# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

# mypy: allow-untyped-defs
import fcntl
import logging
import multiprocessing
import os
import socket
import tempfile
import time

logger = logging.getLogger("aiter")
_GUARD_PROTOCOL = b"flock\n"


def _pid_namespace():
    """Identity of this process's PID namespace, e.g. ``pid:[4026532567]``.

    Containers get distinct namespaces even when they share the host's network
    (and therefore its hostname), which is exactly the case pid+host cannot
    tell apart. Empty string where the kernel does not expose it.
    """
    try:
        return os.readlink("/proc/self/ns/pid")
    except OSError:
        return ""


def _process_start_time(pid):
    """Field 22 of ``/proc/<pid>/stat``: start time in clock ticks since boot.

    Distinguishes a live process from a recycled pid. Empty string where
    unavailable (non-Linux, hidden procfs).
    """
    try:
        with open(f"/proc/{pid}/stat", "rb") as f:
            data = f.read()
        # comm may contain spaces/parens; everything after the last ')' is safe
        return data[data.rindex(b")") + 2 :].split()[19].decode()
    except (OSError, IndexError, ValueError):
        return ""


class FileBaton:
    """A primitive, file-based synchronization utility.

    The lock file records the owning ``pid``, host, **PID namespace and process
    start time** so that a crashed or killed builder (which never reaches
    :meth:`release`) leaves behind a *stale* lock that waiters can detect and
    break, instead of deadlocking forever. This also covers the empty/0-byte
    lock left when a process dies between creating and writing the file.

    ``pid`` + host alone is not an identity. Two cases where it is ambiguous,
    both of which deadlock a build in practice:

    * **Containers sharing the host network** (``docker run --network host``,
      the normal way to run a tuner) report the *host's* hostname while keeping
      their own PID namespace, and every container has a pid 1 and low worker
      pids. A lock leaked by a killed container therefore looks alive to the
      next one, forever -- and with a host-mounted build cache it survives
      restarts, so the wedge is permanent until someone deletes the file.
    * **PID reuse** after the holder dies, on any host.

    The namespace id and start time disambiguate identity. New-format holders
    also retain a kernel ``flock`` for the full build: process death releases
    it automatically, while a paused process keeps it, so cross-namespace
    recovery never has to guess from a heartbeat timeout.
    """

    def __init__(
        self,
        lock_file_path,
        wait_seconds=0.2,
        stale_grace_seconds=10.0,
    ):
        """
        Create a new :class:`FileBaton`.

        Args:
            lock_file_path: The path to the file used for locking.
            wait_seconds: The seconds to periodically sleep (spin) when
                calling ``wait()``.
            stale_grace_seconds: For an orphaned lock with no readable owner
                info (e.g. a 0-byte lock from a crash), how old it must be
                before being treated as stale. Protects the brief window
                between create and write in a healthy builder.
        """
        self.lock_file_path = lock_file_path
        self.wait_seconds = wait_seconds
        self.stale_grace_seconds = stale_grace_seconds
        self.fd = None

    def try_acquire(self):
        """
        Try to atomically create a file under exclusive access and stamp it
        with the current owner (pid + host) for stale detection.

        Returns:
            True if this instance owns the baton, else False.
        """
        # A stale-lock breaker reacquires the baton inside _try_break_stale()
        # before exposing the old lock's absence to other waiters. Let the
        # caller's normal retry loop observe that ownership.
        if self.fd is not None:
            return True
        return self._publish_lock()

    def _publish_lock(self, replace=False):
        """Publish a fully prepared lock, optionally replacing a stale one."""
        # open(O_CREAT, 0o666) publishes a mode filtered by the process umask.
        # A process killed before a following fchmod() can therefore strand a
        # 0600 lock that another cache UID can never probe.  Prepare a private
        # same-directory inode completely, then hard-link it into place.  The
        # link is an atomic no-replace publication and already has mode 0666.
        lock_dir = os.path.dirname(self.lock_file_path) or "."
        lock_name = os.path.basename(self.lock_file_path)
        fd, private_path = tempfile.mkstemp(
            prefix=f".{lock_name}.", suffix=".tmp", dir=lock_dir
        )
        published = False
        try:
            os.fchmod(fd, 0o666)
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            try:
                pid = os.getpid()
                remaining = memoryview(
                    (
                        f"{pid}\n{socket.gethostname()}\n"
                        f"{_pid_namespace()}\n{_process_start_time(pid)}\nflock\n"
                    ).encode()
                )
                while remaining:
                    written = os.write(fd, remaining)
                    if written <= 0:
                        raise OSError("could not write baton owner record")
                    remaining = remaining[written:]
                os.fsync(fd)
            except OSError:
                # Never publish a valid-looking legacy prefix if owner
                # metadata is only partially written. An empty record is
                # protected by the lifetime flock while live and recoverable
                # after process death.
                try:
                    os.ftruncate(fd, 0)
                except OSError:
                    pass
            if replace:
                # The recovery guard excludes release and other breakers.
                # Replacing the stale inode atomically avoids any absent-path
                # interval that a waiter could mistake for completed work.
                os.replace(private_path, self.lock_file_path)
            else:
                try:
                    os.link(private_path, self.lock_file_path)
                except FileExistsError:
                    return False
            self.fd = fd
            published = True
            return True
        finally:
            try:
                os.unlink(private_path)
            except OSError:
                # A leftover private alias cannot block the canonical lock.
                pass
            if not published:
                os.close(fd)

    def wait(self):
        """
        Periodically sleep until the baton is released by its holder, or break
        the lock if its holder is dead.

        Returns:
            True if the holder released the lock normally (its work is done).
            False if a stale lock was broken and this instance atomically took
            its place — the caller should retry ``try_acquire()`` and redo the
            work, since no holder ever finished it.
        """
        logger.info(
            f"[pid={os.getpid()} pname={multiprocessing.current_process().name}] "
            f"waiting for baton release at {self.lock_file_path}"
        )
        while True:
            if not os.path.exists(self.lock_file_path):
                # Normal release removes the lock while holding this guard.
                # Wait for that operation to finish before reporting success.
                # A persistent recovery-guard pathname is harmless; flock tells us
                # whether a guarded operation is actually active.
                sfd = self._try_acquire_steal_guard()
                if sfd is None:
                    if self._try_recover_absent_legacy_guard():
                        logger.warning(
                            f"[pid={os.getpid()}] recovered an abandoned legacy "
                            f"guard at {self.lock_file_path}"
                        )
                        return False
                    time.sleep(self.wait_seconds)
                    continue
                try:
                    completed = not os.path.exists(self.lock_file_path)
                finally:
                    self._release_steal_guard(sfd)
                if completed:
                    return True
                continue
            if self._is_stale() and self._try_break_stale():
                logger.warning(
                    f"[pid={os.getpid()}] broke stale lock at "
                    f"{self.lock_file_path} (dead/abandoned holder)"
                )
                return False
            time.sleep(self.wait_seconds)

    def release(self):
        """Release the baton and remove its file."""
        if self.fd is None:
            return

        # Serialize the verify+unlink step with stale replacement. Without this
        # guard, the path could be replaced after the inode check and before
        # remove(), letting an expired holder delete its successor's lock.
        sfd = None
        while self._owns_lock_path():
            # Finishing the build makes it safe to migrate an ambiguous
            # flock-less legacy marker: a paused old breaker can only remove
            # the lock this owner is about to remove itself.
            sfd = self._try_acquire_steal_guard(allow_legacy_migration=True)
            if sfd is not None:
                break
            time.sleep(self.wait_seconds)
        try:
            if sfd is not None and self._owns_lock_path():
                try:
                    os.remove(self.lock_file_path)
                except FileNotFoundError:
                    pass
        finally:
            os.close(self.fd)
            self.fd = None
            if sfd is not None:
                self._release_steal_guard(sfd)

    def _owns_lock_path(self):
        """Whether the path still names the inode acquired by this instance."""
        if self.fd is None:
            return False
        try:
            owner = os.fstat(self.fd)
            current = os.stat(self.lock_file_path)
        except OSError:
            return False
        return (owner.st_dev, owner.st_ino) == (current.st_dev, current.st_ino)

    def _try_acquire_steal_guard(self, allow_legacy_migration=False):
        """Try to hold the recovery guard; kernel releases it on process exit."""
        guard_path = self.lock_file_path + ".steal"
        while True:
            try:
                sfd = os.open(guard_path, os.O_RDWR)
            except FileNotFoundError:
                sfd = self._publish_guard(guard_path)
                if sfd is not None:
                    return sfd
                continue
            except PermissionError:
                sfd = None
            else:
                try:
                    protocol = os.pread(sfd, len(_GUARD_PROTOCOL), 0)
                except OSError:
                    protocol = b""
                if protocol != _GUARD_PROTOCOL:
                    os.close(sfd)
                    sfd = None
                else:
                    try:
                        fcntl.flock(sfd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    except BlockingIOError:
                        os.close(sfd)
                        return None
                    except OSError:
                        os.close(sfd)
                        raise
                if sfd is not None:
                    return sfd

            # A flock-less legacy marker has no owner identity. It may belong
            # to a paused old process regardless of age, so stale recovery
            # must never replace it while a canonical lock still exists.
            if not allow_legacy_migration:
                return None
            if not self._legacy_steal_guard_can_migrate(guard_path):
                return None

            # Serialize migration among upgraded peers, then atomically
            # replace the legacy inode in place. Older peers keep observing
            # the same .steal pathname and their O_EXCL acquisition remains
            # excluded throughout the replacement.
            migration_path = guard_path + ".migrate"
            mfd = self._try_acquire_protocol_guard(migration_path)
            if mfd is None:
                return None
            retry = False
            try:
                if self._guard_has_protocol(guard_path):
                    retry = True
                elif not self._legacy_steal_guard_can_migrate(guard_path):
                    return None
                else:
                    return self._publish_guard(guard_path, replace=True)
            finally:
                self._release_steal_guard(mfd)
            if retry:
                continue

    @staticmethod
    def _guard_has_protocol(guard_path):
        try:
            with open(guard_path, "rb") as guard:
                return guard.read(len(_GUARD_PROTOCOL)) == _GUARD_PROTOCOL
        except OSError:
            return False

    def _try_acquire_protocol_guard(self, guard_path):
        """Acquire a guard pathname known to use the current protocol."""
        while True:
            try:
                fd = os.open(guard_path, os.O_RDWR)
            except FileNotFoundError:
                fd = self._publish_guard(guard_path)
                if fd is not None:
                    return fd
                continue
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                os.close(fd)
                return None
            return fd

    @staticmethod
    def _publish_guard(guard_path, replace=False):
        """Atomically publish a shared, flocked recovery-guard inode."""
        guard_dir = os.path.dirname(guard_path) or "."
        guard_name = os.path.basename(guard_path)
        fd, private_path = tempfile.mkstemp(
            prefix=f".{guard_name}.", suffix=".tmp", dir=guard_dir
        )
        published = False
        try:
            os.fchmod(fd, 0o666)
            remaining = memoryview(_GUARD_PROTOCOL)
            while remaining:
                written = os.write(fd, remaining)
                if written <= 0:
                    raise OSError("could not write recovery-guard protocol")
                remaining = remaining[written:]
            os.fsync(fd)
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if replace:
                os.replace(private_path, guard_path)
            else:
                try:
                    os.link(private_path, guard_path)
                except FileExistsError:
                    return None
            published = True
            return fd
        finally:
            try:
                os.unlink(private_path)
            except OSError:
                pass
            if not published:
                os.close(fd)

    def _legacy_steal_guard_can_migrate(self, guard_path):
        """Whether a legacy marker passed its grace period and has no flock."""
        try:
            age = time.time() - os.path.getmtime(guard_path)
        except OSError:
            return False
        if age <= self.stale_grace_seconds:
            return False

        # Intermediate versions of this protocol retained a lifetime flock on
        # .steal. Honor it even after the grace period. A restrictive marker
        # from the original O_EXCL-only protocol cannot be opened by a peer
        # UID; after the grace period it is safe to bypass via .steal.flock.
        try:
            fd = os.open(guard_path, os.O_RDWR)
        except OSError:
            return True
        try:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return False
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)
        return True

    def _try_recover_absent_legacy_guard(self):
        """Reacquire work abandoned after a legacy breaker removed the lock.

        With no canonical lock, a paused legacy breaker has already passed its
        destructive step; publishing our lock is safe. Rebuilding is the
        conservative result because the old breaker may have died before it
        could start the replacement work.
        """
        guard_path = self.lock_file_path + ".steal"
        if self._guard_has_protocol(guard_path):
            return False
        if not self._legacy_steal_guard_can_migrate(guard_path):
            return False
        return self.try_acquire()

    @staticmethod
    def _release_steal_guard(sfd):
        fcntl.flock(sfd, fcntl.LOCK_UN)
        os.close(sfd)

    # ---- stale-lock detection ----

    def _read_owner(self):
        """Return (pid, host, ns, start_time, protocol) from the lock file.

        Locks written by an older AITER carry only pid + host; they parse with
        the remaining fields empty and keep the previous behaviour.
        """
        try:
            with open(self.lock_file_path, "r") as f:
                lines = f.read().splitlines()
        except (FileNotFoundError, OSError):
            return None, None, "", "", ""
        if len(lines) < 2 or not lines[0].strip().isdigit():
            return None, None, "", "", ""
        ns = lines[2].strip() if len(lines) > 2 else ""
        start = lines[3].strip() if len(lines) > 3 else ""
        protocol = lines[4].strip() if len(lines) > 4 else ""
        return int(lines[0].strip()), lines[1].strip(), ns, start, protocol

    @staticmethod
    def _pid_alive(pid):
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True  # exists but owned by another user
        return True

    def _owner_flock_released(self):
        """Return True if released, False if held, or None if non-writable."""
        try:
            # NFS emulates flock with byte-range locks and requires a writable
            # descriptor for LOCK_EX. Lock files are explicitly mode 0666 so
            # every cache peer can perform this liveness probe.
            fd = os.open(self.lock_file_path, os.O_RDWR)
        except PermissionError:
            # An old empty lock may predate shared 0666 publication. The
            # caller can recover it by age; protocol=flock records remain
            # conservative when this probe is unavailable.
            return None
        except OSError:
            return False
        try:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return False
            fcntl.flock(fd, fcntl.LOCK_UN)
            return True
        finally:
            os.close(fd)

    def _is_stale(self):
        """A lock is stale if its recorded holder is dead, or if it carries no
        owner info and has outlived the grace period (orphaned/0-byte lock)."""
        pid, host, ns, start, protocol = self._read_owner()
        if pid is None:
            # No readable owner: only trust mtime, and only for our own host
            # cannot be verified, so fall back to an age-based grace window.
            try:
                age = time.time() - os.path.getmtime(self.lock_file_path)
            except OSError:
                return False
            if age <= self.stale_grace_seconds:
                return False
            # None identifies an inaccessible pre-flock orphan. New locks are
            # published mode 0666, so a protocol=flock record never takes this
            # legacy fallback.
            return self._owner_flock_released() is not False
        if protocol == "flock":
            # For new-format locks the kernel-held lifetime lock is the
            # authority across PID/UTS namespaces and hosts sharing an NFS
            # cache. It remains held while a process is paused and is released
            # automatically on process exit.
            return self._owner_flock_released() is True
        if host != socket.gethostname():
            # Legacy locks have no cross-host liveness signal, so never steal
            # one from a different host.
            return False
        my_ns = _pid_namespace()
        if ns and my_ns and ns != my_ns:
            # Same host, different PID namespace: the recorded pid means
            # nothing here. A lifetime flock distinguishes a dead process from
            # a paused one without guessing from a delayed heartbeat. Locks
            # from the pre-flock format are not safe to steal across namespaces.
            return False
        if not self._pid_alive(pid):
            return True
        if start:
            # Same pid, different start time: the original holder died and the
            # pid was recycled.
            current_start = _process_start_time(pid)
            return bool(current_start) and current_start != start
        return False

    def _try_break_stale(self):
        """Atomically break a stale lock under a process-lifetime flock.

        The breaker reacquires the baton before dropping the guard so no waiter
        can mistake the handoff for a successfully completed build. The guard's
        pathname may persist, but a crashed process cannot leave its flock held.
        """
        sfd = self._try_acquire_steal_guard()
        if sfd is None:
            return False
        try:
            # Re-verify under the steal lock to avoid racing a fresh acquire.
            if os.path.exists(self.lock_file_path) and self._is_stale():
                return self._publish_lock(replace=True)
            return False
        finally:
            self._release_steal_guard(sfd)
