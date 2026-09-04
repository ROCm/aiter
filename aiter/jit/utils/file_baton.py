# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

# mypy: allow-untyped-defs
import fcntl
import logging
import multiprocessing
import os
import socket
import time

logger = logging.getLogger("aiter")


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
        try:
            fd = os.open(self.lock_file_path, os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o644)
        except FileExistsError:
            return False
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            os.close(fd)
            try:
                os.remove(self.lock_file_path)
            except FileNotFoundError:
                pass
            raise
        self.fd = fd
        try:
            os.fchmod(self.fd, 0o644)
            pid = os.getpid()
            os.write(
                self.fd,
                (
                    f"{pid}\n{socket.gethostname()}\n"
                    f"{_pid_namespace()}\n{_process_start_time(pid)}\nflock\n"
                ).encode(),
            )
            os.fsync(self.fd)
        except OSError:
            pass
        return True

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
                # A stale-lock breaker removes the old lock while holding this
                # guard and creates its replacement before releasing the guard.
                # A persistent .steal pathname is harmless; flock tells us
                # whether a handoff is actually active.
                sfd = self._try_acquire_steal_guard()
                if sfd is None:
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
            sfd = self._try_acquire_steal_guard()
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

    def _try_acquire_steal_guard(self):
        """Try to hold the recovery guard; kernel releases it on process exit."""
        guard_path = self.lock_file_path + ".steal"
        try:
            sfd = os.open(guard_path, os.O_CREAT | os.O_EXCL | os.O_RDONLY, 0o666)
        except FileExistsError:
            # flock does not require write access on Linux. Opening read-only
            # lets different UIDs sharing a cache coordinate through a marker
            # created with another user's umask.
            sfd = os.open(guard_path, os.O_RDONLY)
        else:
            # Creation mode is filtered by umask; immediately make the
            # persistent rendezvous inode readable by every cache peer.
            os.fchmod(sfd, 0o666)
        try:
            fcntl.flock(sfd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            os.close(sfd)
            return None
        return sfd

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
        """Whether the kernel released the build owner's lifetime lock."""
        try:
            fd = os.open(self.lock_file_path, os.O_RDONLY)
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
            return age > self.stale_grace_seconds and self._owner_flock_released()
        if host != socket.gethostname():
            # Different host (e.g. shared filesystem): can't check liveness,
            # never steal — avoid breaking a live remote builder's lock.
            return False
        if protocol == "flock":
            # For new-format locks the kernel-held lifetime lock is the
            # authority in every local PID namespace. It remains held while a
            # process is paused and is released automatically on process exit.
            return self._owner_flock_released()
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
                try:
                    os.remove(self.lock_file_path)
                except FileNotFoundError:
                    pass
                return self.try_acquire()
            return False
        finally:
            self._release_steal_guard(sfd)
