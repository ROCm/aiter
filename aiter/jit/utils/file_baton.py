# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

# mypy: allow-untyped-defs
import os
import sys
import time
import logging

logger = logging.getLogger("aiter")


def _pid_is_alive(pid: int) -> bool:
    """Return True if a process with ``pid`` exists on the current host."""
    if pid <= 0:
        return False
    if sys.platform == "win32":
        # Lightweight check via OpenProcess without importing ctypes elsewhere.
        import ctypes
        from ctypes import wintypes

        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        STILL_ACTIVE = 259
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.OpenProcess(
            PROCESS_QUERY_LIMITED_INFORMATION, False, wintypes.DWORD(pid)
        )
        if not handle:
            return False
        try:
            exit_code = wintypes.DWORD()
            if kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)) == 0:
                return False
            return exit_code.value == STILL_ACTIVE
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        # Exists but not ours.
        return True
    except OSError:
        return False
    return True


class FileBaton:
    """A primitive, file-based synchronization utility.

    The owning process writes its PID into the lock file so that other
    processes (or the same process on a later run) can detect and clear a
    stale baton left behind by a crash / Ctrl+C / kill.
    """

    def __init__(self, lock_file_path, wait_seconds=0.2):
        """
        Create a new :class:`FileBaton`.

        Args:
            lock_file_path: The path to the file used for locking.
            wait_seconds: The seconds to periodically sleep (spin) when
                calling ``wait()``.
        """
        self.lock_file_path = lock_file_path
        self.wait_seconds = wait_seconds
        self.fd = None

    def _read_owner_pid(self):
        try:
            with open(self.lock_file_path, "r") as f:
                data = f.read().strip()
            return int(data) if data else None
        except (OSError, ValueError):
            return None

    def _try_clear_stale(self):
        """If the lock file's owner PID is gone, remove the lock. Returns True
        if we cleared a stale lock."""
        pid = self._read_owner_pid()
        if pid is None:
            return False
        if _pid_is_alive(pid):
            return False
        try:
            os.remove(self.lock_file_path)
            logger.warning(
                "Removed stale JIT baton %s (owner pid %d no longer alive)",
                self.lock_file_path,
                pid,
            )
            return True
        except OSError:
            return False

    def try_acquire(self):
        """
        Try to atomically create a file under exclusive access.

        Returns:
            True if the file could be created, else False.
        """
        try:
            self.fd = os.open(self.lock_file_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            # First chance: if the current holder is dead, clear and retry.
            if self._try_clear_stale():
                try:
                    self.fd = os.open(
                        self.lock_file_path,
                        os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                    )
                except FileExistsError:
                    return False
            else:
                return False
        try:
            os.write(self.fd, str(os.getpid()).encode("ascii"))
        except OSError:
            pass
        return True

    def wait(self):
        """
        Periodically sleeps for a certain amount until the baton is released.

        The amount of time slept depends on the ``wait_seconds`` parameter
        passed to the constructor. If the baton's owner dies while we wait,
        we clear the stale lock and return so the caller can retry.
        """
        logger.info(f"waiting for baton release at {self.lock_file_path}")
        stale_check_interval = max(5.0, self.wait_seconds * 25)
        next_stale_check = time.monotonic() + stale_check_interval
        while os.path.exists(self.lock_file_path):
            time.sleep(self.wait_seconds)
            if time.monotonic() >= next_stale_check:
                if self._try_clear_stale():
                    return
                next_stale_check = time.monotonic() + stale_check_interval

    def release(self):
        """Release the baton and removes its file."""
        if self.fd is not None:
            try:
                os.close(self.fd)
            except OSError:
                pass
            self.fd = None
        try:
            os.remove(self.lock_file_path)
        except FileNotFoundError:
            pass
