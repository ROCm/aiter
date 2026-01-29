# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

# mypy: allow-untyped-defs
import os
import time
import logging
import signal

logger = logging.getLogger("aiter")

# Default timeout for compilation locks (10 minutes)
# Can be overridden via AITER_COMPILE_TIMEOUT environment variable
DEFAULT_COMPILE_TIMEOUT = int(os.getenv("AITER_COMPILE_TIMEOUT", "600"))


class FileBaton:
    """A primitive, file-based synchronization utility."""

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

    def try_acquire(self):
        """
        Try to atomically create a file under exclusive access.
        Writes the current process PID to the lock file for stale lock detection.

        Returns:
            True if the file could be created, else False.
        """
        try:
            self.fd = os.open(self.lock_file_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            # Write PID to lock file for stale lock detection
            pid_str = str(os.getpid()).encode()
            os.write(self.fd, pid_str)
            os.fsync(self.fd)
            return True
        except FileExistsError:
            return False

    def _is_stale_lock(self):
        """
        Check if the lock file is stale (holding process has died).
        
        Returns:
            True if the lock is stale, False otherwise.
        """
        try:
            # Read PID from lock file
            with open(self.lock_file_path, 'r') as f:
                pid_str = f.read().strip()
                if not pid_str:
                    return True  # Empty file = stale
                
                pid = int(pid_str)
                
            # Check if process is still alive
            # Sending signal 0 doesn't actually send a signal, just checks if process exists
            try:
                os.kill(pid, 0)
                return False  # Process exists
            except OSError:
                return True  # Process doesn't exist
                
        except (FileNotFoundError, ValueError, PermissionError):
            # File disappeared, invalid PID, or permission denied
            return True

    def wait(self, timeout=None):
        """
        Periodically sleeps until the baton is released.
        
        Args:
            timeout: Maximum seconds to wait (None = use DEFAULT_COMPILE_TIMEOUT).
                    Set to 0 or negative for infinite wait (old behavior).
        
        Raises:
            TimeoutError: If timeout is exceeded and lock is not stale.
        """
        # Use default timeout if none specified
        if timeout is None:
            timeout = DEFAULT_COMPILE_TIMEOUT
        
        # Negative or zero timeout means infinite wait (backward compatible)
        infinite_wait = timeout <= 0
        
        start_time = time.time()
        check_interval = max(self.wait_seconds, 1.0)  # Check at least every 1 second
        last_stale_check = start_time
        stale_check_interval = 30.0  # Check for stale locks every 30 seconds
        
        logger.info(f"waiting for baton release at {self.lock_file_path}" +
                   (f" (timeout={timeout}s)" if not infinite_wait else " (no timeout)"))
        
        while os.path.exists(self.lock_file_path):
            elapsed = time.time() - start_time
            
            # Check for stale lock periodically
            if (time.time() - last_stale_check) > stale_check_interval:
                if self._is_stale_lock():
                    logger.warning(f"Detected stale lock (process died), removing: {self.lock_file_path}")
                    try:
                        os.remove(self.lock_file_path)
                        return  # Lock removed, can proceed
                    except FileNotFoundError:
                        return  # Already removed
                    except Exception as e:
                        logger.error(f"Failed to remove stale lock: {e}")
                last_stale_check = time.time()
            
            # Check timeout (only if not infinite wait)
            if not infinite_wait and elapsed > timeout:
                # Final check: is the lock stale?
                if self._is_stale_lock():
                    logger.warning(f"Lock timeout reached but lock is stale, removing: {self.lock_file_path}")
                    try:
                        os.remove(self.lock_file_path)
                        return
                    except FileNotFoundError:
                        return
                    except Exception as e:
                        logger.error(f"Failed to remove stale lock on timeout: {e}")
                        raise TimeoutError(
                            f"Timeout waiting for baton after {timeout}s at {self.lock_file_path}. "
                            f"Lock appears stale but couldn't be removed: {e}"
                        ) from e
                
                # Lock is not stale, compilation is legitimately taking long
                raise TimeoutError(
                    f"Timeout waiting for baton after {timeout}s at {self.lock_file_path}. "
                    f"Process appears to still be running. Consider increasing AITER_COMPILE_TIMEOUT."
                )
            
            time.sleep(check_interval)

    def release(self):
        """Release the baton and removes its file."""
        if self.fd is not None:
            os.close(self.fd)

        os.remove(self.lock_file_path)
