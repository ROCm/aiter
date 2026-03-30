"""Watchdog and health monitoring utilities for the agentic Triton tuning pipeline."""

import subprocess
from pathlib import Path
from typing import List, Optional


class WatchdogTimeout(Exception):
    """Raised when a command exceeds the configured timeout."""

    def __init__(self, message: str, cmd: Optional[List[str]] = None, timeout: Optional[int] = None) -> None:
        super().__init__(message)
        self.cmd = cmd
        self.timeout = timeout


class CommandWatchdog:
    """Runs subprocess commands with a hard timeout, raising WatchdogTimeout on expiry."""

    def __init__(self, timeout: int = 300) -> None:
        self.timeout = timeout

    def run(
        self,
        cmd: List[str],
        check: bool = False,
        env: Optional[dict] = None,
    ) -> subprocess.CompletedProcess:
        """Run *cmd* as a subprocess.

        Parameters
        ----------
        cmd:
            Command and arguments to run.
        check:
            If ``True``, raise :exc:`subprocess.CalledProcessError` when the
            process exits with a non-zero return code.
        env:
            Optional environment mapping passed directly to :func:`subprocess.run`.

        Returns
        -------
        subprocess.CompletedProcess
            The result of the completed process.

        Raises
        ------
        WatchdogTimeout
            If the process does not finish within :attr:`timeout` seconds.
        subprocess.CalledProcessError
            If *check* is ``True`` and the process returns a non-zero exit code.
        """
        kwargs = dict(capture_output=True, text=True, timeout=self.timeout)
        if env is not None:
            kwargs["env"] = env

        try:
            return subprocess.run(cmd, check=check, **kwargs)
        except subprocess.TimeoutExpired:
            raise WatchdogTimeout(
                f"Command timed out after {self.timeout}s: {' '.join(cmd)}",
                cmd=cmd,
                timeout=self.timeout,
            )


class ProgressMonitor:
    """Monitors a local log file for progress by counting pattern occurrences.

    Parameters
    ----------
    log_path:
        Path to the local log file to monitor.
    pattern:
        String to search for in the log file (each occurrence counts as
        one unit of progress).
    completion_marker:
        String whose presence in the file indicates the run is complete.
    """

    def __init__(
        self,
        log_path,
        pattern: str = "screencase",
        completion_marker: str = "Screen complete",
    ) -> None:
        self._log_path = Path(log_path)
        self._pattern = pattern
        self._completion_marker = completion_marker
        self._last_count: int = 0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _read_text(self) -> str:
        """Return the full contents of the log file, or an empty string if missing."""
        try:
            return self._log_path.read_text(encoding="utf-8", errors="replace")
        except (OSError, IOError):
            return ""

    def _count_occurrences(self, text: str, search: str) -> int:
        """Count non-overlapping occurrences of *search* in *text*."""
        if not search:
            return 0
        return text.count(search)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def get_progress_count(self) -> int:
        """Return the total number of pattern matches found in the log file."""
        return self._count_occurrences(self._read_text(), self._pattern)

    def has_progress(self) -> bool:
        """Return ``True`` if at least one pattern match exists in the log file."""
        return self.get_progress_count() > 0

    def has_new_progress(self) -> bool:
        """Return ``True`` if the match count has grown since the last call.

        As a side effect, updates the internal baseline so subsequent calls
        compare against the new count.
        """
        current = self.get_progress_count()
        new_progress = current > self._last_count
        self._last_count = current
        return new_progress

    def is_complete(self) -> bool:
        """Return ``True`` if the completion marker is present in the log file."""
        return self._completion_marker in self._read_text()


class RemoteProgressMonitor:
    """Monitors a remote log file for progress via ``docker exec`` grep calls.

    Parameters
    ----------
    executor:
        A :class:`~.remote.RemoteExecutor` (or compatible) instance used to
        run commands inside the remote container.
    remote_log_path:
        Absolute path to the log file *inside the container*.
    pattern:
        String to search for (each line containing this string counts).
    completion_marker:
        String whose presence in the file indicates the run is complete.
    """

    def __init__(
        self,
        executor,
        remote_log_path: str,
        pattern: str = "screencase",
        completion_marker: str = "Screen complete",
    ) -> None:
        self._executor = executor
        self._remote_log_path = remote_log_path
        self._pattern = pattern
        self._completion_marker = completion_marker
        self._last_count: int = 0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _remote_count(self, search_str: str) -> int:
        """Return the number of lines in the remote log file containing *search_str*.

        Uses ``grep -c`` via :meth:`executor.docker_exec`.  Returns ``0`` on
        any error (file not found, grep finds no matches, executor failure, …).
        """
        try:
            result = self._executor.docker_exec(
                f"grep -c {_shell_quote(search_str)} {self._remote_log_path}",
                check=False,
            )
            # grep -c exits 0 when matches found, 1 when no matches (but file
            # exists), and 2 on error.  We only trust the output when the
            # return code is 0 or 1.
            if result.returncode not in (0, 1):
                return 0
            stripped = result.stdout.strip()
            if stripped.isdigit():
                return int(stripped)
            return 0
        except Exception:
            return 0

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def get_progress_count(self) -> int:
        """Return the total number of lines matching the pattern in the remote log."""
        return self._remote_count(self._pattern)

    def has_progress(self) -> bool:
        """Return ``True`` if at least one matching line exists in the remote log."""
        return self.get_progress_count() > 0

    def has_new_progress(self) -> bool:
        """Return ``True`` if the match count has grown since the last call."""
        current = self.get_progress_count()
        new_progress = current > self._last_count
        self._last_count = current
        return new_progress

    def is_complete(self) -> bool:
        """Return ``True`` if the completion marker is present in the remote log."""
        return self._remote_count(self._completion_marker) > 0


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _shell_quote(s: str) -> str:
    """Minimally quote *s* for safe embedding in a shell grep argument."""
    import shlex
    return shlex.quote(s)
