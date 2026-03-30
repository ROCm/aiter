"""Tests for the watchdog and progress monitoring utilities."""

import subprocess
import sys
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from .watchdog import CommandWatchdog, ProgressMonitor, RemoteProgressMonitor, WatchdogTimeout


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _completed(returncode=0, stdout="", stderr=""):
    r = MagicMock(spec=subprocess.CompletedProcess)
    r.returncode = returncode
    r.stdout = stdout
    r.stderr = stderr
    return r


# ---------------------------------------------------------------------------
# CommandWatchdog
# ---------------------------------------------------------------------------

class TestCommandWatchdogSuccess(unittest.TestCase):
    """Commands that complete within the timeout succeed normally."""

    @patch("subprocess.run")
    def test_returns_completed_process(self, mock_run):
        mock_run.return_value = _completed(returncode=0, stdout="hello\n")
        dog = CommandWatchdog(timeout=10)
        result = dog.run([sys.executable, "-c", "print('hello')"])
        self.assertEqual(result.returncode, 0)
        self.assertEqual(result.stdout, "hello\n")

    @patch("subprocess.run")
    def test_env_passed_through(self, mock_run):
        mock_run.return_value = _completed(returncode=0, stdout="world\n")
        dog = CommandWatchdog(timeout=10)
        env = {"MY_VAR": "world"}
        dog.run([sys.executable, "-c", "import os; print(os.environ['MY_VAR'])"], env=env)
        call_kwargs = mock_run.call_args[1]
        self.assertIn("env", call_kwargs)
        self.assertEqual(call_kwargs["env"], env)

    @patch("subprocess.run")
    def test_timeout_forwarded_to_subprocess(self, mock_run):
        mock_run.return_value = _completed(returncode=0)
        dog = CommandWatchdog(timeout=42)
        dog.run(["true"])
        call_kwargs = mock_run.call_args[1]
        self.assertEqual(call_kwargs.get("timeout"), 42)


class TestCommandWatchdogTimeout(unittest.TestCase):
    """Commands that exceed the timeout raise WatchdogTimeout."""

    @patch("subprocess.run")
    def test_timeout_raises_watchdog_timeout(self, mock_run):
        mock_run.side_effect = subprocess.TimeoutExpired(cmd=["sleep"], timeout=1)
        dog = CommandWatchdog(timeout=1)
        with self.assertRaises(WatchdogTimeout) as ctx:
            dog.run(["sleep", "100"])
        self.assertIn("timed out", str(ctx.exception).lower())

    @patch("subprocess.run")
    def test_watchdog_timeout_carries_cmd(self, mock_run):
        mock_run.side_effect = subprocess.TimeoutExpired(cmd=["sleep"], timeout=1)
        dog = CommandWatchdog(timeout=1)
        with self.assertRaises(WatchdogTimeout) as ctx:
            dog.run(["sleep", "100"])
        self.assertEqual(ctx.exception.cmd, ["sleep", "100"])

    @patch("subprocess.run")
    def test_watchdog_timeout_carries_timeout_value(self, mock_run):
        mock_run.side_effect = subprocess.TimeoutExpired(cmd=["sleep"], timeout=5)
        dog = CommandWatchdog(timeout=5)
        with self.assertRaises(WatchdogTimeout) as ctx:
            dog.run(["sleep", "100"])
        self.assertEqual(ctx.exception.timeout, 5)


class TestCommandWatchdogFailure(unittest.TestCase):
    """Non-zero exit codes are handled according to the check flag."""

    @patch("subprocess.run")
    def test_failure_without_check_returns_result(self, mock_run):
        mock_run.return_value = _completed(returncode=1, stderr="oops")
        dog = CommandWatchdog(timeout=10)
        result = dog.run(["false"], check=False)
        self.assertEqual(result.returncode, 1)

    @patch("subprocess.run")
    def test_failure_with_check_raises_called_process_error(self, mock_run):
        mock_run.side_effect = subprocess.CalledProcessError(
            returncode=2, cmd=["false"], output="", stderr="bad"
        )
        dog = CommandWatchdog(timeout=10)
        with self.assertRaises(subprocess.CalledProcessError) as ctx:
            dog.run(["false"], check=True)
        self.assertEqual(ctx.exception.returncode, 2)

    @patch("subprocess.run")
    def test_check_kwarg_forwarded(self, mock_run):
        mock_run.return_value = _completed(returncode=0)
        dog = CommandWatchdog(timeout=10)
        dog.run(["true"], check=True)
        call_kwargs = mock_run.call_args[1]
        self.assertTrue(call_kwargs.get("check"))


# ---------------------------------------------------------------------------
# ProgressMonitor (local files)
# ---------------------------------------------------------------------------

class TestProgressMonitorHasProgress(unittest.TestCase):
    def setUp(self):
        import tempfile
        self._tmpdir = tempfile.mkdtemp()
        self._log = Path(self._tmpdir) / "tuning.log"

    def tearDown(self):
        import shutil
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def test_no_file_no_progress(self):
        monitor = ProgressMonitor(self._log)
        self.assertFalse(monitor.has_progress())

    def test_empty_file_no_progress(self):
        self._log.write_text("")
        monitor = ProgressMonitor(self._log)
        self.assertFalse(monitor.has_progress())

    def test_single_match_has_progress(self):
        self._log.write_text("line1\nscreencase result\nline3\n")
        monitor = ProgressMonitor(self._log)
        self.assertTrue(monitor.has_progress())

    def test_no_match_no_progress(self):
        self._log.write_text("nothing useful here\n")
        monitor = ProgressMonitor(self._log)
        self.assertFalse(monitor.has_progress())

    def test_custom_pattern(self):
        self._log.write_text("kernel_done 42\n")
        monitor = ProgressMonitor(self._log, pattern="kernel_done")
        self.assertTrue(monitor.has_progress())


class TestProgressMonitorHasNewProgress(unittest.TestCase):
    def setUp(self):
        import tempfile
        self._tmpdir = tempfile.mkdtemp()
        self._log = Path(self._tmpdir) / "tuning.log"

    def tearDown(self):
        import shutil
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def test_first_call_with_matches_returns_true(self):
        self._log.write_text("screencase 1\n")
        monitor = ProgressMonitor(self._log)
        self.assertTrue(monitor.has_new_progress())

    def test_second_call_without_new_data_returns_false(self):
        self._log.write_text("screencase 1\n")
        monitor = ProgressMonitor(self._log)
        monitor.has_new_progress()  # consume initial progress
        self.assertFalse(monitor.has_new_progress())

    def test_new_lines_added_between_checks_returns_true(self):
        self._log.write_text("screencase 1\n")
        monitor = ProgressMonitor(self._log)
        monitor.has_new_progress()  # consume first match
        # Write more matches
        with self._log.open("a") as f:
            f.write("screencase 2\nscreencase 3\n")
        self.assertTrue(monitor.has_new_progress())

    def test_count_increases_then_flattens(self):
        self._log.write_text("screencase A\n")
        monitor = ProgressMonitor(self._log)
        self.assertTrue(monitor.has_new_progress())
        self.assertFalse(monitor.has_new_progress())
        with self._log.open("a") as f:
            f.write("screencase B\n")
        self.assertTrue(monitor.has_new_progress())
        self.assertFalse(monitor.has_new_progress())


class TestProgressMonitorGetProgressCount(unittest.TestCase):
    def setUp(self):
        import tempfile
        self._tmpdir = tempfile.mkdtemp()
        self._log = Path(self._tmpdir) / "tuning.log"

    def tearDown(self):
        import shutil
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def test_count_zero_for_missing_file(self):
        monitor = ProgressMonitor(self._log)
        self.assertEqual(monitor.get_progress_count(), 0)

    def test_count_matches_occurrences(self):
        self._log.write_text("screencase 1\nscreencase 2\nscreencase 3\n")
        monitor = ProgressMonitor(self._log)
        self.assertEqual(monitor.get_progress_count(), 3)

    def test_count_pattern_on_same_line(self):
        self._log.write_text("screencase screencase\n")
        monitor = ProgressMonitor(self._log)
        # str.count finds both non-overlapping occurrences
        self.assertEqual(monitor.get_progress_count(), 2)


class TestProgressMonitorIsComplete(unittest.TestCase):
    def setUp(self):
        import tempfile
        self._tmpdir = tempfile.mkdtemp()
        self._log = Path(self._tmpdir) / "tuning.log"

    def tearDown(self):
        import shutil
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def test_completion_marker_present(self):
        self._log.write_text("screencase 1\nScreen complete\n")
        monitor = ProgressMonitor(self._log)
        self.assertTrue(monitor.is_complete())

    def test_completion_not_detected_without_marker(self):
        self._log.write_text("screencase 1\nstill running\n")
        monitor = ProgressMonitor(self._log)
        self.assertFalse(monitor.is_complete())

    def test_completion_not_detected_for_missing_file(self):
        monitor = ProgressMonitor(self._log)
        self.assertFalse(monitor.is_complete())

    def test_custom_completion_marker(self):
        self._log.write_text("DONE\n")
        monitor = ProgressMonitor(self._log, completion_marker="DONE")
        self.assertTrue(monitor.is_complete())


# ---------------------------------------------------------------------------
# RemoteProgressMonitor
# ---------------------------------------------------------------------------

def _make_executor(count: int = 0, returncode: int = 0):
    """Return a mock executor whose docker_exec returns a grep -c style result."""
    executor = MagicMock()
    executor.docker_exec.return_value = _completed(
        returncode=returncode,
        stdout=str(count) + "\n",
    )
    return executor


class TestRemoteProgressMonitorHasProgress(unittest.TestCase):
    def test_has_progress_when_count_nonzero(self):
        executor = _make_executor(count=3)
        monitor = RemoteProgressMonitor(executor, "/var/log/tuning.log")
        self.assertTrue(monitor.has_progress())

    def test_no_progress_when_count_zero(self):
        # grep -c returns 0 matches (exit 1 per POSIX when no matches)
        executor = _make_executor(count=0, returncode=1)
        monitor = RemoteProgressMonitor(executor, "/var/log/tuning.log")
        self.assertFalse(monitor.has_progress())

    def test_executor_error_means_no_progress(self):
        executor = MagicMock()
        executor.docker_exec.side_effect = Exception("SSH gone")
        monitor = RemoteProgressMonitor(executor, "/var/log/tuning.log")
        self.assertFalse(monitor.has_progress())


class TestRemoteProgressMonitorIsComplete(unittest.TestCase):
    def test_is_complete_when_marker_found(self):
        executor = _make_executor(count=1)
        monitor = RemoteProgressMonitor(executor, "/var/log/tuning.log")
        self.assertTrue(monitor.is_complete())

    def test_not_complete_when_marker_absent(self):
        executor = _make_executor(count=0, returncode=1)
        monitor = RemoteProgressMonitor(executor, "/var/log/tuning.log")
        self.assertFalse(monitor.is_complete())

    def test_completion_check_uses_completion_marker(self):
        executor = _make_executor(count=1)
        monitor = RemoteProgressMonitor(
            executor,
            "/var/log/tuning.log",
            completion_marker="Screen complete",
        )
        monitor.is_complete()
        call_args = executor.docker_exec.call_args[0][0]
        self.assertIn("Screen complete", call_args)


class TestRemoteProgressMonitorNoProgress(unittest.TestCase):
    def test_get_progress_count_returns_zero_on_error(self):
        executor = MagicMock()
        executor.docker_exec.return_value = _completed(returncode=2, stdout="")
        monitor = RemoteProgressMonitor(executor, "/var/log/tuning.log")
        self.assertEqual(monitor.get_progress_count(), 0)

    def test_get_progress_count_returns_count_on_success(self):
        executor = _make_executor(count=5)
        monitor = RemoteProgressMonitor(executor, "/var/log/tuning.log")
        self.assertEqual(monitor.get_progress_count(), 5)

    def test_has_new_progress_detects_increase(self):
        executor = MagicMock()
        executor.docker_exec.side_effect = [
            _completed(returncode=0, stdout="2\n"),
            _completed(returncode=0, stdout="5\n"),
        ]
        monitor = RemoteProgressMonitor(executor, "/var/log/tuning.log")
        monitor.has_new_progress()  # consume baseline (0 -> 2)
        self.assertTrue(monitor.has_new_progress())  # 2 -> 5

    def test_remote_count_uses_pattern_in_grep_command(self):
        executor = _make_executor(count=0, returncode=1)
        monitor = RemoteProgressMonitor(executor, "/var/log/tuning.log", pattern="my_pattern")
        monitor.get_progress_count()
        call_args = executor.docker_exec.call_args[0][0]
        self.assertIn("my_pattern", call_args)
        self.assertIn("grep", call_args)
        self.assertIn("-c", call_args)


# ---------------------------------------------------------------------------
# WatchdogTimeout exception
# ---------------------------------------------------------------------------

class TestWatchdogTimeoutException(unittest.TestCase):
    def test_is_exception(self):
        exc = WatchdogTimeout("timed out")
        self.assertIsInstance(exc, Exception)

    def test_message_in_str(self):
        exc = WatchdogTimeout("command timed out")
        self.assertIn("timed out", str(exc))

    def test_optional_attributes(self):
        exc = WatchdogTimeout("msg", cmd=["sleep", "1"], timeout=5)
        self.assertEqual(exc.cmd, ["sleep", "1"])
        self.assertEqual(exc.timeout, 5)

    def test_defaults_are_none(self):
        exc = WatchdogTimeout("msg")
        self.assertIsNone(exc.cmd)
        self.assertIsNone(exc.timeout)


if __name__ == "__main__":
    unittest.main()
