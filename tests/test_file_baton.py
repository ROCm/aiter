"""FileBaton stale-lock detection, including killed-but-unreaped holders.

The module under test is pure stdlib, so it is loaded straight from source and
these tests run without ROCm on any host. The zombie case is Linux only, since
only Linux exposes process state via /proc.
"""

import importlib.util
import os
import pathlib
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import unittest

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
_BATON_PATH = _REPO_ROOT / "aiter" / "jit" / "utils" / "file_baton.py"
_IS_LINUX = sys.platform.startswith("linux")


def _load_file_baton():
    spec = importlib.util.spec_from_file_location("_aiter_file_baton", _BATON_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.FileBaton


FileBaton = _load_file_baton()


def _write_lock(path, pid):
    path.write_text(f"{pid}\n{socket.gethostname()}\n")


def _proc_state(pid):
    data = pathlib.Path(f"/proc/{pid}/stat").read_bytes()
    return data[data.rfind(b")") + 1 :].split()[0]


class _Zombie:
    """A child process that is killed and deliberately left unreaped."""

    def __init__(self):
        self.proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])

    def kill(self):
        self.proc.send_signal(signal.SIGKILL)
        # Not calling wait() yet: keeping the child unreaped is what makes it a
        # zombie, i.e. the state that used to defeat stale detection.
        deadline = time.time() + 5.0
        while time.time() < deadline:
            try:
                if _proc_state(self.proc.pid) == b"Z":
                    return
            except FileNotFoundError:
                break
            time.sleep(0.01)
        raise AssertionError(f"pid {self.proc.pid} never reached zombie state")

    def reap(self):
        self.proc.wait()


class TestFileBatonStaleDetection(unittest.TestCase):
    def setUp(self):
        self._dir = tempfile.mkdtemp(prefix="file_baton_")
        self.lock_path = pathlib.Path(self._dir) / "lock"

    def tearDown(self):
        shutil.rmtree(self._dir, ignore_errors=True)

    def test_live_holder_is_not_stale(self):
        pid = os.getpid()
        _write_lock(self.lock_path, pid)
        baton = FileBaton(str(self.lock_path))
        self.assertTrue(FileBaton._pid_alive(pid))
        self.assertFalse(baton._is_stale())

    def test_reaped_process_is_stale(self):
        proc = subprocess.Popen([sys.executable, "-c", "pass"])
        proc.wait()  # reaped -> the pid is no longer in the process table
        _write_lock(self.lock_path, proc.pid)
        self.assertFalse(FileBaton._pid_alive(proc.pid))
        self.assertTrue(FileBaton(str(self.lock_path))._is_stale())

    def test_missing_owner_info_falls_back_to_age(self):
        self.lock_path.write_text("")
        baton = FileBaton(str(self.lock_path), stale_grace_seconds=10.0)
        self.assertFalse(baton._is_stale())

    @unittest.skipUnless(_IS_LINUX, "/proc is Linux only")
    def test_zombie_holder_is_stale(self):
        zombie = _Zombie()
        try:
            zombie.kill()
            _write_lock(self.lock_path, zombie.proc.pid)
            # The regression: a zombie answers kill(pid, 0), so it was treated
            # as alive and waiters spun forever.
            self.assertFalse(FileBaton._pid_alive(zombie.proc.pid))
            baton = FileBaton(str(self.lock_path))
            self.assertTrue(baton._is_stale())
            self.assertTrue(baton._try_break_stale())
            self.assertFalse(self.lock_path.exists())
        finally:
            zombie.reap()

    @unittest.skipUnless(_IS_LINUX, "/proc is Linux only")
    def test_wait_breaks_zombie_lock(self):
        zombie = _Zombie()
        try:
            zombie.kill()
            _write_lock(self.lock_path, zombie.proc.pid)
            result = []
            thread = threading.Thread(
                target=lambda: result.append(FileBaton(str(self.lock_path)).wait()),
                daemon=True,
            )
            thread.start()
            # One spin is 0.2 s; give it a few rounds, then fail loudly instead
            # of hanging the suite if stale detection regresses.
            thread.join(timeout=5.0)
            self.assertFalse(thread.is_alive(), "wait() never broke the stale lock")
            self.assertEqual(result, [False])
        finally:
            zombie.reap()


if __name__ == "__main__":
    unittest.main()
