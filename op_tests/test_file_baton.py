# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Regression tests for JIT file-baton stale-owner detection."""

import importlib.util
import os
import socket
import stat
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

# FileBaton has no GPU dependencies. Load the real module directly so this
# suite can also run on Windows and CPU-only Linux without importing torch.
_BATON_PATH = Path(__file__).resolve().parents[1] / "aiter/jit/utils/file_baton.py"
_spec = importlib.util.spec_from_file_location("file_baton_under_test", _BATON_PATH)
file_baton = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = file_baton
_spec.loader.exec_module(file_baton)
FileBaton = file_baton.FileBaton
fcntl = file_baton.fcntl


@unittest.skipIf(fcntl is None, "POSIX flock tests")
class TestFileBaton(unittest.TestCase):

    def test_unsupported_hardlinks_leave_no_published_or_temporary_lock(self):
        import errno

        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "build.lock")
            baton = FileBaton(path)
            for publish in (baton.try_acquire, baton._try_acquire_steal_guard):
                with mock.patch.object(
                    file_baton.os,
                    "link",
                    side_effect=OSError(errno.EOPNOTSUPP, "hard links"),
                ), self.assertRaises(OSError):
                    publish()
                self.assertIsNone(baton.fd)
                self.assertEqual(os.listdir(directory), [])

    def test_unreadable_live_process_start_time_is_not_stale(self):
        with tempfile.TemporaryDirectory() as tempdir:
            path = os.path.join(tempdir, "build.lock")
            with open(path, "w") as fh:
                fh.write(f"123\n{socket.gethostname()}\n\nknown-start\n")

            baton = FileBaton(path)
            with (
                mock.patch.object(baton, "_pid_alive", return_value=True),
                mock.patch(
                    "file_baton_under_test._process_start_time",
                    return_value="",
                ),
            ):
                self.assertFalse(baton._is_stale())

    def test_stale_breaker_reacquires_before_returning(self):
        with tempfile.TemporaryDirectory() as tempdir:
            path = os.path.join(tempdir, "build.lock")
            with open(path, "w"):
                pass
            baton = FileBaton(
                path,
                wait_seconds=0,
                stale_grace_seconds=-1,
            )
            self.assertFalse(baton.wait())
            self.assertTrue(os.path.exists(path))
            self.assertTrue(baton.try_acquire())
            baton.release()

    def test_failed_stale_replacement_leaves_lock_for_retry(self):
        with tempfile.TemporaryDirectory() as tempdir:
            path = os.path.join(tempdir, "build.lock")
            with open(path, "w"):
                pass

            breaker = FileBaton(path, wait_seconds=0, stale_grace_seconds=-1)
            with mock.patch(
                "file_baton_under_test.os.replace",
                side_effect=OSError("simulated death before publication"),
            ), self.assertRaises(OSError):
                breaker.wait()

            # The stale pathname was never removed. A subsequent waiter sees
            # unfinished work, replaces it, and reports that the build must be
            # retried rather than claiming successful completion.
            self.assertTrue(os.path.exists(path))
            retry = FileBaton(path, wait_seconds=0, stale_grace_seconds=-1)
            self.assertFalse(retry.wait())
            self.assertTrue(retry.try_acquire())
            retry.release()

    def test_waiter_does_not_report_completion_during_handoff(self):
        with tempfile.TemporaryDirectory() as tempdir:
            path = os.path.join(tempdir, "build.lock")
            with open(path, "w"):
                pass
            breaker = FileBaton(path)
            waiter = FileBaton(path, wait_seconds=0.001)
            guard = breaker._try_acquire_steal_guard()
            self.assertIsNotNone(guard)
            result = []
            thread = threading.Thread(target=lambda: result.append(waiter.wait()))
            thread.start()
            time.sleep(0.02)
            self.assertTrue(thread.is_alive())

            self.assertTrue(breaker._publish_lock(replace=True))
            breaker._release_steal_guard(guard)
            time.sleep(0.02)
            self.assertTrue(thread.is_alive())
            breaker.release()
            thread.join(1)
            self.assertEqual(result, [True])

    def test_expired_holder_cannot_release_successor(self):
        with tempfile.TemporaryDirectory() as tempdir:
            path = os.path.join(tempdir, "build.lock")
            expired = FileBaton(path)
            successor = FileBaton(path)
            self.assertTrue(expired.try_acquire())

            # Simulate stale recovery replacing the pathname while the expired
            # holder remains alive (for example, a paused container resumes).
            os.remove(path)
            self.assertTrue(successor.try_acquire())

            expired.release()
            self.assertTrue(os.path.exists(path))
            successor.release()

    def test_foreign_namespace_holder_is_not_stolen_while_paused(self):
        with tempfile.TemporaryDirectory() as tempdir:
            path = os.path.join(tempdir, "build.lock")
            holder = FileBaton(path)
            waiter = FileBaton(path)
            self.assertTrue(holder.try_acquire())

            with mock.patch(
                "file_baton_under_test._pid_namespace",
                return_value="pid:[different]",
            ):
                self.assertFalse(waiter._is_stale())
            holder.release()

    def test_incomplete_owner_record_is_not_stolen_while_flock_is_held(self):
        with tempfile.TemporaryDirectory() as tempdir:
            path = os.path.join(tempdir, "build.lock")
            holder = FileBaton(path)
            waiter = FileBaton(path, stale_grace_seconds=-1)
            self.assertTrue(holder.try_acquire())
            os.ftruncate(holder.fd, 0)

            self.assertFalse(waiter._is_stale())
            holder.release()

    def test_foreign_namespace_dead_holder_is_recoverable(self):
        with tempfile.TemporaryDirectory() as tempdir:
            path = os.path.join(tempdir, "build.lock")
            holder = FileBaton(path)
            waiter = FileBaton(path)
            self.assertTrue(holder.try_acquire())
            os.close(holder.fd)  # simulate process death without release()
            holder.fd = None

            with mock.patch(
                "file_baton_under_test._pid_namespace",
                return_value="pid:[different]",
            ):
                self.assertTrue(waiter._is_stale())
                self.assertFalse(waiter.wait())
            waiter.release()

    def test_lifetime_flock_is_authoritative_across_hostnames(self):
        with tempfile.TemporaryDirectory() as tempdir:
            path = os.path.join(tempdir, "build.lock")
            holder = FileBaton(path)
            waiter = FileBaton(path)
            self.assertTrue(holder.try_acquire())

            with mock.patch(
                "file_baton_under_test.socket.gethostname",
                return_value="another-container-hostname",
            ):
                self.assertFalse(waiter._is_stale())
                os.close(holder.fd)  # simulate process death without release()
                holder.fd = None
                self.assertTrue(waiter._is_stale())
                self.assertFalse(waiter.wait())
            waiter.release()

    def test_lock_files_ignore_restrictive_umask_for_cache_peers(self):
        with tempfile.TemporaryDirectory() as tempdir:
            path = os.path.join(tempdir, "build.lock")
            baton = FileBaton(path)
            old_umask = os.umask(0o077)
            try:
                self.assertTrue(baton.try_acquire())
                guard = baton._try_acquire_steal_guard()
            finally:
                os.umask(old_umask)

            self.assertIsNotNone(guard)
            self.assertEqual(stat.S_IMODE(os.stat(path).st_mode), 0o666)
            self.assertEqual(stat.S_IMODE(os.stat(path + ".steal").st_mode), 0o666)
            baton._release_steal_guard(guard)
            baton.release()

    def test_lock_paths_are_shared_before_atomic_publication(self):
        with tempfile.TemporaryDirectory() as tempdir:
            path = os.path.join(tempdir, "build.lock")
            guard_path = path + ".steal"
            published = []
            real_link = os.link

            def checked_link(source, destination):
                # A creator killed before this point can leave only the
                # private temporary name, never an inaccessible lock path.
                self.assertFalse(os.path.exists(destination))
                self.assertEqual(stat.S_IMODE(os.stat(source).st_mode), 0o666)
                published.append(destination)
                return real_link(source, destination)

            baton = FileBaton(path)
            old_umask = os.umask(0o077)
            try:
                with mock.patch(
                    "file_baton_under_test.os.link", side_effect=checked_link
                ):
                    self.assertTrue(baton.try_acquire())
                    guard = baton._try_acquire_steal_guard()
            finally:
                os.umask(old_umask)

            self.assertIsNotNone(guard)
            self.assertEqual(published, [path, guard_path])
            baton._release_steal_guard(guard)
            baton.release()

    def test_nonwritable_legacy_empty_lock_is_recoverable(self):
        with tempfile.TemporaryDirectory() as tempdir:
            path = os.path.join(tempdir, "build.lock")
            with open(path, "w"):
                pass
            baton = FileBaton(path, stale_grace_seconds=-1)
            real_open = os.open

            def deny_legacy_probe(candidate, flags, *args, **kwargs):
                if candidate == path and flags == os.O_RDWR:
                    raise PermissionError("different cache UID")
                return real_open(candidate, flags, *args, **kwargs)

            with mock.patch(
                "file_baton_under_test.os.open",
                side_effect=deny_legacy_probe,
            ):
                self.assertTrue(baton._is_stale())

    def test_live_preversioned_flock_guard_is_still_honored(self):
        with tempfile.TemporaryDirectory() as tempdir:
            path = os.path.join(tempdir, "build.lock")
            legacy_guard_path = path + ".steal"
            legacy_guard = os.open(
                legacy_guard_path, os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o666
            )
            fcntl.flock(legacy_guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
            try:
                baton = FileBaton(path, stale_grace_seconds=-1)
                self.assertIsNone(baton._try_acquire_steal_guard())
            finally:
                os.close(legacy_guard)

    def test_ambiguous_legacy_guard_is_not_migrated_during_stale_break(self):
        with tempfile.TemporaryDirectory() as tempdir:
            path = os.path.join(tempdir, "build.lock")
            with open(path, "w"):
                pass
            with open(path + ".steal", "w"):
                pass

            old = time.time() - 120
            os.utime(path + ".steal", (old, old))
            baton = FileBaton(path, stale_grace_seconds=-1)
            self.assertIsNone(baton._try_acquire_steal_guard())
            with open(path + ".steal", "rb") as legacy_guard:
                self.assertEqual(legacy_guard.read(), b"")


class TestCompletedBaton(unittest.TestCase):
    def test_absent_lock_with_legacy_guard_is_complete(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "build.lock")
            with open(path + ".steal", "w"):
                pass
            with mock.patch.object(
                file_baton.time, "sleep", side_effect=AssertionError("wait must finish")
            ):
                self.assertTrue(FileBaton(path).wait())
            # Completion must neither migrate nor remove an ambiguous guard.
            self.assertTrue(os.path.exists(path + ".steal"))

    def test_absent_lock_with_unwritable_guard_is_complete(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "build.lock")
            guard_path = path + ".steal"
            with open(guard_path, "w"):
                pass
            os.chmod(guard_path, 0o644)
            real_open = os.open

            def open_as_cache_peer(candidate, flags, *args, **kwargs):
                if candidate == guard_path:
                    raise PermissionError("legacy guard owned by another cache UID")
                return real_open(candidate, flags, *args, **kwargs)

            with (
                mock.patch.object(
                    file_baton.os, "open", side_effect=open_as_cache_peer
                ),
                mock.patch.object(
                    file_baton.time,
                    "sleep",
                    side_effect=AssertionError("wait must finish"),
                ),
            ):
                self.assertTrue(FileBaton(path).wait())


class TestWindowsFallback(unittest.TestCase):
    def setUp(self):
        # On Linux this exercises the same fallback; on Windows it is native.
        patcher = mock.patch.object(file_baton, "fcntl", None)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_acquire_contend_release_and_wait(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "build.lock")
            holder, waiter = FileBaton(path), FileBaton(path)
            self.assertTrue(holder.try_acquire())
            try:
                self.assertFalse(waiter.try_acquire())
                self.assertFalse(waiter._is_stale())
                self.assertEqual(len(Path(path).read_text().splitlines()), 2)
                self.assertTrue(FileBaton._pid_alive(os.getpid()))
            finally:
                holder.release()
            self.assertTrue(waiter.wait())
            self.assertFalse(os.path.exists(path))

    def test_dead_owner_is_recovered_and_reacquired(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "build.lock")
            # A child exits without release(), leaving a real dead-owner lock.
            subprocess.run(
                [
                    sys.executable,
                    "-c",
                    """
import importlib.util, sys
spec = importlib.util.spec_from_file_location('baton_child', sys.argv[1])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
module.fcntl = None
assert module.FileBaton(sys.argv[2]).try_acquire()
""",
                    str(_BATON_PATH),
                    path,
                ],
                check=True,
                timeout=10,
            )
            baton = FileBaton(path, wait_seconds=0)
            self.assertFalse(baton.wait())
            self.assertTrue(baton.try_acquire())
            baton.release()
            self.assertFalse(os.path.exists(path + ".steal"))

    def test_empty_orphan_uses_legacy_grace(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "build.lock")
            Path(path).touch()
            baton = FileBaton(path, stale_grace_seconds=-1)
            self.assertFalse(baton.wait())
            self.assertTrue(baton.try_acquire())
            baton.release()

    def test_flock_record_is_not_stolen_without_flock_support(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "build.lock")
            Path(path).write_text("123\nremote\nnamespace\nstart\nflock\n")
            self.assertFalse(FileBaton(path)._is_stale())

    @unittest.skipUnless(sys.platform == "win32", "requires native Windows imports")
    def test_windows_module_import_does_not_require_fcntl(self):
        subprocess.run(
            [
                sys.executable,
                "-c",
                """
import importlib.util, sys
sys.platform = 'win32'
sys.modules['fcntl'] = None
spec = importlib.util.spec_from_file_location('baton_windows', sys.argv[1])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
assert module.fcntl is None
""",
                str(_BATON_PATH),
            ],
            check=True,
            timeout=10,
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
