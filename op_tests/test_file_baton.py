# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Regression tests for JIT file-baton stale-owner detection."""

import fcntl
import os
import socket
import stat
import tempfile
import threading
import time
import unittest
from unittest import mock

from aiter.jit.utils.file_baton import FileBaton


class TestFileBaton(unittest.TestCase):

    def test_unreadable_live_process_start_time_is_not_stale(self):
        with tempfile.TemporaryDirectory() as tempdir:
            path = os.path.join(tempdir, "build.lock")
            with open(path, "w") as fh:
                fh.write(f"123\n{socket.gethostname()}\n\nknown-start\n")

            baton = FileBaton(path)
            with (
                mock.patch.object(baton, "_pid_alive", return_value=True),
                mock.patch(
                    "aiter.jit.utils.file_baton._process_start_time",
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
                "aiter.jit.utils.file_baton.os.replace",
                side_effect=OSError("simulated death before publication"),
            ):
                with self.assertRaises(OSError):
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
                "aiter.jit.utils.file_baton._pid_namespace",
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
                "aiter.jit.utils.file_baton._pid_namespace",
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
                "aiter.jit.utils.file_baton.socket.gethostname",
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
                    "aiter.jit.utils.file_baton.os.link", side_effect=checked_link
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
                "aiter.jit.utils.file_baton.os.open",
                side_effect=deny_legacy_probe,
            ):
                self.assertTrue(baton._is_stale())

    def test_nonwritable_legacy_steal_marker_is_bypassed_after_grace(self):
        with tempfile.TemporaryDirectory() as tempdir:
            path = os.path.join(tempdir, "build.lock")
            legacy_guard_path = path + ".steal"
            with open(legacy_guard_path, "w"):
                pass
            baton = FileBaton(path, stale_grace_seconds=-1)
            real_open = os.open

            def deny_legacy_guard(candidate, flags, *args, **kwargs):
                if candidate == legacy_guard_path and flags == os.O_RDWR:
                    raise PermissionError("different cache UID")
                return real_open(candidate, flags, *args, **kwargs)

            with mock.patch(
                "aiter.jit.utils.file_baton.os.open",
                side_effect=deny_legacy_guard,
            ):
                guard = baton._try_acquire_steal_guard(allow_legacy_migration=True)

            self.assertIsNotNone(guard)
            with open(legacy_guard_path, "rb") as migrated_guard:
                self.assertEqual(migrated_guard.read(), b"flock\n")
            baton._release_steal_guard(guard)

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
                self.assertIsNone(
                    baton._try_acquire_steal_guard(allow_legacy_migration=True)
                )
            finally:
                os.close(legacy_guard)

    def test_ambiguous_legacy_guard_is_not_migrated_during_stale_break(self):
        with tempfile.TemporaryDirectory() as tempdir:
            path = os.path.join(tempdir, "build.lock")
            with open(path, "w"):
                pass
            with open(path + ".steal", "w"):
                pass

            baton = FileBaton(path, stale_grace_seconds=-1)
            self.assertIsNone(baton._try_acquire_steal_guard())
            with open(path + ".steal", "rb") as legacy_guard:
                self.assertEqual(legacy_guard.read(), b"")


if __name__ == "__main__":
    unittest.main(verbosity=2)
