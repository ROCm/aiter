# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Regression tests for JIT file-baton stale-owner detection."""

import os
import socket
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
            # A leftover marker has no live flock and must not wedge recovery.
            with open(path + ".steal", "w"):
                pass
            os.chmod(path + ".steal", 0o444)

            baton = FileBaton(
                path,
                wait_seconds=0,
                stale_grace_seconds=-1,
            )
            self.assertFalse(baton.wait())
            self.assertTrue(os.path.exists(path))
            self.assertTrue(baton.try_acquire())
            baton.release()

    def test_waiter_does_not_report_completion_during_handoff(self):
        with tempfile.TemporaryDirectory() as tempdir:
            path = os.path.join(tempdir, "build.lock")
            breaker = FileBaton(path)
            waiter = FileBaton(path, wait_seconds=0.001)
            guard = breaker._try_acquire_steal_guard()
            self.assertIsNotNone(guard)
            result = []
            thread = threading.Thread(target=lambda: result.append(waiter.wait()))
            thread.start()
            time.sleep(0.02)
            self.assertTrue(thread.is_alive())

            self.assertTrue(breaker.try_acquire())
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


if __name__ == "__main__":
    unittest.main(verbosity=2)
