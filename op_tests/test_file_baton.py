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
                heartbeat_seconds=0,
            )
            self.assertFalse(baton.wait())
            self.assertTrue(os.path.exists(path))
            self.assertTrue(baton.try_acquire())
            baton.release()

    def test_waiter_does_not_report_completion_during_handoff(self):
        with tempfile.TemporaryDirectory() as tempdir:
            path = os.path.join(tempdir, "build.lock")
            breaker = FileBaton(path, heartbeat_seconds=0)
            waiter = FileBaton(path, wait_seconds=0.001, heartbeat_seconds=0)
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

    def test_expired_holder_cannot_touch_or_release_successor(self):
        with tempfile.TemporaryDirectory() as tempdir:
            path = os.path.join(tempdir, "build.lock")
            expired = FileBaton(path, heartbeat_seconds=0)
            successor = FileBaton(path, heartbeat_seconds=0)
            self.assertTrue(expired.try_acquire())

            # Simulate stale recovery replacing the pathname while the expired
            # holder remains alive (for example, a paused container resumes).
            os.remove(path)
            self.assertTrue(successor.try_acquire())
            successor_mtime = os.stat(path).st_mtime_ns

            self.assertTrue(expired._touch_owned_lock())
            self.assertEqual(os.stat(path).st_mtime_ns, successor_mtime)
            expired.release()
            self.assertTrue(os.path.exists(path))
            successor.release()


if __name__ == "__main__":
    unittest.main(verbosity=2)
