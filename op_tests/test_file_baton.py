# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Regression tests for JIT file-baton stale-owner detection."""

import os
import socket
import tempfile
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
                heartbeat_seconds=0,
            )
            self.assertFalse(baton.wait())
            self.assertTrue(os.path.exists(path))
            self.assertTrue(baton.try_acquire())
            baton.release()

    def test_handoff_marker_causes_reacquire_not_completion(self):
        with tempfile.TemporaryDirectory() as tempdir:
            path = os.path.join(tempdir, "build.lock")
            with open(path + ".steal", "w"):
                pass

            baton = FileBaton(path, wait_seconds=0, heartbeat_seconds=0)
            self.assertFalse(baton.wait())
            self.assertTrue(os.path.exists(path))
            self.assertFalse(os.path.exists(path + ".steal"))
            baton.release()

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
