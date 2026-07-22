# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""
Unit tests for mp_tuner polling loop logic.

Simulates async_result behavior without GPU/multiprocessing to verify:
1. consecutive_timeouts tracks correctly and resets on success
2. half-GPU threshold triggers break at the right time
3. KeyError tasks stay in remaining_tasks and get retried after root-cause restart

Run: python3 -m unittest op_tests.test_mp_tuner_logic -v
"""

import time
import unittest
from multiprocessing import TimeoutError as MPTimeoutError


class FakeAsyncResult:
    """Simulates multiprocessing.AsyncResult for testing polling logic."""

    def __init__(self, behavior, value=None):
        """
        behavior: "ok", "timeout_pending", "timeout_expired", "keyerror", "accelerator"
        value: return value for "ok"
        """
        self.behavior = behavior
        self.value = value

    def get(self, timeout=10):
        if self.behavior == "ok":
            return self.value
        elif self.behavior in ("timeout_pending", "timeout_expired"):
            raise MPTimeoutError("timeout")
        elif self.behavior == "keyerror":
            raise KeyError("12345")
        elif self.behavior == "accelerator":
            raise type("AcceleratorError", (Exception,), {})("GPU fault")


def simulate_poll_round(remaining_tasks, task_start_times, mp_num, timeout):
    """
    Simulate one round of the mp_tuner polling loop.
    Returns (completed, dummy_failed, pool_restart_needed, broke_early)
    """
    completed_this_round = []
    dummy_failed_tasks = []
    consecutive_timeouts = 0
    half_gpu = max(1, (mp_num + 1) // 2)
    pool_restart_needed = False
    broke_early = False

    for k, async_result in remaining_tasks:
        try:
            if timeout is not None:
                elapsed = time.time() - task_start_times[k]
                remaining_time = timeout - elapsed
                actual_timeout = max(1, min(10, remaining_time))
            else:
                actual_timeout = 10

            async_result.get(timeout=actual_timeout)
            completed_this_round.append((k, async_result))
            consecutive_timeouts = 0

        except MPTimeoutError:
            if timeout is not None:
                elapsed = time.time() - task_start_times[k]
                if elapsed > timeout:
                    consecutive_timeouts += 1
                    completed_this_round.append((k, async_result))
                    pool_restart_needed = True

                    if consecutive_timeouts >= half_gpu:
                        broke_early = True
                        break
                else:
                    consecutive_timeouts = 0

        except Exception as e:
            error_type = type(e).__name__
            is_mapping_error = error_type == "KeyError"

            if is_mapping_error:
                dummy_failed_tasks.append((k, "mapping error"))
            elif error_type == "AcceleratorError":
                completed_this_round.append((k, async_result))
                pool_restart_needed = True
                broke_early = True
                break
            else:
                completed_this_round.append((k, async_result))

    return completed_this_round, dummy_failed_tasks, pool_restart_needed, broke_early


class TestConsecutiveTimeouts(unittest.TestCase):

    def test_single_timeout_no_break_8gpu(self):
        """1 stuck GPU out of 8: should NOT break early."""
        mp_num = 8
        timeout = 0.0
        now = time.time()
        remaining = [
            (0, FakeAsyncResult("timeout_expired")),
            (1, FakeAsyncResult("ok", [("info", 1.0, 0.0)])),
            (2, FakeAsyncResult("timeout_expired")),
            (3, FakeAsyncResult("ok", [("info", 2.0, 0.0)])),
        ]
        start_times = {k: now - 10 for k, _ in remaining}

        completed, dummy, restart, broke = simulate_poll_round(
            remaining, start_times, mp_num, timeout
        )
        self.assertFalse(broke, "Should NOT break early with interleaved success")
        self.assertTrue(restart, "Should still need restart (at least 1 timeout)")
        self.assertEqual(len(completed), 4, "All tasks should be processed")

    def test_half_gpu_consecutive_triggers_break(self):
        """4 consecutive timeouts with 8 GPUs (half=4): should break."""
        mp_num = 8
        timeout = 0.0
        now = time.time()
        remaining = [
            (0, FakeAsyncResult("timeout_expired")),
            (1, FakeAsyncResult("timeout_expired")),
            (2, FakeAsyncResult("timeout_expired")),
            (3, FakeAsyncResult("timeout_expired")),
            (4, FakeAsyncResult("ok", [("info", 1.0, 0.0)])),
        ]
        start_times = {k: now - 10 for k, _ in remaining}

        completed, dummy, restart, broke = simulate_poll_round(
            remaining, start_times, mp_num, timeout
        )
        self.assertTrue(broke, "Should break after 4 consecutive timeouts (half of 8)")
        self.assertTrue(restart)
        self.assertEqual(len(completed), 4, "Task 4 not polled due to break")

    def test_success_resets_consecutive(self):
        """Success in between resets counter: 3 timeouts, 1 ok, 3 timeouts != break for 8 GPU."""
        mp_num = 8
        timeout = 0.0
        now = time.time()
        remaining = [
            (0, FakeAsyncResult("timeout_expired")),
            (1, FakeAsyncResult("timeout_expired")),
            (2, FakeAsyncResult("timeout_expired")),
            (3, FakeAsyncResult("ok", [("info", 1.0, 0.0)])),
            (4, FakeAsyncResult("timeout_expired")),
            (5, FakeAsyncResult("timeout_expired")),
            (6, FakeAsyncResult("timeout_expired")),
        ]
        start_times = {k: now - 10 for k, _ in remaining}

        completed, dummy, restart, broke = simulate_poll_round(
            remaining, start_times, mp_num, timeout
        )
        self.assertFalse(broke, "Should NOT break: success at task 3 resets counter")
        self.assertTrue(restart, "Still need restart from timeouts")
        self.assertEqual(len(completed), 7)

    def test_2gpu_half_is_1(self):
        """2 GPUs: half=1, single consecutive timeout triggers break."""
        mp_num = 2
        timeout = 0.0
        now = time.time()
        remaining = [
            (0, FakeAsyncResult("timeout_expired")),
            (1, FakeAsyncResult("ok", [("info", 1.0, 0.0)])),
        ]
        start_times = {k: now - 10 for k, _ in remaining}

        completed, dummy, restart, broke = simulate_poll_round(
            remaining, start_times, mp_num, timeout
        )
        self.assertTrue(broke, "2 GPUs: half=1, first timeout should break")
        self.assertEqual(len(completed), 1)

    def test_pending_timeout_resets_consecutive(self):
        """Task not yet expired (still running) resets consecutive count."""
        mp_num = 4
        timeout = 100.0
        now = time.time()
        remaining = [
            (0, FakeAsyncResult("timeout_expired")),
            (1, FakeAsyncResult("timeout_pending")),
            (2, FakeAsyncResult("timeout_expired")),
        ]
        start_times = {
            0: now - 200,
            1: now,
            2: now - 200,
        }

        completed, dummy, restart, broke = simulate_poll_round(
            remaining, start_times, mp_num, timeout
        )
        self.assertFalse(broke, "Pending task resets consecutive, so no break")
        self.assertEqual(len(completed), 2)


class TestKeyErrorHandling(unittest.TestCase):

    def test_keyerror_stays_in_remaining(self):
        """KeyError tasks should NOT be in completed_this_round."""
        mp_num = 4
        timeout = 0.0
        now = time.time()
        remaining = [
            (0, FakeAsyncResult("keyerror")),
            (1, FakeAsyncResult("ok", [("info", 1.0, 0.0)])),
        ]
        start_times = {k: now - 10 for k, _ in remaining}

        completed, dummy, restart, broke = simulate_poll_round(
            remaining, start_times, mp_num, timeout
        )
        completed_ids = {k for k, _ in completed}
        self.assertNotIn(0, completed_ids, "KeyError task should NOT be completed")
        self.assertIn(1, completed_ids, "OK task should be completed")
        self.assertEqual(len(dummy), 1, "KeyError task should be in dummy_failed")
        self.assertFalse(restart, "KeyError alone should NOT trigger restart")

    def test_keyerror_with_timeout_gets_resubmitted(self):
        """KeyError tasks wait for root-cause timeout to trigger restart."""
        mp_num = 2
        timeout = 0.0
        now = time.time()
        remaining = [
            (0, FakeAsyncResult("keyerror")),
            (1, FakeAsyncResult("timeout_expired")),
        ]
        start_times = {k: now - 10 for k, _ in remaining}

        completed, dummy, restart, broke = simulate_poll_round(
            remaining, start_times, mp_num, timeout
        )
        completed_ids = {k for k, _ in completed}
        self.assertNotIn(0, completed_ids, "KeyError task stays for resubmit")
        self.assertIn(1, completed_ids, "Root-cause timeout is completed")
        self.assertTrue(restart, "Timeout should trigger restart")

        new_remaining = [(k, ar) for k, ar in remaining if k not in completed_ids]
        self.assertEqual(len(new_remaining), 1)
        self.assertEqual(
            new_remaining[0][0], 0, "Only KeyError task remains for resubmit"
        )

    def test_keyerror_no_restart_without_root_cause(self):
        """If only KeyError tasks remain, no restart, they keep polling."""
        mp_num = 4
        timeout = 100.0
        now = time.time()
        remaining = [
            (0, FakeAsyncResult("keyerror")),
            (1, FakeAsyncResult("keyerror")),
        ]
        start_times = {k: now for k, _ in remaining}

        completed, dummy, restart, broke = simulate_poll_round(
            remaining, start_times, mp_num, timeout
        )
        self.assertFalse(restart, "No restart without root cause")
        self.assertEqual(len(completed), 0, "Nothing completed")
        self.assertEqual(len(dummy), 2, "Both are mapping errors")


class TestAcceleratorError(unittest.TestCase):

    def test_accelerator_breaks_immediately(self):
        """AcceleratorError should break immediately and trigger restart."""
        mp_num = 4
        timeout = 100.0
        now = time.time()
        remaining = [
            (0, FakeAsyncResult("ok", [("info", 1.0, 0.0)])),
            (1, FakeAsyncResult("accelerator")),
            (2, FakeAsyncResult("ok", [("info", 2.0, 0.0)])),
        ]
        start_times = {k: now for k, _ in remaining}

        completed, dummy, restart, broke = simulate_poll_round(
            remaining, start_times, mp_num, timeout
        )
        self.assertTrue(broke, "AcceleratorError should break")
        self.assertTrue(restart, "AcceleratorError should trigger restart")
        completed_ids = {k for k, _ in completed}
        self.assertIn(0, completed_ids)
        self.assertIn(1, completed_ids)
        self.assertNotIn(2, completed_ids, "Task 2 not reached due to break")


if __name__ == "__main__":
    unittest.main(verbosity=2)
