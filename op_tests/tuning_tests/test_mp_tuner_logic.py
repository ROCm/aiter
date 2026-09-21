# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""
Unit tests for mp_tuner polling loop logic.

Simulates async_result behavior without GPU/multiprocessing to verify:
1. consecutive_timeouts tracks correctly and resets on success
2. half-GPU threshold triggers break at the right time
3. stale PID mappings trigger an immediate restart and retry

Run: python3 -m unittest op_tests.tuning_tests.test_mp_tuner_logic -v
"""

import importlib
import multiprocessing as mp
import time
import unittest
import warnings
from multiprocessing import TimeoutError as MPTimeoutError

import triton  # noqa: F401  # ROCm environments may require Triton before torch.


def _wait_for_release(release, value):
    release.wait(timeout=5)
    return value


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

        except Exception as e:  # noqa: BLE001
            error_type = type(e).__name__
            is_mapping_error = error_type == "KeyError"

            if is_mapping_error:
                dummy_failed_tasks.append((k, "mapping error"))
                pool_restart_needed = True
                broke_early = True
                break
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

        completed, _dummy, restart, broke = simulate_poll_round(
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

        completed, _dummy, restart, broke = simulate_poll_round(
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

        completed, _dummy, restart, broke = simulate_poll_round(
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

        completed, _dummy, _restart, broke = simulate_poll_round(
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

        completed, _dummy, _restart, broke = simulate_poll_round(
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

        completed, dummy, restart, _broke = simulate_poll_round(
            remaining, start_times, mp_num, timeout
        )
        completed_ids = {k for k, _ in completed}
        self.assertNotIn(0, completed_ids, "KeyError task should NOT be completed")
        self.assertNotIn(1, completed_ids, "Polling stops for immediate remap")
        self.assertEqual(len(dummy), 1, "KeyError task should be in dummy_failed")
        self.assertTrue(restart, "Stale PID mapping must trigger restart")

    def test_keyerror_with_timeout_gets_resubmitted(self):
        """Mapping errors restart before unrelated timeout polling."""
        mp_num = 2
        timeout = 0.0
        now = time.time()
        remaining = [
            (0, FakeAsyncResult("keyerror")),
            (1, FakeAsyncResult("timeout_expired")),
        ]
        start_times = {k: now - 10 for k, _ in remaining}

        completed, _dummy, restart, _broke = simulate_poll_round(
            remaining, start_times, mp_num, timeout
        )
        completed_ids = {k for k, _ in completed}
        self.assertNotIn(0, completed_ids, "KeyError task stays for resubmit")
        self.assertNotIn(1, completed_ids, "Polling stops before the timeout task")
        self.assertTrue(restart, "Mapping error should trigger restart")

        new_remaining = [(k, ar) for k, ar in remaining if k not in completed_ids]
        self.assertEqual(len(new_remaining), 2)

    def test_keyerror_restarts_without_another_failure(self):
        """A mapping error is sufficient cause to rebuild the PID map."""
        mp_num = 4
        timeout = 100.0
        now = time.time()
        remaining = [
            (0, FakeAsyncResult("keyerror")),
            (1, FakeAsyncResult("keyerror")),
        ]
        start_times = {k: now for k, _ in remaining}

        completed, dummy, restart, _broke = simulate_poll_round(
            remaining, start_times, mp_num, timeout
        )
        self.assertTrue(restart, "Mapping errors are themselves a restart cause")
        self.assertEqual(len(completed), 0, "Nothing completed")
        self.assertEqual(len(dummy), 1, "Polling stops at the first mapping error")


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

        completed, _dummy, restart, broke = simulate_poll_round(
            remaining, start_times, mp_num, timeout
        )
        self.assertTrue(broke, "AcceleratorError should break")
        self.assertTrue(restart, "AcceleratorError should trigger restart")
        completed_ids = {k for k, _ in completed}
        self.assertIn(0, completed_ids)
        self.assertIn(1, completed_ids)
        self.assertNotIn(2, completed_ids, "Task 2 not reached due to break")


class TestTaskExecutionTiming(unittest.TestCase):

    def test_queued_task_has_no_elapsed_execution_time(self):
        tuner = importlib.import_module("aiter.utility.mp_tuner")
        elapsed_since_start = getattr(tuner, "_elapsed_since_task_start", None)

        self.assertIsNotNone(
            elapsed_since_start,
            "mp_tuner must calculate timeout from the worker execution start",
        )
        self.assertIsNone(elapsed_since_start([0.0], 0, now=100.0))
        self.assertEqual(elapsed_since_start([55.0], 0, now=100.0), 45.0)

    def test_worker_records_start_only_when_task_leaves_queue(self):
        tuner = importlib.import_module("aiter.utility.mp_tuner")
        init_start_times = getattr(tuner, "_init_task_start_times", None)
        run_with_tracking = getattr(tuner, "_run_with_start_tracking", None)

        self.assertIsNotNone(init_start_times)
        self.assertIsNotNone(run_with_tracking)

        # Importing the ROCm torch/Triton stack in a spawned test worker can
        # abort in the dynamic loader before this helper runs. The queue
        # timing behavior under test is independent of the start method.
        start_method = "fork" if "fork" in mp.get_all_start_methods() else "spawn"
        ctx = mp.get_context(start_method)
        start_times = ctx.RawArray("d", 2)
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"This process .* is multi-threaded, use of fork.*",
                category=DeprecationWarning,
            )
            manager = ctx.Manager()
            release = manager.Event()
            pool = ctx.Pool(1, initializer=init_start_times, initargs=(start_times,))
        try:
            first = pool.apply_async(
                run_with_tracking, (0, _wait_for_release, (release, "first"))
            )
            second = pool.apply_async(
                run_with_tracking, (1, _wait_for_release, (release, "second"))
            )

            deadline = time.monotonic() + 5
            while start_times[0] == 0 and time.monotonic() < deadline:
                time.sleep(0.01)

            self.assertGreater(start_times[0], 0)
            self.assertEqual(
                start_times[1],
                0,
                "Queued task must not get a start timestamp",
            )

            release.set()
            self.assertEqual(first.get(timeout=5), "first")
            self.assertEqual(second.get(timeout=5), "second")
            self.assertGreater(start_times[1], 0)
        finally:
            release.set()
            pool.terminate()
            pool.join()
            manager.shutdown()


class TestTaskStartTimeReset(unittest.TestCase):

    def test_reset_clears_only_the_given_slots(self):
        tuner = importlib.import_module("aiter.utility.mp_tuner")
        reset_start_times = getattr(tuner, "_reset_task_start_times", None)

        self.assertIsNotNone(
            reset_start_times,
            "submitting a task must clear its start-time slot, otherwise a "
            "resubmitted task is judged against the previous attempt's timestamp",
        )
        slots = [11.0, 22.0, 33.0]
        reset_start_times(slots, [0, 2])
        self.assertEqual(list(slots), [0, 22.0, 0])


class TestShapeGroupedContract(unittest.TestCase):

    def test_declared_kernel_count_must_match_group_size(self):
        tuner = importlib.import_module("aiter.utility.mp_tuner")

        with self.assertRaisesRegex(
            ValueError, "declares 2 kernels but contains 1 tasks"
        ):
            tuner.work_group({}, False, 0.0, (2, (None,)), [("only-task",)])


class TestWorkerErrorRatio(unittest.TestCase):

    def test_nonfinite_error_ratio_is_rejected(self):
        tuner = importlib.import_module("aiter.utility.mp_tuner")
        merge_error_ratio = getattr(tuner, "_merge_error_ratio", None)

        self.assertIsNotNone(
            merge_error_ratio,
            "worker must reject non-finite comparator error ratios",
        )
        for observed in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(observed=observed):
                self.assertEqual(merge_error_ratio(0.0, observed), 1.0)

    def test_finite_error_ratio_keeps_maximum(self):
        tuner = importlib.import_module("aiter.utility.mp_tuner")
        merge_error_ratio = getattr(tuner, "_merge_error_ratio", None)

        self.assertIsNotNone(merge_error_ratio)
        self.assertEqual(merge_error_ratio(0.1, 0.2), 0.2)
        self.assertEqual(merge_error_ratio(0.2, 0.1), 0.2)


class TestTypedCandidateStatus(unittest.TestCase):

    def test_failure_classification_preserves_oom_and_unsupported(self):
        tuner = importlib.import_module("aiter.utility.mp_tuner")
        classify = tuner._candidate_failure_status
        self.assertEqual(classify(RuntimeError("HIP out of memory")), "oom_runtime")
        self.assertEqual(classify(ValueError("unsupported layout")), "unsupported")
        self.assertEqual(classify(RuntimeError("kernel launch failed")), "crash")

    def test_status_extension_is_opt_in(self):
        tuner = importlib.import_module("aiter.utility.mp_tuner")
        legacy = tuner._format_worker_result("shape", 1.0, 0.0, "ok", False)
        typed = tuner._format_worker_result("shape", 1.0, 0.0, "ok", True)
        self.assertEqual(legacy, ("shape", 1.0, 0.0))
        self.assertEqual(typed, ("shape", 1.0, 0.0, "ok", ""))

    def test_failure_detail_names_the_exception(self):
        tuner = importlib.import_module("aiter.utility.mp_tuner")
        detail = tuner._candidate_failure_detail(
            ValueError("window_size_right is not supported yet")
        )
        self.assertEqual(detail, "ValueError: window_size_right is not supported yet")
        result = tuner._format_worker_result(
            "shape", -1, 1.0, "unsupported", True, detail
        )
        self.assertEqual(result[3:], ("unsupported", detail))

    def test_failure_detail_is_bounded_and_single_line(self):
        tuner = importlib.import_module("aiter.utility.mp_tuner")
        detail = tuner._candidate_failure_detail(
            RuntimeError("line one\n  line two   " + "x" * 500), limit=80
        )
        self.assertEqual(len(detail), len("RuntimeError: ") + 80)
        self.assertNotIn("\n", detail)
        self.assertTrue(detail.endswith("..."))

    def test_failure_detail_survives_an_empty_message(self):
        tuner = importlib.import_module("aiter.utility.mp_tuner")
        self.assertEqual(
            tuner._candidate_failure_detail(KeyboardInterrupt()), "KeyboardInterrupt"
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
