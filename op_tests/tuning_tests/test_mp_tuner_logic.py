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

import importlib
import multiprocessing as mp
import os
import subprocess
import sys
import time
import unittest
from multiprocessing import TimeoutError as MPTimeoutError


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

        completed, _dummy, restart, _broke = simulate_poll_round(
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

        completed, dummy, restart, _broke = simulate_poll_round(
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

        ctx = mp.get_context("spawn")
        start_times = ctx.RawArray("d", 2)
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

    def test_reset_clears_worker_pids(self):
        tuner = importlib.import_module("aiter.utility.mp_tuner")
        slots, pids = [11.0, 22.0], [101, 202]
        tuner._reset_task_start_times(slots, [1], pids)
        self.assertEqual(pids, [101, 0])


class TestTaskWorkerExit(unittest.TestCase):

    def test_exited_worker_is_detected(self):
        tuner = importlib.import_module("aiter.utility.mp_tuner")
        worker_exited = getattr(tuner, "_task_worker_exited", None)
        self.assertIsNotNone(
            worker_exited,
            "a task whose worker died (e.g. GPU fault) must not wait for the timeout",
        )
        proc = subprocess.Popen([sys.executable, "-c", "pass"])
        proc.wait()
        self.assertTrue(worker_exited([proc.pid], 0))
        self.assertFalse(worker_exited([os.getpid()], 0))
        self.assertFalse(worker_exited([0], 0), "a queued task has no worker yet")

    def test_worker_pid_recorded_when_task_starts(self):
        tuner = importlib.import_module("aiter.utility.mp_tuner")
        ctx = mp.get_context("spawn")
        start_times = ctx.RawArray("d", 1)
        pids = ctx.RawArray("i", 1)
        pool = ctx.Pool(
            1,
            initializer=tuner._init_task_start_times,
            initargs=(start_times, pids),
        )
        try:
            worker_pid = pool.apply_async(
                tuner._run_with_start_tracking, (0, os.getpid, ())
            ).get(timeout=60)
            self.assertEqual(pids[0], worker_pid)
            self.assertGreater(start_times[0], 0)
        finally:
            pool.terminate()
            pool.join()


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

    def setUp(self):
        self.tuner = importlib.import_module("aiter.utility.mp_tuner")

    def test_failure_classification_preserves_oom_and_unsupported(self):
        classify = self.tuner._candidate_failure_status
        self.assertEqual(classify(RuntimeError("HIP out of memory")), "oom_runtime")
        self.assertEqual(classify(ValueError("unsupported layout")), "unsupported")
        self.assertEqual(classify(RuntimeError("kernel launch failed")), "crash")

    def test_status_extension_is_opt_in(self):
        result = self.tuner._format_worker_result
        self.assertEqual(result("shape", 1.0, 0.0, "ok", False), ("shape", 1.0, 0.0))
        self.assertEqual(
            result("shape", -1, 1.0, "unsupported", True, "why"),
            ("shape", -1, 1.0, "unsupported", "why"),
        )

    def test_failure_detail_is_one_bounded_line_naming_the_exception(self):
        detail = self.tuner._candidate_failure_detail
        self.assertEqual(
            detail(ValueError("window_size_right is not supported yet")),
            "ValueError: window_size_right is not supported yet",
        )
        self.assertEqual(detail(KeyboardInterrupt()), "KeyboardInterrupt")
        long = detail(RuntimeError("line one\n  line two   " + "x" * 500), limit=80)
        self.assertEqual(len(long), len("RuntimeError: ") + 80)
        self.assertNotIn("\n", long)
        self.assertTrue(long.endswith("..."))


def _accelerator_error(message="illegal memory access"):
    return type("AcceleratorError", (RuntimeError,), {})(message)


class TestResumeAfterFault(unittest.TestCase):
    """A fault costs only the candidate that was running."""

    GROUP = ((("s", "a"),), (("s", "b"),), (("s", "c"),))

    def setUp(self):
        self.tuner = importlib.import_module("aiter.utility.mp_tuner")

    def test_measured_candidates_survive_and_the_rest_are_resubmitted(self):
        for typed in (False, True):
            with self.subTest(return_status=typed):
                measured = {("s", "a"): (("s", "a"), 12.5, 0.0)}
                done, rest = self.tuner._split_after_fault(
                    self.GROUP, measured, "crash", "fault", typed
                )
                self.assertEqual(done[0], measured[("s", "a")])
                failed = (("s", "b"), float("inf"), 1.0)
                self.assertEqual(done[1], (*failed, "crash", "fault")[: len(done[1])])
                self.assertEqual(len(done[1]), 5 if typed else 3)
                self.assertEqual(rest, self.GROUP[2:])

    def test_a_measured_prefix_needs_no_resubmission(self):
        measured = {task[0]: (task[0], 1.0, 0.0) for task in self.GROUP}
        done, rest = self.tuner._measured_prefix(self.GROUP, measured)
        self.assertEqual([r[0] for r in done], [t[0] for t in self.GROUP])
        self.assertEqual(rest, ())

    def test_unhashable_info_is_keyed_by_repr(self):
        group = [((["s"], "a"),), ((["s"], "b"),)]
        measured = {self.tuner._info_key(group[0][0]): "a-result"}
        done, rest = self.tuner._measured_prefix(group, measured)
        self.assertEqual((done, rest), (["a-result"], group[1:]))


def _fake_task(name, data=None):
    gen = (lambda *a, device=None: data) if data is not None else None
    return (("shape", name), gen, (), name, ((),), {}, None, None, {}, "ref")


class TestWorkGroupCandidateFailure(unittest.TestCase):
    """work_group records a candidate that fails before launch and goes on."""

    def run_group(self, tasks, worker_side_effect=None, synchronize=None):
        from unittest import mock

        tuner = importlib.import_module("aiter.utility.mp_tuner")
        queue = []
        worker = mock.Mock(
            side_effect=worker_side_effect
            or (lambda gpu, info, *rest: (info, 1.0, 0.0))
        )
        with (
            mock.patch.object(tuner, "worker", worker),
            mock.patch.object(tuner.torch.cuda, "set_device"),
            mock.patch.object(tuner.torch.cuda, "empty_cache"),
            mock.patch.object(tuner.torch.cuda, "is_initialized", return_value=True),
            mock.patch.object(tuner.torch.cuda, "synchronize", side_effect=synchronize),
        ):
            rets = tuner.work_group(
                {os.getpid(): 0},
                False,
                0.05,
                (len(tasks), ()),
                tasks,
                progress_queue=mock.Mock(put=queue.append),
            )
        return rets, queue

    def test_a_failing_input_costs_only_its_candidate(self):
        def boom(*args, device=None):
            raise ValueError("cannot build inputs")

        tasks = [_fake_task("a"), _fake_task("b"), _fake_task("c")]
        tasks[1] = (tasks[1][0], boom, *tasks[1][2:])
        rets, queue = self.run_group(tasks)
        self.assertEqual([us for _, us, _ in rets], [1.0, float("inf"), 1.0])
        self.assertEqual(queue, rets)

    def test_an_accelerator_fault_still_aborts_the_group(self):
        def worker(gpu, info, *rest):
            if info[1] == "b":
                raise _accelerator_error()
            return info, 1.0, 0.0

        tasks = [_fake_task("a"), _fake_task("b"), _fake_task("c")]
        with self.assertRaises(RuntimeError):
            self.run_group(tasks, worker)

    def test_an_input_failure_that_breaks_the_context_aborts_the_group(self):
        tuner = importlib.import_module("aiter.utility.mp_tuner")

        def boom(*args, device=None):
            raise RuntimeError("HIP error: unspecified launch failure")

        tasks = [_fake_task("a"), _fake_task("b")]
        tasks[0] = (tasks[0][0], boom, *tasks[0][2:])
        with self.assertRaises(tuner.GpuContextLost):
            self.run_group(tasks, synchronize=RuntimeError("sticky"))

    def test_only_a_shape_group_collects_garbage_afterwards(self):
        from unittest import mock

        tuner = importlib.import_module("aiter.utility.mp_tuner")
        for tasks, collected in (([_fake_task("a")], True), (_fake_task("a"), False)):
            with (
                self.subTest(grouped=collected),
                mock.patch.object(tuner, "worker", lambda gpu, info, *r: (info, 1, 0)),
                mock.patch.object(tuner.torch.cuda, "set_device"),
                mock.patch.object(tuner.torch.cuda, "empty_cache") as empty_cache,
                mock.patch.object(tuner.gc, "collect") as collect,
            ):
                tuner.work_group({os.getpid(): 0}, False, 0.05, (1, ()), tasks)
                self.assertEqual(collect.called, collected)
                self.assertEqual(empty_cache.called, collected)


class TestGpuFaultProbe(unittest.TestCase):
    """A fault is recognised by type or a failing synchronize, not by text."""

    def setUp(self):
        self.tuner = importlib.import_module("aiter.utility.mp_tuner")

    def probe(self, exc, synchronize=None):
        from unittest import mock

        with (
            mock.patch.object(
                self.tuner.torch.cuda, "is_initialized", return_value=True
            ),
            mock.patch.object(
                self.tuner.torch.cuda, "synchronize", side_effect=synchronize
            ),
        ):
            self.tuner._raise_if_gpu_fault(exc)

    def test_a_healthy_context_is_not_a_fault_whatever_the_message(self):
        self.probe(RuntimeError("illegal memory access"))

    def test_a_failing_synchronize_or_an_accelerator_error_is_a_fault(self):
        for exc, sync in (
            (ValueError("anything"), RuntimeError("sticky")),
            (_accelerator_error("x"), None),
        ):
            with self.subTest(exc=exc), self.assertRaises(self.tuner.GpuContextLost):
                self.probe(exc, sync)

    def test_the_parent_recognises_a_pickled_fault_by_type(self):
        import pickle

        lost = pickle.loads(pickle.dumps(self.tuner.GpuContextLost("boom")))
        self.assertTrue(self.tuner._is_accelerator_error(lost))
        self.assertFalse(
            self.tuner._is_accelerator_error(RuntimeError("illegal memory access"))
        )


class _FakeAsync:
    def __init__(self, func, args):
        try:
            self.value, self.error = func(*args), None
        except Exception as error:  # noqa: BLE001
            self.value, self.error = None, error

    def ready(self):
        return True

    def get(self, timeout=None):
        if self.error is not None:
            raise self.error
        return self.value


class _FakePool:
    """Runs each submission synchronously in this process."""

    def __init__(self, submissions):
        self.submissions = submissions

    def apply_async(self, func, args=()):
        if func.__name__ == "_run_with_start_tracking":
            self.submissions.append(args[2][4])
        return _FakeAsync(func, args)

    def close(self):
        pass

    terminate = join = close


def _faulting_work_group(
    gpu_map, fast_mode, err_ratio, in_data, tasks, verbose, return_status, queue
):
    """Measures candidates in order and faults on any named 'fault*'."""
    assert in_data[0] == len(tasks)
    tuner = importlib.import_module("aiter.utility.mp_tuner")
    rets = []
    for task in tasks:
        if task[0][1].startswith("fault"):
            raise _accelerator_error()
        ret = tuner._format_worker_result(task[0], 2.0, 0.0, "ok", return_status)
        rets.append(ret)
        queue.put(ret)
    return rets


class TestMpTunerResubmission(unittest.TestCase):
    """mp_tuner's parent loop with an in-process pool."""

    def run_tuner(self, names, work_group=_faulting_work_group, **kwargs):
        import queue as queue_module
        from unittest import mock

        tuner = importlib.import_module("aiter.utility.mp_tuner")
        submissions = []
        manager = mock.Mock(Queue=queue_module.Queue)

        def new_pool(parallel_num, start_times, pids):
            tuner._TASK_START_TIMES, tuner._TASK_PIDS = start_times, pids
            return _FakePool(submissions)

        tasks = [(("shape", name),) for name in names]
        with (
            mock.patch.object(tuner.torch.cuda, "device_count", return_value=1),
            mock.patch.object(tuner.mp, "set_start_method"),
            mock.patch.object(tuner.mp, "Manager", return_value=manager) as start,
            mock.patch.object(tuner, "_new_pool", new_pool),
            mock.patch.object(tuner, "work_group", work_group),
            mock.patch.object(tuner.time, "sleep"),
        ):
            results = tuner.mp_tuner(
                tasks, [(len(tasks), None)], 1, False, True, **kwargs
            )
        self.managers_started = start.call_count
        return results, [[t[0][1] for t in group] for group in submissions]

    def test_a_manager_starts_only_for_a_multi_candidate_group(self):
        for names, started in ((["a"], 0), (["a", "b"], 1)):
            with self.subTest(names=names):
                self.run_tuner(names)
                self.assertEqual(self.managers_started, started)

    def test_untyped_callers_keep_every_other_candidate(self):
        results, submissions = self.run_tuner(["a", "fault", "c", "d"])
        self.assertEqual(
            [(info[1], us) for info, us, _ in results],
            [("a", 2.0), ("fault", float("inf")), ("c", 2.0), ("d", 2.0)],
        )
        self.assertEqual(submissions, [["a", "fault", "c", "d"], ["c", "d"]])

    def test_a_stale_pid_map_restarts_the_pool_and_resubmits_the_group(self):
        attempts = []

        def remapped_work_group(gpu_map, *args):
            attempts.append(gpu_map)
            if len(attempts) == 1:
                raise KeyError("pid not in GPU map")
            return _faulting_work_group(gpu_map, *args)

        results, submissions = self.run_tuner(
            ["a", "b"], work_group=remapped_work_group
        )
        self.assertEqual(
            [(info[1], us) for info, us, _ in results], [("a", 2.0), ("b", 2.0)]
        )
        self.assertEqual(submissions, [["a", "b"], ["a", "b"]])

    def test_each_fault_consumes_a_candidate_until_the_budget(self):
        tuner = importlib.import_module("aiter.utility.mp_tuner")
        names = [f"fault{i}" for i in range(tuner._MAX_FAULTS_WITHOUT_PROGRESS + 2)]
        results, submissions = self.run_tuner(names, return_status=True)
        budget = tuner._MAX_FAULTS_WITHOUT_PROGRESS
        self.assertEqual(
            [status for *_, status, _ in results], ["crash"] * budget + ["not_run"] * 2
        )
        self.assertEqual(len(submissions), budget)
        self.assertIn("in a row", results[-1][4])


if __name__ == "__main__":
    unittest.main(verbosity=2)
