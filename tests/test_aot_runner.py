"""Tests for the shared AOT compile-job runner."""

import ctypes
import os
import pathlib
import sys
import unittest
from unittest.mock import Mock, patch

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import aiter_worker_limits as worker_limits


def _compile_pointer(config):
    if os.environ.get("AITER_MAX_JOBS") != "1":
        raise RuntimeError("nested compilation was not bounded")
    return ctypes.pointer(ctypes.c_int(config))


def _compile_failure(config):
    raise RuntimeError(f"compile failed: {config}")


class _RecordingExecutor:
    """Execute jobs locally while recording the complete submitted sequence."""

    def __init__(self):
        self.jobs = []

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False

    def map(self, process_job, jobs):
        self.jobs = list(jobs)
        return map(process_job, self.jobs)


class AotRunnerTest(unittest.TestCase):
    def test_run_configs_bounds_the_pool_and_initializes_workers(self):
        executor = _RecordingExecutor()
        process_config = Mock()
        with (
            patch.object(
                worker_limits, "ProcessPoolExecutor", return_value=executor
            ) as pool,
            patch.object(
                worker_limits, "get_worker_count_for", return_value=7
            ) as workers,
        ):
            worker_limits.run_configs([1, 2, 3], process_config)

        workers.assert_called_once_with(3)
        pool.assert_called_once_with(
            max_workers=7,
            initializer=worker_limits.configure_worker_subprocesses,
        )
        self.assertEqual(
            executor.jobs, [(process_config, config) for config in [1, 2, 3]]
        )
        self.assertEqual(process_config.call_count, 3)

    def test_run_compile_jobs_submits_every_family_before_consuming(self):
        executor = _RecordingExecutor()
        families = [
            (Mock(), [1]),
            (Mock(), [2, 3]),
            (Mock(), [4, 5, 6]),
        ]
        with (
            patch.object(worker_limits, "ProcessPoolExecutor", return_value=executor),
            patch.object(
                worker_limits, "get_worker_count_for", return_value=6
            ) as workers,
        ):
            worker_limits.run_compile_jobs(families)

        workers.assert_called_once_with(6)
        self.assertEqual(
            executor.jobs,
            [
                (process_config, config)
                for process_config, configs in families
                for config in configs
            ],
        )
        self.assertEqual(
            [process_config.call_count for process_config, _ in families], [1, 2, 3]
        )

    def test_real_pool_keeps_ctypes_results_in_children(self):
        with (
            patch.dict(os.environ, {"AITER_MAX_JOBS": "2"}),
            patch.object(worker_limits, "get_worker_count_for", return_value=2),
        ):
            worker_limits.run_configs([1, 2, 3], _compile_pointer)
            self.assertEqual(os.environ["AITER_MAX_JOBS"], "2")

    def test_real_pool_propagates_compile_failures(self):
        with (
            patch.object(worker_limits, "get_worker_count_for", return_value=1),
            self.assertRaisesRegex(RuntimeError, "compile failed: 7"),
        ):
            worker_limits.run_configs([7], _compile_failure)


if __name__ == "__main__":
    unittest.main(verbosity=2)
