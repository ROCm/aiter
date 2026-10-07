"""Tests for the shared AOT compile-job runner."""

import pathlib
import sys
import unittest
from unittest.mock import Mock, patch

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import aiter_worker_limits as worker_limits


class _RecordingExecutor:
    """Stand-in pool that records how many families exist when consumed."""

    def __init__(self, consumed):
        self.consumed = consumed
        self.maps = []

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False

    def map(self, process_config, configs):
        configs = list(configs)
        self.maps.append((process_config, configs))

        def results():
            self.consumed.append(len(self.maps))
            yield from [None] * len(configs)

        return results()


class AotRunnerTest(unittest.TestCase):
    def test_run_configs_bounds_the_pool_and_initializes_workers(self):
        executor = _RecordingExecutor([])
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
        self.assertEqual(executor.maps, [(process_config, [1, 2, 3])])

    def test_run_compile_jobs_submits_every_family_before_consuming(self):
        consumed = []
        executor = _RecordingExecutor(consumed)
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
        # Every family is mapped before the first iterator is consumed.
        self.assertEqual(consumed, [3, 3, 3])
        self.assertEqual(
            [configs for _, configs in executor.maps], [[1], [2, 3], [4, 5, 6]]
        )

    def test_run_configs_propagates_worker_failures(self):
        executor = Mock()
        executor.__enter__ = Mock(return_value=executor)
        executor.__exit__ = Mock(return_value=False)

        def failed_results():
            raise RuntimeError("compile failed")
            yield

        executor.map.return_value = failed_results()
        with (
            patch.object(worker_limits, "ProcessPoolExecutor", return_value=executor),
            self.assertRaisesRegex(RuntimeError, "compile failed"),
        ):
            worker_limits.run_configs([1], Mock())


if __name__ == "__main__":
    unittest.main(verbosity=2)
