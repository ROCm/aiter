"""Regression tests for the sampling-kernel AOT compile driver."""

import pathlib
import sys
import unittest
from unittest.mock import Mock, patch

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from aiter.aot import sampling as driver


class SamplingAotTest(unittest.TestCase):
    def test_workers_return_none_after_compilation(self):
        with (
            patch.object(driver, "top_k_renorm_probs_compile") as top_k_renorm,
            patch.object(driver, "top_p_sampling_from_probs_compile") as top_p,
            patch.object(
                driver, "top_k_top_p_sampling_from_probs_compile"
            ) as top_k_top_p,
        ):
            self.assertIsNone(
                driver.process_top_k_renorm_config(
                    driver.TopKRenormConfig(vec_size=4, func_name="top_k_renorm_probs")
                )
            )
            self.assertIsNone(
                driver.process_top_p_sampling_config(
                    driver.TopPSamplingConfig(
                        vec_size=4,
                        deterministic=True,
                        func_name="top_p_sampling_from_probs",
                    )
                )
            )
            self.assertIsNone(
                driver.process_top_k_top_p_sampling_config(
                    driver.TopKTopPSamplingConfig(
                        vec_size=4,
                        deterministic=False,
                        func_name="top_k_top_p_sampling_from_probs",
                    )
                )
            )

        top_k_renorm.assert_called_once_with(4)
        top_p.assert_called_once_with(4, True)
        top_k_top_p.assert_called_once_with(4, False)

    def test_all_kernel_families_are_submitted_before_results_are_consumed(self):
        executor = Mock()
        executor.__enter__ = Mock(return_value=executor)
        executor.__exit__ = Mock(return_value=False)
        submitted = []

        def submit(_worker, configs):
            configs = list(configs)
            submitted.append(configs)

            def results():
                self.assertEqual(len(submitted), 3)
                yield from [None] * len(configs)

            return results()

        executor.map.side_effect = submit
        with (
            patch.object(
                driver.concurrent.futures,
                "ProcessPoolExecutor",
                return_value=executor,
            ) as process_pool,
            patch.object(driver, "get_worker_count_for", return_value=20) as workers,
        ):
            driver.main()

        self.assertEqual([len(configs) for configs in submitted], [4, 8, 8])
        workers.assert_called_once_with(20)
        process_pool.assert_called_once_with(
            max_workers=20,
            initializer=driver.configure_worker_subprocesses,
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
