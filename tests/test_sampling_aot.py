"""Regression tests for the sampling-kernel AOT compile driver."""

import pathlib
import sys
import unittest
from unittest.mock import patch

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from aiter.aot import sampling as driver


class SamplingAotTest(unittest.TestCase):
    def test_callbacks_call_the_expected_compilers(self):
        with (
            patch.object(driver, "top_k_renorm_probs_compile") as top_k_renorm,
            patch.object(driver, "top_p_sampling_from_probs_compile") as top_p,
            patch.object(
                driver, "top_k_top_p_sampling_from_probs_compile"
            ) as top_k_top_p,
        ):
            self.assertIs(
                driver.process_top_k_renorm_config(
                    driver.TopKRenormConfig(vec_size=4, func_name="top_k_renorm_probs")
                ),
                top_k_renorm.return_value,
            )
            self.assertIs(
                driver.process_top_p_sampling_config(
                    driver.TopPSamplingConfig(
                        vec_size=4,
                        deterministic=True,
                        func_name="top_p_sampling_from_probs",
                    )
                ),
                top_p.return_value,
            )
            self.assertIs(
                driver.process_top_k_top_p_sampling_config(
                    driver.TopKTopPSamplingConfig(
                        vec_size=4,
                        deterministic=False,
                        func_name="top_k_top_p_sampling_from_probs",
                    )
                ),
                top_k_top_p.return_value,
            )

        top_k_renorm.assert_called_once_with(4)
        top_p.assert_called_once_with(4, True)
        top_k_top_p.assert_called_once_with(4, False)

    def test_main_submits_every_kernel_family_to_the_shared_runner(self):
        with patch.object(driver, "run_compile_jobs") as run_jobs:
            driver.main()

        jobs = run_jobs.call_args.args[0]
        self.assertEqual(
            [process_config for process_config, _ in jobs],
            [
                driver.process_top_k_renorm_config,
                driver.process_top_p_sampling_config,
                driver.process_top_k_top_p_sampling_config,
            ],
        )
        self.assertEqual([len(configs) for _, configs in jobs], [4, 8, 8])


if __name__ == "__main__":
    unittest.main(verbosity=2)
