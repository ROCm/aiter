"""Regression tests for the MLA decode AOT compile driver.

The driver and compile API diverged when both were introduced on 2025-05-29 in
commit 01864fa8e2347421bc5c314de8b584777ea991ec. This driver previously had no
unit-test coverage, allowing the mismatched arguments to remain unnoticed.
"""

import inspect
import pathlib
import sys
import unittest
from unittest.mock import Mock, patch

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from aiter.aot import asm_mla_decode_fwd as driver
from csrc.cpp_itfs.mla import asm_mla_decode_fwd as compile_module


class AsmMlaDecodeAotTest(unittest.TestCase):
    def test_config_fields_match_compile_api(self):
        compile_parameters = tuple(
            inspect.signature(compile_module.compile).parameters
        )[:6]

        self.assertEqual(driver.MLAConfig._fields, compile_parameters)

    def test_process_config_matches_compile_api(self):
        compile_mock = Mock()
        config = driver.MLAConfig(
            gqa_ratio=16,
            page_size=1,
            q_dtype="__hip_bfloat16",
            kv_dtype="__hip_bfloat16",
            num_kv_splits=7,
            v_head_dim=512,
        )

        with patch.object(driver, "compile", compile_mock):
            self.assertIsNone(driver.process_config(config))

        compile_mock.assert_called_once_with(
            16,
            1,
            "__hip_bfloat16",
            "__hip_bfloat16",
            7,
            512,
        )

    def test_main_builds_all_gqa16_split_variants(self):
        executor = Mock()
        executor.__enter__ = Mock(return_value=executor)
        executor.__exit__ = Mock(return_value=False)
        executor.map.return_value = iter([None] * 16)

        with (
            patch.object(
                driver.concurrent.futures,
                "ProcessPoolExecutor",
                return_value=executor,
            ) as process_pool,
            patch.object(driver, "get_worker_count_for", return_value=1) as workers,
        ):
            driver.main()

        process_config, configs = executor.map.call_args.args
        self.assertIs(process_config, driver.process_config)
        self.assertEqual(
            [config.num_kv_splits for config in configs], list(range(1, 17))
        )
        self.assertEqual({config.gqa_ratio for config in configs}, {16})
        self.assertEqual({config.q_dtype for config in configs}, {"__hip_bfloat16"})
        self.assertEqual({config.kv_dtype for config in configs}, {"__hip_bfloat16"})
        process_pool.assert_called_once_with(
            max_workers=1,
            initializer=driver.configure_worker_subprocesses,
        )
        workers.assert_called_once_with(16)

    def test_main_surfaces_worker_compile_errors(self):
        executor = Mock()
        executor.__enter__ = Mock(return_value=executor)
        executor.__exit__ = Mock(return_value=False)

        def failed_result():
            raise RuntimeError("compile failed")
            yield

        executor.map.return_value = failed_result()
        with (
            patch.object(
                driver.concurrent.futures,
                "ProcessPoolExecutor",
                return_value=executor,
            ),
            patch.object(driver, "get_worker_count_for", return_value=1),
            self.assertRaisesRegex(RuntimeError, "compile failed"),
        ):
            driver.main()


if __name__ == "__main__":
    unittest.main(verbosity=2)
