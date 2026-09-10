"""Focused tests for PA-Gluon's centralized worker policy."""

import pathlib
import sys
import unittest
from unittest.mock import MagicMock, patch

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch

from csrc.cpp_itfs import utils as cpp_itfs_utils
from csrc.cpp_itfs.pa_gluon_aot import pa_decode_gluon_aot_prebuild as pa_gluon


class PaGluonProcessLimitTest(unittest.TestCase):
    @patch.object(pa_gluon, "get_worker_count_for", return_value=1)
    def test_pool_uses_shared_worker_policy(self, worker_count):
        executor = MagicMock()
        executor.__enter__.return_value = executor
        executor.__exit__.return_value = False
        executor.map.return_value = []

        with patch.object(
            pa_gluon.concurrent.futures,
            "ProcessPoolExecutor",
            return_value=executor,
        ) as pool:
            result = pa_gluon.run_multi_pa_gluon_test(
                block_sizes=[16],
                head_configs=[(1, 1)],
                context_length=[1],
                batch_sizes=[1],
                query_lengths=[1],
                quant_mode=["per_tensor"],
                trans_v=[False],
                kv_varlen=[False],
                compute_types_quant_q_and_kv=[[torch.bfloat16, False, False]],
                use_torch_flash_ref_options=[True],
                use_aot_impl_options=[True],
                context_partition_size_options=[256],
                sinks_options=[False],
                sliding_window_options=[0],
            )

        worker_count.assert_called_once_with(1)
        self.assertEqual(pool.call_args.kwargs["max_workers"], 1)
        self.assertIs(
            pool.call_args.kwargs["initializer"],
            pa_gluon.configure_worker_subprocesses,
        )
        self.assertEqual(len(result), 0)

    @patch.object(cpp_itfs_utils, "get_worker_count_for", return_value=1)
    def test_nested_make_uses_one_job(self, worker_count):
        self.assertEqual(
            cpp_itfs_utils._make_build_command(3),
            ["make", "build", "-j1"],
        )
        worker_count.assert_called_once_with(3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
