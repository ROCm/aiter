"""Focused tests for PA-Gluon's centralized worker policy."""

import pathlib
import sys
import unittest
from unittest.mock import MagicMock, call, patch

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch

from csrc.cpp_itfs import utils as cpp_itfs_utils
from csrc.cpp_itfs.pa_gluon_aot import pa_decode_gluon_aot_prebuild as pa_gluon


class PaGluonProcessLimitTest(unittest.TestCase):
    def run_one_config(self):
        return pa_gluon.run_multi_pa_gluon_test(
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

    @patch.object(pa_gluon.torch.cuda, "device_count", return_value=3)
    @patch.object(pa_gluon, "get_gpu_worker_count", return_value=1)
    def test_pool_uses_gpu_worker_policy(self, worker_count, device_count):
        executor = MagicMock()
        executor.__enter__.return_value = executor
        executor.__exit__.return_value = False
        executor.map.return_value = []

        with patch.object(
            pa_gluon.concurrent.futures,
            "ProcessPoolExecutor",
            return_value=executor,
        ) as pool:
            result = self.run_one_config()

        device_count.assert_called_once_with()
        worker_count.assert_called_once_with(1, 3)
        self.assertEqual(pool.call_args.kwargs["max_workers"], 1)
        self.assertIs(
            pool.call_args.kwargs["initializer"],
            pa_gluon._init_gpu_worker,
        )
        counter, devices = pool.call_args.kwargs["initargs"]
        self.assertEqual(counter.value, 0)
        self.assertEqual(devices, 3)
        self.assertEqual(len(result), 0)

    def test_prebuild_requires_a_visible_gpu_before_starting_workers(self):
        with (
            patch.object(pa_gluon.torch.cuda, "device_count", return_value=0),
            patch.object(pa_gluon.concurrent.futures, "ProcessPoolExecutor") as pool,
            self.assertRaisesRegex(RuntimeError, "requires at least one visible GPU"),
        ):
            self.run_one_config()
        pool.assert_not_called()

    def test_workers_are_bound_evenly_to_visible_devices(self):
        counter = pa_gluon.multiprocessing.get_context("spawn").Value("i", 0)
        with (
            patch.object(pa_gluon, "configure_worker_subprocesses") as configure,
            patch.object(pa_gluon.torch.cuda, "set_device") as set_device,
        ):
            for _ in range(7):
                pa_gluon._init_gpu_worker(counter, 3)

        self.assertEqual(configure.call_count, 7)
        self.assertEqual(
            set_device.call_args_list,
            [call(device) for device in [0, 1, 2, 0, 1, 2, 0]],
        )

    def test_kernel_allocations_use_the_workers_assigned_device(self):
        with (
            patch.object(pa_gluon.torch.cuda, "current_device", return_value=2),
            patch.object(pa_gluon.torch, "set_default_device") as default_device,
            patch.object(
                pa_gluon.torch, "tensor", side_effect=RuntimeError("allocation reached")
            ) as tensor,
            self.assertRaisesRegex(RuntimeError, "allocation reached"),
        ):
            pa_gluon.run_pa_gluon_test(
                context_length=1,
                batch_size=1,
                num_heads=(1, 1),
                head_size=128,
                block_size=16,
                compute_type=torch.bfloat16,
                query_length=1,
                quant_mode="per_tensor",
                context_partition_size=256,
                trans_v=False,
                kv_varlen=False,
                use_aot_impl=True,
                quant_q=False,
                quant_kv=False,
            )

        default_device.assert_called_once_with("cuda:2")
        self.assertEqual(tensor.call_args.kwargs["device"], "cuda:2")

    @patch.object(cpp_itfs_utils, "get_worker_count_for", return_value=1)
    def test_nested_build_uses_one_job_on_both_platforms(self, worker_count):
        for windows, command in ((False, ["make", "build"]), (True, ["ninja"])):
            with (
                self.subTest(windows=windows),
                patch.object(cpp_itfs_utils, "IS_WINDOWS", windows),
            ):
                self.assertEqual(cpp_itfs_utils._build_command(3), [*command, "-j1"])
        self.assertEqual(worker_count.call_args_list, [call(3), call(3)])


if __name__ == "__main__":
    unittest.main(verbosity=2)
