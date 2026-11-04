"""AOT driver coverage: config wiring, AOT cache reuse, and GPU worker policy."""

import inspect
import os
import pathlib
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, call, patch

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch

from aiter.aot import asm_mla_decode_fwd as driver
from aiter.jit.utils.chip_info import get_gfx
from csrc.cpp_itfs import utils as cpp_itfs_utils
from csrc.cpp_itfs.mla import asm_mla_decode_fwd as compile_module

# Importing pa_gluon probes the GPU arch through Triton, which fails on a GPU-less
# host; only its tests skip there, so the rest of the module still collects.
try:
    from csrc.cpp_itfs.pa_gluon_aot import pa_decode_gluon_aot_prebuild as pa_gluon
except (ImportError, RuntimeError):
    pa_gluon = None
_needs_pa_gluon = unittest.skipIf(pa_gluon is None, "needs a GPU for Triton")

# FlyDSL ships no Windows wheel; its tests skip there instead of breaking collection.
try:
    import flydsl
except ImportError:
    flydsl = None
_needs_flydsl = unittest.skipIf(flydsl is None, "needs FlyDSL")


class AsmMlaDecodeAotTest(unittest.TestCase):
    def test_config_fields_match_compile_api(self):
        compile_parameters = tuple(
            inspect.signature(compile_module.compile).parameters
        )[:6]

        self.assertEqual(driver.MLAConfig._fields, compile_parameters)


# The documented AOT -> run-only flow for one split-K A16W16 row, in fresh
# processes so the run-only one can only hit the disk cache. RUN_ONLY raises on
# a miss instead of compiling, so any drift between the AOT replay and the
# runtime launch (the split-K semaphore dtype once) fails here.
_HGEMM_CSV = (
    "gfx,cu_num,M,N,K,bias,dtype,outdtype,scaleAB,bpreshuffle,libtype,solidx,"
    "splitK,us,kernelName,err_ratio,tflops,bw\n"
    "gfx950,256,2,256,7168,False,torch.bfloat16,torch.bfloat16,False,False,"
    "flydsl,312,7,5.1366,"
    "flydsl_hgemm_abf16_wbf16_bf16_t16x32x128x7_ks7_w1x2x1_bias0_ktail0_gm0_pft_gfx950,"
    "0.0137,1.43,720.26\n"
)
_HGEMM_RUN_ONLY = """
import torch
from aiter.test_common import checkAllclose
from aiter.tuned_gemm import gemm_a16w16
# all-ones inputs: every split-K partial and the sum are exact in bf16
a = torch.ones((2, 7168), device="cuda", dtype=torch.bfloat16)
w = torch.ones((256, 7168), device="cuda", dtype=torch.bfloat16)
ref = (a.float() @ w.float().t()).to(torch.bfloat16)
out = gemm_a16w16(a, w, otype=torch.bfloat16)
err = checkAllclose(ref.float(), out.float(), rtol=0, atol=0, msg="hgemm ks7")
assert err == 0, f"mismatch ratio {err}"
"""


@_needs_flydsl
class FlydslGemmAotCacheReuseTest(unittest.TestCase):
    def setUp(self):
        if not torch.cuda.is_available() or get_gfx() != "gfx950":
            self.skipTest("needs a gfx950 GPU")

    def test_a16w16_aot_cache_serves_a_run_only_process(self):
        """A fresh RUN_ONLY process must load the AOT-built A16W16 kernel, not miss."""
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = os.path.join(tmp, "bf16_tuned_gemm.csv")
            with open(csv_path, "w") as f:
                f.write(_HGEMM_CSV)
            env = dict(
                os.environ,
                FLYDSL_RUNTIME_CACHE_DIR=os.path.join(tmp, "cache"),
                AITER_CONFIG_GEMM_BF16=csv_path,
                GPU_ARCHS="gfx950",
            )
            env.pop("FLYDSL_RUNTIME_RUN_ONLY", None)
            for name, cmd, extra_env in (
                ("AOT", ["-m", "aiter.aot.flydsl.gemm", "--csv", csv_path], {}),
                ("RUN_ONLY", ["-c", _HGEMM_RUN_ONLY], {"FLYDSL_RUNTIME_RUN_ONLY": "1"}),
            ):
                proc = subprocess.run(
                    [sys.executable, *cmd],
                    cwd=_REPO_ROOT,
                    env={**env, **extra_env},
                    capture_output=True,
                    text=True,
                    timeout=900,
                    check=False,
                )
                self.assertEqual(proc.returncode, 0, f"{name}:\n{proc.stderr[-4000:]}")


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

    @_needs_pa_gluon
    @patch.object(torch.cuda, "device_count", return_value=3)
    @patch(
        "csrc.cpp_itfs.pa_gluon_aot.pa_decode_gluon_aot_prebuild.get_gpu_worker_count",
        return_value=1,
    )
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

    @_needs_pa_gluon
    def test_prebuild_requires_a_visible_gpu_before_starting_workers(self):
        with (
            patch.object(pa_gluon.torch.cuda, "device_count", return_value=0),
            patch.object(pa_gluon.concurrent.futures, "ProcessPoolExecutor") as pool,
            self.assertRaisesRegex(RuntimeError, "requires at least one visible GPU"),
        ):
            self.run_one_config()
        pool.assert_not_called()

    @_needs_pa_gluon
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

    @_needs_pa_gluon
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
