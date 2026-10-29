"""AOT driver coverage: config wiring, cache signatures, and GPU worker policy."""

import inspect
import pathlib
import sys
import unittest
from contextlib import nullcontext
from unittest.mock import MagicMock, call, patch

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch

from aiter.aot import asm_mla_decode_fwd as driver
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
    from aiter.aot.flydsl import gemm
    from aiter.aot.flydsl.common import override_env
    from aiter.ops.flydsl.kernels import gemm_a16w16_gfx950 as gemm_kernel
except ImportError:
    gemm = None
_needs_flydsl = unittest.skipIf(gemm is None, "needs FlyDSL")


class AsmMlaDecodeAotTest(unittest.TestCase):
    def test_config_fields_match_compile_api(self):
        compile_parameters = tuple(
            inspect.signature(compile_module.compile).parameters
        )[:6]

        self.assertEqual(driver.MLAConfig._fields, compile_parameters)


def _flydsl_cache_key(jit_func, dispatch_args):
    """The key FlyDSL files a compile of ``jit_func`` under, launch stream excluded.

    Built by FlyDSL itself (pinned in requirements.txt), so every input of the real
    key counts: per-tensor type, dtype, alignment, 32-bit-stride flag, shape and
    stride (dynamic dims as -1), scalar types, the constexpr param, the target, the
    cache-invalidating env vars and the captured globals.
    """
    jit_func._ensure_sig()
    bound = jit_func._sig.bind_partial(*dispatch_args[:-1])
    return jit_func._build_full_cache_key(bound.arguments)


@_needs_flydsl
class FlydslGemmAotTest(unittest.TestCase):
    def test_a16w16_aot_matches_runtime_cache_signature(self):
        """AOT must pass the runtime's argument signature to reuse its compiled cache."""
        m, n, k = 8, 1536, 7168
        gemm_kernel.get_split_k_buffers.cache_clear()
        self.addCleanup(gemm_kernel.get_split_k_buffers.cache_clear)
        for dtype, torch_dtype in (("bf16", torch.bfloat16), ("fp16", torch.float16)):
            for split_k in (1, 7):
                config = {
                    "block_m": 128,
                    "block_n": 128,
                    "block_k": 64,
                    "stages": 2,
                    "split_k": split_k,
                    "m_waves": 2,
                    "n_waves": 2,
                    "k_waves": 1,
                    "group_m": 0,
                    "use_half_tile_interleaved": False,
                }
                # Pin the target so the LDS lookup does not read the host GPU.
                with (
                    self.subTest(dtype=dtype, split_k=split_k),
                    override_env("FLYDSL_GPU_ARCH", "gfx950"),
                    patch.object(gemm, "run_cached") as aot_call,
                    patch.object(gemm_kernel, "run_cached") as runtime_call,
                    patch.object(
                        gemm_kernel, "persistent_alloc", return_value=nullcontext()
                    ),
                ):
                    gemm._compile_hgemm_to_cache(
                        m=m,
                        n=n,
                        k=k,
                        dtype=dtype,
                        out_dtype=dtype,
                        has_bias=False,
                        target_gfx="gfx950",
                        **config,
                    )
                    # CPU inputs keep the split-K buffers off the GPU.
                    gemm_kernel.gemm_a16w16(
                        torch.empty((m, k), device="cpu", dtype=torch_dtype),
                        torch.empty((n, k), device="cpu", dtype=torch_dtype).t(),
                        user_kwargs=config,
                        stream="stream",
                    )
                    aot_args, runtime_args = (
                        c.call_args.kwargs["dispatch_args"]
                        for c in (aot_call, runtime_call)
                    )
                    runtime_key = _flydsl_cache_key(
                        gemm_kernel.gemm_a16w16_gfx950, runtime_args
                    )
                    self.assertEqual(
                        _flydsl_cache_key(gemm_kernel.gemm_a16w16_gfx950, aot_args),
                        runtime_key,
                    )
                    # The key must see a semaphore dtype drift (what this test guards)
                    # and a stride drift (B passed with A's leading dimension).
                    for index, arg in (
                        (4, aot_args[4].to(torch.float32)),
                        (
                            2,
                            gemm_kernel._dynamic_tensor_arg(
                                torch.empty((8, 8), dtype=torch_dtype), 1
                            ),
                        ),
                    ):
                        drifted = list(aot_args)
                        drifted[index] = arg
                        self.assertNotEqual(
                            _flydsl_cache_key(gemm_kernel.gemm_a16w16_gfx950, drifted),
                            runtime_key,
                        )


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
