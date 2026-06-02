# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Auto-tuner for the gfx1250 MXScale (mxfp8) dense GEMM.

FlyDSL is the only backend, so this driver is a FlyDSL-only subclass of
GemmCommonTuner: for every untuned ``(M, N, K)`` shape it benchmarks the
candidate kernels from ``flydsl_gemm_mxscale_gfx1250_common.kernels_list`` and
writes the winner (its encoded ``kernelName``) into the ``mxscale_gfx1250`` tuned
CSV. The public ``aiter.gemm_a8w8_mxscale`` op then routes to that kernel
automatically. MX A8W4 is not integrated this round, so this driver tunes
``--data_format fp8`` only.

Usage::

    python aiter/ops/flydsl/gemm_tune/gemm_mxscale_gfx1250_tune.py \
        --untune_file aiter/configs/a8w8_mxscale_untuned_gemm.csv \
        --tune_file   aiter/configs/a8w8_mxscale_tuned_gemm.csv
"""

import torch

from aiter import dtypes
from aiter.jit.core import AITER_CONFIG_GEMM_MXSCALE, AITER_ROOT_DIR
from aiter.utility.base_tuner import GemmCommonTuner
from aiter.utility.mp_tuner import mp_tuner
from aiter.utility import fp4_utils
from aiter.ops.quant import per_1x32_f8_scale_f8_quant

from aiter.ops.flydsl.utils import is_flydsl_available
from aiter.ops.flydsl.mxscale_layout import SCALE_BLOCK
from aiter.ops.flydsl.gemm_tune.flydsl_gemm_mxscale_gfx1250_common import (
    _build_kernels_list,
    kernel_fits_shape,
)

if is_flydsl_available():
    from aiter.ops.flydsl.mxscale_gemm import (
        flydsl_mxscale_gemm,
        shuffle_weight_mxscale,
    )

# Single source of truth for the data_format -> q_dtype_w CSV key mapping.
from aiter.ops.gemm_op_a8w8 import _Q_DTYPE_W

_OUT_TORCH = {"bf16": torch.bfloat16, "f16": torch.float16}


def _dequant_fp8(q, s):
    qf = q.view(torch.float8_e4m3fn).to(torch.float32)
    sf = fp4_utils.e8m0_to_f32(s.view(torch.uint8)).to(torch.float32)
    return qf * sf.repeat_interleave(SCALE_BLOCK, dim=1)


def run_torch(x, weight, x_scale, w_scale, dtype=torch.bfloat16):
    """Dequant reference GEMM (fp32 math) for the tuner accuracy comparison."""
    a_f = _dequant_fp8(x, x_scale)
    b_f = _dequant_fp8(weight, w_scale)
    return (a_f @ b_f.t()).to(dtype)


def generate_data(m, n, k, seed, out_dtype="bf16", device="cuda"):
    """Build quantized A/B/scales + an output buffer for one tuning trial."""
    torch.manual_seed(seed)
    a = torch.randn((m, k), dtype=torch.bfloat16, device=device)
    b = torch.randn((n, k), dtype=torch.bfloat16, device=device)
    x, x_scale = per_1x32_f8_scale_f8_quant(
        a, quant_dtype=dtypes.fp8, scale_type=dtypes.fp8_e8m0, shuffle=False
    )
    weight, w_scale = per_1x32_f8_scale_f8_quant(
        b, quant_dtype=dtypes.fp8, scale_type=dtypes.fp8_e8m0, shuffle=False
    )
    out = torch.empty(m, n, dtype=_OUT_TORCH[out_dtype], device=device)
    return {
        "x": x,
        "weight": weight,
        "x_scale": x_scale,
        "w_scale": w_scale,
        "out": out,
    }


def _shuffled_weight_for(weight, w_scale, kernel_name, data_format):
    """Pre-shuffle (B, w_scale) once per kernel_name, cached on the weight tensor.

    The tuner generates input data once per shape and reuses it across every
    candidate kernel and every timed iteration. Weight preshuffle is a multi-MB
    layout pass that must not be charged to the kernel's measured latency, so we
    cache the shuffled result (keyed by kernel_name, since the B layout depends on
    the kernel's N/K tiling) on the shared raw weight tensor. The first (warmup)
    call populates it; the timed iterations hit the cache and pass an already
    ``is_shuffled`` weight straight through.
    """
    cache = getattr(weight, "_mxscale_tuner_shuf", None)
    if cache is None:
        cache = {}
        weight._mxscale_tuner_shuf = cache
    hit = cache.get(kernel_name)
    if hit is None:
        hit = shuffle_weight_mxscale(
            weight, w_scale, data_format=data_format, kernel_name=kernel_name
        )
        cache[kernel_name] = hit
    return hit


def run_gemm_mxscale(x, weight, x_scale, w_scale, out, kernel_name, data_format):
    """Run flydsl_mxscale_gemm with a fixed kernel_name; writes into out."""
    weight_s, w_scale_s = _shuffled_weight_for(
        weight, w_scale, kernel_name, data_format
    )
    flydsl_mxscale_gemm(
        x,
        weight_s,
        x_scale,
        w_scale_s,
        data_format=data_format,
        out=out,
        kernel_name=kernel_name,
    )
    return out


class GemmMxScaleGfx1250Tuner(GemmCommonTuner):
    ARG_DEFAULTS = {
        **GemmCommonTuner.ARG_DEFAULTS,
        "tune_file": f"{AITER_CONFIG_GEMM_MXSCALE}",
        "untune_file": f"{AITER_ROOT_DIR}/aiter/configs/a8w8_mxscale_untuned_gemm.csv",
        "config_env_name": "AITER_CONFIG_GEMM_MXSCALE",
    }

    def _clear_op_caches(self):
        from aiter.ops import gemm_op_a8w8 as _op

        _op.clear_mxscale_config_cache()

    def _setup_specific_arguments(self):
        self.parser.add_argument(
            "--data_format",
            type=str,
            default="fp8",
            choices=["fp8"],  # MXFP8 only; MX A8W4 not integrated this round
            help="MXScale data format to tune (fp8=mxfp8)",
        )
        self.parser.add_argument(
            "--out_dtype",
            type=str,
            default="bf16",
            choices=["bf16", "f16"],
            help="Output dtype to tune (run once per dtype your model needs)",
        )

    def calculate(self, results, bpes=None):
        # MXFP8: A and B are 1 byte/element, output 2 bytes/element.
        if bpes is None:
            bpes = (1, 1, 2)
        return super().calculate(results, bpes=bpes)

    def result_to_df(self, results):
        # info carries a trailing libtype (5-tuple), so the base 4-tuple
        # unpack does not apply; mirror the a8w8 bpreshuffle tuner.
        import pandas as pd

        resultdf = pd.DataFrame(columns=self.columns)
        for el in results:
            info, time, err_ratio = el
            keys, kernelId, splitK, kernelName, libtype = info
            if time == self.INVALID_TIME:
                kernelName = "None"
            elif kernelName == "":
                resolved = self.getKernelName(kernelId, libtype)
                kernelName = (
                    "None" if (resolved is None or pd.isna(resolved)) else str(resolved)
                )
            tflops, bw = self.calculate(el)
            key_dict = dict(zip(self.keys, keys))
            key_dict.update(
                {
                    "libtype": [libtype],
                    "kernelId": [kernelId],
                    "splitK": [splitK],
                    "us": [time],
                    "kernelName": [kernelName],
                    "errRatio": [err_ratio],
                    "tflops": [tflops],
                    "bw": [bw],
                }
            )
            temp = pd.DataFrame(key_dict)
            resultdf = (
                temp
                if resultdf.empty
                else pd.concat([resultdf, temp], ignore_index=True)
            )
        return resultdf

    def getKernelName(self, kernelId, libtype="flydsl"):
        """Encoded kernel name for kernelId; libtype is ignored (always flydsl)."""
        kernels = self._kernels_for(self._data_format, self._out_dtype)
        if kernelId not in kernels:
            return None
        return kernels[kernelId].name

    def _kernels_for(self, data_format, out_dtype):
        cache = getattr(self, "_kernels_cache", None)
        if cache is None:
            cache = {}
            self._kernels_cache = cache
        key = (data_format, out_dtype)
        if key not in cache:
            cache[key] = _build_kernels_list(
                data_format=data_format, out_dtype=out_dtype
            )
        return cache[key]

    def get_mxscale_tune_task(self, info_keys, data_format, out_dtype, seed, args):
        gfx, cu_num, M, N, K, q_dtype_w = info_keys
        if (not is_flydsl_available()) or ("flydsl_mxscale_gemm" not in globals()):
            return []
        kernels = self._kernels_for(data_format, out_dtype)
        run_keys = ["x", "weight", "x_scale", "w_scale", "out"]
        ref_keys = ["x", "weight", "x_scale", "w_scale"]
        tasks = []
        for i in sorted(kernels.keys()):
            ki = kernels[i]
            if not kernel_fits_shape(ki, M, N, K):
                continue
            kernel_name = ki.name
            info = (info_keys, i, 0, kernel_name, "flydsl")
            tasks.append(
                (
                    info,
                    generate_data,
                    (M, N, K, seed, out_dtype),
                    run_gemm_mxscale,
                    (run_keys, kernel_name, data_format),
                    {"num_warmup": args.warmup, "num_iters": args.iters},
                    run_torch,
                    (ref_keys, _OUT_TORCH[out_dtype]),
                    {},
                    None,
                    1e-2,
                    1e-2,
                    None,
                    None,
                    ("out",),
                )
            )
        return tasks

    def tune(self, untunedf, tunedf, args):
        self._data_format = args.data_format
        self._out_dtype = args.out_dtype
        mp_num = args.mp
        shape_grouped = args.shape_grouped
        errRatio = args.errRatio
        cu_num = self.get_cu_num()
        gfx = self.get_gfx()
        task = []
        tasks_data = []
        seed = 0
        # q_dtype_w is keyed off the data format being tuned, so the tuned-CSV
        # key matches what the op looks up (gemm_op_a8w8._Q_DTYPE_W) regardless
        # of the value in the untuned CSV.
        q_dtype_w = _Q_DTYPE_W[args.data_format]
        # base pre_process dedups against the untuned CSV's (wrong) q_dtype_w;
        # skip shapes already tuned for this format's real q_dtype_w here.
        already = set()
        if tunedf is not None and not tunedf.empty and "q_dtype_w" in tunedf.columns:
            sub = tunedf[tunedf["q_dtype_w"] == q_dtype_w]
            already = {(int(r.M), int(r.N), int(r.K)) for r in sub.itertuples()}
        for i in range(len(untunedf)):
            M = untunedf.loc[i, "M"]
            N = untunedf.loc[i, "N"]
            K = untunedf.loc[i, "K"]
            if (int(M), int(N), int(K)) in already:
                continue
            seed += 1
            prev = len(task)
            info_keys = (gfx, cu_num, M, N, K, q_dtype_w)
            task.extend(
                self.get_mxscale_tune_task(
                    info_keys, args.data_format, args.out_dtype, seed, args
                )
            )
            tasks_data.append((len(task) - prev, ()))
        ret = []
        if task:
            ret = mp_tuner(
                task,
                tasks_data,
                mp_num,
                False,
                shape_grouped,
                errRatio,
                timeout=args.timeout,
                verbose=args.verbose,
            )
        return ret

    def run_config(self, args):
        """End-to-end check of the routed op against a dequant reference.

        Mirrors the a8w8 bpreshuffle tuner: for each untuned shape it runs the
        public op (which routes through the tuned CSV) and compares to the torch
        reference, reporting per-shape latency + accuracy status.
        """
        import aiter
        from aiter.test_common import run_perftest, checkAllclose

        out_dtype = args.out_dtype
        allowed = args.errRatio
        op = aiter.gemm_a8w8_mxscale
        results = []
        for i in range(len(self.untunedf)):
            row = self.untunedf.iloc[i]
            M, N, K = int(row["M"]), int(row["N"]), int(row["K"])
            shape_str = f"M{M}_N{N}_K{K}_fp8_{out_dtype}"
            try:
                d = generate_data(M, N, K, seed=0, out_dtype=out_dtype)
                ref = run_torch(
                    d["x"],
                    d["weight"],
                    d["x_scale"],
                    d["w_scale"],
                    _OUT_TORCH[out_dtype],
                )
                out, us = run_perftest(
                    op,
                    d["x"],
                    d["weight"],
                    d["x_scale"],
                    d["w_scale"],
                    dtype=_OUT_TORCH[out_dtype],
                )
                err = checkAllclose(out, ref, msg=f"run_config {shape_str}")
                status = "ok" if err <= allowed else f"mismatch:err_ratio={err:.6g}"
                results.append({"shape": shape_str, "e2e_us": us, "status": status})
            except Exception as e:  # noqa: BLE001
                results.append(
                    {"shape": shape_str, "e2e_us": -1, "status": f"error:{e}"}
                )
            finally:
                torch.cuda.empty_cache()
        return results


if __name__ == "__main__":
    key = ["gfx", "cu_num", "M", "N", "K", "q_dtype_w"]
    resultList = [
        "libtype",
        "kernelId",
        "splitK",
        "us",
        "kernelName",
        "tflops",
        "bw",
        "errRatio",
    ]
    tuner = GemmMxScaleGfx1250Tuner(
        "GemmMxScaleGfx1250Tuner",
        key=key,
        resultList=resultList,
        description="Auto-tuner for gfx1250 MXScale (mxfp8) dense GEMM kernels",
    )
    args = tuner.parse_args()
    tuner.run(args, False)
