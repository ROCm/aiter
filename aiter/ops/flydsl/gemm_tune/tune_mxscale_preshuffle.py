# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Tuner for the FlyDSL MXFP4/MXFP6/MXFP8 preshuffle GEMM (gfx950).

Mechanism mirrors csrc/ck_gemm_a8w8_bpreshuffle/gemm_a8w8_bpreshuffle_tune.py:
a GemmCommonTuner subclass driving mp_tuner (multiprocess compile + bench +
checkAllclose). Only the FlyDSL libtype is tuned (this is a standalone FlyDSL op).

Usage:
    HIP_VISIBLE_DEVICES=0 python -m aiter.ops.flydsl.gemm_tune.tune_mxscale_preshuffle \
        --untune_file aiter/configs/mxscale_preshuffle_untuned_gemm.csv
"""

import torch
import pandas as pd

from aiter import dtypes
from aiter.jit.core import AITER_CONFIG_GEMM_MXSCALE_PRESHUFFLE
from aiter.utility.base_tuner import GemmCommonTuner
from aiter.utility.mp_tuner import mp_tuner
from aiter.ops.quant import per_1x32_f4_quant, per_1x32_f8_scale_f8_quant
from aiter.ops.shuffle import shuffle_weight, shuffle_scale_a16w4
from aiter.utility import fp4_utils
from aiter.ops.flydsl.utils import is_flydsl_available
from aiter.ops.flydsl.gemm_tune.flydsl_gemm_mxscale_preshuffle_common import (
    kernels_list,
    candidates_for,
)

if is_flydsl_available():
    from aiter.ops.flydsl.mxscale_preshuffle_kernels import (
        flydsl_mxscale_preshuffle_gemm,
        gemm_mxscale_preshuffle,
    )


def _quant(x_f, dtype):
    if dtype == "fp8":
        return per_1x32_f8_scale_f8_quant(
            x_f, quant_dtype=dtypes.fp8, scale_type=dtypes.fp8_e8m0
        )
    if dtype == "fp4":
        return per_1x32_f4_quant(x_f, quant_dtype=dtypes.fp4x2)
    raise ValueError(
        f"tune supports fp4/fp8 operands only (aiter has no fp6 quant); got {dtype!r}"
    )


def _dequant(codes, scale, dtype, rows):
    sc = fp4_utils.e8m0_to_f32(scale[:rows].repeat_interleave(32, dim=1))
    if dtype == "fp8":
        return codes.float() * sc
    return fp4_utils.mxfp4_to_f32(codes) * sc  # fp4


def generate_data(m, n, k, seed, a_dtype, b_dtype, dtype=dtypes.bf16, device="cuda"):
    torch.manual_seed(seed)
    ma, na = (m + 31) // 32 * 32, (n + 31) // 32 * 32
    a_f = torch.zeros(ma, k, dtype=torch.float32, device=device)
    b_f = torch.zeros(na, k, dtype=torch.float32, device=device)
    a_f[:m] = torch.randn(m, k, device=device)
    b_f[:n] = torch.randn(n, k, device=device)
    a_q, sa = _quant(a_f, a_dtype)
    b_q, sb = _quant(b_f, b_dtype)
    a_codes, b_codes = a_q[:m], b_q[:n]
    b_shuf = shuffle_weight(b_codes, layout=(16, 16))
    a_scale = shuffle_scale_a16w4(sa, 1, False)
    b_scale = shuffle_scale_a16w4(sb, 1, False)
    a_deq = _dequant(a_codes, sa, a_dtype, m)
    b_deq = _dequant(b_codes, sb, b_dtype, n)
    out = torch.empty(m, n, dtype=dtype, device=device)
    return {
        "A": a_codes,
        "B": b_shuf,
        "a_scale": a_scale,
        "b_scale": b_scale,
        "out": out,
        "a_deq": a_deq,
        "b_deq": b_deq,
    }


def run_gemm_flydsl(A, B, a_scale, b_scale, out, kernel_id, a_dtype, b_dtype):
    ki = kernels_list[kernel_id]
    flydsl_mxscale_preshuffle_gemm(
        A,
        B,
        a_scale,
        b_scale,
        out,
        a_dtype=a_dtype,
        b_dtype=b_dtype,
        tile_m=ki.tile_m,
        tile_n=ki.tile_n,
        tile_k=ki.tile_k,
        waves_per_eu=ki.waves_per_eu,
        xcd_swizzle=ki.xcd_swizzle,
        split_k=ki.split_k,
    )
    return out


def run_torch(a_deq, b_deq, dtype=dtypes.bf16):
    return (a_deq @ b_deq.T).to(dtype)


class MxscalePreShuffleTuner(GemmCommonTuner):
    ARG_DEFAULTS = {
        **GemmCommonTuner.ARG_DEFAULTS,
        "tune_file": f"{AITER_CONFIG_GEMM_MXSCALE_PRESHUFFLE}",
        "untune_file": "aiter/configs/mxscale_preshuffle_untuned_gemm.csv",
        "config_env_name": "AITER_CONFIG_GEMM_MXSCALE_PRESHUFFLE",
    }

    def _clear_op_caches(self):
        from aiter.ops.flydsl import mxscale_preshuffle_kernels as _op

        _op._TUNED_CACHE.clear()

    def _setup_specific_arguments(self):
        pass  # only the FlyDSL libtype; no extra CLI knobs

    def calculate(self, results, bpes=(1, 1, 2)):
        # (inbpe, w_bpe, outbpe); fp8=1 byte/code, bf16 out=2
        return super().calculate(results, bpes=bpes)

    def getKernelName(self, kernelId, libtype="flydsl"):
        ki = kernels_list.get(kernelId)
        return ki.name if ki is not None else None

    def get_flydsl_mxscale_tune_task(self, info_keys, seed, args):
        gfx, cu_num, M, N, K, a_dtype, b_dtype = info_keys
        if (
            not is_flydsl_available()
            or "flydsl_mxscale_preshuffle_gemm" not in globals()
        ):
            return []
        gemm_keys = ["A", "B", "a_scale", "b_scale", "out"]
        ref_keys = ["a_deq", "b_deq"]
        tasks = []
        for kid, ki in candidates_for(a_dtype, b_dtype, M, N, K):
            info = (info_keys, kid, ki.split_k, ki.name, "flydsl")
            tasks.append(
                (
                    info,
                    generate_data,
                    (M, N, K, seed, a_dtype, b_dtype),
                    run_gemm_flydsl,
                    (gemm_keys, kid, a_dtype, b_dtype),
                    {"num_warmup": args.warmup, "num_iters": args.iters},
                    run_torch,
                    (ref_keys, dtypes.bf16),
                    {},
                    None,
                    1e-2,
                    0.01,
                    None,
                    None,
                    ("out",),
                )
            )
        return tasks

    def tune(self, untunedf, tunedf, args):
        mp_num = args.mp
        shape_grouped = args.shape_grouped
        errRatio = args.errRatio
        cu_num = self.get_cu_num()
        gfx = self.get_gfx()
        task = []
        tasks_data = []
        seed = 0
        for i in range(len(untunedf)):
            M = int(untunedf.loc[i, "M"])
            N = int(untunedf.loc[i, "N"])
            K = int(untunedf.loc[i, "K"])
            a_dtype = untunedf.loc[i, "a_dtype"]
            b_dtype = untunedf.loc[i, "b_dtype"]
            seed += 1
            info_keys = (gfx, cu_num, M, N, K, a_dtype, b_dtype)
            shape_tasks = self.get_flydsl_mxscale_tune_task(info_keys, seed, args)
            if not shape_tasks:
                # no legal candidate for this shape (e.g. K%128!=0); skip so
                # mp_tuner's shape grouping stays consistent with the task list.
                print(
                    f"[mxscale] skip untuned shape M={M} N={N} K={K} {a_dtype}/{b_dtype}: no legal tile"
                )
                continue
            task.extend(shape_tasks)
            tasks_data.append((len(shape_tasks), ()))
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

    def result_to_df(self, results):
        resultdf = pd.DataFrame(columns=self.columns)
        for el in results:
            info, time, err_ratio = el
            keys, kernelId, splitK, kernelName, libtype = info
            kernelName = (
                "None"
                if time == self.INVALID_TIME
                else (
                    self.getKernelName(kernelId, libtype)
                    if kernelName == ""
                    else kernelName
                )
            )
            tflops, bw = self.calculate(el)
            key_dict = dict(zip(self.keys, keys))
            if len(results) == self.topk:
                print(
                    f"Tuning result for {str(key_dict).strip('{}')} is kernelId={kernelId} "
                    f"{kernelName} {splitK=}, {time}us, {err_ratio=}, {tflops=} TFLOPS, {bw=} GB/s"
                )
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

    def run_config(self, args):
        from aiter.test_common import run_perftest, checkAllclose

        untunedf = self.untunedf
        results = []
        for i in range(len(untunedf)):
            row = untunedf.iloc[i]
            M, N, K = int(row["M"]), int(row["N"]), int(row["K"])
            a_dtype, b_dtype = row["a_dtype"], row["b_dtype"]
            shape_str = f"({M}, {N}, {K}, {a_dtype}, {b_dtype})"
            allowed_err_ratio, allowed_err_ratio_desc = (
                self._get_run_config_err_ratio_limit(row, args)
            )
            try:
                gd = generate_data(M, N, K, 0, a_dtype, b_dtype)

                def _dispatch(A, B, a_scale, b_scale, out):
                    return gemm_mxscale_preshuffle(
                        A, B, a_scale, b_scale, out, a_dtype=a_dtype, b_dtype=b_dtype
                    )

                out, us = run_perftest(
                    _dispatch,
                    gd["A"],
                    gd["B"],
                    gd["a_scale"],
                    gd["b_scale"],
                    gd["out"],
                    num_warmup=args.warmup,
                    num_iters=args.iters,
                )
                ref = run_torch(gd["a_deq"], gd["b_deq"], dtype=dtypes.bf16)
                err_ratio = checkAllclose(
                    out.to(dtypes.bf16), ref, msg=f"run_config {shape_str}"
                )
                status = (
                    "ok"
                    if err_ratio <= allowed_err_ratio
                    else f"mismatch:err_ratio={err_ratio:.6g}(>{allowed_err_ratio_desc})"
                )
                results.append({"shape": shape_str, "e2e_us": us, "status": status})
            except Exception as e:
                results.append(
                    {"shape": shape_str, "e2e_us": -1, "status": f"error:{e}"}
                )
            finally:
                torch.cuda.empty_cache()
        return results


if __name__ == "__main__":
    key = ["gfx", "cu_num", "M", "N", "K", "a_dtype", "b_dtype"]
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
    tuner = MxscalePreShuffleTuner(
        "MxscalePreShuffleTuner",
        key=key,
        resultList=resultList,
        description="gen API for FlyDSL mxscale preshuffle GEMM (a4w4/a6w4/a8w8)",
    )
    args = tuner.parse_args()
    tuner.run(args, False)
