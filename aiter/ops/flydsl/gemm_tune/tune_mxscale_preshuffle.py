# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Tune the gfx950 FlyDSL MX-scale preshuffle GEMM."""

from typing import ClassVar

import pandas as pd
import torch

from aiter import dtypes
from aiter.jit.core import AITER_CONFIG_GEMM_MXSCALE_PRESHUFFLE
from aiter.ops.flydsl.gemm_tune.flydsl_gemm_mxscale_preshuffle_common import (
    candidates_for,
    kernels_list,
)
from aiter.ops.flydsl.utils import is_flydsl_available
from aiter.ops.quant import per_1x32_f4_quant, per_1x32_f8_scale_f8_quant
from aiter.ops.shuffle import shuffle_scale_a16w4, shuffle_weight
from aiter.utility import fp4_utils
from aiter.utility.base_tuner import GemmCommonTuner
from aiter.utility.mp_tuner import mp_tuner

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
        f"tuning supports fp4/fp8 operands only (no fp6 quantizer); got {dtype!r}"
    )


def _dequant(codes, scale, dtype, rows):
    scale_f32 = fp4_utils.e8m0_to_f32(scale[:rows].repeat_interleave(32, dim=1))
    if dtype == "fp8":
        return codes.float() * scale_f32
    return fp4_utils.mxfp4_to_f32(codes) * scale_f32


def generate_data(
    m,
    n,
    k,
    seed,
    a_dtype,
    b_dtype,
    dtype=dtypes.bf16,
    device="cuda",
):
    torch.manual_seed(seed)
    m_aligned = (m + 31) // 32 * 32
    n_aligned = (n + 31) // 32 * 32
    a_f = torch.zeros(m_aligned, k, dtype=torch.float32, device=device)
    b_f = torch.zeros(n_aligned, k, dtype=torch.float32, device=device)
    a_f[:m] = torch.randn(m, k, device=device)
    b_f[:n] = torch.randn(n, k, device=device)
    a_q, scale_a_unshuffled = _quant(a_f, a_dtype)
    b_q, scale_b_unshuffled = _quant(b_f, b_dtype)
    a_codes = a_q[:m]
    b_codes = b_q[:n]
    b_shuffled = shuffle_weight(b_codes, layout=(16, 16))
    scale_a = shuffle_scale_a16w4(scale_a_unshuffled, 1, False)
    scale_b = shuffle_scale_a16w4(scale_b_unshuffled, 1, False)
    a_dequant = _dequant(a_codes, scale_a_unshuffled, a_dtype, m)
    b_dequant = _dequant(b_codes, scale_b_unshuffled, b_dtype, n)
    out = torch.empty(m, n, dtype=dtype, device=device)
    return {
        "A": a_codes,
        "B": b_shuffled,
        "a_scale": scale_a,
        "b_scale": scale_b,
        "out": out,
        "a_deq": a_dequant,
        "b_deq": b_dequant,
    }


def run_gemm_flydsl(A, B, a_scale, b_scale, out, kernel_id, a_dtype, b_dtype):
    instance = kernels_list[kernel_id]
    flydsl_mxscale_preshuffle_gemm(
        A,
        B,
        a_scale,
        b_scale,
        out,
        a_dtype=a_dtype,
        b_dtype=b_dtype,
        tile_m=instance.tile_m,
        tile_n=instance.tile_n,
        tile_k=instance.tile_k,
        waves_per_eu=instance.waves_per_eu,
        xcd_swizzle=instance.xcd_swizzle,
        split_k=instance.split_k,
    )
    return out


def run_torch(a_deq, b_deq, dtype=dtypes.bf16):
    return (a_deq @ b_deq.T).to(dtype)


class MxscalePreShuffleTuner(GemmCommonTuner):
    ARG_DEFAULTS: ClassVar[dict] = {
        **GemmCommonTuner.ARG_DEFAULTS,
        "tune_file": AITER_CONFIG_GEMM_MXSCALE_PRESHUFFLE,
        "untune_file": "aiter/configs/mxscale_preshuffle_untuned_gemm.csv",
        "config_env_name": "AITER_CONFIG_GEMM_MXSCALE_PRESHUFFLE",
    }

    def _clear_op_caches(self):
        from aiter.ops.flydsl import mxscale_preshuffle_kernels as op

        op._TUNED_CACHE.clear()

    def _setup_specific_arguments(self):
        pass

    def calculate(self, results, bpes=(1, 1, 2)):
        return super().calculate(results, bpes=bpes)

    def getKernelName(self, kernelId, libtype="flydsl"):
        instance = kernels_list.get(kernelId)
        return instance.name if instance is not None else None

    def get_flydsl_mxscale_tune_task(self, info_keys, seed, args):
        _gfx, _cu_num, M, N, K, a_dtype, b_dtype = info_keys
        if (
            not is_flydsl_available()
            or "flydsl_mxscale_preshuffle_gemm" not in globals()
        ):
            return []
        gemm_keys = ["A", "B", "a_scale", "b_scale", "out"]
        ref_keys = ["a_deq", "b_deq"]
        tasks = []
        for kernel_id, instance in candidates_for(a_dtype, b_dtype, M, N, K):
            info = (info_keys, kernel_id, instance.split_k, instance.name, "flydsl")
            tasks.append(
                (
                    info,
                    generate_data,
                    (M, N, K, seed, a_dtype, b_dtype),
                    run_gemm_flydsl,
                    (gemm_keys, kernel_id, a_dtype, b_dtype),
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
        del tunedf
        tasks = []
        tasks_data = []
        seed = 0
        for index in range(len(untunedf)):
            M = int(untunedf.loc[index, "M"])
            N = int(untunedf.loc[index, "N"])
            K = int(untunedf.loc[index, "K"])
            a_dtype = untunedf.loc[index, "a_dtype"]
            b_dtype = untunedf.loc[index, "b_dtype"]
            seed += 1
            info_keys = (self.get_gfx(), self.get_cu_num(), M, N, K, a_dtype, b_dtype)
            shape_tasks = self.get_flydsl_mxscale_tune_task(info_keys, seed, args)
            if not shape_tasks:
                print(
                    f"[mxscale] skip M={M} N={N} K={K} "
                    f"{a_dtype}/{b_dtype}: no legal tile"
                )
                continue
            tasks.extend(shape_tasks)
            tasks_data.append((len(shape_tasks), ()))
        if not tasks:
            return []
        return mp_tuner(
            tasks,
            tasks_data,
            args.mp,
            False,
            args.shape_grouped,
            args.errRatio,
            timeout=args.timeout,
            verbose=args.verbose,
        )

    def result_to_df(self, results):
        resultdf = pd.DataFrame(columns=self.columns)
        for result in results:
            info, time_us, err_ratio = result
            keys, kernel_id, split_k, kernel_name, libtype = info
            if time_us == self.INVALID_TIME:
                kernel_name = "None"
            elif kernel_name == "":
                kernel_name = self.getKernelName(kernel_id, libtype)
            tflops, bandwidth = self.calculate(result)
            key_dict = dict(zip(self.keys, keys))
            if len(results) == self.topk:
                print(
                    f"Tuning result for {str(key_dict).strip('{}')} is "
                    f"kernelId={kernel_id} {kernel_name} splitK={split_k}, "
                    f"{time_us}us, err_ratio={err_ratio}, "
                    f"tflops={tflops} TFLOPS, bw={bandwidth} GB/s"
                )
            key_dict.update(
                {
                    "libtype": [libtype],
                    "kernelId": [kernel_id],
                    "splitK": [split_k],
                    "us": [time_us],
                    "kernelName": [kernel_name],
                    "errRatio": [err_ratio],
                    "tflops": [tflops],
                    "bw": [bandwidth],
                }
            )
            frame = pd.DataFrame(key_dict)
            resultdf = (
                frame
                if resultdf.empty
                else pd.concat([resultdf, frame], ignore_index=True)
            )
        return resultdf

    def run_config(self, args):
        from aiter.test_common import checkAllclose, run_perftest

        results = []
        for _, row in self.untunedf.iterrows():
            M, N, K = int(row["M"]), int(row["N"]), int(row["K"])
            a_dtype, b_dtype = row["a_dtype"], row["b_dtype"]
            shape = f"({M}, {N}, {K}, {a_dtype}, {b_dtype})"
            allowed, allowed_description = self._get_run_config_err_ratio_limit(
                row, args
            )
            try:
                data = generate_data(M, N, K, 0, a_dtype, b_dtype)

                def dispatch(
                    A,
                    B,
                    a_scale,
                    b_scale,
                    out,
                    a_dtype=a_dtype,
                    b_dtype=b_dtype,
                ):
                    return gemm_mxscale_preshuffle(
                        A,
                        B,
                        a_scale,
                        b_scale,
                        out,
                        a_dtype=a_dtype,
                        b_dtype=b_dtype,
                    )

                out, time_us = run_perftest(
                    dispatch,
                    data["A"],
                    data["B"],
                    data["a_scale"],
                    data["b_scale"],
                    data["out"],
                    num_warmup=args.warmup,
                    num_iters=args.iters,
                )
                ref = run_torch(data["a_deq"], data["b_deq"], dtype=dtypes.bf16)
                err_ratio = checkAllclose(
                    out.to(dtypes.bf16), ref, msg=f"run_config {shape}"
                )
                status = (
                    "ok"
                    if err_ratio <= allowed
                    else f"mismatch:err_ratio={err_ratio:.6g}(>{allowed_description})"
                )
                results.append(
                    {"shape": shape, "e2e_us": time_us, "status": status}
                )
            except Exception as error:  # noqa: BLE001 - record per-shape failures
                results.append(
                    {"shape": shape, "e2e_us": -1, "status": f"error:{error}"}
                )
            finally:
                torch.cuda.empty_cache()
        return results


if __name__ == "__main__":
    tuner = MxscalePreShuffleTuner(
        "MxscalePreShuffleTuner",
        key=["gfx", "cu_num", "M", "N", "K", "a_dtype", "b_dtype"],
        resultList=[
            "libtype",
            "kernelId",
            "splitK",
            "us",
            "kernelName",
            "tflops",
            "bw",
            "errRatio",
        ],
        description="tune FlyDSL mxscale preshuffle GEMM (a4w4/a6w4/a8w8)",
    )
    tuner.run(tuner.parse_args(), False)
