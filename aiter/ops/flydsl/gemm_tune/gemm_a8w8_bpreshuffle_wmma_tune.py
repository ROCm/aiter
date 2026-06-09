# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Auto-tuner for the gfx1250 (WMMA) a8w8 bpreshuffle GEMM.

FlyDSL-only counterpart of the MFMA tuner in
``csrc/ck_gemm_a8w8_bpreshuffle/gemm_a8w8_bpreshuffle_tune.py``: for each untuned
(M, N, K) it benchmarks the WMMA candidates from
``flydsl_gemm_a8w8_bpreshuffle_wmma_common`` and writes the winner's kernelName
(``flydsl_bpreshuffle_wmma_*``, libtype "flydsl") into the a8w8 bpreshuffle tuned
CSV. The public ``aiter.gemm_a8w8_bpreshuffle`` op then routes to it on gfx1250.

Usage::

    python aiter/ops/flydsl/gemm_tune/gemm_a8w8_bpreshuffle_wmma_tune.py \
        --untune_file aiter/configs/a8w8_bpreshuffle_untuned_gemm.csv \
        --tune_file   aiter/configs/a8w8_bpreshuffle_tuned_gemm.csv
"""

import pandas as pd
import torch

import aiter
from aiter import dtypes
from aiter.jit.core import AITER_CONFIG_GEMM_A8W8_BPRESHUFFLE, AITER_ROOT_DIR
from aiter.utility.base_tuner import GemmCommonTuner
from aiter.utility.mp_tuner import mp_tuner
from aiter.ops.shuffle import shuffle_weight
from aiter.ops.flydsl.utils import is_flydsl_available
from aiter.ops.flydsl.gemm_tune.flydsl_gemm_a8w8_bpreshuffle_wmma_common import (
    kernels_list,
    kernel_fits_shape,
)

if is_flydsl_available():
    from aiter.ops.flydsl.bpreshuffle_gemm_gfx1250 import run_preshuffle_gemm_a8_gfx1250

_OUT_TORCH = {"bf16": torch.bfloat16, "f16": torch.float16}
_Q_DTYPE_W = str(dtypes.fp8)


def run_torch(x, weight, x_scale, w_scale, dtype=torch.bfloat16):
    a = x.to(torch.float32) * x_scale.to(torch.float32)
    b = weight.to(torch.float32) * w_scale.to(torch.float32)
    return (a @ b.t()).to(dtype)


def generate_data(m, n, k, seed, out_dtype="bf16", device="cuda"):
    torch.manual_seed(seed)
    x = torch.randn((m, k), dtype=torch.bfloat16, device=device)
    weight = torch.randn((n, k), dtype=torch.bfloat16, device=device)
    x, x_scale = aiter.pertoken_quant(x, quant_dtype=dtypes.fp8)
    weight, w_scale = aiter.pertoken_quant(weight, quant_dtype=dtypes.fp8)
    weight_shuffle = shuffle_weight(weight, layout=(16, 16))
    out = torch.empty(m, n, dtype=_OUT_TORCH[out_dtype], device=device)
    return {
        "x": x,
        "weight": weight,
        "weight_shuffle": weight_shuffle,
        "x_scale": x_scale,
        "w_scale": w_scale,
        "out": out,
    }


def run_gemm(x, weight_shuffle, x_scale, w_scale, out, kernel_id):
    ki = kernels_list[kernel_id]
    run_preshuffle_gemm_a8_gfx1250(
        x,
        weight_shuffle,
        x_scale,
        w_scale,
        out,
        ki.tile_m,
        ki.tile_n,
        ki.tile_k,
        num_buffers=ki.num_buffers,
        split_k=ki.split_k,
        cluster_m=ki.cluster_m,
        cluster_n=ki.cluster_n,
    )
    return out


class GemmA8W8BpreShuffleWmmaTuner(GemmCommonTuner):
    ARG_DEFAULTS = {
        **GemmCommonTuner.ARG_DEFAULTS,
        "tune_file": f"{AITER_CONFIG_GEMM_A8W8_BPRESHUFFLE}",
        "untune_file": f"{AITER_ROOT_DIR}/aiter/configs/a8w8_bpreshuffle_untuned_gemm.csv",
        "config_env_name": "AITER_CONFIG_GEMM_A8W8_BPRESHUFFLE",
    }

    def _clear_op_caches(self):
        from aiter.ops import gemm_op_a8w8 as _op

        _op.get_GEMM_config_with_quant_type.cache_clear()
        _op._GEMM_QUANT_TYPE_CACHE.clear()
        _op._GEMM_QUANT_TYPE_HAS_GFX.clear()

    def _setup_specific_arguments(self):
        self.parser.add_argument(
            "--out_dtype",
            type=str,
            default="bf16",
            choices=["bf16", "f16"],
            help="Output dtype to tune (run once per dtype your model needs)",
        )

    def calculate(self, results, bpes=(1, 1, 2)):
        return super().calculate(results, bpes=bpes)

    def getKernelName(self, kernelId, libtype="flydsl"):
        ki = kernels_list.get(kernelId)
        return ki.name if ki is not None else None

    def result_to_df(self, results):
        resultdf = pd.DataFrame(columns=self.columns)
        for el in results:
            info, time, err_ratio = el
            keys, kernelId, splitK, kernelName, libtype = info
            if time == self.INVALID_TIME:
                kernelName = "None"
            elif kernelName == "":
                resolved = self.getKernelName(kernelId, libtype)
                kernelName = "None" if resolved is None else str(resolved)
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

    def get_wmma_tune_task(self, info_keys, out_dtype, seed, args):
        gfx, cu_num, M, N, K, q_dtype_w = info_keys
        if not is_flydsl_available():
            return []
        run_keys = ["x", "weight_shuffle", "x_scale", "w_scale", "out"]
        ref_keys = ["x", "weight", "x_scale", "w_scale"]
        tasks = []
        for i in sorted(kernels_list.keys()):
            ki = kernels_list[i]
            if not kernel_fits_shape(ki, M, N, K):
                continue
            info = (info_keys, i, 0, ki.name, "flydsl")
            tasks.append(
                (
                    info,
                    generate_data,
                    (M, N, K, seed, out_dtype),
                    run_gemm,
                    (run_keys, i),
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
        mp_num = args.mp
        shape_grouped = args.shape_grouped
        errRatio = args.errRatio
        cu_num = self.get_cu_num()
        gfx = self.get_gfx()
        out_dtype = args.out_dtype
        task = []
        tasks_data = []
        seed = 0
        already = set()
        if tunedf is not None and not tunedf.empty and "q_dtype_w" in tunedf.columns:
            sub = tunedf[tunedf["q_dtype_w"] == _Q_DTYPE_W]
            already = {(int(r.M), int(r.N), int(r.K)) for r in sub.itertuples()}
        for i in range(len(untunedf)):
            M = untunedf.loc[i, "M"]
            N = untunedf.loc[i, "N"]
            K = untunedf.loc[i, "K"]
            if (
                "q_dtype_w" in untunedf.columns
                and str(untunedf.loc[i, "q_dtype_w"]) != _Q_DTYPE_W
            ):
                continue
            if (int(M), int(N), int(K)) in already:
                continue
            seed += 1
            prev = len(task)
            info_keys = (gfx, cu_num, M, N, K, _Q_DTYPE_W)
            task.extend(self.get_wmma_tune_task(info_keys, out_dtype, seed, args))
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
        from aiter.test_common import run_perftest, checkAllclose

        out_dtype = args.out_dtype
        allowed = args.errRatio
        results = []
        for i in range(len(self.untunedf)):
            row = self.untunedf.iloc[i]
            if "q_dtype_w" in row and str(row["q_dtype_w"]) != _Q_DTYPE_W:
                continue
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
                    aiter.gemm_a8w8_bpreshuffle,
                    d["x"],
                    d["weight_shuffle"],
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
    tuner = GemmA8W8BpreShuffleWmmaTuner(
        "GemmA8W8BpreShuffleWmmaTuner",
        key=key,
        resultList=resultList,
        description="Auto-tuner for gfx1250 (WMMA) a8w8 bpreshuffle GEMM",
    )
    args = tuner.parse_args()
    tuner.run(args, False)
