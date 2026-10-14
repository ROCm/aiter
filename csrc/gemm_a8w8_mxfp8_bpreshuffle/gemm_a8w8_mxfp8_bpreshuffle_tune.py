# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Tune gemm_a8w8_mxfp8_bpreshuffle (FP8 GEMM, 1x32 e8m0 scales) on gfx950.

gfx950 runs two FlyDSL pipelines on unshuffled row-major scales (x_scale
[M, K/32], w_scale [N, K/32]): the mxpsh kernel and the 8-wave 256x256 one
(ids from 1_000_000). Every candidate of both that fits the shape is timed and
the fastest lands in the same table the gfx1250 rows live in, keyed by gfx:

    python3 csrc/gemm_a8w8_mxfp8_bpreshuffle/gemm_a8w8_mxfp8_bpreshuffle_tune.py \\
        -i aiter/configs/a8w8_mxfp8_bpreshuffle_untuned_gemm.csv \\
        -o aiter/configs/model_configs/dsv4_a8w8_mxfp8_bpreshuffle_tuned_gemm.csv
"""

from typing import Any, ClassVar

import pandas as pd
import torch
import torch.nn.functional as F

from aiter import dtypes, logger
from aiter.ops.flydsl.gemm_tune.flydsl_gemm_mxscale_preshuffle_common import (
    MXFP8_ROW_PIPELINES,
)
from aiter.ops.gemm_op_common import with_mxscale_w_scale_block
from aiter.ops.shuffle import shuffle_weight
from aiter.utility.base_tuner import GemmCommonTuner
from aiter.utility.mp_tuner import mp_tuner

TUNED_FILE = "aiter/configs/model_configs/dsv4_a8w8_mxfp8_bpreshuffle_tuned_gemm.csv"
# The block of a shape list without a w_scale_block column, and of a tuned
# table from before the column.
W_SCALE_BLOCK = "1x32"
# w_scale block -> the pipelines tuned for it.
PIPELINES_BY_W_SCALE_BLOCK = {"1x32": MXFP8_ROW_PIPELINES}


def generate_data(m, n, k, seed, w_scale_block=W_SCALE_BLOCK, device="cuda"):
    """fp8 operands with per-1x32 e8m0 scales around 2^-3..2^3, unshuffled."""
    assert w_scale_block == "1x32", f"no {w_scale_block} w_scale data"
    torch.manual_seed(seed)
    x = (torch.randn((m, k), device=device) * 2).to(dtypes.fp8)
    weight = (torch.randn((n, k), device=device) * 2).to(dtypes.fp8)
    x_scale = torch.randint(124, 131, (m, k // 32), dtype=torch.uint8, device=device)
    w_scale = torch.randint(124, 131, (n, k // 32), dtype=torch.uint8, device=device)

    def dequant(q, s):
        return q.to(dtypes.fp32) * torch.exp2(s.float() - 127).repeat_interleave(32, 1)

    return {
        "x": x,
        "weight_shuffle": shuffle_weight(weight, layout=(16, 16)),
        "x_scale": x_scale.view(dtypes.fp8_e8m0),
        "w_scale": w_scale.view(dtypes.fp8_e8m0),
        "out": torch.empty(m, n, dtype=dtypes.bf16, device=device),
        "x_deq": dequant(x, x_scale),
        "w_deq": dequant(weight, w_scale),
    }


def run_torch(x_deq, w_deq, dtype=dtypes.bf16):
    return F.linear(x_deq, w_deq).to(dtype)


def run_gemm_flydsl(x, weight_shuffle, x_scale, w_scale, out, kernel_name):
    from aiter.ops.flydsl.mxscale_preshuffle_kernels import run_gemm_a8w8_mxfp8_gfx950

    return run_gemm_a8w8_mxfp8_gfx950(
        x, weight_shuffle, x_scale, w_scale, out, kernel_name
    )


class GemmA8W8MXFP8BpreshuffleTuner(GemmCommonTuner):
    ARG_DEFAULTS: ClassVar[dict[str, Any]] = {
        **GemmCommonTuner.ARG_DEFAULTS,
        "tune_file": TUNED_FILE,
        "untune_file": "aiter/configs/a8w8_mxfp8_bpreshuffle_untuned_gemm.csv",
        "errRatio": 0.05,
        "batch": 100,
        "profile_file": "",
        "config_env_name": "AITER_CONFIG_GEMM_A8W8_MXFP8_BPRESHUFFLE",
    }

    def getKernelName(self, kernelId):
        # Pipeline id ranges are disjoint (8-wave ids start at 1_000_000).
        for pipe in MXFP8_ROW_PIPELINES:
            if kernelId in pipe.kernels_list:
                return pipe.kernels_list[kernelId].name
        return None

    def _clear_op_caches(self):
        from aiter.ops import gemm_op_a8w8 as _op

        _op._get_mxfp8_bpreshuffle_config.cache_clear()
        _op._load_mxfp8_bpreshuffle_tuned.cache_clear()
        _op._CKGEMM_CONFIG_CACHE.clear()
        _op._CKGEMM_HAS_GFX.clear()

    def _setup_specific_arguments(self):
        pass

    def get_untuned_gemm_list(self, untuned_gemm_file):
        # A plain M,N,K shape list is a 1x32 one (as is a blank cell of a
        # tuned table --all retunes); pre_process slices untunedf down to
        # self.keys.
        df = super().get_untuned_gemm_list(untuned_gemm_file)
        if "w_scale_block" not in df.columns:
            df["w_scale_block"] = W_SCALE_BLOCK
        df["w_scale_block"] = df["w_scale_block"].fillna(W_SCALE_BLOCK)
        unknown = set(df["w_scale_block"]) - set(PIPELINES_BY_W_SCALE_BLOCK)
        if unknown:
            raise ValueError(
                f"{untuned_gemm_file}: no gfx950 mxfp8 kernels tuned for w_scale "
                f"block {sorted(unknown)}; tunable: {sorted(PIPELINES_BY_W_SCALE_BLOCK)}"
            )
        return df

    def get_tuned_gemm_list(self, tuned_gemm_file, columns=None):
        df = super().get_tuned_gemm_list(tuned_gemm_file, columns)
        df = with_mxscale_w_scale_block(df, tuned_gemm_file, default=W_SCALE_BLOCK)
        # A table from before the column gets it where the tuner writes it.
        rest = [c for c in df.columns if c not in self.keys]
        return df[[c for c in self.keys if c in df.columns] + rest]

    def calculate(self, results, bpes=(1, 1, 2)):
        return super().calculate(results, bpes=(1, 1, 2))

    def run_config(self, args):
        from aiter.ops.gemm_op_a8w8 import gemm_a8w8_mxfp8_bpreshuffle
        from aiter.test_common import checkAllclose, run_perftest

        results = []
        for i in range(len(self.untunedf)):
            row = self.untunedf.iloc[i]
            M, N, K = int(row["M"]), int(row["N"]), int(row["K"])
            block = row.get("w_scale_block", W_SCALE_BLOCK)
            shape_str = f"({M}, {N}, {K}, {block})"
            allowed, allowed_desc = self._get_run_config_err_ratio_limit(row, args)
            try:
                d = generate_data(M, N, K, 0, block)
                out, us = run_perftest(
                    gemm_a8w8_mxfp8_bpreshuffle,
                    d["x"],
                    d["weight_shuffle"],
                    d["x_scale"],
                    d["w_scale"],
                    num_warmup=args.warmup,
                    num_iters=args.iters,
                )
                err_ratio = checkAllclose(
                    out,
                    run_torch(d["x_deq"], d["w_deq"]),
                    msg=f"run_config {shape_str}",
                )
                status = (
                    "ok"
                    if err_ratio <= allowed
                    else f"mismatch:err_ratio={err_ratio:.6g}(>{allowed_desc})"
                )
                results.append({"shape": shape_str, "e2e_us": us, "status": status})
            except Exception as e:  # noqa: BLE001
                results.append(
                    {"shape": shape_str, "e2e_us": -1, "status": f"error:{e}"}
                )
            finally:
                torch.cuda.empty_cache()
        return results

    def tune(self, untunedf, tunedf, args):
        gfx = self.get_gfx()
        if gfx != "gfx950":
            logger.warning(
                f"gemm_a8w8_mxfp8_bpreshuffle tuner covers gfx950 only, got {gfx}"
            )
            return []
        cu_num = self.get_cu_num()
        gemm_keys = ["x", "weight_shuffle", "x_scale", "w_scale", "out"]
        ref_args = (["x_deq", "w_deq"], dtypes.bf16)
        run_kwargs = {"num_warmup": args.warmup, "num_iters": args.iters}
        seed = 0

        task = []
        tasks_data = []
        for i in range(len(untunedf)):
            M = int(untunedf.loc[i, "M"])
            N = int(untunedf.loc[i, "N"])
            K = int(untunedf.loc[i, "K"])
            block = str(untunedf.loc[i, "w_scale_block"])
            info_keys = (gfx, cu_num, M, N, K, block)
            # One pass per pipeline of the block, in their order.
            cands = [
                (kernel_id, ki)
                for pipe in PIPELINES_BY_W_SCALE_BLOCK[block]
                for kernel_id, ki in pipe.kernels_list.items()
                if pipe.fits(ki, M, N, K)
            ]
            for kernel_id, ki in cands:
                # kernelName carries the full launch config (incl. split-K and
                # the pipeline), so dispatch never needs kernelId -- it is
                # recorded for reference.
                info = (info_keys, kernel_id, ki.split_k, ki.name, "flydsl")
                task.append(
                    (
                        info,
                        generate_data,
                        (M, N, K, seed, block),
                        run_gemm_flydsl,
                        (gemm_keys, ki.name),
                        dict(run_kwargs),
                        run_torch,
                        ref_args,
                        {},
                        None,
                        1e-2,
                        0.01,
                        None,
                        None,
                        ("out",),
                    )
                )
            if not cands:
                logger.warning(f"no mxfp8 {block} kernel fits M={M} N={N} K={K}")
            tasks_data.append((len(cands), ()))

        if not task:
            return []
        return mp_tuner(
            task,
            tasks_data,
            args.mp,
            False,
            args.shape_grouped,
            args.errRatio,
            timeout=args.timeout,
            verbose=args.verbose,
        )

    def result_to_df(self, results):
        rows = []
        for el in results:
            info, time, err_ratio = el
            keys, kernelId, splitK, kernelName, libtype = info
            if time in (self.INVALID_TIME, self.INF_TIME):
                kernelName = "None"
            tflops, bw = self.calculate(el)
            key_dict = dict(zip(self.keys, keys))
            if len(results) == self.topk:
                print(
                    f"Tuning result for {str(key_dict).strip('{}')} is "
                    f"{kernelName} splitK={splitK}, {time}us, err_ratio={err_ratio}, "
                    f"tflops={tflops} TFLOPS, bw={bw} GB/s"
                )
            key_dict.update(
                {
                    "libtype": [libtype],
                    "kernelId": [kernelId],
                    "splitK": [splitK],
                    "us": [time],
                    "kernelName": [kernelName],
                    "tflops": [tflops],
                    "bw": [bw],
                    "errRatio": [err_ratio],
                }
            )
            rows.append(key_dict)
        if not rows:
            return pd.DataFrame(columns=self.columns)
        return pd.concat([pd.DataFrame(r) for r in rows], ignore_index=True)


if __name__ == "__main__":
    # Same columns as the table the gfx1250 rows already live in.
    key = ["gfx", "cu_num", "M", "N", "K", "w_scale_block"]
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
    tuner = GemmA8W8MXFP8BpreshuffleTuner(
        "a8w8_mxfp8_bpreshuffle_tuned_gemm",
        key=key,
        resultList=resultList,
        description="Tune gemm_a8w8_mxfp8_bpreshuffle (gfx950 FlyDSL mxpsh, 1x32 e8m0)",
    )
    args = tuner.parse_args()
    tuner.run(args, False)
