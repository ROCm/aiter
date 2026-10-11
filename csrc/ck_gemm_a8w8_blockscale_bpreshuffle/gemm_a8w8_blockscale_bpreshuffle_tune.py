# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Tune the FlyDSL single-launch blockscale bpreshuffle kernel (kernel 1).

Candidates come from the same tile/option space as the per-token kernel-1 tuner
(``flydsl_gemm_a8w8_bpreshuffle_common``), restricted to blockscale-legal ones
(fp8, tile_k % 128 == 0, gfx950), including split-K values, and are benchmarked
through ``flydsl_preshuffle_gemm_a8(scale_mode="blockscale")``. Candidate names
carry the ``_smbs`` token, which the runtime dispatch and AOT parse back out.
Every candidate is ``libtype=flydsl``.

Winner rows go to ``aiter/configs/a8w8_blockscale_bpreshuffle_tuned_gemm.csv``
(via ``AITER_CONFIG_GEMM_A8W8_BLOCKSCALE_BPRESHUFFLE``). No tuned rows are
shipped from this file; a real sweep is a separate follow-up.
"""

from dataclasses import replace
from typing import Any, ClassVar

import pandas as pd
import torch
from einops import rearrange

from aiter import dtypes
from aiter.jit.core import AITER_CONFIG_GEMM_A8W8_BLOCKSCALE_BPRESHUFFLE
from aiter.ops.shuffle import shuffle_weight
from aiter.utility.base_tuner import GemmCommonTuner
from aiter.utility.mp_tuner import mp_tuner

# Same import guard as the per-token tuner: stay importable where flydsl cannot
# compile.
try:
    from aiter.ops.flydsl.gemm_tune.flydsl_gemm_a8w8_bpreshuffle_common import (
        k_split_candidates,
        kernel_fits_shape,
        kernels_list_blockscale,
    )
except ImportError:
    print(
        "[FlyDSL] flydsl_gemm_a8w8_bpreshuffle_common.py not found, "
        "flydsl blockscale tuning disabled"
    )
    kernels_list_blockscale = {}

BLOCK_SHAPE = (128, 128)  # (block_n, block_k), matches gemm_a8w8_blockscale_tune.py


def run_torch_blockscale(x, weight, x_scale, w_scale, dtype=dtypes.bf16):
    """fp32 blockscale dequant reference.

    ``x_scale`` is fp32 ``[M, K//128]`` and ``w_scale`` is fp32
    ``[N//128, K//128]`` -- the same layout ``gemm_a8w8_blockscale_tune.py``
    uses (verbatim dequant math, adapted to drop the untimed bias/asm-padding
    concerns that tuner also carries). Kept in fp32 throughout: never cast the
    dequantized operands to bf16 before the matmul.
    """
    block_n, block_k = BLOCK_SHAPE
    m, k = x.shape
    n = weight.shape[0]
    scale_n = (n + block_n - 1) // block_n
    scale_k = (k + block_k - 1) // block_k

    x = x.to(x_scale.dtype).view(m, k // block_k, block_k) * x_scale.unsqueeze(-1)
    x = x.view(m, k)

    w_scale_full = rearrange(
        w_scale.view(-1, 1)
        .repeat(1, block_n * block_k)
        .view(scale_n, scale_k, block_n, block_k),
        "num_blk_n num_blk_k blk_n blk_k -> (num_blk_n blk_n) (num_blk_k blk_k)",
    )
    w_scale_full = w_scale_full[:n, :k]
    weight = weight.to(w_scale_full.dtype) * w_scale_full

    out = torch.nn.functional.linear(x.to(dtypes.fp32), weight.to(dtypes.fp32))
    return out.to(dtype)


def run_gemm_flydsl_blockscale(
    x, weight_shuffle, x_scale, w_scale, out, kernel_id, k_split=1
):
    from aiter.ops.flydsl.gemm_kernels import flydsl_preshuffle_gemm_a8

    ki = kernels_list_blockscale[kernel_id]
    flydsl_preshuffle_gemm_a8(
        x,
        weight_shuffle,
        x_scale,
        w_scale,
        out,
        ki.tile_m,
        ki.tile_n,
        ki.tile_k,
        ki.use_async_copy,
        ki.waves_per_eu,
        ki.xcd_swizzle,
        lds_stage=ki.lds_stage,
        enable_scheduler=ki.enable_scheduler,
        split_k=k_split,
        scale_mode="blockscale",
    )
    return out


def generate_data_blockscale(m, n, k, seed, dtype=dtypes.bf16, device="cuda"):
    """Random blockscale-quantized inputs, scales in the kernel-1 blockscale layout.

    ``x_scale`` is generated ``[M, K//128]`` (the layout
    ``gemm_a8w8_blockscale_tune.py`` uses for its torch-reference dequant) and
    then transposed to ``[K//128, M]`` -- the layout
    ``flydsl_preshuffle_gemm_a8(..., scale_mode="blockscale")`` requires.
    ``w_scale`` needs no transform: ``[N//128, K//128]`` is already what both
    sides expect.
    """
    torch.manual_seed(seed)
    block_n, block_k = BLOCK_SHAPE
    scale_n = (n + block_n - 1) // block_n
    scale_k = (k + block_k - 1) // block_k

    x = (torch.rand((m, k), dtype=dtypes.fp16, device=device) / 10).to(dtypes.fp8)
    weight = (torch.rand((n, k), dtype=dtypes.fp16, device=device) / 10).to(dtypes.fp8)
    x_scale = torch.rand([m, scale_k], dtype=dtypes.fp32, device=device)
    w_scale = torch.rand([scale_n, scale_k], dtype=dtypes.fp32, device=device)
    x_scale_t = x_scale.transpose(0, 1).contiguous()

    weight_shuffle = shuffle_weight(weight, layout=(16, 16))
    out = torch.empty(m, n, dtype=dtype, device=device)
    return {
        "x": x,
        "weight_shuffle": weight_shuffle,
        "x_scale": x_scale_t,
        "w_scale": w_scale,
        "out": out,
        "weight": weight,
        "x_scale_ref": x_scale,
    }


# Small built-in decode-shape smoke list -- for a quick
# ``python3 gemm_a8w8_blockscale_bpreshuffle_tune.py`` sanity sweep only. Both
# shapes satisfy the blockscale K%128==0 / N%128==0 constraint. Do NOT grow
# this into a real tuning shape list here; real tuning is a separate effort.
SMOKE_SHAPES = [
    (1, 2048, 7168),
    (4, 4096, 7168),
]


class GemmA8W8BlockScaleBpreShuffleTuner(GemmCommonTuner):
    ARG_DEFAULTS: ClassVar[dict[str, Any]] = {
        **GemmCommonTuner.ARG_DEFAULTS,
        "tune_file": f"{AITER_CONFIG_GEMM_A8W8_BLOCKSCALE_BPRESHUFFLE}",
        "untune_file": "aiter/configs/a8w8_blockscale_bpreshuffle_untuned_gemm.csv",
        "config_env_name": "AITER_CONFIG_GEMM_A8W8_BLOCKSCALE_BPRESHUFFLE",
    }

    def _setup_specific_arguments(self):
        """No extra flags: single pipeline/libtype."""

    def _clear_op_caches(self):
        from aiter.ops import gemm_op_a8w8 as _op

        _op.get_CKGEMM_config.cache_clear()
        _op._CKGEMM_CONFIG_CACHE.clear()
        _op._CKGEMM_HAS_GFX.clear()

    def calculate(self, results, bpes=(1, 1, 2)):
        ## bpes = (inbpe, w_bpe, outbpe)
        return super().calculate(results, bpes=bpes)

    def getKernelName(self, kernelId, libtype="flydsl"):
        if kernelId in kernels_list_blockscale:
            return kernels_list_blockscale[kernelId].name
        return None

    def get_flydsl_blockscale_tune_task(self, info_keys, seed):
        _gfx, cu_num, M, N, K = info_keys

        gemm_flydsl_keys = ["x", "weight_shuffle", "x_scale", "w_scale", "out"]
        ref_keys = ["x", "weight", "x_scale_ref", "w_scale"]
        tasks = []
        for i in sorted(kernels_list_blockscale):
            ki = kernels_list_blockscale[i]
            if not kernel_fits_shape(ki, M, N, K):
                continue
            for ks in [1] + k_split_candidates(ki, M, N, K, cu_num=cu_num):
                name = replace(ki, k_split=ks).name
                tasks.append(
                    (
                        (info_keys, i, 0 if ks == 1 else ks, name, "flydsl"),
                        generate_data_blockscale,
                        (M, N, K, seed, dtypes.bf16),
                        run_gemm_flydsl_blockscale,
                        (gemm_flydsl_keys, i, ks),
                        {
                            "num_warmup": args.warmup,
                            "num_iters": args.iters,
                        },
                        run_torch_blockscale,
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
        cu_num = self.get_cu_num()
        gfx = self.get_gfx()
        task = []
        tasks_data = []  # [(kernel_nums, datas)]
        for i in range(len(untunedf)):
            M = untunedf.loc[i, "M"]
            N = untunedf.loc[i, "N"]
            K = untunedf.loc[i, "K"]
            prev_task_count = len(task)
            task.extend(
                self.get_flydsl_blockscale_tune_task((gfx, cu_num, M, N, K), i + 1)
            )
            tasks_data.append((len(task) - prev_task_count, ()))
        ret = []
        if task:
            ret = mp_tuner(
                task,
                tasks_data,
                args.mp,
                False,
                args.shape_grouped,
                args.errRatio,
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
                else (self.getKernelName(kernelId) if kernelName == "" else kernelName)
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


if __name__ == "__main__":
    ## use default key and resultList; column order must match the header of
    ## a8w8_blockscale_bpreshuffle_tuned_gemm.csv: no q_dtype_w column (fp8-only).
    key = ["gfx", "cu_num", "M", "N", "K"]
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
    tuner = GemmA8W8BlockScaleBpreShuffleTuner(
        "GemmA8W8BlockScaleBpreShuffleTuner",
        key=key,
        resultList=resultList,
        description="gen API for gemm a8w8 blockscale bpreshuffle flydsl kernel",
    )

    args = tuner.parse_args()

    # If the untuned CSV is empty (no run-specific shapes requested), fall back
    # to the small built-in smoke list rather than tuning nothing.
    if tuner.get_untuned_gemm_list(args.untune_file).empty:
        import os
        import tempfile

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", prefix="aiter_smoke_", delete=False
        ) as tmp:
            tmp.write("M,N,K\n")
            for m, n, k in SMOKE_SHAPES:
                tmp.write(f"{m},{n},{k}\n")
            args.untune_file = tmp.name
        try:
            tuner.run(args, False)
        finally:
            os.remove(args.untune_file)
    else:
        tuner.run(args, False)
