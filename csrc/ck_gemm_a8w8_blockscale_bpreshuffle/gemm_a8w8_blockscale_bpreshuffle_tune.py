# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Tune the FlyDSL blockscale split-K bpreshuffle pipelines.

Adapted from ``gemm_a8w8_bpreshuffle_tune.py`` (the ptpc bpreshuffle tuner),
narrowed to the single-pipeline blockscale mode of the a8w8 split-K family:
``SPLITK_BLOCKSCALE_PIPELINE`` (fp32 in-loop dequant). Unlike ptpc
bpreshuffle, it has one quant dtype (fp8) and one scale layout, so there is
no ``--libtype``/``q_dtype_w`` sweep here -- every candidate is
``libtype=flydsl``.

This is bounded plumbing, not a tuning run: it writes winner rows to
``aiter/configs/a8w8_blockscale_bpreshuffle_tuned_gemm.csv`` (via
``AITER_CONFIG_GEMM_A8W8_BLOCKSCALE_BPRESHUFFLE_FILE``, shared by both scale
modes -- the runtime dispatch parses the scale mode back out of the
kernelName) for whatever shapes are in its untuned CSV / built-in smoke list.
No tuned rows are shipped from this file; a real sweep is a separate
follow-up.
"""

from typing import Any, ClassVar

import torch
from einops import rearrange

from aiter import dtypes
from aiter.jit.core import AITER_CONFIG_GEMM_A8W8_BLOCKSCALE_BPRESHUFFLE
from aiter.ops.flydsl.gemm_tune.flydsl_splitk_bpreshuffle_tuner_common import (
    FlydslSplitKBpreshuffleTuner,
)
from aiter.ops.shuffle import shuffle_weight

# Reuses the same common module as the ptpc bpreshuffle tuner: the pipeline
# and its kernel table are committed there (d0c7422c) precisely so a separate
# blockscale tuner script can import and drive them without duplicating the
# candidate-generation logic. Same import guard as the ptpc tuner: this module
# must stay importable (to name candidates) even where flydsl cannot compile.
try:
    from aiter.ops.flydsl.gemm_tune.flydsl_gemm_a8w8_bpreshuffle_common import (
        SPLITK_BLOCKSCALE_PIPELINE,
        kernels_list_splitk_blockscale,
    )
except ImportError:
    print(
        "[FlyDSL] flydsl_gemm_a8w8_bpreshuffle_common.py not found, "
        "flydsl blockscale split-K tuning disabled"
    )
    SPLITK_BLOCKSCALE_PIPELINE = None
    kernels_list_splitk_blockscale = {}

# Sweep exactly one pipeline each -- kept as tuples (not a bare Pipeline) so
# the tune-task loops below stay shaped like the multi-pipeline ptpc template,
# in case a second pipeline is added to either scale mode later.
FLYDSL_BLOCKSCALE_PIPELINES = (
    (SPLITK_BLOCKSCALE_PIPELINE,) if SPLITK_BLOCKSCALE_PIPELINE is not None else ()
)

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


def run_gemm_flydsl_splitk_blockscale(
    x, weight_shuffle, x_scale, w_scale, out, kernel_id
):
    from aiter.ops.flydsl.kernels.preshuffle_gemm_splitk_op import (
        flydsl_preshuffle_gemm_splitk_a8,
    )

    ki = kernels_list_splitk_blockscale[kernel_id]
    flydsl_preshuffle_gemm_splitk_a8(
        x,
        weight_shuffle,
        x_scale,
        w_scale,
        out,
        ki.tile_m,
        ki.tile_n,
        ki.tile_k,
        ki.split_k,
        use_async_copy=ki.use_async_copy,
        waves_per_eu=ki.waves_per_eu,
        xcd_swizzle=ki.xcd_swizzle,
        lds_stage=ki.lds_stage,
        enable_scheduler=ki.enable_scheduler,
        scale_mode=ki.scale_mode,
        use_m_bounded_store=ki.use_m_bounded_store,
    )
    return out


# Pipeline name -> runner, mirroring
# ``_FLYDSL_PIPELINE_RUNNERS`` in the ptpc bpreshuffle tuner.
_FLYDSL_PIPELINE_RUNNERS = {
    "splitk_blockscale": run_gemm_flydsl_splitk_blockscale,
}


def generate_data_blockscale(m, n, k, seed, dtype=dtypes.bf16, device="cuda"):
    """Random blockscale-quantized inputs, scales in the FlyDSL split-K layout.

    ``x_scale`` is generated ``[M, K//128]`` (the layout
    ``gemm_a8w8_blockscale_tune.py`` uses for its torch-reference dequant) and
    then transposed to ``[K//128, M]`` -- the layout
    ``flydsl_preshuffle_gemm_splitk_a8(..., scale_mode="blockscale")`` requires.
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


class GemmA8W8BlockScaleBpreShuffleTuner(FlydslSplitKBpreshuffleTuner):
    ARG_DEFAULTS: ClassVar[dict[str, Any]] = {
        **FlydslSplitKBpreshuffleTuner.ARG_DEFAULTS,
        "tune_file": f"{AITER_CONFIG_GEMM_A8W8_BLOCKSCALE_BPRESHUFFLE}",
        "untune_file": "aiter/configs/a8w8_blockscale_bpreshuffle_untuned_gemm.csv",
        "config_env_name": "AITER_CONFIG_GEMM_A8W8_BLOCKSCALE_BPRESHUFFLE",
    }

    def _clear_op_caches(self):
        from aiter.ops import gemm_op_a8w8 as _op

        _op.get_CKGEMM_config.cache_clear()
        _op._CKGEMM_CONFIG_CACHE.clear()
        _op._CKGEMM_HAS_GFX.clear()

    def _tune_task_getter_names(self) -> tuple[str, ...]:
        return ("get_flydsl_splitk_blockscale_tune_task",)

    def calculate(self, results, bpes=(1, 1, 2)):
        ## bpes = (inbpe, w_bpe, outbpe)
        return super().calculate(results, bpes=bpes)

    def getKernelName(self, kernelId, libtype="flydsl"):
        if kernelId in kernels_list_splitk_blockscale:
            return kernels_list_splitk_blockscale[kernelId].name
        return None

    def get_flydsl_splitk_blockscale_tune_task(self, info_keys, seed):
        _gfx, _cu_num, M, N, K = info_keys

        if SPLITK_BLOCKSCALE_PIPELINE is None:
            return []

        gemm_flydsl_keys = ["x", "weight_shuffle", "x_scale", "w_scale", "out"]
        ref_keys = ["x", "weight", "x_scale_ref", "w_scale"]
        tasks = []
        for pipe in FLYDSL_BLOCKSCALE_PIPELINES:
            runner_entry = _FLYDSL_PIPELINE_RUNNERS.get(pipe.name)
            if runner_entry is None:
                print(f"[FlyDSL] no runner registered for pipeline {pipe.name!r}")
                continue
            runner = runner_entry
            if not pipe.kernels_list:
                continue
            for i in sorted(pipe.kernels_list.keys()):
                ki = pipe.kernels_list[i]
                if not pipe.fits(ki, M, N, K):
                    continue
                tasks.append(
                    (
                        (info_keys, i, 0, ki.name, "flydsl"),
                        generate_data_blockscale,
                        (M, N, K, seed, dtypes.bf16),
                        runner,
                        (
                            gemm_flydsl_keys,
                            i,
                        ),
                        {
                            "num_warmup": args.warmup,
                            "num_iters": args.iters,
                        },
                        run_torch_blockscale,
                        (
                            ref_keys,
                            dtypes.bf16,
                        ),
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
        description="gen API for gemm a8w8 blockscale bpreshuffle flydsl split-K kernel",
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
