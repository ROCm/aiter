# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Tune the FlyDSL a4w4 (mxfp4) split-K bpreshuffle pipeline.

Sibling of ``gemm_a8w8_blockscale_bpreshuffle_tune.py``, ported for a4w4:
both operands packed 4-bit (fp4/E2M1), scales are 32-block E8M0 bytes rather
than the 128-block fp32 blockscale a8w8 uses (``scale_mode="mxfp4"``). Sweeps
exactly one pipeline (``splitk_mxfp4`` from the a4w4 candidate module), so
there is no ``--libtype``/dtype sweep here -- every candidate is
``libtype=flydsl``.

This is bounded plumbing, not a tuning run: it writes winner rows to
``aiter/configs/a4w4_blockscale_tuned_gemm.csv`` (via
``AITER_CONFIG_GEMM_A4W4``) for whatever shapes are in its untuned CSV /
built-in smoke list. No tuned rows are shipped from this file; a real sweep
is a separate follow-up.
"""

from typing import Any, ClassVar

import torch

from aiter import dtypes
from aiter.jit.core import AITER_CONFIG_GEMM_A4W4
from aiter.ops.flydsl.gemm_tune.flydsl_splitk_bpreshuffle_tuner_common import (
    FlydslSplitKBpreshuffleTuner,
)
from aiter.ops.shuffle import shuffle_scale_w4_cdna4, shuffle_weight_w4_cdna4

# Reuses the a4w4 candidate module committed alongside the dispatch branch
# (gemm_op_a4w4.py's ``libtype=="flydsl"`` path): the kernel table and shape
# predicate live there so this tuner and the dispatcher never duplicate the
# candidate-generation logic. Same import guard as the a8w8 blockscale
# bpreshuffle tuner: this module must stay importable (to name candidates) on
# hosts where flydsl cannot compile.
try:
    from aiter.ops.flydsl.gemm_tune.flydsl_gemm_a4w4_bpreshuffle_common import (
        PIPELINES as A4W4_PIPELINES,
    )
    from aiter.ops.flydsl.gemm_tune.flydsl_gemm_a4w4_bpreshuffle_common import (
        kernels_list_splitk_mxfp4,
    )
except ImportError:
    print(
        "[FlyDSL] flydsl_gemm_a4w4_bpreshuffle_common.py not found, "
        "flydsl a4w4 split-K tuning disabled"
    )
    A4W4_PIPELINES = ()
    kernels_list_splitk_mxfp4 = {}

from aiter.ops.flydsl.utils import is_flydsl_available

BLOCK = 32  # mxfp4 E8M0 scale block size (K elements per scale byte).
FP4_MAX = 6.0
# The 8 magnitudes of the E2M1 codebook (sign is the high nibble bit).
_FP4_MAG = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32)


def fp4_quant(x):
    """Nearest-E2M1 codes (uint8, 0..15) for an fp32 tensor.

    Ported verbatim from ``op_tests/bench_mxfp4_splitk_check.py``, the
    correctness probe for this same kernel family.
    """
    mag = x.abs().unsqueeze(-1)
    code = (mag - _FP4_MAG.to(x.device)).abs().argmin(dim=-1).to(torch.uint8)
    return code | (x < 0).to(torch.uint8) * 8


def fp4_deq(code):
    return _FP4_MAG.to(code.device)[(code & 7).long()] * torch.where(
        code & 8 > 0, -1.0, 1.0
    )


def pack_fp4(code):
    """[..., K] nibble codes -> [..., K/2] bytes, low nibble first."""
    lo = code[..., 0::2]
    hi = code[..., 1::2]
    return (lo | (hi << 4)).contiguous()


def e8m0_quant(x_blocks):
    """(fp4 codes, E8M0 byte, fp32 scale) for a [..., BLOCK] view."""
    amax = x_blocks.abs().amax(dim=-1).clamp_min(1e-30)
    byte = (torch.ceil(torch.log2(amax / FP4_MAX)) + 127).clamp(0, 254).to(torch.uint8)
    scale = torch.exp2(byte.float() - 127.0).unsqueeze(-1)
    code = fp4_quant((x_blocks / scale).clamp(-FP4_MAX, FP4_MAX))
    return code, byte, scale


def run_torch_mxfp4(a_deq, b_deq, dtype=dtypes.bf16):
    """Reference matmul on already-dequantized fp32 operands."""
    return torch.nn.functional.linear(a_deq, b_deq).to(dtype)


def run_gemm_flydsl_splitk_mxfp4(aq, wq, xsb, wsb, out, kernel_id):
    from aiter.ops.flydsl.kernels.preshuffle_gemm_splitk_op import (
        flydsl_preshuffle_gemm_splitk_a8,
    )

    ki = kernels_list_splitk_mxfp4[kernel_id]
    flydsl_preshuffle_gemm_splitk_a8(
        aq,
        wq,
        xsb,
        wsb,
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
        scale_mode="mxfp4",
        use_m_bounded_store=ki.use_m_bounded_store,
        in_dtype="fp4",
    )
    return out


# Pipeline name -> (runner, is-it-usable-right-now), mirroring
# ``_FLYDSL_PIPELINE_RUNNERS`` in the a8w8 blockscale bpreshuffle tuner.
_FLYDSL_PIPELINE_RUNNERS = {
    "splitk_mxfp4": (run_gemm_flydsl_splitk_mxfp4, is_flydsl_available),
}


def generate_data_mxfp4(m, n, k, seed, dtype=dtypes.bf16, device="cuda"):
    """Random mxfp4-quantized inputs, in the dispatch's CDNA4 preshuffle layout.

    Mirrors ``gemm_op_a4w4.py``'s ``libtype=="flydsl"`` branch and
    ``op_tests/bench_mxfp4_splitk_check.py``: A/B quantized to E2M1 codes with
    per-32-block E8M0 scales, B and both scale tensors run through the
    ``shuffle_{weight,scale}_w4_cdna4`` CDNA4 preshuffle, and A-scale rows
    padded to a multiple of 32 before shuffling (same ``m_pad`` padding the
    dispatch path applies to ``A_scale``).
    """
    torch.manual_seed(seed)
    sk = k // BLOCK
    a = (torch.rand((m, k), dtype=dtypes.fp32, device=device) - 0.5) * 4.0
    b = (torch.rand((n, k), dtype=dtypes.fp32, device=device) - 0.5) * 4.0

    a_code, a_e8, a_sc = e8m0_quant(a.view(m, sk, BLOCK))
    b_code, b_e8, b_sc = e8m0_quant(b.view(n, sk, BLOCK))

    a_deq = (fp4_deq(a_code) * a_sc).view(m, k)
    b_deq = (fp4_deq(b_code) * b_sc).view(n, k)

    aq = pack_fp4(a_code.view(m, k)).view(torch.int8)
    wq = shuffle_weight_w4_cdna4(pack_fp4(b_code.view(n, k))).view(torch.int8)

    m_pad = ((m + 31) // 32) * 32
    a_e8_pad = torch.zeros((m_pad, sk), dtype=torch.uint8, device=device)
    a_e8_pad[:m] = a_e8
    xsb = shuffle_scale_w4_cdna4(a_e8_pad).view(torch.int8)
    wsb = shuffle_scale_w4_cdna4(b_e8).view(torch.int8)

    out = torch.empty(m, n, dtype=dtype, device=device)
    return {
        "aq": aq,
        "wq": wq,
        "xsb": xsb,
        "wsb": wsb,
        "out": out,
        "a_deq": a_deq,
        "b_deq": b_deq,
    }


# Small built-in decode-shape smoke list -- for a quick
# ``python3 gemm_a4w4_blockscale_bpreshuffle_tune.py`` sanity sweep only. Both
# shapes satisfy the mxfp4 K%32==0 constraint (BLOCK=32). Do NOT grow this
# into a real tuning shape list here; real tuning is a separate effort.
SMOKE_SHAPES = [
    (1, 512, 4096),
    (4, 4096, 4096),
]


class GemmA4W4BlockScaleBpreShuffleTuner(FlydslSplitKBpreshuffleTuner):
    ARG_DEFAULTS: ClassVar[dict[str, Any]] = {
        **FlydslSplitKBpreshuffleTuner.ARG_DEFAULTS,
        "tune_file": f"{AITER_CONFIG_GEMM_A4W4}",
        "untune_file": "aiter/configs/a4w4_blockscale_untuned_gemm.csv",
        "config_env_name": "AITER_CONFIG_GEMM_A4W4",
    }

    def _clear_op_caches(self):
        from aiter.ops.gemm_op_a4w4 import get_GEMM_config

        get_GEMM_config.cache_clear()
        if hasattr(get_GEMM_config, "gemm_dict"):
            del get_GEMM_config.gemm_dict

    def _tune_task_getter_names(self) -> tuple[str, ...]:
        return ("get_flydsl_splitk_mxfp4_tune_task",)

    def calculate(self, results, bpes=(1 / 2, 1 / 2, 2)):
        ## bpes = (inbpe, w_bpe, outbpe); fp4 packs 2 codes/byte, hence 1/2.
        return super().calculate(results, bpes=bpes)

    def getKernelName(self, kernelId, libtype="flydsl"):
        if kernelId in kernels_list_splitk_mxfp4:
            return kernels_list_splitk_mxfp4[kernelId].name
        return None

    def get_flydsl_splitk_mxfp4_tune_task(self, info_keys, seed):
        _gfx, _cu_num, M, N, K = info_keys

        if not is_flydsl_available() or not A4W4_PIPELINES:
            return []

        gemm_flydsl_keys = ["aq", "wq", "xsb", "wsb", "out"]
        ref_keys = ["a_deq", "b_deq"]
        tasks = []
        for pipe in A4W4_PIPELINES:
            runner_entry = _FLYDSL_PIPELINE_RUNNERS.get(pipe.name)
            if runner_entry is None:
                print(f"[FlyDSL] no runner registered for pipeline {pipe.name!r}")
                continue
            runner, is_available = runner_entry
            if not pipe.kernels_list or not is_available():
                continue
            for i in sorted(pipe.kernels_list.keys()):
                ki = pipe.kernels_list[i]
                if not pipe.fits(ki, M, N, K):
                    continue
                tasks.append(
                    (
                        # splitK column is 0 here, matching the a8w8
                        # blockscale bpreshuffle tuner's convention: the real
                        # split factor is embedded in kernelName, not this
                        # column (gemm_op_a4w4.py parses it back out of the
                        # name string, never reads the splitK column for the
                        # flydsl branch).
                        (info_keys, i, 0, ki.name, "flydsl"),
                        generate_data_mxfp4,
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
                        run_torch_mxfp4,
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
    ## a4w4_blockscale_tuned_gemm.csv (shared with the CK a4w4 tuner).
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
    tuner = GemmA4W4BlockScaleBpreShuffleTuner(
        "GemmA4W4BlockScaleBpreShuffleTuner",
        key=key,
        resultList=resultList,
        description="gen API for gemm a4w4 (mxfp4) split-K bpreshuffle flydsl kernel",
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
