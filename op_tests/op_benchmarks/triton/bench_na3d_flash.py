# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Performance benchmark for 3D neighborhood flash attention (na3d_flash).

Usage examples
--------------
# Sweep default LTX-2.5 shapes, BF16
python bench_na3d_flash.py

# Custom shapes
python bench_na3d_flash.py -s '(1,79,192,192,4,64,11,11,11)'
"""

import argparse

import pandas as pd
import torch

import aiter
from aiter import dtypes
from aiter.jit.utils.chip_info import get_gfx_runtime as get_gfx
from aiter.ops.triton.attention.na3d_flash import na3d_flash_attn
from aiter.test_common import benchmark, checkAllclose, run_perftest
from op_tests.triton_tests.attention.test_na3d_flash import (
    _DEFAULT_SHAPES,
    _na3d_sdpa_exact,
)

SUPPORTED_GFX = ["gfx942", "gfx950"]
_SUPPORTED_DTYPES = (torch.bfloat16,)


@benchmark()
def bench_na3d_flash(B, T, H, W, NH, HD, KT, KH, KW, dtype):
    SEQ = T * H * W
    C = NH * HD
    K = KT * KH * KW  # neighborhood size

    # Build inputs matching the real decoder call: pre-scaled Q, BF16, channels-last.
    scale = HD**-0.5
    q = torch.randn(B, T, H, W, NH, HD, dtype=dtype, device="cuda") * scale
    k = torch.randn(B, T, H, W, NH, HD, dtype=dtype, device="cuda")
    v = torch.randn(B, T, H, W, NH, HD, dtype=dtype, device="cuda")

    # Reference: grouped-SDPA exact path (same as the eager processor fallback).
    # Also used as a timed candidate so the table shows the real kernel speedup.
    ref = _na3d_sdpa_exact(q, k, v, kernel_size=(KT, KH, KW))

    # Logical FLOPs: QK and AV dot products over K neighbors per query.
    #   QK: 2 * B * SEQ * K * C
    #   AV: 2 * B * SEQ * K * C
    flops = 4 * B * SEQ * K * C
    # Logical bytes: minimum data the operation must touch if every element is
    # read/written exactly once: Q, K, V each read once per query, output written once.
    # This is an effective/logical rate, not actual HBM traffic (which varies by
    # implementation and tile size). The same formula is used for both candidates so
    # the comparison is fair; sdpa_exact's lower effective TB/s reflects its higher
    # real bandwidth cost.
    #   reads : Q (SEQ*C) + K (SEQ*K*C) + V (SEQ*K*C)
    #   writes: Out (SEQ*C)
    elem = q.element_size()
    nbytes = (2 * B * SEQ * C + 2 * B * SEQ * K * C) * elem

    candidates = {
        "triton": lambda: na3d_flash_attn(q, k, v, kernel_size=(KT, KH, KW)),
        "sdpa_exact": lambda: _na3d_sdpa_exact(q, k, v, kernel_size=(KT, KH, KW)),
    }

    ret = {"gfx": get_gfx()}
    for name, fn in candidates.items():
        out, us = run_perftest(fn)
        err = checkAllclose(
            ref.float(),
            out.float(),
            rtol=1e-2,
            atol=5e-2,
            msg=f"{name}: na3d_flash (T={T},H={H},W={W},k=({KT},{KH},{KW}))",
        )
        ret[f"{name} us"] = us
        ret[f"{name} TFLOPS"] = flops / us / 1e6
        ret[f"{name} eff TB/s"] = nbytes / us / 1e6
        ret[f"{name} err"] = err
    return ret


def main():
    try:
        gfx = get_gfx()
    except (RuntimeError, KeyError) as e:
        aiter.logger.warning("na3d_flash: could not detect GPU arch, skipping (%s)", e)
        return
    if gfx not in SUPPORTED_GFX:
        aiter.logger.warning("na3d_flash unsupported on %s; skipping", gfx)
        return

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="Benchmark 3D neighborhood flash attention",
    )
    parser.add_argument(
        "-d",
        "--dtype",
        type=dtypes.str2Dtype,
        nargs="*",
        default=[dtypes.d_dtypes["bf16"]],
        help="Input dtype (default: bf16)",
    )
    parser.add_argument(
        "-s",
        "--shapes",
        type=dtypes.str2tuple,
        nargs="*",
        default=_DEFAULT_SHAPES,
        help=(
            "Shapes as (B,T,H,W,NH,HD,KT,KH,KW) tuples.\n"
            "e.g.: -s '(1,79,192,192,4,64,11,11,11)'"
        ),
    )
    args = parser.parse_args()

    for dtype in args.dtype:
        if dtype not in _SUPPORTED_DTYPES:
            aiter.logger.warning(
                "na3d_flash only supports %s; skipping dtype=%s",
                _SUPPORTED_DTYPES,
                str(dtype).replace("torch.", ""),
            )
            continue
        rows = []
        for shape in args.shapes:
            B, T, H, W, NH, HD, KT, KH, KW = shape
            rows.append(bench_na3d_flash(B, T, H, W, NH, HD, KT, KH, KW, dtype))
        df = pd.DataFrame(rows)
        aiter.logger.info(
            "na3d_flash summary (%s):\n%s",
            str(dtype).replace("torch.", ""),
            df.to_markdown(index=False),
        )


if __name__ == "__main__":
    main()
