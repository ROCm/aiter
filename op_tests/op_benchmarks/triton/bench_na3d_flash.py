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

import functools
import itertools

import pandas as pd
import torch
import triton

import aiter
from aiter import dtypes
from aiter.jit.utils.chip_info import get_gfx_runtime as get_gfx
from aiter.ops.triton.attention.na3d_flash import na3d_flash_attn
from aiter.test_common import benchmark, checkAllclose
from op_tests.op_benchmarks.triton.utils.argparse import (
    get_do_bench_graph_kwargs,
    get_parser,
    get_rot_bufs_kwargs,
)
from op_tests.op_benchmarks.triton.utils.benchmark_utils import (
    do_bench_aiter_triton,
    generate_rotating_buffers_pool,
)
from op_tests.triton_tests.attention.test_na3d_flash import (
    _DEFAULT_SHAPES,
    _na3d_sdpa_exact,
)

SUPPORTED_GFX = ["gfx942", "gfx950"]
_SUPPORTED_DTYPES = (torch.bfloat16,)

# Set once in main() from parsed CLI args, read inside bench_na3d_flash. Kept
# out of its parameter list so @benchmark()'s inspect.getcallargs logging
# doesn't turn these into extra dict-valued columns in the result table.
_KWARGS_ROT_BUFS: dict = {}
_KWARGS_DO_BENCH_GRAPH: dict = {}


def _generate_pool_entry(B, T, H, W, NH, HD, dtype):
    scale = HD**-0.5
    q = torch.randn(B, T, H, W, NH, HD, dtype=dtype, device="cuda") * scale
    k = torch.randn(B, T, H, W, NH, HD, dtype=dtype, device="cuda")
    qkv = torch.randn(B, T, H, W, 3 * NH * HD, dtype=dtype, device="cuda")
    v = qkv.chunk(3, dim=-1)[2].view(B, T, H, W, NH, HD)
    return q, k, v


_POOL_FIELDS = ["q", "k", "v"]


def _make_candidate_fns(kernel_size):
    def triton_fn(q, k, v, **_):
        return na3d_flash_attn(q, k, v, kernel_size=kernel_size)

    def sdpa_exact_fn(q, k, v, **_):
        return _na3d_sdpa_exact(q, k, v, kernel_size=kernel_size)

    return {"triton": triton_fn, "sdpa_exact": sdpa_exact_fn}


def _rotating_call(fn, rot_bufs):
    """Manual pool rotation for candidates that can't be CUDA-graph captured.

    ``_na3d_sdpa_exact`` builds its attention mask with fresh, non-pinned
    host->device tensors on every call (see ``_na3d_sdpa_mask``), which
    ``torch.cuda.graph`` capture rejects outright. It still benefits from
    rotating through independent input sets, just not via
    ``do_bench_aiter_triton``'s graph replay.
    """
    pool_len = len(rot_bufs)
    counter = itertools.count()

    def run():
        return fn(**rot_bufs[next(counter) % pool_len])

    return run


@benchmark()
def bench_na3d_flash(B, T, H, W, NH, HD, KT, KH, KW, dtype):
    SEQ = T * H * W
    C = NH * HD
    K = KT * KH * KW  # neighborhood size

    # Match the LTX-2.5 decoder layout so the timed call includes the wrapper
    # overhead users actually pay.  In the fused-QKV processor q/k pass through
    # norm + rope (fresh contiguous tensors), but v is returned directly as
    # qkv.chunk(3, dim=-1)[2].view(...), i.e. NON-contiguous (W-stride = 3*NH*HD).
    # na3d_flash_attn's v.contiguous() is therefore a real full-volume staging copy
    # here, not the no-op it would be for a freshly allocated contiguous v.
    pool_gen = functools.partial(_generate_pool_entry, B, T, H, W, NH, HD, dtype)
    rot_bufs = generate_rotating_buffers_pool(
        [pool_gen], [_POOL_FIELDS], **_KWARGS_ROT_BUFS
    )
    assert not rot_bufs[0]["v"].is_contiguous()  # v.contiguous() copies in-call

    # Reference: grouped-SDPA exact path (same as the eager processor fallback).
    # Also used as a timed candidate so the table shows the real kernel speedup.
    ref = _na3d_sdpa_exact(
        rot_bufs[0]["q"], rot_bufs[0]["k"], rot_bufs[0]["v"], kernel_size=(KT, KH, KW)
    )

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
    elem = rot_bufs[0]["q"].element_size()
    nbytes = (2 * B * SEQ * C + 2 * B * SEQ * K * C) * elem

    candidates = _make_candidate_fns((KT, KH, KW))

    ret = {"gfx": get_gfx()}
    for name, fn in candidates.items():
        out = fn(**rot_bufs[0])
        err = checkAllclose(
            ref.float(),
            out.float(),
            rtol=1e-2,
            atol=5e-2,
            msg=f"{name}: na3d_flash (T={T},H={H},W={W},k=({KT},{KH},{KW}))",
        )
        if name == "sdpa_exact":
            ms = triton.testing.do_bench(_rotating_call(fn, rot_bufs), warmup=5, rep=20)
        else:
            ms = do_bench_aiter_triton(fn, rot_bufs, **_KWARGS_DO_BENCH_GRAPH)
        us = ms * 1000
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

    parser = get_parser(kernel_name="NA3D Flash Attention")
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

    global _KWARGS_ROT_BUFS, _KWARGS_DO_BENCH_GRAPH
    _KWARGS_ROT_BUFS = get_rot_bufs_kwargs(args)
    _KWARGS_DO_BENCH_GRAPH = get_do_bench_graph_kwargs(args)

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
