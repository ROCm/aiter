# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""
Forward-only test / benchmark for the tuned FMHA CSV lookup path.

Uses triton.testing.do_bench for timing — same method as bench_fmha_varlen.py.

Usage examples
--------------
# Batch mode (default, same as bench_fmha_varlen.py -> bench_ck):
  python op_tests/test_tuned_fmha_fwd.py -b 1 -hq 8 -hk 8 -sq 5235 -sk 70771 -d 256 -dv 256 --dtype bf16

# Varlen mode (3D tensors + cu_seqlens, same as bench_fmha_varlen.py -> bench_varlen):
  python op_tests/test_tuned_fmha_fwd.py -b 1 -hq 8 -hk 8 -sq 5235 -sk 70771 -d 256 -dv 256 --dtype bf16 --mode varlen
"""

import argparse
import logging
import sys

import torch
import triton

logging.basicConfig(
    level=logging.INFO,
    format="%(name)s %(levelname)s: %(message)s",
)
logger = logging.getLogger("test_tuned_fmha_fwd")

import aiter
from aiter.ops.mha import mha_fwd
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.mha import find_tuned_fmha_config

# ---------------------------------------------------------------------------

DTYPE_MAP = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
}


def _fwd_flops(batch, hq, sq, sk, hdim_q, hdim_v):
    """FLOPs for fwd attention: 2 * batch * hq * sq * sk * (hdim_q + hdim_v)."""
    return 2.0 * batch * hq * sq * sk * (hdim_q + hdim_v)


# ---------------------------------------------------------------------------
# Batch mode — uses mha_fwd with 4D tensors (same as bench_ck in bench_fmha_varlen.py)
# ---------------------------------------------------------------------------

def run_batch_fwd(batch, hq, hk, sq, sk, hdim_q, hdim_v, dtype_str, causal=True):
    torch_dtype = DTYPE_MAP[dtype_str]
    scale = hdim_q ** -0.5

    gfx = get_gfx()
    cfg = find_tuned_fmha_config(gfx, dtype_str, batch, hq, hk, sq, sk, hdim_q, hdim_v)
    tile = cfg[0] if cfg else None
    num_splits = cfg[1] if cfg else 1
    if tile is None:
        logger.warning(
            "No CSV match for gfx=%s dtype=%s b=%d hq=%d hk=%d sq=%d sk=%d "
            "d=%d dv=%d — will use heuristic path.",
            gfx, dtype_str, batch, hq, hk, sq, sk, hdim_q, hdim_v,
        )
    else:
        logger.info("CSV match: tile_pattern=%s  num_splits=%d", tile, num_splits)

    q = torch.randn(batch, sq, hq, hdim_q, device="cuda", dtype=torch_dtype)
    k = torch.randn(batch, sk, hk, hdim_q, device="cuda", dtype=torch_dtype)
    v = torch.randn(batch, sk, hk, hdim_v, device="cuda", dtype=torch_dtype)

    fn = lambda: mha_fwd(
        q, k, v,
        dropout_p=0.0,
        softmax_scale=scale,
        is_causal=causal,
        window_size_left=-1,
        window_size_right=-1,
        sink_size=0,
        return_softmax_lse=True,
        return_dropout_randval=False,
        num_splits=num_splits,
    )

    # Warmup + sanity
    out = fn()
    torch.cuda.synchronize()
    out_tensor = out[0]
    assert not torch.isnan(out_tensor).any(), "Output contains NaN!"
    assert not torch.isinf(out_tensor).any(), "Output contains Inf!"

    # Benchmark — triton.testing.do_bench (same as bench_fmha_varlen.py)
    avg_ms = triton.testing.do_bench(fn, warmup=200, rep=1000)

    fwd_flop = _fwd_flops(batch, hq, sq, sk, hdim_q, hdim_v)
    tflops = fwd_flop / avg_ms * 1e-9

    logger.info(
        "PASS  batch   b=%-3d hq=%-3d hk=%-3d sq=%-6d sk=%-6d d=%-3d dv=%-3d "
        "dtype=%-4s  %8.4f ms (%8.1f us)  %7.1f TFLOP/s  [%s]",
        batch, hq, hk, sq, sk, hdim_q, hdim_v, dtype_str,
        avg_ms, avg_ms * 1000, tflops,
        f"tile={tile} splits={num_splits}" if tile else "heuristic",
    )
    return True


# ---------------------------------------------------------------------------
# Varlen mode — uses flash_attn_varlen_func with 3D tensors + cu_seqlens
# (same as bench_varlen in bench_fmha_varlen.py)
# ---------------------------------------------------------------------------

def run_varlen_fwd(batch, hq, hk, sq, sk, hdim_q, hdim_v, dtype_str, causal=True):
    torch_dtype = DTYPE_MAP[dtype_str]
    scale = hdim_q ** -0.5

    gfx = get_gfx()
    cfg = find_tuned_fmha_config(gfx, dtype_str, batch, hq, hk, sq, sk, hdim_q, hdim_v)
    tile = cfg[0] if cfg else None
    num_splits = cfg[1] if cfg else 1
    if tile is None:
        logger.warning(
            "No CSV match for gfx=%s dtype=%s b=%d hq=%d hk=%d sq=%d sk=%d "
            "d=%d dv=%d — will use heuristic path.",
            gfx, dtype_str, batch, hq, hk, sq, sk, hdim_q, hdim_v,
        )
    else:
        logger.info("CSV match: tile_pattern=%s  num_splits=%d", tile, num_splits)

    # Build 4D then flatten to 3D — same as bench_fmha_varlen.py -> bench_varlen
    q = torch.randn(batch, sq, hq, hdim_q, device="cuda", dtype=torch_dtype)
    k = torch.randn(batch, sk, hk, hdim_q, device="cuda", dtype=torch_dtype)
    v = torch.randn(batch, sk, hk, hdim_v, device="cuda", dtype=torch_dtype)

    cu_q = torch.arange(0, (batch + 1) * sq, sq, device="cuda", dtype=torch.int32)
    cu_k = torch.arange(0, (batch + 1) * sk, sk, device="cuda", dtype=torch.int32)

    q_flat = q.reshape(-1, hq, hdim_q)
    k_flat = k.reshape(-1, hk, hdim_q)
    v_flat = v.reshape(-1, hk, hdim_v)

    fn = lambda: aiter.flash_attn_varlen_func(
        q_flat, k_flat, v_flat, cu_q, cu_k, sq, sk, causal=causal,
    )

    # Warmup + sanity
    out = fn()
    torch.cuda.synchronize()
    out_tensor = out[0] if isinstance(out, tuple) else out
    assert not torch.isnan(out_tensor).any(), "Output contains NaN!"
    assert not torch.isinf(out_tensor).any(), "Output contains Inf!"

    # Benchmark — triton.testing.do_bench (same as bench_fmha_varlen.py)
    avg_ms = triton.testing.do_bench(fn, warmup=200, rep=1000)

    fwd_flop = _fwd_flops(batch, hq, sq, sk, hdim_q, hdim_v)
    tflops = fwd_flop / avg_ms * 1e-9

    logger.info(
        "PASS  varlen  b=%-3d hq=%-3d hk=%-3d sq=%-6d sk=%-6d d=%-3d dv=%-3d "
        "dtype=%-4s  %8.4f ms (%8.1f us)  %7.1f TFLOP/s  [%s]",
        batch, hq, hk, sq, sk, hdim_q, hdim_v, dtype_str,
        avg_ms, avg_ms * 1000, tflops,
        f"tile={tile} splits={num_splits}" if tile else "heuristic",
    )
    return True


# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="Forward-only test / benchmark for tuned FMHA CSV lookup.",
    )
    parser.add_argument(
        "--mode", type=str, choices=["varlen", "batch"], default="batch",
        help="varlen (3D + cu_seqlens) or batch (4D). Default: batch",
    )
    parser.add_argument("-b", "--batch", type=int, default=1, help="Batch size")
    parser.add_argument("-hq", type=int, default=8, help="Number of Q heads")
    parser.add_argument("-hk", type=int, default=8, help="Number of K heads")
    parser.add_argument("-sq", type=int, default=5235, help="Sequence length Q")
    parser.add_argument("-sk", type=int, default=70771, help="Sequence length K")
    parser.add_argument("-d", "--hdim_q", type=int, default=256, help="Head dim Q")
    parser.add_argument("-dv", "--hdim_v", type=int, default=256, help="Head dim V")
    parser.add_argument(
        "--dtype", type=str, choices=["bf16", "fp16"], default="bf16",
    )
    parser.add_argument("--no-causal", dest="causal", action="store_false", default=True)

    args = parser.parse_args()

    runner = run_varlen_fwd if args.mode == "varlen" else run_batch_fwd
    ok = runner(
        args.batch, args.hq, args.hk, args.sq, args.sk,
        args.hdim_q, args.hdim_v, args.dtype, args.causal,
    )
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
