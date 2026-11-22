# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Benchmark ``pa_prefill_sparse``: sparse paged-prefill attention over two
KV sources (prefix + extend) with per-head sink. The default shapes use the
DSv4-Pro gathered widths (134 and 352 entries per token), variable length.

  # default: the two DSv4 layer classes, variable length
  python op_tests/op_benchmarks/triton/bench_pa_prefill_sparse.py

  # one shape, fixed length, packed fp8
  python op_tests/op_benchmarks/triton/bench_pa_prefill_sparse.py \\
      --shape 4096 64 512 134 1 --no-variable-len --fp8
"""

import argparse
import os
import sys

# Run as a script, import aiter and op_tests from this checkout.
_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if os.path.isdir(os.path.join(_REPO, "aiter")) and _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import torch
import triton

from aiter import logger
from aiter.ops.triton.attention.pa_prefill_sparse import pa_prefill_sparse
from aiter.ops.triton.utils._triton import arch_info
from op_tests.triton_tests.utils.dsv4_sparse_attn import (
    make_pa_prefill_sparse_inputs,
    v4_pack_2buff,
)

DEVICE_ARCH = arch_info.get_arch()

# (T, H, D, prefix_len, extend_len, label)
# prefix_len is the per-token gathered width; 134/352 are the measured DSv4
# cr=128 / cr=4 widths. T is the prefill chunk's token count.
DEFAULT_SHAPES = [
    (1024, 64, 512, 134, 1, "cr128-1k"),
    (1024, 64, 512, 352, 1, "cr4-1k"),
    (4096, 64, 512, 134, 1, "cr128-4k"),
    (4096, 64, 512, 352, 1, "cr4-4k"),
    (16128, 64, 512, 134, 1, "cr128-16k"),
    (16128, 64, 512, 352, 1, "cr4-16k"),
]


def build_inputs(T, H, D, prefix_len, extend_len, variable_len, extend_pool, pool):
    """The test's input builder, with the pool sizes this benchmark controls."""
    # Pool size controls cache residency: the default pools are small and
    # mostly cache-resident; raise --pool until 'reuse' approaches 1.
    total_pages = pool if pool is not None else max(T * 4, prefix_len * 64)
    total_extend = extend_pool if extend_pool is not None else max(T, extend_len * 64)
    q, ukv, p_idx, p_ptr, kv, e_idx, e_ptr, sink, scale = make_pa_prefill_sparse_inputs(
        T,
        H,
        D,
        prefix_len,
        extend_len,
        total_pages=total_pages,
        total_extend_tokens=total_extend,
        variable_len=variable_len,
    )
    return {
        "q": q,
        "unified_kv": ukv,
        "kv_indices_prefix": p_idx,
        "kv_indptr_prefix": p_ptr,
        "kv": kv,
        "kv_indices_extend": e_idx,
        "kv_indptr_extend": e_ptr,
        "attn_sink": sink,
        "softmax_scale": scale,
        # The inputs have no -1 slots, so skip the per-slot sentinel check.
        "has_invalid": False,
    }


def _kv_row_bytes(D, fp8):
    """Bytes of one KV row: D bf16 values, or a 512 B packed row + 64 bf16 RoPE."""
    return 512 + 64 * 2 if fp8 else D * 2


def flops_bytes(T, H, D, n_slots, n_uniq_rows, fp8):
    """FLOPs count every gathered slot. Traffic counts each distinct KV row once,
    so TB/s is a lower bound."""
    flops = 4.0 * H * D * n_slots  # QK (2) + PV (2)
    kv_bytes = float(n_uniq_rows) * _kv_row_bytes(D, fp8)
    # Q in (packed on the fp8 path), O out in bf16.
    q_bytes = float(T) * H * (640 if fp8 else D * 2)
    o_bytes = float(T) * H * D * 2
    return flops, kv_bytes + q_bytes + o_bytes


def bench_one(shape, variable_len, rep, cudagraph, fp8, extend_pool, pool):
    T, H, D, p_len, e_len, label = shape
    inp = build_inputs(T, H, D, p_len, e_len, variable_len, extend_pool, pool)
    p_idx, e_idx = inp["kv_indices_prefix"], inp["kv_indices_extend"]

    if fp8:
        q_packed, q_rope = v4_pack_2buff(inp["q"])
        kv_packed, kv_rope = v4_pack_2buff(inp["unified_kv"])
        args = dict(
            inp,
            q=q_packed,
            unified_kv=kv_packed,
            q_rope=q_rope,
            unified_kv_rope=kv_rope,
        )
        # A 1-wide extend pool is a placeholder, so fp8 runs prefix-only there.
        if e_len > 1:
            ext_packed, ext_rope = v4_pack_2buff(inp["kv"])
            args.update(kv=ext_packed, kv_rope=ext_rope)
        else:
            args.update(kv=None, kv_indices_extend=None, kv_indptr_extend=None)
            e_idx = e_idx[:0]
    else:
        args = inp

    def fn():
        return pa_prefill_sparse(**args)

    try:
        fn()
        torch.cuda.synchronize()
    except Exception as e:  # noqa: BLE001 - report and keep going
        logger.warning("%-12s FAIL %s: %.70s", label, type(e).__name__, e)
        return None

    if cudagraph:
        ms = triton.testing.do_bench_cudagraph(fn, rep=rep)
    else:
        ms = triton.testing.do_bench(fn, warmup=max(1, rep // 4), rep=rep)

    n_slots = p_idx.numel() + e_idx.numel()
    n_uniq = int(torch.unique(p_idx).numel()) + int(torch.unique(e_idx).numel())
    flops, byts = flops_bytes(T, H, D, n_slots, n_uniq, fp8)
    return {
        "label": label,
        "T": T,
        "H": H,
        "D": D,
        "prefix": p_len,
        "extend": e_len,
        "indices": n_slots,
        "uniq": n_uniq,
        "reuse": n_slots / max(1, n_uniq),
        "us": ms * 1e3,
        "tflops": flops / (ms * 1e-3) / 1e12,
        "tbps": byts / (ms * 1e-3) / 1e12,
    }


def main(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--shape",
        type=int,
        nargs=5,
        action="append",
        metavar=("T", "H", "D", "PREFIX_LEN", "EXTEND_LEN"),
        help="Shape to benchmark; repeat for several. PREFIX_LEN is the "
        "per-token gathered width (DSv4: 134 for cr=128, 352 for cr=4). "
        "Default: the six DSv4 shapes.",
    )
    p.add_argument(
        "--variable-len",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Draw per-token lengths in [1, PREFIX_LEN] instead of "
        "a uniform width (default: on).",
    )
    p.add_argument(
        "--fp8",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Run the DSv4 packed-fp8 path instead of bf16 (prefix-only when "
        "EXTEND_LEN is 1).",
    )
    p.add_argument(
        "--pool",
        type=int,
        default=None,
        help="rows in the prefix KV pool. Default is small enough that the "
        "working set is cache-resident; raise it until 'reuse' approaches 1.",
    )
    p.add_argument(
        "--extend-pool",
        type=int,
        default=None,
        help="rows in the extend KV pool. Default sizes it from T.",
    )
    p.add_argument(
        "--rep", type=int, default=50, help="do_bench target in ms (default 50)."
    )
    p.add_argument("--cudagraph", action=argparse.BooleanOptionalAction, default=True)
    a = p.parse_args(argv)

    if DEVICE_ARCH not in ("gfx1250", "gfx1250-strict", "gfx950"):
        logger.warning(
            "pa_prefill_sparse has no tuned path on %s; running the portable "
            "Triton fallback.",
            DEVICE_ARCH,
        )

    shapes = (
        [tuple(s) + (f"T{s[0]}-p{s[3]}",) for s in a.shape]
        if a.shape
        else DEFAULT_SHAPES
    )

    logger.info(
        "arch=%s  fp8=%s  variable_len=%s  cudagraph=%s  rep=%d",
        DEVICE_ARCH,
        a.fp8,
        a.variable_len,
        a.cudagraph,
        a.rep,
    )
    logger.info(
        "%-12s%7s%5s%5s%8s%8s%10s%9s%7s%10s%9s%8s",
        "shape",
        "T",
        "H",
        "D",
        "prefix",
        "extend",
        "indices",
        "uniq",
        "reuse",
        "us",
        "TF/s",
        "TB/s",
    )
    logger.info("%s", "-" * 98)
    rows = []
    for s in shapes:
        r = bench_one(
            s, a.variable_len, a.rep, a.cudagraph, a.fp8, a.extend_pool, a.pool
        )
        if r is None:
            continue
        rows.append(r)
        logger.info(
            "%-12s%7d%5d%5d%8d%8d%10d%9d%7.1f%10.1f%9.1f%8.2f",
            r["label"],
            r["T"],
            r["H"],
            r["D"],
            r["prefix"],
            r["extend"],
            r["indices"],
            r["uniq"],
            r["reuse"],
            r["us"],
            r["tflops"],
            r["tbps"],
        )
    if not rows:
        sys.exit("no shape completed")
    return rows


if __name__ == "__main__":
    main()
