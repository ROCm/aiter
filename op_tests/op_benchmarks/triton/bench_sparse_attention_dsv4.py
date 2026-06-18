# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Benchmark for the DSV4 sparse MLA prefill / decode Triton kernels.
Usage:
  python op_tests/op_benchmarks/triton/bench_sparse_attention_dsv4.py
  python op_tests/op_benchmarks/triton/bench_sparse_attention_dsv4.py --shapes prefill
  python op_tests/op_benchmarks/triton/bench_sparse_attention_dsv4.py --shapes decode

  # Benchmark the Gluon (CDNA4 / gfx950) kernels instead of the Triton ones:
  USE_GLUON=1 python op_tests/op_benchmarks/triton/bench_sparse_attention_dsv4.py

"""

from aiter.ops.triton.gluon.sparse_attention_dsv4 import (
    sparse_attn_prefill_gluon,
    sparse_attn_decode_gluon,
    # sparse_attn_decode_split_gluon,
)
from aiter.ops.triton._triton_kernels.attention.sparse_attention_dsv4 import (
    _sparse_attn_decode_kernel as csa_decode_tl,
    _sparse_attn_prefill_kernel as csa_prefill_tl,
)

import argparse
import importlib.util
import os
import sys

import torch
import triton

USE_GLUON = os.environ.get("USE_GLUON", "0") == "1"
BACKEND = "gluon" if USE_GLUON else "triton"


sparse_attn_prefill_kernel = csa_prefill_tl
sparse_attn_decode_kernel = csa_decode_tl


NOPE_DIM = 448
ROPE_DIM = 64
HEAD_DIM = NOPE_DIM + ROPE_DIM  # 512
NOPE_BLOCK = 512                 # next_power_of_2(448)
NOPE_BYTES = 576                 # 448 fp8 + 128 bf16 RoPE
SCALE_BYTES = 8                  # 8 fp8-exp scales per token
PER_BLOCK_ROW_BYTES = NOPE_BYTES + SCALE_BYTES  # 584


# ---------------------------------------------------------------------------
# Bench data builder
# ---------------------------------------------------------------------------
def _build_csr(num_q: int, max_slots: int, max_topk: int, device: str):
    lens = torch.randint(
        max(1, max_topk // 4),
        max_topk + 1,
        (num_q,),
        dtype=torch.int32,
        device=device,
    )
    flat, ptr = [], [0]
    for i in range(num_q):
        L = int(lens[i].item())
        flat.append(torch.randperm(
            max_slots, device=device, dtype=torch.int32)[:L])
        ptr.append(ptr[-1] + L)
    return torch.cat(flat), torch.tensor(ptr, dtype=torch.int32, device=device), lens


def _build_fp8_cache(num_blocks: int, block_size: int, device: str):
    """Encode bf16 NoPE+RoPE into the 584-byte/row fp8_ds_mla layout the
    decode kernel expects (NoPE fp8 + RoPE bf16 + per-block scale bytes)."""
    total = num_blocks * block_size
    nope_bf16 = torch.randn(
        total, NOPE_DIM, dtype=torch.bfloat16, device=device) * 0.3
    rope_bf16 = torch.randn(
        total, ROPE_DIM, dtype=torch.bfloat16, device=device) * 0.3

    grp = nope_bf16.float().view(total, NOPE_DIM // 64, 64)
    amax = grp.abs().amax(dim=-1).clamp(min=1e-8)
    exp_v = torch.ceil(torch.log2(amax / 224.0)
                       ).clamp(min=-126, max=128) + 127.0
    enc = exp_v.to(torch.uint8)
    sc_f32 = torch.pow(2.0, enc.float() - 127.0)
    fp8 = (grp / sc_f32.unsqueeze(-1)
           ).to(torch.float8_e4m3fn).view(total, NOPE_DIM)
    fp8_bytes = fp8.view(torch.uint8)
    rope_bytes = rope_bf16.view(torch.uint8).view(total, ROPE_DIM * 2)

    cache = torch.zeros(
        num_blocks, block_size, PER_BLOCK_ROW_BYTES, dtype=torch.uint8, device=device
    )
    flat = cache.view(num_blocks, block_size * PER_BLOCK_ROW_BYTES)
    nope_pb = fp8_bytes.view(num_blocks, block_size, NOPE_DIM)
    rope_pb = rope_bytes.view(num_blocks, block_size, ROPE_DIM * 2)
    enc_pb = enc.view(num_blocks, block_size, NOPE_DIM // 64)
    for pos in range(block_size):
        base = pos * NOPE_BYTES
        flat[:, base: base + NOPE_DIM] = nope_pb[:, pos]
        flat[:, base + NOPE_DIM: base + NOPE_BYTES] = rope_pb[:, pos]
    sb0 = block_size * NOPE_BYTES
    for pos in range(block_size):
        sb = sb0 + pos * SCALE_BYTES
        flat[:, sb: sb + (NOPE_DIM // 64)] = enc_pb[:, pos]
    return cache


# ---------------------------------------------------------------------------
# Kernel launchers
# ---------------------------------------------------------------------------
def _launch_prefill(
    q,
    kv,
    indices,
    indptr,
    out,
    num_queries,
    num_heads,
    head_dim,
    has_sink,
    attn_sink,
    scale,
):
    block_d = triton.next_power_of_2(head_dim)
    # Print each kernel input tensor shape (once; this is called every bench rep).
    if not getattr(_launch_prefill, "_shapes_printed", False):
        print("---- _sparse_attn_prefill_kernel input tensor shapes ----")
        for name, t in (
            ("q             ", q),
            ("kv            ", kv),
            ("indices       ", indices),
            ("indptr        ", indptr),
            ("attn_sink     ", attn_sink),
            ("out           ", out),
        ):
            print(f"  {name}: {tuple(t.shape)}  {t.dtype}")
        print(f"  block_d: ", block_d)
        _launch_prefill._shapes_printed = True

    if USE_GLUON:
        # Persistent Gluon kernel: the launcher builds the 1-D grid and passes
        # num_queries (it grid-strides over the (query, head-block) tile space).
        sparse_attn_prefill_gluon(
            q,
            kv,
            indices,
            indptr,
            out,
            scale,
            attn_sink=attn_sink if has_sink else None,
        )
        return

    grid = lambda META: (
        num_queries,
        triton.cdiv(num_heads, META["BLOCK_H"]),
    )
    sparse_attn_prefill_kernel[grid](
        q,
        kv,
        indices,
        indptr,
        attn_sink,
        out,
        q.stride(0), q.stride(1), q.stride(2),
        kv.stride(0), kv.stride(1),
        out.stride(0), out.stride(1), out.stride(2),
        num_heads,
        head_dim,
        kv.shape[0],
        scale,
        HAS_ATTN_SINK=has_sink,
        BLOCK_D=block_d,
    )


def _launch_decode(
    q,
    main_cache,
    main_idx,
    main_indptr,
    extra_cache,
    extra_idx,
    extra_indptr,
    attn_sink,
    out,
    num_q,
    num_heads,
    main_block_size,
    extra_block_size,
    has_extra,
    has_sink,
    scale,
):
    # Print each kernel input tensor shape (once; this is called every bench rep).
    if not getattr(_launch_decode, "_shapes_printed", False):
        print("---- _sparse_attn_decode_kernel input tensor shapes ----")
        for name, t in (
            ("q             ", q),
            ("main_cache    ", main_cache),
            ("main_idx      ", main_idx),
            ("main_indptr   ", main_indptr),
            ("extra_cache   ", extra_cache),
            ("extra_idx     ", extra_idx),
            ("extra_indptr  ", extra_indptr),
            ("attn_sink     ", attn_sink),
            ("out           ", out),
        ):
            print(f"  {name}: {tuple(t.shape)}  {t.dtype}")
        print(f"  has_extra: {has_extra}, has_sink: {has_sink}")
        _launch_decode._shapes_printed = True

    if USE_GLUON:
        # Persistent Gluon decode kernel: the host launcher builds the 1-D grid
        # and passes num_queries (it grid-strides over the tile space itself).
        # sparse_attn_decode_split_gluon(
        sparse_attn_decode_gluon(
            q,
            main_cache,
            main_idx,
            main_indptr,
            out,
            scale,
            extra_cache=extra_cache if has_extra else None,
            extra_indices=extra_idx if has_extra else None,
            extra_indptr=extra_indptr if has_extra else None,
            attn_sink=attn_sink if has_sink else None,
        )
        return

    grid = lambda META: (  # noqa: E731
        num_q,
        triton.cdiv(num_heads, META["BLOCK_H"]),
    )
    sparse_attn_decode_kernel[grid](
        q,
        main_cache,
        main_idx,
        main_indptr,
        extra_cache,
        extra_idx,
        extra_indptr,
        attn_sink,
        out,
        q.stride(0), q.stride(1),
        out.stride(0), out.stride(1),
        main_cache.stride(0),
        extra_cache.stride(0),
        main_cache.shape[0] * main_cache.shape[1],
        extra_cache.shape[0] * extra_cache.shape[1],
        main_block_size,
        extra_block_size,
        scale,
        num_heads,
        HAS_ATTN_SINK=has_sink,
        HAS_EXTRA=has_extra,
        NOPE_DIM=NOPE_DIM,
        NOPE_BLOCK=NOPE_BLOCK,
        ROPE_DIM=ROPE_DIM,
        IS_FNUZ=False,
    )


# ---------------------------------------------------------------------------
# Benchmark drivers
# ---------------------------------------------------------------------------
def _bench(fn, *, warmup=5, reps=50):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    ev0 = torch.cuda.Event(enable_timing=True)
    ev1 = torch.cuda.Event(enable_timing=True)
    ev0.record()
    for _ in range(reps):
        fn()
    ev1.record()
    torch.cuda.synchronize()
    return ev0.elapsed_time(ev1) / reps


def run_prefill_bench(args, device: str):
    print("\n========== PREFILL ==========")
    rows = []
    for cfg in args.prefill_cfgs:
        num_queries, num_heads, num_kv, topk = cfg
        torch.manual_seed(0)
        q = torch.randn(
            num_queries, num_heads, HEAD_DIM, dtype=torch.bfloat16, device=device
        )
        kv = torch.randn(num_kv, HEAD_DIM, dtype=torch.bfloat16, device=device)
        indices, indptr, _ = _build_csr(num_queries, num_kv, topk, device)
        nnz = int(indptr[-1].item())        # number of non-zeros
        scale = 1.0 / (HEAD_DIM ** 0.5)
        attn_sink = torch.empty(1, device=device, dtype=torch.float32)
        out_tri = torch.empty_like(q)

        _launch_prefill(
            q, kv, indices, indptr, out_tri,
            num_queries, num_heads, HEAD_DIM, False, attn_sink, scale,
        )
        torch.cuda.synchronize()

        ms = _bench(
            lambda: _launch_prefill(
                q, kv, indices, indptr, out_tri,
                num_queries, num_heads, HEAD_DIM, False, attn_sink, scale,
            )
        )

        # FLOPS: per query, for each of `nnz` K positions, 2*H*D for QK + 2*H*D for PV.
        flops = 4.0 * num_heads * HEAD_DIM * nnz
        # Bytes: Q [Q,H,D] + KV gather [nnz, D] + out [Q,H,D]
        bytes_moved = (
            q.numel() * q.element_size()
            + nnz * HEAD_DIM * kv.element_size()
            + out_tri.numel() * out_tri.element_size()
        )
        rows.append(
            (num_queries, num_heads, num_kv, topk,
             ms,
             flops / (ms * 1e-3) / 1e12,
             bytes_moved / (ms * 1e-3) / 1e9)
        )
    _print_table(
        "PREFILL",
        ["Q", "H", "Kv", "topk",
         f"{BACKEND} ms", f"{BACKEND} TFLOPS", f"{BACKEND} GB/s"],
        rows,
    )


def run_decode_bench(args, device: str):
    print("\n========== DECODE ==========")
    rows = []
    for cfg in args.decode_cfgs:
        num_q, num_heads, block_size, num_blocks, topk, has_extra = cfg
        has_extra = bool(has_extra)
        torch.manual_seed(0)
        swa_cache = _build_fp8_cache(num_blocks, block_size, device)
        q = torch.randn(num_q, num_heads, HEAD_DIM,
                        dtype=torch.bfloat16, device=device)
        swa_idx, swa_indptr, _ = _build_csr(
            num_q, num_blocks * block_size, topk, device
        )
        swa_nnz = int(swa_indptr[-1].item())

        if has_extra:
            extra_cache = _build_fp8_cache(num_blocks, block_size, device)
            extra_idx, extra_indptr, _ = _build_csr(
                num_q, num_blocks * block_size, topk, device
            )
            extra_nnz = int(extra_indptr[-1].item())
        else:
            extra_cache = swa_cache
            extra_idx = torch.empty(0, dtype=torch.int32, device=device)
            extra_indptr = torch.zeros(
                num_q + 1, dtype=torch.int32, device=device)
            extra_nnz = 0

        attn_sink = torch.empty(1, device=device, dtype=torch.float32)
        scale = 1.0 / (HEAD_DIM ** 0.5)

        out_tri = torch.empty_like(q)

        ms = _bench(
            lambda: _launch_decode(
                q, swa_cache, swa_idx, swa_indptr,
                extra_cache, extra_idx, extra_indptr,
                attn_sink, out_tri, num_q, num_heads,
                block_size, block_size, has_extra, False, scale,
            )
        )

        total_nnz = swa_nnz + extra_nnz
        flops = 4.0 * num_heads * HEAD_DIM * total_nnz
        # K bytes per token: NOPE_DIM fp8 + ROPE_DIM bf16 + SCALE_BYTES → 448+128+8 = 584
        bytes_moved = (
            q.numel() * q.element_size()
            + total_nnz * PER_BLOCK_ROW_BYTES
            + out_tri.numel() * out_tri.element_size()
        )
        rows.append(
            (num_q, num_heads, block_size, num_blocks, topk, int(has_extra),
             ms,
             flops / (ms * 1e-3) / 1e12,
             bytes_moved / (ms * 1e-3) / 1e9)
        )
    _print_table(
        "DECODE",
        ["Q", "H", "blk", "N", "topk", "ext",
         f"{BACKEND} ms", f"{BACKEND} TFLOPS", f"{BACKEND} GB/s"],
        rows,
    )


def _print_table(title, headers, rows):
    def _fmt(x):
        if isinstance(x, float):
            return f"{x:.3f}" if x >= 1 or x == 0 else f"{x:.4f}"
        return str(x)
    cells = [[_fmt(c) for c in r] for r in rows]
    widths = [max(len(h), *(len(c[i]) for c in cells))
              for i, h in enumerate(headers)]
    line = " | ".join(h.rjust(widths[i]) for i, h in enumerate(headers))
    sep = "-+-".join("-" * w for w in widths)
    print(line)
    print(sep)
    for c in cells:
        print(" | ".join(s.rjust(widths[i]) for i, s in enumerate(c)))


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--shapes",
        choices=["all", "prefill", "decode"],
        default="all",
    )
    p.add_argument(
        "--prefill_cfgs",
        nargs="+",
        type=str,
        default=[
            # (num_queries, num_heads, num_kv, topk)
            "4096,128,4096,512",
            "4096,128,4096,1024",
            "8192,128,8192,512",
            "8192,128,8192,1024",
        ],
    )
    p.add_argument(
        "--decode_cfgs",
        nargs="+",
        type=str,
        default=[
            # Format: (num_q, num_heads, block_size, num_blocks, topk, has_extra)
            # num_blocks only needs to satisfy: num_blocks * block_size >= topk

            # ── SWA-only (CR=1, sbs=256): 512 × 256 = 131 072 >> topk ──
            # "1,128,256,512,1024,0",
            "128,128,256,512,1024,0",
            "2048,128,256,512,1024,0",
            "4096,128,256,512,1024,0",
            "8192,128,256,512,1024,0",

            # ── C4A (CR=4, sbs=64): 512 × 64 = 32 768 >> topk ──
            # "1,128,64,512,1024,1",
            # "32,128,64,512,1024,1",
            # "2048,128,64,512,1024,1",
            # "4096,128,64,512,1024,1",

            # ── C128A (CR=128, sbs=2): 2048 × 2 = 4096 >> topk ──
            # "1,128,2,2048,1024,1",
            # "32,128,2,2048,1024,1",
            # "128,128,2,2048,1024,1",
        ],
    )
    args = p.parse_args()
    args.prefill_cfgs = [tuple(int(x) for x in s.split(","))
                         for s in args.prefill_cfgs]
    args.decode_cfgs = [tuple(int(x) for x in s.split(","))
                        for s in args.decode_cfgs]
    return args


def main():
    args = _parse_args()
    device = "cuda"
    print(
        f"GPU: {torch.cuda.get_device_name(0)}  "
        f"({torch.cuda.get_device_properties(0).multi_processor_count} CUs)"
    )
    print(f"Triton: {triton.__version__}")
    print(f"Backend: {'GLUON (gfx950)' if USE_GLUON else 'TRITON'} ")
    # if args.shapes in ("all", "prefill"):
    #     run_prefill_bench(args, device)
    if args.shapes in ("all", "decode"):
        run_decode_bench(args, device)


if __name__ == "__main__":
    main()
