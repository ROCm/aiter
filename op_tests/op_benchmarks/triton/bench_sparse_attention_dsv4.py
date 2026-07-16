# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Benchmark for the DSV4 sparse MLA prefill kernels.

Benchmarks the Triton prefill kernel against the Gluon (CDNA4 / gfx950) prefill
backend and reports the speedup over Triton. The Gluon backend is the unified
`mla_gluon(..., has_pe=False)` entry (HAS_PE=False over the shared `_mla_gluon`
kernel) — the backend the production entrance uses. When Gluon is unavailable
(non-gfx950 arch, or Triton < 3.6), only the Triton perf is reported.

Usage:
  python op_tests/op_benchmarks/triton/bench_sparse_attention_dsv4.py
  python op_tests/op_benchmarks/triton/bench_sparse_attention_dsv4.py --shapes prefill
"""

from aiter.ops.triton._triton_kernels.attention.sparse_attention_dsv4 import (
    _sparse_attn_prefill_kernel as csa_prefill_tl,
)

import argparse

import torch
import triton

# The Gluon prefill kernel is opt-in (gfx950 + Triton >= 3.6). Probe it once at
# import time; the benchmark falls back to Triton-only when unavailable.
try:
    from aiter.ops.triton.gluon.mla_gluon import mla_gluon
    from aiter.jit.utils.chip_info import get_gfx

    HAS_GLUON = get_gfx() == "gfx950"
except ImportError:
    mla_gluon = None
    HAS_GLUON = False


sparse_attn_prefill_kernel = csa_prefill_tl


NOPE_DIM = 448
ROPE_DIM = 64
HEAD_DIM = NOPE_DIM + ROPE_DIM  # 512


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
        flat.append(torch.randperm(max_slots, device=device, dtype=torch.int32)[:L])
        ptr.append(ptr[-1] + L)
    return torch.cat(flat), torch.tensor(ptr, dtype=torch.int32, device=device), lens


def _ref_prefill(q, kv, indices, indptr, scale, attn_sink=None):
    """torch reference: per-query masked softmax attention over the CSR-gathered KV."""
    out = torch.zeros_like(q, dtype=torch.float32)
    for t in range(q.shape[0]):
        s, e = int(indptr[t]), int(indptr[t + 1])
        if e <= s:
            continue
        K = kv[indices[s:e].long()].float()  # [L, D]
        sc = (q[t].float() @ K.t()) * scale  # [H, L]
        if attn_sink is not None:
            m = torch.maximum(sc.max(-1).values, attn_sink.float())
            p = torch.exp(sc - m[:, None])
            denom = p.sum(-1) + torch.exp(attn_sink.float() - m)
        else:
            m = sc.max(-1).values
            p = torch.exp(sc - m[:, None])
            denom = p.sum(-1)
        out[t] = (p / denom[:, None]) @ K
    return out.to(q.dtype)


# ---------------------------------------------------------------------------
# Kernel launchers
# ---------------------------------------------------------------------------
def _launch_prefill(
    backend,
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
    if backend == "gluon":
        mla_gluon(
            q,  # q_nope = combined-D query
            None,  # q_pe unused in prefill mode
            kv,  # kv_c
            out,  # o
            indices,  # page_table = ragged kv_indices
            indptr,  # seq_info = ragged kv_indptr
            scale,
            min_kv_seq_len=float("inf"),  # skip min_kv_seq_len check for decode
            has_pe=False,
            attn_sink=attn_sink if has_sink else None,
        )
        return
    else:  # default triton backend

        def grid(META):
            return (num_queries, triton.cdiv(num_heads, META["BLOCK_H"]))

        sparse_attn_prefill_kernel[grid](
            q,
            kv,
            indices,
            indptr,
            attn_sink,
            out,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            kv.stride(0),
            kv.stride(1),
            out.stride(0),
            out.stride(1),
            out.stride(2),
            num_heads,
            head_dim,
            kv.shape[0],
            scale,
            HAS_ATTN_SINK=has_sink,
            BLOCK_D=block_d,
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


def _time_backend(backend, q, kv, indices, indptr, num_queries, num_heads, scale):
    """Compile + time one backend; returns its mean per-call latency in ms."""
    attn_sink = torch.empty(1, device=q.device, dtype=torch.float32)
    out = torch.empty_like(q)
    _launch_prefill(
        backend,
        q,
        kv,
        indices,
        indptr,
        out,
        num_queries,
        num_heads,
        HEAD_DIM,
        False,
        attn_sink,
        scale,
    )
    torch.cuda.synchronize()
    return _bench(
        lambda: _launch_prefill(
            backend,
            q,
            kv,
            indices,
            indptr,
            out,
            num_queries,
            num_heads,
            HEAD_DIM,
            False,
            attn_sink,
            scale,
        )
    )


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
        nnz = int(indptr[-1].item())  # number of non-zeros
        scale = 1.0 / (HEAD_DIM**0.5)

        # FLOPS: per query, for each of `nnz` K positions, 2*H*D for QK + 2*H*D for PV.
        flops = 4.0 * num_heads * HEAD_DIM * nnz
        # Bytes: Q [Q,H,D] + KV gather [nnz, D] + out [Q,H,D]
        bytes_moved = (
            q.numel() * q.element_size()
            + nnz * HEAD_DIM * kv.element_size()
            + num_queries * num_heads * HEAD_DIM * 2
        )

        def _perf(ms):
            return flops / (ms * 1e-3) / 1e12, bytes_moved / (ms * 1e-3) / 1e9

        tri_ms = _time_backend(
            "triton", q, kv, indices, indptr, num_queries, num_heads, scale
        )
        tri_tflops, tri_gbps = _perf(tri_ms)

        if HAS_GLUON:
            glu_ms = _time_backend(
                "gluon", q, kv, indices, indptr, num_queries, num_heads, scale
            )
            glu_tflops, glu_gbps = _perf(glu_ms)
            speedup = tri_ms / glu_ms if glu_ms > 0 else float("nan")
            rows.append(
                (
                    num_queries,
                    num_heads,
                    num_kv,
                    topk,
                    tri_ms,
                    tri_tflops,
                    glu_ms,
                    glu_tflops,
                    f"{speedup:.2f}x",
                )
            )
        else:
            rows.append(
                (num_queries, num_heads, num_kv, topk, tri_ms, tri_tflops, tri_gbps)
            )

    if HAS_GLUON:
        headers = [
            "Q",
            "H",
            "Kv",
            "topk",
            "triton ms",
            "triton TFLOPS",
            "gluon ms",
            "gluon TFLOPS",
            "speedup",
        ]
    else:
        headers = ["Q", "H", "Kv", "topk", "triton ms", "triton TFLOPS", "triton GB/s"]
    _print_table("PREFILL", headers, rows)


def _print_table(title, headers, rows):
    def _fmt(x):
        if isinstance(x, float):
            return f"{x:.3f}" if x >= 1 or x == 0 else f"{x:.4f}"
        return str(x)

    cells = [[_fmt(c) for c in r] for r in rows]
    widths = [max(len(h), *(len(c[i]) for c in cells)) for i, h in enumerate(headers)]
    header = "| " + " | ".join(h.rjust(widths[i]) for i, h in enumerate(headers)) + " |"
    sep = "| " + " | ".join("-" * widths[i] for i in range(len(headers))) + " |"
    print(header)
    print(sep)
    for c in cells:
        print("| " + " | ".join(s.rjust(widths[i]) for i, s in enumerate(c)) + " |")


def check_correctness(device: str):
    """Quick torch-reference correctness gate, run once before profiling.

    Validates every available backend (Triton, and Gluon on gfx950) with and
    without the attention sink on one small shape. Raises on mismatch so a broken
    kernel fails loudly instead of being silently benchmarked.
    """
    print("\n========== CORRECTNESS ==========")
    num_queries, num_heads, num_kv, topk = 128, 128, 2048, 512
    torch.manual_seed(0)
    q = torch.randn(
        num_queries, num_heads, HEAD_DIM, dtype=torch.bfloat16, device=device
    )
    kv = torch.randn(num_kv, HEAD_DIM, dtype=torch.bfloat16, device=device)
    indices, indptr, _ = _build_csr(num_queries, num_kv, topk, device)
    scale = 1.0 / (HEAD_DIM**0.5)

    backends = ["triton", "gluon"] if HAS_GLUON else ["triton"]
    for backend in backends:
        for has_sink in (False, True):
            attn_sink = (
                torch.randn(num_heads, dtype=torch.float32, device=device)
                if has_sink
                else torch.empty(1, dtype=torch.float32, device=device)
            )
            out = torch.empty_like(q)
            _launch_prefill(
                backend,
                q,
                kv,
                indices,
                indptr,
                out,
                num_queries,
                num_heads,
                HEAD_DIM,
                has_sink,
                attn_sink,
                scale,
            )
            torch.cuda.synchronize()
            ref = _ref_prefill(
                q, kv, indices, indptr, scale, attn_sink if has_sink else None
            )
            max_diff = (out.float() - ref.float()).abs().max().item()
            torch.testing.assert_close(out.float(), ref.float(), atol=1e-2, rtol=1e-2)
            print(
                f"  {backend:6s} sink={str(has_sink):5s}: OK (max|delta|={max_diff:.4f})"
            )


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--shapes",
        choices=["all", "prefill"],
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
    args = p.parse_args()
    args.prefill_cfgs = [tuple(int(x) for x in s.split(",")) for s in args.prefill_cfgs]
    return args


def main():
    args = _parse_args()
    device = "cuda"
    print(
        f"GPU: {torch.cuda.get_device_name(0)}  "
        f"({torch.cuda.get_device_properties(0).multi_processor_count} CUs)"
    )
    print(f"Triton: {triton.__version__}")
    print(
        "Backends: Triton + Gluon MLA-decode prefill (gfx950)"
        if HAS_GLUON
        else "Backends: Triton only (Gluon kernel unavailable)"
    )
    if args.shapes in ("all", "prefill"):
        check_correctness(device)
        run_prefill_bench(args, device)


if __name__ == "__main__":
    main()
