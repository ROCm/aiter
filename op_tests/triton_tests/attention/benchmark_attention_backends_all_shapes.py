#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""
Compare multiple attention backends on all shapes from a unified-attention JSONL trace.

Backends:
- unified_default / unified_force_2d / unified_force_3d
- ck_pa_naive (CK tile decode-only)
- triton_pa_decode (decode-only)
- triton_pa_prefill (prefill-only)
- rocm_pa_common (decode-only, from aiter.ops.attention.paged_attention_common)
- asm_pa_fwd (decode-only, from aiter.ops.attention.pa_fwd_asm)

The script auto-skips unsupported shape/backend combinations.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import json
import math
import random
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import torch
import triton.language as tl

from aiter.ops.triton.attention import unified_attention as ua_mod


DTYPE_MAP: dict[str, torch.dtype] = {
    "torch.float16": torch.float16,
    "torch.bfloat16": torch.bfloat16,
    "torch.float32": torch.float32,
    "torch.int8": torch.int8,
    "torch.float8_e4m3fn": torch.float8_e4m3fn,
    "torch.float8_e4m3fnuz": torch.float8_e4m3fnuz,
}

BACKEND_LIST = [
    "unified_default",
    "unified_force_2d",
    "unified_force_3d",
    "ck_pa_naive",
    "triton_pa_decode",
    "triton_pa_prefill",
    "rocm_pa_common",
    "asm_pa_fwd",
]


@dataclass(frozen=True)
class Signature:
    q_shape: tuple[int, int, int]
    k_shape: tuple[int, int, int, int]
    block_table_shape: tuple[int, int]
    max_seqlen_q: int
    max_seqlen_k: int
    window_size: tuple[int, int]
    num_seqs: int
    q_dtype: str
    k_dtype: str
    softmax_scale: float
    softcap: float
    has_sinks: bool


@dataclass
class Case:
    sig: Signature
    count: int
    observed_kind: str


def parse_dtype(name: str) -> torch.dtype:
    if name not in DTYPE_MAP:
        raise ValueError(f"Unsupported dtype: {name}")
    return DTYPE_MAP[name]


def read_cases(path: Path, dedupe: bool, max_shapes: int) -> list[Case]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if not rows:
        return []

    if not dedupe:
        out = []
        for r in rows:
            sig = Signature(
                q_shape=tuple(r["q_shape"]),
                k_shape=tuple(r["k_shape"]),
                block_table_shape=tuple(r["block_table_shape"]),
                max_seqlen_q=int(r["max_seqlen_q"]),
                max_seqlen_k=int(r["max_seqlen_k"]),
                window_size=tuple(r["window_size"]),
                num_seqs=int(r["num_seqs"]),
                q_dtype=str(r["q_dtype"]),
                k_dtype=str(r["k_dtype"]),
                softmax_scale=float(r["softmax_scale"]),
                softcap=float(r.get("softcap", 0.0)),
                has_sinks=bool(r.get("has_sinks", False)),
            )
            out.append(Case(sig=sig, count=1, observed_kind=str(r["kind"])))
        return out[:max_shapes] if max_shapes > 0 else out

    sig_counter: Counter[Signature] = Counter()
    sig_kind: dict[Signature, str] = {}
    for r in rows:
        sig = Signature(
            q_shape=tuple(r["q_shape"]),
            k_shape=tuple(r["k_shape"]),
            block_table_shape=tuple(r["block_table_shape"]),
            max_seqlen_q=int(r["max_seqlen_q"]),
            max_seqlen_k=int(r["max_seqlen_k"]),
            window_size=tuple(r["window_size"]),
            num_seqs=int(r["num_seqs"]),
            q_dtype=str(r["q_dtype"]),
            k_dtype=str(r["k_dtype"]),
            softmax_scale=float(r["softmax_scale"]),
            softcap=float(r.get("softcap", 0.0)),
            has_sinks=bool(r.get("has_sinks", False)),
        )
        sig_counter[sig] += 1
        sig_kind.setdefault(sig, str(r["kind"]))

    ranked = sig_counter.most_common(max_shapes if max_shapes > 0 else None)
    return [Case(sig=s, count=c, observed_kind=sig_kind[s]) for s, c in ranked]


def synth_q_lens(total_tokens: int, num_seqs: int) -> list[int]:
    if num_seqs <= 0:
        return [total_tokens]
    base = total_tokens // num_seqs
    rem = total_tokens % num_seqs
    return [base + (1 if i < rem else 0) for i in range(num_seqs)]


def build_unified_inputs(sig: Signature, seed: int = 0) -> dict[str, Any]:
    random.seed(seed)
    torch.manual_seed(seed)
    q_dtype = parse_dtype(sig.q_dtype)
    k_dtype = parse_dtype(sig.k_dtype)

    q = torch.randn(*sig.q_shape, dtype=q_dtype, device="cuda")
    k = torch.randn(*sig.k_shape, dtype=k_dtype, device="cuda")
    v = torch.randn(*sig.k_shape, dtype=k_dtype, device="cuda")
    out = torch.empty_like(q)

    q_lens = synth_q_lens(sig.q_shape[0], sig.num_seqs)
    cu = [0]
    for x in q_lens:
        cu.append(cu[-1] + x)
    cu_seqlens_q = torch.tensor(cu, dtype=torch.int32, device="cuda")
    seqused_k = torch.full((sig.num_seqs,), sig.max_seqlen_k, dtype=torch.int32, device="cuda")

    rows, cols = sig.block_table_shape
    if rows != sig.num_seqs:
        rows = sig.num_seqs
    block_table = torch.randint(
        0, sig.k_shape[0], (rows, cols), dtype=torch.int32, device="cuda"
    )
    sinks = (
        torch.randn(sig.q_shape[1], dtype=torch.bfloat16, device="cuda")
        if sig.has_sinks
        else None
    )
    return {
        "q": q,
        "k": k,
        "v": v,
        "out": out,
        "cu_seqlens_q": cu_seqlens_q,
        "max_seqlen_q": sig.max_seqlen_q,
        "seqused_k": seqused_k,
        "max_seqlen_k": sig.max_seqlen_k,
        "softmax_scale": sig.softmax_scale,
        "causal": True,
        "window_size": sig.window_size,
        "block_table": block_table,
        "softcap": sig.softcap,
        "q_descale": None,
        "k_descale": None,
        "v_descale": None,
        "alibi_slopes": None,
        "output_scale": None,
        "qq_bias": None,
        "sinks": sinks,
    }


@contextlib.contextmanager
def override_unified(mode: str):
    old = ua_mod.use_2d_kernel
    if mode == "unified_default":
        yield
        return
    if mode == "unified_force_2d":
        ua_mod.use_2d_kernel = lambda *args, **kwargs: True
    elif mode == "unified_force_3d":
        ua_mod.use_2d_kernel = lambda *args, **kwargs: False
    try:
        yield
    finally:
        ua_mod.use_2d_kernel = old


def cache_layout_decode(k_u: torch.Tensor, v_u: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    # unified cache: [num_blocks, block_size, num_kv_heads, head]
    # decode cache:  [num_blocks, num_kv_heads, block_size, head]
    return k_u.permute(0, 2, 1, 3).contiguous(), v_u.permute(0, 2, 1, 3).contiguous()


def cache_layout_asm(k_u: torch.Tensor, v_u: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    # asm K: [num_blocks, num_kv_heads, head_size/x, block_size, x]
    # asm V: [num_blocks, num_kv_heads, head_size, block_size]
    nb, blk, kvh, hd = k_u.shape
    elem = torch.tensor([], dtype=k_u.dtype).element_size()
    x = min(hd, max(1, 16 // elem))
    if hd % x != 0:
        raise RuntimeError(f"head_size={hd} not divisible by vector size x={x}")
    k = k_u.permute(0, 2, 3, 1).contiguous()  # [nb, kvh, hd, blk]
    k = k.view(nb, kvh, hd // x, x, blk).permute(0, 1, 2, 4, 3).contiguous()
    v = v_u.permute(0, 2, 3, 1).contiguous()
    return k, v


def run_kernel(fn: Callable[[], None], warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    return (t1 - t0) * 1e3 / iters


def bench_backend(
    case: Case, backend: str, warmup: int, iters: int, ignore_sinks: bool
) -> tuple[float | None, str]:
    sig = case.sig
    inp = build_unified_inputs(sig, seed=123)

    # Unified variants are always valid.
    if backend in {"unified_default", "unified_force_2d", "unified_force_3d"}:
        def _call():
            with override_unified(backend):
                ua_mod.unified_attention(**inp)

        return run_kernel(_call, warmup, iters), "ok"

    # Decode-only paths.
    is_decode = sig.max_seqlen_q == 1
    is_global = sig.window_size == (-1, -1)
    no_sinks = not sig.has_sinks
    decode_global_ok = is_decode and is_global and (no_sinks or ignore_sinks)

    if backend == "ck_pa_naive":
        if not decode_global_ok:
            return None, "skip:decode-global-no-sinks-only(use --ignore-sinks)"
        try:
            import aiter
        except Exception as e:  # pragma: no cover
            return None, f"skip:import:{e}"

        q = inp["q"]
        out = torch.empty_like(q)
        k_asm, v_asm = cache_layout_asm(inp["k"], inp["v"])
        block_tables = inp["block_table"]
        context_lens = inp["seqused_k"]

        # quant_algo=0 (NO quant), scales can be empty.
        k_dequant_scales = torch.empty((0,), dtype=torch.float32, device=q.device)
        v_dequant_scales = torch.empty((0,), dtype=torch.float32, device=q.device)

        def _call():
            aiter.pa_fwd_naive(
                q,
                k_asm,
                v_asm,
                block_tables,
                context_lens,
                k_dequant_scales,
                v_dequant_scales,
                sig.max_seqlen_k,
                sig.k_shape[2],  # num_kv_heads
                float(sig.softmax_scale),
                1.0,  # scale_k
                1.0,  # scale_v
                sig.k_shape[1],  # block_size
                0,  # quant_algo = NO
                out,
            )

        status = "ok"
        if sig.has_sinks and ignore_sinks:
            status = "ok:ignore_sinks_semantics"
        return run_kernel(_call, warmup, iters), status

    if backend == "triton_pa_decode":
        if not decode_global_ok:
            return None, "skip:decode-global-no-sinks-only(use --ignore-sinks)"
        from aiter.ops.triton.attention.pa_decode import paged_attention_decode

        q = inp["q"]
        k_u = inp["k"]
        v_u = inp["v"]
        out = torch.empty_like(q)
        k_d, v_d = cache_layout_decode(k_u, v_u)
        seq_lens = inp["seqused_k"]
        block_tables = inp["block_table"]
        max_seq_len = sig.max_seqlen_k
        k_scale = torch.tensor([1.0], dtype=torch.float32, device=q.device)
        v_scale = torch.tensor([1.0], dtype=torch.float32, device=q.device)
        compute_type = tl.bfloat16 if q.dtype == torch.bfloat16 else tl.float16

        def _call():
            paged_attention_decode(
                output=out,
                query=q,
                key_cache=k_d,
                value_cache=v_d,
                seq_lens=seq_lens,
                block_tables=block_tables,
                attn_scale=float(sig.softmax_scale),
                max_seq_len=max_seq_len,
                compute_type=compute_type,
                k_scale=k_scale,
                v_scale=v_scale,
                num_seq_partitions=0,
                alibi_slopes=None,
            )

        status = "ok"
        if sig.has_sinks and ignore_sinks:
            status = "ok:ignore_sinks_semantics"
        return run_kernel(_call, warmup, iters), status

    if backend == "triton_pa_prefill":
        if is_decode:
            return None, "skip:prefill-only"
        from aiter.ops.triton.attention.pa_prefill import context_attention_fwd

        q = inp["q"]
        # token K/V for prefill stream (not in trace): synthetic
        k_tok = torch.randn(q.shape[0], sig.k_shape[2], sig.k_shape[3], dtype=inp["k"].dtype, device=q.device)
        v_tok = torch.randn_like(k_tok)
        out = torch.empty_like(q)
        k_asm, v_asm = cache_layout_asm(inp["k"], inp["v"])
        q_lens = synth_q_lens(sig.q_shape[0], sig.num_seqs)
        b_seq_len = torch.tensor(q_lens, dtype=torch.int32, device=q.device)
        b_start_loc = torch.cat(
            [
                torch.tensor([0], dtype=torch.int32, device=q.device),
                b_seq_len.cumsum(0, dtype=torch.int32),
            ]
        )
        b_loc = inp["block_table"]
        k_scale = torch.tensor([1.0], dtype=torch.float32, device=q.device)
        v_scale = torch.tensor([1.0], dtype=torch.float32, device=q.device)
        sliding = sig.window_size[0] + 1 if sig.window_size != (-1, -1) else 0

        def _call():
            context_attention_fwd(
                q=q,
                k=k_tok,
                v=v_tok,
                o=out,
                kv_cache_dtype="auto",
                k_cache=k_asm,
                v_cache=v_asm,
                b_loc=b_loc,
                b_start_loc=b_start_loc,
                b_seq_len=b_seq_len,
                max_input_len=max(q_lens),
                k_scale=k_scale,
                v_scale=v_scale,
                alibi_slopes=None,
                sliding_window=sliding,
                sm_scale=float(sig.softmax_scale),
                skip_decode=False,
            )

        return run_kernel(_call, warmup, iters), "ok"

    if backend in {"rocm_pa_common", "asm_pa_fwd"}:
        if not decode_global_ok:
            return None, "skip:decode-global-no-sinks-only(use --ignore-sinks)"
        try:
            from aiter.ops import attention as attn_ops
        except Exception as e:  # pragma: no cover
            return None, f"skip:import:{e}"

        q = inp["q"]
        out = torch.empty_like(q)
        k_asm, v_asm = cache_layout_asm(inp["k"], inp["v"])
        seq_lens = inp["seqused_k"]
        block_tables = inp["block_table"]

        if backend == "asm_pa_fwd":
            def _call():
                attn_ops.pa_fwd_asm(
                    Q=q,
                    K=k_asm,
                    V=v_asm,
                    block_tables=block_tables,
                    context_lens=seq_lens,
                    block_tables_stride0=block_tables.stride(0),
                    max_qlen=1,
                    out_=out,
                    # hp=1 can fail heuristic selection for some traced configs.
                    high_precision=0,
                )

            status = "ok"
            if sig.has_sinks and ignore_sinks:
                status = "ok:ignore_sinks_semantics"
            return run_kernel(_call, warmup, iters), status

        # rocm_pa_common path (may internally choose asm/rocm)
        exp_sums = torch.empty((q.shape[0], q.shape[1], math.ceil(sig.max_seqlen_k / 256)), dtype=torch.float32, device=q.device)
        max_logits = torch.empty_like(exp_sums)
        tmp_out = torch.empty((q.shape[0], q.shape[1], exp_sums.shape[-1], q.shape[2]), dtype=torch.float32, device=q.device)

        def _call():
            attn_ops.paged_attention_common(
                Q=q,
                K=k_asm,
                V=v_asm,
                exp_sums=exp_sums,
                max_logits=max_logits,
                tmp_out=tmp_out,
                block_tables=block_tables,
                context_lens=seq_lens,
                block_tables_stride0=block_tables.stride(0),
                scale=float(sig.softmax_scale),
                max_qlen=1,
                max_seq_len=sig.max_seqlen_k,
                out_=out,
                kv_cache_dtype="auto",
                kv_cache_tensor_dtype=inp["k"].dtype,
            )

        status = "ok"
        if sig.has_sinks and ignore_sinks:
            status = "ok:ignore_sinks_semantics"
        return run_kernel(_call, warmup, iters), status

    return None, "skip:unknown-backend"


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare attention backends on JSONL shapes.")
    parser.add_argument("--jsonl", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--max-shapes", type=int, default=0, help="0 means all")
    parser.add_argument(
        "--backends",
        type=str,
        default=",".join(BACKEND_LIST),
        help=f"Comma-separated subset of: {','.join(BACKEND_LIST)}",
    )
    parser.add_argument(
        "--all-rows",
        action="store_true",
        help="Replay every row (no signature dedupe).",
    )
    parser.add_argument(
        "--ignore-sinks",
        action="store_true",
        help="Allow non-sink backends on sink shapes (semantics mismatch; perf-only).",
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=None,
        help="Optional CSV path to store per-result rows.",
    )
    parser.add_argument(
        "--out-summary-json",
        type=Path,
        default=None,
        help="Optional JSON path to store aggregated summaries.",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA is required.")
        return 2

    chosen_backends = [b.strip() for b in args.backends.split(",") if b.strip()]
    for b in chosen_backends:
        if b not in BACKEND_LIST:
            raise ValueError(f"Unknown backend: {b}")

    cases = read_cases(args.jsonl, dedupe=not args.all_rows, max_shapes=args.max_shapes)
    print(f"cases={len(cases)} backends={chosen_backends}")
    print("idx count observed max_q max_k window q_shape                          backend            ms    status")
    print("-" * 132)
    results: list[dict[str, Any]] = []

    for i, case in enumerate(cases):
        s = case.sig
        for backend in chosen_backends:
            try:
                ms, status = bench_backend(
                    case,
                    backend,
                    warmup=args.warmup,
                    iters=args.iters,
                    ignore_sinks=args.ignore_sinks,
                )
            except Exception as e:
                ms = None
                status = f"error:{type(e).__name__}:{str(e).splitlines()[0][:120]}"
            ms_str = f"{ms:8.3f}" if ms is not None else "   n/a  "
            print(
                f"{i:3d} {case.count:5d} {case.observed_kind:7s} "
                f"{s.max_seqlen_q:5d} {s.max_seqlen_k:5d} {str(s.window_size):>12s} "
                f"{str(s.q_shape):>28s} {backend:18s} {ms_str}  {status}"
            )
            results.append(
                {
                    "idx": i,
                    "count": case.count,
                    "observed_kind": case.observed_kind,
                    "max_seqlen_q": s.max_seqlen_q,
                    "max_seqlen_k": s.max_seqlen_k,
                    "window_size": list(s.window_size),
                    "q_shape": list(s.q_shape),
                    "k_shape": list(s.k_shape),
                    "block_table_shape": list(s.block_table_shape),
                    "num_seqs": s.num_seqs,
                    "q_dtype": s.q_dtype,
                    "k_dtype": s.k_dtype,
                    "softmax_scale": s.softmax_scale,
                    "softcap": s.softcap,
                    "has_sinks": s.has_sinks,
                    "backend": backend,
                    "ms": ms,
                    "status": status,
                    "is_decode": s.max_seqlen_q == 1,
                    "is_global": s.window_size == (-1, -1),
                }
            )

    # Aggregations (success-only). Treat "ok:*" as success, too.
    def _is_success(status: str) -> bool:
        return status == "ok" or status.startswith("ok:")

    by_backend: dict[str, dict[str, Any]] = {}
    for backend in chosen_backends:
        rows_ok = [
            r
            for r in results
            if r["backend"] == backend and _is_success(r["status"]) and r["ms"] is not None
        ]
        rows_all = [r for r in results if r["backend"] == backend]
        weighted_den = sum(int(r["count"]) for r in rows_ok)
        weighted_num = sum(float(r["ms"]) * int(r["count"]) for r in rows_ok)
        by_backend[backend] = {
            "ok_rows": len(rows_ok),
            "total_rows": len(rows_all),
            "avg_ms_unweighted": (sum(float(r["ms"]) for r in rows_ok) / len(rows_ok)) if rows_ok else None,
            "avg_ms_weighted_by_trace_count": (weighted_num / weighted_den) if weighted_den > 0 else None,
        }

    # Slice-level aggregates to inspect phase behavior.
    def _slice_stats(name: str, pred) -> dict[str, Any]:
        out: dict[str, Any] = {"name": name, "by_backend": {}}
        for backend in chosen_backends:
            rows = [
                r
                for r in results
                if r["backend"] == backend and _is_success(r["status"]) and r["ms"] is not None and pred(r)
            ]
            if not rows:
                out["by_backend"][backend] = {"ok_rows": 0, "avg_ms_weighted_by_trace_count": None}
                continue
            den = sum(int(r["count"]) for r in rows)
            num = sum(float(r["ms"]) * int(r["count"]) for r in rows)
            out["by_backend"][backend] = {
                "ok_rows": len(rows),
                "avg_ms_weighted_by_trace_count": (num / den) if den > 0 else None,
            }
        return out

    slice_aggs = [
        _slice_stats("prefill", lambda r: not r["is_decode"]),
        _slice_stats("decode_global", lambda r: r["is_decode"] and r["is_global"]),
        _slice_stats("decode_windowed", lambda r: r["is_decode"] and not r["is_global"]),
    ]

    print("\n=== Aggregate by backend (ok rows only) ===")
    for backend in chosen_backends:
        agg = by_backend[backend]
        print(
            f"{backend:18s} ok={agg['ok_rows']:4d}/{agg['total_rows']:4d} "
            f"avg_ms={agg['avg_ms_unweighted'] if agg['avg_ms_unweighted'] is not None else 'n/a'} "
            f"weighted_ms={agg['avg_ms_weighted_by_trace_count'] if agg['avg_ms_weighted_by_trace_count'] is not None else 'n/a'}"
        )

    if args.out_csv is not None:
        args.out_csv.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "idx",
            "count",
            "observed_kind",
            "max_seqlen_q",
            "max_seqlen_k",
            "window_size",
            "q_shape",
            "k_shape",
            "block_table_shape",
            "num_seqs",
            "q_dtype",
            "k_dtype",
            "softmax_scale",
            "softcap",
            "has_sinks",
            "backend",
            "ms",
            "status",
            "is_decode",
            "is_global",
        ]
        with args.out_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        print(f"Wrote per-row results: {args.out_csv}")

    if args.out_summary_json is not None:
        args.out_summary_json.parent.mkdir(parents=True, exist_ok=True)
        summary = {
            "num_cases": len(cases),
            "backends": chosen_backends,
            "aggregate_by_backend": by_backend,
            "aggregate_by_slice": slice_aggs,
        }
        with args.out_summary_json.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"Wrote summary: {args.out_summary_json}")

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
