#!/usr/bin/env python3
# SPDX-License-Identifier: MIT

"""
More realistic backend benchmark:
- no per-call KV layout conversion
- each backend receives KV cache tensors in its native layout

This avoids unrealistically timing layout conversion in the hot path.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import triton.language as tl

import aiter
from aiter.ops.triton.attention import unified_attention as ua_mod
from aiter.ops.triton.attention.pa_decode import paged_attention_decode


DTYPE_MAP = {
    "torch.float16": torch.float16,
    "torch.bfloat16": torch.bfloat16,
    "torch.float32": torch.float32,
}


def is_ok(status: str) -> bool:
    return status == "ok" or status.startswith("ok:")


@dataclass(frozen=True)
class Case:
    idx: int
    q_shape: tuple[int, int, int]
    k_shape: tuple[int, int, int, int]
    block_table_shape: tuple[int, int]
    max_q: int
    max_k: int
    window_size: tuple[int, int]
    num_seqs: int
    q_dtype: str
    k_dtype: str
    softmax_scale: float
    softcap: float
    has_sinks: bool


def parse_dtype(name: str) -> torch.dtype:
    if name not in DTYPE_MAP:
        raise ValueError(f"Unsupported dtype: {name}")
    return DTYPE_MAP[name]


def load_cases(path: Path, max_cases: int) -> list[Case]:
    out: list[Case] = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            out.append(
                Case(
                    idx=i,
                    q_shape=tuple(r["q_shape"]),
                    k_shape=tuple(r["k_shape"]),
                    block_table_shape=tuple(r["block_table_shape"]),
                    max_q=int(r["max_seqlen_q"]),
                    max_k=int(r["max_seqlen_k"]),
                    window_size=tuple(r["window_size"]),
                    num_seqs=int(r["num_seqs"]),
                    q_dtype=str(r["q_dtype"]),
                    k_dtype=str(r["k_dtype"]),
                    softmax_scale=float(r["softmax_scale"]),
                    softcap=float(r.get("softcap", 0.0)),
                    has_sinks=bool(r.get("has_sinks", False)),
                )
            )
            if max_cases > 0 and len(out) >= max_cases:
                break
    return out


def synth_q_lens(total_tokens: int, num_seqs: int) -> list[int]:
    base = total_tokens // num_seqs
    rem = total_tokens % num_seqs
    return [base + (1 if i < rem else 0) for i in range(num_seqs)]


def make_common(case: Case, seed: int) -> dict[str, Any]:
    torch.manual_seed(seed)
    q_dtype = parse_dtype(case.q_dtype)
    kv_dtype = parse_dtype(case.k_dtype)

    q = torch.randn(*case.q_shape, dtype=q_dtype, device="cuda")
    rows, cols = case.block_table_shape
    if rows != case.num_seqs:
        rows = case.num_seqs
    q_lens = synth_q_lens(case.q_shape[0], case.num_seqs)
    cu = [0]
    for x in q_lens:
        cu.append(cu[-1] + x)
    cu_q = torch.tensor(cu, dtype=torch.int32, device="cuda")
    seq_lens = torch.full((case.num_seqs,), case.max_k, dtype=torch.int32, device="cuda")

    # Compact physical blocks to keep benchmark realistic and stable.
    num_blocks_trace, block_size, num_kv_heads, head_size = case.k_shape
    need_blks = (case.max_k + block_size - 1) // block_size
    compact_blocks = max(need_blks * max(case.num_seqs, 1) * 2, 64)
    compact_blocks = min(compact_blocks, num_blocks_trace)

    block_tables = torch.randint(0, compact_blocks, (rows, cols), dtype=torch.int32, device="cuda")

    return {
        "q": q,
        "q_dtype": q_dtype,
        "kv_dtype": kv_dtype,
        "cu_q": cu_q,
        "seq_lens": seq_lens,
        "block_tables": block_tables,
        "compact_blocks": compact_blocks,
        "block_size": block_size,
        "num_kv_heads": num_kv_heads,
        "head_size": head_size,
    }


def make_kv_unified(common: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
    # unified layout: [num_blocks, block_size, num_kv_heads, head]
    k = torch.randn(
        common["compact_blocks"],
        common["block_size"],
        common["num_kv_heads"],
        common["head_size"],
        dtype=common["kv_dtype"],
        device="cuda",
    )
    v = torch.randn_like(k)
    return k, v


def make_kv_decode(common: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
    # decode layout: [num_blocks, num_kv_heads, block_size, head]
    k = torch.randn(
        common["compact_blocks"],
        common["num_kv_heads"],
        common["block_size"],
        common["head_size"],
        dtype=common["kv_dtype"],
        device="cuda",
    )
    v = torch.randn_like(k)
    return k, v


def make_kv_ck(common: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
    # CK naive layout:
    # K [num_blocks, num_kv_heads, head/x, block_size, x]
    # V [num_blocks, num_kv_heads, head, block_size]
    hd = common["head_size"]
    elem = torch.tensor([], dtype=common["kv_dtype"]).element_size()
    x = min(hd, max(1, 16 // elem))
    if hd % x != 0:
        raise RuntimeError(f"head_size={hd} not divisible by x={x}")
    k = torch.randn(
        common["compact_blocks"],
        common["num_kv_heads"],
        hd // x,
        common["block_size"],
        x,
        dtype=common["kv_dtype"],
        device="cuda",
    )
    v = torch.randn(
        common["compact_blocks"],
        common["num_kv_heads"],
        hd,
        common["block_size"],
        dtype=common["kv_dtype"],
        device="cuda",
    )
    return k, v


def bench(fn, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    return (t1 - t0) * 1e3 / iters


def main() -> int:
    ap = argparse.ArgumentParser(description="Benchmark attention backends with native KV layouts.")
    ap.add_argument("--jsonl", type=Path, required=True)
    ap.add_argument("--max-cases", type=int, default=200, help="0 means all rows.")
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--ignore-sinks", action="store_true")
    ap.add_argument(
        "--out-csv",
        type=Path,
        default=Path("/workspaces/workspace/attn_backend_results_native_layouts.csv"),
    )
    args = ap.parse_args()

    cases = load_cases(args.jsonl, args.max_cases)
    if not cases:
        print("No cases.")
        return 1

    rows: list[list[Any]] = []
    hdr = [
        "idx",
        "count",
        "phase",
        "is_decode",
        "max_q",
        "max_k",
        "max_seqlen_q",
        "max_seqlen_k",
        "window_size",
        "backend",
        "ms",
        "status",
    ]

    for i, c in enumerate(cases):
        phase = "decode" if c.max_q == 1 else "prefill"
        common = make_common(c, seed=123 + i)
        is_decode = c.max_q == 1

        def pack_row(backend: str, ms: Any, status: str) -> list[Any]:
            # Keep compatibility with plotting scripts expecting count/max_seqlen_*/is_decode.
            return [
                c.idx,                    # idx
                1,                        # count (native benchmark rows are unweighted)
                phase,                    # phase
                str(is_decode).lower(),   # is_decode
                c.max_q,                  # max_q
                c.max_k,                  # max_k
                c.max_q,                  # max_seqlen_q
                c.max_k,                  # max_seqlen_k
                list(c.window_size),      # window_size
                backend,                  # backend
                ms,                       # ms
                status,                   # status
            ]

        # Unified backends (native unified layout)
        k_u, v_u = make_kv_unified(common)
        sinks = (
            None
            if (args.ignore_sinks and c.has_sinks)
            else (torch.randn(c.q_shape[1], dtype=torch.bfloat16, device="cuda") if c.has_sinks else None)
        )
        base_kwargs = dict(
            q=common["q"],
            k=k_u,
            v=v_u,
            out=torch.empty_like(common["q"]),
            cu_seqlens_q=common["cu_q"],
            max_seqlen_q=c.max_q,
            seqused_k=common["seq_lens"],
            max_seqlen_k=c.max_k,
            softmax_scale=c.softmax_scale,
            causal=True,
            window_size=c.window_size,
            block_table=common["block_tables"],
            softcap=c.softcap,
            q_descale=None,
            k_descale=None,
            v_descale=None,
            alibi_slopes=None,
            output_scale=None,
            qq_bias=None,
            sinks=sinks,
        )

        def run_unified_default():
            ua_mod.unified_attention(**base_kwargs)

        old = ua_mod.use_2d_kernel
        try:
            ms = bench(run_unified_default, args.warmup, args.iters)
            rows.append(pack_row("unified_default", ms, "ok"))
        except Exception as e:
            rows.append(pack_row("unified_default", "", f"error:{type(e).__name__}"))

        try:
            ua_mod.use_2d_kernel = lambda *a, **kw: True
            ms = bench(run_unified_default, args.warmup, args.iters)
            rows.append(pack_row("unified_force_2d", ms, "ok"))
        except Exception as e:
            rows.append(pack_row("unified_force_2d", "", f"error:{type(e).__name__}"))

        try:
            ua_mod.use_2d_kernel = lambda *a, **kw: False
            ms = bench(run_unified_default, args.warmup, args.iters)
            rows.append(pack_row("unified_force_3d", ms, "ok"))
        except Exception as e:
            rows.append(pack_row("unified_force_3d", "", f"error:{type(e).__name__}"))
        finally:
            ua_mod.use_2d_kernel = old

        # Decode-only backends with native layouts.
        decode_global = (c.max_q == 1 and c.window_size == (-1, -1))
        allow_decode = decode_global and (args.ignore_sinks or not c.has_sinks)

        if allow_decode:
            # Triton decode native layout
            try:
                k_d, v_d = make_kv_decode(common)
                out = torch.empty_like(common["q"])
                compute_type = tl.bfloat16 if common["q"].dtype == torch.bfloat16 else tl.float16
                k_scale = torch.tensor([1.0], dtype=torch.float32, device="cuda")
                v_scale = torch.tensor([1.0], dtype=torch.float32, device="cuda")

                def run_decode():
                    paged_attention_decode(
                        output=out,
                        query=common["q"],
                        key_cache=k_d,
                        value_cache=v_d,
                        seq_lens=common["seq_lens"],
                        block_tables=common["block_tables"],
                        attn_scale=c.softmax_scale,
                        max_seq_len=c.max_k,
                        compute_type=compute_type,
                        k_scale=k_scale,
                        v_scale=v_scale,
                        num_seq_partitions=0,
                        alibi_slopes=None,
                    )

                ms = bench(run_decode, args.warmup, args.iters)
                rows.append(pack_row("triton_pa_decode_native", ms, "ok"))
            except Exception as e:
                rows.append(pack_row("triton_pa_decode_native", "", f"error:{type(e).__name__}"))

            # CK naive native layout
            try:
                k_ck, v_ck = make_kv_ck(common)
                out = torch.empty_like(common["q"])
                ks = torch.empty((0,), dtype=torch.float32, device="cuda")
                vs = torch.empty((0,), dtype=torch.float32, device="cuda")

                def run_ck():
                    aiter.pa_fwd_naive(
                        common["q"],
                        k_ck,
                        v_ck,
                        common["block_tables"],
                        common["seq_lens"],
                        ks,
                        vs,
                        c.max_k,
                        common["num_kv_heads"],
                        c.softmax_scale,
                        1.0,
                        1.0,
                        common["block_size"],
                        0,
                        out,
                    )

                ms = bench(run_ck, args.warmup, args.iters)
                rows.append(pack_row("ck_pa_naive_native", ms, "ok"))
            except Exception as e:
                rows.append(pack_row("ck_pa_naive_native", "", f"error:{type(e).__name__}"))
        else:
            rows.append(pack_row("triton_pa_decode_native", "", "skip:decode_global_only"))
            rows.append(pack_row("ck_pa_naive_native", "", "skip:decode_global_only"))

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(hdr)
        w.writerows(rows)
    print(f"Wrote: {args.out_csv}")

    # Quick status summary.
    summary = {}
    for r in rows:
        b = r[9]
        st = r[11]
        summary.setdefault(b, {"ok": 0, "skip": 0, "error": 0})
        if is_ok(st):
            summary[b]["ok"] += 1
        elif st.startswith("skip:"):
            summary[b]["skip"] += 1
        else:
            summary[b]["error"] += 1
    print("Status summary:")
    for b, s in summary.items():
        print(f"- {b}: ok={s['ok']} skip={s['skip']} error={s['error']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
