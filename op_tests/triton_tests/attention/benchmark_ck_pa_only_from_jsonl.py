#!/usr/bin/env python3
# SPDX-License-Identifier: MIT

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import aiter


DTYPE_MAP = {
    "torch.float16": torch.float16,
    "torch.bfloat16": torch.bfloat16,
    "torch.float32": torch.float32,
}


@dataclass(frozen=True)
class Case:
    idx: int
    count: int
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
    has_sinks: bool


def parse_dtype(name: str) -> torch.dtype:
    if name not in DTYPE_MAP:
        raise ValueError(f"Unsupported dtype for CK benchmark: {name}")
    return DTYPE_MAP[name]


def load_cases(path: Path, max_cases: int, ignore_sinks: bool) -> list[Case]:
    out: list[Case] = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            # CK decode naive scope in this benchmark:
            # decode + global (+ optional ignore sinks).
            if int(r["max_seqlen_q"]) != 1:
                continue
            if tuple(r["window_size"]) != (-1, -1):
                continue
            if bool(r.get("has_sinks", False)) and not ignore_sinks:
                continue
            out.append(
                Case(
                    idx=i,
                    count=1,
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


def prepare_case(case: Case) -> dict[str, Any]:
    q_dtype = parse_dtype(case.q_dtype)
    kv_dtype = parse_dtype(case.k_dtype)
    q = torch.randn(*case.q_shape, dtype=q_dtype, device="cuda")

    # Compact cache to avoid huge physical block allocation from trace.
    num_blocks_trace, block_size, num_kv_heads, head_size = case.k_shape
    needed_blocks_per_seq = (case.max_k + block_size - 1) // block_size
    compact_blocks = max(needed_blocks_per_seq * max(case.num_seqs, 1) * 2, 64)
    compact_blocks = min(compact_blocks, num_blocks_trace)

    k_u = torch.randn(
        compact_blocks, block_size, num_kv_heads, head_size, dtype=kv_dtype, device="cuda"
    )
    v_u = torch.randn_like(k_u)

    # Convert to CK layout:
    # K: [num_blocks, num_kv_heads, head_size/x, block_size, x]
    # V: [num_blocks, num_kv_heads, head_size, block_size]
    elem = torch.tensor([], dtype=kv_dtype).element_size()
    x = min(head_size, max(1, 16 // elem))
    if head_size % x != 0:
        raise RuntimeError(f"head_size={head_size} not divisible by vector size x={x}")

    k = k_u.permute(0, 2, 3, 1).contiguous()
    k = k.view(compact_blocks, num_kv_heads, head_size // x, x, block_size)
    k = k.permute(0, 1, 2, 4, 3).contiguous()
    v = v_u.permute(0, 2, 3, 1).contiguous()

    rows, cols = case.block_table_shape
    if rows != case.num_seqs:
        rows = case.num_seqs
    block_tables = torch.randint(0, compact_blocks, (rows, cols), dtype=torch.int32, device="cuda")
    q_lens = synth_q_lens(case.q_shape[0], case.num_seqs)
    context_lens = torch.full((case.num_seqs,), case.max_k, dtype=torch.int32, device="cuda")
    if sum(q_lens) != case.q_shape[0]:
        raise RuntimeError("q lens synthesis mismatch")

    out = torch.empty_like(q)
    k_dequant_scales = torch.empty((0,), dtype=torch.float32, device="cuda")
    v_dequant_scales = torch.empty((0,), dtype=torch.float32, device="cuda")
    return {
        "q": q,
        "k": k,
        "v": v,
        "block_tables": block_tables,
        "context_lens": context_lens,
        "k_dequant_scales": k_dequant_scales,
        "v_dequant_scales": v_dequant_scales,
        "max_seq_len": case.max_k,
        "num_kv_heads": case.k_shape[2],
        "scale_s": case.softmax_scale,
        "scale_k": 1.0,
        "scale_v": 1.0,
        "block_size": case.k_shape[1],
        "quant_algo": 0,
        "out": out,
        "compact_blocks": compact_blocks,
    }


def run_ck(inp: dict[str, Any]) -> None:
    aiter.pa_fwd_naive(
        inp["q"],
        inp["k"],
        inp["v"],
        inp["block_tables"],
        inp["context_lens"],
        inp["k_dequant_scales"],
        inp["v_dequant_scales"],
        inp["max_seq_len"],
        inp["num_kv_heads"],
        inp["scale_s"],
        inp["scale_k"],
        inp["scale_v"],
        inp["block_size"],
        inp["quant_algo"],
        inp["out"],
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="CK-only decode/global benchmark from JSONL shapes.")
    ap.add_argument("--jsonl", type=Path, required=True)
    ap.add_argument("--max-cases", type=int, default=50, help="0 means all eligible cases")
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--ignore-sinks", action="store_true", help="Allow sinks rows as perf-only.")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("CUDA/HIP device required.")
        return 2

    cases = load_cases(args.jsonl, args.max_cases, ignore_sinks=args.ignore_sinks)
    print(f"eligible_ck_cases={len(cases)} (decode+global, ignore_sinks={args.ignore_sinks})")
    if not cases:
        print("No eligible cases. Try --ignore-sinks.")
        return 1

    # First call includes potential JIT compile overhead.
    first = prepare_case(cases[0])
    t0 = time.perf_counter()
    run_ck(first)
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    print(f"first_call_ms_including_jit={(t1 - t0) * 1e3:.3f}")

    per_case = []
    for i, c in enumerate(cases):
        inp = prepare_case(c)
        for _ in range(args.warmup):
            run_ck(inp)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(args.iters):
            run_ck(inp)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        ms = (t1 - t0) * 1e3 / args.iters
        per_case.append(ms)
        print(
            f"case={i:3d} trace_row={c.idx:4d} q={c.q_shape[0]:4d} max_k={c.max_k:4d} "
            f"compact_blocks={inp['compact_blocks']:4d} ms={ms:.4f}"
        )

    avg_ms = sum(per_case) / len(per_case)
    print(f"steady_avg_ms={avg_ms:.4f} over {len(per_case)} cases")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
