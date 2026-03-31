#!/usr/bin/env python3
# SPDX-License-Identifier: MIT

"""
Realistic microbenchmark: unified attention 2D vs 3D on JSONL shapes.

Key differences from the earlier compact-cache version:
- Uses the FULL physical cache size from the trace (not compacted to 64 blocks).
- Block table indices span the full physical block range (realistic TLB/cache pressure).
- Supports --num-blocks-override to test specific cache sizes.
- Reports per-case winner with phase breakdown.
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

from aiter.ops.triton.attention import unified_attention as ua_mod


DTYPE_MAP = {
    "torch.float16": torch.float16,
    "torch.bfloat16": torch.bfloat16,
    "torch.float32": torch.float32,
}


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


def make_inputs(
    case: Case, seed: int, ignore_sinks: bool, num_blocks_override: int
) -> dict[str, Any]:
    torch.manual_seed(seed)
    q_dtype = parse_dtype(case.q_dtype)
    kv_dtype = parse_dtype(case.k_dtype)

    q = torch.randn(*case.q_shape, dtype=q_dtype, device="cuda")

    trace_blocks, block_size, num_kv_heads, head_size = case.k_shape

    if num_blocks_override > 0:
        num_blocks = num_blocks_override
    else:
        num_blocks = trace_blocks

    k = torch.randn(
        num_blocks, block_size, num_kv_heads, head_size, dtype=kv_dtype, device="cuda"
    )
    v = torch.randn_like(k)

    q_lens = synth_q_lens(case.q_shape[0], case.num_seqs)
    cu = [0]
    for x in q_lens:
        cu.append(cu[-1] + x)
    cu_q = torch.tensor(cu, dtype=torch.int32, device="cuda")
    seq_lens = torch.full(
        (case.num_seqs,), case.max_k, dtype=torch.int32, device="cuda"
    )

    rows, cols = case.block_table_shape
    if rows != case.num_seqs:
        rows = case.num_seqs
    block_tables = torch.randint(
        0, num_blocks, (rows, cols), dtype=torch.int32, device="cuda"
    )

    sinks = None
    if case.has_sinks and not ignore_sinks:
        sinks = torch.randn(case.q_shape[1], dtype=torch.bfloat16, device="cuda")

    return {
        "q": q,
        "k": k,
        "v": v,
        "out": torch.empty_like(q),
        "cu_seqlens_q": cu_q,
        "max_seqlen_q": case.max_q,
        "seqused_k": seq_lens,
        "max_seqlen_k": case.max_k,
        "softmax_scale": case.softmax_scale,
        "causal": True,
        "window_size": case.window_size,
        "block_table": block_tables,
        "softcap": case.softcap,
        "q_descale": None,
        "k_descale": None,
        "v_descale": None,
        "alibi_slopes": None,
        "output_scale": None,
        "qq_bias": None,
        "sinks": sinks,
        "_num_blocks": num_blocks,
    }


def bench_case(
    inp: dict[str, Any], force_2d: bool, warmup: int, iters: int
) -> float:
    kw = {k: v for k, v in inp.items() if not k.startswith("_")}
    old = ua_mod.use_2d_kernel
    ua_mod.use_2d_kernel = (
        (lambda *a, **kw: True) if force_2d else (lambda *a, **kw: False)
    )
    try:
        for _ in range(warmup):
            ua_mod.unified_attention(**kw)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            ua_mod.unified_attention(**kw)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        return (t1 - t0) * 1e3 / iters
    finally:
        ua_mod.use_2d_kernel = old


def phase_name(max_q: int, window_size: tuple[int, int]) -> str:
    if max_q == 1:
        return "decode_windowed" if window_size != (-1, -1) else "decode_global"
    return "prefill"


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Realistic unified 2D vs 3D microbenchmark from JSONL shapes."
    )
    ap.add_argument("--jsonl", type=Path, required=True)
    ap.add_argument("--max-cases", type=int, default=0, help="0 means all rows.")
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--ignore-sinks", action="store_true")
    ap.add_argument("--tie-eps-ms", type=float, default=1e-3)
    ap.add_argument(
        "--num-blocks-override",
        type=int,
        default=0,
        help=(
            "Override num physical cache blocks. "
            "0 = use trace's actual k_shape[0] (realistic). "
            "Positive value = use that many blocks for all cases."
        ),
    )
    ap.add_argument(
        "--out-csv",
        type=Path,
        default=Path("/workspaces/workspace/unified_2d_vs_3d_realistic.csv"),
    )
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("CUDA/HIP device required.")
        return 2

    cases = load_cases(args.jsonl, args.max_cases)
    if not cases:
        print("No cases found.")
        return 1

    # Deduplicate: many JSONL rows share the same physical cache size.
    # Allocate KV once per unique (num_blocks, block_size, num_kv_heads, head_size).
    # But each case still has unique q/block_table/seq_lens.

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
        "num_blocks",
        "ms_2d",
        "ms_3d",
        "delta_ms_3d_minus_2d",
        "winner",
        "status",
    ]

    winners = {"2d": 0, "3d": 0, "tie": 0}
    winners_by_phase: dict[str, dict[str, int]] = {}

    for i, c in enumerate(cases):
        phase = phase_name(c.max_q, c.window_size)
        winners_by_phase.setdefault(phase, {"2d": 0, "3d": 0, "tie": 0})
        is_decode = c.max_q == 1

        try:
            inp = make_inputs(
                c, seed=123 + i, ignore_sinks=args.ignore_sinks,
                num_blocks_override=args.num_blocks_override,
            )
            num_blocks = inp["_num_blocks"]

            ms2 = bench_case(inp, force_2d=True, warmup=args.warmup, iters=args.iters)
            ms3 = bench_case(inp, force_2d=False, warmup=args.warmup, iters=args.iters)
            delta = ms3 - ms2
            if abs(delta) <= args.tie_eps_ms:
                winner = "tie"
            else:
                winner = "2d" if ms2 < ms3 else "3d"
            winners[winner] += 1
            winners_by_phase[phase][winner] += 1
            rows.append([
                c.idx, 1, phase, str(is_decode).lower(),
                c.max_q, c.max_k, c.max_q, c.max_k,
                list(c.window_size), num_blocks,
                ms2, ms3, delta, winner, "ok",
            ])

            if (i + 1) % 50 == 0 or i == len(cases) - 1:
                print(
                    f"[{i + 1}/{len(cases)}] "
                    f"2d={winners['2d']} 3d={winners['3d']} tie={winners['tie']}"
                )

        except Exception as e:
            rows.append([
                c.idx, 1, phase, str(is_decode).lower(),
                c.max_q, c.max_k, c.max_q, c.max_k,
                list(c.window_size), 0,
                "", "", "", "", f"error:{type(e).__name__}:{str(e).splitlines()[0][:80]}",
            ])

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(hdr)
        w.writerows(rows)

    total_ok = sum(winners.values())
    print(f"\nWrote: {args.out_csv}")
    print(
        f"rows={len(cases)} ok={total_ok} "
        f"num_blocks_override={args.num_blocks_override} tie_eps_ms={args.tie_eps_ms}"
    )
    if total_ok > 0:
        print(f"winner overall: 2d={winners['2d']} 3d={winners['3d']} tie={winners['tie']}")
    print("winner by phase:")
    for phase in ["prefill", "decode_global", "decode_windowed"]:
        p = winners_by_phase.get(phase, {"2d": 0, "3d": 0, "tie": 0})
        print(f"  {phase}: 2d={p['2d']} 3d={p['3d']} tie={p['tie']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
