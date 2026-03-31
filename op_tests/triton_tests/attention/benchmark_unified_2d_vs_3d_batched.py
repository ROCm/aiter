#!/usr/bin/env python3
# SPDX-License-Identifier: MIT

"""
Unified attention 2D vs 3D microbenchmark with synthetic batched decode.

Key insight: the JSONL trace has num_seqs=1 (single-request capture), but real
vLLM serving uses continuous batching with many concurrent sequences per decode
step. This script generates synthetic batched cases to approximate that regime.

It runs:
  1) JSONL-derived cases (as-is from trace, typically num_seqs=1)
  2) Synthetic batched decode cases with configurable num_seqs, max_k ranges

Both use full physical cache size for realistic memory pressure.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
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


@dataclass
class Case:
    tag: str
    num_seqs: int
    max_q: int
    max_k: int
    num_q_heads: int
    num_kv_heads: int
    head_size: int
    block_size: int
    window_size: tuple[int, int]
    q_dtype: torch.dtype
    kv_dtype: torch.dtype
    softmax_scale: float
    softcap: float
    has_sinks: bool
    num_blocks: int


def load_jsonl_cases(path: Path, max_cases: int) -> list[Case]:
    out: list[Case] = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            q_shape = tuple(r["q_shape"])
            k_shape = tuple(r["k_shape"])
            q_dtype_str = str(r["q_dtype"])
            kv_dtype_str = str(r["k_dtype"])
            if q_dtype_str not in DTYPE_MAP or kv_dtype_str not in DTYPE_MAP:
                continue
            out.append(
                Case(
                    tag=f"jsonl_{i}",
                    num_seqs=int(r["num_seqs"]),
                    max_q=int(r["max_seqlen_q"]),
                    max_k=int(r["max_seqlen_k"]),
                    num_q_heads=q_shape[1],
                    num_kv_heads=k_shape[2],
                    head_size=q_shape[2],
                    block_size=k_shape[1],
                    window_size=tuple(r["window_size"]),
                    q_dtype=DTYPE_MAP[q_dtype_str],
                    kv_dtype=DTYPE_MAP[kv_dtype_str],
                    softmax_scale=float(r["softmax_scale"]),
                    softcap=float(r.get("softcap", 0.0)),
                    has_sinks=bool(r.get("has_sinks", False)),
                    num_blocks=k_shape[0],
                )
            )
            if max_cases > 0 and len(out) >= max_cases:
                break
    return out


def generate_batched_decode_cases(
    num_seqs_list: list[int],
    max_k_list: list[int],
    window_sizes: list[tuple[int, int]],
    num_q_heads: int = 64,
    num_kv_heads: int = 8,
    head_size: int = 64,
    block_size: int = 32,
    num_blocks: int = 32768,
    dtype: torch.dtype = torch.bfloat16,
    softmax_scale: float = 0.125,
    has_sinks: bool = True,
) -> list[Case]:
    out: list[Case] = []
    for ns in num_seqs_list:
        for mk in max_k_list:
            for ws in window_sizes:
                out.append(
                    Case(
                        tag=f"synth_bs{ns}_k{mk}_w{ws[0]}",
                        num_seqs=ns,
                        max_q=1,
                        max_k=mk,
                        num_q_heads=num_q_heads,
                        num_kv_heads=num_kv_heads,
                        head_size=head_size,
                        block_size=block_size,
                        window_size=ws,
                        q_dtype=dtype,
                        kv_dtype=dtype,
                        softmax_scale=softmax_scale,
                        softcap=0.0,
                        has_sinks=has_sinks,
                        num_blocks=num_blocks,
                    )
                )
    return out


def make_inputs(
    case: Case, seed: int, ignore_sinks: bool, num_blocks_override: int
) -> dict[str, Any]:
    torch.manual_seed(seed)

    num_blocks = num_blocks_override if num_blocks_override > 0 else case.num_blocks
    total_q_tokens = case.num_seqs * case.max_q

    q = torch.randn(
        total_q_tokens, case.num_q_heads, case.head_size,
        dtype=case.q_dtype, device="cuda",
    )
    k = torch.randn(
        num_blocks, case.block_size, case.num_kv_heads, case.head_size,
        dtype=case.kv_dtype, device="cuda",
    )
    v = torch.randn_like(k)

    # Build cu_seqlens_q for num_seqs sequences, each with max_q tokens.
    q_lens = [case.max_q] * case.num_seqs
    cu = [0]
    for x in q_lens:
        cu.append(cu[-1] + x)
    cu_q = torch.tensor(cu, dtype=torch.int32, device="cuda")

    seq_lens = torch.full((case.num_seqs,), case.max_k, dtype=torch.int32, device="cuda")

    max_blocks_per_seq = math.ceil(case.max_k / case.block_size)
    block_tables = torch.randint(
        0, num_blocks, (case.num_seqs, max_blocks_per_seq), dtype=torch.int32, device="cuda",
    )

    sinks = None
    if case.has_sinks and not ignore_sinks:
        sinks = torch.randn(case.num_q_heads, dtype=torch.bfloat16, device="cuda")

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


def bench_kernel(
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
        description="Batched unified 2D vs 3D microbenchmark."
    )
    ap.add_argument("--jsonl", type=Path, default=None, help="Optional JSONL for trace-derived cases.")
    ap.add_argument("--max-jsonl-cases", type=int, default=0, help="0 means all JSONL rows.")
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--ignore-sinks", action="store_true")
    ap.add_argument("--tie-eps-ms", type=float, default=1e-3)
    ap.add_argument("--num-blocks-override", type=int, default=0)
    ap.add_argument(
        "--synth-num-seqs",
        type=str,
        default="1,2,4,8,16,32,64,128,256",
        help="Comma-separated num_seqs values for synthetic batched decode.",
    )
    ap.add_argument(
        "--synth-max-k",
        type=str,
        default="512,1024,2048,4096,8192",
        help="Comma-separated max_k values for synthetic decode.",
    )
    ap.add_argument(
        "--synth-num-blocks",
        type=int,
        default=32768,
        help="Physical blocks for synthetic cases.",
    )
    ap.add_argument(
        "--out-csv",
        type=Path,
        default=Path("/workspaces/workspace/unified_2d_vs_3d_batched.csv"),
    )
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("CUDA/HIP device required.")
        return 2

    cases: list[Case] = []

    if args.jsonl is not None:
        jsonl_cases = load_jsonl_cases(args.jsonl, args.max_jsonl_cases)
        cases.extend(jsonl_cases)
        print(f"Loaded {len(jsonl_cases)} JSONL cases.")

    synth_num_seqs = [int(x) for x in args.synth_num_seqs.split(",") if x.strip()]
    synth_max_k = [int(x) for x in args.synth_max_k.split(",") if x.strip()]
    synth_cases = generate_batched_decode_cases(
        num_seqs_list=synth_num_seqs,
        max_k_list=synth_max_k,
        window_sizes=[(-1, -1), (127, 0)],
        num_blocks=args.synth_num_blocks,
    )
    cases.extend(synth_cases)
    print(f"Generated {len(synth_cases)} synthetic batched decode cases.")
    print(f"Total cases: {len(cases)}")

    if not cases:
        print("No cases.")
        return 1

    rows: list[list[Any]] = []
    hdr = [
        "tag",
        "phase",
        "is_decode",
        "num_seqs",
        "max_q",
        "max_k",
        "max_seqlen_q",
        "max_seqlen_k",
        "window_size",
        "num_blocks",
        "num_q_heads",
        "num_kv_heads",
        "ms_2d",
        "ms_3d",
        "delta_ms_3d_minus_2d",
        "speedup_3d_over_2d",
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

            ms2 = bench_kernel(inp, force_2d=True, warmup=args.warmup, iters=args.iters)
            ms3 = bench_kernel(inp, force_2d=False, warmup=args.warmup, iters=args.iters)
            delta = ms3 - ms2
            speedup = ms2 / ms3 if ms3 > 0 else float("inf")
            if abs(delta) <= args.tie_eps_ms:
                winner = "tie"
            else:
                winner = "2d" if ms2 < ms3 else "3d"
            winners[winner] += 1
            winners_by_phase[phase][winner] += 1

            rows.append([
                c.tag, phase, str(is_decode).lower(), c.num_seqs,
                c.max_q, c.max_k, c.max_q, c.max_k,
                list(c.window_size), num_blocks,
                c.num_q_heads, c.num_kv_heads,
                ms2, ms3, delta, speedup, winner, "ok",
            ])

            print(
                f"[{i + 1}/{len(cases)}] {c.tag:40s} "
                f"ns={c.num_seqs:4d} max_k={c.max_k:5d} "
                f"2d={ms2:.4f} 3d={ms3:.4f} "
                f"winner={winner} speedup_3d={speedup:.3f}x"
            )

        except Exception as e:
            rows.append([
                c.tag, phase, str(is_decode).lower(), c.num_seqs,
                c.max_q, c.max_k, c.max_q, c.max_k,
                list(c.window_size), 0,
                c.num_q_heads, c.num_kv_heads,
                "", "", "", "", "",
                f"error:{type(e).__name__}:{str(e).splitlines()[0][:80]}",
            ])
            print(f"[{i + 1}/{len(cases)}] {c.tag:40s} ERROR: {e}")

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(hdr)
        w.writerows(rows)

    total_ok = sum(winners.values())
    print(f"\nWrote: {args.out_csv}")
    print(f"total_cases={len(cases)} ok={total_ok} tie_eps_ms={args.tie_eps_ms}")
    if total_ok > 0:
        print(f"winner overall: 2d={winners['2d']} 3d={winners['3d']} tie={winners['tie']}")
    print("winner by phase:")
    for phase in ["prefill", "decode_global", "decode_windowed"]:
        p = winners_by_phase.get(phase, {"2d": 0, "3d": 0, "tie": 0})
        print(f"  {phase}: 2d={p['2d']} 3d={p['3d']} tie={p['tie']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
