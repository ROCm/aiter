#!/usr/bin/env python3
# SPDX-License-Identifier: MIT

"""
Compare output equivalence across attention backends on a few trace-derived cases.

Default scope is decode+global rows from unified-attention JSONL, since those are
the rows where CK decode and Triton decode are comparable to unified attention.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import triton.language as tl

import aiter
from aiter.ops.triton.attention import unified_attention as ua_mod
from aiter.ops.triton.attention.pa_decode import paged_attention_decode


DTYPE_MAP: dict[str, torch.dtype] = {
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
    max_seqlen_q: int
    max_seqlen_k: int
    window_size: tuple[int, int]
    num_seqs: int
    q_dtype: str
    k_dtype: str
    softmax_scale: float
    softcap: float
    has_sinks: bool


def parse_dtype(name: str) -> torch.dtype:
    if name not in DTYPE_MAP:
        raise ValueError(f"Unsupported dtype in this comparator: {name}")
    return DTYPE_MAP[name]


def load_cases(path: Path, num_cases: int, ignore_sinks: bool) -> list[Case]:
    cases: list[Case] = []
    seen: set[tuple[Any, ...]] = set()
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            # Focus on decode+global, where CK/triton decode kernels are comparable.
            if int(r["max_seqlen_q"]) != 1:
                continue
            if tuple(r["window_size"]) != (-1, -1):
                continue
            if bool(r.get("has_sinks", False)) and not ignore_sinks:
                continue
            key = (
                tuple(r["q_shape"]),
                tuple(r["k_shape"]),
                tuple(r["block_table_shape"]),
                int(r["max_seqlen_k"]),
                int(r["num_seqs"]),
                str(r["q_dtype"]),
                str(r["k_dtype"]),
                float(r["softmax_scale"]),
                float(r.get("softcap", 0.0)),
                bool(r.get("has_sinks", False)),
            )
            if key in seen:
                continue
            seen.add(key)
            cases.append(
                Case(
                    idx=i,
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
            )
            if len(cases) >= num_cases:
                break
    return cases


def synth_q_lens(total_tokens: int, num_seqs: int) -> list[int]:
    base = total_tokens // num_seqs
    rem = total_tokens % num_seqs
    return [base + (1 if i < rem else 0) for i in range(num_seqs)]


def make_inputs(case: Case, seed: int, ignore_sinks: bool) -> tuple[dict[str, Any], dict[str, int]]:
    random.seed(seed)
    torch.manual_seed(seed)

    q_dtype = parse_dtype(case.q_dtype)
    kv_dtype = parse_dtype(case.k_dtype)

    q = torch.randn(*case.q_shape, dtype=q_dtype, device="cuda")

    # Use a compact cache instead of trace's full num_blocks.
    # Trace often logs very large physical caches (e.g. 179896 blocks), but
    # decode for max_k~1k only needs ~32 blocks. Compacting avoids ROCm faults.
    trace_num_blocks, block_size, num_kv_heads, head_size = case.k_shape
    needed_blocks_per_seq = (case.max_seqlen_k + block_size - 1) // block_size
    compact_num_blocks = max(needed_blocks_per_seq * max(case.num_seqs, 1) * 2, 64)
    compact_num_blocks = min(compact_num_blocks, trace_num_blocks)
    k_shape_compact = (compact_num_blocks, block_size, num_kv_heads, head_size)

    k = torch.randn(*k_shape_compact, dtype=kv_dtype, device="cuda")
    v = torch.randn(*k_shape_compact, dtype=kv_dtype, device="cuda")
    out = torch.empty_like(q)

    q_lens = synth_q_lens(case.q_shape[0], case.num_seqs)
    cu = [0]
    for x in q_lens:
        cu.append(cu[-1] + x)
    cu_seqlens_q = torch.tensor(cu, dtype=torch.int32, device="cuda")
    seqused_k = torch.full((case.num_seqs,), case.max_seqlen_k, dtype=torch.int32, device="cuda")

    rows, cols = case.block_table_shape
    if rows != case.num_seqs:
        rows = case.num_seqs
    block_table = torch.randint(
        0, compact_num_blocks, (rows, cols), dtype=torch.int32, device="cuda"
    )

    sinks = None
    if case.has_sinks and not ignore_sinks:
        sinks = torch.randn(case.q_shape[1], dtype=torch.bfloat16, device="cuda")

    inp = {
        "q": q,
        "k": k,
        "v": v,
        "out": out,
        "cu_seqlens_q": cu_seqlens_q,
        "max_seqlen_q": case.max_seqlen_q,
        "seqused_k": seqused_k,
        "max_seqlen_k": case.max_seqlen_k,
        "softmax_scale": case.softmax_scale,
        "causal": True,
        "window_size": case.window_size,
        "block_table": block_table,
        "softcap": case.softcap,
        "q_descale": None,
        "k_descale": None,
        "v_descale": None,
        "alibi_slopes": None,
        "output_scale": None,
        "qq_bias": None,
        "sinks": sinks,
    }
    meta = {
        "trace_num_blocks": trace_num_blocks,
        "compact_num_blocks": compact_num_blocks,
    }
    return inp, meta


def cache_layout_decode(k_u: torch.Tensor, v_u: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return k_u.permute(0, 2, 1, 3).contiguous(), v_u.permute(0, 2, 1, 3).contiguous()


def cache_layout_asm(k_u: torch.Tensor, v_u: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    nb, blk, kvh, hd = k_u.shape
    elem = torch.tensor([], dtype=k_u.dtype).element_size()
    x = min(hd, max(1, 16 // elem))
    if hd % x != 0:
        raise RuntimeError(f"head_size={hd} not divisible by x={x}")
    k = k_u.permute(0, 2, 3, 1).contiguous()  # [nb, kvh, hd, blk]
    k = k.view(nb, kvh, hd // x, x, blk).permute(0, 1, 2, 4, 3).contiguous()
    v = v_u.permute(0, 2, 3, 1).contiguous()
    return k, v


@contextlib.contextmanager
def force_unified(mode: str):
    old = ua_mod.use_2d_kernel
    if mode == "default":
        yield
        return
    ua_mod.use_2d_kernel = (lambda *args, **kwargs: True) if mode == "2d" else (lambda *args, **kwargs: False)
    try:
        yield
    finally:
        ua_mod.use_2d_kernel = old


def run_unified(inp: dict[str, Any], mode: str) -> torch.Tensor:
    out = torch.empty_like(inp["q"])
    kw = dict(inp)
    kw["out"] = out
    with force_unified(mode):
        ua_mod.unified_attention(**kw)
    return out


def run_triton_pa_decode(inp: dict[str, Any], softmax_scale: float) -> torch.Tensor:
    q = inp["q"]
    out = torch.empty_like(q)
    k_d, v_d = cache_layout_decode(inp["k"], inp["v"])
    seq_lens = inp["seqused_k"]
    block_tables = inp["block_table"]
    compute_type = tl.bfloat16 if q.dtype == torch.bfloat16 else tl.float16
    k_scale = torch.tensor([1.0], dtype=torch.float32, device=q.device)
    v_scale = torch.tensor([1.0], dtype=torch.float32, device=q.device)
    paged_attention_decode(
        output=out,
        query=q,
        key_cache=k_d,
        value_cache=v_d,
        seq_lens=seq_lens,
        block_tables=block_tables,
        attn_scale=float(softmax_scale),
        max_seq_len=int(inp["max_seqlen_k"]),
        compute_type=compute_type,
        k_scale=k_scale,
        v_scale=v_scale,
        num_seq_partitions=0,
        alibi_slopes=None,
    )
    return out


def run_ck_pa_naive(inp: dict[str, Any], softmax_scale: float) -> torch.Tensor:
    q = inp["q"]
    k_asm, v_asm = cache_layout_asm(inp["k"], inp["v"])
    out = torch.empty_like(q)
    k_dequant_scales = torch.empty((0,), dtype=torch.float32, device=q.device)
    v_dequant_scales = torch.empty((0,), dtype=torch.float32, device=q.device)
    aiter.pa_fwd_naive(
        q,
        k_asm,
        v_asm,
        inp["block_table"],
        inp["seqused_k"],
        k_dequant_scales,
        v_dequant_scales,
        int(inp["max_seqlen_k"]),
        int(inp["k"].shape[2]),  # num_kv_heads
        float(softmax_scale),
        1.0,
        1.0,
        int(inp["k"].shape[1]),  # block_size
        0,  # quant_algo NO
        out,
    )
    return out


def compare(name: str, ref: torch.Tensor, other: torch.Tensor, atol: float, rtol: float) -> tuple[bool, float, float]:
    diff = (other.float() - ref.float()).abs()
    max_abs = diff.max().item()
    mean_abs = diff.mean().item()
    ok = bool(torch.allclose(other.float(), ref.float(), atol=atol, rtol=rtol))
    print(
        f"  {name:20s} allclose={str(ok):5s} "
        f"max_abs={max_abs:.6f} mean_abs={mean_abs:.6f}"
    )
    return ok, max_abs, mean_abs


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare attention outputs across backends.")
    parser.add_argument("--jsonl", type=Path, required=True)
    parser.add_argument("--num-cases", type=int, default=5)
    parser.add_argument("--atol", type=float, default=2e-2)
    parser.add_argument("--rtol", type=float, default=2e-2)
    parser.add_argument(
        "--ignore-sinks",
        action="store_true",
        help="Drop sinks even if trace has sinks=True so decode backends are comparable.",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA/HIP device required.")
        return 2

    cases = load_cases(args.jsonl, num_cases=args.num_cases, ignore_sinks=args.ignore_sinks)
    if not cases:
        print("No eligible cases found. Try --ignore-sinks.")
        return 1

    print(f"Comparing {len(cases)} case(s) with atol={args.atol} rtol={args.rtol}")
    print("Reference backend: unified_default\n")

    total_checks = 0
    total_pass = 0
    for ci, case in enumerate(cases):
        print(
            f"Case {ci} (trace_row={case.idx}): "
            f"q={case.q_shape} k={case.k_shape} max_q={case.max_seqlen_q} "
            f"max_k={case.max_seqlen_k} window={case.window_size} sinks={case.has_sinks}"
        )
        inp, meta = make_inputs(case, seed=123 + ci, ignore_sinks=args.ignore_sinks)
        print(
            f"  cache blocks: trace={meta['trace_num_blocks']} compact={meta['compact_num_blocks']}"
        )
        torch.cuda.synchronize()

        ref = run_unified(inp, mode="default")
        checks = [
            ("unified_force_2d", lambda: run_unified(inp, mode="2d")),
            ("unified_force_3d", lambda: run_unified(inp, mode="3d")),
            ("triton_pa_decode", lambda: run_triton_pa_decode(inp, case.softmax_scale)),
            ("ck_pa_naive", lambda: run_ck_pa_naive(inp, case.softmax_scale)),
        ]
        for name, fn in checks:
            total_checks += 1
            try:
                out = fn()
                ok, _, _ = compare(name, ref, out, atol=args.atol, rtol=args.rtol)
                total_pass += int(ok)
            except Exception as e:
                print(f"  {name:20s} ERROR: {type(e).__name__}: {str(e).splitlines()[0]}")
        print("")

    print(f"Done: {total_pass}/{total_checks} comparisons passed allclose.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
