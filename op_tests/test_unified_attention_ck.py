# SPDX-License-Identifier: MIT
# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.

"""Correctness + performance tests for CK unified attention.

Shape coverage mirrors `op_tests/triton_tests/attention/test_unified_attention.py`
(the upstream Triton FlashDecoding test), filtered to the configurations the
CK kernel actually supports:

  * head_size  ∈ {64, 128}  (Triton also covers 256; CK has no 256 instance)
  * block_size ∈ {16, 32, 64} (Triton also covers 48; CK ships ps16/ps32/ps64)
  * dtype      ∈ {bf16, fp16} for activations
  * q_dtype    ∈ {None, e4m3}  (per-tensor FP8 quantisation of Q/K/V)
  * num_heads  ∈ {(4,4), (8,1), (16,2), (64,8)}  — MHA + GQA-{2,8}

CK doesn't implement sliding-window, soft-cap, or alibi/sinks, so those
parameters from the Triton test are pinned at "off" here.

Reporting follows `op_tests/test_pa.py`:
  * `@perftest()` for per-kernel timing
  * `@benchmark()` for the per-config wrapper
  * pandas DataFrame summary at the end with both correctness and timings

Example:
  python op_tests/test_unified_attention_ck.py
  python op_tests/test_unified_attention_ck.py --quick           # smoke subset
  python op_tests/test_unified_attention_ck.py --head-size 128   # restrict
  python op_tests/test_unified_attention_ck.py --no-triton       # CK-only

  # Single-shape mode — replaces the standalone ua-test-scripts/test_single_shape.py
  # for ad-hoc correctness/perf checks; expands to one seq_lens batch of
  # `b` identical (sq, sk) sequences. `--num-blocks auto` allocates exactly
  # `b * ceil(sk / block_size)` physical blocks so block_tables hold unique
  # indices (no fake L2 reuse), matching what vLLM/SGLang-style allocators
  # produce in production.
  python op_tests/test_unified_attention_ck.py \\
      -b 64 -sq 1 -sk 128000 \\
      --num-heads 64,8 --head-size 128 --block-size 16 \\
      --dtype bf16 --q-dtype none --num-blocks auto
"""

from __future__ import annotations

import argparse
import gc
from dataclasses import dataclass
from typing import List, Optional, Tuple

import pandas as pd
import torch

import aiter
from aiter import dtypes
from aiter.test_common import benchmark, checkAllclose, perftest


# ----------------------------------------------------------------------------
# CK FP8 dtype: gfx950 uses OCP e4m3fn, gfx94x uses e4m3fnuz. Mirror what the
# CK kernel resolves at compile time (CK_TILE_USE_OCP_FP8) so host-side
# quantisation stays in lock-step with what the kernel consumes.
# ----------------------------------------------------------------------------
def _pick_fp8_dtype() -> torch.dtype:
    try:
        arch = torch.cuda.get_device_properties(0).gcnArchName
    except Exception:
        arch = ""
    return torch.float8_e4m3fn if "gfx950" in arch else torch.float8_e4m3fnuz


# ----------------------------------------------------------------------------
# Config grid — same dimensions as the upstream Triton test, filtered for
# CK support. Set via argparse or override at module level for ad-hoc runs.
# ----------------------------------------------------------------------------
# Default grid: ~48 configs, completes in a few minutes. Keeps every
# dimension that exercises a distinct CK code path (head-size tier,
# block-size instance, FP8 vs full-precision, prefill+decode mix vs
# all-decode), at the smallest grid that touches each.
DEFAULT_NUM_HEADS: List[Tuple[int, int]] = [
    (16, 2),   # GQA-8 (8 query heads per kv head)
    (64, 8),   # GQA-8 at the production shape
]
DEFAULT_HEAD_SIZES: List[int] = [64, 128]
DEFAULT_BLOCK_SIZES: List[int] = [16, 32, 64]
DEFAULT_DTYPES: List[torch.dtype] = [dtypes.bf16, torch.float16]
DEFAULT_Q_DTYPES: List[Optional[str]] = [None, "fp8"]
DEFAULT_NUM_BLOCKS: List[int] = [2048]
DEFAULT_SEQ_LENS: List[List[Tuple[int, int]]] = [
    [(1, 1328), (5, 18), (129, 463)],   # mixed prefill + decode
]

# `--full`: matches the Triton UA test matrix exactly (3 head-configs ×
# 2 num_blocks × 2 seq_lens batches) where CK can run it. ~288 configs.
FULL_NUM_HEADS: List[Tuple[int, int]] = [
    (4, 4),    # MHA
    (16, 2),   # GQA-8
    (64, 8),   # GQA-8 (production)
]
FULL_NUM_BLOCKS: List[int] = [2048, 32768]
FULL_SEQ_LENS: List[List[Tuple[int, int]]] = [
    [(1, 1328), (5, 18), (129, 463)],
    [(1, 523),  (1, 37), (1, 2011)],
]

# `--quick`: smoke subset (~8 configs).
QUICK_NUM_HEADS = [(16, 2)]
QUICK_BLOCK_SIZES = [32]
QUICK_NUM_BLOCKS = [2048]
QUICK_SEQ_LENS = [[(1, 1328), (5, 18), (129, 463)]]


@dataclass
class CaseConfig:
    seq_lens: List[Tuple[int, int]]
    num_heads: Tuple[int, int]
    head_size: int
    block_size: int
    dtype: torch.dtype
    q_dtype: Optional[str]       # None or "fp8"
    num_blocks: int


# ----------------------------------------------------------------------------
# Reference (matches Triton test's `ref_paged_attn` exactly).
# ----------------------------------------------------------------------------
def ref_paged_attn(
    query: torch.Tensor,         # [total_q, num_q_heads, head_size]  (high-precision)
    key_cache: torch.Tensor,     # [num_blocks, block_size, num_kv_heads, head_size]
    value_cache: torch.Tensor,
    query_lens: List[int],
    kv_lens: List[int],
    block_tables: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    num_seqs = len(query_lens)
    block_tables_np = block_tables.cpu().numpy()
    _, block_size, num_kv_heads, head_size = key_cache.shape

    outputs = []
    start = 0
    for i in range(num_seqs):
        query_len = query_lens[i]
        kv_len = kv_lens[i]
        q = query[start:start + query_len]
        q = q * scale

        num_kv_blocks = (kv_len + block_size - 1) // block_size
        block_indices = block_tables_np[i, :num_kv_blocks]

        k = key_cache[block_indices].view(-1, num_kv_heads, head_size)[:kv_len]
        v = value_cache[block_indices].view(-1, num_kv_heads, head_size)[:kv_len]

        # GQA broadcast.
        if q.shape[1] != k.shape[1]:
            qpkv = q.shape[1] // k.shape[1]
            k = torch.repeat_interleave(k, qpkv, dim=1)
            v = torch.repeat_interleave(v, qpkv, dim=1)

        attn = torch.einsum("qhd,khd->hqk", q.float(), k.float())
        mask = torch.triu(
            torch.ones(query_len, kv_len, device=q.device),
            diagonal=kv_len - query_len + 1,
        ).bool()
        attn.masked_fill_(mask, float("-inf"))
        attn = torch.softmax(attn, dim=-1).to(v.dtype)
        out = torch.einsum("hqk,khd->qhd", attn, v)
        outputs.append(out)
        start += query_len

    return torch.cat(outputs, dim=0)


# ----------------------------------------------------------------------------
# Input synthesis. Mirrors the Triton test exactly so the same inputs feed
# both backends and the reference.
# ----------------------------------------------------------------------------
def _make_inputs(cfg: CaseConfig, device: str, seed: int):
    torch.manual_seed(seed)
    num_query_heads, num_kv_heads = cfg.num_heads
    assert num_query_heads % num_kv_heads == 0
    query_lens = [x[0] for x in cfg.seq_lens]
    kv_lens = [x[1] for x in cfg.seq_lens]
    total_q = sum(query_lens)
    max_query_len = max(query_lens)
    max_kv_len = max(kv_lens)
    scale = cfg.head_size ** -0.5

    query = torch.randn(
        total_q, num_query_heads, cfg.head_size, dtype=cfg.dtype, device=device
    )
    key_cache = torch.randn(
        cfg.num_blocks, cfg.block_size, num_kv_heads, cfg.head_size,
        dtype=cfg.dtype, device=device,
    )
    value_cache = torch.randn_like(key_cache)

    cu_seqlens_q = torch.tensor(
        [0] + query_lens, dtype=torch.int32, device=device
    ).cumsum(dim=0, dtype=torch.int32)
    kv_lens_t = torch.tensor(kv_lens, dtype=torch.int32, device=device)
    max_num_blocks_per_seq = (max_kv_len + cfg.block_size - 1) // cfg.block_size
    num_seqs = len(cfg.seq_lens)
    block_tables = torch.randint(
        0, cfg.num_blocks,
        (num_seqs, max_num_blocks_per_seq),
        dtype=torch.int32, device=device,
    )

    # FP8 quantisation: per-tensor symmetric, matching Triton test.
    q_descale = k_descale = v_descale = 1.0
    if cfg.q_dtype == "fp8":
        fp8_dtype = _pick_fp8_dtype()
        fp8_max = float(torch.finfo(fp8_dtype).max)

        def _q(t):
            amax = t.detach().abs().amax().clamp(min=1e-9).item()
            descale = amax / fp8_max
            return (t / descale).clamp_(-fp8_max, fp8_max).to(fp8_dtype), descale

        q_fp8, q_descale = _q(query)
        k_fp8, k_descale = _q(key_cache)
        v_fp8, v_descale = _q(value_cache)
    else:
        q_fp8 = k_fp8 = v_fp8 = None

    return {
        "query": query,
        "key_cache": key_cache,
        "value_cache": value_cache,
        "q_fp8": q_fp8,
        "k_fp8": k_fp8,
        "v_fp8": v_fp8,
        "q_descale": q_descale,
        "k_descale": k_descale,
        "v_descale": v_descale,
        "cu_seqlens_q": cu_seqlens_q,
        "kv_lens": kv_lens_t,
        "kv_lens_list": kv_lens,
        "query_lens": query_lens,
        "block_tables": block_tables,
        "max_query_len": max_query_len,
        "max_kv_len": max_kv_len,
        "scale": scale,
        "total_q": total_q,
    }


def _int32_overflow_possible(num_blocks, block_size, num_kv_heads, head_size) -> bool:
    INT32_MAX = 2 ** 31 - 1
    return num_blocks * block_size * num_kv_heads * head_size > INT32_MAX


def _attn_flops_and_mem_bytes(cfg: "CaseConfig", total_q: int) -> Tuple[int, int]:
    """Theoretical attention FLOPs and HBM-traffic bytes for one launch.

    Used by single-shape mode to report TFLOPs / GB/s (the same back-of-
    the-envelope cost model the standalone test_single_shape.py used, so
    bandwidth numbers stay directly comparable across the consolidation).

    FLOPs: Q·Kᵀ and Attn·V are each `2 · total_q · seqlen_k · head_dim ·
    num_q_heads` for causal/no-mask (the GQA broadcast doesn't change the
    per-q-head MFMA count); softmax/mask are ignored as O(N) relative to
    the O(N·D) matmuls.

    Bytes: Q + K + V at the activation dtype (FP8 halves the K/V traffic
    that dominates decode) + output at bf16 (CK FP8 outputs bf16 too).
    Sum over all sequences in the batch; assumes the batch is N copies of
    the same (sq, sk) shape, which is how single-shape mode synthesises
    its seq_lens list.
    """
    batch = len(cfg.seq_lens)
    sq, sk = cfg.seq_lens[0]
    hq, hk = cfg.num_heads
    d = cfg.head_size
    bytes_per_elem = 1 if cfg.q_dtype == "fp8" else (
        2 if cfg.dtype in (torch.bfloat16, torch.float16) else 4
    )
    bytes_per_out = 2  # bf16 always for the kernel outputs we benchmark
    flops = 4 * total_q * sk * d * hq
    mem_q = total_q * hq * d * bytes_per_elem
    mem_k = batch * sk * hk * d * bytes_per_elem
    mem_v = mem_k
    mem_o = total_q * hq * d * bytes_per_out
    return flops, mem_q + mem_k + mem_v + mem_o


def _compute_num_splits(cfg: "CaseConfig", total_q: int, device: str) -> int:
    """Return what aiter's split-KV wrapper would pick for this shape.

    `_pick_num_splits` only reads tensor shapes + .device, so we can call
    it on empty (no-data) tensors here — no need to materialise the full
    KV cache just to probe the heuristic. Used by single-shape mode to
    tag the report with `num_splits` so the reader can attribute any
    perf surprise to the combine-kernel path vs. the attention kernel.
    """
    from aiter.ops.unified_attention import _pick_num_splits
    hq, hk = cfg.num_heads
    q = torch.empty((max(1, total_q), hq, cfg.head_size),
                    dtype=cfg.dtype, device=device)
    k = torch.empty((1, cfg.block_size, hk, cfg.head_size),
                    dtype=cfg.dtype, device=device)
    seq_lens = torch.empty((len(cfg.seq_lens),),
                           dtype=torch.int32, device=device)
    return int(_pick_num_splits(q, k, seq_lens))


# ----------------------------------------------------------------------------
# Kernel runners — exactly one per backend so @perftest amortises warmup +
# does proper torch profiler timing per the test_pa convention.
#
# IMPORTANT: every GPU tensor the kernel reads is passed as a *positional*
# argument here. @perftest's `device_memory_profiling` sums positional-tensor
# bytes to size its L2-cache rotation buffer; tensors hidden inside dicts or
# kwargs are invisible to it, so it would clamp to its 101-iter ceiling and
# OOM at long-context / large-batch shapes (multi-GB per copy). Keeping the
# bulky tensors positional lets the auto-sizer scale rotations to the GPU's
# free memory, which is what makes the long-context FP8 sweep run end-to-end
# without manual rotation tuning.
# ----------------------------------------------------------------------------
@perftest()
def run_ck(
    out,
    query,
    key_cache,
    value_cache,
    q_fp8,
    k_fp8,
    v_fp8,
    block_tables,
    kv_lens,
    cu_seqlens_q,
    *,
    q_descale,
    k_descale,
    v_descale,
    scale,
    max_query_len,
    mask_type,
    allow_splitkv=True,
):
    from aiter.ops.unified_attention import unified_attention_fwd

    q = q_fp8 if q_fp8 is not None else query
    k = k_fp8 if k_fp8 is not None else key_cache
    v = v_fp8 if v_fp8 is not None else value_cache
    overflow = _int32_overflow_possible(k.shape[0], k.shape[1], k.shape[2], k.shape[3])

    unified_attention_fwd(
        out,
        q,
        k,
        v,
        block_tables,
        kv_lens,
        cu_seqlens_q,
        mask_type=mask_type,
        scale_s=scale,
        scale=1.0,
        scale_k=1.0,
        scale_v=1.0,
        scale_out=1.0,
        cache_ptr_int32_overflow_possible=overflow,
        # By default we let the transparent split-KV wrapper pick num_splits,
        # since that is what production callers will see. Set
        # `allow_splitkv=False` (via the `--no-splitkv` CLI flag) to force
        # the single-launch path — useful for isolating the combine overhead
        # and for A/B-ing the heuristic, not as a correctness workaround.
        allow_splitkv=allow_splitkv,
        q_descale=float(q_descale),
        k_descale=float(k_descale),
        v_descale=float(v_descale),
        max_seqlen_q=max_query_len,
    )
    return out


@perftest()
def run_triton(
    out,
    query,
    key_cache,
    value_cache,
    q_fp8,
    k_fp8,
    v_fp8,
    block_tables,
    kv_lens,
    cu_seqlens_q,
    *,
    q_descale,
    k_descale,
    v_descale,
    scale,
    max_query_len,
    max_kv_len,
):
    from aiter.ops.triton.attention.unified_attention import unified_attention

    q = q_fp8 if q_fp8 is not None else query
    k = k_fp8 if k_fp8 is not None else key_cache
    v = v_fp8 if v_fp8 is not None else value_cache
    fp8 = q_fp8 is not None
    q_descale_t = (
        torch.tensor([q_descale], dtype=dtypes.fp32, device=out.device) if fp8 else None
    )
    k_descale_t = (
        torch.tensor([k_descale], dtype=dtypes.fp32, device=out.device) if fp8 else None
    )
    v_descale_t = (
        torch.tensor([v_descale], dtype=dtypes.fp32, device=out.device) if fp8 else None
    )

    unified_attention(
        q=q,
        k=k,
        v=v,
        out=out,
        cu_seqlens_q=cu_seqlens_q,
        seqused_k=kv_lens,
        max_seqlen_q=max_query_len,
        max_seqlen_k=max_kv_len,
        softmax_scale=scale,
        causal=True,
        window_size=(-1, -1),
        block_table=block_tables,
        softcap=0.0,
        q_descale=q_descale_t,
        k_descale=k_descale_t,
        v_descale=v_descale_t,
    )
    return out


def _ck_dispatch_supported(cfg: CaseConfig) -> Optional[str]:
    """Return None if the kernel can run this config, else a skip reason."""
    if cfg.head_size not in (64, 128):
        return f"head_size={cfg.head_size} not in CK instances"
    if cfg.block_size not in (16, 32, 64):
        return f"block_size={cfg.block_size} not in CK instances"
    if cfg.q_dtype == "fp8" and cfg.block_size < 32:
        return "fp8 path requires block_size >= 32"
    if cfg.num_heads[0] % cfg.num_heads[1] != 0:
        return f"num_heads {cfg.num_heads} not divisible (GQA invariant)"
    return None


# ----------------------------------------------------------------------------
# Per-config test driver — owns input synthesis, runs both backends, runs the
# reference, asserts correctness, and returns a dict the parent loop folds
# into the summary DataFrame.
# ----------------------------------------------------------------------------
@benchmark()
def test_unified_attention_ck(
    seq_lens: List[Tuple[int, int]],
    num_heads: Tuple[int, int],
    head_size: int,
    block_size: int,
    dtype: torch.dtype,
    q_dtype: Optional[str],
    num_blocks: int,
    *,
    seed: int = 0,
    device: str = "cuda",
    run_triton_backend: bool = True,
    skip_reference: bool = False,
    allow_splitkv: bool = True,
) -> dict:
    cfg = CaseConfig(
        seq_lens=seq_lens,
        num_heads=num_heads,
        head_size=head_size,
        block_size=block_size,
        dtype=dtype,
        q_dtype=q_dtype,
        num_blocks=num_blocks,
    )
    skip = _ck_dispatch_supported(cfg)
    if skip is not None:
        aiter.logger.info(f"SKIP {cfg}: {skip}")
        return {"status": f"skipped: {skip}"}

    t = _make_inputs(cfg, device, seed)

    # Output dtype: bf16 for FP8 inputs (matches CK FP8 traits + Triton test),
    # else the activation dtype.
    out_dtype = dtypes.bf16 if cfg.q_dtype == "fp8" else cfg.dtype
    out_ck = torch.empty(
        t["total_q"], num_heads[0], head_size, dtype=out_dtype, device=device
    )

    # NOTE: @perftest deep-copies args for L2-cache rotation, so the output
    # captured for correctness comes from the kernel's return value (the
    # `out` of the *last* rotated copy), not the `out_ck` we passed in.
    out_ck, time_ck = run_ck(
        out_ck,
        t["query"], t["key_cache"], t["value_cache"],
        t["q_fp8"], t["k_fp8"], t["v_fp8"],
        t["block_tables"], t["kv_lens"], t["cu_seqlens_q"],
        q_descale=t["q_descale"], k_descale=t["k_descale"], v_descale=t["v_descale"],
        scale=t["scale"], max_query_len=t["max_query_len"],
        mask_type=2, allow_splitkv=allow_splitkv,
    )

    # Triton kernel for cross-check + perf comparison.
    time_triton = None
    out_triton = None
    if run_triton_backend:
        out_triton_buf = torch.empty_like(out_ck)
        try:
            out_triton, time_triton = run_triton(
                out_triton_buf,
                t["query"], t["key_cache"], t["value_cache"],
                t["q_fp8"], t["k_fp8"], t["v_fp8"],
                t["block_tables"], t["kv_lens"], t["cu_seqlens_q"],
                q_descale=t["q_descale"], k_descale=t["k_descale"], v_descale=t["v_descale"],
                scale=t["scale"], max_query_len=t["max_query_len"],
                max_kv_len=t["max_kv_len"],
            )
        except Exception as e:
            aiter.logger.info(f"Triton run failed for {cfg}: {e}")
            out_triton = None

    # Reference (torch). The reference always consumes the high-precision
    # source tensors — quantisation noise shows up in both kernels' outputs
    # but never in the reference, which is the convention the upstream
    # Triton test uses too.
    atol = 1.5e-1 if cfg.q_dtype == "fp8" else 1.5e-2
    rtol = 1.5e-1 if cfg.q_dtype == "fp8" else 1e-2
    ck_passed = triton_passed = None
    if not skip_reference:
        ref = ref_paged_attn(
            t["query"], t["key_cache"], t["value_cache"],
            t["query_lens"], t["kv_lens_list"], t["block_tables"], t["scale"],
        ).to(out_dtype)

        ck_passed = checkAllclose(
            ref, out_ck, atol=atol, rtol=rtol,
            msg=f"CK     vs ref     | {cfg} | {time_ck:>8.2f} us",
        ) == 0
        if out_triton is not None:
            triton_passed = checkAllclose(
                ref, out_triton, atol=atol, rtol=rtol,
                msg=f"Triton vs ref     | {cfg} | {time_triton:>8.2f} us",
            ) == 0

    speedup = (time_triton / time_ck) if (time_triton is not None) else None
    # num_splits surfaces what the transparent split-KV wrapper picked
    # for this shape. Single-shape mode tags the report with it so any
    # perf surprise can be attributed to the combine-kernel path vs the
    # attention kernel proper; grid mode treats it as just another data
    # column in the DataFrame.
    num_splits = _compute_num_splits(cfg, t["total_q"], device) if allow_splitkv else 1
    return {
        "ck_us":       round(time_ck, 2),
        "triton_us":   round(time_triton, 2) if time_triton is not None else None,
        "ck_vs_tri":   round(speedup, 2) if speedup is not None else None,
        "ck_pass":     ck_passed,
        "triton_pass": triton_passed,
        "num_splits":  num_splits,
        "status":      "ok",
    }


# ----------------------------------------------------------------------------
# Entry point: argparse + cartesian iteration + DataFrame report.
# ----------------------------------------------------------------------------
def _parse_seq_lens(s: str) -> List[Tuple[int, int]]:
    """Parse '1,1328;5,18;129,463' into [(1,1328),(5,18),(129,463)]."""
    out = []
    for chunk in s.split(";"):
        a, b = chunk.split(",")
        out.append((int(a), int(b)))
    return out


def _parse_dtype(s: str) -> torch.dtype:
    return {
        "bf16": dtypes.bf16, "bfloat16": dtypes.bf16,
        "fp16": torch.float16, "float16": torch.float16, "half": torch.float16,
    }[s]


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description=__doc__,
    )
    parser.add_argument("--quick", action="store_true",
                        help="Smoke subset (one head-config, one block size).")
    parser.add_argument("--full", action="store_true",
                        help="Run the full Triton-UA-style matrix "
                             "(~288 configs, ~30 min on a single MI355).")
    parser.add_argument("--head-size", type=int, nargs="*",
                        default=None, help="Restrict head sizes.")
    parser.add_argument("--block-size", type=int, nargs="*",
                        default=None, help="Restrict block sizes.")
    parser.add_argument("--dtype", type=_parse_dtype, nargs="*",
                        default=None, help="Restrict dtypes (bf16, fp16).")
    parser.add_argument("--q-dtype", type=str, nargs="*",
                        default=None, choices=["none", "fp8"],
                        help="Restrict quant dtypes: 'none' (full-precision) "
                             "or 'fp8' (per-tensor e4m3).")
    parser.add_argument("--num-heads", type=str, nargs="*", default=None,
                        help="Restrict (HQ,HK) tuples, e.g. --num-heads 16,2 64,8")
    parser.add_argument("--num-blocks", type=str, nargs="*",
                        default=None, help="Restrict KV-cache pool sizes. "
                                            "Pass integers, or the literal "
                                            "'auto' to size the pool to "
                                            "`batch * ceil(sk / block_size)` "
                                            "so block_tables hold unique "
                                            "physical indices and the "
                                            "benchmark sees no fake L2 "
                                            "reuse (matches vLLM/SGLang-"
                                            "style allocators). 'auto' "
                                            "requires single-shape mode.")
    parser.add_argument("--seq-lens", type=_parse_seq_lens, nargs="*",
                        default=None, help="Restrict seq_lens batches, format: "
                                            "'1,1328;5,18;129,463'")

    # Single-shape shortcut — turns `b` repeated `(sq, sk)` pairs into one
    # seq_lens batch. Replaces the standalone ua-test-scripts/test_single_shape.py
    # script for ad-hoc shape investigations during kernel-side iteration.
    parser.add_argument("-b", "--batch", type=int, default=None,
                        help="Single-shape mode: number of sequences. Requires "
                             "--seq-q and --seq-k; mutually exclusive with "
                             "--seq-lens.")
    parser.add_argument("-sq", "--seq-q", type=int, default=None,
                        help="Single-shape mode: query length per sequence "
                             "(use 1 for decode).")
    parser.add_argument("-sk", "--seq-k", type=int, default=None,
                        help="Single-shape mode: KV (context) length per "
                             "sequence.")
    parser.add_argument("--seed", type=int, default=0)
    # Triton comparison default depends on mode:
    #   - single-shape  → ON  (you almost always want the head-to-head)
    #   - grid (default/quick/full) → OFF (correctness vs ref only — keeps
    #     CK-side regression sweeps fast + uncoupled from Triton's perf)
    # Override explicitly with `--triton` / `--no-triton`. Using
    # BooleanOptionalAction so the unset default (`None`) is distinguishable
    # from an explicit `--no-triton`, which is what lets the per-mode
    # default kick in only when the user hasn't expressed a preference.
    parser.add_argument("--triton", action=argparse.BooleanOptionalAction,
                        default=None,
                        help="Compare against the Triton kernel. Default: "
                             "ON in single-shape mode, OFF for grid runs. "
                             "Use --no-triton to force off (e.g. CK-only "
                             "regression sweeps) or --triton to force on.")
    parser.add_argument("--no-reference", action="store_true",
                        help="Skip the torch reference (perf-only run).")
    parser.add_argument("--no-splitkv", action="store_true",
                        help="Force the CK single-launch path (disable the "
                             "transparent split-KV wrapper). Useful for "
                             "measuring the kernel directly or working "
                             "around split-KV bugs.")
    args = parser.parse_args()

    if args.quick:
        num_heads_cfg  = QUICK_NUM_HEADS
        block_sizes    = QUICK_BLOCK_SIZES
        num_blocks_lst = QUICK_NUM_BLOCKS
        seq_lens_grid  = QUICK_SEQ_LENS
    elif args.full:
        num_heads_cfg  = FULL_NUM_HEADS
        block_sizes    = DEFAULT_BLOCK_SIZES
        num_blocks_lst = FULL_NUM_BLOCKS
        seq_lens_grid  = FULL_SEQ_LENS
    else:
        num_heads_cfg  = DEFAULT_NUM_HEADS
        block_sizes    = DEFAULT_BLOCK_SIZES
        num_blocks_lst = DEFAULT_NUM_BLOCKS
        seq_lens_grid  = DEFAULT_SEQ_LENS

    head_sizes    = DEFAULT_HEAD_SIZES
    dtypes_list   = DEFAULT_DTYPES
    q_dtypes_list = DEFAULT_Q_DTYPES

    if args.head_size:
        head_sizes = args.head_size
    if args.block_size:
        block_sizes = args.block_size
    if args.dtype:
        dtypes_list = args.dtype
    if args.q_dtype:
        q_dtypes_list = [None if x == "none" else x for x in args.q_dtype]
    if args.num_heads:
        num_heads_cfg = [
            tuple(int(x) for x in s.split(","))
            for s in args.num_heads
        ]
    if args.seq_lens:
        seq_lens_grid = args.seq_lens

    # Single-shape mode is mutually exclusive with --seq-lens because
    # both control the same axis. The shortcut just expands to one
    # `seq_lens_grid` entry of `batch` identical `(seq_q, seq_k)` pairs.
    single_shape_flags = [args.batch, args.seq_q, args.seq_k]
    single_shape = any(v is not None for v in single_shape_flags)
    if single_shape:
        if not all(v is not None for v in single_shape_flags):
            parser.error("--batch/--seq-q/--seq-k must be passed together")
        if args.seq_lens:
            parser.error("--batch/--seq-q/--seq-k is mutually exclusive "
                         "with --seq-lens")
        seq_lens_grid = [[(args.seq_q, args.seq_k)] * args.batch]

    # Resolve the mode-dependent Triton default (see CLI comment).
    run_triton = args.triton if args.triton is not None else single_shape

    # `--num-blocks auto` requires a single (batch, sk, block_size) so we
    # can compute the exact pool size. It's most useful precisely in
    # single-shape mode (the only mode where there's one well-defined sk).
    if args.num_blocks:
        if any(x.lower() == "auto" for x in args.num_blocks):
            if len(args.num_blocks) != 1:
                parser.error("--num-blocks auto must be the only --num-blocks "
                             "value (can't mix 'auto' with explicit sizes)")
            if not (all(v is not None for v in single_shape_flags)
                    and len(block_sizes) == 1):
                parser.error("--num-blocks auto requires --batch/--seq-q/"
                             "--seq-k *and* a single --block-size so the "
                             "pool size is unambiguously batch * "
                             "ceil(sk / block_size).")
            pages_per_seq = (args.seq_k + block_sizes[0] - 1) // block_sizes[0]
            num_blocks_lst = [args.batch * pages_per_seq]
        else:
            num_blocks_lst = [int(x) for x in args.num_blocks]

    rows = []
    for seq_lens in seq_lens_grid:
        for nh in num_heads_cfg:
            for hd in head_sizes:
                for bs in block_sizes:
                    for dt in dtypes_list:
                        for qd in q_dtypes_list:
                            for nb in num_blocks_lst:
                                ret = test_unified_attention_ck(
                                    seq_lens, nh, hd, bs, dt, qd, nb,
                                    seed=args.seed,
                                    run_triton_backend=run_triton,
                                    skip_reference=args.no_reference,
                                    allow_splitkv=not args.no_splitkv,
                                )
                                rows.append(ret)
                                # Release the deep-copied rotation buffers
                                # @perftest holds before the next config
                                # claims VRAM. Without this the allocator
                                # reuses the same chunks across configs and
                                # we've observed sporadic NaN outputs from
                                # CK at bs=64 + nh=(16,2) — the root cause
                                # is still under investigation; the cache
                                # flush sidesteps it cleanly here.
                                gc.collect()
                                torch.cuda.empty_cache()

    if single_shape:
        _print_single_shape_report(rows[0], seq_lens_grid[0], num_heads_cfg[0],
                                   head_sizes[0], block_sizes[0],
                                   dtypes_list[0], q_dtypes_list[0],
                                   num_blocks_lst[0])
    else:
        df = pd.DataFrame(rows)
        # `@benchmark()` merges the call kwargs into the row dict; drop
        # the ones that are constant across rows to keep the table
        # readable.
        for noise_col in ("seed", "device", "run_triton_backend",
                           "skip_reference", "allow_splitkv"):
            if noise_col in df.columns:
                df = df.drop(columns=[noise_col])
        aiter.logger.info(
            "unified_attention_ck summary (markdown):\n%s",
            df.to_markdown(index=False),
        )

    # Surface any failure as a non-zero exit so CI flags it. Done in both
    # modes — single-shape failure usually means "this shape is broken,
    # please look", so we still want a non-zero exit there.
    failed_rows = [
        r for r in rows
        if r.get("ck_pass") is False or r.get("triton_pass") is False
    ]
    if failed_rows:
        aiter.logger.error(
            "%d row(s) failed correctness:\n%s",
            len(failed_rows),
            pd.DataFrame(failed_rows).to_markdown(index=False),
        )
        raise SystemExit(1)


def _print_single_shape_report(
    row: dict,
    seq_lens: List[Tuple[int, int]],
    num_heads: Tuple[int, int],
    head_size: int,
    block_size: int,
    dtype: torch.dtype,
    q_dtype: Optional[str],
    num_blocks: int,
) -> None:
    """Pretty-print a single-shape result block.

    Replaces the report that used to live in the standalone
    ua-test-scripts/test_single_shape.py. Output is intentionally
    grep-friendly so downstream sweep scripts (e.g.
    ua-test-scripts/regression_decode.sh) can scrape CK time / bandwidth /
    correctness / split-KV status without parsing the DataFrame.
    """
    batch = len(seq_lens)
    sq, sk = seq_lens[0]
    hq, hk = num_heads
    total_q = batch * sq
    cfg = CaseConfig(
        seq_lens=seq_lens, num_heads=num_heads, head_size=head_size,
        block_size=block_size, dtype=dtype, q_dtype=q_dtype,
        num_blocks=num_blocks,
    )
    flops, mem_bytes = _attn_flops_and_mem_bytes(cfg, total_q)
    mem_gb = mem_bytes / 1e9
    phase = "decode" if sq == 1 else "prefill"

    print()
    print("=" * 80)
    print("CK unified attention — single-shape mode")
    print("=" * 80)
    print("Shape:")
    print(f"  batch         = {batch}")
    print(f"  seqlen_q      = {sq}")
    print(f"  seqlen_k      = {sk}")
    print(f"  num_q_heads   = {hq}")
    print(f"  num_kv_heads  = {hk}")
    print(f"  head_size     = {head_size}")
    print(f"  block_size    = {block_size}")
    print(f"  num_blocks    = {num_blocks}")
    print(f"  dtype         = {dtype}  q_dtype={q_dtype}")
    print(f"  phase         = {phase}")
    print(f"  total_q       = {total_q}")
    num_splits = row.get("num_splits")
    if num_splits is not None:
        active = "ACTIVE" if num_splits > 1 else "off"
        # Heads-up next to the timing block: if num_splits > 1 you're
        # paying for two launches (attention kernel + combine) plus the
        # workspace alloc, so any per-shape under-perf relative to a
        # b=128 sweep number may live in the combine path rather than
        # the attention kernel proper.
        print(f"  split-KV      = num_splits={num_splits} ({active})")
    try:
        gpu = torch.cuda.get_device_properties(0)
        print(f"  GPU           = {gpu.gcnArchName}  "
              f"CUs={gpu.multi_processor_count}  "
              f"HBM={gpu.total_memory >> 20}MB")
    except Exception:
        pass

    print("-" * 80)
    print("Correctness (vs torch reference):")
    ck_p = row.get("ck_pass")
    tr_p = row.get("triton_pass")
    if ck_p is not None:
        print(f"  CK     vs ref:  {'PASS' if ck_p else 'FAIL'}")
    if tr_p is not None:
        print(f"  Triton vs ref:  {'PASS' if tr_p else 'FAIL'}")

    print("-" * 80)
    print("Timing (median of @perftest iters):")
    ck_us = row.get("ck_us")
    tr_us = row.get("triton_us")
    if ck_us is not None:
        ms = ck_us / 1e3
        tflops = (flops / 1e12) / (ms / 1e3)
        bw = mem_gb / (ms / 1e3)
        print(f"  CK time        = {ms:8.4f} ms")
        print(f"  CK Bandwidth   = {bw:8.2f} GB/s")
        print(f"  CK TFLOPs      = {tflops:8.2f}")
    if tr_us is not None:
        ms = tr_us / 1e3
        tflops = (flops / 1e12) / (ms / 1e3)
        bw = mem_gb / (ms / 1e3)
        print(f"  Triton time    = {ms:8.4f} ms")
        print(f"  Triton Bandwidth={bw:8.2f} GB/s")
        print(f"  Triton TFLOPs  = {tflops:8.2f}")
    if ck_us is not None and tr_us is not None:
        speedup = tr_us / ck_us
        winner = "CK" if speedup >= 1.0 else "Triton"
        print(f"  Speedup        = {speedup:.3f}x ({winner} wins)")
    print("=" * 80)


if __name__ == "__main__":
    main()
