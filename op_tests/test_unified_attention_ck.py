# SPDX-License-Identifier: MIT
# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.

"""Correctness + performance tests for CK unified attention.

Shape coverage mirrors `op_tests/triton_tests/attention/test_unified_attention.py`
(the upstream Triton FlashDecoding test), filtered to the configurations the
CK kernel actually supports:

  * head_size  ∈ {64, 128}  (Triton also covers 256; CK has no 256 instance)
  * block_size ∈ {16, 32, 64} (Triton also covers 48; CK ships ps16/ps32/ps64)
  * dtype      ∈ {bf16, fp16, fp8}
                  - bf16 / fp16 → Q/K/V stored at the activation dtype
                  - fp8         → Q/K/V quantised to e4m3 (per-tensor symmetric)
                                  on top of a bf16 source — vLLM / SGLang's
                                  recipe; the same convention the CK FP8
                                  problem traits resolve at compile time.
  * num_heads  ∈ {(4,4), (8,1), (16,2), (64,8)}  — MHA + GQA-{2,8}
  * mask       ∈ {bottom-right causal, top-left causal, no-mask} ×
                 (window_size_left, window_size_right) ∈ any FA/vLLM
                 SWA pair. The default `(mask_type=2, window=(-1,-1))`
                 is plain bottom-right causal (no SWA) and matches the
                 pre-SWA behaviour bit-for-bit.

Reporting follows `op_tests/test_pa.py`:
  * `@perftest()` for per-kernel timing
  * `@benchmark()` for the per-config wrapper
  * pandas DataFrame summary at the end with both correctness and timings

Examples:
  python op_tests/test_unified_attention_ck.py
  python op_tests/test_unified_attention_ck.py --quick           # smoke subset
  python op_tests/test_unified_attention_ck.py --head-size 128   # restrict
  python op_tests/test_unified_attention_ck.py --no-triton       # CK-only

  # SWA on/off perf comparison: run the existing grid twice, once
  # without SWA and once with a 128-token left window. Each shape
  # produces two rows in the DataFrame so the SWA cost is one
  # subtraction away. (`--window` takes one semicolon-separated arg
  # of `L,R` pairs; argparse can't accept a leading `-` as a value.)
  python op_tests/test_unified_attention_ck.py --window='-1,-1;128,0'

  # Curated SWA correctness sweep (ported from the standalone bash
  # scripts in 3rdparty/.../42_unified_attention/script/). Each
  # category is a self-contained shape sweep; 'all' expands to every
  # category. The same @perftest()/@benchmark() machinery times each
  # fixture so the run produces a perf-+-correctness table just like
  # the cartesian-grid mode.
  python op_tests/test_unified_attention_ck.py --swa-fixtures all
  python op_tests/test_unified_attention_ck.py --swa-fixtures smoke fp8

  # Single-shape mode — replaces the standalone ua-test-scripts/test_single_shape.py
  # for ad-hoc correctness/perf checks; expands to one seq_lens batch of
  # `b` identical (sq, sk) sequences. `--num-blocks auto` allocates exactly
  # `b * ceil(sk / block_size)` physical blocks so block_tables hold unique
  # indices (no fake L2 reuse), matching what vLLM/SGLang-style allocators
  # produce in production.
  python op_tests/test_unified_attention_ck.py \\
      -b 64 -sq 1 -sk 128000 \\
      --num-heads 64,8 --head-size 128 --block-size 16 \\
      --dtype bf16 --num-blocks auto

  # Same shape with SWA (128-token left window):
  python op_tests/test_unified_attention_ck.py \\
      -b 64 -sq 1 -sk 128000 \\
      --num-heads 64,8 --head-size 128 --block-size 16 \\
      --dtype bf16 --num-blocks auto --window='128,0'
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
# Single dtype axis combining the kernel's two internal dtype slots:
#   source dtype (the activation dtype the reference + kernel consume)
#   + optional FP8 per-tensor quantisation layered on top.
# `fp8` covers the bf16-source / e4m3-quant path that vLLM and SGLang
# actually use; bf16-source is the only quantisation source CK ships
# traits for (and what the Triton FP8 reference also expects), so we
# don't enumerate `fp16-source fp8` as a separate axis here. If someone
# ever wants it they can hand-edit this list.
DEFAULT_DTYPES: List[Tuple[torch.dtype, Optional[str]]] = [
    (dtypes.bf16,   None),
    (torch.float16, None),
    (dtypes.bf16,   "fp8"),
]
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
    # Mask + sliding-window-attention bounds, FA/vLLM semantics:
    #   mask_type          : 0 = no mask, 1 = top-left causal,
    #                        2 = bottom-right causal (default).
    #   window_size_left   : -1 = unbounded left context.
    #   window_size_right  : -1 = unbounded right context. With
    #                        mask_type != 0 the CK wrapper coerces -1
    #                        to 0 (the standard causal anchor), so
    #                        the default tuple below maps to plain
    #                        bottom-right causal and behaves exactly
    #                        like the pre-SWA test path.
    mask_type: int = 2
    window_size_left: int = -1
    window_size_right: int = -1


# ----------------------------------------------------------------------------
# SWA-aware reference. Generalises the Triton test's `ref_paged_attn` so the
# same helper handles plain causal (mask_type=2, window=(-1,-1)) AND sliding-
# window attention.
#
# Mask semantics mirror FA / vLLM / CK exactly:
#   * is_top_left = (mask_type == 1) — query row q_idx anchors at key col q_idx.
#   * is_top_left = False (mask_type == 2) — bottom-right anchor: query row
#     q_idx anchors at key col (kv_len - query_len + q_idx). Standard FA varlen
#     / Triton SWA convention.
#   * window_size_left  < 0 → no left bound  (unbounded earlier context).
#   * window_size_right < 0 + mask_type != 0 → coerced to 0 here, mirroring
#     the CK wrapper. So default `mask_type=2, window=(-1,-1)` reduces to
#     plain bottom-right causal — bit-identical to the pre-SWA reference.
# ----------------------------------------------------------------------------
def _build_swa_mask(
    query_len: int,
    kv_len: int,
    mask_type: int,
    window_size_left: int,
    window_size_right: int,
    device: torch.device,
) -> torch.Tensor:
    """Returns a (query_len, kv_len) bool tensor; True = masked out (-inf)."""
    if mask_type == 0:
        return torch.zeros((query_len, kv_len), dtype=torch.bool, device=device)
    # Match the CK wrapper: window_size_right<0 + masked → causal anchor (0).
    if window_size_right < 0:
        window_size_right = 0
    q_pos = torch.arange(query_len, device=device).unsqueeze(1)  # [Q, 1]
    k_pos = torch.arange(kv_len, device=device).unsqueeze(0)     # [1, K]
    is_top_left = (mask_type == 1)
    # `center` is the K column index the query is anchored to. For
    # bottom-right (default causal) the last query token anchors at the
    # last KV token.
    center = q_pos if is_top_left else q_pos + (kv_len - query_len)

    valid = torch.ones((query_len, kv_len), dtype=torch.bool, device=device)
    if window_size_left >= 0:
        valid &= (k_pos >= (center - window_size_left))
    if window_size_right >= 0:
        valid &= (k_pos <= (center + window_size_right))
    return ~valid


def ref_paged_attn(
    query: torch.Tensor,         # [total_q, num_q_heads, head_size]  (high-precision)
    key_cache: torch.Tensor,     # [num_blocks, block_size, num_kv_heads, head_size]
    value_cache: torch.Tensor,
    query_lens: List[int],
    kv_lens: List[int],
    block_tables: torch.Tensor,
    scale: float,
    mask_type: int = 2,
    window_size_left: int = -1,
    window_size_right: int = -1,
) -> torch.Tensor:
    num_seqs = len(query_lens)
    block_tables_np = block_tables.cpu().numpy()
    _, block_size, num_kv_heads, head_size = key_cache.shape

    outputs = []
    start = 0
    for i in range(num_seqs):
        query_len = query_lens[i]
        kv_len = kv_lens[i]
        # Keep the whole reduction in fp32 — mixed-dtype einsum crashes,
        # and casting attn mid-reduction loses precision for free.
        q = query[start:start + query_len].float() * scale

        num_kv_blocks = (kv_len + block_size - 1) // block_size
        block_indices = block_tables_np[i, :num_kv_blocks]

        k = key_cache[block_indices].view(-1, num_kv_heads, head_size)[:kv_len]
        v = value_cache[block_indices].view(-1, num_kv_heads, head_size)[:kv_len]

        # GQA broadcast.
        if q.shape[1] != k.shape[1]:
            qpkv = q.shape[1] // k.shape[1]
            k = torch.repeat_interleave(k, qpkv, dim=1)
            v = torch.repeat_interleave(v, qpkv, dim=1)

        attn = torch.einsum("qhd,khd->hqk", q, k.float())
        mask = _build_swa_mask(
            query_len, kv_len,
            mask_type, window_size_left, window_size_right,
            device=q.device,
        )
        attn.masked_fill_(mask, float("-inf"))
        attn = torch.softmax(attn, dim=-1)
        # Pure-mask rows (entirely -inf) appear at sparse-window corners;
        # softmax produces NaN there. The CK kernel writes a zero row
        # (LSE = -inf); match that so the comparison stays meaningful.
        # For plain causal this branch never fires.
        attn = torch.nan_to_num(attn, nan=0.0)
        out = torch.einsum("hqk,khd->qhd", attn, v.float()).to(query.dtype)
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
# SWA fixture data. These are the curated SWA shape + window combinations
# the team uses to gate SWA correctness (ported 1:1 from the standalone
# bash scripts in
# `3rdparty/composable_kernel/example/ck_tile/42_unified_attention/script/`).
# Each `SwaFixture` is a complete config — when `--swa-fixtures CATEGORY` is
# passed on the CLI, the runner iterates over the named lists instead of
# the cartesian default grid. The same `@benchmark()`-decorated driver
# captures the per-fixture CK / Triton timings, so the SWA cases land in
# the same DataFrame report as the rest of the grid.
# ----------------------------------------------------------------------------
@dataclass
class SwaFixture:
    name: str
    seq_lens: List[Tuple[int, int]]
    num_heads: Tuple[int, int]
    head_size: int
    block_size: int
    mask_type: int
    window_size_left: int
    window_size_right: int
    dtype: torch.dtype = dtypes.bf16
    q_dtype: Optional[str] = None
    num_blocks: int = 1024
    category: str = "smoke"


# Block size: the standalone CK example builds `ps=128` instances; aiter
# ships only `ps ∈ {16, 32, 64}` (see `optCompilerConfig.json`). Cases
# that the bash fixtures ran with `-page_blk_size=128` are translated to
# `block_size=64` here — that still exercises the kernel-tile-vs-page
# interaction (for prefill_d64 with kPageBlockSize=32 it means 2 tiles
# per page, i.e. mid-page Step D starts are possible). GPT-OSS shapes
# stay at 32, and the non-page-aligned stress uses 32 + 64 to keep
# coverage of both the page-aligned (1 tile/page) and mid-page (2
# tiles/page) Step D start arithmetic.

# BASELINE_B from the bash script: d=64 GQA-8 prefill (-h_k=1 -nqpkv=8).
_BASE_B_SEQS  = [(400, 400), (256, 256), (512, 512), (128, 128)]
# Pure prefill d=128: q_len > 128 forces prefill_d128 (above the
# decode_d128_m128 threshold).
_PRE128_SEQS  = [(257, 257), (512, 512)]
# Pure d=128 *decode* with SWA. q_len=1 everywhere forces the decode_d128
# tier instead of the prefill tier (the dispatcher's max_seqlen_q
# heuristic routes num_tokens > num_seqs to a prefill tier).
_DECODE_D128_SEQS = [(1, 256)] * 4

_SMOKE_SWA_FIXTURES: List[SwaFixture] = [
    # baseB SWA via xformer-style window (xb:W).
    SwaFixture("baseB xb:64",  _BASE_B_SEQS, (8, 1), 64, 64, 2, 64,  0),
    SwaFixture("baseB xb:128", _BASE_B_SEQS, (8, 1), 64, 64, 2, 128, 0),
    # baseB SWA via FA-style explicit left/right window (b:L,R).
    SwaFixture("baseB b:64,0", _BASE_B_SEQS, (8, 1), 64, 64, 2, 64,  0),
    # Pure prefill SWA on d=128: validates the d=128 prefill SWA instance.
    SwaFixture("prefill d128 xb:64",  _PRE128_SEQS, (8, 1), 128, 64, 2, 64, 0),
    SwaFixture("prefill d128 b:64,0", _PRE128_SEQS, (8, 1), 128, 64, 2, 64, 0),
    # Pure d=128 decode SWA. All three decode tiers (m128 / m32 / m16)
    # now have SWA kernel instances. The dispatcher routes by
    # `max_rows = max_q * num_qpkv`; `q_len=1, num_qpkv=1` lands on
    # decode_d128_m16. To also cover m32 / m128 we use GQA shapes.
    SwaFixture("decode d128 MHA Q=1 xb:64",  _DECODE_D128_SEQS, (8, 8), 128, 64, 2, 64,  0),
    SwaFixture("decode d128 MHA Q=1 xb:128", _DECODE_D128_SEQS, (8, 8), 128, 64, 2, 128, 0),
    SwaFixture("decode d128 MHA Q=1 b:64,0", _DECODE_D128_SEQS, (8, 8), 128, 64, 2, 64,  0),
    # GQA-32 q=1 → max_rows=32 → decode_d128_m32 tier.
    SwaFixture("decode d128 m32 GQA-32 Q=1 xb:64",
               [(1, 256)] * 4, (32, 1), 128, 64, 2, 64, 0),
    # GQA-8 q=16 → max_rows=128 (edge of the m128 bucket).
    SwaFixture("decode d128 m128 GQA-8 Q=16 xb:64",
               [(16, 256)] * 4, (8, 1), 128, 64, 2, 64, 0),
    # MHA q=32 → max_rows=32 → decode_d64_m64 tier.
    SwaFixture("decode d64 m64 MHA Q=32 xb:64",
               [(32, 256)] * 4, (8, 8), 64, 64, 2, 64, 0),
]

# PRE_64 / PRE_128 baselines (always-prefill shapes; force the d=64 /
# d=128 prefill tiers from the dispatcher).
_PRE_64_SEQS  = [(512, 512), (512, 512)]
_PRE_128_SEQS = [(257, 257), (512, 512)]

_EDGE_SWA_FIXTURES: List[SwaFixture] = [
    # Window = 1: only the diagonal cell (and one neighbour). Tests
    # Step D's right-clip collapsing to the smallest non-empty range
    # per Q-row.
    SwaFixture("d64  window=1 (xb:1)",  _PRE_64_SEQS,  (8, 1), 64,  64, 2, 1, 0, category="edge"),
    SwaFixture("d64  window=1 (b:0,0)", _PRE_64_SEQS,  (8, 1), 64,  64, 2, 0, 0, category="edge"),
    SwaFixture("d128 window=1 (xb:1)",  _PRE_128_SEQS, (8, 1), 128, 64, 2, 1, 0, category="edge"),
    SwaFixture("d128 window=1 (b:0,0)", _PRE_128_SEQS, (8, 1), 128, 64, 2, 0, 0, category="edge"),
    # Window >= seq_k: SWA collapses to dense (within the causal half).
    SwaFixture("d64  window>=sk (xb:2048)", _PRE_64_SEQS,  (8, 1), 64,  64, 2, 2048, 0,    category="edge"),
    SwaFixture("d64  window=511,511 (b:)",  _PRE_64_SEQS,  (8, 1), 64,  64, 2, 511,  511,  category="edge"),
    SwaFixture("d128 window>=sk (xb:2048)", _PRE_128_SEQS, (8, 1), 128, 64, 2, 2048, 0,    category="edge"),
    SwaFixture("d128 window=511,511 (b:)",  _PRE_128_SEQS, (8, 1), 128, 64, 2, 511,  511,  category="edge"),
    # Top-left vs bottom-right anchor. Same window radius, different
    # diagonal alignment.
    SwaFixture("d64  top-left (xt:64)",  _PRE_64_SEQS,  (8, 1), 64,  64, 1, 64, 0, category="edge"),
    SwaFixture("d128 top-left (xt:64)",  _PRE_128_SEQS, (8, 1), 128, 64, 1, 64, 0, category="edge"),
    # Asymmetric left/right windows.
    SwaFixture("d64  asymmetric (b:32,8)", _PRE_64_SEQS,  (8, 1), 64,  64, 2, 32, 8, category="edge"),
    SwaFixture("d64  asymmetric (b:8,32)", _PRE_64_SEQS,  (8, 1), 64,  64, 2, 8, 32, category="edge"),
    SwaFixture("d128 asymmetric (b:8,32)", _PRE_128_SEQS, (8, 1), 128, 64, 2, 8, 32, category="edge"),
    # Odd seqlen: 480 = 7.5 pages at block_size=64; last KV tile is a
    # true edge tile.
    SwaFixture("d64  odd s_k=480 (xb:64)",  [(480, 480), (480, 480)], (8, 1), 64,  64, 2, 64, 0, category="edge"),
    SwaFixture("d128 odd s_k=480 (xb:64)",  [(257, 480), (480, 480)], (8, 1), 128, 64, 2, 64, 0, category="edge"),
]

# GPT-OSS fixtures (decode_bs32, page_blk_size=32).
_GPTOSS_SWA_FIXTURES: List[SwaFixture] = [
    SwaFixture("DECODE_BS32 Q=1   xb:128",  [(1, 512)] * 4,    (8, 1), 64, 32, 2, 128, 0, category="gptoss"),
    SwaFixture("DECODE_BS32 Q=1   b:127,0", [(1, 512)] * 4,    (8, 1), 64, 32, 2, 127, 0, category="gptoss"),
    SwaFixture("DECODE_BS32 Q=128 xb:128",  [(128, 1024)] * 4, (8, 1), 64, 32, 2, 128, 0, category="gptoss"),
    SwaFixture("DECODE_BS32 Q=128 b:127,0", [(128, 1024)] * 4, (8, 1), 64, 32, 2, 127, 0, category="gptoss"),
    SwaFixture("DECODE_BS32 QM    xb:128",  [(512, 1024), (1024, 1024), (512, 1024), (1024, 1024)],
               (8, 1), 64, 32, 2, 128, 0, category="gptoss"),
    SwaFixture("DECODE_BS32 QM    b:127,0", [(512, 1024), (1024, 1024), (512, 1024), (1024, 1024)],
               (8, 1), 64, 32, 2, 127, 0, category="gptoss"),
]

# Non-page-aligned stress (page_size != kPageBlockSize tile size).
_NONALIGN_SWA_FIXTURES: List[SwaFixture] = [
    SwaFixture("non-align ps=32  xb:64",      _PRE_64_SEQS,  (8, 1), 64,  32, 2, 64, 0, category="non-align"),
    SwaFixture("non-align ps=64  xb:64",      _PRE_64_SEQS,  (8, 1), 64,  64, 2, 64, 0, category="non-align"),
    SwaFixture("non-align ps=64  b:48,0",     _PRE_64_SEQS,  (8, 1), 64,  64, 2, 48, 0, category="non-align"),
    SwaFixture("non-align d128 ps=64 xb:64",  _PRE_128_SEQS, (8, 1), 128, 64, 2, 64, 0, category="non-align"),
]

# FP8 SWA (per-tensor fp8e4m3 quant + descale, block_size>=32 mandatory).
_FP8_SWA_FIXTURES: List[SwaFixture] = [
    SwaFixture("fp8 prefill d64  xb:64",     _PRE_64_SEQS,  (8, 1), 64,  64, 2, 64,  0,
               q_dtype="fp8", category="fp8"),
    SwaFixture("fp8 prefill d64  b:32,8",    _PRE_64_SEQS,  (8, 1), 64,  64, 2, 32,  8,
               q_dtype="fp8", category="fp8"),
    SwaFixture("fp8 prefill d128 xb:64",     _PRE_128_SEQS, (8, 1), 128, 64, 2, 64,  0,
               q_dtype="fp8", category="fp8"),
    SwaFixture("fp8 prefill d128 xt:64",     _PRE_128_SEQS, (8, 1), 128, 64, 1, 64,  0,
               q_dtype="fp8", category="fp8"),
    SwaFixture("fp8 decode d64  m16  MHA Q=1 xb:128", [(1, 512)] * 4,
               (8, 8), 64, 32, 2, 128, 0, q_dtype="fp8", category="fp8"),
    SwaFixture("fp8 decode d64  m64  MHA Q=32 xb:64", [(32, 256)] * 4,
               (8, 8), 64, 32, 2, 64, 0, q_dtype="fp8", category="fp8"),
    SwaFixture("fp8 decode d64  m128 GQA-8 Q=128 xb:128", [(128, 1024)] * 4,
               (8, 1), 64, 32, 2, 128, 0, q_dtype="fp8", category="fp8"),
    SwaFixture("fp8 decode d128 m16  MHA Q=1 xb:64", [(1, 256)] * 4,
               (8, 8), 128, 64, 2, 64, 0, q_dtype="fp8", category="fp8"),
    SwaFixture("fp8 decode d128 m32  GQA-32 Q=1 xb:64", [(1, 256)] * 4,
               (32, 1), 128, 64, 2, 64, 0, q_dtype="fp8", category="fp8"),
    SwaFixture("fp8 decode d128 m128 GQA-8 Q=16 xb:64", [(16, 256)] * 4,
               (8, 1), 128, 64, 2, 64, 0, q_dtype="fp8", category="fp8"),
]

SWA_FIXTURE_GROUPS: dict = {
    "smoke":     _SMOKE_SWA_FIXTURES,
    "edge":      _EDGE_SWA_FIXTURES,
    "gptoss":    _GPTOSS_SWA_FIXTURES,
    "non-align": _NONALIGN_SWA_FIXTURES,
    "fp8":       _FP8_SWA_FIXTURES,
}


def _expand_swa_fixtures(categories: List[str]) -> List[SwaFixture]:
    """Return the curated SwaFixture list for the named categories.
    `all` expands to every category in `SWA_FIXTURE_GROUPS`."""
    if not categories or "all" in categories:
        categories = list(SWA_FIXTURE_GROUPS.keys())
    out: List[SwaFixture] = []
    for cat in categories:
        out.extend(SWA_FIXTURE_GROUPS[cat])
    return out


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
    window_size_left=-1,
    window_size_right=-1,
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
        window_size_left=window_size_left,
        window_size_right=window_size_right,
    )
    return out


def triton_supports_swa(mask_type: int, window_size_right: int) -> bool:
    """The aiter Triton UA kernel only honours the *left* window
    (`SLIDING_WINDOW = window_size[0] + 1`) and is hard-coded to
    bottom-right causal (`assert causal`). For top-left masks or any
    non-zero / non-negative right window the comparison would be
    apples-to-oranges, so the caller should skip the Triton leg.
    """
    if mask_type == 1:           # top-left causal
        return False
    if window_size_right > 0:    # asymmetric right window
        return False
    return True


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
    window_size_left=-1,
    window_size_right=-1,
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
        # Triton only consumes window_size[0] (left) as
        # `SLIDING_WINDOW = window_size[0] + 1`; window_size[1] is
        # ignored. For non-SWA (window_size_left=-1) we pass (-1, -1)
        # which the kernel reads as "no SLIDING_WINDOW", matching the
        # pre-SWA behaviour bit-for-bit.
        window_size=(window_size_left, window_size_right),
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
    # SWA mask config. Defaults reduce to plain bottom-right causal
    # (the kernel/wrapper coerces window_size_right=-1 to 0 when
    # mask_type != 0), so passing these defaults preserves the pre-SWA
    # behaviour bit-for-bit.
    mask_type: int = 2,
    window_size_left: int = -1,
    window_size_right: int = -1,
) -> dict:
    cfg = CaseConfig(
        seq_lens=seq_lens,
        num_heads=num_heads,
        head_size=head_size,
        block_size=block_size,
        dtype=dtype,
        q_dtype=q_dtype,
        num_blocks=num_blocks,
        mask_type=mask_type,
        window_size_left=window_size_left,
        window_size_right=window_size_right,
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
        mask_type=mask_type,
        window_size_left=window_size_left,
        window_size_right=window_size_right,
        allow_splitkv=allow_splitkv,
    )

    # Triton kernel for cross-check + perf comparison. The aiter Triton
    # UA path is hard-coded to bottom-right causal and only honours
    # `window_size[0]` (the left window). For top-left masks or any
    # non-zero right window the kernels diverge by construction, so
    # we skip the Triton leg there and report just the CK side — the
    # SwaCase report has a `triton_skipped` column to surface this.
    time_triton = None
    out_triton = None
    triton_skipped_reason = None
    if run_triton_backend:
        if not triton_supports_swa(mask_type, window_size_right):
            triton_skipped_reason = (
                f"triton has no top-left / wr>0 SWA "
                f"(mask_type={mask_type}, wr={window_size_right})"
            )
        else:
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
                    window_size_left=window_size_left,
                    window_size_right=window_size_right,
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
            mask_type=mask_type,
            window_size_left=window_size_left,
            window_size_right=window_size_right,
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
        "triton_skip": triton_skipped_reason,
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


def _parse_window_list(s: str) -> List[Tuple[int, int]]:
    """Parse 'L,R[;L,R …]' into a list of (L, R) pairs. Single arg with
    semicolons separates pairs — argparse can't take `-1,-1` directly as
    a positional value because it parses leading `-` as an option flag,
    and `--window=-1,-1` syntax is awkward to remember. The semicolon
    form sidesteps both issues.

    Examples:
      '-1,-1'         → no SWA baseline only.
      '-1,-1;128,0'   → SWA-off + SWA-128 (causal SWA, left=128).
      '64,0;64,8;1,0' → bottom-right causal SWA + asymmetric + degenerate.

    Each side is an int, -1 = unbounded.
    """
    pairs = []
    for chunk in s.split(";"):
        a, b = chunk.split(",")
        pairs.append((int(a), int(b)))
    return pairs


def _parse_dtype(s: str) -> Tuple[torch.dtype, Optional[str]]:
    """Resolve a user-facing dtype name to the (source_dtype, q_dtype) pair
    the kernel call expects.

    The CK problem traits + the Triton FP8 reference both model FP8 as
    "per-tensor symmetric quantisation of bf16 source tensors", so `fp8`
    here is a shorthand for that combo — there is no `fp16-source fp8`
    path. `--dtype fp8` does what the CLI name suggests: Q, K and V are
    quantised to e4m3.
    """
    return {
        "bf16":     (dtypes.bf16,    None),
        "bfloat16": (dtypes.bf16,    None),
        "fp16":     (torch.float16,  None),
        "float16":  (torch.float16,  None),
        "half":     (torch.float16,  None),
        "fp8":      (dtypes.bf16,    "fp8"),
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
                        default=None,
                        help="Restrict dtypes. Values: bf16, fp16, fp8. "
                             "`fp8` means Q/K/V are per-tensor symmetrically "
                             "quantised to e4m3 from a bf16 source — the "
                             "vLLM/SGLang convention the CK FP8 traits + "
                             "Triton FP8 reference both expect.")
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
    parser.add_argument("--allow-splitkv", action="store_true",
                        default=None,
                        help="Force split-KV on. Mostly useful in "
                             "--swa-fixtures mode where the default is "
                             "single-launch (kernel-correctness gate) — "
                             "pass this to switch fixtures into the "
                             "production split-KV path for perf "
                             "comparison. On the cartesian grid the "
                             "default is already on, so this is a no-op "
                             "unless paired with --no-splitkv.")
    # SWA axes. `--window` extends the cartesian grid with one extra
    # dimension: each (window_size_left, window_size_right) pair adds
    # one row per other-axis cell. `(-1, -1)` is the no-SWA baseline,
    # so e.g. `--window -1,-1 128,0` runs both the existing causal grid
    # AND the same grid with a 128-token left window — giving a direct
    # SWA-on vs SWA-off perf comparison per shape.
    parser.add_argument("--window", type=_parse_window_list,
                        default=None,
                        help="Semicolon-separated list of sliding-window "
                             "'L,R' pairs to iterate over (FA/vLLM "
                             "semantics; -1 = unbounded). Default: "
                             "'-1,-1' (no SWA). Example: "
                             "--window='-1,-1;128,0' runs every grid "
                             "config twice — once without SWA and once "
                             "with a 128-token left window.")
    parser.add_argument("--mask-type", type=int, nargs="*", default=None,
                        help="One or more mask types to iterate over: "
                             "0=no mask, 1=top-left causal, 2=bottom-"
                             "right causal (default).")
    # Curated SWA fixture mode — bypasses the cartesian grid and runs
    # the pre-baked shape+window combinations from the SWA correctness
    # sweep (ported from
    # `3rdparty/.../42_unified_attention/script/{smoke,edge}_test_swa.sh`).
    parser.add_argument("--swa-fixtures", type=str, nargs="*",
                        default=None,
                        choices=["smoke", "edge", "gptoss", "non-align",
                                 "fp8", "all"],
                        help="Run the curated SWA shape+window fixtures "
                             "instead of the cartesian grid. Each "
                             "category is a self-contained shape sweep; "
                             "'all' expands to every category.")
    parser.add_argument("--swa-filter", type=str, default=None,
                        help="Substring filter on the SWA fixture name "
                             "(used with --swa-fixtures).")
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
    dtype_pairs   = DEFAULT_DTYPES

    if args.head_size:
        head_sizes = args.head_size
    if args.block_size:
        block_sizes = args.block_size
    if args.dtype:
        dtype_pairs = args.dtype
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

    # SWA axes for the cartesian grid. Defaults preserve pre-SWA behaviour
    # bit-for-bit (one row per shape, plain bottom-right causal).
    window_pairs: List[Tuple[int, int]] = (
        args.window if args.window else [(-1, -1)]
    )
    mask_types   = args.mask_type if args.mask_type else [2]

    rows = []
    # ---- Curated SWA fixture mode -----------------------------------------
    # When --swa-fixtures is set we ignore the cartesian grid + --window /
    # --mask-type entirely. Each fixture is a complete config; the same
    # @benchmark()-decorated driver runs it through the CK / Triton /
    # reference pipeline and the result lands in the same DataFrame.
    #
    # Default-off split-KV for the fixture sweep: this mode is a kernel-
    # correctness gate (ported from the standalone SWA file which forced
    # single-launch for exactly this reason — "we want the kernel output,
    # not the split-KV combine"). The cartesian-grid path keeps the
    # production-typical split-KV default; users who want a split-KV
    # SWA perf comparison should use `--window L,R ...` on the grid.
    # `--no-splitkv` / explicit `--allow-splitkv` still wins on both
    # paths.
    if args.swa_fixtures is not None:
        # Fixture mode: single-launch by default (kernel-correctness gate,
        # matches the old SWA file). `--allow-splitkv` opts into the
        # production split-KV path for perf testing; `--no-splitkv` is
        # the explicit "force off" override (a no-op given the default).
        fixture_allow_splitkv = bool(args.allow_splitkv) and not args.no_splitkv

        fixtures = _expand_swa_fixtures(args.swa_fixtures)
        if args.swa_filter:
            fixtures = [f for f in fixtures if args.swa_filter in f.name]
        if not fixtures:
            parser.error(
                f"No SWA fixtures matched: categories={args.swa_fixtures} "
                f"filter={args.swa_filter!r}"
            )
        for fx in fixtures:
            ret = test_unified_attention_ck(
                fx.seq_lens, fx.num_heads, fx.head_size, fx.block_size,
                fx.dtype, fx.q_dtype, fx.num_blocks,
                seed=args.seed,
                run_triton_backend=run_triton,
                skip_reference=args.no_reference,
                allow_splitkv=fixture_allow_splitkv,
                mask_type=fx.mask_type,
                window_size_left=fx.window_size_left,
                window_size_right=fx.window_size_right,
            )
            # Surface the fixture name + category in the DataFrame for the
            # human reader; both columns are dropped later if empty.
            ret["swa_fixture"]  = fx.name
            ret["swa_category"] = fx.category
            rows.append(ret)
            gc.collect()
            torch.cuda.empty_cache()
    else:
        # ---- Cartesian grid mode -----------------------------------------
        for seq_lens in seq_lens_grid:
            for nh in num_heads_cfg:
                for hd in head_sizes:
                    for bs in block_sizes:
                        for (dt, qd) in dtype_pairs:
                            for nb in num_blocks_lst:
                                for mt in mask_types:
                                    for (wl, wr) in window_pairs:
                                        ret = test_unified_attention_ck(
                                            seq_lens, nh, hd, bs, dt, qd, nb,
                                            seed=args.seed,
                                            run_triton_backend=run_triton,
                                            skip_reference=args.no_reference,
                                            allow_splitkv=not args.no_splitkv,
                                            mask_type=mt,
                                            window_size_left=wl,
                                            window_size_right=wr,
                                        )
                                        rows.append(ret)
                                        # Release the deep-copied rotation
                                        # buffers @perftest holds before
                                        # the next config claims VRAM.
                                        # Without this the allocator reuses
                                        # the same chunks across configs
                                        # and we've observed sporadic NaN
                                        # outputs from CK at bs=64 +
                                        # nh=(16,2) — the root cause is
                                        # still under investigation; the
                                        # cache flush sidesteps it cleanly
                                        # here.
                                        gc.collect()
                                        torch.cuda.empty_cache()

    if single_shape:
        dt0, qd0 = dtype_pairs[0]
        _print_single_shape_report(rows[0], seq_lens_grid[0], num_heads_cfg[0],
                                   head_sizes[0], block_sizes[0],
                                   dt0, qd0, num_blocks_lst[0])
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
