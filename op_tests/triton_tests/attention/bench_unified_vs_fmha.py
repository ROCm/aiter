#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""
Benchmark CK Unified Attention (`unified_attention_fwd`) against the full
set of CK FMHA forward entry points exposed in `aiter.ops.mha`:

    - mha_fwd                (batched 4D, non-paged)
    - mha_varlen_fwd         (varlen, non-paged)
    - mha_varlen_fwd  paged  (varlen + block_table)
    - mha_batch_prefill_func (varlen + CSR-paged: kv_indptr / kv_page_indices)
    - fmha_v3_fwd            (gfx942/gfx950 asm, batched)
    - fmha_v3_varlen_fwd     (gfx942/gfx950 asm, varlen)

All backends consume the same logical (Q, K, V) data, only re-laid-out for
each kernel's expected layout.

Usage:
    python bench_unified_vs_fmha.py                          # full sweep
    python bench_unified_vs_fmha.py --cases decode           # filter
    python bench_unified_vs_fmha.py --verify                 # accuracy check
    python bench_unified_vs_fmha.py --examples               # show example calls
    python bench_unified_vs_fmha.py --head-size 64           # GQA-8 sweep
    python bench_unified_vs_fmha.py --model gpt-oss-120b     # real-model preset
    python bench_unified_vs_fmha.py --model llama3-70b
    python bench_unified_vs_fmha.py --json results.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass, field
from typing import Optional

import torch

# ============================================================================
# Backend imports (lazy / tolerant -- some may be unavailable on older builds)
# ============================================================================

try:
    from aiter.ops.unified_attention import unified_attention_fwd
    HAVE_UA = True
except Exception as e:  # noqa: BLE001
    HAVE_UA = False
    UA_IMPORT_ERR = str(e)

try:
    from aiter.ops.mha import (
        mha_fwd,
        mha_varlen_fwd,
        mha_batch_prefill_func,
        fmha_v3_fwd,
        fmha_v3_varlen_fwd,
    )
    HAVE_MHA = True
except Exception as e:  # noqa: BLE001
    HAVE_MHA = False
    MHA_IMPORT_ERR = str(e)


# ============================================================================
# Case definition
# ============================================================================


@dataclass
class BenchCase:
    tag: str
    num_kv_heads: int
    num_queries_per_kv: int
    head_size: int
    page_blk_size: int
    query_lens: list[int]
    kv_lens: list[int]
    causal: bool = True
    dtype: str = "bf16"  # "bf16" or "fp16"

    @property
    def num_q_heads(self) -> int:
        return self.num_kv_heads * self.num_queries_per_kv

    @property
    def batch(self) -> int:
        return len(self.query_lens)

    @property
    def torch_dtype(self) -> torch.dtype:
        return torch.bfloat16 if self.dtype == "bf16" else torch.float16

    @property
    def is_uniform(self) -> bool:
        """True if all query lens equal and all kv lens equal (4D batched eligible)."""
        return (
            len(set(self.query_lens)) == 1
            and len(set(self.kv_lens)) == 1
        )


# ============================================================================
# Default sweep
# ============================================================================


def _decode_cases(num_kv_heads: int, num_queries_per_kv: int,
                  head_size: int, page_blk_size: int) -> list[BenchCase]:
    cases = []
    gqa_tag = f"h{num_kv_heads}x{num_queries_per_kv}_d{head_size}"
    for batch in [8, 32, 64, 128, 256, 384]:
        for kv_len in [1024, 4096]:
            cases.append(BenchCase(
                tag=f"decode_{gqa_tag}_b{batch}_k{kv_len}",
                num_kv_heads=num_kv_heads,
                num_queries_per_kv=num_queries_per_kv,
                head_size=head_size,
                page_blk_size=page_blk_size,
                query_lens=[1] * batch,
                kv_lens=[kv_len] * batch,
            ))
    return cases


def _prefill_cases(num_kv_heads: int, num_queries_per_kv: int,
                   head_size: int, page_blk_size: int) -> list[BenchCase]:
    cases = []
    gqa_tag = f"h{num_kv_heads}x{num_queries_per_kv}_d{head_size}"
    for s in [512, 1024, 2048, 4096]:
        cases.append(BenchCase(
            tag=f"prefill_{gqa_tag}_b1_q{s}_k{s}",
            num_kv_heads=num_kv_heads,
            num_queries_per_kv=num_queries_per_kv,
            head_size=head_size,
            page_blk_size=page_blk_size,
            query_lens=[s],
            kv_lens=[s],
        ))
    return cases


def _mixed_cases(num_kv_heads: int, num_queries_per_kv: int,
                 head_size: int, page_blk_size: int) -> list[BenchCase]:
    gqa_tag = f"h{num_kv_heads}x{num_queries_per_kv}_d{head_size}"
    return [
        BenchCase(
            tag=f"mixed_{gqa_tag}_chunked_pf",
            num_kv_heads=num_kv_heads,
            num_queries_per_kv=num_queries_per_kv,
            head_size=head_size,
            page_blk_size=page_blk_size,
            query_lens=[256, 1, 512, 1],
            kv_lens=[256, 2048, 512, 4096],
        ),
        BenchCase(
            tag=f"mixed_{gqa_tag}_decode_heavy",
            num_kv_heads=num_kv_heads,
            num_queries_per_kv=num_queries_per_kv,
            head_size=head_size,
            page_blk_size=page_blk_size,
            query_lens=[1] * 16 + [128, 256],
            kv_lens=[2048] * 16 + [128, 256],
        ),
    ]


# ============================================================================
# Real-model presets (sourced from
# `op_tests/op_benchmarks/triton/model_benchmarking_tool/model_shapes.json`)
# ============================================================================

# Each entry is (num_q_heads, num_kv_heads, head_size). Only symmetric (dqk==dv)
# attention configs are listed here -- DeepSeek-R1 *decode* is MLA (dqk=576,
# dv=512) which neither `unified_attention_fwd` nor `mha_fwd` handles, and
# DeepSeek-R1 *prefill* has dqk=192/dv=128 which is not in UA's instance set.
# Llama4 Maverick vision (hdim=88) is also outside UA's set.
MODELS: dict[str, tuple[int, int, int]] = {
    # GPT-OSS 120B: (GQA=8, hdim=64) <- UA's primary hot zone.
    "gpt-oss-120b":   (64,   8,  64),
    # Llama3-family: GQA=4/8/16, all hdim=128.  UA needs (GQA=1) for hdim=128;
    # mha_fwd handles all of these.
    "llama3-8b":      (32,   8, 128),
    "llama3-70b":     (64,   8, 128),
    "llama3-405b":    (128,  8, 128),
    # Llama4 Maverick text path: GQA=5, hdim=128.
    "llama4-maverick": (40,  8, 128),
    # Qwen3-235B: GQA=16, hdim=128.
    "qwen3-235b":     (64,   4, 128),
    # DeepSeek-R1 prefill MHA (asymmetric: dqk=192, dv=128 not modeled here;
    # we use the symmetric dqk=192 case which UA does *not* ship).
    "deepseek-r1-prefill": (128, 128, 192),
}


def make_model_cases(model_name: str) -> list[BenchCase]:
    if model_name not in MODELS:
        raise ValueError(
            f"unknown model {model_name!r}; available: {sorted(MODELS)}")
    h_q, h_kv, d = MODELS[model_name]
    if h_q % h_kv != 0:
        raise ValueError(f"{model_name}: h_q={h_q} not divisible by h_kv={h_kv}")
    gqa = h_q // h_kv
    page = 64 if d <= 128 else 128
    cases: list[BenchCase] = []
    cases += _decode_cases(num_kv_heads=h_kv, num_queries_per_kv=gqa,
                           head_size=d, page_blk_size=page)
    cases += _prefill_cases(num_kv_heads=h_kv, num_queries_per_kv=gqa,
                            head_size=d, page_blk_size=page)
    cases += _mixed_cases(num_kv_heads=h_kv, num_queries_per_kv=gqa,
                          head_size=d, page_blk_size=page)
    return cases


def make_default_cases(head_size: int) -> list[BenchCase]:
    """Default sweep: decode + prefill + mixed for the chosen head size.

    head_size=128 sweeps MHA (GQA=1) and GQA-8 (8 KV heads, 64 Q heads).
    head_size=64 sweeps GQA-8 only (CK-UA's other supported config).
    """
    cases = []
    if head_size == 128:
        # MHA (GQA=1): CK-UA-eligible config.
        cases += _decode_cases(num_kv_heads=8, num_queries_per_kv=1,
                               head_size=128, page_blk_size=64)
        cases += _prefill_cases(num_kv_heads=8, num_queries_per_kv=1,
                                head_size=128, page_blk_size=64)
        cases += _mixed_cases(num_kv_heads=8, num_queries_per_kv=1,
                              head_size=128, page_blk_size=64)
        # GQA-8 with hdim=128: NOT a CK-UA config but covers FMHA family.
        cases += _decode_cases(num_kv_heads=8, num_queries_per_kv=8,
                               head_size=128, page_blk_size=64)
        cases += _prefill_cases(num_kv_heads=8, num_queries_per_kv=8,
                                head_size=128, page_blk_size=64)
    elif head_size == 64:
        # GQA-8: CK-UA-eligible config.
        cases += _decode_cases(num_kv_heads=8, num_queries_per_kv=8,
                               head_size=64, page_blk_size=64)
        cases += _prefill_cases(num_kv_heads=8, num_queries_per_kv=8,
                                head_size=64, page_blk_size=64)
        cases += _mixed_cases(num_kv_heads=8, num_queries_per_kv=8,
                              head_size=64, page_blk_size=64)
    else:
        raise ValueError(f"--head-size must be 64 or 128, got {head_size}")
    return cases


# ============================================================================
# Tensor builders
# ============================================================================


@dataclass
class CaseTensors:
    """All tensor variants for one BenchCase, on the same logical Q/K/V data."""
    case: BenchCase
    device: torch.device

    # Common
    softmax_scale: float = 0.0

    # Q in varlen 3D layout: [total_q, h_q, d]
    q_3d: torch.Tensor = None  # type: ignore
    cu_seqlens_q: torch.Tensor = None  # type: ignore
    cu_seqlens_k: torch.Tensor = None  # type: ignore
    seq_lens_k: torch.Tensor = None  # type: ignore
    max_seqlen_q: int = 0
    max_seqlen_k: int = 0

    # Non-paged varlen K/V: [total_k, h_kv, d]
    k_3d: torch.Tensor = None  # type: ignore
    v_3d: torch.Tensor = None  # type: ignore

    # Paged K/V: [num_blocks, page, h_kv, d]
    k_paged: torch.Tensor = None  # type: ignore
    v_paged: torch.Tensor = None  # type: ignore
    block_tables: torch.Tensor = None  # type: ignore  # [batch, max_pages]
    num_blocks: int = 0

    # CSR paged (mha_batch_prefill)
    kv_indptr: torch.Tensor = None  # type: ignore
    kv_page_indices: torch.Tensor = None  # type: ignore
    kv_last_page_lens: torch.Tensor = None  # type: ignore

    # 4D batched (only when uniform): [b, s, h, d]
    q_4d: Optional[torch.Tensor] = None
    k_4d: Optional[torch.Tensor] = None
    v_4d: Optional[torch.Tensor] = None


def build_tensors(case: BenchCase, device: str = "cuda") -> CaseTensors:
    """Build every layout variant from one shared logical (Q, K, V) tensor.

    Strategy:
        1. Generate contiguous Q (3D varlen) and contiguous K/V (3D varlen).
        2. From the contiguous K/V, scatter into a paged buffer using a random
           block_tables permutation.  Both paged and non-paged paths read
           identical data.
        3. If `case.is_uniform`, also build 4D batched views.
    """
    dtype = case.torch_dtype
    dev = torch.device(device)

    h_q, h_kv, d = case.num_q_heads, case.num_kv_heads, case.head_size
    page = case.page_blk_size

    total_q = sum(case.query_lens)
    total_k = sum(case.kv_lens)

    # Q (varlen 3D)
    q_3d = torch.randn(total_q, h_q, d, dtype=dtype, device=dev)
    cu_seqlens_q = torch.tensor(
        [0] + list(torch.tensor(case.query_lens).cumsum(0).tolist()),
        dtype=torch.int32, device=dev,
    )
    cu_seqlens_k = torch.tensor(
        [0] + list(torch.tensor(case.kv_lens).cumsum(0).tolist()),
        dtype=torch.int32, device=dev,
    )
    seq_lens_k = torch.tensor(case.kv_lens, dtype=torch.int32, device=dev)
    max_seqlen_q = max(case.query_lens)
    max_seqlen_k = max(case.kv_lens)

    # K/V varlen 3D
    k_3d = torch.randn(total_k, h_kv, d, dtype=dtype, device=dev)
    v_3d = torch.randn(total_k, h_kv, d, dtype=dtype, device=dev)

    # Paged buffers: pick num_blocks large enough plus some slack for shuffling.
    pages_per_seq = [(kl + page - 1) // page for kl in case.kv_lens]
    total_pages = sum(pages_per_seq)
    # extra slack so block_tables can use non-trivial permutations
    num_blocks = max(total_pages + 16, total_pages * 2)

    k_paged = torch.zeros(num_blocks, page, h_kv, d, dtype=dtype, device=dev)
    v_paged = torch.zeros(num_blocks, page, h_kv, d, dtype=dtype, device=dev)

    # Permute pages: assign a random unique physical page ID to each (seq, logical_page)
    perm = torch.randperm(num_blocks, device=dev)[:total_pages].to(torch.int32)
    max_pages_per_seq = max(pages_per_seq)
    block_tables = torch.zeros(case.batch, max_pages_per_seq,
                               dtype=torch.int32, device=dev)

    # Fill paged buffers from contiguous KV
    perm_offset = 0
    kv_offset = 0
    for s, (kv_len, npages) in enumerate(zip(case.kv_lens, pages_per_seq)):
        for p_idx in range(npages):
            phys = int(perm[perm_offset + p_idx].item())
            block_tables[s, p_idx] = phys
            tok_start = kv_offset + p_idx * page
            tok_end = min(tok_start + page, kv_offset + kv_len)
            n = tok_end - tok_start
            k_paged[phys, :n] = k_3d[tok_start:tok_end]
            v_paged[phys, :n] = v_3d[tok_start:tok_end]
        perm_offset += npages
        kv_offset += kv_len

    # CSR paged layout for mha_batch_prefill
    kv_indptr = torch.tensor(
        [0] + list(torch.tensor(pages_per_seq).cumsum(0).tolist()),
        dtype=torch.int32, device=dev,
    )
    kv_page_indices = torch.empty(total_pages, dtype=torch.int32, device=dev)
    perm_offset = 0
    for s, npages in enumerate(pages_per_seq):
        for p_idx in range(npages):
            kv_page_indices[perm_offset + p_idx] = block_tables[s, p_idx]
        perm_offset += npages
    kv_last_page_lens = torch.tensor(
        [((kl - 1) % page) + 1 for kl in case.kv_lens],
        dtype=torch.int32, device=dev,
    )

    # Optional 4D batched layout
    q_4d = k_4d = v_4d = None
    if case.is_uniform:
        b = case.batch
        s_q = case.query_lens[0]
        s_k = case.kv_lens[0]
        q_4d = q_3d.view(b, s_q, h_q, d).contiguous()
        k_4d = k_3d.view(b, s_k, h_kv, d).contiguous()
        v_4d = v_3d.view(b, s_k, h_kv, d).contiguous()

    softmax_scale = 1.0 / math.sqrt(d)

    return CaseTensors(
        case=case, device=dev, softmax_scale=softmax_scale,
        q_3d=q_3d, cu_seqlens_q=cu_seqlens_q, cu_seqlens_k=cu_seqlens_k,
        seq_lens_k=seq_lens_k,
        max_seqlen_q=max_seqlen_q, max_seqlen_k=max_seqlen_k,
        k_3d=k_3d, v_3d=v_3d,
        k_paged=k_paged, v_paged=v_paged,
        block_tables=block_tables, num_blocks=num_blocks,
        kv_indptr=kv_indptr, kv_page_indices=kv_page_indices,
        kv_last_page_lens=kv_last_page_lens,
        q_4d=q_4d, k_4d=k_4d, v_4d=v_4d,
    )


# ============================================================================
# Skip checks
# ============================================================================


# CK-UA shipped instances: (head_size, num_queries_per_kv) in {(64,8),(128,1)}
_UA_ELIGIBLE = {(64, 8), (128, 1)}


def ua_skip_reason(case: BenchCase) -> Optional[str]:
    if not HAVE_UA:
        return "UA module unavailable"
    if (case.head_size, case.num_queries_per_kv) not in _UA_ELIGIBLE:
        return f"UA: (d, gqa)=({case.head_size}, {case.num_queries_per_kv}) not in {{(64,8),(128,1)}}"
    if case.page_blk_size < 32:
        return f"UA: page_blk_size {case.page_blk_size} < 32"
    if case.head_size <= 64 and case.page_blk_size != 64:
        return f"UA: hdim<=64 requires page=64, got {case.page_blk_size}"
    if case.head_size > 64 and case.page_blk_size != 32 and case.page_blk_size != 64:
        # Some d=128 instances exist for both 32 and 64; allow.
        pass
    if case.dtype not in ("fp16", "bf16"):
        return f"UA: dtype {case.dtype} not supported"
    return None


def mha_fwd_skip_reason(case: BenchCase) -> Optional[str]:
    if not HAVE_MHA:
        return "mha module unavailable"
    if not case.is_uniform:
        return "mha_fwd: requires uniform (q_len, kv_len) (4D batched)"
    return None


def _gfx_arch() -> str:
    try:
        return torch.cuda.get_device_properties(0).gcnArchName.split(":")[0]
    except Exception:  # noqa: BLE001
        return ""


_GFX = None  # cached


def fmha_v3_skip_reason(case: BenchCase) -> Optional[str]:
    """fmha_v3 ships ASM kernels for gfx942 (MI300X) and gfx950 (MI350).

    The Python selector does not gate on arch by default; only the FP8 sub-path
    requires gfx942 specifically.  Build/load failures on either arch are caught
    and reported as `err:...` by the runner, not pre-skipped here.
    """
    global _GFX
    if not HAVE_MHA:
        return "mha module unavailable"
    if _GFX is None:
        _GFX = _gfx_arch()
    if _GFX not in ("gfx942", "gfx950"):
        return f"fmha_v3: gfx942/gfx950 only (got {_GFX})"
    if case.dtype != "bf16":
        return "fmha_v3: bf16 only"
    if case.head_size not in (128, 192):
        return f"fmha_v3: hdim {case.head_size} not in {{128,192}}"
    return None


def fmha_v3_batched_skip_reason(case: BenchCase) -> Optional[str]:
    base = fmha_v3_skip_reason(case)
    if base:
        return base
    if not case.is_uniform:
        return "fmha_v3_fwd: requires uniform (q_len, kv_len)"
    return None


# ============================================================================
# Timing helper
# ============================================================================


def _bench(fn, warmup: int, iters: int) -> float:
    """Returns ms/iter using cuda events."""
    # Warmup
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


# ============================================================================
# Backend runners
# ============================================================================


def run_unified_attention_fwd(t: CaseTensors, warmup: int, iters: int) -> float:
    case = t.case
    out = torch.empty_like(t.q_3d)
    mask_type = 2 if case.causal else 0

    def _call():
        unified_attention_fwd(
            out, t.q_3d, t.k_paged, t.v_paged,
            t.block_tables, t.seq_lens_k, t.cu_seqlens_q,
            mask_type=mask_type,
            scale_s=t.softmax_scale,
            scale=1.0, scale_k=1.0, scale_v=1.0, scale_out=1.0,
        )

    return _bench(_call, warmup, iters)


def run_mha_fwd(t: CaseTensors, warmup: int, iters: int) -> float:
    case = t.case
    assert t.q_4d is not None and t.k_4d is not None and t.v_4d is not None

    def _call():
        mha_fwd(
            t.q_4d, t.k_4d, t.v_4d,
            dropout_p=0.0, softmax_scale=t.softmax_scale,
            is_causal=case.causal,
            window_size_left=-1, window_size_right=-1, sink_size=0,
            return_softmax_lse=False, return_dropout_randval=False,
        )

    return _bench(_call, warmup, iters)


def run_mha_varlen_fwd_paged(t: CaseTensors, warmup: int, iters: int) -> float:
    case = t.case

    def _call():
        mha_varlen_fwd(
            t.q_3d, t.k_paged, t.v_paged,
            t.cu_seqlens_q, t.cu_seqlens_k,
            t.max_seqlen_q, t.max_seqlen_k, 0,
            0.0, t.softmax_scale, 0.0, False,
            case.causal, -1, -1, 0,
            False, False,
            block_table=t.block_tables,
        )

    return _bench(_call, warmup, iters)


def run_mha_varlen_fwd_nonpaged(t: CaseTensors, warmup: int, iters: int) -> float:
    case = t.case

    def _call():
        mha_varlen_fwd(
            t.q_3d, t.k_3d, t.v_3d,
            t.cu_seqlens_q, t.cu_seqlens_k,
            t.max_seqlen_q, t.max_seqlen_k, 0,
            0.0, t.softmax_scale, 0.0, False,
            case.causal, -1, -1, 0,
            False, False,
            block_table=None,
        )

    return _bench(_call, warmup, iters)


def run_mha_batch_prefill(t: CaseTensors, warmup: int, iters: int) -> float:
    case = t.case

    def _call():
        mha_batch_prefill_func(
            t.q_3d, t.k_paged, t.v_paged,
            t.cu_seqlens_q, t.kv_indptr, t.kv_page_indices,
            t.max_seqlen_q, t.max_seqlen_k,
            dropout_p=0.0, softmax_scale=t.softmax_scale,
            causal=case.causal,
            kv_last_page_lens=t.kv_last_page_lens,
        )

    return _bench(_call, warmup, iters)


def run_fmha_v3_fwd(t: CaseTensors, warmup: int, iters: int) -> float:
    case = t.case
    assert t.q_4d is not None

    def _call():
        fmha_v3_fwd(
            t.q_4d, t.k_4d, t.v_4d,
            0.0, t.softmax_scale,
            case.causal, -1, -1,
            False, False,
            1,  # how_v3_bf16_cvt
        )

    return _bench(_call, warmup, iters)


def run_fmha_v3_varlen_fwd(t: CaseTensors, warmup: int, iters: int) -> float:
    case = t.case

    def _call():
        fmha_v3_varlen_fwd(
            t.q_3d, t.k_3d, t.v_3d,
            t.cu_seqlens_q, t.cu_seqlens_k,
            t.max_seqlen_q, t.max_seqlen_k, 0,
            0.0, t.softmax_scale, 0.0, False,
            case.causal, -1, -1,
            False, False,
            1,  # how_v3_bf16_cvt
        )

    return _bench(_call, warmup, iters)


# Order is also the column order in the printed table.
BACKENDS = [
    ("unified_attention", run_unified_attention_fwd, ua_skip_reason),
    ("mha_fwd",           run_mha_fwd,                mha_fwd_skip_reason),
    ("varlen_paged",      run_mha_varlen_fwd_paged,   lambda c: None if HAVE_MHA else "mha module unavailable"),
    ("varlen_nopage",     run_mha_varlen_fwd_nonpaged, lambda c: None if HAVE_MHA else "mha module unavailable"),
    ("batch_prefill",     run_mha_batch_prefill,      lambda c: None if HAVE_MHA else "mha module unavailable"),
    ("fmha_v3",           run_fmha_v3_fwd,            fmha_v3_batched_skip_reason),
    ("fmha_v3_varlen",    run_fmha_v3_varlen_fwd,     fmha_v3_skip_reason),
]


# ============================================================================
# Examples (for `--examples`)
# ============================================================================


def example_unified_attention_fwd():
    print("=== unified_attention_fwd (CK-UA) ===")
    case = BenchCase(
        tag="example_ua",
        num_kv_heads=8, num_queries_per_kv=1, head_size=128,
        page_blk_size=64,
        query_lens=[1, 1, 1, 1], kv_lens=[256, 256, 256, 256],
    )
    t = build_tensors(case)
    out = torch.empty_like(t.q_3d)
    unified_attention_fwd(
        out, t.q_3d, t.k_paged, t.v_paged,
        t.block_tables, t.seq_lens_k, t.cu_seqlens_q,
        mask_type=2, scale_s=t.softmax_scale,
        scale=1.0, scale_k=1.0, scale_v=1.0, scale_out=1.0,
    )
    torch.cuda.synchronize()
    print(f"  ran. out.abs().mean()={out.abs().mean().item():.6f}\n")


def example_mha_fwd():
    print("=== mha_fwd (CK FMHA, batched 4D) ===")
    case = BenchCase(
        tag="example_mha_fwd",
        num_kv_heads=8, num_queries_per_kv=1, head_size=128,
        page_blk_size=64,
        query_lens=[512] * 2, kv_lens=[512] * 2,
    )
    t = build_tensors(case)
    try:
        out, lse, S, rng = mha_fwd(
            t.q_4d, t.k_4d, t.v_4d,
            dropout_p=0.0, softmax_scale=t.softmax_scale,
            is_causal=True, window_size_left=-1, window_size_right=-1, sink_size=0,
            return_softmax_lse=False, return_dropout_randval=False,
        )
        torch.cuda.synchronize()
        print(f"  ran. out.shape={tuple(out.shape)} mean={out.abs().mean().item():.6f}\n")
    except Exception as e:  # noqa: BLE001
        print(f"  failed: {type(e).__name__}: {str(e).splitlines()[0]}\n")


def example_mha_varlen_fwd():
    print("=== mha_varlen_fwd (CK FMHA, varlen + paged block_table) ===")
    case = BenchCase(
        tag="example_varlen",
        num_kv_heads=8, num_queries_per_kv=1, head_size=128,
        page_blk_size=64,
        query_lens=[256, 1, 512, 1], kv_lens=[256, 2048, 512, 4096],
    )
    t = build_tensors(case)
    try:
        out, lse, S, rng = mha_varlen_fwd(
            t.q_3d, t.k_paged, t.v_paged,
            t.cu_seqlens_q, t.cu_seqlens_k,
            t.max_seqlen_q, t.max_seqlen_k, 0,
            0.0, t.softmax_scale, 0.0, False,
            True, -1, -1, 0,
            False, False,
            block_table=t.block_tables,
        )
        torch.cuda.synchronize()
        print(f"  ran. out.shape={tuple(out.shape)} mean={out.abs().mean().item():.6f}\n")
    except Exception as e:  # noqa: BLE001
        print(f"  failed: {type(e).__name__}: {str(e).splitlines()[0]}\n")


def example_mha_batch_prefill():
    print("=== mha_batch_prefill_func (CK FMHA, CSR-paged) ===")
    case = BenchCase(
        tag="example_batch_prefill",
        num_kv_heads=8, num_queries_per_kv=1, head_size=128,
        page_blk_size=64,
        query_lens=[256, 1, 512, 1], kv_lens=[256, 2048, 512, 4096],
    )
    t = build_tensors(case)
    try:
        out = mha_batch_prefill_func(
            t.q_3d, t.k_paged, t.v_paged,
            t.cu_seqlens_q, t.kv_indptr, t.kv_page_indices,
            t.max_seqlen_q, t.max_seqlen_k,
            dropout_p=0.0, softmax_scale=t.softmax_scale,
            causal=True,
            kv_last_page_lens=t.kv_last_page_lens,
        )
        torch.cuda.synchronize()
        out0 = out[0] if isinstance(out, (list, tuple)) else out
        print(f"  ran. out.shape={tuple(out0.shape)} mean={out0.abs().mean().item():.6f}\n")
    except Exception as e:  # noqa: BLE001
        print(f"  failed: {type(e).__name__}: {str(e).splitlines()[0]}\n")


def example_fmha_v3_fwd():
    print("=== fmha_v3_fwd (gfx942 asm, batched 4D) ===")
    case = BenchCase(
        tag="example_v3", num_kv_heads=8, num_queries_per_kv=1,
        head_size=128, page_blk_size=64,
        query_lens=[512] * 2, kv_lens=[512] * 2, dtype="bf16",
    )
    t = build_tensors(case)
    try:
        out, lse, S, rng = fmha_v3_fwd(
            t.q_4d, t.k_4d, t.v_4d,
            0.0, t.softmax_scale,
            True, -1, -1,
            False, False, 1,
        )
        torch.cuda.synchronize()
        print(f"  ran. out.shape={tuple(out.shape)} mean={out.abs().mean().item():.6f}\n")
    except Exception as e:  # noqa: BLE001
        print(f"  unavailable on this device: {e}\n")


def example_fmha_v3_varlen_fwd():
    print("=== fmha_v3_varlen_fwd (gfx942 asm, varlen) ===")
    case = BenchCase(
        tag="example_v3_varlen", num_kv_heads=8, num_queries_per_kv=1,
        head_size=128, page_blk_size=64,
        query_lens=[512, 1024], kv_lens=[512, 1024], dtype="bf16",
    )
    t = build_tensors(case)
    try:
        out, lse, S, rng = fmha_v3_varlen_fwd(
            t.q_3d, t.k_3d, t.v_3d,
            t.cu_seqlens_q, t.cu_seqlens_k,
            t.max_seqlen_q, t.max_seqlen_k, 0,
            0.0, t.softmax_scale, 0.0, False,
            True, -1, -1,
            False, False, 1,
        )
        torch.cuda.synchronize()
        print(f"  ran. out.shape={tuple(out.shape)} mean={out.abs().mean().item():.6f}\n")
    except Exception as e:  # noqa: BLE001
        print(f"  unavailable on this device: {e}\n")


def show_examples() -> int:
    if not torch.cuda.is_available():
        print("CUDA/HIP device required.")
        return 2
    if not HAVE_UA:
        print(f"Skipping unified_attention_fwd example: {UA_IMPORT_ERR}")
    else:
        example_unified_attention_fwd()
    if not HAVE_MHA:
        print(f"Skipping FMHA examples: {MHA_IMPORT_ERR}")
        return 0
    example_mha_fwd()
    example_mha_varlen_fwd()
    example_mha_batch_prefill()
    gfx = _gfx_arch()
    if gfx not in ("gfx942", "gfx950"):
        print(f"=== fmha_v3_fwd / fmha_v3_varlen_fwd ===\n  skipped: gfx942/gfx950 only (got {gfx})\n")
    else:
        example_fmha_v3_fwd()
        example_fmha_v3_varlen_fwd()
    return 0


# ============================================================================
# Verification
# ============================================================================


def reference_attention(t: CaseTensors) -> torch.Tensor:
    """Per-sequence reference attention using upcast fp32 SDPA on packed
    contiguous K/V (`k_3d`, `v_3d`).  Returns shape [total_q, h_q, d].
    """
    case = t.case
    out = torch.empty_like(t.q_3d, dtype=torch.float32)

    h_q, h_kv, d = case.num_q_heads, case.num_kv_heads, case.head_size
    nqpkv = case.num_queries_per_kv

    cu_q = t.cu_seqlens_q.tolist()
    cu_k = t.cu_seqlens_k.tolist()

    for s in range(case.batch):
        q_start, q_end = cu_q[s], cu_q[s + 1]
        k_start, k_end = cu_k[s], cu_k[s + 1]
        q_len = q_end - q_start
        kv_len = k_end - k_start

        q = t.q_3d[q_start:q_end].float()           # [q_len, h_q, d]
        k = t.k_3d[k_start:k_end].float()           # [kv_len, h_kv, d]
        v = t.v_3d[k_start:k_end].float()           # [kv_len, h_kv, d]

        # repeat KV heads to match Q heads (GQA)
        if nqpkv > 1:
            k = k.repeat_interleave(nqpkv, dim=1)   # [kv_len, h_q, d]
            v = v.repeat_interleave(nqpkv, dim=1)

        # [h_q, q_len, kv_len]
        scores = torch.einsum("qhd,khd->hqk", q, k) * t.softmax_scale

        if case.causal:
            # bottom-right alignment
            i = torch.arange(q_len, device=q.device).view(-1, 1)
            j = torch.arange(kv_len, device=q.device).view(1, -1)
            mask = j > (i + (kv_len - q_len))
            scores.masked_fill_(mask, float("-inf"))

        attn = torch.softmax(scores, dim=-1)
        out[q_start:q_end] = torch.einsum("hqk,khd->qhd", attn, v).to(torch.float32)

    return out


def verify_case(case: BenchCase, atol: float = 5e-2, rtol: float = 5e-2) -> dict:
    """Run all eligible backends against a fp32 reference and report max abs/rel diff."""
    t = build_tensors(case)
    ref = reference_attention(t)

    results = {}

    # Unified attention
    if ua_skip_reason(case) is None:
        out = torch.empty_like(t.q_3d)
        try:
            unified_attention_fwd(
                out, t.q_3d, t.k_paged, t.v_paged,
                t.block_tables, t.seq_lens_k, t.cu_seqlens_q,
                mask_type=(2 if case.causal else 0), scale_s=t.softmax_scale,
                scale=1.0, scale_k=1.0, scale_v=1.0, scale_out=1.0,
            )
            torch.cuda.synchronize()
            diff = (out.float() - ref).abs()
            results["unified_attention"] = (
                diff.max().item(),
                (diff / ref.abs().clamp(min=1e-3)).max().item(),
            )
        except Exception as e:  # noqa: BLE001
            results["unified_attention"] = ("err", str(e))
    else:
        results["unified_attention"] = ("skip", ua_skip_reason(case))

    if not HAVE_MHA:
        return results

    # mha_fwd
    if mha_fwd_skip_reason(case) is None:
        try:
            out, *_ = mha_fwd(
                t.q_4d, t.k_4d, t.v_4d,
                0.0, t.softmax_scale, case.causal, -1, -1, 0,
                False, False,
            )
            torch.cuda.synchronize()
            out_flat = out.reshape(-1, case.num_q_heads, case.head_size)
            diff = (out_flat.float() - ref).abs()
            results["mha_fwd"] = (
                diff.max().item(),
                (diff / ref.abs().clamp(min=1e-3)).max().item(),
            )
        except Exception as e:  # noqa: BLE001
            results["mha_fwd"] = ("err", str(e))
    else:
        results["mha_fwd"] = ("skip", mha_fwd_skip_reason(case))

    # mha_varlen_fwd paged
    try:
        out, *_ = mha_varlen_fwd(
            t.q_3d, t.k_paged, t.v_paged,
            t.cu_seqlens_q, t.cu_seqlens_k,
            t.max_seqlen_q, t.max_seqlen_k, 0,
            0.0, t.softmax_scale, 0.0, False,
            case.causal, -1, -1, 0,
            False, False,
            block_table=t.block_tables,
        )
        torch.cuda.synchronize()
        diff = (out.float() - ref).abs()
        results["varlen_paged"] = (
            diff.max().item(),
            (diff / ref.abs().clamp(min=1e-3)).max().item(),
        )
    except Exception as e:  # noqa: BLE001
        results["varlen_paged"] = ("err", str(e))

    # mha_varlen_fwd non-paged
    try:
        out, *_ = mha_varlen_fwd(
            t.q_3d, t.k_3d, t.v_3d,
            t.cu_seqlens_q, t.cu_seqlens_k,
            t.max_seqlen_q, t.max_seqlen_k, 0,
            0.0, t.softmax_scale, 0.0, False,
            case.causal, -1, -1, 0,
            False, False,
        )
        torch.cuda.synchronize()
        diff = (out.float() - ref).abs()
        results["varlen_nopage"] = (
            diff.max().item(),
            (diff / ref.abs().clamp(min=1e-3)).max().item(),
        )
    except Exception as e:  # noqa: BLE001
        results["varlen_nopage"] = ("err", str(e))

    # mha_batch_prefill
    try:
        out_list = mha_batch_prefill_func(
            t.q_3d, t.k_paged, t.v_paged,
            t.cu_seqlens_q, t.kv_indptr, t.kv_page_indices,
            t.max_seqlen_q, t.max_seqlen_k,
            dropout_p=0.0, softmax_scale=t.softmax_scale,
            causal=case.causal,
            kv_last_page_lens=t.kv_last_page_lens,
        )
        out = out_list[0] if isinstance(out_list, (list, tuple)) else out_list
        torch.cuda.synchronize()
        diff = (out.float() - ref).abs()
        results["batch_prefill"] = (
            diff.max().item(),
            (diff / ref.abs().clamp(min=1e-3)).max().item(),
        )
    except Exception as e:  # noqa: BLE001
        results["batch_prefill"] = ("err", str(e))

    return results


# ============================================================================
# Main
# ============================================================================


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Benchmark CK Unified Attention vs CK FMHA family.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--head-size", type=int, default=128, choices=[64, 128])
    ap.add_argument("--model", default=None, choices=sorted(MODELS),
                    help="Use real-model attention shape preset "
                         "(overrides --head-size).  Shapes sourced from "
                         "op_tests/op_benchmarks/triton/model_benchmarking_tool/"
                         "model_shapes.json.")
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16"])
    ap.add_argument("--cases", nargs="*", default=None,
                    help="Run only cases whose tag contains any of these substrings")
    ap.add_argument("--verify", action="store_true",
                    help="Run accuracy verification against fp32 reference")
    ap.add_argument("--examples", action="store_true",
                    help="Run a minimal example invocation for each backend and exit")
    ap.add_argument("--json", default=None, help="Write results to JSON file")
    args = ap.parse_args()

    if args.examples:
        return show_examples()

    if not torch.cuda.is_available():
        print("CUDA/HIP device required.")
        return 2

    if not HAVE_UA:
        print(f"WARNING: unified_attention_fwd unavailable: {UA_IMPORT_ERR}")
    if not HAVE_MHA:
        print(f"WARNING: mha module unavailable: {MHA_IMPORT_ERR}")
        return 2

    if args.model is not None:
        cases = make_model_cases(args.model)
        h_q, h_kv, d = MODELS[args.model]
        print(f"# model preset: {args.model}  "
              f"(num_q_heads={h_q}, num_kv_heads={h_kv}, head_size={d}, "
              f"GQA={h_q // h_kv})")
    else:
        cases = make_default_cases(args.head_size)
    for c in cases:
        c.dtype = args.dtype
    if args.cases:
        cases = [c for c in cases if any(sub in c.tag for sub in args.cases)]
        if not cases:
            print(f"No cases matched {args.cases}")
            return 1

    # ------------ Verification ------------
    if args.verify:
        print("=" * 80)
        print("ACCURACY VERIFICATION (vs fp32 reference)")
        print("=" * 80)
        any_fail = False
        for case in cases[:5]:  # verify a handful to keep it fast
            res = verify_case(case)
            print(f"\n{case.tag}")
            for backend, val in res.items():
                if isinstance(val[0], str):
                    print(f"  {backend:20s}  {val[0]}: {val[1]}")
                else:
                    abs_d, rel_d = val
                    ok = abs_d < 5e-2
                    if not ok:
                        any_fail = True
                    print(f"  {backend:20s}  max_abs={abs_d:.4e}  max_rel={rel_d:.4e}  {'OK' if ok else 'HIGH-DIFF'}")
        print()
        if any_fail:
            print("*** Some backends had high diff; benchmark numbers may still be valid ***\n")

    # ------------ Benchmark sweep ------------
    print(f"warmup={args.warmup} iters={args.iters} dtype={args.dtype} head_size={args.head_size}")
    print()
    headers = ["case"] + [name for name, _, _ in BACKENDS] + ["best", "UA/best"]
    col_w = [42, 11, 11, 11, 11, 11, 11, 11, 9, 9]
    sep = " ".join("-" * w for w in col_w)

    line = " ".join(f"{h:>{w}s}" for h, w in zip(headers, col_w))
    line = f"{headers[0]:<{col_w[0]}s}" + line[col_w[0]:]
    print(line)
    print(sep)

    all_records = []
    for case in cases:
        try:
            t = build_tensors(case)
        except Exception as e:  # noqa: BLE001
            print(f"{case.tag:<{col_w[0]}s}  build error: {e}")
            continue

        row_times: list[Optional[float]] = []
        skip_reasons: list[Optional[str]] = []

        for name, runner, skip_check in BACKENDS:
            reason = skip_check(case)
            if reason is not None:
                row_times.append(None)
                skip_reasons.append(reason)
                continue
            try:
                ms = runner(t, args.warmup, args.iters)
                row_times.append(ms)
                skip_reasons.append(None)
            except Exception as e:  # noqa: BLE001
                row_times.append(None)
                short = type(e).__name__
                skip_reasons.append(f"err:{short}")

        valid = [m for m in row_times if m is not None]
        best = min(valid) if valid else None
        ua_ms = row_times[0]
        ratio = (ua_ms / best) if (ua_ms is not None and best is not None and best > 0) else None

        cells = [f"{case.tag:<{col_w[0]}s}"]
        for ms, reason, w in zip(row_times, skip_reasons, col_w[1:1 + len(BACKENDS)]):
            if ms is not None:
                cells.append(f"{ms:>{w}.4f}")
            else:
                # show short skip token
                short = (reason or "skip")
                if len(short) > w - 1:
                    short = short[: w - 1]
                cells.append(f"{short:>{w}s}")
        cells.append(f"{best:>{col_w[-2]}.4f}" if best is not None else f"{'n/a':>{col_w[-2]}s}")
        if ratio is not None:
            cells.append(f"{ratio:>{col_w[-1] - 1}.3f}x")
        else:
            cells.append(f"{'n/a':>{col_w[-1]}s}")
        print(" ".join(cells))

        all_records.append({
            "case": case.tag,
            "shape": {
                "batch": case.batch,
                "num_kv_heads": case.num_kv_heads,
                "num_queries_per_kv": case.num_queries_per_kv,
                "head_size": case.head_size,
                "page_blk_size": case.page_blk_size,
                "query_lens": case.query_lens,
                "kv_lens": case.kv_lens,
                "dtype": case.dtype,
                "causal": case.causal,
            },
            "ms": {name: row_times[i] for i, (name, _, _) in enumerate(BACKENDS)},
            "skip": {name: skip_reasons[i] for i, (name, _, _) in enumerate(BACKENDS)},
            "best_ms": best,
            "ua_over_best": ratio,
        })

    print()
    print("Times in ms.  UA/best: <1.0 means UA wins.  'skip:...' means backend doesn't support the shape.")

    if args.json:
        with open(args.json, "w") as f:
            json.dump({
                "args": vars(args),
                "results": all_records,
            }, f, indent=2)
        print(f"Wrote {args.json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
