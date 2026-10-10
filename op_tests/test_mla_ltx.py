# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""MLA decode: large page_id, KV byte offset, and >4 GiB pools (gfx942/gfx950 asm).

Compares torch golden vs mla_decode_fwd.
Follows op_tests/test_quant.py layout (aiter-op-test SKILL).

Examples:
  python op_tests/test_mla_ltx.py
    # default on gfx942: every GFX942_PRESETS row in each of its modes (boundary+split-KV)
    # default on gfx950: PR global-load .co sweep (8 presets, ps, boundary+page16m)
  python op_tests/test_mla_ltx.py --suites kv_address --ps nps
    # gfx942: BF16 16/128 heads, stage-1 only; use --ps ps for persistent only (8 cases each)
  python op_tests/test_mla_ltx.py --preset qh16_fp8_q4 qh16_bf16_q8 --ps nps prefill
    # gfx942: the named presets in the named modes (nps / ps / prefill)
  python op_tests/test_mla_ltx.py --preset qh64_fp8_q1 --ps ps --lse off --page-base 19000000 --ctx 4
  python op_tests/test_mla_ltx.py --preset qh16_fp8_q1 qh64_bf16_q1 --suites boundary --ps ps --lse off
  python op_tests/test_mla_ltx.py --suites page16m -d fp8 -kvd fp8 -n 16,1 --ps ps --lse off --ctx 4
  MLA_PAGE_OOB_NUM_PAGES=3800000 python op_tests/test_mla_ltx.py --suites over4g
"""

from __future__ import annotations

import argparse
import itertools
import os
import sys
from collections.abc import Iterable
from dataclasses import dataclass

import pandas as pd
import torch

import aiter
from aiter import dtypes
from aiter.jit.utils.chip_info import get_gfx
from aiter.test_common import benchmark, checkAllclose, run_perftest

torch.set_default_device("cuda")

SUPPORTED_GFX = ["gfx942", "gfx950"]

# --- Fixed MLA layout (decode absorb, page_size=1) ---
KV_LORA_RANK = 512
QK_ROPE_HEAD_DIM = 64
QK_HEAD_DIM = KV_LORA_RANK + QK_ROPE_HEAD_DIM
V_HEAD_DIM = KV_LORA_RANK
PAGE_SIZE = 1
NHEAD_KV = 1
DECODE_QLEN = 1
BATCH_SIZE = 1
SUB_KV_TILE = 128
SAFE_PAGE_BASE = 1_000
PAGE_ID_16M = 16_000_000
SEED_CHUNK_PAGES = 262_144
SEQ_PAGE_BASE = -1  # sequential kv_indices 0..ctx-1

# Built in main(); read by @benchmark test_mla_ltx.
_KV_POOL: tuple[torch.Tensor, int] | None = None


def _parse_nhead_decode_qlen(value: int | tuple | str) -> tuple[int, int]:
    if isinstance(value, str):
        value = dtypes.str2tuple(value)
    if isinstance(value, int):
        return value, 1
    if isinstance(value, tuple):
        if len(value) == 1:
            return int(value[0]), 1
        if len(value) >= 2:
            return int(value[0]), int(value[1])
    raise ValueError(f"invalid -n / --nhead value: {value!r}")


def _csv_type_name(dtype) -> str:
    if dtype == dtypes.fp8:
        return "fp8"
    if dtype in (dtypes.bf16, torch.bfloat16):
        return "bf16"
    raise ValueError(f"unsupported dtype: {dtype}")


def _dtype_element_size(dtype) -> int:
    return 1 if dtype == dtypes.fp8 else 2


@dataclass
class PointCase:
    page_base: int
    ctx_len: int
    label: str
    suite: str
    max_split: int = 1


@dataclass
class Harness:
    q_dtype: torch.dtype
    kv_dtype: torch.dtype
    nhead: int
    decode_qlen: int = DECODE_QLEN

    @property
    def use_fp8(self) -> bool:
        return self.q_dtype == dtypes.fp8 and self.kv_dtype == dtypes.fp8

    @property
    def bytes_per_page(self) -> int:
        return QK_HEAD_DIM * _dtype_element_size(self.kv_dtype)

    def summary(self) -> str:
        q = _csv_type_name(self.q_dtype)
        kv = _csv_type_name(self.kv_dtype)
        return f"q={q} kv={kv} nhead={self.nhead} decode_qlen={self.decode_qlen}"

    def page_id_last_safe_2g(self) -> int:
        return ((1 << 31) - 1) // self.bytes_per_page

    def page_id_first_over_2g(self) -> int:
        return self.page_id_last_safe_2g() + 1

    def page_id_last_safe_4g(self) -> int:
        return ((1 << 32) - 1) // self.bytes_per_page

    def page_id_first_over_4g(self) -> int:
        return self.page_id_last_safe_4g() + 1

    def mega_ctx_len(self) -> int:
        return self.page_id_first_over_4g() + 16


# (q_dtype, kv_dtype, nhead/Gqa, decode_qlen/qSeqLen)
# Decode absorb rows from hsa/gfx950/mla/mla_asm.csv (prefill=0, causal=1, cprr=0).
# causal=0 rows are the msk0 non-masked builds, which this harness does not cover.
PresetConfig = tuple[torch.dtype, torch.dtype, int, int]

PRESETS: dict[str, PresetConfig] = {
    # fp8 / fp8 (mla_a8w8_*)
    "qh8_fp8_q1": (dtypes.fp8, dtypes.fp8, 8, 1),
    "qh16_fp8_q1": (dtypes.fp8, dtypes.fp8, 16, 1),
    "qh16_fp8_q2": (dtypes.fp8, dtypes.fp8, 16, 2),
    "qh16_fp8_q4": (dtypes.fp8, dtypes.fp8, 16, 4),
    "qh32_fp8_q1": (dtypes.fp8, dtypes.fp8, 32, 1),
    "qh32_fp8_q2": (dtypes.fp8, dtypes.fp8, 32, 2),
    "qh32_fp8_q4": (dtypes.fp8, dtypes.fp8, 32, 4),
    "qh64_fp8_q1": (dtypes.fp8, dtypes.fp8, 64, 1),
    "qh8_fp8_q4": (dtypes.fp8, dtypes.fp8, 8, 4),
    # bf16 / bf16 (mla_a16w16_* / mla_dec_stage1_*)
    "qh8_bf16_q1": (dtypes.bf16, dtypes.bf16, 8, 1),
    "qh8_bf16_q2": (dtypes.bf16, dtypes.bf16, 8, 2),
    "qh16_bf16_q1": (dtypes.bf16, dtypes.bf16, 16, 1),
    "qh16_bf16_q4": (dtypes.bf16, dtypes.bf16, 16, 4),
    "qh16_bf16_q8": (dtypes.bf16, dtypes.bf16, 16, 8),
    "qh32_bf16_q4": (dtypes.bf16, dtypes.bf16, 32, 4),
    "qh64_bf16_q1": (dtypes.bf16, dtypes.bf16, 64, 1),
    "qh128_bf16_q1": (dtypes.bf16, dtypes.bf16, 128, 1),
    # legacy aliases
    "qh16_fp8": (dtypes.fp8, dtypes.fp8, 16, 1),
    "qh64_bf16": (dtypes.bf16, dtypes.bf16, 64, 1),
    # gfx942 only (hsa/gfx942/mla/mla_asm.csv)
    "qh128_fp8_q1": (dtypes.fp8, dtypes.fp8, 128, 1),
    "qh16_a16w8_q1": (dtypes.bf16, dtypes.fp8, 16, 1),
    "qh32_bf16_q1": (dtypes.bf16, dtypes.bf16, 32, 1),
    "qh32_bf16_q2": (dtypes.bf16, dtypes.bf16, 32, 2),
    "qh64_bf16_q2": (dtypes.bf16, dtypes.bf16, 64, 2),
    "qh128_bf16_q8": (dtypes.bf16, dtypes.bf16, 128, 8),
}
DEFAULT_PRESET = "qh16_fp8_q1"

# gfx942 kv_address coverage: preset -> modes, with the .co each mode selects.
# nps = stage-1 split-KV, ps = persistent, prefill = mla_prefill_fwd.
# Not covered: mla_a16w8_qh16_m16x4_n16x1_coex0_mask1_ps_page64_ds32 (page_size 64).
GFX942_PRESETS: dict[str, tuple[str, ...]] = {
    # nps: mla_dec_stage1_bf16_a16w16_subQ16_mqa16, ps: mla_a16w16_qh16_m16x4_n16x1_coex0_mask1_ps
    "qh16_bf16_q1": ("nps", "ps"),
    # nps: mla_dec_stage1_bf16_a16w16_subQ128_mqa128, ps: folded to 16 heads (as qh16_bf16_q1)
    "qh128_bf16_q1": ("nps", "ps"),
    # nps: mla_a16w16_qh16_m16x4_n16x1_coex0_mask1
    "qh16_bf16_q4": ("nps",),
    # nps: mla_a16w16_qh16_m32x4_n16x1_coex0_mask1, prefill: mla_pfl_bf16_a16w16_causal_subQ16_mqa16
    "qh16_bf16_q8": ("nps", "prefill"),
    # prefill: mla_pfl_bf16_a16w16_causal_subQ128_mqa128
    "qh128_bf16_q8": ("prefill",),
    # nps: MLA_A16W16_1TG_4W_32mx1_16nx1_Coex0_Msk1_QH16_Q1
    "qh32_bf16_q1": ("nps",),
    # nps: MLA_A16W16_1TG_4W_32mx1_16nx1_Coex0_Msk1_QH16
    "qh32_bf16_q2": ("nps",),
    # nps: MLA_A16W16_1TG_4W_64mx1_16nx1_Coex0_Msk1_QH16
    "qh64_bf16_q2": ("nps",),
    # ps: mla_dec_stage1_bf16_a16w16_subQ16_mqa16_ps
    "qh8_bf16_q2": ("ps",),
    # ps: mla_a16w8_qh16_m16x4_n16x1_coex0_mask1_ps
    "qh16_a16w8_q1": ("ps",),
    # nps: mla_a8w8_qh16_qseqlen1_gqaratio16, ps: mla_a8w8_qh16_qseqlen1_gqaratio16_ps
    "qh16_fp8_q1": ("nps", "ps"),
    # nps: mla_a8w8_qh16_qseqlen2_gqaratio16, ps: mla_a8w8_qh16_qseqlen2_gqaratio16_ps
    "qh16_fp8_q2": ("nps", "ps"),
    # nps: mla_a8w8_qh64_qseqlen4_gqaratio16, ps: mla_a8w8_qh64_qseqlen4_gqaratio16_ps
    "qh16_fp8_q4": ("nps", "ps"),
    # nps: mla_a8w8_qh128_m32x4_n16x2_msk1, ps: mla_a8w8_qh128_m32x4_n16x2_msk0_ps
    "qh128_fp8_q1": ("nps", "ps"),
}
GFX942_MODES = ("nps", "ps", "prefill")

# PR #4452 global-load .co: decode presets reachable via ps+lse off (skip cprr;
# skip qh16_fp8_q4 nps -> unrefreshed qh64_qseqlen4 alias in mla_asm.csv).
GLOBAL_LOAD_PRESETS: tuple[str, ...] = (
    "qh32_bf16_q4",
    "qh64_bf16_q1",
    "qh16_fp8_q1",
    "qh16_fp8_q4",
    "qh32_fp8_q2",
    "qh32_fp8_q4",
    "qh8_fp8_q4",
    "qh64_fp8_q1",
)


def _default_ctx(h: Harness) -> int:
    """Kernels need ctx >= decode_qlen; use 4 for q1/q2 smoke on ltx suites."""
    return max(4, h.decode_qlen)


def apply_config(
    q_dtype: torch.dtype,
    kv_dtype: torch.dtype,
    nhead: int,
    decode_qlen: int = DECODE_QLEN,
) -> None:
    if q_dtype == dtypes.fp8 and kv_dtype == dtypes.bf16:
        raise ValueError(
            "fp8 Q with bf16 KV is not supported (see test_mla.py check_support)"
        )
    global HARNESS
    HARNESS = Harness(q_dtype, kv_dtype, nhead, decode_qlen)


_q0, _kv0, _n0, _ql0 = PRESETS[DEFAULT_PRESET]
HARNESS = Harness(_q0, _kv0, _n0, _ql0)


def _point_cases_for(h: Harness) -> list[PointCase]:
    p2g = h.page_id_last_safe_2g()
    p2g1 = h.page_id_first_over_2g()
    p4g = h.page_id_last_safe_4g()
    p4g1 = h.page_id_first_over_4g()
    ctx = _default_ctx(h)
    return [
        PointCase(SAFE_PAGE_BASE, ctx, "below_2g_offset", "boundary"),
        PointCase(p2g, ctx, "edge_2g_last_safe", "boundary"),
        PointCase(p2g1, ctx, "edge_2g_first_overflow", "boundary"),
        PointCase(p4g, ctx, "edge_4g_last_safe", "boundary"),
        PointCase(p4g1, ctx, "edge_4g_first_overflow", "boundary"),
        PointCase(p4g1, 16, "seq16_from_4g_overflow", "boundary"),
        PointCase(p4g1, SUB_KV_TILE, "ctx128_at_4g_boundary", "over4g"),
        PointCase(65_409, SUB_KV_TILE, "cross_window_65536_subkv", "pa_window"),
        PointCase(PAGE_ID_16M, ctx, "page_id_16m", "page16m"),
        PointCase(PAGE_ID_16M, SUB_KV_TILE, "page_id_16m_ctx128", "page16m"),
        PointCase(1 << 24, ctx, "page_id_2p24", "page16m"),
        PointCase(19_000_000, ctx, "page_id_19m", "page16m"),
    ]


def _kv_address_cases(h: Harness) -> list[PointCase]:
    # A BF16 row starts below 4 GiB but straddles it at page 3,728,270 (FP8: 7,456,540).
    # The last query token sees every row, so straddling4g ends on that row.
    boundary = (1 << 32) // h.bytes_per_page
    ctx = h.decode_qlen
    return [
        PointCase(boundary - ctx, ctx, "below4g", "kv_address"),
        PointCase(boundary - ctx + 1, ctx, "straddling4g", "kv_address"),
        PointCase(boundary + 1, ctx, "above4g", "kv_address"),
        PointCase(boundary + 1, 256, "above4g-splitkv", "kv_address", max_split=4),
    ]


_POINT_CASES = _point_cases_for(HARNESS)
_SEQUENTIAL_CASES = [(HARNESS.mega_ctx_len(), "sequential_mega_over4g", "mega")]


def ref_masked_attention(query, key, value, scale, dtype, is_causal=True):
    w = torch.einsum("qhd,khd->hqk", query.float(), key.float()) * scale
    if is_causal:
        s_q, s_k = query.shape[0], key.shape[0]
        bias = torch.zeros(s_q, s_k, dtype=query.dtype)
        mask = torch.ones(s_q, s_k, dtype=torch.bool).tril(diagonal=s_k - s_q)
        bias.masked_fill_(mask.logical_not(), float("-inf"))
        w += bias
    w = torch.softmax(w, dim=-1)
    return torch.einsum("hqk,khd->qhd", w.float(), value.float()).to(dtype)


def torch_mla_extend(
    q, kvc_cache, qo_indptr, kv_indptr, kv_indices, sm_scale, dtype, is_causal=True
):
    qs = torch.tensor_split(q, qo_indptr.tolist()[1:])
    kvc = torch.index_select(kvc_cache, 0, kv_indices)
    kvs = torch.tensor_split(kvc, kv_indptr.tolist()[1:])
    outs = []
    for i in range(qo_indptr.shape[0] - 1):
        kvc_i, q_i = kvs[i], qs[i]
        v, _ = torch.split(kvc_i, [KV_LORA_RANK, QK_ROPE_HEAD_DIM], dim=-1)
        outs.append(ref_masked_attention(q_i, kvc_i, v, sm_scale, dtype, is_causal))
    return torch.concat(outs)


def run_torch_mla_decode(
    q,
    kv_buffer,
    qo_indptr,
    kv_indptr,
    kv_indices,
):
    sm = 1.0 / (QK_HEAD_DIM**0.5)
    kv_bf16 = (
        kv_buffer.to(torch.bfloat16) if HARNESS.kv_dtype == dtypes.fp8 else kv_buffer
    )
    return torch_mla_extend(
        q, kv_bf16, qo_indptr, kv_indptr, kv_indices, sm, torch.bfloat16, True
    )


def _seed_pages(kv_buffer: torch.Tensor, ranges: Iterable[tuple[int, int]]) -> None:
    for base, length in ranges:
        if length <= 4096:
            for pid in range(base, base + length):
                gen = torch.Generator(device="cuda")
                gen.manual_seed(pid & 0xFFFFFFFF)
                kv_buffer[pid].copy_(
                    torch.randn(1, QK_HEAD_DIM, device="cuda", generator=gen) * 0.02
                )
            continue
        gen = torch.Generator(device="cuda")
        gen.manual_seed((base & 0xFFFFFFFF) ^ (length & 0xFFFFFFFF))
        for start in range(base, base + length, SEED_CHUNK_PAGES):
            end = min(start + SEED_CHUNK_PAGES, base + length)
            n = end - start
            kv_buffer[start:end].copy_(
                torch.randn(n, 1, QK_HEAD_DIM, device="cuda", generator=gen) * 0.02
            )


def _build_kv_pool(num_pages: int, ranges: list[tuple[int, int]]) -> torch.Tensor:
    kv = torch.zeros(
        (num_pages, NHEAD_KV, QK_HEAD_DIM), dtype=HARNESS.kv_dtype, device="cuda"
    )
    _seed_pages(kv, ranges)
    return kv


def _build_persistent_metadata(
    qo_indptr, kv_indptr, kv_last_page_lens, max_split_per_batch, *_legacy
):
    bs = qo_indptr.shape[0] - 1
    # gfx942 plans FP8 128 heads unfolded only with the real dtypes; gfx950 keeps
    # the BF16 metadata its sweep was validated with.
    if get_gfx() == "gfx942":
        dtype_q, dtype_kv = HARNESS.q_dtype, HARNESS.kv_dtype
    else:
        dtype_q = dtype_kv = dtypes.bf16
    sizes = aiter.get_mla_metadata_info_v1(
        bs,
        HARNESS.decode_qlen,
        HARNESS.nhead,
        dtype_q,
        dtype_kv,
        is_sparse=False,
        fast_mode=True,
        num_kv_splits=max_split_per_batch,
        intra_batch_mode=False,
    )

    def buf(i):
        n, t = sizes[i]
        return torch.empty(n, dtype=t, device="cuda")

    wmd, wi, wis, ri, rfm, rpm = (buf(i) for i in range(6))
    aiter.get_mla_metadata_v1(
        qo_indptr,
        kv_indptr,
        kv_last_page_lens,
        HARNESS.nhead // NHEAD_KV,
        NHEAD_KV,
        False,
        wmd,
        wis,
        wi,
        ri,
        rfm,
        rpm,
        page_size=PAGE_SIZE,
        kv_granularity=max(PAGE_SIZE, 16),
        max_seqlen_qo=HARNESS.decode_qlen,
        uni_seqlen_qo=HARNESS.decode_qlen,
        fast_mode=True,
        max_split_per_batch=max_split_per_batch,
        intra_batch_mode=False,
        dtype_q=dtype_q,
        dtype_kv=dtype_kv,
    )
    return {
        "work_meta_data": wmd,
        "work_indptr": wi,
        "work_info_set": wis,
        "reduce_indptr": ri,
        "reduce_final_map": rfm,
        "reduce_partial_map": rpm,
    }


def _make_indptr(ctx_len: int, page_base: int | None):
    qlen = HARNESS.decode_qlen
    kv_indptr = torch.tensor([0, ctx_len], dtype=torch.int, device="cuda")
    if page_base is None:
        kv_indices = torch.arange(ctx_len, dtype=torch.int, device="cuda")
    else:
        kv_indices = torch.arange(
            page_base, page_base + ctx_len, dtype=torch.int, device="cuda"
        )
    qo_indptr = torch.tensor([0, qlen], dtype=torch.int, device="cuda")
    return qo_indptr, kv_indptr, kv_indices


def run_asm_mla_decode(
    q,
    kv_buffer,
    num_pages,
    out,
    qo_indptr,
    kv_indptr,
    kv_indices,
    *,
    persistent: bool,
    return_lse: bool,
    max_split: int,
):
    sm = 1.0 / (QK_HEAD_DIM**0.5)
    kv_lens = torch.ones(BATCH_SIZE, dtype=torch.int, device="cuda")
    kv_view = kv_buffer.view(num_pages, PAGE_SIZE, NHEAD_KV, QK_HEAD_DIM)
    q_asm = q.to(dtypes.fp8) if HARNESS.use_fp8 else q
    kw = {
        "page_size": PAGE_SIZE,
        "nhead_kv": NHEAD_KV,
        "sm_scale": sm,
        "return_lse": return_lse,
    }
    if HARNESS.use_fp8:
        kw["q_scale"] = torch.ones(1, dtype=torch.float, device="cuda")
    if HARNESS.kv_dtype == dtypes.fp8:
        kw["kv_scale"] = torch.ones(1, dtype=torch.float, device="cuda")
    if persistent:
        kw["num_kv_splits"] = max_split
        kw.update(_build_persistent_metadata(qo_indptr, kv_indptr, kv_lens, max_split))
    elif get_gfx() == "gfx942":
        # Preserve the requested splits even for short regression sequences.
        kw["num_kv_splits"] = max_split
        kw["num_kv_splits_indptr"] = torch.tensor(
            [0, max_split], dtype=torch.int, device="cuda"
        )
    aiter.mla.mla_decode_fwd(
        q_asm,
        kv_view,
        out,
        qo_indptr,
        kv_indptr,
        kv_indices,
        kv_lens,
        HARNESS.decode_qlen,
        g_kv_indptr=None,
        cp_world_size=1,
        cp_rank=0,
        **kw,
    )
    return out


def run_asm_mla_prefill(q, kv_buffer, num_pages, out, qo_indptr, kv_indptr, kv_indices):
    kv_lens = torch.ones(BATCH_SIZE, dtype=torch.int, device="cuda")
    aiter.mla.mla_prefill_fwd(
        q,
        kv_buffer.view(num_pages, PAGE_SIZE, NHEAD_KV, QK_HEAD_DIM),
        out,
        qo_indptr,
        kv_indptr,
        kv_indices,
        kv_lens,
        HARNESS.decode_qlen,
        sm_scale=1.0 / (QK_HEAD_DIM**0.5),
    )
    return out


def _page_offset(page_id: int) -> int:
    return page_id * HARNESS.bytes_per_page


def _pool_bytes(num_pages: int) -> int:
    return num_pages * HARNESS.bytes_per_page


def _need_num_pages(
    points: list[PointCase], seq: list[tuple[int, str]], ctx_override: int
) -> int:
    need = 10_000
    for c in points:
        ctx = ctx_override if ctx_override > 0 else c.ctx_len
        if c.page_base >= 0:
            need = max(need, c.page_base + ctx)
    for ctx, _ in seq:
        need = max(need, ctx + 10_000)
    return need + 1


def _seed_ranges(
    points: list[PointCase], seq: list[tuple[int, str]], ctx_override: int
):
    ranges = []
    for c in points:
        ctx = ctx_override if ctx_override > 0 else c.ctx_len
        if c.page_base >= 0:
            ranges.append((c.page_base, ctx))
    for ctx, _ in seq:
        ranges.append((0, ctx))
    return ranges


def _filter_points(suites: list[str]) -> list[PointCase]:
    if not suites:
        return list(_POINT_CASES)
    want = set(suites)
    return [c for c in _POINT_CASES if c.suite in want]


def _decode_flops_bytes(ctx_len: int) -> tuple[int, int]:
    qlen = HARNESS.decode_qlen
    h = HARNESS.nhead
    d = QK_HEAD_DIM
    dv = V_HEAD_DIM
    flops = 2 * qlen * h * ctx_len * (d + dv)
    elem = _dtype_element_size(HARNESS.kv_dtype)
    nbytes = qlen * h * d * 2 + ctx_len * QK_HEAD_DIM * elem + qlen * h * dv * 2
    return flops, nbytes


@benchmark()
def test_mla_ltx(
    page_base,
    ctx_len,
    label,
    persistent,
    return_lse,
    max_split,
    q_dtype,
    kv_dtype,
    nhead,
    decode_qlen,
    check_kv_address=False,
    prefill=False,
):
    apply_config(q_dtype, kv_dtype, nhead, decode_qlen)
    if _KV_POOL is None:
        raise RuntimeError("KV pool not initialized; call from main() only")

    pool, num_pages = _KV_POOL
    if check_kv_address:
        return _check_kv_address(
            pool, page_base, ctx_len, label, persistent, return_lse, max_split, prefill
        )
    pb = None if page_base == SEQ_PAGE_BASE else page_base
    qo, kv_i, kv_x = _make_indptr(ctx_len, pb)

    qlen = HARNESS.decode_qlen
    gen = torch.Generator(device="cuda")
    gen.manual_seed(
        (page_base & 0xFFFF) ^ (ctx_len & 0xFFFF) ^ int(persistent) ^ int(return_lse)
    )
    q = torch.randn(
        (qlen, HARNESS.nhead, QK_HEAD_DIM), dtype=torch.bfloat16, generator=gen
    )
    ref = run_torch_mla_decode(q, pool, qo, kv_i, kv_x)
    out = torch.empty((qlen, HARNESS.nhead, V_HEAD_DIM), dtype=torch.bfloat16).fill_(-1)

    candidates = {
        "asm": lambda: run_asm_mla_decode(
            q,
            pool,
            num_pages,
            out,
            qo,
            kv_i,
            kv_x,
            persistent=bool(persistent),
            return_lse=bool(return_lse),
            max_split=max_split,
        ),
    }
    flops, nbytes = _decode_flops_bytes(ctx_len)
    max_pid = (page_base + ctx_len - 1) if page_base >= 0 else (ctx_len - 1)
    kv_off = _page_offset(max_pid)

    ret = {
        "gfx": get_gfx(),
        "label": label,
        "kv_off": kv_off,
        "off_gt_4g": int(kv_off >= (1 << 32)),
    }
    for name, fn in candidates.items():
        result, us = run_perftest(fn)
        err = checkAllclose(
            ref.to(dtypes.fp32),
            result.to(dtypes.fp32),
            rtol=3e-2,
            atol=3e-2,
            msg=f"{name}: mla_ltx {label}",
        )
        ret[f"{name} us"] = us
        ret[f"{name} TFLOPS"] = flops / us / 1e6
        ret[f"{name} TB/s"] = nbytes / us / 1e6
        ret[f"{name} err"] = err
    return ret


def _check_kv_address(
    pool, page_base, ctx_len, label, persistent, return_lse, max_split, prefill
):
    """ROCm/aiter#5826: compare identical KV data at low and high offsets."""
    mode = "prefill" if prefill else "persistent" if persistent else "stage1"
    qlen = HARNESS.decode_qlen
    q_name, kv_name = (_csv_type_name(d) for d in (HARNESS.q_dtype, HARNESS.kv_dtype))
    name = f"{label}-{mode}-{HARNESS.nhead}-{q_name}_{kv_name}-q{qlen}"
    generator = torch.Generator(device="cpu").manual_seed(42)
    q_cpu = torch.randn(
        qlen, HARNESS.nhead, QK_HEAD_DIM, generator=generator, device="cpu"
    ).to(HARNESS.q_dtype)
    kv_cpu = torch.randn(ctx_len, QK_HEAD_DIM, generator=generator, device="cpu").to(
        HARNESS.kv_dtype
    )
    # Independent of GPU page indexing, including the reference's KV reads.
    # Causal: query token t sees the first ctx_len - qlen + 1 + t rows.
    scores = (q_cpu.float() @ kv_cpu.float().T) * (QK_HEAD_DIM**-0.5)
    hidden = torch.ones(qlen, ctx_len, dtype=torch.bool, device="cpu")
    hidden = hidden.triu(ctx_len - qlen + 1)
    scores.masked_fill_(hidden[:, None, :], float("-inf"))
    reference = torch.softmax(scores, dim=-1) @ kv_cpu[:, :V_HEAD_DIM].float()
    # The FP8-Q kernels also round P to FP8.
    tol = 6e-2 if HARNESS.use_fp8 else 2e-2
    q = q_cpu.cuda()
    source = kv_cpu.cuda().view(ctx_len, NHEAD_KV, QK_HEAD_DIM)
    pool[SAFE_PAGE_BASE : SAFE_PAGE_BASE + ctx_len].copy_(source)
    placed = pool[page_base : page_base + ctx_len]
    placed.copy_(source)
    torch.testing.assert_close(
        placed.cpu().view_as(kv_cpu).float(), kv_cpu.float(), rtol=0, atol=0
    )

    ret = {"gfx": get_gfx(), "label": label, "passed": True}
    try:
        for control, pool_view, page in (
            ("low-offset control", pool, SAFE_PAGE_BASE),
            ("same-byte rebased control", placed, 0),
            ("pool-global page indices", pool, page_base),
        ):
            qo, kv_i, kv_x = _make_indptr(ctx_len, page)
            out = torch.full(
                (qlen, HARNESS.nhead, V_HEAD_DIM),
                float("nan"),
                dtype=torch.bfloat16,
                device="cuda",
            )
            if prefill:
                run_asm_mla_prefill(
                    q, pool_view, pool_view.shape[0], out, qo, kv_i, kv_x
                )
            else:
                run_asm_mla_decode(
                    q,
                    pool_view,
                    pool_view.shape[0],
                    out,
                    qo,
                    kv_i,
                    kv_x,
                    persistent=bool(persistent),
                    return_lse=bool(return_lse),
                    max_split=max_split,
                )
            torch.testing.assert_close(
                out.float().cpu(), reference, atol=tol, rtol=tol, msg=control
            )
    except AssertionError as exc:
        ret["passed"] = False
        ret["failure"] = str(exc)
        print(f"FAIL {name}: {exc}", flush=True)
    else:
        print(f"PASS {name}", flush=True)
    return ret


def _parse_persistent(values: list[str]) -> list[bool]:
    out: list[bool] = []
    for v in values:
        if v in ("ps", "1", "true", "True"):
            out.append(True)
        elif v in ("nps", "0", "false", "False"):
            out.append(False)
        else:
            raise ValueError(f"unknown persistent mode {v!r}")
    return out


def _parse_lse(values: list[str]) -> list[bool]:
    out: list[bool] = []
    for v in values:
        if v in ("on", "1", "true", "True"):
            out.append(True)
        elif v in ("off", "0", "false", "False"):
            out.append(False)
        else:
            raise ValueError(f"unknown lse mode {v!r}")
    return out


def _sweep_rows(
    points: list[PointCase],
    seq: list[tuple[int, str]],
    ctx_override: int,
    q_dtype,
    kv_dtype,
    nhead,
    decode_qlen,
    persistent_modes: list[bool],
    lse_modes: list[bool],
    max_splits: list[int] | None,
    prefill: bool = False,
) -> list[dict]:
    rows: list[dict] = []
    for c in points:
        ctx = ctx_override if ctx_override > 0 else c.ctx_len
        for persistent, lse, max_split in itertools.product(
            persistent_modes,
            lse_modes,
            max_splits if max_splits is not None else [c.max_split],
        ):
            rows.append(
                test_mla_ltx(
                    c.page_base,
                    ctx,
                    c.label,
                    int(persistent),
                    int(lse),
                    max_split,
                    q_dtype,
                    kv_dtype,
                    nhead,
                    decode_qlen,
                    check_kv_address=c.suite == "kv_address",
                    prefill=prefill,
                )
            )
    for ctx, label in seq:
        for persistent, lse, max_split in itertools.product(
            persistent_modes, lse_modes, max_splits if max_splits is not None else [1]
        ):
            rows.append(
                test_mla_ltx(
                    SEQ_PAGE_BASE,
                    ctx,
                    label,
                    int(persistent),
                    int(lse),
                    max_split,
                    q_dtype,
                    kv_dtype,
                    nhead,
                    decode_qlen,
                )
            )
    return rows


def main():
    gfx = get_gfx()
    if len(sys.argv) == 1 and gfx == "gfx950":
        sys.argv.extend(
            [
                "--preset",
                *GLOBAL_LOAD_PRESETS,
                "--suites",
                "boundary",
                "page16m",
                "--ctx",
                "4",
                "--lse",
                "off",
            ]
        )
    elif len(sys.argv) == 1 and gfx == "gfx942":
        sys.argv.extend(["--preset", *GFX942_PRESETS])

    if gfx not in SUPPORTED_GFX:
        aiter.logger.warning("mla_ltx unsupported on %s; skipping", gfx)
        return

    is_gfx942 = gfx == "gfx942"
    _dq, _dkv, _dn, _dql = PRESETS["qh16_bf16_q1" if is_gfx942 else DEFAULT_PRESET]
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="MLA decode large page_id / >4 GiB KV pool (gfx942/gfx950 asm)",
    )
    parser.add_argument(
        "--preset",
        nargs="*",
        default=[],
        choices=sorted(PRESETS.keys()),
        help="named decode configs (overrides -d/-kvd/-n when set)",
    )
    parser.add_argument(
        "--suites",
        "--suite",
        dest="suites",
        nargs="*",
        default=["kv_address"] if is_gfx942 else [],
        choices=["boundary", "over4g", "pa_window", "mega", "page16m"]
        + (["kv_address"] if is_gfx942 else []),
        help="case groups to sweep (empty = all groups)",
    )
    parser.add_argument(
        "-d",
        "--dtype",
        type=dtypes.str2Dtype,
        nargs="*",
        default=[_dq],
        choices=[dtypes.d_dtypes["bf16"], dtypes.d_dtypes["fp8"]],
        help="Q dtype list",
    )
    parser.add_argument(
        "-kvd",
        "--kv_dtype",
        type=dtypes.str2Dtype,
        nargs="*",
        default=[_dkv],
        choices=[dtypes.d_dtypes["bf16"], dtypes.d_dtypes["fp8"]],
        help="KV dtype list",
    )
    parser.add_argument(
        "-n",
        "--nhead",
        type=dtypes.str2tuple,
        nargs="*",
        default=[(16, 1), (128, 1)] if is_gfx942 else [(_dn, _dql)],
        help="nhead,decode_qlen tuples (same as test_mla.py -n)",
    )
    parser.add_argument(
        "--persistent",
        "--ps",
        dest="persistent",
        nargs="*",
        default=list(GFX942_MODES) if is_gfx942 else ["ps"],
        help="ps / nps sweep, plus prefill on gfx942 (default: all on gfx942, ps on gfx950)",
    )
    parser.add_argument(
        "--lse",
        nargs="*",
        default=["off"] if is_gfx942 else ["off", "on"],
        help="on / off sweep (default: off on gfx942, both on gfx950)",
    )
    parser.add_argument(
        "--ctx",
        type=int,
        nargs="*",
        default=[0],
        help="ctx override (0=use case default)",
    )
    parser.add_argument(
        "--page-base",
        type=int,
        nargs="*",
        default=[0],
        help="if set, single custom point sweep",
    )
    parser.add_argument(
        "--num-kv-splits",
        type=int,
        nargs="*",
        default=None,
        help="split counts (default: 4 for the kv_address split-KV case, 1 otherwise)",
    )
    parser.add_argument(
        "--mega-ctx",
        type=int,
        nargs="*",
        default=[],
        help="sequential ctx lengths (empty=default mega when suites include mega/all)",
    )
    args = parser.parse_args()

    run_prefill = is_gfx942 and "prefill" in args.persistent
    persistent_modes = _parse_persistent([v for v in args.persistent if v != "prefill"])
    lse_modes = _parse_lse(args.lse)
    ctx_overrides = args.ctx if args.ctx else [0]

    if args.preset:
        config_rows = [PRESETS[name] for name in args.preset]
    else:
        config_rows = [
            (q_dtype, kv_dtype, *_parse_nhead_decode_qlen(nhead_spec))
            for q_dtype, kv_dtype, nhead_spec in itertools.product(
                args.dtype, args.kv_dtype, args.nhead
            )
        ]

    gfx942_modes = {PRESETS[name]: modes for name, modes in GFX942_PRESETS.items()}
    if is_gfx942 and any(row not in gfx942_modes for row in config_rows):
        parser.error("gfx942 coverage requires a GFX942_PRESETS config")

    address_results = []
    for q_dtype, kv_dtype, nhead, decode_qlen in config_rows:
        row_persistent, row_prefill = persistent_modes, False
        if is_gfx942:
            modes = gfx942_modes[(q_dtype, kv_dtype, nhead, decode_qlen)]
            row_persistent = [
                p for p in persistent_modes if ("ps" if p else "nps") in modes
            ]
            row_prefill = run_prefill and "prefill" in modes
            if not row_persistent and not row_prefill:
                continue
        apply_config(q_dtype, kv_dtype, nhead, decode_qlen)
        global _POINT_CASES, _SEQUENTIAL_CASES, _KV_POOL
        _POINT_CASES = _point_cases_for(HARNESS)
        if is_gfx942:
            _POINT_CASES.extend(_kv_address_cases(HARNESS))
        _SEQUENTIAL_CASES = [(HARNESS.mega_ctx_len(), "sequential_mega_over4g", "mega")]

        points = _filter_points(args.suites)
        seq: list[tuple[int, str]] = []
        if args.mega_ctx:
            seq = [(c, f"sequential_ctx{c}") for c in args.mega_ctx]
        elif not args.suites or "mega" in args.suites:
            seq = [(c[0], c[1]) for c in _SEQUENTIAL_CASES]

        for page_base in args.page_base:
            if page_base > 0:
                for ctx_ov in ctx_overrides:
                    ctx = ctx_ov if ctx_ov > 0 else _default_ctx(HARNESS)
                    points = [
                        PointCase(page_base, ctx, f"page_id_{page_base}", "page16m")
                    ]
                    seq = []
                    break

        for ctx_override in ctx_overrides:
            needed_pages = _need_num_pages(points, seq, ctx_override)
            num_pages = int(
                os.environ.get(
                    "MLA_PAGE_OOB_NUM_PAGES",
                    str(needed_pages),
                )
            )
            if any(c.suite == "kv_address" for c in points):
                if num_pages < needed_pages:
                    parser.error(
                        f"kv_address requires at least {needed_pages} KV pages"
                    )
                free_bytes, _ = torch.cuda.mem_get_info()
                if free_bytes < _pool_bytes(num_pages) + (1 << 30):
                    aiter.logger.warning(
                        "SKIP kv_address: requires a 4 GiB pool plus 1 GiB of workspace"
                    )
                    continue
            aiter.logger.info(
                "mla_ltx config=%s suites=%s pages=%d pool_GiB=%.2f",
                HARNESS.summary(),
                args.suites,
                num_pages,
                _pool_bytes(num_pages) / 2**30,
            )
            try:
                _KV_POOL = (
                    _build_kv_pool(num_pages, _seed_ranges(points, seq, ctx_override)),
                    num_pages,
                )
            except torch.cuda.OutOfMemoryError as e:
                aiter.logger.warning("mla_ltx OOM building pool: %s", e)
                continue

            rows = _sweep_rows(
                points,
                seq,
                ctx_override,
                q_dtype,
                kv_dtype,
                nhead,
                decode_qlen,
                row_persistent,
                lse_modes,
                args.num_kv_splits,
            )
            if row_prefill:
                rows += _sweep_rows(
                    [c for c in points if c.suite == "kv_address"],
                    [],
                    ctx_override,
                    q_dtype,
                    kv_dtype,
                    nhead,
                    decode_qlen,
                    [False],
                    [False],
                    args.num_kv_splits,
                    prefill=True,
                )
            address_results.extend(row for row in rows if "passed" in row)
            df = pd.DataFrame(rows)
            aiter.logger.info(
                "mla_ltx summary (markdown):\n%s", df.to_markdown(index=False)
            )
            del _KV_POOL
            _KV_POOL = None
            torch.cuda.empty_cache()

    if address_results:
        failed = sum(not row["passed"] for row in address_results)
        print(
            f"gfx942 KV addressing: {len(address_results) - failed} passed, {failed} failed"
        )
        if failed:
            raise AssertionError(f"{failed} KV-address cases failed")


if __name__ == "__main__":
    main()
