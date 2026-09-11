# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Compare gfx1250 page-size-1 persistent decode against page-size-64 ASM on
one shared logical KV cache. Both backends call ``aiter.mla.mla_decode_fwd``
(the ps1 path uses ``AITER_MLA_DECODE_PS1_FLYDSL=1`` for FlyDSL stage-1).

nhead=16 supports q_seq=1/2/4; nhead=32/64/128 support q_seq=1 only on the
persistent kernel (ASM PS64 uses the same logical head count via split-KV).

The two kernels cannot share a KV buffer: page-size-1 stores one token per page
([nope|rope] interleaved, 576 B per page), while page-size-64 stores 64 tokens
per page as [nope_block | rope_block]. This test generates the logical KV once
and packs it into both layouts, so the backends see identical values and their
outputs are directly comparable against a single torch reference.

Two batch shapes:
  定长 (default)  every sequence in the batch carries the same ctx_len. Use a
                  ctx_len that is a multiple of 64 to keep both backends off
                  their tail-masking paths and isolate the layout/pipeline
                  differences.
  变长 (--varlen) sequence lengths are drawn per batch entry. Both backends see
                  the same lengths, but they react differently: the persistent
                  path's planner spreads uneven work across CUs, while the ASM
                  path's static grid gives every sequence the same split budget,
                  so short sequences leave their blocks idle.

Examples:
  python3 op_tests/test_mla_ps1_persistent_vs_asm_ps64.py
  python3 op_tests/test_mla_ps1_persistent_vs_asm_ps64.py -q 1 2 4 -b 4 -c 2048 4096
  python3 op_tests/test_mla_ps1_persistent_vs_asm_ps64.py -n 128 -q 1 -b 4 -c 2048 4096
  python3 op_tests/test_mla_ps1_persistent_vs_asm_ps64.py -n 32 64 -q 1 --scales poc
  python3 op_tests/test_mla_ps1_persistent_vs_asm_ps64.py -b 32 -c 8192 --varlen
  python3 op_tests/test_mla_ps1_persistent_vs_asm_ps64.py --varlen --varlen-min-ratio 0.1
"""

import argparse
import itertools
import os

import pandas as pd
import torch

import aiter
from aiter import dtypes
from aiter.jit.utils.chip_info import get_gfx
from aiter.mla import get_meta_param, mla_decode_fwd
from aiter.test_common import benchmark, checkAllclose, run_perftest

# Same opt-in as production ATOM when wiring FlyDSL page-size-1 persistent decode.
os.environ.setdefault("AITER_MLA_DECODE_PS1_FLYDSL", "1")

torch.set_default_device("cuda")

SUPPORTED_GFX = ["gfx1250"]

SUPPORTED_NUM_Q_HEADS = (16, 32, 64, 128)
SUPPORTED_Q_SEQ_LENS = {
    16: (1, 2, 4),
    32: (1,),
    64: (1,),
    128: (1,),
}
QK_NOPE_HEAD_DIM = 512
QK_ROPE_HEAD_DIM = 64
QK_HEAD_DIM = QK_NOPE_HEAD_DIM + QK_ROPE_HEAD_DIM
V_HEAD_DIM = QK_NOPE_HEAD_DIM

PS64_PAGE_SIZE = 64
# The page-size-64 kernels read Q through a 768 B per-head row stride; the
# page-size-1 persistent kernel reads it unpadded at 576 B. Same values, two
# buffers.
PS64_Q_HEAD_STRIDE = 768

# Planner head count is always 16: the page-size-1 kernel eats the full
# nhead (16 or 128) inside one work item, and passing 128 would trigger the
# host-side 128-to-16-head fold that this kernel does not consume.
METADATA_NUM_Q_HEADS = 16
KV_GRANULARITY = 16
MAX_SPLIT_PER_BATCH = 16


_SEED = 20260909
_PERF_NUM_ITERS = 101
_PERF_NUM_WARMUP = 5

SCALE_MODES = ("unit", "poc")


def _make_fp8_scales(device, scale_mode: str):
    """Scalar q/kv descales passed into mla_decode_fwd (fp8 paths).

    unit: 1.0 / 1.0 — default for apples-to-apples PS1 vs ASM on the same tensors.
    poc:  0.75 / 1.20 — matches test_mla_decode_pagesize64 _make_scales(enabled=True).
    """
    if scale_mode not in SCALE_MODES:
        raise ValueError(f"scale_mode must be one of {SCALE_MODES}, got {scale_mode!r}")
    if scale_mode == "unit":
        q_val, kv_val = 1.0, 1.0
    else:
        q_val, kv_val = 0.75, 1.20
    return (
        torch.tensor([q_val], dtype=torch.float32, device=device),
        torch.tensor([kv_val], dtype=torch.float32, device=device),
    )


def _seed_for(batch, ctx_len, q_seq_len, varlen, min_ratio, nhead=16):
    return (
        _SEED
        + batch * 1009
        + ctx_len * 17
        + q_seq_len * 101
        + nhead * 13
        + int(varlen) * 7919
        + int(min_ratio * 1000) * 31
    )


def _make_seq_lens(batch, ctx_len, varlen, min_ratio):
    """Per-sequence KV lengths.

    Fixed-length: every entry is ctx_len; total_kv = batch * ctx_len.

    Variable-length: total_kv is *still* batch * ctx_len so the two modes
    exercise exactly the same amount of compute. Individual lengths vary:
    each sequence gets at least max(1, round(ctx_len * min_ratio)) tokens;
    the remainder is distributed via a uniform stick-breaking draw (batch-1
    random cut-points in [0, remaining]), so the batch has a genuine
    long/short spread and some sequences may exceed ctx_len.
    """
    if not varlen or batch == 1:
        return [ctx_len] * batch
    low = max(1, round(ctx_len * min_ratio))
    target_total = batch * ctx_len
    remaining = target_total - batch * low
    if remaining <= 0:
        # min_ratio so close to 1 there is no distributable slack.
        return [ctx_len] * batch
    # Place (batch-1) cut-points uniformly in [0, remaining], sort them, then
    # take consecutive gaps as the extra tokens above `low` per sequence.
    cuts = sorted(torch.randint(0, remaining + 1, (batch - 1,)).tolist())
    boundaries = [0] + cuts + [remaining]
    extras = [boundaries[i + 1] - boundaries[i] for i in range(batch)]
    return [low + e for e in extras]


def _pages_per_batch(seq_lens):
    return [(n + PS64_PAGE_SIZE - 1) // PS64_PAGE_SIZE for n in seq_lens]


def _auto_num_splits(batch, seq_lens, q_seq_len, nhead):
    """The split count mla_decode_fwd's non-persistent path would pick.

    Mirrors that wrapper, including the argument quirk that dominates the result:
    it passes `total_kv = kv_indices.shape[0]`, which at page_size=64 is a page
    count rather than a token count.
    """
    num_splits, _ = get_meta_param(
        None,
        batch,
        sum(_pages_per_batch(seq_lens)),
        nhead,
        q_seq_len,
        dtypes.fp8,
    )
    return int(num_splits)


# ---------------------------------------------------------------------------
# Layout packing: one logical KV -> the two on-device layouts
# ---------------------------------------------------------------------------


def _nan_like(reference):
    """A NaN scalar in `reference`'s dtype (fp8 has no Python literal path)."""
    return torch.tensor(float("nan"), dtype=torch.float32, device=reference.device).to(
        reference.dtype
    )


def _pack_kv_pagesize1(kv_logical):
    """[total_kv, 576] -> ([total_kv, 1, 1, 576], token-level block table).

    Every token becomes its own page holding [nope|rope] contiguously, so the
    ragged concatenation already is the logical page order and varlen needs no
    special casing here. Logical token i is scattered to physical page
    indices[i] so the test also covers the gather path.
    """
    total_pages = kv_logical.size(0)
    flat = kv_logical.reshape(total_pages, 1, 1, QK_HEAD_DIM).contiguous()

    indices = torch.randperm(total_pages, device=kv_logical.device).to(torch.int32)
    kv_buffer = torch.empty_like(flat)
    kv_buffer[indices.long()] = flat
    return kv_buffer, indices


def _pack_kv_pagesize64(kv_logical, seq_lens):
    """[total_kv, 576] -> ([num_pages, 64*576], page-level block table).

    Each page holds 64 tokens seg-packed as [64*512 nope | 64*64 rope]. Page
    counts differ per sequence under varlen, so the pages are laid out batch by
    batch. Whatever the last page of a batch does not use is poisoned with NaN,
    so a kernel that ignores kv_last_page_lens shows up as a non-finite output
    rather than a small numeric drift.
    """
    device = kv_logical.device
    pages_per_batch = _pages_per_batch(seq_lens)
    total_pages = sum(pages_per_batch)
    nan_value = _nan_like(kv_logical)

    pages = torch.empty(
        (total_pages, PS64_PAGE_SIZE, QK_HEAD_DIM),
        dtype=kv_logical.dtype,
        device=device,
    )
    kv_cursor = 0
    page_cursor = 0
    for seq_len, page_count in zip(seq_lens, pages_per_batch):
        slots = pages[page_cursor : page_cursor + page_count].reshape(
            page_count * PS64_PAGE_SIZE, QK_HEAD_DIM
        )
        slots[:seq_len] = kv_logical[kv_cursor : kv_cursor + seq_len]
        if seq_len < slots.size(0):
            slots[seq_len:] = nan_value
        kv_cursor += seq_len
        page_cursor += page_count

    packed = torch.cat(
        (
            pages[..., :QK_NOPE_HEAD_DIM].reshape(
                total_pages, PS64_PAGE_SIZE * QK_NOPE_HEAD_DIM
            ),
            pages[..., QK_NOPE_HEAD_DIM:].reshape(
                total_pages, PS64_PAGE_SIZE * QK_ROPE_HEAD_DIM
            ),
        ),
        dim=-1,
    ).contiguous()

    indices = torch.randperm(total_pages, device=device).to(torch.int32)
    kv_buffer = torch.empty_like(packed)
    kv_buffer[indices.long()] = packed
    return kv_buffer, indices


def _pack_q_pagesize64(query):
    """[total_q, nhead, 576] -> the same values behind a 768 B per-head row stride."""
    nhead = query.size(1)
    padded = torch.zeros(
        (query.size(0), nhead, PS64_Q_HEAD_STRIDE),
        dtype=query.dtype,
        device=query.device,
    )
    padded[..., :QK_HEAD_DIM].copy_(query)
    return torch.as_strided(
        padded,
        size=query.shape,
        stride=(nhead * PS64_Q_HEAD_STRIDE, PS64_Q_HEAD_STRIDE, 1),
    )


# ---------------------------------------------------------------------------
# Case construction
# ---------------------------------------------------------------------------


def _prefix_sum(values, device):
    out = torch.zeros(len(values) + 1, dtype=torch.int32, device=device)
    out[1:] = torch.tensor(values, dtype=torch.int32, device=device).cumsum(0)
    return out


def _allocate_ps1_metadata(batch, q_seq_len):
    metadata_info = aiter.get_mla_metadata_info_v1(
        batch,
        q_seq_len,
        METADATA_NUM_Q_HEADS,
        dtypes.fp8,
        dtypes.fp8,
        is_sparse=False,
        fast_mode=True,
        num_kv_splits=MAX_SPLIT_PER_BATCH,
        intra_batch_mode=False,
    )
    return [
        torch.empty(size, dtype=dtype, device="cuda") for size, dtype in metadata_info
    ]


def _build_ps1_metadata(seq_lens, q_seq_len):
    """Plan the persistent work queue.

    At page_size=1 the planner's kv_indptr is token-level, so the ragged
    per-sequence lengths go in directly and the planner is what decides how the
    uneven work is spread over CUs.
    """
    device = torch.device("cuda")
    batch = len(seq_lens)
    qo_indptr = torch.arange(batch + 1, dtype=torch.int32, device=device) * q_seq_len
    kv_indptr = _prefix_sum(seq_lens, device)
    kv_last_page_lens = torch.ones(batch, dtype=torch.int32, device=device)

    (
        work_meta_data,
        work_indptr,
        work_info_set,
        reduce_indptr,
        reduce_final_map,
        reduce_partial_map,
    ) = _allocate_ps1_metadata(batch, q_seq_len)

    aiter.get_mla_metadata_v1(
        qo_indptr,
        kv_indptr,
        kv_last_page_lens,
        METADATA_NUM_Q_HEADS,
        1,
        True,
        work_meta_data,
        work_info_set,
        work_indptr,
        reduce_indptr,
        reduce_final_map,
        reduce_partial_map,
        page_size=1,
        kv_granularity=KV_GRANULARITY,
        max_seqlen_qo=q_seq_len,
        uni_seqlen_qo=q_seq_len,
        fast_mode=True,
        max_split_per_batch=MAX_SPLIT_PER_BATCH,
        intra_batch_mode=False,
        dtype_q_nope=dtypes.fp8,
        dtype_kv_nope=dtypes.fp8,
    )

    num_works = int(work_indptr[-1].item())
    work_info = work_info_set[:num_works]
    partial_locations = work_info[:, 1]
    if bool((work_info[:, 3] - work_info[:, 2] != q_seq_len).any()):
        raise RuntimeError(
            "page-size-1 comparison requires one full query tile per work item"
        )

    # True max_t(n_splits). The FlyDSL reduce compiles its unroll to this, so an
    # over-estimate would cost masked gathers on every tile.
    tile_widths = reduce_indptr[1:] - reduce_indptr[:-1]

    return {
        "work_meta_data": work_meta_data,
        "work_indptr": work_indptr,
        "work_info_set": work_info_set,
        "work_info": work_info,
        "kv_indptr_ps1": kv_indptr,
        "kv_last_page_lens_ps1": kv_last_page_lens,
        "reduce_indptr": reduce_indptr,
        "reduce_final_map": reduce_final_map,
        "reduce_partial_map": reduce_partial_map,
        "num_works": num_works,
        # Work items the planner left whole carry partial_qo_loc == -1 and write
        # straight to the final output, so clamp before sizing the partials.
        "num_partials": max(int(partial_locations.max().item()), 0) + q_seq_len,
        "max_splits": int(tile_widths.max().item()),
    }


def _build_case(seq_lens, num_splits, q_seq_len, nhead, scale_mode="unit"):
    batch = len(seq_lens)
    if batch < 1 or min(seq_lens) < 1:
        raise ValueError(f"batch and every seq_len must be positive, got {seq_lens=}")

    device = torch.device("cuda")
    total_kv = sum(seq_lens)

    query = torch.randn(
        (batch * q_seq_len, nhead, QK_HEAD_DIM),
        dtype=torch.bfloat16,
        device=device,
    ).to(dtypes.fp8)
    # Ragged concatenation: sequence b owns rows [kv_indptr[b], kv_indptr[b+1]).
    kv_logical = torch.randn(
        (total_kv, QK_HEAD_DIM),
        dtype=torch.bfloat16,
        device=device,
    ).to(dtypes.fp8)

    kv_ps1, kv_indices_ps1 = _pack_kv_pagesize1(kv_logical)
    kv_ps64, kv_indices_ps64 = _pack_kv_pagesize64(kv_logical, seq_lens)

    pages_per_batch = _pages_per_batch(seq_lens)
    last_page_lens = [n % PS64_PAGE_SIZE or PS64_PAGE_SIZE for n in seq_lens]

    case = {
        "seq_lens": seq_lens,
        "kv_offsets": [0, *itertools.accumulate(seq_lens)],
        "query_ps1": query,
        "query_ps64": _pack_q_pagesize64(query),
        "kv_logical": kv_logical,
        "kv_ps1": kv_ps1,
        "kv_indices_ps1": kv_indices_ps1,
        "kv_ps64": kv_ps64,
        "kv_indices_ps64": kv_indices_ps64,
        # ASM stage 1 walks a PAGE-level block table, so kv_indptr counts pages.
        "kv_indptr_ps64": _prefix_sum(pages_per_batch, device),
        "kv_last_page_lens_ps64": torch.tensor(
            last_page_lens, dtype=torch.int32, device=device
        ),
        "qo_indptr": (
            torch.arange(batch + 1, dtype=torch.int32, device=device) * q_seq_len
        ),
        # Every sequence gets the same split budget regardless of its length;
        # valid_split_count is what keeps the merge off the unused splits.
        "num_kv_splits_indptr": torch.arange(
            batch + 1, dtype=torch.int32, device=device
        )
        * num_splits,
        "total_kv": total_kv,
        "q_seq_len": q_seq_len,
    }
    q_scale, kv_scale = _make_fp8_scales(device, scale_mode)
    case["q_scale"] = q_scale
    case["kv_scale"] = kv_scale
    case["scale_mode"] = scale_mode
    case.update(_build_ps1_metadata(seq_lens, q_seq_len))
    return case


def _torch_reference(case, softmax_scale):
    """Merged attention over each sequence's whole KV run, from logical values.

    Layout independent by construction, so the same tensor validates both
    backends under either batch shape. Query positions use the same causal tail
    convention as the q_seq-aware persistent and ASM kernels.
    """
    query = case["query_ps1"].float()
    kv_logical = case["kv_logical"].float()
    q_scale = float(case["q_scale"][0])
    kv_scale = float(case["kv_scale"][0])
    score_scale = softmax_scale * q_scale * kv_scale
    offsets = case["kv_offsets"]
    q_seq_len = case["q_seq_len"]
    nhead = query.size(1)
    ref_out = torch.empty(
        (len(case["seq_lens"]) * q_seq_len, nhead, V_HEAD_DIM),
        dtype=torch.float32,
        device=query.device,
    )
    for batch_id in range(len(case["seq_lens"])):
        kv = kv_logical[offsets[batch_id] : offsets[batch_id + 1]]
        for q_pos in range(q_seq_len):
            q_row = batch_id * q_seq_len + q_pos
            valid_kv_len = max(
                case["seq_lens"][batch_id] - (q_seq_len - 1 - q_pos),
                0,
            )
            if valid_kv_len == 0:
                ref_out[q_row].zero_()
                continue
            valid_kv = kv[:valid_kv_len]
            logits = torch.matmul(query[q_row], valid_kv.transpose(0, 1)) * score_scale
            probabilities = torch.softmax(logits, dim=-1)
            ref_out[q_row] = (
                torch.matmul(probabilities, valid_kv[:, :V_HEAD_DIM]) * kv_scale
            )
    return ref_out


# ---------------------------------------------------------------------------
# Backends (both route through aiter.mla.mla_decode_fwd)
# ---------------------------------------------------------------------------


def _decode_output(batch, q_seq_len, nhead):
    return torch.empty((batch * q_seq_len, nhead, V_HEAD_DIM), dtype=torch.bfloat16)


def _run_ps1_mla_decode_fwd(case, q_seq_len, softmax_scale, output):
    mla_decode_fwd(
        case["query_ps1"],
        case["kv_ps1"],
        output,
        case["qo_indptr"],
        case["kv_indptr_ps1"],
        case["kv_indices_ps1"],
        case["kv_last_page_lens_ps1"],
        q_seq_len,
        page_size=1,
        nhead_kv=1,
        sm_scale=softmax_scale,
        work_meta_data=case["work_meta_data"],
        work_indptr=case["work_indptr"],
        work_info_set=case["work_info_set"],
        reduce_indptr=case["reduce_indptr"],
        reduce_final_map=case["reduce_final_map"],
        reduce_partial_map=case["reduce_partial_map"],
        q_scale=case["q_scale"],
        kv_scale=case["kv_scale"],
        causal=True,
    )


def _run_asm_ps64_mla_decode_fwd(case, q_seq_len, num_splits, softmax_scale, output):
    mla_decode_fwd(
        case["query_ps64"],
        case["kv_ps64"].view(-1, PS64_PAGE_SIZE, 1, QK_HEAD_DIM),
        output,
        case["qo_indptr"],
        case["kv_indptr_ps64"],
        case["kv_indices_ps64"],
        case["kv_last_page_lens_ps64"],
        q_seq_len,
        page_size=PS64_PAGE_SIZE,
        nhead_kv=1,
        sm_scale=softmax_scale,
        num_kv_splits=num_splits,
        num_kv_splits_indptr=case["num_kv_splits_indptr"],
        q_scale=case["q_scale"],
        kv_scale=case["kv_scale"],
        causal=True,
    )


@benchmark()
def test_ps1_persistent_vs_asm_ps64(
    batch=4,
    ctx_len=4096,
    q_seq_len=1,
    varlen=False,
    varlen_min_ratio=0.125,
    num_splits=0,
    num_iters=_PERF_NUM_ITERS,
    num_warmup=_PERF_NUM_WARMUP,
    nhead=16,
    scales="unit",
):
    if nhead not in SUPPORTED_NUM_Q_HEADS:
        raise ValueError(f"nhead must be one of {SUPPORTED_NUM_Q_HEADS}, got {nhead}")
    allowed_q = SUPPORTED_Q_SEQ_LENS[nhead]
    if q_seq_len not in allowed_q:
        raise ValueError(
            f"q_seq_len must be one of {allowed_q} for nhead={nhead}, got {q_seq_len}"
        )
    torch.manual_seed(
        _seed_for(batch, ctx_len, q_seq_len, varlen, varlen_min_ratio, nhead)
    )
    seq_lens = _make_seq_lens(batch, ctx_len, varlen, varlen_min_ratio)
    if not num_splits:
        num_splits = _auto_num_splits(batch, seq_lens, q_seq_len, nhead)

    case = _build_case(seq_lens, num_splits, q_seq_len, nhead, scales)
    softmax_scale = 1.0 / (QK_HEAD_DIM**0.5)
    ref_out = _torch_reference(case, softmax_scale)

    nhead = case["query_ps1"].size(1)
    ps1_output = _decode_output(batch, q_seq_len, nhead)
    asm_output = _decode_output(batch, q_seq_len, nhead)

    def run_ps1():
        _run_ps1_mla_decode_fwd(case, q_seq_len, softmax_scale, ps1_output)

    def run_asm():
        _run_asm_ps64_mla_decode_fwd(
            case, q_seq_len, num_splits, softmax_scale, asm_output
        )

    backends = {
        "ps1_mla_decode_fwd": (run_ps1, ps1_output),
        "asm_ps64": (run_asm, asm_output),
    }

    total_kv = case["total_kv"]
    flops = 2 * total_kv * q_seq_len * nhead * (QK_HEAD_DIM + V_HEAD_DIM)
    # Unique KV bytes plus the Q read; the partial traffic differs per backend so
    # it is reported through the measured time rather than folded in here.
    stage1_bytes = total_kv * QK_HEAD_DIM + batch * q_seq_len * nhead * QK_HEAD_DIM

    # Partial traffic the merge has to stream: the fp32 [H, Dv] tile each split
    # wrote, read back once, plus the bf16 result. This is where the persistent
    # path and the ASM path differ most -- the planner emits far more, smaller
    # splits than the ASM grid does.
    partial_element_size = 4
    ps1_reduce_bytes = (
        case["num_partials"] * nhead * (V_HEAD_DIM * partial_element_size + 4)
        + batch * nhead * V_HEAD_DIM * 2 * q_seq_len
    )
    asm_reduce_bytes = (
        0
        if num_splits == 1
        else batch
        * num_splits
        * q_seq_len
        * nhead
        * (V_HEAD_DIM * partial_element_size + 4)
        + batch * q_seq_len * nhead * V_HEAD_DIM * 2
    )

    ret = {
        "scales": scales,
        "nhead": nhead,
        "num_splits": num_splits,
        "varlen": varlen,
        "min_ctx": min(seq_lens),
        "max_ctx": max(seq_lens),
        "total_kv": total_kv,
        "ps1_num_works": case["num_works"],
        "ps1_num_partials": case["num_partials"],
        "ps1_max_splits": case["max_splits"],
    }

    for name, (run_decode, output) in backends.items():
        is_asm = name == "asm_ps64"
        _, total_us = run_perftest(
            run_decode, num_iters=num_iters, num_warmup=num_warmup
        )
        # mla_decode_fwd fuses stage-1 and reduce; keep column names for tables.
        stage1_us = total_us
        reduce_us = 0.0

        assert torch.isfinite(output).all(), f"{name}: non-finite output"
        err = checkAllclose(
            ref_out,
            output.to(torch.float32),
            rtol=6e-2,
            atol=6e-2,
            tol_err_ratio=0.05,
            msg=f"{name}: MLA decode merged output",
        )
        assert err <= 0.05, f"{name}: mismatch ratio {err:.2%} exceeds 5%"

        reduce_bytes = asm_reduce_bytes if is_asm else ps1_reduce_bytes
        ret[f"{name} stage1 us"] = stage1_us
        ret[f"{name} reduce us"] = reduce_us
        ret[f"{name} total us"] = total_us
        ret[f"{name} TFLOPS"] = flops / total_us / 1e6
        ret[f"{name} stage1 TB/s"] = stage1_bytes / stage1_us / 1e6
        ret[f"{name} reduce TB/s"] = (
            reduce_bytes / reduce_us / 1e6 if reduce_us else 0.0
        )
        ret[f"{name} err"] = err

    ret["speedup ps1/asm"] = (
        ret["asm_ps64 total us"] / ret["ps1_mla_decode_fwd total us"]
    )

    # Drop columns that are either fixed per run (infrastructure args) or
    # already encoded in the backend name, so the markdown table stays narrow.
    for key in ("gfx", "num_iters", "num_warmup"):
        ret.pop(key, None)
    return ret


def main():
    if get_gfx() not in SUPPORTED_GFX:
        aiter.logger.warning(
            "ps1-persistent vs asm-ps64 comparison unsupported on %s; skipping",
            get_gfx(),
        )
        return

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="Compare gfx1250 page-size-1 persistent FlyDSL MLA decode "
        "against page-size-64 ASM (nhead=16 q_seq=1/2/4; 32/64/128 q_seq=1).",
    )
    parser.add_argument(
        "-n",
        "--nhead",
        type=int,
        nargs="*",
        choices=SUPPORTED_NUM_Q_HEADS,
        default=[16],
        help="Q head count(s). 32/64/128 only support q_seq=1.\n" "e.g.: -n 32 64 -q 1",
    )
    parser.add_argument(
        "-q",
        "--q-seq",
        type=int,
        nargs="*",
        choices=(1, 2, 3, 4),
        default=None,
        help="Query sequence lengths. Default follows --nhead "
        "(1/2/4 for 16, 1 for 128).\n"
        "e.g.: -q 1 2 4",
    )
    parser.add_argument(
        "-b",
        "--batch",
        type=int,
        nargs="*",
        default=[1, 2, 4, 8, 16],
        help="Batch sizes. e.g.: -b 1 4 32",
    )
    parser.add_argument(
        "-c",
        "--ctx-len",
        type=int,
        nargs="*",
        default=[2048, 4096, 8192],
        help="Context length. 定长 uses it for every sequence; 变长 uses it as\n"
        "the longest sequence. Multiples of 64 keep 定长 off the tail-masking\n"
        "paths. e.g.: -c 2048 4096",
    )
    parser.add_argument(
        "--varlen",
        action="store_true",
        help="Draw a different KV length per sequence (变长) instead of giving\n"
        "every sequence the same ctx_len (定长).",
    )
    parser.add_argument(
        "--varlen-min-ratio",
        type=float,
        default=0.5,
        help="Shortest sequence as a fraction of ctx_len under --varlen.\n"
        "Lower widens the imbalance; too low and the persistent planner starts\n"
        "emitting non-split work items the page-size-1 kernel cannot consume.\n"
        "e.g.: --varlen-min-ratio 0.25",
    )
    parser.add_argument(
        "--split-kv",
        type=int,
        nargs="*",
        default=[0],
        help="KV split counts for the ASM path, 0 = pick per shape like\n"
        "mla_decode_fwd does. The persistent path always plans its own.\n"
        "e.g.: --split-kv 0 1 4",
    )
    parser.add_argument(
        "--scales",
        choices=SCALE_MODES,
        default="unit",
        help="FP8 q/kv scalar descales for both backends.\n"
        "unit: 1.0/1.0 (default). poc: 0.75/1.20 like test_mla_decode_pagesize64.",
    )
    parser.add_argument(
        "--num-iters",
        type=int,
        default=_PERF_NUM_ITERS,
        help="Timed kernel iterations.",
    )
    parser.add_argument(
        "--num-warmup",
        type=int,
        default=_PERF_NUM_WARMUP,
        help="Warmup kernel iterations.",
    )
    args = parser.parse_args()

    rows = []
    for nhead in args.nhead:
        q_seqs = args.q_seq if args.q_seq else list(SUPPORTED_Q_SEQ_LENS[nhead])
        for q_seq_len, batch, ctx_len, num_splits in itertools.product(
            q_seqs, args.batch, args.ctx_len, args.split_kv
        ):
            rows.append(
                test_ps1_persistent_vs_asm_ps64(
                    batch,
                    ctx_len,
                    q_seq_len,
                    args.varlen,
                    args.varlen_min_ratio,
                    num_splits,
                    num_iters=args.num_iters,
                    num_warmup=args.num_warmup,
                    nhead=nhead,
                    scales=args.scales,
                )
            )
    df = pd.DataFrame(rows)
    aiter.logger.info(
        "ps1-persistent vs asm-ps64 summary (markdown):\n%s",
        df.to_markdown(index=False),
    )


if __name__ == "__main__":
    main()
