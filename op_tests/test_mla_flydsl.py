# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Accuracy and timing tests for the gfx1250 MLA decode kernels written in FlyDSL.

Backends:

* PS1: persistent page-size-1 FP8 decode. :func:`aiter.mla.mla_decode_fwd` runs
  it on code objects exported from FlyDSL (``aiter.mla_ps1_fp8_asm_fwd``,
  hsa/gfx1250/mla_dsl). "jit" below is FlyDSL JIT of the PS1 kernel in this
  repository (aiter/ops/flydsl/kernels/mla_gfx1250/mla_pagesize1_fp8_fp8.py),
  a different build of the same computation.
* PS64: the page-size-64 FlyDSL kernel, and the hand-written ASM PS64 kernel
  (``asm_ps64``) both are compared with.

Modes:

* ``ps1`` checks PS1 through :func:`aiter.mla.mla_decode_fwd` against the torch
  reference.
* ``cp`` does the same under round-robin context parallelism: global KV
  position ``p`` lives on rank ``p % W``, each rank runs on its own shard with
  the causal mask applied on global positions, and the ranks' LSE-merged output
  is compared with full-KV attention.
* ``ps1-asm`` checks the PS1 code objects against FlyDSL JIT: every case runs
  once with the code object entry point stood in for by JIT and once on the code
  object; output and LSE must agree to bf16 / fp32 rounding, and the code object
  output must match the reference. Causal masking and the returned LSE are
  swept, since each selects its own code object.
* ``cp-asm`` does the same for round-robin CP, rank by rank.
* ``ps1-vs-asm`` times PS1 against ASM PS64 on the same logical Q/KV, each
  packed into its native layout. Head counts ASM PS64 has no kernel for run
  padded up to one it does (see the ``asm heads`` column: at 96 heads ASM does
  128 heads of work).
* ``ps64`` checks and times the FlyDSL PS64 kernel against ASM PS64.

Every mode sweeps the product of its arguments and skips the combinations its
backends do not support (reported at the end).

Examples:

  python3 op_tests/test_mla_flydsl.py --mode ps1
  python3 op_tests/test_mla_flydsl.py --mode ps1 -b 1 -c 1 63 64 65
  python3 op_tests/test_mla_flydsl.py --mode ps1 --varlen --varlen-min-ratio 0.1
  python3 op_tests/test_mla_flydsl.py --mode cp --num-heads 16 96 \\
      --q-seq-len 2 4 --cp-world-size 2 4 8 -c 3 64 1200 --varlen
  python3 op_tests/test_mla_flydsl.py --mode ps1-asm -c 65 2048 8192 --varlen
  python3 op_tests/test_mla_flydsl.py --mode cp-asm --q-seq-len 1 4 \\
      --cp-world-size 2 4 8 -c 64 2048
  python3 op_tests/test_mla_flydsl.py --mode ps1-vs-asm --num-heads 96 \\
      -b 64 -c 8192 --varlen --varlen-min-ratio 0.1
  python3 op_tests/test_mla_flydsl.py --mode ps64 --split-kv 0 1 2
"""

import argparse
import collections
import contextlib
import itertools
import os

import pandas as pd
import torch

import aiter
from aiter import dtypes
from aiter.jit.utils.chip_info import get_gfx
from aiter.mla import get_meta_param, mla_decode_fwd
from aiter.test_common import checkAllclose, run_perftest

os.environ.setdefault("AITER_MLA_DECODE_PS1_FLYDSL", "1")
torch.set_default_device("cuda")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SUPPORTED_GFX = ("gfx1250",)
MODES = ("ps1", "cp", "ps1-asm", "cp-asm", "ps1-vs-asm", "ps64")

SUPPORTED_NUM_Q_HEADS = (16, 32, 64, 96, 128)
# Query lengths PS1 serves per head count.
PS1_Q_SEQ_LENS = {
    16: (1, 2, 3, 4),
    32: (1,),
    64: (1,),
    96: (1, 2, 3, 4, 6, 8),
    128: (1,),
}
# Query lengths ps1-vs-asm compares at (those ASM PS64 has kernels for).
COMPARE_Q_SEQ_LENS = {
    16: (1, 2, 4),
    32: (1,),
    64: (1,),
    96: (1,),
    128: (1,),
}
# Code objects and FlyDSL JIT are separate builds that round probabilities to
# FP8 at different points, so the odd value lands one FP8 step apart.
JIT_MISMATCH_RATIO = 1e-3
# Head counts ASM PS64 runs padded up to.
ASM_PADDED_HEADS = {96: 128}
# The PS1 metadata planner packs nhead * q_seq query rows per work item up to
# this many; beyond it every work item holds one query position.
PLANNER_PACKED_Q_BUDGET = 128

QK_NOPE_HEAD_DIM = 512
QK_ROPE_HEAD_DIM = 64
QK_HEAD_DIM = QK_NOPE_HEAD_DIM + QK_ROPE_HEAD_DIM
V_HEAD_DIM = QK_NOPE_HEAD_DIM
SOFTMAX_SCALE = 1.0 / (QK_HEAD_DIM**0.5)

PS64_PAGE_SIZE = 64
PS64_NUM_Q_HEADS = 128
PS64_Q_HEAD_STRIDE = 768
KV_GRANULARITY = 16
MAX_SPLIT_PER_BATCH = 16

SCALE_MODES = ("unit", "poc")
_SEED = 20260909
_PERF_NUM_ITERS = 101
_PERF_NUM_WARMUP = 5

# ---------------------------------------------------------------------------
# Cases: sequence lengths, Q/KV, backend layouts and metadata
# ---------------------------------------------------------------------------


def _make_fp8_scales(device, mode):
    if mode == "unit":
        q_value, kv_value = 1.0, 1.0
    elif mode == "poc":
        q_value, kv_value = 0.75, 1.20
    else:
        raise ValueError(f"unsupported scales={mode!r}; expected {SCALE_MODES}")
    return (
        torch.tensor([q_value], dtype=torch.float32, device=device),
        torch.tensor([kv_value], dtype=torch.float32, device=device),
    )


def _seed_for(batch, ctx_len, q_seq_len, varlen, min_ratio, nhead):
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
    """Return positive lengths whose total remains ``batch * ctx_len``."""
    if not varlen or batch == 1:
        return [ctx_len] * batch

    low = max(1, round(ctx_len * min_ratio))
    remaining = batch * (ctx_len - low)
    if remaining <= 0:
        return [ctx_len] * batch

    cuts = sorted(torch.randint(0, remaining + 1, (batch - 1,)).tolist())
    boundaries = [0, *cuts, remaining]
    return [low + boundaries[index + 1] - boundaries[index] for index in range(batch)]


def _min_seq_len(batch, ctx_len, varlen, min_ratio):
    """Shortest length ``_make_seq_lens`` can return for these arguments."""
    if not varlen or batch == 1:
        return ctx_len
    return min(ctx_len, max(1, round(ctx_len * min_ratio)))


def _prefix_sum(values, device):
    result = torch.zeros(len(values) + 1, dtype=torch.int32, device=device)
    result[1:] = torch.tensor(values, dtype=torch.int32, device=device).cumsum(0)
    return result


def _pages_per_seq(seq_lens):
    return [(length + PS64_PAGE_SIZE - 1) // PS64_PAGE_SIZE for length in seq_lens]


def _auto_num_splits(batch, seq_lens, q_seq_len, nhead):
    """Match ``mla_decode_fwd`` PS64 split selection.

    Its ``total_kv`` argument is the number of PS64 page-table entries, not the
    logical token count.
    """
    num_splits, _ = get_meta_param(
        None,
        batch,
        sum(_pages_per_seq(seq_lens)),
        nhead,
        q_seq_len,
        dtypes.fp8,
    )
    return int(num_splits)


def _nan_like(tensor):
    return torch.tensor(float("nan"), dtype=torch.float32, device=tensor.device).to(
        tensor.dtype
    )


def _pack_q_ps64(query):
    """Place Q behind the 768-byte head stride required by gfx1250 ASM PS64."""
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


def _pack_kv_ps1(kv_logical):
    """Pack one logical token per interleaved ``[nope|rope]`` physical page."""
    num_pages = kv_logical.size(0)
    logical_pages = kv_logical.reshape(num_pages, 1, 1, QK_HEAD_DIM).contiguous()
    page_indices = torch.randperm(num_pages, device=kv_logical.device).to(torch.int32)
    kv_buffer = torch.empty_like(logical_pages)
    kv_buffer[page_indices.long()] = logical_pages
    return kv_buffer, page_indices


def _pack_kv_ps64(kv_logical, seq_lens):
    """Pack logical KV into PS64 ``[nope_block|rope_block]`` pages."""
    device = kv_logical.device
    pages_per_seq = _pages_per_seq(seq_lens)
    total_pages = sum(pages_per_seq)
    pages = torch.empty(
        (total_pages, PS64_PAGE_SIZE, QK_HEAD_DIM),
        dtype=kv_logical.dtype,
        device=device,
    )

    kv_offset = 0
    page_offset = 0
    for seq_len, page_count in zip(seq_lens, pages_per_seq):
        slots = pages[page_offset : page_offset + page_count].reshape(
            page_count * PS64_PAGE_SIZE, QK_HEAD_DIM
        )
        slots[:seq_len] = kv_logical[kv_offset : kv_offset + seq_len]
        if seq_len < slots.size(0):
            slots[seq_len:] = _nan_like(kv_logical)
        kv_offset += seq_len
        page_offset += page_count

    packed_pages = torch.cat(
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
    page_indices = torch.randperm(total_pages, device=device).to(torch.int32)
    kv_buffer = torch.empty_like(packed_pages)
    kv_buffer[page_indices.long()] = packed_pages
    return kv_buffer, page_indices


def _allocate_ps1_metadata(batch, q_seq_len, nhead):
    metadata_info = aiter.get_mla_metadata_info_v1(
        batch,
        q_seq_len,
        nhead,
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


def _build_ps1_metadata(
    seq_lens, q_seq_len, nhead, qo_indptr, is_cp_round_robin=False, causal=True
):
    device = qo_indptr.device
    kv_indptr = _prefix_sum(seq_lens, device)
    kv_last_page_lens = torch.ones(len(seq_lens), dtype=torch.int32, device=device)
    (
        work_meta_data,
        work_indptr,
        work_info_set,
        reduce_indptr,
        reduce_final_map,
        reduce_partial_map,
    ) = _allocate_ps1_metadata(len(seq_lens), q_seq_len, nhead)

    aiter.get_mla_metadata_v1(
        qo_indptr,
        kv_indptr,
        kv_last_page_lens,
        nhead,
        1,
        # Above 64 heads the planner folds the causal boundary into each work
        # item's kv_end, so the metadata has to agree with the kernel's mask.
        causal,
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
        is_cp_round_robin=is_cp_round_robin,
        dtype_q_nope=dtypes.fp8,
        dtype_kv_nope=dtypes.fp8,
    )

    num_works = int(work_indptr[-1].item())
    work_info = work_info_set[:num_works]
    qo_tile_size = 1 if nhead * 2 > PLANNER_PACKED_Q_BUDGET else q_seq_len
    if bool((work_info[:, 3] - work_info[:, 2] != qo_tile_size).any()):
        raise RuntimeError(
            f"PS1 nhead={nhead} expects qo tiles of {qo_tile_size}, "
            f"got {sorted(set((work_info[:, 3] - work_info[:, 2]).tolist()))}"
        )

    return {
        "kv_indptr_ps1": kv_indptr,
        "kv_last_page_lens_ps1": kv_last_page_lens,
        "work_meta_data": work_meta_data,
        "work_indptr": work_indptr,
        "work_info_set": work_info_set,
        "reduce_indptr": reduce_indptr,
        "reduce_final_map": reduce_final_map,
        "reduce_partial_map": reduce_partial_map,
        "num_works": num_works,
    }


def _build_logical_case(seq_lens, q_seq_len, nhead, scales):
    device = torch.device("cuda")
    batch = len(seq_lens)
    total_kv = sum(seq_lens)
    query = torch.randn(
        (batch * q_seq_len, nhead, QK_HEAD_DIM),
        dtype=torch.bfloat16,
        device=device,
    ).to(dtypes.fp8)
    kv_logical = torch.randn(
        (total_kv, QK_HEAD_DIM),
        dtype=torch.bfloat16,
        device=device,
    ).to(dtypes.fp8)
    q_scale, kv_scale = _make_fp8_scales(device, scales)
    return {
        "seq_lens": seq_lens,
        "kv_offsets": [0, *itertools.accumulate(seq_lens)],
        "query": query,
        "kv_logical": kv_logical,
        "qo_indptr": (
            torch.arange(batch + 1, dtype=torch.int32, device=device) * q_seq_len
        ),
        "q_scale": q_scale,
        "kv_scale": kv_scale,
        "q_seq_len": q_seq_len,
        "nhead": nhead,
        "total_kv": total_kv,
    }


def _prepare_case(batch, ctx_len, q_seq_len, nhead, varlen, min_ratio, scales):
    if batch < 1 or ctx_len < 1:
        raise ValueError(
            f"batch and ctx_len must be positive, got {batch=}, {ctx_len=}"
        )
    torch.manual_seed(_seed_for(batch, ctx_len, q_seq_len, varlen, min_ratio, nhead))
    seq_lens = _make_seq_lens(batch, ctx_len, varlen, min_ratio)
    return _build_logical_case(seq_lens, q_seq_len, nhead, scales)


def _add_ps1_layout(case, causal=True):
    kv_buffer, page_indices = _pack_kv_ps1(case["kv_logical"])
    case.update(
        {
            "kv_ps1": kv_buffer,
            "kv_indices_ps1": page_indices,
        }
    )
    case.update(
        _build_ps1_metadata(
            case["seq_lens"],
            case["q_seq_len"],
            case["nhead"],
            case["qo_indptr"],
            causal=causal,
        )
    )


def _pad_query_heads(query, padded_heads):
    """Widen Q to `padded_heads`, leaving the added heads zero.

    Attention is independent per head, so the original heads read back unchanged
    from the front of the result and the padding only costs time.
    """
    padded = torch.zeros(
        (query.size(0), padded_heads, query.size(2)),
        dtype=query.dtype,
        device=query.device,
    )
    padded[:, : query.size(1)].copy_(query)
    return padded


def _add_ps64_layout(case, num_splits, asm_heads=None):
    kv_buffer, page_indices = _pack_kv_ps64(case["kv_logical"], case["seq_lens"])
    pages_per_seq = _pages_per_seq(case["seq_lens"])
    asm_query = case["query"]
    if asm_heads is not None and asm_heads != case["nhead"]:
        asm_query = _pad_query_heads(asm_query, asm_heads)
    case.update(
        {
            "asm_heads": asm_query.size(1),
            "query_ps64": _pack_q_ps64(asm_query),
            "kv_ps64": kv_buffer,
            "kv_indices_ps64": page_indices,
            "kv_indptr_ps64": _prefix_sum(pages_per_seq, case["query"].device),
            "kv_last_page_lens_ps64": torch.tensor(
                [
                    length % PS64_PAGE_SIZE or PS64_PAGE_SIZE
                    for length in case["seq_lens"]
                ],
                dtype=torch.int32,
                device=case["query"].device,
            ),
            "num_kv_splits_indptr": (
                torch.arange(
                    len(case["seq_lens"]) + 1,
                    dtype=torch.int32,
                    device=case["query"].device,
                )
                * num_splits
            ),
            "seqused_k": torch.tensor(
                case["seq_lens"], dtype=torch.int32, device=case["query"].device
            ),
        }
    )


def _decode_output(case, nhead=None):
    return torch.empty(
        (
            len(case["seq_lens"]) * case["q_seq_len"],
            case["nhead"] if nhead is None else nhead,
            V_HEAD_DIM,
        ),
        dtype=torch.bfloat16,
    )


# ---------------------------------------------------------------------------
# References and checks
# ---------------------------------------------------------------------------


def _torch_reference(case, softmax_scale, causal=True, return_lse=False):
    query = case["query"].float()
    kv_logical = case["kv_logical"].float()
    q_scale = float(case["q_scale"][0])
    kv_scale = float(case["kv_scale"][0])
    score_scale = softmax_scale * q_scale * kv_scale
    q_seq_len = case["q_seq_len"]
    output = torch.empty(
        (len(case["seq_lens"]) * q_seq_len, case["nhead"], V_HEAD_DIM),
        dtype=torch.float32,
        device=query.device,
    )
    lse = torch.full(
        output.shape[:2], float("-inf"), dtype=torch.float32, device=query.device
    )

    for batch_id, seq_len in enumerate(case["seq_lens"]):
        begin = case["kv_offsets"][batch_id]
        end = case["kv_offsets"][batch_id + 1]
        kv = kv_logical[begin:end]
        for q_pos in range(q_seq_len):
            q_row = batch_id * q_seq_len + q_pos
            valid_kv_len = (
                max(seq_len - (q_seq_len - 1 - q_pos), 0) if causal else seq_len
            )
            if valid_kv_len == 0:
                output[q_row].zero_()
                continue
            valid_kv = kv[:valid_kv_len]
            logits = torch.matmul(query[q_row], valid_kv.transpose(0, 1)) * score_scale
            lse[q_row] = torch.logsumexp(logits, dim=-1)
            probabilities = torch.softmax(logits, dim=-1)
            output[q_row] = (
                torch.matmul(probabilities, valid_kv[:, :V_HEAD_DIM]) * kv_scale
            )
    return (output, lse) if return_lse else output


def _cp_local_positions(seq_len, cp_world_size, cp_rank, device):
    """Global positions of one request's tokens held by ``cp_rank``, in order."""
    return torch.arange(seq_len, device=device)[cp_rank::cp_world_size]


def _torch_reference_cp_rank(case, softmax_scale, cp_world_size, cp_rank):
    """One rank's partial attention under round-robin CP.

    Query row ``i`` of a request sits at global position ``seq_len - q_seq + i``
    and sees the rank's tokens at or before it. Rows that see none of them come
    back as out=0, lse=-inf, which is what the kernel writes for them.
    """
    query = case["query"].float()
    kv_logical = case["kv_logical"].float()
    q_scale = float(case["q_scale"][0])
    kv_scale = float(case["kv_scale"][0])
    score_scale = softmax_scale * q_scale * kv_scale
    q_seq_len = case["q_seq_len"]
    rows = len(case["seq_lens"]) * q_seq_len
    output = torch.zeros(
        (rows, case["nhead"], V_HEAD_DIM), dtype=torch.float32, device=query.device
    )
    lse = torch.full(
        (rows, case["nhead"]), float("-inf"), dtype=torch.float32, device=query.device
    )

    for batch_id, seq_len in enumerate(case["seq_lens"]):
        positions = _cp_local_positions(seq_len, cp_world_size, cp_rank, query.device)
        kv = kv_logical[case["kv_offsets"][batch_id] + positions]
        for q_pos in range(q_seq_len):
            q_row = batch_id * q_seq_len + q_pos
            visible = positions <= seq_len - q_seq_len + q_pos
            if not bool(visible.any()):
                continue
            valid_kv = kv[visible]
            logits = torch.matmul(query[q_row], valid_kv.transpose(0, 1)) * score_scale
            lse[q_row] = torch.logsumexp(logits, dim=-1)
            output[q_row] = (
                torch.matmul(torch.softmax(logits, dim=-1), valid_kv[:, :V_HEAD_DIM])
                * kv_scale
            )
    return output, lse


def _merge_cp_ranks(outputs, lses):
    """Combine per-rank partial attention with the standard LSE merge."""
    lse = torch.stack(lses)
    weights = torch.exp(lse - lse.max(0).values).nan_to_num(0.0)
    merged = (torch.stack([out.float() for out in outputs]) * weights[..., None]).sum(0)
    return merged / weights.sum(0)[..., None]


def _check_output(name, reference, output):
    """Mismatch ratio of ``output`` against ``reference``; at most 5% may differ."""
    assert torch.isfinite(output).all(), f"{name}: non-finite output"
    error = checkAllclose(
        reference,
        output.float(),
        rtol=6e-2,
        atol=6e-2,
        tol_err_ratio=0.05,
        msg=f"{name}: MLA decode output",
    )
    assert error <= 0.05, f"{name}: mismatch ratio {error:.2%} exceeds 5%"
    return error


def _check_lse(name, reference, output):
    finite = torch.isfinite(reference)
    assert bool(
        torch.isneginf(output[~finite]).all()
    ), f"{name}: rows that see no local KV must report lse=-inf"
    if not bool(finite.any()):
        return 0.0
    # Absolute and tight on purpose: one token wrongly let through the round-robin
    # mask moves a row's LSE by roughly 1/(visible tokens), which a relative
    # tolerance on LSE values of ~log(len) would swallow at long context.
    error = checkAllclose(
        reference[finite],
        output[finite],
        rtol=0.0,
        atol=1e-3,
        tol_err_ratio=0.05,
        msg=f"{name}: MLA decode LSE",
    )
    assert error <= 0.05, f"{name}: LSE mismatch ratio {error:.2%} exceeds 5%"
    return error


def _assert_matches_jit(name, jit_tensors, asm_tensors):
    """The code object's output and LSE against FlyDSL JIT's: -inf in the same
    places, and at most JIT_MISMATCH_RATIO of the finite values apart by more
    than bf16 output / fp32 LSE rounding."""
    for index, (jit, asm) in enumerate(zip(jit_tensors, asm_tensors, strict=True)):
        if jit is None or asm is None:
            assert jit is None and asm is None, f"{name}: tensor {index} is missing"
            continue
        tol = 2e-2 if jit.dtype == torch.bfloat16 else 1e-4
        jit, asm = jit.float(), asm.float()
        assert torch.equal(
            torch.isneginf(jit), torch.isneginf(asm)
        ), f"{name}: code object output {index} has -inf where FlyDSL JIT does not"
        finite = torch.isfinite(jit)
        if not bool(finite.any()):
            continue
        close = torch.isclose(asm[finite], jit[finite], atol=tol, rtol=tol)
        ratio = 1.0 - close.float().mean().item()
        assert ratio <= JIT_MISMATCH_RATIO, (
            f"{name}: code object output {index} vs FlyDSL JIT: {ratio:.2e} of "
            f"values differ by more than {tol} (max "
            f"{(asm[finite] - jit[finite]).abs().max().item():.4f})"
        )


# ---------------------------------------------------------------------------
# Launchers
# ---------------------------------------------------------------------------


def _run_ps1(case, softmax_scale, output, causal=True, return_lse=False):
    """Run PS1 decode into ``output``; return the final LSE when asked for."""
    _, final_lse = mla_decode_fwd(
        case["query"],
        case["kv_ps1"],
        output,
        case["qo_indptr"],
        case["kv_indptr_ps1"],
        case["kv_indices_ps1"],
        case["kv_last_page_lens_ps1"],
        case["q_seq_len"],
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
        causal=causal,
        return_lse=return_lse,
    )
    return final_lse


def _run_asm_ps64(case, num_splits, softmax_scale, output):
    mla_decode_fwd(
        case["query_ps64"],
        case["kv_ps64"].view(-1, PS64_PAGE_SIZE, 1, QK_HEAD_DIM),
        output,
        case["qo_indptr"],
        case["kv_indptr_ps64"],
        case["kv_indices_ps64"],
        case["kv_last_page_lens_ps64"],
        case["q_seq_len"],
        page_size=PS64_PAGE_SIZE,
        nhead_kv=1,
        sm_scale=softmax_scale,
        num_kv_splits=num_splits,
        num_kv_splits_indptr=case["num_kv_splits_indptr"],
        q_scale=case["q_scale"],
        kv_scale=case["kv_scale"],
        causal=True,
    )


@contextlib.contextmanager
def _ps1_stage1_backend(use_asm):
    """mla_decode_fwd runs PS1 stage 1 on the exported code objects; with
    ``use_asm=False`` the code object entry point is stood in for by FlyDSL JIT
    of the repository's PS1 kernel. Fails if no launch lands on the backend
    asked for."""
    import aiter.ops.flydsl.mla_kernels as flydsl_mla

    calls = {"asm": 0, "jit": 0}
    asm_fn = aiter.mla_ps1_fp8_asm_fwd

    def asm(*args, **kwargs):
        calls["asm"] += 1
        return asm_fn(*args, **kwargs)

    def jit(
        split_data,
        split_lse,
        final_output,
        final_lse,
        q,
        kv_buffer,
        kv_page_indices,
        work_indptr,
        work_info,
        softmax_scale,
        q_scale,
        kv_scale,
        max_seqlen_q,
        causal,
        qo_indptr,
        kv_indptr,
        g_kv_indptr,
        cp_world_size,
        cp_rank,
    ):
        calls["jit"] += 1
        flydsl_mla.flydsl_mla_pagesize1_fp8_fp8(
            split_data,
            split_lse,
            final_output,
            q,
            kv_buffer,
            kv_page_indices,
            work_indptr,
            work_info,
            softmax_scale,
            q_scale=q_scale,
            kv_scale=kv_scale,
            final_lse=final_lse,
            max_seqlen_q=max_seqlen_q,
            causal=causal,
            qo_indptr=qo_indptr,
            kv_indptr=kv_indptr,
            g_kv_indptr=g_kv_indptr,
            cp_world_size=cp_world_size,
            cp_rank=cp_rank,
        )

    aiter.mla_ps1_fp8_asm_fwd = asm if use_asm else jit
    try:
        yield calls
    finally:
        aiter.mla_ps1_fp8_asm_fwd = asm_fn
    expected, other = ("asm", "jit") if use_asm else ("jit", "asm")
    assert (
        calls[expected] > 0 and calls[other] == 0
    ), f"PS1 stage 1 expected on {expected}, dispatched {calls}"


# ---------------------------------------------------------------------------
# Mode drivers: each runs one case and returns its summary row
# ---------------------------------------------------------------------------


def _row(mode, case, nhead, q_seq_len, **columns):
    """A summary row: the columns every mode shares, then the mode's own."""
    return {
        "mode": mode,
        "batch": len(case["seq_lens"]),
        "min_ctx": min(case["seq_lens"]),
        "max_ctx": max(case["seq_lens"]),
        "nhead": nhead,
        "q_seq": q_seq_len,
        **columns,
    }


def _test_ps1(
    batch,
    ctx_len,
    nhead,
    q_seq_len,
    num_iters,
    num_warmup,
    scales,
    varlen=False,
    min_ratio=0.5,
):
    case = _prepare_case(batch, ctx_len, q_seq_len, nhead, varlen, min_ratio, scales)
    _add_ps1_layout(case)
    reference = _torch_reference(case, SOFTMAX_SCALE)
    output = _decode_output(case)

    def run():
        _run_ps1(case, SOFTMAX_SCALE, output)

    _, us = run_perftest(run, num_iters=num_iters, num_warmup=num_warmup)
    return _row(
        "ps1",
        case,
        nhead,
        q_seq_len,
        us=us,
        err=_check_output("ps1", reference, output),
    )


def _test_ps1_asm(
    batch,
    ctx_len,
    nhead,
    q_seq_len,
    causal,
    return_lse,
    num_iters,
    num_warmup,
    scales,
    varlen=False,
    min_ratio=0.5,
):
    case = _prepare_case(batch, ctx_len, q_seq_len, nhead, varlen, min_ratio, scales)
    _add_ps1_layout(case, causal=causal)
    reference, reference_lse = _torch_reference(
        case, SOFTMAX_SCALE, causal=causal, return_lse=True
    )

    results, timings = {}, {}
    for backend in ("jit", "asm"):
        output = _decode_output(case)

        def run(output=output):
            return _run_ps1(case, SOFTMAX_SCALE, output, causal, return_lse)

        with _ps1_stage1_backend(use_asm=backend == "asm"):
            _, timings[backend] = run_perftest(
                run, num_iters=num_iters, num_warmup=num_warmup
            )
            final_lse = run()
        results[backend] = (output, final_lse)

    name = f"ps1-asm causal={int(causal)} lse={int(return_lse)}"
    _assert_matches_jit(name, results["jit"], results["asm"])
    output, final_lse = results["asm"]
    err = _check_output(name, reference, output)
    if return_lse:
        _check_lse(name, reference_lse, final_lse.float())
    return _row(
        "ps1-asm",
        case,
        nhead,
        q_seq_len,
        causal=int(causal),
        lse=int(return_lse),
        **{"jit us": timings["jit"], "asm us": timings["asm"]},
        **{"asm/jit": timings["asm"] / timings["jit"]},
        err=err,
    )


def _test_cp(
    batch,
    ctx_len,
    nhead,
    q_seq_len,
    cp_world_size,
    num_iters,
    num_warmup,
    scales,
    varlen=False,
    min_ratio=0.5,
    use_asm=None,
    return_tensors=False,
    return_lse=True,
):
    """``use_asm`` pins the PS1 stage 1 backend (None leaves mla_decode_fwd's);
    ``return_tensors`` also returns every rank's output and LSE. Without
    ``return_lse`` the kernel writes no LSE, so the ranks are merged with the
    reference LSE and the returned LSEs are None."""
    case = _prepare_case(batch, ctx_len, q_seq_len, nhead, varlen, min_ratio, scales)
    min_len = min(case["seq_lens"])
    if min_len < q_seq_len:
        raise ValueError(
            f"every request needs KV length >= q_seq={q_seq_len}, got min {min_len}"
        )
    # At max_seqlen_q == 1 the metadata planner emits no work item for a request
    # whose local shard is empty, so its output row would never be written.
    if q_seq_len == 1 and min_len < cp_world_size:
        raise ValueError(
            f"q_seq=1 needs every request's KV length >= cp_world_size="
            f"{cp_world_size}, got min {min_len}"
        )
    kv_buffer, kv_indices = _pack_kv_ps1(case["kv_logical"])
    g_kv_indptr = _prefix_sum(case["seq_lens"], kv_buffer.device)

    backend = (
        contextlib.nullcontext() if use_asm is None else _ps1_stage1_backend(use_asm)
    )
    outputs, lses, merge_lses, rank_errs, rank_us = [], [], [], [], []
    with backend:
        for cp_rank in range(cp_world_size):
            local_positions = [
                _cp_local_positions(seq_len, cp_world_size, cp_rank, kv_buffer.device)
                for seq_len in case["seq_lens"]
            ]
            local_lens = [positions.numel() for positions in local_positions]
            local_indices = torch.cat(
                [
                    kv_indices[case["kv_offsets"][batch_id] + positions]
                    for batch_id, positions in enumerate(local_positions)
                ]
            )
            if local_indices.numel() == 0:
                local_indices = torch.zeros(
                    1, dtype=torch.int32, device=kv_buffer.device
                )
            metadata = _build_ps1_metadata(
                local_lens, q_seq_len, nhead, case["qo_indptr"], is_cp_round_robin=True
            )
            output = _decode_output(case)

            def run(
                output=output,
                metadata=metadata,
                local_indices=local_indices,
                cp_rank=cp_rank,
            ):
                return mla_decode_fwd(
                    case["query"],
                    kv_buffer,
                    output,
                    case["qo_indptr"],
                    metadata["kv_indptr_ps1"],
                    local_indices,
                    metadata["kv_last_page_lens_ps1"],
                    q_seq_len,
                    page_size=1,
                    nhead_kv=1,
                    sm_scale=SOFTMAX_SCALE,
                    work_meta_data=metadata["work_meta_data"],
                    work_indptr=metadata["work_indptr"],
                    work_info_set=metadata["work_info_set"],
                    reduce_indptr=metadata["reduce_indptr"],
                    reduce_final_map=metadata["reduce_final_map"],
                    reduce_partial_map=metadata["reduce_partial_map"],
                    q_scale=case["q_scale"],
                    kv_scale=case["kv_scale"],
                    return_lse=return_lse,
                    g_kv_indptr=g_kv_indptr,
                    cp_world_size=cp_world_size,
                    cp_rank=cp_rank,
                    causal=True,
                )

            _, us = run_perftest(run, num_iters=num_iters, num_warmup=num_warmup)
            _, final_lse = run()
            reference, reference_lse = _torch_reference_cp_rank(
                case, SOFTMAX_SCALE, cp_world_size, cp_rank
            )
            name = f"cp W={cp_world_size} rank={cp_rank} lse={int(return_lse)}"
            rank_errs.append(_check_output(name, reference, output))
            if return_lse:
                _check_lse(name, reference_lse, final_lse.float())
            else:
                assert final_lse is None, f"{name}: LSE returned without return_lse"
            empty_rows = torch.isneginf(reference_lse).all(-1)
            assert bool(
                (output[empty_rows] == 0).all()
            ), f"{name}: rows that see no local KV must be written as zero"
            outputs.append(output)
            lses.append(final_lse.float() if return_lse else None)
            merge_lses.append(lses[-1] if return_lse else reference_lse)
            rank_us.append(us)

    reference = _torch_reference(case, SOFTMAX_SCALE)
    merged = _merge_cp_ranks(outputs, merge_lses)
    row = _row(
        "cp",
        case,
        nhead,
        q_seq_len,
        cp_world=cp_world_size,
        **{"max rank us": max(rank_us), "max rank err": max(rank_errs)},
        **{
            "merged err": _check_output(
                f"cp W={cp_world_size} merged", reference, merged
            )
        },
    )
    return (row, outputs, lses) if return_tensors else row


def _test_cp_asm(
    batch,
    ctx_len,
    nhead,
    q_seq_len,
    cp_world_size,
    num_iters,
    num_warmup,
    scales,
    varlen=False,
    min_ratio=0.5,
    return_lse=True,
):
    args = (batch, ctx_len, nhead, q_seq_len, cp_world_size, num_iters, num_warmup)
    kwargs = {
        "varlen": varlen,
        "min_ratio": min_ratio,
        "return_tensors": True,
        "return_lse": return_lse,
    }
    jit_row, jit_outputs, jit_lses = _test_cp(*args, scales, use_asm=False, **kwargs)
    asm_row, asm_outputs, asm_lses = _test_cp(*args, scales, use_asm=True, **kwargs)
    for cp_rank in range(cp_world_size):
        _assert_matches_jit(
            f"cp-asm W={cp_world_size} rank={cp_rank} lse={int(return_lse)}",
            (jit_outputs[cp_rank], jit_lses[cp_rank]),
            (asm_outputs[cp_rank], asm_lses[cp_rank]),
        )
    shared = ("batch", "min_ctx", "max_ctx", "nhead", "q_seq", "cp_world")
    return {
        "mode": "cp-asm",
        **{key: asm_row[key] for key in shared},
        "lse": int(return_lse),
        "jit max rank us": jit_row["max rank us"],
        "asm max rank us": asm_row["max rank us"],
        "asm/jit": asm_row["max rank us"] / jit_row["max rank us"],
        "merged err": asm_row["merged err"],
    }


def _test_ps1_vs_asm_ps64(
    batch,
    ctx_len,
    nhead,
    q_seq_len,
    num_splits,
    num_iters,
    num_warmup,
    scales,
    varlen=False,
    min_ratio=0.5,
):
    case = _prepare_case(batch, ctx_len, q_seq_len, nhead, varlen, min_ratio, scales)
    asm_heads = ASM_PADDED_HEADS.get(nhead, nhead)
    if not num_splits:
        num_splits = _auto_num_splits(batch, case["seq_lens"], q_seq_len, asm_heads)
    _add_ps1_layout(case)
    _add_ps64_layout(case, num_splits, asm_heads)

    reference = _torch_reference(case, SOFTMAX_SCALE)
    ps1_output = _decode_output(case)
    asm_output = _decode_output(case, nhead=asm_heads)

    def run_ps1():
        _run_ps1(case, SOFTMAX_SCALE, ps1_output)

    def run_asm():
        _run_asm_ps64(case, num_splits, SOFTMAX_SCALE, asm_output)

    _, ps1_us = run_perftest(run_ps1, num_iters=num_iters, num_warmup=num_warmup)
    _, asm_us = run_perftest(run_asm, num_iters=num_iters, num_warmup=num_warmup)
    return _row(
        "ps1-vs-asm",
        case,
        nhead,
        q_seq_len,
        **{
            "asm heads": asm_heads,
            "asm splits": num_splits,
            "ps1 works": case["num_works"],
            "ps1 us": ps1_us,
            "asm us": asm_us,
            "speedup": asm_us / ps1_us,
            "ps1 err": _check_output("ps1", reference, ps1_output),
            "asm err": _check_output("asm_ps64", reference, asm_output[:, :nhead]),
        },
    )


def _test_ps64(
    batch,
    ctx_len,
    num_splits,
    num_iters,
    num_warmup,
    scales,
    varlen=False,
    min_ratio=0.5,
):
    from aiter.ops.flydsl.mla_kernels import (
        flydsl_mla_decode_reduce,
        flydsl_mla_pagesize64_fp8_fp8,
    )

    case = _prepare_case(
        batch,
        ctx_len,
        1,
        PS64_NUM_Q_HEADS,
        varlen,
        min_ratio,
        scales,
    )
    if not num_splits:
        num_splits = _auto_num_splits(batch, case["seq_lens"], 1, PS64_NUM_Q_HEADS)
    _add_ps64_layout(case, num_splits)
    reference = _torch_reference(case, SOFTMAX_SCALE)

    output_shape = (batch, PS64_NUM_Q_HEADS, V_HEAD_DIM)
    split_data = torch.empty(
        (
            output_shape
            if num_splits == 1
            else (batch, num_splits, PS64_NUM_Q_HEADS, V_HEAD_DIM)
        ),
        dtype=torch.bfloat16 if num_splits == 1 else torch.float32,
    )
    split_lse = torch.empty(
        (batch, num_splits, PS64_NUM_Q_HEADS, 1), dtype=torch.float32
    )
    flydsl_output = (
        split_data
        if num_splits == 1
        else torch.empty(output_shape, dtype=torch.bfloat16)
    )
    asm_output = torch.empty(output_shape, dtype=torch.bfloat16)

    def run_flydsl():
        flydsl_mla_pagesize64_fp8_fp8(
            split_data,
            split_lse,
            case["query_ps64"],
            case["kv_ps64"],
            case["kv_indptr_ps64"],
            case["kv_indices_ps64"],
            case["kv_last_page_lens_ps64"],
            case["qo_indptr"],
            case["num_kv_splits_indptr"],
            case["q_scale"],
            case["kv_scale"],
            SOFTMAX_SCALE,
            num_splits,
            page_size=PS64_PAGE_SIZE,
        )
        if num_splits > 1:
            flydsl_mla_decode_reduce(
                split_data,
                split_lse,
                case["seqused_k"],
                flydsl_output,
                num_splits,
                1,
            )

    def run_asm():
        _run_asm_ps64(case, num_splits, SOFTMAX_SCALE, asm_output)

    _, flydsl_us = run_perftest(run_flydsl, num_iters=num_iters, num_warmup=num_warmup)
    _, asm_us = run_perftest(run_asm, num_iters=num_iters, num_warmup=num_warmup)
    return _row(
        "ps64",
        case,
        PS64_NUM_Q_HEADS,
        1,
        splits=num_splits,
        **{
            "flydsl us": flydsl_us,
            "asm us": asm_us,
            "speedup": asm_us / flydsl_us,
            "flydsl err": _check_output("flydsl_ps64", reference, flydsl_output),
            "asm err": _check_output("asm_ps64", reference, asm_output),
        },
    )


# ---------------------------------------------------------------------------
# pytest entry points
# ---------------------------------------------------------------------------


def _skip(reason):
    import pytest

    pytest.skip(reason)


def _require_gfx():
    if get_gfx() not in SUPPORTED_GFX:
        _skip(f"FlyDSL MLA tests need {SUPPORTED_GFX}, running on {get_gfx()}")


def _require_ps1(nhead, q_seq_len):
    if q_seq_len not in PS1_Q_SEQ_LENS[nhead]:
        raise ValueError(
            f"PS1 supports q_seq={PS1_Q_SEQ_LENS[nhead]} "
            f"for nhead={nhead}, got {q_seq_len}"
        )


def test_mla_flydsl(
    page_size=1,
    batch=1,
    ctx_len=65,
    num_splits=0,
    num_q_heads=128,
    q_seq_len=1,
    num_iters=_PERF_NUM_ITERS,
    num_warmup=_PERF_NUM_WARMUP,
):
    """PS1 (``page_size=1``) or FlyDSL PS64 (``page_size=64``) against the reference."""
    _require_gfx()
    if page_size == 1:
        _require_ps1(num_q_heads, q_seq_len)
        _test_ps1(batch, ctx_len, num_q_heads, q_seq_len, num_iters, num_warmup, "poc")
    elif page_size == PS64_PAGE_SIZE:
        if num_q_heads != PS64_NUM_Q_HEADS or q_seq_len != 1:
            raise ValueError("PS64 FlyDSL requires nhead=128 and q_seq_len=1")
        _test_ps64(batch, ctx_len, num_splits, num_iters, num_warmup, "poc")
    else:
        raise ValueError(f"unsupported page_size={page_size}; expected 1 or 64")


def test_ps1_asm_matches_flydsl(
    batch=4,
    ctx_len=2048,
    nhead=96,
    q_seq_len=4,
    causal=True,
    return_lse=True,
    varlen=True,
    varlen_min_ratio=0.5,
    scales="poc",
    num_iters=_PERF_NUM_ITERS,
    num_warmup=_PERF_NUM_WARMUP,
):
    _require_gfx()
    _require_ps1(nhead, q_seq_len)
    _test_ps1_asm(
        batch,
        ctx_len,
        nhead,
        q_seq_len,
        causal,
        return_lse,
        num_iters,
        num_warmup,
        scales,
        varlen,
        varlen_min_ratio,
    )


def test_cp_asm_matches_flydsl(
    batch=4,
    ctx_len=64,
    nhead=96,
    q_seq_len=4,
    cp_world_size=4,
    return_lse=True,
    varlen=True,
    varlen_min_ratio=0.5,
    scales="poc",
    num_iters=_PERF_NUM_ITERS,
    num_warmup=_PERF_NUM_WARMUP,
):
    _require_gfx()
    _require_ps1(nhead, q_seq_len)
    _test_cp_asm(
        batch,
        ctx_len,
        nhead,
        q_seq_len,
        cp_world_size,
        num_iters,
        num_warmup,
        scales,
        varlen,
        varlen_min_ratio,
        return_lse,
    )


def test_cp_asm_no_lse_matches_flydsl():
    test_cp_asm_matches_flydsl(return_lse=False)


def test_ps1_cp_round_robin(
    batch=4,
    ctx_len=64,
    nhead=96,
    q_seq_len=4,
    cp_world_size=4,
    varlen=True,
    varlen_min_ratio=0.5,
    scales="poc",
    num_iters=_PERF_NUM_ITERS,
    num_warmup=_PERF_NUM_WARMUP,
):
    _require_gfx()
    _require_ps1(nhead, q_seq_len)
    _test_cp(
        batch,
        ctx_len,
        nhead,
        q_seq_len,
        cp_world_size,
        num_iters,
        num_warmup,
        scales,
        varlen,
        varlen_min_ratio,
    )


def test_ps1_persistent_vs_asm_ps64(
    batch=4,
    ctx_len=4096,
    nhead=16,
    q_seq_len=1,
    num_splits=0,
    varlen=False,
    varlen_min_ratio=0.5,
    scales="unit",
    num_iters=_PERF_NUM_ITERS,
    num_warmup=_PERF_NUM_WARMUP,
):
    _require_gfx()
    if q_seq_len not in COMPARE_Q_SEQ_LENS[nhead]:
        raise ValueError(
            f"ps1-vs-asm supports q_seq={COMPARE_Q_SEQ_LENS[nhead]} "
            f"for nhead={nhead}, got {q_seq_len}"
        )
    _test_ps1_vs_asm_ps64(
        batch,
        ctx_len,
        nhead,
        q_seq_len,
        num_splits,
        num_iters,
        num_warmup,
        scales,
        varlen,
        varlen_min_ratio,
    )


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------


def _parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description=__doc__,
    )
    parser.add_argument("--mode", choices=MODES, default="ps1")
    parser.add_argument(
        "-b", "--batch", type=int, nargs="+", default=[4], help="Batch sizes."
    )
    parser.add_argument(
        "-c",
        "--ctx-len",
        type=int,
        nargs="+",
        default=[2048, 4096, 8192],
        help="Mean context lengths; try 1/63/64/65 for tail coverage.",
    )
    parser.add_argument(
        "--num-heads",
        type=int,
        nargs="+",
        choices=SUPPORTED_NUM_Q_HEADS,
        default=list(SUPPORTED_NUM_Q_HEADS),
        help="Query head counts (ps64 always runs 128).",
    )
    parser.add_argument(
        "--q-seq-len",
        type=int,
        nargs="+",
        choices=(1, 2, 3, 4, 6, 8),
        default=[1, 2, 3, 4],
        help="Query lengths; each head count runs the ones it supports.",
    )
    parser.add_argument(
        "--split-kv",
        type=int,
        nargs="+",
        default=[0],
        help="ASM/FlyDSL PS64 split counts (ps1-vs-asm, ps64); 0 selects them.",
    )
    parser.add_argument(
        "--cp-world-size",
        type=int,
        nargs="+",
        default=[2, 4, 8],
        help="Round-robin CP world sizes (cp, cp-asm).",
    )
    parser.add_argument(
        "--varlen",
        action="store_true",
        help="Vary the lengths around the mean (see --varlen-min-ratio).",
    )
    parser.add_argument(
        "--varlen-min-ratio",
        type=float,
        default=0.5,
        help="With --varlen, the shortest length as a fraction of the mean.",
    )
    parser.add_argument("--scales", choices=SCALE_MODES, default="unit")
    parser.add_argument("--num-iters", type=int, default=_PERF_NUM_ITERS)
    parser.add_argument("--num-warmup", type=int, default=_PERF_NUM_WARMUP)
    return parser.parse_args()


def _cases(args):
    """A runner for every combination the arguments ask for that the mode's
    backends support, and how many were skipped for each reason."""
    timing = (args.num_iters, args.num_warmup, args.scales)
    lengths = (args.varlen, args.varlen_min_ratio)
    cases, skipped = [], collections.Counter()

    if args.mode == "ps64":
        for batch, ctx_len, num_splits in itertools.product(
            args.batch, args.ctx_len, args.split_kv
        ):
            cases.append(
                lambda b=batch, c=ctx_len, s=num_splits: _test_ps64(
                    b, c, s, *timing, *lengths
                )
            )
        return cases, skipped

    supported = COMPARE_Q_SEQ_LENS if args.mode == "ps1-vs-asm" else PS1_Q_SEQ_LENS
    for nhead, q_seq_len, batch, ctx_len in itertools.product(
        args.num_heads, args.q_seq_len, args.batch, args.ctx_len
    ):
        if q_seq_len not in supported[nhead]:
            skipped[f"q_seq={q_seq_len} unsupported at nhead={nhead}"] += 1
            continue
        shape = (batch, ctx_len, nhead, q_seq_len)
        if args.mode == "ps1":
            cases.append(lambda s=shape: _test_ps1(*s, *timing, *lengths))
        elif args.mode == "ps1-asm":
            for causal, return_lse in itertools.product((True, False), (False, True)):
                cases.append(
                    lambda s=shape, c=causal, r=return_lse: _test_ps1_asm(
                        *s, c, r, *timing, *lengths
                    )
                )
        elif args.mode == "ps1-vs-asm":
            for num_splits in args.split_kv:
                cases.append(
                    lambda s=shape, n=num_splits: _test_ps1_vs_asm_ps64(
                        *s, n, *timing, *lengths
                    )
                )
        else:
            min_len = _min_seq_len(batch, ctx_len, *lengths)
            for cp_world_size in args.cp_world_size:
                # A request needs at least q_seq tokens, and at q_seq=1 every
                # rank must hold one of them (see _test_cp).
                if min_len < q_seq_len or (q_seq_len == 1 and min_len < cp_world_size):
                    skipped["KV too short for q_seq / cp_world_size"] += 1
                    continue
                cp_args = (*shape, cp_world_size, *timing, *lengths)
                if args.mode == "cp":
                    cases.append(lambda a=cp_args: _test_cp(*a))
                    continue
                for return_lse in (True, False):
                    cases.append(
                        lambda a=cp_args, r=return_lse: _test_cp_asm(*a, return_lse=r)
                    )
    return cases, skipped


def main():
    if get_gfx() not in SUPPORTED_GFX:
        aiter.logger.warning("FlyDSL MLA tests unsupported on %s; skipping", get_gfx())
        return

    args = _parse_args()
    cases, skipped = _cases(args)
    rows = [run() for run in cases]
    if skipped:
        aiter.logger.info(
            "FlyDSL MLA %s skipped: %s",
            args.mode,
            ", ".join(f"{reason} (x{count})" for reason, count in skipped.items()),
        )
    aiter.logger.info(
        "FlyDSL MLA %s summary:\n%s",
        args.mode,
        pd.DataFrame(rows).to_markdown(index=False),
    )


if __name__ == "__main__":
    main()
