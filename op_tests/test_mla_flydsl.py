# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Accuracy and timing tests for gfx1250 FlyDSL MLA decode kernels.

Modes:

* ``ps1`` validates persistent page-size-1 FlyDSL through
  :func:`aiter.mla.mla_decode_fwd`.
* ``ps64`` compares the dedicated page-size-64 FlyDSL kernel with ASM PS64.
* ``ps1-vs-asm`` compares persistent FlyDSL PS1 with ASM PS64 using the same
  logical Q/KV values packed into each backend's native layout.

Examples:

  python3 op_tests/test_mla_flydsl.py --mode ps1
  python3 op_tests/test_mla_flydsl.py --mode ps1 -b 1 -c 1 63 64 65
  python3 op_tests/test_mla_flydsl.py --mode ps64 --split-kv 0 1 2
  python3 op_tests/test_mla_flydsl.py --mode ps1-vs-asm \
      --num-heads 32 --q-seq-len 1 --varlen
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
from aiter.test_common import checkAllclose, run_perftest

os.environ.setdefault("AITER_MLA_DECODE_PS1_FLYDSL", "1")
torch.set_default_device("cuda")

SUPPORTED_GFX = ("gfx1250",)
MODES = ("ps1", "ps64", "ps1-vs-asm")
SUPPORTED_NUM_Q_HEADS = (16, 32, 64, 128)
PS1_Q_SEQ_LENS = {
    16: (1, 2, 3, 4),
    32: (1,),
    64: (1,),
    128: (1,),
}
# ASM PS64 has no registered (16, 3) kernel.
COMPARE_Q_SEQ_LENS = {
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
PS64_NUM_Q_HEADS = 128
PS64_Q_HEAD_STRIDE = 768
KV_GRANULARITY = 16
MAX_SPLIT_PER_BATCH = 16

SCALE_MODES = ("unit", "poc")
_SEED = 20260909
_PERF_NUM_ITERS = 101
_PERF_NUM_WARMUP = 5


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


def _build_ps1_metadata(seq_lens, q_seq_len, nhead, qo_indptr):
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
    if bool((work_info[:, 3] - work_info[:, 2] != q_seq_len).any()):
        raise RuntimeError("FlyDSL PS1 requires one full query tile per work item")

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


def _add_ps1_layout(case):
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
        )
    )


def _add_ps64_layout(case, num_splits):
    kv_buffer, page_indices = _pack_kv_ps64(case["kv_logical"], case["seq_lens"])
    pages_per_seq = _pages_per_seq(case["seq_lens"])
    case.update(
        {
            "query_ps64": _pack_q_ps64(case["query"]),
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


def _torch_reference(case, softmax_scale):
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

    for batch_id, seq_len in enumerate(case["seq_lens"]):
        begin = case["kv_offsets"][batch_id]
        end = case["kv_offsets"][batch_id + 1]
        kv = kv_logical[begin:end]
        for q_pos in range(q_seq_len):
            q_row = batch_id * q_seq_len + q_pos
            valid_kv_len = max(seq_len - (q_seq_len - 1 - q_pos), 0)
            if valid_kv_len == 0:
                output[q_row].zero_()
                continue
            valid_kv = kv[:valid_kv_len]
            logits = torch.matmul(query[q_row], valid_kv.transpose(0, 1)) * score_scale
            probabilities = torch.softmax(logits, dim=-1)
            output[q_row] = (
                torch.matmul(probabilities, valid_kv[:, :V_HEAD_DIM]) * kv_scale
            )
    return output


def _decode_output(case):
    return torch.empty(
        (
            len(case["seq_lens"]) * case["q_seq_len"],
            case["nhead"],
            V_HEAD_DIM,
        ),
        dtype=torch.bfloat16,
    )


def _run_ps1(case, softmax_scale, output):
    mla_decode_fwd(
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
        causal=True,
    )


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


def _check_output(name, reference, output):
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


def _prepare_case(batch, ctx_len, q_seq_len, nhead, varlen, min_ratio, scales):
    if batch < 1 or ctx_len < 1:
        raise ValueError(
            f"batch and ctx_len must be positive, got {batch=}, {ctx_len=}"
        )
    torch.manual_seed(_seed_for(batch, ctx_len, q_seq_len, varlen, min_ratio, nhead))
    seq_lens = _make_seq_lens(batch, ctx_len, varlen, min_ratio)
    return _build_logical_case(seq_lens, q_seq_len, nhead, scales)


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
    softmax_scale = 1.0 / (QK_HEAD_DIM**0.5)
    reference = _torch_reference(case, softmax_scale)
    output = _decode_output(case)

    def run():
        _run_ps1(case, softmax_scale, output)

    _, total_us = run_perftest(run, num_iters=num_iters, num_warmup=num_warmup)
    return {
        "mode": "ps1",
        "batch": batch,
        "min_ctx": min(case["seq_lens"]),
        "max_ctx": max(case["seq_lens"]),
        "nhead": nhead,
        "q_seq": q_seq_len,
        "total us": total_us,
        "err": _check_output("ps1", reference, output),
    }


def test_ps1_persistent_vs_asm_ps64(
    batch=4,
    ctx_len=4096,
    q_seq_len=1,
    varlen=False,
    varlen_min_ratio=0.5,
    num_splits=0,
    num_iters=_PERF_NUM_ITERS,
    num_warmup=_PERF_NUM_WARMUP,
    nhead=16,
    scales="unit",
):
    if q_seq_len not in COMPARE_Q_SEQ_LENS[nhead]:
        raise ValueError(
            f"PS1-vs-ASM supports q_seq={COMPARE_Q_SEQ_LENS[nhead]} "
            f"for nhead={nhead}, got {q_seq_len}"
        )
    case = _prepare_case(
        batch,
        ctx_len,
        q_seq_len,
        nhead,
        varlen,
        varlen_min_ratio,
        scales,
    )
    if not num_splits:
        num_splits = _auto_num_splits(batch, case["seq_lens"], q_seq_len, nhead)
    _add_ps1_layout(case)
    _add_ps64_layout(case, num_splits)

    softmax_scale = 1.0 / (QK_HEAD_DIM**0.5)
    reference = _torch_reference(case, softmax_scale)
    ps1_output = _decode_output(case)
    asm_output = _decode_output(case)

    def run_ps1():
        _run_ps1(case, softmax_scale, ps1_output)

    def run_asm():
        _run_asm_ps64(case, num_splits, softmax_scale, asm_output)

    _, ps1_us = run_perftest(run_ps1, num_iters=num_iters, num_warmup=num_warmup)
    _, asm_us = run_perftest(run_asm, num_iters=num_iters, num_warmup=num_warmup)
    return {
        "mode": "ps1-vs-asm",
        "batch": batch,
        "min_ctx": min(case["seq_lens"]),
        "max_ctx": max(case["seq_lens"]),
        "nhead": nhead,
        "q_seq": q_seq_len,
        "asm_splits": num_splits,
        "ps1_works": case["num_works"],
        "ps1 total us": ps1_us,
        "asm total us": asm_us,
        "speedup": asm_us / ps1_us,
        "ps1 err": _check_output("ps1", reference, ps1_output),
        "asm err": _check_output("asm_ps64", reference, asm_output),
    }


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
    softmax_scale = 1.0 / (QK_HEAD_DIM**0.5)
    reference = _torch_reference(case, softmax_scale)

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
            softmax_scale,
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
        _run_asm_ps64(case, num_splits, softmax_scale, asm_output)

    _, flydsl_us = run_perftest(run_flydsl, num_iters=num_iters, num_warmup=num_warmup)
    _, asm_us = run_perftest(run_asm, num_iters=num_iters, num_warmup=num_warmup)
    return {
        "mode": "ps64",
        "batch": batch,
        "min_ctx": min(case["seq_lens"]),
        "max_ctx": max(case["seq_lens"]),
        "splits": num_splits,
        "flydsl total us": flydsl_us,
        "asm total us": asm_us,
        "speedup": asm_us / flydsl_us,
        "flydsl err": _check_output("flydsl_ps64", reference, flydsl_output),
        "asm err": _check_output("asm_ps64", reference, asm_output),
    }


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
    """Backward-compatible entry point for the original PS1/PS64 test."""
    if page_size == 1:
        if q_seq_len not in PS1_Q_SEQ_LENS[num_q_heads]:
            raise ValueError(
                f"PS1 supports q_seq={PS1_Q_SEQ_LENS[num_q_heads]} "
                f"for nhead={num_q_heads}, got {q_seq_len}"
            )
        return _test_ps1(
            batch,
            ctx_len,
            num_q_heads,
            q_seq_len,
            num_iters,
            num_warmup,
            "poc",
        )
    if page_size == PS64_PAGE_SIZE:
        if num_q_heads != PS64_NUM_Q_HEADS or q_seq_len != 1:
            raise ValueError("PS64 FlyDSL requires nhead=128 and q_seq_len=1")
        return _test_ps64(
            batch,
            ctx_len,
            num_splits,
            num_iters,
            num_warmup,
            "poc",
        )
    raise ValueError(f"unsupported page_size={page_size}; expected 1 or 64")


def _parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="Validate and compare gfx1250 FlyDSL MLA decode kernels.",
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
        default=[16, 32, 64, 128],
    )
    parser.add_argument(
        "--q-seq-len",
        type=int,
        nargs="+",
        choices=(1, 2, 3, 4),
        default=[1, 2, 3, 4],
    )
    parser.add_argument(
        "--split-kv",
        type=int,
        nargs="+",
        default=[0],
        help="ASM/PS64 split counts; 0 selects automatically.",
    )
    parser.add_argument("--varlen", action="store_true")
    parser.add_argument("--varlen-min-ratio", type=float, default=0.5)
    parser.add_argument("--scales", choices=SCALE_MODES, default="unit")
    parser.add_argument("--num-iters", type=int, default=_PERF_NUM_ITERS)
    parser.add_argument("--num-warmup", type=int, default=_PERF_NUM_WARMUP)
    return parser.parse_args()


def main():
    if get_gfx() not in SUPPORTED_GFX:
        aiter.logger.warning("FlyDSL MLA tests unsupported on %s; skipping", get_gfx())
        return

    args = _parse_args()
    rows = []
    if args.mode == "ps64":
        for batch, ctx_len, num_splits in itertools.product(
            args.batch, args.ctx_len, args.split_kv
        ):
            rows.append(
                _test_ps64(
                    batch,
                    ctx_len,
                    num_splits,
                    args.num_iters,
                    args.num_warmup,
                    args.scales,
                    args.varlen,
                    args.varlen_min_ratio,
                )
            )
    else:
        supported = PS1_Q_SEQ_LENS if args.mode == "ps1" else COMPARE_Q_SEQ_LENS
        for nhead, q_seq_len, batch, ctx_len, num_splits in itertools.product(
            args.num_heads,
            args.q_seq_len,
            args.batch,
            args.ctx_len,
            args.split_kv if args.mode == "ps1-vs-asm" else [0],
        ):
            if q_seq_len not in supported[nhead]:
                continue
            if args.mode == "ps1":
                rows.append(
                    _test_ps1(
                        batch,
                        ctx_len,
                        nhead,
                        q_seq_len,
                        args.num_iters,
                        args.num_warmup,
                        args.scales,
                        args.varlen,
                        args.varlen_min_ratio,
                    )
                )
            else:
                rows.append(
                    test_ps1_persistent_vs_asm_ps64(
                        batch,
                        ctx_len,
                        q_seq_len,
                        args.varlen,
                        args.varlen_min_ratio,
                        num_splits,
                        args.num_iters,
                        args.num_warmup,
                        nhead,
                        args.scales,
                    )
                )

    aiter.logger.info(
        "FlyDSL MLA %s summary:\n%s",
        args.mode,
        pd.DataFrame(rows).to_markdown(index=False),
    )


if __name__ == "__main__":
    main()
