# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Accuracy and timing tests for gfx1250 FlyDSL MLA decode kernels.

page_size=1 exercises the production path via ``aiter.mla.mla_decode_fwd``
(FlyDSL stage-1 when ``AITER_MLA_DECODE_PS1_FLYDSL=1`` + HIP reduce inside
the wrapper).

page_size=64 compares FlyDSL kernels against ``mla_decode_fwd`` (ASM PS64 +
Triton stage-2). FlyDSL PS64 is not routed through ``mla_decode_fwd`` yet.

For page_size=64 the split count defaults to whatever mla_decode_fwd's heuristic
picks for the shape; --split-kv pins it instead.

Examples:
  python3 op_tests/test_mla_flydsl.py
  python3 op_tests/test_mla_flydsl.py -p 1 -b 1 16 -c 63 64 65 8192
  python3 op_tests/test_mla_flydsl.py -p 64 -b 1 -c 65 --split-kv 1 2
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

SUPPORTED_GFX = ["gfx1250"]
SUPPORTED_PAGE_SIZES = (1, 64)
SUPPORTED_PS1_NUM_Q_HEADS = (16, 32, 64, 128)

NUM_Q_HEADS = 128
QK_NOPE_HEAD_DIM = 512
QK_ROPE_HEAD_DIM = 64
QK_HEAD_DIM = QK_NOPE_HEAD_DIM + QK_ROPE_HEAD_DIM
V_HEAD_DIM = QK_NOPE_HEAD_DIM
Q_HEAD_STRIDE = 768

# The page-size-1 kernel consumes all 128 Q heads in one work item. Passing 16
# to the generic metadata planner prevents its host-side 128-to-16-head folding.
METADATA_NUM_Q_HEADS = 16
KV_GRANULARITY = 16
MAX_SPLIT_PER_BATCH = 16

_SEED = 20260818
_PERF_NUM_ITERS = 101
_PERF_NUM_WARMUP = 5


def _allocate_metadata(batch, q_seq_len):
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


def _build_case_ps1(batch, ctx_len, num_q_heads, q_seq_len):
    if batch < 1 or ctx_len < 1:
        raise ValueError(
            f"batch and ctx_len must be positive, got {batch=}, {ctx_len=}"
        )
    if q_seq_len not in (1, 2, 3, 4):
        raise ValueError(f"q_seq_len must be in [1, 4], got {q_seq_len}")
    if num_q_heads not in SUPPORTED_PS1_NUM_Q_HEADS:
        raise ValueError(
            f"unsupported page-size-1 num_q_heads={num_q_heads}; "
            f"expected one of {SUPPORTED_PS1_NUM_Q_HEADS}"
        )
    if num_q_heads != 16 and q_seq_len != 1:
        raise ValueError(
            f"page-size-1 {num_q_heads}-head case only supports q_seq_len=1"
        )

    torch.manual_seed(_SEED + batch * 1009 + ctx_len * 17 + q_seq_len * 101)
    device = torch.device("cuda")
    total_pages = batch * ctx_len

    query = torch.randn(
        (batch * q_seq_len, num_q_heads, QK_HEAD_DIM),
        dtype=torch.bfloat16,
        device=device,
    ).to(dtypes.fp8)
    logical_kv = torch.randn(
        (total_pages, 1, 1, QK_HEAD_DIM),
        dtype=torch.bfloat16,
        device=device,
    ).to(dtypes.fp8)

    # Scatter logical tokens so the test catches broken kv_page_indices gathers.
    kv_page_indices = torch.randperm(total_pages, device=device).to(torch.int32)
    kv_buffer = torch.empty_like(logical_kv)
    kv_buffer[kv_page_indices.long()] = logical_kv

    qo_indptr = torch.arange(batch + 1, dtype=torch.int32, device=device) * q_seq_len
    kv_indptr = torch.arange(batch + 1, dtype=torch.int32, device=device) * ctx_len
    kv_last_page_lens = torch.ones(batch, dtype=torch.int32, device=device)
    # Non-unit descales so both places the kernel applies them -- the QK
    # temperature and the PV output -- are covered.
    q_scale = torch.tensor([0.75], dtype=torch.float32, device=device)
    kv_scale = torch.tensor([1.20], dtype=torch.float32, device=device)

    (
        work_meta_data,
        work_indptr,
        work_info_set,
        reduce_indptr,
        reduce_final_map,
        reduce_partial_map,
    ) = _allocate_metadata(batch, q_seq_len)
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
    if bool((work_info[:, 3] - work_info[:, 2] != q_seq_len).any()):
        raise RuntimeError(
            "page-size-1 FlyDSL test requires one full query tile per work item"
        )

    return {
        "query": query,
        "kv_buffer": kv_buffer,
        "kv_page_indices": kv_page_indices,
        "qo_indptr": qo_indptr,
        "kv_indptr": kv_indptr,
        "kv_last_page_lens": kv_last_page_lens,
        "q_scale": q_scale,
        "kv_scale": kv_scale,
        "work_meta_data": work_meta_data,
        "work_indptr": work_indptr,
        "work_info_set": work_info_set,
        "reduce_indptr": reduce_indptr,
        "reduce_final_map": reduce_final_map,
        "reduce_partial_map": reduce_partial_map,
        "q_seq_len": q_seq_len,
    }


def _mla_decode_ps1(case, q_seq_len, softmax_scale, output, *, return_lse=False):
    return mla_decode_fwd(
        case["query"],
        case["kv_buffer"],
        output,
        case["qo_indptr"],
        case["kv_indptr"],
        case["kv_page_indices"],
        case["kv_last_page_lens"],
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
        return_lse=return_lse,
        causal=True,
    )


def _mla_decode_ps64(case, num_splits, softmax_scale, output):
    page_size = case["page_size"]
    mla_decode_fwd(
        case["q"],
        case["kv_buffer"].view(-1, page_size, 1, QK_HEAD_DIM),
        output,
        case["qo_indptr"],
        case["kv_indptr"],
        case["kv_page_indices"],
        case["kv_last_page_lens"],
        1,
        page_size=page_size,
        nhead_kv=1,
        sm_scale=softmax_scale,
        num_kv_splits=num_splits,
        num_kv_splits_indptr=case["num_kv_splits_indptr"],
        q_scale=case["q_scale"],
        kv_scale=case["kv_scale"],
        causal=True,
    )


def _torch_merged_reference_ps1(case, batch, ctx_len, softmax_scale):
    """Attention output over each batch's whole KV run.

    Independent of how the planner sliced KV, since the cross-split merge is
    mathematically a plain softmax over the concatenation of the splits.
    """
    query = case["query"].float()
    num_q_heads = query.size(1)
    q_seq_len = case["q_seq_len"]
    kv_buffer = case["kv_buffer"][:, 0, 0].float()
    kv_descale = float(case["kv_scale"][0])
    score_scale = softmax_scale * float(case["q_scale"][0]) * kv_descale
    ref_out = torch.empty(
        (batch * q_seq_len, num_q_heads, V_HEAD_DIM),
        dtype=torch.float32,
        device=query.device,
    )
    for batch_id in range(batch):
        physical_pages = case["kv_page_indices"][
            batch_id * ctx_len : (batch_id + 1) * ctx_len
        ].long()
        kv = kv_buffer.index_select(0, physical_pages)
        for q_pos in range(q_seq_len):
            q_row = batch_id * q_seq_len + q_pos
            valid_kv_len = ctx_len - (q_seq_len - 1 - q_pos)
            valid_kv = kv[:valid_kv_len]
            logits = torch.matmul(query[q_row], valid_kv.transpose(0, 1)) * score_scale
            probabilities = torch.softmax(logits, dim=-1)
            ref_out[q_row] = (
                torch.matmul(probabilities, valid_kv[:, :V_HEAD_DIM]) * kv_descale
            )
    return ref_out


def _test_mla_flydsl_ps1(batch, ctx_len, num_q_heads, q_seq_len, num_iters, num_warmup):
    case = _build_case_ps1(batch, ctx_len, num_q_heads, q_seq_len)
    softmax_scale = 1.0 / (QK_HEAD_DIM**0.5)
    total_q = batch * q_seq_len
    output = torch.empty((total_q, num_q_heads, V_HEAD_DIM), dtype=torch.bfloat16)

    def run_decode():
        _mla_decode_ps1(case, q_seq_len, softmax_scale, output)

    _, total_us = run_perftest(run_decode, num_iters=num_iters, num_warmup=num_warmup)
    assert torch.isfinite(output).all(), "mla_decode_fwd: non-finite output"
    ref_out = _torch_merged_reference_ps1(case, batch, ctx_len, softmax_scale)
    out_err = checkAllclose(
        ref_out,
        output.to(torch.float32),
        rtol=6e-2,
        atol=6e-2,
        tol_err_ratio=0.05,
        msg="mla_decode_fwd: MLA page-size-1 merged output",
    )
    assert out_err <= 0.05, f"merged output mismatch ratio {out_err:.2%} exceeds 5%"

    return {
        "page_size": 1,
        "batch": batch,
        "ctx_len": ctx_len,
        "num_q_heads": num_q_heads,
        "q_seq_len": q_seq_len,
        "total us": total_us,
        "err": out_err,
    }


def _auto_num_splits(batch, ctx_len, page_size):
    """The split count mla_decode_fwd would pick, so the sweep spends its time on
    the shapes production actually runs instead of a pinned count.

    Mirrors that wrapper's non-persistent branch, including the argument quirk
    that dominates the result here: it passes `total_kv = kv_indices.shape[0]`,
    which at page_size=64 is a page count rather than a token count, so the fp8
    min-block cap (ceil(pages / 32) per seq) clamps 64x harder than reading it as
    tokens would suggest. The uniform indptr get_meta_param returns alongside the
    count is the same `arange(batch + 1) * num_splits` that _build_case_ps64
    builds.
    """
    num_pages_per_batch = (ctx_len + page_size - 1) // page_size
    num_splits, _ = get_meta_param(
        None,
        batch,
        batch * num_pages_per_batch,
        NUM_Q_HEADS,
        1,  # max_seqlen_q
        dtypes.fp8,
    )
    return int(num_splits)


def _pack_q(q):
    padded = torch.zeros(
        (q.size(0), NUM_Q_HEADS, Q_HEAD_STRIDE),
        dtype=q.dtype,
        device=q.device,
    )
    padded[..., :QK_HEAD_DIM].copy_(q)
    return torch.as_strided(
        padded,
        size=q.shape,
        stride=(NUM_Q_HEADS * Q_HEAD_STRIDE, Q_HEAD_STRIDE, 1),
    )


def _pack_kv_pages(kv, page_size):
    num_pages = kv.size(0)
    packed = torch.cat(
        (
            kv[..., :QK_NOPE_HEAD_DIM].reshape(num_pages, page_size * QK_NOPE_HEAD_DIM),
            kv[..., QK_NOPE_HEAD_DIM:].reshape(num_pages, page_size * QK_ROPE_HEAD_DIM),
        ),
        dim=-1,
    )
    return packed.contiguous()


def _build_case_ps64(batch, ctx_len, num_splits, page_size):
    torch.manual_seed(_SEED + batch * 1009 + ctx_len * 17 + num_splits)
    device = torch.device("cuda")
    num_pages_per_batch = (ctx_len + page_size - 1) // page_size
    total_pages = batch * num_pages_per_batch
    last_page_len = ctx_len % page_size or page_size

    q_ref = torch.randn(
        (batch, NUM_Q_HEADS, QK_HEAD_DIM),
        dtype=torch.bfloat16,
        device=device,
    ).to(dtypes.fp8)
    q = _pack_q(q_ref)

    kv_logical = torch.randn(
        (total_pages, page_size, 1, QK_HEAD_DIM),
        dtype=torch.bfloat16,
        device=device,
    )
    if last_page_len != page_size:
        last_pages = torch.arange(
            num_pages_per_batch - 1,
            total_pages,
            num_pages_per_batch,
            dtype=torch.int64,
            device=device,
        )
        kv_logical[last_pages, last_page_len:] = float("nan")
    kv_logical = kv_logical.to(dtypes.fp8)

    # Scatter logical pages so the test also covers kv_page_indices addressing.
    kv_page_indices = torch.randperm(total_pages, device=device).to(torch.int32)
    kv_ref = torch.empty_like(kv_logical)
    kv_ref[kv_page_indices.long()] = kv_logical
    kv_buffer = _pack_kv_pages(kv_ref, page_size)

    kv_indptr = (
        torch.arange(batch + 1, dtype=torch.int32, device=device) * num_pages_per_batch
    )
    qo_indptr = torch.arange(batch + 1, dtype=torch.int32, device=device)
    kv_last_page_lens = torch.full(
        (batch,), last_page_len, dtype=torch.int32, device=device
    )
    num_kv_splits_indptr = (
        torch.arange(batch + 1, dtype=torch.int32, device=device) * num_splits
    )
    seqused_k = torch.full((batch,), ctx_len, dtype=torch.int32, device=device)
    q_scale = torch.tensor([0.75], dtype=torch.float32, device=device)
    kv_scale = torch.tensor([1.20], dtype=torch.float32, device=device)

    return {
        "q": q,
        "q_ref": q_ref,
        "kv_buffer": kv_buffer,
        "kv_ref": kv_ref,
        "kv_indptr": kv_indptr,
        "kv_page_indices": kv_page_indices,
        "kv_last_page_lens": kv_last_page_lens,
        "qo_indptr": qo_indptr,
        "num_kv_splits_indptr": num_kv_splits_indptr,
        "seqused_k": seqused_k,
        "q_scale": q_scale,
        "kv_scale": kv_scale,
        "num_pages_per_batch": num_pages_per_batch,
        "last_page_len": last_page_len,
        "page_size": page_size,
    }


def _torch_reference_ps64(case, batch, softmax_scale):
    """Merged (post-reduce) attention output, computed over the whole KV run.

    Independent of how either backend slices KV into splits, since the merge is
    mathematically a plain softmax over the concatenation of all splits.
    """
    ref_out = torch.empty(
        (batch, NUM_Q_HEADS, V_HEAD_DIM),
        dtype=torch.float32,
        device=case["q_ref"].device,
    )
    num_pages = case["num_pages_per_batch"]
    page_size = case["page_size"]
    q_scale = case["q_scale"][0]
    kv_scale = case["kv_scale"][0]

    for batch_id in range(batch):
        q = case["q_ref"][batch_id].float() * q_scale
        page_base = batch_id * num_pages
        kv_chunks = []
        for local_page in range(num_pages):
            physical_page = case["kv_page_indices"][page_base + local_page].long()
            valid_len = (
                case["last_page_len"] if local_page == num_pages - 1 else page_size
            )
            kv_chunks.append(
                case["kv_ref"][physical_page, :valid_len, 0].float() * kv_scale
            )
        kv = torch.cat(kv_chunks, dim=0)
        logits = torch.matmul(q, kv.transpose(0, 1)) * softmax_scale
        probabilities = torch.softmax(logits, dim=-1)
        ref_out[batch_id] = torch.matmul(probabilities, kv[:, :V_HEAD_DIM])

    return ref_out


def _test_mla_flydsl_ps64(batch, ctx_len, page_size, num_splits, num_iters, num_warmup):
    from aiter.ops.flydsl.mla_kernels import (
        flydsl_mla_decode_reduce,
        flydsl_mla_pagesize64_fp8_fp8,
    )

    if not num_splits:
        num_splits = _auto_num_splits(batch, ctx_len, page_size)

    case = _build_case_ps64(batch, ctx_len, num_splits, page_size)
    softmax_scale = 1.0 / (QK_HEAD_DIM**0.5)
    ref_out = _torch_reference_ps64(case, batch, softmax_scale)

    output_shape = (batch, NUM_Q_HEADS, V_HEAD_DIM)
    split_shape = (batch, num_splits, NUM_Q_HEADS, V_HEAD_DIM)
    split_dtype = torch.bfloat16 if num_splits == 1 else torch.float32

    flydsl_split_data = torch.empty(
        output_shape if num_splits == 1 else split_shape,
        dtype=split_dtype,
    )
    flydsl_split_lse = torch.empty(
        (batch, num_splits, NUM_Q_HEADS, 1),
        dtype=torch.float32,
    )
    # num_splits == 1 needs no merge at all: both stage1 kernels write the final
    # bf16 result straight out, so the reduce output aliases the stage1 buffer.
    flydsl_output = (
        flydsl_split_data
        if num_splits == 1
        else torch.empty(output_shape, dtype=torch.bfloat16)
    )
    asm_output = torch.empty(output_shape, dtype=torch.bfloat16)

    def run_flydsl():
        flydsl_mla_pagesize64_fp8_fp8(
            split_data=flydsl_split_data,
            split_lse=flydsl_split_lse,
            q=case["q"],
            kv_buffer=case["kv_buffer"],
            kv_indptr=case["kv_indptr"],
            kv_page_indices=case["kv_page_indices"],
            kv_last_page_lens=case["kv_last_page_lens"],
            qo_indptr=case["qo_indptr"],
            num_kv_splits_indptr=case["num_kv_splits_indptr"],
            q_scale=case["q_scale"],
            kv_scale=case["kv_scale"],
            softmax_scale=softmax_scale,
            num_splits=num_splits,
            page_size=page_size,
        )

    def reduce_flydsl():
        flydsl_mla_decode_reduce(
            flydsl_split_data,
            flydsl_split_lse,
            case["seqused_k"],
            flydsl_output,
            num_splits,
            1,  # num_tokens_per_seq
        )

    def run_asm_decode():
        _mla_decode_ps64(case, num_splits, softmax_scale, asm_output)

    candidates = {
        "flydsl": (run_flydsl, reduce_flydsl, flydsl_output),
        "mla_decode_fwd": (run_asm_decode, None, asm_output),
    }
    ret = {
        "page_size": page_size,
        "batch": batch,
        "ctx_len": ctx_len,
        "num_splits": num_splits,
    }
    for name, (stage1, reduce_fn, actual_out) in candidates.items():

        def run_full(_s1=stage1, _reduce=reduce_fn, _ns=num_splits):
            _s1()
            if _reduce is not None and _ns > 1:
                _reduce()

        _, total_us = run_perftest(run_full, num_iters=num_iters, num_warmup=num_warmup)

        assert torch.isfinite(actual_out).all(), f"{name}: non-finite output"
        err = checkAllclose(
            ref_out,
            actual_out.to(dtypes.fp32),
            rtol=6e-2,
            atol=6e-2,
            tol_err_ratio=0.05,
            msg=f"{name}: MLA decode output",
        )
        assert err <= 0.05, f"{name}: mismatch ratio {err:.2%} exceeds 5%"
        ret[f"{name} total us"] = total_us
        ret[f"{name} err"] = err
    return ret


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
    if page_size == 1:
        return _test_mla_flydsl_ps1(
            batch, ctx_len, num_q_heads, q_seq_len, num_iters, num_warmup
        )
    if page_size == 64:
        if num_q_heads != NUM_Q_HEADS or q_seq_len != 1:
            raise ValueError(
                f"page_size=64 requires num_q_heads={NUM_Q_HEADS} and "
                f"q_seq_len=1, got {num_q_heads=} and {q_seq_len=}"
            )
        return _test_mla_flydsl_ps64(
            batch, ctx_len, page_size, num_splits, num_iters, num_warmup
        )
    raise ValueError(
        f"unsupported page_size={page_size}; expected one of {SUPPORTED_PAGE_SIZES}"
    )


def main():
    if get_gfx() not in SUPPORTED_GFX:
        aiter.logger.warning(
            "flydsl MLA kernels unsupported on %s; skipping", get_gfx()
        )
        return

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="Validate gfx1250 FlyDSL MLA decode kernels",
    )
    parser.add_argument(
        "-p",
        "--page-size",
        type=int,
        nargs="*",
        default=[1],
        choices=SUPPORTED_PAGE_SIZES,
        help="Page sizes. e.g.: -p 1 64",
    )
    parser.add_argument(
        "-b",
        "--batch",
        type=int,
        nargs="*",
        default=[4],
        help="Batch sizes. e.g.: -b 1 4 16",
    )
    parser.add_argument(
        "-c",
        "--ctx-len",
        type=int,
        nargs="*",
        default=[2048, 4096, 8192],
        help="Context lengths. e.g.: -c 63 64 65 1024",
    )
    parser.add_argument(
        "--split-kv",
        type=int,
        nargs="*",
        default=[0],
        help="KV split counts for page_size=64, 0 = pick per shape like "
        "mla_decode_fwd does.\ne.g.: --split-kv 0 1 2 4",
    )
    parser.add_argument(
        "--num-heads",
        type=int,
        nargs="*",
        default=[16, 32, 64, 128],
        choices=SUPPORTED_PS1_NUM_Q_HEADS,
        help="Q head counts for page_size=1. Only 16 heads run q_seq>1;\n"
        "page_size=64 supports only 128.",
    )
    parser.add_argument(
        "--q-seq-len",
        type=int,
        nargs="*",
        default=[1, 2, 3, 4],
        choices=(1, 2, 3, 4),
        help="Query sequence lengths for page_size=1 and 16 heads.",
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
    for page_size, batch, ctx_len in itertools.product(
        args.page_size, args.batch, args.ctx_len
    ):
        if page_size == 64:
            configs = [(num_splits, NUM_Q_HEADS, 1) for num_splits in args.split_kv]
        else:
            configs = [
                (0, num_q_heads, q_seq_len)
                for num_q_heads in args.num_heads
                for q_seq_len in args.q_seq_len
                if num_q_heads == 16 or q_seq_len == 1
            ]
        for num_splits, num_q_heads, q_seq_len in configs:
            rows.append(
                test_mla_flydsl(
                    page_size,
                    batch,
                    ctx_len,
                    num_splits=num_splits,
                    num_q_heads=num_q_heads,
                    q_seq_len=q_seq_len,
                    num_iters=args.num_iters,
                    num_warmup=args.num_warmup,
                )
            )
    df = pd.DataFrame(rows)
    aiter.logger.info(
        "flydsl MLA summary (markdown):\n%s",
        df.to_markdown(index=False),
    )


if __name__ == "__main__":
    main()
