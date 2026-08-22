# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Accuracy and performance test for the gfx1250 page-size-1 FlyDSL MLA kernel.

Examples:
  python3 op_tests/test_mla_pagesize1_flydsl.py
  python3 op_tests/test_mla_pagesize1_flydsl.py -b 1 16 -c 63 64 65 8192
"""

import argparse
import itertools

import pandas as pd
import torch

import aiter
from aiter import dtypes
from aiter.jit.utils.chip_info import get_gfx
from aiter.test_common import benchmark, checkAllclose, run_perftest

torch.set_default_device("cuda")

SUPPORTED_GFX = ["gfx1250"]

PAGE_SIZE = 1
NUM_Q_HEADS = 128
QK_NOPE_HEAD_DIM = 512
QK_ROPE_HEAD_DIM = 64
QK_HEAD_DIM = QK_NOPE_HEAD_DIM + QK_ROPE_HEAD_DIM
V_HEAD_DIM = QK_NOPE_HEAD_DIM

# The kernel consumes all 128 Q heads in one work item. Passing 16 to the
# generic metadata planner prevents its host-side 128-to-16-head folding.
METADATA_NUM_Q_HEADS = 16
KV_GRANULARITY = 16
MAX_SPLIT_PER_BATCH = 16

_SEED = 20260818
_PERF_NUM_ITERS = 101
_PERF_NUM_WARMUP = 5


def _allocate_metadata(batch):
    metadata_info = aiter.get_mla_metadata_info_v1(
        batch,
        1,
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


def _build_case(batch, ctx_len):
    if batch < 1 or ctx_len < 1:
        raise ValueError(
            f"batch and ctx_len must be positive, got {batch=}, {ctx_len=}"
        )

    torch.manual_seed(_SEED + batch * 1009 + ctx_len * 17)
    device = torch.device("cuda")
    total_pages = batch * ctx_len

    query = torch.randn(
        (batch, NUM_Q_HEADS, QK_HEAD_DIM),
        dtype=torch.bfloat16,
        device=device,
    ).to(dtypes.fp8)
    logical_kv = torch.randn(
        (total_pages, PAGE_SIZE, 1, QK_HEAD_DIM),
        dtype=torch.bfloat16,
        device=device,
    ).to(dtypes.fp8)

    # Scatter logical tokens so the test catches broken kv_page_indices gathers.
    kv_page_indices = torch.randperm(total_pages, device=device).to(torch.int32)
    kv_buffer = torch.empty_like(logical_kv)
    kv_buffer[kv_page_indices.long()] = logical_kv

    qo_indptr = torch.arange(batch + 1, dtype=torch.int32, device=device)
    kv_indptr = torch.arange(batch + 1, dtype=torch.int32, device=device) * ctx_len
    kv_last_page_lens = torch.ones(batch, dtype=torch.int32, device=device)

    (
        work_meta_data,
        work_indptr,
        work_info_set,
        reduce_indptr,
        reduce_final_map,
        reduce_partial_map,
    ) = _allocate_metadata(batch)
    aiter.get_mla_metadata_v1(
        qo_indptr,
        kv_indptr,
        kv_last_page_lens,
        METADATA_NUM_Q_HEADS,
        1,
        False,
        work_meta_data,
        work_info_set,
        work_indptr,
        reduce_indptr,
        reduce_final_map,
        reduce_partial_map,
        page_size=PAGE_SIZE,
        kv_granularity=KV_GRANULARITY,
        max_seqlen_qo=1,
        uni_seqlen_qo=1,
        fast_mode=True,
        max_split_per_batch=MAX_SPLIT_PER_BATCH,
        intra_batch_mode=False,
        dtype_q_nope=dtypes.fp8,
        dtype_kv_nope=dtypes.fp8,
    )

    num_works = int(work_indptr[-1].item())
    work_info = work_info_set[:num_works]
    partial_locations = work_info[:, 1]
    if bool((partial_locations < 0).any()):
        raise RuntimeError("page-size-1 FlyDSL stage 1 requires split work items")
    if bool((work_info[:, 3] - work_info[:, 2] != 1).any()):
        raise RuntimeError(
            "page-size-1 FlyDSL kernel only supports one query per work item"
        )

    return {
        "query": query,
        "kv_buffer": kv_buffer,
        "kv_page_indices": kv_page_indices,
        "work_indptr": work_indptr,
        "work_info": work_info,
        "num_pages": total_pages,
        "num_works": num_works,
        "num_partials": int(partial_locations.max().item()) + 1,
    }


def _torch_stage1_reference(case, softmax_scale):
    ref_data = torch.empty(
        (case["num_partials"], NUM_Q_HEADS, V_HEAD_DIM),
        dtype=torch.float32,
        device=case["query"].device,
    )
    ref_lse = torch.empty(
        (case["num_partials"], NUM_Q_HEADS),
        dtype=torch.float32,
        device=case["query"].device,
    )

    query = case["query"].float()
    kv_buffer = case["kv_buffer"][:, 0, 0].float()
    for work_idx in range(case["num_works"]):
        row = case["work_info"][work_idx]
        partial_location = int(row[1].item())
        qo_start = int(row[2].item())
        kv_start = int(row[4].item())
        kv_end = int(row[5].item())

        physical_pages = case["kv_page_indices"][kv_start:kv_end].long()
        kv = kv_buffer.index_select(0, physical_pages)
        logits = torch.matmul(query[qo_start], kv.transpose(0, 1))
        logits *= softmax_scale
        probabilities = torch.softmax(logits, dim=-1)
        ref_data[partial_location] = torch.matmul(probabilities, kv[:, :V_HEAD_DIM])
        ref_lse[partial_location] = torch.logsumexp(logits, dim=-1)

    return ref_data, ref_lse


@benchmark()
def test_mla_pagesize1_flydsl(
    batch=1,
    ctx_len=65,
    num_iters=_PERF_NUM_ITERS,
    num_warmup=_PERF_NUM_WARMUP,
):
    from aiter.ops.flydsl import flydsl_mla_pagesize1_fp8_fp8

    case = _build_case(batch, ctx_len)
    softmax_scale = 1.0 / (QK_HEAD_DIM**0.5)
    ref_data, ref_lse = _torch_stage1_reference(case, softmax_scale)

    split_data = torch.empty(
        (case["num_partials"], NUM_Q_HEADS, V_HEAD_DIM),
        dtype=torch.float32,
    )
    split_lse = torch.empty(
        (case["num_partials"], NUM_Q_HEADS),
        dtype=torch.float32,
    )

    def run_flydsl():
        flydsl_mla_pagesize1_fp8_fp8(
            split_data,
            split_lse,
            case["query"],
            case["kv_buffer"],
            case["kv_page_indices"],
            case["work_indptr"],
            case["work_info"],
            softmax_scale,
        )
        return split_data, split_lse

    (actual_data, actual_lse), us = run_perftest(
        run_flydsl,
        num_iters=num_iters,
        num_warmup=num_warmup,
    )
    assert torch.isfinite(actual_data).all(), "flydsl: non-finite split_data"
    assert torch.isfinite(actual_lse).all(), "flydsl: non-finite split_lse"

    data_err = checkAllclose(
        ref_data,
        actual_data,
        rtol=6e-2,
        atol=6e-2,
        tol_err_ratio=0.05,
        msg="flydsl: MLA page-size-1 split_data",
    )
    lse_err = checkAllclose(
        ref_lse,
        actual_lse,
        rtol=6e-2,
        atol=6e-2,
        tol_err_ratio=0.05,
        msg="flydsl: MLA page-size-1 split_lse",
    )
    err = max(data_err, lse_err)
    assert err <= 0.05, f"flydsl: mismatch ratio {err:.2%} exceeds 5%"

    total_kv = batch * ctx_len
    flops = 2 * total_kv * NUM_Q_HEADS * (QK_HEAD_DIM + V_HEAD_DIM)
    nbytes = (
        case["num_works"] * NUM_Q_HEADS * QK_HEAD_DIM
        + total_kv * (QK_HEAD_DIM + 4)
        + case["num_partials"] * NUM_Q_HEADS * (V_HEAD_DIM * 4 + 4)
        + case["num_works"] * 8 * 4
    )

    return {
        "gfx": get_gfx(),
        "batch": batch,
        "ctx_len": ctx_len,
        "num_works": case["num_works"],
        "flydsl us": us,
        "flydsl TFLOPS": flops / us / 1e6,
        "flydsl TB/s": nbytes / us / 1e6,
        "data err": data_err,
        "lse err": lse_err,
    }


def main():
    if get_gfx() not in SUPPORTED_GFX:
        aiter.logger.warning(
            "flydsl_mla_pagesize1_fp8_fp8 unsupported on %s; skipping", get_gfx()
        )
        return

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="Validate and benchmark gfx1250 FlyDSL MLA page-size-1 stage 1",
    )
    parser.add_argument(
        "-b",
        "--batch",
        type=int,
        nargs="*",
        default=[16],
        help="Batch sizes. e.g.: -b 1 4 16",
    )
    parser.add_argument(
        "-c",
        "--ctx-len",
        type=int,
        nargs="*",
        default=[1024, 2048, 5000, 8192, 16384, 32768],
        help="Context lengths. e.g.: -c 63 64 65 1024",
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

    rows = [
        test_mla_pagesize1_flydsl(
            batch,
            ctx_len,
            num_iters=args.num_iters,
            num_warmup=args.num_warmup,
        )
        for batch, ctx_len in itertools.product(args.batch, args.ctx_len)
    ]
    df = pd.DataFrame(rows)
    aiter.logger.info(
        "flydsl_mla_pagesize1_fp8_fp8 summary (markdown):\n%s",
        df.to_markdown(index=False),
    )


if __name__ == "__main__":
    main()
