# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Accuracy and performance comparison for gfx1250 FlyDSL and ASM MLA stage 1.

Examples:
  python3 op_tests/test_mla_pagesize64_fp8_fp8_flydsl.py
  python3 op_tests/test_mla_pagesize64_fp8_fp8_flydsl.py -b 1 -c 65 --split-kv 1 2
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

PAGE_SIZE = 64
NUM_Q_HEADS = 128
QK_NOPE_HEAD_DIM = 512
QK_ROPE_HEAD_DIM = 64
QK_HEAD_DIM = QK_NOPE_HEAD_DIM + QK_ROPE_HEAD_DIM
V_HEAD_DIM = QK_NOPE_HEAD_DIM
Q_HEAD_STRIDE = 768

_SEED = 20260807
_PERF_NUM_ITERS = 101
_PERF_NUM_WARMUP = 5


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


def _pack_kv_pages(kv):
    num_pages = kv.size(0)
    packed = torch.cat(
        (
            kv[..., :QK_NOPE_HEAD_DIM].reshape(num_pages, PAGE_SIZE * QK_NOPE_HEAD_DIM),
            kv[..., QK_NOPE_HEAD_DIM:].reshape(num_pages, PAGE_SIZE * QK_ROPE_HEAD_DIM),
        ),
        dim=-1,
    )
    return packed.contiguous()


def _build_case(batch, ctx_len, num_splits):
    torch.manual_seed(_SEED + batch * 1009 + ctx_len * 17 + num_splits)
    device = torch.device("cuda")
    num_pages_per_batch = (ctx_len + PAGE_SIZE - 1) // PAGE_SIZE
    total_pages = batch * num_pages_per_batch
    last_page_len = ctx_len % PAGE_SIZE or PAGE_SIZE

    q_ref = torch.randn(
        (batch, NUM_Q_HEADS, QK_HEAD_DIM),
        dtype=torch.bfloat16,
        device=device,
    ).to(dtypes.fp8)
    q = _pack_q(q_ref)

    kv_logical = torch.randn(
        (total_pages, PAGE_SIZE, 1, QK_HEAD_DIM),
        dtype=torch.bfloat16,
        device=device,
    )
    if last_page_len != PAGE_SIZE:
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
    kv_buffer = _pack_kv_pages(kv_ref)

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
        "q_scale": q_scale,
        "kv_scale": kv_scale,
        "num_pages_per_batch": num_pages_per_batch,
        "last_page_len": last_page_len,
    }


def _torch_stage1_reference(case, batch, num_splits, softmax_scale):
    ref_data = torch.empty(
        (batch, num_splits, NUM_Q_HEADS, V_HEAD_DIM),
        dtype=torch.float32,
        device=case["q_ref"].device,
    )
    ref_lse = torch.empty(
        (batch, num_splits, NUM_Q_HEADS, 1),
        dtype=torch.float32,
        device=case["q_ref"].device,
    )
    num_pages = case["num_pages_per_batch"]
    q_scale = case["q_scale"][0]
    kv_scale = case["kv_scale"][0]

    for batch_id in range(batch):
        q = case["q_ref"][batch_id].float() * q_scale
        page_base = batch_id * num_pages
        for split_id in range(min(num_pages, num_splits)):
            kv_chunks = []
            for local_page in range(split_id, num_pages, num_splits):
                physical_page = case["kv_page_indices"][page_base + local_page].long()
                valid_len = (
                    case["last_page_len"] if local_page == num_pages - 1 else PAGE_SIZE
                )
                kv_chunks.append(
                    case["kv_ref"][physical_page, :valid_len, 0].float() * kv_scale
                )
            kv = torch.cat(kv_chunks, dim=0)
            logits = torch.matmul(q, kv.transpose(0, 1)) * softmax_scale
            probabilities = torch.softmax(logits, dim=-1)
            ref_data[batch_id, split_id] = torch.matmul(
                probabilities, kv[:, :V_HEAD_DIM]
            )
            ref_lse[batch_id, split_id, :, 0] = torch.logsumexp(logits, dim=-1)

    return ref_data, ref_lse


@benchmark()
def test_mla_pagesize64_flydsl(batch=1, ctx_len=65, num_splits=1):
    from aiter.ops.flydsl.mla_kernels import (
        flydsl_mla_pagesize64_fp8_fp8,
    )

    case = _build_case(batch, ctx_len, num_splits)
    softmax_scale = 1.0 / (QK_HEAD_DIM**0.5)
    ref_data, ref_lse = _torch_stage1_reference(case, batch, num_splits, softmax_scale)

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
    asm_output = torch.empty(output_shape, dtype=torch.bfloat16)
    asm_split_data = (
        asm_output.view(split_shape)
        if num_splits == 1
        else torch.empty(split_shape, dtype=torch.float32)
    )
    asm_split_lse = torch.empty_like(flydsl_split_lse)
    valid_split_count = torch.full((batch,), num_splits, dtype=torch.int32)

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
            page_size=PAGE_SIZE,
        )
        return flydsl_split_data, flydsl_split_lse

    def run_asm():
        aiter.mla_decode_stage1_asm_fwd(
            case["q"],
            case["kv_buffer"].view(-1, PAGE_SIZE, 1, QK_HEAD_DIM),
            case["qo_indptr"],
            case["kv_indptr"],
            case["kv_page_indices"],
            case["kv_last_page_lens"],
            case["num_kv_splits_indptr"],
            None,
            None,
            None,
            1,
            PAGE_SIZE,
            1,
            softmax_scale,
            asm_split_data,
            asm_split_lse,
            asm_output,
            None,
            case["q_scale"],
            case["kv_scale"],
            None,
            1,
            0,
            valid_split_count,
            int(num_splits > 1),
        )
        return asm_split_data, asm_split_lse

    candidates = {
        "flydsl": run_flydsl,
        "asm": run_asm,
    }
    valid_splits = min(case["num_pages_per_batch"], num_splits)
    expected_data = ref_data[:, :valid_splits]
    if num_splits == 1:
        expected_data = expected_data.to(torch.bfloat16)

    total_q = batch
    total_kv = batch * ctx_len
    flops = 2 * total_kv * NUM_Q_HEADS * (QK_HEAD_DIM + V_HEAD_DIM)
    output_element_size = 2 if num_splits == 1 else 4
    nbytes = (
        total_q * NUM_Q_HEADS * QK_HEAD_DIM
        + total_kv * QK_HEAD_DIM
        + batch * valid_splits * NUM_Q_HEADS * (V_HEAD_DIM * output_element_size + 4)
    )

    ret = {
        "gfx": get_gfx(),
        "valid_splits": valid_splits,
    }
    for name, fn in candidates.items():
        (actual_data, actual_lse), us = run_perftest(
            fn,
            num_iters=_PERF_NUM_ITERS,
            num_warmup=_PERF_NUM_WARMUP,
        )
        actual_data = actual_data.reshape(batch, num_splits, NUM_Q_HEADS, V_HEAD_DIM)[
            :, :valid_splits
        ]
        actual_lse = actual_lse[:, :valid_splits]
        assert torch.isfinite(actual_data).all(), f"{name}: non-finite split_data"
        assert torch.isfinite(actual_lse).all(), f"{name}: non-finite split_lse"

        data_err = checkAllclose(
            expected_data.to(dtypes.fp32),
            actual_data.to(dtypes.fp32),
            rtol=6e-2,
            atol=6e-2,
            tol_err_ratio=0.05,
            msg=f"{name}: MLA stage1 split_data",
        )
        lse_err = checkAllclose(
            ref_lse[:, :valid_splits].to(dtypes.fp32),
            actual_lse.to(dtypes.fp32),
            rtol=6e-2,
            atol=6e-2,
            tol_err_ratio=0.05,
            msg=f"{name}: MLA stage1 split_lse",
        )
        err = max(data_err, lse_err)
        assert err <= 0.05, f"{name}: mismatch ratio {err:.2%} exceeds 5%"

        ret[f"{name} us"] = us
        ret[f"{name} TFLOPS"] = flops / us / 1e6
        ret[f"{name} TB/s"] = nbytes / us / 1e6
        ret[f"{name} err"] = err
        ret[f"{name} data err"] = data_err
        ret[f"{name} lse err"] = lse_err
    return ret


def main():
    if get_gfx() not in SUPPORTED_GFX:
        aiter.logger.warning(
            "flydsl_mla_pagesize64_fp8_fp8 unsupported on %s; skipping", get_gfx()
        )
        return

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="Compare gfx1250 FlyDSL and ASM MLA page-size-64 stage 1",
    )
    parser.add_argument(
        "-b",
        "--batch",
        type=int,
        nargs="*",
        default=[1, 2, 16],
        help="Batch sizes. e.g.: -b 1 4",
    )
    parser.add_argument(
        "-c",
        "--ctx-len",
        type=int,
        nargs="*",
        default=[2048, 5000, 8192],
        help="Context lengths. e.g.: -c 64 65 1024",
    )
    parser.add_argument(
        "--split-kv",
        type=int,
        nargs="*",
        default=[16],
        help="KV split counts. e.g.: --split-kv 1 2 4",
    )
    args = parser.parse_args()

    rows = [
        test_mla_pagesize64_flydsl(batch, ctx_len, num_splits)
        for batch, ctx_len, num_splits in itertools.product(
            args.batch, args.ctx_len, args.split_kv
        )
    ]
    df = pd.DataFrame(rows)
    aiter.logger.info(
        "flydsl_mla_pagesize64_fp8_fp8 summary (markdown):\n%s",
        df.to_markdown(index=False),
    )


if __name__ == "__main__":
    main()
