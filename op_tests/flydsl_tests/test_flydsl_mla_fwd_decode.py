# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Correctness smoke test and benchmark for FlyDSL MLA decode."""

import argparse
import itertools
import math

import pandas as pd
import pytest
import torch

import aiter
from aiter import dtypes
from aiter.jit.utils.chip_info import get_gfx
from aiter.test_common import benchmark, checkAllclose, run_perftest

try:
    from aiter.ops.flydsl.kernels.mla_fwd_decode_m16x8_fp8_fp8 import (
        launch_mla_fwd_decode_m16x8_fp8_fp8,
    )
except (ImportError, AttributeError, RuntimeError, OSError):
    launch_mla_fwd_decode_m16x8_fp8_fp8 = None

torch.set_default_device("cuda")

SUPPORTED_GFX = ["gfx942", "gfx950"]
NUM_HEADS = 128
QK_HEAD_DIM = 576
V_HEAD_DIM = 512


# Pytorch reference, from FlyDSL tests/kernels/test_mla_decode.py

def ref_masked_attention(query, key, value, scale, dtype, q_scale=None, kv_scale=None):
    """Single-sequence MLA attention (no causal mask needed for decode_qlen=1)."""
    s = scale
    if q_scale is not None:
        s *= q_scale
    if kv_scale is not None:
        s *= kv_scale

    attn_weights = torch.einsum("qhd,khd->hqk", query.float(), key.float()) * s
    lse = attn_weights.logsumexp(dim=-1)
    m = attn_weights.max(-1).values
    attn_weights_exp = torch.exp(attn_weights - m.unsqueeze(-1))
    denom = attn_weights_exp.sum(-1)
    out = torch.einsum("hqk,khd->qhd", attn_weights_exp.float(), value.float())
    out = out / denom.transpose(0, 1).unsqueeze(-1)
    if kv_scale is not None:
        out *= kv_scale
    return out.to(dtype), lse


def torch_mla_extend(q, kvc_cache, qo_indptr, kv_indptr, kv_indices, kv_last_page_lens, sm_scale, dtype):
    """Pure-PyTorch paged MLA attention reference."""
    is_fp8_q = q.dtype in (torch.float8_e4m3fn, torch.float8_e4m3fnuz)
    is_fp8_kvc = kvc_cache.dtype in (torch.float8_e4m3fn, torch.float8_e4m3fnuz)

    if is_fp8_q:
        q = q.to(torch.float)
    if is_fp8_kvc:
        kvc_cache = kvc_cache.to(torch.float)

    num_page, page_size, nhead_kv, _ = kvc_cache.shape
    qs = torch.tensor_split(q, qo_indptr.tolist()[1:])
    kvc = torch.index_select(kvc_cache, 0, kv_indices)
    kvs = torch.tensor_split(kvc, kv_indptr.tolist()[1:])
    bs = qo_indptr.shape[0] - 1

    os_list = []
    lses = []
    for i in range(bs):
        cur_num_page = kvs[i].shape[0]
        real_kv_seq_len = (cur_num_page - 1) * page_size + kv_last_page_lens.tolist()[i]
        kvc_i = kvs[i].flatten(0, 1)[:real_kv_seq_len]
        q_i = qs[i]
        k_i = kvc_i
        v_i = kvc_i[:, :, :KV_LORA_RANK]
        o_i, lse_i = ref_masked_attention(q_i, k_i, v_i, sm_scale, dtype)
        os_list.append(o_i)
        lses.append(lse_i)

    o = torch.concat(os_list)
    lse = torch.concat(lses, dim=1).transpose(0, 1)
    return o, lse

def run_torch(
        query,
        kv_buffer,
        kv_page_indices,
        work_indptr,
        work_info_set,
        final_output,
        split_output,
        split_lse
        ):
    """Torch reference for MLA forward decode"""



    pass
    # return ref_output, ref_lse


def _make_inputs(batch, context_len):
    total_kv = batch * context_len
    query = torch.zeros((batch * NUM_HEADS, QK_HEAD_DIM), dtype=dtypes.fp8)
    kv_buffer = torch.zeros((total_kv, QK_HEAD_DIM), dtype=dtypes.fp8)
    kv_page_indices = torch.arange(total_kv, dtype=torch.int32)
    work_indptr = torch.tensor([0, batch], dtype=torch.int32)
    work_info_set = torch.tensor(
        [
            [
                batch_idx,
                batch_idx,
                batch_idx,
                batch_idx + 1,
                batch_idx * context_len,
                (batch_idx + 1) * context_len,
                0,
                0,
            ]
            for batch_idx in range(batch)
        ],
        dtype=torch.int32,
    )
    final_output = torch.empty(
        (batch * NUM_HEADS, V_HEAD_DIM), dtype=torch.bfloat16
    )
    split_output = torch.full(
        (batch * NUM_HEADS, V_HEAD_DIM), float("nan"), dtype=torch.float32
    )
    split_lse = torch.full(
        (batch * NUM_HEADS,), float("nan"), dtype=torch.float32
    )
    return (
        query,
        kv_buffer,
        kv_page_indices,
        work_indptr,
        work_info_set,
        final_output,
        split_output,
        split_lse,
    )


@benchmark()
def test_mla_fwd_decode_m16x8_fp8_fp8(batch=1, context_len=1):
    if not torch.cuda.is_available():
        pytest.skip("ROCm is not available")
    if get_gfx() not in SUPPORTED_GFX:
        pytest.skip(f"MLA decode is unsupported on {get_gfx()}")
    if launch_mla_fwd_decode_m16x8_fp8_fp8 is None:
        pytest.skip("FlyDSL MLA decode is not available")

    (
        query,
        kv_buffer,
        kv_page_indices,
        work_indptr,
        work_info_set,
        final_output,
        split_output,
        split_lse,
    ) = _make_inputs(batch, context_len)
    ref_output, ref_lse = run_torch(batch, context_len, split_output, split_lse)

    props = torch.cuda.get_device_properties(query.device)
    lds_size = getattr(props, "shared_memory_per_multiprocessor", None)
    if lds_size is None:
        lds_size = props.shared_memory_per_block

    def run_flydsl():
        launch_mla_fwd_decode_m16x8_fp8_fp8(
            query,
            kv_buffer,
            kv_page_indices,
            work_indptr,
            work_info_set,
            final_output,
            split_output,
            split_lse,
            1.0 / (QK_HEAD_DIM**0.5),
            1,
            lds_size,
            torch.cuda.current_stream(),
        )
        return split_output, split_lse

    candidates = {"flydsl": run_flydsl}
    flops = 2 * batch * NUM_HEADS * context_len * (QK_HEAD_DIM + V_HEAD_DIM)
    nbytes = sum(
        tensor.numel() * tensor.element_size()
        for tensor in (query, kv_buffer, split_output, split_lse)
    )

    ret = {"gfx": get_gfx()}
    for name, candidate in candidates.items():
        (out, lse), us = run_perftest(candidate)
        output_err = checkAllclose(
            ref_output.to(dtypes.fp32),
            out.to(dtypes.fp32),
            rtol=1e-3,
            atol=1e-3,
            msg=f"{name}: MLA decode output",
        )
        lse_err = checkAllclose(
            ref_lse.to(dtypes.fp32),
            lse.to(dtypes.fp32),
            rtol=1e-3,
            atol=1e-3,
            msg=f"{name}: MLA decode LSE",
        )
        ret[f"{name} us"] = us
        ret[f"{name} TFLOPS"] = flops / us / 1e6
        ret[f"{name} TB/s"] = nbytes / us / 1e6
        ret[f"{name} err"] = max(output_err, lse_err)
    return ret


@pytest.mark.skip(reason="mla_fwd_decode_h16 is not implemented yet")
def test_mla_fwd_decode_h16():
    """Placeholder for the <=16-head MLA decode kernel test."""


def main():
    if get_gfx() not in SUPPORTED_GFX:
        aiter.logger.warning("MLA decode unsupported on %s; skipping", get_gfx())
        return

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="FlyDSL MLA decode test configuration",
    )
    parser.add_argument(
        "-b",
        "--batch",
        type=int,
        nargs="*",
        default=[1],
        help="Batch sizes. Example: -b 1 4",
    )
    parser.add_argument(
        "-s",
        "--context-len",
        type=int,
        nargs="*",
        default=[1],
        help="KV context lengths. Example: -s 1 32",
    )
    args = parser.parse_args()

    rows = [
        test_mla_fwd_decode_m16x8_fp8_fp8(batch, context_len)
        for batch, context_len in itertools.product(args.batch, args.context_len)
    ]
    df = pd.DataFrame(rows)
    aiter.logger.info(
        "MLA decode m16x8 fp8/fp8 summary (markdown):\n%s",
        df.to_markdown(index=False),
    )


if __name__ == "__main__":
    main()
