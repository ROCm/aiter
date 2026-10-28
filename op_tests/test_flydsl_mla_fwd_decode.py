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
NUM_HEADS_KV = 1
QK_HEAD_DIM = 576
V_HEAD_DIM = 512
KV_LORA_RANK = 512
PAGE_SIZE = 1
DECODE_QLEN = 1

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


def run_torch_split(
    q, kvc_cache, kv_indices, work_indptr, work_info_set, split_output, split_lse, sm_scale
):
    """Pure-PyTorch reference for the per-work-item partial outputs.

    Each work item attends ``q[qo_start:qo_end]`` to the KV pages
    ``kv_indices[kv_start:kv_end]`` and writes its normalized output and LSE
    into ``split_output`` / ``split_lse`` (in place) at ``partial_qo_loc``.
    Returns the partial slots written (other slots are left untouched by the
    kernel).
    """
    q = q.to(torch.float)
    kvc_cache = kvc_cache.to(torch.float)
    slots = []

    num_works = work_indptr[-1].item()
    for _, partial_qo_loc, qo_start, qo_end, kv_start, kv_end, _, _ in (
        work_info_set[:num_works].tolist()
    ):
        kvc = kvc_cache[kv_indices[kv_start:kv_end]].flatten(0, 1)
        o, lse = ref_masked_attention(
            q[qo_start:qo_end], kvc, kvc[:, :, :KV_LORA_RANK], sm_scale, torch.float32
        )
        loc = slice(partial_qo_loc, partial_qo_loc + qo_end - qo_start)
        split_output[loc, 0] = o
        split_lse[loc, 0, :, 0] = lse.transpose(0, 1)
        slots.extend(range(loc.start, loc.stop))

    return torch.tensor(slots, dtype=torch.long)


def _make_inputs(batch, context_len, decode_qlen=DECODE_QLEN, max_split_per_batch=32):
    """Build Q, paged KV, and persistent-decode metadata.

    Sequence tables and ``work_indptr`` / ``work_info_set`` come from
    ``aiter.get_mla_metadata_v1``, matching FlyDSL ``tests/kernels/test_mla_decode.py``.
    """
    nhead = NUM_HEADS
    nhead_kv = NUM_HEADS_KV
    page_size = PAGE_SIZE
    fp8 = dtypes.fp8

    seq_lens_qo = torch.full((batch,), decode_qlen, dtype=torch.int)
    qo_indptr = torch.zeros(batch + 1, dtype=torch.int)
    qo_indptr[1:] = torch.cumsum(seq_lens_qo, dim=0)
    total_q = qo_indptr[-1].item()
    max_seqlen_qo = decode_qlen

    kv_block_nums = torch.full(
        (batch,), (context_len + page_size - 1) // page_size, dtype=torch.int
    )
    kv_last_page_lens = torch.ones(batch, dtype=torch.int)
    if context_len % page_size != 0:
        kv_last_page_lens.fill_(context_len % page_size)
    kv_indptr = torch.zeros(batch + 1, dtype=torch.int)
    kv_indptr[1:] = torch.cumsum(kv_block_nums, dim=0)
    num_page = kv_indptr[-1].item()
    kv_page_indices = torch.randperm(num_page, dtype=torch.int)

    query = torch.randn((total_q, nhead, QK_HEAD_DIM), dtype=torch.bfloat16).to(fp8)
    kv_buffer = torch.randn(
        (num_page, page_size, nhead_kv, QK_HEAD_DIM), dtype=torch.bfloat16
    ).to(fp8)

    cu_num = torch.cuda.get_device_properties(query.device).multi_processor_count
    max_split_per_batch = min((cu_num + batch - 1) // batch, max_split_per_batch)

    (
        (work_meta_data_size, work_meta_data_type),
        (work_indptr_size, work_indptr_type),
        (work_info_set_size, work_info_set_type),
        (reduce_indptr_size, reduce_indptr_type),
        (reduce_final_map_size, reduce_final_map_type),
        (reduce_partial_map_size, reduce_partial_map_type),
    ) = aiter.get_mla_metadata_info_v1(
        batch,
        max_seqlen_qo,
        nhead,
        fp8,
        fp8,
        is_sparse=False,
        fast_mode=True,
        num_kv_splits=max_split_per_batch,
        intra_batch_mode=False,
    )

    work_meta_data = torch.empty(work_meta_data_size, dtype=work_meta_data_type)
    work_indptr = torch.empty(work_indptr_size, dtype=work_indptr_type)
    work_info_set = torch.empty(work_info_set_size, dtype=work_info_set_type)
    reduce_indptr = torch.empty(reduce_indptr_size, dtype=reduce_indptr_type)
    reduce_final_map = torch.empty(reduce_final_map_size, dtype=reduce_final_map_type)
    reduce_partial_map = torch.empty(reduce_partial_map_size, dtype=reduce_partial_map_type)

    aiter.get_mla_metadata_v1(
        qo_indptr,
        kv_indptr,
        kv_last_page_lens,
        nhead // nhead_kv,
        nhead_kv,
        False,
        work_meta_data,
        work_info_set,
        work_indptr,
        reduce_indptr,
        reduce_final_map,
        reduce_partial_map,
        kv_granularity=max(page_size, 16),
        max_seqlen_qo=int(max_seqlen_qo),
        uni_seqlen_qo=decode_qlen,
        fast_mode=True,
        max_split_per_batch=max_split_per_batch,
        intra_batch_mode=False,
        dtype_q=fp8,
        dtype_kv=fp8,
    )

    final_output = torch.empty((total_q, nhead, V_HEAD_DIM), dtype=torch.bfloat16)
    num_partial = reduce_partial_map.size(0) * max_seqlen_qo
    split_output = torch.zeros(
        (num_partial, 1, nhead, V_HEAD_DIM), dtype=torch.float32
    )
    split_lse = torch.zeros((num_partial, 1, nhead, 1), dtype=torch.float32)
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


def _has_unsplit_work(work_indptr, work_info_set):
    """True if any work item has ``partial_qo_loc == -1`` (KV range not split).

    The m16x8 kernel always writes split_output/split_lse at ``partial_qo_loc``
    and leaves the final output to ``mla_reduce_v1``; an unsplit work item
    would make it store at a negative offset.
    """
    num_works = work_indptr[-1].item()
    return bool((work_info_set[:num_works, 1] < 0).any().item())


@benchmark()
def test_mla_fwd_decode_m16x8_fp8_fp8(batch=1, context_len=64):
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
    if _has_unsplit_work(work_indptr, work_info_set):
        pytest.skip(
            f"batch={batch}, context_len={context_len}: metadata produced unsplit "
            "work items, which the m16x8 kernel does not support"
        )

    sm_scale = 1.0 / (QK_HEAD_DIM**0.5)
    ref_split_output = split_output.clone()
    ref_split_lse = split_lse.clone()
    slots = run_torch_split(
        query,
        kv_buffer,
        kv_page_indices,
        work_indptr,
        work_info_set,
        ref_split_output,
        ref_split_lse,
        sm_scale,
    )

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
            sm_scale,
            work_indptr.shape[0] - 1,
            lds_size,
            torch.cuda.current_stream(),
        )
        return split_output, split_lse

    candidates = {"flydsl": run_flydsl}
    flops = 2 * batch * NUM_HEADS * context_len * (QK_HEAD_DIM + V_HEAD_DIM)
    nbytes = sum(
        tensor.numel() * tensor.element_size()
        for tensor in (query, kv_buffer, split_output[slots], split_lse[slots])
    )

    ret = {"gfx": get_gfx()}
    for name, candidate in candidates.items():
        (out, lse), us = run_perftest(candidate)
        output_err = checkAllclose(
            ref_split_output,
            out,
            atol=1e-3,
            rtol=1e-3,
            msg=f"{name}: MLA decode split output",
        )
        lse_err = checkAllclose(
            ref_split_lse,
            lse,
            atol=1e-3,
            rtol=1e-3,
            msg=f"{name}: MLA decode split LSE",
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
        default=[64],
        help="KV context lengths. Example: -s 64 2048",
    )
    args = parser.parse_args()

    rows = []
    for batch, context_len in itertools.product(args.batch, args.context_len):
        try:
            rows.append(test_mla_fwd_decode_m16x8_fp8_fp8(batch, context_len))
        except pytest.skip.Exception as e:
            aiter.logger.warning("Skipped: %s", e)
    if not rows:
        return
    df = pd.DataFrame(rows)
    aiter.logger.info(
        "MLA decode m16x8 fp8/fp8 summary (markdown):\n%s",
        df.to_markdown(index=False),
    )


if __name__ == "__main__":
    main()
