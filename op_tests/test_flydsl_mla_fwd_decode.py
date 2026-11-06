# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Correctness smoke test and benchmark for FlyDSL MLA decode."""

import argparse
import itertools

import pandas as pd
import pytest
import torch

import aiter
from aiter import dtypes
from aiter.jit.utils.chip_info import get_gfx
from aiter.test_common import benchmark, checkAllclose, run_perftest

torch.set_default_device("cuda")

SUPPORTED_GFX = ["gfx942", "gfx950"]
NUM_HEADS = 12
NUM_HEADS_KV = 1
QK_HEAD_DIM = 576
V_HEAD_DIM = 512
PAGE_SIZE = 1
DECODE_QLEN = 1

# Pytorch reference, from FlyDSL tests/kernels/test_mla_decode.py

def ref_attention(query, key, value, scale):
    """Single-sequence MLA attention (no causal mask needed for decode_qlen=1)."""
  
    attn_weights = torch.einsum("qhd,khd->hqk", query.float(), key.float()) * scale
    lse = attn_weights.logsumexp(dim=-1)
    m = attn_weights.max(-1).values
    attn_weights_exp = torch.exp(attn_weights - m.unsqueeze(-1))
    denom = attn_weights_exp.sum(-1)
    out = torch.einsum("hqk,khd->qhd", attn_weights_exp.float(), value.float())
    out = out / denom.transpose(0, 1).unsqueeze(-1)

    return out.to(torch.bfloat16), lse


def run_torch(q, kv_cache, qo_indptr, kv_indptr, kv_indices, kv_last_page_lens, sm_scale, out, lse):
    """Pure-PyTorch paged MLA attention reference."""

    num_page, page_size, nhead_kv, _ = kv_cache.shape
    qs = torch.tensor_split(q, qo_indptr.tolist()[1:])
    kvc = torch.index_select(kv_cache, 0, kv_indices)
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
        v_i = kvc_i[:, :, :V_HEAD_DIM]
        o_i, lse_i = ref_attention(q_i, k_i, v_i, sm_scale)
        os_list.append(o_i)
        lses.append(lse_i)

    out[:] = torch.concat(os_list)[:]
    lse[:] = torch.concat(lses, dim=1).transpose(0, 1)[:]
    return out, lse

def run_torch_persistent(query,
        kv_buffer,
        kv_page_indices,
        work_indptr,
        work_info_set,
        final_output,
        split_output,
        split_lse,
        sm_scale):
    """Execute the persistent-decode work list with run_torch, one work at a time.

    Unsplit works (partial_qo_loc < 0) write final_output[qo_start:qo_end];
    split works write split_output / split_lse rows [partial_qo_loc, partial_qo_loc + qo_len).
    Returns the split slot indices that were written.
    """
    page_size = kv_buffer.shape[1]
    work_indptr_l = work_indptr.tolist()
    work_info_l = work_info_set.tolist()
    num_heads = query.shape[1]

    slots = []
    for cu in range(len(work_indptr_l) - 1):
        for work_idx in range(work_indptr_l[cu], work_indptr_l[cu + 1]):
            _, partial_qo_loc, qo_start, qo_end, kv_start, kv_end, _, _ = work_info_l[work_idx]
            qo_len = qo_end - qo_start
            num_pages = kv_end - kv_start
            if qo_len <= 0 or num_pages <= 0:
                continue

            qo_indptr = torch.tensor([0, qo_len], dtype=torch.int32)
            kv_indptr = torch.tensor([0, num_pages], dtype=torch.int32)
            kv_last_page_lens = torch.tensor([page_size], dtype=torch.int32)

            if partial_qo_loc < 0:
                out = final_output[qo_start:qo_end]
                lse = torch.empty((qo_len, num_heads), dtype=torch.float32)
            else:
                out = split_output[partial_qo_loc : partial_qo_loc + qo_len, 0]
                lse = split_lse[partial_qo_loc : partial_qo_loc + qo_len, 0, :, 0]
                slots.extend(range(partial_qo_loc, partial_qo_loc + qo_len))

            run_torch(
                query[qo_start:qo_end],
                kv_buffer,
                qo_indptr,
                kv_indptr,
                kv_page_indices[kv_start:kv_end],
                kv_last_page_lens,
                sm_scale,
                out,
                lse,
            )

    return torch.tensor(sorted(slots), dtype=torch.long)

def get_mla_metadata_torch(qo_indptr, kv_indptr, kv_indices, num_cus, kv_granularity=16, max_split_per_batch=32):
    """Minimal persistent-decode planner (page_size=1).

    Splits each batch's KV range into equal, granularity-aligned chunks and
    spreads the resulting works evenly over ``num_cus`` CUs.

    Returns:
        kv_page_indices: [num_kv_pages] int32, indexed by work ``kv_start:kv_end``.
        work_indptr: [num_cus + 1] int32, CU ``c`` runs works ``work_indptr[c]:work_indptr[c + 1]``.
        work_info_set: [num_works, 8] int32, aiter ``MlaWorkInfo`` layout:
            batch_idx, partial_qo_loc (-1 if unsplit), qo_start, qo_end,
            kv_start, kv_end, kv_offset, pad.
    """
    qo_indptr_l = qo_indptr.tolist()
    kv_indptr_l = kv_indptr.tolist()
    batch_size = len(qo_indptr_l) - 1
    splits_per_batch = min(max_split_per_batch, (num_cus + batch_size - 1) // batch_size)

    works = []
    partial_qo_loc = 0
    for b in range(batch_size):
        qo_start, qo_end = qo_indptr_l[b], qo_indptr_l[b + 1]
        kv_begin, kv_end = kv_indptr_l[b], kv_indptr_l[b + 1]
        kv_blocks = (kv_end - kv_begin + kv_granularity - 1) // kv_granularity
        num_splits = max(1, min(splits_per_batch, kv_blocks))
        chunk = max(1, (kv_blocks + num_splits - 1) // num_splits) * kv_granularity

        for kv_start in range(kv_begin, max(kv_end, kv_begin + 1), chunk):
            split_end = min(kv_start + chunk, kv_end)
            if num_splits == 1:
                loc = -1
            else:
                loc = partial_qo_loc
                partial_qo_loc += qo_end - qo_start
            works.append([b, loc, qo_start, qo_end, kv_start, split_end, kv_end - split_end, 0])

    work_info_set = torch.tensor(works, dtype=torch.int32)
    work_indptr = (torch.arange(num_cus + 1, dtype=torch.int32) * len(works)) // num_cus
    kv_page_indices = kv_indices.to(torch.int32).contiguous()
    return kv_page_indices, work_indptr.to(torch.int32), work_info_set



def _make_inputs(batch_size, ctx_len, decode_qlen, q_nhead, q_head_dim, kv_nhead, kv_head_dim, page_size, q_dtype, kv_dtype, out_dtype):
    """Build Q, paged KV, and persistent-decode metadata.
    """

    kv_max_sz = 65536 * 32
    num_page = (kv_max_sz + page_size - 1) // page_size

    # -- Sequence metadata --
    # seq_lens_kv = torch.full((batch_size,), ctx_len, dtype=torch.int)
    kv_block_nums = torch.full((batch_size,), (ctx_len + page_size - 1) // page_size, dtype=torch.int)
    kv_last_page_lens = torch.ones(batch_size, dtype=torch.int)
    if ctx_len % page_size != 0:
        kv_last_page_lens.fill_(ctx_len % page_size)

    kv_indptr = torch.zeros(batch_size + 1, dtype=torch.int)
    kv_indptr[1:] = torch.cumsum(kv_block_nums, dim=0)
    num_page = kv_indptr[-1].item()
    kv_indices = torch.randperm(num_page, dtype=torch.int)

    seq_lens_qo = torch.full((batch_size,), decode_qlen, dtype=torch.int)
    qo_indptr = torch.zeros(batch_size + 1, dtype=torch.int)
    qo_indptr[1:] = torch.cumsum(seq_lens_qo, dim=0)
    total_q = qo_indptr[-1].item()
    max_seqlen_qo = decode_qlen

    # -- KV buffer and Q --
    kv_buffer = torch.randn((num_page, page_size, kv_nhead, QK_HEAD_DIM), dtype=torch.bfloat16)
    kv_buffer = kv_buffer.to(kv_dtype)

    q = torch.randn((total_q, q_nhead, QK_HEAD_DIM), dtype=torch.bfloat16)
    q = q.to(q_dtype)

    sm_scale = 1.0 / (QK_HEAD_DIM**0.5)

    num_cus = torch.cuda.get_device_properties(q.device).multi_processor_count
    kv_page_indices, work_indptr, work_info_set = get_mla_metadata_torch(
        qo_indptr, kv_indptr, kv_indices, num_cus, kv_granularity=max(page_size, 16)
    )

    num_partial_slots = max(1, int((work_info_set[:, 1] >= 0).sum()) * decode_qlen)
    final_output = torch.empty((total_q, q_nhead, V_HEAD_DIM), dtype=out_dtype)
    split_output = torch.empty((num_partial_slots, 1, q_nhead, V_HEAD_DIM), dtype=torch.float32)
    split_lse = torch.empty((num_partial_slots, 1, q_nhead, 1), dtype=torch.float32)

    return (
        q,
        kv_buffer,
        kv_page_indices,
        work_indptr,
        work_info_set,
        final_output,
        split_output,
        split_lse,
    )


@benchmark()
def test_mla_fwd_decode(batch=1, context_len=64):
    if not torch.cuda.is_available():
        pytest.skip("ROCm is not available")
    if get_gfx() not in SUPPORTED_GFX:
        pytest.skip(f"MLA decode is unsupported on {get_gfx()}")

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

    sm_scale = 1.0 / (QK_HEAD_DIM**0.5)
    ref_final_output = final_output.clone()
    ref_split_output = split_output.clone()
    ref_split_lse = split_lse.clone()
    slots = run_torch_persistent(
        query,
        kv_buffer,
        kv_page_indices,
        work_indptr,
        work_info_set,
        ref_final_output,
        ref_split_output,
        ref_split_lse,
        sm_scale,
    )

    props = torch.cuda.get_device_properties(query.device)
    lds_size = getattr(props, "shared_memory_per_multiprocessor", None)
    if lds_size is None:
        lds_size = props.shared_memory_per_block


    candidates = {"torch": run_torch}
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
            rows.append(test_mla_fwd_decode(batch, context_len))
        except pytest.skip.Exception as e:
            aiter.logger.warning("Skipped: %s", e)
    if not rows:
        return
    df = pd.DataFrame(rows)
    aiter.logger.info(
        "MLA decode summary (markdown):\n%s",
        df.to_markdown(index=False),
    )


if __name__ == "__main__":
    main()
