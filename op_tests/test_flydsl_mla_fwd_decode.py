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
from aiter.ops.flydsl.kernels.mla_fwd_decode_h16 import mla_fwd_decode_persistent
from aiter.test_common import benchmark, checkAllclose, run_perftest

torch.set_default_device("cuda")

SUPPORTED_GFX = ["gfx942", "gfx950"]

# Pytorch reference, from FlyDSL tests/kernels/test_mla_decode.py


def ref_attention(query, key, value, scale, mask):
    """Single-sequence masked MLA attention.

    ``mask`` is a bool [s_q, s_k] tensor; query row i attends to key j iff
    mask[i, j]. Rows with no visible key return out = 0 and lse = -inf, so they
    drop out of an LSE merge.
    """

    attn_weights = torch.einsum("qhd,khd->hqk", query.float(), key.float()) * scale
    attn_weights.masked_fill_(~mask, float("-inf"))
    lse = attn_weights.logsumexp(dim=-1)
    m = attn_weights.max(-1).values.nan_to_num(neginf=0.0)
    attn_weights_exp = torch.exp(attn_weights - m.unsqueeze(-1))
    denom = attn_weights_exp.sum(-1)
    out = torch.einsum("hqk,khd->qhd", attn_weights_exp.float(), value.float())
    # denom >= 1 for any row with a visible key; fully masked rows have out == 0.
    out = out / denom.clamp(min=1.0).transpose(0, 1).unsqueeze(-1)

    return out, lse


def run_torch(
    q,
    kv_cache,
    qo_indptr,
    kv_indptr,
    kv_indices,
    kv_last_page_lens,
    sm_scale,
    out,
    lse,
    kv_tail_lens=None,
):
    """Pure-PyTorch paged MLA attention reference.

    ``kv_tail_lens[i]`` is the number of tokens of sequence i that follow the
    KV pages given here (non-zero when only a split chunk is passed); the
    causal mask is placed as if those tokens were present.
    """

    num_page, page_size, nhead_kv, _ = kv_cache.shape
    v_head_dim = out.shape[-1]
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
        v_i = kvc_i[:, :, :v_head_dim]
        tail = 0 if kv_tail_lens is None else kv_tail_lens[i]
        s_q = q_i.shape[0]
        # Query row r sits at chunk-local position (real_kv_seq_len + tail - s_q + r).
        mask = torch.ones(s_q, real_kv_seq_len, dtype=torch.bool).tril(
            diagonal=real_kv_seq_len + tail - s_q
        )
        o_i, lse_i = ref_attention(q_i, k_i, v_i, sm_scale, mask)
        os_list.append(o_i)
        lses.append(lse_i)

    out[:] = torch.concat(os_list)[:]
    lse[:] = torch.concat(lses, dim=1).transpose(0, 1)[:]
    return out, lse


def run_torch_persistent(
    query,
    kv_buffer,
    kv_page_indices,
    work_indptr,
    work_info_set,
    kv_last_page_lens,
    final_output,
    split_output,
    split_lse,
    sm_scale,
):
    """Execute the persistent-decode work list with run_torch, one work at a time.

    Unsplit works (partial_qo_loc < 0) write final_output[qo_start:qo_end];
    split works write split_output / split_lse rows [partial_qo_loc, partial_qo_loc + qo_len).
    Only the work that ends its batch (kv_offset == 0) may have a partial last page.
    Returns the split slot indices that were written.
    """
    page_size = kv_buffer.shape[1]
    work_indptr_l = work_indptr.tolist()
    work_info_l = work_info_set.tolist()
    last_page_lens_l = kv_last_page_lens.tolist()
    num_heads = query.shape[1]

    slots = []
    for cu in range(len(work_indptr_l) - 1):
        for work_idx in range(work_indptr_l[cu], work_indptr_l[cu + 1]):
            (
                batch_idx,
                partial_qo_loc,
                qo_start,
                qo_end,
                kv_start,
                kv_end,
                kv_offset,
                _,
            ) = work_info_l[work_idx]
            qo_len = qo_end - qo_start
            num_pages = kv_end - kv_start
            if qo_len <= 0 or num_pages <= 0:
                continue

            qo_indptr = torch.tensor([0, qo_len], dtype=torch.int32)
            kv_indptr = torch.tensor([0, num_pages], dtype=torch.int32)
            last_page_len = (
                last_page_lens_l[batch_idx] if kv_offset == 0 else page_size
            )
            work_last_page_lens = torch.tensor([last_page_len], dtype=torch.int32)
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
                work_last_page_lens,
                sm_scale,
                out,
                lse,
                kv_tail_lens=[kv_offset],
            )

    return torch.tensor(sorted(slots), dtype=torch.long)


def merge_splits_torch(work_info_set, final_output, split_output, split_lse):
    """LSE-combine split works into final_output[qo_start:qo_end]."""
    groups = {}
    for _, loc, qo_start, qo_end, *_ in work_info_set.tolist():
        if loc >= 0:
            groups.setdefault((qo_start, qo_end), []).append(loc)

    for (qo_start, qo_end), locs in groups.items():
        qo_len = qo_end - qo_start
        o = torch.stack([split_output[loc : loc + qo_len, 0] for loc in locs])
        lse = torch.stack([split_lse[loc : loc + qo_len, 0, :, 0] for loc in locs])
        weights = torch.exp(lse - lse.logsumexp(dim=0)).unsqueeze(-1)
        final_output[qo_start:qo_end] = (weights * o).sum(dim=0)
    return final_output


def get_mla_metadata_torch(
    qo_indptr,
    kv_indptr,
    kv_indices,
    kv_last_page_lens,
    page_size,
    num_cus,
    kv_granularity=16,
    max_split_per_batch=32,
):
    """Minimal persistent-decode planner.

    KV ranges and ``kv_granularity`` are in pages. Splits each batch's KV range
    into equal, granularity-aligned chunks and spreads the resulting works
    evenly over ``num_cus`` CUs.

    Returns:
        kv_page_indices: [num_kv_pages] int32, indexed by work ``kv_start:kv_end``.
        work_indptr: [num_cus + 1] int32, CU ``c`` runs works ``work_indptr[c]:work_indptr[c + 1]``.
        work_info_set: [num_works, 8] int32, aiter ``MlaWorkInfo`` layout:
            batch_idx, partial_qo_loc (-1 if unsplit), qo_start, qo_end,
            kv_start, kv_end, kv_offset, pad.
    """
    qo_indptr_l = qo_indptr.tolist()
    kv_indptr_l = kv_indptr.tolist()
    last_page_lens_l = kv_last_page_lens.tolist()
    batch_size = len(qo_indptr_l) - 1
    splits_per_batch = min(
        max_split_per_batch, (num_cus + batch_size - 1) // batch_size
    )

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
            pages_left = kv_end - split_end
            kv_offset = (
                0
                if pages_left == 0
                else (pages_left - 1) * page_size + last_page_lens_l[b]
            )
            works.append(
                [b, loc, qo_start, qo_end, kv_start, split_end, kv_offset, 0]
            )

    work_info_set = torch.tensor(works, dtype=torch.int32)
    work_indptr = (torch.arange(num_cus + 1, dtype=torch.int32) * len(works)) // num_cus
    kv_page_indices = kv_indices.to(torch.int32).contiguous()
    return kv_page_indices, work_indptr.to(torch.int32), work_info_set


def _make_inputs(
    batch_size,
    ctx_len,
    decode_qlen,
    q_nhead,
    kv_nhead,
    qkv_head_dim,
    rope_dim,
    page_size,
    q_dtype,
    kv_dtype,
    out_dtype,
):
    """Build Q, paged KV, and persistent-decode metadata."""

    # -- Sequence metadata --
    kv_block_nums = torch.full(
        (batch_size,), (ctx_len + page_size - 1) // page_size, dtype=torch.int
    )
    kv_last_page_lens = torch.full((batch_size,), page_size, dtype=torch.int)
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

    # -- KV buffer and Q --
    qk_rope_dim = qkv_head_dim + rope_dim
    kv_buffer = torch.randn(
        (num_page, page_size, kv_nhead, qk_rope_dim), dtype=torch.bfloat16
    )
    kv_buffer = kv_buffer.to(kv_dtype)

    q = torch.randn((total_q, q_nhead, qk_rope_dim), dtype=torch.bfloat16)
    q = q.to(q_dtype)

    sm_scale = 1.0 / (qk_rope_dim**0.5)

    num_cus = torch.cuda.get_device_properties(q.device).multi_processor_count
    kv_page_indices, work_indptr, work_info_set = get_mla_metadata_torch(
        qo_indptr,
        kv_indptr,
        kv_indices,
        kv_last_page_lens,
        page_size,
        num_cus,
        kv_granularity=max(1, 16 // page_size),
    )

    num_partial_slots = max(1, int((work_info_set[:, 1] >= 0).sum()) * decode_qlen)
    # Zero-filled so rows a work list leaves untouched compare equal.
    final_output = torch.zeros((total_q, q_nhead, qkv_head_dim), dtype=out_dtype)
    split_output = torch.zeros(
        (num_partial_slots, 1, q_nhead, qkv_head_dim), dtype=torch.float32
    )
    split_lse = torch.zeros((num_partial_slots, 1, q_nhead, 1), dtype=torch.float32)

    return (
        q,
        kv_buffer,
        qo_indptr,
        kv_indptr,
        kv_indices,
        kv_last_page_lens,
        kv_page_indices,
        work_indptr,
        work_info_set,
        sm_scale,
        final_output,
        split_output,
        split_lse,
    )


@benchmark()
def test_mla_fwd_decode(
        batch_size,
        context_len,
        decode_qlen,
        q_nhead,
        kv_nhead,
        qkv_head_dim,
        rope_dim,
        page_size,
        q_dtype,
        kv_dtype,
        out_dtype,
        num_iters=2,
    ):
    if not torch.cuda.is_available():
        pytest.skip("ROCm is not available")
    if get_gfx() not in SUPPORTED_GFX:
        pytest.skip(f"MLA decode is unsupported on {get_gfx()}")
    if context_len < decode_qlen:
        pytest.skip("context_len must cover all decode_qlen query tokens")

    (
        query,
        kv_buffer,
        qo_indptr,
        kv_indptr,
        kv_indices,
        kv_last_page_lens,
        kv_page_indices,
        work_indptr,
        work_info_set,
        sm_scale,
        final_output,
        split_output,
        split_lse,
    ) = _make_inputs(
        batch_size,
        context_len,
        decode_qlen,
        q_nhead,
        kv_nhead,
        qkv_head_dim,
        rope_dim,
        page_size,
        q_dtype,
        kv_dtype,
        out_dtype,
    )

    ref_final_output = final_output.clone()
    ref_split_output = split_output.clone()
    ref_split_lse = split_lse.clone()
    slots = run_torch_persistent(
        query,
        kv_buffer,
        kv_page_indices,
        work_indptr,
        work_info_set,
        kv_last_page_lens,
        ref_final_output,
        ref_split_output,
        ref_split_lse,
        sm_scale,
    )

    # Sanity-check the work-list reference against whole-sequence attention.
    full_output = torch.empty(final_output.shape, dtype=torch.float32)
    full_lse = torch.empty((query.shape[0], q_nhead), dtype=torch.float32)
    run_torch(
        query,
        kv_buffer,
        qo_indptr,
        kv_indptr,
        kv_indices,
        kv_last_page_lens,
        sm_scale,
        full_output,
        full_lse,
    )
    merged_output = merge_splits_torch(
        work_info_set,
        ref_final_output.float(),
        ref_split_output,
        ref_split_lse,
    )
    checkAllclose(
        full_output,
        merged_output,
        msg="torch: work-list reference vs whole-sequence attention",
    )

    props = torch.cuda.get_device_properties(query.device)
    lds_size = getattr(props, "shared_memory_per_multiprocessor", None)
    if lds_size is None:
        lds_size = props.shared_memory_per_block

    candidates = {
        "torch": lambda: run_torch_persistent(
            query,
            kv_buffer,
            kv_page_indices,
            work_indptr,
            work_info_set,
            kv_last_page_lens,
            final_output,
            split_output,
            split_lse,
            sm_scale,
        ),
        "flydsl": lambda: mla_fwd_decode_persistent(
            query,
            kv_buffer,
            kv_page_indices,
            work_indptr,
            work_info_set,
            kv_last_page_lens,
            final_output,
            split_output,
            split_lse,
            sm_scale,
        ),
    }
    flops = 2 * batch_size * q_nhead * decode_qlen * context_len * (2 * qkv_head_dim + rope_dim)
    nbytes = sum(
        tensor.numel() * tensor.element_size()
        for tensor in (query, kv_buffer, split_output[slots], split_lse[slots])
    )

    ret = {"gfx": get_gfx()}
    for name, candidate in candidates.items():
        for tensor in (final_output, split_output, split_lse):
            tensor.zero_()
        _, us = run_perftest(candidate, num_iters=num_iters)
        final_err = checkAllclose(
            ref_final_output,
            final_output,
            msg=f"{name}: MLA decode final output",
        )
        output_err = checkAllclose(
            ref_split_output,
            split_output,
            atol=1e-3,
            rtol=1e-3,
            msg=f"{name}: MLA decode split output",
        )
        lse_err = checkAllclose(
            ref_split_lse,
            split_lse,
            atol=1e-3,
            rtol=1e-3,
            msg=f"{name}: MLA decode split LSE",
        )
        ret[f"{name} us"] = us
        ret[f"{name} TFLOPS"] = flops / us / 1e6
        ret[f"{name} TB/s"] = nbytes / us / 1e6
        ret[f"{name} err"] = max(final_err, output_err, lse_err)
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
    parser.add_argument(
        "-q",
        "--decode-qlen",
        type=int,
        nargs="*",
        default=[1],
        help="Decode query lengths. Example: -q 1 2",
    )
    parser.add_argument(
        "-n",
        "--num-heads",
        type=int,
        nargs="*",
        default=[12],
        help="Numbers of query heads. Example: -n 12 16",
    )
    parser.add_argument(
        "-nk",
        "--num-heads-kv",
        type=int,
        nargs="*",
        default=[1],
        help="Numbers of KV heads. Example: -nk 1",
    )
    parser.add_argument(
        "-rd",
        "--rope-dim",
        type=int,
        nargs="*",
        default=[64],
        help="RoPE dimensions. Example: -rd 64",
    )
    parser.add_argument(
        "-qkv",
        "--qkv-head-dim",
        type=int,
        nargs="*",
        default=[512],
        help="QKV head dimensions (nope, i.e. LoRA dim). Example: -qkv 512",
    )
    parser.add_argument(
        "-p",
        "--page-size",
        type=int,
        nargs="*",
        default=[1],
        help="KV page sizes. Example: -p 1 16",
    )
    parser.add_argument(
        "-kvd",
        "--kv-dtype",
        type=dtypes.str2Dtype,
        choices=[dtypes.bf16, dtypes.fp8],
        nargs="*",
        default=[dtypes.bf16],
        metavar="{bf16,fp8}",
        help="KV cache dtypes. Example: -kvd bf16 fp8",
    )
    parser.add_argument(
        "-i",
        "--num-iters",
        type=int,
        default=2,
        help="Benchmark iterations per candidate. Example: -i 101",
    )

    args = parser.parse_args()

    rows = []
    for (
        batch,
        context_len,
        decode_qlen,
        num_heads,
        num_heads_kv,
        qkv_head_dim,
        rope_dim,
        page_size,
        kv_dtype,
    ) in itertools.product(
        args.batch,
        args.context_len,
        args.decode_qlen,
        args.num_heads,
        args.num_heads_kv,
        args.qkv_head_dim,
        args.rope_dim,
        args.page_size,
        args.kv_dtype,
    ):
        try:
            rows.append(
                test_mla_fwd_decode(
                    batch,
                    context_len,
                    decode_qlen,
                    num_heads,
                    num_heads_kv,
                    qkv_head_dim,
                    rope_dim,
                    page_size,
                    dtypes.bf16,
                    kv_dtype,
                    dtypes.bf16,
                    num_iters=args.num_iters,
                )
            )
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
