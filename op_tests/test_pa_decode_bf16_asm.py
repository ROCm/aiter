# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Functional + perf test for pa_decode_bf16_asm (FP8 paged-attention decode, gfx1250).

Ops layer:  aiter.pa_decode_bf16_asm  (wraps SP3 PA_DECODE_D64_1TG_4W_PS)

Kernel properties (see the reference host file sched2/pa_ps.cpp):
  * head_dim=64, page_size=256, gqa=8.
  * FP8 Q **and** FP8 paged KV cache; bf16 output.
  * per-tensor scalar dequant scales for Q/K/V (softmax scale folded into
    key_scale by the wrapper).
  * persistent / split-KV; GPT-OSS style attention sink (no-op here).

Style mirrors op_tests/test_pa_ps.py: a torch host reference is compared against
the kernel via aiter.test_common.checkAllclose (no pytest), driven by argparse
over a config grid.  Supports arbitrary kv_len (multi-page): the persistent
kernel splits KV across workgroups and writes partials into split_o/split_lse,
then pa_reduce_v1 combines them into the final output (same compose as
pa_persistent_fwd).
"""

import argparse
import itertools
import random
from typing import Tuple

import pandas as pd
import torch

import aiter
from aiter import dtypes
from aiter.test_common import benchmark, checkAllclose, perftest

torch.set_default_device("cuda")

PA_HEAD_DIM = 64
PA_PAGE_SIZE = 256
PA_GQA_RATIO = 8

fp8 = torch.float8_e4m3fn


def ceil_div(a, b):
    return (a + b - 1) // b


def make_persistent_metadata(
    batch, kv_head_num, gqa, qo_indptr, kv_indptr, context_lens, page_size,
    max_qlen, uni_qlen, device,
):
    """Generate work + reduce metadata via aiter's GPU helper.

    The PA-decode kernel is the persistent / split-KV variant: every workgroup
    reads work_indptr+work_info to find its (batch, q-head, kv-range) slice, and
    the reduce maps tell pa_reduce_v1 how to merge split partials.  All MUST be
    non-null.  aiter's `WorkInfo` union (csrc/include/ps.h) is byte-identical to
    the sched2 ps::WORK_INFO this kernel was authored against, so the metadata is
    directly consumable.
    """
    (
        (work_meta_data_size, work_meta_data_type),
        (work_indptr_size, work_indptr_type),
        (work_info_set_size, work_info_set_type),
        (reduce_indptr_size, reduce_indptr_type),
        (reduce_final_map_size, reduce_final_map_type),
        (reduce_partial_map_size, reduce_partial_map_type),
    ) = aiter.get_pa_metadata_info_v1(batch, kv_head_num)

    work_metadata_ptrs = torch.empty(work_meta_data_size, dtype=work_meta_data_type, device=device)
    work_indptr = torch.empty(work_indptr_size, dtype=work_indptr_type, device=device)
    work_info = torch.empty(work_info_set_size, dtype=work_info_set_type, device=device)
    reduce_indptr = torch.empty(reduce_indptr_size, dtype=reduce_indptr_type, device=device)
    reduce_final_map = torch.empty(reduce_final_map_size, dtype=reduce_final_map_type, device=device)
    reduce_partial_map = torch.empty(reduce_partial_map_size, dtype=reduce_partial_map_type, device=device)

    aiter.get_pa_metadata_v1(
        qo_indptr,
        kv_indptr,
        context_lens,
        gqa,
        kv_head_num,
        True,
        work_metadata_ptrs,
        work_indptr,
        work_info,
        reduce_indptr,
        reduce_final_map,
        reduce_partial_map,
        kv_granularity=max(page_size, 16),
        block_size=page_size,
        max_seqlen_qo=int(max_qlen),
        uni_seqlen_qo=int(uni_qlen),
        fast_mode=True,
        max_split_per_batch=-1,
    )
    return (
        work_indptr,
        work_info,
        reduce_indptr,
        reduce_final_map,
        reduce_partial_map,
    )


def ref_pa_decode(
    Q, K, V, kv_indices, kv_indptr, context_lens, gqa,
    query_scale, key_scale, value_scale, softmax_scale,
):
    """Torch host reference for the gfx1250 PA-decode kernel (no sink, mtp=0).

    De-interleaves the tiled paged FP8 K/V into token-major [token, head, dim]
    (matching test_pa_ps.py's k-cache reconstruction / asm_V_shuffle), dequants
    with the per-tensor scales, then does softmax attention per (batch, kv_head,
    gqa) over the whole context (multi-page via kv_indptr/kv_indices) and returns
    bf16.
    """
    num_pages, kv_head_num = K.shape[0], K.shape[1]
    head_dim = Q.shape[-1]
    page_size = V.shape[2] * V.shape[4]  # (page_size//16) * 16
    batch, qlen = Q.shape[0], Q.shape[1]
    device = Q.device

    # K[p,h,d//16,tok,d%16] -> K_tm[p,h,tok,d];  V[p,h,tok//16,d,tok%16] -> V_tm[p,h,tok,d]
    K_tm = (
        K.float().permute(0, 1, 3, 2, 4).reshape(num_pages, kv_head_num, page_size, head_dim)
    )
    V_tm = (
        V.float().permute(0, 1, 2, 4, 3).reshape(num_pages, kv_head_num, page_size, head_dim)
    )
    Qf = Q.float()  # [batch, qlen, kv_head, gqa, head_dim]
    out = torch.empty_like(Qf)

    for b in range(batch):
        ctx = int(context_lens[b].item())
        pages = kv_indices[int(kv_indptr[b]) : int(kv_indptr[b + 1])].long()
        tok_page = pages.repeat_interleave(page_size)[:ctx]
        tok_off = torch.arange(ctx, device=device) % page_size
        Kc = K_tm[tok_page, :, tok_off, :] * key_scale     # [ctx, kv_head, head_dim]
        Vc = V_tm[tok_page, :, tok_off, :] * value_scale   # [ctx, kv_head, head_dim]
        for ql in range(qlen):
            q = Qf[b, ql] * query_scale  # [kv_head, gqa, head_dim]
            logits = torch.einsum("hgd,thd->hgt", q, Kc) * softmax_scale
            w = torch.softmax(logits.float(), dim=-1)
            out[b, ql] = torch.einsum("hgt,thd->hgd", w, Vc)
    return out.to(torch.bfloat16)


@perftest(num_rotate_args=1)
def run_pa_stage(
    Q, K, V, kv_indices, context_lens, softmax_scale, kv_indptr,
    gqa, mtp, query_scale, key_scale, value_scale, qo_indptr,
    work_indptr, work_info, split_o, split_lse,
):
    # PA stage only: writes O directly for non-split work items and partials into
    # split_o/split_lse for split (multi-page) ones.  (num_rotate_args=1 so the
    # split buffers after the call belong to the same call whose `out` we keep.)
    return aiter.pa_decode_bf16_asm(
        Q, K, V, kv_indices, context_lens, softmax_scale, kv_indptr,
        gqa=gqa, mtp=mtp,
        query_scale=query_scale, key_scale=key_scale, value_scale=value_scale,
        qo_indptr=qo_indptr, work_indptr=work_indptr, work_info=work_info,
        split_o=split_o, split_lse=split_lse,
    )


def cpu_reduce(out, split_o, split_lse, reduce_indptr, reduce_final_map, reduce_partial_map, gqa):
    """Host (torch) replacement for pa_reduce_v1 (gfx1250 reduce kernel WIP).

    Merges the kernel's split partials into the final O in place, mirroring
    sched2 common_pa_ps.h::paged_attention_reduce: per reduce group,
        final_lse = max_lse + log(sum exp(lse - max_lse))
        final_o   = sum_p partial_o_p * exp(lse_p - final_lse)
    Rows written directly to O by the kernel (partial_o_loc == -1, not in any
    reduce group) are left untouched.  Empty/padded groups (start >= end) skip.
    """
    batch, qlen, kv_head_num = out.shape[0], out.shape[1], out.shape[2]
    head_dim = out.shape[-1]
    q_head_num = kv_head_num * gqa
    total_s = batch * qlen

    out_flat = out.view(total_s, q_head_num, head_dim)
    so = split_o.reshape(split_o.shape[0], q_head_num, head_dim).float()
    sl = split_lse.reshape(split_lse.shape[0], q_head_num).float()

    rip = reduce_indptr.to(torch.int64).tolist()
    rfm = reduce_final_map.to(torch.int64).reshape(-1, 2).tolist()
    rpm = reduce_partial_map.to(torch.int64)

    for g in range(len(rip) - 1):
        s0, s1 = rip[g], rip[g + 1]
        if s1 <= s0:  # padded/empty group
            continue
        qo_start, qo_end = rfm[g][0], rfm[g][1]
        base = rpm[s0:s1]  # [P] partial_o_loc base for each split
        for seq_id in range(qo_start, qo_end):
            locs = base + (seq_id - qo_start)
            lses = sl[locs]                       # [P, q_head_num]
            m = lses.max(dim=0).values            # [q_head_num]
            s = torch.exp(lses - m).sum(dim=0)    # [q_head_num]
            final_lse = m + torch.log(s)          # [q_head_num]
            scale = torch.exp(lses - final_lse)   # [P, q_head_num]
            o = (so[locs] * scale.unsqueeze(-1)).sum(dim=0)  # [q_head_num, head_dim]
            out_flat[seq_id] = o.to(out_flat.dtype)
    return out


@benchmark()
def test_pa_decode(
    batch: int,
    kv_head_num: int,
    ctx_len: int,
    scales: Tuple[float, float, float],
    varlen: bool = False,
) -> dict:
    """Random FP8 paged inputs (arbitrary kv_len) vs the torch host reference."""
    gqa = PA_GQA_RATIO
    head_dim = PA_HEAD_DIM
    page_size = PA_PAGE_SIZE
    mtp = 0
    qlen_with_mtp = mtp + 1
    q_head_num = kv_head_num * gqa
    device = "cuda"

    query_scale, key_scale, value_scale = scales
    softmax_scale = 1.0 / (head_dim**0.5)

    # ---- KV lengths + paged block tables (mirrors test_pa_ps.py) ----
    seq_lens_kv = torch.empty(batch, dtype=torch.int32, device=device)
    if varlen:
        for i in range(batch):
            seq_lens_kv[i] = max(int(random.uniform(1, ctx_len)), 1)
    else:
        seq_lens_kv.fill_(ctx_len)

    max_blocks_per_seq = ceil_div(int(seq_lens_kv.max().item()), page_size)
    max_blocks = max_blocks_per_seq * batch
    block_tables = torch.randperm(max_blocks, device=device).to(torch.int32).reshape(
        batch, max_blocks_per_seq
    )

    actual_blocks = ceil_div(seq_lens_kv, page_size)
    kv_indptr = torch.zeros(batch + 1, dtype=torch.int32, device=device)
    kv_indptr[1:] = torch.cumsum(actual_blocks, dim=0)
    kv_indices = torch.cat(
        [block_tables[i, : int(actual_blocks[i].item())] for i in range(batch)]
    ).to(torch.int32)

    qo_indptr = torch.arange(
        0, (batch + 1) * qlen_with_mtp, qlen_with_mtp, dtype=torch.int32, device=device
    )

    num_phys_pages = max_blocks
    # Keep magnitudes modest so FP8 e4m3 represents them well.
    Q = (0.5 * torch.randn(batch, qlen_with_mtp, kv_head_num, gqa, head_dim, device=device)).to(fp8)
    K = (0.5 * torch.randn(num_phys_pages, kv_head_num, head_dim // 16, page_size, 16, device=device)).to(fp8)
    V = (0.5 * torch.randn(num_phys_pages, kv_head_num, page_size // 16, head_dim, 16, device=device)).to(fp8)

    # ---- persistent metadata + split scratch ----
    max_qlen = qlen_with_mtp
    (
        work_indptr,
        work_info,
        reduce_indptr,
        reduce_final_map,
        reduce_partial_map,
    ) = make_persistent_metadata(
        batch, kv_head_num, gqa, qo_indptr, kv_indptr, seq_lens_kv,
        page_size, max_qlen, qlen_with_mtp, device,
    )
    split_rows = reduce_partial_map.size(0) * max_qlen
    # Init split_lse=-inf / split_o=0 so any split the kernel leaves unwritten
    # (e.g. an empty-kv split the scheduler created for a short sequence)
    # contributes exp(lse-max)=0 and is ignored by the host reduce.
    split_o = torch.zeros((split_rows, 1, q_head_num, head_dim), dtype=dtypes.fp32, device=device)
    split_lse = torch.full(
        (split_rows, 1, q_head_num, 1), float("-inf"), dtype=dtypes.fp32, device=device
    )

    # PA stage on GPU (timed); reduce on host (gfx1250 reduce kernel is WIP).
    out, us = run_pa_stage(
        Q, K, V, kv_indices, seq_lens_kv, softmax_scale, kv_indptr,
        gqa, mtp, query_scale, key_scale, value_scale, qo_indptr,
        work_indptr, work_info, split_o, split_lse,
    )
    torch.cuda.synchronize()
    out = cpu_reduce(
        out, split_o, split_lse,
        reduce_indptr, reduce_final_map, reduce_partial_map, gqa,
    )

    ref = ref_pa_decode(
        Q, K, V, kv_indices, kv_indptr, seq_lens_kv, gqa,
        query_scale, key_scale, value_scale, softmax_scale,
    )

    # FP8 inputs + bf16 output + exp2/log2e + MFMA accumulation -> loose tol.
    err = checkAllclose(
        ref.float(),
        out.float(),
        atol=2e-2,
        rtol=2e-2,
        msg="[torch vs pa_decode_bf16_asm][fp8]: us......",
    )

    # us_pa = PA kernel stage only (reduce runs on host until the gfx1250 reduce
    # kernel lands, so it is excluded from the kernel latency number).
    return {"max_kv": int(seq_lens_kv.max().item()), "us_pa": us, "err": err}


parser = argparse.ArgumentParser(
    formatter_class=argparse.RawTextHelpFormatter,
    description="config input of pa_decode_bf16_asm test",
)
parser.add_argument(
    "-b",
    "--batch_size",
    type=int,
    nargs="*",
    default=[1, 2, 4],
    help="""Batch size.
    e.g. -b 1 2 4""",
)
parser.add_argument(
    "-kvh",
    "--kv_head_num",
    type=int,
    nargs="*",
    default=[1, 2],
    help="""Number of KV heads (q heads = kv_head_num * gqa(8)).
    e.g. -kvh 1 2""",
)
parser.add_argument(
    "-c",
    "--ctx_len",
    type=int,
    nargs="*",
    default=[64, 256, 1024, 4097, 10240],
    help="""Context length (arbitrary; multi-page when > 256).
    e.g. -c 256 4097""",
)
parser.add_argument(
    "--varlen",
    action="store_true",
    help="""Variable kv seqlens per batch (random in [1, ctx_len]). Default: False.
    --varlen # enable""",
)
parser.add_argument(
    "--scaled",
    action="store_true",
    help="""Use non-trivial per-tensor scales (0.5/2.0/1.5) instead of unit.
    --scaled # enable""",
)
args = parser.parse_args()

l_scales = [(0.5, 2.0, 1.5)] if args.scaled else [(1.0, 1.0, 1.0)]

df = []
for batch, kv_head_num, ctx_len, scales in itertools.product(
    args.batch_size, args.kv_head_num, args.ctx_len, l_scales
):
    ret = test_pa_decode(batch, kv_head_num, ctx_len, scales, args.varlen)
    df.append(ret)
df = pd.DataFrame(df)
df_md = df.to_markdown(index=False)
aiter.logger.info("pa_decode_bf16_asm summary (markdown):\n%s", df_md)
df.to_csv("pa_decode_bf16_asm.csv")
