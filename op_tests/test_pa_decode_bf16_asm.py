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
over a config grid.  Supports arbitrary kv_len (multi-page) via split-KV.

The gfx1250 split-KV reduce kernel is WIP, so the PA stage runs on GPU and the
LSE merge runs on host in cpu_reduce (which faithfully matches aiter
csrc/kernels/mla/reduce.cu).  The summary also reports n_partial / unwritten to
flag any partial slots the kernel did not write (a kernel/metadata mismatch).
"""

import argparse
import itertools
import os
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


def make_no_split_metadata(batch, kv_head_num, gqa, qo_indptr, kv_indptr, device):
    """Hand-built NO-SPLIT metadata in the SP3 kernel's (sched2) convention:
    one work item per (batch, kv_head), partial_o_loc=-1 (direct-to-O), covering
    the whole context.  work_info = 8-dword WORK_INFO:
      [batch_idx, partial_o_loc, qo_start, qo_end, kv_start, kv_end, kv_offset,
       q_head_range=pack(q_head_start,q_head_end)]
    kv_start/kv_end = global block indices (kv_indptr-based); qo_* global qo idx.
    """
    qo = qo_indptr.tolist()
    kvp = kv_indptr.tolist()
    works = []
    for b in range(batch):
        for h in range(kv_head_num):
            qhs, qhe = h * gqa, (h + 1) * gqa
            q_head_range = ((qhe & 0xFFFF) << 16) | (qhs & 0xFFFF)
            works.append([b, -1, qo[b], qo[b + 1], kvp[b], kvp[b + 1], 0, q_head_range])
    num_work = len(works)
    work_info = torch.tensor(works, dtype=torch.int32, device=device).reshape(-1)
    num_tg = torch.cuda.get_device_properties(device).multi_processor_count
    work_indptr = torch.tensor(
        [(i * num_work) // num_tg for i in range(num_tg + 1)],
        dtype=torch.int32, device=device,
    )
    return work_indptr, work_info


def make_sched2_metadata(
    batch, kv_head_num, gqa, qo_indptr, kv_indptr, context_lens,
    block_size, qlen_granularity, available_tgs, device, is_causal=False,
):
    """Faithful Python port of sched2 common_ps.h generate_metadata +
    generate_reduce_info (the convention the SP3 PA_DECODE kernel was authored
    against).  aiter.get_pa_metadata_v1 encodes work_info differently (the SP3
    kernel reads zeros from it), so we build it directly.

    work_info = 8-dword WORK_INFO: [batch_idx, partial_o_loc, qo_start, qo_end,
    kv_start, kv_end, kv_offset, q_head_range=pack(qhs,qhe)].  partial_o_loc=-1
    means write straight to O (single-TG tile); otherwise the row in split_o.
    Returns (work_indptr, work_info, reduce_indptr, reduce_final_map,
    reduce_partial_map, split_rows).
    """
    qhead_granularity = gqa
    kvlen_granularity = block_size          # pa_ps.cpp passes PA_PAGE_SIZE
    blocks_per_unit = kvlen_granularity // block_size  # = 1
    SPLIT_KV_OVERHEAD = 0
    qo = qo_indptr.tolist()
    kvp = kv_indptr.tolist()
    ctx = context_lens.tolist()
    num_head_k = kv_head_num

    # Step 1: query tiles (single cluster, one work = one Q-tile x one q-head).
    qtiles = []  # [batch_idx, qo_start, qo_end, num_blocks, effective_kv_len]
    total_units = 0
    for b in range(batch):
        qo_len = qo[b + 1] - qo[b]
        kv_len = ctx[b]
        q_off = 0
        while q_off < qo_len:
            lqs, lqe = q_off, min(q_off + qlen_granularity, qo_len)
            ekv = min(kv_len - qo_len + lqe, kv_len) if is_causal else kv_len
            num_units = ceil_div(ekv, kvlen_granularity)
            qtiles.append([b, lqs + qo[b], lqe + qo[b], num_units * blocks_per_unit, ekv])
            total_units += num_units
            q_off += qlen_granularity

    average = total_units // available_tgs
    reminder = total_units % available_tgs

    # Step 2: distribute split units across TGs (mirrors kn_generate_metadata).
    work_info = []
    work_indptr = [0] * (available_tgs + 1)
    cur_tile = cur_block = partial_tile_idx = 0
    for tg in range(available_tgs):
        for kho in range(num_head_k):
            qhs, qhe = kho * qhead_granularity, (kho + 1) * qhead_granularity
            qhr = ((qhe & 0xFFFF) << 16) | (qhs & 0xFFFF)
            sv_tile, sv_block, sv_pidx = cur_tile, cur_block, partial_tile_idx
            cap = (average + 1) * blocks_per_unit if tg < reminder else average * blocks_per_unit
            while cur_tile < len(qtiles) and cap > 0:
                bt, qs, qe, nblk, ekv = qtiles[cur_tile]
                remaining_blocks = nblk - cur_block
                remaining_kv = ekv - cur_block * block_size
                kv_start = cur_block + kvp[bt]
                if remaining_kv <= cap * block_size + SPLIT_KV_OVERHEAD:
                    consuming = remaining_blocks
                    if cur_block == 0:
                        ploc = -1  # whole tile in one TG -> direct to O
                    else:
                        ploc = qlen_granularity * partial_tile_idx
                        partial_tile_idx += 1
                    kv_end = min(kv_start + consuming, kvp[bt + 1])
                    work_info.append([bt, ploc, qs, qe, kv_start, kv_end, 0, qhr])
                    cur_tile += 1
                    cur_block = 0
                else:
                    consuming = cap
                    ploc = qlen_granularity * partial_tile_idx
                    partial_tile_idx += 1
                    kv_end = min(kv_start + consuming, kvp[bt + 1])
                    kv_off = ctx[bt] - (kv_end - kvp[bt]) * block_size
                    work_info.append([bt, ploc, qs, qe, kv_start, kv_end, kv_off, qhr])
                    cur_block += consuming
                cap -= consuming
            if kho != num_head_k - 1:  # kheads share the same split layout
                cur_tile, cur_block, partial_tile_idx = sv_tile, sv_block, sv_pidx
        work_indptr[tg + 1] = len(work_info)

    # Reduce info: group partials by (qo_start, qo_end), dedup across kheads.
    reduce_map = {}
    for w in work_info:
        if w[1] == -1:
            continue
        reduce_map.setdefault((w[2], w[3]), set()).add(w[1])
    reduce_indptr = [0]
    reduce_final_map = []
    reduce_partial_map = []
    nrw = 0
    for key in sorted(reduce_map.keys()):
        plocs = sorted(reduce_map[key])
        nrw += len(plocs)
        reduce_indptr.append(nrw)
        reduce_final_map.append([key[0], key[1]])
        reduce_partial_map.extend(plocs)

    plocs_all = [w[1] for w in work_info if w[1] != -1]
    split_rows = (max(plocs_all) + qlen_granularity) if plocs_all else 1

    def _t(lst, n=None):
        t = torch.tensor(lst if lst else [0], dtype=torch.int32, device=device)
        return t if lst else t[:0]

    return (
        torch.tensor(work_indptr, dtype=torch.int32, device=device),
        _t([x for w in work_info for x in w]),
        torch.tensor(reduce_indptr, dtype=torch.int32, device=device),
        _t([x for p in reduce_final_map for x in p]),
        _t(reduce_partial_map),
        split_rows,
    )


def cpu_reduce(out, split_o, split_lse, reduce_indptr, reduce_final_map, reduce_partial_map, gqa):
    """Host reduce (matches aiter csrc/kernels/mla/reduce.cu convention, natural log):
        global_lse = max_lse + log(sum exp(lse - max_lse))
        out        = sum_p partial_o_p * exp(lse_p - global_lse)
    partial_lse layout [row, head]; partial_output [row, head, dv]; row = loc+local_seq.
    Only reduced rows are touched; padded groups (start>=end) skipped.
    Returns (out, n_referenced, n_unwritten) for diagnostics.
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

    n_ref = 0
    n_unwritten = 0
    for g in range(len(rip) - 1):
        s0, s1 = rip[g], rip[g + 1]
        if s1 <= s0:
            continue
        qo_start, qo_end = rfm[g][0], rfm[g][1]
        base = rpm[s0:s1]
        for seq_id in range(qo_start, qo_end):
            locs = base + (seq_id - qo_start)
            lses = sl[locs]                       # [P, q_head_num]
            n_ref += lses.numel()
            n_unwritten += int(torch.isinf(lses).sum().item())
            m = lses.max(dim=0).values
            s = torch.exp(lses - m).sum(dim=0)
            global_lse = m + torch.log(s)
            scale = torch.exp(lses - global_lse)
            o = (so[locs] * scale.unsqueeze(-1)).sum(dim=0)
            out_flat[seq_id] = o.to(out_flat.dtype)
    return out, n_ref, n_unwritten


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
    # PA stage: direct-to-O for non-split work items, partials -> split_o/split_lse
    # for split (multi-page) ones.  num_rotate_args=1 so the split buffers after
    # the call belong to the same call whose `out` we keep.
    return aiter.pa_decode_bf16_asm(
        Q, K, V, kv_indices, context_lens, softmax_scale, kv_indptr,
        gqa=gqa, mtp=mtp,
        query_scale=query_scale, key_scale=key_scale, value_scale=value_scale,
        qo_indptr=qo_indptr, work_indptr=work_indptr, work_info=work_info,
        split_o=split_o, split_lse=split_lse,
    )


@benchmark()
def test_pa_decode(
    batch: int,
    kv_head_num: int,
    ctx_len: int,
    scales: Tuple[float, float, float],
    varlen: bool = False,
    no_split: bool = False,
) -> dict:
    """Random FP8 paged inputs (arbitrary kv_len) vs the torch host reference."""
    gqa = PA_GQA_RATIO
    head_dim = PA_HEAD_DIM
    page_size = PA_PAGE_SIZE
    mtp = 0
    qlen_with_mtp = mtp + 1
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

    max_qlen = qlen_with_mtp
    q_head_num = kv_head_num * gqa

    if no_split:
        # Hand-built sched2-convention metadata: every work item writes direct to
        # O, no split/reduce.  Isolates kernel correctness from aiter metadata.
        work_indptr, work_info = make_no_split_metadata(
            batch, kv_head_num, gqa, qo_indptr, kv_indptr, device,
        )
        out, us = run_pa_stage(
            Q, K, V, kv_indices, seq_lens_kv, softmax_scale, kv_indptr,
            gqa, mtp, query_scale, key_scale, value_scale, qo_indptr,
            work_indptr, work_info, None, None,
        )
        torch.cuda.synchronize()
        n_ref = n_unwritten = 0
    else:
        # ---- sched2-convention split-KV metadata + scratch (host reduce; gfx1250 reduce WIP) ----
        num_cu = torch.cuda.get_device_properties(device).multi_processor_count
        (
            work_indptr, work_info,
            reduce_indptr, reduce_final_map, reduce_partial_map, split_rows,
        ) = make_sched2_metadata(
            batch, kv_head_num, gqa, qo_indptr, kv_indptr, seq_lens_kv,
            page_size, qlen_with_mtp, num_cu, device,
        )
        # -inf lse / 0 o so any split the kernel leaves unwritten is inert in reduce.
        split_o = torch.zeros((split_rows, 1, q_head_num, head_dim), dtype=dtypes.fp32, device=device)
        split_lse = torch.full(
            (split_rows, 1, q_head_num, 1), float("-inf"), dtype=dtypes.fp32, device=device
        )

        out, us = run_pa_stage(
            Q, K, V, kv_indices, seq_lens_kv, softmax_scale, kv_indptr,
            gqa, mtp, query_scale, key_scale, value_scale, qo_indptr,
            work_indptr, work_info, split_o, split_lse,
        )
        torch.cuda.synchronize()

        out, n_ref, n_unwritten = cpu_reduce(
            out, split_o, split_lse,
            reduce_indptr, reduce_final_map, reduce_partial_map, gqa,
        )

    ref = ref_pa_decode(
        Q, K, V, kv_indices, kv_indptr, seq_lens_kv, gqa,
        query_scale, key_scale, value_scale, softmax_scale,
    )

    if os.environ.get("AITER_PA_DEBUG", "0") == "1":
        of = out.float()
        rf = ref.float()
        o0 = of[0, 0, 0, 0, :8]
        r0 = rf[0, 0, 0, 0, :8]
        print(f"\n[DEBUG] b={batch} kvh={kv_head_num} ctx={ctx_len} scales={scales} "
              f"n_partial={n_ref} unwritten={n_unwritten}")
        print(f"[DEBUG] out[0,0,0,0,:8] = {[round(x,4) for x in o0.tolist()]}")
        print(f"[DEBUG] ref[0,0,0,0,:8] = {[round(x,4) for x in r0.tolist()]}")
        print(f"[DEBUG] out/ref          = {[round(x,4) for x in (o0/r0).tolist()]}")
        # per (kv_head, gqa) max abs err -> reveals per-head / gqa-broadcast issues
        perhead = (of - rf).abs().amax(dim=(0, 1, 4))  # [kv_head, gqa]
        print(f"[DEBUG] per-head maxerr  =\n{perhead}")
        print(f"[DEBUG] |out| mean={of.abs().mean():.4f}  |ref| mean={rf.abs().mean():.4f}")

    # FP8 inputs + bf16 output + exp2/log2e + MFMA accumulation -> loose tol.
    err = checkAllclose(
        ref.float(),
        out.float(),
        atol=2e-2,
        rtol=2e-2,
        msg="[torch vs pa_decode_bf16_asm][fp8]: us......",
    )

    # Diagnostics: n_ref = partial (row,head) entries the reduce read; unwritten =
    # those still -inf (kernel never wrote them -> kernel/metadata split mismatch).
    return {
        "max_kv": int(seq_lens_kv.max().item()),
        "us_pa": us,
        "n_partial": n_ref,
        "unwritten": n_unwritten,
        "err": err,
    }


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
parser.add_argument(
    "--no-split",
    dest="no_split",
    action="store_true",
    help="""Use hand-built no-split metadata (direct-to-O, no reduce). Isolates
    kernel correctness from aiter.get_pa_metadata_v1. Default: False.""",
)
args = parser.parse_args()

l_scales = [(0.5, 2.0, 1.5)] if args.scaled else [(1.0, 1.0, 1.0)]

df = []
for batch, kv_head_num, ctx_len, scales in itertools.product(
    args.batch_size, args.kv_head_num, args.ctx_len, l_scales
):
    ret = test_pa_decode(batch, kv_head_num, ctx_len, scales, args.varlen, args.no_split)
    df.append(ret)
df = pd.DataFrame(df)
df_md = df.to_markdown(index=False)
aiter.logger.info("pa_decode_bf16_asm summary (markdown):\n%s", df_md)
df.to_csv("pa_decode_bf16_asm.csv")
