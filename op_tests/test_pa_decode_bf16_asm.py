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
LSE merge runs on host in cpu_reduce (which matches aiter csrc/kernels/mla/reduce.cu).
"""

import argparse
import itertools
import random
import sys
from typing import Optional, Tuple

import pandas as pd
import torch

import aiter
from aiter import dtypes
from aiter.test_common import benchmark, checkAllclose, perftest

current_gfx = aiter.get_gfx()
if current_gfx != "gfx1250":
    print(f"Skipping test_pa_decode_bf16_asm.py: requires gfx1250, got {current_gfx}")
    sys.exit(0)

torch.set_default_device("cuda")

PA_HEAD_DIM = 64
PA_PAGE_SIZE = 256
PA_GQA_RATIO = 8
PA_TILE_Q = 32  # kernel TileQ; mtp must be < PA_TILE_Q / gqa (= 4)

fp8 = dtypes.fp8


def ceil_div(a, b):
    return (a + b - 1) // b


def rms_rel_err(ref, out):
    """RMS error normalized by peak magnitude — the metric the standalone uses
    (fmha_check_result check_mode=1): nrms = sqrt(mean((ref-out)^2)) / max|.|.
    Robust to the per-element noise that big scales (peaked softmax + fp8/exp2)
    amplify, which per-element atol/rtol over-penalizes."""
    a = ref.float()
    b = out.float()
    mag = max(a.abs().max().item(), b.abs().max().item(), 1e-9)
    return ((a - b).pow(2).mean().sqrt() / mag).item()


PA_FP8_MAX = (
    448.0  # OCP E4M3 (non-FNUZ, bias=7) max finite — the kernel's P-quant clamp.
)


def quant_fp8_p_per_row(w):
    """Round softmax weights P to FP8 e4m3 with a per-row (per-query) max scale, to
    match the kernel's second WMMA.

    The SP3 kernel (and csim's golden single_work_decode) quantize the attention
    probabilities to FP8 before the P@V matmul: per row r it forms
    scale_r = max_t|P[r,t]| / 448, casts P/scale_r to e4m3, runs the FP8 matmul,
    then dequants by scale_r.  The fp32 reference skipped this, so it sat below the
    kernel's accuracy and the kernel's FP8-P rounding read as error.

    `w` has the token dim last ([..., t]); the per-row max is taken over t.  Note
    this equals quantizing the *unnormalized* exp() values: the softmax sum cancels
    through scale_r, and the row max element (exp(0)=1 before normalize) maps to 448.
    Single-shot over the whole context, so for >256-token rows it is a close (not
    bit-exact) model of the kernel's per-256-tile rescaled quant."""
    amax = w.abs().amax(dim=-1, keepdim=True).clamp_min(1e-20)
    scale = amax / PA_FP8_MAX
    w_fp8 = (w / scale).clamp(-PA_FP8_MAX, PA_FP8_MAX).to(fp8).float()
    return w_fp8 * scale


def make_sched2_metadata(
    batch,
    kv_head_num,
    gqa,
    qo_indptr,
    kv_indptr,
    context_lens,
    block_size,
    qlen_granularity,
    available_tgs,
    device,
    is_causal=True,
):
    """Python port of sched2 common_ps.h generate_metadata + generate_reduce_info
    (the convention the SP3 PA_DECODE kernel was authored against).

    work_info = 8-dword WORK_INFO: [batch_idx, partial_o_loc, qo_start, qo_end,
    kv_start, kv_end, kv_offset, q_head_range=pack(qhs,qhe)].  A tile handled by a
    single TG uses partial_o_loc=-1 (direct-to-O); split tiles emit partials + a
    reduce group.  Returns (work_indptr, work_info, reduce_indptr,
    reduce_final_map, reduce_partial_map, split_rows).
    """
    qhead_granularity = gqa
    kvlen_granularity = block_size
    blocks_per_unit = kvlen_granularity // block_size
    qo = qo_indptr.tolist()
    kvp = kv_indptr.tolist()
    ctx = context_lens.tolist()
    num_head_k = kv_head_num

    # Step 1: query tiles (one work = one Q-tile x one q-head).
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
            qtiles.append(
                [b, lqs + qo[b], lqe + qo[b], num_units * blocks_per_unit, ekv]
            )
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
            cap = (
                (average + 1) * blocks_per_unit
                if tg < reminder
                else average * blocks_per_unit
            )
            while cur_tile < len(qtiles) and cap > 0:
                bt, qs, qe, nblk, ekv = qtiles[cur_tile]
                remaining_blocks = nblk - cur_block
                remaining_kv = ekv - cur_block * block_size
                kv_start = cur_block + kvp[bt]
                if remaining_kv <= cap * block_size:
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

    def _t(lst):
        return (
            torch.tensor(lst, dtype=torch.int32, device=device)
            if lst
            else torch.zeros(0, dtype=torch.int32, device=device)
        )

    return (
        torch.tensor(work_indptr, dtype=torch.int32, device=device),
        _t([x for w in work_info for x in w]),
        torch.tensor(reduce_indptr, dtype=torch.int32, device=device),
        _t([x for p in reduce_final_map for x in p]),
        _t(reduce_partial_map),
        split_rows,
    )


def cpu_reduce(
    out, split_o, split_lse, reduce_indptr, reduce_final_map, reduce_partial_map, gqa
):
    """Host reduce (matches aiter csrc/kernels/mla/reduce.cu, natural log):
        global_lse = max_lse + log(sum exp(lse - max_lse))
        out        = sum_p partial_o_p * exp(lse_p - global_lse)
    partial_lse layout [row, head]; partial_output [row, head, dv]; row = loc+local_seq.
    Only reduced rows are touched; direct-O rows (written by the kernel) are left alone.
    """
    batch, qlen, kv_head_num = out.shape[0], out.shape[1], out.shape[2]
    head_dim = out.shape[-1]
    q_head_num = kv_head_num * gqa
    out_flat = out.view(batch * qlen, q_head_num, head_dim)
    so = split_o.reshape(split_o.shape[0], q_head_num, head_dim).float()
    sl = split_lse.reshape(split_lse.shape[0], q_head_num).float()

    rip = reduce_indptr.to(torch.int64).tolist()
    rfm = reduce_final_map.to(torch.int64).reshape(-1, 2).tolist()
    rpm = reduce_partial_map.to(torch.int64)

    for g in range(len(rip) - 1):
        s0, s1 = rip[g], rip[g + 1]
        if s1 <= s0:
            continue
        qo_start, qo_end = rfm[g][0], rfm[g][1]
        base = rpm[s0:s1]
        for seq_id in range(qo_start, qo_end):
            locs = base + (seq_id - qo_start)
            lses = sl[locs]
            m = lses.max(dim=0).values
            s = torch.exp(lses - m).sum(dim=0)
            global_lse = m + torch.log(s)
            scale = torch.exp(lses - global_lse)
            o = (so[locs] * scale.unsqueeze(-1)).sum(dim=0)
            out_flat[seq_id] = o.to(out_flat.dtype)
    return out


def ref_pa_decode(
    Q,
    K,
    V,
    kv_indices,
    kv_indptr,
    context_lens,
    gqa,
    query_scale,
    key_scale,
    value_scale,
    softmax_scale,
    sink=None,
):
    """Torch host reference for the gfx1250 PA-decode kernel (no sink, mtp=0).

    De-interleaves the tiled paged FP8 K/V into token-major [token, head, dim]
    (matching test_pa_ps.py's k-cache reconstruction / asm_V_shuffle), dequants
    with the per-tensor scales, then does softmax attention per (batch, kv_head,
    gqa) over the whole context (multi-page via kv_indptr/kv_indices) -> bf16.

    The softmax probabilities P are rounded to FP8 e4m3 (per-row max scale) before
    the P@V product via quant_fp8_p_per_row, mirroring the kernel's second WMMA
    (and csim's single_work_decode golden); skipping it leaves the reference more
    accurate than the kernel, so the kernel's FP8-P rounding shows up as error.

    For mtp>0 the SP3 kernel applies a per-MTP-position causal border (var CAUSAL,
    setup_mask_border): query position i (0..mtp) attends only to the first
    `seq_len - mtp + i` tokens (token p masked when p >= border).  Note this is
    the kernel's convention; the sched2 CPU ref single_work_decode uses one extra
    token (`ctx - mtp + 1 + i`), but the GPU follows the no-`+1` border above.
    For mtp=0 this is the full context (no-op).

    sink (optional): per-Q-head fp32 logits in the kernel's PRE-SCALE raw domain,
    shape [kv_head_num*gqa].  It adds one virtual logit `s_eff*sink_raw` (s_eff =
    query_scale*key_scale*softmax_scale) to each row's softmax denominator; the
    sink has no value, so it only shrinks the real-token weights.
    """
    num_pages, kv_head_num = K.shape[0], K.shape[1]
    head_dim = Q.shape[-1]
    page_size = V.shape[2] * V.shape[4]  # (page_size//16) * 16
    batch, qlen = Q.shape[0], Q.shape[1]
    mtp = qlen - 1
    device = Q.device
    s_eff = query_scale * key_scale * softmax_scale
    sink_hg = (
        (sink.float().view(kv_head_num, gqa) * s_eff) if sink is not None else None
    )

    # K[p,h,d//16,tok,d%16] -> K_tm[p,h,tok,d];  V[p,h,tok//16,d,tok%16] -> V_tm[p,h,tok,d]
    K_tm = (
        K.float()
        .permute(0, 1, 3, 2, 4)
        .reshape(num_pages, kv_head_num, page_size, head_dim)
    )
    V_tm = (
        V.float()
        .permute(0, 1, 2, 4, 3)
        .reshape(num_pages, kv_head_num, page_size, head_dim)
    )
    Qf = Q.float()  # [batch, qlen, kv_head, gqa, head_dim]
    out = torch.empty_like(Qf)

    for b in range(batch):
        ctx = int(context_lens[b].item())
        pages = kv_indices[int(kv_indptr[b]) : int(kv_indptr[b + 1])].long()
        tok_page = pages.repeat_interleave(page_size)[:ctx]
        tok_off = torch.arange(ctx, device=device) % page_size
        # IMPORTANT: keep K/V/Q raw (fp8 dequant, UNSCALED) and apply the scales
        # AFTER the fp32 dot/PV, matching the kernel (it folds q/k scales into
        # scl_log2e applied to Q.K, and value_scale at finalize).  Scaling the
        # operands first changes the fp32 accumulation magnitude and diverges
        # badly for large scales (the dot sums ~scale^2 terms instead of ~1).
        Kc = K_tm[tok_page, :, tok_off, :]  # [ctx, kv_head, head_dim] raw
        Vc = V_tm[tok_page, :, tok_off, :]  # [ctx, kv_head, head_dim] raw
        for ql in range(qlen):
            # SP3 causal border (no +1): MTP position ql attends to the first
            # `ctx - mtp + ql` tokens.  When this is <= 0 the row is FULLY masked
            # (no valid KV) and the kernel outputs 0 (max-init / L=0 guard).  Do
            # NOT clamp to a minimum of 1 here: that makes the reference attend to
            # token 0 and diverge from the GPU for tiny kv_seq_len + mtp>0 (e.g.
            # kv_seq_len=1, mtp=2 -> MTP rows 0,1 are fully masked: GPU=0, but a
            # clamped ref would expect V[0]).  The emu host (pa_setNEG_INF_MQA)
            # masks these rows to 0, which is why the same case passes on emu.
            valid = min(ctx - mtp + ql, ctx)
            if valid <= 0:
                out[b, ql] = 0
                continue
            q = Qf[b, ql]  # [kv_head, gqa, head_dim] raw
            logits = (
                torch.einsum("hgd,thd->hgt", q, Kc[:valid]) * s_eff
            )  # raw dot, then scale
            if sink_hg is not None:
                logits = torch.cat(
                    [logits, sink_hg.unsqueeze(-1)], dim=-1
                )  # +1 sink logit
                w = torch.softmax(logits.float(), dim=-1)[
                    ..., :valid
                ]  # drop sink weight
            else:
                w = torch.softmax(logits.float(), dim=-1)
            # Match the kernel: the P@V WMMA quantizes the real-token probabilities
            # to FP8 per-row (after the sink is dropped — the sink has no value), so
            # quant here, not before, to mirror it.
            w = quant_fp8_p_per_row(w)
            out[b, ql] = torch.einsum("hgt,thd->hgd", w, Vc[:valid]) * value_scale
    return out.to(torch.bfloat16)


@perftest(num_rotate_args=1)
def run_pa_stage(
    Q,
    K,
    V,
    kv_indices,
    context_lens,
    softmax_scale,
    kv_indptr,
    gqa,
    mtp,
    query_scale,
    key_scale,
    value_scale,
    qo_indptr,
    work_indptr,
    work_info,
    split_o,
    split_lse,
    sink,
):
    # PA stage: direct-to-O for non-split work items, partials -> split_o/split_lse
    # for split (multi-page) ones (merged on host by cpu_reduce).  sink=None ->
    # wrapper fills a -inf no-op buffer (kernel always reads the sink slot).
    return aiter.pa_decode_bf16_asm(
        Q,
        K,
        V,
        kv_indices,
        context_lens,
        softmax_scale,
        kv_indptr,
        gqa=gqa,
        mtp=mtp,
        query_scale=query_scale,
        key_scale=key_scale,
        value_scale=value_scale,
        qo_indptr=qo_indptr,
        work_indptr=work_indptr,
        work_info=work_info,
        split_o=split_o,
        split_lse=split_lse,
        sink=sink,
    )


@benchmark()
def test_pa_decode(
    batch: int,
    kv_head_num: int,
    ctx_len: int,
    mtp: int = 0,
    scales: Optional[Tuple[float, float, float]] = None,
    varlen: bool = False,
    use_sink: bool = False,
) -> dict:
    """Random FP8 paged inputs (arbitrary kv_len) vs the torch host reference.

    scales=None -> random per-tensor q/k/v scales; otherwise the given (q,k,v).
    mtp -> multi-token-predict layers (qlen = mtp+1); kernel requires mtp < 4.
    use_sink -> pass a random per-Q-head sink (pre-scale raw logits) to the kernel
    and include the matching sink term in the reference (needs the sink-enabled
    kernel binary; with the current .co the kernel ignores it and this fails).
    """
    gqa = PA_GQA_RATIO
    head_dim = PA_HEAD_DIM
    page_size = PA_PAGE_SIZE
    assert (
        mtp < PA_TILE_Q // gqa
    ), f"kernel requires mtp < {PA_TILE_Q // gqa}, got {mtp}"
    qlen_with_mtp = mtp + 1
    q_head_num = kv_head_num * gqa
    device = "cuda"
    torch.manual_seed(0)

    if scales is None:
        query_scale = round(random.uniform(0.5, 2.0), 4)
        key_scale = round(random.uniform(0.5, 2.0), 4)
        value_scale = round(random.uniform(0.5, 2.0), 4)
    else:
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
    block_tables = (
        torch.randperm(max_blocks, device=device)
        .to(torch.int32)
        .reshape(batch, max_blocks_per_seq)
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
    Q = (
        0.5
        * torch.randn(batch, qlen_with_mtp, kv_head_num, gqa, head_dim, device=device)
    ).to(fp8)
    K = (
        0.5
        * torch.randn(
            num_phys_pages, kv_head_num, head_dim // 16, page_size, 16, device=device
        )
    ).to(fp8)
    V = (
        0.5
        * torch.randn(
            num_phys_pages, kv_head_num, page_size // 16, head_dim, 16, device=device
        )
    ).to(fp8)

    # ---- sched2-convention split-KV metadata + scratch (host reduce; gfx1250 reduce WIP) ----
    num_cu = torch.cuda.get_device_properties(device).multi_processor_count
    (
        work_indptr,
        work_info,
        reduce_indptr,
        reduce_final_map,
        reduce_partial_map,
        split_rows,
    ) = make_sched2_metadata(
        batch,
        kv_head_num,
        gqa,
        qo_indptr,
        kv_indptr,
        seq_lens_kv,
        page_size,
        qlen_with_mtp,
        num_cu,
        device,
    )
    # -inf lse / 0 o so any split the kernel leaves unwritten is inert in reduce.
    split_o = torch.zeros(
        (split_rows, 1, q_head_num, head_dim), dtype=dtypes.fp32, device=device
    )
    split_lse = torch.full(
        (split_rows, 1, q_head_num, 1), float("-inf"), dtype=dtypes.fp32, device=device
    )

    # Sink: per-Q-head logits in the kernel's pre-scale raw domain.  The kernel is
    # sink-enabled (always merges the sink slot), so ALWAYS pass finite values:
    # real (~Q.K range) when use_sink, else a finite large-negative no-op (NOT
    # None/-inf).  The same buffer goes to the kernel and the reference.
    if use_sink:
        sink = (torch.randn(q_head_num, device=device) * 2.0).to(dtypes.fp32)
    else:
        sink = torch.full((q_head_num,), -1.0e30, dtype=dtypes.fp32, device=device)

    out, us = run_pa_stage(
        Q,
        K,
        V,
        kv_indices,
        seq_lens_kv,
        softmax_scale,
        kv_indptr,
        gqa,
        mtp,
        query_scale,
        key_scale,
        value_scale,
        qo_indptr,
        work_indptr,
        work_info,
        split_o,
        split_lse,
        sink,
    )
    torch.cuda.synchronize()
    out = cpu_reduce(
        out,
        split_o,
        split_lse,
        reduce_indptr,
        reduce_final_map,
        reduce_partial_map,
        gqa,
    )

    ref = ref_pa_decode(
        Q,
        K,
        V,
        kv_indices,
        kv_indptr,
        seq_lens_kv,
        gqa,
        query_scale,
        key_scale,
        value_scale,
        softmax_scale,
        sink,
    )

    # Per-element check (detailed report) + RMS/peak (the kernel's actual
    # acceptance metric).  Big scales -> razor-sharp softmax: per-element noise
    # (exp2 vs exp) grows but RMS stays tiny, so judge correctness by nrms.
    err = checkAllclose(
        ref.float(),
        out.float(),
        atol=2e-2,
        rtol=2e-2,
        msg="[torch vs pa_decode_bf16_asm][fp8]: us......",
    )
    nrms = rms_rel_err(ref, out)

    return {
        "max_kv": int(seq_lens_kv.max().item()),
        "mtp": mtp,
        "sink": use_sink,
        "qkv_scale": (query_scale, key_scale, value_scale),
        "us": us,
        "err": err,
        "nrms": nrms,
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
    default=[1, 3, 8, 64],
    help="""Batch size.
    e.g. -b 1 3 8 64""",
)
parser.add_argument(
    "-kvh",
    "--kv_head_num",
    type=int,
    nargs="*",
    default=[1, 8],
    help="""Number of KV heads (q heads = kv_head_num * gqa(8)).
    e.g. -kvh 1 8""",
)
parser.add_argument(
    "-c",
    "--ctx_len",
    type=int,
    nargs="*",
    default=[7, 256, 1024, 4097, 16384],
    help="""Context length (arbitrary; multi-page when > 256).
    e.g. -c 256 4097""",
)
parser.add_argument(
    "-m",
    "--mtp",
    type=int,
    nargs="*",
    default=[0],
    help="""Multi-token-predict layers (qlen = mtp+1). Kernel requires mtp < 4.
    e.g. -m 0 1 2 3""",
)
parser.add_argument(
    "--varlen",
    action="store_true",
    help="""Variable kv seqlens per batch (random in [1, ctx_len]). Default: False.""",
)
parser.add_argument(
    "--scales",
    type=float,
    nargs=3,
    default=None,
    metavar=("Q", "K", "V"),
    help="""Per-tensor q/k/v dequant scales by hand, e.g. --scales 0.5 2.0 1.5.
    Default: random scales per config.""",
)
parser.add_argument(
    "--sink",
    action="store_true",
    help="""Enable GPT-OSS attention sink: random per-Q-head sink logits passed to
    the kernel + matching sink term in the reference. Requires the sink-enabled
    kernel binary; with the current .co the kernel ignores it and this fails.""",
)
args = parser.parse_args()

df = []
for batch, kv_head_num, ctx_len, mtp in itertools.product(
    args.batch_size, args.kv_head_num, args.ctx_len, args.mtp
):
    ret = test_pa_decode(
        batch,
        kv_head_num,
        ctx_len,
        mtp,
        tuple(args.scales) if args.scales is not None else None,
        args.varlen,
        args.sink,
    )
    df.append(ret)
df = pd.DataFrame(df)
df_md = df.to_markdown(index=False)
aiter.logger.info("pa_decode_bf16_asm summary (markdown):\n%s", df_md)
df.to_csv("pa_decode_bf16_asm.csv")
