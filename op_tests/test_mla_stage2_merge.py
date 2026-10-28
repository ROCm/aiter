# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Standalone perf/correctness test for the triton cross-split MLA merge kernel
``_fwd_kernel_stage2_asm`` (aiter/mla.py).

Like ``test_mla_reduce.py`` this runs NO attention stage: it synthesizes the
per-split partial buffers (``logits`` = Mid_O, ``attn_lse`` = Mid_lse) and the
split/kv metadata directly, launches only the stage-2 merge kernel exactly as
``mla_decode_fwd_v4_nm`` does (same args, strides, constexprs), and compares
against a pure-torch FlashAttention log-sum-exp reference.

The kernel merges ``num_kv_splits`` partial (out, lse) pairs per (token, head)
into the final attention output. It is memory-bound in decode (small M, wide
V_HEAD_DIM=512), so TB/s is the metric that matters; TFLOPS is reported too.

Two candidates:
  ``base`` -- the production ``_fwd_kernel_stage2_asm``. Tuning showed it is
    memory-latency bound (num_stages irrelevant, waves_per_eu=4 optimal) and
    that ``num_warps=1`` beats the old ``num_warps=4`` by 1.7-1.9x on gfx950
    decode shapes -- that value now ships in aiter/mla.py.
  ``vec`` -- a loop-free variant that loads the whole [MAX_KV_SPLITS, BLOCK_DV]
    tile and reduces in-register. ~5-16% faster than base for SMALL num_kv_splits
    (2-4), equal/worse for large N (register pressure). Kept here as a validated
    alternative; NOT wired into production (the absolute win is <~1.3us on a
    ~1ms decode and it would need a second kernel + a per-N dispatch).
  ``dchunk`` -- ``_fwd_kernel_stage2_dchunk``, what ``mla_decode_fwd_v4_nm`` runs
    when num_seqs * num_heads <= 2048: ``vec`` with Lv split across a third grid
    axis, so the tile shrinks to [MAX_KV_SPLITS, D_CHUNK] (no register pressure
    at large N) and the CTA count grows by Lv / D_CHUNK.
"""

import argparse
import itertools

import pandas as pd
import torch
import triton
import triton.language as tl

import aiter
from aiter import dtypes
from aiter.jit.utils.chip_info import get_gfx
from aiter.mla import _fwd_kernel_stage2_asm, _launch_stage2_dchunk
from aiter.test_common import assertAllclose, benchmark, run_perftest

torch.set_default_device("cuda")
torch.manual_seed(0)

# No arch gate: _fwd_kernel_stage2_asm is pure triton (tl.load/store/exp/maximum,
# fp32 math, the AMD-generic waves_per_eu hint) with no arch-specific intrinsics,
# so it runs on every gfx target. get_gfx() is still recorded per row so the table
# is self-describing across cards.

# DeepSeek MLA absorbed V head dim; the kernel tiles the whole row at once.
V_HEAD_DIM = 512
# Per-split KV granularity for gqa128 single-token decode (mla.py:1511, else-branch).
MGC = 16


@triton.jit
def _stage2_merge_vec(
    Mid_O,  # [T, N, H, Lv] fp32
    Mid_lse,  # [T, N, H, 1] fp32
    O,
    qo_indptr,
    kv_indptr,
    kv_last_page_lens,
    num_kv_splits_indptr,
    valid_split_count,
    stride_mid_ob: tl.int64,
    stride_mid_oh: tl.int64,
    stride_mid_os: tl.int64,
    stride_obs: tl.int64,
    stride_oh: tl.int64,
    page_size: tl.constexpr,
    KV_INDPTR_IS_PAGE_LEVEL: tl.constexpr,
    USE_VALID_SPLIT_COUNT_REDUCE: tl.constexpr,
    BATCH_NUM: tl.constexpr,
    MAX_KV_SPLITS: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    Lv: tl.constexpr,
    mgc: tl.constexpr,
):
    """Loop-free merge for SMALL num_kv_splits: load the whole [MAX_KV_SPLITS,
    BLOCK_DV] partial tile at once and reduce over the split axis in-register.

    Only viable when MAX_KV_SPLITS is small (tile = MAX_KV_SPLITS*BLOCK_DV fp32
    registers); for N=2..4 this removes the online-softmax loop + its per-split
    setup and issues both split loads concurrently. One (token, head) per CTA.
    """
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)
    offs_d = tl.arange(0, BLOCK_DV)
    mask_d = offs_d < Lv
    splits = tl.arange(0, MAX_KV_SPLITS)

    cur_qo_start = tl.load(qo_indptr + cur_batch)
    cur_qo_end = tl.load(qo_indptr + cur_batch + 1)
    cur_split_start = tl.load(num_kv_splits_indptr + cur_batch)
    cur_split_end = tl.load(num_kv_splits_indptr + cur_batch + 1)
    cur_kv_start = tl.load(kv_indptr + cur_batch)
    cur_kv_end = tl.load(kv_indptr + cur_batch + 1)
    cur_kv_seq_len = cur_kv_end - cur_kv_start
    if KV_INDPTR_IS_PAGE_LEVEL:
        cur_kv_seq_len = (cur_kv_seq_len - 1) * page_size + tl.load(
            kv_last_page_lens + cur_batch
        )
    num_valid = tl.minimum(
        cur_split_end - cur_split_start, tl.cdiv(cur_kv_seq_len, mgc)
    )
    if USE_VALID_SPLIT_COUNT_REDUCE:
        num_valid = tl.minimum(num_valid, tl.load(valid_split_count + cur_batch))
    valid = splits < num_valid

    lse_base = cur_head * stride_mid_oh
    for cur_qo in range(cur_qo_start, cur_qo_end):
        qo_lse = cur_qo * stride_mid_ob + lse_base
        lse = tl.load(
            Mid_lse + qo_lse + splits * stride_mid_os,
            mask=valid,
            other=-float("inf"),
        )  # [MAX_KV_SPLITS]
        m = tl.max(lse, axis=0)
        alpha = tl.where(valid, tl.exp(lse - m), 0.0)  # [MAX_KV_SPLITS]
        z = tl.sum(alpha, axis=0)

        v_off = (
            qo_lse * Lv + splits[:, None] * stride_mid_os * Lv + offs_d[None, :]
        )  # [MAX_KV_SPLITS, BLOCK_DV]
        tv = tl.load(
            Mid_O + v_off,
            mask=valid[:, None] & mask_d[None, :],
            other=0.0,
        )  # [MAX_KV_SPLITS, BLOCK_DV]
        acc = tl.sum(alpha[:, None] * tv, axis=0)  # [BLOCK_DV]

        tl.store(
            O + cur_qo * stride_obs + cur_head * stride_oh + offs_d,
            acc / z,
            mask=mask_d,
        )


def run_torch(logits, attn_lse, num_valid, out_dtype):
    """Reference cross-split flash merge.

    logits    : [T, N, H, Dv] fp32  per-split V partials
    attn_lse  : [T, N, H]     fp32  per-split log-sum-exp
    num_valid : int                 splits actually written (rest are stale)
    Returns   : [T, H, Dv] out_dtype
    """
    lg = logits[:, :num_valid].to(torch.float32)  # [T, n, H, Dv]
    ls = attn_lse[:, :num_valid, :, 0].to(torch.float32)  # [T, n, H]
    m = ls.max(dim=1, keepdim=True).values  # [T, 1, H]
    w = torch.exp(ls - m)  # [T, n, H]
    z = w.sum(dim=1)  # [T, H]
    acc = (w.unsqueeze(-1) * lg).sum(dim=1)  # [T, H, Dv]
    out = acc / z.unsqueeze(-1)
    return out.to(out_dtype)


@benchmark()
def test_stage2_merge(
    num_seqs, num_heads, num_kv_splits, dv, dtype, warps=1, stages=2, wpe=4
):
    T = num_seqs  # decode: one q token per seq (max_seqlen_q == 1)
    N = num_kv_splits
    H = num_heads
    Lv = dv
    BLOCK_DV = 1 << (Lv - 1).bit_length()  # next_power_of_2(Lv)

    # KV length chosen so every one of the N splits is valid (kv_len >= N * mgc):
    # num_valid = min(N, cdiv(kv_len, mgc)) == N.
    kv_len = N * MGC
    num_valid = N

    # --- Synthesize per-split partials (Mid_O / Mid_lse), contiguous ---------
    logits = torch.randn((T, N, H, Lv), dtype=dtypes.fp32)
    attn_lse = torch.randn((T, N, H, 1), dtype=dtypes.fp32) * 2.0
    # Stale (never-written) split slots must not perturb the result; scribble
    # NaN there to prove the num_valid masking in both kernel and reference.
    if num_valid < N:
        logits[:, num_valid:] = float("nan")
        attn_lse[:, num_valid:] = float("nan")

    out_dtype = dtypes.bf16

    # --- Metadata (uniform decode, page_size=1 token-level indptr) -----------
    qo_indptr = torch.arange(0, T + 1, dtype=dtypes.i32)
    kv_indptr = torch.arange(0, T + 1, dtype=dtypes.i32) * kv_len
    kv_last_page_lens = torch.ones(T, dtype=dtypes.i32)
    num_kv_splits_indptr = torch.arange(0, T + 1, dtype=dtypes.i32) * N
    valid_split_count = torch.full((T,), N, dtype=dtypes.i32)
    final_lse_buf = torch.empty((1,), dtype=dtypes.fp32)

    out_base = torch.empty((T, H, Lv), dtype=out_dtype)
    out_vec = torch.empty((T, H, Lv), dtype=out_dtype)
    out_dchunk = torch.empty((T, H, Lv), dtype=out_dtype)
    max_kv_splits = 1 << (N - 1).bit_length()  # next_power_of_2(N)

    ref = run_torch(logits, attn_lse, num_valid, out_dtype)

    def run_base():
        _fwd_kernel_stage2_asm[(T, H)](
            logits,
            attn_lse,
            out_base,
            final_lse_buf,
            qo_indptr,
            kv_indptr,
            kv_last_page_lens,
            num_kv_splits_indptr,
            valid_split_count,
            attn_lse.stride(0),  # stride_mid_ob
            attn_lse.stride(2),  # stride_mid_oh
            attn_lse.stride(1),  # stride_mid_os
            out_base.stride(0),  # stride_obs
            out_base.stride(1),  # stride_oh
            0,  # stride_lse_bs (HAS_FINAL_LSE=False)
            page_size=1,
            KV_INDPTR_IS_PAGE_LEVEL=False,
            MAYBE_FINAL_OUT=True,
            HAS_FINAL_LSE=False,
            USE_VALID_SPLIT_COUNT_REDUCE=int(N > 1),
            BATCH_NUM=T,
            BLOCK_DV=BLOCK_DV,
            Lv=Lv,
            mgc=MGC,
            num_warps=warps,
            num_stages=stages,
            waves_per_eu=wpe,
        )
        return out_base

    def run_vec():
        _stage2_merge_vec[(T, H)](
            logits,
            attn_lse,
            out_vec,
            qo_indptr,
            kv_indptr,
            kv_last_page_lens,
            num_kv_splits_indptr,
            valid_split_count,
            attn_lse.stride(0),
            attn_lse.stride(2),
            attn_lse.stride(1),
            out_vec.stride(0),
            out_vec.stride(1),
            page_size=1,
            KV_INDPTR_IS_PAGE_LEVEL=False,
            USE_VALID_SPLIT_COUNT_REDUCE=int(N > 1),
            BATCH_NUM=T,
            MAX_KV_SPLITS=max_kv_splits,
            BLOCK_DV=BLOCK_DV,
            Lv=Lv,
            mgc=MGC,
            num_warps=warps,
            num_stages=stages,
            waves_per_eu=wpe,
        )
        return out_vec

    def run_dchunk():
        _launch_stage2_dchunk(
            logits,
            attn_lse,
            out_dchunk,
            qo_indptr,
            kv_indptr,
            num_kv_splits_indptr,
            valid_split_count,
            T,
            H,
            N,
            Lv,
            MGC,
        )
        return out_dchunk

    # Merge reads N partial (V + lse) per (token, head), writes one V row.
    read_bytes = T * H * N * (Lv + 1) * 4
    write_bytes = T * H * Lv * out_base.element_size()
    flops = T * H * N * Lv * 2  # weighted-accumulate FMA dominates
    ref_fp32 = ref.to(dtypes.fp32)

    ret = {"gfx": get_gfx(), "warps": warps, "stages": stages, "wpe": wpe}
    for name, fn in (("base", run_base), ("vec", run_vec), ("dchunk", run_dchunk)):
        o, us = run_perftest(fn)
        err = assertAllclose(
            ref_fp32,
            o.to(dtypes.fp32),
            rtol=1e-2,
            atol=1e-2,
            msg=f"{name} T={T} H={H} N={N}",
        )
        ret[f"{name} us"] = us
        ret[f"{name} TFLOPS"] = flops / us / 1e6
        ret[f"{name} TB/s"] = (read_bytes + write_bytes) / us / 1e6
        ret[f"{name} err"] = err
    return ret


def test_stage2_merge_empty_seq(num_heads=16, num_kv_splits=8):
    """num_valid == 0 (kv_len == 0, valid_split_count == 0, or no splits in
    num_kv_splits_indptr) must give O == 0 in both merges, not 0/0 = NaN. The
    unread split slots hold NaN, like a never-written torch.empty buffer."""
    T, H, N, Lv = 4, num_heads, num_kv_splits, V_HEAD_DIM
    logits = torch.full((T, N, H, Lv), float("nan"), dtype=dtypes.fp32)
    attn_lse = torch.full((T, N, H, 1), float("nan"), dtype=dtypes.fp32)
    logits[0] = torch.randn((N, H, Lv), dtype=dtypes.fp32)
    attn_lse[0] = torch.randn((N, H, 1), dtype=dtypes.fp32)
    ref = run_torch(logits[:1], attn_lse[:1], N, dtypes.fp32)[0]

    def indptr(lens):
        return torch.tensor([0] + lens, dtype=dtypes.i32).cumsum(0).to(dtypes.i32)

    # seq0 valid, seq1 kv_len == 0, seq2 valid_split_count == 0, seq3 no splits
    qo_indptr = torch.arange(0, T + 1, dtype=dtypes.i32)
    kv_indptr = indptr([N * MGC, 0, N * MGC, N * MGC])
    num_kv_splits_indptr = indptr([N, N, N, 0])
    valid_split_count = torch.tensor([N, N, 0, N], dtype=dtypes.i32)

    out_dchunk = torch.full((T, H, Lv), 7.0, dtype=dtypes.bf16)
    _launch_stage2_dchunk(
        logits,
        attn_lse,
        out_dchunk,
        qo_indptr,
        kv_indptr,
        num_kv_splits_indptr,
        valid_split_count,
        T,
        H,
        N,
        Lv,
        MGC,
    )
    out_base = torch.full((T, H, Lv), 7.0, dtype=dtypes.bf16)
    _fwd_kernel_stage2_asm[(T, H)](
        logits,
        attn_lse,
        out_base,
        torch.empty((1,), dtype=dtypes.fp32),
        qo_indptr,
        kv_indptr,
        None,
        num_kv_splits_indptr,
        valid_split_count,
        attn_lse.stride(0),
        attn_lse.stride(2),
        attn_lse.stride(1),
        out_base.stride(0),
        out_base.stride(1),
        0,
        page_size=1,
        KV_INDPTR_IS_PAGE_LEVEL=False,
        MAYBE_FINAL_OUT=False,
        HAS_FINAL_LSE=False,
        USE_VALID_SPLIT_COUNT_REDUCE=1,
        BATCH_NUM=T,
        BLOCK_DV=triton.next_power_of_2(Lv),
        Lv=Lv,
        mgc=MGC,
        num_warps=1,
        num_stages=2,
        waves_per_eu=4,
    )
    for name, o in (("dchunk", out_dchunk), ("base", out_base)):
        assert torch.isfinite(o).all(), f"{name}: non-finite O for num_valid == 0"
        assert bool((o[1:] == 0).all()), f"{name}: O != 0 for num_valid == 0"
        assertAllclose(ref, o[0].float(), msg=f"{name} empty-seq batch seq0 ")


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="config input of test",
    )
    parser.add_argument(
        "-b",
        "--batch",
        type=int,
        nargs="*",
        default=[1, 16, 64, 128, 256],
        help="num_seqs (decode tokens); drives reduce-CTA fan-out",
    )
    parser.add_argument(
        "--heads",
        type=int,
        nargs="*",
        default=[16, 128],
        help="num_heads (gqa-flattened); 128 = gqa128 tp1, 16 = tp8",
    )
    parser.add_argument(
        "--splits",
        type=int,
        nargs="*",
        default=[2, 4, 8, 16, 32],
        help="num_kv_splits merged per (token, head)",
    )
    parser.add_argument(
        "-d",
        "--dtype",
        type=dtypes.str2Dtype,
        nargs="*",
        default=[dtypes.bf16],
        help="partial/output dtype axis (output is bf16)",
    )
    parser.add_argument(
        "--warps",
        type=int,
        nargs="*",
        default=[1],  # production value (see module docstring); 4 for A/B tuning
        help="num_warps launch tuning axis for the merge kernel",
    )
    parser.add_argument(
        "--stages",
        type=int,
        nargs="*",
        default=[2],
        help="num_stages (software-pipeline depth of the split loop)",
    )
    parser.add_argument(
        "--wpe",
        type=int,
        nargs="*",
        default=[4],
        help="waves_per_eu occupancy hint",
    )
    args = parser.parse_args()

    for heads in args.heads:
        test_stage2_merge_empty_seq(heads)

    for dtype in args.dtype:
        df = []
        for wpe, stages, warps, heads, splits, batch in itertools.product(
            args.wpe,
            args.stages,
            args.warps,
            args.heads,
            args.splits,
            args.batch,
        ):
            df.append(
                test_stage2_merge(
                    batch, heads, splits, V_HEAD_DIM, dtype, warps, stages, wpe
                )
            )
        df = pd.DataFrame(df)
        aiter.logger.info(
            "_fwd_kernel_stage2_asm merge summary (markdown):\n%s",
            df.to_markdown(index=False),
        )


if __name__ == "__main__":
    main()
