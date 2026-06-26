#!/usr/bin/env python
# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""aiter op-test for ``fused_qk_norm_rope_group_quant`` (DeepSeek-V4, no paged cache).

Validates the fused Q/K RMSNorm + GPT-J RoPE + group-quant kernel against a
pure-torch reference, reports per-config kernel us / bandwidth utilization, and
compares throughput against ``flydsl_qk_norm_rope_quant`` (the MLIR-JIT producer
for the same V4 shape).

V4 CSA training-QAT layout: NoPE fp8 (1x64 e8m0 group scale), PE bf16 (NOT
quantized). Token-contiguous outputs, no slot_mapping / paged cache. Q mirrors
K when fp8; bf16 Q stays a plain [.,H,512] rotated tensor (ATOM sparse_attn default).

K outputs (v4 nm asm reader layout), per (token, kv_head):
    k_nope_scale_buff [T, NK, 512] fp8:
        [0   : 448)  nope fp8
        [448 : 462)  e8m0 scale, 14 B = 7 groups each written twice (s0,s0,..,s6,s6)
        [462 : 512)  pad (zero)
    k_rope_buff       [T, NK, 64]  bf16:  rotated K-PE (not quantized)

Q outputs (fp8): same split -- q_nope_scale_buff [T, H, 512], plus a separate
q_rope_buff [T, H, 64] bf16. bf16 Q: q_nope_scale_buff is [T, H, 512] bf16, no q_rope_buff.

Usage:
    python op_tests/test_fused_qk_norm_rope_group_quant.py
    python op_tests/test_fused_qk_norm_rope_group_quant.py -T 64 256 1024 --H 16 128
    python op_tests/test_fused_qk_norm_rope_group_quant.py --no-flydsl   # skip the perf compare
"""

import argparse
import itertools
import random

import pandas as pd
import torch

import aiter
from aiter import dtypes
from aiter.ops.flydsl import flydsl_qk_norm_rope_quant
from aiter.test_common import benchmark, checkAllclose, run_perftest
from aiter.utility.fp4_utils import f32_to_mx_e8m0_scale
from aiter.utility.mx_types import MxDtypeInt, MxScaleRoundModeInt

torch.set_default_device("cuda")

_FP8 = dtypes.fp8
_FP8_MAX = float(torch.finfo(_FP8).max)
_DEV = "cuda"
PE_BYTE_OFFSET = 464
# MI355X HBM3e peak. Used only for the "%peak" perf column.
_PEAK_BW_GBPS = 8000.0


# ============================================================================
# Reference (pure torch)
# ============================================================================


def _cos_sin(max_pos, rope_dim, dtype):
    """Build a RoPE cos/sin table: [max_pos, rope_dim/2]."""
    inv_freq = 1.0 / (
        10000 ** (torch.arange(0, rope_dim, 2, device=_DEV).float() / rope_dim)
    )
    freqs = torch.einsum(
        "i,j->ij", torch.arange(max_pos, device=_DEV).float(), inv_freq
    )
    return freqs.cos().to(dtype).contiguous(), freqs.sin().to(dtype).contiguous()


def _apply_gptj_rope(pe, cos, sin, pos, *, is_neox):
    """Rotate the PE part [.., rope_dim]. is_neox: half-split pairing (x=[:rd/2], y=[rd/2:]);
    else GPT-J adjacent pairing (x=even, y=odd)."""
    T, n_heads = pe.shape[0], pe.shape[1]
    rope_dim = pe.shape[-1]
    c = cos[pos].float().view(T, 1, rope_dim // 2)
    s = sin[pos].float().view(T, 1, rope_dim // 2)
    if is_neox:
        x, y = pe[..., : rope_dim // 2], pe[..., rope_dim // 2 :]
        return torch.cat([x * c - y * s, x * s + y * c], -1)
    x, y = pe[..., 0::2], pe[..., 1::2]
    return torch.stack([x * c - y * s, x * s + y * c], -1).reshape(T, n_heads, rope_dim)


def _norm_rope_nope_fp8(x, weight, cos, sin, pos, eps, *, is_neox, group_size):
    """Pure-torch reference for one stream (Q or K):
        RMSNorm(+weight) -> split nope/pe -> RoPE(pe) -> nope 1xG e8m0 fp8 quant, pe stays bf16.
    Returns (nope_fp8, scale_e8m0[.,n_heads,n_groups] uint8, pe_bf16, full_rotated_bf16).
    """
    T, n_heads, head_dim = x.shape
    rope_dim = cos.shape[-1] * 2
    nope_dim = head_dim - rope_dim
    n_groups = nope_dim // group_size

    normed = x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + eps)
    if weight is not None:
        normed = normed * weight.float()
    nope, pe = normed[..., :nope_dim], normed[..., nope_dim:]
    pe_rotated = _apply_gptj_rope(pe, cos, sin, pos, is_neox=is_neox)

    # nope: per-group amax -> e8m0 scale (MX RoundUp, FP8 E4M3) -> fp8. Uses the shared
    # reference helper (== the kernel's fp_f32_to_e8m0_scale<RoundUp, FP8_E4M3>).
    amax = (
        nope.reshape(T, n_heads, n_groups, group_size).abs().amax(-1).clamp_min(1e-12)
    )
    scale_e8m0 = f32_to_mx_e8m0_scale(
        amax, mode=MxScaleRoundModeInt.RoundUp, dtype=MxDtypeInt.FP8_E4M3
    ).view(
        torch.uint8
    )  # reinterpret the e8m0 byte (== biased exponent), not numeric cast
    inv_scale = 1.0 / (scale_e8m0.to(torch.int32) << 23).view(torch.float32)
    nope_fp8 = (
        nope
        * inv_scale.unsqueeze(-1)
        .expand(T, n_heads, n_groups, group_size)
        .reshape(T, n_heads, nope_dim)
    ).to(_FP8)

    full_rotated_bf16 = torch.cat([nope, pe_rotated], -1).to(torch.bfloat16)
    return nope_fp8, scale_e8m0, pe_rotated.to(torch.bfloat16), full_rotated_bf16


def _ref(q, kv, kw, cos, sin, pos, eps, *, is_neox, q_group_size, k_group_size):
    """V4 Q/K reference. Q is weightless (V4-Pro); K uses the RMSNorm weight kw.
    fp8 Q mirrors K (nope fp8 + pe bf16); the full bf16-rotated Q is the bf16-Q reference.

    NOTE: only Q honours ``quant_group_size``; the kernel's K NoPE quant is always
    1x64 (the v4 asm reader's 14-byte scale format is hardcoded), so the K reference
    uses ``k_group_size`` (= 64) independently of the Q group size.
    """
    q_nope_fp8, q_scale, q_pe_bf16, q_full_bf16 = _norm_rope_nope_fp8(
        q, None, cos, sin, pos, eps, is_neox=is_neox, group_size=q_group_size
    )
    k_nope_fp8, k_scale, k_pe_bf16, _ = _norm_rope_nope_fp8(
        kv, kw, cos, sin, pos, eps, is_neox=is_neox, group_size=k_group_size
    )
    return q_nope_fp8, q_scale, q_pe_bf16, k_nope_fp8, k_scale, k_pe_bf16, q_full_bf16


# ============================================================================
# Main test (per-config): accuracy + perf + flydsl comparison
# ============================================================================


@benchmark()
def test_fused_qk_norm_rope_group_quant(
    T, H, D, RD, *, is_neox, q_fp8, G, GK=64, compare_flydsl=True
):
    NK = 1  # DeepSeek V4 is MQA: exactly one KV head (kernel hardcodes this)
    torch.manual_seed(0)
    random.seed(0)
    nope = D - RD
    eps = 1e-6
    # Q honours the quant_group_size (G); K NoPE is always 1x64 in the kernel, so use a
    # separate GK (=64) for the K reference / scale-readback layout.
    n_groups_q = nope // G  # Q e8m0 scale groups (7 for D=512, G=64; 14 for G=32)
    n_groups_k = nope // GK  # K e8m0 scale groups (always 7 for D=512)
    entry = D  # k_nope_scale_buff is 512 B (head_dim): nope(448) + 2*n_groups dup e8m0 + pad

    cos, sin = _cos_sin(max(T, 64) + 4, RD, torch.bfloat16)
    pos = torch.randint(0, cos.shape[0] - 1, (T,), dtype=torch.int64, device=_DEV)
    q = (torch.randn(T, H, D, device=_DEV) * 0.1).bfloat16()
    kv = (torch.randn(T, NK, D, device=_DEV) * 0.1).bfloat16()
    kw = (torch.randn(D, device=_DEV).abs() + 0.5).bfloat16()

    # Reference
    ref_q_nope, ref_q_scale, ref_q_pe, ref_k_nope, ref_k_scale, ref_k_pe, ref_q_bf16 = (
        _ref(
            q,
            kv,
            kw,
            cos,
            sin,
            pos,
            eps,
            is_neox=is_neox,
            q_group_size=G,
            k_group_size=GK,
        )
    )

    # Kernel + perf (pre-allocate outputs so we time the kernel, not torch.empty)
    # fp8 Q: q_nope_scale_buff (512 B, zeroed pad) + separate q_rope_buff bf16.
    q_nope_scale_buff = (
        torch.zeros(T, H, entry, dtype=_FP8, device=_DEV)
        if q_fp8
        else torch.empty(T, H, D, dtype=torch.bfloat16, device=_DEV)
    )
    q_rope_buff = (
        torch.empty(T, H, RD, dtype=torch.bfloat16, device=_DEV) if q_fp8 else None
    )
    k_nope_scale_buff = torch.zeros(T, NK, entry, dtype=_FP8, device=_DEV)
    k_rope_buff = torch.empty(T, NK, RD, dtype=torch.bfloat16, device=_DEV)
    (q_nope_scale_buff, q_rope_buff, k_nope_scale_buff, k_rope_buff), us = run_perftest(
        aiter.fused_qk_norm_rope_group_quant,
        q,
        kv,
        kw,
        pos,
        cos,
        sin,
        eps,
        is_neox=is_neox,
        q_out_dtype=(_FP8 if q_fp8 else torch.bfloat16),
        q_nope_scale_buff=q_nope_scale_buff,
        q_rope_buff=q_rope_buff,
        k_nope_scale_buff=k_nope_scale_buff,
        k_rope_buff=k_rope_buff,
        quant_group_size=G,
        scale_dtype="e8m0",
    )

    # --- Accuracy ---
    # Q: fp8 -> dequant vs fp8 ref; bf16 -> direct compare (Q not quantized)
    if q_fp8:
        # fp8 Q mirrors K: q_nope_scale_buff = nope fp8 [0:nope) + inline dup e8m0 scale
        # [nope:nope+2*n_groups); rotated Q-PE bf16 in the separate q_rope_buff.
        assert q_rope_buff is not None, "fp8 Q must produce q_rope_buff"
        q_nope_got = q_nope_scale_buff[..., :nope]
        q_scale_pairs = (
            q_nope_scale_buff.view(torch.uint8)[..., nope : nope + 2 * n_groups_q]
            .contiguous()
            .reshape(T, H, n_groups_q, 2)
        )
        got_scale_f32 = (q_scale_pairs[..., 0].to(torch.int32) << 23).view(
            torch.float32
        )
        ref_scale_f32 = (ref_q_scale.to(torch.int32) << 23).view(torch.float32)
        q_deq = q_nope_got.float() * got_scale_f32.unsqueeze(-1).expand(
            T, H, n_groups_q, G
        ).reshape(T, H, nope)
        ref_q_deq = ref_q_nope.float() * ref_scale_f32.unsqueeze(-1).expand(
            T, H, n_groups_q, G
        ).reshape(T, H, nope)
        err_q = checkAllclose(q_deq, ref_q_deq, atol=0.05, rtol=0.02, msg="Q-nope fp8")
        checkAllclose(
            q_scale_pairs[..., 0].float(),
            ref_q_scale.float(),
            atol=0.0,
            rtol=0.0,
            msg="Q scale e8m0",
        )
        checkAllclose(
            q_scale_pairs[..., 1].float(),
            ref_q_scale.float(),
            atol=0.0,
            rtol=0.0,
            msg="Q scale e8m0 dup",
        )
        checkAllclose(
            q_rope_buff.float(), ref_q_pe.float(), atol=0.01, rtol=0.01, msg="Q-pe bf16"
        )
    else:
        assert q_rope_buff is None, "bf16 Q must NOT produce q_rope_buff"
        assert q_nope_scale_buff.dtype == torch.bfloat16
        err_q = checkAllclose(
            q_nope_scale_buff.float(),
            ref_q_bf16.float(),
            atol=0.02,
            rtol=0.01,
            msg="Q bf16",
        )

    # K nope fp8 from k_nope_scale_buff[..., 0:nope]; e8m0 scale @[nope:nope+2*n_groups),
    # written as each tile-scale duplicated x2 (s0,s0,s1,s1,...). Take the first of
    # each pair for dequant; verify BOTH halves equal the reference scale.
    k_nope_got = k_nope_scale_buff[..., :nope]
    k_scale_pairs = (
        k_nope_scale_buff.view(torch.uint8)[..., nope : nope + 2 * n_groups_k]
        .contiguous()
        .reshape(T, NK, n_groups_k, 2)
    )
    got_k_scale_f32 = (k_scale_pairs[..., 0].to(torch.int32) << 23).view(torch.float32)
    ref_k_scale_f32 = (ref_k_scale.to(torch.int32) << 23).view(torch.float32)
    k_deq = k_nope_got.float() * got_k_scale_f32.unsqueeze(-1).expand(
        T, NK, n_groups_k, GK
    ).reshape(T, NK, nope)
    ref_k_deq = ref_k_nope.float() * ref_k_scale_f32.unsqueeze(-1).expand(
        T, NK, n_groups_k, GK
    ).reshape(T, NK, nope)
    err_k = checkAllclose(k_deq, ref_k_deq, atol=0.05, rtol=0.02, msg="K-nope fp8")
    checkAllclose(
        k_scale_pairs[..., 0].float(),
        ref_k_scale.float(),
        atol=0.0,
        rtol=0.0,
        msg="K scale e8m0",
    )
    checkAllclose(
        k_scale_pairs[..., 1].float(),
        ref_k_scale.float(),
        atol=0.0,
        rtol=0.0,
        msg="K scale e8m0 dup",
    )

    # K pe bf16 (NOT quantized) from the separate k_rope_buff
    err_kpe = checkAllclose(
        k_rope_buff.float(), ref_k_pe.float(), atol=0.01, rtol=0.01, msg="K-pe bf16"
    )

    # --- flydsl perf comparison ---
    # Match flydsl's quant mode to ours so the comparison is apples-to-apples:
    #   q_fp8=True  -> flydsl fp8 (1x64 e8m0)   [both write fp8 Q]
    #   q_fp8=False -> flydsl bf16 (quant off)  [both write bf16 Q]
    # (comparing bf16-Q against fp8-flydsl would just measure the 2x Q write.)
    fly_us = float("nan")
    if compare_flydsl:
        try:
            _, fly_us = run_perftest(
                flydsl_qk_norm_rope_quant,
                q.view(T, H * D),
                kv.view(T, D),
                kw,
                cos,
                sin,
                pos,
                num_q_heads=H,
                head_dim=D,
                rope_head_dim=RD,
                quant=q_fp8,
                quant_group_size=(G if q_fp8 else None),
                scale_dtype=("e8m0" if q_fp8 else "fp32"),
            )
        except Exception:
            fly_us = float("nan")

    # --- Bandwidth (effective): read q+kv+kw, write Q + K (nope+scale+rope) ---
    bytes_in = T * H * D * 2 + T * NK * D * 2 + D * 2
    if q_fp8:
        bytes_out = (
            T * H * (nope + 2 * n_groups_q) + T * H * RD * 2
        )  # Q nope+scale + Q rope bf16
    else:
        bytes_out = T * H * D * 2  # Q bf16 full head
    bytes_out += (
        T * NK * (nope + 2 * n_groups_k) + T * NK * RD * 2
    )  # K nope+scale + K rope bf16
    gbps = (bytes_in + bytes_out) / (us * 1e-6) / 1e9
    ratio = (us / fly_us) if fly_us == fly_us and fly_us > 0 else float("nan")

    # Only metrics here; the @benchmark decorator already echoes the call args
    # (T, H, D, RD, is_neox, q_fp8, G, NK, ...) as columns.
    return {
        "hip_us": round(us, 3),
        "flydsl_us": (round(fly_us, 3) if fly_us == fly_us else None),
        "hip/flydsl": (round(ratio, 3) if ratio == ratio else None),
        "GB/s": round(gbps, 0),
        "%peak": round(gbps / _PEAK_BW_GBPS * 100, 1),
        "err_q": err_q,
        "err_k": err_k,
        "err_kpe": err_kpe,
    }


# ============================================================================
# SWA fused ring-cache write test
# ============================================================================
#
# Decode-only fusion: the post-norm/rope K row is ALSO scattered into a per-request
# sliding-window ring that mirrors the main K output's two-buffer split:
#   swa_nope[slot, pos % cache_size, :] = k_nope_scale_buff[t]   (nope fp8 + dup e8m0 + pad)
#   swa_rope[slot, pos % cache_size, :] = k_rope_buff[t]         (rope bf16)
# where slot = state_slot_mapping[batch_id_per_token[t]]; batch_id == -1 (CG-pad) skips.
#
# Verification strategy: run the kernel, then REPLAY the scatter in python from the
# kernel's own main K outputs and compare byte-exact to the SWA ring. This validates
# the ring addressing + the verbatim entry copy (incl. the duplicated scale + pad)
# independently of the quant math (which the main test above already checks vs ref).


def _build_swa_batch(T, cache_size):
    """Collision-free decode-ish batch layout: split the first ``real`` tokens into
    ``bs`` contiguous sequences (distinct ring slot per seq, consecutive positions ->
    distinct pos%cache_size within a seq), plus a couple of trailing CG-pad (bid=-1)
    tokens that must be skipped by the kernel. Returns int32 batch_id [T], int64
    positions [T], int32 state_slot_mapping [bs], num_slots, bs, n_pad."""
    n_pad = min(2, T - 1) if T > 1 else 0
    real = T - n_pad
    # Pick bs so each seq has <= cache_size tokens (consecutive positions -> distinct
    # pos%cache_size within a seq => collision-free ring writes). At least 4 seqs when
    # there are enough tokens, to exercise multi-seq slot indirection.
    min_bs = (real + cache_size - 1) // cache_size  # ceil(real / cache_size)
    bs = max(min(4, real), min_bs)
    counts = [real // bs + (1 if i < real % bs else 0) for i in range(bs)]
    assert max(counts) <= cache_size, "per-seq token count must be <= cache_size"
    bid_list, pos_list = [], []
    for i, cnt in enumerate(counts):
        base = int(torch.randint(0, 64, (1,)).item())
        for j in range(cnt):
            bid_list.append(i)
            pos_list.append(base + j)
    for _ in range(n_pad):
        bid_list.append(-1)
        pos_list.append(0)
    bid = torch.tensor(bid_list, dtype=torch.int32, device=_DEV)
    pos = torch.tensor(pos_list, dtype=torch.int64, device=_DEV)
    num_slots = bs + 2  # extra slots to exercise the slot indirection
    slot_map = torch.randperm(num_slots, device=_DEV)[:bs].to(torch.int32)
    return bid, pos, slot_map, num_slots, bs, n_pad


@benchmark()
def test_fused_qk_norm_rope_group_quant_swa(T, H, D, RD, *, is_neox, q_fp8, G, GK=64):
    NK = 1
    torch.manual_seed(0)
    random.seed(0)
    nope = D - RD
    eps = 1e-6
    cache_size = 128  # DeepSeek-V4 sliding window
    n_groups_k = nope // GK
    entry = D

    bid, pos, slot_map, num_slots, bs, n_pad = _build_swa_batch(T, cache_size)
    cos, sin = _cos_sin(int(pos.max().item()) + 4, RD, torch.bfloat16)
    q = (torch.randn(T, H, D, device=_DEV) * 0.1).bfloat16()
    kv = (torch.randn(T, NK, D, device=_DEV) * 0.1).bfloat16()
    kw = (torch.randn(D, device=_DEV).abs() + 0.5).bfloat16()

    # K reference (vs the main cache outputs)
    ref_k_nope, ref_k_scale, ref_k_pe, _ = _norm_rope_nope_fp8(
        kv, kw, cos, sin, pos, eps, is_neox=is_neox, group_size=GK
    )

    # Main K/Q output buffers
    q_nope_scale_buff = (
        torch.zeros(T, H, entry, dtype=_FP8, device=_DEV)
        if q_fp8
        else torch.empty(T, H, D, dtype=torch.bfloat16, device=_DEV)
    )
    q_rope_buff = (
        torch.empty(T, H, RD, dtype=torch.bfloat16, device=_DEV) if q_fp8 else None
    )
    k_nope_scale_buff = torch.zeros(T, NK, entry, dtype=_FP8, device=_DEV)
    k_rope_buff = torch.empty(T, NK, RD, dtype=torch.bfloat16, device=_DEV)

    # SWA ring buffers (zero-init: unwritten cells must stay zero for the byte-compare)
    swa_nope = torch.zeros(num_slots, cache_size, entry, dtype=_FP8, device=_DEV)
    swa_rope = torch.zeros(num_slots, cache_size, RD, dtype=torch.bfloat16, device=_DEV)

    (q_nope_scale_buff, q_rope_buff, k_nope_scale_buff, k_rope_buff), us = run_perftest(
        aiter.fused_qk_norm_rope_group_quant,
        q,
        kv,
        kw,
        pos,
        cos,
        sin,
        eps,
        is_neox=is_neox,
        q_out_dtype=(_FP8 if q_fp8 else torch.bfloat16),
        q_nope_scale_buff=q_nope_scale_buff,
        q_rope_buff=q_rope_buff,
        k_nope_scale_buff=k_nope_scale_buff,
        k_rope_buff=k_rope_buff,
        quant_group_size=G,
        scale_dtype="e8m0",
        swa_nope_scale_buff=swa_nope,
        swa_rope_buff=swa_rope,
        state_slot_mapping=slot_map,
        batch_id_per_token=bid,
    )

    # --- Main K cache still correct (sanity vs reference) ---
    k_nope_got = k_nope_scale_buff[..., :nope]
    k_scale_pairs = (
        k_nope_scale_buff.view(torch.uint8)[..., nope : nope + 2 * n_groups_k]
        .contiguous()
        .reshape(T, NK, n_groups_k, 2)
    )
    got_k_scale_f32 = (k_scale_pairs[..., 0].to(torch.int32) << 23).view(torch.float32)
    ref_k_scale_f32 = (ref_k_scale.to(torch.int32) << 23).view(torch.float32)
    k_deq = k_nope_got.float() * got_k_scale_f32.unsqueeze(-1).expand(
        T, NK, n_groups_k, GK
    ).reshape(T, NK, nope)
    ref_k_deq = ref_k_nope.float() * ref_k_scale_f32.unsqueeze(-1).expand(
        T, NK, n_groups_k, GK
    ).reshape(T, NK, nope)
    err_k = checkAllclose(k_deq, ref_k_deq, atol=0.05, rtol=0.02, msg="K-nope fp8")
    err_kpe = checkAllclose(
        k_rope_buff.float(), ref_k_pe.float(), atol=0.01, rtol=0.01, msg="K-pe bf16"
    )

    # --- SWA ring == replay(main K outputs) ---
    # Build the expected ring by replaying the scatter from the kernel's own main K
    # outputs (last-write-wins; the collision-free layout makes order irrelevant).
    exp_nope = torch.zeros_like(swa_nope)
    exp_rope = torch.zeros_like(swa_rope)
    bid_cpu = bid.tolist()
    pos_cpu = pos.tolist()
    slot_cpu = slot_map.tolist()
    n_written = 0
    for t in range(T):
        b = bid_cpu[t]
        if b < 0:
            continue
        slot = slot_cpu[b]
        ring = pos_cpu[t] % cache_size
        exp_nope[slot, ring] = k_nope_scale_buff[t, 0]
        exp_rope[slot, ring] = k_rope_buff[t, 0]
        n_written += 1

    # Byte-exact compare (pure copy): fp8 entry incl. nope + dup scale + pad, and rope bf16.
    err_swa_nope = checkAllclose(
        swa_nope.view(torch.uint8).float(),
        exp_nope.view(torch.uint8).float(),
        atol=0.0,
        rtol=0.0,
        msg="SWA nope+scale entry (byte-exact)",
    )
    err_swa_rope = checkAllclose(
        swa_rope.float(),
        exp_rope.float(),
        atol=0.0,
        rtol=0.0,
        msg="SWA rope bf16 (exact)",
    )

    return {
        "hip_us": round(us, 3),
        "bs": bs,
        "num_slots": num_slots,
        "n_pad": n_pad,
        "n_written": n_written,
        "err_k": err_k,
        "err_kpe": err_kpe,
        "err_swa_nope": err_swa_nope,
        "err_swa_rope": err_swa_rope,
    }


# ============================================================================
# argparse + matrix sweep
# ============================================================================

parser = argparse.ArgumentParser(
    formatter_class=argparse.RawTextHelpFormatter,
    description="aiter test for fused_qk_norm_rope_group_quant (V4, token-contiguous, no cache).",
)
parser.add_argument(
    "-T",
    "--T",
    type=int,
    nargs="*",
    default=[4, 16, 64, 256, 1024, 4096, 16384],
    help="token-count sweep. e.g. -T 4 64 1024",
)
parser.add_argument(
    "--H",
    type=int,
    nargs="*",
    default=[16, 128],
    help="num-Q-heads-per-rank sweep. e.g. --H 16 128",
)
parser.add_argument("--D", type=int, default=512, help="head_dim (kernel MVP: 512)")
parser.add_argument("--RD", type=int, default=64, help="rope_head_dim (RoPE tail size)")
parser.add_argument(
    "--G",
    type=int,
    nargs="*",
    default=[32, 64],
    choices=[32, 64],
    help="Q quant_group_size sweep (K is always 64). e.g. --G 64",
)
parser.add_argument(
    "--neox", action="store_true", help="also sweep is_neox=True (default: GPT-J only)."
)
parser.add_argument(
    "--no-flydsl", action="store_true", help="skip the flydsl perf comparison."
)
parser.add_argument(
    "--q-bf16", action="store_true", help="use bf16 Q (default: fp8 Q)."
)
parser.add_argument(
    "--swa",
    action="store_true",
    help="run ONLY the fused SWA ring-cache write test (decode-only).",
)
parser.add_argument(
    "--no-swa",
    action="store_true",
    help="skip the fused SWA ring-cache write test.",
)
args = parser.parse_args()

neox_modes = [False, True] if args.neox else [False]

rows = []
q_fp8 = not args.q_bf16
if not args.swa:
    for G, H, neox in itertools.product(args.G, args.H, neox_modes):
        for T in args.T:
            rows.append(
                test_fused_qk_norm_rope_group_quant(
                    T,
                    H,
                    args.D,
                    args.RD,
                    is_neox=neox,
                    q_fp8=q_fp8,
                    G=G,
                    GK=64,
                    compare_flydsl=not args.no_flydsl,
                )
            )

    df = pd.DataFrame(rows)
    aiter.logger.info(
        "fused_qk_norm_rope_group_quant (V4) summary (markdown):\n%s",
        df.to_markdown(index=False),
    )

# --- SWA fused ring-cache write sweep (decode-only; small T to stay off the
# fine-grained xlarge path which does not carry the SWA scatter) ---
if args.swa or not args.no_swa:
    # cap T to the decode/med range (<= 1024) so the coarse kernel (shared K-wave
    # body) runs; the fine-grained xlarge path asserts out SWA by design.
    swa_T = [t for t in args.T if t <= 1024] or [16, 64, 256]
    swa_rows = []
    for G, H, neox in itertools.product(args.G, args.H, neox_modes):
        for T in swa_T:
            swa_rows.append(
                test_fused_qk_norm_rope_group_quant_swa(
                    T,
                    H,
                    args.D,
                    args.RD,
                    is_neox=neox,
                    q_fp8=q_fp8,
                    G=G,
                    GK=64,
                )
            )
    swa_df = pd.DataFrame(swa_rows)
    aiter.logger.info(
        "fused_qk_norm_rope_group_quant SWA ring-write summary (markdown):\n%s",
        swa_df.to_markdown(index=False),
    )
