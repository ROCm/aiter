#!/usr/bin/env python
# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""aiter op-test for ``flydsl_qk_norm_rope_quant``.

Validates the fused RMSNorm + GPT-J RoPE + (optional) FP8 quant kernel
against a pure-torch reference, and reports per-config kernel us /
bandwidth utilization.

Sweeps:
- T (sequence / decode-batch length)
- (group_size, scale_dtype) combos: per-row fp32, 1x128 fp32 / e8m0, 1x64 ...
- with vs without optional ``q_weight``

Usage:
    python op_tests/test_flydsl_qk_norm_rope_quant.py
    python op_tests/test_flydsl_qk_norm_rope_quant.py -T 64 256 1024 -q fp8_1x128_e8m0
    python op_tests/test_flydsl_qk_norm_rope_quant.py --no-quant   # bf16 only
    python op_tests/test_flydsl_qk_norm_rope_quant.py --init zero
    python op_tests/test_flydsl_qk_norm_rope_quant.py --init constant --init-val 0.5
    python op_tests/test_flydsl_qk_norm_rope_quant.py --seed 42 --init uniform --init-scale 0.2
"""

import argparse
import itertools
import math

import torch

import aiter
from aiter import dtypes
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.flydsl import flydsl_qk_norm_rope_quant
from aiter.test_common import benchmark, checkAllclose, print_json_table, run_perftest

torch.set_default_device("cuda")

# The public wrapper dispatches wave64 (gfx942/gfx950) vs wave32 (gfx1250)
# internally, so one test covers every card. Positive allow-list: an unknown
# new card must not silently run an unbuilt kernel.
SUPPORTED_GFX = ["gfx942", "gfx950", "gfx1250"]

# Shared constants (independent of attention shape).
_EPS = 1e-6
_SQRT2 = math.sqrt(2.0)
_FP8_DTYPE = dtypes.fp8
_FP8_MAX = float(torch.finfo(_FP8_DTYPE).max)

# Data-initialisation modes
_INIT_MODES = ("normal", "uniform", "zero", "constant")


def _make_data(shape, *, dtype, device, mode="normal", scale=0.1, val=1.0):
    """Create a data tensor in the requested initialisation mode."""
    if mode == "zero":
        return torch.zeros(shape, dtype=dtype, device=device)
    if mode == "constant":
        return torch.full(shape, val, dtype=dtype, device=device)
    if mode == "uniform":
        t = torch.empty(shape, dtype=torch.float32, device=device).uniform_(-1, 1)
        return (t * scale).to(dtype)
    return torch.randn(shape, dtype=dtype, device=device) * scale


# ============================================================================
# Reference (pure torch)
# ============================================================================


def _rope_tail_ref(x, cos2d, sin2d, pos, *, D, RD):
    """GPT-J pair-interleaved RoPE on the last RD dims."""
    NOPE = D - RD
    T = x.shape[0]
    leading = x.shape[1:-1]
    tail = x[..., NOPE:].reshape(T, *leading, RD // 2, 2)
    c = cos2d[pos].reshape(T, *((1,) * len(leading)), RD // 2)
    s = sin2d[pos].reshape(T, *((1,) * len(leading)), RD // 2)
    even, odd = tail[..., 0], tail[..., 1]
    new_e = even * c - odd * s
    new_o = even * s + odd * c
    tail_new = torch.stack([new_e, new_o], dim=-1).reshape(T, *leading, RD)
    return torch.cat([x[..., :NOPE], tail_new], dim=-1)


def _e8m0_encode_ref(amax_safe):
    """E8M0 block-scale reference, following the project default round mode.

    Delegates the e8m0-byte derivation to the shared CPU helper
    ``fp4_utils.f32_to_mx_e8m0_scale(mode=MX_DEFAULT_ROUND_MODE,
    dtype=FP8_E4M3)`` -- the single source the HIP / FlyDSL kernels mirror --
    so the test follows the project-wide default instead of hard-coding
    RoundUp. (moe_sorting confirms this CPU helper matches the reciprocal-
    multiply HIP kernel byte-for-byte; qk_norm only compares dequant output
    under a loose tolerance, so any rare 1-ULP boundary case is absorbed.)

    Returns ``(byte_uint8, factor_fp32)`` where ``factor = 1 / dequant_scale
    = 2^(127 - byte)`` is the multiplier applied to ``x_norm`` so
    ``out = x_norm * factor`` lands in fp8 range.
    """
    from aiter.utility import fp4_utils
    from aiter.utility.mx_types import MX_DEFAULT_ROUND_MODE, MxDtype

    e8m0_biased = (
        fp4_utils.f32_to_mx_e8m0_scale(
            amax_safe.float(), mode=MX_DEFAULT_ROUND_MODE, dtype=MxDtype.FP8_E4M3
        )
        .view(torch.uint8)
        .to(torch.int64)
    )
    quant_exp = (254 - e8m0_biased).to(torch.int32)
    factor = (quant_exp << 23).view(torch.float32)
    return e8m0_biased.to(torch.uint8), factor


def _flydsl_qk_norm_rope_ref(
    q,
    kv,
    kv_weight,
    cos_cache,
    sin_cache,
    positions,
    *,
    H,
    D,
    RD,
    q_weight=None,
    quant=False,
    quant_group_size=None,
    scale_dtype="fp32",
):
    """Pure-torch reference. Returns same tuple as the kernel."""
    T = q.shape[0]
    G = quant_group_size if quant_group_size is not None else D
    NG = D // G

    q3 = q.view(T, H, D).float()
    kvf = kv.float()
    rstd_q = torch.rsqrt(q3.pow(2).mean(-1, keepdim=True) + _EPS)
    rstd_kv = torch.rsqrt(kvf.pow(2).mean(-1, keepdim=True) + _EPS)
    q_n = q3 * rstd_q
    if q_weight is not None:
        q_n = q_n * q_weight.float()
    kv_n = kvf * rstd_kv * kv_weight.float()

    cos2d = cos_cache.view(cos_cache.shape[0], cos_cache.shape[-1]).float()
    sin2d = sin_cache.view(sin_cache.shape[0], sin_cache.shape[-1]).float()
    q_roped = _rope_tail_ref(q_n, cos2d, sin2d, positions, D=D, RD=RD)
    kv_roped = _rope_tail_ref(kv_n, cos2d, sin2d, positions, D=D, RD=RD)

    if not quant:
        return (
            q_roped.to(torch.bfloat16),
            kv_roped.to(torch.bfloat16),
            None,
            None,
        )

    # Per-group amax of pre-RoPE x_norm x SQRT2 (post-rope upper bound).
    q_groups = q_n.reshape(T, H, NG, G)
    kv_groups = kv_n.reshape(T, NG, G)
    am_q = (q_groups.abs().amax(-1) * _SQRT2).clamp_min(1e-12)
    am_kv = (kv_groups.abs().amax(-1) * _SQRT2).clamp_min(1e-12)

    if scale_dtype == "fp32":
        factor_q = _FP8_MAX / am_q
        factor_kv = _FP8_MAX / am_kv
        scale_q_store = (am_q / _FP8_MAX).to(torch.float32)
        scale_kv_store = (am_kv / _FP8_MAX).to(torch.float32)
    elif scale_dtype == "e8m0":
        scale_q_store, factor_q = _e8m0_encode_ref(am_q)
        scale_kv_store, factor_kv = _e8m0_encode_ref(am_kv)
    else:
        raise ValueError(scale_dtype)

    factor_q_full = factor_q.unsqueeze(-1).expand(*factor_q.shape, G).reshape(T, H, D)
    factor_kv_full = factor_kv.unsqueeze(-1).expand(*factor_kv.shape, G).reshape(T, D)
    q_fp8 = (q_roped * factor_q_full).clamp(-_FP8_MAX, _FP8_MAX).to(_FP8_DTYPE)
    kv_fp8 = (kv_roped * factor_kv_full).clamp(-_FP8_MAX, _FP8_MAX).to(_FP8_DTYPE)
    return q_fp8, kv_fp8, scale_q_store, scale_kv_store


# ============================================================================
# Dequant for fp8 -> fp32 comparison
# ============================================================================


def _dequant(out_fp8, scale, *, D, quant_group_size, scale_dtype):
    T = out_fp8.shape[0]
    leading = out_fp8.shape[1:-1]
    G = quant_group_size if quant_group_size is not None else D
    if scale_dtype == "fp32":
        scale_f = scale.float()
    else:
        # MX e8m0: dequant_scale = 2^(byte - 127) -> bits = (byte << 23)
        bits = scale.to(torch.int32) << 23
        scale_f = bits.view(torch.float32)
    scale_full = scale_f.unsqueeze(-1).expand(*scale_f.shape, G).reshape(T, *leading, D)
    return out_fp8.float() * scale_full


# ============================================================================
# Main test (per-config)
# ============================================================================

# Nominal HBM peak per arch, used only for the "%peak" perf column. A single
# constant made that column meaningless on every card but one.
_PEAK_BW_GBPS_BY_GFX = {
    "gfx942": 5300.0,  # MI300X (MI325X is 6000)
    "gfx950": 8000.0,  # MI355X HBM3E
    "gfx1250": 22000.0,
}


def _pct_peak(gbps):
    """%peak against this card's nominal HBM peak; None when the arch is unknown."""
    peak = _PEAK_BW_GBPS_BY_GFX.get(get_gfx())
    return round(gbps / peak * 100, 1) if peak else None


# Pin the arg-rotation count. Left to itself, run_perftest derives it from
# `free_memory` at call time, so two rows timed in one process rotate a
# different number of times and land in different L2 states -- their `us`
# columns then are not comparable with each other.
_ROTATE = 4


@benchmark()
def test_flydsl_qk_norm_rope_quant(
    T,
    H,
    D,
    RD,
    *,
    quant_group_size,
    scale_dtype,
    q_weighted,
    quant,
    seed=0,
    init_mode="normal",
    init_scale=0.1,
    init_val=1.0,
):
    torch.manual_seed(seed)
    device = torch.device("cuda")

    # Build cos/sin via a YaRN-style table covering all positions in T.
    max_pos = max(T, 64)
    inv_freq = 1.0 / (10000 ** (torch.arange(0, RD, 2, device=device).float() / RD))
    pos_range = torch.arange(max_pos, device=device).float()
    freqs = torch.einsum("i,j->ij", pos_range, inv_freq)
    cos = freqs.cos().to(torch.bfloat16).contiguous()
    sin = freqs.sin().to(torch.bfloat16).contiguous()

    _ikw = {
        "dtype": torch.bfloat16,
        "device": device,
        "mode": init_mode,
        "scale": init_scale,
        "val": init_val,
    }
    q = _make_data((T, H * D), **_ikw)
    # Mimic V4 KV split: kv = strided view into a wider tensor
    Q_LORA = 1536
    qkv_a = _make_data((T, Q_LORA + D), **_ikw)
    _, kv = torch.split(qkv_a, [Q_LORA, D], dim=-1)
    kv_w = torch.randn(D, dtype=torch.bfloat16, device=device).abs() + 0.5
    q_w = (
        (torch.randn(D, dtype=torch.bfloat16, device=device).abs() + 0.5)
        if q_weighted
        else None
    )
    pos = torch.randint(0, max_pos - 1, (T,), dtype=torch.int64, device=device)

    # Reference
    ref_q, ref_kv, ref_qs, ref_ks = _flydsl_qk_norm_rope_ref(
        q,
        kv.contiguous(),
        kv_w,
        cos,
        sin,
        pos,
        H=H,
        D=D,
        RD=RD,
        q_weight=q_w,
        quant=quant,
        quant_group_size=quant_group_size,
        scale_dtype=scale_dtype,
    )

    # Kernel + perf
    (got_q, got_kv, got_qs, got_ks), us = run_perftest(
        flydsl_qk_norm_rope_quant,
        q,
        kv,
        kv_w,
        cos,
        sin,
        pos,
        num_q_heads=H,
        head_dim=D,
        rope_head_dim=RD,
        q_weight=q_w,
        quant=quant,
        quant_group_size=quant_group_size,
        scale_dtype=scale_dtype,
        num_rotate_args=_ROTATE,
    )

    # Accuracy
    if quant:
        deq_kw = {
            "D": D,
            "quant_group_size": quant_group_size,
            "scale_dtype": scale_dtype,
        }
        got_deq = _dequant(got_q, got_qs, **deq_kw)
        ref_deq = _dequant(ref_q, ref_qs, **deq_kw)
        got_kv_deq = _dequant(got_kv, got_ks, **deq_kw)
        ref_kv_deq = _dequant(ref_kv, ref_ks, **deq_kw)
        # Looser tolerance under fp8 + group quant -- pow2 rounding plus bf16
        # RoPE noise pushes per-element diffs into the 0.1-10 range depending
        # on amax. cos-sim (computed via checkAllclose's atol on row sums)
        # remains > 0.999 in all configs we ship.
        rtol, atol = 0.05, 5.0
    else:
        got_deq = got_q.float()
        ref_deq = ref_q.float()
        got_kv_deq = got_kv.float()
        ref_kv_deq = ref_kv.float()
        rtol, atol = 1e-3, 1e-2

    err_q = checkAllclose(
        ref_deq, got_deq, rtol=rtol, atol=atol, msg="Q (rmsnorm+rope+quant)"
    )
    err_kv = checkAllclose(
        ref_kv_deq, got_kv_deq, rtol=rtol, atol=atol, msg="KV (rmsnorm+rope+quant)"
    )

    # Bandwidth-utilization estimate (Q in/out + KV in/out + scales when quant)
    bytes_in = T * H * D * 2 + T * D * 2 + D * 2  # Q + KV + kv_weight (small)
    if q_weighted:
        bytes_in += D * 2
    if quant:
        out_bytes_per_elem = 1
        G = quant_group_size if quant_group_size is not None else D
        NG = D // G
        scale_bytes = (T * H + T) * NG * (4 if scale_dtype == "fp32" else 1)
        bytes_out = T * H * D * out_bytes_per_elem + T * D * out_bytes_per_elem
        bytes_total = bytes_in + bytes_out + scale_bytes
    else:
        bytes_out = T * H * D * 2 + T * D * 2
        bytes_total = bytes_in + bytes_out
    gbps = bytes_total / (us * 1e-6) / 1e9

    return {
        "us": round(us, 3),
        "GB/s": round(gbps, 0),
        "%peak": _pct_peak(gbps),
        "err_q": err_q,
        "err_kv": err_kv,
    }


def test_flydsl_qk_norm_rope_quant_cos_sin_4d():
    """Cover the advertised cos/sin layout that DeepSeek-V4 uses.

    The wrapper docstring states cos/sin caches may have any leading shape
    whose last dim is RD/2 -- DeepSeek-V4 stores them as
    ``[max_pos, 1, 1, RD/2]``. The matrix sweep above only exercises the
    2D ``[max_pos, RD/2]`` shape, so add a single smoke case that reshapes
    cos/sin to 4D and verifies the output is bit-identical to the 2D path.
    """
    torch.manual_seed(0)
    device = torch.device("cuda")

    T, H, D, RD = 16, 16, 512, 64

    max_pos = max(T, 64)
    inv_freq = 1.0 / (10000 ** (torch.arange(0, RD, 2, device=device).float() / RD))
    pos_range = torch.arange(max_pos, device=device).float()
    freqs = torch.einsum("i,j->ij", pos_range, inv_freq)
    cos_2d = freqs.cos().to(torch.bfloat16).contiguous()
    sin_2d = freqs.sin().to(torch.bfloat16).contiguous()
    cos_4d = cos_2d.view(max_pos, 1, 1, RD // 2)
    sin_4d = sin_2d.view(max_pos, 1, 1, RD // 2)

    q = torch.randn(T, H * D, dtype=torch.bfloat16, device=device) * 0.1
    Q_LORA = 1536
    qkv_a = torch.randn(T, Q_LORA + D, dtype=torch.bfloat16, device=device) * 0.1
    _, kv = torch.split(qkv_a, [Q_LORA, D], dim=-1)
    kv_w = torch.randn(D, dtype=torch.bfloat16, device=device).abs() + 0.5
    pos = torch.randint(0, max_pos - 1, (T,), dtype=torch.int64, device=device)

    out_2d = flydsl_qk_norm_rope_quant(
        q,
        kv,
        kv_w,
        cos_2d,
        sin_2d,
        pos,
        num_q_heads=H,
        head_dim=D,
        rope_head_dim=RD,
    )
    out_4d = flydsl_qk_norm_rope_quant(
        q,
        kv,
        kv_w,
        cos_4d,
        sin_4d,
        pos,
        num_q_heads=H,
        head_dim=D,
        rope_head_dim=RD,
    )
    # 2D vs 4D cos/sin must produce bit-identical results -- the wrapper just
    # .view()s the cache; identical underlying storage means identical loads.
    torch.testing.assert_close(out_2d[0], out_4d[0], atol=0.0, rtol=0.0)
    torch.testing.assert_close(out_2d[1], out_4d[1], atol=0.0, rtol=0.0)


# ============================================================================
# Fused SWA cache-write (BF16 only)
# ============================================================================
#
# With ``swa_kv`` provided, the kernel scatters each token's post-norm/rope KV
# row into a second pool in the same launch, fusing what used to be a separate
# ``swa_write`` kernel. Two addressing modes, one per value of ``mode``:
#
#   direct : row = swa_dest_rows[t]                       (caller resolved it)
#   paged  : row = block_tables[bid, pos//B]*B + pos%B    (content-addressed)
#
# The scatter stores the SAME bytes the kernel writes to ``kv_out``, so the
# pool must be bit-exact against a gather built from ``kv_out`` -- no separate
# torch reference is needed or meaningful for the scatter itself.
#
# Every skip path the kernel implements gets a token, because each is a
# distinct gate and a regression in one is invisible to the others:
#
#   both   : batch_id_per_token[t] == -1   CG-pad token
#   both   : row past the pool             stale row from a larger pool
#   direct : swa_dest_rows[t] == -1        caller declined this token
#   paged  : block_tables[...] == -1       block outside the window
#   paged  : blk >= max_blocks             position past the table's end
#
# Not covered: the kernel's ``pos < 0`` gate. A negative position is not a
# legal input to this op -- the main path indexes cos/sin with the raw position
# long before the scatter -- so that gate is defence-in-depth mirroring the C++
# sibling, and feeding one in would make the whole launch undefined rather than
# exercise the gate.

_SWA_BLOCK_SIZE = 8  # small, so a block-index overrun is reachable at modest T
_SWA_MAX_BLOCKS = 4
_SWA_BS = 4  # sequences in the paged block table
_SWA_GUARD_ROWS = 16


def _build_swa_case(T, mode, *, device):
    """Batch layout for the fused SWA scatter, one token per skip path.

    Returns ``(bid, pos, index_tensor, num_rows, dest)`` where ``dest[t]`` is
    the pool row token ``t`` must land on, or ``-1`` if the kernel must skip
    it. ``index_tensor`` is what the caller passes: ``swa_dest_rows`` in direct
    mode, ``swa_block_tables`` in paged mode.
    """
    # Reserve the last three tokens for skip paths; the rest are plain writes.
    n_special = 3 if T >= 6 else 0
    n_write = T - n_special
    bid_l = [t % _SWA_BS for t in range(n_write)]
    pos_l = [t // _SWA_BS for t in range(n_write)]

    if mode == "paged":
        # Plain writes are confined to blocks [0, max_blocks-1), leaving the
        # LAST block free for the out-of-window sentinel. Without that reserve,
        # a large enough T has a plain token allocate the very block the
        # sentinel wants, the sentinel stops being a sentinel, and two tokens
        # land on one row -- which breaks the collision-free premise the
        # replay-and-compare depends on.
        cap = (_SWA_MAX_BLOCKS - 1) * _SWA_BLOCK_SIZE
        assert n_write <= _SWA_BS * cap, f"T={T} exceeds the paged table capacity"
        # t_sent: a block deliberately left at the -1 out-of-window sentinel.
        # t_over: blk past the table. Its overrun lands on flat index
        # bid*max_blocks + blk, i.e. a LATER sequence's row -- point it at an
        # ALLOCATED entry, else the phys<0 gate catches it first and the block
        # bound is never actually exercised.
        if n_special:
            bid_l += [-1, 0, 1]
            pos_l += [
                0,
                (_SWA_MAX_BLOCKS - 1) * _SWA_BLOCK_SIZE,  # the reserved block
                _SWA_MAX_BLOCKS * _SWA_BLOCK_SIZE + 3,  # blk past the table
            ]
        t_sent = n_write + 1 if n_special else -1

        bt = torch.full((_SWA_BS, _SWA_MAX_BLOCKS), -1, dtype=torch.int32)
        nxt = 0
        for t in range(len(bid_l)):
            b, pv = bid_l[t], pos_l[t]
            if b < 0 or t == t_sent:
                continue
            blk = pv // _SWA_BLOCK_SIZE
            if blk >= _SWA_MAX_BLOCKS:
                continue
            if int(bt[b, blk]) < 0:
                bt[b, blk] = nxt
                nxt += 1
        # The overrun token reads table[flat // max_blocks, flat % max_blocks];
        # make sure that entry is real so only the blk bound can stop it.
        if n_special:
            flat = bid_l[-1] * _SWA_MAX_BLOCKS + pos_l[-1] // _SWA_BLOCK_SIZE
            r, c = flat // _SWA_MAX_BLOCKS, flat % _SWA_MAX_BLOCKS
            if r < _SWA_BS and int(bt[r, c]) < 0:
                bt[r, c] = nxt
                nxt += 1
        num_rows = max(nxt, 1) * _SWA_BLOCK_SIZE

        dest = []
        for t in range(len(bid_l)):
            b, pv = bid_l[t], pos_l[t]
            blk = pv // _SWA_BLOCK_SIZE
            if b < 0 or blk >= _SWA_MAX_BLOCKS:
                dest.append(-1)
                continue
            phys = int(bt[b, blk])
            row = phys * _SWA_BLOCK_SIZE + pv % _SWA_BLOCK_SIZE
            dest.append(-1 if phys < 0 or row >= num_rows else row)
        index_t = bt.to(device)
    else:
        if n_special:
            bid_l += [-1, 0, 1]
            pos_l += [0, 0, 0]
        num_rows = max(T + _SWA_GUARD_ROWS, 32)
        # Scattered, NOT identity: a kernel that used the token index instead of
        # the supplied row would still pass an identity mapping.
        rows = torch.randperm(num_rows, device=device)[:T].to(torch.int32)
        dest = [int(r) for r in rows.tolist()]
        if n_special:
            dest[-3] = -1  # bid == -1
            dest[-2] = -1  # caller declined this token
            dest[-1] = num_rows + 5  # stale row past the pool
            rows[-2] = -1
            rows[-1] = num_rows + 5
        for t in range(T):
            if bid_l[t] < 0 or dest[t] < 0 or dest[t] >= num_rows:
                dest[t] = -1
        index_t = rows

    bid = torch.tensor(bid_l, dtype=torch.int32, device=device)
    pos = torch.tensor(pos_l, dtype=torch.int64, device=device)
    return bid, pos, index_t, num_rows, dest


@benchmark()
def test_flydsl_swa_write(
    T, H, D, RD, mode, *, seed=0, init_mode="normal", init_scale=0.1, init_val=1.0
):
    torch.manual_seed(seed)
    device = torch.device("cuda")

    bid, pos, index_t, num_rows, dest = _build_swa_case(T, mode, device=device)
    max_pos = int(pos.max().item()) + 4
    inv_freq = 1.0 / (10000 ** (torch.arange(0, RD, 2, device=device).float() / RD))
    freqs = torch.einsum(
        "i,j->ij", torch.arange(max_pos, device=device).float(), inv_freq
    )
    cos = freqs.cos().to(torch.bfloat16).contiguous()
    sin = freqs.sin().to(torch.bfloat16).contiguous()

    _ikw = {
        "dtype": torch.bfloat16,
        "device": device,
        "mode": init_mode,
        "scale": init_scale,
        "val": init_val,
    }
    q = _make_data((T, H * D), **_ikw)
    # Mimic the V4 KV split: kv is a strided view into a wider tensor, exactly
    # as the model hands it over.
    Q_LORA = 1536
    qkv_a = _make_data((T, Q_LORA + D), **_ikw)
    _, kv = torch.split(qkv_a, [Q_LORA, D], dim=-1)
    kv_w = torch.randn(D, dtype=torch.bfloat16, device=device).abs() + 0.5

    # Carve the pool out of a larger allocation. The out-of-bounds gates prevent
    # a write; they change no in-pool byte, so only a dirtied guard row can show
    # that one regressed.
    G = _SWA_GUARD_ROWS
    # A zero-filled pool cannot distinguish "the kernel wrote a zero row" from
    # "the kernel skipped this row" when --init zero. Seed every row with an
    # unreachable sentinel so the write-mask check remains valid for all data
    # distributions.
    sentinel = torch.finfo(torch.bfloat16).max
    pool = torch.full(
        (G + num_rows + G, D), sentinel, dtype=torch.bfloat16, device=device
    )
    swa_kv = pool[G : G + num_rows]
    mode_kw = (
        {"swa_dest_rows": index_t}
        if mode == "direct"
        else {"swa_block_tables": index_t, "swa_block_size": _SWA_BLOCK_SIZE}
    )

    # Correctness and perf are measured SEPARATELY on purpose. The scatter
    # writes in place, and run_perftest rotates its args across several deep
    # copies of every tensor -- so the guarded pool inspected below would be
    # written by some subset of the iterations while the returned tensors come
    # from another, which made err intermittently non-zero. One untimed launch
    # owns the correctness check; the timed launch writes a throwaway pool and
    # its output is discarded.
    got_q, got_kv, _, _ = flydsl_qk_norm_rope_quant(
        q,
        kv,
        kv_w,
        cos,
        sin,
        pos,
        num_q_heads=H,
        head_dim=D,
        rope_head_dim=RD,
        swa_kv=swa_kv,
        batch_id_per_token=bid,
        **mode_kw,
    )
    _, us = run_perftest(
        flydsl_qk_norm_rope_quant,
        q,
        kv,
        kv_w,
        cos,
        sin,
        pos,
        num_q_heads=H,
        head_dim=D,
        rope_head_dim=RD,
        swa_kv=torch.zeros_like(swa_kv),
        batch_id_per_token=bid,
        **mode_kw,
        num_rotate_args=_ROTATE,
    )

    # The scatter is a verbatim copy of kv_out, so the reference IS kv_out
    # gathered onto the rows the addressing mode selects.
    expected = torch.full_like(swa_kv, sentinel)
    n_written = 0
    for t in range(T):
        if dest[t] < 0:
            continue
        expected[dest[t]] = got_kv[t]
        n_written += 1
    # The point of the layout is that some tokens are skipped; if every token
    # landed, the gates below would be vacuously satisfied.
    assert (
        n_written < T or T < 6
    ), f"{mode}: no skip path exercised at T={T} ({n_written}/{T} written)"

    err_swa = checkAllclose(
        expected.to(dtypes.fp32),
        swa_kv.to(dtypes.fp32),
        rtol=0,
        atol=0,  # pure byte copy: must be exact
        msg=f"SWA pool ({mode})",
    )
    # A skipped token must reach NO row, not merely the right one.
    assert (
        int((swa_kv != sentinel).any(dim=1).sum()) == n_written
    ), f"{mode}: a skipped token still reached the pool"
    assert (pool[:G] == sentinel).all(), f"{mode}: scatter wrote BEFORE the pool"
    assert (pool[G + num_rows :] == sentinel).all(), (
        f"{mode}: scatter wrote PAST the pool"
    )

    # The scatter must not perturb the primary outputs.
    ref_q, ref_kv, _, _ = flydsl_qk_norm_rope_quant(
        q, kv, kv_w, cos, sin, pos, num_q_heads=H, head_dim=D, rope_head_dim=RD
    )
    err_q = checkAllclose(
        ref_q.to(dtypes.fp32), got_q.to(dtypes.fp32), rtol=0, atol=0, msg="Q vs no-SWA"
    )
    err_kv = checkAllclose(
        ref_kv.to(dtypes.fp32),
        got_kv.to(dtypes.fp32),
        rtol=0,
        atol=0,
        msg="KV vs no-SWA",
    )

    # RMSNorm ~4 flop/elem (square, accumulate, mul rstd, mul weight); GPT-J
    # RoPE ~3 flop/elem over the RD tail. Q and KV rows both do both.
    rows = T * H + T
    flops = rows * D * 4 + rows * RD * 3
    # Read q+kv, write q_out+kv_out, plus the scattered rows. bf16 throughout.
    nbytes = (rows * D * 2) * 2 + n_written * D * 2

    return {
        "gfx": get_gfx(),
        "rows_written": n_written,
        "us": round(us, 3),
        "TFLOPS": round(flops / us / 1e6, 2),
        "TB/s": round(nbytes / us / 1e6, 3),
        "err_swa": err_swa,
        "err_q": err_q,
        "err_kv": err_kv,
    }


_QUANT_OPTIONS = {
    "bf16": (False, None, "fp32", False),  # quant_q,kv off
    "fp8_per_row_fp32": (True, None, "fp32", False),
    "fp8_1x128_fp32": (True, 128, "fp32", False),
    "fp8_1x64_fp32": (True, 64, "fp32", False),
    "fp8_1x128_e8m0": (True, 128, "e8m0", False),
    "fp8_1x64_e8m0": (True, 64, "e8m0", False),
    # kernel supported, but no use case yet
    # "fp8_1x32_fp32": (True, 32, "fp32", False),
    # "fp8_1x32_e8m0": (True, 32, "e8m0", False),
}


def main():
    if get_gfx() not in SUPPORTED_GFX:
        aiter.logger.warning(
            "flydsl_qk_norm_rope_quant unsupported on %s; skipping", get_gfx()
        )
        return

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="aiter test for flydsl_qk_norm_rope_quant (V4-Pro decode shape).",
    )
    parser.add_argument(
        "-T",
        "--T",
        type=int,
        nargs="*",
        default=[1, 2, 16, 32, 64, 128, 192, 256, 512, 1024, 16384, 65540],
        help="token-count sweep. e.g. -T 4 64 1024",
    )
    parser.add_argument(
        "--H",
        type=int,
        nargs="*",
        default=[16, 64, 128],
        help="num-Q-heads-per-rank sweep. e.g. --H 16 128",
    )
    parser.add_argument(
        "--D",
        type=int,
        nargs="*",
        default=[512],
        help=(
            "head_dim sweep. Current kernel MVP only supports D=512 (VEC=8); "
            "other D values are rejected with a clear assert until atom-widths + "
            "fp8 packing are generalised."
        ),
    )
    parser.add_argument(
        "--RD",
        type=int,
        default=64,
        help="rope_head_dim (RoPE tail size, single value)",
    )
    parser.add_argument(
        "-q",
        "--quant",
        type=str,
        choices=list(_QUANT_OPTIONS.keys()),
        nargs="*",
        default=list(_QUANT_OPTIONS.keys()),
        help="quant config(s). bf16 = no quant, fp8_<group>_<scale> = quant.",
    )
    parser.add_argument(
        "--qweight",
        action="store_true",
        help="also run each config with optional q_weight=enabled.",
    )
    parser.add_argument(
        "--swa-mode",
        type=str,
        nargs="*",
        default=["direct", "paged"],
        choices=["direct", "paged"],
        help="SWA scatter addressing mode(s) to sweep.",
    )
    parser.add_argument(
        "--no-quant",
        action="store_true",
        help="bf16 only (ignore -q).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="random seed for data initialisation (default: 0).",
    )
    parser.add_argument(
        "--init",
        type=str,
        choices=list(_INIT_MODES),
        default="normal",
        help="data initialisation mode: normal, uniform, zero, constant (default: normal).",
    )
    parser.add_argument(
        "--init-scale",
        type=float,
        default=0.1,
        help="multiplicative scale for normal/uniform data (default: 0.1).",
    )
    parser.add_argument(
        "--init-val",
        type=float,
        default=1.0,
        help="fill value for constant mode (default: 1.0).",
    )
    args = parser.parse_args()

    # Smoke-test the advertised 4D cos/sin layout once before sweeping.
    test_flydsl_qk_norm_rope_quant_cos_sin_4d()

    quant_keys = ["bf16"] if args.no_quant else args.quant
    qweight_modes = [False, True] if args.qweight else [False]
    init_kw = {
        "seed": args.seed,
        "init_mode": args.init,
        "init_scale": args.init_scale,
        "init_val": args.init_val,
    }

    rows = []
    for key, qw_mode, H, D, T in itertools.product(
        quant_keys, qweight_modes, args.H, args.D, args.T
    ):
        quant, group_size, scale_dtype, _ = _QUANT_OPTIONS[key]
        rows.append(
            test_flydsl_qk_norm_rope_quant(
                T,
                H,
                D,
                args.RD,
                quant_group_size=group_size,
                scale_dtype=scale_dtype,
                q_weighted=qw_mode,
                quant=quant,
                **init_kw,
            )
        )
    print_json_table("flydsl_qk_norm_rope_quant summary", rows)

    # Separate arg signature -> its own table (merging would scatter NaNs).
    # The scatter is decode-only and bf16-only; keep T in the decode range.
    swa_rows = []
    for mode, H, D, T in itertools.product(
        args.swa_mode,
        args.H,
        args.D,
        # decode range: the scatter is decode-only, and the paged layout holds
        # _SWA_BS * (max_blocks-1) * block_size == 96 tokens (the last block is
        # reserved for the out-of-window sentinel).
        [t for t in args.T if 8 <= t <= 96] or [16, 64],
    ):
        swa_rows.append(test_flydsl_swa_write(T, H, D, args.RD, mode, **init_kw))
    print_json_table("flydsl_qk_norm_rope_quant fused SWA write summary", swa_rows)


if __name__ == "__main__":
    main()
