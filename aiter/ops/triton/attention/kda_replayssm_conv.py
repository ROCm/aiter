# SPDX-License-Identifier: MIT
"""gfx1250 Gluon: Kimi-K3's speculative-verify short convolution (width 4, SiLU, ``causal_conv1d_update`` with
``num_accepted_tokens``) fused into the CTA-per-(request, value head) ReplaySSM KDA kernel of
``aiter.ops.triton._gluon_kernels.gfx1250.attention.kda_replayssm`` (its helpers are reused unchanged).

With H == HV every CTA owns exactly the 3 x 128 conv channels behind its q, k and v heads, so the conv needs no
cross-CTA traffic: each tap of the window is one masked load whose row pointer picks the conv state (window positions
0..2 = the accepted history at ``num_accepted_tokens - 1``) or x (positions 3..), issued with the record / token
loads. The tap arithmetic, SiLU (rounded to bf16 as the unfused q / k / v are) and the token math run before the
64 KB checkpoint load is issued, so the tap tiles are dead before the checkpoint registers come live and the kernel
fits 128 VGPRs (8 waves an EU) without spills. The three taps the new conv state keeps are parked in LDS; after the
CTA's first barrier (every window read done -- the read and write lines can coincide) each thread writes its own
elements back as ``[hist[1], hist[2], x_0 .. x_{T-1}]`` at the write line. One token chunk: max_query_len <= 8.

Optionally (``f_a`` + ``f_b_weight``) the gate projection ``a = f_b_proj(f_a)`` runs in the kernel too, on the matrix
cores: the CTA's [128, F] weight slice and the step's f_a rows are issued first, staged through row-padded LDS
(alongside only the record tiles, so the LDS peak does not grow) and contracted with one WMMA chain, rounded to bf16
as the unfused GEMM output is. That removes ATOM's f_a copy and f_b_proj GEMM from the verify step."""

import functools
import os

import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

from aiter.ops.triton._gluon_kernels.gfx1250.attention.kda_replayssm import (
    _kda_cm_out,
    _kda_join8,
    _kda_split8,
    _kda_tok_append,
    _kda_tok_loop,
    _kda_tok_math,
    _kda_upd8,
)
from aiter.ops.triton.utils._triton.arch_info import get_arch
from aiter.ops.triton.utils.config_utils import load_config_json, resolve_config_dir


@gluon.jit
def _conv_tap(x, cs, bos, t, m_t, ch, mk, si, off, J: gl.constexpr,
              stride_x_tok: gl.constexpr, stride_cs_slot: gl.constexpr, stride_cs_dim: gl.constexpr,
              stride_cs_pos: gl.constexpr):
    """Window position t + J of the channels ch (raw, bf16): the conv state below 3, x from 3 on."""
    idx = t + J
    in_hist = idx < 3
    if stride_cs_dim == 1:
        # channels contiguous in both sources (the token-major conv state K3 serves): pick the row base once per
        # token row, so the 4 channels of a lane stay one contiguous (vector) load
        row = gl.where(in_hist, cs + si * stride_cs_slot + (off + idx) * stride_cs_pos,
                       x + (bos + idx - 3) * stride_x_tok)
        p = row[:, None] + ch[None, :]
        return gl.load(p, mask=m_t[:, None] & mk, other=0.0)
    else:
        # channel-strided conv state: two masked loads, so the x side stays one contiguous (vector) load per lane
        m = m_t[:, None] & mk
        v_h = gl.load(cs + si * stride_cs_slot + ch[None, :] * stride_cs_dim + (off + idx)[:, None] * stride_cs_pos,
                      mask=m & in_hist[:, None], other=0.0)
        v_x = gl.load(x + (bos + idx - 3)[:, None] * stride_x_tok + ch[None, :], mask=m & (idx >= 3)[:, None],
                      other=0.0)
        return gl.where(in_hist[:, None], v_h, v_x)


@gluon.jit
def _conv_load(x, cs, bos, t0, T, ch0, si, off, conv_ok, o_t, o_kt, K: gl.constexpr,
               stride_x_tok: gl.constexpr, stride_cs_slot: gl.constexpr, stride_cs_dim: gl.constexpr,
               stride_cs_pos: gl.constexpr):
    t = t0 + o_t
    m_t = (t < T) & conv_ok
    ch = ch0 + o_kt
    mk = (o_kt < K)[None, :]
    v0 = _conv_tap(x, cs, bos, t, m_t, ch, mk, si, off, 0, stride_x_tok, stride_cs_slot, stride_cs_dim, stride_cs_pos)
    v1 = _conv_tap(x, cs, bos, t, m_t, ch, mk, si, off, 1, stride_x_tok, stride_cs_slot, stride_cs_dim, stride_cs_pos)
    v2 = _conv_tap(x, cs, bos, t, m_t, ch, mk, si, off, 2, stride_x_tok, stride_cs_slot, stride_cs_dim, stride_cs_pos)
    v3 = _conv_tap(x, cs, bos, t, m_t, ch, mk, si, off, 3, stride_x_tok, stride_cs_slot, stride_cs_dim, stride_cs_pos)
    return v0, v1, v2, v3


@gluon.jit
def _conv_math(v0, v1, v2, v3, cw, ch0, o_kt):
    """y = SiLU(sum_j w[:, j] tap_j), rounded to bf16 as the unfused q / k / v tensors are."""
    ch = ch0 + o_kt
    acc = v0.to(gl.float32) * gl.load(cw + ch * 4 + 0)[None, :]
    acc = gl.fma(v1.to(gl.float32), gl.load(cw + ch * 4 + 1)[None, :], acc)
    acc = gl.fma(v2.to(gl.float32), gl.load(cw + ch * 4 + 2)[None, :], acc)
    acc = gl.fma(v3.to(gl.float32), gl.load(cw + ch * 4 + 3)[None, :], acc)
    y = acc / (1.0 + gl.exp(-acc))
    return y.to(gl.bfloat16).to(gl.float32)


@gluon.jit
def _conv_writeback(cs, so, t0, T, ch0, conv_ok, o_t, o_kt, v1, v2, v3, K: gl.constexpr,
                    stride_cs_slot: gl.constexpr, stride_cs_dim: gl.constexpr, stride_cs_pos: gl.constexpr):
    """New window at line so: positions 0, 1 = the history's last two (token 0's taps 1, 2), 2 + t = x_t (tap 3)."""
    t = t0 + o_t
    m_t = (t < T) & conv_ok & (so >= 0)
    ch = ch0 + o_kt
    mk = (o_kt < K)[None, :]
    base = cs + so * stride_cs_slot + ch[None, :] * stride_cs_dim
    gl.store(base + (2 + t)[:, None] * stride_cs_pos, v3, mask=m_t[:, None] & mk)
    m0 = (m_t & (t == 0))[:, None] & mk
    zt = (t * 0)[:, None]
    gl.store(base + zt * stride_cs_pos, v1, mask=m0)
    gl.store(base + (zt + 1) * stride_cs_pos, v2, mask=m0)


@gluon.jit
def _tok_load_conv(x, cs, a, b, bos, t0, T, i_h, i_hv, si, off, conv_ok, o_t, o_kt,
                   H: gl.constexpr, HV: gl.constexpr, K: gl.constexpr,
                   stride_a_token: gl.constexpr, stride_b_token: gl.constexpr, stride_b_head: gl.constexpr,
                   stride_x_tok: gl.constexpr, stride_cs_slot: gl.constexpr, stride_cs_dim: gl.constexpr,
                   stride_cs_pos: gl.constexpr, HAS_FB: gl.constexpr):
    """The raw loads of one chunk: q / k / v conv taps (x and conv state), a (unless f_b_proj is fused), b."""
    KD: gl.constexpr = H * K
    q0, q1, q2, q3 = _conv_load(x, cs, bos, t0, T, i_h * K, si, off, conv_ok, o_t, o_kt, K,
                                stride_x_tok, stride_cs_slot, stride_cs_dim, stride_cs_pos)
    k0, k1, k2, k3 = _conv_load(x, cs, bos, t0, T, KD + i_h * K, si, off, conv_ok, o_t, o_kt, K,
                                stride_x_tok, stride_cs_slot, stride_cs_dim, stride_cs_pos)
    w0, w1, w2, w3 = _conv_load(x, cs, bos, t0, T, 2 * KD + i_hv * K, si, off, conv_ok, o_t, o_kt, K,
                                stride_x_tok, stride_cs_slot, stride_cs_dim, stride_cs_pos)
    m_t = (t0 + o_t) < T
    tok = bos + t0 + o_t
    m_tk = m_t[:, None] & (o_kt < K)[None, :]
    if HAS_FB:
        r_x = q0  # placeholder: the gate comes from _gate_proj
    else:
        r_x = gl.load(a + tok[:, None] * stride_a_token + i_hv * K + o_kt[None, :], mask=m_tk, other=0.0)
    r_b = gl.load(b + tok * stride_b_token + i_hv * stride_b_head, mask=m_t, other=0.0)
    return q0, q1, q2, q3, k0, k1, k2, k3, w0, w1, w2, w3, r_x, r_b


@gluon.jit
def _gate_issue(fa, wfb, bos, T, i_hv, K: gl.constexpr, F: gl.constexpr, stride_fa_tok: gl.constexpr,
                WARPS: gl.constexpr):
    """The f_b_proj operands, issued first thing: W's [128, F] slice of this head and the step's f_a rows (padded to
    16), rows of F at 16 B a lane."""
    LG: gl.constexpr = gl.BlockedLayout([1, 8], [32 * 8 // F, F // 8], [WARPS, 1], [1, 0])
    o_r = gl.arange(0, K, layout=gl.SliceLayout(1, LG))
    o_f = gl.arange(0, F, layout=gl.SliceLayout(0, LG))
    t_w = gl.load(wfb + (i_hv * K + o_r)[:, None] * F + o_f[None, :])
    o_m = gl.arange(0, 16, layout=gl.SliceLayout(1, LG))
    t_a = gl.load(fa + (bos + o_m)[:, None] * stride_fa_tok + o_f[None, :],
                  mask=(o_m < T)[:, None] & (o_f < F)[None, :], other=0.0)
    return t_w, t_a


@gluon.jit
def _gate_mma(t_w, t_a, BT: gl.constexpr, K: gl.constexpr, F: gl.constexpr, WARPS: gl.constexpr, LT: gl.constexpr,
              SH: gl.constexpr):
    """f_b_proj of this CTA's head on the matrix cores: bf16(f_a[t, :] @ W[i_hv * K + k, :]^T), the eight warps each
    on 16 of the head's 128 channels. Operands go through row-padded LDS tiles (conflict-free 16 B operand reads;
    only the record tiles are live alongside, so the LDS peak stays the kernel's); the result comes back in LT
    through a 4 KB tile."""
    gl.static_assert(WARPS == 8)
    gl.static_assert(K == 128)
    WL: gl.constexpr = gl.amd.AMDWMMALayout(version=3, transposed=True, warp_bases=[[0, 1], [0, 2], [0, 4]],
                                            instr_shape=[16, 16, 32])  # warp bases in 16 x 16 tiles
    OA: gl.constexpr = gl.DotOperandLayout(operand_index=0, parent=WL, k_width=8)
    OB: gl.constexpr = gl.DotOperandLayout(operand_index=1, parent=WL, k_width=8)
    PW: gl.constexpr = gl.PaddedSharedLayout.with_identity_for([[F, 8]], [K, F], [1, 0])
    PA: gl.constexpr = gl.PaddedSharedLayout.with_identity_for([[F, 8]], [16, F], [1, 0])
    s_w = gl.allocate_shared_memory(gl.bfloat16, [K, F], PW, t_w)
    s_a = gl.allocate_shared_memory(gl.bfloat16, [16, F], PA, t_a)
    gl.barrier()
    acc = gl.amd.gfx1250.wmma(s_a.load(OA), s_w.permute((1, 0)).load(OB), gl.zeros([16, K], gl.float32, WL))
    s_g = gl.allocate_shared_memory(gl.bfloat16, [16, K], SH, acc.to(gl.bfloat16))
    gl.barrier()
    return s_g.slice(0, BT).load(LT)

@gluon.jit
def _replayssm_kda_conv_fwd_cta_kernel(
    x, cs, cw, cs_idx, cs_idx_in, n_acc,
    a, b, A_log, dt_bias, o, ckpt, buf_k, buf_u, buf_g, write_pos, slot_idx, cu_seqlens, fa, wfb,
    scale, LOWER_BOUND, T_MAX, CAP,
    H: gl.constexpr, HV: gl.constexpr, K: gl.constexpr, V: gl.constexpr, BH: gl.constexpr, BT: gl.constexpr,
    stride_ckpt_slot: gl.constexpr,
    stride_bufk_slot: gl.constexpr, stride_bufk_pos: gl.constexpr, stride_bufk_hv: gl.constexpr,
    stride_bufu_slot: gl.constexpr, stride_bufu_pos: gl.constexpr, stride_bufu_hv: gl.constexpr,
    stride_bufg_slot: gl.constexpr, stride_bufg_pos: gl.constexpr, stride_bufg_hv: gl.constexpr,
    stride_a_token: gl.constexpr, stride_b_token: gl.constexpr, stride_b_head: gl.constexpr,
    stride_x_tok: gl.constexpr, stride_cs_slot: gl.constexpr, stride_cs_dim: gl.constexpr,
    stride_cs_pos: gl.constexpr, stride_csi: gl.constexpr, stride_csi_in: gl.constexpr,
    USE_QK_L2NORM_IN_KERNEL: gl.constexpr, USE_LOWER_BOUND: gl.constexpr,
    TPW_K: gl.constexpr, WARPS: gl.constexpr, HAS_FB: gl.constexpr, F: gl.constexpr, stride_fa_tok: gl.constexpr,
):
    TPW_V: gl.constexpr = 32 // TPW_K
    RPT: gl.constexpr = V // (TPW_V * WARPS)
    LS: gl.constexpr = gl.BlockedLayout([RPT, 1], [TPW_V, TPW_K], [WARPS, 1], [1, 0])
    LK: gl.constexpr = gl.SliceLayout(0, LS)
    LV: gl.constexpr = gl.SliceLayout(1, LS)
    KB: gl.constexpr = TPW_K
    NB: gl.constexpr = K // TPW_K
    gl.static_assert(TPW_K == 16)
    gl.static_assert(NB == 8)
    gl.static_assert(H == HV)
    L3: gl.constexpr = gl.BlockedLayout([RPT, 1, NB], [TPW_V, TPW_K, 1], [WARPS, 1, 1], [2, 1, 0])
    L4: gl.constexpr = gl.BlockedLayout([RPT, 2, 1, 4], [TPW_V, 1, TPW_K, 1], [WARPS, 1, 1, 1], [3, 1, 2, 0])
    LK2: gl.constexpr = gl.SliceLayout(0, L4)
    SH2: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, [2, 1, 0])
    LR: gl.constexpr = gl.BlockedLayout([2, 4], [8, 4], [1, WARPS], [1, 0])
    LT: gl.constexpr = gl.BlockedLayout([1, K // 32], [1, 32], [WARPS, 1], [1, 0])
    SH: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, [1, 0])
    SH1: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, [0])

    i_nh = gl.program_id(0)
    i_n = i_nh // HV
    i_hv = i_nh % HV
    i_h = i_hv // (HV // H)

    bos = gl.load(cu_seqlens + i_n).to(gl.int64)
    eos = gl.load(cu_seqlens + i_n + 1).to(gl.int64)
    slot = gl.load(slot_idx + i_n).to(gl.int64)
    wp = gl.load(write_pos + gl.maximum(slot, 0)).to(gl.int32)
    si = gl.load(cs_idx_in + i_n * stride_csi_in).to(gl.int64)
    so = gl.load(cs_idx + i_n * stride_csi).to(gl.int64)
    off = (gl.load(n_acc + i_n) - 1).to(gl.int64)
    T = (eos - bos).to(gl.int32)
    if T == 0:
        return
    if slot < 0:
        return
    conv_ok = si >= 0
    o_sp = gl.arange(0, 32 * WARPS, layout=gl.BlockedLayout([1], [32], [WARPS], [0]))
    gl.inline_asm_elementwise("s_setprio 2\nv_mov_b32 $0, $1", "=v,v", [o_sp], dtype=gl.int32, is_pure=False, pack=1)
    h = gl.maximum(wp, 0)
    do_flush = h + 2 * T_MAX > CAP
    base = gl.where(do_flush, 0, h).to(gl.int64)

    # ---- 0. records, chunk-0 conv taps / a / b, then the ckpt ---------------------------------------
    if HAS_FB:  # the f_b_proj operands lead: their chain (LDS, WMMA, LDS) is the longest
        t_w, t_a = _gate_issue(fa, wfb, bos, T, i_hv, K, F, stride_fa_tok, WARPS)
    o_hr = gl.arange(0, BH, layout=gl.SliceLayout(1, LR))
    o_kr = gl.arange(0, K, layout=gl.SliceLayout(0, LR))
    m_r = (o_hr < h)[:, None] & (o_kr < K)[None, :]
    t_g = gl.load(buf_g + slot * stride_bufg_slot + o_hr[:, None] * stride_bufg_pos + i_hv * stride_bufg_hv
                  + o_kr[None, :], mask=m_r, other=0.0).to(gl.float32)
    t_k = gl.load(buf_k + slot * stride_bufk_slot + o_hr[:, None] * stride_bufk_pos + i_hv * stride_bufk_hv
                  + o_kr[None, :], mask=m_r, other=0.0).to(gl.float32)
    LU: gl.constexpr = gl.BlockedLayout([1, 8], [32 * 8 // V, V // 8], [WARPS, 1], [1, 0])
    o_hu = gl.arange(0, BH, layout=gl.SliceLayout(1, LU))
    o_vr = gl.arange(0, V, layout=gl.SliceLayout(0, LU))
    t_u = gl.load(buf_u + slot * stride_bufu_slot + o_hu[:, None] * stride_bufu_pos + i_hv * stride_bufu_hv
                  + o_vr[None, :], mask=(o_hu < h)[:, None] & (o_vr < V)[None, :], other=0.0)
    o_t = gl.arange(0, BT, layout=gl.SliceLayout(1, LT))
    o_kt = gl.arange(0, K, layout=gl.SliceLayout(0, LT))
    b_Alog = gl.load(A_log + i_hv).to(gl.float32)
    b_dtb = gl.load(dt_bias + i_hv * K + o_kt).to(gl.float32)
    q0, q1, q2, q3, k0, k1, k2, k3, w0, w1, w2, w3, r_x, r_b = _tok_load_conv(
        x, cs, a, b, bos, 0, T, i_h, i_hv, si, off, conv_ok, o_t, o_kt, H, HV, K,
        stride_a_token, stride_b_token, stride_b_head, stride_x_tok, stride_cs_slot, stride_cs_dim, stride_cs_pos,
        HAS_FB)
    o_v4 = gl.arange(0, V, layout=gl.SliceLayout(1, gl.SliceLayout(2, gl.SliceLayout(3, L4))))
    o_h4 = gl.arange(0, 2, layout=gl.SliceLayout(0, gl.SliceLayout(2, gl.SliceLayout(3, L4))))
    o_j4 = gl.arange(0, KB, layout=gl.SliceLayout(0, gl.SliceLayout(1, gl.SliceLayout(3, L4))))
    o_e4 = gl.arange(0, 4, layout=gl.SliceLayout(0, gl.SliceLayout(1, gl.SliceLayout(2, L4))))
    b_ctot = gl.sum(t_g, axis=0)
    t_kw = t_k * gl.exp(b_ctot[None, :] - gl.cumsum(t_g, axis=0))
    s_kw = gl.allocate_shared_memory(gl.float32, [BH, K], SH, t_kw)
    s_u = gl.allocate_shared_memory(t_u.dtype, [BH, V], SH, t_u)
    s_dec = gl.allocate_shared_memory(gl.float32, [K], SH1, gl.exp(b_ctot))
    if HAS_FB:  # f_b_proj in-kernel while the taps are in flight
        r_x = _gate_mma(t_w, t_a, BT, K, F, WARPS, LT, SH)
    s_eg = gl.allocate_shared_memory(gl.float32, [BT, K], SH)
    s_kn = gl.allocate_shared_memory(gl.float32, [BT, K], SH)
    s_qn = gl.allocate_shared_memory(gl.float32, [BT, K], SH)
    s_v = gl.allocate_shared_memory(gl.float32, [BT, V], SH)
    s_beta = gl.allocate_shared_memory(gl.float32, [BT, V], SH)
    KD: gl.constexpr = H * K
    # conv + token math before the 64 KB ckpt load is issued, straight from the tap registers; the three taps the
    # new conv state keeps (token 0's taps 1, 2 = the history's last two, tap 3 = x_t) are parked in LDS -- every
    # thread re-reads only its own elements after the main barrier, so the park needs no barrier of its own
    p_q1 = gl.allocate_shared_memory(q1.dtype, [BT, K], SH, q1)
    p_q2 = gl.allocate_shared_memory(q1.dtype, [BT, K], SH, q2)
    p_q3 = gl.allocate_shared_memory(q1.dtype, [BT, K], SH, q3)
    p_k1 = gl.allocate_shared_memory(q1.dtype, [BT, K], SH, k1)
    p_k2 = gl.allocate_shared_memory(q1.dtype, [BT, K], SH, k2)
    p_k3 = gl.allocate_shared_memory(q1.dtype, [BT, K], SH, k3)
    p_w1 = gl.allocate_shared_memory(q1.dtype, [BT, K], SH, w1)
    p_w2 = gl.allocate_shared_memory(q1.dtype, [BT, K], SH, w2)
    p_w3 = gl.allocate_shared_memory(q1.dtype, [BT, K], SH, w3)
    y_q = _conv_math(q0, q1, q2, q3, cw, i_h * K, o_kt)
    y_k = _conv_math(k0, k1, k2, k3, cw, KD + i_h * K, o_kt)
    y_v = _conv_math(w0, w1, w2, w3, cw, 2 * KD + i_hv * K, o_kt)
    a_k, a_g = _kda_tok_math(y_q, y_k, y_v, r_x, r_b, b_dtb, b_Alog, scale, LOWER_BOUND, s_eg, s_kn, s_qn, s_v,
                             s_beta, BT, V, LT, USE_QK_L2NORM_IN_KERNEL, USE_LOWER_BOUND)
    p_ckpt_hv = (ckpt + slot * stride_ckpt_slot + i_hv * V * K + o_v4[:, None, None, None] * K
                 + o_h4[None, :, None, None] * (K // 2) + o_j4[None, None, :, None] * 4 + o_e4[None, None, None, :])
    b_ck = gl.load(p_ckpt_hv)
    o_v = gl.arange(0, V, layout=LV)
    o_k0 = gl.arange(0, KB, layout=LK) * 0
    m_k0 = (gl.arange(0, KB, layout=LK) == 0)[None, :] & (o_v >= 0)[:, None]
    o_j = gl.arange(0, KB, layout=LK)
    m_b1 = gl.where((o_j // 2) % 2 == 1, -1, 0)[None, :] + gl.zeros([V, KB], gl.int32, LS)
    o_vo = (o_v - o_v % RPT)[:, None] + (o_j // 2)[None, :]
    m_o = ((o_v % RPT) == RPT - 1)[:, None] & ((o_j % 2) == 0)[None, :]
    gl.barrier()
    gl.inline_asm_elementwise("s_setprio 0\nv_mov_b32 $0, $1", "=v,v", [o_sp], dtype=gl.int32, is_pure=False, pack=1)
    # every window read of this CTA is done: the new conv state (read and write lines can coincide)
    _conv_writeback(cs, so, 0, T, i_h * K, conv_ok, o_t, o_kt, p_q1.load(LT), p_q2.load(LT),
                    p_q3.load(LT), K, stride_cs_slot, stride_cs_dim, stride_cs_pos)
    _conv_writeback(cs, so, 0, T, KD + i_h * K, conv_ok, o_t, o_kt, p_k1.load(LT),
                    p_k2.load(LT), p_k3.load(LT), K, stride_cs_slot, stride_cs_dim,
                    stride_cs_pos)
    _conv_writeback(cs, so, 0, T, 2 * KD + i_hv * K, conv_ok, o_t, o_kt, p_w1.load(LT),
                    p_w2.load(LT), p_w3.load(LT), K, stride_cs_slot, stride_cs_dim,
                    stride_cs_pos)
    _kda_tok_append(buf_k, buf_g, a_k, a_g, slot, base, 0, T, i_hv, o_t, o_kt, stride_bufk_slot, stride_bufk_pos,
                    stride_bufk_hv, stride_bufg_slot, stride_bufg_pos, stride_bufg_hv, K)
    # ---- 1. rebuild S_h ---------------------------------------------------------------------------
    S0, S1, S2, S3, S4, S5, S6, S7 = _kda_split8(b_ck.to(gl.float32), LS)
    d0, d1, d2, d3, d4, d5, d6, d7 = _kda_split8(s_dec.reinterpret(gl.float32, [2, KB, 4], SH2).load(LK2), LK)
    S0 = S0 * d0[None, :]
    S1 = S1 * d1[None, :]
    S2 = S2 * d2[None, :]
    S3 = S3 * d3[None, :]
    S4 = S4 * d4[None, :]
    S5 = S5 * d5[None, :]
    S6 = S6 * d6[None, :]
    S7 = S7 * d7[None, :]
    n_rep = gl.minimum(h, BH)
    for j in range(n_rep):
        e0, e1, e2, e3, e4, e5, e6, e7 = _kda_split8(
            s_kw.reinterpret(gl.float32, [BH, 2, KB, 4], SH2).index(j).load(LK2), LK)
        b_uj = s_u.reinterpret(t_u.dtype, [BH, V], SH1).index(j).load(LV).to(gl.float32)
        S0, S1, S2, S3, S4, S5, S6, S7 = _kda_upd8(S0, S1, S2, S3, S4, S5, S6, S7, b_uj, e0, e1, e2, e3, e4, e5,
                                                   e6, e7)
    if do_flush:
        gl.store(p_ckpt_hv, _kda_cm_out(_kda_join8(S0, S1, S2, S3, S4, S5, S6, S7, L3), L4).to(
            ckpt.dtype.element_ty))
    # ---- 2. this step's tokens, BT at a time ------------------------------------------------------
    S0, S1, S2, S3, S4, S5, S6, S7 = _kda_tok_loop(
        S0, S1, S2, S3, S4, S5, S6, S7, s_eg, s_kn, s_qn, s_v, s_beta, o, buf_u, bos, 0, gl.minimum(T, BT), slot,
        base, i_hv, o_v, o_k0, m_k0, m_b1, o_vo, m_o, stride_bufu_slot, stride_bufu_pos, stride_bufu_hv, HV, V, BT,
        KB, NB, LK, LV, LK2, SH1, SH2)


@functools.lru_cache(maxsize=8)
def _config(arch, cu_num):
    path = resolve_config_dir("attention", "kda_replayssm", backend="gluon")
    return load_config_json(f"{path}/DEFAULT.json")


def replayssm_kda_conv_gluon(
    x, conv_state, conv_weight, conv_state_indices, num_accepted_tokens, a, b, A_log, dt_bias, ckpt, buf_k, buf_u,
    buf_g, write_pos, slot_idx, cu_seqlens, max_query_len, num_heads, conv_state_indices_in=None, o=None,
    scale=None, use_qk_l2norm_in_kernel=False, lower_bound=None, f_a=None, f_b_weight=None,
):
    """Same call as ``aiter.ops.kda_replayssm.kda_replayssm_decode_conv`` (H == HV == num_heads, K = V = 128): the
    width-4 speculative conv + SiLU of x ``[T, 3 * H * 128]`` (read at ``conv_state_indices_in`` /
    ``num_accepted_tokens - 1``, rewritten at ``conv_state_indices``) fused with the ReplaySSM KDA step. Returns
    ``[T, HV, 128]`` (``o`` if given). With ``f_a`` ``[T, F]`` (rows dense, e.g. a column view of the in-proj output)
    and ``f_b_weight`` ``[H * 128, F]`` (bf16, contiguous) the gate ``a = f_b_proj(f_a)`` is computed in-kernel and ``a`` may
    be None."""
    arch = get_arch()
    if arch not in ("gfx1250", "gfx1250-strict"):
        raise NotImplementedError("gfx1250 only")
    K = V = 128
    H = HV = int(num_heads)
    T_tot = x.shape[0]
    if x.dim() != 2 or x.shape[1] != 3 * H * K or x.stride(-1) != 1:
        raise ValueError("x must be [T, 3 * H * 128] with a contiguous last dim")
    if conv_state.dim() != 3 or conv_state.shape[1] != 3 * H * K or conv_state.shape[2] < 2 + max_query_len:
        raise ValueError("conv_state must be [lines, 3 * H * 128, >= 2 + max_query_len]")
    if tuple(conv_weight.shape) != (3 * H * K, 4) or not conv_weight.is_contiguous():
        raise ValueError("conv_weight must be a contiguous [3 * H * 128, 4]")
    if not 1 <= max_query_len <= 8:
        raise NotImplementedError("one token chunk: max_query_len <= 8")
    if conv_state_indices_in is None:
        conv_state_indices_in = conv_state_indices
    cap = buf_k.shape[2]
    if cap < 2 * max_query_len:
        raise ValueError("record capacity must be at least 2*max_query_len")
    BH = max(16, triton.next_power_of_2(cap - max_query_len))
    if BH > 64:
        raise NotImplementedError("history tiles up to 64 rows")
    if b.dim() == 3:
        b3 = b
    else:
        b3 = b.unsqueeze(0)
    has_fb = f_b_weight is not None
    if has_fb:
        F = f_a.shape[-1]
        if (f_a.dim() != 2 or f_a.shape[0] != T_tot or f_a.stride(-1) != 1 or f_a.dtype != torch.bfloat16
                or tuple(f_b_weight.shape) != (H * K, F) or not f_b_weight.is_contiguous()
                or f_b_weight.dtype != torch.bfloat16 or F % 64 or F > 256):
            raise ValueError("f_a must be [T, F] rows-dense bf16 and f_b_weight a contiguous bf16 [H * 128, F]")
        a, stride_fa = f_a, f_a.stride(0)
        stride_a = 0
    else:
        F, stride_fa = 64, 0
        if a.dim() == 3:
            a = a.unsqueeze(0)
        a = a if a.stride(-1) == 1 and a.stride(-2) == K else a.contiguous()
        stride_a = a.stride(1)
    out = x.new_empty(T_tot, HV, V) if o is None else o
    out4 = out.view(1, out.shape[0], HV, V)
    n = slot_idx.numel()
    if n == 0:
        return out
    cfg = _config(arch, torch.cuda.get_device_properties(x.device).multi_processor_count)
    BT = triton.next_power_of_2(max_query_len)
    _replayssm_kda_conv_fwd_cta_kernel[(n * HV,)](
        x, conv_state, conv_weight, conv_state_indices, conv_state_indices_in, num_accepted_tokens,
        a, b3, A_log, dt_bias, out4, ckpt, buf_k, buf_u, buf_g, write_pos, slot_idx, cu_seqlens,
        f_a if has_fb else a, f_b_weight if has_fb else a,
        K ** -0.5 if scale is None else scale, 0.0 if lower_bound is None else lower_bound, max_query_len, cap,
        H=H, HV=HV, K=K, V=V, BH=BH, BT=BT,
        stride_ckpt_slot=ckpt.stride(0),
        stride_bufk_slot=buf_k.stride(0), stride_bufk_pos=buf_k.stride(2), stride_bufk_hv=buf_k.stride(1),
        stride_bufu_slot=buf_u.stride(0), stride_bufu_pos=buf_u.stride(2), stride_bufu_hv=buf_u.stride(1),
        stride_bufg_slot=buf_g.stride(0), stride_bufg_pos=buf_g.stride(2), stride_bufg_hv=buf_g.stride(1),
        stride_a_token=stride_a, stride_b_token=b3.stride(1), stride_b_head=b3.stride(2),
        stride_x_tok=x.stride(0), stride_cs_slot=conv_state.stride(0), stride_cs_dim=conv_state.stride(1),
        stride_cs_pos=conv_state.stride(2), stride_csi=conv_state_indices.stride(0),
        stride_csi_in=conv_state_indices_in.stride(0),
        USE_QK_L2NORM_IN_KERNEL=use_qk_l2norm_in_kernel, USE_LOWER_BOUND=lower_bound is not None,
        TPW_K=cfg["TPW_K"], WARPS=cfg["num_warps"], HAS_FB=has_fb, F=F, stride_fa_tok=stride_fa,
        num_warps=cfg["num_warps"], waves_per_eu=int(os.environ.get("KDA_CONV_GLUON_WPE", cfg["waves_per_eu"])),
    )
    return out

def replayssm_kda_conv_gluon_supported(x, conv_state, conv_weight, a, b, ckpt, buf_k, buf_u, buf_g, max_query_len, o):
    """Whether :func:`replayssm_kda_conv_gluon` covers the call (gate ``a`` precomputed). Never raises: False sends the
    caller to its other kernels."""
    try:
        if get_arch() not in ("gfx1250", "gfx1250-strict"):
            return False
        K = V = 128
        if o is None or o.dim() != 3 or o.shape[-1] != V or not o.is_contiguous():
            return False
        H = HV = o.shape[-2]
        if x.dim() != 2 or x.shape[1] != 3 * H * K or x.stride(-1) != 1 or o.shape[0] < x.shape[0]:
            return False
        if conv_state.dim() != 3 or conv_state.shape[1] != 3 * H * K or conv_state.shape[2] < 2 + max_query_len:
            return False
        if conv_state.dtype != x.dtype:
            return False
        if tuple(conv_weight.shape) != (3 * H * K, 4) or not conv_weight.is_contiguous():
            return False
        if not 1 <= max_query_len <= 8:
            return False
        if ckpt.dim() != 4 or tuple(ckpt.shape[1:]) != (HV, V, K) or ckpt.stride(-1) != 1 or ckpt.stride(-2) != K \
                or ckpt.stride(1) != V * K:
            return False
        cap = buf_k.shape[2]
        if cap < 2 * max_query_len or max(16, triton.next_power_of_2(cap - max_query_len)) > 64:
            return False
        for buf, last in ((buf_k, K), (buf_u, V), (buf_g, K)):
            if buf.dim() != 4 or buf.shape[1] != HV or buf.shape[2] != cap or buf.shape[3] != last \
                    or buf.stride(-1) != 1:
                return False
        if a is None or a.shape[-2:] != (HV, K) or a.dim() not in (3, 4):
            return False
        if b.dim() not in (2, 3) or b.shape[-1] != HV:
            return False
        cfg = _config(get_arch(), torch.cuda.get_device_properties(x.device).multi_processor_count)
        return cfg["num_warps"] == 8 and cfg["TPW_K"] == 16
    except Exception:  # noqa: BLE001 -- a support probe
        return False
