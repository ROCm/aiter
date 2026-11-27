# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""MHA Forward Prefill kernel — ``m32x8`` design, gfx1250 (MI400 / mi450).

A clean FlyDSL kernel written in the high-level layout-algebra style
(tiled copy / tiled MMA + ``SharedAllocator``).

``m32x8`` names the threadgroup shape: **8 waves per threadgroup**, each wave
owning a **32-row** Q span (2 adjacent 16-row WMMA tiles). gfx1250 runs wave32,
so a threadgroup is ``8 * 32 = 256`` threads and ``BLOCK_M = 32 * 8 = 256`` Q
rows. (The leading ``32`` is per-wave Q rows; ``16`` is the WMMA M dimension.)

Layout support — two device kernels over one shared compute core (option B):
  - ``kn_fmha_fwd_prefill_a8w8_m32x8_thd``  — varlen THD, driven by ``cu_seqlens``.
  - ``kn_fmha_fwd_prefill_a8w8_m32x8_bshd`` — batched BSHD, uniform ``seq_len`` scalar
    (no ``cu_seqlens`` tensors → nothing transient to bake into a CUDA graph).
Both resolve their per-workgroup base offsets + sequence bounds, then call the
layout-agnostic ``_core_attention`` helper.

Scope — v1 (this file is intentionally config-agnostic in its name):
  - ``qk_hdim in {64, 128, 192, 256}`` (D_qk), ``v_hdim in {64, 128, 256}`` (D_v); they
    are independent. ``n_block`` is picked by ``pick_n_block``. v_hdim 256 runs at
    BLOCK_M 128 only: the BLOCK_M 256 O tile does not fit its LDS staging chunk
  - dtype: fp8/bf8 for Q/K/V (independently), bf16 for O
  - grouped-query attention (GQA): ``gqa = nheads_q // nheads_k``
  - causal and non-causal

``qk_hdim``, ``v_hdim`` and the dtype are compile-time (build-time) parameters
captured by the builder closure, so they never appear in the file name and can
be generalized later without changing the runtime kernel signatures.

Target: gfx1250, wave32, 8 waves per threadgroup (256 threads).
"""

import functools
import math
import os
from enum import IntEnum

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl.expr import arith as _fx_arith
from flydsl.expr import const_expr, gpu, rocdl
from flydsl.expr.rocdl import tdm_ops
from flydsl.expr.typing import T
from flydsl.expr.utils.arith import _to_raw as _raw

from aiter.jit.utils.chip_info import get_cu_num, get_lds_capacity_bytes
from aiter.ops.flydsl.kernels import buffer_ops

from ..kernels_common import LOG2E
from ..tensor_shim import _run_compiled

# Q/K/V staging managers (own their LDS swizzles + async copy schedules). They are
# self-contained: this kernel maintains its own arch constants below and passes the
# config each manager needs through its constructor.
#
# Single source of truth for gfx1250 Expert Scheduling Mode 2 (DEP_MODE=2). Lives
# in fmha_buffer_managers. Under mode 2 the LLVM setreg (via the
# amdgpu-expert-scheduling-mode hint, set in _ensure_*_kernel) makes LLVM insert all
# depctr covers itself for the plain intrinsics the kernel emits.
from .fmha_buffer_managers import (
    ENABLE_SCHED_MODE2,
    KManagerPaged,
    KManagerV1,
    KManagerV2,
    ProducerCtx,
    QManagerV1,
    QManagerV2,
    VManagerPaged,
    VManagerV1,
    VManagerV2,
    _ir,
)
from .fmha_utils import (
    BLOCK_SIZE,
    DTYPE_MAP,
    LDS_CHUNK_BYTES,
    LDS_QO_BYTES,
    NUM_Q_TILES_CHOICES,
    NUM_WAVES,
    WAVE_SIZE,
    WMMA_M,
    PagedSpec,
    _alloc_lds,
    _bare_barrier,
    _block_m,
    _concat,
    _ContiguousKVPolicy,
    _copy_fence,
    _copy_wait,
    _epi_state,
    _is_paged_layout_supported,
    _keepalive,
    _kv_drain_depths,
    _kv_head_idx,
    _lane_id,
    _load_seqlen_pair,
    _load_sink_logit,
    _m_tile_idx,
    _num_heads_kv,
    _o_manager,
    _o_pack,
    _o_unpack,
    _packed_tile_indices,
    _PagedKVPolicy,
    _pick_num_q_tiles,
    _ring_drive,
    _ring_num_prefetch,
    _tree_reduce_multi,
    _warp_id,
    _wg_kv_span,
    _zero_slot,
    exp2_fastmath,
    fadd_fastmath,
    fadd_no_reassoc,
    fmax_fastmath,
    fmul_fastmath,
    fsub_fastmath,
    s_buffer_load,
    schedule_bank,
    supports_gqa_ratio,
    virtual_block,
)

# ============================================================================
# Threadgroup / arch constants
# ============================================================================


# "m32x8": each wave owns WMMA_ROW_PER_WAVE adjacent 16-row (WMMA M) Q sub-tiles →
# BLOCK_M = 16 * 2 * 8 = 256 Q rows per threadgroup. Each wave's 2 tiles are
# contiguous: warp i owns rows [i*32, i*32+32) = tiles 2i, 2i+1.
WMMA_N = 16  # kv rows per WMMA tile (the S^T=K@Q^T output's n_block-direction axis)
WMMA_K = 32  # WMMA contraction depth (bf16 v_wmma_f32_16x16x32); d-tile width
# 8-bit WMMA contracts twice as deep for the same 8-VGPR operand: 16x16x64 carries
# 32 B/lane of fp8 where 16x16x32 carries 32 B/lane of bf16 (probe-confirmed on HW).
WMMA_K_B8 = 64
WMMA_ROW_PER_WAVE = 2  # Q WMMA tiles per wave (the "x2" step from m16x8 to m32x8)
BLOCK_M = WMMA_M * WMMA_ROW_PER_WAVE * NUM_WAVES  # 256
# 1 => BLOCK_M 128 ("m16x8"), 2 => BLOCK_M 256. Chosen per call by _pick_num_q_tiles.


# A wave owns WMMA_M * num_q_tiles contiguous packed (seq, q_head_in_group) rows, and Q/O
# describe that tile with ONE rectangular TDM box. That only works when a head group is
# never split across waves: either a wave holds whole groups, or it sits inside one.


# Causal work grows with the M index, so dispatch the heavy tiles first.


class WarpType(IntEnum):
    """Warp-specialization role (compile-time). gfx1250 pairs wave i with wave i+4 on
    one SIMD; the low half (waves 0..3) and high half (waves 4..7) run different
    main-loop preamble orderings so one wave drives memory while its SIMD-mate computes.

    SIMD parity (wave w -> SIMD w%4) is NOT a role: the swapped KV-half read order
    carries it as a runtime term instead, so the body stays 2-way.
    """

    LO = 0
    HI = 1

    @property
    def is_lo(self):
        return self is WarpType.LO


DEFAULT_QK_HDIM = 128
DEFAULT_V_HDIM = 128
DEFAULT_DTYPE = "fp8"
_B8_DTYPES = (fx.Float8E4M3FN, fx.Float8E5M2)
# ``q_dtype`` types Q, ``kv_dtype`` types K and V; the two are independent (the wmma
# takes a per-operand type) so fp8 Q against bf8 KV, or the reverse, is a legal build.
# P and O follow neither: softmax P is always fp8 -- the gemm sees UNNORMALIZED
# exp(S - m), whose row max is exactly 1.0, so the extra mantissa bit beats bf8's
# exponent range (terms under e4m3's floor are negligible against a sum >= 1). The O
# accumulator and store stay 16-bit.
P_DTYPE = fx.Float8E4M3FN
_TORCH_P_DTYPE = {
    fx.Float8E4M3FN: torch.float8_e4m3fn,
    fx.Float8E5M2: torch.float8_e5m2,
}[P_DTYPE]
O_DTYPE = fx.BFloat16
TORCH_O_DTYPE = torch.bfloat16
# Q/K/V inputs this kernel serves; 16-bit belongs to the a16w16 sibling.
_TORCH_IN_DTYPES = {torch.float8_e4m3fn: "fp8", torch.float8_e5m2: "bf8"}
# What THIS kernel serves. DTYPE_MAP is shared with the 16-bit sibling and names every
# dtype, so validation must gate on these, never on the map's keys.
SUPPORTED_DTYPES = tuple(_TORCH_IN_DTYPES.values())
SUPPORTED_TORCH_DTYPES = tuple(_TORCH_IN_DTYPES)
SUPPORTED_QK_HDIM = (64, 128, 192, 256)
SUPPORTED_V_HDIM = (64, 128, 256)

# KV sequence block (columns of one QK GEMM tile). Configurable; 64 for now.
N_BLOCK_CHOICES = (32, 64, 128, 256)
# ---- LDS chunk layout: 12 x 26 KB = 312 KB of the 320 KB budget. ----
# Chunk i sits at LDS_CHUNK_BYTES * i. Every K|V tile is split 2-way along n_block, the
# halves 6 chunks apart so they land in different 64 KB LDS segments:
#
#   K[0][0]   0..26  seg 0      K[0][1] 156..182  seg 2
#   V[0][0]  26..52  seg 0      V[0][1] 182..208  seg 2,3
#   K[1][0]  52..78  seg 0,1    K[1][1] 208..234  seg 3
#   V[1][0]  78..104 seg 1      V[1][1] 234..260  seg 3,4
#   K[2][0] 104..130 seg 1,2    K[2][1] 260..286  seg 4
#   V[2][0] 130..156 seg 2      V[2][1] 286..312  seg 4
#
# Q and O own no LDS -- they time-share KV chunks (see _q_wave_base / _o_wave_base).
KV_LDS_SPLITS = 2  # halves one K (or V) tile is split into along n_block
# Per-wave Q / O region inside a chunk pair. 17 KB covers 32 rows x 256 hdim + pad.

DEFAULT_N_BLOCK = 64
# Auto-picked n_block, widest first: a wider tile must still fit one chunk per split half
# and the register file (128 doubles the live fragment state, spilling above this hdim).
N_BLOCK_PREF = (128, 64)
N_BLOCK_WIDE_MAX_QK_HDIM = 128

# K|V LDS slots the main loop rotates through. 3 is the minimum for the software-pipelined
# body: it reads V(tile_idx-1) from slot 0 and K(tile_idx) from slot 1 while tile_idx+1 is
# written into slot 2. Two slots would need V written into the slot this body still reads,
# which also races the carried ring head.
N_KV_PP = 3

# Softmax runs in log2 units: LOG2E is folded into Q by the Q loader's bf16 multiply, so
# the inner loop is a plain exp2_fastmath(S' - m) and m/LSE live in the log2 domain.

# Deferred oaccu rescale (FAv4, hk_mla spec §9.1.1): corr == 1 when the running max does
# not move, so keep m STALE while the row max stays within RESCALE_THRESHOLD natural
# logits of it and skip the full-width multiply. False (or threshold < 0) always rescales.
ENABLE_DEFER_RESCALE = True
# P = exp(S - m_stale) reaches exp(RESCALE_THRESHOLD), so the threshold is capped by
# P_DTYPE's finite max: e4m3 tops out at 448 (exp 6.10), e5m2 at 57344 (exp 10.96).
# Overflowing e4m3 converts to NaN while the denominator keeps the exact f32 sum, so
# the cap is derived from P_DTYPE rather than written down -- retyping P re-derives it.
RESCALE_THRESHOLD = min(8.0, math.log(torch.finfo(_TORCH_P_DTYPE).max) - 0.1)

# Running-max seed: finite big-negative, not -inf, so a fully-masked row keeps m finite
# and softmax's (m_prev - m_new) / fma(s, .., -m) never do -inf arithmetic (NaN).
# exp2_fastmath(big_neg - real) still underflows to 0. Masked scores themselves stay -inf.
BIG_NEG = -1.0e30

# Compile-time Q/K/V loader select. False = V1 (cluster_load_async), True = V2 (TDM,
# HW OOB, far fewer address VGPRs). The two differ only in transport, so this picks a
# class pair and no call site branches on it. O is selected separately by O_VARIANT.
USE_TDM_LOADER = True

# K/V producer specialization: LO issues every K copy, HI every V copy, each wave taking
# one dense n_block/KV_PRODUCER_WARPS row band, so a half's counter tracks one operand.
KV_PRODUCER_WARPS = NUM_WAVES // 2

# LO/HI anti-phase (FA3 ping-pong). Same tile stream, same barrier count, opposite phase:
#   LO body tile_idx:  gemm(tile_idx)                 | BAR | softmax(tile_idx)
#   HI body tile_idx:  softmax(tile_idx-1) + rescale  | BAR | gemm(tile_idx)
# so one half is in the WMMA stream while the other is in the softmax VALU stream. HI
# carries s_acc instead of P, seeded to -inf so its dead leading softmax is a no-op.
# O writer variant (decoupled from USE_TDM_LOADER): "v1" swizzled LDS + buffer_store (fastest so
# far), "v2" TDM store (padding ignored -> contiguous LDS -> bank conflict, slow), "v3" padded LDS +
# global_store_async_from_lds_b128.

# LDS->VGPR ring for the fused PV+QK WMMA stream (``_pv_qk_gemm``). Holds only RING loads
# live at 4 VGPR each, num_prefetch = RING - 2*LAG in flight, instead of bursting a whole
# tile before the first wmma. Keep RING a multiple of 4 -- an odd wrap flips slot parity,
# costing a cycle on half the wmma.
PVQK_RING = 24
PVQK_LAG = 2
# Ring-head loads issued at the END of a body (under the softmax VALU) and carried across
# the back edge, instead of at the top of the body that consumes them. Clamped to
# num_prefetch; 0 restores the un-carried head. Costs 4 VGPR per carried load.
PVQK_HEAD_CARRY = 20

# NOTE: the remaining tiling constants (chunk sizes, K/V write-tile + V swizzle
# granularity) live inside fmha_buffer_managers.py — they are intrinsic to the
# managers' LDS layouts, so the kernel no longer declares them here.


# ============================================================================
# Small device helpers
# ============================================================================


# ============================================================================
# Compute stages
# ============================================================================


_B8_WMMA = {
    (fx.Float8E4M3FN, fx.Float8E4M3FN): rocdl.wmma_f32_16x16x64_fp8_fp8,
    (fx.Float8E4M3FN, fx.Float8E5M2): rocdl.wmma_f32_16x16x64_fp8_bf8,
    (fx.Float8E5M2, fx.Float8E4M3FN): rocdl.wmma_f32_16x16x64_bf8_fp8,
    (fx.Float8E5M2, fx.Float8E5M2): rocdl.wmma_f32_16x16x64_bf8_bf8,
}


def _load_descale(ptr):
    """One per-tensor fp32 descale. Uniform and loop-invariant, so it hoists."""
    return fx.Float32(fx.ptr_load(fx.get_iter(ptr)))


def _wmma(a, b, c, *, a_dtype, b_dtype):
    """One WMMA: C[16x16 f32] = A @ B + C, operand reuse off, v8 f32 accumulator.

    ``16x16x64`` -- twice bf16's contraction depth for the same 8 VGPRs. A and B may mix
    FP8(E4M3FN) and BF8(E5M2) freely, one intrinsic per combination. Operands arrive as
    v8 i32 (the intrinsics reject float vectors, and nothing on the 8-bit path ever
    materializes an f8 vector type), so the caller names their element types.
    """
    v8f32 = fx.Vector.make_type(8, fx.Float32)
    wmma = _B8_WMMA[(a_dtype, b_dtype)]
    # modC defaults to WMMACModifier::none (== the old modC=0); omit it.
    return wmma(v8f32, _ir(a), _ir(b), _ir(c), reuseA=False, reuseB=False).result


_B8_WMMA128 = {
    (fx.Float8E4M3FN, fx.Float8E4M3FN): rocdl.wmma_f32_16x16x128_fp8_fp8,
    (fx.Float8E4M3FN, fx.Float8E5M2): rocdl.wmma_f32_16x16x128_fp8_bf8,
    (fx.Float8E5M2, fx.Float8E4M3FN): rocdl.wmma_f32_16x16x128_bf8_fp8,
    (fx.Float8E5M2, fx.Float8E5M2): rocdl.wmma_f32_16x16x128_bf8_bf8,
}
# Fuse each pair of adjacent 64-deep contraction steps into one 16x16x128 WMMA (A/B = the
# two 16x16x64 operands concatenated -- HW-checked bit-identical to the chained pair).
# Halves the PV WMMA count and drops QK from 3 to 2 per (kv frag, q tile).
# AITER_FMHA_A8W8_W128=0 restores the 16x16x64 stream.
USE_W128 = os.environ.get("AITER_FMHA_A8W8_W128", "1") != "0"
# Softmax pass 2 in vector form: contracted v8 fma (v_pk_fma_f32 on the WMMA output
# pairs), all fmas -> one uninterrupted exp run -> sums, sched_barrier-separated so no
# TRANS result feeds the next VALU (no v_nop hazard padding).
# AITER_FMHA_A8W8_VEC_SOFTMAX=0 restores the scalar path.
USE_VSM = os.environ.get("AITER_FMHA_A8W8_VEC_SOFTMAX", "1") != "0"
# Softmax row sum by WMMA (all-ones A against the FP8 P that also feeds PV) instead of
# a VALU add tree, on calls without band masking. O is normalized by the sum of the P
# it actually used; LSE then carries P's FP8 rounding (same as the ASM FP8 kernel).
# AITER_FMHA_A8W8_WMMA_ROWSUM=0 restores the f32 VALU row sum everywhere.
USE_WSUM = os.environ.get("AITER_FMHA_A8W8_WMMA_ROWSUM", "1") != "0"
WSUM_MASKED = os.environ.get("AITER_FMHA_A8W8_WMMA_ROWSUM", "1") == "2"
_VSM_NO_REASSOC = (
    _fx_arith.FastMathFlags.nnan
    | _fx_arith.FastMathFlags.ninf
    | _fx_arith.FastMathFlags.nsz
    | _fx_arith.FastMathFlags.contract
    | _fx_arith.FastMathFlags.afn
)


def _wmma128(a, b, c, *, a_dtype, b_dtype):
    """One 16x16x128 WMMA; a/b are v16 i32 (two 16x16x64 operands concatenated)."""
    v8f32 = fx.Vector.make_type(8, fx.Float32)
    r = _B8_WMMA128[(a_dtype, b_dtype)](
        v8f32, _ir(a), _ir(b), _ir(c), reuseA=False, reuseB=False
    )
    return r.result if hasattr(r, "result") else r


_B8_PK8_CVT = {
    fx.Float8E4M3FN: rocdl.cvt_scalef32_pk8_fp8_f32,
    fx.Float8E5M2: rocdl.cvt_scalef32_pk8_bf8_f32,
}


def _convert_vectors(vec_lists, dtype):
    """Convert a list of lists of vectors to ``dtype``.

    ``.to()`` is no good for an 8-bit result: it emits ``arith.truncf``, which has no
    lowering pattern for an f8 vector, so the value stays f8-typed while the pipeline
    legalizes its users to i8 and the bridging cast dies unlegalized. The hardware
    convert sidesteps that and does all 8 elements in one instruction; scale is 1."""
    cvt = _B8_PK8_CVT[dtype]
    v2i32 = fx.Vector.make_type(2, fx.Int32)
    one = _ir(fx.Float32(1.0))
    return [[fx.Vector(cvt(v2i32, _ir(v), one)) for v in row] for row in vec_lists]


def _softmax(
    *,
    s_list,
    m_prev_list,
    d_prev_list,
    lane_idx,
    n_block,
    kv_swap_delta,
    kv_pos_base=None,
    q_max_list=None,
    q_min_list=None,
    kv_len=None,
    s_scale,
    need_sum=True,
):
    """Online-softmax update for one KV tile, for ALL num_q_tiles q-WMMA-tiles this wave owns.

    Every ``*_list`` argument and return is length num_q_tiles (= num_q_tiles_per_wave). The num_q_tiles rows are
    independent (each owns its S, running m/d and mask bounds) but share the tile's K/V,
    so processing them together lets ``_tree_reduce_multi`` interleave their max/sum trees
    and dual-issue the combines, hiding each other's permlanex16 latency. ``s_list`` is
    already in log2 units (the loader folds softmax_scale*LOG2E into Q), so exp is exp2.

    Layout (from ``_pv_qk_gemm``): ``s_list[r]`` is ``num_kv_frags = n_block//WMMA_N`` v8-f32
    accumulators; this lane owns query ``q = warp*16 + l%16`` and, in tile ``kv_frag_idx``, kv rows
    ``kv_frag_idx*16 + (l//16)*8 + [0..8)``. The peer lane ``l^16`` holds the other 8-row half of
    the same q, so rows reduce locally over (kv_frag_idx, i) then across ``shuffle_xor(16)``.

    ``kv_swap_delta`` corrects the swapped half-order read: when the wave takes the tile's
    two halves in the other order, slot ``kv_frag_idx`` holds physical kv-tile ``kv_frag_idx ^ (num_kv_frags/2)``,
    so its absolute offset moves by +/- ``n_block/2``. A runtime Int32 (0 on even SIMDs)
    folded into two per-body mask bases rather than into the num_kv_frags per-slot constants.

    Masking (per element, sequence-relative ``kv_pos = kv_pos_base + (l//16)*8 + kv_frag_idx*16 +
    i``; a None bound is skipped): ``q_max_list[r]`` masks ``kv_pos > q_max`` (band upper
    edge ``q_seq + (kv_len-q_len) + window_right``, clamped to ``kv_len-1`` on the last
    tile to fold the tail); ``q_min_list[r]`` masks ``kv_pos < q_min`` (lower edge, minus
    ``window_left``); ``kv_len`` masks ``kv_pos >= kv_len`` and applies only when q_max is
    None (the standalone non-causal tail).

    Returns ``(p, m_new, d_new, corr)`` plus ``rescale_masks``. p is num_kv_frags v8 **f32** P^T =
    exp(S^T - m_new), NOT narrowed to the wmma element type -- the caller places that with
    ``_convert_vectors``. m_new is STALE (== m_prev) when that row's ballot did not fire (FAv4
    §9.1.1), with corr == 1 and d_new = corr*d_prev + rowsum(p). ``rescale_masks`` is the num_q_tiles
    raw ballots (None when deferral is compiled out), for the caller to fold into its one
    branch condition at the use site.
    """
    num_kv_frags = n_block // WMMA_N
    f32 = T.f32
    neg_inf = fx.Float32(float("-inf"))
    _defer = ENABLE_DEFER_RESCALE and RESCALE_THRESHOLD >= 0.0

    # permlanex16 selectors: identity cross-16 gather (nibbles 0..15) => lane l<->l^16.
    sel_lo, sel_hi = _raw(fx.Int32(0x76543210)), _raw(fx.Int32(0xFEDCBA98))

    def peer(v):  # cross-lane reduce partner: lane l <-> l^16 (the other kv half)
        return fx.Float32(
            rocdl.permlanex16(
                f32,
                _raw(v),
                _raw(v),
                sel_lo,
                sel_hi,
                fi=False,
                bound_control=False,
            )
        )

    kv_half_idx = lane_idx // fx.Int32(
        WMMA_M
    )  # 0/1: which 8-row kv half this lane owns

    # Mask base: swapped reads move the low num_kv_frags/2 slots up by n_block/2 and the high slots
    # down, so one base per half absorbs the whole correction and the per-slot term stays
    # the compile-time kv_frag_idx*WMMA_N + i.
    if kv_pos_base is not None:
        _kv_pos_half_base = kv_pos_base + kv_half_idx * fx.Int32(8)
        _kv_pos_halves = [
            _kv_pos_half_base + kv_swap_delta,
            _kv_pos_half_base - kv_swap_delta,
        ]

    num_q_tiles = len(s_list)
    q_max_list = q_max_list if q_max_list is not None else [None] * num_q_tiles
    q_min_list = q_min_list if q_min_list is not None else [None] * num_q_tiles

    # ---- Pass 1: masked S, flattened (kv_frag_idx, i) order. Built for every row first so the
    # max-trees below emit INTERLEAVED. ----
    s_masked_list = []
    for r in range(num_q_tiles):
        s = s_list[r]
        q_max, q_min = q_max_list[r], q_min_list[r]
        s_masked = []
        for kv_frag_idx in range(num_kv_frags):
            svec = fx.Vector(_ir(s[kv_frag_idx]))
            for i in range(8):
                sval = fx.Float32(svec[i])
                if q_max is not None or q_min is not None or kv_len is not None:
                    kv_pos = _kv_pos_halves[
                        kv_frag_idx >= num_kv_frags // 2
                    ] + fx.Int32(kv_frag_idx * WMMA_N + i)
                    if q_max is not None:
                        ubound = (
                            q_max
                            if kv_len is None
                            else fx.min(q_max, kv_len - fx.Int32(1))
                        )
                        sval = (kv_pos > ubound).select(neg_inf, sval)
                    if q_min is not None:
                        sval = (kv_pos < q_min).select(neg_inf, sval)
                    if kv_len is not None and q_max is None:
                        sval = (kv_pos >= kv_len).select(neg_inf, sval)
                s_masked.append(sval)
        s_masked_list.append(s_masked)

    # ---- Row max: balanced max-trees, interleaved across rows. ----
    max3 = lambda a, b, c: fmax_fastmath(fmax_fastmath(a, b), c)
    local_max_list = _tree_reduce_multi(s_masked_list, max3, fmax_fastmath)

    # ---- Per row: peer reduce + deferred-rescale decision + corr. ----
    m_new_list, corr_list = [], []
    rescale_masks = None if not _defer else []
    for r in range(num_q_tiles):
        m_prev, q_min = m_prev_list[r], q_min_list[r]
        row_max = fmax_fastmath(local_max_list[r], peer(local_max_list[r]))
        # S is left RAW through mask and max tree so the scale has a single consumer
        # (the exp argument) and folds into its subtract. Valid because the scale is
        # positive: max(c*S) == c*max(S). One multiply per row, not per element.
        row_max = fmul_fastmath(row_max, s_scale)
        m_full = fmax_fastmath(m_prev, row_max)

        # Deferred oaccu rescale: keep m STALE while the running max barely moves, so the
        # caller SKIPS the wide `o_acc *= corr`. The ballot promotes the test to
        # wave-uniform for that branch. `>` lowers to ordered OGT, so a fully-masked
        # lane's -inf - -inf = NaN compares false and never forces a rescale.
        if _defer:
            need = fsub_fastmath(row_max, m_prev) > fx.Float32(
                RESCALE_THRESHOLD * LOG2E
            )
            # Select on the per-lane `need`, not the ballot: every lane owns its own q row
            # (the l<->l^16 pair agrees after the peer reduce), so staleness is per-lane
            # and the ballot is only the caller's branch condition. That leaves the branch
            # as the compare's only other consumer, so the fold sinks to the s_cbranch
            # instead of sitting between the max tree and the exp chain.
            m_new = need.select(m_full, m_prev)
            rescale_masks.append(fx.Int32(rocdl.ballot(fx.Int32.ir_type, need)))
        else:
            m_new = m_full

        # log2-domain, so exp2 takes the difference directly. m is seeded to BIG_NEG
        # (finite), so a fully masked row keeps m_new = BIG_NEG -> corr = 1 and
        # p = exp2_fastmath(-inf) = 0, with no (-inf)-(-inf) anywhere and no clamp needed.
        corr = exp2_fastmath(fsub_fastmath(m_prev, m_new))
        m_new_list.append(m_new)
        corr_list.append(corr)

    # ---- Pass 2: p = exp2_fastmath(S - m_new), every row's subs before any exp. The subs pair
    # into v_dual_sub_f32 against their row's shared m_new, and the exp run stays pure
    # TRANS -- the coexecution hazard is TRANS followed by non-TRANS VALU, so an
    # uninterrupted run needs none of the v_nop padding an interleaved one does. ----
    # exp2_fastmath(c*S - m) as one fma: the scale's only consumer, so it contracts.
    neg_m = [fsub_fastmath(fx.Float32(0.0), m_new_list[r]) for r in range(num_q_tiles)]
    if USE_VSM:
        # Vector form: one v8f32 per kv frag. All rows' contracted v8 fmas (4 v_pk_fma_f32
        # each, on the WMMA output's aligned register pairs -- no regalloc moves), then one
        # uninterrupted exp run, then the sums; the sched_barriers keep a TRANS result from
        # being the very next VALU's operand, so no v_nop hazard padding.
        _fm_c = _fx_arith.FastMathFlags.contract
        p_list, local_sum_list = [], []
        d8_all = []
        for r in range(num_q_tiles):
            sc8 = fx.Vector.from_elements([s_scale], fx.Float32).broadcast_to(8)
            nm8 = fx.Vector.from_elements([neg_m[r]], fx.Float32).broadcast_to(8)
            row = []
            for f in range(num_kv_frags):
                s8 = fx.Vector.from_elements(
                    s_masked_list[r][f * 8 : (f + 1) * 8], fx.Float32
                )
                # contract only (no ninf): masked -inf must stay -inf.
                with _fx_arith.fastmath(_fm_c):
                    row.append(s8 * sc8 + nm8)
            d8_all.append(row)
        rocdl.sched_barrier(0)
        for r in range(num_q_tiles):
            p_list.append(
                [
                    fx.Vector.from_elements(
                        [exp2_fastmath(fx.Float32(d8[i])) for i in range(8)],
                        fx.Float32,
                    )
                    for d8 in d8_all[r]
                ]
            )
        rocdl.sched_barrier(0)
        # Row sum: balanced v8 add tree across the frags (v_pk_add_f32), then 8 -> 1.
        for r in range(num_q_tiles if need_sum else 0):
            cur = list(p_list[r])
            with _fx_arith.fastmath(_VSM_NO_REASSOC):
                while len(cur) > 1:
                    nxt = [cur[i] + cur[i + 1] for i in range(0, len(cur) - 1, 2)]
                    if len(cur) % 2:
                        nxt.append(cur[-1])
                    cur = nxt
                v = cur[0]
                v4 = v.shuffle(v, [0, 1, 2, 3]) + v.shuffle(v, [4, 5, 6, 7])
                v2 = v4.shuffle(v4, [0, 1]) + v4.shuffle(v4, [2, 3])
            local_sum_list.append(fadd_no_reassoc(fx.Float32(v2[0]), fx.Float32(v2[1])))
    else:
        # fx.fma, no ninf fast-math: masked s is -inf and must stay -inf, and
        # -inf * (positive finite) + finite is still -inf.
        diff_list = [
            [fx.fma(sv, s_scale, neg_m[r]) for sv in s_masked_list[r]]
            for r in range(num_q_tiles)
        ]
        p_flat_list = [[exp2_fastmath(d) for d in diff] for diff in diff_list]
        p_list = [
            [
                fx.Vector.from_elements(
                    p_flat[kv_frag_idx * 8 : (kv_frag_idx + 1) * 8], fx.Float32
                )
                for kv_frag_idx in range(num_kv_frags)
            ]
            for p_flat in p_flat_list
        ]

        # ---- Row sum: interleaved balanced sum-trees. fadd_t drops reassoc so LLVM's
        # Reassociate does not re-linearize the tree into a serial chain. ----
        add3 = lambda a, b, c: fadd_no_reassoc(fadd_no_reassoc(a, b), c)
        local_sum_list = (
            _tree_reduce_multi(p_flat_list, add3, fadd_no_reassoc) if need_sum else None
        )

    if not need_sum:
        # WSUM: the row sum rides the O accumulator (rescaled with it); d is unused.
        return p_list, m_new_list, list(d_prev_list), corr_list, rescale_masks
    d_new_list = []
    for r in range(num_q_tiles):
        d_new_list.append(
            fadd_fastmath(
                fmul_fastmath(corr_list[r], d_prev_list[r]),
                fadd_fastmath(local_sum_list[r], peer(local_sum_list[r])),
            )
        )
    return p_list, m_new_list, d_new_list, corr_list, rescale_masks


def _pv_qk_gemm(
    *,
    v_emit,
    k_emit,
    p_list,
    q_frags_list,
    v_hdim,
    n_block,
    warp_type,
    o_acc_list=None,
    head=None,
    ring=PVQK_RING,
    ring_lag=PVQK_LAG,
    vld_per_frag=2,
    q_dtype=None,
    kv_dtype=None,
    phase="pvqk",
):
    """GEMM2(u-1) then GEMM1(u) driven by ONE ring over the concatenated V-then-K
    fragment streams, for all num_q_tiles q-WMMA-tiles this wave owns.

    V and K are shared across the q-tiles, so each fragment is shuffled once and fed into
    num_q_tiles independent WMMA chains. Consumption stays strictly sequential -- a full PV pass
    then a full QK pass -- so P dies before the first QK wmma and the two accumulator sets
    are never co-live. What the single ring buys is the refill window: the K loads for
    QK(u) issue ~NP/2 fragments before the transition, while PV's wmma stream still runs.

    GEMM2, O^T = V^T @ P^T, D[M=d, N=q]: **A = V^T** (src_a, transpose-loaded via
    ds_load_tr16_b128, two V-tiles (kv, kv+16) shuffled) and **B = P^T** (src_b, two
    softmax tiles p[2*k_tile_idx], p[2*k_tile_idx+1]). Contracts kv in ``num_contract_tiles = n_block//WMMA_K`` tiles and
    produces ``d_tiles = v_hdim//WMMA_M`` d-tiles on the M axis. Lane ``l`` element ``si``
    of tile ``d_tile_idx`` holds O[q = l%16, d = d_tile_idx*WMMA_M + (l//16)*8 + si] -- the OManager
    fragment layout.

    GEMM1, S^T = K @ Q^T: **K = A-operand**, **Q = B-operand**. Contracts d in
    ``num_k_tiles = qk_hdim//WMMA_K`` tiles and produces ``num_kv_frags = n_block//WMMA_N`` kv-tiles. Lane
    ``l`` element ``si`` holds S^T[kv = kv_tile*WMMA_N + (l//16)*8 + si, q = l%16] (kv on
    the C-row / M axis, q on the C-col / N axis; GPU-verified). Each K fragment is the two
    16-col halves of a d-tile shuffled into a v16 matching the Q frag layout.

    ``p_list``/``q_frags_list``/``o_acc_list`` are length-num_q_tiles lists: per q-tile its bf16 P^T
    B-operands, its num_k_tiles v16-bf16 Q fragments, and (or None) its running ``d_tiles`` v8-f32
    O accumulator, already rescaled by ``corr``, which PV accumulates onto online.
    ``v_emit(j)``/``k_emit(j)`` emit the j-th V transpose-load / K ds_load of the resident
    block in flat ``(d_tile_idx, k_tile_idx, half)`` and ``(kv, d_tile_idx, half)`` order; ``_ring_drive`` calls
    them on demand. ``head`` adopts the NP-deep prefetch the caller issued before softmax,
    so that latency still hides under the softmax VALU.

    Returns ``(out_list, s_acc_list)``: the updated O accumulators, and per q-tile num_kv_frags
    v8-f32 S^T accumulators.
    """
    num_q_tiles = len(p_list)
    d_tiles = v_hdim // WMMA_M
    # Contraction depth follows the V load width: each tile is one WMMA's worth of kv,
    # vld_per_frag transpose loads of 16 rows each (32 kv at 16-bit, 64 at 8-bit).
    num_contract_tiles = n_block // (vld_per_frag * WMMA_N)
    num_kv_frags = n_block // WMMA_N
    num_k_tiles = len(q_frags_list[0])
    # softmax tiles per PV B operand: one per 16 kv rows of the contraction
    num_p_parts = num_kv_frags // num_contract_tiles

    # ``phase`` drops one half of the stream when the caller knows it is dead (the
    # single-KV-tile core runs QK and PV as separate passes around softmax).
    do_pv = phase != "qk"
    do_qk = phase != "pv"
    num_v_frags = d_tiles * num_contract_tiles if do_pv else 0
    num_k_frags = num_kv_frags * num_k_tiles if do_qk else 0
    num_vld = vld_per_frag * num_v_frags

    # WSUM: an extra trailing O tile carries the softmax row sum, accumulated by
    # WMMA against an all-ones A (every element of the tile = sum_kv P8[kv, q]).
    _wsum = o_acc_list is not None and len(o_acc_list[0]) == d_tiles + 1 and do_pv
    out_list = [[None] * (d_tiles + int(_wsum)) for _ in range(num_q_tiles)]
    if not do_pv:
        out_list = [list(row) for row in o_acc_list]
    _ones8 = None
    if _wsum:
        _one_byte = 0x38 if kv_dtype == fx.Float8E4M3FN else 0x3C
        _ones8 = fx.Vector.filled(8, _one_byte * 0x01010101, fx.Int32)
    s_acc_list = [[None] * num_kv_frags for _ in range(num_q_tiles)]
    p_frags = (
        {}
    )  # (q_tile_idx, k_tile_idx) -> B operand, handed to _ring_drive on its last fragment

    _w128_pv = USE_W128 and num_contract_tiles % 2 == 0
    _w128_qk = USE_W128 and num_k_tiles >= 2
    _qk_pairs = num_k_tiles // 2
    _pend_v = {}
    _pend_k = {}
    _q128 = (
        [
            [
                _concat([q_frags_list[q][2 * j], q_frags_list[q][2 * j + 1]])
                for j in range(_qk_pairs)
            ]
            for q in range(num_q_tiles)
        ]
        if _w128_qk
        else None
    )

    def emit(j):
        return v_emit(j) if j < num_vld else k_emit(j - num_vld)

    def consume(i, *parts):
        if i < num_v_frags:
            d_tile_idx, k_tile_idx = divmod(i, num_contract_tiles)
            v_frag = _concat(parts)
            if _w128_pv:
                if k_tile_idx % 2 == 0:
                    _pend_v[d_tile_idx] = v_frag
                    return
                v_frag = _concat([_pend_v.pop(d_tile_idx), v_frag])
                for q_tile_idx in range(num_q_tiles):
                    acc = out_list[q_tile_idx][d_tile_idx]
                    if acc is None:
                        acc = (
                            o_acc_list[q_tile_idx][d_tile_idx]
                            if o_acc_list is not None
                            else fx.Vector.filled(8, 0.0, fx.Float32)
                        )
                    p = p_list[q_tile_idx]
                    kt0 = k_tile_idx - 1
                    p_frag = _concat(
                        [p[num_p_parts * kt0 + t] for t in range(2 * num_p_parts)]
                    )
                    p_frags[(q_tile_idx, k_tile_idx)] = p_frag
                    out_list[q_tile_idx][d_tile_idx] = _wmma128(
                        v_frag, p_frag, acc, a_dtype=kv_dtype, b_dtype=P_DTYPE
                    )
                    rocdl.sched_barrier(0)
                if _wsum and i == num_v_frags - 1:
                    for q_tile_idx in range(num_q_tiles):
                        p = p_list[q_tile_idx]
                        sacc = o_acc_list[q_tile_idx][d_tiles]
                        for kt in range(num_contract_tiles):
                            sacc = _wmma(
                                _ones8,
                                _concat(
                                    [
                                        p[num_p_parts * kt + t]
                                        for t in range(num_p_parts)
                                    ]
                                ),
                                sacc,
                                a_dtype=kv_dtype,
                                b_dtype=P_DTYPE,
                            )
                        out_list[q_tile_idx][d_tiles] = sacc
                        rocdl.sched_barrier(0)
                return
            for q_tile_idx in range(num_q_tiles):
                acc = out_list[q_tile_idx][d_tile_idx]
                if acc is None:
                    acc = (
                        o_acc_list[q_tile_idx][d_tile_idx]
                        if o_acc_list is not None
                        else fx.Vector.filled(8, 0.0, fx.Float32)
                    )
                p = p_list[q_tile_idx]
                p_frag = _concat(
                    [p[num_p_parts * k_tile_idx + t] for t in range(num_p_parts)]
                )
                p_frags[(q_tile_idx, k_tile_idx)] = p_frag
                out_list[q_tile_idx][d_tile_idx] = _wmma(
                    v_frag, p_frag, acc, a_dtype=kv_dtype, b_dtype=P_DTYPE
                )
                rocdl.sched_barrier(0)
            return
        kv, d_tile_idx = divmod(i - num_v_frags, num_k_tiles)
        k_frag = _concat(parts)
        if _w128_qk and d_tile_idx < _qk_pairs * 2:
            if d_tile_idx % 2 == 0:
                _pend_k[0] = k_frag
                return
            a128 = _concat([_pend_k.pop(0), k_frag])
            for q_tile_idx in range(num_q_tiles):
                acc = (
                    s_acc_list[q_tile_idx][kv]
                    if d_tile_idx > 1
                    else fx.Vector.filled(8, 0.0, fx.Float32)
                )
                s_acc_list[q_tile_idx][kv] = _wmma128(
                    a128,
                    _q128[q_tile_idx][d_tile_idx // 2],
                    acc,
                    a_dtype=kv_dtype,
                    b_dtype=q_dtype,
                )
                rocdl.sched_barrier(0)
            return
        for q_tile_idx in range(num_q_tiles):
            acc = (
                s_acc_list[q_tile_idx][kv]
                if d_tile_idx > 0
                else fx.Vector.filled(8, 0.0, fx.Float32)
            )
            s_acc_list[q_tile_idx][kv] = _wmma(
                k_frag,
                q_frags_list[q_tile_idx][d_tile_idx],
                acc,
                a_dtype=kv_dtype,
                b_dtype=q_dtype,
            )
            rocdl.sched_barrier(0)

    def dies(i):
        # The PV half's B operands die on the final d-tile row, which is where the ring is
        # already refilling for QK. Release them through the ring so the refill sees one
        # WAR distance, not two.
        if i >= num_v_frags:
            return ()
        d_tile_idx, k_tile_idx = divmod(i, num_contract_tiles)
        if d_tile_idx != d_tiles - 1:
            return ()
        if _w128_pv and k_tile_idx % 2 == 0:
            return ()
        return [
            p_frags.pop((q_tile_idx, k_tile_idx)) for q_tile_idx in range(num_q_tiles)
        ]

    # Raise wave priority for the whole WMMA stream: under anti-phase the other half is
    # in its softmax VALU here, and the gemm half must win issue arbitration.
    rocdl.s_setprio(2)
    _ring_drive(
        widths=[vld_per_frag] * num_v_frags + [2] * num_k_frags,
        emit=emit,
        consume=consume,
        ring=ring,
        ring_lag=ring_lag,
        head=head,
        dies=dies,
    )
    rocdl.s_setprio(0 if warp_type.is_lo else 1)
    return out_list, s_acc_list


# ============================================================================
# Shared, layout-agnostic compute core
# ============================================================================


# Keys ``_attention_prologue`` shares with the compute cores' kwarg dict.
_PROLOGUE_KEYS = (
    "qk_hdim",
    "v_hdim",
    "n_block",
    "mask_left",
    "mask_right",
    "gqa_ratio",
    "num_q_tiles_per_wave",
    "ptr_Q",
    "ptr_K",
    "ptr_V",
    "softmax_scale",
    "stride_q_seq",
    "stride_k_seq",
    "stride_v_seq",
    "stride_q_head",
    "stride_k_head",
    "stride_v_head",
    "q_start",
    "q_len",
    "kv_start",
    "kv_len",
    "window_left",
    "window_right",
    "q_dtype",
    "kv_dtype",
    "paged",
)


def _build_managers(
    *,
    qk_hdim,
    v_hdim,
    n_block,
    gqa_ratio,
    num_q_tiles_per_wave,
    q_dtype,
    kv_dtype,
    paged=None,
):
    """The Q/K/V staging managers both cores use; USE_TDM_LOADER picks the transport.

    Pure object construction -- no IR is emitted until a manager method is called, so
    sharing this between the cores cannot move the generated code.
    """
    if paged is not None:
        # Q is dense whatever the KV pool looks like; only K/V gather.
        return (
            QManagerV2(
                qk_hdim=qk_hdim,
                gqa_ratio=gqa_ratio,
                num_waves=NUM_WAVES,
                q_tiles_per_wave=num_q_tiles_per_wave,
                elem_dtype=q_dtype,
            ),
            KManagerPaged(
                qk_hdim=qk_hdim,
                n_block=n_block,
                elem_dtype=kv_dtype,
            ),
            VManagerPaged(
                v_hdim=v_hdim,
                n_block=n_block,
                elem_dtype=kv_dtype,
            ),
        )
    if USE_TDM_LOADER:
        return (
            QManagerV2(
                qk_hdim=qk_hdim,
                gqa_ratio=gqa_ratio,
                num_waves=NUM_WAVES,
                q_tiles_per_wave=num_q_tiles_per_wave,
                elem_dtype=q_dtype,
            ),
            KManagerV2(
                qk_hdim=qk_hdim,
                n_block=n_block,
                elem_dtype=kv_dtype,
            ),
            VManagerV2(
                v_hdim=v_hdim,
                n_block=n_block,
                elem_dtype=kv_dtype,
            ),
        )
    return (
        QManagerV1(
            qk_hdim=qk_hdim,
            gqa_ratio=gqa_ratio,
            num_waves=NUM_WAVES,
            q_tiles_per_wave=num_q_tiles_per_wave,
            elem_dtype=q_dtype,
        ),
        KManagerV1(
            qk_hdim=qk_hdim,
            n_block=n_block,
            elem_dtype=kv_dtype,
        ),
        VManagerV1(
            v_hdim=v_hdim,
            n_block=n_block,
            elem_dtype=kv_dtype,
        ),
    )


def _attention_prologue(
    *,
    qk_hdim,
    v_hdim,
    n_block,
    mask_left,
    mask_right,
    gqa_ratio,
    num_q_tiles_per_wave,
    ptr_Q,
    ptr_K,
    ptr_V,
    softmax_scale,
    stride_q_seq,
    stride_k_seq,
    stride_v_seq,
    stride_q_head,
    stride_k_head,
    stride_v_head,
    q_start,
    q_len,
    kv_start,
    kv_len,
    window_left,
    window_right,
    q_dtype,
    kv_dtype,
    paged,  # PagedSpec, or None for a dense KV band
    warp_idx,  # runtime fx.Int32 wave index
    lds_base,  # LDS base (fx.Int32), allocated once by the caller (_alloc_lds)
):
    """Everything both halves do before the first body, traced once per build.

    The halves differ in four things only: which band they copy (LO K, HI V), which
    slot their second copy fills, how many copies fit in Q's global-load shadow, and
    HI's resting priority. All four ride two runtime warp-type branches here, both
    inside Q's global load; everything else -- the managers, the LDS map, Q's own
    load, and this WG's KV span -- is identical and reaches the binary once.
    """
    BLOCK_M = _block_m(num_q_tiles_per_wave)
    lane_idx = _lane_id()
    kv_head, q_head_idx, seq_idx = _packed_tile_indices(
        gqa_ratio, warp_idx, lane_idx, num_q_tiles_per_wave
    )

    # softmax_scale*LOG2E, folded into Q by the loader's bf16 multiply.
    # 8-bit Q cannot carry the scale (no f8 multiply), so softmax applies it to S.
    _q_scale = None
    _s_scale = softmax_scale * fx.Float32(LOG2E)

    # K/V staging: N_KV_PP slots of 2 LDS_CHUNK_BYTES chunks each, every tile split 2-way
    # along n_block into chunks 6 apart (different 64 KB segments). Q time-shares slot 1's
    # chunks, O the two slots the loop has finished with -- see the 12-chunk map up top.
    q_mgr, k_mgr, v_mgr = _build_managers(
        paged=paged,
        qk_hdim=qk_hdim,
        v_hdim=v_hdim,
        n_block=n_block,
        gqa_ratio=gqa_ratio,
        num_q_tiles_per_wave=num_q_tiles_per_wave,
        q_dtype=q_dtype,
        kv_dtype=kv_dtype,
    )
    k_blk_bytes = k_mgr.get_lds_size_in_byte()
    v_blk_bytes = v_mgr.get_lds_size_in_byte()
    assert k_blk_bytes % KV_LDS_SPLITS == 0 and v_blk_bytes % KV_LDS_SPLITS == 0
    # Chunk stride between a tile's two n_block halves (K[slot_idx][0] -> K[slot_idx][1]) and between
    # consecutive slots. Slot slot_idx occupies chunks 2pp, 2pp+1 low and 2pp+6, 2pp+7 high.
    _SPLIT_STRIDE = 6 * LDS_CHUNK_BYTES  # 156 KB
    slot_bytes = 2 * LDS_CHUNK_BYTES  # 52 KB: one slot's K|V pair in one half
    for _label, _bytes in (("K", k_blk_bytes), ("V", v_blk_bytes)):
        assert _bytes // KV_LDS_SPLITS <= LDS_CHUNK_BYTES, (
            f"{_label} split {_bytes // KV_LDS_SPLITS}B exceeds the {LDS_CHUNK_BYTES}B chunk "
            f"(qk_hdim={qk_hdim}, v_hdim={v_hdim}, n_block={n_block})"
        )
    assert 2 * N_KV_PP * slot_bytes <= get_lds_capacity_bytes(
        "gfx1250"
    ), "12-chunk K|V layout over LDS capacity"

    def _k_lds_buf(
        slot_idx,
    ):  # slot ``pp``'s low-half base == K[slot_idx][0] (int or fx.Int32; folds when const)
        if isinstance(slot_idx, int):
            slot_idx = fx.Int32(slot_idx)
        return lds_base + slot_idx * fx.Int32(slot_bytes)

    def _v_lds_buf(slot_idx):  # V[slot_idx][0] == the slot base one chunk in
        return _k_lds_buf(slot_idx) + fx.Int32(LDS_CHUNK_BYTES)

    # Logical slot -> physical chunk pair. Q covers all of physical slot 1 and the low
    # 17 KB of slot 2's K chunks, so physical slot 0 is the only Q-disjoint one; putting
    # the first tile (logical slot 1) there lets the prologue issue it before Q is read.
    _PHYSICAL_SLOT = [2, 0, 1]

    def _k_bufs_at(
        slot,
    ):  # K[.][0], K[.][1] of the slot whose low-half base is ``slot``
        return [slot, slot + fx.Int32(_SPLIT_STRIDE)]

    def _v_bufs_at(slot):
        v_start_desc = slot + fx.Int32(LDS_CHUNK_BYTES)
        return [v_start_desc, v_start_desc + fx.Int32(_SPLIT_STRIDE)]

    # READ side only -- the producer keeps writing split s to chunk s. Odd SIMDs take the
    # two bases in the other order, so the parities are never in the same 64 KB segment
    # set at the same point of a gemm. _split_bufs runs only in the prologue and its
    # results are carried as ds pointers, so this costs a few adds and nothing in-loop.
    assert KV_LDS_SPLITS == 2, "parity order needs a 2-way split"
    _is_odd_warp = warp_idx & fx.Int32(1)
    _read_order_lo = _is_odd_warp * fx.Int32(_SPLIT_STRIDE)
    _read_order = [_read_order_lo, fx.Int32(_SPLIT_STRIDE) - _read_order_lo]
    # Slot kv_frag_idx then holds physical kv-tile kv_frag_idx ^ (num_kv_frags/2); the mask is the only consumer
    # of the absolute index, and it takes the correction as +/- this delta.
    _kv_swap_delta = _is_odd_warp * fx.Int32(n_block // 2)

    def _split_bufs(b):
        return [b + o for o in _read_order]

    def _k_lds_bufs(slot_idx):
        return _split_bufs(_k_lds_buf(slot_idx))

    def _v_lds_bufs(slot_idx):
        return _split_bufs(_v_lds_buf(slot_idx))

    # ---- Q and O own no LDS: both time-share KV chunks in LDS_QO_BYTES per-wave slices.
    # Q sits in the K[1][0] / K[1][1] chunk pairs (52 KB and 208 KB, 4 waves x 17 KB each);
    # it is drained into VGPR and dead before the prologue issues a tile into slot 1 or 2
    # (s_wait_dscnt + barrier below). Wave w -> chunk B iff (w&1) ^ ((w>>2)&1), so the two
    # waves sharing a SIMD (w and w+4) always land in different chunks. ----
    for _label, _bytes in (("Q", q_mgr.warp_lds_size_in_byte()),):
        assert (
            _bytes <= LDS_QO_BYTES
        ), f"{_label} per-wave {_bytes}B over the {LDS_QO_BYTES}B slice"
    _q_chunk_sel = (warp_idx & fx.Int32(1)) ^ ((warp_idx >> fx.Int32(2)) & fx.Int32(1))
    q_lds_warp = (
        _k_lds_buf(1)
        + _q_chunk_sel * fx.Int32(_SPLIT_STRIDE)
        + (warp_idx >> fx.Int32(1)) * fx.Int32(LDS_QO_BYTES)
    )

    q_mgr.load_q_to_vgpr_part1(
        ptr_Q=ptr_Q,
        stride_q_seq=stride_q_seq,
        stride_q_head=stride_q_head,
        q_start=q_start,
        q_len=q_len,
        kv_head=kv_head,
        block_x=_m_tile_idx(),
        warp_idx=warp_idx,
        lane_idx=lane_idx,
        ptr_lds_warp=q_lds_warp,
    )

    # ---- This WG's KV tiles span relative kv [start_tile_idx*n_block, kv_len_wg); mask_left
    # moves start_tile_idx past whole tiles before the WG's min query's band start. The
    # defensive min() keeps start_tile_idx a valid buffer index even for an over-launched WG
    # whose whole band is empty (its per-element masks zero the work anyway).
    m_tile_idx, causal_off, kv_len_wg, num_tiles = _wg_kv_span(
        n_block=n_block,
        mask_right=mask_right,
        gqa_ratio=gqa_ratio,
        num_q_tiles_per_wave=num_q_tiles_per_wave,
        q_len=q_len,
        kv_len=kv_len,
        window_right=window_right,
    )
    last_tile_idx = num_tiles - fx.Int32(1)  # always carries the kv_len tail
    if mask_left:
        wg_min_seq = (m_tile_idx * fx.Int32(BLOCK_M)) // fx.Int32(gqa_ratio)
        kv_lo = fx.max(wg_min_seq + causal_off - window_left, fx.Int32(0))
        start_tile_idx = kv_lo // fx.Int32(n_block)
        start_tile_idx = fx.min(start_tile_idx, last_tile_idx)
    else:
        start_tile_idx = fx.Int32(0)

    def _tile_row0(t):  # clamped to the last tile, never guarded off the end
        return fx.min(t, last_tile_idx) * fx.Int32(n_block)

    # How many rows of tile t are in-bounds: n_block for every tile but the last, whose
    # tail is loop-invariant. Tiles past last_tile_idx read as the last one, matching
    # _tile_row0's clamp. A scalar select keeps this out of the VALU -- the min/max form
    # folds to v_med3_i32, which has no scalar counterpart, so a uniform value would go
    # out to a VGPR and come back through v_readfirstlane_b32 on every body.
    tail_valid = kv_len_wg - last_tile_idx * fx.Int32(n_block)

    def _kv_valid(t):
        return (t < last_tile_idx).select(fx.Int32(n_block), tail_valid)

    # ---- Prologue. All K/V address arithmetic is pure (no memory op until
    # ``.async_load()``), so it is hoisted into the Q global-load shadow and the copies
    # themselves issue last, leaving nothing between them and the prologue barrier.
    #
    # Slot rotation is local to this WG's tile stream, so start_tile_idx never enters the
    # placement. Body tile_idx reads K from slot 1 and V from slot 0, so the first body needs tile
    # start_tile_idx's K AND V in slot 1 (K for QK(start_tile_idx), V for the next body's PV) and
    # only FINITE data in slot 0's V region -- its PV is the dead leading one, p == 0, and
    # 0 * NaN would poison O. Loading start_tile_idx's V there is the cheapest such filler.
    start_row0 = start_tile_idx * fx.Int32(n_block)
    # This wave's slot among its half's producers; it owns one dense
    # n_block/KV_PRODUCER_WARPS row band and copies it alone.
    _producer_warp = warp_idx % fx.Int32(KV_PRODUCER_WARPS)
    _kv_ctx = ProducerCtx(
        producer_warp=_producer_warp,
        num_producer_warps=KV_PRODUCER_WARPS,
        lane_idx=lane_idx,
    )
    if paged is None:
        _kv_policy = _ContiguousKVPolicy(
            tile_row0=_tile_row0, kv_valid=_kv_valid, row_base=kv_start
        )
    else:
        _kv_policy = _PagedKVPolicy(
            tile_row0=_tile_row0,
            kv_valid=_kv_valid,
            ptr_page_indices=paged.ptr_page_indices,
            num_page_entries=paged.num_page_entries,
            page_base=paged.page_base,
            page_size=paged.page_size,
            num_pool_rows=paged.num_pool_rows,
            producer_warp=_producer_warp,
            num_producer_warps=KV_PRODUCER_WARPS,
            n_block=n_block,
        )

    def _k_desc(slot, row0, valid, page_indices=None):
        """K descriptor for one tile, addressed off the slot's low-half base. Pure until
        ``.async_load()``; which transport it carries (async, TDM or paged gather) is the
        manager's business, not this call site's."""
        return k_mgr.load_descriptor(
            ptr_lds=_k_bufs_at(slot),
            ptr_src=ptr_K,
            stride_seq=stride_k_seq,
            stride_head=stride_k_head,
            head=kv_head,
            row0=_kv_policy.row_base + row0,
            valid=valid,
            ctx=_kv_ctx,
            **_kv_policy.desc_kwargs(page_indices),
        )

    def _v_desc(slot, row0, valid, page_indices=None):
        return v_mgr.load_descriptor(
            ptr_lds=_v_bufs_at(slot),
            ptr_src=ptr_V,
            stride_seq=stride_v_seq,
            stride_head=stride_v_head,
            head=kv_head,
            row0=_kv_policy.row_base + row0,
            valid=valid,
            ctx=_kv_ctx,
            **_kv_policy.desc_kwargs(page_indices),
        )

    # LO copies the K band, HI the V band, both of tile start into slot 1; HI owes one more,
    # V(start) into slot 0 for body start's dead PV. Built here because the address math is
    # pure; only the ``.async_load()``s go under the warp-type branch.
    _start_valid = _kv_valid(start_tile_idx)
    _start_pages = _kv_policy.indices_for(start_tile_idx)
    k_start_desc = _k_desc(
        _k_lds_buf(_PHYSICAL_SLOT[1]), start_row0, _start_valid, _start_pages
    )
    v_start_desc = _v_desc(
        _k_lds_buf(_PHYSICAL_SLOT[1]), start_row0, _start_valid, _start_pages
    )

    # Per-half counter usage, off the descriptors' declared counts. A counter this half
    # never touches gets -1 ("do not wait on it"), NOT 0.
    kv_counts = {
        WarpType.LO: (k_start_desc.tensorcnt or -1, k_start_desc.asynccnt or -1),
        WarpType.HI: (v_start_desc.tensorcnt or -1, v_start_desc.asynccnt or -1),
    }
    _drains = {w: _kv_drain_depths(*c) for w, c in kv_counts.items()}
    # Fully-drained depths name only WHICH counters are live, and both bands live on the
    # same one, so one prologue fence serves both halves.
    assert _drains[WarpType.LO] == _drains[WarpType.HI], "prologue fence is per-WG"
    _kv_drain = _drains[WarpType.LO]

    # Under V2 no prologue copy lands on Q's chunks, so all go out before Q is read and
    # overlap its latency. Under V1 Q's own stage shares asynccnt and part2 waits a depth
    # counted over Q's loads alone, so every copy must go after part2.
    if USE_TDM_LOADER:
        _early = {
            WarpType.LO: [k_start_desc],
            WarpType.HI: [v_start_desc],
        }
        _late = {WarpType.LO: [], WarpType.HI: []}
    else:
        _early = {WarpType.LO: [], WarpType.HI: []}
        _late = {WarpType.LO: [k_start_desc], WarpType.HI: [v_start_desc]}

    # Body start reads V(start-1) from logical slot 0 against a zero P. Zero the slot
    # rather than refetching V(start) into it: stale LDS can decode to NaN, and 0*NaN is
    # NaN. Sits in Q's global-load shadow, between its part1 and part2.
    _zero_slot(
        _v_lds_buf(_PHYSICAL_SLOT[0]),
        v_blk_bytes,
        KV_LDS_SPLITS,
        _SPLIT_STRIDE,
        warp_idx * fx.Int32(WAVE_SIZE) + lane_idx,
    )

    _is_lo = warp_idx < fx.Int32(NUM_WAVES // 2)
    if USE_TDM_LOADER:
        # The halves issue different numbers of copies here, so the drain immediate is
        # per-half and rides the same branch. Both sit inside Q's global load.
        @flyc.jit
        def _issue_early(is_lo):
            if is_lo:
                for _desc in _early[WarpType.LO]:
                    _desc.async_load()
                tdm_ops.tensor_wait(sum(d.tensorcnt for d in _early[WarpType.LO]))
            else:
                for _desc in _early[WarpType.HI]:
                    _desc.async_load()
                tdm_ops.tensor_wait(sum(d.tensorcnt for d in _early[WarpType.HI]))

        _issue_early(_is_lo)
    q_frags = q_mgr.load_q_to_vgpr_part2(scale=_q_scale)

    @flyc.jit
    def _issue_late(is_lo):
        if is_lo:
            for _desc in _late[WarpType.LO]:
                _desc.async_load()
        else:
            rocdl.s_setprio(1)
            for _desc in _late[WarpType.HI]:
                _desc.async_load()

    _issue_late(_is_lo)
    rocdl.sched_barrier(0)
    _copy_fence(*_kv_drain)

    return {
        "lane_idx": lane_idx,
        "kv_head": kv_head,
        "q_head_idx": q_head_idx,
        "seq_idx": seq_idx,
        "k_mgr": k_mgr,
        "v_mgr": v_mgr,
        "kv_ctx": _kv_ctx,
        "k_lds_buf": _k_lds_buf,
        "k_bufs_at": _k_bufs_at,
        "v_bufs_at": _v_bufs_at,
        "k_lds_bufs": _k_lds_bufs,
        "v_lds_bufs": _v_lds_bufs,
        "kv_swap_delta": _kv_swap_delta,
        "split_stride": _SPLIT_STRIDE,
        "physical_slot": _PHYSICAL_SLOT,
        "m_tile_idx": m_tile_idx,
        "causal_off": causal_off,
        "num_tiles": num_tiles,
        "last_tile_idx": last_tile_idx,
        "start_tile_idx": start_tile_idx,
        "kv_policy": _kv_policy,
        "kv_counts": kv_counts,
        "kv_drain": _kv_drain,
        "q_frags": q_frags,
    }


schedule_bank.initialized = False
schedule_bank.func = None


def _attention_core_multi_kv_tiles(
    *,
    qk_hdim,
    v_hdim,
    n_block,  # compile-time KV block width (columns of one QK GEMM tile)
    mask_left,  # compile-time: bound the left band edge (finite window_left)
    mask_right,  # compile-time: bound the right band edge (causal or finite window_right)
    return_lse,
    has_sink,  # compile-time: fold a per-head sink logit into the softmax denom
    gqa_ratio,  # compile-time GQA group size = nheads_q // nheads_kv
    num_q_tiles_per_wave,  # compile-time q-WMMA tiles per wave; BLOCK_M = 16*this*8
    ptr_O,
    ptr_Q,
    ptr_K,
    ptr_V,
    ptr_LSE,
    ptr_sink,  # [nheads_q] fp32 per-head sink logits; read only when has_sink
    softmax_scale,
    stride_q_seq,
    stride_k_seq,
    stride_v_seq,
    stride_o_seq,
    stride_q_head,
    stride_k_head,
    stride_v_head,
    stride_o_head,
    # LSE addressing (element strides + per-batch bound), resolved by the caller.
    # Only consumed when return_lse; the caller may pass anything otherwise.
    stride_lse_seq,
    stride_lse_head,
    lse_base_elems,  # first element offset of this batch's LSE slab
    lse_num_records_bytes,  # buffer-resource bound (below the 0x7FFFFFFF drop)
    # Per-batch token ranges (fx.Int32), resolved by the caller:
    q_start,  # first Q token index of this batch in the global tensor
    q_len,  # valid Q tokens in this batch
    kv_start,  # first K/V token index of this batch
    kv_len,  # valid K/V tokens in this batch
    # Sliding-window bounds (runtime fx.Int32, >= 0). window_left read only when
    # mask_left, window_right only when mask_right. Causal == mask_right, window_right=0.
    window_left,
    window_right,
    warp_idx,  # runtime fx.Int32 wave index
    warp_type,  # compile-time WarpType (LO/HI x SIMD parity)
    lds_base,  # LDS base (fx.Int32), allocated once by the caller (_alloc_lds)
    prologue,  # dict from _attention_prologue: managers, LDS map, KV span, Q fragments
    paged=None,  # consumed by the prologue; here only so one kwarg dict serves both
    q_dtype,  # compile-time Q fragment type; K/V are kv_dtype, P is P_DTYPE, O is O_DTYPE
    kv_dtype,  # compile-time K/V fragment type (may differ from q_dtype; both 8-bit)
    v_descale=None,  # the epilogue's; here so one kwarg dict serves both
):
    """Layout-agnostic m32x8 compute, shared by the THD and BSHD kernel entries.

    Takes over from ``_attention_prologue`` (Q already in VGPR, the first tiles already
    in flight) and hands ``_attention_epilogue`` the finished softmax/O state. The caller
    dispatches on runtime ``warp_type``, tracing this body once per compile-time value;
    the two instantiations differ only in their main-loop phase ordering (LO drives the
    K load, HI shadows it), which is why only this middle part is duplicated.
    """
    BLOCK_M = _block_m(num_q_tiles_per_wave)
    lane_idx = prologue["lane_idx"]
    kv_head = prologue["kv_head"]
    q_head_idx = prologue["q_head_idx"]
    seq_idx = prologue["seq_idx"]
    k_mgr = prologue["k_mgr"]
    v_mgr = prologue["v_mgr"]
    _kv_ctx = prologue["kv_ctx"]
    _k_lds_buf = prologue["k_lds_buf"]
    _k_bufs_at = prologue["k_bufs_at"]
    _v_bufs_at = prologue["v_bufs_at"]
    _k_lds_bufs = prologue["k_lds_bufs"]
    _v_lds_bufs = prologue["v_lds_bufs"]
    _kv_swap_delta = prologue["kv_swap_delta"]
    _SPLIT_STRIDE = prologue["split_stride"]
    _PHYSICAL_SLOT = prologue["physical_slot"]
    m_tile_idx = prologue["m_tile_idx"]
    causal_off = prologue["causal_off"]
    num_tiles = prologue["num_tiles"]
    last_tile_idx = prologue["last_tile_idx"]
    start_tile_idx = prologue["start_tile_idx"]
    kv_policy = prologue["kv_policy"]
    _tile_row0 = kv_policy.tile_row0
    _kv_valid = kv_policy.kv_valid
    _kv_drain = prologue["kv_drain"]
    q_frags = prologue["q_frags"]
    # 8-bit Q cannot carry the scale (no f8 multiply); softmax applies it to S.
    _s_scale = softmax_scale * fx.Float32(LOG2E)

    def _get_kv_desc(slot, row0, valid, page_indices=None):
        """This half's BufferOpDescriptor for one tile: LO issues every K copy, HI every
        V copy. Pure until ``.async_load()``; which transport it carries (async, TDM or
        paged gather) is the manager's business, not this call site's."""
        _extra = kv_policy.desc_kwargs(page_indices)
        if warp_type.is_lo:
            return k_mgr.load_descriptor(
                ptr_lds=_k_bufs_at(slot),
                ptr_src=ptr_K,
                stride_seq=stride_k_seq,
                stride_head=stride_k_head,
                head=kv_head,
                row0=kv_policy.row_base + row0,
                valid=valid,
                ctx=_kv_ctx,
                **_extra,
            )
        return v_mgr.load_descriptor(
            ptr_lds=_v_bufs_at(slot),
            ptr_src=ptr_V,
            stride_seq=stride_v_seq,
            stride_head=stride_v_head,
            head=kv_head,
            row0=kv_policy.row_base + row0,
            valid=valid,
            ctx=_kv_ctx,
            **_extra,
        )

    # ---- Loop init: the online-softmax seed and the O accumulators, all iter_args.
    #
    # Seeding m=-inf, d=0, O=0 makes the first tile's corr = exp2_fastmath(m_prev-m_new) = 0 zero
    # the (already-zero) O before its PV adds in — the standard flash seed.
    #
    # Attention sink (compile-time) is one extra ``exp(sink)`` term in the softmax denom.
    # Fold it in by seeding m = sink[q_head]*LOG2E (m is log2-domain) and d = 1.0
    # (= exp(sink-sink)); the rescales carry that d seed to exactly exp(sink - m_final).
    # Without a sink the first tile's corr zeroes the d seed, so d=1 would equal d=0 --
    # the no-sink path keeps d=0 to stay byte-for-byte.
    d_tiles = v_hdim // WMMA_M
    num_q_tiles = num_q_tiles_per_wave
    num_kv_frags = n_block // WMMA_N
    # WSUM: one extra O tile holds the row sum (needs the 128-deep PV pairing).
    # Band-masked (causal / windowed) calls keep the f32 VALU row sum: their short rows (a
    # handful of keys) would carry the fp8 rounding of P into the LSE (up to ~2e-2 measured).
    _WS = int(
        USE_WSUM
        and USE_W128
        and (n_block // (v_mgr.loads_per_frag * WMMA_N)) % 2 == 0
        and (WSUM_MASKED or (not mask_left and not mask_right))
    )
    # Per-q-tile carried state: [m, d, O, P_0 .. P_{num_kv_frags-1}]. The lagging half
    # carries the f32 s_acc in the P slots instead of bf16 P; the slot count is the same.
    # O travels the back edge as ONE tuple (see _o_pack); the P slots do not, because the
    # gemm ring produces and consumes them one at a time.
    _q_state_slots = 2 + 1 + num_kv_frags
    _is_lagging_softmax = not warp_type.is_lo
    if _is_lagging_softmax:
        # Barrier-count filler: re-pairs the two halves' barrier streams.
        _copy_fence(*_kv_drain)
    if has_sink:
        num_heads_q = _num_heads_kv() * fx.Int32(gqa_ratio)
        m_init = [
            _load_sink_logit(ptr_sink, q_head_idx[q_tile_idx], num_heads_q)
            * fx.Float32(LOG2E)
            for q_tile_idx in range(num_q_tiles)
        ]
        d_init = [fx.Float32(1.0) for _ in range(num_q_tiles)]
    else:
        m_init = [fx.Float32(BIG_NEG) for _ in range(num_q_tiles)]
        d_init = [fx.Float32(0.0) for _ in range(num_q_tiles)]
    _init = []
    for q_tile_idx in range(num_q_tiles):
        _init += (
            [
                _raw(m_init[q_tile_idx]),
                _raw(
                    fx.Vector.from_elements(
                        [d_init[q_tile_idx]], fx.Float32
                    ).broadcast_to(8)
                    if _WS
                    else d_init[q_tile_idx]
                ),
            ]
            + [_raw(fx.Vector.filled(8 * d_tiles, 0.0, fx.Float32))]
            + [
                _raw(
                    fx.Vector.filled(8, float("-inf"), fx.Float32)
                    if _is_lagging_softmax
                    else _convert_vectors(
                        [[fx.Vector.filled(8, 0.0, fx.Float32)]], P_DTYPE
                    )[0][0]
                )
                for _ in range(num_kv_frags)
            ]
        )

    # ---- One ds_load base-pointer set per slot plus the slot's byte base, carried as
    # iter_args and LEFT-ROTATED in the yield, so no runtime "% N_KV_PP" is evaluated. ----
    k_lds_ld = [
        k_mgr.ds_load_ptrs(ptr_lds=_k_lds_bufs(_PHYSICAL_SLOT[i]), lane_idx=lane_idx)
        for i in range(N_KV_PP)
    ]
    v_lds_ld = [
        v_mgr.ds_load_ptrs(ptr_lds=_v_lds_bufs(_PHYSICAL_SLOT[i]), lane_idx=lane_idx)
        for i in range(N_KV_PP)
    ]
    _num_k_bases = len(k_lds_ld[0])  # ds bases per K buffer (one per LDS split)
    _num_v_bases = len(v_lds_ld[0])
    _k_bases_idx = len(_init)
    for i in range(N_KV_PP):
        _init = _init + k_lds_ld[i]
    _v_bases_idx = len(_init)
    for i in range(N_KV_PP):
        _init = _init + v_lds_ld[i]
    _slot_bases_idx = len(_init)
    _init = _init + [_raw(_k_lds_buf(_PHYSICAL_SLOT[i])) for i in range(N_KV_PP)]

    # Carried ring head: _num_head_carry of the first body's V loads, issued here (slot 0 is resident
    # -- the prologue fence just drained it) and thereafter at the end of each body.
    _num_k_frags = k_mgr.num_ds_loads() // k_mgr.loads_per_frag
    _num_v_frags = v_mgr.num_ds_loads() // v_mgr.loads_per_frag
    _num_vld = v_mgr.num_ds_loads()
    _vld_group = v_mgr.loads_per_frag  # pin the head in whole V fragments
    _num_prefetch = _ring_num_prefetch(
        _num_vld + k_mgr.num_ds_loads(), PVQK_RING, PVQK_LAG
    )
    # The lagging half opens its body with softmax, so its ring head is issued there and
    # consumed after the barrier -- within one body, nothing to carry.
    # The carry is issued a body early, where only V is resident, so it cannot run past V.
    _num_head_carry = (
        0 if _is_lagging_softmax else min(PVQK_HEAD_CARRY, _num_prefetch, _num_vld)
    )
    assert _num_head_carry <= _num_vld, "carried head must be V loads only"
    _head_carry_idx = len(_init)
    _seed_head = []
    for j in range(_num_head_carry):
        # Same per-pair pin as ``_issue_head`` in the body.
        _seed_head.append(_raw(v_mgr.load_one_to_reg(v_lds_ld[0], j)))
        if len(_seed_head) % _vld_group == 0:
            rocdl.sched_barrier(0)
    _init = _init + _seed_head

    _kv_state_idx = len(_init)
    _init = _init + [
        _raw(v) for v in kv_policy.init_state(start_tile_idx + fx.Int32(1))
    ]

    # ========================================================================
    # Main KV loop -- SOFTWARE-PIPELINED by one tile. The loop variable ``u`` is the
    # SOFTMAX tile, and body tile_idx runs
    #
    #     PV(tile_idx-1)  ->  QK(tile_idx)  ->  softmax(tile_idx)  ->  rescale O by corr(tile_idx)
    #
    # carrying P across the back edge. That puts the two gemms back to back (a shared
    # ring feeds both) and leaves exactly three live slots: V(tile_idx-1) in slot 0, K(tile_idx) in
    # slot 1, tile tile_idx+1's copy landing in slot 2. Slots are selected by the carried ds
    # pointers / bases, left-rotated at the end of each `main_loop`.
    #
    # tile_idx runs [start_tile_idx, num_tiles] -- ONE body more than there are tiles. Both extra
    # half-bodies are dead rather than peeled, an extra trace costing far more than an
    # extra iteration: body start_tile_idx's PV multiplies the seeded P == 0, and body
    # num_tiles' QK/softmax sits at kv_pos_base >= kv_len so every element masks to -inf
    # (m unchanged, corr == 1, P == 0, d unchanged).
    # ========================================================================
    def main_loop(tile_idx, state, *, mask_left, mask_right, kv_len):
        # mask_left/mask_right/kv_len shadow the closure flags: the caller splits the
        # tile stream into a mask-free clean region + boundary loops and passes None for
        # any edge this sub-loop provably doesn't cross. The split points are already
        # expressed in SOFTMAX tiles, so they carry over unchanged. The lagging half runs
        # softmax one tile behind its gemm, hence tile_idx-1.
        softmax_tile_idx = tile_idx - fx.Int32(1) if _is_lagging_softmax else tile_idx
        kv_tile_start = softmax_tile_idx * fx.Int32(
            n_block
        )  # softmax tile's first (batch-relative) kv row

        # Unpack: num_q_tiles independent per-q-tile (m, d, O, P) groups, then the shared ds state.
        m_prev = [
            fx.Float32(state[q_tile_idx * _q_state_slots + 0])
            for q_tile_idx in range(num_q_tiles)
        ]
        d_prev = [
            (
                fx.Float32(0.0)
                if _WS
                else fx.Float32(state[q_tile_idx * _q_state_slots + 1])
            )
            for q_tile_idx in range(num_q_tiles)
        ]
        o_acc = [
            _o_unpack(fx.Vector(state[q_tile_idx * _q_state_slots + 2]), d_tiles)
            + ([fx.Vector(state[q_tile_idx * _q_state_slots + 1])] if _WS else [])
            for q_tile_idx in range(num_q_tiles)
        ]
        # Carried gemm/softmax hand-off: P(tile_idx-1) on the leading half, S(tile_idx-1) on the
        # lagging one (same slot count, different element type).
        carry_prev = [
            [
                fx.Vector(state[q_tile_idx * _q_state_slots + 3 + kv_frag_idx])
                for kv_frag_idx in range(num_kv_frags)
            ]
            for q_tile_idx in range(num_q_tiles)
        ]
        k_slots = [
            list(
                state[
                    _k_bases_idx
                    + i * _num_k_bases : _k_bases_idx
                    + (i + 1) * _num_k_bases
                ]
            )
            for i in range(N_KV_PP)
        ]
        v_slots = [
            list(
                state[
                    _v_bases_idx
                    + i * _num_v_bases : _v_bases_idx
                    + (i + 1) * _num_v_bases
                ]
            )
            for i in range(N_KV_PP)
        ]
        slot_of = [fx.Int32(state[_slot_bases_idx + i]) for i in range(N_KV_PP)]
        head_carry = [
            fx.Vector(state[_head_carry_idx + i]) for i in range(_num_head_carry)
        ]
        # The ring's LDS base pointers: banked with the fragments they feed.
        v_curr = [
            schedule_bank(p, bank=1, reason="ring_lds_addr") for p in v_slots[0]
        ]  # tile tile_idx-1: this body's PV
        k_curr = [
            schedule_bank(p, bank=1, reason="ring_lds_addr") for p in k_slots[1]
        ]  # tile tile_idx:   this body's QK

        # Tile prefetched by this body, CLAMPED to the last tile rather than guarded off
        # past the end. Every body then issues exactly one tile's copies, so the fence
        # count is uniform and the last iteration needs no peel -- cheaper than the extra
        # trace a peel costs, against ~2 dead L2-resident tile loads per workgroup. The
        # clamped re-load lands in the slot body num_tiles reads as its dead K, and O
        # stages past all the slots. Both halves copy tile tile_idx+1 into slot 2 -- K and V
        # occupy disjoint chunks of it, and the slot last held tile tile_idx-2, dead since body
        # tile_idx-2 -- so the distance is the same on both and the slot arithmetic has no half
        # to branch on.
        write_slot = slot_of[N_KV_PP - 1]
        prefetch_tile_idx = tile_idx + fx.Int32(1)
        kv_state_prev = list(
            state[_kv_state_idx : _kv_state_idx + kv_policy.num_state_slots]
        )
        prefetch_row0 = _tile_row0(prefetch_tile_idx)
        prefetch_valid = _kv_valid(prefetch_tile_idx)

        # This half's descriptor for tile ``pf``'s K (LO) or V (HI) into the oldest slot.
        # Built up front: pure, so its address VALU overlaps the drain; only the copy
        # itself must stay after the barrier.
        prefetch_page_indices = kv_policy.indices(kv_state_prev)
        prefetch_desc = _get_kv_desc(
            write_slot, prefetch_row0, prefetch_valid, prefetch_page_indices
        )

        def _k_emit(j):
            return k_mgr.load_one_to_reg(k_curr, j)

        def _v_emit(j):
            return v_mgr.load_one_to_reg(v_curr, j)

        def _issue_head(emit, first, last, carried=()):
            # Pin every fragment's loads in place. The ring consumes one group per
            # fragment, but the scheduler scatters the 20-deep burst freely --
            # in the dumps a partner load drifted 12+ slots back, so the wmma waited on
            # most of the burst (s_wait_dscnt 0x9 / 0x7) instead of the ring's 0x12.
            # NOT _keepalive -- that is a USE, so it drags an s_wait_dscnt 0x0 in with it.
            out = list(carried)
            for j in range(first, last):
                out.append(emit(j))
                if len(out) % _vld_group == 0:
                    rocdl.sched_barrier(0)
            return out

        def _vk_emit(j):
            # The ring runs V then K; past the V loads the head continues into K, matching
            # ``_pv_qk_gemm``'s emit. Both slots are resident, so either is issuable here.
            return _v_emit(j) if j < _num_vld else _k_emit(j - _num_vld)

        def _pvqk_head():
            # The first _num_head_carry loads came from the previous body (issued under its softmax);
            # only the remainder is issued here.
            return _issue_head(_vk_emit, _num_head_carry, _num_prefetch, head_carry)

        def _softmax_phase(s_in, o_in):
            # This lane's query (tile q_tile_idx) attends [q_min, q_max] in batch-relative kv.
            # None bounds are skipped, so a clean-region tile passes all-None and does
            # zero per-element masking. kv_len is passed only on the right-boundary
            # sub-loop, which is also what neutralizes the dead trailing softmax. All num_q_tiles
            # q-tiles go in ONE call so their tree reductions emit interleaved.
            q_max_list = [
                seq_idx[q_tile_idx] + causal_off + window_right if mask_right else None
                for q_tile_idx in range(num_q_tiles)
            ]
            q_min_list = [
                seq_idx[q_tile_idx] + causal_off - window_left if mask_left else None
                for q_tile_idx in range(num_q_tiles)
            ]
            p_list, m_new_list, d_new_list, corr_list, rescale_masks = _softmax(
                s_list=s_in,
                m_prev_list=m_prev,
                d_prev_list=d_prev,
                lane_idx=lane_idx,
                n_block=n_block,
                kv_pos_base=kv_tile_start,
                kv_swap_delta=_kv_swap_delta,
                q_max_list=q_max_list,
                q_min_list=q_min_list,
                kv_len=kv_len,
                s_scale=_s_scale,
                need_sum=not _WS,
            )

            # Rescale each q-tile's running O by this tile's corr. The leading half
            # rescales at the END of the body and the next body's PV accumulates onto the
            # result; the lagging half rescales before its own PV. Same product either way.
            # Under deferral the wide multiply is gated behind a non-divergent scf.if that
            # fires only when a running max actually moved, with ONE branch covering all num_q_tiles
            # rows -- neither row moves in the common case, so the steady state is a single
            # not-taken s_cbranch_vccz, and a row that stayed stale has corr == 1 exactly,
            # making its rescale inside the taken branch the identity.
            corr_vecs = [
                fx.Vector.from_elements(
                    [corr_list[q_tile_idx]], fx.Float32
                ).broadcast_to(8)
                for q_tile_idx in range(num_q_tiles)
            ]
            o_vecs = [
                [
                    fx.Vector(_ir(o_in[q_tile_idx][d_tile_idx]))
                    for d_tile_idx in range(len(o_in[q_tile_idx]))
                ]
                for q_tile_idx in range(num_q_tiles)
            ]
            if rescale_masks is None:
                o_resc_list = [
                    [ov * corr_vecs[q_tile_idx] for ov in o_vecs[q_tile_idx]]
                    for q_tile_idx in range(num_q_tiles)
                ]
            else:
                # Folded HERE, not in _softmax: the ballots are VALU-produced SGPRs, so
                # keeping the s_or/s_cmp at the use site leaves the whole row-max chain
                # between the v_cmp and the turnaround. OR the raw masks, not the per-row
                # booleans -- one s_or_b32 covers all num_q_tiles rows.
                mask_any = rescale_masks[0]
                for _mask in rescale_masks[1:]:
                    mask_any = mask_any | _mask

                @flyc.jit
                def _maybe_rescale_all(o_flat_in, corr_flat, do_rescale):
                    result = o_flat_in
                    if do_rescale:
                        result = [ov * cv for ov, cv in zip(o_flat_in, corr_flat)]
                    return result

                # One flat list: the num_q_tiles rows share the folded condition, so they rescale
                # (or pass through) together.
                o_flat = list(
                    _maybe_rescale_all(
                        [v for row in o_vecs for v in row],
                        [
                            corr_vecs[q_tile_idx]
                            for q_tile_idx in range(num_q_tiles)
                            for _ in range(d_tiles + _WS)
                        ],
                        mask_any != fx.Int32(0),
                    )
                )
                o_resc_list = [
                    o_flat[
                        q_tile_idx
                        * (d_tiles + _WS) : (q_tile_idx + 1)
                        * (d_tiles + _WS)
                    ]
                    for q_tile_idx in range(num_q_tiles)
                ]
            return p_list, m_new_list, d_new_list, o_resc_list

        # Only the drain barrier (the one before tensor_load_to_lds) is required; the wall
        # is pure scheduling. Dropping it halves the in-loop barriers and wins everywhere
        # except non-window-masked qk_hdim 256 (n_block 64), which regresses ~4%.
        # Compile-time constant, so both halves still agree on the barrier count.
        _keep_phase_wall = qk_hdim >= 256 and not (mask_left and mask_right)

        def _phase_barrier():
            # One half leaves the WMMA stream here as the other enters it; the
            # sched_barriers pin that split even when the s_barrier itself is gone.
            rocdl.sched_barrier(0)
            if _keep_phase_wall:
                _bare_barrier()
            rocdl.sched_barrier(0)

        # GEMM2(tile_idx-1) then GEMM1(tile_idx) on one ring: O += P^T(tile_idx-1) @ V(tile_idx-1), then
        # S^T = K(tile_idx) @ Q^T. sched_barrier fences the ring head out of the WMMA stream
        # (no wmma<-ds_load bubble); the ring itself issues the per-fragment s_wait_dscnt.
        if _is_lagging_softmax:
            kv_state_next = kv_policy.advance(prefetch_tile_idx)
            prefetch_desc.async_load()
            p_f32, m_new_list, d_new_list, o_resc = _softmax_phase(carry_prev, o_acc)
            # Head out BEFORE P is narrowed: it is ds_loads with no dependence on P, so
            # issuing it first puts the conversion in its shadow rather than behind it.
            rocdl.sched_barrier(0)
            pvqk_head = _pvqk_head()
            rocdl.sched_barrier(0)
            p_list = _convert_vectors(p_f32, P_DTYPE)
            # Anchor the conversion in THIS block. Its only real use is past the barrier,
            # so MachineSink (which ignores sched_barrier) would otherwise sink every
            # v_cvt_pk_bf16_f32 into the gemm and interleave it with the WMMA.
            _keepalive([v for pt in p_list for v in pt])
            _phase_barrier()
            o_out, s_acc = _pv_qk_gemm(
                v_emit=_v_emit,
                k_emit=_k_emit,
                vld_per_frag=v_mgr.loads_per_frag,
                q_dtype=q_dtype,
                kv_dtype=kv_dtype,
                p_list=p_list,
                q_frags_list=q_frags,
                v_hdim=v_hdim,
                n_block=n_block,
                warp_type=warp_type,
                o_acc_list=o_resc,
                head=pvqk_head,
            )
            carry_next = s_acc
            _copy_fence(*_kv_drain, num_dscnt=_num_head_carry)
            head_next = []
        else:
            kv_state_next = kv_policy.advance(prefetch_tile_idx)
            _copy_fence(*_kv_drain, num_dscnt=_num_head_carry)
            pvqk_head = _pvqk_head()
            prefetch_desc.async_load()
            rocdl.sched_barrier(0)
            o_acc, s_list = _pv_qk_gemm(
                v_emit=_v_emit,
                k_emit=_k_emit,
                vld_per_frag=v_mgr.loads_per_frag,
                q_dtype=q_dtype,
                kv_dtype=kv_dtype,
                p_list=carry_prev,
                q_frags_list=q_frags,
                v_hdim=v_hdim,
                n_block=n_block,
                warp_type=warp_type,
                o_acc_list=o_acc,
                head=pvqk_head,
            )
            _phase_barrier()
            carry_f32, m_new_list, d_new_list, o_out = _softmax_phase(s_list, o_acc)
            # Next body's ring head: V(tile_idx) from the slot this body read K from, resident
            # and already fenced. Mirrors the lagging half -- both issue the head just
            # before the barrier preceding the gemm that consumes it, ahead of the
            # narrowing for the same reason.
            rocdl.sched_barrier(0)
            head_next = _issue_head(
                lambda j: v_mgr.load_one_to_reg(v_slots[1], j), 0, _num_head_carry
            )
            rocdl.sched_barrier(0)
            carry_next = _convert_vectors(carry_f32, P_DTYPE)

        # Yield: num_q_tiles updated (m, d, O, P) groups, then the ds pointers and slot bases
        # left-rotated by one so slot 0 holds tile tile_idx (next body's PV) and the oldest
        # rotates into the write position.
        out = []
        for q_tile_idx in range(num_q_tiles):
            out += (
                [
                    _raw(m_new_list[q_tile_idx]),
                    _raw(o_out[q_tile_idx][d_tiles] if _WS else d_new_list[q_tile_idx]),
                ]
                + [_raw(_o_pack(o_out[q_tile_idx][:d_tiles]))]
                + [_raw(cv) for cv in carry_next[q_tile_idx]]
            )
        rotation = list(range(1, N_KV_PP)) + [0]
        for i in rotation:
            out += k_slots[i]
        for i in rotation:
            out += v_slots[i]
        out += [_raw(slot_of[i]) for i in rotation]
        out += [_raw(h) for h in head_next]
        out += [_raw(v) for v in kv_state_next]
        return out

    # ---- Stream softmax tiles [start_tile_idx, num_tiles] through 3 sub-loops split by the
    # attention band, so interior tiles fully inside the band skip masking. clean_lo_idx /
    # clean_hi_idx are runtime split points, but each sub-loop's mask on/off is COMPILE-TIME.
    # The rotation state threads continuously through all three, leaving the slot
    # assignment intact, and the bounds are already softmax-tile indices, so the pipeline
    # shift only extends the last loop by one body (the dead-QK one).
    #   [start_tile_idx, clean_lo_idx)    left boundary   (emitted only when mask_left)
    #   [clean_lo_idx,   clean_hi_idx)    clean, no mask
    #   [clean_hi_idx,   num_tiles+1) right boundary + kv_len tail + the dead trailing body
    # clean_hi_idx = first tile that could need RIGHT masking = the WG's earliest query's
    # diagonal tile ((min q_max + 1)//n_block). Kept <= last_tile_idx so the tail tile stays in
    # the right loop, and >= start_tile_idx for a valid partition.
    if mask_right:
        wg_min_seq = (m_tile_idx * fx.Int32(BLOCK_M)) // fx.Int32(gqa_ratio)
        qmax_min = fx.max(wg_min_seq + causal_off + window_right, fx.Int32(0))
        clean_hi_idx = (qmax_min + fx.Int32(1)) // fx.Int32(n_block)
    else:
        clean_hi_idx = fx.Int32(num_tiles)
    clean_hi_idx = fx.max(fx.min(clean_hi_idx, last_tile_idx), start_tile_idx)

    # clean_lo_idx = first tile fully at/above the WG's latest query's window start
    # (ceildiv(max q_min, n_block)); clamped into [start_tile_idx, clean_hi_idx].
    if mask_left:
        wg_max_seq = fx.min(
            (m_tile_idx * fx.Int32(BLOCK_M) + fx.Int32(BLOCK_M - 1))
            // fx.Int32(gqa_ratio),
            q_len - fx.Int32(1),
        )
        qmin_max = fx.max(wg_max_seq + causal_off - window_left, fx.Int32(0))
        clean_lo_idx = (qmin_max + fx.Int32(n_block - 1)) // fx.Int32(n_block)
    else:
        clean_lo_idx = start_tile_idx
    clean_lo_idx = fx.min(fx.max(clean_lo_idx, start_tile_idx), clean_hi_idx)
    if mask_left:
        # The lagging half's softmax tile is tile_idx-1, so the clean loop must start one body
        # later or tile clean_lo_idx-1 would cross the left edge unmasked. The left loop
        # absorbs the extra body; over-masking a clean tile is a no-op. Only needed when
        # a left loop exists at all -- with mask_left off, clean_lo_idx == start_tile_idx and
        # shifting would drop the first body entirely.
        clean_lo_idx = fx.min(clean_lo_idx + fx.Int32(1), clean_hi_idx)

    @flyc.jit
    def _run_tiles(state, lo_i32, hi_i32, *, mask_left, mask_right, kv_len):
        final_state = state
        for tile, carried in range(fx.Index(lo_i32), fx.Index(hi_i32), 1, init=state):
            next_state = main_loop(
                fx.Int32(tile),
                list(carried),
                mask_left=mask_left,
                mask_right=mask_right,
                kv_len=kv_len,
            )
            final_state = yield next_state
        return final_state

    def _busy_tiles(state):
        if mask_left:
            state = _run_tiles(
                state,
                start_tile_idx,
                clean_lo_idx,
                mask_left=mask_left,
                mask_right=mask_right,
                kv_len=None,
            )
        state = _run_tiles(
            state,
            clean_lo_idx,
            clean_hi_idx,
            mask_left=None,
            mask_right=None,
            kv_len=None,
        )
        state = _run_tiles(
            state,
            clean_hi_idx,
            fx.Int32(num_tiles) + fx.Int32(1),
            mask_left=mask_left,
            mask_right=mask_right,
            kv_len=kv_len,
        )
        return state

    if IDLE_WAVES and paged is None and qk_hdim < 256:
        # IDLE_WAVES: a wave whose whole 16*num_q_tiles-row Q span lies past q_len (the tail of a
        # partial m-tile) skips the gemm/softmax entirely and only keeps its producer duty
        # (its K or V band copy) and the per-body barriers -- same barrier count, same slot
        # rotation, state left at the seed. Its SIMD partner then has the issue port alone.
        def idle_loop(tile_idx, state):
            slot_of = [fx.Int32(state[_slot_bases_idx + i]) for i in range(N_KV_PP)]
            write_slot = slot_of[N_KV_PP - 1]
            prefetch_tile_idx = tile_idx + fx.Int32(1)
            kv_state_prev = list(
                state[_kv_state_idx : _kv_state_idx + kv_policy.num_state_slots]
            )
            desc = _get_kv_desc(
                write_slot,
                _tile_row0(prefetch_tile_idx),
                _kv_valid(prefetch_tile_idx),
                kv_policy.indices(kv_state_prev),
            )
            kv_state_next = kv_policy.advance(prefetch_tile_idx)
            # qk_hdim < 256 only: no in-loop phase wall, one fence per body per half.
            if _is_lagging_softmax:
                desc.async_load()
                rocdl.sched_barrier(0)
                _copy_fence(*_kv_drain, num_dscnt=_num_head_carry)
            else:
                _copy_fence(*_kv_drain, num_dscnt=_num_head_carry)
                desc.async_load()
                rocdl.sched_barrier(0)
            out = list(state[:_k_bases_idx])
            rotation = list(range(1, N_KV_PP)) + [0]
            for i in rotation:
                out += list(
                    state[
                        _k_bases_idx
                        + i * _num_k_bases : _k_bases_idx
                        + (i + 1) * _num_k_bases
                    ]
                )
            for i in rotation:
                out += list(
                    state[
                        _v_bases_idx
                        + i * _num_v_bases : _v_bases_idx
                        + (i + 1) * _num_v_bases
                    ]
                )
            out += [_raw(slot_of[i]) for i in rotation]
            out += list(state[_head_carry_idx : _head_carry_idx + _num_head_carry])
            out += [_raw(v) for v in kv_state_next]
            return out

        @flyc.jit
        def _idle_tiles(state, lo_i32, hi_i32):
            final_state = state
            for tile, carried in range(
                fx.Index(lo_i32), fx.Index(hi_i32), 1, init=state
            ):
                next_state = idle_loop(fx.Int32(tile), list(carried))
                final_state = yield next_state
            return final_state

        _wave_row0 = (
            m_tile_idx * fx.Int32(BLOCK_M) + warp_idx * fx.Int32(WMMA_M * num_q_tiles)
        ) // fx.Int32(gqa_ratio)

        # The ds_load base pointers (llvm.ptr<3>) cannot cross the scf.if: they are not
        # read after the loop, so only the rest of the state is yielded out of it.
        def _strip_ptrs(s):
            return list(s[:_k_bases_idx]) + list(s[_slot_bases_idx:])

        @flyc.jit
        def _busy_or_idle(busy):
            final_state = _strip_ptrs(_init)
            if busy:
                final_state = _strip_ptrs(_busy_tiles(_init))
            else:
                final_state = _strip_ptrs(
                    _idle_tiles(
                        _init, start_tile_idx, fx.Int32(num_tiles) + fx.Int32(1)
                    )
                )
            return final_state

        _nonptr = list(_busy_or_idle(_wave_row0 < q_len))
        final = (
            _nonptr[:_k_bases_idx]
            + list(_init[_k_bases_idx:_slot_bases_idx])
            + _nonptr[_k_bases_idx:]
        )
    else:
        final = _busy_tiles(_init)
    if not _is_lagging_softmax:
        # Partner for the lagging half's prologue filler -- without it the two halves
        # disagree on the barrier count. It also orders this half's O stores against its
        # OWN last K reads: this half's last in-loop barrier is the PHASE one, which sits
        # BEFORE the gemm, so wave 1 could otherwise store O into final[0]'s K chunk while
        # wave 0 is still reading K out of it.
        _bare_barrier()

    _o_free_slots = [
        fx.Int32(final[_slot_bases_idx + 2]),
        fx.Int32(final[_slot_bases_idx + 0]),
    ]
    o_lds_warp = (
        ((warp_idx & fx.Int32(1)) > fx.Int32(0)).select(
            _o_free_slots[1], _o_free_slots[0]
        )
        + ((warp_idx >> fx.Int32(1)) & fx.Int32(1)) * fx.Int32(LDS_QO_BYTES)
        + (warp_idx >> fx.Int32(2)) * fx.Int32(_SPLIT_STRIDE)
    )
    # The trailing body's dead clamped tile copies are still in flight. HI's targets
    # final[1], a V chunk O never touches; LO's two K copies land in final[2] and final[1],
    # and final[2] IS an O chunk -- so the drain needs a rendezvous behind it, which
    # ``_attention_epilogue`` supplies, since a wave only retires its own copies and O's
    # chunk halves are shared by wave pairs.
    _copy_wait(*_kv_drain)
    _o_fin = [
        _o_unpack(fx.Vector(final[q_tile_idx * _q_state_slots + 2]), d_tiles)
        for q_tile_idx in range(num_q_tiles)
    ]
    if _WS:
        _d_fin = [
            fx.Float32(fx.Vector(final[q * _q_state_slots + 1])[0])
            for q in range(num_q_tiles)
        ]
    else:
        _d_fin = [
            fx.Float32(final[q_tile_idx * _q_state_slots + 1])
            for q_tile_idx in range(num_q_tiles)
        ]
    return _epi_state(
        m_list=[
            fx.Float32(final[q_tile_idx * _q_state_slots + 0])
            for q_tile_idx in range(num_q_tiles)
        ],
        d_list=_d_fin,
        o_list=[o[:d_tiles] for o in _o_fin],
        o_lds_warp=o_lds_warp,
    )


def _attention_core_one_kv_tile(
    *,
    qk_hdim,
    v_hdim,
    n_block,
    mask_left,
    mask_right,
    return_lse,
    has_sink,
    gqa_ratio,
    num_q_tiles_per_wave,
    ptr_O,
    ptr_Q,
    ptr_K,
    ptr_V,
    ptr_LSE,
    ptr_sink,
    softmax_scale,
    stride_q_seq,
    stride_k_seq,
    stride_v_seq,
    stride_o_seq,
    stride_q_head,
    stride_k_head,
    stride_v_head,
    stride_o_head,
    stride_lse_seq,
    stride_lse_head,
    lse_base_elems,
    lse_num_records_bytes,
    q_start,
    q_len,
    kv_start,
    kv_len,
    window_left,
    window_right,
    warp_idx,
    lds_base,
    q_dtype,
    kv_dtype,
    paged=None,  # never paged (paged uses the multi-tile core); here for one kwarg dict
    v_descale=None,  # the epilogue's; here so one kwarg dict serves both
):
    """Whole-attention compute for a workgroup whose band fits in ONE KV tile.

    Warp specialization buys nothing here: with a single tile there is no next tile to
    prefetch, so there is nothing for a producer half to run ahead on and no PV/QK
    ping-pong to anti-phase. All 8 waves run this one trace -- QK, softmax, PV -- and the
    only thing the wave's half still decides is which of K or V it copies from VRAM, a
    runtime branch rather than a second trace.

    Everything the pipelined core needs for a tile stream is gone: no loop, no slot
    rotation, no carried P/S hand-off, no clamped prefetch, no dead half-bodies. The one
    tile lands in physical slot 0 while Q sits in slot 1 and O stages through slots 1|2,
    so the tile is never overwritten and the whole body needs exactly ONE barrier -- the
    fence that publishes the tile.
    """
    lane_idx = _lane_id()
    kv_head, q_head_idx, seq_idx = _packed_tile_indices(
        gqa_ratio, warp_idx, lane_idx, num_q_tiles_per_wave
    )
    # 8-bit Q cannot carry the scale (no f8 multiply), so softmax applies it to S.
    _q_scale = None
    _s_scale = softmax_scale * fx.Float32(LOG2E)
    num_q_tiles = num_q_tiles_per_wave

    q_mgr, k_mgr, v_mgr = _build_managers(
        qk_hdim=qk_hdim,
        v_hdim=v_hdim,
        n_block=n_block,
        gqa_ratio=gqa_ratio,
        num_q_tiles_per_wave=num_q_tiles,
        q_dtype=q_dtype,
        kv_dtype=kv_dtype,
    )
    _SPLIT_STRIDE = 6 * LDS_CHUNK_BYTES
    slot_bytes = 2 * LDS_CHUNK_BYTES
    for _label, _bytes in (
        ("K", k_mgr.get_lds_size_in_byte()),
        ("V", v_mgr.get_lds_size_in_byte()),
    ):
        assert (
            _bytes % KV_LDS_SPLITS == 0 and _bytes // KV_LDS_SPLITS <= LDS_CHUNK_BYTES
        ), (
            f"{_label} split {_bytes // KV_LDS_SPLITS}B exceeds the {LDS_CHUNK_BYTES}B chunk "
            f"(qk_hdim={qk_hdim}, v_hdim={v_hdim}, n_block={n_block})"
        )
    assert q_mgr.warp_lds_size_in_byte() <= LDS_QO_BYTES

    # The tile is physical slot 0: K at chunks 0|6, V at 1|7.
    k_buf = lds_base
    v_buf = lds_base + fx.Int32(LDS_CHUNK_BYTES)
    assert KV_LDS_SPLITS == 2, "parity order needs a 2-way split"
    _is_odd_warp = warp_idx & fx.Int32(1)
    _read_order_lo = _is_odd_warp * fx.Int32(_SPLIT_STRIDE)
    _read_order = [_read_order_lo, fx.Int32(_SPLIT_STRIDE) - _read_order_lo]
    _kv_swap_delta = _is_odd_warp * fx.Int32(n_block // 2)

    def _split_bufs(b):
        return [b + o for o in _read_order]

    _q_chunk_sel = (warp_idx & fx.Int32(1)) ^ ((warp_idx >> fx.Int32(2)) & fx.Int32(1))
    q_lds_warp = (
        lds_base
        + fx.Int32(slot_bytes)
        + _q_chunk_sel * fx.Int32(_SPLIT_STRIDE)
        + (warp_idx >> fx.Int32(1)) * fx.Int32(LDS_QO_BYTES)
    )
    q_mgr.load_q_to_vgpr_part1(
        ptr_Q=ptr_Q,
        stride_q_seq=stride_q_seq,
        stride_q_head=stride_q_head,
        q_start=q_start,
        q_len=q_len,
        kv_head=kv_head,
        block_x=_m_tile_idx(),
        warp_idx=warp_idx,
        lane_idx=lane_idx,
        ptr_lds_warp=q_lds_warp,
    )

    _, causal_off, kv_len_wg, _ = _wg_kv_span(
        n_block=n_block,
        mask_right=mask_right,
        gqa_ratio=gqa_ratio,
        num_q_tiles_per_wave=num_q_tiles,
        q_len=q_len,
        kv_len=kv_len,
        window_right=window_right,
    )

    # Both descriptors are pure; only the copy one of them issues is warp-dependent.
    _kv_ctx = ProducerCtx(
        producer_warp=warp_idx % fx.Int32(KV_PRODUCER_WARPS),
        num_producer_warps=KV_PRODUCER_WARPS,
        lane_idx=lane_idx,
    )
    k_desc = k_mgr.load_descriptor(
        ptr_lds=[k_buf, k_buf + fx.Int32(_SPLIT_STRIDE)],
        ptr_src=ptr_K,
        stride_seq=stride_k_seq,
        stride_head=stride_k_head,
        head=kv_head,
        row0=kv_start,
        valid=kv_len_wg,
        ctx=_kv_ctx,
    )
    v_desc = v_mgr.load_descriptor(
        ptr_lds=[v_buf, v_buf + fx.Int32(_SPLIT_STRIDE)],
        ptr_src=ptr_V,
        stride_seq=stride_v_seq,
        stride_head=stride_v_head,
        head=kv_head,
        row0=kv_start,
        valid=kv_len_wg,
        ctx=_kv_ctx,
    )

    @flyc.jit
    def _issue_kv(is_lo):
        if is_lo:
            k_desc.async_load()
        else:
            v_desc.async_load()

    _is_lo = warp_idx < fx.Int32(NUM_WAVES // 2)
    _num_tdm_copies = max(k_desc.tensorcnt, v_desc.tensorcnt) or -1
    _num_async_copies = max(k_desc.asynccnt, v_desc.asynccnt) or -1
    if USE_TDM_LOADER:
        # The tile misses Q's chunks entirely, so it goes out before Q is read and its
        # global latency overlaps Q's. Waves differ in how many copies they issued, so
        # part2 drains only down to the count EVERY wave has in flight.
        _issue_kv(_is_lo)
        tdm_ops.tensor_wait(min(k_desc.tensorcnt, v_desc.tensorcnt))
        q_frags = q_mgr.load_q_to_vgpr_part2(scale=_q_scale)
    else:
        # V1 puts Q's own stage on asynccnt and part2 waits a depth counted over Q alone.
        q_frags = q_mgr.load_q_to_vgpr_part2(scale=_q_scale)
        _issue_kv(_is_lo)
    # The only barrier in the body: a wave's counters bound its own copies, so this is
    # what publishes the tile to the other seven.
    _copy_fence(*_kv_drain_depths(_num_tdm_copies, _num_async_copies))

    k_lds = k_mgr.ds_load_ptrs(ptr_lds=_split_bufs(k_buf), lane_idx=lane_idx)
    v_lds = v_mgr.ds_load_ptrs(ptr_lds=_split_bufs(v_buf), lane_idx=lane_idx)

    def _k_emit(j):
        return k_mgr.load_one_to_reg(k_lds, j)

    def _v_emit(j):
        return v_mgr.load_one_to_reg(v_lds, j)

    _, s_list = _pv_qk_gemm(
        v_emit=None,
        k_emit=_k_emit,
        vld_per_frag=v_mgr.loads_per_frag,
        q_dtype=q_dtype,
        kv_dtype=kv_dtype,
        p_list=[[] for _ in range(num_q_tiles)],
        q_frags_list=q_frags,
        v_hdim=v_hdim,
        n_block=n_block,
        warp_type=WarpType.LO,
        o_acc_list=[[] for _ in range(num_q_tiles)],
        phase="qk",
    )

    # PV's ring head goes out before softmax so its LDS latency hides under the VALU.
    _num_prefetch = _ring_num_prefetch(v_mgr.num_ds_loads(), PVQK_RING, PVQK_LAG)
    rocdl.sched_barrier(0)
    pv_head = []
    for j in range(_num_prefetch):
        pv_head.append(_v_emit(j))
        if len(pv_head) % v_mgr.loads_per_frag == 0:
            rocdl.sched_barrier(0)
    rocdl.sched_barrier(0)

    if has_sink:
        num_heads_q = _num_heads_kv() * fx.Int32(gqa_ratio)
        m_prev = [
            _load_sink_logit(ptr_sink, q_head_idx[q_tile_idx], num_heads_q)
            * fx.Float32(LOG2E)
            for q_tile_idx in range(num_q_tiles)
        ]
        d_prev = [fx.Float32(1.0) for _ in range(num_q_tiles)]
    else:
        m_prev = [fx.Float32(BIG_NEG) for _ in range(num_q_tiles)]
        d_prev = [fx.Float32(0.0) for _ in range(num_q_tiles)]
    # This tile is both the first and the last, so it always carries the kv_len tail and
    # O needs no rescale: the seed is 0 and corr*0 == 0 whatever the running max does.
    p_f32, m_new_list, d_new_list, _corr, _masks = _softmax(
        s_list=s_list,
        m_prev_list=m_prev,
        d_prev_list=d_prev,
        lane_idx=lane_idx,
        n_block=n_block,
        kv_pos_base=fx.Int32(0),
        kv_swap_delta=_kv_swap_delta,
        q_max_list=[
            seq_idx[q_tile_idx] + causal_off + window_right if mask_right else None
            for q_tile_idx in range(num_q_tiles)
        ],
        q_min_list=[
            seq_idx[q_tile_idx] + causal_off - window_left if mask_left else None
            for q_tile_idx in range(num_q_tiles)
        ],
        kv_len=kv_len,
        s_scale=_s_scale,
    )
    p_list = _convert_vectors(p_f32, P_DTYPE)
    _keepalive([v for pt in p_list for v in pt])
    o_list, _ = _pv_qk_gemm(
        v_emit=_v_emit,
        k_emit=None,
        vld_per_frag=v_mgr.loads_per_frag,
        q_dtype=q_dtype,
        kv_dtype=kv_dtype,
        p_list=p_list,
        q_frags_list=q_frags,
        v_hdim=v_hdim,
        n_block=n_block,
        warp_type=WarpType.LO,
        o_acc_list=None,
        head=pv_head,
        phase="pv",
    )

    # O stages through physical slots 1 and 2 -- Q's chunks, dead since the prologue, and
    # never touched by the tile in slot 0. Nothing is in flight and no wave can still be
    # reading them, so the epilogue needs neither a drain nor a rendezvous.
    o_lds_warp = (
        (_is_odd_warp > fx.Int32(0)).select(
            lds_base + fx.Int32(2 * slot_bytes), lds_base + fx.Int32(slot_bytes)
        )
        + ((warp_idx >> fx.Int32(1)) & fx.Int32(1)) * fx.Int32(LDS_QO_BYTES)
        + (warp_idx >> fx.Int32(2)) * fx.Int32(_SPLIT_STRIDE)
    )
    return _epi_state(
        m_list=m_new_list, d_list=d_new_list, o_list=o_list, o_lds_warp=o_lds_warp
    )


def _store_o_lse(
    *,
    v_descale,
    o_mgr,
    m_list,
    d_list,
    o_list,
    d_tiles,
    num_q_tiles,
    ptr_O,
    stride_o_seq,
    stride_o_head,
    q_start,
    q_len,
    kv_head,
    m_tile_idx,
    warp_idx,
    lane_idx,
    o_lds_warp,
    seq_idx,
    q_head_idx,
    return_lse,
    ptr_LSE,
    lse_base_elems,
    lse_num_records_bytes,
    stride_lse_seq,
    stride_lse_head,
    carry_state=False,
    out_target=None,
):
    """Normalize, stage and store O (and optionally LSE) for this wave's num_q_tiles q-tiles.

    Shared tail of both compute cores; ``o_lds_warp`` is the caller's choice of which
    finished KV chunk this wave stages through.

    ``carry_state`` (compile-time): O/LSE already hold a running attention state over
    other keys (e.g. earlier context chunks). Read it back, fold this launch's partial
    softmax into it with the standard LSE recombination and write the merged O/LSE in
    place -- the separate merge_attn_states pass, fused. A row whose old LSE is -inf
    has no prior state and its old O is never read (it may be uninitialized).

    ``out_target`` (persistent split launch) redirects the O/LSE STORES at runtime: a
    dict with ``o_iter`` (global base), ``o_base_elems``, ``q_start_o``,
    ``stride_o_seq``, ``stride_o_head``, ``lse_rsrc``, ``lse_base_elems``,
    ``stride_lse_seq``, ``stride_lse_head`` -- already selected between the final output
    and the split workspace by the caller -- plus ``carry_ok`` (runtime bool): a split
    part goes to the workspace raw, so the old state is folded in only when it holds.
    The carried state itself is always READ from the final O/LSE."""
    st_o_iter = None
    st_o_base = fx.Int32(0)
    st_q_start = q_start
    st_stride_o_seq = stride_o_seq
    st_stride_o_head = stride_o_head
    st_lse_rsrc = None
    st_lse_base = lse_base_elems
    st_stride_lse_seq = stride_lse_seq
    st_stride_lse_head = stride_lse_head
    carry_ok = None
    if out_target is not None:
        st_o_iter = out_target["o_iter"]
        st_o_base = out_target["o_base_elems"]
        st_q_start = out_target["q_start_o"]
        st_stride_o_seq = out_target["stride_o_seq"]
        st_stride_o_head = out_target["stride_o_head"]
        st_lse_rsrc = out_target["lse_rsrc"]
        st_lse_base = out_target["lse_base_elems"]
        st_stride_lse_seq = out_target["stride_lse_seq"]
        st_stride_lse_head = out_target["stride_lse_head"]
        carry_ok = out_target.get("carry_ok")
    lse_new_list = None
    if carry_state:
        assert return_lse, "carry_state needs the LSE buffer"
        lse_new_list = []
        khalf = lane_idx // fx.Int32(
            WMMA_M
        )  # which 8-column half of each 16-col d-tile
        _neg_inf = fx.Float32(float("-inf"))
        _zero = fx.Float32(0.0)
        lse_ld_rsrc = buffer_ops.create_buffer_resource(
            ptr_LSE, num_records_bytes=lse_num_records_bytes
        )
        # i64 bound, as in _zero_fill_attention; offsets below are i32 bytes.
        o_ld_rsrc = buffer_ops.create_buffer_resource(
            ptr_O,
            num_records_bytes=fx.Int64(q_start + q_len)
            * fx.Int64(stride_o_seq)
            * fx.Int64(2),
        )
    for q_tile_idx in range(num_q_tiles):
        # Normalize this q-tile's O by its running denom d, then reshape+store to VRAM.
        # o_final[d_tile_idx] lane l elem si = sum_kv P[q,kv] V[kv, d_tile_idx*16+(l//16)*8+si]
        # (unnormalized); divide by the per-query denom d (peer-consistent across the
        # lane pair) to finish softmax. OManager masks rows with seq >= q_len.
        d_final = d_list[q_tile_idx]
        o_final = o_list[q_tile_idx]
        if carry_state:
            m_final = m_list[q_tile_idx]
            row_ok = seq_idx[q_tile_idx] < q_len
            lse_off_el = (
                lse_base_elems
                + seq_idx[q_tile_idx] * stride_lse_seq
                + q_head_idx[q_tile_idx] * stride_lse_head
            )
            lse_old = fx.Float32(
                buffer_ops.buffer_load(
                    lse_ld_rsrc,
                    lse_off_el,
                    vec_width=1,
                    dtype=fx.Float32,
                    mask=row_ok,
                    cache_modifier=CARRY_LOAD_CPOL,
                )
            )
            # Everything in log2 units, where m lives.
            lse_old2 = lse_old * fx.Float32(LOG2E)
            has_old = lse_old2 > _neg_inf
            if carry_ok is not None:
                has_old = has_old & carry_ok
            has_cur = d_final > _zero
            lse_cur2 = fx.Float32(has_cur.select(m_final + fx.log2(d_final), _neg_inf))
            mx = fx.Float32((lse_cur2 > lse_old2).select(lse_cur2, lse_old2))
            both_empty = mx == _neg_inf
            safe_mx = fx.Float32(both_empty.select(_zero, mx))
            a_old = fx.Float32(has_old.select(fx.exp2(lse_old2 - safe_mx), _zero))
            # Weight of the UNNORMALIZED o_final: O_cur * exp2(lse_cur2 - mx) with
            # O_cur = o_final / d collapses to o_final * exp2(m - mx).
            e_cur = fx.Float32(has_cur.select(fx.exp2(m_final - safe_mx), _zero))
            tot = a_old + e_cur * d_final
            inv_tot = fx.Float32((tot > _zero).select(fx.Float32(1.0) / tot, _zero))
            scale_cur = e_cur * inv_tot
            if v_descale is not None:
                scale_cur = scale_cur * v_descale
            scale_old = a_old * inv_tot
            lse_new_list.append(
                both_empty.select(
                    _neg_inf, (safe_mx + fx.log2(tot)) * fx.Float32(1.0 / LOG2E)
                )
            )
            # Old O: this lane's 8 contiguous bf16 per d-tile, one b128 each. Rows with
            # no prior state (or past q_len) load nothing (OOB -> 0), so garbage never
            # meets a zero weight as NaN.
            o_row_el = (
                (q_start + seq_idx[q_tile_idx]) * stride_o_seq
                + q_head_idx[q_tile_idx] * stride_o_head
                + khalf * fx.Int32(8)
            )
            ld_ok = row_ok & has_old
            sc_cur_vec = fx.Vector.from_elements([scale_cur], fx.Float32).broadcast_to(
                8
            )
            sc_old_vec = fx.Vector.from_elements([scale_old], fx.Float32).broadcast_to(
                8
            )
            o_norm = []
            for d_tile_idx in range(d_tiles):
                raw = buffer_ops.buffer_load(
                    o_ld_rsrc,
                    (o_row_el + fx.Int32(d_tile_idx * WMMA_M)) // fx.Int32(2),
                    vec_width=4,
                    dtype=fx.Int32,
                    mask=ld_ok,
                    cache_modifier=CARRY_LOAD_CPOL,
                )
                o_old = fx.Vector(raw, (4,), fx.Int32).bitcast(O_DTYPE).to(fx.Float32)
                o_norm.append(o_final[d_tile_idx] * sc_cur_vec + o_old * sc_old_vec)
        else:
            # Fully-masked row (d_final==0): 1/0=inf, o_final=0, 0*inf=NaN -> guard to O=0.
            inv_denom = (d_final > fx.Float32(0.0)).select(
                fx.Float32(1.0) / d_final, fx.Float32(0.0)
            )
            if v_descale is not None:
                inv_denom = inv_denom * v_descale
            inv_denom_vec = fx.Vector.from_elements(
                [inv_denom], fx.Float32
            ).broadcast_to(8)
            o_norm = [
                o_final[d_tile_idx] * inv_denom_vec for d_tile_idx in range(d_tiles)
            ]
        if q_tile_idx > 0:
            rocdl.s_wait_dscnt(0)  # drain prev q-tile's O ring/DS ops before reuse
        o_mgr.store_o_to_vram(
            ptr_O=ptr_O,
            o_base_elems=st_o_base,
            stride_o_seq=st_stride_o_seq,
            stride_o_head=st_stride_o_head,
            q_start=st_q_start,
            q_len=q_len,
            kv_head=kv_head,
            block_x=m_tile_idx,
            warp_idx=warp_idx,
            lane_idx=lane_idx,
            ptr_lds_warp=o_lds_warp,
            o_frags=o_norm,
            qtile=q_tile_idx,
            o_iter=st_o_iter,
        )

    # ---- LSE store (optional). LSE = (m_final + log2(d_final)) / LOG2E: m is carried in
    # log2 units of the scaled score and d is domain-free, so one multiply converts both,
    # matching torch.logsumexp(scale * Q @ K^T, dim=kv). The lane pair (l, l^16) holds
    # each query identically, so store once from the kv_half_idx==0 lanes, masked by
    # seq < q_len; buffer_store redirects mask-drops to byte 0x7FFFFFFF.
    if return_lse:
        khalf0 = (lane_idx // fx.Int32(WMMA_M)) == fx.Int32(0)
        if st_lse_rsrc is not None:
            lse_rsrc = st_lse_rsrc
        else:
            lse_rsrc = buffer_ops.create_buffer_resource(
                ptr_LSE, num_records_bytes=lse_num_records_bytes
            )
        for q_tile_idx in range(num_q_tiles):
            m_final = m_list[q_tile_idx]
            d_final = d_list[q_tile_idx]
            # fx.log2 lowers to the HW v_log_f32 (base-2), matching m's log2 domain.
            if carry_state:
                lse_val = lse_new_list[q_tile_idx]
            else:
                lse_val = (m_final + fx.log2(d_final)) * fx.Float32(1.0 / LOG2E)
            lse_mask = khalf0 & (seq_idx[q_tile_idx] < q_len)
            lse_off_el = (
                st_lse_base
                + seq_idx[q_tile_idx] * st_stride_lse_seq
                + q_head_idx[q_tile_idx] * st_stride_lse_head
            )
            # Pre-mask the offset (OOB rows -> 0x7fffffff) and pass mask=None so the
            # store maps 1:1 to a single buffer_store with masking already SSA-visible.
            lse_off_masked = lse_mask.select(
                lse_off_el * fx.Int32(4), fx.Int32(0x7FFFFFFF)
            )
            buffer_ops.buffer_store(
                lse_val, lse_rsrc, lse_off_masked, mask=None, offset_is_bytes=True
            )


# Keys ``_attention_epilogue`` shares with the compute cores' kwarg dict.
_EPILOGUE_KEYS = (
    "v_hdim",
    "gqa_ratio",
    "num_q_tiles_per_wave",
    "return_lse",
    "ptr_O",
    "ptr_LSE",
    "stride_o_seq",
    "stride_o_head",
    "stride_lse_seq",
    "stride_lse_head",
    "lse_base_elems",
    "lse_num_records_bytes",
    "q_start",
    "q_len",
    "v_descale",
)


def _epi_state_zeros(*, num_q_tiles_per_wave, v_hdim):
    """Type exemplar for the warp-type scf.if. Both branches overwrite it, so the
    constants are dead; the rewriter needs the names bound before the branch."""
    d_tiles = v_hdim // WMMA_M
    num_q_tiles = num_q_tiles_per_wave
    return _epi_state(
        m_list=[fx.Float32(0.0) for _ in range(num_q_tiles)],
        d_list=[fx.Float32(0.0) for _ in range(num_q_tiles)],
        o_list=[
            [fx.Vector.filled(8, 0.0, fx.Float32) for _ in range(d_tiles)]
            for _ in range(num_q_tiles)
        ],
        o_lds_warp=fx.Int32(0),
    )


def _attention_epilogue(
    *,
    v_descale,
    epi_state,
    rendezvous,  # compile-time: the multi core leaves KV copies in O's chunk
    warp_idx,
    v_hdim,
    gqa_ratio,
    num_q_tiles_per_wave,
    return_lse,
    ptr_O,
    ptr_LSE,
    stride_o_seq,
    stride_o_head,
    stride_lse_seq,
    stride_lse_head,
    lse_base_elems,
    lse_num_records_bytes,
    q_start,
    q_len,
    carry_state=False,
    out_target=None,
):
    """The O/LSE tail, traced once per build instead of once per warp type.

    Every core hands back the same flat state -- [m, d, O_0..O_{d_tiles-1}] per q-tile
    plus the LDS chunk this wave stages O through -- and everything else here is a pure
    function of the launch, so the warp-type scf.if can close before it."""
    d_tiles = v_hdim // WMMA_M
    num_q_tiles = num_q_tiles_per_wave
    _q_epi_slots = 2 + d_tiles
    lane_idx = _lane_id()
    kv_head, q_head_idx, seq_idx = _packed_tile_indices(
        gqa_ratio, warp_idx, lane_idx, num_q_tiles
    )
    if rendezvous:
        # O's chunk halves are shared by wave pairs and still hold the trailing dead KV
        # copies; each half drained only its own, so the pair meets before the ds_writes.
        _bare_barrier()
    _store_o_lse(
        v_descale=v_descale,
        o_mgr=_o_manager(
            v_hdim=v_hdim,
            gqa_ratio=gqa_ratio,
            num_q_tiles_per_wave=num_q_tiles,
            elem_dtype=O_DTYPE,
        ),
        m_list=[
            fx.Float32(epi_state[q_tile_idx * _q_epi_slots + 0])
            for q_tile_idx in range(num_q_tiles)
        ],
        d_list=[
            fx.Float32(epi_state[q_tile_idx * _q_epi_slots + 1])
            for q_tile_idx in range(num_q_tiles)
        ],
        o_list=[
            [
                fx.Vector(epi_state[q_tile_idx * _q_epi_slots + 2 + d_tile_idx])
                for d_tile_idx in range(d_tiles)
            ]
            for q_tile_idx in range(num_q_tiles)
        ],
        d_tiles=d_tiles,
        num_q_tiles=num_q_tiles,
        ptr_O=ptr_O,
        stride_o_seq=stride_o_seq,
        stride_o_head=stride_o_head,
        q_start=q_start,
        q_len=q_len,
        kv_head=kv_head,
        m_tile_idx=_m_tile_idx(),
        warp_idx=warp_idx,
        lane_idx=lane_idx,
        o_lds_warp=fx.Int32(epi_state[num_q_tiles * _q_epi_slots]),
        seq_idx=seq_idx,
        q_head_idx=q_head_idx,
        return_lse=return_lse,
        ptr_LSE=ptr_LSE,
        lse_base_elems=lse_base_elems,
        lse_num_records_bytes=lse_num_records_bytes,
        stride_lse_seq=stride_lse_seq,
        stride_lse_head=stride_lse_head,
        carry_state=carry_state,
        out_target=out_target,
    )


def _zero_fill_attention(
    *,
    v_hdim,
    gqa_ratio,
    num_q_tiles_per_wave,
    return_lse,
    has_sink,
    ptr_sink,
    ptr_O,
    ptr_LSE,
    stride_o_seq,
    stride_o_head,
    stride_lse_seq,
    stride_lse_head,
    lse_num_records_bytes,
    q_start,
    q_len,
):
    """q_len>0 with kv_len==0 (cross-attention): softmax over an empty KV set, so O=0 for
    this WG's valid query rows and LSE=-inf -- or LSE=sink[head] with a sink, exp(sink)
    being the only surviving term. Flat coalesced b128 write, no WMMA layout."""
    BLOCK_M = _block_m(num_q_tiles_per_wave)
    thread_idx = _warp_id() * fx.Int32(WAVE_SIZE) + _lane_id()
    kv_head = _kv_head_idx()
    row0 = _m_tile_idx() * fx.Int32(BLOCK_M)
    g = fx.Int32(gqa_ratio)
    _chunk_elems = 8  # bf16 per b128 store
    num_chunks_per_row = v_hdim // _chunk_elems  # b128 chunks per O row

    # i64: an i32 product (large total_q * stride) can overflow negative, then the
    # descriptor sign-extends it to a huge bound, defeating the 0x7FFFFFFF OOB drop.
    o_num_records_bytes = (
        fx.Int64(q_start + q_len) * fx.Int64(stride_o_seq) * fx.Int64(2)
    )
    o_rsrc = buffer_ops.create_buffer_resource(
        ptr_O, num_records_bytes=o_num_records_bytes
    )
    zero_o = fx.Vector.filled(_chunk_elems, 0.0, O_DTYPE)
    for r in range(BLOCK_M * num_chunks_per_row // BLOCK_SIZE):
        chunk_idx = (
            fx.Int32(r * BLOCK_SIZE) + thread_idx
        )  # flat b128-chunk index this round
        packed_row_idx = row0 + chunk_idx // fx.Int32(num_chunks_per_row)
        d = (chunk_idx % fx.Int32(num_chunks_per_row)) * fx.Int32(_chunk_elems)
        seq = packed_row_idx // g
        head = kv_head * g + packed_row_idx % g
        off = (q_start + seq) * stride_o_seq + head * stride_o_head + d
        off_masked = (seq < q_len).select(off * fx.Int32(2), fx.Int32(0x7FFFFFFF))
        buffer_ops.buffer_store(
            zero_o, o_rsrc, off_masked, mask=None, offset_is_bytes=True
        )

    if return_lse:
        lse_rsrc = buffer_ops.create_buffer_resource(
            ptr_LSE, num_records_bytes=lse_num_records_bytes
        )
        packed_row_idx = row0 + thread_idx  # one LSE per packed row, one row per thread
        seq = packed_row_idx // g
        head = kv_head * g + packed_row_idx % g
        if has_sink:
            num_heads_q = _num_heads_kv() * g
            lse_val = _load_sink_logit(ptr_sink, head, num_heads_q)
        else:
            lse_val = fx.Float32(float("-inf"))
        off = (q_start + seq) * stride_lse_seq + head * stride_lse_head
        off_masked = (seq < q_len).select(off * fx.Int32(4), fx.Int32(0x7FFFFFFF))
        if BLOCK_M < BLOCK_SIZE:
            off_masked = (thread_idx < fx.Int32(BLOCK_M)).select(
                off_masked, fx.Int32(0x7FFFFFFF)
            )
        buffer_ops.buffer_store(
            lse_val, lse_rsrc, off_masked, mask=None, offset_is_bytes=True
        )


# ============================================================================
# Builder — one device kernel per (layout, config)
# ============================================================================


def kv_split_bytes(qk_hdim, v_hdim, n_block, kv_dtype):
    """Bytes one half of a 2-way-split K and V tile occupies, straight from the active
    managers -- n_block//KV_LDS_SPLITS rows of hdim*sizeof(elem) plus their per-row pad
    (16 B for K, 32 B for V)."""
    k_cls = KManagerV2 if USE_TDM_LOADER else KManagerV1
    v_cls = VManagerV2 if USE_TDM_LOADER else VManagerV1
    k_mgr = k_cls(qk_hdim=qk_hdim, n_block=n_block, elem_dtype=kv_dtype)
    v_mgr = v_cls(v_hdim=v_hdim, n_block=n_block, elem_dtype=kv_dtype)
    return (
        k_mgr.get_lds_size_in_byte() // KV_LDS_SPLITS,
        v_mgr.get_lds_size_in_byte() // KV_LDS_SPLITS,
    )


def pick_n_block(qk_hdim, v_hdim, kv_dtype):
    """Widest n_block from N_BLOCK_PREF whose split K and V tiles each still fit one
    LDS_CHUNK_BYTES chunk.

    The 16-bit kernel also caps wide n_block at N_BLOCK_WIDE_MAX_QK_HDIM, a capacity
    proxy that does not apply here: 8-bit halves the bytes, and n_block 64 is not merely
    narrow but invalid, since a contraction tile is 64 kv deep and each of the
    KV_LDS_SPLITS splits must hold whole tiles."""
    for candidate_n_block in N_BLOCK_PREF:
        if (
            max(kv_split_bytes(qk_hdim, v_hdim, candidate_n_block, kv_dtype))
            <= LDS_CHUNK_BYTES
        ):
            return candidate_n_block
    raise ValueError(
        f"no n_block in {N_BLOCK_PREF} fits the {LDS_CHUNK_BYTES}B chunk "
        f"(qk_hdim={qk_hdim}, v_hdim={v_hdim})"
    )


# ============================================================================
# Persistent split-KV launch ("thd_ps")
#
# The varlen grid (m-tiles x heads x batch) leaves most of the GPU idle on the MLA
# chunked-prefix calls: every request is in the batch, but only the 1-3 whose cached
# prefix overlaps the chunk have keys, so the work is a few hundred (m-tile, head)
# units of ~256 KV tiles each on 256 CUs -- 1.1 to 2 rounds, with the second mostly
# empty -- while thousands of empty or zero-fill workgroups each claim a CU (the LDS
# footprint is one WG per CU) just to exit.
#
# Instead: one WG per CU loops over a work list decoded in-kernel from cu_seqlens.
# A work item is (batch, m-tile, head, KV part). For a non-causal call each unit's KV
# range is cut into equal parts so the item count is ~PS_ITEMS_PER_WG per WG; a part
# writes its normalized O (bf16) and LSE to a workspace slot, and a small merge pass
# combines the parts by LSE. Units of batches with no keys are zero-fill items.
# ============================================================================

# Cache policy of the carry_state reads of the running O/LSE (0 = default; 16 = device
# scope). Kept as a knob: the persistent launch maps units to WGs (and XCDs) per call.
CARRY_LOAD_CPOL = int(os.environ.get("FMHA_CARRY_LOAD_CPOL", "0"))

# Waves whose whole Q span lies past q_len skip gemm + softmax (producer duty only).
# AITER_FMHA_A8W8_IDLE_WAVES=0 makes them compute the dead rows as before.
IDLE_WAVES = os.environ.get("AITER_FMHA_A8W8_IDLE_WAVES", "1") != "0"
PS_MAX_BATCH = (
    32  # cu_seqlens entries decoded in SGPRs (unrolled); larger batches use the grid
)
PS_MAX_PARTS = 16  # most KV parts per unit
PS_ITEM_OVH_TILES = (
    6  # per-item fixed cost (prologue + epilogue + fence) in KV-tile units
)
PS_MIN_PART_TILES = 16  # never cut a unit's KV below this many n_block tiles
PS_WS_ITEMS = (
    4096  # workspace capacity in split items (256 x v_hdim bf16 + 256 fp32 each)
)
PS_META_HDR = 8  # int32 header words of the merge table (word 0 = split-unit count)
PS_META_STRIDE = 8  # int32 words per split-unit entry


def _ps_cdiv(a, b):
    return (a + b - fx.Int32(1)) // b


def _ps_cdiv_pow2(a, b_const):
    """ceil(a / b) for a compile-time power-of-two b, a >= 0 (shift, no division)."""
    sh = int(b_const).bit_length() - 1
    assert (1 << sh) == b_const
    return (a + fx.Int32(b_const - 1)) >> fx.Int32(sh)


def _wave_sum(v):
    """All-lane sum of a per-lane Int32 (wave32 butterfly); uniform result in an SGPR."""
    for s in (16, 8, 4, 2, 1):
        v = v + fx.Int32(fx.shuffle_xor(v, s, WAVE_SIZE))
    return fx.Int32(rocdl.readfirstlane(T.i32, _raw(v)))


def _wave_excl_scan(v, lane):
    """Exclusive prefix sum over lanes (Hillis-Steele with shuffle_up)."""
    x = v
    for s in (1, 2, 4, 8, 16):
        y = fx.Int32(fx.shuffle_up(x, s, WAVE_SIZE))
        x = x + (lane >= fx.Int32(s)).select(y, fx.Int32(0))
    return x - v


def _ps_lane_geom(*, ptr_cu_q, ptr_cu_k, b, B, H, gqa, block_m, n_block, scalar=False):
    """Batch ``b``'s (q0, ql, k0, kl, mt, kt, units, attn); ``b`` may be per-lane (VGPR
    loads) or wave-uniform (scalar loads). Batches >= B read as empty."""
    valid = b < B
    nrec = (B + fx.Int32(1)) * fx.Int32(4)
    rq = buffer_ops.create_buffer_resource(ptr_cu_q, num_records_bytes=nrec)
    rk = buffer_ops.create_buffer_resource(ptr_cu_k, num_records_bytes=nrec)

    def _ld(r, i):
        return fx.Int32(
            buffer_ops.buffer_load(r, i, vec_width=1, dtype=T.i32, is_scalar=scalar)
        )

    q0 = _ld(rq, b)
    q1 = _ld(rq, b + fx.Int32(1))
    k0 = _ld(rk, b)
    k1 = _ld(rk, b + fx.Int32(1))
    ql = fx.max(valid.select(q1 - q0, fx.Int32(0)), fx.Int32(0))
    kl = fx.max(valid.select(k1 - k0, fx.Int32(0)), fx.Int32(0))
    mt = _ps_cdiv_pow2(ql * fx.Int32(gqa), block_m)
    kt = _ps_cdiv_pow2(kl, n_block)
    units = mt * H
    attn = kt > fx.Int32(0)
    return {
        "q0": q0,
        "ql": ql,
        "k0": k0,
        "kl": kl,
        "mt": mt,
        "kt": kt,
        "units": units,
        "attn": attn,
    }


def _ps_parts(attn, kt, pt):
    return fx.max(attn.select(_ps_cdiv(kt, pt), fx.Int32(1)), fx.Int32(1))


def _ps_setup(
    *,
    ptr_cu_q,
    ptr_cu_k,
    B,
    H,
    G,
    gqa,
    block_m,
    n_block,
    allow_split,
    ws_cap,
    force_pt=None,
):
    """Work-list geometry, one lane per batch (B <= 32), every wave redundantly -- so only
    a few per-lane VGPRs and SGPRs stay live across the item loop.

    A unit is one (batch, m-tile, head). An attention unit (its batch has keys) is cut into
    ``parts`` equal KV ranges; the items of a batch are ordered (head, m-tile, part) -- a
    head's m-tiles, which read the same K/V, run in the same round. ``parts`` is chosen
    per batch as if the batch ran alone: argmin over p of
    ceil(units*p / G) * (ceil(kt/p) + PS_ITEM_OVH_TILES). Batches are then laid out in
    decreasing item length, so a partially filled round is topped up with shorter items
    and round-robin WG w=0 carries the longest item of every round. Units of batches
    without keys are zero-fill units in their own id space (after all attention items).
    """
    lane = _lane_id()
    e = _ps_lane_geom(
        ptr_cu_q=ptr_cu_q,
        ptr_cu_k=ptr_cu_k,
        b=lane,
        B=B,
        H=H,
        gqa=gqa,
        block_m=block_m,
        n_block=n_block,
    )
    units, kt, attn = e["units"], e["kt"], e["attn"]
    a_units = attn.select(units, fx.Int32(0))
    z_units = attn.select(fx.Int32(0), units)
    kt1 = fx.max(kt, fx.Int32(1))
    parts = fx.Int32(1)
    if allow_split:
        best = _ps_cdiv(a_units, G) * (kt1 + fx.Int32(PS_ITEM_OVH_TILES))
        for pp in range(2, PS_MAX_PARTS + 1):
            tpp_p = _ps_cdiv(kt1, fx.Int32(pp))
            c = _ps_cdiv(a_units * fx.Int32(pp), G) * (
                tpp_p + fx.Int32(PS_ITEM_OVH_TILES)
            )
            ok = (c < best) & (tpp_p >= fx.Int32(PS_MIN_PART_TILES))
            parts = ok.select(fx.Int32(pp), parts)
            best = ok.select(c, best)
        if force_pt is not None:
            parts = (force_pt > fx.Int32(0)).select(
                fx.max(_ps_cdiv(kt1, fx.max(force_pt, fx.Int32(1))), fx.Int32(1)), parts
            )
        parts = attn.select(parts, fx.Int32(1))
        # Workspace overflow: fall back to no split anywhere.
        n_split = _wave_sum((parts > fx.Int32(1)).select(a_units * parts, fx.Int32(0)))
        parts = (n_split > ws_cap).select(fx.Int32(1), parts)
    items = a_units * parts
    tpp = attn.select(_ps_cdiv(kt1, parts), fx.Int32(-1))
    valid = lane < B
    key = valid.select(tpp, fx.Int32(-2))
    # Rank batches by decreasing item length (ties: lower batch first); item_base is the
    # exclusive prefix of items in rank order.
    item_base = fx.Int32(0)
    for l in range(WAVE_SIZE):
        kl_ = fx.Int32(fx.shuffle_idx(key, l, WAVE_SIZE))
        il_ = fx.Int32(fx.shuffle_idx(items, l, WAVE_SIZE))
        before = (kl_ > key) | ((kl_ == key) & (fx.Int32(l) < lane))
        item_base = item_base + before.select(il_, fx.Int32(0))
    split = parts > fx.Int32(1)
    split_items = split.select(items, fx.Int32(0))
    split_units = split.select(a_units, fx.Int32(0))
    return {
        "N": _wave_sum(items),
        "ZU": _wave_sum(z_units),
        "n_split_units": _wave_sum(split_units),
        "parts": parts,
        "item_base": item_base,
        "item_end": item_base + items,
        "ws_base": _wave_excl_scan(split_items, lane),
        "unit_base": _wave_excl_scan(split_units, lane),
        "zero_base": _wave_excl_scan(z_units, lane),
    }


def _ps_find(base_vgpr, x, B):
    """The batch (lane) owning id ``x`` in a lane-order exclusive-scan id space: the last
    lane whose base <= x (bases never decrease; an empty batch shares its successor's base
    and loses to it). Returns (batch, base) as SGPRs."""
    lane = _lane_id()
    own = (base_vgpr <= x) & (lane < B)
    mask = fx.Int32(rocdl.ballot(fx.Int32.ir_type, own))
    b = fx.Int32(fx.ctpop(mask)) - fx.Int32(1)
    base = fx.Int32(rocdl.readlane(T.i32, _raw(base_vgpr), _raw(b)))
    return b, base


def _ps_find_range(base_vgpr, end_vgpr, x, B):
    """The batch (lane) whose [base, end) holds ``x`` (ranges disjoint, any lane order)."""
    lane = _lane_id()
    own = (base_vgpr <= x) & (x < end_vgpr) & (lane < B)
    mask = fx.Int32(rocdl.ballot(fx.Int32.ir_type, own))
    b = fx.Int32(fx.cttz(mask))
    base = fx.Int32(rocdl.readlane(T.i32, _raw(base_vgpr), _raw(b)))
    return b, base


def _ps_item(
    st, it, *, zero, ptr_cu_q, ptr_cu_k, B, H, gqa, block_m, n_block, reverse_m
):
    """Decode attention item ``it`` (or zero unit ``it`` when ``zero``)."""
    if zero:
        b, base = _ps_find(st["zero_base"], it, B)
    else:
        b, base = _ps_find_range(st["item_base"], st["item_end"], it, B)
    d = _ps_lane_geom(
        ptr_cu_q=ptr_cu_q,
        ptr_cu_k=ptr_cu_k,
        b=b,
        B=B,
        H=H,
        gqa=gqa,
        block_m=block_m,
        n_block=n_block,
        scalar=True,
    )
    local = it - base
    mt = fx.max(d["mt"], fx.Int32(1))
    if zero:
        parts = fx.Int32(1)
    else:
        parts = fx.Int32(rocdl.readlane(T.i32, _raw(st["parts"]), _raw(b)))
    unit = local // parts
    part = local - unit * parts
    head = unit // mt
    m_tile = unit - head * mt
    if reverse_m:  # causal: the heavy (late) m-tiles first
        m_tile = mt - fx.Int32(1) - m_tile
    d.update(
        batch=b,
        local=local,
        unit=unit,
        part=part,
        head=head,
        m_tile=m_tile,
        parts=parts,
    )
    if not zero:
        tpp = _ps_cdiv(fx.max(d["kt"], fx.Int32(1)), parts)
        d["tile_lo"] = part * tpp
        d["tile_hi"] = fx.min(d["tile_lo"] + tpp, d["kt"])
        d["slot"] = (
            fx.Int32(rocdl.readlane(T.i32, _raw(st["ws_base"]), _raw(b))) + local
        )
        d["split_unit"] = (
            fx.Int32(rocdl.readlane(T.i32, _raw(st["unit_base"]), _raw(b))) + unit
        )
    return d


@flyc.jit
def _ps_attention_body(
    *, core_kwargs, out_target, warp_idx, lds_base, carry_state=False
):
    """One work item's prologue / core / epilogue (the grid entry's body, minus the
    entry checks). LDS is rewritten from scratch, so the caller fences between items."""
    _epi_state_out = _epi_state_zeros(
        num_q_tiles_per_wave=core_kwargs["num_q_tiles_per_wave"],
        v_hdim=core_kwargs["v_hdim"],
    )
    _prologue = _attention_prologue(
        warp_idx=warp_idx,
        lds_base=lds_base,
        **{k: core_kwargs[k] for k in _PROLOGUE_KEYS},
    )
    if warp_idx // fx.Int32(NUM_WAVES // 2) == fx.Int32(0):
        _epi_state_out = _attention_core_multi_kv_tiles(
            warp_idx=warp_idx,
            warp_type=WarpType.LO,
            lds_base=lds_base,
            prologue=_prologue,
            **core_kwargs,
        )
    else:
        _epi_state_out = _attention_core_multi_kv_tiles(
            warp_idx=warp_idx,
            warp_type=WarpType.HI,
            lds_base=lds_base,
            prologue=_prologue,
            **core_kwargs,
        )
    _attention_epilogue(
        epi_state=_epi_state_out,
        rendezvous=True,
        warp_idx=warp_idx,
        out_target=out_target,
        carry_state=carry_state,
        **{k: core_kwargs[k] for k in _EPILOGUE_KEYS},
    )


def _ps_item_fence():
    """The next item's prologue rewrites LDS: retire this item's TDM O stores (they read
    LDS) and every LDS access, then rendezvous."""
    tdm_ops.tensor_wait(0)
    rocdl.s_wait_dscnt(0)
    gpu.barrier()


@functools.cache
def build_fmha_fwd_prefill_a8w8_m32x8(
    *,
    layout: str = "thd",
    qk_hdim: int = DEFAULT_QK_HDIM,
    v_hdim: int = DEFAULT_V_HDIM,
    n_block: int | None = None,
    q_dtype_str: str = DEFAULT_DTYPE,
    kv_dtype_str: str = DEFAULT_DTYPE,
    mask_left: bool = False,
    mask_right: bool = False,
    one_kv_tile: bool = False,
    return_lse: bool = False,
    has_sink: bool = False,
    has_descale: bool = False,
    gqa_ratio: int = 1,
    num_q_tiles_per_wave: int = WMMA_ROW_PER_WAVE,
    page_size: int | None = None,
    carry_state: bool = False,
):
    """Build the m32x8 device kernel for a given layout ("thd" varlen or "bshd" batched)
    plus config; every parameter here is compile-time and baked into the trace.
    ``gqa_ratio`` (= ``nheads_q // nheads_kv``) is among them so the per-lane ``% / //``
    fold to shift/and when it is a power of two.

    ``carry_state`` (thd only, needs ``return_lse``): O/LSE are a running state that
    this launch updates in place (see ``_store_o_lse``); a batch with no keys leaves
    its rows untouched instead of zero-filling them.
    """
    CARRY = bool(carry_state)
    assert not CARRY or (
        layout in ("thd", "thd_ps") and return_lse and not has_sink
    ), "carry_state is thd/thd_ps-only, needs return_lse and no sink"
    assert layout in (
        "thd",
        "thd_ps",
        "bshd",
        "paged",
    ), f"layout must be thd|thd_ps|bshd|paged, got {layout!r}"
    # qk_hdim (D_qk) is a WMMA_K multiple; v_hdim (D_v) a WMMA_M multiple. Independent.
    assert (
        qk_hdim in SUPPORTED_QK_HDIM and v_hdim in SUPPORTED_V_HDIM
    ), f"supports qk_hdim in {SUPPORTED_QK_HDIM} x v_hdim in {SUPPORTED_V_HDIM}, got {qk_hdim}/{v_hdim}"
    assert (
        q_dtype_str in SUPPORTED_DTYPES and kv_dtype_str in SUPPORTED_DTYPES
    ), f"dtypes must be in {list(SUPPORTED_DTYPES)}, got {q_dtype_str!r}/{kv_dtype_str!r}"
    Q_ELEM_DTYPE = DTYPE_MAP[q_dtype_str]
    KV_ELEM_DTYPE = DTYPE_MAP[kv_dtype_str]
    assert gqa_ratio >= 1, f"gqa_ratio must be >= 1, got {gqa_ratio}"
    assert (
        num_q_tiles_per_wave in NUM_Q_TILES_CHOICES
    ), f"num_q_tiles_per_wave must be in {NUM_Q_TILES_CHOICES}, got {num_q_tiles_per_wave}"
    if n_block is None:
        n_block = pick_n_block(qk_hdim, v_hdim, KV_ELEM_DTYPE)
    assert (
        n_block in N_BLOCK_CHOICES
    ), f"n_block must be in {N_BLOCK_CHOICES}, got {n_block}"

    QK_HDIM = qk_hdim
    V_HDIM = v_hdim
    N_BLOCK = int(n_block)
    MASK_LEFT = bool(mask_left)
    MASK_RIGHT = bool(mask_right)
    PAGE_SIZE = None if page_size is None else int(page_size)
    if PAGE_SIZE is not None:
        assert (
            PAGE_SIZE & (PAGE_SIZE - 1) == 0
        ), f"page_size must be a power of two, got {PAGE_SIZE}"
        assert not one_kv_tile, "paged uses the multi-tile core"
    LOG2_PAGE = None if PAGE_SIZE is None else PAGE_SIZE.bit_length() - 1
    HAS_DESCALE = bool(has_descale)
    ONE_KV_TILE = bool(one_kv_tile)
    RET_LSE = bool(return_lse)
    HAS_SINK = bool(has_sink)
    GQA_RATIO = int(gqa_ratio)
    NUM_Q_TILES = int(num_q_tiles_per_wave)

    if layout == "paged":

        @flyc.kernel(known_block_size=[BLOCK_SIZE, 1, 1])
        def kn_fmha_fwd_prefill_a8w8_m32x8_paged(
            ptr_O: fx.Pointer,
            ptr_Q: fx.Pointer,
            ptr_K: fx.Pointer,
            ptr_V: fx.Pointer,
            ptr_LSE: fx.Pointer,
            ptr_sink: fx.Pointer,
            ptr_q_descale: fx.Pointer,
            ptr_k_descale: fx.Pointer,
            ptr_v_descale: fx.Pointer,
            ptr_cu_seqlens_q: fx.Pointer,
            ptr_kv_indptr: fx.Pointer,
            ptr_kv_page_indices: fx.Pointer,
            ptr_kv_last_page_lens: fx.Pointer,
            softmax_scale: fx.Float32,
            stride_q_seq: fx.Int32,
            stride_k_seq: fx.Int32,
            stride_v_seq: fx.Int32,
            stride_o_seq: fx.Int32,
            stride_q_head: fx.Int32,
            stride_k_head: fx.Int32,
            stride_v_head: fx.Int32,
            stride_o_head: fx.Int32,
            stride_lse_seq: fx.Int32,
            stride_lse_head: fx.Int32,
            window_left: fx.Int32,
            window_right: fx.Int32,
            num_page_entries: fx.Int32,
            num_pool_rows: fx.Int32,
        ):
            """Paged entry: Q comes from qo_indptr as in THD, but this batch's KV is a
            page list, so kv_len is derived from the page count and the last page's
            fill. Positions stay sequence-relative (kv_start = 0); the page table
            supplies the pool row."""
            batch = fx.Int32(gpu.block_id("z"))
            q_start, q_end = _load_seqlen_pair(ptr_cu_seqlens_q, batch)
            q_len = q_end - q_start
            page_lo, page_hi = _load_seqlen_pair(ptr_kv_indptr, batch)
            num_pages = page_hi - page_lo
            last_page_len = s_buffer_load(
                ptr_kv_last_page_lens,
                batch,
                1,
                num_records_bytes=fx.Int32(gpu.grid_dim.z) * fx.Int32(4),
            )[0]
            kv_start = fx.Int32(0)
            kv_len = (num_pages > fx.Int32(0)).select(
                ((num_pages - fx.Int32(1)) << fx.Int32(LOG2_PAGE)) + last_page_len,
                fx.Int32(0),
            )

            # LSE is [nheads_q, total_q]. Bound the buffer resource by the last element
            # this batch can touch over BOTH axes; a seq-only bound is exact only when
            # seq is the major axis.
            num_heads_q = gpu.grid_dim.y * fx.Int32(GQA_RATIO)
            lse_base_elems = q_start * stride_lse_seq
            lse_num_records_bytes = (
                fx.Int64(q_start + q_len - fx.Int32(1)) * fx.Int64(stride_lse_seq)
                + fx.Int64(num_heads_q - fx.Int32(1)) * fx.Int64(stride_lse_head)
                + 1
            ) * 4

            # The grid is sized for max_seqlen_q, so in a ragged batch a short sequence
            # leaves whole WGs past its end with no row to compute AND none to zero.
            # Per-tensor descales, as gfx950 does: q/k scale the fp32 logits (so they
            # ride softmax_scale and reach the LSE through m), v scales O alone.
            if const_expr(HAS_DESCALE):
                _softmax_scale = softmax_scale * (
                    _load_descale(ptr_q_descale) * _load_descale(ptr_k_descale)
                )
                _v_descale = _load_descale(ptr_v_descale)
            else:
                _softmax_scale = softmax_scale
                _v_descale = None

            first_row = _m_tile_idx() * fx.Int32(_block_m(NUM_Q_TILES))
            tile_has_rows = first_row < q_len * fx.Int32(GQA_RATIO)
            # An empty batch must not enter the core: kv_len==0 would write O/0 = NaN.
            if (q_len > fx.Int32(0)) & (kv_len > fx.Int32(0)) & tile_has_rows:
                _core_kwargs = {
                    "qk_hdim": QK_HDIM,
                    "v_hdim": V_HDIM,
                    "n_block": N_BLOCK,
                    "mask_left": MASK_LEFT,
                    "mask_right": MASK_RIGHT,
                    "return_lse": RET_LSE,
                    "has_sink": HAS_SINK,
                    "gqa_ratio": GQA_RATIO,
                    "num_q_tiles_per_wave": NUM_Q_TILES,
                    "ptr_O": ptr_O,
                    "ptr_Q": ptr_Q,
                    "ptr_K": ptr_K,
                    "ptr_V": ptr_V,
                    "ptr_LSE": ptr_LSE,
                    "ptr_sink": ptr_sink,
                    "softmax_scale": _softmax_scale,
                    "v_descale": _v_descale,
                    "stride_q_seq": stride_q_seq,
                    "stride_k_seq": stride_k_seq,
                    "stride_v_seq": stride_v_seq,
                    "stride_o_seq": stride_o_seq,
                    "stride_q_head": stride_q_head,
                    "stride_k_head": stride_k_head,
                    "stride_v_head": stride_v_head,
                    "stride_o_head": stride_o_head,
                    "stride_lse_seq": stride_lse_seq,
                    "stride_lse_head": stride_lse_head,
                    "lse_base_elems": lse_base_elems,
                    "lse_num_records_bytes": lse_num_records_bytes,
                    "q_start": q_start,
                    "q_len": q_len,
                    "kv_start": kv_start,
                    "kv_len": kv_len,
                    "window_left": window_left,
                    "window_right": window_right,
                    "q_dtype": Q_ELEM_DTYPE,
                    "kv_dtype": KV_ELEM_DTYPE,
                    "paged": PagedSpec(
                        ptr_page_indices=ptr_kv_page_indices,
                        num_page_entries=num_page_entries,
                        page_base=page_lo,
                        page_size=PAGE_SIZE,
                        num_pool_rows=num_pool_rows,
                    ),
                }
                # Exactly one core reaches the binary: the host decides from the shape (bshd) or
                # from max_seqlen_k (thd) whether every WG sees a single KV tile.
                lds_base = _alloc_lds()
                warp_idx = _warp_id()

                _epi_state_out = _epi_state_zeros(
                    num_q_tiles_per_wave=NUM_Q_TILES, v_hdim=V_HDIM
                )
                if ONE_KV_TILE:
                    _epi_state_out = _attention_core_one_kv_tile(
                        warp_idx=warp_idx, lds_base=lds_base, **_core_kwargs
                    )
                else:
                    # One traced copy of Q's load and the first KV tiles, outside the
                    # warp-type branch; the two halves differ only under its own branches.
                    _prologue = _attention_prologue(
                        warp_idx=warp_idx,
                        lds_base=lds_base,
                        **{k: _core_kwargs[k] for k in _PROLOGUE_KEYS},
                    )
                    # Warp specialization: LO (waves 0..N/2-1) vs HI (N/2..N-1).
                    if warp_idx // fx.Int32(NUM_WAVES // 2) == fx.Int32(0):
                        _epi_state_out = _attention_core_multi_kv_tiles(
                            warp_idx=warp_idx,
                            warp_type=WarpType.LO,
                            lds_base=lds_base,
                            prologue=_prologue,
                            **_core_kwargs,
                        )
                    else:
                        _epi_state_out = _attention_core_multi_kv_tiles(
                            warp_idx=warp_idx,
                            warp_type=WarpType.HI,
                            lds_base=lds_base,
                            prologue=_prologue,
                            **_core_kwargs,
                        )
                # One traced copy of the O/LSE tail, outside the warp-type branch.
                _attention_epilogue(
                    epi_state=_epi_state_out,
                    rendezvous=not ONE_KV_TILE,
                    warp_idx=warp_idx,
                    **{k: _core_kwargs[k] for k in _EPILOGUE_KEYS},
                )
            elif (q_len > fx.Int32(0)) & tile_has_rows:
                # Cross-attention tail: q_len>0 but kv_len==0 -> O=0, LSE=-inf (or sink).
                _zero_fill_attention(
                    v_hdim=V_HDIM,
                    gqa_ratio=GQA_RATIO,
                    num_q_tiles_per_wave=NUM_Q_TILES,
                    return_lse=RET_LSE,
                    has_sink=HAS_SINK,
                    ptr_sink=ptr_sink,
                    ptr_O=ptr_O,
                    ptr_LSE=ptr_LSE,
                    stride_o_seq=stride_o_seq,
                    stride_o_head=stride_o_head,
                    stride_lse_seq=stride_lse_seq,
                    stride_lse_head=stride_lse_head,
                    lse_num_records_bytes=lse_num_records_bytes,
                    q_start=q_start,
                    q_len=q_len,
                )

        return kn_fmha_fwd_prefill_a8w8_m32x8_paged

    if layout == "thd_ps":
        assert not ONE_KV_TILE, "persistent launch uses the multi-tile core"
        assert RET_LSE, "persistent launch always writes LSE (the split merge needs it)"
        BM = _block_m(NUM_Q_TILES)
        # KV splitting needs a part-local band: non-causal, no window, MHA rows (the
        # workspace row layout is one head per unit), no sink (folded into the merge
        # would double-count it).
        SPLIT_OK = (
            (not MASK_LEFT) and (not MASK_RIGHT) and GQA_RATIO == 1 and not HAS_SINK
        )

        @flyc.kernel(known_block_size=[BLOCK_SIZE, 1, 1])
        def kn_fmha_fwd_prefill_a8w8_m32x8_thd_ps(
            ptr_O: fx.Pointer,
            ptr_Q: fx.Pointer,
            ptr_K: fx.Pointer,
            ptr_V: fx.Pointer,
            ptr_LSE: fx.Pointer,
            ptr_sink: fx.Pointer,
            ptr_q_descale: fx.Pointer,
            ptr_k_descale: fx.Pointer,
            ptr_v_descale: fx.Pointer,
            ptr_cu_seqlens_q: fx.Pointer,
            ptr_cu_seqlens_k: fx.Pointer,
            ptr_ws_o: fx.Pointer,
            ptr_ws_lse: fx.Pointer,
            ptr_ws_meta: fx.Pointer,
            softmax_scale: fx.Float32,
            stride_q_seq: fx.Int32,
            stride_k_seq: fx.Int32,
            stride_v_seq: fx.Int32,
            stride_o_seq: fx.Int32,
            stride_q_head: fx.Int32,
            stride_k_head: fx.Int32,
            stride_v_head: fx.Int32,
            stride_o_head: fx.Int32,
            stride_lse_seq: fx.Int32,
            stride_lse_head: fx.Int32,
            window_left: fx.Int32,
            window_right: fx.Int32,
            num_heads_kv: fx.Int32,
            batch_size: fx.Int32,
            ws_items_cap: fx.Int32,
            lse_num_records_bytes: fx.Int32,
            force_pt: fx.Int32,
        ):
            """Persistent varlen THD entry. Non-causal: stream-K -- WG w takes a contiguous
            range of the virtual tile space, so it holds whole units plus at most two
            partial ones (its first and last items, written to workspace slots 2w / 2w+1
            for the merge). Causal: whole units round-robin. Zero-fill units round-robin.
            """
            G = fx.Int32(gpu.grid_dim.x)
            wg = fx.Int32(gpu.block_id("x"))
            H = num_heads_kv
            geo = {
                "ptr_cu_q": ptr_cu_seqlens_q,
                "ptr_cu_k": ptr_cu_seqlens_k,
                "B": batch_size,
                "H": H,
                "gqa": GQA_RATIO,
                "block_m": BM,
                "n_block": N_BLOCK,
            }
            st = _ps_setup(
                G=G,
                allow_split=SPLIT_OK,
                ws_cap=ws_items_cap,
                force_pt=force_pt,
                **geo,
            )
            tid = _warp_id() * fx.Int32(WAVE_SIZE) + _lane_id()

            if const_expr(HAS_DESCALE):
                _softmax_scale = softmax_scale * (
                    _load_descale(ptr_q_descale) * _load_descale(ptr_k_descale)
                )
                _v_descale = _load_descale(ptr_v_descale)
            else:
                _softmax_scale = softmax_scale
                _v_descale = None

            o_addr = fx.Int64(fx.ptrtoint(fx.get_iter(ptr_O)))
            ws_o_addr = fx.Int64(fx.ptrtoint(fx.get_iter(ptr_ws_o)))
            lse_addr = fx.Int64(fx.ptrtoint(fx.get_iter(ptr_LSE)))
            ws_lse_addr = fx.Int64(fx.ptrtoint(fx.get_iter(ptr_ws_lse)))
            ws_lse_bytes = ws_items_cap * fx.Int32(BM * 4)
            o_ptr_ty = fx.PointerType.get(
                elem_ty=O_DTYPE.ir_type,
                address_space=fx.AddressSpace.Global,
                alignment=16,
            )
            lds_base = _alloc_lds()
            warp_idx = _warp_id()

            def _core_kw(d, kv_start, kv_len):
                return {
                    "qk_hdim": QK_HDIM,
                    "v_hdim": V_HDIM,
                    "n_block": N_BLOCK,
                    "mask_left": MASK_LEFT,
                    "mask_right": MASK_RIGHT,
                    "return_lse": RET_LSE,
                    "has_sink": HAS_SINK,
                    "gqa_ratio": GQA_RATIO,
                    "num_q_tiles_per_wave": NUM_Q_TILES,
                    "ptr_O": ptr_O,
                    "ptr_Q": ptr_Q,
                    "ptr_K": ptr_K,
                    "ptr_V": ptr_V,
                    "ptr_LSE": ptr_LSE,
                    "ptr_sink": ptr_sink,
                    "softmax_scale": _softmax_scale,
                    "v_descale": _v_descale,
                    "stride_q_seq": stride_q_seq,
                    "stride_k_seq": stride_k_seq,
                    "stride_v_seq": stride_v_seq,
                    "stride_o_seq": stride_o_seq,
                    "stride_q_head": stride_q_head,
                    "stride_k_head": stride_k_head,
                    "stride_v_head": stride_v_head,
                    "stride_o_head": stride_o_head,
                    "stride_lse_seq": stride_lse_seq,
                    "stride_lse_head": stride_lse_head,
                    "lse_base_elems": d["q0"] * stride_lse_seq,
                    "lse_num_records_bytes": lse_num_records_bytes,
                    "q_start": d["q0"],
                    "q_len": d["ql"],
                    "kv_start": kv_start,
                    "kv_len": kv_len,
                    "window_left": window_left,
                    "window_right": window_right,
                    "q_dtype": Q_ELEM_DTYPE,
                    "kv_dtype": KV_ELEM_DTYPE,
                    "paged": None,
                }

            def _zero_item(d):
                _zero_fill_attention(
                    v_hdim=V_HDIM,
                    gqa_ratio=GQA_RATIO,
                    num_q_tiles_per_wave=NUM_Q_TILES,
                    return_lse=RET_LSE,
                    has_sink=HAS_SINK,
                    ptr_sink=ptr_sink,
                    ptr_O=ptr_O,
                    ptr_LSE=ptr_LSE,
                    stride_o_seq=stride_o_seq,
                    stride_o_head=stride_o_head,
                    stride_lse_seq=stride_lse_seq,
                    stride_lse_head=stride_lse_head,
                    lse_num_records_bytes=lse_num_records_bytes,
                    q_start=d["q0"],
                    q_len=d["ql"],
                )

            meta_rsrc = buffer_ops.create_buffer_resource(
                ptr_ws_meta,
                num_records_bytes=(
                    fx.Int32(PS_META_HDR)
                    + (ws_items_cap // fx.Int32(2)) * fx.Int32(PS_META_STRIDE)
                )
                * fx.Int32(4),
            )
            if const_expr(SPLIT_OK):
                hdr_off = ((wg == fx.Int32(0)) & (tid == fx.Int32(0))).select(
                    fx.Int32(0), fx.Int32(0x7FFFFFFF)
                )
                buffer_ops.buffer_store(
                    st["n_split_units"],
                    meta_rsrc,
                    hdr_off,
                    mask=None,
                    offset_is_bytes=True,
                )
            N = st["N"]
            # Under carry_state a unit without keys is the merge identity: no zero items.
            n_items = N if CARRY else N + st["ZU"]
            n_mine = fx.max(_ps_cdiv(n_items - wg, G), fx.Int32(0))
            for j in range(fx.Index(0), fx.Index(n_mine), 1):
                it = wg + fx.Int32(j) * G
                if it < N:
                    d = _ps_item(st, it, zero=False, reverse_m=MASK_RIGHT, **geo)
                    m_tile = d["m_tile"]
                    kv_lo = d["tile_lo"] * fx.Int32(N_BLOCK)
                    kv_len = fx.min(d["tile_hi"] * fx.Int32(N_BLOCK), d["kl"]) - kv_lo
                    if const_expr(SPLIT_OK):
                        is_ws = d["parts"] > fx.Int32(1)
                        slot = d["slot"]
                        out_target = {
                            "o_iter": fx.inttoptr(
                                o_ptr_ty, is_ws.select(ws_o_addr, o_addr)
                            ),
                            "o_base_elems": is_ws.select(
                                (slot - m_tile) * fx.Int32(BM * V_HDIM), fx.Int32(0)
                            ),
                            "q_start_o": is_ws.select(fx.Int32(0), d["q0"]),
                            "stride_o_seq": is_ws.select(
                                fx.Int32(V_HDIM), stride_o_seq
                            ),
                            "stride_o_head": is_ws.select(fx.Int32(0), stride_o_head),
                            "lse_rsrc": buffer_ops.create_buffer_resource_from_addr(
                                is_ws.select(ws_lse_addr, lse_addr),
                                num_records_bytes=is_ws.select(
                                    ws_lse_bytes, lse_num_records_bytes
                                ),
                            ),
                            "lse_base_elems": is_ws.select(
                                (slot - m_tile) * fx.Int32(BM),
                                d["q0"] * stride_lse_seq,
                            ),
                            "stride_lse_seq": is_ws.select(fx.Int32(1), stride_lse_seq),
                            "stride_lse_head": is_ws.select(
                                fx.Int32(0), stride_lse_head
                            ),
                            # Under carry the running state is folded into part 0 of a
                            # split unit (raw parts otherwise); the merge pass then only
                            # combines parts and never reads the running state.
                            "carry_ok": (d["parts"] == fx.Int32(1))
                            | (d["part"] == fx.Int32(0)),
                        }
                    else:
                        out_target = None
                    with virtual_block(
                        m_tile=m_tile, kv_head=d["head"], num_heads_kv=H
                    ):
                        _ps_attention_body(
                            core_kwargs=_core_kw(d, d["k0"] + kv_lo, kv_len),
                            out_target=out_target,
                            warp_idx=warp_idx,
                            lds_base=lds_base,
                            carry_state=CARRY,
                        )
                    if const_expr(SPLIT_OK):
                        # Part 0 of a split unit publishes its merge entry:
                        # (slot0, parts, out_row0, nrows, head).
                        ent = fx.Int32(PS_META_HDR) + d["split_unit"] * fx.Int32(
                            PS_META_STRIDE
                        )
                        pub = is_ws & (d["part"] == fx.Int32(0)) & (tid < fx.Int32(5))
                        val = (tid == fx.Int32(0)).select(
                            slot,
                            (tid == fx.Int32(1)).select(
                                d["parts"],
                                (tid == fx.Int32(2)).select(
                                    d["q0"] + m_tile * fx.Int32(BM),
                                    (tid == fx.Int32(3)).select(
                                        fx.min(
                                            fx.Int32(BM),
                                            d["ql"] - m_tile * fx.Int32(BM),
                                        ),
                                        d["head"],
                                    ),
                                ),
                            ),
                        )
                        off = pub.select(
                            (ent + tid) * fx.Int32(4), fx.Int32(0x7FFFFFFF)
                        )
                        buffer_ops.buffer_store(
                            val, meta_rsrc, off, mask=None, offset_is_bytes=True
                        )
                    _ps_item_fence()
                else:
                    d = _ps_item(st, it - N, zero=True, reverse_m=False, **geo)
                    with virtual_block(
                        m_tile=d["m_tile"], kv_head=d["head"], num_heads_kv=H
                    ):
                        _zero_item(d)

        return kn_fmha_fwd_prefill_a8w8_m32x8_thd_ps

    if layout == "thd":

        @flyc.kernel(known_block_size=[BLOCK_SIZE, 1, 1])
        def kn_fmha_fwd_prefill_a8w8_m32x8_thd(
            ptr_O: fx.Pointer,
            ptr_Q: fx.Pointer,
            ptr_K: fx.Pointer,
            ptr_V: fx.Pointer,
            ptr_LSE: fx.Pointer,
            ptr_sink: fx.Pointer,
            ptr_q_descale: fx.Pointer,
            ptr_k_descale: fx.Pointer,
            ptr_v_descale: fx.Pointer,
            ptr_cu_seqlens_q: fx.Pointer,
            ptr_cu_seqlens_k: fx.Pointer,
            softmax_scale: fx.Float32,
            stride_q_seq: fx.Int32,
            stride_k_seq: fx.Int32,
            stride_v_seq: fx.Int32,
            stride_o_seq: fx.Int32,
            stride_q_head: fx.Int32,
            stride_k_head: fx.Int32,
            stride_v_head: fx.Int32,
            stride_o_head: fx.Int32,
            stride_lse_seq: fx.Int32,
            stride_lse_head: fx.Int32,
            window_left: fx.Int32,
            window_right: fx.Int32,
            max_seqlen_q: fx.Int32,
            max_seqlen_k: fx.Int32,
        ):
            """Varlen THD entry: this batch's token ranges come from cu_seqlens
            (batch = grid.z)."""
            batch = fx.Int32(gpu.block_id("z"))
            q_start, q_end = _load_seqlen_pair(ptr_cu_seqlens_q, batch)
            kv_start, kv_end = _load_seqlen_pair(ptr_cu_seqlens_k, batch)
            q_len = q_end - q_start
            kv_len = kv_end - kv_start

            # LSE is [nheads_q, total_q]. Bound the buffer resource by the last element
            # this batch can touch over BOTH axes; a seq-only bound is exact only when
            # seq is the major axis.
            num_heads_q = gpu.grid_dim.y * fx.Int32(GQA_RATIO)
            lse_base_elems = q_start * stride_lse_seq
            lse_num_records_bytes = (
                fx.Int64(q_start + q_len - fx.Int32(1)) * fx.Int64(stride_lse_seq)
                + fx.Int64(num_heads_q - fx.Int32(1)) * fx.Int64(stride_lse_head)
                + fx.Int64(1)
            ) * fx.Int64(4)

            # Per-tensor descales, as gfx950 does: q/k scale the fp32 logits (so they
            # ride softmax_scale and reach the LSE through m), v scales O alone.
            if const_expr(HAS_DESCALE):
                _softmax_scale = softmax_scale * (
                    _load_descale(ptr_q_descale) * _load_descale(ptr_k_descale)
                )
                _v_descale = _load_descale(ptr_v_descale)
            else:
                _softmax_scale = softmax_scale
                _v_descale = None

            # An empty batch must not enter the core: kv_len==0 would write O/0 = NaN.
            # Nor a tile past this batch's queries: grid.x is sized by max_seqlen_q, so a
            # shorter sequence's trailing tiles own no packed row. They would stream the
            # whole KV band with every row masked and store nothing. The condition is
            # uniform across the workgroup's waves, so they skip together.
            first_row = _m_tile_idx() * fx.Int32(_block_m(NUM_Q_TILES))
            tile_has_rows = first_row < q_len * fx.Int32(GQA_RATIO)
            if (q_len > fx.Int32(0)) & (kv_len > fx.Int32(0)) & tile_has_rows:
                _core_kwargs = {
                    "qk_hdim": QK_HDIM,
                    "v_hdim": V_HDIM,
                    "n_block": N_BLOCK,
                    "mask_left": MASK_LEFT,
                    "mask_right": MASK_RIGHT,
                    "return_lse": RET_LSE,
                    "has_sink": HAS_SINK,
                    "gqa_ratio": GQA_RATIO,
                    "num_q_tiles_per_wave": NUM_Q_TILES,
                    "ptr_O": ptr_O,
                    "ptr_Q": ptr_Q,
                    "ptr_K": ptr_K,
                    "ptr_V": ptr_V,
                    "ptr_LSE": ptr_LSE,
                    "ptr_sink": ptr_sink,
                    "softmax_scale": _softmax_scale,
                    "v_descale": _v_descale,
                    "stride_q_seq": stride_q_seq,
                    "stride_k_seq": stride_k_seq,
                    "stride_v_seq": stride_v_seq,
                    "stride_o_seq": stride_o_seq,
                    "stride_q_head": stride_q_head,
                    "stride_k_head": stride_k_head,
                    "stride_v_head": stride_v_head,
                    "stride_o_head": stride_o_head,
                    "stride_lse_seq": stride_lse_seq,
                    "stride_lse_head": stride_lse_head,
                    "lse_base_elems": lse_base_elems,
                    "lse_num_records_bytes": lse_num_records_bytes,
                    "q_start": q_start,
                    "q_len": q_len,
                    "kv_start": kv_start,
                    "kv_len": kv_len,
                    "window_left": window_left,
                    "window_right": window_right,
                    "q_dtype": Q_ELEM_DTYPE,
                    "kv_dtype": KV_ELEM_DTYPE,
                    "paged": None,
                }
                # Exactly one core reaches the binary: the host decides from the shape (bshd) or
                # from max_seqlen_k (thd) whether every WG sees a single KV tile.
                lds_base = _alloc_lds()
                warp_idx = _warp_id()

                _epi_state_out = _epi_state_zeros(
                    num_q_tiles_per_wave=NUM_Q_TILES, v_hdim=V_HDIM
                )
                if ONE_KV_TILE:
                    _epi_state_out = _attention_core_one_kv_tile(
                        warp_idx=warp_idx, lds_base=lds_base, **_core_kwargs
                    )
                else:
                    # One traced copy of Q's load and the first KV tiles, outside the
                    # warp-type branch; the two halves differ only under its own branches.
                    _prologue = _attention_prologue(
                        warp_idx=warp_idx,
                        lds_base=lds_base,
                        **{k: _core_kwargs[k] for k in _PROLOGUE_KEYS},
                    )
                    # Warp specialization: LO (waves 0..N/2-1) vs HI (N/2..N-1).
                    if warp_idx // fx.Int32(NUM_WAVES // 2) == fx.Int32(0):
                        _epi_state_out = _attention_core_multi_kv_tiles(
                            warp_idx=warp_idx,
                            warp_type=WarpType.LO,
                            lds_base=lds_base,
                            prologue=_prologue,
                            **_core_kwargs,
                        )
                    else:
                        _epi_state_out = _attention_core_multi_kv_tiles(
                            warp_idx=warp_idx,
                            warp_type=WarpType.HI,
                            lds_base=lds_base,
                            prologue=_prologue,
                            **_core_kwargs,
                        )
                # One traced copy of the O/LSE tail, outside the warp-type branch.
                _attention_epilogue(
                    epi_state=_epi_state_out,
                    rendezvous=not ONE_KV_TILE,
                    warp_idx=warp_idx,
                    carry_state=CARRY,
                    **{k: _core_kwargs[k] for k in _EPILOGUE_KEYS},
                )
            elif (q_len > fx.Int32(0)) & tile_has_rows:
                # Cross-attention tail: q_len>0 but kv_len==0 -> O=0, LSE=-inf (or sink).
                # A tile past the queries has no row to fill, so it skips this too.
                # Under carry_state an empty KV set is the merge identity: the running
                # state stays as it is.
                if const_expr(not CARRY):
                    _zero_fill_attention(
                        v_hdim=V_HDIM,
                        gqa_ratio=GQA_RATIO,
                        num_q_tiles_per_wave=NUM_Q_TILES,
                        return_lse=RET_LSE,
                        has_sink=HAS_SINK,
                        ptr_sink=ptr_sink,
                        ptr_O=ptr_O,
                        ptr_LSE=ptr_LSE,
                        stride_o_seq=stride_o_seq,
                        stride_o_head=stride_o_head,
                        stride_lse_seq=stride_lse_seq,
                        stride_lse_head=stride_lse_head,
                        lse_num_records_bytes=lse_num_records_bytes,
                        q_start=q_start,
                        q_len=q_len,
                    )

        return kn_fmha_fwd_prefill_a8w8_m32x8_thd

    @flyc.kernel(known_block_size=[BLOCK_SIZE, 1, 1])
    def kn_fmha_fwd_prefill_a8w8_m32x8_bshd(
        ptr_O: fx.Pointer,
        ptr_Q: fx.Pointer,
        ptr_K: fx.Pointer,
        ptr_V: fx.Pointer,
        ptr_LSE: fx.Pointer,
        ptr_sink: fx.Pointer,
        ptr_q_descale: fx.Pointer,
        ptr_k_descale: fx.Pointer,
        ptr_v_descale: fx.Pointer,
        softmax_scale: fx.Float32,
        stride_q_seq: fx.Int32,
        stride_k_seq: fx.Int32,
        stride_v_seq: fx.Int32,
        stride_o_seq: fx.Int32,
        stride_q_head: fx.Int32,
        stride_k_head: fx.Int32,
        stride_v_head: fx.Int32,
        stride_o_head: fx.Int32,
        stride_lse_seq: fx.Int32,
        stride_lse_head: fx.Int32,
        stride_lse_batch: fx.Int32,
        window_left: fx.Int32,
        window_right: fx.Int32,
        seq_len_q: fx.Int32,
        seq_len_k: fx.Int32,
    ):
        """Batched BSHD entry: uniform ``seq_len_q``/``seq_len_k`` scalars replace
        cu_seqlens, so nothing is transient and this path is CUDA-graph safe. Token base
        is batch_idx * seq_len (batch = grid.z)."""
        batch = fx.Int32(gpu.block_id("z"))

        # LSE is [B, nheads_q, seq_q]: base = batch*stride_lse_batch; every valid
        # element offset is < base + stride_lse_batch (< the 0x7FFFFFFF drop).
        lse_base_elems = batch * stride_lse_batch
        lse_num_records_bytes = fx.Int64(lse_base_elems + stride_lse_batch) * fx.Int64(
            4
        )

        # Per-tensor descales, as in the thd entry.
        if const_expr(HAS_DESCALE):
            _softmax_scale = softmax_scale * (
                _load_descale(ptr_q_descale) * _load_descale(ptr_k_descale)
            )
            _v_descale = _load_descale(ptr_v_descale)
        else:
            _softmax_scale = softmax_scale
            _v_descale = None

        _core_kwargs = {
            "qk_hdim": QK_HDIM,
            "v_hdim": V_HDIM,
            "n_block": N_BLOCK,
            "mask_left": MASK_LEFT,
            "mask_right": MASK_RIGHT,
            "return_lse": RET_LSE,
            "has_sink": HAS_SINK,
            "gqa_ratio": GQA_RATIO,
            "num_q_tiles_per_wave": NUM_Q_TILES,
            "ptr_O": ptr_O,
            "ptr_Q": ptr_Q,
            "ptr_K": ptr_K,
            "ptr_V": ptr_V,
            "ptr_LSE": ptr_LSE,
            "ptr_sink": ptr_sink,
            "softmax_scale": _softmax_scale,
            "v_descale": _v_descale,
            "stride_q_seq": stride_q_seq,
            "stride_k_seq": stride_k_seq,
            "stride_v_seq": stride_v_seq,
            "stride_o_seq": stride_o_seq,
            "stride_q_head": stride_q_head,
            "stride_k_head": stride_k_head,
            "stride_v_head": stride_v_head,
            "stride_o_head": stride_o_head,
            "stride_lse_seq": stride_lse_seq,
            "stride_lse_head": stride_lse_head,
            "lse_base_elems": lse_base_elems,
            "lse_num_records_bytes": lse_num_records_bytes,
            "q_start": batch * seq_len_q,
            "q_len": seq_len_q,
            "kv_start": batch * seq_len_k,
            "kv_len": seq_len_k,
            "window_left": window_left,
            "window_right": window_right,
            "q_dtype": Q_ELEM_DTYPE,
            "kv_dtype": KV_ELEM_DTYPE,
            "paged": None,
        }
        # Exactly one core reaches the binary: the host decides from the shape (bshd) or
        # from max_seqlen_k (thd) whether every WG sees a single KV tile.
        lds_base = _alloc_lds()
        warp_idx = _warp_id()

        _epi_state_out = _epi_state_zeros(
            num_q_tiles_per_wave=NUM_Q_TILES, v_hdim=V_HDIM
        )
        if ONE_KV_TILE:
            _epi_state_out = _attention_core_one_kv_tile(
                warp_idx=warp_idx, lds_base=lds_base, **_core_kwargs
            )
        else:
            # One traced copy of Q's load and the first KV tiles, outside the
            # warp-type branch; the two halves differ only under its own branches.
            _prologue = _attention_prologue(
                warp_idx=warp_idx,
                lds_base=lds_base,
                **{k: _core_kwargs[k] for k in _PROLOGUE_KEYS},
            )
            # Warp specialization: LO (waves 0..N/2-1) vs HI (N/2..N-1).
            if warp_idx // fx.Int32(NUM_WAVES // 2) == fx.Int32(0):
                _epi_state_out = _attention_core_multi_kv_tiles(
                    warp_idx=warp_idx,
                    warp_type=WarpType.LO,
                    lds_base=lds_base,
                    prologue=_prologue,
                    **_core_kwargs,
                )
            else:
                _epi_state_out = _attention_core_multi_kv_tiles(
                    warp_idx=warp_idx,
                    warp_type=WarpType.HI,
                    lds_base=lds_base,
                    prologue=_prologue,
                    **_core_kwargs,
                )
        # One traced copy of the O/LSE tail, outside the warp-type branch.
        _attention_epilogue(
            epi_state=_epi_state_out,
            rendezvous=not ONE_KV_TILE,
            warp_idx=warp_idx,
            **{k: _core_kwargs[k] for k in _EPILOGUE_KEYS},
        )

    return kn_fmha_fwd_prefill_a8w8_m32x8_bshd


# ============================================================================
# Launch wrappers + host entries
# ============================================================================

_launch_fns = (
    {}
)  # {(layout, mask_left, mask_right, return_lse, has_sink, gqa_ratio): fn}


def _pick_one_kv_tile(max_seqlen_k, qk_hdim, v_hdim, kv_dtype_str):
    """True when the launch bounds every WG to a single KV tile."""
    n_block = pick_n_block(qk_hdim, v_hdim, DTYPE_MAP[kv_dtype_str])
    return int(max_seqlen_k) <= n_block


# XCD balance of the varlen/paged grid (FMHA_ODD_M_TILE_GRID=0 disables it).
ODD_M_TILE_GRID = os.environ.get("FMHA_ODD_M_TILE_GRID", "1") != "0"


def _xcd_balanced_max_seqlen_q(max_seqlen_q, gqa, num_q_tiles_per_wave):
    """``max_seqlen_q`` as the launcher should size grid x from: rounded up so the
    m-tile count is odd.

    The dispatcher deals consecutive workgroup ids to the XCDs round-robin, and grid x
    (the m-tile) is the fastest axis, so the heads of one m-tile sit at ids
    ``x + grid_x * head``. With an even grid_x they reach only a fraction of the XCDs
    -- grid_x 4 on 8 XCDs puts every head of a tile on 2 of them. A ragged batch whose
    sequences own fewer m-tiles than the longest, e.g. a chunked-prefill context chunk
    whose keys belong to one or two sequences, then ran its real WGs on a quarter of
    the GPU (measured 2x on a one-tile chunk). An odd tile count is coprime with any
    power-of-two XCD count, so each tile's heads cover all XCDs; the padding tiles own
    no rows and exit at the kernel's first-row check. The kernels read max_seqlen_q for
    the grid only.
    """
    if not ODD_M_TILE_GRID:
        return max_seqlen_q
    block_m = _block_m(num_q_tiles_per_wave)
    tiles = -(-int(max_seqlen_q) * int(gqa) // block_m)
    if tiles == 0 or tiles % 2:
        return max_seqlen_q
    return -(-(tiles + 1) * block_m // int(gqa))


def _ensure_thd_kernel(
    mask_left: bool,
    mask_right: bool,
    one_kv_tile: bool,
    return_lse: bool,
    has_sink: bool,
    has_descale: bool,
    gqa_ratio: int,
    qk_hdim: int = DEFAULT_QK_HDIM,
    v_hdim: int = DEFAULT_V_HDIM,
    q_dtype_str: str = DEFAULT_DTYPE,
    kv_dtype_str: str = DEFAULT_DTYPE,
    num_q_tiles_per_wave: int = WMMA_ROW_PER_WAVE,
    carry_state: bool = False,
):
    key = (
        "thd",
        bool(mask_left),
        bool(mask_right),
        bool(return_lse),
        bool(has_sink),
        bool(has_descale),
        int(gqa_ratio),
        int(qk_hdim),
        int(v_hdim),
        str(q_dtype_str),
        str(kv_dtype_str),
        int(num_q_tiles_per_wave),
        bool(one_kv_tile),
    ) + ((True,) if carry_state else ())
    if key in _launch_fns:
        return
    kernel = build_fmha_fwd_prefill_a8w8_m32x8(
        carry_state=bool(carry_state),
        layout="thd",
        qk_hdim=qk_hdim,
        v_hdim=v_hdim,
        mask_left=mask_left,
        mask_right=mask_right,
        one_kv_tile=one_kv_tile,
        return_lse=return_lse,
        has_sink=has_sink,
        has_descale=has_descale,
        gqa_ratio=gqa_ratio,
        q_dtype_str=q_dtype_str,
        kv_dtype_str=kv_dtype_str,
        num_q_tiles_per_wave=num_q_tiles_per_wave,
    )
    block_m = _block_m(num_q_tiles_per_wave)

    @flyc.jit
    def _launch(
        ptr_O: fx.Pointer,
        ptr_Q: fx.Pointer,
        ptr_K: fx.Pointer,
        ptr_V: fx.Pointer,
        ptr_LSE: fx.Pointer,
        ptr_sink: fx.Pointer,
        ptr_q_descale: fx.Pointer,
        ptr_k_descale: fx.Pointer,
        ptr_v_descale: fx.Pointer,
        ptr_cu_seqlens_q: fx.Pointer,
        ptr_cu_seqlens_k: fx.Pointer,
        softmax_scale: fx.Float32,
        stride_q_seq: fx.Int32,
        stride_k_seq: fx.Int32,
        stride_v_seq: fx.Int32,
        stride_o_seq: fx.Int32,
        stride_q_head: fx.Int32,
        stride_k_head: fx.Int32,
        stride_v_head: fx.Int32,
        stride_o_head: fx.Int32,
        stride_lse_seq: fx.Int32,
        stride_lse_head: fx.Int32,
        window_left: fx.Int32,
        window_right: fx.Int32,
        max_seqlen_q: fx.Int32,
        max_seqlen_k: fx.Int32,
        num_heads_kv: fx.Int32,
        batch_size: fx.Int32,
        stream: fx.Stream,
    ):
        # 3D grid: x = tiles over (seq, q_head_in_group) per kv-head,
        #          y = kv_head, z = batch. block = 256 (8 waves x wave32).
        grid_x = fx.Index(
            fx.ceildiv(fx.Uint32(max_seqlen_q * gqa_ratio), fx.Uint32(block_m))
        )
        grid_y = fx.Index(num_heads_kv)
        grid_z = fx.Index(batch_size)

        launcher = kernel(
            ptr_O,
            ptr_Q,
            ptr_K,
            ptr_V,
            ptr_LSE,
            ptr_sink,
            ptr_q_descale,
            ptr_k_descale,
            ptr_v_descale,
            ptr_cu_seqlens_q,
            ptr_cu_seqlens_k,
            softmax_scale,
            stride_q_seq,
            stride_k_seq,
            stride_v_seq,
            stride_o_seq,
            stride_q_head,
            stride_k_head,
            stride_v_head,
            stride_o_head,
            stride_lse_seq,
            stride_lse_head,
            window_left,
            window_right,
            max_seqlen_q,
            max_seqlen_k,
        )
        launcher.launch(
            grid=(grid_x, grid_y, grid_z),
            block=(BLOCK_SIZE, 1, 1),
            stream=stream,
        )

    _launch.compile_hints["llvm_options"] = {
        "amdgpu-expert-scheduling-mode": ENABLE_SCHED_MODE2,
        # "amdgpu-sched-strategy": "coexec",  # gfx1250 co-exec sched for warp specialization
    }
    _launch.compile_hints["waves_per_eu"] = 2
    _launch_fns[key] = _launch


def _ensure_thd_ps_kernel(
    mask_left: bool,
    mask_right: bool,
    has_sink: bool,
    has_descale: bool,
    gqa_ratio: int,
    qk_hdim: int,
    v_hdim: int,
    q_dtype_str: str,
    kv_dtype_str: str,
    num_q_tiles_per_wave: int,
    carry_state: bool = False,
):
    key = (
        "thd_ps",
        bool(carry_state),
        bool(mask_left),
        bool(mask_right),
        bool(has_sink),
        bool(has_descale),
        int(gqa_ratio),
        int(qk_hdim),
        int(v_hdim),
        str(q_dtype_str),
        str(kv_dtype_str),
        int(num_q_tiles_per_wave),
    )
    if key in _launch_fns:
        return key
    kernel = build_fmha_fwd_prefill_a8w8_m32x8(
        carry_state=bool(carry_state),
        layout="thd_ps",
        qk_hdim=qk_hdim,
        v_hdim=v_hdim,
        mask_left=mask_left,
        mask_right=mask_right,
        one_kv_tile=False,
        return_lse=True,
        has_sink=has_sink,
        has_descale=has_descale,
        gqa_ratio=gqa_ratio,
        q_dtype_str=q_dtype_str,
        kv_dtype_str=kv_dtype_str,
        num_q_tiles_per_wave=num_q_tiles_per_wave,
    )

    @flyc.jit
    def _launch(
        ptr_O: fx.Pointer,
        ptr_Q: fx.Pointer,
        ptr_K: fx.Pointer,
        ptr_V: fx.Pointer,
        ptr_LSE: fx.Pointer,
        ptr_sink: fx.Pointer,
        ptr_q_descale: fx.Pointer,
        ptr_k_descale: fx.Pointer,
        ptr_v_descale: fx.Pointer,
        ptr_cu_seqlens_q: fx.Pointer,
        ptr_cu_seqlens_k: fx.Pointer,
        ptr_ws_o: fx.Pointer,
        ptr_ws_lse: fx.Pointer,
        ptr_ws_meta: fx.Pointer,
        softmax_scale: fx.Float32,
        stride_q_seq: fx.Int32,
        stride_k_seq: fx.Int32,
        stride_v_seq: fx.Int32,
        stride_o_seq: fx.Int32,
        stride_q_head: fx.Int32,
        stride_k_head: fx.Int32,
        stride_v_head: fx.Int32,
        stride_o_head: fx.Int32,
        stride_lse_seq: fx.Int32,
        stride_lse_head: fx.Int32,
        window_left: fx.Int32,
        window_right: fx.Int32,
        num_heads_kv: fx.Int32,
        batch_size: fx.Int32,
        ws_items_cap: fx.Int32,
        lse_num_records_bytes: fx.Int32,
        force_pt: fx.Int32,
        num_wgs: fx.Int32,
        stream: fx.Stream,
    ):
        launcher = kernel(
            ptr_O,
            ptr_Q,
            ptr_K,
            ptr_V,
            ptr_LSE,
            ptr_sink,
            ptr_q_descale,
            ptr_k_descale,
            ptr_v_descale,
            ptr_cu_seqlens_q,
            ptr_cu_seqlens_k,
            ptr_ws_o,
            ptr_ws_lse,
            ptr_ws_meta,
            softmax_scale,
            stride_q_seq,
            stride_k_seq,
            stride_v_seq,
            stride_o_seq,
            stride_q_head,
            stride_k_head,
            stride_v_head,
            stride_o_head,
            stride_lse_seq,
            stride_lse_head,
            window_left,
            window_right,
            num_heads_kv,
            batch_size,
            ws_items_cap,
            lse_num_records_bytes,
            force_pt,
        )
        launcher.launch(
            grid=(fx.Index(num_wgs), 1, 1),
            block=(BLOCK_SIZE, 1, 1),
            stream=stream,
        )

    _launch.compile_hints["llvm_options"] = {
        "amdgpu-expert-scheduling-mode": ENABLE_SCHED_MODE2,
    }
    _launch.compile_hints["waves_per_eu"] = 2
    _launch_fns[key] = _launch
    return key


_ps_workspaces = {}


def _ps_workspace(device, v_hdim):
    """Split workspace, allocated once per device: partial O (bf16), partial LSE (fp32),
    and the int32 merge table."""
    key = (device, v_hdim)
    ws = _ps_workspaces.get(key)
    if ws is None:
        bm = _block_m(WMMA_ROW_PER_WAVE)
        ws = (
            torch.empty((PS_WS_ITEMS * bm, v_hdim), dtype=TORCH_O_DTYPE, device=device),
            torch.empty((PS_WS_ITEMS * bm,), dtype=torch.float32, device=device),
            torch.zeros(
                (PS_META_HDR + (PS_WS_ITEMS // 2) * PS_META_STRIDE,),
                dtype=torch.int32,
                device=device,
            ),
        )
        _ps_workspaces[key] = ws
    return ws


def _ps_merge(ws, out, lse, num_wgs, block_m, carry_state=False):
    """Combine the split parts of every split unit by LSE into ``out``/``lse``."""
    from .fmha_ps_merge import ps_merge

    ws_o, ws_lse, meta = ws
    ps_merge(
        ws_o,
        ws_lse,
        meta,
        out,
        lse,
        num_programs=4 * num_wgs,
        block_m=block_m,
        meta_hdr=PS_META_HDR,
        meta_stride=PS_META_STRIDE,
        max_parts=PS_MAX_PARTS,
        carry_state=carry_state,
    )


def _ensure_paged_kernel(
    mask_left: bool,
    mask_right: bool,
    return_lse: bool,
    has_sink: bool,
    has_descale: bool,
    gqa_ratio: int,
    page_size: int,
    qk_hdim: int = DEFAULT_QK_HDIM,
    v_hdim: int = DEFAULT_V_HDIM,
    q_dtype_str: str = DEFAULT_DTYPE,
    kv_dtype_str: str = DEFAULT_DTYPE,
    num_q_tiles_per_wave: int = WMMA_ROW_PER_WAVE,
):
    key = (
        "paged",
        bool(mask_left),
        bool(mask_right),
        bool(return_lse),
        bool(has_sink),
        bool(has_descale),
        int(gqa_ratio),
        int(qk_hdim),
        int(v_hdim),
        str(q_dtype_str),
        str(kv_dtype_str),
        int(num_q_tiles_per_wave),
        int(page_size),
    )
    if key in _launch_fns:
        return
    kernel = build_fmha_fwd_prefill_a8w8_m32x8(
        layout="paged",
        page_size=page_size,
        qk_hdim=qk_hdim,
        v_hdim=v_hdim,
        mask_left=mask_left,
        mask_right=mask_right,
        one_kv_tile=False,
        return_lse=return_lse,
        has_sink=has_sink,
        has_descale=has_descale,
        gqa_ratio=gqa_ratio,
        q_dtype_str=q_dtype_str,
        kv_dtype_str=kv_dtype_str,
        num_q_tiles_per_wave=num_q_tiles_per_wave,
    )
    block_m = _block_m(num_q_tiles_per_wave)

    @flyc.jit
    def _launch(
        ptr_O: fx.Pointer,
        ptr_Q: fx.Pointer,
        ptr_K: fx.Pointer,
        ptr_V: fx.Pointer,
        ptr_LSE: fx.Pointer,
        ptr_sink: fx.Pointer,
        ptr_q_descale: fx.Pointer,
        ptr_k_descale: fx.Pointer,
        ptr_v_descale: fx.Pointer,
        ptr_cu_seqlens_q: fx.Pointer,
        ptr_kv_indptr: fx.Pointer,
        ptr_kv_page_indices: fx.Pointer,
        ptr_kv_last_page_lens: fx.Pointer,
        softmax_scale: fx.Float32,
        stride_q_seq: fx.Int32,
        stride_k_seq: fx.Int32,
        stride_v_seq: fx.Int32,
        stride_o_seq: fx.Int32,
        stride_q_head: fx.Int32,
        stride_k_head: fx.Int32,
        stride_v_head: fx.Int32,
        stride_o_head: fx.Int32,
        stride_lse_seq: fx.Int32,
        stride_lse_head: fx.Int32,
        window_left: fx.Int32,
        window_right: fx.Int32,
        max_seqlen_q: fx.Int32,
        max_seqlen_k: fx.Int32,
        num_page_entries: fx.Int32,
        num_pool_rows: fx.Int32,
        num_heads_kv: fx.Int32,
        batch_size: fx.Int32,
        stream: fx.Stream,
    ):
        # 3D grid: x = tiles over (seq, q_head_in_group) per kv-head,
        #          y = kv_head, z = batch. block = 256 (8 waves x wave32).
        grid_x = fx.Index(
            fx.ceildiv(fx.Uint32(max_seqlen_q * gqa_ratio), fx.Uint32(block_m))
        )
        grid_y = fx.Index(num_heads_kv)
        grid_z = fx.Index(batch_size)

        launcher = kernel(
            ptr_O,
            ptr_Q,
            ptr_K,
            ptr_V,
            ptr_LSE,
            ptr_sink,
            ptr_q_descale,
            ptr_k_descale,
            ptr_v_descale,
            ptr_cu_seqlens_q,
            ptr_kv_indptr,
            ptr_kv_page_indices,
            ptr_kv_last_page_lens,
            softmax_scale,
            stride_q_seq,
            stride_k_seq,
            stride_v_seq,
            stride_o_seq,
            stride_q_head,
            stride_k_head,
            stride_v_head,
            stride_o_head,
            stride_lse_seq,
            stride_lse_head,
            window_left,
            window_right,
            num_page_entries,
            num_pool_rows,
        )
        launcher.launch(
            grid=(grid_x, grid_y, grid_z),
            block=(BLOCK_SIZE, 1, 1),
            stream=stream,
        )

    _launch.compile_hints["llvm_options"] = {
        "amdgpu-expert-scheduling-mode": ENABLE_SCHED_MODE2,
        # "amdgpu-sched-strategy": "coexec",  # gfx1250 co-exec sched for warp specialization
    }
    _launch.compile_hints["waves_per_eu"] = 2
    _launch_fns[key] = _launch


def _ensure_bshd_kernel(
    mask_left: bool,
    mask_right: bool,
    one_kv_tile: bool,
    return_lse: bool,
    has_sink: bool,
    has_descale: bool,
    gqa_ratio: int,
    qk_hdim: int = DEFAULT_QK_HDIM,
    v_hdim: int = DEFAULT_V_HDIM,
    q_dtype_str: str = DEFAULT_DTYPE,
    kv_dtype_str: str = DEFAULT_DTYPE,
    num_q_tiles_per_wave: int = WMMA_ROW_PER_WAVE,
):
    key = (
        "bshd",
        bool(mask_left),
        bool(mask_right),
        bool(return_lse),
        bool(has_sink),
        bool(has_descale),
        int(gqa_ratio),
        int(qk_hdim),
        int(v_hdim),
        str(q_dtype_str),
        str(kv_dtype_str),
        int(num_q_tiles_per_wave),
        bool(one_kv_tile),
    )
    if key in _launch_fns:
        return
    kernel = build_fmha_fwd_prefill_a8w8_m32x8(
        layout="bshd",
        qk_hdim=qk_hdim,
        v_hdim=v_hdim,
        mask_left=mask_left,
        mask_right=mask_right,
        one_kv_tile=one_kv_tile,
        return_lse=return_lse,
        has_sink=has_sink,
        has_descale=has_descale,
        gqa_ratio=gqa_ratio,
        q_dtype_str=q_dtype_str,
        kv_dtype_str=kv_dtype_str,
        num_q_tiles_per_wave=num_q_tiles_per_wave,
    )
    block_m = _block_m(num_q_tiles_per_wave)

    @flyc.jit
    def _launch(
        ptr_O: fx.Pointer,
        ptr_Q: fx.Pointer,
        ptr_K: fx.Pointer,
        ptr_V: fx.Pointer,
        ptr_LSE: fx.Pointer,
        ptr_sink: fx.Pointer,
        ptr_q_descale: fx.Pointer,
        ptr_k_descale: fx.Pointer,
        ptr_v_descale: fx.Pointer,
        softmax_scale: fx.Float32,
        stride_q_seq: fx.Int32,
        stride_k_seq: fx.Int32,
        stride_v_seq: fx.Int32,
        stride_o_seq: fx.Int32,
        stride_q_head: fx.Int32,
        stride_k_head: fx.Int32,
        stride_v_head: fx.Int32,
        stride_o_head: fx.Int32,
        stride_lse_seq: fx.Int32,
        stride_lse_head: fx.Int32,
        stride_lse_batch: fx.Int32,
        window_left: fx.Int32,
        window_right: fx.Int32,
        seq_len_q: fx.Int32,
        seq_len_k: fx.Int32,
        num_heads_kv: fx.Int32,
        batch_size: fx.Int32,
        stream: fx.Stream,
    ):
        # 3D grid: x = tiles over (seq, q_head_in_group) per kv-head,
        #          y = kv_head, z = batch. block = 256 (8 waves x wave32).
        grid_x = fx.Index(
            fx.ceildiv(fx.Uint32(seq_len_q * gqa_ratio), fx.Uint32(block_m))
        )
        grid_y = fx.Index(num_heads_kv)
        grid_z = fx.Index(batch_size)

        launcher = kernel(
            ptr_O,
            ptr_Q,
            ptr_K,
            ptr_V,
            ptr_LSE,
            ptr_sink,
            ptr_q_descale,
            ptr_k_descale,
            ptr_v_descale,
            softmax_scale,
            stride_q_seq,
            stride_k_seq,
            stride_v_seq,
            stride_o_seq,
            stride_q_head,
            stride_k_head,
            stride_v_head,
            stride_o_head,
            stride_lse_seq,
            stride_lse_head,
            stride_lse_batch,
            window_left,
            window_right,
            seq_len_q,
            seq_len_k,
        )
        launcher.launch(
            grid=(grid_x, grid_y, grid_z),
            block=(BLOCK_SIZE, 1, 1),
            stream=stream,
        )

    _launch.compile_hints["llvm_options"] = {
        "amdgpu-expert-scheduling-mode": ENABLE_SCHED_MODE2,
        # "amdgpu-sched-strategy": "coexec",  # gfx1250 co-exec sched for warp specialization
    }
    _launch.compile_hints["waves_per_eu"] = 2
    _launch_fns[key] = _launch


def flash_attn_a8w8_m32x8_supported(
    mode,
    q,
    k,
    v,
    *,
    dropout_p=0.0,
    window_size=(-1, -1),
    bias=None,
    alibi_slopes=None,
    sink=None,
    block_table=None,
    return_attn_probs=False,
    kv_indptr=None,
    kv_page_indices=None,
    kv_last_page_lens=None,
) -> bool:
    """Can the m32x8 prefill kernel serve this configuration?

    Metadata only -- no allocation, compile or launch -- so a caller can gate before
    preparing buffers. The ARCH is the caller's business: dispatch on it first and call
    this only on gfx1250. ``mode`` is "varlen", "batch" or "paged": the first two share
    one kernel, while "paged" adds the page-pool screens and reads the kv_* arguments.

    Per-call buffer validation (a caller-provided ``out``) stays with the entry points.
    """
    qk_hdim, v_hdim = q.shape[-1], v.shape[-1]
    num_q, num_kv = q.shape[-2], k.shape[-2]
    if mode == "paged":
        if not _is_paged_layout_supported(
            k, v, kv_indptr, kv_page_indices, kv_last_page_lens
        ):
            return False
    elif block_table is not None:
        return False
    sink_supported = sink is None or (
        torch.is_tensor(sink)
        and sink.dtype == torch.float32
        and sink.device == q.device
    )
    return (
        qk_hdim in SUPPORTED_QK_HDIM
        and v_hdim in SUPPORTED_V_HDIM
        and k.shape[-1] == qk_hdim
        and q.dtype in _TORCH_IN_DTYPES
        and k.dtype in _TORCH_IN_DTYPES
        and v.dtype == k.dtype
        and num_kv > 0
        and num_q % num_kv == 0
        and supports_gqa_ratio(num_q // max(num_kv, 1), v_hdim)
        and dropout_p == 0.0
        and (len(window_size) < 3 or window_size[2] == 0)
        and bias is None
        and alibi_slopes is None
        and sink_supported
        and not return_attn_probs
    )


def flash_attn_varlen_a8w8_m32x8(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),
    out=None,
    return_lse=False,
    sink=None,
    lse=None,
    q_descale=None,
    k_descale=None,
    v_descale=None,
    num_q_tiles_per_wave=None,
    carry_state=False,
    persistent=None,
):
    """Host entry — varlen THD, qk_hdim in {64,128,192,256} / v_hdim in {64,128,256}, fp8 or bf8.

    ``carry_state=True``: ``out`` / ``lse`` (both required, ``return_lse`` implied) hold a
    running attention state of the same queries over OTHER keys -- e.g. the new tokens'
    causal pass, or earlier context chunks. This launch merges its own partial softmax
    into them in place (LSE recombination, as merge_attn_states does), so chunked prefill
    needs no separate merge pass; a sequence with no keys in this launch keeps its rows
    untouched. A row whose ``lse`` is -inf has no prior state and its ``out`` is not read.

    ``persistent``: None takes the persistent split-KV launch for unmasked calls
    (``AITER_FMHA_A8W8_PERSISTENT=0`` disables it) and the grid launch for causal or
    windowed ones; True / False force either.

    ``window_size`` is ``(left, right)`` sliding-window bounds, ``-1`` meaning infinite
    on that side; ``causal`` forces ``right=0``. Only finiteness is baked into the kernel
    (compile-time ``mask_left``/``mask_right``), so one variant serves any magnitude.
    ``sink`` is an optional ``[nheads_q]`` fp32 per-head logit in the scaled-score domain,
    its presence likewise compile-time. ``lse`` is an optional caller-provided
    ``[nheads_q, total_q]`` fp32 buffer, allocated here when ``return_lse`` and None.
    """
    assert (
        q.dtype in SUPPORTED_TORCH_DTYPES and k.dtype in SUPPORTED_TORCH_DTYPES
    ), f"Expected fp8 or bf8, got {q.dtype}/{k.dtype}"
    assert v.dtype == k.dtype, f"k/v dtype must match, got {k.dtype}/{v.dtype}"
    q_dtype_str = _TORCH_IN_DTYPES[q.dtype]
    kv_dtype_str = _TORCH_IN_DTYPES[k.dtype]
    qk_hdim = q.shape[-1]
    assert (
        qk_hdim in SUPPORTED_QK_HDIM
    ), f"Expected qk_hdim in {SUPPORTED_QK_HDIM}, got {qk_hdim}"
    v_hdim = v.shape[-1]
    assert (
        v_hdim in SUPPORTED_V_HDIM
    ), f"Expected v_hdim in {SUPPORTED_V_HDIM}, got {v_hdim}"

    total_q_tokens = q.shape[0]
    batch = cu_seqlens_q.shape[0] - 1
    nheads_q = q.shape[1]
    nheads_k = k.shape[1]
    assert (
        nheads_q % nheads_k == 0
    ), f"nheads_q={nheads_q} must be a multiple of nheads_k={nheads_k}"
    gqa = nheads_q // nheads_k

    has_sink = sink is not None
    if has_sink:
        assert sink.dtype == torch.float32, f"sink must be fp32, got {sink.dtype}"
        assert (
            sink.dim() == 1 and sink.shape[0] == nheads_q
        ), f"sink must be [nheads_q={nheads_q}], got {tuple(sink.shape)}"
    # ptr_sink is only read when has_sink; pass q as a valid placeholder otherwise.
    sink_ptr = sink if has_sink else q

    # Per-tensor fp32 descales, all three or none, as gfx950 requires. Presence is
    # compile-time; the pointers are always passed, with q as a placeholder when unused.
    _descales = (q_descale, k_descale, v_descale)
    has_descale = any(d is not None for d in _descales)
    if has_descale:
        assert all(
            d is not None for d in _descales
        ), "q/k/v_descale must be given together"
        for _name, _d in zip(("q", "k", "v"), _descales):
            assert (
                torch.is_tensor(_d) and _d.dtype == torch.float32
            ), f"{_name}_descale must be a float32 tensor, got {type(_d)}"
            assert _d.numel() == 1, f"{_name}_descale must hold one element"
            assert _d.device == q.device, f"{_name}_descale must be on {q.device}"
    qd_ptr, kd_ptr, vd_ptr = _descales if has_descale else (q, q, q)

    if softmax_scale is None:
        softmax_scale = 1.0 / (q.shape[-1] ** 0.5)

    # Sliding window: causal forces right=0. Finiteness (>=0) is compile-time
    # (mask_left/mask_right); the magnitudes ride along as runtime Int32 args.
    win_left, win_right = int(window_size[0]), int(window_size[1])
    if causal:
        win_right = 0
    mask_left = win_left >= 0
    mask_right = win_right >= 0
    window_left = max(win_left, 0)
    window_right = max(win_right, 0)

    carry_state = bool(carry_state)
    if carry_state:
        assert (
            out is not None and lse is not None
        ), "carry_state merges into the caller's out/lse"
        assert not has_sink, "carry_state does not take a sink"
        assert out.dtype == TORCH_O_DTYPE and tuple(out.shape) == (
            total_q_tokens,
            nheads_q,
            v_hdim,
        ), f"carry_state out must be [{total_q_tokens}, {nheads_q}, {v_hdim}] bf16"
        assert out.stride(2) == 1, "carry_state out needs unit stride on head_dim"
        assert lse.dtype == torch.float32 and tuple(lse.shape) == (
            nheads_q,
            total_q_tokens,
        ), f"carry_state lse must be [{nheads_q}, {total_q_tokens}] fp32"
        return_lse = True

    if out is None:
        out = torch.empty(
            (total_q_tokens, nheads_q, v_hdim),
            dtype=TORCH_O_DTYPE,
            device=q.device,
        )
    if return_lse:
        if lse is None:
            # [nheads_q, total_q] is the aiter varlen LSE convention, what CK and the
            # gfx1250 ASM varlen kernel both return. The kernel is stride-driven, so the
            # layout lives entirely in the two strides below.
            lse = torch.empty(
                (nheads_q, total_q_tokens), dtype=torch.float32, device=q.device
            )
        lse_ptr = lse
        stride_lse_seq = lse.stride(1)
        stride_lse_head = lse.stride(0)
    else:
        lse_ptr = q
        stride_lse_seq = 0
        stride_lse_head = 0

    # Q/K/V/O strides in ELEMENTS (TDM loaders consume them directly).
    stride_q_seq = q.stride(0)
    stride_k_seq = k.stride(0)
    stride_v_seq = v.stride(0)
    stride_o_seq = out.stride(0)
    stride_q_head = q.stride(1)
    stride_k_head = k.stride(1)
    stride_v_head = v.stride(1)
    stride_o_head = out.stride(1)

    if num_q_tiles_per_wave is None:
        num_q_tiles_per_wave = _pick_num_q_tiles(
            nheads_k, batch, max_seqlen_q, gqa, v_hdim
        )
    num_q_tiles_per_wave = int(num_q_tiles_per_wave)

    one_kv_tile = _pick_one_kv_tile(max_seqlen_k, qk_hdim, v_hdim, kv_dtype_str)
    if persistent is None:
        # Masked calls are scheduled but never KV-split, and that schedule loses
        # to the grid on long sequences.
        persistent = (
            os.environ.get("AITER_FMHA_A8W8_PERSISTENT", "1") != "0"
            and not mask_left
            and not mask_right
        )
    if (
        persistent
        and not one_kv_tile
        and batch <= PS_MAX_BATCH
        and num_q_tiles_per_wave == WMMA_ROW_PER_WAVE
        and total_q_tokens > 0
    ):
        if not return_lse:
            # The persistent kernel always writes LSE; give it a scratch buffer.
            lse_ptr = torch.empty(
                (nheads_q, total_q_tokens), dtype=torch.float32, device=q.device
            )
            stride_lse_seq = lse_ptr.stride(1)
            stride_lse_head = lse_ptr.stride(0)
        lse_buf = lse_ptr
        lse_nrec = (
            (total_q_tokens - 1) * stride_lse_seq + (nheads_q - 1) * stride_lse_head + 1
        ) * 4
        key = _ensure_thd_ps_kernel(
            mask_left,
            mask_right,
            has_sink,
            has_descale,
            gqa,
            qk_hdim,
            v_hdim,
            q_dtype_str,
            kv_dtype_str,
            num_q_tiles_per_wave,
            carry_state=carry_state,
        )
        ws = _ps_workspace(q.device, v_hdim)
        num_wgs = int(os.environ.get("AITER_FMHA_A8W8_PS_WGS", "0")) or get_cu_num()
        _run_compiled(
            _launch_fns[key],
            out,
            q,
            k,
            v,
            lse_buf,
            sink_ptr,
            qd_ptr,
            kd_ptr,
            vd_ptr,
            cu_seqlens_q,
            cu_seqlens_k,
            ws[0],
            ws[1],
            ws[2],
            softmax_scale,
            stride_q_seq,
            stride_k_seq,
            stride_v_seq,
            stride_o_seq,
            stride_q_head,
            stride_k_head,
            stride_v_head,
            stride_o_head,
            stride_lse_seq,
            stride_lse_head,
            window_left,
            window_right,
            nheads_k,
            batch,
            PS_WS_ITEMS,
            lse_nrec,
            int(os.environ.get("AITER_FMHA_A8W8_PS_FORCE_PT", "0")),
            num_wgs,
            torch.cuda.current_stream(),
        )
        if not mask_left and not mask_right and gqa == 1 and not has_sink:
            _ps_merge(ws, out, lse_buf, num_wgs, _block_m(num_q_tiles_per_wave))
        if return_lse:
            return out, lse
        return out

    grid_max_seqlen_q = _xcd_balanced_max_seqlen_q(
        max_seqlen_q, gqa, num_q_tiles_per_wave
    )

    _ensure_thd_kernel(
        mask_left,
        mask_right,
        one_kv_tile,
        bool(return_lse),
        has_sink,
        has_descale,
        gqa,
        qk_hdim=qk_hdim,
        v_hdim=v_hdim,
        q_dtype_str=q_dtype_str,
        kv_dtype_str=kv_dtype_str,
        num_q_tiles_per_wave=num_q_tiles_per_wave,
        carry_state=carry_state,
    )

    _run_compiled(
        _launch_fns[
            (
                "thd",
                mask_left,
                mask_right,
                bool(return_lse),
                has_sink,
                has_descale,
                gqa,
                qk_hdim,
                v_hdim,
                q_dtype_str,
                kv_dtype_str,
                num_q_tiles_per_wave,
                one_kv_tile,
            )
            + ((True,) if carry_state else ())
        ],
        out,
        q,
        k,
        v,
        lse_ptr,
        sink_ptr,
        qd_ptr,
        kd_ptr,
        vd_ptr,
        cu_seqlens_q,
        cu_seqlens_k,
        softmax_scale,
        stride_q_seq,
        stride_k_seq,
        stride_v_seq,
        stride_o_seq,
        stride_q_head,
        stride_k_head,
        stride_v_head,
        stride_o_head,
        stride_lse_seq,
        stride_lse_head,
        window_left,
        window_right,
        grid_max_seqlen_q,
        max_seqlen_k,
        nheads_k,
        batch,
        torch.cuda.current_stream(),
    )

    if return_lse:
        return out, lse
    return out


def flash_attn_paged_a8w8_m32x8(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    qo_indptr: torch.Tensor,
    kv_indptr: torch.Tensor,
    kv_page_indices: torch.Tensor,
    kv_last_page_lens: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),
    out=None,
    return_lse=False,
    sink=None,
    lse=None,
    q_descale=None,
    k_descale=None,
    v_descale=None,
    num_q_tiles_per_wave=None,
):
    """Host entry — PAGED KV, qk_hdim in {64,128,192,256} / v_hdim in {64,128,256}, fp8 or bf8.

    ``k``/``v`` are the page pool, ``[num_blocks, page_size, nheads_k, hdim]`` (or
    ``[num_blocks, nheads_k, hdim]`` at page size 1). ``qo_indptr`` gives this batch's Q
    token ranges, as it does for THD; the KV side is the AITER page triple. ``page_size``
    is compile-time, so one build serves one page size.

    ``window_size`` is ``(left, right)`` sliding-window bounds, ``-1`` meaning infinite
    on that side; ``causal`` forces ``right=0``. Only finiteness is baked into the kernel
    (compile-time ``mask_left``/``mask_right``), so one variant serves any magnitude.
    ``sink`` is an optional ``[nheads_q]`` fp32 per-head logit in the scaled-score domain,
    its presence likewise compile-time. ``lse`` is an optional caller-provided
    ``[nheads_q, total_q]`` fp32 buffer, allocated here when ``return_lse`` and None.
    """
    assert (
        q.dtype in SUPPORTED_TORCH_DTYPES and k.dtype in SUPPORTED_TORCH_DTYPES
    ), f"Expected 8-bit q/kv, got {q.dtype}/{k.dtype}"
    assert v.dtype == k.dtype, f"k/v dtype must match, got {k.dtype}/{v.dtype}"
    q_dtype_str = _TORCH_IN_DTYPES[q.dtype]
    kv_dtype_str = _TORCH_IN_DTYPES[k.dtype]
    qk_hdim = q.shape[-1]
    assert (
        qk_hdim in SUPPORTED_QK_HDIM
    ), f"Expected qk_hdim in {SUPPORTED_QK_HDIM}, got {qk_hdim}"
    v_hdim = v.shape[-1]
    assert (
        v_hdim in SUPPORTED_V_HDIM
    ), f"Expected v_hdim in {SUPPORTED_V_HDIM}, got {v_hdim}"

    total_q_tokens = q.shape[0]
    batch = qo_indptr.shape[0] - 1
    assert k.dim() == v.dim() and k.dim() in (3, 4), (
        f"paged k/v must be [num_blocks, page_size, nheads_k, hdim] or "
        f"[num_blocks, nheads_k, hdim]; got {tuple(k.shape)}"
    )
    page_size = 1 if k.dim() == 3 else k.shape[1]
    assert (
        page_size & (page_size - 1) == 0
    ), f"page_size must be a power of two, got {page_size}"
    # pages contiguous => row p*page_size + o is token o of page p; stride(-3) is the
    # per-token stride at both ranks.
    assert k.stride(0) == page_size * k.stride(-3) and v.stride(
        0
    ) == page_size * v.stride(
        -3
    ), "paged k/v need contiguous pages (stride(0) == page_size * stride(-3))"
    num_pool_rows = k.shape[0] * page_size
    num_page_entries = kv_page_indices.numel()
    # Token-row view: what the gather addresses, and one variant per page_size
    # (dispatch specializes on rank, so 3-D and 4-D pools would not share one).
    k = k.as_strided(
        (num_pool_rows, k.shape[-2], k.shape[-1]),
        (k.stride(-3), k.stride(-2), k.stride(-1)),
    )
    v = v.as_strided(
        (num_pool_rows, v.shape[-2], v.shape[-1]),
        (v.stride(-3), v.stride(-2), v.stride(-1)),
    )
    nheads_q = q.shape[1]
    nheads_k = k.shape[-2]
    assert (
        nheads_q % nheads_k == 0
    ), f"nheads_q={nheads_q} must be a multiple of nheads_k={nheads_k}"
    gqa = nheads_q // nheads_k

    has_sink = sink is not None
    if has_sink:
        assert sink.dtype == torch.float32, f"sink must be fp32, got {sink.dtype}"
        assert (
            sink.dim() == 1 and sink.shape[0] == nheads_q
        ), f"sink must be [nheads_q={nheads_q}], got {tuple(sink.shape)}"
    # ptr_sink is only read when has_sink; pass q as a valid placeholder otherwise.
    sink_ptr = sink if has_sink else q

    # Per-tensor fp32 descales, all three or none, as gfx950 requires. Presence is
    # compile-time; the pointers are always passed, with q as a placeholder when unused.
    _descales = (q_descale, k_descale, v_descale)
    has_descale = any(d is not None for d in _descales)
    if has_descale:
        assert all(
            d is not None for d in _descales
        ), "q/k/v_descale must be given together"
        for _name, _d in zip(("q", "k", "v"), _descales):
            assert (
                torch.is_tensor(_d) and _d.dtype == torch.float32
            ), f"{_name}_descale must be a float32 tensor, got {type(_d)}"
            assert _d.numel() == 1, f"{_name}_descale must hold one element"
            assert _d.device == q.device, f"{_name}_descale must be on {q.device}"
    qd_ptr, kd_ptr, vd_ptr = _descales if has_descale else (q, q, q)

    if softmax_scale is None:
        softmax_scale = 1.0 / (q.shape[-1] ** 0.5)

    # Sliding window: causal forces right=0. Finiteness (>=0) is compile-time
    # (mask_left/mask_right); the magnitudes ride along as runtime Int32 args.
    win_left, win_right = int(window_size[0]), int(window_size[1])
    if causal:
        win_right = 0
    mask_left = win_left >= 0
    mask_right = win_right >= 0
    window_left = max(win_left, 0)
    window_right = max(win_right, 0)

    if out is None:
        out = torch.empty(
            (total_q_tokens, nheads_q, v_hdim), dtype=TORCH_O_DTYPE, device=q.device
        )
    if return_lse:
        if lse is None:
            # [nheads_q, total_q] is the aiter varlen LSE convention, what CK and the
            # gfx1250 ASM varlen kernel both return. The kernel is stride-driven, so the
            # layout lives entirely in the two strides below.
            lse = torch.empty(
                (nheads_q, total_q_tokens), dtype=torch.float32, device=q.device
            )
        lse_ptr = lse
        stride_lse_seq = lse.stride(1)
        stride_lse_head = lse.stride(0)
    else:
        lse_ptr = q
        stride_lse_seq = 0
        stride_lse_head = 0

    # Q/K/V/O strides in ELEMENTS (TDM loaders consume them directly).
    stride_q_seq = q.stride(0)
    stride_k_seq = k.stride(-3)  # per-TOKEN stride in the pool
    stride_v_seq = v.stride(-3)
    stride_o_seq = out.stride(0)
    stride_q_head = q.stride(1)
    stride_k_head = k.stride(-2)
    stride_v_head = v.stride(-2)
    stride_o_head = out.stride(1)

    if num_q_tiles_per_wave is None:
        num_q_tiles_per_wave = _pick_num_q_tiles(
            nheads_k, batch, max_seqlen_q, gqa, v_hdim
        )
    num_q_tiles_per_wave = int(num_q_tiles_per_wave)

    _ensure_paged_kernel(
        mask_left,
        mask_right,
        bool(return_lse),
        has_sink,
        has_descale,
        gqa,
        page_size,
        qk_hdim=qk_hdim,
        v_hdim=v_hdim,
        q_dtype_str=q_dtype_str,
        kv_dtype_str=kv_dtype_str,
        num_q_tiles_per_wave=num_q_tiles_per_wave,
    )

    _run_compiled(
        _launch_fns[
            (
                "paged",
                mask_left,
                mask_right,
                bool(return_lse),
                has_sink,
                has_descale,
                gqa,
                qk_hdim,
                v_hdim,
                q_dtype_str,
                kv_dtype_str,
                num_q_tiles_per_wave,
                page_size,
            )
        ],
        out,
        q,
        k,
        v,
        lse_ptr,
        sink_ptr,
        qd_ptr,
        kd_ptr,
        vd_ptr,
        qo_indptr,
        kv_indptr,
        kv_page_indices,
        kv_last_page_lens,
        softmax_scale,
        stride_q_seq,
        stride_k_seq,
        stride_v_seq,
        stride_o_seq,
        stride_q_head,
        stride_k_head,
        stride_v_head,
        stride_o_head,
        stride_lse_seq,
        stride_lse_head,
        window_left,
        window_right,
        _xcd_balanced_max_seqlen_q(max_seqlen_q, gqa, num_q_tiles_per_wave),
        max_seqlen_k,
        num_page_entries,
        num_pool_rows,
        nheads_k,
        batch,
        torch.cuda.current_stream(),
    )

    if return_lse:
        return out, lse
    return out


def flash_attn_batch_a8w8_m32x8(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),
    out=None,
    return_lse=False,
    sink=None,
    lse=None,
    q_descale=None,
    k_descale=None,
    v_descale=None,
    num_q_tiles_per_wave=None,
):
    """Host entry — batched BSHD ``[B, S, H, D]``, qk_hdim in {64,128,192,256} / v_hdim in {64,128,256}, fp8 or bf8.

    Uses the dedicated BSHD kernel with a uniform ``seq_len`` scalar (no cu_seqlens), so
    there is nothing transient to bake into a CUDA graph. ``window_size``, ``causal``,
    ``sink`` and ``lse`` behave as in ``flash_attn_varlen_a8w8_m32x8``; the LSE buffer is
    ``[B, nheads_q, S_q]`` here.
    """
    assert (
        q.dtype in SUPPORTED_TORCH_DTYPES and k.dtype in SUPPORTED_TORCH_DTYPES
    ), f"Expected fp8 or bf8, got {q.dtype}/{k.dtype}"
    assert v.dtype == k.dtype, f"k/v dtype must match, got {k.dtype}/{v.dtype}"
    q_dtype_str = _TORCH_IN_DTYPES[q.dtype]
    kv_dtype_str = _TORCH_IN_DTYPES[k.dtype]
    assert q.dim() == 4, f"Expected 4D BSHD tensor, got rank {q.dim()}"
    qk_hdim = q.shape[-1]
    assert (
        qk_hdim in SUPPORTED_QK_HDIM
    ), f"Expected qk_hdim in {SUPPORTED_QK_HDIM}, got {qk_hdim}"
    v_hdim = v.shape[-1]
    assert (
        v_hdim in SUPPORTED_V_HDIM
    ), f"Expected v_hdim in {SUPPORTED_V_HDIM}, got {v_hdim}"

    batch, seq_len_q, nheads_q, _ = q.shape
    seq_len_k = k.shape[1]
    nheads_k = k.shape[2]
    assert (
        nheads_q % nheads_k == 0
    ), f"nheads_q={nheads_q} must be a multiple of nheads_k={nheads_k}"
    gqa = nheads_q // nheads_k

    has_sink = sink is not None
    if has_sink:
        assert sink.dtype == torch.float32, f"sink must be fp32, got {sink.dtype}"
        assert (
            sink.dim() == 1 and sink.shape[0] == nheads_q
        ), f"sink must be [nheads_q={nheads_q}], got {tuple(sink.shape)}"
    # ptr_sink is only read when has_sink; pass q as a valid placeholder otherwise.
    sink_ptr = sink if has_sink else q

    # Per-tensor fp32 descales, all three or none, as gfx950 requires. Presence is
    # compile-time; the pointers are always passed, with q as a placeholder when unused.
    _descales = (q_descale, k_descale, v_descale)
    has_descale = any(d is not None for d in _descales)
    if has_descale:
        assert all(
            d is not None for d in _descales
        ), "q/k/v_descale must be given together"
        for _name, _d in zip(("q", "k", "v"), _descales):
            assert (
                torch.is_tensor(_d) and _d.dtype == torch.float32
            ), f"{_name}_descale must be a float32 tensor, got {type(_d)}"
            assert _d.numel() == 1, f"{_name}_descale must hold one element"
            assert _d.device == q.device, f"{_name}_descale must be on {q.device}"
    qd_ptr, kd_ptr, vd_ptr = _descales if has_descale else (q, q, q)

    if softmax_scale is None:
        softmax_scale = 1.0 / (q.shape[-1] ** 0.5)

    # Sliding window: causal forces right=0. Finiteness (>=0) is compile-time
    # (mask_left/mask_right); the magnitudes ride along as runtime Int32 args.
    win_left, win_right = int(window_size[0]), int(window_size[1])
    if causal:
        win_right = 0
    mask_left = win_left >= 0
    mask_right = win_right >= 0
    window_left = max(win_left, 0)
    window_right = max(win_right, 0)

    if out is None:
        out = torch.empty(
            (batch, seq_len_q, nheads_q, v_hdim),
            dtype=TORCH_O_DTYPE,
            device=q.device,
        )
    if return_lse:
        if lse is None:
            lse = torch.empty(
                (batch, nheads_q, seq_len_q), dtype=torch.float32, device=q.device
            )
        lse_ptr = lse
        stride_lse_seq = lse.stride(2)
        stride_lse_head = lse.stride(1)
        stride_lse_batch = lse.stride(0)
    else:
        lse_ptr = q
        stride_lse_seq = 0
        stride_lse_head = 0
        stride_lse_batch = 0

    # Empty tensor — skip the launch (host-known dims, no device sync). No queries means
    # no rows to write; no keys means an empty softmax set, so O=0 and LSE=-inf, or
    # LSE=sink[head] with a sink, exp(sink) being the only surviving term.
    if seq_len_q == 0 or seq_len_k == 0:
        if seq_len_q > 0 and seq_len_k == 0:
            out.zero_()
            if return_lse:
                if sink is not None:
                    lse.copy_(
                        sink.to(device=lse.device, dtype=lse.dtype)
                        .view(1, -1, 1)
                        .expand_as(lse)
                    )
                else:
                    lse.fill_(float("-inf"))
        return (out, lse) if return_lse else out

    # BSHD: seq is dim 1, head dim 2; the per-batch base is derived in-kernel as
    # batch_idx * seq_len. Strides in ELEMENTS (TDM loaders consume them directly).
    stride_q_seq = q.stride(1)
    stride_k_seq = k.stride(1)
    stride_v_seq = v.stride(1)
    stride_o_seq = out.stride(1)
    stride_q_head = q.stride(2)
    stride_k_head = k.stride(2)
    stride_v_head = v.stride(2)
    stride_o_head = out.stride(2)

    if num_q_tiles_per_wave is None:
        num_q_tiles_per_wave = _pick_num_q_tiles(
            nheads_k, batch, seq_len_q, gqa, v_hdim
        )
    num_q_tiles_per_wave = int(num_q_tiles_per_wave)

    one_kv_tile = _pick_one_kv_tile(seq_len_k, qk_hdim, v_hdim, kv_dtype_str)

    _ensure_bshd_kernel(
        mask_left,
        mask_right,
        one_kv_tile,
        bool(return_lse),
        has_sink,
        has_descale,
        gqa,
        qk_hdim=qk_hdim,
        v_hdim=v_hdim,
        q_dtype_str=q_dtype_str,
        kv_dtype_str=kv_dtype_str,
        num_q_tiles_per_wave=num_q_tiles_per_wave,
    )

    _run_compiled(
        _launch_fns[
            (
                "bshd",
                mask_left,
                mask_right,
                bool(return_lse),
                has_sink,
                has_descale,
                gqa,
                qk_hdim,
                v_hdim,
                q_dtype_str,
                kv_dtype_str,
                num_q_tiles_per_wave,
                one_kv_tile,
            )
        ],
        out,
        q,
        k,
        v,
        lse_ptr,
        sink_ptr,
        qd_ptr,
        kd_ptr,
        vd_ptr,
        softmax_scale,
        stride_q_seq,
        stride_k_seq,
        stride_v_seq,
        stride_o_seq,
        stride_q_head,
        stride_k_head,
        stride_v_head,
        stride_o_head,
        stride_lse_seq,
        stride_lse_head,
        stride_lse_batch,
        window_left,
        window_right,
        seq_len_q,
        seq_len_k,
        nheads_k,
        batch,
        torch.cuda.current_stream(),
    )

    if return_lse:
        return out, lse
    return out
