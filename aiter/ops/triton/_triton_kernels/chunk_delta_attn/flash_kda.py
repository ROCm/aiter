# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
# Adapted from flash-linear-attention: Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li

"""
Two-kernel fused KDA forward, following the FlashKDA kernel split.

The default ``chunk_delta_attn_fwd`` pipeline runs five kernels (l2norm, beta
sigmoid, gate cumsum, intra, delta_h, gla_o) and materializes ``g_cumsum``,
``Aqk``, ``Akk``, ``w``, ``u`` and ``h`` in HBM between them. This path collapses
the same math into two:

  K1 (``_flash_kda_prepare_kernel``), token-parallel over chunks:
      q/k L2 norm, gate activation + chunk cumsum, beta sigmoid, the decayed
      q/k tiles, the per-chunk gate total, ``Mqk``, and the triangular inverse
      ``(I - L)^-1``. Everything lands in a per-chunk workspace.

  K2 (``_flash_kda_segment_kernel``), sequential over chunks within one
      segment: the delta-rule recurrence and the output projection, reading
      only the K1 workspace plus ``v``/``beta``.

The recurrent state stays in registers across the whole segment in K2, which is
the point of the split: the baseline writes ``h`` for every chunk to HBM and
reads it back in the output kernel.

A segment defaults to the whole sequence. When that leaves the device idle --
K2's block count is only ``n_segments * H * (V / BW)``, so a 12-head model gets
96 blocks against 256 CUs -- the sequence is instead cut into fixed-length
segments that run concurrently. Each segment's state update is affine in its
incoming state, ``h' = A_seg h + b_seg``, and both parts are produced by this
same kernel: seeding it with ``h = 0`` gives ``b_seg``, and seeding it with
``h = I`` and ``v = 0`` propagates a basis whose result is ``A_seg``, avoiding
any explicit per-chunk operator product. A short scan (``n_segments`` steps
instead of ``n_chunks``) then recovers each segment's true incoming state, and a
final pass re-runs the segments to write outputs.

Restrictions (the caller is expected to check these and fall back):
    * ``K == V == 128`` and ``HV == H`` (no GVA).
    * ``chunk_size == 32`` (see ``FLASH_KDA_CHUNK``).
    * The fused-gate sigmoid path only, i.e. ``use_gate_in_kernel`` with
      ``lower_bound`` set, plus in-kernel qk l2norm and beta sigmoid.
    * ``safe_gate``. K1 bounds the intra-chunk decay with a midpoint pivot
      rather than the sub-chunk pivot ``safe_gate`` selects in the default
      intra kernel, so it answers that request by a different route and has no
      unpivoted mode to offer a caller who declined it.
    * Forward only, and no intermediates for a backward pass.
"""

import functools
import math
import os

import torch
import triton
import triton.language as tl

from aiter.ops.triton._triton_kernels.chunk_delta_attn.chunk_delta_attn_utils import (
    CHUNK_DELTA_ATTN_TRITON_AUTOTUNE,
    autotune_cache_kwargs,
    chunk_delta_attn_autotune_configs,
    chunk_delta_attn_tuned_config,
    chunk_delta_attn_tuned_config_shortlist,
    exp,
    exp2,
    input_guard,
    tensor_cache,
)
from aiter.ops.triton._triton_kernels.chunk_delta_attn.fast_launch import fast_launch
from aiter.ops.triton._triton_kernels.chunk_delta_attn.utils.index import (
    prepare_chunk_indices,
)
from aiter.ops.triton.utils._triton import arch_info
from aiter.ops.triton.utils.logger import AiterTritonLogger

_LOGGER = AiterTritonLogger()

# On by default. `flash_kda_supported` decides per call and anything outside the
# restrictions above keeps the default pipeline, so this switch only exists to
# force that pipeline anyway -- to A/B the two, or to back out if the fused path
# regresses on a shape the admission check still accepts.
AITER_FDA_ENABLE: bool = os.getenv("AITER_FDA_ENABLE", "1").lower() in (
    "1",
    "true",
    "yes",
    "on",
)

_GLUON_SEL: str = os.getenv("AITER_FDA_USE_GLUON", "1").lower()
_GLUON_BOTH: bool = _GLUON_SEL in ("1", "true", "yes", "on", "all")
AITER_FDA_USE_GLUON_K1: bool = _GLUON_BOTH or _GLUON_SEL == "k1"
AITER_FDA_USE_GLUON_K2: bool = _GLUON_BOTH or _GLUON_SEL == "k2"

DEVICE_ARCH = arch_info.get_arch()
_GLUON_ARCH_OK: bool = DEVICE_ARCH == "gfx950"

_LOGGED_ROUTES: set = set()

# The K-dimension is consumed as two 64-wide halves so the recurrent state fits
# a pair of [64, BW] register tiles in K2. Relaxing this means reworking K2's
# state blocking, not just the loop bound.
FLASH_KDA_K: int = 128
# C = 32, not 64. The intra-chunk decay factors are re-centered on the chunk
# midpoint, which keeps them in fp32 for a span of about +-70 log2 units; the
# Kimi gate spends roughly 3.6 of those per token, so 32 rows fit and 64 do not.
# See the pivot comment in the prepare kernel.
FLASH_KDA_CHUNK: int = 32

# Width of the diagonal blocks the WY inverse is built from. The Neumann powers
# used to invert them grow with the block width, so this bounds how far the
# intermediates can run ahead of the O(1) result; 16 is also what the fla and
# CUTLASS kernels use.
FLASH_KDA_INV_BLOCK: int = 16

_K1_FALLBACK_CONFIG = triton.Config({}, num_warps=2, num_stages=1)


@triton.heuristics(
    {
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
        "HAS_BIAS": lambda args: args["dt_bias"] is not None,
    }
)
@triton.autotune(
    configs=chunk_delta_attn_autotune_configs(
        [
            triton.Config({}, num_warps=nw, num_stages=ns)
            for nw in [2, 4]
            for ns in [1, 2]
        ],
        default_config=chunk_delta_attn_tuned_config(
            "flash_kda_prepare_kernel", _K1_FALLBACK_CONFIG
        ),
    ),
    key=["H", "K", "C"],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=["T"])
def _flash_kda_prepare_kernel(
    q,
    k,
    g_raw,
    beta_raw,
    A_log,
    dt_bias,
    ws_kd,
    ws_qd,
    ws_kr,
    ws_gt,
    ws_inv_mqk,
    cu_seqlens,
    chunk_indices,
    scale,
    lower_bound,
    T,
    NT,
    TOTAL_TILES,
    H: tl.constexpr,
    K: tl.constexpr,
    C: tl.constexpr,
    BC: tl.constexpr,
    NUM_DOUBLING: tl.constexpr,
    NUM_MERGE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    CM_QKG: tl.constexpr = "",
    CM_WS: tl.constexpr = "",
):
    """Per-chunk prepare: decayed q/k, gate total, Mqk, and (I - L)^-1."""
    i_t = tl.program_id(0).to(tl.int64)
    i_bh = tl.program_id(1).to(tl.int64)
    i_b, i_h = i_bh // H, i_bh % H

    if IS_VARLEN:
        # program_id(0) is already the global chunk index; chunk_indices maps it
        # to (sequence, chunk-within-sequence).
        i_n = tl.load(chunk_indices + i_t * 2).to(tl.int64)
        i_tl = tl.load(chunk_indices + i_t * 2 + 1).to(tl.int64)
        bos = tl.load(cu_seqlens + i_n).to(tl.int64)
        eos = tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T_seq = eos - bos
        g_tile = i_t
    else:
        i_tl = i_t
        bos = i_b * T
        T_seq = T
        g_tile = i_b * NT + i_t

    t_off = i_tl * C
    if t_off >= T_seq:
        return
    actual_len = tl.minimum(C, T_seq - t_off)

    o_c = tl.arange(0, C)
    o_k = tl.arange(0, K)
    m_c = o_c < actual_len
    m_ck = m_c[:, None]

    base = (bos + t_off) * H + i_h
    qk_off = base * K + o_c[:, None] * (H * K) + o_k[None, :]

    b_q = tl.load(q + qk_off, mask=m_ck, other=0.0, cache_modifier=CM_QKG).to(
        tl.float32
    )
    b_k = tl.load(k + qk_off, mask=m_ck, other=0.0, cache_modifier=CM_QKG).to(
        tl.float32
    )

    # L2 normalize rows, matching l2norm_fwd's eps placement.
    b_q = b_q * (1.0 / tl.sqrt(tl.sum(b_q * b_q, axis=1) + 1e-6))[:, None]
    b_k = b_k * (1.0 / tl.sqrt(tl.sum(b_k * b_k, axis=1) + 1e-6))[:, None]

    # Gate: lower_bound * sigmoid(exp(A_log) * (g + dt_bias)), then chunk-local
    # cumsum into log2 space. Same expression as chunk_gate_cumsum so the two
    # paths agree bit-for-bit on the gate.
    b_g = tl.load(g_raw + qk_off, mask=m_ck, other=0.0, cache_modifier=CM_QKG).to(
        tl.float32
    )
    if HAS_BIAS:
        b_g = b_g + tl.load(dt_bias + i_h * K + o_k).to(tl.float32)[None, :]
    b_A = tl.load(A_log + i_h).to(tl.float32)
    b_gate = lower_bound * tl.sigmoid(exp(b_A) * b_g)
    LOG2_E: tl.constexpr = 1.4426950408889634
    b_gcum = tl.cumsum(b_gate, axis=0) * LOG2_E
    b_gcum = tl.where(m_ck, b_gcum, 0.0)

    # gcum is non-positive and monotonically decreasing down the chunk, so every
    # exponent below is <= 0 and exp2 cannot overflow. gcum itself reaches about
    # -460 at C=64 with lower_bound=-5, which is why nothing here forms
    # exp2(-gcum) unclamped.
    last_row = actual_len - 1
    b_g_last = tl.sum(tl.where(o_c[:, None] == last_row, b_gcum, 0.0), axis=0)
    b_g_total = exp2(b_g_last)

    # Decay measured from the chunk start. These only ever multiply the recurrent
    # state, which enters the chunk at row 0, so a row whose gate has decayed past
    # fp32 range genuinely contributes nothing and flushing it to zero is correct.
    b_exp_g = exp2(b_gcum)

    # Stored here rather than at the end of the kernel: these are three C x K
    # tiles, and holding them live across the inversion below costs more
    # registers than K1 has to spare. At H = 64 that alone decided whether the
    # kernel fit in 256 VGPRs, i.e. whether it ran at one or two waves per SIMD.
    #
    # Workspace is [H, TOTAL_TILES, ...] so one sequence's chunks stay contiguous
    # for K2's sequential walk. Tail rows are zeroed rather than left unwritten:
    # K2 accumulates k_restored^T @ U over all C rows, so garbage there would
    # corrupt the recurrent state, not just the masked output rows.
    ws_idx = i_h * TOTAL_TILES + g_tile
    ck_off = ws_idx * C * K + o_c[:, None] * K + o_k[None, :]
    tl.store(
        ws_kd + ck_off,
        tl.where(m_ck, b_k * b_exp_g, 0.0).to(tl.bfloat16),
        cache_modifier=CM_WS,
    )
    tl.store(
        ws_qd + ck_off,
        tl.where(m_ck, b_q * b_exp_g * scale, 0.0).to(tl.bfloat16),
        cache_modifier=CM_WS,
    )
    tl.store(
        ws_kr + ck_off,
        tl.where(m_ck, b_k * exp2(b_g_last[None, :] - b_gcum), 0.0).to(tl.bfloat16),
        cache_modifier=CM_WS,
    )
    tl.store(ws_gt + ws_idx * K + o_k, b_g_total, cache_modifier=CM_WS)

    p_beta = beta_raw + (bos + t_off) * H + i_h + o_c * H
    b_beta = tl.sigmoid(tl.load(p_beta, mask=m_c, other=0.0).to(tl.float32))

    # The intra-chunk matrices need decay *differences* between two rows of the
    # same chunk. Those are O(1) near the diagonal even when each row's decay from
    # the chunk start is not, so they cannot be factored as
    # exp2(gcum[i]) * exp2(-gcum[j]): with the Kimi gate the cumsum drops about
    # 3.6 log2 units per token, so one factor underflows while the other overflows
    # and the product that should have been O(1) collapses. Re-centering on the
    # chunk midpoint halves the span each factor has to cover, which keeps both
    # inside fp32 for C <= 32. C = 64 spans ~280 log2 units and needs a per
    # sub-chunk pivot instead, as the default intra kernel uses.
    o_mid = tl.minimum(C // 2, actual_len - 1)
    b_gp = tl.sum(tl.where(o_c[:, None] == o_mid, b_gcum, 0.0), axis=0)
    b_gm = b_gcum - b_gp[None, :]
    b_dec = exp2(b_gm)
    b_inc = exp2(-b_gm)
    b_k_piv = tl.where(m_ck, b_k * b_dec, 0.0).to(tl.bfloat16)
    b_q_piv = tl.where(m_ck, b_q * b_dec * scale, 0.0).to(tl.bfloat16)
    b_k_inv = tl.where(m_ck, b_k * b_inc, 0.0).to(tl.bfloat16)

    # The WY form needs (I + tril(diag(beta) K K^T, -1))^-1. L is negated here so
    # the doubling below can be written as the plain Neumann series in L; folding
    # the sign in anywhere later would need alternating signs per term.
    # Getting this wrong is nearly invisible at the Kimi default lower_bound=-5:
    # the decay damps L to almost nothing, so (I + L)^-1 and (I - L)^-1 agree to
    # well inside bf16. It only shows up as the gate weakens, which is what the
    # lower_bound sweep in the tests is guarding.
    o_i = tl.arange(0, C)
    b_L = tl.dot(b_k_piv, tl.trans(b_k_inv))
    b_L = tl.where(o_i[:, None] > o_i[None, :], -b_L * b_beta[:, None], 0.0)

    cc_off = ws_idx * 2 * C * C + o_i[:, None] * C + o_i[None, :]
    b_Mqk = tl.dot(b_q_piv, tl.trans(b_k_inv))
    b_Mqk = tl.where(o_i[:, None] >= o_i[None, :], b_Mqk, 0.0)
    tl.store(ws_inv_mqk + cc_off + C * C, b_Mqk, cache_modifier=CM_WS)

    # Invert the BC-wide diagonal blocks via (I+D)(I+D^2)(I+D^4)...(I+D^(BC/2)),
    # then fold the sub-diagonal blocks back in, doubling the block width each
    # round. D is strictly lower triangular, hence nilpotent, and so is
    # INV @ L_off at every merge level (it maps one sub-block into the one below
    # and no further), so both halves are exact in exact arithmetic.
    # Summing I + D + D^2 + D^4 + ... instead would silently drop D^3, D^5, D^6,
    # and every other non-power-of-two term; that is not the inverse, and on this
    # D it is off by the magnitude of the matrix itself.
    #
    # The block split is not about the answer but about the intermediates. L^k
    # counts paths down the chunk, so across all C = 32 rows its entries reach
    # 5e7 once the keys correlate and the gate stops damping L, while the inverse
    # they sum to stays bounded by 1. Nothing survives that cancellation: the
    # flat doubling comes out 400% wrong even in fp32, 1e5 wrong in bf16, and
    # overflows fp16. Confined to BC = 16 rows the same powers peak at 2e3 and
    # the result lands within 2e-4, at an identical matmul count.
    #
    # input_precision is spelled out because an unannotated fp32 dot is not fp32
    # everywhere: tl.dot resolves it to "tf32" on any target that allows it, and
    # gfx942 then selects v_mfma_f32_32x32x4_xf32, which truncates to 19 bits
    # without rounding. On a doubling this cancellation-prone that is 2e1 off.
    b_D = tl.where(o_i[:, None] // BC == o_i[None, :] // BC, b_L, 0.0)
    b_INV = tl.where(o_i[:, None] == o_i[None, :], 1.0, 0.0) + b_D
    b_Dp = tl.dot(b_D, b_D, input_precision="ieee")
    for _ in tl.static_range(NUM_DOUBLING):
        b_INV = b_INV + tl.dot(b_INV, b_Dp, input_precision="ieee")
        b_Dp = tl.dot(b_Dp, b_Dp, input_precision="ieee")

    # fp32 throughout: the partial sums here are what the split keeps small, and
    # narrowing the matmuls to fp16 puts the ill-conditioned case back at 100%
    # error. This is the parallel kernel, so the width is nearly free; only the
    # fp16 store below is seen by K2's serial walk.
    w = BC
    for _ in tl.static_range(NUM_MERGE):
        m_off = (o_i[:, None] // (2 * w) == o_i[None, :] // (2 * w)) & (
            o_i[:, None] // w != o_i[None, :] // w
        )
        b_off = tl.where(m_off, b_L, 0.0)
        b_INV = b_INV + tl.dot(
            b_INV,
            tl.dot(b_off, b_INV, input_precision="ieee"),
            input_precision="ieee",
        )
        w = 2 * w

    tl.store(ws_inv_mqk + cc_off, b_INV, cache_modifier=CM_WS)


_K2_FALLBACK_CONFIG = triton.Config({"BW": 16}, num_warps=2, num_stages=2)

_K2_CONFIGS: list = (
    [
        triton.Config({"BW": BW}, num_warps=nw, num_stages=ns)
        for BW in [16, 32, 64, 128]
        for nw in [2, 4]
        for ns in [1, 2, 3]
    ]
    if CHUNK_DELTA_ATTN_TRITON_AUTOTUNE
    else chunk_delta_attn_tuned_config_shortlist(
        "flash_kda_segment_kernel", [_K2_FALLBACK_CONFIG]
    )
)


def _seg_occupancy_class(num_segs: int) -> int:
    """Bucket of ``num_segs`` for K2's autotune key.

    The grid is ``cdiv(W, BW) * num_segs * H``, so the best BW moves with the
    segment count: one segment per sequence leaves BW=64 with a quarter of the
    blocks BW=16 gets, and picking 64 there costs 2.3x. Keying on the exact
    count would re-tune per batch, and a power-of-two bucket already separates
    the schedules that differ.
    """
    return max(1, num_segs).bit_length()


@triton.autotune(
    configs=_K2_CONFIGS,
    # HAS_V / COMPUTE_OUTPUT are in the key because the three passes below have
    # very different per-iteration cost and must not share a tuned config, and
    # NUM_SEGS_CLASS is there for the same reason -- without it the segmented
    # and unsegmented schedules collide and whichever runs first sets the
    # config for both, a result the on-disk cache then keeps.
    key=["H", "K", "W", "C", "HAS_V", "COMPUTE_OUTPUT", "NUM_SEGS_CLASS"],
    **autotune_cache_kwargs,
)
@triton.jit
def _flash_kda_segment_kernel(
    ws_kd,
    ws_qd,
    ws_kr,
    ws_gt,
    ws_inv_mqk,
    v_input,
    beta_raw,
    out,
    h_in,
    h_out,
    final_state,
    seg_chunk_base,
    seg_nchunks,
    seg_tok_base,
    seg_tok_end,
    seg_seq,
    seg_is_last,
    TOTAL_TILES,
    NUM_SEGS_CLASS,  # tuning discriminator only; see _seg_occupancy_class
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    W: tl.constexpr,
    C: tl.constexpr,
    BW: tl.constexpr,
    INIT_IDENTITY: tl.constexpr,
    HAS_H_IN: tl.constexpr,
    HAS_V: tl.constexpr,
    COMPUTE_OUTPUT: tl.constexpr,
    STORE_H_OUT: tl.constexpr,
    STORE_FINAL: tl.constexpr,
    STATE_V_FIRST: tl.constexpr,
    CM_OUT: tl.constexpr = "",
):
    """Delta-rule recurrence over one segment of chunks.

    The state update is affine in the incoming state, ``h' = A_seg h + b_seg``,
    so the same kernel serves all three passes of the segmented (context
    parallel) schedule by varying only how it is seeded and what it stores:

      * ``h=0``, real ``v``            -> ``b_seg``   (the affine part)
      * ``h=I``, ``v=0`` (``W = K``)   -> ``A_seg``   (the linear part, one
        column block per program, so the operator is obtained by propagating a
        basis rather than by multiplying out per-chunk operators)
      * ``h=h_in`` from the scan, real ``v``, ``COMPUTE_OUTPUT`` -> the output

    With one segment per sequence this degenerates to a plain sequential scan
    and only the third form runs.
    """
    i_w = tl.program_id(0).to(tl.int64)
    i_sh = tl.program_id(1).to(tl.int64)
    i_seg, i_h = i_sh // H, i_sh % H

    chunk_base = tl.load(seg_chunk_base + i_seg).to(tl.int64)
    n_chunks = tl.load(seg_nchunks + i_seg)
    tok_base = tl.load(seg_tok_base + i_seg).to(tl.int64)
    tok_end = tl.load(seg_tok_end + i_seg).to(tl.int64)

    o_c = tl.arange(0, C)
    o_k1 = tl.arange(0, 64)
    o_k2 = 64 + tl.arange(0, 64)
    o_w = i_w * BW + tl.arange(0, BW)
    m_w = o_w < W

    # The state is [K, W]; K is split into two 64-row halves so each tile is a
    # [64, BW] fp32 register block.
    if INIT_IDENTITY:
        b_h1 = tl.where(o_k1[:, None] == o_w[None, :], 1.0, 0.0)
        b_h2 = tl.where(o_k2[:, None] == o_w[None, :], 1.0, 0.0)
    elif HAS_H_IN:
        s_off = (i_seg * H + i_h) * K * W + o_w[None, :]
        b_h1 = tl.load(h_in + s_off + o_k1[:, None] * W, mask=m_w[None, :], other=0.0)
        b_h2 = tl.load(h_in + s_off + o_k2[:, None] * W, mask=m_w[None, :], other=0.0)
        b_h1 = b_h1.to(tl.float32)
        b_h2 = b_h2.to(tl.float32)
    else:
        b_h1 = tl.zeros([64, BW], dtype=tl.float32)
        b_h2 = tl.zeros([64, BW], dtype=tl.float32)

    for j in range(n_chunks):
        ws_idx = i_h * TOTAL_TILES + chunk_base + j
        ck = ws_idx * C * K
        t0 = tok_base + j * C
        m_c = (t0 + o_c) < tok_end
        m_cw = m_c[:, None] & m_w[None, :]

        b_h1_bf = b_h1.to(tl.bfloat16)
        b_h2_bf = b_h2.to(tl.bfloat16)

        b_kd1 = tl.load(ws_kd + ck + o_c[:, None] * K + o_k1[None, :])
        b_kd2 = tl.load(ws_kd + ck + o_c[:, None] * K + o_k2[None, :])
        b_tmp = tl.dot(b_kd1, b_h1_bf) + tl.dot(b_kd2, b_h2_bf)

        vo_off = t0 * H * V + i_h * V + o_c[:, None] * (H * V) + o_w[None, :]
        if HAS_V:
            b_v = tl.load(v_input + vo_off, mask=m_cw, other=0.0).to(tl.float32)
        else:
            b_v = tl.zeros([C, BW], dtype=tl.float32)

        p_beta = beta_raw + t0 * H + i_h + o_c * H
        b_beta = tl.sigmoid(tl.load(p_beta, mask=m_c, other=0.0).to(tl.float32))

        # Tail rows must stay zero: U feeds the state update, which sums over all
        # C rows regardless of how many are real.
        b_u = tl.where(m_c[:, None], (b_v - b_tmp) * b_beta[:, None], 0.0)

        cc = ws_idx * 2 * C * C + o_c[:, None] * C + o_c[None, :]
        b_inv = tl.load(ws_inv_mqk + cc)
        b_U = tl.dot(b_inv, b_u.to(ws_inv_mqk.dtype.element_ty))

        if COMPUTE_OUTPUT:
            b_mqk = tl.load(ws_inv_mqk + cc + C * C)
            b_qd1 = tl.load(ws_qd + ck + o_c[:, None] * K + o_k1[None, :])
            b_qd2 = tl.load(ws_qd + ck + o_c[:, None] * K + o_k2[None, :])
            b_o = tl.dot(b_qd1, b_h1_bf) + tl.dot(b_qd2, b_h2_bf)
            b_o += tl.dot(b_mqk, b_U.to(b_mqk.dtype))
            tl.store(
                out + vo_off,
                b_o.to(out.dtype.element_ty),
                mask=m_cw,
                cache_modifier=CM_OUT,
            )

        b_gt1 = tl.load(ws_gt + ws_idx * K + o_k1).to(tl.float32)
        b_gt2 = tl.load(ws_gt + ws_idx * K + o_k2).to(tl.float32)

        # The recurrence's own matmuls stay in bf16: widening them costs about
        # 40% of the kernel and moved no error bound measurably, since the state
        # they feed is accumulated in fp32 either way and U already carries
        # whatever the inverse above lost.
        b_U_bf = b_U.to(tl.bfloat16)
        b_kr1_t = tl.load(ws_kr + ck + o_k1[:, None] + o_c[None, :] * K)
        b_kr2_t = tl.load(ws_kr + ck + o_k2[:, None] + o_c[None, :] * K)

        b_h1 = b_h1 * b_gt1[:, None] + tl.dot(b_kr1_t, b_U_bf).to(tl.float32)
        b_h2 = b_h2 * b_gt2[:, None] + tl.dot(b_kr2_t, b_U_bf).to(tl.float32)

    if STORE_H_OUT:
        s_off = (i_seg * H + i_h) * K * W + o_w[None, :]
        e_ty = h_out.dtype.element_ty
        tl.store(h_out + s_off + o_k1[:, None] * W, b_h1.to(e_ty), mask=m_w[None, :])
        tl.store(h_out + s_off + o_k2[:, None] * W, b_h2.to(e_ty), mask=m_w[None, :])

    # Not merged into one condition: the outer test is a constexpr, and when it
    # is false final_state is a null pointer and seg_is_last must not be read.
    if STORE_FINAL:  # noqa: SIM102
        if tl.load(seg_is_last + i_seg) == 1:
            i_n = tl.load(seg_seq + i_seg).to(tl.int64)
            if STATE_V_FIRST:
                f_off = (i_n * H + i_h) * V * K + o_w[None, :] * K
                tl.store(final_state + f_off + o_k1[:, None], b_h1, mask=m_w[None, :])
                tl.store(final_state + f_off + o_k2[:, None], b_h2, mask=m_w[None, :])
            else:
                f_off = (i_n * H + i_h) * K * V + o_w[None, :]
                tl.store(
                    final_state + f_off + o_k1[:, None] * V, b_h1, mask=m_w[None, :]
                )
                tl.store(
                    final_state + f_off + o_k2[:, None] * V, b_h2, mask=m_w[None, :]
                )


@triton.heuristics({"HAS_H0": lambda args: args["h0"] is not None})
@triton.jit
def _flash_kda_seg_scan_kernel(
    A_seg,
    b_seg,
    h_in,
    h0,
    seq_seg_off,
    H: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BV: tl.constexpr,
    HAS_H0: tl.constexpr,
):
    """Serial scan across a sequence's segments: ``h <- A_seg h + b_seg``.

    Only runs once per segment rather than once per chunk, so its serial depth
    is the segment count (single digits) instead of the chunk count.
    """
    i_v = tl.program_id(0).to(tl.int64)
    i_nh = tl.program_id(1).to(tl.int64)
    i_n, i_h = i_nh // H, i_nh % H

    s0 = tl.load(seq_seg_off + i_n).to(tl.int64)
    s1 = tl.load(seq_seg_off + i_n + 1).to(tl.int64)

    o_k1 = tl.arange(0, 64)
    o_k2 = 64 + tl.arange(0, 64)
    o_v = i_v * BV + tl.arange(0, BV)
    m_v = o_v < V

    if HAS_H0:
        base = (i_n * H + i_h) * K * V + o_v[None, :]
        b_h1 = tl.load(h0 + base + o_k1[:, None] * V, mask=m_v[None, :], other=0.0)
        b_h2 = tl.load(h0 + base + o_k2[:, None] * V, mask=m_v[None, :], other=0.0)
    else:
        b_h1 = tl.zeros([64, BV], dtype=tl.float32)
        b_h2 = tl.zeros([64, BV], dtype=tl.float32)

    for s in range(s0, s1):
        hb = (s * H + i_h) * K * V + o_v[None, :]
        tl.store(h_in + hb + o_k1[:, None] * V, b_h1, mask=m_v[None, :])
        tl.store(h_in + hb + o_k2[:, None] * V, b_h2, mask=m_v[None, :])

        # A_seg is stored bf16: the dots below consume it as bf16 anyway, and at
        # this block count the scan is bound by how fast it can pull the
        # operator in, not by arithmetic.
        ab = (s * H + i_h) * K * K
        b_A11 = tl.load(A_seg + ab + o_k1[:, None] * K + o_k1[None, :])
        b_A12 = tl.load(A_seg + ab + o_k1[:, None] * K + o_k2[None, :])
        b_A21 = tl.load(A_seg + ab + o_k2[:, None] * K + o_k1[None, :])
        b_A22 = tl.load(A_seg + ab + o_k2[:, None] * K + o_k2[None, :])

        b_h1_bf = b_h1.to(tl.bfloat16)
        b_h2_bf = b_h2.to(tl.bfloat16)
        b_n1 = tl.dot(b_A11, b_h1_bf) + tl.dot(b_A12, b_h2_bf)
        b_n2 = tl.dot(b_A21, b_h1_bf) + tl.dot(b_A22, b_h2_bf)

        b_h1 = b_n1 + tl.load(
            b_seg + hb + o_k1[:, None] * V, mask=m_v[None, :], other=0.0
        )
        b_h2 = b_n2 + tl.load(
            b_seg + hb + o_k2[:, None] * V, mask=m_v[None, :], other=0.0
        )


def flash_kda_supported(
    q: torch.Tensor,
    v: torch.Tensor,
    chunk_size: int,
    safe_gate: bool,
    use_gate_in_kernel: bool,
    use_qk_l2norm_in_kernel: bool,
    use_beta_sigmoid_in_kernel: bool,
    lower_bound: float | None,
    A_log: torch.Tensor | None,
) -> bool:
    """Whether ``flash_kda_fwd`` can serve this call (see the module docstring)."""
    K = q.shape[-1]
    HV, V = v.shape[2], v.shape[-1]
    return (
        chunk_size == FLASH_KDA_CHUNK
        and safe_gate
        and K == FLASH_KDA_K
        and V == FLASH_KDA_K
        and HV == q.shape[2]
        and q.dtype == torch.bfloat16
        and v.dtype == torch.bfloat16
        and use_gate_in_kernel
        and use_qk_l2norm_in_kernel
        and use_beta_sigmoid_in_kernel
        and lower_bound is not None
        and A_log is not None
    )


def _gluon_k1_usable(C: int, K: int) -> bool:
    return _GLUON_ARCH_OK and C == FLASH_KDA_CHUNK and K == FLASH_KDA_K


def _gluon_k2_usable(C: int, K: int, V: int) -> bool:
    """Whether the Gluon K2 can stand in for the Triton one on this call."""
    return (
        _GLUON_ARCH_OK
        and C == FLASH_KDA_CHUNK
        and K == FLASH_KDA_K
        and V == FLASH_KDA_K
    )


def _log_route(k1_gluon: bool, k2_gluon: bool, C: int, K: int, V: int) -> None:
    key = (k1_gluon, k2_gluon, C, K, V)
    if key in _LOGGED_ROUTES:
        return
    _LOGGED_ROUTES.add(key)
    _LOGGER.info(
        "flash_kda route: K1=%s K2=%s (C=%d K=%d V=%d arch=%s)",
        "gluon" if k1_gluon else "triton",
        "gluon" if k2_gluon else "triton",
        C,
        K,
        V,
        DEVICE_ARCH,
    )


# Launchable anywhere: BW=32 needs two warps, the narrowest the state split
# allows. The zero floor keeps the pair self-consistent -- a device with no
# published table takes the wide branch and finds this same schedule there.
_K2_GLUON_FALLBACK = triton.Config({"BW": 32, "MIN_BLOCKS_PER_CU": 0}, num_warps=2)


@functools.cache
def _k2_gluon_schedule(W: int, num_segs: int, H: int) -> tuple[int, int]:
    """``(BW, num_warps)`` for the Gluon K2, which does not autotune.

    The warps can only split BW -- the state is the B operand of ``dot(kd, h)``
    and its K axis is the contraction axis -- so a count above ``BW // 16``
    duplicates columns instead of splitting them, which bounds the pairs the
    config may publish. Which of the two a shape runs is decided by the grid,
    ``cdiv(W, BW) * num_segs * H``: the wide tile reads the per-chunk workspace
    half as many times, but only while there are still enough blocks to fill the
    device, so it is taken above a published blocks-per-CU floor.
    """
    wide = chunk_delta_attn_tuned_config(
        "k2_ab_fused_gluon_wide", _K2_GLUON_FALLBACK, backend="gluon"
    )
    bw = wide.kwargs["BW"]
    blocks = (W // bw) * num_segs * H
    if W % bw == 0 and blocks >= wide.kwargs["MIN_BLOCKS_PER_CU"] * _num_cus():
        return bw, wide.num_warps
    narrow = chunk_delta_attn_tuned_config(
        "k2_ab_fused_gluon_narrow", _K2_GLUON_FALLBACK, backend="gluon"
    )
    return narrow.kwargs["BW"], narrow.num_warps


@functools.cache
def _num_cus(device_index: int = 0) -> int:
    return torch.cuda.get_device_properties(device_index).multi_processor_count


# Blocks pass A should end up with, in units of the CU count.
_SEG_TARGET_BLOCKS = 3
_SEG_MAX_SEGMENTS = 16
_SEG_MAX_CHUNKS = 32
_SEG_MIN_CHUNKS = 64

_SCAN_BV_NARROW = 16
_SCAN_BV_WIDE = 32


def _scan_bv(n_seqs: int, H: int, V: int) -> tuple[int, int]:
    # Not a fill-the-CUs knob: under 16 always loses; this asks only if 16 overshoots.
    if (V // _SCAN_BV_NARROW) * n_seqs * H > _num_cus():
        return _SCAN_BV_WIDE, 2
    return _SCAN_BV_NARROW, 4


def _choose_chunks_per_seg(n_chunks_max: int, n_seqs: int, H: int, V: int) -> int:
    """Segment length in chunks, or ``n_chunks_max`` to disable segmentation.

    The recurrence kernel gets ``n_segments * H * (V / BW)`` blocks, so with one
    segment per sequence a low head count leaves most of the device idle: at
    H=12, V=128 that is 48 blocks against 256 CUs. Segmenting buys blocks by
    turning one pass into three -- pass A, the cross-segment scan, and pass C --
    so what decides the length is the block count it lands on rather than the
    sequence length on its own.
    """
    override = os.getenv("CHUNK_DELTA_ATTN_FLASH_KDA_SEG", "").strip()
    if override:
        val = int(override)
        return n_chunks_max if val <= 0 else val
    blocks = n_seqs * H * max(1, V // 32)
    cus = _num_cus()
    # There has to be idle capacity to absorb the extra passes.
    if blocks > cus:
        return n_chunks_max
    # And enough depth to amortize them, which no amount of idle capacity makes
    # up for.
    if n_chunks_max < _SEG_MIN_CHUNKS:
        return n_chunks_max
    # Enough segments to fill the device, and enough to keep a segment's own
    # serial walk from becoming the limit on a long sequence. Capped over both,
    # since the two extra passes scale with the segment count.
    segs = min(
        _SEG_MAX_SEGMENTS,
        max(_SEG_TARGET_BLOCKS * cus / blocks, n_chunks_max / _SEG_MAX_CHUNKS),
    )
    # To a power of two, the grid the constants were calibrated on.
    return max(1, min(n_chunks_max, 1 << round(math.log2(n_chunks_max / segs))))


@tensor_cache
def _seq_bounds(cu_seqlens: torch.Tensor) -> tuple[tuple[int, int], ...]:
    """``(bos, eos)`` per sequence.

    Cached because reading it is a device-to-host copy issued after the prepare
    kernel has been queued, so an uncached call stalls the host on that kernel
    on every invocation rather than only on the first.
    """
    bounds = cu_seqlens.tolist()
    return tuple((bounds[i], bounds[i + 1]) for i in range(len(bounds) - 1))


@functools.lru_cache(maxsize=64)
def _build_segments(
    seqs: tuple[tuple[int, int], ...],
    C: int,
    chunks_per_seg: int,
    device: torch.device,
):
    """Descriptors for the segmented recurrence.

    Returns ``(desc, seq_seg_off, num_segs, max_segs_per_seq)`` where ``desc`` is
    a ``[6, num_segs]`` int32 tensor holding, per segment: the global chunk index
    it starts at (indexing the K1 workspace), its chunk count, its first and
    one-past-last token, its sequence, and whether it ends that sequence.

    Cached on the sequence bounds: building these costs a Python loop and a host
    to device copy, which at this kernel's runtime is not noise, and a serving
    loop repeats the same shapes.
    """
    chunk_base, nchunks, tok_base, tok_end, seq_id, is_last = [], [], [], [], [], []
    seq_seg_off = [0]
    g_chunk = 0
    for i, (bos, eos) in enumerate(seqs):
        nch = triton.cdiv(eos - bos, C)
        nseg = max(1, triton.cdiv(nch, chunks_per_seg))
        for s in range(nseg):
            c0 = s * chunks_per_seg
            m = min(chunks_per_seg, nch - c0)
            chunk_base.append(g_chunk + c0)
            nchunks.append(m)
            tok_base.append(bos + c0 * C)
            tok_end.append(min(bos + (c0 + m) * C, eos))
            seq_id.append(i)
            is_last.append(1 if s == nseg - 1 else 0)
        seq_seg_off.append(len(chunk_base))
        g_chunk += nch

    desc = torch.tensor(
        [chunk_base, nchunks, tok_base, tok_end, seq_id, is_last],
        dtype=torch.int32,
        device=device,
    )
    off = torch.tensor(seq_seg_off, dtype=torch.int32, device=device)
    max_per_seq = max(
        seq_seg_off[i + 1] - seq_seg_off[i] for i in range(len(seq_seg_off) - 1)
    )
    return desc, off, len(chunk_base), max_per_seq


_prepare_fast = fast_launch(_flash_kda_prepare_kernel)
_segment_fast = fast_launch(_flash_kda_segment_kernel)
_seg_scan_fast = fast_launch(_flash_kda_seg_scan_kernel)


@input_guard
def flash_kda_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor | None,
    scale: float,
    lower_bound: float,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    state_v_first: bool = False,
    cu_seqlens: torch.Tensor | None = None,
    chunk_indices: torch.Tensor | None = None,
    chunks_per_seg: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """
    Two-kernel KDA forward, with an optional segmented (context parallel) scan.

    Args:
        q, k:               ``[B, T, H, 128]`` bf16.
        v:                  ``[B, T, H, 128]`` bf16.
        g:                  Raw gate ``[B, T, H, 128]``; activation and cumsum
                            happen in K1.
        beta:               Raw beta logits ``[B, T, H]``; sigmoid in-kernel.
        A_log:              ``[H]`` fp32.
        dt_bias:            ``[H * 128]`` fp32 or None.
        scale:              Attention scale, folded into the decayed q.
        lower_bound:        Gate lower bound (Kimi uses -5.0).
        initial_state:      ``[N, H, K, V]`` fp32, or ``[N, H, V, K]`` when
                            ``state_v_first``. None starts from zero.
        output_final_state: Allocate and return the final state.
        state_v_first:      V-first state layout.
        cu_seqlens:         ``[N + 1]`` for varlen; ``B`` must be 1.
        chunk_indices:      Output of ``prepare_chunk_indices``; computed here
                            when omitted.
        chunks_per_seg:     Segment length for the parallel scan. None picks a
                            value from the head count; <= 0 disables it.

    Returns:
        ``(o, final_state)`` with ``o`` shaped like ``v``.
    """
    B, T, H, K = q.shape
    V = v.shape[-1]
    C = FLASH_KDA_CHUNK
    inv_block = min(FLASH_KDA_INV_BLOCK, C)
    dev = q.device

    if cu_seqlens is not None:
        if B != 1:
            raise ValueError(f"Varlen mode requires B=1, got {B}.")
        if chunk_indices is None:
            chunk_indices = prepare_chunk_indices(cu_seqlens, C)
        N = cu_seqlens.numel() - 1
        NT = 0
        total_tiles = chunk_indices.shape[0]
    else:
        N = B
        NT = triton.cdiv(T, C)
        total_tiles = B * NT

    if cu_seqlens is not None:
        seqs = _seq_bounds(cu_seqlens)
    else:
        seqs = tuple((b * T, b * T + T) for b in range(B))

    # Longest sequence in chunks: segmentation is judged per sequence, and
    # passing it as the segment length is what disables segmentation. Resolved
    # here rather than next to _build_segments because K1 launches first and the
    # cache modifiers below depend on the answer.
    n_chunks_max = max(triton.cdiv(eos - bos, C) for bos, eos in seqs)
    if chunks_per_seg is None:
        chunks_per_seg = _choose_chunks_per_seg(n_chunks_max, N, H, V)
    elif chunks_per_seg <= 0:
        chunks_per_seg = n_chunks_max
    segmented = n_chunks_max > chunks_per_seg

    CM_LOAD = ".cg"
    CM_STORE = "" if segmented else ".wt"
    CM_OUT_STORE = "" if segmented else ".cs"

    ws_shape = (H * total_tiles, C, K)
    ws_kd = torch.empty(ws_shape, dtype=torch.bfloat16, device=dev)
    ws_qd = torch.empty(ws_shape, dtype=torch.bfloat16, device=dev)
    ws_kr = torch.empty(ws_shape, dtype=torch.bfloat16, device=dev)
    ws_gt = torch.empty(H * total_tiles, K, dtype=torch.float32, device=dev)
    # fp16 rather than bf16, for the mantissa and not the range: the inverse the
    # prepare kernel writes here is bounded by 1, and K2 walks it through a
    # recurrence as long as the sequence, so the 8 mantissa bits of bf16 are what
    # ends up limiting the fused path. On correlated keys with a weak gate, bf16
    # here scores 1.1e-2 against an fp32 recurrence where fp16 scores 2.3e-3,
    # which is the difference between losing to the default pipeline and beating
    # it by 5x. Both are two bytes, and fp32 would cost 1.3-1.5x, since this is
    # read twice per chunk on K2's serial path. The CUTLASS FlashKDA picks fp16
    # here for the same reason.
    ws_inv_mqk = torch.empty(H * total_tiles, 2 * C, C, dtype=torch.float16, device=dev)

    use_gluon_k1 = AITER_FDA_USE_GLUON_K1 and _gluon_k1_usable(C, K)
    if use_gluon_k1:
        from aiter.ops.triton._gluon_kernels.gfx950.chunk_delta_attn.flash_kda_k1 import (
            gluon_k1_prepare,
        )

        gluon_k1_prepare(
            q=q,
            k=k,
            g_raw=g,
            beta_raw=beta,
            A_log=A_log,
            dt_bias=dt_bias,
            ws_kd=ws_kd,
            ws_qd=ws_qd,
            ws_kr=ws_kr,
            ws_gt=ws_gt,
            ws_inv_mqk=ws_inv_mqk,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            scale=scale,
            lower_bound=lower_bound,
            T=T,
            NT=NT,
            TOTAL_TILES=total_tiles,
            H=H,
            K=K,
            C=C,
            BC=inv_block,
            B=B,
            CM_WS=CM_STORE,
            CM_LOAD=CM_LOAD,
        )
    else:
        _prepare_fast[(total_tiles if cu_seqlens is not None else NT, B * H)](
            q=q,
            k=k,
            g_raw=g,
            beta_raw=beta,
            A_log=A_log,
            dt_bias=dt_bias,
            ws_kd=ws_kd,
            ws_qd=ws_qd,
            ws_kr=ws_kr,
            ws_gt=ws_gt,
            ws_inv_mqk=ws_inv_mqk,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            scale=scale,
            lower_bound=lower_bound,
            T=T,
            NT=NT,
            TOTAL_TILES=total_tiles,
            H=H,
            K=K,
            C=C,
            BC=inv_block,
            NUM_DOUBLING=inv_block.bit_length() - 2,
            NUM_MERGE=(C // inv_block).bit_length() - 1,
            CM_QKG=CM_LOAD,
            CM_WS=CM_STORE,
        )

    desc, seq_seg_off, num_segs, max_segs = _build_segments(
        seqs, C, chunks_per_seg, dev
    )
    seg_chunk_base, seg_nchunks, seg_tok_base, seg_tok_end, seg_seq, seg_is_last = desc

    # The kernel indexes h_in by segment, so normalize a V-first initial state
    # once here instead of carrying the layout through three passes.
    h0 = initial_state
    if h0 is not None:
        if state_v_first:
            h0 = h0.transpose(-1, -2)
        h0 = h0.to(torch.float32).contiguous()

    o = torch.empty_like(v)
    final_state = None
    if output_final_state:
        shape = (N, H, V, K) if state_v_first else (N, H, K, V)
        final_state = torch.empty(shape, dtype=torch.float32, device=dev)

    common = {
        "ws_kd": ws_kd,
        "ws_qd": ws_qd,
        "ws_kr": ws_kr,
        "ws_gt": ws_gt,
        "ws_inv_mqk": ws_inv_mqk,
        "beta_raw": beta,
        "seg_chunk_base": seg_chunk_base,
        "seg_nchunks": seg_nchunks,
        "seg_tok_base": seg_tok_base,
        "seg_tok_end": seg_tok_end,
        "seg_seq": seg_seq,
        "seg_is_last": seg_is_last,
        "TOTAL_TILES": total_tiles,
        "NUM_SEGS_CLASS": _seg_occupancy_class(num_segs),
        "H": H,
        "K": K,
        "V": V,
        "C": C,
        "STATE_V_FIRST": state_v_first,
        "CM_OUT": CM_OUT_STORE,
    }

    # Only pass A can go to Gluon, so an unsegmented shape reaches none of it.
    use_gluon_k2 = AITER_FDA_USE_GLUON_K2 and _gluon_k2_usable(C, K, V) and max_segs > 1
    _log_route(use_gluon_k1, use_gluon_k2, C, K, V)

    def _launch_k2(*, W, **kw):
        return _segment_fast[
            lambda meta, _w=W: (triton.cdiv(_w, meta["BW"]), num_segs * H)
        ](W=W, **common, **kw)

    if max_segs > 1:
        # Pass A: b_seg (affine part) and A_seg (linear part), both fully
        # parallel across segments. Neither writes outputs.
        b_seg = torch.empty(num_segs, H, K, V, dtype=torch.float32, device=dev)
        A_seg = torch.empty(num_segs, H, K, K, dtype=torch.bfloat16, device=dev)
        if use_gluon_k2:
            from aiter.ops.triton._gluon_kernels.gfx950.chunk_delta_attn import (
                flash_kda_k2 as _g2,
            )

            bw, nw = _k2_gluon_schedule(V, num_segs, H)
            _g2.k2_ab_fused_fast[(triton.cdiv(V, bw), num_segs * H)](
                ws_kd=ws_kd,
                ws_kr=ws_kr,
                ws_gt=ws_gt,
                ws_inv_mqk=ws_inv_mqk,
                v_input=v,
                beta_raw=beta,
                h_out_b=b_seg,
                h_out_a=A_seg,
                seg_chunk_base=seg_chunk_base,
                seg_nchunks=seg_nchunks,
                seg_tok_base=seg_tok_base,
                seg_tok_end=seg_tok_end,
                TOTAL_TILES=total_tiles,
                H=H,
                K=K,
                V=V,
                C=C,
                BW=bw,
                **_g2.build_layouts(nw),
                num_warps=nw,
            )
        else:
            for buf, width, identity, has_v in (
                (b_seg, V, False, True),
                (A_seg, K, True, False),
            ):
                _launch_k2(
                    v_input=v,
                    out=None,
                    h_in=None,
                    h_out=buf,
                    final_state=None,
                    W=width,
                    INIT_IDENTITY=identity,
                    HAS_H_IN=False,
                    HAS_V=has_v,
                    COMPUTE_OUTPUT=False,
                    STORE_H_OUT=True,
                    STORE_FINAL=False,
                )

        # Pass B: propagate across segments. Depth is the segment count.
        h_in = torch.empty(num_segs, H, K, V, dtype=torch.float32, device=dev)
        BV_SCAN, SCAN_WARPS = _scan_bv(N, H, V)
        _seg_scan_fast[(triton.cdiv(V, BV_SCAN), N * H)](
            A_seg=A_seg,
            b_seg=b_seg,
            h_in=h_in,
            h0=h0,
            seq_seg_off=seq_seg_off,
            H=H,
            K=K,
            V=V,
            BV=BV_SCAN,
            num_warps=SCAN_WARPS,
        )
    else:
        h_in = h0

    # Pass C: re-run each segment from its true incoming state, writing outputs.
    _launch_k2(
        v_input=v,
        out=o,
        h_in=h_in,
        h_out=None,
        final_state=final_state,
        W=V,
        INIT_IDENTITY=False,
        HAS_H_IN=h_in is not None,
        HAS_V=True,
        COMPUTE_OUTPUT=True,
        STORE_H_OUT=False,
        STORE_FINAL=output_final_state,
    )

    return o, final_state
