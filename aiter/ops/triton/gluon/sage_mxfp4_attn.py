"""
Gluon Sage Attention MXFP4 forward kernel for CDNA4 (gfx950).

Unified mfma_scaled [32,32,64] is used for both QK (e2m1) and PV (e4m3 with
unit scales). The inner loop relies on the LLVM scheduler to interleave
buffer_load / LDS / MFMA instructions; on CDNA4 there is no Hopper-style
warp specialization, so larger BLOCK_M and num_warps (e.g. 256/8) give the
backend more independent ILP and tend to win at long sequences.

Loop body order (per iteration n):
  1. global buffer_load V[n], K[n+1] -> regs
  2. v_smem.store(V[n])
  3. QK = mfma_scaled(Q, K[n] from smem)
  4. k_smem.store(K[n+1])      # frees K regs before softmax
  5. bias + online softmax + acc *= alpha
  6. cast P to fp8, load V[n] from smem, PV = mfma_scaled(P, V[n])
  7. global buffer_load + smem.store K_scale[n+1]
  8. load K[n+1] / K_scale[n+1] from smem for next iter
"""

import torch
import triton
import triton.language as tl
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.amd import AMDMFMALayout
from triton.experimental.gluon.language.amd.cdna4 import (
    mfma as mfma_cdna4,
    mfma_scaled,
    get_mfma_scale_layout,
)
from triton.experimental.gluon.language._layouts import (
    DotOperandLayout,
)


# ---------------------------------------------------------------------------
# Inner loop for full blocks (no masking)
# ---------------------------------------------------------------------------

@gluon.jit
def _sage_inner(
    acc, l_i, m_i,
    q_dot, q_scale_dot,
    k_base, v_base, ks_base,
    bias_base,
    stride_kn, stride_kk,
    stride_vk, stride_vn,
    stride_ksn, stride_ksk,
    stride_bn,
    block_start, block_end,
    k_smem, ks_smem, v_smem,
    BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr,
    BLOCK_DMODEL_QK: gl.constexpr, BLOCK_DMODEL_V: gl.constexpr,
    ACTUAL_BLOCK_DMODEL_QK: gl.constexpr, ACTUAL_BLOCK_DMODEL_V: gl.constexpr,
    USE_BIAS: gl.constexpr,
    # Layouts
    blocked_layout: gl.constexpr,
    kt_blocked_layout: gl.constexpr,
    ks_blocked_layout: gl.constexpr,
    kt_dot_layout: gl.constexpr,
    k_scale_layout: gl.constexpr,
    p_dot_layout: gl.constexpr,
    v_dot_layout: gl.constexpr,
    qk_mma_layout: gl.constexpr,
    pv_mma_layout: gl.constexpr,
    qk_mma_offs_n_col: gl.constexpr,
    qk_mma_m_layout: gl.constexpr,
    pv_mma_m_layout: gl.constexpr,
    p_scale_layout: gl.constexpr,
    v_scale_layout: gl.constexpr,
):
    SCALE_GROUP: gl.constexpr = 32
    PADDED_HEAD_QK: gl.constexpr = ACTUAL_BLOCK_DMODEL_QK != BLOCK_DMODEL_QK
    PADDED_HEAD_V: gl.constexpr = ACTUAL_BLOCK_DMODEL_V != BLOCK_DMODEL_V

    # Offset layouts for K, K_scale, V
    kt_offs_d_layout: gl.constexpr = gl.SliceLayout(dim=1, parent=kt_blocked_layout)
    kt_offs_n_layout: gl.constexpr = gl.SliceLayout(dim=0, parent=kt_blocked_layout)
    ks_offs_n_layout: gl.constexpr = gl.SliceLayout(dim=1, parent=ks_blocked_layout)
    ks_offs_s_layout: gl.constexpr = gl.SliceLayout(dim=0, parent=ks_blocked_layout)
    v_offs_n_layout: gl.constexpr = gl.SliceLayout(dim=1, parent=blocked_layout)
    v_offs_d_layout: gl.constexpr = gl.SliceLayout(dim=0, parent=blocked_layout)

    kt_offs_d = gl.arange(0, BLOCK_DMODEL_QK // 2, layout=kt_offs_d_layout)
    kt_offs_n = gl.arange(0, BLOCK_N, layout=kt_offs_n_layout)
    ks_offs_n = gl.arange(0, BLOCK_N, layout=ks_offs_n_layout)
    ks_offs_s = gl.arange(0, BLOCK_DMODEL_QK // SCALE_GROUP, layout=ks_offs_s_layout)
    v_offs_n = gl.arange(0, BLOCK_N, layout=v_offs_n_layout)
    v_offs_d = gl.arange(0, BLOCK_DMODEL_V, layout=v_offs_d_layout)

    if PADDED_HEAD_QK:
        kt_mask = kt_offs_d[:, None] < (ACTUAL_BLOCK_DMODEL_QK // 2)

    # Pre-create unit scales for PV mfma_scaled (constant across iterations)
    p_unit_scale = gl.full([BLOCK_M, BLOCK_N // SCALE_GROUP], 127, dtype=gl.uint8, layout=p_scale_layout)
    v_unit_scale = gl.full([BLOCK_DMODEL_V, BLOCK_N // SCALE_GROUP], 127, dtype=gl.uint8, layout=v_scale_layout)

    # ---- Prologue: load K[0] + K_scale[0] -> smem (loop body will load from smem) ----
    start_n_0 = block_start * BLOCK_N
    kt_offsets_0 = kt_offs_d[:, None] * stride_kk + (start_n_0 + kt_offs_n[None, :]) * stride_kn
    if PADDED_HEAD_QK:
        kt_regs_0 = gl.amd.cdna4.buffer_load(ptr=k_base, offsets=kt_offsets_0, mask=kt_mask)
    else:
        kt_regs_0 = gl.amd.cdna4.buffer_load(ptr=k_base, offsets=kt_offsets_0)
    k_smem.store(kt_regs_0)
    ks_offsets_0 = (start_n_0 + ks_offs_n[:, None]) * stride_ksn + ks_offs_s[None, :] * stride_ksk
    ks_regs_0 = gl.amd.cdna4.buffer_load(ptr=ks_base, offsets=ks_offsets_0)
    ks_smem.store(ks_regs_0)
    kt_dot = k_smem.load(kt_dot_layout)
    ks_dot = ks_smem.load(k_scale_layout)

    # ---- Main loop: V load + QK(preloaded K) + softmax + PV + K[n+1] load ----
    for block_n in tl.range(block_start, block_end):
        start_n = block_n * BLOCK_N
        next_start_n = (block_n + 1) * BLOCK_N

        # Load V[n] and K[n+1] global early (latency hiding)
        v_offsets = (start_n + v_offs_n[:, None]) * stride_vk + v_offs_d[None, :] * stride_vn
        if PADDED_HEAD_V:
            v_mask = v_offs_d[None, :] < ACTUAL_BLOCK_DMODEL_V
            v_regs = gl.amd.cdna4.buffer_load(ptr=v_base, offsets=v_offsets, mask=v_mask)
        else:
            v_regs = gl.amd.cdna4.buffer_load(ptr=v_base, offsets=v_offsets)

        kt_offsets_next = kt_offs_d[:, None] * stride_kk + (next_start_n + kt_offs_n[None, :]) * stride_kn
        if PADDED_HEAD_QK:
            kt_regs_next = gl.amd.cdna4.buffer_load(ptr=k_base, offsets=kt_offsets_next, mask=kt_mask)
        else:
            kt_regs_next = gl.amd.cdna4.buffer_load(ptr=k_base, offsets=kt_offsets_next)

        # Stage V to smem early so LDS write overlaps with QK MMA setup
        v_smem.store(v_regs)

        # QK = mfma_scaled (uses kt_dot/ks_dot loaded at end of prev iter / prologue)
        qk = gl.zeros([BLOCK_M, BLOCK_N], dtype=gl.float32, layout=qk_mma_layout)
        qk = mfma_scaled(
            a=q_dot, a_scale=q_scale_dot, a_format="e2m1",
            b=kt_dot, b_scale=ks_dot, b_format="e2m1", acc=qk,
        )

        # Store K[n+1] to smem mid-body (frees K regs before softmax)
        k_smem.store(kt_regs_next)

        # Bias
        if USE_BIAS:
            bias_offs_n = start_n + gl.arange(0, BLOCK_N, layout=qk_mma_offs_n_col)
            bias_vals = gl.load(bias_base + bias_offs_n * stride_bn)
            qk = qk + bias_vals[None, :]

        # Online softmax
        m_ij = gl.max(qk, axis=1)
        m_new = gl.maximum(m_i, m_ij)
        p = gl.exp2(qk - m_new[:, None])
        l_ij = gl.sum(p, axis=1)
        alpha = gl.exp2(m_i - m_new)
        l_i = l_i * alpha + l_ij
        m_i = m_new

        # Scale acc (no convert_layout — unified MMA layout)
        acc = acc * alpha[:, None]

        # P·V via mfma_scaled e4m3 (unified [32,32,64] layout)
        # Cast + convert P first so it overlaps with the LDS read of V
        p_cast = p.to(gl.float8e4nv)
        p_dot_reg = gl.convert_layout(p_cast, p_dot_layout)
        v_dot = v_smem.load(v_dot_layout)
        acc = mfma_scaled(
            a=p_dot_reg, a_scale=p_unit_scale, a_format="e4m3",
            b=v_dot, b_scale=v_unit_scale, b_format="e4m3", acc=acc,
        )

        # KS load late (after PV) and store, then load K/KS from smem for next iter
        ks_offsets_next = (next_start_n + ks_offs_n[:, None]) * stride_ksn + ks_offs_s[None, :] * stride_ksk
        ks_regs_next = gl.amd.cdna4.buffer_load(ptr=ks_base, offsets=ks_offsets_next)
        ks_smem.store(ks_regs_next)

        kt_dot = k_smem.load(kt_dot_layout)
        ks_dot = ks_smem.load(k_scale_layout)

    return acc, l_i, m_i


# ---------------------------------------------------------------------------
# Masked inner loop (boundary / causal)
# ---------------------------------------------------------------------------

@gluon.jit
def _sage_inner_mask(
    acc, l_i, m_i,
    q_dot, q_scale_dot,
    k_base, v_base,
    ks_base,
    bias_base,
    stride_kn, stride_kk,
    stride_vk, stride_vn,
    stride_ksn, stride_ksk,
    stride_bn,
    seqlen_k, seqlen_q,
    block_start, block_end,
    n_extra_tokens,
    k_smem, ks_smem, v_smem,
    IS_CAUSAL: gl.constexpr,
    BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr,
    BLOCK_DMODEL_QK: gl.constexpr, BLOCK_DMODEL_V: gl.constexpr,
    ACTUAL_BLOCK_DMODEL_QK: gl.constexpr, ACTUAL_BLOCK_DMODEL_V: gl.constexpr,
    USE_BIAS: gl.constexpr,
    start_m,
    # Layouts
    kt_blocked_layout: gl.constexpr,
    blocked_layout: gl.constexpr,
    kt_dot_layout: gl.constexpr,
    k_scale_layout: gl.constexpr,
    p_dot_layout: gl.constexpr,
    v_dot_layout: gl.constexpr,
    qk_mma_layout: gl.constexpr,
    pv_mma_layout: gl.constexpr,
    ks_blocked_layout: gl.constexpr,
    qk_mma_offs_n_col: gl.constexpr,
    qk_mma_offs_m_row: gl.constexpr,
    qk_mma_m_layout: gl.constexpr,
    pv_mma_m_layout: gl.constexpr,
    p_scale_layout: gl.constexpr,
    v_scale_layout: gl.constexpr,
):
    SCALE_GROUP: gl.constexpr = 32
    PADDED_HEAD_QK: gl.constexpr = ACTUAL_BLOCK_DMODEL_QK != BLOCK_DMODEL_QK
    PADDED_HEAD_V: gl.constexpr = ACTUAL_BLOCK_DMODEL_V != BLOCK_DMODEL_V

    kt_offs_d_layout: gl.constexpr = gl.SliceLayout(dim=1, parent=kt_blocked_layout)
    kt_offs_n_layout: gl.constexpr = gl.SliceLayout(dim=0, parent=kt_blocked_layout)
    ks_offs_n_layout: gl.constexpr = gl.SliceLayout(dim=1, parent=ks_blocked_layout)
    ks_offs_s_layout: gl.constexpr = gl.SliceLayout(dim=0, parent=ks_blocked_layout)
    v_offs_n_layout: gl.constexpr = gl.SliceLayout(dim=1, parent=blocked_layout)
    v_offs_d_layout: gl.constexpr = gl.SliceLayout(dim=0, parent=blocked_layout)

    kt_offs_d = gl.arange(0, BLOCK_DMODEL_QK // 2, layout=kt_offs_d_layout)
    kt_offs_n = gl.arange(0, BLOCK_N, layout=kt_offs_n_layout)
    ks_offs_n = gl.arange(0, BLOCK_N, layout=ks_offs_n_layout)
    ks_offs_s = gl.arange(0, BLOCK_DMODEL_QK // SCALE_GROUP, layout=ks_offs_s_layout)
    v_offs_n = gl.arange(0, BLOCK_N, layout=v_offs_n_layout)
    v_offs_d = gl.arange(0, BLOCK_DMODEL_V, layout=v_offs_d_layout)

    seqlen_delta_qk = seqlen_k - seqlen_q

    for block_n in range(block_start, block_end):
        start_n = block_n * BLOCK_N

        # --- Load K with boundary mask ---
        kt_offsets = kt_offs_d[:, None] * stride_kk + (start_n + kt_offs_n[None, :]) * stride_kn
        kt_mask = (start_n + kt_offs_n[None, :]) < seqlen_k
        if PADDED_HEAD_QK:
            kt_mask = kt_mask & (kt_offs_d[:, None] < (ACTUAL_BLOCK_DMODEL_QK // 2))
        kt_global = gl.load(k_base + kt_offsets, mask=kt_mask, other=0)
        k_smem.store(kt_global)

        # --- Load K_scale with boundary mask ---
        ks_offsets = (start_n + ks_offs_n[:, None]) * stride_ksn + ks_offs_s[None, :] * stride_ksk
        ks_mask = (start_n + ks_offs_n[:, None]) < seqlen_k
        ks_global = gl.load(ks_base + ks_offsets, mask=ks_mask, other=0)
        ks_smem.store(ks_global)

        # --- Issue V load early ---
        v_offsets = (start_n + v_offs_n[:, None]) * stride_vk + v_offs_d[None, :] * stride_vn
        v_mask = (start_n + v_offs_n[:, None]) < seqlen_k
        if PADDED_HEAD_V:
            v_mask = v_mask & (v_offs_d[None, :] < ACTUAL_BLOCK_DMODEL_V)
        v_regs = gl.amd.cdna4.buffer_load(ptr=v_base, offsets=v_offsets, mask=v_mask)

        kt_dot = k_smem.load(kt_dot_layout)
        ks_dot = ks_smem.load(k_scale_layout)

        # --- QK ---
        qk = gl.zeros([BLOCK_M, BLOCK_N], dtype=gl.float32, layout=qk_mma_layout)
        if (n_extra_tokens != 0) and (start_n + BLOCK_N == block_end * BLOCK_N):
            bound_offs = start_n + gl.arange(0, BLOCK_N, layout=qk_mma_offs_n_col)
            bound_mask = bound_offs[None, :] < seqlen_k
            qk = gl.where(bound_mask, qk,
                          gl.full([BLOCK_M, BLOCK_N], float("-inf"), dtype=gl.float32, layout=qk_mma_layout))

        qk = mfma_scaled(
            a=q_dot, a_scale=q_scale_dot, a_format="e2m1",
            b=kt_dot, b_scale=ks_dot, b_format="e2m1", acc=qk,
        )

        # --- Causal mask ---
        if IS_CAUSAL:
            causal_offs_n = start_n + gl.arange(0, BLOCK_N, layout=qk_mma_offs_n_col)
            causal_offs_m = start_m * BLOCK_M + gl.arange(0, BLOCK_M, layout=qk_mma_offs_m_row)
            causal_mask = causal_offs_m[:, None] >= (causal_offs_n[None, :] - seqlen_delta_qk)
            qk = gl.where(causal_mask, qk,
                          gl.full([BLOCK_M, BLOCK_N], float("-inf"), dtype=gl.float32, layout=qk_mma_layout))

        # --- Bias ---
        if USE_BIAS:
            bias_offs_n = start_n + gl.arange(0, BLOCK_N, layout=qk_mma_offs_n_col)
            bias_mask = bias_offs_n < seqlen_k
            bias_vals = gl.load(bias_base + bias_offs_n * stride_bn, mask=bias_mask, other=0.0)
            qk = qk + bias_vals[None, :]

        # --- Online softmax ---
        m_ij = gl.max(qk, axis=1)
        m_new = gl.maximum(m_i, m_ij)

        q_shifted = gl.where(
            m_new[:, None] == gl.full([BLOCK_M, BLOCK_N], float("-inf"), dtype=gl.float32, layout=qk_mma_layout),
            gl.full([BLOCK_M, BLOCK_N], float("-inf"), dtype=gl.float32, layout=qk_mma_layout),
            qk - m_new[:, None],
        )
        p = gl.exp2(q_shifted)
        l_ij = gl.sum(p, axis=1)

        m_diff = gl.where(
            m_new == gl.full([BLOCK_M], float("-inf"), dtype=gl.float32, layout=qk_mma_m_layout),
            gl.full([BLOCK_M], float("-inf"), dtype=gl.float32, layout=qk_mma_m_layout),
            m_i - m_new,
        )
        alpha = gl.exp2(m_diff)
        l_i = l_i * alpha + l_ij
        m_i = m_new

        alpha_pv = gl.convert_layout(alpha, pv_mma_m_layout)
        acc = acc * alpha_pv[:, None]

        # --- V store + P*V via mfma_scaled e4m3 ---
        v_smem.store(v_regs)
        v_dot = v_smem.load(v_dot_layout)
        p_cast = p.to(v_dot.dtype)
        p_dot_reg = gl.convert_layout(p_cast, p_dot_layout)
        p_unit_scale = gl.full([BLOCK_M, BLOCK_N // SCALE_GROUP], 127, dtype=gl.uint8, layout=p_scale_layout)
        v_unit_scale = gl.full([BLOCK_DMODEL_V, BLOCK_N // SCALE_GROUP], 127, dtype=gl.uint8, layout=v_scale_layout)
        acc = mfma_scaled(
            a=p_dot_reg, a_scale=p_unit_scale, a_format="e4m3",
            b=v_dot, b_scale=v_unit_scale, b_format="e4m3", acc=acc,
        )

    return acc, l_i, m_i


# ---------------------------------------------------------------------------
# Main kernel
# ---------------------------------------------------------------------------

@gluon.jit
def gluon_sage_mxfp4_fwd(
    Q, K, V,
    bias,
    Q_Descale, K_Descale, V_Descale,
    Out,
    stride_qz, stride_qh, stride_qm,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vk, stride_vn,
    stride_oz, stride_oh, stride_om,
    stride_qsz, stride_qsh, stride_qsm, stride_qsk,
    stride_ksz, stride_ksh, stride_ksn, stride_ksk,
    stride_vsz, stride_vsh,
    stride_bz, stride_bh, stride_bm, stride_bn,
    HQ: gl.constexpr, HK: gl.constexpr,
    ACTUAL_BLOCK_DMODEL_QK: gl.constexpr,
    ACTUAL_BLOCK_DMODEL_V: gl.constexpr,
    MAX_SEQLENS_Q: gl.constexpr,
    MAX_SEQLENS_K: gl.constexpr,
    IS_CAUSAL: gl.constexpr,
    BLOCK_M: gl.constexpr,
    BLOCK_DMODEL_QK: gl.constexpr,
    BLOCK_DMODEL_V: gl.constexpr,
    BLOCK_N: gl.constexpr,
    PRE_LOAD_V: gl.constexpr,
    USE_BIAS: gl.constexpr,
    NUM_STAGES: gl.constexpr,
):
    """
    Gluon Sage Attention MXFP4 Forward Kernel (AMD CDNA4 / gfx950).

    Grid: (cdiv(seqlen_q, BLOCK_M), nheads_q, batch)
    """
    SCALE_GROUP: gl.constexpr = 32
    Q_HEAD_DIV: gl.constexpr = 2
    K_HEAD_DIV: gl.constexpr = 2
    num_warps: gl.constexpr = gl.num_warps()
    threads_per_warp: gl.constexpr = 64

    gl.assume(stride_qz >= 0); gl.assume(stride_qh >= 0); gl.assume(stride_qm >= 0)
    gl.assume(stride_kz >= 0); gl.assume(stride_kh >= 0)
    gl.assume(stride_kn >= 0); gl.assume(stride_kk >= 0)
    gl.assume(stride_vz >= 0); gl.assume(stride_vh >= 0)
    gl.assume(stride_vk >= 0); gl.assume(stride_vn >= 0)
    gl.assume(stride_oz >= 0); gl.assume(stride_oh >= 0); gl.assume(stride_om >= 0)

    start_m = gl.program_id(0)
    off_h_q = gl.program_id(1)
    off_z   = gl.program_id(2)
    off_h_k = off_h_q * HK // HQ

    PADDED_HEAD_QK: gl.constexpr = ACTUAL_BLOCK_DMODEL_QK != BLOCK_DMODEL_QK
    PADDED_HEAD_V: gl.constexpr = ACTUAL_BLOCK_DMODEL_V != BLOCK_DMODEL_V

    # ===================== Layout definitions =====================

    # MMA layout for QK: mfma_scaled with e2m1 (K=64)
    qk_mma_layout: gl.constexpr = AMDMFMALayout(
        version=4, instr_shape=[32, 32, 64],
        transposed=True, warps_per_cta=[num_warps, 1],
    )
    # MMA layout for PV: unified with QK using mfma_scaled fp8 (K=64)
    pv_mma_layout: gl.constexpr = qk_mma_layout

    # Q*K dot operand layouts (mxfp4)
    q_dot_layout: gl.constexpr = DotOperandLayout(
        operand_index=0, parent=qk_mma_layout, k_width=16,
    )
    kt_dot_layout: gl.constexpr = DotOperandLayout(
        operand_index=1, parent=qk_mma_layout, k_width=16,
    )
    q_scale_layout: gl.constexpr = get_mfma_scale_layout(
        q_dot_layout, [BLOCK_M, BLOCK_DMODEL_QK // SCALE_GROUP],
    )
    k_scale_layout: gl.constexpr = get_mfma_scale_layout(
        kt_dot_layout, [BLOCK_N, BLOCK_DMODEL_QK // SCALE_GROUP],
    )

    # P*V dot operand layouts (fp8 with unified [32,32,64] layout)
    pv_k_width: gl.constexpr = 16
    p_dot_layout: gl.constexpr = DotOperandLayout(
        operand_index=0, parent=pv_mma_layout, k_width=pv_k_width,
    )
    v_dot_layout: gl.constexpr = DotOperandLayout(
        operand_index=1, parent=pv_mma_layout, k_width=pv_k_width,
    )
    p_scale_layout: gl.constexpr = get_mfma_scale_layout(
        p_dot_layout, [BLOCK_M, BLOCK_N // SCALE_GROUP],
    )
    v_scale_layout: gl.constexpr = get_mfma_scale_layout(
        v_dot_layout, [BLOCK_DMODEL_V, BLOCK_N // SCALE_GROUP],
    )

    # Blocked layouts for global loads
    blocked_layout: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1, 8], threads_per_warp=[threads_per_warp // 4, 4],
        warps_per_cta=[num_warps, 1], order=[1, 0],
    )
    kt_blocked_layout: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1, 1], threads_per_warp=[1, threads_per_warp],
        warps_per_cta=[1, num_warps], order=[0, 1],
    )
    ks_blocked_layout: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[4, 1], threads_per_warp=[8, 8],
        warps_per_cta=[1, num_warps], order=[0, 1],
    )

    # Slice layouts
    offs_m_layout: gl.constexpr = gl.SliceLayout(dim=1, parent=blocked_layout)
    offs_d_layout: gl.constexpr = gl.SliceLayout(dim=0, parent=blocked_layout)
    qk_mma_m_layout: gl.constexpr = gl.SliceLayout(dim=1, parent=qk_mma_layout)
    pv_mma_m_layout: gl.constexpr = gl.SliceLayout(dim=1, parent=pv_mma_layout)
    qk_mma_offs_n_col: gl.constexpr = gl.SliceLayout(dim=0, parent=qk_mma_layout)
    qk_mma_offs_m_row: gl.constexpr = gl.SliceLayout(dim=1, parent=qk_mma_layout)

    # Shared memory layouts
    q_smem_layout: gl.constexpr = gl.SwizzledSharedLayout(
        vec=16, per_phase=2, max_phase=8, order=[1, 0],
    )
    qs_smem_layout: gl.constexpr = gl.SwizzledSharedLayout(
        vec=1, per_phase=1, max_phase=1, order=[1, 0],
    )
    k_smem_layout: gl.constexpr = gl.SwizzledSharedLayout(
        vec=16, per_phase=2, max_phase=8, order=[0, 1],
    )
    ks_smem_layout: gl.constexpr = gl.SwizzledSharedLayout(
        vec=1, per_phase=1, max_phase=1, order=[0, 1],
    )
    v_smem_layout: gl.constexpr = gl.SwizzledSharedLayout(
        vec=8, per_phase=1, max_phase=16, order=[1, 0],
    )

    # ===================== Offsets =====================

    offs_m = start_m * BLOCK_M + gl.arange(0, BLOCK_M, layout=offs_m_layout)
    offs_d_v = gl.arange(0, BLOCK_DMODEL_V, layout=offs_d_layout)

    q_base = Q + off_z * stride_qz + off_h_q * stride_qh
    k_base = K + off_z * stride_kz + off_h_k * stride_kh
    v_base = V + off_z * stride_vz + off_h_k * stride_vh
    qs_base = Q_Descale + off_z * stride_qsz + off_h_q * stride_qsh
    ks_base = K_Descale + off_z * stride_ksz + off_h_k * stride_ksh

    # ===================== Load Q and Q_scale =====================

    q_smem = gl.allocate_shared_memory(
        gl.uint8, [BLOCK_M, BLOCK_DMODEL_QK // Q_HEAD_DIV], layout=q_smem_layout,
    )
    qs_smem = gl.allocate_shared_memory(
        gl.uint8, [BLOCK_M, BLOCK_DMODEL_QK // SCALE_GROUP], layout=qs_smem_layout,
    )

    q_offs_m_layout: gl.constexpr = gl.SliceLayout(dim=1, parent=blocked_layout)
    q_offs_d_layout: gl.constexpr = gl.SliceLayout(dim=0, parent=blocked_layout)
    q_offs_m = start_m * BLOCK_M + gl.arange(0, BLOCK_M, layout=q_offs_m_layout)
    q_offs_d = gl.arange(0, BLOCK_DMODEL_QK // Q_HEAD_DIV, layout=q_offs_d_layout)

    q_ptrs = q_base + q_offs_m[:, None] * stride_qm + q_offs_d[None, :]
    q_mask = q_offs_m[:, None] < MAX_SEQLENS_Q
    q_global = gl.load(q_ptrs, mask=q_mask, other=0)
    q_smem.store(q_global)
    q_dot = q_smem.load(q_dot_layout)

    qs_scale_layout: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[4, 1], threads_per_warp=[8, 8],
        warps_per_cta=[num_warps, 1], order=[1, 0],
    )
    qs_offs_m_layout: gl.constexpr = gl.SliceLayout(dim=1, parent=qs_scale_layout)
    qs_offs_s_layout: gl.constexpr = gl.SliceLayout(dim=0, parent=qs_scale_layout)
    qs_offs_m = start_m * BLOCK_M + gl.arange(0, BLOCK_M, layout=qs_offs_m_layout)
    qs_offs_s = gl.arange(0, BLOCK_DMODEL_QK // SCALE_GROUP, layout=qs_offs_s_layout)

    qs_ptrs = qs_base + qs_offs_m[:, None] * stride_qsm + qs_offs_s[None, :] * stride_qsk
    qs_mask = qs_offs_m[:, None] < MAX_SEQLENS_Q
    qs_global = gl.load(qs_ptrs, mask=qs_mask, other=0)
    qs_smem.store(qs_global)
    q_scale_dot = qs_smem.load(q_scale_layout)

    # ===================== Allocate K/V shared memory =====================

    k_smem = gl.allocate_shared_memory(
        gl.uint8, [BLOCK_DMODEL_QK // K_HEAD_DIV, BLOCK_N], layout=k_smem_layout,
    )
    ks_smem_buf = gl.allocate_shared_memory(
        gl.uint8, [BLOCK_N, BLOCK_DMODEL_QK // SCALE_GROUP], layout=ks_smem_layout,
    )
    v_smem = gl.allocate_shared_memory(
        V.dtype.element_ty, [BLOCK_N, BLOCK_DMODEL_V], layout=v_smem_layout,
    )

    # ===================== Block ranges =====================

    seqlen_q = MAX_SEQLENS_Q
    seqlen_k = MAX_SEQLENS_K

    n_blocks_total = (seqlen_k + BLOCK_N - 1) // BLOCK_N
    n_extra_tokens = seqlen_k % BLOCK_N
    is_modulo_mn: gl.constexpr = (MAX_SEQLENS_Q % BLOCK_M == 0) and (MAX_SEQLENS_K % BLOCK_N == 0)

    if IS_CAUSAL:
        causal_block_limit = (start_m + 1) * BLOCK_M + seqlen_k - seqlen_q
        n_blocks = gl.minimum(n_blocks_total, (causal_block_limit + BLOCK_N - 1) // BLOCK_N)
        masked_blocks: gl.constexpr = BLOCK_M // BLOCK_N + (not is_modulo_mn)
    else:
        n_blocks = n_blocks_total
        masked_blocks: gl.constexpr = 1 if not is_modulo_mn else 0

    masked_blocks_clamped = gl.minimum(masked_blocks, n_blocks)
    n_full_blocks = n_blocks - masked_blocks_clamped

    # ===================== Init accumulators =====================

    m_i = gl.full([BLOCK_M], float("-inf"), dtype=gl.float32, layout=qk_mma_m_layout)
    l_i = gl.full([BLOCK_M], 1.0, dtype=gl.float32, layout=qk_mma_m_layout)
    acc = gl.zeros([BLOCK_M, BLOCK_DMODEL_V], dtype=gl.float32, layout=pv_mma_layout)

    if USE_BIAS:
        bias_base = bias + off_z * stride_bz + off_h_q * stride_bh + start_m * stride_bm
    else:
        bias_base = bias

    # ===================== Full blocks =====================

    if n_full_blocks > 0:
        acc, l_i, m_i = _sage_inner(
            acc, l_i, m_i,
            q_dot, q_scale_dot,
            k_base, v_base, ks_base, bias_base,
            stride_kn, stride_kk,
            stride_vk, stride_vn,
            stride_ksn, stride_ksk,
            stride_bn,
            0, n_full_blocks,
            k_smem, ks_smem_buf, v_smem,
            BLOCK_M, BLOCK_N,
            BLOCK_DMODEL_QK, BLOCK_DMODEL_V,
            ACTUAL_BLOCK_DMODEL_QK, ACTUAL_BLOCK_DMODEL_V,
            USE_BIAS,
            blocked_layout,
            kt_blocked_layout, ks_blocked_layout,
            kt_dot_layout, k_scale_layout,
            p_dot_layout, v_dot_layout,
            qk_mma_layout, pv_mma_layout,
            qk_mma_offs_n_col,
            qk_mma_m_layout, pv_mma_m_layout,
            p_scale_layout, v_scale_layout,
        )

    # ===================== Masked blocks =====================

    if masked_blocks > 0:
        acc, l_i, m_i = _sage_inner_mask(
            acc, l_i, m_i,
            q_dot, q_scale_dot,
            k_base, v_base, ks_base, bias_base,
            stride_kn, stride_kk,
            stride_vk, stride_vn,
            stride_ksn, stride_ksk,
            stride_bn,
            seqlen_k, seqlen_q,
            n_full_blocks, n_blocks,
            n_extra_tokens,
            k_smem, ks_smem_buf, v_smem,
            IS_CAUSAL,
            BLOCK_M, BLOCK_N,
            BLOCK_DMODEL_QK, BLOCK_DMODEL_V,
            ACTUAL_BLOCK_DMODEL_QK, ACTUAL_BLOCK_DMODEL_V,
            USE_BIAS,
            start_m,
            kt_blocked_layout, blocked_layout,
            kt_dot_layout, k_scale_layout,
            p_dot_layout, v_dot_layout,
            qk_mma_layout, pv_mma_layout,
            ks_blocked_layout,
            qk_mma_offs_n_col, qk_mma_offs_m_row,
            qk_mma_m_layout, pv_mma_m_layout,
            p_scale_layout, v_scale_layout,
        )

    # ===================== Epilogue =====================

    l_recip = 1.0 / l_i
    acc_blocked = gl.convert_layout(acc, blocked_layout)

    vd_offs = gl.arange(0, BLOCK_DMODEL_V, layout=offs_d_layout)
    vd_ptr = V_Descale + off_z * stride_vsz + off_h_k * stride_vsh + vd_offs
    if PADDED_HEAD_V:
        v_descale = gl.load(vd_ptr, mask=vd_offs < ACTUAL_BLOCK_DMODEL_V, other=0.0)
    else:
        v_descale = gl.load(vd_ptr)

    l_recip_blocked = gl.convert_layout(l_recip, offs_m_layout)
    acc_blocked = acc_blocked * l_recip_blocked[:, None] * v_descale[None, :]

    o_base = Out + off_z * stride_oz + off_h_q * stride_oh
    o_ptrs = o_base + offs_m[:, None] * stride_om + offs_d_v[None, :]
    o_mask = offs_m[:, None] < MAX_SEQLENS_Q
    if PADDED_HEAD_V:
        o_mask = o_mask & (offs_d_v[None, :] < ACTUAL_BLOCK_DMODEL_V)

    gl.store(o_ptrs, acc_blocked.to(Out.dtype.element_ty), mask=o_mask)
