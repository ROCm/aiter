# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Gluon kernels of the batch-invariant MXFP4 MoE (gfx950): one FP32 accumulator per
output over K, fixed sums."""

from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.extra import libdevice
from triton.runtime.jit import constexpr_function

from aiter.ops.triton.utils._triton.kernel_repr import make_kernel_repr


@constexpr_function
def _mfma_layout(num_warps):
    # One instruction shape (bf16 16x16x32) for every tile height and batch.
    return gl.amd.AMDMFMALayout(
        version=4,
        instr_shape=[16, 16, 32],
        transposed=True,
        warps_per_cta=[1, num_warps],
    )


@constexpr_function
def _scale_layout_3d(t, bk, bn):
    """[BK/32, 32, BN] layout that reshapes to t ([BK, BN]): a K basis dk goes to (dk //
    32, dk % 32)."""

    def f(b):
        dk, dn = b
        return [dk // 32, 0, dn] if dk >= 32 else [0, dk, dn]

    return gl.DistributedLinearLayout(
        reg_bases=[f(b) for b in t.reg_bases],
        lane_bases=[f(b) for b in t.lane_bases],
        warp_bases=[f(b) for b in t.warp_bases],
        block_bases=[],
        shape=[bk // 32, 32, bn],
    )


@gluon.jit
def _segment_tile(pid, n_seg, SEG_OFFS, TILE_OFFS, BLOCK_M: gl.constexpr):
    """(first row, end row, segment) of flat tile pid; binary search over the tile
    offsets."""
    lo = 0
    hi = n_seg
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if gl.load(TILE_OFFS + mid) <= pid:
            lo = mid
        else:
            hi = mid
    r0 = gl.load(SEG_OFFS + lo) + (pid - gl.load(TILE_OFFS + lo)) * BLOCK_M
    r1 = gl.minimum(gl.load(SEG_OFFS + lo + 1), r0 + BLOCK_M)
    return r0, r1, lo


@gluon.jit
def _upcast(w_s, s_s, DOT_W: gl.constexpr, S_SL: gl.constexpr, DOT_WB: gl.constexpr):
    """bf16 [BK, BN] = fp4 tile * 2^(scale - 127), scale broadcast over 32 K values."""
    w = w_s.load(DOT_W)
    sc = s_s.permute([1, 0]).load(S_SL)
    sc = gl.expand_dims(sc, 1)
    sc = sc.broadcast_to([sc.shape[0], 32, sc.shape[2]])
    sc = sc.reshape([sc.shape[0] * 32, sc.shape[2]])
    sc = gl.convert_layout(sc, DOT_WB, assert_trivial=True)
    return gl.amd.cdna4.scaled_upcast(w, sc, gl.bfloat16, axis=0)


@gluon.jit
def _kloop(
    acc,
    a_base,
    a_off,
    a_mask,
    w_base,
    w_off,
    s_base,
    s_off,
    nk,
    a_s,
    w_s,
    s_s,
    BLOCK_K: gl.constexpr,
    DOT_A: gl.constexpr,
    DOT_W: gl.constexpr,
    S_SL: gl.constexpr,
    DOT_WB: gl.constexpr,
):
    """acc += A tile @ deq(W tile) over nk BLOCK_K steps in K order; the next step's
    loads overlap the MFMAs."""
    a = gl.amd.cdna3.buffer_load(a_base, a_off, mask=a_mask, other=0.0)
    w = gl.amd.cdna3.buffer_load(w_base, w_off)
    sc = gl.amd.cdna3.buffer_load(s_base, s_off)
    for _ in range(nk - 1):
        a_s.store(a)
        w_s.store(w)
        s_s.store(sc)
        a_base += BLOCK_K
        w_base += BLOCK_K // 2
        s_base += BLOCK_K // 32
        a = gl.amd.cdna3.buffer_load(a_base, a_off, mask=a_mask, other=0.0)
        w = gl.amd.cdna3.buffer_load(w_base, w_off)
        sc = gl.amd.cdna3.buffer_load(s_base, s_off)
        acc = gl.amd.cdna4.mfma(
            a_s.load(DOT_A), _upcast(w_s, s_s, DOT_W, S_SL, DOT_WB), acc
        )
    a_s.store(a)
    w_s.store(w)
    s_s.store(sc)
    return gl.amd.cdna4.mfma(
        a_s.load(DOT_A), _upcast(w_s, s_s, DOT_W, S_SL, DOT_WB), acc
    )


@gluon.jit
def _kloop2(
    acc0,
    acc1,
    a_base,
    a_off,
    a_mask,
    w_base,
    w_off,
    w_d,
    s_base,
    s_off,
    s_d,
    nk,
    a_s,
    w_s,
    s_s,
    w1_s,
    s1_s,
    BLOCK_K: gl.constexpr,
    DOT_A: gl.constexpr,
    DOT_W: gl.constexpr,
    S_SL: gl.constexpr,
    DOT_WB: gl.constexpr,
):
    """_kloop for two weight tiles (rows w_d / s_d apart) that share the A tile."""
    a = gl.amd.cdna3.buffer_load(a_base, a_off, mask=a_mask, other=0.0)
    w = gl.amd.cdna3.buffer_load(w_base, w_off)
    sc = gl.amd.cdna3.buffer_load(s_base, s_off)
    w1 = gl.amd.cdna3.buffer_load(w_base + w_d, w_off)
    s1 = gl.amd.cdna3.buffer_load(s_base + s_d, s_off)
    for _ in range(nk - 1):
        a_s.store(a)
        w_s.store(w)
        s_s.store(sc)
        w1_s.store(w1)
        s1_s.store(s1)
        a_base += BLOCK_K
        w_base += BLOCK_K // 2
        s_base += BLOCK_K // 32
        a = gl.amd.cdna3.buffer_load(a_base, a_off, mask=a_mask, other=0.0)
        w = gl.amd.cdna3.buffer_load(w_base, w_off)
        sc = gl.amd.cdna3.buffer_load(s_base, s_off)
        w1 = gl.amd.cdna3.buffer_load(w_base + w_d, w_off)
        s1 = gl.amd.cdna3.buffer_load(s_base + s_d, s_off)
        av = a_s.load(DOT_A)
        acc0 = gl.amd.cdna4.mfma(av, _upcast(w_s, s_s, DOT_W, S_SL, DOT_WB), acc0)
        acc1 = gl.amd.cdna4.mfma(av, _upcast(w1_s, s1_s, DOT_W, S_SL, DOT_WB), acc1)
    a_s.store(a)
    w_s.store(w)
    s_s.store(sc)
    w1_s.store(w1)
    s1_s.store(s1)
    av = a_s.load(DOT_A)
    acc0 = gl.amd.cdna4.mfma(av, _upcast(w_s, s_s, DOT_W, S_SL, DOT_WB), acc0)
    acc1 = gl.amd.cdna4.mfma(av, _upcast(w1_s, s1_s, DOT_W, S_SL, DOT_WB), acc1)
    return acc0, acc1


@constexpr_function
def _load_layouts(block_k, num_warps):
    ka = block_k // 8
    kw = block_k // 32
    return (
        gl.BlockedLayout([1, 8], [64 // ka, ka], [num_warps, 1], [1, 0]),
        gl.BlockedLayout([16, 1], [kw, 64 // kw], [1, num_warps], [0, 1]),
        # 4 scale bytes (32 bits) per thread: the async copy needs 32 or 128 bits.
        (
            gl.BlockedLayout(
                [1, 4],
                [64 // max(block_k // 128, 1), max(block_k // 128, 1)],
                [num_warps, 1],
                [1, 0],
            )
            if block_k >= 128
            else gl.BlockedLayout([1, block_k // 32], [64, 1], [num_warps, 1], [1, 0])
        ),
    )


@gluon.jit
def _tile_offsets(
    r0,
    r1,
    pid_n,
    A_ROWS,
    N,
    stride_am,
    stride_wn,
    stride_sn,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    BLOCK_K: gl.constexpr,
    num_warps: gl.constexpr,
):
    """Offsets of the A rows (gathered with A_ROWS unless None), the W tile [BK/2, BN]
    and the scale tile [BN, BK/32]."""
    LAYOUTS: gl.constexpr = _load_layouts(BLOCK_K, num_warps)
    LA: gl.constexpr = LAYOUTS[0]
    LW: gl.constexpr = LAYOUTS[1]
    LS: gl.constexpr = LAYOUTS[2]
    offs_m = r0 + gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, LA))
    m_mask = offs_m < r1
    if A_ROWS is None:
        rows = offs_m
    else:
        rows = gl.load(A_ROWS + offs_m, mask=m_mask, other=0)
    a_off = (
        rows[:, None] * stride_am
        + gl.arange(0, BLOCK_K, layout=gl.SliceLayout(0, LA))[None, :]
    )
    a_mask = m_mask[:, None] & (a_off >= 0)
    wn = pid_n * BLOCK_N + gl.arange(0, BLOCK_N, layout=gl.SliceLayout(0, LW))
    wn = gl.where(wn < N, wn, 0)
    w_off = (
        wn[None, :] * stride_wn
        + gl.arange(0, BLOCK_K // 2, layout=gl.SliceLayout(1, LW))[:, None]
    )
    sn = pid_n * BLOCK_N + gl.arange(0, BLOCK_N, layout=gl.SliceLayout(1, LS))
    sn = gl.where(sn < N, sn, 0)
    s_off = (
        sn[:, None] * stride_sn
        + gl.arange(0, BLOCK_K // 32, layout=gl.SliceLayout(0, LS))[None, :]
    )
    return a_off, a_mask, w_off, s_off


@constexpr_function
def _shared_layouts():
    return (
        gl.SwizzledSharedLayout(8, 1, 16, order=[1, 0]),
        gl.SwizzledSharedLayout(16, 1, 8, order=[0, 1]),
        gl.SwizzledSharedLayout(1, 1, 1, order=[1, 0]),
    )


@constexpr_function
def _w_smem_layout(bk, swz):
    """Shared layout of the [*, bk/2, bn] fp4 weights; swz: XOR swizzle so one MFMA
    operand read hits 64 banks."""
    if not swz:
        return gl.SwizzledSharedLayout(16, 1, 8, order=[0, 1])
    row = bk // 2
    return gl.SwizzledSharedLayout(
        16, max(256 // row, 1), max(row // 16, 1), order=[0, 1]
    )


_gemm_repr = make_kernel_repr(
    "_moe_mxfp4_bi_gemm_gluon_kernel", ["BLOCK_M", "BLOCK_N", "BLOCK_K"]
)


@gluon.jit(repr=_gemm_repr)
def _moe_mxfp4_bi_gemm_gluon_kernel(
    A,
    A_ROWS,
    W,
    S,
    OUT,
    EXPT_OFFS,
    TILE_OFFS,
    n_experts,
    N,
    K,
    stride_am,
    stride_we,
    stride_wn,
    stride_se,
    stride_sn,
    stride_oc,
    stride_om,
    CH_A,
    CH_W,
    CH_S,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    BLOCK_K: gl.constexpr,
    num_warps: gl.constexpr,
):
    """Plane c (program_id 2) of a grouped MXFP4 GEMM: OUT[c, r] = A[A_ROWS[r], c * CH_A
    : c * CH_A + K] @ deq(W[e])^T."""
    pid_t = gl.program_id(0)
    pid_n = gl.program_id(1)
    c = gl.program_id(2).to(gl.int64)
    if pid_t >= gl.load(TILE_OFFS + n_experts):
        return
    r0, r1, e = _segment_tile(pid_t, n_experts, EXPT_OFFS, TILE_OFFS, BLOCK_M)
    e = e.to(gl.int64)
    MFMA: gl.constexpr = _mfma_layout(num_warps)
    DOT_A: gl.constexpr = gl.DotOperandLayout(operand_index=0, parent=MFMA, k_width=8)
    DOT_W: gl.constexpr = gl.DotOperandLayout(operand_index=1, parent=MFMA, k_width=4)
    DOT_WB: gl.constexpr = gl.DotOperandLayout(operand_index=1, parent=MFMA, k_width=8)
    S_SL: gl.constexpr = gl.SliceLayout(
        1,
        _scale_layout_3d(
            gl.to_linear_layout(DOT_WB, [BLOCK_K, BLOCK_N]), BLOCK_K, BLOCK_N
        ),
    )
    SH: gl.constexpr = _shared_layouts()
    a_s = gl.allocate_shared_memory(gl.bfloat16, [BLOCK_M, BLOCK_K], SH[0])
    w_s = gl.allocate_shared_memory(gl.uint8, [BLOCK_K // 2, BLOCK_N], SH[1])
    s_s = gl.allocate_shared_memory(gl.uint8, [BLOCK_N, BLOCK_K // 32], SH[2])
    a_off, a_mask, w_off, s_off = _tile_offsets(
        r0,
        r1,
        pid_n,
        A_ROWS,
        N,
        stride_am,
        stride_wn,
        stride_sn,
        BLOCK_M,
        BLOCK_N,
        BLOCK_K,
        num_warps,
    )
    acc = gl.zeros((BLOCK_M, BLOCK_N), dtype=gl.float32, layout=MFMA)
    acc = _kloop(
        acc,
        A + c * CH_A,
        a_off,
        a_mask,
        W + e * stride_we + c * CH_W,
        w_off,
        S + e * stride_se + c * CH_S,
        s_off,
        K // BLOCK_K,
        a_s,
        w_s,
        s_s,
        BLOCK_K,
        DOT_A,
        DOT_W,
        S_SL,
        DOT_WB,
    )
    om = r0 + gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, MFMA))
    on = pid_n * BLOCK_N + gl.arange(0, BLOCK_N, layout=gl.SliceLayout(0, MFMA))
    gl.amd.cdna3.buffer_store(
        acc,
        OUT + c * stride_oc,
        om[:, None] * stride_om + on[None, :],
        mask=(om < r1)[:, None] & (on < N)[None, :],
    )


_w2_repr = make_kernel_repr(
    "_moe_mxfp4_bi_w2_tree_gluon_kernel", ["BLOCK_M", "BLOCK_N", "BLOCK_K"]
)


@gluon.jit(repr=_w2_repr)
def _moe_mxfp4_bi_w2_tree_gluon_kernel(
    A,
    W,
    S,
    OUT,
    EXPT_OFFS,
    TILE_OFFS,
    n_experts,
    N,
    KC,
    stride_am,
    stride_we,
    stride_wn,
    stride_se,
    stride_sn,
    stride_om,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    BLOCK_K: gl.constexpr,
    num_warps: gl.constexpr,
):
    """OUT[r] = pair tree of the 8 chunk products A[r, chunk c] @ deq(W[e][:, chunk
    c])^T (one FP32 accumulator each); a bf16 OUT stores bf16(y)."""
    pid_t = gl.program_id(0)
    pid_n = gl.program_id(1)
    if pid_t >= gl.load(TILE_OFFS + n_experts):
        return
    r0, r1, e = _segment_tile(pid_t, n_experts, EXPT_OFFS, TILE_OFFS, BLOCK_M)
    e = e.to(gl.int64)
    MFMA: gl.constexpr = _mfma_layout(num_warps)
    DOT_A: gl.constexpr = gl.DotOperandLayout(operand_index=0, parent=MFMA, k_width=8)
    DOT_W: gl.constexpr = gl.DotOperandLayout(operand_index=1, parent=MFMA, k_width=4)
    DOT_WB: gl.constexpr = gl.DotOperandLayout(operand_index=1, parent=MFMA, k_width=8)
    S_SL: gl.constexpr = gl.SliceLayout(
        1,
        _scale_layout_3d(
            gl.to_linear_layout(DOT_WB, [BLOCK_K, BLOCK_N]), BLOCK_K, BLOCK_N
        ),
    )
    SH: gl.constexpr = _shared_layouts()
    a_s = gl.allocate_shared_memory(gl.bfloat16, [BLOCK_M, BLOCK_K], SH[0])
    w_s = gl.allocate_shared_memory(gl.uint8, [BLOCK_K // 2, BLOCK_N], SH[1])
    s_s = gl.allocate_shared_memory(gl.uint8, [BLOCK_N, BLOCK_K // 32], SH[2])
    a_off, a_mask, w_off, s_off = _tile_offsets(
        r0,
        r1,
        pid_n,
        None,
        N,
        stride_am,
        stride_wn,
        stride_sn,
        BLOCK_M,
        BLOCK_N,
        BLOCK_K,
        num_warps,
    )
    nk = KC // BLOCK_K
    a_base = A
    w_base = W + e * stride_we
    s_base = S + e * stride_se
    for c in gl.static_range(8):
        acc = gl.zeros((BLOCK_M, BLOCK_N), dtype=gl.float32, layout=MFMA)
        acc = _kloop(
            acc,
            a_base + c * KC,
            a_off,
            a_mask,
            w_base + c * (KC // 2),
            w_off,
            s_base + c * (KC // 32),
            s_off,
            nk,
            a_s,
            w_s,
            s_s,
            BLOCK_K,
            DOT_A,
            DOT_W,
            S_SL,
            DOT_WB,
        )
        if c % 2 == 0:
            p = acc
        elif c == 1:
            h0 = p + acc
        elif c == 3:
            h0 = h0 + (p + acc)
        elif c == 5:
            h1 = p + acc
        else:
            y = h0 + (h1 + (p + acc))
    om = r0 + gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, MFMA))
    on = pid_n * BLOCK_N + gl.arange(0, BLOCK_N, layout=gl.SliceLayout(0, MFMA))
    gl.amd.cdna3.buffer_store(
        y.to(OUT.dtype.element_ty),
        OUT,
        om[:, None] * stride_om + on[None, :],
        mask=(om < r1)[:, None] & (on < N)[None, :],
    )


_gu_repr = make_kernel_repr(
    "_moe_mxfp4_bi_gu_situ_gluon_kernel",
    ["HAS_L", "BLOCK_M", "BLOCK_N", "BLOCK_K"],
)


@gluon.jit(repr=_gu_repr)
def _moe_mxfp4_bi_gu_situ_gluon_kernel(
    A,
    A_ROWS,
    W,
    S,
    OUT,
    EXPT_OFFS,
    TILE_OFFS,
    n_experts,
    N,
    K,
    stride_am,
    stride_we,
    stride_wn,
    stride_se,
    stride_sn,
    stride_o,
    INV_B,
    B,
    INV_L,
    L,
    HAS_L: gl.constexpr,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    BLOCK_K: gl.constexpr,
    num_warps: gl.constexpr,
):
    """act = bf16(SiTU(fp32(bf16(gate)), fp32(bf16(up)))) of A[A_ROWS] @ deq(W[e])^T;
    launch without FP fusion."""
    pid_t = gl.program_id(0)
    pid_n = gl.program_id(1)
    if pid_t >= gl.load(TILE_OFFS + n_experts):
        return
    r0, r1, e = _segment_tile(pid_t, n_experts, EXPT_OFFS, TILE_OFFS, BLOCK_M)
    e = e.to(gl.int64)
    MFMA: gl.constexpr = _mfma_layout(num_warps)
    DOT_A: gl.constexpr = gl.DotOperandLayout(operand_index=0, parent=MFMA, k_width=8)
    DOT_W: gl.constexpr = gl.DotOperandLayout(operand_index=1, parent=MFMA, k_width=4)
    DOT_WB: gl.constexpr = gl.DotOperandLayout(operand_index=1, parent=MFMA, k_width=8)
    S_SL: gl.constexpr = gl.SliceLayout(
        1,
        _scale_layout_3d(
            gl.to_linear_layout(DOT_WB, [BLOCK_K, BLOCK_N]), BLOCK_K, BLOCK_N
        ),
    )
    SH: gl.constexpr = _shared_layouts()
    a_s = gl.allocate_shared_memory(gl.bfloat16, [BLOCK_M, BLOCK_K], SH[0])
    w_s = gl.allocate_shared_memory(gl.uint8, [BLOCK_K // 2, BLOCK_N], SH[1])
    s_s = gl.allocate_shared_memory(gl.uint8, [BLOCK_N, BLOCK_K // 32], SH[2])
    w1_s = gl.allocate_shared_memory(gl.uint8, [BLOCK_K // 2, BLOCK_N], SH[1])
    s1_s = gl.allocate_shared_memory(gl.uint8, [BLOCK_N, BLOCK_K // 32], SH[2])
    a_off, a_mask, w_off, s_off = _tile_offsets(
        r0,
        r1,
        pid_n,
        A_ROWS,
        N,
        stride_am,
        stride_wn,
        stride_sn,
        BLOCK_M,
        BLOCK_N,
        BLOCK_K,
        num_warps,
    )
    acc_g = gl.zeros((BLOCK_M, BLOCK_N), dtype=gl.float32, layout=MFMA)
    acc_u = gl.zeros((BLOCK_M, BLOCK_N), dtype=gl.float32, layout=MFMA)
    acc_g, acc_u = _kloop2(
        acc_g,
        acc_u,
        A,
        a_off,
        a_mask,
        W + e * stride_we,
        w_off,
        N * stride_wn,
        S + e * stride_se,
        s_off,
        N * stride_sn,
        K // BLOCK_K,
        a_s,
        w_s,
        s_s,
        w1_s,
        s1_s,
        BLOCK_K,
        DOT_A,
        DOT_W,
        S_SL,
        DOT_WB,
    )
    om = r0 + gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, MFMA))
    on = pid_n * BLOCK_N + gl.arange(0, BLOCK_N, layout=gl.SliceLayout(0, MFMA))
    m = (om < r1)[:, None] & (on < N)[None, :]
    g = acc_g.to(gl.bfloat16).to(gl.float32)
    u = acc_u.to(gl.bfloat16).to(gl.float32)
    sg = gl.div_rn(1.0, 1.0 + libdevice.exp(-g))
    gg = (B * libdevice.tanh(g * INV_B)) * sg
    if HAS_L:
        u = L * libdevice.tanh(u * INV_L)
    gl.amd.cdna3.buffer_store(
        (gg * u).to(gl.bfloat16), OUT, om[:, None] * stride_o + on[None, :], mask=m
    )


@gluon.jit
def _issue(
    a_s,
    w_s,
    s_s,
    i,
    a_base,
    a_off,
    a_mask,
    w_base,
    w_off,
    s_base,
    s_off,
    BLOCK_K: gl.constexpr,
):
    """Async copy of K step i (A, W and scale tiles) into the shared buffers of slot i %
    NUM_BUFFERS."""
    NB: gl.constexpr = a_s.shape[0]
    j = i % NB
    gl.amd.cdna4.async_copy.buffer_load_to_shared(
        a_s.index(j), a_base + i * BLOCK_K, a_off, mask=a_mask, other=0.0
    )
    gl.amd.cdna4.async_copy.buffer_load_to_shared(
        w_s.index(j), w_base + i * (BLOCK_K // 2), w_off
    )
    gl.amd.cdna4.async_copy.buffer_load_to_shared(
        s_s.index(j), s_base + i * (BLOCK_K // 32), s_off
    )
    gl.amd.cdna4.async_copy.commit_group()


@gluon.jit
def _slot_offsets(slot, cols, N, HP, PS, PLANES: gl.constexpr):
    """Offsets of the leaves [slot, cols]: rows of N, or column planes [N // HP, slots,
    HP] PS apart (PLANES)."""
    if PLANES:
        return (
            (cols // HP).to(gl.int64)[None, :] * PS
            + slot[:, None] * HP
            + (cols % HP)[None, :]
        )
    else:
        return slot[:, None] * N + cols[None, :]


_w2a_repr = make_kernel_repr(
    "_moe_mxfp4_bi_w2_tree_async_gluon_kernel",
    [
        "BLOCK_M",
        "BLOCK_N",
        "BLOCK_K",
        "NUM_BUFFERS",
        "N_FIRST",
        "W_SWZ",
        "MMA_K",
        "EP_LEAF",
        "ZB",
        "PLANES",
    ],
)


@gluon.jit(repr=_w2a_repr)
def _moe_mxfp4_bi_w2_tree_async_gluon_kernel(
    A,
    W,
    S,
    OUT,
    EXPT_OFFS,
    TILE_OFFS,
    n_experts,
    N,
    KC,
    stride_am,
    stride_we,
    stride_wn,
    stride_se,
    stride_sn,
    stride_om,
    SLOTS,
    RT,
    ORD,
    TW,
    n_slots,
    max_tiles,
    HP,
    PS,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    BLOCK_K: gl.constexpr,
    NUM_BUFFERS: gl.constexpr,
    num_warps: gl.constexpr,
    N_FIRST: gl.constexpr = False,
    W_SWZ: gl.constexpr = False,
    MMA_K: gl.constexpr = 0,
    EP_LEAF: gl.constexpr = False,
    ZB: gl.constexpr = 16,
    PLANES: gl.constexpr = False,
):
    """The W2 tree kernel with async copies. EP_LEAF: store the leaves w * fp32(bf16(y))
    by slot (PLANES: as column planes), +0 for other ranks' slots."""
    # N_FIRST: grid (N tiles, row tiles); neighbors read rows of one expert.
    pid_t = gl.program_id(1 if N_FIRST else 0)
    pid_n = gl.program_id(0 if N_FIRST else 1)
    if EP_LEAF and pid_t >= max_tiles:
        LZ: gl.constexpr = gl.BlockedLayout([1, 4], [1, 64], [num_warps, 1], [1, 0])
        zs = (pid_t - max_tiles) * ZB + gl.arange(0, ZB, layout=gl.SliceLayout(1, LZ))
        zr = gl.load(RT + zs, mask=zs < n_slots, other=0)
        zh = pid_n * BLOCK_N + gl.arange(0, BLOCK_N, layout=gl.SliceLayout(0, LZ))
        zm = ((zs < n_slots) & (zr < 0))[:, None] & (zh < N)[None, :]
        gl.store(
            SLOTS + _slot_offsets(zs.to(gl.int64), zh, N, HP, PS, PLANES),
            gl.zeros([ZB, BLOCK_N], gl.float32, LZ),
            mask=zm,
        )
        return
    if pid_t >= gl.load(TILE_OFFS + n_experts):
        return
    r0, r1, e = _segment_tile(pid_t, n_experts, EXPT_OFFS, TILE_OFFS, BLOCK_M)
    e = e.to(gl.int64)
    MFMA: gl.constexpr = _mfma_layout(num_warps)
    DOT_A: gl.constexpr = gl.DotOperandLayout(operand_index=0, parent=MFMA, k_width=8)
    DOT_W: gl.constexpr = gl.DotOperandLayout(operand_index=1, parent=MFMA, k_width=4)
    DOT_WB: gl.constexpr = gl.DotOperandLayout(operand_index=1, parent=MFMA, k_width=8)
    MK: gl.constexpr = MMA_K if MMA_K > 0 else BLOCK_K
    SUB: gl.constexpr = BLOCK_K // MK
    S_SL: gl.constexpr = gl.SliceLayout(
        1,
        _scale_layout_3d(gl.to_linear_layout(DOT_WB, [MK, BLOCK_N]), MK, BLOCK_N),
    )
    SH: gl.constexpr = _shared_layouts()
    SH_W: gl.constexpr = _w_smem_layout(BLOCK_K, W_SWZ)
    a_s = gl.allocate_shared_memory(gl.bfloat16, [NUM_BUFFERS, BLOCK_M, BLOCK_K], SH[0])
    w_s = gl.allocate_shared_memory(
        gl.uint8, [NUM_BUFFERS, BLOCK_K // 2, BLOCK_N], SH_W
    )
    s_s = gl.allocate_shared_memory(
        gl.uint8, [NUM_BUFFERS, BLOCK_N, BLOCK_K // 32], SH[2]
    )
    a_off, a_mask, w_off, s_off = _tile_offsets(
        r0,
        r1,
        pid_n,
        None,
        N,
        stride_am,
        stride_wn,
        stride_sn,
        BLOCK_M,
        BLOCK_N,
        BLOCK_K,
        num_warps,
    )
    w_base = W + e * stride_we
    s_base = S + e * stride_se
    nkc = KC // MK
    nk = 8 * KC // BLOCK_K
    for i in gl.static_range(NUM_BUFFERS - 1):
        _issue(
            a_s, w_s, s_s, i, A, a_off, a_mask, w_base, w_off, s_base, s_off, BLOCK_K
        )
    zero = gl.zeros((BLOCK_M, BLOCK_N), dtype=gl.float32, layout=MFMA)
    acc = zero
    s0 = zero
    s1 = zero
    s2 = zero
    y = zero
    for i in range(nk):
        if i + NUM_BUFFERS - 1 < nk:
            _issue(
                a_s,
                w_s,
                s_s,
                i + NUM_BUFFERS - 1,
                A,
                a_off,
                a_mask,
                w_base,
                w_off,
                s_base,
                s_off,
                BLOCK_K,
            )
            gl.amd.cdna4.async_copy.wait_group(NUM_BUFFERS - 1)
        else:
            gl.amd.cdna4.async_copy.wait_group(0)
        j = i % NUM_BUFFERS
        for sub in gl.static_range(SUB):
            acc = gl.amd.cdna4.mfma(
                a_s.index(j).slice(sub * MK, MK, 1).load(DOT_A),
                _upcast(
                    w_s.index(j).slice(sub * (MK // 2), MK // 2, 0),
                    s_s.index(j).slice(sub * (MK // 32), MK // 32, 1),
                    DOT_W,
                    S_SL,
                    DOT_WB,
                ),
                acc,
            )
            q = i * SUB + sub
            if q % nkc == nkc - 1:
                # chunk c done: fold into the fixed pair tree of P0..P7
                c = q // nkc
                if c % 2 == 0:
                    s0 = acc
                else:
                    t = s0 + acc
                    if c == 3:
                        s2 = s1 + t
                    elif c == 7:
                        y = s2 + (s1 + t)
                    else:
                        s1 = t
                acc = zero
    om = r0 + gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, MFMA))
    on = pid_n * BLOCK_N + gl.arange(0, BLOCK_N, layout=gl.SliceLayout(0, MFMA))
    om_mask = om < r1
    if EP_LEAF:
        slot = gl.load(ORD + om, mask=om_mask, other=0)
        w = gl.load(TW + slot, mask=om_mask, other=0.0)
        yf = y.to(gl.bfloat16).to(gl.float32)
        gl.store(
            SLOTS + _slot_offsets(slot.to(gl.int64), on, N, HP, PS, PLANES),
            w[:, None] * yf,
            mask=om_mask[:, None] & (on < N)[None, :],
        )
    else:
        gl.amd.cdna3.buffer_store(
            y.to(OUT.dtype.element_ty),
            OUT,
            om[:, None] * stride_om + on[None, :],
            mask=om_mask[:, None] & (on < N)[None, :],
        )


@gluon.jit
def _issue2(
    a_s,
    w_s,
    s_s,
    w1_s,
    s1_s,
    i,
    a_base,
    a_off,
    a_mask,
    w_base,
    w_off,
    w_d,
    s_base,
    s_off,
    s_d,
    BLOCK_K: gl.constexpr,
):
    """_issue with a second weight tile (w_d / s_d rows apart) on the same A tile."""
    NB: gl.constexpr = a_s.shape[0]
    j = i % NB
    gl.amd.cdna4.async_copy.buffer_load_to_shared(
        a_s.index(j), a_base + i * BLOCK_K, a_off, mask=a_mask, other=0.0
    )
    wb = w_base + i * (BLOCK_K // 2)
    sb = s_base + i * (BLOCK_K // 32)
    gl.amd.cdna4.async_copy.buffer_load_to_shared(w_s.index(j), wb, w_off)
    gl.amd.cdna4.async_copy.buffer_load_to_shared(s_s.index(j), sb, s_off)
    gl.amd.cdna4.async_copy.buffer_load_to_shared(w1_s.index(j), wb + w_d, w_off)
    gl.amd.cdna4.async_copy.buffer_load_to_shared(s1_s.index(j), sb + s_d, s_off)
    gl.amd.cdna4.async_copy.commit_group()


_gua_repr = make_kernel_repr(
    "_moe_mxfp4_bi_gu_situ_async_gluon_kernel",
    [
        "HAS_L",
        "BLOCK_M",
        "BLOCK_N",
        "BLOCK_K",
        "NUM_BUFFERS",
        "N_FIRST",
        "W_SWZ",
    ],
)


@gluon.jit(repr=_gua_repr)
def _moe_mxfp4_bi_gu_situ_async_gluon_kernel(
    A,
    A_ROWS,
    W,
    S,
    OUT,
    EXPT_OFFS,
    TILE_OFFS,
    n_experts,
    N,
    K,
    stride_am,
    stride_we,
    stride_wn,
    stride_se,
    stride_sn,
    stride_o,
    INV_B,
    B,
    INV_L,
    L,
    HAS_L: gl.constexpr,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    BLOCK_K: gl.constexpr,
    NUM_BUFFERS: gl.constexpr,
    num_warps: gl.constexpr,
    N_FIRST: gl.constexpr = False,
    W_SWZ: gl.constexpr = False,
):
    """The gate / up + SiTU kernel with async copies."""
    # N_FIRST: grid (N tiles, row tiles); neighbors read rows of one expert.
    pid_t = gl.program_id(1 if N_FIRST else 0)
    pid_n = gl.program_id(0 if N_FIRST else 1)
    if pid_t >= gl.load(TILE_OFFS + n_experts):
        return
    r0, r1, e = _segment_tile(pid_t, n_experts, EXPT_OFFS, TILE_OFFS, BLOCK_M)
    e = e.to(gl.int64)
    MFMA: gl.constexpr = _mfma_layout(num_warps)
    DOT_A: gl.constexpr = gl.DotOperandLayout(operand_index=0, parent=MFMA, k_width=8)
    DOT_W: gl.constexpr = gl.DotOperandLayout(operand_index=1, parent=MFMA, k_width=4)
    DOT_WB: gl.constexpr = gl.DotOperandLayout(operand_index=1, parent=MFMA, k_width=8)
    S_SL: gl.constexpr = gl.SliceLayout(
        1,
        _scale_layout_3d(
            gl.to_linear_layout(DOT_WB, [BLOCK_K, BLOCK_N]), BLOCK_K, BLOCK_N
        ),
    )
    SH: gl.constexpr = _shared_layouts()
    SH_W: gl.constexpr = _w_smem_layout(BLOCK_K, W_SWZ)
    a_s = gl.allocate_shared_memory(gl.bfloat16, [NUM_BUFFERS, BLOCK_M, BLOCK_K], SH[0])
    w_s = gl.allocate_shared_memory(
        gl.uint8, [NUM_BUFFERS, BLOCK_K // 2, BLOCK_N], SH_W
    )
    s_s = gl.allocate_shared_memory(
        gl.uint8, [NUM_BUFFERS, BLOCK_N, BLOCK_K // 32], SH[2]
    )
    w1_s = gl.allocate_shared_memory(
        gl.uint8, [NUM_BUFFERS, BLOCK_K // 2, BLOCK_N], SH_W
    )
    s1_s = gl.allocate_shared_memory(
        gl.uint8, [NUM_BUFFERS, BLOCK_N, BLOCK_K // 32], SH[2]
    )
    a_off, a_mask, w_off, s_off = _tile_offsets(
        r0,
        r1,
        pid_n,
        A_ROWS,
        N,
        stride_am,
        stride_wn,
        stride_sn,
        BLOCK_M,
        BLOCK_N,
        BLOCK_K,
        num_warps,
    )
    w_base = W + e * stride_we
    s_base = S + e * stride_se
    w_d = N * stride_wn
    s_d = N * stride_sn
    nk = K // BLOCK_K
    acc_g = gl.zeros((BLOCK_M, BLOCK_N), dtype=gl.float32, layout=MFMA)
    acc_u = gl.zeros((BLOCK_M, BLOCK_N), dtype=gl.float32, layout=MFMA)
    for i in gl.static_range(NUM_BUFFERS - 1):
        _issue2(
            a_s,
            w_s,
            s_s,
            w1_s,
            s1_s,
            i,
            A,
            a_off,
            a_mask,
            w_base,
            w_off,
            w_d,
            s_base,
            s_off,
            s_d,
            BLOCK_K,
        )
    for i in range(nk):
        if i + NUM_BUFFERS - 1 < nk:
            _issue2(
                a_s,
                w_s,
                s_s,
                w1_s,
                s1_s,
                i + NUM_BUFFERS - 1,
                A,
                a_off,
                a_mask,
                w_base,
                w_off,
                w_d,
                s_base,
                s_off,
                s_d,
                BLOCK_K,
            )
            gl.amd.cdna4.async_copy.wait_group(NUM_BUFFERS - 1)
        else:
            gl.amd.cdna4.async_copy.wait_group(0)
        j = i % NUM_BUFFERS
        av = a_s.index(j).load(DOT_A)
        acc_g = gl.amd.cdna4.mfma(
            av, _upcast(w_s.index(j), s_s.index(j), DOT_W, S_SL, DOT_WB), acc_g
        )
        acc_u = gl.amd.cdna4.mfma(
            av, _upcast(w1_s.index(j), s1_s.index(j), DOT_W, S_SL, DOT_WB), acc_u
        )
    om = r0 + gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, MFMA))
    on = pid_n * BLOCK_N + gl.arange(0, BLOCK_N, layout=gl.SliceLayout(0, MFMA))
    m = (om < r1)[:, None] & (on < N)[None, :]
    g = acc_g.to(gl.bfloat16).to(gl.float32)
    u = acc_u.to(gl.bfloat16).to(gl.float32)
    sg = gl.div_rn(1.0, 1.0 + libdevice.exp(-g))
    gg = (B * libdevice.tanh(g * INV_B)) * sg
    if HAS_L:
        u = L * libdevice.tanh(u * INV_L)
    gl.amd.cdna3.buffer_store(
        (gg * u).to(gl.bfloat16), OUT, om[:, None] * stride_o + on[None, :], mask=m
    )


@gluon.jit
def _route_buckets(IDS, EMAP, s, valid, n_experts, n_ids, HAS_MAP: gl.constexpr):
    """Local expert of each slot, or n_experts for a slot on another rank or an id
    outside [0, n_ids)."""
    g = gl.load(IDS + s, mask=valid, other=-1).to(gl.int64)
    ok = valid & (g >= 0) & (g < n_ids)
    if HAS_MAP:
        g = gl.load(EMAP + g, mask=ok, other=-1).to(gl.int64)
    else:
        g = gl.where(ok, g, -1)
    return gl.where((g >= 0) & (g < n_experts), g, n_experts).to(gl.int32)


@gluon.jit
def _add_i32(a, b):
    return a + b


_route_repr = make_kernel_repr(
    "_moe_mxfp4_bi_route_gluon_kernel", ["TOP_K", "HAS_MAP", "BLOCK_M", "NB", "BS"]
)


@gluon.jit(repr=_route_repr)
def _moe_mxfp4_bi_route_gluon_kernel(
    IDS,
    EMAP,
    TOK,
    ROWS_TOK,
    ORD,
    EXPT_OFFS,
    TILE_OFFS,
    S,
    n_experts,
    n_ids,
    TOP_K: gl.constexpr,
    HAS_MAP: gl.constexpr,
    BLOCK_M: gl.constexpr,
    NB: gl.constexpr,
    BS: gl.constexpr,
    num_warps: gl.constexpr,
):
    """Stable counting sort of the slots by local expert in one program: TOK, ROWS_TOK
    (-1: other rank), ORD and the expert row / tile offsets."""
    L1: gl.constexpr = gl.BlockedLayout([1], [64], [num_warps], [0])
    eb = gl.arange(0, NB, layout=L1)
    cnt = gl.zeros([NB], gl.int32, layout=L1)
    for c0 in range(0, S, BS):
        s = c0 + gl.arange(0, BS, layout=L1)
        valid = s < S
        b = _route_buckets(IDS, EMAP, s, valid, n_experts, n_ids, HAS_MAP)
        cnt += gl.histogram(b, NB, mask=valid, layout=L1)
    first = gl.associative_scan(cnt, 0, _add_i32) - cnt
    tiles = gl.where(eb < n_experts, (cnt + BLOCK_M - 1) // BLOCK_M, 0)
    first_tile = gl.associative_scan(tiles, 0, _add_i32) - tiles
    gl.store(EXPT_OFFS + eb, first, mask=eb <= n_experts)
    gl.store(TILE_OFFS + eb, first_tile, mask=eb <= n_experts)
    _route_scatter_block(
        IDS,
        EMAP,
        first,
        TOK,
        ROWS_TOK,
        ORD,
        0,
        S,
        n_experts,
        n_ids,
        TOP_K,
        HAS_MAP,
        NB,
        BS,
        num_warps,
    )


_route_count_repr = make_kernel_repr(
    "_moe_mxfp4_bi_route_count_gluon_kernel", ["HAS_MAP", "NB", "BS", "BLK"]
)


@gluon.jit(repr=_route_count_repr)
def _moe_mxfp4_bi_route_count_gluon_kernel(
    IDS,
    EMAP,
    CNT,
    S,
    n_experts,
    n_ids,
    HAS_MAP: gl.constexpr,
    NB: gl.constexpr,
    BS: gl.constexpr,
    BLK: gl.constexpr,
    num_warps: gl.constexpr,
):
    """CNT[b, j] = slots of bucket j in slots [b * BLK, (b + 1) * BLK)."""
    L1: gl.constexpr = gl.BlockedLayout([1], [64], [num_warps], [0])
    b = gl.program_id(0)
    cnt = gl.zeros([NB], gl.int32, layout=L1)
    for c0 in range(b * BLK, gl.minimum((b + 1) * BLK, S), BS):
        s = c0 + gl.arange(0, BS, layout=L1)
        valid = s < S
        cnt += gl.histogram(
            _route_buckets(IDS, EMAP, s, valid, n_experts, n_ids, HAS_MAP),
            NB,
            mask=valid,
            layout=L1,
        )
    gl.store(CNT + b * NB + gl.arange(0, NB, layout=L1), cnt)


_route_scan_repr = make_kernel_repr(
    "_moe_mxfp4_bi_route_scan_gluon_kernel", ["BLOCK_M", "NB"]
)


@gluon.jit(repr=_route_scan_repr)
def _moe_mxfp4_bi_route_scan_gluon_kernel(
    CNT,
    BASE,
    EXPT_OFFS,
    TILE_OFFS,
    nblk,
    n_experts,
    BLOCK_M: gl.constexpr,
    NB: gl.constexpr,
    num_warps: gl.constexpr,
):
    """BASE[b, j] = first sorted row of the bucket-j slots of block b; expert row and
    tile offsets."""
    L1: gl.constexpr = gl.BlockedLayout([1], [64], [num_warps], [0])
    eb = gl.arange(0, NB, layout=L1)
    cnt = gl.zeros([NB], gl.int32, layout=L1)
    for b in range(nblk):
        cnt += gl.load(CNT + b * NB + eb)
    run = gl.associative_scan(cnt, 0, _add_i32) - cnt
    tiles = gl.where(eb < n_experts, (cnt + BLOCK_M - 1) // BLOCK_M, 0)
    gl.store(EXPT_OFFS + eb, run, mask=eb <= n_experts)
    gl.store(
        TILE_OFFS + eb,
        gl.associative_scan(tiles, 0, _add_i32) - tiles,
        mask=eb <= n_experts,
    )
    for b in range(nblk):
        gl.store(BASE + b * NB + eb, run)
        run += gl.load(CNT + b * NB + eb)


@gluon.jit
def _route_scatter_block(
    IDS,
    EMAP,
    base0,
    TOK,
    ROWS_TOK,
    ORD,
    s_lo,
    s_hi,
    n_experts,
    n_ids,
    TOP_K: gl.constexpr,
    HAS_MAP: gl.constexpr,
    NB: gl.constexpr,
    BS: gl.constexpr,
    num_warps: gl.constexpr,
):
    """Sorted rows of the slots [s_lo, s_hi) from the running bucket base (kept in
    shared memory)."""
    L1: gl.constexpr = gl.BlockedLayout([1], [64], [num_warps], [0])
    L2: gl.constexpr = gl.BlockedLayout([1, 1], [8, 8], [num_warps, 1], [1, 0])
    base_s = gl.allocate_shared_memory(
        gl.int32, [NB], gl.SwizzledSharedLayout(1, 1, 1, order=[0])
    )
    base_s.store(base0)
    gl.barrier()
    for c0 in range(s_lo, s_hi, BS):
        s = c0 + gl.arange(0, BS, layout=L1)
        valid = s < s_hi
        b = _route_buckets(IDS, EMAP, s, valid, n_experts, n_ids, HAS_MAP)
        b = gl.where(valid, b, NB)
        bi = gl.convert_layout(b, gl.SliceLayout(1, L2))
        bj = gl.convert_layout(b, gl.SliceLayout(0, L2))
        ii = gl.arange(0, BS, layout=gl.SliceLayout(1, L2))
        jj = gl.arange(0, BS, layout=gl.SliceLayout(0, L2))
        same = (bi[:, None] == bj[None, :]) & (jj[None, :] < ii[:, None])
        rank = gl.convert_layout(gl.sum(same.to(gl.int32), axis=1), L1)
        pos = base_s.gather(gl.minimum(b, NB - 1), 0) + rank
        gl.store(TOK + pos, s // TOP_K, mask=valid)
        gl.store(ORD + pos, s, mask=valid)
        gl.store(ROWS_TOK + s, gl.where(b < n_experts, pos, -1), mask=valid)
        h = gl.histogram(b, NB, mask=valid, layout=L1)
        gl.barrier()
        base_s.store(base_s.load(L1) + h)
        gl.barrier()


_route_scatter_repr = make_kernel_repr(
    "_moe_mxfp4_bi_route_scatter_gluon_kernel", ["TOP_K", "HAS_MAP", "NB", "BS", "BLK"]
)


@gluon.jit(repr=_route_scatter_repr)
def _moe_mxfp4_bi_route_scatter_gluon_kernel(
    IDS,
    EMAP,
    BASE,
    TOK,
    ROWS_TOK,
    ORD,
    S,
    n_experts,
    n_ids,
    TOP_K: gl.constexpr,
    HAS_MAP: gl.constexpr,
    NB: gl.constexpr,
    BS: gl.constexpr,
    BLK: gl.constexpr,
    num_warps: gl.constexpr,
):
    """Stable scatter of the slots of block b (BASE row b from the scan)."""
    b = gl.program_id(0)
    L1: gl.constexpr = gl.BlockedLayout([1], [64], [num_warps], [0])
    base0 = gl.load(BASE + b * NB + gl.arange(0, NB, layout=L1))
    _route_scatter_block(
        IDS,
        EMAP,
        base0,
        TOK,
        ROWS_TOK,
        ORD,
        b * BLK,
        gl.minimum((b + 1) * BLK, S),
        n_experts,
        n_ids,
        TOP_K,
        HAS_MAP,
        NB,
        BS,
        num_warps,
    )


@gluon.jit
def _pair_sum(x):
    """[2k, BH] -> [k, BH]: row i is x[2i] + x[2i + 1]."""
    x = x.reshape([x.shape[0] // 2, 2, x.shape[1]]).permute([0, 2, 1])
    a, b = x.split()
    return a + b


_tree8_repr = make_kernel_repr("_moe_mxfp4_bi_tree8_rows_gluon_kernel", ["BR", "BN"])


@gluon.jit(repr=_tree8_repr)
def _moe_mxfp4_bi_tree8_rows_gluon_kernel(
    P,
    OUT,
    NROWS,
    N,
    stride_pc,
    stride_pm,
    stride_om,
    BR: gl.constexpr,
    BN: gl.constexpr,
    num_warps: gl.constexpr,
):
    """OUT[r] = ((P0 + P1) + (P2 + P3)) + ((P4 + P5) + (P6 + P7)) of the 8 FP32 chunk
    planes P[c, r] for rows < NROWS, stored in OUT's dtype (bf16 rounds once)."""
    L: gl.constexpr = gl.BlockedLayout([1, 4], [1, 64], [num_warps, 1], [1, 0])
    stride_pc = stride_pc.to(gl.int64)  # 7 * stride_pc can pass 2**31
    nrows = gl.load(NROWS)
    r0 = gl.program_id(0) * BR
    if r0 >= nrows:
        return
    offs_r = r0 + gl.arange(0, BR, layout=gl.SliceLayout(1, L))
    offs_n = gl.program_id(1) * BN + gl.arange(0, BN, layout=gl.SliceLayout(0, L))
    m = (offs_r < nrows)[:, None] & (offs_n < N)[None, :]
    p = P + offs_r.to(gl.int64)[:, None] * stride_pm + offs_n[None, :]
    p0 = gl.load(p, mask=m, other=0.0)
    p1 = gl.load(p + stride_pc, mask=m, other=0.0)
    p2 = gl.load(p + 2 * stride_pc, mask=m, other=0.0)
    p3 = gl.load(p + 3 * stride_pc, mask=m, other=0.0)
    p4 = gl.load(p + 4 * stride_pc, mask=m, other=0.0)
    p5 = gl.load(p + 5 * stride_pc, mask=m, other=0.0)
    p6 = gl.load(p + 6 * stride_pc, mask=m, other=0.0)
    p7 = gl.load(p + 7 * stride_pc, mask=m, other=0.0)
    y = ((p0 + p1) + (p2 + p3)) + ((p4 + p5) + (p6 + p7))
    gl.store(
        OUT + offs_r.to(gl.int64)[:, None] * stride_om + offs_n[None, :],
        y.to(OUT.dtype.element_ty),
        mask=m,
    )


_plane_repr = make_kernel_repr("_moe_mxfp4_bi_tree16_plane_gluon_kernel", ["BH"])


@gluon.jit(repr=_plane_repr)
def _moe_mxfp4_bi_tree16_plane_gluon_kernel(
    SLOTS,
    IDS,
    OUT,
    H,
    HP,
    COL0,
    stride_o,
    BH: gl.constexpr,
    num_warps: gl.constexpr,
):
    """OUT[t, COL0 : COL0 + HP] = expert-id ordered pair tree of the 16 leaves in SLOTS
    [T * 16, HP], bf16, -0 -> +0; +0 in the other columns."""
    t = gl.program_id(0).to(gl.int64)
    h0 = gl.program_id(1) * BH
    LV: gl.constexpr = gl.BlockedLayout([1, 1], [16, 4], [1, num_warps], [1, 0])
    ii = gl.arange(0, 16, layout=gl.SliceLayout(1, LV))
    jj = gl.arange(0, 16, layout=gl.SliceLayout(0, LV))
    id_i = gl.load(IDS + t * 16 + ii)
    id_j = gl.load(IDS + t * 16 + jj)
    before = (id_j[None, :] < id_i[:, None]) | (
        (id_j[None, :] == id_i[:, None]) & (jj[None, :] < ii[:, None])
    )
    rank = gl.sum(before.to(gl.int32), axis=1)
    rank_j = gl.convert_layout(rank, gl.SliceLayout(0, LV))
    slot = gl.sum(gl.where(rank_j[None, :] == ii[:, None], jj[None, :], 0), axis=1)
    LCOL: gl.constexpr = gl.BlockedLayout([16, 1], [1, 64], [1, num_warps], [0, 1])
    slot = gl.convert_layout(slot, gl.SliceLayout(1, LCOL))
    offs_h = h0 + gl.arange(0, BH, layout=gl.SliceLayout(0, LCOL))
    inp = (offs_h >= COL0) & (offs_h < COL0 + HP)
    leaf = gl.load(
        SLOTS + (t * 16 + slot)[:, None] * HP + (offs_h - COL0)[None, :],
        mask=inp[None, :],
        other=0.0,
    )
    s = gl.reshape(_pair_sum(_pair_sum(_pair_sum(_pair_sum(leaf)))), [BH])
    o = s.to(gl.bfloat16)
    o = gl.where(o == 0, 0.0, o).to(gl.bfloat16)
    offs_o = h0 + gl.arange(0, BH, layout=o.type.layout)
    gl.store(OUT + t * stride_o + offs_o, o, mask=offs_o < H)


_combine_repr = make_kernel_repr(
    "_moe_mxfp4_bi_combine_gluon_kernel", ["BH", "SCATTER"]
)


@gluon.jit(repr=_combine_repr)
def _moe_mxfp4_bi_combine_gluon_kernel(
    Y,
    W,
    ROWS,
    IDS,
    OUT,
    H,
    stride_y,
    stride_o,
    BH: gl.constexpr,
    num_warps: gl.constexpr,
    SCATTER: gl.constexpr = False,
):
    """Top-16 combine: leaves w * fp32(Y[row]) in expert-id order, pair tree, bf16,
    -0 -> +0; SCATTER: the FP32 leaves by slot instead."""
    t = gl.program_id(0).to(gl.int64)
    h0 = gl.program_id(1) * BH
    LV: gl.constexpr = gl.BlockedLayout([1, 1], [16, 4], [1, num_warps], [1, 0])
    # rank of each slot in expert-id order, then the slot of each rank
    ii = gl.arange(0, 16, layout=gl.SliceLayout(1, LV))
    jj = gl.arange(0, 16, layout=gl.SliceLayout(0, LV))
    id_i = gl.load(IDS + t * 16 + ii)
    id_j = gl.load(IDS + t * 16 + jj)
    before = (id_j[None, :] < id_i[:, None]) | (
        (id_j[None, :] == id_i[:, None]) & (jj[None, :] < ii[:, None])
    )
    rank = gl.sum(before.to(gl.int32), axis=1)  # rank of slot i
    rank_j = gl.convert_layout(rank, gl.SliceLayout(0, LV))
    slot = gl.sum(
        gl.where(rank_j[None, :] == ii[:, None], jj[None, :], 0), axis=1
    )  # slot of rank i
    # 16 leaves of one output in one thread (tree in registers); lanes along H.
    LCOL: gl.constexpr = gl.BlockedLayout([16, 1], [1, 64], [1, num_warps], [0, 1])
    if SCATTER:
        slot = gl.arange(0, 16, layout=gl.SliceLayout(1, LCOL))
    else:
        slot = gl.convert_layout(slot, gl.SliceLayout(1, LCOL))
    r = gl.load(ROWS + t * 16 + slot)
    w = gl.load(W + t * 16 + slot)
    w = gl.where(r >= 0, w, 0.0)
    rr = gl.maximum(r, 0).to(gl.int64)
    offs_h = h0 + gl.arange(0, BH, layout=gl.SliceLayout(0, LCOL))
    m = (r >= 0)[:, None] & (offs_h < H)[None, :]
    yf = gl.load(Y + rr[:, None] * stride_y + offs_h[None, :], mask=m, other=0.0).to(
        gl.float32
    )
    leaf = w[:, None] * yf
    if SCATTER:
        gl.store(
            OUT + (t * 16 + slot.to(gl.int64))[:, None] * stride_o + offs_h[None, :],
            leaf,
            mask=(offs_h < H)[None, :],
        )
        return
    s = _pair_sum(_pair_sum(_pair_sum(_pair_sum(leaf))))
    s = gl.reshape(s, [BH])
    o = s.to(gl.bfloat16)
    o = gl.where(o == 0, 0.0, o).to(gl.bfloat16)
    offs_o = h0 + gl.arange(0, BH, layout=o.type.layout)
    gl.store(OUT + t * stride_o + offs_o, o, mask=offs_o < H)
