# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""gfx942 (MI300X) Gluon kernel for the A16W4 (bf16 x MXFP4) MoE GEMM.

CDNA3 has no scaled MFMA, so the Triton kernel's ``tl.dot_scaled`` is decomposed by the
compiler into a per-element fp4 -> fp8 -> f32 -> *scale -> bf16 chain on the VALU; at decode
shapes that chain, not HBM, bounds the kernel (~5.5 VALU ops per weight element).

This kernel dequantises with ``v_perm_b32`` byte lookups instead:

* the fp4 weight tile is loaded straight into the MFMA B-operand register layout (16-byte
  vector loads, no LDS round trip) by permuting K consistently on both operands;
* one inline-asm block turns each packed dword (8 fp4) into four bf16x2 registers: a byte
  LUT of the e2m1 magnitudes (bias C added so the e8m0 scale can be applied with a
  saturating packed 16-bit subtract), the sign ORed into the high byte, and a byte
  interleave into the exact register pairs the MFMA consumes;
* the e8m0 scale is applied as an exponent-field integer subtract (exact: bf16(fp4 * 2^e)
  needs no rounding), 2 elements per op.

Per 8 weight elements: 24 VALU ops, all 32-bit, versus ~44 for the compiler-generated chain.

Scale range: the bias C = 0x3000 keeps the lookup values below the sign bit and makes the
subtract saturate +-0 to -0.0 for every scale 3 <= e8m0 <= 222 (weights 2^-124 .. 2^95).
"""

import triton
import triton.language as tl
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

from aiter.ops.triton._triton_kernels.common.pid_preprocessing import pid_grid
from aiter.ops.triton._triton_kernels.moe.activations import _swiglu
from aiter.ops.triton._triton_kernels.moe.moe_op_gemm_a16w4 import (
    matmul_launch_metadata,
    xcd_swizzle,
)

# ---------------------------------------------------------------------------------------------
# Dequant inline asm
# ---------------------------------------------------------------------------------------------
# One packed dword in ($8) holds fp4 elements e0..e7 (e_{2b} = low nibble of byte b).
# Constant byte tables ($9..$12, VGPRs hoisted out of the loop): bf16 bit patterns T'[m] of
# the fp4 magnitudes m = 0..7 plus the bias C, split into low bytes (tlo0: m 0..3, tlo1: m 4..7)
# and high bytes (thi0, thi1).  T'[0] = 0x8000 so that +-0 end up as -0.0 after the subtract.
# $13 = the e8m0 scale of this (32-K group, n), zero-extended and replicated in both 16-bit halves.
# D = (C + 127<<7) - (e8m0 << 7) is formed inside the asm (packed 16-bit ops) so the compiler cannot
# narrow it: with a Triton-side uint16 expression LLVM emitted (word >> 1) without masking the low byte.
# Outputs: $0 = (e0,e4), $1 = (e1,e5), $4 = (e2,e6), $3 = (e3,e7) as bf16x2 (low, high half);
# $2 is scratch; $5..$7 are SGPR scratch for selectors/constants (VOP3/VOP3P take no literals on gfx9).
# v_perm_b32 D, S0, S1, SEL: SEL bytes 0-3 select bytes of S1, 4-7 select bytes of S0.
_ASM_DEQUANT = tl.constexpr(
    """
s_mov_b32 $5, 0x06020400
s_mov_b32 $6, 0x07030501
s_mov_b32 $7, 0x00070007
v_and_b32 $4, 0x07070707, $8
v_lshrrev_b32 $2, 4, $8
v_and_b32 $2, 0x07070707, $2
v_perm_b32 $0, $10, $9, $4
v_perm_b32 $1, $12, $11, $4
v_perm_b32 $3, $12, $11, $2
v_perm_b32 $2, $10, $9, $2
v_lshlrev_b32 $4, 4, $8
v_and_b32 $4, 0x80808080, $4
v_or_b32 $1, $4, $1
v_and_b32 $4, 0x80808080, $8
v_or_b32 $3, $4, $3
v_perm_b32 $4, $1, $0, $6
v_perm_b32 $0, $1, $0, $5
v_perm_b32 $1, $3, $2, $5
v_perm_b32 $3, $3, $2, $6
v_pk_lshlrev_b16 $2, $7, $13
s_mov_b32 $5, 0x6f806f80
v_pk_sub_u16 $2, $5, $2
v_pk_sub_i16 $0, $0, $2 clamp
v_pk_sub_i16 $1, $1, $2 clamp
v_pk_sub_i16 $3, $3, $2 clamp
v_pk_sub_i16 $4, $4, $2 clamp
"""
)
_ASM_CONSTRAINTS = tl.constexpr("=&v,=&v,=&v,=&v,=&v,=&s,=&s,=&s,v,v,v,v,v,v")

LUT_BIAS = tl.constexpr(0x3000)
# T'[m] = bf16(fp4 magnitude m) + C: 0x8000, 0x6F00, 0x6F80, 0x6FC0, 0x7000, 0x7040, 0x7080, 0x70C0
TLO0 = tl.constexpr(0xC0800000)  # low bytes of T'[0..3]
TLO1 = tl.constexpr(0xC0804000)  # low bytes of T'[4..7]
THI0 = tl.constexpr(0x6F6F6F80)  # high bytes of T'[0..3]
THI1 = tl.constexpr(0x70707070)  # high bytes of T'[4..7]


def gfx942_a16w4_layouts(block_m, block_n, block_k, num_warps):
    """Register layouts of the gfx942 kernel for one launch config.

    Returns (LOAD_X, LOAD_W, LOAD_S, MFMA):
      LOAD_X  (block_m, block_k) bf16 global-load layout, 16 B per lane along K.
      LOAD_W  (block_k//4, block_n) uint16 view of the packed fp4 tile: lane (g = lane//16,
              n = lane%16) holds 2*KG contiguous uint16 (KG = block_k//32 dwords = 32 B for
              block_k=256); the K bits of this layout are ordered so that the tile can be
              relabelled in-register into the MFMA B operand layout (see the kernel).
      LOAD_S  (KG//4, 1, 4, block_n) uint8 scale layout: lane (g, n) holds the KG//4 e8m0 scales
              of the 32-K groups its weight dwords belong to (groups g*KG//4 .. +KG//4-1 of the tile);
              the size-1 dim is broadcast to the 4 dwords per group in the kernel.
      MFMA    the 16x16x16 bf16 MFMA accumulator layout.
    The kernel checks (at compile time) that these match the dot-operand layouts.
    """
    assert block_k % 128 == 0 and block_k <= 512, block_k
    assert block_n % 16 == 0 and block_n >= 16 * num_warps, (block_n, num_warps)
    KG = block_k // 32
    PK2 = block_k // 4
    nw = num_warps.bit_length() - 1
    assert 1 << nw == num_warps
    nt = (block_n // (16 * num_warps)).bit_length() - 1  # N tiles per warp (register bits)
    warp_n = [16 << i for i in range(nw)]
    tile_n = [16 * num_warps << i for i in range(nt)]
    kreg = (2 * KG).bit_length() - 1  # uint16 elements per lane along K = 2*KG
    load_x = gl.BlockedLayout(
        size_per_thread=[1, 8],
        threads_per_warp=[512 // block_k, block_k // 8],
        warps_per_cta=[num_warps, 1],
        order=[1, 0],
    )
    load_w = gl.DistributedLinearLayout(
        reg_bases=[[1 << i, 0] for i in range(kreg)] + [[0, t] for t in tile_n],
        lane_bases=[[0, 1], [0, 2], [0, 4], [0, 8], [2 * KG, 0], [4 * KG, 0]],
        warp_bases=[[0, w] for w in warp_n],
        block_bases=[],
        shape=[PK2, block_n],
    )
    KGL = KG // 4  # scale groups covered by one lane's KG consecutive dwords
    load_s = gl.DistributedLinearLayout(
        reg_bases=[[1 << i, 0, 0, 0] for i in range(KGL.bit_length() - 1)] + [[0, 0, 0, t] for t in tile_n],
        lane_bases=[[0, 0, 0, 1], [0, 0, 0, 2], [0, 0, 0, 4], [0, 0, 0, 8], [0, 0, 1, 0], [0, 0, 2, 0]],
        warp_bases=[[0, 0, 0, w] for w in warp_n],
        block_bases=[],
        shape=[KGL, 1, 4, block_n],
    )
    mfma = gl.amd.AMDMFMALayout(
        version=3,
        instr_shape=[16, 16, 16],
        transposed=True,
        warps_per_cta=[1, num_warps],
    )
    return load_x, load_w, load_s, mfma


@gluon.jit
def _const_u16_pairs(shape: gl.constexpr, VALUE: gl.constexpr, layout: gl.constexpr):
    """uint16 tensor whose (even, odd) element pairs along dim 0 hold the (low, high) halves of
    VALUE, so that a pack=2 inline-asm operand sees the 32-bit constant VALUE."""
    par = gl.arange(0, shape[0], layout=gl.SliceLayout(1, layout)) % 2
    lo = gl.full(shape, VALUE & 0xFFFF, gl.uint16, layout=layout)
    hi = gl.full(shape, VALUE >> 16, gl.uint16, layout=layout)
    return gl.where((par == 0)[:, None], lo, hi)


@gluon.jit
def _relabel_x(x, BLOCK_M: gl.constexpr, BLOCK_K: gl.constexpr):
    """Physical X tile (8 contiguous K per lane) -> semantic K order of the W operand:
    semantic k = 8d + 2s + h  <->  physical element s + 4h of chunk d."""
    x = gl.reshape(x, (BLOCK_M, BLOCK_K // 8, 2, 4))  # (m, d, h, s): physical e = 4h + s
    x = gl.permute(x, (0, 1, 3, 2))  # (m, d, s, h)
    return gl.reshape(x, (BLOCK_M, BLOCK_K))


@gluon.jit
def _relabel_w(wq, KG: gl.constexpr, BLOCK_N: gl.constexpr, DOT_W16: gl.constexpr):
    """Physical uint16 W tile (lane g: 2*KG contiguous uint16 = KG dwords) -> the k_width=2
    B-operand layout: lane g, register kt holds semantic dword d = g + 4*kt."""
    PK2: gl.constexpr = 8 * KG  # uint16 elements per column = BLOCK_K // 4
    w3 = gl.reshape(wq, (4, KG, 2, BLOCK_N))  # (g, kt, h, n): physical q = 2*KG*g + 2*kt + h
    w3 = gl.permute(w3, (1, 0, 2, 3))  # (kt, g, h, n)
    wu = gl.reshape(w3, (PK2, BLOCK_N))  # u = h + 2g + 8kt
    return gl.convert_layout(wu, DOT_W16, assert_trivial=True)


@gluon.jit
def _dequant_tile(
    wu, sc16, tlo0, tlo1, thi0, thi1, BLOCK_K: gl.constexpr, BLOCK_N: gl.constexpr, DOT_W: gl.constexpr
):
    """wu: (BLOCK_K//4, BLOCK_N) uint16 packed fp4 in the k_width=2 operand layout; sc16: the e8m0
    scale of each element's (group, n), as uint16 in the same layout.  Returns the bf16 (BLOCK_K, BLOCK_N) B operand
    with semantic K = 8d + 2s + h <-> fp4 element s + 4h of dword d."""
    o0, o1, _junk, o3, o2, _s1, _s2, _s3 = gl.inline_asm_elementwise(
        _ASM_DEQUANT,
        _ASM_CONSTRAINTS,
        [wu, tlo0, tlo1, thi0, thi1, sc16],
        dtype=(gl.uint16, gl.uint16, gl.uint16, gl.uint16, gl.uint16, gl.uint16, gl.uint16, gl.uint16),
        is_pure=True,
        pack=2,
    )
    # Each o_s register already holds one MFMA operand pair (register bit 0 = half h), so
    # assemble with h as the innermost join: register bits [h, s0, s1, kt...] == DOT_W.
    DN: gl.constexpr = BLOCK_K // 8
    o0 = gl.permute(gl.reshape(o0, (DN, 2, BLOCK_N)), (0, 2, 1))
    o1 = gl.permute(gl.reshape(o1, (DN, 2, BLOCK_N)), (0, 2, 1))
    o2 = gl.permute(gl.reshape(o2, (DN, 2, BLOCK_N)), (0, 2, 1))
    o3 = gl.permute(gl.reshape(o3, (DN, 2, BLOCK_N)), (0, 2, 1))
    lo0, hi0 = gl.split(o0)
    lo1, hi1 = gl.split(o1)
    lo2, hi2 = gl.split(o2)
    lo3, hi3 = gl.split(o3)
    # nested joins: s = 2*outer + inner (verified on hardware with an identity-activation readout;
    # the layout check below cannot see operand order, only the resulting bases)
    j_lo = gl.reshape(gl.permute(gl.join(gl.join(lo0, lo1), gl.join(lo2, lo3)), (0, 1, 3, 2)), (DN, BLOCK_N, 4))
    j_hi = gl.reshape(gl.permute(gl.join(gl.join(hi0, hi1), gl.join(hi2, hi3)), (0, 1, 3, 2)), (DN, BLOCK_N, 4))
    wb = gl.join(j_lo, j_hi)  # (d, n, s, h)
    wb = gl.permute(wb, (0, 2, 3, 1))  # (d, s, h, n)
    wb = gl.reshape(wb, (BLOCK_K, BLOCK_N))
    return gl.convert_layout(wb, DOT_W, assert_trivial=True).to(gl.bfloat16, bitcast=True)


@gluon.jit(launch_metadata=matmul_launch_metadata)
def _moe_gemm_a16w4_gfx942(
    Y,
    stride_y_k,
    stride_y_m,
    stride_y_n,
    X,
    stride_x_m,
    stride_x_k,
    W,
    stride_w_e,
    stride_w_k,
    stride_w_n,
    WMxScale,
    stride_w_mx_e,
    stride_w_mx_k,
    stride_w_mx_n,
    B,
    stride_b_e,  # Bias
    Gammas,
    N,
    K,  # shapes
    # expt data
    GatherIndx,
    ExptHist,
    ExptOffs,
    ExptOffsSum,
    ExptData,
    # expert-parallel: global->local expert id map (-1 = not on this rank)
    ExpertMap,
    # true grid size
    grid_m,
    grid_n,
    # fused activation function
    APPLY_SWIGLU: gl.constexpr,
    alpha,
    limit,
    ACTIVATION_REDUCTION_N: gl.constexpr,
    SWIGLU_ADD_RESIDUAL: gl.constexpr,
    # MoE config
    N_EXPTS_ACT: gl.constexpr,
    HAS_EXPERT_MAP: gl.constexpr,
    # optimization config
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    BLOCK_K: gl.constexpr,
    GROUP_M: gl.constexpr,
    XCD_SWIZZLE: gl.constexpr,
    SPLIT_K: gl.constexpr,
    W_CACHE_MODIFIER: gl.constexpr,
    SCALE_K_CONTIG: gl.constexpr,
    # layouts (gfx942_a16w4_layouts)
    LOAD_X: gl.constexpr,
    LOAD_W: gl.constexpr,
    LOAD_S: gl.constexpr,
    MFMA_LAYOUT: gl.constexpr,
    num_warps: gl.constexpr,
    UPCAST_INDICES: gl.constexpr = False,
):
    gl.assume(stride_y_k >= 0)
    gl.assume(stride_y_m >= 0)
    gl.assume(stride_y_n >= 0)
    gl.assume(stride_x_m >= 0)
    gl.assume(stride_x_k >= 0)
    gl.assume(stride_w_e >= 0)
    gl.assume(stride_w_n >= 0)
    gl.assume(stride_w_mx_e >= 0)
    gl.assume(stride_w_mx_k >= 0)
    gl.assume(stride_w_mx_n >= 0)
    if B is not None:
        gl.assume(stride_b_e >= 0)
    gl.assume(grid_m >= 0)
    gl.assume(grid_n >= 0)

    gl.static_assert(W.dtype.element_ty == gl.uint8, "mx_weight_ptr must be uint8")
    gl.static_assert(WMxScale.dtype.element_ty == gl.uint8, "mx_scale_ptr must be uint8")
    gl.static_assert(X.dtype.element_ty == gl.bfloat16, "activations must be bf16")
    gl.static_assert(BLOCK_K % 128 == 0, "BLOCK_K must be a multiple of 128")

    KG: gl.constexpr = BLOCK_K // 32  # e8m0 groups (= packed dwords per lane) per tile
    PK2: gl.constexpr = BLOCK_K // 4  # uint16 elements of packed fp4 per column per tile
    OUT_BLOCK_N: gl.constexpr = BLOCK_N // ACTIVATION_REDUCTION_N
    yN = N // ACTIVATION_REDUCTION_N

    DOT_X: gl.constexpr = gl.DotOperandLayout(operand_index=0, parent=MFMA_LAYOUT, k_width=8)
    DOT_W: gl.constexpr = gl.DotOperandLayout(operand_index=1, parent=MFMA_LAYOUT, k_width=8)
    DOT_W16: gl.constexpr = gl.DotOperandLayout(operand_index=1, parent=MFMA_LAYOUT, k_width=2)
    SMEM_X: gl.constexpr = gl.SwizzledSharedLayout(8, 1, 16, order=[1, 0])

    pid = gl.program_id(0)
    if ExptOffsSum is not None and XCD_SWIZZLE > 1:
        # true grid size: skip the padding tiles of the expert data
        padding_m = grid_m - gl.load(ExptOffsSum)
    else:
        padding_m: gl.constexpr = 0

    index_type: gl.constexpr = gl.int64 if UPCAST_INDICES else gl.int32

    unpadded_m = grid_m - padding_m
    gl.assume(unpadded_m >= 0)
    total_actual_tiles = unpadded_m * grid_n * SPLIT_K
    if padding_m > 0 and pid >= total_actual_tiles:
        return

    # swizzle program ids
    pid_emnk = pid
    if XCD_SWIZZLE != 1:
        pid_emnk = xcd_swizzle(pid_emnk, total_actual_tiles, XCD_SWIZZLE)
    pid_mnk = pid_emnk % (unpadded_m * grid_n * SPLIT_K)
    pid_k = pid_mnk % SPLIT_K
    pid_mn = pid_mnk // SPLIT_K
    pid_m, pid_n = pid_grid(pid_mn, unpadded_m, grid_n, GROUP_M)
    if SPLIT_K > 1:
        Y += pid_k.to(index_type) * stride_y_k
    # unpack expert data
    expt_data = gl.load(ExptData + pid_m)
    if expt_data == -1:
        return
    expt_id = expt_data & 0x0000FFFF
    block_id = expt_data >> 16
    M = gl.load(ExptHist + expt_id)
    start_m = gl.load(ExptOffs + expt_id)
    if HAS_EXPERT_MAP:
        w_expt_id = gl.load(ExpertMap + expt_id)
        if w_expt_id < 0:
            return
    else:
        w_expt_id = expt_id
    expt_id, block_id = expt_id.to(index_type), block_id.to(index_type)
    w_expt_id = w_expt_id.to(index_type)
    start_m = start_m.to(index_type)
    pid_n, pid_k = pid_n.to(index_type), pid_k.to(index_type)

    # ---- X pointers (physical K order; the operand is relabelled in registers)
    offs_x_m = BLOCK_M * block_id + gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, LOAD_X))
    offs_x_m = gl.max_contiguous(gl.multiple_of(offs_x_m % M, BLOCK_M), BLOCK_M)
    if GatherIndx is None:
        X += start_m * stride_x_m
    else:
        GatherIndx += start_m
        # no bounds check needed: offs_x_m wraps around M
        offs_x_m = gl.load(GatherIndx + offs_x_m) // N_EXPTS_ACT
    # K permutation at chunk (8-element) granularity: the W tile's lane group g holds the KG
    # physically consecutive dwords KG*g .. KG*g+KG-1, which the B-operand layout interprets as
    # semantic dwords d = g + 4*kt; load X so that semantic chunk d comes from physical chunk
    # KG*(d%4) + d//4 (chunks stay contiguous, so the 16-byte loads survive). The order inside a
    # chunk is fixed up in registers by _relabel_x before the LDS stage.
    k_load = gl.arange(0, BLOCK_K, layout=gl.SliceLayout(0, LOAD_X))
    d_sem = k_load // 8
    offs_x_k = BLOCK_K * pid_k + 8 * (KG * (d_sem % 4) + d_sem // 4) + (k_load % 8)
    offs_x_k = gl.max_contiguous(gl.multiple_of(offs_x_k, 8), 8)
    XPtrs = X + offs_x_m.to(index_type)[:, None] * stride_x_m + offs_x_k.to(index_type)[None, :] * stride_x_k

    # ---- W pointers: uint16 view, lane (g, n) reads 2*KG contiguous uint16 of column n
    W16 = W.to(gl.pointer_type(gl.uint16))
    offs_w_n = pid_n * BLOCK_N + gl.arange(0, BLOCK_N, layout=gl.SliceLayout(0, LOAD_W))
    offs_w_n = gl.max_contiguous(gl.multiple_of(offs_w_n % N, BLOCK_N), BLOCK_N)
    offs_w_q = PK2 * pid_k + gl.arange(0, PK2, layout=gl.SliceLayout(1, LOAD_W))
    offs_w_q = gl.max_contiguous(gl.multiple_of(offs_w_q, 2 * KG), 2 * KG)
    # column stride and expert stride are multiples of 16 bytes (K % 256 == 0): keep the 16-byte
    # vector loads by stating it for the combined offsets
    offs_w = (
        offs_w_q.to(index_type)[:, None]
        + offs_w_n.to(index_type)[None, :] * (stride_w_n // 2)
        + w_expt_id * (stride_w_e // 2)
    )
    offs_w = gl.max_contiguous(gl.multiple_of(offs_w, [8, 8]), [2 * KG, 1])
    WPtrs = W16 + offs_w

    # ---- scale pointers: (KGL, 1, 4, BLOCK_N); lane (g, n) reads the KGL groups of its dwords
    KGL: gl.constexpr = KG // 4
    offs_s_j = gl.arange(0, KGL, layout=gl.SliceLayout(1, gl.SliceLayout(2, gl.SliceLayout(3, LOAD_S))))
    offs_s_g = gl.arange(0, 4, layout=gl.SliceLayout(0, gl.SliceLayout(1, gl.SliceLayout(3, LOAD_S))))
    offs_s_n = pid_n * BLOCK_N + gl.arange(0, BLOCK_N, layout=gl.SliceLayout(0, gl.SliceLayout(1, gl.SliceLayout(2, LOAD_S))))
    offs_s_n = gl.max_contiguous(gl.multiple_of(offs_s_n % N, BLOCK_N), BLOCK_N)
    offs_s_kg = KG * pid_k + offs_s_g.to(index_type)[None, None, :, None] * KGL + offs_s_j.to(index_type)[:, None, None, None]
    if SCALE_K_CONTIG:
        offs_s = offs_s_kg + offs_s_n.to(index_type)[None, None, None, :] * stride_w_mx_n + w_expt_id * stride_w_mx_e
        offs_s = gl.max_contiguous(gl.multiple_of(offs_s, [KGL, 1, 1, 1]), [KGL, 1, 1, 1])
    else:
        offs_s = offs_s_kg * stride_w_mx_k + offs_s_n.to(index_type)[None, None, None, :] * stride_w_mx_n + w_expt_id * stride_w_mx_e
    SPtrs = WMxScale + offs_s

    x_smem = gl.allocate_shared_memory(gl.bfloat16, [BLOCK_M, BLOCK_K], SMEM_X)
    tlo0 = _const_u16_pairs((PK2, BLOCK_N), TLO0, DOT_W16)
    tlo1 = _const_u16_pairs((PK2, BLOCK_N), TLO1, DOT_W16)
    thi0 = _const_u16_pairs((PK2, BLOCK_N), THI0, DOT_W16)
    thi1 = _const_u16_pairs((PK2, BLOCK_N), THI1, DOT_W16)

    acc = gl.zeros((BLOCK_M, BLOCK_N), dtype=gl.float32, layout=MFMA_LAYOUT)
    num_k_iter = K // (BLOCK_K * SPLIT_K)
    for k in range(num_k_iter):
        x = gl.load(XPtrs)
        wq = gl.load(WPtrs, cache_modifier=W_CACHE_MODIFIER)
        sc = gl.load(SPtrs, cache_modifier=W_CACHE_MODIFIER)

        # weights: relabel into the k_width=2 operand layout, spread the scales, dequantise
        wu = _relabel_w(wq, KG, BLOCK_N, DOT_W16)
        sc16 = sc.to(gl.uint16)  # (KGL, 1, 4, BLOCK_N)
        sc16 = gl.reshape(sc16.broadcast_to((KGL, 4, 4, BLOCK_N)), (KG, 4, BLOCK_N))  # dword kt = 4*j + r -> group j
        sc16 = gl.join(sc16, sc16)  # both 16-bit halves of each dword
        # register-local broadcast + relabel into the k_width=2 operand layout (no cross-lane movement)
        sc16 = gl.convert_layout(gl.reshape(gl.permute(sc16, (0, 1, 3, 2)), (PK2, BLOCK_N)), DOT_W16)
        wb = _dequant_tile(wu, sc16, tlo0, tlo1, thi0, thi1, BLOCK_K, BLOCK_N, DOT_W)

        # activations: stage through LDS in the semantic K order of the W operand
        x_smem.store(_relabel_x(x, BLOCK_M, BLOCK_K))
        gl.barrier()
        xa = x_smem.load(DOT_X)
        gl.barrier()

        acc = gl.amd.cdna3.mfma(xa, wb, acc)

        XPtrs += (BLOCK_K * SPLIT_K) * stride_x_k
        WPtrs += PK2 * SPLIT_K
        if SCALE_K_CONTIG:
            SPtrs += KG * SPLIT_K
        else:
            SPtrs += (KG * SPLIT_K) * stride_w_mx_k

    # ---- epilogue
    offs_m = BLOCK_M * block_id + gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, MFMA_LAYOUT))
    offs_y_n = BLOCK_N * pid_n + gl.arange(0, BLOCK_N, layout=gl.SliceLayout(0, MFMA_LAYOUT))
    mask_m = offs_m < M
    mask_n = offs_y_n < N
    if B is not None:
        BPtrs = B + w_expt_id * stride_b_e + offs_y_n
        if pid_k == 0:
            bias = gl.load(BPtrs, mask=mask_n, other=0, cache_modifier=W_CACHE_MODIFIER)
        else:
            bias = gl.full([BLOCK_N], 0, dtype=gl.float32, layout=gl.SliceLayout(0, MFMA_LAYOUT))
        acc = acc + bias[None, :]
    if APPLY_SWIGLU and SPLIT_K == 1:
        out = _swiglu(acc, alpha, limit, ADD_RESIDUAL=SWIGLU_ADD_RESIDUAL)
        gl.static_assert(
            out.shape[1] == OUT_BLOCK_N,
            f"Activation fn out.shape[1] ({out.shape[1]}) doesn't match computed OUT_BLOCK_N ({OUT_BLOCK_N})",
        )
        out = gl.convert_layout(out, MFMA_LAYOUT)
        offs_y_n = OUT_BLOCK_N * pid_n + gl.arange(0, OUT_BLOCK_N, layout=gl.SliceLayout(0, MFMA_LAYOUT))
        mask_n = offs_y_n < yN
    else:
        gl.static_assert(
            ACTIVATION_REDUCTION_N == 1,
            "Activation reduction must be 1 if no activation fn is provided",
        )
        out = acc
    if Gammas is not None:
        gammas = gl.load(Gammas + start_m + offs_m, mask=mask_m, other=0.0)
        out *= gammas[:, None]

    # write-back
    Y += start_m * stride_y_m
    YPtrs = Y + offs_m.to(index_type)[:, None] * stride_y_m + offs_y_n.to(index_type)[None, :] * stride_y_n
    mask = mask_m[:, None] & mask_n[None, :]
    gl.store(YPtrs, out.to(Y.dtype.element_ty), mask=mask)
