# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Gluon (gfx1250) BF16 BMM with the R1 NVFP4 MLA Q quantization in the epilogue.

One persistent workgroup per (head, split) loads the head's weight into
registers and walks row tiles. Each tile goes through two WMMA halves, is
quantized in registers and leaves by TDM stores.
"""

from triton.experimental import gluon
from triton.experimental.gluon import language as gl

from aiter.ops.triton.utils._triton.kernel_repr import make_kernel_repr

_Q_GAIN = gl.constexpr(6.72)
# RN(1 / 6) and RN(1 / fp32(6.72)).
_RCP_6 = gl.constexpr(0.1666666716337204)
_RCP_GAIN = gl.constexpr(0.1488095223903656)


@gluon.jit
def _rcp_approx(x):
    return gl.inline_asm_elementwise(
        "v_rcp_f32 $0, $1", "=v,v", [x], dtype=gl.float32, is_pure=True, pack=1
    )


@gluon.jit
def _block_scale(amax):
    """The E4M3 scale code of a 16-value block (amax / 6 at the gain), its
    divisor scale / gain and RN(1 / divisor). The divisions are a multiply and
    an FMA correction, exact for these operands."""
    rcp6 = gl.full_like(amax, _RCP_6)
    q6 = amax * rcp6
    ideal = gl.fma(gl.fma(gl.full_like(amax, -6.0), q6, amax), rcp6, q6)
    ideal = gl.where(amax == 0, 1.0, ideal)
    gain = gl.full_like(amax, _Q_GAIN)
    t = gl.minimum(gl.maximum(ideal * gain, 1.0 / 512), 448.0)
    s8 = t.to(gl.float8e4nv)
    scale = s8.to(gl.float32)
    scale_code = s8.to(gl.uint8, bitcast=True)
    rcp_gain = gl.full_like(amax, _RCP_GAIN)
    qg = scale * rcp_gain
    div = gl.fma(gl.fma(-gain, qg, scale), rcp_gain, qg)
    # The negation is on r, not on div, so a NaN divisor keeps its sign.
    r = _rcp_approx(div)
    recip = gl.fma(gl.fma(div, -r, gl.full_like(div, 1.0)), r, r)
    return scale_code, div, recip


@gluon.jit
def _quantize_regs(
    acc, c_blocks, s_smem, blk0, BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr
):
    """Rounds ``acc`` to BF16 and quantizes it where it is. In the transposed
    WMMA layout a lane holds 8 columns of a 16-column block."""
    x3 = gl.reshape(acc.to(gl.bfloat16).to(gl.float32), [BLOCK_M, BLOCK_N // 16, 16])
    amax = gl.max(gl.abs(x3), axis=2)
    scale_code, div, recip = _block_scale(amax)
    div3 = gl.expand_dims(div, 2) + gl.zeros_like(x3)
    recip3 = gl.expand_dims(recip, 2) + gl.zeros_like(x3)
    # x / div correctly rounded; the remainder is negated so +-0 keep the sign.
    q0 = x3 * recip3
    q = gl.fma(-gl.fma(div3, q0, -x3), recip3, q0)
    # FP32 -> E2M1 (nearest even, saturating) at scale 2**0, low nibble first.
    one = gl.expand_dims(gl.full_like(amax, 127, dtype=gl.uint8), 2)
    packed = gl.amd.gfx1250.scaled_downcast(q, one, "e2m1", axis=2)
    c_blocks.slice(blk0, BLOCK_N // 16, dim=1).store(packed)
    s_smem.slice(blk0, BLOCK_N // 16, dim=1).store(scale_code.to(gl.uint8))


_fused_bf16_bmm_nvfp4_quant_mla_repr = make_kernel_repr(
    "_fused_bf16_bmm_nvfp4_quant_mla_kernel",
    ["BLOCK_M", "BLOCK_N", "BLOCK_K", "WARP_BASES", "num_warps"],
)


@gluon.jit(repr=_fused_bf16_bmm_nvfp4_quant_mla_repr)
def _fused_bf16_bmm_nvfp4_quant_mla_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    s_ptr,
    M,
    N,
    K,
    TILES,
    stride_ab,
    stride_am,
    stride_bb,
    stride_bn,
    stride_cb,
    stride_cm,
    stride_sb,
    stride_sm,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    BLOCK_K: gl.constexpr,
    SHARED_LAYOUT_A: gl.constexpr,
    SHARED_LAYOUT_B: gl.constexpr,
    WMMA_LAYOUT: gl.constexpr,
    WARP_BASES: gl.constexpr,
    OPERAND_LAYOUT_A: gl.constexpr,
    OPERAND_LAYOUT_B: gl.constexpr,
    num_warps: gl.constexpr,
):
    # The workgroup does rows m0 .. m0 + TILES * BLOCK_M of one head. N is
    # 2 * BLOCK_N and K is BLOCK_K: the head's weight is two B tiles in LDS.
    head = gl.program_id(axis=0)
    m0 = gl.program_id(axis=1) * TILES * BLOCK_M
    n_tiles = gl.minimum(TILES, gl.cdiv(M - m0, BLOCK_M))
    NC: gl.constexpr = BLOCK_N  # code bytes per row, two codes per byte
    NS: gl.constexpr = BLOCK_N // 8  # scales per row
    CODES: gl.constexpr = gl.PaddedSharedLayout.with_identity_for(
        [[NC, 16]], [BLOCK_M, NC], [1, 0]
    )
    SCALES: gl.constexpr = gl.PaddedSharedLayout.with_identity_for(
        [[NS, 16]], [BLOCK_M, NS], [1, 0]
    )
    a_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
        base=a_ptr + head * stride_ab,
        shape=(M, K),
        strides=(stride_am, 1),
        block_shape=(BLOCK_M, BLOCK_K),
        layout=SHARED_LAYOUT_A,
    )
    b_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
        base=b_ptr + head * stride_bb,
        shape=(N, K),
        strides=(stride_bn, 1),
        block_shape=(BLOCK_N, BLOCK_K),
        layout=SHARED_LAYOUT_B,
    )
    c_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
        base=c_ptr + head * stride_cb,
        shape=(M, NC),
        strides=(stride_cm, 1),
        block_shape=(BLOCK_M, NC),
        layout=CODES,
    )
    s_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
        base=s_ptr + head * stride_sb,
        shape=(M, NS),
        strides=(stride_sm, 1),
        block_shape=(BLOCK_M, NS),
        layout=SCALES,
    )
    a_buf = gl.allocate_shared_memory(
        a_ptr.type.element_ty, [2, BLOCK_M, BLOCK_K], SHARED_LAYOUT_A
    )
    b0 = gl.allocate_shared_memory(
        b_ptr.type.element_ty, [BLOCK_N, BLOCK_K], SHARED_LAYOUT_B
    )
    b1 = gl.allocate_shared_memory(
        b_ptr.type.element_ty, [BLOCK_N, BLOCK_K], SHARED_LAYOUT_B
    )
    c_buf = gl.allocate_shared_memory(gl.uint8, [2, BLOCK_M, NC], CODES)
    s_buf = gl.allocate_shared_memory(gl.uint8, [2, BLOCK_M, NS], SCALES)
    gl.amd.gfx1250.tdm.async_load(b_desc, [0, 0], b0)
    gl.amd.gfx1250.tdm.async_load(b_desc, [BLOCK_N, 0], b1)
    gl.amd.gfx1250.tdm.async_load(a_desc, [m0, 0], a_buf.index(0))
    # A wait covers only this warp's part of a TDM load; the barrier after it
    # makes the whole tile visible.
    gl.amd.gfx1250.tdm.async_wait(0)
    gl.barrier()
    zero = gl.zeros((BLOCK_M, BLOCK_N), dtype=gl.float32, layout=WMMA_LAYOUT)
    # Reuse the weight operands across row tiles. Quantize each half before
    # the next WMMA to shorten the accumulator lifetime.
    w0 = b0.permute([1, 0]).load(OPERAND_LAYOUT_B)
    w1 = b1.permute([1, 0]).load(OPERAND_LAYOUT_B)
    for t in range(n_tiles):
        if t + 1 < n_tiles:
            # Every warp read this buffer (tile t - 1) before the last barrier.
            gl.amd.gfx1250.tdm.async_load(
                a_desc, [m0 + (t + 1) * BLOCK_M, 0], a_buf.index((t + 1) % 2)
            )
        cur_a = a_buf.index(t % 2).load(OPERAND_LAYOUT_A)
        acc0 = gl.amd.gfx1250.wmma(cur_a, w0, zero)
        c_blocks = c_buf.index(t % 2).reshape([BLOCK_M, NS, 8])
        _quantize_regs(acc0, c_blocks, s_buf.index(t % 2), 0, BLOCK_M, BLOCK_N)
        acc1 = gl.amd.gfx1250.wmma(cur_a, w1, zero)
        _quantize_regs(
            acc1, c_blocks, s_buf.index(t % 2), BLOCK_N // 16, BLOCK_M, BLOCK_N
        )
        gl.barrier()
        gl.amd.gfx1250.tdm.async_store(
            c_desc, [m0 + t * BLOCK_M, 0], c_buf.index(t % 2)
        )
        gl.amd.gfx1250.tdm.async_store(
            s_desc, [m0 + t * BLOCK_M, 0], s_buf.index(t % 2)
        )
        # TDM ops complete in order: with at most this tile's two stores in
        # flight, the next A tile is in and tile t - 1's buffers are free.
        gl.amd.gfx1250.tdm.async_wait(2)
        gl.barrier()
    gl.amd.gfx1250.tdm.async_wait(0)
