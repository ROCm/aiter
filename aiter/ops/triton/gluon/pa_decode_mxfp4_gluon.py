# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Sparse paged-attention decode over an MXFP4 K/V cache (gfx942 gluon).

Modelled on ``pa_decode_gluon`` (the ``ps=True`` path): fp8 (e4m3fnuz) QK / PV
MFMAs, online softmax per 128/256-token partition, split partials merged by a
small reduce. Specialized to head_dim 128, GQA group 16, page-16 sparse tables.

Each token's packed e2m1 dwords and its four e8m0 chunk scales are loaded
straight into the MFMA operand layouts and decoded in registers to e4m3fnuz
with the chunk scale folded in relative to the token's largest chunk exponent
(``v_perm_b32`` lookup; per-chunk table words from ``v_cvt_pk_fp8_f32``). The
token's 2^(emax-127) then rides the per-token KV-scale path (K: into the
softmax scale; V: the fp8 P rescale), as KV_QUANT_MODE == 1 does for fp8 KV.
Decoded values are exact while representable in e4m3fnuz (chunk exponents
within 6 binades of the token max; RNE into the fp8 subnormals below that).

Cache layout ``kv_cache``: uint8 ``[num_blocks, 2H, 128, 68]``; per (block, head
slot) an 8704-byte page ``[128 tok x 64 B e2m1][128 tok x 4 B e8m0]``; K heads in
slots ``[0, H)``, V heads in ``[H, 2H)``. ``block_tables`` row ``r*H + h`` lists
page-16 entries ``(block*8 + j)*H + h`` (sub-page j = tokens 16j..16j+15).
"""

import torch
import triton
import triton.language as tl
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.language.extra.hip import libdevice as hip_libdevice

from aiter.ops.triton.gluon.pa_decode_gluon import (
    _fused_max_combine,
    get_cdna_version,
)

PAGE_TOKENS = 128
HEAD_DIM = 128
PAGE_BYTES = PAGE_TOKENS * (HEAD_DIM // 2 + HEAD_DIM // 32)  # 8704

# One packed dword $5 (8 e2m1 nibbles, element 2i = low nibble of byte i) ->
# $0 = fp8 bytes of elements 0..3, $1 = elements 4..7. $6 / $7 are the
# per-(token, chunk) LUT words for magnitudes 0..3 / 4..7 (v_perm_b32 picks
# bytes 0-3 from src1, 4-7 from src0). FNUZ: -0 must not become 0x80 (NaN), so
# the sign is applied only where the magnitude byte is nonzero (byte + 0x7F
# sets bit 7 iff byte != 0; magnitudes are <= 0x54 so no carry escapes).
# gfx942 VOP3 takes no literals, hence the v_mov for the selectors.
_DEC_ASM = gl.constexpr("""
v_and_b32 $2, 0x07070707, $5
v_lshrrev_b32 $3, 4, $5
v_and_b32 $3, 0x07070707, $3
v_perm_b32 $2, $7, $6, $2
v_perm_b32 $3, $7, $6, $3
v_add_u32 $4, 0x7F7F7F7F, $2
v_and_b32 $4, 0x80808080, $4
v_lshlrev_b32 $0, 4, $5
v_and_or_b32 $2, $0, $4, $2
v_add_u32 $4, 0x7F7F7F7F, $3
v_and_b32 $4, 0x80808080, $4
v_and_or_b32 $3, $5, $4, $3
v_mov_b32 $4, 0x05010400
v_perm_b32 $0, $3, $2, $4
v_mov_b32 $4, 0x07030602
v_perm_b32 $1, $3, $2, $4
s_nop 4
""")
_DEC_CON = gl.constexpr("=&v,=&v,=&v,=&v,=&v,v,v,v")


@gluon.jit
def _perm(hi, lo, sel):
    """v_perm_b32: byte i of the result = byte sel[i] of the 8-byte {hi, lo}."""
    return gl.inline_asm_elementwise(
        "v_perm_b32 $0, $1, $2, $3",
        "=v,v,v,v",
        [hi, lo, sel],
        dtype=gl.int32,
        is_pure=True,
        pack=1,
    )


@gluon.jit
def _perm_mfma(hi, lo, sel):
    """v_perm_b32 whose result feeds an MFMA operand directly: LLVM's hazard
    recognizer does not see inside inline asm, so wait out VALU -> MFMA here."""
    return gl.inline_asm_elementwise(
        "v_perm_b32 $0, $1, $2, $3\ns_nop 2",
        "=v,v,v,v",
        [hi, lo, sel],
        dtype=gl.int32,
        is_pure=True,
        pack=1,
    )


@gluon.jit
def _dec_dword(d, lo, hi):
    """8 packed e2m1 nibbles (element 2i = low nibble of byte i) -> (fp8 bytes
    of elements 0, 2, 4, 6; of elements 1, 3, 5, 7) using the per-chunk LUT
    words (lo: magnitudes 0..3, hi: 4..7). The sign is OR-ed in unguarded: the
    LUT's magnitude-0 byte is 0x01 (not 0x00), so FNUZ -0 never becomes 0x80."""
    me = _perm(hi, lo, d & 0x07070707)
    mo = _perm(hi, lo, (d >> 4) & 0x07070707)
    return me | ((d << 4) & -0x7F7F7F80), mo | (d & -0x7F7F7F80)


@gluon.jit
def _cvt_pk4(a, b, c, d):
    """Four f32 -> one dword of e4m3fnuz bytes [a, b, c, d] (RNE, saturating)."""
    return gl.inline_asm_elementwise(
        "v_cvt_pk_fp8_f32 $0, $1, $2\nv_cvt_pk_fp8_f32 $0, $3, $4 op_sel:[0,0,1]",
        "=&v,v,v,v,v",
        [a, b, c, d],
        dtype=gl.int32,
        is_pure=True,
        pack=1,
    )


@gluon.jit
def _e2m1_lut(k):
    """k = emax - e_chunk (>= 0) -> e4m3fnuz bytes of the e2m1 magnitudes
    [0, .5, 1, 1.5] and [2, 3, 4, 6] times 2^-k (exact while representable,
    RNE into the fp8 subnormals, zero past them). Magnitude 0 maps to the
    smallest subnormal 2^-10 (byte 0x01), i.e. 2^-10 / 6 of the token's
    largest element: far below fp8 rounding, and it lets the decode OR the
    sign in without a -0 -> NaN guard."""
    sc = ((127 - gl.minimum(k, 100)) << 23).to(gl.float32, bitcast=True)
    lo = _cvt_pk4(
        gl.full(k.shape, 0.0009765625, gl.float32, k.type.layout),
        sc * 0.5,
        sc,
        sc * 1.5,
    )
    hi = _cvt_pk4(sc * 2.0, sc * 3.0, sc * 4.0, sc * 6.0)
    return lo, hi


@gluon.jit
def _lut_words(w, shift):
    """Token scale words w, chunk shift -> per-(token, chunk) LUT words
    (relative to the token's largest chunk exponent)."""
    emax = gl.maximum(
        gl.maximum(w & 255, (w >> 8) & 255),
        gl.maximum((w >> 16) & 255, (w >> 24) & 255),
    )
    emax = gl.minimum(emax, 254)
    ec = (w >> shift) & 255
    return _e2m1_lut(gl.maximum(emax - ec, 0))


@gluon.jit
def _spread4(x, layout: gl.constexpr):
    """[NW, SPW, 16, 4 (chunk)] -> [NW, SPW, 16, 16 (dword g = 4 * chunk + j)]:
    LUT words computed once per (token, chunk), then replicated to the four
    dwords of the chunk by a register/lane shuffle."""
    x = gl.join(gl.join(x, x), gl.join(x, x))
    return gl.convert_layout(gl.reshape(x, [x.shape[0], x.shape[1], 16, 16]), layout)


@gluon.jit
def _decode_v(raw, lo, hi, NW: gl.constexpr, SPW: gl.constexpr):
    """raw int32 [NW, SPW, 16, 16] packed e2m1 (dword g = head dims 8g..8g+7) of
    [warp, sub-page, token, g] with its LUT words lo / hi -> uint8
    [NW, SPW * 16, 128] e4m3fnuz, column n = 16 * e + g holding dim 8g + e."""
    o0, o1 = _dec_dword(raw, lo, hi)
    d = gl.join(o0, o1)  # [4, 4, 16, 16, 2]
    b0 = d.to(gl.uint8)
    b1 = (d >> 8).to(gl.uint8)
    b2 = (d >> 16).to(gl.uint8)
    b3 = (d >> 24).to(gl.uint8)
    x = gl.reshape(gl.join(gl.join(b0, b2), gl.join(b1, b3)), [NW, SPW, 16, 16, 8])
    x = gl.permute(x, [0, 1, 2, 4, 3])  # [w, sp, t, e, g]
    return gl.reshape(x, [NW, SPW * 16, 128])


@gluon.jit
def _decode_k(raw, w, shift):
    """raw int32 [8, PART, 2] (dword 2 qp + h of token n), w int32 [1, PART, 1]
    scale words, shift [8, 1, 1] -> uint8 [128, PART] e4m3fnuz, row = head dim.
    The LUT is computed once per (dword pair, token): both dwords of a pair
    lie in the same 32-dim chunk."""
    emax = gl.maximum(
        gl.maximum(w & 255, (w >> 8) & 255),
        gl.maximum((w >> 16) & 255, (w >> 24) & 255),
    )
    emax = gl.minimum(emax, 254)
    ec = (w >> shift) & 255
    lo, hi = _e2m1_lut(gl.maximum(emax - ec, 0))
    o0, o1 = _dec_dword(raw, lo, hi)
    d = gl.join(o0, o1)  # [8, PART, 2 (half), 2 (even/odd elements)]
    b0 = d.to(gl.uint8)
    b1 = (d >> 8).to(gl.uint8)
    b2 = (d >> 16).to(gl.uint8)
    b3 = (d >> 24).to(gl.uint8)
    x = gl.reshape(gl.join(gl.join(b0, b2), gl.join(b1, b3)), [8, raw.shape[1], 16])
    x = gl.permute(
        x, [0, 2, 1]
    )  # [qp, half * 8 + e, token]; e' = 4 w + m holds element 2 m + w
    return gl.reshape(x, [128, raw.shape[1]])


@gluon.jit
def _token_scale(w):
    emax = gl.maximum(
        gl.maximum(w & 255, (w >> 8) & 255),
        gl.maximum((w >> 16) & 255, (w >> 24) & 255),
    )
    return (gl.minimum(emax, 254) << 23).to(gl.float32, bitcast=True)


@gluon.jit
def _page_entries(bt_ptr, bt_off, start, idx, max_nb):
    """Page-16 table entries start + idx of a row, clamped into the row's valid
    range: no masks / branches, so the loads issue together (a clamped entry
    duplicates a valid page; its tokens are masked out of the softmax)."""
    return gl.load(bt_ptr + bt_off + gl.minimum(start + idx, max_nb - 1))


@gluon.jit
def _entry_bases(e, stride_kv_block, NUM_KV_HEADS: gl.constexpr):
    """Page-16 entry -> (byte offset of its (block, head) page, first token row)."""
    e = e.to(gl.uint32)  # entries are >= 0; unsigned div/mod by constants is shifts
    h = (e % NUM_KV_HEADS).to(gl.int32)
    j = ((e // NUM_KV_HEADS) % 8).to(gl.int32)
    blk = (e // (8 * NUM_KV_HEADS)).to(gl.int32)
    base = blk.to(gl.int64) * stride_kv_block + h * 8704
    return base, j * 16


@gluon.jit
def pa_decode_mxfp4_kernel(
    exp_sums_ptr,  # [num_seqs, 1, max_parts, group]
    max_logits_ptr,
    output_ptr,  # one-shot: [num_seqs, group, D]; else [num_seqs, 1, parts, group, D]
    query_ptr,  # [num_seqs, group, D] bf16
    kv_ptr,  # uint8 cache
    kv_i32_ptr,  # same cache as int32
    block_tables_ptr,  # [num_seqs, >= 128] int32 page-16 entries
    context_lengths_ptr,  # [num_seqs]
    softmax_scale,
    stride_ml_seq,
    stride_ml_part,
    stride_out_seq,
    stride_out_part,
    stride_out_group,
    stride_q_seq,
    stride_q_group,
    stride_kv_block,  # bytes per cache block (2H * 8704)
    stride_bt_seq,
    query_group_size,
    bt_cols,  # entries per block-table row
    NUM_KV_HEADS: gl.constexpr,
    CDNA_VERSION: gl.constexpr,
    ONE_SHOT: gl.constexpr,
    NUM_WARPS: gl.constexpr = 4,
    PART: gl.constexpr = 256,
):
    GROUP: gl.constexpr = 16
    HEAD: gl.constexpr = 128
    NB: gl.constexpr = PART // 16
    COMPUTE_TYPE: gl.constexpr = gl.float8e4b8
    FP8_MAX: gl.constexpr = 240.0
    LOG2_E: gl.constexpr = 1.4426950408889634
    V_OFF: gl.constexpr = NUM_KV_HEADS * 8704

    seq = gl.program_id(0)
    split = gl.program_id(2)

    blocked_query_layout: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1, 8],
        threads_per_warp=[4, 16],
        warps_per_cta=[max(NUM_WARPS // 4, 1), min(NUM_WARPS, 4)],
        order=[1, 0],
    )
    mfma_layout: gl.constexpr = gl.amd.AMDMFMALayout(
        version=CDNA_VERSION,
        instr_shape=[16, 16, 32],
        transposed=True,
        warps_per_cta=[1, NUM_WARPS],
    )
    lhs_layout: gl.constexpr = gl.DotOperandLayout(
        operand_index=0, parent=mfma_layout, k_width=16
    )
    rhs_layout: gl.constexpr = gl.DotOperandLayout(
        operand_index=1, parent=mfma_layout, k_width=16
    )
    qk_linear_layout: gl.constexpr = mfma_layout
    # PV runs as a batched MFMA: warp w owns tokens 64w..64w+63 of the partition
    # and all 128 output columns; the four partial accumulators are summed once
    # at the end. Output column n holds head dim d = 8 * (n % 16) + n // 16, so
    # the eight columns a lane holds per token are one packed e2m1 dword: the
    # V loads stay 64 B-coalesced per token and the token-major -> K-contiguous
    # transpose the MFMA operand needs is register-local.
    NW: gl.constexpr = NUM_WARPS
    TPW: gl.constexpr = PART // NW  # tokens per warp slice (64 or 32)
    SPW: gl.constexpr = TPW // 16  # sub-pages per warp slice
    pv3_layout: gl.constexpr = gl.amd.AMDMFMALayout(
        version=CDNA_VERSION,
        instr_shape=[16, 16, 32],
        transposed=True,
        warps_per_cta=[NW, 1, 1],
    )
    p3_layout: gl.constexpr = gl.DotOperandLayout(
        operand_index=0, parent=pv3_layout, k_width=8
    )
    v3_layout: gl.constexpr = gl.DotOperandLayout(
        operand_index=1, parent=pv3_layout, k_width=8
    )
    # [warp, sub-page in warp, token in sub-page, dword] matching v3_layout
    gl.static_assert((NW != 1) or (PART == 32), "1 warp: 32-token partitions")
    gl.static_assert((NW != 2) or (PART == 64), "2 warps: 64-token partitions")
    if NW == 1:
        vload_layout: gl.constexpr = gl.DistributedLinearLayout(
            reg_bases=[[0, 0, 1, 0], [0, 0, 2, 0], [0, 0, 4, 0]],
            lane_bases=[
                [0, 0, 0, 1],
                [0, 0, 0, 2],
                [0, 0, 0, 4],
                [0, 0, 0, 8],
                [0, 0, 8, 0],
                [0, 1, 0, 0],
            ],
            warp_bases=[],
            block_bases=[],
            shape=[NW, SPW, 16, 16],
        )
        kload_layout: gl.constexpr = gl.DistributedLinearLayout(
            reg_bases=[[0, 0, 1], [4, 0, 0], [0, 16, 0]],
            lane_bases=[
                [0, 1, 0],
                [0, 2, 0],
                [0, 4, 0],
                [0, 8, 0],
                [1, 0, 0],
                [2, 0, 0],
            ],
            warp_bases=[],
            block_bases=[],
            shape=[8, 32, 2],
        )
    elif NW == 2:
        vload_layout: gl.constexpr = gl.DistributedLinearLayout(
            reg_bases=[[0, 0, 1, 0], [0, 0, 2, 0], [0, 0, 4, 0]],
            lane_bases=[
                [0, 0, 0, 1],
                [0, 0, 0, 2],
                [0, 0, 0, 4],
                [0, 0, 0, 8],
                [0, 0, 8, 0],
                [0, 1, 0, 0],
            ],
            warp_bases=[[1, 0, 0, 0]],
            block_bases=[],
            shape=[NW, SPW, 16, 16],
        )
        kload_layout: gl.constexpr = gl.DistributedLinearLayout(
            reg_bases=[[0, 0, 1], [4, 0, 0], [0, 32, 0]],
            lane_bases=[
                [0, 1, 0],
                [0, 2, 0],
                [0, 4, 0],
                [0, 8, 0],
                [1, 0, 0],
                [2, 0, 0],
            ],
            warp_bases=[[0, 16, 0]],
            block_bases=[],
            shape=[8, 64, 2],
        )
    elif NW == 4:
        if PART == 256:
            vload_layout: gl.constexpr = gl.DistributedLinearLayout(
                reg_bases=[[0, 0, 1, 0], [0, 0, 2, 0], [0, 0, 4, 0], [0, 2, 0, 0]],
                lane_bases=[
                    [0, 0, 0, 1],
                    [0, 0, 0, 2],
                    [0, 0, 0, 4],
                    [0, 0, 0, 8],
                    [0, 0, 8, 0],
                    [0, 1, 0, 0],
                ],
                warp_bases=[[1, 0, 0, 0], [2, 0, 0, 0]],
                block_bases=[],
                shape=[NW, SPW, 16, 16],
            )
        else:
            vload_layout: gl.constexpr = gl.DistributedLinearLayout(
                reg_bases=[[0, 0, 1, 0], [0, 0, 2, 0], [0, 0, 4, 0]],
                lane_bases=[
                    [0, 0, 0, 1],
                    [0, 0, 0, 2],
                    [0, 0, 0, 4],
                    [0, 0, 0, 8],
                    [0, 0, 8, 0],
                    [0, 1, 0, 0],
                ],
                warp_bases=[[1, 0, 0, 0], [2, 0, 0, 0]],
                block_bases=[],
                shape=[NW, SPW, 16, 16],
            )
        # [dword pair qp, token, half]: dword q = 2 qp + half (head dims 8q..8q+7)
        if PART == 256:
            kload_layout: gl.constexpr = gl.DistributedLinearLayout(
                reg_bases=[[0, 0, 1], [4, 0, 0], [0, 64, 0], [0, 128, 0]],
                lane_bases=[
                    [0, 1, 0],
                    [0, 2, 0],
                    [0, 4, 0],
                    [0, 8, 0],
                    [1, 0, 0],
                    [2, 0, 0],
                ],
                warp_bases=[[0, 16, 0], [0, 32, 0]],
                block_bases=[],
                shape=[8, 256, 2],
            )
        else:
            kload_layout: gl.constexpr = gl.DistributedLinearLayout(
                reg_bases=[[0, 0, 1], [4, 0, 0], [0, 64, 0]],
                lane_bases=[
                    [0, 1, 0],
                    [0, 2, 0],
                    [0, 4, 0],
                    [0, 8, 0],
                    [1, 0, 0],
                    [2, 0, 0],
                ],
                warp_bases=[[0, 16, 0], [0, 32, 0]],
                block_bases=[],
                shape=[8, 128, 2],
            )
    else:
        vload_layout: gl.constexpr = gl.DistributedLinearLayout(
            reg_bases=[[0, 0, 1, 0], [0, 0, 2, 0], [0, 0, 4, 0]],
            lane_bases=[
                [0, 0, 0, 1],
                [0, 0, 0, 2],
                [0, 0, 0, 4],
                [0, 0, 0, 8],
                [0, 0, 8, 0],
                [0, 1, 0, 0],
            ],
            warp_bases=[[1, 0, 0, 0], [2, 0, 0, 0], [4, 0, 0, 0]],
            block_bases=[],
            shape=[NW, SPW, 16, 16],
        )
        kload_layout: gl.constexpr = gl.DistributedLinearLayout(
            reg_bases=[[1, 0], [8, 0], [0, 128]],
            lane_bases=[[0, 1], [0, 2], [0, 4], [0, 8], [2, 0], [4, 0]],
            warp_bases=[[0, 16], [0, 32], [0, 64]],
            block_bases=[],
            shape=[16, 256],
        )
    shared_p3_layout: gl.constexpr = gl.SwizzledSharedLayout(8, 1, 8, order=[2, 1, 0])
    # compact [warp, sub-page, token, chunk]: one LUT per (token, chunk)
    if NW == 1:
        vlut_layout: gl.constexpr = gl.DistributedLinearLayout(
            reg_bases=[[0, 0, 1, 0]],
            lane_bases=[
                [0, 0, 0, 1],
                [0, 0, 0, 2],
                [0, 0, 2, 0],
                [0, 0, 4, 0],
                [0, 0, 8, 0],
                [0, 1, 0, 0],
            ],
            warp_bases=[],
            block_bases=[],
            shape=[NW, SPW, 16, 4],
        )
    elif NW == 2:
        vlut_layout: gl.constexpr = gl.DistributedLinearLayout(
            reg_bases=[[0, 0, 1, 0]],
            lane_bases=[
                [0, 0, 0, 1],
                [0, 0, 0, 2],
                [0, 0, 2, 0],
                [0, 0, 4, 0],
                [0, 0, 8, 0],
                [0, 1, 0, 0],
            ],
            warp_bases=[[1, 0, 0, 0]],
            block_bases=[],
            shape=[NW, SPW, 16, 4],
        )
    elif PART == 256:
        vlut_layout: gl.constexpr = gl.DistributedLinearLayout(
            reg_bases=[[0, 0, 1, 0], [0, 2, 0, 0]],
            lane_bases=[
                [0, 0, 0, 1],
                [0, 0, 0, 2],
                [0, 0, 2, 0],
                [0, 0, 4, 0],
                [0, 0, 8, 0],
                [0, 1, 0, 0],
            ],
            warp_bases=[[1, 0, 0, 0], [2, 0, 0, 0]],
            block_bases=[],
            shape=[NW, SPW, 16, 4],
        )
    else:
        vlut_layout: gl.constexpr = gl.DistributedLinearLayout(
            reg_bases=[[0, 0, 1, 0]],
            lane_bases=[
                [0, 0, 0, 1],
                [0, 0, 0, 2],
                [0, 0, 2, 0],
                [0, 0, 4, 0],
                [0, 0, 8, 0],
                [0, 1, 0, 0],
            ],
            warp_bases=[[1, 0, 0, 0], [2, 0, 0, 0]],
            block_bases=[],
            shape=[NW, SPW, 16, 4],
        )
    lw_idx = gl.arange(
        0,
        NW,
        layout=gl.SliceLayout(1, gl.SliceLayout(2, gl.SliceLayout(3, vlut_layout))),
    )
    lsp_idx = gl.arange(
        0,
        SPW,
        layout=gl.SliceLayout(0, gl.SliceLayout(2, gl.SliceLayout(3, vlut_layout))),
    )
    lt_idx = gl.arange(
        0,
        16,
        layout=gl.SliceLayout(0, gl.SliceLayout(1, gl.SliceLayout(3, vlut_layout))),
    )
    lc_idx = gl.arange(
        0,
        4,
        layout=gl.SliceLayout(0, gl.SliceLayout(1, gl.SliceLayout(2, vlut_layout))),
    )
    l_sub = lw_idx[:, None] * SPW + lsp_idx[None, :]
    l_shift = (lc_idx * 8)[None, None, None, :]
    # kload: [dword q of the token (head dims 8q..8q+7), token n] matching
    # rhs_layout (k_width 16): a lane holds dwords 2c, 2c+1 of a token.
    col_layout: gl.constexpr = gl.SliceLayout(0, qk_linear_layout)
    row_layout: gl.constexpr = gl.SliceLayout(1, qk_linear_layout)

    vw_idx = gl.arange(
        0,
        NW,
        layout=gl.SliceLayout(1, gl.SliceLayout(2, gl.SliceLayout(3, vload_layout))),
    )
    vsp_idx = gl.arange(
        0,
        SPW,
        layout=gl.SliceLayout(0, gl.SliceLayout(2, gl.SliceLayout(3, vload_layout))),
    )
    vt_idx = gl.arange(
        0,
        16,
        layout=gl.SliceLayout(0, gl.SliceLayout(1, gl.SliceLayout(3, vload_layout))),
    )
    vg_idx = gl.arange(
        0,
        16,
        layout=gl.SliceLayout(0, gl.SliceLayout(1, gl.SliceLayout(2, vload_layout))),
    )
    v_sub = vw_idx[:, None] * SPW + vsp_idx[None, :]  # sub-page within the partition
    v_tok_dw = vt_idx[None, None, :, None] * 16 + vg_idx[None, None, None, :]

    kqp_idx = gl.arange(0, 8, layout=gl.SliceLayout(1, gl.SliceLayout(2, kload_layout)))
    kn_idx = gl.arange(
        0, PART, layout=gl.SliceLayout(0, gl.SliceLayout(2, kload_layout))
    )
    kh_idx = gl.arange(0, 2, layout=gl.SliceLayout(0, gl.SliceLayout(1, kload_layout)))
    k_dw = (kqp_idx * 2)[:, None, None] + kh_idx[None, None, :]
    k_shift = ((kqp_idx // 2) * 8)[:, None, None]

    context_length = gl.load(context_lengths_ptr + seq)
    num_splits = gl.num_programs(2)
    bt_off = seq * stride_bt_seq
    # Split s owns partitions s, s + S, ...: its first table entries do not
    # depend on the context length, so they load alongside it (clamped into the
    # row; entries past the valid ones are replaced by entry 0 below).
    k_ent0 = gl.load(
        block_tables_ptr + bt_off + gl.minimum(split * NB + kn_idx // 16, bt_cols - 1)
    )
    v_ent0 = gl.load(
        block_tables_ptr + bt_off + gl.minimum(split * NB + v_sub, bt_cols - 1)
    )
    l_ent0 = gl.load(
        block_tables_ptr + bt_off + gl.minimum(split * NB + l_sub, bt_cols - 1)
    )
    ent_first = gl.load(block_tables_ptr + bt_off)
    qg = gl.arange(0, GROUP, layout=gl.SliceLayout(1, lhs_layout))
    qd = gl.arange(0, HEAD, layout=gl.SliceLayout(0, lhs_layout))
    q = gl.load(
        query_ptr + seq * stride_q_seq + qg[:, None] * stride_q_group + qd[None, :],
        mask=(qg < query_group_size)[:, None],
        other=0.0,
    )
    num_parts = gl.cdiv(context_length, PART)
    max_nb = gl.cdiv(context_length, 16)

    g_offs = gl.arange(0, GROUP, layout=gl.SliceLayout(1, blocked_query_layout))
    d_offs = gl.arange(0, HEAD, layout=gl.SliceLayout(0, blocked_query_layout))
    qrow_mask = g_offs < query_group_size
    row_mask = gl.convert_layout(qrow_mask, row_layout)
    pv_row_mask = gl.convert_layout(row_mask, gl.SliceLayout(1, mfma_layout))
    out_g = gl.arange(0, GROUP, layout=gl.SliceLayout(1, mfma_layout))
    out_d = gl.arange(0, HEAD, layout=gl.SliceLayout(0, mfma_layout))

    max_logits = gl.full((GROUP,), float("-inf"), dtype=gl.float32, layout=row_layout)
    exp_sums = gl.full((GROUP,), 0.0, dtype=gl.float32, layout=row_layout)
    acc3 = gl.zeros((NW, GROUP, HEAD), dtype=gl.float32, layout=pv3_layout)

    if ONE_SHOT:
        out_offsets = (
            seq * stride_out_seq + g_offs[:, None] * stride_out_group + d_offs[None, :]
        )
        out_mask = qrow_mask[:, None]
    else:
        out_offsets = (
            seq * stride_out_seq
            + split * stride_out_part
            + out_g[:, None] * stride_out_group
            + out_d[None, :]
        )
        out_mask = pv_row_mask[:, None]
        ml_offsets = (
            seq * stride_ml_seq
            + split * stride_ml_part
            + gl.arange(0, GROUP, layout=row_layout)
        )

    if split >= num_parts:
        if not ONE_SHOT:
            gl.store(max_logits_ptr + ml_offsets, max_logits, mask=row_mask)
            gl.store(exp_sums_ptr + ml_offsets, exp_sums, mask=row_mask)
            gl.store(
                output_ptr + out_offsets,
                gl.zeros((GROUP, HEAD), dtype=gl.bfloat16, layout=mfma_layout),
                mask=out_mask,
            )
        return

    # Q straight into the QK lhs operand layout (16 contiguous head dims per
    # lane: two 16-byte loads), no shared-memory staging.
    k_ent = gl.where(split * NB + kn_idx // 16 < max_nb, k_ent0, ent_first)
    v_ent = gl.where(split * NB + v_sub < max_nb, v_ent0, ent_first)
    l_ent = gl.where(split * NB + l_sub < max_nb, l_ent0, ent_first)
    # head dims 8g + (0, 2, 4, 6, 1, 3, 5, 7): the K decode's byte order (QK is
    # invariant to a common permutation of the head dim); register-local.
    q = gl.reshape(
        gl.permute(gl.reshape(q, [GROUP, 16, 4, 2]), [0, 1, 3, 2]), [GROUP, HEAD]
    )
    query_op = gl.convert_layout(q.to(COMPUTE_TYPE), lhs_layout)
    probs_shared = gl.allocate_shared_memory(
        COMPUTE_TYPE, [NW, GROUP, TPW], shared_p3_layout
    )
    col_idx = gl.arange(0, PART, layout=col_layout)

    for p in range(split, num_parts, num_splits):
        kbase, krow = _entry_bases(k_ent, stride_kv_block, NUM_KV_HEADS)
        k_tok = kbase // 4 + krow * 16 + (kn_idx % 16) * 16  # token's first dword
        k_raw = gl.load(kv_i32_ptr + (k_tok[None, :, None] + k_dw))
        k_w = gl.load(kv_i32_ptr + (kbase // 4 + 2048 + krow + kn_idx % 16))[
            None, :, None
        ]
        vbase, vrow = _entry_bases(v_ent, stride_kv_block, NUM_KV_HEADS)
        v_dw = vbase // 4 + V_OFF // 4 + vrow * 16
        v_raw = gl.load(kv_i32_ptr + (v_dw[:, :, None, None] + v_tok_dw))

        k_ent = _page_entries(
            block_tables_ptr, bt_off, (p + num_splits) * NB, kn_idx // 16, max_nb
        )
        lbase, lrow = _entry_bases(l_ent, stride_kv_block, NUM_KV_HEADS)
        l_w = gl.load(
            kv_i32_ptr
            + (
                (lbase // 4 + V_OFF // 4 + 2048 + lrow)[:, :, None, None]
                + lt_idx[None, None, :, None]
            )
        )
        v_ent = _page_entries(
            block_tables_ptr, bt_off, (p + num_splits) * NB, v_sub, max_nb
        )
        l_ent = _page_entries(
            block_tables_ptr, bt_off, (p + num_splits) * NB, l_sub, max_nb
        )
        # ---- K: decode, QK^T ----
        k_op = _decode_k(k_raw, k_w, k_shift)
        k_op = gl.convert_layout(k_op.to(COMPUTE_TYPE, bitcast=True), rhs_layout)
        scores = gl.amd.cdna3.mfma(
            query_op,
            k_op,
            gl.zeros((GROUP, PART), dtype=gl.float32, layout=mfma_layout),
        )
        # ---- V: decode (token-major) ----
        v_lo, v_hi = _lut_words(l_w, l_shift)
        v_op = _decode_v(
            v_raw, _spread4(v_lo, vload_layout), _spread4(v_hi, vload_layout), NW, SPW
        )
        v_op = gl.convert_layout(v_op.to(COMPUTE_TYPE, bitcast=True), v3_layout)

        k_scale = gl.convert_layout(_token_scale(gl.reshape(k_w, [PART])), col_layout)
        v_scale = gl.convert_layout(
            _token_scale(gl.reshape(gl.max(l_w, axis=3), [PART])), col_layout
        )

        scores = gl.convert_layout(scores, layout=qk_linear_layout)
        # log2 units: exp2 below needs no per-element multiply
        scores = ((softmax_scale * LOG2_E) * k_scale[None, :]) * scores
        if (p * PART + PART > context_length) | (query_group_size < GROUP):
            cols = p * PART + col_idx
            mask = row_mask[:, None] & (cols < context_length)[None, :]
            scores = gl.where(mask, scores, -3.4e38)
            v_scale = gl.where(cols < context_length, v_scale, 0.0)

        v_scale_b = tl.broadcast_to(v_scale[None, :], scores.shape)
        cur_max, v_scale_max_2d = gl.reduce(
            (scores, v_scale_b), axis=1, combine_fn=_fused_max_combine
        )
        v_scale_max = gl.max(v_scale_max_2d, axis=0)
        new_max = gl.maximum(max_logits, cur_max)
        alpha = tl.math.exp2(max_logits - new_max)
        probs = tl.math.exp2(scores - new_max[:, None])
        exp_sums = alpha * exp_sums + gl.sum(probs, axis=1)

        inv_vmax = hip_libdevice.fast_dividef(1.0, v_scale_max + 1e-30)
        probs = (v_scale_b * (FP8_MAX * inv_vmax)) * probs
        prob_scale = v_scale_max * (1.0 / FP8_MAX)
        p3 = gl.permute(gl.reshape(probs.to(COMPUTE_TYPE), [GROUP, NW, TPW]), [1, 0, 2])
        probs_shared.store(p3)
        probs_op = probs_shared.load(p3_layout)
        alpha3 = gl.convert_layout(
            alpha, gl.SliceLayout(0, gl.SliceLayout(2, pv3_layout))
        )
        acc3 = acc3 * alpha3[None, :, None]
        pv = gl.amd.cdna3.mfma(
            probs_op,
            v_op,
            gl.zeros((NW, GROUP, HEAD), dtype=gl.float32, layout=pv3_layout),
        )
        acc3 += prob_scale * pv
        max_logits = new_max

    safe = gl.where(exp_sums > 0, exp_sums, 1.0)
    # Sum the per-warp partial accumulators: one shared store, one load with the
    # warp dim in registers. Column n = 16 e + g holds head dim 8 g + e; the
    # shared view is read back as [w, g_row, e, g] so a thread sums 4 warps for
    # 8 consecutive head dims and stores them contiguously.
    red_layout: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[NW, 1, 1, 8],
        threads_per_warp=[1, 4, 16, 1],
        warps_per_cta=[1, NUM_WARPS, 1, 1],
        order=[3, 2, 1, 0],
    )
    acc_smem = gl.allocate_shared_memory(
        gl.float32,
        [NW, GROUP, 8, 16],
        gl.SwizzledSharedLayout(1, 1, 1, order=[3, 2, 1, 0]),
    )
    acc_smem.store(gl.reshape(acc3, [NW, GROUP, 8, 16]))
    acc4 = acc_smem.permute([0, 1, 3, 2]).load(red_layout)  # [w, g_row, g, e]
    accf = gl.sum(acc4, axis=0)  # [GROUP, 16 (g), 8 (e)]
    out_l: gl.constexpr = gl.SliceLayout(0, red_layout)
    inv = gl.convert_layout(1.0 / safe, gl.SliceLayout(1, gl.SliceLayout(2, out_l)))
    accf = (accf * inv[:, None, None]).to(gl.bfloat16)
    # e' = (0, 2, 4, 6, 1, 3, 5, 7) order from the V decode -> natural head dims
    accf = gl.convert_layout(
        gl.reshape(
            gl.permute(gl.reshape(accf, [GROUP, 16, 2, 4]), [0, 1, 3, 2]),
            [GROUP, 16, 8],
        ),
        out_l,
    )
    a_g3 = gl.arange(0, GROUP, layout=gl.SliceLayout(1, gl.SliceLayout(2, out_l)))
    a_q = gl.arange(0, 16, layout=gl.SliceLayout(0, gl.SliceLayout(2, out_l)))
    a_e = gl.arange(0, 8, layout=gl.SliceLayout(0, gl.SliceLayout(1, out_l)))
    a_g = a_g3[:, None, None]
    a_d = a_q[None, :, None] * 8 + a_e[None, None, :]
    if ONE_SHOT:
        a_off = seq * stride_out_seq + a_g * stride_out_group + a_d
    else:
        a_off = (
            seq * stride_out_seq
            + split * stride_out_part
            + a_g * stride_out_group
            + a_d
        )
        gl.store(max_logits_ptr + ml_offsets, max_logits, mask=row_mask)
        gl.store(exp_sums_ptr + ml_offsets, exp_sums, mask=row_mask)
    a_mask = a_g < query_group_size
    gl.store(output_ptr + a_off, accf, mask=a_mask)


@triton.jit
def _pa_decode_mxfp4_reduce_kernel(
    out_ptr,  # [num_seqs, G, D] bf16 (rows of the 3-D output view)
    max_logits_ptr,  # [num_seqs, splits, G] fp32
    exp_sums_ptr,  # [num_seqs, splits, G] fp32
    tmp_ptr,  # [num_seqs, splits, G, D] bf16
    stride_out_seq,
    stride_out_group,
    NSPLIT: tl.constexpr,
    NSPLIT_POW2: tl.constexpr,
    G: tl.constexpr,
    D: tl.constexpr,
):
    """Merge the per-split (max, sum, normalized partial output) of one
    (row, query head)."""
    s = tl.program_id(0).to(tl.int64)
    g = tl.program_id(1)
    sp = tl.arange(0, NSPLIT_POW2)
    d = tl.arange(0, D)
    valid = sp < NSPLIT
    ml_off = s * (NSPLIT * G) + sp * G + g
    m = tl.load(max_logits_ptr + ml_off, mask=valid, other=float("-inf"))
    e = tl.load(exp_sums_ptr + ml_off, mask=valid, other=0.0)
    m_all = tl.max(m, axis=0)
    m_all = tl.where(m_all == float("-inf"), 0.0, m_all)
    w = e * tl.exp2(m - m_all)  # maxima in log2 units
    w_sum = tl.sum(w, axis=0)
    w = w / tl.where(w_sum > 0, w_sum, 1.0)
    o = tl.load(
        tmp_ptr + ml_off[:, None] * D + d[None, :], mask=valid[:, None], other=0.0
    )
    r = tl.sum(o.to(tl.float32) * w[:, None], axis=0)
    tl.store(
        out_ptr + s * stride_out_seq + g * stride_out_group + d,
        r.to(out_ptr.dtype.element_ty),
    )


# ---------------------------------------------------------------------------
# Host launch path. Decode runs once per layer per step with persistent
# buffers, so the launcher keeps (a) the split partials in a per-device buffer
# grown on demand and (b) the resolved launches (compiled kernel + bound
# argument list) keyed by the tensors' pointers / shapes / strides, the
# scalars and the stream, and replays them without Triton's per-call argument
# binding. Shape checks run only when a launch is first resolved.
# ---------------------------------------------------------------------------
_PARTIALS: dict = {}
_LAUNCHES: dict = {}
_MAX_LAUNCHES = 4096
_SPLITS: dict = {}
SMALL1 = 2
SMALL2 = 8


def _partials(device, num_seqs, splits, group, head, dtype):
    n = num_seqs * splits * group
    key = (device, dtype)
    buf = _PARTIALS.get(key)
    if buf is None or buf[0].numel() < n:
        cap = max(n, 1 << 16)
        buf = (
            torch.empty(cap, dtype=torch.float32, device=device),
            torch.empty(cap, dtype=torch.float32, device=device),
            torch.empty(cap * head, dtype=dtype, device=device),
        )
        _PARTIALS[key] = buf
        _LAUNCHES.clear()  # cached launches point into the old buffers
    return (
        buf[0][:n].view(num_seqs, splits, group),
        buf[1][:n].view(num_seqs, splits, group),
        buf[2][: n * head].view(num_seqs, splits, group, head),
    )


class _Ptr:
    """Stand-in for a tensor argument of a cached launch: the HIP launcher only
    calls ``data_ptr()``."""

    __slots__ = ("ptr",)

    def __init__(self, ptr):
        self.ptr = ptr

    def data_ptr(self):
        return self.ptr


def _resolve_launch(kernel, grid, args, kwargs, stream):
    """Launch ``kernel[grid](*args, **kwargs)`` once and return a closure that
    replays exactly that launch (same compiled kernel and bound values)."""
    compiled = kernel[grid](*args, **kwargs)
    try:
        device = triton.runtime.driver.active.get_current_device()
        binder = kernel.device_caches[device][4]
        bound_args, _, _ = binder(*args, **kwargs)
        values = tuple(bound_args.values())
        g = tuple(grid) + (1,) * (3 - len(grid))
        meta = compiled.launch_metadata(g, stream, *values)
        # Keep only the pointers: holding the tensors would pin every buffer a
        # cached launch ever saw (activation-sized q / out included).
        values = tuple(
            _Ptr(v.data_ptr()) if isinstance(v, torch.Tensor) else v for v in values
        )
        knobs = triton.knobs.runtime

        def replay():
            compiled.run(
                g[0],
                g[1],
                g[2],
                stream,
                compiled.function,
                compiled.packed_metadata,
                meta,
                knobs.launch_enter_hook,
                knobs.launch_exit_hook,
                *values,
            )

        return replay
    except Exception:  # noqa: BLE001 - Triton internals moved: no replay
        return None


def _recommended_config(device, num_seqs):
    """-> (splits, partition_size, num_warps)."""
    """(splits, partition_size) for num_seqs = rows * kv heads; sparse rows hold
    <= 16 x 128 tokens. Small grids are latency bound: one 128-token partition
    per program. Larger grids take fewer splits, so a program streams several
    partitions (split s owns s, s + S, ...) with the next one's table entries
    in flight, and the split reduce shrinks. Measured on MI300X (graph replay,
    2048-token rows): B1 q1 16x128; B4 q4 8x128; B8/B16 q4 4x256; B32 q4 2x256."""
    key = (device, num_seqs)
    c = _SPLITS.get(key)
    if c is None:
        if num_seqs <= SMALL1:
            c = (64, 32, 1)
        elif num_seqs <= SMALL2:
            c = (32, 64, 2)
        elif num_seqs <= 16:
            c = (16, 128, 4)
        elif num_seqs <= 32:
            c = (8, 128, 4)
        elif num_seqs <= 128:
            c = (4, 256, 4)
        else:
            c = (2, 256, 4)
        _SPLITS[key] = c
    return c


def pa_decode_mxfp4_gluon(
    output: torch.Tensor,  # [rows, Hq, 128] bf16
    query: torch.Tensor,  # [rows, Hq, 128] bf16
    kv_cache: torch.Tensor,  # uint8 [num_blocks, 2H, 128, 68]
    context_lengths: torch.Tensor,  # int32 [rows * H]
    block_tables: torch.Tensor,  # int32 [rows * H, >= 128], page-16 entries
    softmax_scale: float,
    num_kv_heads: int,
    max_context_partition_num: int | None = None,
    num_warps: int | None = None,
    partition_size: int | None = None,
) -> torch.Tensor:
    """Sparse MXFP4 paged-attention decode; row ``r*H + h`` of ``block_tables``
    / ``context_lengths`` drives query row r, KV head h (heads h*G..h*G+G-1).
    ``max_context_partition_num`` (splits per row) defaults to AITER's
    recommendation; 1 writes ``output`` directly without a reduce.
    ``partition_size`` (tokens per program step, 128 or 256) defaults to 128
    for small grids (shorter per-program latency) and 256 otherwise."""
    stream = torch.cuda.current_stream(query.device).cuda_stream
    key = (
        output.data_ptr(),
        query.data_ptr(),
        query.shape[0],
        query.shape[1],
        kv_cache.data_ptr(),
        kv_cache.shape[0],
        kv_cache.stride(0),
        context_lengths.data_ptr(),
        block_tables.data_ptr(),
        block_tables.stride(0),
        block_tables.shape[1],
        float(softmax_scale),
        num_kv_heads,
        max_context_partition_num,
        num_warps,
        partition_size,
        stream,
    )
    hit = _LAUNCHES.get(key)
    if hit is not None:
        for launch in hit:
            launch()
        return output

    rows, num_q_heads, head = query.shape
    H = num_kv_heads
    group = num_q_heads // H
    if head != HEAD_DIM or group != 16 or num_q_heads != H * group:
        raise ValueError("pa_decode_mxfp4_gluon: head_dim 128 and GQA group 16 only")
    if kv_cache.dtype != torch.uint8 or kv_cache[0].numel() != 2 * H * PAGE_BYTES:
        raise ValueError("MXFP4 K/V cache must be uint8 [blocks, 2H, 128, 68]")
    if kv_cache.stride(-1) != 1 or kv_cache[0].is_contiguous() is False:
        raise ValueError("MXFP4 K/V cache blocks must be contiguous")
    if not (query.is_contiguous() and output.is_contiguous()):
        raise ValueError("query and output must be contiguous")
    if block_tables.dtype != torch.int32 or block_tables.stride(1) != 1:
        raise ValueError("block_tables must be int32 with contiguous rows")
    if context_lengths.dtype != torch.int32:
        raise ValueError("context_lengths must be int32")
    num_seqs = rows * H
    if num_seqs == 0:
        return output
    q_view = query.view(num_seqs, group, head)
    out_view = output.view(num_seqs, group, head)
    if max_context_partition_num:
        splits = max_context_partition_num
        part = partition_size or 256
        num_warps = num_warps or {32: 1, 64: 2}.get(part, 4)
    else:
        splits, part, nw = _recommended_config(query.device, num_seqs)
        if partition_size or num_warps:
            part = partition_size or 128
            nw = num_warps or {32: 1, 64: 2}.get(part, 4)
        num_warps = nw
    one_shot = splits <= 1
    if one_shot:
        exp_sums = max_logits = out_k = out_view  # exp/max unused
        so_seq, so_part, so_group = out_view.stride(0), 0, out_view.stride(1)
        ml_seq = ml_part = 0
    else:
        max_logits, exp_sums, out_k = _partials(
            query.device, num_seqs, splits, group, head, query.dtype
        )
        so_seq, so_part, so_group = out_k.stride(0), out_k.stride(1), out_k.stride(2)
        ml_seq, ml_part = exp_sums.stride(0), exp_sums.stride(1)
    launches = [
        _resolve_launch(
            pa_decode_mxfp4_kernel,
            (num_seqs, 1, splits),
            (
                exp_sums,
                max_logits,
                out_k,
                q_view,
                kv_cache,
                kv_cache.view(torch.int32),
                block_tables,
                context_lengths,
                float(softmax_scale),
                ml_seq,
                ml_part,
                so_seq,
                so_part,
                so_group,
                q_view.stride(0),
                q_view.stride(1),
                kv_cache.stride(0),
                block_tables.stride(0),
                group,
                block_tables.shape[1],
            ),
            {
                "NUM_KV_HEADS": H,
                "CDNA_VERSION": get_cdna_version(),
                "ONE_SHOT": one_shot,
                "NUM_WARPS": num_warps,
                "PART": part,
                "num_warps": num_warps,
            },
            stream,
        )
    ]
    if not one_shot:
        launches.append(
            _resolve_launch(
                _pa_decode_mxfp4_reduce_kernel,
                (num_seqs, group),
                (
                    out_view,
                    max_logits,
                    exp_sums,
                    out_k,
                    out_view.stride(0),
                    out_view.stride(1),
                ),
                {
                    "NSPLIT": splits,
                    "NSPLIT_POW2": triton.next_power_of_2(splits),
                    "G": group,
                    "D": head,
                    "num_warps": 1,
                },
                stream,
            )
        )
    if all(launch is not None for launch in launches):
        if len(_LAUNCHES) >= _MAX_LAUNCHES:
            _LAUNCHES.clear()
        _LAUNCHES[key] = tuple(launches)
    return output
