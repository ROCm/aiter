# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Shared flydsl emitters for the V4 compress-attn kernels.

Single source of truth for the FP8 ``group_fp8`` (V4 nm-asm) scatter tail used by
both the CSA single-kernel (``fused_compress_attn``) and the HCA 2-kernel
(``fused_compress_attn_hca``) paths, on wave64 (VEC=8) and wave32 (VEC=16). Keeping
it here avoids drift between the two kernels' fp8 entry layouts (they MUST stay
byte-identical so the V4 nm-asm sparse-attn reader sees one layout).
"""

from contextlib import contextmanager
from functools import lru_cache

from flydsl.expr import arith, range_constexpr, vector, buffer_ops
from flydsl.expr.arith import ArithValue, CmpFPredicate, CmpIPredicate
from flydsl.expr.typing import T
from flydsl._mlir import ir
from flydsl._mlir.dialects import rocdl, scf
from flydsl.runtime.device import get_rocm_arch

from .tensor_shim import _to_raw
from .quant_utils import emit_mx_e8m0_scale
from aiter.utility.mx_types import (
    MxDtypeInt as _MxD,
    MX_DEFAULT_ROUND_MODE as _MX_DEFAULT_MODE,
)


@contextmanager
def _if_then(if_op):
    """SCF IfOp then-region context manager. Auto-yields empty if missing."""
    with ir.InsertionPoint(if_op.then_block):
        try:
            yield if_op.then_block
        finally:
            blk = if_op.then_block
            if (not blk.operations) or not isinstance(blk.operations[-1], scf.YieldOp):
                scf.YieldOp([])


@lru_cache(maxsize=1)
def group_fp8_mx_dtype():
    """e4m3fnuz on gfx942 (MI300), OCP e4m3fn on gfx950+/gfx1250. Matches the C++
    kHwFp8E4m3Dtype selection so the e8m0 scale + fp8 bytes align across kernels."""
    return _MxD.FP8_E4M3_FNUZ if get_rocm_arch() == "gfx942" else _MxD.FP8_E4M3


def emit_group_fp8_nm_asm_scatter(
    *,
    normed_lane,  # list[VEC] f32: post-norm nope values (this lane's slice)
    rotated_lane,  # list[VEC] f32: post-RoPE pe values (this lane's slice)
    lane,  # i32: within-wave lane id (0..wave_width-1)
    is_rope_t,  # i1: lane >= ROPE_THREAD_LO
    cache_base,  # i32: physical_block*kcache_block_stride + slot*kcache_token_stride
    out_rsrc,  # kv_cache buffer resource (fp8 entry [.., entry])
    krope_base,  # i32: physical_block*krope_block_stride + slot*krope_token_stride
    krope_rsrc,  # k_rope_buff buffer resource (bf16 [.., RD])
    VEC,  # elems/lane (8 wave64, 16 wave32); must be a multiple of 4
    NOPE,  # nope_dim (head_dim - rope_head_dim)
    RTS,  # threads per quant group (= group_size // VEC)
    log2_rts,
    ROPE_THREAD_LO,  # first rope lane (= NOPE // VEC)
    wave_width,  # 64 (wave64) or 32 (wave32) -- shuffle_xor width
    vecVf32,  # T.vec(VEC, f32)
    fm_fast,  # arith.FastMathFlags.fast
):
    """Emit the FP8 nope (1xG e8m0) + inline duplicated e8m0 scale + bf16 rope->separate
    buffer scatter (V4 nm-asm layout). Byte-identical across CSA / HCA / wave32.

    Layout written into ``out_rsrc`` (fp8 entry, 1 byte/elem):
        [0:NOPE)               nope fp8
        [NOPE:NOPE+2*nGroups)  e8m0 group scale, each duplicated x2
    Rotated PE bf16 -> ``krope_rsrc`` at krope_base + (lane-ROPE_THREAD_LO)*VEC.
    """
    f32 = T.f32
    i32 = T.i32
    assert VEC % 4 == 0, f"group_fp8: VEC={VEC} must be a multiple of 4"
    c0f = arith.constant(0.0, type=f32)
    c_neg_uf = arith.constant(-(2.0**-8), type=f32)
    c_zero_i32 = arith.constant(0, type=i32)
    c_one_i32 = arith.constant(1, type=i32)

    # group-amax of |normed| over the RTS-thread group (shuffle_xor within wave)
    amax_g = _to_raw(arith.constant(0.0, type=f32))
    for i in range_constexpr(VEC):
        nv = arith.subf(c0f, normed_lane[i])
        av = arith.maximumf(normed_lane[i], nv)
        amax_g = arith.maximumf(amax_g, av)
    for sh in range_constexpr(log2_rts):
        off = RTS >> (sh + 1)
        peer = _to_raw(ArithValue(amax_g).shuffle_xor(off, wave_width))
        amax_g = arith.maximumf(amax_g, peer)
    e8m0 = emit_mx_e8m0_scale(amax_g, mode=_MX_DEFAULT_MODE, dtype=group_fp8_mx_dtype())
    quant_exp = arith.constant(254, type=i32) - e8m0
    inv_scale = (quant_exp << arith.constant(23, type=i32)).bitcast(f32)

    # -- nope lanes: scaled fp8 + group-leader dup e8m0 byte --
    is_nope = arith.cmpi(
        CmpIPredicate.slt, _to_raw(lane), arith.constant(ROPE_THREAD_LO, type=i32)
    )
    _if_nope = scf.IfOp(is_nope)
    with _if_then(_if_nope):
        safe = []
        for i in range_constexpr(VEC):
            sv = arith.MulFOp(normed_lane[i], inv_scale, fastmath=fm_fast).result
            # e4m3fnuz -0->+0 clamp: small negatives -> +0 (cvt returns NaN otherwise)
            is_tn = arith.andi(
                arith.cmpf(CmpFPredicate.OLT, sv, c0f),
                arith.cmpf(CmpFPredicate.OGT, sv, c_neg_uf),
            )
            safe.append(arith.select(is_tn, c0f, sv))
        # pack VEC fp8 -> VEC/4 dwords (2 cvt_pk_fp8 per dword)
        dwords = []
        for d in range_constexpr(VEC // 4):
            pk = arith.constant(0, type=i32)
            pk = rocdl.cvt_pk_fp8_f32(i32, safe[4 * d + 0], safe[4 * d + 1], pk, 0)
            pk = rocdl.cvt_pk_fp8_f32(i32, safe[4 * d + 2], safe[4 * d + 3], pk, 1)
            dwords.append(pk)
        nope_off = ArithValue(cache_base) + ArithValue(lane) * arith.constant(
            VEC, type=i32
        )
        store_vec = vector.from_elements(T.vec(VEC // 4, i32), dwords)
        buffer_ops.buffer_store(
            store_vec, out_rsrc, _to_raw(nope_off), offset_is_bytes=True
        )
        group_id = ArithValue(lane) >> arith.constant(log2_rts, type=i32)
        lane_in_group = ArithValue(lane) & arith.constant(RTS - 1, type=i32)
        is_leader = arith.cmpi(CmpIPredicate.eq, _to_raw(lane_in_group), c_zero_i32)
        _if_leader = scf.IfOp(is_leader)
        with _if_then(_if_leader):
            e8m0_i8 = arith.TruncIOp(T.i8, e8m0).result
            sc_off = (
                ArithValue(cache_base)
                + arith.constant(NOPE, type=i32)
                + ArithValue(group_id) * arith.constant(2, type=i32)
            )
            buffer_ops.buffer_store(e8m0_i8, out_rsrc, _to_raw(sc_off))
            buffer_ops.buffer_store(
                e8m0_i8, out_rsrc, _to_raw(ArithValue(sc_off) + c_one_i32)
            )

    # -- rope lanes: rotated bf16 -> separate k_rope_buff --
    _if_rope_q = scf.IfOp(is_rope_t)
    with _if_then(_if_rope_q):
        rope_rel = ArithValue(lane) - arith.constant(ROPE_THREAD_LO, type=i32)
        krope_off = ArithValue(krope_base) + ArithValue(rope_rel) * arith.constant(
            VEC, type=i32
        )
        rope_f32 = vector.from_elements(vecVf32, rotated_lane)
        rope_bf16 = rope_f32.truncf(T.vec(VEC, T.bf16))
        dwr = (VEC + 1) // 2
        rope_i32 = vector.bitcast(T.vec(dwr, i32), rope_bf16)
        krope_off_dw = ArithValue(krope_off) >> c_one_i32
        if dwr <= 4:
            # VEC<=8 (wave64): single dwordx{dwr} store.
            buffer_ops.buffer_store(rope_i32, krope_rsrc, _to_raw(krope_off_dw))
        else:
            # VEC=16 (wave32) -> dwr=8: no dwordx8 store; split into 2x dwordx4.
            c4_i32 = arith.constant(4, type=i32)
            lo = vector.extract_strided_slice(
                T.vec(4, i32), rope_i32, offsets=[0], sizes=[4], strides=[1]
            )
            hi = vector.extract_strided_slice(
                T.vec(4, i32), rope_i32, offsets=[4], sizes=[4], strides=[1]
            )
            buffer_ops.buffer_store(lo, krope_rsrc, _to_raw(krope_off_dw))
            buffer_ops.buffer_store(
                hi, krope_rsrc, _to_raw(ArithValue(krope_off_dw) + c4_i32)
            )
