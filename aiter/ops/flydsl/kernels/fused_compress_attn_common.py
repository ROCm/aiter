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

import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import rocdl, scf
from flydsl._mlir.dialects import vector as vector_dialect
from flydsl.expr import arith, range_constexpr
from flydsl.expr.arith import ArithValue
from flydsl.expr.typing import T
from flydsl.runtime.device import get_rocm_arch

from aiter.utility.mx_types import (
    MX_DEFAULT_ROUND_MODE as _MX_DEFAULT_MODE,
)
from aiter.utility.mx_types import (
    MxDtypeInt as _MxD,
)

from .quant_utils import emit_mx_e8m0_scale
from .tensor_shim import _to_raw


def _raw_buffer_store(data, rsrc, byte_offset):
    """Store ``data`` (scalar or vector ir.Value) through the AMD buffer resource
    ``rsrc`` (``!llvm.ptr<8>``) at ``byte_offset`` (i32, in BYTES).

    This is the hardware boundary: the emitter's ``rsrc`` is a raw buffer
    descriptor handed in by callers (built from the fx layout API's
    ``get_buffer_rsrc``), so the store goes straight to the ROCDL raw-buffer
    intrinsic -- the tensor-oriented ``fx.copy`` surface can't take a bare rsrc.
    Mirrors the old vendored byte-offset store path exactly (soffset 0, aux 0).
    """
    # BOUNDARY: raw ROCDL raw-buffer store (bare rsrc, no fx.copy surface).
    off = _to_raw(byte_offset)
    if not (isinstance(off.type, ir.IntegerType) and off.type.width == 32):
        off = (
            fx.Int64(off).to(fx.Int32)
            if isinstance(off.type, ir.IndexType)
            else fx.Int32(off)
        ).ir_value()
    zero = fx.Int32(0).ir_value()
    rocdl.RawPtrBufferStoreOp(_to_raw(data), _to_raw(rsrc), off, zero, zero)


@contextmanager
def _if_then(if_op):
    """SCF IfOp then-region context manager. Auto-yields empty if missing."""
    # BOUNDARY: scf.IfOp then-region for predicated side-effect stores.
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
    vecVf32,  # T.vec(VEC, f32) (unused; kept for signature back-compat)
    fm_fast,  # kept for signature back-compat (fast_fp_math is a launcher hint)
):
    """Emit the FP8 nope (1xG e8m0) + inline duplicated e8m0 scale + bf16 rope->separate
    buffer scatter (V4 nm-asm layout). Byte-identical across CSA / HCA / wave32.

    Layout written into ``out_rsrc`` (fp8 entry, 1 byte/elem):
        [0:NOPE)               nope fp8
        [NOPE:NOPE+2*nGroups)  e8m0 group scale, each duplicated x2
    Rotated PE bf16 -> ``krope_rsrc`` at krope_base + (lane-ROPE_THREAD_LO)*VEC.
    """
    i32 = T.i32
    assert VEC % 4 == 0, f"group_fp8: VEC={VEC} must be a multiple of 4"
    c0f = fx.Float32(0.0)
    c_neg_uf = fx.Float32(-(2.0**-8))

    lane = fx.Int32(lane)

    # group-amax of |normed| over the RTS-thread group (shuffle_xor within wave)
    amax_g = fx.Float32(0.0)
    for i in range_constexpr(VEC):
        # subf kept raw (non-fast) to match HEAD -- ambient fast_fp_math would
        # otherwise flip the flag and change the ISA.
        nv = fx.Float32(arith.subf(_to_raw(c0f), _to_raw(normed_lane[i])))
        av = fx.Float32(normed_lane[i]).maximumf(nv)
        amax_g = amax_g.maximumf(av)
    for sh in range_constexpr(log2_rts):
        off = RTS >> (sh + 1)
        peer = amax_g.shuffle_xor(off, wave_width)
        amax_g = amax_g.maximumf(peer)
    # BOUNDARY: emit_mx_e8m0_scale is a raw IR builder (uses .bitcast(T.i32));
    # feed it a raw ArithValue.
    e8m0 = emit_mx_e8m0_scale(
        ArithValue(amax_g.ir_value()),
        mode=_MX_DEFAULT_MODE,
        dtype=group_fp8_mx_dtype(),
    )
    quant_exp = fx.Int32(254) - fx.Int32(e8m0)
    inv_scale = (quant_exp << fx.Int32(23)).bitcast(fx.Float32)

    # -- nope lanes: scaled fp8 + group-leader dup e8m0 byte --
    is_nope = lane < fx.Int32(ROPE_THREAD_LO)
    # BOUNDARY: scf.IfOp predicated side-effect store region.
    _if_nope = scf.IfOp(is_nope.ir_value())
    with _if_then(_if_nope):
        safe = []
        for i in range_constexpr(VEC):
            sv = fx.Float32(normed_lane[i]) * inv_scale
            # e4m3fnuz -0->+0 clamp: small negatives -> +0 (cvt returns NaN otherwise)
            is_tn = (sv < c0f) & (sv > c_neg_uf)
            safe.append(is_tn.select(c0f, sv))
        # pack VEC fp8 -> VEC/4 dwords (2 cvt_pk_fp8 per dword)
        dwords = []
        for d in range_constexpr(VEC // 4):
            # BOUNDARY: hand-packed rocdl.cvt_pk_fp8_f32 (raw operands).
            pk = fx.Int32(0).ir_value()
            pk = rocdl.cvt_pk_fp8_f32(
                i32, _to_raw(safe[4 * d + 0]), _to_raw(safe[4 * d + 1]), pk, 0
            )
            pk = rocdl.cvt_pk_fp8_f32(
                i32, _to_raw(safe[4 * d + 2]), _to_raw(safe[4 * d + 3]), pk, 1
            )
            dwords.append(pk)
        nope_off = fx.Int32(cache_base) + lane * fx.Int32(VEC)
        store_vec = fx.Vector.from_elements(dwords, fx.Int32)
        # nope_off is already a BYTE offset (fp8 entry, 1 byte/elem).
        _raw_buffer_store(store_vec, out_rsrc, nope_off)
        group_id = lane >> fx.Int32(log2_rts)
        lane_in_group = lane & fx.Int32(RTS - 1)
        is_leader = lane_in_group == fx.Int32(0)
        # BOUNDARY: scf.IfOp predicated side-effect store region.
        _if_leader = scf.IfOp(is_leader.ir_value())
        with _if_then(_if_leader):
            e8m0_i8 = fx.Int32(e8m0).to(fx.Int8)
            sc_off = fx.Int32(cache_base) + fx.Int32(NOPE) + group_id * fx.Int32(2)
            # i8 scale byte: element offset == byte offset.
            _raw_buffer_store(e8m0_i8, out_rsrc, sc_off)
            _raw_buffer_store(e8m0_i8, out_rsrc, sc_off + fx.Int32(1))

    # -- rope lanes: rotated bf16 -> separate k_rope_buff --
    # BOUNDARY: scf.IfOp predicated side-effect store region.
    _if_rope_q = scf.IfOp(_to_raw(is_rope_t))
    with _if_then(_if_rope_q):
        rope_rel = lane - fx.Int32(ROPE_THREAD_LO)
        krope_off = fx.Int32(krope_base) + rope_rel * fx.Int32(VEC)
        rope_f32 = fx.Vector.from_elements(rotated_lane, fx.Float32)
        rope_bf16 = rope_f32.truncf(T.vec(VEC, T.bf16))
        dwr = (VEC + 1) // 2
        rope_i32 = fx.Vector(rope_bf16).bitcast(fx.Int32)
        krope_off_dw = krope_off >> fx.Int32(1)
        # i32 dword store: byte offset == dword index * 4.
        krope_byte = krope_off_dw * fx.Int32(4)
        if dwr <= 4:
            # VEC<=8 (wave64): single dwordx{dwr} store.
            _raw_buffer_store(rope_i32, krope_rsrc, krope_byte)
        else:
            # VEC=16 (wave32) -> dwr=8: no dwordx8 store; split into 2x dwordx4.
            # BOUNDARY: raw vector.extract_strided_slice (no fx sub-vector op).
            lo = vector_dialect.extract_strided_slice(
                T.vec(4, i32), _to_raw(rope_i32), offsets=[0], sizes=[4], strides=[1]
            )
            hi = vector_dialect.extract_strided_slice(
                T.vec(4, i32), _to_raw(rope_i32), offsets=[4], sizes=[4], strides=[1]
            )
            _raw_buffer_store(lo, krope_rsrc, krope_byte)
            # +4 dwords == +16 bytes.
            _raw_buffer_store(hi, krope_rsrc, krope_byte + fx.Int32(16))
