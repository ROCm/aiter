# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Shared flydsl emitters for the V4 compress-attn kernels.

Single source of truth for the FP8 ``group_fp8`` (V4 nm-asm) scatter tail used by
both the CSA single-kernel (``fused_compress_attn``) and the HCA 2-kernel
(``fused_compress_attn_hca``) paths, on wave64 (VEC=8) and wave32 (VEC=16). Keeping
it here avoids drift between the two kernels' fp8 entry layouts (they MUST stay
byte-identical so the V4 nm-asm sparse-attn reader sees one layout).
"""

from functools import lru_cache

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import range_constexpr
from flydsl.expr.typing import T
from flydsl.runtime.device import get_rocm_arch

from aiter.utility.mx_types import (
    MX_DEFAULT_ROUND_MODE as _MX_DEFAULT_MODE,
)
from aiter.utility.mx_types import (
    MxDtypeInt as _MxD,
)

from .quant_utils import emit_mx_e8m0_scale

_AS_GLOBAL = fx.AddressSpace.Global


@lru_cache(maxsize=1)
def group_fp8_mx_dtype():
    """e4m3fnuz on gfx942 (MI300), OCP e4m3fn on gfx950+/gfx1250. Matches the C++
    kHwFp8E4m3Dtype selection so the e8m0 scale + fp8 bytes align across kernels."""
    return _MxD.FP8_E4M3_FNUZ if get_rocm_arch() == "gfx942" else _MxD.FP8_E4M3


def state_slot_byte_offset(slot, slot_stride_f32_elems):
    """`slot * slot_stride` in bytes, computed in 64 bits.

    All four compress-attn kernels (CSA and HCA, wave64 and gfx1250) rebase
    their `kv_state` / `score_state` buffer descriptors onto the program's own
    slot instead of carrying the slot term in the load offset, so a caller may
    hand any of them a strided per-request arena view. A buffer offset is a
    32-bit BYTE offset, so one
    descriptor reaches at most 4 GiB from its base — which the slot term
    alone can exceed once the caller's state tensor is a view whose slot
    stride is a whole per-request arena entry rather than the field's own
    size. `get_element_ptr` adds this in 64-bit pointer arithmetic, so only
    the multiply needs widening, and the remaining offset covers one entry.
    """
    slot_i64 = fx.Int64(fx.Int32(slot))
    stride_i64 = fx.Int64(fx.Int32(slot_stride_f32_elems))
    return (slot_i64 * stride_i64 * fx.Int64(4)).ir_value()  # x sizeof(f32)


def block_base_bytes_i64(physical_block, block_stride, elem_bytes: int = 1):
    """Byte offset of one physical block, computed in 64 bits.

    A buffer descriptor addresses through a 32-bit offset inside a 32-bit
    ``num_records`` window, so one fixed base cannot reach past 4 GiB: the
    offset wraps at 2 GiB and the hardware drops the access at 4 GiB, both
    silently. Shifting the descriptor's BASE per block instead leaves the
    32-bit field holding one block's worth of offset, which no layout makes big.

    How soon that mattered scales with how many layers a block spans.
    DeepSeek-V4's envelope layout puts every layer's rows for one block
    together, so consecutive blocks sit 708 KB apart and block 3,031 wraps; the
    layer-major predecessor still wrapped at block 65,536 for a CSA layer, well
    inside a pool that routinely holds 150,000.

    ``block_stride`` is in elements of the cache's own dtype and may be either
    a runtime value (the caller passed the tensor's stride) or a Python int
    (the packed fp4/preshuffle layouts derive it from compile-time shape
    constants). ``elem_bytes`` converts: 1 for the fp8/uint8 entry caches, 2
    for a bf16 one, 4 for the fp32 per-token scale.
    """
    blk_i64 = fx.Int64(fx.Int32(physical_block))
    if isinstance(block_stride, int):
        stride_i64 = fx.Int64(block_stride)
    else:
        stride_i64 = fx.Int64(fx.Int32(block_stride))
    base = blk_i64 * stride_i64
    if elem_bytes != 1:
        base = base * fx.Int64(elem_bytes)
    return base.ir_value()


def _global_ptr(base_i64, byte_off, elem_ir_type, align):
    """Direct global pointer at ``base_i64 + byte_off``, element type ``elem_ir_type``.

    The block/slot term is already folded into ``base_i64`` (a 64-bit address),
    so no 32-bit V# offset window and no 4 GiB wrap -- the reason the emitter
    doesn't need a buffer resource. ``byte_off`` (i32, the lane's in-entry
    position) is widened and folded into the base too, so the returned pointer
    addresses the store site directly at element 0. The store is plan-bounded
    and in-bounds (no OOB / sentinel), so a raw pointer is correct.

    ``elem_ir_type`` is a *scalar* type (i8 for the e8m0 byte, i32 for the fp8 /
    rope dword vectors); a wider vector value is stored whole through it, with
    ``align`` = the store's byte width so the wide store stays aligned.
    """
    addr = base_i64 + fx.Int64(byte_off)
    pt = fx.PointerType.get(elem_ir_type, address_space=_AS_GLOBAL, alignment=align)
    return fx.inttoptr(pt, addr)


def emit_group_fp8_nm_asm_scatter(
    *,
    normed_lane,  # list[VEC] f32: post-norm nope values (this lane's slice)
    rotated_lane,  # list[VEC] f32: post-RoPE pe values (this lane's slice)
    lane,  # i32: within-wave lane id (0..wave_width-1)
    is_rope_t,  # i1: lane >= ROPE_THREAD_LO
    cache_base,  # i32: byte offset of this token's fp8 entry within its block
    out_base_i64,  # i64: kv_cache base addr + physical_block*block_stride (bytes)
    krope_base,  # i32: byte offset of this token's rope row within its block
    krope_base_i64,  # i64: k_rope_buff base addr + physical_block*block_stride (bytes)
    VEC,  # elems/lane (8 wave64, 16 wave32); must be a multiple of 4
    NOPE,  # nope_dim (head_dim - rope_head_dim)
    RTS,  # threads per quant group (= group_size // VEC)
    log2_rts,
    ROPE_THREAD_LO,  # first rope lane (= NOPE // VEC)
    wave_width,  # 64 (wave64) or 32 (wave32) -- shuffle_xor width
):
    """Emit the FP8 nope (1xG e8m0) + inline duplicated e8m0 scale + bf16 rope->separate
    buffer scatter (V4 nm-asm layout). Byte-identical across CSA / HCA / wave32.

    Stores go through direct global pointers built from ``out_base_i64`` /
    ``krope_base_i64`` (the cache/rope base address with the per-block byte
    offset already folded in). Layout written into kv_cache (fp8 entry,
    1 byte/elem):
        [0:NOPE)               nope fp8
        [NOPE:NOPE+2*nGroups)  e8m0 group scale, each duplicated x2
    Rotated PE bf16 -> k_rope_buff at krope_base + (lane-ROPE_THREAD_LO)*VEC.
    """
    i32 = T.i32
    assert VEC % 4 == 0, f"group_fp8: VEC={VEC} must be a multiple of 4"
    lane = fx.Int32(lane)
    is_rope_t = fx.Boolean(is_rope_t)
    cache_base = fx.Int32(cache_base)
    krope_base = fx.Int32(krope_base)
    normed = [fx.Float32(v) for v in normed_lane]
    c0f = fx.Float32(0.0)
    c_neg_uf = fx.Float32(-(2.0**-8))

    # group-amax of |normed| over the RTS-thread group (shuffle_xor within wave)
    amax_g = fx.Float32(0.0)
    for i in range_constexpr(VEC):
        amax_g = fx.maximumf(amax_g, fx.maximumf(normed[i], c0f - normed[i]))
    for sh in range_constexpr(log2_rts):
        off = RTS >> (sh + 1)
        amax_g = fx.maximumf(amax_g, amax_g.shuffle_xor(off, wave_width))
    e8m0 = emit_mx_e8m0_scale(
        amax_g.ir_value(), mode=_MX_DEFAULT_MODE, dtype=group_fp8_mx_dtype()
    )
    quant_exp = fx.Int32(254) - fx.Int32(e8m0)
    inv_scale = (quant_exp << fx.Int32(23)).bitcast(fx.Float32)

    # -- nope lanes: scaled fp8 + group-leader dup e8m0 byte --
    # Guarded bodies live in local @flyc.jit helpers so a plain `if` lowers to
    # scf.if: the DSL's if-rewrite fires on a decorated body, not on this
    # imported emitter's own source (skill Sec.5).
    @flyc.jit
    def _nope_lanes():
        if lane < fx.Int32(ROPE_THREAD_LO):
            safe = []
            for i in range_constexpr(VEC):
                # inv_scale is fast-fp-math ambient -> `*` == the old
                # MulFOp(fastmath=fast) (byte-identical under _DEFAULT_COMPILE_HINTS).
                sv = normed[i] * inv_scale
                # e4m3fnuz -0->+0 clamp: small negatives -> +0 (cvt returns NaN otherwise)
                is_tn = (sv < c0f) & (sv > c_neg_uf)
                safe.append(is_tn.select(c0f, sv))
            # pack VEC fp8 -> VEC/4 dwords (2 cvt_pk_fp8 per dword)
            dwords = []
            for d in range_constexpr(VEC // 4):
                pk = fx.Int32(0).ir_value()
                pk = fx.rocdl.cvt_pk_fp8_f32(
                    i32, safe[4 * d + 0].ir_value(), safe[4 * d + 1].ir_value(), pk, 0
                )
                pk = fx.rocdl.cvt_pk_fp8_f32(
                    i32, safe[4 * d + 2].ir_value(), safe[4 * d + 3].ir_value(), pk, 1
                )
                dwords.append(fx.Int32(pk))
            nope_off = cache_base + lane * fx.Int32(VEC)
            store_vec = fx.Vector.from_elements(dwords, fx.Int32)
            # VEC fp8 bytes = VEC//4 i32 dwords at byte offset nope_off.
            _global_ptr(out_base_i64, nope_off, i32, VEC).store(store_vec)

            if (lane & fx.Int32(RTS - 1)) == fx.Int32(0):
                e8m0_i8 = fx.Int32(e8m0).to(fx.Int8)
                group_id = lane >> fx.Int32(log2_rts)
                sc_off = cache_base + fx.Int32(NOPE) + group_id * fx.Int32(2)
                # e8m0 duplicated x2: one i8 byte at sc_off and sc_off+1.
                sc_ptr = _global_ptr(out_base_i64, sc_off, T.i8, 1)
                sc_ptr[0] = e8m0_i8
                sc_ptr[1] = e8m0_i8

    _nope_lanes()

    # -- rope lanes: rotated bf16 -> separate k_rope_buff --
    @flyc.jit
    def _rope_lanes():
        if is_rope_t:
            rope_rel = lane - fx.Int32(ROPE_THREAD_LO)
            krope_off = krope_base + rope_rel * fx.Int32(VEC)  # bf16 elements
            rope_f32 = fx.Vector.from_elements(list(rotated_lane), fx.Float32)
            rope_bf16 = rope_f32.truncf(T.vec(VEC, T.bf16))
            dwr = (VEC + 1) // 2
            rope_i32 = rope_bf16.bitcast(fx.Int32)  # -> vec<dwr x i32>
            # bf16 element offset -> byte offset (x2). dwr i32 dwords per rope row.
            krope_byte = krope_off << fx.Int32(1)
            if dwr <= 4:
                # VEC<=8 (wave64): single dwordx{dwr} store.
                _global_ptr(krope_base_i64, krope_byte, i32, 4 * dwr).store(rope_i32)
            else:
                # VEC=16 (wave32) -> dwr=8: no dwordx8 store; split into 2x dwordx4.
                lo = fx.Vector.from_elements([rope_i32[k] for k in range(4)], fx.Int32)
                hi = fx.Vector.from_elements(
                    [rope_i32[k + 4] for k in range(4)], fx.Int32
                )
                _global_ptr(krope_base_i64, krope_byte, i32, 16).store(lo)
                _global_ptr(krope_base_i64, krope_byte + fx.Int32(16), i32, 16).store(
                    hi
                )

    _rope_lanes()
