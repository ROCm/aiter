# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Fused MoE route-map + MX quant + scatter-copy + scale-preshuffle (FlyDSL).

The grouped a8w4/fp4 MoE stage1 input prep is normally four kernels (see
``grouped_moe_gfx1250.py``):

    1. build_route_maps          route i -> grouped row (atomic argsort)
    2. per_1x32 MX quant         hidden(T, model_dim) -> payload + e8m0 scale
    3. scatter_copy_token        payload[token] -> grouped_payload[row]
    4. scatter_preshuffle_scale  scale[token]   -> grouped_scale[row] (WMMA layout)

This kernel fuses all four into one *warp-per-route* pass. Each warp owns one
route ``i = token*topk + k``:

    lane 0   : expert = topk_ids[i]; slot = atomicAdd(counter[expert], 1)
               grouped_row = expert_row_base[expert] + slot (masked: e*max_m,
               contiguous-M: starts[e]); topids_to_rows[i] = grouped_row
    broadcast slot (hence grouped_row) to the whole warp via readlane
    all lanes: quantize token's activation row directly into
               grouped_payload[grouped_row] (fp4 e2m1 or fp8 e4m3) and write the
               e8m0 block scales into grouped_scale in the preshuffled WMMA layout
               for grouped_row -- no per-token intermediates, no rows_to_tokens.

The quant math (per-1x32 E8M0 block scale + f32->e2m1) is shared with
``silu_and_mul_fq.py`` via ``quant_utils``. ``counter`` must be zero-initialised
before launch; after the run ``counter[expert] == masked_m[expert]``.

Layout / intra-warp mapping
---------------------------
``model_dim`` is processed in 32-element MX blocks. Each lane quantizes
``ELEMS_PER_LANE`` (=2) contiguous bf16 columns, so a block spans
``LANES_PER_MX_BLOCK`` (=16) lanes and a wavefront (32 on gfx1250 / 64 on gfx9xx)
covers ``wave_size // 16`` blocks at once. The per-block amax reduction is a
butterfly ``shuffle_xor`` over the block's 16 lanes; the lead lane of each block
(lane_in_block == 0) writes the single e8m0 scale byte.

Scale preshuffle (per grouped row, mirrors
``moe_scatter_copy_preshuffle_scale.py``): for a grouped row at within-expert
position ``slot`` in expert ``e`` and MX block ``mx_block`` (with
``scale_dword = mx_block // 4`` and ``byte_in_dword = mx_block % 4``)::

    scale_tile  = slot // (wmma_rep*16)
    wmma_row    = (slot % (wmma_rep*16)) // 16
    row_lane16  = slot % 16
    out_row     = scale_tile*16 + row_lane16
    dst_dword   = e*(max_m*scale_dwords_per_row)
                  + out_row*(scale_dwords_per_row*wmma_rep)
                  + scale_dword*wmma_rep + wmma_row
    dst_byte    = dst_dword*4 + byte_in_dword

Each warp writes only its own (valid) row; padding rows are never touched, which
matches the existing scatter-copy contract (the masked GEMM, bounded by
``masked_m``, never reads padding payload or scale).

Grid  : (ceil(numel / warps_per_block), 1, 1)   numel = token_num*topk
Block : (BLOCK_THREADS, 1, 1)
"""

from types import SimpleNamespace

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import arith, ptrtoint, range_constexpr, const_expr, rocdl, vector, gpu
from flydsl.expr.typing import T, Int32
from flydsl.expr.arith import ArithValue, CmpIPredicate
from flydsl.compiler.kernel_function import CompilationContext

from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm, scf
from flydsl.expr import buffer_ops
from flydsl.runtime.device import get_rocm_arch

from aiter.ops.flydsl.kernels.quant_utils import emit_f32_to_e2m1, emit_mx_e8m0_scale
from aiter.ops.flydsl.kernels.kernels_common import get_warp_size
from aiter.ops.flydsl.kernels.tensor_shim import ptr_rsrc, MOE_KERNARG_PRELOAD_COUNT

from aiter.utility.mx_types import (
    MxDtypeInt as _MxDtype,
    MX_DEFAULT_ROUND_MODE as _ROUND_MODE,
)

BLOCK_THREADS = 256
ELEMS_PER_LANE = 2  # bf16 columns each lane quantizes -> 1 fp4 byte / 2 fp8 bytes
LANES_PER_MX_BLOCK = 32 // ELEMS_PER_LANE  # 16 lanes cover one 32-element MX block

# Architectures with native scaled-pack f32->fp4/fp8 conversion
# (``v_cvt_scalef32_pk_{fp4,fp8}_f32``). On these the per-block pack folds the
# scale division in (one HW instruction, exact RNE); elsewhere we fall back to
# the portable path (SW e2m1 emitter for fp4 / ``v_cvt_pk_fp8_f32`` for fp8,
# both legal on gfx942 and gfx1250).
#
# NOTE: gfx1250 does *not* have these instructions -- the gfx950 (CDNA4)
# ``v_cvt_scalef32_pk_{fp4,fp8}_f32`` intrinsics have no valid gfx1250 encoding,
# so selecting them on gfx1250 makes the AMDGPU backend abort with an MC
# "Invalid opcode!" assertion at compile time. gfx1250 therefore uses the same
# portable path as gfx942 (matches ``silu_and_mul_fq``).
_NATIVE_SCALED_CVT_ARCHS = ("gfx950",)

# gfx1250 has no 2-element ``v_cvt_scalef32_pk_{fp4,fp8}_f32`` (gfx950-only) but it
# *does* have the 8-element ``v_cvt_scalef32_pk8_{fp4,fp8}_bf16``: 8 bf16 -> packed
# fp4 (i32, 8 nibbles) / fp8 (v2i32, 8 e4m3 bytes), dividing by the e8m0 exponent
# carried in the f32 scale. We emit them via inline asm so they do not depend on
# the MLIR rocdl op lowering.
_PK8_BF16_ARCHS = ("gfx1250",)


def _arch_has_pk8(arch: str) -> bool:
    return arch.startswith(_PK8_BF16_ARCHS)


def _cvt_scalef32_pk8_fp4_bf16(src_v8bf16, scale_f32, *, i32_ty):
    """Native gfx1250 scaled 8x bf16 -> packed fp4 (i32, 8 nibbles).

    ``src_v8bf16`` is a ``vector<8xbf16>`` ir.Value, ``scale_f32`` an f32 whose
    exponent is the e8m0 block scale (value 2^(e8m0-127)); the HW divides each
    input by it and round-to-nearest-even packs the 8 fp4 nibbles into i32.
    """
    return llvm.inline_asm(
        i32_ty,
        [_raw(src_v8bf16), _raw(scale_f32)],
        "v_cvt_scalef32_pk8_fp4_bf16 $0, $1, $2",
        "=v,v,v",
        has_side_effects=False,
    )


def _cvt_scalef32_pk8_fp8_bf16(src_v8bf16, scale_f32, *, v2i32_ty):
    """Native gfx1250 scaled 8x bf16 -> packed fp8 e4m3 (v2i32, 8 bytes).

    Same scale contract as the fp4 form; the HW divides each input by the f32
    scale's exponent and RNE-packs 8 fp8 e4m3 bytes into a 2xi32 vector.
    """
    return llvm.inline_asm(
        v2i32_ty,
        [_raw(src_v8bf16), _raw(scale_f32)],
        "v_cvt_scalef32_pk8_fp8_bf16 $0, $1, $2",
        "=v,v,v",
        has_side_effects=False,
    )


def _raw(value):
    """Unwrap a DSL Numeric to a raw ir.Value (rocdl ops need raw operands)."""
    return value.ir_value() if hasattr(value, "ir_value") else value


def _arch_has_native_scaled_cvt(arch: str) -> bool:
    return arch.startswith(_NATIVE_SCALED_CVT_ARCHS)


def _quant_layout(feat_dim: int, quant_mode: str, wmma_rep: int) -> SimpleNamespace:
    """Shared per-block quant + e8m0 scale-preshuffle geometry.

    ``feat_dim`` is the activation feature dim being quantized along K
    (``model_dim`` for the stage1 route kernel, ``inter_dim`` for the stage2
    grouped kernel). The payload conversion path (gfx1250 native pk8 fp4 /
    gfx950 native pk2 / portable) and the FP8 e8m0 dtype are chosen here from the
    current arch -- not caller arguments. Returns a namespace consumed by both
    builders and by ``_emit_quant_block_loop``.
    """
    if quant_mode not in ("fp4", "fp8"):
        raise NotImplementedError(
            f"quant_mode={quant_mode!r} unsupported (expected 'fp4' or 'fp8')."
        )
    assert feat_dim % 32 == 0, f"feat_dim ({feat_dim}) must be a multiple of 32"
    assert wmma_rep >= 1, "wmma_rep must be >= 1"

    is_fp8 = quant_mode == "fp8"
    arch = str(get_rocm_arch())
    use_native = _arch_has_native_scaled_cvt(arch)
    # gfx1250: native 8-wide pk8 convert for both fp4 and fp8 -> 8 elems/lane
    # (4 lanes per 32-elem MX block) instead of the 2 elems/lane (16 lanes) the
    # SW/pk2 paths use.
    use_pk8 = _arch_has_pk8(arch)
    elems_per_lane = 8 if use_pk8 else ELEMS_PER_LANE
    lanes_per_mx_block = 32 // elems_per_lane

    if is_fp8:
        mx_dtype = (
            _MxDtype.FP8_E4M3_FNUZ if arch.startswith("gfx942") else _MxDtype.FP8_E4M3
        )
        payload_bytes_per_row = feat_dim
        payload_bytes_per_block = 32
        payload_bytes_per_lane = elems_per_lane
    else:
        mx_dtype = _MxDtype.FP4_E2M1
        payload_bytes_per_row = feat_dim // 2
        payload_bytes_per_block = 16
        payload_bytes_per_lane = elems_per_lane // 2

    wave_size = get_warp_size()
    assert BLOCK_THREADS % wave_size == 0
    warps_per_block = BLOCK_THREADS // wave_size
    mx_blocks_per_wave_iter = wave_size // lanes_per_mx_block

    mx_blocks_per_row = feat_dim // 32  # == scale_bytes_per_row (1 e8m0/block)
    scale_bytes_per_row = mx_blocks_per_row
    assert (
        scale_bytes_per_row % 4 == 0
    ), "feat_dim//32 must be a multiple of 4 (dword-packed scale)"
    scale_dwords_per_row = scale_bytes_per_row // 4
    rows_per_tile = wmma_rep * 16
    dst_scale_dwords_per_row = scale_dwords_per_row * wmma_rep
    block_iters = (
        mx_blocks_per_row + mx_blocks_per_wave_iter - 1
    ) // mx_blocks_per_wave_iter

    # Butterfly reduction distances within one MX block (16 lanes for the 2-elem
    # paths, 4 lanes for pk8).
    amax_shuffle_dists = []
    dist = 1
    while dist < lanes_per_mx_block:
        amax_shuffle_dists.append(dist)
        dist *= 2

    native_tag = "pk8" if use_pk8 else ("nat" if use_native else "sw")
    return SimpleNamespace(
        is_fp8=is_fp8,
        arch=arch,
        use_native=use_native,
        use_pk8=use_pk8,
        elems_per_lane=elems_per_lane,
        lanes_per_mx_block=lanes_per_mx_block,
        mx_dtype=mx_dtype,
        payload_bytes_per_row=payload_bytes_per_row,
        payload_bytes_per_block=payload_bytes_per_block,
        payload_bytes_per_lane=payload_bytes_per_lane,
        wave_size=wave_size,
        warps_per_block=warps_per_block,
        mx_blocks_per_wave_iter=mx_blocks_per_wave_iter,
        mx_blocks_per_row=mx_blocks_per_row,
        scale_bytes_per_row=scale_bytes_per_row,
        scale_dwords_per_row=scale_dwords_per_row,
        rows_per_tile=rows_per_tile,
        dst_scale_dwords_per_row=dst_scale_dwords_per_row,
        block_iters=block_iters,
        amax_shuffle_dists=amax_shuffle_dists,
        native_tag=native_tag,
    )


def _emit_quant_block_loop(c: SimpleNamespace) -> None:
    """Emit one warp's per-MX-block quant + e8m0 scale-preshuffle loop.

    ``c`` carries the layout flags, SSA constants/types, buffer resources, the
    source base (``feat_elem_base``), the intra-warp mapping (``block_in_wave``,
    ``lane_in_block``, ``is_block_lead``), and ``c.dests``: a list of destination
    namespaces, each with ``payload_row_byte_base`` and ``scale_row_dword_base``.
    The current callers pass a single destination; keeping this as a list lets a
    future caller experiment with multi-destination scattering without changing
    the quant math. Shared verbatim by both stage1 and stage2; only the preamble
    that computes ``c.dests`` differs.
    """
    i32 = c.i32
    f32 = c.f32
    mx_group_base = getattr(c, "mx_group_base", None)
    if mx_group_base is None:
        mx_group_base = arith.constant(0, type=i32)
    for it in range_constexpr(c.block_iters):
        # MX block (along K) this lane works on this iteration.
        mx_block = (mx_group_base + arith.constant(it, type=i32)) * arith.constant(
            c.mx_blocks_per_wave_iter, type=i32
        ) + c.block_in_wave
        block_in_range = arith.cmpi(
            CmpIPredicate.ult,
            mx_block,
            arith.constant(c.mx_blocks_per_row, type=i32),
        )
        _if_block = scf.IfOp(block_in_range)
        with ir.InsertionPoint(_if_block.then_block):
            if const_expr(c.use_pk8):
                # gfx1250 native pk8: 8 contiguous bf16 cols this lane.
                # col_base = mx_block*32 + lane_in_block*8.
                col_base = (
                    mx_block * arith.constant(32, type=i32)
                    + c.lane_in_block * c.c_elems_per_lane
                )
                # 2 bf16/dword -> 4 dwords; one aligned dwordx4 = 8 bf16.
                hidden_dword = (c.feat_elem_base + col_base) >> c.c1_i32
                dwords4 = buffer_ops.buffer_load(
                    c.hidden_rsrc, hidden_dword, vec_width=4, dtype=i32
                )
                vec8_bf16_ty = T.vec(8, T.bf16)
                vec8_f32_ty = T.vec(8, f32)
                bf16x8 = vector.bitcast(vec8_bf16_ty, dwords4)
                f32x8 = bf16x8.extf(vec8_f32_ty)

                # per-block amax over this lane's 8 elems, then a butterfly
                # shuffle_xor across the block's 4 lanes.
                block_amax = c.c0_f32
                for j in range_constexpr(8):
                    xj = vector.extract(f32x8, static_position=[j], dynamic_position=[])
                    absj = llvm.call_intrinsic(f32, "llvm.fabs.f32", [xj], [], [])
                    block_amax = arith.maximumf(block_amax, absj)
                for dist in c.amax_shuffle_dists:
                    peer_amax = block_amax.shuffle_xor(
                        arith.constant(dist, type=i32), c.c_wave
                    )
                    block_amax = arith.maximumf(block_amax, peer_amax)

                e8m0_scale = emit_mx_e8m0_scale(
                    block_amax, mode=_ROUND_MODE, dtype=c.mx_dtype
                )
                # scale 2^(e8m0-127); the HW divides each input by its exponent
                # and RNE-packs the 8 outputs (fp4: i32 / fp8: v2i32).
                block_scale_f32 = (ArithValue(e8m0_scale) << c.c23_i32).bitcast(f32)
                if const_expr(c.is_fp8):
                    payload_val = _cvt_scalef32_pk8_fp8_bf16(
                        bf16x8, block_scale_f32, v2i32_ty=T.vec(2, i32)
                    )  # v2i32 = 8 fp8 e4m3 bytes
                else:
                    payload_val = _cvt_scalef32_pk8_fp4_bf16(
                        bf16x8, block_scale_f32, i32_ty=i32
                    )  # i32 = 4 fp4x2 bytes
            else:
                # two contiguous bf16 columns: col_base = mx_block*32 + lane_in_block*2
                col_base = (
                    mx_block * arith.constant(32, type=i32)
                    + c.lane_in_block * c.c_elems_per_lane
                )
                hidden_dword = (c.feat_elem_base + col_base) >> c.c1_i32  # 2 bf16/dword

                dword_raw = buffer_ops.buffer_load(
                    c.hidden_rsrc, hidden_dword, vec_width=1, dtype=i32
                )
                vec1_i32_ty = T.vec(1, i32)
                vec2_bf16_ty = T.vec(ELEMS_PER_LANE, T.bf16)
                vec2_f32_ty = T.vec(ELEMS_PER_LANE, f32)
                bf16_pair = vector.bitcast(
                    vec2_bf16_ty, vector.from_elements(vec1_i32_ty, [dword_raw])
                )
                f32_pair = bf16_pair.extf(vec2_f32_ty)
                x0 = vector.extract(f32_pair, static_position=[0], dynamic_position=[])
                x1 = vector.extract(f32_pair, static_position=[1], dynamic_position=[])

                # per-block amax: max over this lane's 2 elems, then a butterfly
                # shuffle_xor across the block's 16 lanes.
                abs0 = llvm.call_intrinsic(f32, "llvm.fabs.f32", [x0], [], [])
                abs1 = llvm.call_intrinsic(f32, "llvm.fabs.f32", [x1], [], [])
                block_amax = arith.maximumf(c.c0_f32, arith.maximumf(abs0, abs1))
                for dist in c.amax_shuffle_dists:
                    peer_amax = block_amax.shuffle_xor(
                        arith.constant(dist, type=i32), c.c_wave
                    )
                    block_amax = arith.maximumf(block_amax, peer_amax)

                e8m0_scale = emit_mx_e8m0_scale(
                    block_amax, mode=_ROUND_MODE, dtype=c.mx_dtype
                )

                # Forward block scale 2^(e8m0-127) = bitcast(e8m0<<23); the native
                # scalef32 ops divide by its *exponent part*. The portable path
                # multiplies by the reciprocal 2^(127-e8m0) then converts.
                if const_expr(c.is_fp8):
                    if const_expr(c.use_native):
                        block_scale_f32 = (ArithValue(e8m0_scale) << c.c23_i32).bitcast(
                            f32
                        )
                        packed = rocdl.cvt_scalef32_pk_fp8_f32(
                            i32,
                            _raw(c.c0_i32),
                            _raw(x0),
                            _raw(x1),
                            _raw(block_scale_f32),
                            0,
                        )
                    else:
                        recip_scale = ((c.c254_i32 - e8m0_scale) << c.c23_i32).bitcast(
                            f32
                        )
                        scaled0 = ArithValue(x0) * recip_scale
                        scaled1 = ArithValue(x1) * recip_scale
                        # v_cvt_pk_fp8_f32: 2 f32 -> 2 fp8 bytes in word 0.
                        packed = rocdl.cvt_pk_fp8_f32(
                            i32, scaled0, scaled1, c.c0_i32, 0
                        )
                    payload_val = arith.trunci(T.i16, ArithValue(packed))  # 2 fp8 B
                else:
                    if const_expr(c.use_native):
                        block_scale_f32 = (ArithValue(e8m0_scale) << c.c23_i32).bitcast(
                            f32
                        )
                        packed = rocdl.cvt_scalef32_pk_fp4_f32(
                            i32,
                            _raw(c.c0_i32),
                            _raw(x0),
                            _raw(x1),
                            _raw(block_scale_f32),
                            0,
                        )
                        payload_val = arith.trunci(T.i8, ArithValue(packed))
                    else:
                        recip_scale = ((c.c254_i32 - e8m0_scale) << c.c23_i32).bitcast(
                            f32
                        )
                        nib0 = emit_f32_to_e2m1(ArithValue(x0) * recip_scale)
                        nib1 = emit_f32_to_e2m1(ArithValue(x1) * recip_scale)
                        packed_byte = ArithValue(nib0) | (ArithValue(nib1) << c.c4_i32)
                        payload_val = arith.trunci(T.i8, packed_byte)  # 1 fp4x2 B

            # One quant result (payload_val + e8m0_scale) is written to every
            # destination row in ``c.dests``. Current kernels pass one destination;
            # the list keeps the store side generic without changing quant math.
            # The block-scale's dword/byte position depends only on ``mx_block``.
            scale_dword = arith.divui(mx_block, c.c4_i32)
            byte_in_dword = mx_block - scale_dword * c.c4_i32
            e8m0_byte = arith.trunci(T.i8, e8m0_scale)
            for dst in c.dests:
                # payload byte offset within grouped_payload. offset_is_bytes=True
                # so the i8 (fp4) / i16 (fp8) / i32 (pk8) store does not rescale
                # this already-byte offset by the data element size.
                payload_byte_off = (
                    dst.payload_row_byte_base
                    + mx_block * c.c_payload_bytes_per_block
                    + c.lane_in_block * c.c_payload_bytes_per_lane
                )
                buffer_ops.buffer_store(
                    payload_val, c.payload_rsrc, payload_byte_off, offset_is_bytes=True
                )

                # one e8m0 byte per block, written by the block's lead lane.
                _if_lead = scf.IfOp(c.is_block_lead)
                with ir.InsertionPoint(_if_lead.then_block):
                    dst_scale_dword = (
                        dst.scale_row_dword_base + scale_dword * c.c_wmma_rep
                    )
                    dst_scale_byte = dst_scale_dword * c.c4_i32 + byte_in_dword
                    buffer_ops.buffer_store(e8m0_byte, c.scale_rsrc, dst_scale_byte)
                    scf.YieldOp([])
            scf.YieldOp([])


def _emit_quant_one_k_group(c: SimpleNamespace, mx_group) -> None:
    """Emit exactly one K group of MX blocks for one warp.

    ``mx_group`` indexes groups of ``mx_blocks_per_wave_iter`` MX blocks. This is
    the K-split entry point used by small-token specializations; the original
    full-row callers keep using ``_emit_quant_block_loop``.
    """
    d = vars(c).copy()
    d["block_iters"] = 1
    d["mx_group_base"] = mx_group
    _emit_quant_block_loop(SimpleNamespace(**d))


def build_moe_fused_route_quant_scatter_module(
    model_dim: int,
    topk: int,
    wmma_rep: int,
    quant_mode: str = "fp4",
    *,
    use_expert_row_base: bool = True,
    max_m: int = 0,
):
    """Return a JIT launcher for the fused route+quant+scatter+preshuffle kernel.

    Parameters
    ----------
    model_dim : int    activation feature dim (must be a multiple of 32).
    topk : int         routes per token (token = route // topk).
    wmma_rep : int     ``warp_tile_m // 16`` (scale preshuffle tile geometry).
    quant_mode : str   ``"fp4"`` (MXFP4 e2m1, payload model_dim//2) or ``"fp8"``
                       (MXFP8 e4m3, payload model_dim).

    The payload conversion path (native ``v_cvt_scalef32_pk_{fp4,fp8}_f32`` vs the
    portable path) is chosen here from the current arch -- gfx950/gfx1250 use the
    native scaled-convert instruction, everything else (incl. gfx942) uses the
    portable path. ``topk_ids`` is int32 (the router's only output dtype).

    The destination row for each route is ``row_base + slot`` and both the
    payload and the e8m0 scale are indexed by that *global* row, so the same
    kernel serves either output layout:

      * masked     : ``row_base = expert*max_m``          -> buffer (E, max_m)
      * contiguous : ``expert_row_base[e] = starts[e]``   -> buffer (1, contiguous_m)
                     (DeepGEMM contiguous-M; ``starts`` is the tile_m-aligned
                     exclusive prefix sum of masked_m)

    Every base must be a multiple of ``wmma_rep*16`` (both forms are) so the
    preshuffle tiling stays consistent.

    Launcher signature::

        (topk_ids, counter, topids_to_rows, hidden, grouped_payload, grouped_scale,
         expert_row_base, numel, grid_blocks, stream=...)

      topk_ids        : (numel,)               int32  flattened expert ids
      counter         : (E,)                   int32  per-expert counter, init 0
                        (== masked_m[expert] after the run)
      topids_to_rows  : (numel,)               int32  out: route -> grouped row
      hidden          : (token_num*model_dim,) bf16   flat activations
      grouped_payload : (n_rows*payload_bytes_per_row,) uint8  out: MX payload
                        (payload_bytes_per_row = model_dim//2 fp4 / model_dim fp8;
                        n_rows = E*max_m masked / contiguous_m contiguous)
      grouped_scale   : (n_rows*(model_dim//32),) uint8  out: preshuffled e8m0
      expert_row_base : (E,)                   int32  per-expert dst row base;
                       ignored for masked layout
    """
    if not use_expert_row_base and max_m <= 0:
        raise ValueError("max_m must be positive when expert_row_base is fused")
    L = _quant_layout(model_dim, quant_mode, wmma_rep)
    is_fp8 = L.is_fp8
    use_native = L.use_native
    use_pk8 = L.use_pk8
    elems_per_lane = L.elems_per_lane
    lanes_per_mx_block = L.lanes_per_mx_block
    mx_dtype = L.mx_dtype
    payload_bytes_per_row = L.payload_bytes_per_row
    payload_bytes_per_block = L.payload_bytes_per_block
    payload_bytes_per_lane = L.payload_bytes_per_lane
    wave_size = L.wave_size
    warps_per_block = L.warps_per_block
    mx_blocks_per_wave_iter = L.mx_blocks_per_wave_iter
    mx_blocks_per_row = L.mx_blocks_per_row
    rows_per_tile = L.rows_per_tile
    dst_scale_dwords_per_row = L.dst_scale_dwords_per_row
    block_iters = L.block_iters
    amax_shuffle_dists = L.amax_shuffle_dists
    topk_is_pow2 = topk > 0 and (topk & (topk - 1)) == 0
    topk_shift = topk.bit_length() - 1 if topk_is_pow2 else 0

    base_tag = "baseptr" if use_expert_row_base else f"basem{max_m}"
    module_name = (
        f"moe_fused_route_quant_scatter_md{model_dim}_tk{topk}_r{wmma_rep}"
        f"_{quant_mode}_{L.native_tag}_{base_tag}"
    )

    @flyc.kernel(name=module_name)
    def fused_kernel(
        topk_ids: fx.Pointer,  # (numel,) int32
        counter: fx.Pointer,  # (E,) int32, init 0
        topids_to_rows: fx.Pointer,  # (numel,) int32 out
        hidden: fx.Pointer,  # (token_num*model_dim,) bf16
        grouped_payload: fx.Pointer,  # (n_rows*payload_bytes_per_row,) uint8 out
        grouped_scale: fx.Pointer,  # preshuffled e8m0 out
        expert_row_base: fx.Pointer,  # (E,) int32 per-expert dst row base
        numel: Int32,
    ):
        i32 = T.i32
        f32 = T.f32

        c0_i32 = arith.constant(0, type=i32)
        c1_i32 = arith.constant(1, type=i32)
        c4_i32 = arith.constant(4, type=i32)
        c16_i32 = arith.constant(16, type=i32)
        c23_i32 = arith.constant(23, type=i32)
        c254_i32 = arith.constant(254, type=i32)
        c0_f32 = arith.constant(0.0, type=f32)

        c_wave = arith.constant(wave_size, type=i32)
        c_topk = arith.constant(topk, type=i32)
        c_topk_shift = arith.constant(topk_shift, type=i32)
        c_model_dim = arith.constant(model_dim, type=i32)
        c_payload_bytes_per_row = arith.constant(payload_bytes_per_row, type=i32)
        c_payload_bytes_per_block = arith.constant(payload_bytes_per_block, type=i32)
        c_payload_bytes_per_lane = arith.constant(payload_bytes_per_lane, type=i32)
        c_dst_scale_dwords_per_row = arith.constant(dst_scale_dwords_per_row, type=i32)
        c_wmma_rep = arith.constant(wmma_rep, type=i32)
        c_rows_per_tile = arith.constant(rows_per_tile, type=i32)
        c_lanes_per_block = arith.constant(lanes_per_mx_block, type=i32)
        c_elems_per_lane = arith.constant(elems_per_lane, type=i32)
        c_max_m = arith.constant(max_m, type=i32)

        tid = ArithValue(fx.thread_idx.x)
        bid = ArithValue(fx.block_idx.x)

        warp_in_block = tid // c_wave
        lane = tid - warp_in_block * c_wave  # tid % wave_size
        route = bid * arith.constant(warps_per_block, type=i32) + warp_in_block

        route_in_range = arith.cmpi(CmpIPredicate.ult, route, ArithValue(numel))
        _if_route = scf.IfOp(route_in_range)
        with ir.InsertionPoint(_if_route.then_block):
            topk_ids_rsrc = ptr_rsrc(topk_ids)
            # expert id for this route (uniform across the warp)
            expert = ArithValue(
                buffer_ops.buffer_load(topk_ids_rsrc, route, vec_width=1, dtype=i32)
            )

            # Lane 0 claims the within-expert slot via atomicAdd, then broadcasts
            # it to the warp. Single-token pow2 cases use the dedicated st_ksplit
            # kernel, so the generic path does not need a runtime numel==topk
            # branch here.
            slot_on_lane0 = arith.constant(0, type=i32)
            if lane == 0:
                counter_base = arith.index_cast(T.index, ptrtoint(counter))
                expert_idx = arith.index_cast(T.index, expert)
                counter_addr = fx.Index(counter_base) + fx.Index(expert_idx) * fx.Index(
                    4
                )
                counter_ptr = buffer_ops.create_llvm_ptr(counter_addr, address_space=1)
                counter_ptr = (
                    counter_ptr._value
                    if hasattr(counter_ptr, "_value")
                    else counter_ptr
                )
                slot_on_lane0 = ArithValue(
                    llvm.AtomicRMWOp(
                        llvm.AtomicBinOp.add,
                        counter_ptr,
                        arith.constant(1, type=i32),
                        llvm.AtomicOrdering.monotonic,
                        syncscope="agent",
                        alignment=4,
                    ).result
                )
            # readlane needs raw ir.Value operands in this FlyDSL build (the
            # /workspace/FlyDSL example's auto-unwrap + T.i32() are a newer API).
            slot = ArithValue(rocdl.readlane(i32, _raw(slot_on_lane0), _raw(c0_i32)))
            slot = ArithValue(slot)

            # Destination row = per-expert base + within-expert slot. Masked
            # layout fuses the former Python-side arange(E)*max_m into this
            # kernel; contiguous-M still loads starts[e] from expert_row_base.
            if const_expr(use_expert_row_base):
                erb_rsrc = ptr_rsrc(expert_row_base)
                row_base = ArithValue(
                    buffer_ops.buffer_load(erb_rsrc, expert, vec_width=1, dtype=i32)
                )
            else:
                row_base = expert * c_max_m
            grouped_row = slot + row_base
            if const_expr(topk_is_pow2):
                token = route >> c_topk_shift
            else:
                token = arith.divui(route, c_topk)

            # topids_to_rows[route] = grouped_row (lane 0 only; warp-uniform value)
            if lane == 0:
                topids_to_rows_rsrc = ptr_rsrc(topids_to_rows)
                buffer_ops.buffer_store(grouped_row, topids_to_rows_rsrc, route)

            # --- per-row scale-preshuffle geometry (uniform; from the *global*
            #     grouped_row so the same math serves both output layouts). Since
            #     every expert base is a multiple of rows_per_tile, tiling by the
            #     global row reproduces the per-expert byte layout exactly. ---
            scale_tile = arith.divui(grouped_row, c_rows_per_tile)
            row_in_tile = grouped_row - scale_tile * c_rows_per_tile
            wmma_row = arith.divui(row_in_tile, c16_i32)
            row_lane16 = row_in_tile - wmma_row * c16_i32
            out_row = scale_tile * c16_i32 + row_lane16
            # dst dword base for out_row; scale_dword*wmma_rep added per block.
            scale_row_dword_base = out_row * c_dst_scale_dwords_per_row + wmma_row

            payload_row_byte_base = grouped_row * c_payload_bytes_per_row
            hidden_elem_base = token * c_model_dim  # bf16 element base for this token

            hidden_rsrc = ptr_rsrc(hidden)
            payload_rsrc = ptr_rsrc(grouped_payload)
            scale_rsrc = ptr_rsrc(grouped_scale)

            # this lane's position inside its MX block group
            block_in_wave = arith.divui(lane, c_lanes_per_block)
            lane_in_block = lane - block_in_wave * c_lanes_per_block
            is_block_lead = arith.cmpi(CmpIPredicate.eq, lane_in_block, c0_i32)

            c = SimpleNamespace(
                i32=i32,
                f32=f32,
                block_iters=block_iters,
                mx_blocks_per_wave_iter=mx_blocks_per_wave_iter,
                mx_blocks_per_row=mx_blocks_per_row,
                amax_shuffle_dists=amax_shuffle_dists,
                is_fp8=is_fp8,
                use_native=use_native,
                use_pk8=use_pk8,
                mx_dtype=mx_dtype,
                c0_i32=c0_i32,
                c1_i32=c1_i32,
                c4_i32=c4_i32,
                c23_i32=c23_i32,
                c254_i32=c254_i32,
                c0_f32=c0_f32,
                c_wave=c_wave,
                c_elems_per_lane=c_elems_per_lane,
                c_payload_bytes_per_block=c_payload_bytes_per_block,
                c_payload_bytes_per_lane=c_payload_bytes_per_lane,
                c_wmma_rep=c_wmma_rep,
                block_in_wave=block_in_wave,
                lane_in_block=lane_in_block,
                is_block_lead=is_block_lead,
                dests=[
                    SimpleNamespace(
                        payload_row_byte_base=payload_row_byte_base,
                        scale_row_dword_base=scale_row_dword_base,
                    )
                ],
                feat_elem_base=hidden_elem_base,
                hidden_rsrc=hidden_rsrc,
                payload_rsrc=payload_rsrc,
                scale_rsrc=scale_rsrc,
            )
            _emit_quant_block_loop(c)
            scf.YieldOp([])

    @flyc.jit
    def launch_fused(
        topk_ids: fx.Pointer,
        counter: fx.Pointer,
        topids_to_rows: fx.Pointer,
        hidden: fx.Pointer,
        grouped_payload: fx.Pointer,
        grouped_scale: fx.Pointer,
        expert_row_base: fx.Pointer,
        numel: fx.Int32,
        grid_blocks: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            pass

        grid_x = arith.index_cast(T.index, grid_blocks)
        fused_kernel(
            topk_ids,
            counter,
            topids_to_rows,
            hidden,
            grouped_payload,
            grouped_scale,
            expert_row_base,
            numel,
        ).launch(
            grid=(grid_x, 1, 1),
            block=(BLOCK_THREADS, 1, 1),
            stream=stream,
        )

    launch_fused.compile_hints = {
        "llvm_options": {
            "amdgpu-kernarg-preload": True,
            "amdgpu-kernarg-preload-count": MOE_KERNARG_PRELOAD_COUNT,
        },
    }

    return launch_fused


def build_moe_fused_route_quant_scatter_st_ksplit_module(
    model_dim: int,
    topk: int,
    wmma_rep: int,
    quant_mode: str = "fp4",
    *,
    use_expert_row_base: bool = True,
    max_m: int = 0,
):
    """Single-token K-split stage1 route+quant+scatter+preshuffle kernel.

    The generic stage1 kernel keeps one warp per route and lets that warp loop
    over the full K row. For token_num == 1 this under-fills the GPU (only topk
    active warps), so this specialization launches one warp per (route, K-group)
    while keeping route-level parallelism. Production topk routing yields distinct
    expert indices per token, so each route's within-expert slot is 0 and the
    per-expert counter value is 1; this avoids both the route-counter atomic and
    the repeated topk scan in every K group.
    """
    if topk <= 0 or (topk & (topk - 1)) != 0:
        raise NotImplementedError(
            "single-token K-split currently requires power-of-two topk"
        )
    if not use_expert_row_base and max_m <= 0:
        raise ValueError("max_m must be positive when expert_row_base is fused")

    L = _quant_layout(model_dim, quant_mode, wmma_rep)
    if not L.use_pk8:
        raise NotImplementedError(
            "single-token K-split is currently enabled only for gfx1250 pk8"
        )

    is_fp8 = L.is_fp8
    use_native = L.use_native
    use_pk8 = L.use_pk8
    elems_per_lane = L.elems_per_lane
    lanes_per_mx_block = L.lanes_per_mx_block
    mx_dtype = L.mx_dtype
    payload_bytes_per_row = L.payload_bytes_per_row
    payload_bytes_per_block = L.payload_bytes_per_block
    payload_bytes_per_lane = L.payload_bytes_per_lane
    wave_size = L.wave_size
    # This specialization is used only for token_num == 1. Use exactly one warp
    # per route in the block so topk < 8 does not leave half of a 256-thread block
    # idle (e.g. topk=4 on wave32 -> 128-thread blocks).
    warps_per_block = topk
    block_threads = topk * wave_size
    mx_blocks_per_wave_iter = L.mx_blocks_per_wave_iter
    mx_blocks_per_row = L.mx_blocks_per_row
    rows_per_tile = L.rows_per_tile
    dst_scale_dwords_per_row = L.dst_scale_dwords_per_row
    amax_shuffle_dists = L.amax_shuffle_dists
    k_groups = L.block_iters

    base_tag = "baseptr" if use_expert_row_base else f"basem{max_m}"
    module_name = (
        f"moe_fused_route_quant_scatter_stks_md{model_dim}_tk{topk}_r{wmma_rep}"
        f"_{quant_mode}_{L.native_tag}_{base_tag}"
    )

    @flyc.kernel(name=module_name)
    def fused_kernel(
        topk_ids: fx.Pointer,  # (topk,) int32
        counter: fx.Pointer,  # (E,) int32 out
        topids_to_rows: fx.Pointer,  # (topk,) int32 out
        hidden: fx.Pointer,  # (model_dim,) bf16
        grouped_payload: fx.Pointer,  # out
        grouped_scale: fx.Pointer,  # out
        expert_row_base: fx.Pointer,  # (E,) int32
        numel: Int32,  # == topk for this specialization
    ):
        i32 = T.i32
        f32 = T.f32

        c0_i32 = arith.constant(0, type=i32)
        c1_i32 = arith.constant(1, type=i32)
        c4_i32 = arith.constant(4, type=i32)
        c16_i32 = arith.constant(16, type=i32)
        c23_i32 = arith.constant(23, type=i32)
        c254_i32 = arith.constant(254, type=i32)
        c0_f32 = arith.constant(0.0, type=f32)

        c_wave = arith.constant(wave_size, type=i32)
        c_payload_bytes_per_row = arith.constant(payload_bytes_per_row, type=i32)
        c_payload_bytes_per_block = arith.constant(payload_bytes_per_block, type=i32)
        c_payload_bytes_per_lane = arith.constant(payload_bytes_per_lane, type=i32)
        c_dst_scale_dwords_per_row = arith.constant(dst_scale_dwords_per_row, type=i32)
        c_wmma_rep = arith.constant(wmma_rep, type=i32)
        c_rows_per_tile = arith.constant(rows_per_tile, type=i32)
        c_lanes_per_block = arith.constant(lanes_per_mx_block, type=i32)
        c_elems_per_lane = arith.constant(elems_per_lane, type=i32)
        c_max_m = arith.constant(max_m, type=i32)

        tid = ArithValue(fx.thread_idx.x)
        bid = ArithValue(fx.block_idx.x)
        k_group = ArithValue(fx.block_idx.y)

        warp_in_block = tid // c_wave
        lane = tid - warp_in_block * c_wave
        route = bid * arith.constant(warps_per_block, type=i32) + warp_in_block

        route_in_range = arith.cmpi(CmpIPredicate.ult, route, ArithValue(numel))
        _if_route = scf.IfOp(route_in_range)
        with ir.InsertionPoint(_if_route.then_block):
            topk_ids_rsrc = ptr_rsrc(topk_ids)
            expert = ArithValue(
                buffer_ops.buffer_load(topk_ids_rsrc, route, vec_width=1, dtype=i32)
            )

            # torch.topk over experts returns distinct expert indices for one
            # token. Therefore each selected expert receives exactly one route:
            # slot=0, counter[expert]=1. This is the key small-token fast path;
            # the generic kernel remains available for non-single-token cases.
            slot = ArithValue(c0_i32)

            is_lane0 = arith.cmpi(CmpIPredicate.eq, lane, c0_i32)
            is_k0 = arith.cmpi(CmpIPredicate.eq, k_group, c0_i32)
            is_lane0_k0 = arith.andi(is_lane0, is_k0)
            if is_lane0_k0:
                counter_rsrc = ptr_rsrc(counter)
                buffer_ops.buffer_store(c1_i32, counter_rsrc, expert)

            if const_expr(use_expert_row_base):
                erb_rsrc = ptr_rsrc(expert_row_base)
                row_base = ArithValue(
                    buffer_ops.buffer_load(erb_rsrc, expert, vec_width=1, dtype=i32)
                )
            else:
                row_base = expert * c_max_m
            grouped_row = slot + row_base

            if is_lane0_k0:
                topids_to_rows_rsrc = ptr_rsrc(topids_to_rows)
                buffer_ops.buffer_store(grouped_row, topids_to_rows_rsrc, route)

            scale_tile = arith.divui(grouped_row, c_rows_per_tile)
            row_in_tile = grouped_row - scale_tile * c_rows_per_tile
            wmma_row = arith.divui(row_in_tile, c16_i32)
            row_lane16 = row_in_tile - wmma_row * c16_i32
            out_row = scale_tile * c16_i32 + row_lane16
            scale_row_dword_base = out_row * c_dst_scale_dwords_per_row + wmma_row
            payload_row_byte_base = grouped_row * c_payload_bytes_per_row

            hidden_rsrc = ptr_rsrc(hidden)
            payload_rsrc = ptr_rsrc(grouped_payload)
            scale_rsrc = ptr_rsrc(grouped_scale)

            block_in_wave = arith.divui(lane, c_lanes_per_block)
            lane_in_block = lane - block_in_wave * c_lanes_per_block
            is_block_lead = arith.cmpi(CmpIPredicate.eq, lane_in_block, c0_i32)

            c = SimpleNamespace(
                i32=i32,
                f32=f32,
                block_iters=1,
                mx_blocks_per_wave_iter=mx_blocks_per_wave_iter,
                mx_blocks_per_row=mx_blocks_per_row,
                amax_shuffle_dists=amax_shuffle_dists,
                is_fp8=is_fp8,
                use_native=use_native,
                use_pk8=use_pk8,
                mx_dtype=mx_dtype,
                c0_i32=c0_i32,
                c1_i32=c1_i32,
                c4_i32=c4_i32,
                c23_i32=c23_i32,
                c254_i32=c254_i32,
                c0_f32=c0_f32,
                c_wave=c_wave,
                c_elems_per_lane=c_elems_per_lane,
                c_payload_bytes_per_block=c_payload_bytes_per_block,
                c_payload_bytes_per_lane=c_payload_bytes_per_lane,
                c_wmma_rep=c_wmma_rep,
                block_in_wave=block_in_wave,
                lane_in_block=lane_in_block,
                is_block_lead=is_block_lead,
                dests=[
                    SimpleNamespace(
                        payload_row_byte_base=payload_row_byte_base,
                        scale_row_dword_base=scale_row_dword_base,
                    )
                ],
                feat_elem_base=c0_i32,
                hidden_rsrc=hidden_rsrc,
                payload_rsrc=payload_rsrc,
                scale_rsrc=scale_rsrc,
            )
            _emit_quant_one_k_group(c, k_group)
            scf.YieldOp([])

    @flyc.jit
    def launch_fused(
        topk_ids: fx.Pointer,
        counter: fx.Pointer,
        topids_to_rows: fx.Pointer,
        hidden: fx.Pointer,
        grouped_payload: fx.Pointer,
        grouped_scale: fx.Pointer,
        expert_row_base: fx.Pointer,
        numel: fx.Int32,
        grid_route_blocks: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        grid_x = arith.index_cast(T.index, grid_route_blocks)
        grid_y = arith.index_cast(T.index, arith.constant(k_groups, type=T.i32))
        fused_kernel(
            topk_ids,
            counter,
            topids_to_rows,
            hidden,
            grouped_payload,
            grouped_scale,
            expert_row_base,
            numel,
        ).launch(
            grid=(grid_x, grid_y, 1),
            block=(block_threads, 1, 1),
            stream=stream,
        )

    launch_fused.compile_hints = {
        "llvm_options": {
            "amdgpu-kernarg-preload": True,
            "amdgpu-kernarg-preload-count": MOE_KERNARG_PRELOAD_COUNT,
        },
    }

    return launch_fused


def build_moe_fused_quant_preshuffle_module(
    feat_dim: int,
    wmma_rep: int,
    quant_mode: str = "fp4",
    skip_padding: bool = False,
):
    """Return a JIT launcher for the fused (grouped) quant + scale-preshuffle kernel.

    The stage2 analog of ``build_moe_fused_route_quant_scatter_module``: the input
    is *already* grouped row-major ``(E, max_m, feat_dim)`` (e.g. the stage1 GEMM
    output), so there is no route map / atomic slot / scatter -- one warp per
    grouped row quantizes that row straight into the grouped MX payload and writes
    the e8m0 block scales into the preshuffled WMMA layout. Replaces
    ``per_1x32_f4_quant`` / MXFP8 quant + ``flydsl_moe_preshuffle_scale``.

    Parameters
    ----------
    feat_dim : int     feature dim being quantized along K (inter_dim for stage2);
                       multiple of 32.
    wmma_rep : int     ``warp_tile_m // 16`` (scale preshuffle tile geometry).
    quant_mode : str   ``"fp4"`` (payload feat_dim//2) or ``"fp8"`` (payload feat_dim).
    skip_padding : bool  when True the kernel reads ``masked_m[expert]`` and skips
                       padding rows (``slot >= masked_m[expert]``) entirely -- no
                       hidden read, no quant, no store. Only valid for the masked
                       ``(E, max_m)`` layout where ``expert = row // max_m``; the
                       caller must pass a real ``masked_m``. When False every one
                       of the ``E*max_m`` rows is quantized (padding included);
                       ``masked_m`` is then ignored (a dummy may be passed).

    Launcher signature::

        (grouped_in, grouped_payload, grouped_scale, masked_m, n_rows, max_m,
         grid_blocks, stream=...)

      grouped_in      : (n_rows*feat_dim,) bf16   flat grouped activations
      grouped_payload : (n_rows*payload_bytes_per_row,) uint8  out: MX payload
      grouped_scale   : (E*(max_m//wmma_rep)*(feat_dim//32)*wmma_rep,) uint8
                        out: preshuffled e8m0 scale
      masked_m        : (E,) int32  per-expert valid row count (read iff skip_padding)
      n_rows          : E*max_m  (padding rows skipped iff skip_padding)
      max_m           : per-expert row capacity (for expert = row // max_m)
    """
    L = _quant_layout(feat_dim, quant_mode, wmma_rep)
    # Unpack into locals so the @kernel closure captures the quant_mode-derived
    # scalars (is_fp8, payload geometry, ...). The JIT disk cache keys on the
    # launch function's source + scalar closure values; if these stayed hidden
    # inside the ``L`` namespace the fp4 and fp8 variants (same feat_dim/wmma_rep)
    # would hash to the same key and silently share one binary.
    is_fp8 = L.is_fp8
    use_native = L.use_native
    use_pk8 = L.use_pk8
    elems_per_lane = L.elems_per_lane
    lanes_per_mx_block = L.lanes_per_mx_block
    mx_dtype = L.mx_dtype
    payload_bytes_per_row = L.payload_bytes_per_row
    payload_bytes_per_block = L.payload_bytes_per_block
    payload_bytes_per_lane = L.payload_bytes_per_lane
    wave_size = L.wave_size
    warps_per_block = L.warps_per_block
    mx_blocks_per_wave_iter = L.mx_blocks_per_wave_iter
    mx_blocks_per_row = L.mx_blocks_per_row
    scale_dwords_per_row = L.scale_dwords_per_row
    rows_per_tile = L.rows_per_tile
    dst_scale_dwords_per_row = L.dst_scale_dwords_per_row
    block_iters = L.block_iters
    amax_shuffle_dists = L.amax_shuffle_dists

    # skip_padding changes the emitted control flow (and the masked_m read), so it
    # must be part of the JIT cache key via the module name -- otherwise the two
    # variants (same feat_dim/wmma_rep/quant_mode) would collide on one binary.
    skip_tag = "skip" if skip_padding else "all"
    module_name = (
        f"moe_fused_quant_preshuffle_fd{feat_dim}_r{wmma_rep}"
        f"_{quant_mode}_{L.native_tag}_{skip_tag}"
    )

    @flyc.kernel(name=module_name)
    def fused_kernel(
        grouped_in: fx.Pointer,  # (n_rows*feat_dim,) bf16
        grouped_payload: fx.Pointer,  # (n_rows*payload_bytes_per_row,) uint8 out
        grouped_scale: fx.Pointer,  # preshuffled e8m0 out
        masked_m: fx.Pointer,  # (E,) int32 valid row count (read iff skip_padding)
        n_rows: Int32,
        max_m: Int32,
    ):
        i32 = T.i32
        f32 = T.f32

        c0_i32 = arith.constant(0, type=i32)
        c1_i32 = arith.constant(1, type=i32)
        c4_i32 = arith.constant(4, type=i32)
        c16_i32 = arith.constant(16, type=i32)
        c23_i32 = arith.constant(23, type=i32)
        c254_i32 = arith.constant(254, type=i32)
        c0_f32 = arith.constant(0.0, type=f32)

        c_wave = arith.constant(wave_size, type=i32)
        c_feat_dim = arith.constant(feat_dim, type=i32)
        c_payload_bytes_per_row = arith.constant(payload_bytes_per_row, type=i32)
        c_payload_bytes_per_block = arith.constant(payload_bytes_per_block, type=i32)
        c_payload_bytes_per_lane = arith.constant(payload_bytes_per_lane, type=i32)
        c_scale_dwords_per_row = arith.constant(scale_dwords_per_row, type=i32)
        c_dst_scale_dwords_per_row = arith.constant(dst_scale_dwords_per_row, type=i32)
        c_wmma_rep = arith.constant(wmma_rep, type=i32)
        c_rows_per_tile = arith.constant(rows_per_tile, type=i32)
        c_lanes_per_block = arith.constant(lanes_per_mx_block, type=i32)
        c_elems_per_lane = arith.constant(elems_per_lane, type=i32)

        tid = ArithValue(fx.thread_idx.x)
        bid = ArithValue(fx.block_idx.x)

        warp_in_block = tid // c_wave
        lane = tid - warp_in_block * c_wave  # tid % wave_size
        # one warp per grouped row (no routing: row == grouped row).
        row = bid * arith.constant(warps_per_block, type=i32) + warp_in_block

        row_in_range = arith.cmpi(CmpIPredicate.ult, row, ArithValue(n_rows))
        _if_row = scf.IfOp(row_in_range)
        with ir.InsertionPoint(_if_row.then_block):
            m = ArithValue(max_m)
            expert = ArithValue(arith.divui(row, m))
            slot = row - expert * m  # row within expert

            def _emit_row():
                # --- per-row scale-preshuffle geometry (uniform; row pos == slot) ---
                scale_tile = arith.divui(slot, c_rows_per_tile)
                row_in_tile = slot - scale_tile * c_rows_per_tile
                wmma_row = arith.divui(row_in_tile, c16_i32)
                row_lane16 = row_in_tile - wmma_row * c16_i32
                out_row = scale_tile * c16_i32 + row_lane16
                scale_row_dword_base = (
                    expert * (m * c_scale_dwords_per_row)
                    + out_row * c_dst_scale_dwords_per_row
                    + wmma_row
                )

                payload_row_byte_base = row * c_payload_bytes_per_row
                feat_elem_base = row * c_feat_dim  # bf16 element base for this row

                hidden_rsrc = ptr_rsrc(grouped_in)
                payload_rsrc = ptr_rsrc(grouped_payload)
                scale_rsrc = ptr_rsrc(grouped_scale)

                block_in_wave = arith.divui(lane, c_lanes_per_block)
                lane_in_block = lane - block_in_wave * c_lanes_per_block
                is_block_lead = arith.cmpi(CmpIPredicate.eq, lane_in_block, c0_i32)

                c = SimpleNamespace(
                    i32=i32,
                    f32=f32,
                    block_iters=block_iters,
                    mx_blocks_per_wave_iter=mx_blocks_per_wave_iter,
                    mx_blocks_per_row=mx_blocks_per_row,
                    amax_shuffle_dists=amax_shuffle_dists,
                    is_fp8=is_fp8,
                    use_native=use_native,
                    use_pk8=use_pk8,
                    mx_dtype=mx_dtype,
                    c0_i32=c0_i32,
                    c1_i32=c1_i32,
                    c4_i32=c4_i32,
                    c23_i32=c23_i32,
                    c254_i32=c254_i32,
                    c0_f32=c0_f32,
                    c_wave=c_wave,
                    c_elems_per_lane=c_elems_per_lane,
                    c_payload_bytes_per_block=c_payload_bytes_per_block,
                    c_payload_bytes_per_lane=c_payload_bytes_per_lane,
                    c_wmma_rep=c_wmma_rep,
                    block_in_wave=block_in_wave,
                    lane_in_block=lane_in_block,
                    is_block_lead=is_block_lead,
                    dests=[
                        SimpleNamespace(
                            payload_row_byte_base=payload_row_byte_base,
                            scale_row_dword_base=scale_row_dword_base,
                        )
                    ],
                    feat_elem_base=feat_elem_base,
                    hidden_rsrc=hidden_rsrc,
                    payload_rsrc=payload_rsrc,
                    scale_rsrc=scale_rsrc,
                )
                _emit_quant_block_loop(c)

            if const_expr(skip_padding):
                # Skip padding rows: the masked GEMM never reads rows beyond
                # masked_m[expert], so quantizing them is pure waste. With high
                # capacity-factor padding this elides most of the work.
                masked_rsrc = ptr_rsrc(masked_m)
                valid = ArithValue(
                    buffer_ops.buffer_load(masked_rsrc, expert, vec_width=1, dtype=i32)
                )
                slot_valid = arith.cmpi(CmpIPredicate.ult, slot, valid)
                _if_valid = scf.IfOp(slot_valid)
                with ir.InsertionPoint(_if_valid.then_block):
                    _emit_row()
                    scf.YieldOp([])
            else:
                _emit_row()
            scf.YieldOp([])

    @flyc.jit
    def launch_fused(
        grouped_in: fx.Pointer,
        grouped_payload: fx.Pointer,
        grouped_scale: fx.Pointer,
        masked_m: fx.Pointer,
        n_rows: fx.Int32,
        max_m: fx.Int32,
        grid_blocks: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        grid_x = arith.index_cast(T.index, grid_blocks)
        fused_kernel(
            grouped_in,
            grouped_payload,
            grouped_scale,
            masked_m,
            n_rows,
            max_m,
        ).launch(
            grid=(grid_x, 1, 1),
            block=(BLOCK_THREADS, 1, 1),
            stream=stream,
        )

    launch_fused.compile_hints = {
        "llvm_options": {
            "amdgpu-kernarg-preload": True,
            "amdgpu-kernarg-preload-count": MOE_KERNARG_PRELOAD_COUNT,
        },
    }

    return launch_fused


def build_moe_fused_quant_preshuffle_route_ksplit_module(
    feat_dim: int,
    wmma_rep: int,
    quant_mode: str = "fp4",
    source_topk: int = 0,
    remap_rows: bool = False,
):
    """Route-indexed K-split grouped quant+preshuffle for small token counts.

    Instead of launching over every row in the (E, max_m) capacity buffer and
    skipping padding, this kernel launches only over routed rows
    (``topids_to_rows``) and K groups. It works for both masked and contiguous-M
    layouts because ``topids_to_rows`` already contains the global row index in
    the actual output/input buffer.
    """
    L = _quant_layout(feat_dim, quant_mode, wmma_rep)
    if not L.use_pk8:
        raise NotImplementedError(
            "route-indexed K-split is currently enabled only for gfx1250 pk8"
        )

    is_fp8 = L.is_fp8
    use_native = L.use_native
    use_pk8 = L.use_pk8
    elems_per_lane = L.elems_per_lane
    lanes_per_mx_block = L.lanes_per_mx_block
    mx_dtype = L.mx_dtype
    payload_bytes_per_row = L.payload_bytes_per_row
    payload_bytes_per_block = L.payload_bytes_per_block
    payload_bytes_per_lane = L.payload_bytes_per_lane
    wave_size = L.wave_size
    warps_per_block = L.warps_per_block
    mx_blocks_per_wave_iter = L.mx_blocks_per_wave_iter
    mx_blocks_per_row = L.mx_blocks_per_row
    rows_per_tile = L.rows_per_tile
    dst_scale_dwords_per_row = L.dst_scale_dwords_per_row
    block_iters = L.block_iters
    amax_shuffle_dists = L.amax_shuffle_dists

    source_tag = f"srctk{source_topk}" if source_topk > 0 else "srcrow"
    remap_tag = "_remap" if remap_rows else ""
    source_topk_is_pow2 = source_topk > 0 and (source_topk & (source_topk - 1)) == 0
    source_topk_shift = source_topk.bit_length() - 1 if source_topk_is_pow2 else 0

    module_name = (
        f"moe_fused_quant_preshuffle_routeks_fd{feat_dim}_r{wmma_rep}"
        f"_{quant_mode}_{L.native_tag}_{source_tag}{remap_tag}"
    )

    @flyc.kernel(name=module_name)
    def fused_kernel(
        grouped_in: fx.Pointer,  # flat grouped activations
        grouped_payload: fx.Pointer,
        grouped_scale: fx.Pointer,
        topids_to_rows: fx.Pointer,  # (numel,) int32 global rows
        row_starts: fx.Pointer,  # (E,) int32, read iff remap_rows
        route_max_m: Int32,  # masked route stride, read iff remap_rows
        numel: Int32,
    ):
        i32 = T.i32
        f32 = T.f32

        c0_i32 = arith.constant(0, type=i32)
        c1_i32 = arith.constant(1, type=i32)
        c4_i32 = arith.constant(4, type=i32)
        c16_i32 = arith.constant(16, type=i32)
        c23_i32 = arith.constant(23, type=i32)
        c254_i32 = arith.constant(254, type=i32)
        c0_f32 = arith.constant(0.0, type=f32)

        c_wave = arith.constant(wave_size, type=i32)
        c_feat_dim = arith.constant(feat_dim, type=i32)
        c_payload_bytes_per_row = arith.constant(payload_bytes_per_row, type=i32)
        c_payload_bytes_per_block = arith.constant(payload_bytes_per_block, type=i32)
        c_payload_bytes_per_lane = arith.constant(payload_bytes_per_lane, type=i32)
        c_dst_scale_dwords_per_row = arith.constant(dst_scale_dwords_per_row, type=i32)
        c_wmma_rep = arith.constant(wmma_rep, type=i32)
        c_rows_per_tile = arith.constant(rows_per_tile, type=i32)
        c_lanes_per_block = arith.constant(lanes_per_mx_block, type=i32)
        c_elems_per_lane = arith.constant(elems_per_lane, type=i32)
        c_source_topk = arith.constant(source_topk, type=i32)
        c_source_topk_shift = arith.constant(source_topk_shift, type=i32)

        tid = ArithValue(fx.thread_idx.x)
        bid = ArithValue(fx.block_idx.x)
        k_group = ArithValue(fx.block_idx.y)

        warp_in_block = tid // c_wave
        lane = tid - warp_in_block * c_wave
        route = bid * arith.constant(warps_per_block, type=i32) + warp_in_block

        route_in_range = arith.cmpi(CmpIPredicate.ult, route, ArithValue(numel))
        _if_route = scf.IfOp(route_in_range)
        with ir.InsertionPoint(_if_route.then_block):
            rows_rsrc = ptr_rsrc(topids_to_rows)
            row = ArithValue(
                buffer_ops.buffer_load(rows_rsrc, route, vec_width=1, dtype=i32)
            )
            if const_expr(remap_rows):
                m = ArithValue(route_max_m)
                expert = ArithValue(arith.divui(row, m))
                slot = row - expert * m
                starts_rsrc = ptr_rsrc(row_starts)
                row = (
                    ArithValue(
                        buffer_ops.buffer_load(
                            starts_rsrc, expert, vec_width=1, dtype=i32
                        )
                    )
                    + slot
                )
                is_lane0 = arith.cmpi(CmpIPredicate.eq, lane, c0_i32)
                is_k0 = arith.cmpi(CmpIPredicate.eq, k_group, c0_i32)
                _if_store = scf.IfOp(arith.andi(is_lane0, is_k0))
                with ir.InsertionPoint(_if_store.then_block):
                    buffer_ops.buffer_store(row, rows_rsrc, route)
                    scf.YieldOp([])

            scale_tile = arith.divui(row, c_rows_per_tile)
            row_in_tile = row - scale_tile * c_rows_per_tile
            wmma_row = arith.divui(row_in_tile, c16_i32)
            row_lane16 = row_in_tile - wmma_row * c16_i32
            out_row = scale_tile * c16_i32 + row_lane16
            scale_row_dword_base = out_row * c_dst_scale_dwords_per_row + wmma_row

            payload_row_byte_base = row * c_payload_bytes_per_row
            if const_expr(source_topk > 0):
                if const_expr(source_topk_is_pow2):
                    source_row = route >> c_source_topk_shift
                else:
                    source_row = arith.divui(route, c_source_topk)
                feat_elem_base = source_row * c_feat_dim
            else:
                feat_elem_base = row * c_feat_dim

            hidden_rsrc = ptr_rsrc(grouped_in)
            payload_rsrc = ptr_rsrc(grouped_payload)
            scale_rsrc = ptr_rsrc(grouped_scale)

            block_in_wave = arith.divui(lane, c_lanes_per_block)
            lane_in_block = lane - block_in_wave * c_lanes_per_block
            is_block_lead = arith.cmpi(CmpIPredicate.eq, lane_in_block, c0_i32)

            c = SimpleNamespace(
                i32=i32,
                f32=f32,
                block_iters=1,
                mx_blocks_per_wave_iter=mx_blocks_per_wave_iter,
                mx_blocks_per_row=mx_blocks_per_row,
                amax_shuffle_dists=amax_shuffle_dists,
                is_fp8=is_fp8,
                use_native=use_native,
                use_pk8=use_pk8,
                mx_dtype=mx_dtype,
                c0_i32=c0_i32,
                c1_i32=c1_i32,
                c4_i32=c4_i32,
                c23_i32=c23_i32,
                c254_i32=c254_i32,
                c0_f32=c0_f32,
                c_wave=c_wave,
                c_elems_per_lane=c_elems_per_lane,
                c_payload_bytes_per_block=c_payload_bytes_per_block,
                c_payload_bytes_per_lane=c_payload_bytes_per_lane,
                c_wmma_rep=c_wmma_rep,
                block_in_wave=block_in_wave,
                lane_in_block=lane_in_block,
                is_block_lead=is_block_lead,
                dests=[
                    SimpleNamespace(
                        payload_row_byte_base=payload_row_byte_base,
                        scale_row_dword_base=scale_row_dword_base,
                    )
                ],
                feat_elem_base=feat_elem_base,
                hidden_rsrc=hidden_rsrc,
                payload_rsrc=payload_rsrc,
                scale_rsrc=scale_rsrc,
            )
            _emit_quant_one_k_group(c, k_group)
            scf.YieldOp([])

    @flyc.jit
    def launch_fused(
        grouped_in: fx.Pointer,
        grouped_payload: fx.Pointer,
        grouped_scale: fx.Pointer,
        topids_to_rows: fx.Pointer,
        row_starts: fx.Pointer,
        route_max_m: fx.Int32,
        numel: fx.Int32,
        grid_route_blocks: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        grid_x = arith.index_cast(T.index, grid_route_blocks)
        grid_y = arith.index_cast(T.index, arith.constant(block_iters, type=T.i32))
        fused_kernel(
            grouped_in,
            grouped_payload,
            grouped_scale,
            topids_to_rows,
            row_starts,
            route_max_m,
            numel,
        ).launch(
            grid=(grid_x, grid_y, 1),
            block=(BLOCK_THREADS, 1, 1),
            stream=stream,
        )

    launch_fused.compile_hints = {
        "llvm_options": {
            "amdgpu-kernarg-preload": True,
            "amdgpu-kernarg-preload-count": MOE_KERNARG_PRELOAD_COUNT,
        },
    }

    return launch_fused


def build_moe_fused_route_psum_quant_scatter_module(
    model_dim: int,
    topk: int,
    wmma_rep: int,
    quant_mode: str = "fp4",
):
    """Return a JIT launcher for the *fully fused* DeepGEMM contiguous-M stage1 prep.

    This is the single-kernel fusion of three previously-separate launches in the
    contiguous-M path (see ``grouped_moe_gfx1250.py``):

        1. ``torch.bincount(flat_experts)``        -> per-expert counts (masked_m)
        2. ``moe_contiguous_psum``                 -> tile-aligned exclusive prefix
                                                      sum (starts) + actual ends (psum)
        3. ``moe_fused_route_quant_scatter``       -> route + MX quant + scatter +
                                                      scale-preshuffle

    A single persistent grid (``num_workers`` resident workgroups) runs three
    phases separated by a hand-rolled grid-wide barrier (FlyDSL has no
    ``grid.sync`` / cooperative launch; this mirrors the global-atomic spin in
    ``splitk_hgemm.py``). Each worker owns a strided slice of the
    ``numel = token_num*topk`` routes (warp-per-route, ``stride =
    num_workers*warps_per_block``):

        Phase 1 (all blocks): ``lane0: atomicAdd(count[expert], 1)`` -> count == masked_m.
        Barrier A: every block-leader arrives on ``barrier[0]``; the last arriver
                   runs Phase 2, the rest spin on the release flag ``barrier[1]``.
        Phase 2 (last block, one thread): serial tile-aligned prefix sum over
                   ``count`` -> ``starts``/``psum`` (logic lifted from
                   ``moe_contiguous_psum``), then publishes ``barrier[1] = 1``.
        Phase 3 (all blocks): ``lane0: slot = atomicAdd(slot_counter[expert], 1)``,
                   ``grouped_row = starts[expert] + slot``, then the shared
                   ``_emit_quant_block_loop`` quantizes + scatters + preshuffles.

    The destination is always the DeepGEMM contiguous-M layout: a single
    ``(1, contiguous_m)`` payload/scale buffer indexed by the global
    ``grouped_row = starts[expert] + slot`` (every ``starts[e]`` is tile_m-aligned,
    hence a multiple of ``wmma_rep*16``, so the preshuffle tiling is consistent).

    Cross-block memory ordering uses ``syncscope="agent"`` atomics plus coherent
    (``sc0 sc1``) global load/store + ``s_waitcnt(0)`` around the release flag, so
    the prefix-sum reads of ``count`` and the Phase-3 reads of ``starts`` observe
    the committed values.

    Launcher signature::

        (topk_ids, count, slot_counter, starts, psum, barrier, topids_to_rows,
         hidden, grouped_payload, grouped_scale, numel, experts, tile_m,
         num_workers, grid_blocks, stream=...)

      topk_ids        : (numel,)               int32  flattened expert ids
      count           : (E,)                   int32  in/out, init 0 (== masked_m)
      slot_counter    : (E,)                   int32  in/out, init 0 (phase-3 slots)
      starts          : (E,)                   int32  out  tile-aligned prefix sum
      psum            : (E,)                   int32  out  starts[e]+count[e]
      barrier         : (2,)                   int32  in/out, init 0 (arrival/release)
      topids_to_rows  : (numel,)               int32  out  route -> grouped row
      hidden          : (token_num*model_dim,) bf16   flat activations
      grouped_payload : (contiguous_m*payload_bytes_per_row,) uint8  out MX payload
      grouped_scale   : (contiguous_m*(model_dim//32),) uint8  out preshuffled e8m0
      experts         : int32  number of experts E (matches count/slot/starts len)
      tile_m          : int32  contiguous-M tile (starts aligned to this)
      num_workers     : int32  resident workgroup count (== grid_blocks)
    """
    L = _quant_layout(model_dim, quant_mode, wmma_rep)
    is_fp8 = L.is_fp8
    use_native = L.use_native
    use_pk8 = L.use_pk8
    elems_per_lane = L.elems_per_lane
    lanes_per_mx_block = L.lanes_per_mx_block
    mx_dtype = L.mx_dtype
    payload_bytes_per_row = L.payload_bytes_per_row
    payload_bytes_per_block = L.payload_bytes_per_block
    payload_bytes_per_lane = L.payload_bytes_per_lane
    wave_size = L.wave_size
    warps_per_block = L.warps_per_block
    mx_blocks_per_wave_iter = L.mx_blocks_per_wave_iter
    mx_blocks_per_row = L.mx_blocks_per_row
    rows_per_tile = L.rows_per_tile
    dst_scale_dwords_per_row = L.dst_scale_dwords_per_row
    block_iters = L.block_iters
    amax_shuffle_dists = L.amax_shuffle_dists

    module_name = (
        f"moe_fused_route_psum_quant_scatter_md{model_dim}_tk{topk}_r{wmma_rep}"
        f"_{quant_mode}_{L.native_tag}"
    )

    # gfx12 split the memory wait counters (s_wait_loadcnt / s_wait_storecnt);
    # gfx9 uses the unified ``s_waitcnt``. The cross-block barrier publishes/reads
    # its scratch (count / starts / psum / release flag) exclusively through
    # agent-scope atomics + plain buffer loads, which is the only reliably
    # L2-coherent cross-CU producer/consumer pattern on gfx1250 (hand-rolled
    # inline-asm coherent global load/store miscompiles here).
    _is_gfx12 = str(L.arch).startswith("gfx12")

    @flyc.kernel(name=module_name)
    def fused_kernel(
        topk_ids: fx.Pointer,  # (numel,) int32
        count: fx.Pointer,  # (E,) int32 in/out (init 0) -> masked_m
        slot_counter: fx.Pointer,  # (E,) int32 in/out (init 0)
        starts: fx.Pointer,  # (E,) int32 out
        psum: fx.Pointer,  # (E,) int32 out
        barrier: fx.Pointer,  # (2,) int32 in/out (init 0): [0]=arrival, [1]=release
        topids_to_rows: fx.Pointer,  # (numel,) int32 out
        hidden: fx.Pointer,  # (token_num*model_dim,) bf16
        grouped_payload: fx.Pointer,  # (contiguous_m*payload_bytes_per_row,) uint8 out
        grouped_scale: fx.Pointer,  # preshuffled e8m0 out
        numel: Int32,
        experts: Int32,
        tile_m: Int32,
        num_workers: Int32,
    ):
        i32 = T.i32
        f32 = T.f32

        c0_i32 = arith.constant(0, type=i32)
        c1_i32 = arith.constant(1, type=i32)
        c4_i32 = arith.constant(4, type=i32)
        c16_i32 = arith.constant(16, type=i32)
        c23_i32 = arith.constant(23, type=i32)
        c254_i32 = arith.constant(254, type=i32)
        c0_f32 = arith.constant(0.0, type=f32)

        c_wave = arith.constant(wave_size, type=i32)
        c_warps_per_block = arith.constant(warps_per_block, type=i32)
        c_topk = arith.constant(topk, type=i32)
        c_model_dim = arith.constant(model_dim, type=i32)
        c_payload_bytes_per_row = arith.constant(payload_bytes_per_row, type=i32)
        c_payload_bytes_per_block = arith.constant(payload_bytes_per_block, type=i32)
        c_payload_bytes_per_lane = arith.constant(payload_bytes_per_lane, type=i32)
        c_dst_scale_dwords_per_row = arith.constant(dst_scale_dwords_per_row, type=i32)
        c_wmma_rep = arith.constant(wmma_rep, type=i32)
        c_rows_per_tile = arith.constant(rows_per_tile, type=i32)
        c_lanes_per_block = arith.constant(lanes_per_mx_block, type=i32)
        c_elems_per_lane = arith.constant(elems_per_lane, type=i32)

        # --- cross-block scratch access helpers (raw !llvm.ptr<1> at elem idx) ---
        def _wait_mem():
            # Drain outstanding global memory ops (loads + stores) so atomics /
            # coherent writes are committed to the L2 coherence point.
            if const_expr(_is_gfx12):
                rocdl.s_wait_loadcnt(0)
                rocdl.s_wait_storecnt(0)
            else:
                rocdl.s_waitcnt(0)

        def _elem_ptr(tensor, elem_idx_i32):
            base = arith.index_cast(T.index, ptrtoint(tensor))
            idx = arith.index_cast(T.index, elem_idx_i32)
            addr = fx.Index(base) + idx * fx.Index(4)
            p = buffer_ops.create_llvm_ptr(addr, address_space=1)
            return p._value if hasattr(p, "_value") else p

        def _atomic_add(tensor, elem_idx_i32, addend):
            ptr = _elem_ptr(tensor, elem_idx_i32)
            return ArithValue(
                llvm.AtomicRMWOp(
                    llvm.AtomicBinOp.add,
                    ptr,
                    addend,
                    llvm.AtomicOrdering.monotonic,
                    syncscope="agent",
                    alignment=4,
                ).result
            )

        tid = ArithValue(fx.thread_idx.x)
        bid = ArithValue(fx.block_idx.x)

        warp_in_block = tid // c_wave
        lane = tid - warp_in_block * c_wave  # tid % wave_size
        route0 = bid * c_warps_per_block + warp_in_block  # first route this warp owns
        stride = ArithValue(num_workers) * c_warps_per_block

        topk_ids_rsrc = ptr_rsrc(topk_ids)
        count_rsrc = ptr_rsrc(count)

        # ============================ Phase 1: count ============================
        # Strided warp-per-route histogram into ``count`` (== masked_m). Loop bounds
        # are warp-uniform (lane-independent) so the post-phase gpu.barrier() is hit
        # by every thread of the block.
        route0_idx = arith.index_cast(T.index, route0)
        numel_idx = arith.index_cast(T.index, ArithValue(numel))
        stride_idx = arith.index_cast(T.index, stride)
        loop1 = scf.ForOp(route0_idx, numel_idx, stride_idx)
        with ir.InsertionPoint(loop1.body):
            route_i32 = arith.index_cast(i32, loop1.induction_variable)
            expert = ArithValue(
                buffer_ops.buffer_load(topk_ids_rsrc, route_i32, vec_width=1, dtype=i32)
            )
            if lane == 0:
                _atomic_add(count, expert, c1_i32)
            scf.YieldOp([])

        # ===================== Barrier A + Phase 2: prefix sum ==================
        gpu.barrier()
        _wait_mem()
        rocdl.sched_barrier(0)

        is_block_leader = arith.cmpi(CmpIPredicate.eq, tid, c0_i32)
        _if_leader = scf.IfOp(is_block_leader)
        with ir.InsertionPoint(_if_leader.then_block):
            my_arrival = _atomic_add(barrier, c0_i32, c1_i32)
            nwm1 = ArithValue(num_workers) - c1_i32
            is_last = arith.cmpi(CmpIPredicate.eq, my_arrival, nwm1)
            is_not_last = arith.cmpi(CmpIPredicate.ne, my_arrival, nwm1)

            # Last arriver: every block has bumped ``count`` (its atomics committed
            # to L2 before its arrival atomic). Serial tile-aligned prefix sum,
            # mirroring moe_contiguous_psum, reading ``count`` coherently.
            _if_last = scf.IfOp(is_last)
            with ir.InsertionPoint(_if_last.then_block):
                tile_v = ArithValue(tile_m)
                tile_minus_1 = tile_v - c1_i32
                e_upper = arith.index_cast(T.index, ArithValue(experts))
                c0_idx = arith.index(0)
                c1_idx = arith.index(1)
                ploop = scf.ForOp(c0_idx, e_upper, c1_idx, [c0_i32])
                with ir.InsertionPoint(ploop.body):
                    e = ploop.induction_variable
                    cur = ploop.inner_iter_args[0]
                    e_i32 = arith.index_cast(i32, e)
                    # This serial prefix sum runs in a single thread of the global
                    # last-arriver block, after the cross-block barrier guarantees
                    # every block's count atomics are committed.
                    #
                    # ``count`` is read with a plain buffer load (the same path
                    # Phase 1/3 use correctly): the hand-rolled inline-asm coherent
                    # load miscompiles inside this loop -- it aliases the count read
                    # with the just-written starts/psum accumulator, producing a
                    # Fibonacci-shaped runaway prefix sum. The count values are
                    # already L2-visible here (post-barrier) so no special load
                    # coherence is needed.
                    #
                    # ``starts``/``psum`` are published with agent-scope atomics
                    # (they are zero-initialised, so atomic-add == atomic write).
                    # This mirrors the count path -- atomic write here + a plain
                    # buffer load in Phase 3 -- which is the only cross-block
                    # producer/consumer pattern that is reliably L2-coherent on
                    # gfx1250; the inline-asm coherent store can linger in this
                    # block's L0 and not be visible to Phase 3 readers in time.
                    cnt = ArithValue(
                        buffer_ops.buffer_load(
                            count_rsrc, e_i32, vec_width=1, dtype=i32
                        )
                    )
                    q = arith.divui(cnt + tile_minus_1, tile_v)
                    aligned = ArithValue(q) * tile_v
                    _atomic_add(starts, e_i32, _raw(cur))
                    _atomic_add(psum, e_i32, _raw(ArithValue(cur) + cnt))
                    next_cur = ArithValue(cur) + ArithValue(aligned)
                    scf.YieldOp([next_cur])
                # Ensure starts/psum land in L2 before the release flag is visible,
                # then publish the release with an agent-scope atomic (the inline-asm
                # coherent store/load barrier is unreliable on gfx1250 -- readers can
                # observe the flag set before starts/psum are visible).
                _wait_mem()
                _atomic_add(barrier, c1_i32, c1_i32)
                scf.YieldOp([])

            # Other blocks: spin on the release flag until the last block publishes.
            _if_nl = scf.IfOp(is_not_last)
            with ir.InsertionPoint(_if_nl.then_block):
                init_cur = arith.constant(0, type=i32)
                w = scf.WhileOp([i32], [init_cur])
                before = ir.Block.create_at_start(w.before, [i32])
                after = ir.Block.create_at_start(w.after, [i32])
                with ir.InsertionPoint(before):
                    cur_w = before.arguments[0]
                    need_wait = arith.cmpi(CmpIPredicate.eq, cur_w, c0_i32)
                    scf.ConditionOp(need_wait, [cur_w])
                with ir.InsertionPoint(after):
                    # Coherent read via agent-scope atomic add of 0 (reliable on
                    # gfx1250, unlike the inline-asm coherent load).
                    rel = _atomic_add(barrier, c1_i32, c0_i32)
                    scf.YieldOp([_raw(rel)])
                scf.YieldOp([])
            scf.YieldOp([])

        # All threads converge here; the leader has observed the release flag, so
        # ``starts`` is committed and visible to coherent reads in Phase 3.
        gpu.barrier()
        _wait_mem()
        rocdl.sched_barrier(0)

        # ===================== Phase 3: route + quant + scatter =================
        # ``starts`` was published to L2 by the last-arriver block (coherent store
        # + release flag). Read it back with a plain buffer load -- the inline-asm
        # coherent load is unreliable here (same miscompile as the Phase 2 prefix
        # sum: it can return a stale 0 instead of the published row base, which
        # scatters the first route of an expert into row 0).
        starts_rd_rsrc = ptr_rsrc(starts)
        hidden_rsrc = ptr_rsrc(hidden)
        payload_rsrc = ptr_rsrc(grouped_payload)
        scale_rsrc = ptr_rsrc(grouped_scale)
        topids_to_rows_rsrc = ptr_rsrc(topids_to_rows)

        loop3 = scf.ForOp(route0_idx, numel_idx, stride_idx)
        with ir.InsertionPoint(loop3.body):
            route_i32 = arith.index_cast(i32, loop3.induction_variable)
            expert = ArithValue(
                buffer_ops.buffer_load(topk_ids_rsrc, route_i32, vec_width=1, dtype=i32)
            )

            # lane 0 claims the within-expert slot and reads the (published) per-
            # expert row base; both are warp-uniform, broadcast via readlane.
            slot_on_lane0 = arith.constant(0, type=i32)
            rowbase_on_lane0 = arith.constant(0, type=i32)
            if lane == 0:
                slot_on_lane0 = _atomic_add(slot_counter, expert, c1_i32)
                rowbase_on_lane0 = buffer_ops.buffer_load(
                    starts_rd_rsrc, _raw(expert), vec_width=1, dtype=i32
                )
            slot = ArithValue(rocdl.readlane(i32, _raw(slot_on_lane0), _raw(c0_i32)))
            row_base = ArithValue(
                rocdl.readlane(i32, _raw(rowbase_on_lane0), _raw(c0_i32))
            )
            grouped_row = slot + row_base
            token = arith.divui(route_i32, c_topk)

            if lane == 0:
                buffer_ops.buffer_store(grouped_row, topids_to_rows_rsrc, route_i32)

            # per-row scale-preshuffle geometry from the global grouped_row.
            scale_tile = arith.divui(grouped_row, c_rows_per_tile)
            row_in_tile = grouped_row - scale_tile * c_rows_per_tile
            wmma_row = arith.divui(row_in_tile, c16_i32)
            row_lane16 = row_in_tile - wmma_row * c16_i32
            out_row = scale_tile * c16_i32 + row_lane16
            scale_row_dword_base = out_row * c_dst_scale_dwords_per_row + wmma_row

            payload_row_byte_base = grouped_row * c_payload_bytes_per_row
            hidden_elem_base = token * c_model_dim

            block_in_wave = arith.divui(lane, c_lanes_per_block)
            lane_in_block = lane - block_in_wave * c_lanes_per_block
            is_block_lead = arith.cmpi(CmpIPredicate.eq, lane_in_block, c0_i32)

            c = SimpleNamespace(
                i32=i32,
                f32=f32,
                block_iters=block_iters,
                mx_blocks_per_wave_iter=mx_blocks_per_wave_iter,
                mx_blocks_per_row=mx_blocks_per_row,
                amax_shuffle_dists=amax_shuffle_dists,
                is_fp8=is_fp8,
                use_native=use_native,
                use_pk8=use_pk8,
                mx_dtype=mx_dtype,
                c0_i32=c0_i32,
                c1_i32=c1_i32,
                c4_i32=c4_i32,
                c23_i32=c23_i32,
                c254_i32=c254_i32,
                c0_f32=c0_f32,
                c_wave=c_wave,
                c_elems_per_lane=c_elems_per_lane,
                c_payload_bytes_per_block=c_payload_bytes_per_block,
                c_payload_bytes_per_lane=c_payload_bytes_per_lane,
                c_wmma_rep=c_wmma_rep,
                block_in_wave=block_in_wave,
                lane_in_block=lane_in_block,
                is_block_lead=is_block_lead,
                dests=[
                    SimpleNamespace(
                        payload_row_byte_base=payload_row_byte_base,
                        scale_row_dword_base=scale_row_dword_base,
                    )
                ],
                feat_elem_base=hidden_elem_base,
                hidden_rsrc=hidden_rsrc,
                payload_rsrc=payload_rsrc,
                scale_rsrc=scale_rsrc,
            )
            _emit_quant_block_loop(c)
            scf.YieldOp([])

    @flyc.jit
    def launch_fused(
        topk_ids: fx.Pointer,
        count: fx.Pointer,
        slot_counter: fx.Pointer,
        starts: fx.Pointer,
        psum: fx.Pointer,
        barrier: fx.Pointer,
        topids_to_rows: fx.Pointer,
        hidden: fx.Pointer,
        grouped_payload: fx.Pointer,
        grouped_scale: fx.Pointer,
        numel: fx.Int32,
        experts: fx.Int32,
        tile_m: fx.Int32,
        num_workers: fx.Int32,
        grid_blocks: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            pass

        grid_x = arith.index_cast(T.index, grid_blocks)
        fused_kernel(
            topk_ids,
            count,
            slot_counter,
            starts,
            psum,
            barrier,
            topids_to_rows,
            hidden,
            grouped_payload,
            grouped_scale,
            numel,
            experts,
            tile_m,
            num_workers,
        ).launch(
            grid=(grid_x, 1, 1),
            block=(BLOCK_THREADS, 1, 1),
            stream=stream,
        )

    launch_fused.compile_hints = {
        "llvm_options": {
            "amdgpu-kernarg-preload": True,
            "amdgpu-kernarg-preload-count": MOE_KERNARG_PRELOAD_COUNT,
        },
    }

    return launch_fused
