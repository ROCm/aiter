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
               grouped_row = expert*max_m + slot; topids_to_rows[i] = grouped_row
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
from flydsl.expr import arith, range_constexpr, const_expr, rocdl, vector
from flydsl.expr.typing import T, Int32
from flydsl.expr.arith import ArithValue, CmpIPredicate
from flydsl.compiler.kernel_function import CompilationContext

from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm, scf
from flydsl.expr import buffer_ops
from flydsl.runtime.device import get_rocm_arch

from aiter.ops.flydsl.kernels.quant_utils import emit_f32_to_e2m1, emit_mx_e8m0_scale
from aiter.ops.flydsl.kernels.kernels_common import get_warp_size

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
        mx_dtype = _MxDtype.FP8_E4M3_FNUZ if arch.startswith("gfx942") else _MxDtype.FP8_E4M3
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

    ``c`` carries the layout flags, SSA constants/types, buffer resources, and
    per-warp bases (``feat_elem_base``, ``payload_row_byte_base``,
    ``scale_row_dword_base``, ``block_in_wave``, ``lane_in_block``,
    ``is_block_lead``). Shared verbatim by the stage1 route kernel and the
    stage2 grouped kernel -- only the preamble that computes the bases differs.
    """
    i32 = c.i32
    f32 = c.f32
    for it in range_constexpr(c.block_iters):
        # MX block (along K) this lane works on this iteration.
        mx_block = (
            arith.constant(it * c.mx_blocks_per_wave_iter, type=i32) + c.block_in_wave
        )
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
                    xj = vector.extract(
                        f32x8, static_position=[j], dynamic_position=[]
                    )
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
                        block_scale_f32 = (
                            ArithValue(e8m0_scale) << c.c23_i32
                        ).bitcast(f32)
                        packed = rocdl.cvt_scalef32_pk_fp8_f32(
                            i32, _raw(c.c0_i32), _raw(x0), _raw(x1),
                            _raw(block_scale_f32), 0,
                        )
                    else:
                        recip_scale = (
                            (c.c254_i32 - e8m0_scale) << c.c23_i32
                        ).bitcast(f32)
                        scaled0 = ArithValue(x0) * recip_scale
                        scaled1 = ArithValue(x1) * recip_scale
                        # v_cvt_pk_fp8_f32: 2 f32 -> 2 fp8 bytes in word 0.
                        packed = rocdl.cvt_pk_fp8_f32(i32, scaled0, scaled1, c.c0_i32, 0)
                    payload_val = arith.trunci(T.i16, ArithValue(packed))  # 2 fp8 B
                else:
                    if const_expr(c.use_native):
                        block_scale_f32 = (
                            ArithValue(e8m0_scale) << c.c23_i32
                        ).bitcast(f32)
                        packed = rocdl.cvt_scalef32_pk_fp4_f32(
                            i32, _raw(c.c0_i32), _raw(x0), _raw(x1),
                            _raw(block_scale_f32), 0,
                        )
                        payload_val = arith.trunci(T.i8, ArithValue(packed))
                    else:
                        recip_scale = (
                            (c.c254_i32 - e8m0_scale) << c.c23_i32
                        ).bitcast(f32)
                        nib0 = emit_f32_to_e2m1(ArithValue(x0) * recip_scale)
                        nib1 = emit_f32_to_e2m1(ArithValue(x1) * recip_scale)
                        packed_byte = ArithValue(nib0) | (ArithValue(nib1) << c.c4_i32)
                        payload_val = arith.trunci(T.i8, packed_byte)  # 1 fp4x2 B

            # payload byte offset within grouped_payload. offset_is_bytes=True
            # so the i8 (fp4) / i16 (fp8) / i32 (pk8) store does not rescale this
            # already-byte offset by the data element size.
            payload_byte_off = (
                c.payload_row_byte_base
                + mx_block * c.c_payload_bytes_per_block
                + c.lane_in_block * c.c_payload_bytes_per_lane
            )
            buffer_ops.buffer_store(
                payload_val, c.payload_rsrc, payload_byte_off, offset_is_bytes=True
            )

            # one e8m0 byte per block, written by the block's lead lane.
            _if_lead = scf.IfOp(c.is_block_lead)
            with ir.InsertionPoint(_if_lead.then_block):
                scale_dword = arith.divui(mx_block, c.c4_i32)
                byte_in_dword = mx_block - scale_dword * c.c4_i32
                dst_scale_dword = c.scale_row_dword_base + scale_dword * c.c_wmma_rep
                dst_scale_byte = dst_scale_dword * c.c4_i32 + byte_in_dword
                e8m0_byte = arith.trunci(T.i8, e8m0_scale)
                buffer_ops.buffer_store(e8m0_byte, c.scale_rsrc, dst_scale_byte)
                scf.YieldOp([])
            scf.YieldOp([])


def build_moe_fused_route_quant_scatter_module(
    model_dim: int,
    topk: int,
    wmma_rep: int,
    quant_mode: str = "fp4",
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

    Launcher signature::

        (topk_ids, counter, topids_to_rows, hidden, grouped_payload, grouped_scale,
         numel, max_m, grid_blocks, stream=...)

      topk_ids        : (numel,)               int32  flattened expert ids
      counter         : (E,)                   int32  per-expert counter, init 0
                        (== masked_m[expert] after the run)
      topids_to_rows  : (numel,)               int32  out: route -> grouped row
      hidden          : (token_num*model_dim,) bf16   flat activations
      grouped_payload : (E*max_m*payload_bytes_per_row,) uint8  out: MX payload
                        (payload_bytes_per_row = model_dim//2 fp4 / model_dim fp8)
      grouped_scale   : (E*(max_m//wmma_rep)*(model_dim//32)*wmma_rep,) uint8
                        out: preshuffled e8m0 scale
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
    scale_dwords_per_row = L.scale_dwords_per_row
    rows_per_tile = L.rows_per_tile
    dst_scale_dwords_per_row = L.dst_scale_dwords_per_row
    block_iters = L.block_iters
    amax_shuffle_dists = L.amax_shuffle_dists

    module_name = (
        f"moe_fused_route_quant_scatter_md{model_dim}_tk{topk}_r{wmma_rep}"
        f"_{quant_mode}_{L.native_tag}"
    )

    @flyc.kernel(name=module_name)
    def fused_kernel(
        topk_ids: fx.Tensor,  # (numel,) int32
        counter: fx.Tensor,  # (E,) int32, init 0
        topids_to_rows: fx.Tensor,  # (numel,) int32 out
        hidden: fx.Tensor,  # (token_num*model_dim,) bf16
        grouped_payload: fx.Tensor,  # (E*max_m*payload_bytes_per_row,) uint8 out
        grouped_scale: fx.Tensor,  # preshuffled e8m0 out
        numel: Int32,
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
        c_topk = arith.constant(topk, type=i32)
        c_model_dim = arith.constant(model_dim, type=i32)
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
        route = bid * arith.constant(warps_per_block, type=i32) + warp_in_block

        route_in_range = arith.cmpi(CmpIPredicate.ult, route, ArithValue(numel))
        _if_route = scf.IfOp(route_in_range)
        with ir.InsertionPoint(_if_route.then_block):
            topk_ids_rsrc = buffer_ops.create_buffer_resource(topk_ids, max_size=True)
            # expert id for this route (uniform across the warp)
            expert = ArithValue(
                buffer_ops.buffer_load(topk_ids_rsrc, route, vec_width=1, dtype=i32)
            )

            # --- lane 0 claims the within-expert slot via atomicAdd, then
            #     broadcasts it to the whole warp. Follows the readlane idiom in
            #     FlyDSL/kernels/dispatch_combine_intranode_kernel.py:218-225:
            #     init to 0, reassign on lane 0, readlane(...). ---
            slot_on_lane0 = arith.constant(0, type=i32)
            if lane == 0:
                counter_base = buffer_ops.extract_base_index(counter, address_space=1)
                expert_idx = arith.index_cast(T.index, expert)
                counter_addr = (
                    fx.Index(counter_base) + fx.Index(expert_idx) * fx.Index(4)
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

            grouped_row = slot + expert * ArithValue(max_m)
            token = arith.divui(route, c_topk)

            # topids_to_rows[route] = grouped_row (lane 0 only; warp-uniform value)
            if lane == 0:
                topids_to_rows_rsrc = buffer_ops.create_buffer_resource(
                    topids_to_rows, max_size=True
                )
                buffer_ops.buffer_store(grouped_row, topids_to_rows_rsrc, route)

            # --- per-row scale-preshuffle geometry (uniform; row position == slot) ---
            scale_tile = arith.divui(slot, c_rows_per_tile)
            row_in_tile = slot - scale_tile * c_rows_per_tile
            wmma_row = arith.divui(row_in_tile, c16_i32)
            row_lane16 = row_in_tile - wmma_row * c16_i32
            out_row = scale_tile * c16_i32 + row_lane16
            # dst dword base for (expert, out_row); scale_dword*wmma_rep added per block
            scale_row_dword_base = (
                expert * (ArithValue(max_m) * c_scale_dwords_per_row)
                + out_row * c_dst_scale_dwords_per_row
                + wmma_row
            )

            payload_row_byte_base = grouped_row * c_payload_bytes_per_row
            hidden_elem_base = token * c_model_dim  # bf16 element base for this token

            hidden_rsrc = buffer_ops.create_buffer_resource(hidden, max_size=True)
            payload_rsrc = buffer_ops.create_buffer_resource(
                grouped_payload, max_size=True
            )
            scale_rsrc = buffer_ops.create_buffer_resource(grouped_scale, max_size=True)

            # this lane's position inside its 16-lane MX block group
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
                payload_row_byte_base=payload_row_byte_base,
                feat_elem_base=hidden_elem_base,
                scale_row_dword_base=scale_row_dword_base,
                hidden_rsrc=hidden_rsrc,
                payload_rsrc=payload_rsrc,
                scale_rsrc=scale_rsrc,
            )
            _emit_quant_block_loop(c)
            scf.YieldOp([])

    @flyc.jit
    def launch_fused(
        topk_ids: fx.Tensor,
        counter: fx.Tensor,
        topids_to_rows: fx.Tensor,
        hidden: fx.Tensor,
        grouped_payload: fx.Tensor,
        grouped_scale: fx.Tensor,
        numel: fx.Int32,
        max_m: fx.Int32,
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
            numel,
            max_m,
        ).launch(
            grid=(grid_x, 1, 1),
            block=(BLOCK_THREADS, 1, 1),
            stream=stream,
        )

    return launch_fused


def build_moe_fused_quant_preshuffle_module(
    feat_dim: int,
    wmma_rep: int,
    quant_mode: str = "fp4",
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

    Launcher signature::

        (grouped_in, grouped_payload, grouped_scale, n_rows, max_m, grid_blocks,
         stream=...)

      grouped_in      : (n_rows*feat_dim,) bf16   flat grouped activations
      grouped_payload : (n_rows*payload_bytes_per_row,) uint8  out: MX payload
      grouped_scale   : (E*(max_m//wmma_rep)*(feat_dim//32)*wmma_rep,) uint8
                        out: preshuffled e8m0 scale
      n_rows          : E*max_m  (all rows are quantized; padding rows are unread)
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

    module_name = (
        f"moe_fused_quant_preshuffle_fd{feat_dim}_r{wmma_rep}"
        f"_{quant_mode}_{L.native_tag}"
    )

    @flyc.kernel(name=module_name)
    def fused_kernel(
        grouped_in: fx.Tensor,  # (n_rows*feat_dim,) bf16
        grouped_payload: fx.Tensor,  # (n_rows*payload_bytes_per_row,) uint8 out
        grouped_scale: fx.Tensor,  # preshuffled e8m0 out
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
        c_dst_scale_dwords_per_row = arith.constant(
            dst_scale_dwords_per_row, type=i32
        )
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

            # --- per-row scale-preshuffle geometry (uniform; row position == slot) ---
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

            hidden_rsrc = buffer_ops.create_buffer_resource(grouped_in, max_size=True)
            payload_rsrc = buffer_ops.create_buffer_resource(
                grouped_payload, max_size=True
            )
            scale_rsrc = buffer_ops.create_buffer_resource(grouped_scale, max_size=True)

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
                payload_row_byte_base=payload_row_byte_base,
                feat_elem_base=feat_elem_base,
                scale_row_dword_base=scale_row_dword_base,
                hidden_rsrc=hidden_rsrc,
                payload_rsrc=payload_rsrc,
                scale_rsrc=scale_rsrc,
            )
            _emit_quant_block_loop(c)
            scf.YieldOp([])

    @flyc.jit
    def launch_fused(
        grouped_in: fx.Tensor,
        grouped_payload: fx.Tensor,
        grouped_scale: fx.Tensor,
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
            n_rows,
            max_m,
        ).launch(
            grid=(grid_x, 1, 1),
            block=(BLOCK_THREADS, 1, 1),
            stream=stream,
        )

    return launch_fused

