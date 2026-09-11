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
    dst_dword   = e*(max_m*scale_dwords_per_row)
                  + scale_tile*(scale_dwords_per_row*wmma_rep*16)
                  + scale_dword*wmma_rep*16
                  + wmma_row*16 + row_lane16
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
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import arith, const_expr, gpu, ptrtoint, range_constexpr, rocdl
from flydsl.expr.arith import ArithValue
from flydsl.expr.typing import Int32, T
from flydsl.runtime.device import get_rocm_arch

from aiter.ops.flydsl.kernels import buffer_ops
from aiter.ops.flydsl.kernels import tdm_ops_gfx1250 as tdm_ops
from aiter.ops.flydsl.kernels.gemm_common_gfx1250 import make_lds_copy_ops
from aiter.ops.flydsl.kernels.kernels_common import (
    create_llvm_ptr,
    format_kernel_name,
    get_warp_size,
)
from aiter.ops.flydsl.kernels.moe_route_maps import DROPPED_ROUTE_ROW
from aiter.ops.flydsl.kernels.quant_utils import emit_f32_to_e2m1, emit_mx_e8m0_scale
from aiter.ops.flydsl.kernels.tensor_shim import (
    AITER_FLYDSL_KERNARG_PRELOAD,
    AITER_FLYDSL_KERNARG_PRELOAD_COUNT,
    buf_scalar_load,
    ptr_buf_tensor,
)
from aiter.utility.mx_types import (
    MX_DEFAULT_ROUND_MODE as _ROUND_MODE,
)
from aiter.utility.mx_types import (
    MxDtypeInt as _MxDtype,
)

BLOCK_THREADS = 256
# Nominal extent of a live destination descriptor; a dead one gets 0 instead.
# It only has to exceed any real buffer, and stays under 2 GiB because the
# descriptor builder sign-extends the size to 64 bits.
_SCALE_RSRC_MAX_BYTES = 0x7FFFFFFF
# TDM staging depth. One chunk is degenerate -- the prologue waits on the whole
# row with nothing to overlap -- and past four the per-chunk tensor_wait and two
# CTA barriers outweigh the smaller transfer. Four also divides every shipped
# model_dim's iteration count, so one depth serves them all.
_TOKEN_MULTIDEST_TDM_CHUNKS = 4
# The K-split aims for this many blocks per CU; past that it stops paying.
_TOKEN_MULTIDEST_BLOCKS_PER_CU = 4
_TOKEN_MULTIDEST_MAX_KSPLIT = 14
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


def _emit_pk8_lane_amax(bf16x8, c):
    """max(|x|) over the 8 bf16 this lane owns, as f32."""
    f32 = c.f32
    f32x8 = bf16x8.extf(T.vec(8, f32))
    acc = c.c0_f32
    for j in range_constexpr(8):
        xj = fx.Vector(f32x8)[j]
        absj = llvm.call_intrinsic(f32, "llvm.fabs.f32", [xj.ir_value()], [], [])
        acc = arith.maximumf(acc, absj)
    return acc


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
    # _emit_quant_block_loop emits the wave iterations without a range check, so
    # every iteration must land inside the row. wave32+pk8 needs feat_dim % 256;
    # wave64 would need 512, hence pinning the wave size too.
    assert wave_size == 32, f"gfx1250 wave32 only, got wave_size={wave_size}"
    assert feat_dim % 256 == 0, f"feat_dim must be a multiple of 256, got {feat_dim}"

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

    ``c`` carries the layout flags, SSA constants/types, the i64 row bases
    (``payload_base``, ``hidden_base``) with their per-row byte strides
    (``payload_bytes_per_row``, ``feat_bytes_per_row``, ``feat_row_i32``), the
    scale resource, the intra-warp mapping (``block_in_wave``, ``lane_in_block``,
    ``is_block_lead``), and ``c.dests``: a list of destination namespaces, each
    with ``payload_row_i32`` and ``scale_row_dword_base``. The current callers
    pass a single destination; keeping this as a list lets a future caller
    experiment with multi-destination scattering without changing the quant math.
    Shared verbatim by both stage1 and stage2; only the preamble that computes
    ``c.dests`` differs.

    With ``c.prequantized`` the source row is already an MX payload plus a
    separate e8m0 row (``c.src_scale_base`` / ``c.src_scale_bytes_per_row``, and
    ``c.feat_bytes_per_row`` sized for the payload, not for bf16): the first pass
    loads what it would otherwise have computed, and the store pass is unchanged.
    """
    i32 = c.i32
    f32 = c.f32
    mx_group_base = getattr(c, "mx_group_base", None)
    if mx_group_base is None:
        mx_group_base = arith.constant(0, type=i32)

    # i64 row base: at >64k tokens a grouped row index times model_dim exceeds the
    # 32-bit buffer voffset (contiguous_m * feat_dim > 2**32), corrupting the store.
    payload_bytes_per_row = c.payload_bytes_per_row
    payload_dests = getattr(c, "payload_dests", c.dests)
    # A zero-length descriptor is how a dead destination is switched off, so
    # the quant pass stays one basic block.
    payload_records = getattr(c, "payload_num_records", payload_bytes_per_row)
    dst_payload = []
    for dst in payload_dests:
        row_addr = (
            c.payload_base + fx.Uint64(dst.payload_row_i32) * payload_bytes_per_row
        )
        dst_payload.append(
            buffer_ops.create_buffer_resource_from_addr(
                row_addr, num_records_bytes=payload_records
            )
        )

    # The hidden/payload SOURCE row stays on the width-agnostic buffer_ops V#
    # (one descriptor per buffer, per-access vec_width). Its pk8 dwordx4 load
    # (16 B/lane) through a lane-unit ptr_buf_tensor would force a copy-atom +
    # fragment whose register bundle survives into the store pass, inflating
    # VGPRs by up to +50 on the fp4/fp8 pk8 modules. Only the single-width
    # scatter *stores* below are on the layout API.
    hidden_row_addr = c.hidden_base + fx.Uint64(c.feat_row_i32) * c.feat_bytes_per_row
    hidden_rsrc = buffer_ops.create_buffer_resource_from_addr(
        hidden_row_addr, num_records_bytes=c.feat_bytes_per_row
    )
    feat_elem_base = arith.constant(0, type=i32)

    # Pre-quantized source: load what the quant pass would have computed, so the
    # store pass below stays shared. That destination arithmetic has already
    # moved once (to WMMA-contiguous); a second copy of it would fail silently,
    # by writing to the wrong offset.
    prequantized = getattr(c, "prequantized", False)
    src_scale_rsrc = None
    if const_expr(prequantized):
        c2_i32 = arith.constant(2, type=i32)
        # Its own stride: a sender pads the e8m0 row (mori pads to 128 B so every
        # token's TDM run starts aligned), so it is a build constant rather than
        # feat_dim // 32.
        src_scale_rsrc = buffer_ops.create_buffer_resource_from_addr(
            c.src_scale_base + fx.Uint64(c.feat_row_i32) * c.src_scale_bytes_per_row,
            num_records_bytes=c.src_scale_bytes_per_row,
        )

    def _mx_block_of(it):
        return (mx_group_base + arith.constant(it, type=i32)) * arith.constant(
            c.mx_blocks_per_wave_iter, type=i32
        ) + c.block_in_wave

    # Optional TDM staging of the hidden row: ``chunk_prefetch`` issues chunk
    # n+1's DMA before chunk n is converted, so the transfer runs under the
    # convert pipeline. The store pass still sees every block at once.
    n_chunks = getattr(c, "hidden_chunks", 1)
    assert c.block_iters % n_chunks == 0
    iters_per_chunk = c.block_iters // n_chunks
    chunk_prefetch = getattr(c, "chunk_prefetch", None)
    hidden_lds_load = getattr(c, "hidden_lds_load", None)

    def _emit_hidden_load(it):
        """This iteration's 8 bf16 (one aligned dwordx4)."""
        col_base = (
            _mx_block_of(it) * arith.constant(32, type=i32)
            + c.lane_in_block * c.c_elems_per_lane
        )
        if const_expr(hidden_lds_load is not None):
            chunk = it // iters_per_chunk
            chunk_elems = c.mx_blocks_per_wave_iter * iters_per_chunk * 32
            return hidden_lds_load(
                c.hidden_lds_idx,
                c.hidden_lds_row_off
                + arith.constant((chunk % 2) * c.hidden_slot_bytes, type=i32)
                + (col_base - arith.constant(chunk * chunk_elems, type=i32))
                * arith.constant(2, type=i32),
            )
        return buffer_ops.buffer_load(
            hidden_rsrc,
            (feat_elem_base + col_base) >> c.c1_i32,
            vec_width=4,
            dtype=i32,
        )

    quant_results = []
    for it in range_constexpr(c.block_iters):
        if const_expr(chunk_prefetch is not None and it % iters_per_chunk == 0):
            chunk_prefetch(it // iters_per_chunk)
        # MX block (along K) this lane works on this iteration.
        mx_block = _mx_block_of(it)
        if const_expr(prequantized):
            # This lane's payload bytes sit at exactly the offset the store pass
            # writes them to, so both sides share the expression and cannot drift.
            # fp8: 8 B/lane = 2 dwords; fp4: 4 B/lane = 1 dword -- the same types
            # the pk8 converts produce, so the store needs no special case.
            byte_off = (
                mx_block * c.c_payload_bytes_per_block
                + c.lane_in_block * c.c_payload_bytes_per_lane
            )
            payload_val = buffer_ops.buffer_load(
                hidden_rsrc,
                byte_off >> c2_i32,
                vec_width=c.payload_dwords_per_lane,
                dtype=i32,
            )
            # Every lane of an MX block loads the same e8m0 byte (one cache line)
            # and only the lead lane stores it. Unconditional on purpose: a value
            # defined inside an scf.if would not dominate the store pass below.
            e8m0_byte = buffer_ops.buffer_load(
                src_scale_rsrc, mx_block, vec_width=1, dtype=T.i8
            )
            # Widen so the store pass's trunci sees the same i32 it does on the
            # quant path, where e8m0 comes out of emit_mx_e8m0_scale as i32.
            e8m0_scale = arith.extui(i32, ArithValue(e8m0_byte))
        elif const_expr(c.use_pk8):
            # gfx1250 native pk8: 8 contiguous bf16 cols this lane.
            # col_base = mx_block*32 + lane_in_block*8.
            col_base = (
                mx_block * arith.constant(32, type=i32)
                + c.lane_in_block * c.c_elems_per_lane
            )
            # 2 bf16/dword -> 4 dwords; one aligned dwordx4 = 8 bf16.
            dwords4 = _emit_hidden_load(it)
            bf16x8 = fx.Vector(dwords4).bitcast(fx.Numeric.from_ir_type(T.bf16))

            # per-block amax over this lane's 8 elems, then a butterfly
            # shuffle_xor across the block's 4 lanes.
            block_amax = _emit_pk8_lane_amax(bf16x8, c)
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
            hidden_dword = (feat_elem_base + col_base) >> c.c1_i32  # 2 bf16/dword

            dword_raw = buffer_ops.buffer_load(
                hidden_rsrc, hidden_dword, vec_width=1, dtype=i32
            )
            vec2_f32_ty = T.vec(ELEMS_PER_LANE, f32)
            bf16_pair = fx.Vector.from_elements(
                [dword_raw], fx.Numeric.from_ir_type(i32)
            ).bitcast(fx.Numeric.from_ir_type(T.bf16))
            f32_pair = bf16_pair.extf(vec2_f32_ty)
            x0 = fx.Vector(f32_pair)[0]
            x1 = fx.Vector(f32_pair)[1]

            # per-block amax: max over this lane's 2 elems, then a butterfly
            # shuffle_xor across the block's 16 lanes.
            abs0 = llvm.call_intrinsic(f32, "llvm.fabs.f32", [x0.ir_value()], [], [])
            abs1 = llvm.call_intrinsic(f32, "llvm.fabs.f32", [x1.ir_value()], [], [])
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
                    block_scale_f32 = (ArithValue(e8m0_scale) << c.c23_i32).bitcast(f32)
                    packed = rocdl.cvt_scalef32_pk_fp8_f32(
                        i32,
                        _raw(c.c0_i32),
                        _raw(x0),
                        _raw(x1),
                        _raw(block_scale_f32),
                        0,
                    )
                else:
                    recip_scale = ((c.c254_i32 - e8m0_scale) << c.c23_i32).bitcast(f32)
                    scaled0 = ArithValue(x0) * recip_scale
                    scaled1 = ArithValue(x1) * recip_scale
                    # v_cvt_pk_fp8_f32: 2 f32 -> 2 fp8 bytes in word 0.
                    packed = rocdl.cvt_pk_fp8_f32(i32, scaled0, scaled1, c.c0_i32, 0)
                payload_val = arith.trunci(T.i16, ArithValue(packed))  # 2 fp8 B
            else:
                if const_expr(c.use_native):
                    block_scale_f32 = (ArithValue(e8m0_scale) << c.c23_i32).bitcast(f32)
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
                    recip_scale = ((c.c254_i32 - e8m0_scale) << c.c23_i32).bitcast(f32)
                    nib0 = emit_f32_to_e2m1(ArithValue(x0) * recip_scale)
                    nib1 = emit_f32_to_e2m1(ArithValue(x1) * recip_scale)
                    packed_byte = ArithValue(nib0) | (ArithValue(nib1) << c.c4_i32)
                    payload_val = arith.trunci(T.i8, packed_byte)  # 1 fp4x2 B

        quant_results.append((mx_block, payload_val, e8m0_scale))

    if const_expr(getattr(c, "scale_vec4", False)):
        # Payload per block as usual, but the row's e8m0 goes out 16 B at a
        # time: the store pass has every block's result live, so two adjacent
        # blocks' dwords pair into one dwordx4.
        for mx_block, payload_val, _e8m0 in quant_results:
            _emit_payload_stores(c, dst_payload, payload_val, mx_block)
        for i in range_constexpr(0, len(quant_results), 2):
            _emit_row_major_scale_vec4(
                c, quant_results[i][0], quant_results[i][2], quant_results[i + 1][2]
            )
        return

    # Stores are a separate pass so the quant pass above stays one basic block and
    # its loads can cluster. That only works while the quant pass is branch-free:
    # a value defined inside a guarded region does not dominate this loop, so
    # putting a per-iteration guard back above would produce invalid IR.
    # One quant result (payload_val + e8m0_scale) is written to every destination
    # row in ``c.dests``.
    for mx_block, payload_val, e8m0_scale in quant_results:
        _emit_quant_result_stores(c, dst_payload, mx_block, payload_val, e8m0_scale)


def _emit_quant_result_stores(c, dst_payload, mx_block, payload_val, e8m0_scale):
    """Write one MX-block result to every destination in ``c.dests``.

    Payload stores are unpredicated. The scale stores share one lead-lane guard
    covering all destinations, so a 6-dest kernel emits one dispatch per MX
    block instead of six.
    """
    # The block-scale's dword/byte position depends only on ``mx_block``.
    scale_dword = fx.Uint32(mx_block) // fx.Uint32(c.c4_i32)
    byte_in_dword = mx_block - scale_dword * c.c4_i32
    e8m0_byte = arith.trunci(T.i8, e8m0_scale)
    _emit_payload_stores(c, dst_payload, payload_val, mx_block)

    row_major_scale = getattr(c, "row_major_scale", False)
    if const_expr(row_major_scale and getattr(c, "scale_pack_dwords", False)):
        _emit_row_major_scale_dwords(c, mx_block, scale_dword, e8m0_scale)
        return

    # one e8m0 byte per block, written by the block's lead lane. This plain
    # helper is not AST-rewritten, so the runtime guard goes through a local
    # @flyc.jit dispatch (a bare Python ``if`` would eval the dynamic Boolean
    # as a host bool).
    def _store_lead_scale():
        for dst in c.dests:
            if const_expr(row_major_scale):
                # (row, feat_dim//32) bytes: each row's scales contiguous, so a
                # warp's six destination writes stay within six rows instead of
                # touching a fresh cache line per MX block.
                dst_scale_byte = (
                    dst.payload_row_i32 * c.c_scale_bytes_per_row + mx_block
                )
            else:
                dst_scale_dword = (
                    dst.scale_row_dword_base + scale_dword * c.c_wmma_rep * 16
                )
                dst_scale_byte = dst_scale_dword * c.c4_i32 + byte_in_dword
            c.scale_t[dst_scale_byte] = e8m0_byte

    @flyc.jit
    def _dispatch_lead_scale():
        if c.is_block_lead:
            _store_lead_scale()

    _dispatch_lead_scale()


def _emit_payload_stores(c, dst_payload, payload_val, mx_block):
    """One MX block's payload bytes to every payload destination.

    The MX payload row stays on the width-agnostic buffer_ops V# (per-access
    byte offset). A lane-unit ptr_buf_tensor store is correct and cheaper on
    most rows but perturbs VGPR alloc by +1..+4 on the fp4/fp8 pk8 modules.
    """
    payload_byte_off = (
        mx_block * c.c_payload_bytes_per_block
        + c.lane_in_block * c.c_payload_bytes_per_lane
    )
    payload_cache = getattr(c, "payload_cache_modifier", 0)
    for rsrc in dst_payload:
        buffer_ops.buffer_store(
            payload_val,
            rsrc,
            payload_byte_off,
            cache_modifier=payload_cache,
            offset_is_bytes=True,
        )


def _pack_block_group_dword(c, e8m0_scale):
    """The dword of e8m0 bytes for the 4 MX blocks around this lane's block.

    Only the lane holding block 4k ends up with them in the right order; the
    caller predicates on that.
    """
    i32 = c.i32
    v = ArithValue(arith.andi(e8m0_scale, arith.constant(0xFF, type=i32)))
    p1 = ArithValue(v.shuffle_xor(c.c4_i32, c.c_wave))
    half = v | (p1 << arith.constant(8, type=i32))
    p2 = ArithValue(half.shuffle_xor(arith.constant(8, type=i32), c.c_wave))
    return half | (p2 << arith.constant(16, type=i32))


def _emit_row_major_scale_vec4(c, mx_block_lo, e8m0_lo, e8m0_hi):
    """Two blocks' worth of row-major e8m0 (16 B) in one dwordx4 store.

    ``mx_block_lo`` is the low iteration's block for this lane. Lane 4k*4 holds
    the low dword of each pair and picks up its partner's upper dword with one
    more xor-shuffle, so a single lane writes all 16 bytes.

    Like the payload, the packed scale stays on the width-agnostic buffer_ops V#:
    ``c.scale_t`` is a byte view, and this store is a dword-indexed dwordx4.
    """
    i32 = c.i32
    lo = _pack_block_group_dword(c, e8m0_lo)
    hi = _pack_block_group_dword(c, e8m0_hi)
    lo_peer = ArithValue(lo).shuffle_xor(arith.constant(16, type=i32), c.c_wave)
    hi_peer = ArithValue(hi).shuffle_xor(arith.constant(16, type=i32), c.c_wave)
    quad = fx.Vector.from_elements(
        [lo, lo_peer, hi, hi_peer], fx.Numeric.from_ir_type(i32)
    )
    scale_dword = fx.Uint32(mx_block_lo) // fx.Uint32(c.c4_i32)

    def _store_quad():
        for dst in c.dests:
            buffer_ops.buffer_store(
                quad,
                c.scale_rsrc,
                dst.payload_row_i32 * c.c_scale_dwords_per_row + scale_dword,
            )

    # Only lane 0 of the wave stores: it holds blocks 4k..4k+3 of both halves.
    is_wave_lead = arith.andi(
        fx.Int32(c.block_in_wave) == c.c0_i32,
        c.is_block_lead,
    )

    @flyc.jit
    def _dispatch_quad():
        if is_wave_lead:
            _store_quad()

    _dispatch_quad()


def _emit_row_major_scale_dwords(c, mx_block, scale_dword, e8m0_scale):
    """Row-major e8m0 as one dword per 4 MX blocks instead of 4 byte stores.

    Only the lane holding block 4k ends up with the bytes in the right order,
    hence the ``block_in_wave % 4 == 0`` predicate.
    """
    i32 = c.i32
    packed = _pack_block_group_dword(c, e8m0_scale)

    def _store_packed():
        for dst in c.dests:
            # buffer_store scales the offset by the stored type, so index dwords.
            buffer_ops.buffer_store(
                packed,
                c.scale_rsrc,
                dst.payload_row_i32 * c.c_scale_dwords_per_row + scale_dword,
            )

    group_lead = arith.andi(
        arith.andi(fx.Int32(c.block_in_wave), arith.constant(3, type=i32)) == c.c0_i32,
        c.is_block_lead,
    )

    @flyc.jit
    def _dispatch_packed():
        if group_lead:
            _store_packed()

    _dispatch_packed()


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
    use_g2l: bool = False,
    weight_dtype: str = "bf16",
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
    w_fx = {"bf16": fx.BFloat16, "f16": fx.Float16}[weight_dtype]

    base_tag = "baseptr" if use_expert_row_base else f"basem{max_m}"
    g2l_tag = f"_g2l_{weight_dtype}" if use_g2l else ""
    module_name = format_kernel_name(
        f"moe_fused_route_quant_scatter_md{model_dim}_tk{topk}_r{wmma_rep}"
        f"_{quant_mode}_{L.native_tag}_{base_tag}{g2l_tag}"
    )

    @flyc.kernel(name=module_name, known_block_size=[BLOCK_THREADS, 1, 1])
    def fused_kernel(
        topk_ids: fx.Pointer,  # (numel,) int32 (GLOBAL expert ids when use_g2l)
        counter: fx.Pointer,  # (E,) int32, init 0
        topids_to_rows: fx.Pointer,  # (numel,) int32 out
        hidden: fx.Pointer,  # (token_num*model_dim,) bf16
        grouped_payload: fx.Pointer,  # (n_rows*payload_bytes_per_row,) uint8 out
        grouped_scale: fx.Pointer,  # preshuffled e8m0 out
        expert_row_base: fx.Pointer,  # (E,) int32 per-expert dst row base
        numel: Int32,
        g2l_lut: fx.Pointer,  # (E_global,) int32 global->local, sentinel=n_buckets
        weight_in: fx.Pointer,  # (numel,) f32 route weights in (used iff use_g2l)
        gather_w: fx.Pointer,  # (numel,) weight_dtype out; kept->cast, drops->0
        n_buckets: Int32,  # sentinel value (== dropped) / local expert count
    ):
        """Write masked or contiguous ``(Mtile, K//128, wmma_rep, 16, 4)`` scales."""
        i32 = T.i32
        f32 = T.f32
        wdt = {"bf16": T.bf16, "f16": T.f16}[weight_dtype]

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
        _c_model_dim = arith.constant(model_dim, type=i32)
        _c_payload_bytes_per_row = arith.constant(payload_bytes_per_row, type=i32)
        c_payload_bytes_per_block = arith.constant(payload_bytes_per_block, type=i32)
        c_payload_bytes_per_lane = arith.constant(payload_bytes_per_lane, type=i32)
        c_dst_scale_dwords_per_row = arith.constant(dst_scale_dwords_per_row, type=i32)
        c_wmma_rep = arith.constant(wmma_rep, type=i32)
        c_rows_per_tile = arith.constant(rows_per_tile, type=i32)
        c_lanes_per_block = arith.constant(lanes_per_mx_block, type=i32)
        c_elems_per_lane = arith.constant(elems_per_lane, type=i32)
        c_max_m = arith.constant(max_m, type=i32)

        tid = fx.Uint32(fx.thread_idx.x)
        bid = fx.Uint32(fx.block_idx.x)

        warp_in_block = tid // c_wave
        lane = tid - warp_in_block * c_wave  # tid % wave_size
        route = bid * arith.constant(warps_per_block, type=i32) + warp_in_block

        route_in_range = fx.Uint32(route) < fx.Uint32(numel)
        if route_in_range:
            # expert id for this route (uniform across the warp)
            expert = fx.Uint32(ptr_buf_tensor(topk_ids)[route])

            # EP global->local remap (warp-uniform), replacing the host
            # cumsum/index/eq/where/masked_fill chain. Dropped (non-local) routes
            # address bucket 0 to keep the atomic in bounds but claim no slot, so
            # masked_m -- and the grouped GEMM's row count with it -- covers only
            # this rank's own routes. They are tagged with DROPPED_ROUTE_ROW, skip
            # the quant+scatter below, and get a zero gather weight.
            is_drop = None
            if const_expr(use_g2l):
                le = ptr_buf_tensor(g2l_lut)[expert]
                is_drop = le == fx.Uint32(n_buckets)
                expert = fx.Uint32(is_drop.select(c0_i32, le))
                # Fused weight cast+mask (warp-uniform: every lane writes the same
                # value to gather_w[route], redundant but race-free). Reads f32
                # weight_in and writes weight_dtype (kept -> cast, dropped -> 0),
                # folding the host topk_weight.to(bf16) copy + masked_fill.
                w_f32 = ptr_buf_tensor(weight_in, fx.Float32)[route]
                w_cast = arith.trunc_f(wdt, w_f32)
                w_out = is_drop.select(arith.constant(0.0, type=wdt), w_cast)
                ptr_buf_tensor(gather_w, w_fx)[route] = w_out

            # Lane 0 claims the within-expert slot via atomicAdd, then broadcasts
            # it to the warp. Single-token pow2 cases use the dedicated st_ksplit
            # kernel, so the generic path does not need a runtime numel==topk
            # branch here.
            if const_expr(use_g2l):
                # Dropped routes add 0, so they take no row in the grouped layout.
                slot_incr = is_drop.select(c0_i32, c1_i32)
            else:
                slot_incr = c1_i32
            slot_on_lane0 = arith.constant(0, type=i32)
            if lane == 0:
                counter_addr = fx.Int64(ptrtoint(counter)) + fx.Int64(expert) * 4
                counter_ptr = create_llvm_ptr(counter_addr)
                counter_ptr = (
                    counter_ptr._value
                    if hasattr(counter_ptr, "_value")
                    else counter_ptr
                )
                slot_on_lane0 = fx.Uint32(
                    llvm.AtomicRMWOp(
                        llvm.AtomicBinOp.add,
                        counter_ptr,
                        _raw(slot_incr),
                        llvm.AtomicOrdering.monotonic,
                        syncscope="agent",
                        alignment=4,
                    ).result
                )
            # readlane needs raw ir.Value operands in this FlyDSL build (the
            # /workspace/FlyDSL example's auto-unwrap + T.i32() are a newer API).
            slot = fx.Uint32(rocdl.readlane(i32, _raw(slot_on_lane0), _raw(c0_i32)))

            # Destination row = per-expert base + within-expert slot. Masked
            # layout fuses the former Python-side arange(E)*max_m into this
            # kernel; contiguous-M still loads starts[e] from expert_row_base.
            if const_expr(use_expert_row_base):
                row_base = fx.Uint32(ptr_buf_tensor(expert_row_base)[expert])
            else:
                row_base = expert * c_max_m
            grouped_row = slot + row_base
            if const_expr(topk_is_pow2):
                token = route >> c_topk_shift
            else:
                token = fx.Uint32(route) // fx.Uint32(c_topk)

            # topids_to_rows[route] = grouped_row (lane 0 only; warp-uniform value).
            # A dropped route claimed no slot, so the row it computed belongs to
            # the bucket-0 route holding that slot -- store the sentinel instead.
            if const_expr(use_g2l):
                row_out = arith.select(
                    _raw(is_drop),
                    arith.constant(DROPPED_ROUTE_ROW, type=i32),
                    _raw(grouped_row),
                )
            else:
                row_out = grouped_row
            if lane == 0:
                ptr_buf_tensor(topids_to_rows)[route] = row_out

            def _emit_row_quant_scatter():
                # --- per-row scale-preshuffle geometry (uniform; from the *global*
                #     grouped_row so the same math serves both output layouts). Since
                #     every expert base is a multiple of rows_per_tile, tiling by the
                #     global row reproduces the per-expert byte layout exactly. ---
                scale_tile = fx.Uint32(grouped_row) // fx.Uint32(c_rows_per_tile)
                row_in_tile = grouped_row - scale_tile * c_rows_per_tile
                wmma_row = fx.Uint32(row_in_tile) // fx.Uint32(c16_i32)
                row_lane16 = row_in_tile - wmma_row * c16_i32
                scale_row_dword_base = (
                    scale_tile * c_dst_scale_dwords_per_row * c16_i32
                    + wmma_row * c16_i32
                    + row_lane16
                )

                payload_base = fx.Int64(ptrtoint(grouped_payload))
                hidden_base = fx.Int64(ptrtoint(hidden))
                scale_t = ptr_buf_tensor(grouped_scale, fx.Int8)

                # this lane's position inside its MX block group
                block_in_wave = fx.Uint32(lane) // fx.Uint32(c_lanes_per_block)
                lane_in_block = lane - block_in_wave * c_lanes_per_block
                is_block_lead = lane_in_block == c0_i32

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
                            payload_row_i32=grouped_row,
                            scale_row_dword_base=scale_row_dword_base,
                        )
                    ],
                    payload_base=payload_base,
                    payload_bytes_per_row=payload_bytes_per_row,
                    hidden_base=hidden_base,
                    feat_bytes_per_row=model_dim * 2,
                    feat_row_i32=token,
                    scale_t=scale_t,
                )
                _emit_quant_block_loop(c)

            if const_expr(use_g2l):
                # Scattering a dropped route would overwrite the payload of the
                # route that owns that row. is_drop is warp-uniform, so the whole
                # warp branches together and the amax shuffles stay well defined.
                is_kept = le != fx.Uint32(n_buckets)
                if is_kept:
                    _emit_row_quant_scatter()
            else:
                _emit_row_quant_scatter()

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
        g2l_lut: fx.Pointer,
        weight_in: fx.Pointer,
        gather_w: fx.Pointer,
        n_buckets: fx.Int32,
        grid_blocks: fx.Int32,
        stream: fx.Stream = fx.Stream(None),  # noqa: B008
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
            g2l_lut,
            weight_in,
            gather_w,
            n_buckets,
        ).launch(
            grid=(grid_x, 1, 1),
            block=(BLOCK_THREADS, 1, 1),
            stream=stream,
        )

    launch_fused.compile_hints = {
        "llvm_options": {
            "amdgpu-kernarg-preload": AITER_FLYDSL_KERNARG_PRELOAD,
            "amdgpu-kernarg-preload-count": AITER_FLYDSL_KERNARG_PRELOAD_COUNT,
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
    module_name = format_kernel_name(
        f"moe_fused_route_quant_scatter_stks_md{model_dim}_tk{topk}_r{wmma_rep}"
        f"_{quant_mode}_{L.native_tag}_{base_tag}"
    )

    @flyc.kernel(name=module_name, known_block_size=[block_threads, 1, 1])
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
        """Write masked or contiguous ``(Mtile, K//128, wmma_rep, 16, 4)`` scales."""
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
        _c_payload_bytes_per_row = arith.constant(payload_bytes_per_row, type=i32)
        c_payload_bytes_per_block = arith.constant(payload_bytes_per_block, type=i32)
        c_payload_bytes_per_lane = arith.constant(payload_bytes_per_lane, type=i32)
        c_dst_scale_dwords_per_row = arith.constant(dst_scale_dwords_per_row, type=i32)
        c_wmma_rep = arith.constant(wmma_rep, type=i32)
        c_rows_per_tile = arith.constant(rows_per_tile, type=i32)
        c_lanes_per_block = arith.constant(lanes_per_mx_block, type=i32)
        c_elems_per_lane = arith.constant(elems_per_lane, type=i32)
        c_max_m = arith.constant(max_m, type=i32)

        tid = fx.Uint32(fx.thread_idx.x)
        bid = fx.Uint32(fx.block_idx.x)
        k_group = fx.Uint32(fx.block_idx.y)

        warp_in_block = tid // c_wave
        lane = tid - warp_in_block * c_wave
        route = bid * arith.constant(warps_per_block, type=i32) + warp_in_block

        route_in_range = fx.Uint32(route) < fx.Uint32(numel)
        if route_in_range:
            expert = fx.Uint32(ptr_buf_tensor(topk_ids)[route])

            # torch.topk over experts returns distinct expert indices for one
            # token. Therefore each selected expert receives exactly one route:
            # slot=0, counter[expert]=1. This is the key small-token fast path;
            # the generic kernel remains available for non-single-token cases.
            slot = fx.Uint32(0)

            is_lane0 = lane == c0_i32
            is_k0 = k_group == c0_i32
            is_lane0_k0 = is_lane0 & is_k0
            if is_lane0_k0:
                ptr_buf_tensor(counter)[expert] = c1_i32

            if const_expr(use_expert_row_base):
                row_base = fx.Uint32(ptr_buf_tensor(expert_row_base)[expert])
            else:
                row_base = expert * c_max_m
            grouped_row = slot + row_base

            if is_lane0_k0:
                ptr_buf_tensor(topids_to_rows)[route] = grouped_row

            scale_tile = fx.Uint32(grouped_row) // fx.Uint32(c_rows_per_tile)
            row_in_tile = grouped_row - scale_tile * c_rows_per_tile
            wmma_row = fx.Uint32(row_in_tile) // fx.Uint32(c16_i32)
            row_lane16 = row_in_tile - wmma_row * c16_i32
            scale_row_dword_base = (
                scale_tile * c_dst_scale_dwords_per_row * c16_i32
                + wmma_row * c16_i32
                + row_lane16
            )

            payload_base = fx.Int64(ptrtoint(grouped_payload))
            hidden_base = fx.Int64(ptrtoint(hidden))
            scale_t = ptr_buf_tensor(grouped_scale, fx.Int8)

            block_in_wave = fx.Uint32(lane) // fx.Uint32(c_lanes_per_block)
            lane_in_block = lane - block_in_wave * c_lanes_per_block
            is_block_lead = lane_in_block == c0_i32

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
                        payload_row_i32=grouped_row,
                        scale_row_dword_base=scale_row_dword_base,
                    )
                ],
                payload_base=payload_base,
                payload_bytes_per_row=payload_bytes_per_row,
                hidden_base=hidden_base,
                feat_bytes_per_row=model_dim * 2,
                feat_row_i32=c0_i32,
                scale_t=scale_t,
            )
            _emit_quant_one_k_group(c, k_group)

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
        stream: fx.Stream = fx.Stream(None),  # noqa: B008
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
            "amdgpu-kernarg-preload": AITER_FLYDSL_KERNARG_PRELOAD,
            "amdgpu-kernarg-preload-count": AITER_FLYDSL_KERNARG_PRELOAD_COUNT,
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

    @flyc.kernel(name=module_name, known_block_size=[BLOCK_THREADS, 1, 1])
    def fused_kernel(
        grouped_in: fx.Pointer,  # (n_rows*feat_dim,) bf16
        grouped_payload: fx.Pointer,  # (n_rows*payload_bytes_per_row,) uint8 out
        grouped_scale: fx.Pointer,  # preshuffled e8m0 out
        masked_m: fx.Pointer,  # (E,) int32 valid row count (read iff skip_padding)
        n_rows: Int32,
        max_m: Int32,
    ):
        """Write scales as ``(E, M//(wmma_rep*16), K//128, wmma_rep, 16, 4)``."""
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
        _c_feat_dim = arith.constant(feat_dim, type=i32)
        _c_payload_bytes_per_row = arith.constant(payload_bytes_per_row, type=i32)
        c_payload_bytes_per_block = arith.constant(payload_bytes_per_block, type=i32)
        c_payload_bytes_per_lane = arith.constant(payload_bytes_per_lane, type=i32)
        c_scale_dwords_per_row = arith.constant(scale_dwords_per_row, type=i32)
        c_dst_scale_dwords_per_row = arith.constant(dst_scale_dwords_per_row, type=i32)
        c_wmma_rep = arith.constant(wmma_rep, type=i32)
        c_rows_per_tile = arith.constant(rows_per_tile, type=i32)
        c_lanes_per_block = arith.constant(lanes_per_mx_block, type=i32)
        c_elems_per_lane = arith.constant(elems_per_lane, type=i32)

        tid = fx.Uint32(fx.thread_idx.x)
        bid = fx.Uint32(fx.block_idx.x)

        warp_in_block = tid // c_wave
        lane = tid - warp_in_block * c_wave  # tid % wave_size
        # one warp per grouped row (no routing: row == grouped row).
        row = bid * arith.constant(warps_per_block, type=i32) + warp_in_block

        row_in_range = fx.Uint32(row) < fx.Uint32(n_rows)
        if row_in_range:
            m = fx.Uint32(max_m)
            expert = fx.Uint32(row) // m
            slot = row - expert * m  # row within expert

            def _emit_row():
                # --- per-row scale-preshuffle geometry (uniform; row pos == slot) ---
                scale_tile = fx.Uint32(slot) // fx.Uint32(c_rows_per_tile)
                row_in_tile = slot - scale_tile * c_rows_per_tile
                wmma_row = fx.Uint32(row_in_tile) // fx.Uint32(c16_i32)
                row_lane16 = row_in_tile - wmma_row * c16_i32
                scale_row_dword_base = (
                    expert * (m * c_scale_dwords_per_row)
                    + scale_tile * c_dst_scale_dwords_per_row * c16_i32
                    + wmma_row * c16_i32
                    + row_lane16
                )

                payload_base = fx.Int64(ptrtoint(grouped_payload))
                hidden_base = fx.Int64(ptrtoint(grouped_in))
                scale_t = ptr_buf_tensor(grouped_scale, fx.Int8)

                block_in_wave = fx.Uint32(lane) // fx.Uint32(c_lanes_per_block)
                lane_in_block = lane - block_in_wave * c_lanes_per_block
                is_block_lead = lane_in_block == c0_i32

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
                            payload_row_i32=row,
                            scale_row_dword_base=scale_row_dword_base,
                        )
                    ],
                    payload_base=payload_base,
                    payload_bytes_per_row=payload_bytes_per_row,
                    hidden_base=hidden_base,
                    feat_bytes_per_row=feat_dim * 2,
                    feat_row_i32=row,
                    scale_t=scale_t,
                )
                _emit_quant_block_loop(c)

            if const_expr(skip_padding):
                # Skip padding rows: the masked GEMM never reads rows beyond
                # masked_m[expert], so quantizing them is pure waste. With high
                # capacity-factor padding this elides most of the work.
                valid = fx.Uint32(ptr_buf_tensor(masked_m)[expert])
                slot_valid = fx.Uint32(slot) < fx.Uint32(valid)
                if slot_valid:
                    _emit_row()
            else:
                _emit_row()

    @flyc.jit
    def launch_fused(
        grouped_in: fx.Pointer,
        grouped_payload: fx.Pointer,
        grouped_scale: fx.Pointer,
        masked_m: fx.Pointer,
        n_rows: fx.Int32,
        max_m: fx.Int32,
        grid_blocks: fx.Int32,
        stream: fx.Stream = fx.Stream(None),  # noqa: B008
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
            "amdgpu-kernarg-preload": AITER_FLYDSL_KERNARG_PRELOAD,
            "amdgpu-kernarg-preload-count": AITER_FLYDSL_KERNARG_PRELOAD_COUNT,
        },
    }

    return launch_fused


def build_moe_fused_quant_preshuffle_route_ksplit_module(
    feat_dim: int,
    wmma_rep: int,
    quant_mode: str = "fp4",
    source_topk: int = 0,
    remap_rows: bool = False,
    ksplit: bool = True,
    prequantized: bool = False,
    src_scale_bytes_per_row: int = 0,
):
    """Route-indexed grouped quant+preshuffle.

    Instead of launching over every row in the (E, max_m) capacity buffer and
    skipping padding, this kernel launches only over routed rows
    (``topids_to_rows``). When ``ksplit=True`` (designed for small token counts
    where grid.x is too small to saturate the GPU), the K-dimension is split
    across ``grid.y = block_iters`` so each workgroup handles one K-group.
    When ``ksplit=False`` (large token counts where grid.x already saturates),
    ``grid.y = 1`` and each warp loops over all K-groups internally.

    ``prequantized`` says ``grouped_in`` is an MX payload the sender already
    produced (fp8 or fp4 EP dispatch) and ``src_scale`` its row-major e8m0 rows,
    ``src_scale_bytes_per_row`` apart. The kernel then keeps the route gather and
    the scale preshuffle and drops only the quant -- the preshuffle cannot move
    to the sender, because its destination is a function of the grouped row THIS
    rank assigns, which no sender knows.
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

    if prequantized:
        assert src_scale_bytes_per_row >= L.scale_bytes_per_row, (
            f"src_scale_bytes_per_row {src_scale_bytes_per_row} cannot hold "
            f"{L.scale_bytes_per_row} e8m0 bytes for feat_dim {feat_dim}"
        )
    # The payload row IS the source row here, so the loop's bounds and its
    # per-lane offsets both come off the payload geometry.
    src_bytes_per_row = payload_bytes_per_row if prequantized else feat_dim * 2
    payload_dwords_per_lane = payload_bytes_per_lane // 4
    if prequantized:
        assert payload_bytes_per_lane % 4 == 0, (
            f"prequantized load is dword-wide; {quant_mode} gives "
            f"{payload_bytes_per_lane} B/lane"
        )

    source_tag = f"srctk{source_topk}" if source_topk > 0 else "srcrow"
    remap_tag = "_remap" if remap_rows else ""
    ksplit_tag = "" if ksplit else "_noKS"
    # In the name because it changes what the kernel READS, not just how fast:
    # two builds with the same feat_dim/quant_mode are not interchangeable.
    prequant_tag = f"_pq{src_scale_bytes_per_row}" if prequantized else ""
    source_topk_is_pow2 = source_topk > 0 and (source_topk & (source_topk - 1)) == 0
    source_topk_shift = source_topk.bit_length() - 1 if source_topk_is_pow2 else 0

    module_name = (
        f"moe_fused_quant_preshuffle_routeks_fd{feat_dim}_r{wmma_rep}"
        f"_{quant_mode}_{L.native_tag}_{source_tag}{remap_tag}{ksplit_tag}"
        f"{prequant_tag}"
    )

    @flyc.kernel(name=module_name, known_block_size=[BLOCK_THREADS, 1, 1])
    def fused_kernel(
        grouped_in: fx.Pointer,  # flat grouped activations
        grouped_payload: fx.Pointer,
        grouped_scale: fx.Pointer,
        topids_to_rows: fx.Pointer,  # (numel,) int32 global rows
        row_starts: fx.Pointer,  # (E,) int32, read iff remap_rows
        route_max_m: Int32,  # masked route stride, read iff remap_rows
        numel: Int32,
        num_valid_routes: fx.Pointer,  # (1,) int32: routes >= this are dead-tail padding (EP dynamic token count); skip
        src_scale: fx.Pointer,  # (tokens, src_scale_bytes_per_row) e8m0, read iff prequantized
    ):
        """Write masked or contiguous ``(Mtile, K//128, wmma_rep, 16, 4)`` scales."""
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
        _c_feat_dim = arith.constant(feat_dim, type=i32)
        _c_payload_bytes_per_row = arith.constant(payload_bytes_per_row, type=i32)
        c_payload_bytes_per_block = arith.constant(payload_bytes_per_block, type=i32)
        c_payload_bytes_per_lane = arith.constant(payload_bytes_per_lane, type=i32)
        c_dst_scale_dwords_per_row = arith.constant(dst_scale_dwords_per_row, type=i32)
        c_wmma_rep = arith.constant(wmma_rep, type=i32)
        c_rows_per_tile = arith.constant(rows_per_tile, type=i32)
        c_lanes_per_block = arith.constant(lanes_per_mx_block, type=i32)
        c_elems_per_lane = arith.constant(elems_per_lane, type=i32)
        c_source_topk = arith.constant(source_topk, type=i32)
        c_source_topk_shift = arith.constant(source_topk_shift, type=i32)

        tid = fx.Uint32(fx.thread_idx.x)
        bid = fx.Uint32(fx.block_idx.x)

        # One warp owns one route, so tid // wave is wave-invariant -- but the
        # backend cannot see that. readfirstlane says it, which keeps `route`
        # (and the row and destination descriptor derived from it) uniform;
        # otherwise every payload store goes through a readfirstlane waterfall.
        warp_in_block = fx.Uint32(rocdl.readfirstlane(i32, fx.Uint32(tid // c_wave)))
        lane = tid - warp_in_block * c_wave
        route = bid * arith.constant(warps_per_block, type=i32) + warp_in_block

        # Dynamic EP token count (capture-safe, no host sync): grid is launched over
        # the static numel routes, but routes >= num_valid_routes (= total_recv*topk)
        # are dead-tail padding rows of the dispatch buffer -> skip the gather+quant.
        # When truncation is disabled the caller passes a null pointer, which must
        # not be dereferenced, so the load is predicated rather than unconditional.
        num_valid_routes_is_set = fx.Int64(ptrtoint(num_valid_routes)) != 0
        valid_route_count = fx.Uint32(numel)
        if num_valid_routes_is_set:
            valid_route_count = fx.Uint32(ptr_buf_tensor(num_valid_routes)[c0_i32])
        route_in_range = fx.Uint32(route) < fx.Uint32(valid_route_count)
        rows_t = ptr_buf_tensor(topids_to_rows)
        # An EP route with no grouped row carries the negative DROPPED_ROUTE_ROW
        # sentinel: a destination row derived from it would overwrite a kept
        # route's payload. Dead-tail routes default to the same sentinel, so one
        # predicate covers both.
        row_raw = fx.Int32(DROPPED_ROUTE_ROW)
        if route_in_range:
            # Scalar (SMEM) load: `route` is wave-uniform, and landing the row in
            # an SGPR is what makes the per-row destination descriptor uniform.
            row_raw = fx.Int32(buf_scalar_load(rows_t, route))
        row_is_mapped = row_raw >= fx.Int32(0)
        if row_is_mapped:
            row = fx.Uint32(row_raw)
            if const_expr(remap_rows):
                m = fx.Uint32(route_max_m)
                expert = fx.Uint32(row) // m
                slot = row - expert * m
                # Overwrites `row`, so it has to stay scalar too.
                row = (
                    fx.Uint32(buf_scalar_load(ptr_buf_tensor(row_starts), expert))
                    + slot
                )
                is_lane0 = lane == c0_i32
                if const_expr(ksplit):
                    k_group = fx.Uint32(fx.block_idx.y)
                    is_k0 = k_group == c0_i32
                    store_cond = is_lane0 & is_k0
                else:
                    store_cond = is_lane0
                if store_cond:
                    rows_t[route] = row

            scale_tile = fx.Uint32(row) // fx.Uint32(c_rows_per_tile)
            row_in_tile = row - scale_tile * c_rows_per_tile
            wmma_row = fx.Uint32(row_in_tile) // fx.Uint32(c16_i32)
            row_lane16 = row_in_tile - wmma_row * c16_i32
            scale_row_dword_base = (
                scale_tile * c_dst_scale_dwords_per_row * c16_i32
                + wmma_row * c16_i32
                + row_lane16
            )

            if const_expr(source_topk > 0):
                if const_expr(source_topk_is_pow2):
                    source_row = route >> c_source_topk_shift
                else:
                    source_row = fx.Uint32(route) // fx.Uint32(c_source_topk)
                feat_row_i32 = source_row
            else:
                feat_row_i32 = row

            scale_t = ptr_buf_tensor(grouped_scale, fx.Int8)
            payload_base = fx.Int64(ptrtoint(grouped_payload))
            hidden_base = fx.Int64(ptrtoint(grouped_in))

            block_in_wave = fx.Uint32(lane) // fx.Uint32(c_lanes_per_block)
            lane_in_block = lane - block_in_wave * c_lanes_per_block
            is_block_lead = lane_in_block == c0_i32

            qc = SimpleNamespace(
                i32=i32,
                f32=f32,
                block_iters=1 if ksplit else block_iters,
                payload_base=payload_base,
                payload_bytes_per_row=payload_bytes_per_row,
                hidden_base=hidden_base,
                feat_bytes_per_row=src_bytes_per_row,
                feat_row_i32=feat_row_i32,
                prequantized=prequantized,
                payload_dwords_per_lane=payload_dwords_per_lane,
                src_scale_base=fx.Int64(ptrtoint(src_scale)),
                src_scale_bytes_per_row=src_scale_bytes_per_row,
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
                        payload_row_i32=row,
                        scale_row_dword_base=scale_row_dword_base,
                    )
                ],
                scale_t=scale_t,
            )
            if const_expr(ksplit):
                k_group_val = fx.Uint32(fx.block_idx.y)
                _emit_quant_one_k_group(qc, k_group_val)
            else:
                _emit_quant_block_loop(qc)

    _grid_y_dim = block_iters if ksplit else 1

    @flyc.jit
    def launch_fused(
        grouped_in: fx.Pointer,
        grouped_payload: fx.Pointer,
        grouped_scale: fx.Pointer,
        topids_to_rows: fx.Pointer,
        row_starts: fx.Pointer,
        route_max_m: fx.Int32,
        numel: fx.Int32,
        num_valid_routes: fx.Pointer,
        src_scale: fx.Pointer,
        grid_route_blocks: fx.Int32,
        stream: fx.Stream = fx.Stream(None),  # noqa: B008
    ):
        grid_x = arith.index_cast(T.index, grid_route_blocks)
        grid_y = arith.index_cast(T.index, arith.constant(_grid_y_dim, type=T.i32))
        fused_kernel(
            grouped_in,
            grouped_payload,
            grouped_scale,
            topids_to_rows,
            row_starts,
            route_max_m,
            numel,
            num_valid_routes,
            src_scale,
        ).launch(
            grid=(grid_x, grid_y, 1),
            block=(BLOCK_THREADS, 1, 1),
            stream=stream,
        )

    launch_fused.compile_hints = {
        "llvm_options": {
            "amdgpu-kernarg-preload": AITER_FLYDSL_KERNARG_PRELOAD,
            "amdgpu-kernarg-preload-count": AITER_FLYDSL_KERNARG_PRELOAD_COUNT,
        },
    }

    return launch_fused


def _token_multidest_scale_base(
    row, c_rows_per_tile, c_dst_scale_dwords_per_row, c16_i32
):
    """Return the preshuffled e8m0 dword base for one grouped row."""
    scale_tile = fx.Uint32(row) // fx.Uint32(c_rows_per_tile)
    row_in_tile = fx.Uint32(row) - scale_tile * c_rows_per_tile
    wmma_row = row_in_tile // fx.Uint32(c16_i32)
    row_lane16 = row_in_tile - wmma_row * c16_i32
    return (
        scale_tile * c_dst_scale_dwords_per_row * c16_i32
        + wmma_row * c16_i32
        + row_lane16
    )


def token_multidest_ksplit(
    feat_dim: int, wmma_rep: int, quant_mode: str, token_num: int
) -> int:
    """How many ways to split the row's K groups over ``grid.y``.

    One warp per token gives only ``token_num / warps_per_block`` blocks, so a
    decode-sized batch leaves most of the GPU idle; splitting K widens the grid
    without duplicating any work. Splitting further than that only multiplies
    the per-block route setup, and it costs the staging pipeline, so it stops
    once the grid is wide enough.
    """
    from aiter.jit.utils.chip_info import get_cu_num

    L = _quant_layout(feat_dim, quant_mode, wmma_rep)
    grid = -(-token_num // L.warps_per_block)
    target = (get_cu_num() or 256) * _TOKEN_MULTIDEST_BLOCKS_PER_CU
    want = -(-target // max(grid, 1))
    best = 1
    for n in range(1, min(want, _TOKEN_MULTIDEST_MAX_KSPLIT) + 1):
        if L.block_iters % n == 0:
            best = n
    return best


def token_multidest_tdm_chunks(
    feat_dim: int, wmma_rep: int, quant_mode: str, ksplit: int = 1
) -> int:
    """TDM staging depth: the deepest the row's geometry allows, up to the tuned one.

    A chunk has to cover a whole number of wave iterations and stay 16 B
    aligned, so the depth has to come from the row's divisors rather than a
    constant. A K-split block has too few iterations left to pay for a pipeline
    and the builder drops staging for it, so report that here too.
    """
    if ksplit > 1:
        return 0
    L = _quant_layout(feat_dim, quant_mode, wmma_rep)
    for n in range(min(_TOKEN_MULTIDEST_TDM_CHUNKS, L.block_iters), 0, -1):
        if L.block_iters % n == 0 and (feat_dim * 2) % (n * 16) == 0:
            return n
    return 0


def build_moe_token_multidest_quant_module(
    feat_dim: int,
    wmma_rep: int,
    topk: int,
    quant_mode: str = "fp4",
    row_major_scale: bool = False,
    tdm_hidden_chunks: int = _TOKEN_MULTIDEST_TDM_CHUNKS,
    ksplit: int = 1,
):
    """Quantize each token once and scatter the result to its ``topk`` routed rows.

    The route-indexed kernel quantizes the same hidden row once per route; this
    one uses a warp per token, computes the MX payload/e8m0 row once, then emits
    every destination. What it saves therefore grows with ``topk``, while what
    it costs -- one buffer descriptor per destination held live across the store
    pass -- grows with it too.
    """
    L = _quant_layout(feat_dim, quant_mode, wmma_rep)
    if not L.use_pk8:
        raise NotImplementedError("token multidest quant requires gfx1250 pk8")
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
    row_iters = L.block_iters
    amax_shuffle_dists = L.amax_shuffle_dists
    # Splitting the row's K groups over grid.y is safe because an MX block scale
    # covers 32 contiguous elements: the slices need no cross-block reduction.
    if row_iters % ksplit:
        raise ValueError(
            f"ksplit={ksplit} must divide the row's {row_iters} wave iterations"
        )
    block_iters = row_iters // ksplit
    # The staged chunks are a pipeline over the *block's* iterations, and a
    # split block has too few left to pay for one.
    if ksplit > 1:
        tdm_hidden_chunks = 0
    if tdm_hidden_chunks and (
        block_iters % tdm_hidden_chunks or (feat_dim * 2) % (tdm_hidden_chunks * 16)
    ):
        raise ValueError(
            f"tdm_hidden_chunks={tdm_hidden_chunks} must divide block_iters="
            f"{block_iters} and leave a 16 B-aligned chunk of {feat_dim * 2} B"
        )
    hidden_chunk_bytes = feat_dim * 2 // tdm_hidden_chunks if tdm_hidden_chunks else 0
    # Row-major e8m0 goes out packed: a dword per 4 MX blocks, widened to a
    # dwordx4 per 8 when the block count pairs up. Both need the pk8 geometry
    # (4 lanes per MX block) to assemble the bytes with xor-shuffles.
    scale_pack_dwords = row_major_scale and lanes_per_mx_block == 4
    scale_vec4 = scale_pack_dwords and block_iters % 2 == 0
    module_name = (
        f"moe_token_multidest_quant_k{topk}_fd{feat_dim}_r{wmma_rep}"
        f"_{quant_mode}_{L.native_tag}"
        f"{'_rmscale' if row_major_scale else ''}"
        f"{'_scpk' if scale_pack_dwords else ''}"
        f"{'_scv4' if scale_vec4 else ''}"
        f"{f'_hidtdm{tdm_hidden_chunks}' if tdm_hidden_chunks else ''}"
        f"{f'_ks{ksplit}' if ksplit > 1 else ''}"
    )

    @flyc.kernel(name=module_name, known_block_size=[BLOCK_THREADS, 1, 1])
    def token_multidest_kernel(
        hidden: fx.Pointer,
        grouped_payload: fx.Pointer,
        grouped_scale: fx.Pointer,
        topids_to_rows: fx.Pointer,
        token_num: Int32,
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
        c_payload_bytes_per_block = arith.constant(payload_bytes_per_block, type=i32)
        c_payload_bytes_per_lane = arith.constant(payload_bytes_per_lane, type=i32)
        c_dst_scale_dwords_per_row = arith.constant(dst_scale_dwords_per_row, type=i32)
        c_wmma_rep = arith.constant(wmma_rep, type=i32)
        c_rows_per_tile = arith.constant(rows_per_tile, type=i32)
        c_lanes_per_block = arith.constant(lanes_per_mx_block, type=i32)
        c_elems_per_lane = arith.constant(elems_per_lane, type=i32)
        c_scale_bytes_per_row = arith.constant(mx_blocks_per_row, type=i32)
        c_scale_dwords_per_row = arith.constant(mx_blocks_per_row // 4, type=i32)

        tid = fx.Uint32(fx.thread_idx.x)
        bid = fx.Uint32(fx.block_idx.x)
        warp_in_block = fx.Uint32(rocdl.readfirstlane(i32, fx.Uint32(tid // c_wave)))
        lane = tid - warp_in_block * c_wave
        token0 = bid * arith.constant(warps_per_block, type=i32)
        token = token0 + warp_in_block

        # Double-buffered hidden staging: [chunk slot 0 | chunk slot 1].
        hslot = warps_per_block * hidden_chunk_bytes if tdm_hidden_chunks else 0
        hidden_lds_load = None
        hidden_lds_idx = None
        hidden_lds_row_off = c0_i32
        chunk_prefetch = None
        if const_expr(tdm_hidden_chunks):
            h_lds = fx.SharedAllocator().allocate(2 * hslot)._ptr
            hidden_lds_idx = fx.index_cast(T.index, ptrtoint(h_lds))
            hidden_lds_load, _ = make_lds_copy_ops(128)
            hidden_lds_row_off = warp_in_block * arith.constant(
                hidden_chunk_bytes, type=i32
            )
            valid_rows = fx.Int32(token_num) - fx.Int32(token0)
            hg_base = fx.recast_iter(fx.Int8, hidden) + fx.Int64(token0) * (
                feat_dim * 2
            )

            def _issue_hidden(chunk):
                shape = (warps_per_block, hidden_chunk_bytes)
                tdm_ops.tensor_load_2d(
                    tdm_ops.make_tensor_descriptor_2d(
                        global_ptr=fx.Tensor(
                            fx.make_view(
                                hg_base + fx.Int64(chunk * hidden_chunk_bytes),
                                fx.make_layout(shape, (feat_dim * 2, 1)),
                            )
                        ),
                        lds_memref=fx.Tensor(
                            fx.make_view(
                                fx.add_offset(h_lds, (chunk % 2) * hslot),
                                fx.make_layout(shape, (hidden_chunk_bytes, 1)),
                            )
                        ),
                        global_offset=(0, 0),
                        tensor_shape=shape,
                        strides=(feat_dim * 2, 1),
                        tile_shape=shape,
                        elem_bytes=1,
                        num_warps=1,
                        oob_outer_bound=valid_rows,
                    )
                )

            is_loader = warp_in_block == fx.Uint32(c0_i32)

            def chunk_prefetch(chunk):
                # Prologue lands chunk 0 outright; every chunk then stages the
                # next one and leaves exactly it in flight, so the DMA runs
                # under this chunk's converts. Two slots mean chunk n+1 reuses
                # chunk n-1's, hence the WAR barrier before the issue; the RAW
                # barrier publishes chunk n, which only wave 0 waited on.
                if const_expr(chunk == 0) and is_loader:
                    _issue_hidden(0)
                    tdm_ops.tensor_wait(0)
                gpu.barrier()
                if is_loader and const_expr(chunk + 1 < tdm_hidden_chunks):
                    _issue_hidden(chunk + 1)
                if const_expr(chunk > 0):
                    if is_loader:
                        tdm_ops.tensor_wait(1 if chunk + 1 < tdm_hidden_chunks else 0)
                    gpu.barrier()

        # CTA-uniform guard (every launched block owns at least one token), so
        # the TDM staging barriers below are reached by all waves. A warp whose
        # token is past the end reads token 0 instead and has its destination
        # descriptors zero-sized, so its stores are dropped by the hardware
        # bounds check rather than by a branch.
        valid = token < fx.Uint32(token_num)
        token_eff = valid.select(token, fx.Uint32(c0_i32))
        pay_records = valid.select(
            arith.constant(payload_bytes_per_row, type=i32), c0_i32
        )
        # Sign-extended to i64 by the descriptor builder, so stay under 2 GiB.
        scale_records = valid.select(
            arith.constant(_SCALE_RSRC_MAX_BYTES, type=i32), c0_i32
        )
        if token0 < fx.Uint32(token_num):
            rows_t = ptr_buf_tensor(topids_to_rows)
            route0 = token_eff * arith.constant(topk, type=i32)
            # Scalar loads: the route is wave-uniform, and landing each row in
            # an SGPR is what keeps its destination descriptor uniform too.
            rows = [
                fx.Uint32(buf_scalar_load(rows_t, route0 + arith.constant(k, type=i32)))
                for k in range_constexpr(topk)
            ]
            scales = [
                _token_multidest_scale_base(
                    row, c_rows_per_tile, c_dst_scale_dwords_per_row, c16_i32
                )
                for row in rows
            ]

            block_in_wave = lane // fx.Uint32(c_lanes_per_block)
            lane_in_block = lane - block_in_wave * c_lanes_per_block
            qc = SimpleNamespace(
                i32=i32,
                f32=f32,
                block_iters=block_iters,
                payload_base=fx.Int64(ptrtoint(grouped_payload)),
                payload_bytes_per_row=payload_bytes_per_row,
                hidden_base=fx.Int64(ptrtoint(hidden)),
                feat_bytes_per_row=feat_dim * 2,
                feat_row_i32=token_eff,
                payload_num_records=pay_records,
                prequantized=False,
                payload_dwords_per_lane=payload_bytes_per_lane // 4,
                src_scale_base=fx.Int64(0),
                src_scale_bytes_per_row=0,
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
                is_block_lead=lane_in_block == c0_i32,
                payload_dests=[SimpleNamespace(payload_row_i32=row) for row in rows],
                dests=[
                    SimpleNamespace(payload_row_i32=row, scale_row_dword_base=sc)
                    for row, sc in zip(rows, scales)
                ],
                # Byte view for the unpacked e8m0 store; the packed dword /
                # dwordx4 row-major stores need a width-agnostic V# instead.
                # Both carry the same zero-on-invalid bound.
                scale_t=ptr_buf_tensor(
                    grouped_scale, fx.Int8, num_records_bytes=scale_records
                ),
                scale_rsrc=buffer_ops.create_buffer_resource_from_addr(
                    fx.Int64(ptrtoint(grouped_scale)),
                    num_records_bytes=scale_records,
                ),
                row_major_scale=row_major_scale,
                c_scale_bytes_per_row=c_scale_bytes_per_row,
                c_scale_dwords_per_row=c_scale_dwords_per_row,
                hidden_chunks=max(1, tdm_hidden_chunks),
                chunk_prefetch=chunk_prefetch,
                hidden_lds_load=hidden_lds_load,
                hidden_lds_idx=hidden_lds_idx,
                hidden_lds_row_off=hidden_lds_row_off,
                hidden_slot_bytes=hslot,
                scale_pack_dwords=scale_pack_dwords,
                scale_vec4=scale_vec4,
                mx_group_base=(
                    fx.Uint32(fx.block_idx.y) * arith.constant(block_iters, type=i32)
                    if const_expr(ksplit > 1)
                    else None
                ),
            )
            _emit_quant_block_loop(qc)

    @flyc.jit
    def launch_token_multidest(
        hidden: fx.Pointer,
        grouped_payload: fx.Pointer,
        grouped_scale: fx.Pointer,
        topids_to_rows: fx.Pointer,
        token_num: fx.Int32,
        grid_blocks: fx.Int32,
        stream: fx.Stream = fx.Stream(None),  # noqa: B008
    ):
        token_multidest_kernel(
            hidden, grouped_payload, grouped_scale, topids_to_rows, token_num
        ).launch(
            grid=(
                arith.index_cast(T.index, grid_blocks),
                arith.index_cast(T.index, arith.constant(ksplit, type=T.i32)),
                1,
            ),
            block=(BLOCK_THREADS, 1, 1),
            stream=stream,
        )

    launch_token_multidest.compile_hints = {
        "llvm_options": {
            "amdgpu-kernarg-preload": AITER_FLYDSL_KERNARG_PRELOAD,
            "amdgpu-kernarg-preload-count": AITER_FLYDSL_KERNARG_PRELOAD_COUNT,
        },
    }
    return launch_token_multidest


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
    ``grid.sync`` / cooperative launch; this uses a global-atomic spin
    protocol). Each worker owns a strided slice of the
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

    module_name = format_kernel_name(
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

    @flyc.kernel(name=module_name, known_block_size=[BLOCK_THREADS, 1, 1])
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
        """Write ``(contiguous_m//(wmma_rep*16), K//128, wmma_rep, 16, 4)`` scales."""
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
        _c_model_dim = arith.constant(model_dim, type=i32)
        _c_payload_bytes_per_row = arith.constant(payload_bytes_per_row, type=i32)
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
            addr = fx.Int64(ptrtoint(tensor)) + fx.Int64(elem_idx_i32) * 4
            p = create_llvm_ptr(addr)
            return p._value if hasattr(p, "_value") else p

        def _atomic_add(tensor, elem_idx_i32, addend):
            ptr = _elem_ptr(tensor, elem_idx_i32)
            return fx.Uint32(
                llvm.AtomicRMWOp(
                    llvm.AtomicBinOp.add,
                    ptr,
                    addend,
                    llvm.AtomicOrdering.monotonic,
                    syncscope="agent",
                    alignment=4,
                ).result
            )

        tid = fx.Uint32(fx.thread_idx.x)
        bid = fx.Uint32(fx.block_idx.x)

        warp_in_block = tid // c_wave
        lane = tid - warp_in_block * c_wave  # tid % wave_size
        route0 = bid * c_warps_per_block + warp_in_block  # first route this warp owns
        stride = fx.Uint32(num_workers) * c_warps_per_block

        topk_ids_t = ptr_buf_tensor(topk_ids)
        count_t = ptr_buf_tensor(count)

        # ============================ Phase 1: count ============================
        # Strided warp-per-route histogram into ``count`` (== masked_m). Loop bounds
        # are warp-uniform (lane-independent) so the post-phase gpu.barrier() is hit
        # by every thread of the block.
        numel_i32 = fx.Uint32(numel)
        for route_i32 in range(route0, numel_i32, stride):
            expert = fx.Uint32(topk_ids_t[route_i32])
            if lane == 0:
                _atomic_add(count, expert, c1_i32)

        # ===================== Barrier A + Phase 2: prefix sum ==================
        gpu.barrier()
        _wait_mem()
        rocdl.sched_barrier(0)

        is_block_leader = tid == c0_i32
        if is_block_leader:
            my_arrival = _atomic_add(barrier, c0_i32, c1_i32)
            nwm1 = fx.Uint32(num_workers) - 1
            is_last = my_arrival == nwm1
            is_not_last = my_arrival != nwm1

            # Last arriver: every block has bumped ``count`` (its atomics committed
            # to L2 before its arrival atomic). Serial tile-aligned prefix sum,
            # mirroring moe_contiguous_psum, reading ``count`` coherently.
            if is_last:
                tile_v = fx.Uint32(tile_m)
                tile_minus_1 = tile_v - c1_i32
                cur = fx.Uint32(0)
                for e_i32 in range(c0_i32, fx.Uint32(experts), 1):
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
                    cnt = fx.Uint32(count_t[e_i32])
                    aligned = (cnt + tile_minus_1) // tile_v * tile_v
                    _atomic_add(starts, e_i32, _raw(cur))
                    _atomic_add(psum, e_i32, _raw(cur + cnt))
                    cur = cur + aligned
                # Ensure starts/psum land in L2 before the release flag is visible,
                # then publish the release with an agent-scope atomic (the inline-asm
                # coherent store/load barrier is unreliable on gfx1250 -- readers can
                # observe the flag set before starts/psum are visible).
                _wait_mem()
                _atomic_add(barrier, c1_i32, c1_i32)

            # Other blocks: spin on the release flag until the last block publishes.
            if is_not_last:
                rel = fx.Uint32(0)
                while rel == 0:
                    # Coherent read via agent-scope atomic add of 0 (reliable on
                    # gfx1250, unlike the inline-asm coherent load).
                    rel = _atomic_add(barrier, c1_i32, c0_i32)

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
        starts_rd_t = ptr_buf_tensor(starts)
        payload_base = fx.Int64(ptrtoint(grouped_payload))
        hidden_base = fx.Int64(ptrtoint(hidden))
        scale_t = ptr_buf_tensor(grouped_scale, fx.Int8)
        topids_to_rows_t = ptr_buf_tensor(topids_to_rows)

        for route_i32 in range(route0, numel_i32, stride):
            expert = fx.Uint32(topk_ids_t[route_i32])

            # lane 0 claims the within-expert slot and reads the (published) per-
            # expert row base; both are warp-uniform, broadcast via readlane.
            slot_on_lane0 = arith.constant(0, type=i32)
            rowbase_on_lane0 = arith.constant(0, type=i32)
            if lane == 0:
                slot_on_lane0 = _atomic_add(slot_counter, expert, c1_i32)
                rowbase_on_lane0 = starts_rd_t[expert]
            slot = fx.Uint32(rocdl.readlane(i32, _raw(slot_on_lane0), _raw(c0_i32)))
            row_base = fx.Uint32(
                rocdl.readlane(i32, _raw(rowbase_on_lane0), _raw(c0_i32))
            )
            grouped_row = slot + row_base
            token = fx.Uint32(route_i32) // fx.Uint32(c_topk)

            if lane == 0:
                topids_to_rows_t[route_i32] = grouped_row

            # per-row scale-preshuffle geometry from the global grouped_row.
            scale_tile = fx.Uint32(grouped_row) // fx.Uint32(c_rows_per_tile)
            row_in_tile = grouped_row - scale_tile * c_rows_per_tile
            wmma_row = fx.Uint32(row_in_tile) // fx.Uint32(c16_i32)
            row_lane16 = row_in_tile - wmma_row * c16_i32
            scale_row_dword_base = (
                scale_tile * c_dst_scale_dwords_per_row * c16_i32
                + wmma_row * c16_i32
                + row_lane16
            )

            block_in_wave = fx.Uint32(lane) // fx.Uint32(c_lanes_per_block)
            lane_in_block = lane - block_in_wave * c_lanes_per_block
            is_block_lead = lane_in_block == c0_i32

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
                        payload_row_i32=grouped_row,
                        scale_row_dword_base=scale_row_dword_base,
                    )
                ],
                payload_base=payload_base,
                payload_bytes_per_row=payload_bytes_per_row,
                hidden_base=hidden_base,
                feat_bytes_per_row=model_dim * 2,
                feat_row_i32=token,
                scale_t=scale_t,
            )
            _emit_quant_block_loop(c)

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
        stream: fx.Stream = fx.Stream(None),  # noqa: B008
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
            "amdgpu-kernarg-preload": AITER_FLYDSL_KERNARG_PRELOAD,
            "amdgpu-kernarg-preload-count": AITER_FLYDSL_KERNARG_PRELOAD_COUNT,
        },
    }

    return launch_fused
