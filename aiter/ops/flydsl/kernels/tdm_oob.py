# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Vendored OOB-capable TDM 2D descriptor builder for gfx1250.

This is a faithful copy of ``flydsl.expr.rocdl.tdm_ops.make_tensor_descriptor_2d``
as of the FlyDSL "add M out-of-bounds support" change, carried here so the
non-tile-aligned-M (OOB) GEMM path works against the *older* flydsl this aiter
build pins, whose ``make_tensor_descriptor_2d`` predates the ``oob_outer_bound``
argument.

The kernel only routes through this fallback when the installed flydsl lacks
native ``oob_outer_bound`` support (see ``_make_tdm_desc`` in
``gemm_fp8fp4_gfx1250``); when flydsl has it, the native builder is used.

To stay robust across flydsl internal-layout changes, every low-level symbol is
sourced from the installed ``tdm_ops`` module namespace (i.e. whatever that
module successfully imported) rather than re-imported from private paths. The
only behavioural delta vs. the pinned builder is the ``oob_outer_bound`` branch
that computes a runtime ``tensor_dim1``; with ``oob_outer_bound=None`` the output
is byte-identical to the original path.
"""

from __future__ import annotations

import math
from typing import Tuple, Union

from flydsl.expr.rocdl import tdm_ops as _tdm

# Reuse whatever the installed tdm_ops bound — keeps us in lock-step with the
# pinned flydsl's lower-level primitives instead of guessing private paths.
ir = _tdm.ir
std_arith = _tdm.std_arith
llvm_dialect = _tdm.llvm_dialect
memref_dialect = _tdm.memref_dialect
arith = _tdm.arith
vector = _tdm.vector
_raw = _tdm._raw
T = _tdm.T
_ArithValue = _tdm._ArithValue
compute_warp_distribution = _tdm.compute_warp_distribution
compute_padding_encoding = _tdm.compute_padding_encoding
TDMDescriptor2D = _tdm.TDMDescriptor2D


def make_tensor_descriptor_2d(
    global_ptr,
    lds_memref,
    global_offset: Tuple,
    tensor_shape: Tuple[int, int],
    strides: Tuple[int, int],
    tile_shape: Tuple[int, int],
    elem_bytes: int = 2,
    pad_interval: int = 0,
    pad_amount: int = 0,
    num_warps: int = 1,
    cache_policy: int = 0,
    pred: int = 1,
    workgroup_mask: Union[int, "ir.Value"] = 0,
    lds_byte_offset=None,
    for_store: bool = False,
    atomic_barrier_enable: bool = False,
    early_timeout: bool = False,
    oob_outer_bound=None,
) -> "TDMDescriptor2D":
    """Build a 2D TDM descriptor (vendored, OOB-capable).

    See ``flydsl.expr.rocdl.tdm_ops.make_tensor_descriptor_2d`` for the full
    argument reference. ``oob_outer_bound`` is the runtime outer-dim global
    extent (real M for a row-major A/C); when given, ``tensor_dim1`` is set to
    the tile-start-relative remaining extent
    ``max(0, oob_outer_bound - (outer_off + warp_off_outer))`` while
    ``tile_dim1`` stays the full per-warp tile, so the partial last tile exceeds
    the tensor bound and the hardware OOB-handles the overhang (fault-safe load,
    zero-fill in LDS). Accepts a Python int or an i32/index ir.Value. ``None``
    keeps ``tensor_dim1 == tile_dim1`` (OOB off) — byte-identical to the
    non-OOB path.
    """
    from flydsl._mlir.dialects import fly as _fly_d

    outer_stride, inner_stride = strides
    outer_tile, inner_tile = tile_shape
    outer_off, inner_off = global_offset

    # The outer (leading-dim) stride may be a compile-time int or a runtime
    # i32/index ir.Value (strided A/C, e.g. a row-slice whose row pitch exceeds
    # the logical inner extent). Normalise to an index value for address math and
    # an i32 value for the descriptor's stride field (sgpr5).
    if isinstance(outer_stride, int):
        outer_stride_idx = arith.index(outer_stride)
        outer_stride_is_runtime = False
    else:
        os_val = (
            outer_stride.ir_value()
            if hasattr(outer_stride, "ir_value")
            else outer_stride
        )
        if not isinstance(os_val, ir.Value):
            raise TypeError(
                f"outer stride must be int or i32/index ir.Value, "
                f"got {type(outer_stride).__name__}"
            )
        if isinstance(os_val.type, ir.IndexType):
            # Wrap raw ir.Value so it supports the _ArithValue ops below (*, cast).
            outer_stride_idx = _ArithValue(os_val)
        elif isinstance(os_val.type, ir.IntegerType) and os_val.type.width == 32:
            outer_stride_idx = arith.index_cast(T.index, os_val)
        else:
            raise TypeError(
                f"outer stride ir.Value must be index or i32, got {os_val.type}"
            )
        outer_stride_is_runtime = True

    # -- Warp distribution --
    warps_per_dim, block_per_warp = compute_warp_distribution(
        [outer_tile, inner_tile],
        num_warps,
    )
    bpw_outer, bpw_inner = block_per_warp
    warps_dim0 = warps_per_dim[0]

    if num_warps > 1:
        # Auto-acquire SGPR wave_id via hardware register (TTMP8[29:25]).
        # This keeps the entire descriptor address chain in SALU,
        from flydsl.expr import rocdl as _rocdl_ext

        _wid_i32 = _rocdl_ext.wave_id()
        wave_id = arith.index_cast(T.index, _wid_i32)
        warp_coord_outer = wave_id % arith.index(warps_dim0)
        warp_coord_inner = wave_id / arith.index(warps_dim0)
        warp_off_outer = warp_coord_outer * arith.index(bpw_outer)
        warp_off_inner = warp_coord_inner * arith.index(bpw_inner)
    else:
        warp_off_outer = arith.index(0)
        warp_off_inner = arith.index(0)

    # -- Global address (byte address for descriptor) --
    glb_ptr_type = ir.Type.parse("!llvm.ptr<1>")
    i64 = ir.IntegerType.get_signless(64)
    a_raw = global_ptr.__extract_to_ir_values__()[0]
    glb_ptr = _fly_d.extract_aligned_pointer_as_index(glb_ptr_type, a_raw)
    glb_base_i64 = _ArithValue(llvm_dialect.ptrtoint(i64, glb_ptr))
    glb_elem_off = (outer_off + warp_off_outer) * outer_stride_idx + (
        inner_off + warp_off_inner
    ) * arith.index(inner_stride)
    glb_byte_off = glb_elem_off * arith.index(elem_bytes)
    glb_byte_off_i64 = arith.index_cast(T.i64, glb_byte_off)
    glb_addr_i64 = glb_base_i64 + glb_byte_off_i64

    # -- LDS address (byte address within shared memory) --
    lds_base_idx = _ArithValue(
        memref_dialect.extract_aligned_pointer_as_index(lds_memref)
    )
    # Compute padded LDS stride (elements) for the outer dim
    if pad_interval > 0 and pad_amount > 0:
        lds_inner_stride = inner_tile + pad_amount  # padded row width
    else:
        lds_inner_stride = inner_tile
    lds_warp_elem_off = warp_off_outer * arith.index(lds_inner_stride) + warp_off_inner
    lds_warp_byte_off = lds_warp_elem_off * arith.index(elem_bytes)
    lds_total_off = lds_base_idx + lds_warp_byte_off
    if lds_byte_offset is not None:
        lds_total_off = lds_total_off + lds_byte_offset
    lds_addr_i32 = arith.index_cast(T.i32, lds_total_off)

    # ================================================================
    # GROUP0 (vector<4xi32>): pred, lds_addr, global_addr_lo/hi
    # ================================================================
    g0_s0 = arith.constant(pred, type=T.i32)
    g0_s1 = lds_addr_i32
    i32 = ir.IntegerType.get_signless(32)
    g0_s2 = _ArithValue(std_arith.TruncIOp(i32, _raw(glb_addr_i64)).result)
    hi_raw = _ArithValue(_raw(glb_addr_i64)).shrui(arith.constant(32, type=T.i64))
    g0_s3 = _ArithValue(std_arith.TruncIOp(i32, _raw(hi_raw)).result) | arith.constant(
        1 << 31, type=T.i32
    )  # type field = 2 in [31:30]
    dgroup0 = vector.from_elements(T.vec(4, T.i32), [g0_s0, g0_s1, g0_s2, g0_s3])

    # ================================================================
    # GROUP1 (vector<8xi32>): config + tensor dims + strides + tile
    # ================================================================
    # Descriptor dim ordering: dim0=innermost, dim1=outermost
    tdim0 = bpw_inner  # innermost extent per warp
    tdim1 = bpw_outer  # outermost extent per warp
    tile_d0 = bpw_inner  # block dim0 per warp
    tile_d1 = bpw_outer  # block dim1 per warp

    # Padding can be applied to the LDS address when copying from memory to LDS,
    #  but not when copying from LDS to memory
    #  (there is no "de-padding" operation; padding is ignored).
    if for_store and pad_interval > 0 and pad_amount > 0:
        tile_d0 += pad_amount
        pad_interval = 0
        pad_amount = 0

    # stride_dim0 in descriptor = outermost stride in elements
    stride0 = outer_stride

    # data_size = log2(elem_bytes)
    data_size_code = int(math.log2(elem_bytes))

    # Padding encoding
    if pad_interval > 0 and pad_amount > 0:
        elem_bits = elem_bytes * 8
        enc_interval, enc_amount = compute_padding_encoding(
            pad_interval, pad_amount, elem_bits
        )
        pad_enable = 1
    else:
        enc_interval, enc_amount = 0, 0
        pad_enable = 0

    # sgpr0: config bitfields
    _abe = 1 if atomic_barrier_enable else 0
    _early_timeout = 1 if early_timeout else 0
    g1_s0_upper = (
        (data_size_code << 16)  # data_size [17:16]
        | (_abe << 18)  # atomic_barrier_enable
        | (0 << 19)  # iterate_enable
        | (pad_enable << 20)  # pad_enable
        | (_early_timeout << 21)  # early_timeout
        | (enc_interval << 22)  # pad_interval [24:22]
        | (enc_amount << 25)  # pad_amount [31:25]
    )

    if isinstance(workgroup_mask, int):
        g1_s0_val = (workgroup_mask & 0xFFFF) | g1_s0_upper
        g1_s0 = arith.constant(g1_s0_val, type=T.i32)
    else:
        upper_const = arith.constant(g1_s0_upper, type=T.i32)
        mask_i32 = arith.andi(workgroup_mask, arith.constant(0xFFFF, type=T.i32))
        g1_s0 = arith.ori(upper_const, mask_i32)

    # sgpr1: atomic_barrier_addr[15:0]=0 | tensor_dim0_lo[31:16]
    g1_s1 = arith.constant((tdim0 & 0xFFFF) << 16, type=T.i32)

    if oob_outer_bound is None:
        # Compile-time tensor_dim1 == tile extent: OOB checking off.
        # sgpr2: tensor_dim0_hi[15:0] | tensor_dim1_lo[31:16]
        g1_s2 = arith.constant(
            ((tdim0 >> 16) & 0xFFFF) | ((tdim1 & 0xFFFF) << 16),
            type=T.i32,
        )
        # sgpr3: tensor_dim1_hi[15:0] | tile_dim0[31:16]
        g1_s3 = arith.constant(
            ((tdim1 >> 16) & 0xFFFF) | (tile_d0 << 16),
            type=T.i32,
        )
    else:
        # Runtime tensor_dim1 = max(0, oob_outer_bound - (outer_off + warp_off_outer)),
        # tile-start-relative (the descriptor's global address already includes the
        # tile/warp start). tile_dim1 (sgpr4) stays the full per-warp tile, so the
        # partial last tile exceeds the tensor bound and the HW OOB-handles the
        # overhang. tensor_dim0 (innermost) and the tile dims stay compile-time.
        if isinstance(oob_outer_bound, int):
            ob_i32 = arith.constant(oob_outer_bound, type=T.i32)
        else:
            ob_i32 = (
                oob_outer_bound.ir_value()
                if hasattr(oob_outer_bound, "ir_value")
                else oob_outer_bound
            )
            if not isinstance(ob_i32, ir.Value):
                raise TypeError(
                    f"oob_outer_bound must be int or i32/index ir.Value, "
                    f"got {type(oob_outer_bound).__name__}"
                )
            if isinstance(ob_i32.type, ir.IndexType):
                ob_i32 = arith.index_cast(T.i32, ob_i32)
            elif not (
                isinstance(ob_i32.type, ir.IntegerType) and ob_i32.type.width == 32
            ):
                raise TypeError(
                    f"oob_outer_bound ir.Value must be index or i32, got {ob_i32.type}"
                )
        start_i32 = arith.index_cast(T.i32, outer_off + warp_off_outer)
        tdim1_rt = arith.maxsi(
            arith.subi(ob_i32, start_i32), arith.constant(0, type=T.i32)
        )
        c16 = arith.constant(16, type=T.i32)
        c_mask16 = arith.constant(0xFFFF, type=T.i32)
        # sgpr2: tensor_dim0_hi[15:0] (const) | tensor_dim1_lo[31:16] (runtime)
        g1_s2 = arith.ori(
            arith.constant((tdim0 >> 16) & 0xFFFF, type=T.i32),
            arith.shli(arith.andi(tdim1_rt, c_mask16), c16),
        )
        # sgpr3: tensor_dim1_hi[15:0] (runtime) | tile_dim0[31:16] (const)
        g1_s3 = arith.ori(
            arith.andi(arith.shrui(tdim1_rt, c16), c_mask16),
            arith.constant(tile_d0 << 16, type=T.i32),
        )

    # sgpr4: tile_dim1[15:0] | tile_dim2[31:16]=0  (always the full per-warp tile)
    g1_s4 = arith.constant(tile_d1 & 0xFFFF, type=T.i32)

    # sgpr5: tensor_dim0_stride (low 32 bits) — stride of outermost dim
    if outer_stride_is_runtime:
        # Runtime leading-dim stride: truncate the index to i32 (strides < 2^31).
        g1_s5 = arith.index_cast(T.i32, outer_stride_idx)
    else:
        g1_s5 = arith.constant(stride0 & 0xFFFFFFFF, type=T.i32)

    # sgpr6-7: for 2D, no higher-dim strides
    g1_s6 = arith.constant(0, type=T.i32)
    g1_s7 = arith.constant(0, type=T.i32)

    dgroup1 = vector.from_elements(
        T.vec(8, T.i32),
        [g1_s0, g1_s1, g1_s2, g1_s3, g1_s4, g1_s5, g1_s6, g1_s7],
    )

    return TDMDescriptor2D(dgroup0=dgroup0, dgroup1=dgroup1)
