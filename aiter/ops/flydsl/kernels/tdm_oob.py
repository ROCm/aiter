# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Compatibility wrapper for OOB-capable gfx1250 TDM descriptors.

The gfx1250 FP8/FP4 GEMM kernel now relies on the newer FlyDSL
``make_tensor_descriptor_2d(..., oob_inner_bound=...)`` API.  The aiter pinned
FlyDSL package may only expose ``oob_outer_bound``.  This module re-exports the
installed ``flydsl.expr.rocdl.tdm_ops`` API and replaces only the descriptor
builder when the installed version lacks inner-dimension OOB support.
"""

from __future__ import annotations

import inspect
import math
from typing import Tuple, Union

from flydsl.expr.rocdl import tdm_ops as _tdm

try:
    from flydsl.expr.meta import dsl_loc_tracing
except ImportError:

    def dsl_loc_tracing(fn):
        return fn


__all__ = list(getattr(_tdm, "__all__", ()))

for _name in __all__:
    globals()[_name] = getattr(_tdm, _name)

if "make_tensor_descriptor_2d" not in __all__:
    __all__.append("make_tensor_descriptor_2d")

_TDM_SIG_PARAMS = inspect.signature(_tdm.make_tensor_descriptor_2d).parameters
_TDM_HAS_INNER_OOB = "oob_inner_bound" in _TDM_SIG_PARAMS


if _TDM_HAS_INNER_OOB:
    make_tensor_descriptor_2d = _tdm.make_tensor_descriptor_2d
else:
    # Reuse the installed module's low-level bindings so the shim stays aligned
    # with the FlyDSL package actually imported by this environment.
    ir = _tdm.ir
    std_arith = _tdm.std_arith
    llvm_dialect = _tdm.llvm_dialect
    memref_dialect = _tdm.memref_dialect
    arith = _tdm.arith
    vector = _tdm.vector
    _raw = _tdm._raw
    T = _tdm.T
    _ArithValue = _tdm._ArithValue
    compute_padding_encoding = _tdm.compute_padding_encoding
    compute_warp_distribution = _tdm.compute_warp_distribution
    TDMDescriptor2D = _tdm.TDMDescriptor2D

    @dsl_loc_tracing
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
        oob_inner_bound=None,
    ) -> "TDMDescriptor2D":
        """Build a 2D TDM descriptor with outer and inner OOB clipping."""
        from flydsl._mlir.dialects import fly as _fly_d

        outer_stride, inner_stride = strides
        outer_tile, inner_tile = tile_shape
        outer_off, inner_off = global_offset

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
                outer_stride_idx = _ArithValue(os_val)
            elif isinstance(os_val.type, ir.IntegerType) and os_val.type.width == 32:
                outer_stride_idx = arith.index_cast(T.index, os_val)
            else:
                raise TypeError(
                    f"outer stride ir.Value must be index or i32, got {os_val.type}"
                )
            outer_stride_is_runtime = True

        warps_per_dim, block_per_warp = compute_warp_distribution(
            [outer_tile, inner_tile],
            num_warps,
        )
        bpw_outer, bpw_inner = block_per_warp
        warps_dim0 = warps_per_dim[0]

        if num_warps > 1:
            from flydsl.expr import rocdl as _rocdl_ext

            wid_i32 = _rocdl_ext.wave_id()
            wave_id = arith.index_cast(T.index, wid_i32)
            warp_coord_outer = wave_id % arith.index(warps_dim0)
            warp_coord_inner = wave_id / arith.index(warps_dim0)
            warp_off_outer = warp_coord_outer * arith.index(bpw_outer)
            warp_off_inner = warp_coord_inner * arith.index(bpw_inner)
        else:
            warp_off_outer = arith.index(0)
            warp_off_inner = arith.index(0)

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

        lds_base_idx = _ArithValue(
            memref_dialect.extract_aligned_pointer_as_index(lds_memref)
        )
        if pad_interval > 0 and pad_amount > 0:
            lds_inner_stride = inner_tile + pad_amount
        else:
            lds_inner_stride = inner_tile
        lds_warp_elem_off = (
            warp_off_outer * arith.index(lds_inner_stride) + warp_off_inner
        )
        lds_warp_byte_off = lds_warp_elem_off * arith.index(elem_bytes)
        lds_total_off = lds_base_idx + lds_warp_byte_off
        if lds_byte_offset is not None:
            lds_total_off = lds_total_off + lds_byte_offset
        lds_addr_i32 = arith.index_cast(T.i32, lds_total_off)

        g0_s0 = arith.constant(pred, type=T.i32)
        g0_s1 = lds_addr_i32
        i32 = ir.IntegerType.get_signless(32)
        g0_s2 = _ArithValue(std_arith.TruncIOp(i32, _raw(glb_addr_i64)).result)
        hi_raw = _ArithValue(_raw(glb_addr_i64)).shrui(arith.constant(32, type=T.i64))
        g0_s3 = _ArithValue(
            std_arith.TruncIOp(i32, _raw(hi_raw)).result
        ) | arith.constant(1 << 31, type=T.i32)
        dgroup0 = vector.from_elements(T.vec(4, T.i32), [g0_s0, g0_s1, g0_s2, g0_s3])

        tdim0 = bpw_inner
        tdim1 = bpw_outer
        tile_d0 = bpw_inner
        tile_d1 = bpw_outer

        if for_store and pad_interval > 0 and pad_amount > 0:
            tile_d0 += pad_amount
            pad_interval = 0
            pad_amount = 0

        stride0 = outer_stride
        data_size_code = int(math.log2(elem_bytes))

        if pad_interval > 0 and pad_amount > 0:
            elem_bits = elem_bytes * 8
            enc_interval, enc_amount = compute_padding_encoding(
                pad_interval, pad_amount, elem_bits
            )
            pad_enable = 1
        else:
            enc_interval, enc_amount = 0, 0
            pad_enable = 0

        abe = 1 if atomic_barrier_enable else 0
        early_timeout_bit = 1 if early_timeout else 0
        g1_s0_upper = (
            (data_size_code << 16)
            | (abe << 18)
            | (0 << 19)
            | (pad_enable << 20)
            | (early_timeout_bit << 21)
            | (enc_interval << 22)
            | (enc_amount << 25)
        )

        if isinstance(workgroup_mask, int):
            g1_s0_val = (workgroup_mask & 0xFFFF) | g1_s0_upper
            g1_s0 = arith.constant(g1_s0_val, type=T.i32)
        else:
            upper_const = arith.constant(g1_s0_upper, type=T.i32)
            mask_i32 = arith.andi(workgroup_mask, arith.constant(0xFFFF, type=T.i32))
            g1_s0 = arith.ori(upper_const, mask_i32)

        def _oob_bound_to_i32(name, bound):
            if isinstance(bound, int):
                return arith.constant(bound, type=T.i32)
            bound_i32 = bound.ir_value() if hasattr(bound, "ir_value") else bound
            if not isinstance(bound_i32, ir.Value):
                raise TypeError(
                    f"{name} must be int or i32/index ir.Value, "
                    f"got {type(bound).__name__}"
                )
            if isinstance(bound_i32.type, ir.IndexType):
                return arith.index_cast(T.i32, bound_i32)
            if not (
                isinstance(bound_i32.type, ir.IntegerType)
                and bound_i32.type.width == 32
            ):
                raise TypeError(
                    f"{name} ir.Value must be index or i32, got {bound_i32.type}"
                )
            return bound_i32

        def _remaining_oob_extent(name, bound, start):
            bound_i32 = _oob_bound_to_i32(name, bound)
            start_i32 = arith.index_cast(T.i32, start)
            return arith.maxsi(
                arith.subi(bound_i32, start_i32), arith.constant(0, type=T.i32)
            )

        if oob_inner_bound is None and oob_outer_bound is None:
            g1_s1 = arith.constant((tdim0 & 0xFFFF) << 16, type=T.i32)
            g1_s2 = arith.constant(
                ((tdim0 >> 16) & 0xFFFF) | ((tdim1 & 0xFFFF) << 16),
                type=T.i32,
            )
            g1_s3 = arith.constant(
                ((tdim1 >> 16) & 0xFFFF) | (tile_d0 << 16),
                type=T.i32,
            )
        else:
            tdim0_rt = (
                None
                if oob_inner_bound is None
                else _remaining_oob_extent(
                    "oob_inner_bound",
                    oob_inner_bound,
                    inner_off + warp_off_inner,
                )
            )
            tdim1_rt = (
                None
                if oob_outer_bound is None
                else _remaining_oob_extent(
                    "oob_outer_bound",
                    oob_outer_bound,
                    outer_off + warp_off_outer,
                )
            )
            c16 = arith.constant(16, type=T.i32)
            c_mask16 = arith.constant(0xFFFF, type=T.i32)

            if tdim0_rt is None:
                g1_s1 = arith.constant((tdim0 & 0xFFFF) << 16, type=T.i32)
                tdim0_hi = arith.constant((tdim0 >> 16) & 0xFFFF, type=T.i32)
            else:
                tdim0_lo = arith.andi(tdim0_rt, c_mask16)
                tdim0_hi = arith.andi(arith.shrui(tdim0_rt, c16), c_mask16)
                g1_s1 = arith.shli(tdim0_lo, c16)

            if tdim1_rt is None:
                tdim1_lo_shifted = arith.constant((tdim1 & 0xFFFF) << 16, type=T.i32)
                tdim1_hi = arith.constant((tdim1 >> 16) & 0xFFFF, type=T.i32)
            else:
                tdim1_lo = arith.andi(tdim1_rt, c_mask16)
                tdim1_lo_shifted = arith.shli(tdim1_lo, c16)
                tdim1_hi = arith.andi(arith.shrui(tdim1_rt, c16), c_mask16)

            g1_s2 = arith.ori(tdim0_hi, tdim1_lo_shifted)
            g1_s3 = arith.ori(
                tdim1_hi,
                arith.constant(tile_d0 << 16, type=T.i32),
            )

        g1_s4 = arith.constant(tile_d1 & 0xFFFF, type=T.i32)

        if outer_stride_is_runtime:
            g1_s5 = arith.index_cast(T.i32, outer_stride_idx)
        else:
            g1_s5 = arith.constant(stride0 & 0xFFFFFFFF, type=T.i32)

        g1_s6 = arith.constant(0, type=T.i32)
        g1_s7 = arith.constant(0, type=T.i32)

        dgroup1 = vector.from_elements(
            T.vec(8, T.i32),
            [g1_s0, g1_s1, g1_s2, g1_s3, g1_s4, g1_s5, g1_s6, g1_s7],
        )

        return TDMDescriptor2D(dgroup0=dgroup0, dgroup1=dgroup1)
