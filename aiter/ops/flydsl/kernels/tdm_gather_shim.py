# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Inlined TDM gather/scatter descriptor builder (gfx1250).

The installed FlyDSL wheel exposes the low-level 5-group intrinsic
``rocdl.tensor_store_from_lds`` / ``rocdl.tensor_load_to_lds`` but NOT the
high-level ``tensor_store_gather`` / ``make_tensor_gather_descriptor`` wrappers
(those live only in the FlyDSL source tree). This module inlines the pure-IR
descriptor packing (ported verbatim from FlyDSL's tdm_ops.py, MI400 ISA
§4.10.3.2 gather mode) so aiter kernels can issue TDM gather-mode
loads/stores against the runtime wheel without rebuilding FlyDSL.

Gather mode: descriptor groups 2/3 carry explicit row indices instead of
higher-dim tensor metadata. Address of row i = global_base + row_index[i] *
(dim0 stride). 32-bit index mode -> up to 8 indices/instruction; 16-bit ->
up to 16. Both load (Global->LDS) and store (LDS->Global) directions.
"""

from __future__ import annotations

from dataclasses import dataclass

from flydsl._mlir import ir
from flydsl._mlir.dialects import arith as std_arith
from flydsl._mlir.dialects import fly as _fly_d
from flydsl._mlir.dialects import llvm as llvm_dialect
from flydsl._mlir.dialects import memref as memref_dialect
from flydsl._mlir.dialects import rocdl, vector
from flydsl.expr import arith
from flydsl.expr.arith import _to_raw as _raw
from flydsl.expr.typing import T, as_ir_value
from flydsl.expr.utils.arith import ArithValue as _ArithValue


@dataclass
class TDMGatherDescriptor:
    """GROUP0..3 for a TDM gather-mode transfer (groups 2/3 = row indices)."""

    dgroup0: object  # vector<4xi32>
    dgroup1: object  # vector<8xi32>
    dgroup2: object  # vector<4xi32> — row indices [0..3] (32b) or [0..7] (16b)
    dgroup3: object  # vector<4xi32> — row indices [4..7] (32b) or [8..15] (16b)


def _zero_dgroup_v8i32():
    z = as_ir_value(arith.constant(0, type=T.i32))
    return vector.from_elements(T.vec(8, T.i32), [z, z, z, z, z, z, z, z])


def make_tensor_gather_dgroup0(
    global_ptr=None,
    lds_memref=None,
    *,
    global_addr_i64=None,
    lds_base_idx=None,
    index_size: int = 32,
    lds_byte_offset=None,
    global_byte_offset=None,
):
    """GROUP0: predicate + LDS addr + global addr lo/hi (the per-issue address
    group). Provide the global base either as ``global_ptr`` (an fx.Tensor) OR
    ``global_addr_i64`` (a raw i64 base VA, e.g. cco ``lsa_ptr``). Provide the
    LDS base either as ``lds_memref`` (an LDS memref) OR ``lds_base_idx`` (a
    precomputed LDS byte-base index). ``global_byte_offset`` (MLIR index) is
    added to the global base."""
    assert index_size in (16, 32), f"index_size must be 16 or 32, got {index_size}"

    i64 = ir.IntegerType.get_signless(64)
    if global_addr_i64 is not None:
        glb_base_i64 = _ArithValue(_raw(global_addr_i64))
    else:
        glb_ptr_type = ir.Type.parse("!llvm.ptr<1>")
        a_raw = global_ptr.__extract_to_ir_values__()[0]
        glb_ptr = _fly_d.extract_aligned_pointer_as_index(glb_ptr_type, a_raw)
        glb_base_i64 = _ArithValue(llvm_dialect.ptrtoint(i64, glb_ptr))
    if global_byte_offset is not None:
        glb_byte_off_i64 = arith.index_cast(T.i64, global_byte_offset)
        glb_base_i64 = glb_base_i64 + glb_byte_off_i64

    if lds_base_idx is not None:
        lds_total_off = _ArithValue(_raw(lds_base_idx))
    else:
        lds_total_off = _ArithValue(
            memref_dialect.extract_aligned_pointer_as_index(lds_memref)
        )
    if lds_byte_offset is not None:
        lds_total_off = lds_total_off + lds_byte_offset
    lds_addr_i32 = arith.index_cast(T.i32, lds_total_off)

    gather_index_bit = 1 if index_size == 32 else 0
    g0_pred = 1 | (gather_index_bit << 30) | (1 << 31)
    g0_s0 = arith.constant(g0_pred, type=T.i32)
    g0_s1 = lds_addr_i32

    i32 = ir.IntegerType.get_signless(32)
    g0_s2 = _ArithValue(std_arith.TruncIOp(i32, _raw(glb_base_i64)).result)
    hi_raw = _ArithValue(_raw(glb_base_i64)).shrui(arith.constant(32, type=T.i64))
    g0_s3 = _ArithValue(std_arith.TruncIOp(i32, _raw(hi_raw)).result) | arith.constant(
        1 << 31, type=T.i32
    )
    return vector.from_elements(
        T.vec(4, T.i32),
        [as_ir_value(g0_s0), as_ir_value(g0_s1), as_ir_value(g0_s2), as_ir_value(g0_s3)],
    )


def make_tensor_gather_descriptor(
    row_indices,
    row_width: int,
    tensor_dim0: int,
    tensor_dim1,
    stride: int,
    global_ptr=None,
    global_addr_i64=None,
    lds_memref=None,
    lds_base_idx=None,
    elem_bytes: int = 1,
    index_size: int = 32,
    gather_tile_dim1=None,
    lds_byte_offset=None,
    global_byte_offset=None,
    workgroup_mask=0,
) -> TDMGatherDescriptor:
    """Build a TDM gather descriptor (no padding).

    row_indices:  list of MLIR i32 Values (<=8 for 32-bit, <=16 for 16-bit).
    row_width:    per-row width in elements (= tile_dim0); row_width*elem_bytes
                  must be a multiple of 4.
    tensor_dim0:  full row width (OOB check).
    tensor_dim1:  full num rows (OOB check); int or runtime i32 Value. Row
                  indices >= tensor_dim1 are dropped by HW (§4.10.3.2).
    stride:       dim0 stride = row stride in elements.
    gather_tile_dim1: optional override for the number of valid indices consumed.
    """
    import math

    assert index_size in (16, 32), f"index_size must be 16 or 32, got {index_size}"
    max_indices = 8 if index_size == 32 else 16
    num_indices = len(row_indices)
    assert (
        0 < num_indices <= max_indices
    ), f"row_indices length {num_indices} exceeds max {max_indices} for {index_size}-bit"
    assert (
        row_width * elem_bytes % 4 == 0
    ), f"row_width*elem_bytes must be mult of 4, got {row_width * elem_bytes}"

    dgroup0 = make_tensor_gather_dgroup0(
        global_ptr=global_ptr,
        global_addr_i64=global_addr_i64,
        lds_memref=lds_memref,
        lds_base_idx=lds_base_idx,
        index_size=index_size,
        lds_byte_offset=lds_byte_offset,
        global_byte_offset=global_byte_offset,
    )

    data_size_code = int(math.log2(elem_bytes))
    pad_enable = 0
    enc_interval, enc_amount = 0, 0

    if isinstance(workgroup_mask, int):
        g1_s0_val = (
            (workgroup_mask & 0xFFFF)
            | (data_size_code << 16)
            | (pad_enable << 20)
            | (enc_interval << 22)
            | (enc_amount << 25)
        )
        g1_s0 = arith.constant(g1_s0_val, type=T.i32)
    else:
        upper = (data_size_code << 16) | (pad_enable << 20)
        g1_s0 = arith.ori(
            arith.constant(upper, type=T.i32),
            arith.andi(workgroup_mask, arith.constant(0xFFFF, type=T.i32)),
        )

    _td1_is_runtime = not isinstance(tensor_dim1, int)
    g1_s1 = arith.constant((tensor_dim0 & 0xFFFF) << 16, type=T.i32)

    if _td1_is_runtime:
        _td0_hi = arith.constant((tensor_dim0 >> 16) & 0xFFFF, type=T.i32)
        _td1_lo = arith.andi(tensor_dim1, arith.constant(0xFFFF, type=T.i32))
        _td1_lo_shifted = arith.shli(_td1_lo, arith.constant(16, type=T.i32))
        g1_s2 = arith.ori(_td0_hi, _td1_lo_shifted)
        _td1_hi = arith.andi(
            arith.shrui(tensor_dim1, arith.constant(16, type=T.i32)),
            arith.constant(0xFFFF, type=T.i32),
        )
        g1_s3 = arith.ori(_td1_hi, arith.constant(row_width << 16, type=T.i32))
    else:
        g1_s2 = arith.constant(
            ((tensor_dim0 >> 16) & 0xFFFF) | ((tensor_dim1 & 0xFFFF) << 16), type=T.i32
        )
        g1_s3 = arith.constant(
            ((tensor_dim1 >> 16) & 0xFFFF) | (row_width << 16), type=T.i32
        )

    if gather_tile_dim1 is None:
        g1_s4 = arith.constant(num_indices & 0xFFFF, type=T.i32)
    elif isinstance(gather_tile_dim1, int):
        g1_s4 = arith.constant(gather_tile_dim1 & 0xFFFF, type=T.i32)
    else:
        g1_s4 = arith.andi(gather_tile_dim1, arith.constant(0xFFFF, type=T.i32))

    g1_s5 = arith.constant(stride & 0xFFFFFFFF, type=T.i32)
    g1_s6 = arith.constant(0, type=T.i32)
    g1_s7 = arith.constant(0, type=T.i32)

    dgroup1 = vector.from_elements(
        T.vec(8, T.i32),
        [
            as_ir_value(v)
            for v in [g1_s0, g1_s1, g1_s2, g1_s3, g1_s4, g1_s5, g1_s6, g1_s7]
        ],
    )

    zero = arith.constant(0, type=T.i32)
    if index_size == 32:
        g2_vals = [row_indices[i] if i < num_indices else zero for i in range(4)]
        g3_vals = [row_indices[i + 4] if (i + 4) < num_indices else zero for i in range(4)]
    else:
        g2_vals = []
        for w in range(4):
            lo = row_indices[w * 2] if w * 2 < num_indices else zero
            hi = row_indices[w * 2 + 1] if w * 2 + 1 < num_indices else zero
            lo_m = arith.andi(lo, arith.constant(0xFFFF, type=T.i32))
            hi_s = arith.shli(
                arith.andi(hi, arith.constant(0xFFFF, type=T.i32)),
                arith.constant(16, type=T.i32),
            )
            g2_vals.append(arith.ori(lo_m, hi_s))
        g3_vals = []
        for w in range(4):
            lo = row_indices[8 + w * 2] if 8 + w * 2 < num_indices else zero
            hi = row_indices[8 + w * 2 + 1] if 8 + w * 2 + 1 < num_indices else zero
            lo_m = arith.andi(lo, arith.constant(0xFFFF, type=T.i32))
            hi_s = arith.shli(
                arith.andi(hi, arith.constant(0xFFFF, type=T.i32)),
                arith.constant(16, type=T.i32),
            )
            g3_vals.append(arith.ori(lo_m, hi_s))

    dgroup2 = vector.from_elements(T.vec(4, T.i32), [as_ir_value(v) for v in g2_vals])
    dgroup3 = vector.from_elements(T.vec(4, T.i32), [as_ir_value(v) for v in g3_vals])

    return TDMGatherDescriptor(
        dgroup0=dgroup0, dgroup1=dgroup1, dgroup2=dgroup2, dgroup3=dgroup3
    )


def tensor_load_gather(desc: TDMGatherDescriptor, cache_policy: int = 0) -> None:
    """Issue a TDM gather load (Global -> LDS) using row indices."""
    dg4 = _raw(_zero_dgroup_v8i32())
    rocdl.tensor_load_to_lds(
        _raw(desc.dgroup0),
        _raw(desc.dgroup1),
        _raw(desc.dgroup2),
        _raw(desc.dgroup3),
        dg4,
        cache_policy,
    )


def tensor_store_gather(desc: TDMGatherDescriptor, cache_policy: int = 0) -> None:
    """Issue a TDM gather store (LDS -> Global) using row indices."""
    dg4 = _raw(_zero_dgroup_v8i32())
    rocdl.tensor_store_from_lds(
        _raw(desc.dgroup0),
        _raw(desc.dgroup1),
        _raw(desc.dgroup2),
        _raw(desc.dgroup3),
        dg4,
        cache_policy,
    )
