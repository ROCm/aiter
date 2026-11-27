# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Vendored TDM gather-mode API for gfx1250 (self-contained, no FlyDSL patch).

Lets aiter kernels issue indexed TDM stores against a *stock* FlyDSL wheel --
i.e. without the FlyDSL-side change that taught
``make_tensor_gather_descriptor`` to accept a fly ``lds_view`` / pointer-based
LDS index. Adapted from FlyDSL's ``expr/rocdl/tdm_ops.py`` (MI400 ISA
S4.10.3.2 gather mode); it bottoms out on the low-level
``rocdl.tensor_store_from_lds`` 5-group intrinsic, which the wheel exposes.
Once these wrappers land upstream, delete this file and import them from
``flydsl.expr.rocdl.tdm_ops`` instead.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import flydsl.expr as fx
from flydsl.expr import rocdl
from flydsl.expr.meta import dsl_loc_tracing
from flydsl.expr.typing import T


def compute_padding_encoding(
    pad_interval_elems: int,
    pad_amount_elems: int,
    elem_bits: int = 16,
) -> tuple[int, int]:
    """TDM descriptor padding bitfield values, per Triton's TDMUtility.cpp.

    The descriptor encodes the interval as ``log2(interval_dwords) - 1`` and the
    amount as ``amount_dwords - 1``.
    """
    dword_bits = 32
    interval_dw = pad_interval_elems * elem_bits // dword_bits
    amount_dw = pad_amount_elems * elem_bits // dword_bits
    if interval_dw <= 0 or amount_dw <= 0:
        return (0, 0)
    assert (
        interval_dw & (interval_dw - 1) == 0
    ), f"padIntervalInDwords must be power-of-2, got {interval_dw}"
    encoded_interval = int(math.log2(interval_dw)) - 1
    encoded_amount = amount_dw - 1
    return (encoded_interval, encoded_amount)


@dataclass
class TDMGatherDescriptor:
    """The four descriptor groups for TDM gather mode.

    Groups 2 and 3 carry row indices instead of higher-dimension tensor
    metadata: up to 8 indices in 32-bit mode, 16 in 16-bit mode.
    """

    dgroup0: object  # vector<4xi32> MLIR Value
    dgroup1: object  # vector<8xi32> MLIR Value
    dgroup2: object  # vector<4xi32> MLIR Value — row indices [0..3] or [0..7]
    dgroup3: object  # vector<4xi32> MLIR Value — row indices [4..7] or [8..15]


def _byte_offset_to_i64(offset):
    """Widen a caller byte offset to i64.

    Accepts a python int, an ``index`` value, or any narrower integer value, so
    kernel bodies can pass a plain expression instead of wrapping it in an
    ``index_cast`` just to satisfy this boundary.
    """
    if isinstance(offset, int):
        return fx.Int64(offset)
    raw = fx.as_ir_value(offset)
    if raw.type == T.index:
        return fx.Int64(raw)
    if raw.type == T.i64:
        return fx.Int64(raw)
    # Byte offsets represented by narrower signless integers were zero-extended.
    return fx.Uint64(raw)


def _zero_dgroup_v8i32():
    return fx.full(8, 0, fx.Int32).ir_value()


@dsl_loc_tracing
def make_tensor_gather_descriptor(
    global_ptr,
    lds_memref,
    row_indices,
    row_width: int,
    tensor_dim0: int,
    tensor_dim1,
    stride: int,
    elem_bytes: int = 1,
    pad_interval: int = 0,
    pad_amount: int = 0,
    index_size: int = 32,
    gather_tile_dim1=None,
    lds_byte_offset=None,
    global_byte_offset=None,
    workgroup_mask=0,
) -> TDMGatherDescriptor:
    """Build a TDM gather descriptor addressing arbitrary rows by index.

    ``tensor_dim1`` (Python int or runtime i32) is the OOB bound: per ISA
    §4.10.3.2 the hardware drops row indices >= it, which is how callers mask
    padding rows. ``gather_tile_dim1`` overrides how many of groups 2/3's
    indices are consumed, so a kernel can keep a fixed index vector while
    shrinking its valid prefix; it defaults to ``len(row_indices)``.
    """
    assert index_size in (16, 32), f"index_size must be 16 or 32, got {index_size}"
    max_indices = 8 if index_size == 32 else 16
    num_indices = len(row_indices)
    assert (
        0 < num_indices <= max_indices
    ), f"row_indices length {num_indices} exceeds max {max_indices} for {index_size}-bit mode"
    assert (
        row_width * elem_bytes % 4 == 0
    ), f"row_width * elem_bytes must be multiple of 4, got {row_width * elem_bytes}"

    dgroup0 = make_tensor_gather_dgroup0(
        global_ptr=global_ptr,
        lds_memref=lds_memref,
        index_size=index_size,
        lds_byte_offset=lds_byte_offset,
        global_byte_offset=global_byte_offset,
    )

    # GROUP 1: config + tensor dims + tile + stride
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

    if isinstance(workgroup_mask, int):
        g1_s0_val = (
            (workgroup_mask & 0xFFFF)
            | (data_size_code << 16)
            | (0 << 18)  # atomic_barrier_enable
            | (0 << 19)  # iterate_enable (ignored in gather)
            | (pad_enable << 20)
            | (0 << 21)  # early_timeout
            | (enc_interval << 22)
            | (enc_amount << 25)
        )
        g1_s0 = fx.Int32(g1_s0_val)
    else:
        upper = (
            (data_size_code << 16)
            | (pad_enable << 20)
            | (enc_interval << 22)
            | (enc_amount << 25)
        )
        g1_s0 = fx.Int32(upper) | (fx.Int32(workgroup_mask) & 0xFFFF)

    # tensor_dim0 (32 bits) packed into sgpr1[31:16] and sgpr2[15:0];
    # tensor_dim1 (32 bits) into sgpr2[31:16] and sgpr3[15:0].
    _td1_is_runtime = not isinstance(tensor_dim1, int)

    g1_s1 = fx.Int32((tensor_dim0 & 0xFFFF) << 16)

    if _td1_is_runtime:
        td1 = fx.Int32(tensor_dim1)
        g1_s2 = fx.Int32((tensor_dim0 >> 16) & 0xFFFF) | ((td1 & 0xFFFF) << 16)
        g1_s3 = (td1.shrui(fx.Int32(16)) & 0xFFFF) | fx.Int32(row_width << 16)
    else:
        g1_s2 = fx.Int32(
            ((tensor_dim0 >> 16) & 0xFFFF) | ((tensor_dim1 & 0xFFFF) << 16)
        )
        g1_s3 = fx.Int32(((tensor_dim1 >> 16) & 0xFFFF) | (row_width << 16))

    # sgpr4: tile_dim1[15:0] — the number of indices consumed from groups 2/3.
    if gather_tile_dim1 is None:
        g1_s4 = fx.Int32(num_indices & 0xFFFF)
    elif isinstance(gather_tile_dim1, int):
        g1_s4 = fx.Int32(gather_tile_dim1 & 0xFFFF)
    else:
        g1_s4 = fx.Int32(gather_tile_dim1) & 0xFFFF

    # sgpr5: tensor_dim0_stride (dim0 stride = row stride in elements)
    g1_s5 = fx.Int32(stride & 0xFFFFFFFF)

    # sgpr6-7: tensor_dim1_stride (ignored in gather mode)
    g1_s6 = fx.Int32(0)
    g1_s7 = fx.Int32(0)

    dgroup1 = fx.Vector.from_elements(
        [g1_s0, g1_s1, g1_s2, g1_s3, g1_s4, g1_s5, g1_s6, g1_s7],
        fx.Int32,
    ).ir_value()

    # GROUP 2 & 3: row indices
    zero = fx.Int32(0)

    if index_size == 32:
        g2_vals = [
            fx.Int32(row_indices[i]) if i < num_indices else zero for i in range(4)
        ]
        g3_vals = [
            fx.Int32(row_indices[i + 4]) if (i + 4) < num_indices else zero
            for i in range(4)
        ]
    else:
        # 16-bit mode packs 2 indices per word: [0..7] in group 2, [8..15] in 3.
        g2_vals = []
        for w in range(4):
            lo_idx = w * 2
            hi_idx = w * 2 + 1
            lo = fx.Int32(row_indices[lo_idx]) if lo_idx < num_indices else zero
            hi = fx.Int32(row_indices[hi_idx]) if hi_idx < num_indices else zero
            g2_vals.append((lo & fx.Int32(0xFFFF)) | ((hi & fx.Int32(0xFFFF)) << 16))
        g3_vals = []
        for w in range(4):
            lo_idx = 8 + w * 2
            hi_idx = 8 + w * 2 + 1
            lo = fx.Int32(row_indices[lo_idx]) if lo_idx < num_indices else zero
            hi = fx.Int32(row_indices[hi_idx]) if hi_idx < num_indices else zero
            g3_vals.append((lo & fx.Int32(0xFFFF)) | ((hi & fx.Int32(0xFFFF)) << 16))

    dgroup2 = fx.Vector.from_elements(g2_vals, fx.Int32).ir_value()
    dgroup3 = fx.Vector.from_elements(g3_vals, fx.Int32).ir_value()

    return TDMGatherDescriptor(
        dgroup0=dgroup0,
        dgroup1=dgroup1,
        dgroup2=dgroup2,
        dgroup3=dgroup3,
    )


@dsl_loc_tracing
def make_tensor_gather_dgroup0(
    global_ptr,
    lds_memref,
    *,
    index_size: int = 32,
    lds_byte_offset=None,
    global_byte_offset=None,
):
    """Build gather descriptor GROUP0 only.

    This is the dynamic address-bearing portion of a TDM gather descriptor.
    Separating it lets kernels hoist static GROUP1/GROUP2/GROUP3 state and
    only rebuild the per-issue address group close to the TDM instruction.
    """
    assert index_size in (16, 32), f"index_size must be 16 or 32, got {index_size}"

    glb_base_i64 = fx.Int64(fx.ptrtoint(fx.get_iter(global_ptr)))
    if global_byte_offset is not None:
        glb_base_i64 += fx.Int64(_byte_offset_to_i64(global_byte_offset))

    # The callers pass either an LDS base index or a FlyDSL view.
    if hasattr(lds_memref, "type") and lds_memref.type == T.index:
        lds_base_i64 = fx.Int64(lds_memref)
    elif hasattr(lds_memref, "__extract_to_ir_values__"):
        lds_base_i64 = fx.Int64(fx.ptrtoint(fx.get_iter(lds_memref)))
    else:
        raise TypeError("lds_memref must be an LDS base index or FlyDSL view")
    lds_total_off = lds_base_i64
    if lds_byte_offset is not None:
        lds_total_off += fx.Int64(lds_byte_offset)
    lds_addr_i32 = fx.Int32(lds_total_off)

    gather_index_bit = 1 if index_size == 32 else 0
    g0_pred = 1 | (gather_index_bit << 30) | (1 << 31)
    g0_s0 = fx.Int32(g0_pred)
    g0_s1 = lds_addr_i32

    g0_s2 = fx.Uint32(glb_base_i64)
    g0_s3 = fx.Uint32(fx.Uint64(glb_base_i64) >> 32) | fx.Uint32(1 << 31)
    return fx.Vector.from_elements(
        [g0_s0, g0_s1, g0_s2, g0_s3],
        fx.Int32,
    ).ir_value()


@dsl_loc_tracing
def tensor_store_gather(
    desc: TDMGatherDescriptor,
    cache_policy: int = 0,
) -> None:
    """Issue one TDM gather store (LDS -> Global) for the descriptor's rows."""
    dg4 = _zero_dgroup_v8i32()
    rocdl.tensor_store_from_lds(
        desc.dgroup0,
        desc.dgroup1,
        desc.dgroup2,
        desc.dgroup3,
        dg4,
        cache_policy=cache_policy if cache_policy else None,
    )
