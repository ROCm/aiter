# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Vendored TDM gather/scatter API for gfx1250 (self-contained, no FlyDSL patch).

This module inlines the high-level TDM *gather-mode* descriptor builder and the
``tdm_gather`` / ``tdm_scatter`` one-call wrappers so aiter kernels can issue
indexed TDM loads/stores against a *stock* FlyDSL wheel -- i.e. without the
FlyDSL-side changes that taught ``make_tensor_gather_descriptor`` to accept a
fly ``lds_view`` / pointer-based LDS index.

The code below is copied verbatim from FlyDSL's ``expr/rocdl/tdm_ops.py`` (the
gather half: MI400 ISA S4.10.3.2 gather mode) with only the relative imports
rewritten to absolute ``flydsl.*`` paths. It bottoms out on the low-level
``rocdl.tensor_load_to_lds`` / ``rocdl.tensor_store_from_lds`` 5-group
intrinsics, which the wheel exposes. Once these wrappers land upstream in the
installed FlyDSL, delete this file and import them from
``flydsl.expr.rocdl.tdm_ops`` instead.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence, Tuple, Union

from flydsl._mlir import ir
from flydsl._mlir.dialects import arith as std_arith
from flydsl._mlir.dialects import llvm as llvm_dialect
from flydsl._mlir.dialects import memref as memref_dialect
from flydsl._mlir.dialects import rocdl, vector
from flydsl.expr import arith
from flydsl.expr.arith import _to_raw as _raw
from flydsl.expr.meta import dsl_loc_tracing
from flydsl.expr.typing import T, as_ir_value
from flydsl.expr.utils.arith import ArithValue as _ArithValue


# --- padding encoding helper (from tdm_ops.compute_padding_encoding) ---

def compute_padding_encoding(
    pad_interval_elems: int,
    pad_amount_elems: int,
    elem_bits: int = 16,
) -> Tuple[int, int]:
    """Compute TDM descriptor padding bitfield values.

    Follows Triton TDMUtility.cpp convention:
      padIntervalInDwords = pad_interval_elems * elem_bits / 32
      padAmountInDwords   = pad_amount_elems   * elem_bits / 32
      encoded_interval    = log2(padIntervalInDwords) - 1
      encoded_amount      = padAmountInDwords - 1

    Args:
        pad_interval_elems: Padding interval in elements (e.g. tile_k = 64).
        pad_amount_elems:   Padding amount in elements (e.g. LDS_PAD = 8).
        elem_bits:          Bits per element (16 for f16/bf16, 32 for f32).

    Returns:
        (encoded_interval, encoded_amount) ready for descriptor bits.
    """
    dword_bits = 32
    interval_dw = pad_interval_elems * elem_bits // dword_bits
    amount_dw = pad_amount_elems * elem_bits // dword_bits
    if interval_dw <= 0 or amount_dw <= 0:
        return (0, 0)
    assert interval_dw & (interval_dw - 1) == 0, f"padIntervalInDwords must be power-of-2, got {interval_dw}"
    encoded_interval = int(math.log2(interval_dw)) - 1
    encoded_amount = amount_dw - 1
    return (encoded_interval, encoded_amount)




# --- gather descriptor dataclass (from tdm_ops.TDMGatherDescriptor) ---

@dataclass
class TDMGatherDescriptor:
    """Holds GROUP0, GROUP1, GROUP2, GROUP3 for TDM gather mode.

    In gather mode, groups 2 and 3 carry row indices instead of
    higher-dimension tensor metadata.

    - 32-bit index mode: up to 8 row indices (4 per group)
    - 16-bit index mode: up to 16 row indices (8 per group)
    """

    dgroup0: object  # vector<4xi32> MLIR Value
    dgroup1: object  # vector<8xi32> MLIR Value
    dgroup2: object  # vector<4xi32> MLIR Value — row indices [0..3] or [0..7]
    dgroup3: object  # vector<4xi32> MLIR Value — row indices [4..7] or [8..15]


# --- internal helper (from tdm_ops._zero_dgroup_v8i32) ---

def _zero_dgroup_v8i32():
    """Create a zero vector<8xi32> for unused descriptor groups."""
    z = arith.constant(0, type=T.i32)
    z = as_ir_value(z)
    return vector.from_elements(T.vec(8, T.i32), [z, z, z, z, z, z, z, z])


@dsl_loc_tracing


# --- gather-mode descriptor builder + wrappers (from tdm_ops, 504-968) ---

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
    workgroup_mask: Union[int, "ir.Value"] = 0,
) -> TDMGatherDescriptor:
    """Build a TDM gather descriptor for loading arbitrary rows from global to LDS.

    In gather mode the TDM fetches rows specified by explicit indices in
    descriptor groups 2 and 3, rather than iterating over contiguous dim1.

    Args:
        global_ptr:    The global tensor pointer (fx.Tensor).
        lds_memref:    The LDS memref base (SmemAllocator base).
        row_indices:   List of row index MLIR i32 Values.  Max 8 for 32-bit
                       mode, max 16 for 16-bit mode.
        row_width:     Width of each row in data_size elements (= tile_dim0).
                       Must be a multiple of 4 bytes.
        tensor_dim0:   Full tensor dimension 0 (row width) for OOB check.
        tensor_dim1:   Full tensor dimension 1 (num rows) for OOB check.
                       Accepts a Python int (compile-time) or an MLIR i32
                       Value / SGPR (runtime).  Per ISA spec §4.10.3.2,
                       row indices >= tensor_dim1 are treated as OOB, so
                       this MUST be >= the actual number of rows (tokens).
        stride:        Stride of dim0 in elements (row stride of the global
                       matrix).
        elem_bytes:    Element size in bytes (1, 2, 4, or 8).
        pad_interval:  Padding interval in elements (0 to disable).
        pad_amount:    Padding amount in elements (0 to disable).
        index_size:    Row index width in bits (16 or 32).
        gather_tile_dim1:
                      Optional override for gather-mode tile_dim1 (the number
                      of valid indices to consume from groups 2/3). Accepts a
                      Python int or runtime MLIR i32 Value / SGPR. Defaults to
                      len(row_indices), preserving the historical behavior.
        lds_byte_offset: Additional LDS byte offset.
        global_byte_offset: Additional global memory byte offset (MLIR index).
                           Used for K-tile column offsets.
        workgroup_mask: Multicast mask.

    Returns:
        TDMGatherDescriptor with groups 0-3 ready for tensor_load_gather.
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

    # ================================================================
    # GROUP 1: config + tensor dims + tile + stride
    # ================================================================
    data_size_code = int(math.log2(elem_bytes))

    if pad_interval > 0 and pad_amount > 0:
        elem_bits = elem_bytes * 8
        enc_interval, enc_amount = compute_padding_encoding(pad_interval, pad_amount, elem_bits)
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
        g1_s0 = arith.constant(g1_s0_val, type=T.i32)
    else:
        upper = (data_size_code << 16) | (pad_enable << 20) | (enc_interval << 22) | (enc_amount << 25)
        g1_s0 = arith.ori(
            arith.constant(upper, type=T.i32),
            arith.andi(workgroup_mask, arith.constant(0xFFFF, type=T.i32)),
        )

    # tensor_dim0 (32 bits) packed into sgpr1[31:16] and sgpr2[15:0]
    # tensor_dim1 (32 bits) packed into sgpr2[31:16] and sgpr3[15:0]
    #
    # tensor_dim1 may be a runtime MLIR i32 value (e.g. num_tokens) —
    # the TDM hardware uses it for OOB checking on gather row indices.
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
            ((tensor_dim0 >> 16) & 0xFFFF) | ((tensor_dim1 & 0xFFFF) << 16),
            type=T.i32,
        )
        g1_s3 = arith.constant(
            ((tensor_dim1 >> 16) & 0xFFFF) | (row_width << 16),
            type=T.i32,
        )

    # sgpr4: tile_dim1[15:0] — in gather mode, this is the number of valid
    # indices consumed from descriptor groups 2/3. Allow kernels to override it
    # at runtime so they can keep a fixed index vector while shrinking the valid
    # prefix for padded MoE tiles.
    if gather_tile_dim1 is None:
        g1_s4 = arith.constant(num_indices & 0xFFFF, type=T.i32)
    elif isinstance(gather_tile_dim1, int):
        g1_s4 = arith.constant(gather_tile_dim1 & 0xFFFF, type=T.i32)
    else:
        g1_s4 = arith.andi(gather_tile_dim1, arith.constant(0xFFFF, type=T.i32))

    # sgpr5: tensor_dim0_stride (dim0 stride = row stride in elements)
    g1_s5 = arith.constant(stride & 0xFFFFFFFF, type=T.i32)

    # sgpr6-7: tensor_dim1_stride (ignored in gather mode)
    g1_s6 = arith.constant(0, type=T.i32)
    g1_s7 = arith.constant(0, type=T.i32)

    dgroup1 = vector.from_elements(
        T.vec(8, T.i32),
        [as_ir_value(v) for v in [g1_s0, g1_s1, g1_s2, g1_s3, g1_s4, g1_s5, g1_s6, g1_s7]],
    )

    # ================================================================
    # GROUP 2 & 3: row indices
    # ================================================================
    zero = arith.constant(0, type=T.i32)

    if index_size == 32:
        # 32-bit mode: group2 has indices [0..3], group3 has [4..7]
        g2_vals = [row_indices[i] if i < num_indices else zero for i in range(4)]
        g3_vals = [row_indices[i + 4] if (i + 4) < num_indices else zero for i in range(4)]
    else:
        # 16-bit mode: pack 2 x 16-bit indices per 32-bit word
        # Group 2: indices [0..7] packed into 4 x i32
        g2_vals = []
        for w in range(4):
            lo_idx = w * 2
            hi_idx = w * 2 + 1
            lo = row_indices[lo_idx] if lo_idx < num_indices else zero
            hi = row_indices[hi_idx] if hi_idx < num_indices else zero
            lo_masked = arith.andi(lo, arith.constant(0xFFFF, type=T.i32))
            hi_shifted = arith.shli(arith.andi(hi, arith.constant(0xFFFF, type=T.i32)), arith.constant(16, type=T.i32))
            g2_vals.append(arith.ori(lo_masked, hi_shifted))
        # Group 3: indices [8..15] packed into 4 x i32
        g3_vals = []
        for w in range(4):
            lo_idx = 8 + w * 2
            hi_idx = 8 + w * 2 + 1
            lo = row_indices[lo_idx] if lo_idx < num_indices else zero
            hi = row_indices[hi_idx] if hi_idx < num_indices else zero
            lo_masked = arith.andi(lo, arith.constant(0xFFFF, type=T.i32))
            hi_shifted = arith.shli(arith.andi(hi, arith.constant(0xFFFF, type=T.i32)), arith.constant(16, type=T.i32))
            g3_vals.append(arith.ori(lo_masked, hi_shifted))

    dgroup2 = vector.from_elements(T.vec(4, T.i32), [as_ir_value(v) for v in g2_vals])
    dgroup3 = vector.from_elements(T.vec(4, T.i32), [as_ir_value(v) for v in g3_vals])

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
    from flydsl._mlir.dialects import fly as _fly_d

    assert index_size in (16, 32), f"index_size must be 16 or 32, got {index_size}"

    glb_ptr_type = ir.Type.parse("!llvm.ptr<1>")
    i64 = ir.IntegerType.get_signless(64)
    a_raw = global_ptr.__extract_to_ir_values__()[0]
    glb_ptr = _fly_d.extract_aligned_pointer_as_index(glb_ptr_type, a_raw)
    glb_base_i64 = _ArithValue(llvm_dialect.ptrtoint(i64, glb_ptr))
    if global_byte_offset is not None:
        glb_byte_off_i64 = arith.index_cast(T.i64, global_byte_offset)
        glb_base_i64 = glb_base_i64 + glb_byte_off_i64

    # lds_memref accepts, in priority order:
    #   * an already-resolved LDS base index (e.g. from a pointer-based
    #     allocator via index_cast(index, ptrtoint(ptr))) -- used directly;
    #   * a fly view (lds_view) -- handled symmetrically to global_ptr above,
    #     extracting its aligned LDS pointer (addrspace 3) via the fly dialect;
    #   * a std MLIR memref -- via memref.extract_aligned_pointer_as_index.
    if hasattr(lds_memref, "type") and isinstance(lds_memref.type, ir.IndexType):
        lds_base_idx = (
            lds_memref if isinstance(lds_memref, _ArithValue) else _ArithValue(lds_memref)
        )
    elif hasattr(lds_memref, "__extract_to_ir_values__"):
        lds_ptr_type = ir.Type.parse("!llvm.ptr<3>")
        lds_raw = lds_memref.__extract_to_ir_values__()[0]
        lds_ptr = _fly_d.extract_aligned_pointer_as_index(lds_ptr_type, lds_raw)
        lds_base_idx = _ArithValue(
            arith.index_cast(T.index, llvm_dialect.ptrtoint(i64, lds_ptr))
        )
    else:
        lds_base_idx = _ArithValue(
            memref_dialect.extract_aligned_pointer_as_index(lds_memref)
        )
    lds_total_off = lds_base_idx
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
    g0_s3 = _ArithValue(std_arith.TruncIOp(i32, _raw(hi_raw)).result) | arith.constant(1 << 31, type=T.i32)
    return vector.from_elements(
        T.vec(4, T.i32), [as_ir_value(g0_s0), as_ir_value(g0_s1), as_ir_value(g0_s2), as_ir_value(g0_s3)]
    )


@dsl_loc_tracing
def tensor_load_gather(
    desc: TDMGatherDescriptor,
    cache_policy: int = 0,
) -> None:
    """Issue a TDM gather load (Global -> LDS) using row indices.

    Uses the 5-group tensor_load_to_lds intrinsic with groups 2 and 3
    carrying the gather row indices.

    Args:
        desc:         TDMGatherDescriptor from make_tensor_gather_descriptor.
        cache_policy: Cache policy (0 = default).
    """
    dg4 = _raw(_zero_dgroup_v8i32())
    rocdl.tensor_load_to_lds(
        _raw(desc.dgroup0),
        _raw(desc.dgroup1),
        _raw(desc.dgroup2),
        _raw(desc.dgroup3),
        dg4,
        cache_policy,
    )


@dsl_loc_tracing
def tensor_store_gather(
    desc: TDMGatherDescriptor,
    cache_policy: int = 0,
) -> None:
    """Issue a TDM gather store (LDS -> Global) using row indices.

    Uses the 5-group tensor_store_from_lds intrinsic with groups 2 and 3
    carrying the gather row indices.

    Args:
        desc:         TDMGatherDescriptor from make_tensor_gather_descriptor.
        cache_policy: Cache policy (0 = default).
    """
    dg4 = _raw(_zero_dgroup_v8i32())
    rocdl.tensor_store_from_lds(
        _raw(desc.dgroup0),
        _raw(desc.dgroup1),
        _raw(desc.dgroup2),
        _raw(desc.dgroup3),
        dg4,
        cache_policy,
    )


# ---------------------------------------------------------------------------
# High-level one-call TDM indexed gather/scatter
#
# These fuse ``make_tensor_gather_descriptor`` with the matching issue op so a
# kernel expresses a hardware TDM indexed transfer in a single call, mirroring
# the ``fx.copy`` / ``fx.gather`` / ``fx.scatter`` naming family.
#
# NOTE: unlike the *software* ``fx.gather`` / ``fx.scatter`` (which expand to a
# per-row loop of ordinary copies), ``tdm_gather`` / ``tdm_scatter`` emit ONE
# hardware TDM instruction that packs up to 8 (32-bit index) / 16 (16-bit
# index) row indices into the descriptor. ``lds_memref`` accepts a fly LDS view
# (lds_view), a std memref, or an LDS base index -- see
# ``make_tensor_gather_dgroup0``.
# ---------------------------------------------------------------------------


def _gather_layout_from_views(global_ptr, lds_memref):
    """Derive the gather descriptor geometry from the fly views themselves.

    ``global_ptr`` / ``lds_memref`` (global_view / lds_view) carry it all in
    their layout + element type:

      tensor_dim1 = global rows       (global_ptr layout dim0 extent)
      tensor_dim0 = global row width  (global_ptr layout dim1 extent)
      stride      = global row stride (global_ptr layout dim0 stride)
      row_width   = LDS tile width    (lds_memref layout dim1 extent)
      elem_bytes  = global elem size  (global_ptr.element_type width)
    """
    from flydsl.expr.primitive import get, get_layout, get_scalar, get_shape, get_stride

    _gl = get_layout(global_ptr)
    _gsh, _gst = get_shape(_gl), get_stride(_gl)
    tensor_dim1 = get_scalar(get(_gsh, 0))
    tensor_dim0 = get_scalar(get(_gsh, 1))
    stride = get_scalar(get(_gst, 0))
    elem_bytes = global_ptr.element_type.width // 8
    row_width = get_scalar(get(get_shape(get_layout(lds_memref)), 1))
    # tensor_dim1 takes make_tensor_gather_descriptor's runtime path (arith.andi
    # on a raw i32); a compile-time int stays int (static path).
    if not isinstance(tensor_dim1, int):
        tensor_dim1 = as_ir_value(tensor_dim1)
    return row_width, tensor_dim0, tensor_dim1, stride, elem_bytes


@dsl_loc_tracing
def tdm_gather(
    global_ptr,
    lds_memref,
    row_indices,
    *,
    index_size: int = 32,
    gather_tile_dim1=None,
    lds_byte_offset=None,
    global_byte_offset=None,
    pad_interval: int = 0,
    pad_amount: int = 0,
    workgroup_mask: Union[int, "ir.Value"] = 0,
    cache_policy: int = 0,
) -> None:
    """Hardware TDM indexed load ``lds[...] = global[row_indices]`` in one call.

    Descriptor geometry (row_width / tensor_dim0 / tensor_dim1 / stride /
    elem_bytes) comes from the ``global_ptr`` / ``lds_memref`` fly views. When a
    field must differ from the views (e.g. an OOB extent), call
    ``make_tensor_gather_descriptor`` + ``tensor_load_gather`` directly instead.
    """
    row_width, tensor_dim0, tensor_dim1, stride, elem_bytes = _gather_layout_from_views(
        global_ptr, lds_memref
    )
    desc = make_tensor_gather_descriptor(
        global_ptr,
        lds_memref,
        row_indices,
        row_width=row_width,
        tensor_dim0=tensor_dim0,
        tensor_dim1=tensor_dim1,
        stride=stride,
        elem_bytes=elem_bytes,
        pad_interval=pad_interval,
        pad_amount=pad_amount,
        index_size=index_size,
        gather_tile_dim1=gather_tile_dim1,
        lds_byte_offset=lds_byte_offset,
        global_byte_offset=global_byte_offset,
        workgroup_mask=workgroup_mask,
    )
    tensor_load_gather(desc, cache_policy)


@dsl_loc_tracing
def tdm_scatter(
    global_ptr,
    lds_memref,
    row_indices,
    *,
    index_size: int = 32,
    gather_tile_dim1=None,
    lds_byte_offset=None,
    global_byte_offset=None,
    pad_interval: int = 0,
    pad_amount: int = 0,
    workgroup_mask: Union[int, "ir.Value"] = 0,
    cache_policy: int = 0,
) -> None:
    """Hardware TDM indexed store ``global[row_indices] = lds[...]`` in one call.

    Descriptor geometry (row_width / tensor_dim0 / tensor_dim1 / stride /
    elem_bytes) comes from the ``global_ptr`` / ``lds_memref`` fly views. When a
    field must differ from the views (e.g. an OOB extent), call
    ``make_tensor_gather_descriptor`` + ``tensor_store_gather`` directly instead.
    """
    row_width, tensor_dim0, tensor_dim1, stride, elem_bytes = _gather_layout_from_views(
        global_ptr, lds_memref
    )
    desc = make_tensor_gather_descriptor(
        global_ptr,
        lds_memref,
        row_indices,
        row_width=row_width,
        tensor_dim0=tensor_dim0,
        tensor_dim1=tensor_dim1,
        stride=stride,
        elem_bytes=elem_bytes,
        pad_interval=pad_interval,
        pad_amount=pad_amount,
        index_size=index_size,
        gather_tile_dim1=gather_tile_dim1,
        lds_byte_offset=lds_byte_offset,
        global_byte_offset=global_byte_offset,
        workgroup_mask=workgroup_mask,
    )
    tensor_store_gather(desc, cache_policy)


# ---------------------------------------------------------------------------
# K-loop hoist helpers
#
# In the MoE GEMM K-reduction loop, only the global "addr_lo" (lane 2 of
# dgroup0) actually advances per K-tile; the LDS layout (lane 1), addr_hi
# (lane 3), predicate (lane 0), and the entire dgroup1 / dgroup2 / dgroup3
# state are K-invariant. By building a base descriptor at K=0 once outside
# the loop and patching only lane 2 inside the loop, we cut the per-iteration
# work to a single vector.insert plus the addr_lo SGPR add.
# ---------------------------------------------------------------------------



