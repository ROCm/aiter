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

from flydsl._mlir import ir
from flydsl._mlir.dialects import arith as std_arith
from flydsl._mlir.dialects import llvm as llvm_dialect
from flydsl._mlir.dialects import memref as memref_dialect
from flydsl._mlir.dialects import rocdl, vector
from flydsl.expr import arith
from flydsl.expr.arith import _to_raw as _raw
from flydsl.expr.meta import dsl_loc_tracing
from flydsl.expr.rocdl import readfirstlane, tdm_ops
from flydsl.expr.typing import T, as_ir_value
from flydsl.expr.utils.arith import ArithValue as _ArithValue


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
        return arith.constant(offset, type=T.i64)
    raw = _raw(offset)
    if isinstance(raw.type, ir.IndexType):
        return arith.index_cast(T.i64, raw)
    if raw.type == T.i64:
        return _ArithValue(raw)
    return _ArithValue(std_arith.ExtUIOp(T.i64, raw).result)


def _zero_dgroup_v8i32():
    z = as_ir_value(arith.constant(0, type=T.i32))
    return vector.from_elements(T.vec(8, T.i32), [z, z, z, z, z, z, z, z])


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
    workgroup_mask: int | ir.Value = 0,
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
        g1_s0 = arith.constant(g1_s0_val, type=T.i32)
    else:
        upper = (
            (data_size_code << 16)
            | (pad_enable << 20)
            | (enc_interval << 22)
            | (enc_amount << 25)
        )
        g1_s0 = arith.ori(
            arith.constant(upper, type=T.i32),
            arith.andi(workgroup_mask, arith.constant(0xFFFF, type=T.i32)),
        )

    # tensor_dim0 (32 bits) packed into sgpr1[31:16] and sgpr2[15:0];
    # tensor_dim1 (32 bits) into sgpr2[31:16] and sgpr3[15:0].
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

    # sgpr4: tile_dim1[15:0] — the number of indices consumed from groups 2/3.
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
        [
            as_ir_value(v)
            for v in [g1_s0, g1_s1, g1_s2, g1_s3, g1_s4, g1_s5, g1_s6, g1_s7]
        ],
    )

    # GROUP 2 & 3: row indices
    zero = arith.constant(0, type=T.i32)

    if index_size == 32:
        g2_vals = [row_indices[i] if i < num_indices else zero for i in range(4)]
        g3_vals = [
            row_indices[i + 4] if (i + 4) < num_indices else zero for i in range(4)
        ]
    else:
        # 16-bit mode packs 2 indices per word: [0..7] in group 2, [8..15] in 3.
        g2_vals = []
        for w in range(4):
            lo_idx = w * 2
            hi_idx = w * 2 + 1
            lo = row_indices[lo_idx] if lo_idx < num_indices else zero
            hi = row_indices[hi_idx] if hi_idx < num_indices else zero
            lo_masked = arith.andi(lo, arith.constant(0xFFFF, type=T.i32))
            hi_shifted = arith.shli(
                arith.andi(hi, arith.constant(0xFFFF, type=T.i32)),
                arith.constant(16, type=T.i32),
            )
            g2_vals.append(arith.ori(lo_masked, hi_shifted))
        g3_vals = []
        for w in range(4):
            lo_idx = 8 + w * 2
            hi_idx = 8 + w * 2 + 1
            lo = row_indices[lo_idx] if lo_idx < num_indices else zero
            hi = row_indices[hi_idx] if hi_idx < num_indices else zero
            lo_masked = arith.andi(lo, arith.constant(0xFFFF, type=T.i32))
            hi_shifted = arith.shli(
                arith.andi(hi, arith.constant(0xFFFF, type=T.i32)),
                arith.constant(16, type=T.i32),
            )
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
        glb_base_i64 = glb_base_i64 + _byte_offset_to_i64(global_byte_offset)

    # lds_memref accepts, in priority order:
    #   * an already-resolved LDS base index (e.g. from a pointer-based
    #     allocator via index_cast(index, ptrtoint(ptr))) -- used directly;
    #   * a fly view (lds_view) -- handled symmetrically to global_ptr above,
    #     extracting its aligned LDS pointer (addrspace 3) via the fly dialect;
    #   * a std MLIR memref -- via memref.extract_aligned_pointer_as_index.
    if hasattr(lds_memref, "type") and isinstance(lds_memref.type, ir.IndexType):
        lds_base_idx = (
            lds_memref
            if isinstance(lds_memref, _ArithValue)
            else _ArithValue(lds_memref)
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
    g0_s3 = _ArithValue(std_arith.TruncIOp(i32, _raw(hi_raw)).result) | arith.constant(
        1 << 31, type=T.i32
    )
    return vector.from_elements(
        T.vec(4, T.i32),
        [
            as_ir_value(g0_s0),
            as_ir_value(g0_s1),
            as_ir_value(g0_s2),
            as_ir_value(g0_s3),
        ],
    )


@dsl_loc_tracing
def tensor_store_gather(
    desc: TDMGatherDescriptor,
    cache_policy: int = 0,
) -> None:
    """Issue one TDM gather store (LDS -> Global) for the descriptor's rows."""
    dg4 = _raw(_zero_dgroup_v8i32())
    cache_policy_attr = (
        ir.IntegerAttr.get(ir.IntegerType.get_signless(32), cache_policy)
        if cache_policy
        else None
    )
    rocdl.tensor_store_from_lds(
        _raw(desc.dgroup0),
        _raw(desc.dgroup1),
        _raw(desc.dgroup2),
        _raw(desc.dgroup3),
        dg4,
        cache_policy=cache_policy_attr,
    )


# --- EP dispatch TDM descriptors (from mori intranode TDM) ---

TDM_MAX_DIM = 0xFFFF

#: A TDM row narrower than this moves at a fraction of peak: the engine issues
#: one descriptor-driven burst per row, so short rows pay the per-row overhead
#: on every one of them. It is a bandwidth floor, not a legality bound -- a
#: 1-row / sub-128B transfer is well-formed and lands correctly.
TDM_ROW_FLOOR_BYTES = 128


def _i32(v):
    """i32 constant from a bit pattern that may set bit 31 (arith wants signed)."""
    v &= 0xFFFFFFFF
    if v >= 1 << 31:
        v -= 1 << 32
    return arith.constant(v, type=T.i32)


def tdm_group1(dim0, dim1, elem_bytes, stride=None):
    """GROUP1 for a ``dim1`` x ``dim0`` tile of ``elem_bytes`` elements.

    ``dim0`` is the contiguous extent of one row and ``stride`` (default
    ``dim0``, i.e. a dense run) the element distance between rows. All three are
    compile-time, so the whole descriptor folds to eight SGPR constants.
    """
    if elem_bytes not in (1, 2, 4, 8):
        raise ValueError(f"TDM data_size encodes 1/2/4/8-byte elements, got {elem_bytes}")
    if not (1 <= dim0 <= TDM_MAX_DIM and 1 <= dim1 <= TDM_MAX_DIM):
        raise ValueError(f"TDM tile dims must be in [1, {TDM_MAX_DIM}], got {dim1}x{dim0}")
    if stride is None:
        stride = dim0
    ds = int(math.log2(elem_bytes))
    return vector.from_elements(
        T.vec(8, T.i32),
        [
            _i32(ds << 16),
            _i32((dim0 & 0xFFFF) << 16),
            _i32(((dim0 >> 16) & 0xFFFF) | ((dim1 & 0xFFFF) << 16)),
            _i32(((dim1 >> 16) & 0xFFFF) | ((dim0 & 0xFFFF) << 16)),
            _i32(dim1 & 0xFFFF),
            _i32(stride & 0xFFFFFFFF),
            _i32(((stride >> 32) & 0xFFFF) | ((dim1 & 0xFFFF) << 16)),
            _i32((dim1 >> 16) & 0xFFFFFFFF),
        ],
    )


#: 128B / 4B: the row floor in elements for the 4-byte metadata runs.
_TDM_ROW_ELEMS_4B = TDM_ROW_FLOOR_BYTES // 4


def tdm_run_shape(n_elems):
    """``(dim0, dim1)`` covering a contiguous run of ``n_elems``, or None.

    mori's ``TdmCheapDim1`` closed form: the largest ``dim1`` in 8/4/2 that
    divides the run and still leaves a row of at least 32 elements (the 128B
    floor at 4 bytes). ``dim0 * dim1 == n_elems`` exactly, so the descriptor
    footprint is the run and cannot spill past it.

    When no 128B-legal tile exists, fall back to the narrowest legal-by-
    construction shape ``(n_elems/2, 2)`` for even ``n_elems >= 4`` -- the same
    branch as C++ ``TdmWholeOrSplit128``. The 128B floor is a bandwidth result,
    not a legality one: a metadata field at 512 tokens is 64B..512B, so half
    bandwidth costs nothing measurable while leaving TDM costs the whole
    pipeline (HIP A/B +10% @512).

    Returns None when no such split exists. The caller must then move the run
    some other way rather than falling back to ``dim1 == 1``: a 1xN tile is not
    a shape the engine accepts here (see the module docstring).
    """
    for d1 in (8, 4, 2):
        if n_elems % d1 == 0 and n_elems // d1 >= _TDM_ROW_ELEMS_4B:
            return n_elems // d1, d1
    if n_elems >= 4 and (n_elems & 1) == 0:
        return n_elems // 2, 2
    return None


def tdm_group0(lds_addr_i32, global_addr_i64):
    """GROUP0 binding a descriptor to one (LDS byte address, global address) pair.

    Both operands must be wave-uniform -- the intrinsic reads them out of SGPRs.
    They are put through ``readfirstlane`` here rather than left to the compiler's
    uniformity analysis: the addresses this kernel feeds in are uniform by
    construction (a warp-strided token id, a ``readlane``-broadcast peer slot) but
    are computed from ``thread_idx``, and a descriptor lane that silently lands in
    a VGPR moves the wrong bytes instead of failing to compile. Every caller is in
    a wave-uniform branch, so lane 0 is always active.
    """
    i32 = ir.IntegerType.get_signless(32)
    addr = arith.unwrap(global_addr_i64)
    lo = arith.trunci(i32, addr)
    hi = arith.ori(
        arith.trunci(i32, arith.shrui(addr, arith.constant(32, type=T.i64))),
        _i32(1 << 31),  # descriptor type field
    )
    return vector.from_elements(
        T.vec(4, T.i32),
        [
            _i32(1),
            readfirstlane(T.i32, arith.unwrap(lds_addr_i32)),
            readfirstlane(T.i32, lo),
            readfirstlane(T.i32, hi),
        ],
    )


def tdm_load(group0, group1, cache_policy=0):
    """Async global -> LDS. Completion is observed with :func:`tdm_wait`."""
    tdm_ops.tensor_load_2d(tdm_ops.TDMDescriptor2D(group0, group1), cache_policy)


def tdm_store(group0, group1, cache_policy=0):
    """Async LDS -> global. Completion is observed with :func:`tdm_wait`."""
    tdm_ops.tensor_store_2d(tdm_ops.TDMDescriptor2D(group0, group1), cache_policy)


def tdm_wait(count=0):
    """Wait until at most ``count`` TDM transfers are still in flight.

    Loads and stores share the one tensor counter and retire in issue order, so
    a partial wait releases the oldest transfers -- there is no way to wait on
    just the loads.
    """
    tdm_ops.tensor_wait(count)
