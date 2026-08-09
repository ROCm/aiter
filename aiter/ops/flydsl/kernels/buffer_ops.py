# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
# Modifications Copyright (C) 2026 Advanced Micro Devices, Inc.

"""AMD buffer load/store operations, vendored into aiter.

flydsl moved these from ``flydsl.expr.buffer_ops`` to its repo-level
``kernels/common/``, which its wheel does not ship, so aiter keeps an
equivalent copy here.

Buffer instructions are an AMD hardware feature (buffer resource descriptor
plus ROCDL intrinsics) providing out-of-bounds protection and better memory
throughput; plain memref load/store is not a substitute.

Descriptor construction delegates to the flydsl layout API
(``fx.rocdl.make_buffer_tensor`` / ``make_buffer_ptr`` + ``get_buffer_rsrc``),
which builds the same V# — identical flags, stride and num_records. What stays
here is the ``(rsrc, element_offset)`` calling convention: the tensor-oriented
``fx.copy`` surface can't express it, and several hundred call sites depend on
it. The remaining raw ops are the hardware boundary itself: the ROCDL
raw-buffer load/store intrinsics, the ``s.buffer.load`` scalar path, and the
byte-GEP in :func:`get_element_ptr`.

Example:
    >>> from aiter.ops.flydsl.kernels import buffer_ops
    >>> import flydsl.expr as fx
    >>>
    >>> rsrc = buffer_ops.create_buffer_resource(A)
    >>> offset = row * fx.Int32(4096) + col
    >>> data = buffer_ops.buffer_load(rsrc, offset, vec_width=4)
    >>> buffer_ops.buffer_store(data, rsrc, offset)
"""

from __future__ import annotations

import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import arith as std_arith
from flydsl._mlir.dialects import llvm, rocdl
from flydsl._mlir.extras import types as T
from flydsl.expr.meta import dsl_loc_tracing


__all__ = [
    "buffer_load",
    "buffer_store",
    "create_buffer_resource",
    "create_buffer_resource_from_addr",
    "create_llvm_ptr",
    "get_element_ptr",
]


def _unwrap_value(value):
    """Recursively unwrap ArithValue or similar wrappers to get the actual MLIR value.

    Handles:
    - FlyDSL ArithValue (has ._value)
    - flyc DSL Numeric like fx.Int32 (has .ir_value() method)
    - flyc ArithValue (is already ir.Value subclass)
    """
    # DSL Numeric (Int32, Float32, etc.) — use ir_value() to materialize
    if hasattr(value, "ir_value") and not isinstance(value, ir.Value):
        return value.ir_value()
    max_depth = 10  # Safety limit
    depth = 0
    while depth < max_depth and not isinstance(value, ir.Value):
        if hasattr(value, "_value"):
            value = value._value
        elif hasattr(value, "value"):
            value = value.value
        else:
            break
        depth += 1
    return value


def _wrap(value: ir.Value):
    """Wrap a raw MLIR value in the matching ``fx`` Numeric class."""
    return fx.Numeric.from_ir_type(value.type)(value)


def _BYTE_PTR_TYPE():
    """Untyped global byte pointer.

    The descriptor is element-agnostic (buffer_load / buffer_store compute
    byte offsets themselves), so this mirrors the opaque ``!llvm.ptr`` the
    raw path used, alignment claim included.
    """
    return fx.PointerType.get(
        fx.Int8.ir_type, address_space=fx.AddressSpace.Global, alignment=1
    )


@dsl_loc_tracing
def _create_i32_constant(value: int) -> ir.Value:
    """Create an i32 constant."""
    if value > 0x7FFFFFFF:
        value = int(value - 2**32)
    return fx.Int32(value).ir_value()


def _to_i32_offset(offset: ir.Value) -> ir.Value:
    """Normalize an already-unwrapped offset value to i32.

    Accepts index (index_cast), i32 (as-is), wider ints e.g. i64 (trunc), and
    narrower ints (sign-extend). Lets callers pass fx.Int64 offsets (element or
    byte) without hitting the ``index_cast i64<->i32`` incompatibility.
    """
    ot = offset.type
    if isinstance(ot, ir.IntegerType) and ot.width == 32:
        return offset
    return _wrap(offset).to(fx.Int32).ir_value()


def _mask_offset(mask, offset: ir.Value) -> ir.Value:
    """Predicate a byte offset: invalid lanes are pushed out of bounds.

    The buffer descriptor's OOB behaviour turns the max offset into a
    dropped store / zero load, which is how masking is expressed here.
    """
    return _wrap(_unwrap_value(mask)).select(
        _wrap(offset), fx.Int32(0x7FFFFFFF)
    ).ir_value()


def _to_i64(value: ir.Value) -> ir.Value:
    """Normalize an already-unwrapped value to i64 (index_cast / sign-extend)."""
    vt = value.type
    if isinstance(vt, ir.IntegerType) and vt.width == 64:
        return value
    return _wrap(value).to(fx.Int64).ir_value()


@dsl_loc_tracing
def _ptr8_to_v4i32(ptr8_val) -> ir.Value:
    """Reinterpret a buffer resource (!llvm.ptr<8>) as a <4 x i32> vector.

    Required by the scalar ``s.buffer.load`` intrinsic, whose resource operand is
    a v4i32 rather than the opaque buffer pointer used by the vector path.
    """
    i128_ty = ir.IntegerType.get_signless(128)
    v4i32_ty = ir.VectorType.get([4], ir.IntegerType.get_signless(32))
    i128_val = llvm.ptrtoint(i128_ty, _unwrap_value(ptr8_val))
    return llvm.bitcast(v4i32_ty, i128_val)


@dsl_loc_tracing
def create_llvm_ptr(value, address_space: int = 0) -> ir.Value:
    """Create an LLVM pointer from an integer or index value."""
    value = _unwrap_value(value)
    if isinstance(value.type, ir.IndexType):
        value = _to_i64(value)
    ptr_type = ir.Type.parse(f"!llvm.ptr<{address_space}>")
    return llvm.IntToPtrOp(ptr_type, value).result


@dsl_loc_tracing
def get_element_ptr(
    base_ptr,
    byte_offset: int | ir.Value | None = None,
    static_byte_offset: int = 0,
    elem_type: ir.Type | None = None,
    no_wrap_flags=None,
) -> ir.Value:
    """Build an LLVM GEP from a base pointer plus byte offsets."""
    _gep_dynamic_index_sentinel = -(2**31)

    base_ptr = _unwrap_value(base_ptr)
    if not isinstance(static_byte_offset, int):
        raise TypeError(
            f"static_byte_offset must be int, got {type(static_byte_offset).__name__}"
        )
    if elem_type is None:
        elem_type = T.i8()
    elif callable(elem_type):
        elem_type = elem_type()

    if byte_offset is None:
        dynamic_indices = []
        raw_constant_indices = [int(static_byte_offset)]
    elif isinstance(byte_offset, int):
        dynamic_indices = []
        raw_constant_indices = [int(byte_offset) + int(static_byte_offset)]
    else:
        offset_val = _unwrap_value(byte_offset)
        if isinstance(offset_val.type, ir.IndexType):
            i64_type = T.i64()
            offset_val = _unwrap_value(
                std_arith.IndexCastOp(i64_type, offset_val).result
            )
        elif not isinstance(offset_val.type, ir.IntegerType):
            raise TypeError(
                "byte_offset must be int, index, or integer-typed MLIR value; "
                f"got {offset_val.type}"
            )

        if static_byte_offset != 0:
            static_type = offset_val.type
            static_attr = ir.IntegerAttr.get(static_type, int(static_byte_offset))
            static_const = _unwrap_value(
                std_arith.ConstantOp(static_type, static_attr).result
            )
            offset_val = _unwrap_value(
                std_arith.AddIOp(offset_val, static_const).result
            )

        dynamic_indices = [offset_val]
        raw_constant_indices = [_gep_dynamic_index_sentinel]

    return llvm.GEPOp(
        base_ptr.type,
        base_ptr,
        dynamic_indices,
        raw_constant_indices,
        elem_type,
        no_wrap_flags,
    ).result


@dsl_loc_tracing
def create_buffer_resource_from_addr(
    addr_i64: ir.Value,
    *,
    num_records_bytes: int | ir.Value | None = None,
) -> ir.Value:
    """Create AMD buffer resource descriptor from a raw i64 device address.

    Useful when working with runtime pointer arrays (e.g. IPC-mapped addresses
    or device-side pointer tables) where no fly.memref is available.
    The full address is encoded as the buffer base; callers should pass
    byte offset 0 to buffer_load / buffer_store.

    Args:
        addr_i64: Raw 64-bit device address (i64 MLIR value).
        num_records_bytes: Optional buffer size in bytes for hardware OOB checking.

    Returns:
        ROCDL buffer resource descriptor (!llvm.ptr<8>).

    Example:
        >>> rsrc = create_buffer_resource_from_addr(raw_addr_i64)
        >>> data = buffer_load(rsrc, i32_zero, vec_width=4, dtype=T.i32)
    """
    if isinstance(num_records_bytes, int):
        num_records_bytes = fx.Int64(max(0, min(num_records_bytes, 0xFFFFFFFF)))
    base_ptr = fx.inttoptr(_BYTE_PTR_TYPE(), fx.Int64(_unwrap_value(addr_i64)))
    buf_ptr = fx.rocdl.make_buffer_ptr(base_ptr, num_records_bytes=num_records_bytes)
    return fx.rocdl.get_buffer_rsrc(buf_ptr)


@dsl_loc_tracing
def create_buffer_resource(
    memref_val,
    stride: int = 0,
    max_size: bool = True,
    *,
    num_records_bytes: int | ir.Value | None = None,
) -> ir.Value:
    """Create an AMD buffer resource descriptor from a tensor.

    Wraps the tensor in a buffer-resource-backed view and hands back the raw
    ROCDL descriptor, so callers keep using the ``(rsrc, element_offset)``
    form of :func:`buffer_load` / :func:`buffer_store`.

    Args:
        memref_val: Tensor to describe.
        stride: Structured-buffer stride; only 0 (contiguous) is supported.
        max_size: Use the max descriptor size (0xFFFFFFFF) instead of the
            layout footprint.
        num_records_bytes: Override the descriptor byte count.

    Returns:
        ROCDL buffer resource descriptor (!llvm.ptr<8>)

    Example:
        >>> rsrc = create_buffer_resource(A)
        >>> data = buffer_load(rsrc, offset)
    """
    if stride != 0:
        raise ValueError(f"only contiguous (stride=0) buffers are supported, got {stride}")
    buf_tensor = fx.rocdl.make_buffer_tensor(
        memref_val, max_size, num_records_bytes=num_records_bytes
    )
    return fx.rocdl.get_buffer_rsrc(fx.get_iter(buf_tensor))


@dsl_loc_tracing
def buffer_load(
    rsrc: ir.Value,
    offset: ir.Value,
    vec_width: int = 4,
    dtype=None,
    mask: ir.Value | None = None,
    cache_modifier: int = 0,
    soffset_bytes: int | ir.Value | None = None,
    is_scalar: bool = False,
) -> ir.Value:
    """AMD buffer load operation.

    Load data from global memory using buffer descriptor and offset.
    Uses hardware-level bounds checking and vectorization.

    Args:
        rsrc: Buffer resource descriptor (!llvm.ptr<8>)
        offset: Offset in elements (i32 type)
        vec_width: Vector width (1, 2, or 4)
        dtype: Element data type (None for f32, or ir.F32Type, etc.)
        mask: Optional mask for predicated load (i1 type)
        cache_modifier: Cache control flags (0 for default)
        soffset_bytes: Optional scalar offset (in BYTES) added by the buffer instruction (soffset).
                      Use this to fold small constant deltas into the instruction instead of emitting
                      extra VGPR address arithmetic.
        is_scalar: Emit a uniform/SGPR scalar load (llvm.amdgcn.s.buffer.load) instead of the
                      vector buffer load. Use only for wave-uniform addresses to route through the
                      SMEM cache and land the result directly in SGPRs. Restricted to vec_width 1 or 4;
                      dtype is forced to i32 (the result is raw i32 dwords). mask and soffset_bytes
                      are not supported in this mode and raise ValueError if provided.

    Returns:
        Loaded data (scalar or vector depending on vec_width)

    Example:
        >>> # Load 4xf32
        >>> data = buffer_load(rsrc, offset, vec_width=4)
        >>>
        >>> # Load with mask
        >>> data = buffer_load(rsrc, offset, vec_width=4, mask=valid)
    """
    # Scalar (uniform) loads return raw i32 dwords; force the element type so the
    # element->byte offset math below uses 4 and the result type is i32 / v4i32.
    if is_scalar:
        if vec_width not in (1, 4):
            raise ValueError(
                f"buffer_load(is_scalar=True): unsupported vec_width={vec_width}"
            )
        if mask is not None or soffset_bytes is not None:
            raise ValueError(
                "buffer_load(is_scalar=True) does not support mask or soffset_bytes"
            )
        dtype = T.i32()
    # Default dtype to f32
    elif dtype is None:
        dtype = T.f32()
    # Accept DSL Numeric class (e.g. fx.Int32) as dtype: unwrap to ir.Type
    elif hasattr(dtype, "ir_type"):
        dtype = dtype.ir_type

    # Unwrap offset first (accept Python ints and DSL Numeric values).
    if isinstance(offset, int):
        offset = _create_i32_constant(offset)
    elif hasattr(offset, "ir_value"):
        offset = offset.ir_value()
    offset = _unwrap_value(offset)

    # Convert offset to i32 if needed (accepts index/i64/i32)
    offset = _to_i32_offset(offset)

    # IMPORTANT: Buffer load offset is in BYTES, not elements!
    # For vec4xf32, each element is 4 bytes, so multiply offset by 4
    element_bytes = dtype.width // 8
    offset = (fx.Int32(offset) * fx.Int32(element_bytes)).ir_value()

    # Apply mask by setting invalid offsets to max
    if mask is not None:
        offset = _mask_offset(mask, offset)

    # Create vector type
    if vec_width == 1:
        result_type = dtype
    else:
        result_type = ir.VectorType.get([vec_width], dtype)

    # Scalar/uniform load path: emit s.buffer.load with a v4i32 resource and the
    # byte offset computed above. Returns i32 (vec_width 1) or v4i32 (vec_width 4).
    if is_scalar:
        rsrc_v4 = _ptr8_to_v4i32(rsrc)
        cache_policy = _create_i32_constant(cache_modifier)
        suffix = "i32" if vec_width == 1 else "v4i32"
        return llvm.call_intrinsic(
            result_type,
            f"llvm.amdgcn.s.buffer.load.{suffix}",
            [rsrc_v4, offset, cache_policy],
            [],
            [],
        )

    # Create instruction offset and aux flags
    if soffset_bytes is None:
        soffset = _create_i32_constant(0)
    else:
        if isinstance(soffset_bytes, int):
            soffset = _create_i32_constant(soffset_bytes)
        else:
            soffset = _to_i32_offset(_unwrap_value(soffset_bytes))
    aux_flags = _create_i32_constant(cache_modifier)

    # Emit buffer load
    load_op = rocdl.RawPtrBufferLoadOp(
        result_type,
        rsrc,
        offset,
        soffset,
        aux_flags,  # soffset (scalar byte offset)  # aux (cache modifiers)
    )

    return load_op.result


@dsl_loc_tracing
def buffer_store(
    data: ir.Value,
    rsrc: ir.Value,
    offset: ir.Value,
    mask: ir.Value | None = None,
    cache_modifier: int = 0,
    *,
    soffset_bytes: int | ir.Value | None = None,
    offset_is_bytes: bool = False,
):
    """AMD buffer store operation.

    Store data to global memory using buffer descriptor and offset.

    Args:
        data: Data to store (scalar or vector)
        rsrc: Buffer resource descriptor (!llvm.ptr<8>)
        offset: Offset in elements (i32 type)
        mask: Optional mask for predicated store (i1 type)
        cache_modifier: Cache control flags (0 for default)

    Example:
        >>> buffer_store(data, rsrc, offset)
        >>>
        >>> # Store with mask
        >>> buffer_store(data, rsrc, offset, mask=valid)
    """
    # Unwrap all inputs (accept DSL Numeric values via ir_value())
    if hasattr(data, "ir_value"):
        data = data.ir_value()
    if isinstance(offset, int):
        offset = _create_i32_constant(offset)
    elif hasattr(offset, "ir_value"):
        offset = offset.ir_value()
    data = _unwrap_value(data)
    rsrc = _unwrap_value(rsrc)
    offset = _unwrap_value(offset)

    # Convert offset to i32 if needed (accepts index/i64/i32)
    offset = _to_i32_offset(offset)

    # IMPORTANT: RawPtrBufferStoreOp offset is in BYTES.
    # For backward compat, `buffer_store()` accepts element offsets by default
    # and scales them to bytes. Set `offset_is_bytes=True` to skip scaling.
    if not offset_is_bytes:
        # Get element size from data type
        data_type = data.type
        if hasattr(data_type, "element_type"):  # Vector type
            element_type = data_type.element_type
        else:  # Scalar type
            element_type = data_type
        element_bytes = element_type.width // 8
        offset = (fx.Int32(offset) * fx.Int32(element_bytes)).ir_value()

    # Apply mask by setting invalid offsets to max
    if mask is not None:
        offset = _mask_offset(mask, offset)

    # Create instruction offset (soffset) and aux flags
    if soffset_bytes is None:
        soffset = _create_i32_constant(0)
    else:
        if isinstance(soffset_bytes, int):
            soffset = _create_i32_constant(int(soffset_bytes))
        else:
            soffset = _to_i32_offset(_unwrap_value(soffset_bytes))
    aux_flags = _create_i32_constant(cache_modifier)

    # Emit buffer store
    rocdl.RawPtrBufferStoreOp(
        data,
        rsrc,
        offset,
        soffset,
        aux_flags,  # soffset (scalar byte offset)  # aux (cache modifiers)
    )
