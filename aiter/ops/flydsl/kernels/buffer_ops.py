# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
# Modifications Copyright (C) 2026 Advanced Micro Devices, Inc.

"""AMD buffer load/store operations, vendored into aiter.

flydsl moved these from ``flydsl.expr.buffer_ops`` to its repo-level
``kernels/common/``, which its wheel does not ship, so aiter keeps an
equivalent copy here. Only published flydsl APIs are used.

Buffer instructions are an AMD hardware feature (buffer resource descriptor
plus ROCDL intrinsics) providing out-of-bounds protection and better memory
throughput; plain memref load/store is not a substitute.

Upstream: FlyDSL ``kernels/common/buffer_ops.py`` @ ROCm/FlyDSL#880.
Behavior matches upstream; formatting and type annotations differ only to
satisfy aiter's lint.

Example:
    >>> from aiter.ops.flydsl.kernels import buffer_ops
    >>> from flydsl.expr import arith
    >>>
    >>> rsrc = buffer_ops.create_buffer_resource(A)
    >>> offset = row * arith.index(4096) + col
    >>> data = buffer_ops.buffer_load(rsrc, offset, vec_width=4)
    >>> buffer_ops.buffer_store(data, rsrc, offset)
"""

from __future__ import annotations

from flydsl._mlir import ir
from flydsl._mlir.dialects import arith as std_arith
from flydsl._mlir.dialects import llvm, rocdl
from flydsl._mlir.extras import types as T
from flydsl.expr.meta import dsl_loc_tracing
from flydsl.runtime.device import is_rdna_arch


def _get_buffer_flags(arch=None):
    """Get AMD buffer resource descriptor (V#) flags word (bits 127:96).

    Constructs the 32-bit flags field for rocdl.make.buffer.rsrc, following the
    same logic as LLVM's AMDGPUToROCDL makeBufferRsrc():
      https://github.com/llvm/llvm-project/blob/main/mlir/lib/Conversion/AMDGPUToROCDL/AMDGPUToROCDL.cpp

    Bit layout (common to all architectures):
      bits [11:0]  - DST_SEL: ignored by raw buffer intrinsics
      bits [14:12] - DATA_FORMAT: must be nonzero, 7 = float
      bits [18:15] - NUM_FORMAT:  must be nonzero, 4 = 32-bit
      bit  [19]    - In nested heap (0)
      bit  [20]    - Behavior on unmap (0 = return 0 / ignore)
      bits [22:21] - Index stride for swizzles (0)
      bit  [23]    - Add thread ID (0)
      bit  [24]    - Reserved: must be 1 on RDNA, 0 on CDNA
      bits [26:25] - Reserved (0)
      bit  [27]    - Non-volatile (CDNA only, 0)
      bits [29:28] - OOB_SELECT (RDNA only): 0=structured, 2=none, 3=check offset
      bits [31:30] - Type (must be 0)

    CDNA (gfx9xx):    (7 << 12) | (4 << 15)                         = 0x20070
    RDNA (gfx10+):    (7 << 12) | (4 << 15) | (1 << 24) | (2 << 28) = 0x21020070
      - bit 24 set to 1 (required on RDNA)
      - OOB_SELECT=2 (no bounds checking, matching LLVM boundsCheck=false)
    """
    import os

    if arch is None:
        arch = os.environ.get("FLYDSL_GPU_ARCH")
    flags = (7 << 12) | (4 << 15)
    if is_rdna_arch(arch):
        flags |= 1 << 24  # reserved bit, must be 1 on RDNA
        flags |= 2 << 28  # OOB_SELECT = 2 (no bounds checking)
    return flags


__all__ = [
    "BufferResourceDescriptor",
    "buffer_load",
    "buffer_store",
    "create_buffer_resource",
    "create_buffer_resource_from_addr",
    "create_llvm_ptr",
    "extract_base_index",
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


@dsl_loc_tracing
def _create_i32_constant(value: int) -> ir.Value:
    """Create i32 constant using standard MLIR arith dialect."""
    i32_type = T.i32()
    if value > 0x7FFFFFFF:
        value = int(value - 2**32)
    attr = ir.IntegerAttr.get(i32_type, value)
    op = std_arith.ConstantOp(i32_type, attr)
    return _unwrap_value(op.result)


@dsl_loc_tracing
def _create_i16_constant(value: int) -> ir.Value:
    """Create i16 constant using standard MLIR arith dialect."""
    i16_type = T.i16()
    attr = ir.IntegerAttr.get(i16_type, value)
    op = std_arith.ConstantOp(i16_type, attr)
    return _unwrap_value(op.result)


@dsl_loc_tracing
def _create_i64_constant(value: int) -> ir.Value:
    """Create i64 constant using standard MLIR arith dialect."""
    i64_type = T.i64()
    attr = ir.IntegerAttr.get(i64_type, value)
    op = std_arith.ConstantOp(i64_type, attr)
    return _unwrap_value(op.result)


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
        i64_type = T.i64()
        value = _unwrap_value(std_arith.IndexCastOp(i64_type, value).result)
    ptr_type = ir.Type.parse(f"!llvm.ptr<{address_space}>")
    return llvm.IntToPtrOp(ptr_type, value).result


@dsl_loc_tracing
def extract_base_index(tensor, address_space: int = 1) -> ir.Value:
    """Extract the base address of a fly.memref as an index value.

    Inverse of :func:`create_llvm_ptr` (index -> ptr). Useful when ISA
    requires a raw pointer instead of a buffer resource descriptor
    (e.g. global_atomic_pk_add_bf16 on gfx942).
    """
    from flydsl._mlir.dialects import fly as _fly
    from flydsl._mlir.dialects import memref as _memref

    raw = _unwrap_value(tensor)
    try:
        ir.MemRefType(raw.type)
        return _memref.extract_aligned_pointer_as_index(raw)
    except ValueError:
        pass

    ptr_type = ir.Type.parse(f"!llvm.ptr<{address_space}>")
    ptr = _fly.extract_aligned_pointer_as_index(ptr_type, raw)
    i64_val = llvm.PtrToIntOp(ir.IntegerType.get_signless(64), ptr).result
    return _unwrap_value(std_arith.IndexCastOp(ir.IndexType.get(), i64_val).result)


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


class BufferResourceDescriptor:
    """AMD Buffer Resource Descriptor

    A buffer resource descriptor contains:
    - base_pointer: Scalar base pointer (wave-uniform, stored in SGPRs)
    - stride: Stride for structured buffers (typically 0 for contiguous)
    - num_records: Buffer size in bytes
    - flags: Data format and access flags

    The descriptor is stored in a special LLVM pointer type (!llvm.ptr<8>)
    """

    def __init__(self, rsrc: ir.Value):
        """Initialize with ROCDL resource descriptor value."""
        self.rsrc = rsrc

    @staticmethod
    @dsl_loc_tracing
    def from_memref(
        memref_val: ir.Value,
        stride: int = 0,
        max_size: bool = True,
        data_format: str = "f32",
        num_records_bytes: int | ir.Value | None = None,
        base_byte_offset: int | ir.Value | None = None,
    ) -> BufferResourceDescriptor:
        """Create buffer resource descriptor from memref.

        Args:
            memref_val: Memref value to create descriptor for
            stride: Stride in elements (0 for contiguous)
            max_size: If True, use max buffer size for flexibility
            num_records_bytes: Override buffer size (in BYTES) used by hardware OOB checking.
                              If provided, this takes precedence over `max_size`.
            base_byte_offset: Optional byte offset added to the descriptor base pointer.
            data_format: Data format ('f32', 'f16', 'i32', etc.)

        Returns:
            BufferResourceDescriptor instance

        Example:
            >>> rsrc = BufferResourceDescriptor.from_memref(A)
        """
        # Extract raw pointer from fly.memref.
        raw_val = _unwrap_value(memref_val)
        from flydsl._mlir.dialects import fly as _fly

        ptr_type = ir.Type.parse("!llvm.ptr")
        base_ptr = _fly.extract_aligned_pointer_as_index(ptr_type, raw_val)
        if base_byte_offset is not None:
            base_ptr = get_element_ptr(base_ptr, byte_offset=base_byte_offset)

        # Create buffer resource descriptor
        flags_val = _get_buffer_flags()
        flags = _create_i32_constant(flags_val)
        stride_val = _create_i16_constant(stride)

        def _num_records_from_memref_type() -> int | None:
            """Best-effort: derive logical buffer size (in bytes) from static memref type."""
            try:
                mt = ir.MemRefType(_unwrap_value(memref_val).type)
                shape = list(mt.shape)
                if any(int(d) < 0 for d in shape):
                    return None
                # Compute element size in bytes (scalar element type).
                elem_t = mt.element_type
                elem_bits = getattr(elem_t, "width", None)
                if elem_bits is None:
                    return None
                elem_bytes = int(elem_bits) // 8
                if elem_bytes <= 0:
                    return None
                num_elems = 1
                for d in shape:
                    num_elems *= int(d)
                return int(num_elems) * int(elem_bytes)
            # best-effort size probe: any failure just means "unknown"
            except Exception:  # noqa: BLE001
                return None

        if num_records_bytes is not None:
            # Caller-provided size in BYTES (preferred for exact hardware OOB behavior).
            if isinstance(num_records_bytes, int):
                nbytes = int(num_records_bytes)
                nbytes = max(0, nbytes)
                # Descriptor uses i32 bytes; clamp to the max representable.
                nbytes = min(nbytes, 0xFFFFFFFF)
                num_records = _create_i64_constant(nbytes)
            else:
                v = _unwrap_value(num_records_bytes)
                i64_type = T.i64()
                if not isinstance(v.type, ir.IntegerType) or v.type.width != 64:
                    if isinstance(v.type, ir.IndexType):
                        op = std_arith.IndexCastOp(i64_type, v)
                    else:
                        op = std_arith.ExtSIOp(i64_type, v)
                    v = _unwrap_value(op.result)
                num_records = v
        elif max_size:
            # Use max for flexibility (hardware will check actual bounds)
            # Note: FlyDSL's rocdl.make.buffer.rsrc requires i32, not i64
            num_records = _create_i64_constant(0xFFFFFFFF)  # FALLBACK_MAX_SIZE
        else:
            # Use the logical memref size (in bytes) for hardware OOB checking.
            nbytes = _num_records_from_memref_type()
            if nbytes is None:
                # Fall back to max-size if we can't infer statically.
                num_records = _create_i64_constant(0xFFFFFFFF)
            else:
                nbytes = min(nbytes, 0xFFFFFFFF)
                num_records = _create_i64_constant(int(nbytes))

        # Create resource descriptor (returns !llvm.ptr<8>)
        rsrc_type = ir.Type.parse("!llvm.ptr<8>")
        rsrc = rocdl.MakeBufferRsrcOp(
            rsrc_type, base_ptr, stride_val, num_records, flags
        ).result

        return BufferResourceDescriptor(rsrc)


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
    addr_i64 = _unwrap_value(addr_i64)
    ptr_type = ir.Type.parse("!llvm.ptr")
    base_ptr = llvm.IntToPtrOp(ptr_type, addr_i64).result
    flags = _create_i32_constant(_get_buffer_flags())
    stride = _create_i16_constant(0)
    if num_records_bytes is None:
        num_records = _create_i64_constant(0xFFFFFFFF)
    elif isinstance(num_records_bytes, int):
        nbytes = max(0, min(int(num_records_bytes), 0xFFFFFFFF))
        num_records = _create_i64_constant(nbytes)
    else:
        num_records = _unwrap_value(num_records_bytes)
        i64_type = T.i64()
        if (
            not isinstance(num_records.type, ir.IntegerType)
            or num_records.type.width != 64
        ):
            if isinstance(num_records.type, ir.IndexType):
                num_records = _unwrap_value(
                    std_arith.IndexCastOp(i64_type, num_records).result
                )
            else:
                num_records = _unwrap_value(
                    std_arith.ExtSIOp(i64_type, num_records).result
                )
    rsrc_type = ir.Type.parse("!llvm.ptr<8>")
    return rocdl.MakeBufferRsrcOp(
        rsrc_type, base_ptr, stride, num_records, flags
    ).result


@dsl_loc_tracing
def create_buffer_resource(
    memref_val: ir.Value,
    stride: int = 0,
    max_size: bool = True,
    *,
    num_records_bytes: int | ir.Value | None = None,
    base_byte_offset: int | ir.Value | None = None,
) -> ir.Value:
    """Create AMD buffer resource descriptor from memref.

    This is a simplified wrapper around BufferResourceDescriptor.from_memref()
    that returns the raw ROCDL resource value.

    Args:
        memref_val: Memref value
        stride: Buffer stride (0 for contiguous)
        max_size: Use maximum buffer size
        num_records_bytes: Override buffer size in bytes.
        base_byte_offset: Optional byte offset added to the descriptor base pointer.

    Returns:
        ROCDL buffer resource descriptor (!llvm.ptr<8>)

    Example:
        >>> rsrc = create_buffer_resource(A)
        >>> data = buffer_load(rsrc, offset)
    """
    desc = BufferResourceDescriptor.from_memref(
        memref_val,
        stride,
        max_size,
        num_records_bytes=num_records_bytes,
        base_byte_offset=base_byte_offset,
    )
    return desc.rsrc


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

    # Convert offset to i32 if needed
    if not isinstance(offset.type, ir.IntegerType) or offset.type.width != 32:
        op = std_arith.IndexCastOp(T.i32(), offset)
        offset = _unwrap_value(op.result)

    # IMPORTANT: Buffer load offset is in BYTES, not elements!
    # For vec4xf32, each element is 4 bytes, so multiply offset by 4
    element_bytes = dtype.width // 8
    bytes_const = _create_i32_constant(element_bytes)
    op = std_arith.MulIOp(offset, bytes_const)
    offset = _unwrap_value(op.result)

    # Apply mask by setting invalid offsets to max
    if mask is not None:
        mask = _unwrap_value(mask)
        max_offset = _create_i32_constant(0x7FFFFFFF)
        op = std_arith.SelectOp(mask, offset, max_offset)
        offset = _unwrap_value(op.result)

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
            soffset = _unwrap_value(soffset_bytes)
            if not isinstance(soffset.type, ir.IntegerType) or soffset.type.width != 32:
                op = std_arith.IndexCastOp(T.i32(), soffset)
                soffset = _unwrap_value(op.result)
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

    # Convert offset to i32 if needed
    if not isinstance(offset.type, ir.IntegerType) or offset.type.width != 32:
        op = std_arith.IndexCastOp(T.i32(), offset)
        offset = _unwrap_value(op.result)

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
        bytes_const = _create_i32_constant(element_bytes)
        op = std_arith.MulIOp(offset, bytes_const)
        offset = _unwrap_value(op.result)

    # Apply mask by setting invalid offsets to max
    if mask is not None:
        mask = _unwrap_value(mask)
        max_offset = _create_i32_constant(0x7FFFFFFF)
        op = std_arith.SelectOp(mask, offset, max_offset)
        offset = _unwrap_value(op.result)

    # Create instruction offset (soffset) and aux flags
    if soffset_bytes is None:
        soffset = _create_i32_constant(0)
    else:
        if isinstance(soffset_bytes, int):
            soffset = _create_i32_constant(int(soffset_bytes))
        else:
            soffset = _unwrap_value(soffset_bytes)
            if not isinstance(soffset.type, ir.IntegerType) or soffset.type.width != 32:
                op = std_arith.IndexCastOp(T.i32(), soffset)
                soffset = _unwrap_value(op.result)
    aux_flags = _create_i32_constant(cache_modifier)

    # Emit buffer store
    rocdl.RawPtrBufferStoreOp(
        data,
        rsrc,
        offset,
        soffset,
        aux_flags,  # soffset (scalar byte offset)  # aux (cache modifiers)
    )
