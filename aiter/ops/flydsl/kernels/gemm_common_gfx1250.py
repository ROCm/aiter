"""Shared utilities for gfx1250 GEMM kernels (fp16 / mxfp4 / mxfp8). """

from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm as llvm_dialect, scf
from flydsl.expr import arith, buffer_ops, gpu, rocdl, tdm_ops, vector
from flydsl.expr.arith import _to_raw as _raw
from flydsl.expr.typing import T
from flydsl.utils.smem_allocator import (
    SmemPtr, get_mlir_type_size, get_op_result_or_value,
)


def get_lds_memref(lds_ptr):
    """Get the raw memref value from SmemPtr or raw memref."""
    if isinstance(lds_ptr, SmemPtr):
        return get_op_result_or_value(lds_ptr.get())
    return get_op_result_or_value(lds_ptr)


def _lds_vec_type(memref, total_bits):
    """Build a vector type matching *memref*'s element type for *total_bits*."""
    raw_mr = arith.unwrap(memref)
    elem_type = ir.MemRefType(raw_mr.type).element_type
    elem_bits = get_mlir_type_size(elem_type) * 8
    n = total_bits // elem_bits
    return ir.VectorType.get([n], elem_type)


def lds_load_b128(memref, elem_off):
    """Load 16 bytes from LDS as ``vector<4×i32>``.

    Automatically adapts to the memref element type (f16, bf16, f32, etc.).
    Produces ``ds_load_b128``.

    Args:
        memref: LDS memref (any 16-bit or 32-bit element type, address-space 3).
        elem_off: Element offset in memref element units.
    """
    vec_ty = _lds_vec_type(memref, 128)
    loaded = vector.load_op(vec_ty, memref, [elem_off])
    return vector.bitcast(ir.VectorType.get([4], ir.IntegerType.get_signless(32)), loaded)


def lds_store_b128(memref, elem_off, data):
    """Store 16 bytes to LDS.

    Bitcasts *data* to match the memref element type, then calls
    ``vector.store``.  Produces ``ds_store_b128``.

    Args:
        memref: LDS memref (any 16-bit or 32-bit element type, address-space 3).
        elem_off: Element offset in memref element units.
        data: Any 128-bit vector (``vec<4×i32>``, ``vec<4×f32>``,
              ``vec<8×f16>``, ``vec<8×bf16>``).
    """
    vec_ty = _lds_vec_type(memref, 128)
    typed_vec = vector.bitcast(vec_ty, data)
    vector.store(typed_vec, memref, [elem_off])


def extract_lds_base_idx(smem_ptr):
    """Extract the absolute LDS byte-base address as an index value."""
    from flydsl._mlir.dialects import memref as _memref

    membuf = get_lds_memref(smem_ptr)
    raw_memref = arith.unwrap(membuf)
    return _memref.extract_aligned_pointer_as_index(raw_memref)


def _raw_lds_ptr(lds_base_idx, byte_offset):
    """Materialize an LLVM LDS pointer from a pre-extracted byte base."""
    from flydsl._mlir.dialects import llvm as _llvm
    from flydsl.expr.arith import ArithValue as _AV

    lds_ptr_ty = ir.Type.parse("!llvm.ptr<3>")
    total_byte = _AV(lds_base_idx) + byte_offset
    addr_i32 = _raw(arith.index_cast(T.i32, total_byte))
    return _llvm.inttoptr(lds_ptr_ty, addr_i32)


def lds_load_b128_raw(lds_base_idx, byte_offset):
    """Load 16 bytes from LDS using a pre-extracted base index (raw LLVM).

    Args:
        lds_base_idx: Index value from ``extract_lds_base_idx``.
        byte_offset: Byte offset (index-type) relative to the base.
    """
    ptr_val = _raw_lds_ptr(lds_base_idx, byte_offset)
    return llvm_dialect.load(ir.VectorType.get([4], ir.IntegerType.get_signless(32)), ptr_val)


def lds_transpose_load_raw(result_type, lds_base_idx, byte_offset):
    """Transpose-load 16 bytes from LDS using a pre-extracted base index."""
    from flydsl._mlir.dialects import rocdl as _rocdl

    ptr_val = _raw_lds_ptr(lds_base_idx, byte_offset)
    return _rocdl.ds_load_tr16_b128(result_type, ptr_val)


def workgroup_barrier(use_cluster=False):
    """Issue the appropriate barrier for LDS visibility.

    Cluster mode layers an inter-workgroup barrier on top of the regular
    workgroup barrier protocol, so call sites can treat it as a single
    "LDS is now readable" fence.
    """
    if use_cluster:
        gpu.cluster_barrier()
    else:
        gpu.barrier()


def pipeline_fence(outstanding=0, use_cluster=False):
    """Fused READY+REUSE fence for gfx1250 multi-buffer pipeline.

    Issues ``s_wait_tensorcnt`` followed by the appropriate barrier.
    """
    tdm_ops.tensor_wait(outstanding)
    workgroup_barrier(use_cluster=use_cluster)


WGP_BARRIER_ID = -1


def pipeline_fence_signal(outstanding=0, use_cluster=False):
    """Signal half of a split barrier fence.

    Issues ``s_wait_tensorcnt`` then ``s_barrier_signal -1``.
    The matching ``pipeline_fence_wait`` must be called later
    (typically mid-compute) before reading the LDS data.

    When *use_cluster* is True the intra-WG barrier is still required
    so that all waves' TDM loads are visible before any wave reads LDS.
    The cluster barrier is layered on top for inter-WG synchronisation.
    """
    tdm_ops.tensor_wait(outstanding)
    rocdl.s_barrier_signal(WGP_BARRIER_ID)
    if use_cluster:
        gpu.cluster_signal_once_per_wg()


def pipeline_fence_wait(use_cluster=False):
    """Wait half of a split barrier fence.

    Issues ``s_barrier_wait -1``.  Must be preceded by a matching
    ``pipeline_fence_signal`` from all waves in the workgroup.
    """
    rocdl.s_barrier_wait(WGP_BARRIER_ID)
    if use_cluster:
        gpu.cluster_wait()


def issue_tdm_loads(*descs, wave_specialized=False, wave_id=None):
    """Emit one or more TDM loads, optionally one descriptor per loader wave."""
    if wave_specialized:
        if wave_id is None:
            wave_id = rocdl.wave_id()
        for idx, desc in enumerate(descs):
            is_loader_wave = arith.cmpi(
                arith.CmpIPredicate.eq,
                wave_id,
                arith.constant(idx, type=T.i32),
            )
            if_op = scf.IfOp(is_loader_wave)
            with ir.InsertionPoint(if_op.then_block):
                tdm_ops.tensor_load_2d(desc)
                scf.YieldOp([])
        return

    for desc in descs:
        tdm_ops.tensor_load_2d(desc)


def store_acc_vec8_to_lds(memref, base_elem_off, imm_elem_off, acc_vec8,
                          out_elem=None):
    """Write one 8-element f32 accumulator sub-vector to LDS.

    For half output (out_elem = T.f16 or T.bf16):
        trunc_f → bitcast(vec<4×i32>) → 1 × lds_store_b128  (16 bytes)
    For f32 output (out_elem = None):
        extract×4 → from_elements(vec<4×f32>) → 2 × lds_store_b128  (32 bytes)

    Args:
        memref: D-output LDS memref (f16 element type).
        base_elem_off: Per-lane base element offset (VGPR).
        imm_elem_off: Compile-time element offset for this sub-tile.
        acc_vec8: ``vector<8×f32>`` accumulator values.
        out_elem: Output element type (``T.f16``, ``T.bf16``, or ``None`` for f32).
    """
    off = base_elem_off + arith.index(imm_elem_off)
    if out_elem is not None:
        h_vec = arith.trunc_f(T.vec(8, out_elem), acc_vec8)
        i32_vec = vector.bitcast(T.vec(4, T.i32), h_vec)
        lds_store_b128(memref, off, i32_vec)
    else:
        for half in range(2):
            vals = [vector.extract(acc_vec8,
                                   static_position=[half * 4 + vi],
                                   dynamic_position=[])
                    for vi in range(4)]
            vec4 = vector.from_elements(T.vec(4, T.f32), vals)
            lds_store_b128(memref, off + arith.index(half * 8), vec4)


def store_acc_vec8_to_buffer(acc_vec8, c_rsrc, addr,
                             out_elem=None, offset_is_bytes=False):
    """Write one 8-element f32 accumulator sub-vector to global memory.

    For half output (out_elem = T.f16 or T.bf16):
        trunc_f → bitcast(vec<4×i32>) → 1 × buffer_store (16 bytes)
    For f32 output (out_elem = None):
        extract×4 → from_elements(vec<4×f32>) → 2 × buffer_store (16 bytes each)

    Args:
        acc_vec8: ``vector<8×f32>`` accumulator values.
        c_rsrc: Buffer resource descriptor for the output matrix.
        addr: Pre-computed address (single value for half, list of 2 for f32).
        out_elem: Output element type (``T.f16``, ``T.bf16``, or ``None`` for f32).
        offset_is_bytes: If True, treat addr as byte offset (half output path).

    Returns:
        Number of addr slots consumed (1 for half, 2 for f32).
    """
    if out_elem is not None:
        h_vec = arith.trunc_f(T.vec(8, out_elem), acc_vec8)
        i32_vec = vector.bitcast(T.vec(4, T.i32), h_vec)
        buffer_ops.buffer_store(i32_vec, c_rsrc, addr,
                                offset_is_bytes=offset_is_bytes)
        return 1
    else:
        for half in range(2):
            vals = [vector.extract(acc_vec8,
                                   static_position=[half * 4 + vi],
                                   dynamic_position=[])
                    for vi in range(4)]
            vec4 = vector.from_elements(T.vec(4, T.f32), vals)
            if isinstance(addr, (list, tuple)):
                buffer_ops.buffer_store(vec4, c_rsrc, addr[half])
            else:
                buffer_ops.buffer_store(vec4, c_rsrc, addr)
        return 2


__all__ = [
    # LDS helpers
    "get_lds_memref",
    # Raw LLVM path
    "extract_lds_base_idx",
    "lds_load_b128_raw",
    "lds_transpose_load_raw",
    # Pipeline
    "workgroup_barrier",
    "pipeline_fence",
    "pipeline_fence_signal",
    "pipeline_fence_wait",
    "issue_tdm_loads",
    # Epilogue
    "store_acc_vec8_to_lds",
    "store_acc_vec8_to_buffer",
]
