# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""FlyDSL dynamic per-tensor scaled quantization.

Implements the equivalent of ``aiter/csrc/kernels/quant_kernels.cu``'s
``dynamic_per_tensor_quant`` -- compute a single global ``scale = max(|x|) /
dtype_max`` and emit ``y = x / scale`` quantized to fp8 (E4M3 FN/FNUZ).

Two GPU kernels are launched in sequence (mirroring the CUDA reference):

  1. ``data_to_scale``:  one block per input row reduces ``max(|x|)`` across
                          the row, then a single lane atomically ``fmax``-
                          updates the global ``scale[0]``.
  2. ``scaled_quant``:   one block per input row reads ``scale[0]`` once,
                          computes ``inv_scale = 1/scale`` and writes the
                          quantized fp8 output.

Design notes
~~~~~~~~~~~~
- ``cols`` is a compile-time constant; the row stride is assumed equal to
  ``cols`` (matches the C++ reference, which uses ``input.numel()/cols``).
- ``cols`` must be a multiple of ``VEC`` (=8) so each vector load is either
  fully in-bounds or fully out-of-bounds. The AMD ``raw_ptr_buffer_load`` op
  returns zero for OOB lanes (via the ``mask`` parameter), which means the
  reduction sees a no-op contribution for tail iterations.
- ``BLOCK_THREADS = 256`` matches the CUDA reference. With wave64 (CDNA),
  that's ``NUM_WAVES = 4`` waves per block. Cross-wave block-reduce uses LDS
  (4 f32 slots) plus wave0 shuffle.
- FP8 packing uses ``v_cvt_pk_fp8_f32`` (``rocdl.cvt_pk_fp8_f32``): two
  invocations per i32 dword pack four f32s into four fp8 bytes.

Public entrypoint
~~~~~~~~~~~~~~~~~
``build_dynamic_per_tensor_quant_module(cols, in_dtype, out_dtype)`` returns a
``@flyc.jit``-compiled launcher ``launch(input, out, scale, rows, stream)``.
Wrap it from ``flydsl_dynamic_per_tensor_quant`` in ``quant_kernels.py``.
"""

from __future__ import annotations

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.compiler.kernel_function import CompilationContext

from flydsl.expr import arith, gpu, vector, range_constexpr
from flydsl.expr import buffer_ops, rocdl
from flydsl.expr.arith import ArithValue, CmpIPredicate
from flydsl.expr.typing import T, Int32
from flydsl.runtime.device import get_rocm_arch as get_hip_arch
from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr

from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm as _llvm, scf as _scf

__all__ = [
    "build_dynamic_per_tensor_quant_module",
    "build_per_1x32_fp4_quant_module",
    "build_per_1x32_fp4_quant_hadamard_module",
    "build_per_1x32_fp4_quant_block_rotation_module",
    "build_per_1x32_fp4_quant_block_rotation_mfma_module",
]

# ----------------------------------------------------------------------------
# Constants matching the CUDA reference (quant_kernels.cu).
# ----------------------------------------------------------------------------
BLOCK_THREADS = 256
WARP_SIZE = 64
NUM_WAVES = BLOCK_THREADS // WARP_SIZE  # 4 on CDNA wave64
# Each thread handles VEC input elements per iteration. VEC=8 corresponds to a
# 16-byte vector load when the input is bf16/fp16.
VEC = 8

# `fp8` here refers to E4M3 (FN/FNUZ depending on arch). Same max magnitude
# either way -- so the scale derivation is identical for both.
_FP8_E4M3_MAX = 448.0


def _dtype_max(out_dtype: str) -> float:
    if out_dtype == "fp8":
        return _FP8_E4M3_MAX
    raise ValueError(f"unsupported output dtype: {out_dtype!r}")


def _input_elem_mlir_type(in_dtype: str):
    """Resolve the bf16/fp16 element type. **Must be called inside an MLIR
    Context** (i.e. from within a ``@flyc.kernel`` body) because flydsl's ``T``
    accessors lazily construct ``ir.Type`` instances that need an active
    context."""
    if in_dtype == "bf16":
        return T.bf16
    if in_dtype in ("fp16", "f16"):
        return T.f16
    raise ValueError(f"unsupported input dtype: {in_dtype!r}")


# ----------------------------------------------------------------------------
# Kernel builder (cached so we don't re-JIT for the same shape/dtype combo).
# ----------------------------------------------------------------------------
@functools.lru_cache(maxsize=None)
def build_dynamic_per_tensor_quant_module(
    cols: int,
    in_dtype: str = "bf16",
    out_dtype: str = "fp8",
):
    """Build (and cache) a launcher for fp8 per-tensor dynamic quant.

    Parameters
    ----------
    cols: int
        Last-dim size of the input (= row stride). Must be a positive multiple
        of ``VEC`` (=8).
    in_dtype: {"bf16", "fp16"}
        Element dtype of the input tensor.
    out_dtype: {"fp8"}
        Output element dtype. Only fp8 (E4M3) is implemented in Step 1.
    """
    if cols <= 0 or (cols % VEC) != 0:
        raise ValueError(
            f"cols must be a positive multiple of VEC={VEC}, got cols={cols}"
        )
    if out_dtype != "fp8":
        raise NotImplementedError(
            f"Step 1 only implements fp8 output; got out_dtype={out_dtype!r}"
        )
    if in_dtype not in ("bf16", "fp16", "f16"):
        raise ValueError(f"unsupported input dtype: {in_dtype!r}")

    elem_bytes_in = 2  # bf16/fp16

    dtype_max_val = _dtype_max(out_dtype)
    inv_dtype_max_val = 1.0 / dtype_max_val

    # Each thread handles VEC elements; block handles BLOCK_THREADS*VEC per iter.
    cols_per_iter = BLOCK_THREADS * VEC
    num_iters = (cols + cols_per_iter - 1) // cols_per_iter

    # ----- LDS allocator for cross-wave block reduce (NUM_WAVES f32s = 16B). -----
    gpu_arch = get_hip_arch()
    # Use a unique sym name per cached build to avoid global collisions.
    sym_tag = f"dq_pt_lds_{cols}_{in_dtype}_{out_dtype}"
    allocator = SmemAllocator(None, arch=gpu_arch, global_sym_name=sym_tag)
    lds_red_offset = allocator._align(allocator.ptr, 16)
    allocator.ptr = lds_red_offset + NUM_WAVES * 4  # NUM_WAVES * sizeof(f32)

    # ------------------------------------------------------------------
    # Kernel A: data_to_scale -- compute global max(|x|)/dtype_max via atomic.
    # ------------------------------------------------------------------
    @flyc.kernel
    def data_to_scale_kernel(
        inp: fx.Tensor,    # (rows, cols)  bf16 / fp16
        scale: fx.Tensor,  # (1,)          f32, pre-zeroed by host
    ):
        bid = fx.block_idx.x  # one block per row
        tid = fx.thread_idx.x

        f32 = T.f32
        i32 = T.i32
        in_elem_ty = _input_elem_mlir_type(in_dtype)
        in_vec_ty = T.vec(VEC, in_elem_ty)

        c0_i32 = arith.constant(0, type=i32)
        c1_i32 = arith.constant(1, type=i32)
        c64_i32 = arith.constant(WARP_SIZE, type=i32)
        c0_f32 = arith.constant(0.0, type=f32)
        c_inv_dmax = arith.constant(inv_dtype_max_val, type=f32)
        cols_i32 = arith.constant(cols, type=i32)
        vec_i32 = arith.constant(VEC, type=i32)

        # NOTE on OOB protection: ``buffer_load(..., mask=...)`` relies on the
        # rsrc's bounds-check to swallow loads, but for dynamic-shape memrefs
        # FlyDSL falls back to ``num_records = 0xFFFFFFFF`` regardless of
        # ``max_size``. We therefore do manual offset clamping + arith.select
        # for the out-of-range lanes (see the loop body below).
        in_rsrc = buffer_ops.create_buffer_resource(inp, max_size=True)
        scale_rsrc = buffer_ops.create_buffer_resource(scale, max_size=True)

        tid_i32 = ArithValue(tid)
        bid_i32 = ArithValue(bid)

        # row_elem_base = bid * cols  (element index of row start)
        row_elem_base = bid_i32 * cols_i32

        # ----- Per-thread max accumulator (lives in registers, no scf scope) -----
        local_max = c0_f32
        for it in range_constexpr(num_iters):
            col_thread = (
                tid_i32 * vec_i32
                + arith.constant(it * cols_per_iter, type=i32)
            )
            in_range = arith.cmpi(CmpIPredicate.ult, col_thread, cols_i32)

            elem_off = row_elem_base + col_thread
            # Clamp the offset to the row start when this lane is OOR so the
            # buffer load always touches in-bounds memory; we still drop the
            # contribution below via ``arith.select``.
            safe_elem_off = arith.select(in_range, elem_off, row_elem_base)
            dw_off = safe_elem_off >> c1_i32  # 2 bf16/fp16 per dword
            vec_dw = VEC * elem_bytes_in // 4  # = 4 dwords for VEC=8

            raw_i32 = buffer_ops.buffer_load(
                in_rsrc, dw_off, vec_width=vec_dw, dtype=i32
            )
            x_vec = vector.bitcast(in_vec_ty, raw_i32)
            x_f32 = x_vec.extf(T.vec(VEC, f32))

            for vi in range_constexpr(VEC):
                v = vector.extract(
                    x_f32, static_position=[vi], dynamic_position=[]
                )
                abs_v = _llvm.call_intrinsic(
                    f32, "llvm.fabs.f32", [v], [], []
                )
                # OOR lanes contribute 0 to the running max.
                safe_abs_v = arith.select(in_range, abs_v, c0_f32)
                local_max = arith.maximumf(local_max, safe_abs_v)

        # ----- Intra-wave reduce via shuffle-xor (wave_size=64). -----
        for sh in [32, 16, 8, 4, 2, 1]:
            off = arith.constant(sh, type=i32)
            peer = local_max.shuffle_xor(off, c64_i32)
            local_max = arith.maximumf(local_max, peer)

        # ----- Cross-wave reduce via LDS + wave0 shuffle. -----
        lane_i32 = tid_i32 & arith.constant(WARP_SIZE - 1, type=i32)
        wave_i32 = tid_i32 >> arith.constant(6, type=i32)  # /64

        # Materialize the LDS view at kernel entry (outside all scf blocks) so
        # it dominates every store/load below. SmemPtr lazily caches the view,
        # so calling .get() here forces emission of memref.view in this scope.
        base_ptr = allocator.get_base()
        lds_red_ptr = SmemPtr(
            base_ptr, lds_red_offset, T.f32, shape=(NUM_WAVES,)
        )
        lds_red_view = lds_red_ptr.get()

        from flydsl._mlir.dialects import memref as _memref

        is_lane0 = arith.cmpi(CmpIPredicate.eq, lane_i32, c0_i32)
        # The if-block writes to LDS only; no value escapes via Python scope.
        _if_l0 = _scf.IfOp(is_lane0)
        with ir.InsertionPoint(_if_l0.then_block):
            wave_idx_index = arith.index_cast(T.index, wave_i32)
            _memref.store(local_max, lds_red_view, [wave_idx_index])
            _scf.YieldOp([])

        gpu.barrier()  # Workgroup-wide sync before wave0 reads partials.

        is_wave0 = arith.cmpi(CmpIPredicate.eq, wave_i32, c0_i32)
        _if_w0 = _scf.IfOp(is_wave0)
        with ir.InsertionPoint(_if_w0.then_block):
            in_range_lane = arith.cmpi(
                CmpIPredicate.ult,
                lane_i32,
                arith.constant(NUM_WAVES, type=i32),
            )
            # Clamp lane to 0 for OOR reads, then mask back to 0 via select.
            safe_lane = arith.select(in_range_lane, lane_i32, c0_i32)
            safe_lane_idx = arith.index_cast(T.index, safe_lane)
            loaded = _memref.load(lds_red_view, [safe_lane_idx])
            partial = arith.select(in_range_lane, loaded, c0_f32)
            # NUM_WAVES <= 64 so xor distances up to 32 are harmless for OOR
            # lanes (which contributed 0 to the running max).
            for sh in [32, 16, 8, 4, 2, 1]:
                off = arith.constant(sh, type=i32)
                peer = partial.shuffle_xor(off, c64_i32)
                partial = arith.maximumf(partial, peer)

            # Lane 0 of wave 0 publishes the block-level max into the global
            # scale via atomic max. Pre-multiplying by inv_dtype_max is
            # mathematically equivalent (positive factor) and avoids a second
            # pass to convert max(|x|) -> scale.
            #
            # NOTE: gfx9xx LLVM backend can't currently select BUFFER_ATOMIC
            # _FMAX on f32, so we mirror the CUDA reference and use an integer
            # atomic-max on the bit pattern of the non-negative f32 value. For
            # non-negative IEEE-754 floats the i32 bit pattern is monotonic in
            # value, so int max == float max.
            is_lane0_w0 = arith.cmpi(CmpIPredicate.eq, lane_i32, c0_i32)
            _if_atom = _scf.IfOp(is_lane0_w0)
            with ir.InsertionPoint(_if_atom.then_block):
                row_scale_f32 = partial * c_inv_dmax
                row_scale_i32 = row_scale_f32.bitcast(i32)
                rocdl.raw_ptr_buffer_atomic_smax(
                    row_scale_i32,
                    scale_rsrc,
                    c0_i32,   # byte offset (= 0, single-element buffer)
                    c0_i32,   # soffset
                    c0_i32,   # aux flags
                )
                _scf.YieldOp([])
            _scf.YieldOp([])

    # ------------------------------------------------------------------
    # Kernel B: scaled_quant -- apply ``y = x / scale`` and pack to fp8.
    # ------------------------------------------------------------------
    @flyc.kernel
    def scaled_quant_kernel(
        inp: fx.Tensor,    # (rows, cols)  bf16 / fp16
        out: fx.Tensor,    # (rows, cols)  fp8  (raw byte buffer)
        scale: fx.Tensor,  # (1,) f32, populated by data_to_scale_kernel
    ):
        bid = fx.block_idx.x
        tid = fx.thread_idx.x

        f32 = T.f32
        i32 = T.i32
        in_elem_ty = _input_elem_mlir_type(in_dtype)
        in_vec_ty = T.vec(VEC, in_elem_ty)

        c0_i32 = arith.constant(0, type=i32)
        c1_i32 = arith.constant(1, type=i32)
        cols_i32 = arith.constant(cols, type=i32)
        vec_i32 = arith.constant(VEC, type=i32)

        # See note in ``data_to_scale_kernel`` -- we use manual
        # offset-clamp + arith.select for OOB protection (the rsrc-mask trick
        # doesn't work for dynamic-shape memrefs).
        in_rsrc = buffer_ops.create_buffer_resource(inp, max_size=True)
        out_rsrc = buffer_ops.create_buffer_resource(out, max_size=True)
        scale_rsrc = buffer_ops.create_buffer_resource(scale, max_size=True)

        tid_i32 = ArithValue(tid)
        bid_i32 = ArithValue(bid)

        # Uniform broadcast: every thread loads the same scale[0] from gmem.
        scale_val = buffer_ops.buffer_load(
            scale_rsrc, c0_i32, vec_width=1, dtype=f32
        )
        # Hardware reciprocal -- precise enough for fp8 quant.
        inv_scale = _llvm.call_intrinsic(
            f32, "llvm.amdgcn.rcp.f32", [scale_val], [], []
        )

        row_elem_base = bid_i32 * cols_i32

        for it in range_constexpr(num_iters):
            col_thread = (
                tid_i32 * vec_i32
                + arith.constant(it * cols_per_iter, type=i32)
            )
            in_range = arith.cmpi(CmpIPredicate.ult, col_thread, cols_i32)

            elem_off = row_elem_base + col_thread
            # Manual OOB clamp (see note above): load from row start when OOR
            # and gate the store with scf.IfOp.
            safe_elem_off = arith.select(in_range, elem_off, row_elem_base)
            dw_off_in = safe_elem_off >> c1_i32  # 2 bf16/fp16 per dword
            vec_dw_in = VEC * elem_bytes_in // 4  # 4

            raw_i32 = buffer_ops.buffer_load(
                in_rsrc, dw_off_in, vec_width=vec_dw_in, dtype=i32
            )
            x_vec = vector.bitcast(in_vec_ty, raw_i32)
            x_f32 = x_vec.extf(T.vec(VEC, f32))

            # Scale each lane element.
            scaled_vals = []
            for vi in range_constexpr(VEC):
                v = vector.extract(
                    x_f32, static_position=[vi], dynamic_position=[]
                )
                scaled_vals.append(v * inv_scale)

            # Pack: VEC=8 → 2 dwords of fp8. Each ``cvt_pk_fp8_f32`` call writes
            # 2 fp8 bytes (one half of an i32 dword), so 2 calls fill a dword.
            packed0 = rocdl.cvt_pk_fp8_f32(
                i32, scaled_vals[0], scaled_vals[1], c0_i32, 0
            )
            packed0 = rocdl.cvt_pk_fp8_f32(
                i32, scaled_vals[2], scaled_vals[3], packed0, 1
            )
            packed1 = rocdl.cvt_pk_fp8_f32(
                i32, scaled_vals[4], scaled_vals[5], c0_i32, 0
            )
            packed1 = rocdl.cvt_pk_fp8_f32(
                i32, scaled_vals[6], scaled_vals[7], packed1, 1
            )

            packed_vec = vector.from_elements(
                T.vec(2, i32), [packed0, packed1]
            )

            # Gate the store on in_range. We deliberately don't use
            # buffer_store(mask=...) because its mask remaps the offset to
            # 0x7FFFFFFF which is < the dynamic-shape rsrc bound (=0xFFFFFFFF).
            _if_store = _scf.IfOp(in_range)
            with ir.InsertionPoint(_if_store.then_block):
                buffer_ops.buffer_store(
                    packed_vec,
                    out_rsrc,
                    elem_off,
                    offset_is_bytes=True,
                )
                _scf.YieldOp([])

    # ------------------------------------------------------------------
    # Host-side JIT launcher: launches both kernels in sequence.
    # ------------------------------------------------------------------
    @flyc.jit
    def launch_dynamic_per_tensor_quant(
        inp: fx.Tensor,
        out: fx.Tensor,
        scale: fx.Tensor,
        rows: Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        # LDS finalize for the reduction scratch.
        allocator.finalized = False
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            allocator.finalize()

        rows_idx = arith.index_cast(T.index, rows)

        scale_launcher = data_to_scale_kernel(inp, scale)
        scale_launcher.launch(
            grid=(rows_idx, 1, 1),
            block=(BLOCK_THREADS, 1, 1),
            stream=stream,
        )

        quant_launcher = scaled_quant_kernel(inp, out, scale)
        quant_launcher.launch(
            grid=(rows_idx, 1, 1),
            block=(BLOCK_THREADS, 1, 1),
            stream=stream,
        )

    return launch_dynamic_per_tensor_quant


# ============================================================================
# MXFP4 (per-1x32) dynamic quantization
# ============================================================================
#
# Equivalent of CUDA ``dynamic_per_group_scaled_quant_kernel`` (group_size=32,
# DTYPE_O=fp4_t) from ``aiter/csrc/kernels/quant_kernels.cu`` (see lines 18-133
# and the dispatch in lines 846-896 of that file).
#
# Algorithm (mirrors the CUDA reference exactly):
#   * Each thread owns one full group of 32 input elements (``thread_data_size
#     == group_size``), so there is **no cross-thread reduction** within a
#     group.
#   * 64 threads per block, so each block produces 64 groups of fp4 output.
#   * Per thread:
#       1. Vector-load 32 bf16/fp16 elements (= 16 dwords = 64 bytes).
#       2. Compute ``absMax`` over those 32 values in registers.
#       3. Apply ``fp4_scale``: round ``absMax`` UP to the next power of 2 via
#          bit manipulation (round-to-nearest-even on the mantissa).
#       4. ``inv_scale = next_pow2 * 0.25`` (== ``next_pow2 / 4``, which is
#          the canonical OCP MX-FP4 dequant scale because the largest pow2 not
#          exceeding ``F4E2M1_MAX = 6.0`` is ``4.0``).
#       5. Write the 8-bit biased exponent of ``inv_scale`` (E8M0) into the
#          scale tensor.
#       6. Use ``v_cvt_scalef32_pk_fp4_{bf16,f16}`` 16 times (4 dwords × 4
#          sel slots) to convert 32 elements -> 16 bytes of MXFP4, applying
#          ``inv_scale`` as the HW scale factor.
#       7. Store 4 dwords (16 bytes) to the output tensor.
#
# No atomics, no LDS, no cross-block sync -- a single kernel launch suffices.
# This is why MXFP4 per-1x32 is fundamentally cheaper than per-tensor: the
# scale is local to each group of 32.

# Block / group geometry constants from the CUDA reference.
GROUP_QUANT_BLOCK_SIZE = 64  # `groupQuantBlockSize` in quant_kernels.cu
FP4_GROUP_SIZE = 32          # only group_size=32 is supported in Step 1


@functools.lru_cache(maxsize=None)
def build_per_1x32_fp4_quant_module(
    cols: int,
    in_dtype: str = "bf16",
    shuffle_scale: bool = False,
):
    """Build (and cache) a launcher for MXFP4 per-1x32 dynamic quant.

    Parameters
    ----------
    cols: int
        Last-dim size of the input (=row stride). Must be a positive multiple
        of ``FP4_GROUP_SIZE`` (=32).
    in_dtype: {"bf16", "fp16"}
        Element dtype of the input tensor.
    shuffle_scale: bool
        If True, write the E8M0 scale bytes in the "shuffled" layout used by
        downstream MXFP4 GEMM kernels. Step 1 currently only supports the
        un-shuffled (logical-order) layout; raise otherwise.

    The returned launcher signature is
    ``launch(inp, out, scale, rows, scaleN_pad, stream)`` where ``out`` is a
    ``uint8`` tensor of shape ``(rows, cols // 2)`` (each byte = 2 fp4) and
    ``scale`` is a ``uint8`` tensor of shape ``(rows, scaleN_pad)`` viewed as
    fp8_e8m0.
    """
    if cols <= 0 or (cols % FP4_GROUP_SIZE) != 0:
        raise ValueError(
            f"cols must be a positive multiple of FP4_GROUP_SIZE="
            f"{FP4_GROUP_SIZE}, got cols={cols}"
        )
    if in_dtype not in ("bf16", "fp16", "f16"):
        raise ValueError(f"unsupported input dtype: {in_dtype!r}")
    if shuffle_scale:
        raise NotImplementedError(
            "shuffle_scale layout not implemented yet; pass shuffle=False"
        )

    elem_bytes_in = 2  # bf16/fp16
    # Per group: 32 elements => 16 packed pairs => 4 dwords of fp4 output.
    PAIRS_PER_GROUP = FP4_GROUP_SIZE // 2     # 16
    DWORDS_PER_GROUP = PAIRS_PER_GROUP // 4   # 4 (each dword holds 8 fp4)
    # gfx950's BUFFER_LOAD can do at most 4xi32 (128b / 16B) per op, so we
    # split the 64-byte group into 4 chunks of 8 elements (= 4 dwords each).
    ELEMS_PER_LOAD = 8                        # 8 bf16/fp16 per load
    LOADS_PER_GROUP = FP4_GROUP_SIZE // ELEMS_PER_LOAD  # 4
    DWORDS_PER_LOAD = ELEMS_PER_LOAD * elem_bytes_in // 4  # = 4

    scale_N = cols // FP4_GROUP_SIZE  # number of groups per row

    @flyc.kernel
    def per_1x32_fp4_quant_kernel(
        inp: fx.Tensor,    # (rows, cols)             bf16 / fp16
        out: fx.Tensor,    # (rows, cols // 2)        uint8 (fp4x2)
        scale: fx.Tensor,  # (rows, scale_N) bytes,   fp8_e8m0
        total_groups: Int32,  # = rows * scale_N (number of fp4 groups to emit)
    ):
        bid = fx.block_idx.x
        tid = fx.thread_idx.x

        f32 = T.f32
        i32 = T.i32
        in_elem_ty = _input_elem_mlir_type(in_dtype)
        # Per-load vector type (8 bf16/fp16 = 4 dwords; legal 128-bit load).
        in_chunk_vec_ty = T.vec(ELEMS_PER_LOAD, in_elem_ty)

        c0_i32 = arith.constant(0, type=i32)
        c1_i32 = arith.constant(1, type=i32)
        c0_f32 = arith.constant(0.0, type=f32)
        c_quarter_f32 = arith.constant(0.25, type=f32)   # 1 / F4E2M1_MAX_POW2 (= 1/4)
        scale_N_i32 = arith.constant(scale_N, type=i32)
        group_size_i32 = arith.constant(FP4_GROUP_SIZE, type=i32)
        out_bytes_per_group_i32 = arith.constant(DWORDS_PER_GROUP * 4, type=i32)
        c_blksz_i32 = arith.constant(GROUP_QUANT_BLOCK_SIZE, type=i32)

        in_rsrc = buffer_ops.create_buffer_resource(inp, max_size=True)
        out_rsrc = buffer_ops.create_buffer_resource(out, max_size=True)
        scale_rsrc = buffer_ops.create_buffer_resource(scale, max_size=True)

        tid_i32 = ArithValue(tid)
        bid_i32 = ArithValue(bid)

        # group_id = block_id * GROUP_QUANT_BLOCK_SIZE + tid
        group_id_i32 = bid_i32 * c_blksz_i32 + tid_i32
        # x = group_id // scale_N (row), y = group_id % scale_N (group-in-row)
        # NB: ArithValue's ``/`` promotes ints to floats; we must use ``//``
        # for integer floor-div on i32.
        x_i32 = group_id_i32 // scale_N_i32
        y_i32 = group_id_i32 % scale_N_i32

        # The kernel is launched with grid = ceil(total_groups / 64), so tail
        # threads in the last block may map to group_id >= total_groups; gate
        # everything on this predicate.
        total_groups_i32 = ArithValue(total_groups)
        in_range_group = arith.cmpi(
            CmpIPredicate.ult, group_id_i32, total_groups_i32
        )

        # ----- Load 32 elements as 4 chunks of vec<8,bf16> (4 dwords each) -----
        # Input element offset = group_id * group_size.
        elem_off_i32 = group_id_i32 * group_size_i32
        # Clamp OOR groups to row 0 so the load always touches valid memory.
        safe_elem_off = arith.select(in_range_group, elem_off_i32, c0_i32)
        in_dw_off_base = safe_elem_off >> c1_i32  # 2 bf16/fp16 per dword

        chunks = []
        for ci in range_constexpr(LOADS_PER_GROUP):
            dw_off_c = in_dw_off_base + arith.constant(
                ci * DWORDS_PER_LOAD, type=i32
            )
            raw_i32_c = buffer_ops.buffer_load(
                in_rsrc, dw_off_c, vec_width=DWORDS_PER_LOAD, dtype=i32
            )
            chunks.append(vector.bitcast(in_chunk_vec_ty, raw_i32_c))

        # ----- Compute absMax across all 32 elements -----
        abs_max = c0_f32
        for ci in range_constexpr(LOADS_PER_GROUP):
            x_chunk_f32 = chunks[ci].extf(T.vec(ELEMS_PER_LOAD, f32))
            for vi in range_constexpr(ELEMS_PER_LOAD):
                v = vector.extract(
                    x_chunk_f32, static_position=[vi], dynamic_position=[]
                )
                abs_v = _llvm.call_intrinsic(
                    f32, "llvm.fabs.f32", [v], [], []
                )
                abs_max = arith.maximumf(abs_max, abs_v)

        # ----- fp4_scale: round absMax UP to next power of 2 (RNE) -----
        # Bit layout of f32: [sign:1][exp:8][mantissa:23]
        u_amax = abs_max.bitcast(i32)
        c_exp_mask = arith.constant(0xFF, type=i32)
        c_23 = arith.constant(23, type=i32)
        c_22 = arith.constant(22, type=i32)
        c_21 = arith.constant(21, type=i32)
        c_lo21_mask = arith.constant(0x1FFFFF, type=i32)
        c_inf_exp = arith.constant(0xFF, type=i32)
        c_1_i32 = c1_i32

        exp = (u_amax >> c_23) & c_exp_mask
        bit22 = (u_amax >> c_22) & c_1_i32
        bit21 = (u_amax >> c_21) & c_1_i32
        lo21 = u_amax & c_lo21_mask

        # round_up = bit22 != 0 AND (bit21 != 0 OR lo21 != 0 OR exp != 0)
        bit22_set = arith.cmpi(CmpIPredicate.ne, bit22, c0_i32)
        bit21_set = arith.cmpi(CmpIPredicate.ne, bit21, c0_i32)
        lo21_set = arith.cmpi(CmpIPredicate.ne, lo21, c0_i32)
        exp_nz = arith.cmpi(CmpIPredicate.ne, exp, c0_i32)
        # OR via arith.ori on i1
        any_low = arith.ori(arith.ori(bit21_set, lo21_set), exp_nz)
        round_up = arith.andi(bit22_set, any_low)
        exp_rounded = exp + arith.select(round_up, c_1_i32, c0_i32)

        # NaN/Inf passthrough: if exp == 0xFF, keep it (don't double-round).
        is_inf_nan = arith.cmpi(CmpIPredicate.eq, exp, c_inf_exp)
        exp_final = arith.select(is_inf_nan, c_inf_exp, exp_rounded)

        next_pow2_i32 = exp_final << c_23
        next_pow2_f32 = next_pow2_i32.bitcast(f32)
        # inv_scale = next_pow2(absMax) * 0.25 (= scale of the group).
        inv_scale = next_pow2_f32 * c_quarter_f32

        # ----- Write the E8M0 scale byte -----
        # E8M0 byte = bits 30..23 of inv_scale (same as `exp_final - 2` for
        # finite values, but bit-extract is simpler and matches CUDA exactly).
        inv_scale_u32 = inv_scale.bitcast(i32)
        e8m0_byte_i32 = (inv_scale_u32 >> c_23) & c_exp_mask
        # Cast to i8 for the byte store.
        e8m0_byte_i8 = arith.trunci(T.i8, e8m0_byte_i32)

        # Scale address = x * scale_N + y, in BYTES (1 byte per group).
        scale_off_i32 = x_i32 * scale_N_i32 + y_i32

        # ----- Convert 32 bf16/fp16 -> 4 dwords of fp4 with HW scale -----
        # Each ``cvt_scalef32_pk_fp4_{bf16,f16}`` consumes a packed-2 input
        # (single 32-bit register), produces 2 fp4 (1 byte) and merges into
        # one of 4 byte-slots of the i32 dword via ``dst_sel_index``.
        if in_dtype == "bf16":
            cvt_op = rocdl.cvt_scalef32_pk_fp4_bf16
        else:  # fp16
            cvt_op = rocdl.cvt_scalef32_pk_fp4_f16
        in_pair_vec_ty = T.vec(2, in_elem_ty)

        # NOTE: we extract pair-by-pair via two scalar extracts + vector.from_elements
        # rather than vector.bitcast, because MLIR vector dialect rejects nested
        # vector-of-vector types (only "scalar-of-vector" is legal).
        # Each input chunk has 8 elements = 4 pairs = exactly 1 output dword.
        out_dwords = []
        for dw in range_constexpr(DWORDS_PER_GROUP):
            packed = c0_i32
            chunk = chunks[dw]  # vec<8, bf16/fp16>
            for sel in range_constexpr(4):
                e0 = vector.extract(
                    chunk, static_position=[sel * 2], dynamic_position=[]
                )
                e1 = vector.extract(
                    chunk, static_position=[sel * 2 + 1], dynamic_position=[]
                )
                pair = vector.from_elements(in_pair_vec_ty, [e0, e1])
                packed = cvt_op(i32, packed, pair, inv_scale, sel)
            out_dwords.append(packed)

        out_vec = vector.from_elements(
            T.vec(DWORDS_PER_GROUP, i32), out_dwords
        )

        # ----- Write outputs only for in-range groups -----
        # Out byte offset = group_id * (group_size / 2) = group_id * 16.
        out_byte_off_i32 = group_id_i32 * out_bytes_per_group_i32

        _if_in = _scf.IfOp(in_range_group)
        with ir.InsertionPoint(_if_in.then_block):
            buffer_ops.buffer_store(
                out_vec, out_rsrc, out_byte_off_i32, offset_is_bytes=True,
            )
            buffer_ops.buffer_store(
                e8m0_byte_i8, scale_rsrc, scale_off_i32, offset_is_bytes=True,
            )
            _scf.YieldOp([])

    @flyc.jit
    def launch_per_1x32_fp4_quant(
        inp: fx.Tensor,
        out: fx.Tensor,
        scale: fx.Tensor,
        num_blocks: Int32,
        total_groups: Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        # num_blocks   = ceil(total_groups / GROUP_QUANT_BLOCK_SIZE)
        # total_groups = rows * scale_N
        # Both are computed host-side because the JIT launch grid must be
        # index-typed.
        num_blocks_idx = arith.index_cast(T.index, num_blocks)

        launcher = per_1x32_fp4_quant_kernel(inp, out, scale, total_groups)
        launcher.launch(
            grid=(num_blocks_idx, 1, 1),
            block=(GROUP_QUANT_BLOCK_SIZE, 1, 1),
            stream=stream,
        )

    return launch_per_1x32_fp4_quant


# ============================================================================
# MXFP4 (per-1x32) dynamic quantization, fused with H_32 Hadamard rotation
# ============================================================================
#
# Same overall structure as ``build_per_1x32_fp4_quant_module`` (1 thread per
# 32-elem group, 64 groups per block, no atomic / no LDS, single kernel
# launch). Differences:
#
#   * **Rotation between load and quantize**: after extf'ing the input to
#     f32, we apply the orthonormal Walsh-Hadamard transform ``H_32 /
#     sqrt(32)`` in-register via a 5-stage radix-2 FFT-style butterfly. The
#     rotated values are what get amax-reduced + quantized into fp4.
#
#   * **Stride order 1 -> 2 -> 4 -> 8 -> 16** (small-to-large) so the early
#     butterflies (intra-chunk: s=1,2,4) only depend on a single 8-element
#     load chunk. This exposes to the MLIR scheduler that compute on chunk 0
#     can issue while loads 1/2/3 are still in flight -- the "interleave
#     matmul with load" pattern (cf. SAGE attention's
#     ``_rotate_quantize_*_kernel`` in the triton path).
#
#   * **f32 conversion path**: post-rotation values are full-precision f32, so
#     we use ``v_cvt_scalef32_pk_fp4_f32`` (not the bf16/f16 packed variant).
#
# Storage layout
# --------------
# Same as the unrotated kernel: ``out`` is fp4x2 (=uint8) of shape
# ``(rows, cols // 2)``, ``scale`` is fp8_e8m0 (=uint8) of shape
# ``(rows, cols // 32)``. The stored scale is the dequant scale for the
# **rotated** values, i.e. the downstream consumer dequantizes via
# ``y_rot = y_fp4 * scale_e8m0`` and then applies the inverse rotation
# ``(H/sqrt(32)) @ y_rot`` (H is symmetric / its own inverse up to
# normalization) to recover an estimate of the original x.

# Hadamard normalization constant: 1 / sqrt(32).
_INV_SQRT_32 = 0.17677669529663687


@functools.lru_cache(maxsize=None)
def build_per_1x32_fp4_quant_hadamard_module(
    cols: int,
    in_dtype: str = "bf16",
    shuffle_scale: bool = False,
):
    """Build (and cache) an H_32-fused MXFP4 per-1x32 dynamic quant launcher.

    Parameters mirror ``build_per_1x32_fp4_quant_module``; see that function
    for buffer-layout details. The returned launcher signature is
    ``launch(inp, out, scale, num_blocks, total_groups, stream)``.
    """
    if cols <= 0 or (cols % FP4_GROUP_SIZE) != 0:
        raise ValueError(
            f"cols must be a positive multiple of FP4_GROUP_SIZE="
            f"{FP4_GROUP_SIZE}, got cols={cols}"
        )
    if in_dtype not in ("bf16", "fp16", "f16"):
        raise ValueError(f"unsupported input dtype: {in_dtype!r}")
    if shuffle_scale:
        raise NotImplementedError(
            "shuffle_scale layout not implemented yet; pass shuffle=False"
        )

    elem_bytes_in = 2  # bf16/fp16
    PAIRS_PER_GROUP = FP4_GROUP_SIZE // 2     # 16
    DWORDS_PER_GROUP = PAIRS_PER_GROUP // 4   # 4
    ELEMS_PER_LOAD = 8
    LOADS_PER_GROUP = FP4_GROUP_SIZE // ELEMS_PER_LOAD     # 4
    DWORDS_PER_LOAD = ELEMS_PER_LOAD * elem_bytes_in // 4  # 4

    scale_N = cols // FP4_GROUP_SIZE

    @flyc.kernel
    def per_1x32_fp4_quant_hadamard_kernel(
        inp: fx.Tensor,
        out: fx.Tensor,
        scale: fx.Tensor,
        total_groups: Int32,
    ):
        bid = fx.block_idx.x
        tid = fx.thread_idx.x

        f32 = T.f32
        i32 = T.i32
        in_elem_ty = _input_elem_mlir_type(in_dtype)
        in_chunk_vec_ty = T.vec(ELEMS_PER_LOAD, in_elem_ty)

        c0_i32 = arith.constant(0, type=i32)
        c1_i32 = arith.constant(1, type=i32)
        c0_f32 = arith.constant(0.0, type=f32)
        c_quarter_f32 = arith.constant(0.25, type=f32)
        c_inv_sqrt32 = arith.constant(_INV_SQRT_32, type=f32)
        scale_N_i32 = arith.constant(scale_N, type=i32)
        group_size_i32 = arith.constant(FP4_GROUP_SIZE, type=i32)
        out_bytes_per_group_i32 = arith.constant(DWORDS_PER_GROUP * 4, type=i32)
        c_blksz_i32 = arith.constant(GROUP_QUANT_BLOCK_SIZE, type=i32)

        in_rsrc = buffer_ops.create_buffer_resource(inp, max_size=True)
        out_rsrc = buffer_ops.create_buffer_resource(out, max_size=True)
        scale_rsrc = buffer_ops.create_buffer_resource(scale, max_size=True)

        tid_i32 = ArithValue(tid)
        bid_i32 = ArithValue(bid)

        group_id_i32 = bid_i32 * c_blksz_i32 + tid_i32
        x_i32 = group_id_i32 // scale_N_i32
        y_i32 = group_id_i32 % scale_N_i32

        total_groups_i32 = ArithValue(total_groups)
        in_range_group = arith.cmpi(
            CmpIPredicate.ult, group_id_i32, total_groups_i32
        )

        # ----- Issue all 4 loads up-front so the compiler can pipeline ------
        # No data dependency between the 4 buffer_load ops, so the AMD ISel
        # scheduler will batch them as 4 back-to-back vmem instructions.
        elem_off_i32 = group_id_i32 * group_size_i32
        safe_elem_off = arith.select(in_range_group, elem_off_i32, c0_i32)
        in_dw_off_base = safe_elem_off >> c1_i32  # 2 elems/dword

        chunks = []
        for ci in range_constexpr(LOADS_PER_GROUP):
            dw_off_c = in_dw_off_base + arith.constant(
                ci * DWORDS_PER_LOAD, type=i32
            )
            raw_i32_c = buffer_ops.buffer_load(
                in_rsrc, dw_off_c, vec_width=DWORDS_PER_LOAD, dtype=i32
            )
            chunks.append(vector.bitcast(in_chunk_vec_ty, raw_i32_c))

        # ----- Extend each chunk to f32 -> 32 scalar SSA values --------------
        # We deliberately extf chunk-by-chunk so the SSA dependency makes it
        # clear to the scheduler that subsequent butterflies on chunk c only
        # need chunk c (not chunks > c). This enables compute-load overlap.
        x_vals = [None] * FP4_GROUP_SIZE
        for ci in range_constexpr(LOADS_PER_GROUP):
            chunk_f32 = chunks[ci].extf(T.vec(ELEMS_PER_LOAD, f32))
            for vi in range_constexpr(ELEMS_PER_LOAD):
                x_vals[ci * ELEMS_PER_LOAD + vi] = vector.extract(
                    chunk_f32, static_position=[vi], dynamic_position=[]
                )

        # ----- H_32 Walsh-Hadamard butterfly (radix-2, 5 stages) -------------
        # Stride order 1 -> 2 -> 4 -> 8 -> 16 (small-to-large): the first
        # three stages only mix elements within one 8-element chunk, so
        # butterflies on chunk 0 can issue while chunks 1/2/3 are still being
        # loaded. The MLIR scheduler picks this up automatically from the SSA
        # def-use graph.
        #
        # NB: inside ``@flyc.kernel`` bodies, the bare ``range()`` builtin
        # creates an SCF loop with ArithValue iv; we want pure Python-level
        # unrolling here so the SSA def-use chain is fully visible to the
        # scheduler. Use ``range_constexpr`` for that.
        for s in (1, 2, 4, 8, 16):
            for i in range_constexpr(FP4_GROUP_SIZE):
                if (i & s) == 0:
                    j = i + s
                    a = x_vals[i]
                    b = x_vals[j]
                    x_vals[i] = a + b
                    x_vals[j] = a - b

        # ----- Normalize: y = H_32 @ x / sqrt(32) ----------------------------
        # 32 fmuls. Cheap relative to the load latency; folding sqrt(32) into
        # the cvt scale factor would save a few cycles but complicates the
        # E8M0 bit-trick (next_pow2 isn't homogeneous under non-pow2 scaling).
        for i in range_constexpr(FP4_GROUP_SIZE):
            x_vals[i] = x_vals[i] * c_inv_sqrt32

        # ----- absMax of the rotated, normalized values ----------------------
        abs_max = c0_f32
        for i in range_constexpr(FP4_GROUP_SIZE):
            abs_v = _llvm.call_intrinsic(
                f32, "llvm.fabs.f32", [x_vals[i]], [], []
            )
            abs_max = arith.maximumf(abs_max, abs_v)

        # ----- fp4_scale: round absMax UP to next power of 2 (RNE) -----------
        u_amax = abs_max.bitcast(i32)
        c_exp_mask = arith.constant(0xFF, type=i32)
        c_23 = arith.constant(23, type=i32)
        c_22 = arith.constant(22, type=i32)
        c_21 = arith.constant(21, type=i32)
        c_lo21_mask = arith.constant(0x1FFFFF, type=i32)
        c_inf_exp = arith.constant(0xFF, type=i32)

        exp = (u_amax >> c_23) & c_exp_mask
        bit22 = (u_amax >> c_22) & c1_i32
        bit21 = (u_amax >> c_21) & c1_i32
        lo21 = u_amax & c_lo21_mask

        bit22_set = arith.cmpi(CmpIPredicate.ne, bit22, c0_i32)
        bit21_set = arith.cmpi(CmpIPredicate.ne, bit21, c0_i32)
        lo21_set = arith.cmpi(CmpIPredicate.ne, lo21, c0_i32)
        exp_nz = arith.cmpi(CmpIPredicate.ne, exp, c0_i32)
        any_low = arith.ori(arith.ori(bit21_set, lo21_set), exp_nz)
        round_up = arith.andi(bit22_set, any_low)
        exp_rounded = exp + arith.select(round_up, c1_i32, c0_i32)

        is_inf_nan = arith.cmpi(CmpIPredicate.eq, exp, c_inf_exp)
        exp_final = arith.select(is_inf_nan, c_inf_exp, exp_rounded)

        next_pow2_i32 = exp_final << c_23
        next_pow2_f32 = next_pow2_i32.bitcast(f32)
        inv_scale = next_pow2_f32 * c_quarter_f32  # = scale of the (rotated) group

        # ----- Write the E8M0 scale byte -------------------------------------
        inv_scale_u32 = inv_scale.bitcast(i32)
        e8m0_byte_i32 = (inv_scale_u32 >> c_23) & c_exp_mask
        e8m0_byte_i8 = arith.trunci(T.i8, e8m0_byte_i32)
        scale_off_i32 = x_i32 * scale_N_i32 + y_i32

        # ----- Pack 32 rotated f32 values into 4 fp4 dwords ------------------
        # Use the f32 cvt because the rotated values aren't representable in
        # bf16/fp16 without an extra rounding step.
        out_dwords = []
        for dw in range_constexpr(DWORDS_PER_GROUP):
            packed = c0_i32
            for sel in range_constexpr(4):
                idx = dw * 4 + sel
                e0 = x_vals[idx * 2]
                e1 = x_vals[idx * 2 + 1]
                # rocdl.cvt_scalef32_pk_fp4_f32 signature:
                #   (res_type, old_vdst, src0, src1, scale, dst_sel_index)
                packed = rocdl.cvt_scalef32_pk_fp4_f32(
                    i32, packed, e0, e1, inv_scale, sel
                )
            out_dwords.append(packed)

        out_vec = vector.from_elements(
            T.vec(DWORDS_PER_GROUP, i32), out_dwords
        )

        out_byte_off_i32 = group_id_i32 * out_bytes_per_group_i32

        _if_in = _scf.IfOp(in_range_group)
        with ir.InsertionPoint(_if_in.then_block):
            buffer_ops.buffer_store(
                out_vec, out_rsrc, out_byte_off_i32, offset_is_bytes=True,
            )
            buffer_ops.buffer_store(
                e8m0_byte_i8, scale_rsrc, scale_off_i32, offset_is_bytes=True,
            )
            _scf.YieldOp([])

    @flyc.jit
    def launch_per_1x32_fp4_quant_hadamard(
        inp: fx.Tensor,
        out: fx.Tensor,
        scale: fx.Tensor,
        num_blocks: Int32,
        total_groups: Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        num_blocks_idx = arith.index_cast(T.index, num_blocks)

        launcher = per_1x32_fp4_quant_hadamard_kernel(
            inp, out, scale, total_groups
        )
        launcher.launch(
            grid=(num_blocks_idx, 1, 1),
            block=(GROUP_QUANT_BLOCK_SIZE, 1, 1),
            stream=stream,
        )

    return launch_per_1x32_fp4_quant_hadamard


# ============================================================================
# MXFP4 (per-1x32) dynamic quantization, fused with a per-block rotation R
# ============================================================================
#
# This is the **general** per-block-rotation variant: the rotation matrix R is
# a runtime tensor of shape ``(scale_N, 32, 32)`` (i.e. one independent (32,
# 32) matrix per fp4 group along the last dim), matching the host-side
# reference:
#
#   def apply_block_rotation(x, R):
#       *lead, N = x.shape
#       B, g, _ = R.shape                    # g == 32, B*g == N
#       xb = x.reshape(*lead, B, g)
#       yb = torch.einsum("...bg,bhg->...bh", xb, R)  # y[h] = sum_g R[h,g] * x[g]
#       return yb.reshape(*lead, N)
#
# So this kernel computes ``y = (R[b] @ x[m, b*32 : b*32+32])`` per group and
# quantizes ``y`` into MXFP4 with an E8M0 scale.
#
# Key perf trick -- LDS reuse of R[b]
# ------------------------------------
# A naive 1-thread-per-group launch would re-read ``R[b]`` (4 KB) from VRAM
# once *per row* -- terrible. We instead launch a 2-D grid where one block
# fixes ``b`` and varies ``m`` over a 64-thread wave, so all 64 threads in the
# block share the same ``R[b]``. We cooperatively load ``R[b]`` into LDS once
# at kernel entry (1024 f32 = 4 KB), barrier, then each thread does its own
# (32, 32)x32 matrix-vector multiply against the LDS copy. With wavefront
# size 64 and uniform LDS read addresses across the wave, the LDS reads are
# coalesced into single-cycle broadcasts.
#
# Grid layout
# -----------
#   grid  = (ceil(rows / 64), scale_N, 1)
#   block = (64, 1, 1)
#   tid.x = local-m within the row chunk
#   bid.x = row-chunk index
#   bid.y = group index ``b`` in ``[0, scale_N)``
#
# Storage layout
# --------------
# Same as the other MXFP4 kernels: ``out`` is fp4x2 (=uint8) of shape
# ``(rows, cols // 2)``, ``scale`` is fp8_e8m0 (=uint8) of shape
# ``(rows, cols // 32)``. The stored scale is the dequant scale of the
# **rotated** values; consumer applies the inverse rotation
# ``R[b]^T`` (assumed orthogonal) on top of the dequant.


@functools.lru_cache(maxsize=None)
def build_per_1x32_fp4_quant_block_rotation_module(
    cols: int,
    in_dtype: str = "bf16",
    rot_dtype: str = "bf16",
    shuffle_scale: bool = False,
):
    """Build (and cache) the per-block-rotated MXFP4 per-1x32 quant launcher.

    Parameters
    ----------
    cols
        Last-dim size of the input tensor; must be a multiple of 32.
    in_dtype
        Input element dtype, ``"bf16"`` or ``"fp16"``.
    rot_dtype
        R element dtype, ``"bf16"`` / ``"fp16"`` / ``"f32"``.
    shuffle_scale
        Reserved for future shuffle layout; must be ``False``.

    Returns
    -------
    A ``@flyc.jit`` launcher with signature
    ``launch(inp, R, out, scale, num_m_blocks, scale_n, rows, stream)``.
    """
    if cols <= 0 or (cols % FP4_GROUP_SIZE) != 0:
        raise ValueError(
            f"cols must be a positive multiple of FP4_GROUP_SIZE="
            f"{FP4_GROUP_SIZE}, got cols={cols}"
        )
    if in_dtype not in ("bf16", "fp16", "f16"):
        raise ValueError(f"unsupported input dtype: {in_dtype!r}")
    if rot_dtype not in ("bf16", "fp16", "f16", "f32", "fp32"):
        raise ValueError(f"unsupported R dtype: {rot_dtype!r}")
    if shuffle_scale:
        raise NotImplementedError(
            "shuffle_scale layout not implemented yet; pass shuffle=False"
        )

    elem_bytes_in = 2  # bf16/fp16
    rot_is_f32 = rot_dtype in ("f32", "fp32")
    rot_elem_bytes = 4 if rot_is_f32 else 2

    PAIRS_PER_GROUP = FP4_GROUP_SIZE // 2     # 16
    DWORDS_PER_GROUP = PAIRS_PER_GROUP // 4   # 4
    ELEMS_PER_LOAD = 8
    LOADS_PER_GROUP = FP4_GROUP_SIZE // ELEMS_PER_LOAD     # 4
    DWORDS_PER_LOAD = ELEMS_PER_LOAD * elem_bytes_in // 4  # 4

    scale_N = cols // FP4_GROUP_SIZE
    g = FP4_GROUP_SIZE                       # 32
    R_NUMEL_PER_BLOCK = g * g                # 1024 f32 in LDS
    R_LDS_BYTES = R_NUMEL_PER_BLOCK * 4      # 4 KB per block (R always cached as f32 in LDS)

    # Cooperative load layout: 64 threads, 1024 R elements -> 16 per thread.
    R_ELEMS_PER_THREAD = R_NUMEL_PER_BLOCK // GROUP_QUANT_BLOCK_SIZE   # 16
    # We do 4 vec4 (= 4 f32 elements) LDS stores per thread.
    LDS_VEC = 4
    R_LDS_STORES_PER_THREAD = R_ELEMS_PER_THREAD // LDS_VEC            # 4

    # ----- LDS allocator: one R[b] copy (4 KB) per block ----------------------
    gpu_arch = get_hip_arch()
    sym_tag = f"fp4_rot_lds_{cols}_{in_dtype}_{rot_dtype}"
    allocator = SmemAllocator(None, arch=gpu_arch, global_sym_name=sym_tag)
    R_lds_offset = allocator._align(allocator.ptr, 16)
    allocator.ptr = R_lds_offset + R_LDS_BYTES

    @flyc.kernel
    def per_1x32_fp4_quant_block_rotation_kernel(
        inp: fx.Tensor,    # (rows, cols)            bf16 / fp16
        rot_R: fx.Tensor,  # (scale_N, 32, 32)       bf16 / fp16 / f32
        out: fx.Tensor,    # (rows, cols // 2)       uint8 (fp4x2)
        scale: fx.Tensor,  # (rows, scale_N)         uint8 (e8m0)
        rows_dyn: Int32,
    ):
        bid_x = fx.block_idx.x  # row-chunk index
        bid_y = fx.block_idx.y  # b in [0, scale_N)
        tid = fx.thread_idx.x

        f32 = T.f32
        i32 = T.i32
        in_elem_ty = _input_elem_mlir_type(in_dtype)
        in_chunk_vec_ty = T.vec(ELEMS_PER_LOAD, in_elem_ty)

        c0_i32 = arith.constant(0, type=i32)
        c1_i32 = arith.constant(1, type=i32)
        c0_f32 = arith.constant(0.0, type=f32)
        c_quarter_f32 = arith.constant(0.25, type=f32)
        c_blksz_i32 = arith.constant(GROUP_QUANT_BLOCK_SIZE, type=i32)
        scale_N_i32 = arith.constant(scale_N, type=i32)
        group_size_i32 = arith.constant(FP4_GROUP_SIZE, type=i32)
        out_bytes_per_group_i32 = arith.constant(DWORDS_PER_GROUP * 4, type=i32)
        c_R_block_elems_i32 = arith.constant(R_NUMEL_PER_BLOCK, type=i32)

        in_rsrc = buffer_ops.create_buffer_resource(inp, max_size=True)
        out_rsrc = buffer_ops.create_buffer_resource(out, max_size=True)
        scale_rsrc = buffer_ops.create_buffer_resource(scale, max_size=True)
        rot_rsrc = buffer_ops.create_buffer_resource(rot_R, max_size=True)

        tid_i32 = ArithValue(tid)
        bid_x_i32 = ArithValue(bid_x)
        bid_y_i32 = ArithValue(bid_y)

        # m = bid_x * 64 + tid; b = bid_y. Each (m, b) is one group.
        m_i32 = bid_x_i32 * c_blksz_i32 + tid_i32
        b_i32 = bid_y_i32
        rows_i32 = ArithValue(rows_dyn)
        in_range_m = arith.cmpi(CmpIPredicate.ult, m_i32, rows_i32)

        # ============================================================
        # Stage 1: cooperative LDS load of R[b] (1024 f32 = 4 KB)
        # ============================================================
        # NB: materialize the LDS view at kernel entry so it dominates every
        # subsequent use (mirrors the pattern used in
        # build_dynamic_per_tensor_quant_module).
        base_ptr = allocator.get_base()
        R_lds_ptr = SmemPtr(
            base_ptr, R_lds_offset, f32, shape=(R_NUMEL_PER_BLOCK,)
        )
        R_lds_view = R_lds_ptr.get()

        # Linear distribution: thread tid handles R[b, ?, ?] elements
        # [tid*16 : tid*16 + 16] of the flat 1024-elem layout.
        # In element units. R element offset within R[b] for this thread:
        #   thread_elem_off = tid * R_ELEMS_PER_THREAD = tid * 16
        # Absolute element offset in R global buffer:
        #   R_b_elem_base = b * R_NUMEL_PER_BLOCK = b * 1024
        #   R_thread_elem_base = R_b_elem_base + thread_elem_off
        R_b_elem_base = b_i32 * c_R_block_elems_i32
        thread_elem_off = tid_i32 * arith.constant(
            R_ELEMS_PER_THREAD, type=i32
        )
        R_thread_elem_base = R_b_elem_base + thread_elem_off

        if rot_is_f32:
            # Load LDS_VEC f32 directly per inner step; LDS_VEC f32 = 16 bytes
            # = 4 dwords. We do R_LDS_STORES_PER_THREAD such vec4 ops.
            for li in range_constexpr(R_LDS_STORES_PER_THREAD):
                step_elem_off = R_thread_elem_base + arith.constant(
                    li * LDS_VEC, type=i32
                )
                # f32 = 4 bytes = 1 dword per element.
                dw_off_li = step_elem_off  # 1 elem per dword
                r_vec_f32 = buffer_ops.buffer_load(
                    rot_rsrc, dw_off_li, vec_width=LDS_VEC, dtype=f32
                )
                lds_idx = thread_elem_off + arith.constant(
                    li * LDS_VEC, type=i32
                )
                lds_idx_index = arith.index_cast(T.index, lds_idx)
                vector.store(
                    r_vec_f32, R_lds_view, [lds_idx_index], alignment=16
                )
        else:
            # bf16 / fp16 R: load 2 dwords = 4 bf16, extf to vec4 f32, store
            # 4 f32 to LDS. R_LDS_STORES_PER_THREAD such steps per thread.
            rot_elem_ty = _input_elem_mlir_type(
                "bf16" if rot_dtype == "bf16" else "fp16"
            )
            for li in range_constexpr(R_LDS_STORES_PER_THREAD):
                step_elem_off = R_thread_elem_base + arith.constant(
                    li * LDS_VEC, type=i32
                )
                # bf16/fp16 = 2 bytes per element, 2 elements per dword.
                dw_off_li = step_elem_off >> c1_i32
                raw_i32_2 = buffer_ops.buffer_load(
                    rot_rsrc, dw_off_li, vec_width=2, dtype=i32
                )
                r_vec_bf = vector.bitcast(
                    T.vec(LDS_VEC, rot_elem_ty), raw_i32_2
                )
                r_vec_f32 = r_vec_bf.extf(T.vec(LDS_VEC, f32))
                lds_idx = thread_elem_off + arith.constant(
                    li * LDS_VEC, type=i32
                )
                lds_idx_index = arith.index_cast(T.index, lds_idx)
                vector.store(
                    r_vec_f32, R_lds_view, [lds_idx_index], alignment=16
                )

        # Workgroup-wide barrier: all R[b] must be visible before matmul.
        gpu.barrier()

        # ============================================================
        # Stage 2: load 32 input elements for this (m, b)
        # ============================================================
        # Element offset into ``inp`` for this group:
        #   elem_off = m * cols + b * 32
        # We clamp m to 0 for OOR threads so the buffer_load always lands in
        # valid memory; the final stores are gated on ``in_range_m``.
        safe_m_i32 = arith.select(in_range_m, m_i32, c0_i32)
        cols_i32 = arith.constant(cols, type=i32)
        elem_off_i32 = (
            safe_m_i32 * cols_i32 + b_i32 * group_size_i32
        )
        in_dw_off_base = elem_off_i32 >> c1_i32  # 2 elems / dword

        chunks = []
        for ci in range_constexpr(LOADS_PER_GROUP):
            dw_off_c = in_dw_off_base + arith.constant(
                ci * DWORDS_PER_LOAD, type=i32
            )
            raw_i32_c = buffer_ops.buffer_load(
                in_rsrc, dw_off_c, vec_width=DWORDS_PER_LOAD, dtype=i32
            )
            chunks.append(vector.bitcast(in_chunk_vec_ty, raw_i32_c))

        # Extend each chunk to f32 and unpack to 32 scalar SSA values.
        x_vals = [None] * FP4_GROUP_SIZE
        for ci in range_constexpr(LOADS_PER_GROUP):
            chunk_f32 = chunks[ci].extf(T.vec(ELEMS_PER_LOAD, f32))
            for vi in range_constexpr(ELEMS_PER_LOAD):
                x_vals[ci * ELEMS_PER_LOAD + vi] = vector.extract(
                    chunk_f32, static_position=[vi], dynamic_position=[]
                )

        # ============================================================
        # Stage 3: 32x32 matrix-vector mul against R_lds (in LDS)
        # ============================================================
        # y[i] = sum_{j=0..31} R_lds[i*32 + j] * x_vals[j]
        # Read R 4 elements at a time via vector.load_op for fewer LDS
        # instructions; the 64 threads in a wave all hit the same LDS
        # address (=> LDS broadcast, single cycle per vec4 load).
        y_vals = [None] * FP4_GROUP_SIZE
        for i in range_constexpr(FP4_GROUP_SIZE):
            acc = c0_f32
            for j_grp in range_constexpr(FP4_GROUP_SIZE // LDS_VEC):
                base_idx = arith.constant(
                    i * FP4_GROUP_SIZE + j_grp * LDS_VEC, type=i32
                )
                base_idx_index = arith.index_cast(T.index, base_idx)
                r4 = vector.load_op(
                    T.vec(LDS_VEC, f32), R_lds_view, [base_idx_index]
                )
                for jj in range_constexpr(LDS_VEC):
                    r_val = vector.extract(
                        r4, static_position=[jj], dynamic_position=[]
                    )
                    x_val = x_vals[j_grp * LDS_VEC + jj]
                    acc = _llvm.call_intrinsic(
                        f32, "llvm.fma.f32", [r_val, x_val, acc], [], []
                    )
            y_vals[i] = acc

        # ============================================================
        # Stage 4: amax + E8M0 scale (same RNE bit-trick as other variants)
        # ============================================================
        abs_max = c0_f32
        for i in range_constexpr(FP4_GROUP_SIZE):
            abs_v = _llvm.call_intrinsic(
                f32, "llvm.fabs.f32", [y_vals[i]], [], []
            )
            abs_max = arith.maximumf(abs_max, abs_v)

        u_amax = abs_max.bitcast(i32)
        c_exp_mask = arith.constant(0xFF, type=i32)
        c_23 = arith.constant(23, type=i32)
        c_22 = arith.constant(22, type=i32)
        c_21 = arith.constant(21, type=i32)
        c_lo21_mask = arith.constant(0x1FFFFF, type=i32)
        c_inf_exp = arith.constant(0xFF, type=i32)

        exp = (u_amax >> c_23) & c_exp_mask
        bit22 = (u_amax >> c_22) & c1_i32
        bit21 = (u_amax >> c_21) & c1_i32
        lo21 = u_amax & c_lo21_mask

        bit22_set = arith.cmpi(CmpIPredicate.ne, bit22, c0_i32)
        bit21_set = arith.cmpi(CmpIPredicate.ne, bit21, c0_i32)
        lo21_set = arith.cmpi(CmpIPredicate.ne, lo21, c0_i32)
        exp_nz = arith.cmpi(CmpIPredicate.ne, exp, c0_i32)
        any_low = arith.ori(arith.ori(bit21_set, lo21_set), exp_nz)
        round_up = arith.andi(bit22_set, any_low)
        exp_rounded = exp + arith.select(round_up, c1_i32, c0_i32)

        is_inf_nan = arith.cmpi(CmpIPredicate.eq, exp, c_inf_exp)
        exp_final = arith.select(is_inf_nan, c_inf_exp, exp_rounded)

        next_pow2_i32 = exp_final << c_23
        next_pow2_f32 = next_pow2_i32.bitcast(f32)
        inv_scale = next_pow2_f32 * c_quarter_f32

        inv_scale_u32 = inv_scale.bitcast(i32)
        e8m0_byte_i32 = (inv_scale_u32 >> c_23) & c_exp_mask
        e8m0_byte_i8 = arith.trunci(T.i8, e8m0_byte_i32)
        # scale address = m * scale_N + b (1 byte per group)
        scale_off_i32 = m_i32 * scale_N_i32 + b_i32

        # ============================================================
        # Stage 5: pack 32 f32 rotated values into 4 fp4 dwords
        # ============================================================
        out_dwords = []
        for dw in range_constexpr(DWORDS_PER_GROUP):
            packed = c0_i32
            for sel in range_constexpr(4):
                idx = dw * 4 + sel
                e0 = y_vals[idx * 2]
                e1 = y_vals[idx * 2 + 1]
                packed = rocdl.cvt_scalef32_pk_fp4_f32(
                    i32, packed, e0, e1, inv_scale, sel
                )
            out_dwords.append(packed)

        out_vec = vector.from_elements(
            T.vec(DWORDS_PER_GROUP, i32), out_dwords
        )

        out_byte_off_i32 = (
            m_i32 * arith.constant(scale_N * DWORDS_PER_GROUP * 4, type=i32)
            + b_i32 * out_bytes_per_group_i32
        )

        _if_in = _scf.IfOp(in_range_m)
        with ir.InsertionPoint(_if_in.then_block):
            buffer_ops.buffer_store(
                out_vec, out_rsrc, out_byte_off_i32, offset_is_bytes=True,
            )
            buffer_ops.buffer_store(
                e8m0_byte_i8, scale_rsrc, scale_off_i32, offset_is_bytes=True,
            )
            _scf.YieldOp([])

    @flyc.jit
    def launch_per_1x32_fp4_quant_block_rotation(
        inp: fx.Tensor,
        rot_R: fx.Tensor,
        out: fx.Tensor,
        scale: fx.Tensor,
        num_m_blocks: Int32,
        rows: Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        # LDS finalize: emit the gpu.module global memref for R[b].
        # Must run on every JIT entry; the allocator is module-scoped.
        allocator.finalized = False
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            allocator.finalize()

        # grid = (num_m_blocks, scale_N, 1); scale_N is compile-time constant.
        num_m_blocks_idx = arith.index_cast(T.index, num_m_blocks)
        scale_N_idx = arith.constant(scale_N, type=T.index)

        launcher = per_1x32_fp4_quant_block_rotation_kernel(
            inp, rot_R, out, scale, rows
        )
        launcher.launch(
            grid=(num_m_blocks_idx, scale_N_idx, 1),
            block=(GROUP_QUANT_BLOCK_SIZE, 1, 1),
            stream=stream,
        )

    return launch_per_1x32_fp4_quant_block_rotation


# ============================================================================
# MFMA-accelerated variant of build_per_1x32_fp4_quant_block_rotation_module.
# ============================================================================
# Mathematical model is identical: ``y[m, b*32+h] = sum_{g} x[m, b*32+g] *
# R[b, h, g]``, followed by per-(m, b) MXFP4 quant of ``y[m, b*32:b*32+32]``.
#
# What changes vs the scalar variant
# ----------------------------------
# * 32x32 mat-vec replaced by 16 invocations of ``v_mfma_f32_16x16x16_{bf16,f16}_1k``
#   (4 M-tiles x 2 N-tiles x 2 K-loop iters, each 16-cycle on gfx942).
# * R[b] cached in LDS as bf16/fp16 (2 KB) instead of extf'd-to-f32 (4 KB).
# * Post-MFMA C tiles transposed via 8 KB LDS so each thread owns a full
#   32-element row (m == bid_x*64 + tid), then the existing amax + E8M0 +
#   ``cvt_scalef32_pk_fp4_f32`` path applies unchanged.
#
# Constraints
# -----------
# * ``in_dtype == rot_dtype`` (MFMA op requires both A/B same fp type).
# * ``rot_dtype`` in {"bf16", "fp16", "f16"} -- no f32 R (no f32 MFMA op
#   on gfx942 for this tile size). Fall back to the scalar variant for f32 R.
# * Block size still 64 threads = 1 wave (MFMA is wave-level).


@functools.lru_cache(maxsize=None)
def build_per_1x32_fp4_quant_block_rotation_mfma_module(
    cols: int,
    in_dtype: str = "bf16",
    rot_dtype: str = "bf16",
    shuffle_scale: bool = False,
    rot_transposed: bool = False,
):
    """MFMA-accelerated per-block-rotated MXFP4 per-1x32 quant launcher.

    Returns a ``@flyc.jit`` launcher with the same signature as
    :func:`build_per_1x32_fp4_quant_block_rotation_module`:

        ``launch(inp, R, out, scale, num_m_blocks, rows, stream)``

    so it is a drop-in replacement when ``in_dtype == rot_dtype`` and
    ``rot_dtype != "f32"``.

    Parameters
    ----------
    rot_transposed : bool, default False
        Selects how the caller-supplied ``rot_R`` tensor of shape
        ``(scale_N, 32, 32)`` is interpreted along its last two dims:

        - ``False`` (default): ``rot_R[b, h, g]`` is the rotation matrix
          ``R`` for block ``b``. The kernel computes
          ``y[m, b*32 + h] = sum_g x[m, b*32 + g] * R[b, h, g]``
          i.e. ``Y = einsum("...bg, bhg -> ...bh", X, rot_R)``
          (equivalent to per-block ``Y[m] = X[m] @ R.T``).

        - ``True``: ``rot_R[b, g, h]`` is the rotation matrix ``R``
          stored transposed along its last two dims (i.e. the caller
          already laid it out as ``R.transpose(-1, -2)``). The kernel
          computes
          ``y[m, b*32 + h] = sum_g x[m, b*32 + g] * R[b, g, h]``
          i.e. ``Y = einsum("...bg, bgh -> ...bh", X, rot_R)``
          (equivalent to per-block ``Y[m] = X[m] @ R``).

        The two modes produce mathematically equivalent results when fed
        with corresponding (transposed vs. non-transposed) ``rot_R``
        tensors. The flag is compile-time and selects a different
        compiled kernel via ``lru_cache``; there is no runtime branch.
        Implementation-wise it only swaps the LDS index formula used for
        the MFMA B-fragment load (one constexpr index swap), so both
        variants have identical ISA cost.
    """
    if cols <= 0 or (cols % FP4_GROUP_SIZE) != 0:
        raise ValueError(
            f"cols must be a positive multiple of FP4_GROUP_SIZE="
            f"{FP4_GROUP_SIZE}, got cols={cols}"
        )
    if in_dtype not in ("bf16", "fp16", "f16"):
        raise ValueError(f"unsupported input dtype: {in_dtype!r}")
    if rot_dtype in ("f32", "fp32"):
        raise ValueError(
            "MFMA path requires bf16/fp16 R; use the scalar "
            "build_per_1x32_fp4_quant_block_rotation_module for f32 R"
        )
    if rot_dtype not in ("bf16", "fp16", "f16"):
        raise ValueError(f"unsupported R dtype: {rot_dtype!r}")
    # Normalize fp16/f16 -> "fp16" for comparison; bf16 stays.
    _norm_in = "bf16" if in_dtype == "bf16" else "fp16"
    _norm_rot = "bf16" if rot_dtype == "bf16" else "fp16"
    if _norm_in != _norm_rot:
        raise ValueError(
            f"MFMA path requires in_dtype == rot_dtype (got "
            f"in_dtype={in_dtype!r}, rot_dtype={rot_dtype!r})"
        )
    if shuffle_scale:
        raise NotImplementedError(
            "shuffle_scale layout not implemented yet; pass shuffle=False"
        )

    PAIRS_PER_GROUP = FP4_GROUP_SIZE // 2       # 16
    DWORDS_PER_GROUP = PAIRS_PER_GROUP // 4     # 4
    scale_N = cols // FP4_GROUP_SIZE
    g = FP4_GROUP_SIZE                          # 32

    # ----- MFMA tile geometry (16x16x16, gfx942 _1k op) -----
    M_TILE = 16
    N_TILE = 16
    K_TILE = 16
    NUM_M_TILES = GROUP_QUANT_BLOCK_SIZE // M_TILE   # 4
    NUM_N_TILES = FP4_GROUP_SIZE // N_TILE           # 2
    NUM_K_TILES = FP4_GROUP_SIZE // K_TILE           # 2
    WMMA_FRAG_VALS = 4   # 4 elements/lane for A, B, C in 16x16x16

    # ----- LDS layout: R[b] (bf16/fp16) + Y tile (f32) ----------------------
    R_NUMEL_PER_BLOCK = g * g                        # 1024
    R_LDS_BYTES = R_NUMEL_PER_BLOCK * 2              # 2 KB (bf16/fp16)
    Y_LDS_BYTES = GROUP_QUANT_BLOCK_SIZE * FP4_GROUP_SIZE * 4  # 8 KB (f32)

    # Cooperative R load: 1024 / 64 = 16 elems/thread, in 2x vec4_i32 (= 8
    # bf16) cooperative loads.
    R_ELEMS_PER_THREAD = R_NUMEL_PER_BLOCK // GROUP_QUANT_BLOCK_SIZE  # 16
    R_LOAD_VEC = 8                                   # bf16 per coop load step
    R_LOAD_STEPS = R_ELEMS_PER_THREAD // R_LOAD_VEC  # 2

    # Per-thread Y read-back: 32 f32 / 4 = 8 vec4 LDS loads.
    Y_READ_VEC = 4
    Y_READ_STEPS = FP4_GROUP_SIZE // Y_READ_VEC      # 8

    # ----- LDS allocator ----------------------------------------------------
    gpu_arch = get_hip_arch()
    sym_tag = f"fp4_rot_mfma_lds_{cols}_{in_dtype}_{rot_dtype}"
    allocator = SmemAllocator(None, arch=gpu_arch, global_sym_name=sym_tag)
    R_lds_offset = allocator._align(allocator.ptr, 16)
    allocator.ptr = R_lds_offset + R_LDS_BYTES
    Y_lds_offset = allocator._align(allocator.ptr, 16)
    allocator.ptr = Y_lds_offset + Y_LDS_BYTES

    @flyc.kernel
    def per_1x32_fp4_quant_block_rotation_mfma_kernel(
        inp: fx.Tensor,    # (rows, cols)             bf16 / fp16
        rot_R: fx.Tensor,  # (scale_N, 32, 32)        bf16 / fp16
        out: fx.Tensor,    # (rows, cols // 2)        uint8 (fp4x2)
        scale: fx.Tensor,  # (rows, scale_N)          uint8 (e8m0)
        rows_dyn: Int32,
    ):
        from flydsl._mlir.dialects import memref as _memref

        bid_x = fx.block_idx.x  # row-chunk index
        bid_y = fx.block_idx.y  # b in [0, scale_N)
        tid = fx.thread_idx.x

        f32 = T.f32
        i32 = T.i32
        i16 = T.i16
        in_elem_ty = _input_elem_mlir_type(in_dtype)
        rot_elem_ty = _input_elem_mlir_type(
            "bf16" if rot_dtype == "bf16" else "fp16"
        )

        # ----- compile-time constants -----
        c0_i32 = arith.constant(0, type=i32)
        c1_i32 = arith.constant(1, type=i32)
        c2_i32 = arith.constant(2, type=i32)
        c4_i32 = arith.constant(4, type=i32)
        c15_i32 = arith.constant(15, type=i32)
        c0_f32 = arith.constant(0.0, type=f32)
        c_quarter_f32 = arith.constant(0.25, type=f32)
        c_blksz_i32 = arith.constant(GROUP_QUANT_BLOCK_SIZE, type=i32)
        scale_N_i32 = arith.constant(scale_N, type=i32)
        group_size_i32 = arith.constant(FP4_GROUP_SIZE, type=i32)
        cols_i32 = arith.constant(cols, type=i32)
        out_bytes_per_group_i32 = arith.constant(DWORDS_PER_GROUP * 4, type=i32)
        c_R_block_elems_i32 = arith.constant(R_NUMEL_PER_BLOCK, type=i32)
        c_R_elems_per_thread_i32 = arith.constant(R_ELEMS_PER_THREAD, type=i32)

        in_rsrc = buffer_ops.create_buffer_resource(inp, max_size=True)
        out_rsrc = buffer_ops.create_buffer_resource(out, max_size=True)
        scale_rsrc = buffer_ops.create_buffer_resource(scale, max_size=True)
        rot_rsrc = buffer_ops.create_buffer_resource(rot_R, max_size=True)

        tid_i32 = ArithValue(tid)
        bid_x_i32 = ArithValue(bid_x)
        bid_y_i32 = ArithValue(bid_y)

        m_base_i32 = bid_x_i32 * c_blksz_i32  # base row of this block
        b_i32 = bid_y_i32
        rows_i32 = ArithValue(rows_dyn)
        m_i32 = m_base_i32 + tid_i32          # this thread's row (for store)
        in_range_m = arith.cmpi(CmpIPredicate.ult, m_i32, rows_i32)

        # Lane decomposition for MFMA: lane t -> (t % 16, t // 16).
        # k_off_in_tile = 4 * (t // 16) (each MFMA tile lane group covers 4
        # consecutive K/M positions).
        lane_mod_16 = tid_i32 & c15_i32
        lane_div_16 = tid_i32 >> c4_i32
        k_off_in_tile = lane_div_16 << c2_i32  # 0, 4, 8, 12

        # ----- LDS views ---------------------------------------------------
        base_ptr = allocator.get_base()
        R_lds_view = SmemPtr(
            base_ptr, R_lds_offset, rot_elem_ty,
            shape=(R_NUMEL_PER_BLOCK,),
        ).get()
        Y_lds_view = SmemPtr(
            base_ptr, Y_lds_offset, f32,
            shape=(GROUP_QUANT_BLOCK_SIZE * FP4_GROUP_SIZE,),
        ).get()

        # ============================================================
        # Stage 1: cooperative LDS load of R[b] (bf16/fp16, 2 KB)
        # ============================================================
        # Thread tid handles flat elements [tid*16, tid*16+16) of rot_R[b].
        # 2 cooperative steps of vec4_i32 (= 8 bf16 each).
        #
        # Target LDS layout in both modes:
        #   R_lds[h*32 + g] == R[b, h, g]   (MFMA-natural; K=g is fast index)
        #
        # rot_transposed == False: rot_R[b, h, g] verbatim -> vec8 store at
        # thread's natural offset is contiguous (compiles to ds_write_b128).
        #
        # rot_transposed == True: rot_R[b, g, h] -> we have to transpose.
        # Each thread's 8-element vec chunk lives at fixed g_src with
        # h_src spanning 8 consecutive values, which scatters to LDS
        # destinations stride-32 apart (R_lds[h_src*32 + g_src]). We
        # decompose the vec8 into 8 scalar stores. Cost is paid ONCE per
        # workgroup before the MFMA loop; the hot path is identical.
        R_b_elem_base = b_i32 * c_R_block_elems_i32
        thread_elem_off = tid_i32 * c_R_elems_per_thread_i32
        R_thread_elem_base = R_b_elem_base + thread_elem_off

        for li in range_constexpr(R_LOAD_STEPS):
            step_elem_off = R_thread_elem_base + arith.constant(
                li * R_LOAD_VEC, type=i32
            )
            dw_off_li = step_elem_off >> c1_i32  # 2 elem / dword
            raw_i32_4 = buffer_ops.buffer_load(
                rot_rsrc, dw_off_li, vec_width=4, dtype=i32
            )
            r_vec_bf = vector.bitcast(
                T.vec(R_LOAD_VEC, rot_elem_ty), raw_i32_4
            )

            if not rot_transposed:
                # ----- contiguous vec8 store, dest = source order -----
                lds_idx = thread_elem_off + arith.constant(
                    li * R_LOAD_VEC, type=i32
                )
                lds_idx_index = arith.index_cast(T.index, lds_idx)
                vector.store(
                    r_vec_bf, R_lds_view, [lds_idx_index], alignment=16
                )
            else:
                # ----- transposed: 8 scalar stride-32 LDS stores -----
                # For this (tid, li), all 8 elements share the same
                #   g_src = (tid*16 + li*R_LOAD_VEC) // 32
                # and span 8 consecutive h_src starting at
                #   h_src_base = (tid*16 + li*R_LOAD_VEC) % 32.
                # Their LDS destinations are
                #   R_lds[(h_src_base + j) * 32 + g_src]   for j in 0..7
                # i.e. an arithmetic progression with stride 32.
                # We compute g_src/h_src_base at runtime (they depend on
                # tid) and emit a constexpr-unrolled loop of 8 scalar
                # stores. The compiler typically pairs adjacent ones into
                # ds_write2_b16 (offset0/offset1 = 0/64, within 256-byte
                # window), so the total cost is ~4 LDS instructions per
                # thread per iteration -- a small one-time penalty.
                flat_base_i32 = thread_elem_off + arith.constant(
                    li * R_LOAD_VEC, type=i32
                )
                g_src_i32 = flat_base_i32 >> arith.constant(
                    5, type=i32
                )  # // 32
                h_src_base_i32 = flat_base_i32 & arith.constant(
                    31, type=i32
                )  # % 32
                dest_base_i32 = (
                    h_src_base_i32 * group_size_i32 + g_src_i32
                )
                for j in range_constexpr(R_LOAD_VEC):
                    elem_bf = vector.extract(
                        r_vec_bf, static_position=[j], dynamic_position=[]
                    )
                    dest_off_i32 = dest_base_i32 + arith.constant(
                        j * FP4_GROUP_SIZE, type=i32
                    )
                    dest_off_idx = arith.index_cast(T.index, dest_off_i32)
                    _memref.store(elem_bf, R_lds_view, [dest_off_idx])

        gpu.barrier()

        # ============================================================
        # Stage 2: MFMA loop, 4 M-tiles x 2 N-tiles x 2 K-iters
        # ============================================================
        # Tile-level layout (16x16x16 _1k, gfx942):
        #   A[m, k]: lane t holds A[t%16, k_off..k_off+3], k_off = 4*(t/16)
        #   B[k, n]: lane t holds B[k_off..k_off+3, t%16],   k_off = 4*(t/16)
        #   C[m, n]: lane t holds C[m_off..m_off+3, t%16],   m_off = 4*(t/16)
        # i.e. all three operands share the same (t%16, t/16) decomposition
        # along their "free" and "stride" axes.
        #
        # We want y[m, h] = sum_g x[m, g] * R[b, h, g] (or
        # sum_g x[m, g] * R[b, g, h] when ``rot_transposed`` is True --
        # the user has handed us R already transposed along its inner
        # two dims). Set A := x (rows m, cols g) and target the
        # MFMA-natural LDS layout R_lds[h*32 + g] == R[b, h, g] in
        # BOTH modes -- ``rot_transposed`` is absorbed at coop-load
        # time, not in the hot path. With R_lds[h*32 + g] the B
        # fragment for lane t at (k_tile, n_tile) reads
        #   R_lds[(n_tile*16 + (t%16)) * 32 + (k_tile*16 + 4*(t/16))
        #         + 0..3]
        # i.e. ``lds_off_b = n_lds * 32 + k_lds`` -- 4 K-consecutive
        # elements per lane, which collapses to a single ds_read_b64
        # (compiler batches across k_tile via ds_read2_b64).

        for m_tile in range_constexpr(NUM_M_TILES):
            # Per-lane row index in this M-tile.
            m_row_i32 = (
                m_base_i32
                + arith.constant(m_tile * 16, type=i32)
                + lane_mod_16
            )
            row_in_range = arith.cmpi(
                CmpIPredicate.ult, m_row_i32, rows_i32
            )
            # OOR rows clamp to row 0 so the buffer_load lands in valid
            # memory; their MFMA outputs are written to LDS (garbage) but
            # never stored out (final store gated on ``in_range_m``).
            safe_m_row = arith.select(row_in_range, m_row_i32, c0_i32)
            # The 32-wide group lives at columns [b*32, b*32 + 32) of the
            # input row -- the input A frags must include the per-b col
            # offset (counterpart of the scalar kernel's ``b * 32`` term).
            row_off_in_inp = safe_m_row * cols_i32 + b_i32 * group_size_i32

            for n_tile in range_constexpr(NUM_N_TILES):
                # ----- zero-init accumulator (vec4 f32 per lane) -----
                acc = vector.from_elements(
                    T.vec(WMMA_FRAG_VALS, f32),
                    [c0_f32] * WMMA_FRAG_VALS,
                )

                for k_tile in range_constexpr(NUM_K_TILES):
                    # ----- A frag: x[m_row, b*32 + k_col : ... +4] -----
                    k_col_i32 = (
                        arith.constant(k_tile * 16, type=i32) + k_off_in_tile
                    )
                    elem_off_a = row_off_in_inp + k_col_i32
                    dw_off_a = elem_off_a >> c1_i32  # 2 elem/dword
                    # vec2 i32 = 4 bytes/lane * 2 = 8 bytes = 4 bf16
                    raw_a = buffer_ops.buffer_load(
                        in_rsrc, dw_off_a, vec_width=2, dtype=i32
                    )
                    a_frag_bf = vector.bitcast(
                        T.vec(WMMA_FRAG_VALS, in_elem_ty), raw_a
                    )

                    # ----- B frag: R_lds[n_local*32 + k_local : ... +4] -----
                    n_lds = (
                        arith.constant(n_tile * 16, type=i32) + lane_mod_16
                    )
                    k_lds = (
                        arith.constant(k_tile * 16, type=i32) + k_off_in_tile
                    )
                    lds_off_b = n_lds * group_size_i32 + k_lds
                    lds_off_b_idx = arith.index_cast(T.index, lds_off_b)
                    b_frag_bf = vector.load_op(
                        T.vec(WMMA_FRAG_VALS, rot_elem_ty),
                        R_lds_view,
                        [lds_off_b_idx],
                    )

                    # ----- MFMA -----
                    if in_dtype == "bf16":
                        a_for_mfma = vector.bitcast(
                            T.vec(WMMA_FRAG_VALS, i16), a_frag_bf
                        )
                        b_for_mfma = vector.bitcast(
                            T.vec(WMMA_FRAG_VALS, i16), b_frag_bf
                        )
                        acc = rocdl.mfma_f32_16x16x16bf16_1k(
                            T.vec(WMMA_FRAG_VALS, f32),
                            [a_for_mfma, b_for_mfma, acc, 0, 0, 0],
                        )
                    else:  # fp16 / f16
                        acc = rocdl.mfma_f32_16x16x16f16(
                            T.vec(WMMA_FRAG_VALS, f32),
                            [a_frag_bf, b_frag_bf, acc, 0, 0, 0],
                        )

                # ----- write C tile to Y_lds (row-major transpose) -----
                # Lane t in tile (m_tile, n_tile) holds 4 elements at
                # (m_local = m_tile*16 + 4*(t/16) + i, n_local = n_tile*16 + t%16)
                # for i in 0..3. Scalar stores; addresses differ by N
                # (=32) along the m axis so no cheap vector store path.
                m_local_base = (
                    arith.constant(m_tile * 16, type=i32) + k_off_in_tile
                )
                n_local_i32 = (
                    arith.constant(n_tile * 16, type=i32) + lane_mod_16
                )
                for i in range_constexpr(WMMA_FRAG_VALS):
                    m_local_i32 = m_local_base + arith.constant(i, type=i32)
                    y_lds_off = m_local_i32 * group_size_i32 + n_local_i32
                    val = vector.extract(
                        acc, static_position=[i], dynamic_position=[]
                    )
                    y_lds_off_idx = arith.index_cast(T.index, y_lds_off)
                    _memref.store(val, Y_lds_view, [y_lds_off_idx])

        gpu.barrier()

        # ============================================================
        # Stage 3: per-thread row gather + amax + E8M0 + cvt fp4
        # ============================================================
        # Thread tid owns Y_lds[tid * 32 + 0..31]. Read 8x vec4 f32.
        y_vals = [None] * FP4_GROUP_SIZE
        tid_row_base = tid_i32 * group_size_i32
        for k in range_constexpr(Y_READ_STEPS):
            lds_off = tid_row_base + arith.constant(k * Y_READ_VEC, type=i32)
            lds_off_idx = arith.index_cast(T.index, lds_off)
            v4 = vector.load_op(T.vec(Y_READ_VEC, f32), Y_lds_view, [lds_off_idx])
            for j in range_constexpr(Y_READ_VEC):
                y_vals[k * Y_READ_VEC + j] = vector.extract(
                    v4, static_position=[j], dynamic_position=[]
                )

        # ----- amax -----
        abs_max = c0_f32
        for i in range_constexpr(FP4_GROUP_SIZE):
            abs_v = _llvm.call_intrinsic(
                f32, "llvm.fabs.f32", [y_vals[i]], [], []
            )
            abs_max = arith.maximumf(abs_max, abs_v)

        # ----- E8M0 scale (same RNE bit-trick as scalar variant) -----
        u_amax = abs_max.bitcast(i32)
        c_exp_mask = arith.constant(0xFF, type=i32)
        c_23 = arith.constant(23, type=i32)
        c_22 = arith.constant(22, type=i32)
        c_21 = arith.constant(21, type=i32)
        c_lo21_mask = arith.constant(0x1FFFFF, type=i32)
        c_inf_exp = arith.constant(0xFF, type=i32)

        exp = (u_amax >> c_23) & c_exp_mask
        bit22 = (u_amax >> c_22) & c1_i32
        bit21 = (u_amax >> c_21) & c1_i32
        lo21 = u_amax & c_lo21_mask

        bit22_set = arith.cmpi(CmpIPredicate.ne, bit22, c0_i32)
        bit21_set = arith.cmpi(CmpIPredicate.ne, bit21, c0_i32)
        lo21_set = arith.cmpi(CmpIPredicate.ne, lo21, c0_i32)
        exp_nz = arith.cmpi(CmpIPredicate.ne, exp, c0_i32)
        any_low = arith.ori(arith.ori(bit21_set, lo21_set), exp_nz)
        round_up = arith.andi(bit22_set, any_low)
        exp_rounded = exp + arith.select(round_up, c1_i32, c0_i32)

        is_inf_nan = arith.cmpi(CmpIPredicate.eq, exp, c_inf_exp)
        exp_final = arith.select(is_inf_nan, c_inf_exp, exp_rounded)

        next_pow2_i32 = exp_final << c_23
        next_pow2_f32 = next_pow2_i32.bitcast(f32)
        inv_scale = next_pow2_f32 * c_quarter_f32

        inv_scale_u32 = inv_scale.bitcast(i32)
        e8m0_byte_i32 = (inv_scale_u32 >> c_23) & c_exp_mask
        e8m0_byte_i8 = arith.trunci(T.i8, e8m0_byte_i32)
        scale_off_i32 = m_i32 * scale_N_i32 + b_i32

        # ----- cvt to fp4 -----
        out_dwords = []
        for dw in range_constexpr(DWORDS_PER_GROUP):
            packed = c0_i32
            for sel in range_constexpr(4):
                idx = dw * 4 + sel
                e0 = y_vals[idx * 2]
                e1 = y_vals[idx * 2 + 1]
                packed = rocdl.cvt_scalef32_pk_fp4_f32(
                    i32, packed, e0, e1, inv_scale, sel
                )
            out_dwords.append(packed)

        out_vec = vector.from_elements(
            T.vec(DWORDS_PER_GROUP, i32), out_dwords
        )

        out_byte_off_i32 = (
            m_i32 * arith.constant(scale_N * DWORDS_PER_GROUP * 4, type=i32)
            + b_i32 * out_bytes_per_group_i32
        )

        _if_in = _scf.IfOp(in_range_m)
        with ir.InsertionPoint(_if_in.then_block):
            buffer_ops.buffer_store(
                out_vec, out_rsrc, out_byte_off_i32, offset_is_bytes=True,
            )
            buffer_ops.buffer_store(
                e8m0_byte_i8, scale_rsrc, scale_off_i32, offset_is_bytes=True,
            )
            _scf.YieldOp([])

    @flyc.jit
    def launch_per_1x32_fp4_quant_block_rotation_mfma(
        inp: fx.Tensor,
        rot_R: fx.Tensor,
        out: fx.Tensor,
        scale: fx.Tensor,
        num_m_blocks: Int32,
        rows: Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        # LDS finalize: emit the gpu.module global memref for R[b] + Y tile.
        allocator.finalized = False
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            allocator.finalize()

        # grid = (num_m_blocks, scale_N, 1); scale_N is compile-time constant.
        num_m_blocks_idx = arith.index_cast(T.index, num_m_blocks)
        scale_N_idx = arith.constant(scale_N, type=T.index)

        launcher = per_1x32_fp4_quant_block_rotation_mfma_kernel(
            inp, rot_R, out, scale, rows
        )
        launcher.launch(
            grid=(num_m_blocks_idx, scale_N_idx, 1),
            block=(GROUP_QUANT_BLOCK_SIZE, 1, 1),
            stream=stream,
        )

    return launch_per_1x32_fp4_quant_block_rotation_mfma

