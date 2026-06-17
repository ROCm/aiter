# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""FlyDSL dynamic per-tensor scaled fp8 quant (E4M3).

Mirrors quant_kernels.cu's ``dynamic_per_tensor_quant``: kernel A reduces
``max(|x|)`` and atomic-updates global scale; kernel B emits ``y = x/scale``.
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


def _waitcnt0_barrier():
    """``s_waitcnt vmcnt(0) lgkmcnt(0)`` + ``s_barrier`` via inline asm.

    Use after a global->LDS DMA whose filled region is shared across waves.
    """
    _llvm.InlineAsmOp(
        res=None,
        operands_=[],
        asm_string="s_waitcnt vmcnt(0) lgkmcnt(0)\ns_barrier",
        constraints="",
        has_side_effects=True,
        is_align_stack=False,
    )


def _waitcnt0():
    """``s_waitcnt vmcnt(0) lgkmcnt(0)`` only -- no ``s_barrier``.

    For a per-wave global->LDS DMA tile (no cross-wave sharing).
    """
    _llvm.InlineAsmOp(
        res=None,
        operands_=[],
        asm_string="s_waitcnt vmcnt(0) lgkmcnt(0)",
        constraints="",
        has_side_effects=True,
        is_align_stack=False,
    )


__all__ = [
    "build_dynamic_per_tensor_quant_module",
    "build_per_1x32_fp4_quant_module",
    "build_per_1x32_fp4_quant_hadamard_module",
    "build_per_1x32_fp4_quant_block_rotation_module",
    "build_per_1x32_fp4_quant_block_rotation_mfma_module",
    "build_per_1x32_fp4_quant_block_rotation_mfma_moe_sorting_module",
]

# Constants matching the CUDA reference (quant_kernels.cu).
BLOCK_THREADS = 256
WARP_SIZE = 64
NUM_WAVES = BLOCK_THREADS // WARP_SIZE  # 4 on CDNA wave64
# VEC elements/thread/iter; VEC=8 == 16-byte vector load for bf16/fp16.
VEC = 8

# E4M3 max magnitude (same for FN/FNUZ), so scale derivation is identical.
_FP8_E4M3_MAX = 448.0


def _dtype_max(out_dtype: str) -> float:
    if out_dtype == "fp8":
        return _FP8_E4M3_MAX
    raise ValueError(f"unsupported output dtype: {out_dtype!r}")


def _input_elem_mlir_type(in_dtype: str):
    """Resolve the bf16/fp16 element type. Must be called inside an MLIR
    Context (a ``@flyc.kernel`` body), since ``T`` needs an active context."""
    if in_dtype == "bf16":
        return T.bf16
    if in_dtype in ("fp16", "f16"):
        return T.f16
    raise ValueError(f"unsupported input dtype: {in_dtype!r}")


# 64-bit addressing helpers for ``rows*cols >= 2**31`` (32-bit buffer offset
# overflow). Selected at build time via ``use_ptr64``; call inside a kernel.
def _global_base_ptr(tensor):
    """Return ``(base_ptr, ptr_ty)`` for addrspace-1 GEP addressing."""
    from flydsl._mlir.dialects import fly as _fly

    ptr_ty = ir.Type.parse("!llvm.ptr<1>")
    return _fly.extract_aligned_pointer_as_index(ptr_ty, tensor), ptr_ty


def _ptr64_load_dwords(base_ptr, ptr_ty, elem_off_index, elem_bytes, vec_dw):
    """64-bit GEP + global load of ``vec_dw`` i32 dwords at byte address
    ``elem_off_index * elem_bytes`` (scalar i32 if vec_dw==1, else vec)."""
    i32 = T.i32
    byte_off = arith.index_cast(
        T.i64, elem_off_index * arith.constant(int(elem_bytes), type=T.index)
    )
    ptr = _llvm.GEPOp(
        ptr_ty,
        base_ptr,
        [byte_off],
        [-2147483648],
        T.i8,
        _llvm.GEPNoWrapFlags.none,
    ).result
    align = int(vec_dw) * 4
    if int(vec_dw) == 1:
        return _llvm.LoadOp(i32, ptr, alignment=align).result
    return _llvm.LoadOp(T.vec(int(vec_dw), i32), ptr, alignment=align).result


def _ptr64_store(value, base_ptr, ptr_ty, byte_off_index, align):
    """64-bit GEP + global store of ``value`` at byte address ``byte_off_index``."""
    byte_off = arith.index_cast(T.i64, byte_off_index)
    ptr = _llvm.GEPOp(
        ptr_ty,
        base_ptr,
        [byte_off],
        [-2147483648],
        T.i8,
        _llvm.GEPNoWrapFlags.none,
    ).result
    _llvm.StoreOp(value, ptr, alignment=int(align))


# Kernel builder (cached so we don't re-JIT for the same shape/dtype combo).
@functools.lru_cache(maxsize=None)
def build_dynamic_per_tensor_quant_module(
    cols: int,
    in_dtype: str = "bf16",
    out_dtype: str = "fp8",
    use_ptr64: bool = False,
):
    """Build (and cache) a launcher for fp8 per-tensor dynamic quant.

    Parameters
    ----------
    cols: int
        Input last-dim (= row stride); positive multiple of ``VEC`` (=8).
    in_dtype: {"bf16", "fp16"}
        Input element dtype.
    out_dtype: {"fp8"}
        Output dtype; only fp8 (E4M3) implemented.
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

    # LDS allocator for cross-wave block reduce (NUM_WAVES f32s = 16B). Unique
    # sym name per cached build to avoid global collisions.
    gpu_arch = get_hip_arch()
    sym_tag = f"dq_pt_lds_{cols}_{in_dtype}_{out_dtype}"
    allocator = SmemAllocator(None, arch=gpu_arch, global_sym_name=sym_tag)
    lds_red_offset = allocator._align(allocator.ptr, 16)
    allocator.ptr = lds_red_offset + NUM_WAVES * 4  # NUM_WAVES * sizeof(f32)

    # Kernel A: data_to_scale -- compute global max(|x|)/dtype_max via atomic.
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

        # OOB protection: the rsrc-mask trick fails for dynamic-shape memrefs
        # (num_records = 0xFFFFFFFF), so we manually clamp + arith.select.
        in_rsrc = buffer_ops.create_buffer_resource(inp, max_size=True)
        scale_rsrc = buffer_ops.create_buffer_resource(scale, max_size=True)

        tid_i32 = ArithValue(tid)
        bid_i32 = ArithValue(bid)

        # row_elem_base = bid * cols  (element index of row start)
        row_elem_base = bid_i32 * cols_i32

        # Per-thread max accumulator (in registers, no scf scope).
        local_max = c0_f32
        for it in range_constexpr(num_iters):
            col_thread = (
                tid_i32 * vec_i32
                + arith.constant(it * cols_per_iter, type=i32)
            )
            in_range = arith.cmpi(CmpIPredicate.ult, col_thread, cols_i32)

            elem_off = row_elem_base + col_thread
            # Clamp OOR lanes to row start (in-bounds load); contribution is
            # dropped below via ``arith.select``.
            safe_elem_off = arith.select(in_range, elem_off, row_elem_base)
            dw_off = safe_elem_off >> c1_i32  # 2 bf16/fp16 per dword
            vec_dw = VEC * elem_bytes_in // 4  # = 4 dwords for VEC=8

            if use_ptr64:
                # ``bid * cols`` overflows i32; read via 64-bit GEP.
                inp_base_ptr, _ptr_ty_as1 = _global_base_ptr(inp)
                safe_col_idx = arith.index_cast(
                    T.index, arith.select(in_range, col_thread, c0_i32)
                )
                elem_idx = arith.index_cast(
                    T.index, bid_i32
                ) * arith.constant(cols, type=T.index) + safe_col_idx
                raw_i32 = _ptr64_load_dwords(
                    inp_base_ptr, _ptr_ty_as1, elem_idx, elem_bytes_in, vec_dw
                )
            else:
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

        # Intra-wave reduce via shuffle-xor (wave_size=64).
        for sh in [32, 16, 8, 4, 2, 1]:
            off = arith.constant(sh, type=i32)
            peer = local_max.shuffle_xor(off, c64_i32)
            local_max = arith.maximumf(local_max, peer)

        # Cross-wave reduce via LDS + wave0 shuffle.
        lane_i32 = tid_i32 & arith.constant(WARP_SIZE - 1, type=i32)
        wave_i32 = tid_i32 >> arith.constant(6, type=i32)  # /64

        # Materialize the LDS view at kernel entry so it dominates every
        # store/load below (forces memref.view emission in this scope).
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
            # NUM_WAVES <= 64 so xor distances are harmless for OOR lanes.
            for sh in [32, 16, 8, 4, 2, 1]:
                off = arith.constant(sh, type=i32)
                peer = partial.shuffle_xor(off, c64_i32)
                partial = arith.maximumf(partial, peer)

            # Lane 0/wave 0 publishes block max (pre-mul inv_dtype_max) via
            # integer atomic-max on the bit pattern (no f32 atomic on gfx9xx).
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

    # Kernel B: scaled_quant -- apply ``y = x / scale`` and pack to fp8.
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

        # See data_to_scale_kernel: manual offset-clamp + arith.select for OOB.
        in_rsrc = buffer_ops.create_buffer_resource(inp, max_size=True)
        out_rsrc = buffer_ops.create_buffer_resource(out, max_size=True)
        scale_rsrc = buffer_ops.create_buffer_resource(scale, max_size=True)

        tid_i32 = ArithValue(tid)
        bid_i32 = ArithValue(bid)

        # Uniform broadcast: every thread loads the same scale[0].
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
            # Manual OOB clamp: load from row start when OOR, gate store later.
            safe_elem_off = arith.select(in_range, elem_off, row_elem_base)
            dw_off_in = safe_elem_off >> c1_i32  # 2 bf16/fp16 per dword
            vec_dw_in = VEC * elem_bytes_in // 4  # 4

            if use_ptr64:
                # ``bid * cols`` overflows i32; read/write via 64-bit GEPs.
                inp_base_ptr, _ptr_ty_as1 = _global_base_ptr(inp)
                out_base_ptr, _ = _global_base_ptr(out)
                row_base_idx = arith.index_cast(
                    T.index, bid_i32
                ) * arith.constant(cols, type=T.index)
                safe_col_idx = arith.index_cast(
                    T.index, arith.select(in_range, col_thread, c0_i32)
                )
                elem_idx_in = row_base_idx + safe_col_idx
                raw_i32 = _ptr64_load_dwords(
                    inp_base_ptr, _ptr_ty_as1, elem_idx_in, elem_bytes_in,
                    vec_dw_in,
                )
            else:
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

            # Pack VEC=8 -> 2 fp8 dwords (each cvt writes 2 bytes; 2 per dword).
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

            # Gate the store on in_range (buffer_store mask remaps offset to
            # 0x7FFFFFFF which is < the dynamic-shape rsrc bound).
            _if_store = _scf.IfOp(in_range)
            with ir.InsertionPoint(_if_store.then_block):
                if use_ptr64:
                    # fp8: 1 byte/elem, byte offset == elem_off (overflows
                    # i32); recompute in 64-bit.
                    elem_idx_out = row_base_idx + arith.index_cast(
                        T.index, col_thread
                    )
                    _ptr64_store(
                        packed_vec, out_base_ptr, _ptr_ty_as1, elem_idx_out, 8
                    )
                else:
                    buffer_ops.buffer_store(
                        packed_vec,
                        out_rsrc,
                        elem_off,
                        offset_is_bytes=True,
                    )
                _scf.YieldOp([])

    # Host-side JIT launcher: launches both kernels in sequence.
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


# MXFP4 per-1x32 dynamic quant (CUDA dynamic_per_group_scaled_quant): 1
# thread/32-elem group -> absMax -> E8M0 (next-pow2 RNE) -> cvt fp4, 1 launch.

# Block / group geometry constants from the CUDA reference.
GROUP_QUANT_BLOCK_SIZE = 64  # `groupQuantBlockSize` in quant_kernels.cu
FP4_GROUP_SIZE = 32          # only group_size=32 is supported in Step 1


@functools.lru_cache(maxsize=None)
def build_per_1x32_fp4_quant_module(
    cols: int,
    in_dtype: str = "bf16",
    shuffle_scale: bool = False,
    use_ptr64: bool = False,
):
    """Build (and cache) a launcher for MXFP4 per-1x32 dynamic quant.

    Parameters
    ----------
    cols: int
        Input last-dim (=row stride); positive multiple of FP4_GROUP_SIZE (32).
    in_dtype: {"bf16", "fp16"}
        Input element dtype.
    shuffle_scale: bool
        Shuffled E8M0 scale layout; unsupported (raises) -- only logical order.

    Launcher: ``launch(inp, out, scale, rows, scaleN_pad, stream)`` with ``out``
    uint8 ``(rows, cols//2)`` (2 fp4/byte) and ``scale`` uint8 e8m0.
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
    # BUFFER_LOAD caps at 4xi32 (16B), so split the 64B group into 4x8 elems.
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
        # x = row, y = group-in-row. Use ``//`` for i32 floor-div (``/``
        # would promote to float).
        x_i32 = group_id_i32 // scale_N_i32
        y_i32 = group_id_i32 % scale_N_i32

        # Tail threads in the last block may map past total_groups; gate them.
        total_groups_i32 = ArithValue(total_groups)
        in_range_group = arith.cmpi(
            CmpIPredicate.ult, group_id_i32, total_groups_i32
        )

        # Load 32 elements as 4 chunks of vec<8,bf16> (4 dwords each).
        elem_off_i32 = group_id_i32 * group_size_i32
        # Clamp OOR groups to row 0 so the load always touches valid memory.
        safe_elem_off = arith.select(in_range_group, elem_off_i32, c0_i32)
        in_dw_off_base = safe_elem_off >> c1_i32  # 2 bf16/fp16 per dword

        chunks = []
        if use_ptr64:
            # ``group_id * group_size`` overflows i32; gather via 64-bit GEP.
            inp_base_ptr, _ptr_ty_as1 = _global_base_ptr(inp)
            safe_gid_idx = arith.index_cast(
                T.index, arith.select(in_range_group, group_id_i32, c0_i32)
            )
            base_elem_idx = safe_gid_idx * arith.constant(
                FP4_GROUP_SIZE, type=T.index
            )
            for ci in range_constexpr(LOADS_PER_GROUP):
                elem_idx_ci = base_elem_idx + arith.constant(
                    ci * ELEMS_PER_LOAD, type=T.index
                )
                raw_i32_c = _ptr64_load_dwords(
                    inp_base_ptr, _ptr_ty_as1, elem_idx_ci, elem_bytes_in,
                    DWORDS_PER_LOAD,
                )
                chunks.append(vector.bitcast(in_chunk_vec_ty, raw_i32_c))
        else:
            for ci in range_constexpr(LOADS_PER_GROUP):
                dw_off_c = in_dw_off_base + arith.constant(
                    ci * DWORDS_PER_LOAD, type=i32
                )
                raw_i32_c = buffer_ops.buffer_load(
                    in_rsrc, dw_off_c, vec_width=DWORDS_PER_LOAD, dtype=i32
                )
                chunks.append(vector.bitcast(in_chunk_vec_ty, raw_i32_c))

        # Compute absMax across all 32 elements.
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

        # fp4_scale: round absMax UP to next power of 2 (RNE). f32 bits:
        # [sign:1][exp:8][mantissa:23].
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

        # E8M0 scale byte = bits 30..23 of inv_scale.
        inv_scale_u32 = inv_scale.bitcast(i32)
        e8m0_byte_i32 = (inv_scale_u32 >> c_23) & c_exp_mask
        e8m0_byte_i8 = arith.trunci(T.i8, e8m0_byte_i32)

        # Scale address = x * scale_N + y, in BYTES (1 byte per group).
        scale_off_i32 = x_i32 * scale_N_i32 + y_i32

        # Convert 32 bf16/fp16 -> 4 dwords of fp4 with HW scale. Each cvt
        # packs 2 fp4 into one of 4 byte-slots via ``dst_sel_index``.
        if in_dtype == "bf16":
            cvt_op = rocdl.cvt_scalef32_pk_fp4_bf16
        else:  # fp16
            cvt_op = rocdl.cvt_scalef32_pk_fp4_f16
        in_pair_vec_ty = T.vec(2, in_elem_ty)

        # Extract pair-by-pair via scalar extracts + vector.from_elements (MLIR
        # rejects nested vector-of-vector). Each 8-elem chunk = 1 output dword.
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

        # Write outputs only for in-range groups. Out byte off = group_id * 16.
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
        # num_blocks = ceil(total_groups/64); total_groups = rows*scale_N.
        # Both computed host-side (launch grid must be index-typed).
        num_blocks_idx = arith.index_cast(T.index, num_blocks)

        launcher = per_1x32_fp4_quant_kernel(inp, out, scale, total_groups)
        launcher.launch(
            grid=(num_blocks_idx, 1, 1),
            block=(GROUP_QUANT_BLOCK_SIZE, 1, 1),
            stream=stream,
        )

    return launch_per_1x32_fp4_quant


# MXFP4 per-1x32 quant fused with H_32 Hadamard rotation: applies the
# orthonormal H_32/sqrt(32) (5-stage radix-2 butterfly) in-register pre-quant.

# Hadamard normalization constant: 1 / sqrt(32).
_INV_SQRT_32 = 0.17677669529663687


@functools.lru_cache(maxsize=None)
def build_per_1x32_fp4_quant_hadamard_module(
    cols: int,
    in_dtype: str = "bf16",
    shuffle_scale: bool = False,
    use_ptr64: bool = False,
):
    """Build (and cache) an H_32-fused MXFP4 per-1x32 dynamic quant launcher.

    Params mirror ``build_per_1x32_fp4_quant_module``. Launcher:
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

        # Issue all 4 loads up-front (no inter-dependency) so the scheduler
        # batches them as back-to-back vmem instructions.
        elem_off_i32 = group_id_i32 * group_size_i32
        safe_elem_off = arith.select(in_range_group, elem_off_i32, c0_i32)
        in_dw_off_base = safe_elem_off >> c1_i32  # 2 elems/dword

        chunks = []
        if use_ptr64:
            # ``group_id * group_size`` overflows i32; gather via 64-bit GEP.
            inp_base_ptr, _ptr_ty_as1 = _global_base_ptr(inp)
            safe_gid_idx = arith.index_cast(
                T.index, arith.select(in_range_group, group_id_i32, c0_i32)
            )
            base_elem_idx = safe_gid_idx * arith.constant(
                FP4_GROUP_SIZE, type=T.index
            )
            for ci in range_constexpr(LOADS_PER_GROUP):
                elem_idx_ci = base_elem_idx + arith.constant(
                    ci * ELEMS_PER_LOAD, type=T.index
                )
                raw_i32_c = _ptr64_load_dwords(
                    inp_base_ptr, _ptr_ty_as1, elem_idx_ci, elem_bytes_in,
                    DWORDS_PER_LOAD,
                )
                chunks.append(vector.bitcast(in_chunk_vec_ty, raw_i32_c))
        else:
            for ci in range_constexpr(LOADS_PER_GROUP):
                dw_off_c = in_dw_off_base + arith.constant(
                    ci * DWORDS_PER_LOAD, type=i32
                )
                raw_i32_c = buffer_ops.buffer_load(
                    in_rsrc, dw_off_c, vec_width=DWORDS_PER_LOAD, dtype=i32
                )
                chunks.append(vector.bitcast(in_chunk_vec_ty, raw_i32_c))

        # Extend each chunk to f32 -> 32 scalar SSA values. extf chunk-by-chunk
        # so the SSA deps expose compute-load overlap to the scheduler.
        x_vals = [None] * FP4_GROUP_SIZE
        for ci in range_constexpr(LOADS_PER_GROUP):
            chunk_f32 = chunks[ci].extf(T.vec(ELEMS_PER_LOAD, f32))
            for vi in range_constexpr(ELEMS_PER_LOAD):
                x_vals[ci * ELEMS_PER_LOAD + vi] = vector.extract(
                    chunk_f32, static_position=[vi], dynamic_position=[]
                )

        # H_32 Walsh-Hadamard butterfly (radix-2, 5 stages, stride 1->16; early
        # stages mix within a chunk). range_constexpr -> Python unroll for SSA.
        for s in (1, 2, 4, 8, 16):
            for i in range_constexpr(FP4_GROUP_SIZE):
                if (i & s) == 0:
                    j = i + s
                    a = x_vals[i]
                    b = x_vals[j]
                    x_vals[i] = a + b
                    x_vals[j] = a - b

        # Normalize: y = H_32 @ x / sqrt(32). 32 fmuls (cheap vs load latency).
        for i in range_constexpr(FP4_GROUP_SIZE):
            x_vals[i] = x_vals[i] * c_inv_sqrt32

        # absMax of the rotated, normalized values.
        abs_max = c0_f32
        for i in range_constexpr(FP4_GROUP_SIZE):
            abs_v = _llvm.call_intrinsic(
                f32, "llvm.fabs.f32", [x_vals[i]], [], []
            )
            abs_max = arith.maximumf(abs_max, abs_v)

        # fp4_scale: round absMax UP to next power of 2 (RNE).
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
        inv_scale = next_pow2_f32 * c_quarter_f32  # = scale of (rotated) group

        # Write the E8M0 scale byte.
        inv_scale_u32 = inv_scale.bitcast(i32)
        e8m0_byte_i32 = (inv_scale_u32 >> c_23) & c_exp_mask
        e8m0_byte_i8 = arith.trunci(T.i8, e8m0_byte_i32)
        scale_off_i32 = x_i32 * scale_N_i32 + y_i32

        # Pack 32 rotated f32 values into 4 fp4 dwords (f32 cvt).
        out_dwords = []
        for dw in range_constexpr(DWORDS_PER_GROUP):
            packed = c0_i32
            for sel in range_constexpr(4):
                idx = dw * 4 + sel
                e0 = x_vals[idx * 2]
                e1 = x_vals[idx * 2 + 1]
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


# MXFP4 per-1x32 quant fused with per-block rotation R (runtime (scale_N,32,32);
# y = R[b] @ x). 2-D grid fixes b per block; R[b] cooperatively cached in LDS.


@functools.lru_cache(maxsize=None)
def build_per_1x32_fp4_quant_block_rotation_module(
    cols: int,
    in_dtype: str = "bf16",
    rot_dtype: str = "bf16",
    shuffle_scale: bool = False,
    use_ptr64: bool = False,
):
    """Build (and cache) the per-block-rotated MXFP4 per-1x32 quant launcher.

    Parameters
    ----------
    cols
        Input last-dim; multiple of 32.
    in_dtype
        Input dtype, ``"bf16"`` / ``"fp16"``.
    rot_dtype
        R dtype, ``"bf16"`` / ``"fp16"`` / ``"f32"``.
    shuffle_scale
        Reserved; must be ``False``.

    Returns a ``@flyc.jit`` launcher
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
    R_LDS_BYTES = R_NUMEL_PER_BLOCK * 4      # 4 KB/block (R cached as f32)

    # Cooperative load: 64 threads, 1024 R elements -> 16/thread = 4 vec4 stores.
    R_ELEMS_PER_THREAD = R_NUMEL_PER_BLOCK // GROUP_QUANT_BLOCK_SIZE   # 16
    LDS_VEC = 4
    R_LDS_STORES_PER_THREAD = R_ELEMS_PER_THREAD // LDS_VEC            # 4

    # LDS allocator: one R[b] copy (4 KB) per block.
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

        # Stage 1: cooperative LDS load of R[b] (1024 f32 = 4 KB).
        # Materialize the LDS view at kernel entry so it dominates all uses.
        base_ptr = allocator.get_base()
        R_lds_ptr = SmemPtr(
            base_ptr, R_lds_offset, f32, shape=(R_NUMEL_PER_BLOCK,)
        )
        R_lds_view = R_lds_ptr.get()

        # Thread tid handles flat R[b] elements [tid*16, tid*16+16); absolute
        # offset = b*1024 + tid*16.
        R_b_elem_base = b_i32 * c_R_block_elems_i32
        thread_elem_off = tid_i32 * arith.constant(
            R_ELEMS_PER_THREAD, type=i32
        )
        R_thread_elem_base = R_b_elem_base + thread_elem_off

        if rot_is_f32:
            # Load LDS_VEC f32 (16B = 4 dwords) per step, R_LDS_STORES_PER_THREAD
            # vec4 ops.
            for li in range_constexpr(R_LDS_STORES_PER_THREAD):
                step_elem_off = R_thread_elem_base + arith.constant(
                    li * LDS_VEC, type=i32
                )
                dw_off_li = step_elem_off  # f32: 1 elem per dword
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
            # bf16/fp16 R: load 2 dwords = 4 bf16, extf to vec4 f32, store to LDS.
            rot_elem_ty = _input_elem_mlir_type(
                "bf16" if rot_dtype == "bf16" else "fp16"
            )
            for li in range_constexpr(R_LDS_STORES_PER_THREAD):
                step_elem_off = R_thread_elem_base + arith.constant(
                    li * LDS_VEC, type=i32
                )
                dw_off_li = step_elem_off >> c1_i32  # 2 elems/dword
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

        # Stage 2: load 32 input elements for this (m, b).
        # elem_off = m*cols + b*32; clamp OOR m to 0 (stores gated later).
        safe_m_i32 = arith.select(in_range_m, m_i32, c0_i32)
        cols_i32 = arith.constant(cols, type=i32)
        elem_off_i32 = (
            safe_m_i32 * cols_i32 + b_i32 * group_size_i32
        )
        in_dw_off_base = elem_off_i32 >> c1_i32  # 2 elems / dword

        chunks = []
        if use_ptr64:
            # ``m * cols`` overflows i32; gather via 64-bit GEP.
            inp_base_ptr, _ptr_ty_as1 = _global_base_ptr(inp)
            base_elem_idx = arith.index_cast(
                T.index, safe_m_i32
            ) * arith.constant(cols, type=T.index) + arith.index_cast(
                T.index, b_i32
            ) * arith.constant(FP4_GROUP_SIZE, type=T.index)
            for ci in range_constexpr(LOADS_PER_GROUP):
                elem_idx_ci = base_elem_idx + arith.constant(
                    ci * ELEMS_PER_LOAD, type=T.index
                )
                raw_i32_c = _ptr64_load_dwords(
                    inp_base_ptr, _ptr_ty_as1, elem_idx_ci, elem_bytes_in,
                    DWORDS_PER_LOAD,
                )
                chunks.append(vector.bitcast(in_chunk_vec_ty, raw_i32_c))
        else:
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

        # Stage 3: 32x32 mat-vec, y[i] = sum_j R_lds[i*32+j]*x_vals[j]. vec4 R
        # reads; all 64 lanes hit the same LDS address (single-cycle broadcast).
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

        # Stage 4: amax + E8M0 scale (same RNE bit-trick as other variants).
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

        # Stage 5: pack 32 f32 rotated values into 4 fp4 dwords.
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
        # LDS finalize: emit the gpu.module global memref for R[b] (per entry).
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


# MFMA variant of build_per_1x32_fp4_quant_block_rotation_module: 32x32 mat-vec
# via 16 MFMA 16x16x16 ops + LDS C-transpose. in_dtype==rot_dtype, bf16/fp16.


@functools.lru_cache(maxsize=None)
def build_per_1x32_fp4_quant_block_rotation_mfma_module(
    cols: int,
    in_dtype: str = "bf16",
    rot_dtype: str = "bf16",
    shuffle_scale: bool = False,
    rot_transposed: bool = False,
    use_ptr64: bool = False,
):
    """MFMA-accelerated per-block-rotated MXFP4 per-1x32 quant launcher.

    Drop-in for :func:`build_per_1x32_fp4_quant_block_rotation_module` when
    ``in_dtype == rot_dtype`` and ``rot_dtype != "f32"`` (same launch sig).

    Parameters
    ----------
    rot_transposed : bool, default False
        ``rot_R`` (scale_N, 32, 32) layout: False -> R[b,h,g] (Y=X@R.T), True ->
        R[b,g,h] (Y=X@R). Equivalent results; swaps one LDS B-frag index.
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
    # Normalize fp16/f16 -> "fp16"; bf16 stays.
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

    # MFMA tile geometry (16x16x16, gfx942 _1k op).
    M_TILE = 16
    N_TILE = 16
    K_TILE = 16
    NUM_M_TILES = GROUP_QUANT_BLOCK_SIZE // M_TILE   # 4
    NUM_N_TILES = FP4_GROUP_SIZE // N_TILE           # 2
    NUM_K_TILES = FP4_GROUP_SIZE // K_TILE           # 2
    WMMA_FRAG_VALS = 4   # 4 elements/lane for A, B, C in 16x16x16

    # LDS layout: R[b] (bf16/fp16) + Y tile (f32).
    R_NUMEL_PER_BLOCK = g * g                        # 1024
    R_LDS_BYTES = R_NUMEL_PER_BLOCK * 2              # 2 KB (bf16/fp16)
    Y_LDS_BYTES = GROUP_QUANT_BLOCK_SIZE * FP4_GROUP_SIZE * 4  # 8 KB (f32)

    # Cooperative R load: 16 elems/thread in 2x vec4_i32 (= 8 bf16) loads.
    R_ELEMS_PER_THREAD = R_NUMEL_PER_BLOCK // GROUP_QUANT_BLOCK_SIZE  # 16
    R_LOAD_VEC = 8                                   # bf16 per coop load step
    R_LOAD_STEPS = R_ELEMS_PER_THREAD // R_LOAD_VEC  # 2

    # Per-thread Y read-back: 32 f32 / 4 = 8 vec4 LDS loads.
    Y_READ_VEC = 4
    Y_READ_STEPS = FP4_GROUP_SIZE // Y_READ_VEC      # 8

    # LDS allocator.
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

        # compile-time constants
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
        # k_off_in_tile = 4 * (t // 16).
        lane_mod_16 = tid_i32 & c15_i32
        lane_div_16 = tid_i32 >> c4_i32
        k_off_in_tile = lane_div_16 << c2_i32  # 0, 4, 8, 12

        # LDS views.
        base_ptr = allocator.get_base()
        R_lds_view = SmemPtr(
            base_ptr, R_lds_offset, rot_elem_ty,
            shape=(R_NUMEL_PER_BLOCK,),
        ).get()
        Y_lds_view = SmemPtr(
            base_ptr, Y_lds_offset, f32,
            shape=(GROUP_QUANT_BLOCK_SIZE * FP4_GROUP_SIZE,),
        ).get()

        # Stage 1: cooperative LDS load of R[b] (2 KB) -> R_lds[h*32+g]==R[b,h,g].
        # Non-transposed: vec8 store. Transposed: 8 scalar stride-32 stores.
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
                # contiguous vec8 store, dest = source order
                lds_idx = thread_elem_off + arith.constant(
                    li * R_LOAD_VEC, type=i32
                )
                lds_idx_index = arith.index_cast(T.index, lds_idx)
                vector.store(
                    r_vec_bf, R_lds_view, [lds_idx_index], alignment=16
                )
            else:
                # transposed: 8 scalar stride-32 stores at R_lds[(h_src_base+j)
                # *32 + g_src] (g_src = flat//32, h_src_base = flat%32).
                flat_base_i32 = thread_elem_off + arith.constant(
                    li * R_LOAD_VEC, type=i32
                )
                g_src_i32 = flat_base_i32 >> arith.constant(5, type=i32)  # //32
                h_src_base_i32 = flat_base_i32 & arith.constant(31, type=i32)  # %32
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

        # Stage 2: MFMA loop, 4 M x 2 N x 2 K tiles. A := x, B :=
        # R_lds[n_lds*32+k_lds] (4 K-consecutive elems/lane -> ds_read_b64).
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
            # OOR rows clamp to row 0 (valid load); their MFMA outputs are
            # never stored (final store gated on ``in_range_m``).
            safe_m_row = arith.select(row_in_range, m_row_i32, c0_i32)
            # Group lives at columns [b*32, b*32+32); A frags add the b*32 col
            # offset.
            if use_ptr64:
                # ``m * cols`` overflows i32; gather A frags via 64-bit GEP.
                inp_base_ptr, _ptr_ty_as1 = _global_base_ptr(inp)
                row_base_elem_idx = arith.index_cast(
                    T.index, safe_m_row
                ) * arith.constant(cols, type=T.index) + arith.index_cast(
                    T.index, b_i32
                ) * arith.constant(FP4_GROUP_SIZE, type=T.index)
            else:
                row_off_in_inp = safe_m_row * cols_i32 + b_i32 * group_size_i32

            for n_tile in range_constexpr(NUM_N_TILES):
                # zero-init accumulator (vec4 f32 per lane)
                acc = vector.from_elements(
                    T.vec(WMMA_FRAG_VALS, f32),
                    [c0_f32] * WMMA_FRAG_VALS,
                )

                for k_tile in range_constexpr(NUM_K_TILES):
                    # A frag: x[m_row, b*32 + k_col : ... +4]
                    k_col_i32 = (
                        arith.constant(k_tile * 16, type=i32) + k_off_in_tile
                    )
                    if use_ptr64:
                        elem_off_a_idx = row_base_elem_idx + arith.index_cast(
                            T.index, k_col_i32
                        )
                        raw_a = _ptr64_load_dwords(
                            inp_base_ptr, _ptr_ty_as1, elem_off_a_idx, 2, 2
                        )
                    else:
                        elem_off_a = row_off_in_inp + k_col_i32
                        dw_off_a = elem_off_a >> c1_i32  # 2 elem/dword
                        # vec2 i32 = 8 bytes = 4 bf16
                        raw_a = buffer_ops.buffer_load(
                            in_rsrc, dw_off_a, vec_width=2, dtype=i32
                        )
                    a_frag_bf = vector.bitcast(
                        T.vec(WMMA_FRAG_VALS, in_elem_ty), raw_a
                    )

                    # B frag: R_lds[n_local*32 + k_local : ... +4]
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

                    # MFMA
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

                # Write C tile to Y_lds (row-major transpose); scalar stores at
                # (m_tile*16 + 4*(t/16) + i, n_tile*16 + t%16), stride-32 along m.
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

        # Stage 3: per-thread row gather + amax + E8M0 + cvt fp4.
        # Thread tid owns Y_lds[tid*32 + 0..31]. Read 8x vec4 f32.
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

        # amax
        abs_max = c0_f32
        for i in range_constexpr(FP4_GROUP_SIZE):
            abs_v = _llvm.call_intrinsic(
                f32, "llvm.fabs.f32", [y_vals[i]], [], []
            )
            abs_max = arith.maximumf(abs_max, abs_v)

        # E8M0 scale (same RNE bit-trick as scalar variant)
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

        # cvt to fp4
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
        # LDS finalize: emit the gpu.module global memref (R[b] + Y tile).
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


# build_per_1x32_fp4_quant_block_rotation_mfma_moe_sorting_module: the MFMA
# rotation+quant kernel fused with MoE scale-sort (input gather/stores remapped).


@functools.lru_cache(maxsize=None)
def build_per_1x32_fp4_quant_block_rotation_mfma_moe_sorting_module(
    cols: int,
    in_dtype: str = "bf16",
    rot_dtype: str = "bf16",
    topk: int = 1,
    rot_transposed: bool = False,
    use_ptr64: bool = False,
    waves_per_wg: int = 1,
    blocks_per_wg: int = 1,
    preload_b: bool = True,
    stage_x_lds: bool = True,
    rot_preshuffled: bool = True,
    swizzle_x_lds: bool = True,
    sched_hints: bool = True,
    persist_m: int = 1,
    xcd_remap: bool = False,
):
    """MFMA rotation + per-1x32 MXFP4 quant fused with MoE scale-sort.

    Same MFMA path as :func:`build_per_1x32_fp4_quant_block_rotation_mfma_module`
    with MoE scale-sort folded in (input gather + output stores remapped).

    Launcher::

        launch(inp, rot_R, sorted_ids, num_valid_ids,
               out, out_scale, num_sorted_blocks, num_tokens, stream)

    Parameters
    ----------
    cols : int
        Activation last-dim (compile-time constant).
    in_dtype, rot_dtype : str
        Element type for ``inp`` / ``rot_R``; must be equal, bf16/fp16 (no f32).
    topk : int, default 1
        Compile-time topk for source-row indexing (1 for stage1, real topk for
        stage2). Each combo gets its own JIT kernel.
    rot_transposed : bool, default False
        ``rot_R`` storage layout; see
        :func:`build_per_1x32_fp4_quant_block_rotation_mfma_module`.
    use_ptr64 : bool, default False
        64-bit GEP gather for ``inp`` when ``rows*cols >= 2**31``; else cheap
        buffer load. Build-time, no runtime branch.

    Performance knobs (layout/schedule-only, bit-exact across settings):

    ``waves_per_wg`` (W)
        Power-of-two waves per WG sharing one R[b]. Needs DPP transpose.
    ``blocks_per_wg`` (K)
        Consecutive distinct-R[b] blocks per WG; K>1 serializes loads (rare win).
    ``preload_b``
        Load B-fragments HBM->VGPR (no R-LDS). ``rot_preshuffled`` is its
        coalesced variant; False keeps the cooperative R[b]->LDS path.
    ``stage_x_lds`` / ``swizzle_x_lds``
        Stage the 64x32 activation tile HBM->LDS; swizzle adds XOR16 anti-conflict.
    ``sched_hints``
        ``sched_group_barrier`` pipeline hints on the MFMA loop.
    ``persist_m``
        Persistent kernel: WG handles persist_m row-block groups; R[b] hoisted.
    ``xcd_remap``
        Remap WG ids for MI300 XCD L2 locality.

    Returns a ``@flyc.jit`` launcher; grid = ``(ceil(num_row_blocks/
    (W*persist_m)), scale_N//K, 1)``, block ``(GROUP_QUANT_BLOCK_SIZE*W, 1, 1)``.
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
            "MFMA+sort path requires bf16/fp16 R; for fp32 R, use the "
            "two-kernel chain (block_rotation + mxfp4_moe_sort)."
        )
    if rot_dtype not in ("bf16", "fp16", "f16"):
        raise ValueError(f"unsupported R dtype: {rot_dtype!r}")
    _norm_in = "bf16" if in_dtype == "bf16" else "fp16"
    _norm_rot = "bf16" if rot_dtype == "bf16" else "fp16"
    if _norm_in != _norm_rot:
        raise ValueError(
            f"MFMA+sort path requires in_dtype == rot_dtype (got "
            f"in_dtype={in_dtype!r}, rot_dtype={rot_dtype!r})"
        )
    if topk < 1:
        raise ValueError(f"topk must be >= 1, got {topk}")

    PAIRS_PER_GROUP = FP4_GROUP_SIZE // 2       # 16
    DWORDS_PER_GROUP = PAIRS_PER_GROUP // 4     # 4
    scale_N = cols // FP4_GROUP_SIZE
    g = FP4_GROUP_SIZE                          # 32

    # Block-group factor: WG handles K consecutive ``b`` blocks (grid y ->
    # scale_N//K). Distinct R[b] per block, so K>1 serializes loads (default 1).
    if not isinstance(blocks_per_wg, int) or blocks_per_wg < 1:
        raise ValueError(
            f"blocks_per_wg must be a positive int, got {blocks_per_wg!r}"
        )
    if scale_N % blocks_per_wg != 0:
        raise ValueError(
            f"blocks_per_wg={blocks_per_wg} must divide scale_N={scale_N} "
            f"(cols={cols})"
        )
    K = blocks_per_wg

    # Number of XCDs on the MI300 die; used by the optional L2-locality remap.
    NUM_XCD = 8

    # Persistent kernel: WG processes persist_m row-block groups along grid-x;
    # R[b] (b == grid-y) is loaded once per WG, only row geometry changes.
    if not isinstance(persist_m, int) or persist_m < 1:
        raise ValueError(f"persist_m must be a positive int, got {persist_m!r}")

    # In-register cross-lane transpose of the MFMA C tile, replacing the 8 KB
    # Y_LDS round-trip + barrier.
    import os as _os
    _DPP_TRANSPOSE = True

    # X-tile LDS staging: read each block's 64x32 activation tile HBM->LDS
    # coalesced, feed A-frags from LDS. Disabled on the ptr64 path.
    _stage_x_env = _os.environ.get("SINGLE_ROT_STAGE_X", "1")
    if _stage_x_env is not None:
        stage_x_lds = (_stage_x_env != "0")
    _staged_x = bool(stage_x_lds) and not use_ptr64

    # XOR16 bank-conflict swizzle on the staged-X tile (row-major [64][32]
    # bf16): XOR the K-byte offset by ((row & 3)*16) on both store and read.
    _swizzle_x = bool(swizzle_x_lds) and _staged_x
    _X_K_BLOCKS16 = (FP4_GROUP_SIZE * 2) // 16   # 4

    # MFMA hot-region software-pipeline hints. Pure perf knob.
    _sched_hints = bool(sched_hints)

    # R[b] pre-shuffle: host permuted R[b] into B-frag order for coalesced
    # HBM->VGPR loads (no R-LDS). Implies preload_b; absorbs rot_transposed.
    rot_preshuffled = bool(rot_preshuffled)
    preload_b = bool(preload_b)
    if rot_preshuffled and not preload_b:
        raise ValueError("rot_preshuffled requires preload_b=True")

    # Multi-wave WG: W waves of 64 lanes, each owns a 64-row block, all sharing
    # R[b] (b == grid-y). Amortizes the R[b] load; needs the DPP transpose.
    W = waves_per_wg
    if (
        not isinstance(W, int)
        or W < 1
        or (W & (W - 1)) != 0
        or (GROUP_QUANT_BLOCK_SIZE * W) > 512
    ):
        raise ValueError(
            f"waves_per_wg must be a power of two in "
            f"[1, {512 // GROUP_QUANT_BLOCK_SIZE}], got {waves_per_wg!r}"
        )
    LOG2_W = W.bit_length() - 1
    if W > 1 and not _DPP_TRANSPOSE:
        raise ValueError(
            "waves_per_wg > 1 requires the in-register DPP transpose."
        )
    BLOCK_DIM = GROUP_QUANT_BLOCK_SIZE * W

    # MFMA tile geometry (16x16x16, gfx942 _1k op) - same as no-sort variant.
    M_TILE = 16
    N_TILE = 16
    K_TILE = 16
    NUM_M_TILES = GROUP_QUANT_BLOCK_SIZE // M_TILE   # 4
    NUM_N_TILES = FP4_GROUP_SIZE // N_TILE           # 2
    NUM_K_TILES = FP4_GROUP_SIZE // K_TILE           # 2
    WMMA_FRAG_VALS = 4

    R_NUMEL_PER_BLOCK = g * g                        # 1024
    R_LDS_BYTES = R_NUMEL_PER_BLOCK * 2              # 2 KB (bf16/fp16)
    R_BLOCK_BYTES = R_NUMEL_PER_BLOCK * 2            # global byte stride b->b+1
    Y_LDS_BYTES = GROUP_QUANT_BLOCK_SIZE * FP4_GROUP_SIZE * 4  # 8 KB f32

    # Cooperative R[b] load uses all BLOCK_DIM = 64*W threads, each loading
    # R_NUMEL/BLOCK_DIM contiguous bf16 elems.
    R_ELEMS_PER_THREAD = R_NUMEL_PER_BLOCK // BLOCK_DIM  # 16 // W
    R_LOAD_VEC = min(8, R_ELEMS_PER_THREAD)             # bf16 per vec
    R_LOAD_STEPS = R_ELEMS_PER_THREAD // R_LOAD_VEC
    R_LOAD_VEC_I32 = R_LOAD_VEC // 2                     # 2 bf16 per i32

    # Direct global->LDS DMA (buffer_load_lds) for non-transposed R[b]: 16 B
    # (dwordx4) per lane, R_DMA_SLOTS chunks; at most 2 waves have work.
    R_DMA_BYTES = 16
    R_DMA_SLOTS = R_LDS_BYTES // R_DMA_BYTES             # 128
    R_DMA_ACTIVE_WAVES = min(W, R_DMA_SLOTS // 64)       # min(W, 2)
    R_DMA_ACTIVE_LANES = R_DMA_ACTIVE_WAVES * 64
    R_DMA_LOADS = (
        (R_DMA_SLOTS + R_DMA_ACTIVE_LANES - 1) // R_DMA_ACTIVE_LANES
    )

    Y_READ_VEC = 4
    Y_READ_STEPS = FP4_GROUP_SIZE // Y_READ_VEC      # 8

    # X-tile LDS staging geometry (one 64x32 tile per wave, row-major).
    X_TILE_ELEMS = GROUP_QUANT_BLOCK_SIZE * FP4_GROUP_SIZE   # 64*32 = 2048
    X_LDS_BYTES = X_TILE_ELEMS * 2                           # 4 KB / wave
    X_DMA_BYTES = 16
    X_DMA_LOADS = X_LDS_BYTES // (64 * X_DMA_BYTES)          # 4 (== NUM_M_TILES)

    gpu_arch = get_hip_arch()
    sym_tag = (
        f"fp4_rot_mfma_sort_lds_{cols}_{in_dtype}_{rot_dtype}_"
        f"t{topk}_rT{int(rot_transposed)}_w{W}_k{K}_pb{int(preload_b)}"
        f"_sx{int(_staged_x)}_psh{int(rot_preshuffled)}_swz{int(_swizzle_x)}"
        f"_sch{int(_sched_hints)}_pm{persist_m}_xcd{int(xcd_remap)}"
    )
    # R-LDS allocated only on the cooperative-LDS path; X-LDS per wave when staged.
    allocator = SmemAllocator(None, arch=gpu_arch, global_sym_name=sym_tag)
    R_lds_offset = allocator._align(allocator.ptr, 16)
    if not preload_b:
        allocator.ptr = R_lds_offset + R_LDS_BYTES
    Y_lds_offset = allocator._align(allocator.ptr, 16)
    if not _DPP_TRANSPOSE:
        allocator.ptr = Y_lds_offset + Y_LDS_BYTES
    X_lds_offset = allocator._align(allocator.ptr, 16)
    if _staged_x:
        allocator.ptr = X_lds_offset + W * X_LDS_BYTES

    @flyc.kernel
    def per_1x32_fp4_quant_block_rotation_mfma_moe_sorting_kernel(
        inp: fx.Tensor,            # (M_src, cols)               bf16 / fp16
        rot_R: fx.Tensor,          # (scale_N, 32, 32)           bf16 / fp16
        out: fx.Tensor,            # (M_src, cols // 2)          uint8 (fp4x2)
        out_scale: fx.Tensor,      # (M_src_pad, scale_N)        uint8 (e8m0)
        num_tokens_dyn: Int32,
    ):
        from flydsl._mlir.dialects import memref as _memref

        bid_x = fx.block_idx.x  # sorted-row chunk index
        bid_y = fx.block_idx.y  # b in [0, scale_N)
        tid = fx.thread_idx.x

        f32 = T.f32
        i32 = T.i32
        i16 = T.i16
        in_elem_ty = _input_elem_mlir_type(in_dtype)
        rot_elem_ty = _input_elem_mlir_type(
            "bf16" if rot_dtype == "bf16" else "fp16"
        )

        # compile-time constants
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
        topk_i32 = arith.constant(topk, type=i32)
        c_K_i32 = arith.constant(K, type=i32)

        in_rsrc = buffer_ops.create_buffer_resource(inp, max_size=True)
        out_rsrc = buffer_ops.create_buffer_resource(out, max_size=True)
        out_scale_rsrc = buffer_ops.create_buffer_resource(
            out_scale, max_size=True
        )
        rot_rsrc = buffer_ops.create_buffer_resource(rot_R, max_size=True)

        # 64-bit ``inp`` gather when ``rows*cols >= 2**31`` (else buffer load);
        # fp4/scale stores keep their <4 GB buffer offsets.
        if use_ptr64:
            from flydsl._mlir.dialects import fly as _fly

            _ptr_ty_as1 = ir.Type.parse("!llvm.ptr<1>")
            inp_base_ptr = _fly.extract_aligned_pointer_as_index(_ptr_ty_as1, inp)
            c_in_elem_bytes_idx = arith.constant(2, type=T.index)  # bf16 / fp16
            cols_idx = arith.constant(cols, type=T.index)
            group_size_idx = arith.constant(FP4_GROUP_SIZE, type=T.index)

        tid_i32 = ArithValue(tid)
        bid_x_i32 = ArithValue(bid_x)
        bid_y_i32 = ArithValue(bid_y)

        # XCD remap (L2 locality): transpose the HW round-robin (xcd, local)
        # grouping so a contiguous run co-locates on one XCD (<8 tail in place).
        if xcd_remap:
            _idx = T.index
            _c8 = arith.constant(NUM_XCD, type=_idx)
            _grid_x = ArithValue(
                arith.index_cast(_idx, ArithValue(gpu.grid_dim.x))
            )
            _pid = (
                ArithValue(arith.index_cast(_idx, bid_x_i32))
                + ArithValue(arith.index_cast(_idx, bid_y_i32)) * _grid_x
            )
            _grid_mn = _grid_x * arith.constant(scale_N // K, type=_idx)
            _per = _grid_mn / _c8
            _full = _per * _c8
            _remapped = (_pid % _c8) * _per + (_pid / _c8)
            _logical = ArithValue(
                arith.select(
                    arith.cmpi(CmpIPredicate.ult, _pid, _full),
                    _remapped, _pid,
                )
            )
            bid_x_i32 = ArithValue(arith.index_cast(i32, _logical % _grid_x))
            bid_y_i32 = ArithValue(arith.index_cast(i32, _logical / _grid_x))

        # Multi-wave: split flat thread id into (wave, lane). Each wave is a
        # full 64-lane wavefront; only `lane` (0..63) feeds the layout math.
        c6_i32 = arith.constant(6, type=i32)
        c63_i32 = arith.constant(63, type=i32)
        lane_i32 = tid_i32 & c63_i32
        wave_i32 = tid_i32 >> c6_i32 if W > 1 else c0_i32

        # WG owns blocks [b_base, b_base+K); all W waves share b = bid_y.
        b_base_i32 = bid_y_i32 * c_K_i32

        num_tokens_i32 = ArithValue(num_tokens_dyn)

        # Lane decomposition for MFMA: lane t -> (t % 16, t // 16).
        lane_mod_16 = lane_i32 & c15_i32
        lane_div_16 = lane_i32 >> c4_i32
        k_off_in_tile = lane_div_16 << c2_i32   # 0, 4, 8, 12

        # Natural-order traversal: wave owns contiguous source rows; out written
        # at the same row, scale in plain row-major (sort is downstream).
        total_rows_i32 = (
            num_tokens_i32 * topk_i32 if topk > 1 else num_tokens_i32
        )
        # Persist only: grid-x ceil-div can overrun; clamp every global row read
        # (no bounds check). Output still gated by store_valid (unclamped row).
        last_valid_row_i32 = (
            total_rows_i32 - c1_i32 if persist_m > 1 else None
        )

        def _clamp_row(row_i32):
            if persist_m > 1:
                return arith.minui(row_i32, last_valid_row_i32)
            return row_i32

        # Row-block geometry, parameterised by ``bidx_eff`` (bid_x for
        # persist_m==1; bid_x*persist_m + p otherwise). R[b] is not touched here.
        def _row_setup(bidx_eff_i32):
            block_id_i32 = (
                bidx_eff_i32 * arith.constant(W, type=i32) + wave_i32 if W > 1
                else bidx_eff_i32
            )
            m_base_i32 = block_id_i32 * c_blksz_i32
            src_row_for_mfma = [
                _clamp_row(
                    m_base_i32 + arith.constant(m_tile * 16, type=i32)
                    + lane_mod_16
                )
                for m_tile in range_constexpr(NUM_M_TILES)
            ]
            src_row_store = m_base_i32 + lane_i32
            store_valid = arith.cmpi(
                CmpIPredicate.ult, src_row_store, total_rows_i32,
            )
            return m_base_i32, src_row_for_mfma, src_row_store, store_valid

        # LDS views. R[b] only on the cooperative-LDS path; Y_LDS only on the
        # (disabled) non-DPP path.
        base_ptr = allocator.get_base()
        if not preload_b:
            R_lds_view = SmemPtr(
                base_ptr, R_lds_offset, rot_elem_ty,
                shape=(R_NUMEL_PER_BLOCK,),
            ).get()
        if not _DPP_TRANSPOSE:
            Y_lds_view = SmemPtr(
                base_ptr, Y_lds_offset, f32,
                shape=(GROUP_QUANT_BLOCK_SIZE * FP4_GROUP_SIZE,),
            ).get()
        if _staged_x:
            X_lds_view = SmemPtr(
                base_ptr, X_lds_offset, in_elem_ty,
                shape=(W * X_TILE_ELEMS,),
            ).get()

        # In-register cross-lane transpose helpers (full 64-lane wave); verified
        # 6-op butterfly sequence (see _xlane_probe.py).
        c_dpp_mask = 0xF
        c_swiz_x4 = arith.constant((4 << 10) | 0x1F, type=i32)
        c_swiz_x8 = arith.constant((8 << 10) | 0x1F, type=i32)
        c_swiz_x20 = arith.constant((20 << 10) | 0x1F, type=i32)
        _no_swizzle = _os.environ.get("SINGLE_ROT_NO_SWIZZLE", "0") == "1"

        def _xshuf(v, d):
            if d == 1:
                return rocdl.update_dpp(i32, v, v, 0xB1, c_dpp_mask,
                                        c_dpp_mask, False)
            if d == 2:
                return rocdl.update_dpp(i32, v, v, 0x4E, c_dpp_mask,
                                        c_dpp_mask, False)
            if d in (4, 8, 20) and not _no_swizzle:
                return rocdl.ds_swizzle(
                    i32, v, {4: c_swiz_x4, 8: c_swiz_x8, 20: c_swiz_x20}[d])
            idx = ((lane_i32 ^ arith.constant(d, type=i32)) & c63_i32) * c4_i32
            return rocdl.ds_bpermute(i32, idx, v)

        def _lane_bit_eq0(lb):
            bit = (lane_i32 >> arith.constant(lb, type=i32)) & c1_i32
            return arith.cmpi(CmpIPredicate.eq, bit, c0_i32)

        def _lane_reg_swap(regs, lb, rb):
            cond0 = _lane_bit_eq0(lb)
            out = list(regs)
            d = 1 << lb
            for a in range_constexpr(16):
                if (a >> rb) & 1:
                    continue
                bb = a | (1 << rb)
                sa = _xshuf(regs[a], d)
                sb = _xshuf(regs[bb], d)
                out[a] = arith.select(cond0, regs[a], sb)
                out[bb] = arith.select(cond0, sa, regs[bb])
            return out

        def _lane_lane_swap(regs, lb1, lb2):
            b1 = (lane_i32 >> arith.constant(lb1, type=i32)) & c1_i32
            b2 = (lane_i32 >> arith.constant(lb2, type=i32)) & c1_i32
            cond_diff = arith.cmpi(CmpIPredicate.ne, b1, b2)
            d = (1 << lb1) | (1 << lb2)
            out = []
            for a in range_constexpr(16):
                sh = _xshuf(regs[a], d)
                out.append(arith.select(cond_diff, sh, regs[a]))
            return out

        def _transpose_c_tile(acc_by_mt):
            regs = [None] * 16
            for mt in range_constexpr(NUM_M_TILES):
                for r in range_constexpr(WMMA_FRAG_VALS):
                    regs[mt * 4 + r] = ArithValue(acc_by_mt[mt][r]).bitcast(i32)
            regs = _lane_reg_swap(regs, 0, 0)
            regs = _lane_reg_swap(regs, 1, 1)
            regs = _lane_reg_swap(regs, 2, 2)
            regs = _lane_lane_swap(regs, 2, 4)
            regs = _lane_reg_swap(regs, 3, 3)
            regs = _lane_lane_swap(regs, 3, 5)
            return [ArithValue(regs[j]).bitcast(f32)
                    for j in range_constexpr(16)]

        # B-fragment / R[b] load helpers, parameterised by block ``b`` (each has
        # a distinct R[b], so every global base carries a ``b * R_NUMEL`` offset).
        def _bfrag_n_k(n_tile, k_tile):
            n_lds = arith.constant(n_tile * 16, type=i32) + lane_mod_16
            k_lds = arith.constant(k_tile * 16, type=i32) + k_off_in_tile
            return n_lds, k_lds

        def _load_bfrag_preshuffled(b_i32, n_tile, k_tile):
            # R[b] pre-shuffled into B-frag order: lane L's 4-bf16 frag is
            # contiguous at b*R_NUMEL + ((n_tile*NUM_K_TILES+k_tile)*64+L)*4.
            base_b = (
                b_i32 * c_R_block_elems_i32
                + arith.constant(
                    (n_tile * NUM_K_TILES + k_tile) * 64 * WMMA_FRAG_VALS,
                    type=i32,
                )
            )
            off_b = base_b + lane_i32 * arith.constant(WMMA_FRAG_VALS, type=i32)
            dw_b = off_b >> c1_i32
            raw_b = buffer_ops.buffer_load(
                rot_rsrc, dw_b, vec_width=WMMA_FRAG_VALS // 2, dtype=i32,
            )
            return vector.bitcast(T.vec(WMMA_FRAG_VALS, rot_elem_ty), raw_b)

        def _load_bfrag_preload_nt(b_i32, n_tile, k_tile):
            # Non-transposed, un-shuffled R[b] HBM->VGPR: each lane loads its 4
            # contiguous-K B-frags at b*R_NUMEL + n_lds*32 + k_lds.
            n_lds, k_lds = _bfrag_n_k(n_tile, k_tile)
            flat_b = (
                b_i32 * c_R_block_elems_i32 + n_lds * group_size_i32 + k_lds
            )
            dw_b = flat_b >> c1_i32
            raw_b = buffer_ops.buffer_load(
                rot_rsrc, dw_b, vec_width=WMMA_FRAG_VALS // 2, dtype=i32,
            )
            return vector.bitcast(T.vec(WMMA_FRAG_VALS, rot_elem_ty), raw_b)

        def _load_bfrag_preload_t(b_i32, n_tile, k_tile):
            # Transposed, un-shuffled R[b] HBM->VGPR: b_frag is column n_lds
            # (rows k_lds..+3) -> stride-32 gather at b*R_NUMEL+(k_lds+i)*32+n_lds.
            c5_i32 = arith.constant(5, type=i32)
            c16_i32 = arith.constant(16, type=i32)
            n_lds, k_lds = _bfrag_n_k(n_tile, k_tile)
            elems = []
            for i in range_constexpr(WMMA_FRAG_VALS):
                flat_i = (
                    b_i32 * c_R_block_elems_i32
                    + (((k_lds + arith.constant(i, type=i32)) << c5_i32)
                       + n_lds)
                )
                dw_i = flat_i >> c1_i32
                raw_i = buffer_ops.buffer_load(
                    rot_rsrc, dw_i, vec_width=1, dtype=i32,
                )
                lo16 = arith.trunci(i16, raw_i)
                hi16 = arith.trunci(i16, raw_i >> c16_i32)
                is_odd = arith.cmpi(CmpIPredicate.ne, flat_i & c1_i32, c0_i32)
                sel16 = arith.select(is_odd, hi16, lo16)
                elems.append(ArithValue(sel16).bitcast(rot_elem_ty))
            return vector.from_elements(
                T.vec(WMMA_FRAG_VALS, rot_elem_ty), elems
            )

        def _load_bfrag(b_i32, n_tile, k_tile):
            if rot_preshuffled:
                return _load_bfrag_preshuffled(b_i32, n_tile, k_tile)
            if not rot_transposed:
                return _load_bfrag_preload_nt(b_i32, n_tile, k_tile)
            return _load_bfrag_preload_t(b_i32, n_tile, k_tile)

        def _load_R_to_lds(b_i32):
            # Cooperative R[b] -> LDS (preload_b == False): non-transposed via
            # direct DMA, transposed via VGPR scatter. Base = b*R_BLOCK_BYTES.
            R_b_elem_base = b_i32 * c_R_block_elems_i32
            if not rot_transposed:
                c64_i32 = arith.constant(64, type=i32)
                c_dma_i32 = arith.constant(R_DMA_BYTES, type=i32)
                c0_i32_dma = arith.constant(0, type=i32)
                c_R_block_bytes_i32 = arith.constant(R_BLOCK_BYTES, type=i32)
                lds_ptr_ty = ir.Type.parse("!llvm.ptr<3>")
                R_base_idx = _memref.extract_aligned_pointer_as_index(
                    R_lds_view
                )
                wave_idx = arith.index_cast(T.index, wave_i32)
                c_wave_bytes_idx = arith.constant(
                    64 * R_DMA_BYTES, type=T.index
                )
                b_byte_base_i32 = b_i32 * c_R_block_bytes_i32

                def _emit_r_dma():
                    for li in range_constexpr(R_DMA_LOADS):
                        lds_addr_idx = (
                            R_base_idx
                            + wave_idx * c_wave_bytes_idx
                            + arith.constant(
                                li * BLOCK_DIM * R_DMA_BYTES, type=T.index
                            )
                        )
                        lds_ptr_i64 = rocdl.readfirstlane(
                            T.i64, arith.index_cast(T.i64, lds_addr_idx)
                        )
                        lds_ptr = _llvm.inttoptr(lds_ptr_ty, lds_ptr_i64)
                        glob_slot_i32 = (
                            arith.constant(li * BLOCK_DIM, type=i32)
                            + wave_i32 * c64_i32
                            + lane_i32
                        )
                        glob_off_i32 = (
                            b_byte_base_i32 + glob_slot_i32 * c_dma_i32
                        )
                        rocdl.raw_ptr_buffer_load_lds(
                            rot_rsrc, lds_ptr, c_dma_i32, glob_off_i32,
                            c0_i32_dma, c0_i32_dma, c0_i32_dma,
                        )

                if W > R_DMA_ACTIVE_WAVES:
                    active_cond = arith.cmpi(
                        CmpIPredicate.ult, wave_i32,
                        arith.constant(R_DMA_ACTIVE_WAVES, type=i32),
                    )
                    _if_dma = _scf.IfOp(active_cond)
                    with ir.InsertionPoint(_if_dma.then_block):
                        _emit_r_dma()
                        _scf.YieldOp([])
                else:
                    _emit_r_dma()
                _waitcnt0_barrier()
            else:
                thread_elem_off = tid_i32 * c_R_elems_per_thread_i32
                for li in range_constexpr(R_LOAD_STEPS):
                    step_elem_off = (
                        R_b_elem_base + thread_elem_off
                        + arith.constant(li * R_LOAD_VEC, type=i32)
                    )
                    dw_off_li = step_elem_off >> c1_i32
                    raw_iv = buffer_ops.buffer_load(
                        rot_rsrc, dw_off_li, vec_width=R_LOAD_VEC_I32,
                        dtype=i32,
                    )
                    r_vec_bf = vector.bitcast(
                        T.vec(R_LOAD_VEC, rot_elem_ty), raw_iv
                    )
                    flat_base_i32 = thread_elem_off + arith.constant(
                        li * R_LOAD_VEC, type=i32
                    )
                    g_src_i32 = flat_base_i32 >> arith.constant(5, type=i32)
                    h_src_base_i32 = flat_base_i32 & arith.constant(
                        31, type=i32
                    )
                    dest_base_i32 = h_src_base_i32 * group_size_i32 + g_src_i32
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

        # MFMA hot-region scheduling hints (gated by sched_hints). Pure perf.
        def _emit_mfma_schedule():
            if not _sched_hints:
                return
            n_mfma = NUM_M_TILES * NUM_N_TILES * NUM_K_TILES   # 16 v_mfma
            n_dsrd = max(1, (NUM_M_TILES * NUM_K_TILES) // 2)  # 4
            # Deferred preshuffled R[b] B-frag loads live here only when persist
            # does not hoist them (persist_m == 1).
            n_vmem = (
                (NUM_N_TILES * NUM_K_TILES)
                if (rot_preshuffled and persist_m == 1) else 0
            )
            if n_vmem > 0:
                rocdl.sched_vmem(n_vmem)
            per = max(1, n_mfma // n_dsrd)
            emitted = 0
            for _i in range_constexpr(n_dsrd):
                rocdl.sched_dsrd(1)
                rocdl.sched_mfma(per)
                emitted += per
            if n_mfma - emitted > 0:
                rocdl.sched_mfma(n_mfma - emitted)
            rocdl.sched_barrier(0)

        # Persist hoist: R[b] is invariant across persist groups, so preload_b
        # caches B-frags in VGPRs once here (LDS path can't hoist, reloads).
        b_frag_by_kk = None
        if persist_m > 1 and preload_b:
            b_frag_by_kk = []
            for kk in range_constexpr(K):
                _b_i32 = b_base_i32 + arith.constant(kk, type=i32)
                _regs = {}
                for n_tile in range_constexpr(NUM_N_TILES):
                    for k_tile in range_constexpr(NUM_K_TILES):
                        _regs[(n_tile, k_tile)] = _load_bfrag(
                            _b_i32, n_tile, k_tile
                        )
                b_frag_by_kk.append(_regs)

        # Persistent row-block loop: WG processes persist_m consecutive groups
        # (bidx_eff = bid_x*persist_m + p). persist_m==1 emits no loop.
        if persist_m > 1:
            _c0_p = arith.constant(0, type=T.index)
            _c1_p = arith.constant(1, type=T.index)
            _c_pm_p = arith.constant(persist_m, type=T.index)
            _for_persist = _scf.ForOp(_c0_p, _c_pm_p, _c1_p)
            _persist_ip = ir.InsertionPoint(_for_persist.body)
            _persist_ip.__enter__()
            _pi_i32 = arith.index_cast(i32, _for_persist.induction_variable)
            bidx_eff_i32 = (
                bid_x_i32 * arith.constant(persist_m, type=i32) + _pi_i32
            )
        else:
            bidx_eff_i32 = bid_x_i32

        m_base_i32, src_row_for_mfma, src_row_store, store_valid = (
            _row_setup(bidx_eff_i32)
        )

        # Per-block loop: WG owns blocks [b_base, b_base+K), each with its own
        # R[b] (LDS path reloads, preload_b keeps B-frags in VGPRs; K>1 serializes).
        for kk in range_constexpr(K):
            b_i32 = b_base_i32 + arith.constant(kk, type=i32)

            # Stage 1: this block's R[b] -> B-fragments (VGPR) or LDS.
            if preload_b:
                if b_frag_by_kk is not None:
                    b_frag_regs = b_frag_by_kk[kk]
                else:
                    b_frag_regs = {}
                    if not rot_preshuffled:
                        # Un-shuffled preload: load all 4 frags before the MFMA.
                        for n_tile in range_constexpr(NUM_N_TILES):
                            for k_tile in range_constexpr(NUM_K_TILES):
                                b_frag_regs[(n_tile, k_tile)] = _load_bfrag(
                                    b_i32, n_tile, k_tile
                                )
                    # Pre-shuffled + persist_m==1: defer to first MFMA use.
            else:
                # Cooperative LDS path: barrier before overwriting shared R-LDS
                # when a slower wave may still read the previous copy.
                if (K > 1 and kk > 0) or persist_m > 1:
                    gpu.barrier()
                _load_R_to_lds(b_i32)

            # Stage X: coalesced HBM->LDS DMA of this block's 64x32 tile (16 B/
            # lane -> row=li*16+lane//4, col=(lane%4)*8; LDS row-major).
            if _staged_x:
                x_lds_ptr_ty = ir.Type.parse("!llvm.ptr<3>")
                X_base_idx = _memref.extract_aligned_pointer_as_index(X_lds_view)
                wave_idx_x = arith.index_cast(T.index, wave_i32)
                c_wave_x_bytes = arith.constant(X_LDS_BYTES, type=T.index)
                c_xdma_bytes = arith.constant(X_DMA_BYTES, type=i32)
                c2b_i32 = arith.constant(2, type=i32)
                c3_i32 = arith.constant(3, type=i32)
                c8_i32 = arith.constant(8, type=i32)
                c16_i32 = arith.constant(16, type=i32)
                c_xkmask_i32 = arith.constant(_X_K_BLOCKS16 - 1, type=i32)
                lane_div4 = lane_i32 >> c2_i32
                col_in_tile = (lane_i32 & c3_i32) * c8_i32
                for li in range_constexpr(X_DMA_LOADS):
                    lds_addr_idx = (
                        X_base_idx
                        + wave_idx_x * c_wave_x_bytes
                        + arith.constant(li * 64 * X_DMA_BYTES, type=T.index)
                    )
                    lds_ptr_i64 = rocdl.readfirstlane(
                        T.i64, arith.index_cast(T.i64, lds_addr_idx)
                    )
                    lds_ptr = _llvm.inttoptr(x_lds_ptr_ty, lds_ptr_i64)
                    row_in_tile = arith.constant(li * 16, type=i32) + lane_div4
                    if _swizzle_x:
                        # XOR16 swizzle: permute this lane's 16 B K-chunk by
                        # ((row & 3)*16) bytes (col in elems -> bytes, XOR, back).
                        col_phys_bytes = col_in_tile * c2b_i32
                        swz_amt = (row_in_tile & c_xkmask_i32) * c16_i32
                        col_sw_elem = (col_phys_bytes ^ swz_amt) >> c1_i32
                    else:
                        col_sw_elem = col_in_tile
                    # Clamp global row (persist_m>1) to avoid reading past input.
                    glob_row = _clamp_row(m_base_i32 + row_in_tile)
                    glob_elem = (
                        glob_row * cols_i32
                        + b_i32 * group_size_i32
                        + col_sw_elem
                    )
                    glob_byte = glob_elem * c2b_i32
                    rocdl.raw_ptr_buffer_load_lds(
                        in_rsrc, lds_ptr, c_xdma_bytes, glob_byte,
                        c0_i32, c0_i32, c0_i32,
                    )
                # Per-wave tile: waitcnt-only suffices (no workgroup barrier).
                _waitcnt0()

            # Stage 2: MFMA loop -- like the no-sort variant but the input row
            # uses the natural ``src_row_for_mfma[m_tile]`` resolved above.
            if use_ptr64:
                b_idx = arith.index_cast(T.index, b_i32)
            # acc_tiles[(mt, nt)] = 4 extracted f32 for the DPP transpose.
            acc_tiles = {}
            # Fence the top of the schedulable MFMA region (paired with the
            # tail hint) so hints only reorder the ds_read/MFMA stream.
            if _sched_hints:
                rocdl.sched_barrier(0)
            for m_tile in range_constexpr(NUM_M_TILES):
                if use_ptr64:
                    # 64-bit element offset of this row's group base.
                    src_row_idx = arith.index_cast(
                        T.index, src_row_for_mfma[m_tile]
                    )
                    row_base_elem_idx = (
                        src_row_idx * cols_idx + b_idx * group_size_idx
                    )
                else:
                    # 32-bit element offset (cheap buffer load).
                    row_off_in_inp = (
                        src_row_for_mfma[m_tile] * cols_i32
                        + b_i32 * group_size_i32
                    )

                for n_tile in range_constexpr(NUM_N_TILES):
                    acc = vector.from_elements(
                        T.vec(WMMA_FRAG_VALS, f32),
                        [c0_f32] * WMMA_FRAG_VALS,
                    )

                    for k_tile in range_constexpr(NUM_K_TILES):
                        k_col_i32 = (
                            arith.constant(k_tile * 16, type=i32)
                            + k_off_in_tile
                        )
                        if _staged_x:
                            # A-fragment from the staged LDS tile: row =
                            # m_tile*16 + lane%16, col = k_col (row-major).
                            x_row_i32 = (
                                arith.constant(m_tile * 16, type=i32)
                                + lane_mod_16
                            )
                            if _swizzle_x:
                                # Mirror the store-side XOR16 swizzle; byte-XOR
                                # collapses to element-XOR by ((row & 3)*8).
                                x_k_col = k_col_i32 ^ (
                                    (x_row_i32
                                     & arith.constant(
                                         _X_K_BLOCKS16 - 1, type=i32))
                                    * arith.constant(8, type=i32)
                                )
                            else:
                                x_k_col = k_col_i32
                            x_lds_off = (
                                arith.index_cast(T.index, wave_i32)
                                * arith.constant(X_TILE_ELEMS, type=T.index)
                                + arith.index_cast(
                                    T.index,
                                    x_row_i32 * group_size_i32 + x_k_col,
                                )
                            )
                            a_frag_bf = vector.load_op(
                                T.vec(WMMA_FRAG_VALS, in_elem_ty),
                                X_lds_view, [x_lds_off],
                            )
                        else:
                            if use_ptr64:
                                # 64-bit byte offset GEP from i64 base + load.
                                elem_off_a_idx = (
                                    row_base_elem_idx
                                    + arith.index_cast(T.index, k_col_i32)
                                )
                                byte_off_a = arith.index_cast(
                                    T.i64, elem_off_a_idx * c_in_elem_bytes_idx
                                )
                                a_ptr = _llvm.GEPOp(
                                    _ptr_ty_as1,
                                    inp_base_ptr,
                                    [byte_off_a],
                                    [-2147483648],
                                    T.i8,
                                    _llvm.GEPNoWrapFlags.none,
                                ).result
                                raw_a = _llvm.LoadOp(
                                    T.vec(2, i32), a_ptr, alignment=8
                                ).result
                            else:
                                elem_off_a = row_off_in_inp + k_col_i32
                                dw_off_a = elem_off_a >> c1_i32
                                raw_a = buffer_ops.buffer_load(
                                    in_rsrc, dw_off_a, vec_width=2, dtype=i32
                                )
                            a_frag_bf = vector.bitcast(
                                T.vec(WMMA_FRAG_VALS, in_elem_ty), raw_a
                            )

                        if preload_b:
                            if (
                                rot_preshuffled
                                and (n_tile, k_tile) not in b_frag_regs
                            ):
                                # Deferred R[b] load at first use (m_tile==0).
                                b_frag_regs[(n_tile, k_tile)] = _load_bfrag(
                                    b_i32, n_tile, k_tile
                                )
                            b_frag_bf = b_frag_regs[(n_tile, k_tile)]
                        else:
                            n_lds = (
                                arith.constant(n_tile * 16, type=i32)
                                + lane_mod_16
                            )
                            k_lds = (
                                arith.constant(k_tile * 16, type=i32)
                                + k_off_in_tile
                            )
                            lds_off_b = n_lds * group_size_i32 + k_lds
                            lds_off_b_idx = arith.index_cast(T.index, lds_off_b)
                            b_frag_bf = vector.load_op(
                                T.vec(WMMA_FRAG_VALS, rot_elem_ty),
                                R_lds_view,
                                [lds_off_b_idx],
                            )

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
                        else:
                            acc = rocdl.mfma_f32_16x16x16f16(
                                T.vec(WMMA_FRAG_VALS, f32),
                                [a_frag_bf, b_frag_bf, acc, 0, 0, 0],
                            )

                    cur = []
                    for i in range_constexpr(WMMA_FRAG_VALS):
                        val = vector.extract(
                            acc, static_position=[i], dynamic_position=[]
                        )
                        cur.append(val)
                        if not _DPP_TRANSPOSE:
                            m_local_base = (
                                arith.constant(m_tile * 16, type=i32)
                                + k_off_in_tile
                            )
                            n_local_i32 = (
                                arith.constant(n_tile * 16, type=i32)
                                + lane_mod_16
                            )
                            m_local_i32 = (
                                m_local_base + arith.constant(i, type=i32)
                            )
                            y_lds_off = (
                                m_local_i32 * group_size_i32 + n_local_i32
                            )
                            y_lds_off_idx = arith.index_cast(
                                T.index, y_lds_off
                            )
                            _memref.store(val, Y_lds_view, [y_lds_off_idx])
                    acc_tiles[(m_tile, n_tile)] = cur

            # Tail scheduling hints for the MFMA hot region (no-op unless
            # sched_hints). Closes with sched_barrier(0).
            _emit_mfma_schedule()

            # Stage 3: build y_vals[0..31] = row (m_base+lane)'s 32 values via
            # in-register cross-lane transpose (DPP path: no Y_LDS, no barrier).
            y_vals = [None] * FP4_GROUP_SIZE
            if _DPP_TRANSPOSE:
                for n_tile in range_constexpr(NUM_N_TILES):
                    acc_by_mt = [
                        acc_tiles[(m_tile, n_tile)]
                        for m_tile in range_constexpr(NUM_M_TILES)
                    ]
                    col_vals = _transpose_c_tile(acc_by_mt)
                    for j in range_constexpr(16):
                        y_vals[n_tile * 16 + j] = col_vals[j]
            else:
                gpu.barrier()
                tid_row_base = lane_i32 * group_size_i32
                for k in range_constexpr(Y_READ_STEPS):
                    lds_off = tid_row_base + arith.constant(
                        k * Y_READ_VEC, type=i32
                    )
                    lds_off_idx = arith.index_cast(T.index, lds_off)
                    v4 = vector.load_op(
                        T.vec(Y_READ_VEC, f32), Y_lds_view, [lds_off_idx],
                    )
                    for j in range_constexpr(Y_READ_VEC):
                        y_vals[k * Y_READ_VEC + j] = vector.extract(
                            v4, static_position=[j], dynamic_position=[],
                        )

            abs_max = c0_f32
            for i in range_constexpr(FP4_GROUP_SIZE):
                abs_v = _llvm.call_intrinsic(
                    f32, "llvm.fabs.f32", [y_vals[i]], [], [],
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

            # cvt to fp4
            out_dwords = []
            for dw in range_constexpr(DWORDS_PER_GROUP):
                packed = c0_i32
                for sel in range_constexpr(4):
                    idx = dw * 4 + sel
                    e0 = y_vals[idx * 2]
                    e1 = y_vals[idx * 2 + 1]
                    packed = rocdl.cvt_scalef32_pk_fp4_f32(
                        i32, packed, e0, e1, inv_scale, sel,
                    )
                out_dwords.append(packed)
            out_vec = vector.from_elements(
                T.vec(DWORDS_PER_GROUP, i32), out_dwords,
            )

            # Stage 4: store at natural source-row addresses; out fp4x2 row =
            # src_row, out_scale row-major [src_row, b]. Scale-sort downstream.
            scale_off_i32 = src_row_store * scale_N_i32 + b_i32

            out_byte_off_i32 = (
                src_row_store
                    * arith.constant(scale_N * DWORDS_PER_GROUP * 4, type=i32)
                + b_i32 * out_bytes_per_group_i32
            )

            _if_store = _scf.IfOp(store_valid)
            with ir.InsertionPoint(_if_store.then_block):
                buffer_ops.buffer_store(
                    out_vec, out_rsrc, out_byte_off_i32,
                    offset_is_bytes=True,
                )
                buffer_ops.buffer_store(
                    e8m0_byte_i8, out_scale_rsrc, scale_off_i32,
                    offset_is_bytes=True,
                )
                _scf.YieldOp([])

        if persist_m > 1:
            # DPP transpose keeps Y in VGPRs (no cross-group barrier); the
            # non-DPP path would need one here.
            if not _DPP_TRANSPOSE:
                gpu.barrier()
            _scf.YieldOp([])
            _persist_ip.__exit__(None, None, None)

    @flyc.jit
    def launch_per_1x32_fp4_quant_block_rotation_mfma_moe_sorting(
        inp: fx.Tensor,
        rot_R: fx.Tensor,
        out: fx.Tensor,
        out_scale: fx.Tensor,
        num_row_blocks: Int32,
        num_tokens: Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        allocator.finalized = False
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            allocator.finalize()

        num_row_blocks_idx = arith.index_cast(T.index, num_row_blocks)
        # Each WG covers K consecutive ``b`` blocks -> grid.y = scale_N // K.
        scale_N_idx = arith.constant(scale_N // K, type=T.index)

        # grid.x = ceil(num_row_blocks / W) then ceil-div by persist_m.
        if W > 1:
            gx_i32 = (
                ArithValue(num_row_blocks)
                + arith.constant(W - 1, type=T.i32)
            ) >> arith.constant(LOG2_W, type=T.i32)
        else:
            gx_i32 = ArithValue(num_row_blocks)
        if persist_m > 1:
            gx_i32 = (
                gx_i32 + arith.constant(persist_m - 1, type=T.i32)
            ) // arith.constant(persist_m, type=T.i32)
        grid_x_idx = arith.index_cast(T.index, gx_i32)

        launcher = per_1x32_fp4_quant_block_rotation_mfma_moe_sorting_kernel(
            inp, rot_R, out, out_scale, num_tokens,
        )
        launcher.launch(
            grid=(grid_x_idx, scale_N_idx, 1),
            block=(BLOCK_DIM, 1, 1),
            stream=stream,
        )

    return launch_per_1x32_fp4_quant_block_rotation_mfma_moe_sorting


# build_per_1x32_fp4_quant_block_single_rotation_mfma_moe_sorting_module: fused
# rotation+quant+sort, ONE shared 32x32 R loaded once per WG, reused over K blocks.


@functools.lru_cache(maxsize=None)
def build_per_1x32_fp4_quant_block_single_rotation_mfma_moe_sorting_module(
    cols: int,
    in_dtype: str = "bf16",
    rot_dtype: str = "bf16",
    topk: int = 1,
    rot_transposed: bool = False,
    use_ptr64: bool = False,
    blocks_per_wg: int = 1,
    waves_per_wg: int = 1,
    preload_b: bool = True,
    stage_x_lds: bool = True,
    rot_preshuffled: bool = True,
    swizzle_x_lds: bool = True,
    sched_hints: bool = True,
    persist_m: int = 1,
    xcd_remap: bool = False,
):
    """Single-shared-R variant of
    :func:`build_per_1x32_fp4_quant_block_rotation_mfma_moe_sorting_module`.

    Same semantics/output, but ``rot_R`` is one ``(32, 32)`` matrix shared by
    all blocks, with each WG processing ``blocks_per_wg`` blocks via one LDS R.

    Launcher (unchanged)::

        launch(inp, rot_R, sorted_ids, num_valid_ids,
               out, out_scale, num_sorted_blocks, num_tokens, stream)

    Parameters
    ----------
    blocks_per_wg : int, default 1
        Consecutive blocks per WG; must divide scale_N. K>1 shrinks grid y to
        scale_N//K and amortizes the shared R-load. LDS is K-independent.
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
            "MFMA+sort path requires bf16/fp16 R; for fp32 R, use the "
            "two-kernel chain (block_rotation + mxfp4_moe_sort)."
        )
    if rot_dtype not in ("bf16", "fp16", "f16"):
        raise ValueError(f"unsupported R dtype: {rot_dtype!r}")
    _norm_in = "bf16" if in_dtype == "bf16" else "fp16"
    _norm_rot = "bf16" if rot_dtype == "bf16" else "fp16"
    if _norm_in != _norm_rot:
        raise ValueError(
            f"MFMA+sort path requires in_dtype == rot_dtype (got "
            f"in_dtype={in_dtype!r}, rot_dtype={rot_dtype!r})"
        )
    if topk < 1:
        raise ValueError(f"topk must be >= 1, got {topk}")

    PAIRS_PER_GROUP = FP4_GROUP_SIZE // 2       # 16
    DWORDS_PER_GROUP = PAIRS_PER_GROUP // 4     # 4
    scale_N = cols // FP4_GROUP_SIZE
    g = FP4_GROUP_SIZE                          # 32

    if not isinstance(blocks_per_wg, int) or blocks_per_wg < 1:
        raise ValueError(
            f"blocks_per_wg must be a positive int, got {blocks_per_wg!r}"
        )
    if scale_N % blocks_per_wg != 0:
        raise ValueError(
            f"blocks_per_wg={blocks_per_wg} must divide scale_N={scale_N} "
            f"(cols={cols})"
        )
    K = blocks_per_wg

    # Persistent kernel factor: WG walks persist_m row-block groups (grid-x
    # shrinks), reusing shared R; persist_m=1 is the non-persistent kernel.
    if not isinstance(persist_m, int) or persist_m < 1:
        raise ValueError(
            f"persist_m must be a positive int, got {persist_m!r}"
        )

    # XCD remap: MI300 has 8 XCDs with private L2. HW round-robins WGs across
    # them; xcd_remap=True inverts that so a contiguous run shares one XCD.
    NUM_XCD = 8

    # In-register cross-lane transpose of the MFMA C tile, replacing the 8 KB
    # Y_LDS round-trip + barrier (LDS drops to R-only ~2 KB).
    import os as _os
    _DPP_TRANSPOSE = True # _os.environ.get("SINGLE_ROT_DPP_TRANSPOSE") == "1"

    # Stage X through LDS (coalesced HBM->LDS DMA, then LDS->VGPR) to avoid
    # strided direct-HBM A-frag loads. Only valid when not use_ptr64.
    _staged_x = bool(stage_x_lds) and not use_ptr64

    # XOR16 bank-conflict swizzle on the staged-X tile: XOR the K-byte offset by
    # ((row & 3)*16), applied symmetrically on store/read so content is unchanged.
    _swizzle_x = bool(swizzle_x_lds) and _staged_x
    # X row is FP4_GROUP_SIZE bf16 = 64 B -> 4 chunks of 16 B.
    _X_K_BLOCKS16 = (FP4_GROUP_SIZE * 2) // 16   # 4

    # Optional software-pipeline hints (sched_group_barrier) on the MFMA hot
    # region: interleave ds_read with MFMA and fence Stage-3/4. Largely A/B.
    _sched_hints = bool(sched_hints)

    # Pre-shuffled R: host permuted R into MFMA B-frag order so the kernel reads
    # it with one coalesced load. Implies preload_b; rot_transposed absorbed.
    if rot_preshuffled and not preload_b:
        raise ValueError("rot_preshuffled requires preload_b=True")

    # Multi-wave WG: W waves of 64 lanes, each owns a 64-row block, all sharing
    # one LDS copy of rot_R. Amortizes the R load; needs the DPP transpose.
    W = waves_per_wg
    # Cap at 8 waves (512-thread WG): each of 64*W threads needs >= 2 of the
    # 1024 bf16 elems for the cooperative R load.
    if (
        not isinstance(W, int)
        or W < 1
        or (W & (W - 1)) != 0
        or (GROUP_QUANT_BLOCK_SIZE * W) > 512
    ):
        raise ValueError(
            f"waves_per_wg must be a power of two in "
            f"[1, {512 // GROUP_QUANT_BLOCK_SIZE}], got {waves_per_wg!r}"
        )
    LOG2_W = W.bit_length() - 1
    if W > 1 and not _DPP_TRANSPOSE:
        raise ValueError(
            "waves_per_wg > 1 requires the in-register DPP transpose "
            "(_DPP_TRANSPOSE) since waves cannot share the Y_LDS tile."
        )
    BLOCK_DIM = GROUP_QUANT_BLOCK_SIZE * W

    M_TILE = 16
    N_TILE = 16
    K_TILE = 16
    NUM_M_TILES = GROUP_QUANT_BLOCK_SIZE // M_TILE   # 4
    NUM_N_TILES = FP4_GROUP_SIZE // N_TILE           # 2
    NUM_K_TILES = FP4_GROUP_SIZE // K_TILE           # 2
    WMMA_FRAG_VALS = 4

    R_NUMEL_PER_BLOCK = g * g                        # 1024 (single shared R)
    R_LDS_BYTES = R_NUMEL_PER_BLOCK * 2              # 2 KB (bf16/fp16)
    Y_LDS_BYTES = GROUP_QUANT_BLOCK_SIZE * FP4_GROUP_SIZE * 4  # 8 KB f32

    # Cooperative R load uses all BLOCK_DIM = 64*W threads; each loads
    # R_NUMEL/BLOCK_DIM contiguous bf16 elems, vectorized at R_LOAD_VEC.
    R_ELEMS_PER_THREAD = R_NUMEL_PER_BLOCK // BLOCK_DIM  # 16 // W
    R_LOAD_VEC = min(8, R_ELEMS_PER_THREAD)              # bf16 per vec
    R_LOAD_STEPS = R_ELEMS_PER_THREAD // R_LOAD_VEC
    R_LOAD_VEC_I32 = R_LOAD_VEC // 2                     # 2 bf16 per i32

    # Direct global->LDS DMA (buffer_load_lds) for non-transposed R: 16 B
    # (dwordx4) per lane, R_DMA_SLOTS chunks; at most 2 waves have work.
    R_DMA_BYTES = 16
    R_DMA_SLOTS = R_LDS_BYTES // R_DMA_BYTES             # 128
    R_DMA_ACTIVE_WAVES = min(W, R_DMA_SLOTS // 64)       # min(W, 2)
    R_DMA_ACTIVE_LANES = R_DMA_ACTIVE_WAVES * 64
    R_DMA_LOADS = (
        (R_DMA_SLOTS + R_DMA_ACTIVE_LANES - 1) // R_DMA_ACTIVE_LANES
    )

    Y_READ_VEC = 4
    Y_READ_STEPS = FP4_GROUP_SIZE // Y_READ_VEC      # 8

    # X-tile LDS staging: one 64x32 tile per wave (row-major), loaded HBM->LDS
    # via buffer_load_lds DMA then read back as MFMA A-fragments.
    X_TILE_ELEMS = GROUP_QUANT_BLOCK_SIZE * FP4_GROUP_SIZE   # 64*32 = 2048
    X_LDS_BYTES = X_TILE_ELEMS * 2                           # 4 KB / wave
    X_DMA_BYTES = 16
    X_DMA_LOADS = X_LDS_BYTES // (64 * X_DMA_BYTES)          # 4 (== NUM_M_TILES)

    gpu_arch = get_hip_arch()
    sym_tag = (
        f"fp4_singrot_mfma_sort_lds_{cols}_{in_dtype}_{rot_dtype}_"
        f"t{topk}_rT{int(rot_transposed)}_k{K}_w{W}_pb{int(preload_b)}"
        f"_sx{int(_staged_x)}_psh{int(rot_preshuffled)}_swz{int(_swizzle_x)}"
        f"_sch{int(_sched_hints)}"
    )
    # R-LDS allocated only on the non-preload path (preload_b loads B-frags
    # straight HBM->VGPR, dropping R-LDS + barrier).
    allocator = SmemAllocator(None, arch=gpu_arch, global_sym_name=sym_tag)
    R_lds_offset = allocator._align(allocator.ptr, 16)
    if not preload_b:
        allocator.ptr = R_lds_offset + R_LDS_BYTES
    Y_lds_offset = allocator._align(allocator.ptr, 16)
    if not _DPP_TRANSPOSE:
        allocator.ptr = Y_lds_offset + Y_LDS_BYTES
    X_lds_offset = allocator._align(allocator.ptr, 16)
    if _staged_x:
        allocator.ptr = X_lds_offset + W * X_LDS_BYTES

    @flyc.kernel
    def per_1x32_fp4_quant_block_single_rotation_mfma_moe_sorting_kernel(
        inp: fx.Tensor,            # (M_src, cols)               bf16 / fp16
        rot_R: fx.Tensor,          # (32, 32)                    bf16 / fp16
        out: fx.Tensor,            # (M_src, cols // 2)          uint8 (fp4x2)
        out_scale: fx.Tensor,      # (M_src_pad, scale_N)        uint8 (e8m0)
        num_tokens_dyn: Int32,
    ):
        from flydsl._mlir.dialects import memref as _memref

        bid_x = fx.block_idx.x  # sorted-row chunk index
        bid_y = fx.block_idx.y  # block-group index in [0, scale_N // K)
        tid = fx.thread_idx.x

        f32 = T.f32
        i32 = T.i32
        i16 = T.i16
        in_elem_ty = _input_elem_mlir_type(in_dtype)
        rot_elem_ty = _input_elem_mlir_type(
            "bf16" if rot_dtype == "bf16" else "fp16"
        )

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
        c_R_elems_per_thread_i32 = arith.constant(R_ELEMS_PER_THREAD, type=i32)
        topk_i32 = arith.constant(topk, type=i32)
        c_K_i32 = arith.constant(K, type=i32)

        in_rsrc = buffer_ops.create_buffer_resource(inp, max_size=True)
        out_rsrc = buffer_ops.create_buffer_resource(out, max_size=True)
        out_scale_rsrc = buffer_ops.create_buffer_resource(
            out_scale, max_size=True
        )
        rot_rsrc = buffer_ops.create_buffer_resource(rot_R, max_size=True)

        if use_ptr64:
            from flydsl._mlir.dialects import fly as _fly

            _ptr_ty_as1 = ir.Type.parse("!llvm.ptr<1>")
            inp_base_ptr = _fly.extract_aligned_pointer_as_index(
                _ptr_ty_as1, inp
            )
            c_in_elem_bytes_idx = arith.constant(2, type=T.index)
            cols_idx = arith.constant(cols, type=T.index)
            group_size_idx = arith.constant(FP4_GROUP_SIZE, type=T.index)

        tid_i32 = ArithValue(tid)
        bid_x_i32 = ArithValue(bid_x)
        bid_y_i32 = ArithValue(bid_y)

        # XCD remap (L2 locality): transpose the HW round-robin (xcd, local)
        # grouping so a contiguous run co-locates on one XCD (<8 tail in place).
        if xcd_remap:
            _idx = T.index
            _c8 = arith.constant(NUM_XCD, type=_idx)
            _grid_x = ArithValue(
                arith.index_cast(_idx, ArithValue(gpu.grid_dim.x))
            )
            _pid = (
                ArithValue(arith.index_cast(_idx, bid_x_i32))
                + ArithValue(arith.index_cast(_idx, bid_y_i32)) * _grid_x
            )
            _grid_mn = _grid_x * arith.constant(scale_N // K, type=_idx)
            _per = _grid_mn / _c8                # blocks per XCD (floor)
            _full = _per * _c8                   # largest mult of 8 <= grid_mn
            _remapped = (_pid % _c8) * _per + (_pid / _c8)
            _logical = ArithValue(
                arith.select(
                    arith.cmpi(CmpIPredicate.ult, _pid, _full),
                    _remapped, _pid,
                )
            )
            bid_x_i32 = ArithValue(arith.index_cast(i32, _logical % _grid_x))
            bid_y_i32 = ArithValue(arith.index_cast(i32, _logical / _grid_x))

        # Multi-wave: split the flat thread id into (wave, lane). Only `lane`
        # (0..63) feeds the lane-based layout math.
        c6_i32 = arith.constant(6, type=i32)
        c63_i32 = arith.constant(63, type=i32)
        lane_i32 = tid_i32 & c63_i32
        wave_i32 = tid_i32 >> c6_i32 if W > 1 else c0_i32

        # This workgroup owns blocks [b_base, b_base + K).
        b_base_i32 = bid_y_i32 * c_K_i32

        num_tokens_i32 = ArithValue(num_tokens_dyn)

        lane_mod_16 = lane_i32 & c15_i32
        lane_div_16 = lane_i32 >> c4_i32
        k_off_in_tile = lane_div_16 << c2_i32   # 0, 4, 8, 12

        # Natural-order traversal: wave owns contiguous source rows; out written
        # at the same row, scale in plain row-major (sort is downstream).
        total_rows_i32 = (
            num_tokens_i32 * topk_i32 if topk > 1 else num_tokens_i32
        )
        # Persist only: grid-x ceil-div can overrun; clamp every global row read
        # (no bounds check). Output still gated by store_valid (unclamped row).
        last_valid_row_i32 = (
            total_rows_i32 - c1_i32 if persist_m > 1 else None
        )

        def _clamp_row(row_i32):
            if persist_m > 1:
                return arith.minui(row_i32, last_valid_row_i32)
            return row_i32

        # Row-block geometry, parameterised by ``bidx_eff`` (bid_x for
        # persist_m==1; bid_x*persist_m + p otherwise). R is loaded once per WG.
        def _row_setup(bidx_eff_i32):
            # Wave w owns sorted-block (bidx_eff*W + w) of 64 rows; m_base is its
            # first sorted_row.
            block_id_i32 = (
                bidx_eff_i32 * arith.constant(W, type=i32) + wave_i32 if W > 1
                else bidx_eff_i32
            )
            m_base_i32 = block_id_i32 * c_blksz_i32
            src_row_for_mfma = []
            for m_tile in range_constexpr(NUM_M_TILES):
                src_row_for_mfma.append(
                    _clamp_row(
                        m_base_i32
                        + arith.constant(m_tile * 16, type=i32)
                        + lane_mod_16
                    )
                )
            # Unclamped store row -> store_valid rejects out-of-range groups.
            src_row_store = m_base_i32 + lane_i32
            store_valid = arith.cmpi(
                CmpIPredicate.ult, src_row_store, total_rows_i32,
            )
            return m_base_i32, src_row_for_mfma, src_row_store, store_valid

        # LDS views.
        base_ptr = allocator.get_base()
        if not preload_b:
            R_lds_view = SmemPtr(
                base_ptr, R_lds_offset, rot_elem_ty,
                shape=(R_NUMEL_PER_BLOCK,),
            ).get()
        if not _DPP_TRANSPOSE:
            Y_lds_view = SmemPtr(
                base_ptr, Y_lds_offset, f32,
                shape=(GROUP_QUANT_BLOCK_SIZE * FP4_GROUP_SIZE,),
            ).get()
        if _staged_x:
            X_lds_view = SmemPtr(
                base_ptr, X_lds_offset, in_elem_ty,
                shape=(W * X_TILE_ELEMS,),
            ).get()

        # In-register cross-lane transpose helpers (full 64-lane wave) on 16 i32
        # SSA values; primitives: ds_bpermute, update_dpp xor1/xor2, ds_swizzle.
        c_dpp_mask = 0xF
        c_swiz_x4 = arith.constant((4 << 10) | 0x1F, type=i32)
        c_swiz_x8 = arith.constant((8 << 10) | 0x1F, type=i32)
        c_swiz_x20 = arith.constant((20 << 10) | 0x1F, type=i32)

        # ds_swizzle disasm crashes the rocprof JSON emitter, so allow forcing
        # ds_bpermute for the within-32 xor stages when capturing an ATT trace.
        _no_swizzle = _os.environ.get("SINGLE_ROT_NO_SWIZZLE") == "1"

        def _xshuf(v, d):
            """Return v held by lane (lane ^ d), full-wave."""
            if d == 1:
                return rocdl.update_dpp(i32, v, v, 0xB1, c_dpp_mask,
                                        c_dpp_mask, False)
            if d == 2:
                return rocdl.update_dpp(i32, v, v, 0x4E, c_dpp_mask,
                                        c_dpp_mask, False)
            if d in (4, 8, 20) and not _no_swizzle:
                return rocdl.ds_swizzle(
                    i32, v, {4: c_swiz_x4, 8: c_swiz_x8, 20: c_swiz_x20}[d])
            # General (crosses 32-lane group) or swizzle disabled: ds_bpermute.
            idx = ((lane_i32 ^ arith.constant(d, type=i32)) & c63_i32) * c4_i32
            return rocdl.ds_bpermute(i32, idx, v)

        def _lane_bit_eq0(lb):
            bit = (lane_i32 >> arith.constant(lb, type=i32)) & c1_i32
            return arith.cmpi(CmpIPredicate.eq, bit, c0_i32)

        def _lane_reg_swap(regs, lb, rb):
            cond0 = _lane_bit_eq0(lb)        # True where lane bit lb == 0
            out = list(regs)
            d = 1 << lb
            for a in range_constexpr(16):
                if (a >> rb) & 1:
                    continue
                b = a | (1 << rb)
                sa = _xshuf(regs[a], d)
                sb = _xshuf(regs[b], d)
                out[a] = arith.select(cond0, regs[a], sb)
                out[b] = arith.select(cond0, sa, regs[b])
            return out

        def _lane_lane_swap(regs, lb1, lb2):
            b1 = (lane_i32 >> arith.constant(lb1, type=i32)) & c1_i32
            b2 = (lane_i32 >> arith.constant(lb2, type=i32)) & c1_i32
            cond_diff = arith.cmpi(CmpIPredicate.ne, b1, b2)
            d = (1 << lb1) | (1 << lb2)
            out = []
            for a in range_constexpr(16):
                sh = _xshuf(regs[a], d)
                out.append(arith.select(cond_diff, sh, regs[a]))
            return out

        def _transpose_c_tile(acc_by_mt):
            """acc_by_mt[mt] = 4 f32 for this n_tile; returns 16 f32 y[j]
            (y[j] at lane t == Y[t][nt*16+j])."""
            regs = [None] * 16
            for mt in range_constexpr(NUM_M_TILES):
                for r in range_constexpr(WMMA_FRAG_VALS):
                    regs[mt * 4 + r] = ArithValue(acc_by_mt[mt][r]).bitcast(i32)
            regs = _lane_reg_swap(regs, 0, 0)
            regs = _lane_reg_swap(regs, 1, 1)
            regs = _lane_reg_swap(regs, 2, 2)
            regs = _lane_lane_swap(regs, 2, 4)
            regs = _lane_reg_swap(regs, 3, 3)
            regs = _lane_lane_swap(regs, 3, 5)
            return [ArithValue(regs[j]).bitcast(f32)
                    for j in range_constexpr(16)]

        # Stage 1: load the single shared R once (reused for all K blocks, no
        # ``b`` offset); preload_b gathers each lane's 4 B-frags from HBM.
        b_frag_regs = {}

        def _bfrag_n_k(n_tile, k_tile):
            n_lds = arith.constant(n_tile * 16, type=i32) + lane_mod_16
            k_lds = arith.constant(k_tile * 16, type=i32) + k_off_in_tile
            return n_lds, k_lds

        def _load_bfrag_preshuffled(n_tile, k_tile):
            # R pre-shuffled into MFMA fragment order: lane L's 4-bf16 B-frag is
            # at flat ((n_tile*NUM_K_TILES+k_tile)*64 + L)*4 -> coalesced read.
            base_b = arith.constant(
                (n_tile * NUM_K_TILES + k_tile) * 64 * WMMA_FRAG_VALS,
                type=i32,
            )
            off_b = base_b + lane_i32 * arith.constant(
                WMMA_FRAG_VALS, type=i32
            )
            dw_b = off_b >> c1_i32
            raw_b = buffer_ops.buffer_load(
                rot_rsrc, dw_b,
                vec_width=WMMA_FRAG_VALS // 2, dtype=i32,
            )
            return vector.bitcast(
                T.vec(WMMA_FRAG_VALS, rot_elem_ty), raw_b
            )

        if rot_preshuffled:
            if persist_m > 1:
                # Persistent kernel: shared R reused across persist groups, so
                # load it once here (before the loop) into b_frag_regs.
                for n_tile in range_constexpr(NUM_N_TILES):
                    for k_tile in range_constexpr(NUM_K_TILES):
                        b_frag_regs[(n_tile, k_tile)] = (
                            _load_bfrag_preshuffled(n_tile, k_tile)
                        )
            else:
                # Defer R's 4 B-frag loads to first use (m_tile==0) so they
                # interleave with the MFMA stream; cached for reuse afterwards.
                pass
        elif not rot_transposed:
            if preload_b:
                # HBM->VGPR: each lane loads its 4 contiguous-K B-frags (no LDS).
                for n_tile in range_constexpr(NUM_N_TILES):
                    for k_tile in range_constexpr(NUM_K_TILES):
                        n_lds, k_lds = _bfrag_n_k(n_tile, k_tile)
                        flat_b = n_lds * group_size_i32 + k_lds
                        dw_b = flat_b >> c1_i32
                        raw_b = buffer_ops.buffer_load(
                            rot_rsrc, dw_b,
                            vec_width=WMMA_FRAG_VALS // 2, dtype=i32,
                        )
                        b_frag_regs[(n_tile, k_tile)] = vector.bitcast(
                            T.vec(WMMA_FRAG_VALS, rot_elem_ty), raw_b
                        )
            else:
                # Direct global->LDS DMA: buffer_load_lds writes 16 B/lane to
                # lds_base + lane*16; per-wave base offset by wave*64*16.
                c64_i32 = arith.constant(64, type=i32)
                c_dma_i32 = arith.constant(R_DMA_BYTES, type=i32)
                c0_i32_dma = arith.constant(0, type=i32)
                lds_ptr_ty = ir.Type.parse("!llvm.ptr<3>")
                R_base_idx = _memref.extract_aligned_pointer_as_index(R_lds_view)
                wave_idx = arith.index_cast(T.index, wave_i32)
                c_wave_bytes_idx = arith.constant(64 * R_DMA_BYTES, type=T.index)

                def _emit_r_dma():
                    for li in range_constexpr(R_DMA_LOADS):
                        # uniform per-wave LDS base
                        lds_addr_idx = (
                            R_base_idx
                            + wave_idx * c_wave_bytes_idx
                            + arith.constant(
                                li * BLOCK_DIM * R_DMA_BYTES, type=T.index
                            )
                        )
                        lds_ptr_i64 = rocdl.readfirstlane(
                            T.i64, arith.index_cast(T.i64, lds_addr_idx)
                        )
                        lds_ptr = _llvm.inttoptr(lds_ptr_ty, lds_ptr_i64)

                        glob_slot_i32 = (
                            arith.constant(li * BLOCK_DIM, type=i32)
                            + wave_i32 * c64_i32
                            + lane_i32
                        )
                        glob_off_i32 = glob_slot_i32 * c_dma_i32
                        rocdl.raw_ptr_buffer_load_lds(
                            rot_rsrc, lds_ptr, c_dma_i32, glob_off_i32,
                            c0_i32_dma, c0_i32_dma, c0_i32_dma,
                        )

                if W > R_DMA_ACTIVE_WAVES:
                    # >2-wave WG: only the first R_DMA_ACTIVE_WAVES waves DMA R.
                    active_cond = arith.cmpi(
                        CmpIPredicate.ult, wave_i32,
                        arith.constant(R_DMA_ACTIVE_WAVES, type=i32),
                    )
                    _if_dma = _scf.IfOp(active_cond)
                    with ir.InsertionPoint(_if_dma.then_block):
                        _emit_r_dma()
                        _scf.YieldOp([])
                else:
                    _emit_r_dma()

                # Wait for the DMA (vmcnt) to land in LDS, then sync the WG.
                _waitcnt0_barrier()
        else:
            if preload_b:
                # Transposed R HBM->VGPR: b_frag is column n_lds (rows k_lds..+3)
                # -> stride-32 gather; load each dword and pick the bf16 half.
                c5_i32 = arith.constant(5, type=i32)
                c16_i32 = arith.constant(16, type=i32)
                for n_tile in range_constexpr(NUM_N_TILES):
                    for k_tile in range_constexpr(NUM_K_TILES):
                        n_lds, k_lds = _bfrag_n_k(n_tile, k_tile)
                        elems = []
                        for i in range_constexpr(WMMA_FRAG_VALS):
                            flat_i = (
                                (k_lds + arith.constant(i, type=i32))
                                << c5_i32
                            ) + n_lds
                            dw_i = flat_i >> c1_i32
                            raw_i = buffer_ops.buffer_load(
                                rot_rsrc, dw_i, vec_width=1, dtype=i32,
                            )
                            # bf16 elem is low/high 16 bits, picked by parity.
                            lo16 = arith.trunci(i16, raw_i)
                            hi16 = arith.trunci(i16, raw_i >> c16_i32)
                            is_odd = arith.cmpi(
                                CmpIPredicate.ne, flat_i & c1_i32, c0_i32
                            )
                            sel16 = arith.select(is_odd, hi16, lo16)
                            elems.append(
                                ArithValue(sel16).bitcast(rot_elem_ty)
                            )
                        b_frag_regs[(n_tile, k_tile)] = vector.from_elements(
                            T.vec(WMMA_FRAG_VALS, rot_elem_ty), elems
                        )
            else:
                # Transposed R: scatter via VGPRs (DMA can't transpose). All
                # BLOCK_DIM threads cooperate; each owns R_ELEMS_PER_THREAD elems.
                thread_elem_off = tid_i32 * c_R_elems_per_thread_i32
                for li in range_constexpr(R_LOAD_STEPS):
                    step_elem_off = thread_elem_off + arith.constant(
                        li * R_LOAD_VEC, type=i32
                    )
                    dw_off_li = step_elem_off >> c1_i32
                    raw_iv = buffer_ops.buffer_load(
                        rot_rsrc, dw_off_li, vec_width=R_LOAD_VEC_I32, dtype=i32
                    )
                    r_vec_bf = vector.bitcast(
                        T.vec(R_LOAD_VEC, rot_elem_ty), raw_iv
                    )
                    flat_base_i32 = thread_elem_off + arith.constant(
                        li * R_LOAD_VEC, type=i32
                    )
                    g_src_i32 = flat_base_i32 >> arith.constant(5, type=i32)
                    h_src_base_i32 = flat_base_i32 & arith.constant(31, type=i32)
                    dest_base_i32 = h_src_base_i32 * group_size_i32 + g_src_i32
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

        # MFMA hot-region scheduling hints (gated by sched_hints) at each block's
        # MFMA loop tail; sched_barrier(0) fences the bottom. Pure perf hint.
        def _emit_mfma_schedule():
            if not _sched_hints:
                return
            n_mfma = NUM_M_TILES * NUM_N_TILES * NUM_K_TILES   # 16 v_mfma
            # Backend packs A-frags into ds_read2st64_b64 (2 each), so the
            # region holds NUM_M_TILES*NUM_K_TILES/2 ds_read instructions.
            n_dsrd = max(1, (NUM_M_TILES * NUM_K_TILES) // 2)  # 4
            # Deferred preshuffled R: 4 B-frag loads as one front-loaded vmem
            # group; persist_m>1 hoists them out so nothing to model.
            n_vmem = (
                (NUM_N_TILES * NUM_K_TILES)
                if (rot_preshuffled and persist_m == 1) else 0
            )  # 4
            if n_vmem > 0:
                rocdl.sched_vmem(n_vmem)
            per = max(1, n_mfma // n_dsrd)                     # 4 MFMA/ds_read
            emitted = 0
            for _i in range_constexpr(n_dsrd):
                rocdl.sched_dsrd(1)
                rocdl.sched_mfma(per)
                emitted += per
            if n_mfma - emitted > 0:
                rocdl.sched_mfma(n_mfma - emitted)
            rocdl.sched_barrier(0)

        # Persistent row-block loop: WG processes persist_m consecutive groups
        # (bidx_eff = bid_x*persist_m + p), R shared. persist_m==1 emits no loop.
        if persist_m > 1:
            _c0_p = arith.constant(0, type=T.index)
            _c1_p = arith.constant(1, type=T.index)
            _c_pm_p = arith.constant(persist_m, type=T.index)
            _for_persist = _scf.ForOp(_c0_p, _c_pm_p, _c1_p)
            _persist_ip = ir.InsertionPoint(_for_persist.body)
            _persist_ip.__enter__()
            _pi_i32 = arith.index_cast(i32, _for_persist.induction_variable)
            bidx_eff_i32 = (
                bid_x_i32 * arith.constant(persist_m, type=i32) + _pi_i32
            )
        else:
            bidx_eff_i32 = bid_x_i32

        m_base_i32, src_row_for_mfma, src_row_store, store_valid = (
            _row_setup(bidx_eff_i32)
        )

        # Per-block loop: K consecutive blocks share the R-LDS above.
        for kk in range_constexpr(K):
            b_i32 = b_base_i32 + arith.constant(kk, type=i32)
            if use_ptr64:
                b_idx = arith.index_cast(T.index, b_i32)

            # Stage X: coalesced HBM->LDS DMA of this block's 64x32 tile (16 B/
            # lane, LDS row-major, XOR16-swizzled when _swizzle_x).
            if _staged_x:
                x_lds_ptr_ty = ir.Type.parse("!llvm.ptr<3>")
                X_base_idx = _memref.extract_aligned_pointer_as_index(X_lds_view)
                wave_idx_x = arith.index_cast(T.index, wave_i32)
                c_wave_x_bytes = arith.constant(X_LDS_BYTES, type=T.index)
                c_xdma_bytes = arith.constant(X_DMA_BYTES, type=i32)
                c2b_i32 = arith.constant(2, type=i32)
                c3_i32 = arith.constant(3, type=i32)
                c8_i32 = arith.constant(8, type=i32)
                c16_i32 = arith.constant(16, type=i32)
                c_xkmask_i32 = arith.constant(_X_K_BLOCKS16 - 1, type=i32)
                lane_div4 = lane_i32 >> c2_i32
                col_in_tile = (lane_i32 & c3_i32) * c8_i32
                for li in range_constexpr(X_DMA_LOADS):
                    lds_addr_idx = (
                        X_base_idx
                        + wave_idx_x * c_wave_x_bytes
                        + arith.constant(li * 64 * X_DMA_BYTES, type=T.index)
                    )
                    lds_ptr_i64 = rocdl.readfirstlane(
                        T.i64, arith.index_cast(T.i64, lds_addr_idx)
                    )
                    lds_ptr = _llvm.inttoptr(x_lds_ptr_ty, lds_ptr_i64)
                    row_in_tile = arith.constant(li * 16, type=i32) + lane_div4
                    if _swizzle_x:
                        # XOR this lane's 16 B K-chunk by ((row & 3)*16) bytes
                        # (col in elems -> bytes, XOR, back) for conflict-free LDS.
                        col_phys_bytes = col_in_tile * c2b_i32
                        swz_amt = (row_in_tile & c_xkmask_i32) * c16_i32
                        col_sw_elem = (col_phys_bytes ^ swz_amt) >> c1_i32
                    else:
                        col_sw_elem = col_in_tile
                    # Clamp global row (persist_m>1) to avoid reading past input.
                    glob_row = _clamp_row(m_base_i32 + row_in_tile)
                    glob_elem = (
                        glob_row * cols_i32
                        + b_i32 * group_size_i32
                        + col_sw_elem
                    )
                    glob_byte = glob_elem * c2b_i32
                    rocdl.raw_ptr_buffer_load_lds(
                        in_rsrc, lds_ptr, c_xdma_bytes, glob_byte,
                        c0_i32, c0_i32, c0_i32,
                    )
                # Per-wave tile: waitcnt-only (no WG barrier). vmcnt(0) mandatory
                # since X-frag ds_read addresses are dynamic.
                _waitcnt0()

            # Stage 2: MFMA loop (per block b); acc_tiles[(mt, nt)] = 4 f32.
            acc_tiles = {}
            # Fence the top of the schedulable MFMA region so hints only reorder
            # ds_read/MFMA, not the staged-X DMA.
            if _sched_hints:
                rocdl.sched_barrier(0)
            for m_tile in range_constexpr(NUM_M_TILES):
                if use_ptr64:
                    src_row_idx = arith.index_cast(
                        T.index, src_row_for_mfma[m_tile]
                    )
                    row_base_elem_idx = (
                        src_row_idx * cols_idx + b_idx * group_size_idx
                    )
                else:
                    row_off_in_inp = (
                        src_row_for_mfma[m_tile] * cols_i32
                        + b_i32 * group_size_i32
                    )

                for n_tile in range_constexpr(NUM_N_TILES):
                    acc = vector.from_elements(
                        T.vec(WMMA_FRAG_VALS, f32),
                        [c0_f32] * WMMA_FRAG_VALS,
                    )

                    for k_tile in range_constexpr(NUM_K_TILES):
                        k_col_i32 = (
                            arith.constant(k_tile * 16, type=i32)
                            + k_off_in_tile
                        )
                        if _staged_x:
                            # A-fragment from the staged LDS tile: row =
                            # m_tile*16 + lane%16, col = k_col_i32 (row-major).
                            x_row_i32 = (
                                arith.constant(m_tile * 16, type=i32)
                                + lane_mod_16
                            )
                            if _swizzle_x:
                                # Mirror the store-side XOR16 swizzle; byte-XOR
                                # collapses to element-XOR by ((row & 3)*8).
                                x_k_col = k_col_i32 ^ (
                                    (x_row_i32
                                     & arith.constant(_X_K_BLOCKS16 - 1, type=i32))
                                    * arith.constant(8, type=i32)
                                )
                            else:
                                x_k_col = k_col_i32
                            x_lds_off = (
                                arith.index_cast(T.index, wave_i32)
                                * arith.constant(X_TILE_ELEMS, type=T.index)
                                + arith.index_cast(
                                    T.index,
                                    x_row_i32 * group_size_i32
                                    + x_k_col,
                                )
                            )
                            a_frag_bf = vector.load_op(
                                T.vec(WMMA_FRAG_VALS, in_elem_ty),
                                X_lds_view, [x_lds_off],
                            )
                        else:
                            if use_ptr64:
                                elem_off_a_idx = (
                                    row_base_elem_idx
                                    + arith.index_cast(T.index, k_col_i32)
                                )
                                byte_off_a = arith.index_cast(
                                    T.i64, elem_off_a_idx * c_in_elem_bytes_idx
                                )
                                a_ptr = _llvm.GEPOp(
                                    _ptr_ty_as1,
                                    inp_base_ptr,
                                    [byte_off_a],
                                    [-2147483648],
                                    T.i8,
                                    _llvm.GEPNoWrapFlags.none,
                                ).result
                                raw_a = _llvm.LoadOp(
                                    T.vec(2, i32), a_ptr, alignment=8
                                ).result
                            else:
                                elem_off_a = row_off_in_inp + k_col_i32
                                dw_off_a = elem_off_a >> c1_i32
                                raw_a = buffer_ops.buffer_load(
                                    in_rsrc, dw_off_a, vec_width=2, dtype=i32
                                )
                            a_frag_bf = vector.bitcast(
                                T.vec(WMMA_FRAG_VALS, in_elem_ty), raw_a
                            )

                        if preload_b:
                            if (
                                rot_preshuffled
                                and (n_tile, k_tile) not in b_frag_regs
                            ):
                                # Deferred R load at first use (m_tile==0), cached.
                                b_frag_regs[(n_tile, k_tile)] = (
                                    _load_bfrag_preshuffled(n_tile, k_tile)
                                )
                            b_frag_bf = b_frag_regs[(n_tile, k_tile)]
                        else:
                            n_lds = (
                                arith.constant(n_tile * 16, type=i32)
                                + lane_mod_16
                            )
                            k_lds = (
                                arith.constant(k_tile * 16, type=i32)
                                + k_off_in_tile
                            )
                            lds_off_b = n_lds * group_size_i32 + k_lds
                            lds_off_b_idx = arith.index_cast(
                                T.index, lds_off_b
                            )
                            b_frag_bf = vector.load_op(
                                T.vec(WMMA_FRAG_VALS, rot_elem_ty),
                                R_lds_view,
                                [lds_off_b_idx],
                            )

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
                        else:
                            acc = rocdl.mfma_f32_16x16x16f16(
                                T.vec(WMMA_FRAG_VALS, f32),
                                [a_frag_bf, b_frag_bf, acc, 0, 0, 0],
                            )

                    m_local_base = (
                        arith.constant(m_tile * 16, type=i32) + k_off_in_tile
                    )
                    n_local_i32 = (
                        arith.constant(n_tile * 16, type=i32) + lane_mod_16
                    )
                    cur = []
                    for i in range_constexpr(WMMA_FRAG_VALS):
                        val = vector.extract(
                            acc, static_position=[i], dynamic_position=[]
                        )
                        cur.append(val)
                        if not _DPP_TRANSPOSE:
                            m_local_i32 = (
                                m_local_base + arith.constant(i, type=i32)
                            )
                            y_lds_off = (
                                m_local_i32 * group_size_i32 + n_local_i32
                            )
                            y_lds_off_idx = arith.index_cast(
                                T.index, y_lds_off
                            )
                            _memref.store(val, Y_lds_view, [y_lds_off_idx])
                    acc_tiles[(m_tile, n_tile)] = cur

            # Tail scheduling hints for the MFMA hot region (no-op unless
            # sched_hints). Closes with sched_barrier(0).
            _emit_mfma_schedule()

            # Stage 3: build y_vals[0..31] = row tid's 32 values.
            y_vals = [None] * FP4_GROUP_SIZE
            if _DPP_TRANSPOSE:
                # In-register cross-lane transpose (no Y_LDS, no barrier).
                for n_tile in range_constexpr(NUM_N_TILES):
                    acc_by_mt = [
                        acc_tiles[(m_tile, n_tile)]
                        for m_tile in range_constexpr(NUM_M_TILES)
                    ]
                    col_vals = _transpose_c_tile(acc_by_mt)
                    for j in range_constexpr(16):
                        y_vals[n_tile * 16 + j] = col_vals[j]
            else:
                gpu.barrier()
                tid_row_base = tid_i32 * group_size_i32
                for k in range_constexpr(Y_READ_STEPS):
                    lds_off = tid_row_base + arith.constant(
                        k * Y_READ_VEC, type=i32
                    )
                    lds_off_idx = arith.index_cast(T.index, lds_off)
                    v4 = vector.load_op(
                        T.vec(Y_READ_VEC, f32), Y_lds_view, [lds_off_idx],
                    )
                    for j in range_constexpr(Y_READ_VEC):
                        y_vals[k * Y_READ_VEC + j] = vector.extract(
                            v4, static_position=[j], dynamic_position=[],
                    )

            abs_max = c0_f32
            for i in range_constexpr(FP4_GROUP_SIZE):
                abs_v = _llvm.call_intrinsic(
                    f32, "llvm.fabs.f32", [y_vals[i]], [], [],
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

            out_dwords = []
            for dw in range_constexpr(DWORDS_PER_GROUP):
                packed = c0_i32
                for sel in range_constexpr(4):
                    idx = dw * 4 + sel
                    e0 = y_vals[idx * 2]
                    e1 = y_vals[idx * 2 + 1]
                    packed = rocdl.cvt_scalef32_pk_fp4_f32(
                        i32, packed, e0, e1, inv_scale, sel,
                    )
                out_dwords.append(packed)
            out_vec = vector.from_elements(
                T.vec(DWORDS_PER_GROUP, i32), out_dwords,
            )

            # Stage 4: store at natural (unsorted) row addresses. Plain
            # row-major scale out_scale[src_row, b] (1 byte/group, stride scale_N).
            scale_off_i32 = src_row_store * scale_N_i32 + b_i32

            out_byte_off_i32 = (
                src_row_store
                    * arith.constant(scale_N * DWORDS_PER_GROUP * 4, type=i32)
                + b_i32 * out_bytes_per_group_i32
            )

            _if_store = _scf.IfOp(store_valid)
            with ir.InsertionPoint(_if_store.then_block):
                buffer_ops.buffer_store(
                    out_vec, out_rsrc, out_byte_off_i32,
                    offset_is_bytes=True,
                )
                buffer_ops.buffer_store(
                    e8m0_byte_i8, out_scale_rsrc, scale_off_i32,
                    offset_is_bytes=True,
                )
                _scf.YieldOp([])

            # Barrier so the next block's MFMA doesn't overwrite Y_LDS before
            # this block's Stage-3 reads finish (K > 1 only).
            if K > 1:
                gpu.barrier()

        if persist_m > 1:
            # Same Y_LDS reuse barrier across persist iterations (only matters
            # for the LDS-transpose path; DPP keeps Y in VGPRs).
            if not _DPP_TRANSPOSE:
                gpu.barrier()
            _scf.YieldOp([])
            _persist_ip.__exit__(None, None, None)

    @flyc.jit
    def launch_per_1x32_fp4_quant_block_single_rotation_mfma_moe_sorting(
        inp: fx.Tensor,
        rot_R: fx.Tensor,
        out: fx.Tensor,
        out_scale: fx.Tensor,
        num_row_blocks: Int32,
        num_tokens: Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        allocator.finalized = False
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            allocator.finalize()

        num_row_blocks_idx = arith.index_cast(T.index, num_row_blocks)
        # grid.x = ceil(num_row_blocks / W) then ceil-div by persist_m.
        if W > 1:
            gx_i32 = (
                ArithValue(num_row_blocks)
                + arith.constant(W - 1, type=T.i32)
            ) >> arith.constant(LOG2_W, type=T.i32)
        else:
            gx_i32 = ArithValue(num_row_blocks)
        if persist_m > 1:
            gx_i32 = (
                gx_i32 + arith.constant(persist_m - 1, type=T.i32)
            ) // arith.constant(persist_m, type=T.i32)
        grid_x_idx = arith.index_cast(T.index, gx_i32)
        grid_y_idx = arith.constant(scale_N // K, type=T.index)

        launcher = (
            per_1x32_fp4_quant_block_single_rotation_mfma_moe_sorting_kernel(
                inp, rot_R, out, out_scale, num_tokens,
            )
        )
        launcher.launch(
            grid=(grid_x_idx, grid_y_idx, 1),
            block=(BLOCK_DIM, 1, 1),
            stream=stream,
        )

    return launch_per_1x32_fp4_quant_block_single_rotation_mfma_moe_sorting

