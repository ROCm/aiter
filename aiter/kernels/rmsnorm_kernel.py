"""RMSNorm kernel builder used by tests.

This file intentionally keeps the kernel builder logic identical to the version
embedded in `tests/kernels/test_rmsnorm.py` (before factoring) to preserve
codegen and performance. Only test-only helpers/imports are removed.
"""

from _mlir import ir

from flydsl.dialects.ext import flir, arith
from flydsl.dialects.ext.python_control_flow import range_constexpr
from . import reduce as reduce_utils
from flydsl.runtime.device import get_rocm_arch as get_hip_arch
from flydsl.utils import SmemAllocator
import _mlir.extras.types as T


KERNEL_NAME = "rmsnorm"


EPS = 1e-5


def dtype_to_elem_type(dtype_str: str):
    if dtype_str == "f32":
        return T.f32()
    if dtype_str == "f16":
        return T.f16()
    if dtype_str == "bf16":
        return T.bf16()
    raise ValueError(f"unsupported dtype: {dtype_str}")


BLOCK_THREADS = 256
WARP_SIZE = 64
VEC_WIDTH = 8
USE_NONTEMPORAL = True
VEC_ALIGN = 16


def build_rmsnorm_module(M: int, N: int, dtype_str: str):
    arch = get_hip_arch()
    DYN = ir.ShapedType.get_dynamic_size()
    allocator = SmemAllocator(None, arch=arch)
    RED_SLOTS = max(1, (BLOCK_THREADS + WARP_SIZE - 1) // WARP_SIZE)
    _state = {}

    class _RMSNorm(flir.MlirModule):
        GPU_MODULE_NAME = "rmsnorm_module"
        GPU_MODULE_TARGETS = [f'#rocdl.target<chip = "{arch}">']

        def init_gpu_module(self):
            elem_type = dtype_to_elem_type(dtype_str)
            compute_type = T.f32()
            _state["elem_type"] = elem_type
            _state["compute_type"] = compute_type
            _state["smem_red"] = allocator.allocate_array(T.f32(), RED_SLOTS)
            # Represent LDS row cache as a 2D (1, N) memref so tensor indexing uses 2 indices.
            _state["smem_row"] = allocator.allocate_tensor((1, N), elem_type)
            allocator.finalize()

        @flir.kernel
        def rmsnorm_kernel(
            self: flir.T.i64,
            Input: lambda: T.memref(DYN, N, _state["elem_type"]),
            Gamma: lambda: T.memref(N, _state["elem_type"]),
            Output: lambda: T.memref(DYN, N, _state["elem_type"]),
            m_in: lambda: T.index(),
        ):
            # Normalize to MLIR index Values early so downstream ops always see `Value`.
            row = flir.const_index(flir.block_idx("x"))
            tid = flir.const_index(flir.thread_idx("x"))
            elem_type = _state["elem_type"]
            compute_type = _state["compute_type"]

            zero_idx = flir.const_index(0)
            # `arith.constant` API takes value + `type=...` (do not pass type positionally).
            n_float = arith.constant(float(N), type=compute_type)
            eps = arith.constant(float(EPS), type=compute_type)
            fm_fast = flir.arith.FastMathFlags.fast

            base_ptr = allocator.get_base()
            s_red = _state["smem_red"](base_ptr).get()
            s_row = _state["smem_row"](base_ptr).get()
            # FLir-style tensor views + tiled copies (like elementwise_add_kernel).
            c0_idx = flir.const_index(0)
            tile_cols = BLOCK_THREADS * VEC_WIDTH  # python int
            tensor_In = flir.make_tensor(Input, shape=(m_in, N), strides=(N, 1))
            tensor_Out = flir.make_tensor(Output, shape=(m_in, N), strides=(N, 1))
            tensor_Gamma = flir.make_tensor(Gamma, shape=(N,), strides=(1,))
            tensor_S = flir.make_tensor(s_row, shape=(1, N), strides=(N, 1))
            gIn = flir.zipped_divide(tensor_In, (1, tile_cols))
            gOut = flir.zipped_divide(tensor_Out, (1, tile_cols))
            gS = flir.zipped_divide(tensor_S, (1, tile_cols))

            block_reduce_add = reduce_utils.make_block_reduce_add(
                tid=tid,
                fm_fast=fm_fast,
                WARP_SIZE=WARP_SIZE,
                RED_SLOTS=RED_SLOTS,
                gpu=flir.gpu_ext,
                arith=arith,
                arith_ops=flir.arith,
                flir=flir,
                T=T,
                ir=ir,
                zero_idx=zero_idx,
            )

            # For small/un-aligned N (N < 2048), use a simple 2-pass global kernel.
            # This avoids relying on the tiled-copy machinery, which is tuned for the 2048-wide tiles.
            if N < tile_cols:
                c_N = flir.const_index(N)
                c_zero = arith.constant(0.0, type=compute_type)
                thread_sumsq = (c_zero)

                # Pass1: sumsq
                for base_idx_int in range_constexpr(0, N, BLOCK_THREADS):
                    c_base = flir.const_index(base_idx_int)
                    idx = c_base + tid
                    is_valid = arith.ult(idx, c_N)
                    thread_sumsq_next = thread_sumsq
                    if is_valid:
                        x_e = flir.memref.load(Input, [(row), arith.as_value(idx)])
                        x = (x_e) if dtype_str == "f32" else flir.arith.extf(compute_type, arith.as_value(x_e))
                        x2 = x * x
                        thread_sumsq_next = thread_sumsq + x2
                    thread_sumsq = thread_sumsq_next

                sum_sq = block_reduce_add(thread_sumsq, s_red)
                mean_sq = sum_sq / n_float
                ms_eps = mean_sq + eps
                rrms = flir.math.rsqrt(arith.as_value(ms_eps))

                # Pass2: normalize + gamma + store
                for base_idx_int in range_constexpr(0, N, BLOCK_THREADS):
                    c_base = flir.const_index(base_idx_int)
                    idx = c_base + tid
                    is_valid = arith.ult(idx, c_N)
                    if is_valid:
                        x_e = flir.memref.load(Input, [(row), arith.as_value(idx)])
                        g_e = flir.memref.load(Gamma, [arith.as_value(idx)])
                        x = (x_e) if dtype_str == "f32" else flir.arith.extf(compute_type, arith.as_value(x_e))
                        g = (g_e) if dtype_str == "f32" else flir.arith.extf(compute_type, arith.as_value(g_e))
                        norm = (arith.ArithValue(x) * rrms).value
                        y = (arith.ArithValue(norm) * g).value
                        y_e = y if dtype_str == "f32" else flir.arith.truncf(elem_type, arith.as_value(y))
                        flir.memref.store((y_e), Output, [(row), arith.as_value(idx)])
                return

            thr_layout = flir.make_ordered_layout((1, BLOCK_THREADS), order=(1, 0))
            val_layout = flir.make_ordered_layout((1, VEC_WIDTH), order=(1, 0))
            copy_atom_e = flir.make_copy_atom(elem_type, vector_size=VEC_WIDTH)
            tiled_copy_e = flir.make_tiled_copy_tv(
                copy_atom_e, thr_layout, val_layout,
                thr_shape=(1, BLOCK_THREADS), val_shape=(1, VEC_WIDTH)
            )
            thr_copy_e = tiled_copy_e.get_slice((tid))

            # Pass0: global -> LDS row cache (1-pass global read)
            for base_idx_int in range_constexpr(0, N, BLOCK_THREADS * VEC_WIDTH):
                c_base = flir.const_index(base_idx_int)
                thread_offset_base = (arith.ArithValue(tid) * VEC_WIDTH).value
                curr_idx = (arith.ArithValue(c_base) + thread_offset_base).value

                tile_safe = (base_idx_int + BLOCK_THREADS * VEC_WIDTH) <= N
                if tile_safe:
                    tile_i = base_idx_int // tile_cols  # python int
                    blkIn = gIn[((row), tile_i)]
                    blkS = gS[(0, tile_i)]
                    thrIn = thr_copy_e.partition_S(blkIn)
                    thrS = thr_copy_e.partition_S(blkS)
                    flir.copy(
                        tiled_copy_e,
                        thrIn,
                        thrS,
                        nontemporal=USE_NONTEMPORAL,
                        alignment=VEC_ALIGN,
                    )
                else:
                    c_N = flir.const_index(N)
                    for k in range_constexpr(VEC_WIDTH):
                        c_k = flir.const_index(k)
                        idx_k = curr_idx + c_k
                        is_valid = arith.ult(idx_k, c_N)
                        if is_valid:
                            v_e = tensor_In[((row), arith.as_value(idx_k))]
                            tensor_S[((c0_idx), arith.as_value(idx_k))] = (v_e)

            flir.gpu_ext.barrier()

            # Pass1: sumsq (from LDS row cache)
            c_zero = arith.constant(0.0, type=compute_type)
            thread_sumsq = (c_zero)

            for base_idx_int in range_constexpr(0, N, BLOCK_THREADS * VEC_WIDTH):
                c_base = flir.const_index(base_idx_int)
                thread_offset_base = (arith.ArithValue(tid) * VEC_WIDTH).value
                curr_idx = (arith.ArithValue(c_base) + thread_offset_base).value

                tile_safe = (base_idx_int + BLOCK_THREADS * VEC_WIDTH) <= N
                if tile_safe:
                    vec_type_e = ir.VectorType.get([VEC_WIDTH], elem_type)
                    vec_e = flir.vector.load(
                        vec_type_e, s_row, [(c0_idx), (curr_idx)], alignment=VEC_ALIGN
                    )
                    vec_type_c = ir.VectorType.get([VEC_WIDTH], compute_type)
                    vec = vec_e if dtype_str == "f32" else flir.arith.extf(vec_type_c, arith.as_value(vec_e))
                    vec2 = (arith.ArithValue(vec) * vec).value
                    red2 = flir.vector.reduction(compute_type, "add", (vec2), fastmath=fm_fast)
                    thread_sumsq = thread_sumsq + red2
                else:
                    c_N = flir.const_index(N)
                    for k in range_constexpr(VEC_WIDTH):
                        c_k = flir.const_index(k)
                        idx_k = curr_idx + c_k
                        is_valid = arith.ult(idx_k, c_N)
                        if is_valid:
                            v_e = tensor_S[((c0_idx), arith.as_value(idx_k))]
                        else:
                            v_e = arith.constant(0.0, type=elem_type)
                        v = (v_e) if dtype_str == "f32" else flir.arith.extf(compute_type, arith.as_value(v_e))
                        v2 = (arith.ArithValue(v) * v).value
                        thread_sumsq = thread_sumsq + v2

            sum_sq = block_reduce_add(thread_sumsq, s_red)
            mean_sq = sum_sq / n_float

            ms_eps = mean_sq + eps
            rrms = flir.math.rsqrt(arith.as_value(ms_eps))

            # Pass2: normalize + gamma + store
            vec_type_e = ir.VectorType.get([VEC_WIDTH], elem_type)
            vec_type_c = ir.VectorType.get([VEC_WIDTH], compute_type)
            rrms_splat = flir.vector.splat(vec_type_c, arith.as_value(rrms))

            # Software pipeline for aligned tiles: prefetch Gamma
            g_pref_e = None
            if N >= BLOCK_THREADS * VEC_WIDTH:
                c_base0 = flir.const_index(0)
                thread_offset0 = (arith.ArithValue(tid) * VEC_WIDTH).value
                curr0 = (arith.ArithValue(c_base0) + thread_offset0).value
                vec_type_e0 = ir.VectorType.get([VEC_WIDTH], elem_type)
                g_pref_e = flir.vector.load(vec_type_e0, Gamma, [arith.as_value(curr0)], alignment=VEC_ALIGN)

            for base_idx_int in range_constexpr(0, N, BLOCK_THREADS * VEC_WIDTH):
                c_base = flir.const_index(base_idx_int)
                thread_offset_base = (arith.ArithValue(tid) * VEC_WIDTH).value
                curr_idx = (arith.ArithValue(c_base) + thread_offset_base).value

                tile_safe = (base_idx_int + BLOCK_THREADS * VEC_WIDTH) <= N
                if tile_safe:
                    # Prefetch next Gamma early (software pipeline)
                    next_base_int = base_idx_int + (BLOCK_THREADS * VEC_WIDTH)
                    if next_base_int < N:
                        c_base_n = flir.const_index(next_base_int)
                        curr_idx_n = (arith.ArithValue(c_base_n) + thread_offset_base).value
                        g_next_e = flir.vector.load(vec_type_e, Gamma, [arith.as_value(curr_idx_n)], alignment=VEC_ALIGN)
                    else:
                        g_next_e = None

                    x_e = flir.vector.load(
                        vec_type_e, s_row, [(c0_idx), arith.as_value(curr_idx)], alignment=VEC_ALIGN
                    )
                    # Gamma is reused across many blocks: do NOT use nontemporal here.
                    g_e = g_pref_e if g_pref_e is not None else flir.vector.load(vec_type_e, Gamma, [arith.as_value(curr_idx)], alignment=VEC_ALIGN)
                    x = x_e if dtype_str == "f32" else flir.arith.extf(vec_type_c, arith.as_value(x_e))
                    g = g_e if dtype_str == "f32" else flir.arith.extf(vec_type_c, arith.as_value(g_e))
                    norm = (arith.ArithValue(x) * rrms_splat).value
                    y = (arith.ArithValue(norm) * g).value
                    y_e = y if dtype_str == "f32" else flir.arith.truncf(vec_type_e, arith.as_value(y))
                    tile_i = base_idx_int // tile_cols  # python int
                    blkOut = gOut[((row), tile_i)]
                    thrOut = thr_copy_e.partition_S(blkOut)
                    frgOut = flir.make_fragment_like(thrOut, elem_type)
                    flir.vector.store(arith.as_value(y_e), frgOut.memref, [c0_idx, c0_idx], alignment=VEC_ALIGN)
                    flir.copy(
                        tiled_copy_e,
                        frgOut,
                        thrOut,
                        nontemporal=USE_NONTEMPORAL,
                        alignment=VEC_ALIGN,
                    )
                    g_pref_e = g_next_e
                else:
                    c_N = flir.const_index(N)
                    for k in range_constexpr(VEC_WIDTH):
                        c_k = flir.const_index(k)
                        idx_k = curr_idx + c_k
                        is_valid = arith.ult(idx_k, c_N)
                        if is_valid:
                            x_e = tensor_S[((c0_idx), arith.as_value(idx_k))]
                            g_e = tensor_Gamma[arith.as_value(idx_k)]
                            x = (x_e) if dtype_str == "f32" else flir.arith.extf(compute_type, arith.as_value(x_e))
                            g = (g_e) if dtype_str == "f32" else flir.arith.extf(compute_type, arith.as_value(g_e))
                            norm = (arith.ArithValue(x) * rrms).value
                            y = (arith.ArithValue(norm) * g).value
                            y_e = y if dtype_str == "f32" else flir.arith.truncf(elem_type, arith.as_value(y))
                            tensor_Out[((row), arith.as_value(idx_k))] = (y_e)

        @flir.jit
        def __call__(
            self: flir.T.i64,
            Input: lambda: T.memref(DYN, N, _state["elem_type"]),
            Gamma: lambda: T.memref(N, _state["elem_type"]),
            Output: lambda: T.memref(DYN, N, _state["elem_type"]),
            m_in: lambda: T.index(),
        ):
            c1 = arith.as_value(flir.arith_ext.index(1))
            gx = arith.as_value(m_in)
            bx = arith.as_value(flir.arith_ext.index(BLOCK_THREADS))
            flir.gpu_ext.LaunchFuncOp(
                ["rmsnorm_module", "rmsnorm_kernel"],
                grid_size=(gx, c1, c1),
                block_size=(bx, c1, c1),
                kernel_operands=[Input, Gamma, Output, m_in],
            )

    return _RMSNorm()


