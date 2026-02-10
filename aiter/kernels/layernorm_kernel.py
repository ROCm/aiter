"""LayerNorm kernel builder used by tests.

This file intentionally keeps the kernel builder logic identical to the version
embedded in `tests/kernels/test_layernorm.py` (before factoring) to preserve
codegen and performance. Only test-only helpers/imports are removed.
"""

from _mlir import ir

from flydsl.dialects.ext import flir, arith
from flydsl.dialects.ext.python_control_flow import range_constexpr
from . import reduce as reduce_utils
from flydsl.runtime.device import get_rocm_arch as get_hip_arch
from flydsl.utils import SmemAllocator
import _mlir.extras.types as T


KERNEL_NAME = "layernorm"

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


def build_layernorm_module(M: int, N: int, dtype_str: str):
    arch = get_hip_arch()
    DYN = ir.ShapedType.get_dynamic_size()
    # gfx950 supports efficient BF16 pack via v_cvt_pk_bf16_f32.
    # gfx942 does *not* support it and tends to lower f32->bf16 to heavier sequences,
    # so we keep the manual pack there for performance.
    USE_HW_CVT_PK_BF16_F32 = (arch == "gfx950") or arch.startswith("gfx95")
    allocator = SmemAllocator(None, arch=arch)

    tile_cols_py = BLOCK_THREADS * VEC_WIDTH

    # Allocate Shared Memory for block reductions (one slot per wave).
    RED_SLOTS = max(1, (BLOCK_THREADS + WARP_SIZE - 1) // WARP_SIZE)
    _state = {}

    class _LayerNorm(flir.MlirModule):
        GPU_MODULE_NAME = "layernorm_module"
        GPU_MODULE_TARGETS = [f'#rocdl.target<chip = "{arch}", abi = "500">']

        def init_gpu_module(self):
            elem_type = dtype_to_elem_type(dtype_str)
            compute_type = T.f32()  # compute in fp32 for stability (and to keep bf16 safe on backend)
            _state["elem_type"] = elem_type
            _state["compute_type"] = compute_type
            _state["smem_red_sum"] = allocator.allocate_array(T.f32(), RED_SLOTS)
            _state["smem_red_sumsq"] = allocator.allocate_array(T.f32(), RED_SLOTS)
            allocator.finalize()

        @flir.kernel
        def layernorm_kernel(
            self: flir.T.i64,
            Input: lambda: T.memref(DYN, N, _state["elem_type"]),
            Gamma: lambda: T.memref(N, _state["elem_type"]),
            Beta: lambda: T.memref(N, _state["elem_type"]),
            Output: lambda: T.memref(DYN, N, _state["elem_type"]),
            m_in: lambda: T.index(),
        ):
            # Normalize to MLIR index Values early so downstream ops always see `Value`.
            row = flir.const_index(flir.block_idx("x"))
            tid = flir.const_index(flir.thread_idx("x"))

            elem_type = _state["elem_type"]
            compute_type = _state["compute_type"]

            zero_idx = flir.const_index(0)
            n_float = arith.constant(float(N), type=compute_type)
            eps = arith.constant(EPS, type=compute_type)
            fm_fast = flir.arith.FastMathFlags.fast

            base_ptr = allocator.get_base()
            s_sum = _state["smem_red_sum"](base_ptr).get()
            s_sumsq = _state["smem_red_sumsq"](base_ptr).get()
            # FLIR-style tensor views + tiled copies (like elementwise_add_kernel).
            c0_idx = flir.const_index(0)
            tile_cols = BLOCK_THREADS * VEC_WIDTH  # python int
            tensor_In = flir.make_tensor(Input, shape=(m_in, N), strides=(N, 1))
            tensor_Out = flir.make_tensor(Output, shape=(m_in, N), strides=(N, 1))
            tensor_Gamma = flir.make_tensor(Gamma, shape=(N,), strides=(1,))
            tensor_Beta = flir.make_tensor(Beta, shape=(N,), strides=(1,))
            gIn = flir.zipped_divide(tensor_In, (1, tile_cols))
            gOut = flir.zipped_divide(tensor_Out, (1, tile_cols))

            thr_layout = flir.make_ordered_layout((1, BLOCK_THREADS), order=(1, 0))
            val_layout = flir.make_ordered_layout((1, VEC_WIDTH), order=(1, 0))
            copy_atom_e = flir.make_copy_atom(elem_type, vector_size=VEC_WIDTH)
            tiled_copy_e = flir.make_tiled_copy_tv(
                copy_atom_e, thr_layout, val_layout,
                thr_shape=(1, BLOCK_THREADS), val_shape=(1, VEC_WIDTH)
            )
            thr_copy_e = tiled_copy_e.get_slice((tid))
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
            block_reduce_add2 = reduce_utils.make_block_reduce_add2(
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

            # Fast-path: keep the original register-row variant for the tuned (N==8192) case.
            if N == (BLOCK_THREADS * VEC_WIDTH * 4):
                num_tiles_py = 4
                # Read Input once into registers (each thread holds 32 fp32 values = 4 vectors),
                # then reuse those registers for reduction + normalize + writeback.
                c_zero = arith.constant(0.0, type=compute_type)
                thread_sum = (c_zero)
                thread_sumsq = (c_zero)
                # Reduce VGPR pressure by caching bf16/f16 payload vectors when possible.
                cache_as_elem = (dtype_str != "f32")
                in_local = []  # bf16/f16: list[vector<VEC_WIDTH x elem_type>]; f32: list[vector<VEC_WIDTH x f32>]

                vec_type_c = ir.VectorType.get([VEC_WIDTH], compute_type)
                vec_type_e = ir.VectorType.get([VEC_WIDTH], elem_type)
                for tile_i in range_constexpr(num_tiles_py):
                    blkIn = gIn[((row), tile_i)]
                    thrIn = thr_copy_e.partition_S(blkIn)
                    frgIn = flir.make_fragment_like(thrIn, elem_type)
                    vec_e = flir.copy(
                        tiled_copy_e,
                        thrIn,
                        frgIn,
                        nontemporal=USE_NONTEMPORAL,
                        alignment=VEC_ALIGN,
                        return_vector=True,
                    )

                    if cache_as_elem:
                        in_local.append(vec_e)
                        # Keep `x` as an ArithValue-like wrapper (so `x * x` works),
                        # but ensure the input is an MLIR Value.
                        x = flir.arith.extf(vec_type_c, arith.as_value(vec_e))
                    else:
                        x = vec_e
                        in_local.append(x)

                    # Avoid Python `*` on MLIR `OpResult` (not overloaded); use ArithValue ops.
                    x_av = arith.ArithValue(arith.as_value(x))
                    x2 = x_av * x_av
                    # `vector.reduction` expects an MLIR Value operand (not an ArithValue wrapper).
                    red = flir.vector.reduction(
                        compute_type, "add", arith.as_value(x), fastmath=fm_fast
                    )
                    red2 = flir.vector.reduction(
                        compute_type, "add", arith.as_value(x2), fastmath=fm_fast
                    )
                    thread_sum = thread_sum + red
                    thread_sumsq = thread_sumsq + red2

                sum_val, sumsq_val = block_reduce_add2(thread_sum, thread_sumsq, s_sum, s_sumsq)

                inv_n = arith.constant(1.0 / float(N), type=compute_type)
                sum_val = arith.ArithValue(arith.as_value(sum_val))
                sumsq_val = arith.ArithValue(arith.as_value(sumsq_val))
                mean = sum_val * inv_n
                mean_sq = sumsq_val * inv_n
                mean2 = mean * mean
                var = mean_sq - mean2
                # Numerical safety: with fast-math and cancellation, `var` can become slightly negative
                # and lead to NaNs in rsqrt for small-N cases. Clamp to >= 0 before adding eps.
                c0_f = arith.constant(0.0, type=compute_type)
                is_neg = var < c0_f
                var = arith.select(is_neg, c0_f, var)

                var_eps = var + eps
                rstd = flir.math.rsqrt(arith.as_value(var_eps), fastmath=fm_fast)

                vec_type_e = ir.VectorType.get([VEC_WIDTH], elem_type)
                vec_type_c = ir.VectorType.get([VEC_WIDTH], compute_type)
                # `vector.splat` expects a raw MLIR Value operand (not wrapper objects).
                mean_splat = flir.vector.splat(vec_type_c, arith.as_value(mean))
                rstd_splat = flir.vector.splat(vec_type_c, arith.as_value(rstd))
                mean_splat_av = arith.ArithValue(arith.as_value(mean_splat))
                rstd_splat_av = arith.ArithValue(arith.as_value(rstd_splat))

                # Pipeline Gamma/Beta loads.
                c_vecw = flir.const_index(VEC_WIDTH)
                # Avoid Python `*` on MLIR Values (OpResult); use explicit arith ops.
                thread_offset_base = flir.arith.MulIOp(
                    arith.as_value(tid), arith.as_value(c_vecw)
                ).result
                c_base0 = flir.const_index(0)
                curr_idx0 = flir.arith.AddIOp(
                    arith.as_value(c_base0), arith.as_value(thread_offset_base)
                ).result
                g_e_cur = flir.vector.load(vec_type_e, Gamma, [arith.as_value(curr_idx0)], alignment=VEC_ALIGN)
                b_e_cur = flir.vector.load(vec_type_e, Beta, [arith.as_value(curr_idx0)], alignment=VEC_ALIGN)
                g_cur = g_e_cur if dtype_str == "f32" else flir.arith.extf(vec_type_c, arith.as_value(g_e_cur))
                b_cur = b_e_cur if dtype_str == "f32" else flir.arith.extf(vec_type_c, arith.as_value(b_e_cur))

                for tile_i in range_constexpr(num_tiles_py):
                    base_idx_int = tile_i * tile_cols
                    c_base = flir.const_index(base_idx_int)
                    curr_idx = flir.arith.AddIOp(
                        arith.as_value(c_base), arith.as_value(thread_offset_base)
                    ).result

                    x = in_local[tile_i]
                    if cache_as_elem:
                        x = flir.arith.extf(vec_type_c, arith.as_value(x))
                    if tile_i + 1 < num_tiles_py:
                        next_base_idx_int = (tile_i + 1) * tile_cols
                        c_base_next = flir.const_index(next_base_idx_int)
                        next_idx = flir.arith.AddIOp(
                            arith.as_value(c_base_next), arith.as_value(thread_offset_base)
                        ).result
                        g_e_next = flir.vector.load(vec_type_e, Gamma, [arith.as_value(next_idx)], alignment=VEC_ALIGN)
                        b_e_next = flir.vector.load(vec_type_e, Beta, [arith.as_value(next_idx)], alignment=VEC_ALIGN)
                        g_next = g_e_next if dtype_str == "f32" else flir.arith.extf(vec_type_c, arith.as_value(g_e_next))
                        b_next = b_e_next if dtype_str == "f32" else flir.arith.extf(vec_type_c, arith.as_value(b_e_next))
                    else:
                        g_next = g_cur
                        b_next = b_cur

                    # Keep math in ArithValue so operator overloads work, unwrap only at MLIR boundaries.
                    x_av = arith.ArithValue(arith.as_value(x))
                    g_av = arith.ArithValue(arith.as_value(g_cur))
                    b_av = arith.ArithValue(arith.as_value(b_cur))
                    y = (x_av - mean_splat_av) * rstd_splat_av
                    y = (y * g_av) + b_av
                    y = arith.as_value(y)

                    if dtype_str == "bf16":
                        if USE_HW_CVT_PK_BF16_F32:
                            out_e = flir.arith.truncf(vec_type_e, (y))
                        else:
                            # Software bf16 pack with round-to-nearest-even (RNE).
                            #
                            # bf16 = round_to_nearest_even(f32) by adding:
                            #   bias = 0x7FFF + ((u >> 16) & 1)
                            # then taking the top 16 bits.
                            vec_i32_ty = ir.VectorType.get([VEC_WIDTH], T.i32())
                            vec4_i32_ty = ir.VectorType.get([VEC_WIDTH // 2], T.i32())
                            vec_bf16_ty = ir.VectorType.get([VEC_WIDTH], elem_type)
                            c16_i32 = arith.constant(16, type=T.i32())
                            c16_i32_v = flir.vector.splat(vec_i32_ty, arith.as_value(c16_i32))

                            u = flir.arith.bitcast(vec_i32_ty, (y))
                            u = arith.as_value(u)

                            upper = arith.shrui(u, c16_i32_v)  # i32 vector (upper 16 bits in low bits)
                            c1_i32 = arith.constant(1, type=T.i32())
                            c1_v = flir.vector.splat(vec_i32_ty, arith.as_value(c1_i32))
                            lsb = arith.andi(upper, arith.as_value(c1_v))

                            c7fff_i32 = arith.constant(0x7FFF, type=T.i32())
                            c7fff_v = flir.vector.splat(vec_i32_ty, arith.as_value(c7fff_i32))
                            bias = arith.ArithValue(arith.as_value(c7fff_v)) + arith.ArithValue(arith.as_value(lsb))
                            u_round = arith.ArithValue(u) + bias

                            bf16_bits = arith.as_value(arith.shrui(u_round, c16_i32_v))

                            even = flir.vector.shuffle(bf16_bits, bf16_bits, mask=[0, 2, 4, 6])
                            odd = flir.vector.shuffle(bf16_bits, bf16_bits, mask=[1, 3, 5, 7])
                            odd_sh = arith.as_value(arith.shli(arith.as_value(odd), flir.vector.splat(vec4_i32_ty, arith.as_value(c16_i32))))
                            packed = arith.as_value(arith.ori(arith.as_value(even), odd_sh))
                            out_e = flir.vector.bitcast(vec_bf16_ty, (packed))
                    else:
                        out_e = y if dtype_str == "f32" else flir.arith.truncf(vec_type_e, (y))

                    blkOut = gOut[((row), tile_i)]
                    thrOut = thr_copy_e.partition_S(blkOut)
                    frgOut = flir.make_fragment_like(thrOut, elem_type)
                    # `vector.store` expects a raw MLIR Value for the stored vector.
                    flir.vector.store(
                        arith.as_value(out_e),
                        frgOut.memref,
                        [c0_idx, c0_idx],
                        alignment=VEC_ALIGN,
                    )
                    flir.copy(
                        tiled_copy_e,
                        frgOut,
                        thrOut,
                        nontemporal=USE_NONTEMPORAL,
                        alignment=VEC_ALIGN,
                    )

                    g_cur = g_next
                    b_cur = b_next
            else:
                # Generic path: 2-pass global implementation supporting arbitrary N (incl. tail).
                # For these small/unaligned-N test cases, correctness & robustness matter more than peak perf.
                c_N = flir.const_index(N)
                c_zero = (arith.constant(0.0, type=compute_type))
                thread_sum = (c_zero)
                thread_sumsq = (c_zero)

                # Pass1: sum + sumsq
                for base_idx_int in range_constexpr(0, N, BLOCK_THREADS):
                    c_base = flir.const_index(base_idx_int)
                    idx = flir.arith.AddIOp(arith.as_value(c_base), arith.as_value(tid)).result
                    is_valid = arith.ult(idx, c_N)
                    thread_sum_next = thread_sum
                    thread_sumsq_next = thread_sumsq
                    if is_valid:
                        x_e = flir.memref.load(Input, [(row), arith.as_value(idx)])
                        x = (x_e) if dtype_str == "f32" else flir.arith.extf(compute_type, arith.as_value(x_e))
                        x_av = arith.ArithValue(arith.as_value(x))
                        x2 = x_av * x_av
                        thread_sum_next = thread_sum + x
                        thread_sumsq_next = thread_sumsq + x2
                    thread_sum, thread_sumsq = thread_sum_next, thread_sumsq_next

                sum_val, sumsq_val = block_reduce_add2(thread_sum, thread_sumsq, s_sum, s_sumsq)

                inv_n = arith.constant(1.0 / float(N), type=compute_type)
                sum_val = arith.ArithValue(arith.as_value(sum_val))
                sumsq_val = arith.ArithValue(arith.as_value(sumsq_val))
                mean = sum_val * inv_n
                mean_sq = sumsq_val * inv_n
                mean2 = mean * mean
                var = mean_sq - mean2
                # Numerical safety: clamp variance to >=0 to avoid NaNs in rsqrt on small-N cases.
                c0_f = arith.constant(0.0, type=compute_type)
                is_neg = var < c0_f
                var = arith.select(is_neg, c0_f, var)
                var_eps = var + eps
                rstd = flir.math.rsqrt(arith.as_value(var_eps), fastmath=fm_fast)

                # Pass2: normalize + affine + store
                for base_idx_int in range_constexpr(0, N, BLOCK_THREADS):
                    c_base = flir.const_index(base_idx_int)
                    idx = flir.arith.AddIOp(arith.as_value(c_base), arith.as_value(tid)).result
                    is_valid = arith.ult(idx, c_N)
                    if is_valid:
                        x_e = flir.memref.load(Input, [(row), arith.as_value(idx)])
                        g_e = flir.memref.load(Gamma, [arith.as_value(idx)])
                        b_e = flir.memref.load(Beta, [arith.as_value(idx)])
                        x = (x_e) if dtype_str == "f32" else flir.arith.extf(compute_type, arith.as_value(x_e))
                        g = (g_e) if dtype_str == "f32" else flir.arith.extf(compute_type, arith.as_value(g_e))
                        b = (b_e) if dtype_str == "f32" else flir.arith.extf(compute_type, arith.as_value(b_e))
                        diff = x - mean
                        norm = diff * rstd
                        scaled = norm * g
                        y = scaled + b
                        if dtype_str == "bf16":
                            y_e = flir.arith.truncf(elem_type, (y))
                        else:
                            y_e = y if dtype_str == "f32" else flir.arith.truncf(elem_type, (y))
                        flir.memref.store((y_e), Output, [(row), (idx)])

        @flir.jit
        def __call__(
            self: flir.T.i64,
            Input: lambda: T.memref(DYN, N, _state["elem_type"]),
            Gamma: lambda: T.memref(N, _state["elem_type"]),
            Beta: lambda: T.memref(N, _state["elem_type"]),
            Output: lambda: T.memref(DYN, N, _state["elem_type"]),
            m_in: lambda: T.index(),
        ):
            c1 = (flir.arith_ext.index(1))
            gx = (m_in)
            bx = (flir.arith_ext.index(BLOCK_THREADS))
            flir.gpu_ext.LaunchFuncOp(
                ["layernorm_module", "layernorm_kernel"],
                grid_size=(gx, c1, c1),
                block_size=(bx, c1, c1),
                kernel_operands=[Input, Gamma, Beta, Output, m_in],
            )

    return _LayerNorm()


