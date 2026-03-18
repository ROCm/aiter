"""LayerNorm kernel builder using the @flyc.kernel API.

LayerNorm(x) = (x - mean) / sqrt(var + eps) * gamma + beta

Two paths:
  - Fast path (N == BLOCK_THREADS * VEC_WIDTH * 4): vectorised tiled copy,
    register caching, pipelined gamma/beta loads.
  - Generic path (arbitrary N): scalar 2-pass implementation.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.compiler.kernel_function import CompilationContext

from flydsl.expr import arith, vector, gpu, range_constexpr
from flydsl.expr.typing import T, Int32

from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr
from flydsl.runtime.device import get_rocm_arch as get_hip_arch

from flydsl._mlir import ir
from flydsl.expr import buffer_ops

KERNEL_NAME = "layernorm"

EPS = 1e-5

BLOCK_THREADS = 256
WARP_SIZE = 64
VEC_WIDTH = 8
USE_NONTEMPORAL = True
VEC_ALIGN = 16


def dtype_to_elem_type(dtype_str: str):
    if dtype_str == "f32":
        return T.f32
    if dtype_str == "f16":
        return T.f16
    if dtype_str == "bf16":
        return T.bf16
    raise ValueError(f"unsupported dtype: {dtype_str}")


def build_layernorm_module(M: int, N: int, dtype_str: str):
    arch = get_hip_arch()
    USE_HW_CVT_PK_BF16_F32 = (arch == "gfx950") or str(arch).startswith("gfx95")

    tile_cols_py = BLOCK_THREADS * VEC_WIDTH

    RED_SLOTS = max(1, (BLOCK_THREADS + WARP_SIZE - 1) // WARP_SIZE)

    elem_bits = 32 if dtype_str == "f32" else 16

    # -- Shared-memory allocation for block reductions ---------------------
    allocator = SmemAllocator(None, arch=arch)
    f32_bytes = 4
    sum_offset = allocator._align(allocator.ptr, 16)
    allocator.ptr = sum_offset + RED_SLOTS * f32_bytes
    sumsq_offset = allocator._align(allocator.ptr, 16)
    allocator.ptr = sumsq_offset + RED_SLOTS * f32_bytes

    # -- GPU kernel --------------------------------------------------------
    @flyc.kernel
    def layernorm_kernel(
        Input: fx.Tensor,
        Gamma: fx.Tensor,
        Beta: fx.Tensor,
        Output: fx.Tensor,
    ):
        bid = fx.block_idx.x
        tid = fx.thread_idx.x

        elem_type = dtype_to_elem_type(dtype_str)
        compute_type = T.f32

        fm_fast = arith.FastMathFlags.fast
        eps_c = arith.constant(EPS, type=compute_type)

        base_ptr = allocator.get_base()
        s_sum = SmemPtr(base_ptr, sum_offset, T.f32, shape=(RED_SLOTS,))
        s_sumsq = SmemPtr(base_ptr, sumsq_offset, T.f32, shape=(RED_SLOTS,))
        s_sum.get()
        s_sumsq.get()

        # -- helpers: wave / block reduction -------------------------------
        def wave_reduce_add(x):
            width_i32 = arith.constant(WARP_SIZE, type=T.i32)
            w = x
            for sh in [32, 16, 8, 4, 2, 1]:
                off = arith.constant(sh, type=T.i32)
                peer = w.shuffle_xor(off, width_i32)
                w = w.addf(peer, fastmath=fm_fast)
            return w

        def block_reduce_add2(val0, val1):
            if RED_SLOTS == 1:
                return wave_reduce_add(val0), wave_reduce_add(val1)

            lane = tid % WARP_SIZE
            wave = tid // WARP_SIZE

            w0 = wave_reduce_add(val0)
            w1 = wave_reduce_add(val1)

            if arith.cmpi(arith.CmpIPredicate.eq, lane, Int32(0)):
                wave_idx = arith.index_cast(T.index, wave)
                s_sum.store(w0, [wave_idx])
                s_sumsq.store(w1, [wave_idx])
            gpu.barrier()

            if arith.cmpi(arith.CmpIPredicate.eq, wave, Int32(0)):
                in_range = lane < RED_SLOTS
                lane_safe = arith.select(in_range, lane, Int32(0))
                lane_safe_idx = arith.index_cast(T.index, lane_safe)
                v0 = s_sum.load([lane_safe_idx])
                v1 = s_sumsq.load([lane_safe_idx])
                z = arith.constant(0.0, type=T.f32)
                ww0 = arith.select(in_range, v0, z)
                ww1 = arith.select(in_range, v1, z)
                ww0 = wave_reduce_add(ww0)
                ww1 = wave_reduce_add(ww1)

                if arith.cmpi(arith.CmpIPredicate.eq, lane, Int32(0)):
                    c0_idx = arith.constant(0, index=True)
                    s_sum.store(ww0, [c0_idx])
                    s_sumsq.store(ww1, [c0_idx])
            gpu.barrier()

            c0_idx = arith.constant(0, index=True)
            return s_sum.load([c0_idx]), s_sumsq.load([c0_idx])

        def compute_mean_rstd(sum_val, sumsq_val):
            from flydsl.expr.arith import ArithValue

            inv_n = arith.constant(1.0 / float(N), type=compute_type)
            s = ArithValue(sum_val)
            ss = ArithValue(sumsq_val)
            mean = s * inv_n
            mean_sq = ss * inv_n
            mean2 = mean * mean
            var = mean_sq - mean2
            c0_f = arith.constant(0.0, type=compute_type)
            is_neg = var < c0_f
            var = is_neg.select(c0_f, var)
            var_eps = ArithValue(var) + eps_c
            rstd = var_eps.rsqrt(fastmath=fm_fast)
            return mean, rstd

        # ==================================================================
        # Fast path: N == BLOCK_THREADS * VEC_WIDTH * 4
        # Uses buffer_load / buffer_store for high-bandwidth vectorised
        # memory access (same approach as preshuffle_gemm).
        # ==================================================================
        if N == (BLOCK_THREADS * VEC_WIDTH * 4) and elem_bits <= 16:
            from flydsl.expr.arith import ArithValue

            num_tiles_py = 4
            tile_cols = BLOCK_THREADS * VEC_WIDTH

            c_zero_f = arith.constant(0.0, type=compute_type)
            thread_sum = c_zero_f
            thread_sumsq = c_zero_f
            cache_as_elem = dtype_str != "f32"
            in_local = []

            vec_type_c = T.vec(VEC_WIDTH, compute_type)
            vec_type_e = T.vec(VEC_WIDTH, elem_type)

            elem_bytes = 4 if dtype_str == "f32" else 2
            vec_dwords = (VEC_WIDTH * elem_bytes) // 4

            in_rsrc = buffer_ops.create_buffer_resource(Input, max_size=True)
            out_rsrc = buffer_ops.create_buffer_resource(Output, max_size=True)
            gamma_rsrc = buffer_ops.create_buffer_resource(Gamma, max_size=True)
            beta_rsrc = buffer_ops.create_buffer_resource(Beta, max_size=True)

            row_soffset = ArithValue(bid) * (N * elem_bytes)
            thr_col_bytes = ArithValue(tid) * (VEC_WIDTH * elem_bytes)

            def _load_vec_buf(rsrc, col_byte_off, soff=None):
                dw = col_byte_off.shrui(arith.constant(2, type=T.i32))
                raw = buffer_ops.buffer_load(
                    rsrc,
                    dw,
                    vec_width=vec_dwords,
                    dtype=T.i32,
                    soffset_bytes=soff,
                )
                if vec_dwords == VEC_WIDTH:
                    return raw.bitcast(vec_type_e)
                return vector.bitcast(vec_type_e, raw)

            def _store_vec_buf(data, rsrc, col_byte_off, soff=None):
                dw = col_byte_off.shrui(arith.constant(2, type=T.i32))
                buffer_ops.buffer_store(
                    data,
                    rsrc,
                    dw,
                    soffset_bytes=soff,
                )

            # -- Pass 1: load input, accumulate sum / sumsq ---------------
            for tile_i in range_constexpr(num_tiles_py):
                col_bytes = ArithValue(thr_col_bytes) + (
                    tile_i * tile_cols * elem_bytes
                )
                vec_e = _load_vec_buf(in_rsrc, col_bytes, soff=row_soffset)

                if cache_as_elem:
                    in_local.append(vec_e)
                    x = vec_e.extf(vec_type_c)
                else:
                    x = vec_e
                    in_local.append(x)

                x_av = ArithValue(x)
                x2 = x_av * x_av
                red = vector.reduction(
                    compute_type,
                    vector.CombiningKind.ADD,
                    x,
                    fastmath=fm_fast,
                )
                red2 = vector.reduction(
                    compute_type,
                    vector.CombiningKind.ADD,
                    x2,
                    fastmath=fm_fast,
                )
                thread_sum = ArithValue(thread_sum) + red
                thread_sumsq = ArithValue(thread_sumsq) + red2

            sum_val, sumsq_val = block_reduce_add2(thread_sum, thread_sumsq)
            mean, rstd = compute_mean_rstd(sum_val, sumsq_val)

            mean_splat = vector.broadcast(vec_type_c, mean)
            rstd_splat = vector.broadcast(vec_type_c, rstd)
            mean_splat_av = ArithValue(mean_splat)
            rstd_splat_av = ArithValue(rstd_splat)

            g_e_cur = _load_vec_buf(gamma_rsrc, thr_col_bytes)
            b_e_cur = _load_vec_buf(beta_rsrc, thr_col_bytes)
            g_cur = g_e_cur if dtype_str == "f32" else g_e_cur.extf(vec_type_c)
            b_cur = b_e_cur if dtype_str == "f32" else b_e_cur.extf(vec_type_c)

            # -- Pass 2: normalize + affine + store -----------------------
            for tile_i in range_constexpr(num_tiles_py):
                if tile_i + 1 < num_tiles_py:
                    next_col_bytes = ArithValue(thr_col_bytes) + (
                        (tile_i + 1) * tile_cols * elem_bytes
                    )
                    g_e_next = _load_vec_buf(gamma_rsrc, next_col_bytes)
                    b_e_next = _load_vec_buf(beta_rsrc, next_col_bytes)
                    g_next = (
                        g_e_next if dtype_str == "f32" else g_e_next.extf(vec_type_c)
                    )
                    b_next = (
                        b_e_next if dtype_str == "f32" else b_e_next.extf(vec_type_c)
                    )
                else:
                    g_next = g_cur
                    b_next = b_cur

                x = in_local[tile_i]
                if cache_as_elem:
                    x = x.extf(vec_type_c)

                x_av = ArithValue(x)
                g_av = ArithValue(g_cur)
                b_av = ArithValue(b_cur)
                y = (x_av - mean_splat_av) * rstd_splat_av
                y = (y * g_av) + b_av
                y_val = y

                if dtype_str == "bf16":
                    if USE_HW_CVT_PK_BF16_F32:
                        out_e = y_val.truncf(vec_type_e)
                    else:
                        vec_i32_ty = T.vec(VEC_WIDTH, T.i32)
                        vec4_i32_ty = T.vec(VEC_WIDTH // 2, T.i32)
                        vec_bf16_ty = T.vec(VEC_WIDTH, elem_type)
                        c16_i32 = arith.constant(16, type=T.i32)
                        c16_v = vector.broadcast(vec_i32_ty, c16_i32)
                        u = y_val.bitcast(vec_i32_ty)
                        upper = u.shrui(c16_v)
                        c1_v = vector.broadcast(
                            vec_i32_ty, arith.constant(1, type=T.i32)
                        )
                        lsb = upper & c1_v
                        c7fff_v = vector.broadcast(
                            vec_i32_ty, arith.constant(0x7FFF, type=T.i32)
                        )
                        bias = ArithValue(c7fff_v) + lsb
                        u_round = u + bias
                        bf16_bits = u_round.shrui(c16_v)
                        even = vector.shuffle(bf16_bits, bf16_bits, [0, 2, 4, 6])
                        odd = vector.shuffle(bf16_bits, bf16_bits, [1, 3, 5, 7])
                        odd_sh = odd << vector.broadcast(vec4_i32_ty, c16_i32)
                        packed = even | odd_sh
                        out_e = vector.bitcast(vec_bf16_ty, packed)
                elif dtype_str == "f32":
                    out_e = y_val
                else:
                    out_e = y_val.truncf(vec_type_e)

                out_col_bytes = ArithValue(thr_col_bytes) + (
                    tile_i * tile_cols * elem_bytes
                )
                i32_vec_ty = T.vec(vec_dwords, T.i32)
                out_vec = (
                    vector.bitcast(i32_vec_ty, out_e)
                    if vec_dwords != VEC_WIDTH
                    else out_e.bitcast(i32_vec_ty)
                )
                _store_vec_buf(out_vec, out_rsrc, out_col_bytes, soff=row_soffset)

                g_cur = g_next
                b_cur = b_next

        else:
            # ==============================================================
            # Generic path: 2-pass scalar implementation for arbitrary N
            # ==============================================================
            from flydsl.expr.arith import ArithValue

            row_in = fx.slice(Input, (bid, None))
            row_out = fx.slice(Output, (bid, None))

            c_zero_f = arith.constant(0.0, type=compute_type)
            thread_sum = c_zero_f
            thread_sumsq = c_zero_f

            copy_atom_s = fx.make_copy_atom(fx.UniversalCopy(elem_bits), elem_bits)
            scalar_reg_ty = fx.MemRefType.get(
                elem_type, fx.LayoutType.get(1, 1), fx.AddressSpace.Register
            )
            scalar_reg_lay = fx.make_layout(1, 1)

            row_div = fx.logical_divide(row_in, fx.make_layout(1, 1))
            gamma_div = fx.logical_divide(Gamma, fx.make_layout(1, 1))
            beta_div = fx.logical_divide(Beta, fx.make_layout(1, 1))
            out_div = fx.logical_divide(row_out, fx.make_layout(1, 1))

            def _load_scalar(divided_tensor, index):
                view = fx.slice(divided_tensor, (None, index))
                r = fx.memref_alloca(scalar_reg_ty, scalar_reg_lay)
                fx.copy_atom_call(copy_atom_s, view, r)
                v = fx.memref_load_vec(r)
                return vector.extract(v, static_position=[0])

            def _store_scalar(divided_tensor, index, val):
                r = fx.memref_alloca(scalar_reg_ty, scalar_reg_lay)
                vec_ty = T.vec(1, elem_type)
                v = vector.from_elements(vec_ty, [val])
                fx.memref_store_vec(v, r)
                view = fx.slice(divided_tensor, (None, index))
                fx.copy_atom_call(copy_atom_s, r, view)

            # -- Pass 1: sum + sumsq --------------------------------------
            for base_idx_int in range_constexpr(0, N, BLOCK_THREADS):
                idx = tid + base_idx_int
                c_N_i32 = Int32(N)
                is_valid = idx < c_N_i32
                c0_i = Int32(0)
                idx_safe = is_valid.select(idx, c0_i)
                x_e = _load_scalar(row_div, idx_safe)
                x = x_e if dtype_str == "f32" else x_e.extf(compute_type)
                x_av = ArithValue(x)
                x2 = x_av * x_av
                x_safe = is_valid.select(x, c_zero_f)
                x2_safe = is_valid.select(x2, c_zero_f)
                thread_sum = ArithValue(thread_sum) + x_safe
                thread_sumsq = ArithValue(thread_sumsq) + x2_safe

            sum_val, sumsq_val = block_reduce_add2(thread_sum, thread_sumsq)
            mean, rstd = compute_mean_rstd(sum_val, sumsq_val)

            # -- Pass 2: normalize + affine + store -----------------------
            for base_idx_int in range_constexpr(0, N, BLOCK_THREADS):
                idx = tid + base_idx_int
                c_N_i32 = Int32(N)
                if arith.cmpi(arith.CmpIPredicate.ult, idx, c_N_i32):
                    x_e = _load_scalar(row_div, idx)
                    g_e = _load_scalar(gamma_div, idx)
                    b_e = _load_scalar(beta_div, idx)
                    x = x_e if dtype_str == "f32" else x_e.extf(compute_type)
                    g = g_e if dtype_str == "f32" else g_e.extf(compute_type)
                    b = b_e if dtype_str == "f32" else b_e.extf(compute_type)
                    diff = ArithValue(x) - mean
                    norm = diff * ArithValue(rstd)
                    scaled = norm * ArithValue(g)
                    y = scaled + ArithValue(b)
                    if dtype_str == "bf16":
                        y_e = y.truncf(elem_type)
                    elif dtype_str == "f32":
                        y_e = y
                    else:
                        y_e = y.truncf(elem_type)
                    _store_scalar(out_div, idx, y_e)

    # -- JIT host launcher -------------------------------------------------
    @flyc.jit
    def launch_layernorm(
        Input: fx.Tensor,
        Gamma: fx.Tensor,
        Beta: fx.Tensor,
        Output: fx.Tensor,
        m_in: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        allocator.finalized = False
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            allocator.finalize()

        idx_m = arith.index_cast(T.index, m_in)
        launcher = layernorm_kernel(Input, Gamma, Beta, Output)
        launcher.launch(
            grid=(idx_m, 1, 1),
            block=(BLOCK_THREADS, 1, 1),
            stream=stream,
        )

    return launch_layernorm
