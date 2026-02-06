#!/usr/bin/env python3
"""
Three standalone FlyDSL sub-kernels that decompose the Paged Attention kernel:
  1. QK MFMA:  Q[16,128] @ K[128,256] -> QK[16,256]  (f32 output)
  2. Softmax:  QK[16,256] -> P[16,256] (bf16), max_logits[16], exp_sums[16]
  3. PV MFMA:  P[16,256] @ V[256,128] -> O[16,128]  (f32 output)

Each is independently testable against a PyTorch reference.

Run:
  python3 aiter/ops/triton/gluon/test_pa_sub_kernels.py
"""

import functools
import math
import torch

import flydsl
from flydsl.dialects.ext import flir
from flydsl.dialects.ext.python_control_flow import range_constexpr
from flydsl.runtime.device import get_rocm_arch as get_hip_arch
from flydsl.utils import SmemAllocator

from _mlir import ir
from flydsl.dialects.ext import arith, gpu, buffer_ops, vector, rocdl, scf
from flydsl.lang.ir.types import T, memref

# ============================================================================
# Constants matching the PA kernel test case
# ============================================================================
M_DIM = 16       # query_groups
K_QK = 128       # head_size
N_QK = 256       # kv_tokens
K_PV = 256       # kv_tokens (K dim for PV)
N_PV = 128       # head_size (N dim for PV)

TOTAL_THREADS = 256
NUM_WAVES = 4
WAVE_SIZE = 64

KPACK = 8        # bf16: 8 elements per 16-byte load
ELEM_BYTES = 2   # bf16

QK_N_PER_WARP = N_QK // NUM_WAVES        # 64
QK_TILES_PER_WARP = QK_N_PER_WARP // 16  # 4
QK_K64_STEPS = K_QK * ELEM_BYTES // 64   # 4

PV_N_PER_WARP = N_PV // NUM_WAVES        # 32
PV_TILES_PER_WARP = PV_N_PER_WARP // 16  # 2
PV_K64_STEPS = K_PV * ELEM_BYTES // 64   # 8

DYN = ir.ShapedType.get_dynamic_size()

LOG2_E = 1.4426950408889634

# ============================================================================
# Sub-kernel 1: QK MFMA
# ============================================================================

@functools.lru_cache(maxsize=8)
def compile_qk_mfma_kernel():
    """Compile QK MFMA: Q[16,128] bf16 @ K[128,256] bf16 -> QK[16,256] f32."""
    gpu_arch = get_hip_arch()
    allocator = SmemAllocator(None, arch=gpu_arch)
    _state = {}

    q_lds_elems = M_DIM * K_QK   # 16 * 128 = 2048

    class _QK_MFMA(flir.MlirModule):
        GPU_MODULE_NAME = "qk_mfma_test"
        GPU_MODULE_TARGETS = [
            f'#rocdl.target<chip = "{gpu_arch}", abi = "500", features = "+sramecc,+xnack">'
        ]

        def init_gpu_module(self):
            _state["lds_q"] = allocator.allocate_array(T.bf16, q_lds_elems)
            allocator.finalize()

        @flir.kernel
        def qk_kernel(
            self: flir.T.i64,
            arg_qk_out: lambda: memref(DYN, T.f32),
            arg_q:      lambda: memref(DYN, T.bf16),
            arg_k:      lambda: memref(DYN, T.bf16),
        ):
            f32 = T.f32; i32 = T.i32; i64 = T.i64; idx = T.index
            vec4_f32 = T.f32x4; vec4_i16 = T.i16x4
            vec8_bf16 = T.bf16x8; vec1_i64 = T.vec(1, i64); vec2_i64 = T.vec(2, i64)

            acc_zero = arith.constant_vector(0.0, vec4_f32)

            out_rsrc = buffer_ops.create_buffer_resource(arg_qk_out, max_size=True)
            q_rsrc   = buffer_ops.create_buffer_resource(arg_q,      max_size=True)
            k_rsrc   = buffer_ops.create_buffer_resource(arg_k,      max_size=True)

            base_ptr = allocator.get_base()
            lds_q = _state["lds_q"](base_ptr).get()

            c_one_idx = arith.constant(1, index=True)
            lds_q_stride = arith.constant(K_QK, index=True)
            shape_lds_q = flir.make_shape(M_DIM, K_QK)
            stride_lds_q = flir.make_stride(lds_q_stride, c_one_idx)
            layout_lds_q = flir.make_layout(shape_lds_q, stride_lds_q)
            q_k_bytes = K_QK * ELEM_BYTES
            k_blocks16_q = arith.constant(q_k_bytes // 16, index=True)

            tx = gpu.thread_id("x")
            wave_lane_layout = flir.make_layout((NUM_WAVES, WAVE_SIZE), stride=(WAVE_SIZE, 1))
            coord_wl = flir.idx2crd(tx, wave_lane_layout)
            wave_id  = flir.get(coord_wl, 0)
            lane_id  = flir.get(coord_wl, 1)
            lane16_layout = flir.make_layout((4, 16), stride=(16, 1))
            coord_l16 = flir.idx2crd(lane_id, lane16_layout)
            lane_div_16 = flir.get(coord_l16, 0)
            lane_mod_16 = flir.get(coord_l16, 1)

            c0 = arith.constant(0, index=True)
            c1 = arith.constant(1, index=True)
            c2 = arith.constant(2, index=True)
            c4 = arith.constant(4, index=True)
            c8 = arith.constant(8, index=True)
            c_eb = arith.constant(ELEM_BYTES, index=True)

            c0_i32 = arith.constant(0, type=i32)

            atom_q_lds = flir.make_copy_atom(T.bf16, vector_size=KPACK)

            mfma_fn = rocdl.mfma_f32_16x16x16bf16_1k

            def mfma_step(acc_in, a_op, b_op):
                return mfma_fn(vec4_f32, [a_op, b_op, acc_in, 0, 0, 0])

            def mfma_k64(acc_in, a0, a1, b0, b1):
                mid = mfma_step(acc_in, a0, b0)
                return mfma_step(mid, a1, b1)

            def to_mfma_operand(vec8_val):
                v_i64x2 = vector.bitcast(vec2_i64, vec8_val)
                h0 = vector.extract(v_i64x2, static_position=[0], dynamic_position=[])
                h1 = vector.extract(v_i64x2, static_position=[1], dynamic_position=[])
                h0_v1 = vector.from_elements(vec1_i64, [h0])
                h1_v1 = vector.from_elements(vec1_i64, [h1])
                return vector.bitcast(vec4_i16, h0_v1), vector.bitcast(vec4_i16, h1_v1)

            # ---- Load Q to LDS ----
            q_tile_k_dwords = K_QK * ELEM_BYTES // 4  # 64
            q_tile_layout = flir.make_layout(
                (M_DIM, q_tile_k_dwords), stride=(q_tile_k_dwords, 1))
            tx_dw_base = tx * c4
            coord_qtile = flir.idx2crd(tx_dw_base, q_tile_layout)
            row_q = flir.get(coord_qtile, 0)
            col_q_dw = flir.get(coord_qtile, 1)
            col_q_elem = col_q_dw * c2

            # Q is [16, 128] bf16, row-major, stride = 128
            q_off_elem = row_q * arith.constant(K_QK, index=True) + col_q_elem
            q_off_dw = q_off_elem / c2
            q_off_i32 = arith.index_cast(i32, q_off_dw)

            q_vec = buffer_ops.buffer_load(q_rsrc, q_off_i32, vec_width=4, dtype=i32)
            q_v8 = vector.bitcast(vec8_bf16, q_vec)

            col_q_bytes = col_q_dw * c4
            col_q_swz = flir.swizzle_xor16(row_q, col_q_bytes, k_blocks16_q)
            col_q_swz_e = col_q_swz / c_eb
            q_store_coord = flir.make_coord(row_q, col_q_swz_e)
            q_store_idx = flir.crd2idx(q_store_coord, layout_lds_q)
            s_view_q = flir.TensorView(
                lds_q, (KPACK,), strides=(1,),
                base_indices=(q_store_idx,), element_type=T.bf16)
            flir.copy(atom_q_lds, q_v8, s_view_q, alignment=16)

            gpu.barrier()

            # ---- QK MFMA ----
            qk_accs = [acc_zero] * QK_TILES_PER_WARP
            col_off_base_bytes = lane_div_16 * arith.constant(KPACK * ELEM_BYTES, index=True)

            # K is stored as [K_QK//8, N_QK, 8] = [16, 256, 8] bf16 row-major
            # stride_hsplit = N_QK * KPACK = 2048, stride_kv = KPACK = 8
            c_stride_k_hsplit = arith.constant(N_QK * KPACK, index=True)
            c_stride_k_kv = arith.constant(KPACK, index=True)

            for ku in range_constexpr(QK_K64_STEPS):
                ku_byte_off = arith.constant(ku * 64, index=True)
                col_base_q = col_off_base_bytes + ku_byte_off
                col_swz_q = flir.swizzle_xor16(lane_mod_16, col_base_q, k_blocks16_q)
                col_swz_q_e = col_swz_q / c_eb
                coord_a = flir.make_coord(lane_mod_16, col_swz_q_e)
                idx_a = flir.crd2idx(coord_a, layout_lds_q)
                loaded_q = vector.load_op(vec8_bf16, lds_q, [idx_a])
                q_a0, q_a1 = to_mfma_operand(loaded_q)

                head_split_base = arith.constant(ku * (32 // KPACK), index=True)
                head_split = head_split_base + lane_div_16

                for ni in range_constexpr(QK_TILES_PER_WARP):
                    kv_col = (wave_id * arith.constant(QK_N_PER_WARP, index=True)
                              + arith.constant(ni * 16, index=True)
                              + lane_mod_16)
                    # K offset: head_split * stride_hsplit + kv_col * stride_kv
                    k_elem_off = head_split * c_stride_k_hsplit + kv_col * c_stride_k_kv
                    k_dw_off = k_elem_off / c2
                    k_off_i32 = arith.index_cast(i32, k_dw_off)

                    k_vec = buffer_ops.buffer_load(k_rsrc, k_off_i32, vec_width=4, dtype=i32)
                    k_v8 = vector.bitcast(vec8_bf16, k_vec)
                    k_b0, k_b1 = to_mfma_operand(k_v8)
                    qk_accs[ni] = mfma_k64(qk_accs[ni], q_a0, q_a1, k_b0, k_b1)

            # ---- Store QK output [16, 256] f32 ----
            c_qk_stride_row = arith.constant(N_QK, index=True)
            for ni in range_constexpr(QK_TILES_PER_WARP):
                for ei in range_constexpr(4):
                    val = vector.extract(qk_accs[ni], static_position=[ei], dynamic_position=[])
                    row = lane_div_16 * c4 + arith.constant(ei, index=True)
                    col = (wave_id * arith.constant(QK_N_PER_WARP, index=True)
                           + arith.constant(ni * 16, index=True) + lane_mod_16)
                    out_off = row * c_qk_stride_row + col
                    out_off_i32 = arith.index_cast(i32, out_off)
                    buffer_ops.buffer_store(val, out_rsrc, out_off_i32)

        @flir.jit
        def __call__(
            self: flir.T.i64,
            arg_qk_out: lambda: memref(DYN, T.f32),
            arg_q:      lambda: memref(DYN, T.bf16),
            arg_k:      lambda: memref(DYN, T.bf16),
        ):
            c1 = arith.constant(1, index=True)
            bdx = arith.constant(TOTAL_THREADS, index=True)
            flir.gpu_ext.LaunchFuncOp(
                ["qk_mfma_test", "qk_kernel"],
                grid_size=(c1, c1, c1),
                block_size=(bdx, c1, c1),
                kernel_operands=[arg_qk_out, arg_q, arg_k],
            )

    m = _QK_MFMA()
    return flydsl.compile(m, waves_per_eu=1)


# ============================================================================
# Sub-kernel 2: Softmax
# ============================================================================

@functools.lru_cache(maxsize=8)
def compile_softmax_kernel():
    """Compile softmax: QK[16,256] f32 -> P[16,256] bf16 + max_logits[16] + exp_sums[16]."""
    gpu_arch = get_hip_arch()
    allocator = SmemAllocator(None, arch=gpu_arch)
    _state = {}

    red_lds_elems = NUM_WAVES * M_DIM  # 4*16 = 64

    class _Softmax(flir.MlirModule):
        GPU_MODULE_NAME = "softmax_test"
        GPU_MODULE_TARGETS = [
            f'#rocdl.target<chip = "{gpu_arch}", abi = "500", features = "+sramecc,+xnack">'
        ]

        def init_gpu_module(self):
            _state["lds_red"] = allocator.allocate_array(T.f32, red_lds_elems)
            allocator.finalize()

        @flir.kernel
        def softmax_kernel(
            self: flir.T.i64,
            arg_p_out:      lambda: memref(DYN, T.bf16),
            arg_ml_out:     lambda: memref(DYN, T.f32),
            arg_es_out:     lambda: memref(DYN, T.f32),
            arg_qk_in:      lambda: memref(DYN, T.f32),
            c_softmax_scale: lambda: T.f32,
            c_context_len:  lambda: T.index,
        ):
            f32 = T.f32; i32 = T.i32; idx = T.index

            p_rsrc  = buffer_ops.create_buffer_resource(arg_p_out, max_size=True)
            ml_rsrc = buffer_ops.create_buffer_resource(arg_ml_out, max_size=True)
            es_rsrc = buffer_ops.create_buffer_resource(arg_es_out, max_size=True)
            qk_rsrc = buffer_ops.create_buffer_resource(arg_qk_in, max_size=True)

            base_ptr = allocator.get_base()
            lds_red = _state["lds_red"](base_ptr).get()

            c_one_idx = arith.constant(1, index=True)
            red_stride = arith.constant(M_DIM, index=True)
            shape_red = flir.make_shape(NUM_WAVES, M_DIM)
            stride_red = flir.make_stride(red_stride, c_one_idx)
            layout_red = flir.make_layout(shape_red, stride_red)

            tx = gpu.thread_id("x")
            wave_lane_layout = flir.make_layout((NUM_WAVES, WAVE_SIZE), stride=(WAVE_SIZE, 1))
            coord_wl = flir.idx2crd(tx, wave_lane_layout)
            wave_id  = flir.get(coord_wl, 0)
            lane_id  = flir.get(coord_wl, 1)
            lane16_layout = flir.make_layout((4, 16), stride=(16, 1))
            coord_l16 = flir.idx2crd(lane_id, lane16_layout)
            lane_div_16 = flir.get(coord_l16, 0)
            lane_mod_16 = flir.get(coord_l16, 1)

            c0 = arith.constant(0, index=True)
            c4 = arith.constant(4, index=True)
            c0_i32 = arith.constant(0, type=i32)
            c64_i32 = arith.constant(64, type=i32)
            c1_i32 = arith.constant(1, type=i32)
            c2_i32 = arith.constant(2, type=i32)
            c4_i32 = arith.constant(4, type=i32)
            c8_i32 = arith.constant(8, type=i32)

            fm_fast = flir.arith.FastMathFlags.fast

            def _shuffle_xor(w, offset):
                raw = gpu.ShuffleOp(
                    arith.as_value(w), arith.as_value(offset),
                    arith.as_value(c64_i32), mode="xor").shuffleResult
                return arith.ArithValue(raw)

            def warp_reduce_max(val):
                w = val
                for off in [c1_i32, c2_i32, c4_i32, c8_i32]:
                    peer = _shuffle_xor(w, off)
                    w = arith.maximum(w, peer)
                return w

            def warp_reduce_sum(val):
                w = val
                for off in [c1_i32, c2_i32, c4_i32, c8_i32]:
                    peer = _shuffle_xor(w, off)
                    w = w + peer
                return w

            def block_reduce_per_row(val, row_idx, op="max"):
                if op == "max":
                    val = warp_reduce_max(val)
                else:
                    val = warp_reduce_sum(val)
                is_lane0 = arith.cmpu(lane_mod_16, c0, "eq")
                red_coord = flir.make_coord(wave_id, row_idx)
                red_idx = flir.crd2idx(red_coord, layout_red)
                if_op = scf.IfOp(is_lane0)
                with if_op:
                    flir.memref.store(arith.as_value(val), lds_red, [arith.as_value(red_idx)])
                gpu.barrier()
                result = val
                for wi in range_constexpr(NUM_WAVES):
                    wi_idx = arith.constant(wi, index=True)
                    rd_coord = flir.make_coord(wi_idx, row_idx)
                    rd_idx = flir.crd2idx(rd_coord, layout_red)
                    rd_val = flir.memref.load(lds_red, [arith.as_value(rd_idx)])
                    if wi == 0:
                        result = rd_val
                    else:
                        if op == "max":
                            result = arith.maximum(result, rd_val)
                        else:
                            result = result + rd_val
                gpu.barrier()
                return result

            # ---- Load QK scores from global memory ----
            # QK is [16, 256] f32 row-major
            c_qk_stride = arith.constant(N_QK, index=True)
            c_negbig = arith.constant(-3.4e38, type=f32)
            c_log2e = arith.constant(LOG2_E, type=f32)

            scores_pr = [[None] * QK_TILES_PER_WARP for _ in range(4)]
            for ni in range_constexpr(QK_TILES_PER_WARP):
                for ei in range_constexpr(4):
                    row = lane_div_16 * c4 + arith.constant(ei, index=True)
                    col = (wave_id * arith.constant(QK_N_PER_WARP, index=True)
                           + arith.constant(ni * 16, index=True) + lane_mod_16)
                    qk_off = row * c_qk_stride + col
                    qk_off_i32 = arith.index_cast(i32, qk_off)
                    s = buffer_ops.buffer_load(qk_rsrc, qk_off_i32, vec_width=1, dtype=f32)

                    # Scale
                    s = s * c_softmax_scale

                    # Boundary mask
                    valid_col = arith.cmpu(col, c_context_len, "ult")
                    row_ok = arith.cmpu(row, arith.constant(M_DIM, index=True), "ult")
                    full_mask = arith.andi(row_ok, valid_col)
                    s = arith.select(full_mask, s, c_negbig)
                    scores_pr[ei][ni] = s

            # ---- Online softmax per row ----
            max_per_ei = [None] * 4
            sum_per_ei = [None] * 4

            for ei in range_constexpr(4):
                row_idx = lane_div_16 * c4 + arith.constant(ei, index=True)

                # Local max across tiles
                local_max = scores_pr[ei][0]
                for ni in range_constexpr(QK_TILES_PER_WARP):
                    if ni > 0:
                        local_max = arith.maximum(local_max, scores_pr[ei][ni])
                global_max = block_reduce_per_row(local_max, row_idx, op="max")
                max_per_ei[ei] = global_max

                # exp2 and sum
                local_sum = arith.constant(0.0, type=f32)
                for ni in range_constexpr(QK_TILES_PER_WARP):
                    s = scores_pr[ei][ni]
                    sub = s - global_max
                    sub_sc = sub * c_log2e
                    prob = flir.math.exp2(arith.as_value(sub_sc), fastmath=fm_fast)
                    prob = arith.ArithValue(prob) if not isinstance(prob, arith.ArithValue) else prob
                    scores_pr[ei][ni] = prob
                    local_sum = local_sum + prob
                global_sum = block_reduce_per_row(local_sum, row_idx, op="sum")
                sum_per_ei[ei] = global_sum

            # ---- Store P (bf16), max_logits, exp_sums ----
            c_p_stride = arith.constant(N_QK, index=True)
            for ni in range_constexpr(QK_TILES_PER_WARP):
                for ei in range_constexpr(4):
                    prob_val = scores_pr[ei][ni]
                    prob_bf16 = arith.trunc_f(T.bf16, prob_val)
                    row = lane_div_16 * c4 + arith.constant(ei, index=True)
                    col = (wave_id * arith.constant(QK_N_PER_WARP, index=True)
                           + arith.constant(ni * 16, index=True) + lane_mod_16)
                    p_off = row * c_p_stride + col
                    p_off_i32 = arith.index_cast(i32, p_off)
                    buffer_ops.buffer_store(prob_bf16, p_rsrc, p_off_i32)

            # max_logits and exp_sums: one writer per row
            is_writer = arith.andi(
                arith.cmpu(wave_id, c0, "eq"),
                arith.cmpu(lane_mod_16, c0, "eq"))
            for ei in range_constexpr(4):
                row = lane_div_16 * c4 + arith.constant(ei, index=True)
                row_ok = arith.cmpu(row, arith.constant(M_DIM, index=True), "ult")
                mask = arith.andi(is_writer, row_ok)
                row_i32 = arith.index_cast(i32, row)
                buffer_ops.buffer_store(max_per_ei[ei], ml_rsrc, row_i32, mask=mask)
                buffer_ops.buffer_store(sum_per_ei[ei], es_rsrc, row_i32, mask=mask)

        @flir.jit
        def __call__(
            self: flir.T.i64,
            arg_p_out:      lambda: memref(DYN, T.bf16),
            arg_ml_out:     lambda: memref(DYN, T.f32),
            arg_es_out:     lambda: memref(DYN, T.f32),
            arg_qk_in:      lambda: memref(DYN, T.f32),
            c_softmax_scale: lambda: T.f32,
            c_context_len:  lambda: T.index,
        ):
            c1 = arith.constant(1, index=True)
            bdx = arith.constant(TOTAL_THREADS, index=True)
            flir.gpu_ext.LaunchFuncOp(
                ["softmax_test", "softmax_kernel"],
                grid_size=(c1, c1, c1),
                block_size=(bdx, c1, c1),
                kernel_operands=[arg_p_out, arg_ml_out, arg_es_out, arg_qk_in,
                                 c_softmax_scale, c_context_len],
            )

    m = _Softmax()
    return flydsl.compile(m, waves_per_eu=1)


# ============================================================================
# Sub-kernel 3: PV MFMA
# ============================================================================

@functools.lru_cache(maxsize=8)
def compile_pv_mfma_kernel():
    """Compile PV MFMA: P[16,256] bf16 @ V[256,128] bf16 -> O[16,128] f32."""
    gpu_arch = get_hip_arch()
    allocator = SmemAllocator(None, arch=gpu_arch)
    _state = {}

    p_lds_elems = M_DIM * K_PV  # 16 * 256 = 4096

    class _PV_MFMA(flir.MlirModule):
        GPU_MODULE_NAME = "pv_mfma_test"
        GPU_MODULE_TARGETS = [
            f'#rocdl.target<chip = "{gpu_arch}", abi = "500", features = "+sramecc,+xnack">'
        ]

        def init_gpu_module(self):
            _state["lds_p"] = allocator.allocate_array(T.bf16, p_lds_elems)
            allocator.finalize()

        @flir.kernel
        def pv_kernel(
            self: flir.T.i64,
            arg_o_out: lambda: memref(DYN, T.f32),
            arg_p:     lambda: memref(DYN, T.bf16),
            arg_v:     lambda: memref(DYN, T.bf16),
        ):
            f32 = T.f32; i32 = T.i32; i64 = T.i64; idx = T.index
            vec4_f32 = T.f32x4; vec4_i16 = T.i16x4
            vec8_bf16 = T.bf16x8; vec1_i64 = T.vec(1, i64); vec2_i64 = T.vec(2, i64)

            acc_zero = arith.constant_vector(0.0, vec4_f32)

            o_rsrc = buffer_ops.create_buffer_resource(arg_o_out, max_size=True)
            p_rsrc = buffer_ops.create_buffer_resource(arg_p, max_size=True)
            v_rsrc = buffer_ops.create_buffer_resource(arg_v, max_size=True)

            base_ptr = allocator.get_base()
            lds_p = _state["lds_p"](base_ptr).get()

            c_one_idx = arith.constant(1, index=True)
            lds_p_stride = arith.constant(K_PV, index=True)
            shape_lds_p = flir.make_shape(M_DIM, K_PV)
            stride_lds_p = flir.make_stride(lds_p_stride, c_one_idx)
            layout_lds_p = flir.make_layout(shape_lds_p, stride_lds_p)

            tx = gpu.thread_id("x")
            wave_lane_layout = flir.make_layout((NUM_WAVES, WAVE_SIZE), stride=(WAVE_SIZE, 1))
            coord_wl = flir.idx2crd(tx, wave_lane_layout)
            wave_id  = flir.get(coord_wl, 0)
            lane_id  = flir.get(coord_wl, 1)
            lane16_layout = flir.make_layout((4, 16), stride=(16, 1))
            coord_l16 = flir.idx2crd(lane_id, lane16_layout)
            lane_div_16 = flir.get(coord_l16, 0)
            lane_mod_16 = flir.get(coord_l16, 1)

            c0 = arith.constant(0, index=True)
            c2 = arith.constant(2, index=True)
            c4 = arith.constant(4, index=True)
            c8 = arith.constant(8, index=True)

            mfma_fn = rocdl.mfma_f32_16x16x16bf16_1k

            def mfma_step(acc_in, a_op, b_op):
                return mfma_fn(vec4_f32, [a_op, b_op, acc_in, 0, 0, 0])

            def mfma_k64(acc_in, a0, a1, b0, b1):
                mid = mfma_step(acc_in, a0, b0)
                return mfma_step(mid, a1, b1)

            def to_mfma_operand(vec8_val):
                vec2_i64_ty = T.vec(2, T.i64)
                vec1_i64_ty = T.vec(1, T.i64)
                v_i64x2 = vector.bitcast(vec2_i64_ty, vec8_val)
                h0 = vector.extract(v_i64x2, static_position=[0], dynamic_position=[])
                h1 = vector.extract(v_i64x2, static_position=[1], dynamic_position=[])
                h0_v1 = vector.from_elements(vec1_i64_ty, [h0])
                h1_v1 = vector.from_elements(vec1_i64_ty, [h1])
                return vector.bitcast(vec4_i16, h0_v1), vector.bitcast(vec4_i16, h1_v1)

            # ---- Load P to LDS ----
            # P is [16, 256] bf16 row-major. Each thread stores its elements to LDS.
            # We use cooperative loading: 256 threads, 16*256=4096 elements, 16 per thread
            p_total_dwords = M_DIM * K_PV * ELEM_BYTES // 4  # 4096*2/4 = 2048
            p_loads_per_thread = p_total_dwords // TOTAL_THREADS  # 2048/256 = 8
            # Each thread loads 8 dwords = 32 bytes = 16 bf16

            p_flat_layout = flir.make_layout(
                (M_DIM, K_PV // 2), stride=(K_PV // 2, 1))  # dword-addressed

            for load_i in range_constexpr(p_loads_per_thread // 4):
                chunk_off = arith.constant(load_i * TOTAL_THREADS * 4, index=True)
                dw_idx = tx * c4 + chunk_off
                coord_p_tile = flir.idx2crd(dw_idx, p_flat_layout)
                p_row = flir.get(coord_p_tile, 0)
                p_col_dw = flir.get(coord_p_tile, 1)
                p_col_elem = p_col_dw * c2
                p_off_elem = p_row * arith.constant(K_PV, index=True) + p_col_elem
                p_off_dw = p_off_elem / c2
                p_off_i32 = arith.index_cast(T.i32, p_off_dw)

                p_vec = buffer_ops.buffer_load(p_rsrc, p_off_i32, vec_width=4, dtype=T.i32)
                p_v8 = vector.bitcast(vec8_bf16, p_vec)

                # Store to LDS at [p_row, p_col_elem] (no swizzle for P)
                p_store_coord = flir.make_coord(p_row, p_col_elem)
                p_store_idx = flir.crd2idx(p_store_coord, layout_lds_p)
                atom_p = flir.make_copy_atom(T.bf16, vector_size=KPACK)
                s_view_p = flir.TensorView(
                    lds_p, (KPACK,), strides=(1,),
                    base_indices=(p_store_idx,), element_type=T.bf16)
                flir.copy(atom_p, p_v8, s_view_p, alignment=16)

            gpu.barrier()

            # ---- Pre-load V ----
            # V is stored as [N_PV, K_PV] = [128, 256] bf16 row-major
            # B operand: V[head_col, kv_token]
            # stride_v_col = K_PV = 256
            c_stride_v_col = arith.constant(K_PV, index=True)

            v_tile_data = []
            for ku_v in range_constexpr(PV_K64_STEPS):
                v_intra_base = arith.constant(ku_v * 32, index=True)
                v_intra_k = v_intra_base + lane_div_16 * c8

                ku_v_packs = []
                for ni_v in range_constexpr(PV_TILES_PER_WARP):
                    v_col = (wave_id * arith.constant(PV_N_PER_WARP, index=True)
                             + arith.constant(ni_v * 16, index=True)
                             + lane_mod_16)
                    # V[v_col, v_intra_k:v_intra_k+8]
                    v_elem_off = v_col * c_stride_v_col + v_intra_k
                    v_dw_off = v_elem_off / c2
                    v_off_i32 = arith.index_cast(T.i32, v_dw_off)
                    v_vec = buffer_ops.buffer_load(v_rsrc, v_off_i32, vec_width=4, dtype=T.i32)
                    v_v8 = vector.bitcast(vec8_bf16, v_vec)
                    vb0, vb1 = to_mfma_operand(v_v8)
                    ku_v_packs.append((vb0, vb1))
                v_tile_data.append(ku_v_packs)

            # ---- PV MFMA ----
            pv_accs = [acc_zero] * PV_TILES_PER_WARP

            for ku_v in range_constexpr(PV_K64_STEPS):
                k_base_p = arith.constant(ku_v * 32, index=True)
                p_col_start = k_base_p + lane_div_16 * c8
                p_coord_lds = flir.make_coord(lane_mod_16, p_col_start)
                p_idx_lds = flir.crd2idx(p_coord_lds, layout_lds_p)
                loaded_p = vector.load_op(vec8_bf16, lds_p, [p_idx_lds])
                p_a0, p_a1 = to_mfma_operand(loaded_p)

                for ni_v in range_constexpr(PV_TILES_PER_WARP):
                    vb0, vb1 = v_tile_data[ku_v][ni_v]
                    pv_accs[ni_v] = mfma_k64(pv_accs[ni_v], p_a0, p_a1, vb0, vb1)

            # ---- Store O [16, 128] f32 ----
            c_o_stride = arith.constant(N_PV, index=True)
            for ni in range_constexpr(PV_TILES_PER_WARP):
                for ei in range_constexpr(4):
                    val = vector.extract(pv_accs[ni], static_position=[ei], dynamic_position=[])
                    row = lane_div_16 * c4 + arith.constant(ei, index=True)
                    col = (wave_id * arith.constant(PV_N_PER_WARP, index=True)
                           + arith.constant(ni * 16, index=True) + lane_mod_16)
                    out_off = row * c_o_stride + col
                    out_off_i32 = arith.index_cast(T.i32, out_off)
                    buffer_ops.buffer_store(val, o_rsrc, out_off_i32)

        @flir.jit
        def __call__(
            self: flir.T.i64,
            arg_o_out: lambda: memref(DYN, T.f32),
            arg_p:     lambda: memref(DYN, T.bf16),
            arg_v:     lambda: memref(DYN, T.bf16),
        ):
            c1 = arith.constant(1, index=True)
            bdx = arith.constant(TOTAL_THREADS, index=True)
            flir.gpu_ext.LaunchFuncOp(
                ["pv_mfma_test", "pv_kernel"],
                grid_size=(c1, c1, c1),
                block_size=(bdx, c1, c1),
                kernel_operands=[arg_o_out, arg_p, arg_v],
            )

    m = _PV_MFMA()
    return flydsl.compile(m, waves_per_eu=1)


# ============================================================================
# Test functions
# ============================================================================

def test_qk_mfma():
    """Test QK MFMA sub-kernel against torch.matmul."""
    print("=" * 60)
    print("TEST 1: QK MFMA  Q[16,128] @ K[128,256] -> QK[16,256]")
    print("=" * 60)

    device = "cuda"
    torch.manual_seed(42)

    # Generate random inputs
    Q = torch.randn(M_DIM, K_QK, dtype=torch.bfloat16, device=device)
    # K_ref: [K_QK, N_QK] = [128, 256] -- each column is a KV token
    K_ref = torch.randn(K_QK, N_QK, dtype=torch.bfloat16, device=device)

    # Reference: Q @ K = [16, 256]
    ref = torch.matmul(Q.float(), K_ref.float())

    # Reformat K for the kernel: [K_QK//8, N_QK, 8]
    # K_cache[hs, kv, e] = K_ref[hs*8+e, kv]
    K_cache = K_ref.view(K_QK // KPACK, KPACK, N_QK).permute(0, 2, 1).contiguous()
    # K_cache shape: [16, 256, 8]

    # Output
    QK_out = torch.zeros(M_DIM, N_QK, dtype=torch.float32, device=device)

    # Compile and run
    exe = compile_qk_mfma_kernel()
    exe(
        QK_out.view(-1),
        Q.contiguous().view(-1),
        K_cache.contiguous().view(-1),
    )
    torch.cuda.synchronize()

    # Compare
    diff = (QK_out - ref).abs()
    print(f"  diff.abs.max  = {diff.max().item():.6e}")
    print(f"  diff.abs.mean = {diff.mean().item():.6e}")
    print(f"  ref.abs.max   = {ref.abs().max().item():.6e}")
    rel_err = diff.max().item() / (ref.abs().max().item() + 1e-12)
    print(f"  relative max  = {rel_err:.6e}")
    passed = diff.max().item() < 1e-1  # bf16 matmul tolerance
    print(f"  PASSED: {passed}")
    return passed, diff.max().item()


def test_softmax():
    """Test softmax sub-kernel against PyTorch reference."""
    print("=" * 60)
    print("TEST 2: Softmax  QK[16,256] -> P[16,256] bf16")
    print("=" * 60)

    device = "cuda"
    torch.manual_seed(42)

    context_len = N_QK  # all 256 tokens valid
    softmax_scale = 1.0 / math.sqrt(K_QK)

    # Generate random QK scores
    QK = torch.randn(M_DIM, N_QK, dtype=torch.float32, device=device)

    # Reference softmax (unnormalized -- kernel stores raw exp2 values, not divided by sum)
    scaled = QK * softmax_scale
    max_vals = scaled.max(dim=1, keepdim=True).values
    shifted = (scaled - max_vals) * LOG2_E
    probs_f32 = torch.pow(2.0, shifted)
    sum_vals = probs_f32.sum(dim=1, keepdim=True)
    P_ref_bf16 = probs_f32.to(torch.bfloat16)
    ml_ref = max_vals.squeeze(1)
    es_ref = sum_vals.squeeze(1)

    # Output tensors
    P_out = torch.zeros(M_DIM, N_QK, dtype=torch.bfloat16, device=device)
    ml_out = torch.zeros(M_DIM, dtype=torch.float32, device=device)
    es_out = torch.zeros(M_DIM, dtype=torch.float32, device=device)

    # Compile and run
    exe = compile_softmax_kernel()
    exe(
        P_out.view(-1),
        ml_out.view(-1),
        es_out.view(-1),
        QK.contiguous().view(-1),
        softmax_scale,
        context_len,
    )
    torch.cuda.synchronize()

    # Compare P
    diff_p = (P_out.float() - P_ref_bf16.float()).abs()
    print(f"  P diff.abs.max  = {diff_p.max().item():.6e}")
    print(f"  P diff.abs.mean = {diff_p.mean().item():.6e}")

    # Compare max_logits
    diff_ml = (ml_out - ml_ref).abs()
    print(f"  max_logits diff.abs.max = {diff_ml.max().item():.6e}")

    # Compare exp_sums
    diff_es = (es_out - es_ref).abs()
    print(f"  exp_sums diff.abs.max   = {diff_es.max().item():.6e}")

    passed = diff_p.max().item() < 1e-2  # bf16 softmax tolerance
    print(f"  PASSED: {passed}")
    return passed, diff_p.max().item()


def test_pv_mfma():
    """Test PV MFMA sub-kernel against torch.matmul."""
    print("=" * 60)
    print("TEST 3: PV MFMA  P[16,256] @ V[256,128] -> O[16,128]")
    print("=" * 60)

    device = "cuda"
    torch.manual_seed(42)

    # Generate random inputs
    P = torch.randn(M_DIM, K_PV, dtype=torch.bfloat16, device=device)
    V_ref = torch.randn(K_PV, N_PV, dtype=torch.bfloat16, device=device)

    # Reference: P @ V = [16, 128]
    ref = torch.matmul(P.float(), V_ref.float())

    # Reformat V for the kernel: V_test[head_col, kv_token] = V_ref[kv_token, head_col]
    # V_test = V_ref.T = [128, 256] bf16
    V_test = V_ref.T.contiguous()

    # Output
    O_out = torch.zeros(M_DIM, N_PV, dtype=torch.float32, device=device)

    # Compile and run
    exe = compile_pv_mfma_kernel()
    exe(
        O_out.view(-1),
        P.contiguous().view(-1),
        V_test.contiguous().view(-1),
    )
    torch.cuda.synchronize()

    # Compare
    diff = (O_out - ref).abs()
    print(f"  diff.abs.max  = {diff.max().item():.6e}")
    print(f"  diff.abs.mean = {diff.mean().item():.6e}")
    print(f"  ref.abs.max   = {ref.abs().max().item():.6e}")
    rel_err = diff.max().item() / (ref.abs().max().item() + 1e-12)
    print(f"  relative max  = {rel_err:.6e}")
    passed = diff.max().item() < 1e-1  # bf16 matmul tolerance
    print(f"  PASSED: {passed}")
    return passed, diff.max().item()


# ============================================================================
# Sub-kernel 4: Integration (QK -> Softmax -> PV in one kernel)
# ============================================================================

@functools.lru_cache(maxsize=8)
def compile_integrated_kernel():
    """Compile integrated QK->softmax->PV kernel matching main PA kernel pattern."""
    gpu_arch = get_hip_arch()
    allocator = SmemAllocator(None, arch=gpu_arch)
    _state = {}

    q_lds_elems = M_DIM * K_QK
    p_lds_elems = M_DIM * K_PV
    red_lds_elems = NUM_WAVES * M_DIM

    class _Integrated(flir.MlirModule):
        GPU_MODULE_NAME = "integrated_test"
        GPU_MODULE_TARGETS = [
            f'#rocdl.target<chip = "{gpu_arch}", abi = "500", features = "+sramecc,+xnack">'
        ]

        def init_gpu_module(self):
            _state["lds_q"] = allocator.allocate_array(T.bf16, q_lds_elems)
            _state["lds_p"] = allocator.allocate_array(T.bf16, p_lds_elems)
            _state["lds_red"] = allocator.allocate_array(T.f32, red_lds_elems)
            allocator.finalize()

        @flir.kernel
        def integrated_kernel(
            self: flir.T.i64,
            arg_o_out:      lambda: memref(DYN, T.bf16),
            arg_ml_out:     lambda: memref(DYN, T.f32),
            arg_es_out:     lambda: memref(DYN, T.f32),
            arg_q:          lambda: memref(DYN, T.bf16),
            arg_k:          lambda: memref(DYN, T.bf16),
            arg_v:          lambda: memref(DYN, T.bf16),
            c_softmax_scale: lambda: T.f32,
            c_context_len:  lambda: T.index,
        ):
            f32 = T.f32; i32 = T.i32; i64 = T.i64; idx = T.index
            vec4_f32 = T.f32x4; vec4_i16 = T.i16x4
            vec8_bf16 = T.bf16x8; vec1_i64 = T.vec(1, i64); vec2_i64 = T.vec(2, i64)

            fm_fast = flir.arith.FastMathFlags.fast
            acc_zero = arith.constant_vector(0.0, vec4_f32)

            o_rsrc  = buffer_ops.create_buffer_resource(arg_o_out, max_size=True)
            ml_rsrc = buffer_ops.create_buffer_resource(arg_ml_out, max_size=True)
            es_rsrc = buffer_ops.create_buffer_resource(arg_es_out, max_size=True)
            q_rsrc  = buffer_ops.create_buffer_resource(arg_q, max_size=True)
            k_rsrc  = buffer_ops.create_buffer_resource(arg_k, max_size=True)
            v_rsrc  = buffer_ops.create_buffer_resource(arg_v, max_size=True)

            base_ptr = allocator.get_base()
            lds_q   = _state["lds_q"](base_ptr).get()
            lds_p   = _state["lds_p"](base_ptr).get()
            lds_red = _state["lds_red"](base_ptr).get()

            c_one_idx = arith.constant(1, index=True)

            # Q LDS layout
            lds_q_stride = arith.constant(K_QK, index=True)
            shape_lds_q  = flir.make_shape(M_DIM, K_QK)
            stride_lds_q = flir.make_stride(lds_q_stride, c_one_idx)
            layout_lds_q = flir.make_layout(shape_lds_q, stride_lds_q)
            q_k_bytes    = K_QK * ELEM_BYTES
            k_blocks16_q = arith.constant(q_k_bytes // 16, index=True)

            # P LDS layout
            lds_p_stride = arith.constant(K_PV, index=True)
            shape_lds_p  = flir.make_shape(M_DIM, K_PV)
            stride_lds_p = flir.make_stride(lds_p_stride, c_one_idx)
            layout_lds_p = flir.make_layout(shape_lds_p, stride_lds_p)

            # Reduction LDS layout
            red_stride   = arith.constant(M_DIM, index=True)
            shape_red    = flir.make_shape(NUM_WAVES, M_DIM)
            stride_red   = flir.make_stride(red_stride, c_one_idx)
            layout_red   = flir.make_layout(shape_red, stride_red)

            tx = gpu.thread_id("x")
            wave_lane_layout = flir.make_layout((NUM_WAVES, WAVE_SIZE), stride=(WAVE_SIZE, 1))
            coord_wl  = flir.idx2crd(tx, wave_lane_layout)
            wave_id   = flir.get(coord_wl, 0)
            lane_id   = flir.get(coord_wl, 1)
            lane16_layout = flir.make_layout((4, 16), stride=(16, 1))
            coord_l16 = flir.idx2crd(lane_id, lane16_layout)
            lane_div_16 = flir.get(coord_l16, 0)
            lane_mod_16 = flir.get(coord_l16, 1)

            c0 = arith.constant(0, index=True)
            c1 = arith.constant(1, index=True)
            c2 = arith.constant(2, index=True)
            c4 = arith.constant(4, index=True)
            c8 = arith.constant(8, index=True)
            c_eb = arith.constant(ELEM_BYTES, index=True)

            c0_i32 = arith.constant(0, type=i32)
            c64_i32 = arith.constant(64, type=i32)
            c1_i32 = arith.constant(1, type=i32)
            c2_i32 = arith.constant(2, type=i32)
            c4_i32 = arith.constant(4, type=i32)
            c8_i32 = arith.constant(8, type=i32)

            atom_q_lds = flir.make_copy_atom(T.bf16, vector_size=KPACK)
            mfma_fn = rocdl.mfma_f32_16x16x16bf16_1k

            def mfma_step(acc_in, a_op, b_op):
                return mfma_fn(vec4_f32, [a_op, b_op, acc_in, 0, 0, 0])

            def mfma_k64(acc_in, a0, a1, b0, b1):
                mid = mfma_step(acc_in, a0, b0)
                return mfma_step(mid, a1, b1)

            def to_mfma_operand(vec8_val):
                v_i64x2 = vector.bitcast(vec2_i64, vec8_val)
                h0 = vector.extract(v_i64x2, static_position=[0], dynamic_position=[])
                h1 = vector.extract(v_i64x2, static_position=[1], dynamic_position=[])
                h0_v1 = vector.from_elements(vec1_i64, [h0])
                h1_v1 = vector.from_elements(vec1_i64, [h1])
                return vector.bitcast(vec4_i16, h0_v1), vector.bitcast(vec4_i16, h1_v1)

            def _shuffle_xor(w, offset):
                raw = gpu.ShuffleOp(
                    arith.as_value(w), arith.as_value(offset),
                    arith.as_value(c64_i32), mode="xor").shuffleResult
                return arith.ArithValue(raw)

            def warp_reduce_max(val):
                w = val
                for off in [c1_i32, c2_i32, c4_i32, c8_i32]:
                    peer = _shuffle_xor(w, off)
                    w = arith.maximum(w, peer)
                return w

            def warp_reduce_sum(val):
                w = val
                for off in [c1_i32, c2_i32, c4_i32, c8_i32]:
                    peer = _shuffle_xor(w, off)
                    w = w + peer
                return w

            def block_reduce_per_row(val, row_idx, op="max"):
                if op == "max":
                    val = warp_reduce_max(val)
                else:
                    val = warp_reduce_sum(val)
                is_lane0 = arith.cmpu(lane_mod_16, c0, "eq")
                red_coord = flir.make_coord(wave_id, row_idx)
                red_idx = flir.crd2idx(red_coord, layout_red)
                if_op = scf.IfOp(is_lane0)
                with if_op:
                    flir.memref.store(arith.as_value(val), lds_red, [arith.as_value(red_idx)])
                gpu.barrier()
                result = val
                for wi in range_constexpr(NUM_WAVES):
                    wi_idx = arith.constant(wi, index=True)
                    rd_coord = flir.make_coord(wi_idx, row_idx)
                    rd_idx = flir.crd2idx(rd_coord, layout_red)
                    rd_val = flir.memref.load(lds_red, [arith.as_value(rd_idx)])
                    if wi == 0:
                        result = rd_val
                    else:
                        if op == "max":
                            result = arith.maximum(result, rd_val)
                        else:
                            result = result + rd_val
                gpu.barrier()
                return result

            # ========== STEP 1: Load Q to LDS ==========
            q_tile_k_dwords = K_QK * ELEM_BYTES // 4
            q_tile_layout = flir.make_layout(
                (M_DIM, q_tile_k_dwords), stride=(q_tile_k_dwords, 1))
            tx_dw_base = tx * c4
            coord_qtile = flir.idx2crd(tx_dw_base, q_tile_layout)
            row_q = flir.get(coord_qtile, 0)
            col_q_dw = flir.get(coord_qtile, 1)
            col_q_elem = col_q_dw * c2

            q_off_elem = row_q * arith.constant(K_QK, index=True) + col_q_elem
            q_off_dw = q_off_elem / c2
            q_off_i32 = arith.index_cast(i32, q_off_dw)

            q_vec = buffer_ops.buffer_load(q_rsrc, q_off_i32, vec_width=4, dtype=i32)
            q_v8 = vector.bitcast(vec8_bf16, q_vec)

            col_q_bytes = col_q_dw * c4
            col_q_swz = flir.swizzle_xor16(row_q, col_q_bytes, k_blocks16_q)
            col_q_swz_e = col_q_swz / c_eb
            q_store_coord = flir.make_coord(row_q, col_q_swz_e)
            q_store_idx = flir.crd2idx(q_store_coord, layout_lds_q)
            s_view_q = flir.TensorView(
                lds_q, (KPACK,), strides=(1,),
                base_indices=(q_store_idx,), element_type=T.bf16)
            flir.copy(atom_q_lds, q_v8, s_view_q, alignment=16)

            gpu.barrier()

            # ========== STEP 2: QK MFMA ==========
            qk_accs = [acc_zero] * QK_TILES_PER_WARP
            col_off_base_bytes = lane_div_16 * arith.constant(KPACK * ELEM_BYTES, index=True)

            c_stride_k_hsplit = arith.constant(N_QK * KPACK, index=True)
            c_stride_k_kv = arith.constant(KPACK, index=True)

            for ku in range_constexpr(QK_K64_STEPS):
                ku_byte_off = arith.constant(ku * 64, index=True)
                col_base_q = col_off_base_bytes + ku_byte_off
                col_swz_q = flir.swizzle_xor16(lane_mod_16, col_base_q, k_blocks16_q)
                col_swz_q_e = col_swz_q / c_eb
                coord_a = flir.make_coord(lane_mod_16, col_swz_q_e)
                idx_a = flir.crd2idx(coord_a, layout_lds_q)
                loaded_q = vector.load_op(vec8_bf16, lds_q, [idx_a])
                q_a0, q_a1 = to_mfma_operand(loaded_q)

                head_split_base = arith.constant(ku * (32 // KPACK), index=True)
                head_split = head_split_base + lane_div_16

                for ni in range_constexpr(QK_TILES_PER_WARP):
                    kv_col = (wave_id * arith.constant(QK_N_PER_WARP, index=True)
                              + arith.constant(ni * 16, index=True)
                              + lane_mod_16)
                    k_elem_off = head_split * c_stride_k_hsplit + kv_col * c_stride_k_kv
                    k_dw_off = k_elem_off / c2
                    k_off_i32 = arith.index_cast(i32, k_dw_off)

                    k_vec = buffer_ops.buffer_load(k_rsrc, k_off_i32, vec_width=4, dtype=i32)
                    k_v8 = vector.bitcast(vec8_bf16, k_vec)
                    k_b0, k_b1 = to_mfma_operand(k_v8)
                    qk_accs[ni] = mfma_k64(qk_accs[ni], q_a0, q_a1, k_b0, k_b1)

            # ========== STEP 3: Softmax ==========
            c_negbig = arith.constant(-3.4e38, type=f32)
            c_log2e = arith.constant(LOG2_E, type=f32)

            scores_pr = [[None] * QK_TILES_PER_WARP for _ in range(4)]
            for ni in range_constexpr(QK_TILES_PER_WARP):
                for ei in range_constexpr(4):
                    s = vector.extract(qk_accs[ni], static_position=[ei], dynamic_position=[])
                    s = s * c_softmax_scale

                    col = (wave_id * arith.constant(QK_N_PER_WARP, index=True)
                           + arith.constant(ni * 16, index=True) + lane_mod_16)
                    valid_col = arith.cmpu(col, c_context_len, "ult")
                    row = lane_div_16 * c4 + arith.constant(ei, index=True)
                    row_ok = arith.cmpu(row, arith.constant(M_DIM, index=True), "ult")
                    full_mask = arith.andi(row_ok, valid_col)
                    s = arith.select(full_mask, s, c_negbig)
                    scores_pr[ei][ni] = s

            max_per_ei = [None] * 4
            sum_per_ei = [None] * 4

            for ei in range_constexpr(4):
                row_idx = lane_div_16 * c4 + arith.constant(ei, index=True)
                local_max = scores_pr[ei][0]
                for ni in range_constexpr(QK_TILES_PER_WARP):
                    if ni > 0:
                        local_max = arith.maximum(local_max, scores_pr[ei][ni])
                global_max = block_reduce_per_row(local_max, row_idx, op="max")
                max_per_ei[ei] = global_max

                local_sum = arith.constant(0.0, type=f32)
                for ni in range_constexpr(QK_TILES_PER_WARP):
                    s = scores_pr[ei][ni]
                    sub = s - global_max
                    sub_sc = sub * c_log2e
                    prob = flir.math.exp2(arith.as_value(sub_sc), fastmath=fm_fast)
                    prob = arith.ArithValue(prob) if not isinstance(prob, arith.ArithValue) else prob
                    scores_pr[ei][ni] = prob
                    local_sum = local_sum + prob
                global_sum = block_reduce_per_row(local_sum, row_idx, op="sum")
                sum_per_ei[ei] = global_sum

            # ========== STEP 4: Store P to LDS element-by-element (like main PA kernel) ==========
            for ei in range_constexpr(4):
                for ni in range_constexpr(QK_TILES_PER_WARP):
                    v_bf = arith.trunc_f(T.bf16, scores_pr[ei][ni])
                    p_row = lane_div_16 * c4 + arith.constant(ei, index=True)
                    p_col = (wave_id * arith.constant(QK_N_PER_WARP, index=True)
                             + arith.constant(ni * 16, index=True)
                             + lane_mod_16)
                    p_coord = flir.make_coord(p_row, p_col)
                    p_idx = flir.crd2idx(p_coord, layout_lds_p)
                    flir.memref.store(arith.as_value(v_bf), lds_p, [arith.as_value(p_idx)])

            gpu.barrier()

            # ========== STEP 5: Pre-load V ==========
            c_stride_v_col = arith.constant(K_PV, index=True)

            v_tile_data = []
            for ku_v in range_constexpr(PV_K64_STEPS):
                v_intra_base = arith.constant(ku_v * 32, index=True)
                v_intra_k = v_intra_base + lane_div_16 * c8

                ku_v_packs = []
                for ni_v in range_constexpr(PV_TILES_PER_WARP):
                    v_col = (wave_id * arith.constant(PV_N_PER_WARP, index=True)
                             + arith.constant(ni_v * 16, index=True)
                             + lane_mod_16)
                    v_elem_off = v_col * c_stride_v_col + v_intra_k
                    v_dw_off = v_elem_off / c2
                    v_off_i32 = arith.index_cast(i32, v_dw_off)
                    v_vec = buffer_ops.buffer_load(v_rsrc, v_off_i32, vec_width=4, dtype=i32)
                    v_v8 = vector.bitcast(vec8_bf16, v_vec)
                    vb0, vb1 = to_mfma_operand(v_v8)
                    ku_v_packs.append((vb0, vb1))
                v_tile_data.append(ku_v_packs)

            # ========== STEP 6: PV MFMA ==========
            pv_accs = [acc_zero] * PV_TILES_PER_WARP
            for ku_v in range_constexpr(PV_K64_STEPS):
                k_base_p = arith.constant(ku_v * 32, index=True)
                p_col_start = k_base_p + lane_div_16 * c8
                p_coord_lds = flir.make_coord(lane_mod_16, p_col_start)
                p_idx_lds = flir.crd2idx(p_coord_lds, layout_lds_p)
                loaded_p = vector.load_op(vec8_bf16, lds_p, [p_idx_lds])
                p_a0, p_a1 = to_mfma_operand(loaded_p)

                for ni_v in range_constexpr(PV_TILES_PER_WARP):
                    vb0, vb1 = v_tile_data[ku_v][ni_v]
                    pv_accs[ni_v] = mfma_k64(pv_accs[ni_v], p_a0, p_a1, vb0, vb1)

            # ========== STEP 7: Normalize and store output ==========
            c_one_f32 = arith.constant(1.0, type=f32)
            c_o_stride = arith.constant(N_PV, index=True)

            for ni in range_constexpr(PV_TILES_PER_WARP):
                for ei in range_constexpr(4):
                    v_e = vector.extract(pv_accs[ni], static_position=[ei], dynamic_position=[])
                    inv_exp = c_one_f32 / sum_per_ei[ei]
                    v_e = v_e * inv_exp
                    v_out = arith.trunc_f(T.bf16, v_e)

                    row = lane_div_16 * c4 + arith.constant(ei, index=True)
                    col = (wave_id * arith.constant(PV_N_PER_WARP, index=True)
                           + arith.constant(ni * 16, index=True)
                           + lane_mod_16)
                    out_off = row * c_o_stride + col
                    out_off_i32 = arith.index_cast(i32, out_off)
                    buffer_ops.buffer_store(v_out, o_rsrc, out_off_i32)

            # Store max_logits and exp_sums
            is_writer = arith.andi(
                arith.cmpu(wave_id, c0, "eq"),
                arith.cmpu(lane_mod_16, c0, "eq"))
            for ei in range_constexpr(4):
                row = lane_div_16 * c4 + arith.constant(ei, index=True)
                row_ok = arith.cmpu(row, arith.constant(M_DIM, index=True), "ult")
                mask = arith.andi(is_writer, row_ok)
                row_i32 = arith.index_cast(i32, row)
                buffer_ops.buffer_store(max_per_ei[ei], ml_rsrc, row_i32, mask=mask)
                buffer_ops.buffer_store(sum_per_ei[ei], es_rsrc, row_i32, mask=mask)

        @flir.jit
        def __call__(
            self: flir.T.i64,
            arg_o_out:      lambda: memref(DYN, T.bf16),
            arg_ml_out:     lambda: memref(DYN, T.f32),
            arg_es_out:     lambda: memref(DYN, T.f32),
            arg_q:          lambda: memref(DYN, T.bf16),
            arg_k:          lambda: memref(DYN, T.bf16),
            arg_v:          lambda: memref(DYN, T.bf16),
            c_softmax_scale: lambda: T.f32,
            c_context_len:  lambda: T.index,
        ):
            c1 = arith.constant(1, index=True)
            bdx = arith.constant(TOTAL_THREADS, index=True)
            flir.gpu_ext.LaunchFuncOp(
                ["integrated_test", "integrated_kernel"],
                grid_size=(c1, c1, c1),
                block_size=(bdx, c1, c1),
                kernel_operands=[arg_o_out, arg_ml_out, arg_es_out,
                                 arg_q, arg_k, arg_v,
                                 c_softmax_scale, c_context_len],
            )

    m = _Integrated()
    return flydsl.compile(m, waves_per_eu=1)


def test_integrated():
    """Test integrated QK->softmax->PV pipeline against PyTorch reference."""
    print("=" * 60)
    print("TEST 4: Integrated  Q[16,128] @ K -> softmax -> @ V -> O[16,128]")
    print("=" * 60)

    device = "cuda"
    torch.manual_seed(42)

    softmax_scale = 1.0 / math.sqrt(K_QK)
    context_len = N_QK

    # Generate random inputs
    Q = torch.randn(M_DIM, K_QK, dtype=torch.bfloat16, device=device)
    K_ref = torch.randn(K_QK, N_QK, dtype=torch.bfloat16, device=device)
    V_ref = torch.randn(K_PV, N_PV, dtype=torch.bfloat16, device=device)

    # PyTorch reference
    qk = torch.matmul(Q.float(), K_ref.float())  # [16, 256]
    scaled = qk * softmax_scale
    max_vals = scaled.max(dim=1, keepdim=True).values
    shifted = (scaled - max_vals) * LOG2_E
    probs_f32 = torch.pow(2.0, shifted)
    sum_vals = probs_f32.sum(dim=1, keepdim=True)
    probs_norm = probs_f32 / sum_vals  # normalized
    probs_bf16 = probs_norm.to(torch.bfloat16)
    O_ref = torch.matmul(probs_bf16.float(), V_ref.float()).to(torch.bfloat16)

    # Reformat K for kernel: [K_QK//8, N_QK, 8]
    K_cache = K_ref.view(K_QK // KPACK, KPACK, N_QK).permute(0, 2, 1).contiguous()
    # Reformat V: [N_PV, K_PV] = [128, 256]
    V_test = V_ref.T.contiguous()

    # Output tensors
    O_out = torch.zeros(M_DIM, N_PV, dtype=torch.bfloat16, device=device)
    ml_out = torch.zeros(M_DIM, dtype=torch.float32, device=device)
    es_out = torch.zeros(M_DIM, dtype=torch.float32, device=device)

    # Compile and run
    exe = compile_integrated_kernel()
    exe(
        O_out.view(-1),
        ml_out.view(-1),
        es_out.view(-1),
        Q.contiguous().view(-1),
        K_cache.contiguous().view(-1),
        V_test.contiguous().view(-1),
        softmax_scale,
        context_len,
    )
    torch.cuda.synchronize()

    # Compare
    diff = (O_out.float() - O_ref.float()).abs()
    print(f"  O diff.abs.max  = {diff.max().item():.6e}")
    print(f"  O diff.abs.mean = {diff.mean().item():.6e}")
    print(f"  O ref.abs.max   = {O_ref.float().abs().max().item():.6e}")
    rel_err = diff.max().item() / (O_ref.float().abs().max().item() + 1e-12)
    print(f"  relative max    = {rel_err:.6e}")

    diff_ml = (ml_out - scaled.max(dim=1).values).abs()
    print(f"  max_logits diff = {diff_ml.max().item():.6e}")
    diff_es = (es_out - probs_f32.sum(dim=1)).abs()
    print(f"  exp_sums diff   = {diff_es.max().item():.6e}")

    passed = diff.max().item() < 5e-3
    print(f"  PASSED: {passed}")
    return passed, diff.max().item()


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print("FlyDSL PA Sub-Kernel Tests")
    print("=" * 60)

    results = {}

    ok1, err1 = test_qk_mfma()
    results["QK MFMA"] = (ok1, err1)
    print()

    ok2, err2 = test_softmax()
    results["Softmax"] = (ok2, err2)
    print()

    ok3, err3 = test_pv_mfma()
    results["PV MFMA"] = (ok3, err3)
    print()

    ok4, err4 = test_integrated()
    results["Integrated"] = (ok4, err4)
    print()

    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    all_passed = True
    for name, (ok, err) in results.items():
        status = "PASS" if ok else "FAIL"
        print(f"  {name:12s}: {status}  (max_diff={err:.6e})")
        if not ok:
            all_passed = False

    if all_passed:
        print("\nAll sub-kernel tests PASSED!")
    else:
        print("\nSome sub-kernel tests FAILED!")

    exit(0 if all_passed else 1)
