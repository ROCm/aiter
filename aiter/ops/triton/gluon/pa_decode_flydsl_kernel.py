# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
#
# FlyDSL implementation of paged attention decode kernel.
# Mirrors paged_attention_decode_v2_gluon_dot_kernel from pa_decode_flydsl.py.
#
# Performance-optimized version:
# - All stride parameters are i32 (not index/i64) to use 32-bit ALU
# - GPU target uses unsafe-math=true for fast exp2 (bare v_exp_f32)
# - Fast-math flags on max/exp2 to eliminate NaN checks

import functools
import torch

import flydsl
from flydsl.dialects.ext import flir
from flydsl.dialects.ext.python_control_flow import range_constexpr
from flydsl.runtime.device import get_rocm_arch as get_hip_arch
from flydsl.utils import SmemAllocator

from _mlir import ir
from _mlir.dialects import vector as _raw_vector
from _mlir.dialects import llvm as _llvm
from flydsl.dialects.ext import arith, gpu, buffer_ops, vector, rocdl, scf
from flydsl.lang.ir.types import T, memref


def _fast_exp2_f32(x):
    """Emit llvm.call_intrinsic @llvm.exp2 with fast-math flags.

    This bypasses math.exp2 -> __ocml_exp2_f32 lowering that generates
    a safe-but-slow 6-instruction pattern.  llvm.call_intrinsic is a
    CallIntrinsicOp (legal in convert-gpu-to-rocdl) and maps directly
    to a single v_exp_f32 instruction on AMDGPU.
    """
    f32 = ir.F32Type.get()
    fm = ir.Attribute.parse('#llvm.fastmath<fast>')
    val = arith.as_value(x) if isinstance(x, arith.ArithValue) else x
    result = _llvm.call_intrinsic(
        f32,            # result type (single)
        'llvm.exp2',    # intrinsic name
        [val],          # args
        [],             # op_bundle_operands
        ir.DenseI32ArrayAttr.get([]),   # op_bundle_sizes
        fastmath_flags=fm,
    )
    return arith.ArithValue(result)


def _make_buffer_rsrc(ptr_i64):
    """Create AMD buffer resource descriptor from a raw i64 pointer.

    This avoids memref parameters (which carry offset+size+stride overhead)
    and directly builds the 128-bit buffer descriptor from a bare pointer.
    """
    # Convert i64 to LLVM pointer
    llvm_ptr = buffer_ops.create_llvm_ptr(ptr_i64, address_space=0)
    # Buffer descriptor constants
    i16_type = ir.IntegerType.get_signless(16)
    i32_type = ir.IntegerType.get_signless(32)
    stride_val = arith.constant(0, type=i16_type)      # contiguous
    num_records = arith.constant(0x7FFFFFFE, type=i32_type)  # max safe size
    flags_val = (7 << 12) | (4 << 15)
    flags = arith.constant(flags_val, type=i32_type)
    rsrc_type = ir.Type.parse('!llvm.ptr<8>')
    rsrc = rocdl.MakeBufferRsrcOp(rsrc_type,
                                   arith.as_value(llvm_ptr),
                                   arith.as_value(stride_val),
                                   arith.as_value(num_records),
                                   arith.as_value(flags)).result
    return rsrc


def _next_power_of_2(x):
    return 1 if x == 0 else 2 ** (x - 1).bit_length()


# ---------------------------------------------------------------------------
# Compile function (one specialised binary per configuration)
# ---------------------------------------------------------------------------

@functools.lru_cache(maxsize=1024)
def compile_pa_decode_kernel(
    *,
    compute_type_str: str,       # "bf16", "fp16", or "fp8"
    query_seq_len: int,
    one_query_group_size: int,
    head_size_pow2: int,
    kv_block_size: int,          # 16 or 64
    context_partition_size: int,  # e.g. 256
    kv_compute_block_size: int,  # e.g. 256
    query_quant_mode: int,       # -1, 0, 1
    kv_quant_mode: int,          # -1, 0, 1
    fp8_max_value: float,
    value_transposed: bool,
    is_causal: bool,
    cdna_version: int,
    waves_per_eu: int = 1,
):
    """Compile a FlyDSL paged-attention decode kernel for the given config."""

    # ---- Derived compile-time constants ------------------------------------
    is_bf16 = compute_type_str == "bf16"
    is_fp16 = compute_type_str == "fp16"
    is_fp8  = compute_type_str == "fp8"
    elem_bytes = 1 if is_fp8 else 2

    QUERY_SEQ_LEN_POW2 = _next_power_of_2(query_seq_len)
    if one_query_group_size <= 16 // QUERY_SEQ_LEN_POW2:
        ONE_QUERY_GROUP_SIZE_POW2 = 16 // QUERY_SEQ_LEN_POW2
    else:
        ONE_QUERY_GROUP_SIZE_POW2 = _next_power_of_2(one_query_group_size)
    QUERY_GROUP_SIZE_POW2 = QUERY_SEQ_LEN_POW2 * ONE_QUERY_GROUP_SIZE_POW2

    HEAD_SIZE_POW2 = head_size_pow2
    KV_BLOCK_SIZE = kv_block_size
    CONTEXT_PARTITION_SIZE = context_partition_size
    KV_COMPUTE_BLOCK_SIZE = kv_compute_block_size

    if is_fp8 or cdna_version == 4:
        MFMA_INSTR_K = 32
    else:
        MFMA_INSTR_K = 16

    if kv_quant_mode >= 0:
        KV_16B_ELEMENT_COUNT = 16
    else:
        KV_16B_ELEMENT_COUNT = 8

    K_HEAD_SIZE_SPLITS = HEAD_SIZE_POW2 // KV_16B_ELEMENT_COUNT
    MAX_NUM_KV_BLOCKS_PER_COMPUTE = KV_COMPUTE_BLOCK_SIZE // KV_BLOCK_SIZE
    KV_COMPUTE_BLOCK_COUNT = CONTEXT_PARTITION_SIZE // KV_COMPUTE_BLOCK_SIZE

    LOG2_E = 1.4426950408889634

    TOTAL_THREADS = 256
    NUM_WAVES = 4
    WAVE_SIZE = 64

    # ---- MFMA tiling -------------------------------------------------------
    qk_m = QUERY_GROUP_SIZE_POW2        # e.g. 16
    qk_k = HEAD_SIZE_POW2               # e.g. 128
    qk_n = KV_COMPUTE_BLOCK_SIZE        # e.g. 256
    qk_n_per_warp = qk_n // NUM_WAVES   # e.g. 64
    qk_tiles_per_warp = qk_n_per_warp // 16  # e.g. 4

    pv_m = QUERY_GROUP_SIZE_POW2
    pv_k = KV_COMPUTE_BLOCK_SIZE        # e.g. 256
    pv_n = HEAD_SIZE_POW2               # e.g. 128
    pv_n_per_warp = pv_n // NUM_WAVES   # e.g. 32
    pv_tiles_per_warp = pv_n_per_warp // 16  # e.g. 2

    # K64 steps (64 bytes per step = 32 bf16 = 2 MFMA-K16 steps for bf16)
    qk_k64_steps = qk_k * elem_bytes // 64   # 128*2/64 = 4 for bf16
    pv_k64_steps = pv_k * elem_bytes // 64   # 256*2/64 = 8 for bf16

    # ---- LDS sizes ---------------------------------------------------------
    q_lds_elems  = qk_m * qk_k          # Q tile
    p_lds_elems  = pv_m * pv_k          # P tile
    red_lds_elems = NUM_WAVES * qk_m     # reduction scratch (cross-wave)

    gpu_arch = get_hip_arch()
    allocator = SmemAllocator(None, arch=gpu_arch)
    _state = {}

    # DYN = ir.ShapedType.get_dynamic_size()  # No longer needed (raw i64 ptrs)

    # ---- Type helpers -------------------------------------------------------
    def _elem_type():
        if is_bf16: return T.bf16
        if is_fp16: return T.f16
        return T.f8

    def _vec8_type():
        if is_bf16: return T.bf16x8
        if is_fp16: return T.f16x8
        return T.vec(8, T.f8)

    def _out_type():
        if is_fp8: return T.bf16
        return _elem_type()

    module_name = (
        f"pa_decode_{compute_type_str}_q{query_seq_len}_g{one_query_group_size}"
        f"_h{head_size_pow2}_b{kv_block_size}_p{context_partition_size}"
        f"_qq{query_quant_mode}_kq{kv_quant_mode}"
        f"_vt{int(value_transposed)}_ca{int(is_causal)}"
    ).replace("-", "m")

    kpack_elems = KV_16B_ELEMENT_COUNT  # 8 for bf16/fp16

    # ========================================================================
    class _PADecode(flir.MlirModule):
        GPU_MODULE_NAME = module_name
        GPU_MODULE_TARGETS = [
            f'#rocdl.target<chip = "{gpu_arch}", abi = "500", features = "+sramecc,+xnack">'
        ]

        def init_gpu_module(self):
            _state["lds_q"]   = allocator.allocate_array(_elem_type(), q_lds_elems)
            _state["lds_p"]   = allocator.allocate_array(_elem_type(), p_lds_elems)
            _state["lds_red"] = allocator.allocate_array(T.f32, red_lds_elems)
            allocator.finalize()

        # ================================================================
        @flir.kernel
        def pa_decode_kernel(
            self: flir.T.i64,
            arg_exp_sums:       lambda: T.i64,
            arg_max_logits:     lambda: T.i64,
            arg_output:         lambda: T.i64,
            arg_query:          lambda: T.i64,
            arg_key_cache:      lambda: T.i64,
            arg_value_cache:    lambda: T.i64,
            arg_block_tables:   lambda: T.i64,
            arg_context_lens:   lambda: T.i64,
            arg_query_scale:    lambda: T.i64,
            arg_key_scale:      lambda: T.i64,
            arg_value_scale:    lambda: T.i64,
            c_softmax_scale:    lambda: T.f32,
            # -- strides as i32 for 32-bit address arithmetic --
            c_stride_ml_seq:    lambda: T.i32,
            c_stride_ml_head:   lambda: T.i32,
            c_stride_ml_part:   lambda: T.i32,
            c_stride_out_seq:   lambda: T.i32,
            c_stride_out_head:  lambda: T.i32,
            c_stride_out_part:  lambda: T.i32,
            c_stride_out_group: lambda: T.i32,
            c_stride_q_bs:      lambda: T.i32,
            c_stride_q_qlen:    lambda: T.i32,
            c_stride_q_kvhead:  lambda: T.i32,
            c_stride_q_group:   lambda: T.i32,
            c_stride_k_block:   lambda: T.i32,
            c_stride_k_head:    lambda: T.i32,
            c_stride_k_hsplit:  lambda: T.i32,
            c_stride_k_belem:   lambda: T.i32,
            c_stride_v_block:   lambda: T.i32,
            c_stride_v_head:    lambda: T.i32,
            c_stride_v_hsz:     lambda: T.i32,
            c_stride_bt_seq:    lambda: T.i32,
            c_stride_qs_bs:     lambda: T.i32,
            c_stride_qs_qlen:   lambda: T.i32,
            c_stride_qs_kvh:    lambda: T.i32,
            c_kv_scale_s0:      lambda: T.i32,
            c_kv_scale_s1:      lambda: T.i32,
            c_head_size:        lambda: T.i32,
        ):
            # ================= Types =================================
            f32   = T.f32
            i32   = T.i32
            i64   = T.i64
            idx   = T.index
            vec4_f32  = T.f32x4
            vec4_i16  = T.i16x4
            vec4_f16  = T.f16x4
            vec8_elem = _vec8_type()
            vec1_i64  = T.vec(1, i64)
            vec2_i64  = T.vec(2, i64)
            vec1_f32  = T.vec(1, f32)
            vec1_i32  = T.vec(1, i32)

            fm_fast = flir.arith.FastMathFlags.fast

            acc_zero = arith.constant_vector(0.0, vec4_f32)

            # ================= Buffer resources (from raw i64 pointers) =
            es_rsrc  = _make_buffer_rsrc(arg_exp_sums)
            ml_rsrc  = _make_buffer_rsrc(arg_max_logits)
            out_rsrc = _make_buffer_rsrc(arg_output)
            q_rsrc   = _make_buffer_rsrc(arg_query)
            k_rsrc   = _make_buffer_rsrc(arg_key_cache)
            v_rsrc   = _make_buffer_rsrc(arg_value_cache)
            bt_rsrc  = _make_buffer_rsrc(arg_block_tables)
            cl_rsrc  = _make_buffer_rsrc(arg_context_lens)

            if query_quant_mode >= 0:
                qs_rsrc = _make_buffer_rsrc(arg_query_scale)
            if kv_quant_mode >= 0:
                ks_rsrc = _make_buffer_rsrc(arg_key_scale)
                vs_rsrc = _make_buffer_rsrc(arg_value_scale)

            # ================= LDS setup =============================
            base_ptr = allocator.get_base()
            lds_q   = _state["lds_q"](base_ptr).get()
            lds_p   = _state["lds_p"](base_ptr).get()
            lds_red = _state["lds_red"](base_ptr).get()

            c_one_idx    = arith.constant(1, index=True)
            lds_q_stride = arith.constant(qk_k, index=True)
            shape_lds_q  = flir.make_shape(qk_m, qk_k)
            stride_lds_q = flir.make_stride(lds_q_stride, c_one_idx)
            layout_lds_q = flir.make_layout(shape_lds_q, stride_lds_q)
            q_k_bytes    = qk_k * elem_bytes
            k_blocks16_q = arith.constant(q_k_bytes // 16, index=True)

            lds_p_stride = arith.constant(pv_k, index=True)
            shape_lds_p  = flir.make_shape(pv_m, pv_k)
            stride_lds_p = flir.make_stride(lds_p_stride, c_one_idx)
            layout_lds_p = flir.make_layout(shape_lds_p, stride_lds_p)

            red_stride   = arith.constant(qk_m, index=True)
            shape_red    = flir.make_shape(NUM_WAVES, qk_m)
            stride_red   = flir.make_stride(red_stride, c_one_idx)
            layout_red   = flir.make_layout(shape_red, stride_red)

            # ================= Thread / block IDs ====================
            tx = gpu.thread_id("x")
            bx = gpu.block_id("x")   # sequence_idx
            by = gpu.block_id("y")   # kv_head_idx
            bz = gpu.block_id("z")   # sequence_partition_idx

            # Convert block IDs to i32 for address arithmetic
            bx_i32 = arith.index_cast(i32, bx)
            by_i32 = arith.index_cast(i32, by)
            bz_i32 = arith.index_cast(i32, bz)

            wave_lane_layout = flir.make_layout((NUM_WAVES, WAVE_SIZE), stride=(WAVE_SIZE, 1))
            coord_wl  = flir.idx2crd(tx, wave_lane_layout)
            wave_id   = flir.get(coord_wl, 0)
            lane_id   = flir.get(coord_wl, 1)

            lane16_layout = flir.make_layout((4, 16), stride=(16, 1))
            coord_l16 = flir.idx2crd(lane_id, lane16_layout)
            lane_div_16 = flir.get(coord_l16, 0)
            lane_mod_16 = flir.get(coord_l16, 1)

            # Index-type constants (for LDS indexing which requires index)
            c0   = arith.constant(0, index=True)
            c1   = arith.constant(1, index=True)
            c2   = arith.constant(2, index=True)
            c4   = arith.constant(4, index=True)
            c8   = arith.constant(8, index=True)
            c_eb = arith.constant(elem_bytes, index=True)

            # i32 constants for address arithmetic
            c0_i32  = arith.constant(0, type=i32)
            c1_i32  = arith.constant(1, type=i32)
            c2_i32  = arith.constant(2, type=i32)
            c4_i32  = arith.constant(4, type=i32)
            c8_i32  = arith.constant(8, type=i32)
            c16_i32 = arith.constant(16, type=i32)
            c32_i32 = arith.constant(32, type=i32)
            c64_i32 = arith.constant(64, type=i32)

            # i32 versions of thread/wave IDs
            wave_id_i32 = arith.index_cast(i32, wave_id)
            lane_id_i32 = arith.index_cast(i32, lane_id)
            lane_div_16_i32 = arith.index_cast(i32, lane_div_16)
            lane_mod_16_i32 = arith.index_cast(i32, lane_mod_16)

            atom_q_lds = flir.make_copy_atom(_elem_type(), vector_size=kpack_elems)

            # ---- MFMA function ----
            if is_bf16:
                mfma_fn = rocdl.mfma_f32_16x16x16bf16_1k
            elif is_fp16:
                mfma_fn = rocdl.mfma_f32_16x16x16f16
            else:
                mfma_fn = rocdl.mfma_f32_16x16x32_fp8_fp8

            def mfma_step(acc_in, a_op, b_op):
                return mfma_fn(vec4_f32, [a_op, b_op, acc_in, 0, 0, 0])

            def mfma_k64(acc_in, a0, a1, b0, b1):
                mid = mfma_step(acc_in, a0, b0)
                return mfma_step(mid, a1, b1)

            def to_mfma_operand(vec8_val):
                """vec8 elem → (op0, op1) for two MFMA-K16 calls."""
                v_i64x2 = vector.bitcast(vec2_i64, vec8_val)
                h0 = vector.extract(v_i64x2, static_position=[0], dynamic_position=[])
                h1 = vector.extract(v_i64x2, static_position=[1], dynamic_position=[])
                h0_v1 = vector.from_elements(vec1_i64, [h0])
                h1_v1 = vector.from_elements(vec1_i64, [h1])
                if is_bf16:
                    return vector.bitcast(vec4_i16, h0_v1), vector.bitcast(vec4_i16, h1_v1)
                else:
                    return vector.bitcast(vec4_f16, h0_v1), vector.bitcast(vec4_f16, h1_v1)

            # ---- Shuffle helper ----
            def _shuffle_xor(w, offset):
                """Warp shuffle XOR returning ArithValue."""
                raw = gpu.ShuffleOp(
                    arith.as_value(w),
                    arith.as_value(offset),
                    arith.as_value(c64_i32),
                    mode="xor",
                ).shuffleResult
                return arith.ArithValue(raw)

            # ---- Per-row reduction across N-dimension (lane_mod_16) ----
            def warp_reduce_n_max(val):
                w = val
                for off in [c1_i32, c2_i32, c4_i32, c8_i32]:
                    peer = _shuffle_xor(w, off)
                    w = arith.maximum(w, peer, fastmath=fm_fast)
                return w

            def warp_reduce_n_sum(val):
                w = val
                for off in [c1_i32, c2_i32, c4_i32, c8_i32]:
                    peer = _shuffle_xor(w, off)
                    w = w + peer
                return w

            def block_reduce_batch_bpermute(vals, op="max"):
                """Batch N reductions: intra-wave shuffle + LDS write + barrier + LDS read + barrier.
                   All N values share the same barrier pair, reducing barrier count from 2*N to 2.
                   vals: list of scalar f32 values (one per row index = lane_div_16*4+i)
                   Returns list of reduced values (same length).
                """
                n = len(vals)
                # Step 1: intra-wave reduction for all values
                reduced = []
                for v in vals:
                    if op == "max":
                        reduced.append(warp_reduce_n_max(v))
                    else:
                        reduced.append(warp_reduce_n_sum(v))

                # Step 2: lane 0 of each 16-lane group writes ALL values to LDS
                is_lane0 = arith.cmpu(lane_mod_16, c0, "eq")
                if_op = scf.IfOp(is_lane0)
                with if_op:
                    for i in range_constexpr(n):
                        row_idx = lane_div_16 * c4 + arith.constant(i, index=True)
                        red_coord = flir.make_coord(wave_id, row_idx)
                        red_idx   = flir.crd2idx(red_coord, layout_red)
                        flir.memref.store(arith.as_value(reduced[i]), lds_red, [arith.as_value(red_idx)])
                gpu.barrier()

                # Step 3: all threads read ALL values from all waves
                results = []
                for i in range_constexpr(n):
                    row_idx = lane_div_16 * c4 + arith.constant(i, index=True)
                    result = reduced[i]
                    for wi in range_constexpr(NUM_WAVES):
                        wi_idx = arith.constant(wi, index=True)
                        rd_coord = flir.make_coord(wi_idx, row_idx)
                        rd_idx   = flir.crd2idx(rd_coord, layout_red)
                        rd_val   = flir.memref.load(lds_red, [arith.as_value(rd_idx)])
                        if wi == 0:
                            result = rd_val
                        else:
                            if op == "max":
                                result = arith.maximum(result, rd_val, fastmath=fm_fast)
                            else:
                                result = result + rd_val
                    results.append(result)
                gpu.barrier()
                return results

            # ============================================================
            # 1) Load context length & early-exit check
            # ============================================================
            cl_off = bx_i32
            context_length_i32 = buffer_ops.buffer_load(cl_rsrc, cl_off, vec_width=1, dtype=i32)
            context_length = arith.index_cast(idx, context_length_i32)

            kv_seq_start = bz * arith.constant(CONTEXT_PARTITION_SIZE, index=True)
            has_work = arith.cmpu(kv_seq_start, context_length, "ult")

            if has_work:
                # ========================================================
                # 2) Load Q tile to LDS  [qk_m, qk_k]
                # ========================================================
                q_tile_k_dwords = qk_k * elem_bytes // 4
                q_tile_layout_l = flir.make_layout(
                    (qk_m, q_tile_k_dwords), stride=(q_tile_k_dwords, 1)
                )
                tx_dw_base = tx * c4
                coord_qtile = flir.idx2crd(tx_dw_base, q_tile_layout_l)
                row_q = flir.get(coord_qtile, 0)
                col_q_dw = flir.get(coord_qtile, 1)

                qgs_pow2 = arith.constant(ONE_QUERY_GROUP_SIZE_POW2, index=True)
                qgs_shift = arith.constant(ONE_QUERY_GROUP_SIZE_POW2.bit_length() - 1, index=True)
                qgs_mask  = arith.constant(ONE_QUERY_GROUP_SIZE_POW2 - 1, index=True)
                qlen_idx = arith.shrui(row_q, qgs_shift)
                grp_idx  = arith.andi(row_q, qgs_mask)
                col_q_elem = col_q_dw * c2

                # Q offset in i32 arithmetic
                row_q_i32 = arith.index_cast(i32, row_q)
                col_q_dw_i32 = arith.index_cast(i32, col_q_dw)
                qlen_idx_i32 = arith.index_cast(i32, qlen_idx)
                grp_idx_i32  = arith.index_cast(i32, grp_idx)
                col_q_elem_i32 = col_q_dw_i32 * c2_i32

                q_off_elem_i32 = (bx_i32 * c_stride_q_bs
                                  + qlen_idx_i32 * c_stride_q_qlen
                                  + by_i32 * c_stride_q_kvhead
                                  + grp_idx_i32 * c_stride_q_group
                                  + col_q_elem_i32)
                q_off_i32 = arith.shrui(q_off_elem_i32, c1_i32)  # unsigned div-by-2

                row_valid = arith.cmpu(
                    row_q,
                    arith.constant(query_seq_len * one_query_group_size, index=True),
                    "ult")
                col_end = col_q_elem + arith.constant(kpack_elems, index=True)
                c_head_size_idx = arith.index_cast(idx, c_head_size)
                col_valid = arith.cmpu(col_end, c_head_size_idx + c1, "ult")
                q_mask = arith.andi(row_valid, col_valid)

                q_vec = buffer_ops.buffer_load(q_rsrc, q_off_i32, vec_width=4, dtype=i32, mask=q_mask)
                q_v8  = vector.bitcast(vec8_elem, q_vec)

                col_q_bytes = col_q_dw * c4
                col_q_swz   = flir.swizzle_xor16(row_q, col_q_bytes, k_blocks16_q)
                col_q_swz_e = arith.shrui(col_q_swz, c1)  # unsigned /2 (elem_bytes=2)
                q_store_coord = flir.make_coord(row_q, col_q_swz_e)
                q_store_idx   = flir.crd2idx(q_store_coord, layout_lds_q)
                s_view_q = flir.TensorView(
                    lds_q, (kpack_elems,), strides=(1,),
                    base_indices=(q_store_idx,), element_type=_elem_type(),
                )
                flir.copy(atom_q_lds, q_v8, s_view_q, alignment=16)

                # ========================================================
                # 3-pre) Prefetch block table + query/kv scales BEFORE Q barrier
                #        so memory latency overlaps with the barrier.
                # ========================================================
                # Issue query scale loads before barrier
                q_scale_val = arith.constant(1.0, type=f32)
                if query_quant_mode == 0:
                    q_scale_val = buffer_ops.buffer_load(qs_rsrc, c0_i32, vec_width=1, dtype=f32)
                elif query_quant_mode == 1:
                    qs_off_i32 = (bx_i32 * c_stride_qs_bs
                                  + qlen_idx_i32 * c_stride_qs_qlen
                                  + by_i32 * c_stride_qs_kvh
                                  + grp_idx_i32)
                    q_scale_val = buffer_ops.buffer_load(qs_rsrc, qs_off_i32, vec_width=1, dtype=f32, mask=row_valid)

                # Pre-load block table entries for ALL compute blocks
                all_block_nums = []
                for cb_idx_pre in range_constexpr(KV_COMPUTE_BLOCK_COUNT):
                    kv_block_start_pre = (bz * arith.constant(CONTEXT_PARTITION_SIZE // KV_BLOCK_SIZE, index=True)
                                          + arith.constant(cb_idx_pre * MAX_NUM_KV_BLOCKS_PER_COMPUTE, index=True))
                    kv_block_start_pre_i32 = arith.index_cast(i32, kv_block_start_pre)
                    bt_base_pre_i32 = bx_i32 * c_stride_bt_seq + kv_block_start_pre_i32
                    cb_block_nums = []
                    for bi in range_constexpr(MAX_NUM_KV_BLOCKS_PER_COMPUTE):
                        bt_off_i32 = bt_base_pre_i32 + arith.constant(bi, type=i32)
                        bn = buffer_ops.buffer_load(bt_rsrc, bt_off_i32, vec_width=1, dtype=i32)
                        cb_block_nums.append(bn)
                    all_block_nums.append(cb_block_nums)

                gpu.barrier()

                # ========================================================
                # 4) Initialise accumulators
                # ========================================================
                pv_accs = [acc_zero] * pv_tiles_per_warp
                max_logit_f32 = [arith.constant(float("-inf"), type=f32) for _ in range(4)]
                exp_sum_f32   = [arith.constant(0.0, type=f32) for _ in range(4)]

                # ========================================================
                # 5) Main compute-block loop
                # ========================================================
                for cb_idx in range_constexpr(KV_COMPUTE_BLOCK_COUNT):
                    kv_sub_start = kv_seq_start + arith.constant(cb_idx * KV_COMPUTE_BLOCK_SIZE, index=True)
                    kv_sub_end = kv_sub_start + arith.constant(KV_COMPUTE_BLOCK_SIZE, index=True)
                    kv_sub_end = arith.select(
                        arith.cmpu(kv_sub_end, context_length, "ult"),
                        kv_sub_end, context_length)

                    num_kv_tokens = kv_sub_end - kv_sub_start
                    c_bsm1 = arith.constant(KV_BLOCK_SIZE - 1, index=True)
                    c_bs   = arith.constant(KV_BLOCK_SIZE, index=True)
                    num_kv_blocks = (num_kv_tokens + c_bsm1) / c_bs

                    # Use pre-loaded block numbers
                    block_nums = all_block_nums[cb_idx]

                    # ---- QK MFMA (K address in i32) ----
                    qk_accs = [acc_zero] * qk_tiles_per_warp
                    col_off_base_bytes = lane_div_16 * arith.constant(kpack_elems * elem_bytes, index=True)

                    for ku in range_constexpr(qk_k64_steps):
                        ku_byte_off = arith.constant(ku * 64, index=True)
                        col_base_q = col_off_base_bytes + ku_byte_off
                        col_swz_q  = flir.swizzle_xor16(lane_mod_16, col_base_q, k_blocks16_q)
                        col_swz_q_e = arith.shrui(col_swz_q, c1)  # unsigned /2 (elem_bytes=2)
                        coord_a    = flir.make_coord(lane_mod_16, col_swz_q_e)
                        idx_a      = flir.crd2idx(coord_a, layout_lds_q)
                        loaded_q   = vector.load_op(vec8_elem, lds_q, [idx_a])
                        q_a0, q_a1 = to_mfma_operand(loaded_q)

                        head_split_i32 = arith.constant(ku * (32 // kpack_elems), type=i32) + lane_div_16_i32

                        for ni in range_constexpr(qk_tiles_per_warp):
                            n_col_i32 = (wave_id_i32 * arith.constant(qk_n_per_warp, type=i32)
                                         + arith.constant(ni * 16, type=i32)
                                         + lane_mod_16_i32)
                            blk_i_i32 = arith.shrui(n_col_i32, arith.constant(KV_BLOCK_SIZE.bit_length() - 1, type=i32))
                            blk_elem_i32 = arith.andi(n_col_i32, arith.constant(KV_BLOCK_SIZE - 1, type=i32))

                            k_bn = block_nums[0]
                            for si in range_constexpr(MAX_NUM_KV_BLOCKS_PER_COMPUTE):
                                if si > 0:
                                    cmp_si = arith.cmpu(blk_i_i32, arith.constant(si, type=i32), "eq")
                                    k_bn = arith.select(cmp_si, block_nums[si], k_bn)

                            # K offset entirely in i32
                            k_elem_off_i32 = (k_bn * c_stride_k_block
                                              + by_i32 * c_stride_k_head
                                              + head_split_i32 * c_stride_k_hsplit
                                              + blk_elem_i32 * c_stride_k_belem)
                            k_off_i32 = arith.shrui(k_elem_off_i32, c1_i32)  # unsigned div-by-2

                            k_vec = buffer_ops.buffer_load(k_rsrc, k_off_i32, vec_width=4, dtype=i32)
                            k_v8  = vector.bitcast(vec8_elem, k_vec)
                            k_b0, k_b1 = to_mfma_operand(k_v8)
                            qk_accs[ni] = mfma_k64(qk_accs[ni], q_a0, q_a1, k_b0, k_b1)

                    # ---- KV quant scales ----
                    k_scale_f32 = arith.constant(1.0, type=f32)
                    v_scale_f32 = arith.constant(1.0, type=f32)
                    if kv_quant_mode == 0:
                        k_scale_f32 = buffer_ops.buffer_load(ks_rsrc, c0_i32, vec_width=1, dtype=f32)
                        v_scale_f32 = buffer_ops.buffer_load(vs_rsrc, c0_i32, vec_width=1, dtype=f32)

                    # ---- Pre-load first V_PREFETCH_STEPS of V (balance latency vs VGPRs) ----
                    V_PREFETCH_STEPS = min(4, pv_k64_steps)
                    v_tile_data = []
                    for ku_v in range_constexpr(V_PREFETCH_STEPS):
                        k_base_v = ku_v * 32
                        v_block_idx = k_base_v // KV_BLOCK_SIZE
                        v_intra_base = k_base_v % KV_BLOCK_SIZE
                        v_bn = block_nums[v_block_idx] if v_block_idx < MAX_NUM_KV_BLOCKS_PER_COMPUTE else block_nums[0]
                        v_intra_k_i32 = arith.constant(v_intra_base, type=i32) + lane_div_16_i32 * c8_i32

                        ku_v_packs = []
                        for ni_v in range_constexpr(pv_tiles_per_warp):
                            v_col_i32 = (wave_id_i32 * arith.constant(pv_n_per_warp, type=i32)
                                         + arith.constant(ni_v * 16, type=i32)
                                         + lane_mod_16_i32)

                            if not value_transposed:
                                v_elem_off_i32 = (v_bn * c_stride_v_block
                                                  + by_i32 * c_stride_v_head
                                                  + v_col_i32 * c_stride_v_hsz
                                                  + v_intra_k_i32)
                            else:
                                v_intra_split_i32 = arith.shrui(v_intra_k_i32, arith.constant(KV_16B_ELEMENT_COUNT.bit_length() - 1, type=i32))
                                v_intra_rem_i32   = arith.andi(v_intra_k_i32, arith.constant(KV_16B_ELEMENT_COUNT - 1, type=i32))
                                v_elem_off_i32 = (v_bn * c_stride_v_block
                                                  + by_i32 * c_stride_v_head
                                                  + v_intra_split_i32 * c_stride_v_hsz
                                                  + v_col_i32 * arith.constant(KV_16B_ELEMENT_COUNT, type=i32)
                                                  + v_intra_rem_i32)

                            v_off_i32 = arith.shrui(v_elem_off_i32, c1_i32)
                            v_vec = buffer_ops.buffer_load(v_rsrc, v_off_i32, vec_width=4, dtype=i32)
                            v_v8  = vector.bitcast(vec8_elem, v_vec)
                            vb0, vb1 = to_mfma_operand(v_v8)
                            ku_v_packs.append((vb0, vb1))
                        v_tile_data.append(ku_v_packs)

                    # ---- Extract QK scores per-element ----
                    c_negbig = arith.constant(-3.4e38, type=f32)
                    c_log2e = arith.constant(LOG2_E, type=f32)

                    kv_block_start = (bz * arith.constant(CONTEXT_PARTITION_SIZE // KV_BLOCK_SIZE, index=True)
                                      + arith.constant(cb_idx * MAX_NUM_KV_BLOCKS_PER_COMPUTE, index=True))
                    kv_col_base = kv_block_start * c_bs

                    scores_pr = [[None] * qk_tiles_per_warp for _ in range(4)]
                    for ni in range_constexpr(qk_tiles_per_warp):
                        for ei in range_constexpr(4):
                            s = vector.extract(qk_accs[ni], static_position=[ei], dynamic_position=[])
                            s = s * c_softmax_scale
                            if query_quant_mode >= 0:
                                s = s * q_scale_val
                            if kv_quant_mode == 0:
                                s = s * k_scale_f32

                            col_abs = (kv_col_base
                                       + wave_id * arith.constant(qk_n_per_warp, index=True)
                                       + arith.constant(ni * 16, index=True)
                                       + lane_mod_16)
                            valid_col = arith.cmpu(col_abs, context_length, "ult")

                            query_row = lane_div_16 * c4 + arith.constant(ei, index=True)
                            if is_causal:
                                q_pos = arith.constant(query_seq_len - 1, index=True) - arith.shrui(query_row, qgs_shift)
                                causal_bound = context_length - q_pos
                                causal_ok = arith.cmpu(col_abs, causal_bound, "ult")
                                valid_col = arith.andi(valid_col, causal_ok)

                            row_ok = arith.cmpu(query_row, arith.constant(query_seq_len * one_query_group_size, index=True), "ult")
                            full_mask = arith.andi(row_ok, valid_col)
                            s = arith.select(full_mask, s, c_negbig)
                            scores_pr[ei][ni] = s

                    # ---- Online softmax per-element (per-row) ----
                    # Phase A: Compute local max for all 4 rows, then batch cross-wave max
                    local_maxes = [None] * 4
                    for ei in range_constexpr(4):
                        local_max = scores_pr[ei][0]
                        for ni in range_constexpr(qk_tiles_per_warp):
                            if ni > 0:
                                local_max = arith.maximum(local_max, scores_pr[ei][ni], fastmath=fm_fast)
                        local_maxes[ei] = local_max

                    # Batched cross-wave max: 1 barrier for all 4 rows
                    global_maxes = block_reduce_batch_bpermute(local_maxes, op="max")

                    # Phase B: Compute new_max, acc_scale, exp, local_sum for all 4 rows
                    new_max_per_ei = [None] * 4
                    acc_scale_per_ei = [None] * 4
                    local_sums = [None] * 4
                    for ei in range_constexpr(4):
                        new_max = arith.maximum(max_logit_f32[ei], global_maxes[ei], fastmath=fm_fast)
                        new_max_per_ei[ei] = new_max

                        diff_max = max_logit_f32[ei] - new_max
                        diff_scaled = diff_max * c_log2e
                        acc_scale_per_ei[ei] = _fast_exp2_f32(diff_scaled)

                        local_sum = arith.constant(0.0, type=f32)
                        for ni in range_constexpr(qk_tiles_per_warp):
                            s = scores_pr[ei][ni]
                            sub = s - new_max
                            sub_sc = sub * c_log2e
                            prob = _fast_exp2_f32(sub_sc)
                            scores_pr[ei][ni] = prob
                            local_sum = local_sum + prob
                        local_sums[ei] = local_sum

                    # Batched cross-wave sum: 1 barrier for all 4 rows
                    global_sums = block_reduce_batch_bpermute(local_sums, op="sum")

                    global_sum_per_ei = global_sums

                    for ei in range_constexpr(4):
                        exp_sum_f32[ei] = acc_scale_per_ei[ei] * exp_sum_f32[ei] + global_sum_per_ei[ei]

                    # ---- Scale PV accumulators per-element ----
                    for ni in range_constexpr(pv_tiles_per_warp):
                        v_list = []
                        for ei in range_constexpr(4):
                            v_e = vector.extract(pv_accs[ni], static_position=[ei], dynamic_position=[])
                            v_e = v_e * acc_scale_per_ei[ei]
                            v_list.append(v_e)
                        pv_accs[ni] = vector.from_elements(vec4_f32, v_list)

                    # ---- FP8 value quant: scale probs by FP8_MAX ----
                    prob_scale_f32 = arith.constant(1.0, type=f32)
                    if kv_quant_mode == 0 and is_fp8:
                        c_fp8max = arith.constant(fp8_max_value, type=f32)
                        for ei in range_constexpr(4):
                            for ni in range_constexpr(qk_tiles_per_warp):
                                scores_pr[ei][ni] = scores_pr[ei][ni] * c_fp8max
                        prob_scale_f32 = v_scale_f32 / c_fp8max

                    # ---- Truncate & store probs to LDS ----
                    for ei in range_constexpr(4):
                        for ni in range_constexpr(qk_tiles_per_warp):
                            v_bf = arith.trunc_f(_elem_type(), scores_pr[ei][ni])
                            p_row = lane_div_16 * c4 + arith.constant(ei, index=True)
                            p_col = (wave_id * arith.constant(qk_n_per_warp, index=True)
                                     + arith.constant(ni * 16, index=True)
                                     + lane_mod_16)
                            p_coord = flir.make_coord(p_row, p_col)
                            p_idx   = flir.crd2idx(p_coord, layout_lds_p)
                            flir.memref.store(arith.as_value(v_bf), lds_p, [arith.as_value(p_idx)])

                    gpu.barrier()

                    # ---- PV MFMA (first V_PREFETCH_STEPS prefetched, rest inline) ----
                    pv_new = [acc_zero] * pv_tiles_per_warp

                    for ku_v in range_constexpr(pv_k64_steps):
                        k_base_p = arith.constant(ku_v * 32, index=True)
                        p_col_start = k_base_p + lane_div_16 * c8
                        p_coord_lds = flir.make_coord(lane_mod_16, p_col_start)
                        p_idx_lds   = flir.crd2idx(p_coord_lds, layout_lds_p)
                        loaded_p    = vector.load_op(vec8_elem, lds_p, [p_idx_lds])
                        p_a0, p_a1  = to_mfma_operand(loaded_p)

                        if ku_v < V_PREFETCH_STEPS:
                            for ni_v in range_constexpr(pv_tiles_per_warp):
                                vb0, vb1 = v_tile_data[ku_v][ni_v]
                                pv_new[ni_v] = mfma_k64(pv_new[ni_v], p_a0, p_a1, vb0, vb1)
                        else:
                            k_base_v = ku_v * 32
                            v_block_idx = k_base_v // KV_BLOCK_SIZE
                            v_intra_base = k_base_v % KV_BLOCK_SIZE
                            v_bn = block_nums[v_block_idx] if v_block_idx < MAX_NUM_KV_BLOCKS_PER_COMPUTE else block_nums[0]
                            v_intra_k_i32 = arith.constant(v_intra_base, type=i32) + lane_div_16_i32 * c8_i32

                            for ni_v in range_constexpr(pv_tiles_per_warp):
                                v_col_i32 = (wave_id_i32 * arith.constant(pv_n_per_warp, type=i32)
                                             + arith.constant(ni_v * 16, type=i32)
                                             + lane_mod_16_i32)

                                if not value_transposed:
                                    v_elem_off_i32 = (v_bn * c_stride_v_block
                                                      + by_i32 * c_stride_v_head
                                                      + v_col_i32 * c_stride_v_hsz
                                                      + v_intra_k_i32)
                                else:
                                    v_intra_split_i32 = arith.shrui(v_intra_k_i32, arith.constant(KV_16B_ELEMENT_COUNT.bit_length() - 1, type=i32))
                                    v_intra_rem_i32   = arith.andi(v_intra_k_i32, arith.constant(KV_16B_ELEMENT_COUNT - 1, type=i32))
                                    v_elem_off_i32 = (v_bn * c_stride_v_block
                                                      + by_i32 * c_stride_v_head
                                                      + v_intra_split_i32 * c_stride_v_hsz
                                                      + v_col_i32 * arith.constant(KV_16B_ELEMENT_COUNT, type=i32)
                                                      + v_intra_rem_i32)

                                v_off_i32 = arith.shrui(v_elem_off_i32, c1_i32)
                                v_vec = buffer_ops.buffer_load(v_rsrc, v_off_i32, vec_width=4, dtype=i32)
                                v_v8  = vector.bitcast(vec8_elem, v_vec)
                                vb0, vb1 = to_mfma_operand(v_v8)
                                pv_new[ni_v] = mfma_k64(pv_new[ni_v], p_a0, p_a1, vb0, vb1)

                    gpu.barrier()

                    # Accumulate PV
                    for ni in range_constexpr(pv_tiles_per_warp):
                        old_vals = []
                        for ei in range_constexpr(4):
                            ov = vector.extract(pv_accs[ni], static_position=[ei], dynamic_position=[])
                            nv = vector.extract(pv_new[ni], static_position=[ei], dynamic_position=[])
                            if kv_quant_mode == 0 and is_fp8:
                                nv = nv * prob_scale_f32
                            combined = ov + nv
                            old_vals.append(combined)
                        pv_accs[ni] = vector.from_elements(vec4_f32, old_vals)

                    for ei in range_constexpr(4):
                        max_logit_f32[ei] = new_max_per_ei[ei]

                # ========================================================
                # 6) Normalise & store output (i32 address arithmetic)
                # ========================================================
                c_one_f32 = arith.constant(1.0, type=f32)

                for ni in range_constexpr(pv_tiles_per_warp):
                    for ei in range_constexpr(4):
                        v_e = vector.extract(pv_accs[ni], static_position=[ei], dynamic_position=[])
                        inv_exp = c_one_f32 / exp_sum_f32[ei]
                        v_e = v_e * inv_exp
                        v_out = arith.trunc_f(_out_type(), v_e)

                        out_row_raw = lane_div_16 * c4 + arith.constant(ei, index=True)
                        out_col = (wave_id * arith.constant(pv_n_per_warp, index=True)
                                   + arith.constant(ni * 16, index=True)
                                   + lane_mod_16)

                        out_qlen_idx = arith.shrui(out_row_raw, qgs_shift)
                        out_grp_idx  = arith.andi(out_row_raw, qgs_mask)
                        out_grp_cont = out_qlen_idx * arith.constant(one_query_group_size, index=True) + out_grp_idx

                        # Output offset in i32
                        out_grp_cont_i32 = arith.index_cast(i32, out_grp_cont)
                        out_col_i32 = arith.index_cast(i32, out_col)
                        out_off_i32 = (bx_i32 * c_stride_out_seq
                                       + by_i32 * c_stride_out_head
                                       + bz_i32 * c_stride_out_part
                                       + out_grp_cont_i32 * c_stride_out_group
                                       + out_col_i32)

                        row_ok = arith.cmpu(out_row_raw, arith.constant(query_seq_len * one_query_group_size, index=True), "ult")
                        col_ok = arith.cmpu(out_col, c_head_size_idx, "ult")
                        out_mask = arith.andi(row_ok, col_ok)

                        buffer_ops.buffer_store(v_out, out_rsrc, out_off_i32, mask=out_mask)

                # ---- Store max_logits, exp_sums (i32 address arithmetic) ----
                is_writer_base = arith.andi(
                    arith.cmpu(wave_id, c0, "eq"),
                    arith.cmpu(lane_mod_16, c0, "eq"))
                for ei in range_constexpr(4):
                    ml_row = lane_div_16 * c4 + arith.constant(ei, index=True)
                    ml_qlen = arith.shrui(ml_row, qgs_shift)
                    ml_grp  = arith.andi(ml_row, qgs_mask)
                    ml_cont = ml_qlen * arith.constant(one_query_group_size, index=True) + ml_grp

                    ml_cont_i32 = arith.index_cast(i32, ml_cont)
                    ml_off_i32 = (bx_i32 * c_stride_ml_seq
                                  + by_i32 * c_stride_ml_head
                                  + bz_i32 * c_stride_ml_part
                                  + ml_cont_i32)

                    ml_row_ok = arith.cmpu(ml_row, arith.constant(query_seq_len * one_query_group_size, index=True), "ult")
                    ml_mask = arith.andi(is_writer_base, ml_row_ok)

                    buffer_ops.buffer_store(max_logit_f32[ei], ml_rsrc, ml_off_i32, mask=ml_mask)
                    buffer_ops.buffer_store(exp_sum_f32[ei], es_rsrc, ml_off_i32, mask=ml_mask)

        # ================================================================
        @flir.jit
        def __call__(
            self: flir.T.i64,
            arg_exp_sums:     lambda: T.i64,
            arg_max_logits:   lambda: T.i64,
            arg_output:       lambda: T.i64,
            arg_query:        lambda: T.i64,
            arg_key_cache:    lambda: T.i64,
            arg_value_cache:  lambda: T.i64,
            arg_block_tables: lambda: T.i64,
            arg_context_lens: lambda: T.i64,
            arg_query_scale:  lambda: T.i64,
            arg_key_scale:    lambda: T.i64,
            arg_value_scale:  lambda: T.i64,
            c_softmax_scale:  lambda: T.f32,
            # -- strides as i32 --
            c_stride_ml_seq:    lambda: T.i32,
            c_stride_ml_head:   lambda: T.i32,
            c_stride_ml_part:   lambda: T.i32,
            c_stride_out_seq:   lambda: T.i32,
            c_stride_out_head:  lambda: T.i32,
            c_stride_out_part:  lambda: T.i32,
            c_stride_out_group: lambda: T.i32,
            c_stride_q_bs:      lambda: T.i32,
            c_stride_q_qlen:    lambda: T.i32,
            c_stride_q_kvhead:  lambda: T.i32,
            c_stride_q_group:   lambda: T.i32,
            c_stride_k_block:   lambda: T.i32,
            c_stride_k_head:    lambda: T.i32,
            c_stride_k_hsplit:  lambda: T.i32,
            c_stride_k_belem:   lambda: T.i32,
            c_stride_v_block:   lambda: T.i32,
            c_stride_v_head:    lambda: T.i32,
            c_stride_v_hsz:     lambda: T.i32,
            c_stride_bt_seq:    lambda: T.i32,
            c_stride_qs_bs:     lambda: T.i32,
            c_stride_qs_qlen:   lambda: T.i32,
            c_stride_qs_kvh:    lambda: T.i32,
            c_kv_scale_s0:      lambda: T.i32,
            c_kv_scale_s1:      lambda: T.i32,
            c_head_size:        lambda: T.i32,
            # -- grid size as index --
            c_num_seqs:         lambda: T.index,
            c_num_kv_heads:     lambda: T.index,
            c_max_part_num:     lambda: T.index,
        ):
            c1 = arith.constant(1, index=True)
            bdx = arith.constant(TOTAL_THREADS, index=True)

            flir.gpu_ext.LaunchFuncOp(
                [module_name, "pa_decode_kernel"],
                grid_size=(c_num_seqs, c_num_kv_heads, c_max_part_num),
                block_size=(bdx, c1, c1),
                kernel_operands=[
                    arg_exp_sums, arg_max_logits, arg_output,
                    arg_query, arg_key_cache, arg_value_cache,
                    arg_block_tables, arg_context_lens,
                    arg_query_scale, arg_key_scale, arg_value_scale,
                    c_softmax_scale,
                    c_stride_ml_seq, c_stride_ml_head, c_stride_ml_part,
                    c_stride_out_seq, c_stride_out_head, c_stride_out_part, c_stride_out_group,
                    c_stride_q_bs, c_stride_q_qlen, c_stride_q_kvhead, c_stride_q_group,
                    c_stride_k_block, c_stride_k_head, c_stride_k_hsplit, c_stride_k_belem,
                    c_stride_v_block, c_stride_v_head, c_stride_v_hsz,
                    c_stride_bt_seq,
                    c_stride_qs_bs, c_stride_qs_qlen, c_stride_qs_kvh,
                    c_kv_scale_s0, c_kv_scale_s1,
                    c_head_size,
                ],
            )

    m = _PADecode()
    return flydsl.compile(m, waves_per_eu=waves_per_eu, unsafe_fp_math=True)


# ---------------------------------------------------------------------------
# Wrapper (matching _paged_attention_decode_v2_with_dot_kernel_reshape_wrapper)
# ---------------------------------------------------------------------------

def _paged_attention_decode_v2_with_dot_kernel_flydsl_wrapper(
    grid,
    exp_sums_ptr,
    max_logits_ptr,
    output_ptr,
    query_ptr,
    key_cache_ptr,
    value_cache_ptr,
    block_tables_ptr,
    context_lengths_ptr,
    softmax_scale,
    query_scale,
    key_scale,
    value_scale,
    stride_max_logits_seq,
    stride_max_logits_head,
    stride_max_logits_part,
    stride_output_seq,
    stride_output_head,
    stride_output_part,
    stride_output_group,
    stride_query_bs,
    stride_query_qlen,
    stride_query_kv_head,
    stride_query_group_size,
    stride_key_block,
    stride_key_head,
    stride_key_head_split,
    stride_key_block_elem,
    stride_value_block,
    stride_value_head_size,
    stride_value_block_elem,
    stride_block_table_seq,
    stride_query_scale_bs,
    stride_query_scale_qlen,
    stride_query_scale_kv_head,
    kv_scale_stride_0,
    kv_scale_stride_1,
    COMPUTE_TYPE,
    query_seq_len,
    query_group_size,
    HEAD_SIZE,
    KV_BLOCK_SIZE,
    KV_16B_ELEMENT_COUNT,
    CONTEXT_PARTITION_SIZE,
    QUERY_QUANT_MODE,
    KV_QUANT_MODE,
    FP8_MAX_VALUE,
    VALUE_TRANSPOSED,
    IS_CAUSAL,
    SLIDING_WINDOW,
    sinks_ptr,
    PS,
    CDNA_VERSION,
):
    """Wrapper function that dispatches to FlyDSL PA kernels."""
    import triton
    import triton.language as tl

    num_sequences, num_kv_heads, num_splits = grid
    HEAD_SIZE_POW2 = triton.next_power_of_2(HEAD_SIZE)

    if KV_BLOCK_SIZE > CONTEXT_PARTITION_SIZE:
        raise NotImplementedError(
            "paged_attention_decode_v2_gluon_large_block_dot_kernel is not yet "
            "implemented in FlyDSL."
        )
    if PS and SLIDING_WINDOW > 0:
        raise NotImplementedError(
            "paged_attention_decode_sliding_window is not yet implemented in FlyDSL."
        )

    if COMPUTE_TYPE == tl.bfloat16:
        ctype_str = "bf16"
    elif COMPUTE_TYPE == tl.float16:
        ctype_str = "fp16"
    elif hasattr(tl, "float8e4b8") and COMPUTE_TYPE == tl.float8e4b8:
        ctype_str = "fp8"
    else:
        ctype_str = "bf16"

    QUERY_SEQ_LEN_POW2 = triton.next_power_of_2(query_seq_len)
    if query_group_size <= 16 // QUERY_SEQ_LEN_POW2:
        ONE_QUERY_GROUP_SIZE_POW2 = 16 // QUERY_SEQ_LEN_POW2
    else:
        ONE_QUERY_GROUP_SIZE_POW2 = triton.next_power_of_2(query_group_size)

    KV_COMPUTE_BLOCK_SIZE = CONTEXT_PARTITION_SIZE

    if QUERY_SEQ_LEN_POW2 * ONE_QUERY_GROUP_SIZE_POW2 == 64:
        wpe = 3
    else:
        wpe = 4

    exe = compile_pa_decode_kernel(
        compute_type_str=ctype_str,
        query_seq_len=query_seq_len,
        one_query_group_size=query_group_size,
        head_size_pow2=HEAD_SIZE_POW2,
        kv_block_size=KV_BLOCK_SIZE,
        context_partition_size=CONTEXT_PARTITION_SIZE,
        kv_compute_block_size=KV_COMPUTE_BLOCK_SIZE,
        query_quant_mode=QUERY_QUANT_MODE,
        kv_quant_mode=KV_QUANT_MODE,
        fp8_max_value=FP8_MAX_VALUE,
        value_transposed=VALUE_TRANSPOSED,
        is_causal=IS_CAUSAL,
        cdna_version=CDNA_VERSION,
        waves_per_eu=wpe,
    )

    def _dptr(t):
        """Get raw data pointer (as int) from tensor, or 0 for None."""
        if t is None:
            return 0
        return t.data_ptr()

    exe(
        _dptr(exp_sums_ptr), _dptr(max_logits_ptr), _dptr(output_ptr),
        _dptr(query_ptr), _dptr(key_cache_ptr), _dptr(value_cache_ptr),
        _dptr(block_tables_ptr), _dptr(context_lengths_ptr),
        _dptr(query_scale), _dptr(key_scale), _dptr(value_scale),
        softmax_scale,
        stride_max_logits_seq, stride_max_logits_head, stride_max_logits_part,
        stride_output_seq, stride_output_head, stride_output_part, stride_output_group,
        stride_query_bs, stride_query_qlen, stride_query_kv_head, stride_query_group_size,
        stride_key_block, stride_key_head, stride_key_head_split, stride_key_block_elem,
        stride_value_block, stride_value_head_size, stride_value_block_elem,
        stride_block_table_seq,
        stride_query_scale_bs, stride_query_scale_qlen, stride_query_scale_kv_head,
        kv_scale_stride_0, kv_scale_stride_1,
        HEAD_SIZE,
        num_sequences, num_kv_heads, num_splits,
    )
    torch.cuda.synchronize()
