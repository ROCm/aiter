# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""
Gated Delta Net K5 hidden-state recurrence kernel -- MFMA 32x32x16 VK fork (基础版).

4 warps (256 threads), 2×2 grid 覆盖 64×64 tile (与 Triton VK warpsPerCTA=[2,2] 一致)。
h_accs 使用 [V, K] layout，GEMM2 trans 零开销。k 使用 token-major [B, T, Hg, K]。
无 pipeline 优化：纯顺序执行，验证正确性。
"""

import math

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import arith, const_expr, gpu, range_constexpr, rocdl, vector
from flydsl.expr.typing import T
from flydsl._mlir import ir
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.runtime.device import get_rocm_arch
from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr

from .tensor_shim import GTensor, STensor, _to_raw

_LOG2E = math.log2(math.e)


def _llvm_lds_ptr_ty():
    return ir.Type.parse("!llvm.ptr<3>")


def _make_fast_exp(g_is_log2_scaled: bool):
    if g_is_log2_scaled:

        def _fast_exp(x):
            return rocdl.exp2(T.f32, x)

    else:

        def _fast_exp(x):
            return rocdl.exp2(T.f32, x * _LOG2E)

    return _fast_exp


def _mfma_bf16_32x32x16(a_bf16x8, b_bf16x8, acc_f32x16):
    """mfma_f32_32x32x16_bf16 (gfx950 CDNA4)。输入直接用 bf16x8，不做 bitcast。"""
    return rocdl.mfma_f32_32x32x16_bf16(
        T.vec(16, T.f32), [a_bf16x8, b_bf16x8, acc_f32x16, 0, 0, 0]
    )


def compile_chunk_gated_delta_h_mfma32_vk(
    *,
    K: int,
    V: int,
    BT: int = 64,
    BV: int = 64,
    H: int,
    Hg: int,
    USE_G: bool = True,
    USE_GK: bool = False,
    USE_INITIAL_STATE: bool = True,
    STORE_FINAL_STATE: bool = True,
    SAVE_NEW_VALUE: bool = True,
    IS_VARLEN: bool = True,
    WU_CONTIGUOUS: bool = True,
    STATE_DTYPE_BF16: bool = False,
    G_IS_LOG2_SCALED: bool = False,
):
    """编译 mfma32 VK 基础版 K5 kernel (4 warps, 2×2 grid)。"""
    assert K <= 256
    assert K % 64 == 0
    assert BV == 64, "mfma32 VK fork 固定 BV=64"
    assert not USE_GK, "mfma32 VK fork 不支持 per-K gates"

    _fast_exp = _make_fast_exp(G_IS_LOG2_SCALED)

    # ===== 核心常量 =====
    WARP_SIZE = 64
    WMMA_K = 16
    MFMA_ACC_SIZE = 16

    # 4 warps 2×2 grid 覆盖 64×64 tile (Triton warpsPerCTA=[2,2])
    NUM_WARPS = 4
    BLOCK_THREADS = NUM_WARPS * WARP_SIZE  # 256

    # h[BV=64, K=128] 拆成 NUM_K_HALVES 个 64×64 K-half
    NUM_K_HALVES = K // 64  # 2
    K_STEPS_PER_HALF = 64 // WMMA_K  # 4 (K-step per K-half for GEMM)
    # 4-warp 2×2 grid 里 warp_row 已经切了 BT (和 V)，所以不需要 BT_MTILES 循环
    # 每 warp 的 GEMM1 output 是单个 32×32 块
    BT_MTILES = 1  # 每 warp 覆盖 1 个 32-high BT tile

    # 每 warp 在每个 K-half 中持有 1 个 acc (32×32, f32x16)
    # 2 K-half → 每 warp 2 个 acc → 32 VGPR
    NUM_H_ACCS = NUM_K_HALVES

    # ===== LDS 布局 =====
    LDS_W_PAD = 4
    LDS_W_STRIDE = K + LDS_W_PAD
    LDS_W_ELEMS = BT * LDS_W_STRIDE
    LDS_W_BYTES = LDS_W_ELEMS * 2

    # lds_k: [K, BT+pad]（BT innermost），GEMM2 A 需要 8 连续 BT
    LDS_K_PAD = 8
    LDS_K_STRIDE = BT + LDS_K_PAD
    LDS_K_ELEMS = K * LDS_K_STRIDE
    LDS_K_BYTES = LDS_K_ELEMS * 2

    LDS_U_PAD = 4
    LDS_U_STRIDE = BV + LDS_U_PAD
    LDS_U_ELEMS = BT * LDS_U_STRIDE
    LDS_U_BYTES = LDS_U_ELEMS * 2

    LDS_G_ELEMS = BT
    LDS_G_BYTES = LDS_G_ELEMS * 4

    # scratch: 64×(64+pad) bf16, 用于 GEMM B operand + h-store + vn-store
    # 4 warps 协作写满 64×64 tile，然后各 warp 按需读取
    LDS_SCRATCH_PAD = 4
    LDS_SCRATCH_STRIDE = 64 + LDS_SCRATCH_PAD  # 68
    LDS_SCRATCH_ELEMS = 64 * LDS_SCRATCH_STRIDE
    LDS_SCRATCH_BYTES = LDS_SCRATCH_ELEMS * 2  # ~8.7 KB

    _K5_KERNEL_REVISION = 2

    GPU_ARCH = get_rocm_arch()
    allocator = SmemAllocator(
        None,
        arch=GPU_ARCH,
        global_sym_name=f"gdn_h_mfma32_vk_4w_smem_v{_K5_KERNEL_REVISION}",
    )
    lds_w_offset = allocator._align(allocator.ptr, 16)
    allocator.ptr = lds_w_offset + LDS_W_BYTES
    lds_k_offset = allocator._align(allocator.ptr, 16)
    allocator.ptr = lds_k_offset + LDS_K_BYTES
    lds_u_offset = allocator._align(allocator.ptr, 16)
    allocator.ptr = lds_u_offset + LDS_U_BYTES
    lds_g_offset = allocator._align(allocator.ptr, 16)
    allocator.ptr = lds_g_offset + LDS_G_BYTES
    lds_scratch_offset = allocator._align(allocator.ptr, 16)
    allocator.ptr = lds_scratch_offset + LDS_SCRATCH_BYTES

    # 协作 load 参数 (256 threads)
    LOAD_VEC_WIDTH = 8

    W_THREADS_PER_ROW = 64 // LOAD_VEC_WIDTH  # 8
    W_ROWS_PER_BATCH = BLOCK_THREADS // W_THREADS_PER_ROW  # 32
    W_NUM_BATCHES = BT // W_ROWS_PER_BATCH  # 2

    # k load: token-major [B, T, Hg, K], K 连续 (stride=1)
    # 每线程读 8 连续 K → 写到 lds_k[K_row, BT_col] 的 K 方向
    # 分工: K/8 = 16 个 K-block, BT = 64 个 token → 16*64 = 1024 对
    # 256 线程 → 4 batch
    K_BLOCKS_PER_TOK = K // LOAD_VEC_WIDTH  # 16
    K_PAIRS_PER_BATCH = BLOCK_THREADS  # 256
    K_TOTAL_PAIRS = BT * K_BLOCKS_PER_TOK  # 1024
    K_TOTAL_PAIRS // K_PAIRS_PER_BATCH  # 4

    U_THREADS_PER_ROW = BV // LOAD_VEC_WIDTH  # 8
    U_ROWS_PER_BATCH = BLOCK_THREADS // U_THREADS_PER_ROW  # 32
    U_NUM_BATCHES = BT // U_ROWS_PER_BATCH  # 2

    @flyc.kernel(name="chunk_gdn_fwd_h_flydsl_mfma32_vk")
    def gdn_h_kernel(
        k_tensor: fx.Pointer,
        v_tensor: fx.Pointer,
        w_tensor: fx.Pointer,
        v_new_tensor: fx.Pointer,
        g_tensor: fx.Pointer,
        gk_tensor: fx.Pointer,
        h_tensor: fx.Pointer,
        h0_tensor: fx.Pointer,
        ht_tensor: fx.Pointer,
        cu_seqlens_tensor: fx.Pointer,
        chunk_offsets_tensor: fx.Pointer,
        T_val: fx.Int32,
        T_flat: fx.Int32,
        N_val: fx.Int32,
    ):
        i_v = fx.block_idx.x
        i_nh = fx.block_idx.y
        i_n = i_nh // fx.Int32(H)
        i_h = i_nh % fx.Int32(H)
        tid = fx.thread_idx.x
        wid = tid // fx.Int32(WARP_SIZE)
        lane = tid % fx.Int32(WARP_SIZE)

        # mfma32 lane 坐标
        lane_row = lane % fx.Int32(32)
        lane_half = lane // fx.Int32(32)

        # 2×2 warp grid
        warp_row = wid // fx.Int32(2)  # V split: 0 or 1
        warp_col = wid % fx.Int32(2)  # K split within K-half: 0 or 1

        # tensor views
        k_ = GTensor(k_tensor, dtype=T.bf16, shape=(-1,))
        v_ = GTensor(v_tensor, dtype=T.bf16, shape=(-1,))
        w_ = GTensor(w_tensor, dtype=T.bf16, shape=(-1,))
        h_ = GTensor(h_tensor, dtype=T.bf16, shape=(-1,))
        g_ = GTensor(g_tensor, dtype=T.f32, shape=(-1,))
        vn_ = GTensor(v_new_tensor, dtype=T.bf16, shape=(-1,))
        state_t = T.bf16 if STATE_DTYPE_BF16 else T.f32
        if const_expr(USE_INITIAL_STATE):
            h0_ = GTensor(h0_tensor, dtype=state_t, shape=(-1,))
        if const_expr(STORE_FINAL_STATE):
            ht_ = GTensor(ht_tensor, dtype=state_t, shape=(-1,))
        if const_expr(IS_VARLEN):
            cu_ = GTensor(cu_seqlens_tensor, dtype=T.i32, shape=(-1,))
            co_ = GTensor(chunk_offsets_tensor, dtype=T.i32, shape=(-1,))

        # LDS views
        lds_base_ptr = allocator.get_base()
        lds_w_ptr = SmemPtr(lds_base_ptr, lds_w_offset, T.bf16, shape=(LDS_W_ELEMS,))
        lds_w = STensor(lds_w_ptr, dtype=T.bf16, shape=(LDS_W_ELEMS,))
        lds_k_ptr = SmemPtr(lds_base_ptr, lds_k_offset, T.bf16, shape=(LDS_K_ELEMS,))
        lds_k = STensor(lds_k_ptr, dtype=T.bf16, shape=(LDS_K_ELEMS,))
        lds_u_ptr = SmemPtr(lds_base_ptr, lds_u_offset, T.bf16, shape=(LDS_U_ELEMS,))
        lds_u = STensor(lds_u_ptr, dtype=T.bf16, shape=(LDS_U_ELEMS,))
        lds_g_ptr = SmemPtr(lds_base_ptr, lds_g_offset, T.f32, shape=(LDS_G_ELEMS,))
        lds_g = STensor(lds_g_ptr, dtype=T.f32, shape=(LDS_G_ELEMS,))
        lds_scratch_ptr = SmemPtr(
            lds_base_ptr, lds_scratch_offset, T.bf16, shape=(LDS_SCRATCH_ELEMS,)
        )
        lds_scratch = STensor(lds_scratch_ptr, dtype=T.bf16, shape=(LDS_SCRATCH_ELEMS,))
        lds_w_memref = lds_w_ptr.get()
        lds_k_memref = lds_k_ptr.get()
        lds_scratch_memref = lds_scratch_ptr.get()

        v8bf16_type = T.vec(8, T.bf16)

        def _lds_read_bf16x8(memref, idx):
            return vector.load_op(v8bf16_type, memref, [idx])

        # 协作 load 线程分工
        w_load_row = tid // fx.Int32(W_THREADS_PER_ROW)
        w_load_col = (tid % fx.Int32(W_THREADS_PER_ROW)) * fx.Int32(LOAD_VEC_WIDTH)
        # k load 分工: tid → (token_idx, k_block_idx)
        # 每 batch 256 对, tid 直接映射到一个 (tok, k_blk)
        # 留到循环内动态计算
        u_load_row = tid // fx.Int32(U_THREADS_PER_ROW)
        u_load_col = (tid % fx.Int32(U_THREADS_PER_ROW)) * fx.Int32(LOAD_VEC_WIDTH)

        # Prologue
        if const_expr(IS_VARLEN):
            bos = cu_[fx.Index(i_n)]
            eos = cu_[fx.Index(i_n) + fx.Index(1)]
            T_local = eos - bos
            NT = (T_local + fx.Int32(BT - 1)) // fx.Int32(BT)
            boh = co_[fx.Index(i_n)]
        else:
            bos = i_n * T_val
            T_local = T_val
            NT = (T_local + fx.Int32(BT - 1)) // fx.Int32(BT)
            boh = i_n * NT

        h_base = (boh * fx.Int32(H) + i_h) * fx.Int32(V * K)
        stride_h = fx.Int32(H * V * K)

        gqa_ratio = H // Hg
        i_hg = i_h // fx.Int32(gqa_ratio)
        # token-major k [B, T_total, Hg, K]（与 Triton VK 一致）
        # stride: tok→tok = Hg*K, hg→hg = K, k→k = 1
        stride_k_tok = fx.Int32(Hg * K)
        k_base = (bos * fx.Int32(Hg) + i_hg) * fx.Int32(K)

        if const_expr(WU_CONTIGUOUS):
            if const_expr(IS_VARLEN):
                v_base = (i_h * T_flat + bos) * fx.Int32(V)
                w_base = (i_h * T_flat + bos) * fx.Int32(K)
            else:
                v_base = ((i_n * fx.Int32(H) + i_h) * T_flat) * fx.Int32(V)
                w_base = ((i_n * fx.Int32(H) + i_h) * T_flat) * fx.Int32(K)
            stride_v = fx.Int32(V)
            stride_w = fx.Int32(K)
        else:
            v_base = (bos * fx.Int32(H) + i_h) * fx.Int32(V)
            w_base = (bos * fx.Int32(H) + i_h) * fx.Int32(K)
            stride_v = fx.Int32(H * V)
            stride_w = fx.Int32(H * K)

        if const_expr(IS_VARLEN):
            vn_base = (i_h * T_flat + bos) * fx.Int32(V)
        else:
            vn_base = ((i_n * fx.Int32(H) + i_h) * T_flat) * fx.Int32(V)

        if const_expr(USE_INITIAL_STATE):
            h0_base = i_nh * fx.Int32(V * K)
        if const_expr(STORE_FINAL_STATE):
            ht_base = i_nh * fx.Int32(V * K)
        if const_expr(USE_G):
            if const_expr(IS_VARLEN):
                g_base = i_h * T_flat + bos
            else:
                g_base = (i_n * fx.Int32(H) + i_h) * T_flat

        # ===== h_accs 初始化 =====
        # 每 warp 2 个 acc (per K-half)
        # h_accs[kh] = h[warp_row*32:(warp_row+1)*32, kh*64+warp_col*32 : +32]
        # #linear1 语义: V = row (register-derived), K = column (lane%32)
        acc_zero = fx.full(MFMA_ACC_SIZE, 0.0, fx.Float32)
        h_accs = [acc_zero] * NUM_H_ACCS

        if const_expr(USE_INITIAL_STATE):
            for kh in range_constexpr(NUM_H_ACCS):
                k_global_base = fx.Int32(kh * 64) + warp_col * fx.Int32(32)
                k_local = lane_row  # lane%32
                k_global = k_global_base + k_local
                v_base_warp = i_v * fx.Int32(BV) + warp_row * fx.Int32(32)
                vals = []
                for ei in range_constexpr(MFMA_ACC_SIZE):
                    v_local = (
                        fx.Int32((ei // 4) * 8)
                        + lane_half * fx.Int32(4)
                        + fx.Int32(ei % 4)
                    )
                    v_global = v_base_warp + v_local
                    h0_off = h0_base + v_global * fx.Int32(K) + k_global
                    val = h0_[fx.Index(h0_off)]
                    if const_expr(STATE_DTYPE_BF16):
                        val = val.extf(T.f32)
                    vals.append(val)
                loaded_vec = vector.from_elements(T.vec(MFMA_ACC_SIZE, T.f32), vals)
                h_accs[kh] = h_accs[kh] + loaded_vec

        # ===== Chunk loop =====
        init_state = [_to_raw(v) for v in h_accs]

        # 辅助: 写 f32x16 acc 到 scratch [64, 64+pad] 的一个 32×32 子块
        def _write_accs_to_scratch(acc_val, v_off, k_off):
            # 整块 f32x16 → bf16x16 转换
            bf_vec = acc_val.truncf(T.vec(MFMA_ACC_SIZE, T.bf16))
            for ei in range_constexpr(MFMA_ACC_SIZE):
                v_local = (
                    fx.Int32((ei // 4) * 8) + lane_half * fx.Int32(4) + fx.Int32(ei % 4)
                )
                k_local = lane_row
                s_off = (
                    (v_off + v_local) * fx.Int32(LDS_SCRATCH_STRIDE) + k_off + k_local
                )
                bf_scalar = vector.extract(
                    bf_vec, static_position=[ei], dynamic_position=[]
                )
                lds_scratch[fx.Index(s_off)] = bf_scalar

        for i_t, state in range(
            fx.Index(0), fx.Index(NT), fx.Index(1), init=init_state
        ):
            h_accs_in = list(state[:NUM_H_ACCS])
            i_t_i32 = fx.Int32(i_t)

            # ========================================
            # Step 1: h-snapshot store (per K-half)
            # 4 warps 写 scratch → barrier → 256 线程协作读 → HBM
            # ========================================
            for kh in range_constexpr(NUM_H_ACCS):
                _write_accs_to_scratch(
                    h_accs_in[kh],
                    warp_row * fx.Int32(32),  # V offset
                    warp_col * fx.Int32(32),  # K offset within 64
                )
                gpu.barrier()

                # 协作读 scratch[64, 64] → HBM h[V, K]
                # 每线程读 8 连续 K → buffer_store_dwordx4
                HT_K8 = 64 // 8  # 8 groups of 8 K per V-row
                HT_TOTAL = 64 * HT_K8  # 512 (row, group) pairs
                HT_ITERS = HT_TOTAL // BLOCK_THREADS  # 2
                for ht_it in range_constexpr(HT_ITERS):
                    grp = fx.Int32(ht_it * BLOCK_THREADS) + tid
                    v_loc = grp // fx.Int32(HT_K8)
                    k8 = (grp % fx.Int32(HT_K8)) * fx.Int32(8)
                    tile8 = lds_scratch.vec_load(
                        (fx.Index(v_loc * fx.Int32(LDS_SCRATCH_STRIDE) + k8),), 8
                    )
                    v_global = i_v * fx.Int32(BV) + v_loc
                    k_global = fx.Int32(kh * 64) + k8
                    h_off = (
                        h_base + i_t_i32 * stride_h + v_global * fx.Int32(K) + k_global
                    )
                    h_.vec_store((fx.Index(h_off),), tile8, 8)
                gpu.barrier()

            # ========================================
            # Step 2-4: 协作 load w/k/u/g → LDS
            # ========================================
            for kb in range_constexpr(K // 64):
                for batch in range_constexpr(W_NUM_BATCHES):
                    row = fx.Int32(batch * W_ROWS_PER_BATCH) + w_load_row
                    abs_row = i_t_i32 * fx.Int32(BT) + row
                    safe_row = (abs_row < T_local).select(abs_row, fx.Int32(0))
                    w_off = (
                        w_base + safe_row * stride_w + fx.Int32(kb * 64) + w_load_col
                    )
                    w_vec = w_.vec_load((fx.Index(w_off),), LOAD_VEC_WIDTH)
                    w_lds = (
                        row * fx.Int32(LDS_W_STRIDE) + fx.Int32(kb * 64) + w_load_col
                    )
                    lds_w.vec_store((fx.Index(w_lds),), w_vec, LOAD_VEC_WIDTH)

            # k load: token-major [B, T, Hg, K] → lds_k[K, BT+pad]（BT innermost）
            # 做转置 load：每线程读 1 个 k 标量，写到 lds_k[K_row, tok_col]
            # 256 线程 × (K*BT / 256) 次 = 128*64/256 = 32 次
            K_SCALAR_TOTAL = K * BT  # 8192
            K_SCALAR_BATCHES = K_SCALAR_TOTAL // BLOCK_THREADS  # 32
            for batch in range_constexpr(K_SCALAR_BATCHES):
                elem_idx = fx.Int32(batch * BLOCK_THREADS) + tid
                ki = elem_idx % fx.Int32(K)  # K 维度
                bt = elem_idx // fx.Int32(K)  # BT 维度
                abs_tok = i_t_i32 * fx.Int32(BT) + bt
                in_b = abs_tok < T_local
                safe_tok = in_b.select(abs_tok, fx.Int32(0))
                k_off = k_base + safe_tok * stride_k_tok + ki
                k_val = k_[fx.Index(k_off)]
                k_lds_off = ki * fx.Int32(LDS_K_STRIDE) + bt
                lds_k[fx.Index(k_lds_off)] = k_val

            for batch in range_constexpr(U_NUM_BATCHES):
                row = fx.Int32(batch * U_ROWS_PER_BATCH) + u_load_row
                abs_row = i_t_i32 * fx.Int32(BT) + row
                safe_row = (abs_row < T_local).select(abs_row, fx.Int32(0))
                u_off = v_base + safe_row * stride_v + i_v * fx.Int32(BV) + u_load_col
                u_vec = v_.vec_load((fx.Index(u_off),), LOAD_VEC_WIDTH)
                u_lds = row * fx.Int32(LDS_U_STRIDE) + u_load_col
                lds_u.vec_store((fx.Index(u_lds),), u_vec, LOAD_VEC_WIDTH)

            if const_expr(USE_G):
                g_row = tid % fx.Int32(BT)
                g_abs = i_t_i32 * fx.Int32(BT) + g_row
                g_in_b = g_abs < T_local
                g_safe = g_in_b.select(g_abs, fx.Int32(0))
                g_val = g_[fx.Index(g_base + g_safe)]
                g_val = g_in_b.select(g_val, fx.Float32(0.0))
                lds_g[fx.Index(g_row)] = g_val

            gpu.barrier()

            # ========================================
            # Step 5: GEMM1 — bv[64,64] = w[64,K] @ h^T[K,64]
            #
            # K=128 分 2 K-half 顺序做，每 K-half:
            #   1. 4 warps 写 h_accs[kh] → scratch[64,64]
            #   2. barrier
            #   3. 4 K-step × dot → 累加到 bv_accs
            # warp(r,c) 计算 bv[r*32:(r+1)*32, c*32:(c+1)*32]
            # ========================================
            bv_accs = [fx.full(MFMA_ACC_SIZE, 0.0, fx.Float32)] * BT_MTILES

            for kh in range_constexpr(NUM_K_HALVES):
                # 写 h_half 到 scratch
                _write_accs_to_scratch(
                    h_accs_in[kh],
                    warp_row * fx.Int32(32),
                    warp_col * fx.Int32(32),
                )
                gpu.barrier()

                # GEMM1 MFMA: bv += w_tile @ h^T_tile
                # warp(r,c) 计算 bv[r*32:(r+1)*32, c*32:(c+1)*32]
                for k_step in range_constexpr(K_STEPS_PER_HALF):
                    # A: w[warp_row*32 + lane%32, kh*64 + k_step*16 + (lane//32)*8 +0..7]
                    a_row = fx.Index(warp_row * 32) + fx.Index(lane_row)
                    a_col = fx.Index(kh * 64 + k_step * 16) + fx.Index(
                        lane_half
                    ) * fx.Index(8)
                    a_idx = a_row * fx.Index(LDS_W_STRIDE) + a_col
                    a_frag = _lds_read_bf16x8(lds_w_memref, a_idx)

                    # B: h^T[k_step*16 + (lane//32)*8, warp_col*32 + lane%32]
                    # scratch[V, K] → 读 h[V=warp_col*32+lane%32, K=k_step*16+(lane//32)*8]
                    b_v = fx.Index(warp_col * 32) + fx.Index(lane_row)
                    b_k = fx.Index(k_step * 16) + fx.Index(lane_half) * fx.Index(8)
                    b_idx = b_v * fx.Index(LDS_SCRATCH_STRIDE) + b_k
                    b_frag = _lds_read_bf16x8(lds_scratch_memref, b_idx)

                    bv_accs[0] = _mfma_bf16_32x32x16(a_frag, b_frag, bv_accs[0])

                gpu.barrier()

            # ========================================
            # Step 6: v_new = u - bv
            # bv_accs 在 #mma D[BT_tile, V_tile] 布局
            # ========================================
            # v_new = u - bv
            # warp(r,c): bv 在 bv_accs[0], D 布局 [BT_sub=32, V_sub=32]
            # D 中: lane%32 → BT_local (M), d[i] → V_local (N)
            # 全局: BT = warp_row*32 + BT_local, V = warp_col*32 + V_local
            u_vals = []
            for ei in range_constexpr(MFMA_ACC_SIZE):
                bt_local = lane_row
                bv_local = (
                    fx.Int32((ei // 4) * 8) + lane_half * fx.Int32(4) + fx.Int32(ei % 4)
                )
                u_bt_global = warp_row * fx.Int32(32) + bt_local
                u_v_global = warp_col * fx.Int32(32) + bv_local
                u_lds_off = u_bt_global * fx.Int32(LDS_U_STRIDE) + u_v_global
                u_elem = lds_u[fx.Index(u_lds_off)]
                u_vals.append(u_elem.extf(T.f32))
            u_vec = vector.from_elements(T.vec(MFMA_ACC_SIZE, T.f32), u_vals)
            vn_val = u_vec - bv_accs[0]
            vn_frags = [vn_val]

            # 修正: 因为 warp(r,c) 只覆盖 1 个 32×32 bv 块，
            # BT_MTILES=2 时 bv_accs 应该只有 1 个元素。
            # 但上面 GEMM1 按 bt_tile 循环了两次... 这在 4-warp 模式下不对。
            # 4 warp 的 2×2 grid 里 warp_row 已经切了 BT，不需要 bt_tile 循环。
            # TODO: 需要修正 GEMM1 循环结构。暂时 bt_tile=0 对应当前 warp 的块。
            # 先跳过这个问题，后续修正。

            # ========================================
            # Step 7: v_new store → HBM
            # ========================================
            if const_expr(SAVE_NEW_VALUE):
                # vn 在 D[BT, V] 布局 → scratch[BT, V] (V innermost)
                vn_val = vn_frags[0]
                vn_bf = vn_val.truncf(T.vec(MFMA_ACC_SIZE, T.bf16))
                for ei in range_constexpr(MFMA_ACC_SIZE):
                    bt_local = lane_row  # M = BT 方向
                    v_local = (
                        fx.Int32((ei // 4) * 8)
                        + lane_half * fx.Int32(4)
                        + fx.Int32(ei % 4)
                    )  # N = V 方向
                    bt_global = warp_row * fx.Int32(32) + bt_local
                    v_global = warp_col * fx.Int32(32) + v_local
                    s_off = bt_global * fx.Int32(LDS_SCRATCH_STRIDE) + v_global
                    lds_scratch[fx.Index(s_off)] = vector.extract(
                        vn_bf, static_position=[ei], dynamic_position=[]
                    )
                gpu.barrier()

                VN_V8 = 64 // 8
                VN_TOTAL = 64 * VN_V8
                VN_ITERS = VN_TOTAL // BLOCK_THREADS
                for vn_it in range_constexpr(VN_ITERS):
                    grp = fx.Int32(vn_it * BLOCK_THREADS) + tid
                    bt_loc = grp // fx.Int32(VN_V8)
                    v8 = (grp % fx.Int32(VN_V8)) * fx.Int32(8)
                    tile8 = lds_scratch.vec_load(
                        (fx.Index(bt_loc * fx.Int32(LDS_SCRATCH_STRIDE) + v8),), 8
                    )
                    abs_bt = i_t_i32 * fx.Int32(BT) + bt_loc
                    vn_off = vn_base + abs_bt * fx.Int32(V) + i_v * fx.Int32(BV) + v8
                    vn_.vec_store((fx.Index(vn_off),), tile8, 8)
                gpu.barrier()

            # ========================================
            # Step 8: Gating
            # ========================================
            if const_expr(USE_G):
                last_bt = (i_t_i32 + fx.Int32(1)) * fx.Int32(BT)
                last_bt = (last_bt < T_local).select(last_bt, T_local)
                last_idx = last_bt - fx.Int32(1)
                g_last_val = g_[fx.Index(g_base + last_idx)]

                exp_g_last = _fast_exp(g_last_val)
                for kh in range_constexpr(NUM_H_ACCS):
                    h_accs_in[kh] = h_accs_in[kh] * exp_g_last

                for bt_tile in range_constexpr(len(vn_frags)):
                    vn_val = vn_frags[bt_tile]
                    gated_vals = []
                    for ei in range_constexpr(MFMA_ACC_SIZE):
                        bt_local = lane_row
                        bt_global = warp_row * fx.Int32(32) + bt_local
                        abs_bt = i_t_i32 * fx.Int32(BT) + bt_global
                        in_bounds = abs_bt < T_local
                        g_row_val = lds_g[fx.Index(bt_global)]
                        gate = _fast_exp(g_last_val - g_row_val)
                        gate = in_bounds.select(gate, fx.Float32(0.0))
                        gated_vals.append(vn_val[ei] * gate)
                    vn_frags[bt_tile] = vector.from_elements(
                        T.vec(MFMA_ACC_SIZE, T.f32), gated_vals
                    )

            # ========================================
            # Step 9: GEMM2 — h[V,K] += trans(k[K,BT] @ vn[BT,BV])
            #
            # 每 K-half: dot(k_half[64,BT], vn[BT,64]) → result[64,64]
            # trans (零开销) → 累加到 h_accs[V,K]
            # ========================================
            for kh in range_constexpr(NUM_K_HALVES):
                # 写 vn_gated 到 scratch: 布局 [V, BT+pad]（BT innermost）
                # 这样 GEMM2 B 的 vec_load(8) 读 8 连续 BT 值
                # vn_frags[0] 在 D[BT, V] 布局: lane%32=BT, d[i]=V
                # warp(r,c): BT=[r*32:(r+1)*32], V=[c*32:(c+1)*32]
                vn_val = vn_frags[0]
                vn_bf = vn_val.truncf(T.vec(MFMA_ACC_SIZE, T.bf16))
                for ei in range_constexpr(MFMA_ACC_SIZE):
                    bt_local = lane_row  # BT 维度
                    bv_local = (
                        fx.Int32((ei // 4) * 8)
                        + lane_half * fx.Int32(4)
                        + fx.Int32(ei % 4)
                    )  # V 维度
                    bt_global = warp_row * fx.Int32(32) + bt_local
                    bv_global = warp_col * fx.Int32(32) + bv_local
                    # scratch[V_row, BT_col]（BT innermost）
                    s_off = bv_global * fx.Int32(LDS_SCRATCH_STRIDE) + bt_global
                    lds_scratch[fx.Index(s_off)] = vector.extract(
                        vn_bf, static_position=[ei], dynamic_position=[]
                    )
                gpu.barrier()

                # GEMM2 MFMA: D[K_sub, V_sub] = k @ vn
                gemm2_acc = fx.full(MFMA_ACC_SIZE, 0.0, fx.Float32)

                for k_step in range_constexpr(K_STEPS_PER_HALF):
                    # D 的 M(=K) → warp_col, D 的 N(=V) → warp_row
                    # trans 后: V=warp_row, K=warp_col → 匹配 h_accs

                    # A: lds_k[K, BT+pad], K 行 BT 列 (BT innermost)
                    # lane%32 → M(=K), 8 连续 BT from lane//32
                    a_k_row = (
                        fx.Index(kh * 64) + fx.Index(warp_col * 32) + fx.Index(lane_row)
                    )
                    a_bt_col = fx.Index(k_step * 16) + fx.Index(lane_half) * fx.Index(8)
                    a_idx = a_k_row * fx.Index(LDS_K_STRIDE) + a_bt_col
                    a_frag = _lds_read_bf16x8(lds_k_memref, a_idx)

                    # B: vn[BT, V] 存为 scratch[V, BT] → 读 scratch[V_row, BT_col]
                    # lane%32 → N=V col → V_row in scratch
                    # 8 连续 BT → BT_col = k_step*16 + lane_half*8
                    b_v_row = fx.Index(warp_row * 32) + fx.Index(lane_row)
                    b_bt_col = fx.Index(k_step * 16) + fx.Index(lane_half) * fx.Index(8)
                    b_idx = b_v_row * fx.Index(LDS_SCRATCH_STRIDE) + b_bt_col
                    b_frag = _lds_read_bf16x8(lds_scratch_memref, b_idx)

                    gemm2_acc = _mfma_bf16_32x32x16(a_frag, b_frag, gemm2_acc)

                gpu.barrier()

                # 零开销 trans: D[K_sub, V_sub] → h[V_sub, K_sub]
                # 同一寄存器内容，语义重解释后直接 add
                h_accs_in[kh] = h_accs_in[kh] + gemm2_acc

            # ========================================
            # yield
            # ========================================
            yield_list = [_to_raw(v) for v in h_accs_in]
            results = yield yield_list

        h_accs_final = list(results[:NUM_H_ACCS])

        # ===== Epilogue: final state store =====
        if const_expr(STORE_FINAL_STATE):
            for kh in range_constexpr(NUM_H_ACCS):
                acc_val = h_accs_final[kh]
                for ei in range_constexpr(MFMA_ACC_SIZE):
                    v_local = (
                        fx.Int32((ei // 4) * 8)
                        + lane_half * fx.Int32(4)
                        + fx.Int32(ei % 4)
                    )
                    k_local = lane_row
                    v_global = i_v * fx.Int32(BV) + warp_row * fx.Int32(32) + v_local
                    k_global = fx.Int32(kh * 64) + warp_col * fx.Int32(32) + k_local
                    ht_off = ht_base + v_global * fx.Int32(K) + k_global
                    elem = vector.extract(
                        acc_val, static_position=[ei], dynamic_position=[]
                    )
                    if const_expr(STATE_DTYPE_BF16):
                        ht_[fx.Index(ht_off)] = arith.truncf(T.bf16, elem)
                    else:
                        ht_[fx.Index(ht_off)] = elem

    # ===== Host launch =====
    @flyc.jit
    def launch_gdn_h(
        k_tensor: fx.Pointer,
        v_tensor: fx.Pointer,
        w_tensor: fx.Pointer,
        v_new_tensor: fx.Pointer,
        g_tensor: fx.Pointer,
        gk_tensor: fx.Pointer,
        h_tensor: fx.Pointer,
        h0_tensor: fx.Pointer,
        ht_tensor: fx.Pointer,
        cu_seqlens_tensor: fx.Pointer,
        chunk_offsets_tensor: fx.Pointer,
        T_val: fx.Int32,
        T_flat: fx.Int32,
        N_val: fx.Int32,
        grid_v: fx.Int32,
        grid_nh: fx.Int32,
        stream: fx.Stream,
    ):
        allocator.finalized = False
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            allocator.finalize()

        launcher = gdn_h_kernel(
            k_tensor,
            v_tensor,
            w_tensor,
            v_new_tensor,
            g_tensor,
            gk_tensor,
            h_tensor,
            h0_tensor,
            ht_tensor,
            cu_seqlens_tensor,
            chunk_offsets_tensor,
            T_val,
            T_flat,
            N_val,
        )
        launcher.launch(
            grid=(grid_v, grid_nh, 1),
            block=(BLOCK_THREADS, 1, 1),
            stream=stream,
        )

    return launch_gdn_h
