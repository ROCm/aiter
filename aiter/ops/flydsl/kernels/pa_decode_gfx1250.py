# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Paged-attention decode kernel for gfx1250 (WMMA, 4-warp).

Layout:
  query        [num_seqs, num_q_heads, HEAD_SIZE]             bf16/f16
  key_cache    [num_blocks, num_kv_heads, KV_BLOCK_SIZE, HEAD_SIZE]
  value_cache  [num_blocks, num_kv_heads, KV_BLOCK_SIZE, HEAD_SIZE]
  block_tables [num_seqs, max_num_blocks_per_seq]             i32
  seq_lens     [num_seqs]                                     i32

Per workgroup (grid = num_seqs x num_kv_heads x num_partitions):
  - 128 threads = 4 wave32 warps.
  - Each warp redundantly computes QK (8 WMMAs for HEAD=128, KVB=32).
  - Each warp handles `HEAD_SIZE/16/NUM_WARPS` N-tiles of PV.

Requirements:
  HEAD_SIZE         multiple of NUM_WARPS*WAVE_SIZE = 128
  KV_BLOCK_SIZE     multiple of WMMA_K = 32  (no block gather in this step)
  QUERY_GROUP_SIZE  exactly 16 (== WMMA_M)
  HEAD_SIZE         multiple of NUM_WARPS*WMMA_N = 64

WMMA instruction: wmma_f32_16x16x32_bf16 (or f16), wave32.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx

from flydsl._mlir import ir
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import arith, buffer_ops, gpu, range_constexpr, rocdl, tdm_ops, vector
from flydsl.expr.arith import _to_raw as _raw
from flydsl.expr.typing import T
from flydsl.expr.numeric import Float32 as fxFloat32, Int32 as fxInt32
from flydsl.runtime.device import get_rocm_arch as get_hip_arch
from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr

WAVE_SIZE = 32      # gfx1250 wave32
NUM_WARPS = 4       # 4 warps = 128 threads
WMMA_M = 16
WMMA_N = 16
WMMA_K = 32


def compile_pa_decode_main(
    *,
    HEAD_SIZE: int = 128,
    KV_BLOCK_SIZE: int = 32,
    QUERY_GROUP_SIZE: int = 16,
    PARTITION_SIZE: int = 256,
    dtype: str = "bf16",
):
    if HEAD_SIZE % 32 != 0:
        raise ValueError(f"HEAD_SIZE must be multiple of 32, got {HEAD_SIZE}")
    if KV_BLOCK_SIZE % 32 != 0:
        raise ValueError(f"KV_BLOCK_SIZE must be multiple of 32, got {KV_BLOCK_SIZE}")
    if PARTITION_SIZE % KV_BLOCK_SIZE != 0:
        raise ValueError(
            f"PARTITION_SIZE {PARTITION_SIZE} must be multiple of KV_BLOCK_SIZE {KV_BLOCK_SIZE}"
        )
    if QUERY_GROUP_SIZE != WMMA_M:
        raise ValueError(f"QUERY_GROUP_SIZE must be exactly {WMMA_M}, got {QUERY_GROUP_SIZE}")
    if dtype not in ("bf16", "f16"):
        raise ValueError(f"dtype must be 'bf16' or 'f16', got {dtype!r}")
    # Each warp owns HEAD_SIZE/NUM_WARPS output cols — must be multiple of WMMA_N.
    if HEAD_SIZE % (NUM_WARPS * WMMA_N) != 0:
        raise ValueError(
            f"HEAD_SIZE must be multiple of NUM_WARPS*WMMA_N="
            f"{NUM_WARPS * WMMA_N}, got {HEAD_SIZE}"
        )

    block_threads = NUM_WARPS * WAVE_SIZE   # 128
    K_QK_TILES = HEAD_SIZE // WMMA_K        # HEAD/32 = 4
    N_QK_TILES = KV_BLOCK_SIZE // WMMA_N    # KVB/16 = 2 (for KVB=32)
    K_PV_TILES = KV_BLOCK_SIZE // WMMA_K    # KVB/32 = 1 (for KVB=32)
    HEAD_PER_WARP = HEAD_SIZE // NUM_WARPS  # 32 for HEAD=128
    N_PV_TILES_PER_WARP = HEAD_PER_WARP // WMMA_N  # 2 for HEAD=128
    BLOCKS_PER_PARTITION = PARTITION_SIZE // KV_BLOCK_SIZE

    elem_bytes = 2  # bf16/f16

    gpu_arch = str(get_hip_arch(timeout_s=300))
    assert gpu_arch.startswith("gfx1250"), f"Expected gfx1250, got {gpu_arch}"

    # LDS layout (2-stage double-buffer for K and V):
    #   Q     [QG=16,  HEAD]    dtype       (persistent)
    #   K[0]  [KVB,    HEAD]    dtype       (stage 0)
    #   V[0]  [KVB,    HEAD]    dtype       (stage 0)
    #   K[1]  [KVB,    HEAD]    dtype       (stage 1)
    #   V[1]  [KVB,    HEAD]    dtype       (stage 1)
    #   P     [QG=16,  KVB]     dtype
    allocator = SmemAllocator(None, arch=gpu_arch, global_sym_name="pa_decode_main_lds")
    q_lds_offset = 0
    q_lds_bytes = QUERY_GROUP_SIZE * HEAD_SIZE * elem_bytes
    kv_one_bytes = KV_BLOCK_SIZE * HEAD_SIZE * elem_bytes
    k_stage_offsets = [q_lds_offset + q_lds_bytes + 2 * i * kv_one_bytes for i in range(2)]
    v_stage_offsets = [k + kv_one_bytes for k in k_stage_offsets]
    p_lds_offset = q_lds_offset + q_lds_bytes + 4 * kv_one_bytes
    p_lds_bytes = QUERY_GROUP_SIZE * KV_BLOCK_SIZE * elem_bytes
    allocator.ptr = p_lds_offset + p_lds_bytes

    q_lds_elems = QUERY_GROUP_SIZE * HEAD_SIZE
    kv_lds_elems = KV_BLOCK_SIZE * HEAD_SIZE
    p_lds_elems = QUERY_GROUP_SIZE * KV_BLOCK_SIZE

    # Striped-load counts (Q load stays as buffer_load; K/V go through TDM)
    Q_TOTAL_ELEMS = QUERY_GROUP_SIZE * HEAD_SIZE
    Q_ELEMS_PER_THREAD = (Q_TOTAL_ELEMS + block_threads - 1) // block_threads

    @flyc.kernel
    def kernel_pa_decode_main(
        arg_out: fx.Tensor,
        arg_max_logits: fx.Tensor,
        arg_exp_sums: fx.Tensor,
        arg_query: fx.Tensor,
        arg_key_cache: fx.Tensor,
        arg_value_cache: fx.Tensor,
        arg_block_tables: fx.Tensor,
        arg_seq_lens: fx.Tensor,
        i32_qk_scale: fx.Int32,
        i32_num_seqs: fx.Int32,
        i32_num_kv_heads: fx.Int32,
        i32_num_parts: fx.Int32,
        i32_max_blocks_per_seq: fx.Int32,
    ):
        elem_ty = T.bf16 if dtype == "bf16" else T.f16
        wmma_op = (
            rocdl.wmma_f32_16x16x32_bf16
            if dtype == "bf16"
            else rocdl.wmma_f32_16x16x32_f16
        )

        tx = gpu.thread_id("x")
        seq_idx = gpu.block_id("x")
        kv_head = gpu.block_id("y")
        part_idx = gpu.block_id("z")

        warp_id = tx / arith.index(WAVE_SIZE)
        lane_id = tx % arith.index(WAVE_SIZE)
        lane_kgrp = lane_id / arith.index(WMMA_M)   # 0 or 1
        lane16 = lane_id % arith.index(WMMA_M)       # 0..15

        num_kv_heads = arith.index_cast(T.index, i32_num_kv_heads.ir_value())
        num_parts = arith.index_cast(T.index, i32_num_parts.ir_value())
        max_blocks = arith.index_cast(T.index, i32_max_blocks_per_seq.ir_value())

        # --- Runtime strides ---
        num_q_heads = num_kv_heads * arith.index(QUERY_GROUP_SIZE)
        stride_q_seq = num_q_heads * arith.index(HEAD_SIZE)
        stride_q_head = arith.index(HEAD_SIZE)

        stride_kv_blk = num_kv_heads * arith.index(KV_BLOCK_SIZE * HEAD_SIZE)
        stride_kv_head = arith.index(KV_BLOCK_SIZE * HEAD_SIZE)
        stride_kv_tok = arith.index(HEAD_SIZE)

        stride_bt_seq = max_blocks

        stride_o_seq = num_kv_heads * num_parts * arith.index(QUERY_GROUP_SIZE * HEAD_SIZE)
        stride_o_head = num_parts * arith.index(QUERY_GROUP_SIZE * HEAD_SIZE)
        stride_o_part = arith.index(QUERY_GROUP_SIZE * HEAD_SIZE)
        stride_o_row = arith.index(HEAD_SIZE)

        stride_lse_seq = num_kv_heads * num_parts * arith.index(QUERY_GROUP_SIZE)
        stride_lse_head = num_parts * arith.index(QUERY_GROUP_SIZE)
        stride_lse_part = arith.index(QUERY_GROUP_SIZE)

        # --- Buffer resources ---
        q_rsrc = buffer_ops.create_buffer_resource(arg_query, max_size=True)
        k_rsrc = buffer_ops.create_buffer_resource(arg_key_cache, max_size=True)
        v_rsrc = buffer_ops.create_buffer_resource(arg_value_cache, max_size=True)
        bt_rsrc = buffer_ops.create_buffer_resource(arg_block_tables, max_size=True)
        sl_rsrc = buffer_ops.create_buffer_resource(arg_seq_lens, max_size=True)
        out_rsrc = buffer_ops.create_buffer_resource(arg_out, max_size=True)
        ml_rsrc = buffer_ops.create_buffer_resource(arg_max_logits, max_size=True)
        es_rsrc = buffer_ops.create_buffer_resource(arg_exp_sums, max_size=True)

        # Sequence length
        seq_len_i32 = buffer_ops.buffer_load(sl_rsrc, seq_idx, vec_width=1, dtype=T.i32)
        seq_len = arith.index_cast(T.index, seq_len_i32)

        # --- LDS setup ---
        base = allocator.get_base()
        q_lds = SmemPtr(base, q_lds_offset, elem_ty, shape=(q_lds_elems,))
        k_lds_stages = [
            SmemPtr(base, k_stage_offsets[i], elem_ty, shape=(kv_lds_elems,))
            for i in range(2)
        ]
        v_lds_stages = [
            SmemPtr(base, v_stage_offsets[i], elem_ty, shape=(kv_lds_elems,))
            for i in range(2)
        ]
        p_lds = SmemPtr(base, p_lds_offset, elem_ty, shape=(p_lds_elems,))
        q_lds.get()
        k_lds_stages[0].get()
        k_lds_stages[1].get()
        v_lds_stages[0].get()
        v_lds_stages[1].get()
        p_lds.get()

        # ---------------- Load Q into LDS (striped across 128 threads) ----------------
        q_seq_base = seq_idx * stride_q_seq + kv_head * arith.index(QUERY_GROUP_SIZE) * stride_q_head
        for e in range_constexpr(Q_ELEMS_PER_THREAD):
            idx = tx + arith.index(e * block_threads)
            in_range = arith.cmpi(
                arith.CmpIPredicate.slt, _raw(idx), _raw(arith.index(Q_TOTAL_ELEMS))
            )
            safe_idx = arith.select(in_range, _raw(idx), _raw(arith.index(0)))
            row = safe_idx / arith.index(HEAD_SIZE)
            col = safe_idx % arith.index(HEAD_SIZE)
            g_off = q_seq_base + row * stride_q_head + col
            v_i16 = buffer_ops.buffer_load(q_rsrc, g_off, vec_width=1, dtype=T.i16)
            v_dt = arith.bitcast(elem_ty, v_i16)
            q_lds.store(v_dt, [row * arith.index(HEAD_SIZE) + col])

        gpu.barrier()

        # --- qk_scale in log2 domain (so we can use exp2) ---
        qk_scale_f32 = arith.bitcast(T.f32, _raw(i32_qk_scale))
        LOG2E = arith.constant(1.4426950408889634, type=T.f32)
        qk_scale_log2_scalar = arith.mulf(qk_scale_f32, LOG2E)
        qk_scale_log2_vec = vector.broadcast(T.vec(8, T.f32), qk_scale_log2_scalar)

        # --- Running state: per-lane 8 rows (indexed by mi), 8 scalars each ---
        neg_inf_f32 = arith.constant(float("-inf"), type=T.f32)
        zero_f32 = arith.constant(0.0, type=T.f32)

        m_state = [neg_inf_f32] * 8     # per-row max, for rows [lane_kgrp*8 + mi]
        l_state = [zero_f32] * 8        # per-row exp_sum

        # PV accumulator: per warp N_PV_TILES_PER_WARP vec<8xf32>
        pv_accs = [
            arith.constant_vector(0.0, T.vec(8, T.f32))
            for _ in range(N_PV_TILES_PER_WARP)
        ]

        def _load_wmma_A_frag(lds_ptr, row_base_idx, k_base_elem, row_stride):
            """Load a vec<16xelem> WMMA A fragment.

            Layout: per lane vals[k0*8 + k1] at (row=row_base_idx, K-col = k_base_elem +
            (k0*2 + lane_kgrp) * 8 + k1), k0 in [0,1], k1 in [0,7].
            """
            vals = []
            for k0 in range_constexpr(2):
                for k1 in range_constexpr(8):
                    kk = (
                        arith.index(k_base_elem)
                        + (arith.index(k0 * 2) + lane_kgrp) * arith.index(8)
                        + arith.index(k1)
                    )
                    off = row_base_idx * arith.index(row_stride) + kk
                    vals.append(lds_ptr.load([off]))
            return vector.from_elements(T.vec(16, elem_ty), vals)

        # Pre-load Q fragments (one per K-sub-tile, held in registers across the loop).
        q_frags = []
        for ks in range_constexpr(K_QK_TILES): # each warp has a complete copy of Q
            q_frags.append(_load_wmma_A_frag(q_lds, lane16, ks * WMMA_K, HEAD_SIZE))

        # ---------------- TDM async load helpers ----------------
        def _phys_blk(logical_blk_idx):
            bt_off = seq_idx * stride_bt_seq + logical_blk_idx
            phys_i32 = buffer_ops.buffer_load(bt_rsrc, bt_off, vec_width=1, dtype=T.i32)
            return arith.index_cast(T.index, phys_i32)

        def _issue_kv_load(kv_tensor, lds_memref, phys_blk):
            """Issue a TDM async 2D load of [KVB, HEAD] from kv_tensor into lds_memref.

            Treat kv_tensor flat-view as a 2D tensor of HEAD-strided rows. The starting
            row index for block ``phys_blk`` at head ``kv_head`` is
            ``phys_blk * num_kv_heads * KVB + kv_head * KVB``.
            """
            outer_off = (phys_blk * num_kv_heads + kv_head) * arith.index(KV_BLOCK_SIZE)
            desc = tdm_ops.make_tensor_descriptor_2d(
                global_ptr=kv_tensor,
                lds_memref=lds_memref,
                global_offset=(outer_off, arith.index(0)),
                tensor_shape=(KV_BLOCK_SIZE, HEAD_SIZE),
                strides=(HEAD_SIZE, 1),
                tile_shape=(KV_BLOCK_SIZE, HEAD_SIZE),
                elem_bytes=elem_bytes,
                num_warps=NUM_WARPS,
            )
            tdm_ops.tensor_load_2d(desc)

        # ---------------- Prologue: issue loads for iteration 0 ----------------
        logical_blk0 = part_idx * arith.index(BLOCKS_PER_PARTITION)
        phys_blk0 = _phys_blk(logical_blk0)
        _issue_kv_load(arg_key_cache, k_lds_stages[0].get(), phys_blk0)
        _issue_kv_load(arg_value_cache, v_lds_stages[0].get(), phys_blk0)

        # ---------------- Main loop over KV blocks ----------------
        for b_in_part in range_constexpr(BLOCKS_PER_PARTITION): # need to fix this it is wasting compute in case actual_num_blocks < blocks_per_partition
            cur_stage = b_in_part % 2
            nxt_stage = (b_in_part + 1) % 2
            logical_blk = part_idx * arith.index(BLOCKS_PER_PARTITION) + arith.index(b_in_part)
            block_first_tok = logical_blk * arith.index(KV_BLOCK_SIZE)

            # Prefetch next block's K, V if it exists
            if b_in_part < BLOCKS_PER_PARTITION - 1:
                next_logical_blk = (
                    part_idx * arith.index(BLOCKS_PER_PARTITION) + arith.index(b_in_part + 1)
                )
                next_phys_blk = _phys_blk(next_logical_blk)
                _issue_kv_load(arg_key_cache, k_lds_stages[nxt_stage].get(), next_phys_blk)
                _issue_kv_load(arg_value_cache, v_lds_stages[nxt_stage].get(), next_phys_blk)
                # Wait for current stage loads (let 2 next loads remain outstanding)
                tdm_ops.tensor_wait(2)
            else:
                tdm_ops.tensor_wait(0)
            gpu.barrier()

            # Current stage pointers
            k_lds = k_lds_stages[cur_stage]
            v_lds = v_lds_stages[cur_stage]

            # ------------------------ QK WMMA ------------------------
            qk_accs = [
                arith.constant_vector(0.0, T.vec(8, T.f32)) for _ in range(N_QK_TILES) # 16x16 tiles there are N_QK_TILES such tiles. each thread owns 8 elems per tile
            ]
            for n_tile in range_constexpr(N_QK_TILES):
                n_row_lds = arith.index(n_tile * WMMA_N) + lane16
                for ks in range_constexpr(K_QK_TILES):
                    k_frag = _load_wmma_A_frag(
                        k_lds, n_row_lds, ks * WMMA_K, HEAD_SIZE
                    )
                    qk_accs[n_tile] = wmma_op(
                        T.vec(8, T.f32),
                        q_frags[ks],
                        k_frag,
                        qk_accs[n_tile],
                        signA=False,
                        signB=False,
                        modC=0,
                        reuseA=False,
                        reuseB=False,
                    ).result

            # Scale by qk_scale * log2e so we can use exp2 later.
            for n_tile in range_constexpr(N_QK_TILES):
                qk_accs[n_tile] = arith.mulf(qk_accs[n_tile], qk_scale_log2_vec)

            # Mask out-of-range tokens (col = block_first_tok + n_tile*16 + lane16)
            neg_inf_vec8 = arith.constant_vector(float("-inf"), T.vec(8, T.f32))
            for n_tile in range_constexpr(N_QK_TILES):
                tok_abs = block_first_tok + arith.index(n_tile * WMMA_N) + lane16
                in_range = arith.cmpi(
                    arith.CmpIPredicate.slt, _raw(tok_abs), _raw(seq_len)
                )
                qk_accs[n_tile] = arith.select(
                    in_range, _raw(qk_accs[n_tile]), _raw(neg_inf_vec8)
                )

            # ------------------------ Online softmax ------------------------
            # Step 1: per-lane, per-row: local max across N-tiles (scalar).
            row_local_max = []
            for mi in range_constexpr(8):
                cur = vector.extract(
                    qk_accs[0], static_position=[mi], dynamic_position=[]
                )
                if N_QK_TILES > 1:
                    for n_tile in range_constexpr(1, N_QK_TILES):
                        v = vector.extract(
                            qk_accs[n_tile], static_position=[mi], dynamic_position=[]
                        )
                        cur = arith.maximumf(cur, v)
                row_local_max.append(cur)

            # Step 2: reduce across lane16 dim (xor 1,2,4,8) → same-row lanes agree.
            row_max_reduced = []
            for mi in range_constexpr(8):
                x = fxFloat32(row_local_max[mi])
                for sh in [1, 2, 4, 8]:
                    peer = x.shuffle_xor(fxInt32(sh), fxInt32(WAVE_SIZE))
                    x = x.maximumf(peer)
                row_max_reduced.append(x.ir_value())

            # Step 3: update m_state, compute alpha for PV rescale.
            alphas = []
            new_m_list = []
            for mi in range_constexpr(8):
                new_m_mi = arith.maximumf(m_state[mi], row_max_reduced[mi])
                alpha_mi = rocdl.exp2(T.f32, arith.subf(m_state[mi], new_m_mi))
                alphas.append(alpha_mi)
                new_m_list.append(new_m_mi)

            # Step 4: compute p = exp2(qk - new_m) in each acc, also accumulate per-row sum.
            # p_accs: same shape as qk_accs but with p values.
            # row_sum_partial[mi] = per-lane partial sum over N-tiles for row mi.
            row_sum_partial = [zero_f32] * 8
            p_accs = []
            for n_tile in range_constexpr(N_QK_TILES):
                vals_new = []
                for mi in range_constexpr(8):
                    q_ij = vector.extract(
                        qk_accs[n_tile], static_position=[mi], dynamic_position=[]
                    )
                    p_ij = rocdl.exp2(T.f32, arith.subf(q_ij, new_m_list[mi]))
                    vals_new.append(p_ij)
                    row_sum_partial[mi] = arith.addf(row_sum_partial[mi], p_ij)
                p_accs.append(
                    vector.from_elements(T.vec(8, T.f32), vals_new)
                )

            # Step 5: reduce row_sum_partial across lane16 dim.
            row_sum_reduced = []
            for mi in range_constexpr(8):
                x = fxFloat32(row_sum_partial[mi])
                for sh in [1, 2, 4, 8]:
                    peer = x.shuffle_xor(fxInt32(sh), fxInt32(WAVE_SIZE))
                    x = x + peer
                row_sum_reduced.append(x.ir_value())

            # Step 6: update l_state: l_new = alpha * l_old + row_sum_reduced.
            for mi in range_constexpr(8):
                l_state[mi] = arith.addf(
                    arith.mulf(alphas[mi], l_state[mi]), row_sum_reduced[mi]
                )
                m_state[mi] = new_m_list[mi]

            # Step 7: Rescale PV accumulator by alpha (per-row).
            # pv_accs[n][mi] corresponds to row (lane_kgrp*8 + mi). Multiply by alphas[mi].
            for pv_n in range_constexpr(N_PV_TILES_PER_WARP):
                new_vals = []
                for mi in range_constexpr(8):
                    v = vector.extract(
                        pv_accs[pv_n], static_position=[mi], dynamic_position=[]
                    )
                    v = arith.mulf(v, alphas[mi])
                    new_vals.append(v)
                pv_accs[pv_n] = vector.from_elements(T.vec(8, T.f32), new_vals)

            # Step 8: Write P to LDS in bf16 layout [QG=16, KVB]. Per lane 8 rows × N_QK_TILES cols.
            # Offset: (row = lane_kgrp*8 + mi) * KVB + (col = n_tile*16 + lane16)
            # All 4 warps write redundantly (same P values).
            for n_tile in range_constexpr(N_QK_TILES):
                p_vec = p_accs[n_tile]
                # cast vec<8xf32> → vec<8xbf16>
                p_half = arith.trunc_f(T.vec(8, elem_ty), p_vec)
                for mi in range_constexpr(8):
                    row = lane_kgrp * arith.index(8) + arith.index(mi)
                    col = arith.index(n_tile * WMMA_N) + lane16
                    off = row * arith.index(KV_BLOCK_SIZE) + col
                    v = vector.extract(
                        p_half, static_position=[mi], dynamic_position=[]
                    )
                    p_lds.store(v, [off])

            gpu.barrier()

            # ------------------------ PV WMMA ------------------------
            # Each warp owns N_PV_TILES_PER_WARP output N-tiles (HEAD cols).
            # global_n_tile = warp_id * N_PV_TILES_PER_WARP + pv_n
            for pv_n in range_constexpr(N_PV_TILES_PER_WARP):
                global_n_tile = warp_id * arith.index(N_PV_TILES_PER_WARP) + arith.index(pv_n)
                for ks in range_constexpr(K_PV_TILES):
                    # P fragment from P LDS: A operand [M=16, K=32]
                    # row = lane16, K-col = ks*32 + kk
                    p_frag = _load_wmma_A_frag(
                        p_lds, lane16, ks * WMMA_K, KV_BLOCK_SIZE
                    )
                    # V fragment from KV LDS: B operand [K=32, N=16]
                    # KV LDS layout [KVB, HEAD]. B operand wants row=K (KVB axis), col=N (HEAD axis).
                    # Per lane: at (K-row=kk, N-col=lane16).
                    # kk here indexes KVB via ks*32 + relative_kk.
                    # N col = global_n_tile*16 + lane16 (HEAD axis).
                    v_vals = []
                    for k0 in range_constexpr(2):
                        for k1 in range_constexpr(8):
                            kk = (
                                arith.index(ks * WMMA_K)
                                + (arith.index(k0 * 2) + lane_kgrp) * arith.index(8)
                                + arith.index(k1)
                            )
                            n_col = global_n_tile * arith.index(WMMA_N) + lane16
                            off = kk * arith.index(HEAD_SIZE) + n_col
                            v_vals.append(v_lds.load([off]))
                    v_frag = vector.from_elements(T.vec(16, elem_ty), v_vals)

                    pv_accs[pv_n] = wmma_op(
                        T.vec(8, T.f32),
                        p_frag,
                        v_frag,
                        pv_accs[pv_n],
                        signA=False,
                        signB=False,
                        modC=0,
                        reuseA=False,
                        reuseB=False,
                    ).result

            gpu.barrier()

        # ---------------- Write per-partition results ----------------
        out_base = (
            seq_idx * stride_o_seq
            + kv_head * stride_o_head
            + part_idx * stride_o_part
        )
        lse_base = (
            seq_idx * stride_lse_seq
            + kv_head * stride_lse_head
            + part_idx * stride_lse_part
        )

        # Accumulator: each warp writes its N_PV_TILES_PER_WARP × vec<8xf32>
        # WMMA C layout: row = lane_kgrp*8 + mi, col = lane16 within the 16-col N-tile
        for pv_n in range_constexpr(N_PV_TILES_PER_WARP):
            global_n_tile = warp_id * arith.index(N_PV_TILES_PER_WARP) + arith.index(pv_n)
            for mi in range_constexpr(8):
                row = lane_kgrp * arith.index(8) + arith.index(mi)
                col = global_n_tile * arith.index(WMMA_N) + lane16
                off = out_base + row * stride_o_row + col
                val = vector.extract(
                    pv_accs[pv_n], static_position=[mi], dynamic_position=[]
                )
                buffer_ops.buffer_store(val, out_rsrc, off)

        # max_logits / exp_sums: per-lane 8 rows, only lane16==0 writes to avoid 16× redundant writes.
        # But they all have same values, so benign-race writes are fine.
        for mi in range_constexpr(8):
            row_idx = lane_kgrp * arith.index(8) + arith.index(mi)
            off = lse_base + row_idx
            buffer_ops.buffer_store(m_state[mi], ml_rsrc, off)
            buffer_ops.buffer_store(l_state[mi], es_rsrc, off)

    cache_tag = (HEAD_SIZE, KV_BLOCK_SIZE, QUERY_GROUP_SIZE, PARTITION_SIZE, dtype, NUM_WARPS)

    @flyc.jit
    def launch_pa_decode_main(
        arg_out: fx.Tensor,
        arg_max_logits: fx.Tensor,
        arg_exp_sums: fx.Tensor,
        arg_query: fx.Tensor,
        arg_key_cache: fx.Tensor,
        arg_value_cache: fx.Tensor,
        arg_block_tables: fx.Tensor,
        arg_seq_lens: fx.Tensor,
        i32_qk_scale: fx.Int32,
        i32_num_seqs: fx.Int32,
        i32_num_kv_heads: fx.Int32,
        i32_num_parts: fx.Int32,
        i32_max_blocks_per_seq: fx.Int32,
        stream: fx.Stream,
    ):
        _ = cache_tag
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            allocator.finalized = False
            allocator.finalize()

        gx = _raw(arith.index_cast(T.index, i32_num_seqs.ir_value()))
        gy = _raw(arith.index_cast(T.index, i32_num_kv_heads.ir_value()))
        gz = _raw(arith.index_cast(T.index, i32_num_parts.ir_value()))

        kernel_pa_decode_main(
            arg_out, arg_max_logits, arg_exp_sums,
            arg_query, arg_key_cache, arg_value_cache,
            arg_block_tables, arg_seq_lens,
            i32_qk_scale, i32_num_seqs, i32_num_kv_heads,
            i32_num_parts, i32_max_blocks_per_seq,
        ).launch(
            grid=(gx, gy, gz),
            block=(block_threads, 1, 1),
            stream=stream,
        )

    return launch_pa_decode_main


# ============================================================================
# Reduce kernel (unchanged; 1 warp, scalar)
# ============================================================================

def compile_pa_decode_reduce(
    *,
    HEAD_SIZE: int = 128,
    QUERY_GROUP_SIZE: int = 16,
    PARTITION_SIZE: int = 256,
    dtype: str = "bf16",
):
    if HEAD_SIZE % 16 != 0:
        raise ValueError("HEAD_SIZE must be multiple of 16")
    if dtype not in ("bf16", "f16"):
        raise ValueError(f"dtype must be 'bf16' or 'f16', got {dtype!r}")
    if HEAD_SIZE % WAVE_SIZE != 0:
        raise ValueError(f"HEAD_SIZE must be multiple of {WAVE_SIZE}")

    block_threads = WAVE_SIZE
    COLS_PER_THREAD = HEAD_SIZE // WAVE_SIZE

    allocator = SmemAllocator(None, arch="gfx1250", global_sym_name="pa_decode_red_lds")
    allocator.ptr = 4

    @flyc.kernel
    def kernel_pa_decode_reduce(
        arg_out: fx.Tensor,
        arg_tmp_out: fx.Tensor,
        arg_max_logits: fx.Tensor,
        arg_exp_sums: fx.Tensor,
        arg_seq_lens: fx.Tensor,
        i32_num_seqs: fx.Int32,
        i32_num_kv_heads: fx.Int32,
        i32_num_parts: fx.Int32,
    ):
        elem_ty = T.bf16 if dtype == "bf16" else T.f16

        tx = gpu.thread_id("x")
        seq_idx = gpu.block_id("x")
        kv_head = gpu.block_id("y")

        num_kv_heads = arith.index_cast(T.index, i32_num_kv_heads.ir_value())
        num_parts = arith.index_cast(T.index, i32_num_parts.ir_value())
        num_q_heads = num_kv_heads * arith.index(QUERY_GROUP_SIZE)

        sl_rsrc = buffer_ops.create_buffer_resource(arg_seq_lens, max_size=True)
        seq_len_i32 = buffer_ops.buffer_load(sl_rsrc, seq_idx, vec_width=1, dtype=T.i32)
        seq_len = arith.index_cast(T.index, seq_len_i32)
        part_size_idx = arith.index(PARTITION_SIZE)
        num_parts_actual = (seq_len + part_size_idx - arith.index(1)) / part_size_idx

        stride_tmp_seq = num_kv_heads * num_parts * arith.index(QUERY_GROUP_SIZE * HEAD_SIZE)
        stride_tmp_head = num_parts * arith.index(QUERY_GROUP_SIZE * HEAD_SIZE)
        stride_tmp_part = arith.index(QUERY_GROUP_SIZE * HEAD_SIZE)
        stride_tmp_row = arith.index(HEAD_SIZE)

        stride_lse_seq = num_kv_heads * num_parts * arith.index(QUERY_GROUP_SIZE)
        stride_lse_head = num_parts * arith.index(QUERY_GROUP_SIZE)
        stride_lse_part = arith.index(QUERY_GROUP_SIZE)

        stride_out_seq = num_q_heads * arith.index(HEAD_SIZE)
        stride_out_head = arith.index(HEAD_SIZE)

        tmp_rsrc = buffer_ops.create_buffer_resource(arg_tmp_out, max_size=True)
        ml_rsrc = buffer_ops.create_buffer_resource(arg_max_logits, max_size=True)
        es_rsrc = buffer_ops.create_buffer_resource(arg_exp_sums, max_size=True)
        out_rsrc = buffer_ops.create_buffer_resource(arg_out, max_size=True)

        zero_f32 = arith.constant(0.0, type=T.f32)
        neg_inf_f32 = arith.constant(float("-inf"), type=T.f32)
        zero_idx = arith.index(0)
        one_idx = arith.index(1)

        for r in range_constexpr(QUERY_GROUP_SIZE):
            lse_row_base = (
                seq_idx * stride_lse_seq
                + kv_head * stride_lse_head
                + arith.index(r)
            )
            tmp_row_base = (
                seq_idx * stride_tmp_seq
                + kv_head * stride_tmp_head
                + arith.index(r) * stride_tmp_row
            )

            # Pass 1: global max + online exp_sum
            init_state = [neg_inf_f32, zero_f32]
            for p, state in range(zero_idx, num_parts_actual, one_idx, init=init_state):
                old_m = state[0]
                old_l = state[1]
                lse_off = lse_row_base + p * stride_lse_part
                this_m = buffer_ops.buffer_load(ml_rsrc, lse_off, vec_width=1, dtype=T.f32)
                this_l = buffer_ops.buffer_load(es_rsrc, lse_off, vec_width=1, dtype=T.f32)
                new_m = arith.maximumf(old_m, this_m)
                alpha_old = rocdl.exp2(T.f32, arith.subf(old_m, new_m))
                alpha_this = rocdl.exp2(T.f32, arith.subf(this_m, new_m))
                new_l = arith.addf(
                    arith.mulf(alpha_old, old_l), arith.mulf(alpha_this, this_l)
                )
                results1 = yield [new_m, new_l]
            g_max = results1[0]
            g_exp_sum = results1[1]

            # Pass 2: weighted sum
            for c in range_constexpr(COLS_PER_THREAD):
                col = tx + arith.index(c * WAVE_SIZE)

                init_acc = [zero_f32, zero_f32]
                for p, state in range(zero_idx, num_parts_actual, one_idx, init=init_acc):
                    old_acc = state[0]
                    lse_off = lse_row_base + p * stride_lse_part
                    this_m = buffer_ops.buffer_load(ml_rsrc, lse_off, vec_width=1, dtype=T.f32)
                    tmp_off = tmp_row_base + p * stride_tmp_part + col
                    this_v = buffer_ops.buffer_load(tmp_rsrc, tmp_off, vec_width=1, dtype=T.f32)
                    alpha = rocdl.exp2(T.f32, arith.subf(this_m, g_max))
                    new_acc = arith.addf(old_acc, arith.mulf(alpha, this_v))
                    results2 = yield [new_acc, state[1]]
                final_acc = arith.divf(results2[0], g_exp_sum)

                out_off = (
                    seq_idx * stride_out_seq
                    + (kv_head * arith.index(QUERY_GROUP_SIZE) + arith.index(r)) * stride_out_head
                    + col
                )
                final_half = arith.trunc_f(elem_ty, final_acc)
                final_i16 = arith.bitcast(T.i16, final_half)
                buffer_ops.buffer_store(final_i16, out_rsrc, out_off)

    cache_tag = (HEAD_SIZE, QUERY_GROUP_SIZE, PARTITION_SIZE, dtype)

    @flyc.jit
    def launch_pa_decode_reduce(
        arg_out: fx.Tensor,
        arg_tmp_out: fx.Tensor,
        arg_max_logits: fx.Tensor,
        arg_exp_sums: fx.Tensor,
        arg_seq_lens: fx.Tensor,
        i32_num_seqs: fx.Int32,
        i32_num_kv_heads: fx.Int32,
        i32_num_parts: fx.Int32,
        stream: fx.Stream,
    ):
        _ = cache_tag
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            allocator.finalized = False
            allocator.finalize()

        gx = _raw(arith.index_cast(T.index, i32_num_seqs.ir_value()))
        gy = _raw(arith.index_cast(T.index, i32_num_kv_heads.ir_value()))

        kernel_pa_decode_reduce(
            arg_out, arg_tmp_out, arg_max_logits, arg_exp_sums,
            arg_seq_lens, i32_num_seqs, i32_num_kv_heads, i32_num_parts,
        ).launch(
            grid=(gx, gy, 1),
            block=(block_threads, 1, 1),
            stream=stream,
        )

    return launch_pa_decode_reduce


__all__ = ["compile_pa_decode_main", "compile_pa_decode_reduce"]
