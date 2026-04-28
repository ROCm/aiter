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
  - QK is warp-split along the N (KVB) axis: each warp owns
    N_QK_TILES_PER_WARP = (KV_COMPUTE_BLOCK_SIZE/WMMA_N) / NUM_WARPS contiguous
    N-tiles, doing N_QK_TILES_PER_WARP * (HEAD_SIZE/WMMA_K) WMMAs.
  - Online softmax is reduced cross-warp through LDS scratch (max + sum slabs)
    aliased over the dead Q LDS region.
  - PV is warp-split along the N (HEAD) axis: each warp owns
    N_PV_TILES_PER_WARP = HEAD_SIZE/NUM_WARPS/WMMA_N output N-tiles, doing
    N_PV_TILES_PER_WARP * (KV_COMPUTE_BLOCK_SIZE/WMMA_K) WMMAs.

Requirements:
  HEAD_SIZE             multiple of NUM_WARPS*WAVE_SIZE = 128
  KV_BLOCK_SIZE         multiple of 16
  KV_COMPUTE_BLOCK_SIZE multiple of WMMA_K = 32, multiple of KV_BLOCK_SIZE
  PARTITION_SIZE        multiple of KV_COMPUTE_BLOCK_SIZE
  BLOCKS_PER_COMPUTE = KV_COMPUTE_BLOCK_SIZE / KV_BLOCK_SIZE  (any positive int;
                                                              BPC > 4 issues multiple buffer_loads)
  QUERY_GROUP_SIZE      exactly 16 (== WMMA_M)

WMMA instruction: wmma_f32_16x16x32_bf16 (or f16), wave32.

Each outer loop iteration processes one "compute tile" of KV_COMPUTE_BLOCK_SIZE
tokens, built by gathering BLOCKS_PER_COMPUTE physical blocks from the paged KV
cache. One vectorized block_tables load fetches all phys-block IDs per tile;
BLOCKS_PER_COMPUTE TDM ops DMA each block into its sub-slice of the K/V LDS
stage. 2-stage double buffer overlaps the next tile's loads with current compute.
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
    KV_COMPUTE_BLOCK_SIZE: int = 64,
    dtype: str = "bf16",
):
    if HEAD_SIZE % 32 != 0:
        raise ValueError(f"HEAD_SIZE must be multiple of 32, got {HEAD_SIZE}")
    # if KV_BLOCK_SIZE % 16 != 0:
    #     raise ValueError(f"KV_BLOCK_SIZE must be multiple of 16, got {KV_BLOCK_SIZE}")
    if KV_COMPUTE_BLOCK_SIZE % WMMA_K != 0:
        raise ValueError(
            f"KV_COMPUTE_BLOCK_SIZE must be multiple of WMMA_K={WMMA_K}, got {KV_COMPUTE_BLOCK_SIZE}"
        )
    if KV_COMPUTE_BLOCK_SIZE % KV_BLOCK_SIZE != 0:
        raise ValueError(
            f"KV_COMPUTE_BLOCK_SIZE {KV_COMPUTE_BLOCK_SIZE} must be multiple of "
            f"KV_BLOCK_SIZE {KV_BLOCK_SIZE}"
        )
    if PARTITION_SIZE % KV_COMPUTE_BLOCK_SIZE != 0:
        raise ValueError(
            f"PARTITION_SIZE {PARTITION_SIZE} must be multiple of "
            f"KV_COMPUTE_BLOCK_SIZE {KV_COMPUTE_BLOCK_SIZE}"
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
    # QK warp-split along N requires N_QK_TILES divisible by NUM_WARPS.
    if KV_COMPUTE_BLOCK_SIZE % (NUM_WARPS * WMMA_N) != 0:
        raise ValueError(
            f"KV_COMPUTE_BLOCK_SIZE must be multiple of NUM_WARPS*WMMA_N="
            f"{NUM_WARPS * WMMA_N}, got {KV_COMPUTE_BLOCK_SIZE}"
        )

    BLOCKS_PER_COMPUTE = KV_COMPUTE_BLOCK_SIZE // KV_BLOCK_SIZE
    # Physical block IDs are loaded via buffer_load with vec_width ∈ {1, 2, 4} on AMD.
    # For BPC > 4 we issue ceil(BPC/4) loads of size ≤ 4 each.
    BT_VEC_WIDTH = 4 if BLOCKS_PER_COMPUTE >= 4 else BLOCKS_PER_COMPUTE
    BT_NUM_LOADS = (BLOCKS_PER_COMPUTE + BT_VEC_WIDTH - 1) // BT_VEC_WIDTH
    BT_TAIL_WIDTH = BLOCKS_PER_COMPUTE - (BT_NUM_LOADS - 1) * BT_VEC_WIDTH

    block_threads = NUM_WARPS * WAVE_SIZE          # 128
    K_QK_TILES = HEAD_SIZE // WMMA_K               # HEAD/32 = 4
    N_QK_TILES = KV_COMPUTE_BLOCK_SIZE // WMMA_N   # compute/16 (e.g. 4 for compute=64)
    N_QK_TILES_PER_WARP = N_QK_TILES // NUM_WARPS  # 1 for compute=64, 2 for 128, 4 for 256
    K_PV_TILES = KV_COMPUTE_BLOCK_SIZE // WMMA_K   # compute/32 (e.g. 2 for compute=64)
    HEAD_PER_WARP = HEAD_SIZE // NUM_WARPS         # 32 for HEAD=128
    N_PV_TILES_PER_WARP = HEAD_PER_WARP // WMMA_N  # 2 for HEAD=128
    COMPUTES_PER_PARTITION = PARTITION_SIZE // KV_COMPUTE_BLOCK_SIZE

    elem_bytes = 2  # bf16/f16

    gpu_arch = str(get_hip_arch(timeout_s=300))
    assert gpu_arch.startswith("gfx1250"), f"Expected gfx1250, got {gpu_arch}"

    # NUM_KV_STAGES = 2 enables a software-pipelined prefetch of the next compute
    # tile's K/V into a separate LDS stage while the current tile is being
    # consumed. With COMPUTES_PER_PARTITION == 1 there's no "next tile" to
    # overlap with, so we collapse to a single stage and save half the K/V LDS.
    NUM_KV_STAGES = 2 if COMPUTES_PER_PARTITION > 1 else 1

    # LDS layout (NUM_KV_STAGES double-buffer for K and V, sized by KV_COMPUTE_BLOCK_SIZE):
    #   Q     [QG=16,  HEAD]                dtype       (persistent until read into q_frags)
    #   K[0]  [KV_COMPUTE_BLOCK_SIZE, HEAD] dtype       (stage 0, holds BLOCKS_PER_COMPUTE blocks)
    #   V[0]  [KV_COMPUTE_BLOCK_SIZE, HEAD] dtype       (stage 0)
    #   K[1]  [KV_COMPUTE_BLOCK_SIZE, HEAD] dtype       (stage 1, only if NUM_KV_STAGES==2)
    #   V[1]  [KV_COMPUTE_BLOCK_SIZE, HEAD] dtype       (stage 1, only if NUM_KV_STAGES==2)
    #   P     [QG=16, KV_COMPUTE_BLOCK_SIZE] dtype
    #
    # Cross-warp reduce scratch [NUM_WARPS, QG] f32 is aliased with Q LDS (Q is
    # unused after q_frags are read into registers at prologue; scratch only
    # lives during the softmax step, so they never overlap in time).
    allocator = SmemAllocator(None, arch=gpu_arch, global_sym_name="pa_decode_main_lds")
    q_lds_elems = QUERY_GROUP_SIZE * HEAD_SIZE
    kv_lds_elems = KV_COMPUTE_BLOCK_SIZE * HEAD_SIZE
    p_lds_elems = QUERY_GROUP_SIZE * KV_COMPUTE_BLOCK_SIZE
    q_lds_offset = 0
    q_lds_bytes = q_lds_elems * elem_bytes
    kv_one_bytes = kv_lds_elems * elem_bytes
    k_stage_offsets = [
        q_lds_offset + q_lds_bytes + 2 * i * kv_one_bytes
        for i in range(NUM_KV_STAGES)
    ]
    v_stage_offsets = [k + kv_one_bytes for k in k_stage_offsets]
    p_lds_offset = q_lds_offset + q_lds_bytes + 2 * NUM_KV_STAGES * kv_one_bytes
    p_lds_bytes = p_lds_elems * elem_bytes
    allocator.ptr = p_lds_offset + p_lds_bytes


    # Cross-warp reduce scratch: two slabs (one for max, one for sum) of
    # NUM_WARPS * QG f32 each. Alias both into Q's LDS region.
    scratch_slab_elems = NUM_WARPS * QUERY_GROUP_SIZE   # f32 elements per slab
    scratch_max_lds_offset = q_lds_offset
    scratch_sum_lds_offset = q_lds_offset + scratch_slab_elems * 4

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
        # ---- Minimal pre-check setup ----
        # Only what's needed to evaluate the early-exit guard. Anything else
        # (other GPU IDs, lane decomposition, strides, the rest of the buffer
        # resources, LDS setup, etc.) lives inside the `if` so we don't pay
        # for it on workgroups whose partition is past seq_len.
        seq_idx = gpu.block_id("x")
        part_idx = gpu.block_id("z")

        sl_rsrc = buffer_ops.create_buffer_resource(arg_seq_lens, max_size=True)
        seq_len_i32 = buffer_ops.buffer_load(sl_rsrc, seq_idx, vec_width=1, dtype=T.i32)
        seq_len = arith.index_cast(T.index, seq_len_i32)

        # Early-exit guard: if this partition starts past seq_len, every token in
        # this workgroup's tile is out of range. The cmpi MUST be inlined into
        # the if test (see note in the kernel docstring) so FlyDSL's AST
        # rewriter converts this to scf.if rather than evaluating bool() at
        # compile time.
        part_first_tok = part_idx * arith.index(PARTITION_SIZE)
        if arith.cmpi(
            arith.CmpIPredicate.slt, _raw(part_first_tok), _raw(seq_len)
        ):
            elem_ty = T.bf16 if dtype == "bf16" else T.f16
            wmma_op = (
                rocdl.wmma_f32_16x16x32_bf16
                if dtype == "bf16"
                else rocdl.wmma_f32_16x16x32_f16
            )

            tx = gpu.thread_id("x")
            kv_head = gpu.block_id("y")

            warp_id = tx / arith.index(WAVE_SIZE)
            lane_id = tx % arith.index(WAVE_SIZE)
            lane_kgrp = lane_id / arith.index(WMMA_M)   # 0 or 1
            lane16 = lane_id % arith.index(WMMA_M)       # 0..15

            num_kv_heads = arith.index_cast(T.index, i32_num_kv_heads.ir_value())
            num_parts = arith.index_cast(T.index, i32_num_parts.ir_value())
            max_blocks = arith.index_cast(T.index, i32_max_blocks_per_seq.ir_value())

            # --- Runtime strides ---
            num_q_heads = num_kv_heads * arith.index(QUERY_GROUP_SIZE)

            stride_bt_seq = max_blocks

            stride_o_seq = num_kv_heads * num_parts * arith.index(QUERY_GROUP_SIZE * HEAD_SIZE)
            stride_o_head = num_parts * arith.index(QUERY_GROUP_SIZE * HEAD_SIZE)
            stride_o_part = arith.index(QUERY_GROUP_SIZE * HEAD_SIZE)
            stride_o_row = arith.index(HEAD_SIZE)

            stride_lse_seq = num_kv_heads * num_parts * arith.index(QUERY_GROUP_SIZE)
            stride_lse_head = num_parts * arith.index(QUERY_GROUP_SIZE)
            stride_lse_part = arith.index(QUERY_GROUP_SIZE)

            # --- Buffer resources (sl_rsrc already created above) ---
            bt_rsrc = buffer_ops.create_buffer_resource(arg_block_tables, max_size=True)
            out_rsrc = buffer_ops.create_buffer_resource(arg_out, max_size=True)
            ml_rsrc = buffer_ops.create_buffer_resource(arg_max_logits, max_size=True)
            es_rsrc = buffer_ops.create_buffer_resource(arg_exp_sums, max_size=True)
            # --- LDS setup ---
            base = allocator.get_base()
            q_lds = SmemPtr(base, q_lds_offset, elem_ty, shape=(q_lds_elems,))
            k_lds_stages = [
                SmemPtr(base, k_stage_offsets[i], elem_ty, shape=(kv_lds_elems,))
                for i in range(NUM_KV_STAGES)
            ]
            v_lds_stages = [
                SmemPtr(base, v_stage_offsets[i], elem_ty, shape=(kv_lds_elems,))
                for i in range(NUM_KV_STAGES)
            ]
            p_lds = SmemPtr(base, p_lds_offset, elem_ty, shape=(p_lds_elems,))
            # Cross-warp reduce scratch slabs (aliased with Q LDS region, f32)
            scratch_max_lds = SmemPtr(
                base, scratch_max_lds_offset, T.f32, shape=(scratch_slab_elems,)
            )
            scratch_sum_lds = SmemPtr(
                base, scratch_sum_lds_offset, T.f32, shape=(scratch_slab_elems,)
            )
            q_lds.get()
            k_lds_stages[0].get()
            v_lds_stages[0].get()
            if NUM_KV_STAGES == 2:
                k_lds_stages[1].get()
                v_lds_stages[1].get()
            p_lds.get()
            scratch_max_lds.get()
            scratch_sum_lds.get()

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

            # lane decomposition for ds_load_tr16_b128 (B-operand transposed read)
            lane8     = lane16 % arith.index(8)
            lane_ngrp = lane16 / arith.index(8)

            def _load_wmma_B_frag_tr(lds_ptr, n_col_base, k_base_elem, row_stride):
                """Load a vec<16xelem> WMMA B fragment via 2 × ds_load_tr16_b128.

                For tensors stored in LDS as [K-rows, N-cols] row-major (K = matmul-K
                = the contraction dim, N = matmul-N), where WMMA B-operand wants
                per-lane: 1 N-col × 16 K-rows. ds_load_tr16_b128 does the K↔N
                transpose in hardware while reading row-major LDS.

                Pre-transpose addressing per lane (lane_kgrp, lane16):
                  lane8 = lane16 % 8, lane_ngrp = lane16 / 8
                  LDS row = k_base_elem + lane_kgrp*8 + lane8     (K-row this lane reads)
                  LDS col = n_col_base + lane_ngrp * 8           (8-col chunk in N)
                One transposed load delivers vec<8xelem> = (8 K-rows × 1 N-col) per lane.
                Two calls (k_half = 0, 1) cover 16 K-rows per lane = full K=32 fragment.
                """
                lds_mem = lds_ptr.get()
                k_row_in_lane = lane_kgrp * arith.index(8) + lane8
                n_col_in_lane = n_col_base + lane_ngrp * arith.index(8)
                base = (
                    (arith.index(k_base_elem) + k_row_in_lane) * arith.index(row_stride)
                    + n_col_in_lane
                )
                chunks = []
                for k_half in range_constexpr(2):
                    k_row_extra = arith.index(k_half * 16) * arith.index(row_stride)
                    elem_off = base + k_row_extra
                    v = rocdl.lds_transpose_load(
                        T.vec(8, elem_ty), lds_mem, elem_off, elem_bytes
                    )
                    chunks.append(v)
                return vector.shuffle(chunks[0], chunks[1], list(range(16)))

            def _load_wmma_A_frag(lds_ptr, row_base_idx, k_base_elem, row_stride):
                """Load a vec<16xelem> WMMA A fragment via 2 × ds_read_b128.

                Per-lane layout: 16 elements at (row = row_base_idx,
                K-col = k_base_elem + (k0*2 + lane_kgrp)*8 + k1), k0 ∈ [0,1], k1 ∈ [0,7].

                For each k0 the 8 K-elements (k1 = 0..7) are K-contiguous in LDS, so
                one vector.load_op of vec<8xelem> covers them — lowers to a single
                ds_read_b128 (16 bytes). Two such loads + a shuffle give the full
                vec<16xelem> WMMA A-operand fragment, replacing 16 scalar ds_read_u16's.
                """
                lds_mem = lds_ptr.get()
                chunks = []
                for k0 in range_constexpr(2):
                    kk_base = (
                        arith.index(k_base_elem)
                        + (arith.index(k0 * 2) + lane_kgrp) * arith.index(8)
                    )
                    elem_off = row_base_idx * arith.index(row_stride) + kk_base
                    chunk = vector.load_op(T.vec(8, elem_ty), lds_mem, [elem_off])
                    chunks.append(chunk)
                return vector.shuffle(chunks[0], chunks[1], list(range(16)))

            # ---------------- TDM async load helpers ----------------
            # Total logical blocks in a partition (used to index block_tables).
            BLOCKS_PER_PARTITION = COMPUTES_PER_PARTITION * BLOCKS_PER_COMPUTE

            # Number of physical blocks this seq actually uses. We use this to
            # mask any block-table read whose logical index lies past the
            # seq's live block count to phys_blk = 0 (a valid physical block —
            # block 0 always exists in the cache). This is gluon's defensive
            # block-level mask: even though `bt_rsrc.max_size=True` and the
            # vLLM zero-init contract already keep us safe, this guarantees
            # the TDM never sees an undefined `phys_blk` in adversarial cases
            # (uninitialized block_tables, custom allocators where 0 isn't
            # block 0, etc.).
            KVB_idx = arith.index(KV_BLOCK_SIZE)
            live_blocks = (seq_len + KVB_idx - arith.index(1)) / KVB_idx
            zero_i32 = arith.constant(0, type=T.i32)

            def _phys_blks_for_compute(compute_iter_idx):
                """Load BLOCKS_PER_COMPUTE physical block IDs via vectorized buffer_load(s).

                For compute iteration `compute_iter_idx`, we need physical block indices at
                logical positions [base, base + BLOCKS_PER_COMPUTE) where
                base = part_idx * BLOCKS_PER_PARTITION + compute_iter_idx * BLOCKS_PER_COMPUTE.

                AMD's buffer_load vec_width is capped at 4 (128 bits for i32). For
                BPC > 4 we issue BT_NUM_LOADS loads of width BT_VEC_WIDTH (with the
                last load possibly narrower = BT_TAIL_WIDTH).

                Loaded IDs whose logical index is past `live_blocks` are masked to 0.

                Returns: list of `T.index` values, one per block in the tile.
                """
                base_logical = (
                    part_idx * arith.index(BLOCKS_PER_PARTITION)
                    + compute_iter_idx * arith.index(BLOCKS_PER_COMPUTE)
                )
                bt_base = seq_idx * stride_bt_seq + base_logical

                out = []
                for ldi in range_constexpr(BT_NUM_LOADS):
                    # Last load uses BT_TAIL_WIDTH (may be < BT_VEC_WIDTH); all others use BT_VEC_WIDTH.
                    this_width = BT_TAIL_WIDTH if ldi == BT_NUM_LOADS - 1 else BT_VEC_WIDTH
                    bt_off = bt_base + arith.index(ldi * BT_VEC_WIDTH)
                    if this_width == 1:
                        phys_i32 = buffer_ops.buffer_load(
                            bt_rsrc, bt_off, vec_width=1, dtype=T.i32
                        )
                        logical_idx = base_logical + arith.index(ldi * BT_VEC_WIDTH)
                        in_range = arith.cmpi(
                            arith.CmpIPredicate.slt, _raw(logical_idx), _raw(live_blocks)
                        )
                        phys_i32 = arith.select(in_range, _raw(phys_i32), _raw(zero_i32))
                        out.append(arith.index_cast(T.index, phys_i32))
                    else:
                        phys_vec = buffer_ops.buffer_load(
                            bt_rsrc, bt_off, vec_width=this_width, dtype=T.i32
                        )
                        for b in range_constexpr(this_width):
                            elem = vector.extract(
                                phys_vec, static_position=[b], dynamic_position=[]
                            )
                            logical_idx = base_logical + arith.index(ldi * BT_VEC_WIDTH + b)
                            in_range = arith.cmpi(
                                arith.CmpIPredicate.slt, _raw(logical_idx), _raw(live_blocks)
                            )
                            elem = arith.select(in_range, _raw(elem), _raw(zero_i32))
                            out.append(arith.index_cast(T.index, elem))
                return out

            def _issue_kv_load_single_block(kv_tensor, lds_memref, phys_blk, lds_byte_off):
                """Issue one TDM async 2D load of [KVB, HEAD] into LDS at byte offset
                `lds_byte_off` relative to the base of `lds_memref`."""
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
                    lds_byte_offset=lds_byte_off,
                )
                tdm_ops.tensor_load_2d(desc)

            def _issue_kv_tile_loads(k_lds_mem, v_lds_mem, phys_blks_list):
                """Issue TDM loads for one compute tile: BLOCKS_PER_COMPUTE K loads
                followed by BLOCKS_PER_COMPUTE V loads (K-then-V issue order).

                The split issue order matters: tensor_wait drains in FIFO, so
                issuing all K's before all V's lets us drain just the K's later
                (via tensor_wait(BPC + outstanding-after)) while V is still in
                flight, overlapping V's DMA with QK + softmax compute.

                Each block lands in a sub-region of the LDS stage (offset by block-index rows).
                """
                one_block_bytes = KV_BLOCK_SIZE * HEAD_SIZE * elem_bytes
                for b in range_constexpr(BLOCKS_PER_COMPUTE):
                    lds_sub_off = arith.index(b * one_block_bytes)
                    _issue_kv_load_single_block(
                        arg_key_cache, k_lds_mem, phys_blks_list[b], lds_sub_off
                    )
                for b in range_constexpr(BLOCKS_PER_COMPUTE):
                    lds_sub_off = arith.index(b * one_block_bytes)
                    _issue_kv_load_single_block(
                        arg_value_cache, v_lds_mem, phys_blks_list[b], lds_sub_off
                    )

            # Fence counts: each compute tile issues 2*BLOCKS_PER_COMPUTE TDM ops (K+V for each block).
            FENCE_OUTSTANDING = 2 * BLOCKS_PER_COMPUTE

            def _issue_q_load():
                """TDM load Q[seq, kv_head*QG : kv_head*QG + QG, :] into q_lds.

                Query is flat 1D view of [num_seqs, num_q_heads, HEAD_SIZE]. As a 2D
                view of HEAD-strided rows, the starting row for this (seq, kv_head)
                is seq_idx*num_q_heads + kv_head*QG.
                """
                q_outer_off = (
                    seq_idx * num_q_heads
                    + kv_head * arith.index(QUERY_GROUP_SIZE)
                )
                desc = tdm_ops.make_tensor_descriptor_2d(
                    global_ptr=arg_query,
                    lds_memref=q_lds.get(),
                    global_offset=(q_outer_off, arith.index(0)),
                    tensor_shape=(QUERY_GROUP_SIZE, HEAD_SIZE),
                    strides=(HEAD_SIZE, 1),
                    tile_shape=(QUERY_GROUP_SIZE, HEAD_SIZE),
                    elem_bytes=elem_bytes,
                    num_warps=NUM_WARPS,
                )
                tdm_ops.tensor_load_2d(desc)

            # ---------------- Prologue: Q + compute tile 0 loads (concurrent) ----------------
            _issue_q_load()
            phys_blks_tile0 = _phys_blks_for_compute(arith.index(0)) # load physical blocks for compute iter 1
            _issue_kv_tile_loads(k_lds_stages[0].get(), v_lds_stages[0].get(), phys_blks_tile0)
            # Drain ONLY Q so q_frags can be read; tile 0 stays in flight and overlaps
            # with q_frags reads. Outstanding here = 1 (Q) + 2*BPC (tile 0) = 1 + FENCE_OUTSTANDING.
            # tensor_wait(FENCE_OUTSTANDING) drains down to FENCE_OUTSTANDING, which
            # drains the oldest op (Q) and leaves tile 0 still in flight. The next
            # tensor_wait inside the loop will drain tile 0 before compute.
            tdm_ops.tensor_wait(FENCE_OUTSTANDING)
            gpu.barrier()

            # Pre-load Q fragments from LDS (held in registers for the entire kernel).
            q_frags = []
            for ks in range_constexpr(K_QK_TILES):  # each warp has a complete copy of Q
                q_frags.append(_load_wmma_A_frag(q_lds, lane16, ks * WMMA_K, HEAD_SIZE))

            # ---------------- Main loop over compute tiles in partition ----------------
            for compute_iter in range_constexpr(COMPUTES_PER_PARTITION):
                cur_stage = compute_iter % NUM_KV_STAGES
                nxt_stage = (compute_iter + 1) % NUM_KV_STAGES
                tile_first_tok = (
                    part_idx * arith.index(PARTITION_SIZE)
                    + arith.index(compute_iter * KV_COMPUTE_BLOCK_SIZE)
                )

                # Prefetch next compute tile's BLOCKS_PER_COMPUTE K+V loads if it exists.
                # _issue_kv_tile_loads issues all K's first, then all V's.
                # FIFO drain semantics: wait_for_K drains everything down to (V's of cur tile + any prefetched).
                if compute_iter < COMPUTES_PER_PARTITION - 1:
                    next_iter_idx = arith.index(compute_iter + 1)
                    next_phys_blks = _phys_blks_for_compute(next_iter_idx)
                    _issue_kv_tile_loads(
                        k_lds_stages[nxt_stage].get(),
                        v_lds_stages[nxt_stage].get(),
                        next_phys_blks,
                    )
                    # Outstanding before wait_for_K: cur K + cur V + next K + next V = 4*BPC.
                    # After wait_for_K: cur V + next K + next V = 3*BPC.
                    outstanding_after_k_drain = 3 * BLOCKS_PER_COMPUTE
                    # After wait_for_V: next K + next V = 2*BPC.
                    outstanding_after_v_drain = 2 * BLOCKS_PER_COMPUTE
                else:
                    # Last iter: no prefetch. Outstanding = cur K + cur V = 2*BPC.
                    # After wait_for_K: cur V = BPC. After wait_for_V: 0.
                    outstanding_after_k_drain = BLOCKS_PER_COMPUTE
                    outstanding_after_v_drain = 0

                # Wait for current tile's K (V continues loading in background).
                tdm_ops.tensor_wait(outstanding_after_k_drain)
                gpu.barrier()

                # Current stage pointers
                k_lds = k_lds_stages[cur_stage]
                v_lds = v_lds_stages[cur_stage]

                # ------------------------ QK WMMA (warp-split by N-tile) ------------------------
                # Each warp owns N_QK_TILES_PER_WARP N-tiles in a contiguous block:
                #   warp w handles global N-tiles [w * N_QK_TILES_PER_WARP, (w+1) * N_QK_TILES_PER_WARP).
                # qk_accs[n_local] is this warp's N_QK_TILES_PER_WARP[n_local] slice of S.
                qk_accs = [
                    arith.constant_vector(0.0, T.vec(8, T.f32))
                    for _ in range(N_QK_TILES_PER_WARP)
                ]
                for n_local in range_constexpr(N_QK_TILES_PER_WARP):
                    n_tile_global = (
                        warp_id * arith.index(N_QK_TILES_PER_WARP) + arith.index(n_local)
                    )
                    n_row_lds = n_tile_global * arith.index(WMMA_N) + lane16
                    for ks in range_constexpr(K_QK_TILES):
                        k_frag = _load_wmma_A_frag(
                            k_lds, n_row_lds, ks * WMMA_K, HEAD_SIZE
                        )
                        qk_accs[n_local] = wmma_op(
                            T.vec(8, T.f32),
                            q_frags[ks],
                            k_frag,
                            qk_accs[n_local],
                            signA=False,
                            signB=False,
                            modC=0,
                            reuseA=False,
                            reuseB=False,
                        ).result

                # Scale by qk_scale * log2e so we can use exp2 later.
                for n_local in range_constexpr(N_QK_TILES_PER_WARP):
                    qk_accs[n_local] = arith.mulf(qk_accs[n_local], qk_scale_log2_vec)

                # Mask out-of-range tokens. For lane's col within the warp's slice:
                #   global_n_tile = warp_id * N_QK_TILES_PER_WARP + n_local
                #   tok_abs = tile_first_tok + global_n_tile * 16 + lane16
                neg_inf_vec8 = arith.constant_vector(float("-inf"), T.vec(8, T.f32))
                for n_local in range_constexpr(N_QK_TILES_PER_WARP):
                    n_tile_global = (
                        warp_id * arith.index(N_QK_TILES_PER_WARP) + arith.index(n_local)
                    )
                    tok_abs = tile_first_tok + n_tile_global * arith.index(WMMA_N) + lane16
                    in_range = arith.cmpi(
                        arith.CmpIPredicate.slt, _raw(tok_abs), _raw(seq_len)
                    )
                    qk_accs[n_local] = arith.select(
                        in_range, _raw(qk_accs[n_local]), _raw(neg_inf_vec8)
                    )

                # ------------------------ Online softmax (cross-warp) ------------------------
                # Step 1: per-lane local max across this warp's N_QK_TILES_PER_WARP N-tiles.
                row_local_max = []
                for mi in range_constexpr(8):
                    cur = vector.extract(
                        qk_accs[0], static_position=[mi], dynamic_position=[]
                    )
                    if N_QK_TILES_PER_WARP > 1:
                        for n_local in range_constexpr(1, N_QK_TILES_PER_WARP):
                            v = vector.extract(
                                qk_accs[n_local], static_position=[mi], dynamic_position=[]
                            )
                            cur = arith.maximumf(cur, v)
                    row_local_max.append(cur)

                # Step 2: reduce within-warp across lane16 (butterfly 1,2,4,8) → warp's partial max.
                warp_max = []
                for mi in range_constexpr(8):
                    x = fxFloat32(row_local_max[mi])
                    for sh in [1, 2, 4, 8]:
                        peer = x.shuffle_xor(fxInt32(sh), fxInt32(WAVE_SIZE))
                        x = x.maximumf(peer)
                    warp_max.append(x.ir_value())

                # Step 3: cross-warp reduce via LDS scratch. Each warp writes its 16
                # partials (one per row of QG), barrier, then all warps read NUM_WARPS
                # partials and scalar-reduce.
                # Scratch layout: scratch_max_lds[w * QG + row] = warp_w's partial for that row.
                # Each lane within a warp has the same `warp_max[mi]` (butterfly made it so),
                # so any lane can do the write. To keep it tidy, only lane0 (lane_id==0) writes.
                # But benign-race stores are cheap; letting all 32 lanes write the same value
                # is fine on AMD LDS and avoids a scf.if. We go with that.
                for mi in range_constexpr(8):
                    row = lane_kgrp * arith.index(8) + arith.index(mi)
                    off = warp_id * arith.index(QUERY_GROUP_SIZE) + row
                    scratch_max_lds.store(warp_max[mi], [off])
                gpu.barrier()

                # Read all NUM_WARPS partials for each of this lane's 8 rows, combine.
                row_max_reduced = []
                for mi in range_constexpr(8):
                    row = lane_kgrp * arith.index(8) + arith.index(mi)
                    acc = fxFloat32(
                        scratch_max_lds.load([arith.index(0) + row])
                    )
                    for w in range_constexpr(1, NUM_WARPS):
                        partial = fxFloat32(
                            scratch_max_lds.load([arith.index(w * QUERY_GROUP_SIZE) + row])
                        )
                        acc = acc.maximumf(partial)
                    row_max_reduced.append(acc.ir_value())

                # Step 4: update m_state, compute alpha for PV rescale.
                alphas = []
                new_m_list = []
                for mi in range_constexpr(8):
                    new_m_mi = arith.maximumf(m_state[mi], row_max_reduced[mi])
                    alpha_mi = rocdl.exp2(T.f32, arith.subf(m_state[mi], new_m_mi))
                    alphas.append(alpha_mi)
                    new_m_list.append(new_m_mi)

                # Step 5: compute p = exp2(qk - new_m) for this warp's N-tiles,
                # accumulate per-lane row_sum over those tiles.
                row_sum_partial = [zero_f32] * 8
                p_accs = []
                for n_local in range_constexpr(N_QK_TILES_PER_WARP):
                    vals_new = []
                    for mi in range_constexpr(8):
                        q_ij = vector.extract(
                            qk_accs[n_local], static_position=[mi], dynamic_position=[]
                        )
                        p_ij = rocdl.exp2(T.f32, arith.subf(q_ij, new_m_list[mi]))
                        vals_new.append(p_ij)
                        row_sum_partial[mi] = arith.addf(row_sum_partial[mi], p_ij)
                    p_accs.append(
                        vector.from_elements(T.vec(8, T.f32), vals_new)
                    )

                # Step 6: butterfly reduce row_sum_partial within warp → warp's partial sum.
                warp_sum = []
                for mi in range_constexpr(8):
                    x = fxFloat32(row_sum_partial[mi])
                    for sh in [1, 2, 4, 8]:
                        peer = x.shuffle_xor(fxInt32(sh), fxInt32(WAVE_SIZE))
                        x = x + peer
                    warp_sum.append(x.ir_value())

                # Step 7: cross-warp sum reduce via LDS.
                for mi in range_constexpr(8):
                    row = lane_kgrp * arith.index(8) + arith.index(mi)
                    off = warp_id * arith.index(QUERY_GROUP_SIZE) + row
                    scratch_sum_lds.store(warp_sum[mi], [off])
                gpu.barrier()

                row_sum_reduced = []
                for mi in range_constexpr(8):
                    row = lane_kgrp * arith.index(8) + arith.index(mi)
                    acc = fxFloat32(
                        scratch_sum_lds.load([arith.index(0) + row])
                    )
                    for w in range_constexpr(1, NUM_WARPS):
                        partial = fxFloat32(
                            scratch_sum_lds.load([arith.index(w * QUERY_GROUP_SIZE) + row])
                        )
                        acc = acc + partial
                    row_sum_reduced.append(acc.ir_value())

                # Step 8: update l_state.
                for mi in range_constexpr(8):
                    l_state[mi] = arith.addf(
                        arith.mulf(alphas[mi], l_state[mi]), row_sum_reduced[mi]
                    )
                    m_state[mi] = new_m_list[mi]

                # Step 9: Rescale PV accumulator by alpha (per-row).
                for pv_n in range_constexpr(N_PV_TILES_PER_WARP):
                    new_vals = []
                    for mi in range_constexpr(8):
                        v = vector.extract(
                            pv_accs[pv_n], static_position=[mi], dynamic_position=[]
                        )
                        v = arith.mulf(v, alphas[mi])
                        new_vals.append(v)
                    pv_accs[pv_n] = vector.from_elements(T.vec(8, T.f32), new_vals)

                # Step 10: Write this warp's P slice to LDS.
                # Each warp owns N-tiles [warp_id*NpW, (warp_id+1)*NpW).
                # LDS col = n_tile_global * 16 + lane16 within [QG, KV_COMPUTE_BLOCK_SIZE].
                for n_local in range_constexpr(N_QK_TILES_PER_WARP):
                    p_vec = p_accs[n_local]
                    p_half = arith.trunc_f(T.vec(8, elem_ty), p_vec)
                    n_tile_global = (
                        warp_id * arith.index(N_QK_TILES_PER_WARP) + arith.index(n_local)
                    )
                    for mi in range_constexpr(8):
                        row = lane_kgrp * arith.index(8) + arith.index(mi)
                        col = n_tile_global * arith.index(WMMA_N) + lane16
                        off = row * arith.index(KV_COMPUTE_BLOCK_SIZE) + col
                        v = vector.extract(
                            p_half, static_position=[mi], dynamic_position=[]
                        )
                        p_lds.store(v, [off])

                # Wait for current tile's V (its DMA was overlapped with QK + softmax + P-LDS write).
                # The barrier doubles as the LDS-fence for the P-LDS writes above.
                tdm_ops.tensor_wait(outstanding_after_v_drain)
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
                            p_lds, lane16, ks * WMMA_K, KV_COMPUTE_BLOCK_SIZE
                        )
                        # V fragment from KV LDS: B operand [K=32, N=16].
                        # V LDS layout is [KVB, HEAD] row-major; matmul-K = KVB rows,
                        # matmul-N = HEAD cols. ds_load_tr16_b128 (via the helper)
                        # handles the K↔N transpose in hardware.
                        v_frag = _load_wmma_B_frag_tr(
                            v_lds,
                            global_n_tile * arith.index(WMMA_N),  # n_col_base in HEAD dim
                            ks * WMMA_K,                          # k_base_elem in KVB dim
                            HEAD_SIZE,                            # row stride (LDS layout [KVB, HEAD])
                        )

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

    cache_tag = (HEAD_SIZE, KV_BLOCK_SIZE, QUERY_GROUP_SIZE, PARTITION_SIZE,
                 KV_COMPUTE_BLOCK_SIZE, dtype, NUM_WARPS)

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
