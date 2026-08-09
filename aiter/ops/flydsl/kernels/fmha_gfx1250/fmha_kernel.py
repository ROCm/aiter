"""
Forward Kernel -- gfx1250, Unified FMHA Implementation.

Top-level kernel file: prologue, core loop interleaving engines, epilogue,
and public API (compile_fmha_fwd, flash_attn_varlen_d192_gfx1250).

Utilities (constants, schedule tables, namespace classes) live in fmha_utils.py.

Target: gfx1250 (MI450), wave32, 4 waves per TG (1TG), 1024 shared VGPRs.
Causal mask always on. num_tiles = bx + 1 (triangular).
"""

from __future__ import annotations

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm as llvm_dialect
from flydsl._mlir.dialects import scf
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import arith, rocdl
from aiter.ops.flydsl.kernels import buffer_ops, vector
from flydsl.expr.primitive import const_expr, range_constexpr
from flydsl.expr.rocdl import tdm_ops
from flydsl.expr.typing import T, Vector as Vec
from flydsl.utils.smem_allocator import SmemAllocator

from ..layout_utils import idx2crd as idx2crd
from ..tensor_shim import _run_compiled

from .fmha_utils import *  # constants, classes, prologue helpers
from .fmha_utils import _ep_finish  # underscore name, not covered by star import



def compile_fmha_fwd(*, is_causal: bool = False, return_lse: bool = False):
    """Compile FMHA kernel variant. Cached per (is_causal, return_lse)."""
    IS_CAUSAL = int(is_causal)
    RETURN_LSE = int(return_lse)

    @flyc.kernel(known_block_size=[BLOCK_SIZE, 1, 1])
    def fmha_fwd_kernel(
        ptr_O: fx.Pointer,
        ptr_Q: fx.Pointer,
        ptr_K: fx.Pointer,
        ptr_V: fx.Pointer,
        ptr_LSE: fx.Pointer,
        ptr_cu_seqlens_q: fx.Pointer,
        ptr_cu_seqlens_k: fx.Pointer,
        scalar_f: fx.Float32,
        stride_q_seq: fx.Int32,
        stride_k_seq: fx.Int32,
        stride_v_seq: fx.Int32,
        stride_o_seq: fx.Int32,
        stride_q_head: fx.Int32,
        stride_k_head: fx.Int32,
        stride_v_head: fx.Int32,
        stride_o_head: fx.Int32,
        gqa: fx.Int32,
        max_seqlen_q: fx.Int32,
        max_seqlen_k: fx.Int32,
    ):
        """D128 BF16 FMHA Forward — full kernel with dynamic KV loop.

        Structure (search for SECTION N to jump):
          SECTION 1  Prologue — HW setup, Q load, LDS address gen
          SECTION 2  Prologue — Tile 0 QK + partial softmax (no PV)
          SECTION 3  Dynamic KV loop (scf.for_)
            3a       Main loop — non-causal tiles
            3b       Main loop — causal tiles
            fmha_pipeline()  GEMM1(QK) + softmax + GEMM2(PV) pipeline
          SECTION 4  Epilogue — endtile + rescale + bf16 convert + store
            _ep_finish()  Final tile post-processing

        Loop-carried state (60 SSA values through scf.for_):
          [0..15]  o_tiles[4][4] v8f32      output accumulators
          [16..19] old_max[4] f32           previous row max
          [20..23] row_sums[4] f32          row sum for softmax
          [24..31] kv_tiles[4][2] v16bf16   K fragments for next WMMA
          [32..39] local_max[4], delta[4]   softmax state
          [40..55] sp_tiles[16] v8f32       QK score tiles per SU
          [56..59] ping-pong LDS bases      k_cur, v_cur, k_next, v_next
        """
        # Kernel params are Numeric (fx.Int32 / fx.Float32) — Python operators
        # work directly, no arith.unwrap needed.

        # Per-head byte strides are now runtime parameters (stride_q/k/v/o_head).
        # actual_q_len / actual_kv_len are derived later from cu_seqlens (THD).

        ty = get_types()

        # ================================================================
        # SECTION 1: Prologue — HW Setup + Q Load + Address Gen
        #
        # 1. Set wave scheduling mode
        # 2. Compute thread/block/grid IDs with XCD remap
        # 3. Load cu_seqlens → actual_q_len, actual_kv_len
        # 4. OOB protection: build per-warp TDM descriptors
        # 5. Guard: skip workgroups with no valid Q rows
        # 6. Zero-fill output for seqlen_k == 0 batches
        # ================================================================

        setreg(2074, 2)  # WAVE_SCHED_MODE = 2
        rocdl.s_nop(0)

        tx = fx.Int32(fx.thread_idx.x)
        lane_id = tx & 31
        wave_id = tx >> 5

        # Grid layout: [B, num_m, H] — batch on x, m-block on y, head on z.
        # Software XCD remap (HipKittens style):
        #   flat wgid = raw_x + gdx*(raw_y + gdy*raw_z)
        # This converts hardware round-robin XCD assignment to chunked assignment:
        #   XCD i gets wgids [i*(NUM_WGS/8) .. (i+1)*(NUM_WGS/8)-1]
        # → workgroups with nearby new_wgid share the same XCD → K/V cache locality.
        _NUM_XCDS = 8
        raw_bx = fx.Int32(fx.block_idx.x)  # raw batch
        raw_by = fx.Int32(fx.block_idx.y)  # raw m-block
        raw_bz = fx.Int32(fx.block_idx.z)  # raw head
        gdx = fx.Int32(fx.grid_dim.x)  # B
        gdy = fx.Int32(fx.grid_dim.y)  # M
        gdz = fx.Int32(fx.grid_dim.z)  # H
        # flat wgid
        wgid = raw_bx + gdx * raw_by + gdx * gdy * raw_bz
        # total workgroups
        num_wgs = gdx * gdy * gdz
        # Guard: only remap when num_wgs is a positive multiple of NUM_XCDS.
        # Otherwise (num_wgs/8 truncates), the formula maps distinct wgids to
        # the same new_wgid → workgroup collision and skipped tiles.
        wgs_per_xcd = num_wgs // _NUM_XCDS
        num_wgs_rem = num_wgs % _NUM_XCDS
        is_gt = num_wgs > _NUM_XCDS
        is_mul = num_wgs_rem == 0
        do_remap = is_gt & is_mul
        new_wgid_remapped = (wgid % _NUM_XCDS) * wgs_per_xcd + wgid // _NUM_XCDS
        new_wgid = do_remap.select(new_wgid_remapped, wgid)
        # decompose back to 3D: x = new%gdx, y = (new/gdx)%gdy, z = new/(gdx*gdy)
        new_bx = new_wgid % gdx
        new_tmp = new_wgid // gdx
        new_by = new_tmp % gdy
        new_bz = new_tmp // gdy
        bz = new_bx  # batch    (grid.x)
        bx = new_by  # m-block  (grid.y)
        by = new_bz  # head     (grid.z)

        m_start = bx * TILE_N

        # THD Step 2: load cu_seqlens -> q_start_tok,
        # k_start_tok (in tokens, broadcast as SGPR)
        i32_ty = T.i32
        i64_ty = T.i64
        gptr_ty = glb_ptr_ty()

        def load_cu_seqlen_scalar(ptr_tensor, idx_i32):
            """Load cu_seqlens[idx] → SGPR (uniform across wavefront)."""
            base_i64 = ptr_base_i64(ptr_tensor)
            byte_off_64 = fx.Int64(idx_i32 * 4)
            addr_ptr = llvm_dialect.inttoptr(gptr_ty, base_i64 + byte_off_64)
            return rocdl.readfirstlane(T.i32, llvm_dialect.load(i32_ty, addr_ptr))

        q_start_tok = load_cu_seqlen_scalar(ptr_cu_seqlens_q, bz)
        q_end_tok = load_cu_seqlen_scalar(ptr_cu_seqlens_q, bz + 1)
        k_start_tok = load_cu_seqlen_scalar(ptr_cu_seqlens_k, bz)
        k_end_tok = load_cu_seqlen_scalar(ptr_cu_seqlens_k, bz + 1)
        actual_q_len = q_end_tok - q_start_tok
        actual_kv_len = k_end_tok - k_start_tok

        # ================================================================
        # OOB protection: pre-compute per-block dg1 lists for TDM loads
        # ================================================================
        stride_k_elems_oob = stride_k_seq >> 1
        stride_v_elems_oob = stride_v_seq >> 1
        K_CFG_OOB = (1 << 16) | K_TDM_CONFIG
        V_CFG_OOB = (1 << 16) | V_TDM_CONFIG
        m_start_raw = m_start

        k_oob_dg1 = [
            TDM.make_kv_dg1_with_oob(
                K_CFG_OOB,
                QK_HDIM,
                8,
                stride_k_elems_oob,
                TDM.per_warp_oob_dim1(
                    actual_kv_len - su * 32,
                    wave_id,
                    8,
                ),
                dim0_stride=200,
            )
            for su in range(CNT_SU)
        ]
        v_oob_dg1 = [
            TDM.make_kv_dg1_with_oob(
                V_CFG_OOB,
                128,
                8,
                stride_v_elems_oob,
                TDM.per_warp_oob_dim1(
                    actual_kv_len - su * 32,
                    wave_id,
                    8,
                ),
            )
            for su in range(CNT_SU)
        ]
        # THD: clamp q_remain to ≥ 0 so excess workgroups (m_start >= actual_q_len)
        # write nothing.
        q_remain_raw = actual_q_len - m_start_raw
        q_remain_o = arith.maxsi(q_remain_raw, arith.constant(0, type=T.i32))
        o_oob_dim1 = TDM.per_warp_oob_dim1(q_remain_o, wave_id, 32)

        # THD: skip workgroups whose m_start >= actual_q_len (no valid tokens for this
        # tile).
        # Everything below (Q load / K/V load / core loop / O writeback) is gated by
        # wg_valid.
        wg_valid = m_start_raw < actual_q_len
        # ---- seqlen_k == 0: zero-fill output for valid Q rows, then skip compute ----
        kv_is_zero = actual_kv_len == 0
        need_zero = wg_valid & kv_is_zero
        if need_zero:
            i64z = T.i64
            glbpz = glb_ptr_ty()

            o_base_z = ptr_base_i64(ptr_O)
            tid_z = wave_id * WAVE_SIZE + lane_id
            zero_v4 = fx.constant_vector(0, T.vec(4, T.i32))
            q_rows = actual_q_len - m_start_raw
            q_rows_clamped = arith.maxsi(q_rows, arith.constant(0, type=T.i32))
            q_tok_z = q_start_tok + m_start_raw + tid_z
            valid_z = tid_z < q_rows_clamped
            if valid_z:
                elem_off_z = by * stride_o_head + q_tok_z * stride_o_seq
                byte_off_z = elem_off_z * 2
                byte_off_z64 = fx.Int64(byte_off_z)
                o_addr_z = o_base_z + byte_off_z64
                for chunk_z in fx.range_constexpr(V_HDIM // 8):
                    chunk_addr = o_addr_z + fx.Int64(chunk_z * 16)
                    ptr_z = llvm_dialect.inttoptr(glbpz, chunk_addr)
                    llvm_dialect.store(zero_v4, ptr_z)
                # LSE = -inf for seqlen_k==0 rows
                if const_expr(RETURN_LSE):
                    neg_inf_zl = arith.constant(float("-inf"), type=T.f32)
                    lse_base_z = ptr_base_i64(ptr_LSE)
                    # lse layout: (total_q, nheads), elem_off = tok * nheads + head
                    lse_elem_z = q_tok_z * gdz + by
                    lse_byte_z = lse_elem_z * 4
                    lse_byte_z64 = fx.Int64(lse_byte_z)
                    lse_addr_z = lse_base_z + lse_byte_z64
                    lse_ptr_z = llvm_dialect.inttoptr(glbpz, lse_addr_z)
                    llvm_dialect.store(neg_inf_zl, lse_ptr_z)

        # Gate the main compute path: need valid Q rows AND non-zero K/V length.
        kv_nonzero = actual_kv_len > 0
        wg_valid_compute = wg_valid & kv_nonzero
        if wg_valid_compute:
            # Q resource descriptor with OOB protection (THD: batch is implicit in
            # q_start_tok)
            q_tok = q_start_tok + bx * 128
            q_offset = q_tok * stride_q_seq + by * stride_q_head
            q_num_bytes = q_end_tok * stride_q_seq
            q_rsrc = buffer_ops.create_buffer_resource(
                ptr_Q, num_records_bytes=q_num_bytes
            )

            # Q load → q_frags[4 bank][2 frag] v16bf16
            # Pass q_offset (byte offset for this workgroup's Q tile) so WG>0 reads
            # correct rows
            q_frags_raw = phase4_q_load(
                lane_id,
                q_rsrc,
                stride_q_seq,
                wave_id,
                q_tile_offset_bytes=q_offset,
            )
            rocdl.sched_barrier(0)

            # Bridge prologue q_frags[4 bank][frags_per_bank] → core loop q_tiles[4
            # msb][Q_WMMA_PER_MSB]
            # q_msb only takes values {0, 2} (k//Q_WMMA_PER_MSB = 0 for all k <
            # total_k_iters).
            # Each q_msb merges TWO adjacent banks (lo+hi K-col halves):
            #   q_msb=0 (sp_msb∈{0,1}): banks[0] (K cols 0..QK_HDIM/2) + banks[1] (K
            #   cols QK_HDIM/2..QK_HDIM)
            #   q_msb=2 (sp_msb∈{2,3}): banks[2] + banks[3]
            # frags_per_bank = len(q_frags_raw[0]) = Q_WMMA_PER_MSB//2 (3 for 192-dim)
            frags_per_bank = len(q_frags_raw[0])  # e.g. 3 for 192-dim, 2 for 128-dim
            q_frags = [[None] * Q_WMMA_PER_MSB for _ in range(NUM_MSB)]
            pad = [None] * (Q_WMMA_PER_MSB - 2 * frags_per_bank)
            q_frags[0] = q_frags_raw[0] + q_frags_raw[1] + pad
            q_frags[2] = q_frags_raw[2] + q_frags_raw[3] + pad

            # Head index (for GQA)
            head_index = head_index_div(by, gqa)

            # K/V base offsets
            # THD: K/V batch offset = k_start_tok * stride_{k,v}_seq
            k_offset = k_start_tok * stride_k_seq + head_index * stride_k_head
            v_offset = k_start_tok * stride_v_seq + head_index * stride_v_head

            # SmemAllocator bases → i32 LDS addresses
            k_a_base_i32 = extract_lds_base_i32(lds_alloc_k_a.get_base())
            k_b_base_i32 = extract_lds_base_i32(lds_alloc_k_b.get_base())
            v_a_base_i32 = extract_lds_base_i32(lds_alloc_v_a.get_base())
            v_b_base_i32 = extract_lds_base_i32(lds_alloc_v_b.get_base())

            # K/V LDS address generation — from SmemAllocator bases
            # kv_lds_addrs_a[0..3]=K_a, [4..7]=V_a  (ping / blk=0)
            # kv_lds_addrs_b[0..3]=K_b, [4..7]=V_b  (pong / blk=1)
            rocdl.sched_barrier(0)
            kv_lds_addrs_a = build_kv_lds_addrs(lane_id, k_a_base_i32, v_a_base_i32)
            kv_lds_addrs_b = build_kv_lds_addrs(lane_id, k_b_base_i32, v_b_base_i32)

            stride_k_32 = stride_k_seq * 32
            stride_v_32 = stride_v_seq * 32

            # SGPR state: softmax scale
            log2e_val = arith.constant(1.4426950408889634, type=T.f32)
            scale = log2e_val * scalar_f
            scale_pair = fx.vector.broadcast(T.vec(2, T.f32), scale)

            sgpr_state = {
                "s_log2e_scl": scale,
                "s_log2e_scl_pair": scale_pair,
            }

            # Build kernel context for extracted pipeline functions
            ctx = {
                "ty": ty,
                "lane_id": lane_id,
                "wave_id": wave_id,
                "m_start": m_start,
                "bx": bx,
                "by": by,
                "gdz": gdz,
                "ptr_K": ptr_K,
                "ptr_V": ptr_V,
                "ptr_O": ptr_O,
                "ptr_LSE": ptr_LSE,
                "scalar_f": scalar_f,
                "stride_k_seq": stride_k_seq,
                "stride_v_seq": stride_v_seq,
                "stride_o_seq": stride_o_seq,
                "stride_o_head": stride_o_head,
                "stride_k_32": stride_k_32,
                "stride_v_32": stride_v_32,
                "k_offset": k_offset,
                "v_offset": v_offset,
                "actual_kv_len": actual_kv_len,
                "actual_q_len": actual_q_len,
                "q_start_tok": q_start_tok,
                "o_oob_dim1": o_oob_dim1,
                "q_frags": q_frags,
                "sgpr_state": sgpr_state,
                "RETURN_LSE": RETURN_LSE,
            }

            # ================================================================
            # SECTION 2: Prologue -- Tile 0 QK + Partial Softmax (no PV)
            #
            # 1. TDM load K(tile 0) → K_a LDS
            # 2. QK_pure for all 4 SUs (no interleaving, first tile)
            # 3. TDM load V(tile 0) → V_a LDS
            # 4. Apply causal + KV OOB masks
            # 5. sp_tiles → sp_pairs → Softmax PART0+PART1+PART2 (first half)
            # 6. Prefetch K(tile 1) → K_b
            # 7. Build initial iter_args for dynamic loop
            # ================================================================
            #
            #   1. TDM K(tile 0, 4 SUs) → wait → QK_pure (64 WMMAs)
            #   2. TDM V(tile 0, 4 SUs) — overlapped with softmax
            #   3. sp_tiles → sp_pairs → softmax PART0+PART1 only (no PART2)
            #   4. TDM K(tile 1) — prefetch K only, V(tile 0) stays in LDS
            #   5. Wait K(tile 1) → LDS K(su=0) — preload for core_loop entry
            #
            # No PV in prologue. PART2 + PV run in core_loop iterations.

            zero_f32 = arith.constant(0.0, type=T.f32)
            neg_inf = arith.constant(float("-inf"), type=T.f32)
            zero_v8f32 = fx.constant_vector(0.0, T.vec(8, T.f32))

            # -- 2a: Load K(tile 0) → K_a (ping buffer) --
            rocdl.sched_barrier(0)
            TDM.load_k_only(
                ptr_K,
                k_offset,
                stride_k_seq,
                stride_k_32,
                wave_id,
                k_a_base_i32,
                oob_dg1_list=k_oob_dg1,
            )
            rocdl.sched_barrier(0)

            # -- 2b: QK_pure for all 4 SUs --
            all_su_sp_tiles = []
            for su in fx.range_constexpr(CNT_SU):
                kv_tiles_su = Fragment.load_k_su(ty, kv_lds_addrs_a, 0, su)
                fresh_sp = []
                for msb in fx.range_constexpr(NUM_MSB):
                    fresh_sp.append([zero_v8f32])
                fresh_sp = qk_gemm_pure(ty, 0, su, q_frags, kv_tiles_su, fresh_sp)
                all_su_sp_tiles.append(fresh_sp)

            # -- 2c: Load V(tile 0) → V_a (ping buffer) --
            TDM.load_v_only(
                ptr_V,
                v_offset,
                stride_v_seq,
                stride_v_32,
                wave_id,
                v_a_base_i32,
                oob_dg1_list=v_oob_dg1,
            )

            # -- 2d': causal mask on prologue tile (n_start=0) --
            # Bottom-right aligned: shift n_start by -(sk - sq) so the diagonal
            # is anchored at the bottom-right corner of the QK matrix.
            causal_offset = actual_kv_len - actual_q_len
            if const_expr(IS_CAUSAL):
                pro_causal_n = -causal_offset
                apply_causal_mask(ctx, all_su_sp_tiles, pro_causal_n)

            # -- 2d'': KV OOB mask on prologue tile --
            apply_kv_oob_mask(ctx, all_su_sp_tiles, actual_kv_len)

            # -- 2d: sp_tiles → sp_pairs --
            sp_pairs_all_pro = Softmax.tiles_to_pairs(all_su_sp_tiles)

            # -- 2e: Softmax PART0+PART1 only (no PART2) --
            # Pin each MSB's scalar state to its own VGPR bank for the
            # "全-bankN" (0x00/0x55/0xAA/0xFF) MSB allocation pattern.
            softmax_state_pro = {
                "old_max": [set_vgpr_bank(neg_inf, m) for m in range(NUM_MSB)],
                "local_max": [set_vgpr_bank(neg_inf, m) for m in range(NUM_MSB)],
                "delta": [set_vgpr_bank(zero_f32, m) for m in range(NUM_MSB)],
                "exp_delta": [None] * NUM_MSB,
                "cur_max_log2e": [None] * NUM_MSB,
                "cur_max_log2e_1": [None] * NUM_MSB,
                "cur_max_log2e_scalar": [None] * NUM_MSB,
                "cur_max_log2e_dup": [None] * NUM_MSB,
                "vgpr_log2e_scl_pair": [None] * NUM_MSB,
                "exp_delta_dup": [None] * NUM_MSB,
                "row_sums": [set_vgpr_bank(zero_f32, m) for m in range(NUM_MSB)],
                "p_bf16": [[], [], [], []],
                "sp_pairs_prev": sp_pairs_all_pro,
            }
            Softmax.part01_only(ty, 0, sp_pairs_all_pro, softmax_state_pro, sgpr_state)

            # -- 2e': PART2 first half for tile 0 --
            # Runs ops 0..PART2_SPLIT-1: setup(7)+pkfma(16)+pair_exp(8) = 31 ops/MSB.
            # 4 MSBs × 8 pair_exp = 32 pair_exp + 4 exp_delta (unavoidable pipeline
            # overhead).
            # sp_pairs_all_pro[m][0..15] are partially modified (pkfma+8-exp applied).
            # pro_exp_delta seeds the O-rescale iter_arg for the first core_loop
            # iteration.
            pro_part2_ops = Softmax.build_all_part2_ops(
                ty, 0, sp_pairs_all_pro, softmax_state_pro, sgpr_state
            )
            for m in fx.range_constexpr(NUM_MSB):
                for op in pro_part2_ops[m][:PART2_SPLIT]:
                    op()

            # -- 2f: Prefetch K(tile 1) → K_b if available --
            tile_n_const = TILE_N
            ctx["tile_n_const"] = tile_n_const
            ctx["zero_v8f32"] = zero_v8f32
            kv_tiles_avail = (actual_kv_len + (TILE_N - 1)) // tile_n_const
            if const_expr(IS_CAUSAL):
                # Causal (bottom-right aligned): the last Q row (m_start + TILE_N - 1)
                # attends to the last K row (actual_kv_len - 1).  The first Q row in
                # this tile attends to K starting at position
                #   (actual_kv_len - actual_q_len) + m_start.
                # So the number of KV tiles needed is:
                sk_sq_diff = actual_kv_len - actual_q_len
                sk_sq_tiles = (sk_sq_diff + (TILE_N - 1)) // tile_n_const
                bx_plus_1 = bx + 1
                causal_tiles = bx_plus_1 + sk_sq_tiles
                num_tiles = arith.minui(causal_tiles.ir_value(), kv_tiles_avail)
            else:
                # Non-causal: iterate over all KV tiles.
                num_tiles = kv_tiles_avail
            num_tiles_idx = arith.index_cast(T.index, num_tiles)
            # Loop runs N-2 iterations (tiles 1..N-2); endtile handled in epilogue.
            num_tiles_minus1 = num_tiles - 1
            num_tiles_minus1_idx = arith.index_cast(T.index, num_tiles_minus1)

            # Load K(tile 1) → K_b for core_loop first iteration.
            # For num_tiles=1 (128x128), K_b is never used (loop runs 0 iterations).
            rocdl.sched_barrier(0)
            k_tile1_stride = tile_n_const * stride_k_seq
            k_tile1_offset = k_offset + k_tile1_stride
            kv_remain_t1 = actual_kv_len - TILE_N
            k_tile1_oob_dg1 = [
                TDM.make_kv_dg1_with_oob(
                    K_CFG_OOB,
                    QK_HDIM,
                    8,
                    stride_k_elems_oob,
                    TDM.per_warp_oob_dim1(
                        kv_remain_t1 - su * 32,
                        wave_id,
                        8,
                    ),
                    dim0_stride=200,
                )
                for su in range(CNT_SU)
            ]
            TDM.load_k_only(
                ptr_K,
                k_tile1_offset,
                stride_k_seq,
                stride_k_32,
                wave_id,
                k_b_base_i32,
                oob_dg1_list=k_tile1_oob_dg1,
            )
            rocdl.sched_barrier(0)

            # -- 2g: Load K(su=0) from K_b for core_loop entry --
            kv_tiles_init = load_initial_kv_tiles(ty, kv_lds_addrs_b, blk=0, su=0)

            # Prologue results: PART0+PART1+PART2 first half done.
            # old_max = local_max (set by PART2 setup op0); row_sums rescaled
            # (×exp_delta).
            pro_old_max = [
                softmax_state_pro["old_max"][m] for m in fx.range_constexpr(NUM_MSB)
            ]
            pro_row_sums = [
                softmax_state_pro["row_sums"][m] for m in fx.range_constexpr(NUM_MSB)
            ]
            pro_local_max = [
                softmax_state_pro["local_max"][m] for m in fx.range_constexpr(NUM_MSB)
            ]
            pro_delta = [
                softmax_state_pro["delta"][m] for m in fx.range_constexpr(NUM_MSB)
            ]

            # Partial sp_pairs after first half: yield as separate lo+hi f32 scalars.
            # Prologue runs pair_exp sequentially (correct v2f32), so extractelement is
            # safe here.
            pro_partial_sp_lo_flat = []
            pro_partial_sp_hi_flat = []
            for m in fx.range_constexpr(NUM_MSB):
                for i in fx.range_constexpr(N_SP_PAIRS):
                    pair = Vec(sp_pairs_all_pro[m][i], dtype=fx.Float32)
                    pro_partial_sp_lo_flat.append(pair[0].ir_value())
                    pro_partial_sp_hi_flat.append(pair[1].ir_value())
            pro_partial_sp_flat = pro_partial_sp_lo_flat + pro_partial_sp_hi_flat
            # exp_delta from PART2 setup (used by first core_loop iteration's O
            # rescale).
            pro_exp_delta = [
                softmax_state_pro["exp_delta"][m] for m in fx.range_constexpr(NUM_MSB)
            ]

            # Flatten kv_tiles_init[4 msb][2] → 8 v16bf16
            kv_flat_init = []
            for msb in fx.range_constexpr(NUM_MSB):
                for k in fx.range_constexpr(N_WMMA_K_TILES):
                    kv_flat_init.append(kv_tiles_init[msb][k])

            # Flatten prologue sp_tiles per SU [CNT_SU][NUM_MSB][1] → 16 v8f32
            sp_flat_init = []
            for su in fx.range_constexpr(CNT_SU):
                for msb in fx.range_constexpr(NUM_MSB):
                    sp_flat_init.append(all_su_sp_tiles[su][msb][0])

            # ================================================================
            # SECTION 3: Dynamic KV Loop — scf.for_ from tile 1
            #
            # Pipeline: K is one tile AHEAD of V in LDS.
            #   Iteration i: GEMM1 on K(i+1), GEMM2 on V(i)
            #   After fmha_pipeline: TDM V(i+1) + K(i+2)
            #
            # Two sub-loops for causal attention:
            #   3a: non-causal tiles [1, first_causal_tile) — no mask
            #   3b: causal tiles [first_causal_tile, num_tiles-1) — with mask
            # ================================================================
            #
            # Pipeline layout:
            #   - K in LDS is one tile AHEAD of V in LDS
            #   - Prologue loaded K(tile 0)+V(tile 0), did QK, prefetched K(tile 1)
            #   - Iteration i: GEMM1 on K(tile i+1), GEMM2 on V(tile i)
            #   - After core_loop: TDM V(tile i+1) + K(tile i+2)
            #
            # O tiles start as zeros (no PV in prologue).

            # ================================================================
            # Init iter_args layout (dynamic — offsets depend on N_WMMA_K_TILES):
            #   [0..15]            o_tiles[4][4] v8f32                  = 16 values
            #   [16..19]           old_max[4] f32                        = 4 values
            #   [20..23]           row_sums[4] f32                       = 4 values
            #   [24..24+KV-1]      kv_tiles[NUM_MSB][N_WMMA_K_TILES]    = KV values
            #   [24+KV..+3]        local_max[4] f32                      = 4 values
            #   [24+KV+4..+7]      delta[4] f32                          = 4 values
            #   [24+KV+8..+23]     sp_tiles[CNT_SU*NUM_MSB] v8f32       = 16 values
            #   [24+KV+24..+27]    ping-pong bases (i32)                 = 4 values
            #   [24+KV+28..+91]    partial_sp_pairs[4][16] v2f32        = 64 values
            #                      (PART2 first half output, double-buffered pipeline)
            #   [24+KV+92..+95]    exp_delta[4] f32                     = 4 values
            # ================================================================
            _KV_SIZE = NUM_MSB * N_WMMA_K_TILES  # 12 for 192-dim, 8 for 128-dim
            _OFF_LOCAL_MAX = 24 + _KV_SIZE  # 36 for 192-dim
            _OFF_DELTA = _OFF_LOCAL_MAX + NUM_MSB
            _OFF_SP = _OFF_DELTA + NUM_MSB
            _OFF_PP = _OFF_SP + CNT_SU * NUM_MSB
            _OFF_PSP = _OFF_PP + 4  # partial_sp lo: 64 f32
            _PSP_SIZE = NUM_MSB * N_SP_PAIRS  # = 64 (lo half)
            _OFF_PSP_HI = _OFF_PSP + _PSP_SIZE  # partial_sp hi: 64 f32
            _OFF_PED = _OFF_PSP_HI + _PSP_SIZE  # exp_delta: 4 f32
            o_flat_init = [zero_v8f32] * (NUM_MSB * N_PV_WMMA_N)

            # Ping-pong bases for iteration 1:
            #   K_cur = K_b (tile 1 K), V_cur = V_a (tile 0 V)
            #   K_next = K_a (TDM K target), V_next = V_b (TDM V target)
            pp_init = [k_b_base_i32, v_a_base_i32, k_a_base_i32, v_b_base_i32]

            init_args = (
                o_flat_init
                + pro_old_max
                + pro_row_sums
                + kv_flat_init
                + pro_local_max
                + pro_delta
                + sp_flat_init
                + pp_init
                + pro_partial_sp_flat
                + pro_exp_delta
            )

            # ---- Split point for causal loops (chunked-prefill: sq < sk) ----
            # Tiles below first_causal_tile are fully under the diagonal and
            # need no causal mask.  Tiles at or above it cross the diagonal.
            # When sq == sk, first_causal_tile == num_tiles so loop 1 covers
            # everything and loop 2 runs zero iterations — no regression.
            if const_expr(IS_CAUSAL):
                # Tile t is fully below the diagonal when
                #   (t+1)*TILE_N - 1 <= causal_offset + m_start
                # i.e. t < floor((causal_offset + m_start) / TILE_N) + 1
                # Since m_start = bx * TILE_N this equals bx + floor(causal_offset /
                # TILE_N).
                # Clamp to [1, num_tiles-1) since loop 1 starts at tile 1.
                first_causal_tile = bx + causal_offset // tile_n_const
                first_causal_tile = arith.maxsi(
                    first_causal_tile.ir_value(), arith.constant(1, type=T.i32)
                )
                first_causal_tile = arith.minui(first_causal_tile, num_tiles_minus1)
                first_causal_tile_idx = arith.index_cast(T.index, first_causal_tile)
            else:
                first_causal_tile_idx = num_tiles_minus1_idx

            # ---- tile_iteration: shared loop body for SECTION 3a / 3b ----
            # ================================================================
            # SECTION 3a: Main KV loop — non-causal tiles [1, first_causal_tile)
            # ================================================================
            for tile_idx, iter_args, loop1_results in scf.for_(
                arith.index(1),
                first_causal_tile_idx,
                arith.index(1),
                iter_args=init_args,
            ):
                yield tile_iteration(ctx, tile_idx, iter_args)

            # ================================================================
            # SECTION 3b: Main KV loop -- causal tiles [first_causal_tile, num_tiles-1)
            # ================================================================
            for tile_idx, iter_args, loop_results in scf.for_(
                first_causal_tile_idx,
                num_tiles_minus1_idx,
                arith.index(1),
                iter_args=loop1_results,
            ):
                # Causal mask: compute n_start for this tile
                tile_idx_i32 = arith.index_cast(T.i32, tile_idx)
                causal_n = tile_idx_i32 * tile_n_const - causal_offset
                yield tile_iteration(ctx, tile_idx, iter_args, causal_n_start=causal_n)

            # ================================================================
            # SECTION 4: Epilogue — post_process + div_cvt + write_out
            #
            # 1. Unpack final loop results (o_tiles, softmax state)
            # 2. Process endtile (last KV tile): core_loop + ep_finish
            # 3. Final O-rescale: O *= 1/row_sum
            # 4. Convert f32 → bf16 and store to LDS
            # 5. TDM store LDS → global output
            # 6. (Optional) Compute and store LSE
            # ================================================================
            #
            #   fmha_post_process(is_odd):
            #     softmax stages 4..7 (PART2) → complete last tile's softmax
            #     LDS V(su=0..3) → PV_pure (64 WMMAs)
            #   fmha_div_cvt():
            #     row_sums cross-MSB reduce → LSE = max*scale + log(sum)
            #     O = O * rcp(row_sum) → cvt_pk_bf16
            #   lds_store_D_LSE() + TDM_store_D_LSE()
            #
            # For tile_n=128: blk=0 always, no is_odd branching.

            # ---- 4a: Unpack loop results ----
            # Pin each MSB's values to bank=d/msb for full-bank MSB pattern.
            ep_o_tiles = []
            for d in fx.range_constexpr(NUM_MSB):
                row = []
                for n in fx.range_constexpr(N_PV_WMMA_N):
                    row.append(set_vgpr_bank(loop_results[d * N_PV_WMMA_N + n], d))
                ep_o_tiles.append(row)

            ep_old_max = [
                set_vgpr_bank(loop_results[16 + i], i)
                for i in fx.range_constexpr(NUM_MSB)
            ]
            ep_row_sums = [
                set_vgpr_bank(loop_results[20 + i], i)
                for i in fx.range_constexpr(NUM_MSB)
            ]
            ep_local_max = [
                set_vgpr_bank(loop_results[_OFF_LOCAL_MAX + i], i)
                for i in fx.range_constexpr(NUM_MSB)
            ]
            ep_delta = [
                set_vgpr_bank(loop_results[_OFF_DELTA + i], i)
                for i in fx.range_constexpr(NUM_MSB)
            ]

            # Epilogue ping-pong: V_cur after swap has last tile's V data
            ep_k_cur_base = loop_results[_OFF_PP]
            ep_v_cur_base = loop_results[_OFF_PP + 1]
            ep_kv_lds_addrs = build_kv_lds_addrs(
                lane_id,
                ep_k_cur_base,
                ep_v_cur_base,
            )

            # ---- 4b: Unpack partial_sp_pairs:
            # reconstruct v2f32 from separate lo+hi f32 ----
            ep_partial_sp_lo = [
                loop_results[_OFF_PSP + i] for i in fx.range_constexpr(_PSP_SIZE)
            ]
            ep_partial_sp_hi = [
                loop_results[_OFF_PSP_HI + i] for i in fx.range_constexpr(_PSP_SIZE)
            ]
            ep_partial_sp_pairs = []
            for m in fx.range_constexpr(NUM_MSB):
                ep_pairs = [
                    make_v2f32(
                        ep_partial_sp_lo[m * N_SP_PAIRS + i],
                        ep_partial_sp_hi[m * N_SP_PAIRS + i],
                        m,
                    )
                    for i in fx.range_constexpr(N_SP_PAIRS)
                ]
                ep_partial_sp_pairs.append(ep_pairs)

            # ---- 4b': Extra state for endtile (N>=2) ----
            # K(N-1) fragments from loop (GEMM2 stage 3 loaded from K_next).
            ep_kv_tiles_flat = [
                loop_results[24 + i] for i in fx.range_constexpr(_KV_SIZE)
            ]
            ep_kv_tiles = []
            for m in fx.range_constexpr(NUM_MSB):
                row = [
                    set_vgpr_bank(
                        ep_kv_tiles_flat[m * N_WMMA_K_TILES + k],
                        m,
                    )
                    for k in fx.range_constexpr(N_WMMA_K_TILES)
                ]
                ep_kv_tiles.append(row)

            # K_next / V_next LDS bases for endtile
            # core_loop kv_lds_addrs_next.
            ep_k_next_base = loop_results[_OFF_PP + 2]
            ep_v_next_base = loop_results[_OFF_PP + 3]
            ep_kv_lds_addrs_next = build_kv_lds_addrs(
                lane_id,
                ep_k_next_base,
                ep_v_next_base,
            )

            # V(N-1) global offset for endtile GEMM1 TDM.
            num_tiles_m1_ep = num_tiles - 1
            ep_v_endtile_offset = v_offset + num_tiles_m1_ep * tile_n_const * stride_v_seq

            # ia_exp_delta for endtile core_loop: exp_delta from last loop GEMM2.
            ia_exp_delta = [
                set_vgpr_bank(loop_results[_OFF_PED + m], m)
                for m in fx.range_constexpr(NUM_MSB)
            ]

            # ---- 4c: s_wait_idle + barrier ----
            emit_void("s_wait_idle")
            rocdl.s_barrier_signal(-1)
            rocdl.s_barrier_wait(-1)

            # ---- 4c': _ep_finish — PART2 + PV_pure + div_cvt + TDM store D ----
            # o_tiles:           [[v8f32]*N_PV_WMMA_N]*NUM_MSB — accumulated O
            # sp_pairs_in:       [[v2f32]*N_SP_PAIRS]*NUM_MSB  — PART2 first-half input
            # exp_delta_rescale: [f32]*NUM_MSB                 — exp_delta for O
            # rescale before PV
            # v_base_for_pv:     i32                           — V LDS base for PV_pure
            # old_max_in:        [f32]*NUM_MSB                 — max across all tiles
            # seen so far
            # local_max_in:      [f32]*NUM_MSB                 — local max (same as
            # old_max after PART0+1)
            # delta_in:          [f32]*NUM_MSB                 — delta (for PART2 setup)
            # row_sums_in:       [f32]*NUM_MSB                 — row_sums accumulated
            # to this point
            # ---- 4c'': endtile dispatch — if N>=2: core_loop + ep_finish, else:
            # ep_finish ----
            is_multi = num_tiles >= 2

            if is_multi:  # N>=2: endtile core_loop then ep_finish
                # All variables defined fresh inside THEN — not state variables.
                et_sp_t = [
                    [set_vgpr_bank(zero_v8f32, m)]
                    for m in fx.range_constexpr(NUM_MSB)
                ]
                et_sfx = {
                    "old_max": list(ep_old_max),
                    "local_max": list(ep_local_max),
                    "delta": list(ep_delta),
                    "exp_delta": [None] * NUM_MSB,
                    "cur_max_log2e": [None] * NUM_MSB,
                    "cur_max_log2e_1": [None] * NUM_MSB,
                    "cur_max_log2e_scalar": [None] * NUM_MSB,
                    "cur_max_log2e_dup": [None] * NUM_MSB,
                    "vgpr_log2e_scl_pair": [None] * NUM_MSB,
                    "exp_delta_dup": [None] * NUM_MSB,
                    "row_sums": list(ep_row_sums),
                    "p_bf16": [[], [], [], []],
                    "sp_pairs_prev": [
                        [
                            ep_partial_sp_pairs[m][i]
                            for i in fx.range_constexpr(N_SP_PAIRS)
                        ]
                        for m in fx.range_constexpr(NUM_MSB)
                    ],
                }
                # DBG: row_sums entering endtile core_loop (= ep_row_sums from
                # loop_results)

                et_tdm = {
                    "v_g0": fx.constant_vector(0, T.vec(4, T.i32)),
                    "v_g1": fx.constant_vector(0, T.vec(8, T.i32)),
                    "k_g0": fx.constant_vector(0, T.vec(4, T.i32)),
                    "k_g1": fx.constant_vector(0, T.vec(8, T.i32)),
                    "v_salu_queue": [],
                    "k_salu_queue": [],
                }
                et_o = [
                    [ep_o_tiles[d][n] for n in range(N_PV_WMMA_N)]
                    for d in range(NUM_MSB)
                ]
                if const_expr(IS_CAUSAL):
                    et_tile_n_start = (arith.index_cast(T.i32, num_tiles_idx) - 1) * TILE_N
                    et_causal_ns = et_tile_n_start - causal_offset
                else:
                    et_causal_ns = None
                et_kv_remain = actual_kv_len - (num_tiles - 1) * tile_n_const
                _et_V_CFG_OOB = (1 << 16) | V_TDM_CONFIG
                _et_stride_v_elems = stride_v_seq >> 1
                et_v_oob_dg1 = [
                    TDM.make_kv_dg1_with_oob(
                        _et_V_CFG_OOB,
                        128,
                        8,
                        _et_stride_v_elems,
                        TDM.per_warp_oob_dim1(
                            et_kv_remain - su * 32,
                            wave_id,
                            8,
                        ),
                    )
                    for su in range(CNT_SU)
                ]
                _, _, et_o, _, et_psp_lo, et_psp_hi, et_ped = fmha_pipeline_ctx(
                    ctx,
                    ty,
                    False,
                    q_frags,
                    ep_kv_tiles,
                    et_sp_t,
                    et_o,
                    ep_kv_lds_addrs,
                    et_tdm,
                    et_sfx,
                    sgpr_state,
                    gemm2=True,
                    tdm_v_offset=ep_v_endtile_offset,
                    tdm_v_target=ep_v_next_base,
                    tdm_k_offset=None,
                    kv_lds_addrs_next=ep_kv_lds_addrs_next,
                    gemm1_tdm_is_v=True,
                    ia_exp_delta=ia_exp_delta,
                    causal_n_start=et_causal_ns,
                    endtile_v_oob_dg1=et_v_oob_dg1,
                    kv_oob_cols=et_kv_remain,
                )
                # Pass updated softmax state (old_max/local_max/delta/row_sums after
                # PART0+1
                # for the endtile tile) so _ep_finish can correctly run PART2 second
                # half.
                # Reconstruct v2f32 sp_pairs for _ep_finish from safe f32 lo+hi scalars.
                et_psp = []
                for rpsm in fx.range_constexpr(NUM_MSB):
                    rpairs = [
                        make_v2f32(
                            et_psp_lo[rpsm * N_SP_PAIRS + rpi],
                            et_psp_hi[rpsm * N_SP_PAIRS + rpi],
                            rpsm,
                        )
                        for rpi in fx.range_constexpr(N_SP_PAIRS)
                    ]
                    et_psp.append(rpairs)
                rocdl.s_wait_tensorcnt(0)
                rocdl.s_barrier_signal(-1)
                rocdl.s_barrier_wait(-1)
                _ep_finish(
                    ctx,
                    et_o,
                    et_psp,
                    et_ped,
                    ep_v_next_base,
                    et_sfx["old_max"],
                    et_sfx["local_max"],
                    et_sfx["delta"],
                    et_sfx["row_sums"],
                    ep_k_cur_base,
                )
            else:  # N=1: original epilogue flow
                _ep_finish(
                    ctx,
                    [
                        [ep_o_tiles[d][n] for n in range(N_PV_WMMA_N)]
                        for d in range(NUM_MSB)
                    ],
                    ep_partial_sp_pairs,
                    [loop_results[_OFF_PED + m] for m in fx.range_constexpr(NUM_MSB)],
                    ep_v_cur_base,
                    list(ep_old_max),
                    list(ep_local_max),
                    list(ep_delta),
                    list(ep_row_sums),
                    ep_k_cur_base,
                )

    return fmha_fwd_kernel


# Launch wrapper + PyTorch entry point

HEAD_DIM_QK = 192
HEAD_DIM_V = 128
BLOCK_M = 128
KV_TILE_N = 128
BPP = 2  # bytes per element (bf16)

launch_fns = {}  # {(is_causal, return_lse): launch_fn}


def patch_reusable_slot_specs():
    import ctypes
    from flydsl.expr.numeric import Float32, Float64

    if not hasattr(Float32, "_reusable_slot_spec"):

        @classmethod
        def _f32_slot_spec(cls, arg):
            return ctypes.c_float, lambda a: a.value if hasattr(a, "value") else a

        Float32._reusable_slot_spec = _f32_slot_spec
        Float32._reusable_ctype = ctypes.c_float

    if not hasattr(Float64, "_reusable_slot_spec"):

        @classmethod
        def _f64_slot_spec(cls, arg):
            return ctypes.c_double, lambda a: a.value if hasattr(a, "value") else a

        Float64._reusable_slot_spec = _f64_slot_spec
        Float64._reusable_ctype = ctypes.c_double


def ensure_kernel(is_causal: bool, return_lse: bool = False):
    key = (is_causal, return_lse)
    if key in launch_fns:
        return

    patch_reusable_slot_specs()

    kernel = compile_fmha_fwd(is_causal=is_causal, return_lse=return_lse)

    @flyc.jit
    def _launch(
        ptr_O: fx.Pointer,
        ptr_Q: fx.Pointer,
        ptr_K: fx.Pointer,
        ptr_V: fx.Pointer,
        ptr_LSE: fx.Pointer,
        ptr_cu_seqlens_q: fx.Pointer,
        ptr_cu_seqlens_k: fx.Pointer,
        scalar_f: fx.Float32,
        stride_q_seq: fx.Int32,
        stride_k_seq: fx.Int32,
        stride_v_seq: fx.Int32,
        stride_o_seq: fx.Int32,
        stride_q_head: fx.Int32,
        stride_k_head: fx.Int32,
        stride_v_head: fx.Int32,
        stride_o_head: fx.Int32,
        gqa: fx.Int32,
        max_seqlen_q: fx.Int32,
        max_seqlen_k: fx.Int32,
        num_heads: fx.Int32,
        batch_size: fx.Int32,
        stream: fx.Stream,
    ):
        lds_alloc_k_a.finalized = False
        lds_alloc_k_b.finalized = False
        lds_alloc_v_a.finalized = False
        lds_alloc_v_b.finalized = False
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            lds_alloc_k_a.finalize()
            lds_alloc_k_b.finalize()
            lds_alloc_v_a.finalize()
            lds_alloc_v_b.finalize()

        num_tg = fx.Index((max_seqlen_q + (BLOCK_M - 1)) // BLOCK_M)
        grid_x = fx.Index(batch_size)
        grid_z = fx.Index(num_heads)

        launcher = kernel(
            ptr_O,
            ptr_Q,
            ptr_K,
            ptr_V,
            ptr_LSE,
            ptr_cu_seqlens_q,
            ptr_cu_seqlens_k,
            scalar_f,
            stride_q_seq,
            stride_k_seq,
            stride_v_seq,
            stride_o_seq,
            stride_q_head,
            stride_k_head,
            stride_v_head,
            stride_o_head,
            gqa,
            max_seqlen_q,
            max_seqlen_k,
        )
        launcher.launch(
            grid=(grid_x, num_tg, grid_z),
            block=(BLOCK_SIZE, 1, 1),
            stream=stream,
        )

    _launch.compile_hints["llvm_options"] = {"amdgpu-expert-scheduling-mode": True}
    launch_fns[key] = _launch


def flash_attn_varlen_d192_gfx1250(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    softmax_scale=None,
    causal=False,
    out=None,
    return_lse=False,
):
    assert q.dtype == torch.bfloat16, f"Expected bf16, got {q.dtype}"
    assert q.shape[-1] == HEAD_DIM_QK, (
        f"Expected headdim_qk={HEAD_DIM_QK}," f" got {q.shape[-1]}"
    )
    assert v.shape[-1] == HEAD_DIM_V, (
        f"Expected headdim_v={HEAD_DIM_V}," f" got {v.shape[-1]}"
    )

    total_q_tokens = q.shape[0]
    batch = cu_seqlens_q.shape[0] - 1
    nheads_q = q.shape[1]
    nheads_k = k.shape[1]
    gqa = nheads_q // nheads_k

    if softmax_scale is None:
        softmax_scale = 1.0 / (HEAD_DIM_QK**0.5)

    if out is None:
        out = torch.empty(
            (total_q_tokens, nheads_q, HEAD_DIM_V),
            dtype=torch.bfloat16,
            device=q.device,
        )
    if return_lse:
        lse = torch.empty(
            (total_q_tokens, nheads_q), dtype=torch.float32, device=q.device
        )
    else:
        lse = torch.empty(
            (batch, nheads_q, max_seqlen_q), dtype=torch.float32, device=q.device
        )

    stride_q_seq = q.stride(0) * BPP
    stride_k_seq = k.stride(0) * BPP
    stride_v_seq = v.stride(0) * BPP
    stride_o_seq = out.stride(0)
    stride_q_head = q.stride(1) * BPP
    stride_k_head = k.stride(1) * BPP
    stride_v_head = v.stride(1) * BPP
    stride_o_head = out.stride(1)

    ensure_kernel(bool(causal), bool(return_lse))

    _run_compiled(
        launch_fns[(bool(causal), bool(return_lse))],
        out,
        q,
        k,
        v,
        lse,
        cu_seqlens_q,
        cu_seqlens_k,
        softmax_scale,
        stride_q_seq,
        stride_k_seq,
        stride_v_seq,
        stride_o_seq,
        stride_q_head,
        stride_k_head,
        stride_v_head,
        stride_o_head,
        gqa,
        max_seqlen_q,
        max_seqlen_k,
        nheads_q,
        batch,
        torch.cuda.current_stream(),
    )

    if return_lse:
        return out, lse
    return out
