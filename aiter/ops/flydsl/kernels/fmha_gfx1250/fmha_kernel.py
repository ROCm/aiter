"""
Forward Kernel -- gfx1250, Unified FMHA Implementation.
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
from aiter.ops.flydsl.kernels import buffer_ops
from flydsl.expr.primitive import const_expr, range_constexpr
from flydsl.expr.typing import T, Vector as Vec

from ..tensor_shim import _run_compiled

from .fmha_utils import *  # constants, classes, prologue helpers
from .fmha_utils import _ep_finish  # underscore name, not covered by star import


def compile_fmha_fwd(*, is_causal: bool = False, return_lse: bool = False):
    """Compile FMHA kernel variant. Cached per (is_causal, return_lse)."""
    IS_CAUSAL = int(is_causal)
    RETURN_LSE = int(return_lse)

    @flyc.kernel(known_block_size=[BLOCK_SIZE, 1, 1])
    def fmha_fwd_kernel(
        ptr_O: fx.Tensor,
        ptr_Q: fx.Tensor,
        ptr_K: fx.Tensor,
        ptr_V: fx.Tensor,
        ptr_LSE: fx.Tensor,
        ptr_cu_seqlens_q: fx.Tensor,
        ptr_cu_seqlens_k: fx.Tensor,
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
        """D128 BF16 FMHA Forward -- full kernel with dynamic KV loop."""
        ty = get_types()
        setreg(2074, 2)  # WAVE_SCHED_MODE = 2
        rocdl.s_nop(0)
        tx = fx.Int32(fx.thread_idx.x)
        lane_id = tx & 31
        wave_id = tx >> 5

        # XCD remap: flat wgid → chunked assignment so nearby wgids share the
        # same XCD for K/V cache locality.
        _NUM_XCDS = 8
        raw_bx = fx.Int32(fx.block_idx.x)
        raw_by = fx.Int32(fx.block_idx.y)
        raw_bz = fx.Int32(fx.block_idx.z)
        gdx = fx.Int32(fx.grid_dim.x)
        gdy = fx.Int32(fx.grid_dim.y)
        gdz = fx.Int32(fx.grid_dim.z)
        wgid = raw_bx + gdx * raw_by + gdx * gdy * raw_bz
        num_wgs = gdx * gdy * gdz
        # Only remap when num_wgs is a positive multiple of NUM_XCDS to avoid
        # workgroup collision from truncated division.
        wgs_per_xcd = num_wgs // _NUM_XCDS
        num_wgs_rem = num_wgs % _NUM_XCDS
        is_gt = num_wgs > _NUM_XCDS
        is_mul = num_wgs_rem == 0
        do_remap = is_gt & is_mul
        new_wgid_remapped = (wgid % _NUM_XCDS) * wgs_per_xcd + wgid // _NUM_XCDS
        new_wgid = do_remap.select(new_wgid_remapped, wgid)
        new_bx = new_wgid % gdx
        new_tmp = new_wgid // gdx
        new_by = new_tmp % gdy
        new_bz = new_tmp // gdy
        bz = new_bx  # batch    (grid.x)
        bx = new_by  # m-block  (grid.y)
        by = new_bz  # head     (grid.z)
        m_start = bx * TILE_N
        i32_ty = T.i32
        i64_ty = T.i64
        gptr_ty = glb_ptr_ty()

        def load_cu_seqlen_scalar(ptr_tensor, idx_i32):
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

        stride_k_elems_oob = stride_k_seq >> 1
        stride_v_elems_oob = stride_v_seq >> 1
        K_CFG_OOB = (1 << 16) | K_TDM_CONFIG
        V_CFG_OOB = (1 << 16) | V_TDM_CONFIG
        m_start_raw = m_start
        k_oob_dg1 = TDM.build_oob_dg1_list(
            K_CFG_OOB,
            QK_HDIM,
            stride_k_elems_oob,
            actual_kv_len,
            wave_id,
            dim0_stride=200,
        )
        v_oob_dg1 = TDM.build_oob_dg1_list(
            V_CFG_OOB,
            128,
            stride_v_elems_oob,
            actual_kv_len,
            wave_id,
        )
        q_remain_raw = actual_q_len - m_start_raw
        q_remain_o = arith.maxsi(q_remain_raw, arith.constant(0, type=T.i32))
        o_oob_dim1 = TDM.per_warp_oob_dim1(q_remain_o, wave_id, 32)
        wg_valid = m_start_raw < actual_q_len
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
                if const_expr(RETURN_LSE):
                    neg_inf_zl = arith.constant(float("-inf"), type=T.f32)
                    lse_base_z = ptr_base_i64(ptr_LSE)
                    lse_elem_z = q_tok_z * gdz + by
                    lse_byte_z = lse_elem_z * 4
                    lse_byte_z64 = fx.Int64(lse_byte_z)
                    lse_addr_z = lse_base_z + lse_byte_z64
                    lse_ptr_z = llvm_dialect.inttoptr(glbpz, lse_addr_z)
                    llvm_dialect.store(neg_inf_zl, lse_ptr_z)

        kv_nonzero = actual_kv_len > 0
        wg_valid_compute = wg_valid & kv_nonzero
        if wg_valid_compute:
            q_tok = q_start_tok + bx * 128
            q_offset = q_tok * stride_q_seq + by * stride_q_head
            q_num_bytes = q_end_tok * stride_q_seq
            q_rsrc = buffer_ops.create_buffer_resource(
                ptr_Q, num_records_bytes=q_num_bytes
            )
            q_frags_raw = phase4_q_load(
                lane_id,
                q_rsrc,
                stride_q_seq,
                wave_id,
                q_tile_offset_bytes=q_offset,
            )
            rocdl.sched_barrier(0)

            # Bridge prologue q_frags[4 bank][frags_per_bank] to core loop
            # q_tiles[4 msb][Q_WMMA_PER_MSB]. Each q_msb merges TWO adjacent
            # banks (lo+hi K-col halves).
            frags_per_bank = len(q_frags_raw[0])
            q_frags = [[None] * Q_WMMA_PER_MSB for _ in range(NUM_MSB)]
            pad = [None] * (Q_WMMA_PER_MSB - 2 * frags_per_bank)
            q_frags[0] = q_frags_raw[0] + q_frags_raw[1] + pad
            q_frags[2] = q_frags_raw[2] + q_frags_raw[3] + pad
            head_index = head_index_div(by, gqa)
            k_offset = k_start_tok * stride_k_seq + head_index * stride_k_head
            v_offset = k_start_tok * stride_v_seq + head_index * stride_v_head
            k_a_base_i32 = extract_lds_base_i32(lds_alloc_k_a.get_base())
            k_b_base_i32 = extract_lds_base_i32(lds_alloc_k_b.get_base())
            v_a_base_i32 = extract_lds_base_i32(lds_alloc_v_a.get_base())
            v_b_base_i32 = extract_lds_base_i32(lds_alloc_v_b.get_base())
            rocdl.sched_barrier(0)
            kv_lds_addrs_a = build_kv_lds_addrs(lane_id, k_a_base_i32, v_a_base_i32)
            kv_lds_addrs_b = build_kv_lds_addrs(lane_id, k_b_base_i32, v_b_base_i32)
            stride_k_32 = stride_k_seq * 32
            stride_v_32 = stride_v_seq * 32
            log2e_val = arith.constant(1.4426950408889634, type=T.f32)
            scale = log2e_val * scalar_f
            scale_pair = fx.vector.broadcast(T.vec(2, T.f32), scale)
            sgpr_state = {
                "s_log2e_scl": scale,
                "s_log2e_scl_pair": scale_pair,
            }
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

            zero_f32 = arith.constant(0.0, type=T.f32)
            neg_inf = arith.constant(float("-inf"), type=T.f32)
            zero_v8f32 = fx.constant_vector(0.0, T.vec(8, T.f32))
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
            all_su_sp_tiles = []
            for su in fx.range_constexpr(CNT_SU):
                kv_tiles_su = Fragment.load_k_su(ty, kv_lds_addrs_a, 0, su)
                fresh_sp = [[zero_v8f32] for msb in fx.range_constexpr(NUM_MSB)]
                fresh_sp = qk_gemm_pure(ty, 0, su, q_frags, kv_tiles_su, fresh_sp)
                all_su_sp_tiles.append(fresh_sp)
            TDM.load_v_only(
                ptr_V,
                v_offset,
                stride_v_seq,
                stride_v_32,
                wave_id,
                v_a_base_i32,
                oob_dg1_list=v_oob_dg1,
            )
            # Bottom-right aligned causal: shift n_start by -(sk - sq)
            causal_offset = actual_kv_len - actual_q_len
            if const_expr(IS_CAUSAL):
                pro_causal_n = -causal_offset
                apply_causal_mask(ctx, all_su_sp_tiles, pro_causal_n)
            apply_kv_oob_mask(ctx, all_su_sp_tiles, actual_kv_len)
            sp_pairs_all_pro = Softmax.tiles_to_pairs(all_su_sp_tiles)
            softmax_state_pro = make_softmax_state(
                [set_vgpr_bank(neg_inf, m) for m in range(NUM_MSB)],
                [set_vgpr_bank(neg_inf, m) for m in range(NUM_MSB)],
                [set_vgpr_bank(zero_f32, m) for m in range(NUM_MSB)],
                [set_vgpr_bank(zero_f32, m) for m in range(NUM_MSB)],
                sp_pairs_prev=sp_pairs_all_pro,
            )
            Softmax.part01_only(ty, 0, sp_pairs_all_pro, softmax_state_pro, sgpr_state)
            # PART2 first half for tile 0: ops 0..PART2_SPLIT-1
            pro_part2_ops = Softmax.build_all_part2_ops(
                ty, 0, sp_pairs_all_pro, softmax_state_pro, sgpr_state
            )
            for m in fx.range_constexpr(NUM_MSB):
                for op in pro_part2_ops[m][:PART2_SPLIT]:
                    op()

            tile_n_const = TILE_N
            ctx["tile_n_const"] = tile_n_const
            ctx["zero_v8f32"] = zero_v8f32
            kv_tiles_avail = (actual_kv_len + (TILE_N - 1)) // tile_n_const
            if const_expr(IS_CAUSAL):
                sk_sq_diff = actual_kv_len - actual_q_len
                sk_sq_tiles = (sk_sq_diff + (TILE_N - 1)) // tile_n_const
                bx_plus_1 = bx + 1
                causal_tiles = bx_plus_1 + sk_sq_tiles
                num_tiles = arith.minui(causal_tiles.ir_value(), kv_tiles_avail)
            else:
                num_tiles = kv_tiles_avail
            num_tiles_idx = arith.index_cast(T.index, num_tiles)
            num_tiles_minus1 = num_tiles - 1
            num_tiles_minus1_idx = arith.index_cast(T.index, num_tiles_minus1)
            rocdl.sched_barrier(0)
            k_tile1_stride = tile_n_const * stride_k_seq
            k_tile1_offset = k_offset + k_tile1_stride
            kv_remain_t1 = actual_kv_len - TILE_N
            k_tile1_oob_dg1 = TDM.build_oob_dg1_list(
                K_CFG_OOB,
                QK_HDIM,
                stride_k_elems_oob,
                kv_remain_t1,
                wave_id,
                dim0_stride=200,
            )
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
            kv_tiles_init = load_initial_kv_tiles(ty, kv_lds_addrs_b, blk=0, su=0)
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
            pro_partial_sp_lo_flat = []
            pro_partial_sp_hi_flat = []
            for m in fx.range_constexpr(NUM_MSB):
                for i in fx.range_constexpr(N_SP_PAIRS):
                    pair = Vec(sp_pairs_all_pro[m][i], dtype=fx.Float32)
                    pro_partial_sp_lo_flat.append(pair[0].ir_value())
                    pro_partial_sp_hi_flat.append(pair[1].ir_value())
            pro_partial_sp_flat = pro_partial_sp_lo_flat + pro_partial_sp_hi_flat
            pro_exp_delta = [
                softmax_state_pro["exp_delta"][m] for m in fx.range_constexpr(NUM_MSB)
            ]
            kv_flat_init = [
                kv_tiles_init[msb][k]
                for msb in fx.range_constexpr(NUM_MSB)
                for k in fx.range_constexpr(N_WMMA_K_TILES)
            ]
            sp_flat_init = [
                all_su_sp_tiles[su][msb][0]
                for su in fx.range_constexpr(CNT_SU)
                for msb in fx.range_constexpr(NUM_MSB)
            ]

            _KV_SIZE = NUM_MSB * N_WMMA_K_TILES
            _OFF_LOCAL_MAX = 24 + _KV_SIZE
            _OFF_DELTA = _OFF_LOCAL_MAX + NUM_MSB
            _OFF_SP = _OFF_DELTA + NUM_MSB
            _OFF_PP = _OFF_SP + CNT_SU * NUM_MSB
            _OFF_PSP = _OFF_PP + 4
            _PSP_SIZE = NUM_MSB * N_SP_PAIRS
            _OFF_PSP_HI = _OFF_PSP + _PSP_SIZE
            _OFF_PED = _OFF_PSP_HI + _PSP_SIZE
            o_flat_init = [zero_v8f32] * (NUM_MSB * N_PV_WMMA_N)
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

            if const_expr(IS_CAUSAL):
                first_causal_tile = bx + causal_offset // tile_n_const
                first_causal_tile = arith.maxsi(
                    first_causal_tile.ir_value(), arith.constant(1, type=T.i32)
                )
                first_causal_tile = arith.minui(first_causal_tile, num_tiles_minus1)
                first_causal_tile_idx = arith.index_cast(T.index, first_causal_tile)
            else:
                first_causal_tile_idx = num_tiles_minus1_idx

            for tile_idx, iter_args, loop1_results in scf.for_(
                arith.index(1),
                first_causal_tile_idx,
                arith.index(1),
                iter_args=init_args,
            ):
                yield tile_iteration(ctx, tile_idx, iter_args)

            for tile_idx, iter_args, loop_results in scf.for_(
                first_causal_tile_idx,
                num_tiles_minus1_idx,
                arith.index(1),
                iter_args=loop1_results,
            ):
                tile_idx_i32 = arith.index_cast(T.i32, tile_idx)
                causal_n = tile_idx_i32 * tile_n_const - causal_offset
                yield tile_iteration(ctx, tile_idx, iter_args, causal_n_start=causal_n)

            ep_o_tiles = [
                [
                    set_vgpr_bank(loop_results[d * N_PV_WMMA_N + n], d)
                    for n in fx.range_constexpr(N_PV_WMMA_N)
                ]
                for d in fx.range_constexpr(NUM_MSB)
            ]
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
            ep_k_cur_base = loop_results[_OFF_PP]
            ep_v_cur_base = loop_results[_OFF_PP + 1]
            ep_kv_lds_addrs = build_kv_lds_addrs(
                lane_id,
                ep_k_cur_base,
                ep_v_cur_base,
            )
            ep_partial_sp_lo = [
                loop_results[_OFF_PSP + i] for i in fx.range_constexpr(_PSP_SIZE)
            ]
            ep_partial_sp_hi = [
                loop_results[_OFF_PSP_HI + i] for i in fx.range_constexpr(_PSP_SIZE)
            ]
            ep_partial_sp_pairs = [
                [
                    make_v2f32(
                        ep_partial_sp_lo[m * N_SP_PAIRS + i],
                        ep_partial_sp_hi[m * N_SP_PAIRS + i],
                        m,
                    )
                    for i in fx.range_constexpr(N_SP_PAIRS)
                ]
                for m in fx.range_constexpr(NUM_MSB)
            ]
            ep_kv_tiles = [
                [
                    set_vgpr_bank(loop_results[24 + m * N_WMMA_K_TILES + k], m)
                    for k in fx.range_constexpr(N_WMMA_K_TILES)
                ]
                for m in fx.range_constexpr(NUM_MSB)
            ]
            ep_k_next_base = loop_results[_OFF_PP + 2]
            ep_v_next_base = loop_results[_OFF_PP + 3]
            ep_kv_lds_addrs_next = build_kv_lds_addrs(
                lane_id,
                ep_k_next_base,
                ep_v_next_base,
            )
            num_tiles_m1_ep = num_tiles - 1
            ep_v_endtile_offset = (
                v_offset + num_tiles_m1_ep * tile_n_const * stride_v_seq
            )
            ia_exp_delta = [
                set_vgpr_bank(loop_results[_OFF_PED + m], m)
                for m in fx.range_constexpr(NUM_MSB)
            ]
            emit_void("s_wait_idle")
            rocdl.s_barrier_signal(-1)
            rocdl.s_barrier_wait(-1)
            is_multi = num_tiles >= 2

            if is_multi:
                et_sp_t = [
                    [set_vgpr_bank(zero_v8f32, m)] for m in fx.range_constexpr(NUM_MSB)
                ]
                et_sfx = make_softmax_state(
                    ep_old_max,
                    ep_local_max,
                    ep_delta,
                    ep_row_sums,
                    sp_pairs_prev=[
                        [
                            ep_partial_sp_pairs[m][i]
                            for i in fx.range_constexpr(N_SP_PAIRS)
                        ]
                        for m in fx.range_constexpr(NUM_MSB)
                    ],
                )
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
                    et_tile_n_start = (
                        arith.index_cast(T.i32, num_tiles_idx) - 1
                    ) * TILE_N
                    et_causal_ns = et_tile_n_start - causal_offset
                else:
                    et_causal_ns = None
                et_kv_remain = actual_kv_len - (num_tiles - 1) * tile_n_const
                _et_V_CFG_OOB = (1 << 16) | V_TDM_CONFIG
                _et_stride_v_elems = stride_v_seq >> 1
                et_v_oob_dg1 = TDM.build_oob_dg1_list(
                    _et_V_CFG_OOB,
                    128,
                    _et_stride_v_elems,
                    et_kv_remain,
                    wave_id,
                )
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
                et_psp = [
                    [
                        make_v2f32(
                            et_psp_lo[rpsm * N_SP_PAIRS + rpi],
                            et_psp_hi[rpsm * N_SP_PAIRS + rpi],
                            rpsm,
                        )
                        for rpi in fx.range_constexpr(N_SP_PAIRS)
                    ]
                    for rpsm in fx.range_constexpr(NUM_MSB)
                ]
                tdm_wait_and_barrier()
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
            else:
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
        ptr_O: fx.Tensor,
        ptr_Q: fx.Tensor,
        ptr_K: fx.Tensor,
        ptr_V: fx.Tensor,
        ptr_LSE: fx.Tensor,
        ptr_cu_seqlens_q: fx.Tensor,
        ptr_cu_seqlens_k: fx.Tensor,
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
        f"Expected headdim_qk={HEAD_DIM_QK}, got {q.shape[-1]}"
    )
    assert v.shape[-1] == HEAD_DIM_V, (
        f"Expected headdim_v={HEAD_DIM_V}, got {v.shape[-1]}"
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
