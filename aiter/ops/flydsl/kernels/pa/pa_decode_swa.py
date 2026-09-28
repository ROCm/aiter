# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""FlyDSL sliding-window paged attention decode kernel."""

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import arith, const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import T
from aiter.ops.flydsl.kernels.pa.pa_common import _compute_block_base_dw_i64
from aiter.ops.flydsl.kernels import dpp_utils
from aiter.ops.flydsl.kernels.kernels_common import LOG2E, get_warp_size
from aiter.ops.flydsl.kernels.pa.utils import (
    cdiv,
    exp2_f32_fast,
    rcp_f32,
    udiv_const,
    unflatten_k,
    urem_const,
)
from aiter.ops.flydsl.kernels.pa.utils import copy_load as _copy_load
from aiter.ops.flydsl.kernels.pa.utils import copy_store as _copy_store
from aiter.ops.flydsl.kernels.pa.utils import global_pointer_from_addr as _global_pointer_from_addr
from aiter.ops.flydsl.kernels.pa.utils import load_global_16b as _load_global_16b

# ── Kernel geometry constants ────────────────────────────────────────
KV_BLOCK_SIZE = 1024  # physical page size (matches SP3 kBlockSize)
KV_COMPUTE_BLOCK = 256  # tile size (matches SP3 kTileKV)
NUM_WARPS = 4
WARP_SIZE = get_warp_size()
BLOCK_THREADS = NUM_WARPS * WARP_SIZE  # 256 on CDNA
MFMA_N = 16
MFMA_K = 32

TOKENS_PER_WARP = KV_COMPUTE_BLOCK // NUM_WARPS  # 64
TLOOP = TOKENS_PER_WARP // MFMA_N  # 4
ROWS_PER_WARP = WARP_SIZE // MFMA_N  # 4
FP8_ELEMS_16B = 16  # 16 FP8 per 16-byte load
QKHE_PER_FETCH = FP8_ELEMS_16B * ROWS_PER_WARP  # 64

VTLOOP = NUM_WARPS  # 4
Q_ELEMS_PER_LANE = 8
Q_CHUNKS_PER_LANE = Q_ELEMS_PER_LANE // 4

# LDS sizes
PROB_ROW_STRIDE_BYTES = 40  # 32 data + 8 padding -> 0 bank conflict
LDS_LOGITS_BYTES = NUM_WARPS * 4 * MFMA_N * PROB_ROW_STRIDE_BYTES  # 10240
LDS_SOFTMAX_BYTES = 2 * NUM_WARPS * MFMA_N * 4  # 512
LDS_SCALE_V_PADDING = 4  # break K/V same-bank paired writes
LDS_SCALE_V_OFFSET = KV_COMPUTE_BLOCK + LDS_SCALE_V_PADDING
LDS_SCALE_BYTES = (LDS_SCALE_V_OFFSET + KV_COMPUTE_BLOCK) * 4  # K/V per-token scale staging

FP8_MAX = 240.0
_FLAT_BUFFER_ELEMENTS = 1 << 30

# Tiles per block (1024 tokens / 256 tokens per tile = 4, matches SP3 kNumBlockTiles)
TILES_PER_BLOCK = KV_BLOCK_SIZE // KV_COMPUTE_BLOCK  # 4


def _get_sw_mtp_group_count(query_length: int, query_group_size: int) -> int:
    return cdiv(query_length * query_group_size, MFMA_N)


def _load_k_flat(
    k_global_ptr,
    k_copy_atom,
    k_register,
    k_block_base_dw_i64,
    tile_token_offset_i32,
    k_tok_thread_base,
    c_tok_stride_dw,
    k_he_off_dw,
    *,
    sched_vmem_after_load=True,
    qkhe_loop: int = 2,
):
    k_flat = []
    tile_tok_base = tile_token_offset_i32 + k_tok_thread_base

    for td in range_constexpr(TLOOP):
        kbo = tile_tok_base + fx.Int32(td * MFMA_N)
        kbo_dw = kbo * c_tok_stride_dw
        for qkhe in range_constexpr(qkhe_loop):
            ka_dw = k_block_base_dw_i64 + fx.Int64(kbo_dw + k_he_off_dw[qkhe])
            k2 = _load_global_16b(k_global_ptr, ka_dw * fx.Int64(4), k_copy_atom, k_register)
            if const_expr(sched_vmem_after_load):
                rocdl.sched_barrier(rocdl.mask_vmem_rd)
            k2_words = fx.Vector(k2)
            k_flat.append(k2_words[0])
            k_flat.append(k2_words[1])

    return k_flat


def _build_pa_k_thread_invariants(
    warp_id,
    lane16id,
    rowid,
    *,
    qkhe_loop: int = 2,
):
    c_tokens_per_warp = fx.Int32(TOKENS_PER_WARP)
    k_tok_thread_base = warp_id * c_tokens_per_warp + lane16id
    c_tok_stride_dw = fx.Int32(FP8_ELEMS_16B // 4)
    c_he_stride_dw = fx.Int32(KV_BLOCK_SIZE * FP8_ELEMS_16B // 4)
    k_he_off_dw = [rowid * c_he_stride_dw + fx.Int32(qkhe * 4) * c_he_stride_dw for qkhe in range(qkhe_loop)]
    return k_tok_thread_base, c_tok_stride_dw, k_he_off_dw


def _compute_sw_mtp_group_state(
    lane16id,
    local_qhead_idx,
    *,
    mtp_group_idx,
    query_length,
    query_group_size,
):
    g_off = mtp_group_idx * MFMA_N
    lane_pair_raw = lane16id + fx.Int32(g_off)
    c_total_pairs = fx.Int32(query_length * query_group_size)
    c_pair_max = fx.Int32(query_length * query_group_size - 1)
    c_ql_m1 = fx.Int32(query_length - 1)

    if const_expr((query_length * query_group_size) % MFMA_N == 0):
        lane_pair = lane_pair_raw
    else:
        lane_pair = arith.select(lane_pair_raw < c_total_pairs, lane_pair_raw, c_pair_max)
    qi_raw = udiv_const(lane_pair, query_group_size)
    if const_expr((query_length * query_group_size) % MFMA_N == 0):
        qi_val = qi_raw
    else:
        qi_val = arith.select(qi_raw < c_ql_m1, qi_raw, c_ql_m1)
    qhi_pos = urem_const(lane_pair, query_group_size)

    lqh_pair_raw = local_qhead_idx + fx.Int32(g_off)
    if const_expr((query_length * query_group_size) % MFMA_N == 0):
        lqh_pair = lqh_pair_raw
    else:
        lqh_pair = arith.select(lqh_pair_raw < c_total_pairs, lqh_pair_raw, c_pair_max)
    lqi_raw = udiv_const(lqh_pair, query_group_size)
    if const_expr((query_length * query_group_size) % MFMA_N == 0):
        qi_for_q = lqi_raw
    else:
        qi_for_q = arith.select(lqi_raw < c_ql_m1, lqi_raw, c_ql_m1)
    local_qhead_idx_for_q = urem_const(lqh_pair, query_group_size)
    return qi_val, qhi_pos, qi_for_q, local_qhead_idx_for_q


@flyc.jit
def _finish_q_fragments(
    logits_base,
    softmax_base,
    q_chunks,
    lane16id,
    rowid,
    local_qhead_idx,
    *,
    head_size: int,
    qkhe_loop: int,
    q_lanes_per_head: int,
):
    c_head_dw = fx.Int32(head_size // 4)
    lds_q_base = local_qhead_idx * c_head_dw + lane16id * 2
    abs_mask = fx.Vector.filled(4, 0x7FFFFFFF, fx.Int32)
    c_zero_f = fx.Float32(0.0)
    c_one_f = fx.Float32(1.0)
    q_f32_chunks = []
    local_max = c_zero_f
    for q_src in q_chunks:
        q_f32 = fx.Vector(q_src).to(fx.Float32)
        q_f32_chunks.append(q_f32)
        q_i32 = q_f32.bitcast(fx.Int32)
        q_abs_i32 = q_i32 & abs_mask
        q_abs = q_abs_i32.bitcast(fx.Float32)
        chunk_max = q_abs.reduce("max")
        local_max = local_max.maximumf(chunk_max)

    for sh in [8, 4, 2, 1]:
        local_max = local_max.maximumf(dpp_utils.dpp_xor_f32(local_max, sh))
    query_scale_lane = fx.Float32(
        arith.select(
            local_max > c_zero_f,
            local_max * fx.Float32(1.0 / FP8_MAX).ir_value(),
            c_one_f,
        )
    )
    inv_query_scale = rcp_f32(query_scale_lane)
    q_words = []
    for q_f32 in q_f32_chunks:
        p = q_f32 * inv_query_scale
        lo = rocdl.cvt_pk_fp8_f32(T.i32, p[0], p[1], fx.Int32(0), False)
        q_words.append(rocdl.cvt_pk_fp8_f32(T.i32, p[2], p[3], lo, True))
    q_w0, q_w1 = q_words

    if lane16id == fx.Int32(0):
        fx.ptr_store(
            fx.Vector.from_elements([query_scale_lane], dtype=fx.Float32),
            softmax_base + local_qhead_idx,
        )

    v01 = fx.Vector.from_elements([q_w0, q_w1], dtype=fx.Int32)
    if const_expr(q_lanes_per_head < MFMA_N):
        if lane16id < fx.Int32(q_lanes_per_head):
            fx.ptr_store(v01, logits_base + lds_q_base)
    else:
        fx.ptr_store(v01, logits_base + lds_q_base)

    q_frags = []
    gpu.barrier()
    query_scale_lane = fx.ptr_load(softmax_base + lane16id, result_type=fx.Vector.make_type(1, fx.Float32))[0]
    for qkhe in range_constexpr(qkhe_loop):
        for qkr in range_constexpr(2):
            lds_rd = lane16id * fx.Int32(head_size // 8) + fx.Int32(qkhe * 8) + rowid * fx.Int32(2) + fx.Int32(qkr)
            q_v1 = fx.ptr_load(
                fx.recast_iter(fx.Int64, logits_base) + (lds_rd), result_type=fx.Vector.make_type(1, fx.Int64)
            )
            q_frags.append(q_v1[0])
    return q_frags, query_scale_lane


def _prefetch_sw_mtp_group_query(
    q_tiles,
    q_copy_atom,
    q_register,
    batch_idx,
    kv_h,
    stride_q_seq,
    stride_q_head,
    lane16id,
    local_qhead_idx,
    *,
    mtp_group_idx,
    query_length,
    query_group_size,
    q_lanes_per_head,
):
    c_query_length = fx.Int32(query_length)
    c_query_group_size = fx.Int32(query_group_size)
    qi_val, qhi_pos, qi_for_q, local_qhead_idx_for_q = _compute_sw_mtp_group_state(
        lane16id,
        local_qhead_idx,
        mtp_group_idx=mtp_group_idx,
        query_length=query_length,
        query_group_size=query_group_size,
    )
    q_row = batch_idx * c_query_length + qi_for_q
    q_base = q_row * stride_q_seq + (kv_h * c_query_group_size + local_qhead_idx_for_q) * stride_q_head
    q_load_lane = lane16id
    if const_expr(q_lanes_per_head < MFMA_N):
        q_load_lane = (lane16id < fx.Int32(q_lanes_per_head)).select(lane16id, fx.Int32(0))
    q_elem = q_base + q_load_lane * fx.Int32(Q_ELEMS_PER_LANE)
    q_chunks = [
        _copy_load(q_tiles, q_elem + fx.Int32(qwi * 4), q_copy_atom, q_register)
        for qwi in range_constexpr(Q_CHUNKS_PER_LANE)
    ]
    return qi_val, qhi_pos, q_chunks


def _normalize_pa_output(running_sum, outs):
    safe_sum = arith.select(running_sum > fx.Float32(0.0), running_sum, fx.Float32(1.0))
    inv_sum = fx.Float32(rcp_f32(safe_sum))
    return [out * inv_sum for out in outs]


def _make_pa_phase_helpers(
    *,
    trans_v,
    per_token_kv,
    kv_h,
    v_global_ptr,
    v_copy_atom,
    v_register,
    ks_tiles,
    vs_tiles,
    scale_copy_atom,
    scale_register,
    logits_base,
    softmax_base,
    scale_base,
    stride_ks_block,
    stride_ks_head,
    softmax_scale_base,
    k_scale_val,
    v_scale_val,
    warp_id,
    lane16id,
    rowid,
    head_size: int = 128,
):
    qkhe_loop = head_size // QKHE_PER_FETCH
    vhe_loop = head_size // MFMA_N // NUM_WARPS
    c_mfma_n = fx.Int32(MFMA_N)

    vhead_elems = [fx.Int32(vhe * NUM_WARPS * MFMA_N) + warp_id * c_mfma_n + lane16id for vhe in range(vhe_loop)]
    v_tok_thread_off = [fx.Int32(vt * TOKENS_PER_WARP) + rowid * c_mfma_n for vt in range(VTLOOP)]
    if const_expr(trans_v):
        vhead_elem_dw = [vhead_elems[vhe] * fx.Int32(FP8_ELEMS_16B // 4) for vhe in range(vhe_loop)]
    else:
        vhead_elem_dw = [vhead_elems[vhe] * fx.Int32(KV_BLOCK_SIZE // 4) for vhe in range(vhe_loop)]

    kv_tok_thread_base = warp_id * fx.Int32(TOKENS_PER_WARP) + rowid * 4
    rowid_8x8 = rowid >> fx.Int32(1)
    offset_in_slot = rowid & fx.Int32(1)
    prob_row_i32 = PROB_ROW_STRIDE_BYTES // 4
    prob_row_i64 = PROB_ROW_STRIDE_BYTES // 8
    prob_wr_thread_base = (
        warp_id * fx.Int32(4 * MFMA_N * prob_row_i32)
        + lane16id * fx.Int32(prob_row_i32)
        + rowid_8x8 * fx.Int32(2)
        + offset_in_slot
    )
    pv_prob_read_base = rowid * fx.Int32(MFMA_N * prob_row_i64) + lane16id * fx.Int32(prob_row_i64)

    sm_lane_wave_base = lane16id * fx.Int32(NUM_WARPS)
    sm_max_off = sm_lane_wave_base + warp_id
    sm_sum_off = fx.Int32(NUM_WARPS * MFMA_N) + sm_lane_wave_base + warp_id
    sm_rd_max_offs = [sm_lane_wave_base + fx.Int32(w) for w in range(NUM_WARPS)]
    sm_rd_sum_offs = [fx.Int32(NUM_WARPS * MFMA_N) + sm_lane_wave_base + fx.Int32(w) for w in range(NUM_WARPS)]

    sm_vmax_wr_off = None
    sm_vmax_rd_offs = None
    if const_expr(per_token_kv):
        sm_vmax_wr_off = fx.Int32(2 * NUM_WARPS * MFMA_N) + sm_lane_wave_base + warp_id
        sm_vmax_rd_offs = [fx.Int32(2 * NUM_WARPS * MFMA_N) + sm_lane_wave_base + fx.Int32(w) for w in range(NUM_WARPS)]

    c_w = fx.Int32(WARP_SIZE)
    neg_inf = fx.Float32(float("-inf"))
    zero_f = fx.Float32(0.0)

    # Sliding-window decode always needs an upper-bound mask: even for a
    # single query, the tail block can contain tokens beyond context_len.
    pv_prob_i64_elem_offs = []
    for vt in range_constexpr(VTLOOP):
        for j in range_constexpr(2):
            p_elem = fx.Int32(vt * 4 * MFMA_N * (PROB_ROW_STRIDE_BYTES // 8)) + pv_prob_read_base + fx.Int32(j)
            pv_prob_i64_elem_offs.append(p_elem)

    def _load_kv_scale_scalars(tile_token_offset_i32, phys_block):
        if const_expr(per_token_kv):
            scale_block_base = phys_block * stride_ks_block + kv_h * stride_ks_head
            scale_stage_token = warp_id * fx.Int32(WARP_SIZE) + rowid * fx.Int32(MFMA_N) + lane16id
            scale_global_token = tile_token_offset_i32 + scale_stage_token
            k_scale_scalar = _copy_load(
                ks_tiles,
                scale_block_base + scale_global_token,
                scale_copy_atom,
                scale_register,
            )[0]
            v_scale_scalar = _copy_load(
                vs_tiles,
                scale_block_base + scale_global_token,
                scale_copy_atom,
                scale_register,
            )[0]
            return k_scale_scalar, v_scale_scalar
        return None

    def _load_v_and_scales(
        v_block_base_dw,
        tile_token_offset_i32,
        *,
        preloaded_scale_scalars=None,
    ):
        if const_expr(per_token_kv):
            scale_stage_token = warp_id * fx.Int32(WARP_SIZE) + rowid * fx.Int32(MFMA_N) + lane16id
            k_scale_scalar, v_scale_scalar = preloaded_scale_scalars
            fx.ptr_store(
                fx.Vector.from_elements([k_scale_scalar], dtype=fx.Float32),
                scale_base + scale_stage_token,
            )
            fx.ptr_store(
                fx.Vector.from_elements([v_scale_scalar], dtype=fx.Float32),
                scale_base + (fx.Int32(LDS_SCALE_V_OFFSET) + scale_stage_token),
            )
            rocdl.sched_barrier(rocdl.mask_vmem_rd)

        v_results = []
        for vt in range_constexpr(VTLOOP):
            vhe_data = []
            for vhe in range_constexpr(vhe_loop):
                v_token_in_block = tile_token_offset_i32 + v_tok_thread_off[vt]
                if const_expr(trans_v):
                    vt_group = v_token_in_block >> fx.Int32(4)
                    va_dw_delta = vt_group * fx.Int32(head_size * FP8_ELEMS_16B // 4) + vhead_elem_dw[vhe]
                else:
                    va_dw_delta = vhead_elem_dw[vhe] + (v_token_in_block >> fx.Int32(2))
                va_byte = (v_block_base_dw + fx.Int64(va_dw_delta)) * fx.Int64(4)
                v_i64x2 = _load_global_16b(v_global_ptr, va_byte, v_copy_atom, v_register)
                rocdl.sched_barrier(rocdl.mask_vmem_rd)
                vhe_data.append(v_i64x2)
            v_results.append(vhe_data)

        return v_results

    def _scale_row_base(td: int):
        return kv_tok_thread_base + fx.Int32(td * MFMA_N)

    def _load_k_scale_vec(td: int):
        return fx.ptr_load(scale_base + (_scale_row_base(td)), result_type=fx.Vector.make_type(4, fx.Float32))

    def _load_v_scale_vec(td: int):
        return fx.ptr_load(
            scale_base + (fx.Int32(LDS_SCALE_V_OFFSET) + _scale_row_base(td)),
            result_type=fx.Vector.make_type(4, fx.Float32),
        )

    def _store_vmax_warp(partition_start, *, seq_end=None):
        if const_expr(per_token_kv):
            kv_tok_base = partition_start + kv_tok_thread_base if const_expr(seq_end is not None) else None
            v_max_warp = zero_f
            for td in range_constexpr(TLOOP):
                vs = fx.Vector(_load_v_scale_vec(td))
                masked_values = []
                for i in range_constexpr(4):
                    vs_i = vs[i]
                    if const_expr(kv_tok_base is not None):
                        kv_tok = kv_tok_base + fx.Int32(td * MFMA_N + i)
                        vs_i = arith.select(kv_tok < seq_end, vs_i, zero_f)
                    masked_values.append(vs_i)
                vs = fx.Vector.from_elements(masked_values, dtype=fx.Float32)
                v_max_warp = v_max_warp.maximumf(vs.reduce("max"))
            for sh in [32, 16]:
                v_max_warp = v_max_warp.maximumf(v_max_warp.shuffle_xor(fx.Int32(sh), c_w))
            fx.ptr_store(
                fx.Vector.from_elements([v_max_warp], dtype=fx.Float32),
                softmax_base + sm_vmax_wr_off,
            )

    def _token_vec_i32(kv_tok_base, td: int):
        kv_tok_td_base = kv_tok_base + fx.Int32(td * MFMA_N)
        return fx.Vector.from_elements(
            [kv_tok_td_base + fx.Int32(i) for i in range_constexpr(4)],
            dtype=fx.Int32,
        )

    def _apply_token_mask_vec(logit_vec, td: int, kv_tok_base, causal_bound, seq_start, false_value):
        tok_vec = _token_vec_i32(kv_tok_base, td)
        in_range = (tok_vec < causal_bound) & (tok_vec >= seq_start)
        false_vec = fx.Vector.from_elements([false_value], dtype=fx.Float32).broadcast_to(4)
        return in_range.select(logit_vec, false_vec)

    def _qk_and_intra_softmax(
        k_ops,
        partition_start,
        q_frags,
        causal_bound,
        query_scale_lane=None,
        *,
        seq_start,
    ):
        query_scale_vec = fx.Vector.from_elements(
            [query_scale_lane * softmax_scale_base],
            dtype=fx.Float32,
        ).broadcast_to(4)
        d_out = []
        for td in range_constexpr(TLOOP):
            acc = fx.Vector.filled(4, 0.0, fx.Float32)
            for k_step in range_constexpr(qkhe_loop * 2):
                acc = rocdl.mfma_f32_16x16x32_fp8_fp8(T.f32x4, [k_ops[td][k_step], q_frags[k_step], acc, 0, 0, 0])
            if const_expr(per_token_kv):
                k_scale_vec = _load_k_scale_vec(td)
                d_out.append(acc * (k_scale_vec * query_scale_vec))
            else:
                d_out.append(acc * (query_scale_vec * k_scale_val))

        kv_tok_base = partition_start + kv_tok_thread_base
        qk_max = neg_inf
        for td in range_constexpr(TLOOP):
            logits_vec = _apply_token_mask_vec(d_out[td], td, kv_tok_base, causal_bound, seq_start, neg_inf)
            d_out[td] = logits_vec
            qk_max = qk_max.maximumf(fx.Vector(logits_vec).reduce("max"))
        for sh in [32, 16]:
            qk_max = qk_max.maximumf(qk_max.shuffle_xor(fx.Int32(sh), c_w))
        fx.ptr_store(
            fx.Vector.from_elements([qk_max], dtype=fx.Float32),
            softmax_base + sm_max_off,
        )

        exp_sum = zero_f
        safe_qk_max = arith.select(qk_max > neg_inf, qk_max, zero_f)
        for td in range_constexpr(TLOOP):
            diff_vec = fx.Vector(d_out[td]) - safe_qk_max
            p_vec = exp2_f32_fast(diff_vec * fx.Float32(LOG2E))
            exp_sum = exp_sum + fx.Vector(p_vec).reduce("add")
            d_out[td] = p_vec
        for sh in [32, 16]:
            exp_sum = exp_sum + exp_sum.shuffle_xor(fx.Int32(sh), c_w)
        fx.ptr_store(
            fx.Vector.from_elements([exp_sum], dtype=fx.Float32),
            softmax_base + sm_sum_off,
        )

        return d_out

    def _cross_warp_softmax_and_prob_pack(d_out, rmax, rsum, outs):
        partition_max = neg_inf
        partition_sum = zero_f
        warp_rescale_factors = []
        max_vec = fx.ptr_load(softmax_base + (sm_rd_max_offs[0]), result_type=fx.Vector.make_type(4, fx.Float32))
        for w in range_constexpr(NUM_WARPS):
            w_max = max_vec[w]
            partition_max = partition_max.maximumf(w_max)
            warp_rescale_factors.append(w_max)
        sum_vec = fx.ptr_load(softmax_base + (sm_rd_sum_offs[0]), result_type=fx.Vector.make_type(4, fx.Float32))
        for w in range_constexpr(NUM_WARPS):
            diff_w = warp_rescale_factors[w] - partition_max
            diff_w = arith.select(partition_max > neg_inf, diff_w, zero_f)
            wf = exp2_f32_fast(diff_w * fx.Float32(LOG2E).ir_value())
            w_sum = sum_vec[w]
            wf_sum = arith.mulf(arith.unwrap(w_sum), arith.unwrap(wf), fastmath=arith.FastMathFlags.contract)
            partition_sum = arith.addf(arith.unwrap(partition_sum), wf_sum, fastmath=arith.FastMathFlags.contract)
            warp_rescale_factors[w] = wf

        my_warp_rescale = warp_rescale_factors[0]
        for w in range_constexpr(1, NUM_WARPS):
            my_warp_rescale = arith.select(
                warp_id == fx.Int32(w),
                warp_rescale_factors[w],
                my_warp_rescale,
            )

        new_rmax = rmax.maximumf(partition_max)
        accum_scale = arith.select(
            rmax > neg_inf,
            exp2_f32_fast((rmax - new_rmax) * fx.Float32(LOG2E).ir_value()),
            zero_f,
        )
        part_to_new = arith.select(
            partition_max > neg_inf,
            exp2_f32_fast((partition_max - new_rmax) * fx.Float32(LOG2E).ir_value()),
            zero_f,
        )

        accum_sum = arith.mulf(arith.unwrap(accum_scale), arith.unwrap(rsum), fastmath=arith.FastMathFlags.contract)
        partition_sum_scaled = arith.mulf(
            arith.unwrap(partition_sum),
            arith.unwrap(part_to_new),
            fastmath=arith.FastMathFlags.contract,
        )
        rsum = arith.addf(accum_sum, partition_sum_scaled, fastmath=arith.FastMathFlags.contract)
        rmax = new_rmax
        for vhe in range_constexpr(vhe_loop):
            outs[vhe] = outs[vhe] * fx.Float32(accum_scale)

        if const_expr(per_token_kv):
            v_max_global = zero_f
            vmax_vec = fx.ptr_load(softmax_base + (sm_vmax_rd_offs[0]), result_type=fx.Vector.make_type(4, fx.Float32))
            for w in range_constexpr(NUM_WARPS):
                w_vmax = vmax_vec[w]
                v_max_global = v_max_global.maximumf(w_vmax)
            v_max_scaled = v_max_global * fx.Float32(1.0 / FP8_MAX).ir_value()
            v_max_safe_scaled = v_max_scaled + fx.Float32(1e-8 / FP8_MAX).ir_value()
            norm_factor = rcp_f32(v_max_safe_scaled)
            prob_scale = my_warp_rescale
            v_correction = v_max_scaled * part_to_new
            for td in range_constexpr(TLOOP):
                d_out[td] = d_out[td] * (_load_v_scale_vec(td) * fx.Float32(prob_scale * norm_factor))
        else:
            prob_scale = my_warp_rescale * part_to_new
            v_correction = v_scale_val
            for td in range_constexpr(TLOOP):
                d_out[td] = d_out[td] * fx.Float32(prob_scale)

        for td in range_constexpr(TLOOP):
            pv = fx.Vector(d_out[td])
            lo = rocdl.cvt_pk_fp8_f32(T.i32, pv[0], pv[1], fx.Int32(0), False)
            pk = rocdl.cvt_pk_fp8_f32(T.i32, pv[2], pv[3], lo, True)
            elem_base = prob_wr_thread_base + fx.Int32(td * MFMA_N * (PROB_ROW_STRIDE_BYTES // 4))
            pk_vec = fx.Vector.from_elements([pk], dtype=fx.Int32)
            fx.ptr_store(pk_vec, logits_base + elem_base)
        return rmax, rsum, outs, v_correction

    def _pv_mfma(v_ops, outs, v_correction):
        v_correction = fx.Float32(v_correction)
        fm_contract = arith.FastMathFlags.contract
        v_correction_vec = fx.Vector.from_elements([v_correction], dtype=fx.Float32).broadcast_to(4)
        for vhe in range_constexpr(vhe_loop):
            tmp_out = fx.Vector.filled(4, 0.0, fx.Float32)
            for vt in range_constexpr(VTLOOP):
                v_i64x2 = fx.Vector(v_ops[vt][vhe])
                for j in range_constexpr(2):
                    p_elem_off = pv_prob_i64_elem_offs[vt * 2 + j]
                    p_i64 = fx.ptr_load(
                        fx.recast_iter(fx.Int64, logits_base) + (p_elem_off),
                        result_type=fx.Vector.make_type(1, fx.Int64),
                    )[0]
                    tmp_out = rocdl.mfma_f32_16x16x32_fp8_fp8(
                        T.f32x4,
                        [
                            v_i64x2[j],
                            p_i64,
                            tmp_out,
                            0,
                            0,
                            0,
                        ],
                    )
            outs[vhe] = arith.addf(
                arith.mulf(tmp_out, v_correction_vec, fastmath=fm_contract),
                outs[vhe],
                fastmath=fm_contract,
            )
        return outs

    return (
        _load_kv_scale_scalars,
        _load_v_and_scales,
        _store_vmax_warp,
        _qk_and_intra_softmax,
        _cross_warp_softmax_and_prob_pack,
        _pv_mfma,
    )


@functools.lru_cache(maxsize=256)
def compile_pa_decode_sw_reduce(
    *,
    max_context_partition_num: int,
    query_seq_len: int,
    query_group_size: int,
    head_size: int,
    output_dtype_str: str,
    logits_dtype_str: str = "bf16",
):
    # Partition partials (`logits`) are read at this dtype; defaults to bf16
    # (this kernel's original, still-default behavior for existing callers).
    # Callers with an fp8 query should still keep this at bf16 (fp8 has too
    # little precision for a re-accumulated intermediate) -- only real f16/f32
    # queries should pick a matching non-bf16 value here.
    if logits_dtype_str == "f32":
        LOGITS_DTYPE = fx.Float32
    elif logits_dtype_str == "f16":
        LOGITS_DTYPE = fx.Float16
    else:
        LOGITS_DTYPE = fx.BFloat16
    if output_dtype_str == "f32":
        OUTPUT_DTYPE = fx.Float32
    elif output_dtype_str == "f16":
        OUTPUT_DTYPE = fx.Float16
    else:
        OUTPUT_DTYPE = fx.BFloat16
    block_threads = head_size
    assert block_threads > 0, "head_size must be positive"
    assert block_threads <= 1024, "head_size must fit in one workgroup"
    reduce_width = 1 if max_context_partition_num <= 1 else 1 << ((max_context_partition_num - 1).bit_length())
    reduce_shuffle_offsets = [off for off in [32, 16, 8, 4, 2, 1] if off < reduce_width]
    red_slots = max(1, (block_threads + WARP_SIZE - 1) // WARP_SIZE)

    @fx.struct
    class SharedStorage:
        red: fx.Array[fx.Float32, red_slots, 16]
        part_weights: fx.Array[fx.Float32, max_context_partition_num, 16]

    @flyc.kernel(known_block_size=(block_threads, 1, 1))
    def pa_decode_sw_reduce_kernel(
        # Raw-pointer kernargs: bare i64 data_ptr() (strides are explicit args).
        output_ptr: fx.Int64,
        exp_sums_ptr: fx.Int64,
        max_logits_ptr: fx.Int64,
        logits_ptr: fx.Int64,
        stride_output_bs: fx.Int32,
        stride_output_len: fx.Int32,
        stride_output_kv_head: fx.Int32,
        stride_output_group_size: fx.Int32,
        stride_exp_sums_seq: fx.Int32,
        stride_exp_sums_head: fx.Int32,
        stride_exp_sums_part: fx.Int32,
        stride_logits_seq: fx.Int32,
        stride_logits_head: fx.Int32,
        stride_logits_part: fx.Int32,
        stride_logits_group: fx.Int32,
    ):
        tid = fx.Int32(gpu.thread_id("x"))
        batch_idx = fx.Int32(gpu.block_id("x"))
        kv_head_idx = fx.Int32(gpu.block_id("y"))
        eqgs_idx = fx.Int32(gpu.block_id("z"))

        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        red_scratch = lds.red.view(fx.make_layout(red_slots, 1))
        if const_expr(max_context_partition_num > WARP_SIZE):
            part_weights_lds = lds.part_weights.view(fx.make_layout(max_context_partition_num, 1))

        def _divide_addr(addr, dtype):
            pointer = _global_pointer_from_addr(addr, dtype, alignment=dtype.width // 8)
            flat = fx.make_view(pointer, fx.make_layout(_FLAT_BUFFER_ELEMENTS, 1))
            return fx.logical_divide(
                fx.rocdl.make_buffer_tensor(flat),
                fx.make_layout(1, 1),
            )

        output = _divide_addr(output_ptr, OUTPUT_DTYPE)
        exp_sums = _divide_addr(exp_sums_ptr, fx.Float32)
        max_logits = _divide_addr(max_logits_ptr, fx.Float32)
        logits = _divide_addr(logits_ptr, LOGITS_DTYPE)

        copy_f32 = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Float32)
        copy_logits = fx.make_copy_atom(
            fx.rocdl.BufferCopy32b() if LOGITS_DTYPE.width == 32 else fx.rocdl.BufferCopy16b(),
            LOGITS_DTYPE,
        )
        copy_output = fx.make_copy_atom(
            fx.rocdl.BufferCopy32b() if OUTPUT_DTYPE.width == 32 else fx.rocdl.BufferCopy16b(),
            OUTPUT_DTYPE,
        )
        f32_register = fx.make_rmem_tensor(1, fx.Float32)
        logits_register = fx.make_rmem_tensor(1, LOGITS_DTYPE)
        output_register = fx.make_rmem_tensor(1, OUTPUT_DTYPE)

        c_zero_f = fx.Float32(0.0)
        c_one_f = fx.Float32(1.0)
        c_neg_inf = fx.Float32(float("-inf"))
        c_log2e = fx.Float32(LOG2E)
        fm_fast = arith.FastMathFlags.fast

        c_w = fx.Int32(WARP_SIZE)
        c_wave_mask = fx.Int32(WARP_SIZE - 1)
        c_red_slots = fx.Int32(red_slots)
        lane = tid & c_wave_mask
        wave = fx.Int32(tid >> fx.Int32(6))

        def _wave_reduce_max_full(val):
            red = val
            for sh in [32, 16, 8, 4, 2, 1]:
                red = red.maximumf(red.shuffle_xor(fx.Int32(sh), c_w))
            return red

        def _wave_reduce_sum_full(val):
            red = val
            for sh in [32, 16, 8, 4, 2, 1]:
                red = red.addf(
                    red.shuffle_xor(fx.Int32(sh), c_w),
                    fastmath=fm_fast,
                )
            return red

        def _block_reduce(val, mode):
            if const_expr(red_slots == 1):
                return _wave_reduce_max_full(val) if const_expr(mode == "max") else _wave_reduce_sum_full(val)

            neutral = c_neg_inf if const_expr(mode == "max") else c_zero_f
            w = _wave_reduce_max_full(val) if const_expr(mode == "max") else _wave_reduce_sum_full(val)

            if lane == 0:
                wave_idx = fx.Int32(wave)
                fx.memref_store(w, red_scratch, wave_idx)
            gpu.barrier()

            if wave == 0:
                in_range = lane < c_red_slots
                lane_safe = arith.select(in_range, lane, 0)
                lane_safe_idx = fx.Int32(lane_safe)
                red_val = fx.memref_load(red_scratch, lane_safe_idx)
                red_val = arith.select(in_range, red_val, neutral)
                red_val = (
                    _wave_reduce_max_full(red_val) if const_expr(mode == "max") else _wave_reduce_sum_full(red_val)
                )
                if lane == 0:
                    fx.memref_store(red_val, red_scratch, 0)
            gpu.barrier()

            return fx.memref_load(red_scratch, 0)

        if const_expr(max_context_partition_num <= WARP_SIZE):
            c_part_num = fx.Int32(max_context_partition_num)
            c_reduce_width = fx.Int32(reduce_width)

            def _wave_reduce_max(val):
                red = val
                for sh in reduce_shuffle_offsets:
                    red = red.maximumf(red.shuffle_xor(fx.Int32(sh), c_w))
                return red

            def _wave_reduce_sum(val):
                red = val
                for sh in reduce_shuffle_offsets:
                    red = red.addf(
                        red.shuffle_xor(fx.Int32(sh), c_w),
                        fastmath=fm_fast,
                    )
                return red

            lane_in_range = lane < c_part_num
            lane_in_reduce = lane < c_reduce_width
            part_sum = c_zero_f
            part_max = c_neg_inf
            if lane_in_reduce:
                part_i32 = arith.select(lane_in_range, lane, 0)
                es_off = (
                    batch_idx * stride_exp_sums_seq
                    + kv_head_idx * stride_exp_sums_head
                    + part_i32 * stride_exp_sums_part
                    + eqgs_idx
                )
                part_sum_raw = _copy_load(exp_sums, es_off, copy_f32, f32_register)[0]
                part_max_raw = _copy_load(max_logits, es_off, copy_f32, f32_register)[0]
                part_sum = arith.select(lane_in_range, part_sum_raw, c_zero_f)
                part_max = arith.select(lane_in_range, part_max_raw, c_neg_inf)

            global_max = _wave_reduce_max(part_max)
            part_scale = arith.select(
                lane_in_range,
                exp2_f32_fast((part_max - global_max) * c_log2e),
                c_zero_f,
            )
            scaled_sum = part_sum * part_scale
            global_exp_sum = _wave_reduce_sum(scaled_sum)
            safe_global_exp_sum = arith.select(
                global_exp_sum > c_zero_f,
                global_exp_sum,
                c_one_f,
            )
            inv_global_exp_sum = rcp_f32(safe_global_exp_sum)
            weight_local = scaled_sum * inv_global_exp_sum
            weight_local_i32 = arith.bitcast(T.i32, arith.unwrap(weight_local))

            acc = c_zero_f
            for part_idx in range_constexpr(max_context_partition_num):
                part_i32 = fx.Int32(part_idx)
                bcast_addr = part_i32 * 4
                weight_i32 = rocdl.ds_bpermute(T.i32, arith.unwrap(bcast_addr), arith.unwrap(weight_local_i32))
                weight = arith.bitcast(T.f32, weight_i32)
                logits_off = (
                    batch_idx * stride_logits_seq
                    + kv_head_idx * stride_logits_head
                    + part_i32 * stride_logits_part
                    + eqgs_idx * stride_logits_group
                    + tid
                )
                part_logits_raw = _copy_load(logits, logits_off, copy_logits, logits_register)[0]
                part_logits = fx.Float32(part_logits_raw)
                acc = acc + part_logits * weight
        else:
            # Fallback for unusually large sliding-window partition counts.
            global_max = c_neg_inf
            for chunk_base in range(0, max_context_partition_num, block_threads):
                chunk_size = min(block_threads, max_context_partition_num - chunk_base)
                c_chunk_size = fx.Int32(chunk_size)
                c_chunk_base = fx.Int32(chunk_base)
                in_chunk = tid < c_chunk_size
                part_i32 = arith.select(in_chunk, tid + c_chunk_base, 0)
                es_off = (
                    batch_idx * stride_exp_sums_seq
                    + kv_head_idx * stride_exp_sums_head
                    + part_i32 * stride_exp_sums_part
                    + eqgs_idx
                )
                part_max_raw = _copy_load(max_logits, es_off, copy_f32, f32_register)[0]
                part_max = arith.select(in_chunk, part_max_raw, c_neg_inf)
                chunk_max = _block_reduce(part_max, "max")
                global_max = global_max.maximumf(chunk_max)

            global_exp_sum = c_zero_f
            for chunk_base in range(0, max_context_partition_num, block_threads):
                chunk_size = min(block_threads, max_context_partition_num - chunk_base)
                c_chunk_size = fx.Int32(chunk_size)
                c_chunk_base = fx.Int32(chunk_base)
                in_chunk = tid < c_chunk_size
                part_i32 = arith.select(in_chunk, tid + c_chunk_base, 0)
                es_off = (
                    batch_idx * stride_exp_sums_seq
                    + kv_head_idx * stride_exp_sums_head
                    + part_i32 * stride_exp_sums_part
                    + eqgs_idx
                )
                part_sum_raw = _copy_load(exp_sums, es_off, copy_f32, f32_register)[0]
                part_max_raw = _copy_load(max_logits, es_off, copy_f32, f32_register)[0]
                part_sum = arith.select(in_chunk, part_sum_raw, c_zero_f)
                part_max = arith.select(in_chunk, part_max_raw, c_neg_inf)
                part_scale = arith.select(
                    in_chunk,
                    exp2_f32_fast((part_max - global_max) * c_log2e),
                    c_zero_f,
                )
                chunk_sum = _block_reduce(part_sum * part_scale, "sum")
                global_exp_sum = global_exp_sum + chunk_sum

            safe_global_exp_sum = arith.select(
                global_exp_sum > c_zero_f,
                global_exp_sum,
                c_one_f,
            )
            inv_global_exp_sum = rcp_f32(safe_global_exp_sum)

            for chunk_base in range(0, max_context_partition_num, block_threads):
                chunk_size = min(block_threads, max_context_partition_num - chunk_base)
                c_chunk_size = fx.Int32(chunk_size)
                c_chunk_base = fx.Int32(chunk_base)
                in_chunk = tid < c_chunk_size
                part_i32 = arith.select(in_chunk, tid + c_chunk_base, 0)
                es_off = (
                    batch_idx * stride_exp_sums_seq
                    + kv_head_idx * stride_exp_sums_head
                    + part_i32 * stride_exp_sums_part
                    + eqgs_idx
                )
                part_sum_raw = _copy_load(exp_sums, es_off, copy_f32, f32_register)[0]
                part_max_raw = _copy_load(max_logits, es_off, copy_f32, f32_register)[0]
                part_sum = arith.select(in_chunk, part_sum_raw, c_zero_f)
                part_max = arith.select(in_chunk, part_max_raw, global_max)
                part_scale = exp2_f32_fast((part_max - global_max) * c_log2e)
                weight = part_sum * part_scale * inv_global_exp_sum
                if in_chunk:
                    part_idx_idx = fx.Int32(part_i32)
                    fx.memref_store(weight, part_weights_lds, part_idx_idx)

            gpu.barrier()

            acc = c_zero_f
            for part_idx in range_constexpr(max_context_partition_num):
                part_i32 = fx.Int32(part_idx)
                part_idx_idx = fx.Int32(part_idx)
                weight = fx.memref_load(part_weights_lds, part_idx_idx)
                logits_off = (
                    batch_idx * stride_logits_seq
                    + kv_head_idx * stride_logits_head
                    + part_i32 * stride_logits_part
                    + eqgs_idx * stride_logits_group
                    + tid
                )
                part_logits_raw = _copy_load(logits, logits_off, copy_logits, logits_register)[0]
                part_logits = fx.Float32(part_logits_raw)
                acc = acc + part_logits * weight

        query_idx = udiv_const(eqgs_idx, query_group_size)
        group_idx = urem_const(eqgs_idx, query_group_size)
        out_off = (
            batch_idx * stride_output_bs
            + query_idx * stride_output_len
            + kv_head_idx * stride_output_kv_head
            + group_idx * stride_output_group_size
            + tid
        )
        out_val = acc if const_expr(output_dtype_str == "f32") else acc.to(OUTPUT_DTYPE)
        _copy_store(
            output,
            out_off,
            copy_output,
            output_register,
            fx.Vector.from_elements([out_val], dtype=OUTPUT_DTYPE),
        )

    @flyc.jit
    def launch_pa_decode_sw_reduce(
        output: fx.Int64,
        exp_sums: fx.Int64,
        max_logits: fx.Int64,
        logits: fx.Int64,
        stride_output_bs,
        stride_output_len,
        stride_output_kv_head,
        stride_output_group_size,
        stride_exp_sums_seq,
        stride_exp_sums_head,
        stride_exp_sums_part,
        stride_logits_seq,
        stride_logits_head,
        stride_logits_part,
        stride_logits_group,
        batch_size,
        num_kv_heads,
        stream: fx.Stream = fx.Stream(None),
    ):
        pa_decode_sw_reduce_kernel(
            output,
            exp_sums,
            max_logits,
            logits,
            stride_output_bs,
            stride_output_len,
            stride_output_kv_head,
            stride_output_group_size,
            stride_exp_sums_seq,
            stride_exp_sums_head,
            stride_exp_sums_part,
            stride_logits_seq,
            stride_logits_head,
            stride_logits_part,
            stride_logits_group,
        ).launch(
            grid=(batch_size, num_kv_heads, query_seq_len * query_group_size),
            block=(block_threads, 1, 1),
            stream=stream,
        )

    return {
        "launch": launch_pa_decode_sw_reduce,
        "kernel": pa_decode_sw_reduce_kernel,
    }


# =====================================================================
# =====================================================================
# compile_pa_decode_sw — Sliding Window kernel with one CTA per 256-token tile
# Grid = (batch_size, num_kv_heads, max_context_partition_num)
# Each block handles one 256-token context partition. `partition_idx` is decoded
# into (physical_block, 256-token sub-tile) after applying the sliding-window offset.
# Uses block_tables for physical block lookup instead of kv_page_indices.
# Output: exp_sums, max_logits, temporary_output -> reduced by a separate kernel.
# =====================================================================
@functools.lru_cache(maxsize=256)
def compile_pa_decode_sw(
    sliding_window: int,  # required > 0 -- baked as compile-time constant
    max_context_partition_num: int,
    softmax_scale=None,
    trans_v=False,
    query_group_size=16,
    per_token_kv=False,
    query_length: int = 1,
    query_input_dtype: str = "bf16",
    head_dim: int = 128,
):
    """Compile a Gluon-style partitioned PA decode kernel for sliding window.

    Grid = (batch_size, num_kv_heads * mtp_groups, max_context_partition_num).
    Each GPU block processes one 256-token partition selected from the visible KV
    region: the sliding tail window.
    sliding_window and max_context_partition_num are compile-time constants.
    """
    assert sliding_window > 0, "compile_pa_decode_sw requires sliding_window > 0"
    if query_input_dtype not in ("bf16", "f16"):
        raise ValueError("`compile_pa_decode_sw` only supports bf16/f16 query inputs.")
    if head_dim not in (64, 128):
        raise ValueError(f"`compile_pa_decode_sw` only supports head_dim 64 or 128, got {head_dim}.")
    fuse_partitions = max_context_partition_num <= 1
    _HEAD = head_dim
    _QKHELOOP = head_dim // QKHE_PER_FETCH
    _VHELOOP = head_dim // MFMA_N // NUM_WARPS
    _Q_LANES_PER_HEAD = head_dim // Q_ELEMS_PER_LANE
    _QUERY_DTYPE = fx.BFloat16 if query_input_dtype == "bf16" else fx.Float16
    if softmax_scale is None:
        softmax_scale = 1.0 / (head_dim**0.5)
    _softmax_scale = float(softmax_scale)
    _bs = KV_BLOCK_SIZE  # 1024
    _mtp_groups = _get_sw_mtp_group_count(query_length, query_group_size)

    LDS_VMAX_BYTES = NUM_WARPS * MFMA_N * 4 if const_expr(per_token_kv) else 0
    LDS_SOFTMAX_TOTAL = LDS_SOFTMAX_BYTES + LDS_VMAX_BYTES

    if per_token_kv:

        @fx.struct
        class SharedStorage:
            logits: fx.Array[fx.Int32, LDS_LOGITS_BYTES // 4, 16]
            softmax: fx.Array[fx.Float32, LDS_SOFTMAX_TOTAL // 4, 16]
            scale: fx.Array[fx.Float32, LDS_SCALE_BYTES // 4, 16]

    else:

        @fx.struct
        class SharedStorage:
            logits: fx.Array[fx.Int32, LDS_LOGITS_BYTES // 4, 16]
            softmax: fx.Array[fx.Float32, LDS_SOFTMAX_TOTAL // 4, 16]

    @flyc.kernel(known_block_size=(BLOCK_THREADS, 1, 1))
    def pa_decode_sw_kernel(
        # Raw-pointer kernargs: bare i64 data_ptr() (strides are explicit args).
        exp_sums_ptr: fx.Int64,  # [batch, kv_heads, max_parts, eqgs] f32
        max_logits_ptr: fx.Int64,  # [batch, kv_heads, max_parts, eqgs] f32
        tmp_out_ptr: fx.Int64,  # [batch, kv_heads, max_parts, eqgs, head_size] bf16
        out_ptr: fx.Int64,  # [batch, query_length, kv_heads, query_group_size, head_size] bf16
        query_ptr: fx.Int64,
        key_cache_ptr: fx.Int64,
        value_cache_ptr: fx.Int64,
        block_tables_ptr: fx.Int64,  # [batch, max_blocks_per_seq] i32
        context_lengths_ptr: fx.Int64,
        key_scale_ptr: fx.Int64,
        value_scale_ptr: fx.Int64,
        stride_q_seq: fx.Int32,
        stride_q_head: fx.Int32,
        stride_k_block: fx.Int32,
        stride_k_head: fx.Int32,
        stride_v_block: fx.Int32,
        stride_v_head: fx.Int32,
        stride_es_seq: fx.Int32,
        stride_es_head: fx.Int32,
        stride_es_part: fx.Int32,
        stride_to_seq: fx.Int32,
        stride_to_head: fx.Int32,
        stride_to_part: fx.Int32,
        stride_to_group: fx.Int32,
        stride_out_bs: fx.Int32,
        stride_out_len: fx.Int32,
        stride_out_kv_head: fx.Int32,
        stride_out_group_size: fx.Int32,
        stride_bt_seq: fx.Int32,
        stride_ks_block: fx.Int32,
        stride_ks_head: fx.Int32,
    ):
        tid = fx.Int32(gpu.thread_id("x"))
        batch_idx = fx.Int32(gpu.block_id("x"))
        grid_y = fx.Int32(gpu.block_id("y"))
        kv_h = udiv_const(grid_y, _mtp_groups)
        mtp_group_from_grid = urem_const(grid_y, _mtp_groups)
        partition_idx = fx.Int32(gpu.block_id("z"))
        lane16id = tid & fx.Int32(15)
        rowid = (tid >> fx.Int32(4)) & fx.Int32(3)
        warp_id = fx.Int32(tid >> fx.Int32(6))

        def _divide_addr(addr, dtype):
            pointer = _global_pointer_from_addr(addr, dtype, alignment=dtype.width // 8)
            flat = fx.make_view(pointer, fx.make_layout(_FLAT_BUFFER_ELEMENTS, 1))
            return fx.logical_divide(
                fx.rocdl.make_buffer_tensor(flat),
                fx.make_layout(1, 1),
            )

        q_tiles = _divide_addr(query_ptr, _QUERY_DTYPE)
        context_lengths = _divide_addr(context_lengths_ptr, fx.Int32)
        block_tables = _divide_addr(block_tables_ptr, fx.Int32)
        exp_sums = _divide_addr(exp_sums_ptr, fx.Float32)
        max_logits = _divide_addr(max_logits_ptr, fx.Float32)
        tmp_out_tiles = _divide_addr(tmp_out_ptr, fx.BFloat16)
        out_tiles = _divide_addr(out_ptr, fx.BFloat16)
        key_scales = _divide_addr(key_scale_ptr, fx.Float32)
        value_scales = _divide_addr(value_scale_ptr, fx.Float32)

        copy_i32 = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Int32)
        copy_f32 = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Float32)
        copy_q = fx.make_copy_atom(fx.rocdl.BufferCopy64b(), _QUERY_DTYPE)
        copy_bf16x4 = fx.make_copy_atom(fx.rocdl.BufferCopy64b(), fx.BFloat16)

        i32_register = fx.make_rmem_tensor(1, fx.Int32)
        f32_register = fx.make_rmem_tensor(1, fx.Float32)
        q_register = fx.make_rmem_tensor(4, _QUERY_DTYPE)
        bf16x4_register = fx.make_rmem_tensor(4, fx.BFloat16)

        k_global_ptr = _global_pointer_from_addr(key_cache_ptr, fx.Uint8, alignment=16)
        v_global_ptr = _global_pointer_from_addr(value_cache_ptr, fx.Uint8, alignment=16)
        global_copy_16b = fx.make_copy_atom(fx.UniversalCopy128b(), fx.Uint8)
        global_register_16b = fx.make_rmem_tensor(16, fx.Uint8)

        context_len = _copy_load(context_lengths, batch_idx, copy_i32, i32_register)[0]
        if const_expr(per_token_kv):
            k_scale_val = fx.Float32(1.0)
            v_scale_val = fx.Float32(1.0)
        else:
            k_scale_val = _copy_load(key_scales, 0, copy_f32, f32_register)[0]
            v_scale_val = _copy_load(value_scales, 0, copy_f32, f32_register)[0]

        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        logits_base = lds.logits.ptr
        softmax_base = lds.softmax.ptr
        scale_base = None
        if const_expr(per_token_kv):
            scale_base = lds.scale.ptr

        _softmax_scale_const = fx.Float32(_softmax_scale)
        NEG_INF = fx.Float32(float("-inf"))
        ZERO_F = fx.Float32(0.0)
        c_cps = fx.Int32(KV_COMPUTE_BLOCK)
        c_bs = fx.Int32(_bs)

        local_qhead_idx = warp_id * 4 + rowid
        _k_tok_thread_base, _c_tok_stride_dw, _k_he_off_dw = _build_pa_k_thread_invariants(
            warp_id,
            lane16id,
            rowid,
            qkhe_loop=_QKHELOOP,
        )

        # ── Context length and partition mapping ──
        # Visible tiles cover the union of all per-query sliding windows.

        num_tiles_for_seq = (context_len + c_cps - 1) >> fx.Int32(8)
        seq_start_global = context_len - query_length - sliding_window
        seq_start_global = arith.select(seq_start_global > 0, seq_start_global, 0)
        tail_start_tile = seq_start_global >> fx.Int32(8)
        visible_tile_count = num_tiles_for_seq - tail_start_tile
        tile_partition_idx_raw = tail_start_tile + partition_idx

        _is_valid = partition_idx < visible_tile_count

        _k_head_off = kv_h * stride_k_head
        _v_head_off = kv_h * stride_v_head

        (
            _load_kv_scale_scalars,
            _load_v_and_scales,
            _store_vmax_warp,
            _qk_and_intra_softmax,
            _cross_warp_softmax_and_prob_pack,
            _pv_mfma,
        ) = _make_pa_phase_helpers(
            trans_v=trans_v,
            per_token_kv=per_token_kv,
            kv_h=kv_h,
            v_global_ptr=v_global_ptr,
            v_copy_atom=global_copy_16b,
            v_register=global_register_16b,
            ks_tiles=key_scales,
            vs_tiles=value_scales,
            scale_copy_atom=copy_f32,
            scale_register=f32_register,
            logits_base=logits_base,
            softmax_base=softmax_base,
            scale_base=scale_base,
            stride_ks_block=stride_ks_block,
            stride_ks_head=stride_ks_head,
            softmax_scale_base=_softmax_scale_const,
            k_scale_val=k_scale_val,
            v_scale_val=v_scale_val,
            warp_id=warp_id,
            lane16id=lane16id,
            rowid=rowid,
            head_size=_HEAD,
        )

        def _process_block_split(
            rmax,
            rsum,
            outs,
            k_ops,
            preloaded_v_and_scales,
            q_frags,
            causal_bound,
            query_scale_lane,
            seq_start,
            partition_start,
        ):
            """Process one 256-token tile inside the selected physical block."""
            v0_ops = preloaded_v_and_scales
            d_out_0 = _qk_and_intra_softmax(
                k_ops,
                partition_start,
                q_frags,
                causal_bound,
                query_scale_lane=query_scale_lane,
                seq_start=seq_start,
            )
            gpu.barrier()
            rmax, rsum, outs, vc0 = _cross_warp_softmax_and_prob_pack(d_out_0, rmax, rsum, outs)
            gpu.barrier()
            outs = _pv_mfma(v0_ops, outs, vc0)
            return rmax, rsum, outs

        def _store_partition_results(eqgs_lane, running_sum, running_max, outelems_norm):
            for vhe in range_constexpr(_VHELOOP):
                hs_base = fx.Int32(vhe * NUM_WARPS * MFMA_N) + warp_id * fx.Int32(MFMA_N) + rowid * 4
                to_off = (
                    batch_idx * stride_to_seq
                    + kv_h * stride_to_head
                    + partition_idx * stride_to_part
                    + eqgs_lane * stride_to_group
                    + hs_base
                )
                out_bf16 = fx.Vector(outelems_norm[vhe]).to(fx.BFloat16)
                _copy_store(
                    tmp_out_tiles,
                    to_off,
                    copy_bf16x4,
                    bf16x4_register,
                    out_bf16,
                )

            es_off = batch_idx * stride_es_seq + kv_h * stride_es_head + partition_idx * stride_es_part + eqgs_lane
            _copy_store(
                exp_sums,
                es_off,
                copy_f32,
                f32_register,
                fx.Vector.from_elements([running_sum], dtype=fx.Float32),
            )
            _copy_store(
                max_logits,
                es_off,
                copy_f32,
                f32_register,
                fx.Vector.from_elements([running_max], dtype=fx.Float32),
            )

        def _store_group_results(qi_val, qhi_pos, running_sum, running_max, outs):
            outelems_norm = _normalize_pa_output(running_sum, outs)
            eqgs_lane = qi_val * fx.Int32(query_group_size) + qhi_pos
            _store_partition_results(eqgs_lane, running_sum, running_max, outelems_norm)

        def _store_fused_group_results(qi_val, qhi_pos, running_sum, outs):
            outelems_norm = _normalize_pa_output(running_sum, outs)
            for vhe in range_constexpr(_VHELOOP):
                hs_base = fx.Int32(vhe * NUM_WARPS * MFMA_N) + warp_id * fx.Int32(MFMA_N) + rowid * 4
                out_off = (
                    batch_idx * stride_out_bs
                    + qi_val * stride_out_len
                    + kv_h * stride_out_kv_head
                    + qhi_pos * stride_out_group_size
                    + hs_base
                )
                out_bf16 = fx.Vector(outelems_norm[vhe]).to(fx.BFloat16)
                _copy_store(
                    out_tiles,
                    out_off,
                    copy_bf16x4,
                    bf16x4_register,
                    out_bf16,
                )

        def _write_empty_partition():
            zero_output = [fx.Vector.filled(4, 0.0, fx.Float32) for _ in range_constexpr(_VHELOOP)]
            qi_val, qhi_pos, _, _ = _compute_sw_mtp_group_state(
                lane16id,
                local_qhead_idx,
                mtp_group_idx=mtp_group_from_grid,
                query_length=query_length,
                query_group_size=query_group_size,
            )
            eqgs_lane = qi_val * fx.Int32(query_group_size) + qhi_pos
            _store_partition_results(eqgs_lane, ZERO_F, NEG_INF, zero_output)

        def _run_valid_partition():
            def _get_tile_metadata(tile_partition_idx_value, tile_valid):
                safe_tile_partition_idx = (
                    arith.select(tile_valid, tile_partition_idx_value, fx.Int32(0))
                    if const_expr(fuse_partitions)
                    else tile_partition_idx_value
                )
                tile_context_len = (
                    arith.select(tile_valid, context_len, fx.Int32(0)) if const_expr(fuse_partitions) else context_len
                )
                tile_seq_partition_idx = safe_tile_partition_idx >> fx.Int32(2)
                tile_block_split_idx = safe_tile_partition_idx & fx.Int32(TILES_PER_BLOCK - 1)
                tile_token_offset_local = tile_block_split_idx * c_cps
                tile_kv_seq_start = tile_seq_partition_idx * c_bs + tile_token_offset_local
                tile_bt_off = batch_idx * stride_bt_seq + tile_seq_partition_idx
                tile_phys_block = _copy_load(block_tables, tile_bt_off, copy_i32, i32_register)[0]
                return tile_token_offset_local, tile_kv_seq_start, tile_context_len, tile_phys_block

            def _load_tile(tile_metadata, tile_scale_scalars):
                tile_token_offset_local, tile_kv_seq_start, tile_context_len, tile_phys_block = tile_metadata
                tile_k_base = _compute_block_base_dw_i64(tile_phys_block, stride_k_block, _k_head_off)

                tile_k_flat = _load_k_flat(
                    k_global_ptr,
                    global_copy_16b,
                    global_register_16b,
                    tile_k_base,
                    tile_token_offset_local,
                    _k_tok_thread_base,
                    _c_tok_stride_dw,
                    _k_he_off_dw,
                    qkhe_loop=_QKHELOOP,
                )

                tile_v_base = _compute_block_base_dw_i64(tile_phys_block, stride_v_block, _v_head_off)
                tile_v_ops = _load_v_and_scales(
                    tile_v_base,
                    tile_token_offset_local,
                    preloaded_scale_scalars=tile_scale_scalars,
                )
                _store_vmax_warp(tile_kv_seq_start, seq_end=tile_context_len)
                return (
                    unflatten_k(tile_k_flat, qkhe_loop=_QKHELOOP),
                    tile_v_ops,
                    tile_kv_seq_start,
                    tile_context_len,
                )

            mtp_prefetch = _prefetch_sw_mtp_group_query(
                q_tiles,
                copy_q,
                q_register,
                batch_idx,
                kv_h,
                stride_q_seq,
                stride_q_head,
                lane16id,
                local_qhead_idx,
                mtp_group_idx=mtp_group_from_grid,
                query_length=query_length,
                query_group_size=query_group_size,
                q_lanes_per_head=_Q_LANES_PER_HEAD,
            )
            tile_valid = fx.Int32(0) < visible_tile_count if const_expr(fuse_partitions) else True
            tile_partition_idx = tail_start_tile if const_expr(fuse_partitions) else tile_partition_idx_raw
            prefetched_tile_metadata = _get_tile_metadata(tile_partition_idx, tile_valid)
            prefetched_tile_scale_scalars = _load_kv_scale_scalars(
                prefetched_tile_metadata[0],
                prefetched_tile_metadata[3],
            )
            qi_val, qhi_pos, q_chunks = mtp_prefetch
            q_frags, query_scale_lane = _finish_q_fragments(
                logits_base,
                softmax_base,
                q_chunks,
                lane16id,
                rowid,
                local_qhead_idx,
                head_size=_HEAD,
                qkhe_loop=_QKHELOOP,
                q_lanes_per_head=_Q_LANES_PER_HEAD,
            )
            (
                tile_k_ops,
                tile_v_and_scales,
                tile_kv_seq_start,
                tile_context_len,
            ) = _load_tile(prefetched_tile_metadata, prefetched_tile_scale_scalars)
            attention_context_len = tile_context_len if const_expr(fuse_partitions) else context_len
            causal_bound = attention_context_len + fx.Int32(1 - query_length) + qi_val
            seq_start = attention_context_len - fx.Int32(query_length + sliding_window) + qi_val
            outs = [fx.Vector.filled(4, 0.0, fx.Float32) for _ in range_constexpr(_VHELOOP)]
            running_max, running_sum, outs = _process_block_split(
                NEG_INF,
                ZERO_F,
                outs,
                tile_k_ops,
                tile_v_and_scales,
                q_frags,
                causal_bound,
                query_scale_lane,
                seq_start,
                tile_kv_seq_start,
            )
            if const_expr(fuse_partitions):
                _store_fused_group_results(qi_val, qhi_pos, running_sum, outs)
            else:
                _store_group_results(qi_val, qhi_pos, running_sum, running_max, outs)

        if const_expr(fuse_partitions):
            _run_valid_partition()
        else:
            if _is_valid:
                _run_valid_partition()
            else:
                _write_empty_partition()

    @flyc.jit
    def launch_pa_decode_sw(
        es: fx.Int64,
        ml: fx.Int64,
        to: fx.Int64,
        out: fx.Int64,
        q: fx.Int64,
        kc: fx.Int64,
        vc: fx.Int64,
        bt: fx.Int64,
        cl: fx.Int64,
        ks: fx.Int64,
        vs: fx.Int64,
        s_q_seq: fx.Int32,
        s_q_head: fx.Int32,
        s_k_block: fx.Int32,
        s_k_head: fx.Int32,
        s_v_block: fx.Int32,
        s_v_head: fx.Int32,
        s_es_seq: fx.Int32,
        s_es_head: fx.Int32,
        s_es_part: fx.Int32,
        s_to_seq: fx.Int32,
        s_to_head: fx.Int32,
        s_to_part: fx.Int32,
        s_to_group: fx.Int32,
        s_out_bs: fx.Int32,
        s_out_len: fx.Int32,
        s_out_kv_head: fx.Int32,
        s_out_group_size: fx.Int32,
        s_bt_seq: fx.Int32,
        s_ks_block: fx.Int32,
        s_ks_head: fx.Int32,
        gx: fx.Int32,
        gy: fx.Int32,
        gz: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        pa_decode_sw_kernel(
            es,
            ml,
            to,
            out,
            q,
            kc,
            vc,
            bt,
            cl,
            ks,
            vs,
            s_q_seq,
            s_q_head,
            s_k_block,
            s_k_head,
            s_v_block,
            s_v_head,
            s_es_seq,
            s_es_head,
            s_es_part,
            s_to_seq,
            s_to_head,
            s_to_part,
            s_to_group,
            s_out_bs,
            s_out_len,
            s_out_kv_head,
            s_out_group_size,
            s_bt_seq,
            s_ks_block,
            s_ks_head,
        ).launch(grid=(gx, gy, gz), block=(BLOCK_THREADS, 1, 1), stream=stream)

    return {
        "launch": launch_pa_decode_sw,
        "kernel": pa_decode_sw_kernel,
    }
