# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 FlyDSL Project Contributors
# ruff: noqa: B023
# Nested @flyc.jit functions execute immediately inside compile-time loops.
"""Aligned common-row fusion for MegaMoE Stage2.

The Stage1 fanout planner lays the selected experts' common routes out in the
same order.  This kernel consumes those two contiguous sections, evaluates the
two experts sequentially with one accumulator set, and scatters their weighted
sum once.  The ordinary Stage2 kernel concurrently handles all residual rows.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import range_constexpr, rocdl
from flydsl.expr.typing import Int8, T
from flydsl.runtime.device import get_rocm_arch

from ..mxfp4_gemm_common import _fabs_f32 as fabs_f32
from ..mxfp4_gemm_common import lds_typed_ptr, lds_vec_load
from ..tensor_shim import (
    _preload_compiled,
    _run_compiled,
    buf_copy_store,
    ptr_buf_tensor,
)
from .gemm2 import (
    _resolve_g2_knobs,
    gemm2_compute_v2,
    issue_a_load_lds_dt,
    kStages,
)
from .mega_moe_stage2 import (
    _fp8_scale_for_leader,
    _stage2_lds_bytes,
)

_BUFFER_OFFSET_ABI_BYTES = 1 << 31
ALIGNED_PAIR_SCATTER_VEC = 16


def _store_accumulator_tile(lds_base, accm, wave, lane, *, BM, BN):
    """CShuffle one unweighted accumulator tile to an f32 LDS slab."""
    k_m_chunks = BM // 16
    num_acc_n = (BN // 4) // 16
    wave_n = BN // 4
    lane_div_16 = lane // 16
    lane_mod_16 = lane % 16
    output = lds_typed_ptr(lds_base, T.f32, align=4)
    for i in range_constexpr(k_m_chunks):
        row_base = fx.Int32(i * 16) + lane_div_16 * fx.Int32(4)
        for j in range_constexpr(num_acc_n):
            col = wave * fx.Int32(wave_n) + fx.Int32(j * 16) + lane_mod_16
            vec = fx.Vector(accm[i][j])
            for v in range_constexpr(4):
                output[(row_base + v) * fx.Int32(BN) + col] = fx.Float32(vec[v])
    fx.barrier()


# fmt: off
def _pair_routewise_half_scatter(lds_a, lds_b, n_block_idx, wave, lane, *,
    N_OUT, BM, BN, npes, topk, log2_max_tok, mask_max_tok, recv_cap,
    comb_inp_nbytes, lds_packed_a_off, lds_packed_b_off,
    lds_weight_a_off, lds_weight_b_off, lds_peer_off):
# fmt: on
    """Quantize/scatter A and B independently with one half-wave each."""
    token_nbytes = N_OUT + N_OUT // 32
    second_half = lane >= fx.Int32(32)
    half_lane = lane & fx.Int32(31)
    selected_lds = second_half.select(lds_b, lds_a)
    packed_off = second_half.select(
        fx.Int32(lds_packed_b_off), fx.Int32(lds_packed_a_off)
    )
    weight_off = second_half.select(
        fx.Int32(lds_weight_b_off), fx.Int32(lds_weight_a_off)
    )

    for row_iter in range_constexpr(BM // 4):
        row = wave + fx.Int32(row_iter * 4)
        row_byte_off = row * fx.Int32(4)
        packed = fx.Int32(
            fx.ptr_load(
                lds_typed_ptr(packed_off + row_byte_off, T.i32, align=4)
            )
        )
        weight = fx.Float32(
            fx.ptr_load(
                lds_typed_ptr(weight_off + row_byte_off, T.f32, align=4)
            )
        )
        token = packed & fx.Int32(0x00FFFFFF)
        slot = packed.shrui(fx.Int32(24)) & fx.Int32(0xFF)
        dest_pe = token >> fx.Int32(log2_max_tok)
        dest_lid = token & fx.Int32(mask_max_tok)
        valid = (
            (token < fx.Int32(recv_cap))
            & (slot < fx.Int32(topk))
            & (dest_pe < fx.Int32(npes))
        )
        safe_peer = valid.select(dest_pe, fx.Int32(0))
        peer_base = fx.Int64(
            fx.ptr_load(
                lds_typed_ptr(
                    fx.Int32(lds_peer_off) + safe_peer * fx.Int32(8),
                    T.i64,
                    align=8,
                )
            )
        )
        payload_buf = ptr_buf_tensor(
            peer_base,
            fx.Int32,
            unit_elems=ALIGNED_PAIR_SCATTER_VEC // 4,
            num_records_bytes=comb_inp_nbytes,
        )
        row_base = (dest_lid * fx.Int32(topk) + slot) * fx.Int32(
            token_nbytes
        )
        active = half_lane < fx.Int32(BN // ALIGNED_PAIR_SCATTER_VEC)
        col = active.select(half_lane * fx.Int32(ALIGNED_PAIR_SCATTER_VEC), fx.Int32(0))
        idx0 = row * fx.Int32(BN) + col
        values_raw = fx.Vector(
            lds_vec_load(
                selected_lds,
                idx0 * fx.Int32(4),
                fx.Vector.make_type(ALIGNED_PAIR_SCATTER_VEC, fx.Float32),
                fx.Float32,
                align=16,
            )
        )
        values = [
            fx.Float32(values_raw[q]) * fx.Float32(weight)
            for q in range_constexpr(ALIGNED_PAIR_SCATTER_VEC)
        ]
        local_max = fabs_f32(values[0])
        for q in range_constexpr(1, ALIGNED_PAIR_SCATTER_VEC):
            local_max = local_max.maximumf(fabs_f32(values[q]))
        max_bits = local_max.bitcast(fx.Int32)
        for xor_lane in (1, 2):
            if xor_lane < 32 // ALIGNED_PAIR_SCATTER_VEC:
                remote_bits = rocdl.ds_bpermute(
                    T.i32,
                    (lane ^ fx.Int32(xor_lane)) * fx.Int32(4),
                    max_bits,
                )
                local_max = local_max.maximumf(
                    fx.Int32(remote_bits).bitcast(fx.Float32)
                )
                max_bits = local_max.bitcast(fx.Int32)
        scale_group_lanes = 32 // ALIGNED_PAIR_SCATTER_VEC
        leader_lane = lane & fx.Int32(~(scale_group_lanes - 1))
        scale_leader = active & (
            (half_lane & fx.Int32(scale_group_lanes - 1)) == fx.Int32(0)
        )
        leader_e8m0 = _fp8_scale_for_leader(scale_leader, local_max)
        e8m0 = fx.Int32(
            rocdl.ds_bpermute(
                T.i32,
                leader_lane * fx.Int32(4),
                leader_e8m0,
            )
        )
        block_scale = (e8m0 << fx.Int32(23)).bitcast(fx.Float32)
        packed_ty = T.vec(2, T.i16)
        packed_words = []
        for word in range_constexpr(ALIGNED_PAIR_SCATTER_VEC // 4):
            packed_word = fx.Vector.filled(2, 0, fx.Int16).ir_value()
            for pair in range_constexpr(2):
                value = word * 4 + pair * 2
                packed_word = rocdl.cvt_scalef32_pk_fp8_f32(
                    packed_ty,
                    packed_word,
                    values[value].ir_value(),
                    values[value + 1].ir_value(),
                    block_scale.ir_value(),
                    pair,
                )
            packed_words.append(fx.Vector(packed_word).bitcast(fx.Int32)[0])
        payload = fx.Vector.from_elements(packed_words, fx.Int32)
        payload_off = (valid & active).select(
            row_base + n_block_idx * fx.Int32(BN) + col,
            fx.Int32(comb_inp_nbytes),
        )
        buf_copy_store(
            payload_buf,
            payload_off // fx.Int32(ALIGNED_PAIR_SCATTER_VEC),
            payload,
            fx.Int32,
            unit_elems=ALIGNED_PAIR_SCATTER_VEC // 4,
            cache_modifier=2,
        )

        @flyc.jit
        def store_scale_if_leader():
            if scale_leader:
                scale_off = valid.select(
                    row_base
                    + fx.Int32(N_OUT)
                    + n_block_idx * fx.Int32(BN // 32)
                    + half_lane // fx.Int32(scale_group_lanes),
                    fx.Int32(comb_inp_nbytes),
                )
                scale_buf = ptr_buf_tensor(
                    peer_base,
                    fx.Int8,
                    num_records_bytes=comb_inp_nbytes,
                )
                buf_copy_store(
                    scale_buf,
                    scale_off,
                    e8m0.to(fx.Int8),
                    fx.Int8,
                    cache_modifier=2,
                )

        store_scale_if_leader()


# fmt: off
def compile_mega_moe_stage2_aligned_pair(*, model_dim: int, inter_dim: int,
    experts: int, topk: int, rank: int, npes: int, max_tok: int,
    recv_cap: int, comb_inp_nbytes: int, BM: int = 32,
    SBM: int = 128, BN: int = 256, BK: int = 256, INTER_MAX: int = 8192,
    use_nt: bool = False, cu_num: int = 722, g2_bhoist=True,
    g2_ascale_pf=True):
# fmt: on
    """Compile the production two-expert common-row Stage2 kernel."""
    arch = str(get_rocm_arch() or "")
    if not arch.startswith("gfx95"):
        raise RuntimeError(
            f"MegaMoE aligned-pair Stage2 requires CDNA4, got {arch or 'unknown'}"
        )
    if BM not in (16, 32, 64) or BN not in (128, 256, 512) or BK != 256:
        raise ValueError(
            "aligned-pair Stage2 requires BM16/32/64, BN128/256/512, and BK256"
        )
    if SBM % BM:
        raise ValueError("Stage1 SBM must be a multiple of pair BM")
    if model_dim % BN or inter_dim % BK:
        raise ValueError("model/inter dimensions must exactly tile BN/BK")
    if not 0 < comb_inp_nbytes < _BUFFER_OFFSET_ABI_BYTES:
        raise ValueError("aligned-pair P2P output exceeds the 32-bit buffer ABI")
    if BN % ALIGNED_PAIR_SCATTER_VEC:
        raise ValueError(
            f"aligned-pair BN must be divisible by {ALIGNED_PAIR_SCATTER_VEC}"
        )
    a_stages = kStages + 1
    compute_lds_bytes = _stage2_lds_bytes(BM, BN, BK, "fp8", a_stages)
    second_compute_off = compute_lds_bytes
    lds_packed_a_off = compute_lds_bytes * 2
    lds_packed_b_off = lds_packed_a_off + BM * 4
    lds_weight_a_off = lds_packed_b_off + BM * 4
    lds_weight_b_off = lds_weight_a_off + BM * 4
    lds_peer_off = lds_weight_b_off + BM * 4
    lds_bytes = lds_peer_off + npes * 8
    if lds_bytes > 160 * 1024:
        raise ValueError(
            f"aligned-pair LDS use {lds_bytes} exceeds 160 KiB"
        )

    log2_max_tok = max_tok.bit_length() - 1
    if max_tok <= 0 or max_tok & (max_tok - 1):
        raise ValueError("max_tok must be a positive power of two")
    mask_max_tok = max_tok - 1
    expert_offset = rank * experts
    total_experts = npes * experts
    total_segments = total_experts + npes
    k_bytes = inter_dim
    kh_tile_a = BK
    g2_bhoist, g2_ascale_pf, _, _, _, _ = _resolve_g2_knobs(
        g2_bhoist,
        g2_ascale_pf,
        0,
        False,
        False,
    )

    @fx.struct
    class SharedStorage:
        buf: fx.Array[Int8, lds_bytes, 16]

    kernel_name = (
        f"megamoe_stage2_aligned_pair_runtime_t{BM}x{BN}x{BK}_cu{cu_num}"
        f"_seqacc4_sv{ALIGNED_PAIR_SCATTER_VEC}_ms1"
        f"_nt{int(use_nt)}_bh{int(g2_bhoist)}apf{int(g2_ascale_pf)}"
        "_rtv3_rq2_hw1"
    )

    # fmt: off
    @flyc.kernel(name=kernel_name, known_block_size=[256, 1, 1])
    def kernel(arg_aq: fx.Int64, arg_ascale: fx.Int64, arg_bq: fx.Int64,
        arg_bscale: fx.Int64, arg_stids: fx.Int64, arg_sweights: fx.Int64,
        arg_expert_tile_end: fx.Int64, arg_count_matrix: fx.Int64,
        arg_pair_config: fx.Int64, arg_parity: fx.Int64,
        arg_p2p_comb_inp: fx.Int64, i32_max_m_blocks: fx.Int32,
        i32_inter: fx.Int32, i32_hidden: fx.Int32):
    # fmt: on
        tx = fx.thread_idx.x
        bx = fx.block_idx.x
        lane = tx % fx.Int32(64)
        wave = rocdl.readfirstlane(T.i32, tx // fx.Int32(64))
        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        lds_compute = fx.Int32(fx.ptrtoint(lds.buf.ptr))
        lds_compute_b = lds_compute + fx.Int32(second_compute_off)

        expert_tile_end = ptr_buf_tensor(arg_expert_tile_end, fx.Int32)
        count_matrix = ptr_buf_tensor(arg_count_matrix, fx.Int32)
        pair_config = ptr_buf_tensor(arg_pair_config, fx.Int32)
        parity_buf = ptr_buf_tensor(arg_parity, fx.Int32)
        active_parity = parity_buf[fx.Int32(0)]
        packed_pair = pair_config[
            active_parity * fx.Int32(npes) + fx.Int32(rank)
        ]
        fx.rocdl.s_waitcnt(0)
        # Every lane loads the same entry; broadcasting a lane-0 SSA value
        # after divergent control flow can select the wrong experts.
        pair_a_lane = packed_pair & fx.Int32(0xFF)
        pair_b_lane = packed_pair.shrui(fx.Int32(8)) & fx.Int32(0xFF)
        pair_enabled_lane = (
            (packed_pair & fx.Int32(1 << 16)) != fx.Int32(0)
        ).select(fx.Int32(1), fx.Int32(0))
        pair_a_rt = fx.Int32(rocdl.readfirstlane(T.i32, pair_a_lane))
        pair_b_rt = fx.Int32(rocdl.readfirstlane(T.i32, pair_b_lane))
        pair_enabled = fx.Int32(
            rocdl.readfirstlane(T.i32, pair_enabled_lane)
        )
        global_a = fx.Int32(expert_offset) + pair_a_rt
        global_b = fx.Int32(expert_offset) + pair_b_rt
        group_a_lane = fx.Int32(0)
        group_b_lane = fx.Int32(0)
        group_rows_lane = fx.Int32(0)
        if lane == fx.Int32(0):
            safe_prev_a = (pair_a_rt > fx.Int32(0)).select(
                pair_a_rt - fx.Int32(1), fx.Int32(0)
            )
            safe_prev_b = (pair_b_rt > fx.Int32(0)).select(
                pair_b_rt - fx.Int32(1), fx.Int32(0)
            )
            prev_a = expert_tile_end[safe_prev_a] * fx.Int32(SBM)
            prev_b = expert_tile_end[safe_prev_b] * fx.Int32(SBM)
            group_a_lane = (pair_a_rt > fx.Int32(0)).select(
                prev_a, fx.Int32(0)
            )
            group_b_lane = (pair_b_rt > fx.Int32(0)).select(
                prev_b, fx.Int32(0)
            )
            group_count = fx.Int32(0)
            group_column = fx.Int32(total_experts + rank)
            for source in range_constexpr(npes):
                group_count = group_count + count_matrix[
                    fx.Int32(source * total_segments) + group_column
                ]
            group_rows_lane = pair_enabled.select((
                (group_count + fx.Int32(SBM - 1)) // fx.Int32(SBM)
            ) * fx.Int32(SBM), fx.Int32(0))
        group_a = fx.Int32(rocdl.readfirstlane(T.i32, group_a_lane))
        group_b = fx.Int32(rocdl.readfirstlane(T.i32, group_b_lane))
        group_rows = fx.Int32(rocdl.readfirstlane(T.i32, group_rows_lane))
        total_m_blocks = group_rows // fx.Int32(BM)
        peer_table = ptr_buf_tensor(arg_p2p_comb_inp, fx.Int64)
        if tx < fx.Int32(npes):
            peer = peer_table[tx]
            fx.ptr_store(
                peer,
                lds_typed_ptr(
                    fx.Int32(lds_peer_off) + tx * fx.Int32(8),
                    T.i64,
                    align=8,
                ),
            )
        fx.barrier()

        n_block = bx // fx.Int32(cu_num)
        m_slot = bx - n_block * fx.Int32(cu_num)
        diff = total_m_blocks - m_slot
        remaining = (diff > fx.Int32(0)).select(diff, fx.Int32(0))
        iterations = (
            remaining + fx.Int32(cu_num - 1)
        ) // fx.Int32(cu_num)
        stids = ptr_buf_tensor(arg_stids, fx.Int32)
        sweights = ptr_buf_tensor(arg_sweights, fx.Float32)

        def issue_all_a_loads(m_row, lds_base):
            for slot in range_constexpr(kStages):
                issue_a_load_lds_dt(
                    arg_aq,
                    lds_base,
                    slot,
                    slot,
                    m_row,
                    wave,
                    lane,
                    True,
                    kh_tile_a,
                    fx.Int32(k_bytes),
                    BM=BM,
                )

        def load_metadata(m_row_a, m_row_b):
            if tx < fx.Int32(BM):
                packed_a = stids[m_row_a + tx]
                packed_b = stids[m_row_b + tx]
                weight_a = sweights[m_row_a + tx]
                weight_b = sweights[m_row_b + tx]
                fx.ptr_store(
                    packed_a,
                    lds_typed_ptr(
                        fx.Int32(lds_packed_a_off) + tx * fx.Int32(4),
                        T.i32,
                        align=4,
                    ),
                )
                fx.ptr_store(
                    packed_b,
                    lds_typed_ptr(
                        fx.Int32(lds_packed_b_off) + tx * fx.Int32(4),
                        T.i32,
                        align=4,
                    ),
                )
                fx.ptr_store(
                    weight_a,
                    lds_typed_ptr(
                        fx.Int32(lds_weight_a_off) + tx * fx.Int32(4),
                        T.f32,
                        align=4,
                    ),
                )
                fx.ptr_store(
                    weight_b,
                    lds_typed_ptr(
                        fx.Int32(lds_weight_b_off) + tx * fx.Int32(4),
                        T.f32,
                        align=4,
                    ),
                )

        def run_pair(m_block):
            m_row_a = group_a + m_block * fx.Int32(BM)
            m_row_b = group_b + m_block * fx.Int32(BM)
            fx.barrier()
            load_metadata(m_row_a, m_row_b)
            issue_all_a_loads(m_row_a, lds_compute)
            rocdl.sched_barrier(0)
            # fmt: off
            acc_a, _, _, _ = gemm2_compute_v2(
                lds_compute, arg_ascale, arg_bq, arg_bscale, fx.Int64(0), arg_aq,
                i32_max_m_blocks, fx.Int32(0), lane, wave, i32_inter, i32_hidden,
                fx.Int32(0), fx.Int32(0), BM=BM, BN=BN, BK=BK, use_nt=use_nt,
                INTER_MAX=INTER_MAX, aStages=a_stages, a_dtype="fp8", SBM=BM,
                g2_bhoist=g2_bhoist, g2_ascale_pf=g2_ascale_pf,
                expert_offset=expert_offset, explicit_m_row=m_row_a,
                explicit_n_block=n_block, explicit_expert=global_a)
            # fmt: on

            _store_accumulator_tile(
                lds_compute, acc_a, wave, lane, BM=BM, BN=BN
            )
            issue_all_a_loads(m_row_b, lds_compute_b)
            rocdl.sched_barrier(0)
            # fmt: off
            acc_b, _, _, _ = gemm2_compute_v2(
                lds_compute_b, arg_ascale, arg_bq, arg_bscale, fx.Int64(0), arg_aq,
                i32_max_m_blocks, fx.Int32(0), lane, wave, i32_inter, i32_hidden,
                fx.Int32(0), fx.Int32(0), BM=BM, BN=BN, BK=BK, use_nt=use_nt,
                INTER_MAX=INTER_MAX, aStages=a_stages, a_dtype="fp8", SBM=BM,
                g2_bhoist=g2_bhoist, g2_ascale_pf=g2_ascale_pf,
                expert_offset=expert_offset, explicit_m_row=m_row_b,
                explicit_n_block=n_block, explicit_expert=global_b)
            _store_accumulator_tile(
                lds_compute_b, acc_b, wave, lane, BM=BM, BN=BN
            )
            _pair_routewise_half_scatter(
                lds_compute, lds_compute_b, n_block, wave, lane,
                N_OUT=model_dim, BM=BM, BN=BN, npes=npes, topk=topk,
                log2_max_tok=log2_max_tok, mask_max_tok=mask_max_tok,
                recv_cap=recv_cap, comb_inp_nbytes=comb_inp_nbytes,
                lds_packed_a_off=lds_packed_a_off,
                lds_packed_b_off=lds_packed_b_off,
                lds_weight_a_off=lds_weight_a_off,
                lds_weight_b_off=lds_weight_b_off,
                lds_peer_off=lds_peer_off,
            )
            # fmt: on

        def swizzle_pair_block(m_block):
            # Multiplication modulo M is a permutation whenever the factor is
            # coprime to M. Pick a small prime which does not divide the
            # runtime group size; this spreads initially resident CTAs across
            # all source-rank row bands without changing coverage.
            factor = (total_m_blocks % fx.Int32(17) != fx.Int32(0)).select(
                fx.Int32(17),
                (total_m_blocks % fx.Int32(13) != fx.Int32(0)).select(
                    fx.Int32(13),
                    (total_m_blocks % fx.Int32(11) != fx.Int32(0)).select(
                        fx.Int32(11),
                        (total_m_blocks % fx.Int32(7) != fx.Int32(0)).select(
                            fx.Int32(7),
                            (total_m_blocks % fx.Int32(5) != fx.Int32(0)).select(
                                fx.Int32(5), fx.Int32(1)
                            ),
                        ),
                    ),
                ),
            )
            return (m_block * factor) % total_m_blocks

        for iteration in range(fx.Int32(0), iterations, fx.Int32(1)):
            m_block = m_slot + iteration * fx.Int32(cu_num)
            run_pair(swizzle_pair_block(m_block))

    # fmt: off
    @flyc.jit
    def launch(arg_aq: fx.Int64, arg_ascale: fx.Int64, arg_bq: fx.Int64,
        arg_bscale: fx.Int64, arg_stids: fx.Int64, arg_sweights: fx.Int64,
        arg_expert_tile_end: fx.Int64, arg_count_matrix: fx.Int64,
        arg_pair_config: fx.Int64, arg_parity: fx.Int64,
        arg_p2p_comb_inp: fx.Int64, i32_max_m_blocks: fx.Int32,
        i32_inter: fx.Int32, i32_hidden: fx.Int32, stream: fx.Stream):
    # fmt: on
        grid_x = fx.Int32(cu_num) * (i32_hidden // fx.Int32(BN))
        kernel(
            arg_aq,
            arg_ascale,
            arg_bq,
            arg_bscale,
            arg_stids,
            arg_sweights,
            arg_expert_tile_end,
            arg_count_matrix,
            arg_pair_config,
            arg_parity,
            arg_p2p_comb_inp,
            i32_max_m_blocks,
            i32_inter,
            i32_hidden,
        ).launch(
            grid=(grid_x, 1, 1),
            block=(256, 1, 1),
            stream=stream,
        )

    return launch


_PAIR_LAUNCH_CACHE = {}


# fmt: off
def run_mega_moe_stage2_aligned_pair(arg_aq, arg_ascale, arg_bq, arg_bscale,
    arg_stids, arg_sweights, arg_expert_tile_end, arg_count_matrix,
    arg_pair_config, arg_parity, arg_p2p, row_capacity,
    i32_inter, i32_hidden, stream, **compile_kw):
# fmt: on
    """Compile/cache and launch the aligned common-row Stage2 kernel."""
    key = tuple(sorted(compile_kw.items()))
    launch = _PAIR_LAUNCH_CACHE.get(key)
    if launch is None:
        launch = compile_mega_moe_stage2_aligned_pair(**compile_kw)
        _PAIR_LAUNCH_CACHE[key] = launch
    bm = int(compile_kw.get("BM", 64))
    max_m_blocks = (int(row_capacity) + bm - 1) // bm
    _run_compiled(
        launch,
        arg_aq,
        arg_ascale,
        arg_bq,
        arg_bscale,
        arg_stids,
        arg_sweights,
        arg_expert_tile_end,
        arg_count_matrix,
        arg_pair_config,
        arg_parity,
        arg_p2p,
        fx.Int32(max_m_blocks),
        fx.Int32(i32_inter),
        fx.Int32(i32_hidden),
        stream,
    )


# fmt: off
def preload_mega_moe_stage2_aligned_pair(arg_aq, arg_ascale, arg_bq, arg_bscale,
    arg_stids, arg_sweights, arg_expert_tile_end, arg_count_matrix,
    arg_pair_config, arg_parity, arg_p2p, row_capacity,
    i32_inter, i32_hidden, stream, **compile_kw):
# fmt: on
    """Compile and load the aligned-pair Stage2 kernel without dispatching it."""
    key = tuple(sorted(compile_kw.items()))
    launch = _PAIR_LAUNCH_CACHE.get(key)
    if launch is None:
        launch = compile_mega_moe_stage2_aligned_pair(**compile_kw)
        _PAIR_LAUNCH_CACHE[key] = launch
    bm = int(compile_kw.get("BM", 64))
    max_m_blocks = (int(row_capacity) + bm - 1) // bm
    return _preload_compiled(
        launch,
        arg_aq,
        arg_ascale,
        arg_bq,
        arg_bscale,
        arg_stids,
        arg_sweights,
        arg_expert_tile_end,
        arg_count_matrix,
        arg_pair_config,
        arg_parity,
        arg_p2p,
        fx.Int32(max_m_blocks),
        fx.Int32(i32_inter),
        fx.Int32(i32_hidden),
        stream,
    )
