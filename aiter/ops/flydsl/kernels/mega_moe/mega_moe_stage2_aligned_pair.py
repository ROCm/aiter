# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 FlyDSL Project Contributors
# ruff: noqa: B023
# Nested @flyc.jit functions execute immediately inside compile-time loops.
"""Aligned common-row fusion prototype for MegaMoE Stage2.

The Stage1 fanout planner lays the selected experts' common routes out in the
same order.  This kernel consumes those two contiguous sections, evaluates the
two experts sequentially with one accumulator set, and scatters their weighted
sum once.  It is intentionally isolated from the normal Stage2 path while the
performance and numerical gates are evaluated.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, range_constexpr, rocdl
from flydsl.expr.typing import Int8, T
from flydsl.runtime.device import get_rocm_arch

from aiter.ops.flydsl.kernels import buffer_ops

from ..mxfp4_gemm_common import _fabs_f32 as fabs_f32
from ..mxfp4_gemm_common import lds_typed_ptr, lds_vec_load
from ..tensor_shim import _run_compiled
from .gemm2 import (
    _resolve_g2_knobs,
    gemm2_compute_v2,
    issue_a_load_lds_dt,
    kStages,
)
from .mega_moe_stage2 import (
    _fp8_scale_for_leader,
    _stage2_lds_bytes,
    p2p_scatter_epilog,
)

_BUFFER_OFFSET_ABI_BYTES = 1 << 31


def _store_weighted_accumulator(
    lds_output_base,
    accm,
    lds_weight_off,
    wave,
    lane,
    *,
    BM,
    BN,
    store_bf16=False,
):
    """CShuffle one accumulator tile to LDS after applying route weights."""
    k_m_chunks = BM // 16
    num_acc_n = (BN // 4) // 16
    wave_n = BN // 4
    lane_div_16 = lane // 16
    lane_mod_16 = lane % 16
    output = lds_typed_ptr(
        lds_output_base,
        T.bf16 if store_bf16 else T.f32,
        align=2 if store_bf16 else 4,
    )

    for i in range_constexpr(k_m_chunks):
        row_base = fx.Int32(i * 16) + lane_div_16 * fx.Int32(4)
        weights = [
            fx.ptr_load(
                lds_typed_ptr(
                    fx.Int32(lds_weight_off) + (row_base + v) * fx.Int32(4),
                    T.f32,
                    align=4,
                )
            )
            for v in range_constexpr(4)
        ]
        for j in range_constexpr(num_acc_n):
            col = wave * fx.Int32(wave_n) + fx.Int32(j * 16) + lane_mod_16
            vec = fx.Vector(accm[i][j])
            for v in range_constexpr(4):
                value = fx.Float32(vec[v]) * fx.Float32(weights[v])
                output[(row_base + v) * fx.Int32(BN) + col] = (
                    fx.BFloat16(value) if const_expr(store_bf16) else value
                )


def _store_weighted_pair_accumulator(
    lds_output_base,
    acc_a,
    acc_b,
    lds_weight_a_off,
    lds_weight_b_off,
    wave,
    lane,
    *,
    BM,
    BN,
):
    """CShuffle the weighted sum of two accumulator tiles directly to LDS."""
    k_m_chunks = BM // 16
    num_acc_n = (BN // 4) // 16
    wave_n = BN // 4
    lane_div_16 = lane // 16
    lane_mod_16 = lane % 16
    output = lds_typed_ptr(lds_output_base, T.f32, align=4)

    for i in range_constexpr(k_m_chunks):
        row_base = fx.Int32(i * 16) + lane_div_16 * fx.Int32(4)
        weights_a = [
            fx.ptr_load(
                lds_typed_ptr(
                    fx.Int32(lds_weight_a_off) + (row_base + v) * fx.Int32(4),
                    T.f32,
                    align=4,
                )
            )
            for v in range_constexpr(4)
        ]
        weights_b = [
            fx.ptr_load(
                lds_typed_ptr(
                    fx.Int32(lds_weight_b_off) + (row_base + v) * fx.Int32(4),
                    T.f32,
                    align=4,
                )
            )
            for v in range_constexpr(4)
        ]
        for j in range_constexpr(num_acc_n):
            col = wave * fx.Int32(wave_n) + fx.Int32(j * 16) + lane_mod_16
            vec_a = fx.Vector(acc_a[i][j])
            vec_b = fx.Vector(acc_b[i][j])
            for v in range_constexpr(4):
                value = fx.Float32(vec_a[v]) * fx.Float32(weights_a[v]) + fx.Float32(
                    vec_b[v]
                ) * fx.Float32(weights_b[v])
                output[(row_base + v) * fx.Int32(BN) + col] = value


# fmt: off
def _pair_scatter_epilog(lds_compute_base, lds_first_base, acc_second, n_block_idx, wave, lane, *,
    N_OUT, BM, BN, npes, topk, log2_max_tok, mask_max_tok, recv_cap,
    comb_inp_nbytes, lds_packed_a_off, lds_packed_b_off, lds_weight_b_off,
    lds_peer_off, first_bf16=False, combined_in_lds=False,
    second_in_lds=False, scatter_active=None, scatter_vec=8):
# fmt: on
    """Add the aligned weighted rows, quantize once, and scatter one route."""
    if const_expr(not combined_in_lds and not second_in_lds):
        _store_weighted_accumulator(
            lds_compute_base,
            acc_second,
            lds_weight_b_off,
            wave,
            lane,
            BM=BM,
            BN=BN,
        )
    fx.barrier()

    token_nbytes = N_OUT + N_OUT // 32

    for row_iter in range_constexpr(BM // 4):
        row = wave + fx.Int32(row_iter * 4)
        row_byte_off = row * fx.Int32(4)
        packed_a = fx.ptr_load(
            lds_typed_ptr(
                fx.Int32(lds_packed_a_off) + row_byte_off,
                T.i32,
                align=4,
            )
        )
        packed_b = fx.ptr_load(
            lds_typed_ptr(
                fx.Int32(lds_packed_b_off) + row_byte_off,
                T.i32,
                align=4,
            )
        )
        packed_a = fx.Int32(rocdl.readfirstlane(T.i32, packed_a.ir_value()))
        packed_b = fx.Int32(rocdl.readfirstlane(T.i32, packed_b.ir_value()))
        token_a = packed_a & fx.Int32(0x00FFFFFF)
        token_b = packed_b & fx.Int32(0x00FFFFFF)
        slot_a = packed_a.shrui(fx.Int32(24)) & fx.Int32(0xFF)
        slot_b = packed_b.shrui(fx.Int32(24)) & fx.Int32(0xFF)
        dest_pe = token_a >> fx.Int32(log2_max_tok)
        dest_lid = token_a & fx.Int32(mask_max_tok)
        valid = (
            (token_a == token_b)
            & (token_a < fx.Int32(recv_cap))
            & (slot_a < fx.Int32(topk))
            & (slot_b < fx.Int32(topk))
            & (dest_pe < fx.Int32(npes))
        )
        if const_expr(scatter_active is not None):
            valid = valid & scatter_active
        safe_peer = valid.select(dest_pe, fx.Int32(0))
        peer_base = fx.ptr_load(
            lds_typed_ptr(
                fx.Int32(lds_peer_off) + safe_peer * fx.Int32(8),
                T.i64,
                align=8,
            )
        )
        peer_base = rocdl.readfirstlane(T.i64, peer_base.ir_value())
        destination = buffer_ops.create_buffer_resource_from_addr(
            peer_base,
            num_records_bytes=comb_inp_nbytes,
        )

        rep_row_base = (dest_lid * fx.Int32(topk) + slot_a) * fx.Int32(
            token_nbytes
        )
        secondary_row_base = (
            dest_lid * fx.Int32(topk) + slot_b
        ) * fx.Int32(token_nbytes)
        active = lane < fx.Int32(BN // scatter_vec)
        if const_expr(scatter_active is not None):
            active = active & scatter_active
        col = active.select(lane * fx.Int32(scatter_vec), fx.Int32(0))
        idx0 = row * fx.Int32(BN) + col
        if const_expr(combined_in_lds):
            first_v = None
        elif const_expr(first_bf16):
            first_v = fx.Vector(
                lds_vec_load(
                    lds_first_base,
                    idx0 * fx.Int32(2),
                    fx.Vector.make_type(scatter_vec, fx.BFloat16),
                    fx.BFloat16,
                    align=16,
                )
            )
        else:
            first_v = fx.Vector(
                lds_vec_load(
                    lds_first_base,
                    idx0 * fx.Int32(4),
                    fx.Vector.make_type(scatter_vec, fx.Float32),
                    fx.Float32,
                    align=16,
                )
            )
        second_v = fx.Vector(
            lds_vec_load(
                lds_compute_base,
                idx0 * fx.Int32(4),
                fx.Vector.make_type(scatter_vec, fx.Float32),
                fx.Float32,
                align=16,
            )
        )
        if const_expr(combined_in_lds):
            values = [
                fx.Float32(second_v[q]) for q in range_constexpr(scatter_vec)
            ]
        else:
            values = [
                fx.Float32(first_v[q]) + fx.Float32(second_v[q])
                for q in range_constexpr(scatter_vec)
            ]
        local_max = fabs_f32(values[0])
        for q in range_constexpr(1, scatter_vec):
            local_max = local_max.maximumf(fabs_f32(values[q]))
        max_bits = local_max.bitcast(fx.Int32)
        for xor_lane in (1, 2):
            if xor_lane < 32 // scatter_vec:
                remote_bits = rocdl.ds_bpermute(
                    T.i32,
                    (lane ^ fx.Int32(xor_lane)) * fx.Int32(4),
                    max_bits,
                )
                local_max = local_max.maximumf(
                    fx.Int32(remote_bits).bitcast(fx.Float32)
                )
                max_bits = local_max.bitcast(fx.Int32)
        scale_group_lanes = 32 // scatter_vec
        leader_lane = lane & fx.Int32(~(scale_group_lanes - 1))
        scale_leader = active & (
            (lane & fx.Int32(scale_group_lanes - 1)) == fx.Int32(0)
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
        for word in range_constexpr(scatter_vec // 4):
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
            packed_words.append(
                fx.Vector(packed_word).bitcast(fx.Int32)[0]
            )
        payload = fx.Vector.from_elements(
            packed_words,
            fx.Int32,
        )
        payload_off = (valid & active).select(
            rep_row_base + n_block_idx * fx.Int32(BN) + col,
            fx.Int32(comb_inp_nbytes),
        )
        buffer_ops.buffer_store(
            payload.ir_value(),
            destination,
            payload_off,
            offset_is_bytes=True,
            cache_modifier=2,
        )

        @flyc.jit
        def store_scales():
            if scale_leader:
                scale_col = lane // fx.Int32(scale_group_lanes)
                rep_scale_off = valid.select(
                    rep_row_base
                    + fx.Int32(N_OUT)
                    + n_block_idx * fx.Int32(BN // 32)
                    + scale_col,
                    fx.Int32(comb_inp_nbytes),
                )
                secondary_scale_off = valid.select(
                    secondary_row_base
                    + fx.Int32(N_OUT)
                    + n_block_idx * fx.Int32(BN // 32)
                    + scale_col,
                    fx.Int32(comb_inp_nbytes),
                )
                buffer_ops.buffer_store(
                    e8m0.to(fx.Int8),
                    destination,
                    rep_scale_off,
                    offset_is_bytes=True,
                    cache_modifier=2,
                )
                buffer_ops.buffer_store(
                    fx.Int8(0),
                    destination,
                    secondary_scale_off,
                    offset_is_bytes=True,
                    cache_modifier=2,
                )

        store_scales()


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
    lds_weight_a_off, lds_weight_b_off, lds_peer_off, scatter_vec=16):
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
        destination = buffer_ops.create_buffer_resource_from_addr(
            peer_base,
            num_records_bytes=comb_inp_nbytes,
        )
        row_base = (dest_lid * fx.Int32(topk) + slot) * fx.Int32(
            token_nbytes
        )
        active = half_lane < fx.Int32(BN // scatter_vec)
        col = active.select(half_lane * fx.Int32(scatter_vec), fx.Int32(0))
        idx0 = row * fx.Int32(BN) + col
        values_raw = fx.Vector(
            lds_vec_load(
                selected_lds,
                idx0 * fx.Int32(4),
                fx.Vector.make_type(scatter_vec, fx.Float32),
                fx.Float32,
                align=16,
            )
        )
        values = [
            fx.Float32(values_raw[q]) * fx.Float32(weight)
            for q in range_constexpr(scatter_vec)
        ]
        local_max = fabs_f32(values[0])
        for q in range_constexpr(1, scatter_vec):
            local_max = local_max.maximumf(fabs_f32(values[q]))
        max_bits = local_max.bitcast(fx.Int32)
        for xor_lane in (1, 2):
            if xor_lane < 32 // scatter_vec:
                remote_bits = rocdl.ds_bpermute(
                    T.i32,
                    (lane ^ fx.Int32(xor_lane)) * fx.Int32(4),
                    max_bits,
                )
                local_max = local_max.maximumf(
                    fx.Int32(remote_bits).bitcast(fx.Float32)
                )
                max_bits = local_max.bitcast(fx.Int32)
        scale_group_lanes = 32 // scatter_vec
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
        for word in range_constexpr(scatter_vec // 4):
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
        buffer_ops.buffer_store(
            payload.ir_value(),
            destination,
            payload_off,
            offset_is_bytes=True,
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
                buffer_ops.buffer_store(
                    e8m0.to(fx.Int8),
                    destination,
                    scale_off,
                    offset_is_bytes=True,
                    cache_modifier=2,
                )

        store_scale_if_leader()


# fmt: off
def compile_mega_moe_stage2_aligned_pair(*, model_dim: int, inter_dim: int,
    experts: int, topk: int, rank: int, npes: int, max_tok: int,
    recv_cap: int, comb_inp_nbytes: int, pair_mask: int, BM: int = 64,
    runtime_pair: bool = False,
    SBM: int = 128, BN: int = 256, BK: int = 256, INTER_MAX: int = 8192,
    use_nt: bool = True, cu_num: int = 112, g2_bhoist=True,
    g2_ascale_pf=True, first_bf16: bool = False,
    diagnostic_no_scatter: bool = False, include_residual: bool = False,
    pair_work_weight: int = 2, lds_reserve_bytes: int = 0,
    dual_accumulator: bool = False, parallel_experts: bool = False,
    scatter_vec: int = 8, m_swizzle: bool = False):
# fmt: on
    """Compile the two-expert aligned common-row Stage2 prototype."""
    arch = str(get_rocm_arch() or "")
    if not arch.startswith("gfx95"):
        raise RuntimeError(
            f"MegaMoE aligned-pair Stage2 requires CDNA4, got {arch or 'unknown'}"
        )
    if not runtime_pair and (pair_mask <= 0 or pair_mask.bit_count() != 2):
        raise ValueError("aligned-pair Stage2 requires exactly two selected experts")
    if BM not in (16, 32, 64) or BN not in (128, 256, 512) or BK != 256:
        raise ValueError(
            "the aligned-pair prototype requires BM16/32/64, BN128/256/512, and BK256"
        )
    if SBM % BM:
        raise ValueError("Stage1 SBM must be a multiple of pair BM")
    if model_dim % BN or inter_dim % BK:
        raise ValueError("model/inter dimensions must exactly tile BN/BK")
    if not 0 < comb_inp_nbytes < _BUFFER_OFFSET_ABI_BYTES:
        raise ValueError("aligned-pair P2P output exceeds the 32-bit buffer ABI")
    if runtime_pair:
        pair_a, pair_b = 0, 1
    else:
        pair_a = (pair_mask & -pair_mask).bit_length() - 1
        pair_b = (pair_mask ^ (1 << pair_a)).bit_length() - 1
    if pair_b >= experts:
        raise ValueError("aligned-pair expert is outside the local expert range")
    if not 1 <= pair_work_weight <= 8:
        raise ValueError("pair work weight must be in [1, 8]")
    if scatter_vec not in (8, 16) or BN % scatter_vec or 32 % scatter_vec:
        raise ValueError("aligned-pair scatter_vec must be 8 or 16 and divide BN/32")
    if parallel_experts and include_residual:
        raise ValueError("parallel experts currently require a separate residual kernel")
    a_stages = kStages + 1
    compute_lds_bytes = _stage2_lds_bytes(BM, BN, BK, "fp8", a_stages)
    second_compute_off = compute_lds_bytes
    lds_packed_a_off = compute_lds_bytes * 2
    lds_packed_b_off = lds_packed_a_off + BM * 4
    lds_weight_a_off = lds_packed_b_off + BM * 4
    lds_weight_b_off = lds_weight_a_off + BM * 4
    lds_peer_off = lds_weight_b_off + BM * 4
    lds_bytes = lds_peer_off + npes * 8
    if lds_reserve_bytes < 0 or lds_reserve_bytes % 16:
        raise ValueError("aligned-pair LDS reservation must be non-negative and 16-byte aligned")
    allocated_lds_bytes = max(lds_bytes, lds_reserve_bytes)
    if allocated_lds_bytes > 160 * 1024:
        raise ValueError(
            f"aligned-pair LDS use {allocated_lds_bytes} exceeds 160 KiB"
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
        buf: fx.Array[Int8, allocated_lds_bytes, 16]

    kernel_name = (
        f"megamoe_stage2_aligned_pair_{'runtime' if runtime_pair else f'e{pair_a}x{pair_b}'}"
        f"_t{BM}x{BN}x{BK}_cu{cu_num}_seqacc4_fa16{int(first_bf16)}"
        f"_ns{int(diagnostic_no_scatter)}_ur{int(include_residual)}"
        f"_wp1_pw{pair_work_weight}_lr{lds_reserve_bytes}"
        f"_da{int(dual_accumulator)}_pe{int(parallel_experts)}_sv{scatter_vec}"
        f"_ms{int(m_swizzle)}"
        f"_nt{int(use_nt)}_bh{int(g2_bhoist)}apf{int(g2_ascale_pf)}"
        f"_rtv3{int(runtime_pair)}_rq2_hw1"
    )

    # fmt: off
    @flyc.kernel(name=kernel_name, known_block_size=[512 if parallel_experts else 256, 1, 1])
    def kernel(arg_aq: fx.Int64, arg_ascale: fx.Int64, arg_bq: fx.Int64,
        arg_bscale: fx.Int64, arg_stids: fx.Int64, arg_sweights: fx.Int64,
        arg_eids: fx.Int64, arg_cumsum: fx.Int64, arg_trb: fx.Int64,
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

        expert_tile_end = buffer_ops.create_buffer_resource_from_addr(
            arg_expert_tile_end
        )
        count_matrix = buffer_ops.create_buffer_resource_from_addr(arg_count_matrix)
        pair_a_lane = fx.Int32(pair_a)
        pair_b_lane = fx.Int32(pair_b)
        pair_enabled_lane = fx.Int32(1)
        if const_expr(runtime_pair):
            pair_config = buffer_ops.create_buffer_resource_from_addr(
                arg_pair_config
            )
            parity_rsrc = buffer_ops.create_buffer_resource_from_addr(arg_parity)
            active_parity = buffer_ops.buffer_load(
                parity_rsrc, fx.Int32(0), vec_width=1, dtype=fx.Int32
            )
            packed_pair = buffer_ops.buffer_load(
                pair_config,
                active_parity * fx.Int32(npes) + fx.Int32(rank),
                vec_width=1,
                dtype=fx.Int32,
            )
            fx.rocdl.s_waitcnt(0)
            # Keep this load wave-uniform.  The former lane-0-only SSA value
            # was broadcast with readfirstlane after leaving a divergent
            # region; that produced the correct host-visible pair table but
            # intermittently selected the wrong experts inside Stage2.
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
            prev_a = buffer_ops.buffer_load(
                expert_tile_end, safe_prev_a, vec_width=1, dtype=fx.Int32
            ) * fx.Int32(SBM)
            prev_b = buffer_ops.buffer_load(
                expert_tile_end, safe_prev_b, vec_width=1, dtype=fx.Int32
            ) * fx.Int32(SBM)
            group_a_lane = (pair_a_rt > fx.Int32(0)).select(
                prev_a, fx.Int32(0)
            )
            group_b_lane = (pair_b_rt > fx.Int32(0)).select(
                prev_b, fx.Int32(0)
            )
            group_count = fx.Int32(0)
            group_column = fx.Int32(total_experts + rank)
            for source in range_constexpr(npes):
                group_count = group_count + buffer_ops.buffer_load(
                    count_matrix,
                    fx.Int32(source * total_segments) + group_column,
                    vec_width=1,
                    dtype=fx.Int32,
                )
            group_rows_lane = pair_enabled.select((
                (group_count + fx.Int32(SBM - 1)) // fx.Int32(SBM)
            ) * fx.Int32(SBM), fx.Int32(0))
        group_a = fx.Int32(rocdl.readfirstlane(T.i32, group_a_lane))
        group_b = fx.Int32(rocdl.readfirstlane(T.i32, group_b_lane))
        group_rows = fx.Int32(rocdl.readfirstlane(T.i32, group_rows_lane))
        total_m_blocks = group_rows // fx.Int32(BM)
        all_m_blocks = fx.Int32(0)
        if const_expr(include_residual):
            cumsum = buffer_ops.create_buffer_resource_from_addr(arg_cumsum)
            total_rows = buffer_ops.buffer_load(
                cumsum,
                fx.Int32(0),
                vec_width=1,
                dtype=fx.Int32,
            )
            all_m_blocks = (
                total_rows + fx.Int32(BM - 1)
            ) // fx.Int32(BM)
        peer_table = buffer_ops.create_buffer_resource_from_addr(arg_p2p_comb_inp)
        if tx < fx.Int32(npes):
            peer = buffer_ops.buffer_load(
                peer_table, tx, vec_width=1, dtype=fx.Int64
            )
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
        stids = buffer_ops.create_buffer_resource_from_addr(arg_stids)
        sweights = buffer_ops.create_buffer_resource_from_addr(arg_sweights)
        tile_row_base = buffer_ops.create_buffer_resource_from_addr(arg_trb)

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
                packed_a = buffer_ops.buffer_load(
                    stids, m_row_a + tx, vec_width=1, dtype=fx.Int32
                )
                packed_b = buffer_ops.buffer_load(
                    stids, m_row_b + tx, vec_width=1, dtype=fx.Int32
                )
                weight_a = buffer_ops.buffer_load(
                    sweights, m_row_a + tx, vec_width=1, dtype=fx.Float32
                )
                weight_b = buffer_ops.buffer_load(
                    sweights, m_row_b + tx, vec_width=1, dtype=fx.Float32
                )
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
            if const_expr(parallel_experts):
                second_half = wave >= fx.Int32(4)
                local_wave = second_half.select(wave - fx.Int32(4), wave)
                selected_lds = second_half.select(lds_compute_b, lds_compute)
                selected_row = second_half.select(m_row_b, m_row_a)
                selected_expert = second_half.select(global_b, global_a)
                selected_weight_off = second_half.select(
                    fx.Int32(lds_weight_b_off), fx.Int32(lds_weight_a_off)
                )
                selected_packed_off = second_half.select(
                    fx.Int32(lds_packed_b_off), fx.Int32(lds_packed_a_off)
                )
                fx.barrier()
                load_metadata(m_row_a, m_row_b)
                for slot in range_constexpr(kStages):
                    issue_a_load_lds_dt(
                        arg_aq,
                        selected_lds,
                        slot,
                        slot,
                        selected_row,
                        local_wave,
                        lane,
                        True,
                        kh_tile_a,
                        fx.Int32(k_bytes),
                        BM=BM,
                    )
                rocdl.sched_barrier(0)
                # fmt: off
                acc, _, _, _ = gemm2_compute_v2(
                    selected_lds, arg_ascale, arg_bq, arg_bscale, fx.Int64(0), arg_aq,
                    i32_max_m_blocks, fx.Int32(0), lane, local_wave, i32_inter, i32_hidden,
                    fx.Int32(0), fx.Int32(0), BM=BM, BN=BN, BK=BK, use_nt=use_nt,
                    INTER_MAX=INTER_MAX, aStages=a_stages, a_dtype="fp8", SBM=BM,
                    g2_bhoist=g2_bhoist, g2_ascale_pf=g2_ascale_pf,
                    expert_offset=expert_offset, explicit_m_row=selected_row,
                    explicit_n_block=n_block, explicit_expert=selected_expert)
                # fmt: on
                if const_expr(not diagnostic_no_scatter):
                    # The two four-wave groups compute and quantize their routes
                    # independently.  This keeps the original route-wise FP8
                    # contract while overlapping the two expert GEMMs.
                    p2p_scatter_epilog(
                        selected_lds, acc, n_block, local_wave, lane,
                        N_OUT=model_dim, BM=BM, BN=BN, npes=npes, topk=topk,
                        log2_max_tok=log2_max_tok,
                        mask_max_tok=mask_max_tok, recv_cap=recv_cap,
                        comb_inp_nbytes=comb_inp_nbytes,
                        lds_packed_off=selected_packed_off,
                        lds_weight_off=selected_weight_off,
                        lds_peer_off=lds_peer_off,
                        p2p_quant_type="fp8_blockwise_1x32",
                        scatter_vec=scatter_vec,
                    )
                return
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
            if const_expr(not diagnostic_no_scatter):
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
                    scatter_vec=scatter_vec,
                )
            # fmt: on

        def swizzle_pair_block(m_block):
            if const_expr(not m_swizzle):
                return m_block
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

        def run_single(m_block):
            m_row = m_block * fx.Int32(BM)
            sort_block = m_row // fx.Int32(SBM)
            row_in_sort_block = m_row - sort_block * fx.Int32(SBM)
            metadata_row = buffer_ops.buffer_load(
                tile_row_base,
                sort_block,
                vec_width=1,
                dtype=fx.Int32,
            ) + row_in_sort_block
            fx.barrier()
            if tx < fx.Int32(BM):
                packed = buffer_ops.buffer_load(
                    stids,
                    metadata_row + tx,
                    vec_width=1,
                    dtype=fx.Int32,
                )
                weight = buffer_ops.buffer_load(
                    sweights,
                    metadata_row + tx,
                    vec_width=1,
                    dtype=fx.Float32,
                )
                fx.ptr_store(
                    packed,
                    lds_typed_ptr(
                        fx.Int32(lds_packed_a_off) + tx * fx.Int32(4),
                        T.i32,
                        align=4,
                    ),
                )
                fx.ptr_store(
                    weight,
                    lds_typed_ptr(
                        fx.Int32(lds_weight_a_off) + tx * fx.Int32(4),
                        T.f32,
                        align=4,
                    ),
                )
            issue_all_a_loads(m_row, lds_compute)
            rocdl.sched_barrier(0)
            unit = m_block * (i32_hidden // fx.Int32(BN)) + n_block
            # fmt: off
            acc, _, output_n_block, _ = gemm2_compute_v2(
                lds_compute, arg_ascale, arg_bq, arg_bscale, arg_eids, arg_aq,
                i32_max_m_blocks, unit, lane, wave, i32_inter, i32_hidden,
                fx.Int32(0), fx.Int32(0), BM=BM, BN=BN, BK=BK, use_nt=use_nt,
                INTER_MAX=INTER_MAX, aStages=a_stages, a_dtype="fp8", SBM=SBM,
                g2_bhoist=g2_bhoist, g2_ascale_pf=g2_ascale_pf,
                expert_offset=expert_offset)
            p2p_scatter_epilog(
                lds_compute, acc, output_n_block, wave, lane, N_OUT=model_dim,
                BM=BM, BN=BN, npes=npes, topk=topk,
                log2_max_tok=log2_max_tok, mask_max_tok=mask_max_tok,
                recv_cap=recv_cap, comb_inp_nbytes=comb_inp_nbytes,
                lds_packed_off=lds_packed_a_off,
                lds_weight_off=lds_weight_a_off, lds_peer_off=lds_peer_off,
                p2p_quant_type="fp8_blockwise_1x32", scatter_vec=scatter_vec)
            # fmt: on

        if const_expr(include_residual):
            group_a_block = group_a // fx.Int32(BM)
            group_b_block = group_b // fx.Int32(BM)
            second_threshold = group_b_block - total_m_blocks
            residual_blocks = all_m_blocks - total_m_blocks * fx.Int32(2)
            weighted_total = (
                total_m_blocks * fx.Int32(pair_work_weight) + residual_blocks
            )
            pair_slots = (
                fx.Int32(cu_num)
                * total_m_blocks
                * fx.Int32(pair_work_weight)
                + weighted_total
                - fx.Int32(1)
            ) // weighted_total
            max_pair_slots = (residual_blocks > fx.Int32(0)).select(
                fx.Int32(cu_num - 1),
                fx.Int32(cu_num),
            )
            pair_slots = (pair_slots < fx.Int32(1)).select(
                fx.Int32(1),
                pair_slots,
            )
            pair_slots = (pair_slots > max_pair_slots).select(
                max_pair_slots,
                pair_slots,
            )
            residual_slots = fx.Int32(cu_num) - pair_slots
            pair_before = m_slot * pair_slots // fx.Int32(cu_num)
            pair_through = (
                (m_slot + fx.Int32(1)) * pair_slots // fx.Int32(cu_num)
            )
            is_pair_slot = pair_through > pair_before
            if is_pair_slot:
                pair_slot = pair_before
                pair_remaining = total_m_blocks - pair_slot
                pair_iterations = (
                    pair_remaining + pair_slots - fx.Int32(1)
                ) // pair_slots
                for iteration in range(
                    fx.Int32(0), pair_iterations, fx.Int32(1)
                ):
                    run_pair(
                        swizzle_pair_block(pair_slot + iteration * pair_slots)
                    )
            else:
                residual_slot = m_slot - pair_before
                residual_remaining = residual_blocks - residual_slot
                residual_iterations = (
                    residual_remaining + residual_slots - fx.Int32(1)
                ) // residual_slots
                for iteration in range(
                    fx.Int32(0), residual_iterations, fx.Int32(1)
                ):
                    residual = residual_slot + iteration * residual_slots
                    after_a = residual >= group_a_block
                    after_b = residual >= second_threshold
                    normal_block = (
                        residual
                        + after_a.select(total_m_blocks, fx.Int32(0))
                        + after_b.select(total_m_blocks, fx.Int32(0))
                    )
                    run_single(normal_block)
        else:
            for iteration in range(fx.Int32(0), iterations, fx.Int32(1)):
                m_block = m_slot + iteration * fx.Int32(cu_num)
                run_pair(swizzle_pair_block(m_block))

    # fmt: off
    @flyc.jit
    def launch(arg_aq: fx.Int64, arg_ascale: fx.Int64, arg_bq: fx.Int64,
        arg_bscale: fx.Int64, arg_stids: fx.Int64, arg_sweights: fx.Int64,
        arg_eids: fx.Int64, arg_cumsum: fx.Int64, arg_trb: fx.Int64,
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
            arg_eids,
            arg_cumsum,
            arg_trb,
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
            block=(512 if parallel_experts else 256, 1, 1),
            stream=stream,
        )

    return launch


_PAIR_LAUNCH_CACHE = {}


# fmt: off
def run_mega_moe_stage2_aligned_pair(arg_aq, arg_ascale, arg_bq, arg_bscale,
    arg_stids, arg_sweights, arg_eids, arg_cumsum, arg_trb,
    arg_expert_tile_end, arg_count_matrix, arg_pair_config, arg_parity,
    arg_p2p, row_capacity,
    i32_inter, i32_hidden, stream, **compile_kw):
# fmt: on
    """Compile/cache and launch the aligned common-row prototype."""
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
        arg_eids,
        arg_cumsum,
        arg_trb,
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
    arg_stids, arg_sweights, arg_eids, arg_cumsum, arg_trb,
    arg_expert_tile_end, arg_count_matrix, arg_pair_config, arg_parity,
    arg_p2p, row_capacity, i32_inter, i32_hidden, stream, **compile_kw):
# fmt: on
    """Compile and load the aligned-pair Stage2 kernel without dispatching it."""
    key = tuple(sorted(compile_kw.items()))
    launch = _PAIR_LAUNCH_CACHE.get(key)
    if launch is None:
        launch = compile_mega_moe_stage2_aligned_pair(**compile_kw)
        _PAIR_LAUNCH_CACHE[key] = launch
    bm = int(compile_kw.get("BM", 64))
    max_m_blocks = (int(row_capacity) + bm - 1) // bm
    return launch.preload(
        arg_aq,
        arg_ascale,
        arg_bq,
        arg_bscale,
        arg_stids,
        arg_sweights,
        arg_eids,
        arg_cumsum,
        arg_trb,
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
