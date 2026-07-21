# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Fused RMSNorm, 3D-MRoPE, optional per-tensor FP8 quantization, and
shuffle-layout paged-KV-cache writes in FlyDSL.

This implementation uses separate Q and KV kernels and supports a subset of
``aiter.fused_qk_norm_mrope_3d_cache_pts_quant_shuffle``. Unsupported
configurations raise an error.

The Q kernel assigns each head to a 32-lane RMSNorm group. NEOX partners are
exchanged with a lane shuffle, and head-independent position and cos/sin loads
are hoisted out of the head loop.

The KV kernel stages one page and KV head in LDS. Complete, aligned,
contiguously mapped pages use 16-byte stores; other mappings use an
elementwise scatter. The cache layouts are
``K [blocks, heads, D/x, block_size, x]`` and
``V [blocks, heads, block_size/x, D, x]``.
"""

import math
from functools import lru_cache

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl.expr import const_expr, gpu, range_constexpr
from flydsl.expr.typing import T

from aiter.ops.flydsl.utils import get_shared_memory_per_block
from aiter.utility import dtypes as aiter_dtypes

from .kernels_common import get_warp_size
from .tensor_shim import _run_compiled

# RMSNorm uses the production kernel's 32-lane logical groups on both wave32
# and wave64 targets.
WAVE = get_warp_size()
_LOG2_WAVE = int(math.log2(WAVE))

RMS_GROUP = 32
_LOG2_RMS_GROUP = int(math.log2(RMS_GROUP))

# 512 threads gives good occupancy and covers the common FP8
# block_size=64, x=16, D=128 write grains without a tail iteration.
KV_THREADS = 512

# A cache run is always 16 B: 16 FP8 elements or 8 bf16 elements.
_RUN_BYTES = 16

# HIP exposes gridDim.y as a 16-bit dimension.
_MAX_GRID_Y = 65535


def _ceil_div(a: int, b: int) -> int:
    return -(-a // b)


# ============================================================================
# Cache-layout helpers.
# ============================================================================
def _is_contiguous_from_dim1(t: torch.Tensor) -> bool:
    """Whether every cache block is internally contiguous.

    Dim 0 may have padding/interleaving, as with a
    ``[num_blocks, 2, ...]`` allocation sliced along dim 1.
    """
    expected = 1
    for dim in range(t.dim() - 1, 0, -1):
        if t.shape[dim] != 1 and t.stride(dim) != expected:
            return False
        expected *= t.shape[dim]
    return True


def rms_reduce_add(x, lane, broadcast_half=True):
    v = x
    for sh_exp in range_constexpr(_LOG2_RMS_GROUP):
        off = RMS_GROUP // (2 << sh_exp)
        peer = v.shuffle_xor(off, RMS_GROUP)
        v = v.addf(peer, fastmath=fx.FastMathFlags.fast)
    if const_expr(WAVE > RMS_GROUP and broadcast_half):
        other_half = v.shuffle_xor(RMS_GROUP, WAVE)
        return (lane < RMS_GROUP).select(v, other_half)
    return v


def mrope_cos_sin(
    col,
    tok,
    positions_t,
    cos_t,
    sin_t,
    mrope_section,
    is_interleaved,
):
    """Gather 3D-MRoPE cos/sin for a column in ``[0, D/2)``.

    Interleaved mode assigns eligible columns to section ``col % 3`` and
    otherwise uses section 0. Non-interleaved mode uses contiguous sections.
    """
    if const_expr(is_interleaved):
        mid = col % 3
        is_mid1 = mid == 1
        boundary = is_mid1.select(
            fx.Int32(mrope_section[1] * 3), fx.Int32(mrope_section[2] * 3)
        )
        in_range = col < boundary
        use_mid = (mid != 0) & in_range
        sect_idx = use_mid.select(mid, fx.Int32(0))
    else:
        in_section_0 = col < mrope_section[0]
        in_section_1 = col < mrope_section[0] + mrope_section[1]
        sect_idx = in_section_0.select(
            fx.Int32(0), in_section_1.select(fx.Int32(1), fx.Int32(2))
        )

    pos = positions_t[sect_idx, tok].to(fx.Int32)
    cos_v = fx.Float32(cos_t[pos, col])
    sin_v = fx.Float32(sin_t[pos, col])
    return cos_v, sin_v


@lru_cache(maxsize=1)
def _fp8_range():
    fp8_dtype = aiter_dtypes.fp8
    fp8_max = float(torch.finfo(fp8_dtype).max)
    fp8_min = float(torch.finfo(fp8_dtype).min)
    return fp8_min, fp8_max


def _align_up(value: int, alignment: int) -> int:
    return ((value + alignment - 1) // alignment) * alignment


# Q kernel
# Upper bounds tuned on MI355X for D=128. Head batching amortizes the
# head-independent position and cos/sin loads.
Q_WAVES_PER_BLOCK = 2
Q_HEAD_ITERS = 16


def _q_launch_config(num_heads_q: int) -> tuple[int, int]:
    """Waves per Q workgroup and heads walked per wave, capped by head count.

    Heads are batched only as far as ``num_heads_q`` reaches, so a small head
    count does not spend its iterations on guarded-off work.
    """
    rows_per_wave = WAVE // RMS_GROUP
    waves = max(1, min(Q_WAVES_PER_BLOCK, num_heads_q // rows_per_wave))
    iters = max(1, min(Q_HEAD_ITERS, num_heads_q // (waves * rows_per_wave)))
    return waves, iters


def _build_q_kernel(
    *,
    head_size: int,
    num_heads_q: int,
    mrope_section: list[int],
    eps: float,
    is_interleaved: bool,
    gemma_norm: bool,
    match_hip: bool,
    waves_per_block: int,
    head_iters: int,
):
    D = head_size
    H_Q = num_heads_q
    HALF = D // 2
    # One head is owned by one 32-lane RMSNorm group: lane ``rl`` holds the
    # ``PROD_VEC_SIZE`` contiguous columns starting at ``PROD_VEC_SIZE * rl``,
    # which is the distribution the production reduction order is defined on.
    # The NEOX partner of every column that lane ``rl`` holds lives in lane
    # ``rl ^ 16``, so the pair values come from one lane shuffle rather than a
    # second read of the row.
    PROD_VEC_SIZE = D // RMS_GROUP
    PAIRS_PER_LANE = PROD_VEC_SIZE // 2
    PARTNER_XOR = RMS_GROUP // 2
    ROWS_PER_WAVE = WAVE // RMS_GROUP
    Q_THREADS = waves_per_block * WAVE
    HEADS_PER_BLOCK = waves_per_block * ROWS_PER_WAVE * head_iters
    N_HEAD_BLOCKS = _ceil_div(H_Q, HEADS_PER_BLOCK)
    NEEDS_HEAD_GUARD = N_HEAD_BLOCKS * HEADS_PER_BLOCK != H_Q

    kname = (
        f"qk_norm_mrope_q_D{D}_H{H_Q}_w{waves_per_block}h{head_iters}"
        f"{'_bf16_truncate' if match_hip else '_f32'}_flydsl"
    )

    @flyc.kernel(name=kname, known_block_size=[Q_THREADS, 1, 1])
    def kernel(
        qkv: fx.Tensor,  # [T, H_Q+H_K+H_V, D] bf16, contig
        positions: fx.Tensor,  # [3, T] i64, arbitrary 2-D layout
        cos_sin: fx.Tensor,  # [max_pos, D] bf16 (cos=[:, :D/2], sin=[:, D/2:])
        q_norm_w: fx.Tensor,  # [D] bf16
        q_out: fx.Tensor,  # [T, H_Q, D] bf16
        token_offset: fx.Int32,
    ):
        fm_fast = fx.FastMathFlags.fast
        cos_t = fx.Tensor(
            fx.make_view(
                fx.get_iter(cos_sin),
                fx.make_layout((cos_sin.shape[0], HALF), cos_sin.stride),
            )
        )
        sin_t = fx.Tensor(
            fx.make_view(
                fx.get_iter(cos_sin) + HALF,
                fx.make_layout((cos_sin.shape[0], HALF), cos_sin.stride),
            )
        )
        layout_rms_values = fx.make_layout(PROD_VEC_SIZE, stride=1)
        qkv_rms_tiles = fx.logical_divide(
            qkv, fx.make_tile(None, None, layout_rms_values)
        )
        copy_rms = fx.make_copy_atom(
            fx.UniversalCopy(PROD_VEC_SIZE * fx.BFloat16.width), fx.BFloat16
        )

        bid_head = fx.block_idx.x
        bid_t = fx.block_idx.y
        tok = bid_t + token_offset
        t = fx.thread_idx.x
        wid = t // WAVE
        lane = t % WAVE
        rl = lane % RMS_GROUP
        row_in_wave = lane // RMS_GROUP

        is_low = rl < PARTNER_XOR
        pair_base = (rl % PARTNER_XOR) * PROD_VEC_SIZE + is_low.select(
            fx.Int32(0), fx.Int32(PAIRS_PER_LANE)
        )

        # Head-independent work, hoisted out of the head loop: every head of
        # this token shares the same mrope cos/sin gather and norm weights.
        cols, cos_vs, sin_vs, w0s, w1s = [], [], [], [], []
        for p in range_constexpr(PAIRS_PER_LANE):
            col = pair_base + p
            cos_v, sin_v = mrope_cos_sin(
                col,
                tok,
                positions,
                cos_t,
                sin_t,
                mrope_section,
                is_interleaved,
            )
            cols.append(col)
            cos_vs.append(cos_v)
            sin_vs.append(sin_v)
            w0 = fx.Float32(q_norm_w[col])
            w1 = fx.Float32(q_norm_w[col + HALF])
            if const_expr(gemma_norm):
                w0 = w0 + 1.0
                w1 = w1 + 1.0
            w0s.append(w0)
            w1s.append(w1)

        def compute_head(head):
            rms_tile = fx.slice(qkv_rms_tiles, (tok, head, (None, rl)))
            rms_reg = fx.make_rmem_tensor(layout_rms_values, fx.BFloat16)
            fx.copy_atom_call(copy_rms, rms_tile, rms_reg)
            own = rms_reg.load().to(fx.Float32)

            sumsq_local = fx.Float32(0.0)
            for i in range_constexpr(PROD_VEC_SIZE):
                value = own[i]
                sumsq_local = sumsq_local + value * value
            sumsq = rms_reduce_add(sumsq_local, rl, broadcast_half=False)
            rstd = fx.rsqrt(sumsq * (1.0 / D) + eps, fastmath=fm_fast)

            # The partner lane's columns complete this lane's NEOX pairs.
            peer = [
                own[i].shuffle_xor(PARTNER_XOR, RMS_GROUP)
                for i in range_constexpr(PROD_VEC_SIZE)
            ]

            for p in range_constexpr(PAIRS_PER_LANE):
                x0 = is_low.select(own[p], peer[PAIRS_PER_LANE + p])
                x1 = is_low.select(peer[p], own[PAIRS_PER_LANE + p])
                if const_expr(match_hip):
                    # Match the production bf16 RMSNorm materialization before
                    # applying RoPE in fp32.
                    xn0 = (x0 * rstd * w0s[p]).to(fx.BFloat16).to(fx.Float32)
                    xn1 = (x1 * rstd * w1s[p]).to(fx.BFloat16).to(fx.Float32)
                else:
                    xn0 = x0 * rstd * w0s[p]
                    xn1 = x1 * rstd * w1s[p]

                o0 = xn0 * cos_vs[p] - xn1 * sin_vs[p]
                o1 = xn1 * cos_vs[p] + xn0 * sin_vs[p]
                q_out[tok, head, cols[p]] = o0.to(fx.BFloat16)
                q_out[tok, head, cols[p] + HALF] = o1.to(fx.BFloat16)

        for i in range_constexpr(head_iters):
            # Waves of a block sit on adjacent heads at every step, so their
            # concurrent loads stay within one contiguous qkv region.
            head = (
                bid_head * HEADS_PER_BLOCK
                + (i * waves_per_block + wid) * ROWS_PER_WAVE
                + row_in_wave
            )
            if const_expr(NEEDS_HEAD_GUARD):
                if head < H_Q:
                    compute_head(head)
            else:
                compute_head(head)

    @flyc.jit
    def launch(
        qkv: fx.Tensor,
        positions_storage: fx.Tensor,
        cos_sin: fx.Tensor,
        q_norm_w: fx.Tensor,
        q_out: fx.Tensor,
        num_tokens: fx.Int32,
        token_offset: fx.Int32,
        positions_stride_0: fx.Int32,
        positions_stride_1: fx.Int32,
        stream: fx.Stream = fx.Stream(None),  # noqa: B008
    ):
        positions = fx.Tensor(
            fx.make_view(
                fx.get_iter(positions_storage),
                fx.make_layout(
                    (3, num_tokens + token_offset),
                    (positions_stride_0, positions_stride_1),
                ),
            )
        )
        k = kernel(
            qkv,
            positions,
            cos_sin,
            q_norm_w,
            q_out,
            token_offset,
        )
        k.launch(
            grid=(N_HEAD_BLOCKS, fx.Int64(num_tokens), 1),
            block=(Q_THREADS, 1, 1),
            stream=stream,
        )

    return launch


# KV kernel: compute into LDS, then write the shuffle-layout cache.
def _build_kv_kernel(
    *,
    num_heads_q: int,
    num_heads_kv: int,  # num_heads_k == num_heads_v (validated by caller)
    head_size: int,
    mrope_section: list[int],
    eps: float,
    block_size: int,
    x: int,
    emit_flat_kv: bool,
    is_interleaved: bool,
    gemma_norm: bool,
    match_hip: bool,
    cache_is_fp8: bool,
    k_cache_block_stride: int,
    v_cache_block_stride: int,
):
    H_Q, H_KV, D = num_heads_q, num_heads_kv, head_size
    HALF = D // 2
    PROD_VEC_SIZE = D // RMS_GROUP
    PAIR_LANES = min(WAVE, HALF)
    PAIRS_PER_LANE = HALF // PAIR_LANES
    WAVES_PER_BLOCK = KV_THREADS // WAVE
    # A phase group must cover both the RMS reduction group and the active
    # pair lanes. For D=64, the pair layout has 32 active lanes and lets each
    # wave process two token rows without duplicating pair work.
    COMPUTE_GROUP = max(RMS_GROUP, PAIR_LANES)
    COMPUTE_GROUPS_PER_BLOCK = KV_THREADS // COMPUTE_GROUP
    PHASE1_ITERS = _ceil_div(block_size, COMPUTE_GROUPS_PER_BLOCK)
    CACHE_FX_TYPE = fx.Int8 if cache_is_fp8 else fx.BFloat16

    STAGE_ELEMS = block_size * D
    K_TOTAL_RUNS = (D // x) * block_size
    K_ITERS = _ceil_div(K_TOTAL_RUNS, KV_THREADS)
    V_TOTAL_RUNS = (block_size // x) * D
    V_ITERS = _ceil_div(V_TOTAL_RUNS, KV_THREADS)
    SCATTER_ELEMS = block_size * D
    SCATTER_ITERS = _ceil_div(SCATTER_ELEMS, KV_THREADS)

    @fx.struct
    class SharedStorage:
        k_lds: fx.Array[CACHE_FX_TYPE, STAGE_ELEMS, 16]
        v_lds: fx.Array[CACHE_FX_TYPE, STAGE_ELEMS, 16]
        mapping_ok: fx.Array[fx.Int32, 1, 4]

    _name_parts = [
        "qk_norm_mrope_kv_shuffle",
        f"H{H_KV}",
        f"D{D}",
        f"bs{block_size}",
        f"x{x}",
        "fp8" if cache_is_fp8 else "bf16",
    ]
    if emit_flat_kv:
        _name_parts.append("kvout")
    _name_parts.append("bf16_truncate" if match_hip else "f32")
    _name_parts.append("flydsl")
    kname = "_".join(_name_parts)

    @flyc.kernel(name=kname, known_block_size=[KV_THREADS, 1, 1])
    def kernel(
        qkv: fx.Tensor,  # [T, H_Q+H_K+H_V, D] bf16, contig
        positions: fx.Tensor,  # [3, T] i64, arbitrary 2-D layout
        cos_sin: fx.Tensor,  # [max_pos, D] bf16
        k_norm_w: fx.Tensor,  # [D] bf16
        k_cache: fx.Tensor,  # typed shuffle-layout FP8 or bf16 K cache
        v_cache: fx.Tensor,  # typed shuffle-layout FP8 or bf16 V cache
        slot_mapping: fx.Tensor,  # [T] i64
        k_scale: fx.Tensor,  # [1] f32 (per-tensor scale; no host sync)
        v_scale: fx.Tensor,  # [1] f32
        k_out: fx.Tensor,  # [T, H_K, D] cache dtype (dummy unless emit_flat_kv)
        v_out: fx.Tensor,  # [T, H_V, D] cache dtype (dummy unless emit_flat_kv)
        num_tokens: fx.Int32,
        page_block_offset: fx.Int32,
    ):
        fm_fast = fx.FastMathFlags.fast
        cos_t = fx.Tensor(
            fx.make_view(
                fx.get_iter(cos_sin),
                fx.make_layout((cos_sin.shape[0], HALF), cos_sin.stride),
            )
        )
        sin_t = fx.Tensor(
            fx.make_view(
                fx.get_iter(cos_sin) + HALF,
                fx.make_layout((cos_sin.shape[0], HALF), cos_sin.stride),
            )
        )
        layout_tx_wave_lane = fx.make_layout((WAVES_PER_BLOCK, WAVE), stride=(WAVE, 1))
        # Two logical ownership maps of D are used:
        #  * RMSNorm: 32 lanes own contiguous D/32-element vectors.
        #  * NEOX pairs: active lanes own strided columns in [0, D/2), with
        #    [pair, half] kept as one value mode.
        layout_rms_values = fx.make_layout(PROD_VEC_SIZE, stride=1)
        layout_pair_tv = fx.make_layout(
            (PAIR_LANES, (PAIRS_PER_LANE, 2)),
            stride=(1, (PAIR_LANES, HALF)),
        )
        qkv_rms_tiler = fx.make_tile(None, None, layout_rms_values)
        layout_stage = fx.make_layout((block_size, D), stride=(D, 1))
        layout_k_runs = fx.make_layout((D // x, block_size), stride=(block_size, 1))
        layout_v_runs = fx.make_layout((block_size // x, D), stride=(D, 1))
        layout_stage_runs = fx.make_layout((block_size, D // x, x), stride=(D, x, 1))
        layout_run = fx.make_layout(x, stride=1)
        layout_k_cache = fx.make_layout(
            (H_KV, D // x, block_size, x),
            stride=(D * block_size, block_size * x, x, 1),
        )
        layout_v_cache = fx.make_layout(
            (H_KV, block_size // x, D, x),
            stride=(block_size * D, D * x, x, 1),
        )
        copy_rms = fx.make_copy_atom(
            fx.UniversalCopy(PROD_VEC_SIZE * fx.BFloat16.width), fx.BFloat16
        )
        copy_128b = fx.make_copy_atom(fx.UniversalCopy128b(), CACHE_FX_TYPE)

        def load_rms_sumsq(tile):
            reg = fx.make_rmem_tensor(layout_rms_values, fx.BFloat16)
            fx.copy_atom_call(copy_rms, tile, reg)
            values = reg.load().to(fx.Float32)
            acc = fx.Float32(0.0)
            for i in range_constexpr(PROD_VEC_SIZE):
                value = values[i]
                acc = acc + value * value
            return acc

        def page_is_coalescible(tok0, lane):
            base_slot = slot_mapping[tok0]
            full_page = tok0 + block_size <= num_tokens
            valid_base = (
                full_page and (base_slot >= 0) and ((base_slot % block_size) == 0)
            )
            mapping_valid = valid_base.select(fx.Int32(1), fx.Int32(0))
            if full_page:
                for check_it in range_constexpr(_ceil_div(block_size, WAVE)):
                    token_local = lane + WAVE * check_it
                    if token_local < block_size:
                        tok = tok0 + token_local
                        slot = slot_mapping[tok]
                        expected = base_slot + token_local
                        mapping_valid = mapping_valid & (slot == expected).select(
                            fx.Int32(1), fx.Int32(0)
                        )
            for sh_exp in range_constexpr(_LOG2_WAVE):
                off = WAVE // (2 << sh_exp)
                peer_valid = mapping_valid.shuffle_xor(off, WAVE)
                mapping_valid = mapping_valid & peer_valid
            return mapping_valid

        def _fp8_clamp(value: fx.Float32):
            fp8_min, fp8_max = _fp8_range()
            # Note: min(), max() here cannot be traced by FlyDSL correctly
            # FIXME: NaN values will be clamped to min, max. Check whether this matches intended semantics
            vout = value if value > fp8_min else fp8_min  # noqa: FURB136
            vout = vout if vout < fp8_max else fp8_max  # noqa: FURB136
            return vout

        def quant_pair_fp8(v0, v1, scale):
            s0 = _fp8_clamp(v0 * scale)
            s1 = _fp8_clamp(v1 * scale)

            packed = fx.Int32(fx.rocdl.cvt_pk_fp8_f32(T.i32, s0, s1, fx.Int32(0), 0))
            byte0 = packed.to(fx.Int8)
            byte1 = (packed >> 8).to(fx.Int8)
            return byte0, byte1

        head = fx.block_idx.x  # kv head 0..H_KV-1
        blk = fx.block_idx.y  # logical page index
        t = fx.thread_idx.x  # 0..KV_THREADS-1

        if const_expr(cache_is_fp8):
            # Precompute 1 / scale
            k_scale_value = 1.0 / k_scale[0]
            v_scale_value = 1.0 / v_scale[0]
        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        k_lds = lds.k_lds
        v_lds = lds.v_lds
        k_lds_view = k_lds.view(layout_stage)
        v_lds_view = v_lds.view(layout_stage)
        mapping_ok = lds.mapping_ok

        tok0 = (blk + page_block_offset) * block_size

        # The final logical page may be partial, so guard token-sized inputs
        # and outputs. Phase 2 sends partial pages through the scatter path.
        coord_wl = fx.idx2crd(t, layout_tx_wave_lane)
        wid = fx.Int32(fx.get(coord_wl, 0))
        lane = fx.Int32(fx.get(coord_wl, 1))
        compute_group = t // COMPUTE_GROUP
        pair_lane = t % PAIR_LANES

        # Partition phase-1 tensors by their RMS and NEOX-pair ownership,
        # then retain only this thread's pair lane across all token rows.
        qkv_rms_tiles = fx.logical_divide(qkv, qkv_rms_tiler)
        qkv_pair_view = fx.composition(qkv, fx.make_tile(None, None, layout_pair_tv))
        weight_pair_view = fx.composition(k_norm_w, layout_pair_tv)
        k_lds_pair_view = fx.composition(k_lds_view, fx.make_tile(None, layout_pair_tv))
        v_lds_pair_view = fx.composition(v_lds_view, fx.make_tile(None, layout_pair_tv))
        lane_coord = (pair_lane, None)
        qkv_lane_pairs = fx.slice(qkv_pair_view, (None, None, lane_coord))
        w_lane_pairs = fx.slice(weight_pair_view, lane_coord)
        k_lds_lane_pairs = fx.slice(k_lds_pair_view, (None, lane_coord))
        v_lds_lane_pairs = fx.slice(v_lds_pair_view, (None, lane_coord))
        lane_pair_cols = [
            pair_lane + p * PAIR_LANES for p in range_constexpr(PAIRS_PER_LANE)
        ]
        if const_expr(emit_flat_kv):
            k_out_pair_view = fx.composition(
                k_out, fx.make_tile(None, None, layout_pair_tv)
            )
            v_out_pair_view = fx.composition(
                v_out, fx.make_tile(None, None, layout_pair_tv)
            )
            k_out_lane_pairs = fx.slice(k_out_pair_view, (None, None, lane_coord))
            v_out_lane_pairs = fx.slice(v_out_pair_view, (None, None, lane_coord))
        for it in range_constexpr(PHASE1_ITERS):
            token_local = compute_group + COMPUTE_GROUPS_PER_BLOCK * it
            if token_local < block_size:
                tok = tok0 + token_local
                if tok < num_tokens:
                    if const_expr(emit_flat_kv):
                        flat_slot_valid = slot_mapping[tok] >= 0
                    sumsq_local = fx.Float32(0.0)
                    # D=64 uses one row per 32-lane half; larger heads use
                    # one row per native wave and broadcast the lower half's
                    # reduction result.
                    if pair_lane < RMS_GROUP:
                        rms_tile = fx.slice(
                            qkv_rms_tiles,
                            (tok, H_Q + head, (None, pair_lane)),
                        )
                        sumsq_local = load_rms_sumsq(rms_tile)
                    sumsq = rms_reduce_add(
                        sumsq_local,
                        pair_lane,
                        broadcast_half=PAIR_LANES > RMS_GROUP,
                    )
                    rstd = fx.rsqrt(sumsq * (1.0 / D) + eps, fastmath=fm_fast)

                    k_pairs = fx.slice(qkv_lane_pairs, (tok, H_Q + head, None))
                    v_pairs = fx.slice(qkv_lane_pairs, (tok, H_Q + H_KV + head, None))
                    k_lds_pairs = fx.slice(k_lds_lane_pairs, (token_local, None))
                    v_lds_pairs = fx.slice(v_lds_lane_pairs, (token_local, None))

                    # Define these outside the compile-time branch for tracing.
                    k_out_pairs = (
                        fx.slice(k_out_lane_pairs, (tok, head, None))
                        if const_expr(emit_flat_kv)
                        else None
                    )
                    v_out_pairs = (
                        fx.slice(v_out_lane_pairs, (tok, head, None))
                        if const_expr(emit_flat_kv)
                        else None
                    )
                    for p in range_constexpr(PAIRS_PER_LANE):
                        col = lane_pair_cols[p]
                        k0 = fx.Float32(k_pairs[p, 0])
                        k1 = fx.Float32(k_pairs[p, 1])
                        w0 = fx.Float32(w_lane_pairs[p, 0])
                        w1 = fx.Float32(w_lane_pairs[p, 1])
                        if const_expr(gemma_norm):
                            w0 = w0 + 1.0
                            w1 = w1 + 1.0
                        if const_expr(match_hip):
                            # Production materializes the weighted RMSNorm
                            # result as bf16 before RoPE.
                            xn0 = (k0 * rstd * w0).to(fx.BFloat16).to(fx.Float32)
                            xn1 = (k1 * rstd * w1).to(fx.BFloat16).to(fx.Float32)
                        else:
                            xn0 = k0 * rstd * w0
                            xn1 = k1 * rstd * w1

                        cos_v, sin_v = mrope_cos_sin(
                            col,
                            tok,
                            positions,
                            cos_t,
                            sin_t,
                            mrope_section,
                            is_interleaved,
                        )
                        o0 = xn0 * cos_v - xn1 * sin_v
                        o1 = xn1 * cos_v + xn0 * sin_v
                        if const_expr(cache_is_fp8):
                            # Production assigns RoPE output to vec_t<bf16>
                            # before converting that vector to FP8.
                            o0 = o0.to(fx.BFloat16)
                            o1 = o1.to(fx.BFloat16)
                            kb0, kb1 = quant_pair_fp8(o0, o1, k_scale_value)
                        else:
                            kb0 = o0.to(fx.BFloat16)
                            kb1 = o1.to(fx.BFloat16)
                        k_lds_pairs[p, 0] = kb0
                        k_lds_pairs[p, 1] = kb1
                        if const_expr(emit_flat_kv) and flat_slot_valid:
                            k_out_pairs[p, 0] = kb0
                            k_out_pairs[p, 1] = kb1

                        v0 = fx.Float32(v_pairs[p, 0])
                        v1 = fx.Float32(v_pairs[p, 1])
                        if const_expr(cache_is_fp8):
                            vb0, vb1 = quant_pair_fp8(v0, v1, v_scale_value)
                        else:
                            vb0 = v0.to(fx.BFloat16)
                            vb1 = v1.to(fx.BFloat16)
                        v_lds_pairs[p, 0] = vb0
                        v_lds_pairs[p, 1] = vb1
                        if const_expr(emit_flat_kv) and flat_slot_valid:
                            v_out_pairs[p, 0] = vb0
                            v_out_pairs[p, 1] = vb1

        # Wave 0 checks whether the whole logical page maps to one aligned
        # physical page. Other mappings use the scatter path.
        if wid == 0:
            mapping_valid = page_is_coalescible(tok0, lane)
            if lane == 0:
                mapping_ok[0] = mapping_valid
        gpu.barrier()
        can_coalesce = mapping_ok[0] != 0

        if can_coalesce:
            block_id = slot_mapping[tok0] // block_size
            # The block strides are compile-time-specialized. This supports
            # views such as kv[:, 0] / kv[:, 1], whose blocks are internally
            # packed but separated by the other cache plane.
            k_block_base = block_id * k_cache_block_stride
            v_block_base = block_id * v_cache_block_stride
            k_cache_block = fx.Tensor(
                fx.make_view(fx.get_iter(k_cache) + k_block_base, layout_k_cache)
            )
            v_cache_block = fx.Tensor(
                fx.make_view(fx.get_iter(v_cache) + v_block_base, layout_v_cache)
            )
            k_lds_runs = k_lds.ptr.view(layout_stage_runs)
            for it in range_constexpr(K_ITERS):
                r = t + KV_THREADS * it
                if r < K_TOTAL_RUNS:
                    coord_k_run = fx.idx2crd(r, layout_k_runs)
                    chunk_k = fx.get(coord_k_run, 0)
                    block_off = fx.get(coord_k_run, 1)
                    src_k = fx.slice(k_lds_runs, (block_off, chunk_k, None))
                    dst_k = fx.slice(k_cache_block, (head, chunk_k, block_off, None))
                    reg_k = fx.make_rmem_tensor(layout_run, CACHE_FX_TYPE)
                    fx.copy_atom_call(copy_128b, src_k, reg_k)
                    vec_k = fx.memref_load_vec(reg_k)
                    fx.ptr_store(vec_k, fx.get_iter(dst_k))

            for it in range_constexpr(V_ITERS):
                r = t + KV_THREADS * it
                if r < V_TOTAL_RUNS:
                    coord_v_run = fx.idx2crd(r, layout_v_runs)
                    tile = fx.get(coord_v_run, 0)
                    d = fx.get(coord_v_run, 1)
                    vals = [v_lds_view[tile * x + j, d] for j in range_constexpr(x)]
                    vec_x = fx.Vector.from_elements(vals, CACHE_FX_TYPE)
                    dst_v = fx.slice(v_cache_block, (head, tile, d, None))
                    fx.ptr_store(vec_x, fx.get_iter(dst_v))
        else:
            # Generic HIP-compatible scatter for arbitrary/decode mappings.
            # Duplicate slots have the same last-writer-unspecified semantics
            # as the production token-wise kernel.
            for it in range_constexpr(SCATTER_ITERS):
                elem = t + KV_THREADS * it
                if elem < SCATTER_ELEMS:
                    coord_stage = fx.idx2crd(elem, layout_stage)
                    token_local = fx.get(coord_stage, 0)
                    d = fx.get(coord_stage, 1)
                    tok = tok0 + fx.Int32(token_local)
                    if tok < num_tokens:
                        slot = slot_mapping[tok]
                        if slot >= 0:
                            block_id = slot // block_size
                            block_off = slot % block_size
                            k_block_base = block_id * k_cache_block_stride
                            v_block_base = block_id * v_cache_block_stride
                            k_cache_block = fx.Tensor(
                                fx.make_view(
                                    fx.get_iter(k_cache) + k_block_base, layout_k_cache
                                )
                            )
                            v_cache_block = fx.Tensor(
                                fx.make_view(
                                    fx.get_iter(v_cache) + v_block_base, layout_v_cache
                                )
                            )
                            k_cache_block[head, d // x, block_off, d % x] = k_lds_view[
                                token_local, d
                            ]
                            v_cache_block[head, block_off // x, d, block_off % x] = (
                                v_lds_view[token_local, d]
                            )

    @flyc.jit
    def launch(
        qkv: fx.Tensor,
        positions_storage: fx.Tensor,
        cos_sin: fx.Tensor,
        k_norm_w: fx.Tensor,
        k_cache: fx.Tensor,
        v_cache: fx.Tensor,
        slot_mapping: fx.Tensor,
        k_scale: fx.Tensor,
        v_scale: fx.Tensor,
        k_out: fx.Tensor,
        v_out: fx.Tensor,
        num_tokens: fx.Int32,
        num_page_blocks: fx.Int32,
        page_block_offset: fx.Int32,
        positions_stride_0: fx.Int32,
        positions_stride_1: fx.Int32,
        stream: fx.Stream = fx.Stream(None),  # noqa: B008
    ):
        positions = fx.Tensor(
            fx.make_view(
                fx.get_iter(positions_storage),
                fx.make_layout(
                    (3, num_tokens), (positions_stride_0, positions_stride_1)
                ),
            )
        )
        k = kernel(
            qkv,
            positions,
            cos_sin,
            k_norm_w,
            k_cache,
            v_cache,
            slot_mapping,
            k_scale,
            v_scale,
            k_out,
            v_out,
            num_tokens,
            page_block_offset,
        )
        k.launch(
            grid=(H_KV, fx.Int64(num_page_blocks), 1),
            block=(KV_THREADS, 1, 1),
            stream=stream,
        )

    return launch


# Cached compilation and public API
@lru_cache(maxsize=32)
def _compile_q(
    *,
    head_size,
    num_heads_q,
    mrope_section,
    eps,
    is_interleaved,
    gemma_norm,
    match_hip,
    waves_per_block,
    head_iters,
):
    return _build_q_kernel(
        head_size=head_size,
        num_heads_q=num_heads_q,
        mrope_section=list(mrope_section),
        eps=eps,
        is_interleaved=is_interleaved,
        gemma_norm=gemma_norm,
        match_hip=match_hip,
        waves_per_block=waves_per_block,
        head_iters=head_iters,
    )


@lru_cache(maxsize=32)
def _compile_kv(
    *,
    num_heads_q,
    num_heads_kv,
    head_size,
    mrope_section,
    eps,
    block_size,
    x,
    emit_flat_kv,
    is_interleaved,
    gemma_norm,
    match_hip,
    cache_is_fp8,
    k_cache_block_stride,
    v_cache_block_stride,
):
    return _build_kv_kernel(
        num_heads_q=num_heads_q,
        num_heads_kv=num_heads_kv,
        head_size=head_size,
        mrope_section=list(mrope_section),
        eps=eps,
        block_size=block_size,
        x=x,
        emit_flat_kv=emit_flat_kv,
        is_interleaved=is_interleaved,
        gemma_norm=gemma_norm,
        match_hip=match_hip,
        cache_is_fp8=cache_is_fp8,
        k_cache_block_stride=k_cache_block_stride,
        v_cache_block_stride=v_cache_block_stride,
    )


def flydsl_fused_qk_norm_mrope_3d_cache_pts_quant_shuffle(
    qkv: fx.Tensor,
    qw: fx.Tensor,
    kw: fx.Tensor,
    cos_sin: fx.Tensor,
    positions: fx.Tensor,
    num_tokens: int,
    num_heads_q: int,
    num_heads_k: int,
    num_heads_v: int,
    head_size: int,
    is_neox_style: bool,
    mrope_section_: list[int],
    is_interleaved: bool,
    eps: float,
    q_out: fx.Tensor,
    k_cache: fx.Tensor,
    v_cache: fx.Tensor,
    slot_mapping: fx.Tensor,
    per_tensor_k_scale: fx.Tensor,
    per_tensor_v_scale: fx.Tensor,
    k_out: fx.Tensor | None,
    v_out: fx.Tensor | None,
    return_kv: bool,
    use_shuffle_layout: bool,
    block_size: int,
    x: int,
    rotary_dim: int = 0,
    gemma_norm: bool = False,
    match_hip: bool = True,
    stream: fx.Stream = fx.Stream(None),  # noqa: B008
) -> None:
    """FlyDSL drop-in for ``aiter.fused_qk_norm_mrope_3d_cache_pts_quant_shuffle``.

    Results are written in place to ``q_out``, the caches, and optionally
    ``k_out``/``v_out``.

    Args:
        qkv: ``[T, H_q+H_k+H_v, D]`` (or ``[T, (H_q+H_k+H_v)*D]``) bf16,
            contiguous.
        qw, kw: RMSNorm weights for Q / K, shape ``[D]``, bf16.
        cos_sin: ``[max_pos, D]`` bf16 (``cos = [:, :D/2]``, ``sin =
            [:, D/2:]``).
        positions: 3D-MRoPE position IDs, shape ``[3, T]``, int64. Both
            strides are honored.
        num_tokens: T; must equal ``qkv.shape[0]``.
        num_heads_q/k/v: Per-rank head counts; K and V counts must match.
        head_size: D; supported values are 64, 128, and 256.
        is_neox_style: must be ``True`` (NEOX pair layout ``(i, i+D/2)``;
            GPT-J pairing is unsupported).
        mrope_section_: 3-entry list summing to ``head_size // 2``
            (temporal/height/width).
        is_interleaved: Select interleaved or contiguous MRoPE sections.
        eps: RMSNorm epsilon.
        q_out: Contiguous bf16 output with ``T * H_q * D`` elements.
        k_cache, v_cache: shuffle-layout FP8 or bf16 paged KV cache buffers.
            Each block must be internally contiguous; dim 0 may be strided.
        slot_mapping: ``[T]`` int64 physical cache slots. Aligned contiguous
            pages use coalesced writes; other mappings use scatter writes.
        per_tensor_k_scale, per_tensor_v_scale: One-element float32 device
            tensors, read without a host synchronization.
        k_out, v_out: Optional contiguous outputs in the cache dtype, written
            when ``return_kv=True``.
        return_kv: see ``k_out``/``v_out``.
        use_shuffle_layout: Must be ``True``.
        block_size: KV cache page size (tokens/page).
        x: Innermost shuffle run: 16 for FP8 or 8 for bf16.
        rotary_dim: Must be ``0`` or ``head_size``; partial rotary is unsupported.
        gemma_norm: Use Gemma's ``1 + weight`` RMSNorm convention.
        match_hip: Round normalized values to bf16 before RoPE to match HIP;
            otherwise retain fp32 intermediates.
        stream: Launch stream; defaults to the current stream.
    """
    if not is_neox_style:
        raise NotImplementedError(
            "is_neox_style=False (GPT-J interleaved-pair RoPE) is not "
            "implemented; only the NEOX pair layout (i, i+D/2) is supported."
        )
    if not use_shuffle_layout:
        raise NotImplementedError(
            "use_shuffle_layout=False is not implemented -- this kernel only "
            "writes the shuffle-layout KV cache."
        )
    if rotary_dim not in (0, head_size):
        raise NotImplementedError(
            f"partial rotary (rotary_dim={rotary_dim} != head_size="
            f"{head_size}, and != 0) is not implemented; only full rotary "
            "is supported."
        )
    if k_cache.dtype != v_cache.dtype:
        raise TypeError("k_cache/v_cache must have the same dtype")
    cache_is_fp8 = k_cache.dtype == aiter_dtypes.fp8
    cache_is_bf16 = k_cache.dtype == aiter_dtypes.bf16
    if not cache_is_fp8 and not cache_is_bf16:
        raise TypeError(
            "k_cache/v_cache must use the architecture-native AITER FP8 dtype "
            f"({aiter_dtypes.fp8}) or {aiter_dtypes.bf16}, got {k_cache.dtype}"
        )
    expected_x = _RUN_BYTES // k_cache.element_size()
    if x != expected_x:
        raise ValueError(
            f"x={x} does not match the 16-byte shuffle run for "
            f"{k_cache.dtype}; expected x={expected_x}"
        )
    if num_heads_k != num_heads_v:
        raise ValueError(
            f"num_heads_k ({num_heads_k}) must equal num_heads_v "
            f"({num_heads_v}) -- K and V share one grid_x in the fused "
            "KV-cache-write kernel."
        )
    if head_size not in (64, 128, 256):
        raise ValueError(f"head_size ({head_size}) must be one of 64, 128, or 256")
    if len(mrope_section_) != 3:
        raise ValueError(
            f"mrope_section_ must have exactly 3 entries (3D-mrope), got "
            f"{len(mrope_section_)}"
        )
    if sum(mrope_section_) != head_size // 2:
        raise ValueError(
            f"sum(mrope_section_)={sum(mrope_section_)} must equal "
            f"head_size//2={head_size // 2}"
        )
    if (block_size % x != 0) or (block_size == 0):
        raise ValueError(f"block_size ({block_size}) must be a multiple of x ({x})")
    if (head_size * block_size) % 16 != 0:
        raise ValueError(
            f"head_size*block_size ({head_size * block_size}) must be a "
            "multiple of 16 (dwordx4 K-cache run size)"
        )
    lds_bytes = _align_up(2 * head_size * block_size * k_cache.element_size() + 4, 16)
    lds_limit = get_shared_memory_per_block(qkv.device)
    if lds_bytes > lds_limit:
        raise ValueError(
            f"head_size={head_size} x block_size={block_size} needs "
            f"{lds_bytes} B of LDS staging, "
            f"exceeding the {lds_limit} B hardware limit."
        )
    if qkv.dtype != aiter_dtypes.bf16:
        raise TypeError(f"qkv must be bf16, got {qkv.dtype}")
    if qw.dtype != aiter_dtypes.bf16 or kw.dtype != aiter_dtypes.bf16:
        raise TypeError("qw/kw must be bf16")
    if positions.dtype != aiter_dtypes.i64:
        raise TypeError(f"positions must be int64, got {positions.dtype}")
    if positions.shape != (3, num_tokens):
        raise ValueError(
            f"positions shape {tuple(positions.shape)} != (3, {num_tokens})"
        )
    if slot_mapping.dtype != aiter_dtypes.i64:
        raise TypeError(f"slot_mapping must be int64, got {slot_mapping.dtype}")
    if slot_mapping.shape != (num_tokens,):
        raise ValueError(
            f"slot_mapping shape {tuple(slot_mapping.shape)} != ({num_tokens},)"
        )
    if (
        per_tensor_k_scale.dtype != torch.float32
        or per_tensor_v_scale.dtype != torch.float32
    ):
        raise TypeError("per_tensor_k_scale/per_tensor_v_scale must be float32")
    if per_tensor_k_scale.numel() != 1 or per_tensor_v_scale.numel() != 1:
        raise ValueError(
            "per_tensor_k_scale/per_tensor_v_scale must contain one element"
        )
    if return_kv and (k_out is None or v_out is None):
        raise ValueError("return_kv=True requires k_out and v_out to be provided")

    H_Q, H_K, H_V, D = num_heads_q, num_heads_k, num_heads_v, head_size
    total_heads = H_Q + H_K + H_V
    if num_tokens < 0:
        raise ValueError(f"num_tokens must be non-negative, got {num_tokens}")
    if qkv.dim() == 3:
        expected_qkv_shape = (num_tokens, total_heads, D)
    elif qkv.dim() == 2:
        expected_qkv_shape = (num_tokens, total_heads * D)
    else:
        raise ValueError(
            "qkv must be 2-D [T, (H_q+H_k+H_v)*D] or 3-D [T, H_q+H_k+H_v, D]"
        )
    if tuple(qkv.shape) != expected_qkv_shape:
        raise ValueError(
            f"qkv shape {tuple(qkv.shape)} != expected {expected_qkv_shape}"
        )
    if not qkv.is_contiguous():
        raise ValueError("qkv must be contiguous")
    qkv_flat = qkv.view(num_tokens, total_heads, D)
    if qw.shape != (D,) or kw.shape != (D,):
        raise ValueError(f"qw/kw must both have shape ({D},)")
    if not qw.is_contiguous() or not kw.is_contiguous():
        raise ValueError("qw/kw must be contiguous")
    if (
        cos_sin.dtype != aiter_dtypes.bf16
        or cos_sin.dim() != 2
        or cos_sin.shape[1] != D
    ):
        raise TypeError(f"cos_sin must be 2-D bf16 with trailing dimension {D}")
    if not cos_sin.is_contiguous():
        raise ValueError("cos_sin must be contiguous")
    if q_out.dtype != aiter_dtypes.bf16 or q_out.numel() != num_tokens * H_Q * D:
        raise ValueError(f"q_out must be bf16 with {num_tokens * H_Q * D} elements")
    if not q_out.is_contiguous():
        raise ValueError("q_out must be contiguous")
    if not _is_contiguous_from_dim1(k_cache) or not _is_contiguous_from_dim1(v_cache):
        raise ValueError(
            "k_cache/v_cache must be contiguous within each block (dims >= 1)"
        )
    if not positions.is_cuda or not slot_mapping.is_cuda:
        raise ValueError("positions and slot_mapping must be device tensors")
    tensors = [
        qw,
        kw,
        cos_sin,
        positions,
        q_out,
        k_cache,
        v_cache,
        slot_mapping,
        per_tensor_k_scale,
        per_tensor_v_scale,
    ]
    if any(t.device != qkv.device for t in tensors):
        raise ValueError("all tensor arguments must be on the same device as qkv")
    if return_kv:
        if k_out.dtype != k_cache.dtype or v_out.dtype != v_cache.dtype:
            raise TypeError("k_out/v_out dtype must match the corresponding cache")
        if k_out.numel() != num_tokens * H_K * D:
            raise ValueError(f"k_out must contain {num_tokens * H_K * D} elements")
        if v_out.numel() != num_tokens * H_V * D:
            raise ValueError(f"v_out must contain {num_tokens * H_V * D} elements")
        if not k_out.is_contiguous() or not v_out.is_contiguous():
            raise ValueError("k_out/v_out must be contiguous")
        if k_out.device != qkv.device or v_out.device != qkv.device:
            raise ValueError("k_out/v_out must be on the same device as qkv")

    # A zero-token call has no valid Q or KV launch grid. Validation above is
    # still performed so malformed empty inputs fail consistently.
    if num_tokens == 0:
        return

    q_waves_per_block, q_head_iters = _q_launch_config(H_Q)
    q_launch = _compile_q(
        head_size=D,
        num_heads_q=H_Q,
        mrope_section=tuple(mrope_section_),
        eps=eps,
        is_interleaved=is_interleaved,
        gemma_norm=gemma_norm,
        match_hip=match_hip,
        waves_per_block=q_waves_per_block,
        head_iters=q_head_iters,
    )
    # FlyDSL cannot auto-adapt a layout-dynamic tensor when none of its axes
    # has unit stride (for example positions_storage[:, ::2]). Pass a
    # unit-stride view of the same storage and reconstruct the original 2-D
    # layout in the JIT launchers without copying.
    positions_storage_elems = (
        positions.untyped_storage().nbytes() // positions.element_size()
        - positions.storage_offset()
    )
    positions_storage = positions.as_strided((positions_storage_elems,), (1,))
    positions_stride_0 = positions.stride(0)
    positions_stride_1 = positions.stride(1)
    q_out_view = q_out.view(num_tokens, H_Q, D)
    for token_offset in range(0, num_tokens, _MAX_GRID_Y):
        chunk_tokens = min(_MAX_GRID_Y, num_tokens - token_offset)
        _run_compiled(
            q_launch,
            qkv_flat,
            positions_storage,
            cos_sin,
            qw,
            q_out_view,
            chunk_tokens,
            token_offset,
            positions_stride_0,
            positions_stride_1,
            stream,
        )

    packed_block_elems = H_K * D * block_size
    # Preserve the historical shape-agnostic behavior for contiguous flat
    # buffers. For non-contiguous cache views, dim 0 denotes the physical
    # cache block and its stride includes any interleaved plane/padding.
    k_cache_block_stride = (
        packed_block_elems if k_cache.is_contiguous() else k_cache.stride(0)
    )
    v_cache_block_stride = (
        packed_block_elems if v_cache.is_contiguous() else v_cache.stride(0)
    )
    for name, stride in (
        ("k_cache", k_cache_block_stride),
        ("v_cache", v_cache_block_stride),
    ):
        if stride < packed_block_elems:
            raise ValueError(
                f"{name} block stride ({stride}) is smaller than one packed "
                f"shuffle-layout block ({packed_block_elems} elements)"
            )
        if (stride * k_cache.element_size()) % _RUN_BYTES != 0:
            raise ValueError(
                f"{name} block stride ({stride} elements) must preserve "
                f"{_RUN_BYTES}-byte cache-run alignment"
            )

    kv_launch = _compile_kv(
        num_heads_q=H_Q,
        num_heads_kv=H_K,
        head_size=D,
        mrope_section=tuple(mrope_section_),
        eps=eps,
        block_size=block_size,
        x=x,
        emit_flat_kv=return_kv,
        is_interleaved=is_interleaved,
        gemma_norm=gemma_norm,
        match_hip=match_hip,
        cache_is_fp8=cache_is_fp8,
        k_cache_block_stride=k_cache_block_stride,
        v_cache_block_stride=v_cache_block_stride,
    )
    # Include a final partial page; the kernel handles it with guarded staging
    # and scatter writes.
    num_page_blocks = _ceil_div(num_tokens, block_size)
    k_out_arg = (
        k_out.view(num_tokens, H_K, D) if return_kv else k_cache.new_empty((1, 1, 1))
    )
    v_out_arg = (
        v_out.view(num_tokens, H_V, D) if return_kv else v_cache.new_empty((1, 1, 1))
    )
    for page_block_offset in range(0, num_page_blocks, _MAX_GRID_Y):
        chunk_page_blocks = min(_MAX_GRID_Y, num_page_blocks - page_block_offset)
        _run_compiled(
            kv_launch,
            qkv_flat,
            positions_storage,
            cos_sin,
            kw,
            k_cache,
            v_cache,
            slot_mapping,
            per_tensor_k_scale.reshape(1),
            per_tensor_v_scale.reshape(1),
            k_out_arg,
            v_out_arg,
            num_tokens,
            chunk_page_blocks,
            page_block_offset,
            positions_stride_0,
            positions_stride_1,
            stream,
        )
