# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""FlyDSL popularity and min-unique finalize kernels for HERD routing."""

from functools import cache

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import Float32, Int32, Int64, const_expr, gpu, range_constexpr
from flydsl.expr import rocdl as fly_rocdl

from aiter.ops.flydsl.kernels.kernels_common import (
    atomic_add_i32,
    kernel_signature,
    uint32_to_int32,
)
from aiter.ops.flydsl.kernels.tensor_shim import buf_copy_atom, ptr_buf_tensor

_FINALIZE_THREADS = 256
_FUSED_THREADS = 1024
# Per-lane register tile of the fused selector: E / lanes-per-row scores.
_FUSED_MAX_EXPERTS_PER_LANE = 16
# Sorted keys each lane keeps from its one pass over its scores.
_FUSED_LANE_DEPTH = 3
_INT32_MIN = -2147483648
_KEY_HI = uint32_to_int32(0xFFFF0000)
_KEY_LO = 0xFFFF
_KEY_NAN = uint32_to_int32(0x7FFF0000)
_NAN_HALVES = uint32_to_int32(0x80008000)


def _key_value(key):
    """The fp32 score a fused-selector key was built from."""
    hi = key & Int32(_KEY_HI)
    sign = hi >> Int32(31)
    mag = (hi ^ sign) - sign
    return (mag | (sign & Int32(_INT32_MIN))).bitcast(Float32)


def _tree_max(xs):
    while len(xs) > 1:
        xs = [fx.max(xs[i], xs[i + 1]) for i in range(0, len(xs) - 1, 2)] + (
            [xs[-1]] if len(xs) % 2 else []
        )
    return xs[0]


def _kept_routes(ids, values, pops, renormalize, apply_softmax, routed_scaling_factor):
    """Drop the least popular of K+1 candidates and weight the kept K.

    The drop is the argmin over (popularity, value, expert id). Returns per
    candidate whether it is kept, its output slot (rank by ascending expert
    id among the kept), and its weight.
    """
    kp1 = len(ids)
    drop = Int32(0)
    drop_id = ids[0]
    drop_value = values[0]
    drop_pop = pops[0]
    for j in range_constexpr(1, kp1):
        take = (pops[j] < drop_pop) | (
            (pops[j] == drop_pop)
            & (
                (values[j] < drop_value)
                | ((values[j] == drop_value) & (ids[j] < drop_id))
            )
        )
        drop = take.select(Int32(j), drop)
        drop_id = take.select(ids[j], drop_id)
        drop_value = take.select(values[j], drop_value)
        drop_pop = take.select(pops[j], drop_pop)
    keeps = [drop != Int32(j) for j in range_constexpr(kp1)]

    if apply_softmax:
        kept_max = Float32(-3.0e38)
        for j in range_constexpr(kp1):
            kept_max = fx.max(kept_max, keeps[j].select(values[j], Float32(-3.0e38)))
        values = [fx.exp(values[j] - kept_max) for j in range_constexpr(kp1)]
    kept_sum = Float32(0.0)
    for j in range_constexpr(kp1):
        kept_sum = kept_sum + keeps[j].select(values[j], Float32(0.0))
    if apply_softmax:
        scale = routed_scaling_factor / kept_sum
    elif renormalize:
        scale = routed_scaling_factor / fx.max(kept_sum, Float32(1e-20))
    else:
        scale = routed_scaling_factor

    ranks = []
    for j in range_constexpr(kp1):
        rank = Int32(0)
        for other in range_constexpr(kp1):
            before = keeps[other] & (ids[other] < ids[j])
            rank = rank + before.select(Int32(1), Int32(0))
        ranks.append(rank)
    weights = [values[j] * scale for j in range_constexpr(kp1)]
    return keeps, ranks, weights


@cache
def build_herd_finalize_module(
    topk: int,
    num_experts: int,
    renormalize: bool,
    round_to_bf16: bool,
    block_threads: int = _FINALIZE_THREADS,
):
    """Build the fused popularity, min-unique, and Top-K output launcher."""
    if topk < 1 or topk + 1 > num_experts:
        raise ValueError(
            f"invalid HERD geometry: topk={topk}, num_experts={num_experts}"
        )
    if block_threads not in (64, 128, 256, 512, 1024):
        raise ValueError(f"unsupported HERD block size {block_threads}")
    kp1 = topk + 1

    @fx.struct
    class SharedStorage:
        popularity: fx.Array[Int32, num_experts, 16]

    @flyc.kernel(
        name="herd_finalize_"
        + kernel_signature(
            k=topk,
            e=num_experts,
            rn=renormalize,
            rb=round_to_bf16,
            blk=block_threads,
        ),
        known_block_size=[block_threads, 1, 1],
    )
    def finalize_kernel(
        candidate_ids: fx.Tensor,
        candidate_values: fx.Tensor,
        topk_weights: fx.Tensor,
        topk_ids: fx.Tensor,
        topk_weights_stride: Int32,
        topk_ids_stride: Int32,
        rows: Int32,
        routed_scaling_factor: Float32,
    ):
        tid = fx.thread_idx.x
        storage = fx.SharedAllocator().allocate(SharedStorage)
        popularity = storage.popularity.peek().view(fx.make_layout(num_experts, 1))
        topk_weights_flat = ptr_buf_tensor(topk_weights, elem=Float32)
        topk_ids_flat = ptr_buf_tensor(topk_ids, elem=Int32)

        for expert in range(tid, Int32(num_experts), Int32(block_threads)):
            popularity[expert] = Int32(0)
        gpu.barrier()

        total = rows * Int32(kp1)
        for idx in range(tid, total, Int32(block_threads)):
            candidate_row = idx // Int32(kp1)
            candidate_col = idx - candidate_row * Int32(kp1)
            expert = candidate_ids[candidate_row, candidate_col]
            atomic_add_i32(popularity, Int32(1), expert, "workgroup")
        gpu.barrier()

        row = tid
        if row < rows:
            ids = [candidate_ids[row, j] for j in range_constexpr(kp1)]
            values = [candidate_values[row, j] for j in range_constexpr(kp1)]
            pops = [popularity[ids[j]] for j in range_constexpr(kp1)]
            keeps, ranks, weights = _kept_routes(
                ids, values, pops, renormalize, False, routed_scaling_factor
            )
            for j in range_constexpr(kp1):
                if keeps[j]:
                    weight = weights[j]
                    if const_expr(round_to_bf16):
                        weight = weight.to(fx.BFloat16).to(Float32)
                    topk_ids_flat[row * topk_ids_stride + ranks[j]] = ids[j]
                    topk_weights_flat[row * topk_weights_stride + ranks[j]] = weight

    @flyc.jit
    def launch_finalize(
        candidate_ids: fx.Tensor,
        candidate_values: fx.Tensor,
        topk_weights: fx.Tensor,
        topk_ids: fx.Tensor,
        topk_weights_stride: fx.Int32,
        topk_ids_stride: fx.Int32,
        rows: Int32,
        routed_scaling_factor: Float32,
        stream: fx.Stream,
    ):
        finalize_kernel(
            candidate_ids,
            candidate_values,
            topk_weights,
            topk_ids,
            topk_weights_stride,
            topk_ids_stride,
            rows,
            routed_scaling_factor,
        ).launch(
            grid=(1, 1, 1),
            block=(block_threads, 1, 1),
            stream=stream,
        )

    return launch_finalize


@cache
def build_herd_fused_softmax_module(
    topk: int,
    num_experts: int,
    max_rows: int,
    wave_size: int,
    block_threads: int = _FUSED_THREADS,
    depth: int = _FUSED_LANE_DEPTH,
):
    """Build a one-launch flat-topk softmax HERD router over BF16 logits.

    Mirrors Triton's `routing_minunique`: Top-(K+1) on the raw logits with
    batch popularity, then the min-unique drop and a softmax over the kept K.
    One workgroup holds the whole batch, so the popularity never leaves LDS and
    the two-kernel candidate + finalize split costs a launch instead.

    The workgroup is bound by instruction issue on its one CU, so the
    selection is built to issue little:

    * Every row gets a group of `lanes` lanes; lane s holds experts
      [s * epl, (s + 1) * epl) as 32-bit keys, the order-preserving BF16 bits
      above the inverted expert id, so one integer max picks the score and
      breaks ties toward the lower id.
    * Each lane keeps its sorted top-`depth` from a single pass; the K+1
      rounds then compare lane heads only. A lane that runs dry refills from
      its remaining keys under a wave-uniform branch, which is rare: it needs
      `depth` of a row's winners to share a lane.
    * The rounds touch no LDS. Lane j of a group then stores the row's j-th
      winner and adds its popularity in one instruction per wave.
    """
    kp1 = topk + 1
    if topk < 1 or kp1 > num_experts:
        raise ValueError(
            f"invalid HERD geometry: topk={topk}, num_experts={num_experts}"
        )
    if wave_size not in (32, 64):
        raise ValueError(f"wave size must be 32 or 64, got {wave_size}")
    if block_threads % wave_size or not max_rows <= block_threads <= 1024:
        raise ValueError(f"unsupported HERD block size {block_threads}")
    lanes = min(wave_size, block_threads // max_rows)
    if lanes & (lanes - 1) or num_experts % lanes or lanes < kp1:
        raise ValueError(
            f"{block_threads} threads over {max_rows} rows leave {lanes} lanes per "
            f"row, which must be a power of two dividing num_experts={num_experts} "
            f"and at least K+1={kp1}"
        )
    epl = num_experts // lanes
    vec = min(epl, 8)
    if epl > _FUSED_MAX_EXPERTS_PER_LANE or epl % vec or vec % 2:
        raise ValueError(
            f"num_experts={num_experts} over {lanes} lanes per row gives {epl} "
            f"experts per lane; the fused selector takes an even count up to "
            f"{_FUSED_MAX_EXPERTS_PER_LANE} in whole 16-byte loads"
        )
    depth = min(depth, epl, kp1)
    shifts = [lanes >> (s + 1) for s in range(lanes.bit_length() - 1)]
    slot_stride = max(8, kp1)
    LaneMask = Int64 if wave_size == 64 else Int32

    @fx.struct
    class SharedStorage:
        popularity: fx.Array[Int32, num_experts, 16]
        cand_keys: fx.Array[Int32, max_rows * slot_stride, 16]

    @flyc.kernel(
        name="herd_fused_softmax_"
        + kernel_signature(
            k=topk,
            e=num_experts,
            m=max_rows,
            wave=wave_size,
            blk=block_threads,
            d=depth,
        ),
        known_block_size=[block_threads, 1, 1],
    )
    def fused_kernel(
        gating: fx.Tensor,
        topk_weights: fx.Tensor,
        topk_ids: fx.Tensor,
        topk_weights_stride: Int32,
        topk_ids_stride: Int32,
        rows: Int32,
        routed_scaling_factor: Float32,
    ):
        tid = fx.thread_idx.x
        sub = tid % Int32(lanes)
        group_row = tid // Int32(lanes)
        width = Int32(wave_size)
        neg = Int32(_INT32_MIN)
        storage = fx.SharedAllocator().allocate(SharedStorage)
        popularity = storage.popularity.peek().view(fx.make_layout(num_experts, 1))
        cand_keys = storage.cand_keys.peek().view(
            fx.make_layout(max_rows * slot_stride, 1)
        )
        topk_weights_flat = ptr_buf_tensor(topk_weights, elem=Float32)
        topk_ids_flat = ptr_buf_tensor(topk_ids, elem=Int32)

        # Groups past the batch read the last row so every lane runs the same
        # shuffles; only their writes are masked.
        live = group_row < rows
        src_row = fx.min(group_row, rows - Int32(1))
        col0 = sub * Int32(epl)
        # One descriptor over the whole batch: groups of a wave read different
        # rows, and a per-row descriptor would not be wave-uniform.
        score_units = ptr_buf_tensor(gating, elem=fx.BFloat16, unit_elems=vec)
        unit0 = src_row * Int32(num_experts // vec) + sub * Int32(epl // vec)
        fragments = []
        for c in range_constexpr(epl // vec):
            src = fx.slice(score_units, (unit0 + Int32(c), None))
            fragment = fx.make_fragment_like(src)
            fx.copy(buf_copy_atom(vec * 2, fx.BFloat16), src, fragment)
            fragments.append(fragment)

        for expert in range(tid, Int32(num_experts), Int32(block_threads)):
            popularity[expert] = Int32(0)

        # Order-preserving key of each BF16 score, two per dword: magnitudes
        # negated for negative scores, so -0 and +0 meet at 0.
        id_base = Int32(_KEY_LO) - col0
        keys = [None] * epl
        nan_bits = []
        nan_any = Int32(0)
        for c in range_constexpr(epl // vec):
            v = fx.Vector(fx.memref_load_vec(fragments[c])).bitcast(fx.Int16)
            sign = v >> fx.Int16(15)
            mag = v & fx.Int16(0x7FFF)
            ordered = ((mag ^ sign) - sign).bitcast(Int32)
            # Bit 15 of a half is set iff its magnitude is above +inf.
            nan = (mag.bitcast(Int32) + Int32(0x007F007F)) & Int32(_NAN_HALVES)
            for p in range_constexpr(vec // 2):
                nan_bits.append(Int32(nan[p]))
                nan_any = nan_any | Int32(nan[p])
                i = c * vec + 2 * p
                dw = Int32(ordered[p])
                keys[i] = (dw << Int32(16)) | (id_base - Int32(i))
                keys[i + 1] = (dw & Int32(_KEY_HI)) | (id_base - Int32(i + 1))
        # NaN ranks above every number and all NaNs tie, as in ord_signed_f32.
        key_vec = fx.Vector.from_elements(keys, Int32)
        if LaneMask(fly_rocdl.ballot(LaneMask.ir_type, nan_any != Int32(0))) != (
            LaneMask(0)
        ):
            fixed = []
            for i in range_constexpr(epl):
                bit = Int32(0x8000 if i % 2 == 0 else _INT32_MIN)
                is_nan = (nan_bits[i // 2] & bit) != Int32(0)
                fixed.append(
                    is_nan.select(Int32(_KEY_NAN) | (id_base - Int32(i)), keys[i])
                )
            key_vec = fx.Vector.from_elements(fixed, Int32)
        keys = [Int32(key_vec[i]) for i in range_constexpr(epl)]

        heads = [neg] * depth
        for i in range_constexpr(epl):
            x = keys[i]
            heads = [fx.max(heads[0], x)] + [
                fx.max(heads[h], fx.min(heads[h - 1], x))
                for h in range_constexpr(1, depth)
            ]
        last = Int32(2147483647)

        winners = []
        for t in range_constexpr(kp1):
            if const_expr(t >= depth):
                dry = heads[0] == neg
                head = heads[0]
                if LaneMask(fly_rocdl.ballot(LaneMask.ir_type, dry)) != LaneMask(0):
                    below = [
                        (keys[i] < last).select(keys[i], neg)
                        for i in range_constexpr(epl)
                    ]
                    head = dry.select(_tree_max(below), head)
                heads[0] = head
            group_key = heads[0]
            for sh in range_constexpr(len(shifts)):
                group_key = fx.max(
                    group_key, group_key.shuffle_xor(Int32(shifts[sh]), width)
                )
            winners.append(group_key)
            mine = heads[0] == group_key
            last = mine.select(group_key, last)
            heads = [
                mine.select(heads[h + 1], heads[h]) for h in range_constexpr(depth - 1)
            ] + [mine.select(neg, heads[depth - 1])]

        my_key = winners[0]
        for j in range_constexpr(1, kp1):
            my_key = (sub == Int32(j)).select(winners[j], my_key)
        gpu.barrier()
        if live & (sub < Int32(kp1)):
            cand_keys[group_row * Int32(slot_stride) + sub] = my_key
            atomic_add_i32(
                popularity,
                Int32(1),
                Int32(_KEY_LO) - (my_key & Int32(_KEY_LO)),
                "workgroup",
            )
        gpu.barrier()

        row = tid
        if row < rows:
            base = row * Int32(slot_stride)
            ks = [cand_keys[base + Int32(j)] for j in range_constexpr(kp1)]
            ids = [
                Int32(_KEY_LO) - (ks[j] & Int32(_KEY_LO)) for j in range_constexpr(kp1)
            ]
            values = [_key_value(ks[j]) for j in range_constexpr(kp1)]
            pops = [popularity[ids[j]] for j in range_constexpr(kp1)]
            keeps, ranks, weights = _kept_routes(
                ids, values, pops, False, True, routed_scaling_factor
            )
            for j in range_constexpr(kp1):
                if keeps[j]:
                    topk_ids_flat[row * topk_ids_stride + ranks[j]] = ids[j]
                    topk_weights_flat[row * topk_weights_stride + ranks[j]] = weights[j]

    @flyc.jit
    def launch_fused(
        gating: fx.Tensor,
        topk_weights: fx.Tensor,
        topk_ids: fx.Tensor,
        topk_weights_stride: fx.Int32,
        topk_ids_stride: fx.Int32,
        rows: Int32,
        routed_scaling_factor: Float32,
        stream: fx.Stream,
    ):
        fused_kernel(
            gating,
            topk_weights,
            topk_ids,
            topk_weights_stride,
            topk_ids_stride,
            rows,
            routed_scaling_factor,
        ).launch(
            grid=(1, 1, 1),
            block=(block_threads, 1, 1),
            stream=stream,
        )

    return launch_fused
