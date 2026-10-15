# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""FlyDSL popularity and min-unique finalize kernels for HERD routing."""

from functools import cache

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import Float32, Int32, Int64, const_expr, gpu, range_constexpr
from flydsl.expr import rocdl as fly_rocdl
from flydsl.expr.typing import T
from flydsl.runtime.device import get_rocm_arch

from aiter.ops.flydsl.kernels.kernels_common import (
    atomic_add_i32,
    kernel_signature,
    uint32_to_int32,
)
from aiter.ops.flydsl.kernels.tensor_shim import (
    buf_base_i64,
    buf_copy_atom,
    ptr_buf_tensor,
)

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
# The sqrtsoftplus selector holds fp32 scores, so its keys are 64-bit and a lane
# may own up to 32 of them.
_FUSED_SQRTSOFTPLUS_MAX_EXPERTS_PER_LANE = 32
_INT64_MIN = -(1 << 63)
_FLT_MAX = 3.4028234663852886e38
_LOG2E = 1.4426950408889634
_LN2 = 0.6931471805599453
_SOFTPLUS_THRESHOLD = 20.0


def _key_value(key):
    """The fp32 score a fused-selector key was built from."""
    hi = key & Int32(_KEY_HI)
    sign = hi >> Int32(31)
    mag = (hi ^ sign) - sign
    return (mag | (sign & Int32(_INT32_MIN))).bitcast(Float32)


def _sqrt_softplus(x):
    """sqrt(softplus(x)) as aiter's HIP topk_gating computes it.

    +Inf clamps to FLT_MAX so the score stays finite; NaN passes through.
    """
    x = (x > Float32(_FLT_MAX)).select(Float32(_FLT_MAX), x)
    soft = fx.log2(Float32(1.0) + fx.exp2(x * Float32(_LOG2E))) * Float32(_LN2)
    return fx.sqrt((x > Float32(_SOFTPLUS_THRESHOLD)).select(x, soft))


def _shuffle_xor_i64(x, offset, width):
    hi = Int32(x >> Int64(32)).shuffle_xor(offset, width)
    lo = Int32(x).shuffle_xor(offset, width)
    return (Int64(hi) << Int64(32)) | (Int64(lo) & Int64(0xFFFFFFFF))


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


@cache
def sqrtsoftplus_workspace_elems(num_experts: int, max_rows: int, topk: int) -> int:
    """Int32 workspace of the fused sqrtsoftplus router; must start zeroed.

    [0, E) batch popularity, [E] arrival counter, then max_rows rows of
    candidates, each an int64 of its fp32 score above its expert id. The last
    workgroup returns the first two to zero.
    """
    return _sqrtsoftplus_cand_base(num_experts) + 2 * max_rows * max(8, topk + 1)


def _sqrtsoftplus_cand_base(num_experts: int) -> int:
    # In int32 elements; even, so the int64 candidates stay 8-byte aligned.
    return num_experts + 16


@cache
def build_herd_fused_sqrtsoftplus_module(
    topk: int,
    num_experts: int,
    max_rows: int,
    wave_size: int,
    renormalize: bool,
    rows_per_block: int = 4,
    depth: int = _FUSED_LANE_DEPTH,
):
    """Build a one-launch sqrtsoftplus HERD router over BF16 logits.

    Same contract as the topk_gating candidates + finalize pair: Top-(K+1) on
    sqrt(softplus(x)) + fp32 bias, batch popularity, then the min-unique drop
    and the kept K weighted by their unbiased score.

    Scoring every expert of every row is too much issue for one CU, so each
    workgroup selects the candidates of `rows_per_block` rows, one 32-lane
    group per row, and publishes them with the popularity to `workspace`. The
    last workgroup to arrive runs the drop for the whole batch and zeroes the
    popularity and the counter for the next call, so launches on one stream
    must not overlap.

    Everything crossing workgroups moves through agent-scope atomics, read
    back by atomics too: an agent-scope fence would write back or invalidate
    the whole L2 of a multi-XCD part, which costs more than the rest of the
    drop.

    Keys are 64-bit, the order-preserving fp32 biased score above the inverted
    expert id, because a 32-bit key would round the score at the K+1 boundary.
    Each candidate is published with its unbiased score, so the weights match
    the reference bit for bit without the drop re-reading the logits.
    """
    kp1 = topk + 1
    if topk < 1 or kp1 > num_experts:
        raise ValueError(
            f"invalid HERD geometry: topk={topk}, num_experts={num_experts}"
        )
    if wave_size not in (32, 64):
        raise ValueError(f"wave size must be 32 or 64, got {wave_size}")
    lanes = 32
    if num_experts % lanes or lanes < kp1:
        raise ValueError(
            f"{lanes} lanes per row need num_experts={num_experts} to be a "
            f"multiple of {lanes} and K+1={kp1} <= {lanes}"
        )
    block_threads = rows_per_block * lanes
    if block_threads % wave_size or block_threads > 1024:
        raise ValueError(f"unsupported HERD block size {block_threads}")
    epl = num_experts // lanes
    vec = 8
    bias_vec = 4
    if epl > _FUSED_SQRTSOFTPLUS_MAX_EXPERTS_PER_LANE or epl % vec:
        raise ValueError(
            f"num_experts={num_experts} over {lanes} lanes per row gives {epl} "
            f"experts per lane; the fused selector takes a multiple of {vec} up to "
            f"{_FUSED_SQRTSOFTPLUS_MAX_EXPERTS_PER_LANE}"
        )
    depth = min(depth, epl, kp1)
    shifts = [lanes >> (s + 1) for s in range(lanes.bit_length() - 1)]
    slot_stride = max(8, kp1)
    counter_slot = num_experts
    cand_base = _sqrtsoftplus_cand_base(num_experts)
    LaneMask = Int64 if wave_size == 64 else Int32
    split_wait_counters = str(get_rocm_arch()).startswith("gfx12")

    @fx.struct
    class SharedStorage:
        ticket: fx.Array[Int32, 1, 16]
        scores: fx.Array[Float32, rows_per_block * num_experts, 16]

    @flyc.kernel(
        name="herd_fused_sqrtsoftplus_"
        + kernel_signature(
            k=topk,
            e=num_experts,
            m=max_rows,
            wave=wave_size,
            rpb=rows_per_block,
            d=depth,
            rn=renormalize,
        ),
        known_block_size=[block_threads, 1, 1],
    )
    def fused_kernel(
        gating: fx.Tensor,
        correction_bias: fx.Tensor,
        topk_weights: fx.Tensor,
        topk_ids: fx.Tensor,
        workspace: fx.Tensor,
        topk_weights_stride: Int32,
        topk_ids_stride: Int32,
        rows: Int32,
        num_blocks: Int32,
        routed_scaling_factor: Float32,
    ):
        tid = fx.thread_idx.x
        sub = tid % Int32(lanes)
        local_row = tid // Int32(lanes)
        group_row = fx.block_idx.x * Int32(rows_per_block) + local_row
        width = Int32(wave_size)
        neg = Int64(_INT64_MIN)
        storage = fx.SharedAllocator().allocate(SharedStorage)
        ticket = storage.ticket.peek().view(fx.make_layout(1, 1))
        lds_scores = storage.scores.peek().view(
            fx.make_layout(rows_per_block * num_experts, 1)
        )
        topk_weights_flat = ptr_buf_tensor(topk_weights, elem=Float32)
        topk_ids_flat = ptr_buf_tensor(topk_ids, elem=Int32)
        ws = ptr_buf_tensor(workspace, elem=Int32)
        ws_base = buf_base_i64(workspace)

        def ws_atomic(op, index, value):
            ptr = fx.inttoptr(
                fx.PointerType.get(T.i32, fx.AddressSpace.Global, 4),
                ws_base + fx.Int64(index) * fx.Int64(4),
            )
            return Int32(
                op(
                    ptr,
                    value,
                    syncscope=fly_rocdl.SyncScope.Agent,
                    ordering=fx.AtomicOrdering.Monotonic,
                )
            )

        def cand_atomic(op, index, value):
            ptr = fx.inttoptr(
                fx.PointerType.get(T.i64, fx.AddressSpace.Global, 8),
                ws_base + fx.Int64(cand_base * 4) + fx.Int64(index) * fx.Int64(8),
            )
            return Int64(
                op(
                    ptr,
                    value,
                    syncscope=fly_rocdl.SyncScope.Agent,
                    ordering=fx.AtomicOrdering.Monotonic,
                )
            )

        def wait_memory():
            if const_expr(split_wait_counters):
                fly_rocdl.s_wait_loadcnt(0)
                fly_rocdl.s_wait_storecnt(0)
            else:
                fly_rocdl.s_waitcnt(0)

        # Groups past the batch read the last row so every lane runs the same
        # shuffles; only their writes are masked.
        live = group_row < rows
        src_row = fx.min(group_row, rows - Int32(1))
        col0 = sub * Int32(epl)
        score_units = ptr_buf_tensor(gating, elem=fx.BFloat16, unit_elems=vec)
        unit0 = src_row * Int32(num_experts // vec) + sub * Int32(epl // vec)
        fragments = []
        for c in range_constexpr(epl // vec):
            src = fx.slice(score_units, (unit0 + Int32(c), None))
            fragment = fx.make_fragment_like(src)
            fx.copy(buf_copy_atom(vec * 2, fx.BFloat16), src, fragment)
            fragments.append(fragment)
        bias_units = ptr_buf_tensor(correction_bias, elem=Float32, unit_elems=bias_vec)
        bias_unit0 = sub * Int32(epl // bias_vec)
        bias_fragments = []
        for c in range_constexpr(epl // bias_vec):
            src = fx.slice(bias_units, (bias_unit0 + Int32(c), None))
            fragment = fx.make_fragment_like(src)
            fx.copy(buf_copy_atom(bias_vec * 4, Float32), src, fragment)
            bias_fragments.append(fragment)

        # BF16 widens to fp32 by moving its bits to the high half of a dword.
        logits = []
        for c in range_constexpr(epl // vec):
            pairs = fx.Vector(fx.memref_load_vec(fragments[c])).bitcast(Int32)
            for p in range_constexpr(vec // 2):
                dw = Int32(pairs[p])
                logits.append((dw << Int32(16)).bitcast(Float32))
                logits.append((dw & Int32(_KEY_HI)).bitcast(Float32))
        biases = []
        for c in range_constexpr(epl // bias_vec):
            b = fx.Vector(fx.memref_load_vec(bias_fragments[c]))
            biases.extend(Float32(b[q]) for q in range_constexpr(bias_vec))

        id_base = Int32(_KEY_LO) - col0
        scores = [_sqrt_softplus(logits[i]) for i in range_constexpr(epl)]
        # The candidate writers below fetch their winner's score by expert id.
        score_row = local_row * Int32(num_experts) + col0
        for i in range_constexpr(epl):
            lds_scores[score_row + Int32(i)] = scores[i]
        keys = []
        for i in range_constexpr(epl):
            biased = scores[i] + biases[i]
            # NaN takes the lowest real key, so it loses to every number.
            biased = (biased == biased).select(biased, Float32(float("-inf")))
            bits = biased.bitcast(Int32)
            sign = bits >> Int32(31)
            ordered = ((bits & Int32(0x7FFFFFFF)) ^ sign) - sign
            keys.append((Int64(ordered) << Int64(32)) | Int64(id_base - Int32(i)))

        heads = [neg] * depth
        for i in range_constexpr(epl):
            x = keys[i]
            heads = [fx.max(heads[0], x)] + [
                fx.max(heads[h], fx.min(heads[h - 1], x))
                for h in range_constexpr(1, depth)
            ]
        last = Int64(0x7FFFFFFFFFFFFFFF)

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
                    group_key, _shuffle_xor_i64(group_key, Int32(shifts[sh]), width)
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
        my_id = Int32(_KEY_LO) - (Int32(my_key) & Int32(_KEY_LO))
        gpu.barrier()
        if live & (sub < Int32(kp1)):
            score = lds_scores[local_row * Int32(num_experts) + my_id]
            cand_atomic(
                fx.llvm.atomic_xchg,
                group_row * Int32(slot_stride) + sub,
                (Int64(score.bitcast(Int32)) << Int64(32)) | Int64(my_id),
            )
            ws_atomic(fx.llvm.atomic_add, my_id, Int32(1))
        # The arrival must not overtake this workgroup's candidates.
        wait_memory()
        gpu.barrier()
        if tid == Int32(0):
            ticket[0] = ws_atomic(fx.llvm.atomic_add, Int32(counter_slot), Int32(1))
        gpu.barrier()

        if ticket[0] == num_blocks - Int32(1):
            for row in range(tid, rows, Int32(block_threads)):
                cands = [
                    cand_atomic(
                        fx.llvm.atomic_add,
                        row * Int32(slot_stride) + Int32(j),
                        Int64(0),
                    )
                    for j in range_constexpr(kp1)
                ]
                ids = [Int32(c) for c in cands]
                values = [Int32(c >> Int64(32)).bitcast(Float32) for c in cands]
                pops = [
                    ws_atomic(fx.llvm.atomic_add, ids[j], Int32(0))
                    for j in range_constexpr(kp1)
                ]
                keeps, ranks, weights = _kept_routes(
                    ids, values, pops, renormalize, False, routed_scaling_factor
                )
                for j in range_constexpr(kp1):
                    if keeps[j]:
                        topk_ids_flat[row * topk_ids_stride + ranks[j]] = ids[j]
                        topk_weights_flat[row * topk_weights_stride + ranks[j]] = (
                            weights[j]
                        )
            gpu.barrier()
            for expert in range(tid, Int32(num_experts + 1), Int32(block_threads)):
                ws[expert] = Int32(0)

    @flyc.jit
    def launch_fused(
        gating: fx.Tensor,
        correction_bias: fx.Tensor,
        topk_weights: fx.Tensor,
        topk_ids: fx.Tensor,
        workspace: fx.Tensor,
        topk_weights_stride: fx.Int32,
        topk_ids_stride: fx.Int32,
        rows: Int32,
        num_blocks: Int32,
        routed_scaling_factor: Float32,
        stream: fx.Stream,
    ):
        fused_kernel(
            gating,
            correction_bias,
            topk_weights,
            topk_ids,
            workspace,
            topk_weights_stride,
            topk_ids_stride,
            rows,
            num_blocks,
            routed_scaling_factor,
        ).launch(
            grid=(num_blocks, 1, 1),
            block=(block_threads, 1, 1),
            stream=stream,
        )

    return launch_fused
