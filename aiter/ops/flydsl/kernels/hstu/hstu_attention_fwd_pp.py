# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""hstu_attention_fwd_pp - ping-pong HSTU forward (gfx942).

Same math as the fallback kernel, `hstu_attention_fwd.py`, scheduled so the two
waves sharing a SIMD sit in opposite phases:

    phase M: GEMM1 of tile i, GEMM2 of tile i-1, their LDS operand reads, and
             this group's K/V staging.
    phase V: the mask and the SiLU gate (LDS reads of the sigmoid table and the
             window bias only).

8 waves form two groups (waves i and i+4 share a SIMD) and `waves_per_eu=2`
keeps one workgroup per CU; a one-barrier skew keeps the groups in opposite
phases, so the matrix core and the VALU stay busy together. The kernel is
VALU-bound (the gate), so longer phases per barrier win.

* K and V are staged global -> VGPR -> ds_write (K XOR-swizzled, V transposed),
  so a load can span two phases. g0 stages K and g1 stages V through ONE
  unbranched load with a selected base pointer, which is why
  head_dim == hidden_dim is required.
* The staging tiling is derived on one group's 256 threads: the other group is
  in phase V when a tile is staged.
* sigmoid is read from a table in LDS, and a causal window without a contextual
  prefix is an additive bias loaded as GEMM1's initial accumulator, so phase V
  has no transcendental and, under a window, no mask.

Tiles: (512,32,8,2) for d=64 without a window; (256,32,8,2) under a window or
at d=128, where 512 spills. MI300X, bf16, B=1024 H=4 N=16384, against the
fallback: 1.49x at d=64, 1.13x at d=64 W=512, 2.21x at d=128.

Accuracy: the score stays fp32 from the MFMA to the gate; alpha*-log2e is folded
into the sigmoid table's index constant and alpha/N into the epilogue. sigmoid
comes from a nearest-node table (relative error < 4e-3, about bf16's rounding) that
ends in an exact 0, so silu's negative tail vanishes as it should. Compiled
with fast/unsafe FP math.
"""

import functools
import math

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import Vector as Vec
from flydsl.runtime.device import get_rocm_arch

from aiter.ops.flydsl.kernels.hstu.hstu_attention_common import (
    _LOG2E,
    MFMA_ELEMS_PER_LANE,
    MFMA_K,
    MFMA_LANE_K,
    MFMA_M,
    MFMA_N,
    NUM_GRID_GROUPS,
    WARP_SIZE,
    _arch_dma_params,
    _dtype_to_elem_type,
    _mfma_params_for_dim,
    bind_mfma_accs,
    exp2_f32,
)

# SLP vectorization off: a packed `v_pk_*_f32` waits for both halves' producers,
# which halves the independent chains in the latency-bound gate (the backward
# measured 342.9 ms with SLP off against 375.7 ms with it on).
_PP_COMPILE_HINTS = {"llvm_options": {"vectorize-slp": False}}


def phase_barrier():
    """Workgroup barrier the scheduler cannot move code across.

    The `sched_barrier` fences stop VALU from sinking into the next phase. Not
    `gpu.barrier()`: its implicit `lgkmcnt(0)` is placed by hand at the end of
    phase M instead of at every phase boundary.
    """
    rocdl.sched_barrier(0)
    rocdl.s_barrier()
    rocdl.sched_barrier(0)


def _bias_slot_map(max_attn_len: int, kv_subtiles: int):
    """Layout of the window-bias table: v = U / 16 -> slot.

    Returns (fold, jk, jt, br, n_slot, slot, vrep): `slot(v)` is v's slot,
    `vrep(s)` a v whose pattern slot s holds. When the all-keep run
    [kv_subtiles, jk] covers a whole KV tile it is folded into one slot.
    """
    W, KS = max_attn_len, kv_subtiles
    jk = (W - 15) // 16  # last v whose block is entirely kept
    jt = (W + 15) // 16  # last v whose block is not entirely dropped
    fold = jk >= KS
    if fold:
        br = jt + KS - jk

        def slot(v):
            return min(max(v, -1), KS) + 1 + min(max(v - jk, 0), br)

        def vrep(s):
            return s - 1 if s <= KS + 1 else jk + s - KS - 1

        return fold, jk, jt, br, KS + 2 + br, slot, vrep

    def slot(v):
        return min(max(v, -1), jt + KS) + 1

    def vrep(s):
        return s - 1

    return fold, jk, jt, 0, jt + KS + 2, slot, vrep


def _bias_pattern(v: int, max_attn_len: int) -> tuple:
    """Drop mask of the 16x16 block at v = U / 16, row-major (kv a, query b)."""
    return tuple(
        (16 * v + b - a) < 0 or (16 * v + b - a) > max_attn_len
        for a in range(MFMA_M)
        for b in range(MFMA_N)
    )


def _bias_slot_check_range(max_attn_len: int, kv_subtiles: int):
    """The v values where `_bias_slot_map` can disagree with the pattern.

    Folded, every v in [kv_subtiles, jk] maps to the one all-keep slot and
    every v - ng there is all-keep, so only the two edges need checking and the
    cost no longer grows with max_attn_len.
    """
    fold, jk, jt, *_ = _bias_slot_map(max_attn_len, kv_subtiles)
    full = range(-4, jt + kv_subtiles + 8)
    if not fold:
        return list(full)
    lo = range(-4, kv_subtiles + 5)
    hi = range(max(lo.stop, jk - kv_subtiles - 4), full.stop)
    return list(lo) + list(hi)


def _check_bias_slots(max_attn_len: int, kv_subtiles: int, vs) -> None:
    """Assert that every v in `vs` maps to a slot holding its own pattern."""
    _, _, _, _, n_slot, slot, vrep = _bias_slot_map(max_attn_len, kv_subtiles)
    for v in vs:
        s = slot(v)
        assert 0 <= s < n_slot, (v, s)
        assert all(
            _bias_pattern(v - ng, max_attn_len)
            == _bias_pattern(vrep(s) - ng, max_attn_len)
            for ng in range(kv_subtiles)
        ), (v, s)


def validate_hstu_attention_fwd_pp(
    num_heads: int,
    head_dim: int,
    hidden_dim: int,
    causal: bool,
    max_attn_len: int,
    contextual_seq_len: int,
    has_targets: bool,
    alpha: float,
    dtype_str: str,
    *,
    block_m: int,
    block_n: int,
    num_waves: int,
    waves_per_eu: int,
    arch: str | None = None,
) -> None:
    """Raise ValueError for any problem or tile this kernel cannot serve.

    The dispatcher falls back to `hstu_attention_fwd` only on this ValueError,
    so a geometry that builds but fails at launch must be refused here: a
    failed compiled launch is printed, not raised.
    """
    if arch is None:
        arch = get_rocm_arch()
    # The schedule is tuned to CDNA3's MFMA and transcendental rates.
    if not arch.startswith("gfx942"):
        raise ValueError(
            f"hstu attention fwd pp requires gfx942, got {arch!r} "
            "(use hstu_attention_fwd for other targets)"
        )
    if dtype_str != "bf16":
        raise ValueError(
            f"hstu attention fwd pp requires bf16, got {dtype_str!r} "
            "(use hstu_attention_fwd for f16)"
        )
    # One load serves K and V through a selected base pointer.
    if head_dim != hidden_dim:
        raise ValueError(
            f"hstu attention fwd pp requires head_dim == hidden_dim, got "
            f"{head_dim} != {hidden_dim}"
        )
    # Measured on gfx942 with flydsl 0.5.0: head_dim=192 does not tile the
    # staging pass, head_dim=256 and block_n=64 fail at launch
    # (hipErrorIllegalState), and block_n=16 asserts while building.
    if head_dim not in (64, 128):
        raise ValueError(
            f"hstu attention fwd pp requires head_dim in (64, 128), got {head_dim}"
        )
    if block_n != 32:
        raise ValueError(f"hstu attention fwd pp requires block_n == 32, got {block_n}")
    if num_waves != 8:
        raise ValueError(
            f"hstu attention fwd pp requires num_waves == 8 (2 per SIMD inside one "
            f"barrier domain), got {num_waves}"
        )
    if block_m % (num_waves * MFMA_M) != 0:
        raise ValueError(
            f"block_m {block_m} must be a multiple of num_waves*MFMA_M ({num_waves * MFMA_M})"
        )
    if block_m < 256:
        raise ValueError(
            f"hstu attention fwd pp requires block_m >= 256, got {block_m}. At 128 "
            "the kernel fits 4 waves/SIMD, so a second workgroup co-resides and "
            "the phase alternation no longer owns the SIMD."
        )
    if waves_per_eu < 0:
        raise ValueError(f"waves_per_eu must be non-negative, got {waves_per_eu}")
    if contextual_seq_len < 0:
        raise ValueError(
            f"contextual_seq_len must be non-negative, got {contextual_seq_len}"
        )
    if max_attn_len < 0:
        raise ValueError(f"max_attn_len must be non-negative, got {max_attn_len}")
    if num_heads <= 0:
        raise ValueError(f"num_heads must be positive, got {num_heads}")
    if not math.isfinite(alpha):
        raise ValueError(f"alpha must be finite, got {alpha}")


@functools.lru_cache(maxsize=16384)
def build_hstu_attention_fwd_pp(
    num_heads: int,
    head_dim: int,
    hidden_dim: int,
    causal: bool,
    max_attn_len: int,
    contextual_seq_len: int,
    has_targets: bool,
    alpha: float,
    dtype_str: str,
    *,
    block_m: int = 512,
    block_n: int = 32,
    num_waves: int = 8,
    waves_per_eu: int = 2,
):
    validate_hstu_attention_fwd_pp(
        num_heads,
        head_dim,
        hidden_dim,
        causal,
        max_attn_len,
        contextual_seq_len,
        has_targets,
        alpha,
        dtype_str,
        block_m=block_m,
        block_n=block_n,
        num_waves=num_waves,
        waves_per_eu=waves_per_eu,
    )

    BLOCK_M = block_m
    BLOCK_N = block_n
    NUM_WAVES = num_waves
    BLOCK_THREADS = NUM_WAVES * WARP_SIZE
    # g0 = waves 0-3 stages K, g1 = waves 4-7 stages V; waves i and i+4 share a
    # SIMD. The split is unbalanced (at d=64, K is 2 ds_write_b64 per lane and
    # V 8 transposed ds_write_b16), but staging half the rows of both in each
    # group measured 0.8% slower: two dwordx2 loads instead of one dwordx4.
    GRP_WAVES = NUM_WAVES // 2
    GRP_THREADS = GRP_WAVES * WARP_SIZE
    assert GRP_THREADS == 256
    ROWS_PER_WAVE = BLOCK_M // NUM_WAVES
    Q_SUBTILES = ROWS_PER_WAVE // MFMA_M
    KV_SUBTILES = BLOCK_N // MFMA_N
    WAVES_PER_EU = waves_per_eu
    MFMA_DIM_K, MFMA_DIM_LANE_K, MFMA_DIM_LANE_K_LOG2 = _mfma_params_for_dim(head_dim)

    # K LDS swizzle geometry.
    _, _, K_SWZ_ROWS, K_SWZ_SHIFT = _arch_dma_params()

    elem_dtype = _dtype_to_elem_type(dtype_str)
    has_window = max_attn_len > 0
    has_contextual = contextual_seq_len > 0

    K_STEPS = head_dim // MFMA_DIM_K
    D_CHUNKS = hidden_dim // MFMA_M  # GEMM2 output chunks per lane
    # head_dim is 64 or 128, so the K LDS row pitch is head_dim itself and the
    # XOR swizzle (period 64) stays within a row.

    # V is stored transposed, [d, kv], so GEMM2's B operand (4 consecutive kv)
    # is one ds_read_b64; the +8 pad spreads those reads across banks.
    V_T_STRIDE = BLOCK_N + 8

    # Staging tiling, derived on one group's threads (the other group is in
    # phase V): an 8-wide (dwordx4) load per lane, NUM_BATCHES_KV per tile.
    VEC_KV = 8
    assert (BLOCK_N * hidden_dim) % (GRP_THREADS * VEC_KV) == 0
    assert VEC_KV % MFMA_DIM_LANE_K == 0
    THREADS_PER_ROW_KV = hidden_dim // VEC_KV
    ROWS_PER_BATCH_KV = GRP_THREADS // THREADS_PER_ROW_KV
    assert BLOCK_N % ROWS_PER_BATCH_KV == 0
    NUM_BATCHES_KV = BLOCK_N // ROWS_PER_BATCH_KV

    # ---- LDS rotation slots ------------------------------------------------
    # Phase M(i) runs GEMM1(i) and GEMM2(i-1) as one burst, so it reads K(i)
    # and V(i-1) and writes K(i+1) and V(i): two slots per tensor, keyed on the
    # sub-trip position u (K reads u%2 and writes (u+1)%2; V reads (u+1)%2 and
    # writes u%2). Deeper prefetch is held in registers, not in a third slot.
    # One fx.Array field per slot keeps every access inside one LDS global.
    K_SLOTS = 2
    V_SLOTS = 2
    # Unrolled by the slot count so every sub-trip names a literal slot; loop
    # trip counts must stay multiples of UNROLL to keep the slot parity.
    UNROLL = 2
    assert UNROLL % K_SLOTS == 0 and UNROLL % V_SLOTS == 0

    # ---- The MFMA burst's unit stream --------------------------------------
    # A unit is one ds_read plus the Q_SUBTILES MFMAs that consume it:
    #   QK unit j -> (ng, ks) = (j % KV_SUBTILES, j // KV_SUBTILES)
    #   PV unit j -> (c,  ng) = (j % D_CHUNKS,    j // D_CHUNKS)
    # so consecutive units of one GEMM chain on different accumulators.
    N_QK_UNITS = KV_SUBTILES * K_STEPS
    N_PV_UNITS = D_CHUNKS * KV_SUBTILES
    assert N_QK_UNITS == N_PV_UNITS  # head_dim == hidden_dim
    N_UNITS = N_QK_UNITS + N_PV_UNITS

    # Rolling operand window: the MFMAs of step s consume the read issued at
    # step s - PREFETCH_DEPTH, so no MFMA waits on its own ds_read. Depths 1-4
    # measured within 0.2% of each other at (512,32).
    PREFETCH_DEPTH = 2
    # Window slots are indexed by step, not by unit (UNIT_ORDER is a
    # permutation), so any PREFETCH_DEPTH below WIN_RING is hazard-free.
    WIN_RING = 4
    assert 0 <= PREFETCH_DEPTH < WIN_RING <= N_UNITS

    # V slot rotation: vslot p holds d-chunks 1..D-1 of tile p and d-chunk 0 of
    # tile p+1. The tail of M(i) can then prefetch the next phase's leading
    # chunk-0 PV units from the slot it is already reading, so the cross-phase
    # prefetch needs no third slot. Only the tile a lane stages changes
    # (`kv_stage_bump`); the store target does not.
    V_ROT_DT = 1

    # ---- sigmoid(g) lookup table in LDS ------------------------------------
    # Replaces exp2 -> add -> rcp with one ds_read. Nearest node, h = 0.01125,
    # keeps sigmoid's relative error under bf16's 2^-8. Below g = -8 sigmoid
    # rounds to 1 in bf16; above g = +16, p = g*sigmoid is below bf16's noise
    # floor. 2135 entries plus a final sigmoid = 0 that every g past the range
    # clamps to, so silu's negative tail goes to 0 instead of growing with |s|
    # (the dropped |p| is at most 16/(1+2^16) in g units, under the table's own
    # error). Below the range the clamped index gives sigmoid = 1, so the gate
    # stays exact however negative g is; a table of p could not do that.
    _LUT_GLO, _LUT_GHI = -8.0, 16.0
    _LUT_H = 0.01125
    _LUT_N = math.ceil((_LUT_GHI - _LUT_GLO) / _LUT_H) + 1
    _LUT_SCALE = 1.0 / _LUT_H
    _LUT_BIAS = -_LUT_GLO / _LUT_H + 0.5  # +0.5 so truncation == round-to-nearest

    # ---- The windowed mask as an additive bias on GEMM1 --------------------
    # Under a causal window without a contextual prefix, a 16x16 (qg, ng)
    # sub-block's mask depends only on U = q_blk - kv_lo (a multiple of 16):
    # all-keep, all-drop, or one of at most three edge patterns. Those bias
    # tiles (0 kept, BIAS_HUGE dropped) are built once per workgroup in LDS
    # (N_BSLOT * KV_SUBTILES KB, 16 KB at W=512) and loaded as GEMM1's initial
    # accumulator. A dropped score hits an extra LUT entry holding sigmoid = 0,
    # so phase V runs the unmasked gate. Phase M(i) computes tile i+1's slot
    # rows in SALU under its MFMAs, and phase V(i) reads the tiles so they land
    # under the barrier. Target rows, whose window hangs off max_id rather than
    # the row, read the all-zero slot and are masked per element.
    BIAS_MASK = has_window and causal and not has_contextual
    # The accumulator holds the raw score s, and the table index grows with
    # s*alpha*-log2e, so a dropped score is pushed the opposite way to alpha.
    BIAS_HUGE = -math.copysign(2.0**40, alpha) if alpha != 0 else 0.0
    N_BINIT = KV_SUBTILES * Q_SUBTILES if BIAS_MASK else 0
    # The MFMA after which phase M starts the next tile's slot SALU; the first
    # QK MFMAs keep their shadows for their LDS reads. At (256,32), starting
    # after MFMA 2 measured 1.046x, after 4 or 8 1.065x.
    BIAS_SIDE_MFMA = 2 * Q_SUBTILES
    if BIAS_MASK:
        _KS = KV_SUBTILES
        BIAS_FOLD, _JK, _JT, _BR, N_BSLOT, _, _ = _bias_slot_map(max_attn_len, _KS)
        _check_bias_slots(max_attn_len, _KS, _bias_slot_check_range(max_attn_len, _KS))
        # One extra all-zero slot for target q-blocks.
        BIAS_KEEP_SLOT = N_BSLOT
        N_BSLOT += 1
        BIAS_ROWS = N_BSLOT * _KS * WARP_SIZE
    else:
        BIAS_ROWS = 1
    # One extra entry, sigmoid = 0, at the top clamp.
    _LUT_TAB = _LUT_N + 1
    _LUT_ROUNDS = (_LUT_TAB + BLOCK_THREADS - 1) // BLOCK_THREADS

    # Phase V issues a tile's LUT reads in two groups, the first GATE_SPLIT
    # elements and then the rest, so the first group's multiplies start while
    # the second group's reads drain. 8 gives the fewest s_waitcnt in phase V
    # (4, 12, 16 and 20 were all worse).
    GATE_SPLIT = 8

    # The burst's tail prefetches the next phase's first XPHASE_DEPTH units,
    # which must be chunk-0 PV units: only V is rotated.
    XPHASE_DEPTH = PREFETCH_DEPTH
    _PV_HEAD = [j for j in range(N_PV_UNITS) if (j % D_CHUNKS) < V_ROT_DT]
    assert (
        len(_PV_HEAD) >= XPHASE_DEPTH
    ), f"V_ROT_DT={V_ROT_DT} covers {len(_PV_HEAD)} PV units, need {XPHASE_DEPTH}"
    _HEAD = [("pv", j) for j in _PV_HEAD[:XPHASE_DEPTH]]
    # The rest alternates QK and PV 1:1, two independent accumulator chains
    # for the matrix pipe.
    _REST_PV = [("pv", j) for j in range(N_PV_UNITS) if ("pv", j) not in _HEAD]
    _REST_QK = [("qk", j) for j in range(N_QK_UNITS)]
    UNIT_ORDER = _HEAD + [
        spec
        for pair in zip(_REST_QK, _REST_PV + [None] * len(_REST_QK))
        for spec in pair
        if spec is not None
    ]
    assert len(UNIT_ORDER) == N_UNITS, (len(UNIT_ORDER), N_UNITS)
    assert sorted(UNIT_ORDER) == sorted(
        [("qk", j) for j in range(N_QK_UNITS)] + [("pv", j) for j in range(N_PV_UNITS)]
    ), "UNIT_ORDER must be a permutation of the units"
    assert all(
        k == "pv" and (j % D_CHUNKS) < V_ROT_DT for k, j in UNIT_ORDER[:XPHASE_DEPTH]
    )

    @fx.struct
    class SharedStorage:
        k0: fx.Array[elem_dtype, BLOCK_N * head_dim, 16]
        k1: fx.Array[elem_dtype, BLOCK_N * head_dim, 16]
        v0: fx.Array[elem_dtype, hidden_dim * V_T_STRIDE, 16]
        v1: fx.Array[elem_dtype, hidden_dim * V_T_STRIDE, 16]
        lut: fx.Array[fx.Float32, _LUT_TAB, 16]
        mbias: fx.Array[fx.Float32, BIAS_ROWS * MFMA_ELEMS_PER_LANE, 16]

    # ---- Device Kernel ----
    @flyc.kernel(known_block_size=[BLOCK_THREADS, 1, 1])
    def hstu_attention_fwd_pp_kernel(
        q: fx.Tensor,
        k: fx.Tensor,
        v: fx.Tensor,
        seq_offsets: fx.Tensor,
        num_targets: fx.Tensor,
        out: fx.Tensor,
        num_q_tiles: fx.Int32,
        hz_per_group: fx.Int32,
        hz_total: fx.Int32,
        inv_n: fx.Float32,
    ) -> None:
        compute_type = fx.Float32.ir_type
        c_zero_dim_pack = fx.full(MFMA_DIM_LANE_K, 0.0, elem_dtype).ir_value()

        # gfx942: QK and PV both use the 16x16x16 atom.
        dim_mfma_acc, mfma_acc = bind_mfma_accs(
            elem_dtype,
            (MFMA_DIM_K, MFMA_DIM_LANE_K),
            (MFMA_K, MFMA_LANE_K),
        )

        # ---- Thread / lane indices ----
        tid = fx.Int32(gpu.thread_idx.x)
        wave_id = tid // fx.Int32(WARP_SIZE)
        lane = tid % fx.Int32(WARP_SIZE)
        lane_mod_16 = lane % fx.Int32(MFMA_N)
        lane_div_16 = lane // fx.Int32(MFMA_N)

        # Wave-uniform group predicate; `readfirstlane` lets the divergence
        # analysis see that, so the duty branch is a scalar branch. Raw rocdl:
        # FlyDSL 0.5.0 has no typed uniformity API.
        tid_u = fx.Int32(rocdl.readfirstlane(fx.Int32.ir_type, tid))
        is_g0 = tid_u < fx.Int32(GRP_THREADS)
        grp_tid = tid - is_g0.select(fx.Int32(0), fx.Int32(GRP_THREADS))

        # ---- Group-major grid decode -> (batch_idx, head_idx, q_tile_idx) ----
        block_id = fx.Int32(gpu.block_idx.x)
        grid_group = block_id % fx.Int32(NUM_GRID_GROUPS)
        pos_in_group = block_id // fx.Int32(NUM_GRID_GROUPS)
        local_hz_idx = pos_in_group // num_q_tiles
        q_tile_idx = pos_in_group % num_q_tiles
        hz_idx = grid_group * hz_per_group + local_hz_idx

        # The last grid group is padded past batch*num_heads; padding blocks read
        # hz_idx 0 and get seq_len 0, so they store nothing.
        block_valid = hz_idx < hz_total
        hz_idx = block_valid.select(hz_idx, fx.Int32(0))
        batch_idx = hz_idx // fx.Int32(num_heads)
        head_idx = hz_idx % fx.Int32(num_heads)

        # ---- Sequence bounds + id clamps (target tail) ----
        seq_start = fx.Int32(seq_offsets[batch_idx])
        seq_len = fx.Int32(seq_offsets[batch_idx + fx.Int32(1)]) - seq_start
        seq_len = block_valid.select(seq_len, fx.Int32(0))

        num_target = fx.Int32(0)
        if has_targets:
            num_target = fx.Int32(num_targets[batch_idx])

        # Contextual shift first, then the target-tail clamp, as in to_id.
        max_id = seq_len
        if has_contextual:
            max_id = seq_len - fx.Int32(contextual_seq_len) + fx.Int32(1)
        if has_targets:
            max_id = (num_target > fx.Int32(0)).select(max_id - num_target, max_id)

        # g-wide in-row loads. (row, head) goes through the tensor's i64 layout
        # first, since row*row_stride can exceed int32; the in-row access is i32.
        def grouped_loader(t, dim, g):
            in_row = fx.make_layout((dim // g, g), (g, 1))

            def load(row_i64, head_val, colgrp):
                sub = t[row_i64, head_val, None]
                return fx.make_view(fx.get_iter(sub), in_row)[colgrp, None].load()

            return load

        # GEMM1's contraction order: step 2p+h, lane group g -> columns
        # 32*p + 8*g + 4*h, so each lane loads its Q row as 16-byte pieces
        # (half the instructions of the natural 16*ks + 4*g order) on the
        # prologue's critical path. K's LDS read follows the same map.
        assert MFMA_DIM_K == 16 and MFMA_DIM_LANE_K == 4 and K_STEPS % 2 == 0
        Q_VEC = 2 * MFMA_DIM_LANE_K

        def qk_col_grp(ks):
            """Column group (MFMA_DIM_LANE_K units) a lane feeds at k-step `ks`."""
            return fx.Int32((ks // 2) * 8 + ks % 2) + lane_div_16 * fx.Int32(2)

        q_load = grouped_loader(q, head_dim, Q_VEC)

        # g0 loads K and g1 loads V through ONE unbranched load: the loaded
        # vector is carried across the loop's yield, which a value defined
        # inside an `scf.if` would not dominate. k and v have identical layouts
        # (head_dim == hidden_dim), so the groups differ only by a wave-uniform
        # base-pointer delta added to one address chain.
        _kv_b0 = fx.get_iter(k[fx.Int64(0), fx.Int32(0), None])
        _kv_b1 = fx.get_iter(v[fx.Int64(0), fx.Int32(0), None])
        kv_base_delta = is_g0.select(
            fx.Int64(0),
            fx.Int64(fx.ptrtoint(_kv_b1)) - fx.Int64(fx.ptrtoint(_kv_b0)),
        )

        def kv_row_addr(row_i64):
            """This group's per-lane ROW address (base + row + head), as i64."""
            it = fx.get_iter(k[row_i64, head_idx, None])
            return fx.Int64(fx.ptrtoint(it)) + kv_base_delta

        # Buffer descriptor based at (seq_start, head) with `num_records` set to
        # this sequence's extent: a staged row past the sequence reads back zero
        # in hardware instead of a compare/select/cndmask chain in phase M, and
        # the carried address is a 32-bit voffset. The load still lands in VGPRs
        # (not `buffer_load ... lds`), since K and V are relayouted on store.
        assert (num_heads * head_dim) % VEC_KV == 0
        KV_ROW_GRPS = (num_heads * head_dim) // VEC_KV  # voffset groups per token
        kv_desc = fx.rocdl.make_buffer_ptr(
            fx.inttoptr(_kv_b0.type, kv_row_addr(fx.Int64(seq_start))),
            num_records_bytes=fx.Int64(seq_len) * fx.Int64(KV_ROW_GRPS * VEC_KV * 2),
        )
        kv_buf = fx.make_view(
            kv_desc,
            fx.make_layout((seq_len * fx.Int32(KV_ROW_GRPS), VEC_KV), (VEC_KV, 1)),
        )

        # LDS views indexed [row, col_grp, :] in MFMA_LANE_K-element groups, so
        # the hot path never divides. k_smem is K[kv, d] swizzled, v_smem is
        # V[d, kv] transposed. Taken here so they dominate both KV loops.
        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        k_layout = fx.make_layout(
            (BLOCK_N, head_dim // MFMA_DIM_LANE_K, MFMA_DIM_LANE_K),
            (head_dim, MFMA_DIM_LANE_K, 1),
        )
        v_layout = fx.make_layout(
            (hidden_dim, V_T_STRIDE // MFMA_LANE_K, MFMA_LANE_K),
            (V_T_STRIDE, MFMA_LANE_K, 1),
        )
        # Indexed by literal slot at every use, so each access is one fx.Array.
        k_smem = [lds.k0.view(k_layout), lds.k1.view(k_layout)]
        v_smem = [lds.v0.view(v_layout), lds.v1.view(v_layout)]
        lut_mem = lds.lut.view(fx.make_layout((_LUT_TAB, 1), (1, 1)))
        mbias_mem = lds.mbias.view(
            fx.make_layout((BIAS_ROWS, MFMA_ELEMS_PER_LANE), (MFMA_ELEMS_PER_LANE, 1))
        )

        # K swizzle col ^ ((row & (ROWS-1)) << SHIFT), applied to the group
        # index: with SHIFT >= LOG2 and group-aligned columns, XOR commutes
        # with the group division.
        assert K_SWZ_SHIFT >= MFMA_DIM_LANE_K_LOG2

        # V swizzle: the pad spreads the reads, but the transposed stores of
        # lanes VEC_KV apart in d collide 16-way, and no pad compatible with
        # the grouped view fixes that. XOR the kv group with d // VEC_KV
        # instead: stores drop to 4-way, reads stay 4-way, and the groups stay
        # inside [0, BLOCK_N / MFMA_LANE_K), clear of the pad.
        V_SWZ_GRPS = BLOCK_N // MFMA_LANE_K
        assert (
            V_SWZ_GRPS & (V_SWZ_GRPS - 1) == 0
        ), "V swizzle needs a power-of-2 group count"

        def v_swz_key(d_row):
            return (d_row // fx.Int32(VEC_KV)) & fx.Int32(V_SWZ_GRPS - 1)

        def v_swz_grp(d_row, kv_grp):
            return kv_grp ^ v_swz_key(d_row)

        def k_swz_grp(tile_row, col_grp):
            return col_grp ^ (
                (tile_row & fx.Int32(K_SWZ_ROWS - 1))
                << fx.Int32(K_SWZ_SHIFT - MFMA_DIM_LANE_K_LOG2)
            )

        def q_blk_off(qg):
            """First row of this wave's `qg`-th 16-row block, within the Q tile.

            Interleaved (wave w owns blocks w, w+8, ...) rather than contiguous,
            so every wave straddles the causal diagonal and the barrier after
            phase V waits on a balanced max over waves. On its own that costs
            1.2%; with `gate_tile_split` skipping blocks it nets 114.10 ms
            against 114.71 for the contiguous mapping.
            """
            return (wave_id + fx.Int32(qg * NUM_WAVES)) * fx.Int32(MFMA_M)

        def q_raw_load(qt_idx):
            """This wave's raw Q operand packs for Q tile `qt_idx`, flat [ks][qg].

            Out-of-sequence rows read row 0 and are zeroed at the consumer; a
            select here would pin the load's wait to its issue point."""
            pairs = {}
            for p in range_constexpr(K_STEPS // 2):
                for qg in range_constexpr(Q_SUBTILES):
                    row = qt_idx * fx.Int32(BLOCK_M) + q_blk_off(qg) + lane_mod_16
                    safe = (row < seq_len).select(seq_start + row, seq_start)
                    pairs[(p, qg)] = Vec(
                        q_load(
                            fx.Int64(safe),
                            head_idx,
                            fx.Int32(p * (32 // Q_VEC)) + lane_div_16,
                        )
                    )
            raws = []
            for ks in range_constexpr(K_STEPS):
                h = ks % 2
                for qg in range_constexpr(Q_SUBTILES):
                    v8 = pairs[(ks // 2, qg)]
                    raws.append(
                        Vec.from_elements(
                            [
                                v8[h * MFMA_DIM_LANE_K + j]
                                for j in range_constexpr(MFMA_DIM_LANE_K)
                            ],
                            elem_dtype,
                        ).ir_value()
                    )
            return raws

        O_VEC = 8  # bf16 per 16-byte store
        O_CH = hidden_dim // O_VEC  # 16-byte chunks per output row
        # One pass stages MFMA_M rows x O_PCH chunks = one 16-byte read per lane.
        O_PCH = WARP_SIZE // MFMA_M
        O_PASSES = O_CH // O_PCH
        O_PASS_CHUNKS = D_CHUNKS // O_PASSES  # GEMM2 d-chunks per pass
        O_SWZ = O_PCH - 1  # chunk XOR (row // 4): the 4 rows a write spans
        assert O_CH % O_PCH == 0 and D_CHUNKS % O_PASSES == 0 and O_PCH == 4
        _O_WAVE = MFMA_M * O_PCH * O_VEC  # scratch elements per wave
        assert BLOCK_N * head_dim >= GRP_WAVES * _O_WAVE
        assert hidden_dim * V_T_STRIDE >= GRP_WAVES * _O_WAVE
        o_row_layout = fx.make_layout((O_CH, O_VEC), (O_VEC, 1))
        o_scr = [
            arr.view(
                fx.make_layout(
                    (n // (O_PCH * O_VEC), O_PCH, O_VEC), (O_PCH * O_VEC, O_VEC, 1)
                )
            )
            for arr, n in (
                (lds.k1, BLOCK_N * head_dim),
                (lds.v0, hidden_dim * V_T_STRIDE),
            )
        ]
        o_slot_row = (wave_id % fx.Int32(GRP_WAVES)) * fx.Int32(MFMA_M)

        q_raw = q_raw_load(q_tile_idx)

        q_tile_base = q_tile_idx * fx.Int32(BLOCK_M)

        def q_blk_base(qg):
            """First row of this wave's `qg`-th 16-row block of the Q tile."""
            return q_tile_base + q_blk_off(qg)

        # Epilogue: O goes out through a wave-private LDS transpose. GEMM2's C
        # fragment has one query row per (lane_div_16, e) and one d per lane,
        # so a direct store is a 2-byte scatter that clogs the vector memory
        # pipe; staged, each half of a q-subtile's d range is one 16-byte store
        # per lane. Scratch: g0 in K slot 1, g1 in V slot 0 (the groups'
        # epilogues overlap). Nothing else touches those slots by then.
        def store_o(results, grp):
            scr = o_scr[grp]
            for qg in range_constexpr(Q_SUBTILES):
                for hp in range_constexpr(O_PASSES):
                    for e in range_constexpr(MFMA_ELEMS_PER_LANE):
                        row = lane_div_16 * fx.Int32(MFMA_LANE_K) + fx.Int32(e)
                        for cc in range_constexpr(O_PASS_CHUNKS):
                            c = hp * O_PASS_CHUNKS + cc
                            ch = fx.Int32(
                                cc * (MFMA_M // O_VEC)
                            ) + lane_mod_16 // fx.Int32(O_VEC)
                            sw = ch ^ (lane_div_16 & fx.Int32(O_SWZ))
                            val = (Vec(results[c * Q_SUBTILES + qg])[e] * inv_n).to(
                                elem_dtype
                            )
                            scr[o_slot_row + row, sw, lane_mod_16 % fx.Int32(O_VEC)] = (
                                val
                            )
                    r = lane // fx.Int32(O_PCH)
                    ch = lane % fx.Int32(O_PCH)
                    vec = scr[
                        o_slot_row + r,
                        ch ^ ((r // fx.Int32(MFMA_LANE_K)) & fx.Int32(O_SWZ)),
                        None,
                    ].load()
                    grow = q_blk_base(qg) + r
                    if grow < seq_len:
                        orow = out[fx.Int64(seq_start + grow), head_idx, None]
                        fx.make_view(fx.get_iter(orow), o_row_layout)[
                            ch + fx.Int32(hp * O_PCH), None
                        ].store(vec)

        # ---- Q rows / bounds per query sub-tile ----
        q_rows = []
        q_in_bounds = []
        for qg in range_constexpr(Q_SUBTILES):
            local = q_blk_base(qg) + lane_mod_16
            q_rows.append(local)
            q_in_bounds.append(local < seq_len)

        # ---- Q B-operand packs (register-resident, per query sub-tile); q_packs[ks][qg] ----
        q_packs = []
        for ks in range_constexpr(K_STEPS):
            per_qg = []
            for qg in range_constexpr(Q_SUBTILES):
                raw = q_raw[ks * Q_SUBTILES + qg]
                per_qg.append(q_in_bounds[qg].select(raw, c_zero_dim_pack))
            q_packs.append(per_qg)

        # ---- Score gate ----
        # With g = s * alpha * -log2e, sigmoid(alpha*s) = 1 / (1 + 2^g). The
        # gate computes p = s * sigmoid(alpha*s) on the fp32 score s; g is only
        # ever formed inside the table index FMA, and alpha rides the
        # epilogue's 1/N, since silu(alpha*s) = alpha * p.
        c_one_f = fx.Float32(1.0)
        c_zero_f = fx.Float32(0.0)
        # Column id for a position past the end of the sequence: larger than
        # any query id, small enough that a query id minus it cannot wrap.
        COL_OOB = fx.Int32(1 << 30)

        def _rcp(x):
            return fx.Float32(fx.rocdl.rcp(compute_type, x.ir_value()))

        # LUT index by the magic-number trick: biased into [2^21, 2^22), the
        # float's mantissa holds 4 * index (ULP 0.25), so the index is one bit
        # extract instead of a convert and a shift. The bias is rounded to the
        # 0.25 ULP, which raises sigmoid's max relative error from 3.906e-3 to
        # 3.998e-3, still under bf16's 7.812e-3.
        _LUT_MAGIC = float(2**21)
        _LUT_BIG = _LUT_MAGIC + round(_LUT_BIAS * 4.0) / 4.0
        c_lut_scale = fx.Float32(_LUT_SCALE * alpha * -_LOG2E)
        c_lut_big = fx.Float32(_LUT_BIG)
        c_lut_lo = fx.Float32(_LUT_MAGIC)
        # The top clamp lands on the sigmoid = 0 entry, so a score past the
        # range, or one dropped by the window bias, gates to exactly 0.
        c_lut_hi = fx.Float32(_LUT_MAGIC + float(_LUT_TAB - 1))

        def lut_index(sv):
            """Table index of sigmoid(alpha*s) for the raw score s: three
            full-rate VALU ops (index FMA, clamp, bit extract)."""
            t = sv * c_lut_scale + c_lut_big
            t = fx.clampf(t, c_lut_lo, c_lut_hi)  # v_med3_f32
            return (t.bitcast(fx.Int32) >> fx.Int32(2)) & fx.Int32(0x1FFFFF)

        def lut_load(idx):
            return Vec(lut_mem[idx, None].load())[0]

        def lut_sigma(sv):
            return lut_load(lut_index(sv))

        def build_sigma_lut():
            """Fill the table once per workgroup (~5 rounds over BLOCK_THREADS)."""
            for r in range_constexpr(_LUT_ROUNDS):
                _idx = tid + fx.Int32(r * BLOCK_THREADS)
                # Surplus threads of the last round rewrite the last entry with
                # the same bits.
                _idx = (_idx < fx.Int32(_LUT_TAB)).select(_idx, fx.Int32(_LUT_TAB - 1))
                _gv = _idx.to(fx.Float32) * fx.Float32(_LUT_H) + fx.Float32(_LUT_GLO)
                _sg = _rcp(c_one_f + exp2_f32(_gv))
                _sg = (_idx < fx.Int32(_LUT_N)).select(_sg, c_zero_f)
                lut_mem[_idx, fx.Int32(0)] = _sg

        def silu_scale_batch(s_list):
            """p = s * sigmoid(alpha*s) for raw scores; a zeroed (masked) score
            gives p = 0 exactly."""
            sig = [lut_sigma(sv) for sv in s_list]  # all reads before any multiply
            return [sv * si for sv, si in zip(s_list, sig)]

        def to_id(x):
            """Raw position -> masked id (contextual prefix shift, then target-tail clamp)."""
            xid = x
            if has_contextual:
                xid = xid - fx.Int32(contextual_seq_len - 1)
                xid = (xid < fx.Int32(0)).select(fx.Int32(0), xid)
            if has_targets:
                xid = (xid > max_id).select(max_id, xid)
            return xid

        def pack_p(vals):
            """Pack 4 f32 into a bf16 MFMA operand by truncation.

            Truncation is a byte selection (the high half of each f32), so one
            `v_perm_b32` packs a pair; the shift/and/or spelling costs two VALU.
            Not a convert: gfx942 has no `v_cvt_pk_bf16_f32`.
            """
            # Bytes 7,6 of src0 (hi) and 3,2 of src1 (lo): the two high halves.
            sel = fx.Int32(0x07060302)

            def bf16_pair(lo_f32, hi_f32):
                return fx.Int32(
                    rocdl.perm_b32(
                        hi_f32.bitcast(fx.Int32), lo_f32.bitcast(fx.Int32), sel
                    )
                )

            pairs = [bf16_pair(vals[0], vals[1]), bf16_pair(vals[2], vals[3])]
            return Vec.from_elements(pairs, fx.Int32).bitcast(elem_dtype).ir_value()

        q_row_ids = [to_id(q_rows[qg]) for qg in range_constexpr(Q_SUBTILES)]

        # ---- The causal mask as one per-row border -------------------------
        # to_id is monotone, so `to_id(col) < to_id(q)` holds below one raw
        # column per row: border = min(q + 1, id_ceiling), and the mask is
        # `col < border` with no per-column to_id. The +1 admits the diagonal
        # for unclamped rows; a target row stops at the ceiling and needs the
        # diagonal back (`border_keep`). A contextual-prefix row (id 0) attends
        # the whole prefix: the same border without the causal term.
        _CTX_S = contextual_seq_len - 1 if has_contextual else 0
        _id_ceiling = max_id + fx.Int32(_CTX_S) if has_targets else COL_OOB
        q_border = []
        for qg in range_constexpr(Q_SUBTILES):
            _b = q_rows[qg] + fx.Int32(1)
            if const_expr(has_targets):
                _b = (_b < _id_ceiling).select(_b, _id_ceiling)
            if const_expr(has_contextual):
                _b = (q_row_ids[qg] > fx.Int32(0)).select(_b, _id_ceiling)
            q_border.append(_b)

        # A window adds a low bound, col_id >= q_id - W. Without a contextual
        # prefix that is col >= min(q, max_id) - W in raw columns: one compare.
        BORDER_MASK = causal and not has_contextual

        def _win_lo(r):
            if const_expr(has_targets):
                r = (r < _id_ceiling).select(r, _id_ceiling)
            return r - fx.Int32(max_attn_len)

        q_lo = []
        if const_expr(has_window and BORDER_MASK):
            q_lo = [_win_lo(q_rows[qg]) for qg in range_constexpr(Q_SUBTILES)]

        # Wave-uniform q-block bounds for `gate_tile_split`'s per-sub-block
        # branches. Pinned through readfirstlane, since the divergence analysis
        # treats wave_id as per-lane; used only in the predicates. Raw rocdl:
        # no typed uniformity API in FlyDSL 0.5.0.
        def _sgpr32(x):
            return fx.Int32(rocdl.readfirstlane(fx.Int32.ir_type, x.ir_value()))

        q_blk_lo = [_sgpr32(q_blk_base(qg)) for qg in range_constexpr(Q_SUBTILES)]
        q_blk_hi = [x + fx.Int32(MFMA_M - 1) for x in q_blk_lo]
        # The border is monotone in the row only without a contextual prefix.
        SUBBLK_SKIP = causal and not has_contextual
        if const_expr(SUBBLK_SKIP):

            def _row_border(r):
                b = r + fx.Int32(1)
                if const_expr(has_targets):
                    b = (b < _id_ceiling).select(b, _id_ceiling)
                return b

            q_blk_blo = [_row_border(x) for x in q_blk_lo]
            q_blk_bhi = [_row_border(x) for x in q_blk_hi]
            if const_expr(has_window):
                q_blk_wlo = [_win_lo(x) for x in q_blk_lo]
                q_blk_whi = [_win_lo(x) for x in q_blk_hi]

        # ---- BIAS_MASK: table build, per-tile slot pick, target blocks ------
        if const_expr(BIAS_MASK):
            _B_ROWS_PER_SLOT = KV_SUBTILES * WARP_SIZE
            _B_ROUNDS = (BIAS_ROWS + BLOCK_THREADS - 1) // BLOCK_THREADS
            c_bias_huge = fx.Float32(BIAS_HUGE)

            def build_bias_table():
                """Fill every slot's bias tiles: row = (slot, ng, lane), four
                f32 per row in the lane's own accumulator order."""
                for r in range_constexpr(_B_ROUNDS):
                    row = tid + fx.Int32(r * BLOCK_THREADS)
                    # Surplus threads rewrite the last row with the same bits.
                    row = (row < fx.Int32(BIAS_ROWS)).select(
                        row, fx.Int32(BIAS_ROWS - 1)
                    )
                    slot = row // fx.Int32(_B_ROWS_PER_SLOT)
                    ng = (row // fx.Int32(WARP_SIZE)) % fx.Int32(KV_SUBTILES)
                    ln = row % fx.Int32(WARP_SIZE)
                    if const_expr(BIAS_FOLD):
                        vrep = (slot <= fx.Int32(KV_SUBTILES + 1)).select(
                            slot - fx.Int32(1),
                            slot + fx.Int32(_JK - KV_SUBTILES - 1),
                        )
                    else:
                        vrep = slot - fx.Int32(1)
                    # d = q - c = U + b - a, b = lane%16 (query), a = 4*(lane/16)+i (kv)
                    d0 = (
                        (vrep - ng) * fx.Int32(MFMA_M)
                        + ln % fx.Int32(MFMA_N)
                        - (ln // fx.Int32(MFMA_N)) * fx.Int32(MFMA_LANE_K)
                    )
                    vals = []
                    for i in range_constexpr(MFMA_ELEMS_PER_LANE):
                        d = d0 - fx.Int32(i)
                        drop = (d < fx.Int32(0)) | (d > fx.Int32(max_attn_len))
                        drop = drop & (slot != fx.Int32(BIAS_KEEP_SLOT))
                        vals.append(drop.select(c_bias_huge, c_zero_f))
                    mbias_mem[row, None].store(Vec.from_elements(vals, fx.Float32))

            # The bounds are i32 bits; the unsigned forms reinterpret them.
            def _smax(a, b):
                return fx.max(a, fx.Int32(b))

            def _umin(a, b):
                return fx.Int32(fx.min(fx.Uint32(a), fx.Uint32(b)))

            def _umax(a, b):
                return fx.Int32(fx.max(fx.Uint32(a), fx.Uint32(b)))

            def bias_row_pieces(kv_start_fn):
                """The slot-row computation as small SALU pieces that phase M runs
                one per MFMA shadow, in order; the rows end up in `st["rows"]`.

                Each clamp is a signed max then an unsigned min: a same-sign pair
                fuses into v_med3_i32, which has no SALU form."""
                st = {"rows": [None] * Q_SUBTILES}

                def _start():
                    st["kv"] = kv_start_fn()

                # At most ~3 SALU per piece: what fits in one MFMA's 12-cycle
                # shadow.
                def _p0(qg):
                    u = q_blk_lo[qg] - st["kv"]
                    st[("u", qg)] = u
                    st[("lo", qg)] = _smax(u, -MFMA_M)

                def _p1(qg):
                    if const_expr(BIAS_FOLD):
                        st[("hi", qg)] = _smax(
                            st[("u", qg)] - fx.Int32(MFMA_M * _JK), -(MFMA_M - 1)
                        )
                        st[("lo", qg)] = _umin(
                            st[("lo", qg)] + fx.Int32(MFMA_M),
                            MFMA_M * (KV_SUBTILES + 1),
                        )
                    else:
                        st[("lo", qg)] = _umin(
                            st[("lo", qg)] + fx.Int32(MFMA_M),
                            MFMA_M * (_JT + KV_SUBTILES + 1),
                        )

                def _p2(qg):
                    if const_expr(BIAS_FOLD):
                        # Bounds 15 off the multiples of 16, re-aligned by a
                        # mask: with exact bounds instcombine turns the umin
                        # into smin and the pair fuses into v_med3 anyway.
                        st[("hi", qg)] = _umin(
                            st[("hi", qg)] + fx.Int32(MFMA_M - 1),
                            MFMA_M * _BR + MFMA_M - 1,
                        ) & fx.Int32(-MFMA_M)

                def _p3(qg):
                    s16 = st[("lo", qg)]
                    if const_expr(BIAS_FOLD):
                        s16 = s16 + st[("hi", qg)]
                    row0 = s16 * fx.Int32(_B_ROWS_PER_SLOT // MFMA_M)
                    if const_expr(has_targets):
                        # The zero slot is the LAST one, so a max selects it.
                        row0 = _umax(row0, q_blk_tgt_row[qg])
                    st["rows"][qg] = row0

                pieces = [_start]
                for qg in range_constexpr(Q_SUBTILES):
                    pieces += [functools.partial(f, qg) for f in (_p0, _p1, _p2, _p3)]
                return pieces, st

            def bias_rows(kv_start):
                """Per q-subtile, the first table row of this tile's slot."""
                pieces, st = bias_row_pieces(lambda: kv_start)
                for f in pieces:
                    f()
                return st["rows"]

            def bias_init(rows):
                """GEMM1's initial accumulators, s_accs[ng][qg], read from the table."""
                return [
                    [
                        mbias_mem[rows[qg] + fx.Int32(ng * WARP_SIZE) + lane, None]
                        .load()
                        .ir_value()
                        for qg in range_constexpr(Q_SUBTILES)
                    ]
                    for ng in range_constexpr(KV_SUBTILES)
                ]

            if const_expr(has_targets):
                # q-blocks holding a target row; loop-invariant, and false
                # outside the sequence's last Q tile.
                _has_tgt_rows = max_id < seq_len
                q_blk_tgt = [
                    _has_tgt_rows & (q_blk_hi[qg] >= max_id) & (q_blk_lo[qg] < seq_len)
                    for qg in range_constexpr(Q_SUBTILES)
                ]
                q_blk_tgt_row = [
                    x.select(fx.Int32(BIAS_KEEP_SLOT * _B_ROWS_PER_SLOT), fx.Int32(0))
                    for x in q_blk_tgt
                ]
                wave_tgt = q_blk_tgt[0]
                for qg in range_constexpr(1, Q_SUBTILES):
                    wave_tgt = wave_tgt | q_blk_tgt[qg]

        # ---- KV range: causal upper bound + active predicate ----
        q_start = q_tile_idx * fx.Int32(BLOCK_M)
        q_end = q_start + fx.Int32(BLOCK_M)
        base_upper = seq_len
        if causal:
            clamped = (q_end < seq_len).select(q_end, seq_len)
            if has_contextual:
                # Prefix rows (id 0) attend every col_id < max_id, so the
                # prefix block sweeps to seq_len.
                ctx_block = q_start < fx.Int32(contextual_seq_len)
                base_upper = ctx_block.select(seq_len, clamped)
            else:
                base_upper = clamped
        active = q_start < seq_len
        kv_upper = active.select(base_upper, fx.Int32(0))
        n_tiles = (kv_upper + fx.Int32(BLOCK_N - 1)) // fx.Int32(BLOCK_N)

        # ---- Sliding-window lower bound: skip fully-masked low KV tiles ----
        # In effective IDs (the whole contextual prefix is ID 0): the bound maps
        # back to a raw position only when positive; otherwise the window
        # reaches the prefix and the sweep starts at tile 0.
        kv_tile_start = fx.Int32(0)
        if has_window:
            id_lower = to_id(q_start) - fx.Int32(max_attn_len)
            kv_lower = id_lower
            if has_contextual:
                kv_lower = id_lower + fx.Int32(contextual_seq_len - 1)
            kv_lower = (id_lower > fx.Int32(0)).select(kv_lower, fx.Int32(0))
            kv_tile_start = kv_lower // fx.Int32(BLOCK_N)

        N_ACC = D_CHUNKS * Q_SUBTILES
        c_zero_v4f32 = fx.full(MFMA_ELEMS_PER_LANE, 0.0, fx.Float32).ir_value()

        # ---- Group-staged K/V prefetch (group-local thread ids) ----
        kv_row_in_batch = grp_tid // fx.Int32(THREADS_PER_ROW_KV)
        kv_lane_in_row = grp_tid % fx.Int32(THREADS_PER_ROW_KV)
        kv_col = kv_lane_in_row * fx.Int32(VEC_KV)
        # The same column in MFMA_LANE_K groups, for the swizzled K store.
        kv_col_grp = kv_lane_in_row * fx.Int32(VEC_KV // MFMA_DIM_LANE_K)

        # V rotation per lane: a lane's VEC_KV d values sit in one d-chunk, so
        # rotated lanes (chunk < V_ROT_DT) stage tile kv_tile + 1 and the rest
        # stage kv_tile. g0 stages K(kv_tile + 1), which is not rotated.
        kv_rot_lane = (kv_col // fx.Int32(MFMA_M)) < fx.Int32(V_ROT_DT)
        kv_stage_bump = (is_g0 | kv_rot_lane).select(fx.Int32(1), fx.Int32(0))

        # The staging voffset is carried and advanced by a host constant per
        # trip (one add instead of rebuilding the address), in VEC_KV-element
        # groups, which is what the descriptor view is indexed in.
        STAGE_ADV = fx.Int32(BLOCK_N * KV_ROW_GRPS)

        def stage_seed(kv_tile):
            """[voffset] for the tile this lane stages at `kv_tile`."""
            out = []
            for b in range_constexpr(NUM_BATCHES_KV):
                row = kv_row_in_batch + fx.Int32(b * ROWS_PER_BATCH_KV)
                tok = (kv_tile + kv_stage_bump) * fx.Int32(BLOCK_N) + row
                out.append(tok * fx.Int32(KV_ROW_GRPS) + kv_col // fx.Int32(VEC_KV))
            return out

        def stage_advance(voffs):
            return [x + STAGE_ADV for x in voffs]

        def async_load_kv_regs(voffs):
            """Issue this group's coalesced global load to registers (non-blocking).

            Rows past the sequence read back zero from the descriptor; no select
            on the loaded value, which would pull its vmcnt wait into this phase.
            """
            return [
                kv_buf[voffs[b], None].load().ir_value()
                for b in range_constexpr(NUM_BATCHES_KV)
            ]

        def store_k_regs_to_lds(vecs, ks_mem, part=None):
            """g0 only: write K vecs to LDS at XOR-swizzled columns.

            `part` selects one store, so `mfma_burst` can place them one per
            MFMA shadow."""
            _n = -1
            for b in range_constexpr(NUM_BATCHES_KV):
                row = kv_row_in_batch + fx.Int32(b * ROWS_PER_BATCH_KV)
                for h in range_constexpr(VEC_KV // MFMA_DIM_LANE_K):
                    _n += 1
                    if const_expr(part is not None and _n != part):
                        continue
                    col_grp = kv_col_grp + fx.Int32(h)
                    half = Vec.from_elements(
                        [
                            Vec(vecs[b])[h * MFMA_DIM_LANE_K + j]
                            for j in range_constexpr(MFMA_DIM_LANE_K)
                        ],
                        elem_dtype,
                    )
                    ks_mem[row, k_swz_grp(row, col_grp), None].store(half)

        def store_v_regs_to_lds(vecs, vs_mem, part=None):
            """g1 only: write V vecs to LDS transposed to V[d, kv] so GEMM2 can read 4
            consecutive kv per ds_read_b64. The lane's VEC_KV d-contiguous elements
            scatter to kv-fixed, d-varying LDS slots."""
            # The swizzle key, (kv_col + j) // VEC_KV, is the same for every j;
            # hoisted, it saves 24 VGPR (and 5 spills) over computing it per j.
            vkey = v_swz_key(kv_col)
            _n = -1
            for b in range_constexpr(NUM_BATCHES_KV):
                kv_row = kv_row_in_batch + fx.Int32(b * ROWS_PER_BATCH_KV)
                swz_grp = (kv_row // fx.Int32(MFMA_LANE_K)) ^ vkey
                kv_lane = kv_row % fx.Int32(MFMA_LANE_K)
                vv = Vec(vecs[b])
                for j in range_constexpr(VEC_KV):
                    _n += 1
                    if const_expr(part is not None and _n != part):
                        continue
                    vs_mem[kv_col + fx.Int32(j), swz_grp, kv_lane] = vv[j]

        # ---- Per-unit operand readers and MFMAs (see UNIT_ORDER) ----
        def read_k_unit(ng, ks, ks_mem):
            """One K A-operand pack: sub-tile `ng`, contraction step `ks`."""
            local_k_row = fx.Int32(ng * MFMA_M) + lane_mod_16
            k_col_grp = qk_col_grp(ks)
            return ks_mem[local_k_row, k_swz_grp(local_k_row, k_col_grp), None].load()

        def read_v_unit(c, ng, vs_mem):
            """One V B-operand pack (4 consecutive kv): d-chunk `c`, sub-tile `ng`."""
            d_col = fx.Int32(c * MFMA_M) + lane_mod_16
            # kv_lane = ng*MFMA_M + lane_div_16*MFMA_LANE_K, in MFMA_LANE_K-group units.
            kv_grp = fx.Int32(ng * (MFMA_M // MFMA_LANE_K)) + lane_div_16
            return vs_mem[d_col, v_swz_grp(d_col, kv_grp), None].load().ir_value()

        def gemm1_unit(s_accs, ng, ks, k_pack, hook=None):
            """The Q_SUBTILES MFMAs of one QK unit, chained on s_accs[ng][*]."""
            for qg in range_constexpr(Q_SUBTILES):
                # A=K, B=Q, so the result P lands in GEMM2's A-operand layout
                # and stays in registers.
                s_accs[ng][qg] = dim_mfma_acc(
                    Vec(k_pack).ir_value(), q_packs[ks][qg], s_accs[ng][qg]
                )
                if const_expr(hook is not None):
                    hook()
            return s_accs

        def gemm2_unit(o_acc, p_packs, c, ng, v_pack, hook=None):
            """The Q_SUBTILES MFMAs of one PV unit, chained on o_acc[c*Q+*]."""
            for qg in range_constexpr(Q_SUBTILES):
                off = c * Q_SUBTILES + qg
                o_acc[off] = mfma_acc(p_packs[ng][qg], v_pack, o_acc[off])
                if const_expr(hook is not None):
                    hook()
            return o_acc

        def read_unit(spec, ks_mem, vs_mem):
            """The one `ds_read` of unit `spec`, from the right slot."""
            kind, j = spec
            if const_expr(kind == "qk"):
                return read_k_unit(j % KV_SUBTILES, j // KV_SUBTILES, ks_mem)
            return read_v_unit(j % D_CHUNKS, j // D_CHUNKS, vs_mem)

        def apply_unit(spec, operand, o_acc, s_accs, p_packs, hook=None):
            """The Q_SUBTILES MFMAs of unit `spec`, consuming `operand`."""
            kind, j = spec
            if const_expr(kind == "qk"):
                s_accs = gemm1_unit(
                    s_accs, j % KV_SUBTILES, j // KV_SUBTILES, operand, hook
                )
            else:
                o_acc = gemm2_unit(
                    o_acc, p_packs, j % D_CHUNKS, j // D_CHUNKS, operand, hook
                )
            return o_acc, s_accs

        def mfma_burst(
            o_acc,
            p_packs,
            carried,
            ks_mem,
            vs_mem,
            tail_store=None,
            bias=None,
            side=None,
        ):
            """Phase M's MFMA stream: GEMM1 of this tile and GEMM2 of the previous
            one, walked in `UNIT_ORDER`. Returns the raw scores for phase V."""
            if const_expr(bias is not None):
                s_accs = bias
            else:
                s_accs = [
                    [
                        fx.full(MFMA_ELEMS_PER_LANE, 0.0, fx.Float32).ir_value()
                        for _ in range_constexpr(Q_SUBTILES)
                    ]
                    for _ in range_constexpr(KV_SUBTILES)
                ]
            # The read window stays PREFETCH_DEPTH units ahead; `sched_barrier`
            # around each step stops the scheduler hoisting every read to the
            # head of the phase.
            #
            # Staging stores go one per MFMA shadow (12 cycles), after the
            # n_store MFMAs before the last one; at d=64 that is g0's 2
            # ds_write_b64 after #-3..#-2 and g1's 8 ds_write_b16 after
            # #-9..#-2. Only the cross-phase prefetch reads are issued behind
            # them, and those are waited on at the phase barrier anyway.
            _n_mfma = [0]
            _n_store = 0 if tail_store is None else tail_store[0]
            _first = N_UNITS * Q_SUBTILES - _n_store - 1

            _side_res = [None]
            if const_expr(side is not None):
                _side_m0, _side_pieces, _side_st = side

            def _hook():
                m = _n_mfma[0]
                _n_mfma[0] = m + 1
                if const_expr(side is not None):
                    j = m - _side_m0
                    if const_expr(0 <= j < len(_side_pieces)):
                        # One SALU piece per MFMA shadow, fenced so the
                        # scheduler cannot regroup them.
                        rocdl.sched_barrier(0)
                        _side_pieces[j]()
                        if const_expr(j == len(_side_pieces) - 1):
                            # Pin the rows here: without a use in phase M,
                            # machine sinking moves the pieces into phase V
                            # next to `bias_init`. readfirstlane is that use;
                            # it folds away on an SGPR value, so the ISA is
                            # that of an empty inline-asm pin. No value-pin API
                            # in FlyDSL 0.5.0; `disable-machine-sink` instead
                            # grows the d=128 spills.
                            _side_res[0] = [_sgpr32(x) for x in _side_st["rows"]]
                        rocdl.sched_barrier(0)
                if const_expr(tail_store is None or m < _first):
                    return
                k = m - _first
                if const_expr(k >= _n_store):
                    return
                if const_expr(k == 0):
                    rocdl.s_waitcnt(vmcnt=0)
                # Fenced individually, or the scheduler regroups the stores.
                rocdl.sched_barrier(0)
                tail_store[1](k)
                rocdl.sched_barrier(0)

            win = [None for _ in range_constexpr(WIN_RING)]
            # Seeded from the previous phase's tail prefetch.
            for s in range_constexpr(XPHASE_DEPTH):
                win[s % WIN_RING] = carried[s]
            for s in range_constexpr(XPHASE_DEPTH, PREFETCH_DEPTH):
                win[s % WIN_RING] = read_unit(UNIT_ORDER[s], ks_mem, vs_mem)
            nxt = [None for _ in range_constexpr(XPHASE_DEPTH)]
            for step in range_constexpr(N_UNITS):
                rocdl.sched_barrier(0)
                o_acc, s_accs = apply_unit(
                    UNIT_ORDER[step],
                    win[step % WIN_RING],
                    o_acc,
                    s_accs,
                    p_packs,
                    _hook,
                )
                sn = step + PREFETCH_DEPTH
                if const_expr(sn < N_UNITS):
                    win[sn % WIN_RING] = read_unit(UNIT_ORDER[sn], ks_mem, vs_mem)
                else:
                    # The tail reads the next phase's first units from this
                    # phase's `vs_mem` (V_ROT_DT), never the slot being written.
                    nxt[sn - N_UNITS] = read_unit(
                        UNIT_ORDER[sn - N_UNITS], ks_mem, vs_mem
                    )
                rocdl.sched_barrier(0)
            return o_acc, s_accs, nxt, _side_res[0]

        c_zero_pack = fx.full(MFMA_LANE_K, 0.0, elem_dtype).ir_value()

        def border_keep(col, qg):
            """Keep predicate of raw column `col` for q-subtile `qg`, border form."""
            keep = col < q_border[qg]
            if const_expr(has_window):
                keep = keep & (col >= q_lo[qg])
            if const_expr(has_targets):
                # A target row's id is clamped, so its border stops at the
                # ceiling and the diagonal has to come back by hand.
                keep = keep | (col == q_rows[qg])
            return keep

        def mask_target_blocks(kv_start, s_accs):
            """BIAS_MASK with targets: per-element mask on the q-blocks holding
            target rows, behind a loop-invariant wave-uniform branch."""
            sa = [
                s_accs[ng][qg]
                for ng in range_constexpr(KV_SUBTILES)
                for qg in range_constexpr(Q_SUBTILES)
            ]

            def _fix(sa):
                out = []
                for ng in range_constexpr(KV_SUBTILES):
                    kv_base = (
                        kv_start
                        + fx.Int32(ng * MFMA_M)
                        + lane_div_16 * fx.Int32(MFMA_LANE_K)
                    )
                    for qg in range_constexpr(Q_SUBTILES):
                        cur = Vec(sa[ng * Q_SUBTILES + qg])
                        vals = []
                        for i in range_constexpr(MFMA_ELEMS_PER_LANE):
                            keep = border_keep(kv_base + fx.Int32(i), qg)
                            keep = keep | ~q_blk_tgt[qg]
                            vals.append(keep.select(cur[i], c_zero_f))
                        out.append(Vec.from_elements(vals, fx.Float32).ir_value())
                return out

            if wave_tgt:
                sa = _fix(sa)
            return [
                [sa[ng * Q_SUBTILES + qg] for qg in range_constexpr(Q_SUBTILES)]
                for ng in range_constexpr(KV_SUBTILES)
            ]

        def gate_tile_split(kv_start, s_accs):
            """Phase V for a masked tile, per 16x16 (qg, ng) sub-block.

            A wave-uniform SALU branch per sub-block: entirely masked -> zero
            pack, no gate; entirely kept -> gate without the mask; straddling
            -> both. The barrier after phase V pays the max over waves, which
            is why the q-blocks are interleaved (`q_blk_off`).
            """
            p_packs = []
            for ng in range_constexpr(KV_SUBTILES):
                kv_lo = kv_start + fx.Int32(ng * MFMA_M)
                kv_hi = kv_lo + fx.Int32(MFMA_M - 1)
                kv_base = kv_lo + lane_div_16 * fx.Int32(MFMA_LANE_K)
                col_raw = [
                    kv_base + fx.Int32(i) for i in range_constexpr(MFMA_ELEMS_PER_LANE)
                ]
                col_raw = [(c < seq_len).select(c, COL_OOB) for c in col_raw]
                row = []
                for qg in range_constexpr(Q_SUBTILES):
                    cur = s_accs[ng][qg]
                    s_vals = [Vec(cur)[i] for i in range_constexpr(MFMA_ELEMS_PER_LANE)]
                    # Above the diagonal (past every row's border and own
                    # token), or past the end of the sequence.
                    all_drop = (kv_lo >= seq_len) | (
                        (kv_lo > q_blk_hi[qg]) & (kv_lo >= q_blk_bhi[qg])
                    )
                    # Below it: in the sequence and under the lowest row's
                    # border.
                    all_keep = (kv_hi < q_blk_blo[qg]) & (kv_hi < seq_len)
                    if const_expr(has_window):
                        # Below the lowest row's window, or inside the
                        # highest row's.
                        all_drop = all_drop | (kv_hi < q_blk_wlo[qg])
                        all_keep = all_keep & (kv_lo >= q_blk_whi[qg])
                    # Pre-bound: FlyDSL's `if` yields the names it assigns
                    # and needs their values before the branch.
                    pk = c_zero_pack
                    if all_drop:
                        pk = c_zero_pack
                    elif all_keep:
                        pk = pack_p(silu_scale_batch(s_vals))
                    else:
                        kept = [
                            border_keep(col_raw[i], qg).select(s_vals[i], c_zero_f)
                            for i in range_constexpr(MFMA_ELEMS_PER_LANE)
                        ]
                        pk = pack_p(silu_scale_batch(kept))
                    row.append(pk)
                p_packs.append(row)
            return p_packs

        def gate_tile(kv_start, s_accs, apply_mask=True):
            """Phase V: mask and SiLU gate -> P packs.

            apply_mask=False skips the mask; only valid for a tile entirely in
            the sequence and entirely kept (the causal unmasked prefix)."""
            if const_expr(apply_mask and SUBBLK_SKIP):
                return gate_tile_split(kv_start, s_accs)

            # The whole tile's masked scores in (ng, qg, i) order, so every
            # aligned run of MFMA_ELEMS_PER_LANE is one accumulator.
            flat = []
            for ng in range_constexpr(KV_SUBTILES):
                if const_expr(apply_mask):
                    kv_base = (
                        kv_start
                        + fx.Int32(ng * MFMA_M)
                        + lane_div_16 * fx.Int32(MFMA_LANE_K)
                    )
                    col_raw = [
                        kv_base + fx.Int32(i)
                        for i in range_constexpr(MFMA_ELEMS_PER_LANE)
                    ]
                    # Fold the sequence bound into the column: an out-of-seq
                    # column gets a position no query equals and an id no
                    # query exceeds, so every term of the mask rejects it.
                    col_in_seq = [
                        col_raw[i] < seq_len
                        for i in range_constexpr(MFMA_ELEMS_PER_LANE)
                    ]
                    if const_expr(BORDER_MASK):
                        # Border form: raw columns, no per-column to_id.
                        col_raw = [
                            col_in_seq[i].select(col_raw[i], COL_OOB)
                            for i in range_constexpr(MFMA_ELEMS_PER_LANE)
                        ]
                    else:
                        col_raw = [
                            col_in_seq[i].select(col_raw[i], fx.Int32(-1))
                            for i in range_constexpr(MFMA_ELEMS_PER_LANE)
                        ]
                        col_id = [
                            col_in_seq[i].select(to_id(col_raw[i]), COL_OOB)
                            for i in range_constexpr(MFMA_ELEMS_PER_LANE)
                        ]
                for qg in range_constexpr(Q_SUBTILES):
                    cur = s_accs[ng][qg]
                    s_vals = [Vec(cur)[i] for i in range_constexpr(MFMA_ELEMS_PER_LANE)]

                    if const_expr(apply_mask):
                        # Traced and used within this iteration, so B023's
                        # late binding does not apply.
                        def keep_col(i):
                            """causal * window * contextual * target mask for (qg, col i)."""
                            if const_expr(BORDER_MASK):
                                return border_keep(col_raw[i], qg)  # noqa: B023
                            dist = q_row_ids[qg] - col_id[i]  # noqa: B023
                            if not causal:
                                # Symmetric distance; a window is |q - col| <= W.
                                dist = (dist > fx.Int32(0)).select(dist, -dist)
                            # The diagonal compares raw positions: a query keeps
                            # its own token even where to_id collapses ids.
                            keep = (q_rows[qg] == col_raw[i]) | (  # noqa: B023
                                dist > fx.Int32(0)
                            )
                            if has_window:
                                keep = keep & (dist <= fx.Int32(max_attn_len))
                            if has_contextual:
                                # Row id 0 attends the contextual prefix.
                                ctx = (q_row_ids[qg] == fx.Int32(0)) & (  # noqa: B023
                                    col_id[i] < max_id  # noqa: B023
                                )
                                keep = keep | ctx
                            return keep

                        s_vals = [
                            keep_col(i).select(s_vals[i], c_zero_f)
                            for i in range_constexpr(MFMA_ELEMS_PER_LANE)
                        ]
                    flat.extend(s_vals)

            # LUT reads in two software-pipelined groups (GATE_SPLIT): each
            # group's address math covers the other's reads, and each group is
            # multiplied and packed completely before the next.
            _h = min(GATE_SPLIT, len(flat) - MFMA_ELEMS_PER_LANE)
            rocdl.sched_barrier(0)
            _i0 = [lut_index(x) for x in flat[:_h]]
            rocdl.sched_barrier(0)
            _s0 = [lut_load(x) for x in _i0]
            rocdl.sched_barrier(0)
            _i1 = [lut_index(x) for x in flat[_h:]]
            rocdl.sched_barrier(0)
            _s1 = [lut_load(x) for x in _i1]
            rocdl.sched_barrier(0)
            _E = MFMA_ELEMS_PER_LANE
            _pk = []
            for _src, _sig in ((flat[:_h], _s0), (flat[_h:], _s1)):
                rocdl.sched_barrier(0)
                _g = [_src[k] * _sig[k] for k in range_constexpr(len(_src))]
                rocdl.sched_barrier(0)
                _pk += [
                    pack_p(_g[j * _E : (j + 1) * _E])
                    for j in range_constexpr(len(_src) // _E)
                ]
                rocdl.sched_barrier(0)
            return [
                [_pk[ng * Q_SUBTILES + qg] for qg in range_constexpr(Q_SUBTILES)]
                for ng in range_constexpr(KV_SUBTILES)
            ]

        # ==== The two phases ====
        N_PACK = KV_SUBTILES * Q_SUBTILES

        def mfma_phase(
            o_acc, p_prev, kv_regs, st_o, pre, kv_tile, u, g0, masked=False, binit=()
        ):
            """Phase M: both GEMMs, their LDS operand reads, and this group's
            staging. `u` is the sub-trip position the LDS slots are keyed on."""
            # Priority over the SIMD partner (in phase V) for the MFMA burst.
            # Raw rocdl: FlyDSL 0.5.0 has no wave-priority API.
            rocdl.s_setprio(1)
            k_cur = u % K_SLOTS
            k_fill = (u + 1) % K_SLOTS
            v_cur = u % V_SLOTS
            v_prev = (u + 1) % V_SLOTS

            # GEMM1 of this tile and GEMM2 of the previous one (`p_prev`, V slot
            # v_prev). Tiles still enter the accumulator in ascending order.
            _N_STORE_K = NUM_BATCHES_KV * (VEC_KV // MFMA_DIM_LANE_K)
            _N_STORE_V = NUM_BATCHES_KV * VEC_KV

            def _store_k(k):
                store_k_regs_to_lds(kv_regs, k_smem[k_fill], part=k)

            def _store_v(k):
                store_v_regs_to_lds(kv_regs, v_smem[v_cur], part=k)

            if const_expr(g0):
                _ts = (_N_STORE_K, _store_k)
            else:
                _ts = (_N_STORE_V, _store_v)

            # BIAS_MASK: this tile's bias arrived from the previous phase V;
            # compute the next tile's slot rows under this burst's MFMAs.
            _bias = None
            _side = None
            if const_expr(masked and BIAS_MASK):
                _bias = unflatten(list(binit))
                _pcs, _pst = bias_row_pieces(
                    lambda: (kv_tile + fx.Int32(1)) * fx.Int32(BLOCK_N)
                )
                _side = (BIAS_SIDE_MFMA, _pcs, _pst)
            o_acc, s_accs, pre_next, brows_next = mfma_burst(
                o_acc,
                p_prev,
                pre,
                k_smem[k_cur],
                v_smem[v_prev],
                tail_store=_ts,
                bias=_bias,
                side=_side,
            )
            rocdl.sched_group_barrier(rocdl.mask_dswr, NUM_BATCHES_KV, 0)

            # Loaded now, stored in M(i+1): one tile further out.
            st_o = stage_advance(st_o)
            kv_regs_next = async_load_kv_regs(st_o)

            # Must be here, not at the end of phase V: the partner reads this
            # phase's LDS writes in the very next barrier interval (g1 reads
            # K(i+1); g0's tail prefetches V), and s_barrier does not retire LDS.
            # Deferring it is a race that passes most runs. It also orders this
            # phase's ds_reads before M(i+1)'s writes to the same slots. No
            # vmcnt: the staging load lands in registers.
            rocdl.s_waitcnt(lgkmcnt=0)
            rocdl.s_setprio(0)
            return o_acc, s_accs, kv_regs_next, st_o, pre_next, brows_next

        def gate_phase(kv_start, s_accs, apply_mask=True, brows=None, binit=()):
            """Phase V: the mask and the SiLU gate, plus (BIAS_MASK) the reads of
            the next tile's bias tiles."""
            if const_expr(apply_mask and BIAS_MASK):
                # The window is already in s_accs; only target q-blocks are
                # masked here.
                if const_expr(has_targets):
                    s_accs = mask_target_blocks(kv_start, s_accs)
                apply_mask = False
            p = gate_tile(kv_start, s_accs, apply_mask=apply_mask)
            if const_expr(brows is not None):
                # Next tile's GEMM1 initial accumulators, read under the barrier.
                bi = bias_init(brows)
                binit = [
                    bi[ng][qg]
                    for ng in range_constexpr(KV_SUBTILES)
                    for qg in range_constexpr(Q_SUBTILES)
                ]
            return [p[ng][qg] for ng in range_constexpr(KV_SUBTILES) for qg in range_constexpr(Q_SUBTILES)], list(binit)  # fmt: skip

        def unflatten(flat):
            return [
                [flat[ng * Q_SUBTILES + qg] for qg in range_constexpr(Q_SUBTILES)]
                for ng in range_constexpr(KV_SUBTILES)
            ]

        # Pure causal: tiles entirely below the diagonal skip the mask, giving an
        # unmasked prefix [kv_tile_start, unmasked_end) and a masked remainder.
        # With targets the prefix ends at min(q_start, max_id).
        CAUSAL_SPLIT = causal and not has_window and not has_contextual

        def split_carry(it):
            """Unpack one loop's carried values: accumulators, P packs, staged
            registers and voffsets, cross-phase prefetch, bias init."""
            it_list = list(it) if isinstance(it, (list, tuple)) else [it]
            o = N_ACC + N_PACK  # staged vectors
            a = o + NUM_BATCHES_KV  # staged voffsets (i32)
            pre = a + NUM_BATCHES_KV  # cross-phase operand prefetch
            return (
                [it_list[i] for i in range_constexpr(N_ACC)],
                unflatten([it_list[N_ACC + i] for i in range_constexpr(N_PACK)]),
                [it_list[o + i] for i in range_constexpr(NUM_BATCHES_KV)],
                [it_list[a + i] for i in range_constexpr(NUM_BATCHES_KV)],
                [it_list[pre + i] for i in range_constexpr(XPHASE_DEPTH)],
                [it_list[pre + XPHASE_DEPTH + i] for i in range_constexpr(N_BINIT)],
            )

        if active:
            acc_init = [c_zero_v4f32 for _ in range_constexpr(N_ACC)]

            # Prefix/remainder boundary; the prefix is empty unless CAUSAL_SPLIT.
            if const_expr(CAUSAL_SPLIT):
                split_pos = q_start
                if has_targets:
                    split_pos = (q_start < max_id).select(q_start, max_id)
                unmasked_end = split_pos // fx.Int32(BLOCK_N)
                unmasked_end = (unmasked_end > kv_tile_start).select(
                    unmasked_end, kv_tile_start
                )
                unmasked_end = (unmasked_end < n_tiles).select(unmasked_end, n_tiles)
                # Round down to a multiple of UNROLL, for slot parity and
                # because only the masked loop can pad; the tile handed back
                # is masked as a no-op.
                unmasked_end = kv_tile_start + (
                    ((unmasked_end - kv_tile_start) // fx.Int32(UNROLL))
                    * fx.Int32(UNROLL)
                )
            else:
                unmasked_end = kv_tile_start

            # ---- Prologue: K(first), and V(first)'s chunk 0 for the rotation ----
            first = kv_tile_start
            _t_first = fx.Int32(first) * fx.Int32(BLOCK_N)
            _boot = async_load_kv_regs(stage_seed(fx.Int32(first) - kv_stage_bump))
            # The carried staging set M(first) stores; the LUT and bias table
            # are built under both loads, since nothing else hides the prologue.
            carry_o = stage_seed(fx.Int32(first))
            carry_regs = async_load_kv_regs(carry_o)
            build_sigma_lut()
            if const_expr(BIAS_MASK):
                build_bias_table()
            # Only the bootstrap load; M(first)'s stores wait for the carried set.
            rocdl.s_waitcnt(vmcnt=NUM_BATCHES_KV)
            if is_g0:
                # The loop writes K(i+1), never K(first).
                store_k_regs_to_lds(_boot, k_smem[0])
            else:
                # vslot 1 (read by M(first)) holds chunk 0 of tile `first` and
                # chunks 1..D-1 of the nonexistent tile first-1, which must be
                # zero: GEMM2 multiplies P(-1) = 0 by them, 0 * NaN is NaN, and
                # LDS is not cleared between workgroups.
                store_v_regs_to_lds(
                    [
                        kv_rot_lane.select(
                            _boot[b], fx.full(VEC_KV, 0.0, elem_dtype).ir_value()
                        )
                        for b in range_constexpr(NUM_BATCHES_KV)
                    ],
                    v_smem[1],
                )

            # The skew: g0 takes one extra barrier here (g1 repays it after the
            # loops), so g0 is in phase M whenever g1 is in phase V. It must come
            # before the publish fence: trip 0 reads both groups' tiles, so the
            # fence has to be the last thing either group does before the loop.
            if is_g0:
                phase_barrier()
            rocdl.s_waitcnt(lgkmcnt=0)
            phase_barrier()

            # M(first)'s PV is on the nonexistent tile first-1: zero packs.
            pre_init = [c_zero_pack for _ in range_constexpr(XPHASE_DEPTH)]
            # Under BIAS_MASK the prefix is empty, so the first tile is unmasked_end.
            binit_init = []
            if const_expr(BIAS_MASK):
                _bi = bias_init(bias_rows(unmasked_end * fx.Int32(BLOCK_N)))
                binit_init = [
                    _bi[ng][qg]
                    for ng in range_constexpr(KV_SUBTILES)
                    for qg in range_constexpr(Q_SUBTILES)
                ]
            carry_init = (
                acc_init
                + [c_zero_pack for _ in range_constexpr(N_PACK)]
                + carry_regs
                + carry_o
                + pre_init
                + binit_init
            )

            # Trip counts, both multiples of UNROLL (slot parity).
            n_unmasked = unmasked_end - fx.Int32(kv_tile_start)
            # Plus one drain trip so the last tile's P meets a GEMM2; trips past
            # n_tiles are fully masked and add nothing.
            n_masked = (
                (n_tiles + fx.Int32(1) - unmasked_end + fx.Int32(UNROLL - 1))
                // fx.Int32(UNROLL)
            ) * fx.Int32(UNROLL)

            # ---- KV loops + epilogue, duplicated per staging duty ----
            # Branching on is_g0 inside every phase M cost 3.3%. The arms run to
            # the end of the block so nothing escapes the `if` (the loops yield,
            # so they cannot move into a helper). Both arms take the same
            # barriers.
            if is_g0:
                loop_results = carry_init
                # ---- Unmasked prefix (empty unless CAUSAL_SPLIT) ----
                for _o0, it in range(
                    fx.Int32(0),
                    n_unmasked // fx.Int32(UNROLL),
                    fx.Int32(1),
                    init=carry_init,
                ):  # ty: ignore
                    o_acc, p_prev, kv_regs, st_o, pre, binit = split_carry(it)
                    base = fx.Int32(kv_tile_start) + fx.Int32(_o0) * fx.Int32(UNROLL)
                    for u in range_constexpr(UNROLL):
                        # Rotated: the gate phase's closing barrier opens the next
                        # sub-trip, so the loop back edge sits in phase V.
                        if const_expr(u == 0):
                            phase_barrier()
                        kv_tile = base + fx.Int32(u)
                        o_acc, s_accs, kv_regs, st_o, pre, brows = mfma_phase(
                            o_acc,
                            p_prev,
                            kv_regs,
                            st_o,
                            pre,
                            kv_tile,
                            u,
                            True,
                            binit=binit,
                        )
                        phase_barrier()
                        p_flat, binit = gate_phase(
                            kv_tile * fx.Int32(BLOCK_N),
                            s_accs,
                            apply_mask=False,
                            brows=brows,
                            binit=binit,
                        )
                        p_prev = unflatten(p_flat)
                        if const_expr(u < UNROLL - 1):
                            phase_barrier()
                    loop_results = yield (o_acc + p_flat + kv_regs + st_o + pre + binit)

                # ---- Masked remainder + drain ----
                for _o1, it in range(
                    fx.Int32(0),
                    n_masked // fx.Int32(UNROLL),
                    fx.Int32(1),
                    init=loop_results,
                ):  # ty: ignore
                    o_acc, p_prev, kv_regs, st_o, pre, binit = split_carry(it)
                    base = unmasked_end + fx.Int32(_o1) * fx.Int32(UNROLL)
                    for u in range_constexpr(UNROLL):
                        if const_expr(u == 0):
                            phase_barrier()
                        kv_tile = base + fx.Int32(u)
                        o_acc, s_accs, kv_regs, st_o, pre, brows = mfma_phase(
                            o_acc,
                            p_prev,
                            kv_regs,
                            st_o,
                            pre,
                            kv_tile,
                            u,
                            True,
                            masked=True,
                            binit=binit,
                        )
                        phase_barrier()
                        p_flat, binit = gate_phase(
                            kv_tile * fx.Int32(BLOCK_N),
                            s_accs,
                            apply_mask=True,
                            brows=brows,
                            binit=binit,
                        )
                        p_prev = unflatten(p_flat)
                        if const_expr(u < UNROLL - 1):
                            phase_barrier()
                    loop_results = yield (o_acc + p_flat + kv_regs + st_o + pre + binit)

                # Closes the last gate phase (the loop is rotated).
                phase_barrier()

                results = (
                    list(loop_results)
                    if isinstance(loop_results, (list, tuple))
                    else [loop_results]
                )
                store_o(results, 0)
            else:
                loop_results = carry_init
                # ---- Unmasked prefix (empty unless CAUSAL_SPLIT) ----
                for _o0, it in range(
                    fx.Int32(0),
                    n_unmasked // fx.Int32(UNROLL),
                    fx.Int32(1),
                    init=carry_init,
                ):  # ty: ignore
                    o_acc, p_prev, kv_regs, st_o, pre, binit = split_carry(it)
                    base = fx.Int32(kv_tile_start) + fx.Int32(_o0) * fx.Int32(UNROLL)
                    for u in range_constexpr(UNROLL):
                        if const_expr(u == 0):
                            phase_barrier()
                        kv_tile = base + fx.Int32(u)
                        o_acc, s_accs, kv_regs, st_o, pre, brows = mfma_phase(
                            o_acc,
                            p_prev,
                            kv_regs,
                            st_o,
                            pre,
                            kv_tile,
                            u,
                            False,
                            binit=binit,
                        )
                        phase_barrier()
                        p_flat, binit = gate_phase(
                            kv_tile * fx.Int32(BLOCK_N),
                            s_accs,
                            apply_mask=False,
                            brows=brows,
                            binit=binit,
                        )
                        p_prev = unflatten(p_flat)
                        if const_expr(u < UNROLL - 1):
                            phase_barrier()
                    loop_results = yield (o_acc + p_flat + kv_regs + st_o + pre + binit)

                # ---- Masked remainder + drain ----
                for _o1, it in range(
                    fx.Int32(0),
                    n_masked // fx.Int32(UNROLL),
                    fx.Int32(1),
                    init=loop_results,
                ):  # ty: ignore
                    o_acc, p_prev, kv_regs, st_o, pre, binit = split_carry(it)
                    base = unmasked_end + fx.Int32(_o1) * fx.Int32(UNROLL)
                    for u in range_constexpr(UNROLL):
                        if const_expr(u == 0):
                            phase_barrier()
                        kv_tile = base + fx.Int32(u)
                        o_acc, s_accs, kv_regs, st_o, pre, brows = mfma_phase(
                            o_acc,
                            p_prev,
                            kv_regs,
                            st_o,
                            pre,
                            kv_tile,
                            u,
                            False,
                            masked=True,
                            binit=binit,
                        )
                        phase_barrier()
                        p_flat, binit = gate_phase(
                            kv_tile * fx.Int32(BLOCK_N),
                            s_accs,
                            apply_mask=True,
                            brows=brows,
                            binit=binit,
                        )
                        p_prev = unflatten(p_flat)
                        if const_expr(u < UNROLL - 1):
                            phase_barrier()
                    loop_results = yield (o_acc + p_flat + kv_regs + st_o + pre + binit)

                # Closes the last gate phase (the loop is rotated).
                phase_barrier()

                # g1 repays the skew barrier.
                phase_barrier()

                results = (
                    list(loop_results)
                    if isinstance(loop_results, (list, tuple))
                    else [loop_results]
                )
                store_o(results, 1)

    _hstu_compile_hints = {
        "fast_fp_math": True,
        "unsafe_fp_math": True,
    }

    @flyc.jit
    def launch_hstu_attention_fwd_pp(
        max_seq_len: fx.Int32,
        batch: fx.Int32,
        q: fx.Tensor,
        k: fx.Tensor,
        v: fx.Tensor,
        seq_offsets: fx.Tensor,
        num_targets: fx.Tensor,
        out: fx.Tensor,
        stream: fx.Stream,
    ) -> None:
        c_num_heads = fx.Int32(num_heads)
        c_ngg = fx.Int32(NUM_GRID_GROUPS)
        num_q_tiles = (max_seq_len + fx.Int32(BLOCK_M - 1)) // fx.Int32(BLOCK_M)
        hz_total = batch * c_num_heads
        hz_per_group = (hz_total + fx.Int32(NUM_GRID_GROUPS - 1)) // c_ngg
        grid_blocks = num_q_tiles * hz_per_group * c_ngg

        # Epilogue scale alpha/N: silu(alpha*s) = alpha * p, normalized by the
        # global max_seq_len, not the row's length.
        inv_n = (fx.Float32(1.0) / max_seq_len) * fx.Float32(alpha)

        hstu_attention_fwd_pp_kernel(
            q,
            k,
            v,
            seq_offsets,
            num_targets,
            out,
            num_q_tiles,
            hz_per_group,
            hz_total,
            inv_n,
            value_attrs={
                "rocdl.waves_per_eu": WAVES_PER_EU,
                "rocdl.flat_work_group_size": f"{BLOCK_THREADS},{BLOCK_THREADS}",
            },
        ).launch(
            grid=(grid_blocks, 1, 1),
            block=BLOCK_THREADS,
            smem=0,
            stream=stream,
        )

    launch_hstu_attention_fwd_pp.compile_hints = _hstu_compile_hints
    return flyc.compile[_PP_COMPILE_HINTS](launch_hstu_attention_fwd_pp)
