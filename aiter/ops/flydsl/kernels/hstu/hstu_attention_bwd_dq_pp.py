# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Ping-pong-pipelined variant of the query-owned HSTU backward dQ kernel.

Same mathematics as `hstu_attention_bwd_dq.py` -- see that module's docstring --
but scheduled so the two waves resident on each SIMD are held in *opposite*
phases:

    phase M : the 48 MFMAs, their LDS operand feed, and this group's DMA.
    phase V : the mask and the SiLU' gate.  No memory operations at all.

One wave group is in M while the other is in V, so the matrix core and the VALU
pipe are both busy continuously.  Without this the two resident waves sit in
lockstep on a shared `gpu.barrier()`: they idle together and they burst MFMA
together, contending for the one matrix core.

Ported from `hstu_attention_bwd_dvdk_pp.py`, which carries the long-form
rationale for every scheduling device used here.  The structure mirrors it with
the ownership inverted -- dQ owns query rows and streams K/V, so the two staging
duties are K and V rather than Q and dO.

Three structural points, each load-bearing:

* **GEMM3 consumes the previous tile's dS.**  `dQ += dS * K` cannot run in the
  same phase as the GEMMs that produce `dS` -- the gate sits between them.  At
  trip `i` GEMM3 therefore uses `ds_prev` and K tile `i-1`, which merges it with
  tile `i`'s GEMM1/GEMM2 into one contiguous 48-MFMA burst.  Seeding `ds_prev`
  to zero makes trip 0 self-starting and the extra trip the drain, so no
  iteration is peeled.
* **The streamed K tile is 3-deep, not 2.**  Slot `i-1` is still being read by
  GEMM3 in the very phase the DMA fills slot `i+1`; at depth 2 those are the
  same slot, and the conflict is cross-group (`vmcnt`/`lgkmcnt` are per-wave),
  so it cannot be fenced away.  V is 2-deep by need but 3-deep in fact, so that
  `UNROLL` is 3 rather than `lcm(3, 2)`.
* **No transposed tiles.**  GEMM3's B operand wants K with the kv axis packed,
  which the row-major tile serves as a 4-element gather.  Those 16 scalar
  `ds_read_u16` sit in phase M, which is the *non*-critical phase, and the dV/dK
  effort established that LDS work there is off the critical path (removing 79%
  of it measured slower).  Revisit only if a trace shows M >= V.

`validate_hstu_attention_bwd_dq_pp` hard-gates the supported shape: this kernel
is specialized for gfx942 / bf16 / head_dim == hidden_dim == 64 and the
`(512, 16, 8)` tile, and raises for anything else.

Why that tile, measured (`tmp/hstu_repro/control_dq.py`, B=1024 H=4 N=16384):
going to 8 waves in one workgroup costs **+12.8%** against the tuned
`(128, 32, 2, 2)` before any ping-pong, almost all of it in the step from two
co-resident workgroups to one.  `ROWS_PER_WAVE` must stay 64 -- halving it to 32
costs +39% to +55%, because each streamed K/V tile then amortizes over half the
math while every per-tile cost is unchanged.  With `ROWS_PER_WAVE = 64` fixed,
`BLOCK_M = 8 * 64 = 512` follows, and `BLOCK_N = 16` is the only width that
leaves register headroom (`BLOCK_N = 32` is already at 256 VGPR before the
pipeline adds anything).  **The baseline this kernel has to beat is the 262.7 ms
of the unmodified kernel at this same tile, not the 225.2 ms tuned number.**
"""

import functools
import math as host_math

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir.dialects import llvm
from flydsl.expr import arith, const_expr, gpu, range_constexpr, rocdl
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
    decode_lane,
    exp2_f32,
    grouped_loader,
    make_lds_dma,
    phase_barrier,
    swz_col,
)

# NOTE: these are read inside the traced kernel body. FlyDSL's disk cache keys
# on the kernel function's source text and its closure values -- a module global
# is NEITHER, so editing one here does NOT invalidate a cached binary and the
# next run silently measures the old schedule. When tuning them, either
# `rm -rf ~/.flydsl/cache` or set FLYDSL_RUNTIME_ENABLE_CACHE=0.
#
# Depth of the rolling LDS operand window, in operand groups.
#
# Swept with `tmp/hstu_repro/sweep_dq_pp.py` (which checks accuracy at every
# point -- an unchecked sweep on the dV/dK twin once reported a 1% "win" from a
# kernel with 23-43% mismatch). There is a cliff below 9:
#     W=11  209.3 / 210.6     W=7  228.3
#     W=9   208.1 / 208.4     W=5  231.2      W=3  228.2
# 9 of 12 leaves the last three groups' reads to issue inside the burst, which
# costs nothing and holds two fewer operands live -- worth having at 255 VGPR.
PREFETCH_W = 9
# Operand groups per MFMA phase: 3 GEMMs x (head_dim / MFMA_M) chunks.
NUM_OPERAND_GROUPS = 12

# Issue order of those groups, as a permutation.
#
# Groups 0..3 are GEMM3, whose four chunks touch four different `dq_acc`
# registers and so have no chain between them; they lead because both their
# operands (`ds_prev` in registers, K tile i-1 in LDS) are ready at phase start.
#
# Groups 4..7 (GEMM1) and 8..11 (GEMM2) each accumulate over `ks` into the SAME
# register per query sub-tile -- a 4-deep dependency chain with only
# Q_OWNED_SUBTILES (4) independent MFMAs between consecutive links. At ~4 cy of
# issue each that is 16 cy against a ~16 cy MFMA result latency, right at the
# edge. Interleaving the two independent GEMMs doubles the spacing to 8 MFMAs.
# Pure reordering: every accumulator still sees its `ks` steps in the same
# sequence, so the arithmetic is unchanged.
GROUP_ORDER = [0, 1, 2, 3, 4, 8, 5, 9, 6, 10, 7, 11]
assert sorted(GROUP_ORDER) == list(range(NUM_OPERAND_GROUPS))

# Operand-group step at which each of this group's DMA passes is fired. Spread
# through the burst's slack so the load has most of the phase to return.
DMA_FIRE_STEPS = (2, 5)

# Wave priority in each phase. The two anti-phased waves on a SIMD compete for
# issue slots, and `s_setprio` decides who wins. The dV/dK twin raises priority
# during M -- but the phase that matters is whichever is LONGER, since the
# barrier interval is max(M, V), and here that is V.
PRIO_M = 1
PRIO_V = 0

# Where the causal/window mask is computed. It needs no GEMM result, so it can
# sit in either phase; the barrier interval is max(M, V) and V is the longer
# one here, so moving it to M is arithmetically free. It was a measured LOSS on
# the dV/dK twin -- but that kernel's M had no slack, and this one's has 308 cy.
# Either way it must land AFTER the sched_barrier-fenced MFMA burst: interleaved
# between the fenced groups it is pinned between MFMA issues and stretches the
# stream (+9% there).
MASK_IN_M = False

# Apply the XOR column swizzle to the streamed V tile as well as K.
#
# KEEP THIS FALSE. The conflicts it removes are real and large, and removing
# them makes the kernel 10.5% SLOWER. Left as a knob so the measurement is
# reproducible rather than re-derived.
#
# The diagnosis: V's row stride is head_dim = 64 elements = 32 dwords = exactly
# the LDS bank count, so `bank` is independent of the row and GEMM2's operand
# read puts all 16 rows on one bank -- a 16-way conflict, against the 2x floor
# K reaches with the swizzle. Counters agree: over the whole kernel
# SQ_LDS_BANK_CONFLICT / SQ_LDS_IDX_ACTIVE = 79%, 4.7x the conflict-free floor,
# essentially all of it this one read.
#
# Measured, ATT phase split (`tmp/run_att.sh dq_pp`, `tmp/att_phases.py`):
#
#                     M phase            V phase           dq
#     unswizzled      1676 cy (1119)     1992 cy (252)     206.8 ms
#     swizzled        1324 cy ( 752)     2864 cy (582)     228.6 ms
#
# The swizzle does exactly what it promises -- M drops 21% and its LDS stall a
# third. It buys nothing, because **M is not the critical phase**: the barrier
# interval is max(M, V) and V already led by 316 cy. Meanwhile V degrades by
# 872 cy and the per-trip total rises 3668 -> 4188. Two contributing costs: the
# XOR makes V's addresses non-affine, so the backend can no longer fuse
# `ds_read_b64` pairs into `ds_read2_b64` (3 fused + 18 single -> 0 + 24), and
# the address arithmetic adds VALU to M (28.7 -> 33.6). Neither is spilling:
# 255 VGPR, 0 spills, both ways.
#
# The general rule this is an instance of: bank conflicts are worth measuring
# to find, and worth fixing only in the phase that sets `max(M, V)`.
SWIZZLE_V = False


def validate_hstu_attention_bwd_dq_pp(
    num_heads: int,
    head_dim: int,
    hidden_dim: int,
    causal: bool,
    max_attn_len: int,
    contextual_seq_len: int,
    has_targets: bool,
    alpha: float,
    dtype_str: str,
    max_seq_len: int,
    *,
    block_m: int,
    block_n: int,
    num_waves: int,
    waves_per_eu: int,
    arch: str | None = None,
) -> None:
    """Hard gate for the ping-pong dQ kernel.

    Deliberately strict: this kernel hardcodes a single measured shape, so
    anything outside it must fall back to `build_hstu_attention_bwd_dq` rather
    than silently compile a variant that was never measured.
    """
    if arch is None:
        arch = get_rocm_arch()
    if not (arch or "").startswith("gfx942"):
        raise ValueError(
            f"hstu bwd dq ping-pong requires gfx942, got {arch!r}. The schedule "
            "assumes 16x16x16 MFMA timing and a dword LDS DMA."
        )
    if dtype_str != "bf16":
        raise ValueError(f"hstu bwd dq ping-pong requires bf16, got {dtype_str!r}")
    if not causal:
        raise ValueError("hstu_attention_bwd only supports causal attention")
    if head_dim != 64 or hidden_dim != 64:
        raise ValueError(
            "hstu bwd dq ping-pong requires head_dim == hidden_dim == 64, got "
            f"{head_dim}/{hidden_dim}. Both groups share one staging code path, "
            "which relies on the K and V tiles being structurally identical."
        )
    if (block_m, block_n, num_waves) != (512, 16, 8):
        raise ValueError(
            "hstu bwd dq ping-pong requires (block_m, block_n, num_waves) = "
            f"(512, 16, 8), got ({block_m}, {block_n}, {num_waves}). 8 waves give "
            "2 per SIMD inside one barrier domain; 512 keeps 64 query rows per "
            "wave, and halving that costs 39-55% to lost reuse."
        )
    if num_heads <= 0:
        raise ValueError(f"num_heads must be positive, got {num_heads}")
    if max_seq_len <= 0:
        raise ValueError(f"max_seq_len must be positive, got {max_seq_len}")
    if waves_per_eu < 0:
        raise ValueError(f"waves_per_eu must be non-negative, got {waves_per_eu}")
    if not host_math.isfinite(alpha):
        raise ValueError(f"alpha must be finite, got {alpha}")
    if contextual_seq_len < 0:
        raise ValueError(
            f"contextual_seq_len must be non-negative, got {contextual_seq_len}"
        )
    if max_attn_len < 0:
        raise ValueError(f"max_attn_len must be non-negative, got {max_attn_len}")


@functools.lru_cache(maxsize=16384)
def build_hstu_attention_bwd_dq_pp(
    num_heads: int,
    head_dim: int,
    hidden_dim: int,
    causal: bool,
    max_attn_len: int,
    contextual_seq_len: int,
    has_targets: bool,
    alpha: float,
    dtype_str: str,
    max_seq_len: int,
    *,
    block_m: int = 512,
    block_n: int = 16,
    num_waves: int = 8,
    waves_per_eu: int = 2,
    has_perm: bool = False,
):
    validate_hstu_attention_bwd_dq_pp(
        num_heads,
        head_dim,
        hidden_dim,
        causal,
        max_attn_len,
        contextual_seq_len,
        has_targets,
        alpha,
        dtype_str,
        max_seq_len,
        block_m=block_m,
        block_n=block_n,
        num_waves=num_waves,
        waves_per_eu=waves_per_eu,
    )

    # BLOCK_M = owned query tile (dQ output rows). BLOCK_N = streamed KV tile.
    BLOCK_M = block_m
    BLOCK_N = block_n
    NUM_WAVES = num_waves
    BLOCK_THREADS = NUM_WAVES * WARP_SIZE
    ROWS_PER_WAVE = BLOCK_M // NUM_WAVES
    Q_OWNED_SUBTILES = ROWS_PER_WAVE // MFMA_M
    WAVES_PER_EU = waves_per_eu
    MFMA_QK_K, MFMA_QK_LANE_K, _ = _mfma_params_for_dim(head_dim)
    MFMA_DA_K, MFMA_DA_LANE_K, _ = _mfma_params_for_dim(hidden_dim)
    DMA_BYTES, DMA_ELEMS, K_SWZ_ROWS, K_SWZ_SHIFT = _arch_dma_params()

    elem_dtype = _dtype_to_elem_type(dtype_str)
    has_window = max_attn_len > 0
    has_contextual = contextual_seq_len > 0

    # head_dim == hidden_dim == 64 (gated above), so every per-dimension count
    # collapses to one value and the K and V tiles are structurally identical.
    K_STEPS = head_dim // MFMA_QK_K
    DK_STEPS = hidden_dim // MFMA_DA_K
    HC_CHUNKS = head_dim // MFMA_M
    STRIDE = head_dim
    assert K_STEPS == DK_STEPS == HC_CHUNKS == 4
    assert MFMA_QK_LANE_K == MFMA_DA_LANE_K == MFMA_LANE_K
    assert BLOCK_N == MFMA_N, "KV_SUBTILES must be 1"
    assert Q_OWNED_SUBTILES == 4
    assert NUM_OPERAND_GROUPS == 3 * HC_CHUNKS

    num_q_tiles = (max_seq_len + BLOCK_M - 1) // BLOCK_M
    stride_qk_n = num_heads * head_dim

    # ---- ping-pong groups -------------------------------------------------
    # g0 = waves 0..3 stages K; g1 = waves 4..7 stages V.  Wave i and wave i+4
    # share a SIMD, which is the pair the phase alternation (and `s_setprio`)
    # arbitrates.
    GRP_WAVES = NUM_WAVES // 2
    GRP_THREADS = GRP_WAVES * WARP_SIZE
    assert GRP_THREADS == 256

    # ---- LDS geometry -----------------------------------------------------
    # Three slots per streamed tensor, every slot its own `fx.Array` field --
    # see SharedStorage.
    K_SLOTS = 3
    V_SLOTS = 3
    # The loop body is unrolled by the slot count so each sub-trip names a
    # *literal* slot. That is what keeps the slot index compile-time, which is
    # what keeps each access provably inside one LDS global.
    UNROLL = K_SLOTS
    assert UNROLL % K_SLOTS == 0 and UNROLL % V_SLOTS == 0
    TILE_ELEMS = BLOCK_N * STRIDE

    tile_elems_per_pass = GRP_THREADS * DMA_ELEMS
    assert TILE_ELEMS % tile_elems_per_pass == 0
    NUM_DMA = TILE_ELEMS // tile_elems_per_pass
    assert NUM_DMA == len(DMA_FIRE_STEPS)
    PAIRS_PER_ROW = STRIDE // DMA_ELEMS

    N_ACC = HC_CHUNKS * Q_OWNED_SUBTILES
    N_PACK = Q_OWNED_SUBTILES

    # One field per ROTATION SLOT, not one array per tensor with the slot as an
    # offset. Each `fx.Array` field becomes its own LDS global, and only then
    # can the backend prove that the `buffer_load_lds` filling slot i+1 is
    # disjoint from the `ds_read`s feeding the MFMAs out of slots i and i-1 --
    # which is the entire point of multiple buffering: the write and the reads
    # have to proceed at the same time.
    #
    # With one fused array and a runtime slot offset the accesses are only
    # provably inside "the k array", so every read may-alias the in-flight DMA
    # and the backend serializes them. Measured on the dV/dK twin: ~8%.
    @fx.struct
    class SharedStorage:
        k0: fx.Array[elem_dtype, TILE_ELEMS, 16]
        k1: fx.Array[elem_dtype, TILE_ELEMS, 16]
        k2: fx.Array[elem_dtype, TILE_ELEMS, 16]
        v0: fx.Array[elem_dtype, TILE_ELEMS, 16]
        v1: fx.Array[elem_dtype, TILE_ELEMS, 16]
        v2: fx.Array[elem_dtype, TILE_ELEMS, 16]

    @flyc.kernel(known_block_size=[BLOCK_THREADS, 1, 1])
    def hstu_attention_bwd_dq_pp(
        q: fx.Tensor,
        k: fx.Tensor,
        v: fx.Tensor,
        do: fx.Tensor,
        seq_offsets: fx.Tensor,
        num_targets: fx.Tensor,
        perm: fx.Tensor,
        dq: fx.Tensor,
        hz_per_group: fx.Int32,
        hz_total: fx.Int32,
    ) -> None:
        elem_type = elem_dtype.ir_type

        # QK and V*dO use architecture-native dimension-axis MFMAs. The dQ
        # reduction stays 16-deep to reuse dS without lane shuffles.
        # Matching MFMA shapes share one accumulator binding.
        qk_mfma_acc, da_mfma_acc, mfma_acc = bind_mfma_accs(
            elem_dtype,
            (MFMA_QK_K, MFMA_QK_LANE_K),
            (MFMA_DA_K, MFMA_DA_LANE_K),
            (MFMA_K, MFMA_LANE_K),
        )

        tid = fx.Int32(gpu.thread_idx.x)
        wave_id, _lane, lane_div_16, lane_mod_16 = decode_lane(
            tid, NUM_WAVES, WARP_SIZE, MFMA_N
        )

        # The group predicate must be provably uniform or the backend will not
        # select `buffer_load_lds` on a selected descriptor. readfirstlane makes
        # the divergence analysis treat it as scalar.
        tid_u = fx.Int32(rocdl.readfirstlane(fx.Int32.ir_type, tid))
        is_g0 = tid_u < fx.Int32(GRP_THREADS)
        grp_tid = tid - is_g0.select(fx.Int32(0), fx.Int32(GRP_THREADS))
        grp_wave = grp_tid // fx.Int32(WARP_SIZE)

        # ---- Group-major grid decode -> (batch_idx, head_idx, q_tile_idx) ----
        block_id = fx.Int32(gpu.block_idx.x)
        grid_group = block_id % fx.Int32(NUM_GRID_GROUPS)
        pos_in_group = block_id // fx.Int32(NUM_GRID_GROUPS)
        local_hz_idx = pos_in_group // fx.Int32(num_q_tiles)
        q_tile_idx = pos_in_group % fx.Int32(num_q_tiles)
        hz_idx = grid_group * hz_per_group + local_hz_idx
        # Clamp padded grid blocks for safe metadata reads, then mark them
        # inactive. `block_valid` is block-uniform, so an inactive workgroup
        # skips the loop -- and therefore its whole barrier sequence -- in BOTH
        # groups together. Any group-dependent skip here would hang the kernel
        # rather than return wrong numbers.
        block_valid = hz_idx < hz_total
        hz_idx = block_valid.select(hz_idx, fx.Int32(0))
        batch_idx = hz_idx // fx.Int32(num_heads)
        head_idx = hz_idx % fx.Int32(num_heads)
        if const_expr(has_perm):
            batch_idx = fx.Int32(perm[batch_idx])

        seq_start = fx.Int32(seq_offsets[batch_idx])
        seq_len = fx.Int32(seq_offsets[batch_idx + fx.Int32(1)]) - seq_start
        seq_len = block_valid.select(seq_len, fx.Int32(0))

        # Apply contextual shifting before target-tail clamping.
        num_target = fx.Int32(0)
        if has_targets:
            num_target = fx.Int32(num_targets[batch_idx])
        max_id = seq_len
        if has_contextual:
            max_id = seq_len - fx.Int32(contextual_seq_len) + fx.Int32(1)
        if has_targets:
            max_id = (num_target > fx.Int32(0)).select(max_id - num_target, max_id)

        def to_id(x):
            xid = x
            if has_contextual:
                xid = xid - fx.Int32(contextual_seq_len - 1)
                xid = (xid < fx.Int32(0)).select(fx.Int32(0), xid)
            if has_targets:
                xid = (xid > max_id).select(max_id, xid)
            return xid

        c_zero_qk_pack = Vec.filled(MFMA_QK_LANE_K, 0.0, elem_dtype).ir_value()
        c_zero_da_pack = Vec.filled(MFMA_DA_LANE_K, 0.0, elem_dtype).ir_value()
        c_zero_pack = Vec.filled(MFMA_LANE_K, 0.0, elem_dtype).ir_value()
        c_zero_v4f32 = Vec.filled(MFMA_ELEMS_PER_LANE, 0.0, fx.Float32).ir_value()

        q_load = grouped_loader(q, head_dim, MFMA_QK_LANE_K)
        do_load = grouped_loader(do, hidden_dim, MFMA_DA_LANE_K)

        q_head_offset = head_idx * fx.Int32(head_dim)
        k_base_byte_offset = (
            fx.Int64(seq_start) * fx.Int64(stride_qk_n) + fx.Int64(q_head_offset)
        ) * fx.Int64(2)

        # ---- LDS views ----------------------------------------------------
        # One view per (tensor, slot), each built straight off its own field's
        # pointer. The slot is a Python index resolved at trace time, so every
        # access is statically inside exactly one LDS global and the backend can
        # cross-noalias them.
        #
        # Do NOT reintroduce a runtime slot offset here, and never form an LDS
        # address with `fx.inttoptr` or a `select` between two pointers: both
        # break the provenance walk, after which SIInsertWaitcnts cannot prove a
        # `ds_read` misses the in-flight `buffer_load_lds` and plants a
        # `vmcnt(0)` in front of every operand read (measured on the dV/dK twin:
        # 19.8 M stall cycles, matrix-core duty down to 22%).
        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        tile_layout = fx.make_layout(
            (BLOCK_N, STRIDE // MFMA_LANE_K, MFMA_LANE_K),
            (STRIDE, MFMA_LANE_K, 1),
        )
        k_views = [
            lds.k0.view(tile_layout),
            lds.k1.view(tile_layout),
            lds.k2.view(tile_layout),
        ]
        v_views = [
            lds.v0.view(tile_layout),
            lds.v1.view(tile_layout),
            lds.v2.view(tile_layout),
        ]

        def k_swz_col(tile_row, col):
            return swz_col(tile_row, col, K_SWZ_ROWS, K_SWZ_SHIFT)

        # ---- Owned Q rows / bounds per query sub-tile ----
        q_wave_base = q_tile_idx * fx.Int32(BLOCK_M) + wave_id * fx.Int32(ROWS_PER_WAVE)
        q_rows = []
        q_in_bounds = []
        for qg in range_constexpr(Q_OWNED_SUBTILES):
            local = q_wave_base + fx.Int32(qg * MFMA_M) + lane_mod_16
            q_rows.append(local)
            q_in_bounds.append(local < seq_len)
        q_row_ids = [to_id(q_rows[qg]) for qg in range_constexpr(Q_OWNED_SUBTILES)]

        # ---- Resident Q / dO B-operand packs, loaded once ------------------
        q_packs = []  # q_packs[ks][qg], B operand of GEMM1
        for ks in range_constexpr(K_STEPS):
            q_col = fx.Int32(ks * MFMA_QK_K) + lane_div_16 * fx.Int32(MFMA_QK_LANE_K)
            per_qg = []
            for qg in range_constexpr(Q_OWNED_SUBTILES):
                safe = q_in_bounds[qg].select(seq_start + q_rows[qg], seq_start)
                raw = q_load(
                    fx.Int64(safe), head_idx, q_col // fx.Int32(MFMA_QK_LANE_K)
                ).ir_value()
                # Fold `alpha` into Q once, here, so GEMM1 yields the SCALED
                # score and the gate's `sc = s * alpha` disappears -- 16
                # element-multiplies per gate phase. The dV/dK twin folds into
                # K instead, because there K is the register-resident operand;
                # here the polarity is inverted (K is streamed through LDS, and
                # `buffer_load_lds` cannot transform data in flight), so Q is
                # the one to scale. Only GEMM1 reads q_packs.
                #
                # dQ's own chain rule is untouched: dQ = alpha * dZ * K still
                # needs the alpha that the epilogue applies -- a DIFFERENT
                # alpha from this one, so both must stay.
                #
                # The cost is that Q is bf16, so bf16(alpha*Q) rounds a second
                # time -- an error the fp32 reference does not have.
                scaled = (Vec(raw).to(fx.Float32) * fx.Float32(alpha)).to(elem_dtype)
                per_qg.append(
                    q_in_bounds[qg].select(scaled.ir_value(), c_zero_qk_pack)
                )
            q_packs.append(per_qg)

        do_packs = []  # do_packs[ks][qg], B operand of GEMM2
        for ks in range_constexpr(DK_STEPS):
            d_col = fx.Int32(ks * MFMA_DA_K) + lane_div_16 * fx.Int32(MFMA_DA_LANE_K)
            per_qg = []
            for qg in range_constexpr(Q_OWNED_SUBTILES):
                safe = q_in_bounds[qg].select(seq_start + q_rows[qg], seq_start)
                raw = do_load(
                    fx.Int64(safe), head_idx, d_col // fx.Int32(MFMA_DA_LANE_K)
                ).ir_value()
                per_qg.append(q_in_bounds[qg].select(raw, c_zero_da_pack))
            do_packs.append(per_qg)

        c_neg_log2e = fx.Float32(-_LOG2E)
        c_zero_f = fx.Float32(0.0)
        # dS no longer carries 1/N (see gate_phase), so the dQ store applies
        # alpha/N in the one multiply it was already doing.
        c_dq_scale = fx.Float32(alpha / max_seq_len)

        # ---- Streamed KV range: causal upper bound, window lower bound ----
        q_start = q_tile_idx * fx.Int32(BLOCK_M)
        q_end = q_start + fx.Int32(BLOCK_M)
        active = q_start < seq_len
        clamped = (q_end < seq_len).select(q_end, seq_len)
        # Logical row zero attends the full contextual prefix above its diagonal.
        base_upper = clamped
        if has_contextual:
            ctx_block = q_start < fx.Int32(contextual_seq_len)
            base_upper = ctx_block.select(seq_len, clamped)
        kv_upper = active.select(base_upper, fx.Int32(0))
        n_tiles = (kv_upper + fx.Int32(BLOCK_N - 1)) // fx.Int32(BLOCK_N)

        # Sliding-window lower bound: skip fully-masked low KV tiles.
        kv_tile_start = fx.Int32(0)
        if has_window:
            eff_q_low = (q_start < max_id).select(q_start, max_id)
            kv_lower = eff_q_low - fx.Int32(max_attn_len)
            kv_lower = (kv_lower > fx.Int32(0)).select(kv_lower, fx.Int32(0))
            win_tile_start = kv_lower // fx.Int32(BLOCK_N)
            kv_tile_start = win_tile_start
            if has_contextual:
                ctx_prefix_block = q_start < fx.Int32(contextual_seq_len)
                kv_tile_start = ctx_prefix_block.select(fx.Int32(0), win_tile_start)

        # ---- staging duty: global -> LDS ----------------------------------
        # g0 stages K, g1 stages V. The two duties are built as separate
        # closures and dispatched on a wave-uniform `if`, rather than selecting
        # between two pointers -- see the provenance note above.
        _dma_atom, _dma_ptr_ty, _rebased_buffer_div = make_lds_dma(DMA_BYTES, elem_type)
        c_stride_n = fx.Int32(stride_qk_n)
        c_dma_elems = fx.Int32(DMA_ELEMS)
        c_pairs_per_row = fx.Int32(PAIRS_PER_ROW)

        def make_dma(tensor, base_byte_off, dst_views, swizzled):
            # num_records spans the whole tensor, and the row clamp below folds
            # every out-of-range row onto token 0. That is deliberately NOT the
            # dV/dK twin's seq-len-bounded descriptor: an out-of-range
            # `buffer_load ... lds` does not deposit zero, it leaves the LDS
            # slot UNWRITTEN, which would make the tail rows of the last tile
            # whatever the previous dispatch left there. Clamping instead means
            # every slot is always fully written with finite data, and no
            # prologue zeroing of the DMA-filled tiles is needed at all.
            div = _rebased_buffer_div(
                fx.get_iter(tensor), base_byte_off, max_seq_len * stride_qk_n
            )
            rows, gcols = [], []
            for d in range_constexpr(NUM_DMA):
                pair = grp_tid + fx.Int32(d * GRP_THREADS)
                row = pair // c_pairs_per_row
                col = (pair % c_pairs_per_row) * c_dma_elems
                rows.append(row)
                gcols.append(k_swz_col(row, col) if const_expr(swizzled) else col)
            # The DMA destination must be wave-uniform in an SGPR, so it is the
            # one LDS address that still goes through `inttoptr` -- that is how
            # `m0` is written. One base per slot, so the slot never becomes a
            # runtime offset.
            lane0 = []
            for view in dst_views:
                wave_byte = fx.Int32(
                    fx.ptrtoint(fx.get_iter(view))
                ) + grp_wave * fx.Int32(WARP_SIZE * DMA_BYTES)
                lane0.append(rocdl.readfirstlane(fx.Int32.ir_type, wave_byte))

            def fire(slot, kv_start, d):
                row = rows[d]
                in_bounds = (kv_start + row) < seq_len
                local_tok = in_bounds.select(kv_start + row, fx.Int32(0))
                src_elem = local_tok * c_stride_n + gcols[d]
                lds_byte = fx.Int32(lane0[slot]) + fx.Int32(
                    d * GRP_THREADS * DMA_BYTES
                )
                dst = fx.make_view(
                    fx.inttoptr(_dma_ptr_ty, lds_byte), fx.make_layout(1, 1)
                )
                fx.copy(_dma_atom, fx.slice(div, (None, fx.Int32(src_elem))), dst)

            return fire

        # K is column-swizzled for the MFMA reads; V stays row-major, exactly as
        # in the shipping kernel.
        dma_k = make_dma(k, k_base_byte_offset, k_views, True)
        dma_v = make_dma(v, k_base_byte_offset, v_views, SWIZZLE_V)

        def k_zero(slot):
            """Zero the one K slot trip 0's GEMM3 reads but no DMA ever fills.

            `ds_prev` is seeded to zero so trip 0 self-starts, but `0 * NaN` is
            NaN and would poison `dq_acc` for the whole sweep, so that slot has
            to hold finite data. Only g0 does this: g0 owns K, and a
            cross-group zeroing would race the DMA (`lgkmcnt` is per-wave, so
            g1's `ds_write` can retire after g0's `buffer_load ... lds`).
            """
            z = Vec(c_zero_pack)
            row = grp_tid // fx.Int32(STRIDE // MFMA_LANE_K)
            colgrp = grp_tid % fx.Int32(STRIDE // MFMA_LANE_K)
            k_views[slot][row, colgrp, None].store(z)

        # ---- the 12 MFMA operand groups ------------------------------------
        # g 0..3   GEMM3 dQ: B = K[kv, hc] tile i-1 (gather)  A = ds_prev (reg)
        # g 4..7   GEMM1 S^T: A = K[kv, d]  tile i            B = q_packs  (reg)
        # g 8..11  GEMM2 dA : A = V[kv, d]  tile i            B = do_packs (reg)
        def read_operand(g, k_cur, k_prev):
            if const_expr(g < HC_CHUNKS):
                # GEMM3's B operand wants B[k = kv, n = hc]: lane (m = lane%16,
                # grp = lane/16) supplies hc = c*16 + lane%16 at the four kv
                # rows grp*4 .. +3. Those are STRIDE elements apart in the
                # row-major tile, hence a 4-element gather rather than one
                # vector read. It is the only gathered operand in the burst and
                # it lives in the non-critical phase.
                view = k_views[k_prev]
                hc_col = fx.Int32(g * MFMA_M) + lane_mod_16
                kv_lane = lane_div_16 * fx.Int32(MFMA_LANE_K)
                elems = []
                for i in range_constexpr(MFMA_LANE_K):
                    kv_row = kv_lane + fx.Int32(i)
                    col = k_swz_col(kv_row, hc_col)
                    elems.append(
                        view[
                            kv_row,
                            col // fx.Int32(MFMA_QK_LANE_K),
                            col % fx.Int32(MFMA_QK_LANE_K),
                        ]
                    )
                return Vec.from_elements(elems, elem_dtype).ir_value()
            row = lane_mod_16
            if const_expr(g < 2 * HC_CHUNKS):
                col = fx.Int32((g - HC_CHUNKS) * MFMA_QK_K) + lane_div_16 * fx.Int32(
                    MFMA_QK_LANE_K
                )
                view = k_views[k_cur]
                return (
                    view[row, k_swz_col(row, col) // fx.Int32(MFMA_QK_LANE_K), None]
                    .load()
                    .ir_value()
                )
            col = fx.Int32((g - 2 * HC_CHUNKS) * MFMA_DA_K) + lane_div_16 * fx.Int32(
                MFMA_DA_LANE_K
            )
            view = v_views[k_cur]
            if const_expr(SWIZZLE_V):
                col = k_swz_col(row, col)
            return view[row, col // fx.Int32(MFMA_DA_LANE_K), None].load().ir_value()

        def fmul_scalar(a, b):
            """Multiply that the backend cannot fold into `v_pk_mul_f32`.

            `v_mul_legacy_f32` has no packed form, so a pair of them stays two
            scalar VALU ops. Semantics differ from `fmul` only for 0 * inf /
            0 * NaN (legacy gives 0); every operand here -- the score
            accumulator, a sigmoid in (0,1], dA -- is finite.
            """
            return fx.Float32(
                llvm.call_intrinsic(
                    fx.Float32.ir_type,
                    "llvm.amdgcn.fmul.legacy",
                    [fx.Float32(a).ir_value(), fx.Float32(b).ir_value()],
                    [],
                    [],
                )
            )

        def fmul_const_scalar(a, b):
            """`a * b` as a scalar `v_mul_f32`, for a CONSTANT `b`.

            `fmul_scalar`'s legacy trick leaks when the other operand is a
            finite non-zero CONSTANT: `fmul_legacy` then equals `fmul`,
            InstCombine drops the intrinsic, and SLP re-packs the pair. This is
            the gate's last `v_pk_mul_f32` and the reason `-log2e` was folded
            into Q instead.

            NO `s_nop`, even though `a` is a GEMM1 accumulator needing 7 wait
            states. Checked in the ISA rather than assumed: the gate's first
            accumulator read sits after EIGHT further MFMAs plus `s_setprio`,
            two `s_waitcnt` and `s_barrier`, and `phase_barrier`'s
            `sched_barrier(0)` pins them there. The `16 x s_nop 6` that made
            this unpacking look like a net loss were never required.

            `b` takes the `s` constraint so the constant is read as VOP2 src0
            and no VGPR holds it.
            """
            from flydsl._mlir import ir
            from flydsl._mlir.dialects import llvm as _llvm

            return fx.Float32(
                _llvm.inline_asm(
                    ir.F32Type.get(),
                    [fx.Float32(a).ir_value(), fx.Float32(b).ir_value()],
                    "v_mul_f32 $0, $2, $1",
                    "=v,v,s",
                    has_side_effects=False,
                )
            )

        def addone_scalar(x):
            """`1 + x` as a scalar `v_add_f32`, via inline asm.

            There is no non-packable f32 add: `v_add_f32` pairs fold into
            `v_pk_add_f32`. A packed op is a JOIN -- it cannot issue until BOTH
            halves' producers retire -- and this one's producer is `v_exp_f32`.
            Measured on the dV/dK twin at 75.6 cy of stall per packed op
            against 2.1 cy for a packed multiply on the same datapath in the
            same phase; removing it was that kernel's single biggest win
            (-8.7%).
            """
            from flydsl._mlir import ir
            from flydsl._mlir.dialects import llvm as _llvm

            return fx.Float32(
                _llvm.inline_asm(
                    ir.F32Type.get(),
                    [fx.Float32(x).ir_value()],
                    # The `s_nop 0` is LOAD-BEARING. The operand is always a
                    # `v_exp_f32` result, and a VALU trans op needs one
                    # software wait state before a consumer reads it. The
                    # compiler inserts that automatically -- but its hazard
                    # recognizer treats INLINEASM as opaque and cannot see that
                    # this block reads the exp's destination. Without the nop
                    # the kernel reads a stale register, and on the dV/dK twin
                    # that showed up as max|err|/peak 0.33 on ~0.3% of elements
                    # under the CAUSAL mask while the hstu mask still passed --
                    # so an hstu-only accuracy run does not validate this.
                    #
                    # One wait state is exactly what is required, confirmed by
                    # asking the compiler: llc -mcpu=gfx942 on
                    # `fadd (llvm.amdgcn.exp2.f32 x), 1.0` emits
                    # `v_exp_f32 / s_nop 0 / v_add_f32`.
                    "s_nop 0\n\tv_add_f32 $0, 1.0, $1",
                    "=v,v",
                    has_side_effects=False,
                )
            )

        def fma_scalar(a, b, c):
            """`a*b + c` as a scalar `v_fma_f32`, via inline asm.

            Written with plain operators this contracts to `v_fma_f32` and is
            then re-vectorised into `v_pk_fma_f32` -- a join on three chain
            values, one of which is a `v_rcp_f32` result.

            No `s_nop` here, unlike `addone_scalar`. `c` is the rcp result and
            a trans op does need one software wait state -- but the other two
            operands are computed FROM that same result (`silu = sc*sig`,
            `b = emu*sig`), so this cannot issue until at least two
            instructions after the rcp. The data dependency supplies the wait
            state the hazard recognizer cannot see.
            """
            from flydsl._mlir import ir
            from flydsl._mlir.dialects import llvm as _llvm

            return fx.Float32(
                _llvm.inline_asm(
                    ir.F32Type.get(),
                    [fx.Float32(a).ir_value(), fx.Float32(b).ir_value(),
                     fx.Float32(c).ir_value()],
                    "v_fma_f32 $0, $1, $2, $3",
                    "=v,v,v,v",
                    has_side_effects=False,
                )
            )

        def rcp_scalar(x):
            """Scalar reciprocal, kept off the packed datapath.

            Also avoids `1.0 / x`, which even under fastmath can emit a Newton
            refinement -- two extra packed ops at the head of the chain.
            """
            return fx.Float32(
                llvm.call_intrinsic(
                    fx.Float32.ir_type,
                    "llvm.amdgcn.rcp.f32",
                    [fx.Float32(x).ir_value()],
                    [],
                    [],
                )
            )

        def pack_bf16_perm(vals):
            """Pack 4 fp32 lane values into a bf16 MFMA operand in 2 `v_perm_b32`.

            `pack_mfma_frag` spends shift+and+or per pair (3 VALU x 2 pairs = 6).
            `v_perm_b32` selects 4 arbitrary bytes out of the {src0, src1} pair
            in one instruction, and bf16 truncation IS a byte selection: the
            high halves of the two f32s, i.e. bytes 7,6 and 3,2. Bit-for-bit the
            same result as the shift/mask path -- gfx942 has no
            `v_cvt_pk_bf16_f32` (CDNA4 only), so this is the cheapest form.
            """
            sel = fx.Int32(0x07060302)
            words = []
            for i in range_constexpr(MFMA_ELEMS_PER_LANE // 2):
                lo = fx.Float32(vals[2 * i]).bitcast(fx.Int32)
                hi = fx.Float32(vals[2 * i + 1]).bitcast(fx.Int32)
                words.append(
                    fx.Int32(
                        llvm.call_intrinsic(
                            fx.Int32.ir_type,
                            "llvm.amdgcn.perm",
                            [hi.ir_value(), lo.ir_value(), sel.ir_value()],
                            [],
                            [],
                        )
                    )
                )
            return Vec.from_elements(words, fx.Int32).bitcast(elem_dtype).ir_value()

        def keep_masks(kv_start):
            """Causal / window / contextual mask, as keep[qg][i] predicates.

            The C fragment of GEMM1 is C[m = kv, n = q]: element i of lane
            (grp = lane/16, m = lane%16) is kv = kv_start + grp*4 + i and
            q = qg*16 + m. So the four kv ids are shared across every query
            sub-tile and are hoisted out of the qg loop.
            """
            kv_base = kv_start + lane_div_16 * fx.Int32(MFMA_LANE_K)
            kv_raw = [
                kv_base + fx.Int32(i) for i in range_constexpr(MFMA_ELEMS_PER_LANE)
            ]
            kv_in_seq = [
                kv_raw[i] < seq_len for i in range_constexpr(MFMA_ELEMS_PER_LANE)
            ]
            kv_ids = [to_id(kv_raw[i]) for i in range_constexpr(MFMA_ELEMS_PER_LANE)]
            out = []
            for qg in range_constexpr(Q_OWNED_SUBTILES):
                per_qg = []
                for i in range_constexpr(MFMA_ELEMS_PER_LANE):
                    # `kv_id < q_id` rather than `q_id - kv_id > 0`: the
                    # subtract is a VALU op whose only consumer is the
                    # comparison, and the comparison takes both operands
                    # directly. Materialise `dist` only where the window needs
                    # its value.
                    keep = (q_rows[qg] == kv_raw[i]) | (kv_ids[i] < q_row_ids[qg])
                    if has_window:
                        dist = q_row_ids[qg] - kv_ids[i]
                        keep = keep & (dist <= fx.Int32(max_attn_len))
                    if has_contextual:
                        ctx = (q_row_ids[qg] == fx.Int32(0)) & (kv_ids[i] < max_id)
                        keep = keep | ctx
                    per_qg.append(keep & kv_in_seq[i] & q_in_bounds[qg])
                out.append(per_qg)
            return out

        # ---- phase M -------------------------------------------------------
        def mfma_phase(dq_acc, ds_prev, kv_tile, k_cur, k_prev, k_fill):
            """Three GEMMs, the operand feed, and this group's DMA."""
            rocdl.s_setprio(PRIO_M)
            kv_start = fx.Int32(kv_tile) * fx.Int32(BLOCK_N)
            s_accs = [c_zero_v4f32 for _ in range_constexpr(Q_OWNED_SUBTILES)]
            da_accs = [c_zero_v4f32 for _ in range_constexpr(Q_OWNED_SUBTILES)]

            # Rolling operand window, a ring of PREFETCH_W + 1 slots indexed by
            # STEP, not by group value. `win[g % (W+1)]` is only correct when
            # the ring is as wide as GROUP_ORDER: the order is a permutation, so
            # a narrower ring can land a later group's read in a slot whose
            # occupant has not been consumed yet, which is silently wrong.
            # Keyed by step, the reuse distance is exactly W+1 against a W
            # lookahead, so every W is correct.
            win = [None for _ in range_constexpr(PREFETCH_W + 1)]
            for _s in range_constexpr(PREFETCH_W):
                win[_s % (PREFETCH_W + 1)] = read_operand(
                    GROUP_ORDER[_s], k_cur, k_prev
                )

            for step in range_constexpr(NUM_OPERAND_GROUPS):
                g = GROUP_ORDER[step]
                rocdl.sched_barrier(0)
                op = win[step % (PREFETCH_W + 1)]
                for qg in range_constexpr(Q_OWNED_SUBTILES):
                    if const_expr(g < HC_CHUNKS):
                        off = g * Q_OWNED_SUBTILES + qg
                        dq_acc[off] = mfma_acc(ds_prev[qg], op, dq_acc[off])
                    elif const_expr(g < 2 * HC_CHUNKS):
                        s_accs[qg] = qk_mfma_acc(
                            op, q_packs[g - HC_CHUNKS][qg], s_accs[qg]
                        )
                    else:
                        da_accs[qg] = da_mfma_acc(
                            op, do_packs[g - 2 * HC_CHUNKS][qg], da_accs[qg]
                        )

                sn = step + PREFETCH_W
                if const_expr(sn < NUM_OPERAND_GROUPS):
                    win[sn % (PREFETCH_W + 1)] = read_operand(
                        GROUP_ORDER[sn], k_cur, k_prev
                    )

                # The DMA fire is spread through the burst's slack; it is VMEM,
                # so it does not interact with the LDS retirement order. The
                # duty branch is wave-uniform (`is_g0` comes from readfirstlane)
                # and side-effect-only, so it lowers to a scalar branch.
                for d in range_constexpr(NUM_DMA):
                    if const_expr(step == DMA_FIRE_STEPS[d]):
                        if is_g0:
                            dma_k(k_fill, kv_start + fx.Int32(BLOCK_N), d)
                        else:
                            dma_v(k_fill, kv_start + fx.Int32(BLOCK_N), d)
                rocdl.sched_barrier(0)

            keep = None
            if const_expr(MASK_IN_M):
                # After the burst, never inside it.
                rocdl.sched_barrier(0)
                keep = keep_masks(kv_start)
                rocdl.sched_barrier(0)

            rocdl.s_setprio(PRIO_V)
            # The tile i+1 DMA fired above must land before the mid barrier, not
            # merely before this group's own next read: with the half-iteration
            # skew the PARTNER's M(i+1) runs in the interval this barrier opens,
            # and `vmcnt` is per-wave. `lgkmcnt(0)` retires this phase's reads
            # before the next trip's DMA can overwrite the slot.
            rocdl.s_waitcnt(vmcnt=0)
            rocdl.s_waitcnt(lgkmcnt=0)
            return dq_acc, s_accs, da_accs, kv_start, keep

        # ---- phase V -------------------------------------------------------
        def gate_phase(s_accs, da_accs, kv_start, keep):
            """SiLU' gate: dS = keep ? silu'(Z) * dA : 0, packed for GEMM3.

            GEMM1 and GEMM2 both finished in phase M, so `grad` never has to
            outlive the element it belongs to -- unlike the non-ping-pong
            kernel, which carries `g_meta` across a barrier and 32 MFMAs.
            """
            # Batched a stage at a time across ALL Q_OWNED_SUBTILES *
            # MFMA_ELEMS_PER_LANE elements, not one element at a time. The gate
            # is a ~10-deep serial dependency chain per element with two
            # quarter-rate transcendentals in it, so the only way to fill the
            # VALU pipe is to give the scheduler every independent chain at
            # once. Batching in halves measured +12% on the dV/dK twin.
            n = Q_OWNED_SUBTILES * MFMA_ELEMS_PER_LANE
            rng = range_constexpr(n)
            # The mask needs no GEMM result, so it can sit in either phase --
            # but it must NOT go inside the fenced MFMA burst, where the
            # sched_barriers pin it between MFMA issues and stretch the stream
            # (measured +9% on the twin).
            if const_expr(not MASK_IN_M):
                keep = keep_masks(kv_start)
            flat_keep = [
                keep[j // MFMA_ELEMS_PER_LANE][j % MFMA_ELEMS_PER_LANE] for j in rng
            ]
            with arith.fastmath(arith.FastMathFlags.fast):
                s = [
                    Vec(s_accs[j // MFMA_ELEMS_PER_LANE])[j % MFMA_ELEMS_PER_LANE]
                    for j in rng
                ]
                da = [
                    Vec(da_accs[j // MFMA_ELEMS_PER_LANE])[j % MFMA_ELEMS_PER_LANE]
                    for j in rng
                ]
                # `sc` arrives already scaled from GEMM1 (alpha is folded into
                # q_packs), so only the exp2 argument needs a multiply.
                sc = s
                emu = [exp2_f32(fmul_const_scalar(sc[j], c_neg_log2e)) for j in rng]
                sig = [rcp_scalar(addone_scalar(emu[j])) for j in rng]
                silu = [fmul_scalar(sc[j], sig[j]) for j in rng]
                # silu'(z) = sig*(1 + z*(1 - sig)) = sig + silu*(1 - sig).
                # The second form is a multiply plus an FMA, against the first
                # form's sub/mul/add/mul, and it reuses `silu`.
                # 1 - sig == emu * sig exactly, since sig = 1/(1+emu). Using the
                # product keeps this off the packed add datapath too.
                grad = [
                    fma_scalar(silu[j], fmul_scalar(emu[j], sig[j]), sig[j])
                    for j in rng
                ]
                # No `1/N` here: dS is linear in it and so is its only consumer
                # (GEMM3), so it rides along to the dQ epilogue and is folded
                # into that store's existing scale. Saves one multiply per
                # element.
                gated = [fmul_scalar(grad[j], da[j]) for j in rng]
                # The select is load-bearing, not a tidy-up: a masked element
                # legitimately produces NaN (emu = inf, sig = 0), and masking
                # earlier would propagate it into dQ.
                ds_vals = [flat_keep[j].select(gated[j], c_zero_f) for j in rng]
            e = MFMA_ELEMS_PER_LANE
            return [
                pack_bf16_perm(ds_vals[qg * e : qg * e + e])
                for qg in range_constexpr(Q_OWNED_SUBTILES)
            ]

        # ---- the pipeline ---------------------------------------------------
        if active:
            # Trip 0's GEMM3 reads the K slot no DMA has filled. Its A operand
            # (`ds_prev`) is zero, but `0 * NaN` is NaN and would poison
            # `dq_acc` for the whole sweep, so zero that slot. With UNROLL = 3
            # the first sub-trip reads slot 2 and fills slot 1, so slot 2 is the
            # one. `lgkmcnt(0)` before the barrier publishes it to the partner
            # group, which reads the same slot in its own trip-0 M phase.
            if is_g0:
                k_zero(2)
            rocdl.s_waitcnt(lgkmcnt=0)

            # Prime the pipeline with tile T0 into slot 0; the loop fires one
            # tile ahead from there.
            _t0 = kv_tile_start * fx.Int32(BLOCK_N)
            for d in range_constexpr(NUM_DMA):
                if is_g0:
                    dma_k(0, _t0, d)
                else:
                    dma_v(0, _t0, d)
            rocdl.s_waitcnt(vmcnt=0)
            rocdl.s_waitcnt(lgkmcnt=0)
            phase_barrier()

            # The skew: g0 burns one extra barrier so that from here on g0 is in
            # phase M exactly when g1 is in phase V, and vice versa. Barrier
            # counts stay equal overall -- g1 repays it after the loop.
            if is_g0:
                phase_barrier()

            acc_init = [c_zero_v4f32 for _ in range_constexpr(N_ACC)] + [
                c_zero_pack for _ in range_constexpr(N_PACK)
            ]
            loop_results = acc_init
            # The body is unrolled UNROLL times so every slot index below is a
            # literal, which is what keeps each LDS access inside one global.
            # The trip count is rounded UP to a multiple of UNROLL rather than
            # peeled: a tile past the sweep has every `keep` false -- it is
            # either past `seq_len` or entirely above the causal diagonal -- so
            # its gate emits zero packs and the next trip's GEMM3 adds exactly
            # nothing. That is the same property the drain trip relies on.
            n_trips = n_tiles + fx.Int32(1) - kv_tile_start
            n_outer = (n_trips + fx.Int32(UNROLL - 1)) // fx.Int32(UNROLL)
            for _outer, it in range(
                fx.Int64(0), fx.Int64(n_outer), fx.Int64(1), init=acc_init
            ):  # ty: ignore
                it_list = list(it) if isinstance(it, (list, tuple)) else [it]
                dq_acc = [it_list[i] for i in range_constexpr(N_ACC)]
                ds_prev = [it_list[N_ACC + i] for i in range_constexpr(N_PACK)]
                base_tile = kv_tile_start + fx.Int32(_outer) * fx.Int32(UNROLL)

                # Loop rotated: the closing barrier of each gate phase is issued
                # at the TOP of the next sub-trip rather than the bottom of this
                # one, so the scf.for back edge -- increment, 64-bit trip
                # compare, branch -- lands in the tail of the gate phase instead
                # of at the head of the MFMA phase, where it would sit on a WAR
                # hazard against the operand reads.
                for u in range_constexpr(UNROLL):
                    if const_expr(u == 0):
                        phase_barrier()
                    kv_tile = base_tile + fx.Int32(u)
                    # Literal slots: tile i lives in i % K_SLOTS, GEMM3 reads
                    # tile i-1's, and the DMA fills tile i+1's.
                    k_cur = u % K_SLOTS
                    k_prev = (u + K_SLOTS - 1) % K_SLOTS
                    k_fill = (u + 1) % K_SLOTS
                    dq_acc, s_accs, da_accs, kv_start, keep = mfma_phase(
                        dq_acc, ds_prev, kv_tile, k_cur, k_prev, k_fill
                    )
                    phase_barrier()
                    ds_prev = gate_phase(s_accs, da_accs, kv_start, keep)
                    if const_expr(u < UNROLL - 1):
                        phase_barrier()

                loop_results = yield (dq_acc + ds_prev)

            # Closes the last iteration's gate phase, which the rotation left
            # without a trailing barrier inside the loop.
            phase_barrier()

            if not is_g0:
                phase_barrier()

            # ---- Epilogue: store dQ (alpha/N applied here) ----
            # MFMA lanes map directly to query rows and head-dimension columns.
            results = (
                list(loop_results)
                if isinstance(loop_results, (list, tuple))
                else [loop_results]
            )
            with arith.fastmath(arith.FastMathFlags.fast):
                for qg in range_constexpr(Q_OWNED_SUBTILES):
                    q_row_base = (
                        q_wave_base
                        + fx.Int32(qg * MFMA_M)
                        + lane_div_16 * fx.Int32(MFMA_LANE_K)
                    )
                    for e in range_constexpr(MFMA_ELEMS_PER_LANE):
                        q_row_e = q_row_base + fx.Int32(e)
                        if q_row_e < seq_len:
                            for c in range_constexpr(HC_CHUNKS):
                                ov = results[c * Q_OWNED_SUBTILES + qg]
                                hc_col = fx.Int32(c * MFMA_M) + lane_mod_16
                                val = (Vec(ov)[e] * c_dq_scale).to(elem_dtype)
                                dq[fx.Int64(seq_start + q_row_e), head_idx, hc_col] = (
                                    val
                                )

    @flyc.jit
    def launch_hstu_attention_bwd_dq_pp(
        batch: fx.Int32,
        q: fx.Tensor,
        k: fx.Tensor,
        v: fx.Tensor,
        do: fx.Tensor,
        seq_offsets: fx.Tensor,
        num_targets: fx.Tensor,
        perm: fx.Tensor,
        dq: fx.Tensor,
        stream: fx.Stream,
    ) -> None:
        c_ngg = fx.Int32(NUM_GRID_GROUPS)
        hz_total = batch * fx.Int32(num_heads)
        hz_per_group = (hz_total + fx.Int32(NUM_GRID_GROUPS - 1)) // c_ngg
        grid = fx.Int32(num_q_tiles) * hz_per_group * c_ngg
        hstu_attention_bwd_dq_pp(
            q,
            k,
            v,
            do,
            seq_offsets,
            num_targets,
            perm,
            dq,
            hz_per_group,
            hz_total,
            value_attrs={
                "passthrough": [
                    ["denormal-fp-math-f32", "preserve-sign,preserve-sign"],
                    ["no-nans-fp-math", "true"],
                    ["unsafe-fp-math", "true"],
                ],
                "rocdl.waves_per_eu": WAVES_PER_EU,
                "rocdl.flat_work_group_size": f"{BLOCK_THREADS},{BLOCK_THREADS}",
            },
        ).launch(
            grid=grid,
            block=BLOCK_THREADS,
            smem=0,
            stream=stream,
        )

    return launch_hstu_attention_bwd_dq_pp
