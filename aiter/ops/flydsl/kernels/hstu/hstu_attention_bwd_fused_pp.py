# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Fused dV/dK/dQ HSTU backward kernel (gfx942, ping-pong, split-K dQ).

Built from `hstu_attention_bwd_dvdk_pp.py`. On top of the dV/dK ping-pong
schedule it also produces dQ = dS.K as a PARTIAL over this workgroup's KV
range, written to an HBM scratch that `hstu_attention_bwd_dq_reduce.py` sums.
Split-K and not atomics because fp atomics are a 0.40 TB/s wall while plain
stores run at 3.56 TB/s -- 8.8x, measured.

Geometry is (256, 16, 8), NOT the twin's 384. 384 spills 21 registers --
KV_OWNED_SUBTILES=3 costs 74 VGPR over KV=2 and the dQ state adds ~46 more --
and measures 503.7 ms against 461.0 at 256, even though it removes the
geometry penalty entirely (nodq 220.4 vs dvdk_pp's 219.2) and cuts the
reduction pass by a third (36.9 vs 52.0 ms). The spills cost +58 ms on the
kernel, more than both gains together. BLOCK_M quantises to 128 (8 waves x
MFMA_M 16), so 256 and 384 are the only candidates; there is nothing between.

(256, 16, 8) is only trustworthy as of e2a81264d, which fixed a prologue
publish-ordering bug that made every geometry except 384 nondeterministic.

Everything below this line is inherited from the twin.

Ping-pong-pipelined variant of the KV-owned HSTU backward dV/dK kernel.

Same mathematics as `hstu_attention_bwd.py` -- see that module's docstring --
but scheduled so the two waves resident on each SIMD are held in *opposite*
phases:

    phase M : the 48 MFMAs, their LDS operand feed, the Q/dO DMA, the
              transpose build, and the mask.
    phase V : the fused SiLU / SiLU' gate.  No memory operations at all.

One wave group is in M while the other is in V, so the matrix core and the VALU
pipe are both busy continuously.  Without this, the two resident waves sit in
*lockstep* on the shared `gpu.barrier()`: they idle together and they burst MFMA
together, contending for the one matrix core (measured: 27.9% matrix-core busy,
50.7% of all stall in `s_waitcnt`).

The schedule is ported from the gfx950 fp8 FMHA ping-pong kernel.  Only the
schedule -- no CDNA4 instruction is used here.

Three structural points, each load-bearing:

* **The chained GEMMs consume the previous tile's fragments.**  GEMM1/GEMM3 at
  trip `i` use `p_packs[i-1]`/`ds_packs[i-1]`, which is what merges GEMM0/2 of
  tile `i` and GEMM1/3 of tile `i-1` into one contiguous MFMA burst.  Seeding
  those packs to zero makes trip 0 self-starting and the extra trip the drain,
  so no iteration is peeled.
* **The transpose build lives in phase M, not phase V.**  With a half-iteration
  skew, a tile written in one group's V phase is published only at the next
  barrier, but the partner's consuming M phase overlaps that very V phase -- an
  unsynchronised cross-group RAW race.  Building in M puts one barrier between
  the write and the earliest partner read.
* **Row-major Q/dO are 3-deep and fired two tiles ahead.**  At depth 2 the load
  gets only the tail of one MFMA phase to return, which reinstates the very
  `vmcnt` stall this kernel exists to remove.

`validate_hstu_attention_bwd_fused_pp` hard-gates the supported shape: this
kernel is specialized for gfx942 / bf16 / head_dim == hidden_dim == 64 and the
`(384, 16, 8)` tile, and raises for anything else.
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
# Depth of the rolling LDS operand window, in operand groups.  The MFMA at group
# `g` consumes the pack read at `g - PREFETCH_W`, so each read has
# PREFETCH_W * KV_OWNED_SUBTILES MFMAs of cover.  The fp8 reference uses depth 2
# against a 32-cycle `32x32x64`; `16x16x16_bf16` is only ~16 cycles, so the
# depth has to be larger here to hide the same ~130-cycle ds_read.
PREFETCH_W = 15
# Operand groups per MFMA phase: 4 GEMMs x (head_dim / MFMA_M) chunks.
NUM_OPERAND_GROUPS = 16

# Issue order of those groups, as a permutation.
#
# In natural order the four chunks of one GEMM run back to back, and GEMM0 /
# GEMM2 accumulate over `ks` into the SAME register per `og` -- a 4-deep
# dependency chain with only KV_OWNED_SUBTILES (3) independent MFMAs between
# consecutive links. At ~4 cy of issue each that is 12 cy against a ~16 cy MFMA
# result latency, so every link stalls.
#
# Interleaving the two independent GEMMs of each pair (0 with 2, 1 with 3)
# doubles the spacing to 6 MFMAs, which covers the latency. Pure reordering:
# every accumulator still sees its `ks` steps in the same sequence, so the
# arithmetic is unchanged.
GROUP_ORDER = [0, 4, 1, 5, 2, 6, 3, 7, 8, 12, 9, 13, 10, 14, 11, 15]
assert sorted(GROUP_ORDER) == list(range(NUM_OPERAND_GROUPS))


def validate_hstu_attention_bwd_fused_pp(
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
    """Hard gate for the ping-pong kernel.

    Deliberately strict: this kernel hardcodes a single tuned shape, so anything
    outside it must fall back to `build_hstu_attention_bwd_dvdk` rather than
    silently compile a variant that was never measured.
    """
    if arch is None:
        arch = get_rocm_arch()
    if not (arch or "").startswith("gfx942"):
        raise ValueError(
            f"hstu bwd dvdk ping-pong requires gfx942, got {arch!r}. The schedule "
            "assumes 16x16x16 MFMA timing and a dword LDS DMA."
        )
    if dtype_str != "bf16":
        raise ValueError(f"hstu bwd dvdk ping-pong requires bf16, got {dtype_str!r}")
    if not causal:
        raise ValueError("hstu_attention_bwd only supports causal attention")
    if head_dim != 64 or hidden_dim != 64:
        raise ValueError(
            "hstu bwd dvdk ping-pong requires head_dim == hidden_dim == 64, got "
            f"{head_dim}/{hidden_dim}. Both groups share one duty code path, which "
            "relies on the Q and dO tiles being structurally identical."
        )
    if (block_m, block_n, num_waves) != (256, 16, 8):
        raise ValueError(
            "hstu bwd fused requires (block_m, block_n, num_waves) = "
            f"(256, 16, 8), got ({block_m}, {block_n}, {num_waves}). 8 waves give "
            "2 per SIMD inside one barrier domain. BLOCK_M quantises to 128 "
            "(8 waves x MFMA_M 16), so 256 and 384 are the only candidates. "
            "384 is the dV/dK twin's shape and removes this kernel's 15 ms "
            "geometry penalty, but it spills 21 registers and measures "
            "503.7 ms against 461.0 at 256 (214 VGPR, 0 spills)."
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
def build_hstu_attention_bwd_fused_pp(
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
    block_m: int = 256,
    block_n: int = 16,
    num_waves: int = 8,
    waves_per_eu: int = 2,
    has_perm: bool = False,
    dq_mode="full",
):
    validate_hstu_attention_bwd_fused_pp(
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

    BLOCK_M = block_m
    BLOCK_N = block_n
    NUM_WAVES = num_waves
    BLOCK_THREADS = NUM_WAVES * WARP_SIZE
    ROWS_PER_WAVE = BLOCK_M // NUM_WAVES
    KV_OWNED_SUBTILES = ROWS_PER_WAVE // MFMA_M
    WAVES_PER_EU = waves_per_eu
    MFMA_QK_K, MFMA_QK_LANE_K, _ = _mfma_params_for_dim(head_dim)
    MFMA_DA_K, MFMA_DA_LANE_K, _ = _mfma_params_for_dim(hidden_dim)
    DMA_BYTES, DMA_ELEMS, K_SWZ_ROWS, K_SWZ_SHIFT = _arch_dma_params()

    elem_dtype = _dtype_to_elem_type(dtype_str)
    ELEM_BYTES = DMA_BYTES // DMA_ELEMS
    is_bf16 = True
    has_window = max_attn_len > 0
    has_contextual = contextual_seq_len > 0

    # head_dim == hidden_dim == 64 (gated above), so every per-dimension count
    # collapses to one value and the Q and dO tiles are structurally identical.
    K_STEPS = head_dim // MFMA_QK_K
    D_CHUNKS = hidden_dim // MFMA_M
    DK_STEPS = hidden_dim // MFMA_DA_K
    HC_CHUNKS = head_dim // MFMA_M
    STRIDE = head_dim
    assert K_STEPS == D_CHUNKS == DK_STEPS == HC_CHUNKS == 4
    assert MFMA_QK_LANE_K == MFMA_DA_LANE_K == MFMA_LANE_K
    assert BLOCK_N == MFMA_N, "Q_STREAM_SUBTILES must be 1"
    assert NUM_OPERAND_GROUPS == 4 * D_CHUNKS

    num_kv_tiles = (max_seq_len + BLOCK_M - 1) // BLOCK_M
    stride_qk_n = num_heads * head_dim
    stride_do_n = num_heads * hidden_dim

    # ---- ping-pong groups -------------------------------------------------
    # g0 = waves 0..3 stages Q and builds Q^T; g1 = waves 4..7 stages dO and
    # builds dO^T.  Wave i and wave i+4 share a SIMD, which is the pair the
    # phase alternation (and `s_setprio`) arbitrates.
    GRP_WAVES = NUM_WAVES // 2
    GRP_THREADS = GRP_WAVES * WARP_SIZE
    assert GRP_THREADS == 256

    # ---- LDS geometry -----------------------------------------------------
    # Both the row-major tiles and the transposes are 2-deep, and every slot is
    # its own `fx.Array` field -- see SharedStorage.
    RM_SLOTS = 2
    T_SLOTS = 2
    # The loop body is unrolled by the slot count so each sub-trip names a
    # *literal* slot. That is what keeps the slot index compile-time, which is
    # what keeps each access provably inside one LDS global.
    UNROLL = RM_SLOTS
    assert UNROLL % RM_SLOTS == 0 and UNROLL % T_SLOTS == 0
    RM_TILE_ELEMS = BLOCK_N * STRIDE
    # Fragment-transpose scratch: one 16x16 bf16 region per (wave, subtile).
    # PER-WAVE PRIVATE -- a wave transposes only its own kv rows -- so it
    # needs no barrier, only `s_waitcnt lgkmcnt`, and it does not rotate, so
    # it does not force UNROLL to lcm(...).
    # dQ reduction buffer: one BLOCK_N x head_dim fp32 tile per UNROLL slot.
    # The row stride is padded by 4 dwords. Unpadded, the four quarter-waves of
    # a `ds_add_f32` sit 4*head_dim = 256 dwords apart, which is 0 mod 64 banks
    # -- a 4-way conflict on every one of the 16 adds per lane. +4 shifts each
    # quarter by 4 banks so all 64 lanes land on distinct ones.
    DQ_PAD = head_dim + 4
    # One PRIVATE slot per wave, written with ds_write. `ds_add_f32` into a
    # shared slot is 55x a ds_write_b32 on gfx942 -- 737 vs 13 ns per op per
    # wave, measured in tmp/hstu_repro/bench_lds_fadd.py, and unchanged by
    # giving the waves private addresses or by the bank padding, so it is the
    # instruction and not contention. It cost 1520 ms of a 1915 ms kernel.
    DQW_ELEMS = NUM_WAVES * BLOCK_N * DQ_PAD
    # Group-exchange buffer. Two rows past the tile: row BLOCK_N is g0's write
    # sink and row BLOCK_N+1 is a permanently-zero source for g1's read, which
    # is how the exchange stays branch-free.
    DQG_SINK, DQG_ZERO = BLOCK_N, BLOCK_N + 1
    DQG_ELEMS = (BLOCK_N + 2) * DQ_PAD
    # 4 contiguous outputs per lane: 16 rows x 16 lanes x 4 = BLOCK_N*head_dim.
    DQ_PER_LANE = BLOCK_N * head_dim // (NUM_WAVES // 2 * WARP_SIZE)
    DQ_LANES_PER_ROW = head_dim // DQ_PER_LANE

    rm_tile_elems_per_pass = GRP_THREADS * DMA_ELEMS
    assert RM_TILE_ELEMS % rm_tile_elems_per_pass == 0
    NUM_DMA = RM_TILE_ELEMS // rm_tile_elems_per_pass
    PAIRS_PER_ROW = STRIDE // DMA_ELEMS

    # One transpose unit per (transposed row, 4-wide query group).
    assert BLOCK_N % MFMA_LANE_K == 0
    T_QGROUPS = BLOCK_N // MFMA_LANE_K
    T_UNITS = head_dim * T_QGROUPS
    assert T_UNITS == GRP_THREADS, "transpose must be exactly one pass per group"

    # Transposed tile, laid out so one lane's FOUR chunk operands are one
    # contiguous 32-byte run.
    #
    # The MFMA B operand for chunk c gives lane (m = lane%16, g = lane/16) the
    # 4 bf16 at [d = c*16 + m][q = g*4 .. +3] -- 8 bytes.  Indexed the obvious
    # way ([d][q], stride BLOCK_N+pad) the four chunks sit 16 rows apart, so
    # feeding the burst costs 4 x `ds_read_b64` per tile, 8 across q_t + do_t.
    #
    # Order it [g][m][c][q] instead and those four 8-byte pieces become
    # adjacent: 32 contiguous bytes per lane, 2 x `ds_read_b128`, 4 reads total.
    # Nothing is permuted -- the run already decomposes into the four operands
    # in chunk order, so the accumulator <-> dV/dK column mapping is unchanged
    # and the epilogue does not move.  (`d` is the GEMM's output N dimension
    # here, not its reduction axis; what makes this free is that each chunk's
    # 8 bytes stays self-contained, not that the order would be summed away.)
    T_B128_ELEMS = 16 // ELEM_BYTES  # one ds_read_b128
    T_ROW_ELEMS = D_CHUNKS * MFMA_LANE_K  # all four chunks for one lane
    assert T_ROW_ELEMS % T_B128_ELEMS == 0
    T_HALVES = T_ROW_ELEMS // T_B128_ELEMS
    CHUNKS_PER_B128 = T_B128_ELEMS // MFMA_LANE_K
    PACK_DWORDS = MFMA_LANE_K * ELEM_BYTES // 4
    # Pad each q-group by one b128 so that (a) every (g, m) start stays 16-byte
    # aligned -- without which the backend cannot form a b128 at all -- and
    # (b) the 64 lanes spread over all 32 banks.  Unpadded, lanes m and m+4
    # share a bank quad and the read doubles to 16 dwords/bank.
    T_GPAD = T_B128_ELEMS
    T_GSTRIDE = MFMA_M * T_ROW_ELEMS + T_GPAD
    assert (T_GSTRIDE * ELEM_BYTES) % 16 == 0
    assert D_CHUNKS * MFMA_M == head_dim
    T_TILE_ELEMS = T_QGROUPS * T_GSTRIDE

    N_ACC_DV = D_CHUNKS * KV_OWNED_SUBTILES
    N_ACC_DK = HC_CHUNKS * KV_OWNED_SUBTILES
    N_ACC = N_ACC_DV + N_ACC_DK
    N_PACK = KV_OWNED_SUBTILES

    # One field per ROTATION SLOT, not one array per tensor with the slot as an
    # offset. Each `fx.Array` field becomes its own LDS global, and only then
    # can the backend prove that the `buffer_load_lds` filling slot 1 is
    # disjoint from the `ds_read`s feeding the MFMAs out of slot 0 -- which is
    # the entire point of double buffering: the write and the reads have to
    # proceed at the same time.
    #
    # With one fused array and a runtime slot offset the accesses are only
    # provably inside "the q array", so every read may-alias the in-flight DMA
    # and the backend serializes them. Measured: that cost ~8% here, on top of
    # the separate `inttoptr` provenance bug (see the LDS views below).
    @fx.struct
    class SharedStorage:
        q0: fx.Array[elem_dtype, RM_TILE_ELEMS, 16]
        q1: fx.Array[elem_dtype, RM_TILE_ELEMS, 16]
        do0: fx.Array[elem_dtype, RM_TILE_ELEMS, 16]
        do1: fx.Array[elem_dtype, RM_TILE_ELEMS, 16]
        q_t0: fx.Array[elem_dtype, T_TILE_ELEMS, 16]
        q_t1: fx.Array[elem_dtype, T_TILE_ELEMS, 16]
        do_t0: fx.Array[elem_dtype, T_TILE_ELEMS, 16]
        do_t1: fx.Array[elem_dtype, T_TILE_ELEMS, 16]
        dqw: fx.Array[fx.Float32, DQW_ELEMS, 16]
        dqg: fx.Array[fx.Float32, DQG_ELEMS, 16]

    @flyc.kernel(known_block_size=[BLOCK_THREADS, 1, 1])
    def hstu_attention_bwd_fused_pp(
        q: fx.Tensor,
        k: fx.Tensor,
        v: fx.Tensor,
        do: fx.Tensor,
        seq_offsets: fx.Tensor,
        num_targets: fx.Tensor,
        perm: fx.Tensor,
        out_dv: fx.Tensor,
        out_dk: fx.Tensor,
        # ---- split-K dQ. Plumbed now, consumed once the dQ GEMM lands. -----
        # `dq_part` is the fp32 partial scratch; `part_off` holds the
        # per-(batch, kv_tile) base offsets into it for THIS pass.
        # `kv_tile_lo/hi` bound the pass's kv tiles for kv-chunking.
        dq_part: fx.Tensor,
        part_off: fx.Tensor,
        hz_per_group: fx.Int32,
        hz_total: fx.Int32,
        kv_tile_lo: fx.Int32,
        kv_tile_hi: fx.Int32,
    ) -> None:
        elem_type = elem_dtype.ir_type

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

        block_id = fx.Int32(gpu.block_idx.x)
        grid_group = block_id % fx.Int32(NUM_GRID_GROUPS)
        # kv-chunking: a pass covers only kv tiles [kv_tile_lo, kv_tile_hi),
        # so the grid is sized to that span and the tile index is biased back
        # up. Splitting on KV and not q is load-bearing -- dV/dK sum over all
        # q, so each pass gets the FULL q range for its kv rows and writes
        # them out complete; only dQ spans passes, which is exactly what the
        # reduction already handles. A q split would leave dV/dK with half
        # their contributions and make every workgroup run in both passes.
        n_pass_tiles = kv_tile_hi - kv_tile_lo
        pos_in_group = block_id // fx.Int32(NUM_GRID_GROUPS)
        local_hz_idx = pos_in_group // n_pass_tiles
        kv_tile_idx = kv_tile_lo + pos_in_group % n_pass_tiles
        hz_idx = grid_group * hz_per_group + local_hz_idx
        block_valid = hz_idx < hz_total
        hz_idx = block_valid.select(hz_idx, fx.Int32(0))
        batch_idx = hz_idx // fx.Int32(num_heads)
        head_idx = hz_idx % fx.Int32(num_heads)

        if const_expr(has_perm):
            batch_idx = fx.Int32(perm[batch_idx])

        seq_start = fx.Int32(seq_offsets[batch_idx])
        seq_len = fx.Int32(seq_offsets[batch_idx + fx.Int32(1)]) - seq_start
        seq_len = block_valid.select(seq_len, fx.Int32(0))

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

        k_load = grouped_loader(k, head_dim, MFMA_QK_LANE_K)
        v_load = grouped_loader(v, hidden_dim, MFMA_DA_LANE_K)

        q_head_offset = head_idx * fx.Int32(head_dim)
        q_base_byte_offset = (
            fx.Int64(seq_start) * fx.Int64(stride_qk_n) + fx.Int64(q_head_offset)
        ) * fx.Int64(2)
        do_base_byte_offset = (
            fx.Int64(seq_start) * fx.Int64(stride_do_n)
            + fx.Int64(head_idx) * fx.Int64(hidden_dim)
        ) * fx.Int64(2)

        # ---- LDS views ----------------------------------------------------
        # One view per (tensor, slot), each built straight off its own field's
        # pointer. The slot is a Python index resolved at trace time, so every
        # access is statically inside exactly one LDS global and the backend can
        # cross-noalias them -- that is what lets slot 1's DMA and slot 0's
        # ds_reads overlap instead of being serialized.
        #
        # Do NOT reintroduce a runtime slot offset here, and never form an LDS
        # address with `fx.inttoptr` or a `select` between two pointers: both
        # break the provenance walk, after which SIInsertWaitcnts cannot prove a
        # `ds_read` misses the in-flight `buffer_load_lds` and plants a
        # `vmcnt(0)` in front of every operand read (measured: 19.8 M stall
        # cycles, matrix-core duty down to 22%).
        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        rm_layout = fx.make_layout(
            (BLOCK_N, STRIDE // MFMA_LANE_K, MFMA_LANE_K),
            (STRIDE, MFMA_LANE_K, 1),
        )
        # Two views of the SAME array: the transpose writes one chunk's 4 bf16
        # (`ds_write_b64`), the MFMA feed reads two chunks at a time
        # (`ds_read_b128`).  Same global, so the aliasing is still exact.
        t_wr_layout = fx.make_layout(
            (T_QGROUPS, MFMA_M, D_CHUNKS, MFMA_LANE_K),
            (T_GSTRIDE, T_ROW_ELEMS, MFMA_LANE_K, 1),
        )
        t_rd_layout = fx.make_layout(
            (T_QGROUPS, MFMA_M, T_HALVES, T_B128_ELEMS),
            (T_GSTRIDE, T_ROW_ELEMS, T_B128_ELEMS, 1),
        )
        q_views = [lds.q0.view(rm_layout), lds.q1.view(rm_layout)]
        do_views = [lds.do0.view(rm_layout), lds.do1.view(rm_layout)]
        q_t_wr = [lds.q_t0.view(t_wr_layout), lds.q_t1.view(t_wr_layout)]
        do_t_wr = [lds.do_t0.view(t_wr_layout), lds.do_t1.view(t_wr_layout)]
        q_t_rd = [lds.q_t0.view(t_rd_layout), lds.q_t1.view(t_rd_layout)]
        do_t_rd = [lds.do_t0.view(t_rd_layout), lds.do_t1.view(t_rd_layout)]

        # 4-D, not 3-D: a lane's DQ_PER_LANE-wide run must be one `None` slice
        # so the flush reads it as a single `ds_read_b128`. Indexed per element
        # it comes out as DQ_PER_LANE/2 `ds_read2_b32` instead -- twice the LDS
        # instructions for the same bytes, which the trace put at 9.6 M cycles
        # (10.3% of all stall) once the scalar adds stopped it fusing wider.
        dqw = lds.dqw.view(
            fx.make_layout(
                (NUM_WAVES, BLOCK_N, head_dim // DQ_PER_LANE, DQ_PER_LANE),
                (BLOCK_N * DQ_PAD, DQ_PAD, DQ_PER_LANE, 1),
            )
        )
        dqg = lds.dqg.view(
            fx.make_layout(
                (BLOCK_N + 2, head_dim // DQ_PER_LANE, DQ_PER_LANE),
                (DQ_PAD, DQ_PER_LANE, 1),
            )
        )
        # Flat, for the prologue clear only.
        dqg_flat = lds.dqg.view(fx.make_layout((DQG_ELEMS,), (1,)))

        def q_swz_col(tile_row, col):
            return swz_col(tile_row, col, K_SWZ_ROWS, K_SWZ_SHIFT)

        # ---- K / V into registers, once ------------------------------------
        c_zero_qk_pack = Vec.filled(MFMA_QK_LANE_K, 0.0, elem_dtype).ir_value()
        c_zero_da_pack = Vec.filled(MFMA_DA_LANE_K, 0.0, elem_dtype).ir_value()

        kv_wave_base = kv_tile_idx * fx.Int32(BLOCK_M) + wave_id * fx.Int32(
            ROWS_PER_WAVE
        )
        kv_rows = []
        kv_in_bounds = []
        for og in range_constexpr(KV_OWNED_SUBTILES):
            local = kv_wave_base + fx.Int32(og * MFMA_M) + lane_mod_16
            kv_rows.append(local)
            kv_in_bounds.append(local < seq_len)

        k_packs = []
        for ks in range_constexpr(K_STEPS):
            k_col = fx.Int32(ks * MFMA_QK_K) + lane_div_16 * fx.Int32(MFMA_QK_LANE_K)
            per_og = []
            for og in range_constexpr(KV_OWNED_SUBTILES):
                safe = kv_in_bounds[og].select(seq_start + kv_rows[og], seq_start)
                raw = k_load(
                    fx.Int64(safe), head_idx, k_col // fx.Int32(MFMA_QK_LANE_K)
                ).ir_value()
                # Fold `alpha` into K once, here, so GEMM0 yields the SCALED
                # score and the gate's `sc = s * alpha` disappears -- 12
                # element-multiplies per gate phase, 6 v_pk_mul. Only GEMM0
                # reads k_packs, and dK's chain rule is unchanged: sc =
                # Q.(aK)^T still gives d(sc)/dK = a.Q, which the epilogue's
                # `c_dk_scale = alpha/N` already applies.
                #
                # The cost is that K is bf16, so bf16(alpha*K) rounds a second
                # time -- an error the fp32 reference does not have.
                scaled = (Vec(raw).to(fx.Float32) * fx.Float32(alpha)).to(elem_dtype)
                per_og.append(
                    kv_in_bounds[og].select(scaled.ir_value(), c_zero_qk_pack)
                )
            k_packs.append(per_og)

        v_packs = []
        for ks in range_constexpr(DK_STEPS):
            v_col = fx.Int32(ks * MFMA_DA_K) + lane_div_16 * fx.Int32(MFMA_DA_LANE_K)
            per_og = []
            for og in range_constexpr(KV_OWNED_SUBTILES):
                safe = kv_in_bounds[og].select(seq_start + kv_rows[og], seq_start)
                raw = v_load(
                    fx.Int64(safe), head_idx, v_col // fx.Int32(MFMA_DA_LANE_K)
                ).ir_value()
                per_og.append(kv_in_bounds[og].select(raw, c_zero_da_pack))
            v_packs.append(per_og)

        c_alpha = fx.Float32(alpha)
        c_inv_n = fx.Float32(1.0 / max_seq_len)
        c_neg_log2e = fx.Float32(-_LOG2E)
        c_alpha_neg_log2e = fx.Float32(alpha * -_LOG2E)
        c_one_f = fx.Float32(1.0)
        c_zero_f = fx.Float32(0.0)
        # dS no longer carries 1/N (see gate_phase), so the dK store applies
        # alpha/N in the one multiply it was already doing.
        c_dk_scale = fx.Float32(alpha / max_seq_len)
        c_zero_v4f32 = Vec.filled(MFMA_ELEMS_PER_LANE, 0.0, fx.Float32).ir_value()
        c_zero_pack = Vec.filled(MFMA_LANE_K, 0.0, elem_dtype).ir_value()

        kv_owned_ids = [to_id(kv_rows[og]) for og in range_constexpr(KV_OWNED_SUBTILES)]

        kv_start_row = kv_tile_idx * fx.Int32(BLOCK_M)
        kv_end_row = kv_start_row + fx.Int32(BLOCK_M)
        active = kv_start_row < seq_len

        q_upper = seq_len
        if has_window and not has_contextual:
            win_upper = kv_end_row + fx.Int32(max_attn_len)
            if has_targets:
                win_upper = (win_upper <= max_id).select(win_upper, seq_len)
            q_upper = (win_upper < seq_len).select(win_upper, seq_len)
        q_upper = active.select(q_upper, fx.Int32(0))
        n_q_tiles = (q_upper + fx.Int32(BLOCK_N - 1)) // fx.Int32(BLOCK_N)
        q_tile_start = kv_start_row // fx.Int32(BLOCK_N)
        if has_contextual:
            q_tile_start = fx.Int32(0)

        # ---- staging duty: global -> LDS ----------------------------------
        # g0 stages Q, g1 stages dO. The two duties are built as separate
        # closures and dispatched on a wave-uniform `if`, rather than selecting
        # between two pointers -- see the provenance note above.
        _dma_atom, _dma_ptr_ty, _rebased_buffer_div = make_lds_dma(DMA_BYTES, elem_type)
        c_stride_n = fx.Int32(stride_qk_n)
        c_dma_elems = fx.Int32(DMA_ELEMS)
        c_pairs_per_row = fx.Int32(PAIRS_PER_ROW)

        def make_dma(tensor, base_byte_off, dst_views, swizzled):
            div = _rebased_buffer_div(
                # num_records = THIS sequence's extent, not the tensor's, so a
                # stray voffset can never read another sequence's tokens. It is
                # a backstop only: the descriptor bound DROPS an out-of-range
                # `buffer_load ... lds` instead of zeroing it, so `fire` still
                # has to clamp the row itself.
                fx.get_iter(tensor),
                base_byte_off,
                seq_len * stride_qk_n,
            )
            # The out-of-range row has to land somewhere defined -- see `fire`.
            # Both the lane's offset within a tile and the clamp target are
            # loop-invariant, so build the whole element offset here and let the
            # loop add only the tile base.
            last_row_elem = (seq_len - fx.Int32(1)) * c_stride_n
            rowoff, bound = [], []
            for d in range_constexpr(NUM_DMA):
                pair = grp_tid + fx.Int32(d * GRP_THREADS)
                row = pair // c_pairs_per_row
                col = (pair % c_pairs_per_row) * c_dma_elems
                gcol = q_swz_col(row, col) if const_expr(swizzled) else col
                rowoff.append(row * c_stride_n + gcol)
                bound.append(last_row_elem + gcol)
            # The DMA destination must be wave-uniform in an SGPR, so it is the
            # one LDS address that still goes through `inttoptr` -- that is how
            # `m0` is written and it is what the shipping kernel does too. One
            # base per slot, so the slot never becomes a runtime offset.
            lane0 = []
            for v in dst_views:
                wave_byte = fx.Int32(fx.ptrtoint(fx.get_iter(v))) + grp_wave * fx.Int32(
                    WARP_SIZE * DMA_BYTES
                )
                lane0.append(rocdl.readfirstlane(fx.Int32.ir_type, wave_byte))

            def fire(slot, q_start, d):
                # A row past the end must still fetch SOMETHING -- an
                # out-of-range `buffer_load ... lds` leaves the LDS slot
                # unwritten rather than zeroed, and a stale NaN there poisons
                # dV through `0 * Inf` (see `rm_zero`). Which row it reads is
                # free, because the gate masks these rows out, so clamp to the
                # sequence's LAST row rather than selecting token 0: that keeps
                # the whole computation in the element-offset domain and drops
                # the parallel token-index chain the bound check used to need.
                raw = q_start * c_stride_n + rowoff[d]
                src_elem = (raw > bound[d]).select(bound[d], raw)
                lds_byte = fx.Int32(lane0[slot]) + fx.Int32(
                    d * GRP_THREADS * DMA_BYTES
                )
                dst = fx.make_view(
                    fx.inttoptr(_dma_ptr_ty, lds_byte), fx.make_layout(1, 1)
                )
                fx.copy(_dma_atom, fx.slice(div, (None, fx.Int32(src_elem))), dst)

            return fire

        dma_q = make_dma(q, q_base_byte_offset, q_views, True)
        dma_do = make_dma(do, do_base_byte_offset, do_views, True)

        # ---- staging duty: LDS -> LDS transpose ----------------------------
        t_unit_row = grp_tid // fx.Int32(T_QGROUPS)
        t_unit_qg = grp_tid % fx.Int32(T_QGROUPS)
        t_unit_qbase = t_unit_qg * fx.Int32(MFMA_LANE_K)
        # The unit's transposed row `d` splits into the chunk that owns it and
        # the lane within that chunk -- which is exactly the [g][m][c] address.
        t_unit_m = t_unit_row % fx.Int32(MFMA_M)
        t_unit_c = t_unit_row // fx.Int32(MFMA_M)

        def make_transpose(rm_views_, t_views_, swizzled):
            def run(rm_slot, t_slot):
                view = rm_views_[rm_slot]
                elems = []
                for i in range_constexpr(MFMA_LANE_K):
                    row = t_unit_qbase + fx.Int32(i)
                    col = (
                        q_swz_col(row, t_unit_row)
                        if const_expr(swizzled)
                        else t_unit_row
                    )
                    elems.append(
                        view[
                            row,
                            col // fx.Int32(MFMA_LANE_K),
                            col % fx.Int32(MFMA_LANE_K),
                        ]
                    )
                t_views_[t_slot][t_unit_qg, t_unit_m, t_unit_c, None].store(
                    Vec.from_elements(elems, elem_dtype)
                )

            return run

        transpose_q = make_transpose(q_views, q_t_wr, True)
        transpose_do = make_transpose(do_views, do_t_wr, True)

        def rm_zero(owns_q):
            """Zero this group's row-major Q/dO tiles once, in the prologue.

            With the buffer descriptor sized to the sequence, a `buffer_load
            ... lds` whose row is past the end does not write LDS at all -- it
            does NOT deposit zero (measured: dropping the gate's
            `q_raw < seq_len` on that assumption gave 13-35% mismatch). So the
            tail rows of the last tile are simply left as they were. That is
            fine from the second tile on, where they hold an earlier tile's
            finite data and the gate masks them -- but on the first tile they
            would be whatever the previous dispatch left, and a NaN there meets
            a masked zero in GEMM1/GEMM3 and poisons dV/dK for the whole sweep.
            Same hazard, and same fix, as `transpose_zero` above.

            Each group zeroes ONLY the tensor it DMAs. Zeroing both would be
            a cross-group race: `lgkmcnt` is per-wave, so g1's `ds_write` to
            the Q tile can retire after g0's `buffer_load ... lds` has filled
            it, erasing the tile. Measured as 0.7-1.4% mismatch.
            """
            z = Vec(c_zero_pack)
            row = grp_tid // fx.Int32(STRIDE // MFMA_LANE_K)
            colgrp = grp_tid % fx.Int32(STRIDE // MFMA_LANE_K)
            views = q_views if const_expr(owns_q) else do_views
            for slot in range_constexpr(RM_SLOTS):
                views[slot][row, colgrp, None].store(z)

        def transpose_zero(t_slot):
            # The T_GPAD tail of each q-group is never read, so zeroing the
            # written footprint is enough to keep trip 0's 0 * x clean.
            z = Vec(c_zero_pack)
            q_t_wr[t_slot][t_unit_qg, t_unit_m, t_unit_c, None].store(z)
            do_t_wr[t_slot][t_unit_qg, t_unit_m, t_unit_c, None].store(z)

        # ---- the 16 MFMA operand groups ------------------------------------
        # g 0..3   GEMM1 dV: B = dO^T[tile i-1]        A = p_prev   (register)
        # g 4..7   GEMM3 dK: B = Q^T [tile i-1]        A = ds_prev  (register)
        # g 8..11  GEMM0 S : A = Q    [tile i]         B = k_packs  (register)
        # g 12..15 GEMM2 dA: A = dO   [tile i]         B = v_packs  (register)
        def read_operand(g, rm_slot, t_slot, halves):
            if const_expr(g < 2 * D_CHUNKS):
                # One `ds_read_b128` serves CHUNKS_PER_B128 operand groups; the
                # second group slices its 2 dwords out of the half the first
                # one already loaded.  `halves` memoises at TRACE time, so the
                # sharing is structural -- it does not depend on the backend
                # CSE-ing two reads across the sched_barriers between them.
                which = 0 if const_expr(g < D_CHUNKS) else 1
                c = g if const_expr(g < D_CHUNKS) else g - D_CHUNKS
                h, sub = c // CHUNKS_PER_B128, c % CHUNKS_PER_B128
                key = (which, h, t_slot)
                if key not in halves:
                    views = do_t_rd if const_expr(which == 0) else q_t_rd
                    halves[key] = (
                        Vec(views[t_slot][lane_div_16, lane_mod_16, h, None].load())
                        .bitcast(fx.Int32)
                    )
                v = halves[key]
                words = [
                    v[sub * PACK_DWORDS + i] for i in range_constexpr(PACK_DWORDS)
                ]
                return (
                    Vec.from_elements(words, fx.Int32).bitcast(elem_dtype).ir_value()
                )
            q_row = lane_mod_16
            if const_expr(g < 3 * D_CHUNKS):
                col = fx.Int32((g - 2 * D_CHUNKS) * MFMA_QK_K) + lane_div_16 * fx.Int32(
                    MFMA_QK_LANE_K
                )
                view = q_views[rm_slot]
                return (
                    view[q_row, q_swz_col(q_row, col) // fx.Int32(MFMA_QK_LANE_K), None]
                    .load()
                    .ir_value()
                )
            col = fx.Int32((g - 3 * D_CHUNKS) * MFMA_DA_K) + lane_div_16 * fx.Int32(
                MFMA_DA_LANE_K
            )
            view = do_views[rm_slot]
            return (
                view[q_row, q_swz_col(q_row, col) // fx.Int32(MFMA_DA_LANE_K), None]
                .load()
                .ir_value()
            )

        def fmul_scalar(a, b):
            """Multiply that the backend cannot fold into `v_pk_mul_f32`.

            EXPERIMENT: `v_mul_legacy_f32` has no packed form, so a pair of them
            stays two scalar VALU ops. Semantics differ from `fmul` only for
            0 * inf / 0 * NaN (legacy gives 0); every operand here -- the score
            accumulator, alpha, a sigmoid in (0,1], dA -- is finite, so the
            results are identical.
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

            `fmul_scalar`'s legacy trick leaks here. `fmul_legacy` differs from
            `fmul` only on 0*inf and 0*NaN, so when the other operand is a
            finite non-zero CONSTANT the two are provably identical, InstCombine
            drops the intrinsic, and SLP re-packs the pair. Measured: this one
            call site was every `v_pk_mul_f32` left in the loop (12 of them,
            24 elements) while the three `fmul_scalar` calls with two SSA
            operands all stayed `v_mul_legacy_f32`.

            `b` takes the `s` constraint so the constant lands in an SGPR and
            is read as VOP2 src0 -- no VGPR is burned to hold it.

            No `s_nop`, unlike `addone_scalar`: `a` is a GEMM0 accumulator, and
            an MFMA result needs 7 wait states before a VALU reads it, but the
            transpose tail, `s_setprio`, `s_waitcnt` and `s_barrier` all sit
            between the two phases and `phase_barrier`'s `sched_barrier(0)`
            pins them there. (Checked in the ISA, not assumed.)
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
            `v_pk_add_f32`, and every FMA/legacy rewrite folds back to the same
            add. Inline asm is the only way to pin it scalar.

            This one op is worth pinning. Measured in the V phase, `1 + emu` is
            6 x `v_pk_add_f32` carrying 75.6 cy of stall each -- 451 cy of a
            2356 cy phase (19%) -- while the OTHER packed opcode in the same
            phase, on the same datapath, stalls 2.1 cy. The difference is not
            the datapath, it is that a packed op is a JOIN: it cannot issue
            until BOTH halves' `v_exp_f32` retire. Scalar, element j's add
            fires on its own exp alone.
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
                    # the kernel reads a stale register: dv/dk max|err|/peak
                    # 0.33 on ~0.3% of elements under the causal mask, while
                    # the hstu mask still passed, so it is easy to miss.
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

        def fadd_scalar(a, b):
            """`a + b` as a scalar `v_add_f32`, blocking the SLP repack.

            Same motivation as `fmul_const_scalar`. The dQ flush accumulates
            DQ_PER_LANE=4 independent lanes of fp32; SLP pairs them into
            `v_pk_add_f32`, which halves the number of independent dependency
            chains the scheduler can interleave. This phase is latency-bound,
            not VALU-throughput-bound, so that is a loss -- the same finding as
            the gate's packing.

            No `s_nop`: both operands come from VALU/LDS, never an MFMA
            accumulator, so there is no wait state for the hazard recogniser to
            miss inside the INLINEASM it cannot see into.
            """
            from flydsl._mlir import ir
            from flydsl._mlir.dialects import llvm as _llvm

            return fx.Float32(
                _llvm.inline_asm(
                    ir.F32Type.get(),
                    [fx.Float32(a).ir_value(), fx.Float32(b).ir_value()],
                    "v_add_f32 $0, $1, $2",
                    "=v,v,v",
                    has_side_effects=False,
                )
            )

        def fma_scalar(a, b, c):
            """`a*b + c` as a scalar `v_fma_f32`, via inline asm.

            Written with plain operators this contracts to `v_fma_f32` and is
            then re-vectorised into `v_pk_fma_f32` -- a join on three chain
            values, which is the stall this gate was restructured to avoid (it
            also spilled 3 VGPRs). Inline asm pins it scalar.

            No `s_nop` here, unlike `addone_scalar`. `c` is a `v_rcp_f32`
            result and a VALU trans op does need one software wait state --
            but the other two operands are computed FROM that same result
            (`silu = sc*sig`, `b = emu*sig`), so this cannot issue until at
            least two instructions after the rcp. The data dependency supplies
            the wait state that the hazard recognizer cannot see.
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
            """Scalar reciprocal, kept off the packed datapath."""
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

        # Identity in B-operand layout: lane l element i is 1.0 exactly when
        # 4*(l/16)+i == l%16. Loop-invariant, 2 VGPR.
        c_ident = pack_bf16_perm(
            [
                (
                    lane_div_16 * fx.Int32(MFMA_LANE_K) + fx.Int32(_i) == lane_mod_16
                ).select(fx.Float32(1.0), fx.Float32(0.0))
                for _i in range_constexpr(MFMA_ELEMS_PER_LANE)
            ]
        )

        def transpose_frag(frag):
            """Transpose a 16x16 fragment with ONE MFMA, no LDS and no shuffle.

            The A and B operand layouts are transposes of each other -- A is
            `lane l elem i <-> [l%16][4*(l/16)+i]`, B is `[4*(l/16)+i][l%16]`
            -- and the C accumulator shares B's. So feeding a B-layout fragment
            of X as the A operand makes it denote X^T, and `X^T . I = X^T`
            lands in C layout, which read back IS the A layout of X. Verified
            exhaustively: 256 of 256 lane-slots.

            Bit-exact. The input is already bf16, `x * 1.0` accumulated in fp32
            is exact, and the pack back to bf16 is exact.

            This replaces an LDS round trip -- `ds_write`, fence, four strided
            `ds_read_u16`, fence, per subtile. Done inside one phase that cost
            16.4 M cycles, 16.7% of all stall in the dispatch-150 trace, since
            a wave cannot see its own LDS write without draining `lgkmcnt`.
            Splitting the write into the previous gate phase removed those
            waits and was still SLOWER (+28 ms): `lgkmcnt` is in-order, so
            reads issued before the operand window join the first MFMA's wait.
            The matrix core sidesteps the counter altogether.
            """
            _c = mfma_acc(frag, c_ident, c_zero_v4f32)
            return pack_bf16_perm(
                [Vec(_c)[_e] for _e in range_constexpr(MFMA_ELEMS_PER_LANE)]
            )

        def keep_masks(q_start, og):
            q_base = q_start + lane_div_16 * fx.Int32(MFMA_LANE_K)
            out = []
            for i in range_constexpr(MFMA_ELEMS_PER_LANE):
                q_raw = q_base + fx.Int32(i)
                q_id = to_id(q_raw)
                # `kv_id < q_id` rather than `q_id - kv_id > 0`: the subtract
                # is a VALU op whose only consumer is the comparison, and the
                # comparison can take both operands directly. Materialise
                # `dist` only when the window actually needs its value.
                # Measured: 12 x `v_sub_u32` per gate phase, 48 of 1045 cy.
                keep = (q_raw == kv_rows[og]) | (kv_owned_ids[og] < q_id)
                if has_window:
                    dist = q_id - kv_owned_ids[og]
                    keep = keep & (dist <= fx.Int32(max_attn_len))
                if has_contextual:
                    ctx = (q_id == fx.Int32(0)) & (kv_owned_ids[og] < max_id)
                    keep = keep | ctx
                out.append(keep & (q_raw < seq_len) & kv_in_bounds[og])
            return out

        # ---- dQ split-K reduction + flush ----------------------------------
        # Scratch base for this (batch, kv_tile); -1 marks a tile the sequence
        # does not have, which only an out-of-range q row could reach and those
        # are steered to the trash row below.
        dq_poff = part_off[
            fx.Int64(batch_idx * fx.Int32(num_kv_tiles) + kv_tile_idx)
        ]
        # 256 lanes per group, 4 contiguous outputs each.
        dq_gtid = (wave_id % fx.Int32(NUM_WAVES // 2)) * fx.Int32(WARP_SIZE) + _lane
        dq_frow = dq_gtid // fx.Int32(DQ_LANES_PER_ROW)
        dq_fvec = dq_gtid % fx.Int32(DQ_LANES_PER_ROW)
        dq_fcol = dq_fvec * fx.Int32(DQ_PER_LANE)
        dq_wbase = is_g0.select(fx.Int32(0), fx.Int32(NUM_WAVES // 2))
        # g1 writes its group sum to the row g0 will read; g0 writes to the
        # sink row and reads the real one. g1 reads the permanently-zero row,
        # so it adds nothing. No branch, and neither group idles.
        dq_wrow = is_g0.select(fx.Int32(DQG_SINK), dq_frow)
        dq_rrow = is_g0.select(dq_frow, fx.Int32(DQG_ZERO))

        def dq_reduce_flush(q_tile_v):
            """Sum the 8 per-wave dQ partials for q tile `q_tile_v - 1` and store.

            Runs in phase V, not phase M, and that placement is forced. Within
            one M phase a group's four waves have no mutual visibility -- there
            is no barrier inside a phase -- so a wave cannot read a slot another
            wave of its own group just wrote. The intervals, with g1 one ahead:

                A    g1 M(i)    writes slots 4-7
                A+1  g0 M(i)    writes slots 0-3 | g1 V(i)  sums 4-7 -> dqg
                A+2  g1 M(i+1)  rewrites 4-7     | g0 V(i)  sums 0-3 + dqg, stores

            Each group reads its OWN slots one interval after writing them and
            one interval before overwriting them, so ONE set of eight slots is
            enough -- two sets would be 65,536 B and would not fit beside the
            16,640 B the dV/dK schedule already holds.

            g1 stores too, to the trash row. Its 256 lanes collapse onto the 64
            dwords of this workgroup's single head there, so the wasted HBM
            traffic is 256 B per tile; only the issue slot is paid.
            """
            # One `ds_read_b128` per source wave, then SCALAR adds: the wide
            # load and the unpacked arithmetic are independent wins and the
            # naive form gives up one to get the other.
            # ALL four reads first, then the adds. Interleaved, the compiler
            # rotates only two destination buffers, so read 3 cannot issue
            # until the adds have drained read 1 -- a read/wait/add chain with
            # ILP 2. The trace charged 4.87 M cycles to that third
            # `ds_read_b128` alone (5.7% of all stall, 239/exec). Four live
            # buffers cost 16 VGPR against 46 spare, and push every `lgkmcnt`
            # further from the read it covers.
            _vs = [
                Vec(dqw[dq_wbase + fx.Int32(w), dq_frow, dq_fvec, None].load())
                for w in range_constexpr(NUM_WAVES // 2)
            ]
            acc = [fx.Float32(0.0) for _ in range_constexpr(DQ_PER_LANE)]
            for w in range_constexpr(NUM_WAVES // 2):
                for j in range_constexpr(DQ_PER_LANE):
                    acc[j] = fadd_scalar(acc[j], _vs[w][j])
            dqg[dq_wrow, dq_fvec, None].store(
                Vec.from_elements(acc, fx.Float32)
            )
            # No fence between the write and the read: NEITHER group reads what
            # it just wrote. g1 writes `dq_frow` and reads the permanently-zero
            # row; g0 writes the sink row and reads the row g1 published a full
            # barrier ago. A defensive lgkmcnt(0) here is a whole LDS round
            # trip on the critical path of every gate phase.
            _g = Vec(dqg[dq_rrow, dq_fvec, None].load())
            for j in range_constexpr(DQ_PER_LANE):
                acc[j] = fadd_scalar(acc[j], _g[j])

            # The partial belongs to tile q_tile_v - 1: phase M consumes
            # `ds_prev`, produced by the PREVIOUS trip's gate phase.
            _qa = (q_tile_v - fx.Int32(1)) * fx.Int32(BLOCK_N) + dq_frow
            # `dq_part` is [row][H*D] and the offsets are in ROWS. A flat
            # element layout is the same memory but overflows the launch ABI,
            # which packs dynamic extents as i32: the scratch is 21e9 fp32 per
            # pass at the benchmark shape.
            _r = fx.Int64(dq_poff) + fx.Int64(_qa - kv_start_row)
            # Upper bound is the jagged tail; the lower one catches the first
            # trip, whose `ds_prev` is the zero pack from `acc_init` and whose
            # nominal tile is q_tile_start-1.
            _r = (_qa < seq_len).select(_r, fx.Int64(0))
            _r = (_qa >= kv_start_row).select(_r, fx.Int64(0))
            _r = is_g0.select(_r, fx.Int64(0))
            _hc = head_idx * fx.Int32(head_dim) + dq_fcol
            for j in range_constexpr(DQ_PER_LANE):
                # Scalar, not `acc[j] * c_inv_n`: packed, this was the single
                # largest non-barrier stall in the dispatch-150 trace --
                # 12.4 M cycles, 12.4% of all stall, 317 cy/exec.
                dq_part[_r, _hc + fx.Int32(j)] = fmul_const_scalar(
                    acc[j], c_inv_n
                )  # fmt: skip

        # ---- phase M -------------------------------------------------------
        def mfma_phase(dv, dk, p_prev, ds_prev, q_tile, rm_cur, rm_fill, t_cur,
                       t_prev, kt_packs):
            """Four GEMMs, the operand feed, this group's DMA and transpose, the mask.

            `kt_packs` is a PARAMETER, unlike every other loop-invariant here
            (`k_packs`, `v_packs`, ...), because it is built after the prologue
            fence and so lives inside the `if active:` branch -- which the AST
            rewriter lowers to its own function scope. Closing over it from out
            here is a NameError, not a late binding.
            """
            # dS in the layout the dQ GEMM's A operand needs. Per-tile, unlike
            # kt_packs: one scratch slot per subtile, written and read inside
            # this phase by the owning wave only, so no barrier is involved and
            # the slots never cross one.
            if const_expr(dq_mode == "notrans"):
                # TIMING ONLY, RESULTS ARE WRONG: feeds the dQ GEMM the
                # untransposed dS to price the transpose on its own.
                ds_t = [ds_prev[og] for og in range_constexpr(KV_OWNED_SUBTILES)]
            elif const_expr(dq_mode != "nodq"):
                ds_t = [
                    transpose_frag(ds_prev[og])
                    for og in range_constexpr(KV_OWNED_SUBTILES)
                ]

            rocdl.s_setprio(1)
            q_start = fx.Int32(q_tile) * fx.Int32(BLOCK_N)
            s_accs = [c_zero_v4f32 for _ in range_constexpr(KV_OWNED_SUBTILES)]
            da_accs = [c_zero_v4f32 for _ in range_constexpr(KV_OWNED_SUBTILES)]

            # Prime the rolling window before the first MFMA.  Group 0 is GEMM1,
            # whose dO^T operand was published at the earliest barrier.
            # Ring of PREFETCH_W + 1 slots indexed by STEP, not by group value.
            # `g % (W+1)` is only correct when the ring is as wide as
            # GROUP_ORDER: the order is an interleaved permutation, so for
            # W <= 11 a later group's read lands in a slot whose occupant has
            # not been consumed yet -- at W=9, g12's read is issued at step 0
            # into g2's slot and g2 is not consumed until step 4, which gives
            # silently wrong results. Keyed by step, the slot reuse distance is
            # exactly W+1 against a W lookahead, so every W is correct.
            win = [None for _ in range_constexpr(PREFETCH_W + 1)]
            halves = {}
            for _s in range_constexpr(PREFETCH_W):
                _g = GROUP_ORDER[_s]
                win[_s % (PREFETCH_W + 1)] = read_operand(_g, rm_cur, t_prev, halves)


            for step in range_constexpr(NUM_OPERAND_GROUPS):
                g = GROUP_ORDER[step]
                rocdl.sched_barrier(0)
                op = win[step % (PREFETCH_W + 1)]
                for og in range_constexpr(KV_OWNED_SUBTILES):
                    if const_expr(g < D_CHUNKS):
                        off = g * KV_OWNED_SUBTILES + og
                        dv[off] = mfma_acc(p_prev[og], op, dv[off])
                    elif const_expr(g < 2 * D_CHUNKS):
                        off = (g - D_CHUNKS) * KV_OWNED_SUBTILES + og
                        dk[off] = mfma_acc(ds_prev[og], op, dk[off])
                    elif const_expr(g < 3 * D_CHUNKS):
                        s_accs[og] = qk_mfma_acc(
                            op, k_packs[g - 2 * D_CHUNKS][og], s_accs[og]
                        )
                    else:
                        da_accs[og] = da_mfma_acc(
                            op, v_packs[g - 3 * D_CHUNKS][og], da_accs[og]
                        )

                sn = step + PREFETCH_W
                if const_expr(sn < NUM_OPERAND_GROUPS):
                    gn = GROUP_ORDER[sn]
                    win[sn % (PREFETCH_W + 1)] = read_operand(
                        gn, rm_cur, t_prev, halves
                    )

                # The DMA fire is spread through the burst's slack; it is VMEM,
                # so it does not interact with the LDS retirement order below.
                # The duty branch is wave-uniform (`is_g0` comes from
                # readfirstlane) and side-effect-only, so it lowers to a scalar
                # branch.
                if const_expr(step == 2):
                    if is_g0:
                        dma_q(rm_fill, q_start + fx.Int32(BLOCK_N), 0)
                    else:
                        dma_do(rm_fill, q_start + fx.Int32(BLOCK_N), 0)
                if const_expr(step == 6):
                    if is_g0:
                        dma_q(rm_fill, q_start + fx.Int32(BLOCK_N), 1)
                    else:
                        dma_do(rm_fill, q_start + fx.Int32(BLOCK_N), 1)
                rocdl.sched_barrier(0)

            # Build this tile's transpose AFTER every operand read has been
            # issued, never in the middle of the burst.
            #
            # `lgkmcnt` retires IN ORDER, so these 5 ops (4 scalar ds_read_u16 +
            # 1 ds_write_b64) sitting at group 3 put themselves on the critical
            # path of all ten operand reads issued behind them: each of those
            # could only retire once a 2-byte strided gather had. That is what
            # produced lgkmcnt(0)/(1) in front of MFMAs that needed only
            # lgkmcnt(3), and it is why deepening PREFETCH_W did nothing -- the
            # window was never the constraint, queue position was.
            #
            # Placing it here exposes its own latency instead of everyone
            # else's, and leaves slot lifetimes untouched (still: write t_cur,
            # read t_prev, both within phase M), so it adds no cross-group
            # hazard.
            rocdl.sched_barrier(0)
            if is_g0:
                transpose_q(rm_cur, t_cur)
            else:
                transpose_do(rm_cur, t_cur)
            rocdl.sched_barrier(0)

            # ABLATION (`dq_mode`): "nodq" drops all dQ work, "nostore" keeps
            # the GEMM and the LDS reduction but not the HBM flush.
            if const_expr(dq_mode != "nodq"):
                # ---- dQ: partial over THIS workgroup's KV range ----------------
                # dQ[q][d] = sum_kv dS[q][kv].K[kv][d]. Both operands are
                # register-resident (transposed dS, kt_packs), so this needs no
                # operand-window group -- at PREFETCH_W=15 every window slot is
                # live at once and an LDS-fed group would add live registers
                # one-for-one.
                #
                # C[m=q][n=d] over 4 d-chunks x KV_OWNED_SUBTILES kv steps.
                rocdl.sched_barrier(0)
                # Operands SWAPPED: this computes dQ^T, C[m=d][n=q]. Same
                # arithmetic for free, different lane layout -- a lane then
                # holds 4 CONSECUTIVE d of ONE q row, so the staging write
                # below is one `ds_write_b128` per chunk instead of 4 scattered
                # dwords. Unswapped, C is [m=q][n=d]: a lane holds one d across
                # 4 q rows, strided by DQ_PAD, and the flush's read wants the
                # other axis -- only one side could ever be contiguous.
                #
                # A = kt_packs reads as K^T[d][kv], B = ds_t as dS^T[kv][q],
                # and K^T.dS^T = (dS.K)^T = dQ^T. Checked against the lane maps.
                dq_acc = [c_zero_v4f32 for _ in range_constexpr(K_STEPS)]
                for ks in range_constexpr(K_STEPS):
                    for og in range_constexpr(KV_OWNED_SUBTILES):
                        dq_acc[ks] = mfma_acc(
                            kt_packs[ks][og], ds_t[og], dq_acc[ks]
                        )  # fmt: skip

                # Park this wave's partial in its PRIVATE slot. Phase V sums
                # the eight of them; see `dq_reduce_flush` for why the reduction
                # cannot happen here.
                #
                # No scale: `c_inv_n` is applied once at the store, 4 elements
                # per lane instead of 16. Alpha needs nothing -- `kt_packs` is
                # transposed from the alpha-scaled `k_packs`, and
                # ds.(aK) = a.(ds.K) is exactly the chain rule. Do NOT also
                # apply dq_pp's `alpha/max_seq_len`; that folds alpha into Q.
                for ks in range_constexpr(K_STEPS):
                    # q = l%16, d = ks*MFMA_M + MFMA_LANE_K*(l/16) + e, so the
                    # four elements are one contiguous run of the
                    # (.., BLOCK_N, head_dim/VEC, VEC) view.
                    dqw[
                        wave_id,
                        lane_mod_16,
                        fx.Int32(ks * MFMA_M // DQ_PER_LANE) + lane_div_16,
                        None,
                    ].store(Vec(dq_acc[ks]))  # fmt: skip
            rocdl.sched_barrier(0)

            rocdl.s_setprio(0)
            # The tile i+1 DMA fired above must land before the mid barrier, not
            # merely before this group's own next read: with the half-iteration
            # skew the PARTNER's M(i+1) runs in the interval this barrier opens,
            # and `vmcnt` is per-wave. Firing at groups 2 and 6 of 16 leaves it
            # most of the burst to return. `lgkmcnt(0)` publishes the transpose
            # write and retires this phase's reads before the next trip's DMA.
            rocdl.s_waitcnt(vmcnt=0)
            rocdl.s_waitcnt(lgkmcnt=0)
            return dv, dk, s_accs, da_accs, q_start

        # ---- phase V -------------------------------------------------------
        def gate_phase(s_accs, da_accs, q_start):
            """Fused SiLU / SiLU' gate: produces both P and dS from one sigmoid.

            GEMM0 and GEMM2 both finished in phase M, so `grad` never has to
            outlive the element it belongs to -- unlike the non-ping-pong kernel,
            which carries `s_meta` across a barrier and 24 MFMAs.

            Also carries the dQ cross-wave reduction. That is the one memory
            operation in this phase, and it is here because phase M cannot host
            it (see `dq_reduce_flush`) -- but it also lands well: the partner
            group is in phase M on the matrix core, so the LDS and store units
            are otherwise idle for the whole interval.
            """
            if const_expr(dq_mode in ("full", "dswrite", "notrans")):
                dq_reduce_flush(q_start // fx.Int32(BLOCK_N))
            # Batched a stage at a time across ALL KV_OWNED_SUBTILES *
            # MFMA_ELEMS_PER_LANE elements, not one element at a time. The gate
            # is a ~10-deep serial dependency chain per element with two
            # quarter-rate transcendentals in it, so the only way to fill the
            # VALU pipe is to give the scheduler every independent chain at
            # once. Measured 10 cy per VALU op when this was written per
            # element -- latency-bound, not throughput-bound.
            n = KV_OWNED_SUBTILES * MFMA_ELEMS_PER_LANE
            rng = range_constexpr(n)
            # The mask needs no GEMM result, so it can sit in either phase --
            # but it must NOT go inside the fenced MFMA burst. Interleaved
            # between the groups, these ~33 VALU ops are pinned by the
            # sched_barriers and stretch the MFMA stream: measured +9%.
            keep = [keep_masks(q_start, og) for og in range_constexpr(KV_OWNED_SUBTILES)]
            flat_keep = [
                keep[j // MFMA_ELEMS_PER_LANE][j % MFMA_ELEMS_PER_LANE] for j in rng
            ]
            with arith.fastmath(arith.FastMathFlags.fast):
                s = [Vec(s_accs[j // MFMA_ELEMS_PER_LANE])[j % MFMA_ELEMS_PER_LANE]
                     for j in rng]
                da = [Vec(da_accs[j // MFMA_ELEMS_PER_LANE])[j % MFMA_ELEMS_PER_LANE]
                      for j in rng]
                # Both derived straight from `s`, not `emu` from `sc`: that
                # serialised two multiplies at the head of a chain whose next
                # link is a 16-cycle transcendental. Same op count, one less
                # link, and the two multiplies now issue in parallel.
                # `sc` arrives already scaled from GEMM0 (alpha is folded into
                # k_packs), so only the exp2 argument needs a multiply.
                sc = s
                emu = [exp2_f32(fmul_const_scalar(sc[j], c_neg_log2e)) for j in rng]
                sig = [rcp_scalar(addone_scalar(emu[j])) for j in rng]
                silu = [fmul_scalar(sc[j], sig[j]) for j in rng]
                # silu'(z) = sig*(1 + z*(1 - sig)) = sig + silu*(1 - sig).
                # The second form is a subtract plus an FMA, against the first
                # form's sub/mul/add/mul -- 2 VALU per element instead of 4, and
                # it reuses `silu`, which is already live.
                # 1 - sig == emu * sig exactly, since sig = 1/(1+emu). Using
                # the product keeps this off the packed add datapath too.
                grad = [
                    fma_scalar(silu[j], fmul_scalar(emu[j], sig[j]), sig[j])
                    for j in rng
                ]
                # No `c_inv_n` here: dS is linear in it, so it rides along to the
                # dK epilogue and is folded into that store's existing scale.
                # Saves one multiply per element.
                gated = [fmul_scalar(grad[j], da[j]) for j in rng]
                p_vals = [flat_keep[j].select(silu[j], c_zero_f) for j in rng]
                ds_vals = [flat_keep[j].select(gated[j], c_zero_f) for j in rng]
            e = MFMA_ELEMS_PER_LANE
            p_packs = [
                pack_bf16_perm(p_vals[og * e : og * e + e])
                for og in range_constexpr(KV_OWNED_SUBTILES)
            ]
            ds_packs = [
                pack_bf16_perm(ds_vals[og * e : og * e + e])
                for og in range_constexpr(KV_OWNED_SUBTILES)
            ]
            return p_packs, ds_packs

        # ---- the pipeline ---------------------------------------------------
        if active:
            # Trip 0's GEMM1/GEMM3 read the transpose of a tile that does not
            # exist. Their A operands are zero, but `0 * NaN` is NaN and would
            # poison the accumulators for the whole sweep, so zero that slot.
            transpose_zero(1)
            # Must retire BEFORE the tile-0 DMA is fired: these `ds_write`s and
            # the `buffer_load ... lds` below target the same LDS, and without
            # the fence the zeroing can land after the load and erase it.
            # (`transpose_zero` needs no such fence -- it writes q_t/do_t,
            # which the DMA never touches.)
            if is_g0:
                rm_zero(True)
            else:
                rm_zero(False)
            rocdl.s_waitcnt(lgkmcnt=0)
            # Prime the pipeline with tile T0; the loop fires one tile ahead.
            _t0 = q_tile_start * fx.Int32(BLOCK_N)
            if is_g0:
                dma_q(0, _t0, 0)
                dma_q(0, _t0, 1)
            else:
                dma_do(0, _t0, 0)
                dma_do(0, _t0, 1)
            # The skew: g0 burns one extra barrier so that from here on g0 is in
            # phase M exactly when g1 is in phase V, and vice versa. Barrier
            # counts stay equal overall -- g1 repays it after the loop.
            #
            # It MUST come before the prologue's DMA fence, not after. Each
            # group stages only its own tensor (g0 Q, g1 dO) but trip 0 reads
            # BOTH, so the fence that publishes them has to be the last thing
            # either group does before the loop. With the skew after the fence,
            # g1 enters M(0) a full interval ahead while g0 still sits at the
            # skew barrier, and g0's trip-0 GEMM2 reads a dO tile that g1 staged
            # without an intervening publish -- nondeterministic dA, hence dV
            # and dK, confined to the q rows g1's waves 2-3 wrote.
            #
            # Invisible at the tuned (384,16,8) shape, which is why this
            # survived: measured 0/63 runs there, against 36/63 at
            # (256,16,8) and 10/27 at (512,16,8). Output is bitwise unchanged
            # at every shape -- this reorders publication, not arithmetic.
            if is_g0:
                phase_barrier()
            rocdl.s_waitcnt(vmcnt=0)
            rocdl.s_waitcnt(lgkmcnt=0)
            phase_barrier()

            # K in the kv-as-reduction layout, for the dQ GEMM's B operand.
            # Loop-invariant, so paid ONCE per workgroup -- and it avoids
            # re-reading K from HBM in a second layout. Transposed from the
            # ALPHA-SCALED k_packs so alpha rides along: ds.(aK) = a.(ds.K) is
            # exactly the chain rule, leaving only 1/N for the flush. Do NOT
            # also apply dq_pp's `alpha/max_seq_len`; that kernel folds alpha
            # into Q, not K.
            #
            # Placed AFTER the prologue fence, not beside k_packs: up there it
            # runs while the zeroing and the staging DMA are still being set
            # up, and it corrupted dv/dk (5.4k elements). Here nothing is in
            # flight.
            # Only `dqg` needs clearing, and only its permanently-zero row --
            # but zero the whole thing, it is one short loop and leaves no
            # assumption about which rows g1 reads. `dqw` is fully overwritten
            # by every M phase before anything reads it.
            for _z in range_constexpr(
                (DQG_ELEMS + BLOCK_THREADS - 1) // BLOCK_THREADS
            ):
                _zi = tid + fx.Int32(_z * BLOCK_THREADS)
                _zi = (_zi < fx.Int32(DQG_ELEMS)).select(_zi, fx.Int32(0))
                dqg_flat[_zi] = fx.Float32(0.0)
            rocdl.s_waitcnt(lgkmcnt=0)

            kt_packs = []
            for ks in range_constexpr(K_STEPS):
                per_og = []
                for og in range_constexpr(KV_OWNED_SUBTILES):
                    per_og.append(transpose_frag(k_packs[ks][og]))
                kt_packs.append(per_og)


            acc_init = [c_zero_v4f32 for _ in range_constexpr(N_ACC)] + [
                c_zero_pack for _ in range_constexpr(2 * N_PACK)
            ]
            loop_results = acc_init
            # The body is unrolled UNROLL times so every slot index below is a
            # literal, which is what keeps each LDS access inside one global.
            # The trip count is rounded UP to a multiple of UNROLL rather than
            # peeled: a tile past `q_upper` has every `keep` false, so its gate
            # emits zero packs and the next trip's GEMM1/GEMM3 add exactly
            # nothing. That is the same property the drain trip already relies
            # on, so the padding is free of correctness risk and costs <1 trip.
            n_trips = n_q_tiles + fx.Int32(1) - q_tile_start
            n_outer = (n_trips + fx.Int32(UNROLL - 1)) // fx.Int32(UNROLL)
            for _outer, it in range(
                fx.Int64(0), fx.Int64(n_outer), fx.Int64(1), init=acc_init
            ):  # ty: ignore
                it_list = list(it) if isinstance(it, (list, tuple)) else [it]
                dv = [it_list[i] for i in range_constexpr(N_ACC_DV)]
                dk = [it_list[N_ACC_DV + i] for i in range_constexpr(N_ACC_DK)]
                p_prev = [it_list[N_ACC + i] for i in range_constexpr(N_PACK)]
                ds_prev = [
                    it_list[N_ACC + N_PACK + i] for i in range_constexpr(N_PACK)
                ]
                base_tile = q_tile_start + fx.Int32(_outer) * fx.Int32(UNROLL)

                # Loop rotated: the closing barrier of each gate phase is
                # issued at the TOP of the next sub-trip rather than the bottom
                # of this one. The scf.for back edge -- increment, 64-bit trip
                # compare, branch -- therefore lands in the tail of the gate
                # phase instead of at the head of the MFMA phase.
                #
                # It was costing 184 cy/exec there: the trip test needs the
                # bound in a VGPR (v_cmp takes only one SGPR operand) and the
                # allocator chose v[108:109], the registers `ds_read2_b64`
                # overwrites a few instructions later, so the copy stalled on a
                # WAR hazard. Ahead of the barrier that hazard resolves during
                # the rendezvous wait, which is time the wave was spending
                # anyway.
                for u in range_constexpr(UNROLL):
                    if const_expr(u == 0):
                        phase_barrier()
                    q_tile = base_tile + fx.Int32(u)
                    # Literal slots: tile i lives in i % RM_SLOTS, its transpose
                    # in i % T_SLOTS, and GEMM1/GEMM3 read tile i-1's.
                    rm_cur = u % RM_SLOTS
                    rm_fill = (u + 1) % RM_SLOTS
                    t_cur = u % T_SLOTS
                    t_prev = (u + 1) % T_SLOTS
                    dv, dk, s_accs, da_accs, q_start = mfma_phase(
                        dv, dk, p_prev, ds_prev, q_tile, rm_cur, rm_fill, t_cur,
                        t_prev, kt_packs,
                    )  # fmt: skip
                    phase_barrier()
                    p_prev, ds_prev = gate_phase(s_accs, da_accs, q_start)
                    if const_expr(u < UNROLL - 1):
                        phase_barrier()

                loop_results = yield (dv + dk + p_prev + ds_prev)

            # Closes the last iteration's gate phase, which the rotation left
            # without a trailing barrier inside the loop.
            phase_barrier()

            # No drain: every tile's partial is flushed by g0's gate phase in
            # the same trip that produced it, and the last gate phase is inside
            # the loop.

            if not is_g0:
                phase_barrier()

            results = (
                list(loop_results)
                if isinstance(loop_results, (list, tuple))
                else [loop_results]
            )
            with arith.fastmath(arith.FastMathFlags.fast):
                for og in range_constexpr(KV_OWNED_SUBTILES):
                    kv_row_base = (
                        kv_wave_base
                        + fx.Int32(og * MFMA_M)
                        + lane_div_16 * fx.Int32(MFMA_LANE_K)
                    )
                    for e in range_constexpr(MFMA_ELEMS_PER_LANE):
                        kv_row_e = kv_row_base + fx.Int32(e)
                        if kv_row_e < seq_len:
                            for c in range_constexpr(D_CHUNKS):
                                ov = results[c * KV_OWNED_SUBTILES + og]
                                col = fx.Int32(c * MFMA_M) + lane_mod_16
                                val = (Vec(ov)[e] * c_inv_n).to(elem_dtype)
                                out_dv[
                                    fx.Int64(seq_start + kv_row_e), head_idx, col
                                ] = val
                            for c in range_constexpr(HC_CHUNKS):
                                ov = results[N_ACC_DV + c * KV_OWNED_SUBTILES + og]
                                col = fx.Int32(c * MFMA_M) + lane_mod_16
                                val = (Vec(ov)[e] * c_dk_scale).to(elem_dtype)
                                out_dk[
                                    fx.Int64(seq_start + kv_row_e), head_idx, col
                                ] = val

    @flyc.jit
    def launch_hstu_attention_bwd_fused_pp(
        batch: fx.Int32,
        q: fx.Tensor,
        k: fx.Tensor,
        v: fx.Tensor,
        do: fx.Tensor,
        seq_offsets: fx.Tensor,
        num_targets: fx.Tensor,
        perm: fx.Tensor,
        out_dv: fx.Tensor,
        out_dk: fx.Tensor,
        dq_part: fx.Tensor,
        part_off: fx.Tensor,
        kv_tile_lo: fx.Int32,
        kv_tile_hi: fx.Int32,
        stream: fx.Stream,
    ) -> None:
        c_ngg = fx.Int32(NUM_GRID_GROUPS)
        hz_total = batch * fx.Int32(num_heads)
        hz_per_group = (hz_total + fx.Int32(NUM_GRID_GROUPS - 1)) // c_ngg
        grid = (kv_tile_hi - kv_tile_lo) * hz_per_group * c_ngg
        hstu_attention_bwd_fused_pp(
            q,
            k,
            v,
            do,
            seq_offsets,
            num_targets,
            perm,
            out_dv,
            out_dk,
            dq_part,
            part_off,
            hz_per_group,
            hz_total,
            kv_tile_lo,
            kv_tile_hi,
            value_attrs={
                "passthrough": [
                    ["denormal-fp-math-f32", "preserve-sign,preserve-sign"],
                    ["no-nans-fp-math", "true"],
                    ["unsafe-fp-math", "true"],
                ],
                "rocdl.waves_per_eu": WAVES_PER_EU,
                "rocdl.flat_work_group_size": f"{BLOCK_THREADS},{BLOCK_THREADS}",
            },
        ).launch(grid=grid, block=BLOCK_THREADS, smem=0, stream=stream)

    return launch_hstu_attention_bwd_fused_pp
