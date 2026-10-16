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
`(256, 16, 8)` tile, and raises for anything else.

The second half of this module is the host side of the same contract: the
split-K reduction kernel that sums the dQ partials, and the scratch-offset and
kv-chunking tables that both kernels index. They live here rather than in a
module of their own because the compact scratch layout is one contract, written
by the kernel above and read by the kernel below.
"""

import functools
import math as host_math

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
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
    swz_col,
)


def phase_barrier():
    """Workgroup barrier that the machine scheduler may not move code across.

    Separates this kernel's MFMA and VALU phases. The ``sched_barrier`` fences
    are load-bearing: ``s_barrier`` alone does not stop pure VALU sinking, so
    part of the gate migrates into the next MFMA phase and the two phases stop
    being disjoint. Deliberately not ``gpu.barrier()``, whose memory fence adds
    an ``s_waitcnt lgkmcnt(0)`` -- this kernel places its waits by hand.
    """
    rocdl.sched_barrier(0)
    rocdl.s_barrier()
    rocdl.sched_barrier(0)


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
#
# But NOT as large as it will go. At W = NUM_OPERAND_GROUPS - 1 the priming
# loop issues every read but one before the burst, and they queue on the LDS
# pipe -- 8 cycles of bank time each, times eight waves sharing the unit, with
# no MFMA between them to absorb it. Lower W scatters them INTO the burst.
# Swept on the benchmark shape (fused kernel, median of 3):
#
#     W = 15   360.2 ms        W = 6    351.4 ms   <- best
#     W = 11   354.0 ms        W = 4    353.1 ms
#     W =  8   352.0 ms
#
# The curve is flat between 6 and 8 and turns back up by 4, where the cover
# per read gets too short. An earlier sweep of this constant looked only at
# VGPR (flat at every value) and concluded it was not a lever; it is one for
# TIME.
PREFETCH_W = 6

# Compile hints for this kernel's binary, applied through `flyc.compile[...]`
# so FlyDSL scopes the LLVM cl::opt around this pipeline run only.
#
# SLP vectorization is what forms the `<2 x float>` ops the backend lowers to
# `v_pk_*_f32`. A packed op is a JOIN -- it cannot issue until BOTH halves'
# producers retire -- and this kernel's gate is latency-bound, so packing halves
# the independent chains the scheduler can interleave.
#
# Measured at B=1024/H=4/N=16384/d=64 targets=300, fused kernel only, with the
# split arm resident so the memory pressure matches the benchmark:
#
#     SLP on (default)        375.7 / 378.3 ms
#     slp-max-reg-size=32     378.2 ms        (capping width does not help)
#     vectorize-slp=false     342.9 / 340.3 / 342.8 ms   <- this
#
# 340 ms also matches what hand-pinned inline asm achieved (339.3) before the
# gate was rewritten in plain arithmetic, so the knob buys back the whole
# difference without any instruction pinning.
#
# NOTE: this must be measured by LATENCY, not by reading the dumped ISA. The
# pinned and plain builds have identical opcode histograms -- under this
# kernel's `unsafe-fp-math` the backend already selects `v_mul_legacy_f32` for
# a plain `fmul` -- and differ only in scheduling.
_FUSED_COMPILE_HINTS = {"llvm_options": {"vectorize-slp": False}}
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
    """Hard gate for the fused kernel.

    Deliberately strict: this kernel hardcodes a single tuned shape, so anything
    outside it must fall back to the `build_hstu_attention_bwd_dvdk` +
    `build_hstu_attention_bwd_dq` pair rather than silently compile a variant
    that was never measured.
    """
    if arch is None:
        arch = get_rocm_arch()
    if not (arch or "").startswith("gfx942"):
        raise ValueError(
            f"hstu bwd fused requires gfx942, got {arch!r}. The schedule "
            "assumes 16x16x16 MFMA timing and a dword LDS DMA."
        )
    if dtype_str != "bf16":
        raise ValueError(f"hstu bwd fused requires bf16, got {dtype_str!r}")
    if not causal:
        raise ValueError("hstu_attention_bwd only supports causal attention")
    if head_dim != 64 or hidden_dim != 64:
        raise ValueError(
            "hstu bwd fused requires head_dim == hidden_dim == 64, got "
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
    if contextual_seq_len != 0:
        # Measured wrong, not merely untested: at B=32/H=4/N=1024 with
        # contextual_seq_len=64 the fused dQ misses the fp32 oracle by 0.85
        # relative against a 0.03 tolerance, with a ZEROED scratch so it is the
        # mask arithmetic, not uninitialised partials.
        raise ValueError(
            "hstu bwd fused does not support a contextual prefix, got "
            f"contextual_seq_len={contextual_seq_len}. The dvdk + dq pair does."
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
    # Group-exchange buffer. Past the tile sit BLOCK_N sink rows for g0's
    # discarded write and one permanently-zero row for g1's read, which is how
    # the exchange stays branch-free.
    #
    # A SINGLE sink row would be a 4-way bank conflict for half the workgroup:
    # the row index is then constant across lanes, so all four quarter-waves
    # address the same 64 dwords (measured 64 of 256 distinct). One sink row
    # per flush row restores g1's spread -- consecutive rows are DQ_PAD apart,
    # which is 4 banks.
    # One row past the tile: a permanently-zero source for g1's read. The 16
    # SINK rows are gone -- g0 no longer writes a dead row, it simply does not
    # write, now that the two tails are a uniform `if is_g0:` branch.
    DQG_ZERO = BLOCK_N
    DQG_ROWS = BLOCK_N + 1
    DQG_ELEMS = DQG_ROWS * DQ_PAD
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
                (DQG_ROWS, head_dim // DQ_PER_LANE, DQ_PER_LANE),
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

        c_inv_n = fx.Float32(1.0 / max_seq_len)
        c_neg_log2e = fx.Float32(-_LOG2E)
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
            for view in dst_views:
                wave_byte = fx.Int32(
                    fx.ptrtoint(fx.get_iter(view))
                ) + grp_wave * fx.Int32(WARP_SIZE * DMA_BYTES)
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
                lds_byte = fx.Int32(lane0[slot]) + fx.Int32(d * GRP_THREADS * DMA_BYTES)
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
            def run(rm_slot, t_slot, between=None):
                """`between` is emitted after the gather and before the store.

                It is a callback and not just a statement ordering because this
                runs inside a RUNTIME `if is_g0:` branch, and the frontend will
                not let values defined in a branch escape it.
                """
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
                if between is not None:
                    between()
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
                    halves[key] = Vec(
                        views[t_slot][lane_div_16, lane_mod_16, h, None].load()
                    ).bitcast(fx.Int32)
                v = halves[key]
                words = [v[sub * PACK_DWORDS + i] for i in range_constexpr(PACK_DWORDS)]
                return Vec.from_elements(words, fx.Int32).bitcast(elem_dtype).ir_value()
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

        # ---- gate arithmetic ------------------------------------------------
        # Plain operators, deliberately. These six were inline `v_mul_f32` /
        # `v_add_f32` / `v_fma_f32` asm blocks plus `llvm.amdgcn.fmul.legacy`
        # and `llvm.amdgcn.rcp.f32`, pinned to keep the gate off the packed
        # datapath -- the gate is latency-bound and a packed op is a JOIN, so
        # it cannot issue until BOTH halves' producers retire.
        #
        # An ATT capture of the executed code object says that rationale no
        # longer holds for THIS kernel. Pinned and plain compile to the same
        # opcode mix -- 48 `v_mul_legacy_f32`, 16 `v_pk_mul_f32`, 16
        # `v_fma_f32`, 114 `s_nop` in both -- because under the kernel's
        # `unsafe-fp-math` attribute the backend already selects legacy
        # multiply for a plain `fmul`. The code objects differ only in
        # scheduling and register assignment.
        #
        # That residual is worth 0.85% over the ten benchmark shapes (+2.2% at
        # the worst, B=1024/H=4 causal), measured. Traded away on purpose: six
        # escapes into MLIR internals -- including a function-local
        # `from flydsl._mlir import ir` and an `s_nop` hand-placed because the
        # hazard recognizer cannot see into `INLINEASM` -- are a standing
        # maintenance cost against a rationale the hardware no longer agrees
        # with. Kept as named seams so re-pinning one is a local change.
        def fmul_scalar(a, b):
            return fx.Float32(a) * fx.Float32(b)

        def fmul_const_scalar(a, b):
            """`a * b` for a loop-invariant constant `b`."""
            return fx.Float32(a) * fx.Float32(b)

        def addone_scalar(x):
            """`1 + x`, where `x` is always a `v_exp_f32` result.

            The pinned form needed an explicit `s_nop 0` here: a VALU trans op
            requires one software wait state before a consumer reads it, and
            the compiler inserts that automatically EXCEPT into inline asm,
            which its hazard recognizer treats as opaque. Reading the stale
            register showed up as dv/dk max|err|/peak 0.33 on ~0.3% of elements
            under the causal mask while the hstu mask still passed. Plain
            arithmetic hands that duty back to the compiler.
            """
            return fx.Float32(1.0) + fx.Float32(x)

        def fadd_scalar(a, b):
            return fx.Float32(a) + fx.Float32(b)

        def fma_scalar(a, b, c):
            """`a*b + c`; contracts to `v_fma_f32`."""
            return fx.Float32(a) * fx.Float32(b) + fx.Float32(c)

        def rcp_scalar(x):
            """Reciprocal. `unsafe-fp-math` lowers this to `v_rcp_f32`."""
            return fx.Float32(1.0) / fx.Float32(x)

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
        dq_poff = part_off[fx.Int64(batch_idx * fx.Int32(num_kv_tiles) + kv_tile_idx)]
        # 256 lanes per group, 4 contiguous outputs each.
        dq_gtid = (wave_id % fx.Int32(NUM_WAVES // 2)) * fx.Int32(WARP_SIZE) + _lane
        dq_frow = dq_gtid // fx.Int32(DQ_LANES_PER_ROW)
        dq_fvec = dq_gtid % fx.Int32(DQ_LANES_PER_ROW)
        dq_fcol = dq_fvec * fx.Int32(DQ_PER_LANE)
        dq_wbase = is_g0.select(fx.Int32(0), fx.Int32(NUM_WAVES // 2))
        # g1 writes its group sum to the row g0 will read; g0 writes to the
        # sink row and reads the real one. g1 reads the permanently-zero row,
        # so it adds nothing. No branch, and neither group idles.
        dq_rrow = is_g0.select(dq_frow, fx.Int32(DQG_ZERO))

        def dq_flush_issue():
            """Issue the five flush reads and nothing else.

            Called at the TOP of phase V and consumed at the BOTTOM, so the
            whole SiLU/SiLU' chain -- ~10 deep per element with two
            quarter-rate transcendentals -- sits between each read and its
            `s_waitcnt`. Nothing in the gate touches these registers, so it is
            free cover; the cost is 20 VGPR live across the phase.
            """
            return (
                [
                    Vec(dqw[dq_wbase + fx.Int32(w), dq_frow, dq_fvec, None].load())
                    for w in range_constexpr(NUM_WAVES // 2)
                ],
                Vec(dqg[dq_rrow, dq_fvec, None].load()),
            )

        def dq_absorb(acc, v):
            """Fold one staged slot into the running sum.

            Called BETWEEN gate stages, one slot at a time, so the backend
            puts each `s_waitcnt lgkmcnt(N)` at a different point in the chain
            instead of clustering all five where the reduction used to sit.
            It also lets each slot's 4 VGPRs die as it is consumed, instead of
            all 20 living to the end of the phase.
            """
            return [fadd_scalar(acc[j], v[j]) for j in range_constexpr(DQ_PER_LANE)]

        def dq_reduce_flush(q_tile_v, _issued):
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
            # The four per-wave slots were already folded in by `dq_absorb`,
            # interleaved with the gate. Only the group exchange is left.
            #
            # That fifth read was issued with the other four and depends on
            # nothing computed here -- the other group published its row a full
            # barrier ago -- and the two rows are disjoint, so it is safe ahead
            # of the write below. g1 writes `dq_frow` and reads the
            # permanently-zero row; g0 writes a sink row and reads the one g1
            # published last interval. Neither group reads what it wrote.
            acc, _g = _issued
            # The own-group sum is kept aside so the exchange write can be
            # issued LAST, after the HBM store -- as far from the five reads as
            # this phase allows. Next to them it stalled 3.06 M cycles (3.8% of
            # all stall, 142/exec) with no dependency to explain it: its inputs
            # are two VALU adds old. That is LDS-pipe contention, both with its
            # own reads and with the partner group's phase M, so distance is
            # the only lever. Costs 4 VGPR to hold the pre-exchange sum.
            # The two tails are now an `if is_g0:` branch, not two writes to
            # dead destinations. `is_g0` is wave-UNIFORM, so this is a scalar
            # branch, and the kernel already branches on it for transpose_q vs
            # transpose_do.
            #
            # What this removes: g1's HBM store to the trash row, and g0's
            # `dqg` write to a sink row. The 16 sink rows go with them.
            #
            # The HBM store has to be `fx.generic_store`, not
            # `dq_part[...] = v`. A subscript ASSIGNMENT inside a dynamic `if`
            # makes the AST rewriter treat `dq_part` as branch-assigned and
            # demand an scf.if result for it; a call has no assignment target.
            # That restriction is what forced the trash rows originally.
            #
            # The `dqg` READ stays unconditional at the head of the phase: g1
            # reads the permanently-zero row and adds nothing. Moving it in
            # here would cost it the gate's latency cover, which is worth far
            # more than the one read.
            _own = list(acc)
            _qa = (q_tile_v - fx.Int32(1)) * fx.Int32(BLOCK_N) + dq_frow
            _r = fx.Int64(dq_poff) + fx.Int64(_qa - kv_start_row)
            _r = (_qa < seq_len).select(_r, fx.Int64(0))
            _r = (_qa >= kv_start_row).select(_r, fx.Int64(0))
            _hc = head_idx * fx.Int32(head_dim) + dq_fcol
            if is_g0:
                for j in range_constexpr(DQ_PER_LANE):
                    fx.generic_store(
                        fx.get_iter(dq_part)
                        + _r * fx.Int64(num_heads * head_dim)
                        + fx.Int64(_hc + fx.Int32(j)),
                        fmul_const_scalar(
                            fadd_scalar(_own[j], _g[j]), c_inv_n
                        ),
                    )  # fmt: skip
            else:
                dqg[dq_frow, dq_fvec, None].store(Vec.from_elements(_own, fx.Float32))

        # ---- phase M -------------------------------------------------------
        def mfma_phase(
            dv, dk, p_prev, ds_prev, q_tile, rm_cur, rm_fill, t_cur, t_prev, kt_packs
        ):
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
            ds_t = [
                transpose_frag(ds_prev[og]) for og in range_constexpr(KV_OWNED_SUBTILES)
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
            # The dQ GEMM runs BEFORE the q_t/do_t transpose build, not after.
            #
            # That build reads an LDS tile `buffer_load ... lds` fills, and the
            # backend cannot tell which bytes the outstanding DMAs will touch,
            # so it guards the read with `vmcnt(0)`. The trace charged 2.57 M
            # cycles to that one wait (3.1% of stall, 61/exec) -- the largest
            # non-barrier item inside phase M. Two things follow:
            #
            #  - the eight dQ MFMAs now run BEFORE it, giving this phase's DMA
            #    ~128 more cycles to land before anything has to wait on it;
            #  - the forced `vmcnt(0)` ends up adjacent to the phase's own
            #    closing `vmcnt(0)`, so the second one is free instead of
            #    being a second full drain.
            #
            # Nothing about the transpose's placement rules changes: it is
            # still inside phase M (phase V would be an unsynchronised
            # cross-group RAW on the tile), still after every operand read has
            # issued, and the slot lifetimes are untouched -- write t_cur, read
            # t_prev, both within this phase.
            def _dq_stage_chunk(ks):
                # q = l%16, d = ks*MFMA_M + MFMA_LANE_K*(l/16) + e, so the four
                # elements are one contiguous run of the
                # (.., BLOCK_N, head_dim/VEC, VEC) view.
                dqw[
                    wave_id,
                    lane_mod_16,
                    fx.Int32(ks * MFMA_M // DQ_PER_LANE) + lane_div_16,
                    None,
                ].store(Vec(dq_acc[ks]))  # fmt: skip

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
                # Stage the PREVIOUS chunk here, interleaved with the GEMM
                # rather than four-in-a-row afterwards. Its accumulator
                # chain closed KV_OWNED_SUBTILES MFMAs ago, which is the
                # hazard cover an MFMA result needs before a non-MFMA
                # consumer, and spreading the writes keeps four b128 off
                # the LDS pipe back to back. The LAST chunk has no
                # following MFMA, so it stays in the transpose build.
                if const_expr(ks >= 1):
                    _dq_stage_chunk(ks - 1)

            # Source order alone does NOT produce the interleave: the
            # scheduler is free to sink all three stores together, and the
            # trace showed it doing exactly that -- nine MFMAs, then three
            # `ds_write_b128` back to back, then the last MFMA.
            #
            # `sched_group_barrier` states the pattern the scheduler must
            # form instead: KV_OWNED_SUBTILES MFMAs, one DS write, repeat.
            # Unlike `sched_barrier(0)` this constrains only the shape of
            # the groups, so the scheduler keeps its freedom WITHIN each
            # one and across the rest of the phase.
            for ks in range_constexpr(K_STEPS):
                rocdl.sched_mfma(KV_OWNED_SUBTILES)
                if const_expr(ks >= 1):
                    rocdl.sched_dswr(1)

            rocdl.sched_barrier(0)

            # The dQ staging writes are emitted INSIDE the transpose build,
            # between its gather and its store.
            #
            # They read dQ accumulators directly, and an MFMA result costs a
            # non-MFMA consumer wait states -- the hazard recogniser emits
            # `s_nop 3`/`s_nop 6` and the writes absorb the rest. The four
            # `ds_read_u16` ahead of them cover that. Sitting AFTER the whole
            # transpose instead, they landed immediately before the phase's
            # closing `lgkmcnt(0)` with nothing to cover their own retirement;
            # here the transpose's own wait and store do it.
            #
            # Neither side has a bank conflict to fix: both are 64-lane b128,
            # 256 dwords over 32 banks, and both measure exactly 8 accesses on
            # the busiest bank -- the floor. T_GPAD and DQ_PAD=68 are what buy
            # that; removing either doubles the busiest bank to 16.
            def _dq_stage():
                # Only the last chunk is left: the other three went out
                # interleaved with the dQ GEMM above. This one has no following
                # MFMA to cover it, so it keeps the transpose's four
                # `ds_read_u16` as its hazard cover.
                _dq_stage_chunk(K_STEPS - 1)

            rocdl.sched_barrier(0)
            if is_g0:
                transpose_q(rm_cur, t_cur, _dq_stage)
            else:
                transpose_do(rm_cur, t_cur, _dq_stage)
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
            # PINNED at the very head of phase V by the `sched_barrier`
            # below. Source placement alone does not hold: the scheduler sinks
            # these loads toward their use to shorten the live range of the 20
            # VGPRs they fill, and it did so from every position tried --
            # measured at 54 VALU ops deep even when written right after the
            # exp2 batch. The fence is the only way to choose the issue cycle.
            _dq_vs, _dq_g = dq_flush_issue()
            _dq_acc = [fx.Float32(0.0) for _ in range_constexpr(DQ_PER_LANE)]
            rocdl.sched_barrier(0)

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
            keep = [
                keep_masks(q_start, og) for og in range_constexpr(KV_OWNED_SUBTILES)
            ]
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
                # Both derived straight from `s`, not `emu` from `sc`: that
                # serialised two multiplies at the head of a chain whose next
                # link is a 16-cycle transcendental. Same op count, one less
                # link, and the two multiplies now issue in parallel.
                # `sc` arrives already scaled from GEMM0 (alpha is folded into
                # k_packs), so only the exp2 argument needs a multiply.
                sc = s
                emu = [exp2_f32(fmul_const_scalar(sc[j], c_neg_log2e)) for j in rng]
                _dq_acc = dq_absorb(_dq_acc, _dq_vs[0])
                sig = [rcp_scalar(addone_scalar(emu[j])) for j in rng]
                _dq_acc = dq_absorb(_dq_acc, _dq_vs[1])
                silu = [fmul_scalar(sc[j], sig[j]) for j in rng]
                _dq_acc = dq_absorb(_dq_acc, _dq_vs[2])
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
                _dq_acc = dq_absorb(_dq_acc, _dq_vs[3])
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
            # Consumed LAST, so the gate above is the reads' cover.
            dq_reduce_flush(q_start // fx.Int32(BLOCK_N), (_dq_acc, _dq_g))
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
            for _z in range_constexpr((DQG_ELEMS + BLOCK_THREADS - 1) // BLOCK_THREADS):
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
                ds_prev = [it_list[N_ACC + N_PACK + i] for i in range_constexpr(N_PACK)]
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

    return flyc.compile[_FUSED_COMPILE_HINTS](launch_hstu_attention_bwd_fused_pp)


# --------------------------------------------------------------------------- #
# Split-K reduction for the dQ partials
#
# The kernel above produces dQ as a PARTIAL per KV tile, because a workgroup
# there owns a KV range and `dQ[q][d] = sum_kv dS[q][kv].K[kv][d]` reduces over
# kv. This half sums those partials.
#
# Split-K rather than atomics: `global_atomic_add_f32` is a hard 0.40 TB/s wall
# on gfx942 while `global_store_dwordx2` runs at 3.56 TB/s -- 8.8x, measured,
# and independent of contention, atomic width and instruction count. The atomic
# design needed 337 GB of traffic, about twice the entire backward pass.
#
# Scratch layout, built by `build_partial_offsets` below and compact by
# necessity -- the dense `[max_tiles][L][H][D]` alternative is 331 GB at the
# production shape:
#
#     offset[b][t]  prefix sum over (b, t) of (S_b - t*BLOCK_M)   -- in ROWS
#     dq_part[offset[b][t] + (q_row - t*BLOCK_M)][h * D + d]
#
# `dq_part` is 2-D rather than a flat element array only because the launch ABI
# packs dynamic extents as i32 and the per-pass scratch is 21e9 fp32 at the
# production shape. The bytes are identical.
#
# kv tile `t` only serves q rows `[t*BLOCK_M, S_b)` -- under the causal mask
# nothing above the diagonal contributes -- so q row `r` sums tiles
# `0 ..= r // BLOCK_M`. All `ROWS_PER_BLOCK` rows of a workgroup share that
# count because `BLOCK_M % ROWS_PER_BLOCK == 0`, which is what lets one runtime
# loop cover the whole workgroup.
#
# The scale is already applied by the fused kernel's flush, so this kernel only
# sums and converts.
# --------------------------------------------------------------------------- #

# 16 q rows per workgroup, 16 fp32 accumulators per thread.
ROWS_PER_BLOCK = 16
# Elements per thread per access. A thread-per-(h, d) mapping is coalesced but
# issues a separate `global_load_dword` for every row, because consecutive rows
# are H*D elements apart and nothing fuses. Giving each thread VEC CONSECUTIVE
# columns instead lets the backend widen them to dwordx4 (and the bf16 stores
# to dwordx2), which is 4x fewer memory instructions for exactly the same
# bytes. VEC must divide head_dim so a thread never straddles two heads.
VEC = 4
# One reserved row at the head of the scratch. The fused kernel sends q rows
# beyond seq_len there instead of branching around the store.
TRASH_ROWS = 1


def validate_hstu_attention_bwd_dq_reduce(num_heads, head_dim, block_m, dtype_str):
    """Gate the supported shape, as the fused kernel's own validator does."""
    arch = get_rocm_arch()
    if arch != "gfx942":
        raise ValueError(f"hstu bwd dq_reduce is specialized for gfx942, got {arch}")
    if dtype_str != "bf16":
        raise ValueError(f"hstu bwd dq_reduce is specialized for bf16, got {dtype_str}")
    if block_m % ROWS_PER_BLOCK != 0:
        raise ValueError(
            f"block_m ({block_m}) must be a multiple of ROWS_PER_BLOCK "
            f"({ROWS_PER_BLOCK}): a workgroup's rows share one tile count."
        )
    if num_heads * head_dim % WARP_SIZE != 0:
        raise ValueError(
            f"num_heads*head_dim ({num_heads * head_dim}) must be a multiple of "
            f"the wave size ({WARP_SIZE})"
        )
    if head_dim % VEC != 0:
        raise ValueError(
            f"head_dim ({head_dim}) must be a multiple of VEC ({VEC}) so a "
            f"thread's run of columns stays inside one head"
        )
    hd = num_heads * head_dim
    row_group = max(hd // (hd // VEC), 1)
    if ROWS_PER_BLOCK % row_group != 0:
        raise ValueError(
            f"ROWS_PER_BLOCK ({ROWS_PER_BLOCK}) must be a multiple of the "
            f"{row_group} rows a workgroup spans at VEC={VEC}"
        )


@functools.cache
def build_hstu_attention_bwd_dq_reduce(
    num_heads,
    head_dim,
    block_m,
    max_tiles,
    dtype_str,
    *,
    max_attn_len=0,
    has_targets=False,
    first_pass=True,
    last_pass=True,
):
    """Sum the dQ partials of kv tiles [tile_lo, tile_hi) into dQ.

    `first_pass`/`last_pass` support kv-chunking (G > 1), where the fused kernel
    is run several times over disjoint KV tile groups into the same reused
    scratch: every pass but the first accumulates into an fp32 carry, and only
    the last converts to bf16.
    """
    validate_hstu_attention_bwd_dq_reduce(num_heads, head_dim, block_m, dtype_str)
    elem_dtype = _dtype_to_elem_type(dtype_str)
    HD = num_heads * head_dim
    ROWS = ROWS_PER_BLOCK
    # Threads tile the (row, column-run) grid: LANES_PER_ROW threads cover one
    # row's VEC-element runs, so the workgroup spans ROW_GROUP rows at a time
    # and sweeps its block in SLICES steps. THREADS stays HD -- making it
    # HD//VEC would be a single wave per workgroup carrying ROWS*VEC
    # accumulators.
    THREADS = HD
    LANES_PER_ROW = HD // VEC
    ROW_GROUP = max(THREADS // LANES_PER_ROW, 1)
    SLICES = ROWS // ROW_GROUP

    @flyc.kernel(known_block_size=[THREADS, 1, 1])
    def hstu_attention_bwd_dq_reduce(
        dq_part: fx.Tensor,
        part_off: fx.Tensor,
        red_map: fx.Tensor,
        seq_offsets: fx.Tensor,
        num_targets: fx.Tensor,
        dq_acc: fx.Tensor,
        out_dq: fx.Tensor,
        tile_lo: fx.Int32,
        tile_hi: fx.Int32,
    ) -> None:
        tid = fx.Int32(gpu.thread_id("x"))
        bid = fx.Int32(gpu.block_id("x"))

        # (batch, row block) comes from a host table rather than
        # `bid // max_blocks`. The sequences are jagged -- mean length is about
        # half of max at the production shape -- so a rectangular grid would
        # spend roughly half its workgroups re-reducing rows that do not exist.
        b = red_map[fx.Int64(bid) * fx.Int64(2)]
        blk = red_map[fx.Int64(bid) * fx.Int64(2) + fx.Int64(1)]

        seq_start = fx.Int32(seq_offsets[fx.Int32(b)])
        seq_len = fx.Int32(seq_offsets[fx.Int32(b) + fx.Int32(1)]) - seq_start
        row0 = fx.Int32(blk) * fx.Int32(ROWS)
        # Target rows have their position id CLAMPED to max_id, so they attend
        # to kv near max_id however far down the sequence they sit. Only read
        # num_targets when a window makes the distinction matter.
        max_id = seq_len
        if const_expr(max_attn_len > 0 and has_targets):
            _nt = fx.Int32(num_targets[fx.Int32(b)])
            max_id = (_nt > fx.Int32(0)).select(seq_len - _nt, seq_len)

        row_sub = tid // fx.Int32(LANES_PER_ROW)
        col = (tid % fx.Int32(LANES_PER_ROW)) * fx.Int32(VEC)
        h = col // fx.Int32(head_dim)
        d0 = col % fx.Int32(head_dim)

        # The tail rows of the last block are clamped onto the last real row,
        # so they read (and later rewrite) a valid element with the value that
        # row would get anyway. Bounding the WRITE instead would need a dynamic
        # `if` around a tensor subscript store, which makes the AST rewriter
        # demand an scf.if result for the output tensor.
        rr = []
        for s in range_constexpr(SLICES):
            _r = row0 + fx.Int32(s * ROW_GROUP) + row_sub
            rr.append((_r < seq_len).select(_r, seq_len - fx.Int32(1)))

        # Every row of this workgroup lies in the same kv tile, so one count
        # serves all of them: BLOCK_M is a multiple of ROWS.
        n_t = row0 // fx.Int32(block_m) + fx.Int32(1)
        n_t = (n_t < tile_hi).select(n_t, tile_hi)

        # Under a sliding window a q row is served by a CONTIGUOUS span of kv
        # tiles, not by every tile below it, and the fused kernel writes a
        # partial only for the tiles it serves. Both bounds matter: this one
        # skips tiles that contribute nothing, and the per-row predicate below
        # keeps the sum off elements that were never written.
        #
        # `row0` is the workgroup's FIRST row, so this bound is the loosest any
        # of its rows needs -- correctness comes from the predicate, this is
        # only work avoided. Truncating division is fine: the clamp at 0 covers
        # the negative case either way.
        t_start = fx.Int32(tile_lo)
        if const_expr(max_attn_len > 0):
            # Anchor the span on the CLAMPED id, not the raw row. A target row
            # has id == max_id, so it is served from (max_id - window) upward --
            # still one contiguous span, just anchored lower down than the row
            # itself. Taking the raw row would skip the tiles it actually needs;
            # giving up and starting at tile_lo (as this did) reads the whole
            # causal span and throws most of it away, which cost 50.4 ms of
            # reduction against 12.6 for the same window without targets.
            _idr = row0
            if const_expr(has_targets):
                _idr = (row0 < max_id).select(row0, max_id)
            _wlo = (_idr - fx.Int32(max_attn_len)) // fx.Int32(block_m)
            _wlo = (_wlo > fx.Int32(0)).select(_wlo, fx.Int32(0))
            t_start = (_wlo > t_start).select(_wlo, t_start)

        acc0 = [fx.Float32(0.0) for _ in range_constexpr(SLICES * VEC)]
        if const_expr(not first_pass):
            for s in range_constexpr(SLICES):
                for j in range_constexpr(VEC):
                    acc0[s * VEC + j] = dq_acc[
                        fx.Int64(seq_start + rr[s]), col + fx.Int32(j)
                    ]

        for _t, it in range(
            fx.Int64(t_start), fx.Int64(n_t), fx.Int64(1), init=acc0
        ):  # ty: ignore
            # The loop-carried state comes back as raw ArithValue, which has
            # none of the DSL numeric methods; re-wrap before using it.
            acc = [
                fx.Float32(x)
                for x in (list(it) if isinstance(it, (list, tuple)) else [it])
            ]
            t_i = fx.Int32(_t)
            base = fx.Int64(part_off[fx.Int64(b) * fx.Int64(max_tiles) + fx.Int64(t_i)])
            row_in_slot = row0 - t_i * fx.Int32(block_m)
            nxt = []
            for s in range_constexpr(SLICES):
                _ri = row_in_slot + (rr[s] - row0)
                if const_expr(max_attn_len > 0):
                    # Written iff this tile serves the row. Measured, not
                    # assumed: NaN-poisoning the scratch found every unwritten
                    # element inside what this excludes (3,826 unwritten, all
                    # within the 4,034 excluded). A `select`, not a multiply --
                    # an unwritten slot may hold a NaN and 0 * NaN is NaN.
                    _hi = (t_i + fx.Int32(1)) * fx.Int32(block_m) + fx.Int32(
                        max_attn_len
                    )
                    _served = rr[s] < _hi
                    if const_expr(has_targets):
                        # Mirrors the fused kernel's `q_upper` reopening: once
                        # a tile's window reaches past max_id it serves EVERY
                        # row, because the clamped target rows land there.
                        _served = _served | (_hi > max_id)
                for j in range_constexpr(VEC):
                    _val = dq_part[base + fx.Int64(_ri), col + fx.Int32(j)]
                    if const_expr(max_attn_len > 0):
                        _val = _served.select(_val, fx.Float32(0.0))
                    nxt.append(acc[s * VEC + j] + _val)
            acc = yield nxt

        res = [
            fx.Float32(x)
            for x in (list(acc) if isinstance(acc, (list, tuple)) else [acc])
        ]
        for s in range_constexpr(SLICES):
            for j in range_constexpr(VEC):
                if const_expr(last_pass):
                    out_dq[fx.Int64(seq_start + rr[s]), h, d0 + fx.Int32(j)] = res[
                        s * VEC + j
                    ].to(elem_dtype)
                else:
                    dq_acc[fx.Int64(seq_start + rr[s]), col + fx.Int32(j)] = res[
                        s * VEC + j
                    ]

    @flyc.jit
    def launch_hstu_attention_bwd_dq_reduce(
        n_blocks: fx.Int32,
        dq_part: fx.Tensor,
        part_off: fx.Tensor,
        red_map: fx.Tensor,
        seq_offsets: fx.Tensor,
        num_targets: fx.Tensor,
        dq_acc: fx.Tensor,
        out_dq: fx.Tensor,
        tile_lo: fx.Int32,
        tile_hi: fx.Int32,
        stream: fx.Stream,
    ) -> None:
        hstu_attention_bwd_dq_reduce(
            dq_part,
            part_off,
            red_map,
            seq_offsets,
            num_targets,
            dq_acc,
            out_dq,
            tile_lo,
            tile_hi,
        ).launch(grid=(n_blocks,), block=(THREADS,), smem=0, stream=stream)

    return launch_hstu_attention_bwd_dq_reduce


# --------------------------------------------------------------------------- #
# Host-side scratch tables
#
# All three are pure functions of the sequence lengths and the tile geometry,
# and all three are built on the input's own device: these tables are indexed
# by the kernels above, so a CPU round-trip would cost a sync per call for no
# benefit. The two unavoidable syncs are the scratch element count and the
# reduction's grid size, both of which the host needs in order to allocate.
# --------------------------------------------------------------------------- #


def build_partial_offsets(
    seq_offsets, block_m, *, max_seq_len, tile_lo=0, tile_hi=None
):
    """Compact scratch offsets, one slot per (batch, kv_tile).

    kv tile `t` of a sequence of length S serves q rows [t*block_m, S), so its
    slot is `S - t*block_m` rows. A DENSE [max_tiles][L][H][D] layout is 331 GB
    at the production shape and does not fit -- the compactness is load-bearing.

    Returns `(offsets, total_rows)` with `offsets` a flat int64 table indexed
    `b * max_tiles + t`, and -1 where tile t does not exist for sequence b (so
    an out-of-range read is a visible fault, not silent corruption).

    Only the kv tiles of THIS pass get a slot; the rest are -1. The table keeps
    the full `max_tiles` stride because the kernel indexes it with its own
    `num_kv_tiles`, but the offsets PACK from the start of the buffer so each
    pass reuses the same scratch.
    """
    device = seq_offsets.device
    lengths = (seq_offsets[1:] - seq_offsets[:-1]).to(torch.int64)
    # Stride is ceil(max_seq_len/block_m), NOT the max over actual lengths: the
    # kernel indexes this table with its own `num_kv_tiles`, which is derived
    # from max_seq_len. Deriving it from the realised lengths instead makes the
    # two disagree whenever no sequence reaches max_seq_len, and every batch
    # beyond the first reads the wrong slot.
    max_tiles = (max_seq_len + block_m - 1) // block_m
    t_idx = torch.arange(max_tiles, dtype=torch.int64, device=device)
    rows = (lengths[:, None] - t_idx[None, :] * block_m).clamp(min=0)

    if tile_hi is None:
        tile_hi = max_tiles
    in_pass = (t_idx >= tile_lo) & (t_idx < tile_hi)
    flat = torch.where(in_pass[None, :], rows, torch.zeros_like(rows)).reshape(-1)

    off = torch.zeros_like(flat)
    off[1:] = torch.cumsum(flat, 0)[:-1]
    # Every offset is shifted past the reserved trash row at the head.
    off = off + TRASH_ROWS
    off = torch.where(flat > 0, off, torch.full_like(off, -1))
    return off, int(flat.sum()) + TRASH_ROWS


def tile_partial_counts(seq_offsets, block_m, max_seq_len):
    """Partial ROWS each kv tile contributes, as host ints.

    Costs ONE device sync, and is the only thing the kv-chunking search needs:
    a span's scratch is just the sum of its tiles' counts. Computing this once
    lets every candidate G be priced in host arithmetic, instead of building a
    full offset table per candidate to discover it does not fit.
    """
    device = seq_offsets.device
    lengths = (seq_offsets[1:] - seq_offsets[:-1]).to(torch.int64)
    max_tiles = (max_seq_len + block_m - 1) // block_m
    t_idx = torch.arange(max_tiles, dtype=torch.int64, device=device)
    counts = (lengths[:, None] - t_idx[None, :] * block_m).clamp(min=0).sum(0)
    return counts.tolist()


def balanced_tile_groups(counts, groups):
    """Split the kv tiles into `groups` spans of roughly equal PARTIAL COUNT.

    kv tile t serves `S_b - t*block_m` q rows, so low tiles produce far more
    partials than high ones. Splitting the tile INDEX range evenly overflows
    group 0's scratch while the last group sits half empty; at the production
    shape the equal-count split puts only about the first 29% of tiles in
    group 0.

    `counts` comes from `tile_partial_counts`; this is pure host arithmetic.
    """
    max_tiles = len(counts)
    cum, running = [], 0
    for c in counts:
        running += c
        cum.append(running)
    total = cum[-1]

    spans, lo = [], 0
    for g in range(groups):
        if g == groups - 1:
            hi = max_tiles
        else:
            target = total * (g + 1) // groups
            hi = next((i for i, c in enumerate(cum) if c >= target), max_tiles - 1) + 1
            # Leave at least one tile for each remaining group.
            hi = max(lo + 1, min(hi, max_tiles - (groups - 1 - g)))
        spans.append((lo, hi))
        lo = hi
    return spans


def span_scratch_rows(counts, spans):
    """Rows the largest pass needs, trash row included. Host arithmetic."""
    return max(sum(counts[lo:hi]) for lo, hi in spans) + TRASH_ROWS


def build_reduce_map(seq_offsets, device):
    """Host table: one (batch, row_block) pair per REAL row block.

    Returns a flat int32 tensor of `2 * n_blocks` and `n_blocks`.
    """
    lengths = (seq_offsets[1:] - seq_offsets[:-1]).to(torch.int64)
    nb = torch.div(lengths + ROWS_PER_BLOCK - 1, ROWS_PER_BLOCK, rounding_mode="floor")
    n_blocks = int(nb.sum())
    b = torch.repeat_interleave(
        torch.arange(len(nb), dtype=torch.int64, device=nb.device), nb
    )
    # per-batch block index: a ramp restarted at each sequence boundary
    starts = torch.cumsum(nb, 0) - nb
    blk = torch.arange(
        n_blocks, dtype=torch.int64, device=nb.device
    ) - torch.repeat_interleave(starts, nb)
    m = torch.stack([b, blk], dim=1).reshape(-1).to(torch.int32)
    return m.to(device), n_blocks
