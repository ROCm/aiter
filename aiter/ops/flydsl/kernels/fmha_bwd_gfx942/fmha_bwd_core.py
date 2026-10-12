# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""FlyDSL FMHA backward for varlen THD, causal, bf16, gfx942, any head dim in [64, 192].

``d_qk`` and ``d_v`` are BUILD PARAMETERS (``build(n_split, d_qk, d_v)``), not module constants:
everything derived from them below is a Python-level value folded at trace time, so each shape
compiles its own fully specialised kernel set.  Supported range is ``64 <= d <= 192`` with
``d % 8 == 0`` -- the same head dims ``can_impl_fmha_v3_bwd`` accepts, plus the MLA-style
``d_qk=192 / d_v=128`` this file was originally written for.  See ``_geom``.

The MFMA primitive used everywhere is ``v_mfma_f32_16x16x16_bf16_1k``:

    C[16m, 16n] += A[16m, 16k] . B[16n, 16k]^T

The default (two-role) ``k_bwd`` carries TWO JOB TYPES on one grid.  Each output element is
written exactly once by exactly one workgroup -- no atomics, fully deterministic -- at the cost
of computing the score matrix once per job type.  For ``d <= 128``, non-deterministic, unsplit
launches the FUSED single-role variant below replaces it (dQ accumulated with bf16 atomics):

  dK/dV job  one workgroup per (key block of 128, sequence, head); streams query tiles of 32.
             S = Q.K^T and dP = dO.V^T contract over d, so those operands come straight out of
             memory; dV = P^T.dO and dK = dS^T.Q contract over the query index, so dO^T and Q^T
             are staged transposed in LDS while P^T / dS^T come free from the accumulators.

  dQ job     one workgroup per (query block of 128, sequence, head); streams key tiles of 32.
             Computes the scores TRANSPOSED (S^T = K.Q^T) so the dS^T fragment is, for free,
             the dS operand that dQ = dS.K needs; only K^T is staged in LDS.

  pre        ``k_delta``  D = rowsum(dO * O), fp32, laid out [H, T] like lse.

Bounds: every global tensor is read and written through a buffer tensor (``_buffer_view``)
whose descriptor carries an exact ``num_records``, so out-of-range loads return 0 and stores
steered past the end are dropped by the hardware.  The causal tails, the empty-workgroup case
and every epilogue mask rely on this instead of predication.
"""

import functools
import os

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir.dialects import llvm as _mlir_llvm
from flydsl._mlir.dialects._llvm_enum_gen import AtomicBinOp as _LLVMBinOp
from flydsl._mlir.dialects._llvm_enum_gen import AtomicOrdering as _LLVMOrd
from flydsl.expr import range_constexpr
from flydsl.expr.primitive import const_expr

_LLVM_FADD = _LLVMBinOp.fadd
_LLVM_MONOTONIC = _LLVMOrd.monotonic

DQK = 192  # default build shape: the MLA-style d_qk
DV = 128  # default build shape: the MLA-style d_v
D_MIN = 64
D_MAX = 192
D_STEP = 8
LOG2E = 1.4426950408889634

ROWS_DELTA = 32  # rows reduced per delta workgroup (2 unrolled 16-row passes)
NUM_THREADS = 512

KEYS_PER_WG = 128  # dK/dV job: keys owned by a workgroup (16 per wave)
QUERY_TILE = 32  # dK/dV job: queries streamed per iteration
LD_TOK_DKDV = QUERY_TILE + 4  # padded leading dim of the transposed LDS tiles

QUERIES_PER_WG = 128  # dQ job: queries owned by a workgroup (16 per wave)
KEY_TILE = 32  # dQ job: keys streamed per iteration
LD_TOK_DQ = KEY_TILE + 4

DELTA_THREADS = 256  # threads per k_delta workgroup
RED_THREADS = 256  # threads per split-K reduction workgroup
RED_VEC = 8  # bf16 lanes per thread (one dwordx4 per partial slab)

NUM_WAVES = NUM_THREADS // 64  # 8

# ---- FUSED SINGLE-ROLE 5-GEMM BACKWARD -------------------------------------------------------
# Row stride (in bf16 elements) of the LDS dS exchange tile the fused role uses to hand dS from
# the C-fragment orientation (row = key) to the operand orientation dQ = dS.K needs (row =
# query).  KEYS_PER_WG keys + 4 pad puts the DWORD row stride at 2 (mod 32), so the 16 lanes of
# a `ds_read_b64` row group tile all 32 banks exactly once.
LD_DS = KEYS_PER_WG + 4

# Calibrated VGPR model for `k_bwd`, fitted against real ISA dumps of THIS revision
# (d=112 -> 218, d=128 -> 222, d=160 -> 256 + 9 spill, d=192 -> 256 + 84 spill):
#
#   vgpr ~= VGPR_FIXED + 6*(n_tiles_qk + n_tiles_v)      accumulators (4) + operand frags (2)
#                      + 2*(n_qk_chunks + n_v_chunks)    the staging prefetch
#                      + 2                               lse / delta prefetch
#
# predicting 214 / 226 / 254 / 278 for those four dims.  The fused role adds one dQ accumulator
# per owned (query tile, d tile) and one register-resident K^T fragment set per owned d tile.
#
# The model is deliberately used as an UPPER bound: measured `vgpr_count` on the FUSED build
# is 125 / 152 / 168 / 190 / 190 / 236 / 232 at d = 64 / 72 / 96 / 104 / 112 / 120 / 128, all
# with `vgpr_spill_count = 0` (ISA dump, one head dim per dump), i.e. it over-predicts by
# ~20 and therefore only ever excludes a shape that would in fact have fitted.  Everything
# from d = 136 up needs TWO d tiles per wave and falls out of the budget on its own.
VGPR_FIXED = 120
VGPR_BUDGET = 256


def supported_head_dim(d: int) -> bool:
    """True for a head dim this file can build a kernel set for."""
    return D_MIN <= d <= D_MAX and d % D_STEP == 0


class _Geom:
    """Every trace-time constant that depends on the head dims.

    THREE widths per head dim, and the difference between them is the whole of the head-dim
    generalization:

      ``d``    the REAL head dim and therefore the row stride of q/k/v/dO/dq/dk/dv in memory.
               All global addressing uses it.
      ``d_p``  ``d`` rounded up to the 16-element MFMA tile.  The MFMA loops run
               ``d_p // 16`` tiles, so a ``d`` that is an odd multiple of 8 (72, 88, ...) pays
               for one half-empty tile -- against the 1.2x-1.8x the ASM backward pays at the
               same head dim, since it only ships hd64 / hd128 / hd192 code objects and zero-pads
               everything in between up to the next one.
      ``d_s``  ``d`` rounded up to 64, the LDS tile width.  The staging loops give each of the
               512 threads one 4-element column chunk and step it by 64, so the LDS tiles are
               sized in units of 64 and the padding columns ``[d, d_s)`` are simply staged as
               ZERO (their global loads are steered past ``num_records``).  That is what lets
               the MFMA loops read tiles beyond ``d`` without masking either operand, and it
               keeps the staging code, the ``+4`` row pad and the XOR swizzle byte-for-byte the
               code that was tuned at 192/128.

    The ``+4`` row pad on the natural tiles is conflict-optimal for EVERY supported head dim,
    not just 192: ``d_s`` is a multiple of 64, so the dword row stride ``d_s/2 + 2`` is always
    ``2 (mod 4)`` (what a 64-lane ``ds_read_b64`` wants) and never a multiple of 32 dwords (the
    16-way conflict the original 192-element stride hit).
    """

    __slots__ = (
        "d_qk",
        "d_v",
        "qk_p",
        "v_p",
        "qk_s",
        "v_s",
        "n_tiles_qk",
        "n_tiles_v",
        "frags_per_row_qk",
        "frags_per_row_v",
        "ld_qk",
        "ld_v",
        "n_qk_chunks",
        "n_v_chunks",
        "off_q_nat",
        "off_q_t",
        "off_do_nat",
        "off_do_t",
        "off_k_nat",
        "off_k_t",
        "off_v_nat",
        "arena",
        "arena_dkdv",
        "off_ds",
        "ld_dqs",
        "off_dqs",
        "dq_tiles_per_wave",
        "est_vgpr",
        "est_vgpr_fused",
        "fusable",
    )

    def __init__(self, d_qk: int, d_v: int):
        assert supported_head_dim(d_qk), f"unsupported d_qk {d_qk}"
        assert supported_head_dim(d_v), f"unsupported d_v {d_v}"
        self.d_qk = d_qk
        self.d_v = d_v
        self.qk_p = -(-d_qk // 16) * 16
        self.v_p = -(-d_v // 16) * 16
        self.qk_s = -(-d_qk // 64) * 64
        self.v_s = -(-d_v // 64) * 64

        self.n_tiles_qk = self.qk_p // 16  # MFMA tiles along d_qk
        self.n_tiles_v = self.v_p // 16  # MFMA tiles along d_v
        # Row strides in 4-element load tiles: one token's d_qk / d_v span as seen through the
        # 4-wide buffer views the operands are read from.  These are REAL-d strides.
        self.frags_per_row_qk = d_qk // 4
        self.frags_per_row_v = d_v // 4
        self.ld_qk = self.qk_s + 4  # row stride of the natural d_qk LDS tile
        self.ld_v = self.v_s + 4
        self.n_qk_chunks = self.qk_s // 64  # 64-column staging chunks per row
        self.n_v_chunks = self.v_s // 64

        # shared bf16 LDS arena; the two job branches are mutually exclusive so they alias
        self.off_q_nat = 0  # dK/dV: Q natural   [QUERY_TILE, ld_qk]
        # Q^T  [qk_s, LD_TOK_DKDV]
        self.off_q_t = self.off_q_nat + QUERY_TILE * self.ld_qk
        self.off_do_nat = self.off_q_t + self.qk_s * LD_TOK_DKDV  # dO natural
        # dO^T [v_s, LD_TOK_DKDV]
        self.off_do_t = self.off_do_nat + QUERY_TILE * self.ld_v
        dkdv_end = self.off_do_t + self.v_s * LD_TOK_DKDV

        self.off_k_nat = 0  # dQ: K natural   [KEY_TILE, ld_qk]
        self.off_k_t = self.off_k_nat + KEY_TILE * self.ld_qk  # K^T  [qk_s, LD_TOK_DQ]
        # V natural (never transposed)
        self.off_v_nat = self.off_k_t + self.qk_s * LD_TOK_DQ
        dq_end = self.off_v_nat + KEY_TILE * self.ld_v

        self.arena = max(dkdv_end, dq_end)
        assert (
            self.arena * 2 <= 64 * 1024
        ), f"d_qk={d_qk} / d_v={d_v} needs a {self.arena * 2} B LDS arena (> 64 KiB)"

        # ---- FUSED-ROLE FEASIBILITY, a purely COMPUTED predicate --------------------------
        # The fused role deletes job B, so its arena is job A's alone plus the dS exchange tile.
        self.arena_dkdv = dkdv_end
        self.off_ds = dkdv_end
        # dQ output tiles per streamed query tile = (QUERY_TILE/16) x n_tiles_qk, spread over
        # NUM_WAVES waves along d.  More than one d tile per wave doubles BOTH the dQ
        # accumulator set and the register-resident K^T set, which no supported head dim can
        # pay for -- so that is the first half of the gate.
        self.dq_tiles_per_wave = -(-self.n_tiles_qk // NUM_WAVES)
        nt = self.n_tiles_qk + self.n_tiles_v
        self.est_vgpr = (
            VGPR_FIXED + 6 * nt + 2 * (self.n_qk_chunks + self.n_v_chunks) + 2
        )
        self.est_vgpr_fused = (
            self.est_vgpr
            + 4 * (QUERY_TILE // 16) * self.dq_tiles_per_wave  # dQ accumulators
            + 2 * (KEYS_PER_WG // 16) * self.dq_tiles_per_wave  # K^T fragments
            + 4  # the strength-reduced atomic address
        )
        # dQ staging tile, bf16 [QUERY_TILE, ld_dqs].  Its dword row stride is
        # 8 mod 16: the pack side stores 8 rows x 8 consecutive dwords per instruction, which that
        # stride spreads over all 32 banks.  At least `qk_p` wide so a half-real last d tile
        # stores into padding.
        self.ld_dqs = 2 * (-(-(self.qk_p // 2 - 8) // 16) * 16 + 8)
        self.off_dqs = self.off_ds + QUERY_TILE * LD_DS
        fused_lds = (
            self.arena_dkdv + QUERY_TILE * (LD_DS + self.ld_dqs)
        ) * 2 + 2 * QUERY_TILE * 4
        self.fusable = (
            self.dq_tiles_per_wave == 1
            and self.est_vgpr_fused <= VGPR_BUDGET
            and fused_lds <= 64 * 1024
        )


@functools.lru_cache(maxsize=None)
def _geom(d_qk: int, d_v: int) -> _Geom:
    return _Geom(d_qk, d_v)


def fused_role_enabled(
    n_split: int,
    d_qk: int,
    d_v: int,
    deterministic: bool = False,
    fuse_split: bool = False,
) -> bool:
    """True when ``build`` will emit the FUSED single-role kernel for this shape.

    The host needs this for two things only: the grid carries dK/dV blocks ALONE, and ``dq``
    must be pre-zeroed because the fused role ACCUMULATES into it.  The fused role's dQ is an
    atomic sum across key blocks and so not bit-reproducible; ``deterministic`` rules it out.

    ``n_split > 1`` is the two-role split-K kernel UNLESS ``fuse_split`` asks for the FUSED
    split-K variant: the fused role streams chunk ``split`` of each key block's query range,
    dQ still goes straight into ``dq`` with bf16 atomics, and only the dK/dV partials go to the
    bf16 workspace that ``k_red`` sums in fixed slab order (so dk/dv stay bit-exact call to call).
    """
    return bool(
        OPT_FUSE
        and not deterministic
        and (n_split == 1 or fuse_split)
        and _geom(d_qk, d_v).fusable
    )


# ---- VALU-reduction ablation knobs -----------------------------------------------------------
# The default is the measured winner: neither `amask` nor `bpeel` wins alone, together they are
# -2.7 % on the main case.  `ascale` is a measured LOSS (+3.6 %) kept only so it stays
# re-checkable -- do not enable it.
_OPTS = {
    x
    for x in os.environ.get("AITER_FMHA_BWD_FLYDSL_OPT", "amask,bpeel,fuse").split(",")
    if x
}
# `fuse` collapses the two workgroup roles into ONE KV-outer role carrying all five GEMMs.  Drop
# it from AITER_FMHA_BWD_FLYDSL_OPT (or pass `deterministic=True`) to get the deterministic
# two-role kernel back.
OPT_FUSE = "fuse" in _OPTS
OPT_AMASK = "amask" in _OPTS  # dK/dV: one unsigned compare instead of 2 signed + and
# dK/dV: LOG2E onto the LDS lse, `scale` onto the dK epilogue
OPT_ASCALE = "ascale" in _OPTS
# dQ: peel the causal-diagonal trips out of the streaming loop
OPT_BPEEL = "bpeel" in _OPTS
# Independent accumulator chains the fused dQ contraction over KEYS_PER_WG keys is split into.
# One chain is 8 back-to-back DEPENDENT MFMAs and nothing else is in flight at that point.
DQ_CHAINS = 2


# ---- INNERMOST SCORE-BODY SPELLING (emission order / pack / C-seed) --------------------------
# Everything in this block is local to the innermost score body of the two jobs.  It never
# changes WHAT is computed, only the ORDER the instructions are emitted in, how the bf16 pack is
# spelled, and which register the dP accumulator starts in.  Each item is a separate const_expr
# constant so it can be ablated byte-for-byte.
#
# The kernel is C1 latency-bound (active 21.7 % / dependency wait 49.6 % / issue wait 28.7 %,
# MFMA pipe 36.6 % busy, LDS bank conflicts exactly 0, HBM 16 % of peak), so the lever is chain
# DEPTH and ds_read -> MFMA OVERLAP, not instruction census.
OPT_PINVMEM = True  # pin the next tile's buffer_loads where they were emitted
OPT_PACKPERM = True  # one v_perm_b32 per bf16 pair instead of and/shift/or
OPT_DFOLD = (
    False  # dK/dV: seed the dP accumulator with -delta (measured neutral-to-loss)
)
OPT_XSDP = True  # interleave the two independent score MFMA chains
# dK/dV: issue the lse / delta ds_read_b128 ABOVE the two score chains instead of below them.
# Implied by `dfold`, which needs the delta fragment as the dP accumulator's seed.
OPT_HOIST = True

# `sched_group_barrier` round robin, "<ds_reads>:<mfmas>" per job.  Empty = LLVM's own order.
# SWEPT ON THIS BASELINE, not copied from a prior wave.  Full `--full-benchmark`
# TARGET_GEOMEAN_VS_DEFAULT (baseline 1.053; the whole score-body block without the round robin
# reads 1.102):
#   job A alone (B off): 2:3 -> 1.1195 | 3:4 -> 1.1080 | 2:4 -> 1.1003 | 1:2 -> 1.0911
#                        3:3 -> 1.0601 | 2:2 -> 1.0566 | 4:4 -> 1.0622 | 6:8 -> 1.0524 | 8:8 -> 1.0454
#   job B at A = 2:3:    3:4 -> 1.1493 | 4:6 -> 1.1417 | 2:3 -> 1.1413 | 3:5 -> 1.1403
#                        4:5 -> 1.1375 | 6:8 -> 1.1309 | 1:2 -> 1.0999 | off -> 1.1207
#   job A at B = 3:4:    2:3 -> 1.1455 | 3:5 -> 1.1432 | 3:4 -> 1.1378 | 2:4 -> 1.1258
# The sharp edge is MFMA-per-group vs reads-per-group, NOT the head dim: every job-A group with
# m/r == 1 (2:2, 3:3, 4:4, 8:8) blows `near_d192_deep` up from 0.85 ms to ~1.45 ms, while every
# m/r >= 1.5 keeps it at 0.84-0.86.  The chosen pair is a single head-dim-INDEPENDENT constant
# that is at or near the optimum at every one of the 9 correctness head dims.
_SCHED_A = "2:3"
_SCHED_B = "3:4"


def _rm(s):
    """Parse "<reads>:<mfmas>"; anything else disables the round robin."""
    if ":" not in s:
        return (0, 0)
    f = tuple(int(x) for x in s.split(":"))
    return (f[0], f[1])


_SCHED_RM_A = _rm(_SCHED_A)
_SCHED_RM_B = _rm(_SCHED_B)
# The FUSED single-role body (dK/dV *and* dQ in one pass) is a different region: it carries the
# dQ MFMAs and the dS `ds_write`s on top of job A's reads, so its round robin is a separate
# constant swept on the fused build itself -- and how much dQ work the region carries depends
# on whether every wave owns a dQ d-tile.  Measured on the fused build (full `--full-benchmark`):
# with the dQ pass DENSE (`n_tiles_qk == NUM_WAVES`, d = 120/128) the region wants a WIDER
# round robin than job A's -- 3:4 reads 0.560 / 1.031 ms on the two d128 cases against 0.607 /
# 1.116 with no round robin at all.  With the dQ pass SPARSE (d <= 112, some waves own no d
# tile) the same round robin is a LOSS -- d64 0.366 / d72 0.459 / d96 0.498 with 2:3 against
# 0.292 / 0.387 / 0.446 with LLVM's own order -- because the forced groups serialise a region
# whose MFMA count the model over-states.  So: round robin only when the dQ pass is dense.
_SCHED_RM_AF = _rm("off")
_SCHED_RM_AFD = _rm("off")

# ---- FUSED-BUILD per-iteration instruction trims ---------------------------------------------
# The SMALL fused build is the one the host pins to >= 2 co-resident workgroups per CU
# (`wgs_per_cu >= 2`, a COMPUTED register / LDS footprint property, see `_fused_wgs_per_cu` in the
# launcher): few d tiles per score element, so its non-MFMA work per MFMA is highest and
# issue-slot trims move it.  The WIDE fused build (1 workgroup per CU, d = 72..128) takes only
# the trims that measured positive there.  The two-role kernel emits none of them.
#
# Stage -lse*LOG2E and -delta*scale in LDS (one VALU per staged row) so the score chain is
# logit = fma(s, scale*LOG2E, -lse') and dS = P * fma(dP, scale, -delta'), each as a packed
# v_pk_fma_f32 over register-aligned halves of the accumulators.  SMALL only: -12 % at d72 WIDE.
OPT_FLSE = True
# dQ bf16 pair pack as perm -> DPP -> perm (3 VALU per pair, no lane-parity selects).  SMALL only
# (neutral on WIDE).
OPT_DQPACK2 = True
# dQ flush: no first-trip / row-past-end selects where provably dead, uniform SGPR base plus a
# 32-bit VGPR byte offset (global saddr form, no 64-bit VALU address math per atomic).  The
# staging tile is zeroed once up front so no trip needs the `keep` select.  Both fused builds
# (+1..5 % on WIDE).
OPT_FLUSH_LEAN = True
# dQ pass spread over BOTH query halves: wave w owns query half w % 2 of d tile w // 2, so when
# 2 * n_tiles_qk <= NUM_WAVES every wave (not just n_tiles_qk of them) carries dQ MFMAs and each
# reads half the dS fragments.  No extra register-resident K^T set.  The branch-free pass is
# fenced into its own scheduling region (sched_barrier), which is what keeps it spill-free at the
# 128-VGPR pin (unfenced it spilled 8 VGPRs inside the q loop).  SMALL only.
OPT_DQHALF = True
# WIDE build: issue every LDS read of the dQ flush before its first atomic (an ordered atomic is
# a scheduling fence, so read -> wait -> atomic serialised per pass).
OPT_WFLBATCH = True
# Measured dead ends on these builds: peeling the causal-diagonal query tiles into their own
# loop (spills at the 128-VGPR pin, flat), and reading the whole dS panel of the dQ pass up front
# with the two query halves' chains interleaved (flat to negative).

# LLVM sched_barrier / sched_group_barrier mask bits.
_SCHED_MFMA = 0x008
_SCHED_DSRD = 0x100
_SCHED_ALL = 0xFFF
_SCHED_NO_VMEM_RD = _SCHED_ALL & ~(0x010 | 0x020)  # everything BUT VMEM reads may cross


def _pin_vmem():
    """Pin the NEXT tile's global prefetch at the point the source emitted it.

    Both streaming loops issue the next tile's ``buffer_load``s immediately after the second
    ``fx.barrier()`` so their ~600-cycle latency is covered by this tile's MFMA run.  LLVM's
    post-RA scheduler is free to SINK them to the bottom of the body, which leaves the
    ``s_waitcnt vmcnt`` in front of the next iteration's LDS staging fully exposed -- i.e. the
    software pipeline exists only in the source.  Clearing just the two VMEM-read bits lets
    everything else (MFMA / DS / VALU / SALU) still move across, so the fence costs nothing but
    the placement of the loads.
    """
    fx.rocdl.sched_barrier(_SCHED_NO_VMEM_RD)


def _sched_pipe(nread, nmfma, rm):
    """Emit a ds_read(r) : MFMA(m) round robin over a body of `nread` reads / `nmfma` MFMAs.

    A run of ``sched_group_barrier`` calls describes a scheduling PIPELINE for the region they
    are emitted in: `r` ds_reads, then `m` MFMAs, repeated.  Without it LLVM clusters the body
    into read runs and MFMA runs, and every MFMA run stalls on the last read of the run before
    it.  The group sizes are NOT copied from a prior baseline -- they are swept (see report).
    """
    r, m = rm
    if r <= 0 or m <= 0:
        return
    ngrp = max(1, min((nread + r - 1) // r, (nmfma + m - 1) // m))
    for _ in range_constexpr(ngrp):
        fx.rocdl.sched_group_barrier(_SCHED_DSRD, r, 0)
        fx.rocdl.sched_group_barrier(_SCHED_MFMA, m, 0)


# --------------------------------------------------------------------------- small helpers
def _mfma(a, b, acc):
    """One ``v_mfma_f32_16x16x16_bf16_1k`` on bare v4 register values."""

    mma = fx.make_mma_atom(fx.rocdl.MFMA(16, 16, 16, fx.BFloat16))
    fa = fx.make_rmem_tensor(fx.make_layout(4, 1), fx.BFloat16)
    fb = fx.make_rmem_tensor(fx.make_layout(4, 1), fx.BFloat16)
    fc = fx.make_rmem_tensor(fx.make_layout(4, 1), fx.Float32)
    fx.memref_store_vec(a, fa)
    fx.memref_store_vec(b, fb)
    fx.memref_store_vec(acc, fc)
    fx.mma_atom_call(mma, fc, fa, fb, fc)
    return fx.memref_load_vec(fc)


def _exp2(x):
    return fx.Float32(fx.rocdl.exp2(fx.Float32.ir_type, x.ir_value()))


def _zero4():
    return fx.Vector.filled(4, 0.0, fx.Float32)


def _elem(acc4, i):
    """Element ``i`` of an fp32 MFMA accumulator / LDS fragment."""
    return fx.Vector(acc4)[i]


# v_perm_b32 byte selector that builds one TRUNCATING bf16 pair in ONE instruction.  The op
# reads an 8-byte source {src0 = bytes 7:4, src1 = bytes 3:0} and picks one byte per output
# byte; with src1 = a, src0 = b we want {b[3], b[2], a[3], a[2]} -- the high halves of both
# fp32s with `a` in the low dword half -- i.e. byte selector 7,6,3,2.
_PERM_BF16PACK = 0x07060302


def _pack4_trunc(vals):
    """4 fp32 -> one v4bf16 MFMA operand fragment.

    ``packperm``: one ``v_perm_b32`` per pair (0.5 VALU per score element, dependency depth 1)
    instead of and + shift + or (1.5 VALU per element, depth 2).  Paid TWICE per element in the
    dK/dV job (P^T and dS^T) and once in the dQ job.  Bit-identical result: both forms keep the
    high 16 bits of each fp32 and drop the low 16.
    """
    if const_expr(OPT_PACKPERM):
        sel = fx.Int32(_PERM_BF16PACK)

        def _pair(a, b):
            return fx.Int32(
                fx.rocdl.perm_b32(b.bitcast(fx.Int32), a.bitcast(fx.Int32), sel)
            )

        packed = fx.Vector.from_elements(
            [_pair(vals[0], vals[1]), _pair(vals[2], vals[3])], dtype=fx.Int32
        )
        return packed.bitcast(fx.BFloat16)

    c_mask = fx.Int32(0xFFFF0000)

    def _pair(a, b):
        a_bits = a.bitcast(fx.Int32)
        b_bits = b.bitcast(fx.Int32)
        return (b_bits & c_mask) | fx.arith.shrui(a_bits, fx.Int32(16))

    packed = fx.Vector.from_elements(
        [_pair(vals[0], vals[1]), _pair(vals[2], vals[3])], dtype=fx.Int32
    )
    return packed.bitcast(fx.BFloat16)


def _to_bf16(v):
    """fp32 -> bf16 in TWO VALU ops (add + shift), for the epilogue's scalar stores."""
    bits = v.bitcast(fx.Uint32) + fx.Uint32(0x00008000)
    return (bits >> fx.Uint32(16)).to(fx.Uint16).bitcast(fx.BFloat16)


def _buffer_view(tensor, nbytes, fx_dt, vec=1):
    """Flat ``(ntile, vec)`` buffer-tensor view with an EXACT num_records.
    ``num_records`` is a 32-bit BYTE count and ``nbytes`` is computed in i32, so this addressing
    tops out at 2 GiB per tensor; the launcher asserts that up front rather than letting it wrap.
    """
    ntile = (1 << 31) // (vec * (fx_dt.width // 8))
    buf = fx.rocdl.make_buffer_tensor(tensor, num_records_bytes=fx.Int64(nbytes))
    return fx.Tensor(
        fx.make_view(fx.get_iter(buf), fx.make_layout((ntile, vec), (vec, 1)))
    )


def _atom(fx_dt, vec):
    """Widest single buffer copy atom covering a ``vec``-wide ``fx_dt`` fragment."""
    return fx.make_copy_atom(fx.rocdl.BufferCopy(vec * fx_dt.width), fx_dt)


def _ldv(buf, tile, fx_dt, vec):
    """``vec`` elements at element offset ``tile * vec``, as a raw vector value."""
    frag = fx.make_rmem_tensor(fx.make_layout(vec, 1), fx_dt)
    fx.copy_atom_call(_atom(fx_dt, vec), fx.slice(buf, (tile, None)), frag)
    return frag.load().ir_value()


def _ld1(buf, idx, fx_dt):
    """One ``fx_dt`` element at element index ``idx``, typed."""
    frag = fx.make_rmem_tensor(fx.make_layout(1, 1), fx_dt)
    fx.copy_atom_call(_atom(fx_dt, 1), fx.slice(buf, (idx, None)), frag)
    return frag.load()[0]


def _stv(val, buf, tile, fx_dt, vec):
    """Store the ``vec``-wide vector ``val`` at element offset ``tile * vec``."""
    frag = fx.make_rmem_tensor(fx.make_layout(vec, 1), fx_dt)
    fx.memref_store_vec(val, frag)
    fx.copy_atom_call(_atom(fx_dt, vec), frag, fx.slice(buf, (tile, None)))


def _st1(val, buf, idx, fx_dt):
    """Store one ``fx_dt`` value at element index ``idx``."""
    frag = fx.make_rmem_tensor(fx.make_layout(1, 1), fx_dt)
    fx.memref_store_vec(fx.Vector.from_elements([val], dtype=fx_dt), frag)
    fx.copy_atom_call(_atom(fx_dt, 1), frag, fx.slice(buf, (idx, None)))


def _bf16x4(ptr):
    return fx.ptr_load(ptr, result_type=fx.Vector.make_type(4, fx.BFloat16))


def _global_bf16_ptr(tensor):
    """The tensor's base as a PAIR-ALIGNED raw bf16 global pointer.

    ``global_atomic_pk_add_bf16`` is a 32-bit atomic and a bf16 memref carries align 2, which
    makes the backend refuse to select the packed form; re-typing the base at align 4 is what
    the other FlyDSL kernels that use this atom do.
    """
    return fx.inttoptr(
        fx.PointerType.get(fx.BFloat16.ir_type, fx.AddressSpace.Global, 4),
        fx.ptrtoint(fx.get_iter(tensor)),
    )


def _atompk2(v2, base, idx):
    """``global_atomic_pk_add_bf16`` of one bf16 PAIR at element index ``idx``.

    DELIBERATELY the ``atomicrmw`` (flat/global) form, not a buffer atom: gfx942 has
    ``global_atomic_pk_add_bf16`` but not the buffer-addressed one.  Losing the buffer resource
    means losing its free out-of-range masking, so every caller steers the INDEX onto a slot it
    is allowed to touch and relies on the value being exactly +0.0 there.
    """
    _mlir_llvm.atomicrmw(
        _LLVM_FADD,
        fx.to_llvm_ptr(base + idx),
        fx.Vector(v2).ir_value(),
        _LLVM_MONOTONIC,
        syncscope="agent",
        alignment=4,
    )


_DPP_QUAD_1032 = 0xB1  # quad_perm:[1,0,3,2] -- swap lane l with lane l^1


def _dqpack(acc4, sel, odd):
    """One dQ accumulator tile -> the TWO bf16 pairs of this lane, for the LDS staging tile.

    ``acc4`` is the accumulator of ``dQ = dS . K^T``: lane ``l`` holds FOUR consecutive query
    rows (``lane_k + 0..3``) at ONE ``d`` (``lane_row``).  A bf16 pair needs two ADJACENT ``d``
    in one dword and that partner lives in lane ``l ^ 1``, one ``quad_perm:[1,0,3,2]`` DPP move
    away.  The pair then SPLITS the four rows: the even lane hands out rows ``lk+0`` / ``lk+2``,
    the odd lane ``lk+1`` / ``lk+3``, so each row is produced exactly once.

    The fp32 -> bf16 conversion is round-to-nearest (``+0x8000`` on the bit pattern before
    ``v_perm_b32`` keeps the high half): dQ partials accumulate ACROSS key blocks, so a
    truncating pack would bias every one of them toward zero and the bias would compound.
    """
    out = []
    av = fx.Vector(acc4)
    rnd = [
        (av[e].bitcast(fx.Uint32) + fx.Uint32(0x00008000)).bitcast(fx.Int32)
        for e in range_constexpr(4)
    ]
    par = None
    for j in range_constexpr(2):
        own = odd.select(rnd[2 * j + 1], rnd[2 * j])
        oth = odd.select(rnd[2 * j], rnd[2 * j + 1])
        # `update_dpp`'s first operand is tied to the destination register; feeding the PREVIOUS
        # granule's `par` (dead by then) instead of `own` saves the `v_mov_b32` the allocator
        # would otherwise need to keep `own` live.
        seed = oth if par is None else par
        par = fx.Int32(
            fx.rocdl.update_dpp(
                fx.Int32.ir_type,
                seed.ir_value(),
                oth.ir_value(),
                _DPP_QUAD_1032,
                0xF,
                0xF,
                False,
            )
        )
        out.append(
            fx.Vector.from_elements(
                [fx.Int32(fx.rocdl.perm_b32(par, own, sel))], dtype=fx.Int32
            ).bitcast(fx.BFloat16)
        )
    return out


_PERM_DQ_EVEN = (
    0x05040100  # {src0[1:0], src1[1:0]}: partner's low bf16 high, own low bf16 low
)
_PERM_DQ_ODD = (
    0x03020706  # {src1[3:2], src0[3:2]}: own high bf16 high, partner's high bf16 low
)


def _dqpack2(acc4, sel2):
    """``_dqpack`` in 3 VALU per output pair: local pack -> DPP swap -> lane-selected perm.

    Each lane first packs its OWN two rounded rows ``2j`` / ``2j+1`` into one dword ``w``
    (``{hi: row 2j+1, lo: row 2j}``) and swaps it with lane ``l ^ 1``.  The even lane (low d of
    the pair) then keeps row ``2j`` of both (own low half, partner's low half), the odd lane row
    ``2j+1`` of both (own high half, partner's high half) -- one ``v_perm_b32`` with a per-lane
    selector, so the output layout is exactly ``_dqpack``'s and no lane-parity select is needed.
    Same round-to-nearest (``+0x8000``) as ``_dqpack``.
    """
    av = fx.Vector(acc4)
    rnd = [
        (av[e].bitcast(fx.Uint32) + fx.Uint32(0x00008000)).bitcast(fx.Int32)
        for e in range_constexpr(4)
    ]
    lsel = fx.Int32(_PERM_BF16PACK)
    out = []
    for j in range_constexpr(2):
        w = fx.Int32(fx.rocdl.perm_b32(rnd[2 * j + 1], rnd[2 * j], lsel))
        # The DPP destination is tied to its `old` operand; every lane is written (full row /
        # bank masks), so seeding it with the now-dead rounded row avoids a v_mov copy of `w`.
        wp = fx.Int32(
            fx.rocdl.update_dpp(
                fx.Int32.ir_type,
                rnd[2 * j].ir_value(),
                w.ir_value(),
                _DPP_QUAD_1032,
                0xF,
                0xF,
                False,
            )
        )
        out.append(
            fx.Vector.from_elements(
                [fx.Int32(fx.rocdl.perm_b32(wp, w, sel2))], dtype=fx.Int32
            ).bitcast(fx.BFloat16)
        )
    return out


def _v2(a, b):
    return fx.Vector.from_elements([a, b], dtype=fx.Float32)


def _packn_rne(vals):
    """2N fp32 -> one v(2N)bf16, round-to-nearest (add 0x8000 before the truncating pack).

    Same 2-op-per-element trick as ``_to_bf16`` but producing a wide packed vector, for the split-K
    reduction kernel's vectorised bf16 stores.
    """
    c_mask = fx.Int32(0xFFFF0000)
    rnd = fx.Int32(0x00008000)

    def _pair(a, b):
        a_bits = a.bitcast(fx.Int32) + rnd
        b_bits = b.bitcast(fx.Int32) + rnd
        return (b_bits & c_mask) | fx.arith.shrui(a_bits, fx.Int32(16))

    npair = len(vals) // 2
    packed = fx.Vector.from_elements(
        [_pair(vals[2 * i], vals[2 * i + 1]) for i in range(npair)], dtype=fx.Int32
    )
    return packed.bitcast(fx.BFloat16)


@functools.lru_cache(maxsize=None)
def build(
    n_split=1, d_qk=DQK, d_v=DV, deterministic=False, fuse_split=False, wgs_per_cu=0
):
    """Compile the kernel set for one ``(n_split, d_qk, d_v, deterministic)``.

    ``n_split`` = split-K factor along the STREAMED index.

    ``n_split == 1`` is the two-kernel path (k_delta + k_bwd writing bf16 straight into dq/dk/dv)
    and every Python-level branch below is written so that this variant emits exactly the code
    it emitted before split-K existed.

    ``n_split > 1`` compiles a SECOND, independently cached variant used only for small workloads
    (see ``_split()`` in aiter/ops/flydsl/fmha_bwd_gfx942.py), where every workgroup is
    co-resident and the makespan is the LONGEST SINGLE WORKGROUP rather than the total work.

    ``d_qk`` / ``d_v`` select the head dims; see ``_Geom`` for the three widths they induce.
    ``deterministic`` rules out the fused single-role kernel (see ``fused_role_enabled``).
    ``fuse_split`` selects the FUSED split-K variant for ``n_split > 1``.  ``wgs_per_cu`` > 0
    pins k_bwd's register budget to that many co-resident workgroups per CU (``waves_per_eu``);
    the host passes the occupancy its makespan model assumed, so the compiler cannot silently
    halve it.
    Every name they feed is a plain Python constant here, so the body below reads exactly as it
    did when the dims were module constants.
    """
    g = _geom(d_qk, d_v)
    DQK, DV = g.d_qk, g.d_v  # real head dims -> all global addressing
    LD_QK, LD_V = g.ld_qk, g.ld_v  # LDS row strides (tile width `d_s`, plus the +4 pad)
    N_TILES_QK, N_TILES_V = g.n_tiles_qk, g.n_tiles_v
    FRAGS_PER_ROW_QK, FRAGS_PER_ROW_V = g.frags_per_row_qk, g.frags_per_row_v
    N_QK_CHUNKS, N_V_CHUNKS = g.n_qk_chunks, g.n_v_chunks
    OFF_Q_NAT, OFF_Q_T, OFF_DO_NAT, OFF_DO_T = (
        g.off_q_nat,
        g.off_q_t,
        g.off_do_nat,
        g.off_do_t,
    )
    OFF_K_NAT, OFF_K_T, OFF_V_NAT = g.off_k_nat, g.off_k_t, g.off_v_nat
    ARENA = g.arena
    # Index of the one staging chunk that straddles the end of a real row, or -1 when d is a
    # multiple of 64 and every chunk is wholly inside it.  Chunks before it are unconditionally
    # valid; there is never a chunk wholly past it, because the tile width is `ceil(d/64)*64`.
    PART_CHUNK_QK = DQK // 64 if DQK % 64 else -1
    PART_CHUNK_V = DV // 64 if DV % 64 else -1
    # Elements of the last MFMA tile that are real, when d is an odd multiple of 8.
    REM_QK = DQK % 16
    REM_V = DV % 16

    # ---- FUSED SINGLE-ROLE 5-GEMM SELECTION ----------------------------------------------
    # The two-role kernel issues 7*s^2*d MAC against the 5*s^2*d causal minimum, because job B
    # recomputes S and dP purely to reach dQ = dS.K.  The fused role collapses both roles into
    # ONE KV-outer role that forms S / dP / dS exactly once per (query tile, key tile) and
    # carries all five GEMMs.  dQ is the one gradient a KV-outer role cannot own, so it
    # accumulates across key blocks with `global_atomic_pk_add_bf16` straight into the caller's
    # bf16 dq -- WHICH BREAKS BIT-DETERMINISM OF dq (fp32 add is not associative); dk/dv are
    # untouched.  That is the same trade AMD's own ASM v3 backward makes.  Restricted to
    # n_split == 1: the co-resident split-K path stays fully deterministic.
    FUSED = fused_role_enabled(n_split, d_qk, d_v, deterministic, fuse_split)
    N_KEY_TILES = KEYS_PER_WG // 16  # key tiles a wave contracts dQ over
    # d tiles < waves leaves the tail waves with no dQ tile to own; they skip the dQ pass.
    DQ_ALL_WAVES = N_TILES_QK == NUM_WAVES
    # A K^T row is invalid when it falls past `d_qk` -- either because the last MFMA tile is
    # half real (d % 16 == 8) or because this wave owns no d tile at all.
    KT_MASK = bool(REM_QK) or not DQ_ALL_WAVES
    # Job A's ds_read : MFMA round robin.  Three regions, three constants: the two-role job A
    # (`_SCHED_RM_A`), the fused role with a DENSE dQ pass (`_SCHED_RM_AFD`) and the fused role
    # with a SPARSE one (`_SCHED_RM_AF`, LLVM's own order by default).  See the table above the
    # constants.
    SCHED_RM_JOB_A = (
        (_SCHED_RM_AFD if DQ_ALL_WAVES else _SCHED_RM_AF) if FUSED else _SCHED_RM_A
    )
    # The small-footprint fused build: the host pinned it to >= 2 workgroups per CU from its
    # computed register / LDS footprint (module notes at OPT_FLSE).
    SMALL = bool(FUSED and wgs_per_cu >= 2)
    WIDE = bool(FUSED and not SMALL)
    FLSE = SMALL and OPT_FLSE and not OPT_ASCALE and not OPT_DFOLD
    LEAN = FUSED and OPT_FLUSH_LEAN
    DQPACK2 = SMALL and OPT_DQPACK2
    FLBATCH = WIDE and OPT_WFLBATCH
    # dQ over both query halves when the d tiles fill at most half the waves.
    DQ_HALF = SMALL and OPT_DQHALF and 2 * N_TILES_QK <= NUM_WAVES
    if DQ_HALF:
        KT_MASK = bool(REM_QK) or 2 * N_TILES_QK < NUM_WAVES
    ARENA = g.arena_dkdv if FUSED else ARENA
    OFF_DS = g.off_ds
    OFF_DQS, LD_DQS = g.off_dqs, g.ld_dqs
    # Flush side of the staging tile: the tile's QUERY_TILE x d_qk/2 bf16 pairs, dealt to the
    # workgroup's threads pair-fastest so one wave instruction covers whole contiguous rows.
    DQS_PAIRS = DQK // 2
    DQS_TOTAL = QUERY_TILE * DQS_PAIRS
    DQS_PASSES = -(-DQS_TOTAL // NUM_THREADS)

    # LDS storage layouts
    if FUSED:

        @fx.struct
        class SmemBwd:
            arena: fx.Array[
                fx.BFloat16,
                ARENA + QUERY_TILE * (LD_DS + LD_DQS),
                16,
            ]
            lse_s: fx.Array[fx.Float32, QUERY_TILE, 16]
            del_s: fx.Array[fx.Float32, QUERY_TILE, 16]

    else:

        @fx.struct
        class SmemBwd:
            arena: fx.Array[fx.BFloat16, ARENA, 16]
            lse_s: fx.Array[fx.Float32, QUERY_TILE, 16]
            del_s: fx.Array[fx.Float32, QUERY_TILE, 16]

    # delta = rowsum(dO*O)
    # rows covered by one 256-thread pass (16 lanes per row)
    ROWS_PER_PASS = DELTA_THREADS // 16
    # passes per workgroup -> ROWS_DELTA rows per workgroup
    PASSES_PER_WG = ROWS_DELTA // ROWS_PER_PASS
    # 8-element tiles spanning one row of dO / O, and how many passes of 16 lanes that needs.
    TILES_PER_ROW_V = DV // 8
    DELTA_CHUNKS = -(-TILES_PER_ROW_V // 16)

    # dq zeroing, folded into the delta pre-pass on the fused build (see `k_delta`): the fused
    # role ACCUMULATES dQ into the caller's tensor, so it must start at zero, and k_delta is a
    # pure streaming kernel with spare instruction slots.
    Z_TILES_PER_ROW = DQK // 8  # 8-element bf16 tiles per dq row
    Z_TILES_PER_WG = ROWS_DELTA * Z_TILES_PER_ROW
    Z_PASSES = -(-Z_TILES_PER_WG // DELTA_THREADS)

    def _delta_body(DO, O, DEL, DQZ, Tlen, Hn):
        """D = rowsum(dO * O).

        The 16 partial sums of a row live in 16 CONSECUTIVE LANES of one wave, so the reduction
        is a pure cross-lane XOR butterfly (4 shuffles): no LDS array and no ``fx.barrier()``.
        The row groups are unrolled PASSES_PER_WG-deep with every global load issued up front,
        because this kernel is pure streaming bandwidth.  DESIGN.md §7

        16 lanes cover ``TILES_PER_ROW_V`` 8-element tiles of a row in ``DELTA_CHUNKS`` passes.
        At d_v = 128 that is exactly one pass with every lane busy; a narrower d_v idles the
        lanes past the end and a wider one takes a second pass, in both cases by steering the
        out-of-row tiles past ``num_records`` so they read 0 and drop out of the sum.
        """
        tid = fx.Int32(fx.thread_idx.x)
        bid = fx.Int32(fx.block_idx.x)
        n_rows = Tlen * Hn
        g_do = _buffer_view(DO, n_rows * (DV * 2), fx.BFloat16, 8)
        g_o = _buffer_view(O, n_rows * (DV * 2), fx.BFloat16, 8)
        g_delta = _buffer_view(DEL, n_rows * 4, fx.Float32)

        lane_in_row = tid % 16
        row_in_group = tid // 16
        oob_tile_v = n_rows * TILES_PER_ROW_V  # one tile past the end of dO / O

        # The u-th pass steps a whole ROWS_PER_PASS x d_v element block, a compile-time delta
        # that folds into the load's immediate offset field.
        if const_expr(TILES_PER_ROW_V == 16):
            tile = bid * (ROWS_DELTA * TILES_PER_ROW_V) + tid
        else:
            tile = (
                bid * (ROWS_DELTA * TILES_PER_ROW_V)
                + row_in_group * TILES_PER_ROW_V
                + lane_in_row
            )

        def _row_tiles(buf, u):
            """This lane's ``DELTA_CHUNKS`` 8-element tiles of pass ``u``'s row."""
            base = tile + u * (ROWS_PER_PASS * TILES_PER_ROW_V)
            out = []
            for c in range_constexpr(DELTA_CHUNKS):
                idx = base + c * 16
                if const_expr(c * 16 + 16 > TILES_PER_ROW_V):
                    idx = (lane_in_row < fx.Int32(TILES_PER_ROW_V - c * 16)).select(
                        idx, oob_tile_v
                    )
                out.append(_ldv(buf, idx, fx.BFloat16, 8))
            return out

        do_vecs = [_row_tiles(g_do, u) for u in range_constexpr(PASSES_PER_WG)]
        o_vecs = [_row_tiles(g_o, u) for u in range_constexpr(PASSES_PER_WG)]

        if const_expr(FUSED):
            # Issued right behind the delta loads so the zero stream overlaps them; the
            # workgroup covers exactly the ROWS_DELTA dq rows whose delta it computes.
            g_dqz = _buffer_view(DQZ, n_rows * (DQK * 2), fx.BFloat16, 8)
            oob_tile_z = n_rows * Z_TILES_PER_ROW
            zero8 = fx.Vector.filled(8, 0.0, fx.BFloat16)
            for c in range_constexpr(Z_PASSES):
                off = c * DELTA_THREADS + tid
                idx = bid * Z_TILES_PER_WG + off
                if const_expr((c + 1) * DELTA_THREADS > Z_TILES_PER_WG):
                    idx = (off < fx.Int32(Z_TILES_PER_WG)).select(idx, oob_tile_z)
                _stv(zero8, g_dqz, idx, fx.BFloat16, 8)

        for u in range_constexpr(PASSES_PER_WG):
            # Even/odd lanes accumulate into two independent fp32 chains, so the 8-element dot
            # product is not one serial dependency line.
            e0 = fx.Float32(0.0)
            e1 = fx.Float32(0.0)
            for chunk in range_constexpr(DELTA_CHUNKS):
                do8 = fx.Vector(do_vecs[u][chunk])
                o8 = fx.Vector(o_vecs[u][chunk])
                for c in range_constexpr(4):
                    e0 = e0 + fx.Float32(do8[2 * c]) * fx.Float32(o8[2 * c])
                    e1 = e1 + fx.Float32(do8[2 * c + 1]) * fx.Float32(o8[2 * c + 1])
            acc = e0 + e1
            for s in range_constexpr(4):
                acc = acc + fx.gpu.shuffle_xor(acc, 1 << s, 64)
            idx = bid * ROWS_DELTA + u * ROWS_PER_PASS + row_in_group
            ok = (lane_in_row == 0) & (idx < n_rows)
            t = idx // Hn
            h = idx % Hn
            # Only lane 0 of each row group holds the reduced sum; the other 15 and any row
            # past the end are steered one element past DEL so the hardware drops the store.
            _st1(acc, g_delta, ok.select(h * Tlen + t, n_rows), fx.Float32)

    # TWO SIGNATURES, one body.  `_delta_body` is a plain Python emitter, so the two kernels
    # below trace the SAME code -- but the two-role build never sees the `dq` argument, and on
    # the smallest split-K shape (5 dispatches, ~90 us total) one extra tensor to marshal per
    # launch measured a reproducible +2.7 %.
    if FUSED:

        @flyc.kernel(known_block_size=[DELTA_THREADS, 1, 1])
        def k_delta(
            DO: fx.Tensor,
            O: fx.Tensor,
            DEL: fx.Tensor,
            DQZ: fx.Tensor,
            Tlen: fx.Int32,
            Hn: fx.Int32,
        ):
            _delta_body(DO, O, DEL, DQZ, Tlen, Hn)

        @flyc.jit
        def launch_delta(
            DO: fx.Tensor,
            O: fx.Tensor,
            DEL: fx.Tensor,
            DQZ: fx.Tensor,
            Tlen: fx.Int32,
            Hn: fx.Int32,
            nblk: fx.Int32,
            stream: fx.Stream,
        ):
            k_delta(DO, O, DEL, DQZ, Tlen, Hn).launch(
                grid=(nblk, 1, 1), block=(DELTA_THREADS, 1, 1), stream=stream
            )

    else:

        @flyc.kernel(known_block_size=[DELTA_THREADS, 1, 1])
        def k_delta(
            DO: fx.Tensor, O: fx.Tensor, DEL: fx.Tensor, Tlen: fx.Int32, Hn: fx.Int32
        ):
            _delta_body(DO, O, DEL, None, Tlen, Hn)

        @flyc.jit
        def launch_delta(
            DO: fx.Tensor,
            O: fx.Tensor,
            DEL: fx.Tensor,
            Tlen: fx.Int32,
            Hn: fx.Int32,
            nblk: fx.Int32,
            stream: fx.Stream,
        ):
            k_delta(DO, O, DEL, Tlen, Hn).launch(
                grid=(nblk, 1, 1), block=(DELTA_THREADS, 1, 1), stream=stream
            )

    # the fused-schedule backward kernel
    @flyc.kernel(known_block_size=[NUM_THREADS, 1, 1])
    def k_bwd(
        Q: fx.Tensor,
        K: fx.Tensor,
        V: fx.Tensor,
        DO: fx.Tensor,
        LSE: fx.Tensor,
        DEL: fx.Tensor,
        CU: fx.Tensor,
        DQO: fx.Tensor,
        DKO: fx.Tensor,
        DVO: fx.Tensor,
        Tlen: fx.Int32,
        Hn: fx.Int32,
        nseq: fx.Int32,
        n_dkdv_blocks: fx.Int32,
        n_blocks: fx.Int32,
        scale: fx.Float32,
        interleave: fx.Int32,
    ):
        tid = fx.Int32(fx.thread_idx.x)
        lane = tid % 64
        wave = tid // 64
        lane_row = lane % 16
        lane_k = (lane // 16) * 4
        lane_k_frag = lane // 16  # lane_k as a 4-element tile index

        blk_x = fx.Int32(fx.block_idx.x)
        head = blk_x % Hn
        blk_y = fx.Int32(fx.block_idx.y)
        # SEQUENCE ROTATION.  MI300 deals workgroups out statically: linear id l goes to XCD
        # l % 8 and, inside it, to shader engine (l / 8) % 4.  With the plain `blk_x // Hn`
        # decode and Hn % 8 == 0 every round of blk_y puts sequence s on the SAME shader engine,
        # so a ragged batch piles its long sequences onto one SE and its makespan depends on
        # the ORDER of the sequences (d128 [300..4096]h8: 0.60-0.83 ms).  Rotating the sequence
        # by blk_y is a bijection within each round and spreads every sequence over all SEs
        # (0.55 ms, order-insensitive).
        seq_idx = (blk_x // Hn + blk_y) % nseq

        n_rows = Tlen * Hn
        # Q / K / V / dO are only ever read as 4-element MFMA fragments, so their views are
        # tiled 4 wide and every offset into them below is a TILE index (element offset / 4).
        gQ = _buffer_view(Q, n_rows * (DQK * 2), fx.BFloat16, 4)
        gK = _buffer_view(K, n_rows * (DQK * 2), fx.BFloat16, 4)
        if const_expr(FUSED):
            # 1-wide view of K: the dQ role's K^T fragments are four scalar loads strided by a
            # token, not a contiguous 4-element run along d.
            gK1 = _buffer_view(K, n_rows * (DQK * 2), fx.BFloat16)
        gV = _buffer_view(V, n_rows * (DV * 2), fx.BFloat16, 4)
        gDO = _buffer_view(DO, n_rows * (DV * 2), fx.BFloat16, 4)
        gLSE = _buffer_view(LSE, n_rows * 4, fx.Float32)
        gDEL = _buffer_view(DEL, n_rows * 4, fx.Float32)
        gCU = _buffer_view(CU, (nseq + 1) * 4, fx.Int32)
        # The FUSED role (split or not) accumulates straight into the caller's dq, never a slab.
        gDQ = _buffer_view(
            DQO, n_rows * (DQK * 2 * (1 if FUSED else n_split)), fx.BFloat16
        )
        gDK = _buffer_view(DKO, n_rows * (DQK * 2 * n_split), fx.BFloat16)
        gDV = _buffer_view(DVO, n_rows * (DV * 2 * n_split), fx.BFloat16)
        if const_expr(n_split > 1):
            slab_qk = n_rows * DQK  # elements per partial slab of dq / dk
            slab_v = n_rows * DV  # elements per partial slab of dv
        # Total elements behind each output view; any index >= this is past num_records, so
        # the hardware drops the store.  These are the masked-store steer targets.
        oob_qk = n_rows * (DQK * n_split)  # dq / dk, all slabs
        oob_v = n_rows * (DV * n_split)  # dv, all slabs
        # ...and the same one-past-the-end index in the 4-wide INPUT views, which is where a
        # read of a head-dim padding column is steered so the hardware returns 0 instead of the
        # next token's data (the descriptors bound the whole tensor, not one row).
        oob_frag_qk = n_rows * FRAGS_PER_ROW_QK
        oob_frag_v = n_rows * FRAGS_PER_ROW_V

        seq_lo = _ld1(gCU, seq_idx, fx.Int32)
        seq_hi = _ld1(gCU, seq_idx + 1, fx.Int32)
        seq_len = seq_hi - seq_lo

        scale_log2e = scale * fx.Float32(LOG2E)

        lds = fx.SharedAllocator().allocate(SmemBwd).peek()
        lds_arena = lds.arena.ptr
        lds_lse = lds.lse_s.ptr
        lds_delta = lds.del_s.ptr
        if const_expr(FUSED):
            lds_ds = lds_arena + OFF_DS  # [QUERY_TILE, LD_DS] dS exchange tile
            lds_dqs = lds_arena + OFF_DQS  # [QUERY_TILE, LD_DQS] dQ staging tile

        stage_row = tid // 16
        stage_col = (tid % 16) * 4
        stage_col_frag = tid % 16  # stage_col as a 4-element tile index
        stage_tok = tid % QUERY_TILE

        # ---- XOR swizzle of the TRANSPOSED LDS tiles (Q^T, dO^T, K^T) ---------------------
        # Those tiles are filled by scalar ds_write_b16 with adjacent lanes stepping the tile ROW
        # index d, which lands the 16 lanes of a row group on only FOUR banks.  Padding cannot fix
        # it (the dword stride is always even), so the TOKEN axis is permuted instead, at the
        # 4-element granularity of the ds_read_b64 that consumes the tile:
        #
        #     element(d, tok)  ->  d*LD_TOK_DKDV + ((SW(d) ^ (tok>>2)) << 2) + (tok & 3)
        #     SW(d) = (d >> 4) & 7
        #
        # SW keys on bit 4 and up of d ON PURPOSE: that makes it constant across the 16 lanes of a
        # read (so the reads cannot regress) while varying across the storing lanes (so the stores
        # spread over 16 banks).  Swizzling on (d>>2)&7 instead is the obvious alternative and was
        # measured 3% SLOWER -- do not re-try it.
        swz_lane = (tid % 16) >> 2
        swz_store_even = ((swz_lane ^ (stage_row >> 2)) << 2) + (stage_row & 3)
        swz_store_odd = swz_store_even ^ 16
        swz_lane_base = (lane >> 4) << 2
        swz_read = [swz_lane_base ^ (k << 2) for k in range_constexpr(4)]

        def _store_col(d_chunk):
            """Swizzled token-column of a staging store of 64-element chunk `d_chunk`."""
            return swz_store_odd if (d_chunk & 1) else swz_store_even

        def _read_col(d_tile, tok_tile):
            """Swizzled token-column of an MFMA read of row block `d_tile`, tokens `tok_tile`."""
            return swz_read[d_tile & 3] + 16 * (((d_tile >> 2) & 1) ^ tok_tile)

        # ---- head-dim padding ---------------------------------------------------------------
        # An LDS tile is `d_s` (= ceil(d/64)*64) columns wide but a row of the tensor is only `d`,
        # so the staging chunk that straddles the end of the row has to write ZERO for the
        # columns past it.  Steering the LOAD past num_records is how that zero is produced --
        # free, and it keeps the store side byte-identical to the 192/128 code.
        #
        # With the LDS padding zeroed, NEITHER MFMA operand needs masking anywhere below: every
        # score GEMM takes its A operand from LDS, so a padding column contributes 0 * (whatever
        # the register-resident B operand happened to load) = 0, and every output GEMM's padding
        # lands on the d axis of the C fragment, which the epilogue drops (`_out_base`).  The
        # register-resident operands are masked anyway, once per workgroup outside the streaming
        # loop, so a NaN in a caller's tensor cannot leak in through a padding column.
        stage_col_ok_qk = (
            (stage_col + PART_CHUNK_QK * 64) < fx.Int32(DQK)
            if PART_CHUNK_QK >= 0
            else None
        )
        stage_col_ok_v = (
            (stage_col + PART_CHUNK_V * 64) < fx.Int32(DV)
            if PART_CHUNK_V >= 0
            else None
        )
        # Last-MFMA-tile lane predicates, for an odd multiple of 8 where that tile is half real.
        lane_k_ok_qk = lane_k < fx.Int32(REM_QK) if REM_QK else None
        lane_k_ok_v = lane_k < fx.Int32(REM_V) if REM_V else None
        lane_row_ok_qk = lane_row < fx.Int32(REM_QK) if REM_QK else None
        lane_row_ok_v = lane_row < fx.Int32(REM_V) if REM_V else None

        def _stage_idx(idx, d_chunk, part_chunk, ok, oob):
            """Staging-load tile index, steered out of range on a padding column."""
            if const_expr(d_chunk == part_chunk):
                return ok.select(idx, oob)
            return idx

        def _frag_idx(idx, d_tile, n_tiles, rem, ok, oob):
            """Register-resident operand tile index, steered out of range past `d`."""
            if const_expr(rem and d_tile == n_tiles - 1):
                return ok.select(idx, oob)
            return idx

        def _out_base(base, d_tile, n_tiles, rem, ok, oob):
            """Epilogue store base, steered past num_records on a padding element of `d`."""
            if const_expr(rem and d_tile == n_tiles - 1):
                return ok.select(base + d_tile * 16, oob)
            return base + d_tile * 16

        # ---- GRID DECODE: MERGED-LPT interleave of the two job types -------------------------
        # Both job types' per-block work is an arithmetic sequence with the SAME slope of 4
        # iterations per index step, so the true longest-processing-time-first order over their
        # UNION is a 1:1 interleave of the two descending lists:
        #   blk_y even -> dK/dV block blk_y/2 ; blk_y odd -> dQ block (n_dq_blocks-1 - blk_y/2)
        # The machine then always holds a mix of both job types.  Simply CONCATENATING the two
        # lists instead was measured to cost the full SUM of the two job types' standalone times.
        #
        # The n_paired / tail_idx arithmetic covers n_dkdv_blocks != n_dq_blocks, unreachable
        # while max_seqlen_q == max_seqlen_k but the kernel must stay general.
        if const_expr(FUSED):
            # ONE role: the grid carries dK/dV blocks only and the host sized it that way.
            is_dkdv = True
            dkdv_blk = blk_y
            dq_blk = fx.Int32(0)
        else:
            n_dq_blocks = n_blocks - n_dkdv_blocks
            n_paired = (n_dkdv_blocks < n_dq_blocks).select(n_dkdv_blocks, n_dq_blocks)
            pair_idx = blk_y // 2
            tail_idx = blk_y - n_paired * 2
            in_paired = blk_y < n_paired * 2
            use_interleave = interleave > 0
            is_dkdv = use_interleave.select(
                in_paired.select(
                    ((blk_y % 2) == 0).select(fx.Int32(1), fx.Int32(0)),
                    (n_dkdv_blocks > n_dq_blocks).select(fx.Int32(1), fx.Int32(0)),
                ),
                (blk_y < n_dkdv_blocks).select(fx.Int32(1), fx.Int32(0)),
            )
            dkdv_blk = use_interleave.select(
                in_paired.select(pair_idx, n_paired + tail_idx), blk_y
            )
            dq_blk = use_interleave.select(
                in_paired.select(
                    n_dq_blocks - 1 - pair_idx, n_dq_blocks - 1 - n_paired - tail_idx
                ),
                (n_blocks - 1) - blk_y,
            )

        if const_expr(n_split > 1):
            # The host multiplied the block counts by n_split, so the decoded index carries the
            # split index in its low digit.  All splits of one block cost the same, so the
            # division keeps both job lists descending as the merged-LPT decode assumes.
            dkdv_split = dkdv_blk % n_split
            dkdv_blk = dkdv_blk // n_split
            dq_split = dq_blk % n_split
            dq_blk = dq_blk // n_split

        if is_dkdv > 0:
            # dK/dV JOB
            lds_q = lds_arena + OFF_Q_NAT
            lds_qt = lds_arena + OFF_Q_T
            lds_do = lds_arena + OFF_DO_NAT
            lds_dot = lds_arena + OFF_DO_T

            key_block_base = dkdv_blk * KEYS_PER_WG

            if key_block_base < seq_len:
                # --- K / V operand fragments for this wave's 16 keys, register-resident -------
                key_in_blk = wave * 16 + lane_row  # key row inside the block
                key_row = seq_lo + key_block_base + key_in_blk
                # The 12 K / 8 V loads differ only by a CONSTANT d-tile delta, so one vgpr address
                # is computed here and each load folds its delta into the immediate offset field.
                k_frag_base = (key_row * Hn + head) * FRAGS_PER_ROW_QK + lane_k_frag
                v_frag_base = (key_row * Hn + head) * FRAGS_PER_ROW_V + lane_k_frag
                k_frags = []
                for d_tile in range_constexpr(N_TILES_QK):
                    k_frags.append(
                        _ldv(
                            gK,
                            _frag_idx(
                                k_frag_base + d_tile * 4,
                                d_tile,
                                N_TILES_QK,
                                REM_QK,
                                lane_k_ok_qk,
                                oob_frag_qk,
                            ),
                            fx.BFloat16,
                            4,
                        )
                    )
                v_frags = []
                for d_tile in range_constexpr(N_TILES_V):
                    v_frags.append(
                        _ldv(
                            gV,
                            _frag_idx(
                                v_frag_base + d_tile * 4,
                                d_tile,
                                N_TILES_V,
                                REM_V,
                                lane_k_ok_v,
                                oob_frag_v,
                            ),
                            fx.BFloat16,
                            4,
                        )
                    )

                if const_expr(FUSED):
                    # ---- K^T operand set for dQ = dS . K^T, register-resident -------------
                    # dQ contracts over the KEY, so its B operand is K TRANSPOSED: lane l must
                    # hold K^T[d = dt*16 + lane_row][key = kt*16 + lane_k + 0..3], i.e. four
                    # keys at ONE d -- four scalar loads per key tile, strided by one token.
                    # Paid ONCE per workgroup (the key block is what a KV-outer role owns),
                    # against seq_len/QUERY_TILE streamed iterations.
                    if const_expr(DQ_HALF):
                        # wave w: query half w % 2 of d tile w // 2
                        dq_qsub = wave & fx.Int32(1)
                        dq_d_tile = wave >> 1
                    else:
                        dq_d_tile = wave  # this wave's d tile of the dQ output
                    kt_d = dq_d_tile * 16 + lane_row
                    kt_base = (
                        (seq_lo + key_block_base + lane_k) * Hn + head
                    ) * DQK + kt_d
                    kt_step = Hn * DQK
                    kt_ok = kt_d < fx.Int32(DQK) if KT_MASK else None
                    kt_frags = []
                    for kt in range_constexpr(N_KEY_TILES):
                        vals = []
                        for i in range_constexpr(4):
                            idx = kt_base + (kt * 16 + i) * kt_step
                            if const_expr(KT_MASK):
                                idx = kt_ok.select(idx, oob_qk)
                            vals.append(_ld1(gK1, idx, fx.BFloat16))
                        kt_frags.append(
                            fx.Vector.from_elements(vals, dtype=fx.BFloat16)
                        )
                    # ---- dQ staging ------------------------------------------------------
                    # The dQ pass packs its bf16 tile into an LDS staging tile and the NEXT
                    # iteration flushes it with every thread adding a pair of whole-row
                    # neighbours, instead of each wave atomically adding its own 16-d column
                    # slab (twice the requests, one 32 B sector each).
                    #
                    # Pack side: `_dqpack` hands out the PAIR (lane_row & ~1, +1) of d for the
                    # query row `lane_k + 2j + (lane_row & 1)`; the pair base and the row
                    # parity are lane-varying but loop-invariant.
                    dq_par = lane_row & fx.Int32(1)
                    # v_perm byte selector that puts the lane's OWN value on its own d: even
                    # lane -> own in the LOW half, odd lane -> own in the HIGH half.
                    dq_sel = (dq_par > 0).select(
                        fx.Int32(0x03020706), fx.Int32(0x07060302)
                    )
                    dq_odd = dq_par > 0
                    dq_ptr = _global_bf16_ptr(DQO)
                    dqs_w = [
                        (lane_k + 2 * j + dq_par) * LD_DQS
                        + dq_d_tile * 16
                        + (lane_row & fx.Int32(~1))
                        for j in range_constexpr(2)
                    ]
                    if const_expr(DQPACK2):
                        dq_sel2 = dq_odd.select(
                            fx.Int32(_PERM_DQ_ODD), fx.Int32(_PERM_DQ_EVEN)
                        )
                    if const_expr(DQ_HALF):
                        # the wave's query half folded into the pack address and into its
                        # dS A-fragment row
                        dqs_w = [w + dq_qsub * (16 * LD_DQS) for w in dqs_w]
                        dq_ds_rd = (dq_qsub * 16 + lane_row) * LD_DS + lane_k
                    # Flush side, loop-invariant: pass p's (row, pair) of this thread.  A
                    # thread past the tile in a partial last pass reads row 0 and adds 0.
                    dqs_row, dqs_rd, dqs_g, dqs_g0, dqs_ok = [], [], [], [], []
                    for p in range_constexpr(DQS_PASSES):
                        f = tid + p * NUM_THREADS
                        ok = None
                        if const_expr((p + 1) * NUM_THREADS > DQS_TOTAL):
                            ok = f < fx.Int32(DQS_TOTAL)
                            f = ok.select(f, tid % DQS_PAIRS)
                        r = f // DQS_PAIRS
                        c = (f % DQS_PAIRS) * 2
                        dqs_row.append(r)
                        dqs_rd.append(r * LD_DQS + c)
                        dqs_g0.append(head * DQK + c)
                        dqs_g.append(r * kt_step + dqs_g0[-1])
                        dqs_ok.append(ok)
                    dqs_zero = fx.Vector.from_elements(
                        [fx.Int32(0)], dtype=fx.Int32
                    ).bitcast(fx.BFloat16)

                    def _dq_flush(tile, keep):
                        """Atomically add the staged dQ tile of query tile `tile` into dq.

                        `keep` False (a traced bool, or None for always) adds +0.0 instead --
                        the first iteration has no staged tile yet, and issuing the same
                        atomics anyway keeps the loop's vmcnt bookkeeping uniform.  A row past
                        the end of the sequence is steered onto the sequence's first row: its
                        staged value is exactly +0.0 (the causal predicate zeroes its dS), and
                        `global_atomic_pk_add_bf16` has no buffer-resource bounds check.
                        """
                        tile_g = (seq_lo + tile) * kt_step
                        for p in range_constexpr(DQS_PASSES):
                            v2 = fx.ptr_load(
                                lds_dqs + dqs_rd[p],
                                result_type=fx.Vector.make_type(2, fx.BFloat16),
                            )
                            if const_expr(keep is not None):
                                v2 = keep.select(v2, dqs_zero)
                            if const_expr(dqs_ok[p] is not None):
                                v2 = dqs_ok[p].select(v2, dqs_zero)
                            idx = ((tile + dqs_row[p]) < seq_len).select(
                                tile_g + dqs_g[p],
                                seq_lo * kt_step + dqs_g0[p],
                            )
                            _atompk2(v2, dq_ptr, idx)

                    def _dq_pass():
                        """dQ = dS . K^T for the streamed query tile, packed to bf16 into the
                        LDS staging tile that the next iteration's `_dq_flush` adds into dq.

                        The 8 waves partition the output along d (wave w owns d tile w), so ONE
                        wave contracts over ALL KEYS_PER_WG keys of the block and its dQ tile is
                        COMPLETE the moment it is formed -- no cross-wave reduction, and the
                        only accumulation left is the one across key blocks that the atomics do.
                        """
                        fx.barrier()

                        def _body():
                            for q_sub in range_constexpr(QUERY_TILE // 16):
                                chains = [_zero4() for _ in range_constexpr(DQ_CHAINS)]
                                for kt in range_constexpr(N_KEY_TILES):
                                    c = kt % DQ_CHAINS
                                    chains[c] = _mfma(
                                        _bf16x4(
                                            lds_ds
                                            + (
                                                (q_sub * 16 + lane_row) * LD_DS
                                                + kt * 16
                                                + lane_k
                                            )
                                        ),
                                        kt_frags[kt],
                                        chains[c],
                                    )
                                dq_a = chains[0]
                                for c in range_constexpr(1, DQ_CHAINS):
                                    dq_a = (
                                        fx.Vector(dq_a) + fx.Vector(chains[c])
                                    ).ir_value()
                                packed = (
                                    _dqpack2(dq_a, dq_sel2)
                                    if const_expr(DQPACK2)
                                    else _dqpack(dq_a, dq_sel, dq_odd)
                                )
                                for j, v2 in enumerate(packed):
                                    fx.ptr_store(
                                        v2, lds_dqs + (q_sub * 16 * LD_DQS + dqs_w[j])
                                    )

                        def _body_half():
                            """dQ of this wave's query half for its one d tile."""
                            chains = [_zero4() for _ in range_constexpr(DQ_CHAINS)]
                            for kt in range_constexpr(N_KEY_TILES):
                                c = kt % DQ_CHAINS
                                chains[c] = _mfma(
                                    _bf16x4(lds_ds + (dq_ds_rd + kt * 16)),
                                    kt_frags[kt],
                                    chains[c],
                                )
                            dq_a = chains[0]
                            for c in range_constexpr(1, DQ_CHAINS):
                                dq_a = (
                                    fx.Vector(dq_a) + fx.Vector(chains[c])
                                ).ir_value()
                            packed = (
                                _dqpack2(dq_a, dq_sel2)
                                if const_expr(DQPACK2)
                                else _dqpack(dq_a, dq_sel, dq_odd)
                            )
                            for j, v2 in enumerate(packed):
                                fx.ptr_store(v2, lds_dqs + dqs_w[j])

                        if const_expr(DQ_HALF):
                            if const_expr(2 * N_TILES_QK == NUM_WAVES):
                                # Branch-free: fence the pass off into its own scheduling
                                # region, as the per-wave branch did, so the machine scheduler
                                # does not stretch live ranges across the whole body.
                                fx.rocdl.sched_barrier(0)
                                _body_half()
                                fx.rocdl.sched_barrier(0)
                            else:
                                if wave < fx.Int32(2 * N_TILES_QK):
                                    _body_half()
                        elif const_expr(DQ_ALL_WAVES):
                            _body()
                        else:
                            if wave < fx.Int32(N_TILES_QK):
                                _body()

                # software-pipelined staging
                def _load_query_tile(q0):
                    frags = []
                    row = (seq_lo + q0 + stage_row) * Hn + head
                    for d_chunk in range_constexpr(N_QK_CHUNKS):
                        frags.append(
                            _ldv(
                                gQ,
                                _stage_idx(
                                    row * FRAGS_PER_ROW_QK
                                    + stage_col_frag
                                    + d_chunk * 16,
                                    d_chunk,
                                    PART_CHUNK_QK,
                                    stage_col_ok_qk,
                                    oob_frag_qk,
                                ),
                                fx.BFloat16,
                                4,
                            )
                        )
                    for d_chunk in range_constexpr(N_V_CHUNKS):
                        frags.append(
                            _ldv(
                                gDO,
                                _stage_idx(
                                    row * FRAGS_PER_ROW_V
                                    + stage_col_frag
                                    + d_chunk * 16,
                                    d_chunk,
                                    PART_CHUNK_V,
                                    stage_col_ok_v,
                                    oob_frag_v,
                                ),
                                fx.BFloat16,
                                4,
                            )
                        )
                    frags.append(
                        _ld1(gLSE, head * Tlen + seq_lo + q0 + stage_tok, fx.Float32)
                    )
                    frags.append(
                        _ld1(gDEL, head * Tlen + seq_lo + q0 + stage_tok, fx.Float32)
                    )
                    return frags

                N_Q_CHUNKS = N_QK_CHUNKS
                N_DO_CHUNKS = N_V_CHUNKS
                N_PREFETCH_DKDV = N_Q_CHUNKS + N_DO_CHUNKS + 2
                if const_expr(n_split == 1):
                    q_lo = key_block_base
                    q_hi = seq_len
                else:
                    # Take chunk `dkdv_split` of this block's full streamed range
                    # [key_block_base, seq_len), cut on QUERY_TILE boundaries so the causal mask
                    # logic is untouched.
                    q_ntiles = (
                        seq_len - key_block_base + (QUERY_TILE - 1)
                    ) // QUERY_TILE
                    q_lo = (
                        key_block_base
                        + ((q_ntiles * dkdv_split) // n_split) * QUERY_TILE
                    )
                    q_hi_split = (
                        key_block_base
                        + ((q_ntiles * (dkdv_split + 1)) // n_split) * QUERY_TILE
                    )
                    q_hi = (q_hi_split < seq_len).select(q_hi_split, seq_len)
                init_state = [
                    _zero4() for _ in range_constexpr(N_TILES_QK + N_TILES_V)
                ] + _load_query_tile(q_lo)
                out_state = init_state
                # Fused-build trims (module notes at OPT_FLSE).  All compile-time: the two-role
                # build sees FLSE = LEAN = False and traces exactly the body it traced before.
                if const_expr(FLSE):
                    # scale*LOG2E is a VALU product of a kernel argument; readfirstlane puts
                    # the (uniform) value back in an SGPR so the splat costs no VGPR pair.
                    sl2e_s = fx.Float32(
                        fx.rocdl.readfirstlane(
                            fx.Float32.ir_type, scale_log2e.ir_value()
                        )
                    )
                    sl2e_2 = _v2(sl2e_s, sl2e_s)
                    scale_2 = _v2(scale, scale)
                    neg_scale = fx.Float32(0.0) - scale
                if const_expr(LEAN):
                    # Pass p's (row, pair).  When the passes step whole rows (DQS_PAIRS divides
                    # NUM_THREADS and no pass is partial) pass p is pass 0 shifted by a UNIFORM
                    # row step, so ONE set of lane values serves every pass and the per-pass
                    # delta is scalar / an LDS immediate.
                    DQS_ROWSTEP = (
                        NUM_THREADS // DQS_PAIRS
                        if NUM_THREADS % DQS_PAIRS == 0
                        and all(o is None for o in dqs_ok)
                        else 0
                    )
                    NLANEV = 1 if DQS_ROWSTEP else DQS_PASSES
                    fl_row = [dqs_row[p] for p in range_constexpr(NLANEV)]
                    fl_rd = [dqs_rd[p] for p in range_constexpr(NLANEV)]
                    # byte offset of this lane's pair inside row 0 of the sequence's head slab
                    fl_hc = [dqs_g0[p] * 2 for p in range_constexpr(NLANEV)]
                    fl_gb = [
                        fl_row[p] * (kt_step * 2) + fl_hc[p]
                        for p in range_constexpr(NLANEV)
                    ]
                    # Zero the staging tile once, so a flush with nothing staged before it (the
                    # first trip, an empty split chunk's final flush) adds +0.0 without a per-trip `keep` select.  The loop-top
                    # barrier (or the post-loop one) orders it before any read.
                    for p in range_constexpr(DQS_PASSES):
                        fx.ptr_store(dqs_zero, lds_dqs + dqs_rd[p])
                    dq_base64 = fx.Int64(fx.ptrtoint(fx.get_iter(DQO)))

                    def _dq_flush_saddr(tile):
                        """`_dq_flush` without `keep` (the staging tile starts zeroed) and with
                        the address as the uniform tensor base plus a ZERO-EXTENDED 32-bit byte
                        offset (dq is < 2 GiB, the launcher asserts it), which the backend
                        selects as the global `saddr` form: no 64-bit VALU math per atomic.
                        """
                        seq_b = seq_lo * (kt_step * 2)
                        tile_b = (seq_lo + tile) * (kt_step * 2)
                        if const_expr(FLBATCH):
                            # every read first: the atomics below are ordered memory ops the
                            # scheduler will not hoist a read over
                            fl_v2 = [
                                fx.ptr_load(
                                    lds_dqs
                                    + (
                                        fl_rd[0 if DQS_ROWSTEP else p]
                                        + p * DQS_ROWSTEP * LD_DQS
                                    ),
                                    result_type=fx.Vector.make_type(2, fx.BFloat16),
                                )
                                for p in range_constexpr(DQS_PASSES)
                            ]
                        for p in range_constexpr(DQS_PASSES):
                            lp = 0 if DQS_ROWSTEP else p
                            dr = p * DQS_ROWSTEP  # uniform row delta of this pass
                            voff = (tile_b + dr * (kt_step * 2)) + fl_gb[lp]
                            voff = ((tile + dr + fl_row[lp]) < seq_len).select(
                                voff, seq_b + fl_hc[lp]
                            )
                            if const_expr(FLBATCH):
                                v2 = fl_v2[p]
                            else:
                                v2 = fx.ptr_load(
                                    lds_dqs + (fl_rd[lp] + dr * LD_DQS),
                                    result_type=fx.Vector.make_type(2, fx.BFloat16),
                                )
                            if const_expr(dqs_ok[p] is not None):
                                w = fx.Vector(v2).bitcast(fx.Int32)[0]
                                w = dqs_ok[p].select(w, fx.Int32(0))
                                v2 = fx.Vector.from_elements(
                                    [w], dtype=fx.Int32
                                ).bitcast(fx.BFloat16)
                            _mlir_llvm.atomicrmw(
                                _LLVM_FADD,
                                fx.to_llvm_ptr(
                                    fx.inttoptr(
                                        fx.PointerType.get(
                                            fx.BFloat16.ir_type,
                                            fx.AddressSpace.Global,
                                            4,
                                        ),
                                        dq_base64
                                        + fx.Int64(
                                            fx.arith.extui(
                                                fx.Int64.ir_type, voff.ir_value()
                                            )
                                        ),
                                    )
                                ),
                                fx.Vector(v2).ir_value(),
                                _LLVM_MONOTONIC,
                                syncscope="agent",
                                alignment=4,
                            )

                def _q_iter(state, q_base):
                    """One streamed query tile."""
                    dk_acc = [state[i] for i in range_constexpr(N_TILES_QK)]
                    dv_acc = [state[N_TILES_QK + i] for i in range_constexpr(N_TILES_V)]
                    staged = [
                        state[N_TILES_QK + N_TILES_V + i]
                        for i in range_constexpr(N_PREFETCH_DKDV)
                    ]

                    fx.barrier()
                    for d_chunk in range_constexpr(N_Q_CHUNKS):
                        d_col = stage_col + d_chunk * 64
                        q_vec = fx.Vector(staged[d_chunk])
                        fx.ptr_store(q_vec, lds_q + (stage_row * LD_QK + d_col))
                        for elem in range_constexpr(4):
                            fx.ptr_store(
                                q_vec[elem],
                                lds_qt
                                + ((d_col + elem) * LD_TOK_DKDV + _store_col(d_chunk)),
                            )
                    for d_chunk in range_constexpr(N_DO_CHUNKS):
                        d_col = stage_col + d_chunk * 64
                        do_vec = fx.Vector(staged[N_Q_CHUNKS + d_chunk])
                        fx.ptr_store(do_vec, lds_do + (stage_row * LD_V + d_col))
                        for elem in range_constexpr(4):
                            fx.ptr_store(
                                do_vec[elem],
                                lds_dot
                                + ((d_col + elem) * LD_TOK_DKDV + _store_col(d_chunk)),
                            )
                    if const_expr(FLSE):
                        # -lse*LOG2E: the score chain's addend (one v_fma per element).
                        fx.ptr_store(
                            fx.Float32(staged[N_Q_CHUNKS + N_DO_CHUNKS])
                            * fx.Float32(-LOG2E),
                            lds_lse + stage_tok,
                        )
                    elif const_expr(OPT_ASCALE):
                        # LOG2E folded onto lse ONCE per staged row (1 VALU/thread/iteration)
                        # instead of once per score element (8/thread/iteration).
                        fx.ptr_store(
                            fx.Float32(staged[N_Q_CHUNKS + N_DO_CHUNKS])
                            * fx.Float32(LOG2E),
                            lds_lse + stage_tok,
                        )
                    else:
                        fx.ptr_store(
                            staged[N_Q_CHUNKS + N_DO_CHUNKS], lds_lse + stage_tok
                        )
                    if const_expr(OPT_DFOLD):
                        # NEGATED once per staged row (1 VALU/thread/iteration) so the dP
                        # accumulator can be SEEDED with it: `dP - delta` then costs nothing per
                        # score element and one node leaves the dP -> dS dependency chain.
                        fx.ptr_store(
                            fx.Float32(0.0)
                            - fx.Float32(staged[N_Q_CHUNKS + N_DO_CHUNKS + 1]),
                            lds_delta + stage_tok,
                        )
                    elif const_expr(FLSE):
                        # -delta*scale: dS = P * fma(dP, scale, -delta*scale).
                        fx.ptr_store(
                            fx.Float32(staged[N_Q_CHUNKS + N_DO_CHUNKS + 1])
                            * neg_scale,
                            lds_delta + stage_tok,
                        )
                    else:
                        fx.ptr_store(
                            staged[N_Q_CHUNKS + N_DO_CHUNKS + 1], lds_delta + stage_tok
                        )
                    fx.barrier()
                    if const_expr(LEAN):
                        # Staging tile pre-zeroed: the first trip adds +0.0 without `keep`.
                        not_first = q_base > q_lo
                        _dq_flush_saddr(not_first.select(q_base - QUERY_TILE, q_lo))
                    elif const_expr(FUSED):
                        # The previous iteration's dQ tile, BEFORE this iteration's prefetch so
                        # the loop-top vmcnt waits never have to cover these atomics' latency.
                        not_first = q_base > q_lo
                        _dq_flush(
                            not_first.select(q_base - QUERY_TILE, q_lo), not_first
                        )
                    # prefetch for the next iteration
                    next_tile = _load_query_tile(q_base + QUERY_TILE)
                    if const_expr(OPT_PINVMEM):
                        _pin_vmem()

                    # this lane's key column (C-frag n index)
                    key_col = key_block_base + key_in_blk
                    # per-iteration base of the unsigned causal+tail test (see OPT_AMASK)
                    mask_base = (q_base + lane_k) - key_col
                    keys_to_end = seq_len - key_col  # loop-invariant; LICM hoists it
                    if const_expr(FUSED):
                        # A key PAST the end of the sequence makes `keys_to_end` negative, and
                        # the unsigned compare below then reads it as ~2^32 and lets every
                        # score through.  The two-role kernel does not care (that key's dK/dV
                        # store is steered past num_records anyway), but the fused role feeds
                        # its dS into the LDS exchange tile, where it would pollute the dQ of
                        # every real query.  Clamping here is one loop-invariant v_max.
                        keys_to_end = (keys_to_end > 0).select(keys_to_end, fx.Int32(0))
                    for q_sub in range_constexpr(QUERY_TILE // 16):
                        # OPT_HOIST: the lse / delta LDS reads are issued ABOVE both MFMA
                        # chains, so their latency overlaps the chains instead of being exposed
                        # after them, and under OPT_DFOLD the delta fragment IS the dP
                        # accumulator's seed register (no copy -- freshly loaded VGPRs).
                        def _lse4():
                            return fx.ptr_load(
                                lds_lse + (q_sub * 16 + lane_k),
                                result_type=fx.Vector.make_type(4, fx.Float32),
                            )

                        def _delta4():
                            return fx.ptr_load(
                                lds_delta + (q_sub * 16 + lane_k),
                                result_type=fx.Vector.make_type(4, fx.Float32),
                            )

                        if const_expr(OPT_HOIST):
                            lse4 = _lse4()
                            delta4 = _delta4()

                        def _a_q(d_tile):
                            return _bf16x4(
                                lds_q
                                + (
                                    (q_sub * 16 + lane_row) * LD_QK
                                    + d_tile * 16
                                    + lane_k
                                )
                            )

                        def _a_do(d_tile):
                            return _bf16x4(
                                lds_do
                                + (
                                    (q_sub * 16 + lane_row) * LD_V
                                    + d_tile * 16
                                    + lane_k
                                )
                            )

                        # `delta4` holds -delta under OPT_DFOLD and its element axis IS the
                        # accumulator's element axis (both are the query q_sub*16 + lane_k + i),
                        # so v_mfma's arbitrary C operand absorbs the subtraction.
                        s_acc = _zero4()
                        dp_acc = delta4 if const_expr(OPT_DFOLD) else _zero4()
                        if const_expr(OPT_XSDP):
                            # S = Q.K^T and dP = dO.V^T are two INDEPENDENT serial MFMA chains
                            # of N_TILES_QK / N_TILES_V accumulate steps.  Emitted one after the
                            # other each chain stalls on its own previous MFMA; round-robined,
                            # each fills the other's accumulate latency.  Pure emission order --
                            # identical instruction census, identical arithmetic.
                            for d_tile in range_constexpr(max(N_TILES_QK, N_TILES_V)):
                                if const_expr(d_tile < N_TILES_QK):
                                    s_acc = _mfma(_a_q(d_tile), k_frags[d_tile], s_acc)
                                if const_expr(d_tile < N_TILES_V):
                                    dp_acc = _mfma(
                                        _a_do(d_tile), v_frags[d_tile], dp_acc
                                    )
                        else:
                            for d_tile in range_constexpr(N_TILES_QK):
                                s_acc = _mfma(_a_q(d_tile), k_frags[d_tile], s_acc)
                            for d_tile in range_constexpr(N_TILES_V):
                                dp_acc = _mfma(_a_do(d_tile), v_frags[d_tile], dp_acc)

                        if const_expr(not OPT_HOIST):
                            lse4 = _lse4()
                            delta4 = _delta4()

                        p_vals = []
                        ds_vals = []
                        if const_expr(FLSE):
                            # Element pairs (0,1) / (2,3) are register-aligned halves of the
                            # accumulators and of the ds_read_b128 lse / delta fragments, so
                            # each <2 x float> fma / mul lowers to one v_pk_* with no moves.
                            sv = fx.Vector(s_acc)
                            dpv = fx.Vector(dp_acc)
                            lv = fx.Vector(lse4)
                            tv = fx.Vector(delta4)
                            for h in range_constexpr(2):
                                e0 = 2 * h
                                e1 = 2 * h + 1
                                lg = fx.Vector(
                                    fx.fma(
                                        _v2(sv[e0], sv[e1]), sl2e_2, _v2(lv[e0], lv[e1])
                                    )
                                )
                                pp = [_exp2(lg[0]), _exp2(lg[1])]
                                for k in range_constexpr(2):
                                    in_causal = fx.Uint32(
                                        mask_base + (q_sub * 16 + e0 + k)
                                    ) < fx.Uint32(keys_to_end)
                                    pp[k] = in_causal.select(pp[k], fx.Float32(0.0))
                                tt = fx.Vector(
                                    fx.fma(
                                        _v2(dpv[e0], dpv[e1]),
                                        scale_2,
                                        _v2(tv[e0], tv[e1]),
                                    )
                                )
                                ds2 = fx.Vector(_v2(pp[0], pp[1]) * tt)
                                p_vals += pp
                                ds_vals += [ds2[0], ds2[1]]
                        for i in range_constexpr(0 if FLSE else 4):
                            if const_expr(OPT_AMASK):
                                in_causal = fx.Uint32(
                                    mask_base + (q_sub * 16 + i)
                                ) < fx.Uint32(keys_to_end)
                            else:
                                query_idx = q_base + q_sub * 16 + lane_k + i
                                in_causal = (query_idx < seq_len) & (
                                    key_col <= query_idx
                                )
                            if const_expr(OPT_ASCALE):
                                # lse already carries LOG2E (folded at the LDS store): one v_fma.
                                logit = _elem(s_acc, i) * scale_log2e - _elem(lse4, i)
                            else:
                                logit = (
                                    _elem(s_acc, i) * scale - _elem(lse4, i)
                                ) * fx.Float32(LOG2E)
                            p_raw = _exp2(logit)
                            # Both mask forms are fx predicates, so one select serves either.
                            p_val = in_causal.select(p_raw, fx.Float32(0.0))
                            dp_i = (
                                _elem(dp_acc, i)
                                if const_expr(OPT_DFOLD)
                                else _elem(dp_acc, i) - _elem(delta4, i)
                            )
                            if const_expr(OPT_ASCALE):
                                # `scale` rides on the dK epilogue instead (dK = dS^T.Q is
                                # linear in it; dV = P^T.dO must NOT be scaled).
                                ds_val = p_val * dp_i
                            else:
                                ds_val = p_val * dp_i * scale
                            p_vals.append(p_val)
                            ds_vals.append(ds_val)
                        p_frag = _pack4_trunc(p_vals)  # == P^T operand (row = j, k = i)
                        # == dS^T operand (row = j, k = i)
                        ds_frag = _pack4_trunc(ds_vals)

                        for d_tile in range_constexpr(N_TILES_V):
                            do_t_frag = _bf16x4(
                                lds_dot
                                + (
                                    (d_tile * 16 + lane_row) * LD_TOK_DKDV
                                    + _read_col(d_tile, q_sub)
                                )
                            )
                            dv_acc[d_tile] = _mfma(p_frag, do_t_frag, dv_acc[d_tile])
                        for d_tile in range_constexpr(N_TILES_QK):
                            q_t_frag = _bf16x4(
                                lds_qt
                                + (
                                    (d_tile * 16 + lane_row) * LD_TOK_DKDV
                                    + _read_col(d_tile, q_sub)
                                )
                            )
                            dk_acc[d_tile] = _mfma(ds_frag, q_t_frag, dk_acc[d_tile])

                        if const_expr(FUSED):
                            # dS hand-off.  The accumulator orientation is C[query][key] with
                            # the KEY on the lane axis, so dQ -- which contracts over the key --
                            # cannot read dS out of registers.  It goes through LDS exactly
                            # once per (query tile, key tile), TRANSPOSED on the write side:
                            # four `ds_write_b16` per lane, 16 consecutive keys per row group,
                            # which is the 128 B/cycle LDS data floor for half-dword stores.
                            ds_b = fx.Vector(ds_frag)
                            for i in range_constexpr(4):
                                fx.ptr_store(
                                    ds_b[i],
                                    lds_ds
                                    + ((q_sub * 16 + lane_k + i) * LD_DS + key_in_blk),
                                )

                    # 2 ds_read fragments per MFMA pair over the whole body; the group sizes
                    # are swept on THIS baseline (see report), never copied from a prior one.
                    # The FUSED role carries dQ's MFMAs and the dS ds_writes in the same
                    # region, so it gets its own round-robin constant, and that constant
                    # depends on whether the dQ pass is dense (every wave owns a d tile).
                    _sched_pipe(
                        2 * (2 * (N_TILES_QK + N_TILES_V) + 2),
                        4 * (N_TILES_QK + N_TILES_V),
                        SCHED_RM_JOB_A,
                    )
                    if const_expr(FUSED):
                        _dq_pass()

                    return dk_acc + dv_acc + next_tile

                for q_base_raw, state in range(q_lo, q_hi, QUERY_TILE, init=init_state):
                    out_state = yield _q_iter(state, fx.Int32(q_base_raw))

                if const_expr(LEAN):
                    # The last staged dQ tile.  The staging tile was pre-zeroed, so an EMPTY
                    # split chunk (fewer query tiles than splits) adds +0.0; its tile index is
                    # clamped to q_lo and the row check steers anything past the end.
                    fx.barrier()
                    last_tile = q_lo + ((q_hi - q_lo - 1) // QUERY_TILE) * QUERY_TILE
                    if const_expr(n_split > 1):
                        last_tile = (q_hi > q_lo).select(last_tile, q_lo)
                    _dq_flush_saddr(last_tile)
                elif const_expr(FUSED):
                    # The last iteration's staged dQ tile.
                    fx.barrier()
                    if const_expr(n_split == 1):
                        _dq_flush(
                            q_lo + ((q_hi - q_lo - 1) // QUERY_TILE) * QUERY_TILE, None
                        )
                    else:
                        # A split chunk can be EMPTY (fewer query tiles than splits): the loop
                        # then never ran, nothing was staged, and the flush must add +0.0.
                        # The tile index is clamped here, outside the loop, so it stays in the
                        # sequence whatever the trip count.
                        has_trips = q_hi > q_lo
                        last_tile = has_trips.select(
                            q_lo + ((q_hi - q_lo - 1) // QUERY_TILE) * QUERY_TILE, q_lo
                        )
                        _dq_flush(last_tile, has_trips)

                # epilogue: C fragments are (m = key row, n = d)
                for i in range_constexpr(4):
                    key_idx = key_block_base + wave * 16 + lane_k + i
                    in_seq = key_idx < seq_len
                    out_row = seq_lo + key_idx
                    if const_expr(n_split == 1):
                        dk_base = in_seq.select(
                            (out_row * Hn + head) * DQK + lane_row, oob_qk
                        )
                        dv_base = in_seq.select(
                            (out_row * Hn + head) * DV + lane_row, oob_v
                        )
                    else:
                        # Partial into slab `dkdv_split`.  Every (token, d, split) slot is written
                        # by exactly one workgroup, so the workspace needs no zeroing.
                        dk_base = in_seq.select(
                            dkdv_split * slab_qk
                            + (out_row * Hn + head) * DQK
                            + lane_row,
                            oob_qk,
                        )
                        dv_base = in_seq.select(
                            dkdv_split * slab_v + (out_row * Hn + head) * DV + lane_row,
                            oob_v,
                        )
                    for d_tile in range_constexpr(N_TILES_QK):
                        _st1(
                            _to_bf16(_elem(out_state[d_tile], i)),
                            gDK,
                            _out_base(
                                dk_base,
                                d_tile,
                                N_TILES_QK,
                                REM_QK,
                                lane_row_ok_qk,
                                oob_qk,
                            ),
                            fx.BFloat16,
                        )
                    for d_tile in range_constexpr(N_TILES_V):
                        _st1(
                            _to_bf16(_elem(out_state[N_TILES_QK + d_tile], i)),
                            gDV,
                            _out_base(
                                dv_base, d_tile, N_TILES_V, REM_V, lane_row_ok_v, oob_v
                            ),
                            fx.BFloat16,
                        )
        else:
            # dQ JOB
            lds_k = lds_arena + OFF_K_NAT
            lds_kt = lds_arena + OFF_K_T
            lds_v = lds_arena + OFF_V_NAT

            # dq_blk is reversed by the grid decode: this job's cost grows with the query block
            # index, so reversing keeps its half of the list descending like the other one.
            query_block_base = dq_blk * QUERIES_PER_WG
            # EMPTY-WORKGROUP EARLY EXIT -- see the dK/dV job's note.
            if query_block_base < seq_len:

                # --- Q / dO operand fragments for this wave's 16 queries, register-resident ---
                query_in_blk = wave * 16 + lane_row
                query_idx = query_block_base + query_in_blk
                query_row = seq_lo + query_idx
                q_frag_base = (query_row * Hn + head) * FRAGS_PER_ROW_QK + lane_k_frag
                do_frag_base = (query_row * Hn + head) * FRAGS_PER_ROW_V + lane_k_frag
                q_frags = []
                for d_tile in range_constexpr(N_TILES_QK):
                    q_frags.append(
                        _ldv(
                            gQ,
                            _frag_idx(
                                q_frag_base + d_tile * 4,
                                d_tile,
                                N_TILES_QK,
                                REM_QK,
                                lane_k_ok_qk,
                                oob_frag_qk,
                            ),
                            fx.BFloat16,
                            4,
                        )
                    )
                do_frags = []
                for d_tile in range_constexpr(N_TILES_V):
                    do_frags.append(
                        _ldv(
                            gDO,
                            _frag_idx(
                                do_frag_base + d_tile * 4,
                                d_tile,
                                N_TILES_V,
                                REM_V,
                                lane_k_ok_v,
                                oob_frag_v,
                            ),
                            fx.BFloat16,
                            4,
                        )
                    )
                # Pre-scaled once per workgroup so the inner loop costs one v_fma per score
                # element instead of mul + sub + mul.
                lse_scaled = _ld1(
                    gLSE, head * Tlen + query_row, fx.Float32
                ) * fx.Float32(LOG2E)
                delta_scaled = _ld1(gDEL, head * Tlen + query_row, fx.Float32) * scale
                query_in_seq = query_idx < seq_len

                # A key j is needed iff j <= i for some owned query i, so j < query_block_base +
                # QUERIES_PER_WG, and j < seq_len.
                query_end = query_block_base + QUERIES_PER_WG
                key_end_full = (query_end < seq_len).select(query_end, seq_len)
                key_end = (query_block_base < seq_len).select(key_end_full, fx.Int32(0))

                def _load_key_tile(k0):
                    frags = []
                    row = (seq_lo + k0 + stage_row) * Hn + head
                    for d_chunk in range_constexpr(N_QK_CHUNKS):
                        frags.append(
                            _ldv(
                                gK,
                                _stage_idx(
                                    row * FRAGS_PER_ROW_QK
                                    + stage_col_frag
                                    + d_chunk * 16,
                                    d_chunk,
                                    PART_CHUNK_QK,
                                    stage_col_ok_qk,
                                    oob_frag_qk,
                                ),
                                fx.BFloat16,
                                4,
                            )
                        )
                    for d_chunk in range_constexpr(N_V_CHUNKS):
                        frags.append(
                            _ldv(
                                gV,
                                _stage_idx(
                                    row * FRAGS_PER_ROW_V
                                    + stage_col_frag
                                    + d_chunk * 16,
                                    d_chunk,
                                    PART_CHUNK_V,
                                    stage_col_ok_v,
                                    oob_frag_v,
                                ),
                                fx.BFloat16,
                                4,
                            )
                        )
                    return frags

                N_K_CHUNKS = N_QK_CHUNKS
                N_PREFETCH_DQ = N_K_CHUNKS + N_V_CHUNKS
                if const_expr(n_split == 1):
                    k_lo = fx.Int32(0)
                    k_hi = key_end
                else:
                    k_ntiles = (key_end + (KEY_TILE - 1)) // KEY_TILE
                    k_lo = ((k_ntiles * dq_split) // n_split) * KEY_TILE
                    k_hi_split = ((k_ntiles * (dq_split + 1)) // n_split) * KEY_TILE
                    k_hi = (k_hi_split < key_end).select(k_hi_split, key_end)
                init_state = [
                    _zero4() for _ in range_constexpr(N_TILES_QK)
                ] + _load_key_tile(k_lo)

                def _stream_key_tile(state, key_base, masked):
                    """One streamed key tile.  `masked` selects the causal-diagonal variant.

                    With OPT_BPEEL the caller runs the mask-free variant over the prefix of key
                    tiles that lie wholly below the causal diagonal -- all but the last <=4 trips
                    -- saving 4 VALU per score element there.
                    """
                    dq_acc = [state[i] for i in range_constexpr(N_TILES_QK)]
                    staged = [
                        state[N_TILES_QK + i] for i in range_constexpr(N_PREFETCH_DQ)
                    ]

                    fx.barrier()
                    for d_chunk in range_constexpr(N_K_CHUNKS):
                        d_col = stage_col + d_chunk * 64
                        k_vec = fx.Vector(staged[d_chunk])
                        fx.ptr_store(k_vec, lds_k + (stage_row * LD_QK + d_col))
                        for elem in range_constexpr(4):
                            fx.ptr_store(
                                k_vec[elem],
                                lds_kt
                                + ((d_col + elem) * LD_TOK_DQ + _store_col(d_chunk)),
                            )
                    for d_chunk in range_constexpr(N_V_CHUNKS):
                        d_col = stage_col + d_chunk * 64
                        fx.ptr_store(
                            staged[N_K_CHUNKS + d_chunk],
                            lds_v + (stage_row * LD_V + d_col),
                        )
                    fx.barrier()
                    # prefetch for the next iteration
                    next_tile = _load_key_tile(key_base + KEY_TILE)
                    if const_expr(OPT_PINVMEM):
                        _pin_vmem()

                    for key_sub in range_constexpr(KEY_TILE // 16):

                        def _b_k(d_tile):
                            return _bf16x4(
                                lds_k
                                + (
                                    (key_sub * 16 + lane_row) * LD_QK
                                    + d_tile * 16
                                    + lane_k
                                )
                            )

                        def _b_v(d_tile):
                            return _bf16x4(
                                lds_v
                                + (
                                    (key_sub * 16 + lane_row) * LD_V
                                    + d_tile * 16
                                    + lane_k
                                )
                            )

                        s_t_acc = _zero4()
                        dp_t_acc = _zero4()
                        if const_expr(OPT_XSDP):
                            for d_tile in range_constexpr(max(N_TILES_QK, N_TILES_V)):
                                if const_expr(d_tile < N_TILES_QK):
                                    s_t_acc = _mfma(
                                        _b_k(d_tile), q_frags[d_tile], s_t_acc
                                    )
                                if const_expr(d_tile < N_TILES_V):
                                    dp_t_acc = _mfma(
                                        _b_v(d_tile), do_frags[d_tile], dp_t_acc
                                    )
                        else:
                            for d_tile in range_constexpr(N_TILES_QK):
                                s_t_acc = _mfma(_b_k(d_tile), q_frags[d_tile], s_t_acc)
                            for d_tile in range_constexpr(N_TILES_V):
                                dp_t_acc = _mfma(
                                    _b_v(d_tile), do_frags[d_tile], dp_t_acc
                                )

                        ds_vals = []
                        for i in range_constexpr(4):
                            logit = _elem(s_t_acc, i) * scale_log2e - lse_scaled
                            p_raw = _exp2(logit)
                            if const_expr(masked):
                                key_idx = key_base + key_sub * 16 + lane_k + i
                                if const_expr(OPT_BPEEL):
                                    in_causal = key_idx <= query_idx
                                else:
                                    in_causal = query_in_seq & (key_idx <= query_idx)
                                p_val = in_causal.select(p_raw, fx.Float32(0.0))
                            else:
                                p_val = p_raw
                            ds_vals.append(
                                p_val * (_elem(dp_t_acc, i) * scale - delta_scaled)
                            )
                        # == dS operand (row = i, k = j)
                        ds_frag = _pack4_trunc(ds_vals)

                        for d_tile in range_constexpr(N_TILES_QK):
                            k_t_frag = _bf16x4(
                                lds_kt
                                + (
                                    (d_tile * 16 + lane_row) * LD_TOK_DQ
                                    + _read_col(d_tile, key_sub)
                                )
                            )
                            dq_acc[d_tile] = _mfma(ds_frag, k_t_frag, dq_acc[d_tile])
                    _sched_pipe(
                        2 * (2 * N_TILES_QK + N_TILES_V),
                        2 * (2 * N_TILES_QK + N_TILES_V),
                        _SCHED_RM_B,
                    )
                    return dq_acc + next_tile

                out_state = init_state
                if const_expr(OPT_BPEEL):
                    # peel point = query_block_base clamped into this split's [k_lo, k_hi)
                    peel_point = (query_block_base < k_hi).select(
                        query_block_base, k_hi
                    )
                    peel_point = (peel_point > k_lo).select(peel_point, k_lo)
                    # mask-free prefix: key tiles wholly below the diagonal
                    for key_base_raw, state in range(
                        k_lo, peel_point, KEY_TILE, init=init_state
                    ):
                        out_state = yield _stream_key_tile(
                            state, fx.Int32(key_base_raw), False
                        )
                    # <=4 diagonal-straddling trips keep the full predicate
                    mid_state = out_state
                    for key_base_raw, state in range(
                        peel_point, k_hi, KEY_TILE, init=mid_state
                    ):
                        out_state = yield _stream_key_tile(
                            state, fx.Int32(key_base_raw), True
                        )
                else:
                    for key_base_raw, state in range(
                        k_lo, k_hi, KEY_TILE, init=init_state
                    ):
                        out_state = yield _stream_key_tile(
                            state, fx.Int32(key_base_raw), True
                        )

                for i in range_constexpr(4):
                    query_idx_out = query_block_base + wave * 16 + lane_k + i
                    in_seq = query_idx_out < seq_len
                    out_row = seq_lo + query_idx_out
                    if const_expr(n_split == 1):
                        dq_base = in_seq.select(
                            (out_row * Hn + head) * DQK + lane_row, oob_qk
                        )
                    else:
                        dq_base = in_seq.select(
                            dq_split * slab_qk + (out_row * Hn + head) * DQK + lane_row,
                            oob_qk,
                        )
                    for d_tile in range_constexpr(N_TILES_QK):
                        _st1(
                            _to_bf16(_elem(out_state[d_tile], i)),
                            gDQ,
                            _out_base(
                                dq_base,
                                d_tile,
                                N_TILES_QK,
                                REM_QK,
                                lane_row_ok_qk,
                                oob_qk,
                            ),
                            fx.BFloat16,
                        )

    # Occupancy pin (8 waves per workgroup over 4 SIMDs = 2 waves per SIMD per workgroup).
    BWD_ATTRS = (
        {"value_attrs": {"rocdl.waves_per_eu": 2 * wgs_per_cu}}
        if wgs_per_cu > 0
        else {}
    )

    @flyc.jit
    def launch_bwd(
        Q: fx.Tensor,
        K: fx.Tensor,
        V: fx.Tensor,
        DO: fx.Tensor,
        LSE: fx.Tensor,
        DEL: fx.Tensor,
        CU: fx.Tensor,
        DQO: fx.Tensor,
        DKO: fx.Tensor,
        DVO: fx.Tensor,
        Tlen: fx.Int32,
        Hn: fx.Int32,
        scale: fx.Float32,
        n_dkdv_blocks: fx.Int32,
        n_blocks: fx.Int32,
        nseq: fx.Int32,
        interleave: fx.Int32,
        stream: fx.Stream,
    ):
        k_bwd(
            Q,
            K,
            V,
            DO,
            LSE,
            DEL,
            CU,
            DQO,
            DKO,
            DVO,
            Tlen,
            Hn,
            nseq,
            n_dkdv_blocks,
            n_blocks,
            scale,
            interleave,
            **BWD_ATTRS,
        ).launch(
            grid=(nseq * Hn, n_blocks, 1), block=(NUM_THREADS, 1, 1), stream=stream
        )

    if const_expr(n_split == 1):
        return launch_delta, launch_bwd

    # ------------------------------------------ split-K reduction (n_split > 1 variant only)
    @flyc.kernel(known_block_size=[RED_THREADS, 1, 1])
    def k_red(
        WS: fx.Tensor,
        OQ: fx.Tensor,
        OK: fx.Tensor,
        OV: fx.Tensor,
        n_qk: fx.Int32,
        n_v: fx.Int32,
    ):
        """dq / dk / dv = bf16( sum_p partial_p ) in ONE dispatch; ``grid.y`` picks the tensor.

        ``WS`` is one allocation laid out ``[dq: n_split slabs of n_qk][dk: n_split slabs of
        n_qk][dv: n_split slabs of n_v]``, so the slabs a workgroup reads are pure index
        arithmetic on ``grid.y``.  Perfectly coalesced dwordx4 streams; every n is a multiple of
        RED_VEC, so a thread's 8-element group is wholly inside or wholly outside num_records.

        The three OUTPUT pointers stay separate (the caller owns dq/dk/dv), so the store is
        triplicated and steered: slot ``x`` is written at ``tile + (y-x)^2 * n_qk/RED_VEC``,
        in range only for ``x == y`` -- the other two land past num_records (n_v <= n_qk) and
        the hardware drops them.  The dv row's grid is sized for n_qk; its tiles past n_v
        likewise store out of range.
        """
        tid = fx.Int32(fx.thread_idx.x)
        bid = fx.Int32(fx.block_idx.x)
        which = fx.Int32(fx.block_idx.y)
        tiles_qk = n_qk // RED_VEC
        tiles_v = n_v // RED_VEC
        # slab stride: n_qk for dq/dk, n_v for dv (which == 2)
        if const_expr(FUSED):
            # FUSED split-K: dq was accumulated in place, so WS is [dk slabs][dv slabs] only and
            # grid.y = 2 covers dk (which 1) and dv (which 2).
            which = which + 1
            ws_elems = n_qk + n_v
            ws_which = which - 1
        else:
            ws_elems = n_qk * 2 + n_v
            ws_which = which
        is_v = which // 2
        tiles_slab = tiles_qk + is_v * (tiles_v - tiles_qk)
        gWS = _buffer_view(WS, ws_elems * (2 * n_split), fx.BFloat16, RED_VEC)
        tile = bid * RED_THREADS + tid
        base = ws_which * (n_split * tiles_qk) + tile
        partials = []
        for slab in range_constexpr(n_split):
            partials.append(
                fx.Vector(_ldv(gWS, base + tiles_slab * slab, fx.BFloat16, RED_VEC))
            )
        vals = []
        for e in range_constexpr(RED_VEC):
            acc = fx.Float32(partials[0][e])
            for slab in range_constexpr(1, n_split):
                acc = acc + fx.Float32(partials[slab][e])
            vals.append(acc)
        packed = _packn_rne(vals)
        for x in range_constexpr(1 if FUSED else 0, 3):
            off = which - fx.Int32(x)
            n_out = n_v if x == 2 else n_qk
            _stv(
                packed,
                _buffer_view([OQ, OK, OV][x], n_out * 2, fx.BFloat16, RED_VEC),
                tile + off * off * tiles_qk,
                fx.BFloat16,
                RED_VEC,
            )

    @flyc.jit
    def launch_red(
        WS: fx.Tensor,
        OQ: fx.Tensor,
        OK: fx.Tensor,
        OV: fx.Tensor,
        n_qk: fx.Int32,
        n_v: fx.Int32,
        nblk: fx.Int32,
        stream: fx.Stream,
    ):
        k_red(WS, OQ, OK, OV, n_qk, n_v).launch(
            grid=(nblk, 2 if FUSED else 3, 1), block=(RED_THREADS, 1, 1), stream=stream
        )

    return launch_delta, launch_bwd, launch_red
