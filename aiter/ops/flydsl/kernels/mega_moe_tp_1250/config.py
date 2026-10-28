# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Compile-time layout of the gfx1250 MegaMoE TP kernel: ctrl ints, peer flags, the
launch configs of the three dispatch paths."""

from __future__ import annotations

from dataclasses import dataclass

MAX_TP = 8
WAVE = 32
# a cross-rank / grid wait that runs longer than this (steady-counter ticks, 100 MHz)
# gives up and sets a sticky error bit (2 s)
DEADLINE = 200_000_000

# Local ctrl ints (one int32 array per rank, zeroed once). Counters that every CTA
# hits sit on their own 128 B line. Every counter is monotonic across launches and
# compared against epoch-derived targets (wrap-safe), so nothing is reset per launch
# except the routing counters (zeroed by CTA 0 once they are dead).
LINE = 32
CT_EPOCH = 0
CT_ERR = 1
CT_CNT = 8 * LINE  # [E] routes per expert
CT_FILL = CT_CNT + 1024  # [E] routes placed per expert
E_MAX = 1024
# Grid sync points (all CTAs arrive once per launch; a set per epoch parity, the
# other parity's zeroed by every launch for the next, so launches of kernels with
# different CTA counts may alternate): two-level arrival counters, a
# line per group of SYNC_GS CTAs plus a top counter (the groups' last arrivers) and a
# flag (the epoch, stored by the grid's last arriver) the waiters poll. Same-address
# atomics serialize in L2, so a flat counter of ~1000 CTAs costs > 10 us.
SYNC_GS = 32
SYNC_NG = 64  # max groups (NCTA <= 2048)
SYNC_TOP = SYNC_NG * LINE
SYNC_FLAG = SYNC_TOP + LINE
SYNC_INTS = SYNC_FLAG + LINE
SY_AG, SY_RS, SY_DONE = 0, 1, 2  # done pushing AG rows / RS rows; done
SY_BAR = 3  # grid barrier j uses sync point SY_BAR + j - 1
N_SYNC = 8
CT_SYNC = CT_FILL + E_MAX
CTRL_INTS = CT_SYNC + 2 * N_SYNC * SYNC_INTS  # [epoch parity][sync point]
# debug (MEGA_TS=1 at compile): per CTA, TS_N int64 steady-counter stamps at the
# phase boundaries, stored after the ctrl ints
TS_N = 24
TS_INTS = 8192 * TS_N * 2

# Peer counter ints (in the symmetric arena): every CTA that pushed rows to the ranks
# adds 1 at each rank once its rows are written back; [epoch parity] (a launch zeroes
# the other parity's for the next launch: no peer touches those before it is done)
FL_AG = 0  # + par * LINE: AG rows (+ routing) landed here
FL_RS = 2 * LINE  # + par * LINE: RS rows landed here
FL_ID = 4 * LINE  # + par * LINE: AG ids / weights landed here (ag_split)
FL_Y = 6 * LINE  # + par * LINE: reduced rows ready (AR two-step Y)
FLAG_INTS = 8 * LINE

ERR_AG, ERR_RS, ERR_BAR = 1, 2, 4  # grid barrier j (1..4) sets ERR_BAR << (j - 1)
ERR_DF = 64  # a per-m-tile wait (dataflow) timed out


def tile_counts_stride(cm: int) -> int:
    """Ints per set of per-m-tile counters (m-tiles of >= 16 rows of cm)."""
    return cm // 16 + 1


# after the per-m-tile counters: [epoch parity][AG, RS][group] writer counts
WG_INTS = 2 * 3 * SYNC_NG  # [parity][AG, RS, ID][group]


def q_off(cm: int) -> int:
    """the task queue counters ([epoch parity], a line each) after the group counts"""
    return 4 * tile_counts_stride(cm) + WG_INTS


def pl_off(cm: int) -> int:
    """[epoch parity][expert] rows placed (df0), after the queue counters"""
    return q_off(cm) + 2 * LINE


def tcnt_ints(cm: int) -> int:
    return pl_off(cm) + 2 * E_MAX


@dataclass(frozen=True)
class GemmCfg:
    """One GEMM's tile config (the gfx1250 TDM grouped GEMM)."""

    tile_m: int = 64
    tile_n: int = 256
    tile_k: int = 128
    m_warp: int = 1
    n_warp: int = 4
    num_buffers: int = 3
    next_stage_prefetch: int = 0
    # instruction-scheduling groups; 0: the tile's default (gemm_tile)
    mma_group: int = 0
    mma_first_group: int = 0
    ds_first_n: int = -1
    c_sep: int = 0  # output staging in its own LDS region; store not waited per task


@dataclass(frozen=True)
class LaunchCfg:
    """One kernel variant (a dispatch path): its GEMM tiles and CTAs per CU."""

    path: str = "mid"  # small | mid | big
    g1: GemmCfg = GemmCfg()
    g2: GemmCfg = GemmCfg()
    cta_per_cu: int = 4
    place_rpw: int = 16  # routes a wave places per batch (R2)
    df1: bool = True  # G1 -> G2 by per-m-tile completion counts, not a grid barrier
    dyn: bool = True  # G1 / G2 tiles from one atomic queue (else static round-robin)
    df2: bool = False  # G2 -> RS by per-m-tile completion counts, not a grid barrier
    local_hist: bool = False  # every CTA counts all routes (no barrier 1; small M)
    expert_sched: bool = False  # LLVM amdgpu-expert-scheduling-mode
    rs_pull: bool = False  # RS partials kept in the own arena, owners read them (Y)
    y_tdm: bool = False  # Y by TDM tiles (LDS staged); measured slower (remote 2-D TDM)
    y_unroll: int = 1  # Y units a wave loads at once (2 and 4 measured slower)
    ag_split: bool = False  # AG ids first (routing starts), rows after (placement)
    g2_q8: bool = False  # G2 route rows as MXFP8 (half the gout write + RS read)
    df0: bool = False  # placement -> G1 by per-expert placed counts, not barrier 2
    ar: bool = False  # replicated in / out (MoE + all-reduce); the small path
    ar_two: bool = False  # AR: Y as reduce-scatter (own T/tp rows) + all-gather
    spread: bool = False  # static GEMM task order spread over CUs
    waves_per_eu: int = 0  # compile hint (0: none)
    expert_hint: bool = True  # tasks get their expert from an LDS psum bisect
    rs_q8: bool = False  # RS partials as int8 + e8m0 per 32 (else bf16; no gain measured)

    @property
    def align_m(self) -> int:
        return max(self.g1.tile_m, self.g2.tile_m)
