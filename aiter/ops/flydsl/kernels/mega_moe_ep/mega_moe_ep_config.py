# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Compile-time layout, shape limits and the per-instance ``KernelCtx`` of the fused
EP MegaMoE kernel.

Mirrors :mod:`mega_moe_tp.mega_moe_tp_config`; the epoch-flag arena here carries the
EP *dispatch* / *combine* readiness epochs (an all-to-all scatter completes a receive
tile; a combine partial lands back at the source) instead of the TP all-gather /
reduce-scatter ones. The ``KernelCtx`` container itself is reused from the TP config.

The in-kernel dispatch planner (``schedule.build_schedule``) is the TP dynamic planner
adapted to expert parallelism: the active-expert space is this rank's ``EPR`` *local*
experts, and the routing it scans is the dispatched ``(token, expert)`` pairs this rank
received, not the whole all-gathered routing.
"""

from __future__ import annotations

import functools
import math

import flydsl.expr as fx

from ..kernels_common import ceildiv
from ..mega_moe_tp.common import DEADLINE, MAX_TP
from ..mega_moe_tp.mega_moe_tp_config import KernelCtx

__all__ = [
    "B_COMBINE",
    "B_DISPATCH",
    "B_LOADER",
    "CTRL_ERR",
    "CTRL_INTS",
    "C_BARCNT",
    "C_BARGEN",
    "C_CNT",
    "C_DYNP",
    "C_GEXP",
    "C_NACT",
    "C_NCH",
    "C_PLAN",
    "C_UL",
    "C_UNIT",
    "C_UROWS",
    "C_USEQ",
    "DYN_MAX",
    "ERR_COMBINE",
    "ERR_COMPUTE",
    "ERR_DISPATCH",
    "FLAG_COMBINE",
    "FLAG_COMB_CNT",
    "FLAG_DISPATCH",
    "FLAG_DISPM",
    "FLAG_DISPM_CNT",
    "FLAG_DISP_CNT",
    "FLAG_INIT",
    "FLAG_INIT_CNT",
    "FLAG_INTS",
    "GBAR_CNT",
    "GBAR_GEN",
    "MAX_NPES",
    "NCHA",
    "NCTA_MAX",
    "NT",
    "NTT",
    "NW",
    "N_XCD",
    "UL_MAX",
    "ep_shape_supported",
    "kernel_config",
    "mega_moe_ep_consts",
]

MAX_NPES = MAX_TP
NCTA_MAX = 256
N_XCD = 8

# Block geometry: one CTA per CU, eight 64-lane waves, warp-specialized by tid.
NW = 4  # compute waves
NT = NW * 64  # 256 compute threads
NTT = NT + 4 * 64  # 512 threads (8 waves): 4 compute + dispatch + loader + 2 combine

# Role bands by tid (see mega_moe_ep_kernel.roles):
#   [0, NT)                   compute_units (local-expert GEMM1 + act + GEMM2)
#   [NT, NT+64)               dispatch_send (a2a scatter tokens to expert-owner ranks)
#   [B_LOADER, B_LOADER+NLOAD) a_loader (TP-style: stage A K-tiles into the NAB-bank ring)
#   [B_LOADER+NLOAD, NTT)     idle
B_DISPATCH = NT
B_LOADER = NT + 64
B_COMBINE = NT + 128
# The dedicated A-loader occupies NLOAD_W waves starting at B_LOADER (producer of the
# NAB-bank A-ring the compute waves consume); more waves = more A-staging throughput.
NLOAD_W = 1
NLOAD = NLOAD_W * 64

# GEMM geometry (the compute waves reuse the mega_moe stage1/stage2 MFMA atoms; see
# gemm.py). The GEMM runs on the NW=4 compute waves (NT=256 threads). TILE_K is the
# MFMA K-step byte granularity; TILE_N the GEMM1 output column tile over the gate|up
# halves (2*inter_dim). FP4 activations halve the per-K-step A bytes.
KCS = 2  # A K-chunks staged together
TILE_K = 256
TILE_N = 256
# A-ring banks for the TP-style dedicated A-loader prefetch (producer/consumer depth):
# the GEMM1/GEMM2 compute waves read A K-tiles from this many LDS banks while the loader
# wave stages the next ones ahead.  >=2 (double-buffer); bounded by the LDS budget.
NAB = 4

# Up to this many received (token, expert) pairs per rank use the dynamic planner
# (the one ported here); larger batches want an LB-style two-phase planner (future).
DYN_MAX = 256
# Chunk-table slots (active-expert row chunks of 16*mt rows) the dynamic planner emits.
# Also the per-echunk capacity of the global inter/gout/rowmap/rowidx scratch. Larger
# than the #CU grid (256) so high-bs launches whose chunk count exceeds the grid run the
# GEMM/combine in multiple CTA waves (overlap room) instead of hanging on overflow.
# Covers v4_pro up to ~bs 10K/rank (nch ~= epr*ceil(R_e/RG)); raise for larger prefill.
NCHA = 1024

# Epoch-flag arena: ints in the symmetric arena ``off_flag`` region. Two banks selected
# by epoch parity so a launch never races the previous one's slots. Each readiness tile
# has an arrival *counter* (senders ``g_add_agent`` into it; the last arriver zeroes it
# and stamps the paired *ready* flag with the launch epoch) and a *ready* flag (value =
# epoch, compared wrap-safe with ``common.before``; waiters ``spin_sys_ge``). The
# counter self-resets each launch; ``reset()`` clears residue after a timeout.
#   FLAG_DISPATCH + parity*NCTA_MAX + tile   dispatch receive tile ready (epoch)
#   FLAG_DISP_CNT + parity*NCTA_MAX + tile   dispatch receive tile arrival counter
#   FLAG_COMBINE  + parity*NCTA_MAX + tile   combine partial tile ready (epoch)
#   FLAG_COMB_CNT + parity*NCTA_MAX + tile   combine partial tile arrival counter
# NOTE: per-tile granularity is the scaffold's bound; the final tile->slot mapping may
# be refined once dispatch_send fixes the receive-tile count.
FLAG_DISPATCH = 0
FLAG_DISP_CNT = FLAG_DISPATCH + 2 * NCTA_MAX
FLAG_COMBINE = FLAG_DISP_CNT + 2 * NCTA_MAX
FLAG_COMB_CNT = FLAG_COMBINE + 2 * NCTA_MAX
# FLAG_INIT + parity*NCTA_MAX + 0        post-init all-to-all barrier ready (epoch)
# FLAG_INIT_CNT + parity*NCTA_MAX + 0    post-init arrival counter (npes*n_cta)
FLAG_INIT = FLAG_COMB_CNT + 2 * NCTA_MAX
FLAG_INIT_CNT = FLAG_INIT + 2 * NCTA_MAX
# FLAG_DISPM + parity*NCTA_MAX + tile    dispatch META (routing-only) ready (epoch) --
# FLAG_DISPM_CNT + parity*NCTA_MAX + tile    its arrival counter. The routing (recv_le)
# is scattered + signalled BEFORE the payload so the owner plans (pre_pos / dyn_plan /
# gather, all routing-only) while the big payload a2a is still in flight.
FLAG_DISPM = FLAG_INIT_CNT + 2 * NCTA_MAX
FLAG_DISPM_CNT = FLAG_DISPM + 2 * NCTA_MAX
FLAG_INTS = FLAG_DISPM_CNT + 2 * NCTA_MAX

# Global control ints (device tensor): a sticky watchdog error word plus room.
CTRL_ERR = 0
ERR_DISPATCH, ERR_COMPUTE, ERR_COMBINE = 1, 2, 4
# Device grid-barrier slots (agent-scope counter + generation in the local ctrl tensor;
# the multi-pass GEMM syncs ALL CTAs of this rank between passes). Self-resetting like
# the LDS cbar: the last arriver zeroes the counter and bumps the generation. Zeroed
# once at engine init; the generation grows monotonically (wrap-safe == compare).
GBAR_CNT, GBAR_GEN = 16, 17
CTRL_INTS = 64

# LDS control slots (ints at ``L_CTL``; see schedule.py). The cbar generation barrier,
# the dynamic plan's bookkeeping and the per-CTA unit list live here.
C_BARCNT, C_BARGEN = 0, 1  # 4-compute-wave LDS generation barrier
C_NACT, C_NCH, C_PLAN, C_DYNP = 2, 3, 4, 5  # #active experts, #units, plan-ready, P
C_UROWS, C_USEQ, C_GEXP, C_CNT = 6, 7, 8, 9  # current unit rows / seq / gather cache
# slots 10-11 free (were the comm-band overlap's C_GREADY/C_COMM, now that combine is
# fused into the GEMM2 epilogue -- gemm2_combine -- at P==1).
C_UNIT = 12  # [C_UNIT, C_UNIT+1): this CTA's unit range [lo, hi)
C_UL = 16  # per-CTA unit list base (records of 4 ints: expert_chunk, i0, icnt, kind)
UL_MAX = (128 - C_UL) // 4

# A-loader ring (producer/consumer) control slots, in their OWN LDS region ``L_RCTL`` so
# the NAB per-bank counters don't crowd the unit-list area of ``L_CTL``.  The dedicated
# loader wave publishes RC_ASEQ[bank] (an A K-tile is staged) and spins on RC_AFREE[bank]
# (all NW compute waves released that bank's prior generation); the compute waves spin on
# RC_ASEQ before reading and bump RC_AFREE after consuming.  RC_G1GO/RC_G2GO are the
# per-pass go flags (compute sets them once the A source for that GEMM is live).
RC_ASEQ = 0  # [RC_ASEQ, RC_ASEQ + NAB): produced seq (monotonic tile count) per ring bank
RC_AFREE = NAB  # [RC_AFREE, RC_AFREE + NAB): consumer release count per ring bank
RC_G1GO = 2 * NAB  # GEMM1 A-stream go (compute sets it after dispatch_wait + gather)
RC_G2GO = 2 * NAB + 1  # GEMM2 A-stream go (Stage 2)
RC_INTS = 2 * NAB + 8  # ints in L_RCTL (headroom for future loader bookkeeping)

ACT_SET = ("silu", "situv2")


def ep_shape_supported(
    model_dim: int, inter_dim: int, experts: int, world_size: int
) -> bool:
    """Whether the EP kernel can tile this shape (gfx950)."""
    return (
        model_dim % 512 == 0
        and inter_dim % 128 == 0
        and inter_dim >= 256
        and experts % world_size == 0
        and 1 <= world_size <= MAX_NPES
    )


@functools.cache
def mega_moe_ep_consts(
    H: int, I: int, MT: int, EPR: int, TMAX: int, a8: bool = False
) -> dict:
    """Per-instance LDS layout (bytes): one row tile's activation + E8M0 scale +
    intermediate staging, plus the dynamic planner's tables (active-expert bitmap /
    dense list, per-local-expert counts, chunk table, wave-prefix scratch, control).

    TODO(compute): grow the A / B / scale / intermediate region into the full ring that
    ``mega_moe_tp_consts`` carries once the GEMM device code lands.
    """
    RG = MT * 16  # rows per GEMM tile (sort_block_m)
    a_row = H if a8 else H // 2  # activation bytes per row (FP8 vs packed FP4)
    a_scale = H // 32  # E8M0 scale bytes per row
    inter_row = I if a8 else I // 2
    NBW = ceildiv(EPR, 32)  # 32-bit words of the active-local-expert bitmap
    # GEMM geometry (mirrors compile_gemm1 so the stage1/stage2 atoms stage into the
    # layout they expect). FP4 halves the per-K-step A bytes; the A LDS pool
    # double-buffers the K tile (and aliases the cshuffle epilogue). The pool is sized
    # for the LARGER of the two cshuffle tiles: GEMM1 writes RG*(TILE_N/2) f32 (gate|up
    # paired), GEMM2 writes RG*TILE_N f32 (down-proj, tile_n2=TILE_N, no pairing) -- so
    # the pool uses the GEMM2 cshuffle (RG*TILE_N*4), which also fits GEMM1's.
    a_k_step = TILE_K if a8 else TILE_K // 2
    a_lds_size = RG * a_k_step
    # The A pool is JUST the NAB-bank A-ring now (the cshuffle C-tile moved to its own
    # L_C region for both GEMM1 and GEMM2, so L_A no longer needs RG*TILE_N f32 -- that
    # aliasing was the pre-loader design).  Sizing it to the ring alone reclaims the
    # (RG*TILE_N*4 - NAB*a_lds_size) bytes (~32 KB at MT4) for headroom.
    lds_pool = NAB * a_lds_size
    c = {
        "RG": RG,
        "A_ROW": a_row,
        "A_SCALE": a_scale,
        "INTER_ROW": inter_row,
        "NBW": NBW,
        "KS1": H // 128,
        "KS2": I // 128,
        "G2": H // 64 // NW,
        "CH1": ceildiv(H // 32, 8),
        "CH2": ceildiv(I // 32, 8),
        "M_REPEAT": RG // 16,
        "N_TILES": (2 * I) // TILE_N,
        "NUM_ACC_N": (TILE_N // NW) // 16,
        "A_K_STEP": a_k_step,
        "K_ITERS": H // TILE_K,
        "A_LDS_I32": a_lds_size // 4,
        "NAB": NAB,
    }
    off = 0

    def take(n: int) -> int:
        nonlocal off
        off = ceildiv(off, 128) * 128
        start, off = off, off + n
        return start

    # compute staging (reuses the stage1/stage2 GEMM atoms' expected LDS). The GEMM1
    # quantized intermediate goes to a GLOBAL buffer (like stage1->stage2), not LDS: a
    # full RG x inter tile would blow the 160 KB budget for FP8. The cshuffle epilogue
    # aliases the A pool (compile_gemm1 pattern); GEMM2 reads the intermediate from
    # global. TODO(tune): keep it per-inter-piece in LDS to drop the global round-trip.
    c["L_A"] = take(lds_pool)  # A NAB-bank ring (GEMM1 + GEMM2 loader); cshuffle is in L_C
    c["L_AS"] = take(RG * a_scale)  # A-scale (E8M0) of one row tile (compute-side stage)
    # GEMM1's cshuffle C-tile gets its OWN region: the A-loader runs ahead into the L_A
    # ring during the epilogue, so the C-tile can no longer alias L_A (it still aliases
    # L_A in GEMM2, which has no loader yet).  Sized for the larger GEMM2 cshuffle too.
    c["L_C"] = take(RG * TILE_N * 4)
    c["L_RCTL"] = take(RC_INTS * 4)  # A-loader ring counters (RC_* slots)
    # dynamic planner tables
    c["L_DYN"] = take((2 * NBW + EPR) * 4)  # bitmap words | prefix words | dense list
    c["L_DCNT"] = take(EPR * 4)  # received rows per local expert
    c["L_WSUM"] = take(NW * 2 * 4)  # wave-prefix scratch (count + offset)
    c["L_DCH"] = take(NCHA * 4)  # chunk table: expert | ce<<16 | c<<24
    c["L_EOFF"] = take(EPR * 4)  # per-expert row offset (gather)
    c["L_RIX"] = take(
        RG * 4
    )  # gathered source_key (payload row) list of one unit chunk
    c["L_CTL"] = take(128 * 4)  # control slots (C_* above)
    c["LDS_BYTES"] = ceildiv(off, 128) * 128
    return c


def kernel_config(
    *,
    H: int,
    I: int,
    TOPK: int,
    MT: int,
    TMAX: int,
    E: int,
    EPR: int,
    NPES: int,
    act: str = "silu",
    situ_beta: float = 1.0,
    situ_linear_beta: float = 1.0,
    swiglu_limit: float | None = None,
    a8: bool = False,
    indexed_payload: bool = False,
):
    """Derived compile-time values of one ``compile_mega_moe_ep`` instance.

    ``indexed_payload`` mirrors the dispatch fix: at the indexed-capable MTPR class
    every rank receives payload keyed by source token (row = rank*mtpr + token,
    bounded by npes*mtpr), overflow-proof under any cross-rank routing skew.
    """
    if act not in ACT_SET:
        raise ValueError(f"mega_moe_ep act must be one of {ACT_SET}, got {act!r}")
    if swiglu_limit is None:
        swiglu_limit = float("inf")
    c = mega_moe_ep_consts(H, I, MT, EPR, TMAX, a8)
    RG, LDS_BYTES, NBW = c["RG"], c["LDS_BYTES"], c["NBW"]
    L_A, L_AS = c["L_A"], c["L_AS"]
    L_DYN, L_DCNT, L_WSUM = c["L_DYN"], c["L_DCNT"], c["L_WSUM"]
    L_DCH, L_EOFF, L_RIX, L_CTL = c["L_DCH"], c["L_EOFF"], c["L_RIX"], c["L_CTL"]
    L_C, L_RCTL = c["L_C"], c["L_RCTL"]
    KS1, KS2, G2 = c["KS1"], c["KS2"], c["G2"]
    CH1, CH2, M_REPEAT = c["CH1"], c["CH2"], c["M_REPEAT"]
    N_TILES, NUM_ACC_N = c["N_TILES"], c["NUM_ACC_N"]
    A_K_STEP, K_ITERS, A_LDS_I32 = c["A_K_STEP"], c["K_ITERS"], c["A_LDS_I32"]
    NAB = c["NAB"]
    # GEMM1 inter-piece choices the planner balances over. BRING-UP: fixed at P=1 (no
    # inter split). unit_tile fuses GEMM1->GEMM2 per unit and GEMM2 contracts the full
    # inter dim, so one unit must own the whole inter dim; P>1 would feed GEMM2 an
    # incomplete intermediate. TODO(tune): separate GEMM1/GEMM2 phases over a shared full
    # intermediate to re-enable P>1 ([p for p in range(1, KS2+1) if KS2 % p == 0]).
    DYN_PS = [1]
    rch = RG  # dynamic row chunk = one row tile (16*mt)
    const_expr = fx.const_expr
    name = (
        f"mega_moe_ep_h{H}_i{I}_e{E}_epr{EPR}_k{TOPK}_mt{MT}_t{TMAX}_np{NPES}_{act}"
        + ("_a8" if a8 else "_a4")
        + ("_ix" if indexed_payload else "")
        + (f"_lim{swiglu_limit:g}" if math.isfinite(swiglu_limit) else "")
    )
    layout_tag = f"{CTRL_INTS}/{FLAG_INTS}/{DEADLINE}/{NCTA_MAX}"
    return KernelCtx(**locals())
