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
import os

import flydsl.expr as fx

from ..kernels_common import ceildiv
from ..mega_moe_tp.common import DEADLINE, MAX_TP
from ..mega_moe_tp.mega_moe_tp_config import KernelCtx

__all__ = [
    "CTRL_ERR",
    "C_BARCNT",
    "C_BARGEN",
    "C_CNT",
    "C_DYNP",
    "C_GEXP",
    "C_NACT",
    "C_NCH",
    "C_UL",
    "C_UROWS",
    "C_USEQ",
    "DYN_MAX",
    "EPOCH_SLOT",
    "ERR_COMBINE",
    "ERR_COMPUTE",
    "ERR_DISPATCH",
    "FLAG_COMB_CNT",
    "FLAG_DISPATCH",
    "FLAG_DISPM",
    "FLAG_DISPM_CNT",
    "FLAG_DISP_CNT",
    "FLAG_INIT",
    "FLAG_INIT_CNT",
    "FLAG_INTS",
    "FLAG_SRCM",
    "GBAR_CNT",
    "GBAR_GEN",
    "GBAR_X",
    "GBAR_XG",
    "MAGG_G",
    "MAGG_X",
    "MAX_NPES",
    "NCTA_MAX",
    "NT",
    "NTT",
    "NW",
    "N_XCD",
    "UL_MAX",
    "ctrl_ints",
    "ep_shape_supported",
    "inter_rows",
    "kernel_config",
    "mega_moe_ep_consts",
    "nch_max",
]

MAX_NPES = MAX_TP
NCTA_MAX = 256
N_XCD = 8

# Block geometry: one CTA per CU, eight 64-lane waves, all of them compute MFMA (2 MFMA
# waves/SIMD); the a2a dispatch and the combine scatter run serially on the same waves.
NW = 8
NT = NW * 64
NTT = NT

# GEMM geometry (the compute waves reuse the mega_moe stage1/stage2 MFMA atoms; see
# gemm.py). TILE_K is the MFMA K-step byte granularity; TILE_N the GEMM1 output column
# tile over the gate|up halves (2*inter_dim). FP4 activations halve the per-K-step A bytes.
KCS = 2  # A K-chunks staged together
TILE_K = 256
# Each of the 8 compute waves owns TILE_N / NW = 64 output cols, i.e. 32 per gate|up half
# = exactly one 1x32 MX quant group (NUM_ACC_N=4, gui_n=2): the register GEMM1 epilogue +
# the chunked GEMM2 cshuffle need the per-wave group to be in-wave.
TILE_N = 512

# Up to this many received (token, expert) pairs per rank use the dynamic planner
# (the one ported here); larger batches want an LB-style two-phase planner (future).
DYN_MAX = 256
# The per-echunk tables (chunk lookup, ECNT counters, inter / inters / rowmap / rowidx)
# are sized by nch_max() / inter_rows().


def nch_max(tmax: int, topk: int, rg: int, epr: int) -> int:
    """Worst-case echunk count of one launch: every received (token, route) pair --
    at most ``tmax * topk`` (all ranks routing to this one) -- in ``rg``-row chunks,
    plus one partial chunk per local expert."""
    return ceildiv(tmax * topk, rg) + epr


def inter_rows(tmax: int, topk: int, epr: int, rg_max: int = 128) -> int:
    """Worst-case rows of the per-echunk scratch (inter / inters / rowmap / rowidx):
    every received pair plus each local expert's partial last chunk at the largest RG.
    """
    return tmax * topk + epr * rg_max


def ctrl_ints(tmax: int, topk: int, epr: int, rg_min: int = 32) -> int:
    """ctrl ints covering ECNT1 / ECNT2 of any row tile >= ``rg_min``."""
    return ECNT1 + 2 * nch_max(tmax, topk, rg_min, epr)


# Epoch-flag arena: ints in the symmetric arena ``off_flag`` region. Two banks selected
# by epoch parity so a launch never races the previous one's slots. Each readiness tile
# has an arrival *counter* (senders ``g_add_agent`` into it; the last arriver zeroes it
# and stamps the paired *ready* flag with the launch epoch) and a *ready* flag (value =
# epoch, compared wrap-safe with ``common.before``; waiters ``spin_sys_ge``). The
# counter self-resets each launch; ``reset()`` clears residue after a timeout.
#   FLAG_DISPATCH + parity*NCTA_MAX + tile   dispatch receive tile ready (epoch)
#   FLAG_DISP_CNT + parity*NCTA_MAX + tile   dispatch receive tile arrival counter
#   FLAG_SRCM     + parity*NCTA_MAX + src    source rank src's m (written before its META
#                                            arrival; ranks may differ)
#   FLAG_COMB_CNT + parity*NCTA_MAX + 0      combine partial arrival counter: the source
#                                            waits its own m*TOPK*COMB_W (zeroed by the
#                                            launch before its next use)
# NOTE: per-tile granularity is the scaffold's bound; the final tile->slot mapping may
# be refined once dispatch_send fixes the receive-tile count.
FLAG_DISPATCH = 0
FLAG_DISP_CNT = FLAG_DISPATCH + 2 * NCTA_MAX
FLAG_SRCM = FLAG_DISP_CNT + 2 * NCTA_MAX
FLAG_COMB_CNT = FLAG_SRCM + 2 * NCTA_MAX
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
# FLAG_PAYC + parity*NCTA_MAX + le    routes to local expert le whose payload row
# has landed (senders add after draining; the owner compares with its L_DCNT[le] and its
# last CTA out zeroes the parity's counters).
FLAG_PAYC = FLAG_DISPM_CNT + 2 * NCTA_MAX
# SBX (small-batch protocol), one bank, slot = the peer rank; a launch never
# confuses a past one's stamp (epochs only grow / the tag differs):
#   FLAG_SBM + src   src's header: tag | its m (the routing words carry the same tag)
#   FLAG_SBP + src   src's payload rows to this rank landed (epoch)
#   FLAG_SBD + own   owner rank ``own`` finished its whole tile queue (epoch): every
#                    part row it scattered to this rank landed
# The SBX tag is the epoch's low 32 - SBX_TB bits above SBX_TB value bits.  Routing
# words start (and are re-scrubbed once per tag period, communication.sbx_scrub) as -1:
# all-ones value bits never pass, so a word unwritten for a whole period can't match.
SBX_TB = 9
FLAG_SBM = FLAG_PAYC + 2 * NCTA_MAX
FLAG_SBP = FLAG_SBM + MAX_NPES
FLAG_SBD = FLAG_SBP + MAX_NPES
FLAG_INTS = FLAG_SBD + MAX_NPES

# Global control ints (device tensor): a sticky watchdog error word plus room.
CTRL_ERR = 0
ERR_DISPATCH, ERR_COMPUTE, ERR_COMBINE = 1, 2, 4
# Device grid-barrier slots (agent-scope counter + generation in the local ctrl tensor;
# the multi-pass GEMM syncs ALL CTAs of this rank between passes). Self-resetting like
# the LDS cbar: the last arriver zeroes the counter and bumps the generation. Zeroed
# once at engine init; the generation grows monotonically (wrap-safe == compare).
GBAR_CNT, GBAR_GEN = 16, 17
# [GBAR_X, GBAR_X + N_XCD): per-XCD arrival counters (2-level grid_bar, XCD barrier);
# [GBAR_XG, GBAR_XG + N_XCD): per-XCD generations of the XCD-local barrier.
GBAR_X, GBAR_XG = 48, 56
# [MAGG_X, MAGG_X + N_XCD) + MAGG_G: per-XCD / per-rank counters that fold this rank's
# CTAs' dispatch META arrivals into one arrival per owner (self-resetting).
MAGG_X, MAGG_G = 18, 26
# Launch epoch (device-side, so a captured CUDA graph replays with a fresh one): every
# CTA reads it at entry; CTA 0 bumps it once the plan grid barrier shows all have.
EPOCH_SLOT = 27
# Per-echunk tile counters: each GEMM1 / GEMM2 tile of an echunk adds 1 once its slice of
# inter / part is written, so a GEMM2 tile waits only its own echunk's N_TILES GEMM1 tiles.
# The echunk's last GEMM2 tile zeroes both (every waiter has passed by then).
# [ECNT1, ECNT1 + NCHM): GEMM1 tiles done per echunk; [ECNT2, ECNT2 + NCHM): GEMM2.
# NCHM = nch_max() of the instance, so ECNT2 = ECNT1 + NCHM is per-instance (kc).
ECNT1 = 64
# Tile queue: XCD x owns a contiguous echunk range and its CTAs pull single n-tiles from
# an XCD-local atomic counter -- all GEMM1 tiles, then all GEMM2 tiles (each waits its
# echunk's N_TILES GEMM1 tiles; the echunk's last GEMM2 tile arrives the combine).  No
# idle CTAs, no partial rounds, no grid barrier between the GEMMs.  An echunk's producers and consumers share the XCD's
# L2, so drained stores + the counter suffice; a CTA whose hardware XCC is not
# bid % N_XCD reports ERR_COMPUTE.  The queue runs expert-major (an expert's n-tiles,
# its echunks minor, so the expert's A stays in L2 across its n-tiles).
QCNT = 32  # [QCNT, QCNT + N_XCD): per-XCD tile-queue counters (self-resetting)
QDONE = QCNT + N_XCD  # CTAs past the tile queue (the last one resets FLAG_PAYC)
assert QDONE < ECNT1
# Dispatch/GEMM overlap: the payload goes in LB = ceil(EPR / N_XCD) rounds -- round g:
# the (token, owner) rows whose smallest le % LB over the token's local experts on that
# owner is g, so expert le is complete after round le % LB, about one expert per owner
# XCD per round -- each counting its landed routes into the owner's FLAG_PAYC[le].
# Every CTA sends the first rounds with the routing (before plan: the plan's grid
# barriers would wait on any CTA still sending), then NSX CTAs per XCD send the rest
# while the others start the tile queue.  The queue walks the XCD's experts in send-round
# order, and a GEMM1 tile waits its expert's count.
# The up-front rounds are picked per launch from m: OVG0_HI at m >= OVG0_MIN, else
# OVG0_LO.
# Owner-split senders (non-SDMA): sender CTA b sends only owner b % npes's rounds, so a
# destination whose ingress a hot round saturates no longer holds back the other owners'
# later rounds.  Owners taking over HVT % of a strided token sample's round-0 sends --
# except the sample's most-loaded owner -- skip the up-front rounds (they would stall
# every CTA ahead of the plan's grid barrier on that owner's ingress); their senders
# send those rounds first.  Deferring the most-loaded owner too would leave its senders
# alone on the busiest ingress.
NSX = 4
HVT = 30
OVG0_LO, OVG0_HI, OVG0_MIN = 2, 1, 1024
# Large kernel: a local expert with more than XHOT/8 of an even XCD share of echunks is
# hot -- its echunks are split evenly over the XCDs (the rest as before), and each XCD
# walks its segments in payload round order, so a hot expert sent in a late round no
# longer idles the XCDs it alone fills (schedule._seg_table).  A low threshold splits
# every expert, so each XCD streams all experts' weights.
XHOT_LARGE = 4
# [SBX_SCNT, SBX_SCNT + MAX_NPES): SBX payload senders to owner d done (self-resetting;
# the MAGG slots, unused by SBX)
SBX_SCNT = MAGG_X

# AITER_MEGAMOE_EP_SDMA=1 (large kernel, experimental, default off): the payload rounds go
# by MORI device SDMA.
# Every sender packs round g's rows per owner p contiguously at the compacted row
# off(p, g) + pos (pos from per-(p, g) counters, then a grid barrier), into its own
# arena's stage region (p == rank: straight into its payload); the round's last CTA posts
# one put per (owner, payload / scale) and the owner waits the per-(source, queue)
# signal counts instead of FLAG_PAYC.  The owner finds a row through recv_pos.  The
# extra counting pass and grid barrier cost small batches.
SDMA_ENV = int(os.environ.get("AITER_MEGAMOE_EP_SDMA", "0"))
SD_QP, SD_QS = 1, 2  # SDMA queues: payload rows, their scales
SD_GMAX = 8  # >= payload rounds (ceil(EPR / N_XCD))
# sdst (local int32 tensor) slots
SD_CNT = 0  # [MAX_NPES * SD_GMAX]: rows per (owner, round) (self-resetting)
SD_RD = (
    SD_CNT + MAX_NPES * SD_GMAX
)  # [SD_GMAX]: CTAs done packing round g (self-resetting)
SD_TURN = (
    SD_RD + SD_GMAX
)  # the next round to post (puts in round order; self-resetting)
SD_SELF = SD_TURN + 1  # epoch * SD_GMAX + rounds of this rank's own rows written
SD_LAND = SD_SELF + 1  # epoch * SD_GMAX + rounds some CTA saw landed from every source
SD_BASE = (
    SD_SELF + 7
)  # [MAX_NPES * 2]: this launch's signal base per (source, SD_QP / SD_QS)
SD_INTS = SD_BASE + 2 * MAX_NPES
SD_DUM = 64  # bytes per source of the dummy-put target (a round without rows)
# SDMA: the packing is local, so it takes more up-front rounds and sender CTAs
SD_NSX, SD_OVG0 = 8, 2


def sbx_on(SB):
    """SB kernel small-batch protocol (SBX) -- no init barrier, no host recv fill, no META /
    PAYC counting: each sender writes its header (m) and whole routing block, all
    epoch-tagged, then its payload rows (FLAG_SBP); each owner stamps FLAG_SBD at every
    source once its tile queue is done, which the source's combine waits.  The next launch
    then only starts after every owner finished this one, so the inputs are safe
    single-banked; the part region is epoch-parity double-banked."""
    return bool(SB)


def pb_on(SB):
    """Payload token buckets per source (kernel_config PB): 4 in the large kernel, none
    in the SB kernel."""
    return 0 if SB else 4


def a_banks(SB):
    """GEMM K-loop A-ring banks: 4 (one barrier per two K-steps, A gathered two K-steps
    ahead), except the SB kernel's 2 (its few K-loops per CTA do not amortize the deeper
    ring; the large kernel needs it at large m)."""
    return 2 if SB else 4


# LDS control slots (ints at ``L_CTL``; see schedule.py). The cbar generation barrier,
# the dynamic plan's bookkeeping and the per-CTA unit list live here.
C_BARCNT, C_BARGEN = 0, 1  # compute-wave LDS generation barrier
C_NACT, C_NCH, C_DYNP = 2, 3, 5  # #active experts, #units, P
C_UROWS, C_USEQ, C_GEXP, C_CNT = 6, 7, 8, 9  # current unit rows / seq / gather cache
C_ELAST = 10  # old value of this CTA's last echunk-counter add (tid0 -> waves)
C_QIDX = 11  # the tile index tid0 pulled from the XCD queue (tid0 -> waves)
C_ROT = 14  # the expert-major walk's start group (in echunks from e_lo)
C_SDL = 12  # SDMA: epoch * SD_GMAX + payload rounds this CTA saw landed (tid0)
C_UL = 16  # per-CTA unit list base (records of 4 ints: expert_chunk, i0, icnt, kind)
UL_MAX = (128 - C_UL) // 4

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
    H: int, I: int, MT: int, EPR: int, TMAX: int, a8: bool = False, SB: bool = False
) -> dict:
    """Per-instance LDS layout (bytes): one row tile's activation + E8M0 scale +
    intermediate staging, plus the dynamic planner's tables (active-expert bitmap /
    dense list, per-local-expert counts, chunk table, wave-prefix scratch, control)."""
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
    # The A pool is the inline-A GEMMs' bank ring (gemm.py A_NBK)
    lds_pool = a_banks(SB) * a_lds_size
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
    # global.
    c["L_A"] = take(lds_pool)  # A bank ring (GEMM1 + GEMM2); cshuffle is in L_C
    c["L_AS"] = take(
        RG * a_scale
    )  # A-scale (E8M0) of one row tile (compute-side stage)
    # GEMM2's cshuffle C-tile gets its OWN region (the A ring is staged ahead during the
    # epilogue, so it can't alias L_A).  GEMM1 uses a register epilogue (no C-tile);
    # GEMM2's cshuffle is processed one MFMA m-tile (16 rows) at a time, so L_C only
    # needs 16 x TILE_N f32 (32 KB, not RG x TILE_N).
    # The saving lets the whole block fit twice per CU (2 MFMA waves/SIMD) while the
    # shuffled layout (coalesced combine P2P) is preserved.  See gemm.py
    # _epilogue_g2_combine (CROWS=16 per-mi chunking).
    c["L_C"] = take(16 * TILE_N * 4)
    # dynamic planner tables
    c["L_DYN"] = take((2 * NBW + EPR) * 4)  # bitmap words | prefix words | dense list
    c["L_DCNT"] = take(EPR * 4)  # received rows per local expert
    c["L_WSUM"] = take(NW * 2 * 4)  # wave-prefix scratch (count + offset)
    # chunk prefix: first echunk of the j-th active expert (dense order), [nact] = nch
    c["L_DCH"] = take((EPR + 1) * 4)
    # echunk -> active index j, one byte per echunk (4 per word) for the first 4*NDIXW
    # echunks; later echunks binary-search L_DCH.  j <= EPR + 3 must fit in a byte.
    c["NDIXW"] = 2048 if EPR <= 252 else 0
    c["L_DIX"] = take(c["NDIXW"] * 4)
    c["L_EOFF"] = take(EPR * 4)  # per-expert row offset (gather)
    c["L_RIX"] = take(
        RG * 4
    )  # gathered source_key (payload row) list of one unit chunk
    c["L_CTL"] = take(128 * 4)  # control slots (C_* above)
    # large kernel (XHOT): this XCD's echunk range + segments (schedule._seg_table),
    # [X0, X1, nseg, pad] | starts[EPR + 1] | experts[EPR] | chunk offsets[EPR]
    c["L_SEG"] = None if SB else take((4 + 3 * EPR + 1) * 4)
    # large kernel: per-(local expert, token bucket) inclusive position prefix
    c["L_PBC"] = take(EPR * pb_on(SB) * 4) if pb_on(SB) else None
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
    SB: bool = False,
    WNT: bool = False,
):
    """Derived compile-time values of one ``compile_mega_moe_ep`` instance.

    ``SB`` selects the small-batch variant (mtpr <= SB_MAX, so all ranks agree): a
    separate kernel whose latency-bound prologue / sync path differs from the
    large-batch one.  ``WNT`` (SB only) streams the W1 / W2 loads non-temporally.

    Every rank receives payload keyed by source token (row = rank*mtpr + token,
    bounded by npes*mtpr), overflow-proof under any cross-rank routing skew.
    """
    if act not in ACT_SET:
        raise ValueError(f"mega_moe_ep act must be one of {ACT_SET}, got {act!r}")
    if swiglu_limit is None:
        swiglu_limit = float("inf")
    c = mega_moe_ep_consts(H, I, MT, EPR, TMAX, a8, SB)
    RG, LDS_BYTES, NBW = c["RG"], c["LDS_BYTES"], c["NBW"]
    L_A, L_AS = c["L_A"], c["L_AS"]
    L_DYN, L_DCNT, L_WSUM = c["L_DYN"], c["L_DCNT"], c["L_WSUM"]
    L_DCH, L_EOFF, L_RIX, L_CTL = c["L_DCH"], c["L_EOFF"], c["L_RIX"], c["L_CTL"]
    L_DIX, NDIXW = c["L_DIX"], c["NDIXW"]
    L_C = c["L_C"]
    KS1, KS2, G2 = c["KS1"], c["KS2"], c["G2"]
    CH1, CH2, M_REPEAT = c["CH1"], c["CH2"], c["M_REPEAT"]
    N_TILES, NUM_ACC_N = c["N_TILES"], c["NUM_ACC_N"]
    A_K_STEP, K_ITERS, A_LDS_I32 = c["A_K_STEP"], c["K_ITERS"], c["A_LDS_I32"]
    # GEMM1 inter-piece choices the planner balances over: fixed at P=1 (no inter
    # split).  GEMM2 contracts the full inter dim, so one unit must own the whole inter
    # dim; P>1 would feed GEMM2 an incomplete intermediate.
    DYN_PS = [1]
    rch = RG  # dynamic row chunk = one row tile (16*mt)
    NCHM = nch_max(TMAX, TOPK, RG, EPR)
    ECNT2 = ECNT1 + NCHM
    CE_BITS = ceildiv(TMAX, RG).bit_length()  # bits of one expert's chunk count
    # inter (/ inters) past 2 GB: rebase its 64-bit base per echunk so the 32-bit buffer
    # offsets stay within one echunk; rowmap / rowidx stay 32-bit
    REBASE_AUTO = inter_rows(TMAX, TOPK, EPR) * (I if a8 else I // 2) >= 2**31
    # mt=8 (a4w4): the non-rebased per-echunk offset math spills its K-loop, so it
    # always rebases
    INTER_REBASE = REBASE_AUTO or MT == 8
    SDMAD = not SB and bool(SDMA_ENV)
    NSXK = SD_NSX if SDMAD else NSX  # payload sender CTAs per XCD
    OVG0K = SD_OVG0 if SDMAD else None  # up-front payload rounds (None: by m, OVG0_*)
    OWS = (
        not SB and not SDMAD
    )  # owner-split senders + hot-owner up-front deferral (NSX)
    assert not OWS or NSXK * N_XCD >= MAX_NPES
    assert not SDMAD or ceildiv(EPR, N_XCD) <= SD_GMAX
    XHOT = 0 if SB else XHOT_LARGE
    # Large kernel: per-(local expert, token bucket) payload readiness.  Every payload
    # round goes in PB token buckets (bucket j: the source's tokens [j m / PB,
    # (j + 1) m / PB)), each counted into FLAG_PAYC[le * PB + j] after its drain; pre_pos
    # orders an expert's positions bucket-major and a GEMM1 tile waits only the buckets
    # holding its rows.  Hot (XHOT) experts are split over the XCDs chunk-strided (XCD x:
    # chunks x, x + 8, ...) and walked in sub-groups, so an expert that holds most of a
    # rank's rows starts its GEMM1 before all of its payload has landed.
    PB = pb_on(SB)
    PBK = max(PB, 1)
    # a sender of m tokens drains every PB / nb buckets (nb = min(PB, pow2 <= m // PBT):
    # a drain per bucket costs more than it overlaps at small m); a hot segment of ng
    # echunks runs in min(PB, max(1, ng // PBSG)) sub-groups
    PBT = 2048
    PBSG = 4
    assert not PB or (
        XHOT
        and not SDMAD
        and EPR * PB <= NCTA_MAX
        and PB <= N_XCD
        and PB & (PB - 1) == 0
        and (TMAX // NPES) % PB == 0
    )
    assert NCHM * RG * 4 < 2**31
    const_expr = fx.const_expr
    SBXP = sbx_on(SB)
    A_NBK = a_banks(SB)
    # SB kernel: cbar = wait_lgkm0 + s_barrier instead of the LDS generation barrier
    # (every wave of the CTA computes, so the hardware barrier covers them all).  The
    # large kernel keeps the LDS barrier: s_barrier there spills its mt8 K-loop.
    HWBAR = bool(SB)
    # GEMM2 n-tile: no quant group ties it to the wave, and the tile queue ends on GEMM2
    # tiles, so the SB mt2 kernel halves it to shorten the queue's tail (the mt4 SB
    # kernel keeps TILE_N).
    TILE_N2 = 256 if SB and MT == 2 else TILE_N
    # SB tile queue: pull the next tile after the K-loop, before this tile's epilogue,
    # instead of after its drain.  The large kernel keeps the pull after the drain.
    QLA = bool(SB)
    # WNT: W1 / W2 buffer loads sc0 nt (each SB weight tile is read about once).  The host
    # picks it from m >= SB_WNT_MIN.
    assert SB or not WNT
    BNT = 3 if WNT else 0
    # m / local expert in SBX_TB bits; all-ones value bits = a scrubbed routing word
    assert not SBXP or max(TMAX // NPES, EPR) < (1 << SBX_TB) - 1
    L_SEG, L_PBC = c["L_SEG"], c["L_PBC"]
    assert not XHOT or EPR <= 64  # _seg_table: one lane per active expert
    name = (
        f"mega_moe_ep_h{H}_i{I}_e{E}_epr{EPR}_k{TOPK}_mt{MT}_t{TMAX}_np{NPES}_{act}"
        + ("_a8" if a8 else "_a4")
        + ("_sb" if SB else "")
        + ("_wnt" if WNT else "")
        + ("_rb" if INTER_REBASE and not REBASE_AUTO else "")
        + (f"_lim{swiglu_limit:g}" if math.isfinite(swiglu_limit) else "")
        + ("_sdma" if SDMAD else "")
    )
    layout_tag = f"{ECNT2 + NCHM}/{FLAG_INTS}/{DEADLINE}/{NCTA_MAX}"
    return KernelCtx(**locals(), NW=NW, NT=NT, NTT=NTT, TILE_N=TILE_N)
