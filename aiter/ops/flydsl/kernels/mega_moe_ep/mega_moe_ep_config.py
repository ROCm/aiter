# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Compile-time layout, shape limits and the per-instance ``KernelCtx`` of the fused
EP MegaMoE kernel.

The epoch-flag arena carries the *dispatch* / *combine* readiness epochs (an
all-to-all scatter completes a receive tile; a combine partial lands back at the
source).  The in-kernel planner (``schedule.build_schedule``) works over this rank's
``EPR`` *local* experts and the ``(token, expert)`` pairs this rank received.
"""

from __future__ import annotations

import functools
import math
import os

import flydsl.expr as fx

from ..kernels_common import ceildiv
from ..mega_moe_tp.common import MAX_TP
from ..mega_moe_tp.mega_moe_tp_config import KernelCtx

__all__ = [
    "CTL_BAR_COUNT",
    "CTL_BAR_GEN",
    "CTL_N_ACTIVE",
    "CTL_N_CHUNKS",
    "CTRL_ERR",
    "EPOCH_SLOT",
    "ERR_COMBINE",
    "ERR_COMPUTE",
    "ERR_DISPATCH",
    "FLAG_COMBINE_COUNT",
    "FLAG_INIT",
    "FLAG_INIT_COUNT",
    "FLAG_INTS",
    "FLAG_ROUTING_COUNT",
    "FLAG_ROUTING_READY",
    "FLAG_SRC_TOKENS",
    "GRID_BAR_COUNT",
    "GRID_BAR_GEN",
    "GRID_BAR_XCD_COUNT",
    "GRID_BAR_XCD_GEN",
    "MAX_CTAS",
    "MAX_RANKS",
    "META_RANK_COUNT",
    "META_XCD_COUNT",
    "NT",
    "NW",
    "N_XCD",
    "ctrl_ints",
    "ep_shape_supported",
    "inter_rows",
    "kernel_config",
    "max_chunks",
    "mega_moe_ep_consts",
]

MAX_RANKS = MAX_TP
MAX_CTAS = 256
N_XCD = 8
# Local experts per rank: the large kernel's HOT_SPLIT segment table gives each lane of
# one wave EPR / 64 of them, and its FLAG_PAYLOAD_COUNT slots (MAX_CTAS per parity) take
# EPR * TOKEN_BUCKETS.
MAX_LOCAL_EXPERTS = 256

# Block geometry: one CTA per CU, eight 64-lane waves, all of them compute MFMA (2 MFMA
# waves/SIMD); the a2a dispatch and the combine scatter run serially on the same waves.
NW = 8
NT = NW * 64

# GEMM geometry (the compute waves reuse the mega_moe stage1/stage2 MFMA atoms; see
# gemm.py). TILE_K is the MFMA K-step byte granularity; TILE_N the GEMM1 output column
# tile over the gate|up halves (2*inter_dim). FP4 activations halve the per-K-step A bytes.
TILE_K = 256
# Each of the 8 compute waves owns TILE_N / NW = 64 output cols, i.e. 32 per gate|up half
# = exactly one 1x32 MX quant group (NUM_ACC_N=4, gui_n=2): the register GEMM1 epilogue +
# the chunked GEMM2 cshuffle need the per-wave group to be in-wave.
TILE_N = 512

# The per-echunk tables (chunk lookup, GEMM1_DONE / GEMM2_DONE counters, inter /
# inter_scale / chunk_payload_row / chunk_recv_slot) are sized by max_chunks() /
# inter_rows().


def max_chunks(tmax: int, topk: int, rg: int, epr: int) -> int:
    """Worst-case echunk count of one launch: every received (token, route) pair --
    at most ``tmax * topk`` (all ranks routing to this one) -- in ``rg``-row chunks,
    plus one partial chunk per local expert."""
    return ceildiv(tmax * topk, rg) + epr


def inter_rows(tmax: int, topk: int, epr: int, rg_max: int = 128) -> int:
    """Worst-case rows of the per-echunk scratch (inter / inter_scale /
    chunk_payload_row / chunk_recv_slot): every received pair plus each local expert's
    partial last chunk at the largest TILE_M."""
    return tmax * topk + epr * rg_max


def ctrl_ints(tmax: int, topk: int, epr: int, rg_min: int = 32) -> int:
    """ctrl ints covering GEMM1_DONE / GEMM2_DONE of any row tile >= ``rg_min``."""
    return GEMM1_DONE + 2 * max_chunks(tmax, topk, rg_min, epr)


# Epoch-flag arena: ints in the symmetric arena ``off_flag`` region. Two banks selected
# by epoch parity so a launch never races the previous one's slots. A readiness slot
# has an arrival *counter* (senders ``g_add_agent`` into it; the last arriver zeroes it
# and stamps the paired *ready* flag with the launch epoch) and a *ready* flag (value =
# epoch, compared wrap-safe with ``common.before``; waiters ``spin_sys_ge``). The
# counter self-resets each launch.
#   FLAG_SRC_TOKENS + parity*MAX_CTAS + src   source rank src's m (written before its
#                                             META arrival; ranks may differ)
#   FLAG_COMBINE_COUNT + parity*MAX_CTAS + 0  combine partial arrival counter: the
#                                             source waits its own m*TOPK*COMBINE_UNITS
#                                             (zeroed by the launch before its next use)
FLAG_SRC_TOKENS = 0
FLAG_COMBINE_COUNT = FLAG_SRC_TOKENS + 2 * MAX_CTAS
# FLAG_INIT + parity*MAX_CTAS + 0        post-init all-to-all barrier ready (epoch)
# FLAG_INIT_COUNT + parity*MAX_CTAS + 0    post-init arrival counter (n_ranks: block 0
# of
#                                        each rank)
FLAG_INIT = FLAG_COMBINE_COUNT + 2 * MAX_CTAS
FLAG_INIT_COUNT = FLAG_INIT + 2 * MAX_CTAS
# FLAG_ROUTING_READY + parity*MAX_CTAS + tile    dispatch META (routing-only) ready
# (epoch) -- FLAG_ROUTING_COUNT + parity*MAX_CTAS + tile    its arrival counter. The
# routing (recv_expert) is scattered + signalled BEFORE the payload so the owner plans
# (assign_positions / plan / gather, all routing-only) while the big payload a2a is
# still in flight.
FLAG_ROUTING_READY = FLAG_INIT_COUNT + 2 * MAX_CTAS
FLAG_ROUTING_COUNT = FLAG_ROUTING_READY + 2 * MAX_CTAS
# FLAG_PAYLOAD_COUNT + parity*MAX_CTAS + le    routes to local expert le whose payload
# row has landed (senders add after draining; the owner compares with its
# LDS_EXPERT_ROWS[le] and its last CTA out zeroes the parity's counters).
FLAG_PAYLOAD_COUNT = FLAG_ROUTING_COUNT + 2 * MAX_CTAS
# Small-batch protocol (the SMALL_BATCH kernel): no init barrier, no host routing fill,
# no META / FLAG_PAYLOAD_COUNT counting.  Each sender writes its header (m) and whole
# routing block, all epoch-tagged, then its payload rows (FLAG_SMALL_PAYLOAD); each
# owner stamps FLAG_SMALL_DONE at every source once its tile queue is done, which the
# source's combine waits.  The next launch then only starts after every owner finished
# this one, so the inputs are safe single-banked; the part region is epoch-parity
# double-banked.  One flag bank, slot = the peer rank; a launch never confuses a past
# one's stamp (epochs only grow / the tag differs):
#   FLAG_SMALL_HEADER + src    src's header: tag | its m (its routing words: the tag)
#   FLAG_SMALL_PAYLOAD + src   src's payload rows to this rank landed (epoch)
#   FLAG_SMALL_DONE + own      owner rank ``own`` finished its whole tile queue (epoch):
#                              every part row it scattered to this rank landed
# The small-batch tag is the epoch's low 32 - TAG_SHIFT bits above TAG_SHIFT value bits
# (m up to 1022, so every small-batch engine's max_tok_per_rank fits).  Routing words
# start (and are re-scrubbed once per tag period, communication.small_scrub) as -1:
# all-ones value bits never pass, so a word unwritten for a whole period can't match.
TAG_SHIFT = 10
# small-batch kernel at inter_dim <= SMALL_SHORT_INTER (a short GEMM2 K loop): the mt2
# kernel keeps the full GEMM2 n-tile, and the host streams the weights non-temporally
# only up to mega_moe_ep.SMALL_STREAM_MAX_SHORT_INTER tokens.
SMALL_SHORT_INTER = 2048
FLAG_SMALL_HEADER = FLAG_PAYLOAD_COUNT + 2 * MAX_CTAS
FLAG_SMALL_PAYLOAD = FLAG_SMALL_HEADER + MAX_RANKS
FLAG_SMALL_DONE = FLAG_SMALL_PAYLOAD + MAX_RANKS
FLAG_INTS = FLAG_SMALL_DONE + MAX_RANKS

# Global control ints (device tensor): a sticky error word plus room.
CTRL_ERR = 0
ERR_DISPATCH, ERR_COMPUTE, ERR_COMBINE = 1, 2, 4
# Device grid-barrier slots (agent-scope counter + generation in the local ctrl tensor;
# the multi-pass GEMM syncs ALL CTAs of this rank between passes). Self-resetting like
# the LDS cta_barrier: the last arriver zeroes the counter and bumps the generation.
# Zeroed once at engine init; the generation grows monotonically (wrap-safe == compare).
GRID_BAR_COUNT, GRID_BAR_GEN = 16, 17
# [GRID_BAR_XCD_COUNT, GRID_BAR_XCD_COUNT + N_XCD): per-XCD arrival counters (2-level
# grid_barrier, XCD barrier); [GRID_BAR_XCD_GEN, GRID_BAR_XCD_GEN + N_XCD): per-XCD
# generations of the XCD-local barrier.
GRID_BAR_XCD_COUNT, GRID_BAR_XCD_GEN = 48, 56
# [META_XCD_COUNT, META_XCD_COUNT + N_XCD) + META_RANK_COUNT: per-XCD / per-rank
# counters that fold this rank's CTAs' dispatch META arrivals into one arrival per owner
# (self-resetting).
META_XCD_COUNT, META_RANK_COUNT = 18, 26
# Launch epoch (device-side, so a captured CUDA graph replays with a fresh one): every
# CTA reads it at entry; CTA 0 bumps it once the plan grid barrier shows all have.
EPOCH_SLOT = 27
# Per-echunk tile counters: each GEMM1 / GEMM2 tile of an echunk adds 1 once its slice
# of inter / part is written, so a GEMM2 tile waits only its own echunk's N_TILES GEMM1
# tiles. The echunk's last GEMM2 tile zeroes both (every waiter has passed by then).
# [GEMM1_DONE, GEMM1_DONE + MAX_CHUNKS): GEMM1 tiles done per echunk; [GEMM2_DONE,
# GEMM2_DONE + MAX_CHUNKS): GEMM2. MAX_CHUNKS = max_chunks() of the instance, so
# GEMM2_DONE = GEMM1_DONE + MAX_CHUNKS is per-instance (kc).
GEMM1_DONE = 64
# Tile queue: XCD x owns a contiguous echunk range and its CTAs pull single n-tiles from
# an XCD-local atomic counter -- all GEMM1 tiles, then all GEMM2 tiles (each waits its
# echunk's N_TILES GEMM1 tiles; the echunk's last GEMM2 tile arrives the combine).  No
# idle CTAs, no partial rounds, no grid barrier between the GEMMs.  An echunk's
# producers and consumers share the XCD's L2, so drained stores + the counter
# suffice.  That needs every bid % N_XCD class on one
# hardware XCC; a HW queue other than the default stream's rotates the bid -> XCC map by
# a constant, which keeps it.  The queue runs expert-major (an expert's n-tiles,
# its echunks minor, so the expert's A stays in L2 across its n-tiles).
# [QUEUE_NEXT, QUEUE_NEXT + N_XCD): per-XCD tile-queue counters (self-resetting)
QUEUE_NEXT = 32
# CTAs past the tile queue (the last one resets FLAG_PAYLOAD_COUNT)
QUEUE_EXITED = QUEUE_NEXT + N_XCD
assert QUEUE_EXITED < GEMM1_DONE
# OR of each CTA's bit (hardware XCC - bid) % N_XCD; the last CTA past the tile queue
# reports ERR_COMPUTE if more than one offset is set, then zeroes it.
XCC_OFFSETS = 28
assert EPOCH_SLOT < XCC_OFFSETS < QUEUE_NEXT
# Dispatch/GEMM overlap: the payload goes in PAYLOAD_ROUNDS = ceil(EPR / N_XCD) rounds
# -- round g: the (token, owner) rows whose smallest le % PAYLOAD_ROUNDS over the
# token's local experts on that owner is g, so expert le is complete after round le %
# PAYLOAD_ROUNDS, about one expert per owner XCD per round -- each counting its landed
# routes into the owner's FLAG_PAYLOAD_COUNT[le]. Every CTA sends the first rounds with
# the routing (before plan: the plan's grid barriers would wait on any CTA still
# sending), then SENDERS_PER_XCD CTAs per XCD send the rest while the others start the
# tile queue.  The queue walks the XCD's experts in send-round order, and a GEMM1 tile
# waits its expert's count. The up-front rounds are picked per launch from m:
# UPFRONT_ROUNDS_LARGE_M at m >= LARGE_M, else UPFRONT_ROUNDS. Owner-split senders:
# sender CTA b sends only owner b % n_ranks's rounds, so a destination whose ingress a
# hot round saturates does not hold back the other owners' later rounds.  Owners taking
# over HOT_OWNER_PCT % of a strided token sample's round-0 sends -- except the sample's
# most-loaded owner -- skip the up-front rounds (they would stall every CTA ahead of the
# plan's grid barrier on that owner's ingress); their senders send those rounds first.
# Deferring the most-loaded owner too would leave its senders alone on the busiest
# ingress.
SENDERS_PER_XCD = 4
HOT_OWNER_PCT = 30
UPFRONT_ROUNDS, UPFRONT_ROUNDS_LARGE_M, LARGE_M = 2, 1, 1024
# Large kernel: a local expert with more than HOT_SPLIT/8 of an even XCD share of
# echunks is hot -- its echunks are split evenly over the XCDs (the others stay whole),
# and each XCD walks its segments in payload round order, so a hot expert sent in a late
# round does not idle the XCDs it alone fills (schedule._hot_segments).  A low threshold
# splits every expert, so each XCD streams all experts' weights.
HOT_SPLIT_LARGE = 4
# [SMALL_SENDERS_DONE, SMALL_SENDERS_DONE + MAX_RANKS): small-batch payload senders to
# owner d done (self-resetting; the META_XCD_COUNT slots, unused by the small-batch
# kernel)
SMALL_SENDERS_DONE = META_XCD_COUNT

# Cross-rank waits spin until their flag arrives: a rank may reach the launch seconds
# after its peers (a JIT compile, a host stall), and a wait that gives up would run on
# with missing rows and stale flags.  AITER_MEGAMOE_EP_WATCHDOG=1 (debug) bounds every
# such wait by DEADLINE instead and traps the kernel on expiry, after setting CTRL_ERR.
WATCHDOG = os.environ.get("AITER_MEGAMOE_EP_WATCHDOG", "0") == "1"


def token_buckets(SMALL_BATCH, EPR):
    """Payload token buckets per source (kernel_config TOKEN_BUCKETS): none in the
    small-batch kernel; in the large kernel 4, halved while EPR * TOKEN_BUCKETS exceeds
    the MAX_CTAS FLAG_PAYLOAD_COUNT slots."""
    if SMALL_BATCH:
        return 0
    buckets = 4
    while buckets > 1 and EPR * buckets > MAX_CTAS:
        buckets //= 2
    return buckets


def a_banks(SMALL_BATCH):
    """GEMM K-loop A-ring banks: 4 (one barrier per two K-steps, A gathered two K-steps
    ahead), except the small-batch kernel's 2 (its few K-loops per CTA do not amortize
    the deeper ring; the large kernel needs it at large m)."""
    return 2 if SMALL_BATCH else 4


# LDS control slots (ints at ``LDS_CTRL``; see schedule.py): the cta_barrier generation
# barrier and the dynamic plan's bookkeeping.
CTL_BAR_COUNT, CTL_BAR_GEN = 0, 1  # compute-wave LDS generation barrier
CTL_N_ACTIVE, CTL_N_CHUNKS = 2, 3  # #active experts, #units
CTL_PREV_COUNT = 4  # old value of this CTA's last echunk-counter add (tid0 -> waves)
CTL_QUEUE_INDEX = 5  # the tile index tid0 pulled from the XCD queue (tid0 -> waves)
CTL_WALK_START = 6  # the expert-major walk's start group (in echunks from chunk_lo)
CTL_INTS = 8

ACT_SET = ("silu", "situv2")


def ep_shape_supported(
    model_dim: int, inter_dim: int, experts: int, world_size: int
) -> bool:
    """Whether the EP kernel can tile this shape (gfx950)."""
    return (
        model_dim % 512 == 0
        and inter_dim % TILE_K == 0
        and inter_dim >= TILE_K
        and experts % world_size == 0
        and 1 <= world_size <= MAX_RANKS
        and experts // world_size <= MAX_LOCAL_EXPERTS
    )


@functools.cache
def mega_moe_ep_consts(
    H: int, I: int, MT: int, EPR: int, a8: bool = False, SMALL_BATCH: bool = False
) -> dict:
    """Per-instance LDS layout (bytes): one row tile's activation + E8M0 scale +
    intermediate staging, plus the dynamic planner's tables (active-expert bitmap /
    dense list, per-local-expert counts, chunk table, wave-prefix scratch, control)."""
    TILE_M = MT * 16  # rows per GEMM tile (sort_block_m)
    BITMAP_WORDS = ceildiv(EPR, 32)  # 32-bit words of the active-local-expert bitmap
    # FP4 halves the per-K-step A bytes.
    a_k_step = TILE_K if a8 else TILE_K // 2
    a_lds_size = TILE_M * a_k_step
    # The A pool is the inline-A GEMMs' bank ring (gemm.py A_BANKS)
    lds_pool = a_banks(SMALL_BATCH) * a_lds_size
    c = {
        "TILE_M": TILE_M,
        "BITMAP_WORDS": BITMAP_WORDS,
        "M_REPEAT": TILE_M // 16,
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

    # compute staging (the layout the gemm_util atoms expect). The GEMM1
    # quantized intermediate goes to a GLOBAL buffer, not LDS: a
    # full TILE_M x inter tile would blow the 160 KB budget for FP8; GEMM2 reads it from
    # global.
    # A bank ring (GEMM1 + GEMM2); cshuffle is in LDS_C_TILE
    c["LDS_A_RING"] = take(lds_pool)
    # A-scale (E8M0) of one row tile (compute-side stage): GEMM1 stages H // 32 bytes
    # per row, GEMM2 I // 32
    c["LDS_A_SCALE"] = take(TILE_M * max(H, I) // 32)
    # GEMM2's cshuffle C-tile gets its OWN region (the A ring is staged ahead during the
    # epilogue, so it can't alias LDS_A_RING).  GEMM1 uses a register epilogue (no
    # C-tile); GEMM2's cshuffle is processed one MFMA m-tile (16 rows) at a time, so
    # LDS_C_TILE only needs 16 x TILE_N f32 (32 KB, not TILE_M x TILE_N) and the
    # shuffled layout (coalesced combine P2P) is preserved.  See gemm.py
    # _epilogue_g2_combine (CROWS=16 per-mi chunking).
    c["LDS_C_TILE"] = take(16 * TILE_N * 4)
    # dynamic planner tables
    # bitmap words | prefix words | dense list
    c["LDS_ACTIVE"] = take((2 * BITMAP_WORDS + EPR) * 4)
    c["LDS_EXPERT_ROWS"] = take(EPR * 4)  # received rows per local expert
    c["LDS_WAVE_SUM"] = take(NW * 2 * 4)  # wave-prefix scratch (count + offset)
    # chunk prefix: first echunk of the j-th active expert (dense order), [nact] = nch
    c["LDS_GROUP_START"] = take((EPR + 1) * 4)
    # echunk -> active index j, one byte per echunk (4 per word) for the first
    # 4*CHUNK_GROUP_WORDS echunks; later echunks binary-search LDS_GROUP_START.  j <=
    # EPR + 3 must fit in a byte.
    c["CHUNK_GROUP_WORDS"] = 2048 if EPR <= 252 else 0
    c["LDS_CHUNK_GROUP"] = take(c["CHUNK_GROUP_WORDS"] * 4)
    # gathered source_key (payload row) list of one unit chunk
    c["LDS_GATHER"] = take(TILE_M * 4)
    c["LDS_CTRL"] = take(CTL_INTS * 4)  # control slots (CTL_* above)
    # large kernel (HOT_SPLIT): this XCD's echunk range + segments
    # (schedule._hot_segments), [X0, X1, nseg, pad] | starts[EPR + 1] | experts[EPR] |
    # chunk offsets[EPR]
    c["LDS_SEGMENTS"] = None if SMALL_BATCH else take((4 + 3 * EPR + 1) * 4)
    # large kernel: per-(local expert, token bucket) inclusive position prefix
    c["LDS_BUCKET_END"] = (
        take(EPR * token_buckets(SMALL_BATCH, EPR) * 4)
        if token_buckets(SMALL_BATCH, EPR)
        else None
    )
    c["LDS_BYTES"] = ceildiv(off, 128) * 128
    return c


def kernel_config(
    *,
    H: int,
    I: int,
    TOPK: int,
    MT: int,
    TOKEN_SLOTS: int,
    E: int,
    EPR: int,
    N_RANKS: int,
    act: str = "silu",
    situ_beta: float = 1.0,
    situ_linear_beta: float = 1.0,
    swiglu_limit: float | None = None,
    a8: bool = False,
    SMALL_BATCH: bool = False,
    STREAM_WEIGHTS: bool = False,
):
    """Derived compile-time values of one ``compile_mega_moe_ep`` instance.

    ``SMALL_BATCH`` selects the small-batch variant (mtpr <= SMALL_MAX_TOKENS, so all
    ranks agree): a separate kernel whose latency-bound prologue / sync path differs
    from the large-batch one.  ``STREAM_WEIGHTS`` (small-batch only) streams the W1 / W2
    loads non-temporally.

    Every rank receives payload keyed by source token (row = rank*mtpr + token,
    bounded by n_ranks*mtpr), overflow-proof under any cross-rank routing skew.
    """
    if act not in ACT_SET:
        raise ValueError(f"mega_moe_ep act must be one of {ACT_SET}, got {act!r}")
    # A limit <= 0 means no SwiGLU clamp, as in the oracle.
    if swiglu_limit is None or swiglu_limit <= 0:
        swiglu_limit = float("inf")
    c = mega_moe_ep_consts(H, I, MT, EPR, a8, SMALL_BATCH)
    TILE_M, LDS_BYTES, BITMAP_WORDS = c["TILE_M"], c["LDS_BYTES"], c["BITMAP_WORDS"]
    LDS_A_RING, LDS_A_SCALE = c["LDS_A_RING"], c["LDS_A_SCALE"]
    LDS_ACTIVE, LDS_EXPERT_ROWS, LDS_WAVE_SUM = (
        c["LDS_ACTIVE"],
        c["LDS_EXPERT_ROWS"],
        c["LDS_WAVE_SUM"],
    )
    LDS_GROUP_START, LDS_GATHER, LDS_CTRL = (
        c["LDS_GROUP_START"],
        c["LDS_GATHER"],
        c["LDS_CTRL"],
    )
    LDS_CHUNK_GROUP, CHUNK_GROUP_WORDS = c["LDS_CHUNK_GROUP"], c["CHUNK_GROUP_WORDS"]
    LDS_C_TILE = c["LDS_C_TILE"]
    M_REPEAT = c["M_REPEAT"]
    N_TILES, NUM_ACC_N = c["N_TILES"], c["NUM_ACC_N"]
    A_K_STEP, K_ITERS, A_LDS_I32 = c["A_K_STEP"], c["K_ITERS"], c["A_LDS_I32"]
    MAX_CHUNKS = max_chunks(TOKEN_SLOTS, TOPK, TILE_M, EPR)
    GEMM2_DONE = GEMM1_DONE + MAX_CHUNKS
    CHUNK_COUNT_BITS = ceildiv(
        TOKEN_SLOTS, TILE_M
    ).bit_length()  # bits of one expert's chunk count
    # inter (/ inter_scale) past 2 GB: rebase its 64-bit base per echunk so the 32-bit
    # buffer offsets stay within one echunk; chunk_payload_row / chunk_recv_slot stay
    # 32-bit
    REBASE_AUTO = inter_rows(TOKEN_SLOTS, TOPK, EPR) * (I if a8 else I // 2) >= 2**31
    # mt=8 (a4w4): the non-rebased per-echunk offset math spills its K-loop, so it
    # always rebases
    INTER_REBASE = REBASE_AUTO or MT == 8
    # owner-split senders + hot-owner up-front deferral (SENDERS_PER_XCD)
    OWNER_SENDERS = not SMALL_BATCH
    assert not OWNER_SENDERS or SENDERS_PER_XCD * N_XCD >= MAX_RANKS
    WATCHDOG_ON = WATCHDOG
    HOT_SPLIT = 0 if SMALL_BATCH else HOT_SPLIT_LARGE
    # Large kernel: per-(local expert, token bucket) payload readiness.  Every payload
    # round goes in TOKEN_BUCKETS token buckets (bucket j: the source's tokens [j m /
    # TOKEN_BUCKETS, (j + 1) m / TOKEN_BUCKETS)), each counted into
    # FLAG_PAYLOAD_COUNT[le * TOKEN_BUCKETS + j] after its drain; assign_positions
    # orders an expert's positions bucket-major and a GEMM1 tile waits only the buckets
    # holding its rows.  Hot (HOT_SPLIT) experts are split over the XCDs chunk-strided
    # (XCD x: chunks x, x + 8, ...) and walked in sub-groups, so an expert that holds
    # most of a rank's rows starts its GEMM1 before all of its payload has landed.
    TOKEN_BUCKETS = token_buckets(SMALL_BATCH, EPR)
    BUCKET_COUNTERS = max(TOKEN_BUCKETS, 1)
    # a sender of m tokens drains every TOKEN_BUCKETS / nb buckets (nb =
    # min(TOKEN_BUCKETS, pow2 <= m // BUCKET_DRAIN_TOKENS): a drain per bucket costs
    # more than it overlaps at small m); a hot segment of ng echunks runs in
    # min(TOKEN_BUCKETS, max(1, ng // HOT_SUBGROUP_CHUNKS)) sub-groups
    BUCKET_DRAIN_TOKENS = 2048
    HOT_SUBGROUP_CHUNKS = 4
    assert not TOKEN_BUCKETS or (
        HOT_SPLIT
        and EPR * TOKEN_BUCKETS <= MAX_CTAS
        and TOKEN_BUCKETS <= N_XCD
        and TOKEN_BUCKETS & (TOKEN_BUCKETS - 1) == 0
        and (TOKEN_SLOTS // N_RANKS) % TOKEN_BUCKETS == 0
    )
    assert MAX_CHUNKS * TILE_M * 4 < 2**31
    # 32-bit buffer offsets / sizes: payload rows, the combine part slots (mtpr * TOPK
    # rows of H + H/8 bytes) and the expert_recv_slots; gemm_util's row map keeps 24
    # row bits
    assert TOKEN_SLOTS * (H if a8 else H // 2) < 2**31 and TOKEN_SLOTS < 2**24
    assert TOKEN_SLOTS // N_RANKS * TOPK * (H + H // 8) < 2**31
    assert EPR * TOKEN_SLOTS * 4 < 2**31
    const_expr = fx.const_expr
    A_BANKS = a_banks(SMALL_BATCH)
    # small-batch kernel: cta_barrier = wait_lgkm0 + s_barrier instead of the LDS
    # generation barrier (every wave of the CTA computes, so the hardware barrier covers
    # them all).  The large kernel keeps the LDS barrier: s_barrier there spills its mt8
    # K-loop.
    HW_BARRIER = bool(SMALL_BATCH)
    # GEMM2 n-tile: no quant group ties it to the wave, and the tile queue ends on GEMM2
    # tiles, so the small-batch mt2 kernel halves it to shorten the queue's tail (the
    # mt4 small-batch kernel and inter_dim <= SMALL_SHORT_INTER keep TILE_N).
    TILE_N2 = 256 if SMALL_BATCH and MT == 2 and I > SMALL_SHORT_INTER else TILE_N
    # small-batch tile queue: pull the next tile after the K-loop, before this tile's
    # epilogue, instead of after its drain.  The large kernel keeps the pull after the
    # drain.
    PULL_AHEAD = bool(SMALL_BATCH)
    # STREAM_WEIGHTS: W1 / W2 buffer loads sc0 nt (each small-batch weight tile is read
    # about once).  The host picks it from m >= SMALL_STREAM_MIN.
    assert SMALL_BATCH or not STREAM_WEIGHTS
    WEIGHT_AUX = 3 if STREAM_WEIGHTS else 0
    # m / local expert in TAG_SHIFT bits; all-ones value bits = a scrubbed routing word
    assert not SMALL_BATCH or max(TOKEN_SLOTS // N_RANKS, EPR) < (1 << TAG_SHIFT) - 1
    LDS_SEGMENTS, LDS_BUCKET_END = c["LDS_SEGMENTS"], c["LDS_BUCKET_END"]
    assert EPR <= MAX_LOCAL_EXPERTS
    name = (
        f"mega_moe_ep_h{H}_i{I}_e{E}_epr{EPR}_k{TOPK}_mt{MT}"
        + f"_t{TOKEN_SLOTS}_np{N_RANKS}_{act}"
        + (f"_b{situ_beta:g}_{situ_linear_beta:g}" if act == "situv2" else "")
        + ("_a8" if a8 else "_a4")
        + ("_sb" if SMALL_BATCH else "")
        + ("_wnt" if STREAM_WEIGHTS else "")
        + ("_rb" if INTER_REBASE and not REBASE_AUTO else "")
        + (f"_lim{swiglu_limit:g}" if math.isfinite(swiglu_limit) else "")
        + ("_wd" if WATCHDOG_ON else "")
    )
    return KernelCtx(**locals(), NW=NW, NT=NT, TILE_N=TILE_N)
