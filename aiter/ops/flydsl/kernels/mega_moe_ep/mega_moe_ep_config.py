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
    "FLAG_DISP_CNT",
    "FLAG_INTS",
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
#   [0, NT)              compute_units (local-expert GEMM1 + act + GEMM2) + comm_help
#   [NT, NT+64)          dispatch_send (a2a scatter tokens to expert-owner ranks)
#   [NT+64, NT+128)      a_loader (stage dispatched tokens to LDS)
#   [NT+128, NTT)        combine_send (scatter partials back to source) + comm_help
B_DISPATCH = NT
B_LOADER = NT + 64
B_COMBINE = NT + 128

# Up to this many received (token, expert) pairs per rank use the dynamic planner
# (the one ported here); larger batches want an LB-style two-phase planner (future).
DYN_MAX = 256
# Chunk-table slots (active-expert row chunks of 16*mt rows) the dynamic planner emits.
NCHA = 256

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
FLAG_INTS = FLAG_COMB_CNT + 2 * NCTA_MAX

# Global control ints (device tensor): a sticky watchdog error word plus room.
CTRL_ERR = 0
ERR_DISPATCH, ERR_COMPUTE, ERR_COMBINE = 1, 2, 4
CTRL_INTS = 64

# LDS control slots (ints at ``L_CTL``; see schedule.py). The cbar generation barrier,
# the dynamic plan's bookkeeping and the per-CTA unit list live here.
C_BARCNT, C_BARGEN = 0, 1  # 4-compute-wave LDS generation barrier
C_NACT, C_NCH, C_PLAN, C_DYNP = 2, 3, 4, 5  # #active experts, #units, plan-ready, P
C_UROWS, C_USEQ, C_GEXP, C_CNT = 6, 7, 8, 9  # current unit rows / seq / gather cache
C_UNIT = 12  # [C_UNIT, C_UNIT+1): this CTA's unit range [lo, hi)
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
    H: int, I: int, MT: int, EPR: int, TMAX: int, a8: bool = False
) -> dict:
    """Per-instance LDS layout (bytes): one row tile's activation + E8M0 scale +
    intermediate staging, plus the dynamic planner's tables (active-expert bitmap /
    dense list, per-local-expert counts, chunk table, wave-prefix scratch, control).

    TODO(compute): grow the A / B / scale / intermediate region into the full ring that
    ``mega_moe_tp_consts`` carries once the GEMM device code lands.
    """
    RG = MT * 16  # rows per tile
    a_row = H if a8 else H // 2  # activation bytes per row (FP8 vs packed FP4)
    a_scale = H // 32  # E8M0 scale bytes per row
    inter_row = I if a8 else I // 2
    NBW = ceildiv(EPR, 32)  # 32-bit words of the active-local-expert bitmap
    c = {
        "RG": RG,
        "A_ROW": a_row,
        "A_SCALE": a_scale,
        "INTER_ROW": inter_row,
        "NBW": NBW,
    }
    off = 0

    def take(n: int) -> int:
        nonlocal off
        off = ceildiv(off, 128) * 128
        start, off = off, off + n
        return start

    # compute staging (one row tile)
    c["L_A"] = take(RG * a_row)
    c["L_AS"] = take(RG * a_scale)
    c["L_INTER"] = take(RG * inter_row)
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
    L_A, L_AS, L_INTER = c["L_A"], c["L_AS"], c["L_INTER"]
    L_DYN, L_DCNT, L_WSUM = c["L_DYN"], c["L_DCNT"], c["L_WSUM"]
    L_DCH, L_EOFF, L_RIX, L_CTL = c["L_DCH"], c["L_EOFF"], c["L_RIX"], c["L_CTL"]
    # GEMM1 inter-piece choices the planner balances over (full inter dim = KS2 blocks).
    KS2 = I // 128
    DYN_PS = [p for p in range(1, KS2 + 1) if KS2 % p == 0]
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
