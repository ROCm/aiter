# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Layout constants of the big-M kernel (bigm_kernel.py) shared with its host side
(bigm_layer.py): SDMA queues, phase kinds, the ctrl block, row / chunk granularity."""

# SDMA queues per peer (MORI spreads a peer's queues over its two engines by parity; queues 0
# and 7 carry no puts)
QE, QX, QQ, QXS, QS, QID = 1, 2, 3, 4, 5, 6
NQ = 8
QUEUES = (QE, QX, QQ, QXS, QS, QID)
PUTS_PER_CHUNK = {QE: 1, QX: 1, QQ: 1, QXS: 1, QS: 1, QID: 2}

# phase kinds (per chunk), in item order within a chunk (their done-counter slots; 0 and 2
# unused)
P_RB, P_S1, P_S2, P_S3, P_U, P_RQ, P_RF, P_UB = 1, 3, 4, 5, 6, 7, 8, 9
NKIND = 16  # done-counter slots per chunk
ZQ = 4  # rows per RQ / RF item
SRT = 32  # tokens per sort item
# experts with more units than this (in a chunk) go round-robin over the XCD queues, right after
# the shared expert's units; the others whole to the least-loaded queue. A queue then holds at
# most ceil(units / 8) + UQ (see unit_cap) for any routing (hot experts included)
UQ = 16

# ctrl ints (zeroed by the host before every launch)
C_TK = 0  # per-XCD tickets, 32 apart
C_TURN = 256  # per-queue post turns, 32 apart
C_PH = 512  # [chunk][NKIND] done counters, 32 apart
C_CNT = 4096  # [chunk][NE] routes per expert (256 apart)
C_OFF = 6144  # [chunk][NE] route-list offset of each expert
C_FILL = 8192  # [chunk][NE] routes listed so far
C_UN = 10240  # [chunk][8] units of each XCD (32 apart)
C_SB = 13312  # [chunk][RQ, RF][sub-block] done items, 8 apart
C_SBH = 18432  # [chunk][-, RB][head sub-block] done items, 8 apart
C_XW = 28928  # NE > 256: C_CNT / C_OFF / C_FILL here instead, [chunk][NE] 512 apart
CTRL_INTS = C_XW + 3 * 8 * 512
C_ERR = CTRL_INTS  # sticky watchdog word after the per-launch ctrl block (not zeroed per launch)

CMAX = 7  # chunks per launch (ctrl layout)
PAD = 64  # the kernel's local rows (and chunk rows) are multiples of this (moe() pads)
# rows per unit (AITER_BIGM_RG), one CTA (NW waves) per CU: the validated geometry (64-row
# units: 16% slower at 16384 tokens and failing with two CTAs per CU / three chunks)
UNIT_ROWS = 128


def i8_row_bytes(H):
    return H + H // 8


def unit_cap(tp, s, K1, NE, RG):
    """Upper bound of the units one XCD gets for a chunk of s rows per rank."""
    ubound = -(-tp * s * K1 // RG) + NE
    return -(-ubound // 8) + UQ + 1
