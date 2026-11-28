# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Layouts of the buffers one kernel writes and another kernel reads.

Writer and reader import the same names from here.
"""


# The kernel key QX of the x quantization the kernels exchange: one f32 scale per
# (row, 128 columns) (QX 1) or per row (QX 0).
def qx_of(x_quant):
    """The kernel key QX of ``x_quant`` ("per_1x128" or "per_token")."""
    assert x_quant in ("per_1x128", "per_token"), x_quant
    return int(x_quant == "per_1x128")


# ---- main path: workspace / sched of comm.py and main.py (layout: comm.py) ----------
# The EP sizes (ranks, NPES) the kernels take: powers of two of at most 8 (a rank is a
# lane mask into a peer table, and a route's source id keeps 3 bits for its rank).
EP_SIZES = (4, 8)
QUEUES = 8
QSTRIDE = 16  # i32 between two ticket counters (separate cache lines)
SCHED_EXPECTED = QUEUES * QSTRIDE
BLOCK_M = 128
# A buffer the kernels address by row stays below 2 GiB: byte offsets are i32, and a
# masked lane's buffer access goes to offset 0x7FFFFFFF, out of range only below 2 GiB.
BUFFER_BYTES_MAX = 1 << 31
# status (local, i32 [8]): [1] rows the last call needed (prep kernel); [STATUS_EPOCH]
# calls completed (the finish kernel advances it; prep / finish flag the call with it + 1)
STATUS_EPOCH = 2


def ws_layout(E_LOCAL, NPES):
    """(E_GLOBAL, CFLAG, FFLAG, ARRIVED) with ``E_LOCAL`` experts on each of ``NPES``
    ranks: the experts of all ranks and the i32 offsets of the main path's workspace
    regions after the count matrix [NPES][E_GLOBAL] (see comm.py)."""
    assert NPES in EP_SIZES, NPES
    E_GLOBAL = NPES * E_LOCAL
    CFLAG = NPES * E_GLOBAL
    return E_GLOBAL, CFLAG, CFLAG + NPES, CFLAG + 2 * NPES


def row_bits(E_GLOBAL):
    """Row bits of a route's send info (plan kernel -> the main kernel's producers):
    destination row | global expert << ROW_BITS; the expert bits and the sign bit (-1:
    not sent) take the rest of the i32 (256 experts: 23 row bits)."""
    return 31 - (E_GLOBAL - 1).bit_length()


# ---- small path (small_m.py): a rank's symmetric buffer, its control block ----------
SMALL_LINE = 64  # bytes per cross-rank flag / counter (one cache line each)


def small_sym_layout(H, TOPK, TMAX, E_LOCAL, NPES):
    """(byte offset of each region, total bytes) of a rank's symmetric buffer (``NPES``
    ranks): fixed slots
    of TMAX rows per local expert (``rx`` fp8 rows, ``rs`` f32 scale per (row, 128 cols),
    ``rw`` route weight, ``rid`` source id), ``run`` (rows the sources took per expert,
    their atomics), ``done`` (source rank s's producers' count of finished calls, in
    DONE_UNIT steps), ``cin`` (bf16 rows the experts send back, per (token, topk slot) of
    this rank), ``cflag`` (rank s sent all its rows back: the epoch)."""
    assert NPES in EP_SIZES, NPES
    rows = E_LOCAL * TMAX
    sizes = (
        ("rx", rows * H),
        ("rs", rows * (H // 128) * 4),
        ("rw", rows * 4),
        ("rid", rows * 4),
        ("run", E_LOCAL * SMALL_LINE),
        ("done", NPES * SMALL_LINE),
        ("cin", TMAX * TOPK * H * 2),
        ("cflag", NPES * SMALL_LINE),
    )
    off, pos = {}, 0
    for name, nbytes in sizes:
        off[name] = pos
        pos += -(-nbytes // 256) * 256
    return off, pos


def small_max_tiles(TOPK, TMAX, BM, E_LOCAL):
    """m-tiles a rank can receive: TMAX tokens, each with <= min(topk, E_LOCAL) routes
    here."""
    return TMAX * min(TOPK, E_LOCAL) // BM + E_LOCAL


# control block (local, int32; one 64 B line per counter): [SC_EPOCH] the calls completed
# (the epoch of a call is it + 1), the combine count (top, then one per block id % 8), the
# unit counter of each of the 8 unit slices (one per XCD, two per XCD on MI308X), then
# the gemm1 units done of each m-tile
SC_EPOCH, SC_TOP = 0, 32
SC_SUB = 48
SC_HEAD = SC_SUB + 8 * 16
SC_TILE = SC_HEAD + 8 * 16
