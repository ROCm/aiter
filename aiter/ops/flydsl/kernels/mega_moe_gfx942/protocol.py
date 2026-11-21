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
NPES, E_LOCAL = 8, 32
E_GLOBAL = NPES * E_LOCAL
CFLAG = NPES * E_GLOBAL
FFLAG = CFLAG + NPES
ARRIVED = FFLAG + NPES
QUEUES = 8
QSTRIDE = 16  # i32 between two ticket counters (separate cache lines)
SCHED_EXPECTED = QUEUES * QSTRIDE
# a route's send info (plan kernel -> the main kernel's producers): destination row |
# global expert << ROW_BITS; 8 expert bits and the sign bit (-1: not sent) leave 23 row
# bits
ROW_BITS = 23
BLOCK_M = 128
# A buffer the kernels address by row stays below 2 GiB: byte offsets are i32, and a
# masked lane's buffer access goes to offset 0x7FFFFFFF, out of range only below 2 GiB.
BUFFER_BYTES_MAX = 1 << 31
# status (local, i32 [8]): [1] rows the last call needed (prep kernel); [STATUS_EPOCH]
# calls completed (the finish kernel advances it; prep / finish flag the call with it + 1)
STATUS_EPOCH = 2
