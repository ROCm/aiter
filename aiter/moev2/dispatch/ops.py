# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""The MoE op chain: every plan covers it with segments, in order.

NOTE: aiter.moev2 is under active development (MoE v2 migration, not graduated).
Please do not modify this code without coordinating with the moev2 owners (@qli88). See
aiter/moev2/README.md.
"""

import enum


class Op(enum.IntEnum):
    SORT = 0  # route tokens to experts: sorted ids, weights, expert ids, output buffer
    QUANT_A1 = 1  # activations into GEMM1's input format
    GEMM1 = 2  # gate/up projection
    ACT = 3  # activation (and gate * up)
    QUANT_INTER = 4  # intermediate into GEMM2's input format
    GEMM2 = 5  # down projection
    REDUCE = 6  # weighted sum over each token's experts, into the output
    TAIL = 7  # fused epilogue after the MoE (only when the problem has a tail)


def chain(problem) -> tuple:
    """The ops a plan for `problem` must cover."""
    ops = tuple(op for op in Op if op is not Op.TAIL)
    return ops + (Op.TAIL,) if problem.tail is not None else ops
