# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
#
# Backward pass of jagged_dense_bmm_broadcast_add (jdbba).
#
# Given the upstream gradient dOut (L, N) of the forward
#     Out[s:e, :] = Jagged[s:e, :] @ Dense[b] + Bias[b][None, :]
# this module produces, per group b over its packed row slice [s, e):
#     dJagged[s:e, :] = dOut[s:e, :] @ Dense[b].T        (M_b x K)
#     dDense[b]       = Jagged[s:e, :].T @ dOut[s:e, :]   (K x N)
#     dBias[b]        = sum_m dOut[s:e, :]                (N,)
#
# dJagged is a per-row-independent GEMM (contraction over the static N axis).
# dDense and dBias both contract over the dynamic sequence axis m, and are
# computed as a two-pass split-reduction over m (a partials kernel writing fp32
# scratch, then a reduce kernel) to avoid serializing the reduction.
#
# bf16 in/out, fp32 accumulate. Targets CDNA (gfx942 / gfx950) like the forward.

from __future__ import annotations

import flydsl.expr as fx

# Sibling import (script dir is on sys.path[0]); shares the forward tile/shape
# constants so forward and backward stay in lockstep.
from jagged_dense_bmm import BLOCK_K, BLOCK_M, BLOCK_N, K, N, N_BLOCKS  # noqa: F401

# Split factor over the jagged (m) axis for the dDense / dBias reductions. Each
# (group, split) block owns one fp32 scratch slot; the reduce pass sums them.
SPLIT = 4


def grad_jagged(
    dJagged: fx.Tensor,      # out    (L, K)            bf16
    dOut: fx.Tensor,         # grad   (L, N)            bf16
    DENSE: fx.Tensor,        # dense  (n_groups * K, N) bf16  (plain, K-major per group)
    SEQ_OFFSETS: fx.Tensor,  # (n_groups + 1,) int32
    n_groups: int,
    max_seq_len: int,
    stream: fx.Stream = fx.Stream(None),
):
    """dJagged[s:e, :] = dOut[s:e, :] @ Dense[b].T, per group.

    Contraction is over the static N axis, so this is a clean per-row GEMM.
    """
    raise NotImplementedError("grad_jagged kernel not yet implemented")


def grad_bias(
    dBias: fx.Tensor,        # out    (n_groups, N)        bf16
    dOut: fx.Tensor,         # grad   (L, N)               bf16
    SEQ_OFFSETS: fx.Tensor,  # (n_groups + 1,) int32
    n_groups: int,
    max_seq_len: int,
    partials: fx.Tensor = None,  # fp32 scratch (n_groups, split, N)
    split: int = SPLIT,
    stream: fx.Stream = fx.Stream(None),
):
    """dBias[b] = sum_m dOut[s:e, :], per group.

    Two-pass split-reduction over m: a partials pass writes `split` fp32 partial
    row-sums per group, then a reduce pass sums them into bf16 dBias.
    """
    raise NotImplementedError("grad_bias kernels not yet implemented")


def grad_dense(
    dDense: fx.Tensor,       # out    (n_groups, K, N)      bf16
    JAGGED: fx.Tensor,       # jagged (L, K)                bf16
    dOut: fx.Tensor,         # grad   (L, N)                bf16
    SEQ_OFFSETS: fx.Tensor,  # (n_groups + 1,) int32
    n_groups: int,
    max_seq_len: int,
    partials: fx.Tensor = None,  # fp32 scratch (n_groups, split, K, N)
    split: int = SPLIT,
    stream: fx.Stream = fx.Stream(None),
):
    """dDense[b] = Jagged[s:e, :].T @ dOut[s:e, :], per group.

    Contraction is over the dynamic sequence axis m. Two-pass split-reduction:
    a partials pass accumulates fp32 (K, N) partials per (group, split) slot,
    then a reduce pass sums them into bf16 dDense.
    """
    raise NotImplementedError("grad_dense kernels not yet implemented")
