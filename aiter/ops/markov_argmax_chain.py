# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Greedy K-step chain of a low-rank first-order Markov bias, in one launch (gfx1250).

With ``x_0 = ids[0]``, for ``k = 0 .. K-1``::

    rows_out[k] = W1[x_k]
    x_{k+1}     = argmax_v( float(base_logits[k, v]) + W1[x_k] . W2[v] )   # ids[k + 1]

``W1 @ W2^T`` (``W1, W2`` of shape ``[V, R]``) factorizes a ``V x V`` transition
bias. A block drafter for speculative decoding samples its K draft positions
left to right like this, every step's argmax picking the next step's W1 row;
K launches of an argmax over ``base + bias`` compute the same chain.

One persistent launch runs all K steps: one workgroup per CU holds its share
of W2 in LDS and registers for the whole chain, and the workgroups exchange
each step's (max, id) through a workspace in global memory. Each step's dot
products accumulate in fp32 over 32-wide chunks of R in order (one WMMA
chain); the argmax takes the lowest id among equal maxima, never a NaN, and
id 0 for a step without a non-NaN logit.

The exchange needs every workgroup resident at once. Its waits are bounded so
that other work holding CUs cannot hang the launch: a wait that gives up sets
bit ``k`` of ``workspace.err`` and step ``k``'s ids are not to be trusted.
``err`` is sticky; the caller reads it (without synchronizing, if it likes)
and clears it.
"""

import functools
from typing import NamedTuple

import torch

from ..jit.core import compile_ops
from ..jit.utils.build_targets import is_gfx1250_family
from ..jit.utils.chip_info import get_gfx_runtime_for_device

# The ctypes ABI follows the signature positionally; keep it in sync with
# csrc/kernels/markov_argmax_chain.cu.
MD_NAME = "module_markov_argmax_chain"

__all__ = [
    "MARKOV_ARGMAX_CHAIN_MAX_STEPS",
    "MARKOV_ARGMAX_CHAIN_RANKS",
    "MarkovArgmaxChainWorkspace",
    "markov_argmax_chain",
    "markov_argmax_chain_supported",
    "markov_argmax_chain_workspace",
]

# Ranks R the kernel is built for, and the most steps a launch runs (one bit of
# the error word a step).
MARKOV_ARGMAX_CHAIN_RANKS = (128, 256)
MARKOV_ARGMAX_CHAIN_MAX_STEPS = 32

# Mirrors of the kernel's constants (csrc/kernels/markov_argmax_chain.cu).
_MAX_WORKGROUPS = 256  # one polling thread a workgroup
_ID_LIMIT = 1 << 18  # ids travel in an 18-bit field, 0x3ffff the "no id" value
_LDS_BYTES = 327680  # LDS of a gfx1250 workgroup
_LDS_SCRATCH = 512
_MAX_LDS_TILES = 48  # 8 waves x 6 LDS tiles a wave
_VGPR_TILES = 16  # 8 waves x 2 register tiles a wave


def _max_tiles_per_workgroup(rank: int) -> int:
    """16-id vocab tiles one workgroup can hold at ``rank``: LDS, then VGPRs."""
    lds_tiles = (_LDS_BYTES - _LDS_SCRATCH) // (16 * (2 * rank + 16))
    return min(lds_tiles, _MAX_LDS_TILES) + _VGPR_TILES


@functools.cache
def _device_caps(index: int) -> tuple[bool, int]:
    """(gfx1250 family, CU count) of CUDA device ``index``."""
    arch = get_gfx_runtime_for_device(index)
    return (
        is_gfx1250_family(arch),
        torch.cuda.get_device_properties(index).multi_processor_count,
    )


def _device_index(device) -> int:
    if device is None:
        return torch.cuda.current_device()
    device = torch.device(device)
    return torch.cuda.current_device() if device.index is None else device.index


def markov_argmax_chain_supported(
    vocab_size: int,
    rank: int,
    num_steps: int,
    *,
    logits_dtype: torch.dtype = torch.bfloat16,
    table_dtype: torch.dtype = torch.bfloat16,
    id_dtype: torch.dtype = torch.int32,
    num_workgroups: int | None = None,
    device: torch.device | int | str | None = None,
) -> bool:
    """Whether :func:`markov_argmax_chain` runs this shape on ``device``.

    True for: a gfx1250 device; bf16 logits and tables; int32 or int64 ids;
    ``rank`` in ``MARKOV_ARGMAX_CHAIN_RANKS``; ``1 <= num_steps <=
    MARKOV_ARGMAX_CHAIN_MAX_STEPS``; ``vocab_size`` a multiple of 16 below
    2**18; and the vocabulary fitting the grid: each of ``num_workgroups``
    (default: the CU count; at most the CU count and 256) workgroups holds
    ``ceil(vocab_size / 16 / num_workgroups)`` 16-id tiles, at most 54 at
    rank 256 and 64 at rank 128 (221168 / 262128 ids on 256 CUs).

    Layouts are the caller's part (see :func:`markov_argmax_chain`). Does not
    build or launch anything; no available ROCm device answers False.
    """
    if (
        rank not in MARKOV_ARGMAX_CHAIN_RANKS
        or not 1 <= num_steps <= MARKOV_ARGMAX_CHAIN_MAX_STEPS
        or vocab_size < 16
        or vocab_size % 16
        or vocab_size >= _ID_LIMIT
        or logits_dtype != torch.bfloat16
        or table_dtype != torch.bfloat16
        or id_dtype not in (torch.int32, torch.int64)
    ):
        return False
    if torch.version.hip is None or not torch.cuda.is_available():
        return False
    gfx1250, num_cus = _device_caps(_device_index(device))
    if not gfx1250:
        return False
    ncta = num_cus if num_workgroups is None else num_workgroups
    if not 1 <= ncta <= min(num_cus, _MAX_WORKGROUPS):
        return False
    return (vocab_size // 16) // ncta + 1 <= _max_tiles_per_workgroup(rank)


class MarkovArgmaxChainWorkspace(NamedTuple):
    """State a run of :func:`markov_argmax_chain` launches shares.

    ``slots``: ``[max_steps, num_workgroups]`` int64, the exchange slots; its
    column count is the grid. Zeroed once, then kept: each launch reads the
    tag of the one before from it. ``err``: ``[1]`` int32, bit ``k`` set when
    a wait of step ``k`` gave up; sticky until the caller clears it.

    One launch at a time: launches sharing a workspace must be ordered (one
    stream, or synchronized).
    """

    slots: torch.Tensor
    err: torch.Tensor


def markov_argmax_chain_workspace(
    max_steps: int,
    device: torch.device | int | str | None = None,
    num_workgroups: int | None = None,
) -> MarkovArgmaxChainWorkspace:
    """A zeroed workspace for launches of up to ``max_steps`` steps.

    ``num_workgroups`` defaults to the device's CU count, one workgroup a CU.
    """
    index = _device_index(device)
    if num_workgroups is None:
        num_workgroups = torch.cuda.get_device_properties(index).multi_processor_count
    if not 1 <= max_steps <= MARKOV_ARGMAX_CHAIN_MAX_STEPS:
        raise ValueError(
            f"max_steps {max_steps} not in 1..{MARKOV_ARGMAX_CHAIN_MAX_STEPS}"
        )
    dev = torch.device("cuda", index)
    return MarkovArgmaxChainWorkspace(
        torch.zeros(max_steps, num_workgroups, dtype=torch.int64, device=dev),
        torch.zeros(1, dtype=torch.int32, device=dev),
    )


@compile_ops(MD_NAME, fc_name="markov_argmax_chain_fwd", ffi_type="ctypes")
def _markov_argmax_chain_fwd(
    base_logits: torch.Tensor,
    ids: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    rows_out: torch.Tensor,
    slots: torch.Tensor,
    err: torch.Tensor,
) -> None:
    """Write ``ids[1:]`` and ``rows_out``; see :func:`markov_argmax_chain`."""


def markov_argmax_chain(
    base_logits: torch.Tensor,
    ids: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    workspace: MarkovArgmaxChainWorkspace,
    rows_out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run the K-step chain in one launch; return ``rows_out``.

    ``base_logits``: ``[K, V]`` bf16, unit stride along V (any row stride).
    ``ids``: ``[K + 1]`` int32 or int64, any stride; ``ids[0]`` (in
    ``[0, V)``) is the anchor, ``ids[1:]`` receive the K argmaxes.
    ``w1``, ``w2``: ``[V, R]`` bf16, contiguous, 16-byte aligned.
    ``rows_out``: ``[K, R]`` bf16 with unit stride along R and 16-byte
    aligned rows, receives ``W1[x_k]``; allocated when None.
    ``workspace``: from :func:`markov_argmax_chain_workspace`, with at least K
    rows, on the same device.

    The shape must pass :func:`markov_argmax_chain_supported` for the
    workspace's column count; the launcher re-checks every condition and
    raises instead of launching. No host synchronization and no allocation
    besides ``rows_out``, so it can be captured in a CUDA graph (with the
    workspace allocated outside the graph's pool).
    """
    if rows_out is None:
        rows_out = torch.empty(
            base_logits.shape[0], w1.shape[1], dtype=w1.dtype, device=w1.device
        )
    _markov_argmax_chain_fwd(
        base_logits, ids, w1, w2, rows_out, workspace.slots, workspace.err
    )
    return rows_out
