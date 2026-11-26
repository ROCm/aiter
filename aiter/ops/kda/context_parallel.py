# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Chunkwise context parallelism for the KDA recurrence.

Each CP rank holds one contiguous token interval of a packed (THD) batch. Only
the first local sequence can continue a sequence from earlier ranks, and only
the last local sequence can continue into later ranks. The state at a rank
boundary is an affine function of the state at the previous one::

    S_end = M @ S_start + S_ext

``S_ext`` (``[HV, K, V]``) is the end state with a zero start, ``M``
(``[HV, K, K]``) the transition. Both come from the local chunk kernels: the
first is the final state of the local run, the second the final state of a run
with ``v = 0`` and ``initial_state = I``. That second run's output ``O'``
(``[T, HV, K]``) is ``scale * M_t^T q_t``, so the contribution of an incoming
state ``S0`` to the output is ``O' @ S0`` and its gradient is ``O'^T dO``.

Forward, per rank:
    1. local recurrence with a zero incoming state;
    2. transition run over the boundary sequences;
    3. all-gather ``[S_ext, M]`` of every rank's last sequence and chain the
       ``pre_num_ranks`` packets before this rank into ``S0``;
    4. correction ``o[:first_end] += O' @ S0``.
Backward, per rank:
    1. ``D = O'^T dO (+ M^T dht)`` for the first sequence;
    2. all-gather ``[D, M^T]`` and chain the ``post_num_ranks`` packets after
       this rank, in reverse, into the gradient of the last sequence's final
       state;
    3. one local backward with the true incoming state ``S0`` and that ``dht``.

The packet layout and both chains follow FLA's chunkwise CP
(``fla.ops.cp``) and Megatron's cuDNN FROST chain: fp32 throughout.
"""

from __future__ import annotations

import bisect
import itertools
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist

__all__ = [
    "KdaCPBoundary",
    "KdaCPContext",
    "all_gather_boundary_packets",
    "build_kda_cp_context",
    "kda_cp_boundary",
    "kda_state_chain",
    "pack_boundary",
]


@dataclass
class KdaCPContext:
    """Rank-local partition metadata, field-compatible with FLA's ``FLACPContext``.

    Any object with these attributes works as a ``cp_context``, including
    ``fla.ops.cp.build_cp_context``'s result (contiguous layout only).

    ``group`` is a ``torch.distributed`` process group, or any object with
    ``rank()``, ``size()`` and ``all_gather_into_tensor(output, input)``.
    """

    group: Any = None
    cu_seqlens: torch.Tensor | None = None
    cu_seqlens_cpu: torch.Tensor | None = None
    is_first_rank: bool | None = None
    is_last_rank: bool | None = None
    pre_num_ranks: int | None = None
    post_num_ranks: int | None = None
    conv1d_kernel_size: int | None = None
    pre_num_conv_tokens: int | None = None
    layout: str = "contiguous"

    @property
    def num_seqs(self) -> int:
        return 0 if self.cu_seqlens is None else self.cu_seqlens.numel() - 1

    @property
    def is_cp_enabled(self) -> bool:
        return self.group is not None


def _group_rank_size(group) -> tuple[int, int]:
    if isinstance(group, dist.ProcessGroup):
        return dist.get_rank(group=group), dist.get_world_size(group=group)
    return group.rank(), group.size()


def build_kda_cp_context(
    cu_seqlens: torch.Tensor,
    group,
    conv1d_kernel_size: int | None = None,
    cu_seqlens_cpu: torch.Tensor | None = None,
    rank: int | None = None,
    world_size: int | None = None,
) -> KdaCPContext:
    """Split global ``cu_seqlens`` into this rank's contiguous interval.

    Same partition as ``fla.ops.cp.build_cp_context(layout="contiguous")``, for
    environments without FLA: ``T / world_size`` tokens per rank, and the
    neighbour counts of the first and last local sequences.
    """
    if rank is None or world_size is None:
        rank, world_size = _group_rank_size(group)
    bounds = (cu_seqlens if cu_seqlens_cpu is None else cu_seqlens_cpu).tolist()
    total = bounds[-1]
    if total < world_size or total % world_size:
        raise ValueError(
            f"total tokens ({total}) must be a positive multiple of the CP size "
            f"({world_size})."
        )
    part = total // world_size
    start, end = part * rank, part * (rank + 1)
    first = bisect.bisect_right(bounds, start) - 1
    last = bisect.bisect_left(bounds, end)
    local = sorted({min(max(b, start), end) - start for b in bounds[first : last + 1]})
    first_pos = bounds[first] // part
    last_pos = (bounds[last] - 1) // part
    local_cpu = torch.tensor(local, dtype=torch.int32)
    return KdaCPContext(
        group=group,
        cu_seqlens=local_cpu.to(cu_seqlens.device, non_blocking=True),
        cu_seqlens_cpu=local_cpu,
        is_first_rank=rank == first_pos,
        is_last_rank=rank == last_pos,
        pre_num_ranks=rank - first_pos,
        post_num_ranks=last_pos - rank,
        conv1d_kernel_size=conv1d_kernel_size,
        pre_num_conv_tokens=start - bounds[first],
    )


@dataclass(frozen=True)
class KdaCPBoundary:
    """What one rank needs from a ``cp_context``: its peers and its edge sequences.

    ``first_end`` is the local end of the first sequence, ``last_start`` the
    local start of the last one; they coincide with ``0`` / ``num_tokens`` when
    the rank holds a single sequence.
    """

    group: Any
    rank: int
    world_size: int
    pre_num_ranks: int
    post_num_ranks: int
    num_seqs: int
    num_tokens: int
    first_end: int
    last_start: int


def kda_cp_boundary(cp_context, num_tokens: int) -> KdaCPBoundary:
    """Validate a contiguous-layout ``cp_context`` against ``num_tokens`` local tokens."""
    if getattr(cp_context, "layout", "contiguous") != "contiguous":
        raise NotImplementedError(
            "aiter KDA chunkwise CP supports the contiguous layout only, got "
            f"{cp_context.layout!r}."
        )
    group = cp_context.group
    if group is None or cp_context.cu_seqlens is None:
        raise ValueError(
            "KDA chunkwise CP needs cp_context.group and cp_context.cu_seqlens."
        )
    rank, world_size = _group_rank_size(group)
    pre, post = cp_context.pre_num_ranks, cp_context.post_num_ranks
    if pre is None or not 0 <= pre <= rank:
        raise ValueError(f"invalid pre_num_ranks={pre} for CP rank {rank}.")
    if post is None or not 0 <= post < world_size - rank:
        raise ValueError(
            f"invalid post_num_ranks={post} for CP rank {rank} of {world_size}."
        )
    cu_cpu = cp_context.cu_seqlens_cpu
    if cu_cpu is None:
        cu_cpu = cp_context.cu_seqlens.cpu()
    bounds = cu_cpu.tolist()
    if len(bounds) < 2 or bounds[0] != 0 or bounds[-1] != num_tokens:
        raise ValueError(
            f"cp_context.cu_seqlens {bounds} must span the {num_tokens} local tokens."
        )
    if any(a >= b for a, b in itertools.pairwise(bounds)):
        raise ValueError(f"cp_context.cu_seqlens {bounds} must be strictly increasing.")
    return KdaCPBoundary(
        group=group,
        rank=rank,
        world_size=world_size,
        pre_num_ranks=pre,
        post_num_ranks=post,
        num_seqs=len(bounds) - 1,
        num_tokens=num_tokens,
        first_end=bounds[1],
        last_start=bounds[-2],
    )


def pack_boundary(state: torch.Tensor, transition: torch.Tensor) -> torch.Tensor:
    """``[HV, K, V]`` state and ``[HV, K, K]`` transition as one fp32 ``[HV, K, V + K]`` packet."""
    return torch.cat((state.float(), transition.float()), dim=-1).contiguous()


def all_gather_boundary_packets(packet: torch.Tensor, group) -> torch.Tensor:
    """Gather one equal-shaped packet per rank: ``[world_size, *packet.shape]``."""
    packet = packet.contiguous()
    _, world_size = _group_rank_size(group)
    out = packet.new_empty((world_size * packet.shape[0], *packet.shape[1:]))
    if isinstance(group, dist.ProcessGroup):
        dist.all_gather_into_tensor(out, packet, group=group)
    else:
        group.all_gather_into_tensor(out, packet)
    return out.view(world_size, *packet.shape)


def kda_state_chain(
    packets: torch.Tensor, value_dim: int, *, reverse: bool = False
) -> torch.Tensor | None:
    """Compose consecutive ranks' ``[S_ext, M]`` packets.

    Forward (``reverse=False``) walks ``packets`` in order with
    ``S <- M_j @ S + S_ext_j``; the backward walks them last to first with the
    packets' transposed transitions. The first packet's transition is never
    applied, since the chain starts from zero. Returns ``None`` for no packets.
    """
    if packets.shape[0] == 0:
        return None
    order: Sequence[int] = range(packets.shape[0])
    if reverse:
        order = reversed(order)
    state = None
    for j in order:
        h = packets[j, ..., :value_dim]
        state = (
            h.clone()
            if state is None
            else torch.matmul(packets[j, ..., value_dim:], state) + h
        )
    return state
