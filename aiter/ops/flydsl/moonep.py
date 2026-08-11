# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""MoonEP-style planning contracts for the gfx950 FlyDSL prototype.

The first prototype intentionally keeps planning host-driven.  It provides a
small, deterministic reference that the later FlyDSL planner and direct-P2P
dispatch kernels can be checked against field by field.

``hipblaslt_grouped_gemm_reference`` is likewise a correctness baseline.  It
launches one hipBLASLt GEMM per non-empty VM group and is not intended to be the
final high-performance variable-M implementation.
"""

from __future__ import annotations

from dataclasses import dataclass

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch


@dataclass(frozen=True)
class MoonEPPlanConfig:
    """Static dimensions used by the MoonEP-style planner.

    ``num_tokens`` is the number of source tokens on every EP rank.  The
    planner therefore targets exactly ``num_tokens * top_k`` routed entries per
    destination rank.
    """

    rank: int
    world_size: int
    num_tokens: int
    top_k: int
    num_experts: int
    prefetch_slots: int | None = None
    token_padding: int = 1
    # Decode mode: keep every expert on its home rank (no balancing, no
    # migration). See MoonEPPlanGeometry.NO_MIG.
    no_migration: bool = False

    def __post_init__(self) -> None:
        if self.world_size <= 0:
            raise ValueError("world_size must be positive")
        if not 0 <= self.rank < self.world_size:
            raise ValueError(
                f"rank must be in [0, {self.world_size}), got {self.rank}"
            )
        if self.num_tokens <= 0:
            raise ValueError("num_tokens must be positive")
        if self.top_k <= 0:
            raise ValueError("top_k must be positive")
        if self.num_experts <= 0 or self.num_experts % self.world_size != 0:
            raise ValueError(
                "num_experts must be positive and divisible by world_size"
            )
        if self.token_padding <= 0:
            raise ValueError("token_padding must be positive")

        slots = self.experts_per_rank if self.prefetch_slots is None else int(
            self.prefetch_slots
        )
        if slots <= 0:
            raise ValueError("prefetch_slots must be positive")
        object.__setattr__(self, "prefetch_slots", slots)

    @property
    def experts_per_rank(self) -> int:
        return self.num_experts // self.world_size

    @property
    def capacity(self) -> int:
        return self.num_tokens * self.top_k

    @property
    def num_dispatch_rows(self) -> int:
        if self.no_migration:
            # Without balancing a destination is no longer capped at capacity:
            # in the worst case every rank routes all of its entries to experts
            # owned by one rank, so bound by world_size * capacity. Only the
            # local home groups can be non-empty, so the padding term loses its
            # factor of two.
            return self.world_size * self.capacity + self.experts_per_rank * (
                self.token_padding - 1
            )
        # At most one remote home group and one local home group contribute
        # non-empty expert segments to a destination rank.  Each segment may
        # need token_padding - 1 rows of padding.
        padding_rows = (
            2 * self.experts_per_rank * (self.token_padding - 1)
        )
        return self.capacity + padding_rows


@dataclass(frozen=True)
class MoonEPReferencePlan:
    """Reference outputs consumed by the gfx950 prototype.

    ``dst`` has shape ``[S, K]``.  Non-negative entries encode
    ``dest_rank * NvS + local_offset``.  Later top-k entries for the same token
    and destination rank use ``-raw_dst - 1`` so dispatch can skip duplicate
    hidden payloads while retaining the raw destination.
    """

    config: MoonEPPlanConfig
    dst: torch.Tensor
    cu_seqlens: torch.Tensor
    experts_to_copy: torch.Tensor
    zero_fill_ranges: torch.Tensor
    remote_stats: torch.Tensor
    alloc: torch.Tensor
    group_expert_ids: torch.Tensor

    @property
    def N(self) -> int:
        return self.config.capacity

    @property
    def R(self) -> int:
        return self.config.world_size

    @property
    def E(self) -> int:
        return self.config.num_experts

    @property
    def B(self) -> int:
        return int(self.config.prefetch_slots)

    @property
    def NvS(self) -> int:
        return self.config.num_dispatch_rows

    @property
    def K(self) -> int:
        return self.config.top_k

    def clone(self) -> "MoonEPReferencePlan":
        """Clone plan-owned tensors for safe reuse, matching MoonEPCommPlan."""

        return type(self)(
            config=self.config,
            dst=self.dst.clone(),
            cu_seqlens=self.cu_seqlens.clone(),
            experts_to_copy=self.experts_to_copy.clone(),
            zero_fill_ranges=self.zero_fill_ranges.clone(),
            remote_stats=self.remote_stats.clone(),
            alloc=self.alloc.clone(),
            group_expert_ids=self.group_expert_ids.clone(),
        )

    @staticmethod
    def decode_dst(dst: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(raw_dst, is_primary)`` without losing duplicate slots."""

        is_primary = dst >= 0
        raw_dst = torch.where(is_primary, dst, -dst - 1)
        return raw_dst, is_primary


def _validate_planning_inputs(
    config: MoonEPPlanConfig,
    topk_experts: torch.Tensor,
    tokens_per_expert: torch.Tensor,
) -> None:
    expected_topk = (config.num_tokens, config.top_k)
    expected_tpe = (config.world_size, config.num_experts)
    if tuple(topk_experts.shape) != expected_topk:
        raise ValueError(
            f"topk_experts must have shape {expected_topk}, "
            f"got {tuple(topk_experts.shape)}"
        )
    if tuple(tokens_per_expert.shape) != expected_tpe:
        raise ValueError(
            f"tokens_per_expert must have shape {expected_tpe}, "
            f"got {tuple(tokens_per_expert.shape)}"
        )
    if topk_experts.dtype not in (torch.int32, torch.int64):
        raise TypeError("topk_experts must be int32 or int64")
    if tokens_per_expert.dtype not in (torch.int32, torch.int64):
        raise TypeError("tokens_per_expert must be int32 or int64")
    if bool(((topk_experts < 0) | (topk_experts >= config.num_experts)).any()):
        raise ValueError("topk_experts contains an out-of-range expert id")
    if bool((tokens_per_expert < 0).any()):
        raise ValueError("tokens_per_expert must be non-negative")

    row_totals = tokens_per_expert.to(torch.int64).sum(dim=1)
    expected_total = torch.full_like(row_totals, config.capacity)
    if not torch.equal(row_totals.cpu(), expected_total.cpu()):
        raise ValueError(
            "every tokens_per_expert row must sum to num_tokens * top_k"
        )

    local_hist = torch.bincount(
        topk_experts.reshape(-1).to(torch.int64).cpu(),
        minlength=config.num_experts,
    )
    if not torch.equal(
        local_hist,
        tokens_per_expert[config.rank].to(torch.int64).cpu(),
    ):
        raise ValueError(
            "topk_experts histogram does not match tokens_per_expert[rank]"
        )


def build_reference_plan(
    config: MoonEPPlanConfig,
    topk_experts: torch.Tensor,
    tokens_per_expert: torch.Tensor,
) -> MoonEPReferencePlan:
    """Build the deterministic MoonEP plan for one source rank.

    The implementation deliberately runs on CPU.  Inputs may reside on a GPU;
    returned tensors are copied back to ``topk_experts.device`` so they can feed
    the first FlyDSL dispatch prototype directly.
    """

    _validate_planning_inputs(config, topk_experts, tokens_per_expert)

    output_device = topk_experts.device
    topk = topk_experts.to(device="cpu", dtype=torch.int64).contiguous()
    tpe = tokens_per_expert.to(device="cpu", dtype=torch.int64).contiguous()

    rank = config.rank
    world_size = config.world_size
    num_experts = config.num_experts
    experts_per_rank = config.experts_per_rank
    capacity = config.capacity
    num_rows = config.num_dispatch_rows
    num_slots = int(config.prefetch_slots)

    # cumsum for tokens of each expert, last element is total tokens of each expert
    tpe_cumsum = tpe.cumsum(dim=0)
    expert_count = tpe_cumsum[-1]
    
    group_tokens = expert_count.view(world_size, experts_per_rank).sum(dim=1)
    balance = group_tokens - capacity

    # alloc[e, d] is the number of expert-e routed entries executed by d.
    alloc = torch.zeros(num_experts, world_size, dtype=torch.int64)
    for expert in range(num_experts):
        alloc[expert, expert // experts_per_rank] = expert_count[expert]

    # First choose how many entries each overloaded home group moves to each
    # underloaded destination rank.
    quotas_by_home = torch.zeros(world_size, world_size, dtype=torch.int64)
    while not config.no_migration:
        home = int(balance.argmax().item())
        dest = int(balance.argmin().item())
        if int(balance[home].item()) <= 0:
            break
        move = -int(balance[dest].item())
        if move <= 0:
            raise AssertionError("positive surplus without a receiver deficit")
        quotas_by_home[home, dest] = move
        balance[home] -= move
        balance[dest] = 0

    # Resolve each home-group quota into exact expert allocations.
    for home in range(world_size):
        expert_begin = home * experts_per_rank
        expert_end = expert_begin + experts_per_rank
        remaining = expert_count[expert_begin:expert_end].clone()
        quotas = quotas_by_home[home].clone()

        while True:
            dest = int(quotas.argmax().item())
            quota = int(quotas[dest].item())
            if quota <= 0:
                break
            local_expert = int(remaining.argmax().item())
            expert = expert_begin + local_expert
            take = min(int(remaining[local_expert].item()), quota)
            if take <= 0:
                raise AssertionError("receiver quota cannot be satisfied")
            alloc[expert, dest] += take
            alloc[expert, home] -= take
            remaining[local_expert] -= take
            quotas[dest] -= take

    if not torch.equal(alloc.sum(dim=1), expert_count):
        raise AssertionError("per-expert token conservation failed")
    if not config.no_migration:
        expected_capacity = torch.full(
            (world_size,), capacity, dtype=torch.int64
        )
        if not torch.equal(alloc.sum(dim=0), expected_capacity):
            raise AssertionError("destination ranks are not perfectly balanced")

    alloc_cumsum = alloc.cumsum(dim=1)
    expert_offsets = torch.zeros(
        world_size, num_experts, dtype=torch.int64
    )
    all_cu_seqlens = torch.zeros(
        world_size, num_experts + num_slots, dtype=torch.int64
    )
    all_zero_fill = torch.zeros(
        world_size, num_experts + num_slots, 2, dtype=torch.int64
    )
    experts_to_copy = torch.full(
        (world_size, num_slots), -1, dtype=torch.int64
    )
    remote_stats = torch.zeros(world_size, 2, dtype=torch.int64)
    group_expert_ids = torch.full(
        (world_size, num_experts + num_slots), -1, dtype=torch.int64
    )

    for dest in range(world_size):
        local_begin = dest * experts_per_rank
        local_end = local_begin + experts_per_rank
        # alloc[expert,dest] => expert i has n routed to dest 
        # find remote expert which routes to dest
        remote_experts = [
            expert
            for expert in range(num_experts)
            if int(alloc[expert, dest].item()) > 0
            and not local_begin <= expert < local_end
        ]
        remote_experts.sort(
            key=lambda expert: (int(alloc[expert, dest].item()), expert),
            reverse=True,
        )

        # just get num_slot from remote
        remote_stats[dest, 0] = len(remote_experts)
        for slot, expert in enumerate(remote_experts[:num_slots]):
            experts_to_copy[dest, slot] = expert
            remote_stats[expert // experts_per_rank, 1] += 1

        selected = set(remote_experts[:num_slots])
        start = 0
        for group in range(num_experts + num_slots):
            count = 0
            expert = -1
            if group < num_experts:
                if group not in selected:
                    expert = group
                    count = int(alloc[expert, dest].item())
            else:
                slot = group - num_experts
                expert = int(experts_to_copy[dest, slot].item())
                if expert >= 0:
                    count = int(alloc[expert, dest].item())

            padded = (
                (count + config.token_padding - 1)
                // config.token_padding
                * config.token_padding
                if count > 0
                else 0
            )
            end = start + count
            padded_end = start + padded
            all_cu_seqlens[dest, group] = padded_end
            group_expert_ids[dest, group] = expert if count > 0 else -1
            if count > 0:
                expert_offsets[dest, expert] = start
                all_zero_fill[dest, group, 0] = end
                all_zero_fill[dest, group, 1] = padded - count
            start = padded_end

        if start > num_rows:
            raise AssertionError(
                f"destination {dest} layout uses {start} rows, capacity is {num_rows}"
            )

    # Assign each local routed entry to the exact destination segment slot.
    local_seen = torch.zeros(num_experts, dtype=torch.int64)
    dst = torch.empty(config.capacity, dtype=torch.int64)
    for flat_idx, expert_tensor in enumerate(topk.reshape(-1)):
        expert = int(expert_tensor.item())
        previous_sources = (
            0 if rank == 0 else int(tpe_cumsum[rank - 1, expert].item())
        )
        global_expert_index = previous_sources + int(local_seen[expert].item())
        local_seen[expert] += 1

        dest = int(
            torch.searchsorted(
                alloc_cumsum[expert], global_expert_index, right=True
            ).item()
        )
        if dest >= world_size:
            raise AssertionError("no destination rank covers routed entry")
        previous_dest_count = (
            0 if dest == 0 else int(alloc_cumsum[expert, dest - 1].item())
        )
        local_offset = int(expert_offsets[dest, expert].item()) + (
            global_expert_index - previous_dest_count
        )
        raw_dst = dest * num_rows + local_offset
        if raw_dst > torch.iinfo(torch.int32).max:
            raise OverflowError("dst encoding exceeds int32")
        dst[flat_idx] = raw_dst

    # Keep the first top-k entry per destination rank as the payload primary.
    # Later entries retain their raw destination as -raw_dst-1.
    dst = dst.view(config.num_tokens, config.top_k)
    for token in range(config.num_tokens):
        seen_ranks: set[int] = set()
        for k_idx in range(config.top_k):
            raw_dst = int(dst[token, k_idx].item())
            dest = raw_dst // num_rows
            if dest in seen_ranks:
                dst[token, k_idx] = -raw_dst - 1
            else:
                seen_ranks.add(dest)

    def out(tensor: torch.Tensor) -> torch.Tensor:
        return tensor.to(device=output_device, dtype=torch.int32)

    return MoonEPReferencePlan(
        config=config,
        dst=out(dst),
        cu_seqlens=out(all_cu_seqlens[rank]),
        experts_to_copy=out(experts_to_copy),
        zero_fill_ranges=out(all_zero_fill[rank]),
        remote_stats=out(remote_stats[rank]),
        alloc=out(alloc.t().contiguous()),
        group_expert_ids=out(group_expert_ids[rank]),
    )


def hipblaslt_grouped_gemm_reference(
    hidden: torch.Tensor,
    weights: torch.Tensor,
    cu_seqlens: torch.Tensor,
    *,
    solution_index: int = -1,
    output: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run one hipBLASLt GEMM per non-empty variable-M group.

    Args:
        hidden: Contiguous ``[NvS, H]`` BF16 token buffer.
        weights: Contiguous ``[G, H, Hout]`` BF16 weights.
        cu_seqlens: ``[G]`` cumulative padded row ends.
        solution_index: hipBLASLt solution id, ``-1`` asks the extension to
            choose its default solution.
        output: Optional contiguous ``[NvS, Hout]`` BF16 output buffer.

    This helper synchronizes ``cu_seqlens`` to the host and launches one GEMM
    per group.  It is a bring-up/reference path, not a performance endpoint.
    """

    if hidden.dtype != torch.bfloat16 or weights.dtype != torch.bfloat16:
        raise TypeError("hidden and weights must be BF16")
    if hidden.ndim != 2 or weights.ndim != 3 or cu_seqlens.ndim != 1:
        raise ValueError("expected hidden[rows,H], weights[G,H,Hout], cu_seqlens[G]")
    if not hidden.is_contiguous() or not weights.is_contiguous():
        raise ValueError("hidden and weights must be contiguous")
    if weights.shape[0] != cu_seqlens.numel():
        raise ValueError("weights and cu_seqlens group counts differ")
    if hidden.shape[1] != weights.shape[1]:
        raise ValueError("hidden H and weight H differ")
    if hidden.device != weights.device or hidden.device != cu_seqlens.device:
        raise ValueError("hidden, weights, and cu_seqlens must share a device")

    if output is None:
        output = torch.empty(
            hidden.shape[0],
            weights.shape[2],
            dtype=hidden.dtype,
            device=hidden.device,
        )
    if tuple(output.shape) != (hidden.shape[0], weights.shape[2]):
        raise ValueError("output has the wrong shape")
    if output.dtype != torch.bfloat16 or not output.is_contiguous():
        raise ValueError("output must be contiguous BF16")

    from aiter.ops.gradlib import hipb_create_extension, hipb_mm

    hipb_create_extension()
    start = 0
    for group, end in enumerate(cu_seqlens.to(device="cpu").tolist()):
        end = int(end)
        if end < start or end > hidden.shape[0]:
            raise ValueError("cu_seqlens must be monotonic and within hidden")
        if end > start:
            group_out = hipb_mm(
                hidden[start:end],
                weights[group],
                solution_index=solution_index,
                out_dtype=torch.bfloat16,
            )
            output[start:end].copy_(group_out)
        start = end
    return output


def hipblaslt_moonep_grouped_gemm_reference(
    hidden: torch.Tensor,
    home_weights: torch.Tensor,
    prefetched_weights: torch.Tensor,
    cu_seqlens: torch.Tensor,
    group_expert_ids: torch.Tensor,
    *,
    rank: int,
    experts_per_rank: int,
    num_experts: int,
    output: torch.Tensor,
    solution_index: int = -1,
) -> torch.Tensor:
    """Execute MoonEP physical groups using home or dynamic-slot weights.

    Groups ``[0, E)`` must resolve to an expert owned by this rank.  Groups
    ``[E, E+B)`` resolve to the corresponding prefetched slot.  This is the
    correctness baseline for the physical VM layout, not a grouped-GEMM
    performance implementation.
    """

    group_count = num_experts + prefetched_weights.shape[0]
    if cu_seqlens.numel() != group_count or group_expert_ids.numel() != group_count:
        raise ValueError("MoonEP group metadata has the wrong length")
    if tuple(home_weights.shape[1:]) != (hidden.shape[1], output.shape[1]):
        raise ValueError("home expert weight shape is incompatible")
    if tuple(prefetched_weights.shape[1:]) != (hidden.shape[1], output.shape[1]):
        raise ValueError("prefetched expert weight shape is incompatible")

    from aiter.ops.gradlib import hipb_create_extension, hipb_mm

    hipb_create_extension()
    ends = cu_seqlens.to(device="cpu").tolist()
    expert_ids = group_expert_ids.to(device="cpu").tolist()
    start = 0
    for group, (end, expert) in enumerate(zip(ends, expert_ids)):
        end = int(end)
        expert = int(expert)
        if end < start or end > hidden.shape[0]:
            raise ValueError("cu_seqlens must be monotonic and within hidden")
        if end > start:
            if expert < 0:
                raise ValueError("non-empty group is missing its expert id")
            if group < num_experts:
                owner = expert // experts_per_rank
                if owner != rank:
                    raise ValueError(
                        "remote expert group has no dynamic prefetch slot"
                    )
                weight = home_weights[expert % experts_per_rank]
            else:
                weight = prefetched_weights[group - num_experts]
            group_out = hipb_mm(
                hidden[start:end],
                weight,
                solution_index=solution_index,
                out_dtype=torch.bfloat16,
            )
            output[start:end].copy_(group_out)
        start = end
    return output


def hipblaslt_moonep_mlp_reference(
    hidden: torch.Tensor,
    home_gate: torch.Tensor,
    home_up: torch.Tensor,
    home_down: torch.Tensor,
    prefetched_gate: torch.Tensor,
    prefetched_up: torch.Tensor,
    prefetched_down: torch.Tensor,
    cu_seqlens: torch.Tensor,
    group_expert_ids: torch.Tensor,
    *,
    rank: int,
    experts_per_rank: int,
    num_experts: int,
    output: torch.Tensor,
    solution_index: int = -1,
    full_gate: torch.Tensor | None = None,
    full_up: torch.Tensor | None = None,
    full_down: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run ``down(silu(gate(x)) * up(x))`` for every physical VM group."""

    slots = prefetched_gate.shape[0]
    group_count = num_experts + slots
    if cu_seqlens.numel() != group_count or group_expert_ids.numel() != group_count:
        raise ValueError("MoonEP group metadata has the wrong length")
    if prefetched_up.shape[0] != slots or prefetched_down.shape[0] != slots:
        raise ValueError("prefetched gate/up/down slot counts differ")
    hidden_dim = hidden.shape[1]
    intermediate_dim = home_gate.shape[2]
    expected_gate = (hidden_dim, intermediate_dim)
    expected_down = (intermediate_dim, hidden_dim)
    for name, tensor in (
        ("home_gate", home_gate),
        ("home_up", home_up),
        ("prefetched_gate", prefetched_gate),
        ("prefetched_up", prefetched_up),
    ):
        if tuple(tensor.shape[1:]) != expected_gate:
            raise ValueError(f"{name} has the wrong shape")
    for name, tensor in (
        ("home_down", home_down),
        ("prefetched_down", prefetched_down),
    ):
        if tuple(tensor.shape[1:]) != expected_down:
            raise ValueError(f"{name} has the wrong shape")
    if tuple(output.shape) != (hidden.shape[0], hidden_dim):
        raise ValueError("expert MLP output shape mismatch")

    from aiter.ops.gradlib import hipb_create_extension, hipb_mm

    hipb_create_extension()
    ends = cu_seqlens.to(device="cpu").tolist()
    expert_ids = group_expert_ids.to(device="cpu").tolist()
    start = 0
    for group, (end, expert) in enumerate(zip(ends, expert_ids)):
        end = int(end)
        expert = int(expert)
        if end < start or end > hidden.shape[0]:
            raise ValueError("cu_seqlens must be monotonic and within hidden")
        if end > start:
            if expert < 0:
                raise ValueError("non-empty group is missing its expert id")
            if group < num_experts:
                if full_gate is not None and full_up is not None and full_down is not None:
                    gate_w = full_gate[expert]
                    up_w = full_up[expert]
                    down_w = full_down[expert]
                else:
                    owner = expert // experts_per_rank
                    if owner != rank:
                        raise ValueError(
                            "remote expert group has no dynamic prefetch slot or full weight view"
                        )
                    local = expert % experts_per_rank
                    gate_w = home_gate[local]
                    up_w = home_up[local]
                    down_w = home_down[local]
            else:
                slot = group - num_experts
                gate_w = prefetched_gate[slot]
                up_w = prefetched_up[slot]
                down_w = prefetched_down[slot]

            x = hidden[start:end]
            gate = hipb_mm(
                x, gate_w, solution_index=solution_index, out_dtype=torch.bfloat16
            )
            up = hipb_mm(
                x, up_w, solution_index=solution_index, out_dtype=torch.bfloat16
            )
            activated = (
                torch.nn.functional.silu(gate.to(torch.float32))
                * up.to(torch.float32)
            ).to(torch.bfloat16)
            group_out = hipb_mm(
                activated,
                down_w,
                solution_index=solution_index,
                out_dtype=torch.bfloat16,
            )
            output[start:end].copy_(group_out)
        start = end
    return output


class MoonEPGpuPlanner:
    """FlyDSL planner producing the same plan as ``build_reference_plan``.

    ``build_reference_plan`` runs on the host and walks ``num_tokens * top_k``
    routed entries in Python; on the 8k production shape that costs ~800 ms,
    about 34x the whole MoonEP forward.  This planner runs the identical
    algorithm as three FlyDSL launches and keeps every tie-break bit-identical,
    so the two are interchangeable field by field.

    Buffers are allocated once and reused.  The returned plan therefore aliases
    planner-owned storage and is invalidated by the next ``build`` call; use
    ``MoonEPReferencePlan.clone()`` to keep one across calls, exactly like the
    upstream ``MoonEPCommPlan`` reuse contract.
    """

    def __init__(
        self,
        config: MoonEPPlanConfig,
        device: torch.device | str,
        *,
        num_vblocks: int = 128,
        hist_waves_per_block: int = 4,
        dst_block_threads: int = 256,
        dst_blocks: int = 256,
    ) -> None:
        from aiter.ops.flydsl.kernels.moonep_planning import (
            MoonEPPlanGeometry,
            make_moonep_plan_jit,
        )

        self.config = config
        # Normalise "cuda" to "cuda:N" so the input check can compare devices
        # against tensors, which always carry an explicit index.
        self.device = torch.empty(0, device=device).device
        self.geo = MoonEPPlanGeometry(
            rank=config.rank,
            world_size=config.world_size,
            num_tokens=config.num_tokens,
            top_k=config.top_k,
            num_experts=config.num_experts,
            prefetch_slots=int(config.prefetch_slots),
            token_padding=config.token_padding,
            num_dispatch_rows=config.num_dispatch_rows,
            no_migration=config.no_migration,
            num_vblocks=num_vblocks,
            hist_waves_per_block=hist_waves_per_block,
            dst_block_threads=dst_block_threads,
            dst_blocks=dst_blocks,
        )
        self._jits = make_moonep_plan_jit(self.geo)
        self._compiled: list = [None, None, None]

        R = config.world_size
        E = config.num_experts
        B = int(config.prefetch_slots)
        G = E + B

        def _i32(*shape: int) -> torch.Tensor:
            return torch.zeros(shape, dtype=torch.int32, device=self.device)

        sizes = self.geo.scratch_sizes()
        self._order = _i32(sizes["order"])
        self._local_hist = _i32(sizes["local_hist"])
        self._tpe_prefix = _i32(sizes["tpe_prefix"])
        self._alloc_cumsum = _i32(sizes["alloc_cumsum"])
        self._expert_off = _i32(sizes["expert_off"])

        self.dst = _i32(config.num_tokens, config.top_k)
        self.cu_seqlens = _i32(G)
        self.experts_to_copy = _i32(R, B)
        self.zero_fill_ranges = _i32(G, 2)
        self.remote_stats = _i32(2)
        self.alloc = _i32(R, E)
        self.group_expert_ids = _i32(G)

    def _check_inputs(
        self, topk_experts: torch.Tensor, tokens_per_expert: torch.Tensor
    ) -> None:
        cfg = self.config
        if tuple(topk_experts.shape) != (cfg.num_tokens, cfg.top_k):
            raise ValueError(
                f"topk_experts must have shape {(cfg.num_tokens, cfg.top_k)}, "
                f"got {tuple(topk_experts.shape)}"
            )
        if tuple(tokens_per_expert.shape) != (cfg.world_size, cfg.num_experts):
            raise ValueError(
                f"tokens_per_expert must have shape "
                f"{(cfg.world_size, cfg.num_experts)}, "
                f"got {tuple(tokens_per_expert.shape)}"
            )
        for name, t in (
            ("topk_experts", topk_experts),
            ("tokens_per_expert", tokens_per_expert),
        ):
            if t.dtype != torch.int32:
                raise TypeError(f"{name} must be int32 for the GPU planner")
            if not t.is_contiguous():
                raise ValueError(f"{name} must be contiguous")
            if t.device != self.device:
                raise ValueError(f"{name} must live on {self.device}")

    def _run(self, slot: int, args: tuple) -> None:
        """First call compiles *and* runs; later calls take the fast path."""

        if self._compiled[slot] is None:
            self._compiled[slot] = flyc.compile(self._jits[slot], *args)
        else:
            raw = tuple(
                a.value if hasattr(a, "value") else a for a in args[:-1]
            ) + (args[-1],)
            self._compiled[slot](*raw)

    STAGES = ("order_hist", "meta", "dst")

    @property
    def stages(self) -> tuple:
        return self.STAGES

    def stage_args(
        self,
        slot: int,
        topk_experts: torch.Tensor,
        tokens_per_expert: torch.Tensor,
        stream: torch.cuda.Stream,
    ) -> tuple:
        """Launch arguments for one stage; also used by the stage benchmark."""

        p = fx.Int64
        if slot == 0:
            return (
                p(topk_experts.data_ptr()),
                p(self._order.data_ptr()),
                p(self._local_hist.data_ptr()),
                stream,
            )
        if slot == 1:
            return (
                p(tokens_per_expert.data_ptr()),
                p(self._local_hist.data_ptr()),
                p(self._tpe_prefix.data_ptr()),
                p(self._alloc_cumsum.data_ptr()),
                p(self._expert_off.data_ptr()),
                p(self.alloc.data_ptr()),
                p(self.experts_to_copy.data_ptr()),
                p(self.cu_seqlens.data_ptr()),
                p(self.zero_fill_ranges.data_ptr()),
                p(self.group_expert_ids.data_ptr()),
                p(self.remote_stats.data_ptr()),
                stream,
            )
        return (
            p(topk_experts.data_ptr()),
            p(self._order.data_ptr()),
            p(self._local_hist.data_ptr()),
            p(self._tpe_prefix.data_ptr()),
            p(self._alloc_cumsum.data_ptr()),
            p(self._expert_off.data_ptr()),
            p(self.dst.data_ptr()),
            self.config.num_tokens,
            stream,
        )

    def run_stage(
        self,
        slot: int,
        topk_experts: torch.Tensor,
        tokens_per_expert: torch.Tensor,
        stream: torch.cuda.Stream | None = None,
    ) -> None:
        """Run one planning stage.  Stages are only valid in ascending order."""

        if stream is None:
            stream = torch.cuda.current_stream(self.device)
        self._run(slot, self.stage_args(slot, topk_experts, tokens_per_expert, stream))

    def build(
        self, topk_experts: torch.Tensor, tokens_per_expert: torch.Tensor
    ) -> MoonEPReferencePlan:
        self._check_inputs(topk_experts, tokens_per_expert)
        stream = torch.cuda.current_stream(self.device)
        for slot in range(3):
            self._run(
                slot, self.stage_args(slot, topk_experts, tokens_per_expert, stream)
            )

        return MoonEPReferencePlan(
            config=self.config,
            dst=self.dst,
            cu_seqlens=self.cu_seqlens,
            experts_to_copy=self.experts_to_copy,
            zero_fill_ranges=self.zero_fill_ranges,
            remote_stats=self.remote_stats,
            alloc=self.alloc,
            group_expert_ids=self.group_expert_ids,
        )


class MoonEPFusedPlanner:
    """Single-launch FlyDSL planner; same plan, latency-tuned.

    Produces byte-identical output to :class:`MoonEPGpuPlanner` and to
    ``build_reference_plan``.  The difference is the launch structure: block 0
    runs the single-workgroup planning metadata while every other block runs the
    histogram, and two software grid barriers separate the remaining phases.

    Because those barriers spin, **every block must be co-resident**: ``blocks``
    is capped at the device CU count, and the kernel must not run concurrently
    with other work on the same device.  :class:`MoonEPGpuPlanner` has no such
    requirement and stays the safe default.

    Buffers are allocated once and reused; ``clone()`` the returned plan to keep
    it past the next ``build`` call.
    """

    STAGES = ("fused",)

    def __init__(
        self,
        config: MoonEPPlanConfig,
        device: torch.device | str,
        *,
        num_vblocks: int = 128,
        blocks: int = 64,
        waves_per_block: int | None = None,
    ) -> None:
        from aiter.ops.flydsl.kernels.moonep_planning_fused import (
            MoonEPFusedGeometry,
            make_moonep_fused_plan_jit,
        )

        self.config = config
        self.device = torch.empty(0, device=device).device

        num_cu = torch.cuda.get_device_properties(
            self.device.index or 0
        ).multi_processor_count
        if blocks > num_cu:
            raise ValueError(
                f"fused planner: blocks={blocks} exceeds the device CU count "
                f"({num_cu}).  Its grid-wide barriers require every block to be "
                f"co-resident; keep blocks <= {num_cu}."
            )

        self.geo = MoonEPFusedGeometry(
            rank=config.rank,
            world_size=config.world_size,
            num_tokens=config.num_tokens,
            top_k=config.top_k,
            num_experts=config.num_experts,
            prefetch_slots=int(config.prefetch_slots),
            token_padding=config.token_padding,
            num_dispatch_rows=config.num_dispatch_rows,
            num_vblocks=num_vblocks,
            blocks=blocks,
            waves_per_block=waves_per_block,
        )
        self._jit = make_moonep_fused_plan_jit(self.geo)
        self._compiled = None

        R = config.world_size
        E = config.num_experts
        B = int(config.prefetch_slots)
        G = E + B

        def _i32(*shape: int) -> torch.Tensor:
            return torch.zeros(shape, dtype=torch.int32, device=self.device)

        sizes = self.geo.scratch_sizes()
        self._order = _i32(sizes["order"])
        self._local_hist = _i32(sizes["local_hist"])
        self._tpe_prefix = _i32(sizes["tpe_prefix"])
        self._alloc_cumsum = _i32(sizes["alloc_cumsum"])
        self._expert_off = _i32(sizes["expert_off"])
        # Barrier tickets are never reset: each wait is ">= the end of my
        # epoch", so repeated calls just keep counting up.
        self._barrier = torch.zeros(
            sizes["barrier"], dtype=torch.int64, device=self.device
        )

        self.dst = _i32(config.num_tokens, config.top_k)
        self.cu_seqlens = _i32(G)
        self.experts_to_copy = _i32(R, B)
        self.zero_fill_ranges = _i32(G, 2)
        self.remote_stats = _i32(2)
        self.alloc = _i32(R, E)
        self.group_expert_ids = _i32(G)

    _check_inputs = MoonEPGpuPlanner._check_inputs

    def stage_args(
        self,
        slot: int,
        topk_experts: torch.Tensor,
        tokens_per_expert: torch.Tensor,
        stream: torch.cuda.Stream,
    ) -> tuple:
        p = fx.Int64
        return (
            p(topk_experts.data_ptr()),
            p(tokens_per_expert.data_ptr()),
            p(self._order.data_ptr()),
            p(self._local_hist.data_ptr()),
            p(self._tpe_prefix.data_ptr()),
            p(self._alloc_cumsum.data_ptr()),
            p(self._expert_off.data_ptr()),
            p(self.alloc.data_ptr()),
            p(self.experts_to_copy.data_ptr()),
            p(self.cu_seqlens.data_ptr()),
            p(self.zero_fill_ranges.data_ptr()),
            p(self.group_expert_ids.data_ptr()),
            p(self.remote_stats.data_ptr()),
            p(self.dst.data_ptr()),
            p(self._barrier.data_ptr()),
            stream,
        )

    def run_stage(
        self,
        slot: int,
        topk_experts: torch.Tensor,
        tokens_per_expert: torch.Tensor,
        stream: torch.cuda.Stream | None = None,
    ) -> None:
        if stream is None:
            stream = torch.cuda.current_stream(self.device)
        args = self.stage_args(0, topk_experts, tokens_per_expert, stream)
        if self._compiled is None:
            self._compiled = flyc.compile(self._jit, *args)
        else:
            raw = tuple(
                a.value if hasattr(a, "value") else a for a in args[:-1]
            ) + (args[-1],)
            self._compiled(*raw)

    def build(
        self, topk_experts: torch.Tensor, tokens_per_expert: torch.Tensor
    ) -> MoonEPReferencePlan:
        self._check_inputs(topk_experts, tokens_per_expert)
        self.run_stage(0, topk_experts, tokens_per_expert)
        return MoonEPReferencePlan(
            config=self.config,
            dst=self.dst,
            cu_seqlens=self.cu_seqlens,
            experts_to_copy=self.experts_to_copy,
            zero_fill_ranges=self.zero_fill_ranges,
            remote_stats=self.remote_stats,
            alloc=self.alloc,
            group_expert_ids=self.group_expert_ids,
        )


_GPU_PLANNER_CACHE: dict = {}


def build_plan_gpu(
    config: MoonEPPlanConfig,
    topk_experts: torch.Tensor,
    tokens_per_expert: torch.Tensor,
    **planner_kwargs,
) -> MoonEPReferencePlan:
    """Convenience wrapper around a cached :class:`MoonEPGpuPlanner`.

    The returned plan aliases planner-owned buffers; ``clone()`` it to keep one
    beyond the next call for the same ``(config, device)``.
    """

    device = topk_experts.device
    key = (config, str(device), tuple(sorted(planner_kwargs.items())))
    planner = _GPU_PLANNER_CACHE.get(key)
    if planner is None:
        planner = MoonEPGpuPlanner(config, device, **planner_kwargs)
        _GPU_PLANNER_CACHE[key] = planner
    return planner.build(topk_experts, tokens_per_expert)


__all__ = [
    "MoonEPFusedPlanner",
    "MoonEPGpuPlanner",
    "MoonEPPlanConfig",
    "MoonEPReferencePlan",
    "build_plan_gpu",
    "build_reference_plan",
    "hipblaslt_grouped_gemm_reference",
    "hipblaslt_moonep_grouped_gemm_reference",
    "hipblaslt_moonep_mlp_reference",
]
