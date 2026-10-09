# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Inputs of the EP16 MegaMoE tile test (test_megamoe_tile_internode.py).

Process setup with the GPU-occupancy preflight, the route fixtures, the packed
A4W4 weights and the duplicate-expert coalescing of the eager reference.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.distributed as dist


@dataclass(frozen=True)
class Shape:
    tokens: int
    hidden: int
    inter: int
    experts: int
    topk: int
    ep_size: int = 16
    gpus_per_node: int = 8
    activation: str = "silu"

    def validate(self) -> None:
        if self.ep_size != 2 * self.gpus_per_node:
            raise ValueError("the EP16 test runs on two nodes")
        if self.experts <= 0 or self.experts % self.ep_size:
            raise ValueError("experts must be a positive multiple of the EP size")
        if not 1 <= self.topk <= 16:
            raise ValueError("topk must be in [1, 16] for the rank-slot ABI")
        if self.activation not in ("silu", "situv2"):
            raise ValueError("activation must be silu or situv2")

    @property
    def local_experts(self) -> int:
        return self.experts // self.ep_size


@dataclass
class Weights:
    """Packed FP4 expert weights and E8M0 scales in the operator's layouts."""

    w1: torch.Tensor
    w1_scale: torch.Tensor
    w2: torch.Tensor
    w2_scale: torch.Tensor


@dataclass
class SharedInputs:
    x: torch.Tensor
    a_quant: torch.Tensor
    a_scale: torch.Tensor
    route_weights: torch.Tensor
    topk_ids: torch.Tensor
    prepared_weights: Weights
    local_expert_mask: torch.Tensor


# --------------------------------------------------------------------------
# Process setup
# --------------------------------------------------------------------------


def _idle_failures(cards, max_idle_vram):
    """Zero utilization alone does not exclude an idle inference server."""
    devices = {k: v for k, v in cards.items() if k.startswith("card")}
    failures = (
        [] if len(devices) == 8 else [f"expected 8 GPUs, observed {len(devices)}"]
    )
    for name, value in devices.items():
        try:
            used = int(value["VRAM Total Used Memory (B)"])
            busy = int(value["GPU use (%)"])
        except (KeyError, TypeError, ValueError):
            failures.append(f"{name}: incomplete utilization/VRAM evidence")
            continue
        if used > max_idle_vram or busy != 0:
            failures.append(f"{name}: VRAM={used} bytes, GPU use={busy}%")
    return failures


def _guard_idle_gpus(output_dir):
    """Both nodes' GPUs must be idle before any rank allocates; Gloo only.

    Local rank 0 of each node samples rocm-smi up to three times (startup
    activity gets two seconds to settle); MEGAMOE_PREFLIGHT_MAX_IDLE_VRAM sets
    the VRAM an idle card may hold (default 2 GiB).
    """
    max_idle_vram = int(os.environ.get("MEGAMOE_PREFLIGHT_MAX_IDLE_VRAM", 2 << 30))
    local = None
    if int(os.environ["LOCAL_RANK"]) == 0:
        observations = []
        for attempt in range(3):
            observed = {"cards": {}, "failures": []}
            try:
                result = subprocess.run(
                    ["rocm-smi", "--showuse", "--showmeminfo", "vram", "--json"],
                    capture_output=True,
                    text=True,
                    check=True,
                    timeout=20,
                )
                observed["cards"] = json.loads(result.stdout)
                observed["failures"] = _idle_failures(observed["cards"], max_idle_vram)
            except (OSError, ValueError, subprocess.SubprocessError) as error:
                observed["failures"] = [str(error)]
            observations.append(observed)
            if not observed["failures"]:
                break
            if attempt < 2:
                time.sleep(1)
        local = {
            "rank": dist.get_rank(),
            **observations[-1],
            "observations": observations,
        }
    gathered = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, local)
    nodes = [row for row in gathered if row is not None]
    failed = len(nodes) != 2 or any(row["failures"] for row in nodes)
    if local is not None:
        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=True)
        (path / "occupancy_preflight.json").write_text(
            json.dumps({"passed": not failed, "nodes": nodes}, indent=2) + "\n"
        )
    if failed:
        raise RuntimeError(
            "EP16 test window is occupied or unverified; no GPU job "
            f"started. See occupancy_preflight.json: "
            f"{[(r['rank'], r['failures'][:2]) for r in nodes]}"
        )


def setup_dist(preflight_dir):
    """Gloo for coordination (CPU tensors only) and the MORI shmem bootstrap.

    MORI owns the GPU/IB transport, so no NCCL/RCCL fabric is created: it may
    pick different RoCE rails/GID indices.
    """
    import mori.shmem as ms
    import torch._C._distributed_c10d as c10d

    rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    dist.init_process_group("gloo")
    try:
        _guard_idle_gpus(preflight_dir)
    except Exception:
        dist.destroy_process_group()
        raise
    torch.cuda.set_device(local_rank)
    c10d._register_process_group("default", dist.group.WORLD)
    ms.shmem_torch_process_group_init("default")
    return rank, world, torch.device("cuda", local_rank)


# --------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------

# permuted-arbitrary-topk (topk 16): per token eight local-node and eight
# remote-node routes over six ranks, non-adjacent slots per rank, and every
# selected rank sees a repeated exact expert ID.
_ARBITRARY_REMOTE = (0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1)
_ARBITRARY_RANK_DELTA = (0, 3, 2, 3, 0, 5, 2, 5, 0, 3, 6, 7, 0, 5, 6, 7)
_ARBITRARY_EXPERT_VARIANT = (0, 1, 2, 1, 3, 4, 2, 4, 0, 5, 6, 7, 3, 4, 6, 7)


def _permuted_arbitrary_topk(shape, rank):
    token = torch.arange(shape.tokens, dtype=torch.int64).view(-1, 1)
    slot = torch.arange(shape.topk, dtype=torch.int64).view(1, -1)
    node, local_rank = divmod(rank, shape.gpus_per_node)
    remote = torch.tensor(_ARBITRARY_REMOTE, dtype=torch.int64).view(1, -1)
    delta = torch.tensor(_ARBITRARY_RANK_DELTA, dtype=torch.int64).view(1, -1)
    owner_node = torch.where(
        remote != 0, torch.full_like(remote, 1 - node), torch.full_like(remote, node)
    )
    owner = (
        owner_node * shape.gpus_per_node + (local_rank + delta) % shape.gpus_per_node
    ).expand(shape.tokens, -1)
    variant = torch.tensor(_ARBITRARY_EXPERT_VARIANT, dtype=torch.int64).view(1, -1)
    base = (token * 7 + rank * 3) % shape.local_experts
    topk_ids = owner * shape.local_experts + (base + variant) % shape.local_experts
    # Distinct weights per slot, depending on source rank and token.
    numerators = ((token * 17 + slot * 29 + rank * 11) % 251 + 1).to(torch.float32)
    return topk_ids.to(torch.int32), numerators / numerators.sum(dim=1, keepdim=True)


def _balanced_routes(shape, rank, generator, device, *, per_node_group):
    """One route per rank: topk/2 ranks on each node, owners rotated by token.

    per_node_group=False is TestWideEpMoe's cross_node (topk == ep_size);
    True rotates the expert per group of gpus_per_node tokens so every local
    expert gets equal load for topk < ep_size too (eplb-balanced).
    """
    token = rank * shape.tokens + torch.arange(shape.tokens, device=device)
    slot = torch.arange(shape.topk, device=device)
    owner = (slot % 2)[None, :] * shape.gpus_per_node + (
        token[:, None] + slot // 2
    ) % shape.gpus_per_node
    group = token[:, None] // shape.gpus_per_node if per_node_group else token[:, None]
    expert = (group * (shape.topk // 2) + slot // 2) % shape.local_experts
    topk_ids = (owner * shape.local_experts + expert).to(torch.int32)
    route_weights = torch.rand(
        (shape.tokens, shape.topk), generator=generator, device=device
    ).softmax(dim=-1)
    return topk_ids, route_weights


ROUTE_PATTERNS = (
    "rank-balanced-hot",
    "cross_node",
    "eplb-balanced",
    "permuted-arbitrary-topk",
)


def shared_inputs(shape, rank, device, *, route_pattern, seed, prepared_weights):
    """Activations and routes of one case; the weights are passed in."""
    # rank-balanced-hot is bench_mega_moe_v2.py's own generator.
    from op_tests.multigpu_tests.bench_mega_moe_v2 import make_inputs
    from aiter.ops.quant import per_1x32_f4_quant

    if route_pattern not in ROUTE_PATTERNS:
        raise ValueError(f"unsupported route pattern {route_pattern!r}")
    x, route_weights, topk_ids = make_inputs(
        shape.tokens,
        rank,
        shape.ep_size,
        shape.hidden,
        shape.experts,
        shape.topk,
        "rank-balanced-hot",
        0.6,
        device,
    )
    if seed != 1234:
        generator = torch.Generator(device=device).manual_seed(seed + rank)
        x.normal_(generator=generator)
    if route_pattern in ("cross_node", "eplb-balanced"):
        if route_pattern == "cross_node" and shape.topk != shape.ep_size:
            raise ValueError("cross_node needs topk == ep_size; use eplb-balanced")
        if shape.topk % 2 or shape.topk // 2 > shape.gpus_per_node:
            raise ValueError("eplb routing needs even topk <= 2 * gpus_per_node")
        generator = torch.Generator(device=device).manual_seed(seed + rank)
        x.normal_(generator=generator)
        topk_ids, route_weights = _balanced_routes(
            shape,
            rank,
            generator,
            device,
            per_node_group=route_pattern == "eplb-balanced",
        )
    elif route_pattern == "permuted-arbitrary-topk":
        if shape.topk != 16:
            raise ValueError("the permuted fixture is defined for topk=16 only")
        ids, weights = _permuted_arbitrary_topk(shape, rank)
        topk_ids, route_weights = ids.to(device), weights.to(device)
    a_quant, a_scale = per_1x32_f4_quant(x, shuffle=False)
    local_mask = torch.zeros(shape.experts, dtype=torch.int32, device=device)
    local_mask[rank * shape.local_experts : (rank + 1) * shape.local_experts] = 1
    return SharedInputs(
        x, a_quant, a_scale, route_weights, topk_ids, prepared_weights, local_mask
    )


# --------------------------------------------------------------------------
# Weights and the reference
# --------------------------------------------------------------------------


def prepare_weights(shape, rank, device, *, seed):
    """Random FP4 w1/w2 with E8M0 scales, in the operator's packed layouts."""
    from aiter.ops.quant import per_1x32_f4_quant
    from aiter.ops.shuffle import (
        shuffle_scale_a16w4,
        shuffle_weight,
        shuffle_weight_a16w4,
    )
    from aiter.utility.fp4_utils import e8m0_shuffle

    generator = torch.Generator(device=device).manual_seed(90_000 + rank + seed - 1234)
    # One BF16 source at a time: holding both while FP4 quantization builds its
    # FP32 workspace costs several GiB of peak memory.
    w1 = torch.randn(
        (shape.local_experts, 2 * shape.inter, shape.hidden),
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    )
    w1.mul_(shape.hidden**-0.25)
    w1q, w1s = per_1x32_f4_quant(w1, shuffle=False)
    del w1
    w1q, w1s = shuffle_weight(w1q, layout=(16, 16)), e8m0_shuffle(w1s)
    torch.cuda.empty_cache()
    w2 = torch.randn(
        (shape.local_experts, shape.hidden, shape.inter),
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    )
    w2.mul_(shape.inter**-0.25)
    w2q, w2s = per_1x32_f4_quant(w2, shuffle=False)
    del w2
    w2q = shuffle_weight_a16w4(w2q, 16, False)
    w2s = shuffle_scale_a16w4(w2s, shape.local_experts, False)
    torch.cuda.empty_cache()
    return Weights(w1q, w1s, w2q, w2s)


def coalesce_reference_routes(ids, weights, *, experts_per_rank, num_experts):
    """Make expert IDs unique per token while keeping each expert's contribution.

    Valid because the weights apply after the expert MLP:
    sum_s w_s * F_e(x) == (sum_s w_s) * F_e(x).  The first slot of an expert
    gets the summed weight; repeats become unused experts of the SAME rank with
    zero weight, so every slot's owner (MORI's rank/node dedup) is unchanged.
    For the eager reference only, never a candidate input.
    """
    if experts_per_rank < 1 or num_experts % experts_per_rank:
        raise ValueError("invalid expert partition")
    result_ids, result_weights = [], []
    for row_ids, row_weights in zip(ids, weights, strict=True):
        out_ids, out_weights = list(row_ids), list(row_weights)
        used = set(row_ids)
        slots_by_expert = {}
        for slot, expert in enumerate(row_ids):
            slots_by_expert.setdefault(expert, []).append(slot)
        for expert, slots in slots_by_expert.items():
            out_weights[slots[0]] = math.fsum(row_weights[s] for s in slots)
            for slot in slots[1:]:
                begin = expert // experts_per_rank * experts_per_rank
                spare = next(
                    (
                        e
                        for e in range(begin, begin + experts_per_rank)
                        if e not in used
                    ),
                    None,
                )
                if spare is None:
                    raise ValueError("no unused same-rank expert for zero-weight slot")
                used.add(spare)
                out_ids[slot], out_weights[slot] = spare, 0.0
        result_ids.append(out_ids)
        result_weights.append(out_weights)
    return result_ids, result_weights
