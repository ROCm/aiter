# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""EP16 inter-node MegaMoE test: accuracy and performance in one pass.

Run the same command on both nodes (2 nodes x 8 GPUs), only --node-rank differs::

    python3 op_tests/multigpu_tests/test_megamoe_internode.py \\
        --node-rank 0 --master-addr <node0 ip> --master-port 29500 --tag k3_dsv4 \\
        --network kimi_k3,dsv4 --tpr-list 16384,4096,512,128,32 --fixtures random,eplb

Run with plain ``python3`` it is the launcher: it sets the transport
environment, then starts one torchrun job per network in --network, in order,
each on --master-port + its index.  Output is printed and also written to
<out-dir>/<tag>/node<r>.log; the exit code goes to node<r>.status.  With several
networks each one keeps its results in <out-dir>/<tag>/<network>/results.json
(one network: <out-dir>/<tag>/results.json).

Inside torchrun (RANK set) it is one network's worker.  Every (TPR, fixture)
case runs in that process, largest TPR first:

  check      fused eager output vs the MORI A4W4 reference built on the same
             weights (relL2), then CUDA-graph replays vs eager, with changed
             inputs and swapped routing in between (--no-check skips all)
  total      GPU events around the full graph replay, including quantization
             and communication; per-rank min/median, mean over all 16 ranks.
             Mega and SmallOp use the same warmup and sampling procedure
  breakdown  fused only: torch.profiler over --prof-replays more replays;
             per-kernel rows from the last --prof-tail of them.  The small-op
             graph is not profiled: under torch.profiler its replay now and then
             deadlocks inside HIP (a roctracer completion callback against the
             launching thread), which hangs all 16 ranks

--timing picks total and/or breakdown; --paths picks the fused operator and/or
the small-op baseline (quant + MORI dispatch + tuned fused_moe + MORI combine).

  --network kimi_k3 --tpr-list 128,256,512,1024 --fixtures eplb,random
  --network dsv4 --tpr-list 512 --paths smallop --timing total
  --network kimi_k3 --tpr-list 128,512 --fixtures permuted --timing none
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import math
import os
import re
import signal
import statistics
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

REPO_ROOT = Path(__file__).resolve().parents[2]

NETWORKS = {
    "kimi_k3": {
        "model_dim": 3584,
        "inter_dim": 3072,
        "experts": 896,
        "topk": 16,
        "activation": "situv2",
        "swiglu_limit": 0.0,
        "smallop_quant": "a4w4",
        "fmoe_csv": "aiter/configs/model_configs/kimik3_a4w4_tuned_fmoe.csv",
    },
    "dsv4": {
        "model_dim": 7168,
        "inter_dim": 3072,
        "experts": 384,
        "topk": 6,
        "activation": "silu",
        "swiglu_limit": 10.0,
        "smallop_quant": "a8w4",
        "fmoe_csv": "aiter/configs/model_configs/dsv4_fp8fp4_tuned_fmoe.csv",
    },
    # DeepSeek-V4.1-Flash (config.json: hidden 5120, moe_intermediate 2304, 384 routed
    # experts, top-6, SwiGLU limit 10).  The fused_moe tables carry no EP16 row
    # (24 experts/rank): run the small-op baseline with --allow-tune-miss.
    "dsv41_flash": {
        "model_dim": 5120,
        "inter_dim": 2304,
        "experts": 384,
        "topk": 6,
        "activation": "silu",
        "swiglu_limit": 10.0,
        "smallop_quant": "a4w4",
        "fmoe_csv": "aiter/configs/model_configs/dsv41_fp4_tuned_fmoe.csv",
    },
}
EP_SIZE, GPUS_PER_NODE = 16, 8

# Per-shape knobs come from the paired tune table (mega_moe_internode_tuned.csv,
# else the built-in rule).  A sweep passes --set <key>=<its own table>; the
# guard resolves the same table, so it stays on for those.
TILE_ENV_KEYS = ("AITER_CONFIG_MEGAMOE_INTERNODE",)

SMALLOP_ENV = {
    "a4w4": {"AITER_SITUV2_A4W4": "1", "AITER_SITUV2_A8W4": "0"},
    # Otherwise Silu+interleave picks a bf16 activation for small M.
    "a8w4": {"AITER_BF16_FP8_MOE_BOUND": "0", "ATOM_MOE_GU_ITLV": "1"},
}

# MORI launch geometry of the two small-op paths (dispatch blocks, RDMA blocks).
MORI_BLOCKS = {"a4w4": (96, 64), "a8w4": (256, 128)}


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--network",
        default="kimi_k3",
        help=f"comma separated, run in order; one of {sorted(NETWORKS)}",
    )
    p.add_argument("--model-dim", type=int)
    p.add_argument("--inter-dim", type=int)
    p.add_argument("--experts", type=int)
    p.add_argument("--topk", type=int)
    p.add_argument("--activation", "--act", choices=("silu", "situv2"))
    p.add_argument("--situ-beta", type=float, default=1.0)
    p.add_argument("--situ-linear-beta", type=float, default=1.0)
    p.add_argument("--tpr-list", default="512", help="tokens per rank, comma separated")
    p.add_argument(
        "--fixtures",
        default="eplb,random",
        help="route fixtures, comma separated: eplb, random, permuted",
    )
    p.add_argument(
        "--route-file",
        default="",
        help="file fixture: DIR:KEY:CHUNK (routes dumped from a serving run)",
    )
    p.add_argument(
        "--paths", default="fused,smallop", help="fused and/or smallop, comma separated"
    )
    p.add_argument(
        "--timing",
        default="total,breakdown",
        help="total and/or breakdown, comma separated; none = accuracy only",
    )
    p.add_argument("--check", dest="check", action="store_true", default=True)
    p.add_argument(
        "--no-check",
        dest="check",
        action="store_false",
        help="skip the eager reference and the replay checks",
    )
    p.add_argument("--rel-l2-threshold", type=float, default=5e-2)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--iters", type=int, default=40)
    p.add_argument(
        "--prof-replays",
        type=int,
        default=20,
        help="replays recorded by torch.profiler for the breakdown",
    )
    p.add_argument(
        "--prof-tail",
        type=int,
        default=5,
        help="last profiled replays used for the breakdown statistics",
    )
    p.add_argument(
        "--comm-quant",
        choices=("fp8", "none"),
        default="fp8",
        help="node-partial format: fp8 quantizes both local and remote partials; none keeps BF16",
    )
    p.add_argument("--smallop-quant", choices=("a4w4", "a8w4"))
    p.add_argument("--fused-quant", choices=("a4w4", "a8w4"), default="a4w4")
    p.add_argument("--fmoe-csv", help="fused_moe tune table for the small-op baseline")
    p.add_argument(
        "--allow-tune-miss",
        action="store_true",
        help="do not fail when the small-op fused_moe misses its tune table",
    )
    p.add_argument(
        "--set",
        dest="env_set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="env override (repeatable); anything but a tile tune table "
        "disables the kernel-name guard",
    )
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--out-dir", default="trace_data/internode")
    p.add_argument("--tag", default="run")
    p.add_argument(
        "--max-idle-vram-gib",
        type=float,
        default=2.0,
        help="preflight: VRAM an idle GPU may already hold",
    )
    g = p.add_argument_group("launcher (plain python3 entry)")
    g.add_argument("--node-rank", type=int, choices=(0, 1))
    g.add_argument("--master-addr", help="node0 address")
    g.add_argument(
        "--master-port",
        type=int,
        default=29500,
        help="network i of --network uses this + i",
    )
    g.add_argument("--timeout", type=int, default=3600, help="seconds per network")
    g.add_argument(
        "--socket-ifname",
        default="enp193s0f1np1",
        help="Gloo / MORI bootstrap interface",
    )
    g.add_argument(
        "--torchrun-wrap",
        default="",
        help="torchrun --no-python wrapper (e.g. rocprofv3 --att on one rank); "
        "the test script and its args follow it",
    )
    args = p.parse_args(argv)
    args.networks = list(dict.fromkeys(v for v in args.network.split(",") if v))
    if not args.networks or set(args.networks) - set(NETWORKS):
        p.error(
            f"--network takes a comma separated subset of {sorted(NETWORKS)}, "
            f"got {args.network!r}"
        )
    if "RANK" in os.environ and len(args.networks) != 1:
        p.error("a torchrun worker runs one network")
    args.network = args.networks[0]
    args.paths = split_choices(p, "--paths", args.paths, ("fused", "smallop"))
    args.timing = split_choices(
        p, "--timing", args.timing, ("total", "breakdown", "none")
    )
    args.timing.discard("none")
    fixtures = split_choices(
        p, "--fixtures", args.fixtures, ("eplb", "random", "permuted", "file")
    )
    # The CCO symmetric memory is pooled and only grows, so the largest window
    # goes first: TPR descending, and eplb (route cap 1) last within a TPR.
    args.fixtures = sorted(fixtures, key=lambda f: (f == "eplb", f))
    args.tprs = sorted({int(v) for v in args.tpr_list.split(",") if v}, reverse=True)
    if not args.tprs:
        p.error("--tpr-list is empty")
    if not 1 <= args.prof_tail <= args.prof_replays or args.iters < 1:
        p.error("need iters >= 1 and 1 <= prof-tail <= prof-replays")
    return args


def split_choices(parser, flag, value, allowed):
    items = {v for v in value.split(",") if v}
    if not items or items - set(allowed):
        parser.error(
            f"{flag} takes a comma separated subset of {allowed}, got {value!r}"
        )
    return items


def resolve_network(args):
    net = dict(NETWORKS[args.network])
    for key in ("model_dim", "inter_dim", "experts", "topk", "activation"):
        if getattr(args, key) is not None:
            net[key] = getattr(args, key)
    if args.smallop_quant:
        net["smallop_quant"] = args.smallop_quant
        if args.network == "dsv41_flash" and args.smallop_quant == "a8w4":
            net["fmoe_csv"] = "aiter/configs/model_configs/dsv41_fp8fp4_tuned_fmoe.csv"
    if args.fmoe_csv:
        net["fmoe_csv"] = args.fmoe_csv
    return net


def apply_env(args, net):
    """Must run before any aiter import: some flags are read at import."""
    # A table path left in the caller's shell would silently replace the tuned
    # tables; sweeps pass theirs through --set.
    for key in TILE_ENV_KEYS:
        os.environ.pop(key, None)
    for item in args.env_set:
        key, sep, value = item.partition("=")
        if not sep:
            raise ValueError(f"--set expects KEY=VALUE, got {item!r}")
        os.environ[key] = value
    csv = Path(net["fmoe_csv"])
    os.environ["AITER_CONFIG_FMOE"] = str(csv if csv.is_absolute() else REPO_ROOT / csv)
    os.environ.update(SMALLOP_ENV[net["smallop_quant"]])
    os.environ.setdefault("AITER_LOG_LEVEL", "INFO")


def fixture_route(fixture, topk):
    """Map a fixture name to the input route pattern and the route cap per rank."""
    if fixture == "eplb":
        return ("cross_node" if topk == EP_SIZE else "eplb-balanced"), 1
    if fixture == "permuted":
        if topk != 16:
            raise ValueError("the permuted fixture tables are defined for topk=16 only")
        return "permuted-arbitrary-topk", 4
    if fixture == "file":
        return "file", topk
    return "rank-balanced-hot", topk


# --------------------------------------------------------------------------
# Input shapes and tensors
# --------------------------------------------------------------------------


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


def _guard_idle_gpus(output_dir, max_idle_vram):
    """Both nodes' GPUs must be idle before any rank allocates; Gloo only.

    Local rank 0 of each node samples rocm-smi up to three times (startup
    activity gets two seconds to settle); ``max_idle_vram`` (bytes) is the VRAM
    an idle card may hold.
    """
    import torch.distributed as dist

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


def setup_dist(preflight_dir, max_idle_vram=2 << 30):
    """Gloo for coordination (CPU tensors only) and the MORI shmem bootstrap.

    MORI owns the GPU/IB transport, so no NCCL/RCCL fabric is created: it may
    pick different RoCE rails/GID indices.
    """
    import mori.shmem as ms
    import torch
    import torch._C._distributed_c10d as c10d
    import torch.distributed as dist

    rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    dist.init_process_group("gloo")
    try:
        _guard_idle_gpus(preflight_dir, max_idle_vram)
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
    import torch

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
    import torch

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
    "file",
)

# "file": replay routes dumped from a serving run.  Set by the test driver:
# (directory, key such as "s3_l45", chunk index); each rank reads
# <dir>/<key>_rank<r>.pt and takes rows [chunk*tokens, (chunk+1)*tokens).
ROUTE_FILE = None


def _balanced_perm(d, key, experts, ep_size):
    """EPLB-r0-like placement: logical expert -> physical id, greedy by load."""
    import torch

    load = torch.zeros(experts, dtype=torch.long)
    for r in range(ep_size):
        ids = torch.load(f"{d}/{key}_rank{r}.pt")["ids"].long().flatten()
        load += torch.bincount(ids[ids >= 0], minlength=experts)
    epr = experts // ep_size
    rank_load = [0] * ep_size
    slots = [0] * ep_size
    phys = torch.empty(experts, dtype=torch.int32)
    for e in torch.argsort(load, descending=True).tolist():
        r = min(
            (r for r in range(ep_size) if slots[r] < epr), key=lambda r: rank_load[r]
        )
        phys[e] = r * epr + slots[r]
        slots[r] += 1
        rank_load[r] += int(load[e])
    return phys


def _file_routes(shape, rank, device):
    import torch

    d, key, chunk = ROUTE_FILE[:3]
    blob = torch.load(f"{d}/{key}_rank{rank}.pt")
    ids, wts = blob["ids"].to(torch.int32), blob["wts"].to(torch.float32)
    if len(ROUTE_FILE) > 3 and ROUTE_FILE[3]:
        phys = _balanced_perm(d, ROUTE_FILE[3], shape.experts, shape.ep_size)
        ids = torch.where(ids >= 0, phys[ids.clamp_min(0).long()], ids)
    rows = ids[chunk * shape.tokens : (chunk + 1) * shape.tokens]
    w = wts[chunk * shape.tokens : (chunk + 1) * shape.tokens]
    epr = shape.local_experts
    invalid_pad = len(ROUTE_FILE) > 4 and ROUTE_FILE[4] == "inv"
    # Pad (and -1) slots: invalid id -1 ("inv") or this rank's local experts, weight 0.
    pad = (
        torch.full((shape.topk,), -1, dtype=torch.int32)
        if invalid_pad
        else rank * epr + torch.arange(shape.topk, dtype=torch.int32) % epr
    )
    out_ids = pad.repeat(shape.tokens, 1)
    out_w = torch.zeros(shape.tokens, shape.topk)
    n = rows.shape[0]
    valid = rows >= 0
    out_ids[:n] = torch.where(valid, rows, pad)
    out_w[:n] = torch.where(valid, w, 0.0)
    return out_ids.to(device).contiguous(), out_w.to(device).contiguous()


def shared_inputs(shape, rank, device, *, route_pattern, seed, prepared_weights):
    """Activations and routes of one case; the weights are passed in."""
    import torch

    # rank-balanced-hot is bench_mega_moe_v2.py's own generator.
    from aiter.ops.quant import per_1x32_f4_quant
    from op_tests.multigpu_tests.bench_mega_moe_v2 import make_inputs

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
    balanced = route_pattern in ("cross_node", "eplb-balanced")
    if seed != 1234 or balanced:
        generator = torch.Generator(device=device).manual_seed(seed + rank)
        x.normal_(generator=generator)
    if balanced:
        if route_pattern == "cross_node" and shape.topk != shape.ep_size:
            raise ValueError("cross_node needs topk == ep_size; use eplb-balanced")
        if shape.topk % 2 or shape.topk // 2 > shape.gpus_per_node:
            raise ValueError("eplb routing needs even topk <= 2 * gpus_per_node")
        topk_ids, route_weights = _balanced_routes(
            shape,
            rank,
            generator,
            device,
            per_node_group=route_pattern == "eplb-balanced",
        )
    elif route_pattern == "file":
        topk_ids, route_weights = _file_routes(shape, rank, device)
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
    import torch

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


# --------------------------------------------------------------------------
# Tile configuration and the kernel-name guard
# --------------------------------------------------------------------------


def stage2_tile(net, tpr, s1):
    """GEMM2 (kernel1) N tile, resolved exactly as the operator does."""
    from aiter.jit.utils.chip_info import get_cu_num, get_gfx_runtime
    from aiter.ops.flydsl.kernels.mega_moe_internode.mega_moe_internode_config import (
        lookup_tune,
        select_gemm2_bm,
        select_gemm2_bn,
        select_gemm2_grid,
        select_stage2_sdma,
    )

    tuned = lookup_tune(
        gfx=get_gfx_runtime(),
        cu_num=get_cu_num(),
        token=tpr * net["topk"],
        model_dim=net["model_dim"],
        inter_dim=net["inter_dim"],
        expert=net["experts"] // EP_SIZE,
        topk=net["topk"],
    )
    bn, bn_source = select_gemm2_bn(tuned)
    cu, cu_source = select_gemm2_grid(tuned)
    source = (
        bn_source
        if bn_source == cu_source
        else f"gemm2_bn={bn_source},gemm2_cu={cu_source}"
    )
    remote_first = bool(s1["split_local"])
    sdma, comm_source = select_stage2_sdma(tpr, tuned)
    sdma = sdma and remote_first
    bm, _ = select_gemm2_bm(
        tuned, tile_group=s1["tile_group"], fused=remote_first, sdma=sdma
    )
    # 0 means kernel1 launches the operator's worker_blocks (256) CTAs.
    return {
        "gemm2_bm": bm,
        "gemm2_bn": bn,
        "gemm2_cu": cu,
        "grid": cu or 256,
        "intra_comm": ("sdma" if sdma else "xgmi") if remote_first else "-",
        "comm_source": comm_source,
        "source": source,
    }


def tile_fragments(tile):
    """(must contain, must not contain) kernel-name fragments for a tile config."""
    g = tile["tile_group"]
    # The first GMM1 ticket (8) is a kernel constant: always _cf8.
    # _slg / _nsl stands for split_local together with its companion switches.
    has = ([f"_tg{g}"] if g != 1 else []) + [
        "_cf8",
        f"_fos{tile['fanout_ctas_per_dest']}",
        "_slg" if tile["split_local"] else "_nsl",
    ]
    lacks = ([] if g != 1 else ["_tg"]) + [
        "_gbn",
        "_nsl" if tile["split_local"] else "_slg",
    ]
    return has, lacks


def check_fused_config(stage1_name, stage2_names, comm_quant, tile, s2tile):
    """Kernel-name guard.  stage2_names come from the profiler (None without it)."""
    problems = []
    has, lacks = tile_fragments(tile)
    missing = [f for f in has if f not in stage1_name]
    if missing:
        problems.append(f"fused_stage1 missing {missing}")
    extra = [f for f in lacks if f in stage1_name]
    if extra:
        problems.append(f"fused_stage1 has {extra}, tile config {tile}")
    if stage2_names is None:
        return problems
    # A fused stage2 kernel carries both names: megamoe_s2f_<k1>__<k2>.
    split = []
    for n in stage2_names:
        if n.startswith("megamoe_one_"):
            # megamoe_one_<stage1>__<k1>__<k2>: Stage1 is checked by name above.
            split += ["megamoe_" + x for x in n[len("megamoe_one_") :].split("__")[1:]]
            split.append("")  # count as fused (grid = resident CTAs)
        elif n.startswith("megamoe_s2f_"):
            a, _, b = n[len("megamoe_s2f_") :].partition("__")
            split += ["megamoe_" + a, "megamoe_" + b]
        else:
            split.append(n)
    fused_s2 = len(split) != len(stage2_names)
    stage2_names = split
    k2 = [n for n in stage2_names if n.startswith("megamoe_k2")]
    if len(k2) != 1:
        problems.append(f"expected one megamoe_k2 kernel, got {k2}")
    elif ("_qr8" in k2[0]) != (comm_quant == "fp8"):
        problems.append(f"k2 rail fp8 mismatch: {k2[0]}")
    k1 = [n for n in stage2_names if n.startswith("megamoe_k1")]
    want = (
        f"_t{s2tile['gemm2_bm']}x{s2tile['gemm2_bn']}x256_",
        f"_cu{s2tile['grid']}_",
    )
    if fused_s2:  # the fused grid is the resident CTA count, not the table's
        want = want[:1]
    if len(k1) != 1:
        problems.append(f"expected one megamoe_k1 kernel, got {k1}")
    elif any(w not in k1[0] for w in want):
        problems.append(
            f"kernel1 missing {[w for w in want if w not in k1[0]]} "
            f"(stage2 tile {s2tile}): {k1[0][:90]}"
        )
    return problems


class TuneLog(logging.Handler):
    """Collect fused_moe 2-stage config selections ('using 2stage ...')."""

    def __init__(self):
        super().__init__(logging.INFO)
        self.lines = []

    def emit(self, record):
        msg = record.getMessage()
        if "using 2stage" in msg:
            self.lines.append(msg)

    def take(self, q_dtype_a):
        # The MORI A4W4 reference also calls fused_moe (FP4 activations); only
        # lookups with the small-op's activation dtype count.
        lines = [
            m for m in self.lines if f"'{q_dtype_a}', 'torch.float4_e2m1fn_x2'" in m
        ]
        self.lines = []
        misses = [m for m in lines if "using 2stage default" in m]
        return {
            "hits": len(lines) - len(misses),
            "misses": len(misses),
            "example": (misses or lines or [""])[0][:300],
        }


# --------------------------------------------------------------------------
# Paths: the fused operator and the MORI small-op baselines
# --------------------------------------------------------------------------


# The CCO Communicator is per process: every operator of the run borrows it and
# only creates/frees its own window.  Rebuilding it per case hangs the second
# case on the GPU.  Its symmetric memory is pooled too: after ccoMemFree a new
# ccoMemAlloc runs out of VMM after about two TPR512 windows, so the pool keeps
# one block and only replaces it with a larger one.
class _BorrowedMemory:
    def __init__(self, memory):
        self._memory = memory

    @property
    def ptr(self):
        return self._memory.ptr

    @property
    def size(self):
        return self._memory.size

    def close(self):
        pass


class PooledCommunicator:
    def __init__(self, rank, world, per_rank_vmm=16 * 1024**3):
        import torch.distributed as dist
        from mori.cco import Communicator, UniqueId

        payload = [bytes(Communicator.get_unique_id()) if rank == 0 else None]
        dist.broadcast_object_list(payload, src=0)
        self._comm = Communicator.init(
            world, rank, UniqueId.from_bytes(payload[0]), per_rank_vmm=per_rank_vmm
        )
        self._memory = None

    def __getattr__(self, name):
        return getattr(self._comm, name)

    def alloc_mem(self, size):
        if self._memory is None or self._memory.size < int(size):
            if self._memory is not None:
                self._memory.close()
            self._memory = self._comm.alloc_mem(int(size))
        return _BorrowedMemory(self._memory)

    def destroy(self):
        if self._memory is not None:
            self._memory.close()
            self._memory = None
        self._comm.destroy()


class MoriSmallOp:
    """Quant + MORI InterNodeV1LL dispatch + fused_moe + MORI combine.

    a4w4: FP4 activations on the fused operator's own weights; it is also the
    accuracy reference of the fused path.  a8w4 (DSV4): FP8 activations and
    the gate/up interleaved FP4 weights of test_wide_ep_moe.py.  No fake expert
    slot is appended: fused_moe keys its tune table on topk_ids.shape[1], and
    the EP16 rows of the model tables use the real TopK.
    """

    def __init__(
        self,
        quant,
        shape,
        shared,
        rank,
        *,
        swiglu_limit,
        situ_beta=1.0,
        situ_linear_beta=1.0,
        a8w4_weights=None,
        separated_weights=False,
    ):
        import mori
        import torch

        import aiter
        from aiter.fused_moe import fused_moe
        from aiter.ops.flydsl.moe_common import GateMode

        self.torch, self.quant, self.shape, self.shared = torch, quant, shape, shared
        self.rank = rank
        self.situ_beta = float(situ_beta)
        self.situ_linear_beta = float(situ_linear_beta)
        self._native_a8_reference = quant == "a8w4" and separated_weights
        self.blocks, self.rdma_blocks = MORI_BLOCKS[quant]
        w = shared.prepared_weights
        self.weights = (
            (w.w1, w.w1_scale, w.w2, w.w2_scale)
            if quant == "a4w4" or separated_weights
            else a8w4_weights
        )
        self.activation = (
            aiter.ActivationType.Situv2
            if shape.activation == "situv2"
            else aiter.ActivationType.Silu
        )
        self.gate_mode = (
            GateMode.SEPARATED.value
            if quant == "a4w4" or separated_weights
            else GateMode.INTERLEAVE.value
        )
        if quant == "a4w4":
            from aiter.ops.quant import dynamic_per_group_scaled_quant

            self._quant_op = dynamic_per_group_scaled_quant
            data_type = shared.a_quant.dtype
        else:
            from aiter.ops.flydsl.kernels.mega_moe.quant import per_1x32_mx_quant

            self._quant_op = per_1x32_mx_quant
            data_type = torch.float8_e4m3fn
        self.swiglu_limit = float(swiglu_limit)
        self._fused_moe = fused_moe
        config = mori.ops.EpDispatchCombineConfig(
            data_type=data_type,
            rank=rank,
            world_size=EP_SIZE,
            hidden_dim=shape.hidden,
            scale_dim=shape.hidden // 32,
            scale_type_size=1,
            max_token_type_size=torch.bfloat16.itemsize,
            max_num_inp_token_per_rank=shape.tokens,
            max_total_recv_tokens=0,
            num_experts_per_rank=shape.local_experts,
            num_experts_per_token=shape.topk,
            kernel_type=mori.ops.EpDispatchCombineKernelType.InterNodeV1LL,
            warp_num_per_block=8,
            block_num=self.blocks,
            rdma_block_num=self.rdma_blocks,
            gpu_per_node=GPUS_PER_NODE,
            quant_type="none",
        )
        self.op = mori.ops.EpDispatchCombineOp(config)

    def forward(self, *, coalesce_duplicates=False):
        torch, shared = self.torch, self.shared
        route_ids, route_weights = shared.topk_ids, shared.route_weights
        if coalesce_duplicates:
            # The reference sorter keeps one slot per (token, expert), so repeated
            # expert IDs lose contributions; the eager reference sums their
            # weights instead (valid since doweight_stage1=False).  The candidate
            # keeps the original slots.
            ids, weights = coalesce_reference_routes(
                route_ids.cpu().tolist(),
                route_weights.cpu().tolist(),
                experts_per_rank=self.shape.local_experts,
                num_experts=self.shape.experts,
            )
            route_ids = torch.tensor(
                ids, dtype=route_ids.dtype, device=route_ids.device
            )
            route_weights = torch.tensor(
                weights, dtype=route_weights.dtype, device=route_weights.device
            )
        if self.quant == "a4w4":
            self._quant_op(
                shared.a_quant, shared.x, shared.a_scale, 32, shuffle_scale=False
            )
            x_q, x_scale = shared.a_quant, shared.a_scale
        else:
            x_q, x_scale = self._quant_op(shared.x, quant_mode="fp8")
        dispatched, recv_weights, recv_scales, recv_ids, recv_tokens = self.op.dispatch(
            x_q,
            route_weights,
            x_scale,
            route_ids,
            block_num=self.blocks,
            rdma_block_num=self.rdma_blocks,
            warp_per_block=8,
        )
        w1, w1_scale, w2, w2_scale = self.weights
        import aiter

        if self._native_a8_reference:
            local_out = self._reference_a8(
                dispatched, recv_scales, recv_ids, recv_weights, recv_tokens
            )
        else:
            local_out = self._fused_moe(
                dispatched,
                w1,
                w2,
                recv_weights,
                recv_ids,
                shared.local_expert_mask,
                activation=self.activation,
                quant_type=aiter.QuantType.per_1x32,
                doweight_stage1=False,
                w1_scale=w1_scale,
                w2_scale=w2_scale,
                a1_scale=recv_scales,
                num_local_tokens=recv_tokens,
                dtype=torch.bfloat16,
                swiglu_limit=self.swiglu_limit,
                gate_mode=self.gate_mode,
                **(
                    {"beta": self.situ_beta, "linear_beta": self.situ_linear_beta}
                    if self.shape.activation == "situv2"
                    else {}
                ),
            )
        result = self.op.combine(
            local_out,
            None,
            route_ids,
            block_num=self.blocks,
            rdma_block_num=self.rdma_blocks,
            warp_per_block=4,
        )
        return result[0] if isinstance(result, tuple) else result

    def _reference_a8(self, dispatched, scales, ids, weights, num_tokens):
        """Independent FP32 matmuls on the candidate's quantized weights.

        The small-op selector uses FP4 for separated SiLU weights, so it
        cannot serve as a native A8 reference for this ABI. MORI retains
        dispatch/combine ownership; PyTorch checks the expert MLP arithmetic.
        """
        from aiter.utility.fp4_utils import e8m0_to_f32, mxfp4_to_f32

        torch, shape = self.torch, self.shape
        e, h, k = shape.local_experts, shape.hidden, shape.inter
        if not hasattr(self, "_reference_dense_weights"):
            w1, s1, w2, s2 = (w.view(torch.uint8) for w in self.weights)
            q1 = (
                w1.reshape(e, 2 * k // 16, h // 64, 2, 16, 16)
                .permute(0, 1, 4, 2, 3, 5)
                .contiguous()
                .reshape(e, 2 * k, h // 2)
            )
            q2 = (
                w2.reshape(e, h // 16, k // 128, 4, 16, 16)
                .permute(0, 1, 4, 2, 3, 5)
                .contiguous()
                .reshape(e, h, k // 2)
            )
            sc1 = (
                s1.reshape(e * 2 * k // 32, h // 256, 4, 16, 2, 2)
                .permute(0, 5, 3, 1, 4, 2)
                .contiguous()
                .reshape(e, 2 * k, h // 32)
            )
            sc2 = (
                s2.reshape(e, h // 32, k // 256, 4, 16, 2, 2)
                .permute(0, 1, 6, 4, 2, 5, 3)
                .contiguous()
                .reshape(e, h, k // 32)
            )
            self._reference_dense_weights = (
                mxfp4_to_f32(q1) * e8m0_to_f32(sc1).repeat_interleave(32, dim=-1),
                mxfp4_to_f32(q2) * e8m0_to_f32(sc2).repeat_interleave(32, dim=-1),
            )
        w1, w2 = self._reference_dense_weights
        n = int(num_tokens.item()) if hasattr(num_tokens, "item") else int(num_tokens)
        x = dispatched[:n].float() * e8m0_to_f32(scales[:n]).repeat_interleave(
            32, dim=-1
        )
        out = torch.zeros(
            (dispatched.shape[0], h), dtype=torch.float32, device=dispatched.device
        )
        for expert in range(e):
            row, slot = torch.where(ids[:n] == self.rank * e + expert)
            if row.numel() == 0:
                continue
            gate, up = (x[row] @ w1[expert].T).chunk(2, dim=-1)
            if shape.activation == "situv2":
                activated = (
                    self.situ_beta
                    * torch.tanh(gate / self.situ_beta)
                    * torch.sigmoid(gate)
                    * self.situ_linear_beta
                    * torch.tanh(up / self.situ_linear_beta)
                )
            else:
                if self.swiglu_limit > 0:
                    gate, up = gate.clamp(max=self.swiglu_limit), up.clamp(
                        -self.swiglu_limit, self.swiglu_limit
                    )
                activated = gate * torch.sigmoid(gate) * up
            aq, asc = self._quant_op(activated.bfloat16(), quant_mode="fp8")
            a = aq.float() * e8m0_to_f32(asc).repeat_interleave(32, dim=-1)
            values = ((a @ w2[expert].T) * weights[row, slot, None]).bfloat16().float()
            out.index_add_(0, row, values)
        return out.bfloat16()


# --------------------------------------------------------------------------
# Graph capture, timing and the profiler breakdown
# --------------------------------------------------------------------------


def capture_graph(torch, dist, fn, device):
    """One forward per replay, captured on a side stream after three warm calls."""
    stream = torch.cuda.Stream(device=device)
    stream.wait_stream(torch.cuda.current_stream(device))
    with torch.cuda.stream(stream):
        for _ in range(3):
            fn()
    stream.synchronize()
    dist.barrier()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        output = fn()
    return graph, output


def time_replays(torch, dist, graph, device, warmup, iters):
    """Full-forward GPU event samples in microseconds, identical for both paths."""
    for _ in range(warmup):
        graph.replay()
    torch.cuda.synchronize(device)
    dist.barrier()
    samples = []
    begin = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(iters):
        begin.record()
        graph.replay()
        end.record()
        end.synchronize()
        samples.append(begin.elapsed_time(end) * 1000)
    dist.barrier()
    return samples


def kernel_class(name):
    if name.startswith("megamoe_one_"):  # Stage1 + fused stage2 in one launch
        return "one"
    if "megamoe_stage1" in name:
        return "stage1"
    if name.startswith("megamoe_k1"):  # fused kernel1: GEMM2 + push
        return "gemm2"
    if name.startswith("megamoe_k2"):  # fused kernel2: node reduce + rail + combine
        return "k2"
    if name.startswith("megamoe_s2f_"):  # kernel1 + kernel2 in one launch
        return "stage2"
    if "quant" in name:
        return "quant"
    return "other"


REPLAY_MARK = re.compile(r"megamoe_replay(\d+)$")


def profile_breakdown(torch, graph, device, replays, tail, trace_file):
    """Stage durations from the same complete replay with minimum kernel sum.

    Select within the last ``tail`` replays on each rank. Stage2 includes
    GEMM2 and combine; its kernel sum excludes gaps between separate launches.
    The quant + Stage1 + Stage2 sum is a profiled measurement; the independent
    GPU-event total is used for the SmallOp comparison. Incomplete profiler
    replays are skipped. Single-kernel mode exposes only its combined duration.
    """
    from torch.profiler import ProfilerActivity, profile, record_function

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for i in range(replays):
            with record_function(f"megamoe_replay{i}"):
                graph.replay()
        torch.cuda.synchronize(device)
    prof.export_chrome_trace(str(trace_file))
    events = json.loads(Path(trace_file).read_text())["traceEvents"]
    windows = []
    for e in events:
        m = REPLAY_MARK.match(e.get("name", ""))
        if (
            m
            and e.get("cat") == "gpu_user_annotation"
            and e.get("ph") == "X"
            and int(m.group(1)) >= replays - tail
        ):
            windows.append((float(e["ts"]), float(e["ts"]) + float(e["dur"])))
    if not windows:
        return (
            None,
            f"profiler recorded none of replays {replays - tail}..{replays - 1}",
        )
    kernels = [e for e in events if e.get("ph") == "X" and e.get("cat") == "kernel"]
    samples, names = [], set()
    for begin, end in windows:
        active = [e for e in kernels if begin <= float(e["ts"]) < end]
        if not active:
            continue
        row = {}
        for e in active:
            key = kernel_class(e["name"])
            row[key] = row.get(key, 0.0) + float(e["dur"])
            names.add(e["name"])
        if "quant" not in row:
            continue
        if "one" in row:
            row["device_total"] = row["quant"] + row["one"]
        elif "stage1" in row and ("stage2" in row or ("gemm2" in row and "k2" in row)):
            if "stage2" not in row:
                row["stage2"] = row["gemm2"] + row["k2"]
            row["device_total"] = row["quant"] + row["stage1"] + row["stage2"]
        else:
            continue
        row["kernel_sum"] = sum(float(e["dur"]) for e in active)
        ends = [float(e["ts"]) + float(e["dur"]) for e in active]
        row["span"] = max(ends) - min(float(e["ts"]) for e in active)
        samples.append(row)
    if not samples:
        return None, "no complete quant/Stage1/Stage2 replay was recorded"
    return min(samples, key=lambda row: row["kernel_sum"]), sorted(names)


def gpu_stats(rank_samples):
    """Preserve every rank's samples and summarize full-forward GPU latency."""
    mins = [min(s) for s in rank_samples]
    medians = [statistics.median(s) for s in rank_samples]
    return {
        "metric": "full-forward GPU events, including quantization and communication",
        "min": statistics.mean(mins),
        "median": statistics.mean(medians),
        "pooled_min": min(mins),
        "worst_median": max(medians),
        "ranks": [
            {"rank": rank, "samples_us": samples}
            for rank, samples in enumerate(rank_samples)
        ],
    }


# --------------------------------------------------------------------------
# Runner
# --------------------------------------------------------------------------


class Runner:
    def __init__(self, args, net):
        import torch
        import torch.distributed as dist

        global ROUTE_FILE

        self.torch, self.dist, self.args, self.net = torch, dist, args, net
        if args.route_file:
            parts = args.route_file.split(":")  # DIR:KEY:CHUNK[:BALANCE_FROM_KEY[:inv]]
            ROUTE_FILE = (
                parts[0],
                parts[1],
                int(parts[2]),
                parts[3] if len(parts) > 3 else "",
                parts[4] if len(parts) > 4 else "",
            )
        self.out_root = Path(args.out_dir) / args.tag
        self.out_root.mkdir(parents=True, exist_ok=True)
        self.rank, world, self.device = setup_dist(
            str(self.out_root / "preflight"), int(args.max_idle_vram_gib * 2**30)
        )
        if world != EP_SIZE:
            raise ValueError(f"EP16 needs {EP_SIZE} ranks, got {world}")
        from aiter import logger

        self.tune_log = TuneLog()
        logger.addHandler(self.tune_log)
        self.guard = all(
            item.partition("=")[0] in TILE_ENV_KEYS for item in args.env_set
        )
        self.comm = None  # created by the first fused case
        # Weights depend on the shape only: prepare them once for every case.
        self.weights = prepare_weights(
            self.shape(1), self.rank, self.device, seed=args.seed
        )
        self.a8w4_weights = None
        if "smallop" in args.paths and net["smallop_quant"] == "a8w4":
            from op_tests.multigpu_tests.test_wide_ep_moe import _quantize_local_weights

            self.a8w4_weights, _ = _quantize_local_weights(
                net["model_dim"],
                net["inter_dim"],
                net["experts"] // EP_SIZE,
                self.rank,
                args.seed,
                self.device,
                quant="a8w4",
            )
        self.results = []

    def shape(self, tpr):
        net = self.net
        shape = Shape(
            tokens=tpr,
            hidden=net["model_dim"],
            inter=net["inter_dim"],
            experts=net["experts"],
            topk=net["topk"],
            ep_size=EP_SIZE,
            gpus_per_node=GPUS_PER_NODE,
            activation=net["activation"],
        )
        shape.validate()
        return shape

    def log(self, msg):
        if self.rank == 0:
            print(msg, flush=True)

    def gather(self, obj):
        rows = [None] * EP_SIZE
        self.dist.all_gather_object(rows, obj)
        return rows

    def rel_l2(self, label, reference, actual, threshold):
        """Rank-max relL2; every rank gets the same value, so the same verdict."""
        torch = self.torch
        ref, out = reference.float(), actual.float()
        value = float((out - ref).norm() / ref.norm().clamp_min(1e-12))
        if not (torch.isfinite(ref).all() and torch.isfinite(out).all()):
            value = float("inf")
        worst = torch.tensor(value, dtype=torch.float64)
        self.dist.all_reduce(worst, op=self.dist.ReduceOp.MAX)
        value = float(worst.item())
        return {
            "label": label,
            "rel_l2": value,
            "threshold": threshold,
            "ok": value < threshold,
        }

    def check_routes(self, shape, shared, fixture, cap):
        torch, epr = self.torch, shape.local_experts
        ids = shared.topk_ids.long()
        valid = ids >= 0  # -1 = invalid slot (no route)
        owners = torch.div(ids.clamp_min(0), epr, rounding_mode="floor")
        per_rank = torch.zeros(
            (shape.tokens, EP_SIZE), dtype=torch.int64, device=owners.device
        )
        per_rank.scatter_add_(1, owners, valid.long())
        if int(per_rank.max()) > cap:
            raise ValueError(
                f"{fixture}: {int(per_rank.max())} routes to one rank, cap {cap}"
            )
        if fixture == "eplb":
            counts = torch.bincount(
                shared.topk_ids.flatten().long(), minlength=shape.experts
            ).cpu()
            self.dist.all_reduce(counts)
            per_rank = counts.view(EP_SIZE, epr)
            padded = (per_rank + 31) // 32 * 32
            if (
                len(set(per_rank.sum(1).tolist())) != 1
                or len(set(padded.sum(1).tolist())) != 1
                or len({tuple(sorted(r)) for r in per_rank.tolist()}) != 1
            ):
                raise AssertionError("eplb inputs are not balanced across ranks")

    # ---- one case ---------------------------------------------------------

    def run_case(self, tpr, fixture):
        args, net = self.args, self.net
        shape = self.shape(tpr)
        route, cap = fixture_route(fixture, shape.topk)
        shared = shared_inputs(
            shape,
            self.rank,
            self.device,
            route_pattern=route,
            seed=args.seed,
            prepared_weights=self.weights,
        )
        self.check_routes(shape, shared, fixture, cap)
        rec = {"tpr": tpr, "fixture": fixture}
        quant = net["smallop_quant"]
        reference = None
        if (args.check and "fused" in args.paths) or (
            quant == "a4w4" and "smallop" in args.paths
        ):
            reference = MoriSmallOp(
                args.fused_quant if "fused" in args.paths else "a4w4",
                shape,
                shared,
                self.rank,
                swiglu_limit=net["swiglu_limit"],
                separated_weights=True,
                situ_beta=args.situ_beta,
                situ_linear_beta=args.situ_linear_beta,
            )
        if "fused" in args.paths:
            rec["fused"] = self.run_fused(shape, shared, cap, reference, fixture)
        if "smallop" in args.paths:
            small = (
                reference
                if quant == "a4w4"
                and reference is not None
                and reference.quant == quant
                else MoriSmallOp(
                    quant,
                    shape,
                    shared,
                    self.rank,
                    swiglu_limit=net["swiglu_limit"],
                    a8w4_weights=self.a8w4_weights,
                    situ_beta=args.situ_beta,
                    situ_linear_beta=args.situ_linear_beta,
                )
            )
            rec["smallop"] = self.run_smallop(shape, small, fixture)
            del small
        del reference, shared
        self.tune_log.lines.clear()
        gc.collect()
        self.torch.cuda.empty_cache()
        return rec

    def run_fused(self, shape, shared, cap, reference, fixture):
        from aiter.ops.flydsl.kernels.mega_moe_internode import MegaMoEInternode

        torch, dist, args, device = self.torch, self.dist, self.args, self.device
        tokens = shape.tokens
        w = shared.prepared_weights
        if self.comm is None:
            # TPR 16384 needs a ~30 GiB window (K3/DSv4); the pool lives for the
            # whole run, so size it for the largest TPR of the run.
            vmm = 40 * 1024**3 if max(self.args.tprs) > 4096 else 16 * 1024**3
            self.comm = PooledCommunicator(self.rank, EP_SIZE, per_rank_vmm=vmm)

        op = MegaMoEInternode(
            rank=self.rank,
            world_size=EP_SIZE,
            model_dim=shape.hidden,
            inter_dim=shape.inter,
            experts=shape.experts,
            topk=shape.topk,
            quant=args.fused_quant,
            w1=w.w1,
            w1_scale=w.w1_scale,
            w2=w.w2,
            w2_scale=w.w2_scale,
            max_tok_per_rank=tokens,
            swiglu_limit=self.net["swiglu_limit"],
            act=shape.activation,
            situ_beta=args.situ_beta,
            situ_linear_beta=args.situ_linear_beta,
            comm_quant=args.comm_quant,
            communicator=self.comm,
            max_routes_per_token_per_rank=cap,
        )
        res = {
            "quant": args.fused_quant,
            "act": op.act,
            "situ_beta": op.situ_beta,
            "situ_linear_beta": op.situ_linear_beta,
            "reference": (
                "mori_dispatch_pytorch_a8_mlp_mori_combine"
                if args.fused_quant == "a8w4"
                else "mori_fused_moe"
            ),
            "stage1_kernel": op.stage1_kernel_name,
            "stage1_tile": dict(op._s1_tile),
            "stage2_tile": stage2_tile(self.net, tokens, op._s1_tile),
            "checks": [],
        }
        coalesce = fixture == "permuted"
        graph = None
        do_check = args.check
        try:
            call = lambda: op.forward(shared.x, shared.route_weights, shared.topk_ids)

            def ref_forward():
                out = reference.forward(coalesce_duplicates=coalesce)[:tokens].clone()
                torch.cuda.synchronize(device)
                dist.barrier()
                return out

            checks = res["checks"]
            thr = args.rel_l2_threshold
            if do_check:
                expected = ref_forward()
            eager = call()[:tokens].clone()
            torch.cuda.synchronize(device)
            if do_check:
                checks.append(self.rel_l2("eager_vs_mori", expected, eager, thr))
            graph, out = capture_graph(torch, dist, call, device)

            def replay_vs(label, ref, threshold):
                graph.replay()
                torch.cuda.synchronize(device)
                checks.append(self.rel_l2(label, ref, out[:tokens], threshold))

            if do_check:
                # Single replays alternate the generation parity, so a stage that
                # froze its parity at capture shows up as last generation's output.
                replay_vs("replay_vs_eager", eager, 1e-2)
                original = shared.x.clone()
                shared.x.mul_(-0.75)
                replay_vs("changed_input", ref_forward(), thr)
                shared.x.copy_(original)
                replay_vs("restored_input", eager, 1e-2)
                # Swapping the two expert nodes keeps each token's per-rank route
                # count; local-only routes become remote and stale payload must clear.
                # Invalid (-1) slots stay invalid: shifting them would route every
                # pad slot of every token to one expert (duplicate ids, overflow).
                original_ids = shared.topk_ids.clone()
                shared.topk_ids.copy_(
                    torch.where(
                        original_ids >= 0,
                        (original_ids + shape.experts // 2) % shape.experts,
                        original_ids,
                    )
                )
                replay_vs("swapped_routing", ref_forward(), thr)
                shared.topk_ids.copy_(original_ids)
                replay_vs("restored_routing", eager, 1e-2)
            ok = all(c["ok"] for c in checks)
            if ok:
                self.time_path(res, graph, f"fused_t{tokens}_{fixture}")
                if do_check:
                    replay_vs("after_timing", eager, 1e-2)
                    self.check_variable_rows(op, shared, res)
                    ok = all(c["ok"] for c in checks)
            res["ok"] = ok
            if self.rank == 0 and self.guard:
                res["config_problems"] = check_fused_config(
                    op.stage1_kernel_name,
                    res.get("kernel_names"),
                    args.comm_quant,
                    res["stage1_tile"],
                    res["stage2_tile"],
                )
        finally:
            graph = None  # free the graph before the operator's window
            torch.cuda.synchronize(device)
            dist.barrier()
            op.close()
        return res

    def check_variable_rows(self, op, shared, res):
        """Compare unequal rank-local lengths to explicitly padded input."""
        torch, dist = self.torch, self.dist
        lengths = (0, 1, op.mtpr // 2, max(0, op.mtpr - 1), op.mtpr)
        tokens = lengths[self.rank % len(lengths)]
        x = torch.zeros_like(shared.x)
        ids = torch.full_like(shared.topk_ids, -1)
        weights = torch.zeros_like(shared.route_weights)
        x[:tokens].copy_(shared.x[:tokens])
        ids[:tokens].copy_(shared.topk_ids[:tokens])
        weights[:tokens].copy_(shared.route_weights[:tokens])
        expected = op.forward(x, weights, ids, slice_output=False).clone()
        actual = op.forward(
            x[:tokens], weights[:tokens], ids[:tokens], slice_output=False
        )
        torch.cuda.synchronize(self.device)
        res["checks"].append(
            self.rel_l2("variable_rows_full_output", expected, actual, 1e-2)
        )
        res["rank_input_rows"] = self.gather(tokens)
        q, scales = op.quantize(x[:tokens])
        for label, call in (
            (
                "variable_rows",
                lambda: op.forward(x[:tokens], weights[:tokens], ids[:tokens]),
            ),
            (
                "variable_rows_prequant",
                lambda: op.forward_prequant(q, scales, weights[:tokens], ids[:tokens]),
            ),
        ):
            graph, out = capture_graph(torch, dist, call, self.device)
            shapes_ok = self.gather(tuple(out.shape) == (tokens, op.model_dim))
            if not all(shapes_ok):
                raise AssertionError(f"{label}: output shape did not follow input rows")
            for replay in range(2):
                graph.replay()
                torch.cuda.synchronize(self.device)
                res["checks"].append(
                    self.rel_l2(f"{label}_replay{replay}", expected[:tokens], out, 1e-2)
                )
            del graph

        # Every rank may be empty simultaneously. Poison the complete output
        # before replay so an empty return cannot hide unwritten capacity rows.
        zeros = torch.zeros_like(expected)
        empty_x, empty_weights, empty_ids = x[:0], weights[:0], ids[:0]
        empty_q, empty_scales = op.quantize(empty_x)
        for label, call in (
            (
                "all_empty_rows",
                lambda: op.forward(
                    empty_x, empty_weights, empty_ids, slice_output=False
                ),
            ),
            (
                "all_empty_rows_prequant",
                lambda: op.forward_prequant(
                    empty_q,
                    empty_scales,
                    empty_weights,
                    empty_ids,
                    slice_output=False,
                ),
            ),
        ):
            graph, out = capture_graph(torch, dist, call, self.device)
            for replay in range(2):
                out.fill_(7.0)
                graph.replay()
                torch.cuda.synchronize(self.device)
                res["checks"].append(
                    self.rel_l2(f"{label}_replay{replay}", zeros, out, 1e-2)
                )
            del graph

    def run_smallop(self, shape, path, fixture):
        res = {"quant": path.quant}
        if self.args.timing:
            graph, _ = capture_graph(self.torch, self.dist, path.forward, self.device)
            self.time_path(
                res, graph, f"smallop_t{shape.tokens}_{fixture}", profile=False
            )
            del graph
        else:  # still look the config up in the fused_moe tune table
            path.forward()
            self.torch.cuda.synchronize(self.device)
        q_a = (
            "torch.float8_e4m3fn" if path.quant == "a8w4" else "torch.float4_e2m1fn_x2"
        )
        tune = self.gather(self.tune_log.take(q_a))
        res["tune"] = {
            "hits": sum(t["hits"] for t in tune),
            "misses": sum(t["misses"] for t in tune),
            "example": tune[0]["example"],
        }
        res["ok"] = bool(res["tune"]["misses"] == 0 or self.args.allow_tune_miss)
        return res

    def time_path(self, res, graph, case, profile=True):
        args = self.args
        if "total" in args.timing:
            samples = time_replays(
                self.torch, self.dist, graph, self.device, args.warmup, args.iters
            )
            res["total"] = gpu_stats(self.gather(samples))
        if profile and "breakdown" in args.timing:
            if "total" not in args.timing:
                for _ in range(args.warmup):
                    graph.replay()
                self.torch.cuda.synchronize(self.device)
            self.dist.barrier()
            trace = self.out_root / f"{case}_rank{self.rank}.json"
            rows, names = profile_breakdown(
                self.torch, graph, self.device, args.prof_replays, args.prof_tail, trace
            )
            gathered = self.gather(rows)
            if all(gathered):
                keys = set.intersection(*(set(r) for r in gathered))
                res["breakdown"] = {
                    k: statistics.mean(r[k] for r in gathered) for k in keys
                }
                res["breakdown_ranks"] = gathered
                res["kernel_names"] = names
            else:
                res["breakdown_error"] = (
                    names if rows is None else "missing on some ranks"
                )

    def run(self):
        failed = False
        for tpr in self.args.tprs:
            for fixture in self.args.fixtures:
                rec = self.run_case(tpr, fixture)
                rec["compare"] = compare(rec)
                self.results.append(rec)
                if self.rank == 0:
                    failed |= print_case(self.net, self.args, rec)
                    self.write_results()  # every case, so a later hang keeps it
        if self.rank == 0:
            print_summary(self.args, self.results)
            print(f"[RESULTS] {self.write_results()}", flush=True)
        return failed

    def write_results(self):
        out = self.out_root / "results.json"
        out.write_text(
            json.dumps(
                {
                    "args": {
                        k: sorted(v) if isinstance(v, set) else v
                        for k, v in vars(self.args).items()
                    },
                    "network": self.net,
                    "results": self.results,
                },
                indent=2,
            )
            + "\n"
        )
        return out

    def close(self):
        self.dist.barrier()
        if self.comm is not None:
            self.comm.destroy()
        self.dist.destroy_process_group()


# --------------------------------------------------------------------------
# Report (rank 0)
# --------------------------------------------------------------------------


def fmt(value):
    return "-" if value is None else f"{value:.1f}"


def compare(rec):
    """Speedup (small-op / fused) of one case's total time."""
    fused, small = rec.get("fused") or {}, rec.get("smallop") or {}
    ft, st = (fused.get("total") or {}).get("min"), (small.get("total") or {}).get(
        "min"
    )
    return {"speedup": st / ft if ft and st else None}


def fmt_x(value):
    return "-" if value is None else f"{value:.2f}x"


def print_case(net, args, rec):
    fused, small = rec.get("fused"), rec.get("smallop")
    p = lambda msg: print(msg, flush=True)
    p(f"[CASE] TPR={rec['tpr']} fixture={rec['fixture']}")
    failed = False
    if fused:
        t1, t2 = fused["stage1_tile"], fused["stage2_tile"]
        p(
            f"  tile      stage1 G={t1['tile_group']} (BM={32 * t1['tile_group']}) "
            f"split_local={t1['split_local']} fanout_ctas_per_dest={t1['fanout_ctas_per_dest']} [{t1['source']}]"
            f" | stage2 BM={t2['gemm2_bm']} BN={t2['gemm2_bn']} grid={t2['grid']} [{t2['source']}]"
            f" intra={t2.get('intra_comm', '-')} [{t2.get('comm_source', '-')}]"
        )
        p(f"  kernel    {fused['stage1_kernel']}")
        if fused["checks"]:
            state = "PASS" if all(c["ok"] for c in fused["checks"]) else "FAIL"
            p(
                f"  check     {state} "
                + "  ".join(
                    f"{c['label']}={c['rel_l2']:.4g}"
                    + ("" if c["ok"] else f"(>={c['threshold']})")
                    for c in fused["checks"]
                )
            )
        for problem in fused.get("config_problems", []):
            p(f"  GUARD FAIL {problem}")
        failed |= not fused["ok"] or bool(fused.get("config_problems"))
    ft = (fused or {}).get("total")
    st = (small or {}).get("total")
    if ft or st:
        speed = f"  speedup {fmt_x(rec['compare']['speedup'])}" if ft and st else ""
        p(
            f"  GPU total (us, full forward including quant and communication; "
            f"per-rank min / median, mean over ranks)  "
            f"fused {fmt((ft or {}).get('min'))} / {fmt((ft or {}).get('median'))}"
            f"  | small-op {fmt((st or {}).get('min'))} / {fmt((st or {}).get('median'))}{speed}"
        )
    fb = (fused or {}).get("breakdown")
    if fused and fused.get("breakdown_error"):
        p(f"  breakdown fused unavailable: {fused['breakdown_error']}")
    if fb:
        p(
            f"  breakdown (us, GPU kernels; same minimum-kernel-sum replay per rank "
            f"from {args.prof_replays - args.prof_tail}..{args.prof_replays - 1}, "
            f"mean over ranks; profiling separate from GPU total)"
        )
        for label, key in (
            ("quant", "quant"),
            ("stage1", "stage1"),
            ("stage2", "stage2"),
            ("quant+s1+s2", "device_total"),
            ("one kernel", "one"),
            ("replay span", "span"),
        ):
            if key in fb:
                p(f"    {label:<12} {fmt(fb[key]):>8}")
    if small:
        t = small["tune"]
        # Cached metadata emits no lookup log, which alone cannot prove a hit.
        state = "MISS" if t["misses"] else "HIT" if t["hits"] else "NO LOOKUP LOG"
        p(
            f"  small-op  {small['quant']} fused_moe tune {state} (hits={t['hits']} "
            f"misses={t['misses']}) csv={os.environ.get('AITER_CONFIG_FMOE')}"
        )
        if state == "MISS":
            p(f"    {t['example']}")
        failed |= not small["ok"]
    return failed


def print_summary(args, results):
    print(
        f"[SUMMARY] network={args.network} (us; GPU totals include quantization and "
        f"communication, mean of per-rank minima; speedup = SmallOp GPU / Mega GPU; "
        f"quant/stage1/stage2/sum come from separate profiling)",
        flush=True,
    )
    print(
        f"  {'TPR':>5} {'fixture':<9} {'check':<5} {'relL2':>8} "
        f"{'quant':>9} {'stage1':>9} {'stage2':>9} {'q+s1+s2':>9} "
        f"{'Mega GPU':>9} {'SmallOp GPU':>11} {'speedup':>8}",
        flush=True,
    )
    for rec in results:
        fused, small = rec.get("fused") or {}, rec.get("smallop") or {}
        checks = fused.get("checks") or []
        state = (
            "-"
            if not checks
            else (
                "PASS"
                if all(c["ok"] for c in checks) and not fused.get("config_problems")
                else "FAIL"
            )
        )
        rel = next((c["rel_l2"] for c in checks if c["label"] == "eager_vs_mori"), None)
        ft, st = (fused.get("total") or {}).get("min"), (small.get("total") or {}).get(
            "min"
        )
        fb = fused.get("breakdown") or {}
        print(
            f"  {rec['tpr']:>5} {rec['fixture']:<9} {state:<5} "
            f"{'-' if rel is None else f'{rel:.5f}':>8} "
            f"{fmt(fb.get('quant')):>9} {fmt(fb.get('stage1')):>9} "
            f"{fmt(fb.get('stage2')):>9} {fmt(fb.get('device_total')):>9} "
            f"{fmt(ft):>9} {fmt(st):>11} {fmt_x(rec['compare']['speedup']):>8}",
            flush=True,
        )


def main(argv=None):
    args = parse_args(argv)
    net = resolve_network(args)
    apply_env(args, net)
    runner = Runner(args, net)
    runner.log(
        f"[CONFIG] network={args.network} shape={net} tpr={args.tprs} "
        f"fixtures={args.fixtures} paths={sorted(args.paths)} "
        f"timing={sorted(args.timing) or 'none'} check={args.check} "
        f"comm_quant={args.comm_quant} set={args.env_set}"
    )
    try:
        failed = runner.run()
    finally:
        runner.close()
    return 1 if failed else 0


# --------------------------------------------------------------------------
# Launcher: plain python3 entry, one torchrun job per network
# --------------------------------------------------------------------------

# Transport environment of the two-node ionic setup; a value already exported
# by the caller wins for the setdefault ones.
LAUNCH_ENV_DEFAULT = {
    "MORI_DEVICE_NIC": "ionic",
    "MORI_RDMA_DEVICES": "^rocep193s0f0,rocep193s0f1",
    "MORI_IB_GID_INDEX": "1",
    "MORI_SHMEM_HEAP_SIZE": "40G",
    # SDMA queues for the fused stage2 at max_tok_per_rank >= 2048; without
    # them the operator keeps the xGMI push.
    "MORI_ENABLE_SDMA": "1",
}
LAUNCH_ENV_FORCED = {
    "OMP_NUM_THREADS": "1",
    "MORI_NUM_QP_PER_PE": "2",
    "MORI_EP_LAUNCH_CONFIG_MODE": "AUTO",
    "AITER_MOE_EXPERT_BALANCE": "true",
    "AMD_SERIALIZE_KERNEL": "0",
    "FLYDSL_RUNTIME_ENABLE_CACHE": "1",
}


def _run_job(cmd, env, log, timeout):
    """Run one torchrun job, copying its output to stdout and ``log``.

    On timeout the whole process group gets SIGTERM, then SIGKILL 30 s later.
    """
    proc = subprocess.Popen(
        cmd,
        env=env,
        cwd=REPO_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    expired = threading.Event()

    def kill():
        expired.set()
        for sig, wait in ((signal.SIGTERM, 30), (signal.SIGKILL, None)):
            try:
                os.killpg(proc.pid, sig)
            except ProcessLookupError:
                return
            if wait is None:
                return
            try:
                proc.wait(wait)
                return
            except subprocess.TimeoutExpired:
                pass

    timer = threading.Timer(timeout, kill)
    timer.start()
    try:
        for line in proc.stdout:
            sys.stdout.buffer.write(line)
            sys.stdout.flush()
            log.write(line)
            log.flush()
        rc = proc.wait()
    finally:
        timer.cancel()
    if expired.is_set():
        msg = f"[LAUNCH] timeout after {timeout}s, rc={rc}\n".encode()
        sys.stdout.buffer.write(msg)
        log.write(msg)
        rc = rc or 124
    return rc


def launch(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    args = parse_args(argv)
    if args.node_rank is None or not args.master_addr:
        raise SystemExit(
            "launcher needs --node-rank and --master-addr "
            "(or start the script under torchrun yourself)"
        )
    out_root = REPO_ROOT / args.out_dir / args.tag  # the jobs run in REPO_ROOT
    status = out_root / f"node{args.node_rank}.status"
    if status.exists():
        raise SystemExit(f"run tag already exists: {status}")
    out_root.mkdir(parents=True, exist_ok=True)

    env = dict(os.environ)
    for key, value in LAUNCH_ENV_DEFAULT.items():
        env.setdefault(key, value)
    env.update(LAUNCH_ENV_FORCED)
    for key in ("GLOO_SOCKET_IFNAME", "MORI_SOCKET_IFNAME", "NCCL_SOCKET_IFNAME"):
        env[key] = args.socket_ifname
    env["PYTHONPATH"] = os.pathsep.join(
        v for v in (str(REPO_ROOT), env.get("PYTHONPATH", "")) if v
    )

    multi = len(args.networks) > 1
    rcs = {}
    with open(out_root / f"node{args.node_rank}.log", "wb") as log:
        for i, network in enumerate(args.networks):
            tag = f"{args.tag}/{network}" if multi else args.tag
            cmd = [
                sys.executable,
                "-u",
                "-m",
                "torch.distributed.run",
                "--nnodes=2",
                f"--nproc-per-node={GPUS_PER_NODE}",
                f"--node-rank={args.node_rank}",
                f"--master-addr={args.master_addr}",
                f"--master-port={args.master_port + i}",
                "--max-restarts=0",
            ]
            if args.torchrun_wrap:
                cmd += ["--no-python", args.torchrun_wrap]
            cmd += [
                os.path.abspath(__file__),
                *argv,
                "--network",
                network,
                "--tag",
                tag,
            ]
            head = (
                f"[LAUNCH] {time.strftime('%T')} network={network} ({i + 1}/"
                f"{len(args.networks)}) node_rank={args.node_rank} "
                f"port={args.master_port + i} tag={tag}\n"
            ).encode()
            sys.stdout.buffer.write(head)
            sys.stdout.flush()
            log.write(head)
            rcs[network] = _run_job(cmd, env, log, args.timeout)
        tail = (
            f"[LAUNCH] {time.strftime('%T')} done "
            + " ".join(f"{n}={rc}" for n, rc in rcs.items())
            + "\n"
        ).encode()
        sys.stdout.buffer.write(tail)
        log.write(tail)
    rc = next((rc for rc in rcs.values() if rc), 0)
    status.write_text(f"{rc}\n")
    return rc


if __name__ == "__main__":
    sys.exit(main() if "RANK" in os.environ else launch())
