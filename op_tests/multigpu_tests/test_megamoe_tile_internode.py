# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""EP16 inter-node MegaMoE tile test: accuracy and performance in one pass.

Launch one torchrun per node (2 nodes x 8 GPUs); see
scripts/megamoe_tile/run_internode_test.sh.  Every (TPR, fixture) case of one
network runs in this process, largest TPR first:

  check      fused eager output vs the MORI A4W4 reference built on the same
             weights (relL2), then CUDA-graph replays vs eager, with changed
             inputs and swapped routing in between (--no-check skips all)
  total      bench_mega_moe.py timing: each replay is synchronized and timed
             with the host clock; per-rank min/median, mean over ranks
  breakdown  torch.profiler over --prof-replays more replays; per-kernel rows
             from the last --prof-tail of them

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
import os
import re
import statistics
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

NETWORKS = {
    "kimi_k3": dict(
        model_dim=3584, inter_dim=3072, experts=896, topk=16,
        activation="situv2", swiglu_limit=0.0, smallop_quant="a4w4",
        fmoe_csv="aiter/configs/model_configs/kimik3_a4w4_tuned_fmoe.csv",
    ),
    "dsv4": dict(
        model_dim=7168, inter_dim=3072, experts=384, topk=6,
        activation="silu", swiglu_limit=10.0, smallop_quant="a8w4",
        fmoe_csv="aiter/configs/model_configs/dsv4_fp8fp4_tuned_fmoe.csv",
    ),
}
EP_SIZE, GPUS_PER_NODE = 16, 8

# Per-shape knobs come from the tune tables (megamoe_tile_stage{1,2}_tuned.csv,
# else the built-in rule).  A sweep passes --set <key>=<its own table>; the
# guard resolves the same table, so it stays on for those.
TILE_ENV_KEYS = ("AITER_CONFIG_MEGAMOE_TILE_STAGE1", "AITER_CONFIG_MEGAMOE_TILE_STAGE2")

SMALLOP_ENV = {
    "a4w4": {"AITER_SITUV2_A4W4": "1", "AITER_SITUV2_A8W4": "0"},
    # Otherwise Silu+interleave picks a bf16 activation for small M.
    "a8w4": {"AITER_BF16_FP8_MOE_BOUND": "0", "ATOM_MOE_GU_ITLV": "1"},
}

# MORI launch geometry of the two small-op paths (dispatch blocks, RDMA blocks).
MORI_BLOCKS = {"a4w4": (96, 64), "a8w4": (256, 128)}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--network", choices=sorted(NETWORKS), default="kimi_k3")
    p.add_argument("--model-dim", type=int)
    p.add_argument("--inter-dim", type=int)
    p.add_argument("--experts", type=int)
    p.add_argument("--topk", type=int)
    p.add_argument("--activation", choices=("silu", "situv2"))
    p.add_argument("--tpr-list", default="512", help="tokens per rank, comma separated")
    p.add_argument("--fixtures", default="eplb,random",
                   help="route fixtures, comma separated: eplb, random, permuted")
    p.add_argument("--paths", default="fused,smallop",
                   help="fused and/or smallop, comma separated")
    p.add_argument("--timing", default="total,breakdown",
                   help="total and/or breakdown, comma separated; none = accuracy only")
    p.add_argument("--check", dest="check", action="store_true", default=True)
    p.add_argument("--no-check", dest="check", action="store_false",
                   help="skip the eager reference and the replay checks")
    p.add_argument("--rel-l2-threshold", type=float, default=5e-2)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--iters", type=int, default=40)
    p.add_argument("--prof-replays", type=int, default=20,
                   help="replays recorded by torch.profiler for the breakdown")
    p.add_argument("--prof-tail", type=int, default=5,
                   help="last profiled replays used for the breakdown statistics")
    p.add_argument("--rail-fp8", dest="rail_fp8", action="store_true", default=True)
    p.add_argument("--no-rail-fp8", dest="rail_fp8", action="store_false")
    p.add_argument("--smallop-quant", choices=("a4w4", "a8w4"))
    p.add_argument("--fmoe-csv", help="fused_moe tune table for the small-op baseline")
    p.add_argument("--allow-tune-miss", action="store_true",
                   help="do not fail when the small-op fused_moe misses its tune table")
    p.add_argument("--set", dest="env_set", action="append", default=[], metavar="KEY=VALUE",
                   help="env override (repeatable); anything but a tile tune table "
                        "disables the kernel-name guard")
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--out-dir", default="trace_data/internode")
    p.add_argument("--tag", default="run")
    args = p.parse_args(argv)
    args.paths = split_choices(p, "--paths", args.paths, ("fused", "smallop"))
    args.timing = split_choices(p, "--timing", args.timing, ("total", "breakdown", "none"))
    args.timing.discard("none")
    fixtures = split_choices(p, "--fixtures", args.fixtures, ("eplb", "random", "permuted"))
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
        parser.error(f"{flag} takes a comma separated subset of {allowed}, got {value!r}")
    return items


def resolve_network(args):
    net = dict(NETWORKS[args.network])
    for key in ("model_dim", "inter_dim", "experts", "topk", "activation"):
        if getattr(args, key) is not None:
            net[key] = getattr(args, key)
    if args.smallop_quant:
        net["smallop_quant"] = args.smallop_quant
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
    return "rank-balanced-hot", topk


# --------------------------------------------------------------------------
# Tile configuration and the kernel-name guard
# --------------------------------------------------------------------------

def stage1_tile(net, tpr):
    from aiter.ops.flydsl.kernels.megamoe_tile.stage1_tune import resolve_stage1_tile

    return resolve_stage1_tile(token=tpr * net["topk"], model_dim=net["model_dim"],
                               inter_dim=net["inter_dim"], expert=net["experts"] // EP_SIZE,
                               topk=net["topk"])


def stage2_tile(net, tpr):
    """GEMM2 (kernel1) N tile, resolved exactly as the operator does."""
    from aiter.jit.utils.chip_info import get_cu_num, get_gfx_runtime
    from aiter.ops.flydsl.kernels.megamoe_tile.stage2_tune import (
        lookup_stage2_tune, resolve_gemm2_bn, resolve_gemm2_cu)

    tuned = lookup_stage2_tune(gfx=get_gfx_runtime(), cu_num=get_cu_num(),
                               token=tpr * net["topk"], model_dim=net["model_dim"],
                               inter_dim=net["inter_dim"], expert=net["experts"] // EP_SIZE,
                               topk=net["topk"])
    bn, bn_source = resolve_gemm2_bn(tuned)
    cu, cu_source = resolve_gemm2_cu(tuned)
    source = bn_source if bn_source == cu_source else f"gemm2_bn={bn_source},gemm2_cu={cu_source}"
    # 0 means kernel1 launches the operator's worker_blocks (256) CTAs.
    return {"gemm2_bn": bn, "gemm2_cu": cu, "grid": cu or 256, "source": source}


def tile_fragments(tile):
    """(must contain, must not contain) kernel-name fragments for a tile config."""
    g = tile["tile_group"]
    # gmm1_bn (256) and compute_first (8) are constants: no _gbn, always _cf8.
    # _slg / _nsl stands for split_local together with its companion switches.
    has = ([f"_tg{g}"] if g != 1 else []) + [
        f"_cf{tile['compute_first']}", f"_fos{tile['fanout_shards']}",
        "_slg" if tile["split_local"] else "_nsl"]
    lacks = ([] if g != 1 else ["_tg"]) + ["_gbn", "_nsl" if tile["split_local"] else "_slg"]
    return has, lacks


def check_fused_config(stage1_name, stage2_names, rail_fp8, tile, s2tile):
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
    k2 = [n for n in stage2_names if n.startswith("megamoe_k2")]
    if len(k2) != 1:
        problems.append(f"expected one megamoe_k2 kernel, got {k2}")
    elif ("_qr8" in k2[0]) != rail_fp8:
        problems.append(f"k2 rail fp8 mismatch: {k2[0]}")
    k1 = [n for n in stage2_names if n.startswith("megamoe_stage2_compact")]
    want = (f"_t32x{s2tile['gemm2_bn']}x256_", f"_p1cu{s2tile['grid']}s")
    if len(k1) != 1:
        problems.append(f"expected one megamoe_stage2_compact kernel, got {k1}")
    elif any(w not in k1[0] for w in want):
        problems.append(f"kernel1 missing {[w for w in want if w not in k1[0]]} "
                        f"(stage2 tile {s2tile}): {k1[0][:90]}")
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
        lines = [m for m in self.lines if f"'{q_dtype_a}', 'torch.float4_e2m1fn_x2'" in m]
        self.lines = []
        misses = [m for m in lines if "using 2stage default" in m]
        return {"hits": len(lines) - len(misses), "misses": len(misses),
                "example": (misses or lines or [""])[0][:300]}


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
    def __init__(self, rank, world):
        import torch.distributed as dist
        from mori.cco import Communicator, UniqueId

        payload = [bytes(Communicator.get_unique_id()) if rank == 0 else None]
        dist.broadcast_object_list(payload, src=0)
        self._comm = Communicator.init(world, rank, UniqueId.from_bytes(payload[0]),
                                       per_rank_vmm=16 * 1024**3)
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

    def __init__(self, quant, shape, shared, rank, *, swiglu_limit, a8w4_weights=None):
        import aiter
        import mori
        import torch
        from aiter.fused_moe import fused_moe
        from aiter.ops.flydsl.moe_common import GateMode

        self.torch, self.quant, self.shape, self.shared = torch, quant, shape, shared
        self.blocks, self.rdma_blocks = MORI_BLOCKS[quant]
        if quant == "a4w4":
            from aiter.ops.quant import dynamic_per_group_scaled_quant

            self._quant_op = dynamic_per_group_scaled_quant
            w = shared.prepared_weights
            self.weights = (w.w1, w.w1_scale, w.w2, w.w2_scale)
            data_type = shared.a_quant.dtype
            self.activation = (aiter.ActivationType.Situv2 if shape.activation == "situv2"
                               else aiter.ActivationType.Silu)
            self.gate_mode = GateMode.SEPARATED.value
        else:
            from aiter.ops.flydsl.kernels.mega_moe.quant import per_1x32_mx_quant

            self._quant_op = per_1x32_mx_quant
            self.weights = a8w4_weights
            data_type = torch.float8_e4m3fn
            self.activation = aiter.ActivationType.Silu
            self.gate_mode = GateMode.INTERLEAVE.value
        self.swiglu_limit = float(swiglu_limit)
        self._fused_moe = fused_moe
        config = mori.ops.EpDispatchCombineConfig(
            data_type=data_type, rank=rank, world_size=EP_SIZE,
            hidden_dim=shape.hidden, scale_dim=shape.hidden // 32, scale_type_size=1,
            max_token_type_size=torch.bfloat16.itemsize,
            max_num_inp_token_per_rank=shape.tokens, max_total_recv_tokens=0,
            num_experts_per_rank=shape.local_experts, num_experts_per_token=shape.topk,
            kernel_type=mori.ops.EpDispatchCombineKernelType.InterNodeV1LL,
            warp_num_per_block=8, block_num=self.blocks, rdma_block_num=self.rdma_blocks,
            gpu_per_node=GPUS_PER_NODE, quant_type="none",
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
            from op_tests.multigpu_tests.megamoe_tile_ep16_inputs import (
                coalesce_reference_routes)

            ids, weights = coalesce_reference_routes(
                route_ids.cpu().tolist(), route_weights.cpu().tolist(),
                experts_per_rank=self.shape.local_experts, num_experts=self.shape.experts)
            route_ids = torch.tensor(ids, dtype=route_ids.dtype, device=route_ids.device)
            route_weights = torch.tensor(weights, dtype=route_weights.dtype,
                                         device=route_weights.device)
        if self.quant == "a4w4":
            self._quant_op(shared.a_quant, shared.x, shared.a_scale, 32, shuffle_scale=False)
            x_q, x_scale = shared.a_quant, shared.a_scale
        else:
            x_q, x_scale = self._quant_op(shared.x, quant_mode="fp8")
        dispatched, recv_weights, recv_scales, recv_ids, recv_tokens = self.op.dispatch(
            x_q, route_weights, x_scale, route_ids,
            block_num=self.blocks, rdma_block_num=self.rdma_blocks, warp_per_block=8)
        w1, w1_scale, w2, w2_scale = self.weights
        import aiter

        local_out = self._fused_moe(
            dispatched, w1, w2, recv_weights, recv_ids, shared.local_expert_mask,
            activation=self.activation, quant_type=aiter.QuantType.per_1x32,
            doweight_stage1=False, w1_scale=w1_scale, w2_scale=w2_scale,
            a1_scale=recv_scales, num_local_tokens=recv_tokens, dtype=torch.bfloat16,
            swiglu_limit=self.swiglu_limit, gate_mode=self.gate_mode)
        result = self.op.combine(local_out, None, route_ids, block_num=self.blocks,
                                 rdma_block_num=self.rdma_blocks, warp_per_block=4)
        return result[0] if isinstance(result, tuple) else result


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
    """bench_mega_moe.py: one sample = replay + synchronize, host wall clock (us)."""
    for _ in range(warmup):
        graph.replay()
    torch.cuda.synchronize(device)
    dist.barrier()
    samples = []
    for _ in range(iters):
        t0 = time.perf_counter()
        graph.replay()
        torch.cuda.synchronize(device)
        samples.append((time.perf_counter() - t0) * 1e6)
    dist.barrier()
    return samples


def kernel_class(name):
    if "megamoe_tile_ep16_stage1" in name:
        return "stage1"
    if name.startswith("megamoe_stage2_"):        # fused kernel1: GEMM2 + push
        return "gemm2"
    if name.startswith("megamoe_k2"):             # fused kernel2: node reduce + rail + combine
        return "k2"
    if name.startswith("EpDispatch"):
        return "dispatch"
    if "moe_sorting" in name or "mxfp4_moe_sort" in name:
        return "sorting"
    if name.startswith("mfma_moe1") or "moe1_" in name:
        return "gemm1"
    if name.startswith("gemm2_") or "moe2_" in name:
        return "gemm2"
    if name.startswith("EpCombine"):
        return "combine"
    if "quant" in name:
        return "quant"
    return "other"


REPLAY_MARK = re.compile(r"megamoe_replay(\d+)$")


def profile_breakdown(torch, graph, device, replays, tail, trace_file):
    """Per-rank minimum of each row over the last ``tail`` profiled replays.

    Returns (rows, kernel names) or (None, error) when the profiler recorded
    none of those replays; it drops one now and then, so a partial window is used.
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
        if (m and e.get("cat") == "gpu_user_annotation" and e.get("ph") == "X"
                and int(m.group(1)) >= replays - tail):
            windows.append((float(e["ts"]), float(e["ts"]) + float(e["dur"])))
    if not windows:
        return None, f"profiler recorded none of replays {replays - tail}..{replays - 1}"
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
        ends = [float(e["ts"]) + float(e["dur"]) for e in active]
        row["span"] = max(ends) - min(float(e["ts"]) for e in active)
        # Stage2 = first GEMM2 start to last k2/combine end, gaps included.
        first = [float(e["ts"]) for e in active if kernel_class(e["name"]) == "gemm2"]
        last = [float(e["ts"]) + float(e["dur"]) for e in active
                if kernel_class(e["name"]) in ("k2", "combine")]
        if first and last:
            row["stage2"] = max(last) - min(first)
        if "stage1" not in row and all(k in row for k in ("dispatch", "sorting", "gemm1")):
            row["stage1"] = row["dispatch"] + row["sorting"] + row["gemm1"]
        samples.append(row)
    if not samples:
        return None, "no kernels inside the profiled replays"
    keys = set.intersection(*(set(s) for s in samples))
    return {k: min(s[k] for s in samples) for k in keys}, sorted(names)


def wall_stats(rank_samples):
    mins = [min(s) for s in rank_samples]
    medians = [sorted(s)[len(s) // 2] for s in rank_samples]
    return {"min": statistics.mean(mins), "median": statistics.mean(medians),
            "pooled_min": min(mins), "worst_median": max(medians)}


# --------------------------------------------------------------------------
# Runner
# --------------------------------------------------------------------------

class Runner:
    def __init__(self, args, net):
        import torch
        import torch.distributed as dist
        from op_tests.multigpu_tests import megamoe_tile_ep16_inputs as inputs

        self.torch, self.dist, self.args, self.net = torch, dist, args, net
        self.inputs = inputs
        self.out_root = Path(args.out_dir) / args.tag
        self.out_root.mkdir(parents=True, exist_ok=True)
        self.rank, world, self.device = inputs.setup_dist(str(self.out_root / "preflight"))
        if world != EP_SIZE:
            raise ValueError(f"EP16 needs {EP_SIZE} ranks, got {world}")
        self.tune_log = TuneLog()
        logging.getLogger("aiter").addHandler(self.tune_log)
        self.guard = all(item.partition("=")[0] in TILE_ENV_KEYS for item in args.env_set)
        self.comm = None          # created by the first fused case
        # Weights depend on the shape only: prepare them once for every case.
        self.weights = inputs.prepare_weights(self.shape(1), self.rank, self.device,
                                              seed=args.seed)
        self.a8w4_weights = None
        if "smallop" in args.paths and net["smallop_quant"] == "a8w4":
            from op_tests.multigpu_tests.test_wide_ep_moe import _quantize_local_weights

            self.a8w4_weights, _ = _quantize_local_weights(
                net["model_dim"], net["inter_dim"], net["experts"] // EP_SIZE,
                self.rank, args.seed, self.device, quant="a8w4")
        self.results = []

    def shape(self, tpr):
        net = self.net
        shape = self.inputs.Shape(
            tokens=tpr, hidden=net["model_dim"], inter=net["inter_dim"],
            experts=net["experts"], topk=net["topk"], ep_size=EP_SIZE,
            gpus_per_node=GPUS_PER_NODE, activation=net["activation"])
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
        return {"label": label, "rel_l2": value, "threshold": threshold,
                "ok": value < threshold}

    def check_routes(self, shape, shared, fixture, cap):
        torch, epr = self.torch, shape.local_experts
        owners = torch.div(shared.topk_ids.long(), epr, rounding_mode="floor")
        per_rank = torch.zeros((shape.tokens, EP_SIZE), dtype=torch.int64, device=owners.device)
        per_rank.scatter_add_(1, owners, torch.ones_like(owners))
        if int(per_rank.max()) > cap:
            raise ValueError(f"{fixture}: {int(per_rank.max())} routes to one rank, cap {cap}")
        if fixture == "eplb":
            counts = torch.bincount(shared.topk_ids.flatten().long(), minlength=shape.experts).cpu()
            self.dist.all_reduce(counts)
            per_rank = counts.view(EP_SIZE, epr)
            padded = (per_rank + 31) // 32 * 32
            if (len(set(per_rank.sum(1).tolist())) != 1 or len(set(padded.sum(1).tolist())) != 1
                    or len({tuple(sorted(r)) for r in per_rank.tolist()}) != 1):
                raise AssertionError("eplb inputs are not balanced across ranks")

    # ---- one case ---------------------------------------------------------

    def run_case(self, tpr, fixture):
        args, net = self.args, self.net
        shape = self.shape(tpr)
        route, cap = fixture_route(fixture, shape.topk)
        shared = self.inputs.shared_inputs(shape, self.rank, self.device,
                                           route_pattern=route, seed=args.seed,
                                           prepared_weights=self.weights)
        self.check_routes(shape, shared, fixture, cap)
        rec = {"tpr": tpr, "fixture": fixture}
        quant = net["smallop_quant"]
        reference = None
        if (args.check and "fused" in args.paths) or (quant == "a4w4" and "smallop" in args.paths):
            reference = MoriSmallOp("a4w4", shape, shared, self.rank,
                                    swiglu_limit=net["swiglu_limit"])
        if "fused" in args.paths:
            rec["fused"] = self.run_fused(shape, shared, cap, reference, fixture)
        if "smallop" in args.paths:
            small = reference if quant == "a4w4" else MoriSmallOp(
                "a8w4", shape, shared, self.rank, swiglu_limit=net["swiglu_limit"],
                a8w4_weights=self.a8w4_weights)
            rec["smallop"] = self.run_smallop(shape, small, fixture)
            del small
        del reference, shared
        self.tune_log.lines.clear()
        gc.collect()
        self.torch.cuda.empty_cache()
        return rec

    def run_fused(self, shape, shared, cap, reference, fixture):
        from aiter.ops.flydsl.kernels.megamoe_tile import MegaMoETileA4W4

        torch, dist, args, device = self.torch, self.dist, self.args, self.device
        tokens = shape.tokens
        w = shared.prepared_weights
        if self.comm is None:
            self.comm = PooledCommunicator(self.rank, EP_SIZE)
        op = MegaMoETileA4W4(
            communicator=self.comm, rank=self.rank, world_size=EP_SIZE,
            model_dim=shape.hidden, inter_dim=shape.inter, experts=shape.experts,
            topk=shape.topk, quant="a4w4", w1=w.w1, w1_scale=w.w1_scale, w2=w.w2,
            w2_scale=w.w2_scale, max_tok_per_rank=tokens,
            max_routes_per_token_per_rank=cap, mega_scheme="hierarchical",
            swiglu_limit=self.net["swiglu_limit"], activation=shape.activation,
            device_generation=True, comm_quant_rail="fp8" if args.rail_fp8 else "none")
        res = {"stage1_kernel": op.stage1_kernel_name,
               "stage1_tile": stage1_tile(self.net, tokens),
               "stage2_tile": stage2_tile(self.net, tokens), "checks": []}
        coalesce = fixture == "permuted"
        graph = None
        try:
            call = lambda: op.forward(shared.x, shared.route_weights, shared.topk_ids)

            def ref_forward():
                out = reference.forward(coalesce_duplicates=coalesce)[:tokens].clone()
                torch.cuda.synchronize(device)
                dist.barrier()
                return out

            checks = res["checks"]
            thr = args.rel_l2_threshold
            if args.check:
                expected = ref_forward()
            eager = call()[:tokens].clone()
            torch.cuda.synchronize(device)
            if args.check:
                checks.append(self.rel_l2("eager_vs_mori", expected, eager, thr))
            graph, out = capture_graph(torch, dist, call, device)

            def replay_vs(label, ref, threshold):
                graph.replay()
                torch.cuda.synchronize(device)
                checks.append(self.rel_l2(label, ref, out[:tokens], threshold))

            if args.check:
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
                original_ids = shared.topk_ids.clone()
                shared.topk_ids.copy_((original_ids + shape.experts // 2) % shape.experts)
                replay_vs("swapped_routing", ref_forward(), thr)
                shared.topk_ids.copy_(original_ids)
                replay_vs("restored_routing", eager, 1e-2)
            ok = all(c["ok"] for c in checks)
            if ok:
                self.time_path(res, graph, f"fused_t{tokens}_{fixture}")
                if args.check:
                    replay_vs("after_timing", eager, 1e-2)
                    ok = checks[-1]["ok"]
            res["ok"] = ok
            if self.rank == 0 and self.guard:
                res["config_problems"] = check_fused_config(
                    op.stage1_kernel_name, res.get("kernel_names"), args.rail_fp8,
                    res["stage1_tile"], res["stage2_tile"])
        finally:
            graph = None      # free the graph before the operator's window
            torch.cuda.synchronize(device)
            dist.barrier()
            op.close()
        return res

    def run_smallop(self, shape, path, fixture):
        res = {"quant": path.quant}
        if self.args.timing:
            graph, _ = capture_graph(self.torch, self.dist, path.forward, self.device)
            self.time_path(res, graph, f"smallop_t{shape.tokens}_{fixture}")
            del graph
        else:  # still look the config up in the fused_moe tune table
            path.forward()
            self.torch.cuda.synchronize(self.device)
        q_a = "torch.float8_e4m3fn" if path.quant == "a8w4" else "torch.float4_e2m1fn_x2"
        tune = self.gather(self.tune_log.take(q_a))
        res["tune"] = {"hits": sum(t["hits"] for t in tune),
                       "misses": sum(t["misses"] for t in tune), "example": tune[0]["example"]}
        res["ok"] = bool(res["tune"]["misses"] == 0 or self.args.allow_tune_miss)
        return res

    def time_path(self, res, graph, case):
        args = self.args
        if "total" in args.timing:
            samples = time_replays(self.torch, self.dist, graph, self.device,
                                   args.warmup, args.iters)
            res["total"] = wall_stats(self.gather(samples))
        if "breakdown" in args.timing:
            trace = self.out_root / f"{case}_rank{self.rank}.json"
            rows, names = profile_breakdown(self.torch, graph, self.device,
                                            args.prof_replays, args.prof_tail, trace)
            gathered = self.gather(rows)
            if all(gathered):
                keys = set.intersection(*(set(r) for r in gathered))
                res["breakdown"] = {k: statistics.mean(r[k] for r in gathered) for k in keys}
                res["kernel_names"] = names
            else:
                res["breakdown_error"] = names if rows is None else "missing on some ranks"

    def run(self):
        failed = False
        for tpr in self.args.tprs:
            for fixture in self.args.fixtures:
                rec = self.run_case(tpr, fixture)
                rec["compare"] = compare(rec)
                self.results.append(rec)
                if self.rank == 0:
                    failed |= print_case(self.net, self.args, rec)
        if self.rank == 0:
            print_summary(self.args, self.results)
            out = self.out_root / "results.json"
            out.write_text(json.dumps({"args": {k: sorted(v) if isinstance(v, set) else v
                                                for k, v in vars(self.args).items()},
                                       "network": self.net, "results": self.results},
                                      indent=2) + "\n")
            print(f"[RESULTS] {out}", flush=True)
        return failed

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


def overlap_rate(compute, comm, fused):
    """bench_mega_moe.py's overlap rate: the share of the shorter of the small-op's
    back-to-back compute and communication kernels the fused kernels hide
    (1.0 = fully hidden, 0.0 = none, < 0 = fused is slower than not overlapping)."""
    if None in (compute, comm, fused) or min(compute, comm) <= 0:
        return None
    return (compute + comm - fused) / min(compute, comm)


def compare(rec):
    """Speedups (small-op / fused) and overlap rates of one case."""
    fused, small = rec.get("fused") or {}, rec.get("smallop") or {}
    ft, st = (fused.get("total") or {}).get("min"), (small.get("total") or {}).get("min")
    fb, sb = fused.get("breakdown") or {}, small.get("breakdown") or {}
    ratio = lambda a, b: a / b if a and b else None
    add = lambda *v: None if None in v else sum(v)
    s1 = (add(sb.get("sorting"), sb.get("gemm1")), sb.get("dispatch"), fb.get("stage1"))
    s2 = (sb.get("gemm2"), sb.get("combine"), add(fb.get("gemm2"), fb.get("k2")))
    return {
        "speedup": ratio(st, ft),
        "speedup_stage1": ratio(sb.get("stage1"), fb.get("stage1")),
        "speedup_stage2": ratio(sb.get("stage2"), fb.get("stage2")),
        "overlap_stage1": overlap_rate(*s1),
        "overlap_stage2": overlap_rate(*s2),
        "overlap": overlap_rate(*(add(a, b) for a, b in zip(s1, s2))),
    }


def fmt_x(value):
    return "-" if value is None else f"{value:.2f}x"


def fmt_pct(value):
    return "-" if value is None else f"{100 * value:.0f}%"


def print_case(net, args, rec):
    fused, small = rec.get("fused"), rec.get("smallop")
    p = lambda msg: print(msg, flush=True)
    p(f"[CASE] TPR={rec['tpr']} fixture={rec['fixture']}")
    failed = False
    if fused:
        t1, t2 = fused["stage1_tile"], fused["stage2_tile"]
        p(f"  tile      stage1 G={t1['tile_group']} (BM={32 * t1['tile_group']}) "
          f"split_local={t1['split_local']} fanout_shards={t1['fanout_shards']} [{t1['source']}]"
          f" | stage2 BN={t2['gemm2_bn']} grid={t2['grid']} [{t2['source']}]")
        p(f"  kernel    {fused['stage1_kernel']}")
        if fused["checks"]:
            state = "PASS" if all(c["ok"] for c in fused["checks"]) else "FAIL"
            p(f"  check     {state} " + "  ".join(
                f"{c['label']}={c['rel_l2']:.4g}" + ("" if c["ok"] else f"(>={c['threshold']})")
                for c in fused["checks"]))
        for problem in fused.get("config_problems", []):
            p(f"  GUARD FAIL {problem}")
        failed |= not fused["ok"] or bool(fused.get("config_problems"))
    ft = (fused or {}).get("total")
    st = (small or {}).get("total")
    if ft or st:
        speed = f"  speedup {fmt_x(rec['compare']['speedup'])}" if ft and st else ""
        p(f"  total     (us, per-rank min / median, mean over ranks)  "
          f"fused {fmt((ft or {}).get('min'))} / {fmt((ft or {}).get('median'))}"
          f"  | small-op {fmt((st or {}).get('min'))} / {fmt((st or {}).get('median'))}{speed}")
    fb = (fused or {}).get("breakdown")
    sb = (small or {}).get("breakdown")
    for name, r in (("fused", fused), ("small-op", small)):
        if r and r.get("breakdown_error"):
            p(f"  breakdown {name} unavailable: {r['breakdown_error']}")
    if fb or sb:
        fb, sb = fb or {}, sb or {}
        p(f"  breakdown (us, per-rank min over profiled replays "
          f"{args.prof_replays - args.prof_tail}..{args.prof_replays - 1}, mean over ranks)"
          f"   fused | small-op")
        rows = (("quant", "quant", "quant"),
                ("stage1", "stage1", "stage1"),
                ("  dispatch/sorting/gemm1", None, None),
                ("GEMM2", "gemm2", "gemm2"),
                ("k2 / combine", "k2", "combine"),
                ("stage2", "stage2", "stage2"),
                ("replay span", "span", "span"))
        for label, fk, sk in rows:
            if fk is None:
                if all(k in sb for k in ("dispatch", "sorting", "gemm1")):
                    p(f"    {label:<24} {'':>8} | {fmt(sb['dispatch'])} + {fmt(sb['sorting'])}"
                      f" + {fmt(sb['gemm1'])}")
                continue
            p(f"    {label:<24} {fmt(fb.get(fk)):>8} | {fmt(sb.get(sk)):>8}"
              + (f"  {fmt_x(sb[sk] / fb[fk])}" if fb.get(fk) and sb.get(sk) else ""))
        c = rec["compare"]
        p(f"  overlap   (bench_mega_moe: (compute + comm - fused) / min(compute, comm))  "
          f"stage1 {fmt_pct(c['overlap_stage1'])} (dispatch vs sorting+gemm1)  "
          f"stage2 {fmt_pct(c['overlap_stage2'])} (combine vs gemm2)  "
          f"all {fmt_pct(c['overlap'])}")
    if small:
        t = small["tune"]
        # fused_moe logs a lookup only the first time it sees a shape; a later
        # case of the same TPR reuses that (already checked) selection.
        state = "MISS" if t["misses"] else "HIT" if t["hits"] else "HIT (cached)"
        p(f"  small-op  {small['quant']} fused_moe tune {state} (hits={t['hits']} "
          f"misses={t['misses']}) csv={os.environ.get('AITER_CONFIG_FMOE')}")
        if state == "MISS":
            p(f"    {t['example']}")
        failed |= not small["ok"]
    return failed


def print_summary(args, results):
    print(f"[SUMMARY] network={args.network} (us; total = per-rank min, mean over ranks)",
          flush=True)
    print("  speedup = small-op / fused; s1/s2 from the breakdown rows; overlap = "
          "bench_mega_moe.py's rate, (compute + comm - fused) / min(compute, comm) of the "
          "small-op kernels; > 100% = fused beats a perfect overlap of them", flush=True)
    print(f"  {'TPR':>5} {'fixture':<9} {'check':<5} {'relL2':>8} {'fused':>8} "
          f"{'small-op':>9} {'speedup':>8} {'f.stage1':>9} {'f.stage2':>9} "
          f"{'s1 x':>6} {'s2 x':>6} {'ovl s1':>7} {'ovl s2':>7} {'ovl':>5}", flush=True)
    for rec in results:
        fused, small = rec.get("fused") or {}, rec.get("smallop") or {}
        checks = fused.get("checks") or []
        state = ("-" if not checks else
                 "PASS" if all(c["ok"] for c in checks) and not fused.get("config_problems")
                 else "FAIL")
        rel = next((c["rel_l2"] for c in checks if c["label"] == "eager_vs_mori"), None)
        ft, st = (fused.get("total") or {}).get("min"), (small.get("total") or {}).get("min")
        fb, c = fused.get("breakdown") or {}, rec["compare"]
        print(f"  {rec['tpr']:>5} {rec['fixture']:<9} {state:<5} "
              f"{'-' if rel is None else f'{rel:.5f}':>8} {fmt(ft):>8} {fmt(st):>9} "
              f"{fmt_x(c['speedup']):>8} {fmt(fb.get('stage1')):>9} {fmt(fb.get('stage2')):>9} "
              f"{fmt_x(c['speedup_stage1']):>6} {fmt_x(c['speedup_stage2']):>6} "
              f"{fmt_pct(c['overlap_stage1']):>7} {fmt_pct(c['overlap_stage2']):>7} "
              f"{fmt_pct(c['overlap']):>5}", flush=True)


def main(argv=None):
    args = parse_args(argv)
    net = resolve_network(args)
    apply_env(args, net)
    runner = Runner(args, net)
    runner.log(f"[CONFIG] network={args.network} shape={net} tpr={args.tprs} "
               f"fixtures={args.fixtures} paths={sorted(args.paths)} "
               f"timing={sorted(args.timing) or 'none'} check={args.check} "
               f"rail_fp8={args.rail_fp8} set={args.env_set}")
    try:
        failed = runner.run()
    finally:
        runner.close()
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
