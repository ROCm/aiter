# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""EP16 inter-node MegaMoE tile test: single-op, functional and performance modes.

Launch one torchrun per node (2 nodes x 8 GPUs); see
scripts/megamoe_tile/run_internode_test.sh.

  --mode op    the fused operator alone, one route fixture
               (--part full: eager check vs MORI + CUDA-graph timing,
                --part stage1: fused Stage1 only, graph-timed)
  --mode func  correctness and CUDA-graph replay checks over route fixtures
  --mode perf  fused operator vs small-op baseline (quant + MORI dispatch +
               tuned fused_moe + MORI combine), both captured and profiled
               in this process, per TPR

Shapes come from --network presets or explicit overrides; nothing assumes a
particular hidden size, expert count or topk.  The graph protocol (warmup,
replay checks, torch.profiler capture, rank statistics) is the existing
megamoe_stage2_graph.run_graph_profile, driven once per path.
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import statistics
import sys
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

# Best-known fused configuration (single-kernel Stage1 + two-kernel Stage2).
# Every entry is required; a missing one silently compiles a different kernel.
FUSED_BEST_ENV = {
    "MEGAMOE_TWO_KERNEL": "1",
    "MEGAMOE_TK_ARRIVAL": "1",
    "MEGAMOE_TK_RAIL": "1",
    "MEGAMOE_TK_WAIT_REMOTE": "1",
    "MEGAMOE_TK_EXPERT_MAJOR": "1",
    "MEGAMOE_TK_K1_TIME": "0",
    "MEGAMOE_TK_K2_TIME": "0",
    "MEGAMOE_TK_H1_PHYS": "1",
    "MEGAMOE_TK_S1_SPLIT_LOCAL": "1",
    "MEGAMOE_TK_S1_ROUTE_GBATCH": "1",
    "MEGAMOE_TK_S1_ASCALE_GATHER": "1",
    "MEGAMOE_TK_S1_RAIL_SOA": "1",
    "MEGAMOE_TK_S1_RAIL_QPS": "4",
    "MEGAMOE_TK_S1_T0_NOFAN": "1",
    "MEGAMOE_TK_S1_FAN1_DIRECT": "1",
    "MEGAMOE_TK_S1_FANOUT_SHARDS": "32",
    "MEGAMOE_TK_S1_WIDE_WAIT": "1",
    "MEGAMOE_TK_S1_WIDE_FANOUT_WAIT": "1",
    "MEGAMOE_TK_S1_DEFER_RECV": "1",
    "MEGAMOE_TK_S1_HOIST_WAIT": "1",
    "MEGAMOE_TK_S1_RAIL_POST_OFFT0": "1",
    "MEGAMOE_TK_S1_SORTCOPY_K2": "1",
    "MEGAMOE_TK_S1_GB_P3": "2",
    "MEGAMOE_TK_S1_META_OPT": "1",
    "MEGAMOE_TK_S1_RAIL_POST_CTAS": "1",
    "MEGAMOE_TK_S1_CREDIT_ASYNC": "1",
    "MEGAMOE_TK_S1_TILE_GROUP": "4",
    "MEGAMOE_TK_S1_KERNEL_SPLIT": "0",
    "MEGAMOE_TK_S1_GATE_SLEEP": "127",
    "MEGAMOE_TK_S1_EARLY_LOCAL_GMM": "1",
    "MEGAMOE_TK_S1_FAN2_SHARDS": "16",
    "MEGAMOE_TK_S1_PUB_RELAXED": "1",
    "MEGAMOE_TK_S1_CLAIM_RELAXED": "1",
    "MEGAMOE_TK_S1_COMPUTE_FIRST": "8",
    "MEGAMOE_TK_GMM1_LDS_SCOPES": "1",
    "MEGAMOE_TK_GMM1_EPI_SWZ": "1",
    "MEGAMOE_TK_GMM1_NOFENCE_BAR": "1",
    "MEGAMOE_TK_GMM1_NEXT_CLAIM": "1",
    "MEGAMOE_TK_S1_POST_NOFAN": "1",
}
# Kernel-name fragments the best configuration must produce.
FUSED_BEST_STAGE1_FRAGMENTS = (
    "_widewait", "_widefan", "_cf8", "_f1d", "_t0nf", "_cra", "_soa4", "_pcta",
    "_gb", "_asg", "_pofft0", "_sck2", "_gp2", "_mo1", "_slg", "_h1p",
    "_gs127", "_elg", "_f2s16", "_tg4", "_expertmajor", "_fos32", "_defrecv",
    "_hoistwait", "_lsc_nfb", "_esw", "_nxc2", "_pnf", "_prx", "_crx",
)

SMALLOP_ENV = {
    "a4w4": {"AITER_SITUV2_A4W4": "1", "AITER_SITUV2_A8W4": "0"},
    # Otherwise Silu+interleave picks a bf16 activation for small M.
    "a8w4": {"AITER_BF16_FP8_MOE_BOUND": "0", "ATOM_MOE_GU_ITLV": "1"},
}

# Production candidate knobs (run_case_relaxed.sh + run_stage2_graph_ep16.sh).
CANDIDATE_ARGS = [
    "--candidate-mode", "full", "--mori-mode", "gmm2_combine",
    "--stage1-workers", "256", "--stage2-workers", "256",
    "--candidate-node-accumulation-mode", "rank_local",
    "--candidate-rank-accumulation-mode", "reduce_push",
    "--candidate-rank-reduce-blocks", "56",
    "--candidate-node-reduce-vec-bytes", "16",
    "--candidate-node-reduce-load-schedule", "load_first",
    "--candidate-gmm-work-swizzle", "n_major_window",
    "--candidate-window-n-groups", "1",
    "--candidate-ready-granularity", "group",
    "--candidate-final-combine-blocks", "14",
    "--candidate-node-reduce-blocks", "16",
    "--candidate-node-reduce-token-owner-fastpath",
    "--candidate-rank-push-batch-invariants",
    "--candidate-rank-push-acquire-cohort", "2",
    "--candidate-rank-epilogue-barrier", "per_row",
    "--graph-comparison-statistic", "pooled_min",
]

TIMING_KEYS = (
    "input_quant_us", "fused_stage1_us", "fused_stage2_us", "dispatch_us",
    "sorting_us", "gemm1_us", "gemm2_us", "combine_us", "stage2_span_us",
    "pipeline_span_us",
)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=("op", "func", "perf"), required=True)
    p.add_argument("--network", choices=sorted(NETWORKS), default="kimi_k3")
    p.add_argument("--model-dim", type=int)
    p.add_argument("--inter-dim", type=int)
    p.add_argument("--experts", type=int)
    p.add_argument("--topk", type=int)
    p.add_argument("--activation", choices=("silu", "situv2"))
    p.add_argument("--tpr-list", default="512", help="tokens per rank, comma separated")
    p.add_argument("--fixtures", default="eplb,permuted",
                   help="func mode route fixtures: eplb, permuted, random")
    p.add_argument("--fixture", default="eplb", help="op/perf mode route fixture")
    p.add_argument("--part", choices=("full", "stage1"), default="full",
                   help="op mode: whole fused op, or fused Stage1 only")
    p.add_argument("--config", choices=("best", "env"), default="best",
                   help="best: force the best-known fused env; env: use the caller's env")
    p.add_argument("--set", dest="env_set", action="append", default=[], metavar="KEY=VALUE",
                   help="env override applied after --config best (repeatable); disables the "
                        "kernel-name guard, since the variant is deliberately not the best config")
    p.add_argument("--rail-fp8", dest="rail_fp8", action="store_true", default=True)
    p.add_argument("--no-rail-fp8", dest="rail_fp8", action="store_false")
    p.add_argument("--smallop-quant", choices=("a4w4", "a8w4"))
    p.add_argument("--fmoe-csv", help="fused_moe tune table for the small-op baseline")
    p.add_argument("--allow-tune-miss", action="store_true",
                   help="do not fail perf mode when fused_moe misses its tune table")
    p.add_argument("--skip-fused", action="store_true", help="perf mode: small-op only")
    p.add_argument("--skip-smallop", action="store_true", help="perf mode: fused only")
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--iters", type=int, default=40)
    p.add_argument("--tail-iters", type=int, default=20)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--rel-l2-threshold", type=float,
                   help="relL2 gate vs MORI; default 5e-2 in func mode, 1.0 (production) otherwise")
    p.add_argument("--out-dir", default="trace_data/internode")
    p.add_argument("--tag", default="run")
    return p.parse_args(argv)


def resolve_network(args):
    net = dict(NETWORKS[args.network])
    for key, attr in (("model_dim", "model_dim"), ("inter_dim", "inter_dim"),
                      ("experts", "experts"), ("topk", "topk"),
                      ("activation", "activation")):
        value = getattr(args, attr)
        if value is not None:
            net[key] = value
    if args.smallop_quant:
        net["smallop_quant"] = args.smallop_quant
    if args.fmoe_csv:
        net["fmoe_csv"] = args.fmoe_csv
    return net


def apply_env(args, net):
    """Must run before any aiter/operator import: some flags are read at import."""
    if args.config == "best":
        os.environ.update(FUSED_BEST_ENV)
        if args.rail_fp8:
            os.environ["MEGAMOE_TK_COMM_QUANT_RAIL"] = "fp8"
        else:
            os.environ.pop("MEGAMOE_TK_COMM_QUANT_RAIL", None)
    for item in args.env_set:
        key, sep, value = item.partition("=")
        if not sep:
            raise ValueError(f"--set expects KEY=VALUE, got {item!r}")
        os.environ[key] = value
    os.environ["MEGAMOE_EXPERTS"] = str(net["experts"])
    os.environ["MEGAMOE_TOPK"] = str(net["topk"])
    os.environ["MEGAMOE_INTER"] = str(net["inter_dim"])
    os.environ["MEGAMOE_EP_SIZE"] = "16"
    # silu clamp (DSV4 10.0); read by the fused operator and its MORI A4W4 reference.
    os.environ["MEGAMOE_SWIGLU_LIMIT"] = str(float(net["swiglu_limit"]))
    csv = Path(net["fmoe_csv"])
    os.environ["AITER_CONFIG_FMOE"] = str(csv if csv.is_absolute() else REPO_ROOT / csv)
    os.environ.update(SMALLOP_ENV[net["smallop_quant"]])
    os.environ.setdefault("AITER_LOG_LEVEL", "INFO")


def set_tpr_env(args, tpr):
    if args.config == "best":
        # One dispatch chunk per rank (k1 -9%); never below one record per QP.
        os.environ["MEGAMOE_TK_S1_RECORDS_PER_CHUNK"] = str(max(4, tpr))


def fixture_route(fixture, topk, ep_size):
    """Map a generic fixture name to the harness route pattern and route cap."""
    if fixture == "eplb":
        return ("cross_node" if topk == ep_size else "eplb-balanced"), 1
    if fixture == "permuted":
        if topk != 16:
            raise ValueError("the permuted fixture tables are defined for topk=16 only")
        return "permuted-arbitrary-topk", 4
    if fixture == "random":
        return "rank-balanced-hot", topk
    raise ValueError(f"unknown fixture {fixture!r}")


def harness_argv(args, net, *, path, tpr, fixture, trace_dir, replay_checks):
    route, cap = fixture_route(fixture, net["topk"], 16)
    argv = [
        "--path", path, "--cuda-graph",
        "--hidden", str(net["model_dim"]), "--activation", net["activation"],
        "--tokens", str(tpr), "--seed", str(args.seed),
        "--warmup", str(args.warmup), "--iters", str(args.iters),
        "--tail-iters", str(args.tail_iters),
        "--route-pattern", route, "--max-routes-per-token-per-rank", str(cap),
        "--graph-rel-l2-threshold", str(args.rel_l2_threshold),
        "--torch-profiler-dir", str(trace_dir),
        *CANDIDATE_ARGS,
        # The batch invariant needs whole source-token batches.
        "--candidate-rank-push-batch-size", str(min(64, tpr)),
    ]
    if replay_checks:
        argv.append("--graph-check-routing-replay")
    if fixture == "permuted" and path == "candidate":
        argv.append("--graph-coalesce-reference-duplicates")
    return argv


class TuneLog(logging.Handler):
    """Collect fused_moe 2-stage config selections ('using 2stage ...')."""

    def __init__(self):
        super().__init__(logging.INFO)
        self.lines = []

    def emit(self, record):
        msg = record.getMessage()
        if "using 2stage" in msg:
            self.lines.append(msg)

    def verdict(self, q_dtype_a=None):
        # The fused path's MORI A4W4 reference also calls fused_moe (FP4
        # activations); only lookups with the small-op's activation dtype count.
        lines = [m for m in self.lines if q_dtype_a is None or f"'{q_dtype_a}', 'torch.float4_e2m1fn_x2'" in m]
        hits = [m for m in lines if "using 2stage default" not in m]
        misses = [m for m in lines if "using 2stage default" in m]
        return {"hits": len(hits), "misses": len(misses),
                "example": (misses or hits or [""])[0][:300]}


class MoriA8W4Baseline:
    """Quant(MXFP8) + MORI InterNodeV1LL dispatch + A8W4 fused_moe + MORI combine.

    Same structure as MoriFusedMoeBaselinePath (the A4W4 small-op), with the
    DSV4 A8W4 settings of test_wide_ep_moe.py: FP8 activations, gate/up
    interleaved FP4 weights from its _quantize_local_weights, SiLU and the
    model swiglu_limit.  Unlike TestWideEpMoe no fake expert slot is appended:
    fused_moe keys its tune table on topk_ids.shape[1], and the EP16 rows of
    the model tables use the real TopK.
    """

    name = "quant_mori_a8_dispatch_fused_moe_combine"
    stage_names = ("quant_dispatch_fused_moe_combine",)
    full_stage_field = stage_names[0]

    def __init__(self, shape, shared, rank, world, *, swiglu_limit, seed):
        import aiter
        import mori
        from aiter.fused_moe import fused_moe
        from aiter.ops.flydsl.kernels.mega_moe.quant import per_1x32_mx_quant
        from aiter.ops.flydsl.moe_common import GateMode
        from op_tests.multigpu_tests.test_wide_ep_moe import _quantize_local_weights

        device = shared.x.device
        (self.w1, self.w1_scale, self.w2, self.w2_scale), _ = _quantize_local_weights(
            shape.hidden, shape.inter, shape.local_experts, rank, seed, device, quant="a8w4")
        self.torch = __import__("torch")
        self.torch.cuda.empty_cache()
        config = mori.ops.EpDispatchCombineConfig(
            data_type=self.torch.float8_e4m3fn, rank=rank, world_size=world,
            hidden_dim=shape.hidden, scale_dim=shape.hidden // 32, scale_type_size=1,
            max_token_type_size=self.torch.bfloat16.itemsize,
            max_num_inp_token_per_rank=shape.tokens, max_total_recv_tokens=0,
            num_experts_per_rank=shape.local_experts, num_experts_per_token=shape.topk,
            kernel_type=mori.ops.EpDispatchCombineKernelType.InterNodeV1LL,
            warp_num_per_block=8, block_num=256, rdma_block_num=128,
            gpu_per_node=shape.gpus_per_node, quant_type="none",
        )
        self.shape, self.shared, self.rank = shape, shared, rank
        self.op = mori.ops.EpDispatchCombineOp(config)
        self._fused_moe = fused_moe
        self._quant = per_1x32_mx_quant
        self._activation = aiter.ActivationType.Silu
        self._quant_type = aiter.QuantType.per_1x32
        self._gate_mode = GateMode.INTERLEAVE.value
        self._swiglu_limit = float(swiglu_limit)

    def _forward(self, *, validate_recv=False, debug_sync=False,
                 reference_coalesce_duplicates=False):
        if reference_coalesce_duplicates:
            raise ValueError("the A8W4 small-op is a timing path, not a coalesced oracle")
        route_ids, route_weights = self.shared.topk_ids, self.shared.route_weights
        x_q, x_scale = self._quant(self.shared.x, quant_mode="fp8")
        dispatched, recv_weights, recv_scales, recv_ids, recv_tokens = self.op.dispatch(
            x_q, route_weights, x_scale, route_ids,
            block_num=256, rdma_block_num=128, warp_per_block=8,
        )
        local_out = self._fused_moe(
            dispatched, self.w1, self.w2, recv_weights, recv_ids,
            self.shared.local_expert_mask,
            activation=self._activation, quant_type=self._quant_type,
            doweight_stage1=False, w1_scale=self.w1_scale, w2_scale=self.w2_scale,
            a1_scale=recv_scales, num_local_tokens=recv_tokens,
            dtype=self.torch.bfloat16, swiglu_limit=self._swiglu_limit,
            gate_mode=self._gate_mode,
        )
        result = self.op.combine(local_out, None, route_ids,
                                 block_num=256, rdma_block_num=128, warp_per_block=4)
        return result[0] if isinstance(result, tuple) else result


class SmallOpPath:
    """The graph driver only touches .baseline on the MORI path."""

    def __init__(self, baseline):
        self.baseline = baseline

    def close(self):
        return None


def split_fused_stage2(trace_file, tail):
    """Per-rank minimum of the two Stage2 kernels and Stage1 over the tail."""
    events = json.loads(Path(trace_file).read_text())["traceEvents"]
    kernels = [e for e in events if e.get("ph") == "X"
               and e.get("cat") in ("kernel", "gpu_op", "Kernel")]
    classes = {"gemm2": ("megamoe_stage2_compact", "megamoe_stage2_fixedslot"),
               "k2": ("megamoe_k2",), "stage1": ("megamoe_tile_ep16_stage1",)}
    out = {}
    for key, prefixes in classes.items():
        hits = sorted((e for e in kernels if e["name"].startswith(prefixes)),
                      key=lambda e: e["ts"])[-tail:]
        out[key] = min(e["dur"] for e in hits) if hits else None
    return out


def kernel_names(trace_dir, rank):
    samples = json.loads((Path(trace_dir) / f"rank{rank}.samples.json").read_text())["samples"]
    names = {}
    for sample in samples:
        for key, values in sample.get("kernel_names", {}).items():
            names.setdefault(key, set()).update(values)
    return {k: sorted(v) for k, v in names.items()}


def check_fused_config(names, args):
    """Kernel-name guard for --config best (rank 0 only)."""
    problems = []
    if args.config != "best" or args.env_set:
        return problems
    stage1 = names.get("fused_stage1", [])
    if len(stage1) != 1:
        problems.append(f"expected one fused_stage1 kernel, got {stage1}")
    else:
        missing = [f for f in FUSED_BEST_STAGE1_FRAGMENTS if f not in stage1[0]]
        if missing:
            problems.append(f"fused_stage1 missing {missing}")
    k2 = [n for n in names.get("fused_stage2", []) if n.startswith("megamoe_k2")]
    if len(k2) != 1:
        problems.append(f"expected one megamoe_k2 kernel, got {k2}")
    elif ("_qr8" in k2[0]) != args.rail_fp8:
        problems.append(f"k2 rail fp8 mismatch: {k2[0]}")
    return problems


def metric_table(summary):
    metrics = summary["timing"]["metrics"]
    table = {}
    for key in TIMING_KEYS:
        m = metrics.get(key)
        if m and m.get("rank_mean_of_min", 0) > 0:
            table[key] = {"rmm": m["rank_mean_of_min"], "pm": m["pooled_min"],
                          "cv": m["rank_mean_cv"]}
    return table


class Runner:
    def __init__(self, args, net):
        import torch
        import torch.distributed as dist
        from op_tests.multigpu_tests import bench_megamoe_tile_ep16_stage2_breakdown as bd
        from op_tests.multigpu_tests.bench_megamoe_tile_ep16_dual_path import (
            _setup_dist, _shared_inputs,
        )
        from op_tests.multigpu_tests.megamoe_stage2_graph import run_graph_profile

        self.torch, self.dist, self.bd = torch, dist, bd
        self.args, self.net = args, net
        self._shared_inputs = _shared_inputs
        self._run_graph_profile = run_graph_profile
        self.out_root = Path(args.out_dir) / args.tag
        self.out_root.mkdir(parents=True, exist_ok=True)
        self.rank, self.world, _local, self.device = _setup_dist(
            needs_mori=True, gpu_preflight_dir=str(self.out_root / "preflight"))
        if self.world != 16:
            raise ValueError(f"EP16 needs 16 ranks, got {self.world}")
        self.tune_log = TuneLog()
        logging.getLogger("aiter").addHandler(self.tune_log)
        self.results = []

    def log(self, msg):
        if self.rank == 0:
            print(msg, flush=True)

    def gather(self, obj):
        rows = [None] * self.world
        self.dist.all_gather_object(rows, obj)
        return rows

    def _prepare(self, path, tpr, fixture, name, replay_checks):
        trace_dir = self.out_root / name
        argv = harness_argv(self.args, self.net, path=path, tpr=tpr, fixture=fixture,
                            trace_dir=trace_dir, replay_checks=replay_checks)
        bargs = self.bd.build_parser().parse_args(argv)
        shape, contract = self.bd.prepare_run(bargs)
        return bargs, shape, contract, trace_dir

    def _shared(self, bargs, shape):
        return self._shared_inputs(shape, self.rank, self.world, self.device,
                                   route_pattern=bargs.route_pattern, seed=bargs.seed)

    def _profile(self, path, shared, shape, bargs, contract):
        """Run the graph protocol; returns (summary or None, error or None) on rank 0."""
        error = None
        try:
            self._run_graph_profile(path, shared, shape, bargs, contract,
                                    self.rank, self.world, self.device)
        except AssertionError as exc:  # numerical checks fail on every rank together
            error = str(exc)
        summary = None
        summary_file = Path(bargs.torch_profiler_dir) / "summary.json"
        if self.rank == 0 and error is None and summary_file.exists():
            summary = json.loads(summary_file.read_text())
        return summary, error

    def run_fused(self, tpr, fixture, name, *, replay_checks, stage1_only=False):
        set_tpr_env(self.args, tpr)
        if stage1_only:
            os.environ["MEGAMOE_TK_S1_ONLY"] = "1"
            os.environ.setdefault("MEGAMOE_TK_S1_ONLY_REPLAYS", str(self.args.iters))
        else:
            os.environ.pop("MEGAMOE_TK_S1_ONLY", None)
        bargs, shape, contract, trace_dir = self._prepare(
            "candidate", tpr, fixture, name, replay_checks)
        shared = self._shared(bargs, shape)
        path = self.bd.build_path(bargs, shape, shared, self.rank, self.world, self.device)
        try:
            summary, error = self._profile(path, shared, shape, bargs, contract)
        finally:
            self.dist.barrier()
            path.close()
            os.environ.pop("MEGAMOE_TK_S1_ONLY", None)
        record = {"path": "fused", "tpr": tpr, "fixture": fixture, "name": name,
                  "error": error}
        if stage1_only:
            record["note"] = "stage1-only: see MEGAMOE_S1_ONLY line in the log"
            return record
        split = None
        if error is None:
            local = split_fused_stage2(trace_dir / f"rank{self.rank}.json",
                                       self.args.tail_iters)
            rows = self.gather(local)
            split = {k: statistics.mean(r[k] for r in rows if r[k] is not None)
                     for k in ("gemm2", "k2", "stage1") if all(r[k] is not None for r in rows)}
            split.update({f"{k}_pm": min(r[k] for r in rows)
                          for k in ("gemm2", "k2") if all(r[k] is not None for r in rows)})
        if self.rank == 0 and summary is not None:
            record["timing"] = metric_table(summary)
            record["correctness"] = summary.get("correctness", [])
            names = kernel_names(trace_dir, 0)
            record["kernel_names"] = names
            record["config_problems"] = check_fused_config(names, self.args)
            record["split"] = split
        del shared
        gc.collect()
        self.torch.cuda.empty_cache()
        return record

    def run_smallop(self, tpr, fixture, name):
        bargs, shape, contract, trace_dir = self._prepare("mori", tpr, fixture, name, False)
        shared = self._shared(bargs, shape)
        quant = self.net["smallop_quant"]
        if quant == "a4w4":
            path = self.bd.build_path(bargs, shape, shared, self.rank, self.world, self.device)
        else:
            baseline = MoriA8W4Baseline(shape, shared, self.rank, self.world,
                                        swiglu_limit=self.net["swiglu_limit"],
                                        seed=self.args.seed)
            path = SmallOpPath(baseline)
        try:
            summary, error = self._profile(path, shared, shape, bargs, contract)
            op = path.baseline.op
            launch = {"mode": getattr(op, "launch_config_mode", None),
                      "dispatch": str(getattr(op, "_cached_dispatch_launch", None)),
                      "combine": str(getattr(op, "_cached_combine_launch", None))}
            q_a = "torch.float8_e4m3fn" if quant == "a8w4" else "torch.float4_e2m1fn_x2"
            tune = self.gather(self.tune_log.verdict(q_a))
        finally:
            self.dist.barrier()
            path.close()
        record = {"path": f"smallop_{quant}", "tpr": tpr, "fixture": fixture,
                  "name": name, "error": error}
        if self.rank == 0:
            record["mori_launch"] = launch
            record["tune"] = {"hits": sum(t["hits"] for t in tune),
                              "misses": sum(t["misses"] for t in tune),
                              "example": tune[0]["example"]}
            if summary is not None:
                record["timing"] = metric_table(summary)
                record["kernel_names"] = kernel_names(trace_dir, 0)
        del path, shared
        gc.collect()
        self.torch.cuda.empty_cache()
        return record

    def finish(self):
        self.log("")
        if self.rank == 0:
            out = self.out_root / "results.json"
            out.write_text(json.dumps({"args": vars(self.args), "network": self.net,
                                       "results": self.results}, indent=2) + "\n")
            self.log(f"[RESULTS] {out}")
        self.dist.barrier()
        self.dist.destroy_process_group()


def fmt(value):
    return "-" if value is None else f"{value:.1f}"


def print_perf(runner, tpr, fused, small):
    if runner.rank != 0:
        return
    ft, st = fused.get("timing", {}), small.get("timing", {})
    sp = fused.get("split") or {}
    get = lambda t, k: (t.get(k) or {}).get("rmm")
    s1_small = None
    if all(get(st, k) for k in ("dispatch_us", "sorting_us", "gemm1_us")):
        s1_small = sum(get(st, k) for k in ("dispatch_us", "sorting_us", "gemm1_us"))
    pipe_f, pipe_s = get(ft, "pipeline_span_us"), get(st, "pipeline_span_us")
    speed = f"{pipe_s / pipe_f:.2f}x" if pipe_f and pipe_s else "-"
    print(f"[PERF] TPR={tpr} (rank_mean_of_min, us)  fused | small-op", flush=True)
    print(f"  stage1            {fmt(get(ft, 'fused_stage1_us')):>8} | {fmt(s1_small):>8}"
          f"  (dispatch {fmt(get(st, 'dispatch_us'))} + sorting {fmt(get(st, 'sorting_us'))}"
          f" + gemm1 {fmt(get(st, 'gemm1_us'))})", flush=True)
    print(f"  stage2 GEMM2      {fmt(sp.get('gemm2')):>8} | {fmt(get(st, 'gemm2_us')):>8}", flush=True)
    print(f"  stage2 k2/combine {fmt(sp.get('k2')):>8} | {fmt(get(st, 'combine_us')):>8}", flush=True)
    print(f"  stage2 total      {fmt(get(ft, 'fused_stage2_us')):>8} | {fmt(get(st, 'stage2_span_us')):>8}", flush=True)
    print(f"  pipeline          {fmt(pipe_f):>8} | {fmt(pipe_s):>8}  speedup {speed}", flush=True)
    for rec in (fused, small):
        if rec.get("error"):
            print(f"  ERROR {rec['path']}: {rec['error']}", flush=True)
    for problem in fused.get("config_problems", []):
        print(f"  CONFIG-GUARD FAIL: {problem}", flush=True)
    if "tune" in small:
        t = small["tune"]
        state = "HIT" if t["misses"] == 0 and t["hits"] > 0 else "MISS"
        print(f"  small-op fused_moe tune: {state} (hits={t['hits']} misses={t['misses']}) "
              f"csv={os.environ.get('AITER_CONFIG_FMOE')}", flush=True)
        if state == "MISS":
            print(f"    {t['example']}", flush=True)
    if "mori_launch" in small:
        print(f"  MORI launch: {small['mori_launch']}", flush=True)


def main(argv=None):
    args = parse_args(argv)
    if args.rel_l2_threshold is None:
        args.rel_l2_threshold = 5e-2 if args.mode == "func" else 1.0
    net = resolve_network(args)
    apply_env(args, net)
    tprs = [int(v) for v in args.tpr_list.split(",") if v]
    runner = Runner(args, net)
    runner.log(f"[CONFIG] mode={args.mode} network={args.network} shape={net} "
               f"config={args.config} rail_fp8={args.rail_fp8} tpr={tprs} set={args.env_set}")
    failed = False
    try:
        for tpr in tprs:
            if args.mode == "op":
                rec = runner.run_fused(tpr, args.fixture, f"op_{args.part}_t{tpr}",
                                       replay_checks=False,
                                       stage1_only=args.part == "stage1")
                runner.results.append(rec)
                if runner.rank == 0:
                    print(f"[OP] TPR={tpr} part={args.part} "
                          + json.dumps({k: rec.get(k) for k in
                                        ("error", "timing", "split", "correctness",
                                         "config_problems")}), flush=True)
                failed |= bool(rec.get("error") or rec.get("config_problems"))
            elif args.mode == "func":
                for fixture in [f for f in args.fixtures.split(",") if f]:
                    rec = runner.run_fused(tpr, fixture, f"func_{fixture}_t{tpr}",
                                           replay_checks=True)
                    runner.results.append(rec)
                    if runner.rank == 0:
                        state = "FAIL" if rec.get("error") else "PASS"
                        print(f"[FUNC] {state} TPR={tpr} fixture={fixture}", flush=True)
                        for check in rec.get("correctness", []):
                            print(f"    {check['label']}: {check['rank_max_rel_l2']:.6g}"
                                  f" (threshold {check['threshold']})", flush=True)
                        if rec.get("error"):
                            print(f"    {rec['error']}", flush=True)
                    failed |= bool(rec.get("error"))
            else:
                fused = {} if args.skip_fused else runner.run_fused(
                    tpr, args.fixture, f"perf_fused_t{tpr}", replay_checks=False)
                small = {} if args.skip_smallop else runner.run_smallop(
                    tpr, args.fixture, f"perf_small_t{tpr}")
                runner.results += [r for r in (fused, small) if r]
                print_perf(runner, tpr, fused, small)
                failed |= bool(fused.get("error") or small.get("error")
                               or fused.get("config_problems"))
                if runner.rank == 0 and small.get("tune", {}).get("misses") and not args.allow_tune_miss:
                    failed = True
    finally:
        runner.finish()
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
