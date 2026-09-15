# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Compare Mori EP and MegaMoEV2 with the same v4_pro A8W4 CUDA Graph workload."""

from __future__ import annotations

import argparse
import os
from dataclasses import replace
from pathlib import Path

os.environ.setdefault("MORI_EP_LAUNCH_CONFIG_MODE", "AUTO")
os.environ.setdefault("MORI_SHMEM_HEAP_SIZE", "40G")

import mori
import mori.shmem as ms
import torch
import torch.distributed as dist
from torch.profiler import ProfilerActivity, profile

import aiter
from aiter import dtypes
from aiter.fused_moe import fused_moe
from aiter.ops.flydsl.kernels.mega_moe import MegaMoEV2
from aiter.ops.flydsl.moe_common import GateMode
from aiter.ops.shuffle import shuffle_scale_a16w4, shuffle_weight_a16w4

MODEL_DIM = 7168
INTER_DIM = 3072
EXPERTS = 384
TOPK = 6
SWIGLU_LIMIT = 10.0

PERF_GUARD_MIN_SPEEDUP = {
    (512, "uniform"): 140.0,
    (512, "rank-mixed-skew"): 110.0,
    (8192, "uniform"): 50.0,
    (8192, "rank-mixed-skew"): 40.0,
}


def setup_dist():
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("cpu:gloo,cuda:nccl", device_id=device)
    import torch._C._distributed_c10d as c10d

    c10d._register_process_group("default", dist.group.WORLD)
    ms.shmem_torch_process_group_init("default")
    return rank, world, device


def barrier():
    torch.cuda.synchronize()
    ms.shmem_barrier_all()


def make_inputs(tokens, rank, world, model_dim, experts, topk, route, hot_bias, device):
    local_experts = experts // world
    generator = torch.Generator(device=device).manual_seed(1234 + rank)
    x = torch.randn(
        (tokens, model_dim), dtype=torch.bfloat16, device=device, generator=generator
    )
    scores = torch.randn(
        (tokens, experts), dtype=torch.float32, device=device, generator=generator
    )
    if route == "hot-rank0":
        scores[:, :local_experts] += hot_bias
    values, ids = torch.topk(scores, topk, dim=-1)
    if route in ("rank-balanced-hot", "rank-balanced-last", "rank-mixed-skew"):
        destination_scores = torch.rand(
            (tokens, world), device=device, generator=generator
        )
        destination = torch.topk(destination_scores, topk, dim=-1).indices
        if route == "rank-balanced-last":
            hot = torch.ones_like(destination, dtype=torch.bool)
        elif route == "rank-mixed-skew":
            hot = destination < world // 2
        else:
            hot = (
                torch.rand((tokens, topk), device=device, generator=generator)
                < hot_bias
            )
        cold_expert = torch.randint(
            1, local_experts, (tokens, topk), device=device, generator=generator
        )
        hot_expert = local_experts - 1 if route == "rank-balanced-last" else 0
        ids = destination * local_experts + torch.where(hot, hot_expert, cold_expert)
        values = torch.randn(
            (tokens, topk), dtype=torch.float32, device=device, generator=generator
        )
    return (
        x.contiguous(),
        values.softmax(dim=-1).contiguous(),
        ids.to(torch.int32).contiguous(),
    )


def make_weights(local_experts, model_dim, inter_dim, rank, device):
    generator = torch.Generator(device=device).manual_seed(9000 + rank)
    quantize = aiter.get_torch_quant(aiter.QuantType.per_1x32)
    w1 = torch.randn(
        (local_experts, 2 * inter_dim, model_dim),
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    )
    w1.mul_(model_dim**-0.25)
    w1_q, w1_scale = quantize(w1, quant_dtype=dtypes.fp4x2)
    del w1
    w1_q = w1_q.view(local_experts, 2 * inter_dim, model_dim // 2)
    w1_q = shuffle_weight_a16w4(w1_q, 16, True).contiguous()
    w1_scale = shuffle_scale_a16w4(w1_scale, local_experts, True).contiguous()

    w2 = torch.randn(
        (local_experts, model_dim, inter_dim),
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    )
    w2.mul_(inter_dim**-0.25)
    w2_q, w2_scale = quantize(w2, quant_dtype=dtypes.fp4x2)
    del w2
    w2_q = w2_q.view(local_experts, model_dim, inter_dim // 2)
    w2_q = shuffle_weight_a16w4(w2_q, 16, False).contiguous()
    w2_scale = shuffle_scale_a16w4(w2_scale, local_experts, False).contiguous()
    torch.cuda.empty_cache()
    return w1_q, w1_scale, w2_q, w2_scale


def make_weights_tp(experts, model_dim, inter_dim, world, rank, device):
    """TP-sharded weights: every rank holds ALL experts, sharded along inter_dim.

    Rebuilds each EP owner rank's full-precision weights from the same seeds
    `make_weights` uses (9000 + owner), then takes this rank's inter_dim shard,
    so the result is numerically comparable to the EP (mori_body) weights for
    accuracy checks. The raw (pre-shuffle) w1 is always block-concatenated
    GGUU — rows [0, inter_dim) = gate, [inter_dim, 2*inter_dim) = up, regardless
    of GateMode — so the inter_dim shard [lo, hi) must be taken from BOTH
    halves separately and re-concatenated; GateMode.INTERLEAVE's row-pairwise
    layout is produced later, inside shuffle_weight_a16w4 itself.
    """
    local_experts_ep = experts // world
    inter_shard = inter_dim // world
    lo, hi = rank * inter_shard, (rank + 1) * inter_shard
    quantize = aiter.get_torch_quant(aiter.QuantType.per_1x32)

    w1_shards = []
    w2_shards = []
    for owner in range(world):
        generator = torch.Generator(device=device).manual_seed(9000 + owner)
        w1_full = torch.randn(
            (local_experts_ep, 2 * inter_dim, model_dim),
            dtype=torch.bfloat16,
            device=device,
            generator=generator,
        )
        w1_full.mul_(model_dim**-0.25)
        w2_full = torch.randn(
            (local_experts_ep, model_dim, inter_dim),
            dtype=torch.bfloat16,
            device=device,
            generator=generator,
        )
        w2_full.mul_(inter_dim**-0.25)
        w1_shards.append(
            torch.cat(
                (
                    w1_full[:, lo:hi, :],
                    w1_full[:, inter_dim + lo : inter_dim + hi, :],
                ),
                dim=1,
            ).clone()
        )
        w2_shards.append(w2_full[:, :, lo:hi].clone())
        del w1_full, w2_full
    w1_shard = torch.cat(w1_shards, dim=0).contiguous()
    w2_shard = torch.cat(w2_shards, dim=0).contiguous()
    del w1_shards, w2_shards

    w1_q, w1_scale = quantize(w1_shard, quant_dtype=dtypes.fp4x2)
    del w1_shard
    w1_q = w1_q.view(experts, 2 * inter_shard, model_dim // 2)
    w1_q = shuffle_weight_a16w4(w1_q, 16, True).contiguous()
    w1_scale = shuffle_scale_a16w4(w1_scale, experts, True).contiguous()

    w2_q, w2_scale = quantize(w2_shard, quant_dtype=dtypes.fp4x2)
    del w2_shard
    w2_q = w2_q.view(experts, model_dim, inter_shard // 2)
    w2_q = shuffle_weight_a16w4(w2_q, 16, False).contiguous()
    w2_scale = shuffle_scale_a16w4(w2_scale, experts, False).contiguous()
    torch.cuda.empty_cache()
    return w1_q, w1_scale, w2_q, w2_scale


class TpGroupShim:
    """Minimal ``tp_group`` for ``comm_fused_moe_host``.

    The comm-fused host runtime only needs the rank/world/device triple plus an
    object broadcast to hand out the MORI communicator id, so the bench uses the
    default process group directly instead of standing up
    ``aiter.dist.parallel_state``.
    """

    def __init__(self, rank, world, device):
        self.rank_in_group = rank
        self.world_size = world
        self.device = device

    def broadcast_object(self, obj=None, src=0):
        payload = [obj]
        dist.broadcast_object_list(payload, src=src)
        return payload[0]


def capture(body):
    barrier()
    body()
    barrier()
    graph = torch.cuda.CUDAGraph()
    stream = torch.cuda.Stream()
    with torch.cuda.graph(graph, stream=stream):
        body()
    for _ in range(5):
        graph.replay()
    barrier()
    return graph


def _fmt_ms(ms):
    return f"{ms[0]:.4f}/{ms[1]:.4f}ms"


def time_graph(graph, iters, device):
    barrier()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        graph.replay()
    end.record()
    torch.cuda.synchronize()
    local_ms = start.elapsed_time(end) / iters
    mean = torch.tensor(local_ms, dtype=torch.float64, device=device)
    maximum = mean.clone()
    dist.all_reduce(mean, op=dist.ReduceOp.SUM)
    dist.all_reduce(maximum, op=dist.ReduceOp.MAX)
    return float(mean.item() / dist.get_world_size()), float(maximum.item())


def profile_graph(graph, name, rank, out_dir, replays=3):
    barrier()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        dist.barrier()
        for _ in range(replays):
            graph.replay()
        torch.cuda.synchronize()
    path = Path(out_dir)
    path.mkdir(parents=True, exist_ok=True)
    prof.export_chrome_trace(str(path / f"{name}_rank{rank}.json"))
    barrier()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=8192)
    parser.add_argument("--rank-tokens", default="")
    parser.add_argument("--config-tokens", type=int, default=0)
    parser.add_argument("--mtpr", type=int, default=8192)
    parser.add_argument("--model-dim", type=int, default=MODEL_DIM)
    parser.add_argument("--inter-dim", type=int, default=INTER_DIM)
    parser.add_argument("--experts", type=int, default=EXPERTS)
    parser.add_argument("--topk", type=int, default=TOPK)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument(
        "--route",
        choices=(
            "uniform",
            "hot-rank0",
            "rank-balanced-hot",
            "rank-balanced-last",
            "rank-mixed-skew",
        ),
        default="uniform",
    )
    parser.add_argument("--hot-bias", type=float, default=0.6)
    parser.add_argument("--stage2-strided", action="store_true")
    parser.add_argument("--stage2-persist-cu", type=int, default=0)
    parser.add_argument("--stage2-skew-cu", type=int, default=0)
    parser.add_argument("--disable-stage2-skew", action="store_true")
    parser.add_argument("--stage1-payload-chunk-rows", type=int, default=0)
    parser.add_argument("--stage1-tile-ready", action="store_true")
    parser.add_argument("--disable-stage1-tile-ready", action="store_true")
    parser.add_argument("--stage1-internal-grouping", action="store_true")
    parser.add_argument("--stage1-work-shards", type=int, default=0)
    parser.add_argument("--stage1-dispatch-cu", type=int, default=0)
    parser.add_argument("--stage1-grid-mult", type=int, default=0)
    parser.add_argument("--stage1-b-nt", type=int, default=-1)
    parser.add_argument("--stage1-tile-resource", action="store_true")
    parser.add_argument("--check-variant", action="store_true")
    parser.add_argument("--profile-dir", default="")
    parser.add_argument("--mega-only", action="store_true")
    parser.add_argument("--perf-guard", action="store_true")
    parser.add_argument(
        "--tp",
        action="store_true",
        help=(
            "TP accounting: NCCL AG(x)+fused_moe baseline vs two-launch "
            "gather+GEMM1 e2e (local quant, ids AG, sort, bulk P2P gather, "
            "token-id GEMM1, GEMM2+RS). Accuracy vs Mori is checked on the "
            "two-launch e2e path at --tokens."
        ),
    )
    parser.add_argument(
        "--tp-sweep",
        default="",
        help=(
            "Comma-separated per-rank token counts for the TP breakdown "
            "(e.g. 64,256,512,2048). Empty uses --tokens only. --mtpr must "
            "cover the largest value."
        ),
    )
    args = parser.parse_args()

    rank, world, device = setup_dist()
    if world != 8:
        raise ValueError("This comparison requires eight ranks")
    if args.experts % world:
        raise ValueError(f"experts={args.experts} must be divisible by world={world}")
    rank_tokens = [int(value) for value in args.rank_tokens.split(",") if value]
    if rank_tokens and len(rank_tokens) != world:
        raise ValueError(f"--rank-tokens requires {world} comma-separated values")
    tokens = rank_tokens[rank] if rank_tokens else args.tokens
    tp_sweep = (
        [int(value) for value in args.tp_sweep.split(",") if value]
        if args.tp_sweep
        else [tokens]
    )
    if args.tp:
        if not tp_sweep:
            raise ValueError("--tp-sweep is empty")
        if min(tp_sweep) <= 0:
            raise ValueError("--tp-sweep values must be positive")
        if max(tp_sweep) > args.mtpr:
            raise ValueError(
                f"--mtpr={args.mtpr} must cover max(--tp-sweep)={max(tp_sweep)}"
            )
    local_experts = args.experts // world
    x, route_weights, ids = make_inputs(
        tokens,
        rank,
        world,
        args.model_dim,
        args.experts,
        args.topk,
        args.route,
        args.hot_bias,
        device,
    )
    route_counts = torch.zeros(world, dtype=torch.int64, device=device)
    route_counts.scatter_add_(
        0,
        ids.flatten().to(torch.int64) // local_experts,
        torch.ones_like(ids.flatten(), dtype=torch.int64),
    )
    dist.all_reduce(route_counts, op=dist.ReduceOp.SUM)
    expert_counts = torch.bincount(
        ids.flatten().to(torch.int64), minlength=args.experts
    )
    dist.all_reduce(expert_counts, op=dist.ReduceOp.SUM)
    w1, w1_scale, w2, w2_scale = make_weights(
        local_experts, args.model_dim, args.inter_dim, rank, device
    )

    mega = MegaMoEV2(
        rank=rank,
        world_size=world,
        model_dim=args.model_dim,
        inter_dim=args.inter_dim,
        experts=args.experts,
        topk=args.topk,
        quant="a8w4",
        w1=w1,
        w1_scale=w1_scale,
        w2=w2,
        w2_scale=w2_scale,
        max_tok_per_rank=args.mtpr,
        swiglu_limit=SWIGLU_LIMIT,
    )
    default_select_config = mega._select_config
    variant_select_config = None
    if (
        args.stage2_strided
        or args.stage2_persist_cu
        or args.stage2_skew_cu
        or args.disable_stage2_skew
        or args.stage1_payload_chunk_rows
        or args.stage1_tile_ready
        or args.disable_stage1_tile_ready
        or args.stage1_internal_grouping
        or args.stage1_work_shards
        or args.stage1_dispatch_cu
        or args.stage1_grid_mult
        or args.stage1_b_nt >= 0
        or args.stage1_tile_resource
        or args.config_tokens
    ):

        def select_strided_config(tokens):
            config = default_select_config(args.config_tokens or tokens)
            stage1 = config.stage1
            stage2 = config.stage2
            if args.stage1_payload_chunk_rows:
                stage1 = replace(
                    stage1, payload_chunk_rows=args.stage1_payload_chunk_rows
                )
            if args.stage1_tile_ready:
                stage1 = replace(stage1, payload_tile_ready=True)
            if args.disable_stage1_tile_ready:
                stage1 = replace(stage1, payload_tile_ready=False)
            if args.stage1_internal_grouping:
                stage1 = replace(
                    stage1, external_grouping=False, external_counting=False
                )
            if args.stage1_work_shards:
                stage1 = replace(stage1, work_shards=args.stage1_work_shards)
            if args.stage1_dispatch_cu:
                stage1 = replace(stage1, num_dispatch_cu=args.stage1_dispatch_cu)
            if args.stage1_grid_mult:
                stage1 = replace(stage1, grid_mult=args.stage1_grid_mult)
            if args.stage1_b_nt >= 0:
                stage1 = replace(stage1, b_nt=args.stage1_b_nt)
            if args.stage1_tile_resource:
                stage1 = replace(stage1, use_tile_resource=True)
            if (
                args.stage2_strided
                or args.stage2_persist_cu
                or args.stage2_skew_cu
                or args.disable_stage2_skew
            ):
                stage2 = replace(
                    stage2,
                    persist_strided=args.stage2_strided,
                    persist_cu=args.stage2_persist_cu or stage2.persist_cu,
                    skew_cu=(
                        0
                        if args.disable_stage2_skew
                        else args.stage2_skew_cu or stage2.skew_cu
                    ),
                )
            config = replace(
                config,
                stage1=stage1,
                stage2=stage2,
            )
            mega._active_config = config
            return config

        variant_select_config = select_strided_config
        mega._select_config = variant_select_config

    mori_cfg = mori.ops.EpDispatchCombineConfig(
        data_type=torch.bfloat16,
        rank=rank,
        world_size=world,
        hidden_dim=args.model_dim,
        scale_dim=0,
        scale_type_size=0,
        max_token_type_size=torch.bfloat16.itemsize,
        max_num_inp_token_per_rank=args.mtpr,
        num_experts_per_rank=local_experts,
        num_experts_per_token=args.topk,
        warp_num_per_block=16,
        block_num=128,
        gpu_per_node=world,
    )
    mori_op = mori.ops.EpDispatchCombineOp(mori_cfg)
    expert_mask = torch.zeros(args.experts, dtype=torch.int32, device=device)
    expert_mask[rank * local_experts : (rank + 1) * local_experts] = 1
    holders = {}

    def mori_body():
        dispatched, recv_weights, _, recv_ids, recv_tokens = mori_op.dispatch(
            x, route_weights, None, ids
        )
        local_out = fused_moe(
            dispatched,
            w1,
            w2,
            recv_weights,
            recv_ids,
            expert_mask,
            quant_type=aiter.QuantType.per_1x32,
            num_local_tokens=recv_tokens,
            w1_scale=w1_scale,
            w2_scale=w2_scale,
            a1_scale=None,
            dtype=torch.bfloat16,
            swiglu_limit=SWIGLU_LIMIT,
            gate_mode=GateMode.INTERLEAVE.value,
        )
        holders["mori"] = mori_op.combine(local_out, None, ids)[0]

    def mega_body():
        holders["mega"] = mega(x, route_weights, ids)

    mori_graph = None if args.mega_only else capture(mori_body)
    print(f"[STEP] rank={rank} mori-capture-done", flush=True)
    mega_graph = capture(mega_body)
    print(f"[STEP] rank={rank} mega-capture-done", flush=True)
    mori_ms = (
        (float("nan"), float("nan"))
        if mori_graph is None
        else time_graph(mori_graph, args.iters, device)
    )
    mega_ms = time_graph(mega_graph, args.iters, device)

    x_q, x_scale = mega.quantize(x)

    def mega_stage1():
        mega._run_fused_stage1(x_q, route_weights, x_scale, ids)

    stage1_graph = capture(mega_stage1)
    print(f"[STEP] rank={rank} stage1-capture-done", flush=True)
    mega_stage1()
    barrier()

    def mega_stage2():
        holders["stage2"] = mega._run_stage2(tokens, None, True, mega._active_config)

    stage2_graph = capture(mega_stage2)
    print(f"[STEP] rank={rank} stage2-capture-done", flush=True)
    stage1_ms = time_graph(stage1_graph, args.iters, device)
    mega_stage1()
    barrier()
    stage2_ms = time_graph(stage2_graph, args.iters, device)

    rel_l2 = None
    if args.check_variant:
        if variant_select_config is None:
            raise ValueError("--check-variant requires a Stage2 variant")
        mega._select_config = default_select_config
        reference = mega(x, route_weights, ids).clone()
        barrier()
        mega._select_config = variant_select_config
        candidate = mega(x, route_weights, ids).clone()
        barrier()
        rel_l2 = (
            candidate.float() - reference.float()
        ).norm() / reference.float().norm()
        dist.all_reduce(rel_l2, op=dist.ReduceOp.MAX)

    tp_rel_l2 = None
    tp_ms = (float("nan"), float("nan"))
    tp_stage1_ms = (float("nan"), float("nan"))
    tp_nccl_ms = (float("nan"), float("nan"))
    tp_rows = []
    tp_comm_fused = os.environ.get("AITER_TP_COMM_FUSED", "0") == "1"
    if args.tp:
        if rank_tokens:
            raise ValueError(
                "--tp requires equal per-rank tokens (no --rank-tokens) in this "
                "first version"
            )
        if mori_graph is None:
            raise ValueError("--tp accuracy check requires Mori (no --mega-only)")
        if tp_comm_fused and tp_sweep != [tokens]:
            raise ValueError("AITER_TP_COMM_FUSED is not compatible with --tp-sweep")
        from functools import partial

        import flydsl.expr as fx

        from aiter.fused_moe import (
            _flydsl_v2_stage2_wrapper,
            _fused_moe_impl,
            get_padded_M,
            moe_sorting,
        )
        from aiter.ops.flydsl.kernels.mega_moe.quant import per_1x32_mx_quant
        from aiter.ops.flydsl.kernels.mega_moe.tp_incremental_push import (
            TpIncrementalWorkspace,
            compile_tp_incremental_push,
            run_tp_incremental_push,
        )
        from aiter.ops.flydsl.kernels.mega_moe.tp_incremental_schedule import (
            expected_token_counts,
            make_tile_row_base,
            publish_order_from_topk,
        )
        from aiter.ops.flydsl.kernels.mega_moe.tp_incremental_stage1 import (
            compile_tp_incremental_fused,
            run_tp_incremental_fused,
            run_tp_two_launch_stage1,
        )
        from aiter.ops.flydsl.kernels.mega_moe.tp_token_gemm1 import (
            compile_tp_token_gemm1,
            gemm1_token_kernel,
        )
        from aiter.ops.flydsl.moe_kernels import (
            build_flydslv2_gemm2_name,
            pick_flydsl_stage2_tile_k,
        )

        w1_tp, w1_scale_tp, w2_tp, w2_scale_tp = make_weights_tp(
            args.experts, args.model_dim, args.inter_dim, world, rank, device
        )
        inter_shard = args.inter_dim // world
        sort_block_m = 32
        fused_kwargs = dict(
            model_dim=args.model_dim,
            inter_dim=inter_shard,
            expert_offset=0,
            sort_block_m=sort_block_m,
            tile_n=256,
            tile_k=256,
            num_cu=torch.cuda.get_device_properties(device).multi_processor_count,
            swiglu_limit=SWIGLU_LIMIT,
        )
        print(f"[STEP] rank={rank} tp-incremental-compile", flush=True)
        compile_tp_incremental_fused(
            model_dim=args.model_dim,
            inter_dim=inter_shard,
            npes=world,
            topk=args.topk,
            num_producers=8,
            swiglu_limit=SWIGLU_LIMIT,
        )
        compile_tp_incremental_push(
            npes=world,
            model_dim=args.model_dim,
            row_major=False,
            num_producers=32,
            num_waves=4,
        )
        compile_tp_token_gemm1(
            model_dim=args.model_dim,
            inter_dim=inter_shard,
            expert_offset=0,
            sort_block_m=sort_block_m,
            tile_n=256,
            tile_k=256,
            swiglu_limit=SWIGLU_LIMIT,
        )
        workspace = TpIncrementalWorkspace(
            rank=rank,
            npes=world,
            max_m_local=max(tp_sweep),
            model_dim=args.model_dim,
            num_experts=args.experts,
            device=device,
        )
        handshake_src = torch.zeros(1, dtype=torch.int32, device=device)
        handshake_dst = torch.empty(world, dtype=torch.int32, device=device)
        stage2_kn = build_flydslv2_gemm2_name(
            "fp8",
            "fp4",
            "bf16",
            tm=sort_block_m,
            tn=256,
            tk=pick_flydsl_stage2_tile_k(inter_shard),
            epilog="atomic",
            persist=False,
            use_nt=True,
            sbm=sort_block_m,
        )

        def tp_metadata_transform(metadata):
            return replace(
                metadata,
                skip_inter_quant=True,
                fuse_quant="fp8",
                block_m=sort_block_m,
                stage2=partial(
                    _flydsl_v2_stage2_wrapper,
                    kernelName=stage2_kn,
                    model_dim=args.model_dim,
                    inter_dim=inter_shard,
                    num_experts=args.experts,
                ),
            )

        fused_runner = None
        if tp_comm_fused:
            from aiter.ops.flydsl.comm_fused_moe_host import (
                create_flydsl_comm_fused_runners,
            )

            bucket = int(get_padded_M(world * tokens))
            if bucket != world * tokens:
                raise ValueError(
                    f"AITER_TP_COMM_FUSED needs an exact padded-M bucket, but "
                    f"world_tokens={world * tokens} pads to {bucket}"
                )
            runners = create_flydsl_comm_fused_runners(
                tp_group=TpGroupShim(rank, world, device),
                model_dim=args.model_dim,
                inter_dim=inter_shard,
                experts=args.experts,
                topk=args.topk,
            )
            if bucket not in runners:
                raise ValueError(f"no comm-fused config for M={bucket}")
            fused_runner = runners[bucket]
            print(f"[STEP] rank={rank} tp-comm-fused-runner m={bucket}", flush=True)

        def _time(body, label):
            print(f"[STEP] rank={rank} {label}", flush=True)
            graph = capture(body)
            print(f"[STEP] rank={rank} {label}-done", flush=True)
            ms = time_graph(graph, args.iters, device)
            del graph
            return ms

        def _fx_stream():
            return fx.Stream(torch.cuda.current_stream().cuda_stream)

        def measure_tp_size(m_local, x_loc, wts_loc, ids_loc):
            world_tokens = world * m_local
            x_loc = x_loc.contiguous()
            wts_loc = wts_loc.contiguous()
            ids_loc = ids_loc.contiguous()
            x_g = torch.empty(
                (world_tokens, args.model_dim), dtype=torch.bfloat16, device=device
            )
            route_weights_g = torch.empty(
                (world_tokens, args.topk), dtype=torch.float32, device=device
            )
            ids_g = torch.empty(
                (world_tokens, args.topk), dtype=torch.int32, device=device
            )
            tp_local_out = torch.empty(
                (m_local, args.model_dim), dtype=torch.bfloat16, device=device
            )
            tp_nccl_local = torch.empty_like(tp_local_out)
            dist.all_gather_into_tensor(route_weights_g, wts_loc)
            dist.all_gather_into_tensor(ids_g, ids_loc)
            expected = expected_token_counts(ids_g, args.experts).to(
                dtype=torch.int32, device=device
            )
            sorted_ids0, _, sorted_expert_ids0, num_valid_ids0, _ = moe_sorting(
                ids_g,
                route_weights_g,
                args.experts,
                args.model_dim,
                torch.bfloat16,
                sort_block_m,
            )
            num_valid = int(num_valid_ids0[0].item())
            tile_row_base = make_tile_row_base(num_valid, sort_block_m, device=device)
            publish_order = (
                publish_order_from_topk(ids_loc).to(torch.int32).contiguous()
            )
            n_m = num_valid // sort_block_m
            expert_ids0 = sorted_expert_ids0[:n_m].contiguous()
            sorted_rows = max(
                sorted_ids0.shape[0], sorted_expert_ids0.shape[0] * sort_block_m
            )
            scale_cols = (inter_shard // 32 + 7) // 8 * 8
            padded_rows = (sorted_rows + 255) // 256 * 256
            gemm1_out = torch.zeros(
                (sorted_rows, inter_shard), dtype=torch.float8_e4m3fn, device=device
            )
            gemm1_scale = torch.zeros(
                (padded_rows, scale_cols), dtype=torch.uint8, device=device
            )
            gemm1_scale_e8m0 = gemm1_scale.view(dtypes.fp8_e8m0)
            x_fp8 = torch.empty(
                (m_local, args.model_dim), dtype=torch.float8_e4m3fn, device=device
            )
            x_scale = torch.empty(
                (m_local, args.model_dim // 32), dtype=torch.uint8, device=device
            )
            rx = workspace.rx[: world_tokens + 1]
            rx_scale = workspace.rx_scale[: world_tokens + 1]

            def _quant_local_x():
                return per_1x32_mx_quant(
                    x_loc, quant_mode="fp8", out=x_fp8, scale=x_scale
                )

            def _run_fused_stage1():
                return run_tp_incremental_fused(
                    workspace,
                    x_fp8,
                    x_scale,
                    ids_loc,
                    gemm1_out,
                    w1_tp,
                    w1_scale_tp,
                    tile_row_base,
                    expert_ids0,
                    sorted_ids0,
                    gemm1_scale,
                    expected,
                    num_valid,
                    world_tokens,
                    m_local,
                    _fx_stream(),
                    publish_order=publish_order,
                    **fused_kwargs,
                )

            def _run_push():
                return run_tp_incremental_push(
                    workspace,
                    x_fp8,
                    x_scale,
                    ids_loc,
                    m_local,
                    _fx_stream(),
                    publish_order=publish_order,
                )

            def _run_token_gemm1():
                return gemm1_token_kernel(
                    gemm1_out,
                    rx,
                    w1_tp,
                    rx_scale,
                    w1_scale_tp,
                    tile_row_base,
                    expert_ids0,
                    sorted_ids0,
                    gemm1_scale,
                    num_valid,
                    world_tokens,
                    _fx_stream(),
                    **fused_kwargs,
                )

            def tp_stage1_override(*, sorted_ids, sorted_expert_ids, **_kwargs):
                _quant_local_x()
                run_tp_two_launch_stage1(
                    workspace,
                    x_fp8,
                    x_scale,
                    ids_loc,
                    gemm1_out,
                    w1_tp,
                    w1_scale_tp,
                    tile_row_base,
                    sorted_expert_ids[:n_m].contiguous(),
                    sorted_ids,
                    gemm1_scale,
                    num_valid,
                    world_tokens,
                    m_local,
                    _fx_stream(),
                    **fused_kwargs,
                )
                return gemm1_out, gemm1_scale_e8m0

            def tp_stage2_override(*, ordinary_stage2, stage2_args, stage2_kwargs):
                return fused_runner(
                    stage2_args=stage2_args,
                    stage2_kwargs=stage2_kwargs,
                    shared_partial=stage2_args[6],
                    ordinary_stage2=ordinary_stage2,
                    all_gather=False,
                )

            def _tp_fused_moe():
                return _fused_moe_impl(
                    x_g,
                    w1_tp,
                    w2_tp,
                    route_weights_g,
                    ids_g,
                    quant_type=aiter.QuantType.per_1x32.value,
                    w1_scale=w1_scale_tp,
                    w2_scale=w2_scale_tp,
                    a1_scale=None,
                    dtype=torch.bfloat16,
                    swiglu_limit=SWIGLU_LIMIT,
                    gate_mode=GateMode.INTERLEAVE.value,
                    _metadata_transform=tp_metadata_transform,
                    _stage1_override=tp_stage1_override,
                    _stage2_override=(
                        tp_stage2_override if fused_runner is not None else None
                    ),
                )

            def tp_nccl_body():
                dist.all_gather_into_tensor(x_g, x_loc)
                dist.all_gather_into_tensor(route_weights_g, wts_loc)
                dist.all_gather_into_tensor(ids_g, ids_loc)
                tp_out = fused_moe(
                    x_g,
                    w1_tp,
                    w2_tp,
                    route_weights_g,
                    ids_g,
                    quant_type=aiter.QuantType.per_1x32,
                    w1_scale=w1_scale_tp,
                    w2_scale=w2_scale_tp,
                    a1_scale=None,
                    dtype=torch.bfloat16,
                    swiglu_limit=SWIGLU_LIMIT,
                    gate_mode=GateMode.INTERLEAVE.value,
                )
                dist.reduce_scatter_tensor(tp_nccl_local, tp_out, op=dist.ReduceOp.SUM)

            def tp_body():
                workspace.zero_handshake()
                dist.all_gather_into_tensor(route_weights_g, wts_loc)
                dist.all_gather_into_tensor(ids_g, ids_loc)
                tp_out = _tp_fused_moe()
                if fused_runner is not None:
                    holders["tp"] = tp_out
                    return
                dist.reduce_scatter_tensor(tp_local_out, tp_out, op=dist.ReduceOp.SUM)
                holders["tp"] = tp_local_out

            def tp_stage1_body():
                workspace.zero_handshake()
                dist.all_gather_into_tensor(route_weights_g, wts_loc)
                dist.all_gather_into_tensor(ids_g, ids_loc)
                _quant_local_x()
                _run_push()
                _run_token_gemm1()

            def tp_quant_body():
                _quant_local_x()

            def tp_ag_meta_body():
                dist.all_gather_into_tensor(route_weights_g, wts_loc)
                dist.all_gather_into_tensor(ids_g, ids_loc)

            def _handshake_sync():
                workspace.zero_handshake()
                dist.all_gather_into_tensor(handshake_dst, handshake_src)

            def tp_fused_k_body():
                _handshake_sync()
                _run_fused_stage1()

            def tp_gather_k_body():
                _handshake_sync()
                _run_push()

            def tp_two_launch_body():
                _handshake_sync()
                _run_push()
                _run_token_gemm1()

            def tp_gemm1_k_body():
                _run_token_gemm1()

            tag = f"m{m_local}"
            nccl_e2e = _time(tp_nccl_body, f"tp-nccl-e2e-{tag}")
            two_e2e = _time(tp_body, f"tp-two-e2e-{tag}")
            tp_saved = holders["tp"].clone() if m_local == tokens else None
            lumped_s1 = _time(tp_stage1_body, f"tp-lumped-s1-{tag}")
            quant = _time(tp_quant_body, f"tp-quant-{tag}")
            ag_meta = _time(tp_ag_meta_body, f"tp-ag-meta-{tag}")
            _quant_local_x()
            fused_k = _time(tp_fused_k_body, f"tp-fused-k-{tag}")
            gather_k = _time(tp_gather_k_body, f"tp-gather-k-{tag}")
            two_launch = _time(tp_two_launch_body, f"tp-two-launch-{tag}")
            gemm1_k = _time(tp_gemm1_k_body, f"tp-gemm1-k-{tag}")
            row = {
                "tokens": m_local,
                "nccl_e2e": nccl_e2e,
                "two_e2e": two_e2e,
                "lumped_s1": lumped_s1,
                "quant": quant,
                "ag_meta": ag_meta,
                "fused_k": fused_k,
                "gather_k": gather_k,
                "two_launch": two_launch,
                "gemm1_k": gemm1_k,
            }
            if rank == 0:
                print(
                    f"[TP-BREAKDOWN] tokens={m_local} "
                    f"nccl_e2e={_fmt_ms(nccl_e2e)} two_e2e={_fmt_ms(two_e2e)} "
                    f"lumped_s1={_fmt_ms(lumped_s1)} quant={_fmt_ms(quant)} "
                    f"ag_meta={_fmt_ms(ag_meta)} fused_k={_fmt_ms(fused_k)} "
                    f"gather_k={_fmt_ms(gather_k)} two_launch={_fmt_ms(two_launch)} "
                    f"gemm1_k={_fmt_ms(gemm1_k)}",
                    flush=True,
                )
            return row, tp_saved

        for m_local in tp_sweep:
            if m_local == tokens:
                x_loc, wts_loc, ids_loc = x, route_weights, ids
            else:
                x_loc, wts_loc, ids_loc = make_inputs(
                    m_local,
                    rank,
                    world,
                    args.model_dim,
                    args.experts,
                    args.topk,
                    args.route,
                    args.hot_bias,
                    device,
                )
            row, tp_saved = measure_tp_size(m_local, x_loc, wts_loc, ids_loc)
            tp_rows.append(row)
            if m_local == tokens:
                tp_ms = row["two_e2e"]
                tp_stage1_ms = row["lumped_s1"]
                tp_nccl_ms = row["nccl_e2e"]
                barrier()
                mori_graph.replay()
                torch.cuda.synchronize()
                mori_ref = holders["mori"][:tokens]
                tp_rel_l2 = (
                    tp_saved.float() - mori_ref.float()
                ).norm() / mori_ref.float().norm()
                dist.all_reduce(tp_rel_l2, op=dist.ReduceOp.MAX)

    if args.profile_dir:
        if mori_graph is not None:
            profile_graph(mori_graph, f"mori_{args.route}", rank, args.profile_dir)
        profile_graph(mega_graph, f"mega_{args.route}", rank, args.profile_dir)
    speedup = (mori_ms[1] / mega_ms[1] - 1.0) * 100.0
    guard_floor = None
    if args.perf_guard:
        if args.mega_only or rank_tokens or args.mtpr != 8192:
            raise ValueError(
                "--perf-guard requires Mori, equal rank tokens, and mtpr=8192"
            )
        if (args.model_dim, args.inter_dim, args.experts, args.topk) != (
            MODEL_DIM,
            INTER_DIM,
            EXPERTS,
            TOPK,
        ):
            raise ValueError("--perf-guard requires the v4_pro shape")
        guard_floor = PERF_GUARD_MIN_SPEEDUP.get((args.tokens, args.route))
        if guard_floor is None:
            raise ValueError(
                f"no performance guard for tokens={args.tokens}, route={args.route}"
            )
    guard_pass = guard_floor is None or speedup >= guard_floor
    if rank == 0:
        print(f"[ROUTES] per-destination-rank={route_counts.tolist()}", flush=True)
        print(
            f"[EXPERTS] active={(expert_counts > 0).sum().item()} max_routes={expert_counts.max().item()} "
            f"mean_routes={expert_counts.float().mean().item():.1f}",
            flush=True,
        )
        if rel_l2 is not None:
            print(
                f"[ACCURACY] variant_vs_default_rel_l2={rel_l2.item():.6e}", flush=True
            )
        if tp_rel_l2 is not None:
            print(f"[ACCURACY] tp_vs_mori_rel_l2={tp_rel_l2.item():.6e}", flush=True)
        if tp_rows:
            print(
                "[TP-TABLE] tokens nccl_e2e two_e2e lumped_s1 quant ag_meta "
                "fused_k gather_k two_launch gemm1_k",
                flush=True,
            )
            for row in tp_rows:
                print(
                    f"[TP-TABLE] {row['tokens']} "
                    f"{_fmt_ms(row['nccl_e2e'])} {_fmt_ms(row['two_e2e'])} "
                    f"{_fmt_ms(row['lumped_s1'])} {_fmt_ms(row['quant'])} "
                    f"{_fmt_ms(row['ag_meta'])} {_fmt_ms(row['fused_k'])} "
                    f"{_fmt_ms(row['gather_k'])} {_fmt_ms(row['two_launch'])} "
                    f"{_fmt_ms(row['gemm1_k'])}",
                    flush=True,
                )
        print(
            f"[RESULT] route={args.route} hot_bias={args.hot_bias} tokens={tokens} "
            f"rank_tokens={rank_tokens or 'same'} mtpr={args.mtpr} "
            f"shape={args.model_dim}x{args.inter_dim} epr={local_experts} topk={args.topk} "
            f"mori_e2e={mori_ms[0]:.4f}/{mori_ms[1]:.4f}ms "
            f"mega_e2e={mega_ms[0]:.4f}/{mega_ms[1]:.4f}ms speedup={speedup:.2f}% "
            f"stage1={stage1_ms[0]:.4f}/{stage1_ms[1]:.4f}ms "
            f"stage2_combine={stage2_ms[0]:.4f}/{stage2_ms[1]:.4f}ms "
            f"tp_nccl_e2e={tp_nccl_ms[0]:.4f}/{tp_nccl_ms[1]:.4f}ms "
            f"tp_e2e={tp_ms[0]:.4f}/{tp_ms[1]:.4f}ms "
            f"tp_stage1={tp_stage1_ms[0]:.4f}/{tp_stage1_ms[1]:.4f}ms "
            f"tp_path={'comm_fused_rs' if tp_comm_fused else 'two_launch+nccl_rs'} rank-mean/max",
            flush=True,
        )
        if guard_floor is not None:
            status = "PASS" if guard_pass else "FAIL"
            print(
                f"[PERF-GUARD] {status} speedup={speedup:.2f}% minimum={guard_floor:.2f}%",
                flush=True,
            )
    if args.tp:
        # MORI shmem_finalize hangs after this bench; timings are already printed.
        barrier()
        os._exit(0 if guard_pass else 1)
    ms.shmem_finalize()
    dist.destroy_process_group()
    if not guard_pass:
        raise AssertionError(f"speedup {speedup:.2f}% is below {guard_floor:.2f}%")


if __name__ == "__main__":
    main()
