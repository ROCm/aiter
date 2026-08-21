# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Compare Mori EP and MegaMoEV2 with the same v4_pro A8W4 CUDA Graph workload."""

from __future__ import annotations

import argparse
import hashlib
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


def load_route_artifact(path: str, rank: int, device: torch.device):
    artifact_dir = Path(path)
    matches = sorted(artifact_dir.glob(f"*rank{rank}.pt"))
    if len(matches) != 1:
        raise ValueError(
            f"expected one rank{rank} route artifact in {artifact_dir}, got {matches}"
        )
    payload = torch.load(matches[0], map_location="cpu", weights_only=False)
    ids = payload["expert_ids"].to(device=device, dtype=torch.int32).contiguous()
    weights = (
        payload["route_weights"].to(device=device, dtype=torch.float32).contiguous()
    )
    if ids.ndim != 2 or weights.shape != ids.shape:
        raise ValueError(
            f"invalid route artifact shapes ids={tuple(ids.shape)} weights={tuple(weights.shape)}"
        )
    return weights, ids, payload.get("metadata", {})


def tensor_hash(tensor: torch.Tensor) -> str:
    raw = tensor.detach().contiguous().view(torch.int16).cpu().numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


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
    return (
        float(mean.item() / dist.get_world_size()),
        float(maximum.item()),
        float(local_ms),
    )


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
    parser.add_argument("--print-rank-times", action="store_true")
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
    parser.add_argument("--stage2-block-m", type=int, default=0)
    parser.add_argument("--stage2-block-n", type=int, default=0)
    parser.add_argument("--stage2-block-k", type=int, default=0)
    parser.add_argument("--disable-stage2-skew", action="store_true")
    parser.add_argument("--stage1-payload-chunk-rows", type=int, default=0)
    parser.add_argument("--stage1-work-shards", type=int, default=0)
    parser.add_argument("--stage1-prepare-cu", type=int, default=0)
    parser.add_argument("--stage1-prepare-quant-cu", type=int, default=0)
    parser.add_argument("--stage1-dispatch-cu", type=int, default=0)
    parser.add_argument("--stage1-grid-mult", type=int, default=0)
    parser.add_argument("--stage1-b-nt", type=int, default=-1)
    parser.add_argument("--stage1-tile-resource", action="store_true")
    parser.add_argument(
        "--stage1-fanout-masks",
        default="",
        help="comma-separated local-expert masks, one per destination",
    )
    parser.add_argument("--route-artifacts", default="")
    parser.add_argument("--output-hash", action="store_true")
    parser.add_argument("--check-variant", action="store_true")
    parser.add_argument("--profile-dir", default="")
    parser.add_argument("--mega-only", action="store_true")
    parser.add_argument("--perf-guard", action="store_true")
    parser.add_argument(
        "--separate-quant-prepare",
        action="store_true",
        help="bench-only A/B: launch the stock quant kernel before compact prepare",
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
    artifact_metadata = None
    if args.route_artifacts:
        route_weights, ids, artifact_metadata = load_route_artifact(
            args.route_artifacts, rank, device
        )
        tokens = int(ids.shape[0])
        if ids.shape[1] != args.topk:
            raise ValueError(
                f"artifact topk={ids.shape[1]} does not match --topk={args.topk}"
            )
        if x.shape[0] != tokens:
            x, _, _ = make_inputs(
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

    fanout_masks = tuple(
        int(item, 0) for item in args.stage1_fanout_masks.split(",") if item
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
        fanout_masks=fanout_masks,
    )
    default_select_config = mega._select_config
    variant_select_config = None
    if (
        args.stage2_strided
        or args.stage2_persist_cu
        or args.stage2_skew_cu
        or args.stage2_block_m
        or args.stage2_block_n
        or args.stage2_block_k
        or args.disable_stage2_skew
        or args.stage1_payload_chunk_rows
        or args.stage1_work_shards
        or args.stage1_prepare_cu
        or args.stage1_prepare_quant_cu
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
            if args.stage1_work_shards:
                stage1 = replace(stage1, work_shards=args.stage1_work_shards)
            if args.stage1_prepare_cu:
                stage1 = replace(stage1, prepare_cu=args.stage1_prepare_cu)
            if args.stage1_prepare_quant_cu:
                stage1 = replace(stage1, prepare_quant_cu=args.stage1_prepare_quant_cu)
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
                or args.stage2_block_m
                or args.stage2_block_n
                or args.stage2_block_k
                or args.disable_stage2_skew
            ):
                stage2 = replace(
                    stage2,
                    block_m=args.stage2_block_m or stage2.block_m,
                    block_n=args.stage2_block_n or stage2.block_n,
                    block_k=args.stage2_block_k or stage2.block_k,
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
        if args.separate_quant_prepare:
            x_q_e2e, x_scale_e2e = mega.quantize(x)
            holders["mega"] = mega._run_joint(
                x_q_e2e,
                x_scale_e2e,
                route_weights,
                ids,
                tokens,
                None,
                True,
            )
        else:
            holders["mega"] = mega(x, route_weights, ids)

    mori_graph = None if args.mega_only else capture(mori_body)
    print(f"[STEP] rank={rank} mori-capture-done", flush=True)
    mega_graph = capture(mega_body)
    print(f"[STEP] rank={rank} mega-capture-done", flush=True)
    mori_ms = (
        (float("nan"), float("nan"), float("nan"))
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

    rank_times = None
    if args.print_rank_times:
        rank_times = [None] * world
        dist.all_gather_object(
            rank_times,
            {
                "mega": mega_ms[2],
                "stage1": stage1_ms[2],
                "stage2": stage2_ms[2],
            },
        )

    output_digests = None
    if args.output_hash:
        local_digest = tensor_hash(holders["mega"])
        output_digests = [None] * world
        dist.all_gather_object(output_digests, local_digest)

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
        if artifact_metadata is not None:
            print(
                f"[ARTIFACT] call={artifact_metadata.get('call_index')} "
                f"layer={artifact_metadata.get('layer_id')}",
                flush=True,
            )
        if output_digests is not None:
            print(f"[OUTPUT-HASH] per-rank={output_digests}", flush=True)
        if rank_times is not None:
            print(f"[RANK-TIMES] {rank_times}", flush=True)
        if rel_l2 is not None:
            print(
                f"[ACCURACY] variant_vs_default_rel_l2={rel_l2.item():.6e}", flush=True
            )
        print(
            f"[RESULT] route={args.route} hot_bias={args.hot_bias} tokens={tokens} "
            f"rank_tokens={rank_tokens or 'same'} mtpr={args.mtpr} "
            f"shape={args.model_dim}x{args.inter_dim} epr={local_experts} topk={args.topk} "
            f"mori_e2e={mori_ms[0]:.4f}/{mori_ms[1]:.4f}ms "
            f"mega_e2e={mega_ms[0]:.4f}/{mega_ms[1]:.4f}ms speedup={speedup:.2f}% "
            f"stage1={stage1_ms[0]:.4f}/{stage1_ms[1]:.4f}ms "
            f"stage2_combine={stage2_ms[0]:.4f}/{stage2_ms[1]:.4f}ms rank-mean/max",
            flush=True,
        )
        if guard_floor is not None:
            status = "PASS" if guard_pass else "FAIL"
            print(
                f"[PERF-GUARD] {status} speedup={speedup:.2f}% minimum={guard_floor:.2f}%",
                flush=True,
            )
    ms.shmem_finalize()
    dist.destroy_process_group()
    if not guard_pass:
        raise AssertionError(f"speedup {speedup:.2f}% is below {guard_floor:.2f}%")


if __name__ == "__main__":
    main()
