# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Layout and scheduling ablations for Flash EP2/EP4, with cold-cache timing."""

import argparse
import json
import os
import random
import re
from dataclasses import asdict, replace
from pathlib import Path

from bench_mega_moe_v2 import (
    MegaMoEV2,
    barrier,
    capture,
    dist,
    make_inputs,
    make_weights,
    mori,
    ms,
    profile_graph,
    run_online_mori,
    setup_dist,
    time_graph,
    torch,
)


def layouts(base):
    result = [base]
    for bm, bn, waves in (
        (16, 128, 2),
        (16, 256, 2),
        (16, 256, 4),
        (16, 512, 4),
        (32, 128, 4),
        (32, 64, 2),
        (32, 128, 2),
        (32, 256, 4),
        (32, 256, 8),
        (32, 512, 8),
        (64, 256, 4),
        (64, 128, 2),
        (64, 128, 4),
        (64, 256, 8),
        (64, 512, 8),
        (128, 256, 8),
        (128, 128, 4),
    ):
        result.append(
            replace(
                base,
                stage1=replace(
                    base.stage1,
                    sort_block_m=bm,
                    tile_n=bn,
                    num_waves=waves,
                    mfma_amajor=bm >= 32,
                ),
                stage2=replace(base.stage2, block_m=min(bm, 64)),
            )
        )
    return list(dict.fromkeys(result))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", default="1,8,32,64,128,256,512,1024")
    parser.add_argument("--mtpr", type=int, default=1024)
    parser.add_argument("--iters", type=int, default=15)
    parser.add_argument("--route", choices=("uniform", "hot-rank0"), default="uniform")
    parser.add_argument("--output", required=True)
    parser.add_argument("--schedule", action="store_true")
    parser.add_argument("--paired-k", action="store_true")
    parser.add_argument("--baseline", action="store_true")
    parser.add_argument("--seed-config-json")
    parser.add_argument("--kernel-knobs", action="store_true")
    parser.add_argument("--transport", action="store_true")
    parser.add_argument("--selected-only", action="store_true")
    parser.add_argument("--profile-dir", default="")
    args = parser.parse_args()
    seed_records = (
        json.loads(Path(args.seed_config_json).read_text())
        if args.seed_config_json
        else []
    )
    os.environ["MORI_EP_LAUNCH_CONFIG_MODE"] = "MANUAL"
    os.environ["AITER_BF16_FP8_MOE_BOUND"] = "0"
    rank, world, device = setup_dist()
    if world not in (2, 4):
        raise ValueError("Flash requires EP2 or EP4")
    w1, s1, w2, s2 = make_weights(384 // world, 5120, 2304, rank, device)
    mega = MegaMoEV2(
        rank=rank,
        world_size=world,
        model_dim=5120,
        inter_dim=2304,
        experts=384,
        topk=6,
        quant="a8w4",
        w1=w1,
        w1_scale=s1,
        w2=w2,
        w2_scale=s2,
        max_tok_per_rank=args.mtpr,
        swiglu_limit=10.0,
    )
    default_select = mega._select_config
    mori_op = None
    expert_mask = torch.zeros(385, dtype=torch.int32, device=device)
    expert_mask[rank * mega.epr : (rank + 1) * mega.epr] = 1
    if args.baseline:
        mori_op = mori.ops.EpDispatchCombineOp(
            mori.ops.EpDispatchCombineConfig(
                data_type=torch.bfloat16,
                rank=rank,
                world_size=world,
                hidden_dim=5120,
                scale_dim=0,
                scale_type_size=4,
                max_token_type_size=2,
                max_num_inp_token_per_rank=args.mtpr,
                num_experts_per_rank=mega.epr,
                num_experts_per_token=6,
                warp_num_per_block=16,
                block_num=80,
                rdma_block_num=0,
                kernel_type=mori.ops.EpDispatchCombineKernelType.IntraNode,
                gpu_per_node=world,
            )
        )
    cache = torch.zeros(512 * 1024 * 1024, dtype=torch.uint8, device=device)
    cache_graph = capture(lambda: cache.add_(1))
    records = []

    def tune_case(tokens):
        x, weights, ids = make_inputs(
            tokens, rank, world, 5120, 384, 6, args.route, 8.0, device
        )
        counts = torch.bincount(ids.flatten().long(), minlength=384)
        dist.all_reduce(counts)
        local_counts = counts[rank * mega.epr : (rank + 1) * mega.epr]
        received = (
            (ids[:, :, None] // mega.epr == torch.arange(world, device=device))
            .any(1)
            .sum(0)
        )
        dist.all_reduce(received)
        mega._select_config = default_select
        reference = mega(x, weights, ids, config_tokens=tokens).clone()
        barrier()
        base = default_select(tokens)
        if seed_records:
            from aiter.ops.flydsl.kernels.mega_moe.mega_moe_config import (
                MegaMoEConfig,
                Stage1Config,
                Stage2Config,
            )

            seed = min(
                (row for row in seed_records if row["source_m"] == tokens),
                key=lambda row: row["median_max_rank_ms"],
            )["config"]
            base = MegaMoEConfig(
                stage1=Stage1Config(**seed["stage1"]),
                stage2=Stage2Config(**seed["stage2"]),
                p2p_quant=seed["p2p_quant"],
                combine_blocks=seed.get("combine_blocks", 128),
                combine_waves=seed.get("combine_waves", 8),
            )
        baseline_ms = None
        if mori_op is not None:

            def baseline_body():
                return run_online_mori(
                    mori_op,
                    x,
                    weights,
                    ids,
                    expert_mask,
                    (w1, s1, w2, s2),
                    64,
                    4,
                    tokens * world,
                )

            baseline_output = baseline_body().clone()
            error = (reference.float() - baseline_output.float()).square().sum()
            norm = baseline_output.float().square().sum()
            dist.all_reduce(error)
            dist.all_reduce(norm)
            if float((error / norm).sqrt()) >= 0.1:
                raise AssertionError("MegaMoE vs MoRI+AITER exceeds relL2=0.1")
            baseline_graph = capture(baseline_body)
            baseline_ms = time_graph(
                baseline_graph, args.iters * 2, device, cache_graph
            )[1]
        if args.paired_k:
            base = replace(base, stage1=replace(base.stage1, pair_k=True))
        results = []
        active = base

        def select(_):
            mega._active_config = active
            return active

        mega._select_config = select

        def body():
            return mega(x, weights, ids, config_tokens=tokens)

        def evaluate(configs, phase):
            nonlocal active
            configs = list(dict.fromkeys(configs))
            random.Random(tokens + len(results)).shuffle(configs)
            graphs = []
            for config in configs:
                active = config
                candidate = body()
                error = (candidate.float() - reference.float()).square().sum()
                norm = reference.float().square().sum()
                dist.all_reduce(error)
                dist.all_reduce(norm)
                rel_l2 = float((error / norm).sqrt())
                if rel_l2 >= 0.001:
                    raise AssertionError(
                        f"EP{world} M{tokens} {config}: relL2={rel_l2}"
                    )
                graphs.append((config, capture(body), rel_l2))
            # Reverse the order on the second pass to share clock/thermal drift.
            timings = {config: [] for config, _, _ in graphs}
            for order in (graphs, list(reversed(graphs))):
                for config, graph, _ in order:
                    active = config
                    timings[config].append(
                        time_graph(graph, args.iters, device, cache_graph)[1]
                    )
            for config, _, rel_l2 in graphs:
                bm = config.stage1.sort_block_m
                tiles = (local_counts + bm - 1) // bm
                weight_bytes = float(tiles.sum()) * 3 * 5120 * 2304 * (0.5 + 1 / 32)
                lower_bytes = (
                    float((local_counts > 0).sum()) * 3 * 5120 * 2304 * (0.5 + 1 / 32)
                )
                ms_max = sum(timings[config]) / len(timings[config])
                record = {
                    "ep": world,
                    "source_m": tokens,
                    "mtpr": args.mtpr,
                    "fixed": mega._s1_fixed_slot,
                    "route": args.route,
                    "received_unique": received.tolist(),
                    "phase": phase,
                    "config": asdict(config),
                    "rel_l2": rel_l2,
                    "median_max_rank_ms": ms_max,
                    "passes": timings[config],
                    "baseline_ms": baseline_ms,
                    "rank0_row_utilization": float(local_counts.sum())
                    / max(1, int(tiles.sum()) * bm),
                    "rank0_ideal_weight_bytes": lower_bytes,
                    "rank0_tile_weight_bytes": weight_bytes,
                    "rank0_ideal_weight_gbps": lower_bytes / (ms_max * 1e6),
                }
                results.append((ms_max, config))
                if rank == 0:
                    records.append(record)
                    Path(args.output).write_text(json.dumps(records, indent=2) + "\n")
                    print(
                        f"[TUNE] EP{world} M={tokens} {phase} {ms_max * 1000:.1f}us "
                        f"W13={bm}x{config.stage1.tile_n}/{config.stage1.num_waves}w "
                        f"W2={config.stage2.block_m}x{config.stage2.block_n} "
                        f"rows={record['rank0_row_utilization']:.3f} relL2={rel_l2:.6f}",
                        flush=True,
                    )

        evaluate(
            [base] if args.transport or args.selected_only else layouts(base), "layout"
        )
        if args.selected_only:
            if args.profile_dir:
                active = base
                profile_graph(
                    capture(body),
                    f"mega_m{tokens}",
                    rank,
                    args.profile_dir,
                    cache_graph=cache_graph,
                )
            barrier()
            return
        if args.transport:
            evaluate(
                [
                    replace(
                        base,
                        stage1=replace(
                            base.stage1, payload_chunk_rows=chunk, prepare_quant_cu=cu
                        ),
                    )
                    for chunk in (128, 256, 512, 1024)
                    for cu in (32, 64, 128)
                ],
                "prepare_payload",
            )
            best = min(results, key=lambda item: item[0])[1]
            evaluate(
                [
                    replace(best, stage2=replace(best.stage2, block_m=bm, block_n=bn))
                    for bm in (16, 32, 64)
                    if bm <= best.stage1.sort_block_m
                    for bn in (128, 256, 512)
                ],
                "W2_rows",
            )
        best = min(results, key=lambda item: item[0])[1]
        evaluate(
            [
                replace(
                    best,
                    stage2=replace(
                        best.stage2,
                        block_n=n,
                        persist_cu=cu,
                        use_nt=nt,
                        persist_strided=strided,
                    ),
                )
                for n, cu, nt, strided in (
                    (128, 256, True, False),
                    (256, 128, True, False),
                    (256, 192, True, False),
                    (256, 256, False, False),
                    (256, 256, True, True),
                    (512, 256, True, False),
                    (512, 128, True, False),
                )
            ],
            "W2",
        )
        if best.stage1.sort_block_m >= 64:
            evaluate(
                [
                    replace(best, stage2=replace(best.stage2, block_m=bm, block_n=bn))
                    for bm in (32, 64)
                    for bn in (128, 256)
                ],
                "W2_rows",
            )
        if args.schedule:
            best = min(results, key=lambda item: item[0])[1]
            dispatch = (4, 8, 16, 32, 64, 128)
            evaluate(
                [
                    replace(best, stage1=replace(best.stage1, num_dispatch_cu=d))
                    for d in dispatch
                    if d % world == 0
                ],
                "dispatch",
            )
            best = min(results, key=lambda item: item[0])[1]
            evaluate(
                [
                    replace(
                        best,
                        stage1=replace(
                            best.stage1,
                            grid_mult=g,
                            work_shards=s,
                            b_nt=nt,
                            waves_per_eu_hint=wpe,
                        ),
                    )
                    for g, s, nt, wpe in (
                        (1, 1, 3, 2),
                        (1, 2, 3, 2),
                        (1, 8, 3, 2),
                        (2, 4, 3, 2),
                        (3, 4, 3, 2),
                        (1, 4, 0, 2),
                        (1, 4, 3, 1),
                        (1, 4, 3, 4),
                    )
                ],
                "W13_schedule",
            )
        if args.kernel_knobs:
            best = min(results, key=lambda item: item[0])[1]
            evaluate(
                [
                    replace(best, stage1=replace(best.stage1, **updates))
                    for updates in (
                        {"mfma_amajor": not best.stage1.mfma_amajor},
                        {"async_a_copy": False},
                        {"pipe_weights": False},
                        {"swizzle_a": False},
                    )
                ],
                "W13_pipeline",
            )
            best = min(results, key=lambda item: item[0])[1]
            evaluate(
                [
                    replace(best, stage2=replace(best.stage2, **updates))
                    for updates in (
                        {"b_hoist": False},
                        {"ascale_prefetch": False},
                        {"spatial_partition": 0},
                        {"spatial_partition": 102},
                        {"spatial_partition": 202},
                        {"spatial_partition": 802},
                        {"bf16_lds": True},
                    )
                ],
                "W2_pipeline",
            )
        best = min(results, key=lambda item: item[0])[1]
        combine_results = []
        for blocks, waves in (
            (16, 4),
            (32, 4),
            (64, 4),
            (128, 4),
            (128, 8),
            (256, 4),
            (256, 8),
        ):
            evaluate(
                [replace(best, combine_blocks=blocks, combine_waves=waves)],
                f"combine_{blocks}_{waves}",
            )
            combine_results.append((results[-1][0], blocks, waves))
        if rank == 0:
            print(f"[COMBINE_WINNER] M={tokens} {min(combine_results)}", flush=True)
        mega.comb_op.cfg = mega.comb_cfg
        if rank == 0:
            best_ms, best = min(results, key=lambda item: item[0])
            print(
                f"[WINNER] EP{world} M={tokens} {best_ms * 1000:.1f}us {best}",
                flush=True,
            )
            if baseline_ms is not None:
                print(
                    f"[BASELINE] M={tokens} MoRI+AITER={baseline_ms * 1000:.1f}us "
                    f"ratio={baseline_ms / best_ms:.3f}",
                    flush=True,
                )
        barrier()

    for tokens in map(int, args.tokens.split(",")):
        tune_case(tokens)
    if args.selected_only:
        from aiter.ops.flydsl.kernels.mega_moe.mega_moe_stage1 import (
            compile_mega_moe_stage1_bundle,
        )
        from aiter.ops.flydsl.kernels.mega_moe.mega_moe_stage2 import _G2_LAUNCH_CACHE

        launcher = compile_mega_moe_stage1_bundle(
            model_dim=mega.model_dim,
            inter_dim=mega.inter_dim,
            rank=mega.rank,
            experts_per_rank=mega.epr,
            fuse_npes=mega.world_size,
            fuse_topk=mega.topk,
            fuse_cap=mega._s1_cap,
            fuse_mtpr=mega.mtpr,
            fuse_scale_dim=mega._s1_scale_dim,
            fixed_slot_dispatch=mega._s1_fixed_slot,
            num_cu=mega._s1_num_cu,
            swiglu_limit=mega.swiglu_limit,
            tile_state_stride=mega._s1_tile_state_stride,
            variants=mega._bundle_plan.stage1_variants,
        )
        resources = {}
        artifact_index = 0
        for launch in (launcher, *_G2_LAUNCH_CACHE.values()):
            for artifact in launch._mem_cache.values():
                ir = artifact._ir_text
                for binary in re.finditer(r'bin = "((?:[^"\\]|\\.)*)"', ir):
                    encoded = binary[1]
                    data = bytearray()
                    pos = 0
                    while pos < len(encoded):
                        if encoded[pos] == "\\":
                            if encoded[pos + 1] in ('"', "\\"):
                                data.append(ord(encoded[pos + 1]))
                                pos += 2
                            else:
                                data.append(int(encoded[pos + 1 : pos + 3], 16))
                                pos += 3
                        else:
                            data.append(ord(encoded[pos]))
                            pos += 1
                    Path(
                        args.output + f".rank{rank}.{artifact_index}.hsaco"
                    ).write_bytes(data)
                    artifact_index += 1
                for match in re.finditer(r'gpu.kernel_metadata<"([^"]+)"', ir):
                    start = ir.find("metadata = {", match.end())
                    end = ir.find("}", start)
                    resources[match[1]] = {
                        key: int(value)
                        for key, value in re.findall(
                            r"(\w+) = (\d+) : i64", ir[start:end]
                        )
                    }
        Path(args.output + f".rank{rank}.resources.json").write_text(
            json.dumps(resources, indent=2) + "\n"
        )
    ms.shmem_finalize()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
