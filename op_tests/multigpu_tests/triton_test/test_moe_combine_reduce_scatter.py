# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import argparse
import multiprocessing as mp

import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F

import aiter
from aiter import logger
from aiter.dist.parallel_state import get_tp_group
from aiter.dist.utils import get_distributed_init_method, get_ip, get_open_port
from aiter.fused_moe import _flydsl_v2_stage2_wrapper, moe_sorting
from aiter.ops.triton.comms.fused.moe_combine_reduce_scatter import (
    MoeCombineReduceScatter,
    fused_moe_combine_reduce_scatter,
)
from aiter.ops.triton.utils._triton import arch_info
from aiter.test_common import ensure_spawn_method

WORLD_SIZES = [2, 4, 8]
BLOCK_M = 128
# (hidden, topk, experts, token counts). A count that is not a multiple of the world
# size gives padding rows. With few tokens, some ranks own no token.
SHAPES = [
    (6144, 4, 128, [1, 3, 1000, 2343, 4096, 5857, 8192, 14394, 16383]),
    (3072, 4, 128, [7, 2343, 4096]),
    (5120, 2, 32, [5, 2343, 4096]),
]
STAGE2_KERNEL = "flydsl_moe2_layout_afp4_wfp4_bf16_t128x256x256_scatter_sbm128"
STAGE2_KERNEL_REDUCE = "flydsl_moe2_layout_afp4_wfp4_bf16_t128x256x256_reduce_sbm128"
STAGE2_INTER = 768

SKIP = arch_info.get_arch() != "gfx950"
pytestmark = pytest.mark.skipif(SKIP, reason="needs gfx950")


def _routing(tokens, topk, experts, hidden, seed, device, group):
    """A real moe_sorting of random routes, the same on all ranks."""
    gen = torch.Generator(device=device).manual_seed(seed)
    logits = torch.randn(
        (tokens, experts), generator=gen, dtype=torch.float32, device=device
    )
    ids = logits.topk(topk, dim=1).indices.to(torch.int32)
    weights = 2 * torch.rand(
        (tokens, topk), generator=gen, dtype=torch.float32, device=device
    )
    sort = moe_sorting(
        ids,
        weights,
        experts,
        hidden,
        torch.bfloat16,
        BLOCK_M,
        accumulate=False,
        output_aux="opus",
    )
    sort = list(sort)
    for i in (0, 1, 2, 3, 6):  # sorted ids, weights, expert ids, counts, reverse
        sort[i] = sort[i].contiguous()
        dist.broadcast(sort[i], 0, group=group)
    return sort


def _reference(combined, tokens, group):
    """The rows of this rank of the fp32 sum of the combined rows in rank order."""
    world = dist.get_world_size(group)
    rows = -(-tokens // world)
    padded = F.pad(combined, (0, 0, 0, rows * world - tokens)).contiguous()
    parts = torch.empty_like(padded)
    dist.all_to_all_single(parts, padded, group=group)
    parts = parts.view(world, rows, -1).float()
    total = parts[0]
    for src in range(1, world):
        total = total + parts[src]
    return total.to(torch.bfloat16)


def _combine(inter, rev, sw, tokens, topk, experts, hidden):
    out = torch.empty((tokens, hidden), dtype=torch.bfloat16, device=inter.device)
    aiter.mxfp4_moe_scatter_reduce(
        flat_out=inter,
        reverse_sorted=rev,
        sorted_weights=sw,
        out=out,
        NE=experts,
        TOPK=topk,
        D_HIDDEN=hidden,
        MB=BLOCK_M,
    )
    return out


def _bad(out, ref):
    return int((out.view(torch.int16) != ref.view(torch.int16)).sum())


def _run(inter, rev, sw, tokens, comm):
    """The fused op on a poisoned output (0xFFFF), so each row must be written."""
    rows = -(-tokens // comm.world)
    out = torch.full((rows, comm.hidden), -1, dtype=torch.int16, device=inter.device)
    return fused_moe_combine_reduce_scatter(
        inter, rev, sw, tokens, comm, out=out.view(torch.bfloat16)
    )


def _check_shape(rank, device, tp, hidden, topk, experts, token_counts, failures):
    group = tp.device_group
    comm = MoeCombineReduceScatter(tp.cpu_group, max(token_counts), hidden, topk)
    try:
        for i, tokens in enumerate(token_counts):
            sort = _routing(tokens, topk, experts, hidden, i, device, group)
            sw, rev = sort[1], sort[6]
            if i == 2:  # skipped routes, as expert parallel makes them
                rev[::7] = -1
            gen = torch.Generator(device=device).manual_seed(100 * i + rank)
            inters = [
                torch.randn(
                    (sw.numel(), hidden),
                    generator=gen,
                    dtype=torch.bfloat16,
                    device=device,
                )
                for _ in range(2)
            ]
            refs = [
                _reference(
                    _combine(x, rev, sw, tokens, topk, experts, hidden), tokens, group
                )
                for x in inters
            ]
            # 2 launches back to back, with no barrier between them.
            torch.cuda.synchronize()
            dist.barrier(group=tp.cpu_group)
            outs = [_run(x, rev, sw, tokens, comm) for x in inters]
            for j, (out, ref) in enumerate(zip(outs, refs)):
                if _bad(out, ref):
                    failures.append(
                        f"world {comm.world} rank {rank} hidden {hidden} topk {topk} "
                        f"tokens {tokens} "
                        f"launch {j}: {_bad(out, ref)} values differ"
                    )
            if tokens == 2343:
                _check_graph(rank, comm, inters, refs, rev, sw, tokens, failures)
        if comm.error() != [0, 0, 0, 0]:
            failures.append(f"rank {rank} hidden {hidden}: error {comm.error()}")
    finally:
        comm.close()


def _check_graph(rank, comm, inters, refs, rev, sw, tokens, failures):
    """2 launches in a HIP graph, replayed 3 times on new data."""
    static = [x.clone() for x in inters]
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        outs = [
            fused_moe_combine_reduce_scatter(x, rev, sw, tokens, comm) for x in static
        ]
    for rep in range(3):
        for x, new in zip(static, inters[rep % 2 :] + inters[: rep % 2]):
            x.copy_(new)
        graph.replay()
        for out, ref in zip(outs, refs[rep % 2 :] + refs[: rep % 2]):
            if _bad(out, ref):
                failures.append(f"rank {rank} tokens {tokens}: graph replay {rep}")


def _check_stage2(rank, device, tp, failures):
    """Stage 2 with scatter_intermediate feeds the fused op. The result is the
    combine of stage 2, then the rank-order reduce-scatter."""
    hidden, topk, experts, tokens = 6144, 4, 128, 2343
    sort = _routing(tokens, topk, experts, hidden, 7, device, tp.device_group)
    sorted_ids, sw, expert_ids, counts, rev = (sort[i] for i in (0, 1, 2, 3, 6))
    rows = sorted_ids.numel()
    gen = torch.Generator(device=device).manual_seed(rank)
    # Random fp4 bytes with unit e8m0 scales: only the data path is under test.
    a2 = torch.randint(
        0,
        256,
        (rows, STAGE2_INTER // 2),
        generator=gen,
        dtype=torch.uint8,
        device=device,
    )
    w2 = torch.randint(
        0,
        256,
        (experts, hidden, STAGE2_INTER // 2),
        generator=gen,
        dtype=torch.uint8,
        device=device,
    )
    a2_scale = torch.full(
        ((rows + BLOCK_M) * STAGE2_INTER // 32,), 127, dtype=torch.uint8, device=device
    )
    w2_scale = torch.full(
        (experts * hidden * STAGE2_INTER // 32,), 127, dtype=torch.uint8, device=device
    )

    def stage2(scatter_intermediate, kernel=STAGE2_KERNEL):
        out = torch.full((tokens, hidden), -1, dtype=torch.int16, device=device)
        out = out.view(torch.bfloat16)
        _flydsl_v2_stage2_wrapper(
            a2,
            None,
            w2,
            sorted_ids,
            expert_ids,
            counts,
            out,
            topk,
            kernelName=kernel,
            model_dim=hidden,
            inter_dim=STAGE2_INTER,
            num_experts=experts,
            w2_scale=w2_scale,
            a2_scale=a2_scale,
            sorted_weights=sw,
            block_m=BLOCK_M,
            reverse_sorted=rev,
            scatter_intermediate=scatter_intermediate,
        )
        return out

    ref = _reference(stage2(None), tokens, tp.device_group)
    items = {}
    untouched = stage2(items)
    if bool((untouched.view(torch.int16) != -1).any()):
        failures.append(f"rank {rank}: stage 2 wrote out with scatter_intermediate")
    if (items.get("topk"), items.get("M"), items.get("model_dim")) != (
        topk,
        tokens,
        hidden,
    ):
        failures.append(f"rank {rank}: scatter_intermediate has {sorted(items)}")
        return
    comm = MoeCombineReduceScatter(tp.cpu_group, tokens, hidden, topk)
    try:
        torch.cuda.synchronize()
        dist.barrier(group=tp.cpu_group)
        out = _run(
            items["inter"],
            items["reverse_sorted"],
            items["sorted_weights"],
            tokens,
            comm,
        )
        if _bad(out, ref):
            failures.append(f"rank {rank}: stage 2 + fused op: {_bad(out, ref)} differ")
        try:
            _run(items["inter"], rev, sw[:0], tokens, comm)
            failures.append(f"rank {rank}: the fused op took empty sorted_weights")
        except AssertionError:
            pass
    finally:
        comm.close()
    # The same dict in a call with another epilogue: the call clears the dict and
    # writes out.
    complete = stage2(items, STAGE2_KERNEL_REDUCE)
    if items:
        failures.append(f"rank {rank}: stale scatter_intermediate {sorted(items)}")
    if bool((complete.view(torch.int16) == -1).all(dim=1).any()):
        failures.append(f"rank {rank}: the reduce epilogue left rows of out")


def _check_timeout(rank, device, tp, failures):
    """The last rank skips a launch. The other ranks must stop at the time limit and
    write the error record of each rank. After reset, a launch is correct again."""
    hidden, topk, experts, tokens = 6144, 4, 128, 4096
    group = tp.device_group
    sort = _routing(tokens, topk, experts, hidden, 9, device, group)
    sw, rev = sort[1], sort[6]
    gen = torch.Generator(device=device).manual_seed(rank)
    inter = torch.randn(
        (sw.numel(), hidden), generator=gen, dtype=torch.bfloat16, device=device
    )
    ref = _reference(
        _combine(inter, rev, sw, tokens, topk, experts, hidden), tokens, group
    )
    comm = MoeCombineReduceScatter(tp.cpu_group, tokens, hidden, topk, timeout_s=0.05)
    try:
        torch.cuda.synchronize()
        dist.barrier(group=tp.cpu_group)
        if rank != comm.world - 1:
            _run(inter, rev, sw, tokens, comm)
        torch.cuda.synchronize()
        dist.barrier(group=tp.cpu_group)  # the other ranks stopped
        record = comm.error()
        if record[0] != 1 or record[3] != 2:
            failures.append(f"rank {rank}: error record {record} after a time-out")
        comm.reset()
        out = _run(inter, rev, sw, tokens, comm)
        if _bad(out, ref) or comm.error() != [0, 0, 0, 0]:
            failures.append(f"rank {rank}: launch after reset is wrong")
    finally:
        comm.close()


def _worker(rank, world_size, init_method):
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    aiter.init_dist_env(world_size, rank, distributed_init_method=init_method)
    tp = get_tp_group()
    failures = []
    try:
        for hidden, topk, experts, token_counts in SHAPES:
            _check_shape(
                rank, device, tp, hidden, topk, experts, token_counts, failures
            )
        _check_stage2(rank, device, tp, failures)
        _check_timeout(rank, device, tp, failures)
        return failures
    finally:
        aiter.destroy_dist_env()


@pytest.mark.parametrize("world_size", WORLD_SIZES)
def test_moe_combine_reduce_scatter(world_size):
    if torch.cuda.device_count() < world_size:
        pytest.skip(f"needs {world_size} GPUs")
    ensure_spawn_method()
    init_method = get_distributed_init_method(get_ip(), get_open_port())
    with mp.Pool(processes=world_size) as pool:
        results = [
            pool.apply_async(_worker, (rank, world_size, init_method))
            for rank in range(world_size)
        ]
        failures = [f for r in results for f in r.get(timeout=1800)]
    assert not failures, "\n".join(failures)


if __name__ == "__main__":  # CI runs the multi-GPU tests as scripts
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-t",
        "--world-size",
        type=int,
        nargs="+",
        default=[w for w in WORLD_SIZES if w <= torch.cuda.device_count()],
        choices=WORLD_SIZES,
    )
    args = parser.parse_args()
    if SKIP or not args.world_size:
        logger.warning("test_moe_combine_reduce_scatter skipped: needs gfx950 GPUs")
    for world_size in [] if SKIP else args.world_size:
        test_moe_combine_reduce_scatter(world_size)
        logger.info("test_moe_combine_reduce_scatter: world size %d passed", world_size)
