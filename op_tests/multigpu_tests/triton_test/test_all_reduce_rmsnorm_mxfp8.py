# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import argparse
import multiprocessing as mp

import pytest
import torch
import torch.distributed as dist

import aiter
from aiter import logger
from aiter.dist.parallel_state import get_tp_group
from aiter.dist.utils import get_distributed_init_method, get_ip, get_open_port
from aiter.ops.triton.comms.fused.all_reduce_rmsnorm_mxfp8 import (
    AllReduceRmsnormMxfp8,
    fused_all_reduce_rmsnorm_mxfp8,
)
from aiter.ops.triton.utils._triton import arch_info
from aiter.test_common import ensure_spawn_method

WORLD_SIZES = [2, 4, 8]
EPS = 1e-6
# (hidden, gemma, token counts). A count that is not a multiple of the world size
# gives ranks that own fewer rows, or no row.
SHAPES = [
    (6144, True, [1, 3, 24, 64, 96, 128, 192, 256]),
    (7168, False, [5, 64, 255]),
    (2880, True, [7, 64, 130]),
    (4096, False, [2, 128]),
]
F32_TINY = torch.finfo(torch.float32).tiny

SKIP = arch_info.get_arch() != "gfx950"
pytestmark = pytest.mark.skipif(SKIP, reason="needs gfx950")


def _reference(parts, residual, weight, eps, gemma):
    """The outputs of the op, on the CPU. The CPU gives IEEE fp32 division and square
    root, and the sum of squares has the order of the kernel."""
    tokens, hidden = residual.shape
    h = parts[0].float()
    for p in parts[1:]:
        h = h + p.float()
    s = h.to(torch.bfloat16).float() + residual.float()
    res_out = s.to(torch.bfloat16)
    # Thread t of hidden / 16 holds the columns 8t.. and hidden / 2 + 8t...
    threads = hidden // 16
    v = s.view(tokens, 2, threads, 8).permute(0, 2, 1, 3).reshape(tokens, threads, 16)
    sq = v[..., 0] * v[..., 0]
    for i in range(1, 16):
        sq = sq + v[..., i] * v[..., i]
    waves = -(-threads // 64)
    lanes = torch.zeros((tokens, waves * 64), dtype=torch.float32, device="cpu")
    lanes[:, :threads] = sq
    lanes = lanes.view(tokens, waves, 64)
    lane_ids = torch.arange(64, device="cpu")
    for off in (32, 16, 8, 4, 2, 1):
        lanes = lanes + lanes[..., lane_ids ^ off]
    total = lanes[:, 0, 0]
    for w in range(1, waves):
        total = total + lanes[:, w, 0]
    var = total / torch.tensor(float(hidden), dtype=torch.float32, device="cpu")
    eps32 = torch.tensor(eps, dtype=torch.float32, device="cpu")
    one = torch.tensor(1.0, dtype=torch.float32, device="cpu")
    rstd = one / torch.sqrt(var + eps32)
    w = weight.float()
    if gemma:
        w = w + 1.0
    y = ((s * rstd[:, None]) * w).to(torch.bfloat16).float()
    groups = y.view(tokens, hidden // 32, 32)
    amax = torch.clamp(groups.abs().amax(-1), min=F32_TINY)
    bits = (amax / 448.0).view(torch.int32)
    ex = (bits >> 23) & 0xFF
    scale = torch.clamp(ex + ((bits & 0x7FFFFF) != 0).to(torch.int32), 0, 254)
    scale = torch.where(ex == 0, 0, scale)
    inv = ((254 - scale) << 23).view(torch.float32)
    x_q = (groups * inv[..., None]).view(tokens, hidden).to(torch.float8_e4m3fn)
    return x_q, scale.to(torch.uint8), res_out


def _bad(outs, refs):
    """The number of bytes of the outputs that differ from the reference."""
    return sum(
        int((o.cpu().view(torch.uint8) != r.view(torch.uint8)).sum())
        for o, r in zip(outs, refs)
    )


def _inputs(tokens, hidden, gemma, seed, rank, device, group):
    """x of this rank, the residual and weight of all ranks, and the reference."""
    gen = torch.Generator(device=device).manual_seed(1000 * seed + rank)
    x = torch.randn(
        (tokens, hidden), generator=gen, dtype=torch.float32, device=device
    ).to(torch.bfloat16)
    gen.manual_seed(1000 * seed + 999)
    residual = torch.randn(
        (tokens, hidden), generator=gen, dtype=torch.float32, device=device
    ).to(torch.bfloat16)
    weight = 0.1 * torch.randn(
        (hidden,), generator=gen, dtype=torch.float32, device=device
    )
    weight = (weight if gemma else weight + 1.0).to(torch.bfloat16)
    if tokens > 2:  # a zero row: all-zero groups, rstd = 1 / sqrt(eps)
        x[1] = 0
        residual[1] = 0
    parts = [torch.empty_like(x) for _ in range(dist.get_world_size(group))]
    dist.all_gather(parts, x, group=group)
    ref = _reference([p.cpu() for p in parts], residual.cpu(), weight.cpu(), EPS, gemma)
    return x, residual, weight, ref


def _check_shape(rank, device, tp, hidden, gemma, token_counts, failures):
    group = tp.device_group
    comm = AllReduceRmsnormMxfp8(tp.cpu_group, max(token_counts), hidden)
    tag = f"world {comm.world} rank {rank} hidden {hidden}"
    try:
        for i, tokens in enumerate(token_counts):
            data = [
                _inputs(tokens, hidden, gemma, 2 * i + j, rank, device, group)
                for j in range(2)
            ]
            # 2 launches back to back, with no barrier between them.
            torch.cuda.synchronize()
            dist.barrier(group=tp.cpu_group)
            outs = [
                fused_all_reduce_rmsnorm_mxfp8(x, r, w, EPS, comm, gemma=gemma)
                for x, r, w, _ in data
            ]
            for j, (out, d) in enumerate(zip(outs, data)):
                if _bad(out, d[3]):
                    failures.append(
                        f"{tag} tokens {tokens} launch {j}: {_bad(out, d[3])} bytes differ"
                    )
        _check_graph(rank, device, tp, comm, gemma, token_counts, failures)
        if comm.error() != [0, 0, 0, 0]:
            failures.append(f"{tag}: error {comm.error()}")
    finally:
        comm.close()


def _check_graph(rank, device, tp, comm, gemma, token_counts, failures):
    """3 launches of mixed sizes in a HIP graph, replayed 3 times on new data."""
    hidden = comm.hidden
    sizes = [token_counts[-1], token_counts[0], token_counts[len(token_counts) // 2]]
    data = [
        [
            _inputs(t, hidden, gemma, 100 + 3 * rep + k, rank, device, tp.device_group)
            for k, t in enumerate(sizes)
        ]
        for rep in range(3)
    ]
    static = [tuple(t.clone() for t in d[:3]) for d in data[0]]
    torch.cuda.synchronize()
    dist.barrier(group=tp.cpu_group)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        outs = [
            fused_all_reduce_rmsnorm_mxfp8(x, r, w, EPS, comm, gemma=gemma)
            for x, r, w in static
        ]
    for rep in range(3):
        for st, d in zip(static, data[rep]):
            for t, new in zip(st, d[:3]):
                t.copy_(new)
        torch.cuda.synchronize()
        dist.barrier(group=tp.cpu_group)
        graph.replay()
        for k, (out, d) in enumerate(zip(outs, data[rep])):
            if _bad(out, d[3]):
                failures.append(
                    f"world {comm.world} rank {rank} hidden {hidden}: graph replay {rep}"
                    f" launch {k}: {_bad(out, d[3])} bytes differ"
                )


def _check_sentinel(rank, device, tp, failures):
    """A 0xFFFF (NaN) value in x must not look like an empty slot: no wait times
    out, and the other rows are exact."""
    hidden, tokens = 6144, 64
    x, residual, weight, ref = _inputs(
        tokens, hidden, True, 7, rank, device, tp.device_group
    )
    x.view(torch.int16)[tokens - 1, 5] = -1  # 0xFFFF
    comm = AllReduceRmsnormMxfp8(tp.cpu_group, tokens, hidden)
    try:
        torch.cuda.synchronize()
        dist.barrier(group=tp.cpu_group)
        out = fused_all_reduce_rmsnorm_mxfp8(x, residual, weight, EPS, comm, gemma=True)
        bad = _bad([o[: tokens - 1] for o in out], [r[: tokens - 1] for r in ref])
        if bad or comm.error() != [0, 0, 0, 0]:
            failures.append(f"rank {rank}: 0xFFFF data: {bad} bytes, {comm.error()}")
    finally:
        comm.close()


def _check_timeout(rank, device, tp, failures):
    """The last rank skips a launch. The other ranks must stop at the time limit and
    write the error record of each rank. After reset, a launch is correct again."""
    hidden, tokens = 6144, 128
    x, residual, weight, ref = _inputs(
        tokens, hidden, True, 9, rank, device, tp.device_group
    )
    comm = AllReduceRmsnormMxfp8(tp.cpu_group, tokens, hidden, timeout_s=0.05)
    try:
        torch.cuda.synchronize()
        dist.barrier(group=tp.cpu_group)
        if rank != comm.world - 1:
            fused_all_reduce_rmsnorm_mxfp8(x, residual, weight, EPS, comm, gemma=True)
        torch.cuda.synchronize()
        dist.barrier(group=tp.cpu_group)  # the other ranks stopped
        record = comm.error()
        if record[0] != 1 or record[3] not in (1, 2):
            failures.append(f"rank {rank}: error record {record} after a time-out")
        comm.reset()
        out = fused_all_reduce_rmsnorm_mxfp8(x, residual, weight, EPS, comm, gemma=True)
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
        for hidden, gemma, token_counts in SHAPES:
            _check_shape(rank, device, tp, hidden, gemma, token_counts, failures)
        _check_sentinel(rank, device, tp, failures)
        _check_timeout(rank, device, tp, failures)
        return failures
    finally:
        aiter.destroy_dist_env()


@pytest.mark.parametrize("world_size", WORLD_SIZES)
def test_all_reduce_rmsnorm_mxfp8(world_size):
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
        logger.warning("test_all_reduce_rmsnorm_mxfp8 skipped: needs gfx950 GPUs")
    for world_size in [] if SKIP else args.world_size:
        test_all_reduce_rmsnorm_mxfp8(world_size)
        logger.info("test_all_reduce_rmsnorm_mxfp8: world size %d passed", world_size)
