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
# gives ranks that own fewer rows, or no row. The HIP graph check uses the first shape.
SHAPES = [
    (6144, True, [1, 3, 64, 256]),
    (7168, False, [5, 64, 255]),
    (2880, True, [7, 64, 130]),
    (4096, False, [2, 128]),
]
# The sentinel and time-out checks use the comm size of the first shape, so they launch
# a kernel that each rank has already compiled.
MAX_TOKENS = max(SHAPES[0][2])

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
    x_q, scale = _mxfp8(y)
    return x_q, scale, res_out


def _mxfp8(y):
    """MXFP8 of the fp32 y: e4m3 values and an E8M0 scale for each group of 32. The
    group amax rounds to a power of 2 in fp32, up from 1.75 x 2^e (2^128 is inf). The
    scale is this power / 2^8, clamped to 2^-127 .. 2^127."""
    tokens, hidden = y.shape
    groups = y.view(tokens, hidden // 32, 32).double()
    amax = groups.abs().amax(-1)
    mant, exp = torch.frexp(amax)  # amax = mant * 2^exp, 0.5 <= mant < 1
    pow2 = torch.exp2((exp - 1 + (mant >= 0.875).to(torch.int32)).double())
    pow2 = torch.where(amax > 0, pow2, 0.0).float().double()
    k = (torch.log2(pow2) - 8).clamp(-127, 127)
    x_q = (groups * torch.exp2(-k)[..., None]).float().view(tokens, hidden)
    return x_q.to(torch.float8_e4m3fn), (k + 127).to(torch.uint8)


def _steps(a, b):
    """The number of rounding steps between the bf16 or e4m3 values a and b. A value
    that is not finite must have the same bits as the reference."""
    bits = 8 * a.element_size()
    ia, ib = (
        t.view(torch.int16 if bits == 16 else torch.uint8).int() & ((1 << bits) - 1)
        for t in (a, b)
    )
    mag = (1 << (bits - 1)) - 1
    ka, kb = (torch.where(i >> (bits - 1) == 1, -(i & mag), i & mag) for i in (ia, ib))
    finite = torch.isfinite(a.float()) & torch.isfinite(b.float())
    return torch.where(finite, (ka - kb).abs(), (ia != ib).int() << 16)


def _diff(outs, refs):
    """A message for each output that is not close to the reference. A bf16 or e4m3
    value can be one rounding step from the reference, in 1 of 10,000 values (at least
    8). The scale bytes must be equal."""
    msgs = []
    for name, o, r in zip(("x_q", "x_scale", "residual_out"), outs, refs):
        o = o.cpu()
        if o.dtype == torch.uint8:
            steps, limit = (o.int() - r.int()).abs(), 0
        else:
            steps, limit = _steps(o, r), 1
        n, worst = int((steps > 0).sum()), int(steps.max())
        if worst > limit or n > max(8, steps.numel() // 10000):
            msgs.append(f"{name}: {n} values differ, by up to {worst} steps")
    return msgs


def generate_all_reduce_rmsnorm_mxfp8_inputs(tokens, hidden, gemma, seed, rank, device):
    """x of this rank, and the residual and norm weight of all ranks (bf16). The test
    and the benchmark use these inputs. With more than 2 tokens, row 1 is 0."""
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
    return x, residual, weight


def _inputs(tokens, hidden, gemma, seed, rank, device, group):
    """The inputs of this rank and the reference."""
    x, residual, weight = generate_all_reduce_rmsnorm_mxfp8_inputs(
        tokens, hidden, gemma, seed, rank, device
    )
    parts = [torch.empty_like(x) for _ in range(dist.get_world_size(group))]
    dist.all_gather(parts, x, group=group)
    ref = _reference([p.cpu() for p in parts], residual.cpu(), weight.cpu(), EPS, gemma)
    return x, residual, weight, ref


def _check_shape(rank, device, tp, hidden, gemma, token_counts, graph, failures):
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
                for msg in _diff(out, d[3]):
                    failures.append(f"{tag} tokens {tokens} launch {j}: {msg}")
        if graph:
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
            for msg in _diff(out, d[3]):
                failures.append(
                    f"world {comm.world} rank {rank} hidden {hidden}: graph replay {rep}"
                    f" launch {k}: {msg}"
                )


def _check_sentinel(rank, device, tp, failures):
    """A 0xFFFF (NaN) value in x must not look like an empty slot: no wait times
    out, and the other rows are correct."""
    hidden, tokens = 6144, 64
    x, residual, weight, ref = _inputs(
        tokens, hidden, True, 7, rank, device, tp.device_group
    )
    x.view(torch.int16)[tokens - 1, 5] = -1  # 0xFFFF
    comm = AllReduceRmsnormMxfp8(tp.cpu_group, MAX_TOKENS, hidden)
    try:
        torch.cuda.synchronize()
        dist.barrier(group=tp.cpu_group)
        out = fused_all_reduce_rmsnorm_mxfp8(x, residual, weight, EPS, comm, gemma=True)
        msgs = _diff([o[: tokens - 1] for o in out], [r[: tokens - 1] for r in ref])
        if msgs or comm.error() != [0, 0, 0, 0]:
            failures.append(f"rank {rank}: 0xFFFF data: {msgs}, error {comm.error()}")
    finally:
        comm.close()


# Group amax values at the edges of the MXFP8 scale rule, and their scale bytes.
EDGE_SCALES = [
    (1.75 * 2.0**-119, 1),  # the smallest amax with a scale above 2^-127
    (4e-36, 1),
    (5.2e-36, 1),
    (1.75 * 2.0**-118, 2),
    (2.0**-100, 19),
    (1.75 / 8, 117),
    (1.7421875, 119),
    (1.75, 120),
    (28.0, 124),
    (448.0, 128),
    (2.0**100, 219),
    (1.75 * 2.0**127, 254),  # amax rounds to 2^128 (inf) in fp32
    (3.3e38, 254),
]


def _check_scales(rank, device, tp, failures):
    """Groups of 32 with an amax from EDGE_SCALES. Rank 0 has rows of +-1, the other
    ranks 0, and the residual is 0: then y = +-weight, and the weight sets the amax."""
    hidden, tokens = 6144, 4
    gen = torch.Generator(device=device).manual_seed(77)
    signs = torch.randint(0, 2, (tokens, hidden), generator=gen, device=device)
    ones = (2 * signs - 1).to(torch.bfloat16)
    x = ones if rank == 0 else torch.zeros_like(ones)
    residual = torch.zeros_like(ones)
    u = 0.5 + 0.5 * torch.rand(
        (hidden // 32, 32), generator=gen, dtype=torch.float32, device=device
    )
    g = torch.arange(hidden // 32, device=device)
    u[g, g % 32] = 1.0  # one value of each group is the amax
    amax = torch.tensor([a for a, _ in EDGE_SCALES], dtype=torch.float32, device=device)
    weight = (amax[g % len(EDGE_SCALES)][:, None] * u).view(hidden).to(torch.bfloat16)
    parts = [ones.cpu()] + [torch.zeros_like(ones.cpu())] * (tp.world_size - 1)
    ref = _reference(parts, residual.cpu(), weight.cpu(), EPS, False)
    want = torch.tensor([b for _, b in EDGE_SCALES], dtype=torch.uint8)
    want = want[torch.arange(hidden // 32) % len(EDGE_SCALES)]
    assert torch.equal(ref[1], want.expand(tokens, -1)), "EDGE_SCALES or _mxfp8"
    comm = AllReduceRmsnormMxfp8(tp.cpu_group, tokens, hidden)
    try:
        torch.cuda.synchronize()
        dist.barrier(group=tp.cpu_group)
        out = fused_all_reduce_rmsnorm_mxfp8(x, residual, weight, EPS, comm)
        msgs = _diff(out, ref)
        if msgs or comm.error() != [0, 0, 0, 0]:
            failures.append(f"rank {rank}: edge scales: {msgs}, error {comm.error()}")
    finally:
        comm.close()


def _check_timeout(rank, device, tp, failures):
    """The last rank skips a launch. The other ranks must stop at the time limit and
    write the error record of each rank. After reset, a launch is correct again."""
    hidden, tokens = 6144, 128
    x, residual, weight, ref = _inputs(
        tokens, hidden, True, 9, rank, device, tp.device_group
    )
    comm = AllReduceRmsnormMxfp8(tp.cpu_group, MAX_TOKENS, hidden)
    limit = comm.timeout_us
    try:
        torch.cuda.synchronize()
        dist.barrier(group=tp.cpu_group)
        comm.timeout_us = 50_000  # 50 ms for the launch that the last rank skips
        if rank != comm.world - 1:
            fused_all_reduce_rmsnorm_mxfp8(x, residual, weight, EPS, comm, gemma=True)
        torch.cuda.synchronize()
        dist.barrier(group=tp.cpu_group)  # the other ranks stopped
        record = comm.error()
        if record[0] != 1 or record[3] not in (1, 2):
            failures.append(f"rank {rank}: error record {record} after a time-out")
        # The launch after reset has the default limit: a rank that starts late must
        # not stop the others.
        comm.timeout_us = limit
        comm.reset()
        out = fused_all_reduce_rmsnorm_mxfp8(x, residual, weight, EPS, comm, gemma=True)
        msgs = _diff(out, ref)
        if msgs or comm.error() != [0, 0, 0, 0]:
            failures.append(f"rank {rank}: launch after reset: {msgs}, {comm.error()}")
    finally:
        comm.close()


def _worker(rank, world_size, init_method):
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    aiter.init_dist_env(world_size, rank, distributed_init_method=init_method)
    tp = get_tp_group()
    failures = []
    # 8 ranks run only the first shape and the checks that launch its kernel. Each other
    # kernel variant compiles on all 8 ranks, which takes most of the CI time. 2 and 4
    # ranks run all checks.
    full = world_size < 8
    try:
        for i, (hidden, gemma, token_counts) in enumerate(
            SHAPES if full else SHAPES[:1]
        ):
            _check_shape(
                rank, device, tp, hidden, gemma, token_counts, i == 0, failures
            )
        _check_sentinel(rank, device, tp, failures)
        _check_timeout(rank, device, tp, failures)
        if full:
            _check_scales(rank, device, tp, failures)
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
