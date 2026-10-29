# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import multiprocessing as mp

import pytest
import torch
import torch.distributed as dist

import aiter
from aiter import logger
from aiter.dist.parallel_state import get_tp_group
from aiter.dist.utils import get_distributed_init_method, get_ip, get_open_port
from aiter.ops.triton.comms.all_reduce import all_reduce_free, all_reduce_init
from aiter.ops.triton.comms.fused.gemm_all_reduce_rmsnorm import (
    gemm_all_reduce_rmsnorm,
)
from aiter.ops.triton.gemm.basic.gemm_a16w16 import gemm_a16w16
from aiter.ops.triton.utils._triton import arch_info
from aiter.test_common import ensure_spawn_method

WORLD_SIZE = 4
HIDDEN = 2048
K = 1024
EPS = 1e-6
TOKENS = [1, 2, 3, 4, 8, 16, 17, 32, 48, 49, 64, 100, 128, 256, 512, 1024, 2048]

SKIP = torch.cuda.device_count() < WORLD_SIZE or arch_info.get_arch() != "gfx1250"
pytestmark = pytest.mark.skipif(SKIP, reason="needs 4 gfx1250 GPUs")


def _inputs(tokens, rank, device):
    """This rank's x and w, the residual and the weight, and the fp32 references of
    residual_out and out. y of each rank is bf16, and the 4 are added in rank order,
    as in the kernel."""
    gen = torch.Generator(device=device).manual_seed(tokens)
    xs = [
        torch.randn((tokens, K), generator=gen, dtype=torch.bfloat16, device=device)
        for _ in range(WORLD_SIZE)
    ]
    # A std of 1 / sqrt(K) keeps y near the size of the residual.
    ws = [
        torch.randn((HIDDEN, K), generator=gen, dtype=torch.bfloat16, device=device)
        * K**-0.5
        for _ in range(WORLD_SIZE)
    ]
    shape = (tokens, HIDDEN)
    residual = torch.randn(shape, generator=gen, dtype=torch.bfloat16, device=device)
    w = torch.randn(HIDDEN, generator=gen, dtype=torch.bfloat16, device=device)
    weight = 0.1 * w  # a Gemma weight: 1 + weight is near 1
    ys = [gemm_a16w16(x, wr).float() for x, wr in zip(xs, ws)]
    s = ys[0] + ys[1] + ys[2] + ys[3] + residual.float()
    out = s * torch.rsqrt(s.pow(2).mean(-1, keepdim=True) + EPS) * (1 + weight.float())
    return xs[rank], ws[rank], residual, weight, s, out


def _compare(name, outs, ref_residual, ref_out, group):
    failures = []
    if not torch.equal(outs[1], ref_residual.to(torch.bfloat16)):
        failures.append(f"{name}: residual_out is not bit-exact")
    try:
        torch.testing.assert_close(outs[0], ref_out.to(torch.bfloat16))
    except AssertionError as e:
        failures.append(f"{name}: out: {e}")
    for t in outs:
        parts = [torch.empty_like(t.view(torch.uint8)) for _ in range(WORLD_SIZE)]
        dist.all_gather(parts, t.view(torch.uint8), group=group)
        if not all(torch.equal(parts[0], p) for p in parts):
            failures.append(f"{name}: the ranks have different results")
    return failures


def _worker(rank, init_method):
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    aiter.init_dist_env(WORLD_SIZE, rank, distributed_init_method=init_method)
    tp = get_tp_group()
    group = tp.cpu_group
    cases = [_inputs(tokens, rank, device) for tokens in TOKENS]
    inboxes, calls = all_reduce_init(group, max(TOKENS) * HIDDEN * 2)
    failures = []
    try:
        # First, while the inboxes are empty: the call numbers wrap from 2^30 to 1.
        # Calls of 64 to 67 tokens, so that each call has new inputs.
        calls[:-1].fill_(2**30 - 2)
        for i in range(4):
            x, w, residual, weight, *ref = _inputs(64 + i, rank, device)
            outs = gemm_all_reduce_rmsnorm(
                x, w, residual, weight, EPS, True, group, inboxes, calls
            )
            name = f"rank {rank} call {i} at the counter wrap"
            failures += _compare(name, outs, *ref, tp.device_group)

        for tokens, (x, w, residual, weight, *ref) in zip(TOKENS, cases):
            outs = gemm_all_reduce_rmsnorm(
                x, w, residual, weight, EPS, True, group, inboxes, calls
            )
            name = f"rank {rank} tokens {tokens}"
            failures += _compare(name, outs, *ref, tp.device_group)

        # All token counts in one HIP graph, replayed 10 times.
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            outs = [
                gemm_all_reduce_rmsnorm(x, w, r, wt, EPS, True, group, inboxes, calls)
                for x, w, r, wt, _, _ in cases
            ]
        for _ in range(10):
            graph.replay()
        for tokens, o, (_, _, _, _, *ref) in zip(TOKENS, outs, cases):
            name = f"rank {rank} graph tokens {tokens}"
            failures += _compare(name, o, *ref, tp.device_group)
        if calls[-1].item():
            failures.append(f"rank {rank}: {calls[-1].item()} polls timed out")
    finally:
        all_reduce_free(group, inboxes)
        aiter.destroy_dist_env()
    return failures


def test_gemm_all_reduce_rmsnorm():
    ensure_spawn_method()
    init_method = get_distributed_init_method(get_ip(), get_open_port())
    with mp.Pool(processes=WORLD_SIZE) as pool:
        results = [
            pool.apply_async(_worker, (rank, init_method)) for rank in range(WORLD_SIZE)
        ]
        failures = [f for r in results for f in r.get(timeout=1800)]
    assert not failures, "\n".join(failures[:20])


if __name__ == "__main__":  # CI runs the multi-GPU tests as scripts
    if SKIP:
        logger.warning("test_gemm_all_reduce_rmsnorm skipped: needs 4 gfx1250 GPUs")
    else:
        test_gemm_all_reduce_rmsnorm()
