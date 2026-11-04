# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import multiprocessing as mp

import pytest
import torch

import aiter
from aiter import logger
from aiter.dist.parallel_state import get_tp_group
from aiter.dist.utils import get_distributed_init_method, get_ip, get_open_port
from aiter.ops.triton.comms.all_reduce import all_reduce_free, all_reduce_init
from aiter.ops.triton.comms.fused.argmax_exchange import (
    argmax_exchange_max_bytes,
    argmax_exchange_pick,
)
from aiter.ops.triton.gemm.basic.gemm_a16w16 import (
    gemm_a16w16,
    gemm_a16w16_argmax_keys,
)
from aiter.ops.triton.utils._triton import arch_info
from aiter.test_common import ensure_spawn_method

WORLD_SIZE = 4
N, K = 62080, 2048  # a Qwen3.5 LM head shard at TP4
ROWS = [1, 5, 13, 16, 17, 64, 80, 300, 16, 1]

SKIP = torch.cuda.device_count() < WORLD_SIZE or arch_info.get_arch() != "gfx1250"
pytestmark = pytest.mark.skipif(SKIP, reason="needs 4 gfx1250 GPUs")


def _case(m, seed, ties, device):
    """x and the full (world * N, K) head; with ties, equal maxima in every shard."""
    gen = torch.Generator(device=device).manual_seed(seed)
    w = torch.randn(
        (WORLD_SIZE * N, K), generator=gen, dtype=torch.float32, device=device
    )
    w = (w * 0.02).to(torch.bfloat16)
    if ties:
        w[[5, N + 7920, 2 * N + 3, 3 * N - 1, 4 * N - 1]] = w[17] * 10
    x = torch.randn((m, K), generator=gen, dtype=torch.float32, device=device)
    return x.to(torch.bfloat16), w


def _worker(rank, init_method):
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    aiter.init_dist_env(WORLD_SIZE, rank, distributed_init_method=init_method)
    group = get_tp_group().cpu_group
    inboxes, calls = all_reduce_init(group, argmax_exchange_max_bytes(max(ROWS)))
    failures = []
    try:
        for i, m in enumerate(ROWS):
            for ties in (False, True):
                x, w = _case(m, i, ties, device)
                ref = torch.argmax(gemm_a16w16(x, w), dim=1).to(torch.int32)
                shard = w[rank * N : (rank + 1) * N]
                out = torch.empty(m, dtype=torch.int32, device=device)
                keys = gemm_a16w16_argmax_keys(x, shard)
                argmax_exchange_pick(keys, rank * N, out, group, inboxes, calls)
                if not torch.equal(out, ref):
                    failures.append(f"rank {rank}: {m} rows, ties {ties} are wrong")
        # The same calls in a HIP graph, with new inputs on every replay.
        x, w = _case(16, 99, True, device)
        shard = w[rank * N : (rank + 1) * N].contiguous()
        xin = torch.empty_like(x)
        out = torch.empty(16, dtype=torch.int32, device=device)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            keys = gemm_a16w16_argmax_keys(xin, shard)
            argmax_exchange_pick(keys, rank * N, out, group, inboxes, calls)
        for rep in range(10):
            gen = torch.Generator(device=device).manual_seed(500 + rep)
            new_x = torch.randn(
                (16, K), generator=gen, dtype=torch.float32, device=device
            )
            xin.copy_(new_x.to(torch.bfloat16))
            graph.replay()
            ref = torch.argmax(gemm_a16w16(xin, w), dim=1).to(torch.int32)
            if not torch.equal(out, ref):
                failures.append(f"rank {rank}: graph replay {rep} is wrong")
        if calls[-1].item():
            failures.append(f"rank {rank}: {calls[-1].item()} polls timed out")
        return failures
    finally:
        all_reduce_free(group, inboxes)
        aiter.destroy_dist_env()


def test_argmax_exchange_pick():
    ensure_spawn_method()
    init_method = get_distributed_init_method(get_ip(), get_open_port())
    with mp.Pool(processes=WORLD_SIZE) as pool:
        results = [
            pool.apply_async(_worker, (rank, init_method)) for rank in range(WORLD_SIZE)
        ]
        failures = [f for r in results for f in r.get(timeout=1800)]
    assert not failures, "\n".join(failures)


if __name__ == "__main__":  # CI runs the multi-GPU tests as scripts
    if SKIP:
        logger.warning("test_argmax_exchange_pick skipped: needs 4 gfx1250 GPUs")
    else:
        test_argmax_exchange_pick()
