# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import multiprocessing as mp
import os

import pytest
import torch
import torch.distributed as dist
import triton

import aiter
from aiter import logger
from aiter.dist.communication_op import tensor_model_parallel_fused_allreduce_rmsnorm
from aiter.dist.parallel_state import get_tp_group
from aiter.dist.utils import get_distributed_init_method, get_ip, get_open_port
from aiter.ops.triton.comms.fused.reduce_scatter_rmsnorm_all_gather import (
    reduce_scatter_rmsnorm_all_gather,
    reduce_scatter_rmsnorm_all_gather_free,
    reduce_scatter_rmsnorm_all_gather_init,
)
from aiter.ops.triton.utils._triton import arch_info
from aiter.test_common import ensure_spawn_method

WORLD_SIZE = 4
HIDDEN = 2048
EPS = 1e-6
# Prefill row counts: above the Gluon AR's 8 MiB, ragged and up to the box size.
TOKENS = [2049, 3149, 8116, 13620, 16384]

SKIP = torch.cuda.device_count() < WORLD_SIZE or arch_info.get_arch() != "gfx1250"
pytestmark = pytest.mark.skipif(SKIP, reason="needs 4 gfx1250 GPUs")


def _inputs(tokens, rank, device):
    """This rank's x, the residual and the weight, and the references of
    residual_out and out. The 4 inputs are added in rank order, as in the kernel,
    and the norm reads the bf16 residual_out."""
    gen = torch.Generator(device=device).manual_seed(tokens)
    shape = (tokens, HIDDEN)
    xs = [
        torch.randn(shape, generator=gen, dtype=torch.bfloat16, device=device)
        for _ in range(WORLD_SIZE)
    ]
    residual = torch.randn(shape, generator=gen, dtype=torch.bfloat16, device=device)
    w = torch.randn(HIDDEN, generator=gen, dtype=torch.bfloat16, device=device)
    weight = 0.1 * w  # a Gemma weight: 1 + weight is near 1
    s = xs[0].float() + xs[1].float() + xs[2].float() + xs[3].float()
    y = (residual.float() + s).to(torch.bfloat16)
    yf = y.float()
    out = (
        yf * torch.rsqrt(yf.pow(2).mean(-1, keepdim=True) + EPS) * (1 + weight.float())
    )
    return xs[rank], residual, weight, y, out


def _owned(tokens, rank, programs, device):
    """The rows that this rank owns: quarter `rank` of each program's slice."""
    sr = 4 * triton.cdiv(triton.cdiv(tokens, programs), 4)
    qr = sr // 4
    mask = torch.zeros(tokens, dtype=torch.bool, device=device)
    for b in range(programs):
        lo = b * sr + rank * qr
        if lo >= tokens:
            break
        mask[lo : min(lo + qr, tokens)] = True
    return mask


def _compare(name, outs, ref_residual, ref_out, owned, group):
    failures = []
    out, residual_out = outs
    if not torch.equal(residual_out[owned], ref_residual[owned]):
        failures.append(f"{name}: residual_out of the own rows is not bit-exact")
    try:
        torch.testing.assert_close(out, ref_out.to(torch.bfloat16))
    except AssertionError as e:
        failures.append(f"{name}: out: {e}")
    parts = [torch.empty_like(out.view(torch.uint8)) for _ in range(WORLD_SIZE)]
    dist.all_gather(parts, out.view(torch.uint8), group=group)
    if not all(torch.equal(parts[0], p) for p in parts):
        failures.append(f"{name}: the ranks have different results")
    return failures


def _worker(rank, init_method):
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    aiter.init_dist_env(WORLD_SIZE, rank, distributed_init_method=init_method)
    tp = get_tp_group()
    group = tp.cpu_group
    state = reduce_scatter_rmsnorm_all_gather_init(group, max(TOKENS))
    cases = [_inputs(tokens, rank, device) for tokens in TOKENS]
    owned = [_owned(t, rank, state.programs, device) for t in TOKENS]
    failures = []
    try:
        # Each size 3 times: the calls alternate between the two inbox banks.
        for tokens, (x, residual, weight, *ref), own in zip(TOKENS, cases, owned):
            for call in range(3):
                outs = reduce_scatter_rmsnorm_all_gather(
                    x, residual, weight, EPS, True, state
                )
                name = f"rank {rank} tokens {tokens} call {call}"
                failures += _compare(name, outs, *ref, own, tp.device_group)

        # All sizes in one HIP graph, replayed 10 times.
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            outs = [
                reduce_scatter_rmsnorm_all_gather(x, r, w, EPS, True, state)
                for x, r, w, _, _ in cases
            ]
        for _ in range(10):
            graph.replay()
        for tokens, o, (_, _, _, *ref), own in zip(TOKENS, outs, cases, owned):
            name = f"rank {rank} graph tokens {tokens}"
            failures += _compare(name, o, *ref, own, tp.device_group)
        if state.err.item():
            failures.append(f"rank {rank}: {state.err.item()} polls timed out")
    finally:
        reduce_scatter_rmsnorm_all_gather_free(group, state)
        aiter.destroy_dist_env()
    return failures


def test_reduce_scatter_rmsnorm_all_gather():
    ensure_spawn_method()
    init_method = get_distributed_init_method(get_ip(), get_open_port())
    with mp.Pool(processes=WORLD_SIZE) as pool:
        results = [
            pool.apply_async(_worker, (rank, init_method)) for rank in range(WORLD_SIZE)
        ]
        failures = [f for r in results for f in r.get(timeout=1800)]
    assert not failures, "\n".join(failures[:20])


def _dispatch_worker(rank, init_method):
    os.environ["AITER_PREFILL_RS_AG"] = "1"  # read when the TP communicator is made
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    aiter.init_dist_env(WORLD_SIZE, rank, distributed_init_method=init_method)
    tp = get_tp_group()
    failures = []
    try:
        state = tp.device_communicator.rs_rmsnorm_ag
        if state is None:
            failures.append(f"rank {rank}: the TP communicator has no RS + AG state")
        else:
            for tokens in (2049, 8116):
                x, residual, weight, *ref = _inputs(tokens, rank, device)
                outs = tensor_model_parallel_fused_allreduce_rmsnorm(
                    x, residual, weight, EPS, gemma_norm=True
                )
                own = _owned(tokens, rank, state.programs, device)
                name = f"rank {rank} dispatch tokens {tokens}"
                failures += _compare(name, outs, *ref, own, tp.device_group)
    finally:
        aiter.destroy_dist_env()
    return failures


def test_reduce_scatter_rmsnorm_all_gather_dispatch():
    ensure_spawn_method()
    init_method = get_distributed_init_method(get_ip(), get_open_port())
    with mp.Pool(processes=WORLD_SIZE) as pool:
        results = [
            pool.apply_async(_dispatch_worker, (rank, init_method))
            for rank in range(WORLD_SIZE)
        ]
        failures = [f for r in results for f in r.get(timeout=1800)]
    assert not failures, "\n".join(failures[:20])


if __name__ == "__main__":  # CI runs the multi-GPU tests as scripts
    if SKIP:
        logger.warning(
            "test_reduce_scatter_rmsnorm_all_gather skipped: needs 4 gfx1250 GPUs"
        )
    else:
        test_reduce_scatter_rmsnorm_all_gather()
        test_reduce_scatter_rmsnorm_all_gather_dispatch()
