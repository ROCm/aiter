# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import multiprocessing as mp
import os

import pytest
import torch
import torch.distributed as dist

import aiter
from aiter import logger
from aiter.dist.communication_op import (
    tensor_model_parallel_all_reduce,
    tensor_model_parallel_fused_allreduce_rmsnorm,
)
from aiter.dist.parallel_state import get_tp_group
from aiter.dist.utils import get_distributed_init_method, get_ip, get_open_port
from aiter.ops.triton.comms.all_reduce import all_reduce_free, all_reduce_init
from aiter.ops.triton.comms.fused.all_reduce_rmsnorm_quant import (
    all_reduce_rmsnorm_quant,
)
from aiter.ops.triton.utils._triton import arch_info
from aiter.test_common import ensure_spawn_method

WORLD_SIZE = 4
HIDDEN = 2048
EPS = 1e-6
TOKENS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]

SKIP = torch.cuda.device_count() < WORLD_SIZE or arch_info.get_arch() != "gfx1250"
pytestmark = pytest.mark.skipif(SKIP, reason="needs 4 gfx1250 GPUs")


def _inputs(tokens, rank, device):
    """This rank's x, the residual and the weight, and the fp32 references of
    residual_out and out. The 4 inputs are added in rank order, as in the kernel."""
    gen = torch.Generator(device=device).manual_seed(tokens)
    shape = (tokens, HIDDEN)
    xs = [
        torch.randn(shape, generator=gen, dtype=torch.bfloat16, device=device)
        for _ in range(WORLD_SIZE)
    ]
    residual = torch.randn(shape, generator=gen, dtype=torch.bfloat16, device=device)
    w = torch.randn(HIDDEN, generator=gen, dtype=torch.bfloat16, device=device)
    weight = 0.1 * w  # a Gemma weight: 1 + weight is near 1
    s = xs[0].float() + xs[1].float() + xs[2].float() + xs[3].float() + residual.float()
    out = s * torch.rsqrt(s.pow(2).mean(-1, keepdim=True) + EPS) * (1 + weight.float())
    return xs[rank], residual, weight, s, out


def _compare(name, outs, ref_residual, ref_out, group):
    failures = []
    if not torch.equal(outs[1], ref_residual.to(torch.bfloat16)):
        failures.append(f"{name}: residual_out is not bit-exact")
    try:
        torch.testing.assert_close(outs[0], ref_out.to(torch.bfloat16))
    except AssertionError as e:
        failures.append(f"{name}: out: {e}")
    if len(outs) == 4:
        step = outs[3].repeat_interleave(128, dim=1)
        error = (outs[2].float() * step - ref_out).abs()
        # e4m3 has 3 mantissa bits. Its smallest step is 2^-9 of the scale.
        if not bool((error <= 2.0**-4 * ref_out.abs() + 2.0**-9 * step).all()):
            failures.append(f"{name}: fp8 values are wrong")
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
            x, residual, weight, *ref = _inputs(64 + i, rank, device)
            outs = all_reduce_rmsnorm_quant(
                x, residual, weight, EPS, True, i % 2 == 1, group, inboxes, calls
            )
            name = f"rank {rank} call {i} at the counter wrap"
            failures += _compare(name, outs, *ref, tp.device_group)

        for quant in (False, True):
            for tokens, (x, residual, weight, *ref) in zip(TOKENS, cases):
                outs = all_reduce_rmsnorm_quant(
                    x, residual, weight, EPS, True, quant, group, inboxes, calls
                )
                name = f"rank {rank} tokens {tokens} quant {quant}"
                failures += _compare(name, outs, *ref, tp.device_group)

            # All token counts in one HIP graph, replayed 10 times.
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                outs = [
                    all_reduce_rmsnorm_quant(
                        x, r, w, EPS, True, quant, group, inboxes, calls
                    )
                    for x, r, w, _, _ in cases
                ]
            for _ in range(10):
                graph.replay()
            for tokens, o, (_, _, _, *ref) in zip(TOKENS, outs, cases):
                name = f"rank {rank} graph tokens {tokens} quant {quant}"
                failures += _compare(name, o, *ref, tp.device_group)
        if calls[-1].item():
            failures.append(f"rank {rank}: {calls[-1].item()} polls timed out")
    finally:
        all_reduce_free(group, inboxes)
        aiter.destroy_dist_env()
    return failures


def test_all_reduce_rmsnorm_quant():
    ensure_spawn_method()
    init_method = get_distributed_init_method(get_ip(), get_open_port())
    with mp.Pool(processes=WORLD_SIZE) as pool:
        results = [
            pool.apply_async(_worker, (rank, init_method)) for rank in range(WORLD_SIZE)
        ]
        failures = [f for r in results for f in r.get(timeout=1800)]
    assert not failures, "\n".join(failures[:20])


def _dispatch_worker(rank, init_method):
    os.environ["AITER_GLUON_FUSED_AR"] = "1"  # read when the TP communicator is made
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    aiter.init_dist_env(WORLD_SIZE, rank, distributed_init_method=init_method)
    tp = get_tp_group()
    failures = []
    try:
        if tp.device_communicator.gluon_ar is None:
            failures.append(f"rank {rank}: the TP communicator has no Gluon inboxes")
        for tokens in (1, 8, 128, 1024, 2048):  # 2048 rows at hidden 2048 are 8 MiB
            x, residual, weight, *ref = _inputs(tokens, rank, device)
            outs = tensor_model_parallel_fused_allreduce_rmsnorm(
                x, residual, weight, EPS, gemma_norm=True
            )
            name = f"rank {rank} dispatch tokens {tokens}"
            failures += _compare(name, outs, *ref, tp.device_group)

        # The plain all-reduce, up to 4 MiB. Bit-exact only on the Gluon path.
        if tp.device_communicator.gluon_ar_plain is None:
            failures.append(f"rank {rank}: no Gluon inboxes for the plain all-reduce")
        for tokens in (1, 8, 128, 1024):  # 1024 rows at hidden 2048 are 4 MiB
            gen = torch.Generator(device=device).manual_seed(1000 + tokens)
            shape = (tokens, HIDDEN)
            xs = [
                torch.randn(shape, generator=gen, dtype=torch.bfloat16, device=device)
                for _ in range(WORLD_SIZE)
            ]
            ref = xs[0].float() + xs[1].float() + xs[2].float() + xs[3].float()
            out = tensor_model_parallel_all_reduce(xs[rank])
            bad = out.view(torch.int16) != ref.to(torch.bfloat16).view(torch.int16)
            if bool(bad.any()):
                failures.append(f"rank {rank} plain dispatch tokens {tokens}: wrong")
    finally:
        aiter.destroy_dist_env()
    return failures


def test_fused_allreduce_rmsnorm_dispatch():
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
        logger.warning("test_all_reduce_rmsnorm_quant skipped: needs 4 gfx1250 GPUs")
    else:
        test_all_reduce_rmsnorm_quant()
