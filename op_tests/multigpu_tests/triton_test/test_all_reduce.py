# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import ctypes
import multiprocessing as mp

import pytest
import torch
import torch.distributed as dist

import aiter
from aiter import logger
from aiter.dist.cuda_wrapper import find_loaded_library
from aiter.dist.parallel_state import get_tp_group
from aiter.dist.utils import get_distributed_init_method, get_ip, get_open_port
from aiter.ops.triton.comms.all_reduce import (
    _LINES,
    all_reduce,
    all_reduce_free,
    all_reduce_init,
)
from aiter.ops.triton.utils._triton import arch_info
from aiter.test_common import ensure_spawn_method

WORLD_SIZE = 4
HIDDEN = 2048
# Token counts of decode and prefill, then 2 sizes with a partial last chunk.
NUMELS = [t * HIDDEN for t in (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048)]
NUMELS += [3 * HIDDEN + 8, 600 * HIDDEN + 8]
MIXED = [8, 1, 2048, 2, 64, 4, 512, 8, 16, 1024, 32, 128, 256, 1, 8]

SKIP = torch.cuda.device_count() < WORLD_SIZE or arch_info.get_arch() != "gfx1250"
pytestmark = pytest.mark.skipif(SKIP, reason="needs 4 gfx1250 GPUs")


def _inputs(numel, seed, rank, device):
    """This rank's input and the fp32 sum of all inputs in rank order, as bf16."""
    gen = torch.Generator(device=device).manual_seed(seed)
    xs = [
        torch.randn(numel, generator=gen, dtype=torch.bfloat16, device=device)
        for _ in range(WORLD_SIZE)
    ]
    ref = xs[0].float() + xs[1].float() + xs[2].float() + xs[3].float()
    return xs[rank], ref.to(torch.bfloat16)


def _bad(out, ref):
    return int((out.view(torch.int16) != ref.view(torch.int16)).sum())


def _stale_tail(rank, group, inboxes, calls, call):
    """After an odd call on 3.1 chunks: does a unit of chunk 3 that a peer sent into
    bank 1 of this rank's inbox carry another tag?"""
    torch.cuda.synchronize()
    dist.barrier(group=group)  # the peers have finished the call
    hip = ctypes.CDLL(find_loaded_library("libamdhip64"))
    slot = 4 * _LINES * ((calls.numel() - 1) // 2)  # int32 words, as in all_reduce
    units = (ctypes.c_uint16 * (8 * _LINES))()  # 3 values and the tag a unit
    for peer in range(WORLD_SIZE):
        if peer != rank:
            src = inboxes[rank] + 4 * ((4 + peer) * slot + 4 * 3 * _LINES)
            assert hip.hipMemcpy(units, ctypes.c_void_p(src), 16 * _LINES, 2) == 0
            if any(tag != call % 32768 + 1 for tag in units[3::4]):
                return True
    return False


def _worker(rank, init_method):
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    aiter.init_dist_env(WORLD_SIZE, rank, distributed_init_method=init_method)
    tp = get_tp_group()
    group = tp.cpu_group
    inboxes, calls = all_reduce_init(group, 2 * max(NUMELS))
    failures = []
    try:
        # First, while the inboxes are empty: the call numbers wrap from 2^30 to 1.
        calls[:-1].fill_(2**30 - 2)
        for i in range(4):
            x, ref = _inputs(64 * HIDDEN, 2000 + i, rank, device)
            if _bad(all_reduce(x, group, inboxes, calls), ref):
                failures.append(f"rank {rank}: call {i} at the counter wrap is wrong")

        # A last dim that is odd in int32 words.
        x, ref = _inputs(12, 3000, rank, device)
        if _bad(all_reduce(x.view(4, 3), group, inboxes, calls), ref.view(4, 3)):
            failures.append(f"rank {rank}: shape (4, 3) is wrong")

        for i, numel in enumerate(NUMELS):
            x, ref = _inputs(numel, i, rank, device)
            out = all_reduce(x, group, inboxes, calls)
            outs = [torch.empty_like(out) for _ in range(WORLD_SIZE)]
            dist.all_gather(outs, out, group=tp.device_group)
            if _bad(out, ref) or any(_bad(o, out) for o in outs):
                failures.append(f"rank {rank} numel {numel}: wrong or rank-dependent")

        # Mixed sizes back to back, then the same calls in a HIP graph.
        cases = [
            _inputs(t * HIDDEN, 100 + i, rank, device) for i, t in enumerate(MIXED)
        ]
        outs = [all_reduce(x, group, inboxes, calls) for x, _ in cases]
        if sum(_bad(o, ref) for o, (_, ref) in zip(outs, cases)):
            failures.append(f"rank {rank}: mixed sizes are wrong")
        xs = [torch.empty_like(x) for x, _ in cases]
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            outs = [all_reduce(x, group, inboxes, calls) for x in xs]
        for rep in range(20):
            refs = []
            for i, (t, x) in enumerate(zip(MIXED, xs)):
                new_x, ref = _inputs(t * HIDDEN, 1000 * rep + i, rank, device)
                x.copy_(new_x)
                refs.append(ref)
            graph.replay()
            if sum(_bad(o, ref) for o, ref in zip(outs, refs)):
                failures.append(f"rank {rank}: graph replay {rep} is wrong")

        # Last, with full inboxes: 4 chunks at call 1, then 3.1 chunks across the
        # counter wrap, then 4 chunks at call 1 again. Chunk 3 must not keep call 1.
        full, part = 4 * 6 * _LINES, 3 * 6 * _LINES + 20  # 6 values a line
        for step, (numel, start) in enumerate(
            [(full, 0), (part, 2**30 - 2), (part, None), (full, None)]
        ):
            if start is not None:
                calls[:-1].fill_(start)
            x, ref = _inputs(numel, 4000 + step, rank, device)
            if _bad(all_reduce(x, group, inboxes, calls), ref):
                failures.append(f"rank {rank}: full-partial-full step {step} is wrong")
            if step == 1 and _stale_tail(rank, group, inboxes, calls, 2**30 - 1):
                failures.append(f"rank {rank}: the tail of chunk 3 kept an old call")
        if calls[-1].item():
            failures.append(f"rank {rank}: {calls[-1].item()} polls timed out")
        return failures
    finally:
        all_reduce_free(group, inboxes)
        aiter.destroy_dist_env()


def test_all_reduce():
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
        logger.warning("test_all_reduce skipped: needs 4 gfx1250 GPUs")
    else:
        test_all_reduce()
