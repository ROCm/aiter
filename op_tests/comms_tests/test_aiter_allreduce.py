# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import argparse
import logging
import os
from multiprocessing import Pool, freeze_support, set_start_method
from typing_extensions import Optional

import torch
import torch.distributed as dist

from aiter import dtypes
from aiter.dist.parallel_state import (
    destroy_distributed_environment,
    destroy_model_parallel,
    ensure_model_parallel_initialized,
    get_tp_group,
    init_distributed_environment,
)
from aiter.dist.utils import get_distributed_init_method, get_open_port
from aiter.test_common import benchmark, checkAllclose, perftest

logger = logging.getLogger("aiter")

set_start_method("spawn", force=True)


def allreduce_aiter(
    tp_size,
    pp_size,
    rankID,
    x,
    distributed_init_method: Optional[str] = None,
):
    device = torch.device(f"cuda:{rankID}")
    torch.cuda.set_device(device)

    init_distributed_environment(
        world_size=tp_size,
        rank=rankID,
        distributed_init_method=distributed_init_method,
    )
    ensure_model_parallel_initialized(tp_size, pp_size)
    x = x.to(device)

    group = get_tp_group().device_group
    dist.all_reduce(torch.zeros(1).cuda(), group=group)
    torch.cuda.synchronize()

    from aiter.ops.triton.comms.communicator import AiterCommunicator

    comm = AiterCommunicator(group=group, device=device)
    if comm.disabled:
        raise RuntimeError(f"AiterCommunicator disabled on rank {rankID}")

    if not comm.should_allreduce(x):
        raise RuntimeError(
            f"AiterCommunicator rejected tensor: shape={tuple(x.shape)} "
            f"dtype={x.dtype} size={x.numel() * x.element_size()}"
        )

    @perftest()
    def run_ar(x):
        return comm.all_reduce(x)

    out = run_ar(x)

    if dist.is_initialized():
        destroy_model_parallel()
        destroy_distributed_environment()
        torch.cuda.empty_cache()
    return out


def allreduce_aiter_cudagraph(
    tp_size,
    pp_size,
    rankID,
    x,
    num_replays,
    distributed_init_method: Optional[str] = None,
):
    """Capture the AiterCommunicator all-reduce into a cudagraph and replay it
    many times back-to-back, then return the final output.

    This is the path eager testing can't reach. The gluon kernel elides its end
    barrier "under graph capture, when the next replay's start barrier covers
    the writes" — an optimization that only activates while capturing and whose
    correctness assumption is only tested across a *sequence* of replays (exactly
    vLLM's decode loop: capture the step once, replay every token). Critically we
    do NOT synchronize between replays — a per-replay sync would serialize the
    ranks and mask the very race we're trying to surface (replay N+1 reducing
    before replay N's symmetric-heap writes have landed on all peers).
    """
    device = torch.device(f"cuda:{rankID}")
    torch.cuda.set_device(device)

    init_distributed_environment(
        world_size=tp_size,
        rank=rankID,
        distributed_init_method=distributed_init_method,
    )
    ensure_model_parallel_initialized(tp_size, pp_size)
    x = x.to(device)

    group = get_tp_group().device_group
    dist.all_reduce(torch.zeros(1).cuda(), group=group)
    torch.cuda.synchronize()

    from aiter.ops.triton.comms.communicator import AiterCommunicator

    comm = AiterCommunicator(group=group, device=device)
    if comm.disabled:
        raise RuntimeError(f"AiterCommunicator disabled on rank {rankID}")
    if not comm.should_allreduce(x):
        raise RuntimeError(
            f"AiterCommunicator rejected tensor: shape={tuple(x.shape)} "
            f"dtype={x.dtype} size={x.numel() * x.element_size()}"
        )

    # Warm up eagerly so first-call allocations (workspace, symmetric buffers)
    # happen before capture — graph capture can't perform them cleanly.
    for _ in range(3):
        out = comm.all_reduce(x)
    torch.cuda.synchronize()

    # Capture the all-reduce; `out` is now the static buffer replays write to.
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = comm.all_reduce(x)

    # Replay back-to-back, no inter-replay sync (see docstring). Each replay
    # reduces the same fixed input, so the result must equal `ref` every time;
    # an elided-barrier race corrupts the steady state and the final `out` drifts.
    for _ in range(num_replays):
        graph.replay()
    torch.cuda.synchronize()

    result = out.clone()

    if dist.is_initialized():
        destroy_model_parallel()
        destroy_distributed_environment()
        torch.cuda.empty_cache()
    return result


@benchmark()
def test_allreduce_aiter(
    tp_size,
    pp_size,
    shape,
    dtype,
    distributed_init_method: Optional[str] = None,
):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "49373"
    pool = Pool(processes=tp_size)
    ref = torch.zeros(shape, dtype=dtype)
    rets = []
    for i in range(tp_size):
        x = torch.randn(shape, dtype=dtype)
        ref += x
        rets.append(
            pool.apply_async(
                allreduce_aiter,
                args=(tp_size, pp_size, i, x, distributed_init_method),
            )
        )
    pool.close()
    pool.join()
    rets = [el.get() for el in rets]
    for out, us in rets:
        msg = f"test_allreduce_aiter: {shape=} {dtype=} {us:>8.2f}"
        checkAllclose(ref, out.to(ref), msg=msg)


def test_allreduce_aiter_cudagraph(
    tp_size,
    pp_size,
    shape,
    dtype,
    num_replays=1000,
    distributed_init_method: Optional[str] = None,
):
    """Correctness of the all-reduce under cudagraph capture + many replays —
    the serving regime the eager test above can't reach. Same reference as the
    eager test (sum of the per-rank inputs); each rank captures the AR and
    replays it `num_replays` times, and the replayed output must still match."""
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "49373"
    pool = Pool(processes=tp_size)
    ref = torch.zeros(shape, dtype=dtype)
    rets = []
    for i in range(tp_size):
        x = torch.randn(shape, dtype=dtype)
        ref += x
        rets.append(
            pool.apply_async(
                allreduce_aiter_cudagraph,
                args=(tp_size, pp_size, i, x, num_replays, distributed_init_method),
            )
        )
    pool.close()
    pool.join()
    rets = [el.get() for el in rets]
    for out in rets:
        msg = f"test_allreduce_aiter_cudagraph: {shape=} {dtype=} replays={num_replays}"
        checkAllclose(ref, out.to(ref), msg=msg)


l_dtype = ["fp16", "bf16"]
l_shape = [(4, 8192), (128, 8192), (256, 8192)]

parser = argparse.ArgumentParser(description="config input of test")
parser.add_argument(
    "-d",
    "--dtype",
    type=str,
    choices=l_dtype,
    nargs="?",
    const=None,
    default=None,
    help="data type",
)
parser.add_argument(
    "-s",
    "--shape",
    type=dtypes.str2tuple,
    nargs="?",
    const=None,
    default=None,
    help="shape. e.g. -s 128,8192",
)


if __name__ == "__main__":
    freeze_support()
    args = parser.parse_args()
    if args.dtype is None:
        l_dtype = [dtypes.d_dtypes[key] for key in l_dtype]
    else:
        l_dtype = [dtypes.d_dtypes[args.dtype]]
    if args.shape is not None:
        l_shape = [args.shape]
    # Eager first (fast sanity that the reduction math is right), then under
    # cudagraph capture + replay (the regime that actually ships). Both must
    # pass for the kernel to "work" — no modes to pick, just run it.
    for dtype in l_dtype:
        for shape in l_shape:
            test_allreduce_aiter(
                8, 1, shape, dtype,
                distributed_init_method=get_distributed_init_method("127.0.0.1", get_open_port()),
            )
            test_allreduce_aiter_cudagraph(
                8, 1, shape, dtype,
                distributed_init_method=get_distributed_init_method("127.0.0.1", get_open_port()),
            )
