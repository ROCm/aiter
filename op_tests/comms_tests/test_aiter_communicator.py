# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
"""Correctness of AiterCommunicator's collective ops — all_reduce and all_gather
— in both eager mode and under cudagraph capture + replay.

One question: does the iris-backed communicator produce correct results? Eager
alone is not enough — the gluon kernels elide barriers under graph capture, and a
race there only shows across a sequence of replays (vLLM captures the decode step
once and replays it every token). So every op is checked eager AND captured +
replayed back-to-back; passing both is what "the op works" means.
"""

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
from aiter.test_common import checkAllclose

logger = logging.getLogger("aiter")

set_start_method("spawn", force=True)

# Replays for the cudagraph correctness check. Back-to-back with no inter-replay
# sync is what stresses an elided end barrier (replay N+1 must not start before
# replay N's symmetric-heap writes land); a per-replay sync would hide the race.
NUM_REPLAYS = 1000

OPS = ["all_reduce", "all_gather"]


def _make_op(comm, op_name, x):
    """The collective under test as a zero-arg closure over the rank's input,
    after enforcing the communicator's own should_* precondition."""
    if op_name == "all_reduce":
        if not comm.should_allreduce(x):
            raise RuntimeError(
                f"AiterCommunicator rejected all_reduce: shape={tuple(x.shape)} dtype={x.dtype}"
            )
        return lambda: comm.all_reduce(x)
    if op_name == "all_gather":
        if not comm.should_allgather(x):
            raise RuntimeError(
                f"AiterCommunicator rejected all_gather: shape={tuple(x.shape)} dtype={x.dtype}"
            )
        return lambda: comm.all_gather(x)
    raise ValueError(f"unknown op {op_name!r}")


def run_comm(
    tp_size,
    pp_size,
    rankID,
    x,
    op_name,
    capture,
    distributed_init_method: Optional[str] = None,
):
    """One rank: init distributed, build the communicator, run `op_name` either
    eagerly or under cudagraph capture + NUM_REPLAYS back-to-back replays, and
    return the result for the driver to check against the reference."""
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

    op = _make_op(comm, op_name, x)

    # Warm up eagerly so first-call allocations (workspace, symmetric buffers)
    # happen before capture — graph capture can't perform them cleanly.
    for _ in range(3):
        out = op()
    torch.cuda.synchronize()

    if capture:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = op()
        for _ in range(NUM_REPLAYS):
            graph.replay()
        torch.cuda.synchronize()

    result = out.clone()

    if dist.is_initialized():
        destroy_model_parallel()
        destroy_distributed_environment()
        torch.cuda.empty_cache()
    return result


def reference(op_name, inputs, dim=-1):
    """What every rank should hold afterwards. all_reduce = elementwise sum
    (accumulated in fp32 so the reference itself doesn't eat bf16 rounding);
    all_gather = concat of the per-rank inputs along `dim`, rank-ordered (the
    AiterCommunicator.all_gather contract)."""
    if op_name == "all_reduce":
        acc = torch.zeros_like(inputs[0], dtype=torch.float32)
        for x in inputs:
            acc += x.to(torch.float32)
        return acc.to(inputs[0].dtype)
    if op_name == "all_gather":
        return torch.cat(inputs, dim=dim)
    raise ValueError(f"unknown op {op_name!r}")


def tolerance(op_name, dtype):
    """all_gather is pure data movement → effectively exact. all_reduce sums
    world_size values, and bf16's 7-bit mantissa (ULP ~8x fp16's) makes tree-vs-
    sequential accumulation diverge by a few ULPs — benign, but it needs a
    dtype-aware absolute tolerance so a *correct* bf16 reduce isn't flagged. A
    real reduction bug produces garbage orders of magnitude beyond this."""
    if op_name == "all_gather":
        return 1e-3
    return 0.1 if dtype == torch.bfloat16 else 0.01


def test_communicator(
    tp_size,
    pp_size,
    shape,
    dtype,
    op_name,
    capture,
    distributed_init_method: Optional[str] = None,
):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "49373"
    inputs = [torch.randn(shape, dtype=dtype) for _ in range(tp_size)]
    ref = reference(op_name, inputs)
    pool = Pool(processes=tp_size)
    rets = [
        pool.apply_async(
            run_comm,
            args=(tp_size, pp_size, i, inputs[i], op_name, capture, distributed_init_method),
        )
        for i in range(tp_size)
    ]
    pool.close()
    pool.join()
    rets = [el.get() for el in rets]
    mode = "cudagraph" if capture else "eager"
    atol = tolerance(op_name, dtype)
    for out in rets:
        msg = f"AiterCommunicator.{op_name} [{mode}]: {shape=} {dtype=}"
        checkAllclose(ref, out.to(ref), atol=atol, rtol=0.01, msg=msg)


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
    # Every collective the communicator offers, eager then cudagraph capture +
    # replay. No modes to pick — run it; the checkAllclose lines are the answer.
    for op_name in OPS:
        for dtype in l_dtype:
            for shape in l_shape:
                for capture in (False, True):
                    test_communicator(
                        8, 1, shape, dtype, op_name, capture,
                        distributed_init_method=get_distributed_init_method(
                            "127.0.0.1", get_open_port()
                        ),
                    )
