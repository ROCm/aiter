# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import argparse
import logging
import os
from multiprocessing import Pool, freeze_support, set_start_method

import pandas as pd
import torch
import torch.distributed as dist

from aiter import dtypes
from aiter.dist.communication_op import tensor_model_parallel_all_reduce
from aiter.dist.parallel_state import (
    destroy_distributed_environment,
    destroy_model_parallel,
    ensure_model_parallel_initialized,
    get_tp_group,
    graph_capture,
    init_distributed_environment,
    set_custom_all_reduce,
)
from aiter.dist.utils import get_distributed_init_method, get_ip, get_open_port
from aiter.test_common import benchmark, checkAllclose, perftest

logger = logging.getLogger("aiter")

set_start_method("spawn", force=True)


def allreduce_custom(
    tp_size,
    pp_size,
    rankID,
    x,
    withGraph=False,
    distributed_init_method: str | None = None,
):
    device = torch.device(f"cuda:{rankID}")
    torch.cuda.set_device(device)
    # init
    logger.info(f"RANK: {rankID} {tp_size} init_process_group...")
    set_custom_all_reduce(True)
    init_distributed_environment(
        world_size=tp_size,
        rank=rankID,
        distributed_init_method=distributed_init_method,
    )
    ensure_model_parallel_initialized(tp_size, pp_size)
    x = x.to(device)
    # dist.barrier(device_ids=[i for i in range(tp_size)])

    # warmup and align all gpu
    group = get_tp_group().device_group
    dist.all_reduce(torch.zeros(1).cuda(), group=group)
    torch.cuda.synchronize()

    if withGraph:
        graph = torch.cuda.CUDAGraph()
        with graph_capture() as gc, torch.cuda.graph(graph, stream=gc.stream):
            out = tensor_model_parallel_all_reduce(x)
        out.fill_(0)

        @perftest()
        def run_ca():
            graph.replay()

        _, us = run_ca()
        out = (out, us)
    else:

        @perftest()
        def run_ca(x):
            return tensor_model_parallel_all_reduce(x)

        out = run_ca(x)

    # destroy
    if dist.is_initialized():
        destroy_model_parallel()
        destroy_distributed_environment()
        torch.cuda.empty_cache()
    return out


@benchmark()
def test_allreduce_custom(
    tp_size,
    pp_size,
    shape,
    dtype,
    withGraph=False,
    distributed_init_method: str | None = None,
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
                allreduce_custom,
                args=(tp_size, pp_size, i, x, withGraph, distributed_init_method),
            )
        )
    pool.close()
    pool.join()
    rets = [el.get() for el in rets]
    all_us = [us for _, us in rets]
    max_err = 0.0
    for out, us in rets:
        msg = f"test_allreduce_custom: {shape=} {dtype=} {withGraph=} {us:>8.2f}"
        err = checkAllclose(ref, out.to(ref), msg=msg)
        max_err = max(max_err, err)
    return {
        "min_us": min(all_us),
        "max_us": max(all_us),
        "err": max_err,
    }


def _allreduce_tail_regression(
    rank, tp_size, shapes, dtype, iterations, distributed_init_method
):
    from aiter.dist.device_communicators.custom_all_reduce import CustomAllreduce

    torch.set_num_threads(1)
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")
    dist.init_process_group(
        "gloo",
        world_size=tp_size,
        rank=rank,
        init_method=distributed_init_method,
    )
    ca_comm = None
    try:
        ca_comm = CustomAllreduce(dist.group.WORLD, device)
        assert not ca_comm.disabled, "The tail regression requires custom allreduce"

        for shape in shapes:
            x = torch.zeros(shape, dtype=dtype, device=device)
            assert ca_comm.should_custom_ar(x), f"Custom allreduce rejected {shape=}"
            # Warm up the actual eager path before capturing its graph variant.
            ca_comm.custom_all_reduce(x, use_new=True)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with ca_comm.capture(), torch.cuda.graph(graph, stream=stream):
                # Register a graph-owned intermediate, as in a captured model,
                # while the external input remains mutable between replays.
                graph_out = ca_comm.custom_all_reduce(x.clone(), use_new=True)
            torch.cuda.current_stream().wait_stream(stream)
            assert graph_out is not None

            generator = torch.Generator().manual_seed(2026)
            for iteration in range(iterations):
                # Independent rank inputs, with sums exactly representable even
                # in BF16. A FP32 reference avoids tolerances hiding stale data.
                inputs = torch.randint(
                    -8, 9, (tp_size, *shape), generator=generator, dtype=torch.int16
                )
                ref = inputs.sum(dim=0, dtype=torch.float32).to(device, dtype)
                x.copy_(inputs[rank])
                eager_out = ca_comm.custom_all_reduce(x, use_new=True)
                assert eager_out is not None
                context = f"{rank=} {shape=} {dtype=} {iteration=}"
                torch.testing.assert_close(
                    eager_out,
                    ref,
                    rtol=0,
                    atol=0,
                    msg=lambda msg, context=context: f"{context} eager\n{msg}",
                )
                # Change the input on every replay and poison the output so
                # replaying stale data or skipping a tail store cannot pass.
                graph_out.fill_(float("nan"))
                graph.replay()
                torch.testing.assert_close(
                    graph_out,
                    ref,
                    rtol=0,
                    atol=0,
                    msg=lambda msg, context=context: f"{context} graph\n{msg}",
                )
            if rank == 0:
                logger.info(
                    "tail regression passed: %s, %s, TP%d, %d eager + graph runs",
                    shape,
                    dtype,
                    tp_size,
                    iterations,
                )
        # All ranks must finish reading peer buffers before any are freed.
        dist.barrier()
    finally:
        if ca_comm is not None:
            ca_comm.close()
        dist.destroy_process_group()


def test_allreduce_tail_regression(tp_size, dtype, iterations=20, shapes=None):
    # 1025x512 reproduced a TP4 BF16 graph-replay failure. The adjacent
    # dimensions exercise both partial and complete blocks in reduce-scatter.
    if shapes is None:
        shapes = [(1023, 512), (1024, 512), (1025, 512)]
        if tp_size in (4, 8):
            pack_size = 16 // torch.empty((), dtype=dtype).element_size()
            # The optimized kernel assigns 512 / tp_size packs to each rank
            # per tile. Check one active lane, one inactive lane, and a size
            # not divisible across ranks (the naive path). These small cases
            # use an uncapped grid; the original shapes exceed the grid cap.
            # 32 tiles is above the two-stage threshold for both TP4 and TP8.
            base_packs = 32 * 512
            shapes.extend(
                ((base_packs + offset) * pack_size,)
                for offset in (-tp_size, tp_size, 1)
            )
    torch.multiprocessing.spawn(
        _allreduce_tail_regression,
        args=(
            tp_size,
            shapes,
            dtype,
            iterations,
            get_distributed_init_method(get_ip(), get_open_port()),
        ),
        nprocs=tp_size,
        join=True,
    )


# Custom-AR size cutoff (bytes). Mirrors _DEFAULT_CAR_MAX_SIZE in
# aiter/dist/device_communicators/custom_all_reduce.py and honors the same
# AITER_CUSTOM_AR_MAX_SIZE override. Inputs at or below this run on the custom
# kernels (larger ones fall back to RCCL), so the swept size list stops here.
_DEFAULT_CAR_MAX_BYTES = 8192 * 8192


def _car_max_bytes() -> int:
    e = os.environ.get("AITER_CUSTOM_AR_MAX_SIZE", "")
    try:
        v = int(e)
        if v > 0:
            return v
    except ValueError:
        pass
    return _DEFAULT_CAR_MAX_BYTES


def gen_sizes(dtype) -> list[int]:
    """Element counts to sweep: [1024, 2048, 4096] then 7168*k / 8192*k for
    k = 1, 2, 4, 8, ... plus the flattened (1023/1024/1025, 512) tail cases,
    up to the largest size custom_all_reduce serves."""
    itemsize = torch.empty(0, dtype=dtype).element_size()
    max_numel = _car_max_bytes() // itemsize
    sizes = [n for n in (1024, 2048, 4096) if n <= max_numel]
    k = 1
    while True:
        added = False
        for base in (7168, 8192):
            n = base * k
            if n <= max_numel:
                sizes.append(n)
                added = True
        if not added:
            break
        k *= 2
    # Include partial tiles around the aligned 1024x512 case. Flattening the
    # contiguous inputs preserves the kernel dispatch and reported tail shape.
    sizes.extend(n for n in (1023 * 512, 1024 * 512, 1025 * 512) if n <= max_numel)
    return sorted(set(sizes))


l_dtype = ["bf16", "fp16", "fp32"]

parser = argparse.ArgumentParser(description="config input of test")
parser.add_argument(
    "-d",
    "--dtype",
    type=str,
    choices=l_dtype,
    default="bf16",
    help="data type (default: bf16)",
)
parser.add_argument(
    "-t",
    "--tp-size",
    type=int,
    choices=[2, 4, 6, 8],
    default=8,
    help="number of GPUs / tensor-parallel size (default: 8)",
)
parser.add_argument(
    "-m",
    "--mode",
    type=str,
    choices=["graph", "eager"],
    default="graph",
    help="execution mode (default: graph)",
)
parser.add_argument(
    "-s",
    "--shape",
    type=dtypes.str2tuple,
    nargs="?",
    const=None,
    default=None,
    help="single shape override, e.g. -s 128,8192 (default: swept size list)",
)
parser.add_argument(
    "--regression",
    action="store_true",
    help="run only tail correctness checks in both eager and graph modes",
)
parser.add_argument(
    "--iterations",
    type=int,
    default=20,
    help="number of changing-input iterations per regression shape (default: 20)",
)


if __name__ == "__main__":
    freeze_support()
    args = parser.parse_args()
    if args.iterations < 1:
        parser.error("--iterations must be positive")
    dtype = dtypes.d_dtypes[args.dtype]
    if args.regression or args.shape is None:
        test_allreduce_tail_regression(
            args.tp_size,
            dtype,
            args.iterations,
            shapes=[args.shape] if args.shape is not None else None,
        )
    if args.regression:
        raise SystemExit(0)
    with_graph = args.mode == "graph"
    if args.shape is not None:
        l_shape = [args.shape]
    else:
        l_shape = gen_sizes(dtype)
    df = []
    for shape in l_shape:
        ret = test_allreduce_custom(
            args.tp_size,
            1,
            shape,
            dtype,
            withGraph=with_graph,
            distributed_init_method=get_distributed_init_method(
                get_ip(), get_open_port()
            ),
        )
        df.append(ret)
    df = pd.DataFrame(df)
    show_cols = [
        "tp_size",
        "shape",
        "dtype",
        "withGraph",
        "min_us",
        "max_us",
        "err",
    ]
    show_cols = [c for c in show_cols if c in df.columns]
    logger.info(
        "custom allreduce summary (markdown):\n%s",
        df[show_cols].to_markdown(index=False),
    )
