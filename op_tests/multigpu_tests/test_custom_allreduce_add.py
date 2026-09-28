# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Exact-result tests for CustomAllreduce.all_reduce_add and the all-reduce
kernel dispatch around its size cutovers.

For every size, in eager and CUDA-graph mode:
  * all_reduce(x) and all_reduce_add(x, y) on integer-valued inputs must equal
    an fp32 reference exactly;
  * all_reduce_add(x, y) on random inputs must be bitwise equal to
    all_reduce(x + y).
A final graph interleaves plain and fused calls of different sizes, since the
LL, LL128 and 2-stage kernels share scratch and signal state.
"""

import argparse
import logging

import torch
import torch.distributed as dist

from aiter import dtypes
from aiter.dist.utils import get_distributed_init_method, get_ip, get_open_port

logger = logging.getLogger("aiter")

KB = 1024
# Each cutover and one step past it, for sizes divisible by world_size * 16
# (+0 / +128 B) and not divisible (-16 / +16 B, 1-stage or naive 2-stage).
# gfx950 aligned cutovers: plain LL 40 / 64 KB, fused LL 64 / 80 / 96 KB,
# fused LL128 384 KB. Unaligned: plain LL 64 / 80 KB, fused LL 96 / 112 KB,
# fused LL128 160 / 768 KB, 1-stage 96 / 256 KB. Original 1-stage limits:
# 80 / 160 KB.
SIZES_BYTES = sorted(
    {16 * KB, 32 * KB + 16, 1024 * KB + 16, 4096 * KB}
    | {cap * KB + off for cap in (40, 64, 80, 96, 160, 384, 1024) for off in (0, 128)}
    | {cap * KB + off for cap in (64, 80, 96, 112, 160, 256, 768) for off in (-16, 16)}
)
INTERLEAVE_BYTES = [16 * KB, 80 * KB, 64 * KB + 128, 384 * KB, 1024 * KB]


def _capture(fn, stream):
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        fn()
    return graph


def _int_inputs(gen, n_inputs, tp_size, numel):
    # Sums of up to 16 values in [-8, 8] are exact in bf16.
    return torch.randint(
        -8, 9, (n_inputs, tp_size, numel), generator=gen, dtype=torch.int16
    )


def _check_size(ca, rank, tp_size, dtype, numel, iterations, gen, keep):
    device = torch.device(f"cuda:{rank}")
    x = torch.empty(numel, dtype=dtype, device=device)
    y = torch.empty_like(x)
    assert ca.should_custom_ar_add(x, y), f"custom AR rejected {numel=} {dtype=}"
    # Graph outputs are IPC-registered with address dedup, and every rank must
    # register the same count. Preallocated outputs keep that rank-independent.
    out_plain, out_add, out_ref = (torch.empty_like(x) for _ in range(3))

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with ca.capture():
        g_plain = _capture(
            lambda: ca.all_reduce(x, out=out_plain, registered_input=True), stream
        )
        g_add = _capture(lambda: ca.all_reduce_add(x, y, out=out_add), stream)
        g_ref = _capture(
            lambda: ca.all_reduce(x + y, out=out_ref, registered_input=True), stream
        )
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    keep.append((x, y, out_plain, out_add, out_ref, g_plain, g_add, g_ref))

    def replay(graph, out):
        out.fill_(float("nan"))
        graph.replay()
        return out

    for it in range(iterations):
        ctx = f"{rank=} {numel=} {dtype=} {it=}"
        ints = _int_inputs(gen, 2, tp_size, numel)
        ref_plain = ints[0].sum(0, dtype=torch.float32).to(device, dtype)
        ref_add = ints.sum((0, 1), dtype=torch.float32).to(device, dtype)
        x.copy_(ints[0, rank])
        y.copy_(ints[1, rank])
        for name, got, ref in (
            ("eager all_reduce", ca.custom_all_reduce(x), ref_plain),
            ("eager all_reduce_add", ca.custom_all_reduce_add(x, y), ref_add),
            ("graph all_reduce", replay(g_plain, out_plain), ref_plain),
            ("graph all_reduce_add", replay(g_add, out_add), ref_add),
        ):
            torch.testing.assert_close(
                got, ref, rtol=0, atol=0, msg=lambda m: f"{ctx} {name}\n{m}"
            )

        rnd = torch.randn((2, tp_size, numel), generator=gen).to(dtype)
        x.copy_(rnd[0, rank])
        y.copy_(rnd[1, rank])
        eager_add = ca.custom_all_reduce_add(x, y)
        assert torch.equal(eager_add, ca.custom_all_reduce(x + y)), f"{ctx} eager"
        replay(g_add, out_add)
        replay(g_ref, out_ref)
        assert torch.equal(out_add, out_ref), f"{ctx} graph"
        assert torch.equal(out_add, eager_add), f"{ctx} graph vs eager"


def _check_interleaved(ca, rank, tp_size, dtype, iterations, gen, keep):
    device = torch.device(f"cuda:{rank}")
    itemsize = torch.empty((), dtype=dtype).element_size()
    numels = [nbytes // itemsize for nbytes in INTERLEAVE_BYTES]
    xs = [torch.empty(n, dtype=dtype, device=device) for n in numels]
    ys = [torch.empty_like(t) for t in xs]
    # Two rounds of (plain, fused) per size.
    calls = [(i, fused) for _ in range(2) for i in range(len(xs)) for fused in (0, 1)]
    outs = [torch.empty_like(xs[i]) for i, _ in calls]

    def body():
        for (i, fused), out in zip(calls, outs):
            if fused:
                ca.all_reduce_add(xs[i], ys[i], out=out)
            else:
                ca.all_reduce(xs[i], out=out, registered_input=True)

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with ca.capture():
        graph = _capture(body, stream)
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    keep.append((xs, ys, outs, graph))

    for it in range(iterations):
        refs = []
        for i, n in enumerate(numels):
            ints = _int_inputs(gen, 2, tp_size, n)
            xs[i].copy_(ints[0, rank])
            ys[i].copy_(ints[1, rank])
            refs.append(
                [
                    ints[0].sum(0, dtype=torch.float32).to(device, dtype),
                    ints.sum((0, 1), dtype=torch.float32).to(device, dtype),
                ]
            )
        for out in outs:
            out.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        for (i, fused), out in zip(calls, outs):
            assert torch.equal(
                out, refs[i][fused]
            ), f"{rank=} interleaved {it=} {numels[i]=} {dtype=} {fused=}"


def _run(rank, tp_size, dtype_list, iterations, distributed_init_method):
    from aiter.dist.device_communicators.custom_all_reduce import CustomAllreduce

    torch.set_num_threads(1)
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")
    dist.init_process_group(
        "gloo", world_size=tp_size, rank=rank, init_method=distributed_init_method
    )
    ca = None
    keep = []
    try:
        ca = CustomAllreduce(dist.group.WORLD, device)
        assert not ca.disabled, "requires custom allreduce"
        assert ca.supports_all_reduce_add, "all_reduce_add not available"
        gen = torch.Generator().manual_seed(2026)
        for dtype in dtype_list:
            itemsize = torch.empty((), dtype=dtype).element_size()
            for nbytes in SIZES_BYTES:
                _check_size(
                    ca, rank, tp_size, dtype, nbytes // itemsize, iterations, gen, keep
                )
            _check_interleaved(ca, rank, tp_size, dtype, iterations, gen, keep)
            if rank == 0:
                logger.info(
                    "all_reduce_add passed: %s, TP%d, %d sizes + interleaved, "
                    "%d eager + graph runs each",
                    dtype,
                    tp_size,
                    len(SIZES_BYTES),
                    iterations,
                )
        # All ranks must finish reading peer buffers before any are freed.
        torch.cuda.synchronize()
        dist.barrier()
    finally:
        del keep
        if ca is not None:
            ca.close()
        dist.destroy_process_group()


def test_allreduce_add(tp_size, dtype_list, iterations=4):
    torch.multiprocessing.spawn(
        _run,
        args=(
            tp_size,
            dtype_list,
            iterations,
            get_distributed_init_method(get_ip(), get_open_port()),
        ),
        nprocs=tp_size,
        join=True,
    )


parser = argparse.ArgumentParser(description="all_reduce_add correctness test")
parser.add_argument(
    "-d",
    "--dtype",
    type=str,
    nargs="+",
    choices=["bf16", "fp16", "fp32"],
    default=["bf16", "fp16", "fp32"],
    help="data types (default: all)",
)
parser.add_argument(
    "-t",
    "--tp-size",
    type=int,
    choices=[2, 4, 8],
    default=8,
    help="number of GPUs / tensor-parallel size (default: 8)",
)
parser.add_argument(
    "--iterations",
    type=int,
    default=4,
    help="changing-input iterations per size (default: 4)",
)


if __name__ == "__main__":
    args = parser.parse_args()
    if args.iterations < 1:
        parser.error("--iterations must be positive")
    test_allreduce_add(
        args.tp_size, [dtypes.d_dtypes[d] for d in args.dtype], args.iterations
    )
