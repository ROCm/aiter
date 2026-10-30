# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Exact-result tests for CustomAllreduce.all_reduce_add and the all-reduce
kernel dispatch around its size cutovers.

For every size, in eager and CUDA-graph mode:
  * all_reduce(x) and all_reduce_add(x, y) on integer-valued inputs must equal
    an fp32 reference exactly, also with out aliasing x or y;
  * all_reduce_add(x, y) on random inputs must be bitwise equal to
    all_reduce(x + y).
A final graph interleaves plain and fused calls of different sizes, since the
LL, LL128 and 2-stage kernels share scratch and signal state. Inputs or outputs
whose physical layout differs from a contiguous one must be rejected. The fp16
fp8-quant path must match a separate add bitwise.

--bench prints CUDA-graph timings instead; running it on a base build and on
this one compares their dispatch.
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
# gfx950 aligned cutovers: plain LL 40 / 64 KB, fused LL 64 / 80 KB, fused
# LL128 208 KB. Unaligned: plain LL 48 / 56 / 72 KB, plain LL128 96 KB, fused
# LL 80 / 96 KB, fused LL128 128 / 384 KB, 1-stage 104 / 256 KB. Original
# 1-stage limits: 80 / 160 KB. 16 / 48 / 112 B have fewer 16-byte packs than
# ranks, which 2-stage can't split (fp32 bypasses LL).
SIZES_BYTES = sorted(
    {16, 48, 112, 16 * KB, 32 * KB + 16, 1024 * KB + 16, 4096 * KB}
    | {
        cap * KB + off
        for cap in (40, 64, 80, 160, 208, 256, 384, 1024)
        for off in (0, 128)
    }
    | {
        cap * KB + off
        for cap in (48, 56, 72, 80, 96, 104, 128, 160, 256, 384)
        for off in (-16, 16)
    }
)
# Includes back-to-back LL128 calls of different sizes (TP2: fused 128 KB - 16,
# 112 KB + 16, 208 KB, 256 KB + 16, 88 KB + 16, plain 88 KB + 16; TP4: fused
# 128 KB - 16, 112 KB + 16), which share the LL128 scratch.
INTERLEAVE_BYTES = [
    16 * KB,
    80 * KB,
    128 * KB - 16,
    112 * KB + 16,
    208 * KB,
    256 * KB + 16,
    88 * KB + 16,
    64 * KB + 128,
    1024 * KB,
]
FP8_QUANT_NUMELS = [128 * 2048, 512 * 2048]
# --bench sizes, each aligned and +16 B.
BENCH_SIZES_KB = (*range(16, 129, 16), 160, 256, 384, 512, 768, 1024, 4096, 16384)
BENCH_ROUNDS = 7


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
                got,
                ref,
                rtol=0,
                atol=0,
                msg=lambda m, label=f"{ctx} {name}": f"{label}\n{m}",
            )
        # out may alias inp or addend, e.g. residual = all_reduce_add(x, residual).
        for alias in ("inp", "addend"):
            a, b = x.clone(), y.clone()
            dst = a if alias == "inp" else b
            ca.all_reduce_add(a, b, out=dst)
            assert torch.equal(dst, ref_add), f"{ctx} out aliases {alias}"

        rnd = torch.randn((2, tp_size, numel), generator=gen).to(dtype)
        x.copy_(rnd[0, rank])
        y.copy_(rnd[1, rank])
        eager_add = ca.custom_all_reduce_add(x, y)
        assert torch.equal(eager_add, ca.custom_all_reduce(x + y)), f"{ctx} eager"
        replay(g_add, out_add)
        replay(g_ref, out_ref)
        assert torch.equal(out_add, out_ref), f"{ctx} graph"
        assert torch.equal(out_add, eager_add), f"{ctx} graph vs eager"


def _check_fp8_quant(ca, rank, tp_size, iterations, gen):
    """fp16 open_fp8_quant path (numel >= 128 * 2048): the staged sum must
    quantize exactly like a separate add."""
    device = torch.device(f"cuda:{rank}")
    for numel in FP8_QUANT_NUMELS:
        for it in range(iterations):
            ctx = f"{rank=} fp8 quant {numel=} {it=}"
            rnd = torch.rand((2, tp_size, numel), generator=gen) * 2 - 1
            x = rnd[0, rank].to(device, torch.float16)
            y = rnd[1, rank].to(device, torch.float16)
            fused = ca.custom_all_reduce_add(x, y, open_fp8_quant=True)
            separate = ca.custom_all_reduce(x + y, open_fp8_quant=True)
            assert fused is not None and separate is not None, ctx
            assert torch.equal(fused, separate), f"{ctx} fused vs separate"
            ref = rnd.to(torch.float16).float().sum((0, 1)).to(device)
            torch.testing.assert_close(
                fused.float(),
                ref,
                rtol=0,
                atol=0.1 * ref.abs().max().item(),
                msg=lambda m, label=ctx: f"{label} vs fp32 reference\n{m}",
            )


def _check_layouts(ca, rank, dtype):
    """Inputs or outputs with a different physical layout must be rejected:
    the kernels index them by flat offset."""
    device = torch.device(f"cuda:{rank}")
    x = torch.full((16, 16), rank + 1, device=device, dtype=dtype)
    y_t = (torch.arange(256, device=device) % 17).to(dtype).reshape(16, 16).t()
    assert ca.should_custom_ar(x)
    assert not ca.should_custom_ar_add(x, y_t), f"{rank=} {dtype=} transposed addend"
    assert not ca.should_custom_ar_add(y_t, x), f"{rank=} {dtype=} transposed inp"
    assert ca.custom_all_reduce_add(x, y_t) is None
    y = y_t.contiguous()
    out_t = torch.empty_like(x).t()
    out_wide = torch.empty_like(x, dtype=torch.float64)
    for case, args, kwargs in (
        ("transposed addend", (x, y_t), {}),
        ("transposed inp", (y_t, x), {}),
        ("transposed out", (x, y), {"out": out_t}),
        ("fp64 out", (x, y), {"out": out_wide}),
    ):
        try:
            ca.all_reduce_add(*args, **kwargs)
        except ValueError:
            continue
        raise AssertionError(f"{rank=} {dtype=} all_reduce_add accepted {case}")


def _check_interleaved(ca, rank, tp_size, dtype, iterations, gen, keep):
    device = torch.device(f"cuda:{rank}")
    itemsize = torch.empty((), dtype=dtype).element_size()
    numels = [nbytes // itemsize for nbytes in INTERLEAVE_BYTES]
    xs = [torch.empty(n, dtype=dtype, device=device) for n in numels]
    ys = [torch.empty_like(t) for t in xs]
    # Two rounds of (plain, fused) per size, then fused only, so LL128 calls of
    # different sizes run back to back without a 2-stage barrier in between.
    calls = [(i, fused) for _ in range(2) for i in range(len(xs)) for fused in (0, 1)]
    calls += [(i, 1) for i in range(len(xs))]
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


def _bench_size(ca, rank, dtype, numel, stream, keep):
    device = torch.device(f"cuda:{rank}")
    iters = 50 if numel * dtype.itemsize <= 1024 * KB else 10
    x = torch.randn(numel, device=device).to(dtype)
    y = torch.randn_like(x)
    s = torch.empty_like(x)

    def separate(o):
        torch.add(x, y, out=s)
        ca.all_reduce(s, out=o, registered_input=True)

    variants = [lambda o: ca.all_reduce(x, out=o, registered_input=True), separate]
    # Absent on base builds without all_reduce_add, which --bench compares against.
    if getattr(ca, "supports_all_reduce_add", False):
        variants.append(lambda o: ca.all_reduce_add(x, y, out=o))
    graphs = []
    for fn in variants:
        outs = [torch.empty_like(x) for _ in range(iters)]
        stream.wait_stream(torch.cuda.current_stream())
        with ca.capture():
            graphs.append(
                _capture(lambda fn=fn, outs=outs: [fn(o) for o in outs], stream)
            )
        torch.cuda.current_stream().wait_stream(stream)
        keep.append(outs)
    keep += [x, y, s, *graphs]
    torch.cuda.synchronize()

    # Alternating rounds so all variants see the same clocks and peer timing.
    samples = [[] for _ in graphs]
    for _ in range(BENCH_ROUNDS):
        for k, graph in enumerate(graphs):
            for _ in range(3):
                graph.replay()
            torch.cuda.synchronize()
            dist.barrier()
            start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
            start.record()
            for _ in range(10):
                graph.replay()
            end.record()
            end.synchronize()
            samples[k].append(start.elapsed_time(end) * 1000 / (10 * iters))
    times = [sorted(v)[BENCH_ROUNDS // 2] for v in samples]
    return times + [float("nan")] * (3 - len(times))


def _bench(ca, rank, tp_size, dtype_list, keep):
    stream = torch.cuda.Stream()
    rows = []
    for dtype in dtype_list:
        for kb in BENCH_SIZES_KB:
            for off in (0, 16):
                nbytes = kb * KB + off
                times = _bench_size(
                    ca, rank, dtype, nbytes // dtype.itemsize, stream, keep
                )
                rows.append((str(dtype).removeprefix("torch."), nbytes, times))
    all_rows = [None] * tp_size
    dist.all_gather_object(all_rows, rows)
    if rank == 0:
        lines = ["tp,dtype,bytes,all_reduce_us,add_all_reduce_us,all_reduce_add_us"]
        for i, (dname, nbytes, _) in enumerate(rows):
            times = [max(r[i][2][j] for r in all_rows) for j in range(3)]
            lines.append(
                f"{tp_size},{dname},{nbytes}," + ",".join(f"{t:.2f}" for t in times)
            )
        logger.info(
            "graph time per call, median of rounds, max over ranks:\n%s",
            "\n".join(lines),
        )


def _test(ca, rank, tp_size, dtype_list, iterations, keep):
    assert ca.supports_all_reduce_add, "all_reduce_add not available"
    gen = torch.Generator().manual_seed(2026)
    for dtype in dtype_list:
        for nbytes in SIZES_BYTES:
            numel = nbytes // dtype.itemsize
            _check_size(ca, rank, tp_size, dtype, numel, iterations, gen, keep)
        _check_interleaved(ca, rank, tp_size, dtype, iterations, gen, keep)
        _check_layouts(ca, rank, dtype)
        if dtype == torch.float16:
            _check_fp8_quant(ca, rank, tp_size, iterations, gen)
        if rank == 0:
            logger.info(
                "all_reduce_add passed: %s, TP%d, %d sizes + interleaved, "
                "%d eager + graph runs each",
                dtype,
                tp_size,
                len(SIZES_BYTES),
                iterations,
            )


def _run(rank, tp_size, dtype_list, iterations, bench, distributed_init_method):
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
        if bench:
            _bench(ca, rank, tp_size, dtype_list, keep)
        else:
            _test(ca, rank, tp_size, dtype_list, iterations, keep)
        # All ranks must finish reading peer buffers before any are freed.
        torch.cuda.synchronize()
        dist.barrier()
    finally:
        del keep
        if ca is not None:
            ca.close()
        dist.destroy_process_group()


def test_allreduce_add(tp_size, dtype_list, iterations=4, bench=False):
    torch.multiprocessing.spawn(
        _run,
        args=(
            tp_size,
            dtype_list,
            iterations,
            bench,
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
parser.add_argument(
    "--bench",
    action="store_true",
    help="instead of testing, print graph timings of all_reduce, "
    "add + all_reduce and all_reduce_add as CSV",
)


if __name__ == "__main__":
    args = parser.parse_args()
    if args.iterations < 1:
        parser.error("--iterations must be positive")
    test_allreduce_add(
        args.tp_size,
        [dtypes.d_dtypes[d] for d in args.dtype],
        args.iterations,
        args.bench,
    )
