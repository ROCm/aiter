# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Lamport one-shot / two-shot path of custom all-reduce: correctness and the
numbers to tune its size ranges (CustomAllreduce::lamportKernel).

Every tp size runs twice, as two process groups:
  lamport on   the default. tensor_model_parallel_all_reduce takes the
               lamport kernel the dispatch picks ("lamport" columns; "kernel"
               says which), and each lamport kernel is also forced on the same
               input ("one_shot" / "two_shot" columns, empty where the size
               does not fit that kernel).
  lamport off  AITER_CUSTOM_AR_DISABLE_LAMPORT=1, the original dispatch
               ("orig" columns).
Production all-reduces run as CUDA graphs on both capture paths:
  registered   the input is IPC-registered; the original dispatch takes
               1-stage / 2-stage.
  copy_in      the input is first copied into the registered pool (what eager
               mode and expandable_segments do); the original dispatch takes
               LL / LL128 for small messages. A message the lamport path takes
               skips that copy, it reads only the local input.
The forced kernels need no registration, so they are timed once per shape.

Lamport results must match an fp32 rank-order sum bit for bit ("err" is the
mismatching fraction, 0 expected); "orig err" is checkAllclose's at bf16
tolerance. Every case also injects the 0xffff sentinel bit pattern on one rank,
which must not hang and must come out as NaN.
"""

import argparse
import os
from multiprocessing import Pool, freeze_support, set_start_method

import pandas as pd
import torch

import aiter
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
from aiter.jit.utils.chip_info import get_gfx
from aiter.test_common import benchmark, checkAllclose

SUPPORTED_GFX = ["gfx942", "gfx950"]
DISABLE_LAMPORT_ENV = "AITER_CUSTOM_AR_DISABLE_LAMPORT"
PACK_BYTES = 16
# kLamportMaxPacks in custom_all_reduce.cuh: one-shot positions per inbox slot
MAX_POSITIONS = 512 * 80
# kLamportTwoShotMaxBytes in custom_all_reduce.cuh: largest two-shot message
TWO_SHOT_MAX_BYTES = 2 << 20
# default sweep: m doubles up to this, so every n gets one size past the
# largest two-shot, enough to see the original dispatch take over (above that
# it is the original dispatch only); larger sizes via -m / -s
SWEEP_MAX_BYTES = 2 * TWO_SHOT_MAX_BYTES
KERNELS = {1: "one_shot", 2: "two_shot"}
CAPTURE_PATHS = ["registered", "copy_in"]
SENTINEL = -1  # 0xffff as int16
CALLS_PER_GRAPH = 10
GRAPH_REPLAYS = 20


def oneShotMaxBytes():
    return MAX_POSITIONS * PACK_BYTES


def twoShotMaxBytes(tp):
    return TWO_SHOT_MAX_BYTES


def twoShotMaxPositions(tp):
    """lamportTwoShotMaxPart: two-shot positions per chunk of the largest message."""
    return TWO_SHOT_MAX_BYTES // PACK_BYTES // tp


def fitsKernel(kernel, tp, nbytes):
    """Whether a message of nbytes can run on a lamport kernel at all."""
    if kernel == 1:
        return nbytes % PACK_BYTES == 0 and nbytes <= oneShotMaxBytes()
    return nbytes % (PACK_BYTES * tp) == 0 and nbytes <= twoShotMaxBytes(tp)


def defaultMs(tp, n, dtype):
    """1, 2, 4, ... up to the largest power of two within SWEEP_MAX_BYTES."""
    ms, m = [], 1
    while m * n * dtype.itemsize <= SWEEP_MAX_BYTES:
        ms.append(m)
        m *= 2
    return ms


def roundLikeKernel(t, dtype):
    # module_custom_all_reduce builds with -DOPUS_FP32_to_BF16_DEFAULT from the
    # env, default 5 (software round-to-nearest-even); 2 truncates on gfx94x.
    # gfx950 converts with the round-to-nearest-even instruction either way.
    truncates = os.environ.get("OPUS_FP32_to_BF16_DEFAULT", "5") == "2"
    if dtype == dtypes.bf16 and get_gfx() == "gfx942" and truncates:
        return (t.view(torch.int32) >> 16).to(torch.int16).view(dtype)
    return t.to(dtype)


def runTorch(xs, dtype):
    """Reference only: fp32 sum in rank order, one rounding at the end."""
    acc = xs[0].to(dtypes.fp32)
    for x in xs[1:]:
        acc += x.to(dtypes.fp32)
    return roundLikeKernel(acc, dtype)


def makeInputs(tp, rank, case_idx, m, n, dtype):
    """Every rank's input from one seed, so each rank has the reference."""
    gen = torch.Generator(device="cuda").manual_seed(case_idx)
    xs = [
        torch.randn((m, n), dtype=dtype, device="cuda", generator=gen)
        for _ in range(tp)
    ]
    # one rank sends the sentinel bit pattern; it must not hang and the output
    # there must be NaN
    xs[case_idx % tp].view(torch.int16).view(-1)[[0, -1]] = SENTINEL
    return xs[rank], xs


def bitErr(out, ref):
    """0 if out matches ref bit for bit off the sentinel positions and is NaN
    on them, else the fraction of mismatching elements (1 if a NaN is lost)."""
    nan_mask = torch.isnan(ref)
    if not torch.isnan(out[nan_mask]).all():
        return 1.0
    bad = out[~nan_mask].view(torch.int16) != ref[~nan_mask].view(torch.int16)
    return bad.sum().item() / max(bad.numel(), 1)


def timeGraph(fn, graphs):
    """Capture CALLS_PER_GRAPH calls in one graph, return (out, us per call).

    The graph and its buffers are kept in `graphs`: custom all-reduce caches
    peer IPC addresses of captured buffers by local pointer.
    """
    graph = torch.cuda.CUDAGraph()
    with graph_capture() as gc, torch.cuda.graph(graph, stream=gc.stream):
        for _ in range(CALLS_PER_GRAPH):
            out = fn()
    graphs.append((graph, out))
    for _ in range(3):
        graph.replay()
    torch.cuda.synchronize()
    get_tp_group().barrier()
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(GRAPH_REPLAYS)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(GRAPH_REPLAYS)]
    for s, e in zip(start_events, end_events):
        s.record()
        graph.replay()
        e.record()
    torch.cuda.synchronize()
    ms = sorted(s.elapsed_time(e) for s, e in zip(start_events, end_events))[
        GRAPH_REPLAYS // 2
    ]
    return out, ms * 1000 / CALLS_PER_GRAPH


def forceKernel(ca, x, kernel):
    out = torch.empty_like(x)
    aiter.lamportAllReduce(ca._ptr, x, out, kernel)
    return out


@benchmark()
def testLamportAllReduce(
    tp, m, n, dtype, path, lamport, case_idx=0, rank=0, ca=None, graphs=None
):
    x, xs = makeInputs(tp, rank, case_idx, m, n, dtype)
    ref = runTorch(xs, dtype)
    nbytes = x.numel() * x.element_size()
    ret = {"gfx": get_gfx()}
    # algorithm bandwidth: message bytes over time. An all-reduce does only
    # (tp - 1) * numel adds, so no TFLOPS column.
    if not lamport:
        out, us = timeGraph(lambda: tensor_model_parallel_all_reduce(x), graphs)
        ret["orig us"] = us
        ret["orig TB/s"] = nbytes / us / 1e6
        ret["orig err"] = checkAllclose(
            ref.to(dtypes.fp32),
            out.to(dtypes.fp32),
            mask=~torch.isnan(ref),
            printLog=rank == 0,
            msg=f"orig: tp{tp} {path} {m=} {n=}",
        )
        return ret

    kernel = aiter.lamportKernel(ca._ptr, x.numel())
    ret["kernel"] = KERNELS.get(kernel, "orig")
    candidates = {"lamport": lambda: tensor_model_parallel_all_reduce(x)}
    # the forced kernels need no registration, so they run on one path only
    if path == CAPTURE_PATHS[0]:
        for k, name in KERNELS.items():
            if fitsKernel(k, tp, nbytes):
                candidates[name] = lambda k=k: forceKernel(ca, x, k)
    for name, fn in candidates.items():
        out, us = timeGraph(fn, graphs)
        ret[f"{name} us"] = us
        ret[f"{name} TB/s"] = nbytes / us / 1e6
        # outside the lamport ranges "lamport" is the original dispatch, which
        # is not bitwise rank-order on every path
        if name == "lamport" and kernel == 0:
            ret[f"{name} err"] = checkAllclose(
                ref.to(dtypes.fp32),
                out.to(dtypes.fp32),
                mask=~torch.isnan(ref),
                printLog=rank == 0,
                msg=f"{name}: tp{tp} {path} {m=} {n=}",
            )
        else:
            ret[f"{name} err"] = bitErr(out, ref)
    return ret


def checkSizeChanges(ca, tp, rank, dtype):
    """Eager forced calls that shrink and regrow the message, per kernel, in
    positions per inbox slot: every block of the kernel's largest message, one
    lane, partial blocks (65 is block 1 lane 0), and for the two-shot also half
    its largest message and one past it, so positions that sit out a round must
    still be re-armed. Three passes cycle every buffer of every block."""
    pack_numel = PACK_BYTES // dtype.itemsize
    errs = []
    for kernel in KERNELS:
        if kernel == 1:
            top = MAX_POSITIONS
            positions = [top, 1, 5120, 192, 65, 64, top]
        else:
            top = twoShotMaxPositions(tp)
            positions = [
                top,
                1,
                top // 2,
                top // 2 + 1,
                5120,
                192,
                65,
                64,
                top,
            ]
        # a two-shot position stands for one pack of each of the tp chunks
        chunks = 1 if kernel == 1 else tp
        for i, pos in enumerate(positions * 3):
            x, xs = makeInputs(tp, rank, 1000 + i, 1, pos * pack_numel * chunks, dtype)
            errs.append(bitErr(forceKernel(ca, x, kernel), runTorch(xs, dtype)))
    return max(errs)


def checkUnalignedInput(ca, tp, rank, dtype):
    """Eager all-reduces on the copy-in path where odd ranks pass an input 2B
    past a 16B boundary: those ranks must copy it into the pool, the others
    skip the copy, and all must still run the same lamport kernel (no hang,
    bitwise result)."""
    errs = []
    for i, (m, n) in enumerate([(1, 4096), (64, 8192)]):
        x, xs = makeInputs(tp, rank, 2000 + i, m, n, dtype)
        assert (
            aiter.lamportKernel(ca._ptr, x.numel()) != 0
        ), f"{m}x{n} is not a lamport size"
        if rank % 2 == 1:
            buf = torch.empty(x.numel() + PACK_BYTES, dtype=dtype, device=x.device)
            x = buf[1 : 1 + x.numel()].view(m, n).copy_(x)
            assert x.data_ptr() % PACK_BYTES != 0
        errs.append(bitErr(tensor_model_parallel_all_reduce(x), runTorch(xs, dtype)))
    return max(errs)


def worker(tp, rank, shapes, dtype, lamport, init_method):
    if lamport:
        os.environ.pop(DISABLE_LAMPORT_ENV, None)
    else:
        os.environ[DISABLE_LAMPORT_ENV] = "1"
    torch.cuda.set_device(rank)
    set_custom_all_reduce(True)
    init_distributed_environment(
        world_size=tp, rank=rank, distributed_init_method=init_method
    )
    ensure_model_parallel_initialized(tp, 1)
    ca = get_tp_group().device_communicator.ca_comm
    assert (
        ca.lamport_enabled == lamport
    ), f"lamport_enabled={ca.lamport_enabled}, want {lamport}"

    resize_err = checkSizeChanges(ca, tp, rank, dtype) if lamport else None
    unaligned_err = checkUnalignedInput(ca, tp, rank, dtype) if lamport else None
    graphs = []
    rows = []
    for path in CAPTURE_PATHS:
        # copy_in: a host opt-out of the registered capture path
        ca.enable_register_for_capturing = path == "registered"
        for i, (m, n) in enumerate(shapes):
            row = testLamportAllReduce(
                tp,
                m,
                n,
                dtype,
                path,
                lamport,
                case_idx=i,
                rank=rank,
                ca=ca,
                graphs=graphs,
            )
            for k in ("case_idx", "rank", "ca", "graphs", "lamport"):
                row.pop(k, None)
            if lamport:
                row["resize err"] = resize_err
                row["unaligned err"] = unaligned_err
            rows.append(row)

    torch.cuda.synchronize()
    get_tp_group().barrier()
    del graphs
    destroy_model_parallel()
    destroy_distributed_environment()
    return rows


def runTp(tp, shapes, dtype, lamport, timeout):
    """One process group of tp ranks; per case the slowest rank's time and the
    worst rank's error."""
    init_method = get_distributed_init_method(get_ip(), get_open_port())
    with Pool(processes=tp) as pool:
        jobs = [
            pool.apply_async(worker, (tp, rank, shapes, dtype, lamport, init_method))
            for rank in range(tp)
        ]
        per_rank = [j.get(timeout=timeout) for j in jobs]
    rows = []
    for case in zip(*per_rank):
        row = dict(case[0])
        for k in row:
            if k.endswith((" us", " err")):
                row[k] = max(r[k] for r in case)
            elif k.endswith(" TB/s"):
                row[k] = min(r[k] for r in case)
        rows.append(row)
    return rows


def main():
    if get_gfx() not in SUPPORTED_GFX:
        aiter.logger.warning(
            "lamport all-reduce unsupported on %s; skipping", get_gfx()
        )
        return

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="lamport path of custom all-reduce: correctness and tuning numbers",
    )
    parser.add_argument(
        "-d",
        "--dtype",
        type=dtypes.str2Dtype,
        nargs="*",
        default="bf16,",
        help="data type, e.g. -d bf16",
    )
    parser.add_argument(
        "-t",
        "--tp",
        type=int,
        nargs="*",
        default=[2, 4, 8],
        help="tensor parallel sizes (2/4/8), e.g. -t 8",
    )
    parser.add_argument(
        "-n",
        type=int,
        nargs="*",
        default=[7168, 8192],
        help="hidden sizes, e.g. -n 7168",
    )
    parser.add_argument(
        "-m",
        type=int,
        nargs="*",
        default=None,
        help="token counts, default 1, 2, 4, ... up to 4MB for each n",
    )
    parser.add_argument(
        "-s",
        "--mn",
        type=dtypes.str2tuple,
        nargs="*",
        default=None,
        help="explicit (tokens, hidden) shapes, overrides -m/-n, e.g. -s 16,4096",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=1800,
        help="seconds per process group before declaring a hang",
    )
    args = parser.parse_args()

    keys = ["tp", "m", "n", "dtype", "path"]
    for dtype in args.dtype:
        df = []
        for tp in args.tp:
            if args.mn is not None:
                shapes = args.mn
            else:
                shapes = [
                    (m, n)
                    for n in args.n
                    for m in (args.m if args.m is not None else defaultMs(tp, n, dtype))
                ]
            on = pd.DataFrame(runTp(tp, shapes, dtype, True, args.timeout))
            off = pd.DataFrame(runTp(tp, shapes, dtype, False, args.timeout))
            df.append(on.merge(off.drop(columns=["gfx"]), on=keys))
        df = pd.concat(df, ignore_index=True)
        df["KB"] = df["m"] * df["n"] * dtype.itemsize // 1024
        lead = keys + ["KB", "gfx", "kernel"]
        df = df[lead + [c for c in df.columns if c not in lead]]
        aiter.logger.info(
            "lamport all-reduce summary (markdown):\n%s", df.to_markdown(index=False)
        )


if __name__ == "__main__":
    freeze_support()
    set_start_method("spawn", force=True)
    main()
