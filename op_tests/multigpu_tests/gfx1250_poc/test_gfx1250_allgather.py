# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""gfx1250 all-gather: the kernel the production route runs, and how fast.

Each rank gathers its tensor of --shape along --dim, TP2 / TP4. Candidates:
  prod     tensor_model_parallel_all_gather(use_custom=True), wherever it
           routes: the Lamport kernel up to the Lamport cap, else the LL or
           barrier kernels, else RCCL
  lamport  CustomAllreduce.lamport_all_gather, where should_lamport_ag holds
  car_ag   CustomAllreduce.all_gather_unreg (ints as same-width floats): the LL
           or barrier kernels through the copy-in pool, as the sweeps before
           the production route timed them
  car_reg  prod with the Lamport cap at 0, i.e. the route before Lamport: the
           same kernels, but a captured input is read in place (registered)
  rccl     dist.all_gather_into_tensor, then the transpose for other dims; ints
           as same-width floats, bytes as fp16 pairs

Dispatch: one eager call of each production entry per case, under the
profiler. "prod" / "dc" name the gather kernel that
tensor_model_parallel_all_gather / CudaCommunicator.all_gather (ATOM's
dcp_all_gather) launched (lamport, ag_ll, ag_gfx1250_* for the barrier
kernels, rccl), "copy" the ranks whose prod call also copied the input (an
unaligned input into the registered pool before the Lamport kernel; the
copy-in pool before an eager barrier kernel). "expect" / "dc expect" restate
the routing rules independently; "route ok" says every rank ran what they
give, copies included.

Correctness: bit for bit against the concatenated inputs, with the inputs and
edge cases of op_tests/multigpu_tests/test_ag_lamport.py.

Timing: a CUDA graph of CALLS calls is replayed REPLAYS times; median per rank,
slowest rank reported. TB/s counts the gathered output bytes.
"""

import argparse
import contextlib
import itertools
import os
import re
import statistics
import sys
from multiprocessing import Pool, set_start_method

import pandas as pd
import torch
import torch.distributed as dist

import aiter
from aiter import dtypes
from aiter.dist.device_communicators.custom_all_reduce import CustomAllreduce
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
from aiter.jit.utils.chip_info import get_gfx_runtime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from test_ag_lamport import (
    EDGE_CASES,
    bit_mismatch,
    cap_cases,
    gather_dc,
    gather_tp,
    make_inputs,
)

SUPPORTED_GFX = ["gfx1250"]
CALLS = 43
REPLAYS = 20
LL_AG_MAX_BYTES = 4 << 20  # kLLArMaxBytes: the C++ all_gather's LL range
_INT_TO_FP_VIEW = CustomAllreduce._INT_TO_FP_VIEW
# dtypes the barrier kernels take, ints by their float view
BARRIER_DTYPES = (
    torch.float32,
    torch.float16,
    torch.bfloat16,
    torch.int32,
    torch.int16,
)


def time_graph(fn):
    """Per-call us of fn: CALLS calls in one graph, median of REPLAYS replays."""
    graph = torch.cuda.CUDAGraph()
    with graph_capture() as gc:
        fn()  # warm up on the capture stream
        with torch.cuda.graph(graph, stream=gc.stream):
            for _ in range(CALLS):
                fn()
    torch.cuda.synchronize()
    get_tp_group().barrier()
    graph.replay()  # align the ranks
    events = [
        (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
        for _ in range(REPLAYS)
    ]
    for start, end in events:
        start.record()
        graph.replay()
        end.record()
    torch.cuda.synchronize()
    us = statistics.median(s.elapsed_time(e) * 1000 / CALLS for s, e in events)
    graph.reset()
    return us


@contextlib.contextmanager
def lamport_off(ca):
    """The routes as they were before Lamport: the Lamport cap at 0, what
    AITER_CUSTOM_AR_LAMPORT_RMS_MAX_SIZE=0 sets."""
    cap = ca._lamport_max_bytes
    ca._lamport_max_bytes = 0
    try:
        yield
    finally:
        ca._lamport_max_bytes = cap


def rccl_all_gather(x, dim):
    # RCCL's int32 all-gather here takes ms inside a graph and from 64 x 7168
    # leaves the process erroring on later launches (hipErrorInvalidValue), so
    # ints go as same-width floats and bytes as fp16 pairs: a gather is a copy
    view = _INT_TO_FP_VIEW.get(x.dtype)
    if view is None and x.element_size() == 1:
        view = torch.float16
    if view is not None:
        return rccl_all_gather(x.view(view), dim).view(x.dtype)
    tp = get_tp_group()
    out = torch.empty((tp.world_size,) + x.shape, dtype=x.dtype, device=x.device)
    dist.all_gather_into_tensor(out, x, group=tp.device_group)
    d = dim % x.dim()
    if d == 0:
        return out.reshape(-1, *x.shape[1:])
    return out.movedim(0, d).reshape(
        *x.shape[:d], tp.world_size * x.shape[d], *x.shape[d + 1 :]
    )


def car_all_gather(ca, x, dim):
    view = _INT_TO_FP_VIEW.get(x.dtype) or x.dtype
    return ca.all_gather_unreg(x.view(view), dim=dim).view(x.dtype)


def expect_route(x, dim, cap, cag_max, entry):
    """The gather kernel the routing rules give, restated from where they live:
    GroupCoordinator.all_gather ("tp") or CudaCommunicator.all_gather ("dc"),
    CustomAllreduce.should_lamport_ag, then the C++ all_gather (LL up to 4 MiB
    of whole 8B packets, else the barrier kernels)."""
    d = dim % x.dim()
    nbytes = x.numel() * x.element_size()
    row = nbytes if d == 0 else x.shape[-1] * x.element_size()
    custom = d == 0 or (d == x.dim() - 1 and row % 16 == 0)
    if entry == "dc":
        custom = custom and nbytes % 16 == 0 and nbytes <= cag_max
    if not custom:
        return "rccl"
    if row % 16 == 0 and 0 < nbytes <= cap:
        return "lamport"
    if nbytes <= LL_AG_MAX_BYTES and nbytes % 8 == 0 and row % 8 == 0:
        return "ag_ll"
    return "barrier"


def expect_copy(route, misaligned_rank):
    """Whether an eager call on that route copies this rank's input first; None
    where it does not matter (RCCL)."""
    return {
        "lamport": misaligned_rank,
        "ag_ll": False,
        "barrier": True,
        "rccl": None,
    }[route]


def launched(fn):
    """(gather kernel, whether the input was copied) of one eager call of fn."""
    torch.cuda.synchronize()
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA]
    ) as prof:
        fn()
        torch.cuda.synchronize()
    kernels, copied = [], False
    for e in prof.events():
        if e.device_type != torch.autograd.DeviceType.CUDA:
            continue
        name = e.name
        if "lamport_ag_gfx1250" in name:
            k = "lamport"
        elif "ag_ll_gfx1250" in name:
            k = "ag_ll"
        elif m := re.search(r"ag_gfx1250_\w+", name):
            k = m.group(0)
        elif (
            "nccl" in name.lower() or "meta::comms::" in name or name == "BatchStreamOp"
        ):
            # RCCL 2.30 gathers with meta::comms::dda* kernels, or with copies
            # and stream memory ops (BatchStreamOp), which aiter never issues
            k = "rccl"
        else:
            copied |= name.startswith("Memcpy")
            continue
        if k not in kernels:
            kernels.append(k)
    return "+".join(kernels) or "none", copied


def route_ok(kernel, copied, route, want_copy):
    if route == "barrier":
        hit = kernel.startswith("ag_gfx1250_") and "+" not in kernel
    else:
        hit = kernel == route
    return hit and (want_copy is None or copied == want_copy)


def run_case(ca, tp, rank, shape, dtype, dim, misaligned):
    seed = sum(shape) * 1000 + len(shape) * 10 + dim + 2
    x, ref, want = make_inputs(tp, rank, seed, shape, dtype, dim)
    if misaligned and rank % 2:
        storage = torch.empty(x.numel() + 1, dtype=dtype, device="cuda")
        x = storage[1:].view(x.shape).copy_(x)
    nbytes = x.numel() * x.element_size()
    cap = ca._lamport_max_bytes
    exp = expect_route(x, dim, cap, ca._cag_max_size, "tp")
    exp_dc = expect_route(x, dim, cap, ca._cag_max_size, "dc")
    misaligned_rank = misaligned and rank % 2 == 1

    ret = {
        "tp": tp,
        "shape": str(tuple(shape)),
        "dtype": str(dtype)[6:],
        "dim": dim,
        "KB/rank": nbytes / 1024,
        "misaligned": misaligned,
        "expect": exp,
        "dc expect": exp_dc,
    }
    kernel, copied = launched(lambda: gather_tp(x, dim))
    kernel_dc, copied_dc = launched(lambda: gather_dc(x, dim))
    ret["prod"], ret["dc"], ret["copy"] = kernel, kernel_dc, copied
    ret["route ok"] = route_ok(
        kernel, copied, exp, expect_copy(exp, misaligned_rank)
    ) and route_ok(kernel_dc, copied_dc, exp_dc, expect_copy(exp_dc, misaligned_rank))

    # name -> (fn, its reference, whether it runs with the Lamport cap at 0)
    cands = {"prod": (lambda: gather_tp(x, dim), want if exp == "lamport" else ref)}
    if ca.should_lamport_ag(x, dim):
        cands["lamport"] = (lambda: ca.lamport_all_gather(x, dim), want)
    # the kernels before Lamport, where they take the dtype: int64 has no float
    # view, and the barrier kernels take only 2- and 4-byte types
    old = expect_route(x, dim, 0, ca._cag_max_size, "tp")
    if (old == "ag_ll" and x.dtype != torch.int64) or (
        old == "barrier" and x.dtype in BARRIER_DTYPES
    ):
        cands["car_ag"] = (lambda: car_all_gather(ca, x, dim), ref)
        cands["car_reg"] = (lambda: gather_tp(x, dim), ref)
    cands["rccl"] = (lambda: rccl_all_gather(x, dim), ref)

    for name, (fn, good) in cands.items():
        with lamport_off(ca) if name == "car_reg" else contextlib.nullcontext():
            ret[f"{name} bitdiff"] = bit_mismatch(fn(), good)
    out_bytes = nbytes * tp
    for name, (fn, _) in cands.items():
        with lamport_off(ca) if name == "car_reg" else contextlib.nullcontext():
            us = time_graph(fn)
        ret[f"{name} us"] = us
        ret[f"{name} TB/s"] = out_bytes / us / 1e6
    return ret


def worker(tp, rank, cases, init_method):
    torch.cuda.set_device(rank)
    set_custom_all_reduce(True)
    init_distributed_environment(
        world_size=tp, rank=rank, distributed_init_method=init_method
    )
    ensure_model_parallel_initialized(tp, 1)
    ca = get_tp_group().device_communicator.ca_comm
    assert ca is not None and not ca.disabled, "custom allreduce is disabled"
    assert ca._lamport_max_bytes > 0, "the Lamport inbox is off"
    dist.all_reduce(torch.zeros(1, device="cuda"), group=get_tp_group().device_group)

    cases = cases + [c[:4] for c in cap_cases(ca._lamport_max_bytes)]
    rows = [run_case(ca, tp, rank, *c) for c in cases]

    torch.cuda.synchronize()
    get_tp_group().barrier()
    destroy_model_parallel()
    destroy_distributed_environment()
    return rows


def merge_ranks(per_rank):
    """Slowest rank's time, the worst mismatch of any rank, the kernels every
    rank ran, the ranks that copied."""
    merged = []
    for cells in zip(*per_rank):
        row = dict(cells[0])
        for k in row:
            if k.endswith((" us", "bitdiff")):
                row[k] = max(c[k] for c in cells)
            elif k.endswith(" TB/s"):
                row[k] = min(c[k] for c in cells)
        for k in ("prod", "dc"):
            row[k] = "|".join(dict.fromkeys(c[k] for c in cells))
        copies = [r for r, c in enumerate(cells) if c["copy"]]
        row["copy"] = (
            "all" if len(copies) == len(cells) else ",".join(map(str, copies)) or "-"
        )
        row["route ok"] = all(c["route ok"] for c in cells)
        merged.append(row)
    return merged


def run_tp(tp, cases):
    init_method = get_distributed_init_method(get_ip(), get_open_port())
    with Pool(processes=tp) as pool:
        jobs = [
            pool.apply_async(worker, (tp, r, cases, init_method)) for r in range(tp)
        ]
        # a protocol bug hangs instead of failing
        results = [j.get(timeout=7200) for j in jobs]
    return merge_ranks(results)


def main():
    if get_gfx_runtime() not in SUPPORTED_GFX:
        aiter.logger.warning(
            "gfx1250 all-gather sweep unsupported on %s; skipping", get_gfx_runtime()
        )
        return
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="gfx1250 all-gather: production route vs each kernel and RCCL",
    )
    parser.add_argument("-t", "--tp", type=int, nargs="*", default=[2, 4])
    parser.add_argument(
        "-s",
        "--shape",
        type=dtypes.str2tuple,
        nargs="*",
        # [4, BS, 32] a rank at BS (concurrency) 8-128, then hidden 7168 rows
        # from 14 KB to 14 MB a rank (TP4 i32 past the 8 MiB cap from 384)
        default=[(4, bs, 32) for bs in (8, 16, 32, 64, 128)]
        + [(m, 7168) for m in (1, 8, 16, 32, 64, 128, 192, 256, 288, 384, 448, 512)],
        help="per-rank input shape, e.g. -s 4,8,32 64,7168",
    )
    parser.add_argument(
        "-d",
        "--dtype",
        type=dtypes.str2Dtype,
        nargs="*",
        default=[dtypes.i32, dtypes.bf16, dtypes.fp16, dtypes.i16, dtypes.fp32],
    )
    parser.add_argument("--dim", type=int, nargs="*", default=[0, -1])
    args = parser.parse_args()
    ngpus = torch.cuda.device_count()

    cases = [
        (shape, dtype, dim, False)
        for dtype, dim, shape in itertools.product(args.dtype, args.dim, args.shape)
    ] + [c[:4] for c in EDGE_CASES]
    set_start_method("spawn", force=True)
    rows = []
    for tp in args.tp:
        if tp > ngpus:
            aiter.logger.warning("tp%d needs %d GPUs, have %d", tp, tp, ngpus)
            continue
        rows += run_tp(tp, cases)
    df = pd.DataFrame(rows)
    aiter.logger.info(
        "all-gather summary (markdown):\n%s",
        df.to_markdown(index=False, floatfmt=".3g"),
    )
    bad = df[~df["route ok"]]
    if len(bad):
        aiter.logger.warning(
            "%d cases ran another kernel than the routing rules give:\n%s",
            len(bad),
            bad[["tp", "shape", "dtype", "dim", "expect", "prod", "dc expect", "dc"]],
        )
    diffs = df.filter(like="bitdiff").fillna(0).max(axis=1)
    if (diffs > 0).any():
        aiter.logger.warning("bit mismatches:\n%s", df[diffs > 0])


if __name__ == "__main__":
    main()
