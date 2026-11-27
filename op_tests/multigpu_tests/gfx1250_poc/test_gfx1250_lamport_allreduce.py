# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""gfx1250 all-reduce: the Lamport one-shot and two-shot, the kernel the
production route runs, and how fast.

Each rank all-reduces its [m, n] tensor. Candidates:
  prod           tensor_model_parallel_all_reduce, the route production takes:
                 the Lamport one-shot (fp16/bf16) or two-shot (fp32 too) up to
                 tp x the Lamport cap, else up to the plain all-reduce cutoff the
                 LL one-shot or the barrier kernel, else RCCL
  lamport_1shot  CustomAllreduce.lamport_all_reduce: every rank pushes its whole
                 input to every peer (lamport_ar_1shot_gfx1250)
  lamport_2shot  CustomAllreduce.lamport_all_reduce(two_shot=True): a pushed
                 reduce-scatter, then a pushed all-gather (lamport_ar_2shot_gfx1250),
                 a lane all three steps of a pack; up to tp x the one-shot cap
  lamport_split  the same with split=True: the steps in wave groups
  car            CustomAllreduce.custom_all_reduce with the Lamport caps at 0,
                 the custom kernels without Lamport, called directly (so past the
                 plain all-reduce cutoff too): the LL one-shot up to 4 MiB, the
                 barrier kernel above (registered input in a graph); "car route"
                 is the kernel it launched
  rccl           dist.all_reduce, in place on its own buffer

Without Lamport, prod is car up to the plain all-reduce cutoff and rccl above.

Correctness: bit for bit against the rank-order fp32 sum, rounded once (two
NaNs count as equal); rccl sums in another order at TP4. A mixed sequence then
runs both Lamport all-reduces between the Lamport AR + add, AR + RMSNorm and
all-gather on the one shared inbox, with sizes that leave chunks short or
empty and two-shots above the one-shot cap, and a graph replayed with fresh
inputs.

Routing: "expect" restates the routing rules independently, "route" is the
kernel an eager prod call launched (profiled), "route ok" that every rank ran
what they give.

Timing: a CUDA graph of CALLS calls is replayed REPLAYS times; median per rank,
slowest rank reported. TB/s counts the input bytes.
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
from aiter.dist.communication_op import (
    tensor_model_parallel_all_gather,
    tensor_model_parallel_all_reduce,
    tensor_model_parallel_all_reduce_add,
    tensor_model_parallel_fused_allreduce_rmsnorm,
)
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
from test_ag_lamport import make_inputs as ag_inputs

SUPPORTED_GFX = ["gfx1250"]
CALLS = 43
REPLAYS = 20
# (m, n) of the mixed sequence: (1, 8) is one pack (TP4: three empty chunks),
# (3, 1000) and (7, 1000) leave the last chunk short
MIXED_AR = [(48, 7168), (1, 8), (3, 1000), (409, 5120), (7, 1000), (128, 7168)]
MIXED_ADD = [(8, 5120), (3, 1000)]
MIXED_RMS = [(4, 7168), (300, 4096)]
MIXED_AG = [((4, 64, 32), torch.int32, 0), ((48, 7168), torch.bfloat16, -1)]
GRAPH_AR = [(48, 7168, True), (7, 1000, False), (288, 5120, True), (1, 8, True)]
# restated from custom_all_reduce.py / custom_all_reduce_gfx1250.cuh
LAMPORT_1SHOT_MAX_BYTES = {2: 9 << 20, 4: 1280 << 10, 8: 768 << 10}
# fp32 takes the LL one-shot up to here instead, the two-shot above
LAMPORT_FP32_LL_MAX_BYTES = {2: 1 << 20, 4: 0, 8: 0}
# the two-shot is split in (lo, hi], fused elsewhere
LAMPORT_SPLIT_BYTES = {
    2: (2 << 20, 4 << 20),
    4: (4 << 20, 8 << 20),
    8: (0, 4 << 20),
}
LL_AR_MAX_BYTES = 4 << 20


def parse_size(s):
    """'1k' -> 1024, '2m' -> 2 MiB, '4096' -> 4096 bytes."""
    unit = {"k": 1 << 10, "m": 1 << 20}.get(s[-1].lower(), 1)
    return int(s[:-1] if unit > 1 else s) * unit


@contextlib.contextmanager
def lamport_off(ca):
    """The routes as they were before Lamport: the Lamport caps at 0, what
    AITER_CUSTOM_AR_LAMPORT_RMS_MAX_SIZE=0 sets."""
    caps = ca._lamport_max_bytes, ca._lamport_ar_max_bytes
    ca._lamport_max_bytes = ca._lamport_ar_max_bytes = 0
    try:
        yield
    finally:
        ca._lamport_max_bytes, ca._lamport_ar_max_bytes = caps


def expect_route(x, tp, cap, car_max, plain_max):
    """The all-reduce kernel the routing rules give, restated from where they
    live: CudaCommunicator.all_reduce, should_custom_plain_ar,
    select_lamport_ar (cap: the plain all-reduce's Lamport cap), then the C++
    allreduce (the LL one-shot up to 4 MiB, else the barrier kernel)."""
    nbytes = x.numel() * x.element_size()
    if nbytes % 16 != 0 or nbytes > car_max:
        return "rccl"
    if (
        x.dtype in (torch.float16, torch.bfloat16, torch.float32)
        and x.is_contiguous()
        and 0 < nbytes <= tp * cap
    ):
        one_shot_max = min(LAMPORT_1SHOT_MAX_BYTES.get(tp, cap), cap)
        if x.dtype == torch.float32:
            ll_max = LAMPORT_FP32_LL_MAX_BYTES.get(tp, one_shot_max)
            one_shot = None if nbytes <= min(ll_max, LL_AR_MAX_BYTES) else False
        else:
            one_shot = nbytes <= one_shot_max
        if one_shot is not None:
            if one_shot:
                return "1shot"
            lo, hi = LAMPORT_SPLIT_BYTES.get(tp, (0, 0))
            return "split" if lo < nbytes <= hi else "2shot"
    if nbytes > plain_max:
        return "rccl"
    return car_route(nbytes, tp)


def car_route(nbytes, tp):
    """The kernel the C++ allreduce picks without Lamport."""
    return "ll" if nbytes <= LL_AR_MAX_BYTES else "barrier"


def launched(fn):
    """The all-reduce kernel(s) one eager call of fn launched."""
    torch.cuda.synchronize()
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA]
    ) as prof:
        fn()
        torch.cuda.synchronize()
    kernels = []
    for e in prof.events():
        if e.device_type != torch.autograd.DeviceType.CUDA:
            continue
        name = e.name
        if m := re.search(r"lamport_ar_2shot_gfx1250<[^>]*,\s*(\d+)\s*>", name):
            # the last template argument is ROLES: 1 fused, 2 or 3 split
            k = "2shot" if m.group(1) == "1" else "split"
        elif m := re.search(r"lamport_ar_(1shot|2shot)_gfx1250", name):
            k = m.group(1)
        elif "ar_ll_gfx1250" in name:
            k = "ll"
        elif "ar_gfx1250_naive" in name:
            k = "barrier"
        elif (
            "nccl" in name.lower() or "meta::comms::" in name or name == "BatchStreamOp"
        ):
            k = "rccl"
        else:
            continue
        if k not in kernels:
            kernels.append(k)
    return "+".join(kernels) or "none"


def make_inputs(tp, rank, seed, m, n, dtype, nan=False):
    """This rank's input and the reference: the rank-order fp32 sum, rounded
    once."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    xs = torch.randn((tp, m, n), dtype=torch.float32, device="cuda", generator=g)
    xs = xs.to(dtype)
    if nan:
        # 0xffff is the inbox sentinel: it must not hang, its value is NaN
        xs[0].view(torch.int16)[m // 2, n // 3] = -1
    acc = xs[0].float()
    for r in range(1, tp):
        acc = acc + xs[r].float()
    return xs[rank].contiguous(), acc.to(dtype)


def bit_mismatch(a, b):
    """Elements whose bits differ; two NaNs count as equal."""
    if a.shape != b.shape or a.dtype != b.dtype:
        return a.numel()
    t = {1: torch.uint8, 2: torch.int16, 4: torch.int32}[a.element_size()]
    same = (a.view(t) == b.view(t)) | (torch.isnan(a) & torch.isnan(b))
    return int((~same).sum().item())


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


def run_case(ca, tp, rank, m, n, dtype):
    x, ref = make_inputs(tp, rank, n * 1000 + m, m, n, dtype)
    nbytes = x.numel() * x.element_size()
    rccl_buf = x.clone()
    group = get_tp_group().device_group
    cands = {"prod": lambda: tensor_model_parallel_all_reduce(x)}
    if ca.should_lamport_ar(x):
        cands["lamport_1shot"] = lambda: ca.lamport_all_reduce(x)
    if ca.should_lamport_ar(x, two_shot=True):
        cands["lamport_2shot"] = lambda: ca.lamport_all_reduce(x, two_shot=True)
        cands["lamport_split"] = lambda: ca.lamport_all_reduce(
            x, two_shot=True, split=True
        )
    if ca.should_custom_ar(x):

        def car():
            with lamport_off(ca):
                return ca.custom_all_reduce(x)

        cands["car"] = car

    def rccl():
        dist.all_reduce(rccl_buf, group=group)
        return rccl_buf

    ret = {"tp": tp, "m": m, "n": n, "dtype": str(dtype)[6:], "KB": nbytes / 1024}
    ret["expect"] = expect_route(
        x, tp, ca._lamport_ar_max_bytes, ca._car_max_size, ca._plain_ar_max_size
    )
    ret["route"] = launched(cands["prod"])
    ret["route ok"] = ret["route"] == ret["expect"]
    if "car" in cands:
        ret["car route"] = launched(cands["car"])
        ret["route ok"] &= ret["car route"] == car_route(nbytes, tp)
    for name, fn in cands.items():
        ret[f"{name} bitdiff"] = bit_mismatch(fn(), ref)
    rccl_buf.copy_(x)
    ret["rccl bitdiff"] = bit_mismatch(rccl(), ref)
    cands["rccl"] = rccl
    for name, fn in cands.items():
        us = time_graph(fn)
        ret[f"{name} us"] = us
        ret[f"{name} TB/s"] = nbytes / us / 1e6
    return ret


def check_mixed_sequence(ca, tp, rank):
    """Both Lamport all-reduces between the other Lamport ops on the one inbox,
    eager and in a graph, against the reference bit for bit."""
    worst, calls = 0, 0
    g = torch.Generator(device="cuda").manual_seed(4321)
    bf16 = torch.bfloat16

    def ar(seed, m, n, two_shot, nan=False, split=False, dtype=bf16):
        nonlocal worst, calls
        x, ref = make_inputs(tp, rank, seed, m, n, dtype, nan=nan)
        assert ca.should_lamport_ar(x, two_shot), (m, n, two_shot, dtype)
        out = ca.lamport_all_reduce(x, two_shot, split and two_shot)
        worst = max(worst, bit_mismatch(out, ref))
        calls += 1

    def add(m, n):
        x = torch.randn((m, n), device="cuda", generator=g).to(bf16)
        assert ca.should_lamport_ar_add(x, x), (m, n)
        tensor_model_parallel_all_reduce_add(x, x)

    def rms(m, n):
        x = torch.randn((m, n), device="cuda", generator=g).to(bf16)
        res = torch.randn((m, n), device="cuda", generator=g).to(bf16)
        w = torch.randn((n,), device="cuda", generator=g).to(bf16)
        assert ca.should_lamport_ar_rms(x, res, w, n), (m, n)
        tensor_model_parallel_fused_allreduce_rmsnorm(x, res, w, 1e-6)

    def ag(seed, shape, dtype, dim):
        nonlocal worst, calls
        x, _, want = ag_inputs(tp, rank, seed, shape, dtype, dim)
        assert ca.should_lamport_ag(x, dim), (shape, dim)
        out = tensor_model_parallel_all_gather(x, use_custom=True, dim=dim)
        worst = max(worst, bit_mismatch(out, want))
        calls += 1

    for i, (m, n) in enumerate(MIXED_AR * 4):
        # one-shot, fused and split two-shots, every fourth two-shot fp32
        ar(
            i,
            m,
            n,
            two_shot=i % 2 == 1,
            nan=i % 5 == 0,
            split=i % 3 == 1,
            dtype=torch.float32 if i % 4 == 3 else bf16,
        )
        if i % 3 == 0:
            add(*MIXED_ADD[i % len(MIXED_ADD)])
        if i % 4 == 1:
            rms(*MIXED_RMS[i % len(MIXED_RMS)])
        if i % 5 == 2:
            ag(i, *MIXED_AG[i % len(MIXED_AG)])
    # the two-shot back to back, the same and then other sizes, fused and split
    for i in range(6):
        ar(200 + i, *MIXED_AR[i // 2], two_shot=True, split=i % 2 == 1)
    # the two-shot above the one-shot cap, up to tp x the cap: just over it
    # (the last chunks short or empty), the full size, sizes in between; each
    # followed by a one-shot at the cap, which takes every wave of the inbox
    cap = ca._lamport_ar_max_bytes // 2  # bf16 values
    mid = (tp + 1) * cap // 2 // 8 * 8 + 8 * 5
    big = [(1, cap + 8), (1, tp * cap), (1, tp * cap - 8 * 33), (1, mid)]
    for i, (m, n) in enumerate(big):
        ar(300 + i, m, n, two_shot=True, nan=i == 0, split=i % 2 == 0)
        ar(310 + i, 1, cap, two_shot=False)
        rms(*MIXED_RMS[i % len(MIXED_RMS)])
        ar(320 + i, m, n, two_shot=True, split=i % 2 == 1)
        ag(330 + i, *MIXED_AG[i % len(MIXED_AG)])
        add(*MIXED_ADD[i % len(MIXED_ADD)])

    graph_ar = GRAPH_AR + [(1, tp * cap, True), (1, cap, False), (1, cap + 8, True)]
    ins = [make_inputs(tp, rank, 0, m, n, bf16)[0] for m, n, _ in graph_ar]
    graph = torch.cuda.CUDAGraph()
    with graph_capture() as gc, torch.cuda.graph(graph, stream=gc.stream):
        outs = [
            ca.lamport_all_reduce(x, two_shot)
            for x, (_, _, two_shot) in zip(ins, graph_ar)
        ]
    for rep in range(6):
        refs = []
        for j, ((m, n, _), x) in enumerate(zip(graph_ar, ins)):
            nx, ref = make_inputs(tp, rank, 500 + 10 * rep + j, m, n, bf16)
            x.copy_(nx)
            refs.append(ref)
        graph.replay()
        torch.cuda.synchronize()
        for o, ref in zip(outs, refs):
            worst = max(worst, bit_mismatch(o, ref))
            calls += 1
        ar(900 + rep, *MIXED_AR[rep % len(MIXED_AR)], two_shot=rep % 2 == 0)
        add(*MIXED_ADD[rep % len(MIXED_ADD)])
        rms(*MIXED_RMS[rep % len(MIXED_RMS)])
    graph.reset()
    return {"tp": tp, "calls": calls, "max bitdiff": worst}


def worker(tp, rank, cases, init_method):
    torch.cuda.set_device(rank)
    set_custom_all_reduce(True)
    init_distributed_environment(
        world_size=tp, rank=rank, distributed_init_method=init_method
    )
    ensure_model_parallel_initialized(tp, 1)
    ca = get_tp_group().device_communicator.ca_comm
    assert ca is not None and not ca.disabled, "custom allreduce is disabled"
    assert ca._lamport_max_bytes > 0, "the Lamport AR + RMSNorm inbox is off"
    assert ca._lamport_ar_max_bytes > 0, "the Lamport all-reduce is off"
    dist.all_reduce(torch.zeros(1, device="cuda"), group=get_tp_group().device_group)

    mixed = check_mixed_sequence(ca, tp, rank)
    rows = [run_case(ca, tp, rank, *c) for c in cases]

    torch.cuda.synchronize()
    get_tp_group().barrier()
    destroy_model_parallel()
    destroy_distributed_environment()
    return rows, mixed


def merge_ranks(per_rank):
    """Slowest rank's time and the worst mismatch of any rank, case by case."""
    merged = []
    for cells in zip(*per_rank):
        row = dict(cells[0])
        for k in row:
            if k.endswith((" us", "bitdiff")):
                row[k] = max(c[k] for c in cells)
            elif k == "route ok":
                row[k] = all(c[k] for c in cells)
            elif k.endswith(" TB/s"):
                row[k] = min(c[k] for c in cells)
        merged.append(row)
    return merged


def run_tp(tp, cases):
    init_method = get_distributed_init_method(get_ip(), get_open_port())
    with Pool(processes=tp) as pool:
        jobs = [
            pool.apply_async(worker, (tp, r, cases, init_method)) for r in range(tp)
        ]
        # a protocol bug hangs instead of failing
        results = [j.get(timeout=3600) for j in jobs]
    rows = merge_ranks([r[0] for r in results])
    mixed = merge_ranks([[r[1]] for r in results])[0]
    return rows, mixed


def main():
    if get_gfx_runtime() not in SUPPORTED_GFX:
        aiter.logger.warning(
            "gfx1250 Lamport all-reduce unsupported on %s; skipping", get_gfx_runtime()
        )
        return
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="gfx1250 Lamport all-reduce, one-shot and two-shot (PoC)",
    )
    parser.add_argument("-t", "--tp", type=int, nargs="*", default=[2, 4])
    parser.add_argument(
        "-m",
        type=int,
        nargs="*",
        default=[1, 4, 16, 32, 64, 128, 256, 384, 512, 768, 1024],
    )
    parser.add_argument("-n", type=int, nargs="*", default=[7168])
    parser.add_argument(
        "-d", "--dtype", type=dtypes.str2Dtype, nargs="*", default=[dtypes.bf16]
    )
    parser.add_argument(
        "--size",
        type=parse_size,
        nargs="*",
        default=[],
        help="bytes a rank, e.g. 1k 2m (KiB / MiB); a [1, size] tensor each,\n"
        "instead of -m x -n",
    )
    args = parser.parse_args()
    ngpus = torch.cuda.device_count()

    if args.size:
        cases = [
            (1, size // dtype.itemsize, dtype)
            for dtype, size in itertools.product(args.dtype, args.size)
        ]
    else:
        cases = list(itertools.product(args.m, args.n, args.dtype))
    set_start_method("spawn", force=True)
    rows, mixed = [], []
    for tp in args.tp:
        if tp > ngpus:
            aiter.logger.warning("tp%d needs %d GPUs, have %d", tp, tp, ngpus)
            continue
        tp_rows, tp_mixed = run_tp(tp, cases)
        rows += tp_rows
        mixed.append(tp_mixed)
    df = pd.DataFrame(rows)
    aiter.logger.info(
        "Lamport all-reduce summary (markdown):\n%s",
        df.to_markdown(index=False, floatfmt=".3g"),
    )
    bad = df[~df["route ok"]]
    if len(bad):
        aiter.logger.warning(
            "%d cases ran another kernel than the routing rules give:\n%s",
            len(bad),
            bad[["tp", "m", "n", "dtype", "expect", "route"]],
        )
    aiter.logger.info(
        "Lamport all-reduce mixed sequence (markdown):\n%s",
        pd.DataFrame(mixed).to_markdown(index=False),
    )


if __name__ == "__main__":
    main()
