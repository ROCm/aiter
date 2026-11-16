# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""tensor_model_parallel_all_reduce_add: the sum over ranks of each rank's x + y
(the MoE shared-expert add), with the bits of all-reducing the materialized sum.

On gfx1250 it takes the Lamport one-shot kernel up to the Lamport size cap
(AITER_CUSTOM_AR_LAMPORT_RMS_MAX_SIZE), else x + y and then all_reduce.
Candidates, all with the same output:
  prod_route   tensor_model_parallel_all_reduce_add, whatever it routes to
  gluon_fused  the Gluon all-reduce with x2 (PR #7488), with --gluon-pr-dir
  gluon_add+ar x + y, then the plain Gluon all-reduce (TP4 <= 4 MiB)
  car_add+ar   x + y, then the custom all-reduce
  rccl_add+ar  x + y, then RCCL

Correctness: every candidate bit for bit against the rank-order fp32 reference
(two NaNs count as equal). A mixed sequence then interleaves the add with the
Lamport AR + RMSNorm on the one shared inbox, with sizes that leave the last
wave partial, injected 0xffff, fallback calls (past the cap, a broadcast
operand, against the fake's metadata), operands off 16B alignment on the odd
ranks only (every rank must still take the Lamport kernel) and a graph replayed
with fresh inputs.

Timing: a CUDA graph of CALLS calls is replayed REPLAYS times; median per rank,
slowest rank reported (CALLS = 43, the MoE layers of a DeepSeek-V4.1 step).
"""

import argparse
import importlib
import statistics
import sys
from multiprocessing import Pool, set_start_method

import pandas as pd
import torch
import torch.distributed as dist

import aiter
from aiter import dtypes
from aiter.dist.communication_op import (
    tensor_model_parallel_all_reduce_add,
    tensor_model_parallel_fused_allreduce_rmsnorm,
)
from aiter.dist.parallel_state import (
    all_reduce_add_fake,
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

SUPPORTED_GFX = ["gfx1250"]
CALLS = 43
REPLAYS = 20
GLUON_MAX_BYTES = 4 << 20
# (m, n) of the mixed sequence: n = 1000 leaves the last wave partial, (1, 8)
# is one pack
MIXED_ADD = [(409, 5120), (3, 1000), (48, 5120), (1, 8), (300, 1000), (8, 5120)]
MIXED_RMS = [(4, 7168), (300, 4096)]
GRAPH_ADD = [(48, 5120), (7, 1000), (288, 5120), (1, 8)]


def make_inputs(tp, rank, seed, m, n, dtype, nan=False):
    """This rank's x, y and the reference: the rank-order fp32 sum of each rank's
    x + y rounded to the dtype, rounded once."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    xs = torch.randn((2, tp, m, n), dtype=torch.float32, device="cuda", generator=g)
    xs = xs.to(dtype)
    if nan:
        # 0xffff is the inbox sentinel: it must not hang, its value is NaN
        xs[0, 0].view(torch.int16)[m // 2, n // 3] = -1
    sums = xs[0] + xs[1]
    acc = sums[0].float()
    for r in range(1, tp):
        acc = acc + sums[r].float()
    return xs[0, rank].contiguous(), xs[1, rank].contiguous(), acc.to(dtype)


def offset_view(t):
    """t's values one element into a fresh storage, so off 16B alignment."""
    storage = torch.empty(t.numel() + 1, dtype=t.dtype, device=t.device)
    return storage[1:].view(t.shape).copy_(t)


def bit_mismatch(a, b):
    """Elements whose bits differ; two NaNs count as equal."""
    if a.shape != b.shape or a.dtype != b.dtype:
        return a.numel()
    same = (a.view(torch.int16) == b.view(torch.int16)) | (
        torch.isnan(a) & torch.isnan(b)
    )
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


def make_candidates(ca, gluon, x, y):
    tp = get_tp_group()
    cands = {"prod_route": lambda: tensor_model_parallel_all_reduce_add(x, y)}
    nbytes = x.numel() * x.element_size()
    if gluon is not None and nbytes <= GLUON_MAX_BYTES and x.dtype == torch.bfloat16:
        mod, state = gluon
        if mod.__name__ != "aiter.ops.triton.comms.all_reduce":
            cands["gluon_fused"] = lambda: mod.all_reduce(x, tp.cpu_group, *state, x2=y)
        cands["gluon_add+ar"] = lambda: mod.all_reduce(x + y, tp.cpu_group, *state)
    cands["car_add+ar"] = lambda: ca.custom_all_reduce(x + y)

    def rccl():
        s = x + y
        dist.all_reduce(s, group=tp.device_group)
        return s

    cands["rccl_add+ar"] = rccl
    return cands


def test_case(ca, gluon, tp, rank, m, n, dtype):
    x, y, ref = make_inputs(tp, rank, n * 1000 + m, m, n, dtype)
    cands = make_candidates(ca, gluon, x, y)
    ret = {"tp": tp, "m": m, "n": n, "KB": m * n * x.element_size() / 1024}
    ret["lamport"] = ca.should_lamport_ar_add(x, y)
    for name, fn in cands.items():
        ret[f"{name} bitdiff"] = bit_mismatch(fn(), ref)
    for name, fn in cands.items():
        ret[f"{name} us"] = time_graph(fn)
    return ret


def check_mixed_sequence(ca, tp, rank, dtype):
    """The add and the AR + RMSNorm interleaved on the one inbox, eager and in a
    graph, plus fallback calls, against the reference bit for bit."""
    worst, calls = 0, 0
    g = torch.Generator(device="cuda").manual_seed(1234)

    def rms(m, n):
        x = torch.randn((m, n), device="cuda", generator=g).to(dtype)
        res = torch.randn((m, n), device="cuda", generator=g).to(dtype)
        w = torch.randn((n,), device="cuda", generator=g).to(dtype)
        assert ca.should_lamport_ar_rms(x, res, w, n), (m, n)
        tensor_model_parallel_fused_allreduce_rmsnorm(x, res, w, 1e-6)

    def add(seed, m, n, nan=False):
        nonlocal worst, calls
        x, y, ref = make_inputs(tp, rank, seed, m, n, dtype, nan=nan)
        assert ca.should_lamport_ar_add(x, y), (m, n)
        worst = max(
            worst, bit_mismatch(tensor_model_parallel_all_reduce_add(x, y), ref)
        )
        calls += 1

    for i, (m, n) in enumerate(MIXED_ADD * 3):
        add(i, m, n, nan=i % 4 == 0)
        if i % 2:
            rms(*MIXED_RMS[i % len(MIXED_RMS)])

    # fallbacks: past the cap, and a broadcast operand (the fake must match)
    m = ca._lamport_max_bytes // (5120 * dtype.itemsize) + 1
    x, y, ref = make_inputs(tp, rank, 77, m, 5120, dtype)
    assert not ca.should_lamport_ar_add(x, y)
    worst = max(worst, bit_mismatch(tensor_model_parallel_all_reduce_add(x, y), ref))
    y1 = y[:1]
    got = tensor_model_parallel_all_reduce_add(x, y1)
    fake = all_reduce_add_fake(x, y1)
    assert got.shape == fake.shape and got.dtype == fake.dtype, (got.shape, fake.shape)
    calls += 2

    # the odd ranks' operands one element into their storage: every rank must
    # still take the Lamport kernel, the odd ones copying them first
    for m, n in MIXED_ADD[:2]:
        x, y, ref = make_inputs(tp, rank, 300 + m, m, n, dtype)
        if rank % 2:
            x, y = offset_view(x), offset_view(y)
        assert ca.should_lamport_ar_add(x, y), (m, n)
        got = tensor_model_parallel_all_reduce_add(x, y)
        worst = max(worst, bit_mismatch(got, ref))
        calls += 1

    ins = [make_inputs(tp, rank, 0, m, n, dtype)[:2] for m, n in GRAPH_ADD]
    if rank % 2:  # unaligned operands inside the graph as well
        ins[0] = tuple(offset_view(t) for t in ins[0])
    graph = torch.cuda.CUDAGraph()
    with graph_capture() as gc, torch.cuda.graph(graph, stream=gc.stream):
        outs = [tensor_model_parallel_all_reduce_add(x, y) for x, y in ins]
    for rep in range(6):
        refs = []
        for j, ((m, n), (x, y)) in enumerate(zip(GRAPH_ADD, ins)):
            nx, ny, ref = make_inputs(tp, rank, 500 + 10 * rep + j, m, n, dtype)
            x.copy_(nx)
            y.copy_(ny)
            refs.append(ref)
        graph.replay()
        torch.cuda.synchronize()
        for o, ref in zip(outs, refs):
            worst = max(worst, bit_mismatch(o, ref))
            calls += 1
        add(900 + rep, *MIXED_ADD[rep % len(MIXED_ADD)])
        rms(*MIXED_RMS[rep % len(MIXED_RMS)])
    graph.reset()
    return {"tp": tp, "calls": calls, "max bitdiff": worst}


def init_gluon(tp, pr_dir):
    """The Gluon all-reduce module and its inbox, TP4 only; the PR #7488 one
    (with x2) when pr_dir is given."""
    if tp != 4:
        return None
    if pr_dir:
        sys.path.insert(0, pr_dir)
        mod = importlib.import_module("gluon_ar_add")
    else:
        mod = importlib.import_module("aiter.ops.triton.comms.all_reduce")
    return mod, mod.all_reduce_init(get_tp_group().cpu_group, GLUON_MAX_BYTES)


def worker(tp, rank, ms, ns, dtype, pr_dir, init_method):
    torch.cuda.set_device(rank)
    set_custom_all_reduce(True)
    init_distributed_environment(
        world_size=tp, rank=rank, distributed_init_method=init_method
    )
    ensure_model_parallel_initialized(tp, 1)
    ca = get_tp_group().device_communicator.ca_comm
    assert ca is not None and not ca.disabled, "custom allreduce is disabled"
    dist.all_reduce(torch.zeros(1, device="cuda"), group=get_tp_group().device_group)
    gluon = init_gluon(tp, pr_dir)

    rows = [test_case(ca, gluon, tp, rank, m, n, dtype) for n in ns for m in ms]
    mixed = (
        check_mixed_sequence(ca, tp, rank, dtype) if ca._lamport_max_bytes > 0 else {}
    )

    torch.cuda.synchronize()
    get_tp_group().barrier()
    if gluon is not None:
        gluon[0].all_reduce_free(get_tp_group().cpu_group, gluon[1][0])
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
        merged.append(row)
    return merged


def run_tp(tp, ms, ns, dtype, pr_dir):
    init_method = get_distributed_init_method(get_ip(), get_open_port())
    with Pool(processes=tp) as pool:
        jobs = [
            pool.apply_async(worker, (tp, r, ms, ns, dtype, pr_dir, init_method))
            for r in range(tp)
        ]
        # a protocol bug hangs instead of failing
        results = [j.get(timeout=3600) for j in jobs]
    rows = merge_ranks([r[0] for r in results])
    mixed = merge_ranks([[r[1]] for r in results])[0] if results[0][1] else None
    return rows, mixed


def main():
    if get_gfx() not in SUPPORTED_GFX:
        aiter.logger.warning(
            "lamport all-reduce add unsupported on %s; skipping", get_gfx()
        )
        return
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="tensor_model_parallel_all_reduce_add vs the other gfx1250 routes",
    )
    parser.add_argument("-t", "--tp", type=int, nargs="*", default=[2, 4])
    parser.add_argument(
        "-m", type=int, nargs="*", default=[1, 8, 48, 96, 288, 409, 819, 1638]
    )
    parser.add_argument("-n", type=int, nargs="*", default=[5120])
    parser.add_argument(
        "-d", "--dtype", type=dtypes.str2Dtype, nargs="*", default=[dtypes.bf16]
    )
    parser.add_argument(
        "--gluon-pr-dir",
        default="",
        help="directory with PR #7488's Gluon all-reduce as gluon_ar_add.py",
    )
    args = parser.parse_args()
    ngpus = torch.cuda.device_count()

    set_start_method("spawn", force=True)
    for dtype in args.dtype:
        rows, mixed = [], []
        for tp in args.tp:
            if tp > ngpus:
                aiter.logger.warning("tp%d needs %d GPUs, have %d", tp, tp, ngpus)
                continue
            tp_rows, tp_mixed = run_tp(tp, args.m, args.n, dtype, args.gluon_pr_dir)
            rows += tp_rows
            if tp_mixed:
                mixed.append(tp_mixed)
        df = pd.DataFrame(rows)
        aiter.logger.info(
            "all-reduce add %s summary (markdown):\n%s",
            dtype,
            df.to_markdown(index=False, floatfmt=".3g"),
        )
        if mixed:
            aiter.logger.info(
                "all-reduce add %s mixed sequence (markdown):\n%s",
                dtype,
                pd.DataFrame(mixed).to_markdown(index=False),
            )


if __name__ == "__main__":
    main()
