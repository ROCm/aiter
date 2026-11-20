# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""All-reduce of x + y with the add fused into the gfx9 lamport kernels
(CustomAllreduce.lamportAllReduceAdd, POC): correctness and the numbers to
write tensor_model_parallel_all_reduce_add's gfx9 dispatch. Once that dispatch
lands, the cases move into test_ar_add_lamport.py (gfx1250 today).

out = the sum over ranks of each rank's x + y (x + y rounded to the dtype, the
rank-order fp32 sum rounded once): the bits of all_reduce(x + y).

Candidates, all timed as CUDA graphs (CALLS_PER_GRAPH calls a graph, median of
GRAPH_REPLAYS replays, best of TIMING_ROUNDS rounds over all candidates,
slowest rank):
  prod_route         tensor_model_parallel_all_reduce_add, what the model runs
                     today: on gfx9 x + y, then all_reduce (the lamport one-shot
                     / two-shot inside their ranges, the original dispatch past)
  add+lamport_1shot  x + y, then the lamport one-shot forced
  add+lamport_2shot  x + y, then the lamport two-shot forced
  fused_1shot        lamportAllReduceAdd, one-shot (up to 640KB)
  fused_2shot        lamportAllReduceAdd, two-shot (up to 2MB)
  fused_auto         lamportAllReduceAdd, the kernel all_reduce picks for the
                     size (inside the lamport ranges only)
A candidate shows up only where the size fits it.

err columns are the fraction of mismatching elements against the reference,
bit for bit for every lamport candidate (two NaNs count as equal), 0 expected;
prod_route past the lamport ranges takes the original dispatch, whose rank
order may differ, so it is checked with checkAllclose instead. Every case
injects the 0xffff inbox sentinel into one rank's x (must not hang, comes out
NaN). "mixed err" runs, eagerly, the fused kernels interleaved with plain
lamport all-reduces of other sizes on the same inboxes (sizes with a partial
last block, injected 0xffff, odd ranks' operands off 16B alignment) and then a
captured graph replayed with fresh inputs. The run fails (assert) on any
non-zero lamport err.
"""

import argparse
import itertools
from multiprocessing import Pool, freeze_support, set_start_method

import pandas as pd
import torch

import aiter
from aiter import dtypes
from aiter.dist.communication_op import (
    tensor_model_parallel_all_reduce_add,
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
from aiter.test_common import benchmark, checkAllclose

SUPPORTED_GFX = ["gfx942", "gfx950"]
PACK_BYTES = 16
ONE_SHOT_MAX_BYTES = 512 * 80 * PACK_BYTES  # kLamportMaxPacks: the forced one-shot
TWO_SHOT_MAX_BYTES = 2 << 20  # kLamportTwoShotMaxBytes: the forced two-shot
SENTINEL = -1  # 0xffff as int16
CALLS_PER_GRAPH = 10
GRAPH_REPLAYS = 20
TIMING_ROUNDS = 3
DEFAULT_MS = [1, 2, 4, 8, 16, 32, 48, 64, 96, 128, 192, 256]
# (m, n) of the mixed sequence: 3 x 1000 leaves the last block partial, (1, 8)
# is one pack
MIXED = [(409, 1024), (3, 1000), (48, 5120), (1, 8), (64, 7168), (8, 4096)]


def makeInputs(tp, rank, seed, m, n, dtype, nan=True):
    """This rank's x, y and the reference, every rank's operands from one seed:
    the rank-order fp32 sum of each rank's x + y (rounded to the dtype),
    rounded once. With nan, rank seed % tp sends the 0xffff bit pattern."""
    gen = torch.Generator(device="cuda").manual_seed(seed)
    xs = torch.randn((2, tp, m, n), dtype=torch.float32, device="cuda", generator=gen)
    xs = xs.to(dtype)
    if nan:
        flat = xs[0, seed % tp].view(torch.int16).view(-1)
        flat[0] = SENTINEL
        flat[-1] = SENTINEL
    sums = xs[0] + xs[1]
    acc = sums[0].float()
    for r in range(1, tp):
        acc = acc + sums[r].float()
    return xs[0, rank].contiguous(), xs[1, rank].contiguous(), acc.to(dtype)


def bitErr(out, ref):
    """Fraction of elements whose bits differ, NaN matching NaN."""
    if out.shape != ref.shape or out.dtype != ref.dtype:
        return 1.0
    both_nan = torch.isnan(out) & torch.isnan(ref)
    bad = (out.view(torch.int16) != ref.view(torch.int16)) & ~both_nan
    return bad.sum().item() / max(bad.numel(), 1)


def offsetView(t):
    """t's values one element into a fresh storage, so off 16B alignment."""
    storage = torch.empty(t.numel() + 1, dtype=t.dtype, device=t.device)
    return storage[1:].view(t.shape).copy_(t)


def captureGraph(fn, graphs):
    """CALLS_PER_GRAPH calls of fn in one graph, replayed a few times; returns
    (graph, out). graphs keeps graphs and buffers alive: custom all-reduce caches
    peer IPC addresses of captured buffers by local pointer."""
    graph = torch.cuda.CUDAGraph()
    with graph_capture() as gc, torch.cuda.graph(graph, stream=gc.stream):
        for _ in range(CALLS_PER_GRAPH):
            out = fn()
    graphs.append((graph, out))
    for _ in range(3):
        graph.replay()
    torch.cuda.synchronize()
    return graph, out


def timeGraph(graph):
    """us per call: median of GRAPH_REPLAYS replays."""
    get_tp_group().barrier()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(GRAPH_REPLAYS)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(GRAPH_REPLAYS)]
    for s, e in zip(starts, ends):
        s.record()
        graph.replay()
        e.record()
    torch.cuda.synchronize()
    ms = sorted(s.elapsed_time(e) for s, e in zip(starts, ends))[GRAPH_REPLAYS // 2]
    return ms * 1000 / CALLS_PER_GRAPH


def lamportCandidates(ca, tp, nbytes):
    """{name: kernel} of the fused / forced lamport kernels that fit nbytes."""
    out = {}
    if 0 < nbytes <= ONE_SHOT_MAX_BYTES and nbytes % PACK_BYTES == 0:
        out["1shot"] = 1
    if 0 < nbytes <= TWO_SHOT_MAX_BYTES and nbytes % (PACK_BYTES * tp) == 0:
        out["2shot"] = 2
    return out


def lamportPlain(ca, t, kernel):
    out = torch.empty_like(t)
    aiter.lamportAllReduce(ca._ptr, t, out, kernel)
    return out


@benchmark()
def testAllReduceAdd(tp, m, n, dtype, case_idx=0, rank=0, ca=None, graphs=None):
    x, y, ref = makeInputs(tp, rank, case_idx, m, n, dtype)
    nbytes = x.numel() * x.element_size()
    candidates = {"prod_route": lambda: tensor_model_parallel_all_reduce_add(x, y)}
    lamport = lamportCandidates(ca, tp, nbytes)
    for name, k in lamport.items():
        candidates[f"add+lamport_{name}"] = lambda k=k: lamportPlain(ca, x + y, k)
    for name, k in lamport.items():
        candidates[f"fused_{name}"] = lambda k=k: ca.lamportAllReduceAdd(x, y, k)
    if aiter.lamportKernel(ca._ptr, x.numel()) != 0:
        candidates["fused_auto"] = lambda: ca.lamportAllReduceAdd(x, y, 0)

    ret = {"gfx": get_gfx_runtime(), "KB": nbytes / 1024}
    captured = {}
    for name, fn in candidates.items():
        graph, out = captureGraph(fn, graphs)
        captured[name] = graph
        if name == "prod_route":
            ret[f"{name} err"] = checkAllclose(
                ref.to(dtypes.fp32),
                out.to(dtypes.fp32),
                mask=~torch.isnan(ref),
                printLog=rank == 0,
                msg=f"{name}: tp{tp} {m=} {n=}",
            )
        else:
            ret[f"{name} err"] = bitErr(out, ref)
    # timing rounds over all candidates, the best round each: clock and power
    # drift over a case then hits every candidate alike
    for name in candidates:
        ret[f"{name} us"] = float("inf")
    for _ in range(TIMING_ROUNDS):
        for name, graph in captured.items():
            ret[f"{name} us"] = min(ret[f"{name} us"], timeGraph(graph))
    # algorithm bandwidth: the message one rank contributes (x + y, one tensor)
    for name in candidates:
        ret[f"{name} TB/s"] = nbytes / ret[f"{name} us"] / 1e6
    return ret


def checkMixed(ca, tp, rank, dtype):
    """Eager: the fused kernels interleaved with plain lamport all-reduces of
    other sizes on the same inboxes and round counters, injected 0xffff, the
    odd ranks' operands off 16B alignment (every rank still takes the lamport
    kernel, the odd ones through a copy); then a graph of fused calls replayed
    with fresh inputs. Returns the worst bit error."""
    errs = []
    step = 0
    for _ in range(3):
        for m, n in MIXED:
            x, y, ref = makeInputs(tp, rank, 7000 + step, m, n, dtype, nan=step % 3 == 0)
            if step % 4 == 1 and rank % 2:
                x, y = offsetView(x), offsetView(y)
            for k in lamportCandidates(ca, tp, x.numel() * x.element_size()).values():
                errs.append(bitErr(ca.lamportAllReduceAdd(x, y, k), ref))
                # a plain all-reduce of another size on the same inbox
                pm, pn = MIXED[(step + 1) % len(MIXED)]
                px, py, pref = makeInputs(tp, rank, 9000 + step, pm, pn, dtype, nan=False)
                s = px + py
                pk = lamportCandidates(ca, tp, s.numel() * s.element_size())
                if k in pk.values():
                    errs.append(bitErr(lamportPlain(ca, s, k), pref))
            step += 1
    torch.cuda.synchronize()

    # a captured graph replayed with fresh inputs in the same buffers
    graphs = []
    for m, n in [(48, 5120), (3, 1000), (128, 7168)]:
        x, y, _ = makeInputs(tp, rank, 100 + m, m, n, dtype)
        for k in lamportCandidates(ca, tp, x.numel() * x.element_size()).values():
            graph, out = captureGraph(lambda: ca.lamportAllReduceAdd(x, y, k), graphs)
            for seed in range(3):
                nx, ny, ref = makeInputs(tp, rank, 200 + seed * 7 + m, m, n, dtype, nan=seed == 1)
                x.copy_(nx)
                y.copy_(ny)
                get_tp_group().barrier()
                graph.replay()
                torch.cuda.synchronize()
                errs.append(bitErr(out, ref))
    del graphs
    return max(errs)


def worker(tp, rank, shapes, dtype, init_method):
    torch.cuda.set_device(rank)
    set_custom_all_reduce(True)
    init_distributed_environment(world_size=tp, rank=rank, distributed_init_method=init_method)
    ensure_model_parallel_initialized(tp, 1)
    ca = get_tp_group().device_communicator.ca_comm
    assert ca.lamport_enabled, "the lamport path is off (AITER_CUSTOM_AR_DISABLE_LAMPORT?)"

    mixed_err = checkMixed(ca, tp, rank, dtype)
    graphs = []
    rows = []
    for i, (m, n) in enumerate(shapes):
        row = testAllReduceAdd(tp, m, n, dtype, case_idx=i, rank=rank, ca=ca, graphs=graphs)
        for k in ("case_idx", "rank", "ca", "graphs"):
            row.pop(k, None)
        row["mixed err"] = mixed_err
        rows.append(row)

    torch.cuda.synchronize()
    get_tp_group().barrier()
    del graphs
    destroy_model_parallel()
    destroy_distributed_environment()
    return rows


def runTp(tp, shapes, dtype, timeout):
    """One process group; per case the slowest rank's time, the worst error."""
    init_method = get_distributed_init_method(get_ip(), get_open_port())
    with Pool(processes=tp) as pool:
        jobs = [
            pool.apply_async(worker, (tp, rank, shapes, dtype, init_method))
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
    if get_gfx_runtime() not in SUPPORTED_GFX:
        aiter.logger.warning(
            "lamport all-reduce + add (gfx9 POC) unsupported on %s; skipping",
            get_gfx_runtime(),
        )
        return

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="lamport all-reduce of x + y (gfx9 POC): correctness and tuning numbers",
    )
    parser.add_argument(
        "-d", "--dtype", type=dtypes.str2Dtype, nargs="*", default="bf16,",
        help="data type, e.g. -d bf16",
    )
    parser.add_argument(
        "-t", "--tp", type=int, nargs="*", default=[2, 4, 8],
        help="tensor parallel sizes (2/4/8), e.g. -t 8",
    )
    parser.add_argument(
        "-n", type=int, nargs="*", default=[4096, 5120, 7168],
        help="hidden sizes, e.g. -n 7168",
    )
    parser.add_argument(
        "-m", type=int, nargs="*", default=DEFAULT_MS,
        help="token counts; sizes past the 2MB two-shot cap are dropped",
    )
    parser.add_argument(
        "-s", "--mn", type=dtypes.str2tuple, nargs="*", default=None,
        help="explicit (tokens, hidden) shapes, overrides -m/-n, e.g. -s 16,7168",
    )
    parser.add_argument(
        "--timeout", type=int, default=1800,
        help="seconds per process group before declaring a hang",
    )
    args = parser.parse_args()

    for dtype in args.dtype:
        if args.mn is not None:
            shapes = [tuple(s) for s in args.mn]
        else:
            shapes = [
                (m, n)
                for n, m in itertools.product(args.n, args.m)
                if m * n * dtype.itemsize <= TWO_SHOT_MAX_BYTES
            ]
        df = []
        for tp in args.tp:
            df += runTp(tp, shapes, dtype, args.timeout)
        df = pd.DataFrame(df)
        lead = ["tp", "m", "n", "dtype", "KB", "gfx"]
        df = df[lead + [c for c in df.columns if c not in lead]]
        aiter.logger.info(
            "lamport all-reduce + add summary (markdown):\n%s", df.to_markdown(index=False)
        )
        bad = [
            c for c in df.columns
            if c.endswith(" err") and c != "prod_route err" and (df[c].fillna(0) != 0).any()
        ]
        assert not bad, f"lamport candidates not bit exact: {bad}"


if __name__ == "__main__":
    freeze_support()
    set_start_method("spawn", force=True)
    main()
