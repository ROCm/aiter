# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Lamport fused all-reduce + residual add + RMSNorm POC on gfx9
(CustomAllreduce.lamportFusedArRms): correctness and the numbers to write its
dispatch against the existing routes.

Candidates, all timed as CUDA graphs (CALLS_PER_GRAPH calls a graph, median of
GRAPH_REPLAYS replays, best of TIMING_ROUNDS rounds over all candidates,
slowest rank):
  car_fused       tensor_model_parallel_fused_allreduce_rmsnorm, the production
                  route (1-stage / 2-stage / quick reduce / split by size)
  car_1stage      the 1-stage fused kernel forced (pull + two barriers)
  lamport_ar+norm tensor_model_parallel_all_reduce (lamport one-shot / two-shot
                  inside its ranges) then rmsnorm2d_fwd_with_add
  lamport         lamportFusedArRms, kernel and block picked by the C++ side
  flat_row / flat_bsB   flat kernel, a row per block / B packs per block
  ws_gG           wave split kernel, G groups of 64 positions per block
  two_shot        two-shot kernel (chunk g of every row reduced on rank g,
                  broadcast), blocks a row picked by the C++ side
  ts_sS           two-shot kernel, S blocks a row
A lamport variant shows up only where the shape fits it (whole blocks a row,
the 640KB one-shot cap for all but two_shot, at most 128 rows for two_shot).

err columns are the fraction of mismatching elements, 0 expected:
  lamport variants  out and residual_out bit for bit against car_1stage (same
                    rounding steps), residual_out also against the torch
                    reference; NaN matches NaN
  the others        checkAllclose of out against the torch reference
Every case injects the 0xffff sentinel bit pattern on one rank, which must not
hang and must come out as NaN. "resize err" runs the lamport variants eagerly
with the row count shrinking and growing (two_shot also past the one-shot
cap), interleaved with one-shot all-reduces on the same inbox and with
gemma_norm on and off; the hidden sizes run one after another, so two_shot
also sees its rows change length.
"""

import argparse
import itertools
from multiprocessing import Pool, freeze_support, set_start_method

import pandas as pd
import torch

import aiter
from aiter import dtypes, rmsnorm2d_fwd_with_add
from aiter.dist.communication_op import (
    tensor_model_parallel_all_reduce,
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
from aiter.test_common import benchmark, checkAllclose

SUPPORTED_GFX = ["gfx942", "gfx950"]
PACK_BYTES = 16
# kLamportMaxPacks in custom_all_reduce.cuh: the one-shot cap
ONE_SHOT_MAX_BYTES = 512 * 80 * PACK_BYTES
WAVE_POSITIONS = 64  # kLamportWaveSize
EPS = 1e-6
SENTINEL = -1  # 0xffff as int16
CALLS_PER_GRAPH = 10
GRAPH_REPLAYS = 20
TIMING_ROUNDS = 3
# name -> (kernel, block) of lamportFusedArRms
FLAT_BLOCKS = {"flat_row": 0, "flat_bs64": 64, "flat_bs128": 128, "flat_bs256": 256}
WS_GROUPS = {"ws_g1": 1, "ws_g2": 2, "ws_g4": 4, "ws_g8": 8}
TWO_SHOT_MAX_ROWS = 128  # kLamportArRms2MaxRows
TS_SPLITS = {"ts_s1": 1, "ts_s2": 2, "ts_s4": 4}
# default sweep cap: the lamport two-shot all-reduce range at TP4 / TP8
DEFAULT_MAX_BYTES = 1 << 20


def lamportVariants(tp, m, n, dtype):
    """{name: (kernel, block)} of the lamport variants that fit this shape."""
    row_packs = n * dtype.itemsize // PACK_BYTES
    if row_packs % WAVE_POSITIONS != 0 or row_packs > 1024:
        return {}
    out = {}
    if m <= TWO_SHOT_MAX_ROWS:
        out["two_shot"] = (3, 0)
        for name, sp in TS_SPLITS.items():
            if row_packs % (32 * sp) == 0:
                out[name] = (3, sp)
    if m * n * dtype.itemsize > ONE_SHOT_MAX_BYTES:
        return out
    out["lamport"] = (0, 0)
    for name, bs in FLAT_BLOCKS.items():
        if bs == 0 or (bs < row_packs and row_packs % bs == 0):
            out[name] = (1, bs)
    for name, g in WS_GROUPS.items():
        if tp * g <= 16 and row_packs % (g * WAVE_POSITIONS) == 0:
            out[name] = (2, g)
    return out


def roundLikeKernel(t, dtype):
    # module_custom_all_reduce builds with software round-to-nearest-even
    # (OPUS_FP32_to_BF16_DEFAULT=5); gfx950 rounds to nearest even in hardware.
    return t.to(dtype)


def runTorch(xs, res, w, dtype, gemma_norm=False):
    """Reference only: rank-order fp32 sum rounded to dtype, + residual in fp32,
    residual_out rounded, fp32 RMSNorm of the unrounded sum."""
    acc = xs[0].to(dtypes.fp32)
    for x in xs[1:]:
        acc += x.to(dtypes.fp32)
    acc = roundLikeKernel(acc, dtype).to(dtypes.fp32) + res.to(dtypes.fp32)
    res_out = roundLikeKernel(acc, dtype)
    rms = torch.rsqrt(acc.pow(2).mean(-1, keepdim=True) + EPS)
    wf = w.to(dtypes.fp32) + (1.0 if gemma_norm else 0.0)
    return roundLikeKernel(acc * rms * wf, dtype), res_out


def makeInputs(tp, rank, case_idx, m, n, dtype):
    """Every rank's input from one seed, so each rank has the reference; the
    residual and weight are the same on every rank, as in a TP model."""
    gen = torch.Generator(device="cuda").manual_seed(case_idx)
    xs = [
        torch.randn((m, n), dtype=dtype, device="cuda", generator=gen)
        for _ in range(tp)
    ]
    res = torch.randn((m, n), dtype=dtype, device="cuda", generator=gen)
    w = torch.rand((n,), dtype=dtype, device="cuda", generator=gen) + 0.5
    # one rank sends the sentinel bit pattern: no hang, NaN rows out
    xs[case_idx % tp].view(torch.int16).view(-1)[[0, -1]] = SENTINEL
    return xs[rank], xs, res, w


def bitErr(out, ref):
    """Fraction of elements whose bits differ, NaN matching NaN."""
    both_nan = torch.isnan(out) & torch.isnan(ref)
    bad = (out.view(torch.int16) != ref.view(torch.int16)) & ~both_nan
    return bad.sum().item() / max(bad.numel(), 1)


def captureGraph(fn, graphs):
    """Capture CALLS_PER_GRAPH calls in one graph, replay it a few times and
    return (graph, out). Graphs and buffers stay alive in `graphs`: custom
    all-reduce caches peer IPC addresses of captured buffers by local pointer."""
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


def arThenNorm(x, res, w):
    ar = tensor_model_parallel_all_reduce(x)
    out = torch.empty_like(ar)
    res_out = torch.empty_like(ar)
    rmsnorm2d_fwd_with_add(out, ar, res, res_out, w, EPS, False, 0)
    return out, res_out


@benchmark()
def testLamportFusedArRms(tp, m, n, dtype, case_idx=0, rank=0, ca=None, graphs=None):
    x, xs, res, w = makeInputs(tp, rank, case_idx, m, n, dtype)
    ref_out, ref_res = runTorch(xs, res, w, dtype)
    nbytes = x.numel() * x.element_size()
    candidates = {
        "car_fused": lambda: tensor_model_parallel_fused_allreduce_rmsnorm(
            x, res, w, EPS
        ),
        "car_1stage": lambda: ca.custom_fused_ar_rms(x, res, w, EPS, use_1stage=True),
        "lamport_ar+norm": lambda: arThenNorm(x, res, w),
    }
    variants = lamportVariants(tp, m, n, dtype)
    for name, (kernel, block) in variants.items():
        candidates[name] = lambda k=kernel, b=block: ca.lamportFusedArRms(
            x, res, w, EPS, kernel=k, block=b
        )

    ret = {"gfx": get_gfx_runtime()}
    outs = {}
    captured = {}
    for name, fn in candidates.items():
        graph, (out, res_out) = captureGraph(fn, graphs)
        captured[name] = graph
        outs[name] = (out, res_out)
        if name in variants:
            base_out, base_res = outs["car_1stage"]
            ret[f"{name} err"] = max(
                bitErr(out, base_out), bitErr(res_out, base_res), bitErr(res_out, ref_res)
            )
        else:
            ret[f"{name} err"] = checkAllclose(
                ref_out.to(dtypes.fp32),
                out.to(dtypes.fp32),
                mask=~torch.isnan(ref_out),
                printLog=rank == 0,
                msg=f"{name}: tp{tp} {m=} {n=}",
            )
    # timing rounds over all candidates, the best round each: clock and
    # power drift over a case then hits every candidate alike
    for name in candidates:
        ret[f"{name} us"] = float("inf")
    for _ in range(TIMING_ROUNDS):
        for name, graph in captured.items():
            ret[f"{name} us"] = min(ret[f"{name} us"], timeGraph(graph))
    for name in candidates:
        ret[f"{name} TB/s"] = nbytes / ret[f"{name} us"] / 1e6
    return ret


def checkSizeChanges(ca, tp, rank, n, dtype):
    """Eager lamport variants with the row count shrinking and growing, one
    shot all-reduces of odd sizes in between (same inbox and counters) and
    gemma_norm alternating; three passes so every buffer is cycled. Returns
    the worst bit error against car_1stage / the torch reference."""
    row_bytes = n * dtype.itemsize
    top = ONE_SHOT_MAX_BYTES // row_bytes
    big = min(TWO_SHOT_MAX_ROWS, 2 * top)  # two_shot only
    ms = [top, 1, big, 3, top // 2 + 1, 2, big, top, 1]
    pack_numel = PACK_BYTES // dtype.itemsize
    ar_positions = [1, 65, 5000, 192, 512 * 80]
    errs = []
    step = 0
    for _ in range(3):
        for m in ms:
            for name, (kernel, block) in lamportVariants(tp, m, n, dtype).items():
                gemma = step % 2 == 1
                x, xs, res, w = makeInputs(tp, rank, 3000 + step, m, n, dtype)
                out, res_out = ca.lamportFusedArRms(
                    x, res, w, EPS, gemma_norm=gemma, kernel=kernel, block=block
                )
                base_out, base_res = ca.custom_fused_ar_rms(
                    x, res, w, EPS, use_1stage=True, gemma_norm=gemma
                )
                _, ref_res = runTorch(xs, res, w, dtype, gemma)
                errs += [
                    bitErr(out, base_out),
                    bitErr(res_out, base_res),
                    bitErr(res_out, ref_res),
                ]
                # a one-shot all-reduce on the same inbox
                pos = ar_positions[step % len(ar_positions)]
                y, ys, _, _ = makeInputs(tp, rank, 5000 + step, 1, pos * pack_numel, dtype)
                ar_out = torch.empty_like(y)
                aiter.lamportAllReduce(ca._ptr, y, ar_out, 1)
                acc = ys[0].to(dtypes.fp32)
                for t in ys[1:]:
                    acc += t.to(dtypes.fp32)
                errs.append(bitErr(ar_out, roundLikeKernel(acc, dtype)))
                step += 1
    torch.cuda.synchronize()
    return max(errs)


def worker(tp, rank, shapes, dtype, init_method):
    torch.cuda.set_device(rank)
    set_custom_all_reduce(True)
    init_distributed_environment(
        world_size=tp, rank=rank, distributed_init_method=init_method
    )
    ensure_model_parallel_initialized(tp, 1)
    ca = get_tp_group().device_communicator.ca_comm
    assert ca.lamport_enabled, "the lamport path is off (AITER_CUSTOM_AR_DISABLE_LAMPORT?)"

    resize_err = {}
    for n in sorted({n for _, n in shapes}):
        resize_err[n] = checkSizeChanges(ca, tp, rank, n, dtype)
    graphs = []
    rows = []
    for i, (m, n) in enumerate(shapes):
        row = testLamportFusedArRms(
            tp, m, n, dtype, case_idx=i, rank=rank, ca=ca, graphs=graphs
        )
        for k in ("case_idx", "rank", "ca", "graphs"):
            row.pop(k, None)
        row["resize err"] = resize_err[n]
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


def defaultMs(n, dtype):
    """1, 2, 4, ... up to DEFAULT_MAX_BYTES and TWO_SHOT_MAX_ROWS rows."""
    ms, m = [], 1
    while m * n * dtype.itemsize <= DEFAULT_MAX_BYTES and m <= TWO_SHOT_MAX_ROWS:
        ms.append(m)
        m *= 2
    return ms


def main():
    if get_gfx_runtime() not in SUPPORTED_GFX:
        aiter.logger.warning(
            "lamport fused AR + RMSNorm unsupported on %s; skipping", get_gfx_runtime()
        )
        return

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="lamport fused all-reduce + RMSNorm POC: correctness and tuning numbers",
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
        "-n", type=int, nargs="*", default=[4096, 7168, 8192],
        help="hidden sizes (multiple of 512, at most 8192), e.g. -n 7168",
    )
    parser.add_argument(
        "-m", type=int, nargs="*", default=None,
        help="token counts, default 1, 2, 4, ... up to 1MB / 128 rows",
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
        df = []
        for tp in args.tp:
            if args.mn is not None:
                shapes = args.mn
            else:
                shapes = [
                    (m, n)
                    for n in args.n
                    for m in (args.m if args.m is not None else defaultMs(n, dtype))
                ]
            df += runTp(tp, shapes, dtype, args.timeout)
        df = pd.DataFrame(df)
        df["KB"] = df["m"] * df["n"] * dtype.itemsize // 1024
        lead = ["tp", "m", "n", "dtype", "KB", "gfx"]
        df = df[lead + [c for c in df.columns if c not in lead]]
        aiter.logger.info(
            "lamport fused AR + RMSNorm summary (markdown):\n%s",
            df.to_markdown(index=False),
        )
        bad = [
            c for c in df.columns
            if c.endswith(" err") and not c.startswith(("car_", "lamport_ar"))
            and (df[c].fillna(0) != 0).any()
        ]
        assert not bad, f"lamport variants not bit exact: {bad}"


if __name__ == "__main__":
    freeze_support()
    set_start_method("spawn", force=True)
    main()
