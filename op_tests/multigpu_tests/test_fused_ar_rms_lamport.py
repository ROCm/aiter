# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""gfx1250 Lamport one-shot fused allreduce + residual + RMSNorm (+ per-token FP8).

tensor_model_parallel_fused_allreduce_rmsnorm, and with quant_type="per_token"
tensor_model_parallel_fused_allreduce_rmsnorm_quant, take the Lamport kernel up
to its size cap (TP2 16 MiB, TP4 8 MiB; AITER_CUSTOM_AR_LAMPORT_RMS_MAX_SIZE).
Candidates, all with the same (out, residual_out), and the scale with FP8 (-q 1):
  prod_route     the production entry, whatever it routes to
  lamport        CustomAllreduce.lamport_fused_ar_rms(_quant), where the route takes it
  lamport+quant  FP8: lamport_fused_ar_rms + per-token quant, the route before the
                 FP8 Lamport kernel
  car_fused      the barrier fused kernel ar_rmsnorm_gfx1250, one-stage
  car_ar+norm    custom allreduce + rmsnorm2d_fwd_with_add (+ per-token quant), the
                 route below 4 MiB before the Lamport kernel
  rccl+norm      RCCL allreduce + rmsnorm2d_fwd_with_add (+ per-token quant)

Correctness: every candidate against a torch reference (an error ratio of 5%,
checkAllclose's tol_err_ratio, fails), and the Lamport kernel bit for bit
against car_fused (two NaNs count as equal); with FP8 also lamport+quant, and
the bf16/fp16 output of emit_bf16. A second table runs one mixed sequence per TP
on the one shared inbox: row widths and row counts change
call to call (block a row and cluster shapes), bf16 and FP8 calls alternate,
0xffff is injected now and then, the odd ranks hold the input, residual and
weight off 16B alignment (every rank must still take the Lamport kernel), and a
CUDA graph of three sizes is replayed between eager calls. Any bit difference
fails the run.

Timing: run_perftest times kernels on one rank and adds calls of its own, which a
cross-rank collective cannot take. Instead a CUDA graph of CALLS calls is
replayed REPLAYS times back to back; median per rank, slowest rank reported.
"""

import argparse
import itertools
import statistics
from multiprocessing import Pool, set_start_method

import pandas as pd
import torch
import torch.distributed as dist

import aiter
from aiter import dtypes
from aiter.dist.communication_op import (
    tensor_model_parallel_fused_allreduce_rmsnorm,
    tensor_model_parallel_fused_allreduce_rmsnorm_quant,
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
from aiter.jit.utils.chip_info import get_gfx
from aiter.test_common import checkAllclose

SUPPORTED_GFX = ["gfx1250"]
EPS = 1e-6
CALLS = 10
REPLAYS = 20
# (n, m) of the mixed sequence: hidden 256 to 8192 in whole waves of packs, row
# counts on both sides of the CU count (256), where a row stops being a cluster
STRESS_NS = [2048, 7168, 256, 4096, 6144, 8192, 5120]
STRESS_MS = [1, 300, 3, 255, 256, 17, 1]
GRAPH_CASES = [(7168, 4), (4096, 300), (2048, 64)]


def make_inputs(tp, rank, seed, m, n, dtype, nan=False):
    """Every rank makes the inputs of all ranks, so each can build the reference.
    The residual and weight are the same on all ranks, as in a TP model."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    xs = torch.randn((tp, m, n), dtype=torch.float32, device="cuda", generator=g)
    xs = xs.to(dtype)
    residual = torch.randn((m, n), dtype=torch.float32, device="cuda", generator=g)
    residual = residual.to(dtype)
    weight = torch.randn((n,), dtype=torch.float32, device="cuda", generator=g)
    weight = weight.to(dtype)
    if nan:
        # 0xffff is the inbox sentinel: it must not hang, its row is NaN
        xs[0].view(torch.int16)[m // 2, n // 3] = -1
    return xs[rank].contiguous(), xs, residual, weight


def run_torch(xs, residual, weight, eps):
    """Rank-order fp32 sum, rounded to the dtype, then add + RMSNorm in fp32."""
    acc = xs[0].to(dtypes.fp32)
    for r in range(1, xs.shape[0]):
        acc = acc + xs[r].to(dtypes.fp32)
    x = acc.to(xs.dtype).to(dtypes.fp32) + residual.to(dtypes.fp32)
    res_out = x.to(xs.dtype)
    inv_rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    out = (x * inv_rms * weight.to(dtypes.fp32)).to(xs.dtype)
    return out, res_out


def offset_view(t):
    """t's values one element into a fresh storage, so off 16B alignment."""
    storage = torch.empty(t.numel() + 1, dtype=t.dtype, device=t.device)
    return storage[1:].view(t.shape).copy_(t)


_BITS = {1: torch.int8, 2: torch.int16, 4: torch.int32}


def bit_mismatch(a, b):
    """Elements whose bits differ; two NaNs count as equal."""
    assert a.shape == b.shape and a.dtype == b.dtype, (a.shape, a.dtype, b.dtype)
    bits = _BITS[a.element_size()]
    same = (a.view(bits) == b.view(bits)) | (
        torch.isnan(a.float()) & torch.isnan(b.float())
    )
    return int((~same).sum().item())


def pair_mismatch(got, want):
    """The most elements whose bits differ in any output: (out, residual_out),
    and the scale (and the bf16/fp16 output) with FP8."""
    assert len(got) == len(want)
    return max(bit_mismatch(g, w) for g, w in zip(got, want))


def route(quant):
    if quant:
        return lambda x, residual, weight: (
            tensor_model_parallel_fused_allreduce_rmsnorm_quant(
                x, residual, weight, EPS, quant_type="per_token"
            )
        )
    return lambda x, residual, weight: tensor_model_parallel_fused_allreduce_rmsnorm(
        x, residual, weight, EPS
    )


def car_fused(ca, x, residual, weight, quant=False):
    if quant:
        return ca.custom_fused_ar_rms_quant(x, residual, weight, EPS, use_1stage=True)
    return ca.custom_fused_ar_rms(x, residual, weight, EPS, use_1stage=True)


def per_token_quant(out, res_out):
    q, scale = aiter.get_hip_quant(aiter.QuantType.per_Token)(
        out, quant_dtype=dtypes.fp8
    )
    return q, res_out, scale


def norm_with_add(ar, residual, weight, quant=False):
    out = torch.empty_like(ar)
    res_out = torch.empty_like(ar)
    aiter.rmsnorm2d_fwd_with_add(out, ar, residual, res_out, weight, EPS, False, 0)
    return per_token_quant(out, res_out) if quant else (out, res_out)


def make_candidates(ca, x, residual, weight, rccl_buf, quant):
    group = get_tp_group().device_group
    prod = route(quant)
    cands = {"prod_route": lambda: prod(x, residual, weight)}
    if ca.should_lamport_ar_rms(x, residual, weight, x.shape[-1]):
        if quant:
            cands["lamport"] = lambda: ca.lamport_fused_ar_rms_quant(
                x, residual, weight, EPS
            )
            cands["lamport+quant"] = lambda: per_token_quant(
                *ca.lamport_fused_ar_rms(x, residual, weight, EPS)
            )
        else:
            cands["lamport"] = lambda: ca.lamport_fused_ar_rms(x, residual, weight, EPS)
    cands["car_fused"] = lambda: car_fused(ca, x, residual, weight, quant)
    cands["car_ar+norm"] = lambda: norm_with_add(
        ca.custom_all_reduce(x), residual, weight, quant
    )
    # in place on its own buffer, as a model reduces its GEMM output; the buffer
    # keeps growing over the graph replays, which does not change the time
    cands["rccl+norm"] = lambda: (
        dist.all_reduce(rccl_buf, group=group),
        norm_with_add(rccl_buf, residual, weight, quant),
    )[1]
    return cands


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


def test_case(ca, tp, rank, m, n, dtype, quant):
    """One (m, n): every candidate's error and time, Lamport bits vs car_fused."""
    x, xs, residual, weight = make_inputs(tp, rank, n * 1000 + m, m, n, dtype)
    ref = run_torch(xs, residual, weight, EPS)
    rccl_buf = x.clone()
    cands = make_candidates(ca, x, residual, weight, rccl_buf, quant)
    base = car_fused(ca, x, residual, weight, quant)
    # traffic of the op on one rank: input, residual, weight in; out, residual
    # out and the FP8 scales out
    es = x.element_size()
    nbytes = (3 * m * n + n) * es + (m * n + 4 * m if quant else m * n * es)
    ret = {"tp": tp, "m": m, "n": n, "quant": quant, "KB": m * n * es / 1024}
    diff = -1
    if "lamport" in cands:
        diff = pair_mismatch(cands["lamport"](), base)
        if quant:
            diff = max(diff, pair_mismatch(cands["lamport+quant"](), base))
            got = ca.lamport_fused_ar_rms_quant(
                x, residual, weight, EPS, emit_bf16=True
            )
            want = ca.custom_fused_ar_rms_quant(
                x, residual, weight, EPS, use_1stage=True, emit_bf16=True
            )
            diff = max(diff, pair_mismatch(got, want))
    ret["lamport bitdiff"] = diff
    # FP8: the dequantized out is within half a code (2^-4) of the reference
    tol = {"rtol": 0.07, "atol": 1e-2} if quant else {}
    for name, fn in cands.items():
        rccl_buf.copy_(x)
        got = fn()
        out = got[0].float() * got[2] if quant else got[0].float()
        ret[f"{name} err"] = max(
            checkAllclose(ref[0].float(), out, printLog=False, **tol),
            checkAllclose(ref[1].float(), got[1].float(), printLog=False),
        )
    for name, fn in cands.items():
        us = time_graph(fn)
        ret[f"{name} us"] = us
        ret[f"{name} TB/s"] = nbytes / us / 1e6
    return ret


def check_mixed_sequence(ca, tp, rank, dtype):
    """Shapes and bf16/FP8 change call to call on the one inbox, eager and in a
    graph, against car_fused bit for bit. Returns the worst mismatch and the
    calls checked."""
    max_bytes = ca._lamport_max_bytes
    worst, calls = 0, 0

    def check(seed, m, n, quant, nan=False):
        nonlocal worst, calls
        x, _, residual, weight = make_inputs(tp, rank, seed, m, n, dtype, nan=nan)
        assert ca.should_lamport_ar_rms(x, residual, weight, n), (m, n)
        got = route(quant)(x, residual, weight)
        want = car_fused(ca, x, residual, weight, quant)
        worst = max(worst, pair_mismatch(got, want))
        calls += 1

    # the largest row count of each width that fits the cap, then the sequence;
    # the second pass flips which calls are FP8
    cases = [(n, max_bytes // (n * dtype.itemsize)) for n in STRESS_NS]
    cases += list(zip(STRESS_NS, STRESS_MS))
    for i, (n, m) in enumerate(cases * 2):
        check(i, m, n, (i + i // len(cases)) % 2 == 1, nan=i % 3 == 0)

    # the odd ranks' operands one element into their storage: every rank must
    # still take the Lamport kernel, the odd ones copying them first; car_fused
    # gets the aligned originals
    for (n, m), quant in itertools.product(zip(STRESS_NS, STRESS_MS), (False, True)):
        x, _, residual, weight = make_inputs(tp, rank, 9000 + n, m, n, dtype)
        want = car_fused(ca, x, residual, weight, quant)
        if rank % 2:
            x, residual, weight = (offset_view(t) for t in (x, residual, weight))
        assert ca.should_lamport_ar_rms(x, residual, weight, n), (m, n)
        worst = max(worst, pair_mismatch(route(quant)(x, residual, weight), want))
        calls += 1

    # a graph of three sizes, bf16 and FP8, replayed between eager calls of
    # other sizes
    graph_cases = list(itertools.product(GRAPH_CASES, (False, True)))
    graph_in = [
        make_inputs(tp, rank, 7000 + n, m, n, dtype) for (n, m), _ in graph_cases
    ]
    want = [
        car_fused(ca, x, res, w, quant)
        for (x, _, res, w), (_, quant) in zip(graph_in, graph_cases)
    ]
    if rank % 2:  # unaligned operands inside the graph as well
        x, xs, res, w = graph_in[0]
        graph_in[0] = (offset_view(x), xs, offset_view(res), offset_view(w))
    graph = torch.cuda.CUDAGraph()
    with graph_capture() as gc, torch.cuda.graph(graph, stream=gc.stream):
        graph_out = [
            route(quant)(x, res, w)
            for (x, _, res, w), (_, quant) in zip(graph_in, graph_cases)
        ]
    for i in range(4):
        graph.replay()
        torch.cuda.synchronize()
        for got, w in zip(graph_out, want):
            worst = max(worst, pair_mismatch(got, w))
            calls += 1
        n, m = cases[i % len(cases)]
        check(100 + i, m, n, i % 2 == 1)
    graph.reset()
    return {"tp": tp, "calls": calls, "max bitdiff": worst}


def worker(tp, rank, ms, ns, dtype, quants, init_method):
    torch.cuda.set_device(rank)
    set_custom_all_reduce(True)
    init_distributed_environment(
        world_size=tp, rank=rank, distributed_init_method=init_method
    )
    ensure_model_parallel_initialized(tp, 1)
    ca = get_tp_group().device_communicator.ca_comm
    assert ca is not None and not ca.disabled, "custom allreduce is disabled"
    dist.all_reduce(torch.zeros(1, device="cuda"), group=get_tp_group().device_group)

    rows = [
        test_case(ca, tp, rank, m, n, dtype, quant)
        for quant, n, m in itertools.product(quants, ns, ms)
    ]
    mixed = (
        check_mixed_sequence(ca, tp, rank, dtype) if ca._lamport_max_bytes > 0 else {}
    )

    torch.cuda.synchronize()
    get_tp_group().barrier()
    destroy_model_parallel()
    destroy_distributed_environment()
    return rows, mixed


def merge_ranks(per_rank):
    """Slowest rank's time and the worst error of any rank, case by case."""
    merged = []
    for cells in zip(*per_rank):
        row = dict(cells[0])
        for k in row:
            if k.endswith((" us", " err", "bitdiff")):
                row[k] = max(c[k] for c in cells)
            elif k.endswith(" TB/s"):
                row[k] = min(c[k] for c in cells)
        merged.append(row)
    return merged


def run_tp(tp, ms, ns, dtype, quants):
    init_method = get_distributed_init_method(get_ip(), get_open_port())
    with Pool(processes=tp) as pool:
        jobs = [
            pool.apply_async(worker, (tp, r, ms, ns, dtype, quants, init_method))
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
            "lamport fused ar + rmsnorm unsupported on %s; skipping", get_gfx()
        )
        return
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="Lamport fused allreduce + RMSNorm vs the other gfx1250 routes",
    )
    parser.add_argument("-t", "--tp", type=int, nargs="*", default=[2, 4])
    parser.add_argument(
        "-m", type=int, nargs="*", default=[1, 4, 16, 64, 128, 256, 512, 1024]
    )
    parser.add_argument("-n", type=int, nargs="*", default=[2048, 4096, 7168, 8192])
    parser.add_argument(
        "-d", "--dtype", type=dtypes.str2Dtype, nargs="*", default=[dtypes.bf16]
    )
    parser.add_argument(
        "-q",
        "--quant",
        type=int,
        nargs="*",
        choices=[0, 1],
        default=[0, 1],
        help="0: bf16/fp16 out, 1: per-token FP8 out",
    )
    args = parser.parse_args()
    ngpus = torch.cuda.device_count()

    set_start_method("spawn", force=True)
    bad = []
    for dtype in args.dtype:
        rows, mixed = [], []
        for tp in args.tp:
            if tp > ngpus:
                aiter.logger.warning("tp%d needs %d GPUs, have %d", tp, tp, ngpus)
                continue
            quants = [bool(q) for q in args.quant]
            tp_rows, tp_mixed = run_tp(tp, args.m, args.n, dtype, quants)
            rows += tp_rows
            if tp_mixed:
                mixed.append(tp_mixed)
        for quant in sorted({r["quant"] for r in rows}):
            df = pd.DataFrame([r for r in rows if r["quant"] == quant])
            aiter.logger.info(
                "lamport fused ar + rmsnorm%s %s summary (markdown):\n%s",
                " + per-token FP8" if quant else "",
                dtype,
                df.drop(columns="quant").to_markdown(index=False, floatfmt=".3g"),
            )
        if mixed:
            aiter.logger.info(
                "lamport fused ar + rmsnorm %s mixed sequence (markdown):\n%s",
                dtype,
                pd.DataFrame(mixed).to_markdown(index=False),
            )
        bad += [r for r in rows if r["lamport bitdiff"] > 0]
        # an error ratio at checkAllclose's tol_err_ratio fails, as it does there
        bad += [r for r in rows if any(r[k] >= 0.05 for k in r if k.endswith(" err"))]
        bad += [r for r in mixed if r["max bitdiff"] > 0]
    assert not bad, f"off the reference, or the Lamport kernel off car_fused: {bad}"


if __name__ == "__main__":
    main()
