# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Custom all-reduce residual fusion and the fused AR+RMSNorm
``num_norm_rows`` / ``skip_residual`` options.

Inputs are small integers, so every all-reduced and residual-added value is
exact in bf16: those are compared with atol=0, RMSNorm outputs with a bf16
tolerance. Every case is checked eager and on one CUDA graph replay; the
table reports the eager time. Repeated graph replays are not timed: the
reduce-scatter + local RMSNorm path (m > 80) can hang under them, as
test_fused_ar_rms.py -t 8 -s 128,7168 -g 1 does on main.
"""

import argparse
import itertools
import os
import pickle
import tempfile

import pandas as pd
import torch
import torch.distributed as dist

import aiter
from aiter import dtypes
from aiter.dist.utils import get_distributed_init_method, get_ip, get_open_port
from aiter.jit.utils.chip_info import get_gfx
from aiter.test_common import checkAllclose, run_perftest

# gfx1250 has no residual all-reduce kernel (custom_all_reduce_residual -> None).
SUPPORTED_GFX = ["gfx942", "gfx950"]
EPS = 1e-6


def int_inputs(seed, tp, rank, shape, dtype):
    """This rank's input, the fp32 sum over all ranks, and a shared residual."""
    gen = torch.Generator().manual_seed(seed)
    parts = torch.randint(-8, 9, (tp, *shape), generator=gen, dtype=torch.int16)
    residual = torch.randint(-8, 9, shape, generator=gen, dtype=torch.int16)
    reduced = parts.sum(dim=0, dtype=torch.float32).cuda()
    return parts[rank].cuda().to(dtype), reduced, residual.cuda().to(dtype)


def run_torch_rmsnorm(x32, w):
    # Reference only: fp32 math. Not timed, not in the table.
    return x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + EPS) * w.float()


def capture(ca, fn):
    graph = torch.cuda.CUDAGraph()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with ca.capture(), torch.cuda.graph(graph, stream=stream):
        outs = fn()
    torch.cuda.current_stream().wait_stream(stream)
    dist.barrier()
    return graph, outs


def run_case(ca, fn, x, refs, tols, graphs):
    """Check ``fn(x)`` against ``refs`` eagerly and on graph replay, then time
    it eagerly. Returns (us, max err).

    The captured input is a separate buffer refreshed before replay and the
    outputs are poisoned, so replaying stale data cannot pass. ``graphs`` keeps
    every graph and its buffers alive for the whole sweep: custom all-reduce
    caches peer IPC addresses by local pointer, so a freed buffer reused by a
    later capture would replay with stale peer pointers.
    """
    errs = []
    eager = fn(x)
    for out, ref, (rtol, atol) in zip(eager, refs, tols):
        errs.append(checkAllclose(ref, out.float(), rtol=rtol, atol=atol, msg="eager"))

    x_in = torch.zeros_like(x)
    graph, outs = capture(ca, lambda: fn(x_in))
    graphs.append((graph, x_in, outs))
    for out in outs:
        out.fill_(float("nan"))
    x_in.copy_(x)
    graph.replay()
    for out, ref, (rtol, atol) in zip(outs, refs, tols):
        errs.append(checkAllclose(ref, out.float(), rtol=rtol, atol=atol, msg="graph"))

    _, us = run_perftest(fn, x)
    return us, max(errs)


def test_all_reduce_residual(ca, tp, rank, m, n, dtype, graphs):
    x, reduced, residual = int_inputs(m * n, tp, rank, (m, n), dtype)
    ref = (reduced + residual.float()).to(dtype).float()
    candidates = {
        "fused": lambda inp: (ca.custom_all_reduce_residual(inp, residual),),
    }
    # The path the model ran before: custom all-reduce, then a separate add.
    # Skipped at TP6, where plain 1-stage all-reduce faults: THREAD_NUM / 6
    # leaves threads 510-511 reading a seventh rank's (null) pointer.
    if tp != 6:
        candidates["split"] = lambda inp: (ca.custom_all_reduce(inp) + residual,)
    numel = m * n
    flops = numel * tp  # (tp - 1) reduce adds + 1 residual add per element
    nbytes = 3 * numel * x.element_size()  # input, residual, output

    ret = {}
    for name, fn in candidates.items():
        us, err = run_case(ca, fn, x, [ref], [(0, 0)], graphs)
        ret[f"{name} us"] = us
        ret[f"{name} TFLOPS"] = flops / us / 1e6
        ret[f"{name} TB/s"] = nbytes / us / 1e6
        ret[f"{name} err"] = err
    return ret


def test_fused_ar_rms_rows(ca, tp, rank, m, n, norm_div, skip_residual, dtype, graphs):
    norm_rows = m // norm_div
    x, reduced, residual = int_inputs(m * 3 + norm_rows, tp, rank, (m, n), dtype)
    w = (torch.arange(n, device="cuda") % 7 + 1).to(dtype) / 4
    res_ref = reduced if skip_residual else reduced + residual.float()
    out_ref = run_torch_rmsnorm(res_ref, w)[:norm_rows]

    def fused(inp):
        # A caller-provided out holding only the normalized rows, as in the model.
        out = torch.empty((norm_rows, n), dtype=dtype, device="cuda")
        return ca.custom_fused_ar_rms(
            inp,
            residual,
            w,
            EPS,
            False,
            out=out,
            num_norm_rows=norm_rows,
            skip_residual=skip_residual,
        )

    numel = m * n
    flops = numel * tp + norm_rows * n * 4
    nbytes = (3 * numel + norm_rows * n) * x.element_size()
    # 5e-3 is above bf16 output rounding (~0.4%) but catches a ~1% RMS scale error.
    us, err = run_case(ca, fused, x, [out_ref, res_ref], [(5e-3, 5e-3), (0, 0)], graphs)
    return {
        "norm_rows": norm_rows,
        "fused us": us,
        "fused TFLOPS": flops / us / 1e6,
        "fused TB/s": nbytes / us / 1e6,
        "fused err": err,
    }


def expect_error(exc, what, fn):
    try:
        fn()
    except exc:
        return
    raise AssertionError(f"{what} was accepted")


def check_rejections(ca):
    """Unsupported option/buffer combinations raise instead of running."""
    from aiter.ops import custom_all_reduce as ops

    m, n, norm_rows = 16, 7168, 4
    bf16 = dtypes.bf16
    x = torch.ones((m, n), dtype=bf16, device="cuda")
    w = torch.ones(n, dtype=bf16, device="cuda")
    pool = ca._pool["input"]

    def rows(k, dtype=bf16):
        return torch.empty((k, n), dtype=dtype, device="cuda")

    def native_ar_res(out):
        ops.all_reduce_residual(ca._ptr, x, out, x, pool.data_ptr, pool.max_size)

    expect_error(RuntimeError, "undersized out", lambda: native_ar_res(rows(m - 1)))
    expect_error(RuntimeError, "fp32 out", lambda: native_ar_res(rows(m, dtypes.fp32)))
    # Same numel as x, but a strided view the kernels would write densely past.
    strided = torch.empty((m, 2 * n), dtype=bf16, device="cuda")[:, ::2]
    expect_error(RuntimeError, "strided out", lambda: native_ar_res(strided))

    # gfx1250 runs only one-shot all-reduce kernels, at every size.
    saved, ca._is_gfx1250 = ca._is_gfx1250, True
    try:
        assert ca.uses_1stage_ar(rows(4096)), "gfx1250 reported a 2-stage allreduce"
    finally:
        ca._is_gfx1250 = saved

    if ca.world_size == 6:
        return  # fused allreduce rmsnorm has no TP6 kernel

    def native_ar_rms(res_inp, res_out, out, use_1stage=False, skip=False):
        ops.fused_allreduce_rmsnorm(
            ca._ptr,
            x,
            res_inp,
            res_out,
            out,
            w,
            EPS,
            pool.data_ptr,
            pool.max_size,
            use_1stage,
            False,
            norm_rows,
            skip,
        )

    for opts in ({"skip_residual": True}, {"num_norm_rows": norm_rows}):
        expect_error(
            ValueError,
            f"1-stage {opts}",
            lambda opts=opts: ca.custom_fused_ar_rms(x, x, w, EPS, True, **opts),
        )
        # A padded width takes the pad kernel, which has no row options.
        expect_error(
            ValueError,
            f"padded {opts}",
            lambda opts=opts: ca.custom_fused_ar_rms(
                x, x, w, EPS, False, out_hidden_dim=n + 128, **opts
            ),
        )
    expect_error(
        RuntimeError,
        "native 1-stage skip_residual",
        lambda: native_ar_rms(x, rows(m), rows(m), use_1stage=True, skip=True),
    )
    expect_error(
        RuntimeError,
        "out shorter than num_norm_rows",
        lambda: native_ar_rms(x, rows(m), rows(norm_rows - 1)),
    )
    expect_error(
        RuntimeError,
        "res_out shorter than m",
        lambda: native_ar_rms(x, rows(m - 1), rows(norm_rows)),
    )
    expect_error(
        RuntimeError,
        "res_inp shorter than m",
        lambda: native_ar_rms(rows(m - 1), rows(m), rows(norm_rows)),
    )
    expect_error(
        RuntimeError,
        "fp32 out",
        lambda: native_ar_rms(x, rows(m), rows(norm_rows, dtypes.fp32)),
    )
    expect_error(
        RuntimeError,
        "strided res_out",
        lambda: native_ar_rms(x, strided, rows(norm_rows)),
    )
    expect_error(
        RuntimeError,
        "strided out",
        lambda: native_ar_rms(x, rows(m), strided[:norm_rows]),
    )
    # num_norm_rows == m is the default behavior, so 1-stage accepts it.
    ca.custom_fused_ar_rms(x, x, w, EPS, True, num_norm_rows=m)
    torch.cuda.synchronize()


def sweep(rank, tp, ar_cases, rms_cases, dtype, init_method, result_path):
    """Run every case on one rank inside a single process group; rank 0 saves
    every rank's rows to ``result_path``."""
    from aiter.dist.device_communicators.custom_all_reduce import CustomAllreduce

    torch.cuda.set_device(rank)
    dist.init_process_group("gloo", world_size=tp, rank=rank, init_method=init_method)
    ca = None
    graphs = []
    try:
        ca = CustomAllreduce(dist.group.WORLD, torch.device(f"cuda:{rank}"))
        assert not ca.disabled, "requires custom allreduce"
        check_rejections(ca)
        ar_rows = [
            test_all_reduce_residual(ca, tp, rank, m, n, dtype, graphs)
            for m, n in ar_cases
        ]
        rms_rows = [
            test_fused_ar_rms_rows(ca, tp, rank, *case, dtype, graphs)
            for case in rms_cases
        ]
        torch.cuda.synchronize()
        per_rank = [None] * tp
        dist.all_gather_object(per_rank, (ar_rows, rms_rows))
        if rank == 0:
            with open(result_path, "wb") as f:
                pickle.dump(per_rank, f)
    finally:
        graphs.clear()
        if ca is not None:
            ca.close()
        dist.destroy_process_group()


def run_tp(tp, ar_cases, rms_cases, dtype):
    """Run the sweep on ``tp`` ranks; one row per case, slowest rank's time."""
    init_method = get_distributed_init_method(get_ip(), get_open_port())
    # spawn (not a Pool) so a rank that dies fails the run instead of hanging it
    with tempfile.TemporaryDirectory() as tmp:
        result_path = os.path.join(tmp, "rows.pkl")
        torch.multiprocessing.spawn(
            sweep,
            args=(tp, ar_cases, rms_cases, dtype, init_method, result_path),
            nprocs=tp,
            join=True,
        )
        with open(result_path, "rb") as f:
            per_rank = pickle.load(f)

    def merge(table, cases, keys):
        rows = []
        for i, case in enumerate(cases):
            row = {"tp": tp, **dict(zip(keys, case)), "dtype": dtype}
            for rank_rows in (r[table] for r in per_rank):
                for k, v in rank_rows[i].items():
                    # slowest rank: max time and error, min throughput
                    agg = max if k.endswith((" us", " err")) else min
                    row[k] = agg(row.get(k, v), v)
            rows.append(row)
        return rows

    return (
        merge(0, ar_cases, ("m", "n")),
        merge(1, rms_cases, ("m", "n", "norm_div", "skip_residual")),
    )


def main():
    if get_gfx() not in SUPPORTED_GFX:
        aiter.logger.warning("custom allreduce residual unsupported on %s", get_gfx())
        return

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="custom all-reduce residual fusion and fused AR+RMSNorm row options",
    )
    parser.add_argument(
        "-t",
        "--tp",
        type=int,
        nargs="*",
        choices=[2, 4, 6, 8],
        default=[8, 6],
        help="TP sizes; TP6 takes the naive residual kernel (default: 8 6)",
    )
    parser.add_argument(
        "-d",
        "--dtype",
        type=dtypes.str2Dtype,
        nargs="*",
        default=[dtypes.bf16],
        help="data types (default: bf16)",
    )
    parser.add_argument(
        "-s",
        "--mn",
        type=dtypes.str2tuple,
        nargs="*",
        # 1-stage below the size crossover and 2-stage above it; 1x819208 is a
        # 2-stage size whose 16-byte packs do not split evenly across ranks.
        default=[(1, 7168), (8, 7168), (64, 7168), (512, 7168), (1, 819208)],
        help="all_reduce_residual shapes m,n (default: Kimi-K3 hidden 7168)",
    )
    parser.add_argument(
        "--rms_mn",
        type=dtypes.str2tuple,
        nargs="*",
        # 16 rows take the single-kernel 2-stage path, 160 rows
        # reduce-scatter + local RMSNorm. n=2880 (360 lanes, not whole 32-lane
        # groups) must also fall back to reduce-scatter + local RMSNorm.
        default=[(16, 7168), (160, 7168), (32, 2880)],
        help="fused AR+RMSNorm shapes m,n",
    )
    parser.add_argument(
        "--norm_div",
        type=int,
        nargs="*",
        default=[4, 1],
        help="num_norm_rows = m // norm_div",
    )
    parser.add_argument(
        "--skip_residual",
        type=dtypes.str2bool,
        nargs="*",
        default=[True, False],
    )
    args = parser.parse_args()

    tps = [tp for tp in args.tp if tp <= torch.cuda.device_count()]
    if tps != args.tp:
        aiter.logger.warning(
            "skipping TP %s: only %d GPUs",
            sorted(set(args.tp) - set(tps)),
            torch.cuda.device_count(),
        )
    rms_cases = list(itertools.product(args.rms_mn, args.norm_div, args.skip_residual))
    rms_cases = [(m, n, div, skip) for (m, n), div, skip in rms_cases]

    ar_df, rms_df = [], []
    for tp, dtype in itertools.product(tps, args.dtype):
        # fused allreduce rmsnorm has no TP6 kernel
        ar_rows, rms_rows = run_tp(tp, args.mn, rms_cases if tp != 6 else [], dtype)
        ar_df += ar_rows
        rms_df += rms_rows

    ar_df, rms_df = pd.DataFrame(ar_df), pd.DataFrame(rms_df)
    aiter.logger.info(
        "all_reduce_residual summary (markdown):\n%s", ar_df.to_markdown(index=False)
    )
    aiter.logger.info(
        "fused_ar_rms num_norm_rows/skip_residual summary (markdown):\n%s",
        rms_df.to_markdown(index=False),
    )
    errs = pd.concat([ar_df.filter(like=" err"), rms_df.filter(like=" err")], axis=1)
    assert (errs.fillna(0) == 0).all().all(), "mismatch vs. reference; see tables"


if __name__ == "__main__":
    main()
