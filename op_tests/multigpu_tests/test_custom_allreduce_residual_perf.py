# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""A/B timing of the fused custom all-reduce paths against the split paths
they replace, on any aiter tree.

- ``test_ar_residual``: ``custom_all_reduce_residual`` ("fused") vs.
  ``custom_all_reduce`` followed by a separate add ("split"), plus plain
  ``custom_all_reduce`` alone ("ar"). Shapes are ``m, n`` (o_proj output).
- ``test_ar_rms_rows``: ``custom_fused_ar_rms`` with ``num_norm_rows`` and
  ``skip_residual`` ("fused") vs. ``custom_all_reduce`` followed by
  ``aiter.rms_norm`` on the first rows ("split"). Shapes are ``tokens, n``;
  the buffer has ``tokens * rows_per_token`` rows and its first ``tokens``
  rows are normalized, as in Kimi-K3's ``[latent | shared]`` MoE buffer
  (``n = 3584``, 3 rows per token), and residual_out is the input, in place.

The "fused" candidates exist only on trees that have them; elsewhere their
columns are empty, so the same command times base and head. Inputs are small
integers or multiples of 1/64, so reduced values are exact and compared at
atol=0. ``<name> min us`` is the fastest rank's time, ``<name> max us`` the
slowest rank's (mostly rank arrival skew); TFLOPS and TB/s use the fastest
rank. Times are eager by default (``-g 0``). ``-g 1`` times CUDA-graph
replays, which can hang on main too: repeated replays of plain bf16 custom
all-reduce ("ar", "split") hang intermittently, and so do those of the
reduce-scatter + local RMSNorm path (more than 80 rows), as
test_fused_ar_rms.py -t 8 -s 128,7168 -g 1 does.
Shapes above 64 MiB need AITER_CUSTOM_AR_MAX_SIZE raised to stay on custom
all-reduce.
"""

import argparse
import inspect
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
from aiter.test_common import benchmark, checkAllclose, run_perftest

SUPPORTED_GFX = ["gfx942", "gfx950"]
EPS = 1e-6
# Set by sweep() on each rank: the communicator ("ca"), whether to time graph
# replays ("graph") and the list keeping every captured graph alive ("graphs").
# Kept out of the @benchmark args, which are the table's columns.
_RANK_STATE = {}


def has_ar_residual(ca):
    return hasattr(ca, "custom_all_reduce_residual")


def has_row_options(ca):
    return "num_norm_rows" in inspect.signature(ca.custom_fused_ar_rms).parameters


def rank_inputs(seed, tp, shape, scale):
    """This rank's input, the fp32 sum over all ranks and a residual shared by
    every rank: integers in [-8, 8] (scale 1) or [-255, 255] / scale."""
    lim = 8 if scale == 1 else 255
    gen = torch.Generator().manual_seed(seed)
    parts = torch.randint(-lim, lim + 1, (tp, *shape), generator=gen) / scale
    residual = torch.randint(-lim, lim + 1, shape, generator=gen) / scale
    return parts[dist.get_rank()].cuda(), parts.sum(dim=0).cuda(), residual.cuda()


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


def run_case(make, x0, refs, tols):
    """Check the outputs of ``make(x)()`` against ``refs`` on a fresh copy of
    ``x0``, then time it eagerly or, in graph mode, check one replay and time
    the replays. Returns (us, max err)."""
    ca, graphs = _RANK_STATE["ca"], _RANK_STATE["graphs"]
    x = x0.clone()
    fn = make(x)
    errs = [
        checkAllclose(ref, out.float(), rtol=rtol, atol=atol, msg="eager")
        for out, ref, (rtol, atol) in zip(fn(), refs, tols)
    ]
    if _RANK_STATE["graph"]:
        x = x0.clone()
        graph, outs = capture(ca, make(x))
        graphs.append((graph, x, outs))
        for out in outs:
            out.fill_(float("nan"))
        x.copy_(x0)  # an output may alias the input (in-place residual_out)
        graph.replay()
        errs += [
            checkAllclose(ref, out.float(), rtol=rtol, atol=atol, msg="graph")
            for out, ref, (rtol, atol) in zip(outs, refs, tols)
        ]
        fn = graph.replay
    # The closure reads the same buffers every call; num_rotate_args=1 keeps
    # perftest from cloning inputs it cannot see.
    _, us = run_perftest(fn, num_rotate_args=1)
    return us, max(errs)


@benchmark()
def test_ar_residual(tp, m, n, dtype):
    ca = _RANK_STATE["ca"]
    x, reduced, residual = rank_inputs(m * n, tp, (m, n), 64)
    x, residual = x.to(dtype), residual.to(dtype)
    # All-reduce rounds to dtype, then the add rounds again.
    ar_ref = reduced.to(dtype).float()
    res_ref = (ar_ref + residual.float()).to(dtype).float()
    candidates = {
        "ar": (lambda x: lambda: (ca.custom_all_reduce(x),), ar_ref),
        "split": (lambda x: lambda: (ca.custom_all_reduce(x) + residual,), res_ref),
    }
    if has_ar_residual(ca):
        candidates["fused"] = (
            lambda x: lambda: (ca.custom_all_reduce_residual(x, residual),),
            res_ref,
        )
    numel = m * n
    ret = {"gfx": get_gfx()}
    for name, (make, ref) in candidates.items():
        us, err = run_case(make, x, [ref], [(0, 0)])
        adds = tp if name != "ar" else tp - 1  # reduce adds (+ residual add)
        tensors = 3 if name != "ar" else 2  # input (+ residual), output
        ret[f"{name} us"] = us
        ret[f"{name} TFLOPS"] = numel * adds / us / 1e6
        ret[f"{name} TB/s"] = tensors * numel * x.element_size() / us / 1e6
        ret[f"{name} err"] = err
    return ret


@benchmark()
def test_ar_rms_rows(tp, tokens, n, rows_per_token, dtype):
    ca = _RANK_STATE["ca"]
    m = tokens * rows_per_token
    x, reduced, _ = rank_inputs(m * n, tp, (m, n), 1)
    x = x.to(dtype)
    w = (torch.arange(n, device="cuda") % 7 + 1).to(dtype) / 4
    norm_ref = run_torch_rmsnorm(reduced[:tokens], w)
    # The caller-provided out holding only the normalized rows, as in the model.
    out = torch.empty((tokens, n), dtype=dtype, device="cuda")

    def split(x):
        def fn():
            ar = ca.custom_all_reduce(x)
            return aiter.rms_norm(ar[:tokens], w, EPS), ar

        return fn

    def fused(x):
        def fn():
            return ca.custom_fused_ar_rms(
                x,
                x,
                w,
                EPS,
                False,
                residual_out=x,
                out=out,
                num_norm_rows=tokens,
                skip_residual=True,
            )

        return fn

    candidates = {"split": split}
    if has_row_options(ca):
        candidates["fused"] = fused
    numel = m * n
    flops = numel * (tp - 1) + tokens * n * 4
    nbytes = (2 * numel + tokens * n) * x.element_size()  # input, reduced, out
    ret = {"gfx": get_gfx(), "rows": m}
    for name, make in candidates.items():
        # 5e-3 is above bf16 output rounding (~0.4%) but catches a ~1% RMS error.
        us, err = run_case(make, x, [norm_ref, reduced], [(5e-3, 5e-3), (0, 0)])
        ret[f"{name} us"] = us
        ret[f"{name} TFLOPS"] = flops / us / 1e6
        ret[f"{name} TB/s"] = nbytes / us / 1e6
        ret[f"{name} err"] = err
    return ret


def sweep(rank, tp, graph, ar_cases, rms_cases, dtype, init_method, result_path):
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
        _RANK_STATE.update(ca=ca, graph=graph, graphs=graphs)
        ar_rows = [test_ar_residual(tp, m, n, dtype) for m, n in ar_cases]
        rms_rows = [test_ar_rms_rows(tp, *case, dtype) for case in rms_cases]
        torch.cuda.synchronize()
        per_rank = [None] * tp
        dist.all_gather_object(per_rank, (ar_rows, rms_rows))
        if rank == 0:
            with open(result_path, "wb") as f:
                pickle.dump(per_rank, f)
    finally:
        _RANK_STATE.clear()
        graphs.clear()
        if ca is not None:
            ca.close()
        dist.destroy_process_group()


def run_tp(tp, graph, ar_cases, rms_cases, dtype):
    """Run the sweep on ``tp`` ranks; one row per case."""
    init_method = get_distributed_init_method(get_ip(), get_open_port())
    # spawn (not a Pool) so a rank that dies fails the run instead of hanging it
    with tempfile.TemporaryDirectory() as tmp:
        result_path = os.path.join(tmp, "rows.pkl")
        torch.multiprocessing.spawn(
            sweep,
            args=(tp, graph, ar_cases, rms_cases, dtype, init_method, result_path),
            nprocs=tp,
            join=True,
        )
        with open(result_path, "rb") as f:
            per_rank = pickle.load(f)

    def merge(table):
        """Rank 0's @benchmark row with each time as the fastest ("min us") and
        slowest ("max us") rank, throughput from the fastest, the max error."""
        rows = []
        for case_rows in zip(*(r[table] for r in per_rank)):
            row = {"graph": graph}
            for k, v in case_rows[0].items():
                if k.endswith(" us"):
                    vals = [r[k] for r in case_rows]
                    row[k.replace(" us", " min us")] = min(vals)
                    row[k.replace(" us", " max us")] = max(vals)
                elif k.endswith((" TFLOPS", " TB/s", " err")):
                    row[k] = max(r[k] for r in case_rows)
                else:
                    row[k] = v
            rows.append(row)
        return rows

    return merge(0), merge(1)


def main():
    if get_gfx() not in SUPPORTED_GFX:
        aiter.logger.warning("custom allreduce fusion A/B unsupported on %s", get_gfx())
        return

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="A/B timing of fused custom all-reduce paths vs. the split paths",
    )
    parser.add_argument(
        "-t",
        "--tp",
        type=int,
        nargs="*",
        choices=[2, 4, 8],
        default=[8],
        help="TP sizes (default: 8)",
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
        "-g",
        "--graph",
        type=int,
        nargs="*",
        choices=[0, 1],
        default=[0],
        help="0: eager timing, 1: CUDA-graph replay timing (default: 0)",
    )
    parser.add_argument(
        "-s",
        "--mn",
        type=dtypes.str2tuple,
        nargs="*",
        default=[(1, 7168), (16, 7168), (64, 7168), (256, 7168), (1024, 7168)],
        help="test_ar_residual shapes m,n (default: Kimi-K3 hidden 7168)",
    )
    parser.add_argument(
        "--rows_mn",
        type=dtypes.str2tuple,
        nargs="*",
        default=[(1, 3584), (4, 3584), (16, 3584), (24, 3584), (64, 3584)],
        help="test_ar_rms_rows shapes tokens,n (default: Kimi-K3 latent 3584)",
    )
    parser.add_argument(
        "--rows_per_token",
        type=int,
        nargs="*",
        default=[3],
        help="buffer rows per token; the first `tokens` rows are normalized "
        "(default: 3, Kimi-K3 [latent | shared])",
    )
    args = parser.parse_args()

    tps = [tp for tp in args.tp if tp <= torch.cuda.device_count()]
    if tps != args.tp:
        aiter.logger.warning(
            "skipping TP %s: only %d GPUs",
            sorted(set(args.tp) - set(tps)),
            torch.cuda.device_count(),
        )
    rms_cases = [
        (tokens, n, rpt)
        for (tokens, n), rpt in itertools.product(args.rows_mn, args.rows_per_token)
    ]

    ar_df, rms_df = [], []
    for tp, graph, dtype in itertools.product(tps, args.graph, args.dtype):
        ar_rows, rms_rows = run_tp(tp, graph, args.mn, rms_cases, dtype)
        ar_df += ar_rows
        rms_df += rms_rows

    ar_df, rms_df = pd.DataFrame(ar_df), pd.DataFrame(rms_df)
    aiter.logger.info(
        "AR + residual: fused vs. split (markdown):\n%s", ar_df.to_markdown(index=False)
    )
    aiter.logger.info(
        "AR + RMSNorm on the first rows: fused vs. split (markdown):\n%s",
        rms_df.to_markdown(index=False),
    )
    errs = pd.concat([df.filter(like=" err") for df in (ar_df, rms_df)], axis=1)
    assert (errs.fillna(0) == 0).all().all(), "mismatch vs. reference; see tables"


if __name__ == "__main__":
    main()
