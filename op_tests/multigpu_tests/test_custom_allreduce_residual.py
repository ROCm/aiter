# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Correctness + perf tests for custom all-reduce residual fusion and the
fused AR+RMSNorm ``num_norm_rows`` / ``skip_residual`` options.

Inputs are small integers so every sum is exact in bf16; reduced tensors are
compared with atol=0, RMSNorm outputs with a bf16 tolerance. Each sweep point
runs on every rank of a ``torch.multiprocessing`` world; rank 0 collects the
rows and ``main()`` prints one markdown table per test function.
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
from aiter.test_common import benchmark, checkAllclose, run_perftest

logger = aiter.logger

# all_reduce_residual has no gfx1250 kernel (CustomAllreduce sets it to None).
SUPPORTED_GFX = ["gfx942", "gfx950"]

EPS = 1e-6

# Per-rank state filled in by _worker, so the @benchmark fns' call args are
# only the sweep axes (they become the summary table's left columns).
_CTX = {}


def _int_inputs(seed, tp, rank, shape, dtype, device):
    gen = torch.Generator().manual_seed(seed)
    parts = torch.randint(-8, 9, (tp, *shape), generator=gen, dtype=torch.int16)
    residual = torch.randint(-8, 9, shape, generator=gen, dtype=torch.int16)
    x = parts[rank].to(device, dtype)
    reduced = parts.sum(dim=0, dtype=torch.float32).to(device)
    return x, reduced, residual.to(device, dtype)


def run_torch_residual(reduced, residual, dtype):
    """Reference for all_reduce_residual: sum over ranks + residual."""
    return (reduced + residual.float()).to(dtype)


def run_torch_rmsnorm(x32, w):
    """Reference RMSNorm in fp32."""
    return x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + EPS) * w.float()


def _capture_graph(ca, fn):
    """Capture ``fn`` once under ca.capture() (registered IPC inputs, as in the
    model's CUDA graph) and return (graph, outputs)."""
    graph = torch.cuda.CUDAGraph()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with ca.capture(), torch.cuda.graph(graph, stream=stream):
        outs = fn()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    _CTX["graphs"].append(graph)  # keep alive until ca.close()
    return graph, outs if isinstance(outs, tuple) else (outs,)


def _sync_ranks():
    torch.cuda.synchronize()
    dist.barrier()


@benchmark()
def test_all_reduce_residual(tp_size, shape, dtype):
    ca, rank, device = _CTX["ca"], _CTX["rank"], _CTX["device"]
    x, reduced, residual = _int_inputs(shape[0], tp_size, rank, shape, dtype, device)
    ref = run_torch_residual(reduced, residual, dtype)

    # Graph candidate: x_in is captured (and IPC-registered) as a zero buffer,
    # then refreshed with the real input before replay.
    x_in = torch.zeros_like(x)
    graph, (gout,) = _capture_graph(
        ca, lambda: ca.custom_all_reduce_residual(x_in, residual)
    )
    gout.fill_(float("nan"))
    x_in.copy_(x)

    candidates = {
        "eager": (lambda: ca.custom_all_reduce_residual(x, residual), None),
        "graph": (graph.replay, gout),
    }

    numel = x.numel()
    # tp-1 cross-rank adds + 1 residual add per element.
    flops = tp_size * numel
    # Local traffic per rank: read input + residual, write output.
    nbytes = 3 * numel * x.element_size()

    ret = {"gfx": get_gfx()}
    for name, (fn, out_buf) in candidates.items():
        _sync_ranks()
        out, us = run_perftest(fn)
        _sync_ranks()
        out = out if out_buf is None else out_buf
        err = checkAllclose(
            ref.to(dtypes.fp32),
            out.to(dtypes.fp32),
            rtol=0,
            atol=0,
            msg=f"rank{rank} all_reduce_residual {name} {shape=}",
        )
        assert err == 0, f"all_reduce_residual {name} mismatch: {shape=} TP{tp_size}"
        ret[f"{name} us"] = us
        ret[f"{name} TFLOPS"] = flops / us / 1e6
        ret[f"{name} TB/s"] = nbytes / us / 1e6
        ret[f"{name} err"] = err
    return ret


@benchmark()
def test_fused_ar_rms_row_opts(tp_size, m, n, num_norm_rows, skip_residual, dtype):
    ca, rank, device = _CTX["ca"], _CTX["rank"], _CTX["device"]
    w = (torch.arange(n, device=device) % 7 + 1).to(dtype) / 4
    x, reduced, residual = _int_inputs(
        m * 3 + num_norm_rows, tp_size, rank, (m, n), dtype, device
    )
    r32 = reduced if skip_residual else reduced + residual.float()
    ref_norm = run_torch_rmsnorm(r32, w)[:num_norm_rows]

    # The caller preallocates an `out` holding only the normalized rows.
    def call(inp, out):
        return ca.custom_fused_ar_rms(
            inp,
            residual,
            w,
            EPS,
            False,
            out=out,
            num_norm_rows=num_norm_rows,
            skip_residual=skip_residual,
        )

    eager_out = torch.empty((num_norm_rows, n), dtype=dtype, device=device)
    x_in = torch.zeros_like(x)
    graph_out = torch.empty_like(eager_out)
    graph, (_gout, gres) = _capture_graph(ca, lambda: call(x_in, graph_out))
    x_in.copy_(x)

    candidates = {
        "eager": (lambda: call(x, eager_out), eager_out),
        "graph": (graph.replay, graph_out),
    }

    elem = x.element_size()
    # AR adds + residual add + RMSNorm (square, sum, scale, weight) on norm rows.
    flops = tp_size * m * n + 4 * num_norm_rows * n
    # Read input (+ residual unless skipped) + weight, write res_out and out rows.
    nbytes = ((2 if skip_residual else 3) * m * n + n + num_norm_rows * n) * elem

    ret = {"gfx": get_gfx()}
    for name, (fn, out_buf) in candidates.items():
        out_buf.fill_(float("nan"))
        _sync_ranks()
        res, us = run_perftest(fn)
        _sync_ranks()
        res_out = gres if name == "graph" else res[1]
        ctx = f"rank{rank} fused_ar_rms {name} {m=} {num_norm_rows=} {skip_residual=}"
        res_err = checkAllclose(
            r32, res_out.float(), rtol=0, atol=0, msg=f"{ctx} res_out"
        )
        err = checkAllclose(
            ref_norm, out_buf.float(), rtol=2e-2, atol=2e-2, msg=f"{ctx} out"
        )
        assert res_err == 0 and err == 0, f"{ctx}: mismatch TP{tp_size}"
        ret[f"{name} us"] = us
        ret[f"{name} TFLOPS"] = flops / us / 1e6
        ret[f"{name} TB/s"] = nbytes / us / 1e6
        ret[f"{name} err"] = err
    return ret


def check_residual_rejects_bad_out(ca, rank, device, n):
    """The native op validates `out` before launching, not just `residual`."""
    from aiter.ops import custom_all_reduce as ops

    x = torch.ones((4, n), dtype=torch.bfloat16, device=device)
    bad_outs = {
        "undersized": torch.empty((2, n), dtype=x.dtype, device=device),
        "wrong dtype": torch.empty(x.shape, dtype=torch.float32, device=device),
    }
    for name, out in bad_outs.items():
        try:
            ops.all_reduce_residual(
                ca._ptr,
                x,
                out,
                x,
                True,
                ca._pool["input"].data_ptr,
                ca._pool["input"].max_size,
            )
        except RuntimeError:
            pass
        else:
            raise AssertionError(f"all_reduce_residual accepted a {name} out")
    if rank == 0:
        logger.info("all_reduce_residual out checks passed")


def check_uses_1stage_ar_gfx1250(ca, rank, device, n):
    """gfx1250 runs only one-shot allreduce kernels, at every size."""
    big = torch.empty((4096, n), dtype=torch.bfloat16, device=device)
    saved = ca._is_gfx1250
    try:
        ca._is_gfx1250 = True
        assert ca.uses_1stage_ar(big), "gfx1250 reported a 2-stage allreduce"
    finally:
        ca._is_gfx1250 = saved
    if rank == 0:
        logger.info("uses_1stage_ar gfx1250 check passed")


def check_norm_rejects_short_buffers(ca, rank, device, n):
    """The native op checks buffer extents against m and num_norm_rows."""
    from aiter.ops import custom_all_reduce as ops

    m, norm_rows = 16, 4
    x = torch.ones((m, n), dtype=torch.bfloat16, device=device)
    w = torch.ones(n, dtype=torch.bfloat16, device=device)

    def rows(k):
        return torch.empty((k, n), dtype=x.dtype, device=device)

    cases = {
        "out shorter than num_norm_rows": (x, rows(m), rows(norm_rows - 1), False),
        "res_out shorter than m": (x, rows(m - 1), rows(norm_rows), False),
        "res_inp shorter than m": (rows(m - 1), rows(m), rows(norm_rows), False),
    }
    for name, (res_inp, res_out, out, skip) in cases.items():
        try:
            ops.fused_allreduce_rmsnorm(
                ca._ptr,
                x,
                res_inp,
                res_out,
                out,
                w,
                EPS,
                ca._pool["input"].data_ptr,
                ca._pool["input"].max_size,
                False,
                False,
                norm_rows,
                skip,
            )
        except RuntimeError:
            pass
        else:
            raise AssertionError(f"fused_allreduce_rmsnorm accepted {name}")
    if rank == 0:
        logger.info("fused_ar_rms buffer extent checks passed")


def check_1stage_rejects_options(ca, rank, device, n):
    from aiter.ops import custom_all_reduce as ops

    m = 4
    x = torch.ones((m, n), dtype=torch.bfloat16, device=device)
    w = torch.ones(n, dtype=torch.bfloat16, device=device)
    for kwargs in ({"skip_residual": True}, {"num_norm_rows": 2}):
        try:
            ca.custom_fused_ar_rms(x, x, w, EPS, True, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"use_1stage=True accepted {kwargs}")
    # A padded output width takes the pad kernel, which has no row options,
    # including when fused_ar_rms allocates that output itself.
    for kwargs in ({"skip_residual": True}, {"num_norm_rows": 2}):
        try:
            ca.custom_fused_ar_rms(
                x, x, w, EPS, False, out_hidden_dim=n + 128, **kwargs
            )
        except ValueError:
            pass
        else:
            raise AssertionError(f"padded out_hidden_dim accepted {kwargs}")
    # The native op checks too, for callers that bypass the Python wrapper.
    out, res_out = torch.empty_like(x), torch.empty_like(x)
    try:
        ops.fused_allreduce_rmsnorm(
            ca._ptr,
            x,
            x,
            res_out,
            out,
            w,
            EPS,
            ca._pool["input"].data_ptr,
            ca._pool["input"].max_size,
            True,
            False,
            -1,
            True,
        )
    except RuntimeError:
        pass
    else:
        raise AssertionError("native use_1stage=True accepted skip_residual")
    # num_norm_rows == m is a no-op and stays allowed on the 1-stage path.
    ca.custom_fused_ar_rms(x, x, w, EPS, True, num_norm_rows=m)
    torch.cuda.synchronize()
    if rank == 0:
        logger.info("1-stage option checks passed")


def _worker(rank, tp, init_method, args, result_path):
    from aiter.dist.device_communicators.custom_all_reduce import CustomAllreduce

    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")
    dist.init_process_group("gloo", world_size=tp, rank=rank, init_method=init_method)
    ca = None
    _CTX.update(rank=rank, device=device, graphs=[])
    tables = {}
    try:
        ca = CustomAllreduce(dist.group.WORLD, device)
        assert not ca.disabled, "requires custom allreduce"
        _CTX["ca"] = ca

        for dtype in args.dtype:
            # 1-stage below the size crossover, 2-stage above it. The extra 1-D
            # shape is a 2-stage size whose 16-byte packs do not split evenly
            # across ranks, so it takes the naive kernel (as does every size at
            # TP6).
            shapes = [(m, n) for m, n in itertools.product(args.tokens, args.hidden)]
            shapes.append((8 * (tp * 12800 + 1),))
            tables[f"all_reduce_residual {dtype}"] = [
                test_all_reduce_residual(tp, shape, dtype) for shape in shapes
            ]

            # The fused allreduce + RMSNorm op rejects TP6.
            if tp != 6:
                # 16 rows use the single-kernel 2-stage path, 160 rows use
                # reduce-scatter + local RMSNorm.
                tables[f"fused_ar_rms row options {dtype}"] = [
                    test_fused_ar_rms_row_opts(tp, m, n, m // div, skip, dtype)
                    for m, n, div, skip in itertools.product(
                        args.norm_tokens,
                        args.hidden,
                        args.norm_rows_div,
                        args.skip_residual,
                    )
                ]

        n = args.hidden[0]
        check_residual_rejects_bad_out(ca, rank, device, n)
        check_uses_1stage_ar_gfx1250(ca, rank, device, n)
        if tp != 6:
            check_1stage_rejects_options(ca, rank, device, n)
            check_norm_rejects_short_buffers(ca, rank, device, n)
        _sync_ranks()
        if rank == 0:
            with open(result_path, "wb") as f:
                pickle.dump(tables, f)
    finally:
        _CTX["graphs"].clear()
        if ca is not None:
            ca.close()
        dist.destroy_process_group()


def main():
    if get_gfx() not in SUPPORTED_GFX:
        logger.warning(
            "custom allreduce residual unsupported on %s; skipping", get_gfx()
        )
        return

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="custom AR residual fusion / fused AR+RMSNorm row options",
    )
    parser.add_argument(
        "-t",
        "--tp-size",
        type=int,
        choices=[2, 4, 6, 8],
        default=8,
        help="world size / tensor-parallel size (default: 8)",
    )
    parser.add_argument(
        "-d",
        "--dtype",
        type=dtypes.str2Dtype,
        nargs="*",
        default=[dtypes.bf16],
        help="data type(s); inputs are small integers, so sums stay exact\n"
        "e.g.: -d bf16",
    )
    parser.add_argument(
        "-m",
        "--tokens",
        type=int,
        nargs="*",
        default=[1, 8, 64, 512],
        help="all_reduce_residual rows (default: 1 8 64 512)",
    )
    parser.add_argument(
        "-n",
        "--hidden",
        type=int,
        nargs="*",
        default=[7168],
        help="hidden size(s) (default: 7168)",
    )
    parser.add_argument(
        "--norm_tokens",
        type=int,
        nargs="*",
        default=[16, 160],
        help="fused_ar_rms rows; 16 = 2-stage kernel, 160 = RS + local norm",
    )
    parser.add_argument(
        "--norm_rows_div",
        type=int,
        nargs="*",
        default=[4, 1],
        help="num_norm_rows = rows // div (default: 4 1)",
    )
    parser.add_argument(
        "--skip_residual",
        type=dtypes.str2bool,
        nargs="*",
        default=[True, False],
        help="skip_residual values to sweep (default: True False)",
    )
    args = parser.parse_args()

    tp = args.tp_size
    with tempfile.TemporaryDirectory() as tmp:
        result_path = os.path.join(tmp, "rows.pkl")
        torch.multiprocessing.spawn(
            _worker,
            args=(
                tp,
                get_distributed_init_method(get_ip(), get_open_port()),
                args,
                result_path,
            ),
            nprocs=tp,
            join=True,
        )
        with open(result_path, "rb") as f:
            tables = pickle.load(f)

    for name, rows in tables.items():
        df = pd.DataFrame(rows)
        logger.info(
            "%s TP%d summary (markdown):\n%s", name, tp, df.to_markdown(index=False)
        )


if __name__ == "__main__":
    main()
