# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Custom all-reduce residual fusion, the fused AR+RMSNorm
``num_norm_rows`` / ``skip_residual`` options, and the per-token-quant fused
AR+RMSNorm 2-stage kernel against its 1-stage kernel.

Fused AR+RMSNorm inputs are small integers, so every all-reduced and
residual-added value is exact in bf16: those are compared with atol=0, RMSNorm
outputs with a bf16 tolerance. all_reduce_residual inputs are multiples of 1/64
that sum exactly in fp32 but round in bf16, so the reduced value must round
before the residual add, as all-reduce then add does, at both stages. The
per-token-quant case uses random inputs, whose rounding the two kernels must
reproduce bit for bit. Every case is checked eager and on one CUDA graph replay; the
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
from aiter.test_common import benchmark, checkAllclose, run_perftest

# gfx1250 has no residual all-reduce kernel (custom_all_reduce_residual -> None).
SUPPORTED_GFX = ["gfx942", "gfx950"]
EPS = 1e-6
# Set by sweep() on each rank: the communicator ("ca") and the list keeping every
# captured graph alive ("graphs"). Kept out of the @benchmark args, which are
# the table's columns.
_RANK_STATE = {}


def int_inputs(seed, tp, rank, shape, dtype):
    """This rank's input, the fp32 sum over all ranks, and a shared residual."""
    gen = torch.Generator().manual_seed(seed)
    parts = torch.randint(-8, 9, (tp, *shape), generator=gen, dtype=torch.int16)
    residual = torch.randint(-8, 9, shape, generator=gen, dtype=torch.int16)
    reduced = parts.sum(dim=0, dtype=torch.float32).cuda()
    return parts[rank].cuda().to(dtype), reduced, residual.cuda().to(dtype)


def frac_inputs(seed, tp, rank, shape, dtype):
    """Like int_inputs, with multiples of 1/64 below 4 in magnitude: exact in
    bf16 and summed exactly in fp32, but their sum and the residual add round."""
    gen = torch.Generator().manual_seed(seed)
    parts = torch.randint(-255, 256, (tp, *shape), generator=gen) / 64
    residual = torch.randint(-255, 256, shape, generator=gen) / 64
    reduced = parts.sum(dim=0).cuda()
    return parts[rank].cuda().to(dtype), reduced, residual.cuda().to(dtype)


def randn_inputs(seed, tp, rank, shape, dtype):
    """Like int_inputs, with normal values that are not exact in ``dtype``."""
    gen = torch.Generator().manual_seed(seed)
    parts = torch.randn((tp, *shape), generator=gen).to(dtype)
    residual = torch.randn(shape, generator=gen).to(dtype)
    reduced = parts.float().sum(dim=0).cuda()
    return parts[rank].cuda(), reduced, residual.cuda()


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


@benchmark()
def test_all_reduce_residual(tp, m, n, dtype):
    ca, graphs = _RANK_STATE["ca"], _RANK_STATE["graphs"]
    x, reduced, residual = frac_inputs(m * n, tp, dist.get_rank(), (m, n), dtype)
    # All-reduce rounds to dtype, then the add rounds again; both stages must
    # match this, not a single rounding of the fp32 sum plus residual.
    ref = (reduced.to(dtype).float() + residual.float()).to(dtype).float()
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

    ret = {"gfx": get_gfx()}
    for name, fn in candidates.items():
        us, err = run_case(ca, fn, x, [ref], [(0, 0)], graphs)
        ret[f"{name} us"] = us
        ret[f"{name} TFLOPS"] = flops / us / 1e6
        ret[f"{name} TB/s"] = nbytes / us / 1e6
        ret[f"{name} err"] = err
    return ret


@benchmark()
def test_fused_ar_rms_rows(tp, m, n, norm_div, skip_residual, dtype):
    ca, graphs = _RANK_STATE["ca"], _RANK_STATE["graphs"]
    norm_rows = m // norm_div
    x, reduced, residual = int_inputs(
        m * 3 + norm_rows, tp, dist.get_rank(), (m, n), dtype
    )
    w = (torch.arange(n, device="cuda") % 7 + 1).to(dtype) / 4
    res_ref = reduced if skip_residual else reduced + residual.float()
    out_ref = run_torch_rmsnorm(res_ref, w)[:norm_rows]

    # A caller-provided out holding only the normalized rows, as in the model;
    # allocated once so the timed call does not include the allocation.
    out = torch.empty((norm_rows, n), dtype=dtype, device="cuda")

    def fused(inp):
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
        "gfx": get_gfx(),
        "norm_rows": norm_rows,
        "fused us": us,
        "fused TFLOPS": flops / us / 1e6,
        "fused TB/s": nbytes / us / 1e6,
        "fused err": err,
    }


@benchmark()
def test_fused_ar_rms_quant_2stage(tp, m, n, dtype):
    """custom_fused_ar_rms_quant: the 2-stage kernel's fp8 output, scales and
    residual_out are bit-identical to the 1-stage kernel's."""
    ca, graphs = _RANK_STATE["ca"], _RANK_STATE["graphs"]
    # use_1stage=False takes allreduce_fusion_kernel_2stage only when
    # n / pack <= 1024 and the input is <= 512 KiB; otherwise the split kernel.
    pack = 16 // dtype.itemsize
    assert n % pack == 0 and n // pack <= 1024 and m * n * dtype.itemsize <= 512 << 10
    x, reduced, residual = randn_inputs(m * n + 1, tp, dist.get_rank(), (m, n), dtype)
    w = (torch.arange(n, device="cuda") % 7 + 1).to(dtype) / 4

    def fused(use_1stage):
        return lambda inp: ca.custom_fused_ar_rms_quant(
            inp, residual, w, EPS, use_1stage
        )

    ref = [t.float() for t in fused(True)(x)]
    # Sanity-check the 1-stage reference itself: residual_out to bf16 rounding
    # (the all-reduced value is rounded to bf16 before the add), the dequantized
    # output to fp8 rounding.
    res_ref = (reduced.to(dtype).float() + residual.float()).to(dtype).float()
    out_ref = run_torch_rmsnorm(reduced.to(dtype).float() + residual.float(), w)
    ref_err = max(
        checkAllclose(res_ref, ref[1], rtol=1e-2, atol=1e-2, msg="1stage residual_out"),
        checkAllclose(out_ref, ref[0] * ref[2], rtol=0.1, atol=0.05, msg="1stage out"),
    )

    numel = m * n
    flops = numel * tp + numel * 6  # reduce + residual add, rmsnorm + quant
    nbytes = numel * (3 * x.element_size() + 1) + m * 4  # inp, res in/out, fp8, scale
    ret = {"gfx": get_gfx(), "ref err": ref_err}
    for name, use_1stage in {"1stage": True, "2stage": False}.items():
        us, err = run_case(ca, fused(use_1stage), x, ref, [(0, 0)] * 3, graphs)
        ret[f"{name} us"] = us
        ret[f"{name} TFLOPS"] = flops / us / 1e6
        ret[f"{name} TB/s"] = nbytes / us / 1e6
        ret[f"{name} err"] = err
    return ret


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

    def native_ar_res(out, inp=x):
        ops.all_reduce_residual(ca._ptr, inp, out, x, pool.data_ptr, pool.max_size)

    expect_error(RuntimeError, "undersized out", lambda: native_ar_res(rows(m - 1)))
    expect_error(RuntimeError, "fp32 out", lambda: native_ar_res(rows(m, dtypes.fp32)))
    # Same numel as x, but a strided view the kernels would write densely past.
    strided = torch.empty((m, 2 * n), dtype=bf16, device="cuda")[:, ::2]
    expect_error(RuntimeError, "strided out", lambda: native_ar_res(strided))
    # Weak-contiguous but transposed: its storage order is not its row order.
    transposed = torch.ones((n, m), dtype=bf16, device="cuda").t()
    expect_error(
        RuntimeError,
        "transposed inp",
        lambda: native_ar_res(rows(m), inp=transposed),
    )
    assert ca.custom_all_reduce_residual(transposed, x) is None

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
    # skip_residual does not read res_inp, so its shape does not matter.
    native_ar_rms(rows(1)[:, :1], rows(m), rows(norm_rows), skip=True)
    # num_norm_rows == m is the default behavior, so 1-stage accepts it.
    ca.custom_fused_ar_rms(x, x, w, EPS, True, num_norm_rows=m)

    # A padded output width takes fused_allreduce_rmsnorm_pad, which writes
    # every row of out and residual_out. Short buffers are views of full-size
    # ones, so a write past a view would land in memory the test owns.
    pad_n = n + 128

    def native_pad(res_inp, res_out, out):
        ops.fused_allreduce_rmsnorm_pad(
            ca._ptr,
            x,
            res_inp,
            res_out,
            out,
            w,
            EPS,
            pool.data_ptr,
            pool.max_size,
            False,
            False,
        )

    def pad_rows(k, dtype=bf16):
        return torch.empty((m, pad_n), dtype=dtype, device="cuda")[:k]

    expect_error(
        RuntimeError,
        "padded out and residual_out shorter than m",
        lambda: ca.custom_fused_ar_rms(
            x,
            x,
            w,
            EPS,
            False,
            out=pad_rows(norm_rows),
            residual_out=rows(m)[:norm_rows],
        ),
    )
    expect_error(
        RuntimeError,
        "padded out shorter than m",
        lambda: native_pad(x, rows(m), pad_rows(m - 1)),
    )
    expect_error(
        RuntimeError,
        "padded res_out shorter than m",
        lambda: native_pad(x, rows(m)[: m - 1], pad_rows(m)),
    )
    expect_error(
        RuntimeError,
        "padded res_inp shorter than m",
        lambda: native_pad(rows(m)[: m - 1], rows(m), pad_rows(m)),
    )
    expect_error(
        RuntimeError,
        "padded fp32 out",
        lambda: native_pad(x, rows(m), pad_rows(m, dtypes.fp32)),
    )
    expect_error(
        RuntimeError,
        "padded strided res_out",
        lambda: native_pad(x, strided, pad_rows(m)),
    )
    # Full-size caller buffers on the padded path are accepted.
    out, res_out = pad_rows(m), rows(m)
    got = ca.custom_fused_ar_rms(
        x, x, w, EPS, False, out=out, residual_out=res_out, out_hidden_dim=pad_n
    )
    assert got[0] is out and got[1] is res_out, "padded call replaced the buffers"
    torch.cuda.synchronize()


def set_expandable_segments(enabled):
    """Toggle the live allocator setting, as a host does around graph capture;
    keep the environment in step for _expandable_segments_enabled()."""
    value = "True" if enabled else "False"
    torch._C._accelerator_setAllocatorSettings(f"expandable_segments:{value}")
    os.environ["PYTORCH_HIP_ALLOC_CONF"] = f"expandable_segments:{value}"


def check_capture_paths(ca):
    """Capture-time behavior of the new entry points.

    - Warmup (inside ca.capture() but not CUDA-capturing) returns the caller's
      out and residual_out, as the eager and captured calls do, zeroed like the
      buffers it allocates.
    - custom_all_reduce_residual refuses a captured input that cannot be
      IPC-exported with the issue-#4174 error, instead of aborting the process
      when graph buffers are registered at the end of capture.
    """
    from aiter.dist.device_communicators import custom_all_reduce as car

    m, n, norm_rows = 16, 7168, 4
    bf16 = dtypes.bf16
    w = torch.ones(n, dtype=bf16, device="cuda")
    residual = torch.ones((m, n), dtype=bf16, device="cuda")

    if ca.world_size != 6:  # fused allreduce rmsnorm has no TP6 kernel
        x = torch.ones((m, n), dtype=bf16, device="cuda")
        out = torch.full((norm_rows, n), float("nan"), dtype=bf16, device="cuda")
        res_out = torch.full((m, n), float("nan"), dtype=bf16, device="cuda")
        with ca.capture():
            got = ca.custom_fused_ar_rms(
                x,
                x,
                w,
                EPS,
                False,
                out=out,
                residual_out=res_out,
                num_norm_rows=norm_rows,
                skip_residual=True,
            )
        assert got[0] is out and got[1] is res_out, (
            "warmup returned new buffers of shapes "
            f"{tuple(got[0].shape)}, {tuple(got[1].shape)}"
        )
        assert not out.any() and not res_out.any(), "warmup left out unzeroed"

    if not ca.enable_register_for_capturing:
        aiter.logger.warning("registered capture path off; skipping the #4174 check")
        return
    # An input allocated while expandable_segments are on is VMM-backed; the
    # host turns them off for capture, so capture takes the registered path.
    set_expandable_segments(True)
    try:
        stale = torch.empty((m + 1) * n * 64, dtype=bf16, device="cuda")[: m * n]
        stale = stale.view(m, n).fill_(1)
    finally:
        set_expandable_segments(False)
    real = not car._ipc_exportable(stale.data_ptr())
    every_rank = [None] * ca.world_size
    dist.all_gather_object(every_rank, real)
    saved = car._ipc_exportable
    if not all(every_rank):
        # No VMM block on every rank (e.g. torch on this platform does not
        # support expandable_segments); make the input non-exportable on all
        # ranks so every rank raises alike.
        aiter.logger.warning(
            "expandable_segments gave an exportable input; mocking _ipc_exportable"
        )
        car._ipc_exportable = lambda ptr: False
    error = None
    try:
        capture(ca, lambda: ca.custom_all_reduce_residual(stale, residual))
    except RuntimeError as e:
        error = str(e)
    finally:
        car._ipc_exportable = saved
    assert (
        error and "cannot be IPC-exported" in error
    ), f"a non-exportable captured input must raise the #4174 error, got {error!r}"
    stale = None  # release the VMM block before empty_cache()
    torch.cuda.empty_cache()
    # The refused capture leaves the communicator usable.
    x = torch.ones((m, n), dtype=bf16, device="cuda")
    got = ca.custom_all_reduce_residual(x, residual)
    err = checkAllclose(
        torch.full((m, n), ca.world_size + 1.0, device="cuda"),
        got.float(),
        rtol=0,
        atol=0,
        msg="all_reduce_residual after a refused capture",
    )
    assert err == 0, "all_reduce_residual is wrong after a refused capture"
    torch.cuda.synchronize()


def sweep(rank, tp, ar_cases, rms_cases, quant_cases, dtype, init_method, result_path):
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
        _RANK_STATE.update(ca=ca, graphs=graphs)
        check_rejections(ca)
        ar_rows = [test_all_reduce_residual(tp, m, n, dtype) for m, n in ar_cases]
        rms_rows = [test_fused_ar_rms_rows(tp, *case, dtype) for case in rms_cases]
        quant_rows = [
            test_fused_ar_rms_quant_2stage(tp, m, n, dtype) for m, n in quant_cases
        ]
        # Last: it toggles expandable_segments, which later captures must not see.
        check_capture_paths(ca)
        torch.cuda.synchronize()
        per_rank = [None] * tp
        dist.all_gather_object(per_rank, (ar_rows, rms_rows, quant_rows))
        if rank == 0:
            with open(result_path, "wb") as f:
                pickle.dump(per_rank, f)
    finally:
        _RANK_STATE.clear()
        graphs.clear()
        if ca is not None:
            ca.close()
        dist.destroy_process_group()


def run_tp(tp, ar_cases, rms_cases, quant_cases, dtype):
    """Run the sweep on ``tp`` ranks; one row per case, slowest rank's time."""
    init_method = get_distributed_init_method(get_ip(), get_open_port())
    # spawn (not a Pool) so a rank that dies fails the run instead of hanging it
    with tempfile.TemporaryDirectory() as tmp:
        result_path = os.path.join(tmp, "rows.pkl")
        torch.multiprocessing.spawn(
            sweep,
            args=(
                tp,
                ar_cases,
                rms_cases,
                quant_cases,
                dtype,
                init_method,
                result_path,
            ),
            nprocs=tp,
            join=True,
        )
        with open(result_path, "rb") as f:
            per_rank = pickle.load(f)

    def merge(table):
        """Rank 0's @benchmark row (call args first), with every metric taken
        from the slowest rank: max time and error, min throughput."""
        rows = []
        for case_rows in zip(*(r[table] for r in per_rank)):
            row = dict(case_rows[0])
            for rank_row in case_rows[1:]:
                for k, v in rank_row.items():
                    if k.endswith((" us", " err")):
                        row[k] = max(row[k], v)
                    elif k.endswith((" TFLOPS", " TB/s")):
                        row[k] = min(row[k], v)
            rows.append(row)
        return rows

    return merge(0), merge(1), merge(2)


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
        "--quant_mn",
        type=dtypes.str2tuple,
        nargs="*",
        # A 2-stage size (<= 512 KiB) whose row count is not a power of two.
        default=[(17, 4096)],
        help="per-token-quant fused AR+RMSNorm 2-stage vs 1-stage shapes m,n",
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

    ar_df, rms_df, quant_df = [], [], []
    for tp, dtype in itertools.product(tps, args.dtype):
        # fused allreduce rmsnorm has no TP6 kernel
        ar_rows, rms_rows, quant_rows = run_tp(
            tp,
            args.mn,
            rms_cases if tp != 6 else [],
            args.quant_mn if tp != 6 else [],
            dtype,
        )
        ar_df += ar_rows
        rms_df += rms_rows
        quant_df += quant_rows

    ar_df, rms_df = pd.DataFrame(ar_df), pd.DataFrame(rms_df)
    quant_df = pd.DataFrame(quant_df)
    aiter.logger.info(
        "all_reduce_residual summary (markdown):\n%s", ar_df.to_markdown(index=False)
    )
    aiter.logger.info(
        "fused_ar_rms num_norm_rows/skip_residual summary (markdown):\n%s",
        rms_df.to_markdown(index=False),
    )
    aiter.logger.info(
        "fused_ar_rms_quant 2-stage vs 1-stage summary (markdown):\n%s",
        quant_df.to_markdown(index=False),
    )
    errs = pd.concat(
        [df.filter(like=" err") for df in (ar_df, rms_df, quant_df)], axis=1
    )
    assert (errs.fillna(0) == 0).all().all(), "mismatch vs. reference; see tables"


if __name__ == "__main__":
    main()
