# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import os
from typing import Optional

import torch
import torch.distributed as dist
import argparse
import pandas as pd
import statistics
from aiter.utility import dtypes

if os.getenv("AITER_AOT_IMPORT") == "1":
    import aiter
    from aiter.ops import custom_all_reduce
    from aiter.jit.utils.torch_guard import torch_compile_guard
    from aiter.ops.quant import get_hip_quant

    aiter.torch_compile_guard = torch_compile_guard
    aiter.get_hip_quant = get_hip_quant
    for _name in dir(custom_all_reduce):
        if not _name.startswith("_"):
            setattr(aiter, _name, getattr(custom_all_reduce, _name))

from aiter.dist.parallel_state import (
    ensure_model_parallel_initialized,
    init_distributed_environment,
    set_custom_all_reduce,
    get_tp_group,
    graph_capture,
    destroy_model_parallel,
    destroy_distributed_environment,
)
from aiter.dist.utils import get_open_port, get_distributed_init_method, get_ip
from aiter.dist.communication_op import tensor_model_parallel_all_gather
from aiter.test_common import (
    checkAllclose,
    perftest,
    benchmark,
)
from multiprocessing import set_start_method, Pool, freeze_support
import logging

logger = logging.getLogger("aiter")

set_start_method("spawn", force=True)


DTYPE_NAMES = [
    "fp32",
    "fp16",
    "bf16",
    "u64",
    "i64",
    "u32",
    "i32",
    "i16",
    "u8",
    "i8",
]

DTYPE_ALIASES = {
    "float32": "fp32",
    "float16": "fp16",
    "bfloat16": "bf16",
    "uint64_t": "u64",
    "int64_t": "i64",
    "uint32_t": "u32",
    "int32_t": "i32",
    "int16_t": "i16",
    "uint8_t": "u8",
    "int8_t": "i8",
}


def make_input(shape, dtype, rank_id=0):
    if dtype.is_floating_point:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(1000 + rank_id)
        return torch.randn(shape, dtype=dtype, generator=generator)

    numel = 1
    for dim in shape:
        numel *= dim
    base = torch.arange(numel, dtype=torch.int64).reshape(shape)
    base = base + rank_id * (numel + 17)

    if dtype == torch.uint64:
        return base.to(torch.uint64)
    if dtype == torch.int64:
        return base.to(torch.int64)
    if dtype == torch.uint32:
        return (base % (2**31)).to(torch.uint32)
    if dtype == torch.int32:
        return (base % (2**30)).to(torch.int32)
    if dtype == torch.int16:
        return (base % (2**14)).to(torch.int16)
    if dtype == torch.uint8:
        return (base % (2**8)).to(torch.uint8)
    if dtype == torch.int8:
        return (base % (2**7)).to(torch.int8)
    raise ValueError(f"Unsupported dtype for all-gather test: {dtype}")


def check_equal(ref, out, msg):
    if ref.dtype.is_floating_point:
        return checkAllclose(ref, out.to(ref), msg=msg)
    if not torch.equal(ref.cpu(), out.cpu()):
        diff = (ref.cpu() != out.cpu()).nonzero()
        first = diff[:8].flatten().tolist()
        raise AssertionError(f"{msg} integer all-gather mismatch at indices {first}")
    logger.info(f"{msg}[checkEqual \033[32mpassed~\033[0m]")
    return 0.0


def parse_shape_list(value):
    return [dtypes.str2tuple(item) for item in value.split(";") if item.strip()]


def parse_dtype_list(value):
    names = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        names.append(DTYPE_ALIASES.get(item, item))
    return names


def parse_dim_list(value):
    dims = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        dims.append(int(item))
    return dims


def dtype_name(dtype):
    for name, torch_dtype in dtypes.d_dtypes.items():
        if torch_dtype == dtype:
            return name
    return str(dtype)


def unique_dims_for_shape(shape, dims):
    ndim = len(shape)
    seen = set()
    unique_dims = []
    for dim in dims:
        assert -ndim <= dim < ndim, f"Invalid dim ({dim}) for shape {shape}"
        normalized_dim = dim if dim >= 0 else ndim + dim
        if normalized_dim in seen:
            continue
        seen.add(normalized_dim)
        unique_dims.append(dim)
    return unique_dims


def _make_runner(x, dim, use_custom, force_direct_custom):
    custom_dim = dim if dim >= 0 else x.dim() + dim

    def run_once():
        if use_custom and force_direct_custom:
            return get_tp_group()._all_gather_out_place(x, custom_dim)
        return tensor_model_parallel_all_gather(x, use_custom=use_custom, dim=dim)

    return run_once


def _avg_across_ranks(local_us, device, group):
    avg_us = torch.tensor([local_us], dtype=torch.float64, device=device)
    dist.all_reduce(avg_us, op=dist.ReduceOp.AVG, group=group)
    return avg_us.item()


def time_one_iter(run_fn):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    out = run_fn()
    end.record()
    end.synchronize()
    return out, start.elapsed_time(end) * 1000.0


def time_all_gather(x, use_custom, dim, warmup, iters, force_direct_custom=False):
    run_once = _make_runner(x, dim, use_custom, force_direct_custom)
    group = get_tp_group().device_group

    dist.barrier(group=group)
    for _ in range(warmup):
        run_once()
    torch.cuda.synchronize()
    dist.barrier(group=group)

    out = None
    samples = []
    for _ in range(iters):
        out, us = time_one_iter(run_once)
        samples.append(us)
    local_us = statistics.median(samples)
    return out, _avg_across_ranks(local_us, x.device, group)


def time_all_gather_pair(x, dim, warmup, iters, force_direct_custom=False):
    """Interleave baseline (RCCL) and custom timing in a single loop.

    Each iteration times baseline then custom back-to-back so both paths see the
    same clock/thermal/link state, then reports the median of per-iteration
    latencies. This removes ordering bias without rerunning slow rows.
    """
    baseline_run = _make_runner(x, dim, use_custom=False, force_direct_custom=False)
    custom_run = _make_runner(
        x, dim, use_custom=True, force_direct_custom=force_direct_custom
    )
    group = get_tp_group().device_group

    dist.barrier(group=group)
    for _ in range(warmup):
        baseline_run()
        custom_run()
    torch.cuda.synchronize()
    dist.barrier(group=group)

    baseline_out = None
    custom_out = None
    baseline_samples = []
    custom_samples = []
    for _ in range(iters):
        baseline_out, baseline_us = time_one_iter(baseline_run)
        custom_out, custom_us = time_one_iter(custom_run)
        baseline_samples.append(baseline_us)
        custom_samples.append(custom_us)

    baseline_local = statistics.median(baseline_samples)
    custom_local = statistics.median(custom_samples)
    return (
        baseline_out,
        _avg_across_ranks(baseline_local, x.device, group),
        custom_out,
        _avg_across_ranks(custom_local, x.device, group),
    )


def sweep_worker(
    tp_size,
    pp_size,
    rank_id,
    cases,
    warmup,
    iters,
    force_direct_custom=False,
    distributed_init_method: Optional[str] = None,
):
    device = torch.device(f"cuda:{rank_id}")
    torch.cuda.set_device(device)
    set_custom_all_reduce(True)
    init_distributed_environment(
        world_size=tp_size,
        rank=rank_id,
        distributed_init_method=distributed_init_method,
    )
    ensure_model_parallel_initialized(tp_size, pp_size)
    dist.all_reduce(torch.zeros(1, device=device), group=get_tp_group().device_group)
    torch.cuda.synchronize()

    rows = []
    try:
        for shape, dtype_name_value, dim in cases:
            dtype = dtypes.d_dtypes[dtype_name_value]
            x = make_input(shape, dtype, rank_id).to(device)
            normalized_dim = dim if dim >= 0 else x.dim() + dim
            ref_parts = [
                make_input(shape, dtype, rank).to(device) for rank in range(tp_size)
            ]
            ref = torch.cat(ref_parts, dim=normalized_dim)
            input_bytes = x.numel() * x.element_size()
            direct_supported = input_bytes % 16 == 0 and (
                normalized_dim == 0
                or (
                    normalized_dim == x.dim() - 1
                    and x.shape[-1] * x.element_size() % 16 == 0
                )
            )
            runtime_guarded = False
            custom_eligible = direct_supported and not runtime_guarded

            baseline_out = None
            baseline_us = None
            baseline_err = None
            baseline_error = ""
            custom_out = None
            custom_us = None
            custom_err = None
            custom_error = ""

            custom_runnable = not (force_direct_custom and not direct_supported)
            if not custom_runnable:
                custom_error = "direct custom all-gather unsupported for normalized_dim"

            if custom_runnable:
                # Interleave baseline and custom timing so both paths observe the
                # same clock/thermal/link state each iteration.
                try:
                    (
                        baseline_out,
                        baseline_us,
                        custom_out,
                        custom_us,
                    ) = time_all_gather_pair(
                        x,
                        dim=dim,
                        warmup=warmup,
                        iters=iters,
                        force_direct_custom=force_direct_custom,
                    )
                    baseline_err = check_equal(
                        ref,
                        baseline_out.to(ref),
                        msg=f"baseline allgather: {shape=} {dtype=} {dim=}",
                    )
                    custom_err = check_equal(
                        ref,
                        custom_out.to(ref),
                        msg=f"custom allgather: {shape=} {dtype=} {dim=}",
                    )
                except Exception as exc:
                    baseline_error = str(exc).replace("\n", " ")
                    logger.warning(
                        "interleaved allgather unsupported/failed: shape=%s dtype=%s dim=%s error=%s",
                        shape,
                        dtype,
                        dim,
                        baseline_error,
                    )
                    baseline_out = None
                    baseline_us = None
                    custom_out = None
                    custom_us = None
                    try:
                        custom_out, custom_us = time_all_gather(
                            x,
                            use_custom=True,
                            dim=dim,
                            warmup=warmup,
                            iters=iters,
                            force_direct_custom=force_direct_custom,
                        )
                        custom_err = check_equal(
                            ref,
                            custom_out.to(ref),
                            msg=f"custom allgather: {shape=} {dtype=} {dim=}",
                        )
                    except Exception as custom_exc:
                        custom_error = str(custom_exc).replace("\n", " ")
                        logger.warning(
                            "custom allgather unsupported/failed: shape=%s dtype=%s dim=%s error=%s",
                            shape,
                            dtype,
                            dim,
                            custom_error,
                        )
            else:
                # Custom path is unsupported for this dim; still time RCCL alone.
                try:
                    baseline_out, baseline_us = time_all_gather(
                        x,
                        use_custom=False,
                        dim=dim,
                        warmup=warmup,
                        iters=iters,
                    )
                    baseline_err = check_equal(
                        ref,
                        baseline_out.to(ref),
                        msg=f"baseline allgather: {shape=} {dtype=} {dim=}",
                    )
                except Exception as exc:
                    baseline_error = str(exc).replace("\n", " ")
                    logger.warning(
                        "baseline allgather unsupported/failed: shape=%s dtype=%s dim=%s error=%s",
                        shape,
                        dtype,
                        dim,
                        baseline_error,
                    )
            speedup = (
                baseline_us / custom_us
                if direct_supported
                and baseline_us is not None
                and custom_us is not None
                and custom_us > 0
                else None
            )
            api_speedup = (
                baseline_us / custom_us
                if custom_eligible
                and baseline_us is not None
                and custom_us is not None
                and custom_us > 0
                else None
            )
            rows.append(
                {
                    "tp_size": tp_size,
                    "shape": str(tuple(shape)),
                    "dtype": dtype_name_value,
                    "dim": dim,
                    "normalized_dim": normalized_dim,
                    "input_bytes": input_bytes,
                    "direct_supported": direct_supported,
                    "runtime_guarded": runtime_guarded,
                    "custom_eligible": custom_eligible,
                    "baseline_us": baseline_us,
                    "custom_us": custom_us,
                    "speedup": speedup,
                    "api_speedup": api_speedup,
                    "baseline_err": baseline_err,
                    "custom_err": custom_err,
                    "baseline_error": baseline_error,
                    "custom_error": custom_error,
                }
            )
    finally:
        if dist.is_initialized():
            destroy_model_parallel()
            destroy_distributed_environment()
            torch.cuda.empty_cache()
    return rows


def allgather_sweep(
    tp_size,
    pp_size,
    shapes,
    dtype_names,
    dims,
    warmup,
    iters,
    force_direct_custom=False,
    distributed_init_method: Optional[str] = None,
):
    cases = [
        (shape, dtype_name_value, dim)
        for dtype_name_value in dtype_names
        for shape in shapes
        for dim in unique_dims_for_shape(shape, dims)
    ]
    pool = Pool(processes=tp_size)
    rets = [
        pool.apply_async(
            sweep_worker,
            args=(
                tp_size,
                pp_size,
                rank,
                cases,
                warmup,
                iters,
                force_direct_custom,
                distributed_init_method,
            ),
        )
        for rank in range(tp_size)
    ]
    pool.close()
    pool.join()
    return rets[0].get()


def run_allgather(
    tp_size,
    pp_size,
    rankID,
    x,
    withGraph=False,
    use_custom=False,
    dim=0,
    distributed_init_method: Optional[str] = None,
):
    device = torch.device(f"cuda:{rankID}")
    torch.cuda.set_device(device)
    # init
    logger.info(f"RANK: {rankID} {tp_size} init_process_group...")
    set_custom_all_reduce(True)
    init_distributed_environment(
        world_size=tp_size,
        rank=rankID,
        distributed_init_method=distributed_init_method,
    )
    ensure_model_parallel_initialized(tp_size, pp_size)
    x = x.to(device)
    # dist.barrier(device_ids=[i for i in range(tp_size)])

    # warmup and align all gpu
    group = get_tp_group().device_group
    dist.all_reduce(torch.zeros(1).cuda(), group=group)
    torch.cuda.synchronize()

    if withGraph:
        graph = torch.cuda.CUDAGraph()
        with graph_capture() as gc:
            with torch.cuda.graph(graph, stream=gc.stream):
                out = tensor_model_parallel_all_gather(
                    x, use_custom=use_custom, dim=dim
                )
        out.fill_(0)

        @perftest()
        def run_ca():
            graph.replay()

        _, us = run_ca()
        out = (out, us)
    else:

        @perftest()
        def run_ca(x):
            return tensor_model_parallel_all_gather(x, use_custom=use_custom, dim=dim)

        out = run_ca(x)

    # destroy
    if dist.is_initialized():
        destroy_model_parallel()
        destroy_distributed_environment()
        torch.cuda.empty_cache()
    return out


def call_ccl_allgather_naive(
    tp_size,
    pp_size,
    rankID,
    x,
    use_custom=True,
    loop_time=1,
    distributed_init_method: Optional[str] = None,
):
    device = torch.device(f"cuda:{rankID}")
    torch.cuda.set_device(device)
    # init
    logger.info(f"RANK: {rankID} {tp_size} init_process_group...")
    set_custom_all_reduce(True)
    init_distributed_environment(
        world_size=tp_size,
        rank=rankID,
        distributed_init_method=distributed_init_method,
    )
    ensure_model_parallel_initialized(tp_size, pp_size)
    x = x.to(device)

    # warmup and align all gpu
    torch.cuda.synchronize()

    for i in range(loop_time):
        out = tensor_model_parallel_all_gather(x, use_custom=use_custom)

    # destroy
    if dist.is_initialized():
        destroy_model_parallel()
        destroy_distributed_environment()
        torch.cuda.empty_cache()
    return out


def allgather_acctest(
    tp_size,
    pp_size,
    shape,
    dtype,
    use_custom=False,
    distributed_init_method: Optional[str] = None,
):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "49373"
    pool = Pool(processes=tp_size)
    rets = []
    input_list = []
    for i in range(tp_size):
        input = make_input(shape, dtype, i).to("cuda")
        input_list.append(input)
        # print(input)
        rets.append(
            pool.apply_async(
                call_ccl_allgather_naive,
                args=(
                    tp_size,
                    pp_size,
                    i,
                    input,
                    use_custom,
                    1,
                    distributed_init_method,
                ),
            )
            # pool.apply_async(call_aiter_allgather_naive, args=(tp_size, pp_size, i, input, 1))
        )
    pool.close()
    pool.join()
    ref = input_list[0]
    for i in range(tp_size - 1):
        ref = torch.concat((ref, input_list[i + 1]), -1)

    ar_rslt = []
    for i, ret in enumerate(rets):
        rslt = ret.get()
        ar_rslt.append(rslt)
    for i in ar_rslt:
        check_equal(ref, i.to(ref), msg=f"allgather accuracy: {shape=} {dtype=}")


@benchmark()
def allgather_perftest(
    tp_size,
    pp_size,
    shape,
    dtype,
    withGraph=False,
    use_custom=False,
    dim=0,
    distributed_init_method: Optional[str] = None,
):
    print(f"run perf test, use custom allgather {use_custom}")
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "49373"
    pool = Pool(processes=tp_size)
    ref = torch.zeros(shape, dtype=dtype)
    rets = []
    input_list = []
    for i in range(tp_size):
        x = make_input(shape, dtype, i)
        input_list.append(x)
        rets.append(
            pool.apply_async(
                run_allgather,
                args=(
                    tp_size,
                    pp_size,
                    i,
                    x,
                    withGraph,
                    use_custom,
                    dim,
                    distributed_init_method,
                ),
            )
            # pool.apply_async(run_cu, args=(x, weight, eps, i))
        )
    pool.close()
    pool.join()
    ref = input_list[0]
    for i in range(tp_size - 1):
        ref = torch.concat((ref, input_list[i + 1]), dim)

    rets = [el.get() for el in rets]
    all_us = [us for _, us in rets]
    max_err = 0.0
    for out, us in rets:
        msg = f"allgather (use custom {use_custom}): {shape=} {dtype=} {withGraph=} {us:>8.2f}"
        err = check_equal(ref, out.to(ref), msg=msg)
        max_err = max(max_err, err)
    return {
        "min_us": min(all_us),
        "max_us": max(all_us),
        "err": max_err,
    }


l_dtype = DTYPE_NAMES
l_shape = [
    (2,),
    (16,),
    (1345,),
    (4096,),
    (1, 128),
    (8, 1024),
    (128, 7168),
    # exceeds max_size/world_size but satisfies all other custom ag
    # conditions (contiguous, 16-byte aligned) — should fallback to RCCL
    # threshold: 64 MB (2 GPU) / 32 MB (4 GPU) / 16 MB (8 GPU)
    # this shape = 4097*8192*2 bytes ≈ 64.015 MB, exceeds even the 2-GPU threshold
    (4097, 8192),
]

parser = argparse.ArgumentParser(description="config input of test")
parser.add_argument(
    "-d",
    "--dtype",
    type=str,
    nargs="?",
    const=None,
    default=None,
    help=f"Data type or comma-separated dtypes. Choices: {','.join(DTYPE_NAMES)}",
)
parser.add_argument(
    "-s",
    "--shape",
    type=str,
    nargs="?",
    const=None,
    default=None,
    help="Shape or semicolon-separated shapes. e.g. -s '16,;128,8192'",
)
parser.add_argument(
    "-t",
    "--tp-size",
    type=int,
    default=8,
    help="Tensor-parallel world size.",
)
parser.add_argument(
    "--fast-sweep",
    action="store_true",
    help="Initialize each rank once and sweep all shapes/dtypes/dims.",
)
parser.add_argument("--warmup", type=int, default=20, help="Warmup iterations.")
parser.add_argument("--iters", type=int, default=100, help="Timing iterations.")
parser.add_argument(
    "--dim",
    type=str,
    default="0,-1",
    help="Dimension or comma-separated dimensions to gather. Duplicate normalized dims per shape are skipped.",
)
parser.add_argument(
    "--output-csv",
    type=str,
    default=None,
    help="Optional CSV path for sweep results.",
)
parser.add_argument(
    "--force-direct-custom",
    action="store_true",
    help="Benchmark the custom all-gather kernel directly for crossover analysis.",
)


if __name__ == "__main__":
    freeze_support()
    args = parser.parse_args()
    if args.dtype is None:
        l_dtype_names = l_dtype
        l_dtype = [dtypes.d_dtypes[key] for key in l_dtype]
    else:
        l_dtype_names = parse_dtype_list(args.dtype)
        l_dtype = [dtypes.d_dtypes[key] for key in l_dtype_names]
    if args.shape is not None:
        l_shape = parse_shape_list(args.shape)
    l_dim = parse_dim_list(args.dim)
    if args.fast_sweep:
        df = pd.DataFrame(
            allgather_sweep(
                args.tp_size,
                1,
                l_shape,
                l_dtype_names,
                l_dim,
                args.warmup,
                args.iters,
                args.force_direct_custom,
                distributed_init_method=get_distributed_init_method(
                    get_ip(), get_open_port()
                ),
            )
        )
        show_cols = [
            "tp_size",
            "shape",
            "dtype",
            "dim",
            "normalized_dim",
            "input_bytes",
            "direct_supported",
            "runtime_guarded",
            "custom_eligible",
            "baseline_us",
            "custom_us",
            "speedup",
            "api_speedup",
            "baseline_err",
            "custom_err",
            "baseline_error",
            "custom_error",
        ]
        logger.info(
            "allgather fast sweep summary (markdown):\n%s",
            df[show_cols].to_markdown(index=False),
        )
        if args.output_csv:
            df.to_csv(args.output_csv, index=False)
        raise SystemExit(0)

    df = []
    for dtype in l_dtype:
        for shape in l_shape:
            for dim in unique_dims_for_shape(shape, l_dim):
                for use_custom in [False, True]:
                    ret = allgather_perftest(
                        args.tp_size,
                        1,
                        shape,
                        dtype,
                        withGraph=False,
                        use_custom=use_custom,
                        dim=dim,
                        distributed_init_method=get_distributed_init_method(
                            get_ip(), get_open_port()
                        ),
                    )
                    df.append(ret)
    df = pd.DataFrame(df)
    show_cols = [
        "tp_size",
        "shape",
        "dtype",
        "withGraph",
        "use_custom",
        "dim",
        "min_us",
        "max_us",
        "err",
    ]
    show_cols = [c for c in show_cols if c in df.columns]
    logger.info(
        "allgather summary (markdown):\n%s",
        df[show_cols].to_markdown(index=False),
    )
