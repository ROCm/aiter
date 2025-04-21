# SPDX-License-Identifier: MIT
# Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

import torch
import torch.profiler as tpf
import os
import copy
import numpy as np
import pandas as pd
from aiter import logger


def perftest(
    num_iters=101, num_warmup=2, testGraph=False, num_rotate_args=0, needTrace=False
):
    def decorator(func):
        def wrapper(*args, **kwargs):
            num = num_rotate_args
            if num < 1:
                gpu_id = torch.cuda.current_device()
                inputSize = (
                    sum([el.nbytes for el in args if isinstance(el, torch.Tensor)]) + 1
                )
                properties = torch.cuda.get_device_properties(gpu_id)
                cache_size = min(
                    properties.L2_cache_size * 64 * 128,
                    properties.total_memory - inputSize,
                )
                num = (cache_size + inputSize - 1) // inputSize
            num = min(num, num_iters)

            rotate_args = [
                (copy.deepcopy(args), copy.deepcopy(kwargs)) for _ in range(num)
            ]

            run_iters(num_warmup, func, *args, **kwargs)
            if int(os.environ.get("AITER_LOG_MORE", 0)):
                latencies = []
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                for _ in range(num_iters):
                    start_event.record()
                    data = func(*args, **kwargs)
                    end_event.record()
                    end_event.synchronize()
                    latencies.append(start_event.elapsed_time(end_event))
                avg = np.mean(latencies) * 1000
                logger.info(f"avg: {avg} us/iter from cuda.Event")

            if testGraph:
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    data = run_iters_rotate(num_iters, func, rotate_args)
                with tpf.profile(
                    activities=[tpf.ProfilerActivity.CPU, tpf.ProfilerActivity.CUDA],
                    profile_memory=True,
                    with_stack=True,
                    with_modules=True,
                ) as prof:
                    run_iters(1, graph.replay)
                avg = get_trace_perf(prof, num_iters)
                logger.info(f"avg: {avg} us/iter with hipgraph")
            with tpf.profile(
                activities=[tpf.ProfilerActivity.CPU, tpf.ProfilerActivity.CUDA],
                profile_memory=True,
                with_stack=True,
                with_modules=True,
                #  record_shapes=True,
                on_trace_ready=(
                    tpf.tensorboard_trace_handler("./aiter_logs/")
                    if needTrace
                    else None
                ),
            ) as prof:
                data = run_iters_rotate(num_iters, func, rotate_args)

            avg = get_trace_perf(prof, num_iters)

            return data, avg

        return wrapper

    return decorator


def benchmark():
    def decorator(func):
        def wrapper(*args, **kwargs):
            callargs = log_args(func, *args, **kwargs)
            ret = func(*args, **kwargs)
            if ret is not None:
                callargs.update(ret)
            return callargs

        return wrapper

    return decorator


def run_iters(num_iters, func, *args, **kwargs):
    data = None
    for _ in range(num_iters):
        data = func(*args, **kwargs)
    return data


def run_iters_rotate(num_iters, func, rotate_args):
    data = None
    num_rotate_args = len(rotate_args)
    for _ in range(num_iters):
        args, kwargs = rotate_args[_ % num_rotate_args]
        data = func(*args, **kwargs)
    return data


def run_perftest(
    func,
    *args,
    num_iters=101,
    num_warmup=2,
    testGraph=False,
    num_rotate_args=0,
    needTrace=False,
    **kwargs,
):

    @perftest(
        num_iters=num_iters,
        num_warmup=num_warmup,
        testGraph=testGraph,
        num_rotate_args=num_rotate_args,
        needTrace=needTrace,
    )
    def worker(*args, **kwargs):
        return func(*args, **kwargs)

    return worker(*args, **kwargs)


def log_args(func, *args, **kwargs):
    import inspect

    callargs = inspect.getcallargs(func, *args, **kwargs)

    prefix = f"calling {func.__name__}("
    blanks = " " * (len(prefix))

    def getTensorInfo(el):
        if isinstance(el, torch.Tensor):
            return f"{el.shape} {el.dtype} {el.device} {hex(el.data_ptr())}"
        elif isinstance(el, tuple):
            viewNum = 5
            if len(el) > viewNum:
                el = list(el[:viewNum]) + ["..."]
            return f'\n{" "*(len(prefix)+31)}'.join(
                ["("] + [f" {getTensorInfo(e)}" for e in el] + [")"]
            )
        return el

    info = [f"{el:<28} = {getTensorInfo(callargs[el])}" for el in callargs]
    info = f",\n{blanks}".join(info)
    logger.info(f"\n{prefix}{info})")
    return callargs


def get_trace_perf(prof, num_iters):
    assert num_iters > 1
    num_iters -= 1
    df = []
    cols = [
        "name",
        "self_cpu_time_total",
        "self_device_time_total",
        "device_type",
        "device_index",
    ]
    for el in prof.events():
        df.append([getattr(el, x, None) for x in cols])
    df = pd.DataFrame(df, columns=cols)
    df["cnt"] = 1
    rets = []
    for name, d in df.groupby("name", sort=False):
        r = d.iloc[1:][["cnt", "self_cpu_time_total", "self_device_time_total"]].sum()
        if not r.empty:
            device_type = str(d["device_type"].iat[0]).split(".")[-1]
            r["name"] = name
            r["device_type"] = device_type
            r["device_index"] = str(d["device_index"].iat[0])
            if device_type == "CUDA":
                r["device_time_total"] = r["self_device_time_total"]
                r["host_time_total"] = 0
            else:
                r["host_time_total"] = r["self_device_time_total"]
                r["device_time_total"] = 0

        rets.append(r)
    df = pd.DataFrame(rets)

    cols = [
        "name",
        "cnt",
        "host_time_total",
        "device_time_total",
        "device_type",
        "device_index",
    ]
    cols = [el for el in cols if el in df.columns]
    df = df[(df.host_time_total > 0) | (df.device_time_total > 0)]

    timerList = [
        "host_time_total",
        "device_time_total",
    ]
    df = df[cols].sort_values(timerList, ignore_index=True)
    avg_name = "[avg us/iter]"
    for el in timerList:
        df.at[avg_name, el] = df[el].sum() / num_iters
    if int(os.environ.get("AITER_LOG_MORE", 0)):
        pd.set_option("display.max_colwidth", 90)
        logger.info(f"{df}")
    return df.at[avg_name, "device_time_total"]


def checkAllclose(a, b, rtol=1e-2, atol=1e-2, msg="", printNum=8):
    isClose = torch.isclose(a, b, rtol=rtol, atol=atol)
    mask = (~isClose).to("cpu")
    if isClose.all():
        logger.info(f"{msg}[checkAllclose {atol=} {rtol=} \033[32mpassed~\033[0m]")
        return 0
    else:
        num = mask.sum()
        printNum = min(printNum, num)
        percent = (num / a.numel()).item()
        a_msked = a[mask]
        b_msked = b[mask]
        delta = (a_msked - b_msked).abs()
        if percent > 0.01:
            logger.info(
                f"""{msg}[checkAllclose {atol=} {rtol=} \033[31mfailed!\033[0m]
    a    : {a.shape}
           {a_msked[:printNum]}
    b    : {b.shape}
           {b_msked[:printNum]}
    delta:
           {delta[:printNum]}"""
            )
        else:
            logger.info(
                f"""{msg}[checkAllclose {atol=} {rtol=} \033[33mwarning!\033[0m] a and b results are not all close"""
            )
        logger.info(
            f"-->max abs delta:{delta.max()}, delta details: {percent:.1%} ({num} of {a.numel()}) elements"
        )
        return percent


def tensor_dump(x: torch.tensor, name: str, dir="./"):
    x_cpu = x.cpu().view(torch.uint8)
    filename = f"{dir}/{name}.bin"
    x_cpu.numpy().tofile(filename)
    logger.info(f"saving {filename} {x.shape}, {x.dtype}")

    with open(f"{dir}/{name}.meta", "w") as f:
        f.writelines([f"{el}\n" for el in [x.shape, x.dtype]])


def tensor_load(filename: str):
    DWs = np.fromfile(filename, dtype=np.uint32)
    metafile = ".".join(filename.split(".")[:-1]) + ".meta"
    shape, dtype = [eval(line.strip()) for line in open(metafile)]
    return torch.tensor(DWs).view(dtype).view(shape)
