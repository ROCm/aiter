# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import torch
import torch.profiler as tpf
import os
import copy
import numpy as np
import csv
import datetime
import json
import threading
import atexit
import pandas as pd
from aiter import logger

_PERFTEST_CONTEXT = threading.local()

pd.set_option("display.max_rows", 200)

SUMMARY_CSV = "./aiter_perf_summary.csv"
OPS_CSV = "./aiter_perf_ops.csv"
CONFIG_PATH = "ctl_log_config.json"

def cleanup():
    try:
        os.remove(CONFIG_PATH)
        print(f"{CONFIG_PATH} has been deleted.")
    except OSError as e:
        print(f"Error deleting {CONFIG_PATH}: {e}")

if int(os.environ.get("AITER_LOG_MORE", 0)) == 3:
    DEFAULT_CONTROL_LOG_LEVEL = 0
    CONTROL_LOG_MORE = DEFAULT_CONTROL_LOG_LEVEL

    with open(CONFIG_PATH, "w") as f:
        json.dump({"log_level": DEFAULT_CONTROL_LOG_LEVEL}, f)

    atexit.register(cleanup)


def write_csv_header_if_needed(file, headers):
    if not os.path.exists(file):
        with open(file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(headers)

def append_row(file, row):
    with open(file, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(row)

def perftest(
    num_iters=101, num_warmup=2, testGraph=False, num_rotate_args=0, needTrace=False
):
    def decorator(func):
        def wrapper(*args, **kwargs):
            if getattr(_PERFTEST_CONTEXT, "active", False):
                return func(*args, **kwargs)

            if os.path.exists(CONFIG_PATH):
                with open(CONFIG_PATH) as f:
                    cfg = json.load(f)
                    CONTROL_LOG_MORE = cfg.get("log_level", DEFAULT_CONTROL_LOG_LEVEL)
                    if CONTROL_LOG_MORE == 0:
                        return func(*args, **kwargs)

            _PERFTEST_CONTEXT.active = True

            try:

                log_level = int(os.environ.get("AITER_LOG_MORE", 0))
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")

                num = num_rotate_args
                if num < 1:
                    gpu_id = torch.cuda.current_device()
                    iter_used_memory, inputSize, _, _ = device_memory_profiling(func, *args, **kwargs)
                    properties = torch.cuda.get_device_properties(gpu_id)
                    free_memory = torch.cuda.mem_get_info(gpu_id)[0]
                    cache_size = min(
                        getattr(properties, "L2_cache_size", 4096 * 1024) * 64 * 128,
                        (free_memory - iter_used_memory + inputSize) * 0.9,
                    )
                    cache_size = max(cache_size, 0)
                    num = int((cache_size + inputSize - 1) // inputSize)
                num = min(num, num_iters)

                rotate_args = [
                    (copy.deepcopy(args), copy.deepcopy(kwargs)) for _ in range(num - 1)
                ] + [(args, kwargs)]

                run_iters(num_warmup, func, *args, **kwargs)

                input_summary = []
                if log_level >= 3:
                    print(f"\n[PerfTest][ARGS] Function: {func.__name__}")
                for i, a in enumerate(args):
                    if isinstance(a, torch.Tensor):
                        shape = list(a.shape)
                        input_summary.extend([str(shape), str(a.dtype), str(a.device)])
                        if log_level >= 3:
                            print(f"  Arg[{i}] shape={shape}, dtype={a.dtype}, device={a.device}, requires_grad={a.requires_grad}")

                for k, v in kwargs.items():
                    if isinstance(v, torch.Tensor):
                        shape = list(v.shape)
                        input_summary.extend([str(shape), str(v.dtype), str(v.device)])
                        if log_level >= 3:
                            print(f"  Kwarg[{k}] shape={shape}, dtype={v.dtype}, device={v.device}, requires_grad={v.requires_grad}")

                # CUDA Event Timing
                avg_event_us = None
                if log_level >= 1:
                    latencies = []
                    start_event = torch.cuda.Event(enable_timing=True)
                    end_event = torch.cuda.Event(enable_timing=True)
                    for _ in range(num_iters):
                        start_event.record()
                        data = func(*args, **kwargs)
                        end_event.record()
                        end_event.synchronize()
                        latencies.append(start_event.elapsed_time(end_event))
                    avg_event_us = np.mean(latencies) * 1000
                    logger.info(f"[Perf] avg: {avg_event_us:.3f} us/iter [via CUDA Event Timing]")
                    if log_level >= 3:
                        print(f"[PerfTest][CUDA EVENT] Latencies (us): {[f'{latency*1000:.2f}' for latency in latencies]}")

                # CUDA Graph (optional)
                avg_graph_us = None
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
                    avg_graph_us = get_trace_perf(prof, num_iters)
                    logger.info(f"[Perf] avg: {avg_graph_us:.3f} us/iter [via CUDA Graph]")

                # Torch Profiler
                with tpf.profile(
                    activities=[tpf.ProfilerActivity.CPU, tpf.ProfilerActivity.CUDA],
                    profile_memory=True,
                    with_stack=True,
                    with_modules=True,
                    on_trace_ready=(
                        tpf.tensorboard_trace_handler("./aiter_logs/")
                        if needTrace else None
                    ),
                ) as prof:
                    data = run_iters_rotate(num_iters, func, rotate_args)

                avg_prof_us = get_trace_perf(prof, num_iters)
                logger.info(f"[Perf] avg: {avg_prof_us:.3f} us/iter [via Torch Profiler]")

                # Output summary
                output_summary = []
                if isinstance(data, tuple):
                    for i, t in enumerate(data):
                        if isinstance(t, torch.Tensor):
                            output_summary.extend([str(list(t.shape)), str(t.dtype), str(t.device)])
                elif isinstance(data, torch.Tensor):
                    output_summary.extend([str(list(data.shape)), str(data.dtype), str(data.device)])
                else:
                    output_summary.append(str(type(data)))

                if log_level >= 3:
                    print(f"\n[PerfTest][RESULT] Function: {func.__name__}")
                    if len(output_summary) % 3 != 0:
                        print("[PerfTest][WARNING] Unexpected output format, skipping detailed print")
                    else :
                        for i in range(0, len(output_summary), 3):
                            print(f"  Out[{i//3}]: shape={output_summary[i]}, dtype={output_summary[i+1]}, device={output_summary[i+2]}")

                # === CSV write (summary) ===
                write_csv_header_if_needed(SUMMARY_CSV, [
                    "timestamp", "function_name", "avg_time_cuda_event_us",
                    "avg_time_graph_us", "avg_time_profiler_us",
                    "input_summary", "output_summary"
                ])
                append_row(SUMMARY_CSV, [
                    timestamp, func.__name__, avg_event_us, avg_graph_us, avg_prof_us,
                    "|".join(input_summary), "|".join(output_summary)
                ])

                # === CSV write (op-level breakdown) ===
                if log_level >= 3:
                    write_csv_header_if_needed(OPS_CSV, [
                        "timestamp", "function_name", "op_name",
                        "cuda_time_avg_us", "cuda_total_time_us", "self_cpu_time_us"
                    ])
                    for evt in prof.key_averages():
                        # print([d for d in dir(evt)])
                        append_row(OPS_CSV, [
                            timestamp, func.__name__, evt.key,
                            getattr(evt, "device_time", 0),
                            getattr(evt, "device_time_total", 0),
                            getattr(evt, "self_cpu_time_total", 0)
                        ])

                    print("\n[PerfTest][Profiler Breakdown] Top Ops (by CUDA time):")
                    top_ops = prof.key_averages().table(
                        sort_by="self_cuda_time_total", row_limit=10
                    )
                    print(top_ops)

                return data, avg_prof_us
            finally:
                _PERFTEST_CONTEXT.active = False

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


def device_memory_profiling(func, *args, **kwargs):
    gpu_id = torch.cuda.current_device()
    inputSize = (
        sum(
            [
                el.nbytes
                for el in args
                if isinstance(el, torch.Tensor) and el.device.index == gpu_id
            ]
        )
        + 1
    )
    torch.cuda.reset_peak_memory_stats(gpu_id)
    cuda_memory_before = (
        torch.cuda.mem_get_info(gpu_id)[1] - torch.cuda.mem_get_info(gpu_id)[0]
    )
    torch_memory_before = torch.cuda.memory_reserved(gpu_id)
    torch_peak_before = torch.cuda.memory_stats(gpu_id).get(
        "allocated_bytes.all.peak", 0
    )
    non_torch_memory_before = cuda_memory_before - torch_memory_before

    data = func(*args, **kwargs)

    torch.cuda.reset_peak_memory_stats(gpu_id)
    cuda_memory_after = (
        torch.cuda.mem_get_info(gpu_id)[1] - torch.cuda.mem_get_info(gpu_id)[0]
    )
    torch_memory_after = torch.cuda.memory_reserved(gpu_id)
    torch_peak_after = torch.cuda.memory_stats(gpu_id).get(
        "allocated_bytes.all.peak", 0
    )
    non_torch_memory_after = cuda_memory_after - torch_memory_after

    torch_peak_increase = torch_peak_after - torch_peak_before
    non_torch_increase = non_torch_memory_after - non_torch_memory_before
    iter_used_memory = torch_peak_increase + non_torch_increase + inputSize

    return iter_used_memory, inputSize, torch_peak_increase, non_torch_increase


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
                r["device_time_sum"] = r["self_device_time_total"]
                r["host_time_sum"] = 0
            else:
                r["host_time_sum"] = r["self_device_time_total"]
                r["device_time_sum"] = 0

        rets.append(r)
    df = pd.DataFrame(rets)

    cols = [
        "name",
        "cnt",
        "host_time_sum",
        "device_time_sum",
        "device_type",
        "device_index",
    ]
    cols = [el for el in cols if el in df.columns]
    df = df[(df.host_time_sum > 0) | (df.device_time_sum > 0)]

    timerList = [
        "host_time_sum",
        "device_time_sum",
    ]
    df = df[cols].sort_values(timerList, ignore_index=True)
    avg_name = "[avg us/iter]"
    for el in timerList:
        df.at[avg_name, el] = df[el].sum() / num_iters
    if int(os.environ.get("AITER_LOG_MORE", 0)):
        pd.set_option("display.expand_frame_repr", False)
        pd.set_option("display.max_colwidth", 90)
        pd.set_option("display.float_format", "{:,.1f}".format)
        logger.info(f"{df}")
    return df.at[avg_name, "device_time_sum"]


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
