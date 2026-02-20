# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
import torch
import torch.profiler as tpf
import os
import copy
import numpy as np
import pandas as pd
import multiprocessing as mp
from aiter import logger

pd.set_option("display.max_rows", 200)
## debug ##
# pd.set_option("display.max_rows", None)
# pd.set_option("display.max_columns", None)
# pd.set_option("display.width", None)
# pd.set_option("display.max_colwidth", None)
# pd.set_option("display.expand_frame_repr", False)


def ensure_spawn_method():
    """
    Ensure multiprocessing uses 'spawn' start method.

    This is required for CUDA/distributed tests. Only sets the method if
    it hasn't been set yet, avoiding conflicts with existing initialization.

    Usage:
        Called at the beginning of multi-GPU test functions before spawning
        worker processes.
    """
    try:
        current_method = mp.get_start_method(allow_none=True)
        if current_method is None:
            mp.set_start_method("spawn")
        elif current_method != "spawn":
            logger.warning(
                f"Multiprocessing start method already set to '{current_method}', "
                f"expected 'spawn'. This may cause issues with CUDA."
            )
    except RuntimeError:
        # Already set, which is fine
        pass


def perftest(
    num_iters=101, num_warmup=2, testGraph=False, num_rotate_args=0, needTrace=False
):
    def decorator(func):
        def wrapper(*args, **kwargs):
            num = num_rotate_args
            if num < 1:
                gpu_id = torch.cuda.current_device()
                iter_used_memory, inputSize, _, _ = device_memory_profiling(
                    func, *args, **kwargs
                )

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
            torch.cuda.synchronize()
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
                    torch.cuda.empty_cache()
                avg = np.mean(latencies) * 1000
                logger.info(f"avg: {avg} us/iter from cuda.Event")

            with tpf.profile(
                activities=[tpf.ProfilerActivity.CPU, tpf.ProfilerActivity.CUDA],
                profile_memory=False,
                with_stack=False,
                with_modules=True,
                # record_shapes=True,
                on_trace_ready=(
                    tpf.tensorboard_trace_handler(f"./aiter_logs/gpu_id_{gpu_id}")
                    if needTrace
                    else None
                ),
            ) as prof:
                data = run_iters_rotate(num_iters, func, rotate_args)
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
            avg = get_trace_perf(prof, num_iters)

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


def post_process_data(df, num_iters, warm_iter=1):
    """remove abnormal data"""

    device_df = df[df["device_type"].astype(str).str.contains("DeviceType.CUDA")]
    # print("devicedf is ", device_df)
    if device_df.empty:
        return [], 0
    kernels_num = int(len(device_df) / num_iters)

    act_iters = num_iters
    valid_n = len(device_df)
    dropped_indexs = []
    if len(device_df) % num_iters == 0:
        kernels_num = int(len(device_df) / num_iters)
    else:
        ##get correct kernel num
        name_list = device_df["name"].tolist()
        max_kernel_num = 20
        n = len(name_list)
        for step in range(1, min(max_kernel_num, n // 2 + 1)):
            sub_list = [name_list[i] for i in range(step)]
            m = len(sub_list)

            valid_n = int(n / m) * m
            pattern_match = all(
                name_list[i] == sub_list[i % m] for i in range(int(n / m) * m)
            )
            if pattern_match:
                kernels_num = m
                act_iters = valid_n / m
                break
        dropped_indexs = device_df.iloc[valid_n:].index.tolist()
        if kernels_num == 0:
            print("data missed, the time may be inaccurate!")

    test_df = device_df.iloc[:valid_n].reset_index()
    grouped_kernel_df = test_df.groupby(test_df.index // kernels_num, sort=False).agg(
        {"self_device_time_total": "sum", "index": list}
    )

    # rm warm iters
    sum_df = grouped_kernel_df.iloc[warm_iter:].reset_index(drop=True)
    out_range_idx = []
    if num_iters > 30:
        # IQR to remove abnormal data
        k = 1.5
        Q1 = sum_df["self_device_time_total"].quantile(0.25)
        Q3 = sum_df["self_device_time_total"].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - k * IQR
        upper = Q3 + k * IQR
        out_range_idx = sum_df.index[
            (sum_df["self_device_time_total"] < lower)
            | (sum_df["self_device_time_total"] > upper)
        ].tolist()
    out_range_num = len(out_range_idx)

    indices = {idx for i in out_range_idx for idx in sum_df.iloc[i]["index"]}

    index_sublists = grouped_kernel_df["index"].head(warm_iter).tolist()
    indices_to_add = [idx for sublist in index_sublists for idx in sublist]
    indices.update(indices_to_add)
    indices.update(dropped_indexs)
    if int(os.environ.get("AITER_LOG_MORE", 0)):
        logger.info(f"abnormal data indices: {indices}")
        for i in indices:
            logger.info(f"abnormal data: {df.iloc[i]['self_device_time_total']}")
    return list(indices), out_range_num + warm_iter + num_iters - act_iters


def get_trace_perf(prof, num_iters):
    assert num_iters > 1
    warm_iter = 1
    num_iters -= warm_iter
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
    ###remove abnormal data
    dropped_num = warm_iter
    dropped_indexs, dropped_num = post_process_data(
        df, num_iters + warm_iter, warm_iter
    )
    df = df.drop(dropped_indexs)
    iter_init = 0  # warm_iter dropped
    df["cnt"] = 1
    rets = []

    for name, d in df.groupby("name", sort=False):
        kernel_num_per_iter = iter_init
        if str(d["device_type"].iat[0]).split(".")[-1] != "CUDA":
            kernel_num_per_iter = 1
        r = d.iloc[kernel_num_per_iter:][
            ["cnt", "self_cpu_time_total", "self_device_time_total"]
        ].sum()
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
            r["device_time_avg"] = (
                r["device_time_sum"] / r["cnt"] if r["cnt"] > 0 else 0
            )
        rets.append(r)
    df = pd.DataFrame(rets)
    cols = [
        "name",
        "cnt",
        "host_time_sum",
        "device_time_sum",
        "device_time_avg",
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
    actual_iters = num_iters + warm_iter - dropped_num
    if df.empty:
        logger.info("no valida data after post process!")

    avg_name = "[avg us/iter]"
    for el in timerList:
        if el == "host_time_sum":
            df.at[avg_name, el] = df[el].sum() / num_iters
        else:
            df.at[avg_name, el] = df[el].sum() / actual_iters
    if int(os.environ.get("AITER_LOG_MORE", 0)):
        pd.set_option("display.expand_frame_repr", False)
        pd.set_option("display.max_colwidth", 90)
        pd.set_option("display.float_format", "{:,.1f}".format)
        logger.info(f"{df}")
    return df.at[avg_name, "device_time_sum"]


def checkAllclose(
    a, b, rtol=1e-2, atol=1e-2, tol_err_ratio=0.05, msg="", printNum=8, printLog=True
):
    isClose = torch.isclose(a, b, rtol=rtol, atol=atol)

    if isClose.all():
        if printLog:
            logger.info(f"{msg}[checkAllclose {atol=} {rtol=} \033[32mpassed~\033[0m]")
        return 0
    else:
        try:
            mask = ~isClose
            num = mask.sum()
            printNum = min(printNum, num)
            percent = (num / a.numel()).item()
            if not printLog:
                return percent
            a_msked = a[mask]
            b_msked = b[mask]
            delta = (a_msked - b_msked).abs()
        except RuntimeError:
            mask = ~isClose.to("cpu")
            num = mask.sum()
            printNum = min(printNum, num)
            percent = (num / a.numel()).item()
            if not printLog:
                return percent
            a_msked = a[mask]
            b_msked = b[mask]
            delta = (a_msked - b_msked).abs()
        if percent > tol_err_ratio:
            logger.info(f"""{msg}[checkAllclose {atol=} {rtol=} \033[31mfailed!\033[0m]
    a    : {a.shape}
           {a_msked[:printNum]}
    b    : {b.shape}
           {b_msked[:printNum]}
    delta:
           {delta[:printNum]}""")
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
