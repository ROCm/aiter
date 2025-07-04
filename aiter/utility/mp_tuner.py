# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
import torch
import multiprocessing as mp
import time
from aiter.test_common import checkAllclose
from aiter import dtypes


def worker(
    gpuIDMap,
    info,
    func,
    args,
    kwargs,
    # ref_func,
    # ref_args,
    # ref_kwargs,
    ref=None,
    rtol=1e-2,
    atol=1e-2,
    printLog=False,
):
    from aiter.test_common import run_perftest

    pid = mp.current_process().pid
    gpuID = gpuIDMap[pid]
    device = torch.device(f"cuda:{gpuID}")
    torch.cuda.set_device(device)

    args = [el.to(device) if isinstance(el, torch.Tensor) else el for el in args]
    torch.cuda.synchronize()
    max_err_ratio = 0.0
    try:
        res = None
        us = float("inf")
        try:
            res, us = run_perftest(func, *args, **kwargs)
            us = round(us, 4)
        except RuntimeError:
            print(f" info:{info}\t No support")

        torch.cuda.synchronize()

        if ref is not None:
            if isinstance(ref, torch.Tensor):
                ref = [ref]
            if isinstance(res, torch.Tensor):
                res = [res]
            ref = [
                (
                    el.to(device)
                    if isinstance(el, torch.Tensor) and el.device != device
                    else el
                )
                for el in ref
            ]
            for i in range(len(ref)):
                if isinstance(ref[i], torch.Tensor):
                    if res[i].shape != ref[i].shape:
                        res[i] = res[i].view(-1)[: ref[i].numel()].view(ref[i].shape)
                    if ref[i].dtype.itemsize == 1:
                        ref[i] = ref[i].to(dtypes.fp32)
                        res[i] = res[i].to(dtypes.fp32)
                    err_ratio = checkAllclose(
                        ref[i],
                        res[i],
                        atol=atol,
                        rtol=rtol,
                        printLog=printLog,
                        msg=f"info:{info} res[{i}] ",
                    )
                    max_err_ratio = max(max_err_ratio, err_ratio)

    except Exception as e:
        print(f"Error in process:{pid} info:{info}: {e}")
        if res is None and ref is not None:
            print("The output is None, can't match with reference")
        us = float("inf")
        max_err_ratio = 1.0

    return info, us, max_err_ratio


def get_pid():
    time.sleep(3)
    return mp.current_process().pid


def post_process(rets, fast_mode=False):
    print("fast_mode is ", fast_mode)
    if fast_mode:
        return rets
    best_time = -1
    from operator import itemgetter

    sorted_rets = tuple(sorted(rets, key=itemgetter(0)))

    cur_info = sorted_rets[0][0]
    bestConfigs = []
    best_config = sorted_rets[0]
    for info, us, max_err_ratio in sorted_rets:
        if max_err_ratio > 0.01:
            continue
        if info[0] == cur_info[0]:
            if best_time < 0 or us < best_time:
                best_config = [info, us, max_err_ratio]
                best_time = us
        else:
            if best_config[0][1] == -1:
                print(f"No kernel can be used for {info}")
                best_config[1] = "nan"
                best_config[-1] = max_err_ratio
            bestConfigs.append(tuple(best_config))
            best_time = us
            cur_info = info
            best_config = [info, us, max_err_ratio]
    if best_config[0][1] == -1:
        print(f"No kernel can be used for {info}")
        best_config[1] = "nan"
        best_config[-1] = 1
    bestConfigs.append(tuple(best_config))
    return bestConfigs


def work_group(gpuIDMap, fast_mode, in_data, tasks):
    group_task = [tasks] if not isinstance(tasks, list) else tasks
    kernels_num, (input_data) = in_data
    info, func, args, kwargs, ref_func, ref_args, ref_kwargs, ref, *rest = group_task[0]

    updated_ref_args = ref_args if not input_data else input_data[:-1] + ref_args
    if ref is None and not fast_mode:
        ref = ref_func(*updated_ref_args, **ref_kwargs)

    rets = []
    shape_grouped = isinstance(tasks, list)
    solutions = 1 if not shape_grouped else kernels_num
    for i in range(solutions):
        info, func, args, kwargs, ref_func, ref_args, ref_kwargs, ref, *rest = (
            group_task[i]
        )
        work_args = (info, func, input_data + args, kwargs, ref, *rest)
        ret = worker(gpuIDMap, *work_args)
        rets.append(ret)

    return post_process(rets, fast_mode)[0] if shape_grouped else rets[0]


def mp_tuner(tasks, in_datas, mp_num=0, fast_mode=False, shape_grouped=False):
    gpu_num = torch.cuda.device_count()
    mp.set_start_method("spawn", force=True)
    mp_num = gpu_num if mp_num < 1 or mp_num > gpu_num else mp_num
    pool = mp.Pool(processes=mp_num)
    pids = [pool.apply_async(get_pid) for i in range(mp_num)]
    # time.sleep(2)
    task_group = []
    # dispatch per shape to one pid
    if not tasks:
        return []
    if shape_grouped:
        start = 0
        for kernel_nums, _ in in_datas:
            end = start + kernel_nums - 1
            task_group.append(tasks[start : end + 1])
            start = end + 1
    else:
        task_group = tasks
    gpu_map = {el.get(): i for i, el in enumerate(pids)}
    # to get index of input data for task_group
    import numpy as np

    ref_data_index = [i for i in range(len(in_datas))]
    if not shape_grouped:
        cumulative = np.cumsum([size for size, _ in in_datas])
        ref_data_index = np.searchsorted(
            cumulative, np.arange(len(task_group)), side="right"
        )
    rets = [
        pool.apply_async(
            work_group,
            args=(gpu_map, fast_mode, in_datas[ref_data_index[k]], task_group[k]),
        )
        for k in range(len(task_group))
    ]

    pool.close()
    pool.join()
    return (
        [el.get() for el in rets]
        if shape_grouped
        else post_process([el.get() for el in rets], fast_mode)
    )
