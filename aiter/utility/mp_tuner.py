# SPDX-License-Identifier: MIT
# Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.
import torch
import multiprocessing as mp
import os
import pandas as pd
import time


def worker(gpuIDMap, tag, func, args, kwargs):
    from aiter.test_common import run_perftest
    pid = mp.current_process().pid
    gpuID = gpuIDMap[pid]
    args = [el.to("cpu") if isinstance(el, torch.Tensor) else el for el in args]
    torch.cuda.synchronize()

    device = torch.device(f"cuda:{gpuID}")
    torch.cuda.set_device(device)
    args = [el.to(device) if isinstance(el, torch.Tensor) else el for el in args]
    torch.cuda.synchronize()

    try:
        _, us = run_perftest(func, *args, **kwargs)
        torch.cuda.synchronize()
        _ = _.to("cpu")
    except Exception as e:
        print(f"Error in process {pid}: {e}")
        us = float("inf")
        _ = None

    return tag, us, _


def get_pid():
    time.sleep(3)
    return mp.current_process().pid


def mp_tuner(tasks, mp_num=0):
    gpu_num = torch.cuda.device_count()
    mp.set_start_method("spawn", force=True)
    pool = mp.Pool(processes=gpu_num if mp_num < 1 or mp_num > gpu_num else mp_num)
    pids = [pool.apply_async(get_pid) for i in range(gpu_num)]
    # time.sleep(2)

    gpu_map = {el.get(): i for i, el in enumerate(pids)}
    rets = [pool.apply_async(worker, args=(gpu_map, *task)) for task in tasks]

    pool.close()
    pool.join()
    return [el.get() for el in rets]
