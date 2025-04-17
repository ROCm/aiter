# SPDX-License-Identifier: MIT
# Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.
import torch
import multiprocessing as mp
import os
import pandas as pd
import time


def worker(gpuIDMap, tag, func, args, **kwargs):
    from aiter.test_common import run_perftest
    pid = mp.current_process().pid
    gpuID = gpuIDMap[pid]
    args = [el.to("cpu") if isinstance(el, torch.Tensor) else el for el in args]
    torch.cuda.synchronize()

    device = torch.device(f"cuda:{gpuID}")
    torch.cuda.set_device(device)
    args = [el.to(device) if isinstance(el, torch.Tensor) else el for el in args]
    torch.cuda.synchronize()

    _, us = run_perftest(func, *args, **kwargs)
    torch.cuda.synchronize()

    return tag, us, _.to("cpu")


def get_pid():
    time.sleep(3)
    return mp.current_process().pid


def mp_tuner(tasks):
    gpu_num = torch.cuda.device_count()
    mp.set_start_method("spawn", force=True)
    pool = mp.Pool(processes=gpu_num)
    pids = [pool.apply_async(get_pid) for i in range(gpu_num)]
    # time.sleep(2)

    gpu_map = {el.get(): i for i, el in enumerate(pids)}
    rets = [pool.apply_async(worker, args=(gpu_map, *task)) for task in tasks]

    pool.close()
    pool.join()
    return [el.get() for el in rets]
