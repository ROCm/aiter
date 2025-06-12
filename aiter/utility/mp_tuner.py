# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
import torch
import multiprocessing as mp
import time
from aiter.test_common import checkAllclose
from aiter import dtypes


def worker(
    gpuIDMap, tag, func, args, kwargs, ref=None, rtol=1e-2, atol=1e-2, printLog=False
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
        res, us = run_perftest(func, *args, **kwargs)
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
                        msg=f"tag:{tag} res[{i}] ",
                    )
                    max_err_ratio = max(max_err_ratio, err_ratio)

    except Exception as e:
        print(f"Error in process:{pid} tag:{tag}: {e}")
        if res is None and ref is not None:
            print("The output is None, can't match with reference")
        us = float("inf")
        max_err_ratio = 1.0

    return tag, us, max_err_ratio


def get_pid():
    time.sleep(3)
    return mp.current_process().pid


def mp_tuner(tasks, mp_num=0):
    gpu_num = torch.cuda.device_count()
    mp.set_start_method("spawn", force=True)
    mp_num = gpu_num if mp_num < 1 or mp_num > gpu_num else mp_num
    pool = mp.Pool(processes=mp_num)
    pids = [pool.apply_async(get_pid) for i in range(mp_num)]
    # time.sleep(2)

    gpu_map = {el.get(): i for i, el in enumerate(pids)}
    rets = [pool.apply_async(worker, args=(gpu_map, *task)) for task in tasks]

    pool.close()
    pool.join()
    return [el.get() for el in rets]
