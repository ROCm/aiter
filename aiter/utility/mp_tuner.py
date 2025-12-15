# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
import torch
import multiprocessing as mp
import time
import logging
import sys
import os
from multiprocessing import TimeoutError as MPTimeoutError
from aiter.test_common import checkAllclose
from aiter import dtypes

logger = logging.getLogger(__name__)


def worker(
    gpu_id,
    info,
    func,
    args,
    kwargs,
    ref=None,
    rtol=1e-2,
    atol=1e-2,
    printLog=False,
    tol_err_ratio=0.05,
):
    from aiter.test_common import run_perftest

    pid = mp.current_process().pid
    device = torch.device(f"cuda:{gpu_id}")

    max_err_ratio = 0.0
    try:
        torch.cuda.set_device(device)
        args = [el.to(device) if isinstance(el, torch.Tensor) else el for el in args]
        torch.cuda.synchronize()
        res = None
        us = float("inf")
        try:
            # print(f"run_perftest: info:{info}")
            res, us = run_perftest(func, *args, **kwargs)
            us = round(us, 4)
        except RuntimeError as e:
            print(f"run gpu func error: info:{info}\t {e}")
        max_retries = 3
        retry_count = 0

        while us == 0 and retry_count < max_retries:
            print(f"!!!! us = 0, try {retry_count + 1} run")
            res, us = run_perftest(func, *args, **kwargs)
            retry_count += 1
        if us == 0:
            print(f"Warning: try run {max_retries} times, but still get 0!")
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
                        tol_err_ratio=tol_err_ratio,
                        printLog=printLog,
                        msg=f"info:{info} res[{i}] ",
                    )
                    max_err_ratio = max(max_err_ratio, err_ratio)

    except torch.cuda.CudaError as e:
        print(f"CUDA Error in process:{pid} info:{info}: {e}")
        print(f"This might be a GPU hang or memory access fault")
        us = float("inf")
        max_err_ratio = 1.0
    except RuntimeError as e:
        if "CUDA" in str(e) or "HIP" in str(e) or "out of memory" in str(e).lower():
            print(f"GPU Runtime Error in process:{pid} info:{info}: {e}")
            # Try to recover GPU state
            try:
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            except:
                pass
        else:
            print(f"Runtime Error in process:{pid} info:{info}: {e}")
        us = float("inf")
        max_err_ratio = 1.0
    except TimeoutError as e:
        print(f"Timeout in process:{pid} info:{info}: {e}")
        us = float("inf")
        max_err_ratio = 1.0
    except Exception as e:
        print(f"Unexpected Error in process:{pid} info:{info}: {e}")
        import traceback

        traceback.print_exc()
        us = float("inf")
        max_err_ratio = 1.0
    finally:
        # Ensure GPU state is cleaned up
        try:
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        except:
            pass

    return info, us, round(max_err_ratio, 4)


def work_group(gpu_id, fast_mode, err_ratio, in_data, tasks):
    """Work group that processes a batch of related tasks.

    Each work_group runs in a separate process (controlled by maxtasksperchild=1).
    If this process crashes due to GPU memory fault, it won't affect other task groups.
    GPU ID is explicitly passed as a parameter.

    Args:
        gpu_id: GPU device ID to use for this work group
        fast_mode: Whether to skip result comparison
        err_ratio: Error tolerance ratio
        in_data: Input data for tasks
        tasks: Task or list of tasks to execute
    """
    try:
        group_task = [tasks] if not isinstance(tasks, list) else tasks
        kernels_num, (input_data) = in_data
        (
            info,
            gen_data,
            gen_args,
            func,
            args,
            kwargs,
            ref_func,
            ref_args,
            ref_kwargs,
            ref,
            *rest,
        ) = group_task[0]

        pid = mp.current_process().pid
        device = torch.device(f"cuda:{gpu_id}")
        torch.cuda.set_device(device)
        print(f"Process {pid} using GPU {gpu_id} for task group")
    except Exception as e:
        print(f"Error initializing work_group: {e}")
        # Return dummy failed results for all tasks
        if isinstance(tasks, list):
            return [
                (task[0] if task else "unknown", float("inf"), 1.0) for task in tasks
            ]
        else:
            return [(tasks[0] if tasks else "unknown", float("inf"), 1.0)]

    try:
        data = (
            gen_data(*gen_args, device=device)
            if not input_data and gen_data is not None
            else input_data
        )

        assert ref_func is not None or ref is not None or fast_mode != 0
        # ref=None & ref_func=None & fast_mode=1: fast tune, not compare results, do not postprocess,return all results
        # ref=None & fast_mode=0: ref_func should be given and return best result
        # (ref!=None | ref_func!=None) & fast_mode=1: compare results and return all results, but do not postprocess
        # (ref!=None | ref_func!=None) & fast_mode=0: return best result, postprocess
        if ref is None and not fast_mode or (ref_func is not None and fast_mode):
            ref_data_idx, *rest = ([], *ref_args) if not data else ref_args
            updated_ref_args = tuple(data[i] for i in ref_data_idx) + tuple(rest)
            ref = ref_func(*updated_ref_args, **ref_kwargs)
            torch.cuda.synchronize()

        rets = []
        shape_grouped = isinstance(tasks, list)
        solutions = 1 if not shape_grouped else kernels_num
        for i in range(solutions):
            (
                info,
                gen_data,
                gen_args,
                func,
                args,
                kwargs,
                ref_func,
                ref_args,
                ref_kwargs,
                ref_noused,
                *rest,
            ) = group_task[i]
            # either gen_data func or inpur data

            new_args = (
                (tuple(data[i] for i in args[0]) + tuple(args[1:]))
                if gen_data is not None
                else args
            )

            ref = ref if ref_noused is None else ref_noused
            work_args = (
                gpu_id,  # Pass GPU ID as first argument
                info,
                func,
                new_args,
                kwargs,
                ref,
                *rest,
            )

            # Run worker with explicit GPU ID
            ret = worker(*work_args, tol_err_ratio=err_ratio)
            rets.append(ret)
        return rets

    except Exception as e:
        print(f"Critical error in work_group: {e}")
        import traceback

        traceback.print_exc()
        # Return dummy failed results for all tasks in the group
        if isinstance(tasks, list):
            return [
                (task[0] if task else "unknown", float("inf"), 1.0) for task in tasks
            ]
        else:
            return [(tasks[0] if tasks else "unknown", float("inf"), 1.0)]


def mp_tuner(
    tasks,
    in_datas,
    mp_num=0,
    fast_mode=False,
    shape_grouped=False,
    err_ratio=0.05,
    timeout=30,
):
    """Multi-process tuner with GPU fault isolation.

    Each task runs in an isolated process (maxtasksperchild=1) to ensure that
    GPU memory faults or hangs in one task don't affect others. The process pool
    automatically spawns new workers after each task completes or crashes.

    Args:
        tasks: List of tuning tasks
        in_datas: Input data for tasks
        mp_num: Number of parallel processes (0 = use all GPUs)
        fast_mode: Skip result comparison if True
        shape_grouped: Group tasks by shape
        err_ratio: Error tolerance ratio
        timeout: Timeout in seconds for each task group

    Returns:
        List of (info, latency, error_ratio) tuples
    """
    gpu_num = torch.cuda.device_count()
    mp.set_start_method("spawn", force=True)
    mp_num = gpu_num if mp_num < 1 or mp_num > gpu_num else mp_num
    parallel_num = mp_num
    start_idx = 0
    if not tasks:
        return []
    if mp_num == 1 & fast_mode == 0:
        shape_grouped = True

    # Set maxtasksperchild=1 so each task runs in a fresh process
    # This ensures GPU faults in one task don't affect others
    # GPU ID will be explicitly passed to each work_group
    pool = mp.Pool(processes=parallel_num, maxtasksperchild=1)  #

    task_group = []
    # dispatch per shape to one pid
    if shape_grouped:
        start = 0
        for kernel_nums, _ in in_datas:
            end = start + kernel_nums - 1
            task_group.append(tasks[start : end + 1])
            start = end + 1
    else:
        task_group = tasks

    # to get index of input data for task_group
    import numpy as np

    ref_data_index = [i for i in range(len(in_datas))]
    if not shape_grouped:
        cumulative = np.cumsum([size for size, _ in in_datas])
        ref_data_index = np.searchsorted(
            cumulative, np.arange(len(task_group)), side="right"
        )

    # Assign GPU ID for each task group using round-robin distribution
    # This ensures tasks are evenly distributed across all available GPUs
    print(f"Distributing {len(task_group)} task groups across {mp_num} GPUs")

    rets = [
        pool.apply_async(
            work_group,
            args=(
                k % mp_num + start_idx,  # GPU ID (round-robin assignment)
                fast_mode,
                err_ratio,
                in_datas[ref_data_index[k]],
                task_group[k],
            ),
        )
        for k in range(len(task_group))
    ]
    pool.close()

    # Collect results with timeout and error handling
    import itertools

    result = []
    failed_tasks = []

    for k, async_result in enumerate(rets):
        try:
            # Wait for result with timeout
            task_result = async_result.get(timeout=timeout)

            if shape_grouped:
                result.extend(task_result)
            else:
                result.append(task_result[0])

        except MPTimeoutError:
            error_msg = f"[!]  Task {k} timed out after {timeout}s - likely GPU hang or infinite loop"
            print(error_msg)
            logger.error(error_msg)
            failed_tasks.append((k, "timeout"))
            # Add dummy failed result
            if shape_grouped:
                task_info = (
                    task_group[k]
                    if isinstance(task_group[k], list)
                    else [task_group[k]]
                )
                for task in task_info:
                    info = task[0] if len(task) > 0 else f"task_{k}"
                    result.append((info, float("inf"), 1.0))
            else:
                task = task_group[k]
                info = task[0] if len(task) > 0 else f"task_{k}"
                result.append((info, float("inf"), 1.0))

        except Exception as e:
            # Check if it's a process crash (segfault, memory fault, etc.)
            error_type = type(e).__name__
            error_str = str(e)

            if (
                "died" in error_str.lower()
                or "terminated" in error_str.lower()
                or "segmentation" in error_str.lower()
            ):
                error_msg = f"[Crash] Task {k} crashed (likely GPU memory access fault): {error_type} - {e}"
            else:
                error_msg = f"[Failed] Task {k} failed with {error_type}: {e}"

            print(error_msg)
            logger.error(error_msg)
            failed_tasks.append((k, error_type))

            # Add dummy failed result
            if shape_grouped:
                task_info = (
                    task_group[k]
                    if isinstance(task_group[k], list)
                    else [task_group[k]]
                )
                for task in task_info:
                    info = task[0] if len(task) > 0 else f"task_{k}"
                    result.append((info, float("inf"), 1.0))
            else:
                task = task_group[k]
                info = task[0] if len(task) > 0 else f"task_{k}"
                result.append((info, float("inf"), 1.0))

    # Clean up the pool
    try:
        pool.terminate()
        pool.join()
    except Exception as e:
        print(f"Warning: Error during pool cleanup: {e}")

    # Print summary
    if failed_tasks:
        timeout_count = sum(1 for _, reason in failed_tasks if reason == "timeout")
        crash_count = len(failed_tasks) - timeout_count

        summary = (
            f"\n{'='*60}\n"
            f"Tuning Summary:\n"
            f"  Total tasks: {len(rets)}\n"
            f"  Successful: {len(rets) - len(failed_tasks)}\n"
            f"  Failed: {len(failed_tasks)}\n"
            f"    - Timeouts (GPU hang): {timeout_count}\n"
            f"    - Crashes (memory fault): {crash_count}\n"
            f"{'='*60}"
        )
        print(summary)
        logger.warning(f"Failed task indices: {[k for k, _ in failed_tasks]}")
    else:
        print(f"[Done] All {len(rets)} tasks completed successfully")

    return result
