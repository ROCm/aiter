# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
import torch
import multiprocessing as mp
import time
from multiprocessing import TimeoutError as MPTimeoutError
from aiter.test_common import checkAllclose
from aiter import dtypes
from aiter import logger


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
            res, us = run_perftest(func, *args, **kwargs)
            us = round(us, 4)

        except RuntimeError as e:
            print(f"run gpu func warning: info:{info}\t {e}", flush=True)
            us = -1  # not support or error
            max_err_ratio = 1.0
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
                        ref[i] = ref[i].view(torch.uint8).to(dtypes.fp32)
                        res[i] = res[i].view(torch.uint8).to(dtypes.fp32)
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
    except RuntimeError as e:
        if "CUDA" in str(e) or "HIP" in str(e) or "out of memory" in str(e).lower():
            if printLog:
                print(f"GPU Runtime Error in process:{pid} info:{info}: {e}")
            # Try to recover GPU state
            try:
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            except Exception as e:
                if printLog:
                    print(f"Error in process:{pid} info:{info}: {e}")
                pass
        else:
            print(f"Runtime Error in process:{pid} info:{info}: {e}")
        us = -1  # float("inf")
        max_err_ratio = 1.0
    except TimeoutError as e:
        if printLog:
            print(f"Timeout in process:{pid} info:{info}: {e}")
        us = float("inf")
        max_err_ratio = 1.0
    except Exception as e:
        if printLog:
            print(f"Unexpected Error in process:{pid} info:{info}: {e}")
            import traceback

            traceback.print_exc()
        us = -1  # float("inf")
        max_err_ratio = 1.0

    return info, us, round(max_err_ratio, 4)


def work_group(GPUIDMap, fast_mode, err_ratio, in_data, tasks, verbose=False):
    """Work group that processes a batch of related tasks."""
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
    _prev_ref_key = (id(ref_func), ref_args)

    pid = mp.current_process().pid
    gpuID = _gpu_id_from_map(GPUIDMap, pid)
    device = torch.device(f"cuda:{gpuID}")
    torch.cuda.set_device(device)
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

    try:
        # Same GPU index as above (avoid a second GPUIDMap[pid] after worker swap)
        gpu_id = gpuID

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

            if ref_noused is not None:
                ref = ref_noused
            else:
                _cur_key = (id(ref_func), ref_args)
                if _cur_key != _prev_ref_key:
                    ref_data_idx_i, *rest_i = ref_args
                    updated = tuple(data[j] for j in ref_data_idx_i) + tuple(rest_i)
                    ref = ref_func(*updated, **ref_kwargs)
                    torch.cuda.synchronize()
                    _prev_ref_key = _cur_key

            # Extract rtol, atol from rest if available, otherwise use defaults
            rtol = rest[0] if len(rest) > 0 else 1e-2
            atol = rest[1] if len(rest) > 1 else 1e-2

            work_args = (
                gpu_id,
                info,
                func,
                new_args,
                kwargs,
                ref,
                rtol,
                atol,
                verbose,  # Use the verbose from work_group parameter
                err_ratio,  # Use the err_ratio from work_group parameter
            )

            # Run worker with explicit GPU ID
            ret = worker(*work_args)
            rets.append(ret)
        return rets

    except Exception as e:
        print(f"Critical error in work_group: {e}")
        # import traceback

        # traceback.print_exc()
        # Return dummy failed results for all tasks in the group
        if isinstance(tasks, list):
            return [
                (task[0] if task else "unknown", float("inf"), 1.0) for task in tasks
            ]
        else:
            return [(tasks[0] if tasks else "unknown", float("inf"), 1.0)]


def get_pid():
    time.sleep(3)
    return mp.current_process().pid


def _gpu_id_from_map(GPUIDMap, pid):
    """Map worker PID to GPU index.

    If the Pool replaced a worker (crash/OOM), the new PID is missing from
    GPUIDMap. When only one GPU is in use (e.g. --mp 1), use that GPU index.
    """
    if pid in GPUIDMap:
        return GPUIDMap[pid]
    if len(GPUIDMap) == 1:
        return next(iter(GPUIDMap.values()))
    raise KeyError(pid)


def mp_tuner(
    tasks,
    in_datas,
    mp_num=0,
    fast_mode=False,
    shape_grouped=False,
    err_ratio=0.05,
    timeout=None,
    verbose=False,  # print verbose log
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
        timeout: Timeout in seconds for each task group (None = no timeout)

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
    if mp_num == 1 and fast_mode == 0:
        shape_grouped = True
    # time.sleep(2)
    task_group = []
    # dispatch per shape to one pid
    if shape_grouped:
        # Group tasks by info_keys (info[0])
        from collections import OrderedDict

        info_key_groups = OrderedDict()

        for task in tasks:
            # Extract info_keys from task (task[0] is info, task[0][0] is info_keys)
            info_keys = task[0][0] if task and len(task) > 0 else None

            if info_keys not in info_key_groups:
                info_key_groups[info_keys] = []
            info_key_groups[info_keys].append(task)

        # Convert to list of groups
        task_group = list(info_key_groups.values())
        print(
            f"[Task Grouping] Grouped {len(tasks)} tasks into {len(task_group)} groups by info_keys"
        )

        # Update in_datas to reflect the actual group sizes
        # Each group gets one entry with (group_size, original_data)
        new_in_datas = []
        for group_idx, group in enumerate(task_group):
            group_size = len(group)
            # Use the first task's data configuration, or keep original if within bounds
            if group_idx < len(in_datas):
                original_data = (
                    in_datas[group_idx][1] if len(in_datas[group_idx]) > 1 else None
                )
            else:
                original_data = (
                    in_datas[0][1] if in_datas and len(in_datas[0]) > 1 else None
                )
            new_in_datas.append((group_size, original_data))

        in_datas = new_in_datas
        print(
            f"[in_datas] Updated to {len(in_datas)} entries with group sizes: {[size for size, _ in in_datas]}"
        )
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
    else:
        # For shape_grouped, each group directly maps to its in_data entry
        ref_data_index = list(range(len(task_group)))

    print(f"Distributing {len(task_group)} task groups across {mp_num} GPUs")

    from collections import deque

    total_tasks = len(task_group)

    def submit_task(pool, gpu_map, task_idx):
        """Submit one task group to the pool."""
        return pool.apply_async(
            work_group,
            args=(
                gpu_map,
                fast_mode,
                err_ratio,
                in_datas[ref_data_index[task_idx]],
                task_group[task_idx],
                verbose,
            ),
        )

    def create_pool_and_gpu_map():
        pool = mp.Pool(processes=parallel_num)
        pids = [pool.apply_async(get_pid) for i in range(start_idx, mp_num)]
        gpu_map = {el.get(): i + start_idx for i, el in enumerate(pids)}
        return pool, gpu_map

    def fill_active_tasks(
        pool, gpu_map, pending_task_indices, active_tasks, task_start_times
    ):
        """
        Submit only up to `parallel_num` in-flight tasks.

        This ensures timeout measures actual execution time instead of queue wait
        time when many shapes are tuned in a single run.
        """
        while pending_task_indices and len(active_tasks) < parallel_num:
            k = pending_task_indices.popleft()
            active_tasks[k] = submit_task(pool, gpu_map, k)
            task_start_times[k] = time.time()

    pool, gpu_map = create_pool_and_gpu_map()

    result_dict = {}  # Store results by task index
    failed_tasks = []
    pending_task_indices = deque(range(total_tasks))
    active_tasks = {}
    task_start_times = {}
    check_interval = 10  # Check every 10 seconds for responsive polling

    fill_active_tasks(
        pool, gpu_map, pending_task_indices, active_tasks, task_start_times
    )

    timeout_msg = (
        f"timeout={timeout}s each" if timeout is not None else "no timeout limit"
    )
    print(f"Waiting for {total_tasks} task groups to complete ({timeout_msg})...")

    def add_dummy_result(k, results_list):
        """Helper function to add dummy failed result"""
        if shape_grouped:
            task_info = (
                task_group[k] if isinstance(task_group[k], list) else [task_group[k]]
            )
            for task in task_info:
                info = task[0] if len(task) > 0 else f"task_{k}"
                results_list.append((info, float("inf"), 1.0))
        else:
            task = task_group[k]
            info = task[0] if len(task) > 0 else f"task_{k}"
            results_list.append((info, float("inf"), 1.0))

    logged_error_types = (
        set()
    )  # Track error types that already logged to avoid duplicates

    while active_tasks or pending_task_indices:
        completed_this_round = []
        timeout_count_this_round = 0  # Track timeouts in this round
        pool_restart_needed = False

        for k, async_result in list(active_tasks.items()):
            try:
                # Calculate appropriate timeout based on task's remaining time
                if timeout is not None:
                    elapsed = time.time() - task_start_times[k]
                    remaining_time = timeout - elapsed
                    # Use the smaller of check_interval and remaining_time, but at least 1 second
                    actual_timeout = max(1, min(check_interval, remaining_time))
                else:
                    # No timeout set, use default check_interval
                    actual_timeout = check_interval

                # Non-blocking check with dynamic timeout
                task_result = async_result.get(timeout=actual_timeout)

                # Task completed successfully
                result_dict[k] = task_result
                completed_this_round.append(k)
                elapsed = time.time() - task_start_times[k]
                if verbose:
                    print(
                        f"[Done] Task {k}/{total_tasks-1} completed in {elapsed:.1f}s ({len(result_dict)}/{total_tasks} done)"
                    )

            except MPTimeoutError:
                # Check if this specific task has exceeded its timeout (only if timeout is set)
                if timeout is not None:
                    elapsed = time.time() - task_start_times[k]

                    if elapsed > timeout:
                        timeout_count_this_round += 1

                        error_msg = f"[!] Task {k} timed out after {elapsed:.1f}s (limit: {timeout}s) - likely GPU hang or infinite loop"
                        print(error_msg)
                        failed_tasks.append((k, "timeout"))

                        # Add dummy result
                        dummy_results = []
                        add_dummy_result(k, dummy_results)
                        result_dict[k] = (
                            dummy_results if shape_grouped else [dummy_results[0]]
                        )
                        completed_this_round.append(k)

                        # Trigger pool restart for timeout (similar to crash)
                        pool_restart_needed = True

                        # If mp_num tasks timed out, all GPUs are likely stuck - restart immediately
                        if timeout_count_this_round >= mp_num:
                            print(
                                f"\n[!] {timeout_count_this_round} tasks timed out (all {mp_num} GPUs likely stuck)"
                            )
                            print("[!] Triggering immediate pool restart...\n")
                            break

            except Exception as e:
                # Check if it's a process crash (segfault, memory fault, etc.)
                error_type = type(e).__name__

                # Special handling for KeyError (PID mapping issue)
                is_mapping_error = error_type == "KeyError"

                if is_mapping_error:
                    error_msg = f"[Mapping Error] Task {k} - Process PID not in GPU map (triggering pool restart): {error_type} - {e}"
                    dummy_results = []
                    add_dummy_result(k, dummy_results)
                    result_dict[k] = (
                        dummy_results if shape_grouped else [dummy_results[0]]
                    )
                    failed_tasks.append((k, "mapping error"))
                    completed_this_round.append(k)
                    pool_restart_needed = True
                else:
                    error_msg = f"[Failed] Task {k} failed with {error_type}: {e}"
                    dummy_results = []
                    add_dummy_result(k, dummy_results)
                    result_dict[k] = (
                        dummy_results if shape_grouped else [dummy_results[0]]
                    )
                    failed_tasks.append((k, "exception"))
                    completed_this_round.append(k)

                # Only log error once per error type
                if error_type not in logged_error_types:
                    logger.error(error_msg)
                    logged_error_types.add(error_type)

        #
        # Remove completed tasks from active list
        for k in completed_this_round:
            active_tasks.pop(k, None)
            task_start_times.pop(k, None)

        # If pool restart needed due to crash, restart pool and resubmit remaining tasks
        if pool_restart_needed and (active_tasks or pending_task_indices):
            if verbose:
                print(f"\n{'='*60}")
                print("? Pool restart needed due to crash. Restarting pool...")
                print(
                    f"Remaining tasks: {len(active_tasks) + len(pending_task_indices)}"
                )
                print(f"{'='*60}\n")

            # Requeue unfinished active tasks before pending ones to preserve order.
            remaining_task_indices = list(active_tasks.keys())
            pending_task_indices = deque(
                remaining_task_indices + list(pending_task_indices)
            )
            active_tasks = {}
            task_start_times = {}

            # Terminate old pool
            try:
                pool.terminate()
                pool.join()
            except Exception as e:
                print(f"Warning: Error during pool termination: {e}")
            pool, gpu_map = create_pool_and_gpu_map()
            fill_active_tasks(
                pool, gpu_map, pending_task_indices, active_tasks, task_start_times
            )
            print(
                f"Pool restarted. Continuing with {len(active_tasks) + len(pending_task_indices)} remaining tasks...\n"
            )
            continue

        fill_active_tasks(
            pool, gpu_map, pending_task_indices, active_tasks, task_start_times
        )

        # Small sleep to avoid busy waiting
        if active_tasks:
            time.sleep(1)

    # Reconstruct results in original task order
    result = []
    for k in range(total_tasks):
        task_result = result_dict.get(k, [])
        if shape_grouped:
            result.extend(task_result)
        else:
            result.append(task_result[0])

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
            f"  Total tasks: {total_tasks}\n"
            f"  Successful: {total_tasks - len(failed_tasks)}\n"
            f"  Failed: {len(failed_tasks)}\n"
            f"    - Timeouts (GPU hang): {timeout_count}\n"
            f"    - Crashes (memory fault): {crash_count}\n"
            f"{'='*60}"
        )
        logger.warning(summary)

    return result
