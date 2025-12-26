# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
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
            print(f"run gpu func error: info:{info}\t {e}")
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
    except RuntimeError as e:
        if "CUDA" in str(e) or "HIP" in str(e) or "out of memory" in str(e).lower():
            print(f"GPU Runtime Error in process:{pid} info:{info}: {e}")
            # Try to recover GPU state
            try:
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            except Exception as e:
                print(f"Error in process:{pid} info:{info}: {e}")
                pass
        else:
            print(f"Runtime Error in process:{pid} info:{info}: {e}")
        us = -1  # float("inf")
        max_err_ratio = 1.0
    except TimeoutError as e:
        print(f"Timeout in process:{pid} info:{info}: {e}")
        us = float("inf")
        max_err_ratio = 1.0
    except Exception as e:
        print(f"Unexpected Error in process:{pid} info:{info}: {e}")
        # import traceback

        # traceback.print_exc()
        us = -1  # float("inf")
        max_err_ratio = 1.0

    return info, us, round(max_err_ratio, 4)


def work_group(GPUIDMap, fast_mode, err_ratio, in_data, tasks, printLog=False):
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

    pid = mp.current_process().pid
    gpuID = GPUIDMap[pid]
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
        # Retrieve GPU ID from the map
        pid = mp.current_process().pid
        # if pid not in GPUIDMap:
        #    # Fallback: Use round-robin GPU assignment based on PID
        #    gpu_num = torch.cuda.device_count()
        #    gpu_id = pid % gpu_num
        #    warning_msg = (
        #        f"[Warning] Process {pid} not found in GPUIDMap. "
        #        f"Available PIDs: {list(GPUIDMap.keys())}. "
        #        f"Using fallback GPU assignment: GPU {gpu_id}"
        #    )
        #    print(warning_msg)
        #    # Still raise KeyError to trigger pool restart in parent process
        #    raise KeyError(
        #        f"Process {pid} not found in GPUIDMap. Available PIDs: {list(GPUIDMap.keys())}"
        #    )
        gpu_id = GPUIDMap[pid]

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
                gpu_id,
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

    # Helper function to submit tasks to pool
    def submit_tasks(pool, gpu_map, task_indices):
        """Submit tasks to the pool and return async results as a dict"""
        return {
            k: pool.apply_async(
                work_group,
                args=(
                    gpu_map,
                    fast_mode,
                    err_ratio,
                    in_datas[ref_data_index[k]],
                    task_group[k],
                    verbose,
                ),
            )
            for k in task_indices
        }

    # Create initial pool and submit all tasks
    pool = mp.Pool(processes=parallel_num)
    pids = [pool.apply_async(get_pid) for i in range(start_idx, mp_num)]
    gpu_map = {el.get(): i + start_idx for i, el in enumerate(pids)}
    rets_dict = submit_tasks(pool, gpu_map, range(len(task_group)))
    # Convert to list for compatibility with existing code
    rets = [rets_dict[k] for k in range(len(task_group))]
    pool.close()

    result_dict = {}  # Store results by task index
    failed_tasks = []
    remaining_tasks = list(enumerate(rets))

    # Track start time for each task
    task_start_times = {k: time.time() for k, _ in remaining_tasks}
    check_interval = 10  # Check every 10 seconds for responsive polling

    timeout_msg = (
        f"timeout={timeout}s each" if timeout is not None else "no timeout limit"
    )
    print(f"Waiting for {len(remaining_tasks)} tasks to complete ({timeout_msg})...")

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

    # Process tasks as they complete
    pool_restart_needed = False
    logged_error_types = (
        set()
    )  # Track error types that already logged to avoid duplicates

    while remaining_tasks:
        completed_this_round = []
        dummy_failed_tasks = []
        timeout_count_this_round = 0  # Track timeouts in this round

        for k, async_result in remaining_tasks:
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
                completed_this_round.append((k, async_result))
                elapsed = time.time() - task_start_times[k]
                if verbose:
                    print(
                        f"[Done] Task {k}/{len(rets)-1} completed in {elapsed:.1f}s ({len(result_dict)}/{len(rets)} done)"
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
                        completed_this_round.append((k, async_result))

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
                    dummy_failed_tasks.append((k, "mapping error"))
                    # pool_restart_needed = True
                else:
                    error_msg = f"[Failed] Task {k} failed with {error_type}: {e}"
                    # pool_restart_needed = True

                # Only log error once per error type
                if error_type not in logged_error_types:
                    logger.error(error_msg)
                    logged_error_types.add(error_type)

        #
        # Remove completed tasks from remaining list
        for item in completed_this_round:
            remaining_tasks.remove(item)

        # If pool restart needed due to crash, restart pool and resubmit remaining tasks
        if pool_restart_needed and remaining_tasks:
            if verbose:
                print(f"\n{'='*60}")
                print("? Pool restart needed due to crash. Restarting pool...")
                print(f"Remaining tasks: {len(remaining_tasks)}")
                print(f"{'='*60}\n")

            # Terminate old pool
            try:
                pool.terminate()
                pool.join()
            except Exception as e:
                print(f"Warning: Error during pool termination: {e}")
            # Create new pool
            pool = mp.Pool(processes=parallel_num)

            # Recreate gpu_map for new processes (new PIDs)
            pids = [pool.apply_async(get_pid) for i in range(start_idx, mp_num)]
            gpu_map = {el.get(): i + start_idx for i, el in enumerate(pids)}

            # Resubmit remaining tasks
            remaining_task_indices = [k for k, _ in remaining_tasks]
            new_rets_dict = submit_tasks(pool, gpu_map, remaining_task_indices)
            pool.close()

            # Update remaining_tasks with new async results
            remaining_tasks = [(k, new_rets_dict[k]) for k in remaining_task_indices]
            # Reset start times for resubmitted tasks
            for k in remaining_task_indices:
                task_start_times[k] = time.time()

            # Reset pool restart flag
            pool_restart_needed = False
            print(
                f"Pool restarted. Continuing with {len(remaining_tasks)} remaining tasks...\n"
            )

        # Small sleep to avoid busy waiting
        if remaining_tasks:
            time.sleep(1)

    # Reconstruct results in original task order
    result = []
    for k in range(len(rets)):
        task_result = result_dict[k]
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
            f"  Total tasks: {len(rets)}\n"
            f"  Successful: {len(rets) - len(failed_tasks)}\n"
            f"  Failed: {len(failed_tasks)}\n"
            f"    - Timeouts (GPU hang): {timeout_count}\n"
            f"    - Crashes (memory fault): {crash_count}\n"
            f"{'='*60}"
        )
        logger.warning(summary)

    return result
