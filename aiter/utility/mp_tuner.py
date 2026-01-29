# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
import torch
import multiprocessing as mp
import time
import os
import glob
from multiprocessing import TimeoutError as MPTimeoutError
from aiter.test_common import checkAllclose
from aiter import dtypes
from aiter import logger


def get_existing_core_files(directory=None):
    """Get set of existing core dump files in directory (default: cwd)."""
    if directory is None:
        directory = os.getcwd()
    patterns = [
        os.path.join(directory, "core.*"),
        os.path.join(directory, "core"),
        os.path.join(directory, "*.core"),
        os.path.join(directory, "gpucore.*"),  # AMD GPU core dumps
    ]
    existing = set()
    for pattern in patterns:
        existing.update(glob.glob(pattern))
    return existing


def detect_new_core_dumps(baseline_cores, directory=None):
    """Check if new core dump files appeared since baseline was captured."""
    current_cores = get_existing_core_files(directory)
    new_cores = current_cores - baseline_cores
    return new_cores


def cleanup_core_dumps(core_files):
    """Delete core dump files to free disk space."""
    deleted = 0
    for f in core_files:
        try:
            os.remove(f)
            deleted += 1
        except Exception as e:
            print(f"Warning: Could not delete {f}: {e}")
    if deleted > 0:
        print(f"[!] Cleaned up {deleted} core dump files")


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
    try:
        gpu_map = {el.get(timeout=30): i + start_idx for i, el in enumerate(pids)}
    except MPTimeoutError:
        print("[!] Timeout getting worker PIDs - GPUs may be in bad state")
        pool.terminate()
        pool.join()
        raise RuntimeError("Failed to initialize worker pool - GPU reset may be needed")
    rets_dict = submit_tasks(pool, gpu_map, range(len(task_group)))
    # Convert to list for compatibility with existing code
    rets = [rets_dict[k] for k in range(len(task_group))]
    pool.close()

    result_dict = {}  # Store results by task index
    failed_tasks = []
    remaining_tasks = list(enumerate(rets))

    # Track start time for each task
    task_start_times = {k: time.time() for k, _ in remaining_tasks}
    check_interval = 2  # Check every 2 seconds for fast crash detection (was 10)

    # Track retry count per task to avoid infinite loops on consistently crashing kernels
    task_retry_count = {k: 0 for k, _ in remaining_tasks}
    max_retries = 2  # Mark as failed after this many crashes

    # Capture baseline core dump files to detect new ones
    baseline_cores = get_existing_core_files()

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

        # Check for new core dumps - if detected, trigger immediate pool restart
        new_cores = detect_new_core_dumps(baseline_cores)
        if new_cores:
            num_core_dumps = len(new_cores)
            print(f"\n[!] GPU core dump detected ({num_core_dumps} files): {new_cores}")
            print("[!] Checking task status before pool restart...")
            baseline_cores.update(new_cores)  # Update baseline so we don't detect same files again
            cleanup_core_dumps(new_cores)  # Clean up to save disk space
            pool_restart_needed = True
            
            # Try to salvage results from tasks that completed successfully before the crash
            successful_count = 0
            crashed_count = 0
            pending_tasks_to_rerun = []
            exceeded_retry_count = 0
            
            for k, async_result in remaining_tasks:
                if k not in result_dict:
                    if async_result.ready():
                        # Task finished - try to get result (might be success or crash)
                        try:
                            task_result = async_result.get(timeout=0.1)
                            # Success! Save the result
                            result_dict[k] = task_result
                            completed_this_round.append((k, async_result))
                            successful_count += 1
                        except Exception as e:
                            # Task crashed - mark as failed
                            dummy_results = []
                            add_dummy_result(k, dummy_results)
                            result_dict[k] = dummy_results if shape_grouped else [dummy_results[0]]
                            failed_tasks.append((k, "gpu_crash"))
                            completed_this_round.append((k, async_result))
                            crashed_count += 1
                    else:
                        # Task was running but not finished - check retry count
                        task_retry_count[k] = task_retry_count.get(k, 0) + 1
                        if task_retry_count[k] >= max_retries:
                            # Too many retries - mark as permanently failed
                            dummy_results = []
                            add_dummy_result(k, dummy_results)
                            result_dict[k] = dummy_results if shape_grouped else [dummy_results[0]]
                            failed_tasks.append((k, "gpu_crash_max_retries"))
                            completed_this_round.append((k, async_result))
                            exceeded_retry_count += 1
                        else:
                            # Will retry
                            pending_tasks_to_rerun.append(k)
            
            # Remove completed/crashed tasks from remaining
            for item in completed_this_round:
                if item in remaining_tasks:
                    remaining_tasks.remove(item)
            
            print(f"[!] Results: {successful_count} completed OK, {crashed_count} crashed, {exceeded_retry_count} exceeded max retries")
            print(f"[!] Tasks to rerun after pool restart: {len(pending_tasks_to_rerun)}")
            
            # OPTIMIZATION: If # core dumps >= # pending tasks, all pending tasks crashed
            # No need for isolation mode - just mark them all as failed
            if num_core_dumps >= len(pending_tasks_to_rerun) and len(pending_tasks_to_rerun) > 0:
                print(f"[!] {num_core_dumps} core dumps >= {len(pending_tasks_to_rerun)} pending tasks")
                print(f"[!] All pending tasks crashed - skipping isolation mode, marking all as FAILED")
                for task_idx in pending_tasks_to_rerun:
                    dummy_results = []
                    add_dummy_result(task_idx, dummy_results)
                    result_dict[task_idx] = dummy_results if shape_grouped else [dummy_results[0]]
                    failed_tasks.append((task_idx, "gpu_crash_all_pending"))
                # Clear remaining tasks since we've handled them all
                remaining_tasks = [(k, ar) for k, ar in remaining_tasks if k not in pending_tasks_to_rerun]
                pool_restart_needed = False  # No need to restart, we're done with these
            # Skip the normal polling loop and go to pool restart
        else:
            # Normal polling loop - only runs if no core dump detected
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
                        failed_tasks.append((k, "timeout"))
                        completed_this_round.append((k, async_result))

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
            remaining_task_indices = [k for k, _ in remaining_tasks]
            
            # Terminate old pool first
            try:
                pool.terminate()
                pool.join()
            except Exception as e:
                print(f"Warning: Error during pool termination: {e}")
            
            # CRASH ISOLATION: If we crashed with multiple GPUs, fall back to single-GPU
            # serial execution to identify exactly which task is crashing
            if parallel_num > 1 and len(remaining_task_indices) > 0:
                print(f"\n{'='*60}")
                print(f"[!] GPU crash detected with {parallel_num} parallel tasks")
                print("[!] Falling back to SINGLE-GPU serial execution to isolate faulty kernel...")
                print(f"[!] Will run {len(remaining_task_indices)} tasks one at a time")
                print(f"{'='*60}\n")
                
                # Use shorter timeout for isolation mode since we expect crashes to be fast
                isolation_timeout = min(timeout, 60) if timeout else 60
                
                # Create single-GPU pool
                pool = mp.Pool(processes=1)
                pids = [pool.apply_async(get_pid)]
                try:
                    gpu_map = {pids[0].get(timeout=30): start_idx}
                except MPTimeoutError:
                    print("[!] Timeout getting worker PID - GPU may need reset")
                    pool.terminate()
                    pool.join()
                    raise RuntimeError("Failed to create isolation pool - GPU reset may be needed")
                
                # Run tasks ONE BY ONE to isolate crashes
                for task_idx in remaining_task_indices:
                    # Check for existing core dumps before this task
                    pre_task_cores = get_existing_core_files()
                    
                    # Submit single task
                    async_result = pool.apply_async(
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
                    
                    task_start = time.time()
                    task_crashed = False
                    
                    # Wait for this single task with timeout
                    # Poll every 1 second for fast crash detection
                    while True:
                        try:
                            task_result = async_result.get(timeout=1)  # Check every 1s for fast crash detection
                            # Success!
                            result_dict[task_idx] = task_result
                            elapsed = time.time() - task_start
                            print(f"  [OK] Task {task_idx} completed in {elapsed:.1f}s")
                            break
                        except MPTimeoutError:
                            elapsed = time.time() - task_start
                            # Check for new core dumps
                            new_cores = detect_new_core_dumps(pre_task_cores)
                            if new_cores:
                                print(f"  [CRASH] Task {task_idx} caused GPU crash (core dump detected after {elapsed:.1f}s)")
                                cleanup_core_dumps(new_cores)
                                task_crashed = True
                                break
                            # Check timeout
                            if elapsed > isolation_timeout:
                                print(f"  [TIMEOUT] Task {task_idx} timed out after {elapsed:.1f}s")
                                task_crashed = True
                                break
                        except Exception as e:
                            print(f"  [ERROR] Task {task_idx} failed: {e}")
                            task_crashed = True
                            break
                    
                    if task_crashed:
                        # Mark as failed and record dummy result
                        dummy_results = []
                        add_dummy_result(task_idx, dummy_results)
                        result_dict[task_idx] = dummy_results if shape_grouped else [dummy_results[0]]
                        failed_tasks.append((task_idx, "gpu_crash_isolated"))
                        
                        # Restart the single-GPU pool for next task
                        try:
                            pool.terminate()
                            pool.join()
                        except:
                            pass
                        pool = mp.Pool(processes=1)
                        pids = [pool.apply_async(get_pid)]
                        try:
                            gpu_map = {pids[0].get(timeout=30): start_idx}
                        except MPTimeoutError:
                            print("[!] Failed to restart isolation pool - stopping isolation")
                            break
                
                # Done with isolation mode - all remaining tasks processed
                remaining_tasks = []
                pool_restart_needed = False
                
                # Clean up isolation pool
                try:
                    pool.terminate()
                    pool.join()
                except:
                    pass
                
                # Restore parallel pool for any future use (though we should be done)
                pool = mp.Pool(processes=parallel_num)
                pids = [pool.apply_async(get_pid) for i in range(start_idx, mp_num)]
                try:
                    gpu_map = {el.get(timeout=30): i + start_idx for i, el in enumerate(pids)}
                except:
                    pass
                pool.close()
                
                print(f"\n[!] Isolation complete. Processed all {len(remaining_task_indices)} tasks.\n")
                
            else:
                # Already in single-GPU mode or no remaining tasks
                if verbose:
                    print(f"\n{'='*60}")
                    print("? Pool restart needed due to crash. Restarting pool...")
                    print(f"Remaining tasks: {len(remaining_tasks)}")
                    print(f"{'='*60}\n")

                # Create new pool
                pool = mp.Pool(processes=parallel_num)

                # Recreate gpu_map for new processes (new PIDs)
                pids = [pool.apply_async(get_pid) for i in range(start_idx, mp_num)]
                try:
                    gpu_map = {el.get(timeout=30): i + start_idx for i, el in enumerate(pids)}
                except MPTimeoutError:
                    print("[!] Timeout getting worker PIDs after restart - GPUs may need reset")
                    pool.terminate()
                    pool.join()
                    raise RuntimeError("Failed to restart worker pool - GPU reset may be needed")

                # Resubmit remaining tasks
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
        gpu_crash_count = sum(1 for _, reason in failed_tasks if reason == "gpu_crash")
        max_retry_count = sum(1 for _, reason in failed_tasks if reason == "gpu_crash_max_retries")
        isolated_crash_count = sum(1 for _, reason in failed_tasks if reason == "gpu_crash_isolated")
        all_pending_crash_count = sum(1 for _, reason in failed_tasks if reason == "gpu_crash_all_pending")
        other_crash_count = len(failed_tasks) - timeout_count - gpu_crash_count - max_retry_count - isolated_crash_count - all_pending_crash_count
        summary = (
            f"\n{'='*60}\n"
            f"Tuning Summary:\n"
            f"  Total tasks: {len(rets)}\n"
            f"  Successful: {len(rets) - len(failed_tasks)}\n"
            f"  Failed: {len(failed_tasks)}\n"
            f"    - GPU crashes (all pending crashed): {all_pending_crash_count}\n"
            f"    - GPU crashes (isolated via single-GPU): {isolated_crash_count}\n"
            f"    - GPU crashes (core dump): {gpu_crash_count}\n"
            f"    - GPU crashes (max retries exceeded): {max_retry_count}\n"
            f"    - Timeouts (GPU hang): {timeout_count}\n"
            f"    - Other crashes: {other_crash_count}\n"
            f"{'='*60}"
        )
        logger.warning(summary)

    return result
