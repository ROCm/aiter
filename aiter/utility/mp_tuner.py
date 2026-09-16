# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
import gc
import math
import multiprocessing as mp
import time
from multiprocessing import TimeoutError as MPTimeoutError
from queue import Empty

import torch

from aiter import dtypes, logger
from aiter.test_common import checkAllclose

_TASK_START_TIMES = None


def _is_mapping_error(exc: BaseException) -> bool:
    return isinstance(exc, KeyError)


def _is_accelerator_error(exc: BaseException) -> bool:
    if type(exc).__name__ == "AcceleratorError":
        return True
    message = str(exc).lower()
    return any(
        marker in message
        for marker in (
            "illegal memory access",
            "memory access fault",
            "device-side assert",
            "hip error 700",
        )
    )


def _init_task_start_times(task_start_times):
    global _TASK_START_TIMES
    _TASK_START_TIMES = task_start_times


def _run_with_start_tracking(task_index, func, args):
    if _TASK_START_TIMES is None:
        raise RuntimeError("Task start-time storage is not initialized")
    _TASK_START_TIMES[task_index] = time.monotonic()
    return func(*args)


def _elapsed_since_task_start(task_start_times, task_index, now=None):
    started_at = task_start_times[task_index]
    if started_at == 0:
        return None
    current_time = time.monotonic() if now is None else now
    return current_time - started_at


def _reset_task_start_times(task_start_times, task_indices):
    """Mark tasks as queued again, so a resubmit is not judged against the
    timestamp its previous attempt left behind."""
    for k in task_indices:
        task_start_times[k] = 0


def _merge_error_ratio(current, observed):
    if not math.isfinite(observed):
        return 1.0
    return max(current, observed)


def _candidate_failure_status(exc: BaseException, default: str = "crash") -> str:
    message = str(exc).lower()
    if isinstance(exc, torch.cuda.OutOfMemoryError) or "out of memory" in message:
        return "oom_runtime"
    if isinstance(exc, (ValueError, NotImplementedError)) or any(
        marker in message
        for marker in ("not support", "unsupported", "invalid argument", "rejected")
    ):
        return "unsupported"
    if isinstance(exc, TimeoutError):
        return "timeout"
    return default


def _format_worker_result(info, us, max_err_ratio, status, return_status):
    result = (info, us, round(max_err_ratio, 4))
    return (*result, status) if return_status else result


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
    compare_fn=None,
    max_abs_delta=None,
    output_keys=None,
    _arg_key_list=None,
    catastrophic_check=True,
    return_status=False,
):
    from aiter.test_common import run_perftest

    pid = mp.current_process().pid
    device = torch.device(f"cuda:{gpu_id}")
    max_err_ratio = 0.0
    status = "ok"
    try:
        torch.cuda.set_device(device)
        args = [el.to(device) if isinstance(el, torch.Tensor) else el for el in args]
        if output_keys is not None and _arg_key_list is not None:
            for key in output_keys:
                if key in _arg_key_list:
                    idx = _arg_key_list.index(key)
                    if idx < len(args) and isinstance(args[idx], torch.Tensor):
                        # Fill output with NaN before run_perftest so that
                        # warmup runs with this initial state.  If the kernel
                        # does not fully write the output, NaN values survive
                        # through warmup/iters and will be caught by
                        # checkAllclose.
                        args[idx].fill_(float("nan"))
        torch.cuda.synchronize()
        res = None
        us = float("inf")
        try:
            res, us = run_perftest(func, *args, **kwargs)
            us = round(us, 4)

        except (RuntimeError, ValueError) as e:
            if _is_accelerator_error(e):
                raise
            print(f"run gpu func warning: info:{info}\t {e}", flush=True)
            us = -1  # not support or error
            max_err_ratio = 1.0
            status = _candidate_failure_status(e)
        max_retries = 3
        retry_count = 0

        while us == 0 and retry_count < max_retries:
            print(f"!!!! us = 0, try {retry_count + 1} run")
            res, us = run_perftest(func, *args, **kwargs)
            retry_count += 1
        if us == 0:
            print(f"Warning: try run {max_retries} times, but still get 0!")
            us = -1
            max_err_ratio = 1.0
            status = "crash"
        torch.cuda.synchronize()
        if us == -1 or res is None:
            return _format_worker_result(
                info, us, max_err_ratio, status, return_status
            )
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
                    # Skip generic reshape when a custom compare_fn is given: it
                    # handles shape/dtype itself (e.g. v2 stage1 compares fp4-packed
                    # uint8 res against unpacked bf16 ref -- different numel by design).
                    if compare_fn is None and res[i].shape != ref[i].shape:
                        res[i] = res[i].view(-1)[: ref[i].numel()].view(ref[i].shape)
                    if compare_fn is not None:
                        err_ratio = compare_fn(
                            ref[i],
                            res[i],
                            msg=f"info:{info} res[{i}] ",
                            printLog=printLog,
                        )
                    else:
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
                            max_abs_delta=max_abs_delta,
                            catastrophic_check=catastrophic_check,
                        )
                    max_err_ratio = _merge_error_ratio(max_err_ratio, err_ratio)
            if max_err_ratio > tol_err_ratio:
                status = "mismatch"
    except RuntimeError as e:
        if _is_accelerator_error(e):
            raise
        if "CUDA" in str(e) or "HIP" in str(e) or "out of memory" in str(e).lower():
            if printLog:
                print(f"GPU Runtime Error in process:{pid} info:{info}: {e}")
            # Try to recover GPU state
            try:
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            except Exception as e:  # noqa: BLE001  blanket catch is intentional here
                if printLog:
                    print(f"Error in process:{pid} info:{info}: {e}")
        else:
            print(f"Runtime Error in process:{pid} info:{info}: {e}")
        us = -1  # float("inf")
        max_err_ratio = 1.0
        status = _candidate_failure_status(e)
    except TimeoutError as e:
        if printLog:
            print(f"Timeout in process:{pid} info:{info}: {e}")
        us = float("inf")
        max_err_ratio = 1.0
        status = "timeout"
    except Exception as e:  # noqa: BLE001
        if printLog:
            print(f"Unexpected Error in process:{pid} info:{info}: {e}")
            import traceback

            traceback.print_exc()
        us = -1  # float("inf")
        max_err_ratio = 1.0
        status = _candidate_failure_status(e)

    return _format_worker_result(info, us, max_err_ratio, status, return_status)


def work_group(
    GPUIDMap,
    fast_mode,
    err_ratio,
    in_data,
    tasks,
    verbose=False,
    return_status=False,
    progress_queue=None,
):
    """Work group that processes a batch of related tasks."""
    shape_grouped = isinstance(tasks, list)
    group_task = tasks if shape_grouped else [tasks]
    kernels_num, (input_data) = in_data
    expected_tasks = kernels_num if shape_grouped else 1
    if len(group_task) != expected_tasks:
        raise ValueError(
            f"work group declares {kernels_num} kernels but contains "
            f"{len(group_task)} tasks"
        )
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
    assert ref_func is not None or ref is not None or fast_mode != 0
    # ref=None & ref_func=None & fast_mode=1: fast tune, not compare results, do not postprocess,return all results
    # ref=None & fast_mode=0: ref_func should be given and return best result
    # (ref!=None | ref_func!=None) & fast_mode=1: compare results and return all results, but do not postprocess
    # (ref!=None | ref_func!=None) & fast_mode=0: return best result, postprocess
    data = None
    data_key = None
    cached_ref = ref
    cached_ref_key = None

    def make_data_key(cur_gen_data, cur_gen_args):
        def normalize(arg):
            if isinstance(arg, torch.Tensor):
                return ("tensor", arg.data_ptr(), tuple(arg.shape), str(arg.dtype))
            if isinstance(arg, (tuple, list)):
                return tuple(normalize(el) for el in arg)
            if isinstance(arg, dict):
                return tuple(
                    sorted((key, normalize(value)) for key, value in arg.items())
                )
            return arg

        return (id(cur_gen_data), normalize(cur_gen_args))

    def ensure_data(cur_gen_data, cur_gen_args):
        nonlocal data, data_key, cached_ref_key
        cur_data_key = make_data_key(cur_gen_data, cur_gen_args)
        if cur_data_key != data_key:
            data = (
                cur_gen_data(*cur_gen_args, device=device)
                if not input_data and cur_gen_data is not None
                else input_data
            )
            data_key = cur_data_key
            cached_ref_key = None
        return data

    try:
        gpu_id = gpuID

        rets = []
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
            data = ensure_data(gen_data, gen_args)

            new_args = (
                (tuple(data[k] for k in args[0]) + tuple(args[1:]))
                if gen_data is not None
                else args
            )

            if ref_noused is not None:
                ref = ref_noused
            else:
                ref = cached_ref
                _cur_key = (id(ref_func), ref_args, data_key)
                if (
                    ref is None
                    and not fast_mode
                    or (ref_func is not None and fast_mode)
                ) and _cur_key != cached_ref_key:
                    ref_data_keys_i, *rest_i = ref_args
                    updated = tuple(data[k] for k in ref_data_keys_i) + tuple(rest_i)
                    ref = ref_func(*updated, **ref_kwargs)
                    torch.cuda.synchronize()
                    cached_ref = ref
                    cached_ref_key = _cur_key

            # Extract rtol, atol from rest if available, otherwise use defaults.
            # Optional rest[2]: custom compare callable (e.g. cosine diff for a8w4).
            # Optional rest[3]: explicit max_abs_delta for catastrophic error detection.
            # Optional rest[4]: output_keys -- names of output tensors to NaN-init.
            rtol = rest[0] if len(rest) > 0 else 1e-2
            atol = rest[1] if len(rest) > 1 else 1e-2
            compare_fn = rest[2] if len(rest) > 2 and callable(rest[2]) else None
            max_abs_delta = rest[3] if len(rest) > 3 else None
            output_keys = (
                rest[4]
                if len(rest) > 4 and isinstance(rest[4], (list, tuple))
                else None
            )
            arg_key_list = list(args[0]) if gen_data is not None else None

            work_args = (
                gpu_id,
                info,
                func,
                new_args,
                kwargs,
                ref,
                rtol,
                atol,
                verbose,
                err_ratio,
                compare_fn,
                max_abs_delta,
                output_keys,
                arg_key_list,
                True,
                return_status,
            )

            # Run worker with explicit GPU ID
            ret = worker(*work_args)
            rets.append(ret)
            if progress_queue is not None:
                progress_queue.put(ret)
        return rets

    except Exception as e:  # noqa: BLE001
        import traceback

        if _is_accelerator_error(e):
            raise
        print(f"Critical error in work_group: {e!r}")
        traceback.print_exc()
        # Allocation failed before any candidate launched.
        status = (
            "oom_preflight"
            if isinstance(e, torch.cuda.OutOfMemoryError)
            or "out of memory" in str(e).lower()
            else "crash"
        )
        # Return dummy failed results for all tasks in the group.
        if isinstance(tasks, list):
            results = [
                _format_worker_result(
                    task[0] if task else "unknown",
                    float("inf"),
                    1.0,
                    status,
                    return_status,
                )
                for task in tasks
            ]
        else:
            results = [
                _format_worker_result(
                    tasks[0] if tasks else "unknown",
                    float("inf"),
                    1.0,
                    status,
                    return_status,
                )
            ]
        if progress_queue is not None:
            for result in results:
                progress_queue.put(result)
        return results
    finally:
        data = None
        cached_ref = None
        ref = None
        gc.collect()
        try:
            torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001 - a faulted HIP context may reject cleanup
            pass


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
    return_status=False,
    result_callback=None,
):
    """Multi-process tuner with one long-lived worker per selected GPU.

    Shape-grouped callers reuse one input/reference allocation for every
    candidate of a shape. GPU faults and hangs trigger a full pool restart;
    each group releases its tensors and allocator cache before the worker takes
    another shape.

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
    if gpu_num < 1:
        raise RuntimeError("mp_tuner requires at least one visible GPU")
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
        from collections import OrderedDict

        info_key_groups = OrderedDict()
        for task in tasks:
            info_keys = task[0][0] if task and len(task) > 0 else None
            if info_keys not in info_key_groups:
                info_key_groups[info_keys] = []
            info_key_groups[info_keys].append(task)

        task_group = list(info_key_groups.values())
        print(
            f"[Task Grouping] Grouped {len(tasks)} tasks into {len(task_group)} groups by info_keys"
        )

        # in_datas already has one entry per shape from the tuner;
        # just verify cardinality matches and use it directly.
        assert len(task_group) == len(
            in_datas
        ), f"shape_grouped: group count ({len(task_group)}) != in_datas count ({len(in_datas)})"
        ref_data_index = list(range(len(task_group)))
    else:
        task_group = tasks
        import numpy as np

        cumulative = np.cumsum([size for size, _ in in_datas])
        ref_data_index = np.searchsorted(
            cumulative, np.arange(len(task_group)), side="right"
        )

    print(f"Distributing {len(task_group)} task groups across {mp_num} GPUs")

    manager = mp.Manager() if result_callback is not None else None
    progress_queue = manager.Queue() if manager is not None else None
    progress_results = {}

    def publish_progress(result):
        info = result[0]
        progress_results[info] = result
        if result_callback is not None:
            result_callback(result)

    def drain_progress():
        if progress_queue is None:
            return
        while True:
            try:
                publish_progress(progress_queue.get_nowait())
            except Empty:
                return

    # Helper function to submit tasks to pool
    def submit_tasks(pool, gpu_map, task_indices):
        """Submit tasks to the pool and return async results as a dict"""
        task_indices = list(task_indices)
        _reset_task_start_times(task_start_times, task_indices)
        return {
            k: pool.apply_async(
                _run_with_start_tracking,
                args=(
                    k,
                    work_group,
                    (
                        gpu_map,
                        fast_mode,
                        err_ratio,
                        in_datas[ref_data_index[k]],
                        task_group[k],
                        verbose,
                        return_status,
                        progress_queue,
                    ),
                ),
            )
            for k in task_indices
        }

    # Create initial pool and submit all tasks
    task_start_times = mp.RawArray("d", len(task_group))
    pool = mp.Pool(
        processes=parallel_num,
        initializer=_init_task_start_times,
        initargs=(task_start_times,),
    )
    pids = [pool.apply_async(get_pid) for i in range(start_idx, mp_num)]
    gpu_map = {el.get(): i + start_idx for i, el in enumerate(pids)}
    rets_dict = submit_tasks(pool, gpu_map, range(len(task_group)))
    # Convert to list for compatibility with existing code
    rets = [rets_dict[k] for k in range(len(task_group))]
    pool.close()

    result_dict = {}  # Store results by task index
    failed_tasks = []
    remaining_tasks = list(enumerate(rets))

    # Checkpoint-enabled callers drain per-candidate progress every second so
    # an abrupt parent failure loses at most a small in-flight window.
    check_interval = 1 if result_callback is not None else 10

    timeout_msg = (
        f"timeout={timeout}s each" if timeout is not None else "no timeout limit"
    )
    print(f"Waiting for {len(remaining_tasks)} tasks to complete ({timeout_msg})...")

    def add_dummy_result(k, results_list, status="crash"):
        """Helper function to add dummy failed result"""
        if shape_grouped:
            task_info = (
                task_group[k] if isinstance(task_group[k], list) else [task_group[k]]
            )
            published_failure = False
            for task in task_info:
                info = task[0] if len(task) > 0 else f"task_{k}"
                if info in progress_results:
                    results_list.append(progress_results[info])
                    continue
                result = _format_worker_result(
                    info, float("inf"), 1.0, status, return_status
                )
                results_list.append(result)
                # The first unfinished candidate is the one that faulted or
                # timed out. Later candidates never ran; leave them out of the
                # checkpoint so a resume loop can continue with them.
                if not published_failure:
                    publish_progress(result)
                    published_failure = True
        else:
            task = task_group[k]
            info = task[0] if len(task) > 0 else f"task_{k}"
            if info in progress_results:
                results_list.append(progress_results[info])
                return
            result = _format_worker_result(
                info, float("inf"), 1.0, status, return_status
            )
            results_list.append(result)
            publish_progress(result)

    # Process tasks as they complete
    pool_restart_needed = False
    logged_error_types = (
        set()
    )  # Track error types that already logged to avoid duplicates

    while remaining_tasks:
        drain_progress()
        completed_this_round = []
        dummy_failed_tasks = []
        consecutive_timeouts = 0
        half_gpu = max(1, (mp_num + 1) // 2)

        for k, async_result in remaining_tasks:
            try:
                elapsed = _elapsed_since_task_start(task_start_times, k)
                if elapsed is None:
                    # The task is still queued, so it has no execution timeout yet.
                    if not async_result.ready():
                        consecutive_timeouts = 0
                        continue
                    actual_timeout = 0
                elif timeout is not None:
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
                consecutive_timeouts = 0
                elapsed = _elapsed_since_task_start(task_start_times, k)
                if verbose:
                    print(
                        f"[Done] Task {k}/{len(rets) - 1} completed in {elapsed:.1f}s ({len(result_dict)}/{len(rets)} done)"
                    )

            except MPTimeoutError:
                # Check if this specific task has exceeded its timeout (only if timeout is set)
                if timeout is not None:
                    elapsed = _elapsed_since_task_start(task_start_times, k)

                    if elapsed is not None and elapsed > timeout:
                        consecutive_timeouts += 1

                        error_msg = f"[!] Task {k} timed out after {elapsed:.1f}s (limit: {timeout}s) - likely GPU hang or infinite loop"
                        print(error_msg)
                        failed_tasks.append((k, "timeout"))

                        # Add dummy result
                        drain_progress()
                        dummy_results = []
                        add_dummy_result(k, dummy_results, "timeout")
                        result_dict[k] = (
                            dummy_results if shape_grouped else [dummy_results[0]]
                        )
                        completed_this_round.append((k, async_result))

                        # Trigger pool restart for timeout (similar to crash)
                        pool_restart_needed = True

                        # If half the GPUs worth of consecutive timeouts, pool is in bad shape
                        if consecutive_timeouts >= half_gpu:
                            print(
                                f"\n[!] {consecutive_timeouts} consecutive tasks timed out (>= {half_gpu}/{mp_num} GPUs likely stuck)"
                            )
                            print("[!] Triggering immediate pool restart...\n")
                            break
                    else:
                        consecutive_timeouts = 0

            except Exception as e:  # noqa: BLE001
                # Check if it's a process crash (segfault, memory fault, etc.)
                error_type = type(e).__name__
                is_mapping_error = _is_mapping_error(e)
                is_accelerator_error = _is_accelerator_error(e)
                # not restart as this is not root use
                if is_mapping_error:
                    error_msg = f"[Mapping Error] Task {k} - Process PID not in GPU map: {error_type} - {e}"
                    dummy_failed_tasks.append((k, "mapping error"))
                    # A worker was replaced behind the PID-to-GPU map. Restart
                    # the pool and retry this unfinished group with a fresh map.
                    pool_restart_needed = True
                    break
                elif is_accelerator_error:
                    # GPU fault (e.g. illegal memory access): worker returns exception instead of
                    # hanging. Unlike hang->timeout, the faulting worker may stay alive and accept
                    # more tasks on the same bad GPU. Break immediately to trigger restart and
                    # terminate the pool before that worker processes further tasks (same as when
                    # fault used to hang and timeout would eventually break).
                    error_msg = f"\033[1;31m[GPU Fault]\033[0m Task {k} failed with {error_type}: {e}"
                    print(error_msg, flush=True)
                    failed_tasks.append((k, "accelerator error"))
                    drain_progress()
                    dummy_results = []
                    add_dummy_result(k, dummy_results, "crash")
                    result_dict[k] = (
                        dummy_results if shape_grouped else [dummy_results[0]]
                    )
                    completed_this_round.append((k, async_result))
                    pool_restart_needed = True
                    break
                else:
                    error_msg = f"[Failed] Task {k} failed with {error_type}: {e}"
                    failed_tasks.append((k, "unknown error"))

                    # Always record a dummy result so reconstruction never sees an empty list
                    # (previously only timeout path did this; async.get() failures left no result_dict[k]).
                    drain_progress()
                    dummy_results = []
                    add_dummy_result(k, dummy_results)
                    result_dict[k] = (
                        dummy_results if shape_grouped else [dummy_results[0]]
                    )
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
            if verbose:
                print(f"\n{'=' * 60}")
                print(
                    "? Pool restart needed due to crash. Restarting pool...", flush=True
                )
                print(f"Remaining tasks: {len(remaining_tasks)}", flush=True)
                print(f"{'=' * 60}\n", flush=True)

            # Terminate old pool
            try:
                pool.terminate()
                pool.join()
            except Exception as e:  # noqa: BLE001
                print(f"Warning: Error during pool termination: {e}", flush=True)
            # Create new pool
            pool = mp.Pool(
                processes=parallel_num,
                initializer=_init_task_start_times,
                initargs=(task_start_times,),
            )

            # Recreate gpu_map for new processes (new PIDs)
            pids = [pool.apply_async(get_pid) for i in range(start_idx, mp_num)]
            gpu_map = {el.get(): i + start_idx for i, el in enumerate(pids)}

            # Resubmit remaining tasks
            remaining_task_indices = [k for k, _ in remaining_tasks]
            new_rets_dict = submit_tasks(pool, gpu_map, remaining_task_indices)
            pool.close()

            # Update remaining_tasks with new async results
            remaining_tasks = [(k, new_rets_dict[k]) for k in remaining_task_indices]

            # Reset pool restart flag
            pool_restart_needed = False
            print(
                f"Pool restarted. Continuing with {len(remaining_tasks)} remaining tasks...\n",
                flush=True,
            )

        # Small sleep to avoid busy waiting
        if remaining_tasks and not completed_this_round:
            time.sleep(1)

    # Reconstruct results in original task order
    drain_progress()
    result = []
    for k in range(len(rets)):
        task_result = result_dict.get(k, [])
        if not task_result:
            # Defensive fallback: keep output cardinality stable even if a task result is missing.
            dummy_results = []
            add_dummy_result(k, dummy_results)
            task_result = dummy_results if shape_grouped else [dummy_results[0]]
        if shape_grouped:
            result.extend(task_result)
        else:
            result.append(task_result[0])

    # Clean up the pool
    try:
        pool.terminate()
        pool.join()
    except Exception as e:  # noqa: BLE001
        print(f"Warning: Error during pool cleanup: {e}")
    if manager is not None:
        drain_progress()
        manager.shutdown()

    # Print summary
    if failed_tasks:
        timeout_count = sum(1 for _, reason in failed_tasks if reason == "timeout")
        crash_count = len(failed_tasks) - timeout_count
        summary = (
            f"\n{'=' * 60}\n"
            f"Tuning Summary:\n"
            f"  Total tasks: {len(rets)}\n"
            f"  Successful: {len(rets) - len(failed_tasks)}\n"
            f"  Failed: {len(failed_tasks)}\n"
            f"    - Timeouts (GPU hang): {timeout_count}\n"
            f"    - Crashes (memory fault): {crash_count}\n"
            f"{'=' * 60}"
        )
        logger.warning(summary)

    return result
