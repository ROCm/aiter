# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
import gc
import math
import multiprocessing as mp
import os
import time
from multiprocessing import TimeoutError as MPTimeoutError
from queue import Empty

import torch

from aiter import dtypes, logger
from aiter.test_common import checkAllclose
from aiter_worker_limits import configure_worker_subprocesses

_TASK_START_TIMES = None
_TASK_PIDS = None
# rocprofiler-sdk exports these (setenv, so os.environ doesn't see them) when this
# process initializes the GPU runtime. Spawned workers inherit them, and then the
# torch profiler there records no GPU kernels, so every candidate times as 0 us.
_ROCPROFILER_ENV = ("ROCPROFILER_REGISTER_LIBRARY", "HSA_TOOLS_ROCPROFILER_V1_TOOLS")


def _new_pool(parallel_num, task_start_times, task_pids):
    for name in _ROCPROFILER_ENV:
        os.environ.pop(name, None)
        os.unsetenv(name)
    return mp.Pool(
        processes=parallel_num,
        initializer=_init_task_start_times,
        initargs=(task_start_times, task_pids),
    )


class GpuContextLost(RuntimeError):
    """Raised by a worker whose GPU context a candidate broke, whatever the
    original exception type, so the parent restarts the pool."""


def _is_mapping_error(exc: BaseException) -> bool:
    return isinstance(exc, KeyError)


def _is_accelerator_error(exc: BaseException) -> bool:
    # Parent side: by type only. The pickled exception carries no context.
    return type(exc).__name__ in ("AcceleratorError", "GpuContextLost")


def _raise_if_gpu_fault(exc: BaseException):
    """Worker side: raise GpuContextLost if ``exc`` left the GPU context
    unusable, i.e. it is an AcceleratorError or a synchronize now fails."""
    if isinstance(exc, GpuContextLost):
        raise exc
    if type(exc).__name__ == "AcceleratorError":
        raise GpuContextLost(_candidate_failure_detail(exc)) from exc
    if not torch.cuda.is_initialized():
        return
    try:
        torch.cuda.synchronize()
    except Exception as error:
        raise GpuContextLost(_candidate_failure_detail(exc)) from error


def _init_task_start_times(task_start_times, task_pids=None):
    global _TASK_START_TIMES, _TASK_PIDS
    configure_worker_subprocesses()
    _TASK_START_TIMES = task_start_times
    _TASK_PIDS = task_pids


def _run_with_start_tracking(task_index, func, args):
    if _TASK_START_TIMES is None:
        raise RuntimeError("Task start-time storage is not initialized")
    if _TASK_PIDS is not None:
        _TASK_PIDS[task_index] = os.getpid()
    _TASK_START_TIMES[task_index] = time.monotonic()
    return func(*args)


def _task_worker_exited(task_pids, task_index):
    """True once the worker that started this task is gone. A worker killed by a
    GPU memory fault never returns a result, so waiting on it only ends at the
    task timeout (or never, without one)."""
    pid = task_pids[task_index]
    if pid == 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    except PermissionError:
        return False
    return False


def _elapsed_since_task_start(task_start_times, task_index, now=None):
    started_at = task_start_times[task_index]
    if started_at == 0:
        return None
    current_time = time.monotonic() if now is None else now
    return current_time - started_at


def _reset_task_start_times(task_start_times, task_indices, task_pids=None):
    """Mark tasks as queued again, so a resubmit is not judged against the
    timestamp (or worker) its previous attempt left behind."""
    for k in task_indices:
        task_start_times[k] = 0
        if task_pids is not None:
            task_pids[k] = 0


def _merge_error_ratio(current, observed):
    if not math.isfinite(observed):
        return 1.0
    return max(current, observed)


def _candidate_failure_status(exc: BaseException) -> str:
    # Labels a failure only; kernels signal unsupported via RuntimeError text.
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
    return "crash"


def _candidate_failure_detail(exc: BaseException, limit: int = 300) -> str:
    """Exception class and one-line message for the candidate journal."""
    message = " ".join(str(exc).split())
    if len(message) > limit:
        message = message[: limit - 3] + "..."
    return f"{type(exc).__name__}: {message}" if message else type(exc).__name__


def _format_worker_result(info, us, max_err_ratio, status, return_status, detail=""):
    result = (info, us, round(max_err_ratio, 4))
    return (*result, status, detail) if return_status else result


def _failed_result(task, status, detail, return_status):
    info = task[0] if len(task) > 0 else "unknown"
    return _format_worker_result(info, float("inf"), 1.0, status, return_status, detail)


# A group whose resubmissions keep faulting before measuring anything (e.g. its
# reference itself faults) stops being resubmitted after this many faults.
_MAX_FAULTS_WITHOUT_PROGRESS = 5


def _info_key(info):
    try:
        hash(info)
    except TypeError:
        return repr(info)
    return info


def _measured_prefix(pending, measured):
    """Split a group's pending candidates into the results already measured,
    in order, and the candidates still to run. ``measured`` maps
    ``_info_key(info)`` to a result."""
    n = 0
    while (
        n < len(pending)
        and len(pending[n]) > 0
        and _info_key(pending[n][0]) in measured
    ):
        n += 1
    return [measured[_info_key(task[0])] for task in pending[:n]], pending[n:]


def _split_after_fault(pending, measured, status, detail, return_status):
    """Results of a group stopped by a fault: what it measured, then the
    candidate that was running (the first unmeasured one) failed with
    ``status``, and the candidates still to run."""
    done, rest = _measured_prefix(pending, measured)
    if rest:
        done.append(_failed_result(rest[0], status, detail, return_status))
        rest = rest[1:]
    return done, rest


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
    detail = ""
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
            _raise_if_gpu_fault(e)
            print(f"run gpu func warning: info:{info}\t {e}", flush=True)
            us = -1  # not support or error
            max_err_ratio = 1.0
            status = _candidate_failure_status(e)
            detail = _candidate_failure_detail(e)
        max_retries = 3
        retry_count = 0

        while us == 0 and retry_count < max_retries:
            print(f"!!!! us = 0, try {retry_count + 1} run")
            res, us = run_perftest(func, *args, **kwargs)
            retry_count += 1
        if us == 0:
            print(f"Warning: try run {max_retries} times, but still get 0!")
        torch.cuda.synchronize()
        if us == -1 or res is None:
            return _format_worker_result(
                info, us, max_err_ratio, status, return_status, detail
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
        _raise_if_gpu_fault(e)
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
        detail = _candidate_failure_detail(e)
    except TimeoutError as e:
        if printLog:
            print(f"Timeout in process:{pid} info:{info}: {e}")
        us = float("inf")
        max_err_ratio = 1.0
        status = "timeout"
        detail = _candidate_failure_detail(e)
    except Exception as e:  # noqa: BLE001
        _raise_if_gpu_fault(e)
        if printLog:
            print(f"Unexpected Error in process:{pid} info:{info}: {e}")
            import traceback

            traceback.print_exc()
        us = -1  # float("inf")
        max_err_ratio = 1.0
        status = _candidate_failure_status(e)
        detail = _candidate_failure_detail(e)

    return _format_worker_result(info, us, max_err_ratio, status, return_status, detail)


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

    rets = []

    def record(ret):
        rets.append(ret)
        if progress_queue is not None:
            progress_queue.put(ret)

    try:
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
            try:
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
                        updated = tuple(data[k] for k in ref_data_keys_i) + tuple(
                            rest_i
                        )
                        ref = ref_func(*updated, **ref_kwargs)
                        torch.cuda.synchronize()
                        cached_ref = ref
                        cached_ref_key = _cur_key
            except Exception as e:  # noqa: BLE001 - a fault is re-raised below
                import traceback

                _raise_if_gpu_fault(e)
                print(f"Critical error in work_group: {e!r}")
                traceback.print_exc()
                status = (
                    "oom_preflight"
                    if isinstance(e, torch.cuda.OutOfMemoryError)
                    or "out of memory" in str(e).lower()
                    else "crash"
                )
                detail = f"failed before launch: {_candidate_failure_detail(e)}"
                record(_failed_result(group_task[i], status, detail, return_status))
                continue

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
                gpuID,
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
            record(worker(*work_args))
        return rets
    finally:
        data = None
        cached_ref = None
        ref = None
        # Per call this costs ~120 ms, so only a shape group pays it; an
        # ungrouped candidate's fault restarts the pool anyway.
        if shape_grouped:
            gc.collect()
            try:
                torch.cuda.empty_cache()
            except Exception as error:  # noqa: BLE001 - cleanup is best-effort
                logger.debug("cache cleanup after a faulted context failed: %s", error)


def get_pid(delay=3):
    if delay:
        time.sleep(delay)
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
        verbose: Print per-task progress and pool restarts
        return_status: Also return status and detail per candidate

    A candidate that faults the GPU, kills its worker or hangs fails alone: the
    pool restarts and the group's remaining candidates are resubmitted, keeping
    what it had already measured.

    Returns:
        One result per candidate, in task order: (info, latency, error_ratio).
        With return_status, (info, latency, error_ratio, status, detail), where
        status is one of ok, mismatch, unsupported, crash, timeout,
        oom_runtime, oom_preflight (outside a candidate's run, e.g. generating
        inputs) or not_run (never run: its group kept faulting before
        measuring anything).
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

        # in_datas has one entry per shape from the tuner, in task order; a
        # shape without candidates has an entry (count 0) but no group.
        in_datas = [data for data in in_datas if data[0] > 0]
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

    # Per-candidate results, so a faulted group resumes after what it measured.
    # Under spawn the manager process re-imports the tuner script (~1 GB), so
    # start it only when a group holds more than one candidate.
    grouped = any(isinstance(g, list) and len(g) > 1 for g in task_group)
    manager = mp.Manager() if grouped else None
    progress_queue = manager.Queue() if grouped else None
    progress_results = {}
    pending = {
        k: list(group) if isinstance(group, list) else [group]
        for k, group in enumerate(task_group)
    }
    settled = {k: [] for k in pending}
    barren_faults = dict.fromkeys(pending, 0)

    def drain_progress():
        while progress_queue is not None:
            try:
                result = progress_queue.get_nowait()
            except Empty:
                return
            progress_results[_info_key(result[0])] = result

    def settle_measured(k):
        """Move what group k measured out of its pending candidates; True once
        nothing is left to run."""
        drain_progress()
        done, pending[k] = _measured_prefix(pending[k], progress_results)
        settled[k].extend(done)
        if not pending[k]:
            result_dict[k] = settled[k]
        return not pending[k]

    def settle_fault(k, status, detail):
        """Fail the candidate group k was running when it faulted; True once
        nothing is left to resubmit."""
        drain_progress()
        done, rest = _split_after_fault(
            pending[k], progress_results, status, detail, return_status
        )
        barren_faults[k] = barren_faults[k] + 1 if len(done) <= 1 else 0
        if rest and barren_faults[k] >= _MAX_FAULTS_WITHOUT_PROGRESS:
            reason = (
                f"not run: the group faulted {barren_faults[k]} times in a row "
                f"before measuring anything (last: {status})"
            )
            done += [_failed_result(t, "not_run", reason, return_status) for t in rest]
            rest = []
        settled[k].extend(done)
        pending[k] = rest
        if not rest:
            result_dict[k] = settled[k]
        return not rest

    # Helper function to submit tasks to pool
    def submit_tasks(pool, gpu_map, task_indices):
        """Submit tasks to the pool and return async results as a dict"""
        task_indices = list(task_indices)
        _reset_task_start_times(task_start_times, task_indices, task_pids)
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
                        (
                            (len(pending[k]), in_datas[ref_data_index[k]][1])
                            if isinstance(task_group[k], list)
                            else in_datas[ref_data_index[k]]
                        ),
                        (
                            pending[k]
                            if isinstance(task_group[k], list)
                            else task_group[k]
                        ),
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
    task_pids = mp.RawArray("i", len(task_group))
    pool = _new_pool(parallel_num, task_start_times, task_pids)
    pid_delay = 0 if mp_num == 1 else 3
    pids = [
        pool.apply_async(get_pid, args=(pid_delay,)) for i in range(start_idx, mp_num)
    ]
    gpu_map = {el.get(): i + start_idx for i, el in enumerate(pids)}
    rets_dict = submit_tasks(pool, gpu_map, range(len(task_group)))
    # Convert to list for compatibility with existing code
    rets = [rets_dict[k] for k in range(len(task_group))]
    pool.close()

    result_dict = {}  # Store results by task index
    failed_tasks = []
    remaining_tasks = list(enumerate(rets))

    check_interval = 10  # Check every 10 seconds for responsive polling

    timeout_msg = (
        f"timeout={timeout}s each" if timeout is not None else "no timeout limit"
    )
    print(f"Waiting for {len(remaining_tasks)} tasks to complete ({timeout_msg})...")

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
                result_dict[k] = settled[k] + list(task_result)
                completed_this_round.append((k, async_result))
                consecutive_timeouts = 0
                elapsed = _elapsed_since_task_start(task_start_times, k)
                if verbose:
                    print(
                        f"[Done] Task {k}/{len(rets) - 1} completed in {elapsed:.1f}s ({len(result_dict)}/{len(rets)} done)"
                    )

            except MPTimeoutError:
                if _task_worker_exited(task_pids, k):
                    print(
                        f"[!] Task {k} lost its worker (pid {task_pids[k]} exited) - likely a GPU fault",
                        flush=True,
                    )
                    failed_tasks.append((k, "worker exited"))
                    if settle_fault(
                        k,
                        "crash",
                        f"worker process {task_pids[k]} exited; "
                        "likely a GPU memory fault",
                    ):
                        completed_this_round.append((k, async_result))
                    pool_restart_needed = True
                    break
                # Check if this specific task has exceeded its timeout (only if timeout is set)
                if timeout is not None:
                    elapsed = _elapsed_since_task_start(task_start_times, k)

                    if elapsed is not None and elapsed > timeout:
                        consecutive_timeouts += 1

                        error_msg = f"[!] Task {k} timed out after {elapsed:.1f}s (limit: {timeout}s) - likely GPU hang or infinite loop"
                        print(error_msg)
                        failed_tasks.append((k, "timeout"))
                        if settle_fault(
                            k,
                            "timeout",
                            f"exceeded {timeout}s after {elapsed:.1f}s; "
                            "likely GPU hang or infinite loop",
                        ):
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
                    if settle_fault(
                        k,
                        "crash",
                        f"accelerator fault: {_candidate_failure_detail(e)}",
                    ):
                        completed_this_round.append((k, async_result))
                    pool_restart_needed = True
                    break
                else:
                    error_msg = f"[Failed] Task {k} failed with {error_type}: {e}"
                    failed_tasks.append((k, "unknown error"))
                    if settle_fault(k, "crash", _candidate_failure_detail(e)):
                        completed_this_round.append((k, async_result))
                    else:
                        pool_restart_needed = True

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
            # Groups the restart interrupted resume after what they measured.
            remaining_tasks = [
                (k, ar) for k, ar in remaining_tasks if not settle_measured(k)
            ]
            pool_restart_needed = False
            if remaining_tasks:
                # Create new pool
                pool = _new_pool(parallel_num, task_start_times, task_pids)

                # Recreate gpu_map for new processes (new PIDs)
                pids = [pool.apply_async(get_pid) for i in range(start_idx, mp_num)]
                gpu_map = {el.get(): i + start_idx for i, el in enumerate(pids)}

                # Resubmit remaining tasks
                remaining_task_indices = [k for k, _ in remaining_tasks]
                new_rets_dict = submit_tasks(pool, gpu_map, remaining_task_indices)
                pool.close()

                # Update remaining_tasks with new async results
                remaining_tasks = [
                    (k, new_rets_dict[k]) for k in remaining_task_indices
                ]
                print(
                    f"Pool restarted. Continuing with {len(remaining_tasks)} remaining tasks...\n",
                    flush=True,
                )

        # Small sleep to avoid busy waiting
        if remaining_tasks and not completed_this_round:
            time.sleep(1)

    # Reconstruct results in original task order
    result = []
    for k in range(len(rets)):
        task_result = result_dict.get(k)
        if not task_result:
            # Defensive fallback: keep output cardinality stable even if a task result is missing.
            task_result = settled[k] + [
                _failed_result(
                    task,
                    "crash",
                    "no result returned by the worker pool",
                    return_status,
                )
                for task in pending[k]
            ]
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
