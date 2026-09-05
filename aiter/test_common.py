# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
import copy
import json
import multiprocessing as mp
import os

import numpy as np
import pandas as pd
import torch
import torch.profiler as tpf

from aiter import logger

pd.set_option("display.max_rows", 200)
_SMI_LABEL_COUNTS = {}
## debug ##
# pd.set_option("display.max_rows", None)
# pd.set_option("display.max_columns", None)
# pd.set_option("display.width", None)
# pd.set_option("display.max_colwidth", None)
# pd.set_option("display.expand_frame_repr", False)


def print_json_table(name, rows, keep=None):
    """Print benchmark rows as one record-oriented JSON object.

    A single-line object is intentional: parent benchmark drivers can validate
    and forward it without parsing pandas' human-readable table formats.
    """
    if isinstance(rows, pd.DataFrame):
        df = rows.copy()
    else:
        df = pd.DataFrame([row for row in rows if row is not None])
    if not df.empty:
        df = df.replace("", pd.NA).dropna(axis=1, how="all")
        if keep is not None:
            cols = [column for column in keep if column in df.columns]
            cols += [
                column
                for column in df.columns
                if "err_msg" in column and column not in cols
            ]
            df = df[cols]
    records = json.loads(df.to_json(orient="records"))
    print(json.dumps({"name": name, "rows": records}), flush=True)


def ensure_spawn_method():
    """
    Ensure multiprocessing uses 'spawn' start method.

    This is required for CUDA/distributed tests. Only sets the method if
    it hasn't been set yet, avoiding conflicts with existing initialization.

    Usage:
        Called at the beginning of multi-GPU test functions before spawning
        worker processes.
    """
    try:
        current_method = mp.get_start_method(allow_none=True)
        if current_method is None:
            mp.set_start_method("spawn")
        elif current_method != "spawn":
            logger.warning(
                f"Multiprocessing start method already set to '{current_method}', "
                f"expected 'spawn'. This may cause issues with CUDA."
            )
    except RuntimeError:
        # Already set, which is fine
        pass


def perftest(
    num_iters=101,
    num_warmup=2,
    testGraph=False,
    num_rotate_args=0,
    needTrace=False,
    use_cuda_event=False,
):
    def decorator(func):
        def wrapper(*args, **kwargs):
            num = num_rotate_args
            if num < 1:
                gpu_id = torch.cuda.current_device()
                iter_used_memory, inputSize, _, _ = device_memory_profiling(
                    func, *args, **kwargs
                )

                properties = torch.cuda.get_device_properties(gpu_id)
                free_memory = torch.cuda.mem_get_info(gpu_id)[0]
                cache_size = min(
                    getattr(properties, "L2_cache_size", 4096 * 1024) * 64 * 128,
                    (free_memory - iter_used_memory + inputSize) * 0.9,
                )
                cache_size = max(cache_size, 0)
                num = int((cache_size + inputSize - 1) // inputSize)
            num = min(num, num_iters)

            rotate_args = [
                (copy.deepcopy(args), copy.deepcopy(kwargs)) for _ in range(num - 1)
            ] + [(args, kwargs)]
            run_iters(num_warmup, func, *args, **kwargs)
            torch.cuda.synchronize()
            if int(os.environ.get("AITER_LOG_MORE", "0")) or use_cuda_event:
                latencies = []
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                for _ in range(num_iters):
                    start_event.record()
                    data = func(*args, **kwargs)
                    end_event.record()
                    end_event.synchronize()
                    latencies.append(start_event.elapsed_time(end_event))
                    torch.cuda.empty_cache()
                avg = np.mean(latencies) * 1000
                logger.info(f"avg: {avg} us/iter from cuda.Event")
                if use_cuda_event:
                    return data, avg

            with tpf.profile(
                activities=[tpf.ProfilerActivity.CPU, tpf.ProfilerActivity.CUDA],
                profile_memory=False,
                with_stack=False,
                with_modules=True,
                # record_shapes=True,
                on_trace_ready=(
                    tpf.tensorboard_trace_handler(f"./aiter_logs/gpu_id_{gpu_id}")
                    if needTrace
                    else None
                ),
            ) as prof:
                data = run_iters_rotate(num_iters, func, rotate_args)
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
            avg = get_trace_perf(prof, num_iters)

            if testGraph:
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    data = run_iters_rotate(num_iters, func, rotate_args)
                with tpf.profile(
                    activities=[tpf.ProfilerActivity.CPU, tpf.ProfilerActivity.CUDA],
                    profile_memory=True,
                    with_stack=True,
                    with_modules=True,
                ) as prof:
                    run_iters(1, graph.replay)
                avg = get_trace_perf(prof, num_iters)
                logger.info(f"avg: {avg} us/iter with hipgraph")

            if os.environ.get("AITER_SMI_MONITOR", "0") == "1":
                fn_name = getattr(func, "__name__", "kernel")
                skipped = {
                    name.strip()
                    for name in os.environ.get("AITER_SMI_SKIP_FUNCTIONS", "").split(",")
                    if name.strip()
                }
                if fn_name in skipped:
                    return data, avg
                # Import lazily: normal library/test use has no amdsmi dependency.
                # Combo and its child UTs run from the repository root, where the
                # standalone op_tests monitor module is importable.
                try:
                    from op_tests.smi_monitor import replay_with_smi
                except ModuleNotFoundError:
                    # Direct ``python op_tests/foo.py`` puts op_tests itself,
                    # rather than the repository root, at sys.path[0].
                    from smi_monitor import replay_with_smi

                if testGraph:
                    replay = graph.replay
                    # One replay contains num_iters calls captured above.
                    replay_us = avg * num_iters
                else:
                    replay_index = 0

                    def replay():
                        nonlocal replay_index
                        replay_args, replay_kwargs = rotate_args[
                            replay_index % len(rotate_args)
                        ]
                        replay_index += 1
                        return func(*replay_args, **replay_kwargs)

                    replay_us = avg

                case_label = os.environ.get("AITER_SMI_LABEL", "benchmark_case")
                label_key = (case_label, fn_name)
                occurrence = _SMI_LABEL_COUNTS.get(label_key, 0) + 1
                _SMI_LABEL_COUNTS[label_key] = occurrence
                replay_with_smi(
                    replay,
                    label=f"{case_label}/{fn_name}#{occurrence}",
                    synchronize=torch.cuda.synchronize,
                    estimated_us=replay_us,
                )

            return data, avg

        return wrapper

    return decorator


def benchmark():
    def decorator(func):
        def wrapper(*args, **kwargs):
            callargs = log_args(func, *args, **kwargs)
            ret = func(*args, **kwargs)
            if ret is not None:
                callargs.update(ret)
            return callargs

        return wrapper

    return decorator


def device_memory_profiling(func, *args, **kwargs):
    gpu_id = torch.cuda.current_device()
    inputSize = (
        sum(
            [
                el.nbytes
                for el in args
                if isinstance(el, torch.Tensor) and el.device.index == gpu_id
            ]
        )
        + 1
    )
    torch.cuda.reset_peak_memory_stats(gpu_id)
    cuda_memory_before = (
        torch.cuda.mem_get_info(gpu_id)[1] - torch.cuda.mem_get_info(gpu_id)[0]
    )
    torch_memory_before = torch.cuda.memory_reserved(gpu_id)
    torch_peak_before = torch.cuda.memory_stats(gpu_id).get(
        "allocated_bytes.all.peak", 0
    )
    non_torch_memory_before = cuda_memory_before - torch_memory_before

    _ = func(*args, **kwargs)

    torch.cuda.reset_peak_memory_stats(gpu_id)
    cuda_memory_after = (
        torch.cuda.mem_get_info(gpu_id)[1] - torch.cuda.mem_get_info(gpu_id)[0]
    )
    torch_memory_after = torch.cuda.memory_reserved(gpu_id)
    torch_peak_after = torch.cuda.memory_stats(gpu_id).get(
        "allocated_bytes.all.peak", 0
    )
    non_torch_memory_after = cuda_memory_after - torch_memory_after

    torch_peak_increase = torch_peak_after - torch_peak_before
    non_torch_increase = non_torch_memory_after - non_torch_memory_before
    iter_used_memory = torch_peak_increase + non_torch_increase + inputSize

    return iter_used_memory, inputSize, torch_peak_increase, non_torch_increase


def run_iters(num_iters, func, *args, **kwargs):
    data = None
    for _ in range(num_iters):
        data = func(*args, **kwargs)
    return data


def run_iters_rotate(num_iters, func, rotate_args):
    data = None
    num_rotate_args = len(rotate_args)
    for _ in range(num_iters):
        args, kwargs = rotate_args[_ % num_rotate_args]
        data = func(*args, **kwargs)

    return data


def run_perftest(
    func,
    *args,
    num_iters=101,
    num_warmup=2,
    testGraph=False,
    num_rotate_args=0,
    needTrace=False,
    use_cuda_event=False,
    **kwargs,
):
    @perftest(
        num_iters=num_iters,
        num_warmup=num_warmup,
        testGraph=testGraph,
        num_rotate_args=num_rotate_args,
        needTrace=needTrace,
        use_cuda_event=use_cuda_event,
    )
    def worker(*args, **kwargs):
        return func(*args, **kwargs)

    return worker(*args, **kwargs)


def log_args(func, *args, **kwargs):
    import inspect

    callargs = inspect.getcallargs(func, *args, **kwargs)

    prefix = f"calling {func.__name__}("
    blanks = " " * (len(prefix))

    def getTensorInfo(el):
        if isinstance(el, torch.Tensor):
            return f"{el.shape} {el.dtype} {el.device} {hex(el.data_ptr())}"
        elif isinstance(el, tuple):
            viewNum = 5
            if len(el) > viewNum:
                el = list(el[:viewNum]) + ["..."]
            return f'\n{" "*(len(prefix)+31)}'.join(
                ["("] + [f" {getTensorInfo(e)}" for e in el] + [")"]
            )
        return el

    info = [f"{el:<28} = {getTensorInfo(callargs[el])}" for el in callargs]
    info = f",\n{blanks}".join(info)
    logger.info(f"\n{prefix}{info})")
    return callargs


def post_process_data(df, num_iters, warm_iter=1):
    """remove abnormal data"""

    device_df = df[df["device_type"].astype(str).str.contains("DeviceType.CUDA")]
    # print("devicedf is ", device_df)
    if device_df.empty:
        return [], 0
    kernels_num = int(len(device_df) / num_iters)

    act_iters = num_iters
    valid_n = len(device_df)
    dropped_indexs = []
    if len(device_df) % num_iters == 0:
        kernels_num = int(len(device_df) / num_iters)
    else:
        ##get correct kernel num
        name_list = device_df["name"].tolist()
        max_kernel_num = 20
        n = len(name_list)
        for step in range(1, min(max_kernel_num, n // 2 + 1)):
            sub_list = [name_list[i] for i in range(step)]
            m = len(sub_list)

            valid_n = int(n / m) * m
            pattern_match = all(
                name_list[i] == sub_list[i % m] for i in range(int(n / m) * m)
            )
            if pattern_match:
                kernels_num = m
                act_iters = valid_n / m
                break
        dropped_indexs = device_df.iloc[valid_n:].index.tolist()
        if kernels_num == 0:
            print("data missed, the time may be inaccurate!")

    test_df = device_df.iloc[:valid_n].reset_index()
    grouped_kernel_df = test_df.groupby(test_df.index // kernels_num, sort=False).agg(
        {"self_device_time_total": "sum", "index": list}
    )

    # rm warm iters
    sum_df = grouped_kernel_df.iloc[warm_iter:].reset_index(drop=True)
    out_range_idx = []
    if num_iters > 30:
        # IQR to remove abnormal data
        k = 1.5
        Q1 = sum_df["self_device_time_total"].quantile(0.25)
        Q3 = sum_df["self_device_time_total"].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - k * IQR
        upper = Q3 + k * IQR
        out_range_idx = sum_df.index[
            (sum_df["self_device_time_total"] < lower)
            | (sum_df["self_device_time_total"] > upper)
        ].tolist()
    out_range_num = len(out_range_idx)

    indices = {idx for i in out_range_idx for idx in sum_df.iloc[i]["index"]}

    index_sublists = grouped_kernel_df["index"].head(warm_iter).tolist()
    indices_to_add = [idx for sublist in index_sublists for idx in sublist]
    indices.update(indices_to_add)
    indices.update(dropped_indexs)
    if int(os.environ.get("AITER_LOG_MORE", "0")):
        logger.info(f"abnormal data indices: {indices}")
        for i in indices:
            logger.info(f"abnormal data: {df.iloc[i]['self_device_time_total']}")
    return list(indices), out_range_num + warm_iter + num_iters - act_iters


def get_trace_perf(prof, num_iters):
    assert num_iters > 1
    warm_iter = 1
    num_iters -= warm_iter
    df = []
    cols = [
        "name",
        "self_cpu_time_total",
        "self_device_time_total",
        "device_type",
        "device_index",
    ]
    for el in prof.events():
        df.append([getattr(el, x, None) for x in cols])
    df = pd.DataFrame(df, columns=cols)
    ###remove abnormal data
    dropped_num = warm_iter
    dropped_indexs, dropped_num = post_process_data(
        df, num_iters + warm_iter, warm_iter
    )
    df = df.drop(dropped_indexs)
    iter_init = 0  # warm_iter dropped
    df["cnt"] = 1
    rets = []

    for name, d in df.groupby("name", sort=False):
        kernel_num_per_iter = iter_init
        if str(d["device_type"].iat[0]).split(".")[-1] != "CUDA":
            kernel_num_per_iter = 1
        r = d.iloc[kernel_num_per_iter:][
            ["cnt", "self_cpu_time_total", "self_device_time_total"]
        ].sum()
        if not r.empty:
            device_type = str(d["device_type"].iat[0]).split(".")[-1]
            r["name"] = name
            r["device_type"] = device_type
            r["device_index"] = str(d["device_index"].iat[0])
            if device_type == "CUDA":
                r["device_time_sum"] = r["self_device_time_total"]
                r["host_time_sum"] = 0
            else:
                r["host_time_sum"] = r["self_device_time_total"]
                r["device_time_sum"] = 0
            r["device_time_avg"] = (
                r["device_time_sum"] / r["cnt"] if r["cnt"] > 0 else 0
            )
        rets.append(r)
    df = pd.DataFrame(rets)
    cols = [
        "name",
        "cnt",
        "host_time_sum",
        "device_time_sum",
        "device_time_avg",
        "device_type",
        "device_index",
    ]
    cols = [el for el in cols if el in df.columns]
    df = df[(df.host_time_sum > 0) | (df.device_time_sum > 0)]

    timerList = [
        "host_time_sum",
        "device_time_sum",
    ]
    df = df[cols].sort_values(timerList, ignore_index=True)
    actual_iters = num_iters + warm_iter - dropped_num
    if df.empty:
        logger.info("no valida data after post process!")

    avg_name = "[avg us/iter]"
    for el in timerList:
        if el == "host_time_sum":
            df.at[avg_name, el] = df[el].sum() / num_iters
        else:
            df.at[avg_name, el] = df[el].sum() / actual_iters
    if int(os.environ.get("AITER_LOG_MORE", "0")):
        pd.set_option("display.expand_frame_repr", False)
        pd.set_option("display.max_colwidth", 90)
        pd.set_option("display.float_format", "{:,.1f}".format)
        logger.info(f"{df}")
    return df.at[avg_name, "device_time_sum"]


_CATASTROPHIC_REL_THRESHOLD = 0.5


def _relmag_catastrophic(actual_max_delta, b):
    """Relative-magnitude catastrophic heuristic.

    Triggers when ``max(|a - b|) > ref_abs_max * 0.5`` -- i.e. a single
    element diverges by more than half of the reference tensor's peak
    magnitude. Designed to catch real precision regressions in kernels that
    write plausible-looking but wrong values to specific positions (e.g.
    bpreshuffle precision drift, wrong scale/quant, half-broken pipeline).

    By contract, ``catastrophic_check=True`` is opt-in: the caller asserts
    the comparison is *position-sensitive* (no sort/ties permutation
    semantics). For position-insensitive data (sorted topk_ids, sort+gather
    weights with degenerate scores, byte-viewed fp4) this heuristic would
    misfire, so callers MUST NOT enable it there.

    For non-floating-point tensors this returns False -- there is no
    meaningful "magnitude" notion for integer indices/IDs. Callers who want
    a hard cap on integer deltas can still use explicit ``max_abs_delta``.
    """
    if not b.is_floating_point():
        return False
    ref_abs_max = max(b.abs().max().item(), 1.0)
    return actual_max_delta > ref_abs_max * _CATASTROPHIC_REL_THRESHOLD


def _check_catastrophic(actual_max_delta, a, b, max_abs_delta, catastrophic_check):
    """Decide whether a checkAllclose mismatch is "catastrophic" (fail-fast).

    Priority order (returns True at the first hit):

    1. Explicit ``max_abs_delta`` -- opt-in hard cap, takes precedence over
       the relative heuristic for callers that know the acceptable absolute
       magnitude.
    2. ``catastrophic_check=True`` -- enables NaN/Inf detection and the
       relative-magnitude heuristic (delta > ref_max * 0.5). NaN/Inf in
       either tensor is catastrophic (covers tuner NaN sentinel and
       numerically blown-up kernels). Do NOT enable on data that may
       legitimately contain NaN in padding regions.
    3. Otherwise: not catastrophic. The caller gets ``err_ratio`` back via
       the normal return value and decides what to do with it.

    ``torch.isfinite`` is safe on integer / byte tensors (returns all True),
    so this function works uniformly across dtypes.
    """
    if max_abs_delta is not None:
        return actual_max_delta > max_abs_delta
    if catastrophic_check:
        if not torch.isfinite(a).all() or not torch.isfinite(b).all():
            return True
        return _relmag_catastrophic(actual_max_delta, b)
    return False


def _catastrophic_check_silent(a, b, max_abs_delta, catastrophic_check):
    """Same policy as ``_check_catastrophic`` but without an already-computed
    ``actual_max_delta``. Used by the not-printLog (tuner) fast path so we
    avoid materialising masked tensors when ``isclose`` already failed."""
    if max_abs_delta is not None:
        return (a - b).abs().max().item() > max_abs_delta
    if catastrophic_check:
        if not torch.isfinite(a).all() or not torch.isfinite(b).all():
            return True
        return _relmag_catastrophic((a - b).abs().max().item(), b)
    return False


def checkAllclose(
    a,
    b,
    rtol=1e-2,
    atol=1e-2,
    tol_err_ratio=0.05,
    msg="",
    printNum=8,
    printLog=True,
    max_abs_delta=None,
    catastrophic_check=False,
):
    isClose = torch.isclose(a, b, rtol=rtol, atol=atol)

    if isClose.all():
        if printLog:
            logger.info(f"{msg}[checkAllclose {atol=} {rtol=} \033[32mpassed~\033[0m]")
        return 0
    else:
        try:
            mask = ~isClose
            num = mask.sum()
            printNum = min(printNum, num)
            percent = (num / a.numel()).item()
            if not printLog:
                if percent >= tol_err_ratio:
                    return percent
                is_cat = _catastrophic_check_silent(
                    a, b, max_abs_delta, catastrophic_check
                )
                return 1.0 if is_cat else percent
            a_msked = a[mask]
            b_msked = b[mask]
            delta = (a_msked - b_msked).abs()
        except RuntimeError:
            mask = ~isClose.to("cpu")
            num = mask.sum()
            printNum = min(printNum, num)
            percent = (num / a.numel()).item()
            if not printLog:
                if percent >= tol_err_ratio:
                    return percent
                is_cat = _catastrophic_check_silent(
                    a, b, max_abs_delta, catastrophic_check
                )
                return 1.0 if is_cat else percent
            a_msked = a[mask]
            b_msked = b[mask]
            delta = (a_msked - b_msked).abs()

        actual_max_delta = delta.max().item()
        is_catastrophic = _check_catastrophic(
            actual_max_delta, a, b, max_abs_delta, catastrophic_check
        )

        if is_catastrophic:
            logger.info(
                f"""{msg}[checkAllclose {atol=} {rtol=} \033[31mcatastrophic!\033[0m] max abs delta {actual_max_delta:.4f}
    a    : {a.shape}
           {a_msked[:printNum]}
    b    : {b.shape}
           {b_msked[:printNum]}
    delta:
           {delta[:printNum]}"""
            )
        elif percent > tol_err_ratio:
            logger.info(f"""{msg}[checkAllclose {atol=} {rtol=} \033[31mfailed!\033[0m]
    a    : {a.shape}
           {a_msked[:printNum]}
    b    : {b.shape}
           {b_msked[:printNum]}
    delta:
           {delta[:printNum]}""")
        else:
            logger.info(
                f"""{msg}[checkAllclose {atol=} {rtol=} \033[33mwarning!\033[0m] a and b results are not all close"""
            )
        logger.info(
            f"-->max abs delta:{delta.max()}, delta details: {percent:.1%} ({num} of {a.numel()}) elements"
        )
        if is_catastrophic:
            raise AssertionError(
                f"{msg}catastrophic error: max abs delta {actual_max_delta:.4f}, "
                f"{percent:.1%} ({num} of {a.numel()}) elements mismatch"
            )
        return percent


# --------------------------------------------------------------------------- #
# DATA / SCALE init.
#   gen = make_generator(seed)
#   x  = fill(shape, dist, gen, dtype=...)            # bf16 / fp32 / float8
#   s  = fill_scale(shape, dist, gen)                 # float32 block scale
#   xq = fill_fp4(shape, dist, gen)                   # MXFP4 packed e2m1
#   x8 = fill_fp8(shape, dist, gen)                   # MXFP8 e4m3
#   s8 = fill_scale_e8m0(shape, dist, gen)            # MX E8M0 on-wire
#   s4 = fill_scale_e4m3(shape, dist, gen)            # NVFP4 E4M3 on-wire
#
# OCP (Open Compute Project) published the MX microscaling formats: e2m1/e4m3
# data plus a tiny E8M0 (or E4M3) scale per block. fill_fp* emit those on-wire
# buffers. Large 2-D tensors are filled in row chunks (~1 GiB f32 staging).
# --------------------------------------------------------------------------- #
DATA_DISTS = ("zero", "constant", "uniform", "norm")
SCALE_DISTS = DATA_DISTS
SCALE_UNIFORM = (0.5, 2.0)
SCALE_NORM_MEAN, SCALE_NORM_STD = 1.0, 0.25
FP8_E4M3 = torch.float8_e4m3fn
FP4_UNIFORM = (-3.0, 3.0)  # e2m1 max is 6.0; keep headroom
FP8_UNIFORM = (-6.0, 6.0)
E8M0_BIAS = 127
E8M0_NEUTRAL = 0x7F  # 2^0 = 1.0
E4M3_NEUTRAL = 0x38  # e4m3 exp bias -> 1.0
E4M3_SCALE_MEAN, E4M3_SCALE_STD = 0.34375, 0.08
POW2_BINOMIAL_N = 10
E8M0_SCALE_DISTS = ("zero", "constant", "uniform", "norm", "auto", "pow2_binomial")
E4M3_SCALE_DISTS = ("zero", "constant", "uniform", "norm", "auto")
_STAGE_ELEMS = 1 << 28  # 256M f32 = 1 GiB per chunk


def make_generator(seed, device="cuda"):
    """Seeded ``torch.Generator`` -- same seed => bit-identical buffers."""
    return torch.Generator(device=device).manual_seed(int(seed))


def add_data_init_args(
    parser, *, default_dist="uniform", default_scale="constant", default_seed=0
):
    """Attach ``--data-init``, ``--scale-init`` and ``--seed``."""
    parser.add_argument(
        "--data-init",
        dest="data_init",
        nargs="*",
        choices=list(DATA_DISTS),
        default=[default_dist],
        help="DATA init: zero | constant | uniform | norm (N(0,1)). "
        "e.g.: --data-init uniform norm",
    )
    parser.add_argument(
        "--scale-init",
        dest="scale_init",
        nargs="*",
        choices=list(SCALE_DISTS),
        default=[default_scale],
        help="SCALE init (non-negative float): zero | constant(=1) | "
        "uniform U(0.5,2) | norm N(1,0.25). Independent of --data-init.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=default_seed,
        help="RNG seed; same seed -> bit-identical uniform/norm buffers",
    )
    return parser


def _row_chunks(rows, cols):
    """Row slices whose f32 staging stays around _STAGE_ELEMS elements."""
    step = max(_STAGE_ELEMS // max(cols, 1), 1)
    for start in range(0, rows, step):
        yield start, min(start + step, rows)


def _canon_dist(dist, allowed):
    if dist == "gaussian":
        dist = "norm"
    if dist not in allowed:
        raise ValueError(f"dist {dist!r}; choose from {allowed}")
    return dist


def _sample_data_f32(shape, dist, gen, *, lo, hi, device):
    if dist == "uniform":
        return torch.empty(shape, dtype=torch.float32, device=device).uniform_(
            lo, hi, generator=gen
        )
    if dist == "norm":
        return torch.empty(shape, dtype=torch.float32, device=device).normal_(
            0.0, 1.0, generator=gen
        )
    raise ValueError(f"data dist {dist!r} is not continuous; use fill dispatch")


def _sample_scale_f32(shape, dist, gen, *, lo, hi, device):
    if dist == "uniform":
        v = torch.empty(shape, dtype=torch.float32, device=device).uniform_(
            lo, hi, generator=gen
        )
    elif dist == "norm":
        v = torch.empty(shape, dtype=torch.float32, device=device).normal_(
            SCALE_NORM_MEAN, SCALE_NORM_STD, generator=gen
        )
    else:
        raise ValueError(f"scale dist {dist!r} is not continuous; use fill_scale")
    v.clamp_(min=0.0)
    return v


def _fill_sampled(shape, dist, gen, *, dtype, device, uniform, constant, sample_fn):
    if dist == "zero":
        return torch.zeros(shape, dtype=dtype, device=device)
    if dist == "constant":
        return torch.full(shape, constant, dtype=dtype, device=device)
    lo, hi = uniform
    if len(shape) != 2:
        return sample_fn(shape, dist, gen, lo=lo, hi=hi, device=device).to(dtype)
    rows, cols = shape
    out = torch.empty(shape, dtype=dtype, device=device)
    for r0, r1 in _row_chunks(rows, cols):
        v = sample_fn((r1 - r0, cols), dist, gen, lo=lo, hi=hi, device=device)
        out[r0:r1] = v.to(dtype)
        del v
    return out


def fill(
    shape,
    dist,
    gen,
    *,
    dtype=torch.float32,
    device="cuda",
    uniform=(-1.0, 1.0),
    constant=1.0,
):
    """Return a ``dtype`` DATA tensor of ``shape``.

    ``dist`` in {zero, constant, uniform, norm}. ``uniform`` is U(lo, hi);
    ``norm`` / ``gaussian`` is N(0, 1). ``zero`` / ``constant`` ignore ``gen``.
    """
    dist = _canon_dist(dist, DATA_DISTS)
    return _fill_sampled(
        shape,
        dist,
        gen,
        dtype=dtype,
        device=device,
        uniform=uniform,
        constant=constant,
        sample_fn=_sample_data_f32,
    )


def fill_scale(
    shape,
    dist,
    gen,
    *,
    dtype=torch.float32,
    device="cuda",
    uniform=SCALE_UNIFORM,
    constant=1.0,
):
    """Return a non-negative float SCALE tensor of ``shape``.

    Same dist names as ``fill``, sampled independently. ``constant`` defaults
    to 1.0 (neutral). ``norm`` is N(1, 0.25) clamped >= 0 -- not DATA's N(0,1).
    For MX on-wire scales use ``fill_scale_e8m0`` / ``fill_scale_e4m3``.
    """
    dist = _canon_dist(dist, SCALE_DISTS)
    return _fill_sampled(
        shape,
        dist,
        gen,
        dtype=dtype,
        device=device,
        uniform=uniform,
        constant=constant,
        sample_fn=_sample_scale_f32,
    )


def _f32_to_e8m0(v: torch.Tensor) -> torch.Tensor:
    """Round positive floats to the nearest E8M0 on-wire byte (bias 127)."""
    e = torch.zeros_like(v, dtype=torch.int32)
    pos = v > 0
    e[pos] = v[pos].log2().round().to(torch.int32) + E8M0_BIAS
    return e.clamp_(0, 255).to(torch.uint8)


def _popcount64(x: torch.Tensor) -> torch.Tensor:
    """Population count for a non-negative int64 tensor (SWAR bit-hack)."""
    x = x - ((x >> 1) & 0x5555555555555555)
    x = (x & 0x3333333333333333) + ((x >> 2) & 0x3333333333333333)
    x = (x + (x >> 4)) & 0x0F0F0F0F0F0F0F0F
    return (x * 0x0101010101010101) >> 56


def fill_fp4(shape, dist, gen, *, uniform=FP4_UNIFORM, device="cuda", constant=0):
    """MXFP4 on-wire: packed e2m1 ``uint8`` of shape ``(rows, cols // 2)``.

    Samples with the same DATA dists as ``fill``, then round-to-nearest e2m1.
    ``shape`` is the logical ``(rows, cols)``; ``cols`` must be even.
    """
    dist = _canon_dist(dist, DATA_DISTS)
    rows, cols = shape
    assert cols % 2 == 0, f"FP4 needs even columns, got {cols}"
    packed = (rows, cols // 2)
    if dist == "zero":
        return torch.zeros(packed, dtype=torch.uint8, device=device)
    if dist == "constant":
        return torch.full(packed, int(constant), dtype=torch.uint8, device=device)

    from aiter.utility import fp4_utils  # local: fp4_utils pulls in triton

    out = torch.empty(packed, dtype=torch.uint8, device=device)
    for r0, r1 in _row_chunks(rows, cols):
        v = _sample_data_f32(
            (r1 - r0, cols),
            dist,
            gen,
            lo=uniform[0],
            hi=uniform[1],
            device=device,
        )
        out[r0:r1] = fp4_utils.f32_to_mxfp4(v).view(torch.uint8)
        del v
    return out


def fill_fp8(shape, dist, gen, *, uniform=FP8_UNIFORM, device="cuda", constant=0.5):
    """MXFP8 on-wire: e4m3 tensor of ``shape``."""
    dist = _canon_dist(dist, DATA_DISTS)
    if dist == "zero":
        return torch.zeros(shape, dtype=FP8_E4M3, device=device)
    if dist == "constant":
        return torch.full(
            shape, float(constant), dtype=torch.float32, device=device
        ).to(FP8_E4M3)
    if len(shape) != 2:
        v = _sample_data_f32(
            shape, dist, gen, lo=uniform[0], hi=uniform[1], device=device
        )
        return v.to(FP8_E4M3)
    rows, cols = shape
    out = torch.empty(shape, dtype=FP8_E4M3, device=device)
    for r0, r1 in _row_chunks(rows, cols):
        v = _sample_data_f32(
            (r1 - r0, cols),
            dist,
            gen,
            lo=uniform[0],
            hi=uniform[1],
            device=device,
        )
        out[r0:r1] = v.to(FP8_E4M3)
        del v
    return out


def fill_scale_e8m0(
    shape,
    dist="auto",
    gen=None,
    *,
    device="cuda",
    n=POW2_BINOMIAL_N,
    constant=E8M0_NEUTRAL,
):
    """MX E8M0 on-wire ``uint8`` (biased exponent, bias 127).

    ``zero`` / ``constant`` / ``uniform`` / ``norm`` map from our SCALE dists
    (float then round to nearest power-of-two byte). ``auto`` /
    ``pow2_binomial`` match the MX GEMM default: 2^(Binomial(21,0.5)-11).
    """
    if dist == "gaussian":
        dist = "norm"
    if dist not in E8M0_SCALE_DISTS:
        raise ValueError(f"E8M0 scale dist {dist!r}; choose from {E8M0_SCALE_DISTS}")
    if dist == "zero":
        return torch.zeros(shape, dtype=torch.uint8, device=device)
    if dist == "constant":
        return torch.full(shape, int(constant), dtype=torch.uint8, device=device)
    if dist in ("uniform", "norm"):
        v = fill_scale(shape, dist, gen, device=device)
        return _f32_to_e8m0(v)
    # auto / pow2_binomial: Binomial(k, 0.5) == popcount of a uniform k-bit int
    trials = 2 * n + 1
    assert trials <= 24, "pow2_binomial popcount path assumes <= 24 trials"
    bits = torch.randint(
        0, 1 << trials, shape, dtype=torch.int64, device=device, generator=gen
    )
    e = _popcount64(bits).to(torch.int32) - (n + 1)
    return (e + E8M0_BIAS).clamp_(0, 255).to(torch.uint8)


def fill_scale_e4m3(
    shape, dist="auto", gen=None, *, device="cuda", constant=E4M3_NEUTRAL
):
    """NVFP4 / E4M3 on-wire ``uint8``.

    ``auto`` -> N(0.34375, 0.08) clamped >= 0, then cast e4m3 (MX GEMM default).
    ``uniform`` / ``norm`` use ``fill_scale`` then cast. ``constant`` is 0x38
    (1.0).
    """
    if dist == "gaussian":
        dist = "auto"
    if dist not in E4M3_SCALE_DISTS:
        raise ValueError(f"E4M3 scale dist {dist!r}; choose from {E4M3_SCALE_DISTS}")
    if dist == "zero":
        return torch.zeros(shape, dtype=torch.uint8, device=device)
    if dist == "constant":
        return torch.full(shape, int(constant), dtype=torch.uint8, device=device)
    if dist in ("uniform", "norm"):
        v = fill_scale(shape, dist, gen, device=device)
        return v.to(FP8_E4M3).view(torch.uint8)
    v = torch.empty(shape, dtype=torch.float32, device=device).normal_(
        E4M3_SCALE_MEAN, E4M3_SCALE_STD, generator=gen
    )
    v.clamp_(min=0.0)
    return v.to(FP8_E4M3).view(torch.uint8)


def tensor_dump(x: torch.Tensor, name: str, dir="./"):
    x_cpu = x.cpu().view(torch.uint8)
    filename = f"{dir}/{name}.bin"
    x_cpu.numpy().tofile(filename)
    logger.info(f"saving {filename} {x.shape}, {x.dtype}")

    with open(f"{dir}/{name}.meta", "w") as f:
        f.writelines([f"{el}\n" for el in [x.shape, x.dtype]])


def tensor_load(filename: str):
    DWs = np.fromfile(filename, dtype=np.uint32)
    metafile = ".".join(filename.split(".")[:-1]) + ".meta"
    with open(metafile) as fh:
        shape, dtype = [eval(line.strip()) for line in fh]
    return torch.tensor(DWs).view(dtype).view(shape)
