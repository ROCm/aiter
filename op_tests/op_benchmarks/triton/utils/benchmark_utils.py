import inspect
import json
import math
import os
import random
import re
import sys
import tempfile
import time
import warnings
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import torch
import triton
import triton.language as tl

# Base directory where configs are located
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))


def get_shape_benchmark_object(plot_name, args, x_names=None):
    """
    Utility function for returning a triton.testing.Benchmark object to populate.

    Note: This is for benchmarking GEMM kernels without the --model flag. The distinction
    comes in the x_names and x_vals: For models, we use hidden_dim and intermediate_dim
    as args, but if we're just given a shape, we use M, N, K.
    """
    if x_names is None:
        x_names = ["M", "N", "K"]

    if args.shape:
        if len(x_names) == len(args.shape):
            x_vals_list = [args.shape]
        elif len(x_names) == 4 and len(args.shape) == 3:  # for batched GEMM kernels
            if hasattr(args, "B") and args.B is not None:
                x_vals_list = [[args.B] + args.shape]
            else:
                batch_size = 16
                warnings.warn(
                    f"Batch size not specified in --shape or with -B, using default: {batch_size}."
                )
                x_vals_list = [[batch_size] + args.shape]
        else:
            raise ValueError(
                f"Incompatible --shape provided: {args.shape}. Expected a shape that matches {x_names}."
            )
    else:
        x_vals_list = get_x_vals(dims=len(x_names), args=args)

    if args.metric == "time":
        ylabel = "Time (ms)"
    elif args.metric == "throughput":
        ylabel = "Throughput (TFLOPS)"
    elif args.metric == "bandwidth":
        ylabel = "Bandwidth (GB/s)"
    else:
        raise NotImplementedError(f"{args.metric} is not supported")

    evaluation_metric_to_unit = {
        "throughput": "TFLOPS",
        "time": "Time_(ms)",
        "bandwidth": "Bandwidth_(GB/s)",  # spaces break prettytable parsing
    }
    benchmark = triton.testing.Benchmark(
        x_names=x_names,
        x_vals=x_vals_list,
        x_log=True,
        y_log=True,
        line_arg="unit",
        line_vals=[evaluation_metric_to_unit[args.metric]],
        line_names=[evaluation_metric_to_unit[args.metric]],
        styles=[("green", "-")],
        ylabel=ylabel,
        plot_name=plot_name,
        args={"metric": args.metric},
    )
    return benchmark


def get_model_benchmark_object(
    plot_name, args, x_names=None, model_benchmark_shapes_fn=None
):
    """
    Utility function for returning a triton.testing.Benchmark object to populate.

    Note: This is for benchmarking models (e.g with the --model arg).
    """
    if x_names is None:
        x_names = ["M", "hidden_dim", "intermediate_dim", "model_name"]
    if model_benchmark_shapes_fn is None:
        model_benchmark_shapes_fn = model_benchmark_shapes
    if not args.fc1 and not args.fc2:
        # by default, benchmark both
        warnings.warn(
            "No specific layer selected for benchmarking, defaulting to both. To specify a layer, use -fc1 or -fc2."
        )
        args.fc1 = True
        args.fc2 = True
    x_vals_list = model_benchmark_shapes_fn(args)

    if args.metric == "time":
        ylabel = "Time (ms)"
    elif args.metric == "throughput":
        ylabel = "Throughput (TFLOPS)"
    elif args.metric == "bandwidth":
        ylabel = "Bandwidth (GB/s)"
    else:
        raise NotImplementedError(f"{args.metric} is not supported")

    line_names = []
    if args.fc1:
        line_names.append("fc1")
    if args.fc2:
        line_names.append("fc2")
    line_vals = line_names

    mpl_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    benchmark = triton.testing.Benchmark(
        x_names=x_names,
        x_vals=x_vals_list,
        x_log=True,
        y_log=True,
        line_arg="layer",
        line_vals=line_vals,
        line_names=line_names,
        styles=[
            (mpl_colors[i], "-") for i in range(len(line_names))
        ],  # match line names to colors
        ylabel=ylabel,
        plot_name=plot_name,
        args={"metric": args.metric},
    )
    return benchmark


def model_benchmark_shapes(args):
    config_file = args.model_configs
    configs = get_model_configs(config_path=config_file, models=args.model)
    if args.model == "all":
        M_list = [args.M if args.M is not None else 4096]
    else:
        M_list = [args.M] if args.M is not None else [2**i for i in range(15)]
    shapes = []
    for M in M_list:
        for model_name, config in configs.items():
            shapes.append(
                (M, config["hidden_size"], config["intermediate_size"], model_name)
            )

    return shapes


def batched_model_benchmark_shapes(args):
    """
    Generate benchmark shapes when profiling a batched GEMM with the --model arg.
    """
    config_file = args.model_configs
    configs = get_model_configs(config_path=config_file, models=args.model)
    if args.model == "all":
        M_list = [args.M if args.M is not None else 4096]
    else:
        M_list = [args.M] if args.M is not None else [2**i for i in range(15)]
    batch_size = args.B if args.B is not None else 16
    shapes = []
    for M in M_list:
        for model_name, config in configs.items():
            hidden_size = config["hidden_size"]
            intermediate_size = config["intermediate_size"]

            shapes.append(
                (M, hidden_size, intermediate_size, batch_size, model_name)
            )  # rearrange batch to last dim so M is graph x-axis

    return shapes


def get_x_vals(dims: int, args=None):
    """
    Get a default set of benchmarking values (M, N, K).
    """
    assert dims in [3, 4], "Invalid number of dimensions"
    if args.M is not None:
        x_vals = [(args.M, 1280, 8192)]
    else:
        x_vals = [
            (1, 1280, 8192),
            (32, 1280, 8192),
            (64, 1280, 8192),
            (128, 1280, 8192),
            (192, 1280, 8192),
            (256, 1280, 8192),
            (320, 1280, 8192),
            (512, 1280, 8192),
            (1024, 1280, 8192),
            (2048, 1280, 8192),
            (4096, 1280, 8192),
            (8192, 1280, 8192),
            (16384, 1280, 8192),
            (4096, 4096, 4096),
            (4096, 4096, 4160),
        ]
    if dims == 4:
        if hasattr(args, "B") and args.B is not None:
            batch_size = args.B
        else:
            batch_size = 16  # by default
            warnings.warn(
                f"Batch size not specified in --shape or with -B, using default: {batch_size}"
            )
        x_vals = [tuple([batch_size] + list(i)) for i in x_vals]  # (B, M, N, K)
    return x_vals


def get_model_configs(
    config_path="./utils/model_configs.json", models="llama3,mixtral_7B"
):
    """
    Load model names from the configuration file.

    Args:
        config_path (str): User-provided path to the configuration JSON file.
        models: List of model names to retrieve, with pattern <modelfamily_modelsize>. If modelfamily specified only, retrieves all the modelsizes.

    Returns:
        dict: A dictionary of available models and their configurations for the specified families.
    """
    # Resolve config path relative to ./perf-kernels/
    config_path = os.path.join(BASE_DIR, config_path)

    with open(config_path, "r") as f:
        configs = json.load(f)

    # Extract models and their configurations for the specified families
    filtered_configs = {}

    if models == "all":
        models = [model for model in configs]
    else:
        models = models.replace(" ", "").split(",")

    for model in models:
        delimiter = "_" if "_" in model else "-"
        model_specs = model.split(delimiter)
        model_family = model_specs[0]

        if model_family in configs:
            model_size = model_specs[1] if len(model_specs) > 1 else None
            # Check if model filtering is required
            if model_size is None:  # Include all models in the family
                # Include all models in the family
                for model_size, model_configs in configs[model_family].items():
                    filtered_configs[f"{model_family}-{model_size}"] = model_configs
            else:
                if model_size in configs[model_family]:
                    filtered_configs[f"{model_family}-{model_size}"] = configs[
                        model_family
                    ][model_size]

    if not filtered_configs:
        print(f"Warning: No models selected with the provided model names: {models}")

    return filtered_configs


def get_available_models(config_file="utils/model_configs.json", filter=None):
    """
    Load model names from the configuration file.

    Args:
        config_file (str): Path to the configuration JSON file.

    Returns:
        list: A list of available model configs.
    """
    # Resolve config path relative to ./perf-kernels/
    config_path = os.path.join(BASE_DIR, config_file)

    with open(config_path, "r") as f:
        configs = json.load(f)

    models = [
        f"{family}-{model}"
        for family in configs
        for model in configs[family]
        if filter is None or filter in f"{family}-{model}"
    ]

    return models


def parse_vgpr_usage(file_path, table_start="result-table-name"):
    from prettytable import PrettyTable

    with open(file_path, "r") as f:
        lines = f.readlines()

    # Extract VGPR-related information
    vgpr_info = []
    table_lines = []
    in_table = False

    for line in lines:
        # Parse autotuning outputs
        if re.search(r"Autotuning kernel", line):
            vgpr_info.append(line.strip())
        if re.search(r"Triton autotuning for function", line):
            vgpr_info.append(line.strip())

        if re.search(r"\.name:", line):
            vgpr_info.append(line.strip())
        if re.search(r"\.vgpr_count:", line) or re.search(r"\.vgpr_spill_count:", line):
            vgpr_info.append(line.strip())
        # Detect start of table
        if re.match(rf"^\s*{table_start}", line):
            vgpr_info.append(line.strip())
            in_table = True
        elif in_table:
            table_lines.append(line.strip())

    # Print extracted information
    print("\n".join(vgpr_info))
    table = PrettyTable()
    table.field_names = re.split(r" {2,}", table_lines[0].strip())
    [table.add_row(line.split()[1:]) for line in table_lines[1:]]

    print(table)


def print_vgpr(fun, table_start="result-table-name"):
    # Create a temporary file
    with tempfile.NamedTemporaryFile(mode="w+", delete=False) as temp_file:
        output_file = temp_file.name

        # Redirect stdout and stderr to the temporary file
        sys.stdout = temp_file
        sys.stderr = temp_file

        os.environ["AMDGCN_ENABLE_DUMP"] = "1"
        os.environ["TRITON_ALWAYS_COMPILE"] = "1"
        os.environ["TRITON_PRINT_AUTOTUNING"] = "1"
        fun()  # run the function

        sys.stdout.flush()
        sys.stderr.flush()

    # Restore stdout and stderr to normal
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__

    time.sleep(0.5)  # Ensure everything is written before reading

    # Parse and print relevant output
    parse_vgpr_usage(output_file, table_start)

    # Remove the temporary file
    os.unlink(output_file)


def get_dtype_bytes(dtype):
    if dtype in [torch.float16, tl.float16] or dtype in [torch.bfloat16, tl.bfloat16]:
        return 2
    elif dtype in [torch.float32, tl.float32] or dtype == torch.int32:
        return 4
    elif dtype == torch.int64:
        return 8
    elif dtype in [
        torch.float8_e4m3fn,
        torch.float8_e4m3fnuz,
        torch.float8_e5m2fnuz,
        tl.float8e4nv,
        tl.float8e5,
    ]:
        return 1
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")


def get_caller_name_no_ext():
    caller_file = inspect.stack()[1].filename  # full path of caller
    return Path(caller_file).stem  # filename without extension


def run_input_generator(
    gen: Callable[[], tuple], fields: Sequence[str]
) -> dict[str, Any]:
    values = gen()
    if len(values) != len(fields):
        raise ValueError(
            f"{gen!r} returned {len(values)} values, "
            f"expected {len(fields)}: {fields}"
        )
    return {name: val for name, val in zip(fields, values) if name != "_"}


def generate_rotating_buffers_pool(
    gens: Sequence[Callable],
    out_vals_keys: Sequence[Sequence[str]],
    target_mb: int,
    min_pool_len: int,
    max_pool_len: int | None = None,
    print_info: bool = False,
):
    per_set_bytes = 0
    seen_storage_ptrs = set()
    # Keep every generator's output alive for the whole sizing loop below: once
    # a kwargs dict is dropped, its tensors' storages can be freed and the
    # caching allocator can hand the same address to a later gen's unrelated
    # tensor, which would then look like an already-counted alias and get
    # skipped. Release these extra references once sizing is done, before
    # building the real pool.
    keepalive = []
    for gen, fields in zip(gens, out_vals_keys):
        kwargs = run_input_generator(gen, fields)
        keepalive.append(kwargs)
        for v in kwargs.values():
            if not isinstance(v, torch.Tensor):
                continue
            storage = v.untyped_storage()
            ptr = storage.data_ptr()
            # Aliased views (e.g. a reshape of another returned tensor) share
            # backing storage; count it once, by its real size, not once per
            # view, or pool sizing undercounts/overcounts the true footprint.
            if ptr in seen_storage_ptrs:
                continue
            seen_storage_ptrs.add(ptr)
            per_set_bytes += storage.nbytes()
    del keepalive

    if per_set_bytes == 0:
        raise ValueError(
            "per_set_bytes=0, specify gens that generate tensors with size > 0"
        )

    del kwargs, v, storage

    pool_len = max(min_pool_len, math.ceil(target_mb * 1024 * 1024 / per_set_bytes))

    # max_pool_len is needed in case the size of input set is small, then covering
    # target_mb might create pool_len in order of millions. This will require
    # do_bench_aiter_triton to capture one kernel launch per pool entry i.e. a long
    # graph capture. So, we need to bound this.
    if max_pool_len is not None:
        pool_len = min(pool_len, max(max_pool_len, min_pool_len))

    pool = []
    for _ in range(pool_len):
        entry = {}
        for gen, fields in zip(gens, out_vals_keys):
            entry.update(run_input_generator(gen, fields))
        pool.append(entry)

    if print_info:
        print(
            f"Rotating Buffers, per_set_sz(MB):{round(per_set_bytes/(1024*1024),2)}"
            + f" target_sz(MB):{target_mb} pool_len:{pool_len}"
        )
    return pool


def _rotating_buffers_fns(
    fn: Callable, rot_bufs: list[dict[str, torch.Tensor]], n: int
):
    """``n`` callables that rotate through ``rot_bufs`` in a shuffled order.

    The shuffle keeps a full cycle hitting every pool entry once, but not in
    the same fixed sequential order every capture -- useful when per-set
    sizes are small relative to the overall pool target size.
    """
    pool_len = len(rot_bufs)
    order = list(range(pool_len))
    random.Random(0).shuffle(order)

    def _make_fn(idx):
        return lambda: fn(**(rot_bufs[order[idx % pool_len]]))

    return [_make_fn(i) for i in range(n)]


def do_bench_aiter_triton(
    fn: Callable,
    rot_bufs: list[dict[str, torch.Tensor]],
    warmup_iters: int,
    kernels_per_graph: int,
    num_replays: int,
    warmup_replays: int,
) -> float:
    """Rotating-buffer CUDA graph benchmark, single process.

    Captures ``max(kernels_per_graph, len(rot_bufs))`` calls to ``fn`` into
    one CUDA graph, rotating through ``rot_bufs`` so the whole pool is
    covered -- a CUDA graph replays the exact calls captured, so a pool
    bigger than ``kernels_per_graph`` would otherwise leave entries unused.
    Returns the mean per-kernel latency in ms.
    """

    # warmup
    for _ in range(warmup_iters):
        fn(**(rot_bufs[0]))
    torch.cuda.synchronize()

    pool_len = len(rot_bufs)
    actual_kernels_per_graph = max(kernels_per_graph, pool_len)
    fns = _rotating_buffers_fns(fn, rot_bufs, actual_kernels_per_graph)

    # capture graph
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        for f in fns[: min(3, len(fns))]:
            f()
        torch.cuda.synchronize()

        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, stream=stream):
            for f in fns:
                f()
    torch.cuda.synchronize()

    # warmup replays
    for _ in range(warmup_replays):
        g.replay()
    torch.cuda.synchronize()

    # Timed replays
    start_events = []
    end_events = []
    for _ in range(num_replays):
        se = torch.cuda.Event(enable_timing=True)
        ee = torch.cuda.Event(enable_timing=True)
        se.record()
        g.replay()
        ee.record()
        start_events.append(se)
        end_events.append(ee)
    torch.cuda.synchronize()

    times_ms = [se.elapsed_time(ee) for se, ee in zip(start_events, end_events)]
    per_kernel_ms = [t / actual_kernels_per_graph for t in times_ms]

    ms = sum(per_kernel_ms) / len(per_kernel_ms)

    return ms


def do_bench_aiter_triton_synced(
    fn: Callable,
    rot_bufs: list[dict[str, torch.Tensor]],
    warmup_iters: int,
    kernels_per_graph: int,
    num_replays: int,
    warmup_replays: int,
    dist_group,
) -> float:
    """Rotating-buffer CUDA graph benchmark, synced across ranks.

    Use this instead of :func:`do_bench_aiter_triton` when ``fn`` performs
    real cross-rank communication (e.g. a combine/all-to-all step), so every
    rank's capture and replay phases line up. ``dist_group`` is a
    ``torch.distributed`` process group; it is only a barrier at each phase
    boundary (warmup, capture, replay) -- unlike the old
    ``do_bench_cudagraph_synced``, no cross-rank latency estimate needs to
    be agreed on, because ``kernels_per_graph`` is a fixed constant and is
    therefore already identical on every rank.

    Ranks may still generate differently sized rotating buffer pools (e.g.
    differing local shapes), so growing the per-graph call count from the
    local pool size alone would desync ranks on replay. This function agrees
    the call count across ranks itself with one extra all-reduce before
    capture; callers do not need to pre-align pool sizes.

    The returned latency is the local rank's own measurement; callers that
    want one number across ranks must reduce it themselves (e.g.
    ``dist.all_reduce(..., op=MAX)``).
    """
    import torch.distributed as dist

    def _barrier():
        dist.barrier(group=dist_group)

    # warmup
    _barrier()
    for _ in range(warmup_iters):
        fn(**(rot_bufs[0]))
    torch.cuda.synchronize()

    pool_len = len(rot_bufs)
    actual_kernels_per_graph = max(kernels_per_graph, pool_len)

    # Ranks can have differently sized pools; agree on one call count so
    # every rank captures/replays the same number of fn() calls, or
    # collectives inside fn would desync ranks whose pools differ.
    # Keep on the CPU: the all-reduce uses the gloo (CPU) group.
    agreed = torch.tensor([actual_kernels_per_graph], dtype=torch.int64, device="cpu")
    dist.all_reduce(agreed, op=dist.ReduceOp.MAX, group=dist_group)
    actual_kernels_per_graph = int(agreed[0])

    fns = _rotating_buffers_fns(fn, rot_bufs, actual_kernels_per_graph)

    # capture graph
    _barrier()
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        for f in fns[: min(3, len(fns))]:
            f()
        torch.cuda.synchronize()

        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, stream=stream):
            for f in fns:
                f()
    torch.cuda.synchronize()

    # warmup replays
    _barrier()
    for _ in range(warmup_replays):
        g.replay()
    torch.cuda.synchronize()

    # Timed replays
    start_events = []
    end_events = []
    for _ in range(num_replays):
        se = torch.cuda.Event(enable_timing=True)
        ee = torch.cuda.Event(enable_timing=True)
        se.record()
        g.replay()
        ee.record()
        start_events.append(se)
        end_events.append(ee)
    torch.cuda.synchronize()

    times_ms = [se.elapsed_time(ee) for se, ee in zip(start_events, end_events)]
    per_kernel_ms = [t / actual_kernels_per_graph for t in times_ms]

    ms = sum(per_kernel_ms) / len(per_kernel_ms)

    return ms
