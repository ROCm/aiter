import os
import json
import torch
import triton.language as tl
import triton
import warnings
import sys
import time
import tempfile
import re
import matplotlib.pyplot as plt

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

    benchmark = triton.testing.Benchmark(
        x_names=x_names,
        x_vals=x_vals_list,
        x_log=True,
        y_log=True,
        line_arg="provider",
        line_vals=["Triton"],
        line_names=["Triton"],
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
        M_list = [args.M] if args.M is not None else [2**i for i in range(0, 15)]
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
        M_list = [args.M] if args.M is not None else [2**i for i in range(0, 15)]
    batch_size = args.B if args.B is not None else 16
    shapes = []
    for M in M_list:
        for model_name, config in configs.items():
            N = config["intermediate_size"]
            K = config["hidden_size"]

            shapes.append(
                (M, N, K, batch_size, model_name)
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
    config_path="./utils/model_configs.json", models="llama3,mistral_7B"
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
    table.field_names = re.split(r" {1,}", table_lines[0].strip())
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
    if dtype in [torch.float16, tl.float16]:
        return 2
    elif dtype in [torch.bfloat16, tl.bfloat16]:
        return 2
    elif dtype in [torch.float32, tl.float32]:
        return 4
    elif dtype == torch.int32:
        return 4
    elif dtype == torch.int64:
        return 8
    elif dtype in [
        torch.float8_e4m3fnuz,
        torch.float8_e5m2fnuz,
        tl.float8e4,
        tl.float8e5,
    ]:
        return 1
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")
