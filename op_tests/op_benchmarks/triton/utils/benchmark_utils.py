import os
import json
import torch
import triton.language as tl
import sys
import time
import os
import tempfile
import re


# Base directory where configs are located
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))


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
    table.field_names = table_lines[0].split()
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
