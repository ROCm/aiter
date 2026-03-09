# SPDX-License-Identifier: MIT
# Copyright (C) 2018-2026, Advanced Micro Devices, Inc. All rights reserved.

import argparse
import glob
import os
import sys
from collections import defaultdict

import numpy as np
import pandas as pd

pd.set_option("future.no_silent_downcasting", True)

this_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.basename(this_dir)
archs = [el for el in os.environ["AITER_GPU_ARCHS"].split(";")]
archs_supported = [
    os.path.basename(os.path.normpath(path)) for path in glob.glob(f"{this_dir}/*/")
]


content = """// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.
#pragma once
#include <unordered_map>

"""


def infer_cpp_type(series):
    return (
        "int"
        if pd.api.types.is_numeric_dtype(series)
        else "std::string"
    )


def format_cpp_value(value):
    if pd.isna(value):
        return "0"
    if isinstance(value, (int, float, np.integer, np.floating)):
        return f"{int(value):>4}"

    value_str = str(value)
    if value_str.replace(".", "", 1).isdigit():
        return f"{int(float(value_str)):>4}"

    escaped_value = value_str.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped_value}"'

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="generate",
        description="gen API for asm Bf16_gemm kernel",
    )
    parser.add_argument(
        "-m",
        "--module",
        required=True,
        help="""module of ASM kernel,
            e.g.: -m bf16gemm
        """,
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        default="aiter/jit/build",
        required=False,
        help="write all the blobs into a directory",
    )
    args = parser.parse_args()
    cfgs = []

    csv_groups = defaultdict(list)
    for arch in archs_supported:
        for el in glob.glob(
            f"{this_dir}/{arch}/{args.module}/**/*.csv", recursive=True
        ):
            cfgname = os.path.basename(el).split(".")[0]
            csv_groups[cfgname].append({"file_path": el, "arch": arch})

    ## deal with same name csv
    cfgs = []
    combined_cfgs = []
    required_output_columns = {"knl_name", "co_name", "arch"}
    other_columns = []
    other_column_types = {}
    for cfgname, file_info_list in csv_groups.items():
        dfs = []
        for file_info in file_info_list:
            single_file = file_info["file_path"]
            arch = file_info["arch"]
            df = pd.read_csv(single_file)
            # check headers
            headers_list = df.columns.tolist()
            required_input_columns = {"knl_name", "co_name"}
            if not required_input_columns.issubset(headers_list):
                missing = required_input_columns - set(headers_list)
                print(
                    f"ERROR: Invalid assembly CSV format -- {single_file}. Missing required columns: {', '.join(missing)}"
                )
                sys.exit(1)
            df["arch"] = arch  # add arch into df
            dfs.append(df)
        if dfs:
            relpath = os.path.relpath(
                os.path.dirname(single_file), f"{this_dir}/{arch}"
            )
            combine_df = pd.concat(dfs, ignore_index=True).infer_objects(copy=False)
            current_other_columns = [
                col
                for col in combine_df.columns.tolist()
                if col not in required_output_columns
            ]
            for col in current_other_columns:
                col_type = infer_cpp_type(combine_df[col])
                if col not in other_column_types:
                    other_columns.append(col)
                    other_column_types[col] = col_type
                elif col_type == "std::string":
                    other_column_types[col] = col_type
            combined_cfgs.append((cfgname, relpath, combine_df))

    if combined_cfgs:
        other_columns_comma = ", ".join(other_columns)
        other_columns_cpp_def = "\n".join(
            [f"    {other_column_types[col]} {col};" for col in other_columns]
        )
        content += f"""
#define ADD_CFG({other_columns_comma}, arch, path, knl_name, co_name)         \\
    {{                                         \\
        arch knl_name, {{ knl_name, path co_name, arch, {other_columns_comma} }}         \\
    }}

struct {args.module}Config
{{
    std::string knl_name;
    std::string co_name;
    std::string arch;
{other_columns_cpp_def}
}};

using CFG = std::unordered_map<std::string, {args.module}Config>;

"""

    for cfgname, relpath, combine_df in combined_cfgs:
        for col in other_columns:
            if col not in combine_df.columns:
                default_value = (
                    "" if other_column_types[col] == "std::string" else 0
                )
                combine_df[col] = default_value

        cfg = [
            "ADD_CFG("
            + ", ".join(format_cpp_value(row[col]) for col in other_columns)
            + f', "{row["arch"]}", "{relpath}/", "{row["knl_name"]}", "{row["co_name"]}"),'
            for row in combine_df.to_dict("records")
            if row["arch"] in archs
        ]
        cfg_txt = "\n    ".join(cfg) + "\n"

        txt = f"""static CFG cfg_{cfgname} = {{
    {cfg_txt}}};"""
        cfgs.append(txt)

    content += "\n".join(cfgs) + "\n"

    with open(f"{args.output_dir}/asm_{args.module}_configs.hpp", "w") as f:
        f.write(content)
