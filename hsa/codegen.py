# SPDX-License-Identifier: MIT
# Copyright (C) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

import argparse
import glob
import os
from collections import defaultdict

import numpy as np
import pandas as pd
from itertools import chain

pd.set_option("future.no_silent_downcasting", True)

this_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.basename(this_dir)
archs = [el for el in os.environ["AITER_GPU_ARCHS"].split(";")]
archs_supported = [
    os.path.basename(os.path.normpath(path)) for path in glob.glob(f"{this_dir}/*/")
]

code_object_arch_map = {
    ("gfx942", ""): "gfx942",
    ("gfx942", "MI300"): "gfx942",
    ("gfx942", "MI308"): "gfx942_MI308",
    ("gfx950", ""): "gfx950",
}

code_object_arch_layout = dict.fromkeys(code_object_arch_map.values()).keys()
code_object_valid_subarch = set(el[1] for el in code_object_arch_map)


code_object_wrapper_magic = ord("#")


header = """// SPDX-License-Identifier: MIT
// Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#pragma once

#include "aiter_hip_common.h"

"""


def create_co_objects_column(row, args, arch):
    co_objects = []
    base_dir = f"{this_dir}/{arch}/{args.module}/"
    for el in glob.glob(
        f"{this_dir}/{arch}/{args.module}/**/{row['co_name']}", recursive=True
    ):
        subarch_dir = os.path.relpath(os.path.dirname(el), base_dir)
        subarch = "" if subarch_dir not in code_object_valid_subarch else subarch_dir
        co_objects.append({"file_path": el, "arch": (arch, subarch)})
    return co_objects


def generate_struct_definition(args, dfs):
    fields = []
    master = dfs[0]
    for column in master.columns:
        if (
            column == "co_objects"
            or column == "co_name"
            or column == "knl_name"
            or column == "arch"
        ):
            continue
        if pd.api.types.is_integer_dtype(master[column]):
            min_int = min(df[column].min() if not df.empty else 0 for df in dfs)
            max_int = max(df[column].max() if not df.empty else 0 for df in dfs)

            mag_int = max(abs(min_int), abs(max_int)) * (-1 if min_int < 0 else 1)
            container_type = np.min_scalar_type(mag_int)
            bit_length = max(1, int(mag_int).bit_length() + int(mag_int < 0))
            fields.append(f"{container_type}_t {column}: {bit_length};")
            continue
        else:
            max_len = max(
                df[column].str.len().max() if not df.empty else 1 for df in dfs
            )

            min_len = min(
                df[column].str.len().min() if not df.empty else max_len for df in dfs
            )

            fields.append(f"FixedString<{min_len}, {max_len}> {column};")
            continue
    fields_str = "\n".join(fields)
    return f"""struct __attribute__((packed)) {args.module}Config {{
{fields_str}
}};
"""


def generate_code_object_data_with_len(co_objects):
    data = None
    debug_force_code_object_wrapper = True
    if (
        len(co_objects) == 1
        and co_objects[0]["arch"][0] in archs
        and not debug_force_code_object_wrapper
    ):
        data = np.fromfile(co_objects[0]["file_path"], dtype=np.uint8)
    else:
        object_map = dict()
        for co_object in co_objects:
            if co_object["arch"][0] in archs:
                key = code_object_arch_map.get(co_object["arch"])
                if key is None:
                    raise ValueError(
                        f"arch {co_object['arch']} not supported for code object wrapper"
                    )
                if key in object_map:
                    raise ValueError(f"duplicate key {key} found")
                object_map[key] = np.fromfile(co_object["file_path"], dtype=np.uint8)
        offsets = []
        current_offset = 0
        for key in code_object_arch_layout:
            if key in object_map:
                offsets.append(current_offset)
                current_offset += len(object_map[key])
            else:
                offsets.append(-1)
        data = np.concatenate(
            [
                np.array([code_object_wrapper_magic], dtype=np.uint8),
                np.array(offsets, dtype=np.int32).view(dtype=np.uint8),
                *(
                    object_map[key]
                    for key in code_object_arch_layout
                    if key in object_map
                ),
            ]
        )

    return len(data), ", ".join(map("0x{:02x}".format, data))


def generate_common():
    return "\n".join(
        [
            header,
            f"static_assert(AiterAsmKernelCodeObjectWrapper::magic == {code_object_wrapper_magic});",
            f"static_assert(sizeof(AiterAsmKernelCodeObjectWrapper) == {1 + 4 * len(code_object_arch_layout)});",
            *(
                f"static_assert(offsetof(AiterAsmKernelCodeObjectWrapper,{arch}_offset) == {1 + 4 * i});"
                for i, arch in enumerate(code_object_arch_layout)
            ),
            "\n",
        ]
    )


def generate_configs(args):
    content = ""

    csv_groups = defaultdict(list)
    for arch in archs_supported:
        for el in glob.glob(
            f"{this_dir}/{arch}/{args.module}/**/*.csv", recursive=True
        ):
            cfgname = os.path.basename(el).split(".")[0]
            csv_groups[cfgname].append({"file_path": el, "arch": arch})

    ## deal with same name csv
    cfgs = []
    for cfgname, file_info_list in csv_groups.items():
        dfs = []
        for file_info in file_info_list:
            single_file = file_info["file_path"]
            arch = file_info["arch"]
            df = pd.read_csv(single_file)
            # check headers
            headers_list = df.columns.tolist()
            required_columns = {"knl_name", "co_name"}
            if not required_columns.issubset(headers_list):
                missing = required_columns - set(headers_list)
                raise ValueError(
                    f"Invalid assembly CSV format -- {single_file}. Missing required columns: {', '.join(missing)}"
                )

            df["arch"] = arch  # add arch into df
            if arch in archs:
                df["co_objects"] = df.apply(
                    create_co_objects_column, axis=1, args=(args, arch)
                )
            else:
                df["co_objects"] = df.apply(lambda _: [], axis=1)
            dfs.append(df)

        df = pd.concat(dfs, ignore_index=True).fillna(0).infer_objects(copy=False)
        if cfgs:
            df = df[cfgs[0]["table"].columns]  # make sure all have same columns
        cfgs.append({"cfgname": cfgname, "table": df})

    if cfgs:
        dfs = [el["table"] for el in cfgs]
        content = generate_struct_definition(args, dfs)
        for df in dfs:
            df.drop(df[~df["arch"].isin(archs)].index, inplace=True)
        offsets = []
        kernel_data = []
        offset = 0
        for df in dfs:
            for _, knl_name, co_objects in df[["knl_name", "co_objects"]].itertuples():
                co_len, co_data = generate_code_object_data_with_len(co_objects)
                offsets.append(offset)
                kernel_data.append(f"0x{len(knl_name):02x}")
                kernel_data.append(
                    ", ".join(f"0x{b:02x}" for b in knl_name.encode("utf8"))
                )
                kernel_data.append("0x00")
                kernel_data.append(co_data)
                offset += len(knl_name) + 2 + co_len

        content += "\n".join(
            [
                f"constexpr uint8_t kernels_descriptors_{args.module}[] = {{{', '.join(kernel_data)}}};",
                f"constexpr uint32_t kernels_descriptor_offsets_{args.module}[] = {{{', '.join(str(off) for off in offsets)}}};",
                f"AiterAsmKernel<> kernels_{args.module}[{len(offsets)}] = {{}};",
                f"using CFG = AiterAsmKernelConfigMap<{args.module}Config, kernels_{args.module}, kernels_descriptor_offsets_{args.module}, kernels_descriptors_{args.module}>;\n",
            ]
        )

        tables = []
        bias = 0

        for cfg in cfgs:
            df = cfg["table"]

            offsets = []

            for arch in df["arch"].unique():
                mask = df["arch"] == arch
                first_idx = df.index[mask].min()
                last_idx = df.index[mask].max() + 1
                offsets.append((arch, first_idx, last_idx))

            if not df.empty:
                df["comment"] = df.apply(
                    lambda row: f"/* {row.name + bias} {row['arch']} {row['knl_name']} {row['co_name']} */",
                    axis=1,
                )
            df = df.drop(["co_objects", "knl_name", "co_name", "arch"], axis=1)

            for col in df.select_dtypes(exclude=["integer"]).columns:
                if col != "comment":
                    df[col] = '"' + df[col].astype(str) + '"'

            df = df.astype(str)

            comment = (
                f"/* {', '.join(col for col in df.columns if col != 'comment')} */"
            )

            cfgname = cfg["cfgname"]
            entries_str = ", \n".join(
                f"{{{', '.join(row)}}}" for row in df.itertuples(index=False)
            )
            tables.append(
                f"""constexpr AiterAsmKernelConfigMapSized<CFG, {len(df)}> cfg_{cfgname} = {{
{{.kernel_index_bias = {bias}, .entry_count = {len(df)}, .per_arch_offsets = {{{", ".join(f"[static_cast<int>(GPUArchId::{el[0]})] = {{{el[1]}, {el[2]}}}" for el in offsets)}}}}},
/*.entries = */{{
{comment}
{entries_str}}},
}};"""
            )
            bias += len(df)

        content += "\n".join(tables)
    with open(f"{args.output_dir}/asm_{args.module}_configs.hpp", "w") as f:
        f.write(generate_common())
        f.write(content)


def generate_code_objects(args):
    filters = (
        [f"{this_dir}/{{arch}}/{glob_pat}" for glob_pat in args.glob]
        if args.glob
        else [f"{this_dir}/{{arch}}/{args.module}/**/*.co"]
    )
    base = (
        f"{this_dir}/{{arch}}/" if args.glob else f"{this_dir}/{{arch}}/{args.module}/"
    )
    co_groups = defaultdict(list)
    for arch in archs_supported:
        base_dir = base.format(arch=arch)
        for el in set(
            chain.from_iterable(
                glob.glob(filter.format(arch=arch), recursive=True)
                for filter in filters
            )
        ):
            co_name = os.path.basename(el).split(".")[0]
            subarch_dir = os.path.relpath(os.path.dirname(el), base_dir)
            subarch = (
                "" if subarch_dir not in code_object_valid_subarch else subarch_dir
            )
            co_groups[co_name].append({"file_path": el, "arch": (arch, subarch)})

    content = "\n".join(
        [
            f"constexpr uint8_t {co_name}_co[] = {{{generate_code_object_data_with_len(co_objects)[1]}}};"
            for co_name, co_objects in co_groups.items()
        ]
    )

    with open(f"{args.output_dir}/asm_{args.module}_code_objects.hpp", "w") as f:
        f.write(generate_common())
        f.write(content)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="generate",
        description="gen API for asm kernel",
    )
    mode_group = parser.add_mutually_exclusive_group()

    mode_group.add_argument(
        "--configs",
        action="store_const",
        const="configs",
        dest="mode",
        default="configs",
        help="Generate config tables header file",
    )

    mode_group.add_argument(
        "--code-objects",
        action="store_const",
        const="code_objects",
        dest="mode",
        help="Generate kernel co header file",
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
    parser.add_argument(
        "--glob",
        required=False,
        action="append",
        help="glob match for objects when generating blobs header ",
    )
    args = parser.parse_args()
    if args.glob and args.mode != "code_objects":
        parser.error("Argument --glob is only allowed with --code-objects")

    if args.mode == "code_objects":
        generate_code_objects(args)
    else:
        generate_configs(args)
