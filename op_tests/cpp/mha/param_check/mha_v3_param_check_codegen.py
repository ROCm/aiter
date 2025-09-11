#!/usr/bin/env python3

import argparse
from pathlib import Path
import sys
import os
import re
from dataclasses import dataclass


@dataclass(frozen=True)
class CodegenInfo:
    version: str = "gfx942"  # arch version, support gfx942 and gfx950
    kernel_type: str = "fmha_v3_bwd"  # kernel type, support fmha_v3_fwd and fmha_v3_bwd
    func_name: str = (
        "fmha_bwd_v3("  # the param check function will be generated base on the original function name
    )
    target_start_line: int = (
        0  # the original function code snippets(start line) we will handle
    )
    target_end_line: int = (
        0  # the original function code snippets(end line) we will handle
    )
    output_file_name: str = (
        "mha_bwd_param_v3_check.cpp"  # which file the generated function will be writed
    )
    encoding: str = "utf-8"  # the file encoding type


codegen_info_list = [
    CodegenInfo(
        "gfx942",
        "fmha_v3_fwd",
        "fmha_fwd_v3(",
        226,
        394,
        "mha_fwd_param_v3_check.cpp",
    ),
    CodegenInfo(
        "gfx950",
        "fmha_v3_fwd",
        "fmha_fwd_v3(",
        181,
        245,
        "mha_fwd_param_v3_check.cpp",
    ),
    CodegenInfo(
        "gfx942",
        "fmha_v3_bwd",
        "fmha_bwd_v3(",
        842,
        2106,
        "mha_bwd_param_v3_check.cpp",
    ),
    CodegenInfo(
        "gfx950",
        "fmha_v3_bwd",
        "fmha_bwd_v3(",
        1218,
        2319,
        "mha_bwd_param_v3_check.cpp",
    ),
]


def write_param_check_func(work_path: Path, codegen_info: CodegenInfo) -> None:
    file_path = os.path.join(
        work_path, codegen_info.version, codegen_info.kernel_type, "codegen.py"
    )
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"file is not exist: {file_path}")

    with file_path.open(
        "r", encoding=codegen_info.encoding, errors="surrogateescape"
    ) as f:
        lines = f.readlines()

    lines = lines[codegen_info.target_start_line : codegen_info.target_end_line]
    target_func_name = f"{codegen_info.func_name[:-1]}_{codegen_info.version}_check("

    out_lines = []
    for line in lines:
        if "using " in line:
            continue
        new_line = line.replace(codegen_info.func_name, target_func_name)
        if (
            codegen_info.version == "gfx950"
            and codegen_info.kernel_type == "fmha_v3_bwd"
        ):
            new_line = re.sub(
                r"r\s*=\s*fmha_bwd_v3_genl_gfx950.*?(?=\n)", "r = 1;", new_line
            )
            if "r = fmha" in new_line:
                continue
        else:
            new_line = re.sub(r"r\s*=\s*fmha.*?(?=\n)", "r = 1;", new_line)
        out_lines.append(new_line)

    output_file_path = Path(codegen_info.output_file_name)
    with output_file_path.open("a", encoding=codegen_info.encoding, newline="") as f:
        f.writelines(out_lines)

    print(f"target_func:{target_func_name} write in {output_file_path} success!")
    return out_lines


def main():
    parser = argparse.ArgumentParser(
        description="delete the line which has '<', and replace the  'return r' -> 'is_v3=true'"
    )
    parser.add_argument(
        "--work_path",
        help="the input aiter work path to be process",
        default="../../../../hsa/",
    )

    args = parser.parse_args()
    work_path = Path(args.work_path)

    for codegen_info in codegen_info_list:
        try:
            write_param_check_func(work_path, codegen_info)
        except Exception as e:
            print(f"error: {e}", file=sys.stderr)
            sys.exit(1)


if __name__ == "__main__":
    main()
