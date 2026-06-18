# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
import argparse
import os
import sys
import shutil
from pathlib import Path

import pandas as pd

this_dir = os.path.dirname(os.path.abspath(__file__))
AITER_CORE_DIR = (
    os.path.join(os.path.abspath(f"{this_dir}/../../../"), "aiter/jit/utils")
    if os.path.exists(
        os.path.join(os.path.abspath(f"{this_dir}/../../../"), "aiter_meta")
    )
    else os.path.abspath(f"{this_dir}/../../aiter/jit/utils")
)
sys.path.insert(0, AITER_CORE_DIR)
from chip_info import build_tune_dict, write_lookup_header, gen_lookup_header_map  # noqa: E402

from gemm_a8w8_common import (  # noqa: E402
    default_kernels_dict,
    kernelInstance,
    kernels_list,
)
from template_env import stream as stream_template  # noqa: E402


class gemm_a8w8_fwd_codegen:
    def __init__(self, working_path, istune=False):
        self.working_path = working_path
        self.impl_path = os.path.join(working_path, "impl")
        self.instances_path = os.path.join(working_path, "instances")
        self.istune = istune

    def gen_instance(self, k: kernelInstance):
        stream_template(
            "impl.cuh.j2",
            os.path.join(self.impl_path, f"{k.name}.cuh"),
            k=k,
            istune=self.istune,
        )

        if self.istune:
            # Generate both I8 and F8 instances for tuning
            # I8 instances
            for EDtype in ["B16"]:
                stream_template(
                    "instance.cpp.j2",
                    os.path.join(
                        self.instances_path, f"{k.name}_abI8_dB16_e{EDtype}.cpp"
                    ),
                    name=k.name,
                    dtypes=f"I8, B16, {EDtype}",
                )

            # F8 instances
            for EDtype in ["B16"]:
                stream_template(
                    "instance.cpp.j2",
                    os.path.join(
                        self.instances_path, f"{k.name}_abF8_dF32_e{EDtype}.cpp"
                    ),
                    name=k.name,
                    dtypes=f"F8, F32, {EDtype}",
                )
        else:
            for EDtype in ["B16", "F16"]:
                for ABDtype in ["I8", "F8"]:
                    for DDtype in ["F32", EDtype]:
                        stream_template(
                            "instance.cpp.j2",
                            os.path.join(
                                self.instances_path,
                                f"{k.name}_ab{ABDtype}_d{DDtype}_e{EDtype}.cpp",
                            ),
                            name=k.name,
                            dtypes=f"{ABDtype}, {DDtype}, {EDtype}",
                        )

    def gen_lookup_dict(self, kernels_dict):
        kernels_dict = gen_lookup_header_map(kernels_dict, self.istune)
        stream_template(
            "lookup.cuh.j2",
            os.path.join(self.working_path, "gemm_a8w8_lookup.h"),
            kernels_dict=kernels_dict,
        )

    def gen_manifest_head(self, kernels_dict):
        kernel_names = [k.name for k in kernels_dict.values()]

        stream_template(
            "gemm_a8w8_manifest.h.j2",
            os.path.join(self.working_path, "gemm_a8w8_manifest.h"),
            kernel_names=kernel_names,
        )

    def gen_instances(self, kernels_dict):
        if os.path.exists(self.impl_path):
            shutil.rmtree(self.impl_path)
        os.mkdir(self.impl_path)
        if os.path.exists(self.instances_path):
            shutil.rmtree(self.instances_path)
        os.mkdir(self.instances_path)

        for mnk, k in kernels_dict.items():
            self.gen_instance(k)

        self.gen_lookup_dict(kernels_dict)
        self.gen_manifest_head(kernels_dict)


def get_tune_dict(tune_dict_csv):
    if os.path.exists(tune_dict_csv):
        return build_tune_dict(
            pd.read_csv(tune_dict_csv), default_kernels_dict, kernels_list
        )
    return default_kernels_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="generate",
        description="gen API for CK gemm a8w8 kernel",
    )

    # the directory for list_blobs/gen_blobs to write files into
    parser.add_argument(
        "-w",
        "--working_path",
        default="./",
        required=False,
        help="the path where all the blobs are going to be generated",
    )

    parser.add_argument(
        "-f",
        "--tune_file",
        default="aiter/configs/a8w8_tuned_gemm.csv",
        required=False,
        help="tune_file include the result after run gemm_a8w8_tune.py",
    )

    parser.add_argument(
        "--tune", action="store_true", required=False, help="generated tune instances"
    )

    args = parser.parse_args()
    codegen = gemm_a8w8_fwd_codegen(args.working_path, args.tune)

    if args.tune:
        codegen.gen_instances(kernels_list)
    else:
        codegen.gen_instances(get_tune_dict(args.tune_file))
