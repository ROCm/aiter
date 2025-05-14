"""
* Copyright © Advanced Micro Devices, Inc. All rights reserved.
* Copyright (c) 2024, The vLLM team.
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*      http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
"""

import argparse
import json
import os
from pathlib import Path

import torch  # isort: split
import aiter
from aiter import dtypes
import pandas as pd

from GemmTuner import GemmTuner

aiter.rocb_create_extension()
aiter.hipb_create_extension()


def generate_mk_sets(model_dir, tp=1):
    with open(f"{model_dir}/config.json") as f:
        data = json.load(f)
        hidden_size = data["hidden_size"]
        intermediate_size = data["intermediate_size"]
        total_num_heads = data["num_attention_heads"]
        total_num_kv_heads = data["num_key_value_heads"]
        dtype = get_dtype(data["torch_dtype"])
        head_dim = hidden_size // total_num_heads
    return (
        [
            (
                (total_num_heads + (2 * total_num_kv_heads)) * head_dim // tp,
                hidden_size,
            ),
            (hidden_size, hidden_size // tp),
            (intermediate_size * 2 // tp, hidden_size),
            (hidden_size, intermediate_size // tp),
        ],
        hidden_size,
        dtype,
    )


dtypes = {
    "f32": dtypes.fp32,
    "float32": dtypes.fp32,
    "f16": dtypes.fp16,
    "float16": dtypes.fp16,
    "bf16": dtypes.bf16,
    "bfloat16": dtypes.bf16,
    "fp8": dtypes.fp8,
}


def get_dtype(dtype_str):
    if dtype_str is None:
        return None
    if dtype_str.startswith("torch"):
        return getattr(torch, dtype_str.split(".")[1])
    if dtype_str in dtypes:
        return dtypes[dtype_str]
    else:
        print(">>> Warning! Invalid dtype", dtype_str, "using default dtype f16")
    return None


def list_of_ints(arg):
    return list(map(int, arg.split(",")))


def load_input_gemms(input_file):
    if Path(input_file).is_file():
        return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_dir",
        type=str,
        default=os.getenv("GTUNE_MODEL", ""),
        help="Enter the location of your model directory",
    )
    parser.add_argument(
        "--tuned_file",
        type=str,
        default=os.getenv("GTUNE_TUNED", "tuned.csv"),
        help="output file for tuned gemm solutions",
    )
    parser.add_argument(
        "--input_file",
        type=str,
        default=os.getenv("GTUNE_INPUT", None),
        help="list of gemms to tune for, mutually exclusive with model_dir",
    )
    parser.add_argument(
        "--tp",
        type=int,
        default=os.getenv("GTUNE_TP", 1),
        help="Tensor parallelism to be used.",
    )
    parser.add_argument(
        "--indtype",
        type=str,
        default=None,
        choices=["f32", "f16", "bf16", "fp8"],
        help="dtype: f32 f16 bf16 fp8. Use this to override the"
        " input_file or if no input_file provided",
    )
    parser.add_argument(
        "--outdtype",
        type=str,
        choices=["f32", "f16", "bf16", "fp8"],
        help="dtype: f32 f16 bf16 fp8. Use to override the default value,"
        " which is the same as indtype for each shape (see --indtype.)",
    )
    parser.add_argument(
        "--rocblas-decode",
        action="store_true",
        default=False,
        help="forces rocblas solution on decode N=1",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=os.getenv("GTUNE_BATCH_SIZE", 1),
        help="Batch size to tune for",
    )
    parser.add_argument(
        "--nsets",
        type=list_of_ints,
        default=[1, 512, 1024, 2048, 3072, 4096, 8192, 16384],
        help="N sizes to tune for: 1,128,2048",
    )
    parser.add_argument(
        "--all_bias",
        action="store_true",
        help="Tune for both bias and non bias cases,"
        " regardless of what was used"
        " to collect the shapes",
    )
    args = parser.parse_args()

    if args.outdtype is None:
        args.outdtype = args.indtype
    indtype = get_dtype(args.indtype)
    outdtype = get_dtype(args.outdtype)

    gtuner = GemmTuner(indtype, outdtype, args.tuned_file, args.rocblas_decode)
    nsets = [i * args.batch_size for i in args.nsets]
    if args.input_file:
        print(f">>> Loading {args.input_file}")
        if not Path(args.input_file).is_file():
            print(f">>> ERROR: {args.input_file} does not exist.  Exiting")
            exit(1)
        shapes = pd.read_csv(args.input_file).fillna("")
        if "outdtype" not in shapes.columns:
            shapes["outdtype"] = ""
        if "scaleAB" not in shapes.columns:
            shapes["scaleAB"] = False
        for i in range(len(shapes)):
            ds = shapes.iloc[i]
            for bias in [True, False] if args.all_bias else [ds["bias"]]:
                gtuner.add_gemm(
                    ds["M"],
                    ds["N"],
                    ds["K"],
                    indtype=get_dtype(ds["dtype"]),
                    bias=bias,
                    outdtype=get_dtype(ds["outdtype"]),
                    scaleAB=ds["scaleAB"],
                )
    else:
        if not args.model_dir:
            print(">>> Warning! NO MODEL SPECIFIED. Tuning for LL2 13B TP1")
            # LL2 13B sizes
            mksets = [(15360, 5120), (5120, 5120), (27648, 5120), (5120, 13824)]
            gtuner.add_gemm(m=32000, n=1, k=5120)  # logits gemm
            dtype = dtypes.fp16
        else:
            mksets, hidden_size, dtype = generate_mk_sets(args.model_dir, args.tp)
            gtuner.add_gemm(
                m=32000 // args.tp,
                n=1 * args.batch_size,
                k=hidden_size,
                indtype=dtype,
            )  # TODO: Handle cases where vocab_size is not divisible by tp

        for n in sorted(nsets):
            for m, k in mksets:
                gtuner.add_gemm(m, n, k, indtype=dtype)

    gtuner.find_best_sols()
