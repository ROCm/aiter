"""
* Copyright (C) Advanced Micro Devices, Inc. All rights reserved.
* Copyright (C) 2024-2025, The vLLM team.
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

import time
import multiprocessing as mp
import gc

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


def get_dtype(dtype_str: str):
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


def runGemmTuner():
    gtuner = GemmTuner()
    ext_group = gtuner.parser.add_argument_group("extra parameters")
    ext_group.add_argument(
        "--model_dir",
        type=str,
        default=os.getenv("GTUNE_MODEL", ""),
        help="Enter the location of your model directory",
    )
    ext_group.add_argument(
        "--batch_size",
        type=int,
        default=os.getenv("GTUNE_BATCH_SIZE", 1),
        help="Batch size to tune for",
    )
    ext_group.add_argument(
        "--nsets",
        type=list_of_ints,
        default=[1, 512, 1024, 2048, 3072, 4096, 8192, 16384],
        help="N sizes to tune for: 1,128,2048",
    )
    ext_group.add_argument(
        "--tp",
        type=int,
        default=os.getenv("GTUNE_TP", 1),
        help="Tensor parallelism to be used.",
    )
    args = gtuner.parse_args()
    if args.outdtype is None:
        args.outdtype = args.indtype
    indtype = get_dtype(args.indtype)
    args.indtype = indtype
    outdtype = get_dtype(args.outdtype)
    args.outdtype = outdtype
    if not args.untune_file:
        nsets = [i * args.batch_size for i in args.nsets]
        if not args.model_dir:
            print(">>> Warning! NO MODEL SPECIFIED. Tuning for LL2 13B TP1")
            # LL2 13B sizes
            mksets = [(15360, 5120), (5120, 5120), (27648, 5120), (5120, 13824)]
            gtuner.add_gemm(m=32000, n=1, k=5120, indtype=indtype)  # logits gemm
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
        gtuner.untunedf.to_csv("./tmp_untuned.csv", index=False)
        args.untune_file = "./tmp_untuned.csv"
    gtuner.run(args)


def clean():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if hasattr(torch.cuda, "memory_allocated"):
        torch.cuda.synchronize()
    try:
        if hasattr(mp, "resource_tracker"):
            mp.resource_tracker.ensure_running()
            # clean  leaked semaphore objects
            if hasattr(mp.resource_tracker, "_CLEANUP_FUNCS"):
                # be careful
                for name in list(mp.resource_tracker._CLEANUP_FUNCS.keys()):
                    try:
                        mp.resource_tracker._CLEANUP_FUNCS.pop(name)()
                    except:
                        pass
    except Exception as e:
        print(f"Resource cleanup warning: {e}")


if __name__ == "__main__":
    retries = 0
    MAX_TRY = 30
    mp.set_start_method("spawn", force=True)
    while retries <= MAX_TRY:
        try:
            process = mp.Process(target=runGemmTuner, args=(), daemon=False)
            process.start()
            process.join()
            if process.exitcode > 1:
                time.sleep(0.5 * retries)
                print(
                    "!Error when run GemmTuner process exitcode is ", process.exitcode
                )
                clean()
                retries += 1
            else:
                break
        except Exception as e:
            print(f"Process creation failed: {e}")
            retries += 1
            clean()
            time.sleep(1)
        finally:
            if process and process.is_alive():
                process.terminate()
                process.join(timeout=5)

    clean()
    print(f"retried num is {retries}")
