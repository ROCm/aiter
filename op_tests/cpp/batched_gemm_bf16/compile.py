# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# Build the torch-free libbatched_gemm_bf16.so. Mirrors op_tests/cpp/mha/compile.py:
# torch_exclude=True + is_python_module=False produces a plain C-ABI shared
# library with no libtorch / libc10 linkage, consumed by the C++ test harness.
import sys
import os

# !!!!!!!!!!!!!!!! never import aiter
this_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, f"{this_dir}/../../../aiter/")
from jit.core import compile_ops, AITER_CSRC_DIR  # noqa: E402

BGEMM_DIR = f"{AITER_CSRC_DIR}/cpp_itfs/batched_gemm_bf16"


def cmdGenFunc_batched_gemm_bf16():
    return {
        "srcs": [f"{BGEMM_DIR}/batched_gemm_bf16.cu"],
        "md_name": "libbatched_gemm_bf16",
        "blob_gen_cmd": [],
        "flags_extra_cc": ["-DUSE_ROCM=1 -DENABLE_CK=1"],
        "extra_include": [BGEMM_DIR],
        "torch_exclude": True,
        "is_python_module": False,
    }


@compile_ops(
    "libbatched_gemm_bf16",
    fc_name="compile_batched_gemm_bf16",
    gen_func=cmdGenFunc_batched_gemm_bf16,
)
def compile_batched_gemm_bf16(): ...


if __name__ == "__main__":
    print("######## building torch-free libbatched_gemm_bf16")
    compile_batched_gemm_bf16()
