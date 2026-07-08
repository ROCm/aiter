# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
import sys
import os

# !!!!!!!!!!!!!!!! never import aiter
this_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, f"{this_dir}/../../../aiter/")
from jit.core import compile_ops, AITER_CSRC_DIR, AITER_CONFIGS  # noqa: E402


@compile_ops("libgemm_a4w4_blockscale", fc_name="compile_gemm_a4w4_blockscale")
def compile_gemm_a4w4_blockscale() -> None: ...


if __name__ == "__main__":
    compile_gemm_a4w4_blockscale()
