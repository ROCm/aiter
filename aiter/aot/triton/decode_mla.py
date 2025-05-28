# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

from aiter.aot.triton.compile import compile_kernel
from aiter.jit.core import AITER_ROOT_DIR


def compile_kernels():
    compile_kernel(
        f"{AITER_ROOT_DIR}/aiter/ops/triton/decode_mla.py",
        "_fwd_kernel_stage2_asm",
        "*fp32:16,*fp32:16,*bf16:16,*i32:16,i32,i32,i32,i32,i32,i32,i32,16,512,512,64",
        "bs,nheads,1",
        4,
        2,
        "decode_mla_stage2_asm",
        waves_per_eu=4,
        kpack=2,
        matrix_instr_nonkdim=16,
    )


if __name__ == "__main__":
    compile_kernels()
