# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

from triton.tools.compile import compile_kernel, CompileArgs
from aiter.jit.core import AITER_ROOT_DIR


def compile_kernels():
    compile_args = CompileArgs(
        path=f"{AITER_ROOT_DIR}/aiter/mla.py",
        kernel_name="_fwd_kernel_stage2_asm",
        signature="*fp32:16,*fp32:16,*bf16:16,*i32:16,i32,i32,i32,i32,i32,i32,i32,16,512,512,64",
        grid="bs,nheads,1",
        num_warps=4,
        num_stages=2,
        out_name="decode_mla_stage2_asm",
    )
    compile_kernel(compile_args)


if __name__ == "__main__":
    compile_kernels()
