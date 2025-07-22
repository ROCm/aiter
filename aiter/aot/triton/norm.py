# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

from triton.tools.compile import compile_kernel, CompileArgs
from aiter.jit.core import AITER_ROOT_DIR


def compile_kernels():
    for BLOCK_SIZE in [32, 64, 128, 256]:
        compile_args = CompileArgs(
            path=f"{AITER_ROOT_DIR}/aiter/ops/triton/norm.py",
            kernel_name="_layernorm_kernel",
            signature=f"*fp16:16,*fp16:16,*fp16:16,*fp16:16,i32,i32,i32,i32,fp32,{BLOCK_SIZE}",
            grid="M,1,1",
            num_warps=4,
            num_stages=2,
            out_name="layernorm_fwd",
        )
        compile_kernel(compile_args)

        compile_args = CompileArgs(
            path=f"{AITER_ROOT_DIR}/aiter/ops/triton/norm.py",
            kernel_name="_fused_add_layernorm_kernel",
            signature=f"*fp16:16,*fp16:16,*fp16:16,*fp16:16,*fp16:16,*fp16:16,i32,i32,i32,i32,fp32,{BLOCK_SIZE}",
            grid="M,1,1",
            num_warps=4,
            num_stages=2,
            out_name="layernorm2d_fwd_with_add",
        )
        compile_kernel(compile_args)


if __name__ == "__main__":
    compile_kernels()
