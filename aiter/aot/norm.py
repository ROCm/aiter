# SPDX-License-Identifier: MIT
# Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

from aiter.aot.triton_compile import compile_kernel
from aiter.jit.core import AITER_ROOT_DIR


def compile_kernels():
    for BLOCK_SIZE in [32, 64, 128, 256]:
        compile_kernel(f"{AITER_ROOT_DIR}/aiter/ops/triton/norm.py", "_layernorm_kernel", f"*fp16:16,*fp16:16,*fp16:16,*fp16:16,i32,i32,i32,i32,fp32,{BLOCK_SIZE}", "M,1,1", 4, 2, "layer_norm")
        compile_kernel(f"{AITER_ROOT_DIR}/aiter/ops/triton/norm.py", "_fused_add_layernorm_kernel", f"*fp16:16,*fp16:16,*fp16:16,*fp16:16,*fp16:16,*fp16:16,i32,i32,i32,i32,fp32,{BLOCK_SIZE}", "M,1,1", 4, 2, "layernorm2d_fwd_with_add")


if __name__ == "__main__":
    compile_kernels()