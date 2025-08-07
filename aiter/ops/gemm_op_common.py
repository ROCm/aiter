# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

from ..jit.core import (
    compile_ops,
)
import torch


@compile_ops("module_gemm_common")
def get_padded_m(M: int, N: int, K: int, gl: int) -> int: ...


# import math

# def nextPow2(num: int) -> int:
#     if num <= 1:
#         return 1
#     return 1 << (num - 1).bit_length()

# def get_padded_m_py(M: int, N: int, K: int, gl: int) -> int:
#     padded_m = M
    
#     if gl == 0:
#         if M <= 256:
#             padded_m = (M + 15) // 16 * 16
#         elif M <= 1024:
#             padded_m = (M + 31) // 32 * 32
#         elif M <= 4096:
#             padded_m = (M + 63) // 64 * 64
#         else:
#             padded_m = (M + 127) // 128 * 128
    
#     elif gl == 1:
#         padded_m = nextPow2(M)
    
#     return padded_m
