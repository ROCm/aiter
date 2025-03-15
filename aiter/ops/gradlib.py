# SPDX-License-Identifier: MIT
# Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

import os
import torch
from torch import Tensor
from typing import List, Optional
from ..jit.core import compile_ops, CK_DIR, AITER_CSRC_DIR, AITER_ROOT_DIR, AITER_CORE_DIR, AITER_GRADLIB_DIR

@compile_ops("module_hipbsolgemm")
def hipb_create_extension(): ...

@compile_ops("module_hipbsolgemm")
def hipb_destroy_extension(): ...

@compile_ops("module_hipbsolgemm")
def hipb_mm(
    mat1:Tensor,
    mat2:Tensor,
    solution_index:int,
    bias:Optional[Tensor] = None,
    out_dtype:Optional[object]= None,
    scaleA:Optional[Tensor] = None,
    scaleB:Optional[Tensor] = None,
    scaleOut:Optional[Tensor] = None
): ...

@compile_ops("module_hipbsolgemm")
def hipb_findallsols(
    mat1:Tensor,
    mat2:Tensor,
    bias:Optional[Tensor] = None,
    out_dtype:Optional[object]= None,
    scaleA:Optional[Tensor] = None,
    scaleB:Optional[Tensor] = None,
    scaleC:Optional[Tensor] = None
): ...

@compile_ops("module_hipbsolgemm")
def getHipblasltKernelName(): ...



@compile_ops("module_rocsolgemm")
def rocb_create_extension(): ...

@compile_ops("module_rocsolgemm")
def rocb_destroy_extension(): ...

@compile_ops("module_rocsolgemm")
def rocb_mm(
    mat1:Tensor,
    mat2:Tensor,
    solution_index:int=0
): ...

@compile_ops("module_rocsolgemm")
def rocb_findallsols(
    mat1:Tensor,
    mat2:Tensor
): ...