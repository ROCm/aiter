# Copyright (c) 2024 Advanced Micro Devices, Inc.  All rights reserved.
#
# -*- coding:utf-8 -*-
# @Script: norm.py
# @Author: valarLip
# @Email: lingpeng.jin@amd.com
# @Create At: 2024-11-29 16:30:04
# @Last Modified By: valarLip
# @Last Modified At: 2025-01-01 15:24:38
# @Description: This is description.

from torch import Tensor
from typing import List, Optional
from ..jit.core import compile_ops, CK_DIR, ATER_CSRC_DIR, ATER_ROOT_DIR
import torch.nn.functional as F

MD_NAME = "module_norm"

compile_ops_ = {
    "srcs": [
        f"{ATER_CSRC_DIR}/py_itfs_ck/norm_kernels.cu",
        f"{ATER_CSRC_DIR}/py_itfs_cu/asm_layernorm.cpp",
        f"{ATER_CSRC_DIR}/pybind/norm_pybind.cu",
    ],
    "flags_extra_hip": [f'-DATER_ASM_DIR=\\"{ATER_ROOT_DIR}/hsa/\\"'],
    "extra_include": [f"{CK_DIR}/example/ck_tile/02_layernorm2d"],
    "blob_gen_cmd": f"{CK_DIR}/example/ck_tile/02_layernorm2d/generate.py --api fwd --gen_blobs --working_path {{}}",
    "md_name": MD_NAME,
}


@compile_ops(fc_name="layernorm2d_fwd", **compile_ops_)
def layer_norm(
    input: Tensor,
    # normalized_shape: List[int],
    weight: Optional[Tensor] = None,
    bias: Optional[Tensor] = None,
    eps: float = 1e-5,
) -> Tensor: ...


@compile_ops(fc_name="layernorm2d_fwd", **compile_ops_)
def layernorm2d_fwd(
    input: Tensor,
    # normalized_shape: List[int],
    weight: Optional[Tensor] = None,
    bias: Optional[Tensor] = None,
    eps: float = 1e-5,
) -> Tensor: ...


@compile_ops(**compile_ops_)
def layernorm2d_fwd_with_add(
    out: Tensor,
    input: Tensor,
    residual_in: Tensor,
    residual_out: Tensor,
    weight: Tensor,
    bias: Tensor,
    epsilon: float,
): ...


@compile_ops(**compile_ops_)
def layernorm2d_fwd_with_smoothquant(
    out: Tensor,
    input: Tensor,
    xscale: Tensor,
    yscale: Tensor,
    weight: Tensor,
    bias: Tensor,
    epsilon: float,
): ...


@compile_ops(**compile_ops_)
def layernorm2d_fwd_with_add_smoothquant(
    out: Tensor,
    input: Tensor,
    residual_in: Tensor,
    residual_out: Tensor,
    xscale: Tensor,
    yscale: Tensor,
    weight: Tensor,
    bias: Tensor,
    epsilon: float,
): ...


# @compile_ops(**compile_ops_)
# def layernorm2d_fwd_with_dynamicquant(
#     out: Tensor,
#     input: Tensor,
#     yscale: Tensor,
#     weight: Tensor,
#     bias: Tensor,
#     epsilon: float):...

# @compile_ops(**compile_ops_)
# def layernorm2d_fwd_with_add_dynamicquant(
#     out: Tensor,
#     input: Tensor,
#     residual_in: Tensor,
#     residual_out: Tensor,
#     yscale: Tensor,
#     weight: Tensor,
#     bias: Tensor,
#     epsilon: float):...
@compile_ops(**compile_ops_)
def layernorm2d_with_add_asm(
    out: Tensor,
    input: Tensor,
    residual_in: Tensor,
    residual_out: Tensor,
    weight: Tensor,
    bias: Tensor,
    epsilon: float,
): ...
