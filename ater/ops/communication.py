# Copyright (c) 2024 Advanced Micro Devices, Inc.  All rights reserved.
#
# -*- coding:utf-8 -*-
# @Script: communication.py
# @Author: valarLip
# @Email: lingpeng.jin@amd.com
# @Create At: 2024-12-14 15:47:26
# @Last Modified By: valarLip
# @Last Modified At: 2024-12-14 16:41:01
# @Description: This is description.

import torch
from typing import List, Optional
from ..jit.core import compile_ops, CK_DIR, ATER_CSRC_DIR, ATER_ROOT_DIR
MD_NAME = 'module_communication'


@compile_ops(srcs=[f'{ATER_CSRC_DIR}/py_itfs_cu/asm_communication.cpp',
                   f'{ATER_CSRC_DIR}/pybind/communication_asm_pybind.cu'],
             flags_extra_hip=[f'-DATER_ASM_DIR=\\"{ATER_ROOT_DIR}/hsa/\\"'],
             md_name=f"{MD_NAME}_asm")
def commun_fwd_asm(
    out: torch.Tensor,
    input: torch.Tensor,
    val: int,
    op: int) -> None: ...
