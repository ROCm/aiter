# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import os
import tempfile
from jinja2 import Template
from csrc.cpp_itfs.utils import (
    compile_template_op,
    transfer_hsaco,
    AITER_CORE_DIR,
    GPU_ARCH,
    not_built,
    run_lib,
)
import csrc.cpp_itfs.utils as _utils_mod

MD_NAME  = "fmha_prefill_paged_asm"
HSACO    = f"fmha_prefill/f8_fmha_prefill_gfx942_hd128_qkptph_vph_paged_vkcolv_ps.co"
KNL_NAME = "_ZN5aiter25fmha_fwd_hd128_fp8_causalE"

with open(f"{AITER_CORE_DIR}/csrc/cpp_itfs/fmha_prefill_paged_asm.cpp.jinja", "r") as f:
    src_template = Template(f.read())


def compile(func_name: str = None):
    if func_name is None:
        func_name = MD_NAME

    if not_built(func_name):
        hsaco_path = f"{AITER_CORE_DIR}/hsa/{GPU_ARCH}/{HSACO}"
        bin_size, bin_data = transfer_hsaco(hsaco_path)

        # compile_lib unconditionally appends CK_DIR/include; for this pure-ASM
        # kernel CK is not needed, so redirect CK_DIR to a guaranteed-empty dir.
        _orig_ck_dir = _utils_mod.CK_DIR
        with tempfile.TemporaryDirectory() as _tmp:
            _utils_mod.CK_DIR = _tmp
            try:
                result = compile_template_op(
                    src_template,
                    MD_NAME,
                    [f"{AITER_CORE_DIR}/csrc/include"],
                    [],
                    bin_size=bin_size,
                    bin_data=bin_data,
                    kernel_name=KNL_NAME,
                    func_name=func_name,
                )
            finally:
                _utils_mod.CK_DIR = _orig_ck_dir
        return result
    else:
        return run_lib(func_name)
