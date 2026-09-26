# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Compatibility exports for the architecture-neutral MoE tile helpers."""

from .moe_gemm_2stage_utils import (
    FlyObjCache,
    MoETileOps,
    _as_ptr,
    all_copy_atoms,
    all_elements,
    asm_mark,
    atom_tensor,
    atomic_add_bf16,
    div_e,
    div_up,
    dump_ir,
    get_d1_shape,
    inner_most_stride,
    load_fragment,
    make_1d_coord_tensor,
    split_works,
    sub_tensor,
    torch_layout,
    view_as_torch_tensor,
)

__all__ = [
    "FlyObjCache",
    "MoETileOps",
    "_as_ptr",
    "all_copy_atoms",
    "all_elements",
    "asm_mark",
    "atom_tensor",
    "atomic_add_bf16",
    "div_e",
    "div_up",
    "dump_ir",
    "get_d1_shape",
    "inner_most_stride",
    "load_fragment",
    "make_1d_coord_tensor",
    "split_works",
    "sub_tensor",
    "torch_layout",
    "view_as_torch_tensor",
]
