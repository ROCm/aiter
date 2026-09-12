# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

from typing import List

from ..jit.core import compile_ops

MD_NAME = "module_custom_all_reduce_ifoe"


@compile_ops("module_custom_all_reduce_ifoe", develop=True)
def ifoe_alloc_fabric(bytes: int, handle_out_ptr: int) -> int: ...


@compile_ops("module_custom_all_reduce_ifoe", develop=True)
def ifoe_import_fabric(handle_ptr: int, bytes: int) -> int: ...


@compile_ops("module_custom_all_reduce_ifoe", develop=True)
def ifoe_init(
    rank: int,
    world: int,
    self_input_ptr: int,
    self_signal_ptr: int,
    self_bf_ptr: int,
    peer_input_ptrs: List[int],
    peer_signal_ptrs: List[int],
    peer_bf_ptrs: List[int],
) -> int: ...


@compile_ops("module_custom_all_reduce_ifoe", develop=True)
def ifoe_all_reduce(
    ctx: int,
    inp_ptr: int,
    out_ptr: int,
    numel: int,
    elt_size: int,
    mode: int,
    unroll: int,
    blocks: int,
) -> None: ...


@compile_ops("module_custom_all_reduce_ifoe", develop=True)
def ifoe_meta_size() -> int: ...


@compile_ops("module_custom_all_reduce_ifoe", develop=True)
def ifoe_dispose(ctx: int) -> None: ...
