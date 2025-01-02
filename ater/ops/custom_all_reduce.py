from torch import Tensor
from typing import List, Optional
from ..jit.core import compile_ops, CK_DIR, ATER_CSRC_DIR, ATER_ROOT_DIR, get_argsOfBuild
import torch.nn.functional as F

MD_NAME = "module_custom_all_reduce"

compile_ops_ = get_argsOfBuild("module_custom_all_reduce")


@compile_ops(**compile_ops_)
def init_custom_ar(
    out: Tensor, exp_sums: Tensor, handles, offsets, rank: int, full_nvlink: bool
) -> int: ...


@compile_ops(**compile_ops_)
def all_reduce_reg(_fa: int, inp: Tensor, out: Tensor): ...


@compile_ops(**compile_ops_)
def all_reduce_unreg(_fa: int, inp: Tensor,
                     reg_buffer: Tensor, out: Tensor): ...


@compile_ops(**compile_ops_)
def all_reduce_asm(_fa: int, inp: Tensor,
                   reg_buffer: Tensor, reg_sig: Tensor) -> Tensor: ...


@compile_ops(**compile_ops_)
def dispose(_fa: int, inp: Tensor, out: Tensor): ...


@compile_ops(**compile_ops_)
def meta_size() -> int: ...


@compile_ops(**compile_ops_)
def register_buffer(_fa: int, t: Tensor, handles, offsets): ...


@compile_ops(**compile_ops_)
def get_graph_buffer_ipc_meta(_fa: int): ...


@compile_ops(**compile_ops_)
def register_graph_buffers(_fa: int, handles, offsets): ...


@compile_ops(**compile_ops_)
def allocate_meta_buffer(size: int) -> Tensor: ...


@compile_ops(**compile_ops_)
def get_meta_buffer_ipc_handle(inp: Tensor) -> Tensor: ...
