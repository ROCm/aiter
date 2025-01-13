from torch import Tensor
from typing import List, Optional
from ..jit.core import compile_ops, CK_DIR, ATER_CSRC_DIR, ATER_ROOT_DIR
import torch.nn.functional as F

MD_NAME = "module_custom_all_reduce"


@compile_ops("module_custom_all_reduce")
def init_custom_ar(
    out: Tensor, exp_sums: Tensor, handles, offsets, rank: int, full_nvlink: bool
) -> int: ...


@compile_ops("module_custom_all_reduce")
def all_reduce_reg(_fa: int, inp: Tensor, out: Tensor): ...


@compile_ops("module_custom_all_reduce")
def all_reduce_unreg(_fa: int, inp: Tensor,
                     reg_buffer: Tensor, out: Tensor): ...


@compile_ops("module_custom_all_reduce")
def all_reduce_asm_(inp: Tensor,
                    ca: int, reg_sig: Tensor, reg_buffer: Tensor, isGraph: bool) -> Tensor: ...


@compile_ops("module_custom_all_reduce")
def all_reduce_rmsnorm_(
    input: Tensor,
    residual_in: Tensor,
    weight: Tensor,
    bias: Tensor,
    epsilon: float,
    ca: int, reg_sig: Tensor, reg_buffer: Tensor, isGraph: bool) -> List[Tensor]: ...


@compile_ops("module_custom_all_reduce")
def all_reduce_rmsnorm_quant_(
    input: Tensor,
    residual_in: Tensor,
    weight: Tensor,
    xscale: Tensor,
    bias: Tensor,
    epsilon: float,
    ca: int, reg_sig: Tensor, reg_buffer: Tensor, isGraph: bool) -> List[Tensor]: ...


@compile_ops("module_custom_all_reduce")
def dispose(_fa: int, inp: Tensor, out: Tensor): ...


@compile_ops("module_custom_all_reduce")
def meta_size() -> int: ...


@compile_ops("module_custom_all_reduce")
def register_buffer(_fa: int, t: Tensor, handles, offsets): ...


@compile_ops("module_custom_all_reduce")
def get_graph_buffer_ipc_meta(_fa: int): ...


@compile_ops("module_custom_all_reduce")
def register_graph_buffers(_fa: int, handles, offsets): ...


@compile_ops("module_custom_all_reduce")
def allocate_meta_buffer(size: int) -> Tensor: ...


@compile_ops("module_custom_all_reduce")
def get_meta_buffer_ipc_handle(inp: Tensor) -> Tensor: ...
