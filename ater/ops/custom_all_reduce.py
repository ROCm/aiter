from torch import Tensor
from typing import List, Optional
from ..jit.core import compile_ops, CK_DIR, ATER_CSRC_DIR, ATER_ROOT_DIR
import torch.nn.functional as F

MD_NAME = "module_custom_all_reduce"

compile_ops_ = {
    "srcs": [
        f"{ATER_CSRC_DIR}/pybind/custom_all_reduce_pybind.cu",
        f"{ATER_CSRC_DIR}/kernels/custom_all_reduce.cu",
        f'{ATER_CSRC_DIR}/py_itfs_cu/asm_communication.cpp',
    ],
    'flags_extra_hip': [f'-DATER_ASM_DIR=\\"{ATER_ROOT_DIR}/hsa/\\"'],
    "md_name": MD_NAME,
}


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
def all_reduce_asm_(inp: Tensor,
                    ca: int, reg_sig: Tensor, reg_buffer: Tensor, isGraph: bool) -> Tensor: ...


@compile_ops(**compile_ops_)
def all_reduce_layernorm_(
    out: Tensor,
    input: Tensor,
    residual_in: Tensor,
    residual_out: Tensor,
    weight: Tensor,
    bias: Tensor,
    epsilon: float,
    ca: int, reg_sig: Tensor, reg_buffer: Tensor, isGraph: bool) -> Tensor: ...


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
