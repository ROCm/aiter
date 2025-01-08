from torch import Tensor
from typing import List, Optional
from ..jit.core import compile_ops, CK_DIR, ATER_CSRC_DIR
import torch.nn.functional as F

MD_NAME = "module_ater_operator"

compile_ops_ = {
    "srcs": [
        f"{ATER_CSRC_DIR}/pybind/ater_operator_pybind.cu",
        f"{ATER_CSRC_DIR}/include/ater_operator.h.cu",
        f"{ATER_CSRC_DIR}/kernels/ater_operator.cu",
    ],
    "md_name": MD_NAME,
}


@compile_ops(**compile_ops_)
def ater_add(input: Tensor, other: Tensor) -> Tensor: ...


@compile_ops(**compile_ops_)
def ater_sub(input: Tensor, other: Tensor) -> Tensor: ...


@compile_ops(**compile_ops_)
def ater_mul(input: Tensor, other: Tensor) -> Tensor: ...


@compile_ops(**compile_ops_)
def ater_div(input: Tensor, other: Tensor) -> Tensor: ...
