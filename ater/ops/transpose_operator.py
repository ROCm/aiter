from torch import Tensor
from typing import List, Optional
from ..jit.core import compile_ops, CK_DIR, ATER_CSRC_DIR, get_argsOfBuild
import torch.nn.functional as F

MD_NAME = "module_transpose_operator"

compile_ops_ = get_argsOfBuild("module_transpose_operator")


@compile_ops(**compile_ops_)
def transpose_add(input: Tensor, other: Tensor) -> Tensor: ...


@compile_ops(**compile_ops_)
def transpose_sub(input: Tensor, other: Tensor) -> Tensor: ...


@compile_ops(**compile_ops_)
def transpose_mul(input: Tensor, other: Tensor) -> Tensor: ...


@compile_ops(**compile_ops_)
def transpose_div(input: Tensor, other: Tensor) -> Tensor: ...
