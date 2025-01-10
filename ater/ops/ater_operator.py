from torch import Tensor
from typing import List, Optional
from ..jit.core import compile_ops, CK_DIR, ATER_CSRC_DIR
import torch.nn.functional as F

MD_NAME = "module_ater_operator"

@compile_ops("module_ater_operator")
def add(input: Tensor, other: Tensor) -> Tensor: ...


@compile_ops("module_ater_operator")
def sub(input: Tensor, other: Tensor) -> Tensor: ...


@compile_ops("module_ater_operator")
def mul(input: Tensor, other: Tensor) -> Tensor: ...


@compile_ops("module_ater_operator")
def div(input: Tensor, other: Tensor) -> Tensor: ...
