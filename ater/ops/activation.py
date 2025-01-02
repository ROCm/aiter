from torch import Tensor
from typing import List, Optional
from ..jit.core import compile_ops, CK_DIR, ATER_CSRC_DIR, get_argsOfBuild
import torch.nn.functional as F

MD_NAME = "module_activation"

compile_ops_ = get_argsOfBuild("module_activation")


@compile_ops(**compile_ops_)
def silu_and_mul(out: Tensor, input: Tensor): ...


@compile_ops(**compile_ops_)
def gelu_and_mul(out: Tensor, input: Tensor): ...


@compile_ops(**compile_ops_)
def gelu_tanh_and_mul(out: Tensor, input: Tensor): ...
