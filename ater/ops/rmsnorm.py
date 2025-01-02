from torch import Tensor
from typing import List, Optional
from ..jit.core import compile_ops, CK_DIR, ATER_CSRC_DIR, get_argsOfBuild
import torch.nn.functional as F

MD_NAME = "module_rmsnorm"

compile_ops_ = get_argsOfBuild("module_rmsnorm")


@compile_ops(**compile_ops_)
def rms_norm(
    out: Tensor,
    input: Tensor,
    residual_in: Tensor,
    residual_out: Tensor,
    xscale: Tensor,
    yscale: Tensor,
    weight: Tensor,
    bias: Tensor,
    epsilon: float,
): ...


@compile_ops(**compile_ops_)
def fused_add_rms_norm(
    out: Tensor,
    input: Tensor,
    residual_in: Tensor,
    residual_out: Tensor,
    xscale: Tensor,
    yscale: Tensor,
    weight: Tensor,
    bias: Tensor,
    epsilon: float,
): ...
