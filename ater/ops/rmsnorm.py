from torch import Tensor
from typing import List, Optional
from ..jit.core import compile_ops, CK_DIR, ATER_CSRC_DIR
import torch.nn.functional as F

MD_NAME = "module_rmsnorm"



@compile_ops("module_rmsnorm")
def rms_norm_cu(
    out: Tensor,
    input: Tensor,
    weight: Tensor,
    epsilon: float,
): 
    '''
    Cuda version of rmsnorm
    '''
    ...


@compile_ops("module_rmsnorm")
def fused_add_rms_norm_cu(
    input: Tensor,        # input/out 
    residual_in: Tensor,  # residual_in/out 
    weight: Tensor,
    epsilon: float,
): 
    '''
    Cuda version of rmsnorm fused add
    '''
    ...


@compile_ops("module_rmsnorm", fc_name="rmsnorm2d_fwd")
def rms_norm(
    input: Tensor,
    weight: Tensor,
    epsilon: float,
): 
    '''
    CK version of rmsnorm
    '''
    ...

@compile_ops("module_rmsnorm")
def rmsnorm2d_fwd(
    input: Tensor,
    # normalized_shape: List[int],
    weight: Optional[Tensor] = None,
    bias: Optional[Tensor] = None,
    eps: float = 1e-5,
) -> Tensor: ...


@compile_ops("module_rmsnorm")
def rmsnorm2d_fwd_with_add(
    out: Tensor,
    input: Tensor,
    residual_in: Tensor,
    residual_out: Tensor,
    weight: Tensor,
    epsilon: float,
): ...


@compile_ops("module_rmsnorm")
def rmsnorm2d_fwd_with_smoothquant(
    out: Tensor,
    input: Tensor,
    xscale: Tensor,
    yscale: Tensor,
    weight: Tensor,
    epsilon: float,
): ...


@compile_ops("module_rmsnorm")
def rmsnorm2d_fwd_with_add_smoothquant(
    out: Tensor,
    input: Tensor,
    residual_in: Tensor,
    residual_out: Tensor,
    xscale: Tensor,
    yscale: Tensor,
    weight: Tensor,
    epsilon: float,
): ...


# @compile_ops("module_rmsnorm")
# def rmsnorm2d_fwd_with_dynamicquant(
#     out: Tensor,
#     input: Tensor,
#     yscale: Tensor,
#     weight: Tensor,
#     epsilon: float):...


# @compile_ops("module_rmsnorm")
# def rmsnorm2d_fwd_with_add_dynamicquant(
#     out: Tensor,
#     input: Tensor,
#     residual_in: Tensor,
#     residual_out: Tensor,
#     yscale: Tensor,
#     weight: Tensor,
#     epsilon: float):...