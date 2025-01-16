import torch
from torch import Tensor
from typing import List, Optional
from ..jit.core import compile_ops, CK_DIR, ATER_CSRC_DIR
import torch.nn.functional as F

MD_NAME = "module_moe_sorting"



@compile_ops("module_moe_sorting")
def moe_sorting_fwd(input: Tensor, out: Tensor,
                    x_scale: Tensor, y_scale: Tensor): ...