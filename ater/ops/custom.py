from torch import Tensor
from typing import List, Optional
from ..jit.core import compile_ops, CK_DIR, ATER_CSRC_DIR, get_argsOfBuild
import torch.nn.functional as F

MD_NAME = "module_custom"

compile_ops_ = get_argsOfBuild("module_custom")


@compile_ops(**compile_ops_)
def wvSpltK(in_a: Tensor, in_b: Tensor, out_c: Tensor, N_in: int, CuCount: int): ...


@compile_ops(**compile_ops_)
def LLMM1(in_a: Tensor, in_b: Tensor, out_c: Tensor, rows_per_block: int): ...
