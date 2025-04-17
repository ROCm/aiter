from enum import Enum
from torch import Tensor
from typing import List, Optional
from ..jit.core import compile_ops, CK_DIR, AITER_CSRC_DIR


@compile_ops("module_aiter_enum", "ActivationType")
def _ActivationType(dummy): ...


@compile_ops("module_aiter_enum", "QuantType")
def _QuantType(dummy): ...


ActivationType = _ActivationType(0)
QuantType = _QuantType(0)
