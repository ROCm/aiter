from enum import Enum
from typing import List, Optional
from ..jit.core import compile_ops, CK_DIR, AITER_CSRC_DIR


@compile_ops("module_aiter_enum", "ActivationType")
def _ActivationType(dummy): ...


@compile_ops("module_aiter_enum", "QuantType")
def _QuantType(dummy): ...


ActivationType = type(_ActivationType(0))
QuantType = type(_QuantType(0))
