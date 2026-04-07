from ..jit.core import compile_ops

# from enum import Enum as Enum
Enum = int


@compile_ops("module_aiter_core", "ActivationType")
def _ActivationType(dummy): ...


@compile_ops("module_aiter_core", "QuantType")
def _QuantType(dummy): ...


ActivationType = type(_ActivationType(0))
QuantType = type(_QuantType(0))
