from ..jit.core import compile_ops

# from enum import Enum as Enum
Enum = int


@compile_ops("module_aiter_enum", "ActivationType")
def _ActivationType(dummy): ...


@compile_ops("module_aiter_enum", "QuantType")
def _QuantType(dummy): ...


ActivationType = type(_ActivationType(0))
QuantType = type(_QuantType(0))

class FP8KVCacheLayout():
    V32_FP8Sparse = 1
    MODEL1_FP8Sparse = 2

    def get_meta(self):
        # Return: (d, d_nope, d_rope, tile_size, num_tiles)
        return {
            FP8KVCacheLayout.V32_FP8Sparse: (576, 512, 64, 128, 4),
            FP8KVCacheLayout.MODEL1_FP8Sparse: (512, 448, 64, 64, 7)
        }[self]