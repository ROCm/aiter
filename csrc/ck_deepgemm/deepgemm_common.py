# SPDX-License-Identifier: MIT
# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
from dataclasses import dataclass


@dataclass
class kernelInstance:
    BLOCK_SIZE: int
    # GroupCount: int
    MPerBLOCK: int
    NPerBLOCK: int
    KPerBLOCK: int
    WAVE_TILE_M: int
    WAVE_TILE_N: int
    WAVE_TILE_K: int
    WAVE_MAP_M: int
    WAVE_MAP_N: int

    @property
    def name(self) -> str:
        return ("_").join(
            [
                "deepgemm",
                ("x").join(
                    map(
                        lambda x: str(x),
                        [
                            self.BLOCK_SIZE,
                            self.MPerBLOCK,
                            self.NPerBLOCK,
                            self.KPerBLOCK,
                        ],
                    )
                ),
                ("x").join(
                    map(
                        lambda x: str(x),
                        [self.WAVE_TILE_M, self.WAVE_TILE_N, self.WAVE_TILE_K],
                    )
                ),
                ("x").join(map(lambda x: str(x), [self.WAVE_MAP_M, self.WAVE_MAP_N])),
            ]
        )


# fmt: off
kernels_list = {
#   (    M,     N,     K): kernel:        BLOCK_SIZE| MPerBLOCK| NPerBLOCK| KPerBLOCK| WAVE_TILE_M| WAVE_TILE_N| WAVE_TILE_K| WAVE_MAP_M| WAVE_MAP_N|  LOOP_SCHED|PIPELINE_VERSION
    1:                     kernelInstance(       256,       128,       128,       128,           16,         16,          64,         1,           4),
    2:                     kernelInstance(       256,       128,       128,       128,           16,         16,          32,         1,           4),
    3:                     kernelInstance(       256,        32,        64,       256,           16,         16,          64,         1,           4),
    4:                     kernelInstance(       256,        32,        64,       256,           16,         16,          32,         1,           4),
}


default_kernels_dict = {
#   (    M,     N,     K): kernel:        BLOCK_SIZE| MPerBLOCK| NPerBLOCK| KPerBLOCK| WAVE_TILE_M| WAVE_TILE_N| WAVE_MAP_M| WAVE_MAP_N| ABLOCK_TRANSFER| BBLOCK_TRANSFER| CBLOCK_TRANSFER| CBLOCK_SPV| CSHUFFLE_MX| CSHUFFLE_NX|  LOOP_SCHED|PIPELINE_VERSION
    (-1):                     kernelInstance(       256,       128,       128,       128,           16,         16,          64,         1,           4),
    (-2):                     kernelInstance(       256,       128,       128,       128,           16,         16,          32,         1,           4),
    (-3):                     kernelInstance(       256,       32,       64,       256,           16,         16,          64,         1,           4),
    (-4):                     kernelInstance(       256,       32,       64,       256,           16,         16,          32,         1,           4),

}
# fmt: on
