# SPDX-License-Identifier: MIT
# Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.
from dataclasses import dataclass

@dataclass
class KernelParameters:
    BLOCK_SIZE: int
    MPerBLOCK: int
    NPerBLOCK: int
    KPerBLOCK: int
    WAVE_TILE_M: int
    WAVE_TILE_N: int
    WAVE_MAP_M: int
    WAVE_MAP_N: int
    ABLOCK_TRANSFER: list[int]
    BBLOCK_TRANSFER: list[int]
    CBLOCK_TRANSFER: list[int]
    CBLOCK_SPV: list[int]
    CSHUFFLE_MX_PER_WAVE_PERSHUFFLE: int
    CSHUFFLE_NX_PER_WAVE_PERSHUFFLE: int
    LOOP_SCHED: str
    PIPELINE_VERSION: int

    @property
    def name(self) -> str:
        return ("_").join([
            "a8w8_rowwise",
            ("x").join(map(lambda x: str(x), [
                self.BLOCK_SIZE, self.MPerBLOCK, self.NPerBLOCK, self.KPerBLOCK])),
            ("x").join(map(lambda x: str(x), [
                self.WAVE_TILE_M, self.WAVE_TILE_N])),
            ("x").join(map(lambda x: str(x), [
                self.WAVE_MAP_M, self.WAVE_MAP_N])),
            ("x").join(map(lambda x: str(x), self.ABLOCK_TRANSFER)),
            ("x").join(map(lambda x: str(x), self.BBLOCK_TRANSFER)),
            ("x").join(map(lambda x: str(x), self.CBLOCK_TRANSFER)),
            ("x").join(map(lambda x: str(x), self.CBLOCK_SPV)),
            ("x").join(map(lambda x: str(x), [
                self.CSHUFFLE_MX_PER_WAVE_PERSHUFFLE, self.CSHUFFLE_NX_PER_WAVE_PERSHUFFLE])),
            self.LOOP_SCHED.lower(),
            f"v{self.PIPELINE_VERSION}",

        ])


kernels_params_dict = {
#   id: kernel:        BLOCK_SIZE| MPerBLOCK| NPerBLOCK| KPerBLOCK| WAVE_TILE_M| WAVE_TILE_N| WAVE_MAP_M| WAVE_MAP_N| ABLOCK_TRANSFER| BBLOCK_TRANSFER| CBLOCK_TRANSFER| CBLOCK_SPV| CSHUFFLE_MX| CSHUFFLE_NX|   LOOP_SCHED| PIPELINE_VERSION
     0: KernelParameters(       256,       256,       256,        64,          32,          32,          4,          4,      [4, 64, 1],      [4, 64, 1],   [1, 32, 1, 8],  [8, 8, 1],           1,           1,  "Intrawave",    4),
     1: KernelParameters(       256,       256,       256,       128,          32,          32,          4,          4,      [8, 32, 1],      [8, 32, 1],   [1, 32, 1, 8],  [8, 8, 1],           1,           1,  "Intrawave",    3),
     2: KernelParameters(       256,       256,       224,       128,          32,          32,          2,          7,      [8, 32, 1],      [8, 32, 1],   [1, 64, 1, 4],  [8, 8, 1],           2,           1,  "Intrawave",    3),
     3: KernelParameters(       256,       256,       192,       128,          32,          32,          4,          3,      [8, 32, 1],      [8, 32, 1],   [1, 32, 1, 8],  [8, 8, 1],           1,           1,  "Intrawave",    3),
     4: KernelParameters(       256,       256,       160,       128,          32,          32,          2,          5,      [8, 32, 1],      [8, 32, 1],   [1, 64, 1, 4],  [8, 8, 1],           2,           1,  "Intrawave",    3),
     5: KernelParameters(       256,       256,       128,       128,          32,          32,          4,          2,      [8, 32, 1],      [8, 32, 1],   [1, 32, 1, 8],  [8, 8, 1],           1,           1,  "Intrawave",    3),
     6: KernelParameters(       256,       256,        96,       128,          32,          32,          2,          3,      [8, 32, 1],      [8, 32, 1],   [1, 64, 1, 4],  [8, 8, 1],           2,           1,  "Intrawave",    3),
     7: KernelParameters(       256,       256,        64,       128,          32,          32,          4,          1,      [8, 32, 1],      [8, 32, 1],   [1, 32, 1, 8],  [8, 8, 1],           1,           1,  "Intrawave",    3),
     8: KernelParameters(       256,       128,       256,       128,          32,          32,          2,          4,      [8, 32, 1],      [8, 32, 1],   [1, 32, 1, 8],  [8, 8, 1],           1,           1,  "Intrawave",    3),
     9: KernelParameters(       256,       128,       224,       128,          32,          32,          1,          7,      [8, 32, 1],      [8, 32, 1],   [1, 64, 1, 4],  [8, 8, 1],           1,           1,  "Intrawave",    3),
    10: KernelParameters(       256,       128,       192,       128,          32,          32,          2,          3,      [8, 32, 1],      [8, 32, 1],   [1, 32, 1, 8],  [8, 8, 1],           1,           1,  "Intrawave",    3),
    11: KernelParameters(       256,       128,       160,       128,          32,          32,          1,          5,      [8, 32, 1],      [8, 32, 1],   [1, 64, 1, 4],  [8, 8, 1],           1,           1,  "Intrawave",    3),
    12: KernelParameters(       256,       128,       128,       256,          32,          32,          2,          2,      [16, 16, 1],     [16, 16, 1],  [1, 32, 1, 8],  [8, 8, 1],           1,           1,  "Intrawave",    3),
    13: KernelParameters(       256,       128,       128,       128,          32,          32,          2,          2,      [8, 32, 1],      [8, 32, 1],   [1, 32, 1, 8],  [8, 8, 1],           1,           1,  "Intrawave",    3),
    14: KernelParameters(       256,       128,        96,       256,          32,          32,          1,          3,      [16, 16, 1],     [16, 16, 1],  [1, 64, 1, 4],  [8, 8, 1],           1,           1,  "Intrawave",    3),
    15: KernelParameters(       256,       128,        64,       256,          32,          32,          2,          1,      [16, 16, 1],     [16, 16, 1],  [1, 32, 1, 8],  [8, 8, 1],           1,           1,  "Intrawave",    3),
    16: KernelParameters(       256,        64,       256,       128,          32,          32,          1,          4,      [8, 32, 1],      [8, 32, 1],   [1, 32, 1, 8],  [8, 8, 1],           1,           1,  "Intrawave",    3),
    17: KernelParameters(       256,        64,       224,       128,          16,          16,          2,          7,      [8, 32, 1],      [8, 32, 1],   [1, 64, 1, 4],  [8, 8, 1],           2,           1,  "Intrawave",    3),
    18: KernelParameters(       256,        64,       192,       256,          32,          32,          1,          3,      [16, 16, 1],     [16, 16, 1],  [1, 32, 1, 8],  [8, 8, 1],           1,           1,  "Intrawave",    3),
    19: KernelParameters(       256,        64,       192,       128,          32,          32,          1,          3,      [8, 32, 1],      [8, 32, 1],   [1, 32, 1, 8],  [8, 8, 1],           1,           1,  "Intrawave",    3),
    20: KernelParameters(       256,        64,       160,       256,          16,          16,          2,          5,      [16, 16, 1],     [16, 16, 1],  [1, 64, 1, 4],  [8, 8, 1],           2,           1,  "Intrawave",    3),
    21: KernelParameters(       256,        64,       128,       256,          32,          32,          1,          2,      [16, 16, 1],     [16, 16, 1],  [1, 32, 1, 8],  [8, 8, 1],           1,           1,  "Intrawave",    3),
    22: KernelParameters(       256,        64,        96,       256,          16,          16,          2,          3,      [16, 16, 1],     [16, 16, 1],  [1, 64, 1, 4],  [8, 8, 1],           2,           1,  "Intrawave",    3),
    23: KernelParameters(       256,        64,        64,       512,          32,          32,          1,          1,      [32, 8, 1],      [32, 8, 1],   [1, 32, 1, 8],  [8, 8, 1],           1,           1,  "Intrawave",    3),
    24: KernelParameters(       256,        32,       256,       128,          32,          32,          1,          2,      [8, 32, 1],      [8, 32, 1],   [1, 32, 1, 8],  [8, 8, 1],           1,           1,  "Intrawave",    3),
    25: KernelParameters(       256,        32,       224,       256,          16,          16,          1,          7,      [16, 16, 1],     [16, 16, 1],  [1, 32, 1, 8],  [4, 4, 1],           1,           1,  "Intrawave",    3),
    26: KernelParameters(       256,        32,       192,       256,          16,          16,          1,          6,      [16, 16, 1],     [16, 16, 1],  [1, 32, 1, 8],  [8, 8, 1],           1,           2,  "Intrawave",    3),
    27: KernelParameters(       256,        32,       160,       256,          16,          16,          1,          5,      [16, 16, 1],     [16, 16, 1],  [1, 32, 1, 8],  [4, 4, 1],           1,           1,  "Intrawave",    3),
    28: KernelParameters(       256,        32,       128,       256,          32,          32,          1,          1,      [16, 16, 1],     [16, 16, 1],  [1, 32, 1, 8],  [8, 8, 1],           1,           1,  "Intrawave",    3),
    29: KernelParameters(       256,        32,        96,       256,          16,          16,          1,          3,      [16, 16, 1],     [16, 16, 1],  [1, 32, 1, 8],  [4, 4, 1],           1,           1,  "Intrawave",    3),
    30: KernelParameters(       256,        32,        64,       512,          16,          16,          1,          2,      [32, 8, 1],      [32, 8, 1],   [1, 32, 1, 8],  [8, 8, 1],           1,           2,  "Intrawave",    3),
    31: KernelParameters(       256,        16,       256,       128,          16,          16,          1,          4,      [16, 16, 1],     [8, 32, 1],   [1, 16, 1, 16], [8, 8, 1],           1,           2,  "Intrawave",    3),
    32: KernelParameters(       256,        16,       192,       256,          16,          16,          1,          3,      [16, 16, 1],     [16, 16, 1],  [1, 16, 1, 16], [4, 4, 1],           1,           1,  "Intrawave",    3),
    33: KernelParameters(       256,        16,       128,       256,          16,          16,          1,          2,      [16, 16, 1],     [16, 16, 1],  [1, 16, 1, 16], [8, 8, 1],           1,           2,  "Intrawave",    3),
    34: KernelParameters(       256,        16,        64,       512,          16,          16,          1,          1,      [32, 8, 1],      [32, 8, 1],   [1, 16, 1, 16], [4, 4, 1],           1,           1,  "Intrawave",    3),
    35: KernelParameters(       256,       128,       128,       128,          32,          32,          2,          2,      [8, 32, 1],      [8, 32, 1],   [1, 32, 1, 8],  [8, 8, 1],           1,           1,  "Intrawave",    4),
    36: KernelParameters(       256,       128,       128,        64,          32,          32,          2,          2,      [4, 64, 1],      [4, 64, 1],   [1, 32, 1, 8],  [8, 8, 1],           1,           1,  "Intrawave",    4),
    37: KernelParameters(       256,       256,       256,       128,          16,          16,          8,          8,      [8, 32, 1],      [8, 32, 1],   [1, 32, 1, 8],  [8, 8, 1],           1,           2,  "Intrawave",    3),
    38: KernelParameters(       256,       256,       256,        64,          16,          16,          8,          8,      [4, 64, 1],      [4, 64, 1],   [1, 32, 1, 8],  [8, 8, 1],           1,           2,  "Intrawave",    3),
    39: KernelParameters(       256,       224,       256,       128,          16,          16,          7,          8,      [8, 32, 1],      [8, 32, 1],   [1, 32, 1, 8],  [8, 8, 1],           1,           2,  "Intrawave",    3),
    40: KernelParameters(       256,       256,       224,       128,          16,          16,          8,          7,      [8, 32, 1],      [8, 32, 1],   [1, 64, 1, 4],  [8, 8, 1],           2,           1,  "Intrawave",    3),
    41: KernelParameters(       256,       128,       128,       128,          32,          32,          2,          2,      [8, 32, 1],      [8, 32, 1],   [1, 32, 1, 8],  [8, 8, 1],           1,           1,  "Intrawave",    5),
    42: KernelParameters(       256,       128,       256,        64,          32,          32,          2,          4,      [4, 64, 1],      [4, 64, 1],   [1, 32, 1, 8],  [8, 8, 1],           1,           1,  "Interwave",    1),
    43: KernelParameters(       256,       256,       128,        64,          32,          32,          4,          2,      [4, 64, 1],      [4, 64, 1],   [1, 32, 1, 8],  [8, 8, 1],           1,           1,  "Interwave",    1),
    44: KernelParameters(       256,       128,       128,       128,          32,          32,          2,          2,      [8, 32, 1],      [8, 32, 1],   [1, 32, 1, 8],  [8, 8, 1],           1,           1,  "Interwave",    1),
    45: KernelParameters(       256,       128,        64,       128,          32,          32,          2,          1,      [8, 32, 1],      [8, 32, 1],   [1, 32, 1, 8],  [8, 8, 1],           1,           1,  "Intrawave",    3),
    46: KernelParameters(       256,        64,       128,       128,          32,          32,          1,          2,      [8, 32, 1],      [8, 32, 1],   [1, 32, 1, 8],  [8, 8, 1],           1,           1,  "Intrawave",    3),
    47: KernelParameters(       256,        64,        64,       128,          32,          32,          1,          1,      [8, 32, 1],      [8, 32, 1],   [1, 32, 1, 8],  [8, 8, 1],           1,           1,  "Intrawave",    3),
# mem(Intrawave): Latency friendly 
    48: KernelParameters(       128,        32,        16,       128,          16,          16,          1,          1,      [8, 16, 1],      [8, 16, 1],   [1, 16, 1, 8],  [2, 2, 1],           1,           1,  "Intrawave",    1),
    49: KernelParameters(        64,        16,        16,       128,          16,          16,          1,          1,      [8,  8, 1],      [8,  8, 1],   [1, 16, 1, 4],  [4, 4, 1],           1,           1,  "Intrawave",    1),
    50: KernelParameters(       128,        16,        32,       128,          16,          16,          1,          1,      [8, 16, 1],      [8, 16, 1],   [1, 16, 1, 8],  [4, 4, 1],           1,           1,  "Intrawave",    1),
# mem(Intrawave): Memory friendly, Col    
    51: KernelParameters(       256,       256,        32,       128,          32,          32,          2,          1,      [8, 32, 1],      [8, 32, 1],   [1, 32, 1, 8],  [4, 4, 1],           1,           1,  "Intrawave",    2),
    52: KernelParameters(       256,       256,        16,       128,          16,          16,          4,          1,      [8, 32, 1],      [8, 16, 1],   [1, 32, 1, 8],  [2, 2, 1],           1,           1,  "Intrawave",    2),
    53: KernelParameters(       128,       128,        32,       128,          32,          32,          2,          1,      [8, 16, 1],      [8, 16, 1],   [1, 16, 1, 8],  [4, 4, 1],           1,           1,  "Intrawave",    2),
    54: KernelParameters(       128,       128,        16,       128,          16,          16,          4,          1,      [8, 16, 1],      [8, 16, 1],   [1, 16, 1, 8],  [2, 2, 1],           1,           1,  "Intrawave",    2),
    55: KernelParameters(       128,        64,        32,       128,          32,          32,          1,          1,      [8, 16, 1],      [8, 16, 1],   [1, 16, 1, 8],  [4, 4, 1],           1,           1,  "Intrawave",    2),
    56: KernelParameters(       128,        64,        16,       128,          16,          16,          2,          1,      [8, 16, 1],      [8, 16, 1],   [1, 16, 1, 8],  [2, 2, 1],           1,           1,  "Intrawave",    2),
    57: KernelParameters(       128,        32,        16,       128,          16,          16,          1,          1,      [8, 16, 1],      [8, 16, 1],   [1, 16, 1, 8],  [2, 2, 1],           1,           1,  "Intrawave",    2),
    58: KernelParameters(        64,        16,        16,        64,          16,          16,          1,          1,      [4, 16, 1],      [4, 16, 1],   [1, 16, 1, 4],  [4, 4, 1],           1,           1,  "Intrawave",    2),
    59: KernelParameters(        64,        16,        16,       128,          16,          16,          1,          1,      [8,  8, 1],      [8,  8, 1],   [1, 16, 1, 4],  [4, 4, 1],           1,           1,  "Intrawave",    2),
    60: KernelParameters(       128,        16,        32,       128,          16,          16,          1,          1,      [8, 16, 1],      [8, 16, 1],   [1, 16, 1, 8],  [4, 4, 1],           1,           1,  "Intrawave",    2),
    61: KernelParameters(       128,        16,        64,       128,          16,          16,          1,          2,      [8, 16, 1],      [8, 16, 1],   [1, 16, 1, 8],  [4, 4, 1],           1,           1,  "Intrawave",    2),
    62: KernelParameters(       128,        32,        64,       128,          32,          32,          1,          1,      [8, 16, 1],      [8, 16, 1],   [1, 16, 1, 8],  [8, 8, 1],           1,           1,  "Intrawave",    2),
    63: KernelParameters(       128,        16,       128,       128,          16,          16,          1,          4,      [8, 16, 1],      [8, 16, 1],   [1, 16, 1, 8],  [4, 4, 1],           1,           1,  "Interwave",    2),
    64: KernelParameters(       128,        32,       128,       128,          32,          32,          1,          2,      [8, 16, 1],      [8, 16, 1],   [1, 16, 1, 8],  [8, 8, 1],           1,           1,  "Interwave",    2),
    65: KernelParameters(       256,        16,       256,       128,          16,          16,          1,          4,      [8, 16, 1],      [8, 16, 1],   [1, 16, 1, 16], [4, 4, 1],           1,           1,  "Interwave",    2),
    66: KernelParameters(       256,        32,       256,       128,          32,          32,          1,          2,      [8, 32, 1],      [8, 32, 1],   [1, 16, 1, 16], [8, 8, 1],           1,           1,  "Intrawave",    2),
# mem(Interwave): Latency friendly 
    67: KernelParameters(       128,        32,        16,       128,          16,          16,          1,          1,      [8, 16, 1],      [8, 16, 1],   [1, 16, 1, 8],  [2, 2, 1],           1,           1,  "Interwave",    1),
    68: KernelParameters(        64,        16,        16,       128,          16,          16,          1,          1,      [8,  8, 1],      [8,  8, 1],   [1, 16, 1, 4],  [4, 4, 1],           1,           1,  "Interwave",    1),
    69: KernelParameters(       128,        16,        32,       128,          16,          16,          1,          1,      [8, 16, 1],      [8, 16, 1],   [1, 16, 1, 8],  [4, 4, 1],           1,           1,  "Interwave",    1),
# mem(Interwave): Memory friendly, Col    
    70: KernelParameters(       256,       256,        32,       128,          32,          32,          2,          1,      [8, 32, 1],      [8, 32, 1],   [1, 32, 1, 8],  [4, 4, 1],           1,           1,  "Interwave",    2),
    71: KernelParameters(       256,       256,        16,       128,          16,          16,          4,          1,      [8, 32, 1],      [8, 16, 1],   [1, 32, 1, 8],  [2, 2, 1],           1,           1,  "Interwave",    2),
    72: KernelParameters(       128,       128,        32,       128,          32,          32,          2,          1,      [8, 16, 1],      [8, 16, 1],   [1, 16, 1, 8],  [4, 4, 1],           1,           1,  "Interwave",    2),
    73: KernelParameters(       128,       128,        16,       128,          16,          16,          4,          1,      [8, 16, 1],      [8, 16, 1],   [1, 16, 1, 8],  [2, 2, 1],           1,           1,  "Interwave",    2),
    74: KernelParameters(       128,        64,        32,       128,          32,          32,          1,          1,      [8, 16, 1],      [8, 16, 1],   [1, 16, 1, 8],  [4, 4, 1],           1,           1,  "Interwave",    2),
    75: KernelParameters(       128,        64,        16,       128,          16,          16,          2,          1,      [8, 16, 1],      [8, 16, 1],   [1, 16, 1, 8],  [2, 2, 1],           1,           1,  "Interwave",    2),
    76: KernelParameters(       128,        32,        16,       128,          16,          16,          1,          1,      [8, 16, 1],      [8, 16, 1],   [1, 16, 1, 8],  [2, 2, 1],           1,           1,  "Interwave",    2),
    77: KernelParameters(        64,        16,        16,        64,          16,          16,          1,          1,      [4, 16, 1],      [4, 16, 1],   [1, 16, 1, 4],  [4, 4, 1],           1,           1,  "Interwave",    2),
    78: KernelParameters(        64,        16,        16,       128,          16,          16,          1,          1,      [8,  8, 1],      [8,  8, 1],   [1, 16, 1, 4],  [4, 4, 1],           1,           1,  "Interwave",    2),
    79: KernelParameters(       128,        16,        32,       128,          16,          16,          1,          1,      [8, 16, 1],      [8, 16, 1],   [1, 16, 1, 8],  [4, 4, 1],           1,           1,  "Interwave",    2),
    80: KernelParameters(       128,        16,        64,       128,          16,          16,          1,          2,      [8, 16, 1],      [8, 16, 1],   [1, 16, 1, 8],  [4, 4, 1],           1,           1,  "Interwave",    2),
    81: KernelParameters(       128,        32,        64,       128,          32,          32,          1,          1,      [8, 16, 1],      [8, 16, 1],   [1, 16, 1, 8],  [8, 8, 1],           1,           1,  "Interwave",    2),
    82: KernelParameters(       128,        16,       128,       128,          16,          16,          1,          4,      [8, 16, 1],      [8, 16, 1],   [1, 16, 1, 8],  [4, 4, 1],           1,           1,  "Interwave",    2),
    83: KernelParameters(       128,        32,       128,       128,          32,          32,          1,          2,      [8, 16, 1],      [8, 16, 1],   [1, 16, 1, 8],  [8, 8, 1],           1,           1,  "Interwave",    2),
    84: KernelParameters(       256,        16,       256,       128,          16,          16,          1,          4,      [8, 16, 1],      [8, 16, 1],   [1, 16, 1, 16], [4, 4, 1],           1,           1,  "Interwave",    2),
    85: KernelParameters(       256,        32,       256,       128,          32,          32,          1,          2,      [8, 32, 1],      [8, 32, 1],   [1, 16, 1, 16], [8, 8, 1],           1,           1,  "Interwave",    2),
}


default_kernels_dict = {
#   (    M,     N,     K): kernel:        BLOCK_SIZE| MPerBLOCK| NPerBLOCK| KPerBLOCK| WAVE_TILE_M| WAVE_TILE_N| WAVE_MAP_M| WAVE_MAP_N| ABLOCK_TRANSFER| BBLOCK_TRANSFER| CBLOCK_TRANSFER| CBLOCK_SPV| CSHUFFLE_MX| CSHUFFLE_NX|  LOOP_SCHED|PIPELINE_VERSION          
    (-1):                  KernelParameters(        64,        16,        16,       128,           16,         16,          1,          1,      [8, 8,  1],      [8, 8,  1],   [1, 16, 1, 4],  [4, 4, 1],           1,           1, "Interwave",  2),
    (-3):                  KernelParameters(       128,        32,        16,       128,           16,         16,          1,          1,      [8, 16, 1],      [8, 16, 1],   [1, 16, 1, 8],  [2, 2, 1],           1,           1, "Interwave",  2),
    (-4):                  KernelParameters(        64,        16,        16,       256,           16,         16,          1,          1,      [16, 4, 1],      [16, 4, 1],   [1, 16, 1, 4],  [4, 4, 1],           1,           1, "Intrawave",  1),
    (-5):                  KernelParameters(       128,        16,        32,       128,           16,         16,          1,          1,      [8, 16, 1],      [8, 16, 1],   [1, 16, 1, 8],  [2, 2, 1],           1,           1, "Intrawave",  2),
    (-6):                  KernelParameters(       256,       128,       128,       128,           32,         32,          2,          2,      [8, 32, 1],      [8, 32, 1],   [1, 32, 1, 8],  [8, 8, 1],           1,           1, "Interwave",  1),
    (-7):                  KernelParameters(       256,       128,       128,       128,           32,         32,          2,          2,      [8, 32, 1],      [8, 32, 1],   [1, 32, 1, 8],  [8, 8, 1],           1,           1, "Intrawave",  3),
    (-8):                  KernelParameters(       256,       256,       128,        64,           32,         32,          4,          2,      [4, 64, 1],      [4, 64, 1],   [1, 32, 1, 8],  [8, 8, 1],           1,           1, "Interwave",  1),
    (-9):                  KernelParameters(       256,       224,       256,       128,           16,         16,          7,          8,      [8, 32, 1],      [8, 32, 1],   [1, 32, 1, 8],  [8, 8, 1],           1,           2, "Intrawave",  3),
    (-10):                 KernelParameters(       128,        16,        32,       128,           16,         16,          1,          1,      [8, 16, 1],      [8, 16, 1],   [1, 16, 1, 8],  [4, 4, 1],           1,           1, "Intrawave",  2),

}