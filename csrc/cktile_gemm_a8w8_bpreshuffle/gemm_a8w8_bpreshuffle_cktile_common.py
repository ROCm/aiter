# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
from dataclasses import dataclass
import os
import sys

this_dir = os.path.dirname(os.path.abspath(__file__))
AITER_CORE_DIR = os.path.abspath(f"{this_dir}/../../../")
if os.path.exists(os.path.join(AITER_CORE_DIR, "aiter_meta")):
    AITER_CORE_DIR = os.path.join(AITER_CORE_DIR, "aiter/jit/utils")  # pip install mode
else:
    AITER_CORE_DIR = os.path.abspath(
        f"{this_dir}/../../aiter/jit/utils"
    )  # develop mode
sys.path.insert(0, AITER_CORE_DIR)

from chip_info import get_gfx  # noqa: E402


@dataclass
class kernelInstance:
    sTransposeC: bool
    sUseStructuredSparsity: bool
    sTileParitionerGroupNum: int
    sTileParitionerM01: int
    sNumWaveGroups: int
    sDoubleSmemBuffer: bool
    PadM: bool
    PadN: bool
    PadK: bool
    BlockPerCu: int
    MTile: int
    NTile: int
    KTile: int
    MWarp: int
    NWarp: int
    KWarp: int
    MWTile: int
    NWTile: int
    KWTile: int
    sScheduler: str

    @property
    def name(self) -> str:
        return ("_").join(
            [
                "a8w8_bpreshuffle_cktile",
                ("x").join(
                    map(
                        lambda x: str(x),
                        [
                            self.sTransposeC,
                            self.sUseStructuredSparsity,
                            self.sTileParitionerGroupNum,
                            self.sTileParitionerM01,
                            self.sNumWaveGroups,
                            self.sDoubleSmemBuffer,
                            self.PadM,
                            self.PadN,
                            self.PadK,
                            self.BlockPerCu,
                        ],
                    )
                ),
                ("x").join(map(lambda x: str(x), [self.MTile, self.NTile, self.KTile])),
                ("x").join(map(lambda x: str(x), [self.MWarp, self.NWarp, self.KWarp])),
                ("x").join(
                    map(lambda x: str(x), [self.MWTile, self.NWTile, self.KWTile])
                ),
                self.sScheduler.lower(),
            ]
        )


# fmt: off
# kernels_list_str = '''
kernels_list_942 = {
    0: kernelInstance( 0, 0, 8, 4, 1, 0, 0, 0, 0, 1,   128,   128,   128,  1,  4,  1,   16,    16,    64, "Default"),
    1: kernelInstance( 0, 0, 8, 4, 1, 0, 0, 0, 0, 1,   128,   128,   256,  1,  4,  1,   16,    16,    64, "Default"),
    2: kernelInstance( 0, 0, 8, 4, 1, 0, 0, 0, 0, 1,   16,    64,    512,  1,  4,  1,   16,    16,    64, "Default"),
    3: kernelInstance( 0, 0, 8, 4, 1, 0, 0, 0, 0, 1,   16,    128,   512,  1,  4,  1,   16,    16,    64, "Default"),
    4: kernelInstance( 0, 0, 8, 4, 1, 0, 0, 0, 0, 1,   16,    256,   512,  1,  4,  1,   16,    16,    64, "Default"),
    5: kernelInstance( 0, 0, 8, 4, 1, 0, 0, 0, 0, 1,   16,    64,    256,  1,  4,  1,   16,    16,    64, "Default"),
    6: kernelInstance( 0, 0, 8, 4, 1, 0, 0, 0, 0, 1,   16,    128,   256,  1,  4,  1,   16,    16,    64, "Default"),
    7: kernelInstance( 0, 0, 8, 4, 1, 0, 0, 0, 0, 1,   16,    256,   256,  1,  4,  1,   16,    16,    64, "Default"),
    8: kernelInstance( 0, 0, 8, 4, 1, 0, 0, 0, 0, 1,   16,    512,   256,  1,  4,  1,   16,    16,    64, "Default"),
    9: kernelInstance( 0, 0, 8, 4, 1, 0, 0, 0, 0, 1,   32,    64,    512,  1,  4,  1,   16,    16,    64, "Default"),
    10: kernelInstance( 0, 0, 8, 4, 1, 0, 0, 0, 0, 1,  64,    256,   64,   1,  4,  1,   16,    16,    64, "Default"),
    11: kernelInstance( 0, 0, 8, 4, 1, 0, 0, 0, 0, 1,  128,   64,   128,   1,  4,  1,   16,    16,    64, "Default"),
    12: kernelInstance( 0, 0, 8, 4, 1, 0, 0, 0, 0, 1,  128,   64,   256,   1,  4,  1,   16,    16,    64, "Default"),
    13: kernelInstance( 0, 0, 8, 4, 1, 0, 0, 0, 0, 1,  128,   128,   64,   1,  4,  1,   16,    16,    64, "Default"),
    14: kernelInstance( 0, 0, 8, 4, 1, 0, 0, 0, 0, 1,  128,   256,   128,  1,  4,  1,   16,    16,    64, "Default"),
    15: kernelInstance( 0, 0, 8, 4, 1, 0, 0, 0, 0, 1,   32,   64,    128,  1,  4,  1,   16,    16,    64, "Default"),
    16: kernelInstance( 0, 0, 8, 4, 1, 0, 0, 0, 0, 1,   64,   64,    128,  1,  4,  1,   16,    16,    64, "Default"),
    17: kernelInstance( 0, 0, 8, 4, 1, 0, 0, 0, 0, 1,   32,   128,   128,  1,  4,  1,   16,    16,    64, "Default"),
    18: kernelInstance( 0, 0, 8, 4, 1, 0, 0, 0, 0, 1,   64,   128,   128,  1,  4,  1,   16,    16,    64, "Default"),
    19: kernelInstance( 0, 0, 8, 4, 1, 0, 0, 0, 0, 1,   32,   256,   128,  1,  4,  1,   16,    16,    64, "Default"),
    20: kernelInstance( 0, 0, 8, 4, 1, 0, 0, 0, 0, 1,   64,   256,   128,  1,  4,  1,   16,    16,    64, "Default"),
    21: kernelInstance( 0, 0, 8, 4, 1, 0, 0, 0, 0, 1,   32,    64,   256,  1,  4,  1,   16,    16,    64, "Default"),
    22: kernelInstance( 0, 0, 8, 4, 1, 0, 0, 0, 0, 1,   64,    64,   256,  1,  4,  1,   16,    16,    64, "Default"),
    23: kernelInstance( 0, 0, 8, 4, 1, 0, 0, 0, 0, 1,   32,   128,   256,  1,  4,  1,   16,    16,    64, "Default"),
    24: kernelInstance( 0, 0, 8, 4, 1, 0, 0, 0, 0, 1,   64,   128,   256,  1,  4,  1,   16,    16,    64, "Default"),
    25: kernelInstance( 0, 0, 8, 4, 1, 0, 0, 0, 0, 1,   32,   192,   128,  1,  4,  1,   16,    16,    64, "Default"),
    26: kernelInstance( 0, 0, 8, 4, 1, 0, 0, 0, 0, 1,   64,   192,   128,  1,  4,  1,   16,    16,    64, "Default"),
    27: kernelInstance( 0, 0, 8, 4, 1, 0, 0, 0, 0, 1,   128,  192,   128,  1,  4,  1,   16,    16,    64, "Default"),
    28: kernelInstance( 0, 0, 8, 4, 1, 0, 0, 0, 0, 2,   128,  128,   128,  1,  4,  1,   16,    16,    64, "Default"),
    29: kernelInstance( 0, 0, 8, 4, 1, 0, 0, 0, 0, 2,   128,  128,   256,  1,  4,  1,   16,    16,    64, "Default"),
    30: kernelInstance( 0, 0, 8, 4, 1, 0, 0, 0, 0, 2,   16,   64,    512,  1,  4,  1,   16,    16,    64, "Default"),
    31: kernelInstance( 0, 0, 8, 4, 1, 0, 0, 0, 0, 2,   16,   128,   512,  1,  4,  1,   16,    16,    64, "Default"),
    32: kernelInstance( 0, 0, 8, 4, 1, 0, 0, 0, 0, 2,   16,   256,   512,  1,  4,  1,   16,    16,    64, "Default"),
    33: kernelInstance( 0, 0, 8, 4, 1, 0, 0, 0, 0, 2,   16,   64,    256,  1,  4,  1,   16,    16,    64, "Default"),
    34: kernelInstance( 0, 0, 8, 4, 1, 0, 0, 0, 0, 2,   16,   128,   256,  1,  4,  1,   16,    16,    64, "Default"),
    35: kernelInstance( 0, 0, 8, 4, 1, 0, 0, 0, 0, 2,   16,   256,   256,  1,  4,  1,   16,    16,    64, "Default"),
    36: kernelInstance( 0, 0, 8, 4, 1, 0, 0, 0, 0, 2,   16,   512,   256,  1,  4,  1,   16,    16,    64, "Default"),
    37: kernelInstance( 0, 0, 8, 4, 1, 0, 0, 0, 0, 2,   32,   64,    512,  1,  4,  1,   16,    16,    64, "Default"),
    38: kernelInstance( 0, 0, 8, 4, 1, 0, 0, 0, 0, 2,  64,    256,   64,   1,  4,  1,   16,    16,    64, "Default"),
    39: kernelInstance( 0, 0, 8, 4, 1, 0, 0, 0, 0, 2,  128,   64,   128,   1,  4,  1,   16,    16,    64, "Default"),
    40: kernelInstance( 0, 0, 8, 4, 1, 0, 0, 0, 0, 2,  128,   64,   256,   1,  4,  1,   16,    16,    64, "Default"),
    41: kernelInstance( 0, 0, 8, 4, 1, 0, 0, 0, 0, 2,  128,   128,   64,   1,  4,  1,   16,    16,    64, "Default"),
    42: kernelInstance( 0, 0, 8, 4, 1, 0, 0, 0, 0, 2,  128,   256,   128,  1,  4,  1,   16,    16,    64, "Default"),
    43: kernelInstance( 0, 0, 8, 4, 1, 0, 0, 0, 0, 2,   32,   64,    128,  1,  4,  1,   16,    16,    64, "Default"),
    44: kernelInstance( 0, 0, 8, 4, 1, 0, 0, 0, 0, 2,   64,   64,    128,  1,  4,  1,   16,    16,    64, "Default"),
    45: kernelInstance( 0, 0, 8, 4, 1, 0, 0, 0, 0, 2,   32,   128,   128,  1,  4,  1,   16,    16,    64, "Default"),
    46: kernelInstance( 0, 0, 8, 4, 1, 0, 0, 0, 0, 2,   64,   128,   128,  1,  4,  1,   16,    16,    64, "Default"),
    47: kernelInstance( 0, 0, 8, 4, 1, 0, 0, 0, 0, 2,   32,   256,   128,  1,  4,  1,   16,    16,    64, "Default"),
    48: kernelInstance( 0, 0, 8, 4, 1, 0, 0, 0, 0, 2,   64,   256,   128,  1,  4,  1,   16,    16,    64, "Default"),
    49: kernelInstance( 0, 0, 8, 4, 1, 0, 0, 0, 0, 2,   32,    64,   256,  1,  4,  1,   16,    16,    64, "Default"),
    50: kernelInstance( 0, 0, 8, 4, 1, 0, 0, 0, 0, 2,   64,    64,   256,  1,  4,  1,   16,    16,    64, "Default"),
    51: kernelInstance( 0, 0, 8, 4, 1, 0, 0, 0, 0, 2,   32,   128,   256,  1,  4,  1,   16,    16,    64, "Default"),
    52: kernelInstance( 0, 0, 8, 4, 1, 0, 0, 0, 0, 2,   64,   128,   256,  1,  4,  1,   16,    16,    64, "Default"),
    53: kernelInstance( 0, 0, 8, 4, 1, 0, 0, 0, 0, 2,   32,   192,   128,  1,  4,  1,   16,    16,    64, "Default"),
    54: kernelInstance( 0, 0, 8, 4, 1, 0, 0, 0, 0, 2,   64,   192,   128,  1,  4,  1,   16,    16,    64, "Default"),
    55: kernelInstance( 0, 0, 8, 4, 1, 0, 0, 0, 0, 2,   128,  192,   128,  1,  4,  1,   16,    16,    64, "Default"),
    56: kernelInstance( 0, 0, 8, 4, 1, 0, 0, 0, 0, 1,   16,   192,   256,  1,  4,  1,   16,    16,    64, "Default"),
    57: kernelInstance( 0, 0, 8, 4, 1, 0, 0, 0, 0, 1,   32,   192,   256,  1,  4,  1,   16,    16,    64, "Default"),
    58: kernelInstance( 0, 0, 8, 4, 1, 0, 0, 0, 0, 1,   64,   192,   256,  1,  4,  1,   16,    16,    64, "Default"),
    59: kernelInstance( 0, 0, 8, 4, 1, 0, 0, 0, 0, 1,   64,   256,   256,  1,  4,  1,   16,    16,    64, "Default"),
    60: kernelInstance( 0, 0, 8, 4, 1, 0, 0, 0, 0, 1,   32,   256,   256,  1,  4,  1,   16,    16,    64, "Default"),
    61: kernelInstance( 0, 0, 8, 4, 1, 0, 0, 0, 0, 1,   16,   512,   256,  1,  4,  1,   16,    16,    64, "Default"),
    62: kernelInstance( 0, 0, 8, 4, 1, 0, 0, 0, 0, 1,   64,   256,   64,   1,  4,  1,   16,    16,    64, "Default"),
    63: kernelInstance( 0, 0, 8, 4, 1, 0, 0, 0, 0, 1,   16,   256,   512,  1,  4,  1,   16,    16,    64, "Default"),
    64: kernelInstance( 0, 0, 8, 4, 1, 0, 0, 0, 0, 1,   16,   256,   256,  1,  4,  1,   16,    16,    64, "Default"),
    65: kernelInstance( 0, 0, 8, 4, 1, 0, 0, 0, 0, 2,   16,   192,   256,  1,  4,  1,   16,    16,    64, "Default"),
    66: kernelInstance( 0, 0, 8, 4, 1, 0, 0, 0, 0, 2,   32,   192,   256,  1,  4,  1,   16,    16,    64, "Default"),
    67: kernelInstance( 0, 0, 8, 4, 1, 0, 0, 0, 0, 2,   64,   192,   256,  1,  4,  1,   16,    16,    64, "Default"),
    68: kernelInstance( 0, 0, 8, 4, 1, 0, 0, 0, 0, 2,   64,   256,   256,  1,  4,  1,   16,    16,    64, "Default"),
    69: kernelInstance( 0, 0, 8, 4, 1, 0, 0, 0, 0, 2,   32,   256,   256,  1,  4,  1,   16,    16,    64, "Default"),
    70: kernelInstance( 0, 0, 8, 4, 1, 0, 0, 0, 0, 2,   16,   512,   256,  1,  4,  1,   16,    16,    64, "Default"),
    71: kernelInstance( 0, 0, 8, 4, 1, 0, 0, 0, 0, 2,   64,   256,    64,  1,  4,  1,   16,    16,    64, "Default"),
    72: kernelInstance( 0, 0, 8, 4, 1, 0, 0, 0, 0, 2,   16,   256,   512,  1,  4,  1,   16,    16,    64, "Default"),
    73: kernelInstance( 0, 0, 8, 4, 1, 0, 0, 0, 0, 2,   16,   256,   256,  1,  4,  1,   16,    16,    64, "Default"),
    74: kernelInstance( 0, 0, 8, 4, 1, 0, 0, 0, 0, 1,   160,   192,   128, 1,  4,  1,   16,    16,    64, "Default"),
    75: kernelInstance( 0, 0, 8, 4, 1, 0, 0, 0, 0, 1,   48,    128,   256, 1,  4,  1,   16,    16,    64, "Default"),
    76: kernelInstance( 0, 0, 8, 4, 1, 0, 0, 0, 0, 1,   96,    192,   128, 1,  4,  1,   16,    16,    64, "Default"),
    77: kernelInstance( 0, 0, 8, 4, 1, 0, 0, 0, 0, 1,   48,    192,   256, 1,  4,  1,   16,    16,    64, "Default"),
    78: kernelInstance( 0, 0, 8, 4, 1, 0, 0, 0, 0, 1,   48,    64,    256, 1,  4,  1,   16,    16,    64, "Default"),
    79: kernelInstance( 0, 0, 8, 4, 1, 0, 0, 0, 0, 1,   96,    64,    128, 1,  4,  1,   16,    16,    64, "Default"),
    80: kernelInstance( 0, 0, 8, 4, 1, 0, 0, 0, 0, 1,   96,    128,   128, 1,  4,  1,   16,    16,    64, "Default"),
    81: kernelInstance( 0, 0, 8, 4, 1, 0, 0, 0, 0, 1,   96,    192,   128, 1,  4,  1,   16,    16,    64, "Default"),
    82: kernelInstance( 0, 0, 8, 4, 1, 0, 0, 0, 0, 1,   96,    256,   128, 1,  4,  1,   16,    16,    64, "Default"),
    83: kernelInstance( 0, 0, 8, 4, 1, 0, 0, 0, 0, 1,   48,    64,    256, 1,  4,  1,   16,    16,    64, "Default"),
    84: kernelInstance( 0, 0, 8, 4, 1, 0, 0, 0, 0, 1,   80,    64,    256, 1,  4,  1,   16,    16,    64, "Default"),
    85: kernelInstance( 0, 0, 8, 4, 1, 0, 0, 0, 0, 1,   96,    64,    256, 1,  4,  1,   16,    16,    64, "Default"),
    86: kernelInstance( 0, 0, 8, 4, 1, 0, 0, 0, 0, 1,   112,   128,   256, 1,  4,  1,   16,    16,    64, "Default"),
    87: kernelInstance( 0, 0, 8, 4, 1, 0, 0, 0, 0, 1,   112,   64,    256, 1,  4,  1,   16,    16,    64, "Default"),
    88: kernelInstance( 0, 0, 8, 4, 1, 0, 0, 0, 0, 1,   160,   192,   128, 1,  4,  1,   16,    16,    64, "Default"),
    89: kernelInstance( 0, 0, 8, 4, 1, 0, 0, 0, 0, 1,   224,   192,   128, 1,  4,  1,   16,    16,    64, "Default"),
    90: kernelInstance( 0, 0, 8, 4, 1, 0, 0, 0, 0, 1,   256,   192,   128, 1,  4,  1,   16,    16,    64, "Default"),
    91: kernelInstance( 0, 0, 8, 4, 1, 0, 0, 0, 0, 1,   48,    256,   256, 1,  4,  1,   16,    16,    64, "Default"),
    92: kernelInstance( 0, 0, 8, 4, 1, 0, 0, 0, 0, 1,   80,    128,   256, 1,  4,  1,   16,    16,    64, "Default"),
    93: kernelInstance( 0, 0, 8, 4, 1, 0, 0, 0, 0, 1,   224,    64,   128, 1,  4,  1,   16,    16,    64, "Default"),
    94: kernelInstance( 0, 0, 8, 4, 1, 0, 0, 0, 0, 1,   112,   192,   256, 1,  4,  1,   16,    16,    64, "Default"),
    95: kernelInstance( 0, 0, 8, 4, 1, 0, 0, 0, 0, 1,   128,   192,   256, 1,  4,  1,   16,    16,    64, "Default"),
    96: kernelInstance( 0, 0, 8, 4, 1, 0, 0, 0, 0, 1,   192,   128,   128, 1,  4,  1,   16,    16,    64, "Default"),
    97: kernelInstance( 0, 0, 8, 4, 1, 0, 0, 0, 0, 1,   224,   128,   128, 1,  4,  1,   16,    16,    64, "Default"),
    98: kernelInstance( 0, 0, 8, 4, 1, 0, 0, 0, 0, 1,   96,   192,   256,  1,  4,  1,   16,    16,    64, "Default"),
    99: kernelInstance( 0, 0, 8, 4, 1, 0, 0, 0, 0, 1,   96,   128,   256,  1,  4,  1,   16,    16,    64, "Default"),
    100: kernelInstance( 0, 0, 8, 4, 1, 0, 0, 0, 0, 1,  256,  128,   128,  1,  4,  1,   16,    16,    64, "Default"),
    101: kernelInstance( 0, 0, 8, 4, 1, 0, 0, 0, 0, 1,  80,   256,   256,  1,  4,  1,   16,    16,    64, "Default"),
    102: kernelInstance( 0, 0, 8, 4, 1, 0, 0, 0, 0, 1,  192,   64,   128,  1,  4,  1,   16,    16,    64, "Default"),
    103: kernelInstance( 0, 0, 8, 4, 1, 0, 0, 0, 0, 1,  80,   192,   256,  1,  4,  1,   16,    16,    64, "Default"),
    104: kernelInstance( 0, 0, 8, 4, 1, 0, 0, 0, 0, 1,  256,   64,   128,  1,  4,  1,   16,    16,    64, "Default"),
    105: kernelInstance( 0, 0, 8, 4, 1, 0, 0, 0, 0, 1,  112,  256,   256,  1,  4,  1,   16,    16,    64, "Default"),
    106: kernelInstance( 0, 0, 8, 4, 1, 0, 0, 0, 0, 1,  96,   256,   256,  1,  4,  1,   16,    16,    64, "Default"),
    107: kernelInstance( 0, 0, 8, 4, 1, 0, 0, 0, 0, 2,  48,   128,   256,  1,  4,  1,   16,    16,    64, "Default"),
    108: kernelInstance( 0, 0, 8, 4, 1, 0, 0, 0, 0, 2,  96,   192,   128,  1,  4,  1,   16,    16,    64, "Default"),
    109: kernelInstance( 0, 0, 8, 4, 1, 0, 0, 0, 0, 2,  48,   192,   256,  1,  4,  1,   16,    16,    64, "Default"),
    110: kernelInstance( 0, 0, 8, 4, 1, 0, 0, 0, 0, 2,  48,    64,   256,  1,  4,  1,   16,    16,    64, "Default"),
    111: kernelInstance( 0, 0, 8, 4, 1, 0, 0, 0, 0, 2,  96,    64,   128,  1,  4,  1,   16,    16,    64, "Default"),
    112: kernelInstance( 0, 0, 8, 4, 1, 0, 0, 0, 0, 2,   96,   128,  128,  1,  4,  1,   16,    16,    64, "Default"),
    113: kernelInstance( 0, 0, 8, 4, 1, 0, 0, 0, 0, 2,   96,   256,  256,  1,  4,  1,   16,    16,    64, "Default"),
    114: kernelInstance( 0, 0, 8, 4, 1, 0, 0, 0, 0, 2,   96,   192,  128,  1,  4,  1,   16,    16,    64, "Default"),
    115: kernelInstance( 0, 0, 8, 4, 1, 0, 0, 0, 0, 2,   96,   256,  128,  1,  4,  1,   16,    16,    64, "Default"),
    116: kernelInstance( 0, 0, 8, 4, 1, 0, 0, 0, 0, 2,   48,   64,   256,  1,  4,  1,   16,    16,    64, "Default"),
    117: kernelInstance( 0, 0, 8, 4, 1, 0, 0, 0, 0, 2,   80,   64,   256,  1,  4,  1,   16,    16,    64, "Default"),
    118: kernelInstance( 0, 0, 8, 4, 1, 0, 0, 0, 0, 2,   96,   64,   256,  1,  4,  1,   16,    16,    64, "Default"),
    119: kernelInstance( 0, 0, 8, 4, 1, 0, 0, 0, 0, 2,   112,  128,  256,  1,  4,  1,   16,    16,    64, "Default"),
    120: kernelInstance( 0, 0, 8, 4, 1, 0, 0, 0, 0, 2,   112,  64,   256,  1,  4,  1,   16,    16,    64, "Default"),
    121: kernelInstance( 0, 0, 8, 4, 1, 0, 0, 0, 0, 2,   160,  192,  128,  1,  4,  1,   16,    16,    64, "Default"),
    122: kernelInstance( 0, 0, 8, 4, 1, 0, 0, 0, 0, 2,   224,  192,  128,  1,  4,  1,   16,    16,    64, "Default"),
    123: kernelInstance( 0, 0, 8, 4, 1, 0, 0, 0, 0, 2,   256,  192,  128,  1,  4,  1,   16,    16,    64, "Default"),
    124: kernelInstance( 0, 0, 8, 4, 1, 0, 0, 0, 0, 2,   48,   256,  256,  1,  4,  1,   16,    16,    64, "Default"),
    125: kernelInstance( 0, 0, 8, 4, 1, 0, 0, 0, 0, 2,   80,   128,  256,  1,  4,  1,   16,    16,    64, "Default"),
    126: kernelInstance( 0, 0, 8, 4, 1, 0, 0, 0, 0, 2,   224,   64,  128,  1,  4,  1,   16,    16,    64, "Default"),
    127: kernelInstance( 0, 0, 8, 4, 1, 0, 0, 0, 0, 2,   112,  192,  256,  1,  4,  1,   16,    16,    64, "Default"),
    128: kernelInstance( 0, 0, 8, 4, 1, 0, 0, 0, 0, 2,   128,  192,  256,  1,  4,  1,   16,    16,    64, "Default"),
    129: kernelInstance( 0, 0, 8, 4, 1, 0, 0, 0, 0, 2,   192,  128,  128,  1,  4,  1,   16,    16,    64, "Default"),
    130: kernelInstance( 0, 0, 8, 4, 1, 0, 0, 0, 0, 2,   224,  128,  128,  1,  4,  1,   16,    16,    64, "Default"),
    131: kernelInstance( 0, 0, 8, 4, 1, 0, 0, 0, 0, 2,   96,   192,  256,  1,  4,  1,   16,    16,    64, "Default"),
    132: kernelInstance( 0, 0, 8, 4, 1, 0, 0, 0, 0, 2,   96,   128,  256,  1,  4,  1,   16,    16,    64, "Default"),
    133: kernelInstance( 0, 0, 8, 4, 1, 0, 0, 0, 0, 2,   256,  128,  128,  1,  4,  1,   16,    16,    64, "Default"),
    134: kernelInstance( 0, 0, 8, 4, 1, 0, 0, 0, 0, 2,   80,   256,  256,  1,  4,  1,   16,    16,    64, "Default"),
    135: kernelInstance( 0, 0, 8, 4, 1, 0, 0, 0, 0, 2,   192,   64,  128,  1,  4,  1,   16,    16,    64, "Default"),
    136: kernelInstance( 0, 0, 8, 4, 1, 0, 0, 0, 0, 2,   80,   192,  256,  1,  4,  1,   16,    16,    64, "Default"),
    137: kernelInstance( 0, 0, 8, 4, 1, 0, 0, 0, 0, 2,   256,   64,  128,  1,  4,  1,   16,    16,    64, "Default"),
    138: kernelInstance( 0, 0, 8, 4, 1, 0, 0, 0, 0, 2,   112,  256,  256,  1,  4,  1,   16,    16,    64, "Default"),

}
# '''

default_kernels_dict_942 = {
    (-1): kernelInstance(0, 0, 8, 4, 1, 0, 0, 0, 0, 1, 128,   128,   128,  1,  4,  1,   16,    16,    64, "Default"),
    (-2):kernelInstance( 0, 0, 8, 4, 1, 0, 0, 0, 0, 1, 16,    64,    512,  1,  4,  1,   16,    16,    64, "Default"),
    (-3):kernelInstance( 0, 0, 8, 4, 1, 0, 0, 0, 0, 1, 32,    64,    512,  1,  4,  1,   16,    16,    64, "Default"),
    (-4):kernelInstance( 0, 0, 8, 4, 1, 0, 0, 0, 0, 1, 64,    256,   64,   1,  4,  1,   16,    16,    64, "Default"),
    (-5):kernelInstance( 0, 0, 8, 4, 1, 0, 0, 0, 0, 1, 128,   128,   64,   1,  4,  1,   16,    16,    64, "Default"),
    (-6):kernelInstance( 0, 0, 8, 4, 1, 0, 0, 0, 0, 1, 128,   64,   128,   1,  4,  1,   16,    16,    64, "Default"),
    (-7):kernelInstance( 0, 0, 8, 4, 1, 0, 0, 0, 0, 1, 64,   256,   128,   1,  4,  1,   16,    16,    64, "Default"),
}

kernels_list_950 = {
    0: kernelInstance( 0, 0, 8, 4, 1, 0, 0, 0, 0, 1,   128,   128,   128,  1,  4,  1,   16,    16,    128, "Default"),
    1: kernelInstance( 0, 0, 8, 4, 1, 0, 0, 0, 0, 1,   128,   128,   256,  1,  4,  1,   16,    16,    128, "Default"),
    2: kernelInstance( 0, 0, 8, 4, 1, 0, 0, 0, 0, 1,   16,    64,    512,  1,  4,  1,   16,    16,    128, "Default"),
    3: kernelInstance( 0, 0, 8, 4, 1, 0, 0, 0, 0, 1,   16,    128,   512,  1,  4,  1,   16,    16,    128, "Default"),
    4: kernelInstance( 0, 0, 8, 4, 1, 0, 0, 0, 0, 1,   16,    256,   512,  1,  4,  1,   16,    16,    128, "Default"),
    5: kernelInstance( 0, 0, 8, 4, 1, 0, 0, 0, 0, 1,   16,    64,    256,  1,  4,  1,   16,    16,    128, "Default"),
    6: kernelInstance( 0, 0, 8, 4, 1, 0, 0, 0, 0, 1,   16,    128,   256,  1,  4,  1,   16,    16,    128, "Default"),
    7: kernelInstance( 0, 0, 8, 4, 1, 0, 0, 0, 0, 1,   16,    256,   256,  1,  4,  1,   16,    16,    128, "Default"),
    8: kernelInstance( 0, 0, 8, 4, 1, 0, 0, 0, 0, 1,   16,    512,   256,  1,  4,  1,   16,    16,    128, "Default"),
    9: kernelInstance( 0, 0, 8, 4, 1, 0, 0, 0, 0, 1,   32,    64,    512,  1,  4,  1,   16,    16,    128, "Default"),
    10: kernelInstance( 0, 0, 8, 4, 1, 0, 0, 0, 0, 1,  256,   256,   128,  1,  4,  1,   16,    16,    128, "Default"),
    11: kernelInstance( 0, 0, 8, 4, 1, 0, 0, 0, 0, 1,  128,   64,   128,   1,  4,  1,   16,    16,    128, "Default"),
    12: kernelInstance( 0, 0, 8, 4, 1, 0, 0, 0, 0, 1,  128,   64,   256,   1,  4,  1,   16,    16,    128, "Default"),
    13: kernelInstance( 0, 0, 8, 4, 1, 0, 0, 0, 0, 2,  256,  128,   128,   1,  4,  1,   16,    16,    128, "Default"),
    14: kernelInstance( 0, 0, 8, 4, 1, 0, 0, 0, 0, 1,  128,   256,   128,  1,  4,  1,   16,    16,    128, "Default"),
    15: kernelInstance( 0, 0, 8, 4, 1, 0, 0, 0, 0, 1,   32,   64,    128,  1,  4,  1,   16,    16,    128, "Default"),
    16: kernelInstance( 0, 0, 8, 4, 1, 0, 0, 0, 0, 1,   64,   64,    128,  1,  4,  1,   16,    16,    128, "Default"),
    17: kernelInstance( 0, 0, 8, 4, 1, 0, 0, 0, 0, 1,   32,   128,   128,  1,  4,  1,   16,    16,    128, "Default"),
    18: kernelInstance( 0, 0, 8, 4, 1, 0, 0, 0, 0, 1,   64,   128,   128,  1,  4,  1,   16,    16,    128, "Default"),
    19: kernelInstance( 0, 0, 8, 4, 1, 0, 0, 0, 0, 1,   32,   256,   128,  1,  4,  1,   16,    16,    128, "Default"),
    20: kernelInstance( 0, 0, 8, 4, 1, 0, 0, 0, 0, 1,   64,   256,   128,  1,  4,  1,   16,    16,    128, "Default"),
    21: kernelInstance( 0, 0, 8, 4, 1, 0, 0, 0, 0, 1,   32,    64,   256,  1,  4,  1,   16,    16,    128, "Default"),
    22: kernelInstance( 0, 0, 8, 4, 1, 0, 0, 0, 0, 1,   64,    64,   256,  1,  4,  1,   16,    16,    128, "Default"),
    23: kernelInstance( 0, 0, 8, 4, 1, 0, 0, 0, 0, 1,   32,   128,   256,  1,  4,  1,   16,    16,    128, "Default"),
    24: kernelInstance( 0, 0, 8, 4, 1, 0, 0, 0, 0, 1,   64,   128,   256,  1,  4,  1,   16,    16,    128, "Default"),
    25: kernelInstance( 0, 0, 8, 4, 1, 0, 0, 0, 0, 1,   32,   192,   128,  1,  4,  1,   16,    16,    128, "Default"),
    26: kernelInstance( 0, 0, 8, 4, 1, 0, 0, 0, 0, 1,   64,   192,   128,  1,  4,  1,   16,    16,    128, "Default"),
    27: kernelInstance( 0, 0, 8, 4, 1, 0, 0, 0, 0, 1,   128,  192,   128,  1,  4,  1,   16,    16,    128, "Default"),
    28: kernelInstance( 0, 0, 8, 4, 1, 0, 0, 0, 0, 2,   128,  128,   128,  1,  4,  1,   16,    16,    128, "Default"),
    29: kernelInstance( 0, 0, 8, 4, 1, 0, 0, 0, 0, 2,   128,  128,   256,  1,  4,  1,   16,    16,    128, "Default"),
    30: kernelInstance( 0, 0, 8, 4, 1, 0, 0, 0, 0, 2,   16,   64,    512,  1,  4,  1,   16,    16,    128, "Default"),
    31: kernelInstance( 0, 0, 8, 4, 1, 0, 0, 0, 0, 2,   16,   128,   512,  1,  4,  1,   16,    16,    128, "Default"),
    32: kernelInstance( 0, 0, 8, 4, 1, 0, 0, 0, 0, 2,   16,   256,   512,  1,  4,  1,   16,    16,    128, "Default"),
    33: kernelInstance( 0, 0, 8, 4, 1, 0, 0, 0, 0, 2,   16,   64,    256,  1,  4,  1,   16,    16,    128, "Default"),
    34: kernelInstance( 0, 0, 8, 4, 1, 0, 0, 0, 0, 2,   16,   128,   256,  1,  4,  1,   16,    16,    128, "Default"),
    35: kernelInstance( 0, 0, 8, 4, 1, 0, 0, 0, 0, 2,   16,   256,   256,  1,  4,  1,   16,    16,    128, "Default"),
    36: kernelInstance( 0, 0, 8, 4, 1, 0, 0, 0, 0, 2,   16,   512,   256,  1,  4,  1,   16,    16,    128, "Default"),
    37: kernelInstance( 0, 0, 8, 4, 1, 0, 0, 0, 0, 2,   32,   64,    512,  1,  4,  1,   16,    16,    128, "Default"),
    38: kernelInstance( 0, 0, 8, 4, 1, 0, 0, 0, 0, 2,   192,   64,   128,  1,  4,  1,   16,    16,   128, "Default"),
    39: kernelInstance( 0, 0, 8, 4, 1, 0, 0, 0, 0, 2,  128,   64,   128,   1,  4,  1,   16,    16,    128, "Default"),
    40: kernelInstance( 0, 0, 8, 4, 1, 0, 0, 0, 0, 2,  128,   64,   256,   1,  4,  1,   16,    16,    128, "Default"),
    41: kernelInstance( 0, 0, 8, 4, 1, 0, 0, 0, 0, 2,   80,   256,   256,  1,  4,  1,   16,    16,   128, "Default"),
    42: kernelInstance( 0, 0, 8, 4, 1, 0, 0, 0, 0, 2,  128,   256,   128,  1,  4,  1,   16,    16,    128, "Default"),
    43: kernelInstance( 0, 0, 8, 4, 1, 0, 0, 0, 0, 2,   32,   64,    128,  1,  4,  1,   16,    16,    128, "Default"),
    44: kernelInstance( 0, 0, 8, 4, 1, 0, 0, 0, 0, 2,   64,   64,    128,  1,  4,  1,   16,    16,    128, "Default"),
    45: kernelInstance( 0, 0, 8, 4, 1, 0, 0, 0, 0, 2,   32,   128,   128,  1,  4,  1,   16,    16,    128, "Default"),
    46: kernelInstance( 0, 0, 8, 4, 1, 0, 0, 0, 0, 2,   64,   128,   128,  1,  4,  1,   16,    16,    128, "Default"),
    47: kernelInstance( 0, 0, 8, 4, 1, 0, 0, 0, 0, 2,   32,   256,   128,  1,  4,  1,   16,    16,    128, "Default"),
    48: kernelInstance( 0, 0, 8, 4, 1, 0, 0, 0, 0, 2,   64,   256,   128,  1,  4,  1,   16,    16,    128, "Default"),
    49: kernelInstance( 0, 0, 8, 4, 1, 0, 0, 0, 0, 2,   32,    64,   256,  1,  4,  1,   16,    16,    128, "Default"),
    50: kernelInstance( 0, 0, 8, 4, 1, 0, 0, 0, 0, 2,   64,    64,   256,  1,  4,  1,   16,    16,    128, "Default"),
    51: kernelInstance( 0, 0, 8, 4, 1, 0, 0, 0, 0, 2,   32,   128,   256,  1,  4,  1,   16,    16,    128, "Default"),
    52: kernelInstance( 0, 0, 8, 4, 1, 0, 0, 0, 0, 2,   64,   128,   256,  1,  4,  1,   16,    16,    128, "Default"),
    53: kernelInstance( 0, 0, 8, 4, 1, 0, 0, 0, 0, 2,   32,   192,   128,  1,  4,  1,   16,    16,    128, "Default"),
    54: kernelInstance( 0, 0, 8, 4, 1, 0, 0, 0, 0, 2,   64,   192,   128,  1,  4,  1,   16,    16,    128, "Default"),
    55: kernelInstance( 0, 0, 8, 4, 1, 0, 0, 0, 0, 2,   128,  192,   128,  1,  4,  1,   16,    16,    128, "Default"),
    56: kernelInstance( 0, 0, 8, 4, 1, 0, 0, 0, 0, 1,   16,   192,   256,  1,  4,  1,   16,    16,    128, "Default"),
    57: kernelInstance( 0, 0, 8, 4, 1, 0, 0, 0, 0, 1,   32,   192,   256,  1,  4,  1,   16,    16,    128, "Default"),
    58: kernelInstance( 0, 0, 8, 4, 1, 0, 0, 0, 0, 1,   64,   192,   256,  1,  4,  1,   16,    16,    128, "Default"),
    59: kernelInstance( 0, 0, 8, 4, 1, 0, 0, 0, 0, 1,   64,   256,   256,  1,  4,  1,   16,    16,    128, "Default"),
    60: kernelInstance( 0, 0, 8, 4, 1, 0, 0, 0, 0, 1,   32,   256,   256,  1,  4,  1,   16,    16,    128, "Default"),
    61: kernelInstance( 0, 0, 8, 4, 1, 0, 0, 0, 0, 1,   16,   512,   256,  1,  4,  1,   16,    16,    128, "Default"),
    62: kernelInstance( 0, 0, 8, 4, 1, 0, 0, 0, 0, 2,   80,   192,   256,  1,  4,  1,   16,    16,    128, "Default"),
    63: kernelInstance( 0, 0, 8, 4, 1, 0, 0, 0, 0, 1,   16,   256,   512,  1,  4,  1,   16,    16,    128, "Default"),
    64: kernelInstance( 0, 0, 8, 4, 1, 0, 0, 0, 0, 1,   16,   256,   256,  1,  4,  1,   16,    16,    128, "Default"),
    65: kernelInstance( 0, 0, 8, 4, 1, 0, 0, 0, 0, 2,   16,   192,   256,  1,  4,  1,   16,    16,    128, "Default"),
    66: kernelInstance( 0, 0, 8, 4, 1, 0, 0, 0, 0, 2,   32,   192,   256,  1,  4,  1,   16,    16,    128, "Default"),
    67: kernelInstance( 0, 0, 8, 4, 1, 0, 0, 0, 0, 2,   64,   192,   256,  1,  4,  1,   16,    16,    128, "Default"),
    68: kernelInstance( 0, 0, 8, 4, 1, 0, 0, 0, 0, 2,   64,   256,   256,  1,  4,  1,   16,    16,    128, "Default"),
    69: kernelInstance( 0, 0, 8, 4, 1, 0, 0, 0, 0, 2,   32,   256,   256,  1,  4,  1,   16,    16,    128, "Default"),
    70: kernelInstance( 0, 0, 8, 4, 1, 0, 0, 0, 0, 2,   16,   512,   256,  1,  4,  1,   16,    16,    128, "Default"),
    71: kernelInstance( 0, 0, 8, 4, 1, 0, 0, 0, 0, 2,   256,   64,   128,  1,  4,  1,   16,    16,    128, "Default"),
    72: kernelInstance( 0, 0, 8, 4, 1, 0, 0, 0, 0, 2,   16,   256,   512,  1,  4,  1,   16,    16,    128, "Default"),
    73: kernelInstance( 0, 0, 8, 4, 1, 0, 0, 0, 0, 2,   16,   256,   256,  1,  4,  1,   16,    16,    128, "Default"),
    74: kernelInstance( 0, 0, 8, 4, 1, 0, 0, 0, 0, 1,   160,   192,   128, 1,  4,  1,   16,    16,    128, "Default"),
    75: kernelInstance( 0, 0, 8, 4, 1, 0, 0, 0, 0, 1,   48,    128,   256, 1,  4,  1,   16,    16,    128, "Default"),
    76: kernelInstance( 0, 0, 8, 4, 1, 0, 0, 0, 0, 1,   96,    192,   128, 1,  4,  1,   16,    16,    128, "Default"),
    77: kernelInstance( 0, 0, 8, 4, 1, 0, 0, 0, 0, 1,   48,    192,   256, 1,  4,  1,   16,    16,    128, "Default"),
    78: kernelInstance( 0, 0, 8, 4, 1, 0, 0, 0, 0, 1,   48,    64,    256, 1,  4,  1,   16,    16,    128, "Default"),
    79: kernelInstance( 0, 0, 8, 4, 1, 0, 0, 0, 0, 1,   96,    64,    128, 1,  4,  1,   16,    16,    128, "Default"),
    80: kernelInstance( 0, 0, 8, 4, 1, 0, 0, 0, 0, 1,   96,    128,   128, 1,  4,  1,   16,    16,    128, "Default"),
    81: kernelInstance( 0, 0, 8, 4, 1, 0, 0, 0, 0, 1,   96,    192,   128, 1,  4,  1,   16,    16,    128, "Default"),
    82: kernelInstance( 0, 0, 8, 4, 1, 0, 0, 0, 0, 1,   96,    256,   128, 1,  4,  1,   16,    16,    128, "Default"),
    83: kernelInstance( 0, 0, 8, 4, 1, 0, 0, 0, 0, 1,   48,    64,    256, 1,  4,  1,   16,    16,    128, "Default"),
    84: kernelInstance( 0, 0, 8, 4, 1, 0, 0, 0, 0, 1,   80,    64,    256, 1,  4,  1,   16,    16,    128, "Default"),
    85: kernelInstance( 0, 0, 8, 4, 1, 0, 0, 0, 0, 1,   96,    64,    256, 1,  4,  1,   16,    16,    128, "Default"),
    86: kernelInstance( 0, 0, 8, 4, 1, 0, 0, 0, 0, 1,   112,   128,   256, 1,  4,  1,   16,    16,    128, "Default"),
    87: kernelInstance( 0, 0, 8, 4, 1, 0, 0, 0, 0, 1,   112,   64,    256, 1,  4,  1,   16,    16,    128, "Default"),
    88: kernelInstance( 0, 0, 8, 4, 1, 0, 0, 0, 0, 1,   160,   192,   128, 1,  4,  1,   16,    16,    128, "Default"),
    89: kernelInstance( 0, 0, 8, 4, 1, 0, 0, 0, 0, 1,   224,   192,   128, 1,  4,  1,   16,    16,    128, "Default"),
    90: kernelInstance( 0, 0, 8, 4, 1, 0, 0, 0, 0, 1,   256,   192,   128, 1,  4,  1,   16,    16,    128, "Default"),
    91: kernelInstance( 0, 0, 8, 4, 1, 0, 0, 0, 0, 1,   48,    256,   256, 1,  4,  1,   16,    16,    128, "Default"),
    92: kernelInstance( 0, 0, 8, 4, 1, 0, 0, 0, 0, 1,   80,    128,   256, 1,  4,  1,   16,    16,    128, "Default"),
    93: kernelInstance( 0, 0, 8, 4, 1, 0, 0, 0, 0, 1,   224,    64,   128, 1,  4,  1,   16,    16,    128, "Default"),
    94: kernelInstance( 0, 0, 8, 4, 1, 0, 0, 0, 0, 1,   112,   192,   256, 1,  4,  1,   16,    16,    128, "Default"),
    95: kernelInstance( 0, 0, 8, 4, 1, 0, 0, 0, 0, 1,   128,   192,   256, 1,  4,  1,   16,    16,    128, "Default"),
    96: kernelInstance( 0, 0, 8, 4, 1, 0, 0, 0, 0, 1,   192,   128,   128, 1,  4,  1,   16,    16,    128, "Default"),
    97: kernelInstance( 0, 0, 8, 4, 1, 0, 0, 0, 0, 1,   224,   128,   128, 1,  4,  1,   16,    16,    128, "Default"),
    98: kernelInstance( 0, 0, 8, 4, 1, 0, 0, 0, 0, 1,   96,   192,   256,  1,  4,  1,   16,    16,    128, "Default"),
    99: kernelInstance( 0, 0, 8, 4, 1, 0, 0, 0, 0, 1,   96,   128,   256,  1,  4,  1,   16,    16,    128, "Default"),
    100: kernelInstance( 0, 0, 8, 4, 1, 0, 0, 0, 0, 1,  256,  128,   128,  1,  4,  1,   16,    16,    128, "Default"),
    101: kernelInstance( 0, 0, 8, 4, 1, 0, 0, 0, 0, 1,  80,   256,   256,  1,  4,  1,   16,    16,    128, "Default"),
    102: kernelInstance( 0, 0, 8, 4, 1, 0, 0, 0, 0, 1,  192,   64,   128,  1,  4,  1,   16,    16,    128, "Default"),
    103: kernelInstance( 0, 0, 8, 4, 1, 0, 0, 0, 0, 1,  80,   192,   256,  1,  4,  1,   16,    16,    128, "Default"),
    104: kernelInstance( 0, 0, 8, 4, 1, 0, 0, 0, 0, 1,  256,   64,   128,  1,  4,  1,   16,    16,    128, "Default"),
    105: kernelInstance( 0, 0, 8, 4, 1, 0, 0, 0, 0, 1,  112,  256,   256,  1,  4,  1,   16,    16,    128, "Default"),
    106: kernelInstance( 0, 0, 8, 4, 1, 0, 0, 0, 0, 1,  96,   256,   256,  1,  4,  1,   16,    16,    128, "Default"),
    107: kernelInstance( 0, 0, 8, 4, 1, 0, 0, 0, 0, 2,  48,   128,   256,  1,  4,  1,   16,    16,    128, "Default"),
    108: kernelInstance( 0, 0, 8, 4, 1, 0, 0, 0, 0, 2,  96,   192,   128,  1,  4,  1,   16,    16,    128, "Default"),
    109: kernelInstance( 0, 0, 8, 4, 1, 0, 0, 0, 0, 2,  48,   192,   256,  1,  4,  1,   16,    16,    128, "Default"),
    110: kernelInstance( 0, 0, 8, 4, 1, 0, 0, 0, 0, 2,  48,    64,   256,  1,  4,  1,   16,    16,    128, "Default"),
    111: kernelInstance( 0, 0, 8, 4, 1, 0, 0, 0, 0, 2,  96,    64,   128,  1,  4,  1,   16,    16,    128, "Default"),
    112: kernelInstance( 0, 0, 8, 4, 1, 0, 0, 0, 0, 2,   96,   128,  128,  1,  4,  1,   16,    16,    128, "Default"),
    113: kernelInstance( 0, 0, 8, 4, 1, 0, 0, 0, 0, 2,   96,   256,  256,  1,  4,  1,   16,    16,    128, "Default"),
    114: kernelInstance( 0, 0, 8, 4, 1, 0, 0, 0, 0, 2,   96,   192,  128,  1,  4,  1,   16,    16,    128, "Default"),
    115: kernelInstance( 0, 0, 8, 4, 1, 0, 0, 0, 0, 2,   96,   256,  128,  1,  4,  1,   16,    16,    128, "Default"),
    116: kernelInstance( 0, 0, 8, 4, 1, 0, 0, 0, 0, 2,   48,   64,   256,  1,  4,  1,   16,    16,    128, "Default"),
    117: kernelInstance( 0, 0, 8, 4, 1, 0, 0, 0, 0, 2,   80,   64,   256,  1,  4,  1,   16,    16,    128, "Default"),
    118: kernelInstance( 0, 0, 8, 4, 1, 0, 0, 0, 0, 2,   96,   64,   256,  1,  4,  1,   16,    16,    128, "Default"),
    119: kernelInstance( 0, 0, 8, 4, 1, 0, 0, 0, 0, 2,   112,  128,  256,  1,  4,  1,   16,    16,    128, "Default"),
    120: kernelInstance( 0, 0, 8, 4, 1, 0, 0, 0, 0, 2,   112,  64,   256,  1,  4,  1,   16,    16,    128, "Default"),
    121: kernelInstance( 0, 0, 8, 4, 1, 0, 0, 0, 0, 2,   160,  192,  128,  1,  4,  1,   16,    16,    128, "Default"),
    122: kernelInstance( 0, 0, 8, 4, 1, 0, 0, 0, 0, 2,   224,  192,  128,  1,  4,  1,   16,    16,    128, "Default"),
    123: kernelInstance( 0, 0, 8, 4, 1, 0, 0, 0, 0, 2,   256,  192,  128,  1,  4,  1,   16,    16,    128, "Default"),
    124: kernelInstance( 0, 0, 8, 4, 1, 0, 0, 0, 0, 2,   48,   256,  256,  1,  4,  1,   16,    16,    128, "Default"),
    125: kernelInstance( 0, 0, 8, 4, 1, 0, 0, 0, 0, 2,   80,   128,  256,  1,  4,  1,   16,    16,    128, "Default"),
    126: kernelInstance( 0, 0, 8, 4, 1, 0, 0, 0, 0, 2,   224,   64,  128,  1,  4,  1,   16,    16,    128, "Default"),
    127: kernelInstance( 0, 0, 8, 4, 1, 0, 0, 0, 0, 2,   112,  192,  256,  1,  4,  1,   16,    16,    128, "Default"),
    128: kernelInstance( 0, 0, 8, 4, 1, 0, 0, 0, 0, 2,   128,  192,  256,  1,  4,  1,   16,    16,    128, "Default"),
    129: kernelInstance( 0, 0, 8, 4, 1, 0, 0, 0, 0, 2,   192,  128,  128,  1,  4,  1,   16,    16,    128, "Default"),
    130: kernelInstance( 0, 0, 8, 4, 1, 0, 0, 0, 0, 2,   224,  128,  128,  1,  4,  1,   16,    16,    128, "Default"),
    131: kernelInstance( 0, 0, 8, 4, 1, 0, 0, 0, 0, 2,   96,   192,  256,  1,  4,  1,   16,    16,    128, "Default"),
    132: kernelInstance( 0, 0, 8, 4, 1, 0, 0, 0, 0, 2,   96,   128,  256,  1,  4,  1,   16,    16,    128, "Default"),


}

default_kernels_dict_950 = {
    (-1): kernelInstance( 0, 0, 8, 4, 1, 0, 0, 0, 0,1, 128,   256,   256,  1,  4,  1,   16,    16,    128, "Default"),
    (-2): kernelInstance( 0, 0, 8, 4, 1, 0, 0, 0, 0,1, 16,    64,    512,  1,  4,  1,   16,    16,    128, "Default"),
    (-3): kernelInstance( 0, 0, 8, 4, 1, 0, 0, 0, 0,1, 32,    64,    512,  1,  4,  1,   16,    16,    128, "Default"),
    (-4): kernelInstance( 0, 0, 8, 4, 1, 0, 0, 0, 0,1, 128,   128,   128,  1,  4,  1,   16,    16,    64, "Default"),
}

# fmt: on

arch = get_gfx()
if arch == "gfx942":
    kernels_list = kernels_list_942
    default_kernels_dict = default_kernels_dict_942
else:
    kernels_list = kernels_list_950
    default_kernels_dict = default_kernels_dict_950
