# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
from dataclasses import dataclass


@dataclass
class kernelInstance:
    BLOCK_SIZE: int
    MPerBLOCK: int
    NPerBLOCK: int
    KPerBLOCK: int
    AK1: int
    BK1: int
    MPerXDL: int
    NPerXDL: int
    WAVE_MAP_M: int
    WAVE_MAP_N: int
    ABLOCK_TRANSFER: list[int]
    BBLOCK_TRANSFER: list[int]
    CSHUFFLE_MX_PER_WAVE_PERSHUFFLE: int
    CSHUFFLE_NX_PER_WAVE_PERSHUFFLE: int
    CBLOCK_TRANSFER: list[int]
    CBLOCK_SPV: list[int]
    PIPELINE_Sched: str
    PIPELINE_VERSION: int

    @property
    def name(self) -> str:
        return ("_").join(
            [
                "a8w8_bpreshuffle",
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
                ("x").join(map(lambda x: str(x), [self.AK1, self.BK1])),
                ("x").join(map(lambda x: str(x), [self.MPerXDL, self.NPerXDL])),
                ("x").join(map(lambda x: str(x), self.ABLOCK_TRANSFER)),
                ("x").join(map(lambda x: str(x), self.BBLOCK_TRANSFER)),
                ("x").join(map(lambda x: str(x), self.CBLOCK_TRANSFER)),
                ("x").join(map(lambda x: str(x), self.CBLOCK_SPV)),
                ("x").join(
                    map(
                        lambda x: str(x),
                        [
                            self.CSHUFFLE_MX_PER_WAVE_PERSHUFFLE,
                            self.CSHUFFLE_NX_PER_WAVE_PERSHUFFLE,
                        ],
                    )
                ),
                self.PIPELINE_Sched.lower(),
                f"v{self.PIPELINE_VERSION}",
            ]
        )


# fmt: off
# kernels_list_str = '''
kernels_list = {
    # clang-format off
     ###############|Block|   MPer|  NPer|  KPer| AK1| BK1|MPer| NPer| MXdl| NXdl|  ABlockTransfer|  BBlockTransfer|    CShuffle|    CShuffle|     CBlockTransferClusterLengths|  CBlockTransfer|  Block-wiseGemm|     Block-wiseGemm|
     ###############| Size|  Block| Block| Block|    |    | XDL|  XDL|  Per|  Per|   ThreadCluster|   ThreadCluster| MXdlPerWave| NXdlPerWave| _MBlock_MXdlPerWave_MWaveMPerXdl| ScalarPerVector|    Pipeline    |           Pipeline|
     ###############|     |       |      |      |    |    |    |     | Wave| Wave| Lengths_K0_M_K1| Lengths_K0_N_K1|  PerShuffle|  PerShuffle| _NBlock_NXdlPerWave_NWaveNPerXdl|   _NWaveNPerXdl|    Scheduler   |           Verision|
     ###############|     |       |      |      |    |    |    |     |     |     |                |                |            |            |                                 |                |                |                   |

        # Compute friendly
    0: kernelInstance( 256,    128,   128,   128,  16,  16,  16,   16,    8,    2,    [8, 32, 1],        [8, 32, 1],              2,           1,                 [1, 32, 1, 8],       [8, 8, 1],        "Intrawave",         3),
    1: kernelInstance( 256,    128,   128,   256,  16,  16,  16,   16,    4,    4,    [16, 16, 1],       [16, 16, 1],             1,           2,                 [1, 32, 1, 8],       [8, 8, 1],        "Intrawave",         3),
    2: kernelInstance( 256,    256,   256,   128,  16,  16,  16,   16,    8,    8,    [8, 32, 1],        [8, 32, 1],              1,           2,                 [1, 32, 1, 8],       [8, 8, 1],        "Intrawave",         3),
    3: kernelInstance( 256,    256,   128,   128,  16,  16,  16,   16,    8,    4,    [8, 32, 1],        [8, 32, 1],              1,           2,                 [1, 32, 1, 8],       [8, 8, 1],        "Intrawave",         3),
    4: kernelInstance( 256,    192,   128,   128,  16,  16,  16,   16,    6,    4,    [8, 32, 1],        [8, 32, 1],              1,           2,                 [1, 32, 1, 8],       [8, 8, 1],        "Intrawave",         3),

    5: kernelInstance( 256,     16,    64,   512,  16,  16,  16,   16,    1,    1,    [32, 8, 1],        [32, 8, 1],              1,           1,                 [1, 16, 1, 16],      [4, 4, 1],        "Intrawave",         1),
    6: kernelInstance( 256,     16,   128,   512,  16,  16,  16,   16,    1,    2,    [32, 8, 1],        [32, 8, 1],              1,           2,                 [1, 16, 1, 16],      [8, 8, 1],        "Intrawave",         1),
    7: kernelInstance( 256,     16,   256,   512,  16,  16,  16,   16,    1,    4,    [32, 8, 1],        [32, 8, 1],              1,           2,                 [1, 16, 1, 16],      [8, 8, 1],        "Intrawave",         1),
    8: kernelInstance( 128,     32,    16,   512,  16,  16,  16,   16,    1,    1,    [32, 4, 1],        [32, 4, 1],              1,           1,                 [1, 32, 1, 4],       [4, 4, 1],        "Intrawave",         1),
    9: kernelInstance( 128,     16,    32,   128,  16,  16,  16,   16,    1,    1,    [8, 16, 1],        [8, 16, 1],              1,           1,                 [1, 16, 1, 8],       [4, 4, 1],        "Intrawave",         1),
   10: kernelInstance( 128,     16,    32,   512,  16,  16,  16,   16,    1,    1,    [32, 4, 1],        [32, 4, 1],              1,           1,                 [1, 16, 1, 8],       [4, 4, 1],        "Intrawave",         1),
   11: kernelInstance( 256,     16,    64,   512,  16,  16,  16,   16,    1,    1,    [32, 8, 1],        [32, 8, 1],              1,           1,                 [1, 16, 1, 16],      [4, 4, 1],        "Intrawave",         1),
   12: kernelInstance( 256,     32,    64,   512,  16,  16,  16,   16,    1,    2,    [32, 8, 1],        [32, 8, 1],              1,           2,                 [1, 32, 1, 8],       [8, 8, 1],        "Intrawave",         1),
   13: kernelInstance( 256,     64,    16,   512,  16,  16,  16,   16,    1,    1,    [32, 8, 1],        [32, 8, 1],              1,           1,                 [1, 64, 1, 4],       [4, 4, 1],        "Intrawave",         1),
   14: kernelInstance( 256,     64,    16,   512,  16,  16,  16,   16,    1,    1,    [32, 8, 1],        [32, 8, 1],              1,           1,                 [1, 64, 1, 4],       [4, 4, 1],        "Intrawave",         1),
   15: kernelInstance( 256,     16,    64,   256,  16,  16,  16,   16,    1,    1,    [16, 16, 1],       [16, 16, 1],             1,           1,                 [1, 16, 1, 16],      [4, 4, 1],        "Intrawave",         1),
   16: kernelInstance( 256,     16,   128,   256,  16,  16,  16,   16,    1,    2,    [16, 16, 1],       [16, 16, 1],             1,           2,                 [1, 16, 1, 16],      [8, 8, 1],        "Intrawave",         1),
   17: kernelInstance( 256,     16,   256,   256,  16,  16,  16,   16,    1,    4,    [16, 16, 1],       [16, 16, 1],             1,           2,                 [1, 16, 1, 16],      [8, 8, 1],        "Intrawave",         1),
   18: kernelInstance( 256,     16,   512,   256,  16,  16,  16,   16,    1,    8,    [16, 16, 1],       [16, 16, 1],             1,           2,                 [1, 16, 1, 16],      [8, 8, 1],        "Intrawave",         1),

   19: kernelInstance( 256,     16,    64,   512,  16,  16,  16,   16,    1,    1,    [32, 8, 1],        [32, 8, 1],              1,           1,                 [1, 16, 1, 16],      [4, 4, 1],        "Intrawave",         2),
   20: kernelInstance( 256,     16,   128,   512,  16,  16,  16,   16,    1,    2,    [32, 8, 1],        [32, 8, 1],              1,           2,                 [1, 16, 1, 16],      [8, 8, 1],        "Intrawave",         2),
   21: kernelInstance( 256,     16,   256,   512,  16,  16,  16,   16,    1,    4,    [32, 8, 1],        [32, 8, 1],              1,           2,                 [1, 16, 1, 16],      [8, 8, 1],        "Intrawave",         2),
   22: kernelInstance( 128,     32,    16,   512,  16,  16,  16,   16,    1,    1,    [32, 4, 1],        [32, 4, 1],              1,           1,                 [1, 32, 1, 4],       [4, 4, 1],        "Intrawave",         2),
   23: kernelInstance( 128,     16,    32,   128,  16,  16,  16,   16,    1,    1,    [8, 16, 1],        [8, 16, 1],              1,           1,                 [1, 16, 1, 8],       [4, 4, 1],        "Intrawave",         2),
   24: kernelInstance( 128,     16,    32,   512,  16,  16,  16,   16,    1,    1,    [32, 4, 1],        [32, 4, 1],              1,           1,                 [1, 16, 1, 8],       [4, 4, 1],        "Intrawave",         2),
   25: kernelInstance( 256,     16,    64,   512,  16,  16,  16,   16,    1,    1,    [32, 8, 1],        [32, 8, 1],              1,           1,                 [1, 16, 1, 16],      [4, 4, 1],        "Intrawave",         2),
   26: kernelInstance( 256,     32,    64,   512,  16,  16,  16,   16,    1,    2,    [32, 8, 1],        [32, 8, 1],              1,           2,                 [1, 32, 1, 8],       [8, 8, 1],        "Intrawave",         2),
   27: kernelInstance( 256,     64,    16,   512,  16,  16,  16,   16,    1,    1,    [32, 8, 1],        [32, 8, 1],              1,           1,                 [1, 64, 1, 4],       [4, 4, 1],        "Intrawave",         2),
   28: kernelInstance( 256,     64,    16,   512,  16,  16,  16,   16,    1,    1,    [32, 8, 1],        [32, 8, 1],              1,           1,                 [1, 64, 1, 4],       [4, 4, 1],        "Intrawave",         2),
   29: kernelInstance( 256,     16,    64,   256,  16,  16,  16,   16,    1,    1,    [16, 16, 1],       [16, 16, 1],             1,           1,                 [1, 16, 1, 16],      [4, 4, 1],        "Intrawave",         2),
   30: kernelInstance( 256,     16,   128,   256,  16,  16,  16,   16,    1,    2,    [16, 16, 1],       [16, 16, 1],             1,           2,                 [1, 16, 1, 16],      [8, 8, 1],        "Intrawave",         2),
   31: kernelInstance( 256,     16,   256,   256,  16,  16,  16,   16,    1,    4,    [16, 16, 1],       [16, 16, 1],             1,           2,                 [1, 16, 1, 16],      [8, 8, 1],        "Intrawave",         2),
   32: kernelInstance( 256,     16,   512,   256,  16,  16,  16,   16,    1,    8,    [16, 16, 1],       [16, 16, 1],             1,           2,                 [1, 16, 1, 16],      [8, 8, 1],        "Intrawave",         2),

   33: kernelInstance( 256,     256,  256,   128,  16,  16,  16,   16,   16,    4,    [8, 32, 1],        [8, 32, 1],              2,           1,                 [1, 32, 1, 8],       [8, 8, 1],         "Intrawave",         3),
   34: kernelInstance( 256,     256,  224,   128,  16,  16,  16,   16,    8,    7,    [8, 32, 1],        [8, 32, 1],              2,           1,                 [1, 64, 1, 4],       [8, 8, 1],         "Intrawave",         3),
   35: kernelInstance( 256,     256,  192,   128,  16,  16,  16,   16,   16,    3,    [8, 32, 1],        [8, 32, 1],              2,           1,                 [1, 32, 1, 8],       [8, 8, 1],         "Intrawave",         3),
   36: kernelInstance( 256,     256,  160,   128,  16,  16,  16,   16,    8,    5,    [8, 32, 1],        [8, 32, 1],              2,           1,                 [1, 64, 1, 4],       [8, 8, 1],         "Intrawave",         3),
   37: kernelInstance( 256,     256,  128,   128,  16,  16,  16,   16,   16,    2,    [8, 32, 1],        [8, 32, 1],              2,           1,                 [1, 32, 1, 8],       [8, 8, 1],         "Intrawave",         3),
   38: kernelInstance( 256,     256,   96,   128,  16,  16,  16,   16,    8,    3,    [8, 32, 1],        [8, 32, 1],              2,           1,                 [1, 64, 1, 4],       [8, 8, 1],         "Intrawave",         3),
   39: kernelInstance( 256,     256,   64,   128,  16,  16,  16,   16,   16,    1,    [8, 32, 1],        [8, 32, 1],              2,           1,                 [1, 32, 1, 8],       [8, 8, 1],         "Intrawave",         3),
   40: kernelInstance( 256,     224,  256,   128,  16,  16,  16,   16,   14,    4,    [8, 32, 1],        [8, 32, 1],              2,           1,                 [1, 32, 1, 8],       [8, 8, 1],         "Intrawave",         3),
   41: kernelInstance( 256,     224,  224,   128,  16,  16,  16,   16,    7,    7,    [8, 32, 1],        [8, 32, 1],              1,           1,                 [1, 32, 1, 8],       [4, 4, 1],         "Intrawave",         3),
   42: kernelInstance( 256,     224,  192,   128,  16,  16,  16,   16,   14,    3,    [8, 32, 1],        [8, 32, 1],              2,           1,                 [1, 32, 1, 8],       [8, 8, 1],         "Intrawave",         3),
   43: kernelInstance( 256,     224,  160,   128,  16,  16,  16,   16,    7,    5,    [8, 32, 1],        [8, 32, 1],              1,           1,                 [1, 32, 1, 8],       [4, 4, 1],         "Intrawave",         3),
   44: kernelInstance( 256,     224,  128,   128,  16,  16,  16,   16,   14,    2,    [8, 32, 1],        [8, 32, 1],              2,           1,                 [1, 32, 1, 8],       [8, 8, 1],         "Intrawave",         3),
   45: kernelInstance( 256,     224,   96,   128,  16,  16,  16,   16,    7,    3,    [8, 32, 1],        [8, 32, 1],              1,           1,                 [1, 32, 1, 8],       [4, 4, 1],         "Intrawave",         3),
   46: kernelInstance( 256,     224,   64,   128,  16,  16,  16,   16,   14,    1,    [8, 32, 1],        [8, 32, 1],              2,           1,                 [1, 32, 1, 8],       [8, 8, 1],         "Intrawave",         3),
   47: kernelInstance( 256,     192,  256,   128,  16,  16,  16,   16,   12,    4,    [8, 32, 1],        [8, 32, 1],              2,           1,                 [1, 32, 1, 8],       [8, 8, 1],         "Intrawave",         3),
   48: kernelInstance( 256,     192,  224,   128,  16,  16,  16,   16,    6,    7,    [8, 32, 1],        [8, 32, 1],              2,           1,                 [1, 64, 1, 4],       [8, 8, 1],         "Intrawave",         3),
   49: kernelInstance( 256,     192,  192,   128,  16,  16,  16,   16,   12,    3,    [8, 32, 1],        [8, 32, 1],              2,           1,                 [1, 32, 1, 8],       [8, 8, 1],         "Intrawave",         3),
   50: kernelInstance( 256,     192,  160,   128,  16,  16,  16,   16,    6,    5,    [8, 32, 1],        [8, 32, 1],              2,           1,                 [1, 64, 1, 4],       [8, 8, 1],         "Intrawave",         3),
   51: kernelInstance( 256,     192,  128,   128,  16,  16,  16,   16,   12,    2,    [8, 32, 1],        [8, 32, 1],              2,           1,                 [1, 32, 1, 8],       [8, 8, 1],         "Intrawave",         3),
   52: kernelInstance( 256,     192,   96,   128,  16,  16,  16,   16,    6,    3,    [8, 32, 1],        [8, 32, 1],              2,           1,                 [1, 64, 1, 4],       [8, 8, 1],         "Intrawave",         3),
   53: kernelInstance( 256,     192,   64,   128,  16,  16,  16,   16,   12,    1,    [8, 32, 1],        [8, 32, 1],              2,           1,                 [1, 32, 1, 8],       [8, 8, 1],         "Intrawave",         3),
   54: kernelInstance( 256,     160,  256,   128,  16,  16,  16,   16,   10,    4,    [8, 32, 1],        [8, 32, 1],              2,           1,                 [1, 32, 1, 8],       [8, 8, 1],         "Intrawave",         3),
   55: kernelInstance( 256,     160,  224,   128,  16,  16,  16,   16,    5,    7,    [8, 32, 1],        [8, 32, 1],              1,           1,                 [1, 32, 1, 8],       [4, 4, 1],         "Intrawave",         3),
   56: kernelInstance( 256,     160,  192,   128,  16,  16,  16,   16,   10,    3,    [8, 32, 1],        [8, 32, 1],              2,           1,                 [1, 32, 1, 8],       [8, 8, 1],         "Intrawave",         3),
   57: kernelInstance( 256,     160,  160,   128,  16,  16,  16,   16,    5,    5,    [8, 32, 1],        [8, 32, 1],              1,           1,                 [1, 32, 1, 8],       [4, 4, 1],         "Intrawave",         3),
   58: kernelInstance( 256,     160,  128,   128,  16,  16,  16,   16,   10,    2,    [8, 32, 1],        [8, 32, 1],              2,           1,                 [1, 32, 1, 8],       [8, 8, 1],         "Intrawave",         3),
   59: kernelInstance( 256,     160,   96,   128,  16,  16,  16,   16,    5,    3,    [8, 32, 1],        [8, 32, 1],              1,           1,                 [1, 32, 1, 8],       [4, 4, 1],         "Intrawave",         3),
   60: kernelInstance( 256,     160,   64,   128,  16,  16,  16,   16,   10,    1,    [8, 32, 1],        [8, 32, 1],              2,           1,                 [1, 32, 1, 8],       [8, 8, 1],         "Intrawave",         3),
   61: kernelInstance( 256,     128,   96,   128,  16,  16,  16,   16,    4,    3,    [8, 32, 1],        [8, 32, 1],              2,           1,                 [1, 64, 1, 4],       [8, 8, 1],         "Intrawave",         3),
   62: kernelInstance( 256,     128,   64,   128,  16,  16,  16,   16,    8,    1,    [8, 32, 1],        [8, 32, 1],              2,           1,                 [1, 32, 1, 8],       [8, 8, 1],         "Intrawave",         3),
   63: kernelInstance( 256,     128,  128,   256,  16,  16,  16,   16,    8,    2,    [16, 16, 1],       [16, 16, 1],             2,           1,                 [1, 32, 1, 8],       [8, 8, 1],         "Intrawave",         3),
   64: kernelInstance( 256,     128,   96,   256,  16,  16,  16,   16,    4,    3,    [16, 16, 1],       [16, 16, 1],             2,           1,                 [1, 64, 1, 4],       [8, 8, 1],         "Intrawave",         3),
   65: kernelInstance( 256,     128,   64,   256,  16,  16,  16,   16,    8,    1,    [16, 16, 1],       [16, 16, 1],             2,           1,                 [1, 32, 1, 8],       [8, 8, 1],         "Intrawave",         3),
   66: kernelInstance( 256,     128,  256,   128,  16,  16,  16,   16,    8,    4,    [8, 32, 1],        [8, 32, 1],              2,           1,                 [1, 32, 1, 8],       [8, 8, 1],         "Intrawave",         3),
   67: kernelInstance( 256,     128,  224,   128,  16,  16,  16,   16,    4,    7,    [8, 32, 1],        [8, 32, 1],              2,           1,                 [1, 64, 1, 4],       [8, 8, 1],         "Intrawave",         3),
   68: kernelInstance( 256,     128,  192,   128,  16,  16,  16,   16,    8,    3,    [8, 32, 1],        [8, 32, 1],              2,           1,                 [1, 32, 1, 8],       [8, 8, 1],         "Intrawave",         3),
   69: kernelInstance( 256,     128,  160,   128,  16,  16,  16,   16,    4,    5,    [8, 32, 1],        [8, 32, 1],              2,           1,                 [1, 64, 1, 4],       [8, 8, 1],         "Intrawave",         3),
   70: kernelInstance( 256,     128,  128,   128,  16,  16,  16,   16,    8,    2,    [8, 32, 1],        [8, 32, 1],              2,           1,                 [1, 32, 1, 8],       [8, 8, 1],         "Intrawave",         3),
   71: kernelInstance( 256,     128,  128,   64,  16,  16,  16,   16,    8,    2,    [4, 64, 1],        [4, 64, 1],              2,           1,                 [1, 32, 1, 8],       [8, 8, 1],         "Intrawave",         3),
   72: kernelInstance( 128,     32,  256,   64,  16,  16,  16,   16,    2,    8,    [4, 32, 1],        [4, 32, 1],              1,           4,                 [1, 8, 1, 16],       [8, 8, 1],         "Intrawave",         1),
   73: kernelInstance( 256,     64,  256,   64,  16,  16,  16,   16,    4,    4,    [4, 64, 1],        [4, 64, 1],              1,           2,                 [1, 16, 1, 16],       [8, 8, 1],         "Intrawave",         1),
   74: kernelInstance( 64,     16,  256,   64,  16,  16,  16,   16,    1,    16,    [4, 16, 1],        [4, 16, 1],              1,           8,                 [1, 8, 1, 8],       [8, 8, 1],         "Intrawave",         1),
    # clang-format on
}
# '''



default_kernels_dict = {
    # clang-format off
        ##############| Block|   MPer|  NPer|  KPer| AK1| BK1|MPer| NPer| MXdl| NXdl|  ABlockTransfer|  BBlockTransfer|    CShuffle|    CShuffle|     CBlockTransferClusterLengths|  CBlockTransfer|  Block-wiseGemm|     Block-wiseGemm|
        ###############| Size|  Block| Block| Block|    |    | XDL|  XDL|  Per|  Per|   ThreadCluster|   ThreadCluster| MXdlPerWave| NXdlPerWave| _MBlock_MXdlPerWave_MWaveMPerXdl| ScalarPerVector|    Pipeline    |           Pipeline|
        ###############|     |       |      |      |    |    |    |     | Wave| Wave| Lengths_K0_M_K1| Lengths_K0_N_K1|  PerShuffle|  PerShuffle| _NBlock_NXdlPerWave_NWaveNPerXdl|   _NWaveNPerXdl|    Scheduler   |           Verision|
        ###############|     |       |      |      |    |    |    |     |     |     |                |                |            |            |                                 |                |                |                   |

        # Compute friendly
    (-1): kernelInstance( 256, 128,   128,   128,   16,  16,   16,   16,    8,    2,     [8, 32, 1],        [8, 32, 1],            2,           1,                 [1, 32, 1, 8],            [8, 8, 1],        "Intrawave",         3 ),
    (-2): kernelInstance( 128, 16,    32,    128,   16,  16,   16,   16,    1,    1,     [8, 16, 1],        [8, 16, 1],             1,          1,                 [1, 16, 1, 8],            [4, 4, 1],        "Intrawave",         1),
    (-3): kernelInstance( 128, 32,    256,   64,    16,  16,   16,   16,    2,    8,     [4, 32, 1],        [4, 32, 1],             1,          4,                 [1, 8, 1, 16],            [8, 8, 1],         "Intrawave",         1)
    # (-2): kernelInstance( 256, 128,   128,   256,   16,  16,   16,   16,    4,    4,     [16, 16, 1],       [16, 16, 1],           1,           2,                 [1, 32, 1, 8],            [8, 8, 1],        "Intrawave",         3 ),
    # (-3): kernelInstance( 256, 256,   256,   128,   16,  16,   16,   16,    8,    8,     [8, 32, 1],        [8, 32, 1],            1,           2,                 [1, 32, 1, 8],            [8, 8, 1],        "Intrawave",         3),
    # (-4): kernelInstance( 256, 256,   128,   128,   16,  16,   16,   16,    8,    4,     [8, 32, 1],        [8, 32, 1],            1,           2,                 [1, 32, 1, 8],            [8, 8, 1],        "Intrawave",         3),
    # (-5): kernelInstance( 256, 192,   128,  128,    16,  16,   16,   16,    6,    4,     [8, 32, 1],        [8, 32, 1],            1,           2,                 [1, 32, 1, 8],            [8, 8, 1],        "Intrawave",         3)
}
# fmt: on
