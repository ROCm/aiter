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
    CBLOCK_SPV: int
    PIPELINE_Sched: str
    PIPELINE_VERSION: int

    @property
    def name(self) -> str:
        return ("_").join(
            [
                "a4w4_blockscale",
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
                ("x").join(str(self.CBLOCK_SPV)),
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
kernels_list = {
    # clang-format off
    #################| Block|  MPer|  NPer|  KPer| AK1| BK1|MPer| NPer| MXdl| NXdl|  ABlockTransfer|  BBlockTransfer|    CShuffle|    CShuffle|     CBlockTransferClusterLengths|  CBlockTransfer|  Block-wiseGemm|     Block-wiseGemm|
    #################| Size | Block| Block| Block|    |    | XDL|  XDL|  Per|  Per|   ThreadCluster|   ThreadCluster| MXdlPerWave| NXdlPerWave| _MBlock_MXdlPerWave_MWaveMPerXdl| ScalarPerVector|    Pipeline    |           Pipeline|
    #################|      |      |      |      |    |    |    |     | Wave| Wave| Lengths_K0_M_K1| Lengths_K0_N_K1|  PerShuffle|  PerShuffle| _NBlock_NXdlPerWave_NWaveNPerXdl|   _NWaveNPerXdl|    Scheduler   |           Verision|
    #################|      |      |      |      |    |    |    |     |     |     |                |                |            |            |                                 |                |                |                   |
    0:  kernelInstance(256,     64,   128,    128,  16,  16,  16,   16,   4,    2,     [8, 32, 1],      [8, 32, 1],         2,           2,               [1, 32, 1, 8],                  8,          "Intrawave",             3,       ),
    1:  kernelInstance(256,     64,   256,    128,  16,  16,  16,   16,   4,    4,     [8, 32, 1],      [8, 32, 1],         2,           2,               [1, 32, 1, 8],                  8,          "Intrawave",             3,       ),
    2:  kernelInstance(256,     64,   384,    128,  16,  16,  16,   16,   4,    6,     [8, 32, 1],      [8, 32, 1],         2,           2,               [1, 32, 1, 8],                  8,          "Intrawave",             3,       ),
    3:  kernelInstance(256,     64,   512,    128,  16,  16,  16,   16,   4,    8,     [8, 32, 1],      [8, 32, 1],         2,           2,               [1, 32, 1, 8],                  8,          "Intrawave",             3,       ),
    4:  kernelInstance(256,     96,   128,    128,  16,  16,  16,   16,   6,    2,     [8, 32, 1],      [8, 32, 1],         2,           2,               [1, 32, 1, 8],                  8,          "Intrawave",             3,       ),
    5:  kernelInstance(256,     96,   256,    128,  16,  16,  16,   16,   6,    4,     [8, 32, 1],      [8, 32, 1],         2,           2,               [1, 32, 1, 8],                  8,          "Intrawave",             3,       ),
    6:  kernelInstance(256,     96,   384,    128,  16,  16,  16,   16,   6,    6,     [8, 32, 1],      [8, 32, 1],         2,           2,               [1, 32, 1, 8],                  8,          "Intrawave",             3,       ),
    7:  kernelInstance(256,     96,   512,    128,  16,  16,  16,   16,   6,    8,     [8, 32, 1],      [8, 32, 1],         2,           2,               [1, 32, 1, 8],                  8,          "Intrawave",             3,       ),
    8:  kernelInstance(256,    128,   128,    128,  16,  16,  16,   16,   8,    2,     [8, 32, 1],      [8, 32, 1],         2,           2,               [1, 32, 1, 8],                  8,          "Intrawave",             3,       ),
    9:  kernelInstance(256,    128,   256,    128,  16,  16,  16,   16,   8,    4,     [8, 32, 1],      [8, 32, 1],         2,           2,               [1, 32, 1, 8],                  8,          "Intrawave",             3,       ),
    10: kernelInstance(256,    128,   384,    128,  16,  16,  16,   16,   8,    6,     [8, 32, 1],      [8, 32, 1],         2,           2,               [1, 32, 1, 8],                  8,          "Intrawave",             3,       ),
    11: kernelInstance(256,    128,   512,    128,  16,  16,  16,   16,   8,    8,     [8, 32, 1],      [8, 32, 1],         2,           2,               [1, 32, 1, 8],                  8,          "Intrawave",             3,       ),
    12: kernelInstance(256,     64,   128,    128,  16,  16,  16,   16,   4,    2,     [8, 32, 1],      [8, 32, 1],         2,           2,               [1, 16, 1, 16],                 8,          "Intrawave",             3,       ),    
    13: kernelInstance(256,     96,   128,    128,  16,  16,  16,   16,   6,    2,     [8, 32, 1],      [8, 32, 1],         2,           2,               [1, 16, 1, 16],                 8,          "Intrawave",             3,       ),
    }


default_kernels_dict = {
    # clang-format off
    ##################| Block|  MPer|  NPer|  KPer| AK1| BK1|MPer| NPer| MXdl| NXdl|  ABlockTransfer|  BBlockTransfer|    CShuffle|    CShuffle|     CBlockTransferClusterLengths|  CBlockTransfer|  Block-wiseGemm|     Block-wiseGemm|
    ##################| Size| Block| Block| Block|    |    | XDL|  XDL|  Per|  Per|   ThreadCluster|   ThreadCluster| MXdlPerWave| NXdlPerWave| _MBlock_MXdlPerWave_MWaveMPerXdl| ScalarPerVector|    Pipeline    |           Pipeline|
    ##################|     |      |      |      |    |    |    |     | Wave| Wave| Lengths_K0_M_K1| Lengths_K0_N_K1|  PerShuffle|  PerShuffle| _NBlock_NXdlPerWave_NWaveNPerXdl|   _NWaveNPerXdl|    Scheduler   |           Verision|
    ##################|     |      |      |      |    |    |    |     |     |     |                |                |            |            |                                 |                |                |                   |
    -1:  kernelInstance(256,     64,   128,    128,  16,  16,  16,   16,   4,    2,     [8, 32, 1],      [8, 32, 1],         2,           2,               [1, 32, 1, 8],                  8,          "Intrawave",             3,     ),
}
# fmt: on
