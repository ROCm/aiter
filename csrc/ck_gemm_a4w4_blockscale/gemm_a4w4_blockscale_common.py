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

@dataclass
class tileKernelInstance:
    M_Tile: int
    N_Tile: int
    K_Tile: int
    M_Warp: int
    N_Warp: int
    K_Warp: int
    M_Warp_Tile: int
    N_Warp_Tile: int
    K_Warp_Tile: int

    Scheduler: str  # Default, Intrawave, Interwave

    TiledMMAPermuteN: bool
    TransposeC: bool
    UsePersistentKernel: bool

    BlockPerCu: int  # 1..BLOCK_PER_CU_MAX

    # When True, 8-warp kernels read x_scale in row-major layout natively,
    # skipping the host-side transpose.
    AQRowMajor: bool = False

    @property
    def is_eight_warp(self) -> bool:
        return self.M_Warp * self.N_Warp * self.K_Warp == 8 and self.K_Warp_Tile == 128

    @property
    def name(self) -> str:
        """
        Generate a unique name for the kernel instance based on its parameters.
        """

        parts = [
            "a8w8_cktile",
            ("x").join(
                map(
                    lambda x: str(x),
                    [self.M_Tile, self.N_Tile, self.K_Tile],
                )
            ),
            ("x").join(
                map(
                    lambda x: str(x),
                    [self.M_Warp, self.N_Warp, self.K_Warp],
                )
            ),
            ("x").join(
                map(
                    lambda x: str(x),
                    [self.M_Warp_Tile, self.N_Warp_Tile, self.K_Warp_Tile],
                )
            ),
            self.Scheduler.lower(),
            ("x").join(
                map(
                    lambda x: str(int(x)),
                    [
                        self.TiledMMAPermuteN,
                        self.TransposeC,
                        self.UsePersistentKernel,
                    ],
                )
            ),
            str(self.BlockPerCu),
        ]
        if self.AQRowMajor:
            parts.append("aqrm")
        return "_".join(parts)


BLOCK_PER_CU_MAX = 4


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
    14: kernelInstance(256,     32,   128,    128,  16,  16,  16,   16,   2,    2,     [8, 32, 1],      [8, 32, 1],         2,           2,               [1, 16, 1, 16],                 8,          "Intrawave",             3,       ),
    15: kernelInstance(256,     32,   256,    128,  16,  16,  16,   16,   2,    4,     [8, 32, 1],      [8, 32, 1],         2,           4,               [1, 8, 1, 32],                 8,          "Intrawave",             3,       ),
    16: kernelInstance(256,     32,   384,    128,  16,  16,  16,   16,   2,    6,     [8, 32, 1],      [8, 32, 1],         2,           2,               [1, 16, 1, 16],                 8,          "Intrawave",             3,       ),
    17: kernelInstance(256,     32,   512,    128,  16,  16,  16,   16,   2,    8,     [8, 32, 1],      [8, 32, 1],         2,           4,               [1,8, 1, 32],                 8,          "Intrawave",             3,       ),
    18: kernelInstance(256,     256,  256,    128,  16,  16,  16,   16,   16,   4,     [8, 32, 1],      [8, 32, 1],         2,           4,               [1,8, 1, 32],                 8,          "Intrawave",             3,       ),
    19: kernelInstance(256,     256,  256,    128,  16,  16,  16,   16,   8,    8,     [8, 32, 1],      [8, 32, 1],         2,           8,               [1,8, 1, 32],                 8,          "Intrawave",             3,       ),
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

kernels_list_cktile = {
    ######################| M_Tile | N_Tile | K_Tile | M_Warp | N_Warp | K_Warp | M_Warp_Tile | N_Warp_Tile | K_Warp_Tile |   Scheduler   | TiledMMAPermuteN |  TransposeC | UsePersistentKernel | BlockPerCu |
    0:  tileKernelInstance(   128,     128,      128,     1,        4,       1,        32,            32,           128,      "Intrawave",        False,             True,           False,             1      ),

}

default_kernels_dict_cktile = {
    #########################| M_Tile | N_Tile | K_Tile | M_Warp | N_Warp | K_Warp | M_Warp_Tile | N_Warp_Tile | K_Warp_Tile |   Scheduler   | TiledMMAPermuteN |  TransposeC | UsePersistentKernel | BlockPerCu |
    (-1):  tileKernelInstance(   128,     128,      128,     1,        4,       1,        32,            32,           128,      "Intrawave",        False,             True,           False,             1      ),
}

# Name-keyed reverse lookup so codegen can filter the tuned CSV by kernelName,
# matching what the C++ runtime dispatcher uses.
kernels_by_name = {v.name: v for v in kernels_list.values()}
