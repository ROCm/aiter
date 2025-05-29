# SPDX-License-Identifier: MIT
# Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.
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

# TODO - more kernel
kernels_list = {}


default_kernels_dict = {
    # clang-format off
    ##############| Block|  MPer|  NPer|  KPer| AK1| BK1|MPer| NPer| MXdl| NXdl|  ABlockTransfer|  BBlockTransfer|    CShuffle|    CShuffle|     CBlockTransferClusterLengths|  CBlockTransfer|  Block-wiseGemm|     Block-wiseGemm|
    ###############| Size| Block| Block| Block|    |    | XDL|  XDL|  Per|  Per|   ThreadCluster|   ThreadCluster| MXdlPerWave| NXdlPerWave| _MBlock_MXdlPerWave_MWaveMPerXdl| ScalarPerVector|    Pipeline    |           Pipeline|
    ###############|     |      |      |      |    |    |    |     | Wave| Wave| Lengths_K0_M_K1| Lengths_K0_N_K1|  PerShuffle|  PerShuffle| _NBlock_NXdlPerWave_NWaveNPerXdl|   _NWaveNPerXdl|    Scheduler   |           Verision|
    ###############|     |      |      |      |    |    |    |     |     |     |                |                |            |            |                                 |                |                |                   |
    # Compute friendly
    (-1): kernelInstance(
        256,
        128,
        512,
        128,
        16,
        16,
        16,
        16,
        8,
        8,
        [8, 32, 1],
        [8, 32, 1],
        2,
        2,
        [1, 32, 1, 8],
        8,
        "Intrawave",
        3,
    ),
}
