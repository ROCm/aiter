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
    ABLOCK_TRANSFER: list[int]
    BBLOCK_TRANSFER: list[int]
    CBLOCK_TRANSFER: list[int]
    CBLOCK_SPV: list[int]
    PIPELINE_Sched: str
    PIPELINE_VERSION: int

    @property
    def name(self) -> str:
        return ("_").join([
            "a8w8_blockscale",
            ("x").join(map(lambda x: str(x), [
                self.BLOCK_SIZE, self.MPerBLOCK, self.NPerBLOCK, self.KPerBLOCK])),
            ("x").join(map(lambda x: str(x), [
                self.AK1, self.BK1])),
            ("x").join(map(lambda x: str(x), [
                self.MPerXDL, self.NPerXDL])),
            ("x").join(map(lambda x: str(x), self.ABLOCK_TRANSFER)),
            ("x").join(map(lambda x: str(x), self.BBLOCK_TRANSFER)),
            ("x").join(map(lambda x: str(x), self.CBLOCK_TRANSFER)),
            ("x").join(map(lambda x: str(x), self.CBLOCK_SPV)),
            self.PIPELINE_Sched.lower(),
            f"v{self.PIPELINE_VERSION}"
        ])

kernels_list = {
    # clang-format off
        ##############| Block| Scale| Scale| Scale|  MPer|  NPer|  KPer| AK1| BK1|MPer| NPer| MXdl| NXdl|  ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockLds|  BBlockTransfer| BBlockTransfer| BBlockTransfer| BlockTransfer| BBlockTransfer| BBlockTransfer| BBlockLds|    CShuffle|    CShuffle|     CBlockTransferClusterLengths|  CBlockTransfer|  Block-wiseGemm|     Block-wiseGemm|
        ###############| Size| Block| Block| Block| Block| Block| Block|    |    | XDL|  XDL|  Per|  Per|   ThreadCluster|  ThreadCluster| SrcAccessOrder|   SrcVectorDim|      SrcScalar|      DstScalar| AddExtraM|   ThreadCluster|  ThreadCluster| SrcAccessOrder|  SrcVectorDim|      SrcScalar|      DstScalar| AddExtraN| MXdlPerWave| NXdlPerWave| _MBlock_MXdlPerWave_MWaveMPerXdl| ScalarPerVector|    Pipeline    |           Pipeline|
        ###############|     |     M|     N|     K|      |      |      |    |    |    |     | Wave| Wave| Lengths_K0_M_K1|   ArrangeOrder|               |               |      PerVector|   PerVector_K1|          | Lengths_K0_N_K1|   ArrangeOrder|               |              |      PerVector|   PerVector_K1|          |  PerShuffle|  PerShuffle| _NBlock_NXdlPerWave_NWaveNPerXdl|   _NWaveNPerXdl|    Scheduler   |           Verision|
        ###############|     |      |      |      |      |      |      |    |    |    |     |     |     |                |               |               |               |               |               |          |                |               |               |              |               |               |          |            |            |                                 |                |                |                   |

        # Compute friendly
    0: kernelInstance(  256,   1,   128,   128,   128,   128,   128,   16,  16,   32,   32,    2,    2,     S<8, 32, 1>,     S<1, 0, 2>,    S<1, 0, 2>,               2,             16,             16,          0,    S<8, 32, 1>,     S<1, 0, 2>,    S<1, 0, 2>,               2,             16,             16,          0,          1,           1,                   S<1, 32, 1, 8>,     S<8>,  "Intrawave", 3 ),
    1: kernelInstance(  256,   1,   128,   128,   128,    64,   128,   16,  16,   32,   32,    2,    1,     S<8, 32, 1>,     S<1, 0, 2>,    S<1, 0, 2>,               2,             16,             16,          0,    S<8, 32, 1>,     S<1, 0, 2>,    S<1, 0, 2>,               2,             16,             16,          0,          1,           1,                   S<1, 32, 1, 8>,     S<8>,  "Intrawave", 3 ),
    2: kernelInstance(  256,   1,   128,   128,    64,   128,   128,   16,  16,   32,   32,    1,    2,     S<8, 32, 1>,     S<1, 0, 2>,    S<1, 0, 2>,               2,             16,             16,          0,    S<8, 32, 1>,     S<1, 0, 2>,    S<1, 0, 2>,               2,             16,             16,          0,          1,           1,                   S<1, 32, 1, 8>,     S<8>,  "Intrawave", 3 ),
    3: kernelInstance(  256,   1,   128,   128,    64,    64,   128,   16,  16,   32,   32,    1,    1,     S<8, 32, 1>,     S<1, 0, 2>,    S<1, 0, 2>,               2,             16,             16,          0,    S<8, 32, 1>,     S<1, 0, 2>,    S<1, 0, 2>,               2,             16,             16,          0,          1,           1,                   S<1, 32, 1, 8>,     S<8>,  "Intrawave", 3 ),

        # Memory friendly 
    4:  kernelInstance( 256,   1,   128,   128,    16,   256,   128,  8,  16,     16,   16,    1,    4,     S<16, 16, 1>,     S<1, 0, 2>,    S<1, 0, 2>,              2,              8,              8,          0,    S<8, 32, 1>,     S<1, 0, 2>,    S<1, 0, 2>,              2,             16,             16,          0,          1,           2,                 S<1, 16, 1, 16>,            S<8>,  "Intrawave", 1, ),
    5:  kernelInstance( 256,   1,   128,   128,    16,   128,   128,  8,  16,     16,   16,    1,    2,     S<16, 16, 1>,     S<1, 0, 2>,    S<1, 0, 2>,              2,              8,              8,          0,    S<8, 32, 1>,     S<1, 0, 2>,    S<1, 0, 2>,              2,             16,             16,          0,          1,           2,                 S<1, 16, 1, 16>,            S<8>,  "Intrawave", 1, ),
    6:  kernelInstance( 256,   1,   128,   128,    16,    64,   128,  8,  16,     16,   16,    1,    1,     S<16, 16, 1>,     S<1, 0, 2>,    S<1, 0, 2>,              2,              8,              8,          0,    S<8, 32, 1>,     S<1, 0, 2>,    S<1, 0, 2>,              2,             16,             16,          0,          1,           1,                 S<1, 16, 1, 16>,            S<4>,  "Intrawave", 1, ),

    7:  kernelInstance( 256,   1,   128,   128,    16,   128,   256, 16,  16,     16,   16,    1,    2,     S<16, 16, 1>,     S<1, 0, 2>,    S<1, 0, 2>,              2,             16,             16,          0,    S<16, 16, 1>,     S<1, 0, 2>,    S<1, 0, 2>,             2,             16,             16,          0,          1,           2,                 S<1, 16, 1, 16>,            S<8>,  "Intrawave", 1, ),
    8:  kernelInstance( 256,   1,   128,   128,    16,    64,   256, 16,  16,     16,   16,    1,    1,     S<16, 16, 1>,     S<1, 0, 2>,    S<1, 0, 2>,              2,             16,             16,          0,    S<16, 16, 1>,     S<1, 0, 2>,    S<1, 0, 2>,             2,             16,             16,          0,          1,           1,                 S<1, 16, 1, 16>,            S<4>,  "Intrawave", 1, ),

    9:  kernelInstance( 256,   1,   128,   128,    32,   256,   128, 16,  16,     32,   32,    1,    2,     S<8, 32, 1>,     S<1, 0, 2>,    S<1, 0, 2>,               2,             16,             16,          0,    S<8, 32, 1>,     S<1, 0, 2>,    S<1, 0, 2>,              2,             16,             16,          0,          1,           1,                 S<1, 32, 1, 8>,            S<8>,  "Intrawave", 1, ),
    10: kernelInstance( 256,   1,   128,   128,    32,   128,   128, 16,  16,     32,   32,    1,    1,     S<8, 32, 1>,     S<1, 0, 2>,    S<1, 0, 2>,               2,             16,             16,          0,    S<8, 32, 1>,     S<1, 0, 2>,    S<1, 0, 2>,              2,             16,             16,          0,          1,           1,                 S<1, 32, 1, 8>,            S<8>,  "Intrawave", 1, ),
    11: kernelInstance( 256,   1,   128,   128,    32,    64,   128, 16,  16,     16,   16,    2,    1,     S<8, 32, 1>,     S<1, 0, 2>,    S<1, 0, 2>,               2,             16,             16,          0,    S<8, 32, 1>,     S<1, 0, 2>,    S<1, 0, 2>,              2,             16,             16,          0,          2,           1,                 S<1, 32, 1, 8>,            S<8>,  "Intrawave", 1, ),

    12: kernelInstance( 256,   1,   128,   128,    32,   128,   256, 16,  16,     32,   32,    1,    1,     S<16, 16, 1>,     S<1, 0, 2>,    S<1, 0, 2>,              2,             16,             16,          0,    S<16, 16, 1>,     S<1, 0, 2>,    S<1, 0, 2>,             2,             16,             16,          0,          1,           1,                 S<1, 32, 1, 8>,            S<8>,  "Intrawave", 1, ),
    13: kernelInstance( 256,   1,   128,   128,    32,    64,   256, 16,  16,     16,   16,    2,    1,     S<16, 16, 1>,     S<1, 0, 2>,    S<1, 0, 2>,              2,             16,             16,          0,    S<16, 16, 1>,     S<1, 0, 2>,    S<1, 0, 2>,             2,             16,             16,          0,          2,           1,                 S<1, 32, 1, 8>,            S<8>,  "Intrawave", 1, ),

    14: kernelInstance( 256,   1,   128,   128,    64,   256,   128, 16,  16,     32,   32,    2,    2,     S<8, 32, 1>,     S<1, 0, 2>,    S<1, 0, 2>,               2,             16,             16,          0,    S<8, 32, 1>,     S<1, 0, 2>,    S<1, 0, 2>,              2,             16,             16,          0,          1,           1,                 S<1, 32, 1, 8>,            S<8>,  "Intrawave", 1, ),
    15: kernelInstance( 256,   1,   128,   128,    64,   128,   128, 16,  16,     32,   32,    2,    1,     S<8, 32, 1>,     S<1, 0, 2>,    S<1, 0, 2>,               2,             16,             16,          0,    S<8, 32, 1>,     S<1, 0, 2>,    S<1, 0, 2>,              2,             16,             16,          0,          1,           1,                 S<1, 32, 1, 8>,            S<8>,  "Intrawave", 1, ),
    16: kernelInstance( 256,   1,   128,   128,    64,    64,   128, 16,  16,     32,   32,    1,    1,     S<8, 32, 1>,     S<1, 0, 2>,    S<1, 0, 2>,               2,             16,             16,          0,    S<8, 32, 1>,     S<1, 0, 2>,    S<1, 0, 2>,              2,             16,             16,          0,          1,           1,                 S<1, 32, 1, 8>,            S<8>,  "Intrawave", 1, ),

    17: kernelInstance( 256,   1,   128,   128,    64,   128,   256, 16,  16,     32,   32,    2,    1,     S<16, 16, 1>,     S<1, 0, 2>,    S<1, 0, 2>,              2,             16,             16,          0,    S<16, 16, 1>,     S<1, 0, 2>,    S<1, 0, 2>,             2,             16,             16,          0,          1,           1,                 S<1, 32, 1, 8>,            S<8>,  "Intrawave", 1, ),
    18: kernelInstance( 256,   1,   128,   128,    64,    64,   256, 16,  16,     32,   32,    1,    1,     S<16, 16, 1>,     S<1, 0, 2>,    S<1, 0, 2>,              2,             16,             16,          0,    S<16, 16, 1>,     S<1, 0, 2>,    S<1, 0, 2>,             2,             16,             16,          0,          1,           1,                 S<1, 32, 1, 8>,            S<8>,  "Intrawave", 1, )
    # clang-format on
}




default_kernels_dict = {
    # clang-format off
        ##############| Block| Scale| Scale| Scale|  MPer|  NPer|  KPer| AK1| BK1|MPer| NPer| MXdl| NXdl|  ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockLds|  BBlockTransfer| BBlockTransfer| BBlockTransfer| BlockTransfer| BBlockTransfer| BBlockTransfer| BBlockLds|    CShuffle|    CShuffle|     CBlockTransferClusterLengths|  CBlockTransfer|  Block-wiseGemm|     Block-wiseGemm|
        ###############| Size| Block| Block| Block| Block| Block| Block|    |    | XDL|  XDL|  Per|  Per|   ThreadCluster|  ThreadCluster| SrcAccessOrder|   SrcVectorDim|      SrcScalar|      DstScalar| AddExtraM|   ThreadCluster|  ThreadCluster| SrcAccessOrder|  SrcVectorDim|      SrcScalar|      DstScalar| AddExtraN| MXdlPerWave| NXdlPerWave| _MBlock_MXdlPerWave_MWaveMPerXdl| ScalarPerVector|    Pipeline    |           Pipeline|
        ###############|     |     M|     N|     K|      |      |      |    |    |    |     | Wave| Wave| Lengths_K0_M_K1|   ArrangeOrder|               |               |      PerVector|   PerVector_K1|          | Lengths_K0_N_K1|   ArrangeOrder|               |              |      PerVector|   PerVector_K1|          |  PerShuffle|  PerShuffle| _NBlock_NXdlPerWave_NWaveNPerXdl|   _NWaveNPerXdl|    Scheduler   |           Verision|
        ###############|     |      |      |      |      |      |      |    |    |    |     |     |     |                |               |               |               |               |               |          |                |               |               |              |               |               |          |            |            |                                 |                |                |                   |

        # Compute friendly
    (-1): kernelInstance( 256,   1,   128,   128,   16,   128,   256,   16,  16,   16,   16,    1,    2,     S<16, 16, 1>,     S<1, 0, 2>,    S<1, 0, 2>,               2,             16,             16,          0,    S<16, 16, 1>,     S<1, 0, 2>,    S<1, 0, 2>,               2,             16,             16,          0,          1,           2,                   S<1, 16, 1, 16>,     S<8>,  "Intrawave", 1 ),
    
}
