# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices,Inc. All rights reserved.
from dataclasses import dataclass


@dataclass
class TileKernelInstance:
    M_Tile: int
    N_Tile: int
    K_Tile: int
    M_Warp: int
    N_Warp: int
    K_Warp: int
    M_Warp_Tile: int
    N_Warp_Tile: int
    K_Warp_Tile: int
    
    kPadM: bool
    kPadN: bool
    kPadK: bool
    
    Scheduler: str # Default, Intrawave, Interwave
    
    TiledMMAPermuteN: bool
    TransposeC: bool
    DoubleSmemBuffer: bool
    UsePersistentKernel: bool
    
    BlockPerCu: int # 1,2
    
    @property
    def name(self) -> str:
        """
        Generate a unique name for the kernel instance based on its parameters.        
        """
        
        return ("_").join(
            [
                "a8w8_blockscale_tile",
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
                ("x").join(
                    map(
                        lambda x: str(int(x)),
                        [self.kPadM, self.kPadN, self.kPadK],
                    )
                ),
                self.Scheduler.lower(),
                ("x").join(
                    map(
                        lambda x: str(int(x)),
                        [self.TiledMMAPermuteN, self.TransposeC, self.DoubleSmemBuffer, self.UsePersistentKernel],
                    )
                ),
                str(self.BlockPerCu)
            ]
        )


# fmt: off
# Candidate and default kernel instances for tile gemm a8w8 blockscale
# These instances are used for generating the kernel code and tuning.
candidate_kernels_dict_tile = {
    #######################| M_Tile | N_Tile | K_Tile | M_Warp | N_Warp | K_Warp | M_Warp_Tile | N_Warp_Tile | K_Warp_Tile | kPadM | kPadN | kPadK |   Scheduler   | TiledMMAPermuteN |  TransposeC | DoubleSmemBuffer | UsePersistentKernel | BlockPerCu |
    # K_Tile = 128
    0:   TileKernelInstance(  256,     128,      128,     1,        4,       1,        16,            16,          128,      False,  False,  False,   "Intrawave",        False,             False,        False,               False,             2      ),
    1:   TileKernelInstance(   96,     256,      128,     1,        4,       1,        16,            16,          128,      False,  False,  False,   "Intrawave",        False,             False,        False,               False,             1      ),
    2:   TileKernelInstance(  128,     256,      128,     1,        4,       1,        16,            16,          128,      False,  False,  False,   "Intrawave",        False,             False,        False,               False,             1      ),
    3:   TileKernelInstance(  192,     256,      128,     1,        4,       1,        16,            16,          128,      False,  False,  False,   "Intrawave",        False,             False,        False,               False,             1      ),
    4:   TileKernelInstance(  256,     256,      128,     1,        4,       1,        16,            16,          128,      False,  False,  False,   "Intrawave",        False,             False,        False,               False,             1      ),
    5:   TileKernelInstance(   96,     256,      128,     1,        4,       1,        16,            16,          128,      False,  False,  False,   "Intrawave",        False,             False,        False,               False,             2      ),
    6:   TileKernelInstance(  128,     256,      128,     1,        4,       1,        16,            16,          128,      False,  False,  False,   "Intrawave",        False,             False,        False,               False,             2      ),
    7:   TileKernelInstance(  192,     256,      128,     1,        4,       1,        16,            16,          128,      False,  False,  False,   "Intrawave",        False,             False,        False,               False,             2      ),
    8:   TileKernelInstance(  256,     256,      128,     1,        4,       1,        16,            16,          128,      False,  False,  False,   "Intrawave",        False,             False,        False,               False,             2      ),
    # K_Tile = 256
    9:   TileKernelInstance(   16,     256,      256,     1,        4,       1,        16,            16,          128,      False,  False,  False,   "Intrawave",        False,             False,        False,               False,             1      ),
    10:  TileKernelInstance(   32,     256,      256,     1,        4,       1,        16,            16,          128,      False,  False,  False,   "Intrawave",        False,             False,        False,               False,             1      ),
    11:  TileKernelInstance(   48,     256,      256,     1,        4,       1,        16,            16,          128,      False,  False,  False,   "Intrawave",        False,             False,        False,               False,             1      ),
    12:  TileKernelInstance(   64,     256,      256,     1,        4,       1,        16,            16,          128,      False,  False,  False,   "Intrawave",        False,             False,        False,               False,             1      ),
    13:  TileKernelInstance(   80,     256,      256,     1,        4,       1,        16,            16,          128,      False,  False,  False,   "Intrawave",        False,             False,        False,               False,             1      ),
    14:  TileKernelInstance(   96,     256,      256,     1,        4,       1,        16,            16,          128,      False,  False,  False,   "Intrawave",        False,             False,        False,               False,             1      ),
    15:  TileKernelInstance(  112,     256,      256,     1,        4,       1,        16,            16,          128,      False,  False,  False,   "Intrawave",        False,             False,        False,               False,             1      ),
    16:  TileKernelInstance(  128,     256,      256,     1,        4,       1,        16,            16,          128,      False,  False,  False,   "Intrawave",        False,             False,        False,               False,             1      ),
    17:  TileKernelInstance(   16,     256,      256,     1,        4,       1,        16,            16,          128,      False,  False,  False,   "Intrawave",        False,             False,        False,               False,             2      ),
    18:  TileKernelInstance(   32,     256,      256,     1,        4,       1,        16,            16,          128,      False,  False,  False,   "Intrawave",        False,             False,        False,               False,             2      ),
    19:  TileKernelInstance(   48,     256,      256,     1,        4,       1,        16,            16,          128,      False,  False,  False,   "Intrawave",        False,             False,        False,               False,             2      ),
    20:  TileKernelInstance(   64,     256,      256,     1,        4,       1,        16,            16,          128,      False,  False,  False,   "Intrawave",        False,             False,        False,               False,             2      ),
    21:  TileKernelInstance(   80,     256,      256,     1,        4,       1,        16,            16,          128,      False,  False,  False,   "Intrawave",        False,             False,        False,               False,             2      ),
    22:  TileKernelInstance(   96,     256,      256,     1,        4,       1,        16,            16,          128,      False,  False,  False,   "Intrawave",        False,             False,        False,               False,             2      ),
    23:  TileKernelInstance(  112,     256,      256,     1,        4,       1,        16,            16,          128,      False,  False,  False,   "Intrawave",        False,             False,        False,               False,             2      ),
    24:  TileKernelInstance(  128,     256,      256,     1,        4,       1,        16,            16,          128,      False,  False,  False,   "Intrawave",        False,             False,        False,               False,             2      ),
}


default_kernels_dict_tile = {
   #######################| M_Tile | N_Tile | K_Tile | M_Warp | N_Warp | K_Warp | M_Warp_Tile | N_Warp_Tile | K_Warp_Tile | kPadM | kPadN | kPadK |   Scheduler   | TiledMMAPermuteN |  TransposeC | DoubleSmemBuffer | UsePersistentKernel | BlockPerCu |
    -1:  TileKernelInstance(   16,     128,      256,     1,        4,       1,        16,            16,           32,      False,  False,  False,   "Intrawave",        False,             False,        False,               False,             1      ), 
}
# fmt: on
