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

    Scheduler: str  # Default, Intrawave, Interwave

    TiledMMAPermuteN: bool
    TransposeC: bool
    DoubleSmemBuffer: bool
    UsePersistentKernel: bool

    BlockPerCu: int  # 1,2

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
                self.Scheduler.lower(),
                ("x").join(
                    map(
                        lambda x: str(int(x)),
                        [
                            self.TiledMMAPermuteN,
                            self.TransposeC,
                            self.DoubleSmemBuffer,
                            self.UsePersistentKernel,
                        ],
                    )
                ),
                str(self.BlockPerCu),
            ]
        )


# fmt: off
# Candidate and default kernel instances for tile gemm a8w8 blockscale
# These instances are used for generating the kernel code and tuning.
candidate_kernels_dict_tile = {
    #######################| M_Tile | N_Tile | K_Tile | M_Warp | N_Warp | K_Warp | M_Warp_Tile | N_Warp_Tile | K_Warp_Tile |   Scheduler   | TiledMMAPermuteN |  TransposeC | DoubleSmemBuffer | UsePersistentKernel | BlockPerCu |
    # K_Tile = 128, M_Warp x N_Warp = 1 x 4, WarpTile = 16 x 16 x 32
    0:   TileKernelInstance(   16,     128,      128,     1,        4,       1,        16,            16,           32,      "Intrawave",        False,             False,        False,               False,             1      ),
    1:   TileKernelInstance(   32,     128,      128,     1,        4,       1,        16,            16,           32,      "Intrawave",        False,             False,        False,               False,             1      ),
    2:   TileKernelInstance(   64,     128,      128,     1,        4,       1,        16,            16,           32,      "Intrawave",        False,             False,        False,               False,             1      ),
    3:   TileKernelInstance(  128,     128,      128,     1,        4,       1,        16,            16,           32,      "Intrawave",        False,             False,        False,               False,             1      ),
    # K_Tile = 128, M_Warp x N_Warp = 2 x 2, WarpTile = 16 x 16 x 32
    4:   TileKernelInstance(   32,     128,      128,     2,        2,       1,        16,            16,           32,      "Intrawave",        False,             False,        False,               False,             1      ),
    5:   TileKernelInstance(   64,     128,      128,     2,        2,       1,        16,            16,           32,      "Intrawave",        False,             False,        False,               False,             1      ),
    6:   TileKernelInstance(  128,     128,      128,     2,        2,       1,        16,            16,           32,      "Intrawave",        False,             False,        False,               False,             1      ),
    # K_Tile = 128, M_Warp x N_Warp = 1 x 4, WarpTile = 32 x 32 x 16
    7:   TileKernelInstance(   32,     128,      128,     1,        4,       1,        32,            32,           16,      "Intrawave",        False,             False,        False,               False,             1      ),
    8:   TileKernelInstance(   64,     128,      128,     1,        4,       1,        32,            32,           16,      "Intrawave",        False,             False,        False,               False,             1      ),
    9:   TileKernelInstance(  128,     128,      128,     1,        4,       1,        32,            32,           16,      "Intrawave",        False,             False,        False,               False,             1      ),
    # K_Tile = 128, M_Warp x N_Warp = 2 x 2, WarpTile = 32 x 32 x 16
    10:  TileKernelInstance(   64,     128,      128,     2,        2,       1,        32,            32,           16,      "Intrawave",        False,             False,        False,               False,             1      ),
    11:  TileKernelInstance(  128,     128,      128,     2,        2,       1,        32,            32,           16,      "Intrawave",        False,             False,        False,               False,             1      ),
    # K_Tile = 256, M_Warp x N_Warp = 1 x 4, WarpTile = 16 x 16 x 32
    12:  TileKernelInstance(   16,     128,      256,     1,        4,       1,        16,            16,           32,      "Intrawave",        False,             False,        False,               False,             1      ),
    13:  TileKernelInstance(   32,     128,      256,     1,        4,       1,        16,            16,           32,      "Intrawave",        False,             False,        False,               False,             1      ),
    14:  TileKernelInstance(   64,     128,      256,     1,        4,       1,        16,            16,           32,      "Intrawave",        False,             False,        False,               False,             1      ),
    15:  TileKernelInstance(  128,     128,      256,     1,        4,       1,        16,            16,           32,      "Intrawave",        False,             False,        False,               False,             1      ),
    # K_Tile = 256, M_Warp x N_Warp = 2 x 2, WarpTile = 16 x 16 x 32
    16:  TileKernelInstance(   32,     128,      256,     2,        2,       1,        16,            16,           32,      "Intrawave",        False,             False,        False,               False,             1      ),
    17:  TileKernelInstance(   64,     128,      256,     2,        2,       1,        16,            16,           32,      "Intrawave",        False,             False,        False,               False,             1      ),
    18:  TileKernelInstance(  128,     128,      256,     2,        2,       1,        16,            16,           32,      "Intrawave",        False,             False,        False,               False,             1      ),
    # K_Tile = 256, M_Warp x N_Warp = 1 x 4, WarpTile = 32 x 32 x 16
    19:  TileKernelInstance(   32,     128,      256,     1,        4,       1,        32,            32,           16,      "Intrawave",        False,             False,        False,               False,             1      ),
    20:  TileKernelInstance(   64,     128,      256,     1,        4,       1,        32,            32,           16,      "Intrawave",        False,             False,        False,               False,             1      ),
    21:  TileKernelInstance(  128,     128,      256,     1,        4,       1,        32,            32,           16,      "Intrawave",        False,             False,        False,               False,             1      ),
    # K_Tile = 256, M_Warp x N_Warp = 2 x 2, WarpTile = 32 x 32 x 16
    22:  TileKernelInstance(   64,     128,      256,     2,        2,       1,        32,            32,           16,      "Intrawave",        False,             False,        False,               False,             1      ),
    23:  TileKernelInstance(  128,     128,      256,     2,        2,       1,        32,            32,           16,      "Intrawave",        False,             False,        False,               False,             1      ),
}


default_kernels_dict_tile = {
   #######################| M_Tile | N_Tile | K_Tile | M_Warp | N_Warp | K_Warp | M_Warp_Tile | N_Warp_Tile | K_Warp_Tile |   Scheduler   | TiledMMAPermuteN |  TransposeC | DoubleSmemBuffer | UsePersistentKernel | BlockPerCu |
    -1:  TileKernelInstance(  128,     128,      256,     1,        4,       1,        16,            16,           32,       "Intrawave",        False,             False,        False,               False,             1      ), 
}
# fmt: on
