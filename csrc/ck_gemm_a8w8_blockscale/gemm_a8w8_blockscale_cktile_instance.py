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
    UsePersistentKernel: bool

    BlockPerCu: int  # 1,2

    @property
    def name(self) -> str:
        """
        Generate a unique name for the kernel instance based on its parameters.
        """

        return ("_").join(
            [
                "a8w8_blockscale_cktile",
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
        )


# fmt: off
# Candidate and default kernel instances for tile gemm a8w8 blockscale
# These instances are used for generating the kernel code and tuning.
candidate_kernels_cktile_dict = {
    #######################| M_Tile | N_Tile | K_Tile | M_Warp | N_Warp | K_Warp | M_Warp_Tile | N_Warp_Tile | K_Warp_Tile |   Scheduler   | TiledMMAPermuteN |  TransposeC | UsePersistentKernel | BlockPerCu |
    # K_Tile = 128, M_Warp x N_Warp = 1 x 4, WarpTile = 16 x 16 x 32
    0:   TileKernelInstance(   32,     128,      128,     1,        4,       1,        16,            16,           32,      "Intrawave",        False,             False,               False,             1      ),
    1:   TileKernelInstance(   64,     128,      128,     1,        4,       1,        16,            16,           32,      "Intrawave",        False,             False,               False,             1      ),
    2:   TileKernelInstance(  128,     128,      128,     1,        4,       1,        16,            16,           32,      "Intrawave",        False,             False,               False,             1      ),
    # K_Tile = 256, M_Warp x N_Warp = 1 x 4, WarpTile = 16 x 16 x 32
    3:   TileKernelInstance(   32,     128,      256,     1,        4,       1,        16,            16,           32,      "Intrawave",        False,             False,               False,             1      ),
    4:   TileKernelInstance(   64,     128,      256,     1,        4,       1,        16,            16,           32,      "Intrawave",        False,             False,               False,             1      ),
    5:   TileKernelInstance(  128,     128,      256,     1,        4,       1,        16,            16,           32,      "Intrawave",        False,             False,               False,             1      ),
    # K_Tile = 128, M_Warp x N_Warp = 2 x 2, WarpTile = 16 x 16 x 32  
    6:   TileKernelInstance(   64,     128,      128,     2,        2,       1,        16,            16,           32,      "Intrawave",        False,             False,               False,             1      ),
    7:   TileKernelInstance(  128,     128,      128,     2,        2,       1,        16,            16,           32,      "Intrawave",        False,             False,               False,             1      ),
    # K_Tile = 256, M_Warp x N_Warp = 2 x 2, WarpTile = 16 x 16 x 32
    8:   TileKernelInstance(   64,     128,      256,     2,        2,       1,        16,            16,           32,      "Intrawave",        False,             False,               False,             1      ),
    9:   TileKernelInstance(  128,     128,      256,     2,        2,       1,        16,            16,           32,      "Intrawave",        False,             False,               False,             1      ),
    # # K_Tile = 128, M_Warp x N_Warp = 1 x 4, WarpTile = 32 x 32 x 16
    # 10:  TileKernelInstance(   64,     128,      128,     1,        4,       1,        32,            32,           16,      "Intrawave",        False,             False,               False,             1      ),
    # 11:  TileKernelInstance(  128,     128,      128,     1,        4,       1,        32,            32,           16,      "Intrawave",        False,             False,               False,             1      ),
    # # K_Tile = 256, M_Warp x N_Warp = 1 x 4, WarpTile = 32 x 32 x 16
    # 12:  TileKernelInstance(   64,     128,      256,     1,        4,       1,        32,            32,           16,      "Intrawave",        False,             False,               False,             1      ),
    # 13:  TileKernelInstance(  128,     128,      256,     1,        4,       1,        32,            32,           16,      "Intrawave",        False,             False,               False,             1      ),
    # # K_Tile = 128, M_Warp x N_Warp = 2 x 2, WarpTile = 32 x 32 x 16
    # 14:  TileKernelInstance(  128,     128,      128,     2,        2,       1,        32,            32,           16,      "Intrawave",        False,             False,               False,             1      ),
    # # K_Tile = 256, M_Warp x N_Warp = 2 x 2, WarpTile = 32 x 32 x 16
    # 15:  TileKernelInstance(  128,     128,      256,     2,        2,       1,        32,            32,           16,      "Intrawave",        False,             False,               False,             1      ),
}


default_kernels_cktile_dict = {
   #######################| M_Tile | N_Tile | K_Tile | M_Warp | N_Warp | K_Warp | M_Warp_Tile | N_Warp_Tile | K_Warp_Tile |   Scheduler   | TiledMMAPermuteN |  TransposeC | UsePersistentKernel | BlockPerCu |
    -1:  TileKernelInstance(  128,     128,      256,     1,        4,       1,        16,            16,           32,       "Intrawave",        False,             False,               False,             1      ),
}
# fmt: on
