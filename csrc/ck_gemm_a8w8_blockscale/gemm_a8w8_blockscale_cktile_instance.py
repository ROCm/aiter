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
    PreshuffleQuant: bool    #TODO Not sure if this needs to be added to the
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
                            self.PreshuffleQuant,
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
    #######################| M_Tile | N_Tile | K_Tile | M_Warp | N_Warp | K_Warp | M_Warp_Tile | N_Warp_Tile | K_Warp_Tile |   Scheduler   | TiledMMAPermuteN |  TransposeC     | PreshuffleQuant | UsePersistentKernel | BlockPerCu |
    0:   TileKernelInstance(   128,     128,      128,     2,        2,       1,        16,            16,           128,      "Intrawave",        False,             True,               False,              False,             2      ),
    1:   TileKernelInstance(   128,     128,      128,     1,        4,       1,        16,            16,           128,      "Intrawave",        False,             True,               False,              False,             2      ),
    2:   TileKernelInstance(    64,     128,      128,     1,        4,       1,        16,            16,           128,      "Intrawave",        False,             True,               False,              False,             2      ),
    3:   TileKernelInstance(    64,     128,      128,     2,        2,       1,        16,            16,           128,      "Intrawave",        False,             True,               False,              False,             2      ),
    4:   TileKernelInstance(   128,     128,      128,     1,        4,       1,        16,            16,           128,      "Intrawave",        False,             False,               True,              False,             2      ),
}


default_kernels_cktile_dict = {
   #######################| M_Tile | N_Tile | K_Tile | M_Warp | N_Warp | K_Warp | M_Warp_Tile | N_Warp_Tile | K_Warp_Tile |   Scheduler   | TiledMMAPermuteN |  TransposeC | PreshuffleQuant | UsePersistentKernel | BlockPerCu |
    -1:  TileKernelInstance(  128,     128,      256,     1,        4,       1,        16,            16,           32,       "Intrawave",        False,             False,       False,        False,             1      ),
}
# fmt: on
