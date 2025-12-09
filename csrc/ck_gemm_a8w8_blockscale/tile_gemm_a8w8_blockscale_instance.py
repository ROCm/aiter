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
    
    TransposeC: bool
    DoubleSmemBuffer: bool
    UsePersistentKernel: bool
    
    @property
    def name(self) -> str:
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
                        [self.TransposeC, self.DoubleSmemBuffer, self.UsePersistentKernel],
                    )
                ) 
            ]
        )


# fmt: off
# Note: The index of tile kernels should be greater than that of legacy kernels in the manifest file.
# This is to ensure that when both legacy and tile kernels are tuned, the tile kernels are
# preferred during selection.
tile_candidate_kernels_dict = {
    #######################| M_Tile | N_Tile | K_Tile | M_Warp | N_Warp | K_Warp | M_Warp_Tile | N_Warp_Tile | K_Warp_Tile | kPadM | kPadN | kPadK |   Scheduler   | TransposeC | DoubleSmemBuffer | UsePersistentKernel |
    1:   TileKernelInstance(   16,     64,      256,     1,        4,       1,        16,            16,           256,      False,  False,  False,   "Intrawave",     False,          False,            False             )   
}


tile_default_kernels_dict = {
    #######################| M_Tile | N_Tile | K_Tile | M_Warp | N_Warp | K_Warp | M_Warp_Tile | N_Warp_Tile | K_Warp_Tile | kPadM | kPadN | kPadK |   Scheduler   | MemoryOperation | TransposeC | DoubleSmemBuffer | UsePersistentKernel |
    -1:   TileKernelInstance(   16,     64,      256,     1,        4,       1,        16,            16,           256,      True,  True,    True,   "Intrawave",       "Set",         False,          False,            True             )   
}
# fmt: on
