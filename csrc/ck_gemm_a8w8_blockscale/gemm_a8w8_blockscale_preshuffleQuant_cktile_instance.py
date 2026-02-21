# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices,Inc. All rights reserved.
from dataclasses import dataclass
import os
import sys

this_dir = os.path.dirname(os.path.abspath(__file__))
AITER_CORE_DIR = os.path.abspath(f"{this_dir}/../../../")
if os.path.exists(os.path.join(AITER_CORE_DIR, "aiter_meta")):
    AITER_CORE_DIR = os.path.join(AITER_CORE_DIR, "aiter/jit/utils")  # pip install mode
else:
    AITER_CORE_DIR = os.path.abspath(
        f"{this_dir}/../../aiter/jit/utils"
    )  # develop mode
sys.path.insert(0, AITER_CORE_DIR)

from chip_info import get_gfx  # noqa: E402

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
kernels_list_942 = {
    #######################| M_Tile | N_Tile | K_Tile | M_Warp | N_Warp | K_Warp | M_Warp_Tile | N_Warp_Tile | K_Warp_Tile |   Scheduler   | TiledMMAPermuteN |  TransposeC | UsePersistentKernel | BlockPerCu |
    0:   TileKernelInstance(   128,     128,      128,     1,        4,       1,        16,            16,           64,     "Intrawave",         False,            False,           False,             1      ),
    1:   TileKernelInstance(    16,     128,      256,     1,        4,       1,        16,            16,           64,     "Intrawave",         False,            False,           False,             1      ),
}

kernels_list_95x = {
    #######################| M_Tile | N_Tile | K_Tile | M_Warp | N_Warp | K_Warp | M_Warp_Tile | N_Warp_Tile | K_Warp_Tile |   Scheduler   | TiledMMAPermuteN |  TransposeC | UsePersistentKernel | BlockPerCu |
    #  0:   TileKernelInstance(    16,     128,      256,     1,        4,       1,        16,            16,          128,      "Intrawave",        False,            False,           False,             1      ),
    #  1:   TileKernelInstance(    16,     128,      256,     1,        4,       1,        16,            16,          128,      "Intrawave",        False,            False,           False,             2      ),
    #  2:   TileKernelInstance(    16,     128,      256,     1,        4,       1,        16,            16,           64,      "Intrawave",        False,            False,           False,             1      ),
    #  3:   TileKernelInstance(    16,     128,      256,     1,        4,       1,        16,            16,           64,      "Intrawave",        False,            False,           False,             2      ),
     0:   TileKernelInstance(    32,     128,      128,     1,        4,       1,        16,            16,          128,      "Intrawave",        False,            False,           False,             1      ),
     1:   TileKernelInstance(    32,     128,      128,     1,        4,       1,        16,            16,          128,      "Intrawave",        False,            False,           False,             2      ),
     2:   TileKernelInstance(    32,     128,      128,     1,        4,       1,        16,            16,           64,      "Intrawave",        False,            False,           False,             1      ),
     3:   TileKernelInstance(    32,     128,      128,     1,        4,       1,        16,            16,           64,      "Intrawave",        False,            False,           False,             2      ),
     4:   TileKernelInstance(   128,     128,      128,     1,        4,       1,        16,            16,          128,      "Intrawave",        False,            False,           False,             1      ),
     5:   TileKernelInstance(   128,     128,      128,     1,        4,       1,        16,            16,          128,      "Intrawave",        False,            False,           False,             1      ),
     6:   TileKernelInstance(   128,     128,      128,     2,        2,       1,        16,            16,          128,      "Intrawave",        False,            False,           False,             2      ),
    # 11:   TileKernelInstance(   192,     256,      128,     4,        2,       1,        16,            16,          128,      "Intrawave",        False,            False,           False,             1      ),
}
default_kernels_cktile_dict_preshuffleQuant = {
    #######################| M_Tile | N_Tile | K_Tile | M_Warp | N_Warp | K_Warp | M_Warp_Tile | N_Warp_Tile | K_Warp_Tile |   Scheduler   | TiledMMAPermuteN |  TransposeC  | UsePersistentKernel | BlockPerCu |
    -1:  TileKernelInstance(   128,     128,      128,     1,        4,       1,        16,            16,          64,      "Intrawave",        False,              False,           False,             1      ),
}
# fmt: on

arch = get_gfx()
if arch.startswith("gfx95"):
    candidate_kernels_cktile_dict_preshuffleQuant = kernels_list_95x
else:
    candidate_kernels_cktile_dict_preshuffleQuant = kernels_list_942