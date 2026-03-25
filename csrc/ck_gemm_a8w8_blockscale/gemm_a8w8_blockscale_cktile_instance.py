# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices,Inc. All rights reserved.
from copy import copy
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

    BlockPerCu: int  # 1..BLOCK_PER_CU_MAX

    AQRowMajor: bool = False  # When True, 8-warp kernels read x_scale in
    # row-major layout natively, skipping the host-side transpose.

    @property
    def is_eight_warp(self) -> bool:
        return (
            self.M_Warp * self.N_Warp * self.K_Warp == 8
            and self.K_Warp_Tile == 128
        )

    @property
    def name(self) -> str:
        """
        Generate a unique name for the kernel instance based on its parameters.
        """

        parts = [
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
        if self.AQRowMajor:
            parts.append("aqrm")
        return "_".join(parts)


BLOCK_PER_CU_MAX = 4


def expand_blockpercu(base_dict, max_bpc=BLOCK_PER_CU_MAX, field_name="BlockPerCu"):
    """Expand kernel instances with BlockPerCu 1..max_bpc variants.

    For each unique tile configuration (all fields except BlockPerCu),
    creates variants for every BPC value in 1..max_bpc that doesn't
    already exist in base_dict.
    """
    expanded = dict(base_dict)
    configs = {}  # tile_config_key -> {bpc: id, ...}
    for idx, k in base_dict.items():
        key = tuple(v for f, v in vars(k).items() if f != field_name)
        configs.setdefault(key, {})[getattr(k, field_name)] = idx
    next_id = max(base_dict.keys()) + 1
    for key, existing_bpcs in configs.items():
        template = base_dict[next(iter(existing_bpcs.values()))]
        for bpc in range(1, max_bpc + 1):
            if bpc not in existing_bpcs:
                inst = copy(template)
                inst.BlockPerCu = bpc
                expanded[next_id] = inst
                next_id += 1
    return expanded


# fmt: off
# Candidate and default kernel instances for tile gemm a8w8 blockscale
# These instances are used for generating the kernel code and tuning.
kernels_list_942 = {
    #######################| M_Tile | N_Tile | K_Tile | M_Warp | N_Warp | K_Warp | M_Warp_Tile | N_Warp_Tile | K_Warp_Tile |   Scheduler   | TiledMMAPermuteN |  TransposeC | UsePersistentKernel | BlockPerCu |
    0:   TileKernelInstance(   128,     128,      128,     1,        4,       1,        16,            16,           64,     "Intrawave",         False,             True,           False,             1      ),
    1:   TileKernelInstance(    16,     128,      256,     1,        4,       1,        16,            16,           64,     "Intrawave",         False,             True,           False,             1      ),
}

kernels_list_95x = {
    #######################| M_Tile | N_Tile | K_Tile | M_Warp | N_Warp | K_Warp | M_Warp_Tile | N_Warp_Tile | K_Warp_Tile |   Scheduler   | TiledMMAPermuteN |  TransposeC | UsePersistentKernel | BlockPerCu |
     0:   TileKernelInstance(    16,     128,      256,     1,        4,       1,        16,            16,          128,      "Intrawave",        False,             True,           False,             1      ),
     1:   TileKernelInstance(    16,     128,      256,     1,        4,       1,        16,            16,          128,      "Intrawave",        False,             True,           False,             2      ),
     2:   TileKernelInstance(    16,     128,      256,     1,        4,       1,        16,            16,           64,      "Intrawave",        False,             True,           False,             1      ),
     3:   TileKernelInstance(    16,     128,      256,     1,        4,       1,        16,            16,           64,      "Intrawave",        False,             True,           False,             2      ),
     4:   TileKernelInstance(    32,     128,      128,     1,        4,       1,        16,            16,          128,      "Intrawave",        False,             True,           False,             1      ),
     5:   TileKernelInstance(    32,     128,      128,     1,        4,       1,        16,            16,          128,      "Intrawave",        False,             True,           False,             2      ),
     6:   TileKernelInstance(    32,     128,      128,     1,        4,       1,        16,            16,           64,      "Intrawave",        False,             True,           False,             1      ),
     7:   TileKernelInstance(    32,     128,      128,     1,        4,       1,        16,            16,           64,      "Intrawave",        False,             True,           False,             2      ),
     8:   TileKernelInstance(   128,     128,      128,     1,        4,       1,        16,            16,          128,      "Intrawave",        False,             True,           False,             1      ),
     9:   TileKernelInstance(   128,     128,      128,     1,        4,       1,        16,            16,          128,      "Intrawave",        False,             True,           False,             1      ),
    10:   TileKernelInstance(   128,     128,      128,     2,        2,       1,        16,            16,          128,      "Intrawave",        False,             True,           False,             2      ),
    11:   TileKernelInstance(   192,     256,      128,     4,        2,       1,        16,            16,          128,      "Intrawave",        False,             True,           False,             1      ),
    # 8-warp kernels with AQRowMajor=True: skip host-side x_scale transpose
    12:   TileKernelInstance(   128,     128,      128,     2,        2,       1,        16,            16,          128,      "Intrawave",        False,             True,           False,             2,     AQRowMajor=True),
    13:   TileKernelInstance(   192,     256,      128,     4,        2,       1,        16,            16,          128,      "Intrawave",        False,             True,           False,             1,     AQRowMajor=True),
}

default_kernels_cktile_dict = {
    #######################| M_Tile | N_Tile | K_Tile | M_Warp | N_Warp | K_Warp | M_Warp_Tile | N_Warp_Tile | K_Warp_Tile |   Scheduler   | TiledMMAPermuteN |  TransposeC  | UsePersistentKernel | BlockPerCu |
    -1:  TileKernelInstance(   128,     128,      128,     1,        4,       1,        16,            16,           64,      "Intrawave",        False,             True,           False,             1      ),
}

# fmt: on


arch = get_gfx()
if arch.startswith("gfx95"):
    candidate_kernels_cktile_dict = expand_blockpercu(kernels_list_95x)
else:
    candidate_kernels_cktile_dict = expand_blockpercu(kernels_list_942)

# Name-based reverse lookup for get_tune_dict()
candidate_kernels_by_name = {v.name: v for v in candidate_kernels_cktile_dict.values()}
