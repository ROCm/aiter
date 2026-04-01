# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
from dataclasses import dataclass, field
from typing import List


@dataclass
class OpusGemmInstance:
    BLOCK_SIZE: int
    B_M: int
    B_N: int
    B_K: int
    T_M: int
    T_N: int
    W_M: int
    W_N: int
    W_K: int
    VEC_A: int
    VEC_B: int
    VEC_C: int
    GROUP_M: int
    GROUP_N: int
    GROUP_K: int
    kernel_tag: str
    output_dtypes: List[str] = field(default_factory=lambda: ["fp32_t"])

    @property
    def name(self) -> str:
        return "_".join(
            [
                "opus_gemm",
                "x".join(
                    map(str, [self.BLOCK_SIZE, self.B_M, self.B_N, self.B_K])
                ),
                "x".join(map(str, [self.T_M, self.T_N])),
                "x".join(map(str, [self.W_M, self.W_N, self.W_K])),
                "x".join(map(str, [self.GROUP_M, self.GROUP_N, self.GROUP_K])),
            ]
        )


# fmt: off
kernels_list = {
    1: OpusGemmInstance(512, 256, 256, 128, 4, 2, 16, 16, 128, 16, 16, 4, 1, 128, 128, "a8w8_scale", ["fp32_t"]),
    2: OpusGemmInstance(512, 256, 256, 128, 2, 4, 16, 16, 128, 16, 16, 4, 0, 0, 0,     "a8w8",       ["fp32_t"]),
    3: OpusGemmInstance(512, 256, 256, 64,  2, 4, 16, 16, 32,  8, 8, 4,  0, 0, 0,      "a16w16",     ["fp32_t", "bf16_t"]),
}

default_kernels_dict = {
    (-1): OpusGemmInstance(512, 256, 256, 128, 4, 2, 16, 16, 128, 16, 16, 4, 1, 128, 128, "a8w8_scale", ["fp32_t"]),
    (-2): OpusGemmInstance(512, 256, 256, 128, 2, 4, 16, 16, 128, 16, 16, 4, 0, 0, 0,     "a8w8",       ["fp32_t"]),
    (-3): OpusGemmInstance(512, 256, 256, 64,  2, 4, 16, 16, 32,  8, 8, 4,  0, 0, 0,      "a16w16",     ["fp32_t", "bf16_t"]),
}
# fmt: on
