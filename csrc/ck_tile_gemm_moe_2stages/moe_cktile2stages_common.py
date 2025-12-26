# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
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
class kernelInstance:
    stage: int
    BLOCK_SIZE: int
    MPerBlock: int
    NPerBlock: int
    KPerBlock: int
    WAVE_TILE_M: int
    WAVE_TILE_N: int
    WAVE_TILE_K: int
    WAVE_MAP_M: int
    WAVE_MAP_N: int
    Block_Per_CU: int = 1
    MulRoutedWeight: bool = False
    ActOP: str = "silu"
    QuantType: str = "per_tensor"

    @property
    def name(self) -> str:
        return ("_").join(
            element
            for element in [
                f"moe_cktile2stages_gemm{self.stage}",
                ("x").join(
                    map(
                        lambda x: str(x),
                        [
                            self.BLOCK_SIZE,
                            self.MPerBlock,
                            self.NPerBlock,
                            self.KPerBlock,
                        ],
                    )
                ),
                ("x").join(map(lambda x: str(x), [self.WAVE_MAP_M, self.WAVE_MAP_N])),
                ("x").join(
                    map(
                        lambda x: str(x),
                        [self.WAVE_TILE_M, self.WAVE_TILE_N, self.WAVE_TILE_K],
                    )
                ),
                str(self.Block_Per_CU) + "perCU",
                self.QuantType,
                "MulRoutedWeight" if self.MulRoutedWeight else "",
                "" if (self.stage == 2) else self.ActOP,
            ]
            if element != ""
        )


# fmt: off
# gemm1 out:bf16/fp16 AB:fp8/i8
a8w8_gemm1_kernels_list_gfx950= {
    #  kernel:           stage| BLOCK_SIZE|MPerBLOCK|  NPerBLOCK| KPerBLOCK| WAVE_TILE_M| WAVE_TILE_N| WAVE_TILE_K| WAVE_MAP_M| WAVE_MAP_N|
    # 0: kernelInstance(       1,        256,       32,         64,       256,           16,         16,         128,          1,        4,),
    1: kernelInstance(       1,        256,       32,        128,       128,           16,         16,         128,          1,        4,),
    2: kernelInstance(       1,        256,       64,        128,       128,           16,         16,         128,          1,        4,),
    4: kernelInstance(       1,        256,       64,        128,       256,           16,         16,         128,          1,        4,),
    4: kernelInstance(       1,        256,      128,        128,       128,           16,         16,         128,          1,        4,),
    5: kernelInstance(       1,        256,      128,        128,       128,           16,         16,         128,          1,        4,),
    6: kernelInstance(       1,        256,      256,        128,       128,           16,         16,         128,          1,        4,),
}

# gemm2 out:bf16/fp16 AB:fp8/i8
a8w8_gemm2_kernels_list_gfx950= {
    #  kernel:           stage| BLOCK_SIZE|MPerBLOCK|  NPerBLOCK| KPerBLOCK| WAVE_TILE_M| WAVE_TILE_N| WAVE_TILE_K| WAVE_MAP_M| WAVE_MAP_N|
    0: kernelInstance(       2,        256,       32,        128,       256,           16,         16,         128,          1,        4,),
    1: kernelInstance(       2,        256,       64,        128,       256,           16,         16,         128,          1,        4,),
    2: kernelInstance(       2,        256,      128,        128,       128,           16,         16,         128,          1,        4,),
    3: kernelInstance(       2,        256,      256,        128,       128,           16,         16,         128,          1,        4,),
    4: kernelInstance(       2,        256,      256,        128,       128,           16,         16,         128,          1,        4,),
}


#a8w8
a8w8_gemm1_kernels_list= {
    #  kernel:           stage| BLOCK_SIZE|MPerBLOCK|  NPerBLOCK| KPerBLOCK| WAVE_TILE_M| WAVE_TILE_N| WAVE_TILE_K| WAVE_MAP_M| WAVE_MAP_N|
    # 0: kernelInstance(       1,        256,       32,         64,       256,           16,         16,          64,          1,        4,),
    # 1: kernelInstance(       1,        256,       32,         64,       128,           16,         16,          64,          1,        4,),
    # 2: kernelInstance(       1,        256,       64,         64,       256,           16,         16,          64,          2,        2,),
    # 3: kernelInstance(       1,        256,       64,         64,       128,           16,         16,          64,          1,        4,),
    3: kernelInstance(       1,        256,       64,         128,       128,           16,         16,          64,          1,        4),
    # 4: kernelInstance(       1,        256,      128,         64,       128,           16,         16,          64,          1,        4,),
    # 5: kernelInstance(       1,        256,      128,        128,       128,           16,         16,          64,          1,        4,),
    # 6: kernelInstance(       1,        256,      256,        128,       128,           16,         16,          64,          1,        4,),
}
a8w8_gemm2_kernels_list= {
    #  kernel:           stage| BLOCK_SIZE|MPerBLOCK|  NPerBLOCK| KPerBLOCK| WAVE_TILE_M| WAVE_TILE_N| WAVE_TILE_K| WAVE_MAP_M| WAVE_MAP_N|
    # 0: kernelInstance(       2,        256,       32,         64,       256,           16,         16,          64,          1,        4,),
    # 1: kernelInstance(       2,        256,       64,         64,       256,           16,         16,          64,          1,        4,),
    # 2: kernelInstance(       2,        256,      128,         64,       128,           16,         16,          64,          1,        4,),
    # 3: kernelInstance(       2,        256,      256,         64,       128,           16,         16,          64,          1,        4,),
    # 4: kernelInstance(       2,        256,       64,        128,       256,           16,         16,         128,          1,        4,),
    # 5: kernelInstance(       2,        256,      128,        128,       128,           16,         16,          64,          1,        4,),
    # 6: kernelInstance(       2,        256,      256,        128,       128,           16,         16,          64,          1,        4,),
    # 7: kernelInstance(       2,        256,       32,         64,       128,           16,         16,          64,          1,        4,),
    8: kernelInstance(       2,        256,       64,        128,       128,           16,         16,          64,          1,        4,),
}


# gemm1 out:bf16/fp16 AB:bf16/fp4
a16w4_gemm1_kernels_list_gfx950= {
    #  kernel:           stage| BLOCK_SIZE|MPerBLOCK|  NPerBLOCK| KPerBLOCK| WAVE_TILE_M| WAVE_TILE_N| WAVE_TILE_K| WAVE_MAP_M| WAVE_MAP_N|| BlockPerCU|
    0: kernelInstance(       1,        256,       16,        128,       256,           16,         16,          32,          1,           4,          2,),
    # 5: kernelInstance(       1,        256,       16,        512,       256,           16,         16,          32,          1,           4,          4,),
    1: kernelInstance(       1,        256,       32,        256,       256,           16,         16,          32,          1,           4,          2,),
    3: kernelInstance(       1,        256,       64,        256,       256,           16,         16,          32,          1,           4,          1,),
    # 4: kernelInstance(       1,        256,      128,        256,       256,           16,         16,          32,          1,           4,          1,),
}
# gemm1 out:bf16/fp16 AB:bf16/fp4
a16w4_gemm1_kernels_list= {
    #  kernel:           stage| BLOCK_SIZE|MPerBLOCK|  NPerBLOCK| KPerBLOCK| WAVE_TILE_M| WAVE_TILE_N| WAVE_TILE_K| WAVE_MAP_M| WAVE_MAP_N|| BlockPerCU|
    0: kernelInstance(       1,        256,       16,        128,       256,           16,         16,          32,          1,           4,          2,),
    # 5: kernelInstance(       1,        256,       16,        512,       256,           16,         16,          32,          1,           4,          4,),
    1: kernelInstance(       1,        256,       32,        256,       256,           16,         16,          32,          1,           4,          2,),
    3: kernelInstance(       1,        256,       64,        256,       256,           16,         16,          32,          1,           4,          1,),
    # 4: kernelInstance(       1,        256,      128,        256,       256,           16,         16,          32,          1,           4,          1,),
}
# gemm2 out:bf16/fp16 AB:bf16/fp4
a16w4_gemm2_kernels_list= {
    #  kernel:           stage| BLOCK_SIZE|MPerBLOCK|  NPerBLOCK| KPerBLOCK| WAVE_TILE_M| WAVE_TILE_N| WAVE_TILE_K| WAVE_MAP_M| WAVE_MAP_N| BlockPerCU|
    0: kernelInstance(       2,        256,       16,        128,       256,           16,         16,          32,          1,        4,            2,),
    # 5: kernelInstance(       2,        256,       16,        512,       256,           16,         16,          32,          1,        4,            4,),
    1: kernelInstance(       2,        256,       32,        256,       256,           16,         16,          32,          1,        4,            2,),
    3: kernelInstance(       2,        256,       64,        256,       256,           16,         16,          32,          1,        4,            1,),
    # 4: kernelInstance(       2,        256,      128,        256,       256,           16,         16,          32,          1,        4,            1,),
    # 4: kernelInstance(       2,        256,      256,        256,       256,           16,         16,          32,          1,        4,),
    # 4: kernelInstance(       2,        256,      256,        128,       128,           16,         16,          32,          1,        4,),
}
# gemm2 out:bf16/fp16 AB:bf16/fp4
a16w4_gemm2_kernels_list_gfx950= {
    #  kernel:           stage| BLOCK_SIZE|MPerBLOCK|  NPerBLOCK| KPerBLOCK| WAVE_TILE_M| WAVE_TILE_N| WAVE_TILE_K| WAVE_MAP_M| WAVE_MAP_N| BlockPerCU|
    0: kernelInstance(       2,        256,       16,        128,       256,           16,         16,          32,          1,        4,            2,),
    # 5: kernelInstance(       2,        256,       16,        512,       256,           16,         16,          32,          1,        4,            4,),
    1: kernelInstance(       2,        256,       32,        256,       256,           16,         16,          32,          1,        4,            2,),
    3: kernelInstance(       2,        256,       64,        256,       256,           16,         16,          32,          1,        4,            1,),
    # 4: kernelInstance(       2,        256,      128,        256,       128,           16,         16,          32,          1,        4,            1,),
    # 4: kernelInstance(       2,        256,      256,        256,       256,           16,         16,          32,          1,        4,),
    # 4: kernelInstance(       2,        256,      256,        128,       128,           16,         16,          32,          1,        4,),
}

# gemm1 out:bf16/fp16 AB:fp8/fp4
a8w4_gemm1_kernels_list_gfx950= {
    #  kernel:           stage| BLOCK_SIZE|MPerBLOCK|  NPerBLOCK| KPerBLOCK| WAVE_TILE_M| WAVE_TILE_N| WAVE_TILE_K| WAVE_MAP_M| WAVE_MAP_N| BlockPerCU|
    # 0: kernelInstance(       1,        256,       16,        128,       256,           16,         16,          128,          1,        4,            2,),
    # 5: kernelInstance(       2,        256,       16,        512,       256,           16,         16,          32,          1,        4,            4,),
    1: kernelInstance(       1,        256,       32,        256,       256,           16,         16,          128,          1,        4,            2,),
    3: kernelInstance(       1,        256,       64,        256,       256,           16,         16,          128,          1,        4,            1,),
    # 4: kernelInstance(       2,        256,      128,        256,       128,           16,         16,          32,          1,        4,            1,),
    # 4: kernelInstance(       2,        256,      256,        256,       256,           16,         16,          32,          1,        4,),
    # 4: kernelInstance(       2,        256,      256,        128,       128,           16,         16,          32,          1,        4,),
}
# gemm2 out:bf16/fp16 AB:fp8/fp4
a8w4_gemm2_kernels_list_gfx950= {
    #  kernel:           stage| BLOCK_SIZE|MPerBLOCK|  NPerBLOCK| KPerBLOCK| WAVE_TILE_M| WAVE_TILE_N| WAVE_TILE_K| WAVE_MAP_M| WAVE_MAP_N| BlockPerCU|
    # 0: kernelInstance(       2,        256,       16,        128,       256,           16,         16,          128,          1,        4,            2,),
    # 5: kernelInstance(       2,        256,       16,        512,       256,           16,         16,          32,          1,        4,            4,),
    1: kernelInstance(       2,        256,       32,        256,       256,           16,         16,          128,          1,        4,            2,),
    3: kernelInstance(       2,        256,       64,        256,       256,           16,         16,          128,          1,        4,            1,),
    # 4: kernelInstance(       2,        256,      128,        256,       128,           16,         16,          32,          1,        4,            1,),
    # 4: kernelInstance(       2,        256,      256,        256,       256,           16,         16,          32,          1,        4,),
    # 4: kernelInstance(       2,        256,      256,        128,       128,           16,         16,          32,          1,        4,),
}

# fmt: on
gemm1_kernels_dict = {
    "a8w8_gfx950": a8w8_gemm1_kernels_list_gfx950,
    "a8w8": a8w8_gemm1_kernels_list,
    "a16w4_gfx950": a16w4_gemm1_kernels_list_gfx950,
    "a16w4": a16w4_gemm1_kernels_list,
    "a8w4_gfx950": a8w4_gemm1_kernels_list_gfx950,
}

gemm2_kernels_dict = {
    "a8w8_gfx950": a8w8_gemm2_kernels_list_gfx950,
    "a8w8": a8w8_gemm2_kernels_list,
    "a16w4_gfx950": a16w4_gemm2_kernels_list_gfx950,
    "a16w4": a16w4_gemm2_kernels_list,
    "a8w4_gfx950": a8w4_gemm2_kernels_list_gfx950,
}


a8w8_gfx950_heuristic_dispatch = """#pragma once
// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
#include "moe_cktile2stages.h"
#include "moe_cktile2stages_heuristic_dispatch_common.h"

template <>
MoeKernel moe_gemm1_heuristic_dispatch<ck_tile::fp8_t, ck_tile::fp8_t, float, ck_tile::bf16_t>(int M, int N, int K, int block_m)
{{
    // Apply shape heuristics to find a suitable kernel implementation.
    if (block_m == 32)
    {{
        return {(1, 1)}<ck_tile::fp8_t, ck_tile::fp8_t, float, ck_tile::bf16_t>;
    }}
    else if (block_m == 64)
    {{
        return {(1, 2)}<ck_tile::fp8_t, ck_tile::fp8_t, float, ck_tile::bf16_t>;
    }}
    //else if (block_m == 128)
    //{{
    //    return {(1, 4)}<ck_tile::fp8_t, ck_tile::fp8_t, float, ck_tile::bf16_t>;
    //}}
    //else if (block_m == 256)
    //{{
    //    return {(1, 6)}<ck_tile::fp8_t, ck_tile::fp8_t, float, ck_tile::bf16_t>;
    //}}
    else
    {{
        TORCH_CHECK(
            false,
            "Unsupported block_m value for moe_geem1 heuristic dispatch: ",
            block_m);
    }}
}}

template <>
MoeKernel moe_gemm2_heuristic_dispatch<ck_tile::fp8_t, ck_tile::fp8_t, float, ck_tile::bf16_t>(int M, int N, int K, int block_m)
{{
    // Apply shape heuristics to find a suitable kernel implementation.
    if (block_m == 32)
    {{
        return {(2, 0)}<ck_tile::fp8_t, ck_tile::fp8_t, float, ck_tile::bf16_t>;
    }}
    else if (block_m == 64)
    {{
        return {(2, 1)}<ck_tile::fp8_t, ck_tile::fp8_t, float, ck_tile::bf16_t>;
    }}
    //else if (block_m == 128)
    //{{
    //    return {(2, 2)}<ck_tile::fp8_t, ck_tile::fp8_t, float, ck_tile::bf16_t>;
    //}}
    //else if (block_m == 256)
    //{{
    //    return {(2, 3)}<ck_tile::fp8_t, ck_tile::fp8_t, float, ck_tile::bf16_t>;
    //}}
    else
    {{
        TORCH_CHECK(
            false,
            "Unsupported block_m value for moe_gemm1 heuristic dispatch: ",
            block_m);
    }}
}}
"""

a16w4_gfx950_heuristic_dispatch = """#pragma once
// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
#include "moe_cktile2stages.h"
#include "moe_cktile2stages_heuristic_dispatch_common.h"

template <>
MoeKernel moe_gemm1_heuristic_dispatch<ck_tile::bf16_t, ck_tile::pk_fp4_t, float, ck_tile::bf16_t>(int M, int N, int K, int block_m)
{{
    // Apply shape heuristics to find a suitable kernel implementation.
    if (block_m == 16)
    {{
        return {(1, 0)}<ck_tile::bf16_t, ck_tile::pk_fp4_t, float, ck_tile::bf16_t>;
    }}
    else if (block_m == 32)
    {{
        return {(1, 1)}<ck_tile::bf16_t, ck_tile::pk_fp4_t, float, ck_tile::bf16_t>;
    }}
    else if (block_m == 64)
    {{
        return {(1, 3)}<ck_tile::bf16_t, ck_tile::pk_fp4_t, float, ck_tile::bf16_t>;
    }}
    else
    {{
        TORCH_CHECK(
            false,
            "Unsupported block_m value for moe_geem1 heuristic dispatch: ",
            block_m);
    }}
}}

template <>
MoeKernel moe_gemm2_heuristic_dispatch<ck_tile::bf16_t, ck_tile::pk_fp4_t, float, ck_tile::bf16_t>(int M, int N, int K, int block_m)
{{
    // Apply shape heuristics to find a suitable kernel implementation.
    if (block_m == 16)
    {{
        return {(2, 0)}<ck_tile::bf16_t, ck_tile::pk_fp4_t, float, ck_tile::bf16_t>;
    }}
    else if (block_m == 32)
    {{
        return {(2, 1)}<ck_tile::bf16_t, ck_tile::pk_fp4_t, float, ck_tile::bf16_t>;
    }}
    else if (block_m == 64)
    {{
        return {(2, 3)}<ck_tile::bf16_t, ck_tile::pk_fp4_t, float, ck_tile::bf16_t>;
    }}
    else
    {{
        TORCH_CHECK(
            false,
            "Unsupported block_m value for moe_gemm2 heuristic dispatch: ",
            block_m);
    }}
}}
"""

a16w4_heuristic_dispatch = """#pragma once
// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
#include "moe_cktile2stages.h"
#include "moe_cktile2stages_heuristic_dispatch_common.h"

template <>
MoeKernel moe_gemm1_heuristic_dispatch<ck_tile::bf16_t, ck_tile::pk_fp4_t, float, ck_tile::bf16_t>(int M, int N, int K, int block_m)
{{
    // Apply shape heuristics to find a suitable kernel implementation.
    if (block_m == 16)
    {{
        return {(1, 0)}<ck_tile::bf16_t, ck_tile::pk_fp4_t, float, ck_tile::bf16_t>;
    }}
    else if (block_m == 32)
    {{
        return {(1, 1)}<ck_tile::bf16_t, ck_tile::pk_fp4_t, float, ck_tile::bf16_t>;
    }}
    else if (block_m == 64)
    {{
        return {(1, 3)}<ck_tile::bf16_t, ck_tile::pk_fp4_t, float, ck_tile::bf16_t>;
    }}
    else
    {{
        TORCH_CHECK(
            false,
            "Unsupported block_m value for moe_geem1 heuristic dispatch: ",
            block_m);
    }}
}}

template <>
MoeKernel moe_gemm2_heuristic_dispatch<ck_tile::bf16_t, ck_tile::pk_fp4_t, float, ck_tile::bf16_t>(int M, int N, int K, int block_m)
{{
    // Apply shape heuristics to find a suitable kernel implementation.
    if (block_m == 16)
    {{
        return {(2, 0)}<ck_tile::bf16_t, ck_tile::pk_fp4_t, float, ck_tile::bf16_t>;
    }}
    else if (block_m == 32)
    {{
        return {(2, 1)}<ck_tile::bf16_t, ck_tile::pk_fp4_t, float, ck_tile::bf16_t>;
    }}
    else if (block_m == 64)
    {{
        return {(2, 3)}<ck_tile::bf16_t, ck_tile::pk_fp4_t, float, ck_tile::bf16_t>;
    }}
    else
    {{
        TORCH_CHECK(
            false,
            "Unsupported block_m value for moe_gemm2 heuristic dispatch: ",
            block_m);
    }}
}}
"""

a8w4_gfx950_heuristic_dispatch = """#pragma once
// SPDX-License-Identifier: MIT
// Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
#include "moe_cktile2stages.h"
#include "moe_cktile2stages_heuristic_dispatch_common.h"

template <>
MoeKernel moe_gemm1_heuristic_dispatch<ck_tile::fp8_t, ck_tile::pk_fp4_t, float, ck_tile::bf16_t>(
    int M,
    int N,
    int K,
    int block_m)
{{
    // Apply shape heuristics to find a suitable kernel implementation.
    if (block_m == 32)
    {{
        return {(1, 1)}<ck_tile::fp8_t, ck_tile::pk_fp4_t, float, ck_tile::bf16_t>;
    }}
    else if (block_m == 64)
    {{
        return {(1, 3)}<ck_tile::fp8_t, ck_tile::pk_fp4_t, float, ck_tile::bf16_t>;
    }}
    else
    {{
        TORCH_CHECK(
            false,
            "Unsupported block_m value for moe_geem1 heuristic dispatch: ",
            block_m);
    }}
}}

template <>
MoeKernel moe_gemm2_heuristic_dispatch<ck_tile::fp8_t, ck_tile::pk_fp4_t, float, ck_tile::bf16_t>(
    int M,
    int N,
    int K,
    int block_m)
{{
    // Apply shape heuristics to find a suitable kernel implementation.
    if (block_m == 32)
    {{
        return {(2, 1)}<ck_tile::fp8_t, ck_tile::pk_fp4_t, float, ck_tile::bf16_t>;
    }}
    else if (block_m == 64)
    {{
        return {(2, 3)}<ck_tile::fp8_t, ck_tile::pk_fp4_t, float, ck_tile::bf16_t>;
    }}
    else
    {{
        TORCH_CHECK(
            false,
            "Unsupported block_m value for moe_gemm2 heuristic dispatch: ",
            block_m);
    }}
}}
"""

heuristic_dispatch_dict = {
    "a8w8_gfx950": a8w8_gfx950_heuristic_dispatch,
    # "a8w8": a8w8_gemm2_kernels_list,
    "a16w4_gfx950": a16w4_gfx950_heuristic_dispatch,
    "a16w4": a16w4_heuristic_dispatch,
    "a8w4_gfx950": a8w4_gfx950_heuristic_dispatch,
}


bit8_list = ["f8", "i8", "fp8"]
bit16_list = ["b16", "f16", "bf16", "fp16"]
bit4_list = ["i4", "fp4x2", "fp4"]
QuantType_list = ["no", "per_tensor", "per_token", "per_1x128", "per_1x32"]


def get_gemm1_kernels_list(
    Adtype: str,
    Bdtype: str,
    QuantType: str = "none",
    ActOP: str = "silu",
    MulRoutedWeight: bool = False,
) -> list:
    arch = get_gfx()
    if Adtype.lower() in bit8_list and Bdtype.lower() in bit8_list and Adtype == Bdtype:
        if arch == "gfx950":
            tag = "a8w8_gfx950"
        else:
            tag = "a8w8"
    elif Adtype in bit16_list and Bdtype in bit4_list:
        if arch == "gfx950":
            tag = "a16w4_gfx950"
        else:
            tag = "a16w4"
    elif Adtype.lower() in bit8_list and Bdtype in bit4_list:
        if arch == "gfx950":
            tag = "a8w4_gfx950"
        else:
            raise ValueError(
                f"Unsupported data type combination: {Adtype}, {Bdtype} on {arch}"
            )
    else:
        raise ValueError(f"Unsupported data type combination: {Adtype}, {Bdtype}")
    kernels_list = gemm1_kernels_dict[tag]
    for id, kernel in kernels_list.items():
        kernel.MulRoutedWeight = MulRoutedWeight
        kernel.ActOP = ActOP
        kernel.QuantType = QuantType
        # if tag == "a8w4":
        # kernel.CDEElementOp = "MulABScaleWint4"
        # elif tag == "a8w8blkscale":
        # kernel.CDEElementOp = "MulABScaleExpertWeightA8W8blkscale"
        # elif tag == "a8w8" or tag == "a4w4":
        # kernel.CDEElementOp = "MulABScale"
        # elif tag == "a16w16":
        # if MulRoutedWeight:
        # kernel.CDEElementOp = "TypeCastExpertWeight"
        # else:
        # kernel.CDEElementOp = "TypeCast"
    return tag, kernels_list


def get_gemm2_kernels_list(
    Adtype: str,
    Bdtype: str,
    QuantType: str = "",
    ActOP: str = "",
    MulRoutedWeight: bool = True,
) -> list:
    arch = get_gfx()
    if Adtype in bit8_list and Bdtype in bit8_list and Adtype == Bdtype:
        if arch == "gfx950":
            tag = "a8w8_gfx950"
        else:
            tag = "a8w8"
    elif Adtype in bit16_list and Bdtype in bit4_list:
        if arch == "gfx950":
            tag = "a16w4_gfx950"
        else:
            tag = "a16w4"
    elif Adtype.lower() in bit8_list and Bdtype in bit4_list:
        if arch == "gfx950":
            tag = "a8w4_gfx950"
        else:
            raise ValueError(
                f"Unsupported data type combination: {Adtype}, {Bdtype} on {arch}"
            )
    else:
        raise ValueError(f"Unsupported data type combination: {Adtype}, {Bdtype}")
    kernels_list = gemm2_kernels_dict[tag]
    for id, kernel in kernels_list.items():
        kernel.MulRoutedWeight = MulRoutedWeight
        kernel.ActOP = ""
        kernel.QuantType = QuantType
        # if tag == "a8w4":
        #     kernel.CDEElementOp = "MulABScaleExpertWeightWin4"
        # elif tag == "a8w8blkscale":
        #     kernel.CDEElementOp = "MulABScaleExpertWeightA8W8blkscale"
        # elif tag == "a8w8" or tag == "a4w4":
        #     kernel.CDEElementOp = "MulABScaleExpertWeight"
        # elif tag == "a16w16":
        #     if MulRoutedWeight:
        #         kernel.CDEElementOp = "TypeCastExpertWeight"
        #     else:
        #         kernel.CDEElementOp = "TypeCast"
    return tag, kernels_list


def get_heuristic_dispatch_template(tag):
    if tag not in heuristic_dispatch_dict.keys():
        raise ValueError(f"Unsupported type for heuristic_dispatch: {tag}")
    return heuristic_dispatch_dict[tag]
