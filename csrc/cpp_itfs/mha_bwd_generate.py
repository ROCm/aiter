# SPDX-License-Identifier: MIT
# Copyright (C) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.
# generate kernel instances to speed up compilation

import argparse
import sys
import os
from pathlib import Path
from typing import Optional

this_dir = os.path.dirname(os.path.abspath(__file__))
AITER_CORE_DIR = os.path.abspath(f"{this_dir}/../../")
if os.path.exists(os.path.join(AITER_CORE_DIR, "aiter_meta")):
    AITER_CORE_DIR = os.path.join(AITER_CORE_DIR, "aiter")  # pip install mode
else:
    AITER_CORE_DIR = os.path.abspath(f"{this_dir}/../../aiter")  # develop mode
sys.path.insert(0, AITER_CORE_DIR)

import jit.core
from jit.utils.chip_info import get_gfx

GEN_DIR = ""  # in Cmake, have to generate files in same folder

AITER_API_FILENAME = "mha_bwd.cpp"

AITER_CPP_API = """#include "mha_bwd.h"
#include <iostream>

namespace aiter {{
mha_bwd_traits get_mha_bwd_traits(int head_size_q,
                                  int head_size_v,
                                  std::string dtype,
                                  bool is_group_mode,
                                  mask_enum mask_type,
                                  bias_enum bias_type,
                                  bool has_dbias,
                                  bool has_dropout,
                                  bool is_store_randval,
                                  bool deterministic,
                                  bool use_ext_asm,
                                  bool is_v3_atomic_fp32,
                                  int how_v3_bf16_cvt)
{{
    return mha_bwd_traits(head_size_q,
                          head_size_v,
                          dtype,
                          is_group_mode,
                          mask_type,
                          bias_type,
                          has_dbias,
                          has_dropout,
                          is_store_randval,
                          deterministic,
                          use_ext_asm,
                          is_v3_atomic_fp32,
                          how_v3_bf16_cvt);
}}

// share with varlen(group mode) api
float mha_bwd(mha_bwd_args args,
              const ck_tile::stream_config& stream_config,
              std::string q_dtype_str,
              bool is_group_mode,
              mask_enum mask_type,
              bias_enum bias_type,
              bool has_dbias,
              bool is_store_randval,
              bool deterministic,
              bool use_ext_asm,
              bool is_v3_atomic_fp32,
              int how_v3_bf16_cvt)
{{
    int head_size_q = args.hdim_q;
    int head_size_v = args.hdim_v;
    bool has_dropout = args.p_drop > 0;
    // bool enable_ailib = args.alibi_slopes_ptr == nullptr;
    auto traits = get_mha_bwd_traits(head_size_q,
                                     head_size_v,
                                     q_dtype_str,
                                     is_group_mode,
                                     mask_type,
                                     bias_type,
                                     has_dbias,
                                     has_dropout,
                                     is_store_randval,
                                     deterministic,
                                     use_ext_asm,
                                     is_v3_atomic_fp32,
                                     how_v3_bf16_cvt);
    float t = -1;
    {F_dispatch}
    return t;
}}
}} // namespace aiter

"""

V2_API = "t = fmha_bwd(traits, args, stream_config);"

V3_MULTI_TARGET_API = """std::cout << "========================" << std::endl;
    std::cout << "gpu arch: " << get_gpu_arch() << std::endl;
    if (get_gpu_arch() == "gfx942") {{
        std::cout << "gpu arch 1: " << get_gpu_arch() << std::endl;

        t = fmha_bwd_v3<GPUArch::gfx942>(traits, args, stream_config);
    }} else if (get_gpu_arch() == "gfx950") {{
        std::cout << "gpu arch 2: " << get_gpu_arch() << std::endl;
        t = fmha_bwd_v3<GPUArch::gfx950>(traits, args, stream_config);
    }} else {{
        std::cout << "No supported GPU arch found!" << std::endl;
        return -1;
    }}
"""

def get_v3_api():
    archs = os.getenv("GPU_ARCHS", "native").split(";")
    if archs[0] == "native":
        gfx = get_gfx()
        return f"t = fmha_bwd_v3<GPUArch::{gfx}>(traits, args, stream_config);"
    else:
        archs = [arch.strip() for arch in archs]
        if len(archs) == 1:
            return f"t = fmha_bwd_v3<GPUArch::{archs[0]}>(traits, args, stream_config);"
        else:
            return V3_MULTI_TARGET_API

V3_API = get_v3_api()

COMBINED_API = V3_API + """
    if (t == -1) { t = fmha_bwd(traits, args, stream_config); }
"""

API_MAP = {1: V2_API, 2: V3_API, 3: COMBINED_API}


def write_blobs(output_dir: Optional[str], receipt) -> None:
    if output_dir is None:
        output_dir = Path(__file__).parent
    else:
        output_dir = Path(output_dir) / GEN_DIR

    output_dir.mkdir(parents=True, exist_ok=True)

    api = AITER_CPP_API.format(F_dispatch=API_MAP[receipt])
    (output_dir / AITER_API_FILENAME).write_text(api)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="generate",
        description="gen API for CK fmha kernel",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        required=False,
        help="write all the blobs into a directory",
    )
    parser.add_argument(
        "-r",
        "--receipt",
        default=0,
        required=False,
        help="codegen receipt. 1: generate fmha v2 c++ api\n"
        + "  2: generate fmha v3 c++ api\n"
        + "  3: generate v2 v3 combined api for PREBUILD mode",
    )

    args = parser.parse_args()

    write_blobs(args.output_dir, int(args.receipt))
