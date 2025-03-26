# SPDX-License-Identifier: MIT
# Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.
# generate kernel instances to speed up compilation

import argparse
from pathlib import Path
from typing import List, Optional

GEN_DIR = ""    # in Cmake, have to generate files in same folder

AITER_API_FILENAME = "aiter_fmha_bwd.cpp"

AITER_CPP_API = """#include <iostream>
#include "aiter_fmha_bwd.h"

fmha_bwd_traits_all get_ck_fmha_bwd_traits_all(const mask_info &mask,
    std::string dtype,
    int head_size_q,
    int head_size_v,
    bool has_dropout,
    bool is_group_mode,
    bias_enum bias_type,
    bool deterministic,
    bool has_dbias,
    bool use_ext_asm,
    bool is_v3_atomic_fp32,
    int how_v3_bf16_cvt)
{{
    return fmha_bwd_traits_all(mask,
            dtype,
            head_size_q,
            head_size_v,
            has_dropout,
            is_group_mode,
            bias_type,
            deterministic,
            has_dbias,
            use_ext_asm,
            is_v3_atomic_fp32,
            how_v3_bf16_cvt);
}}

float fmha_bwd_aiter(fmha_bwd_args args,
        const ck_tile::stream_config& stream_config,
        mask_info mask,
        std::string q_dtype_str,
        bool is_group_mode,
        bias_enum bias_type,
        bool deterministic,
        bool has_dbias,
        bool use_ext_asm,
        bool is_v3_atomic_fp32,
        int how_v3_bf16_cvt)
{{
    int head_size_q = args.hdim_q;
    int head_size_v = args.hdim_v;
    bool has_dropout = args.p_drop > 0;
    // bool enable_ailib = args.alibi_slopes_ptr == nullptr;
    auto traits = get_ck_fmha_bwd_traits_all(mask, q_dtype_str, head_size_q, head_size_v, has_dropout, is_group_mode, bias_type, deterministic, has_dbias, use_ext_asm, is_v3_atomic_fp32, how_v3_bf16_cvt);
    float t = -1;
    t = {F_fmha_bwd_api}(traits, args, stream_config);
    return t;
}}
"""

def write_blobs(output_dir: Optional[str], receipt) -> None:
    if output_dir is None:
        output_dir = Path(__file__).parent
    else:
        output_dir = Path(output_dir) / GEN_DIR

    output_dir.mkdir(parents=True, exist_ok=True)

    fmha_bwd_api_name = "fmha_bwd" if receipt == 1 else "fmha_bwd_v3"
    api = AITER_CPP_API.format(F_fmha_bwd_api = fmha_bwd_api_name)
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
        help="write all the blobs into a directory"
    )
    parser.add_argument(
        "-r",
        "--receipt",
        default=0,
        required=False,
        help="codegen receipt. 1: generate fmha v2 c++ api\n"  + \
             "  2: generate fmha v3 c++ api"
    )

    args = parser.parse_args()

    write_blobs(args.output_dir, int(args.receipt))
