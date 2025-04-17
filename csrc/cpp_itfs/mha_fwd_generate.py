# SPDX-License-Identifier: MIT
# Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.
# generate kernel instances to speed up compilation

import argparse
from pathlib import Path
from typing import List, Optional

GEN_DIR = ""    # in Cmake, have to generate files in same folder

AITER_API_FILENAME = "mha_fwd.cpp"

AITER_CPP_API = """#include "mha_fwd.h"

namespace aiter {{
mha_fwd_traits get_mha_fwd_traits(int head_size_q,
                                  int head_size_v,
                                  std::string dtype,
                                  bool is_group_mode,
                                  const mask_info &mask,
                                  bias_enum bias_type,
                                  bool has_lse,
                                  bool has_dropout)
{{
    return mha_fwd_traits(head_size_q,
                          head_size_v,
                          dtype,
                          is_group_mode,
                          mask,
                          bias_type,
                          has_lse,
                          has_dropout);
}}

mha_fwd_splitkv_traits get_mha_fwd_splitkv_traits(int head_size_q,
                                                  int head_size_v,
                                                  std::string dtype,
                                                  bool is_group_mode,
                                                  const mask_info &mask,
                                                  bias_enum bias_type,
                                                  bool has_lse)
{{
    return mha_fwd_splitkv_traits(head_size_q,
                                  head_size_v,
                                  dtype,
                                  is_group_mode,
                                  mask,
                                  bias_type,
                                  has_lse);
}}
{F_dispatch}

}} // namespace aiter

"""

FMHA_FWD_API = """
float mha_fwd(mha_fwd_args args,
              const ck_tile::stream_config& stream_config,
              std::string q_dtype_str,
              bool is_group_mode,
              mask_info mask,
              bias_enum bias_type,
              bool has_lse)
{
    int head_size_q = args.hdim_q;
    int head_size_v = args.hdim_v;
    bool has_dropout = args.p_drop > 0.f;
    auto traits = get_mha_fwd_traits(head_size_q,
                                     head_size_v,
                                     q_dtype_str,
                                     is_group_mode,
                                     mask,
                                     bias_type,
                                     has_lse,
                                     has_dropout);
    return fmha_fwd(traits, args, stream_config);
}"""

FMHA_FWD_SPLITKV_API = """
float mha_fwd_splitkv(mha_fwd_splitkv_args args,
                      const ck_tile::stream_config& stream_config,
                      std::string q_dtype_str,
                      bool is_group_mode,
                      mask_info mask,
                      bias_enum bias_type,
                      bool has_lse)
{
    int head_size_q = args.hdim_q;
    int head_size_v = args.hdim_v;
    auto traits = get_mha_fwd_splitkv_traits(head_size_q,
                                             head_size_v,
                                             q_dtype_str,
                                             is_group_mode,
                                             mask,
                                             bias_type,
                                             has_lse);
    return fmha_fwd_splitkv(traits, args, stream_config);
}"""

API_MAP = {
    1: FMHA_FWD_API,
    2: FMHA_FWD_SPLITKV_API,
    3: FMHA_FWD_API + FMHA_FWD_SPLITKV_API,
}

def write_blobs(output_dir: Optional[str], receipt) -> None:
    if output_dir is None:
        output_dir = Path(__file__).parent
    else:
        output_dir = Path(output_dir) / GEN_DIR

    output_dir.mkdir(parents=True, exist_ok=True)

    api = AITER_CPP_API.format(F_dispatch = API_MAP[receipt])
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
        help="codegen receipt. 1: generate mha_fwd c++ api\n"  + \
                            "  2: generate mha_fwd_splitkv c++ api\n"  + \
                            "  3: generate fmha varlen fwd c++ api, also can be use for PREBUILD"
    )

    args = parser.parse_args()

    write_blobs(args.output_dir, int(args.receipt))
