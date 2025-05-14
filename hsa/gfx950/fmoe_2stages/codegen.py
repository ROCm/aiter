# SPDX-License-Identifier: MIT
# Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

import os
import argparse
import glob
import pandas as pd

this_dir = os.path.dirname(os.path.abspath(__file__))

template = """// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

#define ADD_CFG(M, N, path, name, co)         \\
    {                                         \\
        name, { name, path co, M, N }         \\
    }

struct FMoe2StageConfig
{
    std::string name;
    std::string co_name;
    int tile_M;
    int tile_N;
};

using CFG = std::unordered_map<std::string, FMoe2StageConfig>;

"""
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="generate",
        description="gen API for CK fmha kernel",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        default="aiter/jit/build",
        required=False,
        help="write all the blobs into a directory",
    )
    args = parser.parse_args()

    cfgs = []
    for el in glob.glob(f"{this_dir}/*.csv"):
        df = pd.read_csv(el)
        cfg = [
            f'ADD_CFG({tileM:>4}, {tileN:>4}, "fmoe_2stages/", "{name}", "{co}"),'
            for tileM, tileN, name, co in df.values
        ]
        filename = os.path.basename(el)
        cfgname = filename.split(".")[0]
        cfg_txt = "\n            ".join(cfg) + "\n"

        txt = f"""static CFG cfg_{cfgname} = {{
            {cfg_txt}}};"""
        cfgs.append(txt)
    txt_all = template + "\n".join(cfgs)
    with open(f"{args.output_dir}/asm_moe_2stage_configs.hpp", "w") as f:
        f.write(txt_all)
