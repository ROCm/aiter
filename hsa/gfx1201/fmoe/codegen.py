# SPDX-License-Identifier: MIT
# Copyright (C) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

import os
import argparse
import glob
import pandas as pd

this_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.basename(this_dir)

template = """// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.
#pragma once
#include <unordered_map>

#define ADD_CFG(atm, vskip, smf, tg_num_perCU, ps, subGU_m, subGU_n, path, name, co)         \\
    {                                         \\
        name, { name, path co, atm, vskip, smf, tg_num_perCU, ps, subGU_m, subGU_n }         \\
    }

struct AsmFmoeConfig
{
    std::string name;
    std::string co_name;
    int atm;
    int vskip;
    int smf;
    int tg_num_perCU;
    int ps;
    int subGU_m;
    int subGU_n;
};

using CFG = std::unordered_map<std::string, AsmFmoeConfig>;

"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="generate",
        description="gen API for asm Fused_moe kernel",
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
    for el in glob.glob(f"{this_dir}/**/*.csv", recursive=True):

        df = pd.read_csv(el)
        cfg = [
            f'ADD_CFG({atm}, {vskip},{smf:>4}, {tg_num_perCU:>4}, {ps:>4},{subGU_m:>4}, {subGU_n:>4}, "{base_dir}/{os.path.dirname(os.path.relpath(el, this_dir))}/", "{Name}", "{Co}"),'
            for Name, Co, atm, vskip, smf, tg_num_perCU, ps, subGU_m, subGU_n in df.values
        ]
        filename = os.path.basename(el)
        cfgname = filename.split(".")[0]
        cfg_txt = "\n            ".join(cfg) + "\n"

        txt = f"""static CFG cfg_{cfgname} = {{
            {cfg_txt}}};"""
        cfgs.append(txt)
    txt_all = template + "\n".join(cfgs)
    with open(f"{args.output_dir}/asm_fmoe_configs.hpp", "w") as f:
        f.write(txt_all)
