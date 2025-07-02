# SPDX-License-Identifier: MIT
# Copyright (C) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

import os
import argparse
import glob
import pandas as pd

this_dir = os.path.dirname(os.path.abspath(__file__))

template = """// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.
#pragma once
#include <unordered_map>

#define ADD_CFG(q_type, kv_type, gqa, mtp, msk, hp, path, name, co)         \\
    {                                         \\
        name, { name, path co, q_type, kv_type, gqa, mtp, msk, hp }         \\
    }

struct AsmPaConfig
{
    std::string name;
    std::string co_name;
    std::string q_type;
    std::string kv_type;
    int gqa;
    int mtp;
    int msk;
    int hp;
};

using CFG = std::unordered_map<std::string, AsmPaConfig>;

"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="generate",
        description="gen API for asm PA kernel",
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
            f'ADD_CFG("{qType}", "{kvType}",{Gqa:>4}, {Mtp:>2}, {Msk:>2},{Hp:>2},"pa/", "{Name}", "{Co}"),'
            for qType, kvType, Gqa, Mtp, Msk, Hp, Name, Co in df.values
        ]
        filename = os.path.basename(el)
        cfgname = filename.split(".")[0]
        cfg_txt = "\n            ".join(cfg) + "\n"

        txt = f"""static CFG cfg_{cfgname} = {{
            {cfg_txt}}};"""
        cfgs.append(txt)
    txt_all = template + "\n".join(cfgs)
    with open(f"{args.output_dir}/asm_pa_configs.hpp", "w") as f:
        f.write(txt_all)
