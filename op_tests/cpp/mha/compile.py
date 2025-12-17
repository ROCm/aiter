# SPDX-License-Identifier: MIT
# Copyright (C) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.
import sys
import os
import argparse

# !!!!!!!!!!!!!!!! never import aiter
# from aiter.jit import core
this_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, f"{this_dir}/../../../aiter/")
from jit.core import compile_ops, CK_DIR, AITER_CSRC_DIR, AITER_META_DIR
from jit.utils.chip_info import get_gfx_list


FWD_CODEGEN_CMD = []
BWD_CODEGEN_CMD = [f"{AITER_META_DIR}/hsa/codegen.py -m fmha_v3_bwd --output_dir {{}}"]


def get_asm_dir():
    for gfx in get_gfx_list():
        FWD_ASM_DIR = f"{AITER_META_DIR}/hsa/{gfx}/fmha_v3_fwd"
        if os.path.exists(FWD_ASM_DIR):
            FWD_CODEGEN_CMD.append(f"{FWD_ASM_DIR}/codegen.py --output_dir {{}}")


def cmdGenFunc_mha_fwd(ck_exclude: bool):
    if ck_exclude:
        blob_gen_cmd = [
            f"{AITER_CSRC_DIR}/cpp_itfs/mha_fwd_generate.py --receipt 1 --output_dir {{}}",
        ]
    else:
        blob_gen_cmd = [
            f"{CK_DIR}/example/ck_tile/01_fmha/generate.py -d fwd --receipt 600 --output_dir {{}}",
            f"{CK_DIR}/example/ck_tile/01_fmha/generate.py -d fwd_splitkv --receipt 600 --output_dir {{}}",
            f"{CK_DIR}/example/ck_tile/01_fmha/generate.py -d batch_prefill --receipt 600 --output_dir {{}}",
            f"{AITER_CSRC_DIR}/cpp_itfs/mha_fwd_generate.py --receipt 5 --output_dir {{}}",
        ]
    blob_gen_cmd.extend(FWD_CODEGEN_CMD)
    return {
        "md_name": "libmha_fwd",
        "blob_gen_cmd": blob_gen_cmd,
    }


@compile_ops(
    "libmha_fwd",
    fc_name="compile_mha_fwd",
    gen_func=cmdGenFunc_mha_fwd,
)
def compile_mha_fwd(ck_exclude: bool): ...


def cmdGenFunc_mha_bwd(ck_exclude: bool):
    if ck_exclude:
        blob_gen_cmd = []
    else:
        blob_gen_cmd = [
            f"{CK_DIR}/example/ck_tile/01_fmha/generate.py -d bwd --receipt 600 --output_dir {{}}",
        ]
    blob_gen_cmd.extend(BWD_CODEGEN_CMD)
    flag_use_v3 = "-DONLY_FAV3" if ck_exclude else None
    return {
        "md_name": "libmha_bwd",
        "blob_gen_cmd": blob_gen_cmd,
        "flags_extra_cc": [flag_use_v3],
    }


@compile_ops(
    "libmha_bwd",
    fc_name="compile_mha_bwd",
    gen_func=cmdGenFunc_mha_bwd,
)
def compile_mha_bwd(ck_exclude: bool = False): ...


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="compile",
        description="compile C++ instance with torch excluded",
    )
    parser.add_argument(
        "--api",
        default="",
        required=False,
        help="supply API(s) to generate (default: all). separated by comma.",
    )

    args = parser.parse_args()

    get_asm_dir()
    if args.api == "fwd":
        compile_mha_fwd(False)
    elif args.api == "bwd":
        compile_mha_bwd(False)
    elif args.api == "fwd_v3":
        compile_mha_fwd(True)
    elif args.api == "bwd_v3":
        compile_mha_bwd(True)
    elif args.api == "":
        compile_mha_fwd(False)
        compile_mha_bwd(False)
    else:
        raise ValueError(
            "Invalid input value: only support 'fwd', 'bwd' or default to be ''"
        )
