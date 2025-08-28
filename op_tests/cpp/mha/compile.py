# SPDX-License-Identifier: MIT
# Copyright (C) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.
import sys
import os
import argparse

# !!!!!!!!!!!!!!!! never import aiter
# from aiter.jit import core
this_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, f"{this_dir}/../../../aiter/")
from jit.core import compile_ops, CK_DIR, AITER_CSRC_DIR, get_asm_dir


def cmdGenFunc_mha_fwd(ck_exclude: bool):
    if ck_exclude:
        blob_gen_cmd = [
            f"{AITER_CSRC_DIR}/cpp_itfs/mha_fwd_generate.py --receipt 1 --output_dir {{}}",
            f"{get_asm_dir()}/fmha_v3_fwd/codegen.py --output_dir {{}}",
        ]
    else:
        blob_gen_cmd = [
            f"{CK_DIR}/example/ck_tile/01_fmha/generate.py -d fwd --receipt 600 --output_dir {{}}",
            f"{CK_DIR}/example/ck_tile/01_fmha/generate.py -d fwd_splitkv --receipt 600 --output_dir {{}}",
            f"{CK_DIR}/example/ck_tile/01_fmha/generate.py -d batch_prefill --receipt 600 --output_dir {{}}",
            f"{AITER_CSRC_DIR}/cpp_itfs/mha_fwd_generate.py --receipt 5 --output_dir {{}}",
            f"{get_asm_dir()}/fmha_v3_fwd/codegen.py --output_dir {{}}",
        ]
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
        blob_gen_cmd = [
            f'{AITER_CSRC_DIR}/py_itfs_cu/fmha_bwd_pre_post_kernel_generate.py --filter "*@*_ndeterministic@*_nbias*_dropout*_ndeterministic*" --output_dir {{}}',
            f"{get_asm_dir()}/fmha_v3_bwd/codegen.py --output_dir {{}}",
            f"{AITER_CSRC_DIR}/cpp_itfs/mha_bwd_generate.py --receipt 2 --output_dir {{}}",
        ]
    else:
        blob_gen_cmd = [
            f"{CK_DIR}/example/ck_tile/01_fmha/generate.py -d bwd --receipt 600 --output_dir {{}}",
            f'{AITER_CSRC_DIR}/py_itfs_cu/fmha_bwd_pre_post_kernel_generate.py --filter "*@*_ndeterministic@*_nbias*_dropout*_ndeterministic*" --output_dir {{}}',
            f"{get_asm_dir()}/fmha_v3_bwd/codegen.py --output_dir {{}}",
            f"{AITER_CSRC_DIR}/cpp_itfs/mha_bwd_generate.py --receipt 3 --output_dir {{}}",
        ]
    return {
        "md_name": "libmha_bwd",
        "blob_gen_cmd": blob_gen_cmd,
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

    if args.api == "fwd":
        compile_mha_fwd()
    elif args.api == "bwd":
        compile_mha_bwd()
    elif args.api == "fwd_v3":
        compile_mha_fwd(True)
    elif args.api == "bwd_v3":
        compile_mha_bwd(True)
    elif args.api == "":
        compile_mha_fwd()
        compile_mha_bwd()
    else:
        raise ValueError(
            "Invalid input value: only support 'fwd', 'bwd' or default to be ''"
        )
