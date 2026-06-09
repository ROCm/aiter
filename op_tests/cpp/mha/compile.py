# SPDX-License-Identifier: MIT
# Copyright (C) 2018-2026, Advanced Micro Devices, Inc. All rights reserved.
import sys
import os
import argparse

# !!!!!!!!!!!!!!!! never import aiter
# from aiter.jit import core
this_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, f"{this_dir}/../../../aiter/")
from jit.core import compile_ops, CK_DIR, AITER_CSRC_DIR, AITER_META_DIR  # noqa: E402

FWD_CODEGEN_CMD = [f"{AITER_META_DIR}/hsa/codegen.py -m fmha_v3_fwd --output_dir {{}}"]
BWD_CODEGEN_CMD = [f"{AITER_META_DIR}/hsa/codegen.py -m fmha_v3_bwd --output_dir {{}}"]
# codegen for ASM FWD with sink (gfx1250 bf16 kernels in hsa/gfx1250/fmha_fwd_bf16/)
FWD_SINK_CODEGEN_CMD = [f"{AITER_META_DIR}/hsa/codegen.py -m fmha_fwd_bf16 --output_dir {{}}"]


def cmdGenFunc_mha_fwd(ck_exclude: bool):
    if ck_exclude:
        srcs = [
            f"{AITER_CSRC_DIR}/cpp_itfs/mha_fwd.cu",
            # ASM FWD with sink dispatcher (gfx1250 bf16); provides
            # fmha_fwd_with_sink_asm C-ABI entry point used by
            # benchmark_mha_fwd_v3.cpp
            f"{AITER_CSRC_DIR}/py_itfs_cu/asm_fmha_fwd_with_sink.cu",
        ]
        blob_gen_cmd = list(FWD_SINK_CODEGEN_CMD)
    else:
        srcs = [
            f"{AITER_CSRC_DIR}/cpp_itfs/mha_fwd.cu",
            f"{AITER_CSRC_DIR}/cpp_itfs/mha_fwd_split.cu",
            f"{AITER_CSRC_DIR}/cpp_itfs/mha_fwd_batch_prefill.cu",
        ]
        blob_gen_cmd = [
            f"{CK_DIR}/example/ck_tile/01_fmha/generate.py -d fwd --receipt 600 --output_dir {{}}",
            f"{CK_DIR}/example/ck_tile/01_fmha/generate.py -d fwd_splitkv --receipt 600 --output_dir {{}}",
            f"{CK_DIR}/example/ck_tile/01_fmha/generate.py -d batch_prefill --receipt 600 --output_dir {{}}",
        ]
    blob_gen_cmd.extend(FWD_CODEGEN_CMD)
    flag_use_v3 = (
        "-DFAV3_ON=1 -DENABLE_CK=0" if ck_exclude else "-DFAV3_ON=1 -DFAV2_ON=1"
    )
    # Use a separate library name for the CK-excluded (gfx1250 / ASM-only) path
    # so its JIT blob directory never collides with the full CK build of libmha_fwd.
    # Both share the compile_mha_fwd entry point but produce distinct .so files.
    md_name = "libmha_fwd_asm" if ck_exclude else "libmha_fwd"
    return {
        "srcs": srcs,
        "md_name": md_name,
        "blob_gen_cmd": blob_gen_cmd,
        "flags_extra_cc": [flag_use_v3],
        "torch_exclude": True,
        "is_python_module": False,
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
    flags_extra_cc = ["-DONLY_FAV3", "-DENABLE_CK=0"] if ck_exclude else []
    return {
        "md_name": "libmha_bwd",
        "blob_gen_cmd": blob_gen_cmd,
        "flags_extra_cc": flags_extra_cc,
        "torch_exclude": True,
        "is_python_module": False,
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
