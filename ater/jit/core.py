# Copyright (c) 2024 Advanced Micro Devices, Inc.  All rights reserved.
#
# -*- coding:utf-8 -*-
# @Script: core.py
# @Author: valarLip
# @Email: lingpeng.jin@amd.com
# @Create At: 2024-11-29 15:58:57
# @Last Modified By: valarLip
# @Last Modified At: 2025-01-14 16:15:31
# @Description: This is description.

import os
import sys
import shutil
import time
import importlib
import functools
import traceback
from typing import List, Optional
from torch.utils import cpp_extension
from torch.utils.file_baton import FileBaton
import logging
import json

PREBUILD_KERNELS = False
if os.path.exists(os.path.dirname(os.path.abspath(__file__))+"/ater_.so"):
    ater_ = importlib.import_module(f'{__package__}.ater_')
    PREBUILD_KERNELS = True
logger = logging.getLogger("ater")

PY = sys.executable
this_dir = os.path.dirname(os.path.abspath(__file__))
ATER_ROOT_DIR = os.path.abspath(f"{this_dir}/../../")
ATER_CSRC_DIR = f'{ATER_ROOT_DIR}/csrc'
CK_DIR = os.environ.get("CK_DIR",
                        f"{ATER_ROOT_DIR}/3rdparty/composable_kernel")
bd_dir = f"{this_dir}/build"
# copy ck to build, thus hippify under bd_dir
shutil.copytree(CK_DIR, f'{bd_dir}/ck', dirs_exist_ok=True)
CK_DIR = f'{bd_dir}/ck'


def validate_and_update_archs():
    archs = os.getenv("GPU_ARCHS", "native").split(";")
    # List of allowed architectures
    allowed_archs = ["native", "gfx90a",
                     "gfx940", "gfx941", "gfx942", "gfx1100"]

    # Validate if each element in archs is in allowed_archs
    assert all(
        arch in allowed_archs for arch in archs
    ), f"One of GPU archs of {archs} is invalid or not supported"
    return archs


def check_and_set_ninja_worker():
    max_num_jobs_cores = int(max(1, os.cpu_count()*0.8))
    if int(os.environ.get("MAX_JOBS", '1')) < max_num_jobs_cores:
        import psutil
        # calculate the maximum allowed NUM_JOBS based on free memory
        free_memory_gb = psutil.virtual_memory().available / \
            (1024 ** 3)  # free memory in GB
        # each JOB peak memory cost is ~8-9GB when threads = 4
        max_num_jobs_memory = int(free_memory_gb / 9)

        # pick lower value of jobs based on cores vs memory metric to minimize oom and swap usage during compilation
        max_jobs = max(1, min(max_num_jobs_cores, max_num_jobs_memory))
        max_jobs = str(max_jobs)
        os.environ["MAX_JOBS"] = max_jobs


def rename_cpp_to_cu(els, dst, recurisve=False):
    def do_rename_and_mv(name, src, dst, ret):
        newName = name
        if name.endswith(".cpp") or name.endswith(".cu"):
            newName = name.replace(".cpp", ".cu")
            ret.append(f'{dst}/{newName}')
        shutil.copy(f'{src}/{name}', f'{dst}/{newName}')
    ret = []
    for el in els:
        if not os.path.exists(el):
            continue
        if os.path.isdir(el):
            for entry in os.listdir(el):
                if os.path.isdir(f'{el}/{entry}'):
                    if recurisve:
                        ret += rename_cpp_to_cu([f'{el}/{entry}'],
                                                dst, recurisve)
                    continue
                do_rename_and_mv(entry, el, dst, ret)
        else:
            do_rename_and_mv(os.path.basename(el),
                             os.path.dirname(el), dst, ret)
    return ret


@functools.lru_cache(maxsize=1024)
def get_module(md_name):
    return importlib.import_module(f'{__package__}.{md_name}')


def build_module(md_name, srcs, flags_extra_cc, flags_extra_hip, blob_gen_cmd, extra_include, extra_ldflags, verbose):
    startTS = time.perf_counter()
    try:
        op_dir = f'{bd_dir}/{md_name}'
        logger.info(f'start build [{md_name}] under {op_dir}')

        opbd_dir = f'{op_dir}/build'
        src_dir = f'{op_dir}/build/srcs'
        os.makedirs(src_dir, exist_ok=True)
        sources = rename_cpp_to_cu(srcs, src_dir)

        flags_cc = ["-O3", "-std=c++17"]
        flags_hip = [
            "-DUSE_PROF_API=1",
            "-D__HIP_PLATFORM_HCC__=1",
            "-D__HIP_PLATFORM_AMD__=1",
            "-U__HIP_NO_HALF_CONVERSIONS__",
            "-U__HIP_NO_HALF_OPERATORS__",

            "-mllvm", "-enable-post-misched=0",
            "-mllvm", "-amdgpu-early-inline-all=true",
            "-mllvm", "-amdgpu-function-calls=false",
            "-mllvm", "--amdgpu-kernarg-preload-count=16",
            "-mllvm", "-amdgpu-coerce-illegal-types=1",
            # "-v", "--save-temps",
            "-Wno-unused-result",
            "-Wno-switch-bool",
            "-Wno-vla-cxx-extension",
            "-Wno-undefined-func-template",
        ]
        flags_cc += flags_extra_cc
        flags_hip += flags_extra_hip
        archs = validate_and_update_archs()
        flags_hip += [f"--offload-arch={arch}" for arch in archs]
        check_and_set_ninja_worker()

        def exec_blob(blob_gen_cmd, op_dir, src_dir, sources):
            if blob_gen_cmd:
                blob_dir = f"{op_dir}/blob"
                os.makedirs(blob_dir, exist_ok=True)
                baton = FileBaton(os.path.join(blob_dir, 'lock'))
                if baton.try_acquire():
                    try:
                        os.system(f'{PY} {blob_gen_cmd.format(blob_dir)}')
                    finally:
                        baton.release()
                else:
                    baton.wait()
                sources += rename_cpp_to_cu([blob_dir],
                                            src_dir, recurisve=True)
            return sources

        if isinstance(blob_gen_cmd, list):
            for s_blob_gen_cmd in blob_gen_cmd:
                sources = exec_blob(s_blob_gen_cmd, op_dir, src_dir, sources)
        else:
            sources = exec_blob(blob_gen_cmd, op_dir, src_dir, sources)

        bd_include_dir = f'{op_dir}/build/include'
        os.makedirs(bd_include_dir, exist_ok=True)
        rename_cpp_to_cu([f"{ATER_CSRC_DIR}/include"],
                         bd_include_dir)
        extra_include_paths = [
            f"{CK_DIR}/include",
            f"{CK_DIR}/library/include",
            f"{bd_include_dir}",
        ]+extra_include

        module = cpp_extension.load(
            md_name,
            sources,
            extra_cflags=flags_cc,
            extra_cuda_cflags=flags_hip,
            extra_ldflags=extra_ldflags,
            extra_include_paths=extra_include_paths,
            build_directory=opbd_dir,
            verbose=verbose or int(os.getenv("ATER_LOG_MORE", 0)) > 0,
            with_cuda=True,
            is_python_module=True,
        )
        shutil.copy(f'{opbd_dir}/{md_name}.so', f'{this_dir}')
    except Exception as e:
        logger.error('failed build jit [{}]\n-->[History]: {}'.format(
            md_name,
            ''.join(traceback.format_exception(*sys.exc_info()))
        ))
        sys.exit()
    logger.info(
        f'finish build [{md_name}], cost {time.perf_counter()-startTS:.8f}s')
    return module


def get_args_of_build(ops_name: str):
    d_opt_build_args = {"srcs": [],
                        "md_name": "",
                        "flags_extra_cc": [],
                        "flags_extra_hip": [],
                        "extra_ldflags": None,
                        "extra_include": [],
                        "verbose": False,
                        "blob_gen_cmd": ""
                        }

    def convert(d_ops: dict):
        # judge isASM
        if d_ops["isASM"].lower() == "true":
            d_ops["flags_extra_hip"].append(
                "rf'-DATER_ASM_DIR=\\\"{ATER_ROOT_DIR}/hsa/\\\"'")
        del d_ops["isASM"]
        for k, val in d_ops.items():
            if isinstance(val, list):
                for idx, el in enumerate(val):
                    if isinstance(el, str):
                        val[idx] = eval(el)
                d_ops[k] = val
            elif isinstance(val, str):
                d_ops[k] = eval(val)
            else:
                pass
        return d_ops
    with open(this_dir+"/optCompilerConfig.json", 'r') as file:
        data = json.load(file)
        if isinstance(data, dict):
            # parse all ops
            if ops_name == "all":
                d_all_ops = {"srcs": [],
                             "flags_extra_cc": [],
                             "flags_extra_hip": [],
                             "extra_include": [],
                             "blob_gen_cmd": []}
                # traverse opts
                for ops_name, d_ops in data.items():
                    # Cannot contain tune ops
                    if ops_name.endswith("tune"):
                        continue
                    single_ops = convert(d_ops)
                    for k in d_all_ops.keys():
                        if isinstance(single_ops[k], list):
                            d_all_ops[k] += single_ops[k]
                        elif isinstance(single_ops[k], str) and single_ops[k] != '':
                            d_all_ops[k].append(single_ops[k])

                # print(d_all_ops)
                return d_all_ops
            # no find opt_name in json.
            elif data.get(ops_name) == None:
                print("Not found this operator in 'optCompilerConfig.json'. ")
                return d_opt_build_args
            # parser single opt
            else:
                compile_ops_ = data.get(ops_name)
                return convert(compile_ops_)
        else:
            print("ERROR: pls use dict_format to write 'optCompilerConfig.json'! ")


def compile_ops(ops_name: str, fc_name: Optional[str] = None):
    def decorator(func):
        def wrapper(*args, **kwargs):
            d_args = get_args_of_build(ops_name)
            md_name = d_args["md_name"]
            srcs = d_args["srcs"]
            flags_extra_cc = d_args["flags_extra_cc"]
            flags_extra_hip = d_args["flags_extra_hip"]
            blob_gen_cmd = d_args["blob_gen_cmd"]
            extra_include = d_args["extra_include"]
            extra_ldflags = d_args["extra_ldflags"]
            verbose = d_args["verbose"]
            loadName = fc_name
            if fc_name is None:
                loadName = func.__name__

            try:
                module = None
                if PREBUILD_KERNELS:
                    if hasattr(ater_, loadName):
                        module = ater_
                if module is None:
                    module = get_module(md_name)
            except Exception as e:
                module = build_module(md_name, srcs, flags_extra_cc, flags_extra_hip,
                                      blob_gen_cmd, extra_include, extra_ldflags, verbose)

            return getattr(module, loadName)(*args, **kwargs)
        return wrapper
    return decorator
