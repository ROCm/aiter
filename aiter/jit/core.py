# SPDX-License-Identifier: MIT
# Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

import os
import sys
import shutil
import time
import importlib
import functools
import traceback
from typing import List, Optional
import torch
from torch.utils import cpp_extension
from torch.utils.file_baton import FileBaton
import logging
import json
from packaging.version import parse, Version

PREBUILD_KERNELS = False
if os.path.exists(os.path.dirname(os.path.abspath(__file__))+"/aiter_.so"):
    aiter_ = importlib.import_module(f'{__package__}.aiter_')
    PREBUILD_KERNELS = True
logger = logging.getLogger("aiter")

PY = sys.executable
this_dir = os.path.dirname(os.path.abspath(__file__))

AITER_CORE_DIR = os.path.abspath(f"{this_dir}/../../")

find_aiter = importlib.util.find_spec("aiter")
if find_aiter is not None:
    if find_aiter.submodule_search_locations:
        package_path = find_aiter.submodule_search_locations[0]
    elif find_aiter.origin:
        package_path = find_aiter.origin
    package_path = os.path.dirname(package_path)
    import site
    site_packages_dirs = site.getsitepackages()
    ### develop mode
    if package_path not in site_packages_dirs:
        AITER_ROOT_DIR = AITER_CORE_DIR
    ### install mode
    else:
        AITER_ROOT_DIR = os.path.abspath(f"{AITER_CORE_DIR}/aiter_meta/")
else:
    print("aiter is not installed.")

AITER_CSRC_DIR = f'{AITER_ROOT_DIR}/csrc'
os.environ["AITER_ASM_DIR"] = f'{AITER_ROOT_DIR}/hsa/'
CK_DIR = os.environ.get("CK_DIR",
                        f"{AITER_ROOT_DIR}/3rdparty/composable_kernel")
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
            logger.warning(f'---> {el} not exists!!!!!!')
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


def get_hip_version():
    return parse(torch.version.hip.split()[-1].rstrip('-').replace('-', '+'))


@functools.lru_cache(maxsize=1024)
def get_module(md_name):
    numa_balance_set = os.popen(
        "cat /proc/sys/kernel/numa_balancing").read().strip()
    if numa_balance_set == "1":
        logger.warning("WARNING: NUMA balancing is enabled, which may cause errors. "
                       "It is recommended to disable NUMA balancing by running 'sudo sh -c echo 0 > /proc/sys/kernel/numa_balancing' ")
    return importlib.import_module(f'{__package__}.{md_name}')


def build_module(md_name, srcs, flags_extra_cc, flags_extra_hip, blob_gen_cmd, extra_include, extra_ldflags, verbose):
    startTS = time.perf_counter()
    try:
        op_dir = f'{bd_dir}/{md_name}'
        logger.info(f'start build [{md_name}] under {op_dir}')

        opbd_dir = f'{op_dir}/build'
        src_dir = f'{op_dir}/build/srcs'
        os.makedirs(src_dir, exist_ok=True)
        if os.path.exists(f'{this_dir}/{md_name}.so'):
            os.remove(f'{this_dir}/{md_name}.so')

        sources = rename_cpp_to_cu(srcs, src_dir)

        flags_cc = ["-O3", "-std=c++17"]
        flags_hip = [
            "-DLEGACY_HIPBLAS_DIRECT",
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

            "-fgpu-flush-denormals-to-zero",
        ]

        # Imitate https://github.com/ROCm/composable_kernel/blob/c8b6b64240e840a7decf76dfaa13c37da5294c4a/CMakeLists.txt#L190-L214
        hip_version = get_hip_version()
        if hip_version > Version('5.7.23302'):
            flags_hip += ["-fno-offload-uniform-block"]
        if hip_version > Version('6.1.40090'):
            flags_hip += ["-mllvm", "-enable-post-misched=0"]
        if hip_version > Version('6.2.41132'):
            flags_hip += ["-mllvm", "-amdgpu-early-inline-all=true",
                          "-mllvm", "-amdgpu-function-calls=false"]
        if hip_version > Version('6.2.41133') and hip_version < Version('6.3.00000'):
            flags_hip += ["-mllvm", "-amdgpu-coerce-illegal-types=1"]

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
        rename_cpp_to_cu([f"{AITER_CSRC_DIR}/include"],
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
            verbose=verbose or int(os.getenv("AITER_LOG_MORE", 0)) > 0,
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


def get_args_of_build(ops_name: str, exclue=[]):
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
        # if d_ops["isASM"].lower() == "true":
        #     d_ops["flags_extra_hip"].append(
        #         "rf'-DAITER_ASM_DIR=\\\"{AITER_ROOT_DIR}/hsa/\\\"'")
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
                    # exclude
                    if ops_name in exclue:
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
                logger.warning("Not found this operator in 'optCompilerConfig.json'. ")
                return d_opt_build_args
            # parser single opt
            else:
                compile_ops_ = data.get(ops_name)
                return convert(compile_ops_)
        else:
            logger.warning("ERROR: pls use dict_format to write 'optCompilerConfig.json'! ")


def compile_ops(ops_name: str, fc_name: Optional[str] = None):
    def decorator(func):
        def wrapper(*args, **kwargs):
            loadName = fc_name
            if fc_name is None:
                loadName = func.__name__

            try:
                module = None
                if PREBUILD_KERNELS:
                    if hasattr(aiter_, loadName):
                        module = aiter_
                if module is None:
                    module = get_module(ops_name)
            except Exception as e:
                d_args = get_args_of_build(ops_name)
                md_name = d_args["md_name"]
                srcs = d_args["srcs"]
                flags_extra_cc = d_args["flags_extra_cc"]
                flags_extra_hip = d_args["flags_extra_hip"]
                blob_gen_cmd = d_args["blob_gen_cmd"]
                extra_include = d_args["extra_include"]
                extra_ldflags = d_args["extra_ldflags"]
                verbose = d_args["verbose"]
                module = build_module(md_name, srcs, flags_extra_cc, flags_extra_hip,
                                      blob_gen_cmd, extra_include, extra_ldflags, verbose)
            op = getattr(module, loadName)
            if int(os.getenv("AITER_LOG_MORE", 0)) == 2:
                import inspect
                callargs = inspect.getcallargs(func, *args, **kwargs)

                def getTensorInfo(el):
                    if isinstance(el, torch.Tensor):
                        return f'{el.shape} {el.dtype}'
                    return el

                callargs = [
                    f"\n        {el} = {getTensorInfo(callargs[el])}" for el in callargs]
                logger.info(
                    f"    calling {ops_name}::{loadName}({', '.join(callargs)})")

            return op(*args, **kwargs)
        return wrapper
    return decorator
