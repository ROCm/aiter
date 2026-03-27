# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import functools
import importlib
import json
import logging
import multiprocessing
import os
import re
import shutil
import sys
import time
import traceback
import types
import typing
from typing import Any, Callable, List, Optional

from packaging.version import Version, parse

this_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, f"{this_dir}/utils/")
from chip_info import get_gfx, get_gfx_list  # noqa: E402
from cpp_extension import _jit_compile, get_hip_version  # noqa: E402
from file_baton import FileBaton  # noqa: E402
from torch_guard import torch_compile_guard  # noqa: E402

AITER_REBUILD = int(os.environ.get("AITER_REBUILD", "0"))

aiter_lib = None


def mp_lock(
    lockPath: str,
    MainFunc: Callable,
    FinalFunc: Optional[Callable] = None,
    WaitFunc: Optional[Callable] = None,
):
    """
    Using FileBaton for multiprocessing.
    """
    baton = FileBaton(lockPath)
    if baton.try_acquire():
        try:
            ret = MainFunc()
        finally:
            if FinalFunc is not None:
                FinalFunc()
            baton.release()
    else:
        baton.wait()
        if WaitFunc is not None:
            ret = WaitFunc()
        ret = None
    return ret


logger = logging.getLogger("aiter")

PY = sys.executable
this_dir = os.path.dirname(os.path.abspath(__file__))

AITER_ROOT_DIR = os.path.abspath(f"{this_dir}/../../")
AITER_LOG_MORE = int(os.getenv("AITER_LOG_MORE", 0))
AITER_LOG_TUNED_CONFIG = int(os.getenv("AITER_LOG_TUNED_CONFIG", 0))


# config_env start here
AITER_CONFIG_GEMM_A4W4 = os.getenv(
    "AITER_CONFIG_GEMM_A4W4",
    f"{AITER_ROOT_DIR}/aiter/configs/a4w4_blockscale_tuned_gemm.csv",
)

AITER_CONFIG_GEMM_A8W8 = os.getenv(
    "AITER_CONFIG_GEMM_A8W8",
    f"{AITER_ROOT_DIR}/aiter/configs/a8w8_tuned_gemm.csv",
)

AITER_CONFIG_GEMM_A8W8_BPRESHUFFLE = os.getenv(
    "AITER_CONFIG_GEMM_A8W8_BPRESHUFFLE",
    f"{AITER_ROOT_DIR}/aiter/configs/a8w8_bpreshuffle_tuned_gemm.csv",
)

AITER_CONFIG_GEMM_A8W8_BLOCKSCALE = os.getenv(
    "AITER_CONFIG_GEMM_A8W8_BLOCKSCALE",
    f"{AITER_ROOT_DIR}/aiter/configs/a8w8_blockscale_tuned_gemm.csv",
)

AITER_CONFIG_FMOE = os.getenv(
    "AITER_CONFIG_FMOE",
    f"{AITER_ROOT_DIR}/aiter/configs/tuned_fmoe.csv",
)

AITER_CONFIG_GEMM_A8W8_BLOCKSCALE_BPRESHUFFLE = os.getenv(
    "AITER_CONFIG_GEMM_A8W8_BLOCKSCALE_BPRESHUFFLE",
    f"{AITER_ROOT_DIR}/aiter/configs/a8w8_blockscale_bpreshuffle_tuned_gemm.csv",
)

AITER_CONFIG_A8W8_BATCHED_GEMM = os.getenv(
    "AITER_CONFIG_A8W8_BATCHED_GEMM",
    f"{AITER_ROOT_DIR}/aiter/configs/a8w8_tuned_batched_gemm.csv",
)

AITER_CONFIG_BF16_BATCHED_GEMM = os.getenv(
    "AITER_CONFIG_BF16_BATCHED_GEMM",
    f"{AITER_ROOT_DIR}/aiter/configs/bf16_tuned_batched_gemm.csv",
)

AITER_CONFIG_GEMM_BF16 = os.getenv(
    "AITER_CONFIG_GEMM_BF16",
    f"{AITER_ROOT_DIR}/aiter/configs/bf16_tuned_gemm.csv",
)


class AITER_CONFIG(object):
    @property
    def AITER_CONFIG_GEMM_A4W4_FILE(self):
        return self.get_config_file(
            "AITER_CONFIG_GEMM_A4W4",
            AITER_CONFIG_GEMM_A4W4,
            "a4w4_blockscale_tuned_gemm",
        )

    @property
    def AITER_CONFIG_GEMM_A8W8_FILE(self):
        return self.get_config_file(
            "AITER_CONFIG_GEMM_A8W8", AITER_CONFIG_GEMM_A8W8, "a8w8_tuned_gemm"
        )

    @property
    def AITER_CONFIG_GEMM_A8W8_BPRESHUFFLE_FILE(self):
        return self.get_config_file(
            "AITER_CONFIG_GEMM_A8W8_BPRESHUFFLE",
            AITER_CONFIG_GEMM_A8W8_BPRESHUFFLE,
            "a8w8_bpreshuffle_tuned_gemm",
        )

    @property
    def AITER_CONFIG_GEMM_A8W8_BLOCKSCALE_FILE(self):
        return self.get_config_file(
            "AITER_CONFIG_GEMM_A8W8_BLOCKSCALE",
            AITER_CONFIG_GEMM_A8W8_BLOCKSCALE,
            "a8w8_blockscale_tuned_gemm",
        )

    @property
    def AITER_CONFIG_FMOE_FILE(self):
        return self.get_config_file(
            "AITER_CONFIG_FMOE", AITER_CONFIG_FMOE, "tuned_fmoe"
        )

    @property
    def AITER_CONFIG_GEMM_A8W8_BLOCKSCALE_BPRESHUFFLE_FILE(self):
        return self.get_config_file(
            "AITER_CONFIG_GEMM_A8W8_BLOCKSCALE_BPRESHUFFLE",
            AITER_CONFIG_GEMM_A8W8_BLOCKSCALE_BPRESHUFFLE,
            "a8w8_blockscale_bpreshuffle_tuned_gemm",
        )

    @property
    def AITER_CONFIG_A8W8_BATCHED_GEMM_FILE(self):
        return self.get_config_file(
            "AITER_CONFIG_A8W8_BATCHED_GEMM",
            AITER_CONFIG_A8W8_BATCHED_GEMM,
            "a8w8_tuned_batched_gemm",
        )

    @property
    def AITER_CONFIG_BF16_BATCHED_GEMM_FILE(self):
        return self.get_config_file(
            "AITER_CONFIG_BF16_BATCHED_GEMM",
            AITER_CONFIG_BF16_BATCHED_GEMM,
            "bf16_tuned_batched_gemm",
        )

    @property
    def AITER_CONFIG_GEMM_BF16_FILE(self):
        return self.get_config_file(
            "AITER_CONFIG_GEMM_BF16", AITER_CONFIG_GEMM_BF16, "bf16_tuned_gemm"
        )

    def update_config_files(self, file_path: str, merge_name: str):
        path_list = file_path.split(os.pathsep) if file_path else []
        if len(path_list) <= 1:
            return file_path
        df_list = []
        ## merge config files
        ##example: AITER_CONFIG_GEMM_A4W4="/path1:/path2"
        import pandas as pd

        df_list.append(pd.read_csv(path_list[0]))
        for i, path in enumerate(path_list[1:]):
            if os.path.exists(path):
                df = pd.read_csv(path)
                base_cols = [c for c in df_list[0].columns if c != "_tag"]
                new_cols = [c for c in df.columns if c != "_tag"]
                assert (
                    base_cols == new_cols
                ), f"Column mismatch between {path_list[0]} and {path}, {base_cols}, {new_cols}"

                df_list.append(df)
            else:
                logger.info(f"path {i+1}: {path} (not exist)")
        merge_df = (
            pd.concat([df for df in df_list if not df.empty], ignore_index=True)
            if df_list
            else pd.DataFrame()
        )
        has_tag = "_tag" in merge_df.columns
        if has_tag:
            merge_df["_tag"] = merge_df["_tag"].fillna("")

        ## get keys from untuned file to drop_duplicates
        untuned_name = (
            re.sub(r"(?:_)?tuned$", r"\1untuned", merge_name)
            if re.search(r"(?:_)?tuned$", merge_name)
            else merge_name.replace("tuned", "untuned")
        )
        untuned_path = f"{AITER_ROOT_DIR}/aiter/configs/{untuned_name}.csv"
        if os.path.exists(untuned_path):
            untunedf = pd.read_csv(untuned_path)
            keys = untunedf.columns.to_list()
            if "cu_num" not in keys:
                keys.append("cu_num")
            dedup_keys = keys + ["_tag"] if has_tag else keys
            sorted_df = merge_df.sort_values("us")
            duplicated_mask = sorted_df.duplicated(subset=dedup_keys, keep="first")
            if duplicated_mask.any():
                dup_rows = sorted_df[duplicated_mask]
                logger.warning(
                    f"Dropping {len(dup_rows)} duplicate rows during merge of '{merge_name}':\n"
                    f"{dup_rows.to_string(index=False)}"
                )
            merge_df = sorted_df.drop_duplicates(
                subset=dedup_keys, keep="first"
            ).reset_index(drop=True)
        else:
            logger.warning(
                f"Untuned config file not found: {untuned_path}. Using all columns for deduplication."
            )
        from pathlib import Path

        config_path = Path("/tmp/aiter_configs/")
        if not config_path.exists():
            config_path.mkdir(parents=True, exist_ok=True)
        new_file_path = f"{config_path}/{merge_name}.csv"
        lock_path = f"{new_file_path}.lock"
        tmp_file_path = f"{new_file_path}.tmp"

        def write_config():
            merge_df.to_csv(tmp_file_path, index=False)
            os.replace(tmp_file_path, new_file_path)

        mp_lock(lock_path, write_config)
        return new_file_path

    @functools.lru_cache(maxsize=20)
    def get_config_file(self, env_name, default_file, tuned_file_name):
        config_env_file = os.getenv(env_name)
        # default_file = f"{AITER_ROOT_DIR}/aiter/configs/{tuned_file_name}.csv"
        from pathlib import Path

        if not config_env_file:
            model_config_dir = Path(f"{AITER_ROOT_DIR}/aiter/configs/model_configs/")
            op_tuned_file_list = [
                p
                for p in model_config_dir.glob(f"*{tuned_file_name}*")
                if (p.is_file() and "untuned" not in str(p))
            ]

            if not op_tuned_file_list:
                config_file = default_file
            else:
                tuned_files = ":".join(str(p) for p in op_tuned_file_list)
                tuned_files = default_file + ":" + tuned_files
                logger.info(
                    f"merge tuned file under model_configs/ and configs/ {tuned_files}"
                )
                config_file = self.update_config_files(tuned_files, tuned_file_name)
        else:
            config_file = self.update_config_files(config_env_file, tuned_file_name)
            # print(f"get config file from environment ", config_file)
        return config_file


AITER_CONFIGS = AITER_CONFIG()
# config_env end here

find_aiter = importlib.util.find_spec("aiter")
if find_aiter is not None:
    if find_aiter.submodule_search_locations:
        package_path = find_aiter.submodule_search_locations[0]
    elif find_aiter.origin:
        package_path = find_aiter.origin
    package_path = os.path.dirname(package_path)
    package_parent_path = os.path.dirname(package_path)

    try:
        with open(f"{this_dir}/../install_mode", "r") as f:
            # develop mode
            isDevelopMode = f.read().strip() == "develop"
    except FileNotFoundError:
        # pip install -e
        isDevelopMode = True

    if isDevelopMode:
        AITER_META_DIR = AITER_ROOT_DIR
    else:
        AITER_META_DIR = os.path.abspath(f"{AITER_ROOT_DIR}/aiter_meta/")
else:
    AITER_META_DIR = AITER_ROOT_DIR
    logger.warning("aiter is not installed.")

# honor environment override and fallback if missing
env_meta = os.environ.get("AITER_META_DIR")
if env_meta:
    AITER_META_DIR = os.path.abspath(env_meta)
if not os.path.exists(os.path.join(AITER_META_DIR, "csrc")):
    AITER_META_DIR = AITER_ROOT_DIR

sys.path.insert(0, AITER_META_DIR)
AITER_CSRC_DIR = f"{AITER_META_DIR}/csrc"
AITER_GRADLIB_DIR = f"{AITER_META_DIR}/gradlib"
gfxs = get_gfx_list()
AITER_ASM_DIR = f"{AITER_META_DIR}/hsa/"
os.environ["AITER_ASM_DIR"] = AITER_ASM_DIR

CK_3RDPARTY_DIR = os.environ.get(
    "CK_DIR", f"{AITER_META_DIR}/3rdparty/composable_kernel"
)
CK_HELPER_DIR = f"{AITER_META_DIR}/3rdparty/ck_helper"
CK_DIR = CK_3RDPARTY_DIR
HIP_KITTENS_DIR = os.environ.get(
    "HIP_KITTENS_DIR", f"{AITER_META_DIR}/3rdparty/HipKittens"
)


@functools.lru_cache(maxsize=1)
def get_asm_dir():
    return os.path.join(AITER_ASM_DIR, get_gfx())


@functools.lru_cache(maxsize=1)
def get_user_jit_dir() -> str:
    if "AITER_JIT_DIR" in os.environ:
        path = os.getenv("AITER_JIT_DIR", "")
        os.makedirs(path, exist_ok=True)
        sys.path.insert(0, path)
        return path
    else:
        if os.access(this_dir, os.W_OK):
            return this_dir
    home_jit_dir = f"{os.path.expanduser('~')}/.aiter/{os.path.basename(this_dir)}"
    if not os.path.exists(home_jit_dir):
        shutil.copytree(this_dir, home_jit_dir)
    return home_jit_dir


bd_dir = f"{get_user_jit_dir()}/build"
# copy ck to build, thus hippify under bd_dir
if multiprocessing.current_process().name == "MainProcess":
    os.makedirs(bd_dir, exist_ok=True)
    # if os.path.exists(f"{bd_dir}/ck/library"):
    #     shutil.rmtree(f"{bd_dir}/ck/library")
# CK_DIR = f"{bd_dir}/ck"


def validate_and_update_archs():
    archs = os.getenv("GPU_ARCHS", "native").split(";")
    archs = [arch.strip() for arch in archs]
    # List of allowed architectures
    allowed_archs = [
        "native",
        "gfx90a",
        "gfx940",
        "gfx941",
        "gfx942",
        "gfx1100",
        "gfx1101",
        "gfx1102",
        "gfx1103",
        "gfx1150",
        "gfx1151",
        "gfx1152",
        "gfx1153",
        "gfx1200",
        "gfx1201",
        "gfx1250",
        "gfx950",
        "gfx1250",
    ]

    # Validate if each element in archs is in allowed_archs
    assert all(
        arch in allowed_archs for arch in archs
    ), f"One of GPU archs of {archs} is invalid or not supported"
    return archs


@functools.lru_cache()
def hip_flag_checker(flag_hip: str) -> bool:
    import subprocess

    cmd = (
        ["hipcc"]
        + flag_hip.split()
        + ["-x", "hip", "-E", "-P", "/dev/null", "-o", "/dev/null"]
    )
    try:
        subprocess.check_output(cmd, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError:
        logger.warning(f"Current hipcc not support: {flag_hip}, skip it.")
        return False
    return True


@functools.lru_cache()
def check_LLVM_MAIN_REVISION():
    # for https://github.com/ROCm/ROCm/issues/5646 and https://github.com/ROCm/composable_kernel/pull/3469
    # ck using following logic...
    """#if LLVM_MAIN_REVISION < 554785
    #define CK_TILE_HOST_DEVICE_EXTERN __host__ __device__
    #else
    #define CK_TILE_HOST_DEVICE_EXTERN"""
    import subprocess

    cmd = """echo "#include <tuple>
__host__ __device__ void func(){std::tuple<int, int> t = std::tuple(1, 1);}" | hipcc -x hip -P -c -Wno-unused-command-line-argument -o /dev/null -"""
    try:
        subprocess.check_output(cmd, shell=True, text=True, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError:
        return 554785
    return 554785 - 1


def check_and_set_ninja_worker():
    max_num_jobs_cores = max(1, os.cpu_count() * 0.8)
    import psutil

    # calculate the maximum allowed NUM_JOBS based on free memory
    free_memory_gb = psutil.virtual_memory().available / (1024**3)  # free memory in GB
    max_num_jobs_memory = int(free_memory_gb / 0.5)  # assuming 0.5 GB per job

    # pick lower value of jobs based on cores vs memory metric to minimize oom and swap usage during compilation
    max_jobs = int(max(1, min(max_num_jobs_cores, max_num_jobs_memory)))
    max_jobs_env = os.environ.get("MAX_JOBS")
    if max_jobs_env is not None:
        try:
            max_processes = int(max_jobs_env)
            # too large value
            if max_processes > max_jobs:
                os.environ["MAX_JOBS"] = str(max_jobs)
        # error value
        except ValueError:
            os.environ["MAX_JOBS"] = str(max_jobs)
    # none value
    else:
        os.environ["MAX_JOBS"] = str(max_jobs)


def rename_cpp_to_cu(els, dst, hipify, recursive=False):
    def do_rename_and_mv(name, src, dst, ret):
        newName = name
        if hipify:
            if name.endswith(".cpp") or name.endswith(".cu"):
                newName = name.replace(".cpp", ".cu")
                ret.append(f"{dst}/{newName}")
            shutil.copy(f"{src}/{name}", f"{dst}/{newName}")
        else:
            if name.endswith(".cpp") or name.endswith(".cu"):
                ret.append(f"{src}/{newName}")

    ret = []
    for el in els:
        if not os.path.exists(el):
            logger.warning(f"---> {el} not exists!!!!!!")
            continue
        if os.path.isdir(el):
            for entry in os.listdir(el):
                if os.path.isdir(f"{el}/{entry}"):
                    if recursive:
                        ret += rename_cpp_to_cu(
                            [f"{el}/{entry}"], dst, hipify, recursive
                        )
                    continue
                do_rename_and_mv(entry, el, dst, ret)
        else:
            do_rename_and_mv(os.path.basename(el), os.path.dirname(el), dst, ret)
    return ret


@torch_compile_guard()
def check_numa_custom_op() -> None:
    numa_balance_set = os.popen("cat /proc/sys/kernel/numa_balancing").read().strip()
    if numa_balance_set == "1":
        logger.warning(
            "WARNING: NUMA balancing is enabled, which may cause errors. "
            "It is recommended to disable NUMA balancing by running \"sudo sh -c 'echo 0 > /proc/sys/kernel/numa_balancing'\" "
            "for more details: https://rocm.docs.amd.com/en/latest/how-to/system-optimization/mi300x.html#disable-numa-auto-balancing"
        )


@functools.lru_cache()
def check_numa():
    check_numa_custom_op()


__mds = {}


@torch_compile_guard()
def get_module_custom_op(md_name: str) -> None:
    global __mds
    if md_name not in __mds:
        if "AITER_JIT_DIR" in os.environ:
            __mds[md_name] = importlib.import_module(md_name)
        else:
            __mds[md_name] = importlib.import_module(f"{__package__}.{md_name}")
        logger.info(f"import [{md_name}] under {__mds[md_name].__file__}")
    return


@functools.lru_cache(maxsize=1024)
def get_module(md_name):
    check_numa()
    get_module_custom_op(md_name)
    return __mds[md_name]


rebuilded_list = ["module_aiter_enum"]


def clone_3rdparty(third_party: str) -> None:
    def MainFunc():
        if not os.path.exists(dir_path):
            import subprocess

            def check_git_version(required_major, required_minor):
                try:
                    output = subprocess.check_output(
                        ["git", "--version"], text=True
                    ).strip()
                    import re

                    m = re.search(r"(\d+)\.(\d+)", output)
                    if m:
                        major, minor = int(m.group(1)), int(m.group(2))
                        return (major > required_major) or (
                            major == required_major and minor >= required_minor
                        )
                except Exception as e:
                    logger.warning(f"Failed to check git version: {e}")
                return False

            logger.info(f"Cloning 3rdparty {third_party} to {dir_path}")
            # Check git version for --revision flag support (>=2.49)
            if not check_git_version(2, 49):
                logger.warning(
                    "Your git version does not support the --revision flag (requires >=2.49). Slow path is used for cloning 3rdparty."
                )
                subprocess.call(
                    [
                        "git",
                        "clone",
                        "-q",
                        third_party_info["url"],
                        dir_path,
                    ]
                )
                subprocess.call(
                    [
                        "git",
                        "-C",
                        dir_path,
                        "reset",
                        "-q",
                        "--hard",
                        third_party_info["commit"],
                    ]
                )
                subprocess.call(
                    [
                        "git",
                        "-C",
                        dir_path,
                        "submodule",
                        "update",
                        "-q",
                        "--init",
                        "--recursive",
                    ]
                )
            else:
                # Save current git config value for advice.detachedHead, set to false
                prev_detached_head = None
                try:
                    try:
                        prev_detached_head = subprocess.check_output(
                            ["git", "config", "--get", "advice.detachedHead"], text=True
                        ).strip()
                    except subprocess.CalledProcessError:
                        prev_detached_head = None  # not set before
                    # Set to false before clone
                    subprocess.call(
                        ["git", "config", "--global", "advice.detachedHead", "false"]
                    )

                    subprocess.call(
                        [
                            "git",
                            "clone",
                            "-q",
                            f"--revision={third_party_info['commit']}",
                            "--depth=1",
                            "--recurse-submodules",
                            third_party_info["url"],
                            dir_path,
                        ]
                    )
                finally:
                    # Restore config after clone
                    if prev_detached_head is not None:
                        subprocess.call(
                            [
                                "git",
                                "config",
                                "--global",
                                "advice.detachedHead",
                                prev_detached_head,
                            ]
                        )
                    else:
                        subprocess.call(
                            [
                                "git",
                                "config",
                                "--global",
                                "--unset",
                                "advice.detachedHead",
                            ]
                        )

    if third_party == "HipKittens":
        dir_path = HIP_KITTENS_DIR
        third_party_info = {
            "url": "https://github.com/HazyResearch/HipKittens.git",
            "commit": "b027c06ba935b80a53a7c7f7f82c0f9cbd0bf3cb",
        }
    elif third_party == "ComposableKernel":
        # TODO: ComposableKernel will be supported in the future
        pass

    if "third_party_info" in locals():
        lock_path = f"{bd_dir}/lock_3rdparty_clone_{third_party}"
        mp_lock(lockPath=lock_path, MainFunc=MainFunc)


def rm_module(md_name):
    os.system(f"rm -rf {get_user_jit_dir()}/{md_name}.so")


def clear_build(md_name):
    os.system(f"rm -rf {bd_dir}/{md_name}")


def build_module(
    md_name,
    srcs,
    flags_extra_cc,
    flags_extra_hip,
    blob_gen_cmd,
    extra_include,
    extra_ldflags,
    verbose,
    is_python_module,
    is_standalone,
    torch_exclude,
    third_party,
    hipify=False,
):
    os.makedirs(bd_dir, exist_ok=True)
    lock_path = f"{bd_dir}/lock_{md_name}"
    startTS = time.perf_counter()
    target_name = f"{md_name}.so" if not is_standalone else md_name

    for tp in third_party:
        clone_3rdparty(tp)

    def MainFunc():
        if AITER_REBUILD == 1:
            rm_module(md_name)
            clear_build(md_name)
        elif AITER_REBUILD >= 2:
            rm_module(md_name)
        op_dir = f"{bd_dir}/{md_name}"
        logger.info(f"start build [{md_name}] under {op_dir}")

        opbd_dir = f"{op_dir}/build"
        src_dir = f"{op_dir}/build/srcs"
        os.makedirs(src_dir, exist_ok=True)
        if os.path.exists(f"{get_user_jit_dir()}/{target_name}"):
            os.remove(f"{get_user_jit_dir()}/{target_name}")

        sources = rename_cpp_to_cu(srcs, src_dir, hipify)

        flags_cc = ["-O3", "-std=c++20", "-Wno-unknown-warning-option"]
        flags_hip = [
            "-DLEGACY_HIPBLAS_DIRECT",
            "-DUSE_PROF_API=1",
            "-D__HIP_PLATFORM_HCC__=1",
            "-D__HIP_PLATFORM_AMD__=1",
            "-U__HIP_NO_HALF_CONVERSIONS__",
            "-U__HIP_NO_HALF_OPERATORS__",
            "-mllvm --amdgpu-kernarg-preload-count=16",
            # "-v --save-temps",
            "-Wno-unused-result",
            "-Wno-switch-bool",
            "-Wno-vla-cxx-extension",
            "-Wno-undefined-func-template",
            "-Wno-macro-redefined",
            "-Wno-missing-template-arg-list-after-template-kw",
            "-fgpu-flush-denormals-to-zero",
            f"-DDLLVM_MAIN_REVISION={check_LLVM_MAIN_REVISION()}",
        ]

        # Imitate https://github.com/ROCm/composable_kernel/blob/c8b6b64240e840a7decf76dfaa13c37da5294c4a/CMakeLists.txt#L190-L214
        hip_version = parse(get_hip_version().split()[-1].rstrip("-").replace("-", "+"))
        if hip_version <= Version("6.3.42132"):
            flags_hip += ["-mllvm --amdgpu-enable-max-ilp-scheduling-strategy=1"]
        if hip_version > Version("5.5.00000"):
            flags_hip += ["-mllvm --lsr-drop-solution=1"]
        if hip_version > Version("5.7.23302"):
            flags_hip += ["-fno-offload-uniform-block"]
        if hip_version > Version("6.1.40090"):
            flags_hip += ["-mllvm -enable-post-misched=0"]
        if hip_version > Version("6.2.41132"):
            flags_hip += [
                "-mllvm -amdgpu-early-inline-all=true",
                "-mllvm -amdgpu-function-calls=false",
            ]
        if hip_version > Version("6.2.41133"):
            flags_hip += ["-mllvm -amdgpu-coerce-illegal-types=1"]
        if get_gfx() == "gfx950" and int(os.getenv("AITER_FP4x2", "1")) > 0:
            flags_hip += ["-D__Float4_e2m1fn_x2"]

        if not torch_exclude:
            import torch

            if hasattr(torch, "float4_e2m1fn_x2"):
                flags_hip += ["-DTORCH_Float4_e2m1fn_x2"]

        enable_ck = int(os.environ.get("ENABLE_CK", "1"))
        if not any("ENABLE_CK" in f for f in flags_extra_cc):
            flags_cc.append(f"-DENABLE_CK={enable_ck}")

        flags_cc += flags_extra_cc
        flags_hip += flags_extra_hip
        archs = validate_and_update_archs()
        flags_hip += [f"--offload-arch={arch}" for arch in archs]
        flags_hip = sorted(set(flags_hip))  # remove same flags
        flags_hip = [el for el in flags_hip if hip_flag_checker(el)]
        check_and_set_ninja_worker()

        def exec_blob(blob_gen_cmd, op_dir, src_dir, sources):
            if blob_gen_cmd:
                blob_dir = f"{op_dir}/blob/"
                os.makedirs(blob_dir, exist_ok=True)
                if AITER_LOG_MORE:
                    logger.info(f"exec_blob ---> {PY} {blob_gen_cmd.format(blob_dir)}")
                os.system(f"{PY} {blob_gen_cmd.format(blob_dir)}")
                sources += rename_cpp_to_cu([blob_dir], src_dir, hipify, recursive=True)
            return sources

        if isinstance(blob_gen_cmd, list):
            for s_blob_gen_cmd in blob_gen_cmd:
                sources = exec_blob(s_blob_gen_cmd, op_dir, src_dir, sources)
        else:
            sources = exec_blob(blob_gen_cmd, op_dir, src_dir, sources)

        extra_include_paths = []

        _is_ckfree = not os.path.isdir(CK_3RDPARTY_DIR)
        if not _is_ckfree:
            extra_include_paths += [
                f"{CK_HELPER_DIR}",
                f"{CK_3RDPARTY_DIR}/include",
                f"{CK_3RDPARTY_DIR}/library/include",
            ]
        else:
            # When CK is not available, define AITER_CK_FREE for all modules
            # so headers use lightweight shims instead of ck_tile/core.hpp
            flags_cc.append("-DAITER_CK_FREE=1")

        if os.path.isdir(HIP_KITTENS_DIR):
            extra_include_paths += [
                f"{HIP_KITTENS_DIR}/include",
            ]

        extra_include_paths = [p for p in extra_include_paths if os.path.isdir(str(p))]

        if not hipify:
            _extra_inc = extra_include
            if _is_ckfree:
                _extra_inc = [p for p in extra_include if os.path.isdir(str(p))]
            extra_include_paths += [
                f"{AITER_CSRC_DIR}/include",
                f"{op_dir}/blob",
            ] + _extra_inc
            if not is_standalone:
                extra_include_paths += [f"{AITER_CSRC_DIR}/include/torch"]
        else:
            old_bd_include_dir = f"{op_dir}/build/include"
            extra_include_paths.append(old_bd_include_dir)
            os.makedirs(old_bd_include_dir, exist_ok=True)
            rename_cpp_to_cu(
                [f"{AITER_CSRC_DIR}/include"] + extra_include,
                old_bd_include_dir,
                hipify,
            )

            if not is_standalone:
                bd_include_dir = f"{op_dir}/build/include/torch"
                os.makedirs(bd_include_dir, exist_ok=True)
                rename_cpp_to_cu(
                    [f"{AITER_CSRC_DIR}/include/torch"],
                    bd_include_dir,
                    hipify,
                )

        try:
            _jit_compile(
                md_name,
                sorted(set(sources)),
                extra_cflags=flags_cc,
                extra_cuda_cflags=flags_hip,
                extra_ldflags=extra_ldflags,
                extra_include_paths=extra_include_paths,
                build_directory=opbd_dir,
                verbose=verbose or AITER_LOG_MORE > 0,
                with_cuda=True,
                is_python_module=is_python_module,
                is_standalone=is_standalone,
                torch_exclude=torch_exclude,
                hipify=hipify,
            )
            if is_python_module and not is_standalone:
                shutil.copy(f"{opbd_dir}/{target_name}", f"{get_user_jit_dir()}")
            else:
                shutil.copy(
                    f"{opbd_dir}/{target_name}", f"{AITER_ROOT_DIR}/op_tests/cpp/mha"
                )
        except Exception as e:
            tag = f"\033[31mfailed jit build [{md_name}]\033[0m"
            logger.error(
                f"{tag}\u2193\u2193\u2193\u2193\u2193\u2193\u2193\u2193\u2193\u2193\n-->[History]: {{}}{tag}\u2191\u2191\u2191\u2191\u2191\u2191\u2191\u2191\u2191\u2191".format(
                    re.sub(
                        "error:",
                        "\033[31merror:\033[0m",
                        "-->".join(traceback.format_exception(*sys.exc_info())),
                        flags=re.I,
                    ),
                )
            )
            raise RuntimeError(
                f"[aiter] build [{md_name}] under {opbd_dir} failed !!!!!!"
            ) from e

    def FinalFunc():
        logger.info(
            f"\033[32mfinish build [{md_name}], cost {time.perf_counter()-startTS:.1f}s \033[0m"
        )

    mp_lock(lockPath=lock_path, MainFunc=MainFunc, FinalFunc=FinalFunc)


def _get_ck_exclude_modules():
    """Return set of module names that require CK and should be excluded in CK-free builds.

    Combines two detection methods:
    1. Config pattern matching -- modules whose optCompilerConfig.json entry references
       CK_DIR, py_itfs_ck, gen_instances, or generate.py
    2. Hardcoded list -- modules with deep ck_tile:: source-level dependencies that
       aren't caught by config pattern matching

    V3 ASM modules are exempted because they build with shim headers only.
    """
    cfg_path = os.path.join(this_dir, "optCompilerConfig.json")
    try:
        with open(cfg_path, "r", encoding="utf-8") as f:
            config_data = json.load(f)
    except Exception:
        config_data = {}

    # Pattern-matched CK modules
    ck_patterns = ["CK_DIR", "py_itfs_ck", "gen_instances", "generate.py"]
    ck_modules = set()
    for mod_name, mod_cfg in config_data.items():
        mod_str = json.dumps(mod_cfg)
        if any(p in mod_str for p in ck_patterns):
            ck_modules.add(mod_name)

    # V3 ASM modules can build with shim headers -- exempt them
    v3_flags = ["FAV3_ON", "ONLY_FAV3"]
    for mod_name, mod_cfg in config_data.items():
        flags_str = json.dumps(mod_cfg.get("flags_extra_cc", []))
        if any(f in flags_str for f in v3_flags):
            ck_modules.discard(mod_name)

    # Modules with deep ck_tile:: source-level deps not caught by config patterns
    ck_modules |= {
        "module_activation",
        "module_cache",
        "module_custom_all_reduce",
        "module_fused_qk_norm_mrope_cache_quant_shuffle",
        "module_fused_qk_norm_rope_cache_quant_shuffle",
        "module_mla_metadata",
        "module_mla_reduce",
        "module_moe_asm",
        "module_pa",
        "module_pa_metadata",
        "module_pa_ragged",
        "module_pa_v1",
        "module_ps_metadata",
        "module_quant",
        "module_rmsnorm_quant",
        "module_rope_general_bwd",
        "module_rope_general_fwd",
        "module_rope_pos_fwd",
        "module_sample",
        "module_topk_plain",
    }

    return ck_modules


def get_args_of_build(ops_name: str, exclude=[]):
    d_opt_build_args = {
        "srcs": [],
        "md_name": "",
        "flags_extra_cc": [],
        "flags_extra_hip": [],
        "extra_ldflags": None,
        "extra_include": [],
        "verbose": False,
        "is_python_module": True,
        "is_standalone": False,
        "torch_exclude": False,
        "hip_clang_path": None,
        "blob_gen_cmd": "",
        "third_party": [],
    }

    def convert(d_ops: dict):
        for k, val in d_ops.items():
            if isinstance(val, list):
                for idx, el in enumerate(val):
                    if isinstance(el, str):
                        if "torch" in el:
                            import torch as torch
                        val[idx] = eval(el)
                d_ops[k] = val
            elif isinstance(val, str):
                d_ops[k] = eval(val)
            else:
                pass

        # undefined compile features will be replaced with default value
        d_opt_build_args.update(d_ops)
        return d_opt_build_args

    with open(this_dir + "/optCompilerConfig.json", "r") as file:
        data = json.load(file)
        if isinstance(data, dict):
            # parse all ops, return list
            if ops_name == "all":
                # Auto-exclude CK-dependent modules in CK-free builds
                if not os.path.isdir(CK_3RDPARTY_DIR):
                    ck_excludes = _get_ck_exclude_modules()
                    exclude = list(set(exclude) | ck_excludes)
                    logger.info(
                        f"[CK-free] Auto-excluding {len(ck_excludes)} CK-dependent modules"
                    )
                all_ops_list = []
                d_all_ops = {
                    "flags_extra_cc": [],
                    "flags_extra_hip": [],
                    "extra_include": [],
                    "blob_gen_cmd": [],
                }
                # traverse opts
                for ops_name, d_ops in data.items():
                    # Cannot contain tune ops
                    if ops_name.endswith("tune"):
                        continue
                    # exclude
                    if ops_name in exclude:
                        continue
                    single_ops = convert(d_ops)
                    # exclude experimental ops if AITER_ENABLE_EXPERIMENTAL is not set
                    if not os.getenv("AITER_ENABLE_EXPERIMENTAL", False):
                        if single_ops.get("is_experimental", False):
                            continue
                    d_single_ops = {
                        "md_name": ops_name,
                        "srcs": single_ops["srcs"],
                        "flags_extra_cc": single_ops["flags_extra_cc"],
                        "flags_extra_hip": single_ops["flags_extra_hip"],
                        "extra_include": single_ops["extra_include"],
                        "blob_gen_cmd": single_ops["blob_gen_cmd"],
                        "third_party": single_ops["third_party"],
                    }
                    for k in d_all_ops.keys():
                        if isinstance(single_ops[k], list):
                            d_all_ops[k] += single_ops[k]
                        elif isinstance(single_ops[k], str) and single_ops[k] != "":
                            d_all_ops[k].append(single_ops[k])
                    all_ops_list.append(d_single_ops)

                return all_ops_list, d_all_ops
            # no find opt_name in json.
            elif data.get(ops_name) is None:
                logger.warning(
                    "Not found this operator ("
                    + ops_name
                    + ") in 'optCompilerConfig.json'. "
                )
                return d_opt_build_args
            # parser single opt
            else:
                compile_ops_ = data.get(ops_name)
                return convert(compile_ops_)
        else:
            logger.warning(
                "ERROR: pls use dict_format to write 'optCompilerConfig.json'! "
            )


def _is_union(origin):
    """Check for both typing.Union (Optional[X]) and types.UnionType (X | None)."""
    return origin is typing.Union or origin is types.UnionType

MANUAL_SCHEMA_OPS = [
    "register_graph_buffers",
    "module_moe_ck2stages",
    "mha_fwd",
    "fmha_v3_fwd",
    "mha_varlen_fwd",
    "mha_bwd",
    "fmha_v3_bwd",
    "mha_varlen_bwd",
    "fmha_v3_varlen_bwd",
    "mha_batch_prefill",
    "hipb_findallsols",
    "rocb_findallsols",
    "_ActivationType",
    "_QuantType",
    "init_custom_ar",
]

NONE_WRAPPED_OP = [
    "hipb_create_extension",
    "hipb_destroy_extension",
    "getHipblasltKernelName",
    "rocb_create_extension",
    "rocb_destroy_extension",
    "get_meta_buffer_ipc_handle",
    "get_graph_buffer_ipc_meta",
    "_ActivationType",
    "_QuantType",
    "allocate_meta_buffer",
    "dispose",
    "meta_size",
    "get_padded_m",
    "compile_mha_fwd",
    "compile_mha_bwd",
]


def _ctypes_call(func, fc_name, md_name):
    """Build a ctypes-based caller for a torch-free .so module.

    Type-hint to C ABI mapping
    -------------------------------------------------------
    Python annotation     | ctypes type          | C type
    ----------------------|----------------------|---------
    Tensor                | POINTER(aiter_tensor_t) | aiter_tensor_t*
    Optional[Tensor]      | POINTER(aiter_tensor_t) | aiter_tensor_t* (NULL if None)
    int                   | c_int                | int
    Optional[int]         | c_int                | int   (-1 if None)
    str                   | c_char_p             | char* (.encode())
    Optional[str]         | c_char_p             | char* (NULL if None)
    bool                  | c_int                | int   (0 / 1)
    float                 | c_float              | float
    (other)               | c_void_p             | void*
    (auto-appended)       | c_void_p             | hipStream_t
    -------------------------------------------------------
    """
    import ctypes
    import inspect

    import torch

    from ..utility.dtypes import aiter_tensor_t, torch_to_aiter

    _cache = {}
    _arg_checked = False

    def _ensure_loaded():
        if _cache:
            return
        so_path = os.path.join(get_user_jit_dir(), f"{md_name}.so")
        if not os.path.exists(so_path):
            d_args = get_args_of_build(md_name)
            d_args["torch_exclude"] = True
            build_module(
                md_name,
                d_args["srcs"],
                d_args["flags_extra_cc"],
                d_args["flags_extra_hip"],
                d_args["blob_gen_cmd"],
                d_args["extra_include"],
                d_args["extra_ldflags"],
                d_args["verbose"],
                d_args["is_python_module"],
                d_args["is_standalone"],
                d_args["torch_exclude"],
                d_args.get("third_party", []),
            )
        lib = ctypes.CDLL(so_path)
        c_func = getattr(lib, fc_name)

        hints = typing.get_type_hints(func)

        ret_hint = hints.get("return")
        if ret_hint is int:
            c_func.restype = ctypes.c_int
        elif ret_hint is float:
            c_func.restype = ctypes.c_float
        else:
            c_func.restype = None

        argtypes = []
        has_tensor = False
        for pname in inspect.signature(func).parameters:
            hint = hints.get(pname)
            origin = typing.get_origin(hint)
            type_args = typing.get_args(hint)
            if hint is torch.Tensor:
                argtypes.append(ctypes.POINTER(aiter_tensor_t))
                has_tensor = True
            elif _is_union(origin) and torch.Tensor in type_args:
                argtypes.append(ctypes.POINTER(aiter_tensor_t))
                has_tensor = True
            elif _is_union(origin) and int in type_args:
                argtypes.append(ctypes.c_int)
            elif _is_union(origin) and str in type_args:
                argtypes.append(ctypes.c_char_p)
            elif hint is str:
                argtypes.append(ctypes.c_char_p)
            elif hint is bool:
                argtypes.append(ctypes.c_int)
            elif hint is int:
                argtypes.append(ctypes.c_int)
            elif hint is float:
                argtypes.append(ctypes.c_float)
            else:
                argtypes.append(ctypes.c_void_p)
        if has_tensor:
            argtypes.append(ctypes.c_void_p)  # hipStream_t
        c_func.argtypes = argtypes

        _cache["lib"] = lib
        _cache["c_func"] = c_func
        _cache["has_tensor"] = has_tensor

    def _check_args_before_convert(bound_args, hints):
        for pname, value in bound_args.items():
            hint = hints.get(pname)
            origin = typing.get_origin(hint)
            type_args = typing.get_args(hint)

            if hint is torch.Tensor:
                if not isinstance(value, torch.Tensor):
                    raise TypeError(
                        f"{fc_name}: '{pname}' expects torch.Tensor, "
                        f"got {type(value).__name__}"
                    )
            elif _is_union(origin) and torch.Tensor in type_args:
                if value is not None and not isinstance(value, torch.Tensor):
                    raise TypeError(
                        f"{fc_name}: '{pname}' expects Optional[torch.Tensor], "
                        f"got {type(value).__name__}"
                    )
            elif _is_union(origin) and int in type_args:
                if value is not None and not isinstance(value, int):
                    raise TypeError(
                        f"{fc_name}: '{pname}' expects Optional[int], "
                        f"got {type(value).__name__}"
                    )
            elif _is_union(origin) and str in type_args:
                if value is not None and not isinstance(value, str):
                    raise TypeError(
                        f"{fc_name}: '{pname}' expects Optional[str], "
                        f"got {type(value).__name__}"
                    )
            elif hint is str:
                if not isinstance(value, str):
                    raise TypeError(
                        f"{fc_name}: '{pname}' expects str, "
                        f"got {type(value).__name__}"
                    )
            elif hint is bool:
                if not isinstance(value, (bool, int)):
                    raise TypeError(
                        f"{fc_name}: '{pname}' expects bool, "
                        f"got {type(value).__name__}"
                    )
            elif hint is int:
                if not isinstance(value, int):
                    raise TypeError(
                        f"{fc_name}: '{pname}' expects int, "
                        f"got {type(value).__name__}"
                    )
            elif hint is float:
                if not isinstance(value, (float, int)):
                    raise TypeError(
                        f"{fc_name}: '{pname}' expects float, "
                        f"got {type(value).__name__}"
                    )

    def caller(*args, **kwargs):
        nonlocal _arg_checked
        _ensure_loaded()
        c_func = _cache["c_func"]

        sig = inspect.signature(func)
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()
        hints = typing.get_type_hints(func)

        if not _arg_checked:
            _check_args_before_convert(bound.arguments, hints)
            _arg_checked = True

        c_args = []
        aiter_refs = []
        tensor_device = None

        for pname, value in bound.arguments.items():
            hint = hints.get(pname)
            origin = typing.get_origin(hint)
            type_args = typing.get_args(hint)

            if hint is torch.Tensor:
                if tensor_device is None:
                    tensor_device = value.device
                at = torch_to_aiter(value)
                aiter_refs.append(at)
                c_args.append(ctypes.byref(at))
            elif _is_union(origin) and torch.Tensor in type_args:
                if value is not None:
                    if tensor_device is None:
                        tensor_device = value.device
                    at = torch_to_aiter(value)
                    aiter_refs.append(at)
                    c_args.append(ctypes.byref(at))
                else:
                    c_args.append(ctypes.POINTER(aiter_tensor_t)())
            elif _is_union(origin) and int in type_args:
                c_args.append(value if value is not None else -1)
            elif _is_union(origin) and str in type_args:
                c_args.append(value.encode() if value is not None else None)
            elif hint is str:
                c_args.append(value.encode())
            elif hint is bool:
                c_args.append(1 if value else 0)
            elif hint is int:
                c_args.append(ctypes.c_int(value))
            elif hint is float:
                c_args.append(ctypes.c_float(value))
            else:
                c_args.append(value)

        if _cache.get("has_tensor"):
            c_args.append(
                ctypes.c_void_p(torch.cuda.current_stream(tensor_device).cuda_stream)
            )
        return c_func(*c_args)

    return caller


def compile_ops(
    _md_name: str,
    fc_name: Optional[str] = None,
    gen_func: Optional[Callable[..., dict[str, Any]]] = None,
    gen_fake: Optional[Callable[..., Any]] = None,
    ffi_type: str = "pybind",
):
    def decorator(func):
        loadName = fc_name if fc_name is not None else func.__name__

        if ffi_type == "ctypes":
            ctypes_caller = _ctypes_call(func, loadName, _md_name)

            @functools.wraps(func)
            def ctypes_wrapper(*args, **kwargs):
                return ctypes_caller(*args, **kwargs)

            @torch_compile_guard(device="cuda", calling_func_=func)
            def ctypes_custom_wrapper(*args, **kwargs):
                return ctypes_wrapper(*args, **kwargs)

            return ctypes_custom_wrapper
#        func.arg_checked = False
#
#        @functools.wraps(func)
#        def wrapper(*args, custom_build_args={}, **kwargs):
#            loadName = fc_name
#            md_name = _md_name
#            if fc_name is None:
#                loadName = func.__name__
#            try:
#                module = None
#                if gen_func is not None:
#                    custom_build_args.update(gen_func(*args, **kwargs))
#                if PREBUILD_KERNELS:
#                    if hasattr(aiter_, loadName):
#                        module = aiter_
#                elif AITER_REBUILD and md_name not in rebuilded_list:
#                    rebuilded_list.append(md_name)
#                    raise ModuleNotFoundError("")
#                if module is None:
#                    md = custom_build_args.get("md_name", md_name)
#                    module = get_module(md)
#            except ModuleNotFoundError:
#                d_args = get_args_of_build(md_name)
#                d_args.update(custom_build_args)

        elif ffi_type == "pybind":
            func.arg_checked = False

            @functools.wraps(func)
            def wrapper(*args, custom_build_args={}, **kwargs):

                md_name = _md_name
                try:
                    module = None
                    if gen_func is not None:
                        custom_build_args.update(gen_func(*args, **kwargs))
                    elif AITER_REBUILD and md_name not in rebuilded_list:
                        rebuilded_list.append(md_name)
                        raise ModuleNotFoundError("start rebuild")
                    if module is None:
                        md = custom_build_args.get("md_name", md_name)
                        module = get_module(md)
                except ModuleNotFoundError:
                    d_args = get_args_of_build(md_name)
                    d_args.update(custom_build_args)

                    md_name = custom_build_args.get("md_name", md_name)

                    srcs = d_args["srcs"]
                    flags_extra_cc = d_args["flags_extra_cc"]
                    flags_extra_hip = d_args["flags_extra_hip"]
                    blob_gen_cmd = d_args["blob_gen_cmd"]
                    extra_include = d_args["extra_include"]
                    extra_ldflags = d_args["extra_ldflags"]
                    verbose = d_args["verbose"]
                    is_python_module = d_args["is_python_module"]
                    is_standalone = d_args["is_standalone"]
                    torch_exclude = d_args["torch_exclude"]
                    hipify = d_args.get("hipify", False)
                    hip_clang_path = d_args.get("hip_clang_path", None)
                    third_party = d_args.get("third_party", [])
                    prev_hip_clang_path = None
                    if hip_clang_path is not None and os.path.exists(hip_clang_path):
                        prev_hip_clang_path = os.environ.get("HIP_CLANG_PATH", None)
                        os.environ["HIP_CLANG_PATH"] = hip_clang_path

                    build_module(
                        md_name,
                        srcs,
                        flags_extra_cc,
                        flags_extra_hip,
                        blob_gen_cmd,
                        extra_include,
                        extra_ldflags,
                        verbose,
                        is_python_module,
                        is_standalone,
                        torch_exclude,
                        third_party,
                        hipify,
                    )

                    if hip_clang_path is not None:
                        if prev_hip_clang_path is not None:
                            os.environ["HIP_CLANG_PATH"] = prev_hip_clang_path
                        else:
                            os.environ.pop("HIP_CLANG_PATH", None)

                    if is_python_module:
                        module = get_module(md_name)
                    if md_name not in __mds:
                        __mds[md_name] = module

                if isinstance(module, types.ModuleType):
                    op = getattr(module, loadName)
                else:
                    return None

<<<<<<< HEAD
                def check_args():
                    get_asm_dir()
                    import inspect
                    import re

                    import torch

                    enum_types = ["ActivationType", "QuantType"]

                    if not op.__doc__.startswith("Members:"):
                        doc_str = op.__doc__.split("\n")[0]
                        doc_str = re.sub(r"<(.*?)\:.*?>", r"\g<1>", doc_str)
                        doc_str = doc_str.replace("list[", "List[")
                        doc_str = doc_str.replace("tuple[", "Tuple[")
                        doc_str = doc_str.replace("collections.abc.Sequence[", "List[")
                        doc_str = doc_str.replace("typing.SupportsInt", "int")
                        doc_str = doc_str.replace("typing.SupportsFloat", "float")
                        pattern = r"([\w\.]+(?:\[[^\]]+\])?)\s*\|\s*None"
                        doc_str = re.sub(pattern, r"Optional[\1]", doc_str)
                        for el in enum_types:
                            doc_str = re.sub(
                                f" (module_)?aiter.*{el} ", f" {el} ", doc_str
                            )
                        namespace = {
                            "List": List,
                            "Optional": Optional,
                            "torch": torch,
                            "typing": typing,
                        }

                        exec(
                            f"from aiter import*\ndef {doc_str}: pass",
                            namespace,
                        )
                        foo = namespace[doc_str.split("(")[0]]
                        sig = inspect.signature(foo)
                        func.__signature__ = sig
                        ann = {k: v.annotation for k, v in sig.parameters.items()}
                        ann["return"] = sig.return_annotation
                        callargs = inspect.getcallargs(func, *args, **kwargs)
                        for el, arg in callargs.items():
                            expected_type = ann[el]
                            got_type = type(arg)
                            origin = typing.get_origin(expected_type)
                            sub_t = typing.get_args(expected_type)

                            if origin is None:
                                if not isinstance(arg, expected_type) and not (
                                    any(el in str(expected_type) for el in enum_types)
                                    and isinstance(arg, int)
                                ):
                                    raise TypeError(
                                        f"{loadName}: {el} needs to be {expected_type} but got {got_type}"
                                    )
                            elif origin is list:
                                if not isinstance(arg, list):
                                    raise TypeError(
                                        f"{loadName}: {el} needs to be List[{sub_t}] but got {arg}"
                                    )
                            elif origin is typing.Union or origin is types.UnionType:
                                if arg is not None and not isinstance(arg, sub_t):
                                    raise TypeError(
                                        f"{loadName}: {el} needs to be Optional[{sub_t}] but got {arg}"
                                    )
                            else:
                                raise TypeError(f"Unsupported type: {expected_type}")

                        func_hints = typing.get_type_hints(func)
                        if ann["return"] is None:
                            func_hints["return"] = None
                        if ann != func_hints:
                            logger.warning(
                                f"type hints mismatch, override to --> {doc_str}"
                            )
                    return True

                if not func.arg_checked:
                    func.arg_checked = check_args()

                if AITER_LOG_MORE == 2:
                    from ..test_common import log_args

                    log_args(func, *args, **kwargs)

                return op(*args, **kwargs)

            @torch_compile_guard(device="cuda", gen_fake=gen_fake, calling_func_=func)
            def custom_wrapper(*args, **kwargs):
                return wrapper(*args, **kwargs)

            return custom_wrapper

        else:
            raise ValueError(
                f"Unknown ffi_type: {ffi_type!r}, expected 'ctypes' or 'pybind'"
            )

    return decorator
