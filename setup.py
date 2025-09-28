# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import os
import shutil
import sys

from setuptools import Distribution, setup

# !!!!!!!!!!!!!!!! never import aiter
# from aiter.jit import core
this_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, f"{this_dir}/aiter/")
from concurrent.futures import ThreadPoolExecutor

from jit import core
from jit.utils.cpp_extension import IS_HIP_EXTENSION, BuildExtension

ck_dir = os.environ.get("CK_DIR", f"{this_dir}/3rdparty/composable_kernel")
PACKAGE_NAME = "aiter"
BUILD_TARGET = os.environ.get("BUILD_TARGET", "auto")

if BUILD_TARGET == "auto":
    if IS_HIP_EXTENSION:
        IS_ROCM = True
    else:
        IS_ROCM = False
else:
    if BUILD_TARGET == "cuda":
        IS_ROCM = False
    elif BUILD_TARGET == "rocm":
        IS_ROCM = True

FORCE_CXX11_ABI = False

PREBUILD_KERNELS = int(os.environ.get("PREBUILD_KERNELS", 0))


def getMaxJobs():
    # calculate the maximum allowed NUM_JOBS based on cores
    max_num_jobs_cores = max(1, os.cpu_count() * 0.8)
    import psutil

    # calculate the maximum allowed NUM_JOBS based on free memory
    free_memory_gb = psutil.virtual_memory().available / (1024**3)  # free memory in GB
    max_num_jobs_memory = int(free_memory_gb / 0.5)  # assuming 0.5 GB per job

    # pick lower value of jobs based on cores vs memory metric to minimize oom and swap usage during compilation
    max_jobs = int(max(1, min(max_num_jobs_cores, max_num_jobs_memory)))
    return max_jobs


def is_develop_mode():
    for arg in sys.argv:
        if arg == "develop":
            return True
        # pip install -e
        elif "editable" in arg:
            return True
    else:
        return False


if is_develop_mode():
    with open("./aiter/install_mode", "w") as f:
        f.write("develop")
else:
    with open("./aiter/install_mode", "w") as f:
        f.write("install")

if IS_ROCM:
    assert os.path.exists(
        ck_dir
    ), 'CK is needed by aiter, please make sure clone by "git clone --recursive https://github.com/ROCm/aiter.git" or "git submodule sync ; git submodule update --init --recursive"'

    if PREBUILD_KERNELS == 1:
        exclude_ops = [
            "libmha_fwd",
            "libmha_bwd",
            "module_fmha_v3_fwd",
            "module_mha_fwd",
            "module_mha_varlen_fwd",
            "module_mha_batch_prefill",
            "module_fmha_v3_bwd",
            "module_fmha_v3_varlen_bwd",
            "module_fmha_v3_varlen_fwd",
            "module_mha_bwd",
            "module_mha_varlen_bwd",
        ]

        all_opts_args_build, prebuild_link_param = core.get_args_of_build(
            "all", exclude=exclude_ops
        )
        os.system(f"rm -rf {core.get_user_jit_dir()}/build")
        os.system(f"rm -rf {core.get_user_jit_dir()}/*.so")
        prebuild_dir = f"{core.get_user_jit_dir()}/build/aiter_/build"
        core.recopy_ck()
        os.makedirs(prebuild_dir + "/srcs")

        def build_one_module(one_opt_args):
            core.build_module(
                md_name=one_opt_args["md_name"],
                srcs=one_opt_args["srcs"],
                flags_extra_cc=one_opt_args["flags_extra_cc"] + ["-DPREBUILD_KERNELS"],
                flags_extra_hip=one_opt_args["flags_extra_hip"]
                + ["-DPREBUILD_KERNELS"],
                blob_gen_cmd=one_opt_args["blob_gen_cmd"],
                extra_include=one_opt_args["extra_include"],
                extra_ldflags=None,
                verbose=False,
                is_python_module=True,
                is_standalone=False,
                torch_exclude=False,
                prebuild=1,
            )

        # step 1, build *.cu -> module*.so
        prebuid_thread_num = 5
        # Respect MAX_JOBS environment variable, fallback to auto-calculation
        max_jobs = os.environ.get("MAX_JOBS")
        if max_jobs is not None and max_jobs.isdigit() and int(max_jobs) > 0:
            prebuid_thread_num = min(prebuid_thread_num, int(max_jobs))
        else:
            prebuid_thread_num = min(prebuid_thread_num, getMaxJobs())
        os.environ["PREBUILD_THREAD_NUM"] = str(prebuid_thread_num)

        with ThreadPoolExecutor(max_workers=prebuid_thread_num) as executor:
            list(executor.map(build_one_module, all_opts_args_build))

        ck_batched_gemm_folders = [
            f"{this_dir}/csrc/{name}/include"
            for name in os.listdir(f"{this_dir}/csrc")
            if os.path.isdir(os.path.join(f"{this_dir}/csrc", name))
            and name.startswith("ck_batched_gemm")
        ]
        ck_gemm_folders = [
            f"{this_dir}/csrc/{name}/include"
            for name in os.listdir(f"{this_dir}/csrc")
            if os.path.isdir(os.path.join(f"{this_dir}/csrc", name))
            and name.startswith("ck_gemm_a")
        ]
        ck_gemm_inc = ck_batched_gemm_folders + ck_gemm_folders
        for src in ck_gemm_inc:
            dst = f"{prebuild_dir}/include"
            shutil.copytree(src, dst, dirs_exist_ok=True)

        shutil.copytree(
            f"{this_dir}/csrc/include", f"{prebuild_dir}/include", dirs_exist_ok=True
        )

        # step 2, link module*.so -> aiter_.so
        core.build_module(
            md_name="aiter_",
            srcs=[f"{prebuild_dir}/srcs/rocm_ops.cu"],
            flags_extra_cc=prebuild_link_param["flags_extra_cc"]
            + ["-DPREBUILD_KERNELS"],
            flags_extra_hip=prebuild_link_param["flags_extra_hip"]
            + ["-DPREBUILD_KERNELS"],
            blob_gen_cmd=prebuild_link_param["blob_gen_cmd"],
            extra_include=prebuild_link_param["extra_include"],
            extra_ldflags=None,
            verbose=False,
            is_python_module=True,
            is_standalone=False,
            torch_exclude=False,
            prebuild=2,
        )
else:
    raise NotImplementedError("Only ROCM is supported")


if os.path.exists("aiter_meta") and os.path.isdir("aiter_meta"):
    shutil.rmtree("aiter_meta")
## link "3rdparty", "hsa", "csrc" into "aiter_meta"
shutil.copytree("3rdparty", "aiter_meta/3rdparty")
shutil.copytree("hsa", "aiter_meta/hsa")
shutil.copytree("csrc", "aiter_meta/csrc")


class NinjaBuildExtension(BuildExtension):
    def __init__(self, *args, **kwargs) -> None:
        # Respect MAX_JOBS environment variable, fallback to auto-calculation
        max_jobs_env = os.environ.get("MAX_JOBS")
        if max_jobs_env is None:
            # Only calculate max_jobs if MAX_JOBS is not set
            max_jobs = getMaxJobs()
            os.environ["MAX_JOBS"] = str(max_jobs)
        else:
            # Validate the provided MAX_JOBS value
            try:
                int(max_jobs_env)
                if int(max_jobs_env) <= 0:
                    raise ValueError("MAX_JOBS must be a positive integer")
            except ValueError:
                # If invalid, fallback to auto-calculation
                max_jobs = getMaxJobs()
                os.environ["MAX_JOBS"] = str(max_jobs)

        super().__init__(*args, **kwargs)


setup_requires = [
    "packaging",
    "psutil",
    "ninja",
    "setuptools_scm",
]
if PREBUILD_KERNELS == 1:
    setup_requires.append("pandas")


class ForcePlatlibDistribution(Distribution):
    def has_ext_modules(self):
        return True


setup(
    name=PACKAGE_NAME,
    use_scm_version=True,
    packages=["aiter_meta", "aiter"],
    include_package_data=True,
    package_data={
        "": ["*"],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: Unix",
    ],
    # ext_modules=ext_modules,
    cmdclass={"build_ext": NinjaBuildExtension},
    python_requires=">=3.8",
    install_requires=[
        "pybind11>=3.0.1",
        # "ninja",
        "pandas",
        "einops",
        "psutil",
    ],
    setup_requires=setup_requires,
    distclass=ForcePlatlibDistribution,
)

if os.path.exists("aiter_meta") and os.path.isdir("aiter_meta"):
    shutil.rmtree("aiter_meta")
