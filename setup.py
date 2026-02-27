# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import os
import shutil
import sys

from setuptools import Distribution, setup
from setuptools.command.build_ext import build_ext

this_dir = os.path.dirname(os.path.abspath(__file__))
PACKAGE_NAME = "amd-aiter"
BUILD_TARGET = os.environ.get("BUILD_TARGET", "auto")
PREBUILD_KERNELS = int(os.environ.get("PREBUILD_KERNELS", 0))


def getMaxJobs():
    # calculate the maximum allowed NUM_JOBS based on cores
    max_num_jobs_cores = max(1, os.cpu_count() * 0.8)

    try:
        import psutil

        # calculate the maximum allowed NUM_JOBS based on free memory
        free_memory_gb = psutil.virtual_memory().available / (1024**3)
        max_num_jobs_memory = int(free_memory_gb / 0.5)  # assuming 0.5 GB per job
    except ImportError:
        # psutil may not be available during metadata extraction
        max_num_jobs_memory = max_num_jobs_cores

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
    return False


def write_install_mode():
    """Write install_mode so core.py uses aiter_meta/ (install) vs repo root (develop).

    Called here so the file exists when setuptools resolves package_data,
    and again in build_ext.run() to ensure it's written for develop mode too.
    """
    mode = "develop" if is_develop_mode() else "install"
    with open("./aiter/install_mode", "w") as f:
        f.write(mode)


def prepare_packaging():
    """Copy source directories and create package metadata for non-editable installs."""
    if os.path.exists("aiter_meta") and os.path.isdir("aiter_meta"):
        shutil.rmtree("aiter_meta")
    shutil.copytree("3rdparty", "aiter_meta/3rdparty")
    shutil.copytree("hsa", "aiter_meta/hsa")
    shutil.copytree("gradlib", "aiter_meta/gradlib")
    shutil.copytree("csrc", "aiter_meta/csrc")
    open("aiter_meta/__init__.py", "w").close()
    write_install_mode()


prepare_packaging()


class NinjaBuildExtension(build_ext):
    """Custom build_ext that defers expensive operations until run() is called."""

    def run(self):
        import json
        from concurrent.futures import ThreadPoolExecutor

        # Defer heavy imports until build time
        sys.path.insert(0, f"{this_dir}/aiter/")
        from jit import core
        from jit.utils.cpp_extension import IS_HIP_EXTENSION

        # Determine build target
        if BUILD_TARGET == "auto":
            IS_ROCM = IS_HIP_EXTENSION
        elif BUILD_TARGET == "rocm":
            IS_ROCM = True
        else:
            IS_ROCM = False

        if not IS_ROCM:
            raise NotImplementedError("Only ROCM is supported")

        ck_dir = os.environ.get("CK_DIR", f"{this_dir}/3rdparty/composable_kernel")
        assert os.path.exists(ck_dir), (
            "CK is needed by aiter, please make sure clone by "
            '"git clone --recursive https://github.com/ROCm/aiter.git" or '
            '"git submodule sync ; git submodule update --init --recursive"'
        )

        write_install_mode()

        def _load_modules_from_config():
            cfg_path = os.path.join(this_dir, "aiter", "jit", "optCompilerConfig.json")
            try:
                with open(cfg_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception:
                return []
            if isinstance(data, dict):
                return list(data.keys())
            return []

        def get_exclude_ops():
            all_modules = _load_modules_from_config()
            exclude_ops = []

            for module in all_modules:
                if PREBUILD_KERNELS == 1:
                    if (
                        "_tune" in module
                        or module == "module_gemm_mi350_a8w8_blockscale_asm"
                    ):
                        exclude_ops.append(module)
                    if "mha" in module and module not in [
                        "module_fmha_v3_fwd",
                        "module_fmha_v3_varlen_fwd",
                    ]:
                        exclude_ops.append(module)
                elif PREBUILD_KERNELS == 2:
                    # Exclude _bwd, _tune, and specific module
                    if (
                        "_bwd" in module
                        or "_tune" in module
                        or module == "module_gemm_mi350_a8w8_blockscale_asm"
                    ):
                        exclude_ops.append(module)
                elif PREBUILD_KERNELS == 3:
                    # Keep only module_fmha_v3* and module_aiter_enum
                    if not (
                        module.startswith("module_fmha_v3")
                        or module == "module_aiter_enum"
                        or module == "module_gemm_mi350_a8w8_blockscale_asm"
                    ):
                        exclude_ops.append(module)
                else:
                    # Default behavior: exclude tunes and specific mi350 module
                    if (
                        "_tune" in module
                        or module == "module_gemm_mi350_a8w8_blockscale_asm"
                    ):
                        exclude_ops.append(module)

            return exclude_ops

        if PREBUILD_KERNELS != 0:
            has_torch = True
            try:
                import torch as _
            except Exception:
                has_torch = False

            if not has_torch:
                print(
                    "[aiter] PREBUILD_KERNELS set but torch not installed, "
                    "skip precompilation in this environment"
                )
            else:
                from jit.utils.mha_recipes import (
                    get_mha_varlen_prebuild_variants_by_names,
                )
                import glob

                exclude_ops = get_exclude_ops()
                all_opts_args_build, _ = core.get_args_of_build(
                    "all", exclude=exclude_ops
                )

                if PREBUILD_KERNELS == 1:
                    extra_args_build = []

                    req_md_names = [
                        "mha_varlen_fwd_bf16_nlogits_nbias_mask_nlse_ndropout_nskip_nqscale",
                        "mha_varlen_fwd_bf16_nlogits_nbias_nmask_lse_ndropout_nskip_nqscale",
                    ]
                    variants = get_mha_varlen_prebuild_variants_by_names(
                        req_md_names, ck_dir
                    )
                    base_args = core.get_args_of_build("module_mha_varlen_fwd")
                    for v in variants:
                        if not isinstance(base_args, dict) or not base_args.get("srcs"):
                            continue
                        extra_args_build.append(
                            {
                                "md_name": v["md_name"],
                                "srcs": base_args["srcs"],
                                "flags_extra_cc": base_args["flags_extra_cc"],
                                "flags_extra_hip": base_args["flags_extra_hip"],
                                "extra_include": base_args["extra_include"],
                                "blob_gen_cmd": v["blob_gen_cmd"],
                            }
                        )
                    all_opts_args_build.extend(extra_args_build)

                bd = f"{core.get_user_jit_dir()}/build"

                shutil.rmtree(bd, ignore_errors=True)
                for f in glob.glob(f"{core.get_user_jit_dir()}/*.so"):
                    try:
                        os.remove(f)
                    except Exception:
                        pass

                def build_one_module(one_opt_args):
                    flags_cc = list(one_opt_args["flags_extra_cc"]) + [
                        f"-DPREBUILD_KERNELS={PREBUILD_KERNELS}"
                    ]
                    flags_hip = list(one_opt_args["flags_extra_hip"]) + [
                        f"-DPREBUILD_KERNELS={PREBUILD_KERNELS}"
                    ]

                    core.build_module(
                        md_name=one_opt_args["md_name"],
                        srcs=one_opt_args["srcs"],
                        flags_extra_cc=flags_cc,
                        flags_extra_hip=flags_hip,
                        blob_gen_cmd=one_opt_args["blob_gen_cmd"],
                        extra_include=one_opt_args["extra_include"],
                        extra_ldflags=None,
                        verbose=False,
                        is_python_module=True,
                        is_standalone=False,
                        torch_exclude=False,
                    )

                prebuid_thread_num = 5
                max_jobs = os.environ.get("MAX_JOBS")
                if max_jobs is not None and max_jobs.isdigit() and int(max_jobs) > 0:
                    prebuid_thread_num = min(prebuid_thread_num, int(max_jobs))
                else:
                    prebuid_thread_num = min(prebuid_thread_num, getMaxJobs())
                os.environ["PREBUILD_THREAD_NUM"] = str(prebuid_thread_num)

                with ThreadPoolExecutor(max_workers=prebuid_thread_num) as executor:
                    list(executor.map(build_one_module, all_opts_args_build))

        # Set MAX_JOBS for ninja
        max_jobs_env = os.environ.get("MAX_JOBS")
        if max_jobs_env is None:
            max_jobs = getMaxJobs()
            os.environ["MAX_JOBS"] = str(max_jobs)
        else:
            try:
                if int(max_jobs_env) <= 0:
                    raise ValueError("MAX_JOBS must be a positive integer")
            except ValueError:
                max_jobs = getMaxJobs()
                os.environ["MAX_JOBS"] = str(max_jobs)

        # Run the actual build
        super().run()


setup_requires = [
    "packaging",
    "psutil",
    "ninja",
    "setuptools_scm",
]
if PREBUILD_KERNELS != 0:
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
    cmdclass={"build_ext": NinjaBuildExtension},
    python_requires=">=3.8",
    install_requires=[
        "pybind11>=3.0.1",
        "ninja",
        "pandas",
        "einops",
        "psutil",
        "packaging",
    ],
    extras_require={
        # Triton-based communication using Iris
        # Note: Iris is not available on PyPI and must be installed separately
        # Install with: pip install -r requirements-triton-comms.txt
        # (See requirements-triton-comms.txt for pinned Iris version)
        "triton_comms": [],
        # Install all optional dependencies
        "all": [],
    },
    setup_requires=setup_requires,
    distclass=ForcePlatlibDistribution,
)

if os.path.exists("aiter_meta") and os.path.isdir("aiter_meta"):
    shutil.rmtree("aiter_meta")
