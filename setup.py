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
PACKAGE_NAME = "amd-aiter"
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
    free_memory_gb = psutil.virtual_memory().available / (1024**3)
    # free memory in GB
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

    if os.path.exists("aiter_meta") and os.path.isdir("aiter_meta"):
        shutil.rmtree("aiter_meta")
    shutil.copytree("3rdparty", "aiter_meta/3rdparty")
    shutil.copytree("hsa", "aiter_meta/hsa")
    shutil.copytree("gradlib", "aiter_meta/gradlib")
    shutil.copytree("csrc", "aiter_meta/csrc")

    def get_exclude_ops():
        if PREBUILD_KERNELS == 1:
            return [
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
                "module_batched_gemm_bf16_tune",
                "module_batched_gemm_a8w8_tune",
                "module_gemm_a8w8_tune",
                "module_gemm_a8w8_blockscale_tune",
                "module_gemm_a8w8_blockscale_bpreshuffle_tune",
                "module_gemm_a4w4_blockscale_tune",
                "module_gemm_a8w8_bpreshuffle_tune",
                "module_gemm_a8w8_bpreshuffle_cktile_tune",
                "module_gemm_mi350_a8w8_blockscale_asm",
            ]
        elif PREBUILD_KERNELS == 2:
            return [
                "libmha_bwd",
                "module_mha_batch_prefill",
                "module_fmha_v3_bwd",
                "module_fmha_v3_varlen_bwd",
                "module_mha_bwd",
                "module_mha_varlen_bwd",
                "module_batched_gemm_bf16_tune",
                "module_batched_gemm_a8w8_tune",
                "module_gemm_a8w8_tune",
                "module_gemm_a8w8_blockscale_tune",
                "module_gemm_a8w8_blockscale_bpreshuffle_tune",
                "module_gemm_a4w4_blockscale_tune",
                "module_gemm_a8w8_bpreshuffle_tune",
                "module_gemm_a8w8_bpreshuffle_cktile_tune",
                "module_gemm_mi350_a8w8_blockscale_asm",
            ]
        elif PREBUILD_KERNELS == 3:
            return [
                "module_activation",
                "module_attention",
                "module_pa_ragged",
                "module_pa_v1",
                "module_attention_asm",
                "module_pa",
                "module_mla_asm",
                "module_cache",
                "module_custom_all_reduce",
                "module_quick_all_reduce",
                "module_custom",
                "module_gemm_common",
                "module_batched_gemm_bf16",
                "module_batched_gemm_a8w8",
                "module_gemm_a8w8",
                "module_gemm_a8w8_blockscale",
                "module_gemm_a8w8_blockscale_bpreshuffle",
                "module_gemm_a4w4_blockscale",
                "module_gemm_a8w8_bpreshuffle",
                "module_deepgemm",
                "module_gemm_a8w8_bpreshuffle_cktile",
                "module_gemm_a8w8_asm",
                "module_gemm_a16w16_asm",
                "module_gemm_a4w4_asm",
                "module_gemm_a8w8_blockscale_asm",
                "module_gemm_a8w8_blockscale_bpreshuffle_asm",
                "module_gemm_mi350_a8w8_blockscale_asm",
                "module_moe_asm",
                "module_moe_ck2stages",
                "module_moe_cktile2stages",
                "module_moe_sorting",
                "module_moe_topk",
                "module_norm",
                "module_pos_encoding",
                "module_rmsnorm",
                "module_smoothquant",
                "module_batched_gemm_bf16_tune",
                "module_batched_gemm_a8w8_tune",
                "module_gemm_a8w8_tune",
                "module_gemm_a8w8_blockscale_tune",
                "module_gemm_a8w8_blockscale_bpreshuffle_tune",
                "module_gemm_a4w4_blockscale_tune",
                "module_gemm_a8w8_bpreshuffle_tune",
                "module_gemm_a8w8_bpreshuffle_cktile_tune",
                "module_aiter_operator",
                "module_aiter_unary",
                "module_quant",
                "module_sample",
                "module_rope_general_fwd",
                "module_rope_general_bwd",
                "module_rope_pos_fwd",
                "module_fused_mrope_rms",
                # "module_fmha_v3_fwd",
                "module_mha_fwd",
                "module_mha_varlen_fwd",
                # "module_fmha_v3_bwd",
                "module_fmha_v3_varlen_bwd",
                "module_fmha_v3_varlen_fwd",
                "module_mha_bwd",
                "module_mha_varlen_bwd",
                "libmha_fwd",
                "libmha_bwd",
                "module_rocsolgemm",
                "module_hipbsolgemm",
                "module_top_k_per_row",
                "module_mla_metadata",
                "module_mla_reduce",
                "module_topk_plain",
            ]
        else:
            return [
                "module_gemm_mi350_a8w8_blockscale_asm",
                "module_batched_gemm_bf16_tune",
                "module_batched_gemm_a8w8_tune",
                "module_gemm_a8w8_tune",
                "module_gemm_a8w8_blockscale_tune",
                "module_gemm_a8w8_blockscale_bpreshuffle_tune",
                "module_gemm_a4w4_blockscale_tune",
                "module_gemm_a8w8_bpreshuffle_tune",
                "module_gemm_a8w8_bpreshuffle_cktile_tune",
            ]

    exclude_ops = get_exclude_ops()

    has_torch = True
    try:
        import torch as _
    except Exception:
        has_torch = False

    if PREBUILD_KERNELS != 0:
        if not has_torch:
            print(
                "[aiter] PREBUILD_KERNELS set but torch not installed, skip precompilation in this environment"
            )
        else:
            all_opts_args_build, _ = core.get_args_of_build("all", exclude=exclude_ops)

            bd = f"{core.get_user_jit_dir()}/build"
            import glob

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

else:
    raise NotImplementedError("Only ROCM is supported")


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
