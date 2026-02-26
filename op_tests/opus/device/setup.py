# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_CSRC = os.path.normpath(
    os.path.join(_THIS_DIR, "..", "..", "..", "csrc", "include")
)

__ARCH__ = "native"

setup(
    name="opus_device_test",
    ext_modules=[
        CUDAExtension(
            name="opus_device_test",
            sources=[
                os.path.join(_THIS_DIR, "test_mfma.cu"),
                os.path.join(_THIS_DIR, "test_mxfp.cu"),
                os.path.join(_THIS_DIR, "test_load_store_if.cu"),
                os.path.join(_THIS_DIR, "test_vector_add.cu"),
                os.path.join(_THIS_DIR, "test_async_load.cu"),
                os.path.join(_THIS_DIR, "test_dtype_convert.cu"),
                os.path.join(_THIS_DIR, "test_mdiv.cu"),
                os.path.join(_THIS_DIR, "test_numeric_limits.cu"),
                os.path.join(_THIS_DIR, "test_workgroup_barrier.cu"),
                os.path.join(_THIS_DIR, "opus_device_test_ext.cpp"),
            ],
            include_dirs=[_REPO_CSRC, _THIS_DIR],
            extra_compile_args={
                # Limit offload-arch to native GPU to speed up compilation
                "nvcc": [f"--offload-arch={__ARCH__}", "-O3"],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
