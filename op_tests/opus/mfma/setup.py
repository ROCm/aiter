# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
# Repo csrc/include: mfma -> opus -> op_tests -> repo root -> csrc/include
_REPO_CSRC = os.path.normpath(
    os.path.join(_THIS_DIR, "..", "..", "..", "csrc", "include")
)

setup(
    name="opus_mfma",
    ext_modules=[
        CUDAExtension(
            name="opus_mfma",
            sources=[
                os.path.join(_THIS_DIR, "test_opus_mfma.cu"),
                os.path.join(_THIS_DIR, "opus_mfma_ext.cpp"),
            ],
            include_dirs=[_REPO_CSRC, _THIS_DIR],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
