# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Test-only access to the C++ selectors; no AITER op or JIT module is registered."""

import ctypes
import functools
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def load_dispatch_lib(path):
    lib = ctypes.CDLL(str(path))
    lib.f4gemm_test_config.argtypes = [
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_char_p,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.POINTER(ctypes.c_int),
    ]
    lib.f4gemm_test_config.restype = ctypes.c_int
    lib.aiter_get_last_error.argtypes = []
    lib.aiter_get_last_error.restype = ctypes.c_char_p
    lib.f4gemm_test_geometry.argtypes = [
        ctypes.c_int,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_int),
    ]
    lib.f4gemm_test_geometry.restype = None
    lib.f4gemm_test_valid_geometry.argtypes = [ctypes.c_int] * 3
    lib.f4gemm_test_valid_geometry.restype = ctypes.c_bool
    return lib


@functools.lru_cache(None)
def get_dispatch_lib():
    from aiter.jit.core import executable_path, get_gfx_list

    hipcc = Path(executable_path("hipcc"))
    cxx = shutil.which("c++")
    if not hipcc.is_file() or cxx is None:
        raise FileNotFoundError(
            "hipcc and a C++ compiler are required for dispatch tests"
        )
    root = Path(__file__).resolve().parents[1]
    temp = tempfile.TemporaryDirectory(prefix="f4gemm_dispatch_test_")
    output = Path(temp.name)
    archs = get_gfx_list()
    commands = [
        [
            sys.executable,
            str(root / "hsa/codegen.py"),
            "-m",
            "f4gemm",
            "-o",
            str(output),
        ],
        [
            str(hipcc),
            "-x",
            "hip",
            f"--offload-arch={archs[0]}",
            "-std=c++20",
            "-fPIC",
            "-O2",
            "-DUSE_ROCM=1",
            "-DENABLE_CK=0",
            "-Wno-unused-result",
            f"-I{root / 'csrc/include'}",
            f"-I{output}",
            "-c",
            str(root / "op_tests/cpp/f4gemm_dispatch_test.cu"),
            "-o",
            str(output / "dispatch.o"),
        ],
        [
            cxx,
            "-shared",
            str(output / "dispatch.o"),
            f"-L{hipcc.resolve().parent.parent / 'lib'}",
            "-lamdhip64",
            "-o",
            str(output / "dispatch.so"),
        ],
    ]
    env = dict(os.environ, AITER_GPU_ARCHS=";".join(archs))
    for command in commands:
        result = subprocess.run(
            command, env=env, text=True, capture_output=True, check=False
        )
        if result.returncode:
            raise RuntimeError(
                "F4GEMM test-helper build failed:\n" + result.stdout + result.stderr
            )
    lib = load_dispatch_lib(output / "dispatch.so")
    lib._tempdir = temp  # Keep the library available to subprocess tests.
    return lib


def query_launch_config(
    lib, M, N, K, intype="mxfp4", outtype="bf16", apre=0, kernel=None, arch="gfx1250"
):
    values = (ctypes.c_int * 7)()
    status = lib.f4gemm_test_config(
        M,
        N,
        K,
        arch.encode(),
        intype.encode(),
        outtype.encode(),
        apre,
        kernel.encode() if kernel else None,
        values,
    )
    if status != 0:
        raise RuntimeError(lib.aiter_get_last_error().decode())
    return tuple(values)


def get_launch_config(M, N, K, intype, outtype, apre, kernel=None):
    from aiter.jit.utils.chip_info import get_gfx_runtime

    values = query_launch_config(
        get_dispatch_lib(),
        M,
        N,
        K,
        intype,
        outtype,
        apre,
        kernel,
        arch=get_gfx_runtime(),
    )
    return dict(
        zip(
            (
                "cluster_x",
                "cluster_y",
                "grid_x",
                "grid_y",
                "tile_m",
                "tile_n",
                "ps_mode",
            ),
            values,
        )
    )
