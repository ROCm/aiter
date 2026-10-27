# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import os
import subprocess
import sys
import textwrap


def _test_env():
    env = os.environ.copy()
    env["AITER_TRITON_ONLY"] = "1"
    utils_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "aiter", "jit", "utils")
    )
    env["PYTHONPATH"] = os.pathsep.join(
        path for path in (utils_path, env.get("PYTHONPATH")) if path
    )
    return env


def test_str2dtype_base_types_do_not_import_native_enum():
    script = textwrap.dedent("""
        import argparse
        import builtins
        import torch

        original_import = builtins.__import__
        enum_imports = []

        def reject_native_enum(name, globals=None, locals=None, fromlist=(), level=0):
            if name.endswith("ops.enum"):
                enum_imports.append(name)
                raise AssertionError(f"native enum import: {name}")
            return original_import(name, globals, locals, fromlist, level)

        builtins.__import__ = reject_native_enum
        from aiter.utility.dtypes import str2Dtype

        assert str2Dtype("fp16") is torch.float16
        assert str2Dtype("none") is None
        assert str2Dtype("fp16,bf16,") == (torch.float16, torch.bfloat16)
        assert str2Dtype("fp16,") == (torch.float16,)
        assert not enum_imports
        try:
            str2Dtype("not-a-dtype")
        except argparse.ArgumentTypeError:
            pass
        assert enum_imports
        """)
    subprocess.run([sys.executable, "-c", script], check=True, env=_test_env())


def test_get_dtype_fp8_delegates_to_shared_runtime_detector():
    script = textwrap.dedent("""
        from unittest import mock

        import torch
        from aiter.jit.utils import chip_info
        from aiter.utility import dtypes

        with (
            mock.patch.object(torch.version, "hip", "6.3"),
            mock.patch.object(torch.cuda, "is_available", return_value=True),
            mock.patch.object(chip_info, "get_gfx_runtime", return_value="gfx942") as get_gfx,
        ):
            assert dtypes.get_dtype_fp8() is torch.float8_e4m3fnuz
        get_gfx.assert_called_once_with()
    """)
    subprocess.run([sys.executable, "-c", script], check=True, env=_test_env())


def test_get_dtype_fp8_skips_runtime_detection_without_hip_device():
    script = textwrap.dedent("""
        import sys
        from unittest import mock

        import torch
        from aiter.utility import dtypes

        for hip, available in ((None, True), ("6.3", False)):
            with (
                mock.patch.dict(sys.modules, {"aiter.jit.utils.chip_info": None}),
                mock.patch.object(torch.version, "hip", hip),
                mock.patch.object(torch.cuda, "is_available", return_value=available),
            ):
                assert dtypes.get_dtype_fp8() is torch.uint8
    """)
    subprocess.run([sys.executable, "-c", script], check=True, env=_test_env())


if __name__ == "__main__":
    test_str2dtype_base_types_do_not_import_native_enum()
    test_get_dtype_fp8_delegates_to_shared_runtime_detector()
    test_get_dtype_fp8_skips_runtime_detection_without_hip_device()
