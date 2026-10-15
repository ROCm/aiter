# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import os
import subprocess
import sys
import textwrap


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
    env = os.environ.copy()
    env["AITER_TRITON_ONLY"] = "1"
    subprocess.run([sys.executable, "-c", script], check=True, env=env)


if __name__ == "__main__":
    test_str2dtype_base_types_do_not_import_native_enum()
