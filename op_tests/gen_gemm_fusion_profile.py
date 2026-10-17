# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Regenerate the sp_qkv profile JSON without importing aiter/__init__ (no torch needed): python op_tests/gen_gemm_fusion_profile.py"""
import os
import sys
import types

root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "aiter"))
if "aiter" not in sys.modules:
    try:
        import aiter  # noqa: F401
    except Exception:
        for k in [k for k in sys.modules if k == "aiter" or k.startswith("aiter.")]:
            del sys.modules[k]
        pkg = types.ModuleType("aiter")
        pkg.__path__ = [root]
        sys.modules["aiter"] = pkg
from aiter.ops.gemm_fusion.sp_qkv import gen_profile  # noqa: E402

gen_profile.main(sys.argv[1:])
