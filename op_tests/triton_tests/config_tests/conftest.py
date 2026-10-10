# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Test environment configuration scoped strictly to config_tests.

Provides CPU-fallback math shims ONLY when Triton is not installed,
and guards aiter import ONLY when ROCm is not present on the host.
Scoped to config_tests so unrelated op_tests suites are completely unaffected.
"""

import os
import sys
from unittest.mock import MagicMock

_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

# Minimal math fallback ONLY for CPU host environments where Triton is not installed
if "triton" not in sys.modules:
    try:
        import triton  # noqa: F401
    except (ImportError, ModuleNotFoundError):
        _mock_triton = MagicMock()
        _mock_triton.cdiv = lambda a, b: (a + b - 1) // b
        _mock_triton.next_power_of_2 = lambda n: 1 if n <= 1 else 2 ** (n - 1).bit_length()
        sys.modules.setdefault("triton", _mock_triton)
        sys.modules.setdefault("triton.language", MagicMock())
        sys.modules.setdefault("triton.runtime", MagicMock())

_t = sys.modules.get("triton", None)
if _t is not None:
    if not hasattr(_t, "cdiv") or isinstance(_t.cdiv, MagicMock):
        _t.cdiv = lambda a, b: (a + b - 1) // b
    if not hasattr(_t, "next_power_of_2") or isinstance(_t.next_power_of_2, MagicMock):
        _t.next_power_of_2 = lambda n: 1 if n <= 1 else 2 ** (n - 1).bit_length()

# In host environments without ROCm (e.g. macOS / CPU-only dev boxes),
# skip CK/HIP JIT compilation to allow importing aiter config utils.
_rocm_home = os.getenv("ROCM_PATH", "/opt/rocm")
if not os.path.exists(_rocm_home) and not os.path.exists(f"{_rocm_home}/.info/version"):
    os.environ.setdefault("AITER_TRITON_ONLY", "1")
