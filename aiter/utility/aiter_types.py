# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
import ctypes
import re
from pathlib import Path


# support develop and install mode
def _find_aiter_enum_h() -> Path:
    root = Path(__file__).resolve().parents[2]
    candidates = [
        root / "csrc" / "include" / "aiter_enum.h",
        root / "aiter_meta" / "csrc" / "include" / "aiter_enum.h",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(f"aiter_enum.h not found in {[str(c) for c in candidates]}")


_AITER_ENUM_H = _find_aiter_enum_h()
_PREFIX = "AITER_DTYPE_"


# get aiter_dtypes(python) from aiter_enum.h
def _parse_aiter_dtypes(header: Path) -> dict:
    """Parse AiterDtype enum from aiter_enum.h, returns {short_name: int_id}."""
    assert header.exists(), f"Header not found: {header}"
    text = header.read_text()
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
    m = re.search(r"typedef\s+enum\s*\{([^}]+)\}\s*AiterDtype\s*;", text)
    assert m, f"AiterDtype enum not found in {header}"
    result = {}
    next_val = 0
    for line in m.group(1).split("\n"):
        line = re.sub(r"//.*", "", line).strip().rstrip(",")
        if not line:
            continue
        if "=" in line:
            name, val = line.split("=", 1)
            name = name.strip()
            next_val = int(val.strip())
        else:
            name = line
        result[name.removeprefix(_PREFIX)] = next_val
        next_val += 1
    return result


aiter_dtypes = _parse_aiter_dtypes(_AITER_ENUM_H)
"""
expected format of aiter_dtypes: {
                            "fp8":      0,
                            "fp8_e8m0": 1,
                            "fp16":     2,
                            "bf16":     3,
                            "fp32":     4,
                            "i4x2":     5,
                            "fp4x2":    6,
                            "u32":      7,
                            "i32":      8,
                            "i16":      9,
                            "i8":       10,
                        }
"""


class AiterTensor(ctypes.Structure):
    _fields_ = [
        ("ptr", ctypes.c_void_p),
        ("size", ctypes.c_size_t),
        ("ndim", ctypes.c_int),
        ("shape", ctypes.c_int64 * 8),
        ("strides", ctypes.c_int64 * 8),
        ("dtype", ctypes.c_int),
        ("device_id", ctypes.c_int),
    ]
