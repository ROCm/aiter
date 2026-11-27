# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Whether an arch ships hand-written ASM kernels for an operator family.

The answer comes from the kernel metadata under ``hsa/<gfx>/<family>/``, the
same CSVs ``hsa/codegen.py`` turns into the C++ kernel tables, so no arch list
is repeated at dispatch sites. Pure Python: no torch, no GPU.
"""

import functools
import os
from pathlib import Path


def _default_asm_root() -> str:
    # core.py exports its resolved root (installed or in-tree) in the environment.
    return os.environ.get("AITER_ASM_DIR") or str(
        Path(__file__).resolve().parents[3] / "hsa"
    )


@functools.cache
def _has_asm_kernels(family: str, gfx: str, asm_root: str) -> bool:
    family_dir = os.path.join(asm_root, gfx, family)
    try:
        return any(name.endswith(".csv") for name in os.listdir(family_dir))
    except OSError:
        return False


def has_asm_kernels(family: str, gfx: str, asm_root: str | None = None) -> bool:
    """True when ``hsa/<gfx>/<family>/`` lists kernels, e.g. ``("bf16gemm", "gfx950")``.

    ``gfx`` is the exact arch: a variant such as ``gfx1250-strict`` has its own
    directory or none, never its base arch's.
    """
    return _has_asm_kernels(family, gfx, asm_root or _default_asm_root())
