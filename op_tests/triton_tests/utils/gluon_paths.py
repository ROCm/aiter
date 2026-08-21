# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import importlib
import zlib


def gluon_import_path(current: str, legacy: str | None = None):
    """Build a picker that alternates between a relocated gluon module's paths.

    Gluon kernels moved out of aiter/ops/triton/gluon/ into
    aiter/ops/triton/_gluon_kernels/<arch>/, with their host entry points beside
    the triton ones. The old import path still resolves through the
    _BACKWARD_COMPAT_MAP in aiter/ops/triton/__init__.py, so tests should cover
    both. The returned picker takes the test case's parameters and chooses a
    path from a checksum of them, which keeps the choice stable across runs
    while both paths stay covered across a suite.

    Args:
        current: Dotted path of the module's current location.
        legacy: Dotted path of the pre-move location. Defaults to the same
            module name under aiter.ops.triton.gluon.
    """
    if legacy is None:
        legacy = f"aiter.ops.triton.gluon.{current.rsplit('.', 1)[1]}"

    def pick(*case):
        chosen = legacy if zlib.crc32(repr(case).encode()) % 2 else current
        return importlib.import_module(chosen)

    return pick
