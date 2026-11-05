# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2025-2026 FlyDSL Project Contributors

"""Object-oriented building blocks for FP8/BF16 paged-attention kernels."""

from functools import cache
from hashlib import sha256
from pathlib import Path


@cache
def implementation_cache_tag():
    """Cover composed PA sources in compilation and autotuning caches.

    FlyDSL follows function closures, but does not traverse the classes held by
    a composed pipeline. Hash their sources once per process, including the
    shared helpers, kernel, planner, reducer and host wrapper, so changes
    invalidate cached binaries and tuning results. Paths are relative to
    flydsl/ so the key is stable across installations.
    """
    package = Path(__file__).parent
    shared_helpers = (
        "buffer_ops.py",
        "dpp_utils.py",
        "kernels_common.py",
        "pa_decode_kernel.py",
        "pa_decode_plan.py",
        "pa_decode_reduce.py",
        "tensor_shim.py",
        "utils.py",
    )
    sources = (
        sorted(package.glob("*.py"))
        + [package.parent / name for name in shared_helpers]
        + [package.parent.parent / "pa_decode.py"]
    )
    digest = sha256()
    for source in sources:
        digest.update(source.relative_to(package.parent.parent).as_posix().encode())
        digest.update(source.read_bytes())
    return digest.hexdigest()
