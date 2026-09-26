# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2025-2026 FlyDSL Project Contributors

"""Object-oriented building blocks for the FP8 paged-attention kernel."""

from functools import cache
from hashlib import sha256
from pathlib import Path


@cache
def implementation_cache_tag():
    """Include operation classes in FlyDSL's persistent compilation cache.

    FlyDSL follows function closures, but does not traverse the classes held by
    a composed pipeline. Hash their sources once per process, including the
    shared helpers they call, so an operation edit invalidates cached binaries.
    Paths are relative to kernels/ so the key is stable across installations.
    """
    package = Path(__file__).parent
    shared_helpers = (
        "buffer_ops.py",
        "dpp_utils.py",
        "kernels_common.py",
        "tensor_shim.py",
        "utils.py",
    )
    sources = sorted(package.glob("*.py")) + [
        package.parent / name for name in shared_helpers
    ]
    digest = sha256()
    for source in sources:
        digest.update(source.relative_to(package.parent).as_posix().encode())
        digest.update(source.read_bytes())
    return digest.hexdigest()
