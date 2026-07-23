# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2025-2026 FlyDSL Project Contributors

"""Public host dispatch for FlyDSL paged-attention decode."""

from .pa_decode_tile import (
    compile_pa_decode_tile,
    get_recommended_splits,
    pa_decode,
)

__all__ = [
    "compile_pa_decode_tile",
    "get_recommended_splits",
    "pa_decode",
]
