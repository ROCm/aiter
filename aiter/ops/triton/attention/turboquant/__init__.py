# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""
TurboQuant — KV-cache compression for transformer attention.

Phase 1 exports (core quantization infrastructure, no Triton kernels):

  Quantizers:
    TurboQuantMSE    — MSE-optimal key compression (Algorithm 1)
    TurboQuantProd   — Unbiased inner-product key compression (Algorithm 2)
    ValueQuantizer   — Group-wise value quantization

  Compressed data containers:
    CompressedKeys
    CompressedKeysProd
    CompressedValues

  Codebook utilities:
    get_codebook
    pregenerate_all_codebooks

  Rotation/projection utilities:
    get_rotation_matrix
    get_qjl_matrix

  Bit-packing utilities:
    pack_indices
    unpack_indices
    packed_size
    compression_ratio
"""

from .codebook import get_codebook, pregenerate_all_codebooks
from .rotation import get_rotation_matrix, get_qjl_matrix, clear_cache as clear_rotation_cache
from .quantizer import (
    TurboQuantMSE,
    TurboQuantProd,
    ValueQuantizer,
    CompressedKeys,
    CompressedKeysProd,
    CompressedValues,
)
from .utils import pack_indices, unpack_indices, packed_size, compression_ratio

__all__ = [
    # Quantizers
    "TurboQuantMSE",
    "TurboQuantProd",
    "ValueQuantizer",
    # Data containers
    "CompressedKeys",
    "CompressedKeysProd",
    "CompressedValues",
    # Codebook
    "get_codebook",
    "pregenerate_all_codebooks",
    # Rotation
    "get_rotation_matrix",
    "get_qjl_matrix",
    "clear_rotation_cache",
    # Bit-packing
    "pack_indices",
    "unpack_indices",
    "packed_size",
    "compression_ratio",
]
