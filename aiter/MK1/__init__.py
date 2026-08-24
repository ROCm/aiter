# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Optional AITER MK1 persistent-decoder integration."""

from .checkpoint import CheckpointInfo, persistent_checkpoint_bytes, validate_checkpoint
from .config import MK1Config, SupportResult, is_supported
from .errors import (
    BackendLaunchError,
    CheckpointError,
    MK1Error,
    NativeBinaryError,
    PrelaunchError,
    UnsupportedConfigurationError,
)
from .persistent_decoder import (
    AtomCacheBinding,
    PersistentDecoder,
    QuantumRequest,
    QuantumResult,
    write_atom_fp16_cache,
    write_atom_shuffled_cache,
)

__all__ = [
    "AtomCacheBinding",
    "BackendLaunchError",
    "CheckpointError",
    "CheckpointInfo",
    "MK1Config",
    "MK1Error",
    "NativeBinaryError",
    "PersistentDecoder",
    "PrelaunchError",
    "QuantumRequest",
    "QuantumResult",
    "SupportResult",
    "UnsupportedConfigurationError",
    "is_supported",
    "persistent_checkpoint_bytes",
    "validate_checkpoint",
    "write_atom_fp16_cache",
    "write_atom_shuffled_cache",
]
