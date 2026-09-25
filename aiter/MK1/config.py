# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Supported-configuration declarations for MK1."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .errors import UnsupportedConfigurationError

SUPPORTED_GPU_ARCHITECTURES = frozenset({"gfx950"})
SUPPORTED_MODEL_FAMILY = "gpt-oss-120b"
SUPPORTED_MODEL_REVISION = "b5c939de"
SUPPORTED_BACKEND = "gpt_oss_gfx950_v1"
SUPPORTED_CACHE_SCALARS = frozenset({"bfloat16", "float16"})


@dataclass(frozen=True, slots=True)
class MK1Config:
    """Configuration checked before a native decoder is created."""

    device: int = 0
    max_sequence_length: int = 32_768
    cache_scalar: str = "bfloat16"
    batch_size: int = 1
    backend: str = SUPPORTED_BACKEND
    model_family: str = SUPPORTED_MODEL_FAMILY
    model_revision: str = SUPPORTED_MODEL_REVISION
    gpu_architecture: str | None = None
    mode: Literal["auto", "required"] = "auto"


@dataclass(frozen=True, slots=True)
class SupportResult:
    """Machine-readable support decision used by serving runtimes."""

    supported: bool
    reasons: tuple[str, ...] = ()

    @property
    def reason(self) -> str | None:
        return "; ".join(self.reasons) if self.reasons else None

    def require(self) -> None:
        if not self.supported:
            raise UnsupportedConfigurationError(
                self.reason or "unsupported MK1 configuration"
            )


def detect_gpu_architecture(device: int = 0) -> str | None:
    """Return the current HIP architecture without making import-time probes."""

    try:
        import torch

        if not torch.cuda.is_available():
            return None
        value = str(torch.cuda.get_device_properties(device).gcnArchName)
    except (AttributeError, RuntimeError, ValueError):
        return None
    return value.split(":", 1)[0]


def is_supported(config: MK1Config, *, probe_hardware: bool = False) -> SupportResult:
    """Validate the deliberately narrow first-release support envelope."""

    reasons: list[str] = []
    architecture = config.gpu_architecture
    if architecture is None and probe_hardware:
        architecture = detect_gpu_architecture(config.device)
    if architecture is not None and architecture not in SUPPORTED_GPU_ARCHITECTURES:
        reasons.append(f"unsupported GPU architecture: {architecture}")
    if probe_hardware and architecture is None:
        reasons.append("no accessible HIP GPU was detected")
    if config.model_family != SUPPORTED_MODEL_FAMILY:
        reasons.append(f"unsupported model family: {config.model_family}")
    if config.model_revision != SUPPORTED_MODEL_REVISION:
        reasons.append(f"unsupported model revision: {config.model_revision}")
    if config.backend != SUPPORTED_BACKEND:
        reasons.append(f"unsupported backend: {config.backend}")
    if config.cache_scalar not in SUPPORTED_CACHE_SCALARS:
        reasons.append(f"unsupported cache scalar: {config.cache_scalar}")
    if config.batch_size != 1:
        reasons.append("the first MK1 integration requires batch_size=1")
    if config.device < 0:
        reasons.append("device must be non-negative")
    if config.max_sequence_length <= 0:
        reasons.append("max_sequence_length must be positive")
    if config.mode not in ("auto", "required"):
        reasons.append(f"unsupported mode: {config.mode}")
    return SupportResult(not reasons, tuple(reasons))
