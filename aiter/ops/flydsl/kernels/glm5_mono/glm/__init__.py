# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Public API for the GLM-5 indexed decode MonoKernel."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from aiter.ops.flydsl.kernels.glm5_mono.glm.op import Glm5MonoKernel
    from aiter.ops.flydsl.kernels.glm5_mono.weights import LayerWeights

__all__ = ["Glm5MonoKernel", "LayerWeights"]


def __getattr__(name: str):
    """Load GPU wrappers only when callers request them."""

    if name == "Glm5MonoKernel":
        from aiter.ops.flydsl.kernels.glm5_mono.glm.op import Glm5MonoKernel

        return Glm5MonoKernel
    if name == "LayerWeights":
        from aiter.ops.flydsl.kernels.glm5_mono.weights import LayerWeights

        return LayerWeights
    raise AttributeError(name)
