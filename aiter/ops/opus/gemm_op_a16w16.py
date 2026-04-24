# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
"""
Opus a16w16 Python-side bindings.

This module hosts the low-level pybind binding `opus_gemm_a16w16_tune`
(id-based kernel selector, splitK as literal KBatch). The high-level
shape-driven API `gemm_a16w16_opus` (CSV lookup + heuristic fallback +
auto-log of untuned shapes) lands in PR2.

The underlying JIT module is shared with `deepgemm_opus`:
`module_deepgemm_opus` (see aiter/jit/optCompilerConfig.json). Both
pybind entries coexist in the same .so; splitting the Python surface
here does not require a C++ / JIT rebuild.
"""

import torch
from torch import Tensor

from ...jit.core import compile_ops


def _gen_opus_gemm_a16w16_tune_fake_tensors(
    XQ: torch.Tensor,
    WQ: torch.Tensor,
    Y: torch.Tensor,
    kernelId: int = 0,
    splitK: int = 0,
) -> torch.Tensor:
    return Y


@compile_ops(
    "module_deepgemm_opus",
    fc_name="opus_gemm_a16w16_tune",
    gen_fake=_gen_opus_gemm_a16w16_tune_fake_tensors,
)
def opus_gemm_a16w16_tune(
    XQ: torch.Tensor,
    WQ: torch.Tensor,
    Y: torch.Tensor,
    kernelId: int = 0,
    splitK: int = 0,
) -> torch.Tensor: ...


__all__ = ["opus_gemm_a16w16_tune"]
