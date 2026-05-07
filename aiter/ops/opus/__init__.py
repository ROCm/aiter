# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
"""
aiter.ops.opus — opus kernel Python user-facing API.

Per-dtype modules. a16w16 lives here today; a8w8 / a8w8_blockscale
arrive in follow-up PRs. Each module owns its own Python surface and
pybind bindings but shares the underlying JIT module
`module_deepgemm_opus` built from csrc/opus_gemm/.

Public API:
  * gemm_a16w16_opus       — shape-driven wrapper (CSV lookup + C++
                             heuristic fallback). Typical user entry.
  * opus_gemm_a16w16_tune  — id-based low-level binding (tuner / override).

Arch gating
-----------
The a16w16 kernels are gfx950-only today (MFMA-32x32x16, ds_read_b64_tr,
160 KiB LDS). On non-gfx950 devices this package still imports cleanly
so that `import aiter` (and the 30+ other ops imported alongside it in
`aiter/__init__.py`) keeps working. The two public callables are
replaced with stubs that raise a clear ``RuntimeError`` only when the
caller actually invokes them; a single ``RuntimeWarning`` is emitted
at import time.
"""

import warnings

from ._arch import _detect_arch

_SUPPORTED = {"gfx950"}
_FEATURE = "aiter.ops.opus (a16w16)"
_HINT = (
    "opus_gemm uses gfx950-only intrinsics (MFMA, ds_read_b64_tr) and "
    "the 160 KiB LDS budget. Set GPU_ARCHS=gfx950 (or run on a gfx950 "
    "device) to use this module."
)

_arch_ok, _detected_arch = _detect_arch(_SUPPORTED)


def _make_unsupported_arch_stub(name: str):
    """Build a callable that always raises with the detected-arch context."""

    def _stub(*_args, **_kwargs):
        raise RuntimeError(
            f"{name} requires GPU arch in {sorted(_SUPPORTED)}; "
            f"detected {_detected_arch!r}. {_HINT}"
        )

    _stub.__name__ = name
    _stub.__qualname__ = name
    _stub.__doc__ = (
        f"Stub installed because {_FEATURE} is unavailable on this "
        f"device (detected {_detected_arch!r}). Calling it raises "
        f"RuntimeError. The real implementation is exposed only on "
        f"supported archs ({sorted(_SUPPORTED)})."
    )
    return _stub


if _arch_ok:
    from .gemm_op_a16w16 import (  # noqa: E402
        opus_gemm_a16w16_tune,
        gemm_a16w16_opus,
    )
else:
    # Non-supported arch (or unknown / probe failed): warn once and install
    # stubs. We deliberately do NOT raise ImportError here -- raising would
    # propagate up through `from aiter.ops.opus import *` in
    # aiter/__init__.py, where it would be caught by the surrounding
    # try/except and silently disable the 30+ subsequent op imports.
    warnings.warn(
        f"{_FEATURE} is gfx950-only; detected arch={_detected_arch!r}. "
        f"opus_gemm_* calls will raise RuntimeError at invocation. {_HINT}",
        RuntimeWarning,
        stacklevel=2,
    )
    gemm_a16w16_opus = _make_unsupported_arch_stub("gemm_a16w16_opus")
    opus_gemm_a16w16_tune = _make_unsupported_arch_stub("opus_gemm_a16w16_tune")


__all__ = [
    "opus_gemm_a16w16_tune",
    "gemm_a16w16_opus",
]
