# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
"""opus kernel Python user-facing API; a16w16 today, a8w8 in follow-ups.

Public API: `gemm_a16w16_opus` (CSV lookup + C++ heuristic) and
`opus_gemm_a16w16_tune` (id-based binding). On unsupported arch the
two callables become stubs that raise RuntimeError on invocation, so
`import aiter` keeps working alongside its 30+ other ops.
"""

from ._arch import _detect_arch

_SUPPORTED = {"gfx950", "gfx942", "gfx1250"}
_FEATURE = "aiter.ops.opus (a16w16)"
_HINT = (
    "opus_gemm supports gfx950 (MFMA 16x16x32 / ds_read_b64_tr / 160 KiB "
    "LDS) and gfx942 (MFMA 16x16x16 / ds_read_b128 / 64 KiB LDS). Set "
    "GPU_ARCHS to one of these (or run on a matching device) to use this "
    "module."
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
    _stub.__doc__ = f"Stub: {_FEATURE} unavailable on {_detected_arch!r}."
    return _stub


if _arch_ok:
    from .gemm_op_a16w16 import (  # noqa: E402
        opus_gemm_a16w16_tune,
        gemm_a16w16_opus,
        opus_gemm_workspace_init,
    )
else:
    # Don't raise ImportError -- aiter/__init__.py's star-import would catch
    # it and silently disable the 30+ subsequent op imports.
    gemm_a16w16_opus = _make_unsupported_arch_stub("gemm_a16w16_opus")
    opus_gemm_a16w16_tune = _make_unsupported_arch_stub("opus_gemm_a16w16_tune")
    opus_gemm_workspace_init = _make_unsupported_arch_stub("opus_gemm_workspace_init")


__all__ = [
    "opus_gemm_a16w16_tune",
    "gemm_a16w16_opus",
    "opus_gemm_workspace_init",
]
