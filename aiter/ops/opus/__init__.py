# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
"""opus kernel Python user-facing API.

Public API: `gemm_a16w16_opus` (CSV lookup + C++ heuristic) and
`opus_gemm_a16w16_tune` (id-based binding). The gfx942 A8W8 blockscale
bpreshuffle entry is an explicit tune API.
"""

from ._arch import _detect_arch

_SUPPORTED = {"gfx950", "gfx942", "gfx1250"}
_FEATURE = "aiter.ops.opus"
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
    from .bmm_op import bmm_a8w8_mxscale_opus
    from .gemm_op_a16w16 import (
        gemm_a16w16_opus,
        opus_gemm_a16w16_tune,
        opus_gemm_workspace_init,
        opus_gemm_workspace_release,
        opus_gemm_workspace_release_all,
    )

    # gfx950-only, unlike the gemm entries above: _SUPPORTED is wider than this
    # kernel, so the wrappers re-check at call time rather than at import.
    from .pa_mqa_logits_opus import (
        compute_prefill_windows,
        pa_mqa_logits_mxfp4_decode,
        pa_mqa_logits_mxfp4_fwd_decode,
        pa_mqa_logits_mxfp4_fwd_prefill,
        pa_mqa_logits_mxfp4_prefill,
        pa_mqa_logits_mxfp4_prefill_windows,
    )

    def opus_gemm_a8w8_blockscale_bpreshuffle_tune(*args, **kwargs):
        from .gemm_op_a8w8 import (
            opus_gemm_a8w8_blockscale_bpreshuffle_tune as _impl,
        )

        return _impl(*args, **kwargs)

else:
    # Don't raise ImportError -- aiter/__init__.py's star-import would catch
    # it and silently disable the 30+ subsequent op imports.
    gemm_a16w16_opus = _make_unsupported_arch_stub("gemm_a16w16_opus")
    opus_gemm_a16w16_tune = _make_unsupported_arch_stub("opus_gemm_a16w16_tune")
    bmm_a8w8_mxscale_opus = _make_unsupported_arch_stub("bmm_a8w8_mxscale_opus")
    opus_gemm_a8w8_blockscale_bpreshuffle_tune = _make_unsupported_arch_stub(
        "opus_gemm_a8w8_blockscale_bpreshuffle_tune"
    )
    opus_gemm_workspace_init = _make_unsupported_arch_stub("opus_gemm_workspace_init")
    opus_gemm_workspace_release = _make_unsupported_arch_stub(
        "opus_gemm_workspace_release"
    )
    opus_gemm_workspace_release_all = _make_unsupported_arch_stub(
        "opus_gemm_workspace_release_all"
    )
    compute_prefill_windows = _make_unsupported_arch_stub("compute_prefill_windows")
    pa_mqa_logits_mxfp4_decode = _make_unsupported_arch_stub(
        "pa_mqa_logits_mxfp4_decode"
    )
    pa_mqa_logits_mxfp4_fwd_decode = _make_unsupported_arch_stub(
        "pa_mqa_logits_mxfp4_fwd_decode"
    )
    pa_mqa_logits_mxfp4_fwd_prefill = _make_unsupported_arch_stub(
        "pa_mqa_logits_mxfp4_fwd_prefill"
    )
    pa_mqa_logits_mxfp4_prefill = _make_unsupported_arch_stub(
        "pa_mqa_logits_mxfp4_prefill"
    )
    pa_mqa_logits_mxfp4_prefill_windows = _make_unsupported_arch_stub(
        "pa_mqa_logits_mxfp4_prefill_windows"
    )


__all__ = [
    "bmm_a8w8_mxscale_opus",
    "compute_prefill_windows",
    "gemm_a16w16_opus",
    "opus_gemm_a8w8_blockscale_bpreshuffle_tune",
    "opus_gemm_a16w16_tune",
    "opus_gemm_workspace_init",
    "opus_gemm_workspace_release",
    "opus_gemm_workspace_release_all",
    "pa_mqa_logits_mxfp4_decode",
    "pa_mqa_logits_mxfp4_fwd_decode",
    "pa_mqa_logits_mxfp4_fwd_prefill",
    "pa_mqa_logits_mxfp4_prefill",
    "pa_mqa_logits_mxfp4_prefill_windows",
]
