# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Gated DeltaNet dense/packed prefill dispatcher with a Triton fallback.

Dense auto routes are limited to the measured gfx942 domain. Packed varlen can
select the metadata-aware W/U split path. Unsupported inputs are delegated,
unchanged, to ``chunk_gated_delta_rule_opt_vk``. Models that already own a
workload policy may request an implementation family explicitly through
``path``.
"""

from __future__ import annotations

import math
from typing import Literal

import torch

from ._gdn_dense_gfx942_routes import lookup_dense_gfx942_path
from .opus_gdn_c_prefill import (
    OPUS_GDN_C_FUSED,
    OPUS_GDN_C_SPLIT,
    opus_gdn_c_prefill_fwd,
)
from .opus_gdn_wu_prefill import (
    OPUS_GDN_K2_SPLIT,
    OPUS_GDN_K2_WU_FUSED,
    _prepare_opus_gdn_varlen_metadata,
    opus_gdn_wu_prefill_fwd,
)
from .triton.gated_delta_net import chunk_gated_delta_rule_opt_vk

GdnPrefillPath = Literal["auto", "c", "cf", "cs", "wu", "wf", "ws", "triton"]
_EXPLICIT_OPUS_PATHS = frozenset(("c", "cf", "cs", "wu", "wf", "ws"))
_ALL_PATHS = _EXPLICIT_OPUS_PATHS | {"auto", "triton"}
_DENSE_BT = 64
_DENSE_DIM = 128
_MEASURED_GFX = "gfx942"
_MEASURED_CU_COUNT = 80


def _runtime_target(q: torch.Tensor) -> tuple[str, int]:
    properties = torch.cuda.get_device_properties(q.device)
    gfx = properties.gcnArchName.split(":", 1)[0]
    return gfx, int(properties.multi_processor_count)


def _shares_storage(a: torch.Tensor, b: torch.Tensor) -> bool:
    return a.untyped_storage().data_ptr() == b.untyped_storage().data_ptr()


def _is_exact_contiguous_view(a: torch.Tensor, b: torch.Tensor) -> bool:
    return (
        a.data_ptr() == b.data_ptr()
        and tuple(a.shape) == tuple(b.shape)
        and tuple(a.stride()) == tuple(b.stride())
    )


def _opus_input_error(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    o: torch.Tensor | None,
    g: torch.Tensor | None,
    beta: torch.Tensor | None,
    initial_state: torch.Tensor | None,
    *,
    allow_padding: bool,
    use_qk_l2norm_in_kernel: bool,
    cu_seqlens: torch.LongTensor | None,
    use_chunk_hip: bool,
    use_chunk_flydsl: bool,
    state_dtype: torch.dtype | None,
    use_exp2: bool,
    num_decodes: int,
    num_decode_tokens: int,
) -> str | None:
    is_varlen = cu_seqlens is not None
    tensors = (("q", q), ("k", k), ("v", v))
    for name, tensor in tensors:
        if not isinstance(tensor, torch.Tensor):
            return f"{name} must be a torch.Tensor"
        if not tensor.is_cuda:
            return f"{name} must be a HIP tensor"
        if tensor.dtype != torch.bfloat16:
            return f"{name} must have dtype bfloat16"
        if not tensor.is_contiguous():
            return f"{name} must be contiguous"

    if q.ndim != 4:
        return "q must have shape [B, T, H, 128]"
    B, T, H, K = q.shape
    if B <= 0 or T <= 0 or H <= 0:
        return "B, T, and H must be positive"
    if K != _DENSE_DIM:
        return "q feature size must be 128"
    if is_varlen and B != 1:
        return "packed varlen expects B=1"
    expected_vector_shape = (B, T, H, _DENSE_DIM)
    if tuple(k.shape) != expected_vector_shape:
        return "k must match q shape [B, T, H, 128]"
    if tuple(v.shape) != expected_vector_shape:
        return "v must match q shape [B, T, H, 128]"
    if k.device != q.device or v.device != q.device:
        return "q, k, and v must be on the same device"
    if not is_varlen and T % _DENSE_BT and (not allow_padding or o is not None):
        return "T must be divisible by 64"

    if is_varlen:
        if not isinstance(cu_seqlens, torch.Tensor):
            return "cu_seqlens must be a torch.Tensor"
        if cu_seqlens.ndim != 1 or cu_seqlens.numel() < 2:
            return "cu_seqlens must have shape [N + 1]"
        if cu_seqlens.device != q.device:
            return "cu_seqlens must be on the same device as q"
        if cu_seqlens.dtype not in (torch.int32, torch.int64):
            return "cu_seqlens must have dtype int32 or int64"
        if not cu_seqlens.is_contiguous():
            return "cu_seqlens must be contiguous"
        state_batch = cu_seqlens.numel() - 1
    else:
        state_batch = B

    expected_scalar_shape = (B, T, H)
    for name, tensor in (("g", g), ("beta", beta)):
        if not isinstance(tensor, torch.Tensor):
            return f"{name} must be a torch.Tensor"
        if tuple(tensor.shape) != expected_scalar_shape:
            return f"{name} must have shape [B, T, H]"
        if tensor.device != q.device:
            return f"{name} must be on the same device as q"
        if tensor.dtype not in (torch.bfloat16, torch.float32):
            return f"{name} must have dtype bfloat16 or float32"
        if not tensor.is_contiguous():
            return f"{name} must be contiguous"

    if o is not None:
        if not isinstance(o, torch.Tensor):
            return "o must be a torch.Tensor"
        if (
            tuple(o.shape) != expected_vector_shape
            or o.device != q.device
            or o.dtype != torch.bfloat16
            or not o.is_contiguous()
        ):
            return "o must be contiguous bf16 and match v shape/device"

    if initial_state is not None:
        if not isinstance(initial_state, torch.Tensor):
            return "initial_state must be a torch.Tensor"
        expected_state_shape = (state_batch, H, _DENSE_DIM, _DENSE_DIM)
        if tuple(initial_state.shape) != expected_state_shape:
            return f"initial_state must have shape {expected_state_shape}"
        if initial_state.device != q.device:
            return "initial_state must be on the same device as q"
        if initial_state.dtype != torch.float32:
            return "initial_state must have dtype float32"
        if not initial_state.is_contiguous():
            return "initial_state must be contiguous"

    if o is not None:
        for name, read_only in (
            ("q", q),
            ("k", k),
            ("g", g),
            ("beta", beta),
            ("initial_state", initial_state),
        ):
            if read_only is not None and _shares_storage(o, read_only):
                return f"o must not alias {name} storage"
        if _shares_storage(o, v) and not _is_exact_contiguous_view(o, v):
            return "o may alias v only when it is exactly the same view"

    if state_dtype not in (None, torch.float32):
        return "state_dtype must be None or float32"
    if use_qk_l2norm_in_kernel:
        return "use_qk_l2norm_in_kernel is not supported by Opus dense paths"
    if use_chunk_hip or use_chunk_flydsl:
        return "explicit chunk HIP/FlyDSL selection requires the fallback"
    if not use_exp2:
        return "use_exp2=False is not supported by Opus dense paths"
    if num_decodes != 0 or num_decode_tokens != 0:
        return "decode-prefix inputs are not supported by Opus dense paths"
    return None


def _select_wu_path(T: int, batch_heads: int, with_state_io: bool) -> str:
    """Deterministic WS/WF envelope from the dense closeout."""
    if T <= 64:
        return "wf"
    if T <= 256:
        threshold = 24 if with_state_io else 21
    elif T <= 512:
        threshold = 40
    elif T <= 8192:
        threshold = 48
    else:
        threshold = 64
    return "wf" if batch_heads >= threshold else "ws"


def _select_c_path(T: int, batch_heads: int) -> str:
    """Mirror the standalone C backend's conservative deterministic policy."""
    use_split = (T >= 256 and batch_heads <= 20) or (T >= 128 and batch_heads <= 8)
    return "cs" if use_split else "cf"


def select_gdn_prefill_path(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    o: torch.Tensor | None = None,
    g: torch.Tensor | None = None,
    beta: torch.Tensor | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
    use_chunk_hip: bool = False,
    use_chunk_flydsl: bool = False,
    state_dtype: torch.dtype | None = None,
    use_exp2: bool = True,
    num_decodes: int = 0,
    num_decode_tokens: int = 0,
    *,
    path: GdnPrefillPath = "auto",
) -> str:
    """Return the concrete implementation family without launching a kernel."""
    normalized_path = str(path).lower()
    if normalized_path not in _ALL_PATHS:
        raise ValueError(
            f"unsupported path={path!r}; expected one of {sorted(_ALL_PATHS)}"
        )
    if normalized_path == "triton":
        return "triton"

    explicit = normalized_path in _EXPLICIT_OPUS_PATHS
    error = _opus_input_error(
        q,
        k,
        v,
        o,
        g,
        beta,
        initial_state,
        allow_padding=explicit,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        cu_seqlens=cu_seqlens,
        use_chunk_hip=use_chunk_hip,
        use_chunk_flydsl=use_chunk_flydsl,
        state_dtype=state_dtype,
        use_exp2=use_exp2,
        num_decodes=num_decodes,
        num_decode_tokens=num_decode_tokens,
    )
    if error is not None:
        unsafe_alias = error.startswith(("o must not alias", "o may alias"))
        if explicit or unsafe_alias:
            raise ValueError(f"path={normalized_path!r} is unavailable: {error}")
        return "triton"

    is_varlen = cu_seqlens is not None
    if is_varlen and normalized_path in ("c", "cf", "cs", "wf"):
        if explicit:
            raise ValueError(
                f"path={normalized_path!r} is unavailable: packed varlen "
                "currently supports the W/U split path only"
            )
        return "triton"

    gfx, cu_count = _runtime_target(q)
    if explicit:
        if normalized_path in ("c", "cf", "cs") and gfx != "gfx942":
            raise ValueError(f"path={normalized_path!r} requires gfx942, got {gfx}")
        if normalized_path in ("wu", "wf", "ws") and gfx not in (
            "gfx942",
            "gfx950",
        ):
            raise ValueError(
                f"path={normalized_path!r} requires gfx942/gfx950, got {gfx}"
            )
    elif gfx != _MEASURED_GFX or cu_count != _MEASURED_CU_COUNT:
        return "triton"

    B, T, H, _ = q.shape
    if is_varlen:
        # Ragged workloads are not keys in the dense winner table.  The
        # metadata-aware native implementation is the W/U split family.
        assert cu_seqlens is not None
        try:
            total_tokens, _, _, _, _ = _prepare_opus_gdn_varlen_metadata(
                cu_seqlens, _DENSE_BT
            )
            if total_tokens != T:
                raise ValueError("cu_seqlens endpoints must be 0 and total_tokens")
        except ValueError as exc:
            if explicit:
                raise ValueError(
                    f"path={normalized_path!r} is unavailable: {exc}"
                ) from exc
            return "triton"
        return "ws"
    padded_T = ((T + _DENSE_BT - 1) // _DENSE_BT) * _DENSE_BT
    with_state_io = initial_state is not None or output_final_state

    if normalized_path == "c":
        selected = _select_c_path(padded_T, B * H)
    elif normalized_path == "wu":
        selected = _select_wu_path(padded_T, B * H, with_state_io)
    elif explicit:
        selected = normalized_path
    else:
        measured_state: bool | None
        if initial_state is None and not output_final_state:
            measured_state = False
        elif initial_state is not None and output_final_state:
            measured_state = True
        else:
            measured_state = None
        selected = None
        if measured_state is not None:
            selected = lookup_dense_gfx942_path(B, T, H, measured_state)
        if selected is None:
            selected = _select_wu_path(T, B * H, with_state_io)

    if selected in ("cf", "cs") and o is not None and _shares_storage(o, v):
        if explicit:
            raise ValueError(f"path={selected!r} requires o not to alias v storage")
        return "triton"
    return selected


def gdn_prefill(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    o: torch.Tensor | None = None,
    g: torch.Tensor | None = None,
    beta: torch.Tensor | None = None,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
    use_chunk_hip: bool = False,
    use_chunk_flydsl: bool = False,
    state_dtype: torch.dtype | None = None,
    use_exp2: bool = True,
    num_decodes: int = 0,
    num_decode_tokens: int = 0,
    *,
    path: GdnPrefillPath = "auto",
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Run dense/packed GDN prefill through Opus when eligible, else opt_vk.

    ``path="auto"`` uses the exact 477-shape gfx942/80-CU winner table.  A
    table miss uses the fixed W/U WS/WF envelope; it never consults process
    environment variables for family selection. Packed varlen bypasses that
    dense table and selects WS on the validated gfx942/80-CU target.
    ``path="triton"`` forces the canonical fallback. Explicit Opus paths raise
    on unsupported inputs rather than silently changing the requested family.
    For a captured packed-varlen call, prewarm with the same ``cu_seqlens``
    Tensor outside capture, retain it, and keep its offsets immutable across
    every replay; changing the packed partition requires graph recapture.
    """
    selected = select_gdn_prefill_path(
        q,
        k,
        v,
        o=o,
        g=g,
        beta=beta,
        initial_state=initial_state,
        output_final_state=output_final_state,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        cu_seqlens=cu_seqlens,
        use_chunk_hip=use_chunk_hip,
        use_chunk_flydsl=use_chunk_flydsl,
        state_dtype=state_dtype,
        use_exp2=use_exp2,
        num_decodes=num_decodes,
        num_decode_tokens=num_decode_tokens,
        path=path,
    )

    if selected == "triton":
        return chunk_gated_delta_rule_opt_vk(
            q=q,
            k=k,
            v=v,
            o=o,
            g=g,
            beta=beta,
            scale=scale,
            initial_state=initial_state,
            output_final_state=output_final_state,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            cu_seqlens=cu_seqlens,
            use_chunk_hip=use_chunk_hip,
            use_chunk_flydsl=use_chunk_flydsl,
            state_dtype=state_dtype,
            use_exp2=use_exp2,
            num_decodes=num_decodes,
            num_decode_tokens=num_decode_tokens,
        )

    if scale is not None and not math.isfinite(float(scale)):
        raise ValueError("scale must be finite for an Opus path")
    if selected in ("cf", "cs"):
        return opus_gdn_c_prefill_fwd(
            q,
            k,
            v,
            g,
            beta,
            scale=scale,
            initial_state=initial_state,
            output_final_state=output_final_state,
            c_mode=(OPUS_GDN_C_FUSED if selected == "cf" else OPUS_GDN_C_SPLIT),
            out=o,
            use_env_overrides=False,
        )

    return opus_gdn_wu_prefill_fwd(
        q,
        k,
        v,
        g,
        beta,
        scale=scale,
        initial_state=initial_state,
        output_final_state=output_final_state,
        BT=_DENSE_BT,
        BV=64,
        num_warps=8 if selected == "wf" else 4,
        k1_algo=1,
        k2_mode=(OPUS_GDN_K2_WU_FUSED if selected == "wf" else OPUS_GDN_K2_SPLIT),
        out=o,
        use_env_overrides=False,
        cu_seqlens=cu_seqlens,
    )


__all__ = ["GdnPrefillPath", "gdn_prefill", "select_gdn_prefill_path"]
