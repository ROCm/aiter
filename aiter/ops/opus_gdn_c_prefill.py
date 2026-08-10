# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Standalone dense C-input Gated DeltaNet prefill backend."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from ..jit.core import compile_ops

OPUS_GDN_C_AUTO = 0
OPUS_GDN_C_FUSED = 1
OPUS_GDN_C_SPLIT = 2
OPUS_GDN_C_SUPPORTED_MODES = (
    OPUS_GDN_C_AUTO,
    OPUS_GDN_C_FUSED,
    OPUS_GDN_C_SPLIT,
)

_DENSE_CHUNK_SIZE = 64
_DENSE_FEATURE_SIZE = 128


def _shares_storage(a: torch.Tensor, b: torch.Tensor) -> bool:
    return a.untyped_storage().data_ptr() == b.untyped_storage().data_ptr()


@compile_ops("module_opus_gdn_c_prefill")
def _opus_gdn_c_prefill_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    o: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor,
    final_state: torch.Tensor,
    has_initial_state: bool,
    output_final_state: bool,
    c_mode: int,
    use_env_overrides: bool,
) -> None: ...


def opus_gdn_c_prefill_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    c_mode: int = OPUS_GDN_C_AUTO,
    out: torch.Tensor | None = None,
    use_env_overrides: bool = True,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Run the gfx942 dense C-input GDN prefill implementation.

    This backend is fixed to BT=64 and K=V=128. It exposes the two production
    paths directly:

    * c_mode=1: CF, fused recurrence and output.
    * c_mode=2: CS, split recurrence followed by shared dense K6.
    * c_mode=0: a conservative policy measured on an 80-CU gfx942. CS is
      selected for T_padded >= 256 and B*H <= 20 or
      T_padded >= 128 and B*H <= 8; every other shape uses CF.

    Explicit mode 1 or 2 is recommended when the model layer already owns a
    workload-specific dispatcher. Auto deliberately does not extrapolate the
    measured envelope to other architectures or broad unmeasured regions.

    Args:
        q: Query tensor with shape [B, T, H, 128].
        k: Key tensor with shape [B, T, H, 128].
        v: Value tensor with shape [B, T, H, 128].
        g: Log-space gate tensor with shape [B, T, H].
        beta: Update gate tensor with shape [B, T, H].
        scale: Query scale; defaults to 1 / sqrt(128).
        initial_state: Optional fp32 state in [B, H, V, K] layout.
        output_final_state: Return the fp32 final state when true.
        c_mode: 0=auto, 1=CF, or 2=CS.
        out: Optional preallocated contiguous bf16 output. Preallocation
            requires T to be divisible by 64. It must not share storage with
            any input tensor.
        use_env_overrides: Honor low-level kernel tuning environment variables
            when true. Production dispatchers should pass false to use the
            published kernel defaults regardless of process environment.

    Returns:
        A pair (output, final_state). Output has the original unpadded
        sequence length; final_state is None unless requested.
    """
    if c_mode not in OPUS_GDN_C_SUPPORTED_MODES:
        raise ValueError(
            f"Unsupported c_mode={c_mode}; expected one of "
            f"{OPUS_GDN_C_SUPPORTED_MODES}"
        )
    if not isinstance(q, torch.Tensor) or q.ndim != 4:
        raise ValueError("q must be a tensor with shape [B, T, H, 128]")
    if q.shape[-1] != _DENSE_FEATURE_SIZE:
        raise ValueError(f"q feature size must be {_DENSE_FEATURE_SIZE}")
    if not q.is_cuda:
        raise ValueError("q must be a HIP tensor")
    device_gfx = torch.cuda.get_device_properties(q.device).gcnArchName.split(":", 1)[0]
    if device_gfx != "gfx942":
        raise ValueError(
            f"opus_gdn_c_prefill currently requires gfx942, got {device_gfx}"
        )

    B, T, H, K = q.shape
    if B <= 0 or T <= 0 or H <= 0:
        raise ValueError(f"B, T, and H must be positive, got {(B, T, H)}")
    expected_vector_shape = (B, T, H, _DENSE_FEATURE_SIZE)
    expected_scalar_shape = (B, T, H)
    for name, tensor in (("k", k), ("v", v)):
        if (
            not isinstance(tensor, torch.Tensor)
            or tuple(tensor.shape) != expected_vector_shape
        ):
            raise ValueError(
                f"{name} must have shape {expected_vector_shape}, got "
                f"{getattr(tensor, 'shape', None)}"
            )
        if tensor.device != q.device:
            raise ValueError(f"{name} must be on the same device as q")
    for name, tensor in (("g", g), ("beta", beta)):
        if (
            not isinstance(tensor, torch.Tensor)
            or tuple(tensor.shape) != expected_scalar_shape
        ):
            raise ValueError(
                f"{name} must have shape {expected_scalar_shape}, got "
                f"{getattr(tensor, 'shape', None)}"
            )
        if tensor.device != q.device:
            raise ValueError(f"{name} must be on the same device as q")

    if scale is None:
        scale = K**-0.5

    pad_len = (-T) % _DENSE_CHUNK_SIZE
    if out is not None:
        if pad_len:
            raise ValueError("a preallocated output requires T to be divisible by 64")
        if not isinstance(out, torch.Tensor):
            raise ValueError("out must be a torch.Tensor")
        if (
            tuple(out.shape) != expected_vector_shape
            or out.dtype != torch.bfloat16
            or out.device != q.device
            or not out.is_contiguous()
        ):
            raise ValueError(
                "out must be a contiguous bf16 tensor matching v shape/device"
            )
        for name, read_only in (
            ("q", q),
            ("k", k),
            ("v", v),
            ("g", g),
            ("beta", beta),
            ("initial_state", initial_state),
        ):
            if isinstance(read_only, torch.Tensor) and _shares_storage(out, read_only):
                raise ValueError(f"out must not alias {name} storage")

    q = q.contiguous().to(torch.bfloat16)
    k = k.contiguous().to(torch.bfloat16)
    v = v.contiguous().to(torch.bfloat16)
    g = g.contiguous().float()
    beta = beta.contiguous().float()

    if pad_len:
        q = F.pad(q, (0, 0, 0, 0, 0, pad_len))
        k = F.pad(k, (0, 0, 0, 0, 0, pad_len))
        v = F.pad(v, (0, 0, 0, 0, 0, pad_len))
        g = F.pad(g, (0, 0, 0, pad_len))
        beta = F.pad(beta, (0, 0, 0, pad_len))

    padded_T = T + pad_len
    output = (
        out
        if out is not None
        else torch.empty((B, padded_T, H, 128), dtype=torch.bfloat16, device=q.device)
    )
    has_initial_state = initial_state is not None
    if has_initial_state:
        expected_state_shape = (B, H, 128, 128)
        if tuple(initial_state.shape) != expected_state_shape:
            raise ValueError(
                f"initial_state must have shape {expected_state_shape}, got "
                f"{tuple(initial_state.shape)}"
            )
        if initial_state.device != q.device:
            raise ValueError("initial_state must be on the same device as q")
        initial = initial_state.contiguous().float()
    else:
        initial = torch.empty(0, dtype=torch.float32, device=q.device)

    final = (
        torch.empty((B, H, 128, 128), dtype=torch.float32, device=q.device)
        if output_final_state
        else torch.empty(0, dtype=torch.float32, device=q.device)
    )
    _opus_gdn_c_prefill_fwd(
        q,
        k,
        v,
        g,
        beta,
        output,
        float(scale),
        initial,
        final,
        has_initial_state,
        output_final_state,
        c_mode,
        use_env_overrides,
    )

    if pad_len:
        output = output[:, :T]
    return output, final if output_final_state else None
