# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Fused Gated DeltaNet decode and MTP verify on gfx942 with an fp32 SSM state.

One launch per layer starts from the in_proj outputs in the flat layout of
Qwen3.5 / Qwen3.8 (``qkvz`` = [q | k | v | z], ``ba`` = [b | a]) and replaces the unpack, the causal conv1d update and the
gating + q/k L2 norm + delta-rule recurrence. The Gluon ``fused_gdn_decode_qkvz`` does
the same fusion but is gfx950-only; on gfx942, ``flydsl_gdr_decode`` and
``flydsl_gdr_mtp_sglang`` run only the recurrence and need a separate conv launch.
Callers gate on :func:`gdn_fused_decode_supported`. The interleaved Qwen3-Next layout
(``qkvz_layout="interleaved"``) has the same shapes and is rejected.

fused_qkvzba_conv1d_gated_delta_rule_decode_flydsl
    One token per request. Updates the conv and SSM state pools in place and writes
    o [B, 1, HV, V] and, if given, a contiguous z [B, HV, V] for the gated norm.
fused_qkvzba_conv1d_gated_delta_rule_verify_flydsl
    Chain MTP target verify (T draft tokens per request, topk 1). Writes each step's
    output, conv window and state to the intermediate caches that the rollback reads.
    The SSM state is only read.
"""

from __future__ import annotations

import functools

import torch

from .kernels.gdn_decode_fused import create_gdn_decode_fused_kernel
from .kernels.gdn_verify_fused import create_gdn_verify_fused_kernel
from .kernels.tensor_shim import _run_compiled

__all__ = [
    "fused_qkvzba_conv1d_gated_delta_rule_decode_flydsl",
    "fused_qkvzba_conv1d_gated_delta_rule_verify_flydsl",
    "gdn_fused_decode_supported",
]

_WARP_THREADS_K = 8
_HEAD_DIM = 128
_CONV_WIDTH = 4
_QKVZ_LAYOUT = "flat"
# Per-rank (k heads, v heads) of Qwen3.8 GDN (16/48 heads) at TP1/2/4: the only
# head counts these kernels are tuned and tested for.
_QWEN38_HEADS = ((16, 48), (8, 24), (4, 12))
_COUNTERS: dict = {}


def _arrival_counter(device, n: int) -> torch.Tensor:
    """Zero-initialised per-(request, k-head) counters; the kernel resets what it uses.

    One buffer per device: launches on one device must not overlap on two streams.
    """
    buf = _COUNTERS.get(str(device))
    if buf is None or buf.numel() < n:
        # A zero-fill recorded during graph capture would not run before other graphs
        # use the buffer, so it must be created eagerly (warm-up runs before capture).
        assert (
            not torch.cuda.is_current_stream_capturing()
        ), "GDN decode arrival counter must be allocated before CUDA graph capture"
        buf = torch.zeros(max(n, 1 << 16), dtype=torch.int32, device=device)
        _COUNTERS[str(device)] = buf
    return buf


def _dt(t: torch.Tensor) -> str:
    assert t.dtype in (torch.bfloat16, torch.float32)
    return "bf16" if t.dtype == torch.bfloat16 else "f32"


def _check_index(t: torch.Tensor):
    assert t.dtype == torch.int32 and t.dim() == 1 and t.stride(0) == 1


@functools.cache
def _launcher(factory, device_index: int, config: tuple, kwargs: tuple):
    """One launcher per GPU: a compiled handle built on one device does not launch
    on another, and ``_run_compiled`` caches the first handle it builds."""
    return factory.__wrapped__(*config, **dict(kwargs))


def _launch(factory, config: tuple, kwargs: tuple, device: torch.device, *args):
    with torch.cuda.device(device):
        exe = _launcher(factory, torch.cuda.current_device(), config, kwargs)
        _run_compiled(exe, *args, torch.cuda.current_stream())


def _check_layout(qkvz_layout: str):
    # Raised, not asserted: python -O strips an assert, and an interleaved qkvz has the
    # same shapes as the flat one, so nothing else would stop it.
    if qkvz_layout != _QKVZ_LAYOUT:
        raise ValueError(
            "the FlyDSL kernels read the flat [q|k|v|z] layout (Qwen3.5/3.8), "
            f"got qkvz_layout={qkvz_layout!r}"
        )


def _check_heads(H: int, HV: int):
    # Raised, not asserted: python -O strips an assert.
    if (H, HV) not in _QWEN38_HEADS:
        raise ValueError(
            f"the FlyDSL GDN kernels serve the Qwen3.8 head counts {_QWEN38_HEADS}, "
            f"got (num_k_heads, num_v_heads) = ({H}, {HV})"
        )


def _require_supported(name: str, *probe_args):
    # The ops re-run the probe, so nothing reaches the kernels off gfx942 or off the
    # Qwen3.8 shapes even when a caller skips gdn_fused_decode_supported().
    if not gdn_fused_decode_supported(*probe_args):
        raise ValueError(
            f"{name}: unsupported arch or tensors (gfx942 and the Qwen3.8 shapes "
            "only); gate on gdn_fused_decode_supported()"
        )


def _check_params(A_log: torch.Tensor, dt_bias: torch.Tensor, H: int, HV: int):
    assert A_log.dtype == torch.float32 and A_log.is_contiguous()
    assert dt_bias.is_contiguous() and A_log.numel() == dt_bias.numel() == HV
    assert H > 0 and HV % H == 0


def _check_state(initial_state: torch.Tensor, scale: float):
    assert initial_state.dim() == 4, "SSM state pool must be [slots, HV, V, K]"
    HV, V, K = initial_state.shape[-3:]
    assert K == V == 128 and abs(scale - K**-0.5) < 1e-12
    assert initial_state.dtype == torch.float32 and initial_state.stride(-1) == 1
    assert initial_state.stride(-2) == K and initial_state.stride(-3) == V * K
    return HV, V, K


def gdn_fused_decode_supported(
    projected_qkvz: torch.Tensor,
    projected_ba: torch.Tensor,
    conv_states: torch.Tensor,
    conv_weights: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    ssm_states: torch.Tensor,
    num_k_heads: int,
    qkvz_layout: str = "flat",
) -> bool:
    """True when the fused decode / verify kernels can serve these tensors.

    Requires gfx942, the Qwen3.8 per-rank head counts (``num_k_heads``, ``HV``) in
    ``(16, 48), (8, 24), (4, 12)`` (TP1/2/4), the flat ``[q|k|v|z]`` in_proj
    layout (``qkvz_layout="flat"``; the interleaved Qwen3-Next layout returns False),
    bf16 activations and conv cache, width-4 conv weights, fp32 ``A_log``, an fp32 SSM
    pool ``[slots, HV, 128, 128]`` with contiguous heads and 128-wide heads. Callers
    keep their unfused path when this returns False.
    """
    if qkvz_layout != _QKVZ_LAYOUT:
        return False
    try:
        from aiter.jit.utils.chip_info import get_gfx_runtime

        if get_gfx_runtime() != "gfx942":
            return False
    except Exception:  # noqa: BLE001 - no device or unknown arch
        return False
    D, H = _HEAD_DIM, num_k_heads
    if ssm_states.dim() != 4 or ssm_states.shape[-2:] != (D, D) or H <= 0:
        return False
    HV = ssm_states.shape[1]
    return (
        (H, HV) in _QWEN38_HEADS
        and projected_qkvz.dtype == projected_ba.dtype == torch.bfloat16
        and projected_qkvz.stride(-1) == projected_ba.stride(-1) == 1
        and projected_qkvz.shape[-1] >= 2 * H * D + 2 * HV * D
        and projected_ba.shape[-1] >= 2 * HV
        and conv_states.dtype == conv_weights.dtype == torch.bfloat16
        and conv_states.dim() == 3
        and conv_states.shape[-1] == _CONV_WIDTH - 1
        and conv_weights.is_contiguous()
        and tuple(conv_weights.shape) == (2 * H * D + HV * D, _CONV_WIDTH)
        and A_log.dtype == torch.float32
        and A_log.is_contiguous()
        and dt_bias.dtype in (torch.bfloat16, torch.float32)
        and dt_bias.is_contiguous()
        and A_log.numel() == dt_bias.numel() == HV
        and ssm_states.dtype == torch.float32
        and ssm_states.stride()[1:] == (D * D, D, 1)
    )


def fused_qkvzba_conv1d_gated_delta_rule_decode_flydsl(
    projected_qkvz: torch.Tensor,
    projected_ba: torch.Tensor,
    conv_states: torch.Tensor,
    conv_weights: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor,
    out: torch.Tensor,
    ssm_state_indices: torch.Tensor,
    num_k_heads: int,
    z_out: torch.Tensor | None = None,
    use_qk_l2norm_in_kernel: bool = True,
    qkvz_layout: str = "flat",
) -> tuple[torch.Tensor, torch.Tensor]:
    """``ssm_state_indices`` indexes both the SSM and the conv state pools (-1 = pad)."""
    assert use_qk_l2norm_in_kernel, "the FlyDSL kernel always L2-normalises q and k"
    _check_layout(qkvz_layout)
    B = projected_qkvz.shape[0]
    HV, V, K = _check_state(initial_state, scale)
    H = num_k_heads
    assert projected_qkvz.dim() == projected_ba.dim() == 2
    assert projected_qkvz.stride(-1) == 1 and projected_ba.stride(-1) == 1
    assert projected_qkvz.shape[1] >= 2 * H * K + 2 * HV * V
    assert projected_ba.shape[1] >= 2 * HV
    assert projected_qkvz.dtype == projected_ba.dtype == out.dtype == torch.bfloat16
    _check_heads(H, HV)
    _require_supported(
        "fused_qkvzba_conv1d_gated_delta_rule_decode_flydsl",
        projected_qkvz,
        projected_ba,
        conv_states,
        conv_weights,
        A_log,
        dt_bias,
        initial_state,
        H,
        qkvz_layout,
    )
    _check_params(A_log, dt_bias, H, HV)
    assert conv_weights.is_contiguous() and conv_weights.shape[-1] == 4
    assert conv_weights.dtype == torch.bfloat16
    assert conv_states.dim() == 3 and conv_states.shape[-1] == 3
    assert conv_states.dtype == torch.bfloat16
    _check_index(ssm_state_indices)
    write_z = z_out is not None
    assert out.is_contiguous() and out.numel() == B * HV * V
    if write_z:
        assert z_out.is_contiguous() and z_out.numel() == B * HV * V
        assert z_out.dtype == torch.bfloat16
    device = projected_qkvz.device
    _launch(
        create_gdn_decode_fused_kernel,
        (H, HV, K, _dt(dt_bias), _WARP_THREADS_K),
        (("write_z", write_z),),
        device,
        projected_qkvz,
        projected_ba,
        conv_weights,
        conv_states,
        _arrival_counter(device, B * H),
        A_log.view(-1),
        dt_bias.view(-1),
        initial_state,
        ssm_state_indices,
        out.view(-1),
        (z_out if write_z else out).view(-1),
        projected_qkvz.stride(0),
        projected_ba.stride(0),
        conv_states.stride(0),
        conv_states.stride(1),
        conv_states.stride(2),
        initial_state.stride(0),
        HV * V,
        B,
    )
    return out, initial_state


def fused_qkvzba_conv1d_gated_delta_rule_verify_flydsl(
    projected_qkvz: torch.Tensor,
    projected_ba: torch.Tensor,
    conv_states: torch.Tensor,
    conv_weights: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor,
    out: torch.Tensor,
    ssm_state_indices: torch.Tensor,
    intermediate_states: torch.Tensor,
    intermediate_state_indices: torch.Tensor,
    intermediate_conv_window: torch.Tensor,
    num_k_heads: int,
    num_draft_tokens: int,
    z_out: torch.Tensor,
    qkvz_layout: str = "flat",
) -> torch.Tensor:
    """Chain target verify, rows ordered [request, draft token]. ``ssm_state_indices``
    indexes the SSM and conv state pools (-1 = pad); ``intermediate_state_indices``
    indexes the rows of ``intermediate_states`` [row, step, HV, V, K] and
    ``intermediate_conv_window`` [row, step, conv_dim, 3]. The SSM state is read only.
    """
    _check_layout(qkvz_layout)
    T = num_draft_tokens
    assert projected_qkvz.dim() == projected_ba.dim() == 2
    assert projected_qkvz.shape[0] % T == 0
    B = projected_qkvz.shape[0] // T
    HV, V, K = _check_state(initial_state, scale)
    H = num_k_heads
    assert projected_qkvz.stride(-1) == 1 and projected_ba.stride(-1) == 1
    assert projected_qkvz.shape[1] >= 2 * H * K + 2 * HV * V
    assert projected_ba.shape[1] >= 2 * HV
    assert projected_qkvz.dtype == projected_ba.dtype == out.dtype == torch.bfloat16
    _check_heads(H, HV)
    _require_supported(
        "fused_qkvzba_conv1d_gated_delta_rule_verify_flydsl",
        projected_qkvz,
        projected_ba,
        conv_states,
        conv_weights,
        A_log,
        dt_bias,
        initial_state,
        H,
        qkvz_layout,
    )
    _check_params(A_log, dt_bias, H, HV)
    assert conv_weights.is_contiguous() and conv_weights.shape[-1] == 4
    assert conv_weights.dtype == conv_states.dtype == torch.bfloat16
    assert conv_states.dim() == 3 and conv_states.shape[-1] == 3
    assert (
        intermediate_conv_window.dim() == 4 and intermediate_conv_window.shape[-1] == 3
    )
    assert intermediate_conv_window.dtype == torch.bfloat16
    assert intermediate_conv_window.shape[1] >= T
    assert intermediate_states.dtype == torch.float32
    assert intermediate_states.dim() == 5 and intermediate_states.shape[1] >= T
    assert intermediate_states.shape[2:] == (HV, V, K)
    assert intermediate_states[0, 0].is_contiguous()
    _check_index(ssm_state_indices)
    _check_index(intermediate_state_indices)
    assert ssm_state_indices.numel() >= B and intermediate_state_indices.numel() >= B
    assert out.is_contiguous() and out.numel() == B * T * HV * V
    assert z_out.is_contiguous() and z_out.numel() == B * T * HV * V
    assert z_out.dtype == torch.bfloat16
    w = intermediate_conv_window
    device = projected_qkvz.device
    _launch(
        create_gdn_verify_fused_kernel,
        (H, HV, T, K, _dt(dt_bias), _WARP_THREADS_K),
        (),
        device,
        projected_qkvz,
        projected_ba,
        conv_weights,
        conv_states,
        w,
        _arrival_counter(device, B * H),
        A_log.view(-1),
        dt_bias.view(-1),
        initial_state,
        intermediate_states,
        ssm_state_indices,
        intermediate_state_indices,
        out.view(-1),
        z_out.view(-1),
        projected_qkvz.stride(0),
        projected_ba.stride(0),
        conv_states.stride(0),
        conv_states.stride(1),
        conv_states.stride(2),
        w.stride(0),
        w.stride(1),
        w.stride(2),
        w.stride(3),
        initial_state.stride(0),
        intermediate_states.stride(0),
        intermediate_states.stride(1),
        B,
    )
    return out
