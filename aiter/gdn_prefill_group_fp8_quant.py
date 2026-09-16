# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Qwen3-Next GDN (gated delta-net) prefill: fused entry point that dispatches to
the single-launch Gluon kernel when it is supported, and falls back to an
aiter-native Python composition otherwise.

Two backends live behind one signature:

* **Gluon fast path** -- ``ops.triton.gated_delta_net.fused_gdn_prefill_qkvz``,
  the fused gfx950 kernel that folds split + conv + SiLU + QK-norm + gating +
  the chunked delta scan + gated RMSNorm + group-128 FP8 quant into a tight set
  of launches. This is where the measured prefill uplift lives, so it is the
  default when ``fused_gdn_prefill_qkvz_supported`` says the call is covered
  (gfx950, Triton >= 3.8, and an in-range ``(tokens, batch)`` / dtype contract).

* **Composition fallback** -- the four aiter pieces chained at the Python level
  across the shared conv/recurrent pools:

    1. ``causal_conv1d_split_qkv_hip_fn``  -- causal conv1d (silu), splits packed
                                              ``[Q|K|V]``, advances ``conv_state``.
    2. ``fused_gdn_gating_and_sigmoid``    -- ``g = -exp(A_log)*softplus(a+dt_bias)``,
                                              ``beta = sigmoid(b)``.
    3. ``chunk_gated_delta_rule``          -- chunked gated delta recurrence
                                              (q/k L2-normed in-kernel).
    4. ``gated_rmsnorm_fp8_group_quant``   -- per-head gated RMSNorm + fp8 group
                                              (size-128) quant.

  Several launches, no cross-kernel fusion -- not single-kernel perf, but a
  correct, Triton-3.7-safe, non-gfx950-safe entry point. It runs whenever the
  fast kernel is absent or does not cover the call.

Both backends share the signature and the 5-tuple return, so the choice is
invisible to callers (SIKL's ``sikl::gdn_prefill_group_fp8_quant`` op inherits
the fast path for free). Semantics match sglang's ``Qwen3GatedDeltaNet`` prefill
(``srt/layers/attention/linear/gdn_backend.py``).

    (bf16_out, conv_state, delta_state, fp8_out, scales) = gdn_prefill_group_fp8_quant(
        projected_qkvz, projected_ba, conv_state, delta_state,
        cache_indices, cu_seqlens, has_initial_state,
        conv_weight, conv_bias, a_log, dt_bias, norm_weight,
        scale=head_k_dim ** -0.5, eps=norm.eps,
    )

Constraints (inherited from the tail kernel): ``head_k_dim == head_v_dim == 128``
and group size 128.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from aiter.ops.causal_conv1d_fwd_split_qkv import causal_conv1d_split_qkv_hip_fn
from aiter.ops.gated_rmsnorm_fp8_group_quant import gated_rmsnorm_fp8_group_quant
from aiter.ops.triton._triton_kernels.gated_delta_rule.fused_qkvzba_split import (
    fused_qkvzba_split_reshape_cat_prefill,
)
from aiter.ops.triton._triton_kernels.gated_delta_rule.prefill.fused_gdn_gating_prefill import (
    fused_gdn_gating_and_sigmoid,
)
from aiter.ops.triton.gated_delta_net.gated_delta_rule import chunk_gated_delta_rule

__all__ = ["gdn_prefill_group_fp8_quant", "FP8PrecisionConfig", "FP8_E4M3_FN"]

_HEAD_DIM = 128  # gated_rmsnorm_fp8_group_quant supports only head_dim == 128.
_GROUP_SIZE = 128
_BACKENDS = ("auto", "gluon", "compose")


@dataclass(frozen=True)
class FP8PrecisionConfig:
    """Mirror of the Artemis fp8 precision descriptor so a caller can pass the
    same object to either backend. Only ``dtype`` steers the aiter tail kernel;
    the group-quant clamp (``group_quant_max``) is applied inside it."""

    dtype: torch.dtype
    max_finite: float
    group_quant_max: float


FP8_E4M3_FN = FP8PrecisionConfig(torch.float8_e4m3fn, 448.0, 448.0)


def _fast_ops():
    """The Gluon fast kernel and its per-call support predicate, or ``(None, None)``.

    Imported lazily and fail-closed: on any older Triton / non-gfx950 build where
    the module is absent, the caller drops to the composition. The import itself
    is light -- the Gluon/HIP tiles only load once a covered call dispatches.
    """
    try:
        from aiter.ops.triton.gated_delta_net.fused_gdn_prefill_qkvz import (
            fused_gdn_prefill_qkvz,
            fused_gdn_prefill_qkvz_supported,
        )
    except (ImportError, ModuleNotFoundError):
        return None, None
    return fused_gdn_prefill_qkvz, fused_gdn_prefill_qkvz_supported


def gdn_prefill_group_fp8_quant(
    projected_qkvz: torch.Tensor,
    projected_ba: torch.Tensor,
    conv_state: torch.Tensor,
    delta_state: torch.Tensor,
    cache_indices: torch.Tensor,
    cu_seqlens: torch.Tensor,
    has_initial_state: torch.Tensor,
    conv_weight: torch.Tensor,
    conv_bias: torch.Tensor | None,
    a_log: torch.Tensor,
    dt_bias: torch.Tensor,
    norm_weight: torch.Tensor,
    *,
    scale: float,
    eps: float = 1.0e-6,
    precision_config: FP8PrecisionConfig = FP8_E4M3_FN,
    return_bf16: bool = True,
    store_state_transposed: bool = False,
    backend: str = "auto",
) -> tuple[
    torch.Tensor | None, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]:
    """Fused GDN prefill: conv1d -> gating -> chunk gated-delta -> gated RMSNorm
    + fp8 group-quant.

    Dispatches to the single-launch Gluon kernel when it covers the call, else
    runs the four-kernel composition. Both return the same 5-tuple.

    Args:
        projected_qkvz: ``[T, D_qkvz]`` in-projection output (packed q/k/v/z).
        projected_ba: ``[T, 2*H_v]`` in-projection output (packed b/a).
        conv_state: ``[num_lines, conv_dim, width-1]`` bf16, advanced in place.
        delta_state: ``[num_lines, H_v, K, V]`` fp32 recurrent pool; the rows
            named by ``cache_indices`` are read as the initial state and
            overwritten with the final state.
        cache_indices: ``[N]`` int32 per-sequence pool slots.
        cu_seqlens: ``[N+1]`` int32 cumulative sequence lengths (query_start_loc).
        has_initial_state: ``[N]`` bool; whether each sequence continues a prior
            conv window.
        conv_weight: ``[conv_dim, 4]`` bf16 depthwise conv weights.
        conv_bias: ``[conv_dim]`` bf16 or None.
        a_log, dt_bias: ``[H_v]`` gating parameters.
        norm_weight: ``[128]`` gated-RMSNorm weight.
        scale: query scale for the delta rule (typically ``head_k_dim ** -0.5``).
        eps: RMSNorm epsilon.
        precision_config: fp8 output dtype descriptor. Only ``float8_e4m3fn`` is
            covered by the Gluon fast path; any other dtype falls back to the
            composition (which honors it).
        return_bf16: if True, return a bf16 output in slot 0; if False, ``None``.
            The two backends produce *different* bf16: the Gluon fast path returns
            its native pre-quant gated-RMSNorm output (higher precision, emitted
            for free); the composition returns the dequant ``fp8 * scale``
            reconstruction. ``fp8_out`` / ``scales`` -- the block-fp8 ``out_proj``
            inputs -- are the same semantics on both paths.
        store_state_transposed: layout of the advanced recurrent state written to
            ``delta_state``. False (default) = aiter-native ``[N, H, K, V]`` (pairs
            with aiter's GDN decode). True = the Artemis Gluon transpose
            ``[N, H, V, K]`` (its in-place pool convention). Honored identically on
            both backends: the Gluon kernel stores its native transposed layout, so
            the fast path transposes the written rows back when False.
        backend: ``"auto"`` (default) uses the Gluon kernel when supported and
            falls back otherwise; ``"gluon"`` forces it and raises if unsupported;
            ``"compose"`` forces the composition (portability / reference / A/B).

    Returns:
        ``(bf16_out, conv_state, delta_state, fp8_out, scales)`` where
        ``fp8_out`` is ``[T, H_v*128]`` and ``scales`` is ``[T, H_v]`` fp32.
    """
    if backend not in _BACKENDS:
        raise ValueError(f"backend must be one of {_BACKENDS}, got {backend!r}")
    if projected_qkvz.dtype != torch.bfloat16:
        raise TypeError("projected_qkvz must be bfloat16.")

    if backend != "compose":
        run_fast, supported = _fast_ops()
        if run_fast is None:
            if backend == "gluon":
                raise RuntimeError(
                    "backend='gluon' requested but aiter fused GDN prefill kernel "
                    "is unimportable (needs gfx950 + Triton >= 3.8)."
                )
        else:
            ok, reason = supported(
                projected_qkvz,
                projected_ba,
                conv_state,
                delta_state,
                cache_indices,
                cu_seqlens,
                has_initial_state,
                conv_weight,
                conv_bias,
                precision_config.dtype,
            )
            if ok:
                return _run_gluon(
                    run_fast,
                    projected_qkvz,
                    projected_ba,
                    conv_state,
                    delta_state,
                    cache_indices,
                    cu_seqlens,
                    has_initial_state,
                    conv_weight,
                    conv_bias,
                    a_log,
                    dt_bias,
                    norm_weight,
                    scale=scale,
                    eps=eps,
                    return_bf16=return_bf16,
                    store_state_transposed=store_state_transposed,
                )
            if backend == "gluon":
                raise ValueError(
                    f"backend='gluon' requested but the call is not covered: {reason}"
                )
            # backend == "auto": fall through to the composition below.

    return _compose(
        projected_qkvz,
        projected_ba,
        conv_state,
        delta_state,
        cache_indices,
        cu_seqlens,
        has_initial_state,
        conv_weight,
        conv_bias,
        a_log,
        dt_bias,
        norm_weight,
        scale=scale,
        eps=eps,
        precision_config=precision_config,
        return_bf16=return_bf16,
        store_state_transposed=store_state_transposed,
    )


def _run_gluon(
    run_fast,
    projected_qkvz: torch.Tensor,
    projected_ba: torch.Tensor,
    conv_state: torch.Tensor,
    delta_state: torch.Tensor,
    cache_indices: torch.Tensor,
    cu_seqlens: torch.Tensor,
    has_initial_state: torch.Tensor,
    conv_weight: torch.Tensor,
    conv_bias: torch.Tensor,
    a_log: torch.Tensor,
    dt_bias: torch.Tensor,
    norm_weight: torch.Tensor,
    *,
    scale: float,
    eps: float,
    return_bf16: bool,
    store_state_transposed: bool,
) -> tuple[
    torch.Tensor | None, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]:
    """Fast path: one fused Gluon launch, then reconcile the state layout / bf16
    slot with this module's contract."""
    bf16_out, conv_state, delta_state, fp8_out, scales = run_fast(
        projected_qkvz,
        projected_ba,
        conv_state,
        delta_state,
        cache_indices,
        cu_seqlens,
        has_initial_state,
        conv_weight,
        conv_bias,
        a_log,
        dt_bias,
        norm_weight,
        scale=scale,
        eps=eps,
    )
    # The Gluon kernel writes recurrent state in its native [N, H, V, K] layout
    # (== store_state_transposed=True). Restore aiter-native [N, H, K, V] when the
    # caller did not ask for the transposed pool, so the flag stays a pure layout
    # choice independent of which backend ran. head_k == head_v == 128, so the rows
    # are square and the transpose is well-defined.
    if not store_state_transposed:
        idx = cache_indices.to(torch.long)
        delta_state.index_copy_(
            0, idx, delta_state.index_select(0, idx).transpose(-1, -2).contiguous()
        )
    return (bf16_out if return_bf16 else None), conv_state, delta_state, fp8_out, scales


def _compose(
    projected_qkvz: torch.Tensor,
    projected_ba: torch.Tensor,
    conv_state: torch.Tensor,
    delta_state: torch.Tensor,
    cache_indices: torch.Tensor,
    cu_seqlens: torch.Tensor,
    has_initial_state: torch.Tensor,
    conv_weight: torch.Tensor,
    conv_bias: torch.Tensor | None,
    a_log: torch.Tensor,
    dt_bias: torch.Tensor,
    norm_weight: torch.Tensor,
    *,
    scale: float,
    eps: float,
    precision_config: FP8PrecisionConfig,
    return_bf16: bool,
    store_state_transposed: bool,
) -> tuple[
    torch.Tensor | None, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]:
    """Fallback path: the four aiter pieces chained across the shared pools."""
    device = projected_qkvz.device
    seq_len = projected_qkvz.shape[0]

    # --- Infer head geometry from shapes (matches the Gluon kernel, which is
    # likewise specialized purely by signature). head_k == head_v == 128.
    num_v_heads = projected_ba.shape[1] // 2
    conv_dim = conv_weight.shape[0]
    value_dim = num_v_heads * _HEAD_DIM
    key_dim = (conv_dim - value_dim) // 2
    num_k_heads = key_dim // _HEAD_DIM
    if key_dim * 2 + value_dim != conv_dim:
        raise ValueError(
            f"conv_dim={conv_dim} is inconsistent with num_v_heads={num_v_heads} "
            f"(head_dim={_HEAD_DIM}); expected 2*key_dim + value_dim."
        )
    if num_v_heads % num_k_heads != 0:
        raise ValueError(
            f"num_v_heads={num_v_heads} must be a multiple of num_k_heads={num_k_heads}."
        )

    # --- 1. Split the packed projections into conv input + z/b/a (fused).
    mixed_qkv, z, b, a = fused_qkvzba_split_reshape_cat_prefill(
        projected_qkvz, projected_ba, num_k_heads, num_v_heads, _HEAD_DIM, _HEAD_DIM
    )

    # --- 2. Causal conv1d with silu; splits Q|K|V and advances conv_state.
    # The HIP kernel wants x laid out as [conv_dim, T].
    q, k, v = causal_conv1d_split_qkv_hip_fn(
        mixed_qkv.transpose(0, 1).contiguous(),
        conv_weight,
        conv_bias,
        conv_state,
        cu_seqlens,
        key_dim,
        value_dim,
        cache_indices=cache_indices,
        has_initial_state=has_initial_state,
        activation="silu",
    )

    # --- Reshape to [B=1, T, H, head] and expand GQA q/k up to the value heads
    # (each value head in a group shares its group's q/k). The delta-rule state
    # pool is sized for H_v, so all of q/k/v/g/beta must speak H_v.
    q = q.view(1, seq_len, num_k_heads, _HEAD_DIM)
    k = k.view(1, seq_len, num_k_heads, _HEAD_DIM)
    v = v.view(1, seq_len, num_v_heads, _HEAD_DIM)
    if num_v_heads != num_k_heads:
        groups = num_v_heads // num_k_heads
        q = q.repeat_interleave(groups, dim=2)
        k = k.repeat_interleave(groups, dim=2)

    # --- 3. Gating: g (log-space decay) and beta (sigmoid gate), per value head.
    g, beta = fused_gdn_gating_and_sigmoid(a_log, a, b, dt_bias)
    g = g.view(1, seq_len, num_v_heads)
    beta = beta.view(1, seq_len, num_v_heads)

    # --- 4. Chunked gated delta recurrence over the packed sequence. Gather the
    # per-sequence initial states, run, and scatter the final states back.
    initial_state = delta_state.index_select(0, cache_indices.to(torch.long))
    core_attn_out, final_state = chunk_gated_delta_rule(
        q.contiguous(),
        k.contiguous(),
        v.contiguous(),
        g,
        beta,
        scale=scale,
        initial_state=initial_state,
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
        cu_seqlens=cu_seqlens,
    )
    if store_state_transposed:
        final_state = final_state.transpose(-1, -2).contiguous()
    delta_state.index_copy_(
        0, cache_indices.to(torch.long), final_state.to(delta_state.dtype)
    )

    # --- 5. Gated RMSNorm + fp8 group quantization tail.
    core = core_attn_out.reshape(seq_len, num_v_heads, _HEAD_DIM)
    fp8_out = torch.empty(
        (seq_len, num_v_heads * _HEAD_DIM), dtype=precision_config.dtype, device=device
    )
    scales = torch.empty((seq_len, num_v_heads), dtype=torch.float32, device=device)
    gated_rmsnorm_fp8_group_quant(
        fp8_out, scales, core, z, norm_weight, eps, _GROUP_SIZE, False
    )

    # --- 6. Optional bf16 reconstruction of the quantized output.
    bf16_out = None
    if return_bf16:
        bf16_out = (
            (
                fp8_out.view(seq_len, num_v_heads, _HEAD_DIM).to(torch.float32)
                * scales[:, :, None]
            )
            .to(torch.bfloat16)
            .reshape(seq_len, num_v_heads * _HEAD_DIM)
        )

    return bf16_out, conv_state, delta_state, fp8_out, scales
