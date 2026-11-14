# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Host dispatch for the gfx1250 one-token KDA decode path."""

import torch

from aiter.ops.triton._gluon_kernels.gfx1250.attention.kda_decode import (
    kda_one_token_decode_kernel,
    load_one_token_config,
)
from aiter.ops.triton._triton_kernels.kimi_delta_attn.fast_launch import fast_launch

H = 12
K = 128
V = 128
LOWER_BOUND = -5.0
_MISSING = object()
_cached_config = _MISSING
_launch = fast_launch(kda_one_token_decode_kernel)


def _config(overrides):
    """Resolve the one-token bucket. ``None`` means the generic kernel."""
    global _cached_config
    if overrides is not None and not overrides.get("use_one_token_decode", False):
        return None
    if overrides:
        return load_one_token_config(overrides)
    if _cached_config is _MISSING:
        _cached_config = load_one_token_config(None)
    return _cached_config


def _one_token_sequences(starts, n_tokens):
    """True when every sequence length is 0 or 1.

    The bounds live on device. Reading them syncs, which is what keeps a longer
    sequence on the generic kernel instead of dropping its extra tokens. CUDA
    graph capture cannot sync, so a captured call is required to already be a
    one-token batch; the eager path still checks.
    """
    if torch.cuda.is_current_stream_capturing():
        return True
    bounds = starts.tolist()
    if bounds[0] < 0 or bounds[-1] > n_tokens:
        return False
    return all(0 <= nxt - cur <= 1 for cur, nxt in zip(bounds, bounds[1:]))


def try_kda_one_token_decode(
    q,
    k,
    v,
    g,
    beta,
    A_log,
    dt_bias,
    initial_state,
    cu_seqlens,
    ssm_state_indices,
    out,
    *,
    config,
    state_v_first,
    inplace_final_state,
    num_accepted_tokens,
    cache_state_updates,
    pad_slot_guard,
    conv_state,
    out_gate,
    use_qk_l2norm_in_kernel,
    use_gate_in_kernel,
    use_beta_sigmoid_in_kernel,
    allow_neg_eigval,
    lower_bound,
    scale,
):
    """Return output for the supported one-token decode, or None."""
    cfg = _config(config)
    if cfg is None:
        return None
    if not (
        state_v_first is not False
        and inplace_final_state
        and num_accepted_tokens is None
        and not cache_state_updates
        and not pad_slot_guard
        and conv_state is None
        and out_gate is None
        and use_qk_l2norm_in_kernel
        and use_gate_in_kernel
        and use_beta_sigmoid_in_kernel
        and not allow_neg_eigval
        and lower_bound == LOWER_BOUND
        and (scale is None or scale == K**-0.5)
    ):
        return None
    if (
        initial_state is None
        or cu_seqlens is None
        or ssm_state_indices is None
        or A_log is None
        or dt_bias is None
    ):
        return None
    if not q.is_cuda or q.ndim != 4 or q.device.index != torch.cuda.current_device():
        return None
    n = q.shape[1]
    if not (
        q.shape == (1, n, H, K)
        and k.shape == q.shape
        and v.shape == q.shape
        and g.shape == q.shape
        and beta.shape == (1, n, H)
        and initial_state.shape[1:] == (H, V, K)
        and initial_state.stride()[1:] == (V * K, K, 1)
        and initial_state.stride(0) % K == 0
        and initial_state.dtype == torch.float32
        and all(x.dtype == torch.bfloat16 for x in (q, k, v, g, beta))
        and all(x.is_contiguous() for x in (q, k, v, A_log, dt_bias, cu_seqlens))
        and g.stride()[1:] == (H * K, K, 1)
        and beta.stride(-1) == 1
        and A_log.numel() == H
        and dt_bias.numel() == H * K
        and A_log.dtype == dt_bias.dtype == torch.float32
        and cu_seqlens.dtype == ssm_state_indices.dtype == torch.int32
        and cu_seqlens.ndim == ssm_state_indices.ndim == 1
        and cu_seqlens.numel() == n + 1
        and ssm_state_indices.numel() == n
        and V % cfg["BV"] == 0
    ):
        return None
    tensors = (
        q,
        k,
        v,
        g,
        beta,
        A_log,
        dt_bias,
        out,
        initial_state,
        cu_seqlens,
        ssm_state_indices,
    )
    device = q.device
    if not all(x.device == device for x in tensors if x is not None):
        return None
    if any(x.data_ptr() % 16 for x in (q, k, v, g, dt_bias, initial_state)):
        return None
    if out is not None and not (
        out.shape == v.shape
        and out.is_contiguous()
        and out.dtype == v.dtype
        and out.data_ptr() % 16 == 0
    ):
        return None
    if not _one_token_sequences(cu_seqlens, n):
        return None
    storages = [x.untyped_storage() if x is not None else None for x in tensors]
    if out is not None:
        out_ptr = storages[7].data_ptr()
        if any(
            storage.data_ptr() == out_ptr for storage in storages[:7] + storages[8:]
        ):
            return None
    else:
        out = torch.empty_like(v)
    grid = (V // cfg["BV"]) * H * ssm_state_indices.numel()
    _launch[(grid,)](
        q_ptr=q,
        k_ptr=k,
        v_ptr=v,
        g_ptr=g,
        beta_ptr=beta,
        A_log_ptr=A_log,
        dt_bias_ptr=dt_bias,
        o_ptr=out,
        state_ptr=initial_state,
        cu_seqlens_ptr=cu_seqlens,
        state_indices_ptr=ssm_state_indices,
        stride_indices_seq=ssm_state_indices.stride(0),
        stride_g_token=g.stride(1),
        stride_beta_token=beta.stride(1),
        stride_o_token=out.stride(1),
        stride_slot_rows=initial_state.stride(0) // K,
        lower_bound=LOWER_BOUND,
        scale=K**-0.5,
        H=H,
        K=K,
        V=V,
        BV=cfg["BV"],
        CH=cfg["CH"],
        NUM_WARPS=cfg["num_warps"],
        SK=cfg["SK"],
        num_warps=cfg["num_warps"],
        enable_fp_fusion=True,
    )
    return out
