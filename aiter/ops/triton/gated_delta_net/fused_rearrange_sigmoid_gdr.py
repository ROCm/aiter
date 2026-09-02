# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Songlin Yang, Yu Zhang
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
#
# Adapted from flash-linear-attention / vLLM (see _triton_kernels copy).

from __future__ import annotations

import functools
import os

import torch
import triton

from aiter.ops.triton._triton_kernels.gated_delta_rule.decode.fused_rearrange_sigmoid_gdr import (
    fused_rearrange_sigmoid_gated_delta_rule_update_kernel,
)


@functools.lru_cache(maxsize=1)
def _flydsl_gdr_available() -> bool:
    # Probes the module the fast path imports below: `aiter.ops.flydsl` raises
    # on a flydsl that is missing or too old, so a bad install is a False here
    # rather than an exception out of a Triton entry point.
    try:
        from aiter.ops.flydsl import linear_attention_kernels  # noqa: F401

        return True
    except (ImportError, OSError, RuntimeError):
        return False


def _flydsl_gdr_enabled() -> bool:
    """Opt-in gate for the FlyDSL gated-delta-rule MTP port.

    Off by default, so Triton keeps serving every call. Not memoized, so the env
    var can be toggled at runtime; only the availability probe is cached.
    """
    return os.environ.get("AITER_GDR_FLYDSL", "") == "1" and _flydsl_gdr_available()


def _uniform_draft_window(cu_seqlens: torch.Tensor | None, total_tokens: int) -> int:
    """Draft length if every sequence has the same one, else -1.

    The FlyDSL kernel bakes the window into the kernel it builds, so it can only
    take a packed batch that unflattens to ``[N, T, ...]``. A ragged batch is a
    different kernel per row, so it stays on Triton.
    """
    if cu_seqlens is None:
        return -1
    # One device-to-host copy; the window has to be known on the host to pick
    # the kernel, so the comparisons are done here rather than one .item() each.
    lens = (cu_seqlens[1:] - cu_seqlens[:-1]).to("cpu")
    if lens.numel() == 0:
        return -1
    first = int(lens[0])
    if first <= 0 or int(lens.min()) != first or int(lens.max()) != first:
        return -1
    if first * lens.numel() != total_tokens:
        return -1
    return first


def _try_flydsl_mtp(
    *,
    A_log,
    a,
    b,
    dt_bias,
    qkv,
    key_dim,
    value_dim,
    head_k_dim,
    head_v_dim,
    softplus_beta,
    softplus_threshold,
    scale,
    initial_state,
    inplace_final_state,
    cu_seqlens,
    ssm_state_indices,
    num_accepted_tokens,
    use_qk_l2norm_in_kernel,
    is_kda,
    core_attn_out,
):
    """Route a speculative-verify call to the FlyDSL chain kernel, or decline.

    Returns ``(out, final_state)`` when it took the call and ``None`` when the
    caller should fall through to Triton. Declining is the common case: this
    covers the MTP verify shape and nothing else.
    """
    if is_kda or not inplace_final_state or initial_state is None:
        return None
    if num_accepted_tokens is None or ssm_state_indices is None:
        return None
    if ssm_state_indices.ndim != 2:
        return None
    if qkv.ndim != 2 or qkv.stride(1) != 1:
        return None
    # The gating constants are compiled into the kernel, so only the default
    # pair is routed.
    if float(softplus_beta) != 1.0 or float(softplus_threshold) != 20.0:
        return None
    if scale is not None and abs(float(scale) - head_k_dim**-0.5) > 1e-12:
        return None

    total_tokens = qkv.shape[0]
    window = _uniform_draft_window(cu_seqlens, total_tokens)
    if window <= 0:
        return None
    n_seq = total_tokens // window

    H = key_dim // head_k_dim
    HV = value_dim // head_v_dim
    stride_qkv_l = qkv.stride(0)
    base = qkv.storage_offset()

    # q / k / v are strided views into the packed projection rather than copies;
    # the kernel takes their strides as build parameters.
    q = qkv.as_strided(
        (n_seq, window, H, head_k_dim),
        (window * stride_qkv_l, stride_qkv_l, head_k_dim, 1),
        base,
    )
    k = qkv.as_strided(
        (n_seq, window, H, head_k_dim),
        (window * stride_qkv_l, stride_qkv_l, head_k_dim, 1),
        base + key_dim,
    )
    v = qkv.as_strided(
        (n_seq, window, HV, head_v_dim),
        (window * stride_qkv_l, stride_qkv_l, head_v_dim, 1),
        base + 2 * key_dim,
    )

    if a.ndim != 2 or b.ndim != 2 or a.stride(1) != 1 or b.stride(1) != 1:
        return None
    a_view = a.as_strided(
        (n_seq, window, HV), (window * a.stride(0), a.stride(0), 1), a.storage_offset()
    )
    b_view = b.as_strided(
        (n_seq, window, HV), (window * b.stride(0), b.stride(0), 1), b.storage_offset()
    )

    from aiter.ops.flydsl.linear_attention_kernels import (
        _flydsl_gdr_mtp_supported,
        flydsl_gdr_mtp,
    )

    idx = ssm_state_indices.to(torch.int32)
    nacc = num_accepted_tokens.to(torch.int32)
    if not _flydsl_gdr_mtp_supported(q, k, v, initial_state, idx, nacc):
        return None
    if a_view.dtype != qkv.dtype or b_view.dtype != qkv.dtype:
        return None
    if dt_bias.dtype != qkv.dtype:
        return None

    out = (
        core_attn_out[: total_tokens * HV * head_v_dim].view(
            n_seq, window, HV, head_v_dim
        )
        if core_attn_out is not None
        else qkv.new_empty(n_seq, window, HV, head_v_dim)
    )
    flydsl_gdr_mtp(
        query=q,
        key=k,
        value=v,
        a=a_view,
        b=b_view,
        dt_bias=dt_bias,
        A_log=A_log,
        state=initial_state,
        out=out,
        ssm_state_indices=idx,
        num_accepted_tokens=nacc,
        use_qk_l2norm=use_qk_l2norm_in_kernel,
    )
    # Same rank as the Triton path below, which returns [1, T, HV, V].
    return out.view(1, total_tokens, HV, head_v_dim), initial_state


def fused_rearrange_sigmoid_gated_delta_rule(
    A_log: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    dt_bias: torch.Tensor,
    qkv: torch.Tensor,
    key_dim: int,
    value_dim: int,
    head_k_dim: int,
    head_v_dim: int,
    beta: float = 1.0,
    threshold: float = 20.0,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    inplace_final_state: bool = True,
    cu_seqlens: torch.LongTensor | None = None,
    ssm_state_indices: torch.Tensor | None = None,
    num_accepted_tokens: torch.Tensor | None = None,
    use_qk_l2norm_in_kernel: bool = False,
    is_kda: bool = False,
    core_attn_out: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Fused Triton sigmoid-gated delta rule over packed QKV (decode-oriented).
    """
    expected_shape = (qkv.shape[0], key_dim * 2 + value_dim)
    assert (
        qkv.shape == expected_shape
    ), f"expect qkv to be in shape {expected_shape}, got {qkv.shape}"

    # FlyDSL port (opt-in). Only the speculative-verify shape is routed;
    # everything else falls through to Triton below unchanged.
    if _flydsl_gdr_enabled():
        routed = _try_flydsl_mtp(
            A_log=A_log,
            a=a,
            b=b,
            dt_bias=dt_bias,
            qkv=qkv,
            key_dim=key_dim,
            value_dim=value_dim,
            head_k_dim=head_k_dim,
            head_v_dim=head_v_dim,
            softplus_beta=beta,
            softplus_threshold=threshold,
            scale=scale,
            initial_state=initial_state,
            inplace_final_state=inplace_final_state,
            cu_seqlens=cu_seqlens,
            ssm_state_indices=ssm_state_indices,
            num_accepted_tokens=num_accepted_tokens,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            is_kda=is_kda,
            core_attn_out=core_attn_out,
        )
        if routed is not None:
            return routed

    if scale is None:
        scale = head_k_dim**-0.5
    else:
        assert scale > 0, "scale must be positive"

    B = 1
    T = qkv.shape[0]
    H = key_dim // head_k_dim
    HV = value_dim // head_v_dim
    K = head_k_dim
    V = head_v_dim
    N = B if cu_seqlens is None else len(cu_seqlens) - 1

    BK, BV = triton.next_power_of_2(K), min(triton.next_power_of_2(V), 32)
    NK, NV = triton.cdiv(K, BK), triton.cdiv(V, BV)
    assert NK == 1, "NK > 1 is not supported yet"
    num_stages = 3
    num_warps = 4

    if inplace_final_state and ssm_state_indices is None:
        raise ValueError(
            "ssm_state_indices is required when inplace_final_state=True "
            "(kernel indexes final state slots per token)."
        )

    o = (
        core_attn_out[: NK * B * T * HV * V].view(NK, B, T, HV, V)
        if core_attn_out is not None
        else qkv.new_empty(NK, B, T, HV, V)
    )
    if inplace_final_state:
        if initial_state is None:
            raise ValueError("initial_state is required when inplace_final_state=True")
        final_state = initial_state
    else:
        st_dtype = initial_state.dtype if initial_state is not None else qkv.dtype
        final_state = qkv.new_empty(T, HV, V, K, dtype=st_dtype)

    stride_init_state_token = (
        int(initial_state.stride(0)) if initial_state is not None else 0
    )
    stride_final_state_token = int(final_state.stride(0))

    if ssm_state_indices is None:
        stride_indices_seq, stride_indices_tok = 1, 1
    elif ssm_state_indices.ndim == 1:
        stride_indices_seq, stride_indices_tok = ssm_state_indices.stride(0), 1
    else:
        stride_indices_seq, stride_indices_tok = ssm_state_indices.stride()

    stride_qkv_l, stride_qkv_hd = qkv.stride()

    grid = (NK, NV, N * HV)
    fused_rearrange_sigmoid_gated_delta_rule_update_kernel[grid](
        A_log=A_log,
        a=a.contiguous(),
        b=b.contiguous(),
        dt_bias=dt_bias,
        beta=beta,
        threshold=threshold,
        qkv=qkv,
        o=o,
        h0=initial_state,
        ht=final_state,
        cu_seqlens=cu_seqlens,
        ssm_state_indices=ssm_state_indices,
        num_accepted_tokens=num_accepted_tokens,
        scale=scale,
        N=N,
        T=T,
        B=B,
        H=H,
        HV=HV,
        K=K,
        V=V,
        BK=BK,
        BV=BV,
        stride_qkv_l=stride_qkv_l,
        stride_qkv_hd=stride_qkv_hd,
        stride_init_state_token=stride_init_state_token,
        stride_final_state_token=stride_final_state_token,
        stride_indices_seq=stride_indices_seq,
        stride_indices_tok=stride_indices_tok,
        INPLACE_FINAL_STATE=inplace_final_state,
        USE_QK_L2NORM_IN_KERNEL=use_qk_l2norm_in_kernel,
        IS_KDA=is_kda,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    o = o.squeeze(0)
    return o, final_state
