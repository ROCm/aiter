# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Host launch of the gfx1250 KDA gating kernel for ATOM's gating entry."""

import torch

from aiter.ops.triton._gluon_kernels.gfx1250.attention.kda_gating import (
    get_kda_gating_config,
    kda_gating_kernel,
)
from aiter.ops.triton._triton_kernels.kimi_delta_attn.fast_launch import fast_launch

_launch = fast_launch(kda_gating_kernel)
_ACT_DTYPES = (torch.bfloat16, torch.float16)


def try_kda_gating(
    *,
    A_log: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    dt_bias: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    o: torch.Tensor | None,
    scale: float | None,
    initial_state: torch.Tensor | None,
    inplace_final_state: bool,
    cu_seqlens: torch.Tensor | None,
    ssm_state_indices: torch.Tensor | None,
    num_accepted_tokens: torch.Tensor | None,
    use_qk_l2norm_in_kernel: bool,
    ssm_state_indices_in: torch.Tensor | None,
    lower_bound: float | None,
) -> torch.Tensor | None:
    """Run ATOM's KDA gating recurrence with ``kda_gating_kernel``.

    The arguments are those of ``fused_sigmoid_gating_delta_rule_update``.
    The kernel updates ``initial_state`` in place. It writes ``o``, or a new
    tensor with the shape of ``v`` when ``o`` is None.

    Limits: K = V = 128 and HV is a multiple of H. The activations are BF16 or
    FP16 with unit stride in the last dimension. The state is FP32 and V-first,
    ``[slots, HV, V, K]``, with contiguous slots. ``cu_seqlens`` and the slot
    tables are int32. The kernel needs q/k L2 normalization and
    ``inplace_final_state``. A 1D slot table has one slot for each sequence,
    so each sequence has 0 or 1 tokens. Spec verify needs a 2D slot table and
    int32 ``num_accepted_tokens``.

    Returns:
        The output tensor, or None, without a launch, when the kernel does not
        cover the inputs.
    """
    if not (
        inplace_final_state
        and use_qk_l2norm_in_kernel
        and initial_state is not None
        and cu_seqlens is not None
        and ssm_state_indices is not None
        and lower_bound is not None
    ):
        return None
    if q.ndim != 4 or q.shape[0] != 1 or v.ndim != 4:
        return None
    _, T, H, K = q.shape
    HV, V = v.shape[2], v.shape[3]
    if not (K == 128 and V == 128 and HV % H == 0):
        return None
    if not (
        k.shape == q.shape
        and v.shape[:2] == q.shape[:2]
        and a.shape == (1, T, HV, K)
        and b.shape == (1, T, HV)
        and q.dtype in _ACT_DTYPES
        and all(x.dtype == q.dtype for x in (k, v, a, b))
        and q.stride()[2:] == (K, 1)
        and k.stride()[2:] == (K, 1)
        and v.stride()[2:] == (V, 1)
        and a.stride()[2:] == (K, 1)
        and b.stride(2) == 1
    ):
        return None
    state = initial_state
    if not (
        state.dtype == torch.float32
        and state.ndim == 4
        and state.shape[1:] == (HV, V, K)
        and state.stride()[1:] == (V * K, K, 1)
    ):
        return None
    if not (
        A_log.numel() == HV
        and A_log.is_contiguous()
        and dt_bias.numel() == HV * K
        and dt_bias.is_contiguous()
        and cu_seqlens.dtype == torch.int32
        and cu_seqlens.ndim == 1
        and cu_seqlens.is_contiguous()
    ):
        return None
    write = ssm_state_indices
    read = write if ssm_state_indices_in is None else ssm_state_indices_in
    if not (
        write.dtype == torch.int32
        and read.dtype == torch.int32
        and read.shape == write.shape
        and read.stride() == write.stride()
        and write.ndim in (1, 2)
        and write.stride(-1) == 1
    ):
        return None
    multi_token = write.ndim == 2
    spec = num_accepted_tokens is not None
    if spec and not (
        multi_token
        and num_accepted_tokens.dtype == torch.int32
        and num_accepted_tokens.is_contiguous()
    ):
        return None
    if o is None:
        o = torch.empty_like(v)
        stride_o_tok = o.stride(1)
    elif o.dtype != v.dtype or o.stride()[-2:] != (V, 1):
        return None
    elif o.shape == v.shape:
        stride_o_tok = o.stride(1)
    elif o.shape == v.shape[1:]:
        stride_o_tok = o.stride(0)
    else:
        return None
    n_seq = cu_seqlens.numel() - 1
    if n_seq <= 0:
        return o
    if scale is None:
        scale = K**-0.5
    config = get_kda_gating_config(n_seq * HV, multi_token)
    num_warps = config.pop("num_warps")
    _launch[(n_seq * HV * (V // config["BV"]),)](
        q_ptr=q,
        k_ptr=k,
        v_ptr=v,
        a_ptr=a,
        b_ptr=b,
        A_log_ptr=A_log,
        dt_bias_ptr=dt_bias,
        o_ptr=o,
        state_ptr=state,
        cu_ptr=cu_seqlens,
        write_ptr=write,
        read_ptr=read,
        accepted_ptr=num_accepted_tokens if spec else cu_seqlens,
        stride_idx_seq=write.stride(0),
        stride_q_tok=q.stride(1),
        stride_k_tok=k.stride(1),
        stride_v_tok=v.stride(1),
        stride_a_tok=a.stride(1),
        stride_b_tok=b.stride(1),
        stride_o_tok=stride_o_tok,
        stride_slot=state.stride(0),
        lower_bound=float(lower_bound),
        scale=float(scale),
        H=H,
        HV=HV,
        K=K,
        V=V,
        NUM_WARPS=num_warps,
        MULTI_TOKEN=multi_token,
        IS_SPEC=spec,
        **config,
        num_warps=num_warps,
        enable_fp_fusion=True,
    )
    return o
