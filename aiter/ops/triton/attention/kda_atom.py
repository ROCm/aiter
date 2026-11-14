# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""ATOM's fused sigmoid-gating delta-rule entry on the gfx1250 KDA kernels.

The parameter list matches ``fused_sigmoid_gating_delta_rule_update`` in ATOM:
``a`` is the gate input and ``b`` is the beta logit. The scalar ``beta`` and
``threshold`` belong to ATOM's softplus gate and are not read when
``lower_bound`` is set.

A distinct ``ssm_state_indices_in`` reads the initial state from those slots
and writes each token into ``ssm_state_indices``. Ordinary decode passes two
buffers even when they name the same slots, so the kernel always receives both
pointers. Comparing the tensors on device would synchronize a captured graph.
"""

import torch

from aiter.ops.triton.utils._triton.arch_info import get_arch


def fused_sigmoid_gating_delta_rule_update(
    A_log: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    dt_bias: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    o: torch.Tensor | None = None,
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
    ssm_state_indices_in: torch.Tensor | None = None,
    lower_bound: float | None = None,
):
    """Run ATOM's KDA gating recurrence.

    ``o`` may be ``[T, HV, V]`` or ``v``'s shape. A ``[T, HV, V]`` buffer is
    unsqueezed for the kernel and squeezed on return. With ``o is None`` the
    returned activation is ``[1, T, HV, V]`` when ``v`` is.

    One token per sequence on gfx1250, with ``H = HV = 12``, ``K = V = 128``,
    BF16 activations, FP32 V-first state and ``lower_bound=-5``, uses
    ``kda_one_token_decode_kernel``. Speculative verify and every other
    supported shape use the packed kernel. Both paths load
    ``ssm_state_indices_in`` and store ``ssm_state_indices``.

    A read slot below 0 leaves that sequence's output and state untouched. A
    write slot below 0 still writes the output. Slot 0 is a real slot.
    """
    del beta, threshold  # softplus-only; the lower-bound gate does not read them
    if get_arch() not in ("gfx1250", "gfx1250-strict"):
        raise NotImplementedError(
            "fused_sigmoid_gating_delta_rule_update requires gfx1250"
        )
    if not is_kda or lower_bound is None:
        raise NotImplementedError(
            "fused_sigmoid_gating_delta_rule_update implements the per-channel "
            "KDA gate `lower_bound * sigmoid(exp(A_log) * (a + dt_bias))`"
        )

    from aiter.ops.triton.attention.kda import fused_recurrent_kda

    squeeze = False
    out = o
    if out is not None:
        if out.shape == v.shape[1:] and out.ndim + 1 == v.ndim:
            out = out.unsqueeze(0)
            squeeze = True
        if out.shape != v.shape or not out.is_contiguous():
            raise ValueError(
                "o must be contiguous and either v's shape or v without the "
                f"batch dimension, got {tuple(out.shape)} for v {tuple(v.shape)}"
            )

    result, final_state = fused_recurrent_kda(
        q.contiguous(),
        k.contiguous(),
        v.contiguous(),
        a.contiguous(),
        b.contiguous(),
        A_log=A_log,
        dt_bias=dt_bias,
        initial_state=initial_state,
        scale=scale,
        output_final_state=True,
        inplace_final_state=inplace_final_state,
        state_v_first=True,
        cu_seqlens=cu_seqlens,
        ssm_state_indices=ssm_state_indices,
        ssm_state_indices_in=ssm_state_indices_in,
        num_accepted_tokens=num_accepted_tokens,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        use_gate_in_kernel=True,
        use_beta_sigmoid_in_kernel=True,
        lower_bound=lower_bound,
        out=out,
        atom_slots=True,
    )
    if squeeze:
        result = result.squeeze(0)
    return result, final_state
