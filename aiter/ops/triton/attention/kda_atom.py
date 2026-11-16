# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""ATOM's fused sigmoid-gating delta-rule entry on the gfx1250 KDA gating kernel.

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
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Run ATOM's KDA gating recurrence, or return None.

    ``kda_gating_kernel`` runs decode, fork and speculative verify on gfx1250
    for the per-channel KDA gate ``lower_bound * sigmoid(exp(A_log) * (a +
    dt_bias))`` with ``K = V = 128``, BF16 or FP16 activations, an FP32
    V-first state, int32 slot tables and ``inplace_final_state``
    (``try_kda_gating`` lists all the limits). Any other input, including
    another architecture and ATOM's softplus gate, returns None without a
    launch, and the caller runs its own kernel.

    ``o`` may be ``[T, HV, V]`` or ``v``'s shape and is returned as given. With
    ``o is None`` the returned activation has ``v``'s shape.

    A read slot below 0 leaves that sequence's output and state untouched. A
    write slot below 0 still writes the output. Slot 0 is a real slot.

    Returns:
        ``(o, initial_state)`` with the state updated in place, or None.
    """
    del beta, threshold  # softplus-only; the lower-bound gate does not read them
    if get_arch() not in ("gfx1250", "gfx1250-strict") or not is_kda:
        return None

    from aiter.ops.triton.attention.kda_gating import try_kda_gating

    out = try_kda_gating(
        A_log=A_log,
        a=a,
        b=b,
        dt_bias=dt_bias,
        q=q,
        k=k,
        v=v,
        o=o,
        scale=scale,
        initial_state=initial_state,
        inplace_final_state=inplace_final_state,
        cu_seqlens=cu_seqlens,
        ssm_state_indices=ssm_state_indices,
        num_accepted_tokens=num_accepted_tokens,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        ssm_state_indices_in=ssm_state_indices_in,
        lower_bound=lower_bound,
    )
    if out is None:
        return None
    return out, initial_state
