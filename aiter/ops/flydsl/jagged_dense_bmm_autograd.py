# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""End-to-end autograd op for jagged_dense_bmm_broadcast_add (jdbba).

Pairs the FlyDSL forward (``jagged_dense_bmm_dispatched``) with the backward
(``jagged_dense_bmm_bwd_dispatched``) as a single ``torch.autograd.Function`` so
the op trains through ``out.backward()`` (the real HSTU path), matching the
recsys harness's validated ``flydsl_autograd`` provider.

Per group b over its packed row slice [s, e):
    Out[s:e] = Jagged[s:e] @ Dense[b] + Bias[b][None, :]     (forward)
    dJagged / dDense / dBias                                  (backward)

LAYOUT RECONCILIATION (the one place fwd↔bwd layouts are bridged):
  The caller holds ``dense`` in the natural ``(n_groups, K, N)`` weight layout.
    * forward kernel wants a TALL, pre-transposed ``(n_groups*N, K)`` buffer
      (N-major per group) + a flat ``(n_groups*N,)`` bias -> built here per call;
    * backward wrapper wants the PLAIN ``(n_groups*K, N)`` K-major dense and
      returns ``dDense`` already shaped ``(n_groups, K, N)`` -> it takes the
      natural ``dense`` directly, so no transpose is needed on the backward side.
  So this Function saves the natural ``(n_groups, K, N)`` dense for backward and
  only pre-transposes for the forward launch. Gradients come back in the caller's
  input layouts: ``dJagged (L, K)``, ``dDense (n_groups, K, N)``, ``dBias (n_groups, N)``.

CONSTRAINTS (inherited from the backward, Phase 0/1):
  * D = K = N is a compile-time constant pinned per process (single-D-per-process).
  * D=512 is currently blocked at large L by the FORWARD's int32 offset overflow
    (L≈7.9M) -- a separate forward-kernel fix (see the integration plan Risks).
    The backward already guards that offset; only the forward faults. So this
    autograd op is validated at D=256 today; D=512 training awaits the forward fix.
"""

from __future__ import annotations

from typing import Optional

import torch

from .jagged_dense_bmm_bwd_dispatch import jagged_dense_bmm_bwd_dispatched
from .jagged_dense_bmm_dispatch_v2 import jagged_dense_bmm_dispatched
from .kernels.jagged_dense_bmm_gen import BLOCK_M as _FWD_BLOCK_M

__all__ = ["jagged_dense_bmm_autograd"]


class _JaggedDenseBmmBroadcastAdd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, jagged, dense, bias, seq_offsets, n_groups, max_seq_len,
                stream, uniform_seqlen):
        import flydsl.compiler as flyc

        n_groups = int(n_groups)
        max_seq_len = int(max_seq_len)
        _, K, N = dense.shape
        L = jagged.shape[0]
        device = jagged.device

        # The Function drives autograd itself, so all kernel I/O runs on DETACHED
        # tensors (FlyDSL's from_dlpack rejects grad-requiring tensors). Detached
        # copies are also what we save for backward (we only need the values).
        jagged_k = jagged.detach()
        dense_k = dense.detach()
        bias_k = bias.detach()

        # fwd contract: tall pre-transposed (n_groups*N, K) dense + flat bias.
        dense_tall = dense_k.transpose(1, 2).reshape(n_groups * N, K).contiguous()
        bias_flat = bias_k.reshape(n_groups * N).contiguous()

        # Output padded by BLOCK_M so a partial tail-tile store stays in-bounds;
        # every returned row [0, L) is written by the kernel, so empty is safe.
        out = torch.empty(L + _FWD_BLOCK_M, N, dtype=torch.bfloat16, device=device)
        tC = flyc.from_dlpack(out).mark_layout_dynamic(leading_dim=1, divisibility=8)
        tA = flyc.from_dlpack(jagged_k).mark_layout_dynamic(leading_dim=1, divisibility=8)
        jagged_dense_bmm_dispatched(
            tC, tA, dense_tall, bias_flat, seq_offsets, n_groups, max_seq_len,
            stream=stream, uniform_seqlen=uniform_seqlen,
        )

        ctx.save_for_backward(jagged_k, dense_k, seq_offsets)
        ctx.n_groups = n_groups
        ctx.max_seq_len = max_seq_len
        ctx.stream = stream
        return out[:L]

    @staticmethod
    def backward(ctx, grad_out):
        jagged, dense, seq_offsets = ctx.saved_tensors  # already detached in forward
        # The backward kernels consume a contiguous bf16 (L, N) upstream grad; it is
        # a grad tensor (no grad tracking) but may be non-contiguous / non-bf16.
        grad_out = grad_out.detach()
        if grad_out.dtype != torch.bfloat16:
            grad_out = grad_out.to(torch.bfloat16)
        grad_out = grad_out.contiguous()

        d_jagged, d_dense, d_bias = jagged_dense_bmm_bwd_dispatched(
            jagged, dense, grad_out, seq_offsets,
            n_groups=ctx.n_groups, max_seq_len=ctx.max_seq_len, stream=ctx.stream,
        )
        # Grads for (jagged, dense, bias, seq_offsets, n_groups, max_seq_len,
        # stream, uniform_seqlen). Only the first three are differentiable.
        return d_jagged, d_dense, d_bias, None, None, None, None, None


def jagged_dense_bmm_autograd(
    jagged: torch.Tensor,        # (L, K)           bf16, packed rows (requires_grad ok)
    dense: torch.Tensor,         # (n_groups, K, N) bf16 weight (requires_grad ok)
    bias: torch.Tensor,          # (n_groups, N)    bf16 bias   (requires_grad ok)
    seq_offsets: torch.Tensor,   # (n_groups + 1,)  int32 prefix-sum offsets
    n_groups: Optional[int] = None,
    max_seq_len: Optional[int] = None,
    stream=None,
    uniform_seqlen: bool = True,
):
    """Autograd-enabled jdbba: returns ``Out (L, N)`` with a fused backward.

    ``Out[s:e] = Jagged[s:e] @ Dense[b] + Bias[b]`` per group b. Backpropagates to
    ``jagged`` / ``dense`` / ``bias`` via the FlyDSL backward. ``max_seq_len`` sizes
    the forward/backward M grid envelope; if omitted it is derived from
    ``seq_offsets`` (one device→host sync — pass it to avoid that). ``uniform_seqlen``
    routes the forward's XCD-remap/kernel-variant choice (scheduling only, not
    correctness); pass ``False`` for skewed/genrec deployment sequence lengths.

    D = K = N is pinned per process (single-D-per-process). D=512 at large L is
    blocked by the forward int32-overflow (see module docstring); use D=256 today.
    """
    if n_groups is None:
        n_groups = dense.shape[0]
    n_groups = int(n_groups)

    if dense.ndim != 3 or dense.shape[1] != dense.shape[2]:
        raise ValueError(
            f"dense must be (n_groups, D, D) with a square dense dim; got {tuple(dense.shape)}."
        )
    if dense.shape[0] != n_groups:
        raise ValueError(f"dense.shape[0]={dense.shape[0]} != n_groups={n_groups}.")
    if bias.shape != (n_groups, dense.shape[2]):
        raise ValueError(
            f"bias must be (n_groups, N)=({n_groups}, {dense.shape[2]}); got {tuple(bias.shape)}."
        )
    if seq_offsets.numel() != n_groups + 1:
        raise ValueError(
            f"seq_offsets must have n_groups+1={n_groups + 1} entries, got {seq_offsets.numel()}."
        )
    if seq_offsets.dtype != torch.int32:
        raise ValueError(f"seq_offsets must be int32, got {seq_offsets.dtype}.")

    if max_seq_len is None:
        max_seq_len = int((seq_offsets[1:] - seq_offsets[:-1]).max().item())

    return _JaggedDenseBmmBroadcastAdd.apply(
        jagged, dense, bias, seq_offsets, n_groups, int(max_seq_len), stream, uniform_seqlen,
    )
