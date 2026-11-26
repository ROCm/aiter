# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Recurrence-level KDA: the chunk fwd/bwd behind one autograd op, with chunkwise CP.

``kda_recurrence`` is the KDA core (``o = KDA(q, k, v, g, beta)``) without the
projections, conv or output norm around it, which is what Megatron's
``kda_backend`` swaps. The local chunk kernels come from a ``KdaKernels``
pair, Triton or FlyDSL; ``cp_context`` adds chunkwise context parallelism
(``aiter.ops.kda.context_parallel``).
"""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass

import torch

from aiter.ops.kda.context_parallel import (
    KdaCPBoundary,
    all_gather_boundary_packets,
    kda_cp_boundary,
    kda_state_chain,
    pack_boundary,
)

__all__ = [
    "KdaKernels",
    "KdaOptions",
    "get_kda_kernels",
    "kda_cp_backward_finish",
    "kda_cp_backward_packet",
    "kda_cp_forward_finish",
    "kda_cp_forward_local",
    "kda_recurrence",
    "kda_transition",
]

# The backward recomputes with the default pipeline's chunk size when unset.
_BWD_DEFAULT_CHUNK_SIZE = 64


@dataclass(frozen=True)
class KdaOptions:
    """Everything besides the tensors that selects the recurrence."""

    scale: float
    use_qk_l2norm_in_kernel: bool = False
    use_gate_in_kernel: bool = False
    use_beta_sigmoid_in_kernel: bool = False
    safe_gate: bool = False
    lower_bound: float | None = None
    chunk_size: int | None = None

    def kwargs(self) -> dict:
        return {
            "scale": self.scale,
            "use_qk_l2norm_in_kernel": self.use_qk_l2norm_in_kernel,
            "use_gate_in_kernel": self.use_gate_in_kernel,
            "use_beta_sigmoid_in_kernel": self.use_beta_sigmoid_in_kernel,
            "safe_gate": self.safe_gate,
            "lower_bound": self.lower_bound,
            "chunk_size": self.chunk_size,
        }


@dataclass(frozen=True)
class KdaKernels:
    """A local (single-rank) chunk forward and backward.

    ``fwd(q, k, v, g, beta, *, A_log, dt_bias, initial_state,
    output_final_state, cu_seqlens, **KdaOptions.kwargs())`` returns
    ``(o, final_state)`` without building a graph; ``final_state`` is fp32
    ``[N, HV, K, V]``.

    ``bwd(do, dht, q, k, v, g, beta, *, A_log, dt_bias, initial_state,
    cu_seqlens, **KdaOptions.kwargs())`` returns
    ``(dq, dk, dv, dg, dbeta, dA_log, ddt_bias, dh0)``. ``dht`` and
    ``initial_state`` may be ``None``.
    """

    name: str
    fwd: Callable
    bwd: Callable


def _triton_fwd(fwd_backend: str):
    def fwd(q, k, v, g, beta, **kw):
        from aiter.ops.triton.kimi_delta_attn import chunk_kimi_delta_attn

        if fwd_backend == "auto":
            from aiter.ops.flydsl.kda_chunk_fwd import kda_fwd_flydsl_supported

            supported = kda_fwd_flydsl_supported(
                H=q.shape[2],
                HV=v.shape[2],
                K=q.shape[-1],
                V=v.shape[-1],
                chunk_size=kw["chunk_size"],
                dtype=q.dtype,
            ) and (q.dtype == k.dtype == v.dtype)
            backend = "flydsl" if supported else "triton"
        else:
            backend = fwd_backend
        with torch.no_grad():
            return chunk_kimi_delta_attn(q, k, v, g, beta, fwd_backend=backend, **kw)

    return fwd


def _triton_bwd(bwd_backend: str):
    def bwd(do, dht, q, k, v, g, beta, *, chunk_size, **kw):
        from aiter.ops.triton._triton_kernels.kimi_delta_attn import (
            chunk_delta_attn_bwd,
        )

        return chunk_delta_attn_bwd(
            do=do,
            dht=dht,
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            chunk_size=chunk_size or _BWD_DEFAULT_CHUNK_SIZE,
            backend=bwd_backend,
            **kw,
        )

    return bwd


def get_kda_kernels(backend: str | KdaKernels | None = None) -> KdaKernels:
    """Resolve ``"triton"``, ``"flydsl"`` or ``"auto"``; a ``KdaKernels`` passes through.

    ``None`` reads ``AITER_KDA_BACKEND`` (default ``"triton"``). ``"flydsl"``
    runs the FlyDSL forward (gfx950, bf16, ``K = V = 128``, chunk 64, no GVA;
    other calls raise) and the FlyDSL backward, which falls back to Triton per
    call. ``"auto"`` takes the FlyDSL forward only where it is supported.
    """
    if isinstance(backend, KdaKernels):
        return backend
    if backend is None:
        backend = os.environ.get("AITER_KDA_BACKEND", "triton")
    if backend == "triton":
        return KdaKernels("triton", _triton_fwd("triton"), _triton_bwd("triton"))
    if backend in ("flydsl", "auto"):
        return KdaKernels(backend, _triton_fwd(backend), _triton_bwd("flydsl"))
    raise ValueError(
        f"KDA backend must be 'triton', 'flydsl' or 'auto', got {backend!r}."
    )


def kda_transition(
    q, k, g, beta, A_log, dt_bias, *, value_dtype, segments, kernels, opts
):
    """``(O', M)`` of token ``segments`` (``[(start, end), ...]``), one per segment.

    Runs ``kernels.fwd`` with ``v = 0`` and ``initial_state = I`` over the
    segments packed as varlen sequences: ``M`` (``[n, HV, K, K]`` fp32) is
    each segment's final state and ``O'`` (``[1, sum(len), HV, K]``) its
    output, ``scale * M_t^T q_t`` per token.
    """
    K, HV = q.shape[-1], g.shape[2]

    def cat(x):
        return torch.cat([x[:, s:e] for s, e in segments], dim=1)

    lens = [e - s for s, e in segments]
    cu = torch.tensor([0] + lens, dtype=torch.int32).cumsum(0).to(torch.int32)
    total = cu[-1].item()
    v0 = torch.zeros(1, total, HV, K, dtype=value_dtype, device=q.device)
    h0 = (
        torch.eye(K, dtype=torch.float32, device=q.device)
        .expand(len(segments), HV, K, K)
        .contiguous()
    )
    o, m = kernels.fwd(
        cat(q).contiguous(),
        cat(k).contiguous(),
        v0,
        cat(g).contiguous(),
        cat(beta).contiguous(),
        A_log=A_log,
        dt_bias=dt_bias,
        initial_state=h0,
        output_final_state=True,
        cu_seqlens=cu.to(q.device),
        **opts.kwargs(),
    )
    return o, m.float()


def kda_cp_forward_local(
    q, k, v, g, beta, A_log, dt_bias, *, cu_seqlens, bnd: KdaCPBoundary, kernels, opts
):
    """Rank-local forward: zero-start recurrence, boundary transitions, fwd packet.

    Returns ``(o_local, ht_local, o_first, m_first, packet)``. ``o_first`` /
    ``m_first`` (``O'`` and ``M`` of the first sequence) are ``None`` unless an
    earlier rank feeds this one; ``packet`` is ``[S_ext, M]`` of the last
    sequence, zeros when no later rank reads it.
    """
    HV, K, V = v.shape[2], q.shape[-1], v.shape[-1]
    o, ht = kernels.fwd(
        q,
        k,
        v,
        g,
        beta,
        A_log=A_log,
        dt_bias=dt_bias,
        initial_state=None,
        output_final_state=True,
        cu_seqlens=cu_seqlens,
        **opts.kwargs(),
    )
    need_first, need_last = bnd.pre_num_ranks > 0, bnd.post_num_ranks > 0
    if bnd.num_seqs == 1:
        segments = [(0, bnd.num_tokens)] if need_first or need_last else []
    else:
        segments = [(0, bnd.first_end)] * need_first + [
            (bnd.last_start, bnd.num_tokens)
        ] * need_last
    o_first = m_first = m_last = None
    if segments:
        o_t, m = kda_transition(
            q,
            k,
            g,
            beta,
            A_log,
            dt_bias,
            value_dtype=v.dtype,
            segments=segments,
            kernels=kernels,
            opts=opts,
        )
        if need_first:
            o_first, m_first = o_t[:, : bnd.first_end], m[0]
        m_last = m[-1]
    if need_last:
        packet = pack_boundary(ht[-1], m_last)
    else:
        packet = torch.zeros(HV, K, V + K, dtype=torch.float32, device=q.device)
    return o, ht, o_first, m_first, packet


def kda_cp_forward_finish(o, ht, o_first, m_first, gathered, *, bnd: KdaCPBoundary):
    """Chain the earlier ranks' packets into ``S0`` and correct the first sequence.

    Returns ``(o, final_state, h0)``: ``h0`` is the true incoming state per
    local sequence (``[N, HV, K, V]`` fp32, zero but the first), or ``None``
    on a rank nothing feeds.
    """
    V = o.shape[-1]
    r, pre = bnd.rank, bnd.pre_num_ranks
    s0 = kda_state_chain(gathered[r - pre : r], V)
    if s0 is None:
        return o, ht, None
    end = bnd.first_end
    corr = torch.einsum("bthk,hkv->bthv", o_first.float(), s0)
    o = o.clone()
    o[:, :end] = (o[:, :end].float() + corr).to(o.dtype)
    ht = ht.clone()
    ht[0] += torch.matmul(m_first, s0)
    h0 = torch.zeros_like(ht)
    h0[0] = s0
    return o, ht, h0


def kda_cp_backward_packet(do, dht, o_first, m_first, *, key_dim, bnd: KdaCPBoundary):
    """``[D, M^T]`` of the first sequence: its state gradient without later ranks.

    ``D = O'^T dO + M^T dht[0]``; zeros when no earlier rank reads it.
    """
    HV, V = do.shape[2], do.shape[-1]
    if bnd.pre_num_ranks == 0:
        return torch.zeros(
            HV, key_dim, V + key_dim, dtype=torch.float32, device=do.device
        )
    end = bnd.first_end
    d = torch.einsum("bthk,bthv->hkv", o_first.float(), do[:, :end].float())
    m_t = m_first.transpose(-1, -2)
    if dht is not None:
        d = d + torch.matmul(m_t, dht[0].float())
    return pack_boundary(d, m_t)


def kda_cp_backward_finish(dht, gathered, *, num_seqs, bnd: KdaCPBoundary):
    """The final-state gradient per local sequence: ``dht`` plus the later ranks' chain."""
    V = gathered.shape[-1] - gathered.shape[-2]
    r, post = bnd.rank, bnd.post_num_ranks
    ds = kda_state_chain(gathered[r + 1 : r + 1 + post], V, reverse=True)
    if ds is None:
        return dht
    if dht is None:
        dht = ds.new_zeros(num_seqs, *ds.shape)
    else:
        dht = dht.float().clone()
    dht[-1] += ds
    return dht


class _KdaRecurrenceFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q,
        k,
        v,
        g,
        beta,
        A_log,
        dt_bias,
        initial_state,
        cu_seqlens,
        kernels,
        opts,
        bnd,
        output_final_state,
    ):
        o_first = m_first = None
        if bnd is None:
            o, ht = kernels.fwd(
                q,
                k,
                v,
                g,
                beta,
                A_log=A_log,
                dt_bias=dt_bias,
                initial_state=initial_state,
                output_final_state=output_final_state,
                cu_seqlens=cu_seqlens,
                **opts.kwargs(),
            )
            h0 = initial_state
        else:
            o, ht, o_first, m_first, packet = kda_cp_forward_local(
                q,
                k,
                v,
                g,
                beta,
                A_log,
                dt_bias,
                cu_seqlens=cu_seqlens,
                bnd=bnd,
                kernels=kernels,
                opts=opts,
            )
            gathered = all_gather_boundary_packets(packet, bnd.group)
            o, ht, h0 = kda_cp_forward_finish(
                o, ht, o_first, m_first, gathered, bnd=bnd
            )
        ctx.save_for_backward(
            q, k, v, g, beta, A_log, dt_bias, h0, cu_seqlens, o_first, m_first
        )
        ctx.kernels, ctx.opts, ctx.bnd = kernels, opts, bnd
        ctx.has_initial_state = initial_state is not None
        if not output_final_state:
            ht = None
        return o.to(q.dtype), ht

    @staticmethod
    def backward(ctx, do, dht):
        q, k, v, g, beta, A_log, dt_bias, h0, cu_seqlens, o_first, m_first = (
            ctx.saved_tensors
        )
        bnd = ctx.bnd
        do = do.contiguous()
        if dht is not None:
            dht = dht.contiguous()
        if bnd is not None:
            packet = kda_cp_backward_packet(
                do, dht, o_first, m_first, key_dim=q.shape[-1], bnd=bnd
            )
            gathered = all_gather_boundary_packets(packet, bnd.group)
            dht = kda_cp_backward_finish(
                dht,
                gathered,
                num_seqs=bnd.num_seqs,
                bnd=bnd,
            )
        dq, dk, dv, dg, dbeta, dA_log, ddt_bias, dh0 = ctx.kernels.bwd(
            do,
            dht,
            q,
            k,
            v,
            g,
            beta,
            A_log=A_log,
            dt_bias=dt_bias,
            initial_state=h0,
            cu_seqlens=cu_seqlens,
            **ctx.opts.kwargs(),
        )
        if not ctx.has_initial_state:
            dh0 = None
        elif dh0 is not None:
            dh0 = dh0.to(h0.dtype)
        return (
            dq,
            dk,
            dv,
            dg,
            dbeta,
            dA_log,
            ddt_bias,
            dh0,
            None,
            None,
            None,
            None,
            None,
        )


def kda_recurrence(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    *,
    A_log: torch.Tensor | None = None,
    dt_bias: torch.Tensor | None = None,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    use_gate_in_kernel: bool = False,
    use_beta_sigmoid_in_kernel: bool = False,
    safe_gate: bool = False,
    lower_bound: float | None = None,
    cu_seqlens: torch.Tensor | None = None,
    chunk_size: int | None = None,
    cp_context=None,
    backend: str | KdaKernels | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    r"""KDA recurrence ``o_t = scale * S_t^T q_t``, ``S_t = (I - b k k^T) Diag(e^g) S_{t-1} + b k v^T``.

    Differentiable in every float tensor input (``q``, ``k``, ``v``, ``g``,
    ``beta``, ``A_log``, ``dt_bias``, ``initial_state``) and through the
    returned final state.

    Args:
        q, k: ``[B, T, H, K]``.
        v: ``[B, T, HV, V]``; ``HV`` a multiple of ``H`` (GVA).
        g: ``[B, T, HV, K]``. Raw gate with ``use_gate_in_kernel``, else a
            natural-log decay.
        beta: ``[B, T, HV]``; logits with ``use_beta_sigmoid_in_kernel``.
        A_log: ``[HV]``, required with ``use_gate_in_kernel``.
        dt_bias: ``[HV * K]``.
        initial_state: fp32 ``[N, HV, K, V]``, one per sequence. Not with
            ``cp_context``.
        output_final_state: also return the final ``[N, HV, K, V]`` fp32
            states. Under CP these are the states at the end of the local
            interval, for the sequences this rank holds.
        safe_gate, lower_bound: the bounded gate
            ``lower_bound * sigmoid(exp(A_log) * (g + dt_bias))``.
        cu_seqlens: ``[N + 1]`` for packed sequences (``B == 1``). Ignored under
            CP, where ``cp_context.cu_seqlens`` holds the local boundaries.
        chunk_size: 32 or 64, or ``None`` for the backend's choice.
        cp_context: chunkwise context parallelism (``KdaCPContext``,
            ``build_kda_cp_context`` or FLA's ``build_cp_context``, contiguous
            layout). Every rank of the group must call, forward and backward.
        backend: see ``get_kda_kernels``.

    Returns:
        ``(o, final_state)``, ``o`` like ``v`` in ``q``'s dtype.
    """
    B, T, H, K = q.shape
    HV = v.shape[2]
    if k.shape != q.shape:
        raise ValueError(
            f"q and k must have the same shape, got {q.shape} vs {k.shape}."
        )
    if HV % H:
        raise ValueError(f"HV={HV} must be a multiple of H={H}.")
    if tuple(g.shape) != (B, T, HV, K) or tuple(beta.shape) != (B, T, HV):
        raise ValueError(
            f"g must be {(B, T, HV, K)} and beta {(B, T, HV)}, got "
            f"{tuple(g.shape)} and {tuple(beta.shape)}."
        )
    if use_gate_in_kernel and A_log is None:
        raise ValueError("`A_log` is required with `use_gate_in_kernel=True`.")
    if safe_gate and use_gate_in_kernel and lower_bound is None:
        raise ValueError("`lower_bound` is required with `safe_gate=True`.")
    kernels = get_kda_kernels(backend)
    opts = KdaOptions(
        scale=K**-0.5 if scale is None else scale,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        use_gate_in_kernel=use_gate_in_kernel,
        use_beta_sigmoid_in_kernel=use_beta_sigmoid_in_kernel,
        safe_gate=safe_gate,
        lower_bound=lower_bound,
        chunk_size=chunk_size,
    )

    bnd = None
    if cp_context is not None and getattr(cp_context, "group", None) is not None:
        if B != 1:
            raise ValueError("KDA chunkwise CP needs packed input with B == 1.")
        if initial_state is not None:
            raise NotImplementedError(
                "KDA chunkwise CP derives the incoming state from earlier ranks; "
                "`initial_state` is not supported."
            )
        bnd = kda_cp_boundary(cp_context, T)
        cu_seqlens = cp_context.cu_seqlens
    if cu_seqlens is not None:
        if B != 1:
            raise ValueError("`cu_seqlens` needs B == 1.")
        cu_seqlens = cu_seqlens.to(q.device).contiguous()
    N = B if cu_seqlens is None else cu_seqlens.numel() - 1
    if initial_state is not None and tuple(initial_state.shape) != (
        N,
        HV,
        K,
        v.shape[-1],
    ):
        raise ValueError(
            f"initial_state must be {(N, HV, K, v.shape[-1])}, got "
            f"{tuple(initial_state.shape)}."
        )

    q, k, v, g, beta = (t.contiguous() for t in (q, k, v, g, beta))
    if A_log is not None:
        A_log = A_log.contiguous()
    if dt_bias is not None:
        dt_bias = dt_bias.reshape(-1).contiguous()
    if initial_state is not None:
        initial_state = initial_state.contiguous()
    return _KdaRecurrenceFunction.apply(
        q,
        k,
        v,
        g,
        beta,
        A_log,
        dt_bias,
        initial_state,
        cu_seqlens,
        kernels,
        opts,
        bnd,
        output_final_state,
    )
