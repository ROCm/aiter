# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""End-to-end FlyDSL Linear Attention Prefill APIs (gated delta rule).

This module hosts:

* ``chunk_gated_delta_rule_fwd_h_flydsl`` -- host wrapper around the K5
  hidden-state recurrence FlyDSL kernel (``compile_chunk_gated_delta_h``),
  including PyTorch tensor preparation, BV autotune, kernel cache, and stream
  handling. The kernel-compile module ``kernels.chunk_gated_delta_h`` is kept
  ``torch``-free, mirroring the layering used by ``kernels.gdr_decode``.

* ``flydsl_gdr_prefill`` -- a drop-in replacement for
  ``aiter.ops.triton.gated_delta_net.chunk_gated_delta_rule_opt_vk`` where
  the K5 hidden-state recurrence runs on FlyDSL and the rest of the chunk
  pipeline (K1+K2 fused cumsum/dot-kkt, K3+K4 fused solve-tril/recompute-w-u,
  K6 output) re-uses the existing Triton implementations.
"""

from __future__ import annotations

import torch
import triton

from .kernels.chunk_gated_delta_h import compile_chunk_gated_delta_h

from aiter.ops.triton._triton_kernels.gated_delta_rule.prefill.chunk_o import (
    chunk_fwd_o_opt_vk,
)
from aiter.ops.triton._triton_kernels.gated_delta_rule.prefill.fused_cumsum_kkt import (
    fused_chunk_local_cumsum_scaled_dot_kkt_fwd,
)
from aiter.ops.triton._triton_kernels.gated_delta_rule.prefill.fused_solve_tril_recompute import (
    fused_solve_tril_recompute_w_u,
)
from aiter.ops.triton._triton_kernels.gated_delta_rule.utils.l2norm import (
    l2norm_fwd,
)

__all__ = [
    "chunk_gated_delta_rule_fwd_h_flydsl",
    "flydsl_gdr_prefill",
]


# -- K5 host wrapper (FlyDSL kernel + autotune + tensor prep) -------------

_compiled_kernels = {}
_autotune_cache = {}  # (shape_key) -> best BV
_BV_CANDIDATES = [16, 32, 64]
_AUTOTUNE_WARMUP = 5
_AUTOTUNE_ITERS = 25


def _get_or_compile(
    K,
    V,
    BT,
    BV,
    H,
    Hg,
    use_g,
    use_gk,
    use_h0,
    store_fs,
    save_vn,
    is_varlen,
    wu_contig,
):
    cache_key = (
        K,
        V,
        BT,
        BV,
        H,
        Hg,
        use_g,
        use_gk,
        use_h0,
        store_fs,
        save_vn,
        is_varlen,
        wu_contig,
    )
    if cache_key not in _compiled_kernels:
        _compiled_kernels[cache_key] = compile_chunk_gated_delta_h(
            K=K,
            V=V,
            BT=BT,
            BV=BV,
            H=H,
            Hg=Hg,
            USE_G=use_g,
            USE_GK=use_gk,
            USE_INITIAL_STATE=use_h0,
            STORE_FINAL_STATE=store_fs,
            SAVE_NEW_VALUE=save_vn,
            IS_VARLEN=is_varlen,
            WU_CONTIGUOUS=wu_contig,
        )
    return _compiled_kernels[cache_key]


def _launch_kernel(
    launch_fn,
    BV,
    V,
    N,
    H,
    k,
    u,
    w,
    vn_arg,
    g_arg,
    gk_arg,
    h,
    h0_arg,
    ht_arg,
    cu_arg,
    co_arg,
    T,
    T_flat,
    stream,
):
    grid_v = triton.cdiv(V, BV)
    grid_nh = N * H
    launch_fn(
        k,
        u,
        w,
        vn_arg,
        g_arg,
        gk_arg,
        h,
        h0_arg,
        ht_arg,
        cu_arg,
        co_arg,
        T,
        T_flat,
        N,
        grid_v,
        grid_nh,
        stream,
    )


def chunk_gated_delta_rule_fwd_h_flydsl(
    k: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    g: torch.Tensor | None = None,
    gk: torch.Tensor | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    chunk_size: int = 64,
    save_new_value: bool = True,
    cu_seqlens: torch.LongTensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """FlyDSL K5 host wrapper.

    Signature is API-compatible with
    ``aiter.ops.triton._triton_kernels.gated_delta_rule.prefill.chunk_delta_h.chunk_gated_delta_rule_fwd_h_opt_vk``:

    Args:
        k: [B, T, Hg, K] bf16.
        w: [B, H, T_flat, K] bf16, head-major contiguous layout.
        u: [B, H, T_flat, V] bf16, head-major contiguous layout.
        g: [T_total, H] f32 cumulative gate, or None.
        gk: [T_total, H, K] f32 per-K cumulative gate, or None.
        initial_state: [N, H, V, K] f32, or None.
        output_final_state: whether to return the final hidden state.
        chunk_size: chunk size BT (default 64).
        save_new_value: whether to materialize ``v_new``.
        cu_seqlens: [N+1] LongTensor for variable-length batching, or None.

    Returns:
        (h, v_new, final_state) in VK-ordered layout (``[..., V, K]`` on the
        last two dims).

    BV-tile selection is internal; results of the very first call for a given
    shape are cached in module-level ``_autotune_cache``.
    """
    # Layout is fixed to head-major contiguous (matches Triton VK wrapper).
    wu_contiguous = True
    BV = 0  # 0 => autotune (cache hit on subsequent calls)

    B, T, Hg, K = k.shape
    BT = chunk_size

    H = w.shape[1]
    V = u.shape[-1]
    T_flat = w.shape[2]

    if cu_seqlens is None:
        N, NT, chunk_offsets = B, triton.cdiv(T, BT), None
    else:
        N = len(cu_seqlens) - 1
        lens = cu_seqlens[1:] - cu_seqlens[:-1]
        NT = sum(triton.cdiv(int(seq_len), BT) for seq_len in lens.tolist())
        chunk_offsets = (
            torch.cat(
                [
                    cu_seqlens.new_tensor([0]),
                    triton.cdiv(lens, BT),
                ]
            )
            .cumsum(-1)
            .to(torch.int32)
        )

    assert K <= 256

    h = k.new_empty(B, NT, H, V, K)
    final_state = (
        k.new_empty(N, H, V, K, dtype=torch.float32) if output_final_state else None
    )
    v_new_buf = k.new_empty(B, H, T_flat, V, dtype=u.dtype)
    v_new = v_new_buf if save_new_value else None

    dummy = torch.empty(1, device=k.device, dtype=torch.float32)
    g_arg = g if g is not None else dummy
    gk_arg = gk if gk is not None else dummy
    h0_arg = initial_state if initial_state is not None else dummy
    ht_arg = final_state if final_state is not None else dummy
    vn_arg = v_new_buf
    cu_arg = (
        cu_seqlens.to(torch.int32) if cu_seqlens is not None else dummy.to(torch.int32)
    )
    co_arg = chunk_offsets if chunk_offsets is not None else dummy.to(torch.int32)
    stream = torch.cuda.current_stream()

    use_g = g is not None
    use_gk = gk is not None
    use_h0 = initial_state is not None
    is_varlen = cu_seqlens is not None

    # Resolve BV: explicit > autotune cache > benchmark
    if BV <= 0:
        shape_key = (
            K,
            V,
            BT,
            H,
            Hg,
            T_flat,
            N,
            use_g,
            use_gk,
            use_h0,
            output_final_state,
            save_new_value,
            is_varlen,
            wu_contiguous,
        )

        if shape_key in _autotune_cache:
            BV = _autotune_cache[shape_key]
        else:
            candidates = [bv for bv in _BV_CANDIDATES if bv <= V and V % bv == 0]
            if len(candidates) <= 1:
                BV = candidates[0] if candidates else 16
            else:
                print(f"[K5 autotune] benchmarking BV in {candidates} ...")
                best_bv, best_us = candidates[0], float("inf")
                for bv in candidates:
                    fn = _get_or_compile(
                        K,
                        V,
                        BT,
                        bv,
                        H,
                        Hg,
                        use_g,
                        use_gk,
                        use_h0,
                        output_final_state,
                        save_new_value,
                        is_varlen,
                        wu_contiguous,
                    )
                    for _ in range(_AUTOTUNE_WARMUP):
                        _launch_kernel(
                            fn,
                            bv,
                            V,
                            N,
                            H,
                            k,
                            u,
                            w,
                            vn_arg,
                            g_arg,
                            gk_arg,
                            h,
                            h0_arg,
                            ht_arg,
                            cu_arg,
                            co_arg,
                            T,
                            T_flat,
                            stream,
                        )
                    torch.cuda.synchronize()
                    s = torch.cuda.Event(enable_timing=True)
                    e = torch.cuda.Event(enable_timing=True)
                    s.record()
                    for _ in range(_AUTOTUNE_ITERS):
                        _launch_kernel(
                            fn,
                            bv,
                            V,
                            N,
                            H,
                            k,
                            u,
                            w,
                            vn_arg,
                            g_arg,
                            gk_arg,
                            h,
                            h0_arg,
                            ht_arg,
                            cu_arg,
                            co_arg,
                            T,
                            T_flat,
                            stream,
                        )
                    e.record()
                    torch.cuda.synchronize()
                    us = s.elapsed_time(e) / _AUTOTUNE_ITERS * 1000
                    print(f"  BV={bv:3d}: {us:.2f} us")
                    if us < best_us:
                        best_us = us
                        best_bv = bv
                BV = best_bv
                print(f"[K5 autotune] best BV={BV} ({best_us:.2f} us)")
            _autotune_cache[shape_key] = BV

    launch_fn = _get_or_compile(
        K,
        V,
        BT,
        BV,
        H,
        Hg,
        use_g,
        use_gk,
        use_h0,
        output_final_state,
        save_new_value,
        is_varlen,
        wu_contiguous,
    )
    _launch_kernel(
        launch_fn,
        BV,
        V,
        N,
        H,
        k,
        u,
        w,
        vn_arg,
        g_arg,
        gk_arg,
        h,
        h0_arg,
        ht_arg,
        cu_arg,
        co_arg,
        T,
        T_flat,
        stream,
    )

    return h, v_new, final_state


# -- End-to-end Linear Attention Prefill (FlyDSL K5 + Triton K1-K4, K6) ----


def flydsl_gdr_prefill(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """End-to-end GDN forward where K5 runs on FlyDSL.

    Signature is identical to
    ``aiter.ops.triton.gated_delta_net.chunk_gated_delta_rule_opt_vk`` so that
    the two can be used interchangeably as drop-in backends.

    Pipeline (matches ``chunk_gated_delta_rule_fwd_opt_vk``):

      * K1+K2 fused : ``fused_chunk_local_cumsum_scaled_dot_kkt_fwd``  (Triton)
      * K3+K4 fused : ``fused_solve_tril_recompute_w_u``               (Triton)
      * **K5**      : ``chunk_gated_delta_rule_fwd_h_flydsl``          (FlyDSL)
      * K6          : ``chunk_fwd_o_opt_vk``                           (Triton)

    Args:
        q: queries ``[B, T, H, K]``.
        k: keys ``[B, T, Hg, K]`` (GQA: ``Hg`` may be smaller than ``H``).
        v: values ``[B, T, H, V]``.
        g: log-decays ``[B, T, H]`` (raw, will be cumsum'd by K1).
        beta: betas ``[B, T, H]``.
        scale: attention scale; default ``1 / sqrt(K)``.
        initial_state: optional ``[N, H, V, K]`` (VK layout).
        output_final_state: whether to return the final state.
        use_qk_l2norm_in_kernel: apply L2 normalization to ``q`` and ``k``
            before the chunk pipeline.
        cu_seqlens: ``[N+1]`` cumulative sequence lengths for varlen mode.

    Returns:
        ``(o, final_state)`` where ``o`` is shape ``[B, T, H, V]`` and
        ``final_state`` is ``[N, H, V, K]`` (or ``None``).
    """
    if cu_seqlens is not None:
        if q.shape[0] != 1:
            raise ValueError(
                f"The batch size is expected to be 1 rather than {q.shape[0]} "
                f"when using `cu_seqlens`."
            )
        if initial_state is not None and initial_state.shape[0] != len(cu_seqlens) - 1:
            raise ValueError(
                f"The number of initial states is expected to be equal to the "
                f"number of input sequences, i.e., {len(cu_seqlens) - 1} "
                f"rather than {initial_state.shape[0]}."
            )

    if scale is None:
        scale = k.shape[-1] ** -0.5

    if use_qk_l2norm_in_kernel:
        q, _ = l2norm_fwd(q)
        k, _ = l2norm_fwd(k)

    # -- K1+K2 (Triton) : g_cumsum, A_raw ----------------------------------
    g_cumsum, A_raw = fused_chunk_local_cumsum_scaled_dot_kkt_fwd(
        k=k,
        beta=beta,
        g=g,
        cu_seqlens=cu_seqlens,
    )

    # -- K3+K4 (Triton) : w (head-major), u (head-major) -------------------
    w, u = fused_solve_tril_recompute_w_u(
        A_raw=A_raw,
        k=k,
        v=v,
        beta=beta,
        g_cumsum=g_cumsum,
        cu_seqlens=cu_seqlens,
    )

    # -- K5 (FlyDSL) : h, v_new, final_state -------------------------------
    h, v_new, final_state = chunk_gated_delta_rule_fwd_h_flydsl(
        k=k,
        w=w,
        u=u,
        g=g_cumsum,
        initial_state=initial_state,
        output_final_state=output_final_state,
        cu_seqlens=cu_seqlens,
    )

    # -- K6 (Triton) : o = chunk_fwd_o_opt_vk(q, k, v_new, h, g_cumsum) ----
    o = chunk_fwd_o_opt_vk(
        q=q,
        k=k,
        v=v_new,
        h=h,
        g=g_cumsum,
        scale=scale,
        cu_seqlens=cu_seqlens,
    )

    return o.to(q.dtype), final_state
