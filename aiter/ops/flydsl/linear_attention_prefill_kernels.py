# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""End-to-end FlyDSL Linear Attention Prefill APIs (gated delta rule).

This module hosts:

* ``chunk_gated_delta_rule_fwd_h_flydsl`` -- host wrapper around the K5
  hidden-state recurrence FlyDSL kernel (``compile_chunk_gated_delta_h``).
  Performs PyTorch tensor preparation, chooses ``BV`` with a rule-based
  grid/CU heuristic, manages the compiled kernel cache, and handles the
  launch stream. The kernel-compile module
  ``kernels.chunk_gated_delta_h`` is kept ``torch``-free, mirroring the
  layering used by ``kernels.gdr_decode``.

* ``flydsl_gdr_prefill`` -- a drop-in replacement for
  ``aiter.ops.triton.gated_delta_net.chunk_gated_delta_rule_opt_vk`` where
  the K5 hidden-state recurrence runs on FlyDSL and the rest of the chunk
  pipeline (K1+K2 fused cumsum/dot-kkt, K3+K4 fused solve-tril/recompute-w-u,
  K6 output) re-uses the existing Triton implementations.
"""

from __future__ import annotations

import os

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


# -- K5 host wrapper (FlyDSL kernel + rule-based BV selection) ------------

_compiled_kernels = {}
_BV_CANDIDATES = [16, 32, 64]
_DEFAULT_BV = 16


def _legal_bv_candidates(V: int) -> list[int]:
    return [c for c in _BV_CANDIDATES if c <= V and V % c == 0]


def _grid_ctas(*, H: int, V: int, N: int, BV: int) -> int:
    return max(1, N) * H * ((V + BV - 1) // BV)


def _select_bv_for_grid(*, H: int, V: int, N: int, target_ctas: int) -> int:
    """Choose the largest legal BV whose grid still covers target_ctas."""
    legal = sorted(_legal_bv_candidates(V), reverse=True)
    if not legal:
        return _DEFAULT_BV
    for bv in legal:
        if _grid_ctas(H=H, V=V, N=N, BV=bv) >= target_ctas:
            return bv
    # If even BV=16 cannot reach the target, use it to maximize grid size.
    return legal[-1]


def _target_bv_for_shape(
    *, H: int, Hg: int, T_flat: int, N: int, is_varlen: bool
) -> int | None:
    """Return the calibrated BV regime before legality/grid adjustment."""
    if is_varlen and H == 32 and Hg == 16:
        if N == 2 and 11000 <= T_flat < 15000:
            return 16
        if N == 3 and not (10000 <= T_flat < 12000 or 20000 <= T_flat < 25000):
            return 64
    if is_varlen and H == 16 and T_flat >= 32768 and N >= 7:
        return 64
    return None


def _lookup_tuned_bv(
    dtype_str,
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
    store_fs,
    save_vn,
    is_varlen,
    wu_contig,
):
    """Select ``BV`` with the rule-based grid/CU heuristic."""
    del (
        dtype_str,
        K,
        BT,
        use_g,
        use_gk,
        use_h0,
        store_fs,
        save_vn,
        wu_contig,
    )
    return _heuristic_bv(
        H=H,
        Hg=Hg,
        V=V,
        T_flat=T_flat,
        N=N,
        is_varlen=is_varlen,
    )


def _heuristic_bv(
    *,
    H: int,
    Hg: int,
    V: int,
    T_flat: int,
    N: int,
    is_varlen: bool,
) -> int:
    """Pick a sensible BV for the requested shape. Pure function: no IO, no state.

    Rules calibrated against a 27-point sweep matrix on gfx950 (20 in-csv
    shapes + 7 csv-uncovered probes). The 27 points span H in
    {8,16,24,32,48,64,128} and T_local in [256, 128000]; see
    flydsl_bv_sweep.log + flydsl_heuristic_verify.log.

      * First pick a target CTA count, then choose the largest legal BV whose
        grid ``N * H * ceil(V / BV)`` still reaches that target. Larger BV
        reduces per-CTA overhead; smaller BV exposes more CTAs for CU
        utilization.

      * ``is_varlen=False`` -- target one wave of CTAs over gfx950's 256 CUs.

      * ``is_varlen=True`` -- the target grid depends on (H, T_local) jointly:
          H <= 8:
            short chunks target the BV=64 grid; medium chunks target BV=32;
            long chunks target BV=16.
          H in (8, 16]:
            long chunks target BV=32; shorter chunks target BV=64.
          H == 32, Hg == 16:
            target grid follows the bench333/407 production trace: single
            sequence needs BV=16 grid; N=2/3 use total-T windows; N>=4 has
            enough grid at BV=64.
          H > 16:
            target the BV=64 grid unless a more specific regime above applies.

    Coverage: the rule matches the AOT seed CSV plus the measured bench333 /
    bench407 probes used during calibration. Shapes far outside the sampled
    (H, T_local) grid may still be suboptimal; extend the calibration sweep
    when production reports new shape families.

    Args:
        H: number of v-heads (per TP rank).
        V: head_v_dim.
        T_flat: flat token count fed to the kernel (sum of context lens
            in varlen, ``B*T`` otherwise).
        N: number of sequences in the batch (varlen) or batch size.
        is_varlen: whether the kernel runs in variable-length mode.
        Hg: number of k-heads (per TP rank). Currently only used to scope
            trace-calibrated rules to the K5 H=32/Hg=16 family.

    Returns:
        A BV from ``_BV_CANDIDATES`` that satisfies ``BV <= V`` and
        ``V % BV == 0``. If the rule's first choice is illegal for this
        V (rare: V<16 or V not divisible by 16), falls back to the
        largest legal candidate, then finally to ``_DEFAULT_BV``.
    """
    target_bv = _target_bv_for_shape(
        H=H, Hg=Hg, T_flat=T_flat, N=N, is_varlen=is_varlen
    )
    target_ctas = (
        _grid_ctas(H=H, V=V, N=N, BV=target_bv) if target_bv is not None else 256
    )
    return _select_bv_for_grid(H=H, V=V, N=N, target_ctas=target_ctas)


def _k5_exp2_prescaled_enabled() -> bool:
    """Read FLYDSL_K5_EXP2_PRESCALED env var at every call (no caching).

    When set to a truthy value (1/true/yes/on), the K5 kernel is compiled
    with ``G_IS_LOG2_SCALED=True`` and ``_fast_exp`` drops the per-call
    ``* log2(e)`` multiply. This is a PERF-ONLY probe: outputs are
    incorrect unless K12 has been updated to pre-scale ``g_cumsum``.
    """
    val = os.environ.get("FLYDSL_K5_EXP2_PRESCALED", "")
    return val.strip().lower() in ("1", "true", "yes", "on")


_G_HEAD_MAJOR_TRUTHY = ("1", "true", "yes", "on")
_G_HEAD_MAJOR_FALSY = ("0", "false", "no", "off")


def _k5_g_head_major_enabled() -> bool:
    """Read FLYDSL_K5_G_HEAD_MAJOR env var at every call (no caching).

    Layout for the cumulative gate tensor consumed by K5:

    * ``True`` (DEFAULT): the K5 kernel is compiled with
      ``G_HEAD_MAJOR=True`` and the host wrapper transposes ``g_cumsum``
      from the K1+K2 producer's ``[B, T, H]`` (token-major) layout into
      a head-major tensor ``[B, H, T]`` (or ``[H, T_total]`` in varlen)
      before launch, so that each head's gate values are contiguous in
      HBM (stride=1). This matches the head-major access pattern used
      by the rest of the K5 hot tensors (``w``/``u``/``v_new``).

    * ``False`` (legacy escape hatch): the K5 kernel reads ``g`` with
      the original token-major offsets ``(bos + row) * H + i_h``. Set
      ``FLYDSL_K5_G_HEAD_MAJOR=0`` (or any of ``false / no / off``) to
      restore this path -- useful for A/B parity with older FlyDSL
      revisions or for diagnosing a regression suspected to come from
      the layout switch.

    Recognised values (case-insensitive): truthy ``1/true/yes/on``,
    falsy ``0/false/no/off``. Unset / unrecognised -> default (head-
    major).
    """
    val = os.environ.get("FLYDSL_K5_G_HEAD_MAJOR", "").strip().lower()
    if val in _G_HEAD_MAJOR_FALSY:
        return False
    if val in _G_HEAD_MAJOR_TRUTHY:
        return True
    return True  # default: head-major


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
    state_bf16=False,
    g_head_major=False,
):
    g_log2_scaled = _k5_exp2_prescaled_enabled()
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
        state_bf16,
        g_log2_scaled,
        g_head_major,
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
            STATE_DTYPE_BF16=state_bf16,
            G_IS_LOG2_SCALED=g_log2_scaled,
            G_HEAD_MAJOR=g_head_major,
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
    state_dtype: torch.dtype | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """FlyDSL K5 host wrapper.

    Signature is API-compatible with
    ``aiter.ops.triton._triton_kernels.gated_delta_rule.prefill.chunk_delta_h.chunk_gated_delta_rule_fwd_h_opt_vk``:

    Args:
        k: [B, T, Hg, K] bf16.
        w: [B, H, T_flat, K] bf16, head-major contiguous layout.
        u: [B, H, T_flat, V] bf16, head-major contiguous layout.
        g: [B, T_total, H] f32 cumulative gate produced by K1+K2 (the
            ``fused_chunk_local_cumsum_scaled_dot_kkt_fwd`` token-major
            output), or None. Internally transposed to head-major
            ``[B, H, T_total]`` when ``FLYDSL_K5_G_HEAD_MAJOR`` is on
            (DEFAULT). Set ``FLYDSL_K5_G_HEAD_MAJOR=0`` to keep the
            tensor in token-major layout for both the wrapper and the
            kernel (legacy escape hatch).
        gk: [T_total, H, K] f32 per-K cumulative gate, or None.
        initial_state: [N, H, V, K] f32, or None.
        output_final_state: whether to return the final hidden state.
        chunk_size: chunk size BT (default 64).
        save_new_value: whether to materialize ``v_new``.
        cu_seqlens: [N+1] LongTensor for variable-length batching, or None.

    Returns:
        (h, v_new, final_state) in VK-ordered layout (``[..., V, K]`` on the
        last two dims).

    BV-tile selection is rule-based. ``chunk_gdn_h_tuned.csv`` remains an AOT
    seed list for pre-compilation, but runtime BV selection does not read it.
    """
    # Layout is fixed to head-major contiguous (matches Triton VK wrapper).
    wu_contiguous = True

    # SSM state dtype: derived from ``initial_state.dtype`` when provided,
    # otherwise from ``state_dtype`` kwarg, otherwise default f32 (matches
    # the legacy behaviour). Only ``torch.float32`` and ``torch.bfloat16``
    # are supported by the kernel.
    if initial_state is not None:
        resolved_state_dtype = initial_state.dtype
        if state_dtype is not None and state_dtype != resolved_state_dtype:
            raise ValueError(
                f"state_dtype={state_dtype} conflicts with "
                f"initial_state.dtype={initial_state.dtype}; pass them consistently "
                f"or omit state_dtype."
            )
    elif state_dtype is not None:
        resolved_state_dtype = state_dtype
    else:
        resolved_state_dtype = torch.float32
    if resolved_state_dtype not in (torch.float32, torch.bfloat16):
        raise ValueError(
            f"SSM state dtype must be float32 or bfloat16, got {resolved_state_dtype}."
        )
    state_bf16 = resolved_state_dtype == torch.bfloat16

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
        lens_list = lens.tolist()
        NT = sum(triton.cdiv(int(seq_len), BT) for seq_len in lens_list)
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
        k.new_empty(N, H, V, K, dtype=resolved_state_dtype)
        if output_final_state
        else None
    )
    v_new_buf = k.new_empty(B, H, T_flat, V, dtype=u.dtype)
    v_new = v_new_buf if save_new_value else None

    dummy = torch.empty(1, device=k.device, dtype=torch.float32)

    # G-layout switch (perf probe). When ``FLYDSL_K5_G_HEAD_MAJOR=1`` the K5
    # kernel is compiled with ``G_HEAD_MAJOR=True`` and expects ``g`` to be
    # contiguous along the T dimension for each head. The K1+K2 producer
    # (``fused_chunk_local_cumsum_scaled_dot_kkt_fwd``) emits ``g`` in
    # ``[B, T, H]`` (token-major) layout, so transpose+contiguous it here
    # to ``[B, H, T_flat]`` before launch. The transpose is one extra
    # ``H * T_flat`` f32 element copy per call -- the probe measures
    # whether the kernel-side scalar-load address pattern win exceeds
    # this transpose cost.
    g_head_major = _k5_g_head_major_enabled()
    if g_head_major and g is not None:
        # ``g`` shape from K1+K2: ``[B, T, H]`` (B==1 in varlen). Transpose
        # the last two dims to ``[B, H, T]`` then ``contiguous()`` so the
        # kernel's stride-1 reads land on consecutive HBM addresses.
        g_for_kernel = g.transpose(-2, -1).contiguous()
    else:
        g_for_kernel = g

    g_arg = g_for_kernel if g_for_kernel is not None else dummy
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

    # Resolve BV from the rule-based grid/CU heuristic.
    BV = _lookup_tuned_bv(
        dtype_str=str(k.dtype),
        K=K,
        V=V,
        BT=BT,
        H=H,
        Hg=Hg,
        T_flat=T_flat,
        N=N,
        use_g=use_g,
        use_gk=use_gk,
        use_h0=use_h0,
        store_fs=bool(output_final_state),
        save_vn=bool(save_new_value),
        is_varlen=is_varlen,
        wu_contig=wu_contiguous,
    )

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
        state_bf16=state_bf16,
        g_head_major=g_head_major,
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
