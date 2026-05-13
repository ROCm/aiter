# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""End-to-end FlyDSL Linear Attention Prefill APIs (gated delta rule).

This module hosts:

* ``chunk_gated_delta_rule_fwd_h_flydsl`` -- host wrapper around the K5
  hidden-state recurrence FlyDSL kernel (``compile_chunk_gated_delta_h``).
  Performs PyTorch tensor preparation, looks up ``BV`` from the
  offline-tuned table ``chunk_gdn_h_tuned.csv``, manages the compiled
  kernel cache, and handles the launch stream. The kernel-compile module
  ``kernels.chunk_gated_delta_h`` is kept ``torch``-free, mirroring the
  layering used by ``kernels.gdr_decode``.

* ``flydsl_gdr_prefill`` -- a drop-in replacement for
  ``aiter.ops.triton.gated_delta_net.chunk_gated_delta_rule_opt_vk`` where
  the K5 hidden-state recurrence runs on FlyDSL and the rest of the chunk
  pipeline (K1+K2 fused cumsum/dot-kkt, K3+K4 fused solve-tril/recompute-w-u,
  K6 output) re-uses the existing Triton implementations.
"""

from __future__ import annotations

import csv
import os
from pathlib import Path

import torch
import triton

from flydsl.runtime.device import get_rocm_arch

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


# -- K5 host wrapper (FlyDSL kernel + offline-tuned BV lookup) ------------

_compiled_kernels = {}
_BV_CANDIDATES = [16, 32, 64]
_DEFAULT_BV = 16
_TUNED_FILE = "chunk_gdn_h_tuned.csv"

# (dtype_str, arch, K, V, BT, H, Hg, T_flat, N,
#  use_g, use_gk, use_h0, store_fs, save_vn, is_varlen, wu_contig) -> {"BV": int}
GDN_H_GLOBAL_CONFIG_MAP = None
# Secondary index for the ``is_varlen=False`` nearest-T fallback. Keyed on the
# full shape tuple but with ``T_flat`` removed; value is the list of
# ``(T_flat, cfg)`` pairs sorted by ``T_flat``.
GDN_H_T_INDEX = None
GDN_H_GPU_ARCH = get_rocm_arch()
_GDN_H_FALLBACK_WARNED = set()
_GDN_H_NEAREST_WARNED = set()


def _gdn_h_shape_key_no_T(
    dtype_str,
    arch,
    K,
    V,
    BT,
    H,
    Hg,
    N,
    use_g,
    use_gk,
    use_h0,
    store_fs,
    save_vn,
    is_varlen,
    wu_contig,
):
    """Lookup key with ``T_flat`` removed (used by nearest-T fallback)."""
    return (
        dtype_str,
        arch,
        K,
        V,
        BT,
        H,
        Hg,
        N,
        use_g,
        use_gk,
        use_h0,
        store_fs,
        save_vn,
        is_varlen,
        wu_contig,
    )


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
    """Look up the best ``BV`` for this shape from the offline-tuned table.

    Falls back to ``_DEFAULT_BV`` when no entry matches (with a one-time
    per-shape warning). Mirrors the lookup-table pattern used by
    ``aiter.ops.flydsl.linear_attention_kernels.get_default_kwargs``.
    """
    global GDN_H_GLOBAL_CONFIG_MAP, GDN_H_T_INDEX
    if GDN_H_GLOBAL_CONFIG_MAP is None:
        _dict = {}
        _t_index = {}
        fname = os.path.join(Path(__file__).resolve().parent, _TUNED_FILE)
        if os.path.exists(fname):
            with open(fname, "r", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    # Coerce CSV string fields to native Python types so the
                    # lookup key tuple is byte-for-byte identical to what
                    # the runtime caller passes.
                    obj_arch = row["arch"]
                    obj_dtype = row["dtype"]
                    obj_K = int(row["K"])
                    obj_V = int(row["V"])
                    obj_BT = int(row["BT"])
                    obj_H = int(row["H"])
                    obj_Hg = int(row["Hg"])
                    obj_T_flat = int(row["T_flat"])
                    obj_N = int(row["N"])
                    obj_use_g = row["use_g"] == "True"
                    obj_use_gk = row["use_gk"] == "True"
                    obj_use_h0 = row["use_h0"] == "True"
                    obj_store_fs = row["store_fs"] == "True"
                    obj_save_vn = row["save_vn"] == "True"
                    obj_is_varlen = row["is_varlen"] == "True"
                    obj_wu_contig = row["wu_contig"] == "True"
                    cfg = {"BV": int(row["BV"])}
                    key = (
                        obj_dtype,
                        obj_arch,
                        obj_K,
                        obj_V,
                        obj_BT,
                        obj_H,
                        obj_Hg,
                        obj_T_flat,
                        obj_N,
                        obj_use_g,
                        obj_use_gk,
                        obj_use_h0,
                        obj_store_fs,
                        obj_save_vn,
                        obj_is_varlen,
                        obj_wu_contig,
                    )
                    _dict[key] = cfg
                    sk = _gdn_h_shape_key_no_T(
                        obj_dtype,
                        obj_arch,
                        obj_K,
                        obj_V,
                        obj_BT,
                        obj_H,
                        obj_Hg,
                        obj_N,
                        obj_use_g,
                        obj_use_gk,
                        obj_use_h0,
                        obj_store_fs,
                        obj_save_vn,
                        obj_is_varlen,
                        obj_wu_contig,
                    )
                    _t_index.setdefault(sk, []).append((obj_T_flat, cfg))
        for _v in _t_index.values():
            _v.sort(key=lambda x: x[0])
        GDN_H_GLOBAL_CONFIG_MAP = _dict
        GDN_H_T_INDEX = _t_index

    key = (
        dtype_str,
        GDN_H_GPU_ARCH,
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
    )
    cfg = GDN_H_GLOBAL_CONFIG_MAP.get(key, None)
    if cfg is not None:
        BV = int(cfg["BV"])
        if BV in _BV_CANDIDATES and BV <= V and V % BV == 0:
            return BV

    # Nearest-T fallback (is_varlen=False only): for non-varlen prefill the
    # optimal BV is essentially independent of T (parallel block count is
    # ``H * V/BV``, T only scales the inner loop). When the exact T_flat is
    # not tuned, look up the closest tuned T_flat for the same shape and
    # reuse its BV. ``is_varlen=True`` still falls through to ``_DEFAULT_BV``
    # to avoid mixing N/T_local effects across batches.
    if not is_varlen:
        sk = _gdn_h_shape_key_no_T(
            dtype_str,
            GDN_H_GPU_ARCH,
            K,
            V,
            BT,
            H,
            Hg,
            N,
            use_g,
            use_gk,
            use_h0,
            store_fs,
            save_vn,
            is_varlen,
            wu_contig,
        )
        candidates_for_shape = (
            GDN_H_T_INDEX.get(sk) if GDN_H_T_INDEX is not None else None
        )
        if candidates_for_shape:
            # Tie-break: prefer the smaller T_flat, i.e. err on the side of the
            # short-sequence config (which favours larger BV / lower register
            # pressure regimes) rather than overshooting.
            nearest_T, nearest_cfg = min(
                candidates_for_shape,
                key=lambda tc: (abs(tc[0] - T_flat), tc[0]),
            )
            BV = int(nearest_cfg["BV"])
            if BV in _BV_CANDIDATES and BV <= V and V % BV == 0:
                if sk not in _GDN_H_NEAREST_WARNED:
                    print(
                        f"[K5 lookup] no exact tuned BV for T_flat={T_flat} "
                        f"on shape={sk}; using nearest-T={nearest_T} -> BV={BV}."
                    )
                    _GDN_H_NEAREST_WARNED.add(sk)
                return BV

    # Fallback: pick a candidate that actually divides V, prefer _DEFAULT_BV.
    if key not in _GDN_H_FALLBACK_WARNED:
        print(
            f"[K5 lookup] no tuned BV for {key}, "
            f"falling back to BV={_DEFAULT_BV}. "
            f"Run the offline tuner to add this shape to {_TUNED_FILE}."
        )
        _GDN_H_FALLBACK_WARNED.add(key)
    candidates = [bv for bv in _BV_CANDIDATES if bv <= V and V % bv == 0]
    if not candidates:
        return _DEFAULT_BV
    if _DEFAULT_BV in candidates:
        return _DEFAULT_BV
    return candidates[0]


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
        state_bf16,
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

    BV-tile selection uses an offline-tuned lookup table
    (``chunk_gdn_h_tuned.csv``) -- mirrors the pattern used by
    ``flydsl_gdr_decode``. Shapes not present in the table fall back to
    ``_DEFAULT_BV`` with a one-time warning.
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
        k.new_empty(N, H, V, K, dtype=resolved_state_dtype)
        if output_final_state
        else None
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

    # Resolve BV from the offline-tuned lookup table.
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
