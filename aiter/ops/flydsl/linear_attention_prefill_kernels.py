# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""FlyDSL Linear Attention Prefill host wrappers (gated delta rule).

This module hosts the ``torch``-facing wrappers for the FlyDSL GDN prefill
kernels, keeping their kernel-compile modules under ``kernels.`` free of any
``torch`` dependency (mirroring the layering used by ``kernels.gdr_decode``):

* ``chunk_gated_delta_rule_fwd_h_flydsl`` -- hidden-state recurrence
  (``compile_chunk_gated_delta_h``). Prepares tensors, chooses ``BV`` with a
  rule-based grid/CU heuristic, manages the compiled kernel cache and stream.
* ``gdn_prepare_fwd_flydsl`` -- cumsum, gated KKT, triangular inverse and the
  WY ``w``/``u`` GEMMs fused into one kernel (``compile_gdn_prepare``).
  Allocates the outputs and applies the anti-camping ``grid_x`` padding.

For an end-to-end GDN forward, call
``aiter.ops.triton.gated_delta_net.chunk_gated_delta_rule_opt_vk`` with
``use_chunk_flydsl=True`` (hidden state) and/or ``use_prepare_flydsl=True``
(prepare). The two are independent: the prepare wrapper matches the layout and
exponent-domain contract of the Triton
``fused_chunk_local_cumsum_scaled_dot_kkt_fwd`` +
``fused_solve_tril_recompute_w_u`` pair it replaces, so it composes with any of
the three ``chunk_gated_delta_rule_fwd_h`` backends. Use
``gdn_prepare_flydsl_supported`` to check a problem shape before selecting it.
"""

from __future__ import annotations

import functools
import math

import torch
import triton

from ..triton._triton_kernels.gated_delta_rule.utils import (
    GatedDeltaRulePrefillMetadata,
    prepare_chunk_offsets,
    prepare_num_chunks,
    prepare_rebased_cu_seqlens,
)
from .kernels.chunk_gated_delta_h import compile_chunk_gated_delta_h
from .kernels.gdn_prepare import compile_gdn_prepare
from .kernels.tensor_shim import _run_compiled

# log2(e); g pre-scaled by this constant lets the kernel use exp2(g) in
# place of exp(g) (matches the Triton VK / HIP convention).
_RCP_LN2 = math.log2(math.e)


__all__ = [
    "chunk_gated_delta_rule_fwd_h_flydsl",
    "gdn_prepare_flydsl_supported",
    "gdn_prepare_fwd_flydsl",
]


# -- Hidden-state host wrapper (FlyDSL kernel + rule-based BV selection) ---

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
            trace-calibrated rules to the hidden-state H=32/Hg=16 family.

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
    g_log2_scaled=False,
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
        g_log2_scaled,
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
    _run_compiled(
        launch_fn,
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
    use_exp2: bool = True,
    num_decodes: int = 0,
    num_decode_tokens: int = 0,
    prefill_metadata: GatedDeltaRulePrefillMetadata | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """FlyDSL hidden-state recurrence host wrapper.

    Signature is API-compatible with
    ``aiter.ops.triton._triton_kernels.gated_delta_rule.prefill.chunk_delta_h.chunk_gated_delta_rule_fwd_h_opt_vk``:

    Args:
        k: [B, T, Hg, K] bf16.
        w: [B, H, T_flat, K] bf16, head-major contiguous layout.
        u: [B, H, T_flat, V] bf16, head-major contiguous layout.
        g: [B, H, T_total] f32 cumulative gate, head-major contiguous
            (matches Triton VK / HIP), or None. Must be a
            ``contiguous()`` tensor with stride-1 along the T dimension.
            Caller passes ``g`` in natural-log space; when
            ``use_exp2=True`` the prepare stage is expected to have
            already pre-scaled ``g`` by ``log2(e)`` (i.e. ``g`` is in
            log2 space) -- this matches the Triton VK convention and is
            NOT re-scaled by this wrapper.
        gk: [T_total, H, K] f32 per-K cumulative gate (natural-log
            space), or None. Pre-scaled to log2 space inside the wrapper
            when ``use_exp2=True``, mirroring
            ``chunk_gated_delta_rule_fwd_h_opt_vk``.
        initial_state: [N, H, V, K] f32, or None.
        output_final_state: whether to return the final hidden state.
        chunk_size: chunk size BT (default 64).
        save_new_value: whether to materialize ``v_new``.
        cu_seqlens: [N+1] LongTensor for variable-length batching, or None.
        state_dtype: optional initial/final state dtype (float32 or bfloat16).
        use_exp2: whether ``g`` is in log2 space. Standalone callers pass
            natural-log ``g`` by default; end-to-end prefill passes the Triton
            prepare stage's ``use_exp2`` setting through explicitly.
        num_decodes: number of leading decode-only sequences to skip in
            ``cu_seqlens``. When nonzero, ``cu_seqlens`` is the ORIGINAL,
            cache-stable metadata tensor (decode prefix included) and the
            data tensors (``k/w/u/g/...``) are expected to be pre-sliced to
            the prefill region; the offsets are rebased internally via the
            cached ``prepare_rebased_cu_seqlens``.
        num_decode_tokens: number of leading decode tokens stripped from the
            data tensors; subtracted from the rebased offsets so they index
            from token 0 of the prefill region.

    Returns:
        (h, v_new, final_state) in VK-ordered layout (``[..., V, K]`` on the
        last two dims).

    BV-tile selection is rule-based. ``chunk_gdn_h_tuned.csv`` remains an AOT
    seed list for pre-compilation, but runtime BV selection does not read it.
    """
    # Layout is fixed to head-major contiguous (matches Triton VK wrapper).
    wu_contiguous = True

    g_log2_scaled = bool(use_exp2)

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
        kernel_cu_seqlens = None
    elif prefill_metadata is not None:
        prefill_metadata.validate(
            cu_seqlens=cu_seqlens,
            chunk_size=BT,
            num_decodes=num_decodes,
            num_decode_tokens=num_decode_tokens,
            total_prefill_tokens=T,
            num_sequences=len(cu_seqlens) - 1,
        )
        schedule = prefill_metadata.get_chunk_schedule(
            BT,
            num_decodes=num_decodes,
            num_decode_tokens=num_decode_tokens,
        )
        chunk_offsets = schedule.chunk_offsets
        NT = schedule.total_chunks
        kernel_cu_seqlens = schedule.kernel_cu_seqlens
        N = schedule.n_prefill
    else:
        # Pass the ORIGINAL (cache-stable) cu_seqlens + the decode ints into
        # the cached prologue helpers. They all key on the original tensor's
        # identity, so chunk_offsets / NT / the rebased kernel cu_seqlens are
        # computed ONCE per (cu_seqlens_id, BT, num_decodes, num_decode_tokens)
        # tuple and every subsequent forward is a pure cache hit -> no
        # per-forward D2H. (Passing a freshly-rebased tensor instead would key
        # the offset/num-chunk caches on an unstable identity and re-fire the
        # .tolist()/int() syncs every call.)
        chunk_offsets = prepare_chunk_offsets(
            cu_seqlens, BT, num_decodes, num_decode_tokens
        )
        NT = prepare_num_chunks(cu_seqlens, BT, num_decodes, num_decode_tokens)
        # Rebased kernel-facing cu_seqlens (matches the pre-sliced prefill
        # data). N is the prefill sequence count (len() is a shape read, no
        # sync).
        kernel_cu_seqlens = prepare_rebased_cu_seqlens(
            cu_seqlens, num_decodes, num_decode_tokens
        )
        N = len(kernel_cu_seqlens) - 1

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

    # G layout is fixed to head-major [B, H, T_flat] (matches Triton VK /
    # HIP). The kernel reads ``g`` with stride-1 along the T dim; require
    # the caller to provide a contiguous head-major tensor.
    if g is not None:
        assert g.is_contiguous(), (
            "FlyDSL hidden state: ``g`` must be contiguous (head-major "
            f"[B, H, T_flat] or [H, T_flat]); got strides={g.stride()}, "
            f"shape={tuple(g.shape)}."
        )
        assert g.shape[-1] == T_flat, (
            f"FlyDSL hidden state: ``g.shape[-1]`` must equal T_flat={T_flat}, "
            f"got g.shape={tuple(g.shape)}."
        )
        assert g.shape[-2] == H, (
            f"FlyDSL hidden state: ``g.shape[-2]`` must equal H={H}, "
            f"got g.shape={tuple(g.shape)}."
        )
    g_arg = g if g is not None else dummy

    # Mirror the Triton VK wrapper: when ``use_exp2=True`` the hidden-state
    # kernel interprets ``gk`` in log2 space, so pre-scale by log2(e) here. The
    # kernel-side ``_fast_exp`` for ``gk`` is shared with the ``g`` path;
    # ``g`` itself must already be log2-scaled by the prepare stage when
    # use_exp2 is on.
    if gk is not None:
        gk = gk.contiguous()
        if g_log2_scaled:
            gk = gk * _RCP_LN2
    gk_arg = gk if gk is not None else dummy
    h0_arg = initial_state if initial_state is not None else dummy
    ht_arg = final_state if final_state is not None else dummy
    vn_arg = v_new_buf
    # cu_arg / co_arg are the kernel-facing (rebased) offsets, narrowed to
    # int32. `.to(torch.int32)` is a device-to-device cast (no host sync); the
    # resulting fresh objects are consumed only by the kernel launch, so their
    # identity does not matter for the @tensor_cache helpers above.
    cu_arg = (
        kernel_cu_seqlens.to(torch.int32)
        if kernel_cu_seqlens is not None
        else dummy.to(torch.int32)
    )
    co_arg = (
        chunk_offsets.to(torch.int32)
        if chunk_offsets is not None
        else dummy.to(torch.int32)
    )
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
        g_log2_scaled=g_log2_scaled,
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


# -- GDN prepare host wrapper (single fused FlyDSL kernel) -----------------


def _device_index(t: torch.Tensor) -> int:
    """Concrete ordinal of a CUDA tensor's device (``None`` = the current one)."""
    return t.device.index if t.device.index is not None else torch.cuda.current_device()


def _pad_grid_x_odd(grid_x: int) -> int:
    """Round grid_x up to odd; only upward, since NT columns are required."""
    return grid_x | 1


@functools.cache
def _is_cdna_mfma_arch() -> bool:
    """Whether the live GPU has the CDNA bf16 MFMA the prepare kernel needs.

    ``get_gfx_runtime`` is the repo's runtime dispatch helper (``get_gfx`` is
    for build-time codegen and honors ``GPU_ARCHS``). It raises on archs it does
    not know, which here just means the fused path is unavailable.
    """
    try:
        from aiter.jit.utils.chip_info import get_gfx_runtime

        return get_gfx_runtime().startswith(("gfx94", "gfx95"))
    except Exception:  # noqa: BLE001
        return False


# The launch hands flattened views to FlyDSL's C-ABI packer, which sizes them as
# C ``int``, so a view of 2**31 elements or more raises ``struct.error`` before
# the kernel runs. ``w_bar``/``u_bar`` are the widest views and match
# ``v.numel()`` (the predicate pins ``K == V``), so bounding v bounds them.
_MAX_FLAT_ELEMS = 2**31


def gdn_prepare_flydsl_supported(
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    BT: int = 64,
) -> bool:
    """Whether ``gdn_prepare_fwd_flydsl`` can serve this problem shape.

    The fused kernel covers a deliberately narrow slice of what the Triton
    prepare pair accepts. Callers gate on this to fall back to Triton
    everywhere else instead of tripping an assert inside the compile.
    """
    return (
        BT == 64
        and k.shape[-1] == 128
        and v.shape[-1] == 128
        and k.dtype is torch.bfloat16
        and v.dtype is torch.bfloat16
        and k.is_cuda
        and v.is_cuda
        and _device_index(k) == _device_index(v)
        and v.numel() < _MAX_FLAT_ELEMS
        and _is_cdna_mfma_arch()
    )


def gdn_prepare_fwd_flydsl(
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    cu_seqlens: torch.Tensor | None = None,
    BT: int = 64,
    Hg: int | None = None,
    use_exp2: bool = True,
    num_decodes: int = 0,
    num_decode_tokens: int = 0,
    prefill_metadata: GatedDeltaRulePrefillMetadata | None = None,
    stream=None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """FlyDSL GDN prepare host wrapper (cumsum, KKT, inverse and WY fused).

    Drop-in for the Triton ``fused_chunk_local_cumsum_scaled_dot_kkt_fwd`` +
    ``fused_solve_tril_recompute_w_u`` pair: same inputs, same output layouts
    and exponent domain, one kernel instead of three dispatches. The
    ``A_raw [B, T, H, BT]`` fp32 intermediate those two exchange never
    materializes here.

    Args:
        k: [B, T, Hg, K] bf16.
        v: [B, T, H, V] bf16.
        g: [B, T, H] f32 raw forget-gate increments (natural-log space).
        beta: [B, T, H] f32, post-sigmoid.
        cu_seqlens: [N+1] for variable-length batching, or None for dense.
            When given, ``B`` must be 1 and ``T`` is the packed token count.
        BT: chunk size (must be 64).
        Hg: number of K/V heads; defaults to ``k.shape[2]``. ``H % Hg == 0``.
        use_exp2: when True, publish ``g_cumsum`` in log2 space (pre-scaled by
            ``log2(e)``) so the downstream ``chunk_gated_delta_rule_fwd_h`` and
            ``chunk_fwd_o`` kernels can use ``exp2``. Matches the ``use_exp2``
            convention of the Triton pair above, including its default.
            ``w_bar``/``u_bar`` are unaffected.
        num_decodes / num_decode_tokens: skip a leading decode-only prefix in
            the ORIGINAL ``cu_seqlens`` (the data tensors are expected
            pre-sliced to the prefill region). ``prefill_metadata`` carries the
            rebased offsets, so no per-forward device-to-host read happens.
        prefill_metadata: prebuilt reusable prefill schedule, required whenever
            ``cu_seqlens`` is given and shareable across the GDR layers of one
            batch. The grid shape comes straight off it, keeping the launch off
            the device offsets. Raises when absent for a varlen batch, which the
            ``chunk_gated_delta_rule_*`` entry points pre-empt by warning and
            falling back to the Triton prepare pair.
        stream: launch stream; defaults to the current stream.

    Returns:
        (w_bar [B, H, T, K] bf16, u_bar [B, H, T, V] bf16,
        g_cumsum [B, H, T] f32) -- all head-major contiguous, i.e. stride 1
        along T within a head, as ``chunk_gated_delta_rule_fwd_h`` and
        ``chunk_fwd_o`` require.

    All three outputs are shaped on the caller's ``T``; ``T`` need not be a
    multiple of ``BT``, as the kernel guards the ragged last chunk on
    ``seqlen`` instead of requiring padded inputs.
    """
    B, T, Hg_in, K = k.shape
    H = v.shape[2]
    V = v.shape[3]
    if Hg is None:
        Hg = Hg_in
    assert H % Hg == 0

    # Casting a non-bf16 ``k``/``v`` here would leave ``w``/``u`` disagreeing
    # with ``k``, which ``chunk_gated_delta_rule_fwd_h`` turns into silent
    # mixed precision downstream rather than an error. Callers needing fp16
    # should stay on the Triton prepare pair, which follows the input dtype.
    if k.dtype is not torch.bfloat16 or v.dtype is not torch.bfloat16:
        raise TypeError(
            "gdn_prepare_fwd_flydsl emits bf16 `w_bar`/`u_bar` and therefore "
            f"requires bf16 `k` and `v`; got k={k.dtype}, v={v.dtype}."
        )
    if cu_seqlens is None and (num_decodes or num_decode_tokens):
        raise ValueError(
            "`num_decodes` / `num_decode_tokens` describe a packed varlen batch "
            "and require `cu_seqlens`."
        )
    # Everything past this point assumes the supported slice; unchecked, an
    # unsupported shape surfaces as a compile assert and a cross-device tensor
    # as a segfault inside the launch.
    if not gdn_prepare_flydsl_supported(k, v, BT=BT):
        raise ValueError(
            "gdn_prepare_fwd_flydsl serves bf16 `k`/`v` with K=V=128 and BT=64, "
            "co-resident on one CDNA device, under 2**31 flattened elements; "
            f"got k={tuple(k.shape)} {k.dtype} on {k.device}, "
            f"v={tuple(v.shape)} {v.dtype} on {v.device}, BT={BT}. Gate on "
            "`gdn_prepare_flydsl_supported` and use the Triton prepare pair "
            "wherever it returns False."
        )

    k = k.contiguous()
    v = v.contiguous()
    g = g.contiguous().float()
    beta = beta.contiguous().float()

    is_varlen = cu_seqlens is not None
    if is_varlen:
        assert B == 1
        # The grid is rectangular -- ``NT`` chunk columns by ``num_seqs * H``
        # rows -- so it needs the LONGEST sequence's chunk count, not the
        # flattened total. Only the host-resident schedule carries that, and
        # deriving it from ``cu_seqlens`` would cost a device-to-host read per
        # layer per forward, so refuse it here. Callers coming through
        # ``chunk_gated_delta_rule_*`` warn and use the Triton pair instead.
        if prefill_metadata is None:
            raise ValueError(
                "gdn_prepare_fwd_flydsl needs `prefill_metadata` for a varlen "
                "batch: its launch grid is sized by the longest sequence's chunk "
                "count, which is only available on the host from the prefill "
                "schedule. Build one with "
                "`build_gated_delta_rule_prefill_metadata`, or use the Triton "
                "prepare pair."
            )
        prefill_metadata.validate(
            cu_seqlens=cu_seqlens,
            chunk_size=BT,
            num_decodes=num_decodes,
            num_decode_tokens=num_decode_tokens,
            total_prefill_tokens=T,
            num_sequences=len(cu_seqlens) - 1,
        )
        schedule = prefill_metadata.get_chunk_schedule(
            BT,
            num_decodes=num_decodes,
            num_decode_tokens=num_decode_tokens,
        )
        num_seqs = schedule.n_prefill
        NT = schedule.max_seq_chunks
        cu = schedule.kernel_cu_seqlens.to(
            device=k.device, dtype=torch.int32
        ).contiguous()
    else:
        # The dense build never dereferences ``cu``, so hand it ``k`` as the
        # placeholder pointer instead of allocating a dummy buffer.
        cu = k
        num_seqs = B
        # A ragged last chunk needs no input padding: the kernel bounds every
        # global access on ``seqlen``, so a dense tail behaves exactly like
        # varlen's ragged last chunk.
        NT = (T + BT - 1) // BT

    # Every output position is written by exactly one block, so ``empty`` is
    # safe and skips two fill kernels.
    w_bar = torch.empty(B, H, T, K, dtype=torch.bfloat16, device=k.device)
    u_bar = torch.empty(B, H, T, V, dtype=torch.bfloat16, device=k.device)
    g_cumsum = torch.empty(B, H, T, dtype=torch.float32, device=k.device)

    grid_x = NT
    grid_y = num_seqs * H

    # Anti-camping. With an even grid_x, a skewed varlen batch -- one long
    # sequence among short ones -- spreads its chunk columns unevenly and leaves
    # much of the device idle: measured 25us against 17us at NT=32. An odd
    # grid_x spreads them, and the extra column exceeds every seqlen, so it
    # early-exits and the output is unchanged. Dense batches never skew.
    if is_varlen and NT >= 2:
        grid_x = _pad_grid_x_odd(NT)

    if stream is None:
        stream = torch.cuda.current_stream()

    exe = compile_gdn_prepare(
        BT=BT,
        K=K,
        V=V,
        is_varlen=is_varlen,
        g_scale=_RCP_LN2 if use_exp2 else 1.0,
    )
    _run_compiled(
        exe,
        k.view(-1),
        v.view(-1),
        g.view(-1),
        beta.view(-1),
        cu.view(-1),
        w_bar.view(-1),
        u_bar.view(-1),
        g_cumsum.view(-1),
        T,
        H,
        Hg,
        grid_x,
        grid_y,
        stream,
    )

    return w_bar, u_bar, g_cumsum
