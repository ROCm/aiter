# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""FlyDSL Linear Attention Prefill host wrappers (gated delta rule).

This module hosts the ``torch``-facing wrappers for the FlyDSL GDN prefill
kernels, keeping their kernel-compile modules under ``kernels.`` free of any
``torch`` dependency (mirroring the layering used by ``kernels.gdr_decode``):

* ``chunk_gated_delta_rule_fwd_h_flydsl`` -- K5 hidden-state recurrence
  (``compile_chunk_gated_delta_h``). Prepares tensors, chooses ``BV`` with a
  rule-based grid/CU heuristic, manages the compiled kernel cache and stream.
* ``gdn_prepare_fwd_flydsl`` -- K1..K4 chunk prepare fused into one kernel
  (``compile_gdn_prepare``). Allocates the outputs and applies the
  anti-camping ``grid_x`` padding.

For an end-to-end GDN forward, call
``aiter.ops.triton.gated_delta_net.chunk_gated_delta_rule_opt_vk`` with
``use_chunk_flydsl=True`` (K5) and/or ``use_prepare_flydsl=True`` (K1..K4).
The two are independent: the prepare wrapper matches the layout and
exponent-domain contract of the Triton K1+K2 pair it replaces, so it composes
with any of the three K5 backends. Use ``gdn_prepare_flydsl_supported`` to
check a problem shape before selecting it.
"""

from __future__ import annotations

import functools
import math

import torch
import triton

from ..triton._triton_kernels.gated_delta_rule.utils import (
    GatedDeltaRulePrefillMetadata,
    prepare_chunk_offsets,
    prepare_max_seq_chunks,
    prepare_num_chunks,
    prepare_rebased_cu_seqlens,
)
from .kernels.chunk_gated_delta_h import compile_chunk_gated_delta_h
from .kernels.gdn_prepare import compile_gdn_prepare
from .kernels.tensor_shim import _run_compiled

# log2(e); g pre-scaled by this constant lets the kernel use exp2(g) in
# place of exp(g) (matches the Triton VK / HIP K5 convention).
_RCP_LN2 = math.log2(math.e)


__all__ = [
    "chunk_gated_delta_rule_fwd_h_flydsl",
    "gdn_prepare_flydsl_supported",
    "gdn_prepare_fwd_flydsl",
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
    """FlyDSL K5 host wrapper.

    Signature is API-compatible with
    ``aiter.ops.triton._triton_kernels.gated_delta_rule.prefill.chunk_delta_h.chunk_gated_delta_rule_fwd_h_opt_vk``:

    Args:
        k: [B, T, Hg, K] bf16.
        w: [B, H, T_flat, K] bf16, head-major contiguous layout.
        u: [B, H, T_flat, V] bf16, head-major contiguous layout.
        g: [B, H, T_total] f32 cumulative gate, head-major contiguous
            (matches Triton VK / HIP K5), or None. Must be a
            ``contiguous()`` tensor with stride-1 along the T dimension.
            Caller passes ``g`` in natural-log space; when
            ``use_exp2=True`` the K1+K2 producer is expected to have
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
        use_exp2: whether ``g`` is in log2 space. Standalone K5 callers pass
            natural-log ``g`` by default; end-to-end prefill passes the Triton
            K1 ``use_exp2`` setting through explicitly.
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
    # HIP K5). The kernel reads ``g`` with stride-1 along the T dim; require
    # the caller to provide a contiguous head-major tensor.
    if g is not None:
        assert g.is_contiguous(), (
            "FlyDSL K5: ``g`` must be contiguous (head-major [B, H, T_flat] "
            f"or [H, T_flat]); got strides={g.stride()}, shape={tuple(g.shape)}."
        )
        assert g.shape[-1] == T_flat, (
            f"FlyDSL K5: ``g.shape[-1]`` must equal T_flat={T_flat}, "
            f"got g.shape={tuple(g.shape)}."
        )
        assert g.shape[-2] == H, (
            f"FlyDSL K5: ``g.shape[-2]`` must equal H={H}, "
            f"got g.shape={tuple(g.shape)}."
        )
    g_arg = g if g is not None else dummy

    # Mirror the Triton VK wrapper: when ``use_exp2=True`` the K5 kernel
    # interprets ``gk`` in log2 space, so pre-scale by log2(e) here. The
    # kernel-side ``_fast_exp`` for ``gk`` is shared with the ``g`` path;
    # ``g`` itself must already be log2-scaled by the K1+K2 producer when
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


# -- K1..K4 prepare host wrapper (single fused FlyDSL kernel) --------------


def _device_index(t: torch.Tensor) -> int:
    """Concrete ordinal of a CUDA tensor's device (``None`` = the current one)."""
    return t.device.index if t.device.index is not None else torch.cuda.current_device()


@functools.cache
def _num_xcd(device_index: int) -> int:
    """XCD (Accelerator Complex Die) count on the given device.

    The HW workgroup->XCD dispatch is the dominant partition behind the grid_x
    camping (see the anti-camping padding in the wrapper).  On CDNA each XCD
    holds a fixed number of active CUs, so we recover the XCD count from the
    visible CU count (also correct under CPX/partitioned modes, which expose
    fewer CUs):
        gfx942  MI300X/MI325X = 304 CU, MI300A = 228 CU  @ 38 CU/XCD -> 8 or 6
        gfx950  MI350/MI355   = 256 CU                    @ 32 CU/XCD -> 8
    """
    p = torch.cuda.get_device_properties(device_index)
    arch = getattr(p, "gcnArchName", "").split(":")[0]
    cu_per_xcd = {"gfx942": 38, "gfx950": 32}.get(arch)
    if cu_per_xcd:
        return max(1, round(p.multi_processor_count / cu_per_xcd))
    return 8  # sane CDNA default


def _pad_grid_x_coprime(grid_x: int, num_xcd: int) -> int:
    """Smallest grid_x' >= grid_x that is coprime with num_xcd (>= NT columns
    are required to cover the longest sequence, so we only pad upward)."""
    while num_xcd > 1 and math.gcd(grid_x, num_xcd) != 1:
        grid_x += 1
    return grid_x


@functools.cache
def _is_cdna_mfma_arch(device_index: int) -> bool:
    """Whether the device runs the CDNA bf16 MFMA path the prepare kernel needs."""
    arch = getattr(torch.cuda.get_device_properties(device_index), "gcnArchName", "")
    return arch.split(":")[0].startswith(("gfx94", "gfx95"))


# The launch hands flattened views to FlyDSL's C-ABI packer, which describes
# sizes as C ``int``, so a view of 2**31 elements or more dies with a bare
# ``struct.error`` before the kernel ever runs. ``w_bar``/``u_bar`` are the
# widest views at ``B*H*T*K`` elements -- equal to ``v.numel()`` here, since the
# predicate already pins ``K == V`` -- so bounding v bounds them. Measured on
# gfx950 (H=32, Hg=16, K=V=128): clean at T_flat=524160, raises at 524288, which
# is exactly 2**31 output elements. The Triton pair computes both, so falling
# back keeps that shape working instead of crashing.
_MAX_FLAT_ELEMS = 2**31


def gdn_prepare_flydsl_supported(
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    BT: int = 64,
) -> bool:
    """Whether ``gdn_prepare_fwd_flydsl`` can serve this problem shape.

    The fused kernel covers a deliberately narrow slice of what the Triton
    K1+K2 pair accepts -- ``compile_gdn_prepare`` hard-asserts ``BT=64``,
    ``K=128`` and ``V=128``, the epilogue emits bf16, the GEMMs are CDNA bf16
    MFMA, and the launch ABI caps a flattened view at ``2**31`` elements.
    Callers use this to pick the fused path only where it applies and fall back
    to Triton everywhere else, instead of tripping an assert.
    """
    return (
        BT == 64
        and k.shape[-1] == 128
        and v.shape[-1] == 128
        and k.dtype is torch.bfloat16
        and v.dtype is torch.bfloat16
        and k.is_cuda
        and v.numel() < _MAX_FLAT_ELEMS
        and _is_cdna_mfma_arch(_device_index(k))
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
    """FlyDSL GDN prepare (K1..K4 fused) host wrapper.

    Drop-in for the Triton ``fused_chunk_local_cumsum_scaled_dot_kkt_fwd`` +
    ``fused_solve_tril_recompute_w_u`` pair: same inputs, same output layouts
    and exponent domain, one kernel instead of three dispatches. The
    ``A_raw [B, T, H, BT]`` fp32 intermediate those two exchange never
    materializes here (256 MiB at T=32768, H=32).

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
            ``log2(e)``) so the downstream K5/K6 kernels can use ``exp2``.
            Matches the Triton K1 ``use_exp2`` convention, including its
            default. ``w_bar``/``u_bar`` are unaffected.
        num_decodes / num_decode_tokens: skip a leading decode-only prefix in
            the ORIGINAL ``cu_seqlens`` (the data tensors are expected
            pre-sliced to the prefill region). Offsets are rebased internally
            via the cached prologue helpers, which key on the original
            tensor's identity, so the grid metadata stays cache-warm across
            forward calls and no per-forward device-to-host read happens.
        prefill_metadata: prebuilt reusable K1--K6 schedule. Preferred over the
            cached helpers when several GDR layers share one batch: the grid
            shape then comes straight off the host-resident schedule.
        stream: launch stream; defaults to the current stream.

    Returns:
        (w_bar [B, H, T, K] bf16, u_bar [B, H, T, V] bf16,
        g_cumsum [B, H, T] f32) -- all head-major contiguous, i.e. stride 1
        along T within a head, as K5 and K6 require.

    All three outputs are shaped on the caller's ``T``; ``T`` need not be a
    multiple of ``BT`` (the ragged last chunk is handled by the kernel's
    ``seqlen`` guards, so no input padding is done).

    KKT, the triangular inverse, and the WY GEMMs all run on MFMA bf16
    16x16x16; see ``kernels.gdn_prepare.compile_gdn_prepare`` for the LDS
    layout and the fixed ``s_vT`` bank-swizzle.
    """
    B, T, Hg_in, K = k.shape
    H = v.shape[2]
    V = v.shape[3]
    if Hg is None:
        Hg = Hg_in
    assert H % Hg == 0

    # ``w_bar``/``u_bar`` are emitted as bf16, so silently up/down-casting a
    # non-bf16 ``k``/``v`` here would hand the downstream K5 a ``w``/``u`` whose
    # dtype disagrees with ``k`` -- and K5 derives ``h`` from ``k.dtype`` but
    # ``v_new`` from ``u.dtype``, so the mismatch surfaces as mixed-precision
    # garbage in K6 rather than as an error. Reject it here instead. Callers who
    # need fp16 should stay on the Triton K1+K2 pair, which follows the input
    # dtype.
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

    k = k.contiguous()
    v = v.contiguous()
    g = g.contiguous().float()
    beta = beta.contiguous().float()

    is_varlen = cu_seqlens is not None
    if is_varlen:
        assert B == 1
        # The launch grid is rectangular -- ``NT`` chunk columns by
        # ``num_seqs * H`` rows -- so it needs the LONGEST sequence's chunk
        # count, not the flattened total. Both host ints come from the shared
        # schedule (or the identity-keyed caches), so no per-forward D2H.
        if prefill_metadata is not None:
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
            kernel_cu_seqlens = schedule.kernel_cu_seqlens
            num_seqs = schedule.n_prefill
            NT = schedule.max_seq_chunks
        else:
            # Pass the ORIGINAL (cache-stable) cu_seqlens plus the decode ints
            # into the cached helpers; handing them a freshly-rebased tensor
            # would key the caches on an unstable identity and re-fire the D2H
            # every call.
            kernel_cu_seqlens = prepare_rebased_cu_seqlens(
                cu_seqlens, num_decodes, num_decode_tokens
            )
            num_seqs = len(kernel_cu_seqlens) - 1
            NT = prepare_max_seq_chunks(cu_seqlens, BT, num_decodes, num_decode_tokens)
        # Device-to-device narrowing cast; the result feeds only the launch, so
        # its identity does not matter to the caches above.
        cu = kernel_cu_seqlens.to(device=k.device, dtype=torch.int32).contiguous()
    else:
        # The dense build never dereferences ``cu_t``, so hand it ``k`` as the
        # placeholder pointer instead of allocating a dummy buffer for it.
        cu = k
        num_seqs = B
        # No input padding for a ragged last chunk: every global access in the
        # kernel is already bounded by ``seqlen`` -- the buffer descriptors are
        # clamped to the sequence's last row, so out-of-range loads return zero
        # and out-of-range stores are dropped, in hardware.  A dense tail
        # therefore behaves exactly like varlen's ragged last chunk.  Padding
        # instead cost four input copies plus an oversized output allocation,
        # and forced a trailing ``[:, :T]`` slice that returned non-contiguous
        # tensors for B > 1.
        NT = (T + BT - 1) // BT

    # The kernel writes every output position (dense: row r is written by chunk
    # r // BT; varlen: each packed token is written by exactly one block), so
    # empty is safe and avoids two ~5us fill kernels.
    w_bar = torch.empty(B, H, T, K, dtype=torch.bfloat16, device=k.device)
    u_bar = torch.empty(B, H, T, V, dtype=torch.bfloat16, device=k.device)
    g_cumsum = torch.empty(B, H, T, dtype=torch.float32, device=k.device)

    grid_x = NT
    grid_y = num_seqs * H

    # Anti partition-camping: the HW dispatches workgroups to the GPU's XCDs
    # (8 on MI300X/MI325X/MI350, 6 on MI300A; each XCD = 4 shader engines).
    # When varlen work is skewed (one long sequence among many short ones) and
    # grid_x (== NT) shares a factor with the XCD count, the long sequence's
    # chunk blocks pile onto a few XCDs/SEs while the rest idle -> up to ~8x
    # slower.  Measured (grid_x=16 on 8 XCDs): all heavy work landed on 1 XCD,
    # SQ_BUSY per-SE CV 1.73 / max-mean 7.5x; camping strength tracks
    # gcd(grid_x, num_xcd).  Padding grid_x up to the nearest value coprime
    # with num_xcd stripes the blocks across every XCD (coprime with an even
    # num_xcd => odd => also clears the 4-SE tier).  The extra chunk column(s)
    # are >= every seqlen so they early-exit and the output is bit-identical.
    # Only varlen skews (dense = uniform full chunks).
    if is_varlen and NT >= 2:
        grid_x = _pad_grid_x_coprime(NT, _num_xcd(_device_index(k)))

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
