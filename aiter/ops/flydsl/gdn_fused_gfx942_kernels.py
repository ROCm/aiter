# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Host side logic of the fused gfx942 GDN K5+K6 kernel.

This module owns everything specific to the fused gfx942 build: compile cache, BV/wave selector,
launch, and the fused-vs-separate router.

The master ``linear_attention_prefill_kernels`` module owns the
separate K5 pipeline and the shape/BV machinery both paths share, i.e.,
this module depends on ``linear_attention_prefill_kernels`` but not vice versa.

"""

from __future__ import annotations

import torch

from .. import prefill_batch_metadata as _pbm
from ..gated_delta_rule_fusion import K5K6Fusion
from ..triton._triton_kernels.gated_delta_rule.utils import (
    prepare_chunk_offsets,
    prepare_rebased_cu_seqlens,
)
from . import linear_attention_prefill_kernels as _host
from .kernels.chunk_gated_delta_h_gfx942 import (
    compile_chunk_gated_delta_h_gfx942,
)
from .kernels.chunk_gated_delta_h_gfx942 import (
    select_fused_variant as _gfx942_select_fused_variant,
)
from .kernels.k5_variants import _bv_waves_of_variant, _legal_bv_candidates
from .kernels.tensor_shim import _run_compiled
from .linear_attention_prefill_kernels import (
    _GFX942_MIN_FILL,
    _RCP_LN2,
    _canonical_gate_rank,
    _check_gk_shape,
    _grid_ctas,
    _select_bv_for_grid,
    chunk_gated_delta_rule_fwd_h_flydsl,
)

GatedDeltaRulePrefillMetadata = _pbm.GatedDeltaRulePrefillMetadata

__all__ = [
    "chunk_gated_delta_rule_fwd_h_o_auto",
    "chunk_gated_delta_rule_fwd_h_o_flydsl",
    "fused_k5k6_gfx942_supported",
    "is_fused_k5k6_gfx942_unsupported",
    "should_use_fused_k5k6_gfx942",
]


# -- Fused K5+K6 compile cache + launch (gfx942) -------------------------

_compiled_fused_kernels: dict = {}


# Fused-vs-unfused selection threshold. Fuse when the fused kernel's
# grid fills the device by at least this fraction.
_FUSED_MIN_FILL = 0.45

# Fused wave-widening: auto uses num_waves=8 (NR_SPLIT=2) when it selects BV=64
# and the bv64 grid fills the device well enough that the extra resident waves
# have room to help.
_FUSED_W8_MIN_FILL = 0.55


def is_fused_k5k6_gfx942_unsupported() -> str | None:
    """Why the fused K5+K6 kernel cannot run here, or None if it can."""
    if _host._ARCH != "gfx942":
        return f"the fused K5+K6 kernel is gfx942-only; this device is {_host._ARCH}"
    if _host._device_cu_count() < 304:
        return (
            f"the fused K5+K6 kernel is only validated on MI300X/MI325X-class "
            f"gfx942 (>=304 CUs); this device reports "
            f"{_host._device_cu_count()} CUs"
        )
    return None


def fused_k5k6_gfx942_supported() -> bool:
    """Whether the fused K5+K6 kernel can run on this device at all."""
    return is_fused_k5k6_gfx942_unsupported() is None


def should_use_fused_k5k6_gfx942(*, H: int, N: int, V: int) -> bool:
    """Heuristic: is the fused K5+K6 kernel the faster choice for this shape?

    Assumes the kernel is usable at all. Consult ``fused_k5k6_gfx942_supported`` for applicability.

    Rule (gfx942): fuse iff the fused kernel's grid fill satisfies:

        fill = ceil(V / BV) * N * H / CU_count  >=  _FUSED_MIN_FILL

    where ``BV`` comes from ``_fused_bv_for_shape``.
    """
    if not fused_k5k6_gfx942_supported():
        return False
    bv, _ = _fused_bv_for_shape(H=H, V=V, N=N, variant=None)
    fill = _grid_ctas(H=H, V=V, N=N, BV=bv) / max(_host._device_cu_count(), 1)
    return fill >= _FUSED_MIN_FILL


def _fused_bv_for_shape(*, H, V, N, variant):
    """Return ``(BV, num_waves)`` for the fused kernel.

    An explicit ``variant`` is honoured (incl. ``bv64w8``);
    otherwise the largest legal BV whose grid clears
    the fill bar is chosen, with num_waves=8 when that BV is 64 and the grid is
    high-fill. Wave-widening needs BT/16 (=4) divisible by NR_SPLIT and
    N_REPEAT=BV/16 divisible by NR_SPLIT, so it only applies at BV=64 here.
    """
    _FUSED_MAX_BV = 64
    legal = sorted(
        (b for b in _legal_bv_candidates(V) if b <= _FUSED_MAX_BV), reverse=True
    )
    if not legal:
        raise ValueError(f"fused GDN K5+K6: no legal BV <= {_FUSED_MAX_BV} for V={V}.")
    if variant is not None:
        bv, num_waves = _bv_waves_of_variant(variant)
        if bv not in legal:
            raise ValueError(
                f"fused GDN K5+K6 variant {variant!r} illegal for V={V}; "
                f"legal: {[f'bv{b}' for b in legal]}"
            )
        # NR_SPLIT = num_waves / (BT//16); the kernel asserts it divides
        # N_REPEAT=BV/16 and BT//16, so w8 is only legal at BV=64.
        # The fused kernel's b_A key-split needs N_REPEAT>=NR_SPLIT.
        if num_waves > 4 and bv < 64:
            raise NotImplementedError(
                f"fused GDN K5+K6 variant {variant!r}: wave-widening is only "
                f"supported at BV=64 (needs N_REPEAT >= NR_SPLIT); use bv64w8."
            )
        return bv, num_waves

    if _host._ARCH == "gfx942":
        tuned = _gfx942_select_fused_variant(H=H, N=N, V=V)
        if tuned is not None:
            return _bv_waves_of_variant(tuned)
    bv = min(
        _select_bv_for_grid(
            H=H, V=V, N=N, target_ctas=int(_GFX942_MIN_FILL * _host._device_cu_count())
        ),
        _FUSED_MAX_BV,
    )
    num_waves = 4
    if bv == 64:
        fill64 = _grid_ctas(H=H, V=V, N=N, BV=64) / _host._device_cu_count()
        if fill64 >= _FUSED_W8_MIN_FILL:
            num_waves = 8
    return bv, num_waves


def _run_fused_k5k6_gfx942(
    *,
    q,
    k,
    w,
    u,
    g,
    gk,
    o,
    scale,
    initial_state,
    output_final_state,
    chunk_size,
    cu_seqlens,
    state_dtype,
    use_exp2,
    num_decodes,
    num_decode_tokens,
    prefill_metadata,
    variant,
):
    """gfx942 fused K5+K6 launch. Mirrors the K5 wrapper's input prep."""
    g_log2_scaled = bool(use_exp2)

    if initial_state is not None:
        resolved_state_dtype = initial_state.dtype
        if state_dtype is not None and state_dtype != resolved_state_dtype:
            raise ValueError(
                f"state_dtype={state_dtype} conflicts with "
                f"initial_state.dtype={initial_state.dtype}."
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
        N, chunk_offsets = B, None
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
        kernel_cu_seqlens = schedule.kernel_cu_seqlens
        N = schedule.n_prefill
    else:
        chunk_offsets = prepare_chunk_offsets(
            cu_seqlens, BT, num_decodes, num_decode_tokens
        )
        kernel_cu_seqlens = prepare_rebased_cu_seqlens(
            cu_seqlens, num_decodes, num_decode_tokens
        )
        N = len(kernel_cu_seqlens) - 1

    assert K <= 256, f"fused GDN K5+K6: K={K} > 256 not supported."

    use_g = g is not None
    use_gk = gk is not None
    use_h0 = initial_state is not None
    is_varlen = cu_seqlens is not None

    final_state = (
        k.new_empty(N, H, V, K, dtype=resolved_state_dtype)
        if output_final_state
        else None
    )

    dummy = torch.empty(1, device=k.device, dtype=torch.float32)

    if g is not None:
        assert g.is_contiguous(), "fused K5+K6: g must be contiguous head-major."
        assert (
            g.shape[-1] == T_flat and g.shape[-2] == H
        ), f"fused K5+K6: g must be [.., H={H}, T_flat={T_flat}]; got {tuple(g.shape)}."
        g = _canonical_gate_rank(g, (-1, H, T_flat))
    g_arg = g if g is not None else dummy

    if gk is not None:
        _check_gk_shape(gk, B=B, T_flat=T_flat, H=H, K=K, is_varlen=is_varlen)
        gk = gk.contiguous()
        if g_log2_scaled:
            gk = gk * _RCP_LN2
        gk = _canonical_gate_rank(gk, (-1, T_flat, H, K))
    gk_arg = gk if gk is not None else dummy
    h0_arg = initial_state if initial_state is not None else dummy
    ht_arg = final_state if final_state is not None else dummy

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

    BV, num_waves = _fused_bv_for_shape(H=H, V=V, N=N, variant=variant)
    NR_SPLIT = num_waves // (BT // 16)

    cache_key = (
        K,
        V,
        BT,
        BV,
        H,
        Hg,
        float(scale),
        use_g,
        use_gk,
        use_h0,
        output_final_state,
        is_varlen,
        state_bf16,
        g_log2_scaled,
        NR_SPLIT,
    )
    if cache_key not in _compiled_fused_kernels:
        _compiled_fused_kernels[cache_key] = compile_chunk_gated_delta_h_gfx942(
            K=K,
            V=V,
            BT=BT,
            BV=BV,
            H=H,
            Hg=Hg,
            USE_G=use_g,
            USE_GK=use_gk,
            USE_INITIAL_STATE=use_h0,
            STORE_FINAL_STATE=output_final_state,
            SAVE_NEW_VALUE=False,
            IS_VARLEN=is_varlen,
            WU_CONTIGUOUS=True,
            STATE_DTYPE_BF16=state_bf16,
            G_IS_LOG2_SCALED=g_log2_scaled,
            NR_SPLIT=NR_SPLIT,
            COMPUTE_OUTPUT=True,
            STORE_H=False,
            SCALE=float(scale),
        )
    launch_fn = _compiled_fused_kernels[cache_key]

    grid_v = (V + BV - 1) // BV
    grid_nh = N * H
    # Unified kernel arg order; v_new and h are unused on the fused build.
    _run_compiled(
        launch_fn,
        k,
        u,
        w,
        dummy,
        g_arg,
        gk_arg,
        dummy,
        h0_arg,
        ht_arg,
        cu_arg,
        co_arg,
        q,
        o,
        T,
        T_flat,
        N,
        grid_v,
        grid_nh,
        stream,
    )

    return o, final_state


def chunk_gated_delta_rule_fwd_h_o_flydsl(
    q: torch.Tensor,
    k: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    g: torch.Tensor | None = None,
    gk: torch.Tensor | None = None,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    chunk_size: int = 64,
    cu_seqlens: torch.LongTensor | None = None,
    state_dtype: torch.dtype | None = None,
    use_exp2: bool = True,
    num_decodes: int = 0,
    num_decode_tokens: int = 0,
    prefill_metadata: GatedDeltaRulePrefillMetadata | None = None,
    variant: str | None = None,
    o: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """FlyDSL fused K5+K6 host wrapper (GDN inter-chunk scan + output).

    Fuses the inter-chunk hidden-state recurrence (K5,
    ``chunk_gated_delta_rule_fwd_h_flydsl``) with the inter/intra-chunk output
    (K6, ``chunk_fwd_o_opt_vk``) into a single call. The fused kernel eliminates
    the ``h`` snapshot and ``v_new`` HBM round-trips between the two stages.

    Args:
        q: [B, T, Hg, K] bf16. Query; K6 scales it by ``scale``. Token-major,
            ``Hg``-strided (GQA: ``Hg`` may be < ``H``).
        k: [B, T, Hg, K] bf16. Shared by K5 (state scan) and K6 (q @ k^T).
        w: [B, H, T_flat, K] bf16, head-major contiguous (K5 input).
        u: [B, H, T_flat, V] bf16, head-major contiguous (K5 input).
        g: [B, H, T_flat] or [H, T_flat] f32 cumulative scalar gate, head-major
            contiguous, or None. Same convention as
            ``chunk_gated_delta_rule_fwd_h_flydsl``: when ``use_exp2=True`` the
            caller must have pre-scaled ``g`` to log2 space. Drives both the K5
            decay and the K6 output gate.
        gk: [T_flat, H, K] f32 per-channel cumulative gate (KDA), or None.
            Pre-scaled internally when ``use_exp2=True``. The per-channel decay
            is folded into ``h``/``v_new`` by K5; the K6 output path is ungated
            for the ``gk`` case (matches the Triton K6, which takes only scalar
            ``g``).
        scale: query scale for K6. Defaults to ``K ** -0.5`` when None.
        initial_state: [N, H, V, K] f32, or None.
        output_final_state: whether to return the final hidden state.
        chunk_size: chunk size BT (default 64).
        cu_seqlens: [N+1] LongTensor for variable-length batching, or None.
        state_dtype: optional initial/final state dtype (float32 or bfloat16).
        use_exp2: whether ``g``/``gk`` are in log2 space (see K5 wrapper).
        num_decodes / num_decode_tokens: leading decode prefix skipped in
            ``cu_seqlens`` (data tensors pre-sliced); see K5 wrapper.
        prefill_metadata: prebuilt reusable schedule; see K5 wrapper.
        variant: explicit K5 BV-variant tag, or None for the auto heuristic.
        o: optional pre-allocated [B, T, H, V] output buffer, written in place.
            A fresh buffer is allocated when None.

    Returns:
        (o, final_state) with ``o`` in token-major ``[B, T, H, V]`` (bf16) and
        ``final_state`` in ``[N, H, V, K]`` (``state_dtype``) or None.
    """
    exactly_one_gate = (g is None) != (gk is None)
    if not exactly_one_gate:
        raise ValueError(
            "chunk_gated_delta_rule_fwd_h_o_flydsl: exactly one of g, gk must be "
            "provided."
        )

    # Validate dtype: the fused kernel is bf16-only.
    for name, t in (("q", q), ("k", k), ("w", w), ("u", u)):
        if t.dtype != torch.bfloat16:
            raise ValueError(
                f"chunk_gated_delta_rule_fwd_h_o_flydsl: '{name}' must be bfloat16, "
                f"got {t.dtype}."
            )

    # Validate contiguity: the kernel assumes packed strides throughout.
    for name, t in (("q", q), ("k", k), ("w", w), ("u", u)):
        if not t.is_contiguous():
            raise ValueError(
                f"chunk_gated_delta_rule_fwd_h_o_flydsl: '{name}' must be contiguous."
            )

    B, T, _Hg, K = q.shape
    H = w.shape[1]
    V = u.shape[-1]
    if scale is None:
        scale = K**-0.5

    if o is None:
        # Token-major [B, T, H, V], matching the Triton K6 output layout.
        o = u.new_empty(B, T, H, V, dtype=u.dtype)
    elif o.shape != (B, T, H, V):
        raise ValueError(
            f"chunk_gated_delta_rule_fwd_h_o_flydsl: pre-allocated 'o' must be "
            f"[{B}, {T}, {H}, {V}], got {tuple(o.shape)}."
        )
    elif not o.is_contiguous():
        raise ValueError(
            "chunk_gated_delta_rule_fwd_h_o_flydsl: pre-allocated 'o' must be contiguous."
        )

    if _host._ARCH != "gfx942":
        raise NotImplementedError(
            f"chunk_gated_delta_rule_fwd_h_o_flydsl: the fused GDN K5+K6 kernel "
            f"is implemented for gfx942 only; got arch '{_host._ARCH}'. "
        )

    return _run_fused_k5k6_gfx942(
        q=q,
        k=k,
        w=w,
        u=u,
        g=g,
        gk=gk,
        o=o,
        scale=scale,
        initial_state=initial_state,
        output_final_state=output_final_state,
        chunk_size=chunk_size,
        cu_seqlens=cu_seqlens,
        state_dtype=state_dtype,
        use_exp2=use_exp2,
        num_decodes=num_decodes,
        num_decode_tokens=num_decode_tokens,
        prefill_metadata=prefill_metadata,
        variant=variant,
    )


def chunk_gated_delta_rule_fwd_h_o_auto(
    q: torch.Tensor,
    k: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    g: torch.Tensor | None = None,
    gk: torch.Tensor | None = None,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    chunk_size: int = 64,
    cu_seqlens: torch.LongTensor | None = None,
    state_dtype: torch.dtype | None = None,
    use_exp2: bool = True,
    num_decodes: int = 0,
    num_decode_tokens: int = 0,
    prefill_metadata=None,
    variant: str | None = None,
    o: torch.Tensor | None = None,
    fusion=None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Combined K5+K6 wrapper that applies the fused-vs-separate heuristic.

    Routes to ``chunk_gated_delta_rule_fwd_h_o_flydsl`` (fused, gfx942 only)
    when ``should_use_fused_k5k6_gfx942`` returns True, otherwise falls back to
    FlyDSL K5 + Triton K6 run sequentially.

    Args:
        fusion: ``K5K6Fusion`` enum value (or string / None). None / AUTO lets
            the shape heuristic decide. NEVER forces the separate path. ALWAYS
            forces the fused path and raises an error if the fused kernel is unsupported.
            On non-gfx942 archs, AUTO and NEVER both use the separate path (fused is gfx942-only).
        All other args: same as ``chunk_gated_delta_rule_fwd_h_o_flydsl``.

    Returns:
        (o, final_state) as ``chunk_gated_delta_rule_fwd_h_o_flydsl``.

    Raises:
        ValueError: ``fusion=ALWAYS`` on a device the fused kernel does not support.
    """
    resolved_fusion = K5K6Fusion.coerce(fusion)
    H = w.shape[1]
    V = u.shape[-1]
    N = cu_seqlens.shape[0] - 1 - num_decodes if cu_seqlens is not None else q.shape[0]

    if resolved_fusion is K5K6Fusion.ALWAYS:
        reason = is_fused_k5k6_gfx942_unsupported()
        if reason is not None:
            raise ValueError(f"K5K6Fusion.ALWAYS was requested, but {reason}.")
        use_fused = True
    else:
        use_fused = resolved_fusion is K5K6Fusion.AUTO and should_use_fused_k5k6_gfx942(
            H=H, N=N, V=V
        )

    if use_fused:
        return chunk_gated_delta_rule_fwd_h_o_flydsl(
            q=q,
            k=k,
            w=w,
            u=u,
            g=g,
            gk=gk,
            scale=scale,
            initial_state=initial_state,
            output_final_state=output_final_state,
            chunk_size=chunk_size,
            cu_seqlens=cu_seqlens,
            state_dtype=state_dtype,
            use_exp2=use_exp2,
            num_decodes=num_decodes,
            num_decode_tokens=num_decode_tokens,
            prefill_metadata=prefill_metadata,
            variant=variant,
            o=o,
        )

    # Separate path: FlyDSL K5 + Triton K6.
    h, v_new, final_state = chunk_gated_delta_rule_fwd_h_flydsl(
        k=k,
        w=w,
        u=u,
        g=g,
        gk=gk,
        initial_state=initial_state,
        output_final_state=output_final_state,
        chunk_size=chunk_size,
        save_new_value=True,
        cu_seqlens=cu_seqlens,
        state_dtype=state_dtype,
        use_exp2=use_exp2,
        num_decodes=num_decodes,
        num_decode_tokens=num_decode_tokens,
        prefill_metadata=prefill_metadata,
        variant=variant,
    )
    if scale is None:
        scale = k.shape[-1] ** -0.5
    if o is None:
        B, T = q.shape[:2]
        o = u.new_empty(B, T, H, V)
    from ..triton.gated_delta_net.gated_delta_rule import chunk_fwd_o_opt_vk

    chunk_fwd_o_opt_vk(
        q=q,
        k=k,
        v=v_new,
        o=o,
        h=h,
        g=g,
        scale=scale,
        cu_seqlens=cu_seqlens,
        use_exp2=use_exp2,
        num_decodes=num_decodes,
        num_decode_tokens=num_decode_tokens,
        prefill_metadata=prefill_metadata,
    )
    return o, final_state
