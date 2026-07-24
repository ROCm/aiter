# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""FlyDSL Linear Attention Prefill K5 host wrapper (gated delta rule).

This module hosts ``chunk_gated_delta_rule_fwd_h_flydsl`` -- the host
wrapper around the K5 hidden-state recurrence FlyDSL kernel
(``compile_chunk_gated_delta_h``). It performs PyTorch tensor
preparation, chooses ``BV`` with a rule-based grid/CU heuristic, manages
the compiled kernel cache, and handles the launch stream. The kernel-
compile module ``kernels.chunk_gated_delta_h`` is kept ``torch``-free,
mirroring the layering used by ``kernels.gdr_decode``.

For an end-to-end GDN forward that uses this K5 wrapper, call
``aiter.ops.triton.gated_delta_net.chunk_gated_delta_rule_opt_vk`` with
``use_chunk_flydsl=True``.
"""

from __future__ import annotations

import functools
import math
import os

import torch
import triton
from packaging.version import Version

import flydsl
import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.runtime.device import get_rocm_arch
from .kernels.chunk_gated_delta_h import compile_chunk_gated_delta_h
from .kernels.chunk_gated_delta_h_mfma16_hip import (
    compile_chunk_gated_delta_h_mfma16_hip,
)
from ..triton._triton_kernels.gated_delta_rule.utils import (
    prepare_chunk_offsets,
    prepare_num_chunks,
    prepare_rebased_cu_seqlens,
)

# The K5 kernel passes every tensor slot as ``fx.Pointer`` (raw data pointer).
# The kernel body wraps each one as ``GTensor(..., shape=(-1,))`` and never
# reads the FlyDSL memref shape/stride, so the pointer ABI produces identical
# device code while skipping the per-launch DLPack export + layout-buffer
# packing that the default layout-dynamic ``fx.Tensor`` memref incurs under
# flydsl >=0.2.0. ``fx.Pointer`` host wrapping (``flyc.from_c_void_p`` +
# ``PointerAdaptor`` fast dispatch) only exists from 0.2.0, so this op alone
# requires 0.2.0 (the rest of aiter.ops.flydsl only needs >=0.1.8).
_K5_MIN_FLYDSL_VERSION = Version("0.2.0")
_installed_flydsl_version = Version(getattr(flydsl, "__version__", "0").split("+")[0])
if _installed_flydsl_version < _K5_MIN_FLYDSL_VERSION:
    raise ImportError(
        "FlyDSL K5 linear-attention prefill requires `flydsl` "
        f">=`{_K5_MIN_FLYDSL_VERSION}` (for the fx.Pointer argument ABI), "
        f"but got `{getattr(flydsl, '__version__', 'unknown')}`."
    )


def _as_ptr(t: torch.Tensor):
    """Wrap a torch tensor as a flydsl ``Pointer`` argument (raw data ptr).

    Uses ``fx.Uint8`` element type: the K5 kernel re-types every slot inside
    the body via ``GTensor(ptr, dtype=...)``, so the host-side element type is
    irrelevant to codegen and only needs to be a valid 1-byte unit.
    """
    return flyc.from_c_void_p(fx.Uint8, t.data_ptr())


# log2(e); g pre-scaled by this constant lets the kernel use exp2(g) in
# place of exp(g) (matches the Triton VK / HIP K5 convention).
_RCP_LN2 = math.log2(math.e)


__all__ = [
    "chunk_gated_delta_rule_fwd_h_flydsl",
]


# -- K5 host wrapper (FlyDSL mfma16_hip kernel; BV via hip K5 select) -----

# gfx942 gate: only the mfma16_hip fork toggles the gfx942 GEMM1 ds-scheduling
# (SCHED_GFX942). ``get_rocm_arch()`` may return a feature-suffixed string like
# ``gfx942:sramecc+:xnack-``; normalize before matching.
_IS_GFX942 = get_rocm_arch().split(":")[0].startswith("gfx942")


_INT32_ATTR = "_flydsl_int32_view"


def _as_int32(t: torch.Tensor) -> torch.Tensor:
    """Return an int32 narrowing of ``t``, cached on the tensor itself.

    ``t`` is expected to come from one of the ``@tensor_cache``-decorated
    prologue helpers (so its identity is stable across forwards). The
    cached int32 result lives as an attribute on ``t`` itself, which keeps
    cache invalidation trivially correct: when the upstream cache evicts
    ``t``, the int32 copy is collected with it.
    """
    if t.dtype == torch.int32:
        return t
    cached = getattr(t, _INT32_ATTR, None)
    if cached is None:
        cached = t.to(torch.int32)
        try:
            object.__setattr__(t, _INT32_ATTR, cached)
        except (AttributeError, TypeError):
            # Some tensor subclasses or autograd-tracked tensors disallow
            # ad-hoc attributes; fall back to the uncached cast (still
            # correct, just no longer hot-path-optimised for this caller).
            pass
    return cached


_PROLOGUE_ATTR = "_flydsl_prologue_cache"


def _resolve_prologue(
    cu_seqlens: torch.Tensor,
    BT: int,
    num_decodes: int,
    num_decode_tokens: int,
):
    """Resolve the per-shape varlen prologue in one cached lookup.

    Each of ``prepare_chunk_offsets`` / ``prepare_num_chunks`` /
    ``prepare_rebased_cu_seqlens`` is already ``@tensor_cache``-decorated, so
    every call is "just" a tuple compare + dict lookup. That is still
    ~0.55us each via the upstream Python wrapper (≈1.7us total across the
    three calls), so we collapse them into a single tuple attached
    directly to ``cu_seqlens`` (keyed by ``(BT, num_decodes,
    num_decode_tokens)``). After the first forward on a given
    ``cu_seqlens`` tensor, this is one ``getattr`` + one dict get on a
    tiny dict, ~0.15us.

    The exact minimum segment length (``min_seqlen``) is cached here too.
    Reading it requires a ``min().item()`` host<->device sync (~5us), but the
    sync is paid once per ``cu_seqlens`` tensor, not per forward. That lets
    the BV heuristic keep the min-based balanced-split carve-outs without a
    separate launch-plan cache.

    Returns ``(NT, chunk_offsets, kernel_cu_seqlens, N, min_seqlen)``.
    """
    cache_key = (BT, num_decodes, num_decode_tokens)
    cache = getattr(cu_seqlens, _PROLOGUE_ATTR, None)
    if cache is None:
        cache = {}
        try:
            object.__setattr__(cu_seqlens, _PROLOGUE_ATTR, cache)
        except (AttributeError, TypeError):
            # Subclass disallows ad-hoc attrs; fall through to recomputing
            # (still correct, just slower).
            cache = None
    if cache is not None:
        hit = cache.get(cache_key)
        if hit is not None:
            return hit

    chunk_offsets = prepare_chunk_offsets(
        cu_seqlens, BT, num_decodes, num_decode_tokens
    )
    NT = prepare_num_chunks(cu_seqlens, BT, num_decodes, num_decode_tokens)
    kernel_cu_seqlens = prepare_rebased_cu_seqlens(
        cu_seqlens, num_decodes, num_decode_tokens
    )
    N = len(kernel_cu_seqlens) - 1
    if N >= 1:
        seg_lens = kernel_cu_seqlens[1:] - kernel_cu_seqlens[:-1]
        min_seqlen = int(seg_lens.min().item())
    else:
        min_seqlen = None
    result = (NT, chunk_offsets, kernel_cu_seqlens, N, min_seqlen)
    if cache is not None:
        cache[cache_key] = result
    return result


@functools.lru_cache(maxsize=None)
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
    """Compile (and cache) the K5 kernel for one compile-time config.

    Cached via ``lru_cache`` keyed on the full compile-time constant set,
    mirroring the gemm/moe/gdr_decode flydsl ops. ``maxsize=None`` because
    the number of distinct configs is naturally bounded by the compile-time
    constant combinations.
    """
    return compile_chunk_gated_delta_h(
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


@functools.lru_cache(maxsize=None)
def _get_or_compile_mfma16_hip(
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
    use_state_indices=False,
    sched_gfx942=False,
):
    """Compile (and cache) the mfma16 / HIP-aligned K5 kernel (formerly the "vk"
    fork): 16x16x16 bf16 MFMA + HIP-matching warp partition, writing the public
    VK layout [..., V, K]. Same compile-time config surface as the baseline / KV
    paths; distinct cache namespace.

    ``use_state_indices`` compiles the indexed state-pool variant: the SSM
    ``initial_state`` is a pool ``[pool_size, H, V, K]`` and each sequence's slot
    is gathered from an ``initial_state_indices[N]`` int32 array (with in-place
    final-state write-back into the same pool slot), mirroring the HIP kernel."""
    return compile_chunk_gated_delta_h_mfma16_hip(
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
        USE_STATE_INDICES=use_state_indices,
        SCHED_GFX942=sched_gfx942,
    )


def _resolve_state_dtype(initial_state, state_dtype):
    """Mirror the legacy state-dtype resolution. Cheap; runs every call."""
    if initial_state is not None:
        resolved = initial_state.dtype
        if state_dtype is not None and state_dtype != resolved:
            raise ValueError(
                f"state_dtype={state_dtype} conflicts with "
                f"initial_state.dtype={initial_state.dtype}; pass them "
                f"consistently or omit state_dtype."
            )
    elif state_dtype is not None:
        resolved = state_dtype
    else:
        resolved = torch.float32
    if resolved not in (torch.float32, torch.bfloat16):
        raise ValueError(
            f"SSM state dtype must be float32 or bfloat16, got {resolved}."
        )
    return resolved


# Valid ``_fork`` selectors for ``chunk_gated_delta_rule_fwd_h_flydsl``. Each
# maps to its own compiled kernel + cache namespace (see the dispatch in the
# wrapper body). ``None`` (not listed here) means the baseline kernel.
_K5_FORKS = frozenset({"mfma16_hip"})


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
    initial_state_indices: torch.Tensor | None = None,
    inplace_final_state: bool | None = None,
    _fork: str | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """FlyDSL K5 host wrapper.

    ``_fork`` selects which compiled kernel implementation to run; it is an
    internal routing knob set by the thin per-fork public wrappers
    (``chunk_gated_delta_rule_fwd_h_flydsl_kv`` etc.), not part of the public
    API. Valid values:

      * ``None``        -> baseline kernel (``chunk_gated_delta_h.py``).
      * ``"kv"``        -> KV fork (coalesced [..., K, V] h-store); BV==64 only.
      * ``"mfma16_hip"`` -> mfma16 / HIP-aligned fork (public [..., V, K]
        layout; 16x16x16 MFMA + HIP split-M warp partition; NON-VWARP only);
        BV in {16, 32, 64} (N_REPEAT = BV // 16).
      * ``"mfma16_3wave_opt2"`` -> exact copy of mfma16_2wave_opt1 (KV-in /
        VK-out fork); divergence base for a 3-wave variant.
      * ``"mfma16_2wave_opt1"``  -> un-pipelined KV fork; BV==64 only.
      * ``"naive"``     -> un-pipelined baseline fork; any BV.

    The ``kv``/``mfma16_2wave_opt1`` forks store h in [..., K, V] (coalesced) and the
    wrapper returns a transposed VK view; every other fork writes the public
    VK layout ([..., V, K]) directly. Forks gated on BV==64 fall back to the
    baseline kernel for other BV values.

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

    BV-tile selection prefers an exact match from ``chunk_gdn_h_tuned.csv``
    (the same file AOT uses as its pre-compilation seed list), then falls
    back to the rule-based heuristic for shapes the sparse csv does not cover.
    """
    # All shape/flag-derived launch products (BV, launch_fn, grid dims,
    # output-buffer shapes, ...) are recomputed inline per forward, MoE/GEMM-
    # style: no per-shape launch-plan cache. This is affordable because each
    # individually-expensive input is itself cached -- the compiled kernel via
    # ``_get_or_compile`` (lru_cache), the varlen prologue + exact
    # ``min_seqlen`` via the per-cu_seqlens cache in ``_resolve_prologue``, and
    # the int32 offset views via ``_as_int32``. Everything else here is cheap
    # host-side arithmetic.

    use_g = g is not None
    use_gk = gk is not None
    use_h0 = initial_state is not None
    g_log2_scaled = bool(use_exp2)

    # Indexed state-pool support (mfma16_hip fork only). When
    # ``initial_state_indices`` is given, ``initial_state`` is a pool
    # ``[pool_size, H, V, K]`` and each sequence gathers its slot from the index
    # array; the final state is written back in place into that same pool
    # (mirrors ``chunk_gated_delta_rule_fwd_h_hip_fn``). ``inplace_final_state``
    # defaults to True whenever indices are given.
    use_state_indices = initial_state_indices is not None
    inplace = use_state_indices if inplace_final_state is None else inplace_final_state
    if use_state_indices:
        if _fork != "mfma16_hip":
            raise NotImplementedError(
                "FlyDSL K5: initial_state_indices is only implemented for the "
                f"mfma16_hip fork; got _fork={_fork!r}."
            )
        if initial_state is None:
            raise ValueError(
                "FlyDSL K5: initial_state_indices requires initial_state (the "
                "state pool)."
            )
        if not inplace:
            raise ValueError(
                "FlyDSL K5: initial_state_indices requires in-place final-state "
                "write-back; leave inplace_final_state unset or set it to True."
            )
        if not output_final_state:
            raise ValueError(
                "FlyDSL K5: initial_state_indices requires output_final_state=True "
                "(the indexed path writes the final state back into the pool)."
            )
    elif inplace and initial_state is None:
        raise ValueError("FlyDSL K5: inplace_final_state requires initial_state.")

    # State-dtype validation: cheap, and raises on bad / conflicting dtypes.
    # ``state_bf16`` is the only derived bit the compile key needs.
    resolved_state_dtype = _resolve_state_dtype(initial_state, state_dtype)
    state_bf16 = resolved_state_dtype is torch.bfloat16

    # The KV forks ("kv" and "mfma16_2wave_opt1") consume k pre-transposed to
    # [B, Hg, K, T_flat] (K major, T innermost so GEMM2 sees each BT row
    # contiguous). This is now their REQUIRED input layout -- the caller
    # (upstream / tests) owns the transpose, the host does NOT permute. Every
    # other fork keeps the original token-major [B, T_flat, Hg, K] layout.
    # mfma32_vk 用 token-major k（与 Triton VK 一致），不需要 pre-transpose
    _kv_k_pretransposed = _fork in ("kv", "mfma16_2wave_opt1", "mfma16_3wave_opt2")
    if _kv_k_pretransposed:
        B, Hg, K, T = k.shape
    else:
        B, T, Hg, K = k.shape
    H = w.shape[1]
    V = u.shape[-1]
    T_flat = w.shape[2]
    BT = chunk_size

    assert K <= 256

    if cu_seqlens is None:
        N = B
        NT = triton.cdiv(T, BT)
        chunk_offsets = None
        kernel_cu_seqlens = None
        is_varlen = False
        min_seqlen = None
    else:
        # The exact ``min_seqlen`` comes from the per-``cu_seqlens`` cache, so
        # its ``.item()`` host<->device sync is paid once per cu_seqlens tensor
        # (not per forward) -- letting the heuristic use the exact min without
        # a per-call stall and without a launch-plan cache.
        NT, chunk_offsets, kernel_cu_seqlens, N, min_seqlen = _resolve_prologue(
            cu_seqlens, BT, num_decodes, num_decode_tokens
        )
        is_varlen = True

    # BV 选择：gfx942 / gfx950 统一走 hip K5 的解析式选择（_select_bv_for_varlen /
    # _select_bv_for_dense），与 chunk_gated_delta_rule_fwd_h_hip_fn 逐函数一致，不再
    # 使用 flydsl 自己的 csv-best / heuristic 逻辑（更简单，且两实现选出相同 BV）。
    # lazy import 避免与 hip 模块潜在的循环导入；两侧 chunk_offsets 同源。
    from aiter.ops.chunk_gated_delta_rule_fwd_h import (
        _select_bv_for_dense as _hip_select_bv_for_dense,
        _select_bv_for_varlen as _hip_select_bv_for_varlen,
    )

    if is_varlen:
        BV = _hip_select_bv_for_varlen(chunk_offsets, H)
    else:
        BV = _hip_select_bv_for_dense(B, T_flat, chunk_size, H, k.device)

    if _fork is not None and _fork not in _K5_FORKS:
        raise ValueError(
            f"FlyDSL K5: unknown ``_fork`` {_fork!r}; expected one of "
            f"{sorted(_K5_FORKS)} or None (baseline)."
        )

    # mfma16_hip accepts BV in {16,32,64} and normally uses the hip-aligned BV
    # above. Allow an env override for A/B BV sweeps; the hand-tuned HIP K5
    # reference is fixed at BV=16 (FLYDSL_K5_MFMA16HIP_BV=16 reproduces it).
    if _fork == "mfma16_hip":
        _bv_env = os.environ.get("FLYDSL_K5_MFMA16HIP_BV")
        if _bv_env:
            BV = int(_bv_env)
            assert BV in (16, 32, 64), "mfma16_hip BV must be in {16,32,64}"
            if V % BV != 0:
                raise ValueError(
                    f"FlyDSL K5 mfma16_hip: requires V % BV == 0; got V={V}, BV={BV}."
                )

    # Fork routing: only the mfma16_hip fork (NON-VWARP HIP split-M kernel writing
    # the public VK layout [..., V, K] directly) or the baseline kernel
    # (``_fork=None``). mfma16_hip is fully BV-parameterized (N_REPEAT = BV // 16),
    # accepting BV in {16,32,64}.
    _mfma16_hip_gate = BV in (16, 32, 64)
    _mfma16_hip_active = (_fork == "mfma16_hip") and _mfma16_hip_gate
    # Both mfma16_hip and baseline write the public VK layout directly; no KV
    # transpose view is needed.
    _kv_needs_transpose = False
    if _mfma16_hip_active:
        _compile_fn = _get_or_compile_mfma16_hip
    else:
        _compile_fn = _get_or_compile
    # ``use_state_indices`` is an mfma16_hip-only compile knob; every other
    # fork's ``_get_or_compile*`` shares the common signature without it.
    _extra_compile_kwargs = (
        # SCHED_GFX942 仅在 gfx942 上开启（_IS_GFX942）；其它 arch（含 gfx950）传
        # False，emit 与改动前逐字节一致，且并入 lru_cache key 成为独立编译产物。
        {"use_state_indices": use_state_indices, "sched_gfx942": _IS_GFX942}
        if _mfma16_hip_active
        else {}
    )
    launch_fn = _compile_fn(
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
        True,
        state_bf16=state_bf16,
        g_log2_scaled=g_log2_scaled,
        **_extra_compile_kwargs,
    )

    # Null-arg placeholder for the @flyc.jit slots the kernel ignores on this
    # path (one local scalar tensor allocated per forward, MoE-style -- no
    # global cache). Sized 1 (not 0) so its ``data_ptr()`` is always a valid
    # non-null device address for the launcher's unused arg slots.
    dummy = torch.empty(1, device=k.device, dtype=torch.float32)
    int32_dummy = dummy.to(torch.int32) if not is_varlen else None
    cu_arg = (
        _as_int32(kernel_cu_seqlens) if kernel_cu_seqlens is not None else int32_dummy
    )
    co_arg = _as_int32(chunk_offsets) if chunk_offsets is not None else int32_dummy
    stream = torch.cuda.current_stream(k.device)

    grid_v = triton.cdiv(V, BV)
    grid_nh = N * H

    # h hidden-state layout:
    #   * baseline & mfma16_hip fork -> [B, NT, H, V, K] (true VK, K innermost).
    #   * KV fork (BV==64)    -> [B, NT, H, K, V] (coalesced KV stores); a
    #     transposed VK view is returned below so the public layout stays VK.
    # ``_kv_active`` was set at the compile-fn selection above; the mfma16_hip
    # fork (``_mfma16_hip_active``) writes the public VK layout directly, no view needed.
    h_shape = (B, NT, H, K, V) if _kv_needs_transpose else (B, NT, H, V, K)
    vn_shape = (B, H, T_flat, V)
    vn_dtype = u.dtype
    fs_shape = (N, H, V, K) if output_final_state else None
    fs_dtype = resolved_state_dtype if output_final_state else None
    save_vn = save_new_value

    # G contiguity guard. We only check ``is_contiguous`` (not the full
    # shape): a mismatched ``g.shape[-1]`` vs ``w.shape[2]`` (T_flat) can
    # only happen if the caller breaks the documented [B, H, T_flat]
    # contract -- in which case strides diverge and ``is_contiguous()``
    # catches the common modes (transposed view, slice) for ~50ns.
    if g is not None and not g.is_contiguous():
        raise AssertionError(
            "FlyDSL K5: ``g`` must be contiguous (head-major [B, H, T_flat] "
            f"or [H, T_flat]); got strides={g.stride()}, shape={tuple(g.shape)}."
        )

    # gk pre-scaling: still per-call work (allocates a new tensor). Cannot
    # be cached without aliasing across forwards; an upstream producer
    # change to emit log2-space gk directly would eliminate this entirely.
    if gk is not None:
        gk = gk.contiguous()
        if g_log2_scaled:
            gk = gk * _RCP_LN2

    # mfma16_2wave_opt1 (vn-direct): k already arrives pre-transposed as
    # [B, Hg, K, T_flat] (the caller owns the transpose), so GEMM2 sees each BT
    # row contiguous with no host-side permute/contiguous. The kernel is pinned
    # to BV=64 above, so there is no baseline fallback that would expect the
    # original token-major layout.
    h = k.new_empty(h_shape)
    v_new_buf = k.new_empty(vn_shape, dtype=vn_dtype)
    if fs_shape is None:
        final_state = None
    elif inplace:
        # In-place write-back: the final state aliases the ``initial_state``
        # buffer (the pool when indexed, or the dense [N,H,V,K] state otherwise),
        # so no separate output tensor is allocated. The kernel writes each
        # sequence's slot back into this same buffer.
        final_state = initial_state
    else:
        final_state = k.new_empty(fs_shape, dtype=fs_dtype)

    # The 11 tensor slots, wrapped as raw fx.Pointer args. Keep the torch
    # tensors referenced as locals (``k``/``u``/``h``/... above) so the storage
    # stays alive across the (synchronous) launch -- ``from_c_void_p`` only
    # captures the data pointer, not a reference to the tensor.
    tensor_args = tuple(
        _as_ptr(t)
        for t in (
            k,
            u,
            w,
            v_new_buf,
            g if g is not None else dummy,
            gk if gk is not None else dummy,
            h,
            initial_state if initial_state is not None else dummy,
            final_state if final_state is not None else dummy,
            cu_arg,
            co_arg,
        )
    )

    # The mfma16_hip kernel carries an extra ``state_indices`` slot (12th tensor
    # arg). Always supply it for that fork -- a real int32 [N] index array when
    # indexed, else a 1-elem int32 dummy for the dense path. ``si_i32`` is kept
    # as a local so its storage stays alive across the synchronous launch.
    if _mfma16_hip_active:
        if use_state_indices:
            si_i32 = initial_state_indices.to(torch.int32).contiguous()
            if si_i32.numel() != N:
                raise ValueError(
                    "FlyDSL K5: initial_state_indices length "
                    f"({si_i32.numel()}) must equal the number of sequences "
                    f"N={N}."
                )
        else:
            si_i32 = dummy.to(torch.int32)
        tensor_args = tensor_args + (_as_ptr(si_i32),)

    launch_fn(
        *tensor_args,
        T,
        T_flat,
        N,
        grid_v,
        grid_nh,
        stream,
    )

    # OPT-KV: the optimized ``kv`` fork wrote h as [B, NT, H, K, V]; return a
    # transposed VIEW so the public layout stays VK (zero-copy stride swap).
    # The VK forks and mfma16_2wave_opt1 (h-b128) write the public VK layout
    # ([..., V, K]) directly, so no view is needed.
    if _kv_needs_transpose:
        h = h.transpose(-2, -1)

    return h, (v_new_buf if save_vn else None), final_state


def chunk_gated_delta_rule_fwd_h_flydsl_mfma16_hip(
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
    initial_state_indices: torch.Tensor | None = None,
    inplace_final_state: bool | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """mfma16 / HIP-aligned K5 implementation (formerly the "vk" fork): NON-VWARP
    only -- uses the 16x16x16 bf16 MFMA and the SAME split-M warp partition (BT
    split-M, K split across waves, V not split across warps) as the hand-tuned
    HIP/C++ K5 kernel, writing the public VK layout [..., V, K]. API-identical to
    ``chunk_gated_delta_rule_fwd_h_flydsl``.

    Supports the indexed state-pool contract via ``initial_state_indices`` /
    ``inplace_final_state`` (see ``chunk_gated_delta_rule_fwd_h_flydsl``),
    matching ``chunk_gated_delta_rule_fwd_h_hip_fn``. The returned h is the usual
    VK layout."""
    return chunk_gated_delta_rule_fwd_h_flydsl(
        k,
        w,
        u,
        g=g,
        gk=gk,
        initial_state=initial_state,
        output_final_state=output_final_state,
        chunk_size=chunk_size,
        save_new_value=save_new_value,
        cu_seqlens=cu_seqlens,
        state_dtype=state_dtype,
        use_exp2=use_exp2,
        num_decodes=num_decodes,
        num_decode_tokens=num_decode_tokens,
        initial_state_indices=initial_state_indices,
        inplace_final_state=inplace_final_state,
        _fork="mfma16_hip",
    )
