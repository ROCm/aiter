# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Fused a16w4/a16wi4/a16w16 (bf16 A x mxfp4/int4/bf16 W) 2-stage MoE kernels.

CDNA MFMA pipeline. bf16 A (no A-scale), W1/W2 upconverted to bf16 in-kernel,
non-scaled ``MFMA(16,16,32,bf16)``:

  - stage1 (:mod:`gemm1`): fused gate+up GEMM + SiLU/SiTUv2 -> bf16 intermediate
    ``[sorted_size, inter_dim]`` by sorted position (no requant/scale).
  - stage2 (:mod:`gemm2`): down-proj GEMM + routing-weighted atomic bf16 scatter
    to ``[tokens, model_dim]``.

Reuses the standard sorting/cumsum/m_indices contract and the
shuffle_weight+e8m0_shuffle W layout. Shared low-level helpers live in
:mod:`gemm1` (imported by :mod:`gemm2`); host-side launch glue is defined below.

Launch args are raw device pointers (``fx.Int64``); tensors passed as
``.data_ptr()``.
"""

import torch
from flydsl.runtime.device import get_rocm_arch

from aiter.ops.flydsl.kernels.tensor_shim import _run_compiled

from .gemm1 import compile_gemm1_a16w4_port, gemm1_a16w4_grid
from .gemm2 import compile_gemm2_a16w4_port, gemm2_a16w4_grid

__all__ = [
    "compile_gemm1_a16w4_port",
    "compile_gemm2_a16w4_port",
    "flydsl_a16w4_gemm1",
    "flydsl_a16w4_gemm2",
    "flydsl_a16w4_moe",
    "gemm1_a16w4_grid",
    "gemm2_a16w4_grid",
]


# gfx942 (CDNA3) lacks K=32 mfma_f32_16x16x32_bf16 + v_cvt_pk_bf16_f32 -> K=16 MFMA +
# scalar dequant fallback; gfx950 (CDNA4) uses the K=32 path. compile_gemm{1,2}_a16w4_port
# are @functools.cache'd, so building through them directly (all-keyword) shares one
# compile with moe_kernels.py's AOT/runtime direct calls.


def flydsl_a16w4_gemm1(
    *,
    a_bf16,
    w1_u8,
    w1_scale_u8,
    sorted_expert_ids,
    cumsum_tensor,
    m_indices,
    inter_sorted_bf16,
    n_tokens,
    NE,
    D_HIDDEN,
    D_INTER,
    topk,
    tile_m=32,
    tile_n=None,
    tile_k=256,
    waves_per_eu=None,
    k_batch=1,
    k_wave=1,
    b_nt=None,
    xcd_swizzle=0,
    gate_mode="separated",
    act="silu",
    situ_beta=1.0,
    situ_linear_beta=1.0,
    swiglu_limit=float("inf"),
    w_dtype="fp4",
    w_layout="standard",
    activation_bf16=False,
    stream=None,
):
    """a16w4/a16wi4/a16w16 fused stage1: gate+up GEMM + SiLU -> bf16 intermediate.

    ``w_dtype="fp4"`` (default): W1 mxfp4, ``w1_scale_u8`` = shuffled e8m0. ``"int4"``:
    W1 packed signed int4 (``pack_int8_to_packed_int4(shuffle_weight(w, (16,16)))``),
    ``w1_scale_u8`` groupwise bf16 in the ``shuffle_scale_for_int4`` ``(E, G//2, N_OUT, 2)``
    bf16-pair layout (dword = e*(G//2*N_OUT) + (group//2)*N_OUT + n).
    ``"bf16"``: RAW bf16 W1 preshuffled ``shuffle_weight (16,16)``; ``w1_scale_u8`` unused.

    ``w_layout="standard"`` (default) consumes the N-major GGUU preshuffle. ``"guinterleave"``
    (mxfp4 only) consumes aiter's native GUGU stage1 W1+scale layout
    (``shuffle_weight_a16w4``/``shuffle_scale_a16w4``) directly, with no host relayout.

    ``a_bf16`` is bf16 ``[n_tokens, D_HIDDEN]``. Writes the bf16 intermediate
    ``[sorted_size, D_INTER]`` (by sorted position) into ``inter_sorted_bf16``.

    Tile config: ``tile_m/n/k`` -> BM/TILE_N/TILE_K, ``waves_per_eu`` ->
    rocdl.waves_per_eu, ``b_nt`` -> W-load cache modifier (0=cached, 2=nt),
    ``xcd_swizzle`` -> XCD/HBM grid remap, ``k_wave`` -> intra-block slice-K ({1,2,4}).
    ``k_batch``/``gate_mode`` accepted for parity (only k_batch=1/separated supported).
    Tiles are CSV/registry-driven and always supplied by the caller; ``b_nt`` is
    the W-load cache modifier (0=cached, 2=nt), taken as-is (nt when unset).

    ``situ_beta``/``situ_linear_beta`` (SiTUv2 only) are runtime f32 scalars (nothing baked).

    ``act="silu_clamped"`` computes ``silu(min(gate, limit)) *
    clamp(up, -limit, limit)`` with runtime ``swiglu_limit > 0``; +inf disables
    the clamp. ``activation_bf16=True`` rounds both GEMM accumulators to bf16
    before this activation. Plain ``silu`` and the default rounding are unchanged.
    """
    if k_batch != 1:
        raise NotImplementedError(f"a16w4 gemm1 only supports k_batch=1, got {k_batch}")
    if gate_mode != "separated":
        raise NotImplementedError(
            f"a16w4 gemm1 only supports gate_mode='separated', got {gate_mode!r}"
        )
    if act == "silu_clamped" and not float(swiglu_limit) > 0.0:
        raise ValueError("silu_clamped requires swiglu_limit > 0")

    BM = tile_m
    TILE_K = tile_k
    TILE_N = tile_n
    # Tile config is fully CSV/registry-driven: the caller resolves tile_n/tile_k/
    # k_wave/b_nt/xcd_swizzle from the tuned kernelName1 and always passes them.
    b_cache_mod = 2 if b_nt is None else b_nt
    if D_HIDDEN % TILE_K != 0:
        raise NotImplementedError(
            f"a16w4 gemm1 requires D_HIDDEN (K) % {TILE_K} == 0, got H={D_HIDDEN}"
        )
    if (2 * D_INTER) % 256 != 0:
        raise NotImplementedError(
            f"a16w4 gemm1 requires 2*D_INTER % 256 == 0, got D_INTER={D_INTER}"
        )
    if D_INTER % TILE_N != 0:
        raise NotImplementedError(
            f"a16w4 gemm1 requires D_INTER % TILE_N({TILE_N}) == 0, got D_INTER={D_INTER}"
        )

    launch = compile_gemm1_a16w4_port(
        BM=BM,
        D_HIDDEN=D_HIDDEN,
        D_INTER=D_INTER,
        NE=NE,
        TOPK=topk,
        TILE_N=TILE_N,
        TILE_K=TILE_K,
        act=act,
        b_cache_mod=b_cache_mod,
        xcd_swizzle=xcd_swizzle,
        waves_per_eu=waves_per_eu,
        w_dtype=w_dtype,
        w_layout=w_layout,
        k_wave=k_wave,
        activation_bf16=activation_bf16,
        rocm_arch=str(get_rocm_arch()),
    )
    max_m_blocks = int(sorted_expert_ids.numel())
    grid = gemm1_a16w4_grid(BM, INTER=D_INTER, TILE_N=TILE_N, max_m_blocks=max_m_blocks)
    # SiTUv2 beta/linear_beta + swiglu_limit -> runtime f32 scalars (host precomputes
    # reciprocals; no device rcp). swiglu_limit is the SiTUv2 clamp bound (+inf =
    # no clamp), matching the a8w4/mixed_moe situv2 clamp so a16w4 == a8w4.
    _beta = float(situ_beta)
    _lbeta = float(situ_linear_beta)
    if _beta <= 0.0 or _lbeta <= 0.0:
        raise ValueError(
            f"situ_beta/situ_linear_beta must be > 0, got {_beta!r}/{_lbeta!r}"
        )
    _run_compiled(
        launch,
        a_bf16.data_ptr(),
        w1_u8.data_ptr(),
        w1_scale_u8.data_ptr(),
        sorted_expert_ids.data_ptr(),
        cumsum_tensor.data_ptr(),
        m_indices.data_ptr(),
        int(n_tokens),
        int(grid),
        _beta,
        1.0 / _beta,
        _lbeta,
        1.0 / _lbeta,
        float(swiglu_limit),
        inter_sorted_bf16.data_ptr(),
        torch.cuda.current_stream() if stream is None else stream,
    )
    return inter_sorted_bf16


def flydsl_a16w4_gemm2(
    *,
    inter_sorted_bf16,
    w2_u8,
    w2_scale_u8,
    sorted_expert_ids,
    cumsum_tensor,
    sorted_token_ids,
    sorted_weights,
    flat_out,
    M_logical,
    max_sorted,
    NE,
    D_HIDDEN,
    D_INTER,
    topk,
    tile_m=32,
    tile_n=256,
    tile_k=256,
    waves_per_eu=None,
    k_batch=1,
    b_nt=None,
    xcd_swizzle=1,
    w_dtype="fp4",
    persist=None,
    epilog="atomic",
    stream=None,
    reuse_lds=False,
):
    """a16w4/a16wi4/a16w16 fused stage2 (down-proj). Consumes the bf16 [sorted_size,
    D_INTER] intermediate; scatters routing-weighted bf16 into ``flat_out``.

    Tile config: ``tile_m/n/k`` -> BM/TILE_N/TILE_K, ``waves_per_eu`` ->
    rocdl.waves_per_eu, ``b_nt`` -> W-load cache modifier, ``xcd_swizzle`` -> XCD/HBM
    grid remap. ``k_batch`` for parity (must be 1). ``b_nt`` is the W-load cache
    modifier (0=cached, 2=nt), taken as-is (cached when unset).

    ``reuse_lds=True`` shares the A tile and atomic-epilogue scratch after
    the K-loop barrier. It is opt-in; existing calls retain their LDS allocation.
    """
    if k_batch != 1:
        raise NotImplementedError(f"a16w4 gemm2 only supports k_batch=1, got {k_batch}")

    BM = tile_m
    TILE_N = tile_n
    TILE_K = tile_k
    # Tile config is CSV/registry-driven: the caller resolves tile_n/tile_k/b_nt/
    # xcd_swizzle from the tuned kernelName2 and always passes them.
    if D_INTER % TILE_K != 0:
        raise NotImplementedError(
            f"a16w4 gemm2 requires D_INTER (K) % {TILE_K} == 0, got D_INTER={D_INTER}"
        )
    if D_HIDDEN % TILE_N != 0:
        raise NotImplementedError(
            f"a16w4 gemm2 requires D_HIDDEN (model_dim) % {TILE_N} == 0, got H={D_HIDDEN}"
        )

    # W-load cache modifier (0=cached, 2=nt): CSV/registry-driven via b_nt, taken
    # as-is (cached when unset).
    _b_cache_mod = 0 if b_nt is None else b_nt
    max_m_blocks = int(sorted_expert_ids.numel())
    # Persistent CU-limited grid (opt-in, default OFF; byte-identical when off): does NOT
    # close the E896 gap (padded launch's empty CTAs early-return ~free), kept as an
    # opt-in building block.
    _persist = False if persist is None else bool(persist)
    launch = compile_gemm2_a16w4_port(
        BM=BM,
        NE=NE,
        N_OUT=D_HIDDEN,
        D_INTER=D_INTER,
        TILE_N=TILE_N,
        TILE_K=TILE_K,
        b_cache_mod=_b_cache_mod,
        xcd_swizzle=xcd_swizzle,
        waves_per_eu=waves_per_eu,
        w_dtype=w_dtype,
        persist=_persist,
        rocm_arch=str(get_rocm_arch()),
        epilog=epilog,
        topk=topk,
        reuse_lds=reuse_lds,
    )
    grid = gemm2_a16w4_grid(
        BM, N_OUT=D_HIDDEN, TILE_N=TILE_N, max_m_blocks=max_m_blocks, persist=_persist
    )
    _run_compiled(
        launch,
        inter_sorted_bf16.data_ptr(),
        w2_u8.data_ptr(),
        w2_scale_u8.data_ptr(),
        sorted_expert_ids.data_ptr(),
        cumsum_tensor.data_ptr(),
        sorted_token_ids.data_ptr(),
        sorted_weights.data_ptr(),
        int(M_logical),
        int(max_m_blocks),
        int(grid),
        flat_out.data_ptr(),
        torch.cuda.current_stream() if stream is None else stream,
    )
    return flat_out


def flydsl_a16w4_moe(
    *,
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    swiglu_limit: float,
    logical_inter_dim: int | None = None,
    activation_bf16: bool = False,
    out: torch.Tensor | None = None,
    intermediate: torch.Tensor | None = None,
    tiles=None,
) -> torch.Tensor:
    """Run clamped A16W4 MoE with paired tiles and a same-math fallback.

    Weights and E8M0 scales are in the standard separated AITER preshuffle
    layout, viewed as uint8. Any logical-intermediate padding is already zero
    in both weight matrices. No router scoring, bias, EP or TP collective is
    performed here. Explicit tiles bypass tuned selection. The fallback is
    G1=32x128x256/G2=32x256x128, not generic fused_moe's backend selection.

    Caller-owned output and sorted-intermediate buffers may be reused; output
    is cleared before every atomic reduction, including graph replay.
    """
    from aiter.fused_moe import moe_sorting
    from aiter.jit.utils.chip_info import get_gfx_runtime
    from aiter.ops.flydsl.moe_a16w4_policy import (
        A16W4Tiles,
        get_a16w4_tuned_tiles,
    )

    if get_gfx_runtime() not in ("gfx942", "gfx950"):
        raise NotImplementedError("clamped A16W4 MoE requires gfx942 or gfx950")
    if not float(swiglu_limit) > 0.0:
        raise ValueError("silu_clamped requires swiglu_limit > 0")
    if hidden_states.ndim != 2 or hidden_states.dtype != torch.bfloat16:
        raise ValueError("hidden_states must be a BF16 matrix")
    token_num, model_dim = hidden_states.shape
    if w1.ndim != 3 or w2.ndim != 3:
        raise ValueError("packed weights must be three-dimensional")
    experts, doubled_inter_dim, packed_model_dim = w1.shape
    inter_dim = doubled_inter_dim // 2
    if (
        doubled_inter_dim != 2 * inter_dim
        or 2 * packed_model_dim != model_dim
        or tuple(w2.shape) != (experts, model_dim, inter_dim // 2)
        or model_dim % 256
        or inter_dim % 128
    ):
        raise ValueError("incompatible A16W4 packed weight shapes")
    if logical_inter_dim is None:
        logical_inter_dim = inter_dim
    if not 0 < logical_inter_dim <= inter_dim:
        raise ValueError("logical_inter_dim must be within the padded intermediate")
    if (
        topk_ids.ndim != 2
        or topk_ids.shape[0] != token_num
        or topk_ids.dtype != torch.int32
        or topk_weights.shape != topk_ids.shape
        or topk_weights.dtype != torch.float32
    ):
        raise ValueError("routing requires matching int32 IDs and float32 weights")
    topk = topk_ids.shape[1]
    if token_num <= 0 or not 0 < topk <= experts:
        raise ValueError("token count and top-k must be positive and valid")
    if (
        w1_scale.numel() != experts * 2 * inter_dim * ((model_dim + 255) // 256) * 8
        or w2_scale.numel() != experts * model_dim * ((inter_dim + 255) // 256) * 8
        or any(tensor.dtype != torch.uint8 for tensor in (w1, w2, w1_scale, w2_scale))
    ):
        raise ValueError(
            "packed weights and E8M0 scales must have matching uint8 storage"
        )
    tensors = (hidden_states, w1, w2, w1_scale, w2_scale, topk_ids, topk_weights)
    if any(
        tensor.device != hidden_states.device or not tensor.is_contiguous()
        for tensor in tensors
    ):
        raise ValueError("MoE inputs must be contiguous and on the same device")
    if tiles is None:
        tiles = get_a16w4_tuned_tiles(
            token_num,
            model_dim,
            inter_dim,
            logical_inter_dim,
            experts,
            topk,
            act="silu_clamped",
            activation_bf16=activation_bf16,
            swiglu_limit=swiglu_limit,
        )
        if tiles is None:
            tiles = A16W4Tiles(32, 128, 256, 256, 128)
    if (
        tiles.block_m not in (16, 32, 64, 128)
        or min(tiles.stage1_n, tiles.stage1_k, tiles.stage2_n, tiles.stage2_k) <= 0
        or inter_dim % tiles.stage1_n
        or model_dim % tiles.stage1_k
        or model_dim % tiles.stage2_n
        or inter_dim % tiles.stage2_k
    ):
        raise ValueError("tile pair does not divide the physical MoE dimensions")
    sorted_ids, sorted_weights, sorted_experts, cumsum, _ = moe_sorting(
        topk_ids, topk_weights, experts, model_dim, hidden_states.dtype, tiles.block_m
    )
    sorted_size = sorted_experts.numel() * tiles.block_m
    if intermediate is None:
        intermediate = torch.empty(
            (sorted_size, inter_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
    if out is None:
        out = torch.empty_like(hidden_states)
    for buffer, shape in (
        (out, (token_num, model_dim)),
        (intermediate, (sorted_size, inter_dim)),
    ):
        if (
            tuple(buffer.shape) != shape
            or buffer.dtype != hidden_states.dtype
            or buffer.device != hidden_states.device
            or not buffer.is_contiguous()
        ):
            raise ValueError(
                "output and intermediate buffers must match their sorted shapes"
            )
        if any(
            buffer.untyped_storage().data_ptr() == tensor.untyped_storage().data_ptr()
            for tensor in tensors
        ):
            raise ValueError("MoE output buffers must not alias inputs")
    if out.untyped_storage().data_ptr() == intermediate.untyped_storage().data_ptr():
        raise ValueError("MoE output buffers must not alias each other")
    out.zero_()
    flydsl_a16w4_gemm1(
        a_bf16=hidden_states,
        w1_u8=w1,
        w1_scale_u8=w1_scale,
        sorted_expert_ids=sorted_experts,
        cumsum_tensor=cumsum,
        m_indices=sorted_ids,
        inter_sorted_bf16=intermediate,
        n_tokens=token_num,
        NE=experts,
        D_HIDDEN=model_dim,
        D_INTER=inter_dim,
        topk=topk,
        tile_m=tiles.block_m,
        tile_n=tiles.stage1_n,
        tile_k=tiles.stage1_k,
        act="silu_clamped",
        activation_bf16=activation_bf16,
        swiglu_limit=swiglu_limit,
    )
    return flydsl_a16w4_gemm2(
        inter_sorted_bf16=intermediate,
        w2_u8=w2,
        w2_scale_u8=w2_scale,
        sorted_expert_ids=sorted_experts,
        cumsum_tensor=cumsum,
        sorted_token_ids=sorted_ids,
        sorted_weights=sorted_weights,
        flat_out=out,
        M_logical=token_num,
        max_sorted=sorted_size,
        NE=experts,
        D_HIDDEN=model_dim,
        D_INTER=inter_dim,
        topk=topk,
        tile_m=tiles.block_m,
        tile_n=tiles.stage2_n,
        tile_k=tiles.stage2_k,
        reuse_lds=tiles.reuse_lds,
    )
