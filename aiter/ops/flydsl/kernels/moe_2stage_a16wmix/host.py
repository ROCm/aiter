# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Host-side launch glue for the fused a16w4/a16wi4/a16w16 MoE kernels.

bf16 A (no A-scale), bf16 ``[sorted_size, inter]`` intermediate (no scale). Reuses
the standard sorting/cumsum/m_indices contract and the shuffle_weight+e8m0_shuffle W
layout. Kernel launch args are raw device pointers (``fx.Int64``); tensors are passed
as ``.data_ptr()``.
"""

import csv
import functools
import os
import re

import torch

from aiter.ops.flydsl.kernels.tensor_shim import _run_compiled

from .gemm1 import compile_gemm1_a16w4_port, gemm1_a16w4_grid
from .gemm2 import compile_gemm2_a16w4_port, gemm2_a16w4_grid


@functools.cache
def _get_compiled_gemm1_a16w4(
    BM,
    D_HIDDEN,
    D_INTER,
    NE,
    topk,
    TILE_N,
    TILE_K,
    act,
    b_cache_mod,
    xcd_swizzle,
    waves_per_eu,
    w_dtype="mxfp4",
    k_wave=1,
):
    return compile_gemm1_a16w4_port(
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
        k_wave=k_wave,
    )


@functools.cache
def _get_compiled_gemm2_a16w4(
    BM,
    NE,
    N_OUT,
    D_INTER,
    TILE_N,
    TILE_K,
    b_cache_mod=2,
    xcd_swizzle=1,
    waves_per_eu=None,
    w_dtype="mxfp4",
    persist=False,
):
    return compile_gemm2_a16w4_port(
        BM=BM,
        NE=NE,
        N_OUT=N_OUT,
        D_INTER=D_INTER,
        TILE_N=TILE_N,
        TILE_K=TILE_K,
        b_cache_mod=b_cache_mod,
        xcd_swizzle=xcd_swizzle,
        waves_per_eu=waves_per_eu,
        w_dtype=w_dtype,
        persist=persist,
    )


def _default_tile_n(N, *, w_dtype="mxfp4"):
    """Adaptive N-tile default for the a16w4/a16wi4/a16w16 stages.

    mxfp4/bf16: the fat-wave tile_n=256 is the tuned aiter tile and wins on the large
    Kimi-K3 shapes; keep 256 when N % 256 == 0, else 128. int4: this bandwidth-/grid-
    fill-bound path prefers tile_n=128 (measured faster on both stages across all int4
    MoE shapes tested), falling back to 64 only when N is not a multiple of 128.
    """
    if w_dtype == "int4":
        if N % 128 == 0:
            return 128
        return 64 if N % 64 == 0 else 128
    return 256 if N % 256 == 0 else 128


def a16wi4_recommend_block_m(tokens, experts, topk, *, base_block_m=32):
    """Recommend the routing/gemm1 block_m (tile_m) for the a16wi4 (int4-W) stage1.

    int4 gemm1 stage1 is W-load bound: with all experts active, the per-expert W1
    tile is re-fetched from HBM once per padded m-block. At the *fill point* where
    each expert has exactly two half-full 32-row m-blocks, doubling block_m to 64
    collapses them into ONE full 64-row block, halving the W1 HBM re-reads (measured
    TCC_MISS 3.34e7 -> 1.47e7, ~2.27x, on 7168x512 E384/k8 at tok2048) and cutting
    stage1 latency ~7%. Outside that band block_m=64 either wastes padding (one
    half-empty 64-row block when there is <=1 block/expert) or halves grid
    parallelism with no reuse gain (>=3 blocks/expert), so it is *not* applied.

    The decision keys on ceil(tokens*topk/experts / base_block_m), the average padded
    m-blocks per expert at block_m==base_block_m (== aiter's ``estimated_m_per_expert``
    heuristic): return 2*base_block_m iff that count is exactly 2 (avg tokens/expert
    in (base_block_m, 2*base_block_m]). base_block_m==32 only (the a16wi4 default);
    other block_m are passed through unchanged.

    IMPORTANT: block_m sizes the moe_sorting padding, so the caller MUST build the
    routing buffers with the SAME block_m it passes to gemm1 -- this helper is a
    dispatcher-side recommendation, not a gemm1-internal override.
    """
    if int(base_block_m) != 32 or int(experts) <= 0:
        return int(base_block_m)
    routes = int(tokens) * int(topk)
    # ceil(avg routes-per-expert / 32) == avg padded m-blocks/expert at block_m=32.
    m_blocks_32 = -(-routes // (int(experts) * 32))
    return 64 if m_blocks_32 == 2 else int(base_block_m)


def a16wi4_scale_to_kernel_layout(scale_ng):
    """Re-layout a logical groupwise int4 scale ``[E, N, num_groups]`` into the
    ``(E, N, num_groups//2, 2)`` bf16-pair layout the a16wi4 kernel expects.

    The kernel indexes the scale per-lane by N and reads bf16 pairs (two adjacent
    K-groups per dword): dword index ``= n*(G//2) + group//2``, even group -> low
    bf16, odd group -> high bf16. ``num_groups`` must be even. Input is already
    N-major (``[E, N, G]``); we just pack consecutive group pairs into the last dim.
    """
    E, N, G = scale_ng.shape
    assert G % 2 == 0, f"num_groups must be even for bf16-pair packing, got {G}"
    s = scale_ng.to(torch.bfloat16).contiguous().view(E, N, G // 2, 2).contiguous()
    return s


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
    w_dtype="mxfp4",
    use_csv_config=False,  # opt-in: default uses our tuned tile_n; CSV params for aiter-compare / when requested
    csv_path=None,
    stream=None,
):
    """a16w4/a16wi4/a16w16 fused stage1: gate+up GEMM + SiLU -> bf16 intermediate.

    ``w_dtype="mxfp4"`` (default): W1 is mxfp4 (``w1_scale_u8`` = shuffled e8m0).
    ``w_dtype="int4"`` (a16wi4): W1 is packed signed int4 (same preshuffle byte layout
    as mxfp4) and ``w1_scale_u8`` is the groupwise bf16 scale in the ``(E, N_OUT, G//2,
    2)`` kernel layout (see :func:`a16wi4_scale_to_kernel_layout`). ``w_dtype="bf16"``
    (a16w16): W1 is RAW bf16 ``[E, N_OUT, K]`` preshuffled with ``shuffle_weight
    (layout=(16,16))``; ``w1_scale_u8`` is unused (pass any pointer).

    ``a_bf16`` is the bf16 activation ``[n_tokens, D_HIDDEN]``. Writes the bf16
    intermediate ``[sorted_size, D_INTER]`` (by sorted position) into
    ``inter_sorted_bf16`` (pre-allocated). No A-scale, no intermediate scale.

    Tile config: ``tile_m`` -> BM, ``tile_n`` -> TILE_N, ``tile_k`` -> TILE_K,
    ``waves_per_eu`` -> ``rocdl.waves_per_eu``, ``b_nt`` -> W-load cache modifier
    (0=cached, 2=nt), ``xcd_swizzle`` -> XCD/HBM-channel grid remap, ``k_wave`` ->
    aiter intra-block slice-K (k_wave in {1,2,4}). ``k_batch`` (grid split-K) and
    ``gate_mode`` are accepted for parity but only ``k_batch=1`` /
    ``gate_mode="separated"`` are supported. ``tile_n=None`` picks the largest N tile
    dividing ``D_INTER``. ``b_nt=None`` uses the measured per-M U-shape (nt in the
    mid-band, cached at the ends), keyed on n_tokens.
    """
    if k_batch != 1:
        raise NotImplementedError(f"a16w4 gemm1 only supports k_batch=1, got {k_batch}")
    if gate_mode != "separated":
        raise NotImplementedError(
            f"a16w4 gemm1 only supports gate_mode='separated', got {gate_mode!r}"
        )

    # CSV-driven per-token config (mxfp4 only, opt-in): use aiter's tuned tile geometry
    # for the Kimi-K3 shapes. Falls back to the adaptive default when no CSV row
    # matches; explicit caller overrides (non-default args) win.
    if use_csv_config and w_dtype == "mxfp4":
        cfg = resolve_a16w4_gemm1_config(
            model_dim=D_HIDDEN,
            inter_dim=D_INTER,
            experts=NE,
            topk=topk,
            tokens=int(n_tokens),
            csv_path=csv_path,
        )
        if cfg is not None:
            if tile_n is None:
                tile_n = cfg["tile_n"]
            if tile_k == 256:
                tile_k = cfg["tile_k"]
            if k_wave == 1:
                k_wave = cfg["k_wave"]
            if waves_per_eu is None:
                waves_per_eu = cfg["waves_per_eu"]
            if xcd_swizzle == 0:
                xcd_swizzle = cfg["xcd_swizzle"]
            if b_nt is None:
                b_nt = cfg["b_nt"]

    BM = tile_m
    TILE_K = tile_k
    _m = int(n_tokens)
    b_cache_mod = (2 if (16 <= _m <= 1024) else 0) if b_nt is None else b_nt
    # mxfp4 high-token default (see worktree host.py): TILE_K=128 + xcd_swizzle=1 at
    # n_tokens>=16 is measured faster across the whole M>=16 range on both Kimi-K3
    # shapes; gated at >=16 (TILE_K=128 regresses tok 1..8). Only when the caller left
    # tile_k/xcd at defaults and not use_csv_config; mxfp4 only; needs K % 128 == 0.
    if w_dtype == "mxfp4" and not use_csv_config and _m >= 16 and D_HIDDEN % 128 == 0:
        if tile_k == 256:
            TILE_K = 128
        if xcd_swizzle == 0:
            xcd_swizzle = 1
    TILE_N = tile_n
    if TILE_N is None:
        # gemm1 mxfp4: the fat-wave tile_n=256 spills to VGPR=260 -> 1 wave/SIMD
        # (occupancy cliff), which fully exposes weight-load latency and worsens with
        # token count. tile_n=128 drops VGPR to 178 (2 waves/SIMD) and is measured
        # faster across the whole token range (1..16384) on the Kimi-K3 3584x512 shape
        # (e.g. s1 -7% @tok8192, -10% @tok16384) with no small-M regression. bf16
        # (a16w16, VGPR>=448 regardless) and int4 (already 128) keep _default_tile_n.
        if w_dtype == "mxfp4" and _m <= 2 and not use_csv_config and D_INTER % 64 == 0:
            # Very-low-M (tok 1..2): narrower TILE_N=64 under-fill fix (see worktree
            # host.py); ~0.86-0.89x gemm1-s1 at tok 1..2, gated to tok<=2 (regresses
            # from tok4 up).
            TILE_N = 64
            # ...and 4-way intra-block slice-K (TILE_K=128 + k_wave=4): at M=1 the K-loop
            # is LDS-wait-bound; splitting K 4 ways halves per-wave K-loop depth (~0.83x
            # s1 at tok1-2, matching aiter's tok1 kw4). Needs K % (4*128) == 0.
            if tile_k == 256 and k_wave == 1 and D_HIDDEN % 512 == 0:
                TILE_K = 128
                k_wave = 4
        elif w_dtype == "int4" and BM == 64 and D_INTER % 64 == 0:
            # int4 BM=64 fill point (a16wi4_recommend_block_m collapses two half-full
            # 32-row m-blocks into one 64-row block at tok~1600..3072): TILE_N=128 there
            # pushes VGPR to 349 + 93 AGPR -> 1 wave/SIMD (occupancy 12.3%). TILE_N=64
            # drops to 2 waves/SIMD (23.0%), no spill, ~0.77x gemm1-s1 across the band.
            # NET LOSS for the BM=32 mid band (already 2 waves), so gated to BM=64.
            TILE_N = 64
        elif w_dtype == "mxfp4" and D_INTER % 128 == 0:
            TILE_N = 128
        else:
            TILE_N = _default_tile_n(D_INTER, w_dtype=w_dtype)
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

    launch = _get_compiled_gemm1_a16w4(
        BM,
        D_HIDDEN,
        D_INTER,
        NE,
        topk,
        TILE_N,
        TILE_K,
        act,
        b_cache_mod,
        xcd_swizzle,
        waves_per_eu,
        w_dtype,
        k_wave,
    )
    max_m_blocks = int(sorted_expert_ids.numel())
    grid = gemm1_a16w4_grid(BM, INTER=D_INTER, TILE_N=TILE_N, max_m_blocks=max_m_blocks)
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
    w_dtype="mxfp4",
    use_csv_config=False,  # opt-in: default uses our tuned tile_n; CSV params for aiter-compare / when requested
    csv_path=None,
    persist=None,
    stream=None,
):
    """a16w4/a16wi4/a16w16 fused stage2 (down-proj). Consumes the bf16 [sorted_size,
    D_INTER] intermediate; scatters routing-weighted bf16 into ``flat_out``.

    Tile config: ``tile_m`` -> BM, ``tile_n`` (model_dim N tile) -> TILE_N, ``tile_k``
    (inter K tile) -> TILE_K, ``waves_per_eu`` -> ``rocdl.waves_per_eu``, ``b_nt`` ->
    W-load cache modifier, ``xcd_swizzle`` -> XCD/HBM-channel grid remap. ``k_batch``
    is accepted for parity (must be 1). ``b_nt=None`` keeps the measured per-M U-shape
    (cached at both ends, nt in the middle band).
    """
    if k_batch != 1:
        raise NotImplementedError(f"a16w4 gemm2 only supports k_batch=1, got {k_batch}")

    # CSV-driven per-token config (mxfp4 only, opt-in). Falls back to the adaptive
    # default when no CSV row matches or violates a divisibility constraint; explicit
    # caller overrides win.
    if use_csv_config and w_dtype == "mxfp4":
        cfg = resolve_a16w4_gemm2_config(
            model_dim=D_HIDDEN,
            inter_dim=D_INTER,
            experts=NE,
            topk=topk,
            tokens=int(M_logical),
            csv_path=csv_path,
        )
        if cfg is not None:
            if tile_n == 256 and D_HIDDEN % cfg["tile_n"] == 0:
                tile_n = cfg["tile_n"]
            if tile_k == 256:
                tile_k = cfg["tile_k"]
            if b_nt is None:
                b_nt = cfg["b_nt"]
            if xcd_swizzle == 1:
                xcd_swizzle = cfg["xcd_swizzle"]

    BM = tile_m
    TILE_N = tile_n
    if TILE_N is None:
        # Match gemm1's adaptive default: largest supported N tile dividing the
        # model_dim (down-proj output N). int4 prefers the legacy tile_n=128
        # geometry on the small MoE shapes (see gemm2 tile-config note below).
        TILE_N = _default_tile_n(D_HIDDEN, w_dtype=w_dtype)
    TILE_K = tile_k
    if D_INTER % TILE_K != 0:
        raise NotImplementedError(
            f"a16w4 gemm2 requires D_INTER (K) % {TILE_K} == 0, got D_INTER={D_INTER}"
        )
    if D_HIDDEN % TILE_N != 0:
        raise NotImplementedError(
            f"a16w4 gemm2 requires D_HIDDEN (model_dim) % {TILE_N} == 0, got H={D_HIDDEN}"
        )

    # B (weight) cache modifier per-token: a measured U-shape. Cached (0) wins at both
    # ends (small M: B reused across few M-blocks; large M >= 2048: high L2 residency)
    # while non-temporal streaming (2) wins the mid-band (32..1024). Caller may override
    # via b_nt.
    _m = int(M_logical)
    _b_cache_mod = (0 if (_m <= 16 or _m >= 2048) else 2) if b_nt is None else b_nt
    max_m_blocks = int(sorted_expert_ids.numel())
    # Persistent CU-limited grid (opt-in, default OFF): cap the launch to ~NUM_CU CTAs
    # and loop over the real work-tiles per CTA. Measured to NOT close the E896 perf
    # gap (the padded launch's empty CTAs early-return for ~0 cost), so kept only as an
    # opt-in building block; byte-identical when off.
    _persist = False if persist is None else bool(persist)
    launch = _get_compiled_gemm2_a16w4(
        BM,
        NE,
        D_HIDDEN,
        D_INTER,
        TILE_N,
        TILE_K,
        _b_cache_mod,
        xcd_swizzle,
        waves_per_eu,
        w_dtype,
        _persist,
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


# =============================================================================
# aiter tuned-CSV config loader for bf16-A MoE. Reads aiter's tuned fmoe CSVs
# (kimik3_fp4 for a16w4, kimik2_i4 for a16wi4) and decodes each
# ``flydsl_moe{1,2}_abf16_w{fp4,int4,bf16}`` kernelName into a tile-config dict.
# Only the tile GEOMETRY is used (aiter's gemm bodies differ, so its latency
# columns are not comparable).
# =============================================================================

# kernelName tokens:  flydsl_moe{stage}_abf16_w{fmt}_bf16_t{m}x{n}x{k}
#   [_w{N}]=waves_per_eu  [_xcd{N}]=xcd_swizzle  [_bnt{N}]=b_nt  [_kw{N}]=k_wave
#   [_kb{N}]=k_batch (aiter grid split-K; we map it onto k_wave, see
#   _kwave_from_kbatch). Extra epilogue tokens are ignored for tile-config purposes.
_A16W4_TILE_RE = re.compile(r"_t(\d+)x(\d+)x(\d+)")
_A16W4_W_RE = re.compile(r"_w(\d+)")
_A16W4_XCD_RE = re.compile(r"_xcd(\d+)")
_A16W4_BNT_RE = re.compile(r"_bnt(\d+)")
_A16W4_KW_RE = re.compile(r"_kw(\d+)")
_A16W4_KB_RE = re.compile(r"_kb(\d+)")


def _kwave_from_kbatch(k_batch):
    """Map aiter's grid split-K (``_kb{N}``) onto our intra-block slice-K (k_wave).

    We have no grid split-K, so approximate aiter's split-K with k_wave in {2,4}:
    ``kb==2 -> kw2``, ``kb>2 -> kw4`` (kw only supports {1,2,4}). ``kb<=1 -> kw1``.
    """
    if k_batch <= 1:
        return 1
    return 2 if k_batch == 2 else 4


def _decode_a16w4_kname(kname):
    """Decode an ``abf16_w{fp4,int4,bf16}`` kernelName into a tile-config dict, or None."""
    m = _A16W4_TILE_RE.search(kname)
    if m is None:
        return None
    tile_m, tile_n, tile_k = int(m.group(1)), int(m.group(2)), int(m.group(3))
    w = _A16W4_W_RE.search(kname)
    xcd = _A16W4_XCD_RE.search(kname)
    bnt = _A16W4_BNT_RE.search(kname)
    kw = _A16W4_KW_RE.search(kname)
    kb = _A16W4_KB_RE.search(kname)
    k_batch = int(kb.group(1)) if kb else 1
    # An explicit _kw token wins; otherwise derive k_wave from aiter's split-K
    # (_kb). int4 rows are tuned with grid split-K (kb2/4/7/14), which we have no
    # equivalent of, so we replace it with the intra-block slice-K lever.
    k_wave = int(kw.group(1)) if kw else _kwave_from_kbatch(k_batch)
    return {
        "tile_m": tile_m,
        "tile_n": tile_n,
        "tile_k": tile_k,
        # b_nt default in aiter's namer is 2 when the token is absent (only _bnt0
        # / _bnt{!=2} are named); mirror that.
        "b_nt": int(bnt.group(1)) if bnt else 2,
        "waves_per_eu": int(w.group(1)) if w else None,
        "xcd_swizzle": int(xcd.group(1)) if xcd else 0,
        "k_wave": k_wave,
        "k_batch": k_batch,
    }


@functools.cache
def _load_a16w4_csv(csv_path):
    """Parse the tuned CSV into {(model_dim,inter,E,topk,stage,tokens): cfg}."""
    table = {}
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            try:
                key_shape = (
                    int(row["model_dim"]),
                    int(row["inter_dim"]),
                    int(row["expert"]),
                    int(row["topk"]),
                    int(row["token"]),
                )
            except (KeyError, ValueError):
                continue
            for stage, col in ((1, "kernelName1"), (2, "kernelName2")):
                kname = row.get(col, "")
                # bf16-A rows across all weight formats (fp4 / int4 / bf16). Only
                # the tile GEOMETRY is reused; aiter's gemm bodies differ.
                if not any(
                    w in kname for w in ("abf16_wfp4", "abf16_wint4", "abf16_wbf16")
                ):
                    continue
                cfg = _decode_a16w4_kname(kname)
                if cfg is not None:
                    table[key_shape + (stage,)] = cfg
    return table


def pick_a16w4_config(csv_path, *, model_dim, inter_dim, experts, topk, tokens, stage):
    """Return aiter's tuned tile-config for one (shape, tokens, stage), or None.

    Picks the exact ``tokens`` row if present, else the nearest tuned token
    (largest tuned token <= requested, or the smallest tuned token otherwise) for
    the shape+stage. ``stage`` is 1 (gemm1) or 2 (gemm2).
    """
    table = _load_a16w4_csv(csv_path)
    exact = table.get((model_dim, inter_dim, experts, topk, tokens, stage))
    if exact is not None:
        return exact
    cand = sorted(
        t
        for (md, i, e, k, t, s) in table
        if (md, i, e, k, s) == (model_dim, inter_dim, experts, topk, stage)
    )
    if not cand:
        return None
    le = [t for t in cand if t <= tokens]
    pick = le[-1] if le else cand[0]
    return table[(model_dim, inter_dim, experts, topk, pick, stage)]


# Candidate locations for aiter's tuned fp4 fmoe CSV (env override wins).
_A16W4_CSV_ENV = "FLYDSL_A16W4_TUNED_CSV"
_A16W4_CSV_CANDIDATES = (
    "/root/aiter/aiter/configs/model_configs/kimik3_fp4_tuned_fmoe.csv",
)


@functools.cache
def _default_a16w4_csv_path():
    """Locate aiter's tuned fp4 fmoe CSV, or None if not found.

    ``FLYDSL_A16W4_TUNED_CSV`` overrides the search. The CSV is used only as a
    source of candidate tile geometries for the Kimi-K3 a16w4 shapes.
    """
    env = os.environ.get(_A16W4_CSV_ENV)
    if env:
        return env if os.path.isfile(env) else None
    for p in _A16W4_CSV_CANDIDATES:
        if os.path.isfile(p):
            return p
    return None


def _kw_tile_k_for(cfg, *, K):
    """Return a (k_wave, tile_k, note) triple from ``cfg`` correct for contraction ``K``.

    The gemm1 builder requires ``K % (k_wave * tile_k) == 0``. aiter names its kw4 rows
    with tile_k=256 but relies on K-padding we don't have, so keep the CSV's k_wave and
    pick the largest tile_k in {256,128,64} that divides; if none works, drop k_wave to
    1. ``note`` is a short string when a fallback was applied, else "".
    """
    kw = int(cfg.get("k_wave", 1))
    tk = int(cfg["tile_k"])
    if kw == 1:
        return 1, tk, ""
    if K % (kw * tk) == 0:
        return kw, tk, ""
    for cand_tk in (256, 128, 64):
        if cand_tk <= tk and K % (kw * cand_tk) == 0:
            return kw, cand_tk, f"kw{kw}:tile_k {tk}->{cand_tk}"
    # No tile_k divides for this k_wave; fall back to no slice-K.
    return 1, tk, f"kw{kw}->1 (no divisible tile_k at K={K})"


def resolve_a16w4_gemm1_config(
    *, model_dim, inter_dim, experts, topk, tokens, csv_path=None
):
    """Resolve the per-token gemm1 tile-config for a shape from the tuned CSV.

    Returns a kwargs dict (tile_m, tile_n, tile_k, waves_per_eu, xcd_swizzle,
    b_nt, k_wave) plus a ``_note`` string, or None when no CSV is available or no
    row matches the shape (caller then uses the adaptive default). K=model_dim is
    the gemm1 contraction; the kw/tile_k pair is corrected for it.
    """
    path = csv_path or _default_a16w4_csv_path()
    if path is None:
        return None
    cfg = pick_a16w4_config(
        path,
        model_dim=model_dim,
        inter_dim=inter_dim,
        experts=experts,
        topk=topk,
        tokens=tokens,
        stage=1,
    )
    if cfg is None:
        return None
    # gemm1 requires inter_dim % tile_n == 0; skip CSV tile_n if it does not divide.
    tile_n = int(cfg["tile_n"])
    if inter_dim % tile_n != 0:
        return None
    kw, tile_k, note = _kw_tile_k_for(cfg, K=model_dim)
    return {
        "tile_m": int(cfg["tile_m"]),
        "tile_n": tile_n,
        "tile_k": tile_k,
        "waves_per_eu": cfg.get("waves_per_eu"),
        "xcd_swizzle": int(cfg.get("xcd_swizzle", 0)),
        "b_nt": int(cfg["b_nt"]),
        "k_wave": kw,
        "_note": note,
    }


def resolve_a16w4_gemm2_config(
    *, model_dim, inter_dim, experts, topk, tokens, csv_path=None
):
    """Resolve the per-token gemm2 tile-config for a shape from the tuned CSV.

    gemm2 has no k_wave (fixed 4-wave N-split); the CSV rows are all k_wave=1.
    gemm2 requires D_INTER (K) % tile_k == 0 and model_dim (N) % tile_n == 0; a
    CSV row violating either is skipped (None -> adaptive default).
    """
    path = csv_path or _default_a16w4_csv_path()
    if path is None:
        return None
    cfg = pick_a16w4_config(
        path,
        model_dim=model_dim,
        inter_dim=inter_dim,
        experts=experts,
        topk=topk,
        tokens=tokens,
        stage=2,
    )
    if cfg is None:
        return None
    tile_n = int(cfg["tile_n"])
    tile_k = int(cfg["tile_k"])
    if model_dim % tile_n != 0 or inter_dim % tile_k != 0:
        return None
    return {
        "tile_m": int(cfg["tile_m"]),
        "tile_n": tile_n,
        "tile_k": tile_k,
        "waves_per_eu": cfg.get("waves_per_eu"),
        "xcd_swizzle": int(cfg.get("xcd_swizzle", 1)),
        "b_nt": int(cfg["b_nt"]),
        "_note": "",
    }
