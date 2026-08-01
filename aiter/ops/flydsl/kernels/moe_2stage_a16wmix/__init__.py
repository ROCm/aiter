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

import csv
import functools
import os
import re

import torch

from aiter.ops.flydsl.kernels.tensor_shim import _run_compiled

from .gemm1 import compile_gemm1_a16w4_port, gemm1_a16w4_grid
from .gemm2 import compile_gemm2_a16w4_port, gemm2_a16w4_grid

__all__ = [
    "a16wi4_recommend_block_m",
    "a16wi4_scale_to_kernel_layout",
    "compile_gemm1_a16w4_port",
    "compile_gemm2_a16w4_port",
    "flydsl_a16w4_gemm1",
    "flydsl_a16w4_gemm2",
    "gemm1_a16w4_grid",
    "gemm2_a16w4_grid",
    "pick_a16w4_config",
    "resolve_a16w4_gemm1_config",
    "resolve_a16w4_gemm2_config",
]


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
    """Adaptive N-tile default. mxfp4/bf16: 256 when N % 256 == 0 else 128 (aiter fat
    tile). int4 (bandwidth/grid-fill-bound): prefers 128, falling to 64 when N % 128 != 0.
    """
    if w_dtype == "int4":
        if N % 128 == 0:
            return 128
        return 64 if N % 64 == 0 else 128
    return 256 if N % 256 == 0 else 128


def a16wi4_recommend_block_m(tokens, experts, topk, *, base_block_m=32):
    """Recommend the routing/gemm1 block_m (tile_m) for the a16wi4 (int4-W) stage1.

    int4 gemm1 is W-load bound (W1 re-fetched per padded m-block). At the fill point
    where each expert has exactly two half-full 32-row m-blocks, block_m=64 collapses
    them into one full 64-row block, halving W1 HBM re-reads (~2.27x fewer misses, ~7%
    stage1). Outside that band 64 wastes padding or halves grid parallelism.

    Decides on ceil(tokens*topk/experts / base_block_m) (avg padded m-blocks/expert ==
    aiter's estimated_m_per_expert): return 2*base_block_m iff that count == 2.
    base_block_m==32 only; other block_m pass through.

    IMPORTANT: block_m sizes the moe_sorting padding, so the caller MUST build the
    routing buffers with the SAME block_m passed to gemm1 (this is a dispatcher-side
    recommendation, not a gemm1-internal override).
    """
    if int(base_block_m) != 32 or int(experts) <= 0:
        return int(base_block_m)
    routes = int(tokens) * int(topk)
    # ceil(avg routes-per-expert / 32) == avg padded m-blocks/expert at block_m=32.
    m_blocks_32 = -(-routes // (int(experts) * 32))
    return 64 if m_blocks_32 == 2 else int(base_block_m)


def a16wi4_scale_to_kernel_layout(scale_ng):
    """Re-layout a logical int4 scale ``[E, N, G]`` into the ``(E, N, G//2, 2)``
    bf16-pair layout the kernel expects (dword = n*(G//2) + group//2, even/odd group ->
    lo/hi bf16). ``G`` must be even; input is already N-major.
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

    ``w_dtype="mxfp4"`` (default): W1 mxfp4, ``w1_scale_u8`` = shuffled e8m0. ``"int4"``:
    W1 packed signed int4 (same preshuffle as mxfp4), ``w1_scale_u8`` groupwise bf16 in
    the ``(E, N_OUT, G//2, 2)`` layout (see :func:`a16wi4_scale_to_kernel_layout`).
    ``"bf16"``: RAW bf16 W1 preshuffled ``shuffle_weight (16,16)``; ``w1_scale_u8`` unused.

    ``a_bf16`` is bf16 ``[n_tokens, D_HIDDEN]``. Writes the bf16 intermediate
    ``[sorted_size, D_INTER]`` (by sorted position) into ``inter_sorted_bf16``.

    Tile config: ``tile_m/n/k`` -> BM/TILE_N/TILE_K, ``waves_per_eu`` ->
    rocdl.waves_per_eu, ``b_nt`` -> W-load cache modifier (0=cached, 2=nt),
    ``xcd_swizzle`` -> XCD/HBM grid remap, ``k_wave`` -> intra-block slice-K ({1,2,4}).
    ``k_batch``/``gate_mode`` accepted for parity (only k_batch=1/separated supported).
    ``tile_n=None`` picks the largest N tile dividing D_INTER. ``b_nt=None`` uses the
    per-M U-shape (nt mid-band, cached at ends).
    """
    if k_batch != 1:
        raise NotImplementedError(f"a16w4 gemm1 only supports k_batch=1, got {k_batch}")
    if gate_mode != "separated":
        raise NotImplementedError(
            f"a16w4 gemm1 only supports gate_mode='separated', got {gate_mode!r}"
        )

    # CSV-driven per-token config (mxfp4 only, opt-in): aiter's tuned tile geometry.
    # Falls back to adaptive default on no match; explicit caller overrides win.
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
    # mxfp4 high-token defaults (only when tile_k/xcd_swizzle left at defaults): at
    # n_tokens >= 16, TILE_K=128 (shorter K-tiles) + xcd_swizzle=1 is faster across M>=16.
    # Gated at >=16 (TILE_K=128 regresses launch-bound tok 1..8). mxfp4 only, K % 128 == 0.
    if w_dtype == "mxfp4" and not use_csv_config and _m >= 16 and D_HIDDEN % 128 == 0:
        if tile_k == 256:
            TILE_K = 128
        if xcd_swizzle == 0:
            xcd_swizzle = 1
    TILE_N = tile_n
    if TILE_N is None:
        # gemm1 mxfp4: fat tile_n=256 spills to VGPR=260 (1 wave/SIMD, exposes W-load
        # latency); tile_n=128 -> VGPR 178 (2 waves/SIMD), faster across all M with no
        # small-M regression. bf16 (VGPR>=448) and int4 (already 128) keep _default_tile_n.
        if w_dtype == "mxfp4" and _m <= 2 and not use_csv_config and D_INTER % 64 == 0:
            # Very-low-M (tok 1..2): E896 has ~topk m-blocks of work (GPU under-filled).
            # TILE_N=64 doubles CTAs/m-block (better latency hiding); regresses from tok4.
            TILE_N = 64
            # ...+ 4-way slice-K (TILE_K=128 + k_wave=4): the M=1 K-loop is LDS-wait-bound;
            # splitting K 4 ways halves per-wave K-depth. K % (4*128) == 0 required (3584
            # fails K % 1024 so TILE_K=128, not 256). Only when tile_k/k_wave at defaults.
            if tile_k == 256 and k_wave == 1 and D_HIDDEN % 512 == 0:
                TILE_K = 128
                k_wave = 4
        elif w_dtype == "int4" and BM == 64 and D_INTER % 64 == 0:
            # int4 BM=64 (tok~1600..3072 W1-reuse fill point): TILE_N=128 spills to 1
            # wave/SIMD (occupancy cliff); TILE_N=64 -> 2 waves/SIMD (no spill), ~0.77x
            # across the BM=64 band. Net loss for the BM=32 mid band, so BM=64-gated.
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

    Tile config: ``tile_m/n/k`` -> BM/TILE_N/TILE_K, ``waves_per_eu`` ->
    rocdl.waves_per_eu, ``b_nt`` -> W-load cache modifier, ``xcd_swizzle`` -> XCD/HBM
    grid remap. ``k_batch`` for parity (must be 1). ``b_nt=None`` keeps the per-M
    U-shape (cached at ends, nt mid-band).
    """
    if k_batch != 1:
        raise NotImplementedError(f"a16w4 gemm2 only supports k_batch=1, got {k_batch}")

    # CSV-driven per-token config (mxfp4 only, opt-in). Falls back to adaptive default
    # on no match / divisibility violation; explicit caller overrides win.
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
        # Adaptive default: largest N tile dividing model_dim (int4 prefers 128).
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

    # B cache modifier per-token U-shape: cached (0) at both ends (small M reuse / large
    # M L2 residency), nt (2) mid-band (32..1024). Caller may override via b_nt.
    _m = int(M_logical)
    _b_cache_mod = (0 if (_m <= 16 or _m >= 2048) else 2) if b_nt is None else b_nt
    max_m_blocks = int(sorted_expert_ids.numel())
    # Persistent CU-limited grid (opt-in, default OFF; byte-identical when off): does NOT
    # close the E896 gap (padded launch's empty CTAs early-return ~free), kept as an
    # opt-in building block.
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
# aiter tuned-CSV config loader for bf16-A MoE. Decodes each
# ``flydsl_moe{1,2}_abf16_w{fp4,int4,bf16}`` kernelName into a tile-config dict.
# Only the tile GEOMETRY is used (aiter's gemm bodies differ).
# =============================================================================

# kernelName tokens:  flydsl_moe{stage}_abf16_w{fmt}_bf16_t{m}x{n}x{k}
#   [_w{N}]=waves_per_eu [_xcd{N}]=xcd_swizzle [_bnt{N}]=b_nt [_kw{N}]=k_wave
#   [_kb{N}]=k_batch (aiter grid split-K, mapped onto k_wave; see _kwave_from_kbatch).
_A16W4_TILE_RE = re.compile(r"_t(\d+)x(\d+)x(\d+)")
_A16W4_W_RE = re.compile(r"_w(\d+)")
_A16W4_XCD_RE = re.compile(r"_xcd(\d+)")
_A16W4_BNT_RE = re.compile(r"_bnt(\d+)")
_A16W4_KW_RE = re.compile(r"_kw(\d+)")
_A16W4_KB_RE = re.compile(r"_kb(\d+)")


def _kwave_from_kbatch(k_batch):
    """Map aiter's grid split-K (``_kb{N}``) onto intra-block slice-K: kb<=1 -> 1,
    kb==2 -> 2, kb>2 -> 4 (k_wave only supports {1,2,4})."""
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
    # Explicit _kw wins; else derive k_wave from aiter's split-K (_kb).
    k_wave = int(kw.group(1)) if kw else _kwave_from_kbatch(k_batch)
    return {
        "tile_m": tile_m,
        "tile_n": tile_n,
        "tile_k": tile_k,
        "b_nt": int(bnt.group(1)) if bnt else 2,  # aiter default 2 when token absent
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
                # bf16-A rows across all weight formats (fp4/int4/bf16); tile geometry only.
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

    Exact ``tokens`` row if present, else nearest tuned token (largest <= requested,
    or smallest). ``stage`` is 1 (gemm1) or 2 (gemm2).
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
    """Locate aiter's tuned fp4 fmoe CSV (``FLYDSL_A16W4_TUNED_CSV`` overrides), or None."""
    env = os.environ.get(_A16W4_CSV_ENV)
    if env:
        return env if os.path.isfile(env) else None
    for p in _A16W4_CSV_CANDIDATES:
        if os.path.isfile(p):
            return p
    return None


def _kw_tile_k_for(cfg, *, K):
    """Return (k_wave, tile_k, note) from ``cfg`` correct for contraction ``K``.

    gemm1 requires ``K % (k_wave * tile_k) == 0``. Keep the CSV's k_wave and pick the
    largest tile_k in {256,128,64} that divides; if none works, drop k_wave to 1.
    ``note`` is set when a fallback was applied.
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
    """Resolve the per-token gemm1 tile-config from the tuned CSV.

    Returns a kwargs dict (+ ``_note``), or None when no CSV/row matches (caller uses
    adaptive default). K=model_dim; the kw/tile_k pair is corrected for it.
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
    """Resolve the per-token gemm2 tile-config from the tuned CSV.

    gemm2 has no k_wave (fixed 4-wave N-split). Requires D_INTER % tile_k == 0 and
    model_dim % tile_n == 0; a row violating either is skipped (None -> adaptive default).
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
