"""FP8/FP8 + per-1x128 FP32 blockscale MoE GEMM stage1/stage2 (FlyDSL port).

Thin **adapter** over the upstream FlyDSL implementation in
``_blockscale_moe_gemm_2stage_upstream.py``. The upstream file is a direct
port (import-rewritten copy) of ``FlyDSL/kernels/moe_blockscale_2stage.py``
on the ROCm/FlyDSL repo (introduced in PR #164, FP8-enabled in PR #252,
MI355-tuned in PR #306).

This adapter exposes the wider aiter dispatcher signature expected by
``aiter.ops.flydsl.moe_kernels.compile_flydsl_moe_stage{1,2}`` and maps
it down to the narrower upstream signature.

CK feature surface coverage in this adapter (matched against
``GridwiseMoeGemmBlockScale``):

  Tier A  (supported, forwarded to upstream):
    - out_dtype in {"bf16", "f16"/"fp16"}
    - scale_block_k (default 128, configurable)
    - waves_per_eu (default 2 on gfx950)
    - accumulate (stage2, atomic add vs reduce wrapper)
    - scale_block_m=1, scale_block_n=128 (CK production defaults)

  Tier B  (validated; required to be the default for DSR1 / TP=8 prefill):
    - act in {"silu", "gelu"}  (gelu uses exact erf, CK parity)
    - enable_bias is False     (no bias path in upstream blockscale yet)
    - model_dim_pad == 0       (clean shapes only)
    - inter_dim_pad == 0
    - swiglu_limit == 0.0      (no SiLU clipping yet)
    - sort_block_m: forwarded as a no-op (upstream encodes its own tile_m)

  Tier C  (raises NotImplementedError until upstream grows the knob):
    - act not in {"silu", "gelu"}
    - enable_bias=True
    - persist_m > 1             (no persistent kernel)
    - use_async_copy=False      (upstream always uses async/buffer copy)
    - b_nt != default            (non-temporal B load — TODO)
    - xcd_swizzle != 0           (XCD remap — TODO)
    - any non-clean shape paddings

Future commits in this PR will move ``xcd_swizzle`` and ``b_nt`` from
Tier C to Tier B with small upstream extensions.
"""

from __future__ import annotations

import functools
import logging

from ._blockscale_moe_gemm_2stage_upstream import (
    compile_moe_blockscale_gemm1 as _upstream_compile_gemm1,
    compile_moe_blockscale_gemm2 as _upstream_compile_gemm2,
)

logger = logging.getLogger(__name__)


# CK-aligned defaults from gridwise_moe_gemm_blockscale.hpp:
#   ScaleBlockM=1, ScaleBlockN=128, ScaleBlockK=128.
SCALE_BLOCK_M_DEFAULT = 1
SCALE_BLOCK_N_DEFAULT = 128
SCALE_BLOCK_K_DEFAULT = 128


# ---------------------------------------------------------------------------
# Argument validation helpers
# ---------------------------------------------------------------------------
def _normalize_out_dtype(out_dtype: str) -> str:
    """Map aiter's out_dtype string to the upstream FlyDSL convention.

    aiter passes ``"bf16"``, ``"f16"``, or ``"fp16"``. Upstream expects
    ``"bf16"`` or ``"f16"``. (Stage2 also supports ``"f32"``.)
    """
    s = str(out_dtype).strip().lower()
    if s in ("f16", "fp16", "half"):
        return "f16"
    if s in ("bf16", "bfloat16"):
        return "bf16"
    if s in ("f32", "fp32", "float"):
        return "f32"
    raise ValueError(
        f"blockscale: unsupported out_dtype={out_dtype!r}; "
        "expected one of {'bf16','f16','fp16','f32'}"
    )


def _validate_blockscale_dtypes(a_dtype: str, b_dtype: str) -> None:
    if a_dtype != "fp8" or b_dtype != "fp8":
        raise ValueError(
            "blockscale: only a_dtype='fp8' and b_dtype='fp8' are supported "
            f"(got a_dtype={a_dtype!r}, b_dtype={b_dtype!r}). "
            "For other dtype combinations use compile_mixed_moe_gemm* or "
            "compile_moe_gemm* instead."
        )


def _validate_split_k(
    *, k_batch: int, model_dim: int, tile_k: int, scale_block_k: int, stage: str
) -> None:
    """Validate split-K constraints for stage1.

    Per the split-K plan (Steps 1-5):
      1. K-slice must be divisible by scale_block_k (each CTA processes whole
         scale blocks in K).
      2. K-slice must be divisible by tile_k (no per-CTA tail-tile handling).
    Violations raise ValueError so the caller can fall back to k_batch=1.
    """
    if k_batch <= 0:
        raise ValueError(f"blockscale {stage}: k_batch={k_batch} must be >= 1")
    if k_batch == 1:
        return
    if model_dim % k_batch != 0:
        raise ValueError(
            f"blockscale {stage}: model_dim={model_dim} not divisible by "
            f"k_batch={k_batch}"
        )
    k_per_split = model_dim // k_batch
    if k_per_split % scale_block_k != 0:
        raise ValueError(
            f"blockscale {stage}: k_per_split={k_per_split} (model_dim/k_batch) "
            f"not divisible by scale_block_k={scale_block_k}"
        )
    if k_per_split % tile_k != 0:
        raise ValueError(
            f"blockscale {stage}: k_per_split={k_per_split} not divisible by "
            f"tile_k={tile_k}"
        )
    # The main K-loop processes pairs of tile_k tiles + a 2-tile tail. That
    # requires total_tiles_per_split (=k_per_split/tile_k) to be even and >= 2.
    if (k_per_split // tile_k) < 2 or ((k_per_split // tile_k) % 2) != 0:
        raise ValueError(
            f"blockscale {stage}: k_per_split={k_per_split} / tile_k={tile_k} "
            f"= {k_per_split // tile_k} must be even and >= 2 "
            "(K-loop unrolls in pairs of tile_k tiles)"
        )


def _validate_scale_blocks(
    scale_block_m: int, scale_block_n: int, scale_block_k: int
) -> int:
    """Validate scale-block dims and return the K-block size to forward."""
    if scale_block_m != SCALE_BLOCK_M_DEFAULT:
        raise NotImplementedError(
            f"blockscale: scale_block_m={scale_block_m} not supported "
            f"(only {SCALE_BLOCK_M_DEFAULT} matches CK production layout)"
        )
    if scale_block_n != SCALE_BLOCK_N_DEFAULT:
        raise NotImplementedError(
            f"blockscale: scale_block_n={scale_block_n} not supported "
            f"(only {SCALE_BLOCK_N_DEFAULT} matches CK production layout)"
        )
    if scale_block_k <= 0 or (scale_block_k % 64) != 0:
        raise ValueError(
            f"blockscale: scale_block_k={scale_block_k} must be a positive "
            "multiple of 64"
        )
    return int(scale_block_k)


def _splitk_kslice_is_valid(
    k_batch: int, model_dim: int, tile_k: int, scale_block_k: int = SCALE_BLOCK_K_DEFAULT
) -> bool:
    """Cheap predicate version of ``_validate_split_k`` (no exceptions).

    Returns True iff a ``k_batch`` value satisfies the K-slice constraints
    (model_dim divisible by k_batch, slice divisible by scale_block_k and
    tile_k, and an even number of tile_k tiles per slice >= 2).
    """
    if k_batch <= 1:
        return True
    if model_dim % k_batch != 0:
        return False
    kps = model_dim // k_batch
    if kps % scale_block_k != 0:
        return False
    if kps % tile_k != 0:
        return False
    ttps = kps // tile_k
    return ttps >= 2 and (ttps % 2) == 0


def pick_k_batch_for_blockscale_stage1(
    *,
    token_num: int,
    inter_dim: int,
    topk: int,
    model_dim: int,
    tile_m: int,
    tile_n: int,
    tile_k: int,
    scale_block_k: int = SCALE_BLOCK_K_DEFAULT,
) -> int:
    """Pick ``k_batch`` for FP8 blockscale MoE stage1 from grid occupancy.

    Tuned on DSR1 TP=8 shapes (``model_dim=7168``, ``inter_dim`` in
    ``{256, 512}``, ``E=257``, ``topk=9``) on gfx950. Selector picks from
    ``{1, 2, 4}`` only -- larger split factors are uniformly worse on the
    measured shapes (atomic-add contention dominates).

    Heuristic:
      blocks_m == 1 (decode-1: only one M-tile)   -> k_batch = 4
      base_ctas >= 16 (grid already fills the GPU) -> k_batch = 1
      base_ctas >=  8                              -> k_batch = 2
      base_ctas <   8                              -> k_batch = 4

    where ``blocks_m = ceil(M*topk / tile_m)`` and
    ``base_ctas = blocks_m * ceil(2*inter_dim / tile_n)``.

    The chosen value is then snapped down to the largest valid ``k_batch
    <= desired`` that satisfies the K-slice constraints; otherwise falls
    back to ``1``. Safe to call for any shape -- never raises.
    """
    blocks_m = max(1, (token_num * topk + tile_m - 1) // tile_m)
    blocks_n = max(1, (2 * inter_dim + tile_n - 1) // tile_n)
    base_ctas = blocks_m * blocks_n
    if blocks_m == 1 and base_ctas < 16:
        # Only one M-tile (decode-1 case): occupancy is most starved on
        # the M dimension regardless of N; split-K=4 is uniformly best
        # in the measured sweep.
        desired = 4
    elif base_ctas >= 16:
        desired = 1
    elif base_ctas >= 8:
        desired = 2
    else:
        desired = 4
    for kb in (desired, max(desired // 2, 1), 1):
        if _splitk_kslice_is_valid(kb, model_dim, tile_k, scale_block_k):
            return kb
    return 1


def _reject_tier_c(
    *,
    act: str,
    enable_bias: bool,
    model_dim_pad: int,
    inter_dim_pad: int,
    swiglu_limit: float,
    persist_m: int,
    use_async_copy: bool,
    b_nt: int,
    xcd_swizzle: int,
    stage: str,
) -> None:
    """Tier-B/C gate. Raises NotImplementedError for unsupported requests.

    The DSR1 / TP=8 prefill production path uses only the default values
    for every knob below; any deviation today means upstream has to grow
    that knob first. Each branch points the caller at the future-work item.
    """
    if act not in ("silu", "gelu"):
        raise NotImplementedError(
            f"blockscale {stage}: act={act!r} not supported " "(only 'silu' or 'gelu')"
        )
    if enable_bias:
        raise NotImplementedError(
            f"blockscale {stage}: enable_bias=True not supported "
            "(upstream FlyDSL blockscale has no bias path yet — Tier C)"
        )
    if model_dim_pad != 0:
        raise NotImplementedError(
            f"blockscale {stage}: model_dim_pad={model_dim_pad} != 0 "
            "(clean shapes only — Tier C)"
        )
    if inter_dim_pad != 0:
        raise NotImplementedError(
            f"blockscale {stage}: inter_dim_pad={inter_dim_pad} != 0 "
            "(clean shapes only — Tier C)"
        )
    if swiglu_limit != 0.0:
        raise NotImplementedError(
            f"blockscale {stage}: swiglu_limit={swiglu_limit} != 0.0 "
            "(no clipping in upstream SiLU — Tier C)"
        )
    if persist_m != 1:
        raise NotImplementedError(
            f"blockscale {stage}: persist_m={persist_m} != 1 "
            "(persistent kernel not supported in upstream blockscale — Tier C)"
        )
    # use_async_copy: upstream FlyDSL blockscale always uses async/buffer
    # copy regardless of this flag, so we accept either value as a no-op.
    # (The aiter dispatcher defaults to False; honoring it would change
    # nothing in the generated kernel.)
    del use_async_copy  # noqa: F841 — explicitly mark as intentionally unused
    # b_nt: upstream doesn't honor this knob yet; accept the dispatcher's
    # default (mixed-path default is 2 for stage1, 0 for stage2) silently
    # so the dispatcher doesn't need to know; but flag non-default values.
    if b_nt not in (0, 2):
        raise NotImplementedError(
            f"blockscale {stage}: b_nt={b_nt} not supported "
            "(non-temporal B load hint — Tier C)"
        )
    if xcd_swizzle != 0:
        raise NotImplementedError(
            f"blockscale {stage}: xcd_swizzle={xcd_swizzle} != 0 "
            "(XCD-aware workgroup remap — Tier C)"
        )


# ---------------------------------------------------------------------------
# Stage 1
# ---------------------------------------------------------------------------
@functools.lru_cache(maxsize=1024)
def compile_blockscale_moe_gemm1(
    *,
    model_dim: int,
    inter_dim: int,
    experts: int,
    topk: int,
    tile_m: int,
    tile_n: int,
    tile_k: int,
    doweight_stage1: bool,
    a_dtype: str = "fp8",
    b_dtype: str = "fp8",
    out_dtype: str = "bf16",
    act: str = "silu",
    enable_bias: bool = False,
    model_dim_pad: int = 0,
    inter_dim_pad: int = 0,
    persist_m: int = 1,
    use_async_copy: bool = True,
    waves_per_eu: int = 3,
    k_batch: int = 1,
    b_nt: int = 2,
    xcd_swizzle: int = 0,
    swiglu_limit: float = 0.0,
    scale_block_m: int = SCALE_BLOCK_M_DEFAULT,
    scale_block_n: int = SCALE_BLOCK_N_DEFAULT,
    scale_block_k: int = SCALE_BLOCK_K_DEFAULT,
):
    """Stage1 FP8 blockscale MoE GEMM (aiter dispatcher entry).

    Forwards to upstream ``compile_moe_blockscale_gemm1`` after validating
    the wide aiter signature against what upstream currently supports.

    See module docstring for tier breakdown of supported kwargs.
    """
    _validate_blockscale_dtypes(a_dtype, b_dtype)
    sbk = _validate_scale_blocks(scale_block_m, scale_block_n, scale_block_k)
    _reject_tier_c(
        act=act,
        enable_bias=enable_bias,
        model_dim_pad=model_dim_pad,
        inter_dim_pad=inter_dim_pad,
        swiglu_limit=swiglu_limit,
        persist_m=persist_m,
        use_async_copy=use_async_copy,
        b_nt=b_nt,
        xcd_swizzle=xcd_swizzle,
        stage="stage1",
    )
    _validate_split_k(
        k_batch=int(k_batch),
        model_dim=int(model_dim),
        tile_k=int(tile_k),
        scale_block_k=sbk,
        stage="stage1",
    )

    upstream_out = _normalize_out_dtype(out_dtype)
    if upstream_out == "f32":
        raise ValueError("blockscale stage1: out_dtype='f32' is only valid for stage2")

    # waves_per_eu == 0 (or negative) means "let the backend decide" in some
    # aiter callers. Forward None to upstream in that case so it can apply
    # its own default (2 on gfx950).
    wpe = int(waves_per_eu) if waves_per_eu and waves_per_eu > 0 else None

    return _upstream_compile_gemm1(
        model_dim=model_dim,
        inter_dim=inter_dim,
        experts=experts,
        topk=topk,
        tile_m=tile_m,
        tile_n=tile_n,
        tile_k=tile_k,
        doweight_stage1=doweight_stage1,
        scale_block_k=sbk,
        out_dtype=upstream_out,
        waves_per_eu=wpe,
        act=act,
        k_batch=int(k_batch),
    )


# ---------------------------------------------------------------------------
# Stage 2
# ---------------------------------------------------------------------------
@functools.lru_cache(maxsize=1024)
def compile_blockscale_moe_gemm2(
    *,
    model_dim: int,
    inter_dim: int,
    experts: int,
    topk: int,
    tile_m: int,
    tile_n: int,
    tile_k: int,
    doweight_stage2: bool,
    a_dtype: str = "fp8",
    b_dtype: str = "fp8",
    out_dtype: str = "bf16",
    accumulate: bool = True,
    persist_m: int = 1,
    sort_block_m: int = 0,
    b_nt: int = 0,
    model_dim_pad: int = 0,
    inter_dim_pad: int = 0,
    xcd_swizzle: int = 0,
    enable_bias: bool = False,
    scale_block_m: int = SCALE_BLOCK_M_DEFAULT,
    scale_block_n: int = SCALE_BLOCK_N_DEFAULT,
    scale_block_k: int = SCALE_BLOCK_K_DEFAULT,
):
    """Stage2 FP8 blockscale MoE GEMM (aiter dispatcher entry).

    Forwards to upstream ``compile_moe_blockscale_gemm2`` after validating
    the wide aiter signature. Stage2 has no activation, no swiglu_limit,
    no k_batch, no use_async_copy in the upstream signature.
    """
    _validate_blockscale_dtypes(a_dtype, b_dtype)
    sbk = _validate_scale_blocks(scale_block_m, scale_block_n, scale_block_k)
    # Stage2 doesn't have act/swiglu_limit/k_batch/use_async_copy; pass
    # defaults so _reject_tier_c can stay one function.
    _reject_tier_c(
        act="silu",
        enable_bias=enable_bias,
        model_dim_pad=model_dim_pad,
        inter_dim_pad=inter_dim_pad,
        swiglu_limit=0.0,
        persist_m=persist_m,
        use_async_copy=True,
        b_nt=b_nt,
        xcd_swizzle=xcd_swizzle,
        stage="stage2",
    )
    # sort_block_m: upstream stage2 derives this from tile_m + sorted_ids,
    # so we accept any value and ignore it. Log if it disagrees with tile_m.
    if sort_block_m not in (0, tile_m):
        logger.warning(
            "blockscale stage2: sort_block_m=%d != tile_m=%d; upstream "
            "ignores sort_block_m and uses tile_m. Ensure the caller's "
            "moe_sorting was invoked with block_m=tile_m.",
            sort_block_m,
            tile_m,
        )

    upstream_out = _normalize_out_dtype(out_dtype)

    return _upstream_compile_gemm2(
        model_dim=model_dim,
        inter_dim=inter_dim,
        experts=experts,
        topk=topk,
        tile_m=tile_m,
        tile_n=tile_n,
        tile_k=tile_k,
        doweight_stage2=doweight_stage2,
        scale_block_k=sbk,
        out_dtype=upstream_out,
        accumulate=bool(accumulate),
    )


__all__ = [
    "compile_blockscale_moe_gemm1",
    "compile_blockscale_moe_gemm2",
    "pick_k_batch_for_blockscale_stage1",
    "SCALE_BLOCK_M_DEFAULT",
    "SCALE_BLOCK_N_DEFAULT",
    "SCALE_BLOCK_K_DEFAULT",
]
