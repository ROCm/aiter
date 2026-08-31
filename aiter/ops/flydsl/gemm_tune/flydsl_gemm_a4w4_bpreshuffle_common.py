# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Tune space for the FlyDSL a4w4 (mxfp4) bpreshuffle split-K pipeline, gfx950 only.

Sibling of ``flydsl_gemm_a8w8_bpreshuffle_common.py``, ported for the split-K
family only -- a4w4 has no non-split ``preshuffle``/``8wave`` pipeline yet.
Same ``{kernelId: instance}`` + ``fits(ki, M, N, K)`` pair, listed in
``PIPELINES`` so a future ``--libtype flydsl`` a4w4 tuner run can iterate it the
same way the a8w8 tuner iterates its ``PIPELINES``.

Quant contract differs from a8w8: both operands are packed 4-bit (fp4) codes,
two codes per byte, so a tile's storage footprint is half the fp8/int8 case;
scales are 32-block E8M0 (``scale_block_k=32``), not the 128-block fp32
blockscale a8w8 uses. See ``flydsl_preshuffle_gemm_splitk_a8(..., in_dtype="fp4",
scale_mode="mxfp4")`` in ``aiter/ops/flydsl/kernels/preshuffle_gemm_splitk_op.py``.

Runners stay on the tuner side: this module must remain importable without
flydsl (the tuner reads it to name candidates on hosts that cannot compile).
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

# Generic (dtype-agnostic) helpers ported once in the a8w8 module; reused here
# rather than duplicated, since they do not depend on any a8w8-specific field.
# ``_ki``/``kernel_fits_shape`` are reused directly (not ported) by
# ``kernel_fits_shape_splitk_mxfp4`` below, so the audited fit clauses live in
# exactly one place.
from aiter.ops.flydsl.gemm_tune.flydsl_gemm_a8w8_bpreshuffle_common import (
    _estimate_max_wpe,
    _ki,
    _smem_align,
    _smem_finalize_size,
    get_gfx,
    kernel_fits_shape,
)

_DTYPE_SHORT = {
    "bf16": "B16",
    "fp16": "F16",
}

# Narrow decode tiles -- same shapes as the a8w8 split-K set;
# reused because the tile geometry that suits small-M decode is dtype-agnostic.
_SPLITK_TILES_MXFP4 = [
    (16, 16, 256),
    (16, 16, 512),
    (16, 32, 256),
    (16, 32, 512),
    (32, 16, 256),
    (32, 16, 512),
    (32, 32, 256),
    (32, 32, 512),
]

# Split factors. sk=1 belongs to this family: it is the direct-out build, which
# writes the final output from the GEMM and skips the reduce pass entirely, so it
# is the right candidate wherever the grid is already full (large N).
# Override via AITER_FLYDSL_A4W4_SPLITK_VALS, e.g. "1,2,4,8,16,32" -- a separate
# knob from the a8w8 AITER_FLYDSL_SPLITK_VALS so an a4w4 tuning run can pick a
# different sweep without perturbing the a8w8 one.
_SPLIT_K_VALS_DEFAULT_MXFP4 = (1, 2, 4, 8, 16)


def _resolve_split_k_vals_mxfp4():
    import os

    raw = os.getenv("AITER_FLYDSL_A4W4_SPLITK_VALS")
    if not raw:
        return _SPLIT_K_VALS_DEFAULT_MXFP4
    vals = tuple(int(x) for x in raw.split(",") if x.strip())
    vals = tuple(v for v in vals if v >= 1)
    return vals or _SPLIT_K_VALS_DEFAULT_MXFP4


_SPLIT_K_VALS_MXFP4 = _resolve_split_k_vals_mxfp4()

_ASYNC_COPY_VALS = (0, 1)
_WAVES_PER_EU = (0, 1, 2, 3, 4)
_XCD_SWIZZLE_VALS = (0, 4)
_LDS_STAGES = (2, 1)

KERNEL_ID_BASE_SPLITK_MXFP4 = 4_000_000
NAME_PREFIX_SPLITK_MXFP4 = "flydsl_a4w4_splitk"

# Fixed by the mxfp4 quant contract (see preshuffle_gemm_splitk_op.py:
# scale_block_k=32 if scale_mode == "mxfp4" else 128); not swept.
SCALE_BLOCK_K_MXFP4 = 32


@dataclass
class A4W4SplitKKernelInstance:
    """One a4w4 (mxfp4) split-K candidate.

    Field order (also the ``.name`` token order): tile_m, tile_n, tile_k,
    split_k, dtype (output: "bf16"|"fp16"), use_async_copy, waves_per_eu,
    xcd_swizzle, lds_stage, sScheduler, use_m_bounded_store. ``q_dtype_a`` /
    ``q_dtype_w`` are always "fp4" for this family (both operands packed
    4-bit), kept as fields for structural parity with the a8w8
    ``SplitKKernelInstance`` rather than swept.
    """

    tile_m: int
    tile_n: int
    tile_k: int
    split_k: int
    q_dtype_a: str = "fp4"
    q_dtype_w: str = "fp4"
    dtype: str = "bf16"
    use_async_copy: int = 0
    waves_per_eu: int = 0
    xcd_swizzle: int = 0
    lds_stage: int = 2
    sScheduler: str = "Default"
    use_m_bounded_store: bool = False

    @property
    def enable_scheduler(self) -> bool:
        return str(self.sScheduler).lower() != "off"

    @property
    def name(self) -> str:
        dt = _DTYPE_SHORT.get(self.dtype, self.dtype.upper())
        return "_".join(
            [
                "flydsl",
                "a4w4",
                "splitk",
                "x".join(map(str, [self.tile_m, self.tile_n, self.tile_k])),
                f"sk{self.split_k}",
                dt,
                "x".join(
                    map(
                        str,
                        [
                            self.use_async_copy,
                            self.waves_per_eu,
                            self.xcd_swizzle,
                            self.lds_stage,
                        ],
                    )
                ),
                self.sScheduler.lower(),
                f"sb{SCALE_BLOCK_K_MXFP4}",
                f"mb{int(self.use_m_bounded_store)}",
            ]
        )


def kernel_instance_estimated_lds_bytes_mxfp4(ki: A4W4SplitKKernelInstance) -> int:
    """A-tile LDS estimate for packed-fp4 operands.

    fp4 packs two 4-bit codes per byte, so the A-tile footprint is
    ``tile_m * tile_k / 2`` bytes (``tile_k`` counted in unpacked elements, the
    same convention ``flydsl_preshuffle_gemm_splitk_a8`` uses when it doubles K
    for ``in_dtype == "fp4"``) -- half the fp8/int8 case the a8w8 estimator
    computes.
    """
    a_tile_bytes = (int(ki.tile_m) * int(ki.tile_k)) // 2
    return _smem_finalize_size(_smem_align(a_tile_bytes)) * (
        2 if int(ki.lds_stage) == 2 else 1
    )


def kernel_fits_shape_splitk_mxfp4(
    ki: A4W4SplitKKernelInstance, M: int, N: int, K: int
) -> bool:
    """Whether an a4w4 split-K candidate is worth tuning for this shape.

    Delegates the LDS/divisibility/num_ctas/M-gate clauses to the shared
    ``kernel_fits_shape`` (the same one the a8w8 split-K family uses) via an
    equivalent ``kernelInstance``, passing this family's fp4-specific LDS
    estimate as an explicit override -- ``kernel_fits_shape`` has no dtype
    entry for packed 4-bit operands, so it cannot derive it from
    ``equiv.q_dtype_a`` itself. Proven equivalent to the prior inline
    reimplementation across the full candidate x shape grid (see the S3
    refactor's equivalence check). **Do not tune these clauses here**: they
    decide the search space, and the split_k range / bounded-store checks
    below are this family's own extra constraints, same as
    ``kernel_fits_shape_splitk`` adds for a8w8.
    """
    equiv = _ki(
        ki.tile_m,
        ki.tile_n,
        ki.tile_k,
        ki.use_async_copy,
        ki.waves_per_eu,
        ki.xcd_swizzle,
        lds_stage=ki.lds_stage,
        q_dtype_a=ki.q_dtype_a,
        q_dtype_w=ki.q_dtype_w,
        dtype=ki.dtype,
        scheduler=ki.sScheduler,
    )
    # split_k need not divide the K-tile count: the kernel pads the tail split
    # and zeroes the overflow tiles. It only has to have at least one tile per
    # split.
    if not (
        kernel_fits_shape(
            equiv, M, N, K, lds_bytes=kernel_instance_estimated_lds_bytes_mxfp4(ki)
        )
        and 1 <= ki.split_k <= K // ki.tile_k
    ):
        return False
    # m_pad == M => the bounded-store predicate is a provable no-op; prune.
    return not (ki.use_m_bounded_store and M % ki.tile_m == 0)


def is_splitk_enabled_mxfp4() -> bool:
    """a4w4 split-K candidates are gfx950-only, same as the a8w8 split-K family."""
    return get_gfx().startswith("gfx950")


def _build_kernels_list_splitk_mxfp4(
    tiles=_SPLITK_TILES_MXFP4,
    split_k_vals=_SPLIT_K_VALS_MXFP4,
) -> dict[int, A4W4SplitKKernelInstance]:
    kl: dict[int, A4W4SplitKKernelInstance] = {}
    idx = KERNEL_ID_BASE_SPLITK_MXFP4
    for lds in _LDS_STAGES:
        for wpe in _WAVES_PER_EU:
            for acp in _ASYNC_COPY_VALS:
                for xcd in _XCD_SWIZZLE_VALS:
                    for sk in split_k_vals:
                        for tm, tn, tk in tiles:
                            if wpe > 0 and wpe > _estimate_max_wpe(tm, tn):
                                continue
                            for umbs in (False, True):
                                kl[idx] = A4W4SplitKKernelInstance(
                                    tm,
                                    tn,
                                    tk,
                                    sk,
                                    use_async_copy=acp,
                                    waves_per_eu=wpe,
                                    xcd_swizzle=xcd,
                                    lds_stage=lds,
                                    use_m_bounded_store=umbs,
                                )
                                idx += 1
    return kl


kernels_list_splitk_mxfp4: dict[int, A4W4SplitKKernelInstance] = (
    _build_kernels_list_splitk_mxfp4() if is_splitk_enabled_mxfp4() else {}
)


# ===========================================================================
# The protocol the tuner iterates
# ===========================================================================


@dataclass(frozen=True)
class Pipeline:
    """One tunable backend of the a4w4 bpreshuffle operator.

    ``q_dtypes_w`` is spelled with plain strings rather than ``torch.dtype`` so
    this module stays importable without torch/flydsl, matching the a8w8
    ``Pipeline`` this mirrors.
    """

    name: str
    kernels_list: dict[int, Any]
    fits: Callable[[Any, int, int, int], bool]
    q_dtypes_w: tuple[str, ...]


# Only one pipeline exists for a4w4 today; the tuple form matches a8w8's
# PIPELINES shape so a shared tuner driver can iterate either module the same
# way.
PIPELINES: tuple[Pipeline, ...] = (
    Pipeline(
        "splitk_mxfp4",
        kernels_list_splitk_mxfp4,
        kernel_fits_shape_splitk_mxfp4,
        ("fp4",),
    ),
)
