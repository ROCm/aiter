# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Opus batched-BMM Python bindings.

This module is intentionally separate from `gemm_op_a16w16.py`: BMM callers use
batch-in-the-middle or grouped layouts (for example DSV4 `wo_a`) while the
underlying kernels still live in the shared opus GEMM backend.
"""

import functools

import torch

from ...jit.core import compile_ops


def _gen_bmm_a8w8_scale_fake_tensors(
    x: torch.Tensor,
    wo_a: torch.Tensor,
    Y: torch.Tensor,
    x_scale: torch.Tensor,
    w_scale: torch.Tensor,
    splitK: int = 2,
    kernelId: int = 0,
) -> None:
    # In-place mutation of ``Y``; fake must mirror the void C++ op (full arg
    # list + None return) so torch.compile registers a mutating op, not a
    # tensor-producing one.
    return None


# mmajor fp8 e8m0 mxscale BMM raw binding: x/Y are [M, batch, *], wo_a + w_scale
# batch-major (zero-copy DSV4 wo_a). kid-dispatched; driven by
# bmm_a8w8_mxscale_opus below.
@compile_ops(
    "module_deepgemm_opus",
    fc_name="opus_bmm_a8w8_mxscale",
    gen_fake=_gen_bmm_a8w8_scale_fake_tensors,
    develop=True,
)
def _opus_bmm_a8w8_mxscale_raw(
    x: torch.Tensor,
    wo_a: torch.Tensor,
    Y: torch.Tensor,
    x_scale: torch.Tensor,
    w_scale: torch.Tensor,
    splitK: int = 2,
    kernelId: int = 0,
) -> None:
    # In-place: result written into ``Y``, void return (``-> None`` keeps it
    # torch.compile-safe as a mutating op). Callers read ``Y``.
    ...


# ---- Shape-driven mxscale flatmm BMM (tuned row + heuristic fallback) ------
# The raw binding has no tuning of its own (kernelId=0 -> slow k32 fused). This
# wrapper adds selection: explicit kernelId -> verbatim; else the tuned row the
# family entry looked up; else M-split for large unaligned M; else a coarse M/G
# heuristic.


@functools.cache
def _mxscale_kid_m_align() -> dict[int, int]:
    """kid -> M multiple its launcher requires (1 == it masks a partial M tile).

    Comes from the codegen instance table, which is also what the tuner filters
    candidates on. This used to be a hand-kept kid allowlist here and a second
    hand-kept m_align column in the tuner, and the two disagreed: kid326 was
    dispatched at unaligned M by this file while the tuner never tuned it there,
    which cost ~9% at the wo_a decode shapes.
    """
    from csrc.opus_gemm.opus_gemm_common import a8w8_mxscale_bmm_kernel_lists

    return {
        int(kid): int(inst.m_align)
        for fam in a8w8_mxscale_bmm_kernel_lists
        for kid, inst in fam.items()
    }


def _kid_runs_m(kid: int, m: int) -> bool:
    """True iff kid's launcher accepts this M (unknown kid -> assume it does not).

    Only a tuned row found at a padded M can name a kernel that rejects the
    real, smaller M, so this is what an incoming id is checked against below.
    No tuned winner needs alignment today (all 11 mask their partial M tile),
    but 10 of the 45 codegen instances require M % 128 or % 256, so a re-tune
    can put one in the CSV.
    """
    align = _mxscale_kid_m_align().get(int(kid))
    return align is not None and m % align == 0


def _heuristic_mxscale_kid(g: int, m: int, n: int, k: int) -> int:
    """Coarse M/G kid picker for shapes not in the tuned CSV.

    kid 158 (512x256 preload pipeline) for large-M/high-G, falling back to kid 150
    (256x256 plain) for K>8192 where 158 early-returns; kid 320/640 for small-M;
    kid 653 the general strong mid/small-M pick; kid 0 (k32 fused) for shapes that
    are not tile-aligned in N or K.
    """

    def div(a: int, b: int) -> bool:
        return a % b == 0

    if div(n, 256) and div(k, 128) and (m >= 2048 or (m >= 1024 and g >= 8)):
        # Large M: the preload pipeline (kid158) is the tuned winner across this
        # whole region (CSV picks 158 for every aligned m>=2048). No M alignment
        # needed -- the pipeline family masks its partial trailing tile via buffer
        # OOB. kid158 stages the SFA/SFB scales into LDS and early-returns for
        # K>8192 (SFA_K_MAX), so gate the preload pick at K<=8192 and fall back to
        # the plain 256x256 (kid150) for K>8192. Measured on g=2,n=1024,k=4096:
        # kid150 was 34-51% slower than 158 at the untuned m=2560/3072/3584
        # buckets, and on unaligned M a single kid158 launch beats the sub-tile
        # kid653 by 13-34% (g2/m2624, g8/m1000, g16/m600).
        return 158 if 4096 <= k <= 8192 else 150
    # Sub-tile M: B_M=32/64 tiles mask partial M via buffer OOB, so run any M
    # (no m-alignment needed -- verified 653/321/... run arbitrary unaligned M).
    if m < 64:
        return 640 if (div(n, 64) and div(k, 256)) else 653
    if m <= 256 and k <= 1024 and div(n, 32) and div(k, 256):
        return 320
    if div(n, 64) and div(k, 128):
        return 653
    return 0  # nothing tile-aligned: k32 fused runs arbitrary shapes


def bmm_a8w8_mxscale_opus(
    x: torch.Tensor,
    wo_a: torch.Tensor,
    x_scale: torch.Tensor,
    w_scale: torch.Tensor,
    out: torch.Tensor | None = None,
    dtype: torch.dtype = torch.bfloat16,
    kernelId: int | None = None,
    splitK: int | None = None,
) -> torch.Tensor:
    """Opus fp8 e8m0 mxscale (block-scale) BMM by kernel id.

    mmajor DSV4 wo_a layout: ``x`` [M, G, K] fp8, ``wo_a`` [G, N, K] fp8,
    ``x_scale`` [M, G, K/128], ``w_scale`` [G, N/128, K/128], ``out`` optional
    [M, G, N]. Returns the [M, G, N] output.

    ``kernelId`` None falls back to the shape heuristic: the tuned CSV is read
    one layer up, in batched_gemm_a8w8_mxscale, which hands the tuned id down.
    An id this backend cannot run at this M gets the heuristic too, so the
    caller never has to know the alignment rules; _opus_bmm_a8w8_mxscale_raw is
    the entry that launches an id verbatim. ``splitK`` defaults to 1.
    """
    m, g, k = int(x.shape[0]), int(x.shape[1]), int(x.shape[2])
    n = int(wo_a.shape[1])

    if out is not None:
        Y = out
    else:
        Y = torch.empty((m, g, n), dtype=dtype, device=x.device)

    # A tuned row found at a padded M can name a kernel whose launcher rejects
    # the real, smaller M; drop its splitK along with it and let the heuristic
    # pick instead of letting the launcher throw.
    if kernelId is not None and not _kid_runs_m(int(kernelId), m):
        kernelId = splitK = None
    if kernelId is None:
        kernelId = _heuristic_mxscale_kid(g, m, n, k)
    if splitK is None:
        splitK = 1

    _opus_bmm_a8w8_mxscale_raw(x, wo_a, Y, x_scale, w_scale, int(splitK), int(kernelId))
    return Y


__all__ = [
    "_opus_bmm_a8w8_mxscale_raw",
    "bmm_a8w8_mxscale_opus",
]
