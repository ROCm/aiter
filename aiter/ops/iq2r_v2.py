# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""IQ2R-v2: 2-bit weights as 8-weight codewords with two-level, MFMA-native scales.

Each group of 8 weights is a 9-bit index into a learned 512-entry E4M3
codebook (or a 10-bit index into a 1,024-entry one) plus 8 explicit sign bits,
in the record layout the gfx950 kernels read. The scales are one E8M0 exponent
``d`` per row and 256 input columns, plus a 4-bit *level* per 32 weights. Level ``l`` means
``m[l % 4] * 2**(l // 4)``, ``m[c] = 2**(c / 4)``: a power of two that joins
``d`` in the scaled-MFMA E8M0 operand, and a multiplier that selects one of 4
pre-scaled copies of the codebook, ``E4M3(codebook * m[c])``.

A weight therefore decodes as::

    w = 2**(d - 127) * 2**(l // 4) * E4M3(codebook[index] * m[l % 4]) * sign

and a per-matrix FP32 gain (two for the interleaved gate/up matrix) multiplies
the projection output. Every term is either part of the MFMA operands (E8M0
scales, FP8 codewords) or applied once per output: no extra MFMA and no
per-weight arithmetic. The codebook copies are part of the format, so the
encoder, the reference decoder and the kernels agree bit for bit.

The encoder follows the IQ2_XXS reference quantizer of ggml/llama.cpp (MIT,
Copyright (c) The ggml authors) as adapted for GLM-5 by antirez
(https://github.com/antirez/llama.cpp-deepseek-v4-flash, MIT):

1. Cost: each weight's squared error is weighted by
   ``importance[k] * sqrt(sigma^2 + w^2)``, sigma^2 being the mean square of
   its 256-weight block, so large weights in quiet channels still count.
2. Group scale: per 32 weights, an initial scale from ``make_qp_quants``, then
   13 candidate scales around it; each assigns every 8-vector its cheapest
   codeword and the candidate with the best least-squares fit wins, refit by
   weighted least squares.
3. Levels: per 256 weights, ``d`` and every group's level are searched
   exhaustively (three exponents for ``d``, all 16 levels per group, with full
   codeword re-assignment under each), minimizing the same weighted cost.
4. Gain: the projection output is scaled so the reconstruction's gain
   ``<W, Wq> / <W, W>`` is ``target_gain`` (0.98 by default).

``iq2r_v2_learn_codebook`` learns the per-matrix codebook with
importance-weighted k-means alternating with this encoder, seeded from the
IQ2_XXS grid (one phase at 1x, one at sqrt(2)), and returns it with the
matrix's encoding. A 1,024-entry codebook is seeded with the grid at
``2**(j / 4)``, j = 0..3.

Codebook entries are normal E4M3 values from 2**-6 to the seed's maximum
(bytes 0x08..0x78 at most; ``IQ2R_V2_CODEBOOK_MIN_BYTE``,
``IQ2R_V2_CODEBOOK_MAX_BYTE``). Most gfx950 kernels load only copy 0 and
form copies 1..C-1 with integer byte arithmetic, which equals
``E4M3(codebook * m[c])`` only in that range. The learner keeps its codebooks
there, and :func:`iq2r_v2_check_codebook` rejects others.
"""

from __future__ import annotations

import base64
import math
from dataclasses import dataclass

import torch
from torch import Tensor

from .iq2r_format import (
    IQ2R_ATOM_TWO_METADATA_OFFSET,
    IQ2R_ATOM_TWO_RECORD_BYTES,
    IQ2R_ATOM_TWO_RECORDS_OFFSET,
    IQ2R_ATOMS_PER_TRIPLET,
    IQ2R_CODEBOOK_ENTRIES,
    IQ2R_GROUP_BYTES,
    IQ2R_LANE_RECORD_BYTES,
    IQ2R_N_BLOCKS_PER_GROUP,
    IQ2R_SCALE_BLOCK,
    IQ2R_TILE_N,
    IQ2R_TRIPLET_BYTES,
    IQ2R_VECTOR_SIZE,
    IQ2R_WIDE_CODEBOOK_ENTRIES,
    IQ2R_WIDE_EXT_OFFSET,
    IQ2R_WIDE_TENTH_BYTE,
    iq2r_data_bytes,
    iq2r_index_bits,
    iq2r_k_tiles,
    iq2r_slab_bytes,
    iq2r_slab_k_tiles,
)

# The IQ2_XXS positive-magnitude grid (``iq2xxs_grid`` in ggml-common.h of
# ggml/llama.cpp, https://github.com/ggml-org/llama.cpp, MIT License,
# Copyright (c) The ggml authors) serialized as 256 eight-byte vectors.  Two
# E4M3 phases seed the 512-entry learned IQ2R codebook.
_IQ2_XXS_GRID_BYTES = base64.b64decode(
    "CAgICAgICAgrCAgICAgICBkZCAgICAgICCsICAgICAgrKwgICAgICBkIGQgICAgICBkZCAgICAgICCsICAgICCsIKwgICAgICCsrCAgICAgrKysICAgICBkICBkICAgICBkIGQgICAgICBkZCAgICAgrGRkICAgIGQgrGQgICAgIGSsZCAgICAgICCsICAgIKwgIKwgICAgrKwgrCAgICCsIKysICAgIGQgICBkICAgIGQgIGQgICAgIGQgZCAgIGRkZCBkICAgICAgZGQgICAgZCCsZCAgICCsZKxkICAgICAgIKwgICCsICAgrCAgIKwgrCCsICAgrCAgrKwgICBkICAgIGQgICBkICAgZCAgICBkICBkICBkIKwgIGQgICBkrCAgZCAgICAgZCBkICCsICBkIGQgICCsIGQgZCAgICCsZCBkICBkICCsIGQgICBkIKwgZCAgICBkrCBkICAgZKysIGQgICAgICBkZCAgrCAgIGRkICAgrCAgZGQgICAgrCBkZCAgrGQgZGRkICBkrKxkZGQgICAgIKxkZCAgZCBkrGRkICBkrCAgrGQgICAgZCCsZCAgICAgZKxkICAgZCCsrGQgICBkrKysZCAgICAgICCsICBkZCAgIKwgICCsICAgrCAgIGRkICCsICAgrKwgIKwgIGQgIGQgrCAgIGQgZCCsICAgIGRkIKwgIKwgZGQgrCAgIKwgrCCsICAgZCAgZKwgICAgIGRkrCAgrCAgIKysICAgZGQgrKwgIGQgICAgIGQgIGQgICAgZCAgIGQgICBkIGQgrCAgIGQgICAgZCAgZCAgIKxkICBkICBkIKwgIGQgICBkrCAgZCBkZGSsICBkICAgICBkIGQgIKwgIGQgZCAgIKwgZCBkICAgZGRkIGQgrKxkZGQgZCAgICCsZCBkICBkrCCsIGQgZGQgZKwgZCAgICAgIGRkICCsICAgZGQgICCsICBkZCBkZKwgIGRkIGSsIGQgZGQgICAgrCBkZCAgrGQgZGRkIKwgrGRkZGQgICAgIKxkZCCsZGQgrGRkIGQgICAgrGQgIGQgICCsZCAgIGQgIKxkICAgIGQgrGQgZCAgrCCsZCAgICAgZKxkIGRkICBkrGQgICCsrGSsZCBkIGRkrKxkICAgICAgIKwgrCAgICAgrCCsrCAgICCsICBkIGQgIKwgZCCsZCAgrCAgICCsICCsIKwgIKwgIKwgZKysIGQgrCAgrCBkZCCsICAgICCsIKwgrCAgIKwgrCBkICAgIGSsICBkICAgZKwgICBkICBkrCAgICBkIGSsIKxkZGQgZKwgICAgIGRkrCBkICBkZGSsICBkrGRkZKwgICBkrKxkrCAgrCAgIKysICAgrCAgrKwgIGRkrCCsrCAgZCBkrKysIGQgICAgICBkIGQgICAgIGQgIGQgICAgZCCsZCAgICBkZCCsICAgIGQgZKwgICAgZCAgIGQgICBkIKwgZCAgIGSsZGRkICAgZCAgrGQgICBkZCAgrCAgIGQgZCCsICAgZCAgZKwgICBkICAgIGQgIGQgIKwgZCAgZGQgrGRkICBkICAgrGQgIGRkZCCsZCAgZGQgICCsICBkICBkIKwgIGQgrCBkrCAgZKxkZGSsICBkIKysZKwgIGQgICAgIGQgZCCsICAgZCBkICCsICBkIGQgICCsIGQgZGSsZKwgZCBkrCBkIGRkIGQgZKwgZGQgZCAgICCsZCBkZCAgICCsIGQgZCAgIKwgZCAgZCAgrCBkICAgZCCsIGRkZCBkIKwgZCAgICBkrCBkIKxkZGSsIGRkIKxkZKwgZKwgIKxkrCBkZGQgZKysIGQgIGSsrKwgZCAgICAgIGRkIKwgICAgZGRkIGQgICBkZGSsZCAgIGRkICCsICAgZGQgICCsICBkZCCsIKwgIGRkIGQgIGQgZGSsICBkZCBkZCBkrKxkIGRkZCBkrKwgZGQgIGSsIGRkZKwgZKwgZGRkrKwgIGRkZGRkICAgrGRkZCBkZGSsZGRkICAgICCsZGRkIGQgIKxkZGSsZCAgrGRkIGSsZCCsZGQgICBkZKxkZCCsICCsrGRkIGQgICAgrGQgIGQgICCsZCAgIGQgIKxkIKysZCAgrGQgICAgZCCsZGRkZGRkIKxkIKxkIKwgrGQgIKxkrCCsZCAgICAgZKxkZGQgICBkrGQgIGQgZGSsZKwgZCBkZKxkIGQgrGRkrGSsICBkIKysZCAgICAgICCsrCAgICAgIKysrCAgICAgrGQgIGQgICCsrCAgrCAgIKwgZCAgZCAgrCCsZCBkICCsICAgZGQgIKxkIGQgrCAgrGQgICAgZCCsIGQgICBkIKwgIGQgIGQgrGRkZCAgZCCsICAgZCBkIKwgIKxkIGQgrCAgICBkZCCsrGQgZGRkIKwgZGSsZGQgrGSsICCsZCCsICAgZKxkIKwgIKxkrGQgrKwgICAgrCCsIGQgIGSsIKxkIGQgrKwgrCBkICAgIGSsICBkICAgZKwgZKwgICBkrCAgIGQgIGSsZCCsrCAgZKysZGQgZCBkrCAgIKxkIGSsZGQgZKwgZKwgICAgIGRkrKwgrCAgZGSsIGQgZCBkZKxkIGRkZGRkrGQgIKwgrGSsICCsIGSsZKysICAgICCsrCAgZGQgIKysZGQgrCAgrKxkrCAgZCCsrCAgICCsIKysIKxkICBkrKwgIGRkIKysrCBkICBkrKys="
)

IQ2R_V2_D_BLOCK = 256
IQ2R_V2_GAIN_BYTES = 8  # two FP32 gains: even rows, odd rows
IQ2R_V2_EPS = 1e-15
# Codebook byte range the kernels' in-LDS copy formation reproduces exactly:
# normal E4M3 values 2**-6 (0x08) to 256 (0x78).
IQ2R_V2_CODEBOOK_MIN_BYTE = 0x08
IQ2R_V2_CODEBOOK_MAX_BYTE = 0x78
IQ2R_V2_CODEBOOK_MIN = 2.0**-6


@dataclass(frozen=True, slots=True)
class IQ2RV2Levels:
    """The 16 per-group levels ``multipliers[l % C] * 2**(l // C)``."""

    name: str
    multipliers: tuple[float, ...]

    @property
    def classes(self) -> int:
        return len(self.multipliers)

    @property
    def octaves(self) -> int:
        return 16 // self.classes

    def level(self, index: int) -> float:
        return self.multipliers[index % self.classes] * 2.0 ** (index // self.classes)


# Four quarter-octave mantissas over four octaves: the level set the kernels run.
IQ2R_V2_LEVELS_C4 = IQ2RV2Levels("c4e4q", tuple(2.0 ** (j / 4.0) for j in range(4)))


def iq2r_v2_format_name(
    levels: IQ2RV2Levels, codebook_entries: int = IQ2R_CODEBOOK_ENTRIES
) -> str:
    iq2r_index_bits(codebook_entries)
    return f"iq2r-{codebook_entries}-fullsign-{levels.name}-e8m0d256-native-v2"


# --------------------------------------------------------------------------
# Sizes
# --------------------------------------------------------------------------


def iq2r_v2_d_columns(k: int) -> int:
    if k % IQ2R_V2_D_BLOCK:
        raise ValueError(f"IQ2R-v2 requires K divisible by {IQ2R_V2_D_BLOCK}, got {k}")
    return k // IQ2R_V2_D_BLOCK


# --------------------------------------------------------------------------
# Codebooks
# --------------------------------------------------------------------------


def _e4m3(x: Tensor) -> Tensor:
    return x.to(torch.float8_e4m3fn).float()


def iq2r_v2_seed_codebook(
    device: torch.device | str = "cpu", entries: int = IQ2R_CODEBOOK_ENTRIES
) -> Tensor:
    """IQ2_XXS grid in native units (grid / 8): 512 entries at 1x (0-255) and
    sqrt(2)x (256-511); 1,024 entries ``E4M3(E4M3(grid / 8) * 2**(j / 4))``,
    j = 0..3 (256 each)."""

    iq2r_index_bits(entries)
    grid = torch.tensor(list(_IQ2_XXS_GRID_BYTES), dtype=torch.float32, device=device)
    grid = grid.reshape(256, IQ2R_VECTOR_SIZE) * 0.125
    if entries == IQ2R_WIDE_CODEBOOK_ENTRIES:
        # The phases scale the E4M3-rounded grid (25 / 8 rounds to 3).
        grid = _e4m3(grid)
        phases = [grid * 2.0 ** (j / 4.0) for j in range(4)]
    else:
        phases = [grid, grid * math.sqrt(2.0)]
    return _e4m3(torch.cat(phases)).contiguous()


def iq2r_v2_check_codebook(codebook: Tensor) -> None:
    """Raise unless every entry of an E4M3 ``codebook`` (any shape) is a byte in
    ``IQ2R_V2_CODEBOOK_MIN_BYTE..IQ2R_V2_CODEBOOK_MAX_BYTE`` (see the module
    docstring)."""

    raw = codebook.float().to(torch.float8_e4m3fn).view(torch.uint8)
    low, high = int(raw.min()), int(raw.max())
    if low < IQ2R_V2_CODEBOOK_MIN_BYTE or high > IQ2R_V2_CODEBOOK_MAX_BYTE:
        raise ValueError(
            f"IQ2R-v2 codebook bytes span 0x{low:02x}..0x{high:02x}; the kernels "
            f"need 0x{IQ2R_V2_CODEBOOK_MIN_BYTE:02x}..0x{IQ2R_V2_CODEBOOK_MAX_BYTE:02x}"
        )


def iq2r_v2_codebook_copies(codebook: Tensor, levels: IQ2RV2Levels) -> Tensor:
    """``[C, ..., entries, 8]`` FP32 copies ``E4M3(codebook * m[c])``, the values the kernels read.

    Each product is formed in FP32 and rounded to E4M3 (round to nearest even),
    the same two steps as the kernels' conversion.
    """

    book = codebook.float()
    multipliers = torch.tensor(
        levels.multipliers, dtype=torch.float32, device=book.device
    )
    return _e4m3(book[None] * multipliers.view(-1, *([1] * book.dim())))


# --------------------------------------------------------------------------
# Encoder
# --------------------------------------------------------------------------


def _cost_weights(blocks: Tensor, importance: Tensor) -> Tensor:
    sigma2 = blocks.square().mean(-1, keepdim=True)
    return importance * torch.sqrt(sigma2 + blocks.square())


def _make_qp_quants(x: Tensor, w: Tensor, nmax: int = 4) -> Tensor:
    """ggml's ``make_qp_quants`` per row of ``x`` (non-negative): least-squares scale."""

    mx = x.amax(-1)
    ok = mx >= IQ2R_V2_EPS
    mxs = torch.where(ok, mx, torch.ones_like(mx))
    iscale = nmax / mxs
    best = (
        w * (x - (1 / iscale)[:, None] * torch.round(iscale[:, None] * x)).square()
    ).sum(-1)
    for step in range(-4, 5):
        if step == 0:
            continue
        candidate = (0.1 * step + nmax) / mxs
        q = torch.round(candidate[:, None] * x).clamp(max=nmax)
        mse = (w * (x - (1 / candidate)[:, None] * q).square()).sum(-1)
        better = mse < best
        best = torch.where(better, mse, best)
        iscale = torch.where(better, candidate, iscale)
    q = torch.round(iscale[:, None] * x).clamp(max=nmax)
    sumlx = (w * x * q).sum(-1)
    suml2 = (w * q * q).sum(-1)
    for _ in range(5):
        for i in range(x.shape[1]):
            wi, xi, qi = w[:, i], x[:, i], q[:, i]
            slx = sumlx - wi * xi * qi
            sl2 = suml2 - wi * qi * qi
            valid = (slx > 0) & (sl2 > 0)
            nq = torch.round(
                xi * sl2 / torch.where(valid, slx, torch.ones_like(slx))
            ).clamp(max=nmax)
            slx2 = slx + wi * xi * nq
            sl22 = sl2 + wi * nq * nq
            accept = valid & (nq != qi) & (slx2 * slx2 * suml2 > sumlx * sumlx * sl22)
            q[:, i] = torch.where(accept, nq, qi)
            sumlx = torch.where(accept, slx2, sumlx)
            suml2 = torch.where(accept, sl22, suml2)
    positive = suml2 > 0
    scale = torch.where(
        positive, sumlx / torch.where(positive, suml2, torch.ones_like(suml2)), 0.0
    )
    return torch.where(ok, scale, torch.zeros_like(scale))


def _codeword_scores(
    vectors: Tensor, weights: Tensor, codebook: Tensor
) -> tuple[Tensor, Tensor]:
    """``A[v, c] = sum_j w c^2`` and ``B[v, c] = sum_j w x c``: cost(s) = s^2 A - 2 s B + const."""

    return weights @ codebook.square().t(), (weights * vectors) @ codebook.t()


def _group_scales(
    magnitudes: Tensor, weights: Tensor, codebook: Tensor, candidates: int = 6
) -> Tensor:
    """Per 32-weight group: the least-squares scale of the best of 13 candidates (steps 1-2)."""

    groups = magnitudes.reshape(-1, IQ2R_SCALE_BLOCK)
    group_weights = weights.reshape(-1, IQ2R_SCALE_BLOCK)
    zero = groups.amax(-1) < IQ2R_V2_EPS
    effective_max = 3.0 * _make_qp_quants(groups, group_weights)
    zero |= effective_max <= 0
    a, b = _codeword_scores(
        magnitudes.reshape(-1, IQ2R_VECTOR_SIZE),
        weights.reshape(-1, IQ2R_VECTOR_SIZE),
        codebook,
    )
    b2 = 2 * b

    def fit(scale: Tensor) -> tuple[Tensor, Tensor]:
        index = torch.argmin(scale.repeat_interleave(4)[:, None] * a - b2, dim=-1)
        q2 = a.gather(1, index[:, None]).reshape(-1, 4).sum(-1)
        qx = b.gather(1, index[:, None]).reshape(-1, 4).sum(-1)
        return q2, qx

    best = torch.zeros_like(effective_max)
    best_scale = torch.zeros_like(effective_max)
    for step in range(-candidates, candidates + 1):
        # antirez's inverse-scale steps around 5 grid units (40 native units / 8).
        scale = torch.where(zero, 0.0, effective_max / (5.0 + 0.1 * step))
        q2, qx = fit(scale)
        safe = torch.where(q2 > 0, q2, torch.ones_like(q2))
        criterion = torch.where(q2 > 0, qx * qx / safe, 0.0)
        better = (q2 > 0) & (criterion > best)
        best = torch.where(better, criterion, best)
        best_scale = torch.where(better, qx / safe, best_scale)
    q2, qx = fit(torch.where(zero, 0.0, best_scale))
    refit = torch.where(
        (q2 > 0) & ~zero, qx / torch.where(q2 > 0, q2, torch.ones_like(q2)), 0.0
    )
    return refit.clamp_min(0.0)


@dataclass
class IQ2RV2Encoding:
    """IQ2R-v2 encoding of whole 256-weight blocks (row-major logical order)."""

    indices: Tensor  # [..., 32 per block] int64
    negative: Tensor  # [..., 256 per block] bool
    levels: Tensor  # [..., 8 per block] uint8
    d: Tensor  # [..., 1 per block] uint8, E8M0
    reconstruction: Tensor  # [..., 256 per block] fp32, before the gain
    cost: float


@torch.no_grad()
def _encode_blocks(
    blocks: Tensor,
    block_importance: Tensor,
    codebook: Tensor,
    levels: IQ2RV2Levels,
    chunk_blocks: int,
) -> IQ2RV2Encoding:
    """Encode ``blocks [B, 256]`` with per-weight importance ``block_importance [B, 256]``."""

    device = blocks.device
    copies = iq2r_v2_codebook_copies(codebook, levels)
    classes, octaves = levels.classes, levels.octaves
    nominal_max = levels.level(15)
    total_blocks = blocks.shape[0]
    indices = torch.empty(total_blocks, 32, dtype=torch.int64, device=device)
    level_out = torch.empty(total_blocks, 8, dtype=torch.uint8, device=device)
    d_out = torch.empty(total_blocks, dtype=torch.uint8, device=device)
    reconstruction = torch.empty_like(blocks, dtype=torch.float32)
    cost = 0.0
    for start in range(0, total_blocks, chunk_blocks):
        stop = min(total_blocks, start + chunk_blocks)
        x = blocks[start:stop].float()
        weights = _cost_weights(x, block_importance[start:stop].float())
        magnitudes = x.abs()
        rows = stop - start
        group_scale = _group_scales(magnitudes, weights, copies[0]).reshape(rows, 8)
        vectors = magnitudes.reshape(-1, IQ2R_VECTOR_SIZE)
        vector_weights = weights.reshape(-1, IQ2R_VECTOR_SIZE)
        scores = [
            (a, 2 * b)
            for a, b in (
                _codeword_scores(vectors, vector_weights, copies[c])
                for c in range(classes)
            )
        ]
        # d: the E8M0 exponent putting the largest group at the top level, and its neighbours.
        top = (group_scale.amax(-1) / nominal_max).clamp_min(2.0**-126)
        centre = torch.floor(torch.log2(top))
        best_cost = torch.full((rows,), math.inf, device=device)
        best_d = torch.zeros(rows, device=device)
        best_level = torch.zeros(rows, 8, dtype=torch.uint8, device=device)
        best_index = torch.zeros(rows * 32, dtype=torch.int64, device=device)
        best_recon = torch.zeros(rows, IQ2R_V2_D_BLOCK, device=device)
        for offset in (-1.0, 0.0, 1.0):
            exponent = (centre + offset).clamp(-127, 127 - (octaves - 1))
            group_cost = torch.full((rows, 8), math.inf, device=device)
            group_level = torch.zeros(rows, 8, dtype=torch.uint8, device=device)
            group_index = torch.zeros(rows * 32, dtype=torch.int64, device=device)
            group_recon = torch.zeros(rows, 8, IQ2R_SCALE_BLOCK, device=device)
            for level in range(16):
                a, b2 = scores[level % classes]
                scale = torch.exp2(exponent + level // classes).repeat_interleave(32)
                index = torch.argmin(scale[:, None] * a - b2, dim=-1)
                recon = (copies[level % classes][index] * scale[:, None]).reshape(
                    rows, 8, IQ2R_SCALE_BLOCK
                )
                err = (
                    weights.reshape(rows, 8, -1)
                    * (recon - magnitudes.reshape(rows, 8, -1)).square()
                ).sum(-1)
                better = err < group_cost
                group_cost = torch.where(better, err, group_cost)
                group_level = torch.where(better, level, group_level)
                group_recon = torch.where(better[..., None], recon, group_recon)
                group_index = torch.where(
                    better.repeat_interleave(4, dim=1).reshape(-1), index, group_index
                )
            total = group_cost.sum(-1)
            better = total < best_cost
            best_cost = torch.where(better, total, best_cost)
            best_d = torch.where(better, exponent, best_d)
            best_level = torch.where(better[:, None], group_level, best_level)
            best_recon = torch.where(
                better[:, None], group_recon.reshape(rows, -1), best_recon
            )
            best_index = torch.where(
                better.repeat_interleave(32), group_index, best_index
            )
        indices[start:stop] = best_index.reshape(rows, 32)
        level_out[start:stop] = best_level
        d_out[start:stop] = (best_d + 127).to(torch.uint8)
        reconstruction[start:stop] = torch.where(x < 0, -best_recon, best_recon)
        cost += float(best_cost.sum(dtype=torch.float64))
    return IQ2RV2Encoding(indices, blocks < 0, level_out, d_out, reconstruction, cost)


@torch.no_grad()
def iq2r_v2_encode(
    weight: Tensor,
    importance: Tensor,
    codebook: Tensor,
    levels: IQ2RV2Levels = IQ2R_V2_LEVELS_C4,
    *,
    chunk_blocks: int = 4096,
) -> IQ2RV2Encoding:
    """Encode ``weight [N, K]`` (K % 256 == 0) with a native ``[512, 8]`` or
    ``[1024, 8]`` E4M3 codebook.

    Returns indices ``[N, K/8]``, negative ``[N, K]``, levels ``[N, K/32]``,
    d ``[N, K/256]`` and the reconstruction ``[N, K]`` (before the gain).
    """

    if weight.ndim != 2 or importance.shape != (weight.shape[1],):
        raise ValueError("weight and importance must have shapes [N,K] and [K]")
    n, k = weight.shape
    columns = iq2r_v2_d_columns(k)
    blocks = weight.float().reshape(-1, IQ2R_V2_D_BLOCK)
    block_importance = (
        importance.float()
        .reshape(1, columns, IQ2R_V2_D_BLOCK)
        .expand(n, -1, -1)
        .reshape(-1, IQ2R_V2_D_BLOCK)
    )
    enc = _encode_blocks(blocks, block_importance, codebook, levels, chunk_blocks)
    return IQ2RV2Encoding(
        indices=enc.indices.reshape(n, k // IQ2R_VECTOR_SIZE),
        negative=enc.negative.reshape(n, k),
        levels=enc.levels.reshape(n, k // IQ2R_SCALE_BLOCK),
        d=enc.d.reshape(n, columns),
        reconstruction=enc.reconstruction.reshape(n, k),
        cost=enc.cost,
    )


def iq2r_v2_gains(
    weight: Tensor,
    reconstruction: Tensor,
    target_gain: float = 0.98,
    *,
    interleaved: bool = False,
) -> tuple[float, float]:
    """Output gains (even rows, odd rows) giving ``<W, g*Wq> / <W, W> = target_gain``.

    ``interleaved`` (the gate0, up0, gate1, ... matrix) fits gate and up rows separately.
    """

    def gain(w: Tensor, q: Tensor) -> float:
        w, q = w.double(), q.double()
        return target_gain * float(w.square().sum() / (q * w).sum().clamp_min(1e-30))

    if not interleaved:
        g = gain(weight, reconstruction)
        return g, g
    return gain(weight[0::2], reconstruction[0::2]), gain(
        weight[1::2], reconstruction[1::2]
    )


# --------------------------------------------------------------------------
# Codebook learning
# --------------------------------------------------------------------------


@torch.no_grad()
def iq2r_v2_learn_codebook(
    weight: Tensor,
    importance: Tensor,
    levels: IQ2RV2Levels = IQ2R_V2_LEVELS_C4,
    *,
    rounds: int = 4,
    inner: int = 4,
    sample_vectors: int = 65536,
    seed: int = 0,
    chunk_blocks: int = 4096,
    entries: int = IQ2R_CODEBOOK_ENTRIES,
) -> tuple[Tensor, IQ2RV2Encoding]:
    """Learn one ``[entries, 8]`` (512 or 1,024) native E4M3 codebook for ``weight [N, K]``.

    Importance-weighted k-means on ``sample_vectors / 32`` random 256-weight
    blocks, alternating with the encoder. Each round encodes the sample, then
    runs ``inner`` Lloyd steps at its fixed per-group scales t (centroid =
    cost-weighted mean of t * |x| over t^2, so the update minimizes the
    encoder's own cost). Codewords stay within the seed's maximum, which keeps
    the scale search's reference point, and at or above
    ``IQ2R_V2_CODEBOOK_MIN`` (see the module docstring), and a round is kept
    only if the weighted cost of the whole matrix drops.

    Returns the codebook and ``weight``'s encoding with it, which the last kept
    round has already computed.
    """

    k = weight.shape[1]
    device = weight.device
    columns = iq2r_v2_d_columns(k)
    book = iq2r_v2_seed_codebook(device, entries)
    pin = float(book.max())
    blocks = weight.float().reshape(-1, IQ2R_V2_D_BLOCK)
    block_importance = importance.float().reshape(columns, IQ2R_V2_D_BLOCK)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    count = min(blocks.shape[0], max(1, sample_vectors // 32))
    pick = torch.randperm(blocks.shape[0], generator=generator)[:count].to(device)
    sample = blocks[pick]
    sample_importance = block_importance[pick % columns]
    vectors = sample.abs().reshape(-1, IQ2R_VECTOR_SIZE)
    weights = _cost_weights(sample, sample_importance).reshape(-1, IQ2R_VECTOR_SIZE)
    multipliers = torch.tensor(levels.multipliers, device=device)

    def matrix_encode(b: Tensor) -> IQ2RV2Encoding:
        return iq2r_v2_encode(weight, importance, b, levels, chunk_blocks=chunk_blocks)

    best = matrix_encode(book)
    for _ in range(rounds):
        enc = _encode_blocks(sample, sample_importance, book, levels, chunk_blocks)
        level = enc.levels.long()
        # Scale each vector's plain codeword sees: 2^(d-127) * 2^(l//C) * m[l%C].
        t = (
            torch.exp2(enc.d.float()[:, None] - 127 + (level // levels.classes).float())
            * multipliers[level % levels.classes]
        )
        t = t.repeat_interleave(4, dim=1).reshape(-1)
        index = enc.indices.reshape(-1)
        candidate = book
        for step in range(inner):
            if step:
                a, b = _codeword_scores(vectors, weights, candidate)
                index = torch.argmin(t[:, None] * a - 2 * b, dim=-1)
            numerator = torch.zeros_like(candidate).index_add_(
                0, index, weights * t[:, None] * vectors
            )
            denominator = torch.zeros_like(candidate).index_add_(
                0, index, weights * t[:, None].square()
            )
            occupied = denominator > 0
            candidate = torch.where(
                occupied, numerator / torch.where(occupied, denominator, 1.0), candidate
            )
        candidate = _e4m3(candidate.clamp(IQ2R_V2_CODEBOOK_MIN, pin))
        encoding = matrix_encode(candidate)
        if encoding.cost >= best.cost:
            break
        book, best = candidate, encoding
    return book.contiguous(), best


# --------------------------------------------------------------------------
# LDLQ: Hessian error feedback over K
# --------------------------------------------------------------------------
#
# The encoder above minimises an importance-weighted weight error, which uses
# only the diagonal of the input second moment H = E[x x^T]. LDLQ (QuIP; the
# error feedback of GPTQ) re-chooses the codeword indices and signs of an
# encoding to lower the expected output error trace(dW H dW^T): walking K in
# 8-weight vectors, each vector takes the codeword nearest to its target, and
# the target of the vectors after it absorbs the error through the Cholesky
# factor of H^-1. The scales (d, levels) stay as encoded, so the format,
# bits and kernels do not change.

IQ2R_V2_LDLQ_DAMP = 1.0  # damping, times the mean of H's diagonal
IQ2R_V2_LDLQ_BATCH = 128  # columns per lazy batch; the result does not depend on it


@dataclass(frozen=True)
class IQ2RV2LdlqFactor:
    """Upper Cholesky factor ``u`` of ``H^-1`` [K, K] and the inverses of its
    8 x 8 diagonal blocks ``block_inverses`` [K/8, 8, 8]."""

    u: Tensor
    block_inverses: Tensor


@torch.no_grad()
def iq2r_v2_ldlq_factor(
    hessian: Tensor, damp: float = IQ2R_V2_LDLQ_DAMP
) -> IQ2RV2LdlqFactor:
    """Factor of ``hessian + damp * mean(diag) * I`` (FP64 Cholesky, FP32 result)."""

    k = hessian.shape[0]
    if hessian.shape != (k, k) or k % IQ2R_VECTOR_SIZE:
        raise ValueError("the LDLQ Hessian must be [K, K] with K a multiple of 8")
    h = hessian.double()
    h = h + damp * h.diagonal().mean() * torch.eye(k, dtype=h.dtype, device=h.device)
    u = torch.linalg.cholesky(
        torch.cholesky_inverse(torch.linalg.cholesky(h)), upper=True
    )
    blocks = u.unfold(0, 8, 8).unfold(1, 8, 8).diagonal(dim1=0, dim2=1)
    inverses = torch.linalg.inv(blocks.permute(2, 0, 1))
    return IQ2RV2LdlqFactor(u.float().contiguous(), inverses.float().contiguous())


@torch.no_grad()
def iq2r_v2_ldlq(
    weight: Tensor,
    encoding: IQ2RV2Encoding,
    codebook: Tensor,
    factor: IQ2RV2LdlqFactor,
    levels: IQ2RV2Levels = IQ2R_V2_LEVELS_C4,
    *,
    batch: int = IQ2R_V2_LDLQ_BATCH,
) -> IQ2RV2Encoding:
    """Re-choose ``encoding``'s indices and signs for ``weight [N, K]`` by LDLQ.

    Each 8-weight vector keeps its scale ``2^(d - 127 + l // C)`` and codebook
    copy ``l % C``; it takes the copy's codeword nearest (unweighted) to its
    target's magnitude over that scale, with the target's signs. Rows are
    independent, so an interleaved gate/up matrix is two projections sharing
    ``factor``. Returns a new encoding with the same ``d`` and levels; its
    ``cost`` is the encoding's unweighted squared error.
    """

    n, k = weight.shape
    if k % batch or batch % IQ2R_VECTOR_SIZE:
        raise ValueError("the LDLQ batch must divide K and be a multiple of 8")
    if factor.u.shape != (k, k):
        raise ValueError("the LDLQ factor does not match the weight's K")
    device = weight.device
    classes = levels.classes
    copies = iq2r_v2_codebook_copies(codebook, levels).to(device)  # [C, E, 8]
    entries = copies.shape[1]
    flat = copies.reshape(classes * entries, IQ2R_VECTOR_SIZE)
    norms = copies.square().sum(-1)  # [C, E]
    u = factor.u.to(device)
    block_inverses = factor.block_inverses.to(device)
    level = encoding.levels.long()
    exponent = (encoding.d.float() - 127).repeat_interleave(8, dim=1) + (
        level // classes
    ).float()
    scale = torch.exp2(exponent).repeat_interleave(4, dim=1)  # [N, K/8]
    copy = (level % classes).repeat_interleave(4, dim=1)  # [N, K/8]

    target = weight.float().clone()
    reconstruction = torch.empty_like(target)
    indices = torch.empty(n, k // IQ2R_VECTOR_SIZE, dtype=torch.int64, device=device)
    rows = torch.arange(n, device=device)
    for b0 in range(0, k, batch):
        b1 = b0 + batch
        block = target[:, b0:b1].clone()
        errors = torch.empty(n, batch, device=device)
        for c0 in range(b0, b1, IQ2R_VECTOR_SIZE):
            r0, j = c0 - b0, c0 // IQ2R_VECTOR_SIZE
            r1 = r0 + IQ2R_VECTOR_SIZE
            t = block[:, r0:r1]
            s, c = scale[:, j], copy[:, j]
            dot = (t.abs() / s[:, None]) @ flat.T  # [N, C * E]
            dot = dot.view(n, classes, entries)[rows, c]
            index = torch.argmin(norms[c] - 2 * dot, dim=-1)
            q = copies[c, index] * s[:, None]
            q = torch.where(t < 0, -q, q)
            reconstruction[:, c0 : c0 + IQ2R_VECTOR_SIZE] = q
            indices[:, j] = index
            e = (t - q) @ block_inverses[j]
            errors[:, r0:r1] = e
            if r1 < batch:
                block[:, r1:] -= e @ u[c0 : c0 + IQ2R_VECTOR_SIZE, c0 + 8 : b1]
            target[:, c0 : c0 + IQ2R_VECTOR_SIZE] = t
        if b1 < k:
            target[:, b1:] -= errors @ u[b0:b1, b1:]
    negative = target < 0
    cost = float((weight.double() - reconstruction.double()).square().sum())
    return IQ2RV2Encoding(
        indices=indices,
        negative=negative,
        levels=encoding.levels.clone(),
        d=encoding.d.clone(),
        reconstruction=reconstruction,
        cost=cost,
    )


# --------------------------------------------------------------------------
# Canonical records
# --------------------------------------------------------------------------
#
# Per 16-row N block and 128-wide K tile: 64 lanes of four indices (9-bit for
# a 512-entry codebook, 10-bit for a 1,024-entry one) and 32 sign bits, plus a
# metadata byte per lane and atom whose low nibble holds the indices' ninth
# bits. With 10-bit indices, the tenth bits of atoms 0 and 1 fill the atom-two
# record's fourth metadata byte, and atom 2's a byte per lane after each
# two-K-tile slab (iq2r_format). Its high nibble holds
# ``exponent(b) * C + class(s(b))`` in lane group b, s = (0, 2, 1, 3): the
# lane that supplies block b's MFMA scale keeps b's exponent, and the class of
# the block a lane decodes codewords for sits in that lane or its neighbour
# group, one permlane16 swap away (see iq2r_levels in the kernels).

_PARTNER = (0, 2, 1, 3)


def _tile_bases(
    n_block: Tensor, k_tile: Tensor, k_tiles: int, entries: int
) -> tuple[Tensor, Tensor, Tensor]:
    """Triplet byte offset, atom and tenth-bit word offset of each (N block, K tile)."""

    slab_k_tiles = iq2r_slab_k_tiles(entries)
    block = n_block % IQ2R_N_BLOCKS_PER_GROUP
    atom = block % IQ2R_ATOMS_PER_TRIPLET
    triplet = block // IQ2R_ATOMS_PER_TRIPLET
    slab = (n_block // IQ2R_N_BLOCKS_PER_GROUP) * (
        k_tiles // slab_k_tiles
    ) + k_tile // slab_k_tiles
    slab_base = slab * iq2r_slab_bytes(entries)
    base = (
        slab_base
        + (k_tile % slab_k_tiles) * IQ2R_GROUP_BYTES
        + triplet * IQ2R_TRIPLET_BYTES
    )
    ext = slab_base + IQ2R_WIDE_EXT_OFFSET + triplet * 64
    return base, atom, ext


def _vector_offsets(
    n: int, k: int, device, entries: int = IQ2R_CODEBOOK_ENTRIES
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Byte offsets of every 8-vector's index byte, metadata byte and tenth-bit
    byte, its metadata bit and its tenth bit.

    Shapes ``[N, K/8]``. The tenth-bit offsets are meaningful only for a
    1,024-entry codebook. A vector of row r and columns [8v, 8v+8) sits in lane
    ``g*16 + r%16`` as codeword ``j``, where its 32-block b and position p in the
    block give ``g = 2*(b%2) + p//2`` and ``j = 2*(b//2) + p%2``.
    """

    k_tiles = iq2r_k_tiles(k)
    row = torch.arange(n, device=device)[:, None]
    vec = torch.arange(k // IQ2R_VECTOR_SIZE, device=device)[None, :]
    n_block, rr = row // IQ2R_TILE_N, row % IQ2R_TILE_N
    k_tile, within = vec // 16, vec % 16
    block, position = within // 4, within % 4
    group = 2 * (block % 2) + position // 2
    word = 2 * (block // 2) + position % 2
    lane = group * 16 + rr
    base, atom, ext_base = _tile_bases(n_block, k_tile, k_tiles, entries)
    record = torch.where(
        atom < 2,
        base + lane * 2 * IQ2R_LANE_RECORD_BYTES + atom * IQ2R_LANE_RECORD_BYTES,
        base + IQ2R_ATOM_TWO_RECORDS_OFFSET + lane * IQ2R_ATOM_TWO_RECORD_BYTES,
    )
    metadata = (
        base
        + IQ2R_ATOM_TWO_RECORDS_OFFSET
        + lane * IQ2R_ATOM_TWO_RECORD_BYTES
        + (IQ2R_ATOM_TWO_METADATA_OFFSET + atom)
    )
    # atoms 0-1: the atom-two record's fourth metadata byte; atom 2: the slab's
    # byte of the lane
    tenth_byte = (
        base
        + IQ2R_ATOM_TWO_RECORDS_OFFSET
        + lane * IQ2R_ATOM_TWO_RECORD_BYTES
        + IQ2R_WIDE_TENTH_BYTE
    )
    ext = torch.where(atom < 2, tenth_byte, ext_base + lane)
    ext_bit = torch.where(atom < 2, atom * 4, (k_tile % 2) * 4) + word
    return record + word, metadata, word, ext, ext_bit


def _level_offsets(
    n: int, k: int, device, entries: int = IQ2R_CODEBOOK_ENTRIES
) -> Tensor:
    """``[N, K/32]`` byte offsets of each 32-group's level nibble (lane ``b*16 + r%16``)."""

    k_tiles = iq2r_k_tiles(k)
    row = torch.arange(n, device=device)[:, None]
    blk = torch.arange(k // IQ2R_SCALE_BLOCK, device=device)[None, :]
    n_block, rr = row // IQ2R_TILE_N, row % IQ2R_TILE_N
    k_tile, b = blk // 4, blk % 4
    lane = b * 16 + rr
    base, atom, _ = _tile_bases(n_block, k_tile, k_tiles, entries)
    return (
        base
        + IQ2R_ATOM_TWO_RECORDS_OFFSET
        + lane * IQ2R_ATOM_TWO_RECORD_BYTES
        + (IQ2R_ATOM_TWO_METADATA_OFFSET + atom)
    )


def _nibbles(levels: Tensor, classes: int) -> Tensor:
    """Levels ``[N, K/32]`` -> stored nibbles (exponent of b, class of s(b)) per 128-wide tile."""

    n = levels.shape[0]
    tiles = levels.to(torch.int64).reshape(n, -1, 4)
    partner = tiles[..., list(_PARTNER)]
    return ((tiles // classes) * classes + partner % classes).reshape(n, -1)


def iq2r_v2_write_data(
    indices: Tensor,
    negative: Tensor,
    levels: Tensor,
    level_set: IQ2RV2Levels,
    codebook_entries: int = IQ2R_CODEBOOK_ENTRIES,
) -> Tensor:
    """Canonical data bytes of one matrix from ``indices [N, K/8]``, ``negative [N, K]``, ``levels [N, K/32]``."""

    n, k = negative.shape
    device = indices.device
    if int(indices.max()) >= codebook_entries or int(indices.min()) < 0:
        raise ValueError(f"IQ2R indices must be in 0..{codebook_entries - 1}")
    data = torch.zeros(
        iq2r_data_bytes(n, k, codebook_entries), dtype=torch.int64, device=device
    )
    index_at, metadata_at, word, ext_at, ext_bit = _vector_offsets(
        n, k, device, codebook_entries
    )
    data.index_put_((index_at.flatten(),), (indices & 255).flatten())
    bits = negative.reshape(n, -1, IQ2R_VECTOR_SIZE).to(torch.int64)
    signs = (bits << torch.arange(IQ2R_VECTOR_SIZE, device=device)).sum(-1)
    data.index_put_(((index_at + 4).flatten(),), signs.flatten())
    # Ninth index bits (low nibble) and levels (high nibble) of the metadata bytes.
    data.index_put_(
        (metadata_at.flatten(),),
        (((indices >> 8) & 1) << word).flatten(),
        accumulate=True,
    )
    if codebook_entries == IQ2R_WIDE_CODEBOOK_ENTRIES:
        data.index_put_(
            (ext_at.flatten(),),
            (((indices >> 9) & 1) << ext_bit).flatten(),
            accumulate=True,
        )
    nibbles = _nibbles(levels, level_set.classes)
    data.index_put_(
        (_level_offsets(n, k, device, codebook_entries).flatten(),),
        (nibbles << 4).flatten(),
        accumulate=True,
    )
    return data.to(torch.uint8)


def iq2r_v2_write_auxiliary(
    codebooks: Tensor, d: Tensor, gains: tuple[float, float]
) -> Tensor:
    """Canonical auxiliary bytes: ``codebooks [B, entries, 8]`` E4M3, ``d [N, K/256]``, two FP32 gains."""

    books = codebooks.to(torch.float8_e4m3fn).view(torch.uint8).reshape(-1)
    gain_bytes = torch.tensor(gains, dtype=torch.float32).view(torch.uint8).to(d.device)
    return torch.cat((books.to(d.device), d.reshape(-1).to(torch.uint8), gain_bytes))


def iq2r_v2_read(
    data: Tensor,
    auxiliary: Tensor,
    n: int,
    k: int,
    level_set: IQ2RV2Levels,
    codebook_entries: int = IQ2R_CODEBOOK_ENTRIES,
):
    """Inverse of the writers: ``(indices, negative, levels, codebook, d, gains)``."""

    data = data.to(torch.int64)
    device = data.device
    index_at, metadata_at, word, ext_at, ext_bit = _vector_offsets(
        n, k, device, codebook_entries
    )
    indices = data[index_at] | (((data[metadata_at] >> word) & 1) << 8)
    if codebook_entries == IQ2R_WIDE_CODEBOOK_ENTRIES:
        indices |= ((data[ext_at] >> ext_bit) & 1) << 9
    signs = data[index_at + 4]
    negative = (
        ((signs[..., None] >> torch.arange(IQ2R_VECTOR_SIZE, device=device)) & 1)
        .bool()
        .reshape(n, k)
    )
    nibbles = (data[_level_offsets(n, k, device, codebook_entries)] >> 4).reshape(
        n, -1, 4
    )
    classes = level_set.classes
    # s is its own inverse: class(b) is stored in group s(b).
    levels = (
        (nibbles // classes) * classes + nibbles[..., list(_PARTNER)] % classes
    ).reshape(n, -1)
    levels = levels.to(torch.uint8)
    book_bytes = codebook_entries * IQ2R_VECTOR_SIZE
    codebook = (
        auxiliary[:book_bytes]
        .view(torch.float8_e4m3fn)
        .float()
        .reshape(codebook_entries, IQ2R_VECTOR_SIZE)
    )
    d_end = book_bytes + n * iq2r_v2_d_columns(k)
    d = auxiliary[book_bytes:d_end].reshape(n, -1)
    gains = tuple(
        auxiliary[-IQ2R_V2_GAIN_BYTES:].contiguous().view(torch.float32).tolist()
    )
    return indices, negative, levels, codebook, d, gains


@torch.no_grad()
def iq2r_v2_materialize(
    data: Tensor,
    auxiliary: Tensor,
    n: int,
    k: int,
    levels: IQ2RV2Levels,
    *,
    apply_gain: bool = True,
    codebook_entries: int = IQ2R_CODEBOOK_ENTRIES,
) -> Tensor:
    """Reference decode of one canonical IQ2R-v2 matrix to FP32 ``[N, K]``.

    The interleaved gate/up matrix applies gain 0 to even rows and gain 1 to
    odd rows.
    """

    indices, negative, level, codebook, d, gains = iq2r_v2_read(
        data, auxiliary, n, k, levels, codebook_entries
    )
    copies = iq2r_v2_codebook_copies(codebook, levels)
    level = level.long()
    group_class = level % levels.classes  # [N, K/32]
    vector_class = group_class.repeat_interleave(4, dim=1)
    codes = copies[vector_class, indices]  # [N, K/8, 8]
    exponent = (
        d.float().repeat_interleave(8, dim=1) - 127 + (level // levels.classes).float()
    )
    scale = torch.exp2(exponent).repeat_interleave(4, dim=1)[..., None]
    values = (codes * scale).reshape(n, k)
    values = torch.where(negative, -values, values)
    if apply_gain:
        gain = torch.tensor(gains, dtype=torch.float32, device=data.device)
        values = values * gain[torch.arange(n, device=data.device) % 2][:, None]
    return values
