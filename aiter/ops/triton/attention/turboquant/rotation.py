# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""
Rotation and projection matrices for TurboQuant.

Two matrices are used:

  Π  (Pi)  — random orthogonal rotation matrix.
             Generated via QR decomposition of a random Gaussian matrix.
             Used by TurboQuantMSE: rotate keys before scalar quantization
             so that each coordinate becomes Beta-distributed and nearly
             independent, making per-coordinate Lloyd-Max optimal.

  S        — Gaussian random projection matrix for QJL.
             Each entry drawn i.i.d. from N(0, 1/d).
             Used by TurboQuantProd: project the quantization residual
             to obtain a 1-bit sketch that corrects inner-product bias.
             Intentionally NOT orthogonalized (paper uses Gaussian S).

Both matrices are cached per (head_dim, device) to ensure consistency
within a run and avoid redundant computation.
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch

# Cache: (head_dim, device_str) -> matrix
_PI_CACHE: Dict[Tuple[int, str], torch.Tensor] = {}
_S_CACHE: Dict[Tuple[int, str], torch.Tensor] = {}

# Fixed seed so matrices are reproducible within a process restart.
# A per-layer seed would be more rigorous; this is sufficient for correctness.
_DEFAULT_SEED = 1234


def get_rotation_matrix(
    head_dim: int,
    device: torch.device,
    seed: int = _DEFAULT_SEED,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Return (or create) the cached random orthogonal rotation matrix Π.

    Π is a (head_dim × head_dim) orthogonal matrix generated once per
    (head_dim, device) via QR decomposition of a random Gaussian matrix.
    The same Π must be used for both compression and decompression.

    Args:
        head_dim: Transformer head dimension d.
        device:   Target device.
        seed:     RNG seed for reproducibility.
        dtype:    Float dtype (fp32 recommended for numerical stability).

    Returns:
        Pi: (head_dim, head_dim) orthogonal tensor on `device`.
    """
    key = (head_dim, str(device), seed)
    if key not in _PI_CACHE:
        gen = torch.Generator()
        gen.manual_seed(seed)
        # Draw a random Gaussian matrix and orthogonalize via QR
        G = torch.randn(head_dim, head_dim, generator=gen, dtype=torch.float64)
        Q, _ = torch.linalg.qr(G)
        # Cast to requested dtype and move to device
        _PI_CACHE[key] = Q.to(dtype=dtype, device=device)
    return _PI_CACHE[key]


def get_qjl_matrix(
    head_dim: int,
    device: torch.device,
    seed: int = _DEFAULT_SEED + 1,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Return (or create) the cached Gaussian QJL projection matrix S.

    S is a (head_dim × head_dim) matrix with entries drawn i.i.d. from
    N(0, 1/head_dim).  It is intentionally NOT orthogonalized — the paper's
    QJL construction uses a plain Gaussian S.

    In TurboQuantProd:
      - The residual r = x - x̂  is projected to get sketch = sign(S @ r)
      - At query time: inner-product correction = ||r|| * <sign(S r), sign(S q)> / d

    Args:
        head_dim: Transformer head dimension d.
        device:   Target device.
        seed:     RNG seed (different from Π seed by default).
        dtype:    Float dtype.

    Returns:
        S: (head_dim, head_dim) Gaussian tensor on `device`.
    """
    key = (head_dim, str(device), seed)
    if key not in _S_CACHE:
        gen = torch.Generator()
        gen.manual_seed(seed)
        # i.i.d. N(0, 1/d) entries — plain Gaussian, not orthogonalized
        S = torch.randn(head_dim, head_dim, generator=gen, dtype=torch.float64)
        S = S / math.sqrt(head_dim)
        _S_CACHE[key] = S.to(dtype=dtype, device=device)
    return _S_CACHE[key]


def clear_cache() -> None:
    """Clear all cached rotation and QJL matrices (useful for testing)."""
    _PI_CACHE.clear()
    _S_CACHE.clear()


# ---------------------------------------------------------------------------
# math import at module level (used in get_qjl_matrix)
# ---------------------------------------------------------------------------
import math  # noqa: E402  (moved here to keep top-of-file imports clean)
