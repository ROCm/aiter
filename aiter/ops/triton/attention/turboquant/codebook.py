# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""
Lloyd-Max optimal codebook generation for TurboQuant.

After a random orthogonal rotation Π, each coordinate of a unit-norm
vector follows the distribution (Lemma 1 of the paper):
    f(x) = Γ(d/2) / (√π · Γ((d-1)/2)) · (1 - x²)^((d-3)/2)  for x ∈ [-1, 1]
which is Beta((d-1)/2, (d-1)/2) on [0,1].
Codebook centroids are stored in [-1, 1] space — no sqrt(d) scaling (matches paper).
Lloyd-Max iteration finds the MSE-optimal scalar quantizer for this
distribution, giving us the codebooks used by TurboQuantMSE.

Codebooks are pre-generated and cached as .pt files to avoid the
expensive iterative computation at model initialization time.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import torch
import numpy as np

# Codebook cache directory — user home cache, always writable.
_CONFIGS_DIR = Path.home() / ".cache" / "turboquant"

# Supported configurations
SUPPORTED_HEAD_DIMS = (64, 128, 256)
SUPPORTED_BITS = (2, 3, 4)


def _sample_beta_coords(
    head_dim: int, n_samples: int = 2_000_000, seed: int = 42
) -> np.ndarray:
    """
    Sample the distribution of a single rotated coordinate.

    After rotating a unit-norm vector x (||x||=1) in R^d by a random
    orthogonal matrix, each coordinate y_i satisfies:
        y_i^2 ~ Beta(1/2, (d-1)/2)  (scaled)
    Equivalently, y_i = sign * sqrt(Beta(1/2,(d-1)/2)) which, after
    rescaling by sqrt(d), becomes approximately N(0,1) for large d.

    For the Lloyd-Max codebook we work in the rotated, norm-1 space, so
    each coordinate is in [-1/√d, 1/√d] * √d  ≡ [-1, 1] after normalization.

    In practice we draw samples from a standard Gaussian and normalize to
    unit sphere, then take one coordinate — this is the exact distribution.
    """
    rng = np.random.default_rng(seed)
    # Draw unit-sphere samples, take first coordinate
    z = rng.standard_normal((n_samples, head_dim)).astype(np.float64)
    norms = np.linalg.norm(z, axis=1, keepdims=True)
    z /= norms
    # Return the first-coordinate marginal in [-1, 1] — matching the paper's
    # Beta((d-1)/2, (d-1)/2) distribution directly, no sqrt(d) scaling.
    return z[:, 0]


def _lloyd_max_iteration(
    samples: np.ndarray,
    n_levels: int,
    n_iter: int = 300,
    tol: float = 1e-8,
) -> np.ndarray:
    """
    Lloyd-Max algorithm: find MSE-optimal scalar quantizer boundaries and centroids.

    Args:
        samples:   1-D array of samples from the target distribution.
        n_levels:  Number of quantization levels (2^bits).
        n_iter:    Maximum number of iterations.
        tol:       Convergence tolerance on centroid movement.

    Returns:
        centroids: (n_levels,) array of reconstruction values.
    """
    # Initialize centroids at uniform quantiles
    percentiles = np.linspace(0, 100, n_levels + 2)[1:-1]
    centroids = np.percentile(samples, percentiles)

    for _ in range(n_iter):
        # E-step: assign each sample to nearest centroid
        # Boundaries are midpoints between adjacent centroids
        boundaries = (centroids[:-1] + centroids[1:]) / 2.0
        indices = np.searchsorted(boundaries, samples)

        # M-step: recompute centroids as cluster means
        new_centroids = np.array(
            [
                samples[indices == k].mean() if np.any(indices == k) else centroids[k]
                for k in range(n_levels)
            ]
        )

        # Check convergence
        if np.max(np.abs(new_centroids - centroids)) < tol:
            break
        centroids = new_centroids

    return centroids.astype(np.float32)


def generate_lloyd_max_codebook(
    head_dim: int,
    bits: int,
    n_samples: int = 2_000_000,
    seed: int = 42,
) -> torch.Tensor:
    """
    Generate a Lloyd-Max optimal codebook for the given (head_dim, bits).

    The codebook contains 2^bits reconstruction centroids optimized for
    MSE under the Beta((d-1)/2, (d-1)/2) coordinate distribution that arises
    after the random orthogonal rotation Π.

    Args:
        head_dim:  Transformer head dimension (d).
        bits:      Bits per coordinate (2, 3, or 4).
        n_samples: Number of Monte-Carlo samples for Lloyd-Max.
        seed:      RNG seed for reproducibility.

    Returns:
        codebook: (2**bits,) float32 tensor of sorted centroids.
    """
    try:
        import scipy  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "scipy is required for codebook generation. "
            "Install it with: pip install scipy"
        ) from e

    n_levels = 2**bits
    samples = _sample_beta_coords(head_dim, n_samples=n_samples, seed=seed)
    centroids = _lloyd_max_iteration(samples, n_levels)
    centroids_sorted = np.sort(centroids)
    return torch.from_numpy(centroids_sorted)


def get_codebook_path(head_dim: int, bits: int) -> Path:
    """Return the codebook file path under _CONFIGS_DIR."""
    return _CONFIGS_DIR / f"codebook_d{head_dim}_b{bits}.pt"


def get_codebook(
    head_dim: int,
    bits: int,
    device: Optional[torch.device] = None,
    force_regenerate: bool = False,
) -> torch.Tensor:
    """
    Load or generate a Lloyd-Max codebook for (head_dim, bits).

    Loads from _CONFIGS_DIR if the file exists; otherwise runs Lloyd-Max
    and saves to _CONFIGS_DIR (user home cache — always writable).

    Args:
        head_dim:         Transformer head dimension.
        bits:             Bits per coordinate (2, 3, or 4).
        device:           Target device for the returned tensor.
        force_regenerate: Re-run Lloyd-Max even if a cached file exists.

    Returns:
        codebook: (2**bits,) float32 tensor of reconstruction centroids.
    """
    path = get_codebook_path(head_dim, bits)

    if not force_regenerate and path.exists():
        codebook = torch.load(path, map_location="cpu", weights_only=True)
    else:
        codebook = generate_lloyd_max_codebook(head_dim, bits)
        _CONFIGS_DIR.mkdir(parents=True, exist_ok=True)
        torch.save(codebook, path)

    if device is not None:
        codebook = codebook.to(device)

    return codebook


def pregenerate_codebooks(
    head_dims: tuple = SUPPORTED_HEAD_DIMS,
    bits_list: tuple = SUPPORTED_BITS,
    verbose: bool = True,
) -> None:
    """
    Pre-generate and save codebooks for the given (head_dim, bits) configs.

    Args:
        head_dims: Tuple of head dimensions to generate codebooks for.
                   Defaults to SUPPORTED_HEAD_DIMS = (64, 128, 256).
        bits_list: Tuple of bit-widths to generate codebooks for.
                   Defaults to SUPPORTED_BITS = (2, 3, 4).
        verbose:   Print progress.
    """
    _CONFIGS_DIR.mkdir(parents=True, exist_ok=True)
    for head_dim in head_dims:
        for bits in bits_list:
            path = get_codebook_path(head_dim, bits)
            if path.exists():
                if verbose:
                    print(
                        f"  [skip] codebook_d{head_dim}_b{bits}.pt already exists at {path}"
                    )
                continue
            if verbose:
                print(
                    f"  [gen]  codebook_d{head_dim}_b{bits}.pt ...", end="", flush=True
                )
            cb = generate_lloyd_max_codebook(head_dim, bits)
            torch.save(cb, path)
            if verbose:
                print(
                    f" done  ({cb.shape[0]} levels, range [{cb[0]:.4f}, {cb[-1]:.4f}])"
                )
                print(f"         saved to {path}")


if __name__ == "__main__":
    print("Generating TurboQuant codebooks...")
    pregenerate_codebooks(verbose=True)
    print("All codebooks generated.")
