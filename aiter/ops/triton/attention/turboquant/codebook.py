# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""
Lloyd-Max optimal codebook generation for TurboQuant.

After a random orthogonal rotation Π, each coordinate of a unit-norm
vector follows a Beta(d/2, d/2) distribution (after scaling by √d).
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

# Directory where pre-generated codebooks are stored
_CONFIGS_DIR = Path(__file__).parent / "configs"

# Supported configurations
SUPPORTED_HEAD_DIMS = (64, 128, 256)
SUPPORTED_BITS = (2, 3, 4)


def _beta_pdf(x: np.ndarray, alpha: float, beta: float) -> np.ndarray:
    """Beta distribution PDF (unnormalized for sampling purposes)."""
    from scipy.special import betaln

    log_norm = betaln(alpha, beta)
    log_pdf = (
        (alpha - 1) * np.log(np.clip(x, 1e-300, None))
        + (beta - 1) * np.log(np.clip(1 - x, 1e-300, None))
        - log_norm
    )
    return np.exp(log_pdf)


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
    # Return the first-coordinate marginal, scaled by sqrt(d) for unit variance
    return z[:, 0] * math.sqrt(head_dim)


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
    MSE under the Beta(d/2, d/2) coordinate distribution that arises
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
    """Return the path for a pre-generated codebook file."""
    return _CONFIGS_DIR / f"codebook_d{head_dim}_b{bits}.pt"


def get_codebook(
    head_dim: int,
    bits: int,
    device: Optional[torch.device] = None,
    force_regenerate: bool = False,
) -> torch.Tensor:
    """
    Load or generate a Lloyd-Max codebook for (head_dim, bits).

    Loads from a pre-generated .pt file if available; otherwise generates
    the codebook via Lloyd-Max iteration and saves it to disk.

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
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(codebook, path)

    if device is not None:
        codebook = codebook.to(device)

    return codebook


def pregenerate_all_codebooks(verbose: bool = True) -> None:
    """
    Pre-generate and save all codebooks for standard (head_dim, bits) configs.

    Call this once at repo setup time or in a build step.
    Generates 9 codebooks: head_dim ∈ {64,128,256} × bits ∈ {2,3,4}.
    """
    _CONFIGS_DIR.mkdir(parents=True, exist_ok=True)
    for head_dim in SUPPORTED_HEAD_DIMS:
        for bits in SUPPORTED_BITS:
            path = get_codebook_path(head_dim, bits)
            if path.exists():
                if verbose:
                    print(f"  [skip] codebook_d{head_dim}_b{bits}.pt already exists")
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


if __name__ == "__main__":
    print("Generating TurboQuant codebooks...")
    pregenerate_all_codebooks(verbose=True)
    print("All codebooks generated.")
