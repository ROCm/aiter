# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""
TurboQuant quantizers: Algorithm 1 (MSE) and Algorithm 2 (Prod).

TurboQuantMSE  — Algorithm 1 from the paper.
  compress(x):
    1. Rotate: x_rot = Π @ x
    2. Quantize each coordinate to nearest codebook centroid → indices
    3. Store (indices, ||x||) to allow norm-preserving reconstruction

  decompress(compressed):
    1. Look up centroids for each index
    2. Rotate back: x̂ = Πᵀ @ x_rot_hat  (Π is orthogonal so Π⁻¹ = Πᵀ)

TurboQuantProd — Algorithm 2 from the paper.
  Adds a 1-bit QJL residual correction on top of MSE to produce an
  unbiased inner-product estimator:
    compress(x):
      1. MSE compress as above → (indices, norm)
      2. Compute residual r = x - x̂
      3. QJL sketch: signs = sign(S @ r), res_norm = ||r||
      4. Store (indices, norm, signs, res_norm)

  score(q, compressed):
      MSE score  = q_rot · centroid_lookup(indices)   (where q_rot = Π @ q)
      QJL correction = res_norm * (q_sketch · signs) / d
                                  (where q_sketch = S @ q)
      total = (MSE score + QJL correction) * norm

ValueQuantizer — simple group-wise quantization for V tensors.
  Values are not dot-product scored, so the rotation trick is not needed.
  We use per-group affine quantization (similar to AWQ / GPTQ style).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from .codebook import get_codebook
from .rotation import get_rotation_matrix, get_qjl_matrix
from .utils import pack_indices, unpack_indices


# ---------------------------------------------------------------------------
# Compressed data containers
# ---------------------------------------------------------------------------

@dataclass
class CompressedKeys:
    """
    Stores a batch of compressed key vectors (TurboQuantMSE output).

    Shapes (batch dims = leading dims of original key tensor):
      mse_indices : (*batch, d_packed)  uint8  — packed codebook indices
      norms       : (*batch,)           fp16   — original L2 norms of x
      Pi          : (d, d)              fp32   — rotation matrix (shared)
      bits        : int                        — bits per coordinate
      head_dim    : int                        — original d
    """
    mse_indices: torch.Tensor   # (*batch, d_packed) uint8
    norms: torch.Tensor         # (*batch,)          fp16
    Pi: torch.Tensor            # (d, d)             fp32
    bits: int
    head_dim: int


@dataclass
class CompressedKeysProd(CompressedKeys):
    """
    TurboQuantProd output: MSE fields + QJL residual fields.

      qjl_signs : (*batch, d//8)  uint8  — packed sign bits of S @ r
      res_norms : (*batch,)       fp16   — ||r||
      S         : (d, d)          fp32   — QJL projection matrix (shared)
    """
    qjl_signs: torch.Tensor   # (*batch, d//8)  uint8
    res_norms: torch.Tensor   # (*batch,)        fp16
    S: torch.Tensor           # (d, d)           fp32


@dataclass
class CompressedValues:
    """
    Stores a batch of group-quantized value vectors.

      indices    : (*batch, d_packed)          uint8  — packed quantized ints
      scales     : (*batch, d // group_size)   fp16   — per-group scale
      zeros      : (*batch, d // group_size)   fp16   — per-group zero point
      bits        : int
      group_size  : int
      head_dim    : int
    """
    indices: torch.Tensor    # (*batch, d_packed)         uint8
    scales: torch.Tensor     # (*batch, d // group_size)  fp16
    zeros: torch.Tensor      # (*batch, d // group_size)  fp16
    bits: int
    group_size: int
    head_dim: int


# ---------------------------------------------------------------------------
# TurboQuantMSE — Algorithm 1
# ---------------------------------------------------------------------------

class TurboQuantMSE(nn.Module):
    """
    MSE-optimal key compressor using random rotation + Lloyd-Max codebook.

    Usage::
        quantizer = TurboQuantMSE(head_dim=128, bits=3, device=device)
        compressed = quantizer.compress(k)   # k: (..., d)
        k_hat = quantizer.decompress(compressed)  # (..., d)
    """

    def __init__(
        self,
        head_dim: int,
        bits: int,
        device: Optional[torch.device] = None,
        seed: int = 1234,
    ) -> None:
        super().__init__()
        if bits not in (2, 3, 4):
            raise ValueError(f"bits must be 2, 3, or 4; got {bits}")

        self.head_dim = head_dim
        self.bits = bits
        self.n_levels = 2 ** bits

        dev = device or torch.device("cpu")
        Pi = get_rotation_matrix(head_dim, dev, seed=seed)
        codebook = get_codebook(head_dim, bits, device=dev)

        # Register as buffers so they move with .to(device) / state_dict
        self.register_buffer("Pi", Pi)
        self.register_buffer("codebook", codebook)  # (2^bits,)

    # ------------------------------------------------------------------
    def compress(self, x: torch.Tensor) -> CompressedKeys:
        """
        Compress key vectors.

        Args:
            x: (..., head_dim) float tensor (any leading batch dims).

        Returns:
            CompressedKeys with packed MSE indices and L2 norms.
        """
        batch_shape = x.shape[:-1]
        d = self.head_dim

        # Compute and store L2 norms for reconstruction
        norms = x.norm(dim=-1)  # (...,)

        # Normalize to unit sphere, then rotate.
        # Rotation preserves norms, so we rotate unit vectors.
        x_unit = x.float() / norms.unsqueeze(-1).clamp(min=1e-8)
        x_rot = x_unit @ self.Pi.T  # (..., d) — unit-norm in rotated space

        # Scale by sqrt(d) to match the codebook distribution.
        # Codebook was generated for coordinates scaled to N(0,1);
        # raw rotated coordinates of unit-norm vectors are N(0, 1/d).
        x_rot_scaled = x_rot * math.sqrt(d)  # (..., d) — now O(1), matches codebook

        # Quantize: for each coordinate find the nearest centroid index
        dists = (x_rot_scaled.unsqueeze(-1) - self.codebook).abs()  # (..., d, 2^b)
        indices = dists.argmin(dim=-1).to(torch.uint8)  # (..., d)  values in [0, 2^b)

        # Bit-pack indices
        packed = pack_indices(indices, self.bits)  # (..., d_packed)

        return CompressedKeys(
            mse_indices=packed,
            norms=norms.to(torch.float16),
            Pi=self.Pi,
            bits=self.bits,
            head_dim=d,
        )

    # ------------------------------------------------------------------
    def decompress(self, compressed: CompressedKeys) -> torch.Tensor:
        """
        Decompress key vectors.

        Args:
            compressed: CompressedKeys from compress().

        Returns:
            x_hat: (..., head_dim) float32 reconstructed keys.
        """
        d = compressed.head_dim

        # Unpack indices
        indices = unpack_indices(
            compressed.mse_indices, compressed.bits, d
        )  # (..., d)  int64

        # Lookup centroids (on the sqrt(d) scale used during compression)
        x_rot_scaled_hat = self.codebook[indices]  # (..., d)

        # Unscale: divide by sqrt(d) to recover unit-sphere coordinates
        x_rot_hat = x_rot_scaled_hat / math.sqrt(d)  # (..., d)

        # Rotate back: we stored x_rot = x_unit @ Πᵀ, so x_unit = x_rot @ Π
        x_unit_hat = x_rot_hat @ compressed.Pi  # (..., d) — approximately unit-norm

        # Restore original scale
        norms = compressed.norms.float()  # (...,)
        x_hat = x_unit_hat * norms.unsqueeze(-1)

        return x_hat


# ---------------------------------------------------------------------------
# TurboQuantProd — Algorithm 2
# ---------------------------------------------------------------------------

class TurboQuantProd(TurboQuantMSE):
    """
    Unbiased inner-product key compressor: MSE quantization + QJL residual.

    On top of TurboQuantMSE, this stores a 1-bit QJL sketch of the
    quantization residual r = x - x̂, allowing the estimator:

        <q, x> ≈ MSE_score(q, compressed) + QJL_correction(q, compressed)

    to be unbiased in expectation (Theorem 2 in the paper).

    The QJL correction estimates <q, r> via:
        res_norm * ((S @ q) · sign(S @ r)) / d
    where S is a Gaussian matrix.  Note the query projection (S @ q) is
    kept continuous — only the key-side residual is binarized.  This gives
    a lower-variance estimator than binarizing both sides.

    Usage::
        quantizer = TurboQuantProd(head_dim=128, bits=3, device=device)
        compressed = quantizer.compress(x)
        score = quantizer.inner_product_score(q, compressed)  # unbiased
    """

    def __init__(
        self,
        head_dim: int,
        bits: int,
        device: Optional[torch.device] = None,
        seed: int = 1234,
    ) -> None:
        super().__init__(head_dim, bits, device=device, seed=seed)

        dev = device or torch.device("cpu")
        S = get_qjl_matrix(head_dim, dev, seed=seed + 1)
        self.register_buffer("S", S)

    # ------------------------------------------------------------------
    def compress(self, x: torch.Tensor) -> CompressedKeysProd:  # type: ignore[override]
        """
        Compress key vectors with QJL residual sketch.

        Args:
            x: (..., head_dim) float tensor.

        Returns:
            CompressedKeysProd with MSE fields + QJL fields.
        """
        # MSE step
        base = super().compress(x)

        # Reconstruct x̂ to compute residual
        x_hat = super().decompress(base)   # (..., d)

        # Residual
        r = x.float() - x_hat             # (..., d)

        # QJL sketch: sign(S @ r)
        sketch = r @ self.S.T             # (..., d)  (equiv to (S @ r.T).T)
        signs_bool = (sketch >= 0)         # (..., d)  bool

        # Pack sign bits: 1 byte per 8 dimensions
        d = self.head_dim
        # Reshape to (..., d//8, 8) and pack into uint8
        signs_uint8 = _pack_signs(signs_bool)  # (..., d//8)

        # Residual norms
        res_norms = r.norm(dim=-1).to(torch.float16)  # (...,)

        return CompressedKeysProd(
            mse_indices=base.mse_indices,
            norms=base.norms,
            Pi=base.Pi,
            bits=base.bits,
            head_dim=base.head_dim,
            qjl_signs=signs_uint8,
            res_norms=res_norms,
            S=self.S,
        )

    # ------------------------------------------------------------------
    def inner_product_score(
        self,
        q: torch.Tensor,
        compressed: CompressedKeysProd,
    ) -> torch.Tensor:
        """
        Compute unbiased inner product estimate <q, x> for each key.

        Args:
            q:          (..._q, head_dim) query tensor.
            compressed: CompressedKeysProd from compress(x).
                        Batch dims must be broadcastable with q.

        Returns:
            scores: broadcastable float32 tensor of dot product estimates.
        """
        d = self.head_dim

        # --- MSE score ---
        # <q, k_hat> = k_norm / sqrt(d) * (q @ Πᵀ) · codebook[idx]
        # because k_hat = (codebook[idx] / sqrt(d)) @ Π * k_norm
        q_rot = q.float() @ self.Pi.T          # (..._q, d)
        k_indices = unpack_indices(
            compressed.mse_indices, compressed.bits, d
        )                                       # (..._k, d)
        k_rot_hat = self.codebook[k_indices]    # (..._k, d) — codebook (sqrt(d)) scale
        # Divide by sqrt(d) to account for the coordinate scaling applied at compress time
        mse_score = (q_rot.unsqueeze(-2) * k_rot_hat).sum(dim=-1) / math.sqrt(d)
        mse_score = mse_score * compressed.norms.float()

        # --- QJL correction for <q, r> ---
        # Estimator: ||r|| * (S @ q) · sign(S @ r) / d
        # q_sketch stays continuous (NOT binarized) — lower variance than sign(Sq)·sign(Sr)
        q_sketch = q.float() @ self.S.T        # (..._q, d)  continuous

        k_signs = _unpack_signs(
            compressed.qjl_signs, d
        )                                       # (..._k, d)  bool
        k_signs_float = k_signs.float() * 2 - 1  # {-1, +1}

        # (S @ q) · sign(S @ r)
        qjl_dot = (q_sketch.unsqueeze(-2) * k_signs_float).sum(dim=-1)

        correction = compressed.res_norms.float() * qjl_dot / d

        return mse_score + correction


# ---------------------------------------------------------------------------
# ValueQuantizer — group-wise affine quantization for V tensors
# ---------------------------------------------------------------------------

class ValueQuantizer(nn.Module):
    """
    Group-wise affine quantization for transformer value (V) tensors.

    Values are not inner-product scored against queries at compress time,
    so the rotation + codebook approach used for keys is unnecessary.
    Instead we use simple per-group min-max (affine) quantization.

    Supported:
      - bits=2: 4 levels per group
      - bits=4: 16 levels per group

    Usage::
        vq = ValueQuantizer(bits=4, group_size=32)
        compressed = vq.compress(v)   # v: (..., d)
        v_hat = vq.decompress(compressed)   # (..., d)
    """

    def __init__(self, bits: int = 4, group_size: int = 32) -> None:
        super().__init__()
        if bits not in (2, 4):
            raise ValueError(f"ValueQuantizer bits must be 2 or 4; got {bits}")
        self.bits = bits
        self.group_size = group_size
        self.n_levels = 2 ** bits

    # ------------------------------------------------------------------
    def compress(self, v: torch.Tensor) -> CompressedValues:
        """
        Quantize value vectors group-wise.

        Args:
            v: (..., head_dim) float tensor.

        Returns:
            CompressedValues.
        """
        d = v.shape[-1]
        if d % self.group_size != 0:
            raise ValueError(
                f"head_dim {d} is not divisible by group_size {self.group_size}"
            )

        batch_shape = v.shape[:-1]
        n_groups = d // self.group_size

        # Reshape to (..., n_groups, group_size) for group-wise stats
        v_grouped = v.float().reshape(*batch_shape, n_groups, self.group_size)

        v_min = v_grouped.min(dim=-1).values   # (..., n_groups)
        v_max = v_grouped.max(dim=-1).values   # (..., n_groups)

        # Affine scale and zero-point
        scale = (v_max - v_min) / (self.n_levels - 1)          # (..., n_groups)
        scale = scale.clamp(min=1e-8)
        zero = v_min                                            # (..., n_groups)

        # Quantize: integer in [0, n_levels - 1]
        v_shifted = v_grouped - zero.unsqueeze(-1)              # (..., n_groups, g)
        indices = (v_shifted / scale.unsqueeze(-1)).round().clamp(0, self.n_levels - 1)
        indices = indices.to(torch.uint8)                       # (..., n_groups, g)

        # Flatten back and pack
        indices_flat = indices.reshape(*batch_shape, d)         # (..., d)
        packed = pack_indices(indices_flat, self.bits)          # (..., d_packed)

        return CompressedValues(
            indices=packed,
            scales=scale.to(torch.float16),
            zeros=zero.to(torch.float16),
            bits=self.bits,
            group_size=self.group_size,
            head_dim=d,
        )

    # ------------------------------------------------------------------
    def decompress(self, compressed: CompressedValues) -> torch.Tensor:
        """
        Dequantize value vectors.

        Args:
            compressed: CompressedValues from compress().

        Returns:
            v_hat: (..., head_dim) float32 reconstructed values.
        """
        d = compressed.head_dim
        g = compressed.group_size
        n_groups = d // g

        # Unpack
        indices_flat = unpack_indices(
            compressed.indices, compressed.bits, d
        )  # (..., d)  int64

        batch_shape = indices_flat.shape[:-1]
        indices_grouped = indices_flat.reshape(*batch_shape, n_groups, g)  # (..., n_g, g)

        scale = compressed.scales.float()   # (..., n_groups)
        zero = compressed.zeros.float()     # (..., n_groups)

        # Dequantize
        v_hat = indices_grouped.float() * scale.unsqueeze(-1) + zero.unsqueeze(-1)
        v_hat = v_hat.reshape(*batch_shape, d)

        return v_hat


# ---------------------------------------------------------------------------
# Helper: sign bit-packing
# ---------------------------------------------------------------------------

def _pack_signs(signs_bool: torch.Tensor) -> torch.Tensor:
    """
    Pack a boolean tensor (..., d) into uint8 (..., d//8).

    Bit 0 of byte k corresponds to signs_bool[..., k*8+0], etc.
    Requires d to be a multiple of 8.
    """
    d = signs_bool.shape[-1]
    if d % 8 != 0:
        # Pad to next multiple of 8
        pad = 8 - (d % 8)
        signs_bool = torch.nn.functional.pad(signs_bool, (0, pad), value=False)
        d = signs_bool.shape[-1]

    batch_shape = signs_bool.shape[:-1]
    # (..., d//8, 8) — last dim is 8 bits per byte
    bits = signs_bool.reshape(*batch_shape, d // 8, 8).to(torch.uint8)
    # Pack: multiply by powers of 2 and sum
    powers = torch.tensor([1, 2, 4, 8, 16, 32, 64, 128], dtype=torch.uint8, device=signs_bool.device)
    packed = (bits * powers).sum(dim=-1).to(torch.uint8)  # (..., d//8)
    return packed


def _unpack_signs(packed: torch.Tensor, d: int) -> torch.Tensor:
    """
    Unpack uint8 (..., d//8) back to bool (..., d).
    """
    batch_shape = packed.shape[:-1]
    n_bytes = packed.shape[-1]
    powers = torch.tensor([1, 2, 4, 8, 16, 32, 64, 128], dtype=torch.uint8, device=packed.device)
    # (..., n_bytes, 8)
    bits = ((packed.unsqueeze(-1) & powers) > 0).reshape(*batch_shape, n_bytes * 8)
    # Trim to original d
    return bits[..., :d]
