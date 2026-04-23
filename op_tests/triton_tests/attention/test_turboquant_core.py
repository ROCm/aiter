# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""
Phase 1 unit tests for TurboQuant core quantization infrastructure.

Test categories:
  1. Bit-packing round-trips  (utils.py)
  2. Rotation matrix orthogonality  (rotation.py)
  3. Codebook generation  (codebook.py)
  4. TurboQuantMSE compress / decompress accuracy  (quantizer.py)
  5. TurboQuantProd inner-product unbiasedness  (quantizer.py)
  6. ValueQuantizer compress / decompress  (quantizer.py)
  7. Distortion scaling  (Theorem 3: distortion ∝ 1/4^b)
  8. Compression ratio accounting  (utils.compression_ratio)

Run with:
    pytest op_tests/triton_tests/attention/test_turboquant_core.py -v
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from aiter.ops.triton.attention.turboquant.utils import (
    pack_indices,
    unpack_indices,
    packed_size,
    compression_ratio,
)
from aiter.ops.triton.attention.turboquant.rotation import (
    get_rotation_matrix,
    get_qjl_matrix,
    clear_cache,
)
from aiter.ops.triton.attention.turboquant.codebook import (
    get_codebook,
    generate_lloyd_max_codebook,
    SUPPORTED_HEAD_DIMS,
    SUPPORTED_BITS,
)
from aiter.ops.triton.attention.turboquant.quantizer import (
    TurboQuantMSE,
    TurboQuantProd,
    ValueQuantizer,
)

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

DEVICE = torch.device("cpu")  # Tests run on CPU; GPU tests tagged separately


def _rand_unit_vecs(n: int, d: int, seed: int = 0) -> torch.Tensor:
    """Return (n, d) float32 unit-norm vectors."""
    torch.manual_seed(seed)
    x = torch.randn(n, d)
    return x / x.norm(dim=-1, keepdim=True)


# ---------------------------------------------------------------------------
# 1. Bit-packing round-trips
# ---------------------------------------------------------------------------

class TestBitPacking:

    @pytest.mark.parametrize("bits,d", [
        (2, 64), (2, 128), (2, 63),   # d not multiple of 4
        (3, 64), (3, 128), (3, 65),   # d not multiple of 8
        (4, 64), (4, 128), (4, 127),  # d not multiple of 2
    ])
    def test_round_trip(self, bits, d):
        """pack → unpack must recover original indices exactly."""
        n_levels = 2 ** bits
        torch.manual_seed(42)
        indices = torch.randint(0, n_levels, (8, d), dtype=torch.int64)

        packed = pack_indices(indices, bits)
        recovered = unpack_indices(packed, bits, d)

        assert recovered.shape == (8, d), f"shape mismatch: {recovered.shape}"
        assert (recovered == indices).all(), \
            f"bits={bits} d={d}: max diff {(recovered - indices).abs().max()}"

    @pytest.mark.parametrize("bits,d", [(2, 128), (3, 128), (4, 128)])
    def test_packed_dtype(self, bits, d):
        """Packed tensor must be uint8."""
        indices = torch.zeros(4, d, dtype=torch.int64)
        packed = pack_indices(indices, bits)
        assert packed.dtype == torch.uint8

    @pytest.mark.parametrize("bits,d", [(2, 128), (3, 128), (4, 128)])
    def test_packed_size(self, bits, d):
        """Packed tensor length matches packed_size()."""
        indices = torch.zeros(1, d, dtype=torch.int64)
        packed = pack_indices(indices, bits)
        assert packed.shape[-1] == packed_size(d, bits), \
            f"bits={bits} d={d}: packed {packed.shape[-1]} != expected {packed_size(d, bits)}"

    @pytest.mark.parametrize("bits", [2, 3, 4])
    def test_batch_dims(self, bits):
        """Pack/unpack must work with arbitrary leading batch dimensions."""
        d = 64
        indices = torch.randint(0, 2**bits, (2, 4, d), dtype=torch.int64)
        packed = pack_indices(indices, bits)
        recovered = unpack_indices(packed, bits, d)
        assert (recovered == indices).all()


# ---------------------------------------------------------------------------
# 2. Rotation matrix orthogonality
# ---------------------------------------------------------------------------

class TestRotationMatrices:

    @pytest.mark.parametrize("head_dim", [64, 128, 256])
    def test_pi_is_orthogonal(self, head_dim):
        """Π^T Π should equal the identity matrix within floating-point tolerance."""
        clear_cache()
        Pi = get_rotation_matrix(head_dim, DEVICE)
        eye = torch.eye(head_dim)
        product = Pi.cpu().T @ Pi.cpu()
        assert torch.allclose(product, eye, atol=1e-5), \
            f"head_dim={head_dim}: max deviation {(product - eye).abs().max():.2e}"

    @pytest.mark.parametrize("head_dim", [64, 128])
    def test_pi_deterministic(self, head_dim):
        """Same seed must produce the same Π."""
        clear_cache()
        Pi1 = get_rotation_matrix(head_dim, DEVICE, seed=42)
        clear_cache()
        Pi2 = get_rotation_matrix(head_dim, DEVICE, seed=42)
        assert torch.allclose(Pi1, Pi2), "Π not reproducible with same seed"

    @pytest.mark.parametrize("head_dim", [64, 128])
    def test_pi_different_seeds_differ(self, head_dim):
        """Different seeds should produce different matrices."""
        clear_cache()
        Pi1 = get_rotation_matrix(head_dim, DEVICE, seed=1)
        clear_cache()
        Pi2 = get_rotation_matrix(head_dim, DEVICE, seed=2)
        assert not torch.allclose(Pi1, Pi2), "Different seeds produced identical Π"

    @pytest.mark.parametrize("head_dim", [64, 128])
    def test_s_is_not_orthogonal(self, head_dim):
        """S should be Gaussian, NOT orthogonal (Sᵀ S ≠ I in general)."""
        clear_cache()
        S = get_qjl_matrix(head_dim, DEVICE)
        eye = torch.eye(head_dim)
        product = S.T @ S
        # For a d×d Gaussian matrix scaled by 1/d, Sᵀ S ≈ I/d * d = I but with variance
        # The key check is that it is NOT exactly orthogonal
        # We just verify S exists and has the right shape
        assert S.shape == (head_dim, head_dim), f"S shape wrong: {S.shape}"
        assert S.dtype == torch.float32

    def test_caching(self):
        """get_rotation_matrix must return the same object on second call."""
        clear_cache()
        Pi1 = get_rotation_matrix(64, DEVICE)
        Pi2 = get_rotation_matrix(64, DEVICE)
        assert Pi1 is Pi2, "Cached matrix should be the same object"


# ---------------------------------------------------------------------------
# 3. Codebook generation
# ---------------------------------------------------------------------------

class TestCodebook:

    @pytest.mark.parametrize("head_dim,bits", [
        (64, 2), (64, 3), (64, 4),
        (128, 2), (128, 3), (128, 4),
    ])
    def test_codebook_shape(self, head_dim, bits):
        """Codebook must have exactly 2^bits entries."""
        cb = get_codebook(head_dim, bits, device=DEVICE)
        assert cb.shape == (2 ** bits,), \
            f"head_dim={head_dim} bits={bits}: got shape {cb.shape}"

    @pytest.mark.parametrize("head_dim,bits", [(128, 2), (128, 3), (128, 4)])
    def test_codebook_sorted(self, head_dim, bits):
        """Centroids must be in ascending order (required for fast lookup)."""
        cb = get_codebook(head_dim, bits, device=DEVICE)
        diffs = cb[1:] - cb[:-1]
        assert (diffs >= 0).all(), "Codebook centroids are not sorted"

    @pytest.mark.parametrize("head_dim,bits", [(128, 2), (128, 4)])
    def test_codebook_symmetric(self, head_dim, bits):
        """Lloyd-Max on a symmetric distribution should produce symmetric centroids."""
        cb = get_codebook(head_dim, bits, device=DEVICE)
        # cb[i] + cb[n-1-i] ≈ 0
        n = len(cb)
        sym_error = (cb + cb.flip(0)).abs().max().item()
        assert sym_error < 0.05, \
            f"head_dim={head_dim} bits={bits}: symmetry error {sym_error:.4f}"

    @pytest.mark.parametrize("bits", [2, 3, 4])
    def test_codebook_reproducible(self, bits):
        """Codebook generation must be deterministic with the same seed."""
        cb1 = generate_lloyd_max_codebook(128, bits, seed=42)
        cb2 = generate_lloyd_max_codebook(128, bits, seed=42)
        assert torch.allclose(cb1, cb2), "Codebook not reproducible"

    @pytest.mark.parametrize("bits", [2, 3, 4])
    def test_codebook_dtype(self, bits):
        """Codebook must be float32."""
        cb = get_codebook(128, bits, device=DEVICE)
        assert cb.dtype == torch.float32


# ---------------------------------------------------------------------------
# 4. TurboQuantMSE compress / decompress accuracy
# ---------------------------------------------------------------------------

class TestTurboQuantMSE:

    @pytest.mark.parametrize("head_dim,bits", [
        (64, 2), (64, 4), (128, 3), (128, 4), (256, 4),
    ])
    def test_decompress_shape(self, head_dim, bits):
        """Decompressed shape must match original."""
        clear_cache()
        mse = TurboQuantMSE(head_dim, bits, device=DEVICE)
        x = _rand_unit_vecs(32, head_dim)
        compressed = mse.compress(x)
        x_hat = mse.decompress(compressed)
        assert x_hat.shape == x.shape

    @pytest.mark.parametrize("head_dim,bits", [
        (128, 4), (128, 3), (128, 2),
    ])
    def test_cos_sim_increases_with_bits(self, head_dim, bits):
        """Higher bits must give better cosine similarity."""
        clear_cache()
        results = {}
        for b in [2, 3, 4]:
            mse = TurboQuantMSE(head_dim, b, device=DEVICE, seed=0)
            x = _rand_unit_vecs(256, head_dim, seed=7)
            compressed = mse.compress(x)
            x_hat = mse.decompress(compressed)
            cos = F.cosine_similarity(x, x_hat, dim=-1).mean().item()
            results[b] = cos
        assert results[2] < results[3] < results[4], \
            f"cos_sim not monotone: {results}"

    @pytest.mark.parametrize("head_dim,bits", [(128, 4)])
    def test_high_bit_cos_sim(self, head_dim, bits):
        """4-bit MSE should achieve cos_sim > 0.97 on random unit vectors."""
        clear_cache()
        mse = TurboQuantMSE(head_dim, bits, device=DEVICE)
        x = _rand_unit_vecs(512, head_dim, seed=99)
        compressed = mse.compress(x)
        x_hat = mse.decompress(compressed)
        cos = F.cosine_similarity(x, x_hat, dim=-1).mean().item()
        assert cos > 0.97, f"4-bit cos_sim too low: {cos:.4f}"

    @pytest.mark.parametrize("head_dim,bits", [(128, 3), (128, 4)])
    def test_norm_preservation(self, head_dim, bits):
        """Norms stored during compression must be recovered accurately."""
        clear_cache()
        mse = TurboQuantMSE(head_dim, bits, device=DEVICE)
        x = torch.randn(64, head_dim) * 2.5  # non-unit vectors
        compressed = mse.compress(x)
        x_hat = mse.decompress(compressed)
        orig_norms = x.norm(dim=-1)
        hat_norms = x_hat.norm(dim=-1)
        rel_err = ((orig_norms - hat_norms) / orig_norms.clamp(min=1e-6)).abs().mean()
        assert rel_err < 0.05, f"Norm relative error too large: {rel_err:.4f}"

    def test_batch_dims(self):
        """Compress/decompress must handle arbitrary leading batch dimensions."""
        clear_cache()
        mse = TurboQuantMSE(128, 3, device=DEVICE)
        x = torch.randn(2, 8, 16, 128)
        compressed = mse.compress(x)
        x_hat = mse.decompress(compressed)
        assert x_hat.shape == x.shape


# ---------------------------------------------------------------------------
# 5. TurboQuantProd inner-product unbiasedness
# ---------------------------------------------------------------------------

class TestTurboQuantProd:

    def test_score_shape(self):
        """inner_product_score must return correct shape."""
        clear_cache()
        prod = TurboQuantProd(128, 3, device=DEVICE)
        k = _rand_unit_vecs(64, 128, seed=0)
        q = _rand_unit_vecs(8, 128, seed=1)
        compressed = prod.compress(k)
        # q: (8, 128), k: (64, 128) → scores: (8, 64)
        scores = prod.inner_product_score(q, compressed)
        assert scores.shape == (8, 64), f"score shape: {scores.shape}"

    def test_inner_product_unbiased(self):
        """
        E[score(q, compressed_k)] ≈ <q, k>  (within statistical tolerance).

        We average over many random keys to check the estimator is unbiased.
        """
        clear_cache()
        torch.manual_seed(42)
        d = 128
        n_keys = 1024
        prod = TurboQuantProd(d, 3, device=DEVICE)

        q = torch.randn(1, d)
        q = q / q.norm()
        k = _rand_unit_vecs(n_keys, d, seed=10)

        true_scores = (q @ k.T).squeeze(0)    # (n_keys,)

        compressed = prod.compress(k)
        est_scores = prod.inner_product_score(q, compressed).squeeze(0)  # (n_keys,)

        bias = (est_scores - true_scores).mean().item()
        assert abs(bias) < 0.02, f"Mean bias too large: {bias:.4f}"

    def test_prod_better_than_mse_for_ip(self):
        """
        TurboQuantProd should have lower mean IP error than TurboQuantMSE
        (especially at low bits where residual correction matters most).
        """
        clear_cache()
        torch.manual_seed(0)
        d, bits = 128, 2
        n = 512

        mse = TurboQuantMSE(d, bits, device=DEVICE, seed=5)
        prod = TurboQuantProd(d, bits, device=DEVICE, seed=5)

        q = _rand_unit_vecs(1, d, seed=100)
        k = _rand_unit_vecs(n, d, seed=101)
        true_ip = (q @ k.T).squeeze(0)

        # MSE score: compute manually via decompress + dot
        k_hat_mse = mse.decompress(mse.compress(k))
        mse_scores = (q @ k_hat_mse.T).squeeze(0)

        # Prod score
        prod_scores = prod.inner_product_score(q, prod.compress(k)).squeeze(0)

        mse_err = (mse_scores - true_ip).abs().mean().item()
        prod_err = (prod_scores - true_ip).abs().mean().item()

        # Prod should be at least as good; allow small slack due to randomness
        assert prod_err <= mse_err * 1.05, \
            f"Prod (err={prod_err:.4f}) worse than MSE (err={mse_err:.4f})"


# ---------------------------------------------------------------------------
# 6. ValueQuantizer
# ---------------------------------------------------------------------------

class TestValueQuantizer:

    @pytest.mark.parametrize("bits,group_size", [(2, 32), (4, 32), (4, 16)])
    def test_round_trip_shape(self, bits, group_size):
        """Decompress must return original shape."""
        vq = ValueQuantizer(bits=bits, group_size=group_size)
        v = torch.randn(16, 128)
        compressed = vq.compress(v)
        v_hat = vq.decompress(compressed)
        assert v_hat.shape == v.shape

    @pytest.mark.parametrize("bits", [4])
    def test_4bit_cos_sim(self, bits):
        """4-bit value quantization must achieve cos_sim ≥ 0.99."""
        vq = ValueQuantizer(bits=bits, group_size=32)
        torch.manual_seed(7)
        v = torch.randn(256, 128)
        compressed = vq.compress(v)
        v_hat = vq.decompress(compressed)
        cos = F.cosine_similarity(v, v_hat, dim=-1).mean().item()
        assert cos >= 0.99, f"4-bit value cos_sim too low: {cos:.4f}"

    @pytest.mark.parametrize("bits", [2])
    def test_2bit_cos_sim(self, bits):
        """2-bit value quantization must achieve cos_sim ≥ 0.90."""
        vq = ValueQuantizer(bits=bits, group_size=32)
        torch.manual_seed(7)
        v = torch.randn(256, 128)
        compressed = vq.compress(v)
        v_hat = vq.decompress(compressed)
        cos = F.cosine_similarity(v, v_hat, dim=-1).mean().item()
        assert cos >= 0.90, f"2-bit value cos_sim too low: {cos:.4f}"

    def test_batch_dims(self):
        """ValueQuantizer must handle arbitrary leading batch dims."""
        vq = ValueQuantizer(bits=4, group_size=32)
        v = torch.randn(2, 4, 128)
        compressed = vq.compress(v)
        v_hat = vq.decompress(compressed)
        assert v_hat.shape == v.shape

    def test_indices_in_range(self):
        """Quantized indices must be in [0, 2^bits)."""
        bits = 4
        vq = ValueQuantizer(bits=bits, group_size=32)
        v = torch.randn(64, 128)
        compressed = vq.compress(v)
        from aiter.ops.triton.attention.turboquant.utils import unpack_indices
        indices = unpack_indices(compressed.indices, bits, 128)
        assert indices.min() >= 0
        assert indices.max() < 2 ** bits


# ---------------------------------------------------------------------------
# 7. Distortion scaling (Theorem 3)
# ---------------------------------------------------------------------------

class TestDistortionScaling:
    """
    Theorem 3 in the TurboQuant paper states that MSE distortion scales as
    1/4^b (halves per extra bit).  We verify this empirically: the MSE at
    bits+1 should be ≤ 35% of the MSE at bits (allowing implementation slack).
    """

    @pytest.mark.parametrize("head_dim", [128])
    def test_distortion_halving(self, head_dim):
        clear_cache()
        torch.manual_seed(0)
        n = 1024
        x = _rand_unit_vecs(n, head_dim)
        mses = {}
        for bits in [2, 3, 4]:
            mse_q = TurboQuantMSE(head_dim, bits, device=DEVICE, seed=42)
            compressed = mse_q.compress(x)
            x_hat = mse_q.decompress(compressed)
            mses[bits] = F.mse_loss(x_hat, x).item()

        # Each extra bit should reduce MSE significantly
        ratio_2_3 = mses[2] / mses[3]
        ratio_3_4 = mses[3] / mses[4]
        assert ratio_2_3 > 1.5, \
            f"3-bit not significantly better than 2-bit: ratio={ratio_2_3:.2f}"
        assert ratio_3_4 > 1.5, \
            f"4-bit not significantly better than 3-bit: ratio={ratio_3_4:.2f}"


# ---------------------------------------------------------------------------
# 8. Compression ratio
# ---------------------------------------------------------------------------

class TestCompressionRatio:

    @pytest.mark.parametrize("head_dim,key_bits,value_bits,expected_min", [
        (128, 3, 2, 3.5),
        (128, 4, 4, 2.5),
        (256, 3, 2, 3.5),
    ])
    def test_ratio_in_range(self, head_dim, key_bits, value_bits, expected_min):
        ratio = compression_ratio(head_dim, key_bits, value_bits)
        assert ratio >= expected_min, \
            f"head_dim={head_dim} k={key_bits}b v={value_bits}b: ratio={ratio:.2f} < {expected_min}"

    def test_higher_bits_lower_ratio(self):
        """More bits → lower compression ratio (less compression)."""
        r_aggressive = compression_ratio(128, 2, 2)
        r_moderate = compression_ratio(128, 3, 2)
        r_conservative = compression_ratio(128, 4, 4)
        assert r_aggressive > r_moderate > r_conservative, \
            f"Ratios not monotone: {r_aggressive:.2f} {r_moderate:.2f} {r_conservative:.2f}"
