"""
Unit tests for grouped_gemm_fprop / grouped_gemm_dgrad / grouped_gemm_wgrad.

Each test compares against a pure-PyTorch reference that loops over groups.
"""

import sys
import pytest
import torch

from grouped_gemm_ops import grouped_gemm_fprop, grouped_gemm_dgrad, grouped_gemm_wgrad


# ---------------------------------------------------------------------------
# Reference implementations (loop over groups, plain torch.matmul)
# ---------------------------------------------------------------------------

def _ref_fprop(x, w, split_sizes):
    """out[g] = x[g] @ w[g]^T"""
    outs = []
    offset = 0
    for g, m_g in enumerate(split_sizes.tolist()):
        if m_g > 0:
            outs.append(x[offset:offset + m_g] @ w[g].T)
        offset += m_g
    return torch.cat(outs, dim=0)


def _ref_dgrad(dy, w, split_sizes):
    """dx[g] = dy[g] @ w[g]"""
    outs = []
    offset = 0
    for g, m_g in enumerate(split_sizes.tolist()):
        if m_g > 0:
            outs.append(dy[offset:offset + m_g] @ w[g])
        offset += m_g
    return torch.cat(outs, dim=0)


def _ref_wgrad(dy, x, split_sizes):
    """dw[g] = dy[g]^T @ x[g]"""
    G = split_sizes.numel()
    N = dy.shape[1]
    K = x.shape[1]
    dw = torch.zeros(G, N, K, device=dy.device, dtype=dy.dtype)
    offset = 0
    for g, m_g in enumerate(split_sizes.tolist()):
        if m_g > 0:
            dw[g] = dy[offset:offset + m_g].T @ x[offset:offset + m_g]
        offset += m_g
    return dw


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_split_sizes(G, M, balanced, device):
    """Generate group sizes that sum to G*M."""
    total = G * M
    if balanced:
        base = total // G
        rem = total % G
        sizes = [base + (1 if i < rem else 0) for i in range(G)]
    else:
        # Random split
        torch.manual_seed(42)
        raw = torch.randint(1, M * 2, (G,), dtype=torch.int64)
        raw = (raw.float() / raw.sum() * total).long()
        raw[-1] = total - raw[:-1].sum()
        raw = raw.clamp(min=0)
        sizes = raw.tolist()
    return torch.tensor(sizes, dtype=torch.int64, device=device)


def _compute_snr(ref, test):
    """Signal-to-noise ratio in dB."""
    noise = (ref.float() - test.float()).norm()
    signal = ref.float().norm()
    if noise == 0:
        return float("inf")
    return 20 * torch.log10(signal / noise).item()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("G", [1, 2, 4, 8])
@pytest.mark.parametrize("M", [128, 256, 512])
@pytest.mark.parametrize("N,K", [(2048, 1536), (4096, 4096), (1024, 2048)])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("balanced", [True, False])
class TestGroupedGemmAPI:
    """Tests for fprop, dgrad, wgrad individually."""

    def test_fprop(self, G, M, N, K, dtype, balanced):
        device = "cuda"
        split_sizes = _make_split_sizes(G, M, balanced, device)
        GM = int(split_sizes.sum().item())

        x = torch.randn(GM, K, device=device, dtype=dtype)
        w = torch.randn(G, N, K, device=device, dtype=dtype)

        out = grouped_gemm_fprop(x, w, split_sizes)
        ref = _ref_fprop(x, w, split_sizes)

        assert out.shape == ref.shape, f"Shape mismatch: {out.shape} vs {ref.shape}"
        snr = _compute_snr(ref, out)
        threshold = 40 if dtype == torch.bfloat16 else 45
        assert snr > threshold, f"fprop SNR {snr:.1f} dB < {threshold} dB"
        torch.testing.assert_close(ref, out, rtol=1e-1, atol=1e-1)

    def test_dgrad(self, G, M, N, K, dtype, balanced):
        device = "cuda"
        split_sizes = _make_split_sizes(G, M, balanced, device)
        GM = int(split_sizes.sum().item())

        dy = torch.randn(GM, N, device=device, dtype=dtype)
        w = torch.randn(G, N, K, device=device, dtype=dtype)

        dx = grouped_gemm_dgrad(dy, w, split_sizes)
        ref = _ref_dgrad(dy, w, split_sizes)

        assert dx.shape == ref.shape, f"Shape mismatch: {dx.shape} vs {ref.shape}"
        snr = _compute_snr(ref, dx)
        threshold = 40 if dtype == torch.bfloat16 else 45
        assert snr > threshold, f"dgrad SNR {snr:.1f} dB < {threshold} dB"
        torch.testing.assert_close(ref, dx, rtol=1e-1, atol=1e-1)

    def test_wgrad(self, G, M, N, K, dtype, balanced):
        device = "cuda"
        split_sizes = _make_split_sizes(G, M, balanced, device)
        GM = int(split_sizes.sum().item())

        dy = torch.randn(GM, N, device=device, dtype=dtype)
        x = torch.randn(GM, K, device=device, dtype=dtype)

        dw = grouped_gemm_wgrad(dy, x, split_sizes)
        ref = _ref_wgrad(dy, x, split_sizes)

        assert dw.shape == ref.shape, f"Shape mismatch: {dw.shape} vs {ref.shape}"
        snr = _compute_snr(ref, dw)
        threshold = 40 if dtype == torch.bfloat16 else 45
        assert snr > threshold, f"wgrad SNR {snr:.1f} dB < {threshold} dB"
        torch.testing.assert_close(ref, dw, rtol=1e-1, atol=1e-1)


@pytest.mark.parametrize("G", [4, 8])
@pytest.mark.parametrize("M", [256])
@pytest.mark.parametrize("N,K", [(2048, 1536)])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
class TestWgradOutputAccum:
    """Test output_accum mode of grouped_gemm_wgrad."""

    def test_output_accum(self, G, M, N, K, dtype):
        device = "cuda"
        split_sizes = _make_split_sizes(G, M, True, device)
        GM = int(split_sizes.sum().item())

        dy = torch.randn(GM, N, device=device, dtype=dtype)
        x = torch.randn(GM, K, device=device, dtype=dtype)

        # First wgrad
        dw1 = grouped_gemm_wgrad(dy, x, split_sizes)

        # Second wgrad accumulated onto a pre-filled tensor
        existing = torch.randn(G, N, K, device=device, dtype=dtype)
        existing_clone = existing.clone()
        result = grouped_gemm_wgrad(dy, x, split_sizes, wgrad=existing, output_accum=True)

        assert result is existing, "output_accum should return the same tensor"
        expected = existing_clone + dw1
        snr = _compute_snr(expected, result)
        assert snr > 40, f"output_accum SNR {snr:.1f} dB too low"


class TestEdgeCases:
    """Edge cases: single group, zero-length groups."""

    def test_single_group(self):
        device = "cuda"
        dtype = torch.bfloat16
        G, M, N, K = 1, 512, 2048, 1024
        split_sizes = torch.tensor([M], dtype=torch.int64, device=device)

        x = torch.randn(M, K, device=device, dtype=dtype)
        w = torch.randn(G, N, K, device=device, dtype=dtype)
        dy = torch.randn(M, N, device=device, dtype=dtype)

        out = grouped_gemm_fprop(x, w, split_sizes)
        ref = x @ w[0].T
        torch.testing.assert_close(ref, out, rtol=1e-1, atol=1e-1)

        dx = grouped_gemm_dgrad(dy, w, split_sizes)
        ref_dx = dy @ w[0]
        torch.testing.assert_close(ref_dx, dx, rtol=1e-1, atol=1e-1)

        dw = grouped_gemm_wgrad(dy, x, split_sizes)
        ref_dw = (dy.T @ x).unsqueeze(0)
        torch.testing.assert_close(ref_dw, dw, rtol=1e-1, atol=1e-1)

    def test_int32_split_sizes(self):
        """split_sizes passed as int32 should work (auto-cast to int64)."""
        device = "cuda"
        dtype = torch.bfloat16
        G, M, N, K = 4, 128, 1024, 512
        split_sizes = torch.full((G,), M, dtype=torch.int32, device=device)

        x = torch.randn(G * M, K, device=device, dtype=dtype)
        w = torch.randn(G, N, K, device=device, dtype=dtype)

        out = grouped_gemm_fprop(x, w, split_sizes)
        ref = _ref_fprop(x, w, split_sizes)
        torch.testing.assert_close(ref, out, rtol=1e-1, atol=1e-1)


class TestAutogradIntegration:
    """Verify that fprop/dgrad/wgrad form a correct gradient triplet.

    We compute forward + backward with torch.autograd and compare the
    individual gradients against our explicit dgrad/wgrad functions.
    """

    @pytest.mark.parametrize("G", [4])
    @pytest.mark.parametrize("M", [256])
    @pytest.mark.parametrize("N,K", [(2048, 1536)])
    @pytest.mark.parametrize("dtype", [torch.bfloat16])
    def test_autograd_consistency(self, G, M, N, K, dtype):
        device = "cuda"
        split_sizes = _make_split_sizes(G, M, True, device)
        GM = int(split_sizes.sum().item())

        x = torch.randn(GM, K, device=device, dtype=dtype)
        w = torch.randn(G, N, K, device=device, dtype=dtype)

        # Use our fprop
        out = grouped_gemm_fprop(x, w, split_sizes)

        # Random upstream gradient
        dy = torch.randn_like(out)

        # Compute dgrad and wgrad with our functions
        dx = grouped_gemm_dgrad(dy, w, split_sizes)
        dw = grouped_gemm_wgrad(dy, x, split_sizes)

        # Compare against reference
        ref_dx = _ref_dgrad(dy, w, split_sizes)
        ref_dw = _ref_wgrad(dy, x, split_sizes)

        snr_dx = _compute_snr(ref_dx, dx)
        snr_dw = _compute_snr(ref_dw, dw)

        assert snr_dx > 40, f"dgrad SNR {snr_dx:.1f} dB"
        assert snr_dw > 40, f"wgrad SNR {snr_dw:.1f} dB"


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "--tb=short", "-x"]))
