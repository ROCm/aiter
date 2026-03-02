"""
Determinism test for grouped_gemm_fprop / grouped_gemm_dgrad / grouped_gemm_wgrad.

For each API, we run the kernel twice with identical inputs and check that
the outputs are bit-for-bit identical (atol=0, rtol=0).
"""

import sys
import torch
import pytest

from grouped_gemm_ops import grouped_gemm_fprop, grouped_gemm_dgrad, grouped_gemm_wgrad


def _make_split_sizes(G, M, device):
    total = G * M
    base = total // G
    rem = total % G
    sizes = [base + (1 if i < rem else 0) for i in range(G)]
    return torch.tensor(sizes, dtype=torch.int64, device=device)


@pytest.mark.parametrize("G", [1, 4, 8])
@pytest.mark.parametrize("M", [128, 512])
@pytest.mark.parametrize("N,K", [(2048, 1536), (4096, 4096)])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
class TestDeterminism:

    def test_fprop_determinism(self, G, M, N, K, dtype):
        device = "cuda"
        split_sizes = _make_split_sizes(G, M, device)
        GM = int(split_sizes.sum().item())

        x = torch.randn(GM, K, device=device, dtype=dtype)
        w = torch.randn(G, N, K, device=device, dtype=dtype)

        out1 = grouped_gemm_fprop(x, w, split_sizes)
        out2 = grouped_gemm_fprop(x, w, split_sizes)

        diff = (out1 - out2).abs()
        max_diff = diff.max().item()
        num_diff = (diff != 0).sum().item()
        total = diff.numel()

        print(f"\n[fprop] G={G} M={M} N={N} K={K} {dtype}")
        print(f"  max_diff={max_diff}, mismatched={num_diff}/{total}")

        assert torch.equal(out1, out2), (
            f"fprop NOT deterministic: max_diff={max_diff}, "
            f"mismatched={num_diff}/{total} elements"
        )

    def test_dgrad_determinism(self, G, M, N, K, dtype):
        device = "cuda"
        split_sizes = _make_split_sizes(G, M, device)
        GM = int(split_sizes.sum().item())

        dy = torch.randn(GM, N, device=device, dtype=dtype)
        w = torch.randn(G, N, K, device=device, dtype=dtype)

        dx1 = grouped_gemm_dgrad(dy, w, split_sizes)
        dx2 = grouped_gemm_dgrad(dy, w, split_sizes)

        diff = (dx1 - dx2).abs()
        max_diff = diff.max().item()
        num_diff = (diff != 0).sum().item()
        total = diff.numel()

        print(f"\n[dgrad] G={G} M={M} N={N} K={K} {dtype}")
        print(f"  max_diff={max_diff}, mismatched={num_diff}/{total}")

        assert torch.equal(dx1, dx2), (
            f"dgrad NOT deterministic: max_diff={max_diff}, "
            f"mismatched={num_diff}/{total} elements"
        )

    def test_wgrad_determinism(self, G, M, N, K, dtype):
        device = "cuda"
        split_sizes = _make_split_sizes(G, M, device)
        GM = int(split_sizes.sum().item())

        dy = torch.randn(GM, N, device=device, dtype=dtype)
        x = torch.randn(GM, K, device=device, dtype=dtype)

        dw1 = grouped_gemm_wgrad(dy, x, split_sizes)
        dw2 = grouped_gemm_wgrad(dy, x, split_sizes)

        diff = (dw1 - dw2).abs()
        max_diff = diff.max().item()
        num_diff = (diff != 0).sum().item()
        total = diff.numel()

        print(f"\n[wgrad] G={G} M={M} N={N} K={K} {dtype}")
        print(f"  max_diff={max_diff}, mismatched={num_diff}/{total}")

        assert torch.equal(dw1, dw2), (
            f"wgrad NOT deterministic: max_diff={max_diff}, "
            f"mismatched={num_diff}/{total} elements"
        )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "--tb=short", "-s"]))
