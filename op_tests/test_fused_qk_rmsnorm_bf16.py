# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""
Correctness test for fused_qk_rmsnorm_bf16.

Tests the single-launch fused QK RMSNorm kernel against two sequential
torch.nn.functional.rms_norm calls. This kernel is used in DeepSeek MLA to
normalize q (q_a_layernorm) and k_nope (kv_a_layernorm) in one GPU launch.

Run:
    cd /sgl-workspace/aiter
    pytest -xvs op_tests/test_fused_qk_rmsnorm_bf16.py

    # Single case:
    pytest -xvs -k "T1_q1536_k512" op_tests/test_fused_qk_rmsnorm_bf16.py
"""

import pytest
import torch

try:
    import torch.cuda
    if not torch.cuda.is_available():
        pytest.skip("ROCm/CUDA not available", allow_module_level=True)
except Exception:
    pytest.skip("torch.cuda unavailable", allow_module_level=True)

try:
    from aiter.ops.fused_qk_rmsnorm_group_quant import fused_qk_rmsnorm_bf16
except ImportError:
    pytest.skip(
        "fused_qk_rmsnorm_bf16 not available (module not compiled or not installed)",
        allow_module_level=True,
    )

# DeepSeek-V2/V3/Kimi-K2.5 MLA dimensions
DEEPSEEK_CASES = [
    # (T, q_n, k_n, id)
    (1,   1536, 512,  "T1_q1536_k512"),   # decode batch-1 — primary target
    (4,   1536, 512,  "T4_q1536_k512"),   # decode batch-4
    (32,  1536, 512,  "T32_q1536_k512"),  # decode batch-32
    (128, 1536, 512,  "T128_q1536_k512"), # decode batch-128
    (1025, 1536, 512, "T1025_q1536_k512"), # grid_y=1 path (m > 1024)
    # Smaller dims: verify BlockSize=64 path (max_n <= 512)
    (1,  512, 512,  "T1_q512_k512"),
    # Larger dims: verify BlockSize=128/256 paths
    (1,  768, 512,  "T1_q768_k512"),
    (1, 2048, 512,  "T1_q2048_k512"),
]


def _rms_norm_ref(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """Reference BF16 RMSNorm: upcast to float32, normalize, downcast."""
    n = x.shape[-1]
    x_f = x.float()
    rms = (x_f.pow(2).mean(dim=-1, keepdim=True) + eps).rsqrt()
    return (x_f * rms * weight.float()).bfloat16()


@pytest.mark.parametrize("T,q_n,k_n,case_id", DEEPSEEK_CASES, ids=[c[3] for c in DEEPSEEK_CASES])
def test_fused_qk_rmsnorm_bf16_correctness(T, q_n, k_n, case_id):
    torch.manual_seed(42)
    device = "cuda"
    dtype  = torch.bfloat16
    eps    = 1e-6

    q      = torch.randn(T, q_n, dtype=dtype, device=device)
    k_nope = torch.randn(T, k_n, dtype=dtype, device=device)
    q_w    = torch.randn(q_n,    dtype=dtype, device=device)
    k_w    = torch.randn(k_n,    dtype=dtype, device=device)

    ref_q = _rms_norm_ref(q,      q_w, eps)
    ref_k = _rms_norm_ref(k_nope, k_w, eps)

    q_out, k_out = fused_qk_rmsnorm_bf16(q, q_w, eps, k_nope, k_w, eps)

    assert q_out.shape == q.shape,      f"q_out shape mismatch: {q_out.shape} vs {q.shape}"
    assert k_out.shape == k_nope.shape, f"k_out shape mismatch: {k_out.shape} vs {k_nope.shape}"
    assert q_out.dtype == dtype
    assert k_out.dtype == dtype

    torch.testing.assert_close(q_out, ref_q,  atol=1e-2, rtol=1e-2,
                               msg=f"[{case_id}] q_out mismatch")
    torch.testing.assert_close(k_out, ref_k,  atol=1e-2, rtol=1e-2,
                               msg=f"[{case_id}] k_out mismatch")


def test_fused_qk_rmsnorm_bf16_does_not_modify_inputs():
    """Verify the kernel allocates fresh output buffers (no in-place aliasing)."""
    torch.manual_seed(0)
    q      = torch.randn(4, 1536, dtype=torch.bfloat16, device="cuda")
    k_nope = torch.randn(4, 512,  dtype=torch.bfloat16, device="cuda")
    q_w    = torch.ones(1536, dtype=torch.bfloat16, device="cuda")
    k_w    = torch.ones(512,  dtype=torch.bfloat16, device="cuda")

    q_orig      = q.clone()
    k_nope_orig = k_nope.clone()

    q_out, k_out = fused_qk_rmsnorm_bf16(q, q_w, 1e-6, k_nope, k_w, 1e-6)

    assert not q_out.data_ptr() == q.data_ptr(),           "q_out should not alias q"
    assert not k_out.data_ptr() == k_nope.data_ptr(),      "k_out should not alias k_nope"
    torch.testing.assert_close(q,      q_orig,      msg="q was modified in-place")
    torch.testing.assert_close(k_nope, k_nope_orig, msg="k_nope was modified in-place")


def test_fused_qk_rmsnorm_bf16_non_contiguous_input():
    """Verify the kernel handles non-packed row stride (e.g. slice of latent_cache)."""
    torch.manual_seed(7)
    T, q_n, k_n, rope_dim = 4, 1536, 512, 64
    # Simulate latent_cache = [T, q_n], latent_kv = [T, k_n + rope_dim]
    # k_nope is a slice of latent_kv that is NOT packed (stride(0) = k_n + rope_dim)
    latent_kv = torch.randn(T, k_n + rope_dim, dtype=torch.bfloat16, device="cuda")
    k_nope    = latent_kv[:, :k_n]   # stride(0) = k_n + rope_dim, stride(1) = 1
    q         = torch.randn(T, q_n,  dtype=torch.bfloat16, device="cuda")
    q_w       = torch.randn(q_n,     dtype=torch.bfloat16, device="cuda")
    k_w       = torch.randn(k_n,     dtype=torch.bfloat16, device="cuda")
    eps       = 1e-6

    ref_q = _rms_norm_ref(q,      q_w, eps)
    ref_k = _rms_norm_ref(k_nope, k_w, eps)

    q_out, k_out = fused_qk_rmsnorm_bf16(q, q_w, eps, k_nope, k_w, eps)

    torch.testing.assert_close(q_out, ref_q, atol=1e-2, rtol=1e-2,
                               msg="non-contiguous k_nope: q_out mismatch")
    torch.testing.assert_close(k_out, ref_k, atol=1e-2, rtol=1e-2,
                               msg="non-contiguous k_nope: k_out mismatch")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Test fused_qk_rmsnorm_bf16 kernel for DeepSeek MLA"
    )
    parser.add_argument("--tokens", type=int, default=None, help="Override token count (T)")
    parser.add_argument("--q_n",    type=int, default=None, help="Override q hidden dim")
    parser.add_argument("--k_n",    type=int, default=None, help="Override k hidden dim")
    args = parser.parse_args()

    cases = DEEPSEEK_CASES
    if args.tokens is not None or args.q_n is not None or args.k_n is not None:
        T   = args.tokens if args.tokens is not None else 1
        q_n = args.q_n    if args.q_n    is not None else 1536
        k_n = args.k_n    if args.k_n    is not None else 512
        cases = [(T, q_n, k_n, f"T{T}_q{q_n}_k{k_n}")]

    for T, q_n, k_n, case_id in cases:
        print(f"  {case_id} ... ", end="", flush=True)
        test_fused_qk_rmsnorm_bf16_correctness(T, q_n, k_n, case_id)
        print("PASS")

    print("  does_not_modify_inputs ... ", end="", flush=True)
    test_fused_qk_rmsnorm_bf16_does_not_modify_inputs()
    print("PASS")

    print("  non_contiguous_input ... ", end="", flush=True)
    test_fused_qk_rmsnorm_bf16_non_contiguous_input()
    print("PASS")
