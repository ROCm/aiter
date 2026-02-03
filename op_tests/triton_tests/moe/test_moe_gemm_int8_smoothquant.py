# Unit tests for Int8 SmoothQuant MoE GEMM
# Based on test_moe_gemm_a8w8.py

from dataclasses import dataclass, fields
import pytest
import torch

# Routing utilities
from aiter.ops.triton.moe.moe_routing.routing import routing

# SmoothQuant MoE utilities
from aiter.ops.triton.moe.moe_op_gemm_int8_smoothquant import (
    smoothquant_quantize,
    smoothquant_quantize_torch,
    quantize_weights_int8,
    moe_gemm_int8_smoothquant,
    moe_gemm_int8_smoothquant_torch,
    smoothquant_moe_mlp,
)

# Target-specific utilities
from aiter.ops.triton.utils._triton.arch_info import get_arch


# ---------------
# Initialize data
# ---------------


def alloc_rand(shape, device, dtype):
    """Allocate random tensor with appropriate range for dtype."""
    if dtype == torch.int8:
        return torch.randint(-127, 128, shape, device=device, dtype=dtype)
    return torch.randn(shape, device=device, dtype=dtype)


def alloc_rand_like(x):
    return alloc_rand(x.shape, x.device, x.dtype)


def init_routing_data(
    m, n_expts_tot, n_expts_act, do_gather, do_scatter, device="cuda"
):
    """Initialize routing data for MoE."""
    logits = torch.randn((m, n_expts_tot), dtype=torch.float16, device=device)
    routing_data, gather_idx, scatter_idx = routing(logits, n_expts_act)
    routing_data.gate_scal = None
    gather_idx = gather_idx if do_gather else None
    scatter_idx = scatter_idx if do_scatter else None
    return m, routing_data, gather_idx, scatter_idx


def init_compute_data(
    m,
    n,
    k,
    gindx,
    sindx,
    n_expts_tot,
    n_expts_act,
    has_y_gammas,
    device="cuda",
):
    """Initialize computation data for MoE."""
    torch.manual_seed(0)
    in_m = m * (n_expts_act if gindx is None else 1)
    
    # Create bf16 activations
    x = torch.randn((in_m, k), device=device, dtype=torch.bfloat16)
    
    # Create bf16 weights (will be quantized later)
    w = torch.randn((n_expts_tot, k, n), device=device, dtype=torch.bfloat16)
    
    # Create bias
    bias = torch.randn((n_expts_tot, n), device=device, dtype=torch.float32)
    
    # Create smooth scales (per column)
    smooth_scale = torch.randn((k,), device=device, dtype=torch.float32).abs() + 0.1
    
    if has_y_gammas:
        gamma = 2 ** torch.randint(
            -5, 0, (m * n_expts_act,), device=device, dtype=torch.float32
        )
    else:
        gamma = None
    
    return x, w, bias, smooth_scale, gamma


def assert_close(ref, tri, maxtol=None, rmstol=None, description="--", verbose=True):
    """Compare reference and triton outputs with tolerance."""
    if tri.dtype.itemsize == 1:
        ref_as_type = ref.to(tri.dtype)
        if ref.dtype == tri.dtype:
            assert torch.all(ref_as_type == tri)
            return
        ref = ref_as_type

    if ref.numel() == 0:
        return

    if maxtol is None:
        maxtol = 5e-2  # Slightly higher tolerance for int8 quantization
    if rmstol is None:
        rmstol = 1e-2

    # Cast to float32
    ref = ref.to(torch.float32).detach()
    tri = tri.to(torch.float32).detach()
    assert ref.shape == tri.shape, f"Shape mismatch: {ref.shape=} vs {tri.shape=}"

    # Handle infinites
    inf_mask_ref = torch.isinf(ref)
    inf_mask_tri = torch.isinf(tri)
    assert torch.equal(inf_mask_ref, inf_mask_tri), "Infinite element mismatch"
    refn = torch.where(inf_mask_ref, 0, ref)
    trin = torch.where(inf_mask_tri, 0, tri)

    # Normalize for RMS
    eps = 1.0e-30
    multiplier = 1.0 / (torch.max(torch.abs(refn)) + eps)
    refn *= multiplier
    trin *= multiplier

    ref_rms = torch.sqrt(torch.square(refn).mean()) + eps
    rel_err = torch.abs(refn - trin) / torch.maximum(ref_rms, torch.abs(refn))
    max_err = torch.max(rel_err).item()
    rms_err = torch.sqrt(torch.square(rel_err).mean()).item()

    if verbose:
        print(f"{description} max relative error = {max_err} (threshold = {maxtol})")
        print(f"{description} RMS relative error = {rms_err} (threshold = {rmstol})")

    if max_err > maxtol:
        bad_idxs = torch.nonzero(rel_err > maxtol)
        num_nonzero = bad_idxs.size(0)
        bad_idxs = bad_idxs[:1000]
        print(f"{num_nonzero} / {rel_err.numel()} mismatched elements at {bad_idxs.tolist()[:10]}")
        bad_idxs_tuple = bad_idxs.unbind(-1)
        print("ref values:", ref[tuple(bad_idxs_tuple)][:10].cpu())
        print("tri values:", tri[tuple(bad_idxs_tuple)][:10].cpu())

    assert max_err <= maxtol, f"Max error {max_err} exceeds tolerance {maxtol}"
    assert rms_err <= rmstol, f"RMS error {rms_err} exceeds tolerance {rmstol}"


# ---------------
# Unit tests for SmoothQuant Quantization
# ---------------


@pytest.mark.parametrize("m", [16, 64, 256, 1024])
@pytest.mark.parametrize("k", [512, 1024, 4096])
def test_smoothquant_quantize(m, k, device="cuda"):
    """Test smoothquant quantization against reference."""
    torch.manual_seed(42)
    
    x = torch.randn((m, k), device=device, dtype=torch.bfloat16)
    smooth_scale = torch.randn((k,), device=device, dtype=torch.float32).abs() + 0.1
    
    # Triton implementation
    x_int8_tri, x_scale_tri = smoothquant_quantize(x, smooth_scale)
    
    # Reference implementation
    x_int8_ref, x_scale_ref = smoothquant_quantize_torch(x, smooth_scale)
    
    # Compare scales
    assert_close(x_scale_ref, x_scale_tri, maxtol=1e-5, description="x_scale")
    
    # Compare quantized values (allow small differences due to rounding)
    diff = (x_int8_ref.to(torch.int32) - x_int8_tri.to(torch.int32)).abs()
    assert diff.max() <= 1, f"Int8 values differ by more than 1: max diff = {diff.max()}"


@pytest.mark.parametrize("shape", [(8, 512, 256), (256, 1024, 512)])
def test_quantize_weights(shape, device="cuda"):
    """Test weight quantization."""
    torch.manual_seed(42)
    E, K, N = shape
    
    w = torch.randn((E, K, N), device=device, dtype=torch.bfloat16)
    w_int8, w_scale = quantize_weights_int8(w)
    
    # Verify shapes
    assert w_int8.shape == (E, K, N)
    assert w_scale.shape == (E, N)
    assert w_int8.dtype == torch.int8
    assert w_scale.dtype == torch.float32
    
    # Verify range
    assert w_int8.abs().max() <= 127
    
    # Verify reconstruction error is reasonable
    w_reconstructed = w_int8.to(torch.float32) * w_scale[:, None, :]
    rel_error = (w.to(torch.float32) - w_reconstructed).abs() / (w.to(torch.float32).abs() + 1e-6)
    assert rel_error.mean() < 0.1, f"Mean reconstruction error too high: {rel_error.mean()}"


# ---------------
# Unit tests for MoE GEMM
# ---------------


@dataclass
class Case:
    m: int
    n: int
    k: int
    n_expts_tot: int = 8
    n_expts_act: int = 2


@pytest.mark.parametrize(
    ", ".join(f.name for f in fields(Case)),
    [
        tuple(getattr(case, f.name) for f in fields(Case))
        for case in [
            # Small cases
            Case(16, 256, 256, 8, 2),
            Case(64, 512, 512, 8, 2),
            Case(128, 1024, 512, 8, 4),
            # Medium cases
            Case(256, 2048, 1024, 16, 4),
            Case(512, 4096, 2048, 32, 8),
            # Large MoE configs
            Case(1024, 7168, 4096, 64, 8),
            Case(2048, 4096, 7168, 128, 8),
            # Edge cases
            Case(17, 257, 129, 8, 2),  # Non-power-of-2
            Case(100, 400, 400, 8, 2),
        ]
    ],
)
@pytest.mark.parametrize(
    "do_gather, do_scatter",
    [
        (False, False),
        (True, False),
        (False, True),
        (True, True),
    ],
)
@pytest.mark.parametrize("has_y_gammas", [False, True])
@pytest.mark.parametrize("activation", ["none", "silu", "swiglu"])
def test_moe_gemm_int8(
    m,
    n,
    k,
    do_gather,
    do_scatter,
    has_y_gammas,
    activation,
    n_expts_tot,
    n_expts_act,
    device="cuda",
):
    """Test int8 MoE GEMM against reference implementation."""
    torch.manual_seed(0)
    
    apply_silu = activation == "silu"
    apply_swiglu = activation == "swiglu"

    # Initialize routing
    m, rdata, gindx, sindx = init_routing_data(
        m, n_expts_tot, n_expts_act, do_gather, do_scatter, device=device
    )
    
    # For SwiGLU, we need double-width N (it will be halved)
    actual_n = n * 2 if apply_swiglu else n
    
    # Initialize data
    x, w, bias, smooth_scale, gammas = init_compute_data(
        m, actual_n, k, gindx, sindx, n_expts_tot, n_expts_act, has_y_gammas, device=device
    )
    
    # Quantize activations with smoothquant
    x_int8, x_scale = smoothquant_quantize(x, smooth_scale)
    x_int8_ref, x_scale_ref = smoothquant_quantize_torch(x, smooth_scale)
    
    # Quantize weights
    w_int8, w_scale = quantize_weights_int8(w)
    
    # Reference implementation
    ref_y = moe_gemm_int8_smoothquant_torch(
        x_int8_ref,
        x_scale_ref,
        w_int8,
        w_scale,
        bias,
        rdata,
        gindx,
        sindx,
        gammas,
        apply_silu=apply_silu,
        apply_swiglu=apply_swiglu,
    )
    
    # Triton implementation
    tri_y = moe_gemm_int8_smoothquant(
        x_int8,
        x_scale,
        w_int8,
        w_scale,
        bias,
        rdata,
        gindx,
        sindx,
        gammas,
        out_dtype=torch.float32,  # Use fp32 for comparison
        apply_silu=apply_silu,
        apply_swiglu=apply_swiglu,
    )
    
    # Compare
    # Higher tolerance for int8 quantization + MoE routing
    assert_close(ref_y, tri_y, maxtol=0.1, rmstol=0.05)


@pytest.mark.parametrize(
    "m, n, k",
    [
        (64, 512, 256),
        (256, 2048, 1024),
        (512, 4096, 2048),
    ],
)
@pytest.mark.parametrize("n_expts_tot, n_expts_act", [(8, 2), (32, 4), (64, 8)])
def test_moe_gemm_int8_output_dtype(m, n, k, n_expts_tot, n_expts_act, device="cuda"):
    """Test that output dtype conversion works correctly."""
    torch.manual_seed(42)
    
    # Initialize
    m, rdata, gindx, sindx = init_routing_data(
        m, n_expts_tot, n_expts_act, True, True, device=device
    )
    x, w, bias, smooth_scale, gammas = init_compute_data(
        m, n, k, gindx, sindx, n_expts_tot, n_expts_act, False, device=device
    )
    
    # Quantize
    x_int8, x_scale = smoothquant_quantize(x, smooth_scale)
    w_int8, w_scale = quantize_weights_int8(w)
    
    # Test bf16 output
    y_bf16 = moe_gemm_int8_smoothquant(
        x_int8, x_scale, w_int8, w_scale,
        bias, rdata, gindx, sindx, None,
        out_dtype=torch.bfloat16,
    )
    assert y_bf16.dtype == torch.bfloat16
    
    # Test fp32 output
    y_fp32 = moe_gemm_int8_smoothquant(
        x_int8, x_scale, w_int8, w_scale,
        bias, rdata, gindx, sindx, None,
        out_dtype=torch.float32,
    )
    assert y_fp32.dtype == torch.float32
    
    # Compare (should be close after dtype conversion)
    assert_close(y_fp32, y_bf16.to(torch.float32), maxtol=0.01, rmstol=0.005)


# ---------------
# Integration test for full MLP
# ---------------


@pytest.mark.parametrize("m", [64, 256])
@pytest.mark.parametrize("hidden_dim, intermediate_dim", [(512, 1024), (1024, 2048)])
@pytest.mark.parametrize("n_expts_tot, n_expts_act", [(8, 2), (16, 4)])
def test_smoothquant_moe_mlp_shapes(
    m, hidden_dim, intermediate_dim, n_expts_tot, n_expts_act, device="cuda"
):
    """Test that smoothquant_moe_mlp produces correct output shapes.
    
    With SwiGLU activation:
    - FC1: hidden_dim -> intermediate_dim (double-width for gating)
    - SwiGLU: intermediate_dim -> intermediate_dim // 2 (halves dimension)
    - FC2: intermediate_dim // 2 -> hidden_dim
    """
    torch.manual_seed(42)
    
    # Initialize routing
    logits = torch.randn((m, n_expts_tot), dtype=torch.float16, device=device)
    rdata, gindx, sindx = routing(logits, n_expts_act)
    
    # Input
    x = torch.randn((m, hidden_dim), device=device, dtype=torch.bfloat16)
    
    # FC1: hidden_dim -> intermediate_dim (double-width for gating)
    fc1_smooth_scale = torch.randn((hidden_dim,), device=device, dtype=torch.float32).abs() + 0.1
    w1 = torch.randn((n_expts_tot, hidden_dim, intermediate_dim), device=device, dtype=torch.bfloat16)
    w1_int8, w1_scale = quantize_weights_int8(w1)
    
    # FC2: intermediate_dim // 2 -> hidden_dim (after SwiGLU halves)
    fc2_smooth_scale = torch.randn((intermediate_dim // 2,), device=device, dtype=torch.float32).abs() + 0.1
    w2 = torch.randn((n_expts_tot, intermediate_dim // 2, hidden_dim), device=device, dtype=torch.bfloat16)
    w2_int8, w2_scale = quantize_weights_int8(w2)
    
    # Forward pass with SwiGLU (halves intermediate dimension)
    output = smoothquant_moe_mlp(
        x,
        fc1_smooth_scale,
        w1_int8,
        w1_scale,
        fc2_smooth_scale,
        w2_int8,
        w2_scale,
        rdata,
        gindx,
        sindx,
        apply_swiglu=True,
    )
    
    # Check output shape
    assert output.shape == (m, hidden_dim), f"Expected ({m}, {hidden_dim}), got {output.shape}"
    assert output.dtype == torch.bfloat16


# ---------------
# Numerical stability tests
# ---------------


@pytest.mark.parametrize("scale_magnitude", [1e-3, 1.0, 1e3])
def test_smoothquant_numerical_stability(scale_magnitude, device="cuda"):
    """Test numerical stability with different scale magnitudes."""
    torch.manual_seed(42)
    m, k = 128, 512
    
    x = torch.randn((m, k), device=device, dtype=torch.bfloat16) * scale_magnitude
    smooth_scale = torch.randn((k,), device=device, dtype=torch.float32).abs() + 0.1
    
    x_int8, x_scale = smoothquant_quantize(x, smooth_scale)
    
    # Verify no NaN or Inf
    assert not torch.isnan(x_int8.to(torch.float32)).any(), "NaN in quantized values"
    assert not torch.isinf(x_scale).any(), "Inf in scales"
    assert not torch.isnan(x_scale).any(), "NaN in scales"
    
    # Verify int8 range
    assert x_int8.abs().max() <= 127


@pytest.mark.parametrize("sparsity", [0.0, 0.5, 0.9])
def test_moe_gemm_sparse_routing(sparsity, device="cuda"):
    """Test MoE GEMM with sparse expert routing (some experts get few/no tokens)."""
    torch.manual_seed(42)
    m, n, k = 256, 512, 256
    n_expts_tot, n_expts_act = 16, 2
    
    # Create biased logits to simulate sparse routing
    logits = torch.randn((m, n_expts_tot), dtype=torch.float16, device=device)
    if sparsity > 0:
        # Make some experts very unlikely
        n_inactive = int(n_expts_tot * sparsity)
        logits[:, :n_inactive] -= 100  # Very negative logits
    
    rdata, gindx, sindx = routing(logits, n_expts_act)
    
    # Data
    x = torch.randn((m * n_expts_act, k), device=device, dtype=torch.bfloat16)
    smooth_scale = torch.randn((k,), device=device, dtype=torch.float32).abs() + 0.1
    w = torch.randn((n_expts_tot, k, n), device=device, dtype=torch.bfloat16)
    
    # Quantize
    x_int8, x_scale = smoothquant_quantize(x, smooth_scale)
    w_int8, w_scale = quantize_weights_int8(w)
    
    # Should not crash with sparse routing
    y = moe_gemm_int8_smoothquant(
        x_int8, x_scale, w_int8, w_scale,
        routing_data=rdata,
        gather_indx=gindx,
        scatter_indx=sindx,
    )
    
    # Verify output is valid
    assert not torch.isnan(y).any(), "NaN in output"
    assert not torch.isinf(y).any(), "Inf in output"


if __name__ == "__main__":
    # Run a quick sanity check
    print("Running sanity check...")
    test_smoothquant_quantize(64, 512)
    test_moe_gemm_int8(64, 512, 256, True, True, False, "none", 8, 2)
    test_moe_gemm_int8(64, 512, 256, True, True, False, "silu", 8, 2)
    test_moe_gemm_int8(64, 512, 256, True, True, False, "swiglu", 8, 2)
    print("Sanity check passed!")
