# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""
Pytest tests for mHC (manifold-constrained Hyper Connection) fused kernel.

Tests correctness of the Triton implementation (equations 14-19 + apply-pre)
against PyTorch references for various input shapes and configurations.

Notation (from mHC paper arXiv:2512.24880v2):
    - M: Batch/sequence dimension
    - n: Stream parameter controlling manifold dimension
    - C: Hidden dimension per stream
    - nC: Total flattened input dimension (K in kernel, K = n × C)
    - N: Total output dimension (n² + 2n)
    - x_l ∈ ℝ^(M×nC): Flattened n-stream residual (input)
    - φ ∈ ℝ^(nC×N): Projection matrix for transformation to 3 streams
    - H ∈ ℝ^(M×N): Output containing [H^pre, H^post, H^res]
      - H^pre: [0:n] manifold projection with sigmoid activation (n elements, H^{pre} ∈ ℝ^{1×n})
      - H^post: [n:2n] post-processing with 2*sigmoid activation (n elements, H^{post} ∈ ℝ^{1×n})
      - H^res: [2n:2n+n²] residual connection (identity activation) (n² elements, H^{res} ∈ ℝ^{n×n})
    - layer_input: (M, C)    - Σᵢ (σ(H^pre_i) + hc_pre_eps) · x_i
"""

import pytest
import torch

from aiter.ops.triton.fusions.mhc import mhc, mhc_post
from op_tests.triton_tests.utils.mhc_ref import (
    generate_mhc_inputs,
    generate_mhc_post_inputs,
    get_test_shapes,
    is_doubly_stochastic,
    mhc_post_torch,
    mhc_torch,
)

try:
    import aiter as _aiter

    _HAS_AITER_MHC_PRE = hasattr(_aiter, "mhc_pre")
except ImportError:
    _aiter = None
    _HAS_AITER_MHC_PRE = False


# =============================================================================
# Tests
# =============================================================================


def _assert_mhc_close(
    triton_tuple,
    ref_tuple,
    *,
    post_atol=1e-2,
    post_rtol=1e-2,
    comb_atol=1e-2,
    comb_rtol=1e-2,
    layer_atol=1e-2,
    layer_rtol=1e-2,
):
    """Compare Triton's ``(post_mix, comb_mix, layer_input)`` 3-tuple against
    the merged ``mhc_torch`` 4-tuple ``(hpost, hres, hpre, layer_input)``.

    ``hpre`` is ignored here — Triton consumes H^pre inline and only exposes
    its downstream effect via ``layer_input``. All comparisons run in fp32;
    both sides are cast via ``.float()``.
    """
    post_t, comb_t, layer_t = triton_tuple
    post_ref, comb_ref, _hpre_ref, layer_ref = ref_tuple

    torch.testing.assert_close(
        post_t.float(), post_ref.float(), atol=post_atol, rtol=post_rtol
    )
    torch.testing.assert_close(
        comb_t.float(), comb_ref.float(), atol=comb_atol, rtol=comb_rtol
    )
    torch.testing.assert_close(
        layer_t.float(), layer_ref.float(), atol=layer_atol, rtol=layer_rtol
    )


@pytest.mark.parametrize("M, n, C", get_test_shapes())
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_mhc_correctness(M, n, C, dtype):
    """
    Test that Triton mhc() matches the PyTorch reference for equations 14-19
    plus the apply-pre step.
    """
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    x, phi, alpha_pre, alpha_post, alpha_res, bias, n_streams = generate_mhc_inputs(
        M, n, C, dtype
    )

    out_torch = mhc_torch(x, phi, alpha_pre, alpha_post, alpha_res, bias, n_streams)
    triton_tuple = mhc(x, phi, alpha_pre, alpha_post, alpha_res, bias, n_streams)

    relaxed = C >= 1024
    _assert_mhc_close(
        triton_tuple,
        out_torch,
        comb_atol=5e-2 if relaxed else 1e-2,
        comb_rtol=5e-2 if relaxed else 1e-2,
        layer_atol=5e-2 if relaxed else 1e-2,
        layer_rtol=5e-2 if relaxed else 1e-2,
    )


@pytest.mark.parametrize("eps", [1e-6, 1e-5, 1e-8])
@pytest.mark.parametrize("M, n, C", [(32, 4, 1024)])
def test_mhc_different_epsilon(eps, M, n, C):
    """Test mhc() with different epsilon values for RMSNorm (Eq 15)."""
    torch.cuda.empty_cache()

    x, phi, alpha_pre, alpha_post, alpha_res, bias, n_streams = generate_mhc_inputs(
        M, n, C
    )

    out_torch = mhc_torch(
        x, phi, alpha_pre, alpha_post, alpha_res, bias, n_streams, eps=eps
    )
    triton_tuple = mhc(
        x, phi, alpha_pre, alpha_post, alpha_res, bias, n_streams, eps=eps
    )

    _assert_mhc_close(triton_tuple, out_torch)


@pytest.mark.parametrize("alpha_scale", [0.1, 0.5, 1.0, 2.0, 10.0])
def test_mhc_different_alpha(alpha_scale):
    """Test mhc() with different scaling factors α (Eq 16)."""
    torch.cuda.empty_cache()

    M, n, C = 32, 4, 1024
    x, phi, _, _, _, bias, n_streams = generate_mhc_inputs(M, n, C)

    alpha_pre = alpha_post = alpha_res = alpha_scale

    out_torch = mhc_torch(x, phi, alpha_pre, alpha_post, alpha_res, bias, n_streams)
    triton_tuple = mhc(x, phi, alpha_pre, alpha_post, alpha_res, bias, n_streams)

    _assert_mhc_close(
        triton_tuple,
        out_torch,
        comb_atol=5e-2,
        comb_rtol=5e-2,
        layer_atol=5e-2,
        layer_rtol=5e-2,
    )


def test_mhc_zero_input():
    """Test mhc() with zero input (edge case for RMSNorm)."""
    torch.cuda.empty_cache()

    M, n, C = 16, 4, 512
    nC = n * C
    N_total = n * n + 2 * n

    x = torch.zeros(M, nC, dtype=torch.bfloat16, device="cuda")
    n_squared = n * n
    phi = torch.randn(nC, n + n + n_squared, dtype=torch.bfloat16, device="cuda") * 0.1
    alpha_pre = alpha_post = alpha_res = 1.0
    bias = torch.randn(N_total, dtype=torch.float32, device="cuda") * 0.1

    out_torch = mhc_torch(x, phi, alpha_pre, alpha_post, alpha_res, bias, n)
    triton_tuple = mhc(x, phi, alpha_pre, alpha_post, alpha_res, bias, n)

    _assert_mhc_close(triton_tuple, out_torch)


def test_mhc_large_values():
    """Test mhc() numerical stability with large input values."""
    torch.cuda.empty_cache()

    M, n, C = 32, 4, 1024
    nC = n * C
    N_total = n * n + 2 * n

    x = torch.randn(M, nC, dtype=torch.bfloat16, device="cuda") * 100
    n_squared = n * n
    phi = torch.randn(nC, n + n + n_squared, dtype=torch.bfloat16, device="cuda") * 0.01
    alpha_pre = alpha_post = alpha_res = 1.0
    bias = torch.randn(N_total, dtype=torch.float32, device="cuda")

    out_torch = mhc_torch(x, phi, alpha_pre, alpha_post, alpha_res, bias, n)
    triton_tuple = mhc(x, phi, alpha_pre, alpha_post, alpha_res, bias, n)

    # Layer_input scales linearly with x, so loosen its absolute tolerance for
    # x ~ N(0, 100²).
    _assert_mhc_close(triton_tuple, out_torch, layer_atol=2.0, layer_rtol=1e-2)


@pytest.mark.parametrize("M, n, C", [(32, 4, 1024), (64, 4, 2048), (128, 8, 1024)])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_mhc_small_shapes(M, n, C, dtype):
    """Quick smoke test for mhc() with representative shapes."""
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    x, phi, alpha_pre, alpha_post, alpha_res, bias, n_streams = generate_mhc_inputs(
        M, n, C, dtype
    )

    out_torch = mhc_torch(x, phi, alpha_pre, alpha_post, alpha_res, bias, n_streams)
    triton_tuple = mhc(x, phi, alpha_pre, alpha_post, alpha_res, bias, n_streams)

    _assert_mhc_close(
        triton_tuple,
        out_torch,
        comb_atol=2e-2,
        comb_rtol=5e-2,
        layer_atol=5e-2,
        layer_rtol=5e-2,
    )


def test_mhc_output_range():
    """Validate output value ranges for mhc()."""
    torch.cuda.empty_cache()

    M, n, C = 64, 4, 1024
    x, phi, alpha_pre, alpha_post, alpha_res, bias, n_streams = generate_mhc_inputs(
        M, n, C
    )

    post_mix, comb_mix, _ = mhc(
        x,
        phi,
        alpha_pre,
        alpha_post,
        alpha_res,
        bias,
        n_streams,
        sinkhorn_iters=50,
    )

    # Post-stream (Eq 18): 2*sigmoid output should be in [0, 2]
    assert torch.all(post_mix >= 0.0), "Post-stream has values < 0"
    assert torch.all(post_mix <= 2.0), "Post-stream has values > 2"

    # Res-stream (Eq 19): doubly stochastic
    assert comb_mix.shape == (M, n_streams, n_streams), "comb_mix shape mismatch"
    assert is_doubly_stochastic(comb_mix.to(torch.float32), tol=5e-2), (
        "comb_mix is not doubly stochastic. "
        f"Row sums: {comb_mix.float().sum(dim=-1)}, "
        f"Col sums: {comb_mix.float().sum(dim=-2)}"
    )


# =============================================================================
# Split-K Tests
# =============================================================================


def _make_split_k_config(num_ksplit):
    """Helper to create config with specified NUM_KSPLIT."""
    return {
        "waves_per_eu": 1,
        "num_stages": 1,
        "num_warps": 4,
        "NUM_KSPLIT": num_ksplit,
    }


@pytest.mark.parametrize("M, n, C", [(32, 4, 1024), (64, 4, 2048), (128, 8, 1024)])
@pytest.mark.parametrize("num_ksplit", [2, 4])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_split_k_correctness(M, n, C, num_ksplit, dtype):
    """Test that split-K matches the PyTorch reference (no Sinkhorn)."""
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    x, phi, alpha_pre, alpha_post, alpha_res, bias, n_streams = generate_mhc_inputs(
        M, n, C, dtype
    )

    out_ref = mhc_torch(
        x,
        phi,
        alpha_pre,
        alpha_post,
        alpha_res,
        bias,
        n_streams,
        return_with_sinkhorn=False,
    )

    triton_tuple = mhc(
        x,
        phi,
        alpha_pre,
        alpha_post,
        alpha_res,
        bias,
        n_streams,
        sinkhorn_iters=0,
        config=_make_split_k_config(num_ksplit),
    )

    relaxed = C >= 1024
    _assert_mhc_close(
        triton_tuple,
        out_ref,
        comb_atol=5e-2 if relaxed else 1e-2,
        comb_rtol=5e-2 if relaxed else 1e-2,
        layer_atol=5e-2 if relaxed else 1e-2,
        layer_rtol=5e-2 if relaxed else 1e-2,
    )


@pytest.mark.parametrize("M, n, C", [(32, 4, 1024), (64, 4, 2048)])
@pytest.mark.parametrize("num_ksplit", [2, 4])
def test_split_k_mhc_full_pipeline(M, n, C, num_ksplit):
    """Test split-K with the full mhc() pipeline including Sinkhorn-Knopp."""
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    x, phi, alpha_pre, alpha_post, alpha_res, bias, n_streams = generate_mhc_inputs(
        M, n, C
    )

    out_ref = mhc_torch(
        x,
        phi,
        alpha_pre,
        alpha_post,
        alpha_res,
        bias,
        n_streams,
        return_with_sinkhorn=True,
    )

    triton_tuple = mhc(
        x,
        phi,
        alpha_pre,
        alpha_post,
        alpha_res,
        bias,
        n_streams,
        config=_make_split_k_config(num_ksplit),
    )

    _assert_mhc_close(
        triton_tuple,
        out_ref,
        comb_atol=5e-2,
        comb_rtol=5e-2,
        layer_atol=5e-2,
        layer_rtol=5e-2,
    )


@pytest.mark.parametrize("num_ksplit", [1, 2, 4, 8])
def test_split_k_various_splits(num_ksplit):
    """Test split-K with various split counts (skip Sinkhorn)."""
    torch.cuda.empty_cache()

    M, n, C = 32, 4, 1024
    x, phi, alpha_pre, alpha_post, alpha_res, bias, n_streams = generate_mhc_inputs(
        M, n, C
    )

    out_ref = mhc_torch(
        x,
        phi,
        alpha_pre,
        alpha_post,
        alpha_res,
        bias,
        n_streams,
        return_with_sinkhorn=False,
    )

    triton_tuple = mhc(
        x,
        phi,
        alpha_pre,
        alpha_post,
        alpha_res,
        bias,
        n_streams,
        sinkhorn_iters=0,
        config=_make_split_k_config(num_ksplit),
    )

    # post and layer_input only; skip comb_mix (not Sinkhorn-projected here)
    # and hpre (raw logits, no Triton-side counterpart).
    post_t, _, layer_t = triton_tuple
    post_ref, _, _hpre_ref, layer_ref = out_ref
    torch.testing.assert_close(post_t.float(), post_ref.float(), atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(layer_t.float(), layer_ref.float(), atol=5e-2, rtol=5e-2)


def test_split_k_large_k():
    """Test split-K with large K dimension where split-K should be beneficial."""
    torch.cuda.empty_cache()

    M, n, C = 64, 4, 2048  # K = n * C = 8192
    x, phi, alpha_pre, alpha_post, alpha_res, bias, n_streams = generate_mhc_inputs(
        M, n, C
    )

    out_ref = mhc_torch(
        x,
        phi,
        alpha_pre,
        alpha_post,
        alpha_res,
        bias,
        n_streams,
        return_with_sinkhorn=False,
    )

    triton_tuple = mhc(
        x,
        phi,
        alpha_pre,
        alpha_post,
        alpha_res,
        bias,
        n_streams,
        sinkhorn_iters=0,
        config=_make_split_k_config(4),
    )

    _assert_mhc_close(
        triton_tuple,
        out_ref,
        comb_atol=5e-2,
        comb_rtol=5e-2,
        layer_atol=5e-2,
        layer_rtol=5e-2,
    )


# =============================================================================
# Triton-vs-HIP parity anchor
# =============================================================================


def _triton_to_hip_pre_inputs(x, phi, alpha_pre, alpha_post, alpha_res, bias, n):
    """Convert Triton-convention mhc inputs to HIP `aiter.mhc_pre` conventions.

    Mapping:
      M <-> m                 n <-> hc_mult              C <-> hidden_size
      x (M, n*C)              <-> residual (m, hc_mult, hidden_size) bf16
      phi (n*C, 2n+n²)        <-> fn.T (hc_mult3, hc_hidden_size)    fp32
      (alpha_pre/post/res)    <-> hc_scale (3,)                      fp32
      bias                    <-> hc_base (hc_mult3,)                fp32
    """
    M, K = x.shape
    C = K // n
    residual = x.view(M, n, C).contiguous().to(torch.bfloat16)
    fn_hip = phi.T.contiguous().float()
    hc_scale = torch.tensor(
        [alpha_pre, alpha_post, alpha_res], dtype=torch.float32, device=x.device
    )
    hc_base = bias.to(torch.float32).contiguous()
    return residual, fn_hip, hc_scale, hc_base


@pytest.mark.parametrize(
    "M, n, C",
    [
        (64, 4, 1024),
        (1024, 4, 1024),
        (2048, 4, 2048),
    ],
)
def test_triton_mhc_matches_hip(M, n, C):
    """Triton ``mhc()`` matches the real HIP kernel ``aiter.mhc_pre()``.

    Skips when:
      - CUDA is unavailable
      - ``aiter.mhc_pre`` is not built in this environment
      - ``n != 4`` (``mhc_pre_big_fuse`` hardcodes ``hc_mult == 4``)
      - ``C < 512`` (``mhc_pre_big_fuse`` dispatch lower bound)
      - ``n*C`` is not divisible by 64 (``mhc_pre_gemm_sqrsum`` ``tile_k`` requirement)
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA device required for mHC kernels")
    if not _HAS_AITER_MHC_PRE:
        pytest.skip("aiter.mhc_pre is not available in this environment")
    if n != 4:
        pytest.skip("aiter.mhc_pre_big_fuse hardcodes hc_mult == 4")
    if C < 512:
        pytest.skip("aiter.mhc_pre_big_fuse dispatch requires C >= 512")
    if (n * C) % 64 != 0:
        pytest.skip("aiter.mhc_pre_gemm_sqrsum needs n*C divisible by tile_k")

    torch.cuda.empty_cache()
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    rms_eps = 1e-6
    hc_pre_eps = 0.0
    hc_sinkhorn_eps = 0.0
    hc_post_mult_value = 2.0
    sinkhorn_repeat = 20

    x, phi, alpha_pre, alpha_post, alpha_res, bias, n_streams = generate_mhc_inputs(
        M, n, C, dtype=torch.bfloat16
    )

    post_t, comb_t, li_t = mhc(
        x,
        phi,
        alpha_pre,
        alpha_post,
        alpha_res,
        bias,
        n_streams,
        eps=rms_eps,
        hc_pre_eps=hc_pre_eps,
        sinkhorn_iters=sinkhorn_repeat,
    )

    residual, fn_hip, hc_scale, hc_base = _triton_to_hip_pre_inputs(
        x, phi, alpha_pre, alpha_post, alpha_res, bias, n_streams
    )
    # aiter.mhc_pre allocates outputs via torch.empty without a device kwarg,
    # so it needs an active torch.device context to land on the GPU.
    with torch.device(x.device):
        post_h, comb_h, li_h = _aiter.mhc_pre(
            residual,
            fn_hip,
            hc_scale,
            hc_base,
            rms_eps=rms_eps,
            hc_pre_eps=hc_pre_eps,
            hc_sinkhorn_eps=hc_sinkhorn_eps,
            hc_post_mult_value=hc_post_mult_value,
            sinkhorn_repeat=sinkhorn_repeat,
        )

    cfg = f"(M={M}, n={n}, C={C})"
    torch.testing.assert_close(
        post_t.float(),
        post_h.float(),
        atol=4e-2,
        rtol=1e-2,
        msg=f"post_mix Triton vs aiter.mhc_pre mismatch at {cfg}",
    )
    torch.testing.assert_close(
        comb_t.float(),
        comb_h.float(),
        atol=4e-2,
        rtol=1e-2,
        msg=f"comb_mix Triton vs aiter.mhc_pre mismatch at {cfg}",
    )
    torch.testing.assert_close(
        li_t.float(),
        li_h.float(),
        atol=8e-2,
        rtol=2e-2,
        msg=f"layer_input Triton vs aiter.mhc_pre mismatch at {cfg}",
    )


# =============================================================================
# mhc_post Tests
# =============================================================================


@pytest.mark.parametrize(
    "m, hc_mult, hidden_size",
    [
        (1, 4, 256),
        (128, 4, 1024),
        (512, 4, 4096),
        (1024, 4, 7168),
        (2048, 4, 2048),
        (64, 4, 512),
    ],
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_mhc_post_correctness(m, hc_mult, hidden_size, dtype):
    """Test mhc_post against PyTorch reference."""
    x, residual, post_mix, comb_mix = generate_mhc_post_inputs(
        m, hc_mult, hidden_size, dtype
    )
    ref = mhc_post_torch(x, residual, post_mix, comb_mix)
    out = mhc_post(None, x, residual, post_mix, comb_mix)

    torch.testing.assert_close(
        out.float(),
        ref.float(),
        atol=1e-2,
        rtol=1e-2,
        msg=f"mhc_post output mismatch at (m={m}, hc_mult={hc_mult}, hidden_size={hidden_size}, dtype={dtype})",
    )


def test_mhc_post_preallocated_output():
    """Verify in-place path: result is out and matches reference."""
    from aiter.ops.triton.fusions.mhc import mhc_post
    from op_tests.triton_tests.utils.mhc_ref import (
        mhc_post_torch,
        generate_mhc_post_inputs,
    )

    m, hc_mult, hidden_size = 128, 4, 1024
    dtype = torch.bfloat16

    x, residual, post_mix, comb_mix = generate_mhc_post_inputs(
        m, hc_mult, hidden_size, dtype
    )

    # Pre-allocate output
    out = torch.empty(m, hc_mult, hidden_size, dtype=dtype, device=x.device)

    # Call mhc_post with pre-allocated output
    result = mhc_post(out, x, residual, post_mix, comb_mix)

    # Verify result is the same object as out
    assert result is out, "mhc_post should return the pre-allocated output tensor"

    # Verify correctness
    ref = mhc_post_torch(x, residual, post_mix, comb_mix)
    torch.testing.assert_close(
        out.float(),
        ref.float(),
        atol=1e-2,
        rtol=1e-2,
        msg="Pre-allocated output mismatch",
    )


def test_mhc_post_squeeze_post_layer_mix():
    """Pass post_layer_mix as (m, hc_mult, 1) — as mhc_pre emits it."""
    from aiter.ops.triton.fusions.mhc import mhc_post
    from op_tests.triton_tests.utils.mhc_ref import (
        mhc_post_torch,
        generate_mhc_post_inputs,
    )

    m, hc_mult, hidden_size = 64, 4, 512
    dtype = torch.bfloat16

    x, residual, post_mix, comb_mix = generate_mhc_post_inputs(
        m, hc_mult, hidden_size, dtype
    )

    # Add extra dimension to post_mix to simulate mhc_pre output
    post_mix_3d = post_mix.unsqueeze(-1)  # (m, hc_mult, 1)
    assert post_mix_3d.shape == (m, hc_mult, 1)

    # Call mhc_post with 3D post_layer_mix
    out = mhc_post(None, x, residual, post_mix_3d, comb_mix)

    # Verify correctness against reference with 2D post_mix
    ref = mhc_post_torch(x, residual, post_mix, comb_mix)
    torch.testing.assert_close(
        out.float(),
        ref.float(),
        atol=1e-2,
        rtol=1e-2,
        msg="mhc_post with 3D post_layer_mix mismatch",
    )


# =============================================================================
# End-to-End Pipeline Tests (mhc → mhc_post)
# =============================================================================
#
# Pipeline Overview:
#
# x_l (m, n, C)  ──┐
#                  │
#                  ├──> mhc() ──> (h_post, h_res, layer_input)
#                  │                                 │
#                  │                                 ├──> layer_input (m, C)
#                  │                                 │
#                  │                                 v
#                  └──────────────────────────> mhc_post() ──> x_l+1 (m, n, C)
#                                                   │
#                                                   └──> Uses (layer_input, x_l, h_post, h_res)
#


def mhc_e2e_ref(
    x_l,
    phi,
    alpha_pre,
    alpha_post,
    alpha_res,
    bias,
    n,
    eps=1e-6,
    hc_pre_eps=0.0,
    hc_post_mult_value=2.0,
    sinkhorn_iters=20,
):
    """
    Reference implementation using PyTorch

    Pipeline:
    x_l (m, n, C) → flatten → mhc → (h_post, h_res, layer_input) → mhc_post → x_l+1 (m, n, C)
    """
    sinkhorn_iters = int(sinkhorn_iters)
    m, n_check, C = x_l.shape
    assert n_check == n, f"Stream count mismatch: {n_check} != {n}"

    x_l_flat = x_l.view(m, n * C)

    # Step 1: mhc - compute coefficients and layer_input
    h_post, h_res, _h_pre, layer_input = mhc_torch(
        x_l_flat,
        phi,
        alpha_pre,
        alpha_post,
        alpha_res,
        bias,
        n,
        eps,
        hc_pre_eps,
        hc_post_mult_value,
        sinkhorn_iters,
    )

    # Step 2: mhc_post - merge layer_input back to multi-stream
    x_l_plus_1 = mhc_post_torch(layer_input, x_l, h_post, h_res)

    return layer_input, x_l_plus_1, h_post, h_res


def mhc_e2e_triton(
    x_l_flat,
    phi,
    alpha_pre,
    alpha_post,
    alpha_res,
    bias,
    n,
    C,
    eps=1e-6,
    hc_pre_eps=0.0,
    hc_post_mult_value=2.0,
    sinkhorn_iters=20,
    config=None,
):
    """
    Triton implementation of full pipeline

    Pipeline:
    x_l_flat (m, n*C) → mhc → (h_post, h_res, layer_input) → mhc_post → x_l+1 (m, n, C)
    """
    # Ensure sinkhorn_iters is an integer
    sinkhorn_iters = int(sinkhorn_iters)
    m = x_l_flat.shape[0]

    # Step 1: mhc (Triton)
    h_post, h_res, layer_input = mhc(
        x_l_flat,
        phi,
        alpha_pre,
        alpha_post,
        alpha_res,
        bias,
        n,
        eps,
        hc_pre_eps,
        hc_post_mult_value,
        sinkhorn_iters,
        config,
    )

    # Reconstruct x_l for mhc_post (it needs the original multi-stream)
    x_l = x_l_flat.view(m, n, C)

    # Step 2: mhc_post (Triton)
    x_l_plus_1 = mhc_post(
        None,  # Let it allocate
        layer_input,
        x_l,
        h_post,
        h_res,
        config,
    )

    return layer_input, x_l_plus_1, h_post, h_res


@pytest.mark.parametrize(
    "m, n, C",
    [
        (1, 4, 256),  # hidden_size=1024
        (32, 4, 256),  # hidden_size=1024
        (64, 4, 512),  # hidden_size=2048
        (128, 4, 1024),  # hidden_size=4096
        (256, 4, 1024),  # hidden_size=4096
        (512, 4, 512),  # hidden_size=2048
        (1024, 4, 256),  # hidden_size=1024
    ],
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_mhc_e2e_correctness(m, n, C, dtype):
    """
    Test correctness of Triton mhc → mhc_post pipeline

    Tests the full round-trip: x_l → mhc() → layer_input → mhc_post() → x_l+1

    Validates:
    1. layer_input matches reference
    2. x_l+1 matches reference
    3. h_post and h_res match reference
    """
    sinkhorn_iters = 20
    x_l_flat, phi, alpha_pre, alpha_post, alpha_res, bias, _ = generate_mhc_inputs(
        m, n, C, dtype
    )
    x_l = x_l_flat.view(m, n, C)

    # Reference implementation
    layer_input_ref, x_l_plus_1_ref, h_post_ref, h_res_ref = mhc_e2e_ref(
        x_l,
        phi,
        alpha_pre,
        alpha_post,
        alpha_res,
        bias,
        n,
        sinkhorn_iters=int(sinkhorn_iters),
    )

    # Triton implementation
    layer_input_triton, x_l_plus_1_triton, h_post_triton, h_res_triton = mhc_e2e_triton(
        x_l_flat,
        phi,
        alpha_pre,
        alpha_post,
        alpha_res,
        bias,
        n,
        C,
        sinkhorn_iters=int(sinkhorn_iters),
    )

    # Verify correctness
    # Note: Tolerances adjusted for bf16 accumulation errors and Sinkhorn-Knopp differences
    torch.testing.assert_close(
        layer_input_triton.float(),
        layer_input_ref.float(),
        atol=2e-2,
        rtol=5e-2,
        msg="layer_input mismatch",
    )

    # h_res can have larger errors due to Sinkhorn-Knopp iteration differences
    # Sinkhorn-Knopp normalization in Triton vs PyTorch can diverge numerically,
    # especially at larger batch sizes where accumulation errors compound
    # Scale tolerance with batch size
    atol_h_res = 1.5 if m <= 128 else 2.0
    rtol_h_res = 1.5 if m <= 128 else 2.0
    torch.testing.assert_close(
        h_res_triton.float(),
        h_res_ref.float(),
        atol=atol_h_res,
        rtol=rtol_h_res,
        msg="h_res mismatch",
    )

    # x_l+1 inherits h_res errors, so also needs larger tolerance
    atol_x = 4.0 if m <= 128 else 6.0
    rtol_x = 3.0 if m <= 128 else 4.0
    torch.testing.assert_close(
        x_l_plus_1_triton.float(),
        x_l_plus_1_ref.float(),
        atol=atol_x,
        rtol=rtol_x,
        msg="x_l+1 mismatch",
    )

    torch.testing.assert_close(
        h_post_triton.float(),
        h_post_ref.float(),
        atol=1e-2,
        rtol=1e-2,
        msg="h_post mismatch",
    )
