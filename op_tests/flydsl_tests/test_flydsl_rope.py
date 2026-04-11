#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for FlyDSL fused RoPE + KV Cache kernel.

Tests correctness against the same reference used by the Triton
``test_fused_qk_rope_reshape_and_cache`` (generate_rope_inputs +
ref_rope_sbhd_fwd), and validates KV cache writes.

Category A tests call the FlyDSL kernel directly (supported configs).
Category B tests call the wrapper, verifying Triton fallback correctness.

Usage:
    # Default CI (~47 configs):
    pytest op_tests/flydsl_tests/test_flydsl_rope.py -v -s

    # Full model sweep:
    FLYDSL_ALL_MODELS=1 pytest op_tests/flydsl_tests/test_flydsl_rope.py -v -s

    # With AITER cross-check:
    FLYDSL_BENCH=1 pytest op_tests/flydsl_tests/test_flydsl_rope.py -v -s
"""

import os

import pytest
import torch

from op_tests.test_rope import ref_rope_sbhd_fwd, RotateStyle
from op_tests.triton_tests.rope.test_rope import generate_rope_inputs

# -- FlyDSL availability --
try:
    from aiter.ops.flydsl.utils import is_flydsl_available

    HAS_FLYDSL = is_flydsl_available()
except ImportError:
    HAS_FLYDSL = False

if HAS_FLYDSL:
    try:
        from aiter.ops.flydsl.rope_kernels import (
            flydsl_fused_qk_rope_reshape_and_cache,
        )

        try:
            from kernels.fused_rope_cache_kernel import build_fused_rope_cache_module
        except ImportError:
            from aiter.ops.flydsl.kernels.fused_rope_cache_kernel import (
                build_fused_rope_cache_module,
            )
    except ImportError:
        HAS_FLYDSL = False

# -- AITER Triton availability --
try:
    from aiter.ops.triton.fusions.fused_kv_cache import fused_qk_rope_reshape_and_cache

    HAS_AITER_TRITON = True
except ImportError:
    HAS_AITER_TRITON = False

# -- Constants (aligned with Triton test) --
DEVICE = "cuda:0"
BLOCK_SIZE = 16
X_SIZE = (
    8  # non-flash cache interleave factor (Triton test uses 8; FlyDSL kernel uses 16)
)
ATOL = 1e-1
RTOL = 1e-1

# -- Model configs for sweep --
ALL_MODELS = [
    ("Llama-8B-TP1", 32, 8, 128),
    ("Llama-8B-TP8", 4, 1, 128),
    ("Llama-70B-TP1", 64, 8, 128),
    ("Llama-70B-TP8", 8, 1, 128),
    ("Llama-405B-TP1", 128, 8, 128),
    ("Llama-405B-TP8", 16, 1, 128),
    ("Qwen3-72B-TP1", 64, 4, 128),
    ("Qwen3-72B-TP8", 8, 1, 128),
    ("Qwen3-235B-TP1", 64, 4, 64),
    ("Qwen3-235B-TP8", 8, 1, 64),
]

# -- Kernel cache for Category A (direct kernel) --
_kernel_cache = {}


def _get_kernel(D, QH, KH, block_size, flash_layout, dtype_str):
    key = (D, QH, KH, block_size, flash_layout, dtype_str)
    if key not in _kernel_cache:
        _kernel_cache[key] = build_fused_rope_cache_module(
            head_dim=D,
            num_q_heads=QH,
            num_kv_heads=KH,
            block_size=block_size,
            is_neox=True,
            flash_layout=flash_layout,
            dtype_str=dtype_str,
        )
    return _kernel_cache[key]


# ---------------------------------------------------------------------------
# Core test runner
# ---------------------------------------------------------------------------
def run_test(
    T,
    QH_per_KH,
    KH,
    D,
    rotate_style=RotateStyle.NEOX,
    reuse_freqs_front_part=False,
    flash_layout=True,
    use_offsets=False,
    dtype=torch.bfloat16,
    block_size=BLOCK_SIZE,
    x_size=X_SIZE,
    use_wrapper=False,
):
    """Run FlyDSL kernel and validate Q/K outputs + KV cache against reference.

    When use_wrapper=False: calls the FlyDSL kernel directly (Category A).
    When use_wrapper=True: calls the wrapper which may fallback to Triton (Category B).
    """
    QH = QH_per_KH * KH

    # -- Generate inputs (same helper as Triton test) --
    q, k, _, _, freqs, positions, offsets, cos, sin = generate_rope_inputs(
        B=1,
        S=T,
        H=KH,
        Q=QH_per_KH,
        D=D,
        cached=True,
        reuse_freqs_front_part=reuse_freqs_front_part,
        nope=False,
        pos=True,
        offs=use_offsets,
        two_inputs=True,
        layout="thd",
        dtype=dtype,
    )
    v = torch.randn_like(k)

    # -- Allocate caches --
    num_blocks = max(32, (T + block_size - 1) // block_size + 1)
    if flash_layout:
        key_cache = torch.zeros(
            num_blocks, block_size, KH, D, dtype=dtype, device=DEVICE
        )
        value_cache = torch.zeros(
            num_blocks, block_size, KH, D, dtype=dtype, device=DEVICE
        )
    else:
        key_cache = torch.zeros(
            num_blocks,
            KH,
            D // x_size,
            block_size,
            x_size,
            dtype=dtype,
            device=DEVICE,
        )
        value_cache = torch.zeros(
            num_blocks,
            KH,
            D,
            block_size,
            dtype=dtype,
            device=DEVICE,
        )

    # Slot mapping: random permutation (matches Triton test)
    slot_mapping = torch.randperm(T, device=DEVICE)

    # Scale factors (bf16 path: 1.0)
    k_scale = torch.ones([1], dtype=torch.float32, device=DEVICE)[0]
    v_scale = torch.ones([1], dtype=torch.float32, device=DEVICE)[0]

    # -- Reference: RoPE rotation (same as Triton test) --
    is_neox = rotate_style == RotateStyle.NEOX
    ref_freqs = freqs[
        positions if offsets is None else torch.add(positions, offsets)
    ].squeeze(-2)

    q_ref = ref_rope_sbhd_fwd(
        q.unsqueeze(0),
        ref_freqs,
        rotate_style=rotate_style,
        reuse_freqs_front_part=reuse_freqs_front_part,
        nope_first=False,
    ).squeeze(0)
    k_ref = ref_rope_sbhd_fwd(
        k.unsqueeze(0),
        ref_freqs,
        rotate_style=rotate_style,
        reuse_freqs_front_part=reuse_freqs_front_part,
        nope_first=False,
    ).squeeze(0)

    # -- Reference: KV cache write (same logic as Triton test) --
    ref_key_cache = key_cache.clone()
    ref_value_cache = value_cache.clone()
    slot_t = slot_mapping // block_size
    slot_b = slot_mapping % block_size
    if flash_layout:
        ref_key_cache[slot_t, slot_b] = k_ref
        ref_value_cache[slot_t, slot_b] = v
    else:
        ref_key_cache[slot_t, :, :, slot_b, :] = k_ref.reshape(
            T, KH, D // x_size, x_size
        )
        ref_value_cache[slot_t, :, :, slot_b] = v

    # -- Run kernel under test --
    if use_wrapper:
        q_out, k_out, _, _ = flydsl_fused_qk_rope_reshape_and_cache(
            q.clone(),
            k.clone(),
            v,
            key_cache,
            value_cache,
            slot_mapping.to(torch.int64),
            positions.to(torch.int64),
            cos,
            sin,
            k_scale,
            v_scale,
            is_neox=is_neox,
            flash_layout=flash_layout,
            apply_scale=False,
            offs=offsets,
            output_zeros=False,
        )
    else:
        # Category A: call the FlyDSL kernel directly
        q_out = torch.empty_like(q)
        k_out = torch.empty_like(k)

        cos_2d = cos.squeeze(1).squeeze(1) if cos.ndim == 4 else cos
        sin_2d = sin.squeeze(1).squeeze(1) if sin.ndim == 4 else sin

        dtype_str = "bf16" if dtype == torch.bfloat16 else "f16"
        launch_fn = _get_kernel(D, QH, KH, block_size, flash_layout, dtype_str)
        launch_fn(
            q,
            k,
            v,
            positions.to(torch.int32),
            cos_2d,
            sin_2d,
            slot_mapping.to(torch.int32),
            key_cache,
            value_cache,
            q_out,
            k_out,
            T,
            stream=torch.cuda.current_stream(),
        )
    torch.cuda.synchronize()

    # -- Validate Q and K outputs --
    torch.testing.assert_close(q_out, q_ref, atol=ATOL, rtol=RTOL)
    torch.testing.assert_close(k_out, k_ref, atol=ATOL, rtol=RTOL)

    # -- Validate KV cache at written slots --
    if flash_layout:
        torch.testing.assert_close(
            key_cache[slot_t, slot_b],
            ref_key_cache[slot_t, slot_b],
            atol=ATOL,
            rtol=RTOL,
        )
        torch.testing.assert_close(
            value_cache[slot_t, slot_b],
            ref_value_cache[slot_t, slot_b],
            atol=ATOL,
            rtol=RTOL,
        )
    else:
        torch.testing.assert_close(
            key_cache[slot_t, :, :, slot_b, :],
            ref_key_cache[slot_t, :, :, slot_b, :],
            atol=ATOL,
            rtol=RTOL,
        )
        torch.testing.assert_close(
            value_cache[slot_t, :, :, slot_b],
            ref_value_cache[slot_t, :, :, slot_b],
            atol=ATOL,
            rtol=RTOL,
        )

    # -- Validate full KV cache (ensures no stray writes) --
    torch.testing.assert_close(key_cache, ref_key_cache, atol=ATOL, rtol=RTOL)
    torch.testing.assert_close(value_cache, ref_value_cache, atol=ATOL, rtol=RTOL)


# ===========================================================================
# Category A: Direct FlyDSL kernel tests (flash, bf16, NEOX)
# ===========================================================================


@pytest.mark.skipif(not HAS_FLYDSL, reason="FlyDSL not installed")
@pytest.mark.parametrize("QH_per_KH", [1, 4, 16])
@pytest.mark.parametrize("KH", [1, 8])
@pytest.mark.parametrize("D", [64, 128])
def test_flydsl_kernel_direct(QH_per_KH, KH, D):
    T = 1  # T>1 has FlyDSL soffset issues; decode (T=1) is the target
    """Category A: Direct FlyDSL kernel — flash, bf16, NEOX."""
    run_test(
        T=T,
        QH_per_KH=QH_per_KH,
        KH=KH,
        D=D,
        rotate_style=RotateStyle.NEOX,
        reuse_freqs_front_part=True,
        flash_layout=True,
        use_wrapper=False,
    )


# ===========================================================================
# Category B: Wrapper fallback tests (verify Triton fallback path)
# ===========================================================================


@pytest.mark.skipif(not HAS_FLYDSL, reason="FlyDSL not installed")
@pytest.mark.parametrize("T", [1, 2, 4])
@pytest.mark.parametrize("QH_per_KH,KH", [(1, 1), (1, 8), (16, 1), (16, 8)])
@pytest.mark.parametrize("D", [128])
@pytest.mark.parametrize("reuse_freqs_front_part", [False, True])
def test_wrapper_triton_aligned(T, QH_per_KH, KH, D, reuse_freqs_front_part):
    """Category B: Triton-aligned params via wrapper (may use FlyDSL or fallback)."""
    run_test(
        T=T,
        QH_per_KH=QH_per_KH,
        KH=KH,
        D=D,
        rotate_style=RotateStyle.NEOX,
        reuse_freqs_front_part=reuse_freqs_front_part,
        flash_layout=True,
        use_wrapper=True,
    )


@pytest.mark.skipif(not HAS_FLYDSL, reason="FlyDSL not installed")
@pytest.mark.parametrize("T", [1, 4])
@pytest.mark.parametrize("QH_per_KH,KH", [(1, 8), (16, 1)])
@pytest.mark.parametrize("D", [128])
def test_wrapper_fallback_gptj(T, QH_per_KH, KH, D):
    """Category B: GPT-J rotation → Triton fallback."""
    run_test(
        T=T,
        QH_per_KH=QH_per_KH,
        KH=KH,
        D=D,
        rotate_style=RotateStyle.GPTJ,
        flash_layout=True,
        use_wrapper=True,
    )


@pytest.mark.skipif(not HAS_FLYDSL, reason="FlyDSL not installed")
@pytest.mark.parametrize("T", [1, 4])
@pytest.mark.parametrize("QH_per_KH,KH", [(1, 8), (16, 1)])
@pytest.mark.parametrize("D", [128])
def test_wrapper_fallback_offsets(T, QH_per_KH, KH, D):
    """Category B: Offsets → Triton fallback."""
    run_test(
        T=T,
        QH_per_KH=QH_per_KH,
        KH=KH,
        D=D,
        rotate_style=RotateStyle.NEOX,
        flash_layout=True,
        use_offsets=True,
        use_wrapper=True,
    )


@pytest.mark.skipif(not HAS_FLYDSL, reason="FlyDSL not installed")
@pytest.mark.parametrize("T", [1, 4])
@pytest.mark.parametrize("QH_per_KH,KH", [(1, 8), (16, 1)])
@pytest.mark.parametrize("D", [128])
def test_wrapper_fallback_nonflash(T, QH_per_KH, KH, D):
    """Category B: Non-flash layout → Triton fallback."""
    run_test(
        T=T,
        QH_per_KH=QH_per_KH,
        KH=KH,
        D=D,
        rotate_style=RotateStyle.NEOX,
        flash_layout=False,
        use_wrapper=True,
    )


@pytest.mark.skipif(not HAS_FLYDSL, reason="FlyDSL not installed")
@pytest.mark.parametrize("T", [2, 4])
def test_wrapper_fallback_qh64(T):
    """Category B: QH=64 (WARP_SIZE) with T>1 → Triton fallback."""
    run_test(
        T=T,
        QH_per_KH=8,
        KH=8,
        D=128,
        rotate_style=RotateStyle.NEOX,
        flash_layout=True,
        use_wrapper=True,
    )


@pytest.mark.skipif(not HAS_FLYDSL, reason="FlyDSL not installed")
def test_wrapper_fallback_f16_prefill():
    """Category B: f16 + KH=1 + large Qr + T>1 → falls back (soffset bug)."""
    run_test(
        T=4,
        QH_per_KH=16,
        KH=1,
        D=128,
        rotate_style=RotateStyle.NEOX,
        flash_layout=True,
        dtype=torch.float16,
        use_wrapper=True,
    )


@pytest.mark.skipif(not HAS_FLYDSL, reason="FlyDSL not installed")
@pytest.mark.parametrize("D", [64, 128])
def test_flydsl_kernel_direct_f16(D):
    """Category A: Direct FlyDSL kernel with f16 (T=1 decode)."""
    run_test(
        T=1,
        QH_per_KH=4,
        KH=8,
        D=D,
        rotate_style=RotateStyle.NEOX,
        reuse_freqs_front_part=True,
        flash_layout=True,
        use_wrapper=False,
        dtype=torch.float16,
    )


# ===========================================================================
# Multi-model sweep (opt-in via FLYDSL_ALL_MODELS=1)
# ===========================================================================


@pytest.mark.skipif(not HAS_FLYDSL, reason="FlyDSL not installed")
@pytest.mark.skipif(
    os.environ.get("FLYDSL_ALL_MODELS", "0") != "1",
    reason="Set FLYDSL_ALL_MODELS=1 for multi-model sweep",
)
@pytest.mark.parametrize(
    "model,QH,KH,D",
    [(m, qh, kh, d) for m, qh, kh, d in ALL_MODELS],
    ids=[m for m, _, _, _ in ALL_MODELS],
)
@pytest.mark.parametrize("T", [1, 32, 128])
def test_flydsl_rope_multi_model(model, QH, KH, D, T):
    """Full model sweep via wrapper (fallback for unsupported configs)."""
    QH_per_KH = QH // KH
    run_test(
        T=T,
        QH_per_KH=QH_per_KH,
        KH=KH,
        D=D,
        rotate_style=RotateStyle.NEOX,
        flash_layout=True,
        use_wrapper=True,
    )


# ===========================================================================
# AITER cross-check (opt-in via FLYDSL_BENCH=1)
# ===========================================================================


@pytest.mark.skipif(not HAS_FLYDSL, reason="FlyDSL not installed")
@pytest.mark.skipif(not HAS_AITER_TRITON, reason="AITER Triton not available")
@pytest.mark.skipif(
    os.environ.get("FLYDSL_BENCH", "0") != "1",
    reason="Set FLYDSL_BENCH=1 for AITER cross-check",
)
@pytest.mark.parametrize("T", [1, 32, 128])
@pytest.mark.parametrize("QH_per_KH,KH", [(1, 8), (16, 1)])
@pytest.mark.parametrize("D", [128])
def test_flydsl_vs_aiter_triton(T, QH_per_KH, KH, D):
    """Cross-check: FlyDSL wrapper vs Triton fused_qk_rope_reshape_and_cache."""

    q, k, _, _, freqs, positions, offsets, cos, sin = generate_rope_inputs(
        B=1,
        S=T,
        H=KH,
        Q=QH_per_KH,
        D=D,
        cached=True,
        reuse_freqs_front_part=False,
        nope=False,
        pos=True,
        offs=False,
        two_inputs=True,
        layout="thd",
        dtype=torch.bfloat16,
    )
    v = torch.randn_like(k)

    num_blocks = max(32, (T + BLOCK_SIZE - 1) // BLOCK_SIZE + 1)
    slot_mapping = torch.randperm(T, device=DEVICE)
    k_scale = torch.ones([1], dtype=torch.float32, device=DEVICE)[0]
    v_scale = torch.ones([1], dtype=torch.float32, device=DEVICE)[0]

    key_cache_fly = torch.zeros(
        num_blocks, BLOCK_SIZE, KH, D, dtype=torch.bfloat16, device=DEVICE
    )
    value_cache_fly = torch.zeros_like(key_cache_fly)
    key_cache_tri = key_cache_fly.clone()
    value_cache_tri = value_cache_fly.clone()

    # FlyDSL wrapper
    q_fly, k_fly, _, _ = flydsl_fused_qk_rope_reshape_and_cache(
        q.clone(),
        k.clone(),
        v,
        key_cache_fly,
        value_cache_fly,
        slot_mapping.to(torch.int64),
        positions.to(torch.int64),
        cos,
        sin,
        k_scale,
        v_scale,
        is_neox=True,
        flash_layout=True,
        apply_scale=False,
        output_zeros=False,
    )

    # Triton direct
    q_tri, k_tri, _, _ = fused_qk_rope_reshape_and_cache(
        q.clone(),
        k.clone(),
        v,
        key_cache_tri,
        value_cache_tri,
        slot_mapping,
        positions,
        cos,
        sin,
        k_scale,
        v_scale,
        is_neox=True,
        flash_layout=True,
        apply_scale=False,
        output_zeros=False,
    )
    torch.cuda.synchronize()

    torch.testing.assert_close(q_fly, q_tri, atol=ATOL, rtol=RTOL)
    torch.testing.assert_close(k_fly, k_tri, atol=ATOL, rtol=RTOL)
    torch.testing.assert_close(key_cache_fly, key_cache_tri, atol=ATOL, rtol=RTOL)
    torch.testing.assert_close(value_cache_fly, value_cache_tri, atol=ATOL, rtol=RTOL)
