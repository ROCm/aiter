#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for FlyDSL fused RoPE + KV Cache kernel.

Tests correctness against the same reference used by the Triton
``test_fused_qk_rope_reshape_and_cache`` (generate_rope_inputs +
ref_rope_sbhd_fwd), and validates KV cache writes.

Category A tests call the FlyDSL kernel directly (supported configs).
Category B tests call the wrapper, verifying Triton fallback correctness.
Category C tests verify fp8 KV cache quantization (apply_scale=True).
Category D tests full-dim cos/sin (reuse_freqs_front_part=False).
Category E tests pos_dtype i32 vs i64 (stride-2 view indexing).
Category F tests negative slot mapping (slot < 0 skips KV write).

Usage:
    # Default CI (~72 configs):
    pytest op_tests/flydsl_tests/test_flydsl_rope.py -v -s

    # Full model sweep:
    FLYDSL_ALL_MODELS=1 pytest op_tests/flydsl_tests/test_flydsl_rope.py -v -s

    # With AITER cross-check / benchmarking:
    FLYDSL_BENCH=1 pytest op_tests/flydsl_tests/test_flydsl_rope.py -v -s

    # CLI (runs all configs including fp8):
    PYTHONPATH=./ python op_tests/flydsl_tests/test_flydsl_rope.py
    FLYDSL_BENCH=1 PYTHONPATH=./ python op_tests/flydsl_tests/test_flydsl_rope.py
"""

import os
import time

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

FP8_DTYPE = torch.float8_e4m3fn
MAX_POS = 8192
KV_SCALE = 0.1  # round-trip-friendly: maps fp8 range to bf16 range

# -- Model configs for sweep --
# Configs confirmed supported (no fallback) in ATOM production.
ALL_MODELS = [
    ("Llama-8B-TP1",   32,  8, 128),
    ("Llama-8B-TP8",    4,  1, 128),
    ("Llama-70B-TP1",  64,  8, 128),
    ("Llama-70B-TP8",   8,  1, 128),
    ("Llama-405B-TP1", 128, 8, 128),
    ("Llama-405B-TP8",  16, 1, 128),
    ("Qwen3-72B-TP1",  64,  4, 128),
    ("Qwen3-72B-TP8",   8,  1, 128),
    ("GPT-OSS-TP1",    64,  8,  64),
    ("GPT-OSS-TP8",     8,  1,  64),
]

# -- Kernel cache for Category A (direct kernel) --
_kernel_cache = {}


def _get_kernel(D, QH, KH, block_size, flash_layout, dtype_str,
                apply_scale=False, reuse_freqs_front_part=True, pos_dtype="i32",
                x_size=16):
    key = (D, QH, KH, block_size, flash_layout, dtype_str,
           apply_scale, reuse_freqs_front_part, pos_dtype, x_size)
    if key not in _kernel_cache:
        _kernel_cache[key] = build_fused_rope_cache_module(
            head_dim=D,
            num_q_heads=QH,
            num_kv_heads=KH,
            block_size=block_size,
            is_neox=True,
            flash_layout=flash_layout,
            dtype_str=dtype_str,
            apply_scale=apply_scale,
            reuse_freqs_front_part=reuse_freqs_front_part,
            pos_dtype=pos_dtype,
            x_size=x_size,
        )
    return _kernel_cache[key]


# ---------------------------------------------------------------------------
# Timing helper
# ---------------------------------------------------------------------------
def _bench_gpu_us(fn, warmup=10, iters=100):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1e6 / iters


# ---------------------------------------------------------------------------
# Category A / B core test runner (bf16)
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
        launch_fn = _get_kernel(D, QH, KH, block_size, flash_layout, dtype_str,
                                x_size=x_size)
        _dummy_scale = torch.ones(1, dtype=torch.float32, device=DEVICE)
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
            _dummy_scale,
            _dummy_scale,
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


# ---------------------------------------------------------------------------
# Category C: fp8 KV cache helper
# ---------------------------------------------------------------------------
def _make_fp8_caches(num_tokens, num_kv_heads, head_dim, num_blocks, block_size,
                     flash_layout):
    x_size = 16  # fp8: 16 elements per x-group = 16 bytes
    if flash_layout:
        key_cache = torch.zeros(
            num_blocks, block_size, num_kv_heads, head_dim,
            device=DEVICE, dtype=FP8_DTYPE,
        )
        value_cache = torch.zeros(
            num_blocks, block_size, num_kv_heads, head_dim,
            device=DEVICE, dtype=FP8_DTYPE,
        )
    else:
        key_cache = torch.zeros(
            num_blocks, num_kv_heads, head_dim // x_size, block_size, x_size,
            device=DEVICE, dtype=FP8_DTYPE,
        )
        value_cache = torch.zeros(
            num_blocks, num_kv_heads, head_dim, block_size,
            device=DEVICE, dtype=FP8_DTYPE,
        )
    return key_cache, value_cache


def run_fp8_test(num_tokens, head_dim=64, num_q_heads=8, num_kv_heads=1,
                 block_size=BLOCK_SIZE, max_pos=MAX_POS, flash_layout=True,
                 kv_scale=KV_SCALE):
    """Run FlyDSL fp8 rope kernel and compare against Triton.

    Returns: (ok, max_kc_err, max_vc_err, flydsl_us, triton_us)
    """
    layout_name = "flash" if flash_layout else "non-flash"
    print(f"\n[fp8] M={num_tokens}, BS={block_size}, QH={num_q_heads}, "
          f"KH={num_kv_heads}, D={head_dim}, layout={layout_name}, scale={kv_scale}")

    torch.manual_seed(42)
    q = torch.randn(num_tokens, num_q_heads, head_dim, device=DEVICE, dtype=torch.bfloat16)
    k = torch.randn(num_tokens, num_kv_heads, head_dim, device=DEVICE, dtype=torch.bfloat16)
    v = torch.randn(num_tokens, num_kv_heads, head_dim, device=DEVICE, dtype=torch.bfloat16)
    cos_cache = torch.randn(max_pos, head_dim // 2, device=DEVICE, dtype=torch.bfloat16)
    sin_cache = torch.randn(max_pos, head_dim // 2, device=DEVICE, dtype=torch.bfloat16)
    positions = torch.randint(0, max_pos, (num_tokens,), device=DEVICE, dtype=torch.int32)
    slot_mapping = torch.arange(num_tokens, device=DEVICE, dtype=torch.int32)

    num_blocks = max(32, (num_tokens + block_size - 1) // block_size + 4)
    k_scale = torch.tensor(kv_scale, dtype=torch.float32, device=DEVICE)
    v_scale = torch.tensor(kv_scale, dtype=torch.float32, device=DEVICE)

    # FlyDSL run
    kc_fly, vc_fly = _make_fp8_caches(num_tokens, num_kv_heads, head_dim, num_blocks,
                                       block_size, flash_layout)
    q_fly = torch.empty_like(q)
    k_fly = torch.empty_like(k)

    q_fly_out, k_fly_out, _, _ = flydsl_fused_qk_rope_reshape_and_cache(
        q, k, v, kc_fly, vc_fly,
        slot_mapping, positions,
        cos_cache, sin_cache,
        k_scale, v_scale,
        is_neox=True, flash_layout=flash_layout,
        apply_scale=True,
        offs=None, q_out=q_fly, k_out=k_fly,
        output_zeros=False,
    )
    torch.cuda.synchronize()

    # Triton reference run
    kc_tri, vc_tri = _make_fp8_caches(num_tokens, num_kv_heads, head_dim, num_blocks,
                                       block_size, flash_layout)
    q_tri = torch.empty_like(q)
    k_tri = torch.empty_like(k)
    cos_4d = cos_cache.unsqueeze(1).unsqueeze(1)
    sin_4d = sin_cache.unsqueeze(1).unsqueeze(1)
    pos_i64 = positions.to(torch.int64)
    slots_i64 = slot_mapping.to(torch.int64)

    fused_qk_rope_reshape_and_cache(
        q, k, v, kc_tri, vc_tri,
        slots_i64, pos_i64,
        cos_4d, sin_4d,
        k_scale, v_scale,
        is_neox=True, flash_layout=flash_layout,
        apply_scale=True,
        offs=None, q_out=q_tri, k_out=k_tri,
        output_zeros=False,
    )
    torch.cuda.synchronize()

    q_err = (q_fly_out.float() - q_tri.float()).abs().max().item()
    k_err = (k_fly_out.float() - k_tri.float()).abs().max().item()
    kc_err = (kc_fly.float() - kc_tri.float()).abs().max().item()
    vc_err = (vc_fly.float() - vc_tri.float()).abs().max().item()

    print(f"  vs Triton → q_err={q_err:.4f} k_err={k_err:.4f} "
          f"kc_err={kc_err:.4f} vc_err={vc_err:.4f}")

    flydsl_us = triton_us = 0.0
    if os.environ.get("FLYDSL_BENCH", "0") == "1":
        def run_fly():
            flydsl_fused_qk_rope_reshape_and_cache(
                q, k, v, kc_fly, vc_fly, slot_mapping, positions,
                cos_cache, sin_cache, k_scale, v_scale,
                is_neox=True, flash_layout=flash_layout, apply_scale=True,
                offs=None, q_out=q_fly, k_out=k_fly, output_zeros=False,
            )

        def run_tri():
            fused_qk_rope_reshape_and_cache(
                q, k, v, kc_tri, vc_tri, slots_i64, pos_i64,
                cos_4d, sin_4d, k_scale, v_scale,
                is_neox=True, flash_layout=flash_layout, apply_scale=True,
                offs=None, q_out=q_tri, k_out=k_tri, output_zeros=False,
            )

        flydsl_us = _bench_gpu_us(run_fly)
        triton_us = _bench_gpu_us(run_tri)
        speedup = triton_us / flydsl_us if flydsl_us > 0 else 0
        print(f"  FlyDSL: {flydsl_us:.1f} us  Triton: {triton_us:.1f} us  "
              f"speedup: {speedup:.2f}x")

    atol = kv_scale * 0.1 + 1e-3
    ok = q_err < 1e-2 and k_err < 1e-2 and kc_err < atol and vc_err < atol
    return ok, kc_err, vc_err, flydsl_us, triton_us


# ===========================================================================
# Category A: Direct FlyDSL kernel tests (flash, bf16, NEOX)
# ===========================================================================


@pytest.mark.skipif(not HAS_FLYDSL, reason="FlyDSL not installed")
@pytest.mark.parametrize("num_q_heads,num_kv_heads,head_dim", [
    (32,  8, 128),   # Llama-8B TP1
    (4,   1, 128),   # Llama-8B TP8
    (64,  8, 128),   # Llama-70B TP1
    (8,   1, 128),   # Llama-70B TP8
    (128, 8, 128),   # Llama-405B TP1
    (16,  1, 128),   # Llama-405B TP8
    (64,  4, 128),   # Qwen3-72B TP1
    (64,  8,  64),   # GPT-OSS TP1
    (8,   1,  64),   # GPT-OSS TP8
], ids=[
    "Llama8B-TP1", "Llama8B-TP8",
    "Llama70B-TP1", "Llama70B-TP8",
    "Llama405B-TP1", "Llama405B-TP8",
    "Qwen72B-TP1", "GPTOSS-TP1", "GPTOSS-TP8",
])
def test_flydsl_kernel_direct(num_q_heads, num_kv_heads, head_dim):
    """Category A: Direct FlyDSL kernel — T=1 decode, flash, bf16, NEOX."""
    QH_per_KH = num_q_heads // num_kv_heads
    run_test(
        T=1,
        QH_per_KH=QH_per_KH,
        KH=num_kv_heads,
        D=head_dim,
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
# Category C: fp8 KV cache quantization (apply_scale=True)
# ===========================================================================


@pytest.mark.skipif(not HAS_FLYDSL, reason="FlyDSL not installed")
@pytest.mark.skipif(not HAS_AITER_TRITON, reason="AITER Triton not available")
@pytest.mark.parametrize("num_tokens", [1, 4, 16, 32])
def test_fp8_flash(num_tokens):
    """Category C: fp8 KV cache, flash layout — FlyDSL vs Triton."""
    ok, kc_err, vc_err, _, _ = run_fp8_test(num_tokens, flash_layout=True)
    assert ok, f"FAILED flash M={num_tokens}: kc={kc_err:.4f} vc={vc_err:.4f}"


@pytest.mark.skipif(not HAS_FLYDSL, reason="FlyDSL not installed")
@pytest.mark.skipif(not HAS_AITER_TRITON, reason="AITER Triton not available")
@pytest.mark.parametrize("num_tokens", [1, 4, 16, 32])
def test_fp8_nonflash(num_tokens):
    """Category C: fp8 KV cache, non-flash layout — FlyDSL vs Triton."""
    ok, kc_err, vc_err, _, _ = run_fp8_test(num_tokens, flash_layout=False)
    assert ok, f"FAILED non-flash M={num_tokens}: kc={kc_err:.4f} vc={vc_err:.4f}"


@pytest.mark.skipif(not HAS_FLYDSL, reason="FlyDSL not installed")
@pytest.mark.skipif(not HAS_AITER_TRITON, reason="AITER Triton not available")
@pytest.mark.parametrize("flash_layout", [True, False], ids=["flash", "nonflash"])
def test_fp8_gpt_oss_tp8(flash_layout):
    """Category C: GPT-OSS 120B TP=8 exact config (QH=8, KH=1, D=64)."""
    ok, kc_err, vc_err, _, _ = run_fp8_test(
        num_tokens=1, head_dim=64, num_q_heads=8, num_kv_heads=1,
        flash_layout=flash_layout,
    )
    assert ok, f"FAILED GPT-OSS TP8: kc={kc_err:.4f} vc={vc_err:.4f}"


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


# ===========================================================================
# Category D: full-dim cos/sin (reuse_freqs_front_part=False)
# ===========================================================================


@pytest.mark.skipif(not HAS_FLYDSL, reason="FlyDSL not installed")
@pytest.mark.skipif(not HAS_AITER_TRITON, reason="AITER Triton not available")
@pytest.mark.parametrize("T", [1, 16, 128])
@pytest.mark.parametrize("QH,KH,D", [(8, 1, 64), (8, 1, 128), (64, 8, 128)])
def test_fulldim_cos_sin(T, QH, KH, D):
    """Category D: full-dim cos/sin [max_pos, D] — reuse_freqs_front_part=False."""
    torch.manual_seed(42)
    q = torch.randn(T, QH, D, device=DEVICE, dtype=torch.bfloat16)
    k = torch.randn(T, KH, D, device=DEVICE, dtype=torch.bfloat16)
    v = torch.randn_like(k)
    num_blocks = max(32, (T + BLOCK_SIZE - 1) // BLOCK_SIZE + 4)
    kc_fly = torch.zeros(num_blocks, BLOCK_SIZE, KH, D, device=DEVICE, dtype=torch.bfloat16)
    vc_fly = torch.zeros_like(kc_fly)
    kc_tri = torch.zeros_like(kc_fly)
    vc_tri = torch.zeros_like(vc_fly)
    cos = torch.randn(MAX_POS, D, device=DEVICE, dtype=torch.bfloat16)
    sin = torch.randn(MAX_POS, D, device=DEVICE, dtype=torch.bfloat16)
    pos = torch.randint(0, MAX_POS, (T,), device=DEVICE, dtype=torch.int32)
    slots = torch.arange(T, device=DEVICE, dtype=torch.int32)
    dummy = torch.ones(1, dtype=torch.float32, device=DEVICE)

    q_fly, k_fly, _, _ = flydsl_fused_qk_rope_reshape_and_cache(
        q, k, v, kc_fly, vc_fly, slots, pos, cos, sin, dummy, dummy,
        is_neox=True, flash_layout=True, apply_scale=False, output_zeros=False,
    )
    cos_4d = cos.unsqueeze(1).unsqueeze(1)
    sin_4d = sin.unsqueeze(1).unsqueeze(1)
    q_tri, k_tri, _, _ = fused_qk_rope_reshape_and_cache(
        q, k, v, kc_tri, vc_tri, slots.long(), pos.long(), cos_4d, sin_4d, dummy, dummy,
        is_neox=True, flash_layout=True, apply_scale=False, output_zeros=False,
    )
    torch.cuda.synchronize()
    torch.testing.assert_close(q_fly, q_tri, atol=ATOL, rtol=RTOL)
    torch.testing.assert_close(k_fly, k_tri, atol=ATOL, rtol=RTOL)
    torch.testing.assert_close(kc_fly, kc_tri, atol=ATOL, rtol=RTOL)
    torch.testing.assert_close(vc_fly, vc_tri, atol=ATOL, rtol=RTOL)


# ===========================================================================
# Category E: pos_dtype — i32 vs i64 (stride-2 view indexing)
# ===========================================================================


@pytest.mark.skipif(not HAS_FLYDSL, reason="FlyDSL not installed")
@pytest.mark.parametrize("pos_dtype", ["i32", "i64"])
@pytest.mark.parametrize("num_tokens", [1, 32, 128])
def test_pos_dtype(pos_dtype, num_tokens):
    """Category E: Position tensor dtype — i32 direct vs i64 stride-2 view.

    The i64 path reinterprets each int64 as two i32 words and reads only the
    low word (little-endian). Kernel uses stride-2 indexing when pos_dtype='i64'.
    """
    QH, KH, D = 8, 1, 128
    dtype = torch.bfloat16
    num_blocks = max(32, (num_tokens + BLOCK_SIZE - 1) // BLOCK_SIZE + 4)

    torch.manual_seed(42)
    q = torch.randn(num_tokens, QH, D, device=DEVICE, dtype=dtype)
    k = torch.randn(num_tokens, KH, D, device=DEVICE, dtype=dtype)
    v = torch.randn_like(k)
    cos_cache = torch.randn(MAX_POS, D // 2, device=DEVICE, dtype=dtype)
    sin_cache = torch.randn(MAX_POS, D // 2, device=DEVICE, dtype=dtype)
    positions_i32 = torch.randint(0, MAX_POS, (num_tokens,), device=DEVICE, dtype=torch.int32)
    slot_mapping = torch.arange(num_tokens, device=DEVICE, dtype=torch.int32)
    kc = torch.zeros(num_blocks, BLOCK_SIZE, KH, D, device=DEVICE, dtype=dtype)
    vc = torch.zeros_like(kc)

    if pos_dtype == "i64":
        positions_arg = positions_i32.to(torch.int64).view(torch.int32)
        slot_arg = slot_mapping.to(torch.int64).view(torch.int32)
    else:
        positions_arg = positions_i32
        slot_arg = slot_mapping

    launch_fn = _get_kernel(D, QH, KH, BLOCK_SIZE, True, "bf16",
                            apply_scale=False, reuse_freqs_front_part=True,
                            pos_dtype=pos_dtype)
    dummy = torch.ones(1, dtype=torch.float32, device=DEVICE)
    q_out = torch.empty_like(q)
    k_out = torch.empty_like(k)
    launch_fn(q, k, v, positions_arg, cos_cache, sin_cache, slot_arg,
              kc, vc, q_out, k_out, num_tokens, dummy, dummy,
              stream=torch.cuda.current_stream())
    torch.cuda.synchronize()

    # Reference: use i32 positions for indexing
    cos_ref = cos_cache[positions_i32.long()].unsqueeze(1).to(dtype)
    sin_ref = sin_cache[positions_i32.long()].unsqueeze(1).to(dtype)
    cos_ref = torch.cat([cos_ref, cos_ref], dim=-1)
    sin_ref = torch.cat([sin_ref, sin_ref], dim=-1)
    hd = D
    q1, q2 = q[..., :hd // 2], q[..., hd // 2:]
    k1, k2 = k[..., :hd // 2], k[..., hd // 2:]
    q_ref = torch.cat([q1 * cos_ref[..., :hd // 2] - q2 * sin_ref[..., :hd // 2],
                       q2 * cos_ref[..., hd // 2:] + q1 * sin_ref[..., hd // 2:]], dim=-1)
    k_ref = torch.cat([k1 * cos_ref[..., :hd // 2] - k2 * sin_ref[..., :hd // 2],
                       k2 * cos_ref[..., hd // 2:] + k1 * sin_ref[..., hd // 2:]], dim=-1)

    torch.testing.assert_close(q_out, q_ref, atol=ATOL, rtol=RTOL)
    torch.testing.assert_close(k_out, k_ref, atol=ATOL, rtol=RTOL)


# ===========================================================================
# Category F: negative slot mapping (slot < 0 → skip KV cache write)
# ===========================================================================


@pytest.mark.skipif(not HAS_FLYDSL, reason="FlyDSL not installed")
@pytest.mark.parametrize("num_tokens", [4, 32])
@pytest.mark.parametrize("flash_layout", [True, False], ids=["flash", "nonflash"])
def test_negative_slots(num_tokens, flash_layout):
    """Category F: Odd-indexed slots set to -1; those KV cache positions stay zero."""
    QH, KH, D = 8, 1, 128
    x_size = 16
    dtype = torch.bfloat16
    num_blocks = max(32, (num_tokens + BLOCK_SIZE - 1) // BLOCK_SIZE + 4)

    torch.manual_seed(7)
    q = torch.randn(num_tokens, QH, D, device=DEVICE, dtype=dtype)
    k = torch.randn(num_tokens, KH, D, device=DEVICE, dtype=dtype)
    v = torch.randn_like(k)
    cos_cache = torch.randn(MAX_POS, D // 2, device=DEVICE, dtype=dtype)
    sin_cache = torch.randn(MAX_POS, D // 2, device=DEVICE, dtype=dtype)
    positions = torch.randint(0, MAX_POS, (num_tokens,), device=DEVICE, dtype=torch.int32)
    slot_mapping = torch.arange(num_tokens, device=DEVICE, dtype=torch.int32)
    slot_mapping[1::2] = -1  # odd slots skipped

    if flash_layout:
        kc = torch.zeros(num_blocks, BLOCK_SIZE, KH, D, device=DEVICE, dtype=dtype)
        vc = torch.zeros_like(kc)
    else:
        kc = torch.zeros(num_blocks, KH, D // x_size, BLOCK_SIZE, x_size, device=DEVICE, dtype=dtype)
        vc = torch.zeros(num_blocks, KH, D, BLOCK_SIZE, device=DEVICE, dtype=dtype)

    kc_ref = kc.clone()
    vc_ref = vc.clone()

    launch_fn = _get_kernel(D, QH, KH, BLOCK_SIZE, flash_layout, "bf16",
                            apply_scale=False, reuse_freqs_front_part=True, pos_dtype="i32",
                            x_size=x_size)
    dummy = torch.ones(1, dtype=torch.float32, device=DEVICE)
    q_out = torch.empty_like(q)
    k_out = torch.empty_like(k)
    launch_fn(q, k, v, positions, cos_cache, sin_cache, slot_mapping,
              kc, vc, q_out, k_out, num_tokens, dummy, dummy,
              stream=torch.cuda.current_stream())
    torch.cuda.synchronize()

    # Reference: manually write only even-indexed slots
    cos_ref = torch.cat([cos_cache[positions.long()].unsqueeze(1).to(dtype)] * 2, dim=-1)
    sin_ref = torch.cat([sin_cache[positions.long()].unsqueeze(1).to(dtype)] * 2, dim=-1)
    hd = D
    q1, q2 = q[..., :hd // 2], q[..., hd // 2:]
    k1, k2 = k[..., :hd // 2], k[..., hd // 2:]
    k_rot = torch.cat([k1 * cos_ref[..., :hd // 2] - k2 * sin_ref[..., :hd // 2],
                       k2 * cos_ref[..., hd // 2:] + k1 * sin_ref[..., hd // 2:]], dim=-1)

    for i, slot in enumerate(slot_mapping.cpu().tolist()):
        if slot < 0:
            continue
        bi, bp = slot // BLOCK_SIZE, slot % BLOCK_SIZE
        if flash_layout:
            kc_ref[bi, bp] = k_rot[i]
            vc_ref[bi, bp] = v[i]
        else:
            kc_ref[bi, :, :, bp, :] = k_rot[i].view(KH, D // x_size, x_size)
            vc_ref[bi, :, :, bp] = v[i]

    torch.testing.assert_close(kc, kc_ref, atol=ATOL, rtol=RTOL)
    torch.testing.assert_close(vc, vc_ref, atol=ATOL, rtol=RTOL)


# ===========================================================================
# CLI: run all configs including fp8
# ===========================================================================

def run_bf16_bench(num_tokens, head_dim=128, num_q_heads=8, num_kv_heads=1,
                   block_size=BLOCK_SIZE, max_pos=MAX_POS, flash_layout=True,
                   dtype=torch.bfloat16):
    """Benchmark FlyDSL vs Triton for bf16 KV cache (no fp8 quantization)."""
    dtype_str = "bf16" if dtype == torch.bfloat16 else "f16"
    layout_name = "flash" if flash_layout else "non-flash"

    torch.manual_seed(42)
    q = torch.randn(num_tokens, num_q_heads, head_dim, device=DEVICE, dtype=dtype)
    k = torch.randn(num_tokens, num_kv_heads, head_dim, device=DEVICE, dtype=dtype)
    v = torch.randn(num_tokens, num_kv_heads, head_dim, device=DEVICE, dtype=dtype)
    cos_cache = torch.randn(max_pos, head_dim // 2, device=DEVICE, dtype=dtype)
    sin_cache = torch.randn(max_pos, head_dim // 2, device=DEVICE, dtype=dtype)
    positions = torch.randint(0, max_pos, (num_tokens,), device=DEVICE, dtype=torch.int32)
    slot_mapping = torch.arange(num_tokens, device=DEVICE, dtype=torch.int32)

    num_blocks = max(32, (num_tokens + block_size - 1) // block_size + 4)
    if flash_layout:
        kc = torch.zeros(num_blocks, block_size, num_kv_heads, head_dim, device=DEVICE, dtype=dtype)
        vc = torch.zeros(num_blocks, block_size, num_kv_heads, head_dim, device=DEVICE, dtype=dtype)
    else:
        kc = torch.zeros(num_blocks, num_kv_heads, head_dim // X_SIZE, block_size, X_SIZE, device=DEVICE, dtype=dtype)
        vc = torch.zeros(num_blocks, num_kv_heads, head_dim, block_size, device=DEVICE, dtype=dtype)
    q_out = torch.empty_like(q)
    k_out = torch.empty_like(k)

    cos_4d = cos_cache.unsqueeze(1).unsqueeze(1)
    sin_4d = sin_cache.unsqueeze(1).unsqueeze(1)
    pos_i64 = positions.to(torch.int64)
    slots_i64 = slot_mapping.to(torch.int64)
    _dummy = torch.ones(1, dtype=torch.float32, device=DEVICE)

    def run_fly():
        flydsl_fused_qk_rope_reshape_and_cache(
            q, k, v, kc, vc, slot_mapping, positions,
            cos_cache, sin_cache, _dummy, _dummy,
            is_neox=True, flash_layout=flash_layout, apply_scale=False,
            offs=None, q_out=q_out, k_out=k_out, output_zeros=False,
        )

    def run_tri():
        fused_qk_rope_reshape_and_cache(
            q, k, v, kc, vc, slots_i64, pos_i64,
            cos_4d, sin_4d, _dummy, _dummy,
            is_neox=True, flash_layout=flash_layout, apply_scale=False,
            offs=None, q_out=q_out, k_out=k_out, output_zeros=False,
        )

    # Correctness check
    run_fly(); torch.cuda.synchronize()
    q_fly = q_out.clone(); k_fly = k_out.clone()
    run_tri(); torch.cuda.synchronize()
    q_tri = q_out.clone(); k_tri = k_out.clone()
    q_err = (q_fly.float() - q_tri.float()).abs().max().item()
    k_err = (k_fly.float() - k_tri.float()).abs().max().item()
    ok = q_err < 1e-1 and k_err < 1e-1

    fly_us = _bench_gpu_us(run_fly)
    tri_us = _bench_gpu_us(run_tri)
    speedup = tri_us / fly_us if fly_us > 0 else 0

    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {layout_name:>9s} M={num_tokens:>4d} {dtype_str}  "
          f"q_err={q_err:.4f} k_err={k_err:.4f}  "
          f"FlyDSL={fly_us:.1f}us Triton={tri_us:.1f}us  speedup={speedup:.2f}x")
    return ok, fly_us, tri_us


if __name__ == "__main__":
    import sys

    if not torch.cuda.is_available():
        print("CUDA/ROCm not available.")
        sys.exit(1)

    print("=" * 60)
    print("FlyDSL RoPE test — bf16 (Category A direct)")
    print("=" * 60)
    for D in [64, 128]:
        for KH in [1, 8]:
            for QH_per_KH in [1, 8]:
                run_test(T=1, QH_per_KH=QH_per_KH, KH=KH, D=D,
                         reuse_freqs_front_part=True, flash_layout=True,
                         use_wrapper=False)
                print(f"  PASS  D={D} KH={KH} QH/KH={QH_per_KH} flash T=1")

    print()
    print("=" * 60)
    print("FlyDSL RoPE benchmark — bf16 KV cache vs Triton")
    print("=" * 60)
    bf16_configs = [
        ("GPT-OSS-120B-TP8", 64, 8, 1),
        ("GPT-OSS-120B-TP1", 64, 64, 8),
        ("Llama-70B-TP8", 128, 8, 1),
        ("Llama-405B-TP8", 128, 16, 1),
    ]
    token_counts_bf16 = [1, 4, 16, 32, 128]
    for model, hd, qh, kh in bf16_configs:
        print(f"\n  {model}: QH={qh}, KH={kh}, D={hd}")
        for flash_layout in [True, False]:
            for m in token_counts_bf16:
                if not HAS_AITER_TRITON:
                    print("  [SKIP] AITER Triton not available")
                    break
                run_bf16_bench(m, head_dim=hd, num_q_heads=qh, num_kv_heads=kh,
                               flash_layout=flash_layout)

    print()
    print("=" * 60)
    print("FlyDSL RoPE test — fp8 KV cache (Category C)")
    print("=" * 60)
    configs = [
        ("GPT-OSS-120B-TP8", 64, 8, 1),
        ("GPT-OSS-120B-TP1", 64, 64, 8),
    ]
    token_counts = [1, 4, 16, 32, 128]

    for model, hd, qh, kh in configs:
        print(f"\n{'='*60}")
        print(f"{model}: QH={qh}, KH={kh}, D={hd}")
        print(f"{'='*60}")
        for flash_layout in [True, False]:
            layout = "flash" if flash_layout else "non-flash"
            for m in token_counts:
                ok, kc_err, vc_err, fly_us, tri_us = run_fp8_test(
                    m, head_dim=hd, num_q_heads=qh, num_kv_heads=kh,
                    flash_layout=flash_layout,
                )
                status = "PASS" if ok else "FAIL"
                bench = f"  FlyDSL={fly_us:.1f}us Triton={tri_us:.1f}us" if fly_us > 0 else ""
                print(f"  [{status}] {layout:>9s} M={m:>4d} "
                      f"kc={kc_err:.4f} vc={vc_err:.4f}{bench}")
    print("\nDone.")
