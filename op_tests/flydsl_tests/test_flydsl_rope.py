#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for FlyDSL fused RoPE + KV Cache kernel.

Tests correctness against PyTorch reference implementation and optionally
cross-checks against AITER Triton fused_qk_rope_reshape_and_cache.

Usage:
    # Quick CI (6 configs):
    pytest op_tests/flydsl_tests/test_flydsl_rope.py -v -s

    # Full model sweep:
    FLYDSL_ALL_MODELS=1 pytest op_tests/flydsl_tests/test_flydsl_rope.py -v -s

    # With AITER cross-check + benchmark:
    FLYDSL_BENCH=1 pytest op_tests/flydsl_tests/test_flydsl_rope.py -v -s
"""

import os

import pytest
import torch

# ── FlyDSL availability ──
try:
    from aiter.ops.flydsl.utils import is_flydsl_available

    HAS_FLYDSL = is_flydsl_available()
except ImportError:
    HAS_FLYDSL = False

if HAS_FLYDSL:
    import sys

    # FlyDSL kernels are in the FlyDSL repo, not in aiter
    flydsl_root = os.environ.get("FLYDSL_ROOT", "/home/pensun/FlyDSL")
    if flydsl_root not in sys.path:
        sys.path.insert(0, flydsl_root)
    from kernels.fused_rope_cache_kernel import build_fused_rope_cache_module

# ── AITER Triton availability ──
try:
    from aiter.ops.triton.fusions.fused_kv_cache import fused_qk_rope_reshape_and_cache

    HAS_AITER_TRITON = True
except ImportError:
    HAS_AITER_TRITON = False

# ── Test configuration ──
DEVICE = "cuda:0"
BLOCK_SIZE = 16
MAX_POS = 8192
BF16_ATOL = 1e-2
F16_ATOL = 5e-3
CROSS_ATOL = 1e-2

# ── Model configs ──
# (name, QH, KH, D)
QUICK_MODELS = [
    ("Llama-70B-TP8", 8, 1, 128),
    ("Llama-405B-TP1", 128, 8, 128),
    ("GPT-OSS-TP1", 64, 8, 64),
]

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
    ("GPT-OSS-TP1", 64, 8, 64),
    ("GPT-OSS-TP8", 8, 1, 64),
]

MODELS = ALL_MODELS if os.environ.get("FLYDSL_ALL_MODELS", "0") == "1" else QUICK_MODELS

# ── Cache for compiled kernels ──
_kernel_cache = {}


def _get_kernel(D, QH, KH, block_size, flash_layout, dtype_str, pos_int64):
    key = (D, QH, KH, block_size, flash_layout, dtype_str, pos_int64)
    if key not in _kernel_cache:
        _kernel_cache[key] = build_fused_rope_cache_module(
            head_dim=D,
            num_q_heads=QH,
            num_kv_heads=KH,
            block_size=block_size,
            is_neox=True,
            flash_layout=flash_layout,
            dtype_str=dtype_str,
            pos_int64=pos_int64,
        )
    return _kernel_cache[key]


def pytorch_rope_ref(q, k, cos_2d, sin_2d, positions, D):
    """NeoX-style RoPE reference in pure PyTorch."""
    half = D // 2
    cos = cos_2d[positions.long()].unsqueeze(1)
    sin = sin_2d[positions.long()].unsqueeze(1)
    q1, q2 = q[..., :half], q[..., half:]
    k1, k2 = k[..., :half], k[..., half:]
    q_out = torch.cat([q1 * cos - q2 * sin, q2 * cos + q1 * sin], dim=-1)
    k_out = torch.cat([k1 * cos - k2 * sin, k2 * cos + q1 * sin], dim=-1)
    return q_out, k_out


def run_test(T, QH, KH, D, flash_layout=True, dtype_str="bf16", pos_int64=False):
    """Run FlyDSL kernel and validate against PyTorch reference."""
    torch_dtype = torch.bfloat16 if dtype_str == "bf16" else torch.float16
    atol = BF16_ATOL if dtype_str == "bf16" else F16_ATOL

    # Inputs
    q = torch.randn(T, QH, D, dtype=torch_dtype, device=DEVICE)
    k = torch.randn(T, KH, D, dtype=torch_dtype, device=DEVICE)
    v = torch.randn(T, KH, D, dtype=torch_dtype, device=DEVICE)

    pos_dtype = torch.int64 if pos_int64 else torch.int32
    positions = torch.randint(
        0, min(MAX_POS, 4096), (T,), dtype=pos_dtype, device=DEVICE
    )
    slot_mapping = torch.arange(T, dtype=pos_dtype, device=DEVICE)

    half = D // 2
    cos_cache = torch.randn(MAX_POS, half, dtype=torch_dtype, device=DEVICE)
    sin_cache = torch.randn(MAX_POS, half, dtype=torch_dtype, device=DEVICE)

    num_blocks = max(32, (T + BLOCK_SIZE - 1) // BLOCK_SIZE + 1)
    if flash_layout:
        key_cache = torch.zeros(
            num_blocks, BLOCK_SIZE, KH, D, dtype=torch_dtype, device=DEVICE
        )
        value_cache = torch.zeros(
            num_blocks, BLOCK_SIZE, KH, D, dtype=torch_dtype, device=DEVICE
        )
    else:
        x_pack = 16
        key_cache = torch.zeros(
            num_blocks,
            KH,
            D // x_pack,
            BLOCK_SIZE,
            x_pack,
            dtype=torch_dtype,
            device=DEVICE,
        )
        value_cache = torch.zeros(
            num_blocks, KH, D, BLOCK_SIZE, dtype=torch_dtype, device=DEVICE
        )

    q_out = torch.empty_like(q)
    k_out = torch.empty_like(k)

    # Run FlyDSL kernel
    launch_fn = _get_kernel(D, QH, KH, BLOCK_SIZE, flash_layout, dtype_str, pos_int64)
    stream = torch.cuda.current_stream()
    launch_fn(
        q,
        k,
        v,
        positions,
        cos_cache,
        sin_cache,
        slot_mapping,
        key_cache,
        value_cache,
        q_out,
        k_out,
        T,
        stream=stream,
    )
    torch.cuda.synchronize()

    # PyTorch reference
    q_ref, k_ref = pytorch_rope_ref(q, k, cos_cache, sin_cache, positions, D)

    q_err = (q_out.float() - q_ref.float()).abs().max().item()
    k_err = (k_out.float() - k_ref.float()).abs().max().item()

    assert q_err < atol, f"Q error {q_err:.2e} >= {atol}"
    assert k_err < atol, f"K error {k_err:.2e} >= {atol}"

    # Optional AITER cross-check
    if HAS_AITER_TRITON and os.environ.get("FLYDSL_BENCH", "0") == "1":
        cos_4d = cos_cache.unsqueeze(1).unsqueeze(1)
        sin_4d = sin_cache.unsqueeze(1).unsqueeze(1)
        pos_i64 = positions.to(torch.int64)
        slot_i64 = slot_mapping.to(torch.int64)
        ks = torch.tensor(1.0, device=DEVICE)
        vs = torch.tensor(1.0, device=DEVICE)
        kc_aiter = key_cache.clone() if flash_layout else torch.zeros_like(key_cache)
        vc_aiter = (
            value_cache.clone() if flash_layout else torch.zeros_like(value_cache)
        )
        qa = torch.empty_like(q)
        ka = torch.empty_like(k)

        fused_qk_rope_reshape_and_cache(
            q.clone(),
            k.clone(),
            v,
            kc_aiter,
            vc_aiter,
            slot_i64,
            pos_i64,
            cos_4d,
            sin_4d,
            ks,
            vs,
            is_neox=True,
            flash_layout=flash_layout,
            apply_scale=False,
            q_out=qa,
            k_out=ka,
            output_zeros=False,
        )
        torch.cuda.synchronize()

        q_cross = (q_out.float() - qa.float()).abs().max().item()
        k_cross = (k_out.float() - ka.float()).abs().max().item()
        cross_ok = q_cross < CROSS_ATOL and k_cross < CROSS_ATOL
        status = "MATCH" if cross_ok else "MISMATCH"
        print(f"  AITER cross-check: {status} Q={q_cross:.2e} K={k_cross:.2e}")
        assert cross_ok, f"Cross-check failed: Q={q_cross:.2e} K={k_cross:.2e}"

    return q_err, k_err


# ── Tests ──


@pytest.mark.skipif(not HAS_FLYDSL, reason="FlyDSL not installed")
@pytest.mark.parametrize("num_tokens", [1, 4, 32, 128])
def test_flydsl_rope_flash_int32(num_tokens):
    """Basic correctness with int32 positions, flash layout."""
    run_test(num_tokens, QH=64, KH=8, D=128, flash_layout=True, pos_int64=False)


@pytest.mark.skipif(not HAS_FLYDSL, reason="FlyDSL not installed")
@pytest.mark.parametrize("num_tokens", [1, 4, 32, 128])
def test_flydsl_rope_flash_int64(num_tokens):
    """AITER-compatible: int64 positions, flash layout."""
    run_test(num_tokens, QH=64, KH=8, D=128, flash_layout=True, pos_int64=True)


@pytest.mark.skipif(not HAS_FLYDSL, reason="FlyDSL not installed")
@pytest.mark.parametrize("num_tokens", [1, 32])
def test_flydsl_rope_nonflash(num_tokens):
    """Non-flash (ATOM) cache layout."""
    run_test(num_tokens, QH=64, KH=8, D=128, flash_layout=False, pos_int64=True)


@pytest.mark.skipif(not HAS_FLYDSL, reason="FlyDSL not installed")
@pytest.mark.parametrize("num_tokens", [1, 32])
def test_flydsl_rope_f16(num_tokens):
    """FP16 dtype."""
    run_test(
        num_tokens,
        QH=64,
        KH=8,
        D=128,
        flash_layout=True,
        dtype_str="f16",
        pos_int64=True,
    )


@pytest.mark.skipif(not HAS_FLYDSL, reason="FlyDSL not installed")
@pytest.mark.parametrize("num_tokens", [1, 32])
def test_flydsl_rope_d64(num_tokens):
    """head_dim=64 (GPT-OSS, Qwen3-235B)."""
    run_test(num_tokens, QH=64, KH=8, D=64, flash_layout=True, pos_int64=True)


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
@pytest.mark.parametrize("num_tokens", [1, 32, 128])
def test_flydsl_rope_multi_model(model, QH, KH, D, num_tokens):
    """Full model sweep with AITER-compatible int64 inputs."""
    q_err, k_err = run_test(num_tokens, QH, KH, D, flash_layout=True, pos_int64=True)
    print(f"  [{model}] T={num_tokens} Q_err={q_err:.2e} K_err={k_err:.2e}")
