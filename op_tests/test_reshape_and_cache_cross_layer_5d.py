# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Tests for Tencent Cross-Layer 5D KV Cache support in
``aiter.reshape_and_cache_flash`` / ``aiter.reshape_and_cache_flash_func``.

Verifies that the writer-side kernel correctly handles the non-contiguous
``[NumBlocks, NumKVHeads, PageSize, HeadDim]`` per-layer view that the
framework slices out of the 6D physical KV buffer
``(NumBlocks, NumKVHeads, NumLayers, 2, PageSize, HeadDim)``.

Companion to ``test_batch_prefill.py::test_batch_prefill_cross_layer_5d_*``
which covers the reader side.
"""

import pytest
import torch

import aiter


def _make_cross_layer_5d_buffer(
    num_blocks: int,
    num_kv_heads: int,
    num_layers: int,
    page_size: int,
    head_dim: int,
    layer_idx: int,
    dtype: torch.dtype,
    device: str = "cuda",
):
    """Allocate a 6D physical KV buffer and return both the buffer and the
    per-layer 5D non-contiguous view ``(2, N, H, B, D)`` for ``layer_idx``.

    Mirrors the construction in
    ``Cross-Layer_5D_KV_Cache_Operator_Adaptation_Plan_EN.md``:

      1. Allocate contiguous 6D `(N, H, L, 2, B, D)`.
      2. Permute to logical 6D `(L, 2, N, H, B, D)` (non-contiguous view).
      3. Index by ``layer_idx`` to get the 5D `(2, N, H, B, D)` per-layer view.
    """
    physical = torch.zeros(
        num_blocks,
        num_kv_heads,
        num_layers,
        2,
        page_size,
        head_dim,
        dtype=dtype,
        device=device,
    )
    logical6d = physical.permute(2, 3, 0, 1, 4, 5)  # (L, 2, N, H, B, D)
    return physical, logical6d[layer_idx]


@pytest.mark.parametrize("num_layers", [4, 80])
@pytest.mark.parametrize("layer_idx", [0, 1])
@pytest.mark.parametrize("num_blocks", [4])
@pytest.mark.parametrize("page_size", [16])
@pytest.mark.parametrize("num_kv_heads", [1, 2])
@pytest.mark.parametrize("head_dim", [128])
def test_reshape_and_cache_flash_cross_layer_5d_strides(
    num_layers,
    layer_idx,
    num_blocks,
    page_size,
    num_kv_heads,
    head_dim,
):
    """Metadata-only test: confirm the cross-layer per-layer 5D view (and its
    K/V slices) have exactly the strides documented in section 5.3 of
    ``Cross-Layer_5D_KV_Cache_Operator_Adaptation_Plan_EN.md``."""
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA/HIP device")

    _, kv_view = _make_cross_layer_5d_buffer(
        num_blocks=num_blocks,
        num_kv_heads=num_kv_heads,
        num_layers=num_layers,
        page_size=page_size,
        head_dim=head_dim,
        layer_idx=layer_idx,
        dtype=torch.bfloat16,
    )
    B, D, H, L = page_size, head_dim, num_kv_heads, num_layers

    expected_5d = (B * D, L * 2 * H * B * D, L * 2 * B * D, D, 1)
    assert kv_view.shape == (2, num_blocks, H, B, D)
    assert not kv_view.is_contiguous()
    assert kv_view.stride() == expected_5d, (
        f"5D view stride mismatch: got {kv_view.stride()} expected {expected_5d}"
    )

    expected_4d = (L * 2 * H * B * D, L * 2 * B * D, D, 1)
    k_cache = kv_view[0]
    v_cache = kv_view[1]
    assert k_cache.stride() == expected_4d
    assert v_cache.stride() == expected_4d
    assert k_cache.stride(-1) == 1
    assert v_cache.stride(-1) == 1


def test_reshape_and_cache_flash_func_rejects_layout_mismatch():
    """`reshape_and_cache_flash_func` must reject inconsistent input combos."""
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA/HIP device")

    num_blocks, num_kv_heads, num_layers, page_size, head_dim = 4, 1, 4, 16, 128
    dtype = torch.bfloat16
    device = "cuda"

    _, kv_view = _make_cross_layer_5d_buffer(
        num_blocks=num_blocks,
        num_kv_heads=num_kv_heads,
        num_layers=num_layers,
        page_size=page_size,
        head_dim=head_dim,
        layer_idx=0,
        dtype=dtype,
    )
    num_tokens = 8
    key = torch.zeros(num_tokens, num_kv_heads, head_dim, dtype=dtype, device=device)
    value = torch.zeros_like(key)
    slot_mapping = torch.arange(num_tokens, dtype=torch.int64, device=device)

    with pytest.raises(ValueError, match="not both"):
        aiter.reshape_and_cache_flash_func(
            key,
            value,
            key_cache=kv_view[0],
            value_cache=kv_view[1],
            slot_mapping=slot_mapping,
            kv_cache=kv_view,
        )

    with pytest.raises(ValueError, match="kv_layout"):
        aiter.reshape_and_cache_flash_func(
            key,
            value,
            slot_mapping=slot_mapping,
            kv_cache=kv_view,
            kv_layout=aiter.KV_LAYOUT_LINEAR,
        )

    with pytest.raises(ValueError, match="must pass"):
        aiter.reshape_and_cache_flash_func(
            key,
            value,
            slot_mapping=slot_mapping,
        )


@pytest.mark.parametrize("num_layers", [4])
@pytest.mark.parametrize("layer_idx", [0, 2])
@pytest.mark.parametrize("num_blocks", [4])
@pytest.mark.parametrize("page_size", [16])
@pytest.mark.parametrize("num_kv_heads", [1, 2])
@pytest.mark.parametrize("head_dim", [128])
@pytest.mark.parametrize("num_tokens", [16, 33])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_reshape_and_cache_flash_cross_layer_5d_matches_packed(
    num_layers,
    layer_idx,
    num_blocks,
    page_size,
    num_kv_heads,
    head_dim,
    num_tokens,
    dtype,
):
    """Functional test for Tencent Cross-Layer 5D KV cache writes.

    Runs `reshape_and_cache_flash` twice with identical input K/V tokens:

      A) packed `[N, B, H, D]` reference key_cache/value_cache (legacy path)
      B) cross-layer non-contiguous `[N, H, B, D]` view sliced from the 6D
         physical buffer (LINEAR_HEADS_FIRST path)

    After both runs, the actual K/V data written for each slot must match
    between the two layouts. The cross-layer cache stores data with the
    ``[N, H, B, D]`` axis order, so we transpose `(0, 2, 1, 3)` to compare
    element-by-element against the packed `[N, B, H, D]` reference.
    """
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA/HIP device")
    assert num_tokens <= num_blocks * page_size, (
        "slot indices must stay within the cache"
    )

    torch.manual_seed(0xCAFE05D)
    device = "cuda"

    # Build random input K/V tokens.
    key = torch.randn(num_tokens, num_kv_heads, head_dim, dtype=dtype, device=device)
    value = torch.randn_like(key)

    # Slot mapping: pack tokens densely into the first `num_tokens` slots.
    slot_mapping = torch.arange(num_tokens, dtype=torch.int64, device=device)

    # ---- Path A: packed reference ------------------------------------------
    k_cache_packed = torch.zeros(
        num_blocks, page_size, num_kv_heads, head_dim, dtype=dtype, device=device
    )
    v_cache_packed = torch.zeros_like(k_cache_packed)
    aiter.reshape_and_cache_flash_func(
        key,
        value,
        key_cache=k_cache_packed,
        value_cache=v_cache_packed,
        slot_mapping=slot_mapping,
        kv_cache_dtype="auto",
        kv_layout=aiter.KV_LAYOUT_LINEAR,
    )

    # ---- Path B: cross-layer 5D non-contiguous view ------------------------
    physical, kv_view = _make_cross_layer_5d_buffer(
        num_blocks=num_blocks,
        num_kv_heads=num_kv_heads,
        num_layers=num_layers,
        page_size=page_size,
        head_dim=head_dim,
        layer_idx=layer_idx,
        dtype=dtype,
    )
    aiter.reshape_and_cache_flash_func(
        key,
        value,
        slot_mapping=slot_mapping,
        kv_cache_dtype="auto",
        kv_cache=kv_view,
    )

    # The cross-layer K cache stores in [N, H, B, D] order; reorder to [N, B, H, D]
    # to compare against the packed reference.
    k_xlayer = kv_view[0].permute(0, 2, 1, 3).contiguous()
    v_xlayer = kv_view[1].permute(0, 2, 1, 3).contiguous()

    torch.testing.assert_close(k_xlayer, k_cache_packed, rtol=0, atol=0)
    torch.testing.assert_close(v_xlayer, v_cache_packed, rtol=0, atol=0)

    # Cross-talk check: no OTHER (kv, layer) slot should have been touched in
    # the 6D physical buffer beyond `(layer_idx, kv_idx=0/1)`. Sum-of-absolute
    # values outside the layer-of-interest must be exactly zero (we
    # zero-initialized the buffer above).
    untouched_mask = torch.ones(num_layers, dtype=torch.bool, device=device)
    untouched_mask[layer_idx] = False
    # Reorder physical from (N, H, L, 2, B, D) -> (L, ...) and index untouched layers
    other_layers = physical.permute(2, 3, 0, 1, 4, 5)[untouched_mask]
    assert other_layers.abs().sum().item() == 0.0, (
        "Cross-layer write leaked into other layers"
    )


def test_reshape_and_cache_flash_legacy_path_unchanged():
    """Regression: the legacy default code path (no kv_layout argument) must
    still work bit-exactly. Confirms the kernel signature change is
    backward-compatible for existing callers."""
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA/HIP device")

    torch.manual_seed(0xDEFA017)
    device = "cuda"
    num_blocks, page_size, num_kv_heads, head_dim = 4, 16, 1, 128
    num_tokens = 16
    dtype = torch.bfloat16

    key = torch.randn(num_tokens, num_kv_heads, head_dim, dtype=dtype, device=device)
    value = torch.randn_like(key)
    slot_mapping = torch.arange(num_tokens, dtype=torch.int64, device=device)
    k_cache = torch.zeros(
        num_blocks, page_size, num_kv_heads, head_dim, dtype=dtype, device=device
    )
    v_cache = torch.zeros_like(k_cache)

    k_scale = torch.tensor([1.0], dtype=torch.float32, device=device)
    v_scale = torch.tensor([1.0], dtype=torch.float32, device=device)

    # Call the raw @compile_ops-decorated function the way existing callers do
    # (no kv_layout kwarg). The default in the binding must keep the packed
    # fast path active.
    aiter.reshape_and_cache_flash(
        key, value, k_cache, v_cache, slot_mapping, "auto", k_scale, v_scale
    )

    # Verify the slot writes by reading them back via slot_mapping math.
    for tok_idx in range(num_tokens):
        slot = slot_mapping[tok_idx].item()
        block_idx = slot // page_size
        block_offset = slot % page_size
        torch.testing.assert_close(
            k_cache[block_idx, block_offset], key[tok_idx], rtol=0, atol=0
        )
        torch.testing.assert_close(
            v_cache[block_idx, block_offset], value[tok_idx], rtol=0, atol=0
        )
