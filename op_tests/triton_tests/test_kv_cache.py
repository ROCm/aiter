import torch
import pytest

from op_tests.triton_tests.attention.test_mla import shuffle_kv_buffer
from aiter.ops.triton.kv_cache import cat_and_cache_mla
from aiter.ops.triton.utils.types import e4m3_dtype


@pytest.mark.parametrize("T", [1, 2, 4, 2048])
@pytest.mark.parametrize("KH", [1, 8])
@pytest.mark.parametrize("D_pe", [64])  # For now, D is power of 2. D >= 16
@pytest.mark.parametrize("D_lora", [512])
@pytest.mark.parametrize("num_kv_cahce_tokens", [16384])
@pytest.mark.parametrize("cache_dtype", [torch.bfloat16, torch.uint8])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("shuffled_kv_cache, block_size", [(True, 64), (False, 1)])
def test_fused_qk_rope_cat_and_cache_mla(
    T: int,
    KH: int,
    D_pe: int,
    D_lora: int,
    num_kv_cahce_tokens: int,
    cache_dtype: bool,
    dtype: torch.dtype,
    shuffled_kv_cache: bool,
    block_size: int,
):
    k_lora = torch.randn((T, KH, D_lora), dtype=dtype, device="cuda") / (
        20 if cache_dtype == torch.uint8 else 1
    )
    k_pe = torch.randn((T, KH, D_pe), dtype=dtype, device="cuda") / (
        20 if cache_dtype == torch.uint8 else 1
    )

    if cache_dtype == torch.uint8:
        cache_dtype_actual = e4m3_dtype

    kv_cache = torch.zeros(
        (num_kv_cahce_tokens, KH, D_lora + D_pe), dtype=cache_dtype, device="cuda"
    )

    if cache_dtype == torch.uint8:
        k_scale = torch.randn(
            [
                1,
            ],
            dtype=torch.float32,
            device="cuda",
        )[0]
    else:
        k_scale = torch.ones(
            [
                1,
            ],
            dtype=torch.float32,
            device="cuda",
        )[0]
    slot_mapping = torch.randperm(T, device="cuda")
    kv_cache_og_dtype = kv_cache.dtype

    torch_k_lora = k_lora
    torch_k_pe = k_pe

    torch_kv_cache = kv_cache.clone()
    if cache_dtype == torch.uint8:
        torch_kv_cache = torch_kv_cache.view(cache_dtype_actual)
        torch_k_lora = (torch_k_lora.to(torch.float32) / k_scale).to(cache_dtype_actual)
        torch_k_pe = (torch_k_pe.to(torch.float32) / k_scale).to(cache_dtype_actual)
    else:
        torch_k_lora = torch_k_lora
        torch_k_pe = torch_k_pe

    torch_kv_cache[slot_mapping, :, :] = torch.cat((torch_k_lora, torch_k_pe), dim=-1)
    torch_kv_cache = torch_kv_cache.view(kv_cache_og_dtype)

    triton_kv_cache = kv_cache.clone()
    if cache_dtype == torch.uint8:
        triton_kv_cache = triton_kv_cache.view(cache_dtype_actual)
    if shuffled_kv_cache:
        triton_kv_cache = triton_kv_cache.view(
            num_kv_cahce_tokens // block_size, KH, block_size, D_lora + D_pe
        )
    cat_and_cache_mla(
        k_lora,
        k_pe,
        triton_kv_cache,
        slot_mapping,
        k_scale,
        shuffled_kv_cache=shuffled_kv_cache,
    )
    triton_kv_cache = triton_kv_cache.view(kv_cache_og_dtype)

    if shuffled_kv_cache:
        if cache_dtype == torch.uint8:
            torch_kv_cache = torch_kv_cache.view(cache_dtype_actual)
        torch_kv_cache = shuffle_kv_buffer(
            torch_kv_cache.reshape(
                num_kv_cahce_tokens // block_size, block_size, KH, D_lora + D_pe
            ),
            D_lora,
        )

    if cache_dtype == torch.uint8:
        torch_kv_cache = torch_kv_cache.view(cache_dtype_actual).to(dtype)
        triton_kv_cache = triton_kv_cache.view(cache_dtype_actual).to(dtype)

    if shuffled_kv_cache:
        torch.testing.assert_close(
            torch_kv_cache[slot_mapping // block_size, :, :],
            triton_kv_cache[slot_mapping // block_size, :, :],
            atol=1e-1,
            rtol=1e-1,
        )
    else:
        torch.testing.assert_close(
            torch_kv_cache[slot_mapping, :, :],
            triton_kv_cache[slot_mapping, :, :],
            atol=1e-1,
            rtol=1e-1,
        )

    torch.testing.assert_close(torch_kv_cache, triton_kv_cache, atol=1e-1, rtol=1e-1)
import pytest
import torch

from aiter.ops.triton.kv_cache import reshape_and_cache


def _torch_reference(key, value, key_cache, value_cache, slot_mapping, block_size):
    expected_k = key_cache.clone()
    expected_v = value_cache.clone()
    for tok_idx in range(slot_mapping.shape[0]):
        slot = int(slot_mapping[tok_idx].item())
        if slot < 0:
            continue
        block_id = slot // block_size
        within = slot % block_size
        expected_k[block_id, :, within, :] = key[tok_idx]
        expected_v[block_id, :, within, :] = value[tok_idx]
    return expected_k, expected_v


@pytest.mark.parametrize("num_tokens", [1, 4, 256])
@pytest.mark.parametrize("num_kv_heads", [1, 8])
@pytest.mark.parametrize("block_size", [16, 64])
@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("skip_fraction", [0.0, 0.5])
def test_reshape_and_cache(
    num_tokens, num_kv_heads, block_size, head_dim, skip_fraction
):
    torch.manual_seed(0)
    num_blocks = 32
    dtype = torch.bfloat16
    device = "cuda"

    key = torch.randn(num_tokens, num_kv_heads, head_dim, dtype=dtype, device=device)
    value = torch.randn(num_tokens, num_kv_heads, head_dim, dtype=dtype, device=device)

    key_cache = torch.randn(
        num_blocks, num_kv_heads, block_size, head_dim, dtype=dtype, device=device
    )
    value_cache = torch.randn(
        num_blocks, num_kv_heads, block_size, head_dim, dtype=dtype, device=device
    )

    # Use a permutation (not randint) so slots are unique - duplicate
    # slots race in the parallel kernel; serial reference cannot model that.
    perm = torch.randperm(num_blocks * block_size, device=device)
    slot_mapping = perm[:num_tokens].to(torch.int64)
    if skip_fraction > 0.0:
        num_skip = max(1, int(num_tokens * skip_fraction))
        skip_idx = torch.randperm(num_tokens, device=device)[:num_skip]
        slot_mapping[skip_idx] = -1

    expected_k, expected_v = _torch_reference(
        key, value, key_cache, value_cache, slot_mapping, block_size
    )

    reshape_and_cache(key, value, key_cache, value_cache, slot_mapping)

    torch.testing.assert_close(key_cache, expected_k, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(value_cache, expected_v, atol=1e-2, rtol=1e-2)


def test_reshape_and_cache_all_skipped():
    """All slots = -1: cache must be byte-identical to input."""
    num_tokens, num_kv_heads, block_size, head_dim, num_blocks = 16, 4, 16, 64, 8
    dtype = torch.bfloat16
    device = "cuda"
    key = torch.randn(num_tokens, num_kv_heads, head_dim, dtype=dtype, device=device)
    value = torch.randn(num_tokens, num_kv_heads, head_dim, dtype=dtype, device=device)
    key_cache = torch.randn(
        num_blocks, num_kv_heads, block_size, head_dim, dtype=dtype, device=device
    )
    value_cache = torch.randn(
        num_blocks, num_kv_heads, block_size, head_dim, dtype=dtype, device=device
    )
    slot_mapping = torch.full((num_tokens,), -1, dtype=torch.int64, device=device)

    k_before = key_cache.clone()
    v_before = value_cache.clone()

    reshape_and_cache(key, value, key_cache, value_cache, slot_mapping)

    torch.testing.assert_close(key_cache, k_before)
    torch.testing.assert_close(value_cache, v_before)


def test_reshape_and_cache_empty():
    """num_tokens == 0: must be a no-op without launching the kernel."""
    num_kv_heads, block_size, head_dim, num_blocks = 4, 16, 64, 8
    dtype = torch.bfloat16
    device = "cuda"
    key = torch.empty(0, num_kv_heads, head_dim, dtype=dtype, device=device)
    value = torch.empty(0, num_kv_heads, head_dim, dtype=dtype, device=device)
    key_cache = torch.randn(
        num_blocks, num_kv_heads, block_size, head_dim, dtype=dtype, device=device
    )
    value_cache = torch.randn(
        num_blocks, num_kv_heads, block_size, head_dim, dtype=dtype, device=device
    )
    slot_mapping = torch.empty(0, dtype=torch.int64, device=device)

    k_before = key_cache.clone()
    v_before = value_cache.clone()

    reshape_and_cache(key, value, key_cache, value_cache, slot_mapping)

    torch.testing.assert_close(key_cache, k_before)
    torch.testing.assert_close(value_cache, v_before)
