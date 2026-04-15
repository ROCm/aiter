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
    torch_k_pe_og_dtype = torch_k_pe.clone()
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
