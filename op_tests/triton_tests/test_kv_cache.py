import torch
import pytest

import triton
from op_tests.triton_tests.attention.test_mla import shuffle_kv_buffer
from aiter.ops.triton.kv_cache import cat_and_cache_mla
from op_tests.triton_tests.attention.test_mla import dynamic_nvfp4_quant_kv_buffer
from op_tests.triton_tests.quant.test_quant_mxfp4 import torch_dequant_nvfp4

from aiter.ops.triton.utils.types import e4m3_dtype
from aiter.test_common import checkAllclose
import aiter.ops.triton.utils._triton.arch_info as arch_info

DEVICE_ARCH = arch_info.get_arch()


def split_unshuffle_nvfp4_kv_buffer(kv_buffer, D_lora, D_pe):
    num_blocks, KH, block_size, new_head_size = kv_buffer.shape
    kv_buffer = kv_buffer.reshape(num_blocks * KH, block_size * new_head_size)
    kv_buffer_lora = kv_buffer[:, : block_size * D_lora // 2]
    kv_buffer_lora_scales = kv_buffer[
        :, block_size * D_lora // 2 : block_size * (D_lora // 2 + D_lora // 16)
    ].view(e4m3_dtype)
    kv_buffer_rope = kv_buffer[
        :,
        block_size
        * (D_lora // 2 + D_lora // 16) : block_size
        * (D_lora // 2 + D_lora // 16 + D_pe // 2),
    ]
    kv_buffer_rope_scales = kv_buffer[
        :, block_size * (D_lora // 2 + D_lora // 16 + D_pe // 2) :
    ].view(e4m3_dtype)
    kv_buffer_lora = (
        kv_buffer_lora.reshape(
            (
                -1,
                block_size // 16,
                (D_lora // 2) // (2 * 16),
                2,
                16,
                16,
            )
        )
        .permute(0, 1, 4, 2, 3, 5)
        .reshape(-1, block_size, D_lora // 2)
    )
    kv_buffer_rope = (
        kv_buffer_rope.reshape(
            (
                -1,
                block_size // 16,
                (D_pe // 2) // (2 * 16),
                2,
                16,
                16,
            )
        )
        .permute(0, 1, 4, 2, 3, 5)
        .reshape(-1, block_size, D_pe // 2)
    )
    D_lora_scales = D_lora // 16
    D_pe_scales = D_pe // 16
    D_lora_scales_k_width = max(4, min(16, triton.next_power_of_2(D_lora_scales)))
    D_pe_scales_k_width = max(4, min(16, triton.next_power_of_2(D_pe_scales)))
    kv_buffer_lora_scales = (
        kv_buffer_lora_scales.reshape(
            (
                -1,
                block_size // 128,
                D_lora_scales // D_lora_scales_k_width,
                128 // 4,
                4,
                D_lora_scales_k_width,
            )
        )
        .permute(0, 1, 4, 3, 2, 5)
        .reshape(-1, block_size, D_lora_scales)
    )
    kv_buffer_rope_scales = (
        kv_buffer_rope_scales.reshape(
            (
                -1,
                block_size // 128,
                D_pe_scales // D_pe_scales_k_width,
                128 // 4,
                4,
                D_pe_scales_k_width,
            )
        )
        .permute(0, 1, 4, 3, 2, 5)
        .reshape(-1, block_size, D_pe_scales)
    )
    return kv_buffer_lora, kv_buffer_lora_scales, kv_buffer_rope, kv_buffer_rope_scales


def check_kv_buffer(
    kv_buffer,
    ref_kv_buffer,
    slot_mapping,
    block_size,
    shuffled_kv_cache,
    D_lora,
    D_pe,
    dtype=torch.bfloat16,
):
    cache_dtype = ref_kv_buffer.dtype
    if cache_dtype == torch.uint8:
        # NVFP4 shuffled KV buffer
        assert shuffled_kv_cache
        (
            ref_kv_buffer_lora,
            ref_kv_buffer_lora_scales,
            ref_kv_buffer_rope,
            ref_kv_buffer_rope_scales,
        ) = split_unshuffle_nvfp4_kv_buffer(ref_kv_buffer, D_lora, D_pe)
        kv_buffer_lora, kv_buffer_lora_scales, kv_buffer_rope, kv_buffer_rope_scales = (
            split_unshuffle_nvfp4_kv_buffer(kv_buffer, D_lora, D_pe)
        )
        ref_kv_buffer_lora_dquant = torch_dequant_nvfp4(
            ref_kv_buffer_lora, ref_kv_buffer_lora_scales, out_dtype=dtype
        )
        ref_kv_buffer_rope_dquant = torch_dequant_nvfp4(
            ref_kv_buffer_rope, ref_kv_buffer_rope_scales, out_dtype=dtype
        )
        kv_buffer_lora_dquant = torch_dequant_nvfp4(
            kv_buffer_lora, kv_buffer_lora_scales, out_dtype=dtype
        )
        kv_buffer_rope_dquant = torch_dequant_nvfp4(
            kv_buffer_rope, kv_buffer_rope_scales, out_dtype=dtype
        )
        # Only compare the slots that were actually written via slot_mapping. The
        # dequantized buffers are (num_blocks * KH, block_size, D); most of the
        # buffer is zero-padding that matches trivially and would otherwise
        # dilute the mismatch ratio, making tol_err_ratio meaningless.
        num_blocks, KH = ref_kv_buffer.shape[0], ref_kv_buffer.shape[1]
        slot_t = slot_mapping // block_size
        slot_b = slot_mapping % block_size

        def gather_written_slots(dquant):
            # (num_blocks * KH, block_size, D) -> (num_blocks, KH, block_size, D)
            # -> (T, KH, D) at the (block, slot_in_block) positions of slot_mapping
            d = dquant.shape[-1]
            return dquant.reshape(num_blocks, KH, block_size, d)[slot_t, :, slot_b]

        # NVFP4 is a 4-bit format with only 8 magnitude levels, so a single FP4
        # quantization step (~0.15 at these magnitudes) exceeds atol=0.1. Benign
        # fp32-vs-bf16 / RoPE arithmetic differences between the kernel and this
        # reference can flip a small number of elements to an adjacent FP4 code.
        # Tolerate a tiny fraction of such mismatches instead of requiring an
        # exact match.
        checkAllclose(
            gather_written_slots(ref_kv_buffer_lora_dquant),
            gather_written_slots(kv_buffer_lora_dquant),
            atol=1e-1,
            rtol=1e-1,
            tol_err_ratio=0.05,
            msg="NVFP4 kv_buffer lora dequant",
        )
        checkAllclose(
            gather_written_slots(ref_kv_buffer_rope_dquant),
            gather_written_slots(kv_buffer_rope_dquant),
            atol=1e-1,
            rtol=1e-1,
            tol_err_ratio=0.05,
            msg="NVFP4 kv_buffer rope dequant",
        )
    elif shuffled_kv_cache:
        # FP8 (e4m3) or BF16 shuffled KV buffer
        ref_kv_buffer = ref_kv_buffer.to(dtype)
        kv_buffer = kv_buffer.to(dtype)
        torch.testing.assert_close(
            ref_kv_buffer[slot_mapping // block_size],
            kv_buffer[slot_mapping // block_size],
            atol=1e-1,
            rtol=1e-1,
        )
        torch.testing.assert_close(ref_kv_buffer, kv_buffer, atol=1e-1, rtol=1e-1)
    else:
        # FP8 or BF16 non-shuffled KV buffer
        ref_kv_buffer = ref_kv_buffer.to(dtype)
        kv_buffer = kv_buffer.to(dtype)
        torch.testing.assert_close(
            ref_kv_buffer[slot_mapping],
            kv_buffer[slot_mapping],
            atol=1e-1,
            rtol=1e-1,
        )
        torch.testing.assert_close(ref_kv_buffer, kv_buffer, atol=1e-1, rtol=1e-1)


@pytest.mark.parametrize("T", [1, 2, 4, 256, 2048])
@pytest.mark.parametrize("KH", [1, 16])
@pytest.mark.parametrize("D_pe", [64])  # For now, D is power of 2. D >= 16
@pytest.mark.parametrize("D_lora", [512])
@pytest.mark.parametrize("num_kv_cahce_tokens", [2048])
@pytest.mark.parametrize(
    "cache_dtype, shuffled_kv_cache, block_size",
    [
        (torch.bfloat16, True, 64),
        (torch.bfloat16, False, 1),
        (e4m3_dtype, True, 64),
        (torch.uint8, True, 128),
    ],
)
def test_cat_and_cache_mla(
    T: int,
    KH: int,
    D_pe: int,
    D_lora: int,
    num_kv_cahce_tokens: int,
    cache_dtype: bool,
    shuffled_kv_cache: bool,
    block_size: int,
):
    if cache_dtype == torch.uint8 and DEVICE_ARCH not in (
        "gfx950",
        "gfx1250",
    ):
        pytest.skip("FP4 KV cache is only supported in GFX950 and GFX1250")

    dtype = torch.bfloat16
    k_lora = torch.randn((T, KH, D_lora), dtype=torch.float32, device="cuda") / (
        20 if cache_dtype != torch.bfloat16 else 1
    )
    k_pe = torch.randn((T, KH, D_pe), dtype=torch.float32, device="cuda") / (
        20 if cache_dtype != torch.bfloat16 else 1
    )
    k_lora = k_lora.to(dtype)
    k_pe = k_pe.to(dtype)

    torch_kv_cache = torch.zeros(
        (num_kv_cahce_tokens, KH, D_lora + D_pe), dtype=torch.bfloat16, device="cuda"
    )

    if cache_dtype != torch.bfloat16:
        k_scale = torch.rand(
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

    torch_k_lora = k_lora
    torch_k_pe = k_pe

    if cache_dtype == torch.bfloat16:
        torch_k_lora = torch_k_lora
        torch_k_pe = torch_k_pe
    elif cache_dtype == e4m3_dtype:
        torch_k_lora = (torch_k_lora.to(torch.float32) / k_scale).to(torch.bfloat16)
        torch_k_pe = (torch_k_pe.to(torch.float32) / k_scale).to(torch.bfloat16)
    else:
        torch_k_lora = (torch_k_lora.to(torch.float32) / k_scale).to(torch.bfloat16)
        torch_k_pe = (torch_k_pe.to(torch.float32) / k_scale).to(torch.bfloat16)

    torch_kv_cache[slot_mapping, :, :] = torch.cat((torch_k_lora, torch_k_pe), dim=-1)
    if cache_dtype == torch.uint8:
        torch_kv_cache = torch_kv_cache.reshape(
            num_kv_cahce_tokens // block_size, block_size, KH, D_lora + D_pe
        )
        torch_kv_cache = dynamic_nvfp4_quant_kv_buffer(torch_kv_cache, D_lora)
    elif shuffled_kv_cache:
        torch_kv_cache = shuffle_kv_buffer(
            torch_kv_cache.reshape(
                num_kv_cahce_tokens // block_size, block_size, KH, D_lora + D_pe
            ).to(cache_dtype),
            D_lora,
        )
    else:
        torch_kv_cache = torch_kv_cache.to(cache_dtype)
    triton_kv_cache = torch.zeros_like(torch_kv_cache)
    cat_and_cache_mla(
        k_lora,
        k_pe,
        triton_kv_cache,
        slot_mapping,
        k_scale,
        shuffled_kv_cache=shuffled_kv_cache,
    )

    check_kv_buffer(
        torch_kv_cache,
        triton_kv_cache,
        slot_mapping,
        block_size,
        shuffled_kv_cache,
        D_lora,
        D_pe,
        dtype,
    )
