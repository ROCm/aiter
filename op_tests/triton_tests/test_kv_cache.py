import pytest
import torch

from aiter.ops.triton.kv_cache import cat_and_cache_mla
from aiter.ops.triton.utils._triton import arch_info
from aiter.ops.triton.utils.fp4_kv import (
    fp4_kv_rope_is_fp8,
    fp4_kv_row_bytes,
    fp4_kv_scale_block,
    fp4_kv_scale_k_width,
)
from aiter.ops.triton.utils.types import e4m3_dtype
from aiter.test_common import assertAllclose
from op_tests.triton_tests.attention.test_mla import (
    dynamic_fp4_quant_kv_buffer,
    shuffle_kv_buffer,
)
from op_tests.triton_tests.quant.test_quant_mxfp4 import (
    torch_dequant_mxfp4,
    torch_dequant_nvfp4,
)

DEVICE_ARCH = arch_info.get_arch()


def torch_dequant_fp4(x, scales, fp4_kv_format, out_dtype):
    """Dequantize one region; a region without scales is FP8 rope."""
    if scales.shape[-1] == 0:
        return x.view(e4m3_dtype).to(out_dtype)
    if fp4_kv_format != "nvfp4":
        return torch_dequant_mxfp4(
            x, scales, out_dtype, block_size=fp4_kv_scale_block(fp4_kv_format)
        )
    return torch_dequant_nvfp4(x, scales, out_dtype=out_dtype)


def split_unshuffle_fp4_kv_buffer(kv_buffer, D_lora, D_pe, fp4_kv_format="nvfp4"):
    """(lora, lora scales, rope, rope scales) per token; FP8 rope comes back as
    its E4M3 bytes with empty scales."""
    num_blocks, KH, block_size, new_head_size = kv_buffer.shape
    block = fp4_kv_scale_block(fp4_kv_format)
    rope_fp8 = fp4_kv_rope_is_fp8(fp4_kv_format)
    D_pe_store = D_pe if rope_fp8 else D_pe // 2
    scale_dtype = e4m3_dtype if fp4_kv_format == "nvfp4" else torch.uint8
    kv_buffer = kv_buffer.reshape(num_blocks * KH, block_size * new_head_size)
    kv_buffer_lora = kv_buffer[:, : block_size * D_lora // 2]
    kv_buffer_lora_scales = kv_buffer[
        :, block_size * D_lora // 2 : block_size * (D_lora // 2 + D_lora // block)
    ].view(scale_dtype)
    kv_buffer_rope = kv_buffer[
        :,
        block_size
        * (D_lora // 2 + D_lora // block) : block_size
        * (D_lora // 2 + D_lora // block + D_pe_store),
    ]
    kv_buffer_rope_scales = kv_buffer[
        :, block_size * (D_lora // 2 + D_lora // block + D_pe_store) :
    ].view(scale_dtype)
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
                D_pe_store // (2 * 16),
                2,
                16,
                16,
            )
        )
        .permute(0, 1, 4, 2, 3, 5)
        .reshape(-1, block_size, D_pe_store)
    )
    D_lora_scales = D_lora // block
    if rope_fp8:
        return (
            kv_buffer_lora,
            _unshuffle_fp4_scales(kv_buffer_lora_scales, block_size, D_lora_scales),
            kv_buffer_rope,
            kv_buffer_rope_scales.reshape(num_blocks * KH, block_size, 0),
        )
    return (
        kv_buffer_lora,
        _unshuffle_fp4_scales(kv_buffer_lora_scales, block_size, D_lora_scales),
        kv_buffer_rope,
        _unshuffle_fp4_scales(kv_buffer_rope_scales, block_size, D_pe // block),
    )


def _unshuffle_fp4_scales(scales, block_size, num_scales):
    """Undo the 32x4 scale shuffle: per token, its ``num_scales`` scales."""
    k_width = fp4_kv_scale_k_width(num_scales)
    return (
        scales.reshape(
            (-1, block_size // 128, num_scales // k_width, 128 // 4, 4, k_width)
        )
        .permute(0, 1, 4, 3, 2, 5)
        .reshape(-1, block_size, num_scales)
    )


def split_unshuffle_nvfp4_kv_buffer(kv_buffer, D_lora, D_pe):
    return split_unshuffle_fp4_kv_buffer(kv_buffer, D_lora, D_pe, "nvfp4")


def check_kv_buffer(
    kv_buffer,
    ref_kv_buffer,
    slot_mapping,
    block_size,
    shuffled_kv_cache,
    D_lora,
    D_pe,
    dtype=torch.bfloat16,
    fp4_kv_format="nvfp4",
):
    cache_dtype = ref_kv_buffer.dtype
    if cache_dtype == torch.uint8:
        # NVFP4 / MXFP4 shuffled KV buffer
        assert shuffled_kv_cache
        (
            ref_kv_buffer_lora,
            ref_kv_buffer_lora_scales,
            ref_kv_buffer_rope,
            ref_kv_buffer_rope_scales,
        ) = split_unshuffle_fp4_kv_buffer(ref_kv_buffer, D_lora, D_pe, fp4_kv_format)
        kv_buffer_lora, kv_buffer_lora_scales, kv_buffer_rope, kv_buffer_rope_scales = (
            split_unshuffle_fp4_kv_buffer(kv_buffer, D_lora, D_pe, fp4_kv_format)
        )
        ref_kv_buffer_lora_dquant = torch_dequant_fp4(
            ref_kv_buffer_lora, ref_kv_buffer_lora_scales, fp4_kv_format, dtype
        )
        ref_kv_buffer_rope_dquant = torch_dequant_fp4(
            ref_kv_buffer_rope, ref_kv_buffer_rope_scales, fp4_kv_format, dtype
        )
        kv_buffer_lora_dquant = torch_dequant_fp4(
            kv_buffer_lora, kv_buffer_lora_scales, fp4_kv_format, dtype
        )
        kv_buffer_rope_dquant = torch_dequant_fp4(
            kv_buffer_rope, kv_buffer_rope_scales, fp4_kv_format, dtype
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
        assertAllclose(
            gather_written_slots(ref_kv_buffer_lora_dquant),
            gather_written_slots(kv_buffer_lora_dquant),
            atol=1e-1,
            rtol=1e-1,
            tol_err_ratio=0.05,
            msg=f"{fp4_kv_format} kv_buffer lora dequant",
        )
        assertAllclose(
            gather_written_slots(ref_kv_buffer_rope_dquant),
            gather_written_slots(kv_buffer_rope_dquant),
            atol=1e-1,
            rtol=1e-1,
            tol_err_ratio=0.05,
            msg=f"{fp4_kv_format} kv_buffer rope dequant",
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
    "cache_dtype, shuffled_kv_cache, block_size, fp4_kv_format",
    [
        (torch.bfloat16, True, 64, "nvfp4"),
        (torch.bfloat16, False, 1, "nvfp4"),
        (e4m3_dtype, True, 64, "nvfp4"),
        (torch.uint8, True, 128, "nvfp4"),
        (torch.uint8, True, 128, "mxfp4"),
        (torch.uint8, True, 128, "mxfp4_b16"),
        (torch.uint8, True, 128, "mxfp4_fp8rope"),
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
    fp4_kv_format: str,
):
    if cache_dtype == torch.uint8 and DEVICE_ARCH not in (
        "gfx950",
        "gfx1250",
    ):
        pytest.skip("FP4 KV cache is only supported in GFX950 and GFX1250")

    torch.manual_seed(0)

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
        torch_k_lora = torch_k_lora  # noqa: PLW0127
        torch_k_pe = torch_k_pe  # noqa: PLW0127
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
        torch_kv_cache = dynamic_fp4_quant_kv_buffer(
            torch_kv_cache, D_lora, fp4_kv_format
        )
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
        fp4_kv_format=fp4_kv_format,
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
        fp4_kv_format=fp4_kv_format,
    )


@pytest.mark.parametrize("fp4_kv_format", ["mxfp4", "mxfp4_b16", "mxfp4_fp8rope"])
@pytest.mark.parametrize("T", [1, 37, 256, 2048])
@pytest.mark.parametrize("magnitude", [1.0, 1 / 20, 1e-3])
def test_cat_and_cache_mla_mxfp4_bit_exact(T: int, magnitude: float, fp4_kv_format):
    """With a unit k_scale the kernel quantizes the same BF16 values the torch
    reference does, so every byte of a written slot (codes, E8M0 scales and
    FP8 rope, read back through the shuffle) must match, and unwritten slots
    stay zero."""
    if DEVICE_ARCH not in ("gfx950", "gfx1250"):
        pytest.skip("FP4 KV cache is only supported in GFX950 and GFX1250")
    torch.manual_seed(0)
    D_lora, D_pe, KH, block_size, num_tokens = 512, 64, 1, 128, 4096
    k_lora = (torch.randn((T, KH, D_lora), device="cuda") * magnitude).bfloat16()
    k_pe = (torch.randn((T, KH, D_pe), device="cuda") * magnitude).bfloat16()
    # Outliers exercise scale selection at the top of each 32-value block.
    k_lora[:, :, ::97] *= 9
    slot_mapping = torch.randperm(num_tokens, device="cuda")[:T]

    ref = torch.zeros(
        (num_tokens, KH, D_lora + D_pe), dtype=torch.bfloat16, device="cuda"
    )
    ref[slot_mapping] = torch.cat((k_lora, k_pe), dim=-1)
    ref = dynamic_fp4_quant_kv_buffer(
        ref.reshape(num_tokens // block_size, block_size, KH, D_lora + D_pe),
        D_lora,
        fp4_kv_format,
    )
    assert ref.shape[-1] == fp4_kv_row_bytes(D_lora, D_pe, fp4_kv_format)
    out = torch.zeros_like(ref)
    cat_and_cache_mla(
        k_lora,
        k_pe,
        out,
        slot_mapping,
        torch.ones((), dtype=torch.float32, device="cuda"),
        shuffled_kv_cache=True,
        fp4_kv_format=fp4_kv_format,
    )

    written = torch.zeros(num_tokens, dtype=torch.bool, device="cuda")
    written[slot_mapping] = True
    names = ("lora", "lora scales", "rope", "rope scales")
    ref_parts = split_unshuffle_fp4_kv_buffer(ref, D_lora, D_pe, fp4_kv_format)
    out_parts = split_unshuffle_fp4_kv_buffer(out, D_lora, D_pe, fp4_kv_format)
    for name, ref_part, out_part in zip(names, ref_parts, out_parts):
        ref_part = ref_part.reshape(num_tokens, -1)
        out_part = out_part.reshape(num_tokens, -1)
        mismatch = (out_part[written] != ref_part[written]).sum().item()
        assert mismatch == 0, f"{name}: {mismatch} bytes differ in written slots"
        assert not out_part[~written].any(), f"{name}: unwritten slots modified"
