import torch
import pytest

import triton
from op_tests.test_rope import ref_rope_sbhd_fwd, RotateStyle
from op_tests.triton_tests.rope.test_rope import generate_rope_inputs
from op_tests.triton_tests.attention.test_mla import (
    shuffle_kv_buffer,
    dynamic_nvfp4_quant_kv_buffer,
)
from op_tests.triton_tests.attention.test_unified_attention import (
    shuffle_kv_cache,
    dynamic_nvfp4_quant_kv_cache,
)
from op_tests.triton_tests.quant.test_quant_mxfp4 import torch_dequant_nvfp4
from op_tests.triton_tests.test_kv_cache import check_kv_buffer
from aiter.ops.triton.fusions.fused_kv_cache import (
    fused_qk_rope_cat_and_cache_mla,
    fused_qk_rope_reshape_and_cache,
    fused_qk_rope_cosine_cache_llama,
)
from aiter.ops.triton.utils._triton import arch_info
from aiter.ops.triton.utils.types import e4m3_dtype


def split_unshuffle_nvfp4_kv_cache(key_or_value_cache):
    num_blocks, KH, block_size, new_head_size = key_or_value_cache.shape
    assert new_head_size % 9 == 0
    head_size = new_head_size * 16 // 9
    key_or_value_cache = key_or_value_cache.reshape(
        num_blocks * KH, block_size * new_head_size
    )
    key_or_value_cache_data = key_or_value_cache[:, : block_size * head_size // 2]
    key_or_value_cache_scales = key_or_value_cache[
        :, block_size * head_size // 2 :
    ].view(e4m3_dtype)
    key_or_value_cache_data = (
        key_or_value_cache_data.reshape(
            (
                -1,
                block_size // 16,
                (head_size // 2) // (2 * 16),
                2,
                16,
                16,
            )
        )
        .permute(0, 1, 4, 2, 3, 5)
        .reshape(-1, block_size, head_size // 2)
    )
    head_size_scales = head_size // 16
    head_size_scales_k_width = max(4, min(16, triton.next_power_of_2(head_size_scales)))
    key_or_value_cache_scales = (
        key_or_value_cache_scales.reshape(
            (
                -1,
                block_size // 128,
                head_size_scales // head_size_scales_k_width,
                128 // 4,
                4,
                head_size_scales_k_width,
            )
        )
        .permute(0, 1, 4, 3, 2, 5)
        .reshape(-1, block_size, head_size_scales)
    )
    return key_or_value_cache_data, key_or_value_cache_scales


@pytest.mark.parametrize("T", [1, 2, 4, 2048])
@pytest.mark.parametrize("QH_per_KH", [1, 16])
@pytest.mark.parametrize("KH", [1])
@pytest.mark.parametrize("D_pe", [64])  # For now, D is power of 2. D >= 16
@pytest.mark.parametrize("D_lora", [512])
@pytest.mark.parametrize("num_kv_cahce_tokens", [2048])
@pytest.mark.parametrize("rotate_style", [RotateStyle.GPTJ, RotateStyle.NEOX])
@pytest.mark.parametrize("reuse_freqs_front_part", [False, True])
@pytest.mark.parametrize(
    "cache_dtype, shuffled_kv_cache, block_size",
    [
        (torch.bfloat16, True, 64),
        (torch.bfloat16, False, 1),
        (e4m3_dtype, True, 64),
        (torch.uint8, True, 128),
    ],
)
def test_fused_qk_rope_cat_and_cache_mla(
    T: int,
    QH_per_KH: int,
    KH: int,
    D_pe: int,
    D_lora: int,
    num_kv_cahce_tokens: int,
    rotate_style: int,
    reuse_freqs_front_part: bool,
    cache_dtype: bool,
    shuffled_kv_cache: bool,
    block_size: int,
):
    dtype = torch.bfloat16
    pos = True
    _, _, _, _, freqs, positions, offsets, cos, sin = generate_rope_inputs(
        1,
        T,
        KH,
        QH_per_KH,
        D_pe,
        cached=True,
        reuse_freqs_front_part=reuse_freqs_front_part,
        nope=False,
        pos=pos,
        offs=False,
        two_inputs=True,
        layout="thd",
        dtype=dtype,
    )
    q = torch.randn(
        (T, QH_per_KH * KH, D_lora + D_pe), dtype=torch.float32, device="cuda"
    )
    q_nope, q_pe = q.to(dtype).split((D_lora, D_pe), dim=-1)
    k_lora = torch.randn((T, KH, D_lora), dtype=torch.float32, device=q.device) / (
        20 if cache_dtype != torch.bfloat16 else 1
    )
    k_pe = torch.randn((T, KH, D_pe), dtype=torch.float32, device=q.device) / (
        20 if cache_dtype != torch.bfloat16 else 1
    )
    k_lora = k_lora.to(dtype)
    k_pe = k_pe.to(dtype)

    kv_cache = torch.zeros(
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

    ref_freqs = (
        freqs[positions if offsets is None else torch.add(positions, offsets)].squeeze(
            -2
        )
        if pos
        else freqs
    )

    torch_q_nope = q_nope
    torch_q_pe = q_pe
    torch_k_lora = k_lora
    torch_k_pe = k_pe

    torch_q_pe = ref_rope_sbhd_fwd(
        torch_q_pe.unsqueeze(0),
        ref_freqs,
        rotate_style=rotate_style,
        reuse_freqs_front_part=reuse_freqs_front_part,
        nope_first=False,
    ).squeeze(0)
    torch_k_pe = ref_rope_sbhd_fwd(
        torch_k_pe.unsqueeze(0),
        ref_freqs,
        rotate_style=rotate_style,
        reuse_freqs_front_part=reuse_freqs_front_part,
        nope_first=False,
    ).squeeze(0)

    torch_kv_cache = kv_cache.clone()
    torch_k_pe_og_dtype = torch_k_pe.clone()
    torch_q = torch.cat((torch_q_nope, torch_q_pe), dim=-1)
    torch_decode_q_pe = torch_q_pe
    if cache_dtype == torch.bfloat16:
        torch_k_lora = torch_k_lora
        torch_k_pe = torch_k_pe
    elif cache_dtype == e4m3_dtype:
        torch_k_lora = (torch_k_lora.to(torch.float32) / k_scale).to(torch.bfloat16)
        torch_k_pe = (torch_k_pe.to(torch.float32) / k_scale).to(torch.bfloat16)
    else:
        torch_k_lora = (torch_k_lora.to(torch.float32) / k_scale).to(torch.bfloat16)
        torch_k_pe = (torch_k_pe.to(torch.float32) / k_scale).to(torch.bfloat16)

    torch_zeros = torch.zeros(((T, QH_per_KH * KH, D_lora)), dtype=dtype, device="cuda")
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
    triton_q, triton_decode_q_pe, triton_k_pe, triton_zeros = (
        fused_qk_rope_cat_and_cache_mla(
            q_nope,
            q_pe,
            k_lora,
            k_pe,
            triton_kv_cache,
            slot_mapping,
            positions,
            cos,
            sin,
            k_scale,
            (rotate_style == RotateStyle.NEOX),
            num_decode_toks_for_zeros=T,
            apply_scale=(k_pe.dtype != triton_kv_cache.dtype),
            q_out=None,
            decode_q_pe_out=None,
            k_pe_out=None,
            shuffled_kv_cache=shuffled_kv_cache,
        )
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

    torch.testing.assert_close(torch_q, triton_q, atol=1e-1, rtol=1e-1)
    torch.testing.assert_close(
        torch_decode_q_pe, triton_decode_q_pe, atol=1e-1, rtol=1e-1
    )
    torch.testing.assert_close(torch_k_pe_og_dtype, triton_k_pe, atol=1e-1, rtol=1e-1)
    torch.testing.assert_close(torch_zeros, triton_zeros, atol=0.1, rtol=0.1)


@pytest.mark.parametrize("T", [1, 2, 4, 2048])
@pytest.mark.parametrize("QH_per_KH", [1, 16])
@pytest.mark.parametrize("KH", [1, 8])
@pytest.mark.parametrize("D", [64])  # For now, D is power of 2. D >= 16
@pytest.mark.parametrize("num_blocks", [128])
@pytest.mark.parametrize("rotate_style", [RotateStyle.GPTJ, RotateStyle.NEOX])
@pytest.mark.parametrize("reuse_freqs_front_part", [False, True])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize(
    "cache_dtype, flash_layout, value_shuffle_layout, block_size",
    [
        (torch.bfloat16, True, False, 16),
        (torch.bfloat16, False, False, 16),
        (torch.bfloat16, False, True, 16),
        (e4m3_dtype, True, False, 16),
        (e4m3_dtype, False, False, 16),
        (e4m3_dtype, False, True, 16),
        (e4m3_dtype, False, True, 64),
        (torch.uint8, False, False, 128),
    ],
)
@pytest.mark.parametrize("offs", [False, True])
def test_fused_qk_rope_reshape_and_cache(
    T: int,
    QH_per_KH: int,
    KH: int,
    D: int,
    num_blocks: int,
    rotate_style: int,
    reuse_freqs_front_part: bool,
    block_size: int,
    flash_layout: bool,
    value_shuffle_layout: bool,
    cache_dtype: bool,
    offs: bool,
    dtype: torch.dtype,
):
    pos = True
    q, k, _, _, freqs, positions, offsets, cos, sin = generate_rope_inputs(
        1,
        T,
        KH,
        QH_per_KH,
        D,
        cached=True,
        reuse_freqs_front_part=reuse_freqs_front_part,
        nope=False,
        pos=pos,
        offs=offs,
        two_inputs=True,
        layout="thd",
        dtype=dtype,
    )
    v = torch.randn_like(k)

    if cache_dtype != torch.bfloat16:
        k = k / 20
        v = v / 20

    if cache_dtype == torch.uint8 and arch_info.get_arch() not in ("gfx1250",):
        pytest.skip("FP4 cases only supported on gfx1250")
    elif cache_dtype == e4m3_dtype and arch_info.get_arch() not in (
        "gfx1250",
        "gfx950",
    ):
        pytest.skip("FP8 cases only supported on gfx1250 and gfx950")

    key_cache = torch.zeros(
        (num_blocks, block_size, KH, D), dtype=torch.bfloat16, device="cuda"
    )
    value_cache = torch.zeros(
        (num_blocks, block_size, KH, D), dtype=torch.bfloat16, device="cuda"
    )
    x_size = 16 // key_cache.element_size()
    if cache_dtype != torch.bfloat16:
        k_scale = torch.randn(
            [
                1,
            ],
            dtype=torch.float32,
            device="cuda",
        )[0]
        v_scale = torch.randn(
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
        v_scale = torch.ones(
            [
                1,
            ],
            dtype=torch.float32,
            device="cuda",
        )[0]
    slot_mapping = torch.randperm(T, device="cuda")
    ref_freqs = (
        freqs[positions if offsets is None else torch.add(positions, offsets)].squeeze(
            -2
        )
        if pos
        else freqs
    )

    torch_q = ref_rope_sbhd_fwd(
        q.unsqueeze(0),
        ref_freqs,
        rotate_style=rotate_style,
        reuse_freqs_front_part=reuse_freqs_front_part,
        nope_first=False,
    ).squeeze(0)
    torch_k = ref_rope_sbhd_fwd(
        k.unsqueeze(0),
        ref_freqs,
        rotate_style=rotate_style,
        reuse_freqs_front_part=reuse_freqs_front_part,
        nope_first=False,
    ).squeeze(0)

    torch_key_cache = key_cache.clone()
    torch_value_cache = value_cache.clone()
    slot_t = slot_mapping // block_size
    slot_b = slot_mapping % block_size
    torch_k_og_dtype = torch_k.clone()
    if cache_dtype != torch.bfloat16:
        torch_k = (torch_k.to(torch.float32) / k_scale).to(torch.bfloat16)
        torch_v = (v.to(torch.float32) / v_scale).to(torch.bfloat16)
    else:
        torch_v = v
    torch_zeros = torch.zeros_like(q)
    torch_key_cache[slot_t, slot_b] = torch_k
    torch_value_cache[slot_t, slot_b] = torch_v
    if cache_dtype == torch.uint8:
        torch_key_cache, torch_value_cache = dynamic_nvfp4_quant_kv_cache(
            torch_key_cache, torch_value_cache
        )
    elif not flash_layout:
        torch_key_cache = torch_key_cache.to(cache_dtype)
        torch_value_cache = torch_value_cache.to(cache_dtype)
        if value_shuffle_layout:
            torch_key_cache, torch_value_cache = shuffle_kv_cache(
                torch_key_cache, torch_value_cache
            )
        else:
            torch_key_cache, _ = shuffle_kv_cache(torch_key_cache, torch_value_cache)
            torch_value_cache = (
                torch_value_cache.view(num_blocks, block_size, KH, D)
                .permute(0, 2, 3, 1)
                .contiguous()
            )
    else:
        torch_key_cache = torch_key_cache.to(cache_dtype)
        torch_value_cache = torch_value_cache.to(cache_dtype)

    triton_key_cache = torch.zeros_like(torch_key_cache)
    triton_value_cache = torch.zeros_like(torch_value_cache)
    triton_q, triton_k, triton_key_cache, triton_value_cache, triton_zeros = (
        fused_qk_rope_reshape_and_cache(
            q,
            k,
            v,
            triton_key_cache,
            triton_value_cache,
            slot_mapping,
            positions,
            cos,
            sin,
            k_scale,
            v_scale,
            (rotate_style == RotateStyle.NEOX),
            flash_layout=flash_layout,
            apply_scale=(cache_dtype != torch.bfloat16),
            offs=offsets,
            q_out=q,
            k_out=k,
        )
    )

    torch.testing.assert_close(torch_q, triton_q, atol=1e-1, rtol=1e-1)
    torch.testing.assert_close(torch_k_og_dtype, triton_k, atol=1e-1, rtol=1e-1)
    torch.testing.assert_close(torch_zeros, triton_zeros, atol=0.1, rtol=0.1)

    if cache_dtype == torch.uint8:
        ref_key_cache, ref_key_cache_scales = split_unshuffle_nvfp4_kv_cache(
            torch_key_cache
        )
        ref_value_cache, ref_value_cache_scales = split_unshuffle_nvfp4_kv_cache(
            torch_value_cache
        )
        triton_key_cache, triton_key_cache_scales = split_unshuffle_nvfp4_kv_cache(
            triton_key_cache
        )
        triton_value_cache, triton_value_cache_scales = split_unshuffle_nvfp4_kv_cache(
            triton_value_cache
        )
        ref_key_cache_dquant = torch_dequant_nvfp4(
            ref_key_cache, ref_key_cache_scales, out_dtype=dtype
        )
        ref_value_cache_dquant = torch_dequant_nvfp4(
            ref_value_cache, ref_value_cache_scales, out_dtype=dtype
        )
        triton_key_cache_dquant = torch_dequant_nvfp4(
            triton_key_cache, triton_key_cache_scales, out_dtype=dtype
        )
        triton_value_cache_dquant = torch_dequant_nvfp4(
            triton_value_cache, triton_value_cache_scales, out_dtype=dtype
        )
        torch.testing.assert_close(
            ref_key_cache_dquant,
            triton_key_cache_dquant,
            atol=1e-1,
            rtol=1e-1,
        )
        torch.testing.assert_close(
            ref_value_cache_dquant,
            triton_value_cache_dquant,
            atol=1e-1,
            rtol=1e-1,
        )
    else:
        torch_key_cache = torch_key_cache.to(dtype)
        triton_key_cache = triton_key_cache.to(dtype)
        torch_value_cache = torch_value_cache.to(dtype)
        triton_value_cache = triton_value_cache.to(dtype)
        if flash_layout:
            torch.testing.assert_close(
                torch_key_cache[slot_t, slot_b],
                triton_key_cache[slot_t, slot_b],
                atol=1e-1,
                rtol=1e-1,
                equal_nan=arch_info.get_arch()
                not in ["gfx950"],  # TODO: investigate nan elements for non-gfx950 arch
            )
            torch.testing.assert_close(
                torch_value_cache[slot_t, slot_b],
                triton_value_cache[slot_t, slot_b],
                atol=1e-1,
                rtol=1e-1,
                equal_nan=arch_info.get_arch() not in ["gfx950"],
            )
        else:
            torch.testing.assert_close(
                torch_key_cache[slot_t, :, :, slot_b, :],
                triton_key_cache[slot_t, :, :, slot_b, :],
                atol=1e-1,
                rtol=1e-1,
                equal_nan=arch_info.get_arch() not in ["gfx950"],
            )
            torch.testing.assert_close(
                torch_value_cache[slot_t, :, :, slot_b],
                triton_value_cache[slot_t, :, :, slot_b],
                atol=1e-1,
                rtol=1e-1,
                equal_nan=arch_info.get_arch() not in ["gfx950"],
            )

        torch.testing.assert_close(
            torch_key_cache,
            triton_key_cache,
            atol=1e-1,
            rtol=1e-1,
            equal_nan=arch_info.get_arch() not in ["gfx950"],
        )
        torch.testing.assert_close(
            torch_value_cache,
            triton_value_cache,
            atol=1e-1,
            rtol=1e-1,
            equal_nan=arch_info.get_arch() not in ["gfx950"],
        )


@pytest.mark.parametrize("T", [1, 2, 4, 32])
@pytest.mark.parametrize("QH_per_KH", [1, 4])
@pytest.mark.parametrize("KH", [1, 8])
@pytest.mark.parametrize("D", [64, 128])
@pytest.mark.parametrize("num_kv_cahce_tokens", [256, 16384])
@pytest.mark.parametrize("rotate_style", [RotateStyle.GPTJ, RotateStyle.NEOX])
@pytest.mark.parametrize("reuse_freqs_front_part", [False, True])
@pytest.mark.parametrize("block_size", [16])
@pytest.mark.parametrize("x_size", [8])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_fused_qk_rope_reshape_and_cache_value_shuffle_layout(
    T: int,
    QH_per_KH: int,
    KH: int,
    D: int,
    num_kv_cahce_tokens: int,
    rotate_style: int,
    reuse_freqs_front_part: bool,
    block_size: int,
    x_size: int,
    dtype: torch.dtype,
):
    """Test fused_qk_rope_reshape_and_cache with value_cache in shuffle layout
    [num_blocks, num_kv_heads, block_size // x, head_size, x].
    """
    assert D % x_size == 0
    pos = True
    offs = False
    q, k, _, _, freqs, positions, offsets, cos, sin = generate_rope_inputs(
        1,
        T,
        KH,
        QH_per_KH,
        D,
        cached=True,
        reuse_freqs_front_part=reuse_freqs_front_part,
        nope=False,
        pos=pos,
        offs=offs,
        two_inputs=True,
        layout="thd",
        dtype=dtype,
    )
    v = torch.randn_like(k)

    cache_dtype = torch.bfloat16
    k_scale = torch.ones(1, dtype=torch.float32, device="cuda")[0]
    v_scale = torch.ones(1, dtype=torch.float32, device="cuda")[0]

    num_blocks = num_kv_cahce_tokens
    slot_chunk_dim = block_size // x_size
    key_cache = torch.zeros(
        (num_blocks, KH, D // x_size, block_size, x_size),
        dtype=cache_dtype,
        device="cuda",
    )
    value_cache = torch.zeros(
        (num_blocks, KH, slot_chunk_dim, D, x_size),
        dtype=cache_dtype,
        device="cuda",
    )
    slot_mapping = torch.randint(0, num_blocks * block_size, (T,), device="cuda")

    ref_freqs = freqs[
        positions if offsets is None else torch.add(positions, offsets)
    ].squeeze(-2)
    torch_q = ref_rope_sbhd_fwd(
        q.unsqueeze(0),
        ref_freqs,
        rotate_style=rotate_style,
        reuse_freqs_front_part=reuse_freqs_front_part,
        nope_first=False,
    ).squeeze(0)
    torch_k = ref_rope_sbhd_fwd(
        k.unsqueeze(0),
        ref_freqs,
        rotate_style=rotate_style,
        reuse_freqs_front_part=reuse_freqs_front_part,
        nope_first=False,
    ).squeeze(0)

    slot_t = slot_mapping // block_size
    slot_b = slot_mapping % block_size
    torch_key_cache = key_cache.clone()
    torch_value_cache = value_cache.clone()
    torch_k_og_dtype = torch_k.clone()
    torch_v = v

    for t in range(T):
        st, sb = slot_t[t].item(), slot_b[t].item()
        torch_key_cache[st, :, :, sb, :] = torch_k[t].reshape(KH, D // x_size, x_size)
        slot_chunk = sb // x_size
        x_off = sb % x_size
        torch_value_cache[st, :, slot_chunk, :, x_off] = torch_v[t]
    torch_zeros = torch.zeros_like(q)

    triton_key_cache = key_cache.clone()
    triton_value_cache = value_cache.clone()
    triton_q, triton_k, triton_key_cache, triton_value_cache, triton_zeros_out = (
        fused_qk_rope_reshape_and_cache(
            q,
            k,
            v,
            triton_key_cache,
            triton_value_cache,
            slot_mapping,
            positions,
            cos,
            sin,
            k_scale,
            v_scale,
            (rotate_style == RotateStyle.NEOX),
            flash_layout=False,
            apply_scale=True,
            offs=offsets,
            q_out=None,
            k_out=None,
            output_zeros=True,
        )
    )

    torch.testing.assert_close(torch_q, triton_q, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(torch_k_og_dtype, triton_k, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(torch_zeros, triton_zeros_out, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(
        torch_key_cache,
        triton_key_cache,
        atol=1e-2,
        rtol=1e-2,
    )
    torch.testing.assert_close(
        torch_value_cache,
        triton_value_cache,
        atol=1e-2,
        rtol=1e-2,
    )


# gpt-oss-120b config: hidden_size=2880, num_attention_heads=64, num_key_value_heads=8, head_dim=64
GPT_OSS_120B_HEAD_DIM = 64
GPT_OSS_120B_NUM_ATTENTION_HEADS = 64
GPT_OSS_120B_NUM_KV_HEADS = 8


@pytest.mark.parametrize("T", [1, 4, 16, 64])
@pytest.mark.parametrize("block_size", [16])
@pytest.mark.parametrize("x_size", [8])
@pytest.mark.parametrize("num_kv_cahce_tokens", [256, 4096])
def test_fused_qk_rope_reshape_and_cache_gpt_oss_120b_config_value_shuffle_precision(
    T: int,
    block_size: int,
    x_size: int,
    num_kv_cahce_tokens: int,
):
    """Test fused_qk_rope_reshape_and_cache with gpt-oss-120b config; compare 4D vs 5D value_cache for precision.
    Config: head_dim=64, num_attention_heads=64, num_key_value_heads=8.
    """
    D = GPT_OSS_120B_HEAD_DIM
    QH = GPT_OSS_120B_NUM_ATTENTION_HEADS
    KH = GPT_OSS_120B_NUM_KV_HEADS
    QH_per_KH = QH // KH
    assert D % x_size == 0
    dtype = torch.bfloat16
    rotate_style = RotateStyle.GPTJ
    reuse_freqs_front_part = True
    pos = True
    offs = False

    q, k, _, _, freqs, positions, offsets, cos, sin = generate_rope_inputs(
        1,
        T,
        KH,
        QH_per_KH,
        D,
        cached=True,
        reuse_freqs_front_part=reuse_freqs_front_part,
        nope=False,
        pos=pos,
        offs=offs,
        two_inputs=True,
        layout="thd",
        dtype=dtype,
    )
    v = torch.randn_like(k)
    k_scale = torch.ones(1, dtype=torch.float32, device="cuda")[0]
    v_scale = torch.ones(1, dtype=torch.float32, device="cuda")[0]
    slot_mapping = torch.randint(
        0, num_kv_cahce_tokens * block_size, (T,), device="cuda"
    )

    num_blocks = num_kv_cahce_tokens
    slot_chunk_dim = block_size // x_size

    # 1) Run with 4D value_cache (baseline)
    key_cache_4d = torch.zeros(
        (num_blocks, KH, D // x_size, block_size, x_size),
        dtype=dtype,
        device="cuda",
    )
    value_cache_4d = torch.zeros(
        (num_blocks, KH, D, block_size),
        dtype=dtype,
        device="cuda",
    )
    q_out_4d, k_out_4d, kc_4d, vc_4d, zeros_4d = fused_qk_rope_reshape_and_cache(
        q.clone(),
        k.clone(),
        v.clone(),
        key_cache_4d.clone(),
        value_cache_4d.clone(),
        slot_mapping,
        positions,
        cos,
        sin,
        k_scale,
        v_scale,
        (rotate_style == RotateStyle.NEOX),
        flash_layout=False,
        apply_scale=True,
        offs=offsets,
        q_out=None,
        k_out=None,
        output_zeros=True,
    )

    # 2) Run with 5D value_cache (shuffle layout), same inputs
    key_cache_5d = torch.zeros(
        (num_blocks, KH, D // x_size, block_size, x_size),
        dtype=dtype,
        device="cuda",
    )
    value_cache_5d = torch.zeros(
        (num_blocks, KH, slot_chunk_dim, D, x_size),
        dtype=dtype,
        device="cuda",
    )
    q_out_5d, k_out_5d, kc_5d, vc_5d, zeros_5d = fused_qk_rope_reshape_and_cache(
        q.clone(),
        k.clone(),
        v.clone(),
        key_cache_5d.clone(),
        value_cache_5d.clone(),
        slot_mapping,
        positions,
        cos,
        sin,
        k_scale,
        v_scale,
        (rotate_style == RotateStyle.NEOX),
        flash_layout=False,
        apply_scale=True,
        offs=offsets,
        q_out=None,
        k_out=None,
        output_zeros=True,
    )

    # Compare outputs: q_out, k_out, key_cache, zeros should match exactly (same kernel path for these)
    torch.testing.assert_close(
        q_out_4d, q_out_5d, atol=1e-3, rtol=1e-3, msg="q_out 4D vs 5D"
    )
    torch.testing.assert_close(
        k_out_4d, k_out_5d, atol=1e-3, rtol=1e-3, msg="k_out 4D vs 5D"
    )
    torch.testing.assert_close(
        kc_4d, kc_5d, atol=1e-3, rtol=1e-3, msg="key_cache 4D vs 5D"
    )
    torch.testing.assert_close(
        zeros_4d, zeros_5d, atol=1e-3, rtol=1e-3, msg="zeros_out 4D vs 5D"
    )

    # Compare value_cache slot-by-slot: vc_4d[slot_t,:,:,slot_b] vs vc_5d[slot_t,:,slot_b//x,:,slot_b%x]
    slot_t = slot_mapping // block_size
    slot_b = slot_mapping % block_size
    for i in range(T):
        st, sb = slot_t[i].item(), slot_b[i].item()
        v4 = vc_4d[st, :, :, sb]
        v5 = vc_5d[st, :, sb // x_size, :, sb % x_size]
        torch.testing.assert_close(
            v4,
            v5,
            atol=1e-3,
            rtol=1e-3,
            msg=f"value_cache at slot {i} (block={st}, slot_in_block={sb}) 4D vs 5D",
        )


@pytest.mark.parametrize("T", [1, 2, 4, 128])
@pytest.mark.parametrize("QH_per_KH", [1, 4, 16])
@pytest.mark.parametrize("KH", [1, 8])
@pytest.mark.parametrize("D", [64, 128])  # For now, D is power of 2. D >= 16
@pytest.mark.parametrize("num_kv_cahce_tokens", [8193])
@pytest.mark.parametrize("rotate_style", [RotateStyle.GPTJ])
@pytest.mark.parametrize("reuse_freqs_front_part", [True])
@pytest.mark.parametrize("cache_dtype", [torch.bfloat16])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("flash_layout", [True])
@pytest.mark.parametrize("block_size", [16])
@pytest.mark.parametrize("x_size", [8])  # not used
@pytest.mark.parametrize("offs", [False])
def test_fused_qk_rope_cosine_cache_llama(
    T: int,
    QH_per_KH: int,
    KH: int,
    D: int,
    num_kv_cahce_tokens: int,
    rotate_style: int,
    reuse_freqs_front_part: bool,
    block_size: int,
    x_size: int,
    flash_layout: bool,
    cache_dtype: bool,
    offs: bool,
    dtype: torch.dtype,
):
    pos = True
    q, k, _, _, freqs, positions, offsets, cos, sin = generate_rope_inputs(
        1,
        T,
        KH,
        QH_per_KH,
        D,
        cached=True,
        reuse_freqs_front_part=reuse_freqs_front_part,
        nope=False,
        pos=pos,
        offs=offs,
        two_inputs=True,
        layout="thd",
        dtype=dtype,
    )
    v = torch.randn_like(k)

    if cache_dtype == torch.uint8:
        if arch_info.get_arch() in ["gfx950"]:
            cache_dtype_actual = torch.float8_e4m3fn
        else:
            cache_dtype_actual = torch.float8_e4m3fnuz

    if flash_layout:
        key_cache = torch.zeros(
            (T, num_kv_cahce_tokens, KH, D), dtype=cache_dtype, device="cuda"
        )
        value_cache = torch.zeros(
            (T, num_kv_cahce_tokens, KH, D), dtype=cache_dtype, device="cuda"
        )
    else:
        pytest.skip()

    if cache_dtype == torch.uint8:
        k_scale = torch.randn(
            [
                1,
            ],
            dtype=torch.float32,
            device="cuda",
        )[0]
        v_scale = torch.randn(
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
        v_scale = torch.ones(
            [
                1,
            ],
            dtype=torch.float32,
            device="cuda",
        )[0]
    slot_mapping = torch.randperm(T, device="cuda")
    positions = slot_mapping
    key_cache_og_dtype = key_cache.dtype
    value_cache_og_dtype = value_cache.dtype

    ref_freqs = (
        freqs[positions if offsets is None else torch.add(positions, offsets)].squeeze(
            -2
        )
        if pos
        else freqs
    )

    torch_q = ref_rope_sbhd_fwd(
        q.unsqueeze(0),
        ref_freqs,
        rotate_style=rotate_style,
        reuse_freqs_front_part=reuse_freqs_front_part,
        nope_first=False,
    ).squeeze(0)
    torch_k = ref_rope_sbhd_fwd(
        k.unsqueeze(0),
        ref_freqs,
        rotate_style=rotate_style,
        reuse_freqs_front_part=reuse_freqs_front_part,
        nope_first=False,
    ).squeeze(0)

    torch_key_cache = key_cache.clone()
    torch_value_cache = value_cache.clone()
    # slot_t = slot_mapping // block_size
    # slot_b = slot_mapping % block_size
    slot_t = torch.arange(slot_mapping.shape[0]).to(slot_mapping.device)
    slot_b = slot_mapping
    if cache_dtype == torch.uint8:
        torch_key_cache = torch_key_cache.view(cache_dtype_actual)
        torch_value_cache = torch_value_cache.view(cache_dtype_actual)
        torch_k = (torch_k.to(torch.float32) / k_scale).to(cache_dtype_actual)
        torch_v = (v.to(torch.float32) / v_scale).to(cache_dtype_actual)
    else:
        torch_v = v
    if flash_layout:
        torch_key_cache[slot_t, slot_b] = torch_k
        torch_value_cache[slot_t, slot_b] = torch_v

    torch_key_cache = torch_key_cache.view(key_cache_og_dtype)
    torch_value_cache = torch_value_cache.view(value_cache_og_dtype)

    triton_key_cache = key_cache.clone()
    triton_value_cache = value_cache.clone()
    if cache_dtype == torch.uint8:
        triton_key_cache = triton_key_cache.view(cache_dtype_actual)
        triton_value_cache = triton_value_cache.view(cache_dtype_actual)
    triton_q, triton_key_cache, triton_value_cache = fused_qk_rope_cosine_cache_llama(
        q,
        k,
        v,
        triton_key_cache,
        triton_value_cache,
        slot_mapping,
        positions,
        cos,
        sin,
        k_scale,
        v_scale,
        (rotate_style == RotateStyle.NEOX),
        flash_layout=flash_layout,
        apply_scale=(cache_dtype != torch.bfloat16),
        offs=offsets,
        q_out=q,
    )
    triton_key_cache = triton_key_cache.view(key_cache_og_dtype)
    triton_value_cache = triton_value_cache.view(value_cache_og_dtype)

    torch.testing.assert_close(torch_q, triton_q, atol=1e-1, rtol=1e-1)

    if cache_dtype == torch.uint8:
        torch_key_cache = torch_key_cache.view(cache_dtype_actual).to(dtype)
        triton_key_cache = triton_key_cache.view(cache_dtype_actual).to(dtype)
        torch_value_cache = torch_value_cache.view(cache_dtype_actual).to(dtype)
        triton_value_cache = triton_value_cache.view(cache_dtype_actual).to(dtype)

    if flash_layout:
        torch.testing.assert_close(
            torch_key_cache[slot_t, slot_b],
            triton_key_cache[slot_t, slot_b],
            atol=1e-1,
            rtol=1e-1,
        )
        torch.testing.assert_close(
            torch_value_cache[slot_t, slot_b],
            triton_value_cache[slot_t, slot_b],
            atol=1e-1,
            rtol=1e-1,
        )

    torch.testing.assert_close(torch_key_cache, triton_key_cache, atol=1e-1, rtol=1e-1)
    torch.testing.assert_close(
        torch_value_cache, triton_value_cache, atol=1e-1, rtol=1e-1
    )
