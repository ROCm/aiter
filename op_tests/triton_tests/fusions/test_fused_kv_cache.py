import pytest
import torch
import triton

from aiter.ops.triton.fusions.fused_kv_cache import (
    fused_qk_cat_and_cache_mla,
    fused_qk_rope_cat_and_cache_mla,
    fused_qk_rope_cosine_cache_llama,
    fused_qk_rope_reshape_and_cache,
    reshape_and_cache_k4v8,
)
from aiter.ops.triton.utils._triton import arch_info
from aiter.ops.triton.utils.types import e4m3_dtype
from aiter.test_common import checkAllclose
from op_tests.test_rope import RotateStyle, ref_rope_sbhd_fwd
from op_tests.triton_tests.attention.test_mla import (
    dynamic_nvfp4_quant_kv_buffer,
    shuffle_kv_buffer,
)
from op_tests.triton_tests.attention.test_unified_attention import (
    dynamic_nvfp4_quant_kv_cache,
    shuffle_kv_cache,
)
from op_tests.triton_tests.quant.test_quant_mxfp4 import torch_dequant_nvfp4
from op_tests.triton_tests.rope.test_rope import generate_rope_inputs
from op_tests.triton_tests.test_kv_cache import check_kv_buffer

DEVICE_ARCH = arch_info.get_arch()


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


@pytest.mark.parametrize("T", [1, 8, 2048])
@pytest.mark.parametrize("QH_per_KH", [16])
@pytest.mark.parametrize("KH", [1])
@pytest.mark.parametrize("D_pe", [64])  # For now, D is power of 2. D >= 16
@pytest.mark.parametrize("D_lora", [512])
@pytest.mark.parametrize("num_kv_cahce_tokens", [16384])
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
@pytest.mark.parametrize("upcast_operand", [False, True])
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
    upcast_operand: bool,
):
    if cache_dtype == torch.uint8 and DEVICE_ARCH not in ("gfx1250", "gfx1250-strict"):
        pytest.skip("NVFP4 quantization is only supported on GFX1250")
    torch.manual_seed(0)
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
        torch_k_lora = torch_k_lora  # noqa: PLW0127
        torch_k_pe = torch_k_pe  # noqa: PLW0127
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
    num_decode_toks_for_zeros = T
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
            num_decode_toks_for_zeros=num_decode_toks_for_zeros,
            apply_scale=(k_pe.dtype != triton_kv_cache.dtype),
            q_out=None,
            decode_q_pe_out=None,
            k_pe_out=None,
            shuffled_kv_cache=shuffled_kv_cache,
            upcast_operand=upcast_operand,
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
    if num_decode_toks_for_zeros > 0:
        torch.testing.assert_close(
            torch_decode_q_pe, triton_decode_q_pe, atol=1e-1, rtol=1e-1
        )
        torch.testing.assert_close(torch_zeros, triton_zeros, atol=0.1, rtol=0.1)
    torch.testing.assert_close(torch_k_pe_og_dtype, triton_k_pe, atol=1e-1, rtol=1e-1)


@pytest.mark.parametrize("T", [1, 8, 2048])
@pytest.mark.parametrize("QH", [12, 16])  # 12: Kimi-K3's 96 heads at TP8
@pytest.mark.parametrize("D_pe", [64])
@pytest.mark.parametrize("D_lora", [512])
@pytest.mark.parametrize("num_kv_cahce_tokens", [16384])
@pytest.mark.parametrize(
    "cache_dtype, shuffled_kv_cache, block_size",
    [
        (torch.bfloat16, True, 64),
        (torch.bfloat16, False, 1),
        (e4m3_dtype, True, 64),
        # sglang's MLA pool, as Kimi-K3 runs it: flat fp8 (tokens, 1, D).
        (e4m3_dtype, False, 1),
    ],
)
@pytest.mark.parametrize("mixed_prefill", [False, True])
@pytest.mark.parametrize("output_zeros", ["none", "partial", "all"])
@pytest.mark.parametrize("padded_slots", [False, True])
def test_fused_qk_cat_and_cache_mla(
    T: int,
    QH: int,
    D_pe: int,
    D_lora: int,
    num_kv_cahce_tokens: int,
    cache_dtype: torch.dtype,
    shuffled_kv_cache: bool,
    block_size: int,
    mixed_prefill: bool,
    output_zeros: str,
    padded_slots: bool,
):
    # The gluon kernel has no NoPE form. That also rules out the NVFP4 cache,
    # which is gfx1250-only.
    if DEVICE_ARCH in ("gfx1250", "gfx1250-strict"):
        pytest.skip("fused_qk_cat_and_cache_mla has no gfx1250 kernel")
    torch.manual_seed(0)
    dtype = torch.bfloat16
    KH = 1  # MLA: a single latent KV head
    D_qk_nope = 128
    # q exists only for the decode tokens; with mixed_prefill the trailing
    # tokens just write k (the kernel's tail branch).
    B = max(T // 2, 1) if mixed_prefill else T

    # Layouts as sglang's MLA absorb path hands them over: q_nope is the
    # absorbed BMM output (QH, B, D_lora) seen through a transpose, q_pe is the
    # tail of the q_b_proj split, and k_pe is the tail of the kv_a latent.
    q_nope = torch.randn((QH, B, D_lora), dtype=dtype, device="cuda").transpose(0, 1)
    q_pe = torch.randn((B, QH, D_qk_nope + D_pe), dtype=dtype, device="cuda")[
        ..., D_qk_nope:
    ]
    latent = (
        torch.randn((T, D_lora + D_pe), dtype=torch.float32, device="cuda")
        / (20 if cache_dtype != torch.bfloat16 else 1)
    ).to(dtype)
    k_nope = latent[:, :D_lora].contiguous().unsqueeze(1)
    k_pe = latent[:, D_lora:].unsqueeze(1)

    if cache_dtype != torch.bfloat16:
        k_scale = torch.rand([1], dtype=torch.float32, device="cuda")[0]
    else:
        k_scale = torch.ones([1], dtype=torch.float32, device="cuda")[0]
    slot_mapping = torch.randperm(num_kv_cahce_tokens, device="cuda")[:T]
    if padded_slots:
        # Kernel contract: slot -1 (padding) writes neither the cache nor
        # k_pe_out. Every other token, so the decode and the tail branch both
        # see padded and valid slots.
        slot_mapping[1::2] = -1
    valid = slot_mapping >= 0

    # NoPE reference: plain concats, no rotation anywhere.
    torch_q = torch.cat((q_nope, q_pe), dim=-1)
    torch_k = torch.cat((k_nope, k_pe), dim=-1)
    if cache_dtype != torch.bfloat16:
        torch_k = (torch_k.to(torch.float32) / k_scale).to(torch.bfloat16)
    torch_kv_cache = torch.zeros(
        (num_kv_cahce_tokens, KH, D_lora + D_pe), dtype=torch.bfloat16, device="cuda"
    )
    torch_kv_cache[slot_mapping[valid]] = torch_k[valid]
    if shuffled_kv_cache:
        torch_kv_cache = shuffle_kv_buffer(
            torch_kv_cache.reshape(
                num_kv_cahce_tokens // block_size, block_size, KH, D_lora + D_pe
            ).to(cache_dtype),
            D_lora,
        )
    else:
        torch_kv_cache = torch_kv_cache.to(cache_dtype)
    triton_kv_cache = torch.zeros_like(torch_kv_cache)

    num_decode_toks_for_zeros = {"none": 0, "partial": B // 2, "all": B}[output_zeros]
    # Sentinel-filled outputs catch stray writes: decode_q_pe_out is a view of
    # the first num_decode_toks_for_zeros rows of a B-row buffer, so a write to
    # a row past it lands in the tail; k_pe_out rows of padded slots must stay.
    sentinel = -123.0
    decode_q_pe_buf = torch.full((B, QH, D_pe), sentinel, dtype=dtype, device="cuda")
    k_pe_out = torch.full((T, KH, D_pe), sentinel, dtype=dtype, device="cuda")
    # sglang hands q to the decode kernel in the cache dtype, except with 12
    # heads (Kimi-K3 at TP8), where the aiter gluon MLA decode needs bf16 q.
    q_out_dtype = dtype if QH == 12 else cache_dtype
    triton_q, triton_decode_q_pe, triton_k_pe, triton_zeros = (
        fused_qk_cat_and_cache_mla(
            q_nope,
            q_pe,
            k_nope,
            k_pe,
            triton_kv_cache,
            slot_mapping,
            k_scale,
            num_decode_toks_for_zeros=num_decode_toks_for_zeros,
            apply_scale=(k_pe.dtype != triton_kv_cache.dtype),
            decode_q_pe_out=decode_q_pe_buf[:num_decode_toks_for_zeros],
            k_pe_out=k_pe_out,
            q_out_dtype=q_out_dtype,
            shuffled_kv_cache=shuffled_kv_cache,
        )
    )

    check_kv_buffer(
        torch_kv_cache,
        triton_kv_cache,
        slot_mapping[valid],
        block_size,
        shuffled_kv_cache,
        D_lora,
        D_pe,
        dtype,
    )

    # Everything outside the cache is a pure copy, so bf16 outputs must match
    # bit for bit; an fp8 q_out may differ by one e4m3 ulp in the cast.
    if q_out_dtype == torch.bfloat16:
        q_tol = {"atol": 0, "rtol": 0}
    else:
        q_tol = {"atol": 2**-9, "rtol": 0.125}
    torch.testing.assert_close(
        torch_q.to(q_out_dtype).to(torch.float32),
        triton_q.to(torch.float32),
        **q_tol,
    )
    torch.testing.assert_close(k_pe[valid], triton_k_pe[valid], atol=0, rtol=0)
    assert bool((triton_k_pe[~valid] == sentinel).all()), "padded k_pe_out written"
    n = num_decode_toks_for_zeros
    torch.testing.assert_close(q_pe[:n], triton_decode_q_pe, atol=0, rtol=0)
    assert bool((decode_q_pe_buf[n:] == sentinel).all()), "decode_q_pe_out overrun"
    torch.testing.assert_close(
        torch.zeros_like(q_nope[:n]), triton_zeros, atol=0, rtol=0
    )


@pytest.mark.parametrize("T", [1, 8, 2048])
@pytest.mark.parametrize("QH_per_KH", [16])
@pytest.mark.parametrize("KH", [8])
@pytest.mark.parametrize("D", [64])  # For now, D is power of 2. D >= 16
@pytest.mark.parametrize("num_blocks", [16384])
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
@pytest.mark.parametrize("upcast_operand", [False, True])
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
    upcast_operand: bool,
):
    if cache_dtype == torch.uint8 and DEVICE_ARCH not in ("gfx1250", "gfx1250-strict"):
        pytest.skip("NVFP4 quantization is only supported on GFX1250")
    torch.manual_seed(0)
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

    if cache_dtype == torch.uint8 and arch_info.get_arch() not in (
        "gfx1250",
        "gfx1250-strict",
    ):
        pytest.skip("FP4 cases only supported on gfx1250")
    elif cache_dtype == e4m3_dtype and arch_info.get_arch() not in (
        "gfx1250",
        "gfx1250-strict",
        "gfx950",
    ):
        pytest.skip("FP8 cases only supported on gfx1250 and gfx950")

    key_cache = torch.zeros(
        (num_blocks, block_size, KH, D), dtype=torch.bfloat16, device="cuda"
    )
    value_cache = torch.zeros(
        (num_blocks, block_size, KH, D), dtype=torch.bfloat16, device="cuda"
    )
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
            upcast_operand=upcast_operand,
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
        tol_err_ratio = 0.05
        assert (
            checkAllclose(
                ref_key_cache_dquant,
                triton_key_cache_dquant,
                atol=1e-1,
                rtol=1e-1,
                tol_err_ratio=tol_err_ratio,
                msg="key_cache dequant (nvfp4)",
            )
            <= tol_err_ratio
        )
        assert (
            checkAllclose(
                ref_value_cache_dquant,
                triton_value_cache_dquant,
                atol=1e-1,
                rtol=1e-1,
                tol_err_ratio=tol_err_ratio,
                msg="value_cache dequant (nvfp4)",
            )
            <= tol_err_ratio
        )
    else:
        torch_key_cache = torch_key_cache.to(dtype)
        triton_key_cache = triton_key_cache.to(dtype)
        torch_value_cache = torch_value_cache.to(dtype)
        triton_value_cache = triton_value_cache.to(dtype)
        tol_err_ratio = 0.05
        if flash_layout:
            ref_key = torch_key_cache[slot_t, slot_b]
            tri_key = triton_key_cache[slot_t, slot_b]
            ref_value = torch_value_cache[slot_t, slot_b]
            tri_value = triton_value_cache[slot_t, slot_b]
        else:
            ref_key = torch_key_cache[slot_t, :, :, slot_b, :]
            tri_key = triton_key_cache[slot_t, :, :, slot_b, :]
            ref_value = torch_value_cache[slot_t, :, :, slot_b]
            tri_value = triton_value_cache[slot_t, :, :, slot_b]

        assert (
            checkAllclose(
                ref_key,
                tri_key,
                atol=1e-1,
                rtol=1e-1,
                tol_err_ratio=tol_err_ratio,
                msg="key_cache written slots",
            )
            <= tol_err_ratio
        )
        assert (
            checkAllclose(
                ref_value,
                tri_value,
                atol=1e-1,
                rtol=1e-1,
                tol_err_ratio=tol_err_ratio,
                msg="value_cache written slots",
            )
            <= tol_err_ratio
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

    torch.manual_seed(0)
    q, k, _, _, _freqs, positions, offsets, cos, sin = generate_rope_inputs(
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
    torch.manual_seed(0)
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
    slot_t = torch.arange(slot_mapping.shape[0], device="cuda")
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


# ---------------------------------------------------------------------------
# A8K4V8 cache write: NVFP4 K + FP8 V, both in the shuffled layout the gfx1250
# unified-attention kernels read.
# ---------------------------------------------------------------------------


def _k4v8_ref_k_cache(k_full_bf16: torch.Tensor) -> torch.Tensor:
    """NVFP4 K cache [num_blocks, KH, block_size, D // 2 + D // 16] from a bf16
    [num_blocks, block_size, KH, D] cache, built by the attention tests' builders."""
    block_size = k_full_bf16.shape[1]
    if block_size >= 128:
        return dynamic_nvfp4_quant_kv_cache(k_full_bf16, k_full_bf16)[0]
    from op_tests.op_benchmarks.triton.bench_unified_attention import (
        nvfp4_quant_shuffle_cache,
    )

    return nvfp4_quant_shuffle_cache(k_full_bf16, min(128, block_size))


_E2M1_MAGNITUDES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def _assert_nvfp4_cache_equal_up_to_ties(got, ref, x_full):
    """Compare two shuffled NVFP4 caches [num_blocks, KH, block_size, D // 2 + D // 16].

    The e4m3 scales (and everything outside the e2m1 data) must be bit-exact. An e2m1
    code may differ from the torch reference only by one step at a rounding threshold
    of the scaled value x / scale: the reference computes x * (1 / (amax / 6)) with
    IEEE fp32 division, the kernel (_nvfp4_quant_op) uses Triton's fp32 division, which
    is not correctly rounded, so where x / scale is exactly a threshold (x = amax / 8
    gives 0.75 for bf16 inputs) the two land one ulp apart and round to the two
    neighbouring codes. Every such difference is checked to be exactly that case.
    x_full: the bf16 [num_blocks, block_size, KH, D] cache the reference was built from.
    """
    nb, kh, bs, d_cache = got.shape
    D = d_cache * 16 // 9
    n_data = bs * D // 2
    g = got.reshape(nb, kh, -1)
    r = ref.reshape(nb, kh, -1)
    assert torch.equal(g[..., n_data:], r[..., n_data:]), "NVFP4 scales differ"

    def unshuffle(c):  # inverse of shuffle_weight(arch="gfx950") on [bs, D // 2]
        return (
            c[..., :n_data]
            .reshape(nb, kh, bs // 16, D // 32, 16, 16)
            .permute(0, 1, 2, 4, 3, 5)
            .reshape(nb, kh, bs, D // 2)
        )

    def nibbles(c):
        c = unshuffle(c).to(torch.int32)
        return torch.stack([c & 0xF, c >> 4], dim=-1).reshape(nb, kh, bs, D)

    gn, rn = nibbles(g), nibbles(r)
    diff = gn != rn
    n_diff = int(diff.sum())
    if n_diff == 0:
        return 0
    # the scaled value the reference rounded
    x = x_full.permute(0, 2, 1, 3).float().reshape(nb, kh, bs, D // 16, 16)
    scale = x.abs().amax(dim=-1, keepdim=True) / 6.0
    qx = (x * (1.0 / scale)).reshape(nb, kh, bs, D)[diff].abs()
    mags = torch.tensor(_E2M1_MAGNITUDES, dtype=torch.float32, device=got.device)
    gm, rm = mags[gn[diff] & 7], mags[rn[diff] & 7]
    assert torch.equal(gn[diff] & 8, rn[diff] & 8), "NVFP4 sign differs"
    lo, hi = torch.minimum(gm, rm), torch.maximum(gm, rm)
    idx_lo = torch.searchsorted(mags, lo)
    assert torch.equal(mags[(idx_lo + 1).clamp(max=7)], hi), "codes not adjacent"
    threshold = (lo + hi) / 2
    assert bool(
        ((qx - threshold).abs() <= 4 * torch.finfo(torch.float32).eps * threshold).all()
    ), "NVFP4 code differs away from a rounding threshold"
    return n_diff


def _k4v8_slots(scenario: str, num_blocks: int, block_size: int):
    """Slot mapping: decode = scattered single tokens (+ one padded -1 slot);
    prefill = contiguous runs over a random page table, crossing page boundaries,
    one of them continuing a cached prefix (chunked prefill)."""
    g = torch.Generator(device="cpu").manual_seed(1)
    if scenario == "decode":
        slots = torch.randperm(num_blocks * block_size, generator=g)[:24]
        slots[5] = -1
        return slots.to(torch.int64).cuda()
    pages = torch.randperm(num_blocks, generator=g)
    runs = [  # (first page index, start position, number of tokens)
        (0, 0, 2 * block_size + 37),
        (3, 13, block_size + 20),
        (5, 0, 5),
    ]
    slots = []
    for p0, start, n in runs:
        pos = torch.arange(start, start + n)
        slots.append(pages[p0 + pos // block_size] * block_size + pos % block_size)
    return torch.cat(slots).to(torch.int64).cuda()


def _k4v8_write(T, QH, KH, D, num_blocks, block_size, slot_mapping, seed=0):
    """Write random K/V at slot_mapping with fused_qk_rope_reshape_and_cache into
    caches pre-filled with other random content. Returns the written caches and
    the bf16 / fp8 full caches the reference builders turn into the same bytes."""
    torch.manual_seed(seed)
    q, k, _, _, freqs, positions, _, cos, sin = generate_rope_inputs(
        1,
        T,
        KH,
        QH // KH,
        D,
        cached=True,
        reuse_freqs_front_part=True,
        nope=False,
        pos=True,
        offs=False,
        two_inputs=True,
        layout="thd",
        dtype=torch.bfloat16,
    )
    torch.manual_seed(seed)
    v = torch.randn_like(k)
    k_scale = torch.ones(1, dtype=torch.float32, device="cuda")  # k_descale = 1
    v_scale = (v.abs().amax().float() / torch.finfo(e4m3_dtype).max).reshape(1)

    # what was in the cache before this write
    old_k = torch.randn(
        (num_blocks, block_size, KH, D), dtype=torch.bfloat16, device="cuda"
    )
    old_v = torch.randn(
        (num_blocks, block_size, KH, D), dtype=torch.bfloat16, device="cuda"
    ).to(e4m3_dtype)
    key_cache = _k4v8_ref_k_cache(old_k)
    value_cache = shuffle_kv_cache(old_v, old_v)[1]
    assert key_cache.dtype == torch.uint8 and value_cache.dtype == e4m3_dtype

    _, k_out, key_cache, value_cache, _ = fused_qk_rope_reshape_and_cache(
        q,
        k,
        v,
        key_cache,
        value_cache,
        slot_mapping,
        positions,
        cos,
        sin,
        k_scale,
        v_scale,
        is_neox=True,
        flash_layout=False,
        apply_scale=True,
        q_out=torch.empty_like(q),
        k_out=torch.empty_like(k),
    )

    # reference full caches: the kernel quantizes the roped bf16 K (k_out) / k_scale,
    # and V * (1 / v_scale) cast to fp8
    valid = slot_mapping >= 0
    slot_t = slot_mapping[valid] // block_size
    slot_b = slot_mapping[valid] % block_size
    k_full = old_k.clone()
    k_full[slot_t, slot_b] = (k_out[valid].float() * (1.0 / k_scale)).to(torch.bfloat16)
    v_full = old_v.clone()
    v_full[slot_t, slot_b] = (v[valid].float() * (1.0 / v_scale)).to(e4m3_dtype)
    return {
        "q": q,
        "k": k,
        "v": v,
        "k_out": k_out,
        "positions": positions,
        "freqs": freqs,
        "key_cache": key_cache,
        "value_cache": value_cache,
        "k_full": k_full,
        "v_full": v_full,
        "v_scale": v_scale,
    }


@pytest.mark.parametrize("scenario", ["decode", "prefill"])
@pytest.mark.parametrize("block_size", [64, 128])
@pytest.mark.parametrize("D", [64, 128, 256])
@pytest.mark.parametrize("QH, KH", [(8, 1), (16, 4)])
def test_fused_qk_rope_reshape_and_cache_k4v8(scenario, block_size, D, QH, KH):
    if DEVICE_ARCH not in ("gfx1250",):
        pytest.skip("NVFP4 KV cache is only supported on GFX1250")
    num_blocks = 8
    slot_mapping = _k4v8_slots(scenario, num_blocks, block_size)
    T = slot_mapping.numel()
    r = _k4v8_write(T, QH, KH, D, num_blocks, block_size, slot_mapping)

    # RoPE itself is covered by test_fused_qk_rope_reshape_and_cache; sanity only
    ref_k = ref_rope_sbhd_fwd(
        r["k"].unsqueeze(0),
        r["freqs"][r["positions"]].squeeze(-2),
        rotate_style=RotateStyle.NEOX,
        reuse_freqs_front_part=True,
        nope_first=False,
    ).squeeze(0)
    valid = slot_mapping >= 0  # k_out is only written for valid slots
    torch.testing.assert_close(r["k_out"][valid], ref_k[valid], atol=1e-1, rtol=1e-1)

    # bit-exact over the whole cache: written rows match the reference builders and
    # nothing else (other rows, other pages, the other layout part) was touched
    ref_key_cache = _k4v8_ref_k_cache(r["k_full"])
    ref_value_cache = shuffle_kv_cache(r["v_full"], r["v_full"])[1]
    assert r["key_cache"].shape == ref_key_cache.shape
    assert r["value_cache"].shape == ref_value_cache.shape
    # V (fp8) is bit-exact; K is bit-exact except e2m1 codes at exact rounding ties
    assert torch.equal(
        r["value_cache"].view(torch.uint8), ref_value_cache.view(torch.uint8)
    ), "FP8 V cache differs"
    n_ties = _assert_nvfp4_cache_equal_up_to_ties(
        r["key_cache"], ref_key_cache, r["k_full"]
    )
    n_written = int((slot_mapping >= 0).sum()) * KH * D
    assert n_ties <= n_written // 100, f"{n_ties} tie flips of {n_written} K values"


@pytest.mark.parametrize("block_size", [64, 128])
@pytest.mark.parametrize("QH, KH", [(8, 1), (32, 4)])
@torch.inference_mode()
def test_k4v8_cache_write_unified_attention(block_size, QH, KH):
    """Write an A8K4V8 cache with fused_qk_rope_reshape_and_cache and run
    unified_attention (FP8 Q, NVFP4 K, FP8 V, shuffled) on it, against
    ref_paged_attn on the dequantized K and the fp8 V with its descale."""
    if DEVICE_ARCH not in ("gfx1250",):
        pytest.skip("A8K4V8 attention is only supported on GFX1250")
    from aiter.ops.triton.attention.unified_attention import unified_attention
    from op_tests.op_benchmarks.triton.bench_unified_attention import (
        nvfp4_quant_dequant,
        quantize_to_fp8,
    )
    from op_tests.triton_tests.attention.test_unified_attention import (
        ref_paged_attn,
    )

    D = 128
    num_blocks = 64
    seq_lens = [(1, 1328), (7, 523), (64, 300), (1, 37)]  # (query_len, kv_len)
    max_pages = max((kv + block_size - 1) // block_size for _, kv in seq_lens)
    g = torch.Generator(device="cpu").manual_seed(2)
    pages = torch.randperm(num_blocks, generator=g)
    block_tables = torch.zeros((len(seq_lens), max_pages), dtype=torch.int32)
    slots, p = [], 0
    for i, (_, kv) in enumerate(seq_lens):
        n = (kv + block_size - 1) // block_size
        block_tables[i, :n] = pages[p : p + n]
        p += n
        pos = torch.arange(kv)
        slots.append(
            block_tables[i, pos // block_size].long() * block_size + pos % block_size
        )
    slot_mapping = torch.cat(slots).cuda()
    block_tables = block_tables.cuda()
    T = slot_mapping.numel()

    # the whole KV of every sequence is written by the kernel (prefill-style)
    r = _k4v8_write(T, QH, KH, D, num_blocks, block_size, slot_mapping, seed=3)
    _assert_nvfp4_cache_equal_up_to_ties(
        r["key_cache"], _k4v8_ref_k_cache(r["k_full"]), r["k_full"]
    )
    assert torch.equal(
        r["value_cache"].view(torch.uint8),
        shuffle_kv_cache(r["v_full"], r["v_full"])[1].view(torch.uint8),
    )

    query_lens = [s[0] for s in seq_lens]
    kv_lens = [s[1] for s in seq_lens]
    query = torch.randn(sum(query_lens), QH, D, dtype=torch.bfloat16, device="cuda")
    q_fp8, q_descale = quantize_to_fp8(query)
    cu_q = torch.tensor([0] + query_lens, dtype=torch.int32, device="cuda").cumsum(
        0, dtype=torch.int32
    )
    kv_lens_t = torch.tensor(kv_lens, dtype=torch.int32, device="cuda")
    k_descale = torch.ones(1, dtype=torch.float32, device="cuda")
    v_descale = r["v_scale"]
    scale = D**-0.5
    out = torch.empty(sum(query_lens), QH, D, dtype=torch.bfloat16, device="cuda")
    unified_attention(
        q=q_fp8,
        k=r["key_cache"],
        v=r["value_cache"],
        out=out,
        cu_seqlens_q=cu_q,
        max_seqlen_q=max(query_lens),
        seqused_k=kv_lens_t,
        max_seqlen_k=max(kv_lens),
        softmax_scale=scale,
        causal=True,
        window_size=(-1, -1),
        block_table=block_tables,
        softcap=0,
        q_descale=q_descale,
        k_descale=k_descale,
        v_descale=v_descale,
        shuffled_kv_cache=True,
    )
    ref = ref_paged_attn(
        query=q_fp8.to(torch.float32) * q_descale,
        key_cache=nvfp4_quant_dequant(r["k_full"]),
        value_cache=r["v_full"],
        query_lens=query_lens,
        kv_lens=kv_lens,
        block_tables=block_tables,
        scale=scale,
        out_dtype=torch.bfloat16,
        k_descale=k_descale,
        v_descale=v_descale,
    )
    max_diff = (out.float() - ref.float()).abs().max().item()
    rel_l2 = ((out.float() - ref.float()).norm() / ref.float().norm()).item()
    # control: the same attention on a K cache with the pages' rows rotated must miss
    ref_bad = ref_paged_attn(
        query=q_fp8.to(torch.float32) * q_descale,
        key_cache=nvfp4_quant_dequant(r["k_full"]).roll(1, dims=1),
        value_cache=r["v_full"],
        query_lens=query_lens,
        kv_lens=kv_lens,
        block_tables=block_tables,
        scale=scale,
        out_dtype=torch.bfloat16,
        k_descale=k_descale,
        v_descale=v_descale,
    )
    rel_l2_bad = ((out.float() - ref_bad.float()).norm() / ref.float().norm()).item()
    print(
        f"A8K4V8 e2e block_size={block_size} QH={QH} KH={KH}: max|ref|="
        f"{ref.float().abs().max().item():.4f} max|diff|={max_diff:.5f} "
        f"rel_l2={rel_l2:.5f} (rows-rotated K control: {rel_l2_bad:.4f})"
    )
    assert rel_l2 < 0.05
    torch.testing.assert_close(out.float(), ref.float(), atol=1.5e-1, rtol=1.5e-1)


@pytest.mark.parametrize("scenario", ["decode", "prefill"])
@pytest.mark.parametrize("block_size", [64, 128])
@pytest.mark.parametrize("D", [128, 256])
@pytest.mark.parametrize("KH", [1, 2, 4])
@pytest.mark.parametrize("packed", [False, True])
def test_reshape_and_cache_k4v8(scenario, block_size, D, KH, packed):
    """The write-only A8K4V8 op: bit-exact V and K up to NVFP4 rounding ties against the
    reference builders over the whole cache, and byte-identical to the cache write of
    fused_qk_rope_reshape_and_cache given that kernel's own roped K."""
    if DEVICE_ARCH not in ("gfx1250",):
        pytest.skip("NVFP4 KV cache is only supported on GFX1250")
    num_blocks = 8
    QH = 4 * KH
    slot_mapping = _k4v8_slots(scenario, num_blocks, block_size)
    T = slot_mapping.numel()
    # the fused rope + cache write, as reference bytes and as the source of a roped K
    r = _k4v8_write(T, QH, KH, D, num_blocks, block_size, slot_mapping, seed=7)
    k_roped, v = r["k_out"], r["v"]
    if packed:
        # strided views into one packed [T, (QH + 2 KH) * D] tensor, as ATOM splits qkv
        qkv = torch.empty((T, (QH + 2 * KH) * D), dtype=torch.bfloat16, device="cuda")
        qkv[:, QH * D : (QH + KH) * D] = k_roped.reshape(T, KH * D)
        qkv[:, (QH + KH) * D :] = v.reshape(T, KH * D)
        k_in = qkv[:, QH * D : (QH + KH) * D].view(T, KH, D)
        v_in = qkv[:, (QH + KH) * D :].view(T, KH, D)
    else:
        k_in, v_in = k_roped, v
    k_scale = torch.ones(1, dtype=torch.float32, device="cuda")  # k_descale = 1
    v_scale = r["v_scale"]

    # 1) the same bytes as the fused rope + cache write: re-writing its rows from its own
    #    roped K into its output caches must leave every byte as it is
    key_cache, value_cache = r["key_cache"].clone(), r["value_cache"].clone()
    reshape_and_cache_k4v8(
        k_in, v_in, key_cache, value_cache, slot_mapping, k_scale, v_scale
    )
    assert torch.equal(
        key_cache, r["key_cache"]
    ), "K cache differs from fused_qk_rope_reshape_and_cache"
    assert torch.equal(
        value_cache.view(torch.uint8), r["value_cache"].view(torch.uint8)
    ), "V cache differs from fused_qk_rope_reshape_and_cache"

    # 2) into fresh random caches, against the torch reference builders over the whole cache
    #    (written rows match, nothing else touched)
    torch.manual_seed(11)
    old_k = torch.randn(
        (num_blocks, block_size, KH, D), dtype=torch.bfloat16, device="cuda"
    )
    old_v = torch.randn(
        (num_blocks, block_size, KH, D), dtype=torch.bfloat16, device="cuda"
    ).to(e4m3_dtype)
    key_cache = _k4v8_ref_k_cache(old_k)
    value_cache = shuffle_kv_cache(old_v, old_v)[1]
    reshape_and_cache_k4v8(
        k_in, v_in, key_cache, value_cache, slot_mapping, k_scale, v_scale
    )
    valid = slot_mapping >= 0
    slot_t, slot_b = slot_mapping[valid] // block_size, slot_mapping[valid] % block_size
    k_full = old_k.clone()
    k_full[slot_t, slot_b] = k_roped[valid]
    v_full = old_v.clone()
    v_full[slot_t, slot_b] = (v[valid].float() * (1.0 / v_scale)).to(e4m3_dtype)
    ref_value_cache = shuffle_kv_cache(v_full, v_full)[1]
    assert torch.equal(
        value_cache.view(torch.uint8), ref_value_cache.view(torch.uint8)
    ), "FP8 V cache differs from the reference"
    n_ties = _assert_nvfp4_cache_equal_up_to_ties(
        key_cache, _k4v8_ref_k_cache(k_full), k_full
    )
    n_written = int(valid.sum()) * KH * D
    assert n_ties <= n_written // 100, f"{n_ties} tie flips of {n_written} K values"
