import torch
import pytest

from op_tests.test_rope import ref_rope_sbhd_fwd, RotateStyle
from op_tests.triton_tests.rope.test_rope import generate_rope_inputs
from aiter.ops.triton.fused_kv_cache import (
    fused_qk_rope_cat_and_cache_mla,
    fused_qk_rope_reshape_and_cache,
    fused_qk_rope_cosine_cache_llama,
)
from aiter.ops.triton.utils._triton import arch_info


@pytest.mark.parametrize("T", [1, 2, 4, 2048])
@pytest.mark.parametrize("QH_per_KH", [1, 16])
@pytest.mark.parametrize("KH", [1, 8])
@pytest.mark.parametrize("D", [128])  # For now, D is power of 2. D >= 16
@pytest.mark.parametrize("D_q_nope", [128])
@pytest.mark.parametrize("D_lora", [512])
@pytest.mark.parametrize("num_kv_cahce_tokens", [16384])
@pytest.mark.parametrize("rotate_style", [RotateStyle.GPTJ, RotateStyle.NEOX])
@pytest.mark.parametrize("reuse_freqs_front_part", [False, True])
@pytest.mark.parametrize("cache_dtype", [torch.bfloat16, torch.uint8])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_fused_qk_rope_cat_and_cache_mla(
    T: int,
    QH_per_KH: int,
    KH: int,
    D: int,
    D_q_nope: int,
    D_lora: int,
    num_kv_cahce_tokens: int,
    rotate_style: int,
    reuse_freqs_front_part: bool,
    cache_dtype: bool,
    dtype: torch.dtype,
):
    pos = True
    _, _, _, _, freqs, positions, offsets, cos, sin = generate_rope_inputs(
        1,
        T,
        KH,
        QH_per_KH,
        D,
        cached=True,
        reuse_freqs_front_part=reuse_freqs_front_part,
        nope=False,
        pos=pos,
        offs=False,
        two_inputs=True,
        layout="thd",
        dtype=dtype,
    )
    q = torch.randn((T, QH_per_KH * KH, D_q_nope + D), dtype=dtype, device="cuda")
    q_nope, q_pe = q.split((D_q_nope, D), dim=-1)
    k_lora = torch.randn((T, KH, D_lora), dtype=dtype, device=q.device) / (
        20 if cache_dtype == torch.uint8 else 1
    )
    k_pe = torch.randn((T, KH, D), dtype=dtype, device=q.device) / (
        20 if cache_dtype == torch.uint8 else 1
    )

    if cache_dtype == torch.uint8:
        if arch_info.get_arch() in ["gfx950"]:
            cache_dtype_actual = torch.float8_e4m3fn
        else:
            cache_dtype_actual = torch.float8_e4m3fnuz

    kv_cache = torch.zeros(
        (num_kv_cahce_tokens, KH, D_lora + D), dtype=cache_dtype, device="cuda"
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
    if cache_dtype == torch.uint8:
        torch_kv_cache = torch_kv_cache.view(cache_dtype_actual)
        torch_k_lora = (torch_k_lora.to(torch.float32) / k_scale).to(cache_dtype_actual)
        torch_k_pe = (torch_k_pe.to(torch.float32) / k_scale).to(cache_dtype_actual)
    else:
        torch_k_lora = torch_k_lora
        torch_k_pe = torch_k_pe

    torch_zeros = torch.zeros(((T, QH_per_KH * KH, D_lora)), dtype=dtype, device="cuda")
    torch_kv_cache[slot_mapping, :, :] = torch.cat((torch_k_lora, torch_k_pe), dim=-1)
    torch_kv_cache = torch_kv_cache.view(kv_cache_og_dtype)

    triton_kv_cache = kv_cache.clone()
    if cache_dtype == torch.uint8:
        triton_kv_cache = triton_kv_cache.view(cache_dtype_actual)
    triton_q, triton_decode_q_pe, triton_k_pe, triton_kv_cache, triton_zeros = (
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
            apply_scale=(k_pe.dtype != kv_cache.dtype),
            q_out=None,
            decode_q_pe_out=None,
            k_pe_out=None,
        )
    )
    triton_kv_cache = triton_kv_cache.view(kv_cache_og_dtype)

    torch.testing.assert_close(torch_q, triton_q, atol=1e-1, rtol=1e-1)
    torch.testing.assert_close(
        torch_decode_q_pe, triton_decode_q_pe, atol=1e-1, rtol=1e-1
    )
    torch.testing.assert_close(torch_k_pe_og_dtype, triton_k_pe, atol=1e-1, rtol=1e-1)
    torch.testing.assert_close(torch_zeros, triton_zeros, atol=0.1, rtol=0.1)

    if cache_dtype == torch.uint8:
        torch_kv_cache = torch_kv_cache.view(cache_dtype_actual).to(dtype)
        triton_kv_cache = triton_kv_cache.view(cache_dtype_actual).to(dtype)

    torch.testing.assert_close(
        torch_kv_cache[slot_mapping, :, :],
        triton_kv_cache[slot_mapping, :, :],
        atol=1e-1,
        rtol=1e-1,
    )

    torch.testing.assert_close(torch_kv_cache, triton_kv_cache, atol=1e-1, rtol=1e-1)


@pytest.mark.parametrize("T", [1, 2, 4, 2048])
@pytest.mark.parametrize("QH_per_KH", [1, 16])
@pytest.mark.parametrize("KH", [1, 8])
@pytest.mark.parametrize("D", [64])  # For now, D is power of 2. D >= 16
@pytest.mark.parametrize("num_kv_cahce_tokens", [16384])
@pytest.mark.parametrize("rotate_style", [RotateStyle.GPTJ, RotateStyle.NEOX])
@pytest.mark.parametrize("reuse_freqs_front_part", [False, True])
@pytest.mark.parametrize("cache_dtype", [torch.bfloat16, torch.uint8])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("cache_flash", [False, True])
@pytest.mark.parametrize("block_size", [16])
@pytest.mark.parametrize("x_size", [8])
@pytest.mark.parametrize("offs", [False, True])
def test_fused_qk_rope_reshape_and_cache(
    T: int,
    QH_per_KH: int,
    KH: int,
    D: int,
    num_kv_cahce_tokens: int,
    rotate_style: int,
    reuse_freqs_front_part: bool,
    block_size: int,
    x_size: int,
    cache_flash: bool,
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
            pytest.skip("Skipping FP8 dtype cases non-gfx950")

    if cache_flash:
        key_cache = torch.zeros(
            (num_kv_cahce_tokens, block_size, KH, D), dtype=cache_dtype, device="cuda"
        )
        value_cache = torch.zeros(
            (num_kv_cahce_tokens, block_size, KH, D), dtype=cache_dtype, device="cuda"
        )
    else:
        key_cache = torch.zeros(
            (num_kv_cahce_tokens, KH, D // x_size, block_size, x_size),
            dtype=cache_dtype,
            device="cuda",
        )
        value_cache = torch.zeros(
            (num_kv_cahce_tokens, KH, D, block_size), dtype=cache_dtype, device="cuda"
        )
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
    slot_t = slot_mapping // block_size
    slot_b = slot_mapping % block_size
    torch_k_og_dtype = torch_k.clone()
    if cache_dtype == torch.uint8:
        torch_key_cache = torch_key_cache.view(cache_dtype_actual)
        torch_value_cache = torch_value_cache.view(cache_dtype_actual)
        torch_k = (torch_k.to(torch.float32) / k_scale).to(cache_dtype_actual)
        torch_v = (v.to(torch.float32) / v_scale).to(cache_dtype_actual)
    else:
        torch_v = v
    torch_zeros = torch.zeros_like(q)
    if cache_flash:
        torch_key_cache[slot_t, slot_b] = torch_k
        torch_value_cache[slot_t, slot_b] = torch_v
    else:
        torch_key_cache[slot_t, :, :, slot_b, :] = torch_k.reshape(
            T, KH, D // x_size, x_size
        )
        torch_value_cache[slot_t, :, :, slot_b] = torch_v
    torch_key_cache = torch_key_cache.view(key_cache_og_dtype)
    torch_value_cache = torch_value_cache.view(value_cache_og_dtype)

    triton_key_cache = key_cache.clone()
    triton_value_cache = value_cache.clone()
    if cache_dtype == torch.uint8:
        triton_key_cache = triton_key_cache.view(cache_dtype_actual)
        triton_value_cache = triton_value_cache.view(cache_dtype_actual)
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
            flash_layout=cache_flash,
            apply_scale=(cache_dtype != torch.bfloat16),
            offs=offsets,
            q_out=q,
            k_out=k,
        )
    )
    triton_key_cache = triton_key_cache.view(key_cache_og_dtype)
    triton_value_cache = triton_value_cache.view(value_cache_og_dtype)

    torch.testing.assert_close(torch_q, triton_q, atol=1e-1, rtol=1e-1)
    torch.testing.assert_close(torch_k_og_dtype, triton_k, atol=1e-1, rtol=1e-1)
    torch.testing.assert_close(torch_zeros, triton_zeros, atol=0.1, rtol=0.1)

    if cache_dtype == torch.uint8:
        torch_key_cache = torch_key_cache.view(cache_dtype_actual).to(dtype)
        triton_key_cache = triton_key_cache.view(cache_dtype_actual).to(dtype)
        torch_value_cache = torch_value_cache.view(cache_dtype_actual).to(dtype)
        triton_value_cache = triton_value_cache.view(cache_dtype_actual).to(dtype)

    if cache_flash:
        torch.testing.assert_close(
            torch_key_cache[slot_t, slot_b],
            triton_key_cache[slot_t, slot_b],
            atol=1e-1,
            rtol=1e-1,
            equal_nan=not (
                arch_info.get_arch() in ["gfx950"]
            ),  # TODO: investigate nan elements for non-gfx950 arch
        )
        torch.testing.assert_close(
            torch_value_cache[slot_t, slot_b],
            triton_value_cache[slot_t, slot_b],
            atol=1e-1,
            rtol=1e-1,
            equal_nan=not (arch_info.get_arch() in ["gfx950"]),
        )
    else:
        torch.testing.assert_close(
            torch_key_cache[slot_t, :, :, slot_b, :],
            triton_key_cache[slot_t, :, :, slot_b, :],
            atol=1e-1,
            rtol=1e-1,
            equal_nan=not (arch_info.get_arch() in ["gfx950"]),
        )
        torch.testing.assert_close(
            torch_value_cache[slot_t, :, :, slot_b],
            triton_value_cache[slot_t, :, :, slot_b],
            atol=1e-1,
            rtol=1e-1,
            equal_nan=not (arch_info.get_arch() in ["gfx950"]),
        )

    torch.testing.assert_close(
        torch_key_cache,
        triton_key_cache,
        atol=1e-1,
        rtol=1e-1,
        equal_nan=not (arch_info.get_arch() in ["gfx950"]),
    )
    torch.testing.assert_close(
        torch_value_cache,
        triton_value_cache,
        atol=1e-1,
        rtol=1e-1,
        equal_nan=not (arch_info.get_arch() in ["gfx950"]),
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
@pytest.mark.parametrize("cache_flash", [True])
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
    cache_flash: bool,
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

    if cache_flash:
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
    if cache_flash:
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
        flash_layout=cache_flash,
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

    if cache_flash:
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
