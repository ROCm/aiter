import os
import sys
import torch
import pytest

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from test_rope import ref_rope_sbhd_fwd, RotateStyle
from .test_rope import generate_rope_inputs
from aiter.ops.triton.fused_kv_cache import fused_qk_rope_cosine_cache_llama
from aiter.ops.triton.utils._triton import arch_info


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
