import torch
import pytest

from op_tests.test_rope import RotateStyle
from op_tests.triton_tests.rope.test_rope import generate_rope_inputs
from aiter.ops.triton.fusions.fused_kv_cache import (
    fused_qk_rope_cat_and_cache_mla,
)
from aiter.ops.triton.utils._triton import arch_info

from aiter.ops.triton.fusions.fused_bmm_rope_kv_cache import (
    fused_fp4_bmm_rope_cat_and_cache_mla,
    fused_fp8_bmm_rope_cat_and_cache_mla,
)
from op_tests.triton_tests.gemm.batched.test_batched_gemm_a16wfp4 import (
    generate_batched_gemm_a16wfp4_inputs,
)
from aiter.ops.triton.gemm.batched.batched_gemm_a16wfp4 import (
    batched_gemm_a16wfp4,
)

from op_tests.triton_tests.gemm.batched.test_batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant import (
    generate_batched_gemm_a16w8_inputs,
)
from aiter.ops.triton.gemm.batched.batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant import (
    batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant,
)


@pytest.mark.parametrize("T", [1, 2, 32, 2048])
@pytest.mark.parametrize("QH_per_KH", [16])
@pytest.mark.parametrize("KH", [1, 8])
@pytest.mark.parametrize("D", [128])  # For now, D is power of 2. D >= 16
@pytest.mark.parametrize("D_q_nope", [128])
@pytest.mark.parametrize("D_lora", [512])
@pytest.mark.parametrize("num_kv_cahce_tokens", [16384])
@pytest.mark.parametrize("rotate_style", [RotateStyle.GPTJ, RotateStyle.NEOX])
@pytest.mark.parametrize("reuse_freqs_front_part", [False, True])
@pytest.mark.parametrize("cache_dtype", [torch.bfloat16, torch.uint8])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_fused_fp4_bmm_rope_cat_and_cache_mla(
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
    if not arch_info.is_fp4_avail():
        pytest.skip("MXFP4 is not available on this device")

    _, w_k, _, w_k_scale, _ = generate_batched_gemm_a16wfp4_inputs(
        QH_per_KH * KH, T, D_lora, D_q_nope, dtype, layout="TN", output=False
    )

    _, _, _, _, _, positions, _, cos, sin = generate_rope_inputs(
        1,
        T,
        KH,
        QH_per_KH,
        D,
        cached=True,
        reuse_freqs_front_part=reuse_freqs_front_part,
        nope=False,
        pos=True,
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

    ref_q_nope_out = batched_gemm_a16wfp4(
        q_nope.transpose(0, 1),
        w_k,
        w_k_scale,
        dtype,
        transpose_bm=True,
        prequant=True,
        y_scale=None,
    )

    ref_kv_cache = kv_cache.clone()
    if cache_dtype == torch.uint8:
        ref_kv_cache = ref_kv_cache.view(cache_dtype_actual)
    ref_q, ref_decode_q_pe, ref_k_pe, ref_zeros = fused_qk_rope_cat_and_cache_mla(
        ref_q_nope_out,
        q_pe,
        k_lora,
        k_pe,
        ref_kv_cache,
        slot_mapping,
        positions,
        cos,
        sin,
        k_scale=k_scale,
        is_neox=(rotate_style == RotateStyle.NEOX),
        num_decode_toks_for_zeros=T,
        apply_scale=(k_pe.dtype != kv_cache.dtype),
        decode_q_pe_out=None,
        k_pe_out=None,
    )
    ref_kv_cache = ref_kv_cache.view(kv_cache_og_dtype)

    triton_kv_cache = kv_cache.clone()
    if cache_dtype == torch.uint8:
        triton_kv_cache = triton_kv_cache.view(cache_dtype_actual)
    triton_q, triton_decode_q_pe, triton_k_pe, triton_zeros = (
        fused_fp4_bmm_rope_cat_and_cache_mla(
            q_nope.transpose(0, 1),
            w_k,
            w_k_scale,
            q_pe,
            k_lora,
            k_pe,
            triton_kv_cache,
            slot_mapping,
            positions,
            cos,
            sin,
            y=None,
            transpose_bm=True,
            prequant=True,
            y_scale=None,
            k_scale=k_scale,
            is_neox=(rotate_style == RotateStyle.NEOX),
            q_out_dtype=None,
            num_decode_toks_for_zeros=T,
        )
    )
    triton_kv_cache = triton_kv_cache.view(kv_cache_og_dtype)

    torch.testing.assert_close(ref_q, triton_q, atol=1e-1, rtol=1e-1)
    torch.testing.assert_close(
        ref_decode_q_pe, triton_decode_q_pe, atol=1e-1, rtol=1e-1
    )
    torch.testing.assert_close(ref_k_pe, triton_k_pe, atol=1e-1, rtol=1e-1)
    torch.testing.assert_close(ref_zeros, triton_zeros, atol=0.1, rtol=0.1)

    if cache_dtype == torch.uint8:
        ref_kv_cache = ref_kv_cache.view(cache_dtype_actual).to(dtype)
        triton_kv_cache = triton_kv_cache.view(cache_dtype_actual).to(dtype)

    torch.testing.assert_close(
        ref_kv_cache[slot_mapping, :, :],
        triton_kv_cache[slot_mapping, :, :],
        atol=1e-1,
        rtol=1e-1,
    )

    torch.testing.assert_close(ref_kv_cache, triton_kv_cache, atol=1e-1, rtol=1e-1)


@pytest.mark.parametrize("T", [1, 2, 32, 2048])
@pytest.mark.parametrize("QH_per_KH", [16])
@pytest.mark.parametrize("KH", [1, 8])
@pytest.mark.parametrize("D", [128])
@pytest.mark.parametrize("D_q_nope", [128])
@pytest.mark.parametrize("D_lora", [512])
@pytest.mark.parametrize("num_kv_cahce_tokens", [16384])
@pytest.mark.parametrize("rotate_style", [RotateStyle.GPTJ, RotateStyle.NEOX])
@pytest.mark.parametrize("reuse_freqs_front_part", [False, True])
@pytest.mark.parametrize("cache_dtype", [torch.bfloat16, torch.uint8])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_fused_fp8_bmm_rope_cat_and_cache_mla(
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
    if not arch_info.is_fp8_avail():
        pytest.skip("MXFP8 is not available on this device")

    QH = QH_per_KH * KH

    q_nope, w_k, w_k_scale, _, _ = generate_batched_gemm_a16w8_inputs(
        QH,
        T,
        D_q_nope,
        D_lora,
        dtype,
        has_bias=False,
        output=False,
        layout="TN",
        transpose_bm=True,
    )

    _, _, _, _, _, positions, _, cos, sin = generate_rope_inputs(
        1,
        T,
        KH,
        QH_per_KH,
        D,
        cached=True,
        reuse_freqs_front_part=reuse_freqs_front_part,
        nope=False,
        pos=True,
        offs=False,
        two_inputs=True,
        layout="thd",
        dtype=dtype,
    )

    q_pe = torch.randn((T, QH, D), dtype=dtype, device="cuda") * 0.1
    k_lora = torch.randn((T, KH, D_q_nope), dtype=dtype, device="cuda") / (
        20 if cache_dtype == torch.uint8 else 1
    )
    k_pe = torch.randn((T, KH, D), dtype=dtype, device="cuda") / (
        20 if cache_dtype == torch.uint8 else 1
    )

    if cache_dtype == torch.uint8:
        if arch_info.get_arch() in ["gfx950"]:
            cache_dtype_actual = torch.float8_e4m3fn
        else:
            cache_dtype_actual = torch.float8_e4m3fnuz

    kv_cache = torch.zeros(
        (num_kv_cahce_tokens, KH, D_q_nope + D), dtype=cache_dtype, device="cuda"
    )

    if cache_dtype == torch.uint8:
        k_scale = torch.randn([1], dtype=torch.float32, device="cuda")[0]
    else:
        k_scale = torch.ones([1], dtype=torch.float32, device="cuda")[0]

    slot_mapping = torch.randperm(T, device="cuda")
    kv_cache_og_dtype = kv_cache.dtype

    ref_q_nope_out = (
        batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant(
            q_nope,
            w_k,
            w_k_scale,
            group_size=128,
            bias=None,
            dtype=dtype,
            YQ=None,
            transpose_bm=True,
        )
    )

    ref_kv_cache = kv_cache.clone()
    if cache_dtype == torch.uint8:
        ref_kv_cache = ref_kv_cache.view(cache_dtype_actual)

    ref_q, ref_decode_q_pe, ref_k_pe, ref_zeros = fused_qk_rope_cat_and_cache_mla(
        ref_q_nope_out,
        q_pe,
        k_lora,
        k_pe,
        ref_kv_cache,
        slot_mapping,
        positions,
        cos,
        sin,
        k_scale=k_scale,
        is_neox=(rotate_style == RotateStyle.NEOX),
        num_decode_toks_for_zeros=T,
        apply_scale=(k_pe.dtype != kv_cache.dtype),
        q_out=None,
        decode_q_pe_out=None,
        k_pe_out=None,
    )
    ref_kv_cache = ref_kv_cache.view(kv_cache_og_dtype)

    triton_kv_cache = kv_cache.clone()
    if cache_dtype == torch.uint8:
        triton_kv_cache = triton_kv_cache.view(cache_dtype_actual)

    triton_q, triton_decode_q_pe, triton_k_pe, triton_zeros = (
        fused_fp8_bmm_rope_cat_and_cache_mla(
            q_nope,
            w_k,
            w_k_scale,
            q_pe,
            k_lora,
            k_pe,
            triton_kv_cache,
            slot_mapping,
            positions,
            cos,
            sin,
            group_size=128,
            transpose_bm=True,
            config=None,
            k_scale=k_scale,
            is_neox=(rotate_style == RotateStyle.NEOX),
            q_out_dtype=dtype,
            num_decode_toks_for_zeros=T,
        )
    )
    triton_kv_cache = triton_kv_cache.view(kv_cache_og_dtype)

    torch.testing.assert_close(ref_q, triton_q, atol=1e-1, rtol=1e-1)
    torch.testing.assert_close(
        ref_decode_q_pe, triton_decode_q_pe, atol=1e-1, rtol=1e-1
    )
    torch.testing.assert_close(ref_k_pe, triton_k_pe, atol=1e-1, rtol=1e-1)
    torch.testing.assert_close(ref_zeros, triton_zeros, atol=0.1, rtol=0.1)

    if cache_dtype == torch.uint8:
        ref_kv_cache = ref_kv_cache.view(cache_dtype_actual).to(dtype)
        triton_kv_cache = triton_kv_cache.view(cache_dtype_actual).to(dtype)

    torch.testing.assert_close(
        ref_kv_cache[slot_mapping, :, :],
        triton_kv_cache[slot_mapping, :, :],
        atol=1e-1,
        rtol=1e-1,
    )

    torch.testing.assert_close(ref_kv_cache, triton_kv_cache, atol=1e-1, rtol=1e-1)
