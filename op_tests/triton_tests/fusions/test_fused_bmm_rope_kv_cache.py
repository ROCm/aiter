import pytest
import torch

from aiter.ops.triton.fusions.fused_bmm_rope_kv_cache import (
    fused_fp4_bmm_rope_cat_and_cache_mla,
    fused_fp8_bmm_rope_cat_and_cache_mla,
)
from aiter.ops.triton.fusions.fused_kv_cache import (
    fused_qk_rope_cat_and_cache_mla,
)
from aiter.ops.triton.gemm.batched.batched_gemm_a16wfp4 import (
    batched_gemm_a16wfp4,
)
from aiter.ops.triton.utils._triton import arch_info
from op_tests.test_rope import RotateStyle
from op_tests.triton_tests.gemm.batched.test_batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant import (
    generate_batched_gemm_a16w8_inputs,
)
from op_tests.triton_tests.gemm.batched.test_batched_gemm_a16wfp4 import (
    generate_batched_gemm_a16wfp4_inputs,
)
from op_tests.triton_tests.quant.test_fused_fp8_quant import per_token_fp8_group_quant
from op_tests.triton_tests.rope.test_rope import generate_rope_inputs


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

    torch.manual_seed(0)
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


def ref_fp8_bmm(x, w, w_scale, group_size, dtype):
    """Torch BMM on the kernels' per-token-group FP8 quant of x: [QH, T, K] -> [T, QH, N]."""
    QH, T, K = x.shape
    xq, xs = per_token_fp8_group_quant(x.reshape(-1, K), w.dtype, group_size)
    x = (xq.float() * xs.repeat_interleave(group_size, dim=-1)[:, :K]).view(QH, T, K)
    y = torch.bmm(x, w.float().transpose(1, 2)) * w_scale
    return y.transpose(0, 1).to(dtype)


@pytest.mark.parametrize("T", [1, 2, 32, 2048])
@pytest.mark.parametrize("QH_per_KH", [16])
@pytest.mark.parametrize("KH", [1, 8])
@pytest.mark.parametrize("D", [128])
@pytest.mark.parametrize(
    "D_q_nope, D_lora", [(128, 512), (512, 192)], ids=["N128-K512", "N512-K192"]
)
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
    torch.manual_seed(0)
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
    q_nope = torch.randn_like(q_nope)

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

    ref_q_nope_out = ref_fp8_bmm(q_nope, w_k, w_k_scale, 128, dtype)

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


def _make_fp8_bmm_mla_inputs(
    T, seed=0, q_out_dtype="fp8", num_decode=0, is_neox=False, reuse_freqs=True
):
    if not arch_info.is_fp8_avail():
        pytest.skip("FP8 is not available on this device")

    q_nope, w_k, w_k_scale, _, _ = generate_batched_gemm_a16w8_inputs(
        8, T, 512, 192, torch.bfloat16, False, False, transpose_bm=True
    )
    _, _, _, _, _, positions, _, cos, sin = generate_rope_inputs(
        1,
        T,
        1,
        8,
        64,
        cached=True,
        reuse_freqs_front_part=reuse_freqs,
        nope=False,
        pos=True,
        offs=False,
        two_inputs=True,
        layout="thd",
        dtype=torch.bfloat16,
    )
    torch.manual_seed(seed)
    q_nope.normal_()
    q_nope[..., :128].mul_(0.25)
    w_k.copy_(torch.randn_like(w_k, dtype=torch.bfloat16) * 0.25)
    w_k_scale.fill_(0.5)
    q_pe = torch.randn((T, 8, 64), dtype=torch.bfloat16, device="cuda")
    k_nope = torch.randn((T, 1, 512), dtype=torch.bfloat16, device="cuda")
    k_rope = torch.randn((T, 1, 64), dtype=torch.bfloat16, device="cuda")
    # Leave cache rows on both sides untouched by the call.
    kv_cache = torch.full((T + 7, 1, 576), 0.25, dtype=w_k.dtype, device="cuda")
    return {
        "q_nope": q_nope,
        "w_k": w_k,
        "w_k_scale": w_k_scale,
        "q_pe": q_pe,
        "k_nope": k_nope,
        "k_rope": k_rope,
        "kv_cache": kv_cache,
        "slot_mapping": torch.randperm(T, device="cuda") + 3,
        "positions": positions,
        "cos": cos.roll(seed, dims=0)[:, None, None, :],
        "sin": sin.roll(seed, dims=0)[:, None, None, :],
        "group_size": 128,
        "transpose_bm": True,
        "k_scale": torch.full((), 0.75, dtype=torch.float32, device="cuda"),
        "is_neox": is_neox,
        "q_out_dtype": w_k.dtype if q_out_dtype == "fp8" else q_out_dtype,
        "num_decode_toks_for_zeros": num_decode,
    }


def _reference_fp8_bmm_mla(inputs):
    # Keep the projection in FP32 until the final output cast.
    q_nope = ref_fp8_bmm(
        inputs["q_nope"],
        inputs["w_k"],
        inputs["w_k_scale"],
        inputs["group_size"],
        torch.float32,
    )
    kv_cache = inputs["kv_cache"].clone()
    num_decode = inputs["num_decode_toks_for_zeros"]
    decode_q_pe = torch.empty(
        (num_decode, 8, 64), dtype=inputs["q_nope"].dtype, device=q_nope.device
    )
    q, decode_q_pe, k_pe, zeros = fused_qk_rope_cat_and_cache_mla(
        q_nope,
        inputs["q_pe"],
        inputs["k_nope"],
        inputs["k_rope"],
        kv_cache,
        inputs["slot_mapping"],
        inputs["positions"],
        inputs["cos"],
        inputs["sin"],
        k_scale=inputs["k_scale"],
        is_neox=inputs["is_neox"],
        num_decode_toks_for_zeros=num_decode,
        q_out_dtype=inputs["q_out_dtype"] or torch.bfloat16,
        decode_q_pe_out=decode_q_pe,
    )
    return (q, decode_q_pe, k_pe, zeros.to(inputs["q_nope"].dtype)), kv_cache


def _assert_fp8_bmm_mla_result(inputs, actual, expected, expected_cache):
    assert len(actual) == len(expected) == 4
    for result, reference in zip(actual, expected):
        assert result.shape == reference.shape
        assert result.dtype == reference.dtype
        assert result.device == reference.device

    torch.testing.assert_close(
        actual[0][..., :512].float(),
        expected[0][..., :512].float(),
        atol=1e-2,
        rtol=1e-1,
    )
    torch.testing.assert_close(
        actual[0][..., 512:].float(),
        expected[0][..., 512:].float(),
        atol=0,
        rtol=0,
    )
    for index in (1, 3):
        torch.testing.assert_close(actual[index], expected[index], atol=0, rtol=0)
    # A negative slot leaves the corresponding k_pe output row unspecified.
    valid = inputs["slot_mapping"] >= 0
    torch.testing.assert_close(actual[2][valid], expected[2][valid], atol=0, rtol=0)
    torch.testing.assert_close(
        inputs["kv_cache"].float(), expected_cache.float(), atol=0, rtol=0
    )


def _assert_same_tensor_bytes(actual, expected):
    torch.testing.assert_close(
        actual.contiguous().view(-1).view(torch.uint8),
        expected.contiguous().view(-1).view(torch.uint8),
        atol=0,
        rtol=0,
    )


@pytest.mark.parametrize("T", [6, 12, 64, 65, 1856, 11072, 16385, 32768])
def test_fused_fp8_bmm_mla_fp8_output(T):
    inputs = _make_fp8_bmm_mla_inputs(T)
    expected, expected_cache = _reference_fp8_bmm_mla(inputs)
    actual = fused_fp8_bmm_rope_cat_and_cache_mla(**inputs)
    _assert_fp8_bmm_mla_result(inputs, actual, expected, expected_cache)


@pytest.mark.parametrize(
    "T, q_out_dtype, num_decode, is_neox, reuse_freqs",
    [
        (1856, torch.bfloat16, 0, False, True),
        (1856, None, 0, False, True),
        (129, "fp8", 3, False, True),
        (129, "fp8", 0, True, True),
        (129, "fp8", 0, False, False),
    ],
    ids=["bf16", "default-dtype", "decode", "neox", "full-frequencies"],
)
def test_fused_fp8_bmm_mla_output_variants(
    T, q_out_dtype, num_decode, is_neox, reuse_freqs
):
    inputs = _make_fp8_bmm_mla_inputs(
        T,
        q_out_dtype=q_out_dtype,
        num_decode=num_decode,
        is_neox=is_neox,
        reuse_freqs=reuse_freqs,
    )
    expected, expected_cache = _reference_fp8_bmm_mla(inputs)
    actual = fused_fp8_bmm_rope_cat_and_cache_mla(**inputs)
    _assert_fp8_bmm_mla_result(inputs, actual, expected, expected_cache)


def test_fused_fp8_bmm_mla_negative_slots():
    inputs = _make_fp8_bmm_mla_inputs(129)
    inputs["slot_mapping"][0] = -1
    inputs["slot_mapping"][-1] = -1
    expected, expected_cache = _reference_fp8_bmm_mla(inputs)
    actual = fused_fp8_bmm_rope_cat_and_cache_mla(**inputs)
    _assert_fp8_bmm_mla_result(inputs, actual, expected, expected_cache)


@pytest.mark.parametrize("padded", [False, True])
def test_fused_fp8_bmm_mla_query_views(padded):
    inputs = _make_fp8_bmm_mla_inputs(1856)
    width = 256 if padded else 192
    queries = torch.empty((1856, 8, width), dtype=torch.bfloat16, device="cuda")
    queries[..., :192].copy_(inputs["q_nope"].transpose(0, 1))
    inputs["q_nope"] = queries[..., :192].transpose(0, 1)
    if padded:
        queries[..., 192:].copy_(inputs["q_pe"])
        inputs["q_pe"] = queries[..., 192:]
    before = queries.clone()
    expected, expected_cache = _reference_fp8_bmm_mla(inputs)
    actual = fused_fp8_bmm_rope_cat_and_cache_mla(**inputs)
    _assert_fp8_bmm_mla_result(inputs, actual, expected, expected_cache)
    _assert_same_tensor_bytes(queries, before)


def test_fused_fp8_bmm_mla_shape_alternation_preserves_inputs_and_outputs():
    retained = []
    for seed, T in enumerate((64, 65, 1856, 64)):
        inputs = _make_fp8_bmm_mla_inputs(T, seed=seed)
        before = {
            name: value.clone()
            for name, value in inputs.items()
            if isinstance(value, torch.Tensor) and name != "kv_cache"
        }
        expected, expected_cache = _reference_fp8_bmm_mla(inputs)
        actual = fused_fp8_bmm_rope_cat_and_cache_mla(**inputs)
        _assert_fp8_bmm_mla_result(inputs, actual, expected, expected_cache)
        for name, value in before.items():
            _assert_same_tensor_bytes(inputs[name], value)
        for previous, snapshot in retained:
            for result, saved in zip(previous, snapshot):
                _assert_same_tensor_bytes(result, saved)
        retained.append((actual, tuple(result.clone() for result in actual)))


@pytest.mark.parametrize("T", [1856, 32768])
def test_fused_fp8_bmm_mla_graph_replay_reads_current_inputs(T):
    inputs = _make_fp8_bmm_mla_inputs(T)
    fused_fp8_bmm_rope_cat_and_cache_mla(**inputs)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = fused_fp8_bmm_rope_cat_and_cache_mla(**inputs)

    for factor in (0.5, 2.0):
        inputs["q_nope"].mul_(factor)
        inputs["q_pe"].neg_()
        inputs["k_nope"].add_(0.25)
        inputs["k_rope"].mul_(factor)
        inputs["w_k"].copy_((inputs["w_k"].float() * factor).to(inputs["w_k"].dtype))
        inputs["w_k_scale"].add_(0.25)
        inputs["k_scale"].add_(0.25)
        inputs["positions"].add_(1).remainder_(T)
        inputs["slot_mapping"].copy_(inputs["slot_mapping"].roll(1))
        inputs["cos"].neg_()
        inputs["sin"].neg_()
        inputs["kv_cache"].fill_(0.25)
        expected, expected_cache = _reference_fp8_bmm_mla(inputs)
        graph.replay()
        _assert_fp8_bmm_mla_result(inputs, actual, expected, expected_cache)
