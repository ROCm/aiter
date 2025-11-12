import torch
import pytest
from aiter.ops.triton.fused_mxfp4_quant import (
    fused_flatten_mxfp4_quant,
    fused_rms_mxfp4_quant,
)
from op_tests.triton_tests.test_quant_mxfp4 import torch_dynamic_mxfp4_quant
from op_tests.triton_tests.test_gemm_afp4wfp4 import (
    mxfp4_to_f32,
    e8m0_to_f32,
    SCALE_GROUP_SIZE,
)
from op_tests.triton_tests.test_gemm_afp4wfp4 import shuffle_scales, un_shuffle_scales

torch.manual_seed(0)


def rmsnorm(input, weight, eps=1e-6):
    row_norm = input * input
    row_norm = torch.sum(row_norm, dim=-1)
    norm_factor = torch.rsqrt((row_norm / input.shape[1]) + eps).reshape(-1, 1)
    rms_norm = input * norm_factor * weight.reshape(1, -1)
    return rms_norm


def calculate_target_w_torch(x1, rms1_w, resid1, x2, rms2_w, eps=1e-6, shuffle=False):
    orig_dtype = x1.dtype
    x1 = x1.to(torch.float32)
    rms1_w = rms1_w.to(torch.float32)
    res1_out = None
    if resid1 is not None:
        resid1 = resid1.to(torch.float32)
        x1 = res1_out = x1 + resid1
        res1_out = res1_out.to(orig_dtype)
    x1 = rmsnorm(x1, rms1_w, eps)
    out1_fp4, out1_scale = torch_dynamic_mxfp4_quant(x1)

    out2 = None
    if x2 is not None:
        x2 = x2.to(torch.float32)
        rms2_w = rms2_w.to(torch.float32)
        out2 = rmsnorm(x2, rms2_w, eps).to(orig_dtype)

    if shuffle:
        out1_scale_pad = out1_scale
        M = out1_scale.shape[0]
        N = x1.shape[1]
        scaleM = (M + 255) // 256 * 256
        scaleN_valid = (N + 31) // 32
        scaleN = (scaleN_valid + 7) // 8 * 8
        out1_scale_pad = torch.empty(
            (scaleM, scaleN), dtype=out1_scale.dtype, device=out1_scale.device
        )
        out1_scale_pad[:M, :scaleN_valid] = out1_scale[:M, :scaleN_valid]
        out1_scale = shuffle_scales(out1_scale_pad)
        out1_scale = out1_scale.view(out1_scale.shape[0] * 32, -1)

    return (out1_fp4, out1_scale), out2, res1_out


def convert_mxfp4_to_fp32(x, x_scales):
    x_f32 = mxfp4_to_f32(x)
    x_scales = x_scales.repeat_interleave(SCALE_GROUP_SIZE, dim=1).to(torch.float32)
    x_scales_f32 = e8m0_to_f32(x_scales)[:, : x_f32.shape[1]]
    x_f32 = x_f32 * x_scales_f32
    return x_f32


def generate_fused_rms_quant_data(
    x1_shape=(32, 1536),
    x1_stride=(2112, 1),
    x2_shape=(32, 512),
    x2_stride=(2112, 1),
    inp2=False,
    res1=False,
    dtype=torch.bfloat16,
):
    x1 = torch.randn((x1_shape[0], x1_stride[0]), dtype=dtype, device="cuda")
    x1 = x1[:, : x1_shape[1]]
    x2 = None
    rms2_w = None
    if inp2:
        x2 = torch.randn((x2_shape[0], x2_stride[0]), dtype=dtype, device="cuda")
        x2 = x2[:, : x2_shape[1]]
        rms2_w = torch.randn(x2.shape[1], dtype=dtype, device="cuda")

    rms1_w = torch.randn(x1.shape[1], dtype=dtype, device="cuda")
    resid1 = None
    if res1:
        resid1 = torch.randn_like(x1, dtype=dtype, device="cuda")
    return x1, x2, rms1_w, rms2_w, resid1


@pytest.mark.parametrize("B", [1, 4, 16, 32, 1000, 10000])
@pytest.mark.parametrize("M", [32, 64])
@pytest.mark.parametrize("N", [32, 64, 128])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_flatten_quant(B: int, M: int, N: int, dtype):
    torch.cuda.empty_cache()  # Helps avoid hangs in large tests

    x = torch.randn((B, M, N), dtype=dtype, device="cuda").transpose(0, 1)

    torch_out, torch_scale = torch_dynamic_mxfp4_quant(x.flatten(1, 2))
    triton_out, triton_scale = fused_flatten_mxfp4_quant(x)

    torch.testing.assert_close(triton_scale, torch_scale)
    torch.testing.assert_close(triton_out, torch_out)


@pytest.mark.parametrize(
    "M, N1, N2, stride",
    [
        (M, N1, N2, stride)
        for M in [1, 4, 33, 64, 132, 256]  # TODO: debug for 131072
        for N1, N2, stride in [
            (200, 200, 200),
            (256, 256, 256),
            (256, 256, 2112),
        ]
    ],
)
@pytest.mark.parametrize("inp2", [True, False])
@pytest.mark.parametrize("res1", [True, False])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("shuffle", [True, False])
@pytest.mark.parametrize("scale_shuffle_padding", [True, False])
def test_fused_rms_quant(
    M: int,
    N1: int,
    N2: int,
    stride: int,
    inp2: bool,
    res1: bool,
    dtype,
    shuffle: bool,
    scale_shuffle_padding: bool,
):
    torch.cuda.empty_cache()  # Helps avoid hangs in large tests
    x1, x2, rms1_w, rms2_w, resid1 = generate_fused_rms_quant_data(
        x1_shape=(M, N1),
        x2_shape=(M, N2),
        x1_stride=(stride, 1),
        x2_stride=(stride, 1),
        inp2=inp2,
        res1=res1,
        dtype=dtype,
    )
    (x1_fp4_torch, x1_scales_torch), x2_torch, res1_out_torch = (
        calculate_target_w_torch(x1, rms1_w, resid1, x2, rms2_w, shuffle=shuffle)
    )

    (x1_fp4_triton, x1_scales_triton), x2_triton, res1_out_triton = (
        fused_rms_mxfp4_quant(
            x1,
            rms1_w,
            1e-6,
            x2,
            rms2_w,
            1e-6,
            resid1,
            shuffle=shuffle,
            scale_shuffle_padding=scale_shuffle_padding,
        )
    )

    if shuffle:
        x1_scales_triton = un_shuffle_scales(
            x1_scales_triton.view(x1_scales_triton.shape[0] // 32, -1)
        )
        x1_scales_torch = un_shuffle_scales(
            x1_scales_torch.view(x1_scales_torch.shape[0] // 32, -1)
        )

    scaleN_valid = (N1 + 31) // 32
    x1_scales_triton = x1_scales_triton[:M, :scaleN_valid]
    x1_scales_torch = x1_scales_torch[:M, :scaleN_valid]

    if x2_triton is not None:
        torch.testing.assert_close(x2_torch, x2_triton)

    if res1_out_triton is not None:
        torch.testing.assert_close(res1_out_torch, res1_out_triton)

    res_fp32_torch = convert_mxfp4_to_fp32(x1_fp4_torch, x1_scales_torch)
    res_fp32_triton = convert_mxfp4_to_fp32(x1_fp4_triton, x1_scales_triton)

    torch.testing.assert_close(res_fp32_torch, res_fp32_triton)
