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

torch.manual_seed(0)


def rmsnorm(input, weight, eps=1e-6):
    row_norm = input * input
    row_norm = torch.sum(row_norm, dim=-1)
    norm_factor = torch.rsqrt((row_norm / input.shape[1]) + eps).reshape(-1, 1)
    rms_norm = input * norm_factor * weight.reshape(1, -1)
    return rms_norm


def calculate_target_w_torch(mat1, rms1_w, resid1, mat2, rms2_w, eps=1e-6):
    orig_dtype = mat1.dtype
    mat1 = mat1.to(torch.float32)
    rms1_w = rms1_w.to(torch.float32)
    mat2 = mat2.to(torch.float32)
    rms2_w = rms2_w.to(torch.float32)
    res1_out = None
    if resid1 is not None:
        resid1 = resid1.to(torch.float32)
        mat1 = res1_out = mat1 + resid1
        res1_out = res1_out.to(orig_dtype)
    mat1 = rmsnorm(mat1, rms1_w, eps)
    mat2 = rmsnorm(mat2, rms2_w, eps).to(orig_dtype)
    q_fp4, q_scales = torch_dynamic_mxfp4_quant(mat1)
    return (q_fp4, q_scales), mat2, res1_out


def convert_mxfp4_to_fp32(x, x_scales):
    x_f32 = mxfp4_to_f32(x)
    x_scales = x_scales.repeat_interleave(SCALE_GROUP_SIZE, dim=1).to(torch.float32)
    x_scales_f32 = e8m0_to_f32(x_scales)[:, : x_f32.shape[1]]
    x_f32 = x_f32 * x_scales_f32
    return x_f32


def generate_fused_rms_quant_data(
    mat1_shape=(32, 1536),
    mat1_stride=(2112, 1),
    mat2_shape=(32, 512),
    mat2_stride=(2112, 1),
    residual=False,
    dtype=torch.bfloat16,
):
    mat1 = torch.randn((mat1_shape[0], mat1_stride[0]), dtype=dtype, device="cuda")
    mat1 = mat1[:, : mat1_shape[1]]

    mat2 = torch.randn((mat2_shape[0], mat2_stride[0]), dtype=dtype, device="cuda")
    mat2 = mat2[:, : mat2_shape[1]]

    rms1_w = torch.randn(mat1.shape[1], dtype=dtype, device="cuda")
    rms2_w = torch.randn(mat2.shape[1], dtype=dtype, device="cuda")
    resid1 = None
    if residual:
        resid1 = torch.randn_like(mat1, dtype=dtype, device="cuda")
    return mat1, mat2, rms1_w, rms2_w, resid1


@pytest.mark.parametrize("B", [1, 4, 16, 32, 1000, 10000])
@pytest.mark.parametrize("M", [32, 64])
@pytest.mark.parametrize("N", [32, 64, 128])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_flatten_quant(B: int, M: int, N: int, dtype):
    x = torch.randn((B, M, N), dtype=dtype, device="cuda").transpose(0, 1)

    torch_out, torch_scale = torch_dynamic_mxfp4_quant(x.flatten(1, 2))
    triton_out, triton_scale = fused_flatten_mxfp4_quant(x)

    torch.testing.assert_close(triton_scale, torch_scale)
    torch.testing.assert_close(triton_out, torch_out)


@pytest.mark.parametrize("B", [1, 32, 256])
@pytest.mark.parametrize("M", [128, 132, 2112])
@pytest.mark.parametrize("N", [32, 96])
@pytest.mark.parametrize("stride", [2112])
@pytest.mark.parametrize("skip_second", [True, False])
@pytest.mark.parametrize("residual", [True, False])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_fused_rms_quant(
    B: int, M: int, N: int, stride: int, skip_second: bool, residual: bool, dtype
):
    mat1, mat2, rms1_w, rms2_w, resid1 = generate_fused_rms_quant_data(
        mat1_shape=(B, M),
        mat2_shape=(B, N),
        mat1_stride=(stride, 1),
        mat2_stride=(stride, 1),
        residual=residual,
        dtype=dtype,
    )
    (mat1_fp4_torch, mat1_scales_torch), mat2_torch, res1_out_torch = (
        calculate_target_w_torch(mat1, rms1_w, resid1, mat2, rms2_w)
    )
    if not skip_second:
        if not residual:
            (mat1_fp4_triton, mat1_scales_triton), mat2_triton = fused_rms_mxfp4_quant(
                mat1, rms1_w, 1e-6, mat2, rms2_w, 1e-6, resid1
            )
        else:
            (mat1_fp4_triton, mat1_scales_triton), mat2_triton, res1_out_triton = (
                fused_rms_mxfp4_quant(mat1, rms1_w, 1e-6, mat2, rms2_w, 1e-6, resid1)
            )
    else:
        if not residual:
            (mat1_fp4_triton, mat1_scales_triton) = fused_rms_mxfp4_quant(
                mat1, rms1_w, 1e-6, None, None, None, None
            )
        else:
            (mat1_fp4_triton, mat1_scales_triton), res1_out_triton = (
                fused_rms_mxfp4_quant(mat1, rms1_w, 1e-6, None, None, None, resid1)
            )
    if not skip_second:
        torch.testing.assert_close(mat2_torch, mat2_triton)

    if residual:
        torch.testing.assert_close(res1_out_torch, res1_out_triton)

    res_fp32_torch = convert_mxfp4_to_fp32(mat1_fp4_torch, mat1_scales_torch)
    res_fp32_triton = convert_mxfp4_to_fp32(mat1_fp4_triton, mat1_scales_triton)

    torch.testing.assert_close(res_fp32_torch, res_fp32_triton)
