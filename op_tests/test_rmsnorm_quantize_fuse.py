import torch
import pytest 
from aiter.test_common import (
    checkAllclose,
)

from aiter import (
    rmsnorm2d_fwd,
    silu_and_mul,
)

from aiter.ops.quant import per_tensor_quant_hip

from aiter.ops.triton.fused_fp8_quant import (
    fused_rms_fp8_per_tensor_static_quant,
    fused_silu_mul_fp8_per_tensor_static_quant,
)
import aiter as rocm_aiter

rocm_aiter_fp8_dtype = rocm_aiter.dtypes.fp8

def rmsnorm2d_fwd_(
    x: torch.Tensor, weight: torch.Tensor, eps: float, dim: int
) -> torch.Tensor:
    ori_shape = x.shape
    x = x.reshape(-1, dim)
    return rmsnorm2d_fwd(x, weight, eps).view(ori_shape)


def rmsnorm_fp8_quantization_ref(x, w, x_scale, eps, n, rocm_fp8_dtype):
    rms_out = rmsnorm2d_fwd_(x, w, eps, n)
    quant_out, _ = per_tensor_quant_hip(rms_out, x_scale, rocm_fp8_dtype)
    return quant_out, rms_out


def triton_rmsnorm_fp8_quantization_fuse(x, w, x_scale, eps, rocm_fp8_dtype):
    quant_out, rms_out, _, _ = fused_rms_fp8_per_tensor_static_quant(x, w, eps, x_scale, None, None, eps, 
                                                                     dtype_quant=rocm_fp8_dtype, 
                                                                     res1=None,
                                                                     output_unquantized_inp1=True,
                                                                     intermediate_convert_to_inp1_type=True)
    return quant_out, rms_out


@pytest.mark.parametrize("m, n", [
    (m, n) 
    for m in [1, 2, 4, 8, 256, 1024, 8192]
    for n in [128, 4096, 8192]
])
def test_rmsnorm_quant_fuse(m, n):
    eps = 0.0001
    rocm_fp8_dtype = rocm_aiter_fp8_dtype

    x_shape = (m, n)
    dtype = torch.bfloat16
    x = torch.randn(x_shape, dtype=dtype, device='cuda')
    w = torch.ones(n, dtype=dtype).cuda()

    DTYPE_MAX = (
        torch.finfo(rocm_fp8_dtype).max
        if torch.is_floating_point(x)
        else torch.iinfo(rocm_fp8_dtype).max
    )

    # calculate the correct scale value
    rms_out = rmsnorm2d_fwd_(x, w, eps, n)
    rms_out_abs = torch.abs(rms_out)
    rms_out_abs_max = torch.max(rms_out_abs)
    scale_val = rms_out_abs_max / DTYPE_MAX
    x_scale = torch.tensor((scale_val), dtype=torch.float32, device='cuda')

    fp8_x_ref, rms_out_ref = rmsnorm_fp8_quantization_ref(x, w, x_scale, eps, n, rocm_fp8_dtype)
    fp8_x, rms_out = triton_rmsnorm_fp8_quantization_fuse(x, w, x_scale, eps, rocm_fp8_dtype)

    checkAllclose(rms_out, rms_out_ref)
    checkAllclose(fp8_x.to(torch.float32), fp8_x_ref.to(torch.float32))


def silu_mul_fp8_quantization_ref(x, x_scale, rocm_fp8_dtype):
    m, n2 = x.shape
    assert n2 % 2 == 0
    n = n2 // 2
    silu_out = torch.empty((m, n), dtype=x.dtype, device=x.device)
    silu_and_mul(silu_out, x)
    quant_out, _ = per_tensor_quant_hip(silu_out, x_scale, rocm_fp8_dtype)
    return quant_out


def triton_silu_mul_fp8_quantization_fuse(x, x_scale, rocm_fp8_dtype):
    quant_out = fused_silu_mul_fp8_per_tensor_static_quant(x, x_scale,
                                                      dtype_quant=rocm_fp8_dtype,
                                                      silu_convert_to_inp_type=True)
    return quant_out


@pytest.mark.parametrize("m, n", [
    (m, n) 
    for m in [1, 2, 4, 8, 256, 1024, 8192]
    for n in [128, 4096, 8192]
])
def test_silu_mul_quant_fuse(m, n):
    rocm_fp8_dtype = rocm_aiter_fp8_dtype

    x_shape = (m, 2 * n)
    dtype = torch.bfloat16
    x = torch.randn(x_shape, dtype=dtype, device='cuda')

    DTYPE_MAX = (
        torch.finfo(rocm_fp8_dtype).max
        if torch.is_floating_point(x)
        else torch.iinfo(rocm_fp8_dtype).max
    )

    # calculate the correct scale value
    silu_out = torch.empty((m, n), dtype=x.dtype, device=x.device)
    silu_and_mul(silu_out, x)
    silu_out_abs = torch.abs(silu_out)
    silu_out_abs_max = torch.max(silu_out_abs)
    scale_val = silu_out_abs_max / DTYPE_MAX
    x_scale = torch.tensor((scale_val), dtype=torch.float32, device='cuda')

    fp8_x_ref = silu_mul_fp8_quantization_ref(x, x_scale, rocm_fp8_dtype)
    fp8_x = triton_silu_mul_fp8_quantization_fuse(x, x_scale, rocm_fp8_dtype)

    checkAllclose(fp8_x.to(torch.float32), fp8_x_ref.to(torch.float32))
