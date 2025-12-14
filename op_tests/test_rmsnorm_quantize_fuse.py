from aiter import (
    rmsnorm2d_fwd,
    rmsnorm2d_fwd_with_add,
    rms_norm,
    layernorm2d_fwd,
    layernorm2d_fwd_with_add,
)

from aiter.ops.quant import per_tensor_quant_hip, per_tensor_quant_triton
import torch

from aiter.ops.triton.fused_fp8_quant import fused_rms_fp8_per_tensor_static_quant
import aiter as rocm_aiter
from aiter.ops.triton.rmsnorm import _rmsnorm_forward_with_add, _rmsnorm_forward

rocm_aiter_fp8_dtype = rocm_aiter.dtypes.fp8
print(f"fp8_type = {rocm_aiter_fp8_dtype}")

def rmsnorm2d_fwd_(
    x: torch.Tensor, weight: torch.Tensor, eps: float, dim: int
) -> torch.Tensor:
    ori_shape = x.shape
    x = x.reshape(-1, dim)
    return rmsnorm2d_fwd(x, weight, eps).view(ori_shape)

torch.manual_seed(1)
hidden_size = 8192
x_shape = (2, hidden_size)
dtype = torch.bfloat16
x = torch.randn(x_shape, dtype=dtype, device='cuda')
w = torch.ones(hidden_size, dtype=dtype).cuda()
eps = 0.1

DTYPE_MAX = 448.0
tmp_rms_out = rmsnorm2d_fwd_(x, w, eps, hidden_size)

abs_x = torch.abs(tmp_rms_out)
x_max = torch.max(abs_x)
scale_val = x_max / DTYPE_MAX

x_scale = torch.tensor((scale_val), dtype=torch.float32, device='cuda')

def atom_rmsnorm_quantize():
    rms_out = rmsnorm2d_fwd_(x, w, eps, hidden_size)
    quant_out, scale = per_tensor_quant_hip(rms_out, x_scale, rocm_aiter_fp8_dtype)
    return quant_out, rms_out


def atom_rmsnorm_quant_fusion():
    quant_out, rms_out, _, _ = fused_rms_fp8_per_tensor_static_quant(x, w, eps, x_scale, None, None, eps, 
                                                                     dtype_quant=rocm_aiter_fp8_dtype, 
                                                                     res1=None,
                                                                     output_unquantized_inp1=True,
                                                                     intermediate_convert_to_inp1_type=True)
    return quant_out, rms_out


def test_atom_fusion():
    x_nonfusion_out, rms_out_nonfusion = atom_rmsnorm_quantize()
    x_fusion_out, rms_out_fusion = atom_rmsnorm_quant_fusion()

    torch.testing.assert_close(rms_out_fusion, rms_out_nonfusion)
    torch.testing.assert_close(x_fusion_out.to(torch.float32), x_nonfusion_out.to(torch.float32))
