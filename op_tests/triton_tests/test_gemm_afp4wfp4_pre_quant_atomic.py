import torch
import triton
import pytest
from aiter.ops.triton.gemm_afp4wfp4_pre_quant_atomic import gemm_afp4wfp4_pre_quant
from triton.testing import runtime

# Note this is specified by the HW and cannot be changed.
SCALE_GROUP_SIZE = 32


def generate_gemm_afp4wfp4_pre_quant_inputs(M, N, K):
    torch.manual_seed(5)
    # 34 is two packed e2m1 values 0010 which is 1.0.
    x_low = torch.randint(0, 16, (M, K // 2), dtype=torch.uint8, device="cuda")
    x_high = torch.randint(0, 16, (M, K // 2), dtype=torch.uint8, device="cuda")
    x = x_low | x_high << 4
    x_scales = torch.randint(
        124, 128, (K // SCALE_GROUP_SIZE, M), dtype=torch.uint8, device="cuda"
    ).T

    x_f32 = mxfp4_to_f32(x)
    x_scales = x_scales.repeat_interleave(SCALE_GROUP_SIZE, dim=-1).to(torch.float32)
    x_scales_f32 = e8m0_to_f32(x_scales)
    x_f32 = x_f32 * x_scales_f32
    x = x_f32.to(torch.bfloat16)

    # x = torch.rand((B, M, K), dtype=torch.bfloat16, device="cuda")
    w_low = torch.randint(0, 16, (N, K // 2), dtype=torch.uint8, device="cuda")
    w_high = torch.randint(0, 16, (N, K // 2), dtype=torch.uint8, device="cuda")
    w = w_low | w_high << 4
    # Scale of 1.0 in e8m0, bias 127.
    w_scales = torch.randint(
        124, 128, (K // SCALE_GROUP_SIZE, N), dtype=torch.uint8, device="cuda"
    )
    w_scales = w_scales.T

    return x, w, x_scales, w_scales


def get_x_vals():

    x_vals = [(1024 * v, 1024 * v, 1024 * v) for v in range(1, 9)]
    x_vals += [(4864, 4096, 8192), (9728, 8192, 65536), (4864, 8192, 4160)]
    x_vals += [
        (1, 1280, 8192),
        (32, 1280, 8192),
        (64, 1280, 8192),
        (128, 1280, 8192),
        (192, 1280, 8192),
        (256, 1280, 8192),
        (320, 1280, 8192),
        (512, 1280, 8192),
        (1024, 1280, 8192),
        (2048, 1280, 8192),
        (4096, 1280, 8192),
        (8192, 1280, 8192),
        (16384, 1280, 8192),
        (1, 8192, 1024),
        (32, 8192, 1024),
        (64, 8192, 1024),
        (128, 8192, 1024),
        (192, 8192, 1024),
        (256, 8192, 1024),
        (320, 8192, 1024),
        (512, 8192, 1024),
        (1024, 8192, 1024),
        (2048, 8192, 1024),
        (4096, 8192, 1024),
        (8192, 8192, 1024),
        (16384, 8192, 1024),
    ]
    x_vals += [(2 ** (v - 1), 4096 * v, 4096 * v) for v in range(1, 6)]
    x_vals += [(16, 16384, 3328 * 2), (128, 16384, 3328 * 2)]
    x_vals += [(32, 512, 7168)]
    x_vals += [(1, 1, 32)]  # minimal case
    return x_vals


def mxfp4_to_f32(x):
    # 2 because we pack fp4 in uint8.
    x = x.repeat_interleave(2, dim=-1)
    x[..., ::2] = x[..., ::2] & 0xF
    x[..., 1::2] = x[..., 1::2] >> 4
    mxfp4_list = [
        0.0,
        0.5,
        1.0,
        1.5,
        2.0,
        3.0,
        4.0,
        6.0,
        -0.0,
        -0.5,
        -1.0,
        -1.5,
        -2.0,
        -3.0,
        -4.0,
        -6.0,
    ]
    mxfp4_in_f32 = torch.tensor(mxfp4_list, dtype=torch.float32, device="cuda")
    return mxfp4_in_f32[x.long()]


def e8m0_to_f32(x):
    x_f32 = 2 ** (x.to(torch.float32) - 127)
    x_f32[x_f32 == 128] = float("nan")
    return x_f32


def run_torch(x, w, w_scales, dtype):
    # First convert the x and w inputs to f32.
    x_f32 = x.to(torch.float32)
    w_f32 = mxfp4_to_f32(w)
    # Next convert the e8m0 scales to f32.
    w_scales = w_scales.repeat_interleave(SCALE_GROUP_SIZE, dim=-1).to(torch.float32)
    w_scales_f32 = e8m0_to_f32(w_scales)
    assert w_f32.shape == w_scales_f32.shape
    w_f32 = w_f32 * w_scales_f32
    return torch.mm(x_f32, w_f32.T).to(dtype)


@pytest.mark.parametrize("M, N, K", get_x_vals())
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("output", [True, False])
def test_gemm_afp4_wfp4_pre_quant(M: int, N: int, K: int, dtype, output: bool):
    if triton.runtime.driver.active.get_current_target().arch not in ("gfx950"):
        pytest.skip("MXFP4 not supported on this architecture")

    # TODO resolve this compilation error
    if M == 4864 and N == 8192 and K == 4160:
        pytest.skip("Skipping this config. due to compilation error.")
    
    x, w, _, w_scales = generate_gemm_afp4wfp4_pre_quant_inputs(M, N, K)

    if output:
        out = torch.zeros((M, N), device=x.device, dtype=torch.float32)
        out = gemm_afp4wfp4_pre_quant(x, w, w_scales, torch.float32, out).to(dtype)
    else:
        out = gemm_afp4wfp4_pre_quant(x, w, w_scales, torch.float32).to(dtype)

    torch_out = run_torch(x, w, w_scales, dtype).to(dtype)

    torch.testing.assert_close(torch_out, out)
