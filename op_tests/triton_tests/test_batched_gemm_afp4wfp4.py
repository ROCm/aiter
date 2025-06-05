import torch
import triton
import pytest
from aiter.ops.triton.batched_gemm_afp4wfp4 import batched_gemm_afp4wfp4

# Note this is specified by the HW and cannot be changed.
SCALE_GROUP_SIZE = 32


def generate_batched_gemm_afp4wfp4_inputs(B, M, N, K):
    torch.manual_seed(5)
    # 34 is two packed e2m1 values 0010 which is 1.0.
    x_low = torch.randint(0, 16, (B, M, K // 2), dtype=torch.uint8)
    x_high = torch.randint(0, 16, (B, M, K // 2), dtype=torch.uint8)
    x = x_low | x_high << 4  # Doing this computation with GPU tensors results in NaN
    x = x.to(device="cuda")
    w_low = torch.randint(0, 16, (B, N, K // 2), dtype=torch.uint8, device="cuda")
    w_high = torch.randint(0, 16, (B, N, K // 2), dtype=torch.uint8, device="cuda")
    w = w_low | w_high << 4
    w = w.transpose(1, 2)
    # Scale of 1.0 in e8m0, bias 127.
    x_scales = torch.randint(
        124, 128, (B, K // SCALE_GROUP_SIZE, M), dtype=torch.uint8, device="cuda"
    )
    w_scales = torch.randint(
        124, 128, (B, K // SCALE_GROUP_SIZE, N), dtype=torch.uint8, device="cuda"
    )
    x_scales = x_scales.transpose(1, 2)
    w_scales = w_scales.transpose(1, 2)

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
    # x_vals = [(128, 1024, 4096)]
    x_vals += [(16, 16384, 3328 * 2), (128, 16384, 3328 * 2)]
    x_vals = [(256, 3584, 2112)]
    # add batch dim
    batch_sizes = [1, 2, 3, 5, 7, 8]
    num_batch_sizes = len(batch_sizes)
    x_vals_with_batch = []
    for i, (m, n, k) in enumerate(x_vals):
        b = batch_sizes[i % num_batch_sizes]
        x_vals_with_batch.append((b, m, n, k))
    return x_vals_with_batch


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
    x_f32 = 2 ** ((x - 127).to(torch.float32))
    x_f32[x_f32 == 128] = float("nan")
    return x_f32


def run_torch(x, w, x_scales, w_scales, dtype):
    # First convert the x and w inputs to f32.
    x_f32 = mxfp4_to_f32(x)
    w_f32 = mxfp4_to_f32(w.transpose(-2, -1))
    w_f32 = w_f32.transpose(-2, -1)
    # Next convert the e8m0 scales to f32.
    x_scales = x_scales.repeat_interleave(SCALE_GROUP_SIZE, dim=-1).to(torch.float32)
    x_scales_f32 = e8m0_to_f32(x_scales)
    x_f32 = x_f32 * x_scales_f32
    w_scales = w_scales.repeat_interleave(SCALE_GROUP_SIZE, dim=-1).to(torch.float32)
    w_scales_f32 = e8m0_to_f32(w_scales)
    w_f32 = w_f32 * w_scales_f32.transpose(-2, -1)
    return torch.bmm(x_f32, w_f32).to(dtype)


@pytest.mark.parametrize("B, M, N, K", get_x_vals())
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_batched_gemm_afp4_wfp4(B: int, M: int, N: int, K: int, dtype):
    if triton.runtime.driver.active.get_current_target().arch not in ("gfx950"):
        pytest.skip("MXFP4 not supported on this architecture")

    x, w, x_scales, w_scales = generate_batched_gemm_afp4wfp4_inputs(B, M, N, K)
    out = torch.empty(B, x.shape[1], w.shape[2], device=x.device, dtype=dtype)

    torch_out = run_torch(x, w, x_scales, w_scales, dtype).to(dtype)

    batched_gemm_afp4wfp4(x, w, out, x_scales, w_scales, dtype)

    torch.testing.assert_close(torch_out, out)
