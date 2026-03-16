import torch
import pytest
from aiter.ops.triton.gemm.batched.batched_gemm_afp4wfp4 import batched_gemm_afp4wfp4
import aiter.ops.triton.utils._triton.arch_info as arch_info
from typing import Union

# Note this is specified by the HW and cannot be changed.
SCALE_GROUP_SIZE = 32


def generate_batched_gemm_afp4wfp4_inputs(
    B: int,
    M: int,
    N: int,
    K: int,
    dtype: Union[str, torch.dtype],
    layout: str = "TN",
    output: bool = False,
):
    """
    Returns:
        - x: shape (B, M, K // 2)
        - w: shape (B, N, K // 2)
        - x_scales: shape (B, M, K // SCALE_GROUP_SIZE)
        - w_scales: shape (B, N, K // SCALE_GROUP_SIZE)
    """
    # TODO: Uncomment after pytorch adds support for manual_seed
    # torch.manual_seed(5)
    if layout[0] == "T":
        # 34 is two packed e2m1 values 0010 which is 1.0.
        x_low = torch.randint(0, 16, (B, M, K // 2), dtype=torch.uint8, device="cuda")
        x_high = torch.randint(0, 16, (B, M, K // 2), dtype=torch.uint8, device="cuda")
    else:
        # 34 is two packed e2m1 values 0010 which is 1.0.
        x_low = torch.randint(
            0, 16, (B, K // 2, M), dtype=torch.uint8, device="cuda"
        ).permute(0, 2, 1)
        x_high = torch.randint(
            0, 16, (B, K // 2, M), dtype=torch.uint8, device="cuda"
        ).permute(0, 2, 1)
    x = x_low | x_high << 4  # Doing this computation with GPU tensors results in NaN

    if layout[1] == "N":
        w_low = torch.randint(0, 16, (B, N, K // 2), dtype=torch.uint8, device="cuda")
        w_high = torch.randint(0, 16, (B, N, K // 2), dtype=torch.uint8, device="cuda")
    else:
        w_low = torch.randint(
            0, 16, (B, K // 2, N), dtype=torch.uint8, device="cuda"
        ).permute(0, 2, 1)
        w_high = torch.randint(
            0, 16, (B, K // 2, N), dtype=torch.uint8, device="cuda"
        ).permute(0, 2, 1)
    w = w_low | w_high << 4
    # Scale of 1.0 in e8m0, bias 127.
    x_scales = torch.randint(
        124, 128, (B, K // SCALE_GROUP_SIZE, M), dtype=torch.uint8, device="cuda"
    )
    w_scales = torch.randint(
        124, 128, (B, K // SCALE_GROUP_SIZE, N), dtype=torch.uint8, device="cuda"
    )
    x_scales = x_scales.transpose(1, 2)
    w_scales = w_scales.transpose(1, 2)

    y = None
    if output:
        y = torch.empty(B, M, N, device=x.device, dtype=dtype)

    return x, w, x_scales, w_scales, y


def get_x_vals():
    # Max 10 UTs per file: small (B, M, N, K)
    return [(1, 1, 32, 32), (1, 8, 64, 128), (2, 16, 128, 256), (1, 32, 256, 256),
            (2, 64, 128, 256), (1, 16, 64, 128), (2, 32, 64, 256), (1, 8, 128, 128),
            (1, 32, 32, 64), (2, 16, 256, 128)]


def mxfp4_to_f32(x):
    """
    Converts a tensor of uint8 dtype (in mxfp4 format) to fp32.
    Args:
        - x: shape (..., D) where D is the packed dimension (K // 2)
    """
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
    x_f32 = mxfp4_to_f32(x)  # -> (B, M, K)
    w_f32 = mxfp4_to_f32(w)  # -> (B, N, K)
    # Next convert the e8m0 scales to f32.
    x_scales = x_scales.repeat_interleave(SCALE_GROUP_SIZE, dim=-1).to(
        torch.float32
    )  # -> (B, M, K)
    x_scales_f32 = e8m0_to_f32(x_scales)
    x_f32 = x_f32 * x_scales_f32
    w_scales = w_scales.repeat_interleave(SCALE_GROUP_SIZE, dim=-1).to(
        torch.float32
    )  # -> (B, N, K)
    w_scales_f32 = e8m0_to_f32(w_scales)
    assert w_scales_f32.shape == w_scales.shape
    w_f32 = w_f32 * w_scales_f32
    return torch.bmm(x_f32, w_f32.transpose(1, 2)).to(dtype)


# Max 10 UTs per file
@pytest.mark.parametrize("B, M, N, K", get_x_vals())
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("layout", ["TN"])
def test_batched_gemm_afp4_wfp4(B: int, M: int, N: int, K: int, dtype, layout):
    if not (arch_info.is_fp4_avail()):
        pytest.skip("MXFP4 not supported on this architecture")

    torch.cuda.empty_cache()  # Helps avoid hangs in large tests

    x, w, x_scales, w_scales, out = generate_batched_gemm_afp4wfp4_inputs(
        B, M, N, K, dtype, layout=layout, output=True
    )

    torch_out = run_torch(x, w, x_scales, w_scales, dtype).to(dtype)
    assert torch_out.shape == (B, M, N)

    batched_gemm_afp4wfp4(x, w, x_scales, w_scales, dtype, out)

    torch.testing.assert_close(torch_out, out)
