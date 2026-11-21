import pytest
import torch

from aiter.ops.triton.gemm.batched.batched_gemm_afp8wfp8 import batched_gemm_afp8wfp8
from aiter.ops.triton.utils._triton import arch_info
from aiter.test_common import checkAllclose

# tl.dot_scaled reads one E8M0 scale for each 32 values along K.
SCALE_GROUP_SIZE = 32

# A K-long dot product of signed E4M3 values cancels to near zero on a few
# outputs, where the kernel and the fp32 reference differ by more than any
# per-element atol. As in test_gemm_afp8wfp8, check the outlier ratio;
# catastrophic_check still fails on NaN, Inf or a large delta.
TOL_ERR_RATIO = 0.05


def generate_batched_gemm_afp8wfp8_inputs(
    B: int,
    M: int,
    N: int,
    K: int,
    dtype: torch.dtype,
    layout: str = "TN",
    output: bool = False,
    w_scale_group_size: tuple[int, int] = (1, 32),
):
    """
    Returns:
        - x: E4M3, shape (B, M, K)
        - w: E4M3, shape (B, N, K)
        - x_scales: uint8 E8M0, shape (B, M, K // 32)
        - w_scales: uint8 E8M0, shape (B, ceil(N / n), K // 32) for (n, 32) blocks
        - y: shape (B, M, N) if output, else None

    layout "TN" gives row-major tensors. "MBK" gives x, x_scales and y as
    transposed views of token-major (M, B, K) and (M, B, N) buffers, which is
    how DeepSeek-V4 wo_a uses the kernel.
    """
    torch.manual_seed(5)
    group_n = w_scale_group_size[0]
    if layout == "MBK":
        x = torch.randn((M, B, K), device="cuda").to(torch.float8_e4m3fn)
        x_scales = torch.randint(
            124, 130, (M, B, K // SCALE_GROUP_SIZE), dtype=torch.uint8, device="cuda"
        )
        x, x_scales = x.transpose(0, 1), x_scales.transpose(0, 1)
    else:
        x = torch.randn((B, M, K), device="cuda").to(torch.float8_e4m3fn)
        x_scales = torch.randint(
            124, 130, (B, M, K // SCALE_GROUP_SIZE), dtype=torch.uint8, device="cuda"
        )
    w = torch.randn((B, N, K), device="cuda").to(torch.float8_e4m3fn)
    w_scales = torch.randint(
        124,
        130,
        (B, -(-N // group_n), K // SCALE_GROUP_SIZE),
        dtype=torch.uint8,
        device="cuda",
    )
    y = None
    if output:
        if layout == "MBK":
            y = torch.empty((M, B, N), dtype=dtype, device="cuda").transpose(0, 1)
        else:
            y = torch.empty((B, M, N), dtype=dtype, device="cuda")
    return x, w, x_scales, w_scales, y


def e8m0_to_f32(x):
    return torch.exp2(x.to(torch.float32) - 127)


def run_torch(x, w, x_scales, w_scales, dtype, group_n):
    x_f32 = x.float() * e8m0_to_f32(x_scales).repeat_interleave(SCALE_GROUP_SIZE, -1)
    w_s = e8m0_to_f32(w_scales).repeat_interleave(SCALE_GROUP_SIZE, -1)
    w_s = w_s.repeat_interleave(group_n, -2)[:, : w.shape[1]]
    w_f32 = w.float() * w_s
    return torch.bmm(x_f32, w_f32.transpose(1, 2)).to(dtype)


def get_x_vals():
    # (B, M, N, K): odd M, N and K tails, then the DeepSeek-V4.1 wo_a shape.
    x_vals = [
        (1, 1, 32, 64),
        (2, 7, 80, 96),
        (3, 33, 96, 160),
        (4, 300, 512, 1024),
        (1, 513, 2048, 512),
    ]
    x_vals += [(8, m, 1024, 4096) for m in (1, 17, 192, 2234)]
    return x_vals


@pytest.mark.parametrize("B, M, N, K", get_x_vals())
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("layout", ["TN", "MBK"])
@pytest.mark.parametrize("w_scale_group_size", [(1, 32), (32, 32)])
def test_batched_gemm_afp8wfp8(B, M, N, K, dtype, layout, w_scale_group_size):
    if arch_info.get_arch() != "gfx1250":
        pytest.skip("batched_gemm_afp8wfp8 has configs only for gfx1250")

    x, w, x_scales, w_scales, y = generate_batched_gemm_afp8wfp8_inputs(
        B,
        M,
        N,
        K,
        dtype,
        layout=layout,
        output=True,
        w_scale_group_size=w_scale_group_size,
    )
    torch_out = run_torch(x, w, x_scales, w_scales, dtype, w_scale_group_size[0])

    out = batched_gemm_afp8wfp8(
        x, w, x_scales, w_scales, dtype, y, w_scale_group_size=w_scale_group_size
    )

    assert out.data_ptr() == y.data_ptr()
    err = checkAllclose(
        torch_out,
        out,
        atol=1e-2,
        rtol=1e-2,
        msg=f"batched afp8wfp8 B={B} M={M} N={N} K={K} {layout} {w_scale_group_size} ",
        catastrophic_check=True,
    )
    assert err < TOL_ERR_RATIO, f"{err:.2%} of elements mismatch"
