import pytest
import torch
import triton

from aiter.ops.triton._triton_kernels.gemm.batched.batched_gemm_a16wfp4 import (
    _get_config,
)
from aiter.ops.triton.gemm.batched.batched_gemm_a16wfp4 import (
    batched_gemm_a16wfp4,
)
from aiter.ops.triton.utils._triton import arch_info
from op_tests.triton_tests.gemm.gemm_shapes import get_shapes

# Note this is specified by the HW and cannot be changed.
SCALE_GROUP_SIZE = 32


def generate_batched_gemm_a16wfp4_inputs(B, M, N, K, dtype, layout="TN", output=False):
    """
    Returns:
        - x: (B, M, K)
        - w: (B, N, K)
        - x_scales: (B, M, K // SCALE_GROUP_SIZE)
        - w_scales: (B, N, K // SCALE_GROUP_SIZE)
    """
    torch.manual_seed(5)
    if layout[0] == "T":
        # 34 is two packed e2m1 values 0010 which is 1.0.
        x_low = torch.randint(0, 16, (B, M, K // 2), dtype=torch.uint8, device="cuda")
        x_high = torch.randint(0, 16, (B, M, K // 2), dtype=torch.uint8, device="cuda")
    else:
        x_low = torch.randint(
            0, 16, (B, K // 2, M), dtype=torch.uint8, device="cuda"
        ).permute(0, 2, 1)
        x_high = torch.randint(
            0, 16, (B, K // 2, M), dtype=torch.uint8, device="cuda"
        ).permute(0, 2, 1)

    x = x_low | x_high << 4
    x_scales = torch.randint(
        124, 128, (B, K // SCALE_GROUP_SIZE, M), dtype=torch.uint8, device="cuda"
    )
    x_scales = x_scales.transpose(1, 2)

    x_f32 = mxfp4_to_f32(x)
    x_scales = x_scales.repeat_interleave(SCALE_GROUP_SIZE, dim=-1).to(torch.float32)
    x_scales_f32 = e8m0_to_f32(x_scales)
    x_f32 = x_f32 * x_scales_f32
    x = x_f32.to(torch.bfloat16)

    # x = torch.rand((B, M, K), dtype=torch.bfloat16, device="cuda")

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
    w_scales = torch.randint(
        124, 128, (B, K // SCALE_GROUP_SIZE, N), dtype=torch.uint8, device="cuda"
    )
    # Put w_scales at the head of a buffer whose tail is 0xFF, which is NaN in
    # e8m0. A ragged-K tile that loads scale groups past the end of w_scales
    # then turns the output NaN instead of reading whatever memory follows.
    poisoned = torch.full(
        (2 * B, K // SCALE_GROUP_SIZE, N), 0xFF, dtype=torch.uint8, device="cuda"
    )
    w_scales = poisoned[:B].copy_(w_scales)
    w_scales = w_scales.transpose(1, 2)

    y = None
    if output:
        y = torch.empty(B, M, N, device=x.device, dtype=dtype)

    return x, w, x_scales, w_scales, y


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
    w_f32 = mxfp4_to_f32(w)  # -> (B, N, K)
    # Next convert the e8m0 scales to f32.
    w_scales = w_scales.repeat_interleave(SCALE_GROUP_SIZE, dim=-1).to(
        torch.float32
    )  # -> (B, N, K)
    w_scales_f32 = e8m0_to_f32(w_scales)
    assert w_f32.shape == w_scales_f32.shape
    w_f32 = w_f32 * w_scales_f32
    return torch.bmm(x_f32, w_f32.transpose(1, 2)).to(dtype)


@pytest.mark.parametrize(
    "B, M, N, K",
    get_shapes(k_mult=32, max_size=256 * 512 * 128, batch=[1, 2, 4, 8, 16]),
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("layout", ["TN", "TT", "NN", "NT"])
def test_batched_gemm_a16wfp4(B: int, M: int, N: int, K: int, layout, dtype):
    if not (arch_info.is_fp4_avail()):
        pytest.skip("MXFP4 not supported on this architecture")

    torch.cuda.empty_cache()  # Helps avoid hangs in large tests

    x, w, _x_scales, w_scales, out = generate_batched_gemm_a16wfp4_inputs(
        B, M, N, K, dtype, layout=layout, output=True
    )

    torch_out = run_torch(x, w, w_scales, dtype).to(dtype)

    batched_gemm_a16wfp4(x, w, w_scales, dtype, out, transpose_bm=False, prequant=True)

    torch.testing.assert_close(torch_out, out)


# K = 320 is not a multiple of the BLOCK_SIZE_K (256) that the gfx950 table gives for
# these shapes, so the second and last K block is masked. With as many K blocks as
# pipeline stages, no earlier iteration has written the LDS buffer that block uses.
@pytest.mark.parametrize("M", [16, 32])
def test_batched_gemm_a16wfp4_masked_k_tail_ignores_stale_lds(M: int):
    """The masked K tail must read `other`, not LDS left behind by an earlier kernel."""
    if not (arch_info.is_fp4_avail()):
        pytest.skip("MXFP4 not supported on this architecture")

    B, N, K, dtype = 4, 512, 320, torch.bfloat16
    config, _ = _get_config(M, N, K // 2)
    if arch_info.get_arch() == "gfx950":
        # The masked tail is reached before every pipeline stage has written LDS.
        assert config["NUM_KSPLIT"] == 1
        assert 2 <= triton.cdiv(K, config["BLOCK_SIZE_K"]) <= config["num_stages"]
    # Same config on an even K (no masked lanes) with all-0xFF weights: e8m0 NaN
    # scales fill the LDS buffers that the masked tail of the real GEMM maps to.
    Bp, Mp, Kp = 16, 512, 3 * config["BLOCK_SIZE_K"]
    nan_x = torch.ones((Bp, Mp, Kp), dtype=dtype, device="cuda")
    nan_w = torch.full((Bp, N, Kp // 2), 0xFF, dtype=torch.uint8, device="cuda")
    nan_w_scales = torch.full(
        (Bp, Kp // SCALE_GROUP_SIZE, N), 0xFF, dtype=torch.uint8, device="cuda"
    ).transpose(1, 2)
    nan_out = torch.empty((Bp, Mp, N), dtype=dtype, device="cuda")
    x, w, _x_scales, w_scales, out = generate_batched_gemm_a16wfp4_inputs(
        B, M, N, K, dtype, layout="TN", output=True
    )
    torch_out = run_torch(x, w, w_scales, dtype).to(dtype)
    for _ in range(3):
        batched_gemm_a16wfp4(
            nan_x,
            nan_w,
            nan_w_scales,
            dtype,
            nan_out,
            config=dict(config),
            transpose_bm=False,
            prequant=True,
        )
        batched_gemm_a16wfp4(
            x,
            w,
            w_scales,
            dtype,
            out,
            config=dict(config),
            transpose_bm=False,
            prequant=True,
        )
        torch.testing.assert_close(torch_out, out)


def test_batched_gemm_a16wfp4_fake_honors_transpose_bm():
    """Regression: the fake must match the real kernel's allocation
    branch on ``transpose_bm`` (lines 100-103 of batched_gemm_a16wfp4.py).

    Pre-fix the fake returned ``(Bx, M, N)`` regardless of
    ``transpose_bm``, so under ``torch.compile`` AOTAutograd specialized
    the unbacked SymInt ``M`` of any downstream consumer to the static
    ``Bx`` -- silently producing wrong output (or a GPU memory access
    fault) on cudagraph replay at any other ``M``. Post-fix the fake
    returns ``(M, Bx, N)`` when ``transpose_bm=True``, matching the real
    kernel.

    Pure meta-tensor test: no GPU, no FP4 hardware, no kernel launch,
    no ``torch.compile`` trace -- testing the fake function in isolation
    is the necessary and sufficient condition for the downstream graph
    to be correct.
    """
    from aiter.ops.triton.gemm.batched.batched_gemm_a16wfp4 import (
        batched_gemm_a16wfp4_fake_tensor,
    )

    B, M, N, K = 16, 7, 512, 128

    x = torch.empty((B, M, K), dtype=torch.bfloat16, device="meta")
    w = torch.empty((B, N, K // 2), dtype=torch.uint8, device="meta")
    w_scales = torch.empty((B, N, K // 32), dtype=torch.uint8, device="meta")

    out_t = batched_gemm_a16wfp4_fake_tensor(
        x, w, w_scales, dtype=torch.bfloat16, transpose_bm=True
    )
    assert out_t.shape == (
        M,
        B,
        N,
    ), f"transpose_bm=True must return (M, B, N); got {tuple(out_t.shape)}"

    out_f = batched_gemm_a16wfp4_fake_tensor(
        x, w, w_scales, dtype=torch.bfloat16, transpose_bm=False
    )
    assert out_f.shape == (
        B,
        M,
        N,
    ), f"transpose_bm=False must return (B, M, N); got {tuple(out_f.shape)}"
