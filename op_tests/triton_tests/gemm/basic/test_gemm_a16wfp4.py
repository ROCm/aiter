import pytest
import torch

from aiter.ops.shuffle import shuffle_weight
from aiter.ops.triton.gemm.basic.gemm_a16wfp4 import (
    gemm_a16wfp4,
    gemm_a16wfp4_preshuffle,
)
from aiter.ops.triton.utils._triton import arch_info
from aiter.ops.triton.utils.shuffle import shuffle_scale_gemm
from op_tests.triton_tests.gemm.gemm_shapes import get_shapes

# Note this is specified by the HW and cannot be changed.
SCALE_GROUP_SIZE = 32


def generate_gemm_a16wfp4_inputs(
    M: int,
    N: int,
    K: int,
    output: bool,
    atomic_add: bool,
    dtype: bool,
    layout: str = "TN",
    shuffle: bool = False,
):
    torch.manual_seed(5)
    # 34 is two packed e2m1 values 0010 which is 1.0.
    if layout[0] == "T":
        x_low = torch.randint(0, 16, (M, K // 2), dtype=torch.uint8, device="cuda")
        x_high = torch.randint(0, 16, (M, K // 2), dtype=torch.uint8, device="cuda")
    else:
        x_low = torch.randint(0, 16, (K // 2, M), dtype=torch.uint8, device="cuda").T
        x_high = torch.randint(0, 16, (K // 2, M), dtype=torch.uint8, device="cuda").T
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
    if layout[1] == "N":
        w_low = torch.randint(0, 16, (N, K // 2), dtype=torch.uint8, device="cuda")
        w_high = torch.randint(0, 16, (N, K // 2), dtype=torch.uint8, device="cuda")
    else:
        w_low = torch.randint(0, 16, (K // 2, N), dtype=torch.uint8, device="cuda").T
        w_high = torch.randint(0, 16, (K // 2, N), dtype=torch.uint8, device="cuda").T
    w = w_low | w_high << 4
    # Scale of 1.0 in e8m0, bias 127.
    w_scales = torch.randint(
        124, 128, (K // SCALE_GROUP_SIZE, N), dtype=torch.uint8, device="cuda"
    )
    w_scales = w_scales.T

    if shuffle:
        use_int4 = False
        weight_shuffle_layout = (16, 16)
        w_shuffed = shuffle_weight(
            w, layout=weight_shuffle_layout, use_int4=use_int4
        ).reshape(
            w.shape[0] // weight_shuffle_layout[0],
            w.shape[1] * weight_shuffle_layout[0],
        )

        # CDNA4-only triton kernel -> always the gfx950 scale layout.
        w_scales_shuffled = shuffle_scale_gemm(
            w_scales, arch="gfx950", preshuffle_factor=32, scale_kwidth=8
        )
    else:
        w_shuffed = w
        w_scales_shuffled = w_scales

    y = None
    if output:
        dtype = torch.float32 if atomic_add else dtype
        y = torch.zeros((M, N), device=x.device, dtype=dtype)

    return x, w, w_shuffed, x_scales, w_scales, w_scales_shuffled, y


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


@pytest.mark.parametrize("M, N, K", get_shapes(k_mult=256))
@pytest.mark.parametrize("output", [True, False])
@pytest.mark.parametrize(
    "atomic_add, shuffle, skip_reduce",
    [
        (True, False, False),
        (False, False, False),
        (False, True, False),
        (False, True, True),
    ],
)
def test_gemm_a16wfp4(
    M: int,
    N: int,
    K: int,
    output: bool,
    atomic_add: bool,
    shuffle: bool,
    skip_reduce: bool,
):
    if not (arch_info.is_fp4_avail()):
        pytest.skip("MXFP4 not supported on this architecture")
    if shuffle and (N % 16 != 0 or K % 512 != 0):
        pytest.skip("Shuffled weight and scales need N % 16 == 0 and K % 512 == 0")

    torch.cuda.empty_cache()  # Helps avoid hangs in large tests

    # TODO resolve this compilation error
    if M == 4864 and N == 8192 and K == 4160:
        pytest.skip("Skipping this config. due to compilation error.")

    dtype = torch.bfloat16
    x, w, w_triton, _, w_scales, w_scales_triton, y = generate_gemm_a16wfp4_inputs(
        M,
        N,
        K,
        output=output,
        atomic_add=atomic_add,
        dtype=dtype,
        layout="TN",
        shuffle=shuffle,
    )
    y_dtype = torch.float32 if atomic_add else dtype

    if shuffle:
        if output:
            y = gemm_a16wfp4_preshuffle(
                x,
                w_triton,
                w_scales_triton,
                prequant=True,
                dtype=y_dtype,
                y=y,
                skip_reduce=skip_reduce,
            )
        else:
            y = gemm_a16wfp4_preshuffle(
                x,
                w_triton,
                w_scales_triton,
                prequant=True,
                dtype=y_dtype,
                skip_reduce=skip_reduce,
            )
        if y.dim() == 3:
            y = torch.sum(y, dim=0).to(dtype=dtype)
    else:
        if output:
            y = gemm_a16wfp4(
                x, w_triton, w_scales_triton, atomic_add=atomic_add, dtype=y_dtype, y=y
            ).to(dtype)
        else:
            y = gemm_a16wfp4(
                x, w_triton, w_scales_triton, atomic_add=atomic_add, dtype=y_dtype
            ).to(dtype)

    torch_out = run_torch(x, w, w_scales, dtype).to(dtype)

    torch.testing.assert_close(torch_out, y)


def test_gemm_a16wfp4_scale_tail_mask():
    """Regression for the b_scales K-tail mask.

    A ragged-K tile loads scale groups past the end of ``w_scales``. The
    kernel must mask that tail (``other=127``), else an out-of-bounds ``0xFF``
    e8m0 scale is NaN and ``tl.dot_scaled`` turns the whole output tile NaN,
    even though the masked data is zero (``0 * NaN = NaN``). The ordinary test
    generator lays scale rows back to back, so an unmasked tail reads the next
    row's valid scales and the bug stays hidden. Here ``w_scales`` sits at the
    head of a buffer whose per-row K tail is ``0xFF``, so a missing mask fails
    deterministically instead of reading whatever memory follows.
    """
    if not arch_info.is_fp4_avail():
        pytest.skip("MXFP4 not supported on this architecture")

    # The CI flake shape. 2 * K / 32 = 344 real scale columns, which is not a
    # multiple of the scale-tile width, so the last K iteration reads past them.
    M, N, K = 16, 4096, 11008
    dtype = torch.bfloat16

    torch.cuda.empty_cache()  # Helps avoid hangs in large tests

    x, w, w_triton, _, w_scales, _, y = generate_gemm_a16wfp4_inputs(
        M, N, K, output=True, atomic_add=True, dtype=dtype, layout="TN", shuffle=False
    )

    # Put the real scales at the head of a buffer whose K tail is 0xFF (NaN in
    # e8m0). The kernel reads the tail along the contiguous K axis, so every
    # out-of-bounds scale group lands in this row's poisoned tail.
    num_scale_k = K // SCALE_GROUP_SIZE
    pad = 64  # covers one scale tile (BLOCK_SIZE_K // 32) of over-read
    poisoned = torch.full(
        (N, num_scale_k + pad), 0xFF, dtype=torch.uint8, device="cuda"
    )
    w_scales_poisoned = poisoned[:, :num_scale_k]
    w_scales_poisoned.copy_(w_scales)

    y = gemm_a16wfp4(
        x, w_triton, w_scales_poisoned, atomic_add=True, dtype=torch.float32, y=y
    ).to(dtype)

    assert torch.isfinite(y).all(), "output has NaN/Inf: b_scales K-tail not masked"

    torch_out = run_torch(x, w, w_scales, dtype).to(dtype)
    torch.testing.assert_close(torch_out, y)
