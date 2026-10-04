# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
import functools

import pytest
import torch
import triton

from aiter.ops.shuffle import shuffle_scale, shuffle_weight
from aiter.ops.triton._triton_kernels.gemm.basic.gemm_afp4wfp4 import (
    _get_config as _get_afp4wfp4_config,
)
from aiter.ops.triton.gemm.basic.gemm_afp4wfp4 import (
    gemm_afp4wfp4 as triton_gemm_afp4wfp4,
)
from aiter.ops.triton.gemm.basic.gemm_afp4wfp4 import (
    gemm_afp4wfp4_preshuffle,
)
from aiter.ops.triton.utils._triton import arch_info
from aiter.ops.triton.utils.shuffle import shuffle_scale_gemm
from aiter.ops.triton.utils.types import str_to_torch_dtype

DEVICE_ARCH = arch_info.get_arch()

pytestmark = pytest.mark.skipif(
    not arch_info.is_fp4_avail(), reason="MXFP4 not supported on this architecture"
)


# Note this is specified by the HW and cannot be changed.
SCALE_GROUP_SIZE = 32


def generate_gemm_afp4wfp4_inputs(
    M,
    N,
    K,
    dtype,
    layout="TN",
    output=True,
    shuffle_weight_fg=False,
    shuffle_scales_fg=False,
):
    if shuffle_weight_fg:
        assert (
            shuffle_scales_fg
        ), "weight shuffling is only supported with scale shuffling"

    torch.manual_seed(5)
    if isinstance(dtype, str):
        dtype = str_to_torch_dtype[dtype]

    if layout[0] == "T":
        # 34 is two packed e2m1 values 0010 which is 1.0.
        x_low = torch.randint(0, 16, (M, K // 2), dtype=torch.uint8)
        x_high = torch.randint(0, 16, (M, K // 2), dtype=torch.uint8)
    else:
        x_low = torch.randint(0, 16, (K // 2, M), dtype=torch.uint8).T
        x_high = torch.randint(0, 16, (K // 2, M), dtype=torch.uint8).T

    if layout[1] == "N":
        w_low = torch.randint(0, 16, (N, K // 2), dtype=torch.uint8, device="cuda")
        w_high = torch.randint(0, 16, (N, K // 2), dtype=torch.uint8, device="cuda")
    else:
        w_low = torch.randint(0, 16, (K // 2, N), dtype=torch.uint8, device="cuda").T
        w_high = torch.randint(0, 16, (K // 2, N), dtype=torch.uint8, device="cuda").T

    x = (
        x_high << 4 | x_low
    )  # Doing this computation on GPU tensors results in NaNs, so move it to GPU afterwards
    x = x.to(device="cuda")

    w = w_low | w_high << 4
    # Scale of 1.0 in e8m0, bias 127.
    M_pad = (M + 255) // 256 * 256
    x_scales = torch.randint(
        124, 128, (K // SCALE_GROUP_SIZE, M_pad), dtype=torch.uint8, device="cuda"
    )
    w_scales = torch.randint(
        124, 128, (K // SCALE_GROUP_SIZE, N), dtype=torch.uint8, device="cuda"
    )
    x_scales = x_scales.T
    w_scales = w_scales.T
    if shuffle_scales_fg:
        # Arch-independent aiter.ops.shuffle.shuffle_scale layout (shared with
        # the CK/asm GEMMs): 32-row stripes of 8 k-groups, returned flat as
        # (pad256(rows), pad8(K//32)). The kernels index one row per 32-row
        # stripe, so view it as (pad256(rows)//32, pad8(K//32)*32) -- taken off
        # the shuffled tensor's own shape, which is padded on both dims.
        # M < 32 stays un-shuffled (M, K//32) row-major.
        if M >= 32:
            xs = shuffle_scale(x_scales[:M])
            x_scales_shuffled = xs.view(-1, xs.shape[1] * 32)
        else:
            x_scales_shuffled = x_scales[:M].contiguous()
        ws = shuffle_scale(w_scales)
        w_scales_shuffled = ws.view(-1, ws.shape[1] * 32)
    else:
        x_scales_shuffled = x_scales[:M]
        w_scales_shuffled = w_scales

    if shuffle_weight_fg:
        # aiter.ops.shuffle.shuffle_weight (layout=(16, 16)) returns the (N, K)
        # shuffled weight, byte-identical on both arches; reshape to the
        # (N//16, K*16) layout the kernel consumes
        w_shuffed = shuffle_weight(w).reshape(w.shape[0] // 16, w.shape[1] * 16)
    else:
        w_shuffed = w

    y = None
    if output:
        y = torch.empty((M, N), dtype=dtype).cuda()
        out_dtype = (None,)
    else:
        out_dtype = dtype

    return (
        x,
        w,
        w_shuffed,
        x_scales[:M],
        w_scales,
        x_scales_shuffled,
        w_scales_shuffled,
        out_dtype,
        y,
    )


def get_x_vals():
    x_vals = [(1024 * v, 1024 * v, 1024 * v) for v in (1, 2, 4, 5, 8)]
    x_vals += [(v, 106496, 16384) for v in (150, 256, 4096, 8000)]  # LL3 405B FC1
    x_vals += [(v, 9216, 7168) for v in (128, 192, 4096, 8000)]
    x_vals += [(v, 7168, 4608) for v in (128, 192, 4096, 8000)]
    x_vals += [(v, 2112, 7168) for v in (128, 192, 4096, 8000)]
    x_vals += [(v, 8192, 512) for v in (128, 192, 4096, 8000)]
    x_vals += [(2048, 8192, 4096)]
    x_vals += [(1, 256, 512), (16, 256, 256), (31, 7168, 4608)]  # M < 32 case
    return x_vals


def mxfp4_to_f32(x):
    # 2 because we pack fp4 in uint8.
    x = x.repeat_interleave(2, dim=1)
    x[:, ::2] = x[:, ::2] & 0xF
    x[:, 1::2] = x[:, 1::2] >> 4
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
    w_f32 = mxfp4_to_f32(w)
    # Next convert the e8m0 scales to f32.
    x_scales = x_scales.repeat_interleave(SCALE_GROUP_SIZE, dim=1).to(torch.float32)
    x_scales_f32 = e8m0_to_f32(x_scales)
    x_f32 = x_f32 * x_scales_f32
    w_scales = w_scales.repeat_interleave(SCALE_GROUP_SIZE, dim=1).to(torch.float32)
    w_scales_f32 = e8m0_to_f32(w_scales)
    w_f32 = w_f32 * w_scales_f32
    return torch.mm(x_f32, w_f32.T).to(dtype)


@pytest.mark.parametrize("M, N, K", get_x_vals())
@pytest.mark.parametrize("output", [True, False])
@pytest.mark.parametrize("shuffle_weight_scales", [True, False])
@pytest.mark.parametrize("skip_reduce", [True, False])
@pytest.mark.parametrize("impl", ["triton", "gluon"])
def test_gemm_afp4_wfp4(
    M: int,
    N: int,
    K: int,
    output,
    shuffle_weight_scales,
    skip_reduce,
    impl,
):
    if impl == "gluon" and not arch_info.is_gluon_avail():
        pytest.skip("Gluon implementation is not supported on this GPU.")
    dtype = torch.bfloat16
    # TODO(brunomazzotti): Fix gluon instr shape then enable gluon tests conditionally on 950
    if impl == "gluon":
        pytest.skip("Gluon tests temporarily disabled.")

    if impl == "gluon" and shuffle_weight_scales:
        pytest.skip("Gluon kernel does not have a preshuffled implementation.")

    if shuffle_weight_scales:
        if N % 32 > 0:
            pytest.skip(
                f"N = {N} is not divisible by 32, skip this test for preshuffled weight/scales tests"
            )
        elif K % 256 > 0:
            pytest.skip(
                f"K = {K} is not divisible by 256, skip this test for preshuffled weight/scales tests"
            )

    (
        x,
        w,
        w_triton,
        x_scales,
        w_scales,
        x_scales_triton,
        w_scales_triton,
        _out_dtype,
        y,
    ) = generate_gemm_afp4wfp4_inputs(
        M,
        N,
        K,
        dtype,
        layout="TN",
        output=output,
        shuffle_scales_fg=shuffle_weight_scales,
        shuffle_weight_fg=shuffle_weight_scales,
    )

    torch_out = run_torch(x, w, x_scales, w_scales, dtype).to(dtype)

    if shuffle_weight_scales:
        triton_out = gemm_afp4wfp4_preshuffle(
            x,
            w_triton,
            x_scales_triton,
            w_scales_triton,
            dtype,
            y,
            skip_reduce=skip_reduce,
        )
    else:
        if impl == "triton":
            fn = triton_gemm_afp4wfp4
        elif impl == "gluon":
            fn = functools.partial(triton_gemm_afp4wfp4, backend="gluon")
        else:
            raise ValueError(f"Unknown implementation: {impl}")
        triton_out = fn(
            x,
            w_triton,
            x_scales_triton,
            w_scales_triton,
            dtype,
            y,
            skip_reduce=skip_reduce,
        )

    if triton_out.dim() == 3:
        triton_out = triton_out.sum(dim=0).to(dtype)

    triton.testing.assert_close(torch_out, triton_out)


@pytest.mark.parametrize("M, N, K", [(1, 10240, 8192), (64, 8192, 28672)])
def test_gemm_afp4_wfp4_preshuffle_splitk(M: int, N: int, K: int):
    dtype = torch.bfloat16
    (
        x,
        w,
        w_triton,
        x_scales,
        w_scales,
        x_scales_triton,
        w_scales_triton,
        _out_dtype,
        y,
    ) = generate_gemm_afp4wfp4_inputs(
        M,
        N,
        K,
        dtype,
        layout="TN",
        output=True,
        shuffle_scales_fg=True,
        shuffle_weight_fg=True,
    )

    # _get_config doubles K itself, so pass packed bytes as the wrapper does.
    config, _ = _get_afp4wfp4_config(M, N, K // 2, True, backend="triton")
    config = dict(config)
    config["NUM_KSPLIT"] = 4

    triton_out = gemm_afp4wfp4_preshuffle(
        x,
        w_triton,
        x_scales_triton,
        w_scales_triton,
        dtype,
        y,
        config=config,
    )

    torch_out = run_torch(x, w, x_scales, w_scales, dtype).to(dtype)

    triton.testing.assert_close(torch_out, triton_out)


@pytest.mark.parametrize("M, N, K", get_x_vals())
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("layout", ["TN"])  # "NN", "NT"
@pytest.mark.parametrize("output", [True, False])
def test_gemm_mxfp4_preshuffled_gfx1250(
    M: int,
    N: int,
    K: int,
    dtype,
    layout,
    output,
):
    if DEVICE_ARCH != "gfx1250":
        pytest.skip("Preshuffled gfx1250 kernel only supported on gfx1250")

    if N % 32 > 0:
        pytest.skip(
            f"N = {N} is not divisible by 32, skip this test for preshuffled weight/scales tests"
        )
    if K % 256 > 0:
        pytest.skip(
            f"K = {K} is not divisible by 256, skip this test for preshuffled weight/scales tests"
        )

    (
        x,
        w,
        w_preshuf,
        x_scales,
        w_scales,
        x_scales_shuffled,
        w_scales_shuffled,
        _out_dtype,
        y,
    ) = generate_gemm_afp4wfp4_inputs(
        M,
        N,
        K,
        dtype,
        layout=layout,
        output=output,
        shuffle_scales_fg=True,
        shuffle_weight_fg=True,
    )

    torch_out = run_torch(x, w, x_scales, w_scales, dtype).to(dtype)

    triton_out = gemm_afp4wfp4_preshuffle(
        x,
        w_preshuf,
        x_scales_shuffled,
        w_scales_shuffled,
        dtype,
        y if y is not None else torch.empty_like(torch_out),
    )

    triton.testing.assert_close(torch_out, triton_out)


def tile_scales_gfx1250(scales):
    """scale_layout="gfx1250_tile": the gfx1250 tile of shuffle_scale_gemm, rows padded to 32."""
    pad = (-scales.shape[0]) % 32
    if pad:
        scales = torch.cat([scales, scales.new_zeros(pad, scales.shape[1])])
    return shuffle_scale_gemm(
        scales.contiguous(), arch="gfx1250", preshuffle_factor=32, scale_kwidth=8
    )


@pytest.mark.parametrize(
    "M, N, K",
    [
        (1, 256, 512),
        (16, 256, 256),
        (31, 7168, 4608),
        (128, 9216, 7168),
        (192, 7168, 4608),
        (1024, 1024, 1024),
        (2048, 7168, 16384),
        (4096, 2112, 7168),
    ],
)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_gemm_mxfp4_preshuffled_gfx1250_tiled_scales(M: int, N: int, K: int, dtype):
    if DEVICE_ARCH != "gfx1250":
        pytest.skip("gfx1250_tile scales are only supported on gfx1250")

    (
        x,
        w,
        w_preshuf,
        x_scales,
        w_scales,
        x_scales_shuffled,
        w_scales_shuffled,
        _out_dtype,
        _y,
    ) = generate_gemm_afp4wfp4_inputs(
        M,
        N,
        K,
        dtype,
        output=False,
        shuffle_scales_fg=True,
        shuffle_weight_fg=True,
    )
    # A scales of M < 32 stay un-shuffled row-major in both layouts.
    x_scales_tiled = x_scales.contiguous() if M < 32 else tile_scales_gfx1250(x_scales)
    w_scales_tiled = tile_scales_gfx1250(w_scales)

    default_out = gemm_afp4wfp4_preshuffle(
        x, w_preshuf, x_scales_shuffled, w_scales_shuffled, dtype
    )
    tiled_out = gemm_afp4wfp4_preshuffle(
        x, w_preshuf, x_scales_tiled, w_scales_tiled, dtype, scale_layout="gfx1250_tile"
    )

    # Same tiles and per-tile k-order as the default path: identical bits.
    torch.testing.assert_close(tiled_out, default_out, rtol=0, atol=0)
    torch_out = run_torch(x, w, x_scales, w_scales, dtype).to(dtype)
    triton.testing.assert_close(torch_out, tiled_out)
