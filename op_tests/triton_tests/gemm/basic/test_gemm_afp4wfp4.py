# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
import pytest
import torch
from aiter.ops.triton.gemm.basic.gemm_afp4wfp4 import (
    gemm_afp4wfp4 as triton_gemm_afp4wfp4,
    gemm_afp4wfp4_preshuffle,
)
from aiter.ops.triton.gluon.gemm_afp4wfp4 import gemm_afp4wfp4 as gluon_gemm_afp4wfp4
import aiter.ops.triton.utils._triton.arch_info as arch_info
from aiter.ops.triton.utils.types import str_to_torch_dtype
from aiter.ops.shuffle import shuffle_weight

DEVICE_ARCH = arch_info.get_arch()


def shuffle_scales(scales: torch.Tensor):
    scales_shuffled = scales.clone()
    sm, sn = scales_shuffled.shape
    scales_shuffled = scales_shuffled.view(sm // 32, 2, 16, sn // 8, 2, 4, 1)
    scales_shuffled = scales_shuffled.permute(0, 3, 5, 2, 4, 1, 6).contiguous()
    scales_shuffled = scales_shuffled.view(sm // 32, sn * 32)
    return scales_shuffled


def un_shuffle_scales(scales_shuffled: torch.Tensor):
    scales = scales_shuffled.clone()
    sm, sn = scales.shape
    scales = scales.view(sm * 32, sn // 32)
    sm, sn = scales.shape
    scales = scales.view(sm // 32, sn // 8, 4, 16, 2, 2, 1)
    scales = scales.permute(0, 5, 3, 1, 4, 2, 6).contiguous()
    scales = scales.view(sm, sn)
    return scales


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
        if M >= 32:
            x_scales_shuffled = shuffle_scales(x_scales)
        else:
            x_scales_shuffled = x_scales.contiguous()
        w_scales_shuffled = shuffle_scales(w_scales)
    else:
        x_scales_shuffled = x_scales
        w_scales_shuffled = w_scales

    if shuffle_weight_fg:
        use_int4 = False
        weight_shuffle_layout = (16, 16)
        w_shuffed = shuffle_weight(
            w, layout=weight_shuffle_layout, use_int4=use_int4
        ).reshape(
            w.shape[0] // weight_shuffle_layout[0],
            w.shape[1] * weight_shuffle_layout[0],
        )
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
        x_scales_shuffled[:M],
        w_scales_shuffled,
        out_dtype,
        y,
    )


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
    x_vals += [(256, 3584, 2112)]
    x_vals += [(7, 4608, 7168), (7, 7168, 2304)]
    x_vals += [(v, 106496, 16384) for v in [1, 8, 16, 31, 32, 64, 128, 256]]
    x_vals += [(v, 16384, 53248) for v in [1, 8, 16, 31, 32, 64, 128, 256]]
    x_vals += [(v, 18432, 16384) for v in [1, 8, 16, 31, 32, 64, 128, 256]]
    x_vals += [(v, 16384, 16384) for v in [1, 8, 16, 31, 32, 64, 128, 256]]
    x_vals += [(v, 10240, 8192) for v in [1, 2, 4, 8, 31, 16, 32, 64]]
    x_vals += [(v, 8192, 8192) for v in [1, 2, 4, 8, 31, 16, 32, 64]]
    x_vals += [(v, 57344, 8192) for v in [1, 2, 4, 8, 31, 16, 32, 64]]
    x_vals += [(v, 8192, 28672) for v in [1, 2, 4, 8, 31, 16, 32, 64]]
    x_vals += [(1, 1, 32)]  # minimal case
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


def run_triton(
    x, w, x_scales, w_scales, dtype=torch.bfloat16, y=None, skip_reduce=False, impl=None
):
    return impl(x, w, x_scales, w_scales, dtype, y, skip_reduce=skip_reduce)


@pytest.mark.parametrize("M, N, K", get_x_vals())
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("layout", ["TN", "TT", "NN", "NT"])
@pytest.mark.parametrize("output", [True, False])
@pytest.mark.parametrize(
    "shuffle_weight_scales",
    [True, False],
)
@pytest.mark.parametrize("skip_reduce", [True, False])
@pytest.mark.parametrize("impl", ["triton", "gluon"])
def test_gemm_afp4_wfp4(
    M: int,
    N: int,
    K: int,
    dtype,
    layout,
    output,
    shuffle_weight_scales,
    skip_reduce,
    impl,
):
    if impl == "gluon" and DEVICE_ARCH != "gfx950":
        pytest.skip(
            "Gluon implementation is not supported on this device (requires CDNA4)."
        )

    if impl == "gluon" and shuffle_weight_scales:
        pytest.skip("Gluon kernel does not have a preshuffled implementation.")

    if not (arch_info.is_fp4_avail()):
        pytest.skip("MXFP4 not supported on this architecture")

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
        out_dtype,
        y,
    ) = generate_gemm_afp4wfp4_inputs(
        M,
        N,
        K,
        dtype,
        layout=layout,
        output=output,
        shuffle_scales_fg=shuffle_weight_scales,
        shuffle_weight_fg=shuffle_weight_scales,
    )

    torch_out = run_torch(x, w, x_scales, w_scales, dtype).to(dtype)

    if shuffle_weight_scales:
        if output:
            triton_out = gemm_afp4wfp4_preshuffle(
                x,
                w_triton,
                x_scales_triton,
                w_scales_triton,
                dtype,
                y,
                use_aot=(dtype == torch.bfloat16 and layout == "TN"),
                skip_reduce=skip_reduce,
            )
        else:
            triton_out = gemm_afp4wfp4_preshuffle(
                x,
                w_triton,
                x_scales_triton,
                w_scales_triton,
                dtype,
                use_aot=(dtype == torch.bfloat16 and layout == "TN"),
                skip_reduce=skip_reduce,
            )
        # TODO: remove in the future
        # if output:
        #     triton_out = gemm_afp4wfp4_preshuffled_scales(
        #         x, w_triton, x_scales_triton, w_scales_triton, dtype, y
        #     )
        # else:
        #     triton_out = gemm_afp4wfp4_preshuffled_scales(
        #         x, w_triton, x_scales_triton, w_scales_triton, dtype
        #     )
    else:
        if impl == "triton":
            impl = triton_gemm_afp4wfp4
        elif impl == "gluon":
            impl = gluon_gemm_afp4wfp4
        else:
            raise ValueError(f"Unknown implementation: {impl}")

        if output:
            triton_out = run_triton(
                x,
                w_triton,
                x_scales_triton,
                w_scales_triton,
                dtype,
                y,
                skip_reduce=skip_reduce,
                impl=impl,
            )
        else:
            triton_out = run_triton(
                x,
                w_triton,
                x_scales_triton,
                w_scales_triton,
                dtype,
                skip_reduce=skip_reduce,
                impl=impl,
            )

    if triton_out.dim() == 3:
        triton_out = triton_out.sum(dim=0).to(dtype)

    torch.testing.assert_close(torch_out, triton_out)
