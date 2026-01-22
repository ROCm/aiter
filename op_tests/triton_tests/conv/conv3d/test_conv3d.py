import pytest
import torch

from aiter.ops.triton.conv.conv3d.conv3d_std import conv3d_std
from aiter.ops.triton.conv.conv3d.conv3d_channel_last import conv3d_channel_last

import logging

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(name)s: %(message)s",
    force=True,
)
logging.getLogger("AITER_TRITON").propagate = True


def gen_conv3d_input(
    in_shape,
    weight_shape,
    in_dtype,
    weight_dtype,
    bias,
    stride,
    padding,
    dilation,
    groups,
    device="cuda",
):
    x = torch.randn(*in_shape, dtype=in_dtype, device=device)
    weight = torch.randn(*weight_shape, dtype=weight_dtype, device=device)

    return {
        "args": (x, weight),
        "kwargs": {
            "bias": bias,
            "stride": stride,
            "padding": padding,
            "dilation": dilation,
            "groups": groups,
        },
    }


@pytest.mark.parametrize(
    "batch_size,in_channels,out_channels,depth,height,width,kernel_size,bias,stride,padding,dilation,groups",
    [
        (
            batch_size,
            in_channels,
            out_channels,
            depth,
            height,
            width,
            kernel_size,
            bias,
            stride,
            padding,
            dilation,
            groups,
        )
        for batch_size in (1,)
        for bias in (None,)
        for stride in ((1, 1, 1),)
        for padding in ((0, 0, 0),)
        for dilation in ((1, 1, 1),)
        for groups in (1,)
        for (in_channels, out_channels, depth, height, width, kernel_size) in [
            (384, 384, 3, 162, 92, (3, 3, 3)),
            (96, 96, 6, 1282, 722, (3, 3, 3)),
            (192, 192, 6, 642, 362, (3, 3, 3)),
            (3, 96, 6, 1282, 722, (3, 3, 3)),
            (192, 384, 1, 320, 180, (1, 1, 1)),
            (32, 32, 21, 160, 90, (1, 1, 1)),
            (192, 192, 5, 320, 180, (3, 1, 1)),
            (384, 384, 4, 322, 182, (3, 3, 3)),
            (192, 384, 4, 322, 182, (3, 3, 3)),
            (192, 384, 2, 320, 180, (1, 1, 1)),
        ]
    ],
)
@pytest.mark.parametrize(
    "impl",
    ["triton.conv3d.std", "triton.conv3d.channel.last"],
)
def test_conv3d(
    batch_size,
    in_channels,
    out_channels,
    depth,
    height,
    width,
    kernel_size,
    bias,
    stride,
    padding,
    dilation,
    groups,
    impl: str,
):
    torch.cuda.empty_cache()

    in_dtype = torch.bfloat16
    weight_dtype = torch.bfloat16

    in_shape = (batch_size, in_channels, depth, height, width)
    weight_shape = (out_channels, in_channels // groups, *kernel_size)

    conv3d_inputs = gen_conv3d_input(
        in_shape,
        weight_shape,
        in_dtype,
        weight_dtype,
        bias,
        stride,
        padding,
        dilation,
        groups,
    )

    args = conv3d_inputs["args"]
    kwargs = conv3d_inputs["kwargs"]

    if impl == "triton.conv3d.std":
        output = conv3d_std(*args, **kwargs)
    elif impl == "triton.conv3d.channel.last":
        # Convert weight tensor to channels last 3d format
        # NCDHW -> NDHWC
        args[1].data = args[1].data.to(memory_format=torch.channels_last_3d)
        if not args[1].is_contiguous(memory_format=torch.channels_last_3d):
            raise ValueError(
                "Weight tensor is not in channels last 3d format. received strides: ",
                args[1].data.stride(),
            )
        output = conv3d_channel_last(*args, **kwargs)
    else:
        raise ValueError(f"Unknown implementation: {impl}")

    ref = torch.nn.functional.conv3d(*args, **kwargs)

    torch.testing.assert_close(ref, output, rtol=0.2, atol=0.02)
