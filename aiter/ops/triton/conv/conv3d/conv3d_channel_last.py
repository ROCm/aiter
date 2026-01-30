from typing import Optional, Union
import triton
import torch

from aiter.ops.triton._triton_kernels.conv.conv3d.conv3d_channel_last import (
    _conv3d_channel_last_kernel,
)
from aiter.ops.triton.utils.conv_common import (
    conv3d_output_shape,
    padding_type,
)
from aiter.ops.triton.utils.logger import AiterTritonLogger

_LOGGER: AiterTritonLogger = AiterTritonLogger()


def conv3d_channel_last(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    stride: Optional[tuple[int, int, int]] = (1, 1, 1),
    padding: Optional[Union[tuple[int, int, int], str]] = (0, 0, 0),
    dilation: Optional[tuple[int, int, int]] = (1, 1, 1),
    groups: Optional[int] = 1,
    config: Optional[dict] = None,
):
    """
    3D convolution front-end with optional SAME/explicit padding handling.

    Args:
        x (torch.Tensor): Input tensor shaped (N, C, D, H, W).
        weight (torch.Tensor): Filter tensor shaped (C_out, C_in/groups, Kd, Kh, Kw).
        bias (Optional[torch.Tensor]): Bias vector shaped (C_out,).
        stride (tuple[int, int, int]): Stride for D, H, W.
        padding (tuple[int, int, int] | str): Explicit padding or "SAME".
        dilation (tuple[int, int, int]): Dilation for Kd, Kh, Kw.
        groups (int): Number of groups.

    Returns:
        torch.Tensor: Output tensor shaped (N, C_out, D_out, H_out, W_out).

    Raises:
        AssertionError: If tensor shapes or group constraints are invalid.
        ValueError: If padding mode is unsupported.
        NotImplementedError: Kernel path not implemented yet.
    """
    _LOGGER.info(
        f"CONV3D_CHANNEL_LAST called with x shape: {x.shape}, weight shape: {weight.shape}"
    )

    # Check constraints
    assert (
        x.ndim == 5 and weight.ndim == 5
    ), "Input tensor x must be 5-dimensional, x.shape: {x.shape}, weight.shape: {weight.shape}"
    assert (
        x.shape[1] == groups * weight.shape[1]
    ), "Input channels must be divisible by groups"
    assert bias is None or (
        bias.ndim == 1 and bias.shape[0] == weight.shape[0]
    ), "Bias must be 1-dimensional and match output channels"

    padding_val = (
        padding_type(x, weight, stride, dilation, padding)
        if isinstance(padding, str)
        else padding
    )
    out_shape = conv3d_output_shape(x, weight, stride, padding_val, dilation)

    # Allocate output tensor
    output = torch.empty(out_shape, dtype=x.dtype, device=x.device)

    if config is None:
        config = {
            "BLOCK_N": 256,
            "BLOCK_CI": 16,
            "BLOCK_CO": 64,
            "num_warps": 8,
            "num_stages": 2,
            "waves_per_eu": 1,
        }

    # BLOCK_NI for N x od x oh x ow,
    # BLOCK_CO for oc,
    # one group per cat
    N = x.shape[0]
    oc = weight.shape[0]
    od, oh, ow = out_shape[-3:]

    def grid_fn(META):
        return (
            triton.cdiv(N * od * oh * ow, META["BLOCK_N"]),
            triton.cdiv(oc // groups, META["BLOCK_CO"]),
            groups,
        )

    if bias is None:
        bias = torch.zeros(oc, dtype=x.dtype, device=x.device)

    _LOGGER.info(f"x.stride: {x.stride()}, weight.stride(): {weight.stride()}")
    _LOGGER.info(f"output.shape: {output.shape}, output.stride(): {output.stride()}")

    _conv3d_channel_last_kernel[grid_fn](
        x,
        weight,
        output,
        bias,
        N,
        x.shape[2],
        x.shape[3],
        x.shape[4],
        oc,
        od,
        oh,
        ow,
        *x.stride(),
        *weight.stride(),
        *output.stride(),
        weight.shape[1],
        weight.shape[2],
        weight.shape[3],
        weight.shape[4],
        stride[0],
        stride[1],
        stride[2],
        padding_val[0],
        padding_val[1],
        padding_val[2],
        dilation[0],
        dilation[1],
        dilation[2],
        GROUPS=groups,
        **config,
    )

    if isinstance(padding, str) and padding.upper() == "SAME":
        _LOGGER.info("Output generated with SAME padding.")
        output = output[
            :, :, (od - x.shape[2]) :, (oh - x.shape[3]) :, (ow - x.shape[4]) :
        ]

    return output
