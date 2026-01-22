import torch
import math
from typing import Optional, Union

from aiter.ops.triton.utils.logger import AiterTritonLogger

_LOGGER: AiterTritonLogger = AiterTritonLogger()


def padding_same(i, w, stride, dilation):
    return math.ceil((stride * (i - 1) + 1 + dilation * (w - 1) - i) / 2)


def out_padding(i, w, padding, stride, dilation):
    return int((i + 2 * padding - dilation * (w - 1) - 1) / stride + 1)


def padding_type(
    x: torch.Tensor,
    w: torch.Tensor,
    stride: torch.Tensor,
    dilation: torch.Tensor,
    padding: str,
) -> list[int, int, int]:
    """
    Determines padding-mode padding

    Args:
        x (torch.Tensor): Input tensor shaped (N, C, D, H, W).
        w (torch.Tensor): Weight tensor shaped (C_out, C_in/groups, Kd, Kh, Kw).
        stride (torch.Tensor | tuple[int, int, int]): Strides for D, H, W.
        dilation (torch.Tensor | tuple[int, int, int]): Dilations for Kd, Kh, Kw.

    Returns:
        tuple[list[int, int, int], list[int, int, int]]:
            - padding_inp: padding for (D, H, W)
            - padding_out: output sizes for (D, H, W)

    Notes:
        SAME mode only supports stride 1 for all dimensions.
    """
    padding = padding.upper()
    padding_val = [0, 0, 0]  # default valid padding

    if padding == "SAME":
        _LOGGER.info("Using SAME padding algorithm.")
        assert all(
            s == 1 for s in stride
        ), "SAME padding only supports stride of 1. recieved stride: {stride}"
        padding_val = [
            padding_same(i, w, s, d)
            for i, w, s, d in zip(x.shape[-3:], w.shape[-3:], stride, dilation)
        ]

    return padding_val


def conv3d_output_shape(
    x: torch.Tensor,
    w: torch.Tensor,
    stride: tuple[int, int, int],
    padding: tuple[int, int, int],
    dilation: tuple[int, int, int],
) -> tuple[int, int, int, int, int]:
    """
    Calculate the output shape of a 3D convolution.

    Args:
        x (torch.Tensor): Input tensor shaped (N, C_in, D_in, H_in, W_in).
        w (torch.Tensor): Weight tensor shaped (C_out, C_in/groups, Kd, Kh, Kw).
        stride (tuple[int, int, int]): Strides for D, H, W.
        padding (tuple[int, int, int]): Paddings for D, H, W.
        dilation (tuple[int, int, int]): Dilations for Kd, Kh, Kw.

    Returns:
        tuple[int, int, int, int, int]: Output tensor shape (N, C_out, D_out, H_out, W_out).
    """
    N = x.shape[0]
    C_out = w.shape[0]
    D_out, H_out, W_out = [
        out_padding(i, w, p, s, d)
        for i, w, p, s, d in zip(x.shape[-3:], w.shape[-3:], padding, stride, dilation)
    ]

    return (N, C_out, D_out, H_out, W_out)


def conv3d_total_flops(n, out_c, out_d, out_h, out_w, weight_c, kd, kh, kw, groups=1):
    # weight_c * kd * kh * kw  MACs,and per MAC 2 FLOPs
    in_c = weight_c * groups
    mac = n * out_c * out_d * out_h * out_w * in_c * kd * kh * kw
    return mac * 2  # FLOPs


def tflops(flops, elapsed):
    return flops / (elapsed / 1e3) / 1e12


def conv3d_bytes(
    n,
    out_c,
    out_d,
    out_h,
    out_w,
    weight_c,
    kd,
    kh,
    kw,
    groups,
    in_d,
    in_h,
    in_w,
    dtype_size,
    include_bias=True,
):
    in_c = weight_c * groups
    input_bytes = n * in_c * in_d * in_h * in_w * dtype_size
    weight_bytes = out_c * in_c * kd * kh * kw * dtype_size
    bias_bytes = out_c * dtype_size if include_bias else 0
    output_bytes = n * out_c * out_d * out_h * out_w * dtype_size
    return input_bytes + weight_bytes + bias_bytes + output_bytes


def bandwidth_gbps(total_bytes, elapsed):
    return total_bytes / (elapsed / 1e3) / 1e9


class GPUTimer:
    def __enter__(self):
        self.start = torch.cuda.Event(enable_timing=True)
        self.end = torch.cuda.Event(enable_timing=True)
        self.start.record()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end.record()
        self.end.synchronize()
        self.elapsed_ms = self.start.elapsed_time(self.end)

    @property
    def elapsed(self) -> float:
        # backward compatible alias
        return getattr(self, "elapsed_ms", 0.0)


class ConvGPUTimer(GPUTimer):
    def __init__(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        stride: tuple[int, int, int] = (1, 1, 1),
        padding: Union[tuple[int, int, int], str] = (0, 0, 0),
        dilation: tuple[int, int, int] = (1, 1, 1),
        groups: int = 1,
    ):
        super().__init__()

        # derive shapes
        padding_val = (
            padding_type(x, weight, stride, dilation, padding)
            if isinstance(padding, str)
            else padding
        )
        out_shape = conv3d_output_shape(x, weight, stride, padding_val, dilation)

        self.n = x.shape[0]
        self.out_c = weight.shape[0]
        self.out_d, self.out_h, self.out_w = out_shape[-3:]
        self.weight_c = weight.shape[1]
        self.kd, self.kh, self.kw = weight.shape[2], weight.shape[3], weight.shape[4]
        self.groups = groups
        self.in_d, self.in_h, self.in_w = x.shape[2], x.shape[3], x.shape[4]
        self.dtype_size = torch.finfo(x.dtype).bits // 8
        self.include_bias = bias is not None

    @property
    def computation(self) -> float:
        if not self.elapsed:
            _LOGGER.warning("Elapsed time is zero, cannot compute TFLOPS.")
            return 0.0
        flops = conv3d_total_flops(
            self.n,
            self.out_c,
            self.out_d,
            self.out_h,
            self.out_w,
            self.weight_c,
            self.kd,
            self.kh,
            self.kw,
            self.groups,
        )
        return tflops(flops, self.elapsed)

    @property
    def bandwidth(self) -> float:
        if not self.elapsed:
            return 0.0
        total_bytes = conv3d_bytes(
            self.n,
            self.out_c,
            self.out_d,
            self.out_h,
            self.out_w,
            self.weight_c,
            self.kd,
            self.kh,
            self.kw,
            self.groups,
            self.in_d,
            self.in_h,
            self.in_w,
            self.dtype_size,
            include_bias=self.include_bias,
        )
        return bandwidth_gbps(total_bytes, self.elapsed)
