# SPDX-License-Identifier: MIT
"""BF16 GEMM with FP32 split-K accumulation and a FlyDSL BF16 conversion.

Factory owns one workspace; use a separate launcher for concurrent streams.
Both GEMM (including its zero initialization) and conversion must be timed.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch

from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.flydsl.gemm_kernels import flydsl_hgemm


def build_gemm_bf16_split_fp32(m, n, config=None, *, device=None):
    """Build ``launch(a[M,K], b[N,K], out[M,N])`` with reusable FP32 workspace.

    Inputs/output must be contiguous BF16, without output/input aliasing.
    Create one launcher per concurrent stream and retain it while graphs exist.
    The default tile targets small M; other shapes need explicit tuning.
    """
    if m <= 0 or n <= 0:
        raise ValueError("m and n must be positive")
    config = (
        dict(config)
        if config is not None
        else {
            "block_m": 16,
            "block_n": 32,
            "block_k": 128,
            "stages": 4,
            "split_k": 4,
            "m_waves": 1,
            "n_waves": 2,
            "k_waves": 1,
            "group_m": 0,
            "policy": "ft",
        }
    )
    if config.get("split_k", 1) <= 1:
        raise ValueError("this launcher requires split_k > 1")
    device = (
        torch.device("cuda", torch.cuda.current_device())
        if device is None
        else torch.device(device)
    )
    if device.type != "cuda":
        raise ValueError("split GEMM requires a CUDA/HIP device")
    with torch.cuda.device(device):
        if get_gfx() != "gfx950":
            raise ValueError("split GEMM currently supports gfx950 only")
    workspace = torch.empty((m, n), device=device, dtype=torch.float32)

    @flyc.kernel
    def convert(X: fx.Tensor, Y: fx.Tensor):
        i = fx.block_idx.x * 256 + fx.thread_idx.x
        x = fx.rocdl.make_buffer_tensor(X)
        y = fx.rocdl.make_buffer_tensor(Y)
        if i < m * n:
            row = i // n
            col = i % n
            y[row, col] = x[row, col].to(fx.BFloat16)

    @flyc.jit
    def cast(
        X: fx.Tensor,
        Y: fx.Tensor,
        stream: fx.Stream = fx.Stream(None),  # noqa: B008 - FlyDSL stream annotation
    ):
        convert(X, Y).launch(
            grid=((m * n + 255) // 256, 1, 1), block=(256, 1, 1), stream=stream
        )

    def call(a, b, out):
        if (
            a.ndim != 2
            or b.ndim != 2
            or a.shape[0] != m
            or b.shape[0] != n
            or a.shape[1] != b.shape[1]
            or out.shape != (m, n)
        ):
            raise ValueError("expected a[M,K], b[N,K], and out[M,N]")
        for tensor in (a, b, out):
            if (
                tensor.dtype != torch.bfloat16
                or tensor.device != workspace.device
                or not tensor.is_contiguous()
            ):
                raise ValueError(
                    "split GEMM requires contiguous BF16 tensors on the launcher's device"
                )
        if any(
            out.untyped_storage().data_ptr() == tensor.untyped_storage().data_ptr()
            for tensor in (a, b)
        ):
            raise ValueError("split GEMM does not support output/input aliasing")
        with torch.cuda.device(workspace.device):
            flydsl_hgemm(a, b, out=workspace, out_dtype=torch.float32, **config)
            cast(workspace, out, stream=torch.cuda.current_stream())
        return out

    return call
