# SPDX-License-Identifier: MIT

"""Public BF16 decode GEMM API backed only by unified Wave/BlockMFMA."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import torch

import flydsl.expr as fx
from aiter.jit.utils.chip_info import get_cu_num, get_gfx_runtime

from .gemm_decode_block_mfma import compile_gemm_decode_block_mfma_bf16
from .gemm_decode_common import validate_gemm_decode_tensors
from .gemm_decode_config import (
    ActivationSource,
    BlockMfmaDecodeConfig,
    ContractionMode,
    DecodeArchTraits,
    DecodeConfig,
    DecodePolicy,
    OutputRounding,
    ReductionMode,
    WaveDecodeConfig,
    conservative_wave_config,
    gemm_decode_kernel_name,
    get_decode_arch_traits,
    iter_gemm_decode_configs,
    parse_gemm_decode_kernel_name,
)
from .gemm_decode_wave import compile_gemm_decode_wave_bf16

__all__ = [
    "ActivationSource",
    "BlockMfmaDecodeConfig",
    "ContractionMode",
    "DecodeArchTraits",
    "DecodeConfig",
    "DecodePolicy",
    "OutputRounding",
    "ReductionMode",
    "WaveDecodeConfig",
    "compile_gemm_decode_bf16",
    "gemm_decode_bf16",
    "gemm_decode_bf16_configured",
    "gemm_decode_kernel_name",
    "get_decode_arch_traits",
    "get_gemm_decode_bf16",
    "iter_gemm_decode_configs",
    "launch_gemm_decode_kernel_name",
    "parse_gemm_decode_kernel_name",
]


def compile_gemm_decode_bf16(
    m: int,
    n: int,
    k: int,
    config: DecodeConfig,
    *,
    arch: str,
    num_cus: int | None = None,
):
    """Compile one exact unified ``(arch, M, N, K, config)`` identity."""
    config.validate(m=m, n=n, k=k, arch=arch)
    if isinstance(config, WaveDecodeConfig):
        return compile_gemm_decode_wave_bf16(m, n, k, config, arch)
    if isinstance(config, BlockMfmaDecodeConfig):
        return compile_gemm_decode_block_mfma_bf16(
            m,
            n,
            k,
            config,
            arch,
            num_cus=num_cus,
        )
    raise TypeError(f"unsupported decode config type: {type(config).__name__}")


def _validate_bias(
    bias: torch.Tensor | None,
    *,
    output: torch.Tensor,
    n: int,
) -> None:
    if bias is None:
        return
    if not isinstance(bias, torch.Tensor):
        raise TypeError("bias must be a torch.Tensor")
    if bias.shape != (n,):
        raise ValueError(f"bias must have shape {(n,)}, got {tuple(bias.shape)}")
    if bias.dtype != torch.bfloat16:
        raise ValueError("bias must have dtype torch.bfloat16")
    if bias.device != output.device:
        raise ValueError(f"bias must be on {output.device}, got {bias.device}")
    if not bias.is_contiguous():
        raise ValueError("bias must be contiguous")


@dataclass(frozen=True)
class _ExecutionStream:
    flydsl: fx.Stream
    torch: torch.cuda.Stream | None


def _resolve_execution_stream(
    stream,
    *,
    device: torch.device,
    bias_requested: bool,
) -> _ExecutionStream:
    """Resolve one queue for both the kernel and optional PyTorch bias op."""
    value = stream.value if isinstance(stream, fx.Stream) else stream
    if value is None:
        torch_stream = torch.cuda.current_stream(device=device)
        return _ExecutionStream(fx.Stream(torch_stream), torch_stream)

    if isinstance(value, torch.cuda.Stream):
        if value.device != device:
            raise ValueError(
                f"stream must be on {device}, got {value.device}"
            )
        return _ExecutionStream(fx.Stream(value), value)

    if not isinstance(value, int) and hasattr(value, "cuda_stream"):
        stream_device = getattr(value, "device", device)
        if torch.device(stream_device) != device:
            raise ValueError(
                f"stream must be on {device}, got {stream_device}"
            )
        value = int(value.cuda_stream)

    if isinstance(value, int):
        if value < 0:
            raise ValueError("raw stream pointer must be non-negative")
        if value == 0:
            if bias_requested:
                raise ValueError(
                    "bias cannot use raw stream 0 because PyTorch cannot "
                    "safely wrap the HIP default-stream sentinel"
                )
            return _ExecutionStream(fx.Stream(0), None)
        try:
            torch_stream = torch.cuda.ExternalStream(value, device=device)
        except Exception as error:
            raise ValueError(
                f"invalid raw stream pointer for {device}: {value}"
            ) from error
        return _ExecutionStream(fx.Stream(torch_stream), torch_stream)

    raise TypeError(
        "stream must be None, a torch.cuda.Stream, a non-negative raw "
        "pointer, or fx.Stream wrapping one of those representations"
    )


def _apply_bias(
    output: torch.Tensor,
    bias: torch.Tensor | None,
    stream: _ExecutionStream,
) -> None:
    if bias is None:
        return
    assert stream.torch is not None
    with torch.cuda.stream(stream.torch):
        output.add_(bias)


def gemm_decode_bf16_configured(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    M: int,
    N: int,
    K: int,
    config: DecodeConfig,
    stream: fx.Stream = fx.Stream(None),
    *,
    arch: str | None = None,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Launch one explicit unified Wave/BlockMFMA configuration."""
    runtime_arch = get_gfx_runtime()
    if arch is not None and arch != runtime_arch:
        raise ValueError(
            f"explicit decode arch {arch} does not match runtime architecture "
            f"{runtime_arch}; use compile_gemm_decode_bf16 for compile-only/AOT"
        )
    validate_gemm_decode_tensors(A, B, C, M, N, K, arch=runtime_arch)
    _validate_bias(bias, output=C, n=N)
    execution_stream = _resolve_execution_stream(
        stream,
        device=A.device,
        bias_requested=bias is not None,
    )
    launcher = compile_gemm_decode_bf16(
        M,
        N,
        K,
        config,
        arch=runtime_arch,
        num_cus=get_cu_num(),
    )
    launcher(A, B, C, stream=execution_stream.flydsl)
    _apply_bias(C, bias, execution_stream)
    return C


def gemm_decode_bf16(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    M: int,
    N: int,
    K: int,
    stream: fx.Stream = fx.Stream(None),
    *,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Launch the deterministic legal unified default for exact M in [1, 5].

    Tuned callers should launch their exact stable kernel name. This generic
    API intentionally uses one legality-first Wave configuration rather than a
    second shape heuristic. M > 5 is outside the decode family contract.
    """
    arch = get_gfx_runtime()
    config = conservative_wave_config(M, N, K, arch)
    return gemm_decode_bf16_configured(
        A,
        B,
        C,
        M,
        N,
        K,
        config,
        stream,
        arch=arch,
        bias=bias,
    )


@lru_cache(maxsize=None)
def get_gemm_decode_bf16(config: DecodeConfig | None = None):
    """Return a stable public launcher for a unified config or default path."""

    def launch(A, B, C, M, N, K, stream=fx.Stream(None), *, bias=None):
        if config is None:
            return gemm_decode_bf16(
                A,
                B,
                C,
                M,
                N,
                K,
                stream,
                bias=bias,
            )
        return gemm_decode_bf16_configured(
            A,
            B,
            C,
            M,
            N,
            K,
            config,
            stream,
            bias=bias,
        )

    return launch


def launch_gemm_decode_kernel_name(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    kernel_name: str,
    stream: fx.Stream = fx.Stream(None),
    *,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Parse and launch one stable unified tuned-kernel identity."""
    arch, m, n, k, config = parse_gemm_decode_kernel_name(kernel_name)
    runtime_arch = get_gfx_runtime()
    if runtime_arch != arch:
        raise ValueError(
            f"decode kernel {kernel_name!r} targets {arch}, "
            f"but the runtime device is {runtime_arch}"
        )
    return gemm_decode_bf16_configured(
        A,
        B,
        C,
        m,
        n,
        k,
        config,
        stream,
        arch=arch,
        bias=bias,
    )
