# SPDX-License-Identifier: MIT

"""Dense BF16 × packed MXFP4 linear on RDNA3.5."""

from dataclasses import dataclass
from functools import lru_cache

import flydsl.expr as fx
import torch

from aiter.jit.utils.chip_info import get_cu_num_device, get_gfx_runtime_for_device

from .gemm_mxfp4_w4a16_config import (
    SUPPORTED_ARCHS,
    MXFP4W4A16Config,
    select_mxfp4_w4a16_config,
    validate_config,
)
from .kernels.gemm.mxfp4_w4a16.rdna3.mxfp4_w4a16 import (
    MX_BLOCK_K,
    launch_mxfp4_w4a16,
)
from .kernels.tensor_shim import _run_compiled, current_stream_arg

__all__ = [
    "MXFP4W4A16Config",
    "PreparedMXFP4Weight",
    "flydsl_gemm_mxfp4_w4a16",
    "prepare_gemm_mxfp4_w4a16_weight",
]


@dataclass(frozen=True)
class PreparedMXFP4Weight:
    """Flat N-packed E2M1 codes and K-block-major E8M0 scales."""

    packed: torch.Tensor
    scales: torch.Tensor
    n: int
    k: int


def prepare_gemm_mxfp4_w4a16_weight(
    weight: torch.Tensor, scale: torch.Tensor
) -> PreparedMXFP4Weight:
    """Repack checkpoint E2M1 bytes [N, K/2] and E8M0 scales [N, ceil(K/32)]."""
    if weight.ndim != 2 or scale.ndim != 2:
        raise ValueError("weight and scale must be 2-D")
    if weight.device != scale.device:
        raise ValueError("weight and scale must be on the same device")
    if weight.dtype != torch.uint8 or scale.dtype != torch.uint8:
        raise TypeError("weight and scale must be uint8")

    n, packed_k = weight.shape
    k = 2 * packed_k
    if min(n, k) < 1 or scale.shape != (n, (k + MX_BLOCK_K - 1) // MX_BLOCK_K):
        raise ValueError(
            "weight and scale shapes must agree and N and K must be positive"
        )
    if n * k >= 1 << 31:
        raise ValueError("N * K must fit in a signed 32-bit element offset")

    words = weight.T.contiguous()
    # Unpack only integer nibbles. No dense BF16 intermediate is prepared.
    codes = torch.stack((words & 15, words >> 4), dim=1).reshape(-1)
    packed = codes[::2] | (codes[1::2] << 4)
    return PreparedMXFP4Weight(packed, scale.T.contiguous(), n, k)


def _allocate_workspace(device, shape, config):
    m, n, _ = shape
    block_m = config.group_tile_m * config.wave_groups_m
    block_n = config.group_tile_n * config.wave_groups_n
    tiles = (
        ((m + block_m - 1) // block_m)
        * ((n + block_n - 1) // block_n)
        * config.wave_groups
    )
    partial_count = tiles * config.split_k * config.group_tile_m * config.group_tile_n
    if partial_count >= 1 << 31:
        raise ValueError("Split-K workspace exceeds the 32-bit element offset limit")
    partials = torch.empty(partial_count, dtype=torch.float32, device=device)
    counters = torch.zeros(
        tiles * config.compute_threads, dtype=torch.int32, device=device
    )
    return partials, counters


@lru_cache(maxsize=32)
def _eager_workspace(device, stream, shape, config):
    with torch.cuda.stream(stream):
        return _allocate_workspace(device, shape, config)


def _check_output(out, a, prepared, shape):
    if out.dtype != torch.bfloat16 or out.shape != shape or out.device != a.device:
        raise ValueError(f"out must have shape {shape} and be BF16 on {a.device}")
    if not out.is_contiguous() or out.is_neg():
        raise ValueError("out must be a contiguous tensor without a negative view")
    if torch.is_grad_enabled() and out.requires_grad:
        raise RuntimeError("MXFP4 W4A16 does not support autograd")

    start = out.data_ptr()
    end = start + out.numel() * out.element_size()
    for tensor in (a, prepared.packed, prepared.scales):
        source = tensor.data_ptr()
        if start < source + tensor.numel() * tensor.element_size() and source < end:
            raise ValueError("out must not overlap an input tensor")


def flydsl_gemm_mxfp4_w4a16(
    a: torch.Tensor,
    prepared: PreparedMXFP4Weight,
    *,
    out: torch.Tensor | None = None,
    config: MXFP4W4A16Config | None = None,
    stream: torch.cuda.Stream | None = None,
) -> torch.Tensor:
    """Run dense BF16 × MXFP4 on gfx1151 with BF16 output and FP32 accumulation."""
    if not isinstance(prepared, PreparedMXFP4Weight):
        raise TypeError("prepared must be a PreparedMXFP4Weight")
    if torch.version.hip is None or not a.is_cuda or a.dtype != torch.bfloat16:
        raise TypeError("a must be a BF16 ROCm GPU tensor")
    if a.ndim != 2 or not a.is_contiguous() or a.is_neg():
        raise ValueError("a must be a contiguous row-major matrix")
    if torch.is_grad_enabled() and a.requires_grad:
        raise RuntimeError("MXFP4 W4A16 does not support autograd")

    m, k = a.shape
    n = prepared.n
    if k != prepared.k or min(n, k) < 1:
        raise ValueError(
            "K must match the prepared weight and N and K must be positive"
        )
    for tensor, shape in (
        (prepared.packed, ((n * k + 1) // 2,)),
        (prepared.scales, ((k + MX_BLOCK_K - 1) // MX_BLOCK_K, n)),
    ):
        if tensor.device != a.device or tensor.dtype != torch.uint8:
            raise ValueError(
                "Prepared weights and scales must be uint8 on the input device"
            )
        if tensor.shape != shape or not tensor.is_contiguous() or tensor.is_neg():
            raise ValueError(
                f"Prepared weight storage must be contiguous with shape {shape}"
            )
    if max(2 * m * k, 2 * m * n, n * k) >= 1 << 31:
        raise ValueError("GEMM buffer offsets must fit in signed 32 bits")
    if stream is not None and stream.device != a.device:
        raise ValueError("stream and inputs must be on the same device")

    arch = get_gfx_runtime_for_device(a.device)
    if arch not in SUPPORTED_ARCHS:
        raise RuntimeError(f"FlyDSL MXFP4 W4A16 requires gfx1151, got {arch}")

    shape = (m, n, k)
    supplied_out = out is not None
    if supplied_out:
        _check_output(out, a, prepared, (m, n))

    with torch.cuda.device(a.device), torch.cuda.stream(stream):
        if out is None:
            out = torch.empty((m, n), dtype=torch.bfloat16, device=a.device)
        if m == 0:
            return out

        if config is None:
            cu_count = get_cu_num_device(a.device.index)
            config = select_mxfp4_w4a16_config(arch, *shape, cu_count)
        elif isinstance(config, MXFP4W4A16Config):
            validate_config(config, shape)
        else:
            raise TypeError("config must be an MXFP4W4A16Config")

        partials, counters = fx.Empty(), fx.Empty()
        if config.split_k > 1:
            if torch.cuda.is_current_stream_capturing():
                # Graph-private scratch keeps replays independent of eager calls.
                partials, counters = _allocate_workspace(a.device, shape, config)
            else:
                current = torch.cuda.current_stream(a.device)
                partials, counters = _eager_workspace(a.device, current, shape, config)

        # The validated tensors have fixed dtype, rank and layout for this key.
        _run_compiled(
            launch_mxfp4_w4a16,
            a,
            prepared.packed,
            prepared.scales,
            out,
            partials,
            counters,
            *shape,
            config.group_tile_m,
            config.group_tile_n,
            config.stage_k,
            config.waves_per_group_m,
            config.waves_per_group_n,
            config.wave_groups_m,
            config.wave_groups_n,
            config.k_unroll,
            config.right_cache_modifier,
            config.left_cache_modifier,
            config.left_in_registers,
            config.right_in_registers,
            config.traversal_group_m,
            config.split_k,
            current_stream_arg(a.device),
            specialization_key=(a.device, shape, config),
        )
    if supplied_out:
        torch.autograd.graph.increment_version(out)
    return out
