# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Iris CCL all-reduce implementation (experimental).

Iris uses symmetric heap memory shared across all ranks for direct
GPU-to-GPU reads. This module manages the Iris lifecycle and provides
an all-reduce operation that can be used as a drop-in replacement for
torch.distributed.all_reduce.

Not compatible with CUDA graph capture. Requires --enforce-eager.
"""

from typing import Any, Optional, Tuple

import iris
import torch
from iris.ccl import Config

import logging

__all__ = ["fused_allreduce_add_rms_quant_iris"]

logger = logging.getLogger(__name__)


class IrisManager:
    """Singleton manager for Iris symmetric heap with buffer caching.

    Iris allocates symmetric heap memory that must be shared across all
    ranks. We initialize once and reuse to avoid OOM from repeated
    allocations.
    """

    _instance: Optional["IrisManager"] = None
    _initialized: bool = False

    def __new__(cls) -> "IrisManager":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        if IrisManager._initialized:
            return
        IrisManager._initialized = True

        self._shmem: Any = None
        self._heap_size: int = 2**33  # 8GB default
        self._config: Any = None

        # Buffer cache: (M, N, dtype) -> (iris_input, iris_output, workspace)
        self._buffer_cache: dict[
            tuple[int, int, torch.dtype], tuple[Any, Any, Any]
        ] = {}
        self._max_cached_shapes: int = 16

    def initialize(self, heap_size: Optional[int] = None) -> None:
        """Initialize Iris symmetric heap (call once at startup)."""
        if self._shmem is not None:
            logger.debug("Iris already initialized, skipping")
            return

        if heap_size is not None:
            self._heap_size = heap_size

        cur_rank = (
            torch.distributed.get_rank()
            if torch.distributed.is_initialized()
            else 0
        )
        logger.info(
            "Initializing Iris symmetric heap: "
            f"rank={cur_rank}, heap_size={self._heap_size / 2**30:.1f}GB"
        )

        self._shmem = iris.iris(self._heap_size)
        self._config = Config(all_reduce_variant="one_shot")

        logger.info(f"Iris initialized successfully on rank {cur_rank}")

    @property
    def shmem(self) -> Any:
        """Get the Iris symmetric memory instance (auto-initializes)."""
        if self._shmem is None:
            self.initialize()
        return self._shmem

    @property
    def config(self) -> Any:
        """Get the Iris CCL config."""
        if self._config is None:
            self.initialize()
        return self._config

    def _get_or_create_buffers(
        self,
        M: int,
        N: int,
        dtype: torch.dtype,
    ) -> tuple[Any, Any, Any]:
        """Get cached buffers or create new ones."""
        cache_key = (M, N, dtype)

        if cache_key in self._buffer_cache:
            return self._buffer_cache[cache_key]

        if len(self._buffer_cache) >= self._max_cached_shapes:
            logger.info(
                f"Iris buffer cache full ({len(self._buffer_cache)} shapes), "
                "clearing"
            )
            self._buffer_cache.clear()

        shmem = self.shmem
        config = self.config

        iris_input = shmem.zeros((M, N), dtype=dtype)
        iris_output = shmem.zeros((M, N), dtype=dtype)

        shmem.barrier()
        workspace = shmem.ccl.all_reduce_preamble(
            iris_output, iris_input, config=config
        )
        shmem.barrier()

        self._buffer_cache[cache_key] = (iris_input, iris_output, workspace)

        cur_rank = (
            torch.distributed.get_rank()
            if torch.distributed.is_initialized()
            else 0
        )
        logger.info(
            f"Iris: created buffers for shape ({M}, {N}), "
            f"dtype={dtype}, rank={cur_rank}"
        )

        return iris_input, iris_output, workspace

    def all_reduce(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Perform all-reduce on input tensor using Iris CCL.

        Args:
            input_tensor: Input tensor (M, N) on GPU

        Returns:
            All-reduced tensor (M, N)
        """
        if torch.cuda.is_current_stream_capturing():
            logger.warning(
                "Iris CCL uses host barriers (shmem.barrier) which are not "
                "capturable in CUDA graphs. Use --enforce-eager or switch to "
                "iris_opt which uses device_barrier."
            )

        shmem = self.shmem
        config = self.config

        M, N = input_tensor.shape

        iris_input, iris_output, workspace = self._get_or_create_buffers(
            M, N, input_tensor.dtype
        )

        iris_input.copy_(input_tensor)
        shmem.barrier()
        shmem.ccl.all_reduce(
            iris_output, iris_input, config=config, workspace=workspace
        )
        torch.cuda.synchronize()

        output = torch.empty_like(input_tensor)
        output.copy_(iris_output)

        return output


_iris_manager: Optional[IrisManager] = None


def get_iris_manager() -> IrisManager:
    """Get the global Iris manager instance."""
    global _iris_manager
    if _iris_manager is None:
        _iris_manager = IrisManager()
    return _iris_manager


def initialize_iris(heap_size: Optional[int] = None) -> None:
    """Initialize Iris for all-reduce operations.

    Call this once at model load time before any forward passes.

    Args:
        heap_size: Size of symmetric heap in bytes (default: 8GB)
    """
    get_iris_manager().initialize(heap_size)


def fused_allreduce_add_rms_quant_iris(
    input: torch.Tensor,
    rms_weight: torch.Tensor,
    rms_eps: float,
    quant_scale: torch.Tensor,
    quant_dtype: torch.dtype,
    group_name: str,
    residual: Optional[torch.Tensor] = None,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    Optional[torch.Tensor],
    torch.Tensor,
    torch.Tensor,
]:
    """Iris CCL implementation of fused AllReduce + Add + RMSNorm + FP8 Quant.

    Uses shared Iris symmetric memory for all-reduce, then applies
    RMSNorm and quantization.
    """
    iris_mgr = get_iris_manager()

    # Step 1: All-reduce using Iris
    allreduce_out = iris_mgr.all_reduce(input)

    # Step 2: RMSNorm (with or without residual add)
    if residual is not None:
        residual_out = allreduce_out + residual
        variance = (residual_out.float() ** 2).mean(dim=-1, keepdim=True)
        rrms = torch.rsqrt(variance + rms_eps)
        rms_out = (residual_out.float() * rrms * rms_weight.float()).to(
            input.dtype
        )
    else:
        residual_out = None
        variance = (allreduce_out.float() ** 2).mean(dim=-1, keepdim=True)
        rrms = torch.rsqrt(variance + rms_eps)
        rms_out = (allreduce_out.float() * rrms * rms_weight.float()).to(
            input.dtype
        )

    # Step 3: FP8 Quant - simple per-tensor quantization
    abs_max = rms_out.float().abs().max()
    if quant_dtype == torch.float8_e4m3fn:
        fp8_max = 448.0
    else:
        fp8_max = 57344.0
    quant_scale_out = (abs_max / fp8_max).to(torch.float32)
    quant_out = (rms_out.float() / quant_scale_out).to(quant_dtype)
    quant_scale_out = quant_scale_out.view(1)

    return allreduce_out, rms_out, residual_out, quant_out, quant_scale_out
