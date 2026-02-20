# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Iris optimized all-reduce implementation.

Inlines the Iris one-shot all-reduce Triton kernel and all supporting
code (config, workspace, group info, chiplet transform) directly in
this file. No imports from iris.ccl. Only depends on the iris runtime
for symmetric heap allocation (iris.iris) and the iris Triton language
extensions (iris.load, iris.store).

One-shot: every CTA gathers all remote tiles via iris.load, reduces
locally, and writes the result with a single tl.store. No broadcast
phase. RMSNorm and FP8 quant are separate torch ops after the kernel.

See iris_opt_allreduce.py for the fully-fused single-kernel version
that fuses all-reduce + RMSNorm + quant into one kernel launch.

Uses shmem.device_barrier() (device-side atomics) for CUDA graph compatibility.
"""

from dataclasses import dataclass
from typing import Any, Optional, Tuple

import iris
import torch
import triton
import triton.language as tl

import logging

__all__ = ["fused_allreduce_add_rms_quant_iris_inline"]

logger = logging.getLogger(__name__)


# ============================================================================
# Inlined from iris.ccl.utils
# ============================================================================


@triton.jit()
def chiplet_transform_chunked(
        pid,
        num_workgroups: tl.constexpr,
        num_xcds: tl.constexpr,
        chunk_size: tl.constexpr,
    ):
        """Redistribute workgroups across XCDs in chunks."""
        if pid > (num_workgroups // (num_xcds * chunk_size)) * (
            num_xcds * chunk_size
        ):
            return pid

        local_pid = pid // num_xcds
        chunk_idx = local_pid // chunk_size
        pos_in_chunk = local_pid % chunk_size

        xcd = pid % num_xcds
        new_pid = (
            chunk_idx * num_xcds * chunk_size + xcd * chunk_size + pos_in_chunk
        )
        return new_pid


def extract_group_info(
    shmem: Any,
) -> Tuple[int, int, int, int, int]:
    """Extract rank/group info from iris shmem context.

    Returns: (rank_in_group, rank_global, world_size, rank_start, rank_stride)
    """
    rank_global = shmem.get_rank()
    rank_in_group = rank_global
    world_size = shmem.get_num_ranks()
    rank_start = 0
    rank_stride = 1
    return rank_in_group, rank_global, world_size, rank_start, rank_stride


# ============================================================================
# Inlined from iris.ccl.config (one-shot relevant fields only)
# ============================================================================


@dataclass
class OneShotConfig:
    """Config for the one-shot all-reduce kernel."""

    block_size_m: int = 32
    block_size_n: int = 64
    swizzle_size: int = 4
    comm_sms: int = 64
    num_xcds: Optional[int] = None
    chunk_size: Optional[int] = None

    def __post_init__(self) -> None:
        if self.num_xcds is None:
            self.num_xcds = iris.hip.get_num_xcc()
        if self.chunk_size is None:
            self.chunk_size = self.swizzle_size * self.swizzle_size
            self.chunk_size = min(
                self.chunk_size, self.comm_sms // self.num_xcds
            )


# ============================================================================
# Inlined from iris.ccl.all_reduce (one-shot kernel)
# ============================================================================


@triton.jit
def persistent_all_reduce_one_shot(
    input_ptr,
    output_ptr,
    M,
    N,
    stride_in_m,
    stride_in_n,
    stride_out_m,
    stride_out_n,
    heap_bases: tl.tensor,
    group_rank: tl.constexpr,
    iris_rank: tl.constexpr,
    world_size: tl.constexpr,
    rank_start: tl.constexpr,
    rank_stride: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    COMM_SMS: tl.constexpr,
    NUM_XCDS: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
):
    """One-shot all-reduce: every CTA gathers all tiles from all ranks.

    Each CTA reads its assigned tiles from every rank via iris.load,
    accumulates locally, and writes the reduced result once via tl.store.
    No broadcast phase — all ranks do all tiles (duplicated work, but
    no remote stores needed).
    """
    pid = tl.program_id(0)

    if NUM_XCDS != 1:
        pid = chiplet_transform_chunked(pid, COMM_SMS, NUM_XCDS, CHUNK_SIZE)

    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    total_tiles = num_pid_m * num_pid_n

    acc_dtype = (
        tl.float32
        if output_ptr.type.element_ty != tl.int8
        else tl.int32
    )

    for tile_id in range(pid, total_tiles, COMM_SMS):
        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        group_id = tile_id // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + (
            (tile_id % num_pid_in_group) % group_size_m
        )
        pid_n = (tile_id % num_pid_in_group) // group_size_m

        tl.assume(pid_m >= 0)
        tl.assume(pid_n >= 0)

        rm_base = pid_m * BLOCK_SIZE_M
        rn_base = pid_n * BLOCK_SIZE_N
        rm = rm_base + tl.arange(0, BLOCK_SIZE_M)
        rn = rn_base + tl.arange(0, BLOCK_SIZE_N)
        rm = tl.max_contiguous(
            tl.multiple_of(rm, BLOCK_SIZE_M), BLOCK_SIZE_M
        )
        rn = tl.max_contiguous(
            tl.multiple_of(rn, BLOCK_SIZE_N), BLOCK_SIZE_N
        )
        mask = (rm[:, None] < M) & (rn[None, :] < N)

        input_offset = (
            rm[:, None] * stride_in_m + rn[None, :] * stride_in_n
        )
        output_offset = (
            rm[:, None] * stride_out_m + rn[None, :] * stride_out_n
        )

        acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=acc_dtype)

        for i in range(world_size):
            remote_rank = rank_start + i * rank_stride
            partial = iris.load(
                input_ptr + input_offset,
                iris_rank,
                remote_rank,
                heap_bases,
                mask=mask,
            )
            acc += partial.to(acc_dtype)

        tl.store(
            output_ptr + output_offset,
            acc.to(output_ptr.type.element_ty),
            mask=mask,
        )


# ============================================================================
# Manager and public API
# ============================================================================


class IrisOptManager:
    """Singleton manager for Iris with inlined one-shot all-reduce.

    Calls the one-shot Triton kernel directly instead of going through
    shmem.ccl.all_reduce(). This gives us control over the kernel for
    future fusion.
    """

    _instance: Optional["IrisOptManager"] = None
    _initialized: bool = False

    def __new__(cls) -> "IrisOptManager":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        if IrisOptManager._initialized:
            return
        IrisOptManager._initialized = True

        self._shmem: Any = None
        self._heap_size: int = 2**33  # 8GB default
        self._config: Optional[OneShotConfig] = None

        # Buffer cache: (M, N, dtype) -> (iris_input, iris_output)
        self._buffer_cache: dict[
            tuple[int, int, torch.dtype], tuple[Any, Any]
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
            "Initializing Iris (opt) symmetric heap: "
            f"rank={cur_rank}, heap_size={self._heap_size / 2**30:.1f}GB"
        )

        self._shmem = iris.iris(self._heap_size)
        self._config = OneShotConfig()

        logger.info(f"Iris (opt) initialized successfully on rank {cur_rank}")

    @property
    def shmem(self) -> Any:
        """Get the Iris symmetric memory instance (auto-initializes)."""
        if self._shmem is None:
            self.initialize()
        return self._shmem

    @property
    def config(self) -> OneShotConfig:
        """Get the one-shot config."""
        if self._config is None:
            self.initialize()
        assert self._config is not None
        return self._config

    def _get_or_create_buffers(
        self,
        M: int,
        N: int,
        dtype: torch.dtype,
    ) -> tuple[Any, Any]:
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

        iris_input = shmem.zeros((M, N), dtype=dtype)
        iris_output = shmem.zeros((M, N), dtype=dtype)

        shmem.device_barrier()

        self._buffer_cache[cache_key] = (iris_input, iris_output)

        cur_rank = (
            torch.distributed.get_rank()
            if torch.distributed.is_initialized()
            else 0
        )
        logger.info(
            f"Iris (opt): created buffers for shape ({M}, {N}), "
            f"dtype={dtype}, rank={cur_rank}"
        )

        return iris_input, iris_output

    def all_reduce(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Perform all-reduce using the inlined one-shot kernel.

        Args:
            input_tensor: Input tensor (M, N) on GPU

        Returns:
            All-reduced tensor (M, N)
        """
        shmem = self.shmem
        config = self.config

        M, N = input_tensor.shape

        iris_input, iris_output = self._get_or_create_buffers(
            M, N, input_tensor.dtype
        )

        # Copy input to symmetric heap
        iris_input.copy_(input_tensor)
        shmem.device_barrier()

        # Extract group info
        rank_in_group, rank_global, world_size, rank_start, rank_stride = (
            extract_group_info(shmem)
        )
        heap_bases = shmem.get_heap_bases()

        stride_in_m, stride_in_n = (
            iris_input.stride(0),
            iris_input.stride(1),
        )
        stride_out_m, stride_out_n = (
            iris_output.stride(0),
            iris_output.stride(1),
        )

        # Launch the one-shot kernel directly
        persistent_all_reduce_one_shot[(config.comm_sms,)](
            iris_input,
            iris_output,
            M,
            N,
            stride_in_m,
            stride_in_n,
            stride_out_m,
            stride_out_n,
            heap_bases,
            rank_in_group,
            rank_global,
            world_size,
            rank_start,
            rank_stride,
            config.block_size_m,
            config.block_size_n,
            config.swizzle_size,
            config.comm_sms,
            config.num_xcds,
            config.chunk_size,
            num_warps=8,
            num_stages=1,
            waves_per_eu=1,
        )

        shmem.device_barrier()

        # Copy result back from symmetric heap
        output = torch.empty_like(input_tensor)
        output.copy_(iris_output)

        return output


_iris_inline_manager: Optional[IrisOptManager] = None


def get_iris_inline_manager() -> IrisOptManager:
    """Get the global Iris opt manager instance."""
    global _iris_inline_manager
    if _iris_inline_manager is None:
        _iris_inline_manager = IrisOptManager()
    return _iris_inline_manager


def initialize_iris_inline(heap_size: Optional[int] = None) -> None:
    """Initialize Iris for optimized all-reduce operations.

    Call this once at model load time before any forward passes.

    Args:
        heap_size: Size of symmetric heap in bytes (default: 8GB)
    """
    get_iris_inline_manager().initialize(heap_size)


def fused_allreduce_add_rms_quant_iris_inline(
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
    """Iris optimized AllReduce + Add + RMSNorm + FP8 Quant.

    Uses the inlined one-shot all-reduce kernel, with RMSNorm and
    per-tensor FP8 quant as separate torch ops after the kernel.

    See iris_opt_allreduce.py for the fully-fused single-kernel version.
    """
    iris_mgr = get_iris_inline_manager()

    # Step 1: All-reduce using inlined one-shot kernel
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

    # Step 3: FP8 Quant - per-tensor quantization
    abs_max = rms_out.float().abs().max()
    if quant_dtype == torch.float8_e4m3fn:
        fp8_max = 448.0
    else:
        fp8_max = 57344.0
    quant_scale_out = (abs_max / fp8_max).to(torch.float32)
    quant_out = (rms_out.float() / quant_scale_out).to(quant_dtype)
    quant_scale_out = quant_scale_out.view(1)

    return allreduce_out, rms_out, residual_out, quant_out, quant_scale_out
