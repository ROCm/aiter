# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors
# ruff: noqa: B008, SIM102

"""Device-driven gfx950 grouped TN contractions for SonicMoE backward.

The backward sorter lays every non-empty expert out as one or more contiguous
64-row blocks.  This kernel consumes that layout directly and computes

``output[expert, M, N] = lhs[expert, rows, M].T @ rhs[expert, rows, N]``.

A small device builder emits one ``(expert, first_sorted_row)`` descriptor per
active expert.  Its public queue ABI is ``[count, expert0, row0, ...]`` so dW2
and dW1 can consume the same queue without rebuilding it.  Decode can bypass
the builder and consume the sorter's one-block-per-expert metadata directly.
The GEMM grid is persistent and walks output tiles from either schedule, so no
output tile performs a lower-bound search and no expert metadata is copied to
the host.  Inputs and output are BF16; each tile accumulates all of an expert's
route rows in FP32 before one BF16 materialization.
"""

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import T
from flydsl.expr.utils.arith import _to_raw as _raw

from aiter.ops.flydsl.kernels import buffer_ops
from aiter.ops.flydsl.kernels.tensor_shim import _run_compiled

from .gemm_utils import (
    GFX950_DMA_BYTES,
    GFX950_WAVE_SIZE,
    __barrier,
    async_load_to_lds,
    buffer_load_lds_inline,
    get_wave_lds_offset,
    make_transposed_lds_layout,
    make_wave_lds_ptr,
    transposed_contiguous_idx,
)
from .memory import atomic_add

_BLOCK_THREADS = 256
_SORTED_BLOCK_M = 64
_NUM_CU = 256
_MAX_RESIDENT_THREADS_PER_CU = 1024
_LDS_BYTES_PER_CU = 163840
_MAX_SIGNED_I32 = (1 << 31) - 1
_MAX_BUFFER_BYTES = (1 << 32) - 1
_TOKEN_MASK = 0x00FFFFFF
_ZERO_VECTOR_ELEMENTS = GFX950_DMA_BYTES // 2
_ZERO_BLOCK_THREADS = 1024
_ZERO_MAX_BLOCKS_PER_EXPERT = _NUM_CU


def _global_bf16_ptr(address):
    pointer_type = fx.PointerType.get(
        fx.BFloat16.ir_type,
        address_space=fx.AddressSpace.Global,
        alignment=GFX950_DMA_BYTES,
    )
    return fx.inttoptr(pointer_type, fx.Int64(address))


def _global_f32_ptr(address):
    pointer_type = fx.PointerType.get(
        fx.Float32.ir_type,
        address_space=fx.AddressSpace.Global,
        alignment=GFX950_DMA_BYTES,
    )
    return fx.inttoptr(pointer_type, fx.Int64(address))


def hot_split_descriptor_capacity(
    routes: int,
    num_experts: int,
    split_rows: int,
    min_hot_rows: int,
) -> int:
    """Return a host bound for hot-expert split-K descriptors.

    Only experts with at least ``min_hot_rows`` enter the split queue.  The
    device builder emits ``ceil(frequency / split_rows)`` descriptors for each
    such expert, so the number of hot experts itself is bounded by
    ``routes // min_hot_rows``.
    """

    values = (
        ("routes", routes),
        ("num_experts", num_experts),
        ("split_rows", split_rows),
        ("min_hot_rows", min_hot_rows),
    )
    for name, value in values:
        if not isinstance(value, int) or isinstance(value, bool):
            raise TypeError(f"{name} must be an int, got {type(value).__name__}")
    if routes < 0 or num_experts <= 0 or split_rows <= 0 or min_hot_rows <= 0:
        raise ValueError(
            "routes must be non-negative and split queue dimensions positive"
        )
    if min_hot_rows < split_rows:
        raise ValueError("min_hot_rows must be at least split_rows")
    if routes == 0:
        return 0
    max_hot_experts = min(num_experts, routes // min_hot_rows)
    if max_hot_experts == 0:
        return 0
    # For A non-empty hot experts, sum ceil(c_e / S) is bounded by
    # floor((R + A*(S-1))/S).
    return (routes + max_hot_experts * (split_rows - 1)) // split_rows


def active_expert_descriptor_capacity(routes: int, num_experts: int) -> int:
    """Return a tight host-known upper bound on active expert descriptors."""

    for name, value in (("routes", routes), ("num_experts", num_experts)):
        if not isinstance(value, int) or isinstance(value, bool):
            raise TypeError(f"{name} must be an int, got {type(value).__name__}")
    if routes < 0:
        raise ValueError(f"routes must be non-negative, got {routes}")
    if num_experts <= 0:
        raise ValueError(f"num_experts must be positive, got {num_experts}")
    if routes > _MAX_SIGNED_I32:
        raise ValueError(f"routes exceeds signed int32 metadata capacity: {routes}")
    return min(routes, num_experts)


def active_expert_queue_elements(routes: int, num_experts: int) -> int:
    """Return the int32 storage size for ``[count, (expert, row) * capacity]``."""

    return 1 + 2 * active_expert_descriptor_capacity(routes, num_experts)


@functools.lru_cache(maxsize=128)
def compile_active_expert_queue(
    num_experts: int,
    device_index: int,
):
    """Compile the standalone producer for the shared active-expert queue.

    Sorter metadata length is a runtime extent and must not specialize this
    compiler cache.  The fixed lower-bound trip count covers every legal
    signed-i32 padded-row extent; predicates stop the search once it converges.
    """

    del device_index
    if num_experts <= 0:
        raise ValueError("num_experts must be positive")
    max_metadata_blocks = _MAX_SIGNED_I32 // _SORTED_BLOCK_M
    lower_bound_steps = max_metadata_blocks.bit_length()

    @flyc.kernel(known_block_size=[_BLOCK_THREADS, 1, 1])
    def clear_queue_count(queue_storage: fx.Tensor):
        if gpu.thread_idx.x == fx.Int32(0):
            storage_rsrc = buffer_ops.create_buffer_resource(
                queue_storage, max_size=True
            )
            buffer_ops.buffer_store(fx.Int32(0), storage_rsrc, fx.Int32(0))

    @flyc.kernel(known_block_size=[_BLOCK_THREADS, 1, 1])
    def build_queue(
        expert_frequency: fx.Tensor,
        sorted_expert_ids: fx.Tensor,
        num_valid_ids: fx.Tensor,
        queue_storage: fx.Tensor,
        i32_queue_capacity: fx.Int32,
    ):
        expert = gpu.block_idx.x * fx.Int32(_BLOCK_THREADS) + gpu.thread_idx.x
        if expert < fx.Int32(num_experts):
            frequency_rsrc = buffer_ops.create_buffer_resource(
                expert_frequency, max_size=True
            )
            frequency = fx.Int32(
                buffer_ops.buffer_load(frequency_rsrc, expert, vec_width=1, dtype=T.i32)
            )
            if frequency > fx.Int32(0):
                valid_rsrc = buffer_ops.create_buffer_resource(
                    num_valid_ids, max_size=True
                )
                padded_rows = fx.Int32(
                    buffer_ops.buffer_load(
                        valid_rsrc, fx.Int32(0), vec_width=1, dtype=T.i32
                    )
                )
                metadata_blocks = padded_rows // fx.Int32(_SORTED_BLOCK_M)
                expert_ids_rsrc = buffer_ops.create_buffer_resource(
                    sorted_expert_ids, max_size=True
                )

                lo = fx.Int32(0)
                hi = metadata_blocks
                for _ in range_constexpr(lower_bound_steps):
                    searching = lo < hi
                    mid = (lo + hi) // fx.Int32(2)
                    safe_mid = searching.select(mid, fx.Int32(0))
                    mid_expert = fx.Int32(
                        buffer_ops.buffer_load(
                            expert_ids_rsrc, safe_mid, vec_width=1, dtype=T.i32
                        )
                    )
                    move_right = searching & (mid_expert < expert)
                    lo = move_right.select(mid + fx.Int32(1), lo)
                    move_left = searching & (mid_expert >= expert)
                    hi = move_left.select(mid, hi)

                in_metadata = lo < metadata_blocks
                safe_lo = in_metadata.select(lo, fx.Int32(0))
                found_expert = fx.Int32(
                    buffer_ops.buffer_load(
                        expert_ids_rsrc, safe_lo, vec_width=1, dtype=T.i32
                    )
                )
                if in_metadata & (found_expert == expert):
                    slot = atomic_add(
                        queue_storage,
                        fx.Int32(0),
                        fx.Int32(1),
                        dtype_bytes=4,
                    )
                    queue_index = fx.Int32(slot)
                    if queue_index < i32_queue_capacity:
                        storage_rsrc = buffer_ops.create_buffer_resource(
                            queue_storage, max_size=True
                        )
                        output_index = fx.Int32(1) + queue_index * fx.Int32(2)
                        buffer_ops.buffer_store(expert, storage_rsrc, output_index)
                        buffer_ops.buffer_store(
                            lo * fx.Int32(_SORTED_BLOCK_M),
                            storage_rsrc,
                            output_index + fx.Int32(1),
                        )

    @flyc.jit
    def launch(
        expert_frequency: fx.Tensor,
        sorted_expert_ids: fx.Tensor,
        num_valid_ids: fx.Tensor,
        queue_storage: fx.Tensor,
        i32_queue_capacity: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        clear_queue_count(queue_storage).launch(
            grid=(1, 1, 1),
            block=(_BLOCK_THREADS, 1, 1),
            stream=stream,
        )
        build_queue(
            expert_frequency,
            sorted_expert_ids,
            num_valid_ids,
            queue_storage,
            i32_queue_capacity,
        ).launch(
            grid=((num_experts + _BLOCK_THREADS - 1) // _BLOCK_THREADS, 1, 1),
            block=(_BLOCK_THREADS, 1, 1),
            stream=stream,
        )

    return launch


def grouped_tn_tuning(
    output_m: int, output_n: int
) -> tuple[int, int, int, int, int, int]:
    """Choose ``(BM, BN, BK, K-pad, M-waves, N-waves)`` for grouped TN."""

    if output_m % 64 or output_n % 64:
        raise ValueError("grouped TN requires output dimensions divisible by 64")
    block_m = next(tile for tile in (256, 128, 64) if output_m % tile == 0)
    block_n = next(tile for tile in (256, 128, 64) if output_n % tile == 0)
    m_waves = 4 if block_m == 256 else 2
    n_waves = 4 if block_n == 256 else 2
    return block_m, block_n, 32, 0, m_waves, n_waves


def grouped_dw2_tuning(
    hidden_size: int, intermediate_size: int
) -> tuple[int, int, int, int, int, int]:
    """Semantic alias for dW2 callers of the reusable grouped TN policy."""

    return grouped_tn_tuning(hidden_size, intermediate_size)


@functools.lru_cache(maxsize=64)
def compile_inactive_weight_grad_zero(
    dw1_expert_elements: int,
    dw2_expert_elements: int,
    num_experts: int,
    device_index: int,
    adaptive: bool = False,
    active_count_divisor: int = 1,
    dense_active_ratio: int = 8,
    blocks_per_expert: int = 1,
):
    """Compile an expert-local zero fill for grouped BF16 weight gradients.

    The dense dW1 allocation for the production SonicMoE shape is larger than
    4 GiB.  Address each expert slab with an i64 base and give the buffer
    instruction an expert-local resource, keeping every vector offset within
    the hardware's 32-bit BRSRC window.
    """

    del device_index
    if min(dw1_expert_elements, dw2_expert_elements, num_experts) <= 0:
        raise ValueError("weight-gradient slab sizes and num_experts must be positive")
    if active_count_divisor <= 0:
        raise ValueError("active_count_divisor must be positive")
    if dense_active_ratio <= 0:
        raise ValueError("dense_active_ratio must be positive")
    if (
        not isinstance(blocks_per_expert, int)
        or isinstance(blocks_per_expert, bool)
        or blocks_per_expert <= 0
    ):
        raise ValueError("blocks_per_expert must be a positive int")
    if blocks_per_expert > _ZERO_MAX_BLOCKS_PER_EXPERT:
        raise ValueError(
            f"blocks_per_expert must not exceed {_ZERO_MAX_BLOCKS_PER_EXPERT}"
        )
    max_grid_blocks = _MAX_SIGNED_I32 // _ZERO_BLOCK_THREADS
    if num_experts > max_grid_blocks // blocks_per_expert:
        raise ValueError(
            "inactive-gradient zero launch exceeds signed int32 grid capacity"
        )
    if dw1_expert_elements % _ZERO_VECTOR_ELEMENTS:
        raise ValueError("dW1 expert slabs must have 128-bit size alignment")
    if dw2_expert_elements % _ZERO_VECTOR_ELEMENTS:
        raise ValueError("dW2 expert slabs must have 128-bit size alignment")
    dw1_expert_bytes = dw1_expert_elements * 2
    dw2_expert_bytes = dw2_expert_elements * 2
    if max(dw1_expert_bytes, dw2_expert_bytes) > _MAX_BUFFER_BYTES:
        raise ValueError("each expert-local weight-gradient slab must fit one BRSRC")
    dw1_vectors = dw1_expert_elements // _ZERO_VECTOR_ELEMENTS
    dw2_vectors = dw2_expert_elements // _ZERO_VECTOR_ELEMENTS

    @flyc.kernel(
        name=(
            f"sonic_zero_inactive_weight_grads_e{num_experts}"
            f"_v{dw1_vectors}x{dw2_vectors}"
            f"_a{int(adaptive)}d{active_count_divisor}r{dense_active_ratio}"
            f"_bpe{blocks_per_expert}"
        ),
        known_block_size=[_ZERO_BLOCK_THREADS, 1, 1],
    )
    def zero_inactive_weight_grads_kernel(
        expert_frequency: fx.Tensor,
        active_count_storage: fx.Tensor,
        dw1_base: fx.Int64,
        dw2_base: fx.Int64,
    ):
        block = fx.Int32(gpu.block_idx.x)
        expert = block // fx.Int32(blocks_per_expert)
        expert_block = block % fx.Int32(blocks_per_expert)
        tid = fx.Int32(gpu.thread_idx.x)
        expert_tid = expert_block * fx.Int32(_ZERO_BLOCK_THREADS) + tid
        expert_stride = fx.Int32(blocks_per_expert * _ZERO_BLOCK_THREADS)
        should_clear = fx.Int32(0)
        if const_expr(adaptive):
            active_rsrc = buffer_ops.create_buffer_resource(
                active_count_storage, max_size=True
            )
            active_count = rocdl.readfirstlane(
                T.i32,
                _raw(
                    buffer_ops.buffer_load(
                        active_rsrc,
                        fx.Int32(0),
                        vec_width=1,
                        dtype=T.i32,
                    )
                ),
            ) // fx.Int32(active_count_divisor)
            dense_clear = active_count * fx.Int32(dense_active_ratio) < fx.Int32(
                num_experts
            )
            if dense_clear:
                # All expert CTAs take the same branch.  Clearing active slabs
                # too preserves contiguous write traffic; grouped TN replaces
                # every active element later in the stream.
                should_clear = fx.Int32(1)
            else:
                frequency_rsrc = buffer_ops.create_buffer_resource(
                    expert_frequency,
                    max_size=True,
                )
                frequency = rocdl.readfirstlane(
                    T.i32,
                    _raw(
                        buffer_ops.buffer_load(
                            frequency_rsrc,
                            expert,
                            vec_width=1,
                            dtype=T.i32,
                        )
                    ),
                )
                should_clear = (frequency == fx.Int32(0)).select(
                    fx.Int32(1), fx.Int32(0)
                )
        else:
            frequency_rsrc = buffer_ops.create_buffer_resource(
                expert_frequency, max_size=True
            )
            frequency = rocdl.readfirstlane(
                T.i32,
                _raw(
                    buffer_ops.buffer_load(
                        frequency_rsrc, expert, vec_width=1, dtype=T.i32
                    )
                ),
            )
            should_clear = (frequency == fx.Int32(0)).select(fx.Int32(1), fx.Int32(0))

        if should_clear != fx.Int32(0):
            # Do not form a descriptor over the full dense tensor: dW1 can be
            # larger than 4 GiB, while the AMD buffer offset is only 32 bits.
            # The i64 expert base plus exact expert-local resource keeps both
            # the address and the OOB bound correct for the last expert.
            dw1_addr = dw1_base + fx.Int64(expert) * fx.Int64(dw1_expert_bytes)
            dw2_addr = dw2_base + fx.Int64(expert) * fx.Int64(dw2_expert_bytes)
            dw1_rsrc = buffer_ops.create_buffer_resource_from_addr(
                _raw(dw1_addr),
                num_records_bytes=dw1_expert_bytes,
            )
            dw2_rsrc = buffer_ops.create_buffer_resource_from_addr(
                _raw(dw2_addr),
                num_records_bytes=dw2_expert_bytes,
            )
            # Store raw zero dwords so lowering selects one 128-bit buffer
            # store rather than a typed packed-BF16 sequence.
            zero = fx.Vector.filled(GFX950_DMA_BYTES // 4, 0, fx.Int32)
            for vector_index in range(
                expert_tid,
                fx.Int32(dw1_vectors),
                expert_stride,
            ):
                buffer_ops.buffer_store(
                    zero,
                    dw1_rsrc,
                    vector_index * fx.Int32(GFX950_DMA_BYTES // 4),
                )
            for vector_index in range(
                expert_tid,
                fx.Int32(dw2_vectors),
                expert_stride,
            ):
                buffer_ops.buffer_store(
                    zero,
                    dw2_rsrc,
                    vector_index * fx.Int32(GFX950_DMA_BYTES // 4),
                )

    @flyc.jit
    def launch(
        expert_frequency: fx.Tensor,
        active_count_storage: fx.Tensor,
        dw1_base: fx.Int64,
        dw2_base: fx.Int64,
        stream: fx.Stream = fx.Stream(None),
    ):
        zero_inactive_weight_grads_kernel(
            expert_frequency,
            active_count_storage,
            dw1_base,
            dw2_base,
        ).launch(
            grid=(num_experts * blocks_per_expert, 1, 1),
            block=(_ZERO_BLOCK_THREADS, 1, 1),
            stream=stream,
        )

    return launch


def zero_inactive_weight_grads_flydsl(
    expert_frequency: torch.Tensor,
    dw1: torch.Tensor,
    dw2: torch.Tensor,
    *,
    blocks_per_expert: int = 1,
    stream: torch.cuda.Stream | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Zero only inactive expert slabs before grouped dW1/dW2 overwrite actives."""

    if expert_frequency.ndim != 1:
        raise ValueError("expert_frequency must have shape [E]")
    num_experts = int(expert_frequency.numel())
    if num_experts <= 0:
        raise ValueError("expert_frequency must contain at least one expert")
    if dw1.ndim != 3 or dw2.ndim != 3:
        raise ValueError("dw1 and dw2 must be dense three-dimensional expert weights")
    if int(dw1.shape[0]) != num_experts or int(dw2.shape[0]) != num_experts:
        raise ValueError("weight gradients and expert_frequency must have the same E")
    tensors = (expert_frequency, dw1, dw2)
    if any(tensor.device != expert_frequency.device for tensor in tensors):
        raise ValueError("inactive-gradient zero tensors must share one device")
    if expert_frequency.dtype != torch.int32:
        raise TypeError("expert_frequency must use int32")
    if dw1.dtype != torch.bfloat16 or dw2.dtype != torch.bfloat16:
        raise TypeError("inactive-gradient zero currently requires BF16 outputs")
    if not all(tensor.is_contiguous() for tensor in tensors):
        raise ValueError("inactive-gradient zero tensors must be contiguous")
    dw1_expert_elements = int(dw1.numel()) // num_experts
    dw2_expert_elements = int(dw2.numel()) // num_experts
    if dw1_expert_elements % _ZERO_VECTOR_ELEMENTS:
        raise ValueError("dW1 expert slabs must have 128-bit size alignment")
    if dw2_expert_elements % _ZERO_VECTOR_ELEMENTS:
        raise ValueError("dW2 expert slabs must have 128-bit size alignment")
    if max(dw1_expert_elements, dw2_expert_elements) * 2 > _MAX_BUFFER_BYTES:
        raise ValueError("each expert-local weight-gradient slab must fit one BRSRC")
    if stream is None:
        stream = torch.cuda.current_stream(expert_frequency.device)
    launcher = compile_inactive_weight_grad_zero(
        dw1_expert_elements,
        dw2_expert_elements,
        num_experts,
        expert_frequency.device.index or 0,
        blocks_per_expert=blocks_per_expert,
    )
    _run_compiled(
        launcher,
        expert_frequency,
        expert_frequency,
        dw1.data_ptr(),
        dw2.data_ptr(),
        stream,
    )
    expert_frequency.record_stream(stream)
    dw1.record_stream(stream)
    dw2.record_stream(stream)
    return dw1, dw2


def zero_weight_grads_adaptive_flydsl(
    expert_frequency: torch.Tensor,
    active_count_storage: torch.Tensor,
    dw1: torch.Tensor,
    dw2: torch.Tensor,
    *,
    active_count_divisor: int = 1,
    dense_active_ratio: int = 8,
    blocks_per_expert: int = 1,
    stream: torch.cuda.Stream | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Choose dense-all or inactive-only dW clearing from a device count.

    ``active_count_storage[0] / active_count_divisor`` is the live expert
    count.  Sparse routing clears every slab with contiguous expert-local
    stores; sufficiently dense routing skips slabs that grouped TN overwrites.
    The choice is uniform across the grid and never synchronizes with the host.
    """

    if active_count_storage.ndim != 1 or active_count_storage.numel() < 1:
        raise ValueError("active_count_storage must contain a device count")
    if active_count_storage.device != expert_frequency.device:
        raise ValueError("active_count_storage must share expert_frequency's device")
    if (
        active_count_storage.dtype != torch.int32
        or not active_count_storage.is_contiguous()
    ):
        raise TypeError("active_count_storage must be contiguous int32")
    if not isinstance(active_count_divisor, int) or active_count_divisor <= 0:
        raise ValueError("active_count_divisor must be a positive int")
    if not isinstance(dense_active_ratio, int) or dense_active_ratio <= 0:
        raise ValueError("dense_active_ratio must be a positive int")

    # Reuse the public helper's validation invariants without launching its
    # non-adaptive specialization.
    if expert_frequency.ndim != 1:
        raise ValueError("expert_frequency must have shape [E]")
    num_experts = int(expert_frequency.numel())
    if num_experts <= 0:
        raise ValueError("expert_frequency must contain at least one expert")
    if dw1.ndim != 3 or dw2.ndim != 3:
        raise ValueError("dw1 and dw2 must be dense three-dimensional expert weights")
    if int(dw1.shape[0]) != num_experts or int(dw2.shape[0]) != num_experts:
        raise ValueError("weight gradients and expert_frequency must have the same E")
    tensors = (expert_frequency, active_count_storage, dw1, dw2)
    if any(tensor.device != expert_frequency.device for tensor in tensors):
        raise ValueError("adaptive-zero tensors must share one device")
    if expert_frequency.dtype != torch.int32:
        raise TypeError("expert_frequency must use int32")
    if dw1.dtype != torch.bfloat16 or dw2.dtype != torch.bfloat16:
        raise TypeError("adaptive weight-gradient zero currently requires BF16 outputs")
    if not all(tensor.is_contiguous() for tensor in tensors):
        raise ValueError("adaptive-zero tensors must be contiguous")
    dw1_expert_elements = int(dw1.numel()) // num_experts
    dw2_expert_elements = int(dw2.numel()) // num_experts
    if dw1_expert_elements % _ZERO_VECTOR_ELEMENTS:
        raise ValueError("dW1 expert slabs must have 128-bit size alignment")
    if dw2_expert_elements % _ZERO_VECTOR_ELEMENTS:
        raise ValueError("dW2 expert slabs must have 128-bit size alignment")
    if max(dw1_expert_elements, dw2_expert_elements) * 2 > _MAX_BUFFER_BYTES:
        raise ValueError("each expert-local weight-gradient slab must fit one BRSRC")
    if stream is None:
        stream = torch.cuda.current_stream(expert_frequency.device)
    launcher = compile_inactive_weight_grad_zero(
        dw1_expert_elements,
        dw2_expert_elements,
        num_experts,
        expert_frequency.device.index or 0,
        True,
        active_count_divisor,
        dense_active_ratio,
        blocks_per_expert,
    )
    _run_compiled(
        launcher,
        expert_frequency,
        active_count_storage,
        dw1.data_ptr(),
        dw2.data_ptr(),
        stream,
    )
    expert_frequency.record_stream(stream)
    active_count_storage.record_stream(stream)
    dw1.record_stream(stream)
    dw2.record_stream(stream)
    return dw1, dw2


@functools.lru_cache(maxsize=32)
def compile_hot_split_queues(
    num_experts: int,
    device_index: int,
):
    """Split one active-expert queue into cold and hot split-K queues.

    The input ABI is ``[count, (expert, first_row)*]``.  The cold output keeps
    that ABI.  The split output is
    ``[count, (expert, first_row, valid_rows)*]`` and the hot-expert output is
    ``[count, (expert, first_partition, partition_count)*]``.  All routing
    decisions remain device-side.  The split and hot thresholds are runtime
    scalars so dynamic route counts reuse one compiled queue builder.
    """

    del device_index
    if num_experts <= 0 or num_experts > _BLOCK_THREADS:
        raise ValueError("hot split queue currently requires 1..256 experts")

    @flyc.kernel(known_block_size=[_BLOCK_THREADS, 1, 1])
    def build(
        expert_frequency: fx.Tensor,
        active_queue: fx.Tensor,
        cold_queue: fx.Tensor,
        split_queue: fx.Tensor,
        hot_queue: fx.Tensor,
        i32_cold_capacity: fx.Int32,
        i32_split_capacity: fx.Int32,
        i32_hot_capacity: fx.Int32,
        i32_split_rows: fx.Int32,
        i32_min_hot_rows: fx.Int32,
    ):
        tid = gpu.thread_idx.x
        active_rsrc = buffer_ops.create_buffer_resource(active_queue, max_size=True)
        cold_rsrc = buffer_ops.create_buffer_resource(cold_queue, max_size=True)
        split_rsrc = buffer_ops.create_buffer_resource(split_queue, max_size=True)
        hot_rsrc = buffer_ops.create_buffer_resource(hot_queue, max_size=True)
        frequency_rsrc = buffer_ops.create_buffer_resource(
            expert_frequency, max_size=True
        )
        if tid == fx.Int32(0):
            buffer_ops.buffer_store(fx.Int32(0), cold_rsrc, fx.Int32(0))
            buffer_ops.buffer_store(fx.Int32(0), split_rsrc, fx.Int32(0))
            buffer_ops.buffer_store(fx.Int32(0), hot_rsrc, fx.Int32(0))
        gpu.barrier()

        active_count = fx.Int32(
            buffer_ops.buffer_load(active_rsrc, fx.Int32(0), vec_width=1, dtype=T.i32)
        )
        if tid < active_count:
            source_offset = fx.Int32(1) + tid * fx.Int32(2)
            expert = fx.Int32(
                buffer_ops.buffer_load(
                    active_rsrc, source_offset, vec_width=1, dtype=T.i32
                )
            )
            first_row = fx.Int32(
                buffer_ops.buffer_load(
                    active_rsrc,
                    source_offset + fx.Int32(1),
                    vec_width=1,
                    dtype=T.i32,
                )
            )
            frequency = fx.Int32(
                buffer_ops.buffer_load(frequency_rsrc, expert, vec_width=1, dtype=T.i32)
            )
            if frequency >= i32_min_hot_rows:
                partition_count = (
                    frequency + i32_split_rows - fx.Int32(1)
                ) // i32_split_rows
                first_partition = fx.Int32(
                    atomic_add(
                        split_queue,
                        fx.Int32(0),
                        partition_count,
                        dtype_bytes=4,
                    )
                )
                hot_slot = fx.Int32(
                    atomic_add(hot_queue, fx.Int32(0), fx.Int32(1), dtype_bytes=4)
                )
                if hot_slot < i32_hot_capacity:
                    hot_offset = fx.Int32(1) + hot_slot * fx.Int32(3)
                    buffer_ops.buffer_store(expert, hot_rsrc, hot_offset)
                    buffer_ops.buffer_store(
                        first_partition,
                        hot_rsrc,
                        hot_offset + fx.Int32(1),
                    )
                    buffer_ops.buffer_store(
                        partition_count,
                        hot_rsrc,
                        hot_offset + fx.Int32(2),
                    )
                for local_partition in range(0, partition_count, 1):
                    partition = first_partition + fx.Int32(local_partition)
                    if partition < i32_split_capacity:
                        local_row = fx.Int32(local_partition) * i32_split_rows
                        remaining = frequency - local_row
                        valid_rows = (remaining < i32_split_rows).select(
                            remaining,
                            i32_split_rows,
                        )
                        split_offset = fx.Int32(1) + partition * fx.Int32(3)
                        buffer_ops.buffer_store(expert, split_rsrc, split_offset)
                        buffer_ops.buffer_store(
                            first_row + local_row,
                            split_rsrc,
                            split_offset + fx.Int32(1),
                        )
                        buffer_ops.buffer_store(
                            valid_rows,
                            split_rsrc,
                            split_offset + fx.Int32(2),
                        )
            else:
                cold_slot = fx.Int32(
                    atomic_add(cold_queue, fx.Int32(0), fx.Int32(1), dtype_bytes=4)
                )
                if cold_slot < i32_cold_capacity:
                    cold_offset = fx.Int32(1) + cold_slot * fx.Int32(2)
                    buffer_ops.buffer_store(expert, cold_rsrc, cold_offset)
                    buffer_ops.buffer_store(
                        first_row,
                        cold_rsrc,
                        cold_offset + fx.Int32(1),
                    )

        gpu.barrier()
        if tid == fx.Int32(0):
            cold_count = fx.Int32(
                buffer_ops.buffer_load(cold_rsrc, fx.Int32(0), vec_width=1, dtype=T.i32)
            )
            split_count = fx.Int32(
                buffer_ops.buffer_load(
                    split_rsrc, fx.Int32(0), vec_width=1, dtype=T.i32
                )
            )
            hot_count = fx.Int32(
                buffer_ops.buffer_load(hot_rsrc, fx.Int32(0), vec_width=1, dtype=T.i32)
            )
            buffer_ops.buffer_store(
                (cold_count < i32_cold_capacity).select(cold_count, i32_cold_capacity),
                cold_rsrc,
                fx.Int32(0),
            )
            buffer_ops.buffer_store(
                (split_count < i32_split_capacity).select(
                    split_count, i32_split_capacity
                ),
                split_rsrc,
                fx.Int32(0),
            )
            buffer_ops.buffer_store(
                (hot_count < i32_hot_capacity).select(hot_count, i32_hot_capacity),
                hot_rsrc,
                fx.Int32(0),
            )

    @flyc.jit
    def launch(
        expert_frequency: fx.Tensor,
        active_queue: fx.Tensor,
        cold_queue: fx.Tensor,
        split_queue: fx.Tensor,
        hot_queue: fx.Tensor,
        i32_cold_capacity: fx.Int32,
        i32_split_capacity: fx.Int32,
        i32_hot_capacity: fx.Int32,
        i32_split_rows: fx.Int32,
        i32_min_hot_rows: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        build(
            expert_frequency,
            active_queue,
            cold_queue,
            split_queue,
            hot_queue,
            i32_cold_capacity,
            i32_split_capacity,
            i32_hot_capacity,
            i32_split_rows,
            i32_min_hot_rows,
        ).launch(grid=(1, 1, 1), block=(_BLOCK_THREADS, 1, 1), stream=stream)

    return launch


def build_hot_split_queues_flydsl(
    expert_frequency: torch.Tensor,
    active_queue: torch.Tensor,
    *,
    routes: int,
    split_rows: int,
    min_hot_rows: int,
    cold_queue: torch.Tensor,
    split_queue: torch.Tensor,
    hot_queue: torch.Tensor,
    stream: torch.cuda.Stream | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build device-side cold, split-partition, and hot-expert queues."""

    num_experts = int(expert_frequency.numel())
    cold_capacity = active_expert_descriptor_capacity(routes, num_experts)
    split_capacity = hot_split_descriptor_capacity(
        routes,
        num_experts,
        split_rows,
        min_hot_rows,
    )
    hot_capacity = min(num_experts, routes // min_hot_rows)
    tensors = (
        expert_frequency,
        active_queue,
        cold_queue,
        split_queue,
        hot_queue,
    )
    if any(tensor.device != expert_frequency.device for tensor in tensors):
        raise ValueError("hot split queue tensors must share one device")
    if any(tensor.dtype != torch.int32 for tensor in tensors):
        raise TypeError("hot split queue tensors must use int32")
    if not all(tensor.is_contiguous() and tensor.ndim == 1 for tensor in tensors):
        raise ValueError("hot split queue tensors must be contiguous vectors")
    if active_queue.numel() < 1 + 2 * cold_capacity:
        raise ValueError("active_queue does not cover its route bound")
    if cold_queue.numel() < 1 + 2 * cold_capacity:
        raise ValueError("cold_queue has insufficient capacity")
    if split_queue.numel() < 1 + 3 * split_capacity:
        raise ValueError("split_queue has insufficient capacity")
    if hot_queue.numel() < 1 + 3 * hot_capacity:
        raise ValueError("hot_queue has insufficient capacity")
    if split_capacity == 0 or hot_capacity == 0:
        with torch.cuda.stream(
            torch.cuda.current_stream(expert_frequency.device)
            if stream is None
            else stream
        ):
            cold_queue.copy_(active_queue)
            split_queue[:1].zero_()
            hot_queue[:1].zero_()
        return cold_queue, split_queue, hot_queue
    if stream is None:
        stream = torch.cuda.current_stream(expert_frequency.device)
    launcher = compile_hot_split_queues(
        num_experts,
        expert_frequency.device.index or 0,
    )
    _run_compiled(
        launcher,
        expert_frequency,
        active_queue,
        cold_queue,
        split_queue,
        hot_queue,
        cold_capacity,
        split_capacity,
        hot_capacity,
        split_rows,
        min_hot_rows,
        stream,
    )
    for tensor in tensors:
        tensor.record_stream(stream)
    return cold_queue, split_queue, hot_queue


def grouped_tn_grid_cap(
    block_m: int,
    block_n: int,
    block_k: int,
    stages: int,
    m_waves: int,
    n_waves: int,
    gather_rhs: bool = False,
) -> int:
    """Return a gfx950 persistent-grid cap derived from tile resources.

    Grouped TN is output-stationary, so a CTA retains its FP32 accumulator
    registers until it has reduced the complete expert segment.  The useful
    resident-CTA count is therefore constrained by both its wave footprint
    and the larger of its staged A/B storage and C-shuffle storage.
    """

    values = (block_m, block_n, block_k, stages, m_waves, n_waves)
    if min(values) <= 0:
        raise ValueError("grouped TN grid tuning values must be positive")
    block_threads = m_waves * n_waves * GFX950_WAVE_SIZE
    lds_ab_bytes = stages * (block_m + block_n) * block_k * 2
    if gather_rhs:
        lds_ab_bytes += stages * block_k * 4
    lds_c_bytes = block_m * block_n * 2
    lds_bytes = max(lds_ab_bytes, lds_c_bytes)
    resident_by_threads = _MAX_RESIDENT_THREADS_PER_CU // block_threads
    resident_by_lds = _LDS_BYTES_PER_CU // lds_bytes
    resident_ctas = max(1, min(resident_by_threads, resident_by_lds))
    return _NUM_CU * resident_ctas


def grouped_tn_launch_grid(
    schedule_capacity: int,
    output_m: int,
    output_n: int,
    block_m: int,
    block_n: int,
    block_k: int,
    stages: int,
    m_waves: int,
    n_waves: int,
    gather_rhs: bool = False,
) -> int:
    """Return the host-known persistent launch bound for grouped TN."""

    if schedule_capacity < 0:
        raise ValueError("schedule_capacity must be non-negative")
    output_tiles = (output_m // block_m) * (output_n // block_n)
    max_work = schedule_capacity * output_tiles
    return min(
        max_work,
        grouped_tn_grid_cap(
            block_m,
            block_n,
            block_k,
            stages,
            m_waves,
            n_waves,
            gather_rhs,
        ),
    )


def grouped_tn_queue_3d_launch_grid(
    schedule_capacity: int,
    output_m: int,
    output_n: int,
    block_m: int,
    block_n: int,
    block_k: int,
    stages: int,
    m_waves: int,
    n_waves: int,
    gather_rhs: bool = False,
) -> tuple[int, int, int] | None:
    """Return an occupancy-capped queue 3D dispatch grid.

    The x axis grid-strides over the device queue while y/z directly encode the
    output tile.  If one expert's output tile set already exceeds the persistent
    resource cap, the caller must retain the established flattened 1D schedule.
    """

    if schedule_capacity <= 0:
        return None
    if output_m % block_m or output_n % block_n:
        return None
    num_m_tiles = output_m // block_m
    num_n_tiles = output_n // block_n
    if num_m_tiles > 65535 or num_n_tiles > 65535:
        return None
    output_tiles = num_m_tiles * num_n_tiles
    cap = grouped_tn_grid_cap(
        block_m,
        block_n,
        block_k,
        stages,
        m_waves,
        n_waves,
        gather_rhs,
    )
    if output_tiles > cap:
        return None
    descriptor_lanes = min(schedule_capacity, max(1, cap // output_tiles))
    return descriptor_lanes, num_n_tiles, num_m_tiles


@functools.lru_cache(maxsize=128)
def compile_grouped_tn(
    output_m: int,
    output_n: int,
    num_experts: int,
    block_m: int,
    block_n: int,
    block_k: int,
    k_padding: int,
    m_waves: int,
    n_waves: int,
    device_index: int,
    metadata_direct: bool = False,
    stages: int = 2,
    min_active_experts: int = 0,
    max_active_experts: int | None = None,
    gather_rhs: bool = False,
    filter_expert_rows: bool = False,
    has_max_expert_rows: bool = False,
    active_guard_or_expert_rows: bool = False,
    exclude_hot_experts: bool = False,
    split_k_partials: bool = False,
    queue_3d: bool = False,
):
    """Compile the persistent grouped TN consumer for a prebuilt queue.

    ``block_k`` and ``k_padding`` are separate constexprs so consumers can
    tune the MFMA reduction tile independently from the number of materialized
    sorter rows.  ``k_padding=0`` means the logical reduction bound is the
    actual expert frequency; physical loads still round to ``block_k`` and are
    safe because sorter padding is zero-filled.  ``metadata_direct`` treats
    the schedule tensor as sorter expert IDs and is valid when every active
    expert occupies one 64-row sorter block (the fixed-K T1 fast path).
    ``min_active_experts`` and ``max_active_experts`` optionally guard a
    specialization with the device-resident queue count.  Two disjoint
    guarded launches can therefore select different gfx950 tile profiles
    without synchronizing the routing distribution back to the host.
    ``filter_expert_rows`` enables runtime row bounds, while
    ``has_max_expert_rows`` controls whether the runtime upper bound is used.
    Keeping the numeric bounds out of this cached compiler prevents dynamic
    route counts from creating one JIT specialization per cutoff.  The
    optional OR mode admits every descriptor when the active-count guard
    matches and only row-selected descriptors otherwise; this lets one
    small-tile launch cover both sparse shards and a hot expert in an otherwise
    dense shard.
    ``exclude_hot_experts`` reads a device-produced hot-queue count and omits
    rows at or above a runtime split threshold only when that queue is active.
    A zero hot count therefore leaves the regular contraction unfiltered,
    which lets a device-side cohort gate disable split-K without changing the
    host launch sequence.
    ``gather_rhs`` loads token-major RHS rows through packed sorter token IDs,
    eliminating their otherwise materialized sorter-order copy.
    ``split_k_partials`` consumes
    ``[count, (expert, first_row, valid_rows)*]`` and writes one FP32 output
    matrix per descriptor.  A separate deterministic reduction finalizes those
    matrices into the public BF16 gradient.
    ``queue_3d`` maps the output M/N tiles directly to grid z/y and uses grid x
    only for active queue descriptors.  This specialization is reserved for
    bounded E16 regular-queue profiles; metadata-direct and split-K schedules
    retain the flattened 1D traversal.
    """

    del device_index
    mma_m = 16
    mma_n = 16
    mma_k = 32
    in_data_bytes = 2
    async_load_bytes = GFX950_DMA_BYTES
    async_load_vec_size = async_load_bytes // in_data_bytes
    block_threads = m_waves * n_waves * GFX950_WAVE_SIZE
    ldg_x_threads = block_k // async_load_vec_size
    ldg_a_iters = (block_m * block_k) // (block_threads * async_load_vec_size)
    ldg_b_iters = (block_n * block_k) // (block_threads * async_load_vec_size)
    # Gathered RHS tiles issue one packed-ID VMEM operation in addition to
    # their regular A/B direct-to-LDS operations.  The IDs are staged by the
    # first wave, one dword per lane, instead of being loaded separately by
    # every B load iteration.
    scheduled_vmem_ops = ldg_a_iters + ldg_b_iters + int(gather_rhs)
    mma_m_iters = block_m // (m_waves * mma_m)
    mma_n_iters = block_n // (n_waves * mma_n)
    k_mma_iters = block_k // mma_k
    cshuffle_vec_size = async_load_vec_size
    num_m_tiles = output_m // block_m
    num_n_tiles = output_n // block_n
    output_tiles_per_expert = num_m_tiles * num_n_tiles
    if (
        min(
            output_m,
            output_n,
            num_experts,
            block_m,
            block_n,
            block_k,
            stages,
            m_waves,
            n_waves,
        )
        <= 0
    ):
        raise ValueError("grouped TN dimensions and tuning values must be positive")
    if not isinstance(min_active_experts, int) or min_active_experts < 0:
        raise ValueError("min_active_experts must be a non-negative int")
    if max_active_experts is not None and (
        not isinstance(max_active_experts, int)
        or max_active_experts < min_active_experts
    ):
        raise ValueError(
            "max_active_experts must be None or at least min_active_experts"
        )
    if not isinstance(filter_expert_rows, bool):
        raise TypeError("filter_expert_rows must be a bool")
    if not isinstance(has_max_expert_rows, bool):
        raise TypeError("has_max_expert_rows must be a bool")
    if has_max_expert_rows and not filter_expert_rows:
        raise ValueError("has_max_expert_rows requires filter_expert_rows")
    if not isinstance(active_guard_or_expert_rows, bool):
        raise TypeError("active_guard_or_expert_rows must be a bool")
    if active_guard_or_expert_rows and (
        metadata_direct or max_active_experts is None or not filter_expert_rows
    ):
        raise ValueError(
            "active_guard_or_expert_rows requires queue metadata plus active and row guards"
        )
    if split_k_partials and metadata_direct:
        raise ValueError("split-K partials require queue metadata")
    if not isinstance(queue_3d, bool):
        raise TypeError("queue_3d must be a bool")
    if queue_3d and (num_experts != 16 or metadata_direct or split_k_partials):
        raise ValueError("queue_3d requires a regular E16 active-expert queue")
    if not isinstance(exclude_hot_experts, bool):
        raise TypeError("exclude_hot_experts must be a bool")
    if exclude_hot_experts and (metadata_direct or split_k_partials):
        raise ValueError("hot-expert exclusion requires regular queue metadata")
    if block_k not in (32, 64):
        raise ValueError("grouped TN block_k must be 32 or 64")
    if stages not in (2, 3, 4):
        raise ValueError("grouped TN stages must be 2, 3, or 4")
    if k_padding not in (0, 32, 64):
        raise ValueError("grouped TN k_padding must be 0, 32, or 64")
    if k_padding and k_padding % block_k:
        raise ValueError("grouped TN k_padding must be divisible by block_k")
    if output_m % block_m or output_n % block_n:
        raise ValueError(
            "grouped TN output tiles must divide output dimensions exactly"
        )
    if block_m % (m_waves * mma_m) or block_n % (n_waves * mma_n):
        raise ValueError("grouped TN tile sizes must divide their wave tiling")
    if block_threads > 1024:
        raise ValueError("grouped TN cannot exceed 1024 workgroup threads")
    if block_k % async_load_vec_size:
        raise ValueError("grouped TN block_k must satisfy 128-bit DMA alignment")
    if block_threads % ldg_x_threads:
        raise ValueError("grouped TN thread layout must divide block_k DMA lanes")
    if ldg_a_iters * block_threads * async_load_vec_size != block_m * block_k:
        raise ValueError(
            "grouped TN A tile must have exact whole-workgroup DMA coverage"
        )
    if ldg_b_iters * block_threads * async_load_vec_size != block_n * block_k:
        raise ValueError(
            "grouped TN B tile must have exact whole-workgroup DMA coverage"
        )
    gather_lanes_per_row = block_n // async_load_vec_size
    if gather_rhs and (
        gather_lanes_per_row > GFX950_WAVE_SIZE
        or GFX950_WAVE_SIZE % gather_lanes_per_row
    ):
        raise ValueError(
            "gathered RHS row vectors must form whole groups within one wave"
        )
    token_id_lds_bytes = stages * block_k * 4 if gather_rhs else 0
    lds_ab_bytes = (
        stages * (block_m + block_n) * block_k * in_data_bytes + token_id_lds_bytes
    )
    lds_c_bytes = block_m * block_n * in_data_bytes
    if max(lds_ab_bytes, lds_c_bytes) > 163840:
        raise ValueError("grouped TN tuning exceeds gfx950 LDS capacity")

    if gather_rhs:

        @fx.struct
        class SharedABStorage:
            a: fx.Array[fx.BFloat16, stages * block_m * block_k, 16]
            b: fx.Array[fx.BFloat16, stages * block_n * block_k, 16]
            token_ids: fx.Array[fx.Int32, stages * block_k, 16]

    else:

        @fx.struct
        class SharedABStorage:
            a: fx.Array[fx.BFloat16, stages * block_m * block_k, 16]
            b: fx.Array[fx.BFloat16, stages * block_n * block_k, 16]

    @fx.union
    class SharedStorage:
        ab: SharedABStorage
        c: fx.Array[fx.BFloat16, block_m * block_n, 16]

    @flyc.kernel(
        name=(
            f"sonic_grouped_tn_bf16_m{output_m}_n{output_n}_e{num_experts}"
            f"_bm{block_m}_bn{block_n}_bk{block_k}_s{stages}_kp{k_padding}_w{m_waves}x{n_waves}"
            f"_md{int(metadata_direct)}"
            f"_gr{int(gather_rhs)}"
            f"_skp{int(split_k_partials)}"
            f"_q3d{int(queue_3d)}"
            f"_amin{min_active_experts}_amax{max_active_experts}"
            f"_rf{int(filter_expert_rows)}_rmax{int(has_max_expert_rows)}"
            f"_aor{int(active_guard_or_expert_rows)}"
            f"_hx{int(exclude_hot_experts)}"
        ),
        known_block_size=[block_threads, 1, 1],
    )
    def grouped_tn_kernel(
        lhs_rows: fx.Tensor,
        rhs_rows: fx.Tensor,
        sorted_token_ids: fx.Tensor,
        expert_frequency: fx.Tensor,
        schedule_storage: fx.Tensor,
        num_valid_ids: fx.Tensor,
        output: fx.Tensor,
        i32_min_expert_rows: fx.Int32,
        i32_max_expert_rows: fx.Int32,
        hot_expert_storage: fx.Tensor,
        i32_hot_split_min_rows: fx.Int32,
        tiled_mma: fx.TiledMma,
    ):
        tid = gpu.thread_idx.x
        bid = fx.Int32(gpu.block_idx.x)
        grid_size = fx.Int32(gpu.grid_dim.x)
        storage_rsrc = buffer_ops.create_buffer_resource(
            schedule_storage, max_size=True
        )
        if const_expr(metadata_direct):
            valid_rsrc = buffer_ops.create_buffer_resource(num_valid_ids, max_size=True)
            padded_rows = rocdl.readfirstlane(
                T.i32,
                _raw(
                    buffer_ops.buffer_load(
                        valid_rsrc, fx.Int32(0), vec_width=1, dtype=T.i32
                    )
                ),
            )
            descriptor_count = padded_rows // fx.Int32(_SORTED_BLOCK_M)
        else:
            descriptor_count = rocdl.readfirstlane(
                T.i32,
                _raw(
                    buffer_ops.buffer_load(
                        storage_rsrc,
                        fx.Int32(0),
                        vec_width=1,
                        dtype=T.i32,
                    )
                ),
            )
        work_bound = descriptor_count * fx.Int32(output_tiles_per_expert)
        if const_expr(min_active_experts > 0 and not active_guard_or_expert_rows):
            work_bound = (descriptor_count >= fx.Int32(min_active_experts)).select(
                work_bound,
                fx.Int32(0),
            )
        if const_expr(
            max_active_experts is not None and not active_guard_or_expert_rows
        ):
            work_bound = (descriptor_count <= fx.Int32(max_active_experts)).select(
                work_bound,
                fx.Int32(0),
            )

        storage = fx.SharedAllocator().allocate(SharedStorage)
        smem_a = storage.ab.a.peek().ptr
        smem_b = storage.ab.b.peek().ptr
        # Keep the non-gather specialization's storage type and LDS footprint
        # unchanged.  This alias is never consumed after constexpr folding.
        smem_token_ids = storage.ab.token_ids.peek().ptr if gather_rhs else smem_b
        smem_c = storage.c.peek().ptr
        lhs_base_addr = fx.Int64(fx.ptrtoint(fx.get_iter(lhs_rows)))
        rhs_base_addr = fx.Int64(fx.ptrtoint(fx.get_iter(rhs_rows)))
        output_base_addr = fx.Int64(fx.ptrtoint(fx.get_iter(output)))
        frequency_rsrc = buffer_ops.create_buffer_resource(
            expert_frequency, max_size=True
        )
        rhs_row_count = fx.Int32(fx.get_scalar(rhs_rows.shape[0]))
        rhs_bytes = fx.Int64(rhs_row_count) * fx.Int64(output_n * in_data_bytes)
        full_rhs_rsrc = buffer_ops.create_buffer_resource(
            rhs_rows,
            max_size=False,
            num_records_bytes=_raw(rhs_bytes),
        )
        token_ids_rsrc = buffer_ops.create_buffer_resource(
            sorted_token_ids, max_size=True
        )

        a_read_atom = fx.make_copy_atom(
            fx.rocdl.cdna4.LDSReadTrans16_64b(), fx.BFloat16
        )
        b_read_atom = fx.make_copy_atom(
            fx.rocdl.cdna4.LDSReadTrans16_64b(), fx.BFloat16
        )
        a_lds_layout = make_transposed_lds_layout(block_m, block_k)
        b_lds_layout = make_transposed_lds_layout(block_n, block_k)
        c_lds_layout = fx.make_layout((block_m, block_n), (block_n, 1))
        thr_mma = tiled_mma.thr_slice(tid)
        thr_copy_a = fx.make_tiled_copy_A(a_read_atom, tiled_mma).get_slice(tid)
        thr_copy_b = fx.make_tiled_copy_B(b_read_atom, tiled_mma).get_slice(tid)
        row_coords = fx.make_view(0, fx.make_layout((block_m, block_n), (1, 0)))
        col_coords = fx.make_view(0, fx.make_layout((block_m, block_n), (0, 1)))
        thr_mma_crow = thr_mma.partition_C(row_coords)
        thr_mma_ccol = thr_mma.partition_C(col_coords)
        wave_offset = get_wave_lds_offset(tid, async_load_bytes)

        def run_output_tile_unchecked(work_index):
            if const_expr(queue_3d):
                descriptor_index = work_index
                block_m_index = fx.Int32(gpu.block_idx.z)
                block_n_index = fx.Int32(gpu.block_idx.y)
            else:
                descriptor_index = work_index // fx.Int32(output_tiles_per_expert)
                output_tile = work_index % fx.Int32(output_tiles_per_expert)
                block_m_index = output_tile // fx.Int32(num_n_tiles)
                block_n_index = output_tile % fx.Int32(num_n_tiles)
            if const_expr(split_k_partials):
                descriptor_offset = fx.Int32(1) + descriptor_index * fx.Int32(3)
                expert = rocdl.readfirstlane(
                    T.i32,
                    _raw(
                        buffer_ops.buffer_load(
                            storage_rsrc,
                            descriptor_offset,
                            vec_width=1,
                            dtype=T.i32,
                        )
                    ),
                )
                first_sorted_row = rocdl.readfirstlane(
                    T.i32,
                    _raw(
                        buffer_ops.buffer_load(
                            storage_rsrc,
                            descriptor_offset + fx.Int32(1),
                            vec_width=1,
                            dtype=T.i32,
                        )
                    ),
                )
                frequency = rocdl.readfirstlane(
                    T.i32,
                    _raw(
                        buffer_ops.buffer_load(
                            storage_rsrc,
                            descriptor_offset + fx.Int32(2),
                            vec_width=1,
                            dtype=T.i32,
                        )
                    ),
                )
            elif const_expr(metadata_direct):
                expert = rocdl.readfirstlane(
                    T.i32,
                    _raw(
                        buffer_ops.buffer_load(
                            storage_rsrc,
                            descriptor_index,
                            vec_width=1,
                            dtype=T.i32,
                        )
                    ),
                )
                first_sorted_row = descriptor_index * fx.Int32(_SORTED_BLOCK_M)
            else:
                descriptor_offset = fx.Int32(1) + descriptor_index * fx.Int32(2)
                expert = rocdl.readfirstlane(
                    T.i32,
                    _raw(
                        buffer_ops.buffer_load(
                            storage_rsrc,
                            descriptor_offset,
                            vec_width=1,
                            dtype=T.i32,
                        )
                    ),
                )
                first_sorted_row = rocdl.readfirstlane(
                    T.i32,
                    _raw(
                        buffer_ops.buffer_load(
                            storage_rsrc,
                            descriptor_offset + fx.Int32(1),
                            vec_width=1,
                            dtype=T.i32,
                        )
                    ),
                )
            if const_expr(not split_k_partials):
                frequency = rocdl.readfirstlane(
                    T.i32,
                    _raw(
                        buffer_ops.buffer_load(
                            frequency_rsrc, expert, vec_width=1, dtype=T.i32
                        )
                    ),
                )
            padding_quantum = block_k if k_padding == 0 else k_padding
            padded_k = (
                (frequency + fx.Int32(padding_quantum - 1)) // fx.Int32(padding_quantum)
            ) * fx.Int32(padding_quantum)
            k_tiles = padded_k // fx.Int32(block_k)
            resource_rows = frequency if k_padding == 0 else padded_k
            lhs_addr = lhs_base_addr + fx.Int64(first_sorted_row) * fx.Int64(
                output_m * in_data_bytes
            )
            lhs_rsrc = buffer_ops.create_buffer_resource_from_addr(
                _raw(lhs_addr),
                num_records_bytes=_raw(
                    fx.Int64(resource_rows) * fx.Int64(output_m * in_data_bytes)
                ),
            )
            rhs_rsrc = full_rhs_rsrc
            if const_expr(not gather_rhs):
                rhs_addr = rhs_base_addr + fx.Int64(first_sorted_row) * fx.Int64(
                    output_n * in_data_bytes
                )
                rhs_rsrc = buffer_ops.create_buffer_resource_from_addr(
                    _raw(rhs_addr),
                    num_records_bytes=_raw(
                        fx.Int64(resource_rows) * fx.Int64(output_n * in_data_bytes)
                    ),
                )
            block_m_offset = block_m_index * fx.Int32(block_m)
            block_n_offset = block_n_index * fx.Int32(block_n)

            output_owner = descriptor_index if split_k_partials else expert
            output_data_bytes = 4 if split_k_partials else in_data_bytes
            output_addr = output_base_addr + fx.Int64(output_owner) * fx.Int64(
                output_m * output_n * output_data_bytes
            )
            output_rsrc = buffer_ops.create_buffer_resource_from_addr(
                _raw(output_addr),
                num_records_bytes=output_m * output_n * output_data_bytes,
            )
            expert_out = fx.rocdl.make_buffer_tensor(
                fx.make_view(
                    (
                        _global_f32_ptr(output_addr)
                        if split_k_partials
                        else _global_bf16_ptr(output_addr)
                    ),
                    fx.make_layout((output_m, output_n), (output_n, 1)),
                ),
                max_size=False,
            )
            g_c = fx.flat_divide(expert_out, (block_m, block_n))[
                None,
                None,
                block_m_index,
                block_n_index,
            ]
            s_a = fx.make_view(smem_a, a_lds_layout)
            s_b = fx.make_view(smem_b, b_lds_layout)
            s_c = fx.make_view(smem_c, c_lds_layout)
            frag_a = thr_mma.make_fragment_A(s_a)
            frag_b = thr_mma.make_fragment_B(s_b)
            frag_c = thr_mma.make_fragment_C(g_c)
            frag_a_retile = thr_copy_a.retile(frag_a)
            frag_b_retile = thr_copy_b.retile(frag_b)
            frag_c.fill(0.0)

            async_context = (
                wave_offset,
                tid,
                block_threads,
                async_load_vec_size,
                ldg_x_threads,
                0,
                block_k,
                in_data_bytes,
                async_load_bytes,
            )

            def load_a(k_tile, stage):
                async_load_to_lds(
                    smem_a + stage * block_m * block_k,
                    lhs_rsrc,
                    a_lds_layout,
                    block_m,
                    output_m,
                    block_m_offset,
                    output_m,
                    ldg_a_iters,
                    True,
                    k_tile,
                    async_context,
                )

            def load_b_serial(k_tile, stage):
                if const_expr(gather_rhs):
                    lds_ptr = make_wave_lds_ptr(
                        smem_b + stage * block_n * block_k,
                        wave_offset,
                    )
                    # Every row is consumed by BN/8 contiguous lanes.  Have
                    # one lane fetch its packed token once, retain all this
                    # thread's load-iteration values in VGPRs, then broadcast
                    # within the wave before issuing the dependent 16-byte
                    # row-vector DMAs.  This avoids both redundant metadata
                    # traffic and an LDS index-staging/barrier round trip.
                    packed_lanes = []
                    load_coordinates = []
                    tile_valid = k_tile < k_tiles
                    for load_iter in range_constexpr(ldg_b_iters):
                        global_tid = fx.Int32(block_threads * load_iter) + tid
                        outer_lds_idx = (
                            global_tid % fx.Int32(gather_lanes_per_row)
                        ) * fx.Int32(async_load_vec_size)
                        k_local_idx = global_tid // fx.Int32(gather_lanes_per_row)
                        outer_local_idx = transposed_contiguous_idx(
                            outer_lds_idx,
                            k_local_idx,
                            b_lds_layout,
                            block_n,
                        )
                        sorted_row = (
                            first_sorted_row + k_tile * fx.Int32(block_k) + k_local_idx
                        )
                        safe_sorted_row = tile_valid.select(
                            sorted_row, first_sorted_row
                        )
                        lane = tid % fx.Int32(GFX950_WAVE_SIZE)
                        packed_lane = fx.Int32(0)
                        if lane % fx.Int32(gather_lanes_per_row) == fx.Int32(0):
                            packed_lane = fx.Int32(
                                buffer_ops.buffer_load(
                                    token_ids_rsrc,
                                    safe_sorted_row,
                                    vec_width=1,
                                    dtype=T.i32,
                                )
                            )
                        source_lane = lane - lane % fx.Int32(gather_lanes_per_row)
                        packed_lanes.append((packed_lane, source_lane))
                        load_coordinates.append(
                            (lds_ptr, block_n_offset + outer_local_idx)
                        )
                        if load_iter < ldg_b_iters - 1:
                            lds_ptr = lds_ptr + fx.Int32(
                                block_threads * async_load_bytes
                            )

                    for load_iter in range_constexpr(ldg_b_iters):
                        packed_lane, source_lane = packed_lanes[load_iter]
                        packed = fx.Int32(
                            rocdl.ds_bpermute(
                                T.i32,
                                source_lane * fx.Int32(4),
                                packed_lane,
                            )
                        )
                        target_lds_ptr, global_col = load_coordinates[load_iter]
                        decoded_token = packed & fx.Int32(_TOKEN_MASK)
                        token_valid = tile_valid & (decoded_token < rhs_row_count)
                        token = token_valid.select(decoded_token, rhs_row_count)
                        safe_global_col = token_valid.select(global_col, fx.Int32(0))
                        global_byte = (
                            token * fx.Int32(output_n) + safe_global_col
                        ) * fx.Int32(in_data_bytes)
                        buffer_load_lds_inline(
                            full_rhs_rsrc,
                            target_lds_ptr,
                            global_byte,
                            async_load_bytes,
                        )
                else:
                    async_load_to_lds(
                        smem_b + stage * block_n * block_k,
                        rhs_rsrc,
                        b_lds_layout,
                        block_n,
                        output_n,
                        block_n_offset,
                        output_n,
                        ldg_b_iters,
                        True,
                        k_tile,
                        async_context,
                    )

            def prefetch_token_ids(k_tile, stage):
                if const_expr(gather_rhs):
                    # A single wave writes one packed int32 ID per K row.
                    # buffer_load ... lds adds lane*4 to this uniform m0 base,
                    # so BK32 naturally uses the low half-wave and BK64 the
                    # full wave.  The existing K-stage barrier publishes this
                    # LDS region; no extra workgroup barrier is introduced.
                    tile_valid = k_tile < k_tiles
                    if tid < fx.Int32(block_k):
                        sorted_row = first_sorted_row + k_tile * fx.Int32(block_k) + tid
                        safe_sorted_row = tile_valid.select(
                            sorted_row, first_sorted_row
                        )
                        token_lds_ptr = fx.recast_iter(
                            fx.Int8,
                            smem_token_ids + stage * fx.Int32(block_k),
                        )
                        buffer_load_lds_inline(
                            token_ids_rsrc,
                            token_lds_ptr,
                            safe_sorted_row * fx.Int32(4),
                            4,
                        )

            def load_b_from_prefetched_ids(k_tile, stage):
                if const_expr(gather_rhs):
                    lds_ptr = make_wave_lds_ptr(
                        smem_b + stage * block_n * block_k,
                        wave_offset,
                    )
                    tile_valid = k_tile < k_tiles
                    for load_iter in range_constexpr(ldg_b_iters):
                        global_tid = fx.Int32(block_threads * load_iter) + tid
                        outer_lds_idx = (
                            global_tid % fx.Int32(gather_lanes_per_row)
                        ) * fx.Int32(async_load_vec_size)
                        k_local_idx = global_tid // fx.Int32(gather_lanes_per_row)
                        outer_local_idx = transposed_contiguous_idx(
                            outer_lds_idx,
                            k_local_idx,
                            b_lds_layout,
                            block_n,
                        )
                        lane = tid % fx.Int32(GFX950_WAVE_SIZE)
                        packed_lane = fx.Int32(0)
                        if lane % fx.Int32(gather_lanes_per_row) == fx.Int32(0):
                            packed_lane = fx.Int32(
                                fx.ptr_load(
                                    smem_token_ids
                                    + stage * fx.Int32(block_k)
                                    + k_local_idx
                                )
                            )
                        source_lane = lane - lane % fx.Int32(gather_lanes_per_row)
                        packed = fx.Int32(
                            rocdl.ds_bpermute(
                                T.i32,
                                source_lane * fx.Int32(4),
                                packed_lane,
                            )
                        )
                        decoded_token = packed & fx.Int32(_TOKEN_MASK)
                        token_valid = tile_valid & (decoded_token < rhs_row_count)
                        token = token_valid.select(decoded_token, rhs_row_count)
                        global_col = block_n_offset + outer_local_idx
                        safe_global_col = token_valid.select(global_col, fx.Int32(0))
                        global_byte = (
                            token * fx.Int32(output_n) + safe_global_col
                        ) * fx.Int32(in_data_bytes)
                        buffer_load_lds_inline(
                            full_rhs_rsrc,
                            lds_ptr,
                            global_byte,
                            async_load_bytes,
                        )
                        if load_iter < ldg_b_iters - 1:
                            lds_ptr = lds_ptr + fx.Int32(
                                block_threads * async_load_bytes
                            )

            def compute_stage(read_stage):
                stage_a = fx.make_view(
                    smem_a + read_stage * block_m * block_k,
                    a_lds_layout,
                )
                stage_b = fx.make_view(
                    smem_b + read_stage * block_n * block_k,
                    b_lds_layout,
                )
                copy_a = thr_copy_a.partition_S(stage_a)
                copy_b = thr_copy_b.partition_S(stage_b)
                for k_iter in range_constexpr(k_mma_iters):
                    fx.copy(
                        b_read_atom,
                        copy_b[None, None, k_iter],
                        frag_b_retile[None, None, k_iter],
                    )
                    fx.copy(
                        a_read_atom,
                        copy_a[None, None, k_iter],
                        frag_a_retile[None, None, k_iter],
                    )
                    fx.gemm(
                        tiled_mma,
                        frag_c,
                        frag_a[None, None, k_iter],
                        frag_b[None, None, k_iter],
                        frag_c,
                        traversal_order=fx.GemmTraversalOrder.KNM,
                    )

            for stage in range_constexpr(stages - 1):
                load_b_serial(fx.Int32(stage), stage)
                load_a(fx.Int32(stage), stage)
            rocdl.sched_barrier(0)
            raw_main_loop_end = k_tiles - fx.Int32(stages - 1)
            main_loop_end = raw_main_loop_end
            if const_expr(stages > 2):
                main_loop_end = (raw_main_loop_end > fx.Int32(0)).select(
                    raw_main_loop_end,
                    fx.Int32(0),
                )
            if const_expr(gather_rhs):
                if main_loop_end > fx.Int32(0):
                    prefetch_token_ids(
                        fx.Int32(stages - 1),
                        fx.Int32(stages - 1),
                    )
            for k_tile in range(fx.Int32(0), main_loop_end, fx.Int32(1)):
                current_stage = k_tile % fx.Int32(stages)
                write_stage = (current_stage + fx.Int32(stages - 1)) % fx.Int32(stages)
                # The prefetched metadata is the newest VMEM operation.  A
                # non-zero vmcnt could therefore publish its LDS stage too
                # early.  Stage-2 already waited at vmcnt(0); keep that exact
                # safety rule for experimental deep gather pipelines until a
                # separately ordered metadata counter is available.
                if const_expr(gather_rhs):
                    __barrier(0)
                    load_b_from_prefetched_ids(
                        k_tile + fx.Int32(stages - 1),
                        write_stage,
                    )
                else:
                    __barrier((stages - 2) * (ldg_a_iters + ldg_b_iters))
                    load_b_serial(k_tile + fx.Int32(stages - 1), write_stage)
                load_a(k_tile + fx.Int32(stages - 1), write_stage)
                if const_expr(gather_rhs):
                    if k_tile + fx.Int32(1) < main_loop_end:
                        prefetch_token_ids(
                            k_tile + fx.Int32(stages),
                            current_stage,
                        )
                compute_stage(current_stage)
                rocdl.sched_vmem(scheduled_vmem_ops)
                for _ in range_constexpr(k_mma_iters):
                    rocdl.sched_dsrd(mma_n_iters)
                    rocdl.sched_dsrd(mma_m_iters)
                    for _ in range_constexpr(mma_m_iters):
                        rocdl.sched_mfma(mma_n_iters)
                rocdl.sched_barrier(0)

            current_stage = main_loop_end % fx.Int32(stages)
            for drain in range_constexpr(stages - 1):
                __barrier((stages - 2 - drain) * (ldg_a_iters + ldg_b_iters))
                if const_expr(stages == 2):
                    compute_stage(current_stage)
                else:
                    if fx.Int32(drain) < k_tiles:
                        compute_stage(current_stage)
                current_stage = (current_stage + fx.Int32(1)) % fx.Int32(stages)

            if const_expr(split_k_partials):
                # Each MFMA fragment element has a unique output coordinate.
                # Direct FP32 stores avoid doubling the C-shuffle LDS footprint
                # beyond gfx950's 160-KiB limit.  The deterministic finalize
                # kernel later sums partitions and performs the sole BF16 cast.
                for value_index in range_constexpr(fx.size(frag_c.shape).unpack()):
                    row = fx.get_scalar(thr_mma_crow[value_index])
                    column = fx.get_scalar(thr_mma_ccol[value_index])
                    output_offset = (
                        (block_m_offset + row) * fx.Int32(output_n)
                        + block_n_offset
                        + column
                    )
                    buffer_ops.buffer_store(
                        fx.Float32(frag_c[value_index]),
                        output_rsrc,
                        output_offset,
                    )
            else:
                frag_c_out = fx.make_fragment_like(frag_c, fx.BFloat16)
                frag_c_out.store(frag_c.load().to(fx.BFloat16))
                gpu.barrier()
                for value_index in range_constexpr(fx.size(frag_c_out.shape).unpack()):
                    row = fx.get_scalar(thr_mma_crow[value_index])
                    column = fx.get_scalar(thr_mma_ccol[value_index])
                    s_c[row, column] = frag_c_out[value_index]
                gpu.barrier()

                vectors_per_row = block_n // cshuffle_vec_size
                tile_vectors = block_m * vectors_per_row
                store_iters = (tile_vectors + block_threads - 1) // block_threads
                for store_iter in range_constexpr(store_iters):
                    vector_index = fx.Int32(block_threads * store_iter) + tid
                    if vector_index < fx.Int32(tile_vectors):
                        local_row = vector_index // fx.Int32(vectors_per_row)
                        local_col = (
                            vector_index % fx.Int32(vectors_per_row)
                        ) * fx.Int32(cshuffle_vec_size)
                        global_row = block_m_offset + local_row
                        global_col = block_n_offset + local_col
                        value = fx.ptr_load(
                            smem_c + local_row * fx.Int32(block_n) + local_col,
                            result_type=fx.Vector.make_type(
                                cshuffle_vec_size, fx.BFloat16
                            ),
                        )
                        output_offset = global_row * fx.Int32(output_n) + global_col
                        buffer_ops.buffer_store(value, output_rsrc, output_offset)

        def run_output_tile(work_index):
            if const_expr(not filter_expert_rows and not exclude_hot_experts):
                run_output_tile_unchecked(work_index)
            else:
                descriptor_index = (
                    work_index
                    if const_expr(queue_3d)
                    else work_index // fx.Int32(output_tiles_per_expert)
                )
                if const_expr(metadata_direct):
                    expert = rocdl.readfirstlane(
                        T.i32,
                        _raw(
                            buffer_ops.buffer_load(
                                storage_rsrc,
                                descriptor_index,
                                vec_width=1,
                                dtype=T.i32,
                            )
                        ),
                    )
                else:
                    descriptor_offset = fx.Int32(1) + descriptor_index * fx.Int32(2)
                    expert = rocdl.readfirstlane(
                        T.i32,
                        _raw(
                            buffer_ops.buffer_load(
                                storage_rsrc,
                                descriptor_offset,
                                vec_width=1,
                                dtype=T.i32,
                            )
                        ),
                    )
                frequency = rocdl.readfirstlane(
                    T.i32,
                    _raw(
                        buffer_ops.buffer_load(
                            frequency_rsrc,
                            expert,
                            vec_width=1,
                            dtype=T.i32,
                        )
                    ),
                )
                selected = frequency >= fx.Int32(0)
                if const_expr(filter_expert_rows):
                    selected = frequency >= i32_min_expert_rows
                    if const_expr(has_max_expert_rows):
                        selected = selected & (frequency <= i32_max_expert_rows)
                    if const_expr(active_guard_or_expert_rows):
                        active_selected = descriptor_count >= fx.Int32(
                            min_active_experts
                        )
                        if const_expr(max_active_experts is not None):
                            active_selected = active_selected & (
                                descriptor_count <= fx.Int32(max_active_experts)
                            )
                        selected = selected | active_selected
                if const_expr(exclude_hot_experts):
                    hot_rsrc = buffer_ops.create_buffer_resource(
                        hot_expert_storage,
                        max_size=True,
                    )
                    hot_count = rocdl.readfirstlane(
                        T.i32,
                        _raw(
                            buffer_ops.buffer_load(
                                hot_rsrc,
                                fx.Int32(0),
                                vec_width=1,
                                dtype=T.i32,
                            )
                        ),
                    )
                    selected = selected & (
                        (hot_count == fx.Int32(0))
                        | (frequency < i32_hot_split_min_rows)
                    )
                if selected:
                    run_output_tile_unchecked(work_index)

        if const_expr(queue_3d):
            descriptor_bound = work_bound // fx.Int32(output_tiles_per_expert)
            if bid < descriptor_bound:
                run_output_tile(bid)
            for descriptor_index in range(
                bid + grid_size,
                descriptor_bound,
                grid_size,
            ):
                gpu.barrier()
                run_output_tile(fx.Int32(descriptor_index))
        else:
            if bid < work_bound:
                run_output_tile(bid)
            for work_index in range(bid + grid_size, work_bound, grid_size):
                gpu.barrier()
                run_output_tile(fx.Int32(work_index))

    if gather_rhs:

        @flyc.jit
        def launch(
            lhs_rows: fx.Tensor,
            rhs_rows: fx.Tensor,
            sorted_token_ids: fx.Tensor,
            expert_frequency: fx.Tensor,
            schedule_storage: fx.Tensor,
            num_valid_ids: fx.Tensor,
            output: fx.Tensor,
            i32_min_expert_rows: fx.Int32,
            i32_max_expert_rows: fx.Int32,
            hot_expert_storage: fx.Tensor,
            i32_hot_split_min_rows: fx.Int32,
            i32_grid: fx.Int32,
            stream: fx.Stream = fx.Stream(None),
        ):
            mma_atom = fx.make_mma_atom(fx.rocdl.MFMA(mma_m, mma_n, mma_k, fx.BFloat16))
            tiled_mma = fx.make_tiled_mma(
                mma_atom,
                fx.make_layout((m_waves, n_waves, 1), (n_waves, 1, 0)),
                fx.make_tile(
                    None,
                    None,
                    fx.make_layout((mma_k // 4, 4), (1, mma_k // 4)),
                ),
            )
            grouped_tn_kernel(
                lhs_rows,
                rhs_rows,
                sorted_token_ids,
                expert_frequency,
                schedule_storage,
                num_valid_ids,
                output,
                i32_min_expert_rows,
                i32_max_expert_rows,
                hot_expert_storage,
                i32_hot_split_min_rows,
                tiled_mma,
            ).launch(
                grid=(
                    (i32_grid, num_n_tiles, num_m_tiles)
                    if queue_3d
                    else (i32_grid, 1, 1)
                ),
                block=(block_threads, 1, 1),
                stream=stream,
            )

    else:

        @flyc.jit
        def launch(
            lhs_rows: fx.Tensor,
            rhs_rows: fx.Tensor,
            expert_frequency: fx.Tensor,
            schedule_storage: fx.Tensor,
            num_valid_ids: fx.Tensor,
            output: fx.Tensor,
            i32_min_expert_rows: fx.Int32,
            i32_max_expert_rows: fx.Int32,
            hot_expert_storage: fx.Tensor,
            i32_hot_split_min_rows: fx.Int32,
            i32_grid: fx.Int32,
            stream: fx.Stream = fx.Stream(None),
        ):
            mma_atom = fx.make_mma_atom(fx.rocdl.MFMA(mma_m, mma_n, mma_k, fx.BFloat16))
            tiled_mma = fx.make_tiled_mma(
                mma_atom,
                fx.make_layout((m_waves, n_waves, 1), (n_waves, 1, 0)),
                fx.make_tile(
                    None,
                    None,
                    fx.make_layout((mma_k // 4, 4), (1, mma_k // 4)),
                ),
            )
            grouped_tn_kernel(
                lhs_rows,
                rhs_rows,
                schedule_storage,
                expert_frequency,
                schedule_storage,
                num_valid_ids,
                output,
                i32_min_expert_rows,
                i32_max_expert_rows,
                hot_expert_storage,
                i32_hot_split_min_rows,
                tiled_mma,
            ).launch(
                grid=(
                    (i32_grid, num_n_tiles, num_m_tiles)
                    if queue_3d
                    else (i32_grid, 1, 1)
                ),
                block=(block_threads, 1, 1),
                stream=stream,
            )

    return launch


def grouped_tn_splitk_from_queue_flydsl(
    lhs_rows: torch.Tensor,
    rhs_rows: torch.Tensor,
    expert_frequency: torch.Tensor,
    split_queue: torch.Tensor,
    partials: torch.Tensor,
    *,
    sorted_token_ids: torch.Tensor | None = None,
    block_m: int = 128,
    block_n: int = 128,
    block_k: int = 32,
    m_waves: int = 2,
    n_waves: int = 2,
    stages: int = 3,
    stream: torch.cuda.Stream | None = None,
) -> torch.Tensor:
    """Compute FP32 TN partials from hot-expert split descriptors."""

    if lhs_rows.ndim != 2 or rhs_rows.ndim != 2 or partials.ndim != 3:
        raise ValueError("split-K TN expects lhs[P,M], rhs[P,N], partials[S,M,N]")
    gather_rhs = sorted_token_ids is not None
    if not gather_rhs and lhs_rows.shape[0] != rhs_rows.shape[0]:
        raise ValueError("split-K TN inputs must share their sorted row extent")
    num_experts = int(expert_frequency.numel())
    capacity, output_m, output_n = (int(value) for value in partials.shape)
    if int(lhs_rows.shape[1]) != output_m or int(rhs_rows.shape[1]) != output_n:
        raise ValueError("split-K TN partial width must match its inputs")
    tensors = (lhs_rows, rhs_rows, expert_frequency, split_queue, partials)
    if sorted_token_ids is not None:
        tensors = (*tensors, sorted_token_ids)
    if any(tensor.device != lhs_rows.device for tensor in tensors):
        raise ValueError("split-K TN tensors must share one device")
    if lhs_rows.dtype != torch.bfloat16 or rhs_rows.dtype != torch.bfloat16:
        raise TypeError("split-K TN inputs must use BF16")
    if expert_frequency.dtype != torch.int32 or split_queue.dtype != torch.int32:
        raise TypeError("split-K TN metadata must use int32")
    if sorted_token_ids is not None and sorted_token_ids.dtype != torch.int32:
        raise TypeError("split-K gathered RHS token IDs must use int32")
    if partials.dtype != torch.float32:
        raise TypeError("split-K TN partials must use FP32")
    if not all(tensor.is_contiguous() for tensor in tensors):
        raise ValueError("split-K TN tensors must be contiguous")
    if (
        split_queue.ndim != 1
        or split_queue.numel() < 1
        or (split_queue.numel() - 1) % 3
    ):
        raise ValueError("split_queue must use [count, (expert,row,rows)*] ABI")
    if capacity < (split_queue.numel() - 1) // 3:
        raise ValueError("partials must cover split queue capacity")
    if capacity == 0:
        return partials
    if partials.numel() * partials.element_size() > _MAX_BUFFER_BYTES:
        raise ValueError("split-K partial storage exceeds one gfx950 BRSRC")
    grid = grouped_tn_launch_grid(
        capacity,
        output_m,
        output_n,
        block_m,
        block_n,
        block_k,
        stages,
        m_waves,
        n_waves,
        gather_rhs,
    )
    if stream is None:
        stream = torch.cuda.current_stream(lhs_rows.device)
    launcher = compile_grouped_tn(
        output_m,
        output_n,
        num_experts,
        block_m,
        block_n,
        block_k,
        0,
        m_waves,
        n_waves,
        lhs_rows.device.index or 0,
        metadata_direct=False,
        stages=stages,
        gather_rhs=gather_rhs,
        split_k_partials=True,
    )
    if sorted_token_ids is None:
        _run_compiled(
            launcher,
            lhs_rows,
            rhs_rows,
            expert_frequency,
            split_queue,
            split_queue,
            partials,
            0,
            _MAX_SIGNED_I32,
            split_queue,
            0,
            grid,
            stream,
        )
    else:
        _run_compiled(
            launcher,
            lhs_rows,
            rhs_rows,
            sorted_token_ids,
            expert_frequency,
            split_queue,
            split_queue,
            partials,
            0,
            _MAX_SIGNED_I32,
            split_queue,
            0,
            grid,
            stream,
        )
    for tensor in tensors:
        tensor.record_stream(stream)
    return partials


@functools.lru_cache(maxsize=32)
def compile_hot_split_finalize(
    output_m: int,
    output_n: int,
    num_experts: int,
    device_index: int,
):
    """Compile deterministic FP32-partial reduction into BF16 dW2."""

    del device_index
    vector_width = 4
    output_elements = output_m * output_n
    if min(output_m, output_n, num_experts) <= 0 or output_elements % vector_width:
        raise ValueError(
            "hot split finalize requires positive vector-aligned dimensions"
        )
    vectors_per_expert = output_elements // vector_width

    @flyc.kernel(known_block_size=[_BLOCK_THREADS, 1, 1])
    def finalize(
        split_queue: fx.Tensor,
        hot_queue: fx.Tensor,
        partials: fx.Tensor,
        output: fx.Tensor,
    ):
        tid = gpu.thread_idx.x
        bid = gpu.block_idx.x
        grid_size = gpu.grid_dim.x
        split_rsrc = buffer_ops.create_buffer_resource(split_queue, max_size=True)
        hot_rsrc = buffer_ops.create_buffer_resource(hot_queue, max_size=True)
        partial_rsrc = buffer_ops.create_buffer_resource(partials, max_size=True)
        output_rsrc = buffer_ops.create_buffer_resource(output, max_size=True)
        hot_count = fx.Int32(
            buffer_ops.buffer_load(hot_rsrc, fx.Int32(0), vec_width=1, dtype=T.i32)
        )
        work_bound = hot_count * fx.Int32(vectors_per_expert)

        def reduce_vector(work_index):
            hot_index = work_index // fx.Int32(vectors_per_expert)
            vector_index = work_index % fx.Int32(vectors_per_expert)
            hot_offset = fx.Int32(1) + hot_index * fx.Int32(3)
            expert = fx.Int32(
                buffer_ops.buffer_load(hot_rsrc, hot_offset, vec_width=1, dtype=T.i32)
            )
            first_partition = fx.Int32(
                buffer_ops.buffer_load(
                    hot_rsrc,
                    hot_offset + fx.Int32(1),
                    vec_width=1,
                    dtype=T.i32,
                )
            )
            partition_count = fx.Int32(
                buffer_ops.buffer_load(
                    hot_rsrc,
                    hot_offset + fx.Int32(2),
                    vec_width=1,
                    dtype=T.i32,
                )
            )
            element = vector_index * fx.Int32(vector_width)
            accum = fx.Vector.filled(vector_width, 0.0, fx.Float32)
            # Builder reserves and writes each expert's partitions in ascending
            # row order, so this loop is deterministic despite cross-expert
            # atomic queue placement.
            for local_partition in range(0, partition_count, 1):
                partition = first_partition + fx.Int32(local_partition)
                split_offset = fx.Int32(1) + partition * fx.Int32(3)
                descriptor_expert = fx.Int32(
                    buffer_ops.buffer_load(
                        split_rsrc,
                        split_offset,
                        vec_width=1,
                        dtype=T.i32,
                    )
                )
                partial_offset = partition * fx.Int32(output_elements) + element
                partial = fx.Vector(
                    buffer_ops.buffer_load(
                        partial_rsrc,
                        partial_offset,
                        vec_width=vector_width,
                        dtype=fx.Float32,
                    )
                )
                accum = accum + (descriptor_expert == expert).select(
                    partial,
                    fx.Vector.filled(vector_width, 0.0, fx.Float32),
                )
            output_offset = expert * fx.Int32(output_elements) + element
            buffer_ops.buffer_store(
                accum.to(fx.BFloat16),
                output_rsrc,
                output_offset,
            )

        work_index = bid * fx.Int32(_BLOCK_THREADS) + tid
        stride = grid_size * fx.Int32(_BLOCK_THREADS)
        for current in range(work_index, work_bound, stride):
            reduce_vector(fx.Int32(current))

    @flyc.jit
    def launch(
        split_queue: fx.Tensor,
        hot_queue: fx.Tensor,
        partials: fx.Tensor,
        output: fx.Tensor,
        i32_grid: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        finalize(split_queue, hot_queue, partials, output).launch(
            grid=(i32_grid, 1, 1),
            block=(_BLOCK_THREADS, 1, 1),
            stream=stream,
        )

    return launch


def finalize_hot_splitk_flydsl(
    split_queue: torch.Tensor,
    hot_queue: torch.Tensor,
    partials: torch.Tensor,
    output: torch.Tensor,
    *,
    stream: torch.cuda.Stream | None = None,
) -> torch.Tensor:
    """Reduce hot-expert FP32 partials in partition order and cast once."""

    if partials.ndim != 3 or output.ndim != 3:
        raise ValueError("split-K finalize expects partials[S,M,N], output[E,M,N]")
    if tuple(partials.shape[1:]) != tuple(output.shape[1:]):
        raise ValueError("split-K partial/output matrix dimensions must match")
    tensors = (split_queue, hot_queue, partials, output)
    if any(tensor.device != output.device for tensor in tensors):
        raise ValueError("split-K finalize tensors must share one device")
    if split_queue.dtype != torch.int32 or hot_queue.dtype != torch.int32:
        raise TypeError("split-K finalize queues must use int32")
    if partials.dtype != torch.float32 or output.dtype != torch.bfloat16:
        raise TypeError("split-K finalize requires FP32 partials and BF16 output")
    if not all(tensor.is_contiguous() for tensor in tensors):
        raise ValueError("split-K finalize tensors must be contiguous")
    hot_capacity = (int(hot_queue.numel()) - 1) // 3
    if hot_capacity <= 0:
        return output
    output_elements = int(output.shape[1]) * int(output.shape[2])
    vector_width = 4
    max_work = hot_capacity * (output_elements // vector_width)
    grid = min(1024, max(1, (max_work + _BLOCK_THREADS - 1) // _BLOCK_THREADS))
    if stream is None:
        stream = torch.cuda.current_stream(output.device)
    launcher = compile_hot_split_finalize(
        int(output.shape[1]),
        int(output.shape[2]),
        int(output.shape[0]),
        output.device.index or 0,
    )
    _run_compiled(
        launcher,
        split_queue,
        hot_queue,
        partials,
        output,
        grid,
        stream,
    )
    for tensor in tensors:
        tensor.record_stream(stream)
    return output


def build_active_expert_queue_flydsl(
    expert_frequency: torch.Tensor,
    sorted_expert_ids: torch.Tensor,
    num_valid_ids: torch.Tensor,
    *,
    routes: int,
    queue_storage: torch.Tensor | None = None,
    stream: torch.cuda.Stream | None = None,
) -> torch.Tensor:
    """Build the shared ``[count, (expert, first_row) * capacity]`` queue."""

    if expert_frequency.ndim != 1:
        raise ValueError("expert_frequency must have shape [E]")
    num_experts = int(expert_frequency.numel())
    capacity = active_expert_descriptor_capacity(routes, num_experts)
    required_elements = 1 + 2 * capacity
    metadata = (expert_frequency, sorted_expert_ids, num_valid_ids)
    if any(tensor.dtype != torch.int32 for tensor in metadata):
        raise TypeError("active-expert queue metadata must use int32")
    if any(tensor.device != expert_frequency.device for tensor in metadata):
        raise ValueError("active-expert queue metadata must share one device")
    if not all(tensor.is_contiguous() for tensor in metadata):
        raise ValueError("active-expert queue metadata must be contiguous")
    if sorted_expert_ids.ndim != 1:
        raise ValueError("sorted_expert_ids must be one-dimensional")
    if num_valid_ids.ndim != 1 or num_valid_ids.numel() < 1:
        raise ValueError("num_valid_ids must contain the padded row count")
    if capacity and sorted_expert_ids.numel() < 1:
        raise ValueError("non-empty routes require sorter expert metadata")
    if queue_storage is None:
        queue_storage = torch.empty(
            required_elements,
            dtype=torch.int32,
            device=expert_frequency.device,
        )
    elif (
        queue_storage.dtype != torch.int32
        or queue_storage.device != expert_frequency.device
        or not queue_storage.is_contiguous()
        or queue_storage.ndim != 1
        or queue_storage.numel() < required_elements
    ):
        raise ValueError(
            "queue_storage must be contiguous int32 on the metadata device "
            f"with at least {required_elements} elements"
        )

    if stream is None:
        stream = torch.cuda.current_stream(expert_frequency.device)
    if capacity == 0:
        with torch.cuda.stream(stream):
            queue_storage[:1].zero_()
        queue_storage.record_stream(stream)
        return queue_storage

    launcher = compile_active_expert_queue(
        num_experts,
        expert_frequency.device.index or 0,
    )
    _run_compiled(
        launcher,
        expert_frequency,
        sorted_expert_ids,
        num_valid_ids,
        queue_storage,
        capacity,
        stream,
    )
    queue_storage.record_stream(stream)
    return queue_storage


def grouped_tn_from_queue_flydsl(
    lhs_rows: torch.Tensor,
    rhs_rows: torch.Tensor,
    expert_frequency: torch.Tensor,
    queue_storage: torch.Tensor,
    output: torch.Tensor,
    *,
    sorted_token_ids: torch.Tensor | None = None,
    block_m: int | None = None,
    block_n: int | None = None,
    block_k: int | None = None,
    k_padding: int | None = None,
    m_waves: int | None = None,
    n_waves: int | None = None,
    stages: int = 2,
    min_active_experts: int = 0,
    max_active_experts: int | None = None,
    min_expert_rows: int = 0,
    max_expert_rows: int | None = None,
    active_guard_or_expert_rows: bool = False,
    hot_expert_storage: torch.Tensor | None = None,
    hot_split_min_rows: int | None = None,
    use_queue_3d: bool = False,
    stream: torch.cuda.Stream | None = None,
) -> torch.Tensor:
    """Consume a prebuilt active-expert queue for one grouped TN contraction.

    By default both inputs use the same sorter-padded row dimension.  Passing
    ``sorted_token_ids`` switches the RHS to token-major ``[T,N]`` storage and
    gathers its rows directly into LDS.  ``output`` must already be zero so
    empty experts retain exact-zero gradients.  Optional tuning arguments let
    dW1 and dW2 select independent output/K tiles.  An optional inclusive
    active-expert interval makes the launch a no-op when ``queue_storage[0]``
    falls outside it.  Inclusive expert-row bounds can independently filter
    queue entries without a host frequency readback.  OR mode admits an entry
    when either its row predicate or the active-count predicate matches.
    A hot-expert queue can independently exclude split-K-owned experts.  Its
    zero device count disables that exclusion, allowing a device-side cohort
    gate to keep the regular contraction without changing the launch graph.
    ``use_queue_3d`` maps the output tile to grid y/z and reserves grid x for
    device-queue descriptors.  It is intended for finite, preselected tuning
    profiles; the runtime queue count remains device-side.
    """

    if lhs_rows.ndim != 2 or rhs_rows.ndim != 2 or output.ndim != 3:
        raise ValueError("expected lhs_rows[P,M], rhs_rows[rows,N], and output[E,M,N]")
    num_experts, output_m, output_n = (int(value) for value in output.shape)
    gather_rhs = sorted_token_ids is not None
    if int(lhs_rows.shape[1]) != output_m:
        raise ValueError("lhs_rows width must match output M")
    if not gather_rhs and int(lhs_rows.shape[0]) != int(rhs_rows.shape[0]):
        raise ValueError("lhs_rows and rhs_rows must share P without RHS gather")
    if int(rhs_rows.shape[1]) != output_n:
        raise ValueError("rhs_rows width must match output N")
    tensors = (lhs_rows, rhs_rows, expert_frequency, queue_storage, output) + (
        (sorted_token_ids,) if sorted_token_ids is not None else ()
    )
    if any(tensor.device != lhs_rows.device for tensor in tensors):
        raise ValueError("grouped TN tensors must share one device")
    if (
        lhs_rows.dtype != torch.bfloat16
        or rhs_rows.dtype != torch.bfloat16
        or output.dtype != torch.bfloat16
    ):
        raise TypeError("grouped TN currently requires BF16 inputs and output")
    if expert_frequency.dtype != torch.int32 or queue_storage.dtype != torch.int32:
        raise TypeError("grouped TN metadata must use int32")
    if sorted_token_ids is not None and sorted_token_ids.dtype != torch.int32:
        raise TypeError("sorted_token_ids must use int32")
    if not all(tensor.is_contiguous() for tensor in tensors):
        raise ValueError("grouped TN tensors must be contiguous")
    if tuple(expert_frequency.shape) != (num_experts,):
        raise ValueError("expert_frequency must have shape [E]")
    if (
        queue_storage.ndim != 1
        or queue_storage.numel() < 1
        or (queue_storage.numel() - 1) % 2
    ):
        raise ValueError(
            "queue_storage must use [count, (expert, first_row) * capacity] ABI"
        )
    if sorted_token_ids is not None and (
        sorted_token_ids.ndim != 1 or sorted_token_ids.numel() < lhs_rows.shape[0]
    ):
        raise ValueError("sorted_token_ids must be 1D and cover every lhs sorted row")
    if (
        gather_rhs
        and sorted_token_ids.numel() * sorted_token_ids.element_size()
        > _MAX_BUFFER_BYTES
    ):
        raise ValueError(
            "sorted_token_ids exceeds the gfx950 buffer-resource byte limit"
        )
    if gather_rhs and rhs_rows.shape[0] > _TOKEN_MASK:
        raise ValueError("token-major RHS row count exceeds packed token-id capacity")
    if gather_rhs and rhs_rows.numel() * rhs_rows.element_size() > _MAX_BUFFER_BYTES:
        raise ValueError(
            "token-major RHS exceeds the gfx950 buffer-resource byte limit"
        )
    if (
        not isinstance(min_expert_rows, int)
        or isinstance(min_expert_rows, bool)
        or min_expert_rows < 0
    ):
        raise ValueError("min_expert_rows must be a non-negative int")
    if max_expert_rows is not None and (
        not isinstance(max_expert_rows, int)
        or isinstance(max_expert_rows, bool)
        or max_expert_rows < min_expert_rows
    ):
        raise ValueError("max_expert_rows must be None or at least min_expert_rows")
    filter_expert_rows = min_expert_rows != 0 or max_expert_rows is not None
    has_max_expert_rows = max_expert_rows is not None
    if active_guard_or_expert_rows and (
        max_active_experts is None or not filter_expert_rows
    ):
        raise ValueError("active_guard_or_expert_rows requires active and row guards")
    exclude_hot_experts = hot_expert_storage is not None
    if exclude_hot_experts:
        assert hot_expert_storage is not None
        if hot_expert_storage.device != lhs_rows.device:
            raise ValueError("hot_expert_storage must share the grouped TN device")
        if hot_expert_storage.dtype != torch.int32:
            raise TypeError("hot_expert_storage must use int32")
        if (
            not hot_expert_storage.is_contiguous()
            or hot_expert_storage.ndim != 1
            or hot_expert_storage.numel() < 1
            or (hot_expert_storage.numel() - 1) % 3
        ):
            raise ValueError(
                "hot_expert_storage must use "
                "[count, (expert, first_partition, partition_count) * capacity] ABI"
            )
        if (
            not isinstance(hot_split_min_rows, int)
            or isinstance(hot_split_min_rows, bool)
            or hot_split_min_rows <= 0
        ):
            raise ValueError(
                "hot_split_min_rows must be positive with hot_expert_storage"
            )
    elif hot_split_min_rows is not None:
        raise ValueError("hot_split_min_rows requires hot_expert_storage")
    if not isinstance(use_queue_3d, bool):
        raise TypeError("use_queue_3d must be a bool")
    if use_queue_3d and (
        num_experts != 16
        or gather_rhs
        or min_active_experts != 0
        or max_active_experts is not None
        or filter_expert_rows
        or exclude_hot_experts
    ):
        raise ValueError(
            "queue 3D requires an unguarded E16 contraction with contiguous RHS"
        )

    capacity = (int(queue_storage.numel()) - 1) // 2
    if capacity == 0:
        return output

    (
        default_bm,
        default_bn,
        default_bk,
        default_k_padding,
        default_m_waves,
        default_n_waves,
    ) = grouped_tn_tuning(
        output_m,
        output_n,
    )
    block_m = default_bm if block_m is None else block_m
    block_n = default_bn if block_n is None else block_n
    block_k = default_bk if block_k is None else block_k
    k_padding = default_k_padding if k_padding is None else k_padding
    m_waves = default_m_waves if m_waves is None else m_waves
    n_waves = default_n_waves if n_waves is None else n_waves
    queue_3d_grid = (
        grouped_tn_queue_3d_launch_grid(
            capacity,
            output_m,
            output_n,
            block_m,
            block_n,
            block_k,
            stages,
            m_waves,
            n_waves,
            gather_rhs,
        )
        if use_queue_3d
        else None
    )
    queue_3d = queue_3d_grid is not None
    grid = (
        queue_3d_grid[0]
        if queue_3d_grid is not None
        else grouped_tn_launch_grid(
            capacity,
            output_m,
            output_n,
            block_m,
            block_n,
            block_k,
            stages,
            m_waves,
            n_waves,
            gather_rhs,
        )
    )
    if stream is None:
        stream = torch.cuda.current_stream(lhs_rows.device)
    launcher = compile_grouped_tn(
        output_m,
        output_n,
        num_experts,
        block_m,
        block_n,
        block_k,
        k_padding,
        m_waves,
        n_waves,
        lhs_rows.device.index or 0,
        False,
        stages,
        min_active_experts,
        max_active_experts,
        gather_rhs=gather_rhs,
        filter_expert_rows=filter_expert_rows,
        has_max_expert_rows=has_max_expert_rows,
        active_guard_or_expert_rows=active_guard_or_expert_rows,
        exclude_hot_experts=exclude_hot_experts,
        queue_3d=queue_3d,
    )
    runtime_max_expert_rows = (
        _MAX_SIGNED_I32 if max_expert_rows is None else max_expert_rows
    )
    hot_expert_arg = queue_storage if hot_expert_storage is None else hot_expert_storage
    runtime_hot_split_min_rows = 0 if hot_split_min_rows is None else hot_split_min_rows
    if gather_rhs:
        _run_compiled(
            launcher,
            lhs_rows,
            rhs_rows,
            sorted_token_ids,
            expert_frequency,
            queue_storage,
            queue_storage,
            output,
            min_expert_rows,
            runtime_max_expert_rows,
            hot_expert_arg,
            runtime_hot_split_min_rows,
            grid,
            stream,
        )
    else:
        _run_compiled(
            launcher,
            lhs_rows,
            rhs_rows,
            expert_frequency,
            queue_storage,
            queue_storage,
            output,
            min_expert_rows,
            runtime_max_expert_rows,
            hot_expert_arg,
            runtime_hot_split_min_rows,
            grid,
            stream,
        )
    queue_storage.record_stream(stream)
    if sorted_token_ids is not None:
        sorted_token_ids.record_stream(stream)
    if hot_expert_storage is not None:
        hot_expert_storage.record_stream(stream)
    return output


def grouped_tn_from_metadata_flydsl(
    lhs_rows: torch.Tensor,
    rhs_rows: torch.Tensor,
    expert_frequency: torch.Tensor,
    sorted_expert_ids: torch.Tensor,
    num_valid_ids: torch.Tensor,
    output: torch.Tensor,
    *,
    sorted_token_ids: torch.Tensor | None = None,
    block_m: int | None = None,
    block_n: int | None = None,
    block_k: int | None = None,
    k_padding: int | None = None,
    m_waves: int | None = None,
    n_waves: int | None = None,
    stages: int = 2,
    stream: torch.cuda.Stream | None = None,
) -> torch.Tensor:
    """Run grouped TN directly from one-block-per-expert sorter metadata.

    This is the builder-free fixed-K T1 path.  Callers must guarantee every
    non-empty expert has at most ``_SORTED_BLOCK_M`` rows; otherwise repeated
    expert IDs would race while writing the same output tile.  Passing
    ``sorted_token_ids`` gathers a token-major RHS directly into LDS.
    """

    if lhs_rows.ndim != 2 or rhs_rows.ndim != 2 or output.ndim != 3:
        raise ValueError("expected lhs_rows[P,M], rhs_rows[rows,N], and output[E,M,N]")
    num_experts, output_m, output_n = (int(value) for value in output.shape)
    gather_rhs = sorted_token_ids is not None
    if int(lhs_rows.shape[1]) != output_m:
        raise ValueError("lhs_rows width must match output M")
    if not gather_rhs and int(lhs_rows.shape[0]) != int(rhs_rows.shape[0]):
        raise ValueError("lhs_rows and rhs_rows must share P without RHS gather")
    if int(rhs_rows.shape[1]) != output_n:
        raise ValueError("rhs_rows width must match output N")
    tensors = (
        lhs_rows,
        rhs_rows,
        expert_frequency,
        sorted_expert_ids,
        num_valid_ids,
        output,
    ) + ((sorted_token_ids,) if sorted_token_ids is not None else ())
    if any(tensor.device != lhs_rows.device for tensor in tensors):
        raise ValueError("grouped TN tensors must share one device")
    if (
        lhs_rows.dtype != torch.bfloat16
        or rhs_rows.dtype != torch.bfloat16
        or output.dtype != torch.bfloat16
    ):
        raise TypeError("grouped TN currently requires BF16 inputs and output")
    if any(
        tensor.dtype != torch.int32
        for tensor in (expert_frequency, sorted_expert_ids, num_valid_ids)
    ):
        raise TypeError("grouped TN metadata must use int32")
    if sorted_token_ids is not None and sorted_token_ids.dtype != torch.int32:
        raise TypeError("sorted_token_ids must use int32")
    if not all(tensor.is_contiguous() for tensor in tensors):
        raise ValueError("grouped TN tensors must be contiguous")
    if tuple(expert_frequency.shape) != (num_experts,):
        raise ValueError("expert_frequency must have shape [E]")
    if sorted_expert_ids.ndim != 1:
        raise ValueError("sorted_expert_ids must be one-dimensional")
    if num_valid_ids.ndim != 1 or num_valid_ids.numel() < 1:
        raise ValueError("num_valid_ids must contain the padded row count")
    if sorted_token_ids is not None and (
        sorted_token_ids.ndim != 1 or sorted_token_ids.numel() < lhs_rows.shape[0]
    ):
        raise ValueError("sorted_token_ids must be 1D and cover every lhs sorted row")
    if (
        gather_rhs
        and sorted_token_ids.numel() * sorted_token_ids.element_size()
        > _MAX_BUFFER_BYTES
    ):
        raise ValueError(
            "sorted_token_ids exceeds the gfx950 buffer-resource byte limit"
        )
    if gather_rhs and rhs_rows.shape[0] > _TOKEN_MASK:
        raise ValueError("token-major RHS row count exceeds packed token-id capacity")
    if gather_rhs and rhs_rows.numel() * rhs_rows.element_size() > _MAX_BUFFER_BYTES:
        raise ValueError(
            "token-major RHS exceeds the gfx950 buffer-resource byte limit"
        )
    if sorted_expert_ids.numel() == 0:
        return output

    (
        default_bm,
        default_bn,
        default_bk,
        default_k_padding,
        default_m_waves,
        default_n_waves,
    ) = grouped_tn_tuning(output_m, output_n)
    block_m = default_bm if block_m is None else block_m
    block_n = default_bn if block_n is None else block_n
    block_k = default_bk if block_k is None else block_k
    k_padding = default_k_padding if k_padding is None else k_padding
    m_waves = default_m_waves if m_waves is None else m_waves
    n_waves = default_n_waves if n_waves is None else n_waves
    grid = grouped_tn_launch_grid(
        int(sorted_expert_ids.numel()),
        output_m,
        output_n,
        block_m,
        block_n,
        block_k,
        stages,
        m_waves,
        n_waves,
        gather_rhs,
    )
    if stream is None:
        stream = torch.cuda.current_stream(lhs_rows.device)
    launcher = compile_grouped_tn(
        output_m,
        output_n,
        num_experts,
        block_m,
        block_n,
        block_k,
        k_padding,
        m_waves,
        n_waves,
        lhs_rows.device.index or 0,
        True,
        stages,
        gather_rhs=gather_rhs,
    )
    if gather_rhs:
        _run_compiled(
            launcher,
            lhs_rows,
            rhs_rows,
            sorted_token_ids,
            expert_frequency,
            sorted_expert_ids,
            num_valid_ids,
            output,
            0,
            _MAX_SIGNED_I32,
            num_valid_ids,
            0,
            grid,
            stream,
        )
    else:
        _run_compiled(
            launcher,
            lhs_rows,
            rhs_rows,
            expert_frequency,
            sorted_expert_ids,
            num_valid_ids,
            output,
            0,
            _MAX_SIGNED_I32,
            num_valid_ids,
            0,
            grid,
            stream,
        )
    sorted_expert_ids.record_stream(stream)
    num_valid_ids.record_stream(stream)
    if sorted_token_ids is not None:
        sorted_token_ids.record_stream(stream)
    return output


def grouped_tn_flydsl(
    lhs_rows: torch.Tensor,
    rhs_rows: torch.Tensor,
    expert_frequency: torch.Tensor,
    sorted_expert_ids: torch.Tensor,
    num_valid_ids: torch.Tensor,
    output: torch.Tensor,
    *,
    routes: int,
    descriptor_storage: torch.Tensor | None = None,
    block_m: int | None = None,
    block_n: int | None = None,
    block_k: int | None = None,
    k_padding: int | None = None,
    m_waves: int | None = None,
    n_waves: int | None = None,
    stages: int = 2,
    stream: torch.cuda.Stream | None = None,
) -> torch.Tensor:
    """Build a queue and compute one standalone grouped TN contraction."""

    if routes == 0:
        # Preserve the output identity and exact zeros without compiling or
        # launching metadata/GEMM kernels for an empty ragged batch.
        active_expert_descriptor_capacity(routes, int(output.shape[0]))
        return output
    queue_storage = build_active_expert_queue_flydsl(
        expert_frequency,
        sorted_expert_ids,
        num_valid_ids,
        routes=routes,
        queue_storage=descriptor_storage,
        stream=stream,
    )
    return grouped_tn_from_queue_flydsl(
        lhs_rows,
        rhs_rows,
        expert_frequency,
        queue_storage,
        output,
        block_m=block_m,
        block_n=block_n,
        block_k=block_k,
        k_padding=k_padding,
        m_waves=m_waves,
        n_waves=n_waves,
        stages=stages,
        stream=stream,
    )


def grouped_dw2_flydsl(
    dy: torch.Tensor,
    activation: torch.Tensor,
    expert_frequency: torch.Tensor,
    sorted_expert_ids: torch.Tensor,
    num_valid_ids: torch.Tensor,
    dw2: torch.Tensor,
    **kwargs,
) -> torch.Tensor:
    """Semantic dW2 wrapper around :func:`grouped_tn_flydsl`."""

    return grouped_tn_flydsl(
        dy,
        activation,
        expert_frequency,
        sorted_expert_ids,
        num_valid_ids,
        dw2,
        **kwargs,
    )


__all__ = [
    "active_expert_descriptor_capacity",
    "active_expert_queue_elements",
    "build_active_expert_queue_flydsl",
    "build_hot_split_queues_flydsl",
    "compile_active_expert_queue",
    "compile_grouped_tn",
    "compile_hot_split_finalize",
    "compile_hot_split_queues",
    "compile_inactive_weight_grad_zero",
    "finalize_hot_splitk_flydsl",
    "grouped_dw2_flydsl",
    "grouped_dw2_tuning",
    "grouped_tn_flydsl",
    "grouped_tn_from_metadata_flydsl",
    "grouped_tn_from_queue_flydsl",
    "grouped_tn_grid_cap",
    "grouped_tn_launch_grid",
    "grouped_tn_splitk_from_queue_flydsl",
    "grouped_tn_tuning",
    "hot_split_descriptor_capacity",
    "zero_inactive_weight_grads_flydsl",
    "zero_weight_grads_adaptive_flydsl",
]
