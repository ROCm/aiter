# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Persistent CTA-specialized gfx1250 EP4 stage-1 mega-kernel skeleton.

This module owns the part that is independent of a particular GEMM lowering:

* a bounded (at most one CTA per CU) producer cohort;
* generation-tagged payload readiness and ready-tile queue publication; and
* a persistent consumer work queue which producer CTAs join after finite work.

The current gfx1250 A8W4/A4W4 grouped GEMM is a launch-level ``@flyc.jit``
function and cannot be called from another kernel.  Consequently this module
requires an explicit :class:`Stage1MegaEmitter` implementation at compile time.
It deliberately has no no-op fallback: compiling without an inline GEMM
emitter raises :class:`Stage1MegaKernelUnsupported`.

Planner/allocator ABI (all entries are i32):

``tile_expected[t]``
    Number of producer arrivals needed by tile ``t``.
``tile_ready[t]``
    System-scope arrival counter, incremented with
    :func:`publish_tile_arrival`.
``ready_epoch[t]``
    The final producer release-publishes the plan generation here. Consumers
    dynamically claim GEMM ``(M, N)`` work ids, map each back to its dense M
    tile, and wait on this plane before reading A.

Exact generation equality is intentional. Reusing a generation is unsupported
because it could accept stale readiness. The compact planner increments and
release-publishes the generation on device, so CUDA Graph replay never embeds a
host generation constant.
"""

import functools
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr.typing import Int32, Int64, T
from flydsl.runtime.device import get_rocm_arch

from aiter.ops.flydsl.kernels import communication_ops_utils as comm_ops
from aiter.ops.flydsl.kernels.tensor_shim import _run_compiled

from .compact_plan import (
    COMPACT_TILE_GEN_DW,
    COMPACT_TILE_NTILES_DW,
    COMPACT_TILE_WORK_DW,
    compact_tile_layout,
)
from .config import _WAVE_SIZE as WAVE

EMITTER_ABI_VERSION = 1
SUPPORTED_HIDDEN_DIM = 7168
SUPPORTED_TOPK = (6, 8)
SUPPORTED_WIRES = ("fp8", "fp4")


class Stage1MegaKernelUnsupported(RuntimeError):
    """The requested geometry has no honest inline implementation."""


@dataclass(frozen=True, slots=True)
class Stage1MegaKernelConfig:
    """Compile-time geometry for the EP4 gfx1250 kernel."""

    hidden_dim: int = SUPPORTED_HIDDEN_DIM
    topk: int = 8
    dispatch_wire: str = "fp8"
    world_size: int = 4
    rank: int = 0
    experts_per_rank: int = 32
    tile_m: int = 32
    tile_n: int = 256
    tile_k: int = 256
    gemm_n: int = 6144
    num_cu: int = 256
    producer_ctas: int = 16
    waves_per_cta: int = 4
    compact_cap: int = 1
    tile_count: int = 1
    waves_per_eu_hint: int = 1

    def __post_init__(self) -> None:
        _validate_config(self)


@runtime_checkable
class Stage1MegaEmitter(Protocol):
    """Compile-time callback ABI for dispatch and GEMM lowering.

    Implementations are Python objects captured while FlyDSL traces the kernel.
    Both methods emit device IR and return ``None``.  They must not launch a
    child kernel.  Dynamic control flow inside a method must itself be FlyDSL
    traceable (for example by using ``@flyc.jit``/``comm_ops.traced`` helpers).

    ``emit_producer`` must perform a finite share of compact payload movement
    and call :func:`publish_tile_arrival` once per completed contribution.
    Every workgroup thread calls both methods; emitters may specialize by
    ``tid``/``wave_id`` but must leave the workgroup converged on return.

    ``emit_consumer`` receives one ready GEMM1 ``(M, N)`` work id and may use
    ``user0..user15`` as
    opaque addresses.  The exact tensor layout is owned by the emitter.
    """

    ABI_VERSION: int
    GEMM_CORE_VERSION: int
    LDS_BYTES: int
    WAVES_PER_CTA: int

    def emit_producer(self, **context) -> None:
        ...

    def emit_consumer(self, **context) -> None:
        ...


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _validate_config(config: Stage1MegaKernelConfig) -> None:
    _require(
        int(config.hidden_dim) == SUPPORTED_HIDDEN_DIM,
        f"stage1 mega-kernel supports hidden_dim={SUPPORTED_HIDDEN_DIM} only",
    )
    _require(int(config.topk) in SUPPORTED_TOPK, f"topk must be one of {SUPPORTED_TOPK}")
    _require(
        config.dispatch_wire in SUPPORTED_WIRES,
        f"dispatch_wire must be one of {SUPPORTED_WIRES}",
    )
    _require(int(config.world_size) == 4, "stage1 mega-kernel supports EP4 only")
    _require(0 <= int(config.rank) < 4, "rank must be in [0, 4)")
    _require(int(config.experts_per_rank) > 0, "experts_per_rank must be positive")
    _require(int(config.tile_m) in (16, 32, 64), "tile_m must be 16, 32 or 64")
    _require(int(config.tile_n) > 0 and config.tile_n % 128 == 0, "tile_n must be 128-aligned")
    _require(int(config.gemm_n) > 0, "gemm_n must be positive")
    _require(
        int(config.tile_k) in (128, 256, 512)
        and int(config.hidden_dim) % int(config.tile_k) == 0,
        "tile_k must be 128, 256 or 512 and divide hidden_dim",
    )
    _require(int(config.num_cu) > 0, "num_cu must be positive")
    _require(
        0 < int(config.producer_ctas) < int(config.num_cu),
        "producer_ctas must be in [1, num_cu)",
    )
    _require(
        int(config.waves_per_cta) in (4, 8),
        "waves_per_cta must be 4 or 8",
    )
    _require(
        1 <= int(config.waves_per_eu_hint) <= 4,
        "waves_per_eu_hint must be in [1, 4]",
    )
    _require(int(config.tile_count) > 0, "tile_count must be positive")
    _require(
        int(config.compact_cap) > 0,
        "compact_cap must be positive",
    )
    layout = compact_tile_layout(compact_cap=int(config.compact_cap))
    _require(
        int(config.tile_count) <= layout.tiles_for(int(config.tile_m)),
        "tile_count exceeds compact tile-state capacity",
    )


def _validate_emitter(emitter: Stage1MegaEmitter | None) -> Stage1MegaEmitter:
    if emitter is None:
        raise Stage1MegaKernelUnsupported(
            "gfx1250 grouped quant GEMM is currently a launch-level @flyc.jit "
            "core; compile_stage1_mega_kernel requires an inline "
            "Stage1MegaEmitter instead of pretending that GEMM work is fused"
        )
    if getattr(emitter, "ABI_VERSION", None) != EMITTER_ABI_VERSION:
        raise TypeError(
            "Stage1MegaEmitter ABI mismatch: expected "
            f"{EMITTER_ABI_VERSION}, got {getattr(emitter, 'ABI_VERSION', None)!r}"
        )
    if not isinstance(getattr(emitter, "LDS_BYTES", None), int) or emitter.LDS_BYTES <= 0:
        raise TypeError("emitter.LDS_BYTES must be a positive integer")
    if (
        not isinstance(getattr(emitter, "GEMM_CORE_VERSION", None), int)
        or emitter.GEMM_CORE_VERSION <= 0
    ):
        raise TypeError("emitter.GEMM_CORE_VERSION must be a positive integer")
    if emitter.LDS_BYTES > 320 * 1024 - 128:
        raise ValueError("emitter LDS arena exceeds gfx1250 workgroup capacity")
    if not callable(getattr(emitter, "emit_producer", None)):
        raise TypeError("emitter.emit_producer must be callable")
    if not callable(getattr(emitter, "emit_consumer", None)):
        raise TypeError("emitter.emit_consumer must be callable")
    return emitter


@comm_ops.traced
def publish_tile_arrival(
    *,
    tile_id,
    generation,
    tile_count,
    addr_tile_generation,
    addr_tile_expected,
    addr_tile_ready,
    addr_ready_epoch,
):
    """Publish one payload contribution and release the last-arriving tile.

    This helper is the only supported producer-to-consumer publication path.
    Waiting on the slot generation prevents a producer from incrementing a
    stale counter before the compact planner has reset it. The final RMW is
    unique and observes the other producers' release sequence; it publishes
    the generation directly at the dense tile id.
    """

    valid = (tile_id >= fx.Int32(0)) & (tile_id < fx.Int32(tile_count))
    if valid:
        comm_ops.wait_i32_until_equals(addr_tile_generation, generation)
        comm_ops.fence_system_acquire()
        expected = fx.Int32(
            comm_ops.load_i32_global_system(
                addr_tile_expected + fx.Int64(tile_id) * fx.Int64(4)
            )
        )
        if expected > fx.Int32(0):
            old = fx.Int32(
                comm_ops.atomic_add_system(
                    addr_tile_ready + fx.Int64(tile_id) * fx.Int64(4), fx.Int32(1)
                )
            )
            if old + fx.Int32(1) == expected:
                comm_ops.fence_system_acquire()
                comm_ops.fence_system_release()
                comm_ops.store_i32_system(addr_ready_epoch, tile_id, generation)


@functools.cache
def compile_stage1_mega_kernel(
    *,
    config: Stage1MegaKernelConfig,
    emitter: Stage1MegaEmitter | None,
):
    """Compile and return the host launcher for one emitter/config pair."""

    if not isinstance(config, Stage1MegaKernelConfig):
        raise TypeError("config must be a Stage1MegaKernelConfig")
    emitter = _validate_emitter(emitter)
    if getattr(emitter, "WAVES_PER_CTA", None) != config.waves_per_cta:
        raise ValueError(
            "emitter/config wave-count mismatch: "
            f"{getattr(emitter, 'WAVES_PER_CTA', None)!r} != {config.waves_per_cta}"
        )
    arch = str(get_rocm_arch() or "")
    if not arch.startswith("gfx1250"):
        raise Stage1MegaKernelUnsupported(
            f"stage1 mega-kernel requires gfx1250, got {arch or 'unknown'}"
        )

    block_threads = int(config.waves_per_cta) * WAVE
    grid_x = int(config.num_cu)
    producer_ctas = int(config.producer_ctas)
    tile_count = int(config.tile_count)
    tile_layout = compact_tile_layout(compact_cap=int(config.compact_cap))
    expected_byte_off = tile_layout.expected_dw * 4
    ready_byte_off = tile_layout.ready_dw * 4
    ready_epoch_byte_off = tile_layout.queue_dw * 4

    @fx.struct
    class SharedStorage:
        storage: fx.Array[fx.Int8, emitter.LDS_BYTES + 128, 128]

    kernel_name = (
        f"ep4_stage1_mega_{config.dispatch_wire}_h{config.hidden_dim}"
        f"_k{config.topk}_tm{config.tile_m}_n{config.gemm_n}_tn{config.tile_n}"
        f"_tk{config.tile_k}_p{producer_ctas}_w{config.waves_per_cta}_r{config.rank}"
        f"_abi{EMITTER_ABI_VERSION}_lds{emitter.LDS_BYTES}"
        f"_gc{emitter.GEMM_CORE_VERSION}"
    )

    @flyc.kernel(name=kernel_name, known_block_size=[block_threads, 1, 1])
    def kernel(
        addr_plan_generation: Int64,
        addr_tile_state: Int64,
        user0: Int64,
        user1: Int64,
        user2: Int64,
        user3: Int64,
        user4: Int64,
        user5: Int64,
        user6: Int64,
        user7: Int64,
        user8: Int64,
        user9: Int64,
        user10: Int64,
        user11: Int64,
        user12: Int64,
        user13: Int64,
        user14: Int64,
        user15: Int64,
    ):
        tid = fx.thread_idx.x
        cta = fx.block_idx.x
        wave_id = tid // fx.Int32(WAVE)
        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        claimed_ptr = lds.storage.ptr
        claimed_addr = fx.Int64(fx.ptrtoint(claimed_ptr))
        lds_base_ptr = lds.storage.ptr + 128
        generation = fx.Int32(
            comm_ops.load_i32_global_system(addr_plan_generation)
        )
        addr_tile_generation = addr_tile_state + fx.Int64(COMPACT_TILE_GEN_DW * 4)
        addr_tile_expected = addr_tile_state + fx.Int64(expected_byte_off)
        addr_tile_ready = addr_tile_state + fx.Int64(ready_byte_off)
        addr_ready_epoch = addr_tile_state + fx.Int64(ready_epoch_byte_off)
        addr_work_head = addr_tile_state + fx.Int64(COMPACT_TILE_WORK_DW * 4)

        if tid == fx.Int32(0):
            comm_ops.wait_i32_until_equals(addr_tile_generation, generation)
            comm_ops.fence_system_acquire()
        fx.barrier()
        num_work_tiles = fx.Int32(
            comm_ops.load_i32_global_system(
                addr_tile_state + fx.Int64(COMPACT_TILE_NTILES_DW * 4)
            )
        )
        gemm_n_tiles = (int(config.gemm_n) + int(config.tile_n) - 1) // int(
            config.tile_n
        )
        total_work_items = num_work_tiles * fx.Int32(gemm_n_tiles)

        user_args = (
            user0,
            user1,
            user2,
            user3,
            user4,
            user5,
            user6,
            user7,
            user8,
            user9,
            user10,
            user11,
            user12,
            user13,
            user14,
            user15,
        )
        common = {
            "config": config,
            "tid": tid,
            "wave_id": wave_id,
            "cta_id": cta,
            "generation": generation,
            "num_work_tiles": num_work_tiles,
            "addr_tile_generation": addr_tile_generation,
            "addr_tile_expected": addr_tile_expected,
            "addr_tile_ready": addr_tile_ready,
            "addr_ready_epoch": addr_ready_epoch,
            "lds_base_ptr": lds_base_ptr,
            "user_args": user_args,
        }

        if cta < fx.Int32(producer_ctas):
            emitter.emit_producer(
                **common,
                producer_id=cta,
                producer_ctas=fx.Int32(producer_ctas),
                publish_tile_arrival=publish_tile_arrival,
            )
        # Contract: producer callbacks are finite and converged.  Their CTAs now
        # join the same queue as initially consumer-only CTAs.
        fx.barrier()

        active = fx.Int32(1)
        while active != fx.Int32(0):
            if tid == fx.Int32(0):
                slot = fx.Int32(comm_ops.atomic_add_agent(addr_work_head, fx.Int32(1)))
                comm_ops.store_i32_lds(claimed_addr, slot)
            fx.barrier()
            slot = fx.Int32(comm_ops.load_i32_lds(claimed_addr))
            active = (slot < total_work_items).select(fx.Int32(1), fx.Int32(0))
            if active != fx.Int32(0):
                tiles_per_group = fx.Int32(gemm_n_tiles * 16)
                group = slot // tiles_per_group
                group_first = group * fx.Int32(16)
                in_group = slot - group * tiles_per_group
                remaining = num_work_tiles - group_first
                group_tiles = (remaining < fx.Int32(16)).select(
                    remaining, fx.Int32(16)
                )
                tile_id = group_first + (
                    in_group - (in_group // group_tiles) * group_tiles
                )
                if tid == fx.Int32(0):
                    comm_ops.wait_i32_until_equals(
                        addr_ready_epoch + fx.Int64(tile_id) * fx.Int64(4),
                        generation,
                    )
                    comm_ops.fence_system_acquire()
                    comm_ops.store_i32_lds(claimed_addr, tile_id)
                fx.barrier()
                tile_id = fx.Int32(comm_ops.load_i32_lds(claimed_addr))
                emitter.emit_consumer(
                    **common, queue_slot=slot, work_id=slot, tile_id=tile_id
                )
                fx.barrier()

    @flyc.jit
    def launch(
        addr_plan_generation: Int64,
        addr_tile_state: Int64,
        user0: Int64,
        user1: Int64,
        user2: Int64,
        user3: Int64,
        user4: Int64,
        user5: Int64,
        user6: Int64,
        user7: Int64,
        user8: Int64,
        user9: Int64,
        user10: Int64,
        user11: Int64,
        user12: Int64,
        user13: Int64,
        user14: Int64,
        user15: Int64,
        stream: fx.Stream,
    ):
        kernel(
            addr_plan_generation,
            addr_tile_state,
            user0,
            user1,
            user2,
            user3,
            user4,
            user5,
            user6,
            user7,
            user8,
            user9,
            user10,
            user11,
            user12,
            user13,
            user14,
            user15,
            value_attrs={
                "rocdl.waves_per_eu": int(config.waves_per_eu_hint),
                "rocdl.flat_work_group_size": f"{block_threads},{block_threads}",
            },
        ).launch(
            grid=(grid_x, 1, 1),
            block=(block_threads, 1, 1),
            stream=stream,
        )

    return launch


def run_stage1_mega_kernel(
    *,
    config: Stage1MegaKernelConfig,
    emitter: Stage1MegaEmitter | None,
    addr_plan_generation: int,
    addr_tile_state: int,
    stream,
    user_args: tuple[int, ...] = (),
) -> None:
    """Validate, compile and launch the kernel from ordinary host Python."""

    addresses = (addr_plan_generation, addr_tile_state)
    if any(not isinstance(address, int) or address <= 0 for address in addresses):
        raise ValueError("all protocol addresses must be positive host integers")
    if len(user_args) > 16:
        raise ValueError("Stage1MegaEmitter supports at most sixteen opaque user addresses")
    if any(not isinstance(address, int) or address < 0 for address in user_args):
        raise ValueError("user_args must contain non-negative host integer addresses")
    padded_user_args = tuple(user_args) + (0,) * (16 - len(user_args))

    launch = compile_stage1_mega_kernel(config=config, emitter=emitter)
    _run_compiled(
        launch,
        *addresses,
        *padded_user_args,
        stream,
    )


# Explicit launch spelling for callers which pair it with
# ``compile_stage1_mega_kernel``.  ``run_...`` remains descriptive of the
# compile-on-first-use convenience path.
launch_stage1_mega_kernel = run_stage1_mega_kernel


__all__ = [
    "EMITTER_ABI_VERSION",
    "Stage1MegaEmitter",
    "Stage1MegaKernelConfig",
    "Stage1MegaKernelUnsupported",
    "compile_stage1_mega_kernel",
    "launch_stage1_mega_kernel",
    "publish_tile_arrival",
    "run_stage1_mega_kernel",
]
