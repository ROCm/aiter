# SPDX-License-Identifier: MIT
"""Host-side construction for the AITER-private CCO transport.

CCO windows are collective resources and must exist before the device
communicator snapshot is created.  This helper makes that ordering explicit and
turns off MORI resource pools that MegaMoETile does not use: ready and credit
words live in the registered arena itself.
"""

from __future__ import annotations

import ctypes
from dataclasses import dataclass

from mori.cco import (
    CCODevCommRequirements,
    GDA_CONNECTION_CROSSNODE,
    GDA_CONNECTION_RAIL,
)

from .ops import TEAM_RAIL, TEAM_WORLD


_HIP_MEMCPY_HOST_TO_DEVICE = 1
_HIP_MEMCPY_DEVICE_TO_HOST = 2


def clear_hip_last_error() -> int:
    """Consume a benign sticky HIP setup error before the first Torch launch.

    CCO window/P2P setup may probe unsupported access modes or re-enable peer
    access.  The operation can succeed while HIP's thread-local last-error slot
    remains set; a later unrelated Torch kernel would otherwise report it.
    """

    from mori.jit.hip_driver import _get_hip_lib

    return int(_get_hip_lib().hipGetLastError())


def _hip_copy(dst, src, nbytes: int, kind: int) -> None:
    """Copy to/from a CCO VMM pointer with the HIP runtime.

    CCO fabric allocations are intentionally not exposed as ordinary Torch
    storage: generic Torch fill/zero kernels may reject their VMM mapping.
    """

    from mori.jit.hip_driver import _check, _get_hip_lib

    _check(
        _get_hip_lib().hipMemcpy(
            dst, src, ctypes.c_size_t(nbytes), ctypes.c_int(kind)
        ),
        "hipMemcpy",
    )


def zero_window(local_ptr: int, nbytes: int) -> None:
    if nbytes < 0:
        raise ValueError("nbytes must be non-negative")
    payload = (ctypes.c_uint8 * nbytes)()
    _hip_copy(
        ctypes.c_void_p(local_ptr),
        payload,
        nbytes,
        _HIP_MEMCPY_HOST_TO_DEVICE,
    )


def fill_window_bytes(local_ptr: int, nbytes: int, value: int) -> None:
    """Fill a CCO VMM range with one byte pattern via the HIP runtime."""

    if nbytes < 0:
        raise ValueError("nbytes must be non-negative")
    if not 0 <= int(value) <= 255:
        raise ValueError("fill byte must be in [0,255]")
    payload = (ctypes.c_uint8 * nbytes)()
    ctypes.memset(payload, int(value), nbytes)
    _hip_copy(
        ctypes.c_void_p(local_ptr),
        payload,
        nbytes,
        _HIP_MEMCPY_HOST_TO_DEVICE,
    )


def write_window_u64(local_ptr: int, values) -> None:
    values = tuple(int(value) for value in values)
    payload = (ctypes.c_uint64 * len(values))(*values)
    _hip_copy(
        ctypes.c_void_p(local_ptr),
        payload,
        ctypes.sizeof(payload),
        _HIP_MEMCPY_HOST_TO_DEVICE,
    )


def write_window_u32(local_ptr: int, values) -> None:
    values = tuple(int(value) for value in values)
    payload = (ctypes.c_uint32 * len(values))(*values)
    _hip_copy(
        ctypes.c_void_p(local_ptr),
        payload,
        ctypes.sizeof(payload),
        _HIP_MEMCPY_HOST_TO_DEVICE,
    )


def read_window_u64(local_ptr: int, count: int) -> tuple[int, ...]:
    if count < 0:
        raise ValueError("count must be non-negative")
    payload = (ctypes.c_uint64 * count)()
    _hip_copy(
        payload,
        ctypes.c_void_p(local_ptr),
        ctypes.sizeof(payload),
        _HIP_MEMCPY_DEVICE_TO_HOST,
    )
    return tuple(payload)


def read_window_u32(local_ptr: int, count: int) -> tuple[int, ...]:
    if count < 0:
        raise ValueError("count must be non-negative")
    payload = (ctypes.c_uint32 * count)()
    _hip_copy(
        payload,
        ctypes.c_void_p(local_ptr),
        ctypes.sizeof(payload),
        _HIP_MEMCPY_DEVICE_TO_HOST,
    )
    return tuple(payload)


def read_window_bytes(local_ptr: int, nbytes: int) -> bytes:
    """Copy an arbitrary CCO-window byte range to immutable host bytes."""

    if nbytes < 0:
        raise ValueError("nbytes must be non-negative")
    payload = (ctypes.c_uint8 * nbytes)()
    _hip_copy(
        payload,
        ctypes.c_void_p(local_ptr),
        ctypes.sizeof(payload),
        _HIP_MEMCPY_DEVICE_TO_HOST,
    )
    return bytes(payload)


@dataclass(frozen=True)
class TransportResources:
    """CCO resources owned by the parent ``Communicator``."""

    memory: object
    window: object
    dev_comm: object
    team: str
    num_qp: int


def create_transport_resources(
    comm,
    arena_bytes: int,
    *,
    num_qp: int,
    team: str = TEAM_WORLD,
) -> TransportResources:
    """Collectively allocate/register an arena, then create its GDA DevComm.

    ``TEAM_WORLD`` addresses peers by world rank and uses CROSSNODE
    connectivity. ``TEAM_RAIL`` addresses peers by node index through
    ``CCO_TEAM_GDA`` and requires RAIL connectivity.
    """

    if arena_bytes <= 0:
        raise ValueError("arena_bytes must be positive")
    if num_qp <= 0:
        raise ValueError("num_qp must be positive")
    if team not in (TEAM_WORLD, TEAM_RAIL):
        raise ValueError(f"team must be {TEAM_WORLD!r} or {TEAM_RAIL!r}")

    # This order is part of the adapter contract. Device kernels receive
    # window.handle; memory.ptr/window.local_ptr are data addresses, not handles.
    memory = comm.alloc_mem(arena_bytes)
    window = comm.register_window(memory.ptr, memory.size)

    reqs = CCODevCommRequirements()
    reqs.gda_connection_type = (
        GDA_CONNECTION_RAIL if team == TEAM_RAIL else GDA_CONNECTION_CROSSNODE
    )
    reqs.gda_context_count = num_qp
    reqs.gda_signal_count = 0
    reqs.gda_counter_count = 0
    reqs.lsa_barrier_count = 0
    reqs.rail_gda_barrier_count = 0
    reqs.barrier_count = 0
    dev_comm = comm.create_dev_comm(reqs)

    return TransportResources(
        memory=memory,
        window=window,
        dev_comm=dev_comm,
        team=team,
        num_qp=num_qp,
    )
