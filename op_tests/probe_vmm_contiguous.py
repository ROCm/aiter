# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""Probe: can we stitch R peers' expert weights into ONE row-contiguous VA?

MoonEP's group GEMM addresses experts purely by row index over a contiguous
``[E + B, ...]`` range.  Our pool today is ``[epn + B, ...]`` (per-rank only),
so remote experts are not addressable at all: every remote expert must be
prefetched into a slot, B is a correctness bound rather than a cache size, and
the experts step splits into two ``fused_moe`` calls.  This probe answers
whether HIP VMM lets us build the ``[E + B]`` layout on gfx950:

    reserve (R + 1) slots of ``epn_padded * row_bytes``
      slot 0..R-1  <- peer pe's home experts, row = pe * epn_padded + k
      slot R       <- local prefetch slots (inference: B is small)

Modelled on mori's CCO, which already builds this exact shape: one
``hipMemAddressReserve``, then every imported peer handle ``hipMemMap``'d at
``flatBase + pe * perRankSize + slotOffset`` (``src/cco/cco_init.cpp``,
``mapPeer``).  CCO itself cannot be reused -- it rounds ``perRankSize`` up to
4 GiB so the device side can encode the stride as ``perRankSize >> 32``, which
leaves 4 GiB holes between peers.  Here the stride is exactly one expert group,
so rows stay contiguous across the seam.

Inference only: no gradient buffers, no reduce path.
Needs ROCm >= 7.1.4 (earlier runtimes leak a DRM fd per hipMemUnmap and race in
hsa_amd_vmem_map; see mori 91d1710e / rocm-systems#4363).

Run:
    torchrun --nproc-per-node 8 op_tests/probe_vmm_contiguous.py
    torchrun --nproc-per-node 8 op_tests/probe_vmm_contiguous.py --row-bytes 1048576
"""

from __future__ import annotations

import argparse
import ctypes
import math
import os
import shutil
import socket
import sys

import torch
import torch.distributed as dist

# ---------------------------------------------------------------- HIP bindings

hip = ctypes.CDLL("libamdhip64.so")

ALLOC_TYPE_PINNED = 1
HANDLE_TYPE_POSIX_FD = 1
LOCATION_TYPE_DEVICE = 1
ACCESS_PROT_READWRITE = 3
GRANULARITY_MINIMUM = 0
GRANULARITY_RECOMMENDED = 1
MEMCPY_HTOD = 1
MEMCPY_DTOH = 2
ERR_PEER_ACCESS_ALREADY_ENABLED = 704


class MemLocation(ctypes.Structure):
    _fields_ = [("type", ctypes.c_int), ("id", ctypes.c_int)]


class AllocFlags(ctypes.Structure):
    _fields_ = [
        ("compressionType", ctypes.c_ubyte),
        ("gpuDirectRDMACapable", ctypes.c_ubyte),
        ("usage", ctypes.c_ushort),
        ("reserved", ctypes.c_ubyte * 4),
    ]


class MemAllocationProp(ctypes.Structure):
    _fields_ = [
        ("type", ctypes.c_int),
        ("requestedHandleType", ctypes.c_int),
        ("location", MemLocation),
        ("win32HandleMetaData", ctypes.c_void_p),
        ("allocFlags", AllocFlags),
    ]


class MemAccessDesc(ctypes.Structure):
    _fields_ = [("location", MemLocation), ("flags", ctypes.c_int)]


# Explicit argtypes: ctypes would otherwise pass bare python ints as 32-bit C
# int, which silently truncates the size_t arguments (>4 GiB reservations).
_P, _SZ, _I = ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int
hip.hipMemGetAllocationGranularity.argtypes = [_P, _P, _I]
hip.hipMemAddressReserve.argtypes = [_P, _SZ, _SZ, _P, ctypes.c_ulonglong]
hip.hipMemAddressFree.argtypes = [_P, _SZ]
hip.hipMemCreate.argtypes = [_P, _SZ, _P, ctypes.c_ulonglong]
hip.hipMemMap.argtypes = [_P, _SZ, _SZ, _P, ctypes.c_ulonglong]
hip.hipMemUnmap.argtypes = [_P, _SZ]
hip.hipMemSetAccess.argtypes = [_P, _SZ, _P, _SZ]
hip.hipMemRelease.argtypes = [_P]
hip.hipMemExportToShareableHandle.argtypes = [_P, _P, _I, ctypes.c_ulonglong]
hip.hipMemImportFromShareableHandle.argtypes = [_P, _P, _I]
hip.hipMemcpy.argtypes = [_P, _P, _SZ, _I]
hip.hipDeviceEnablePeerAccess.argtypes = [_I, ctypes.c_uint]
hip.hipGetErrorString.argtypes = [_I]
hip.hipGetErrorString.restype = ctypes.c_char_p


def check(err: int, what: str) -> None:
    if err != 0:
        raise RuntimeError(
            f"{what} failed: hipError {err} ({hip.hipGetErrorString(err).decode()})"
        )


def make_prop(dev: int) -> MemAllocationProp:
    p = MemAllocationProp()
    p.type = ALLOC_TYPE_PINNED
    p.requestedHandleType = HANDLE_TYPE_POSIX_FD
    p.location.type = LOCATION_TYPE_DEVICE
    p.location.id = dev
    return p


def granularity(dev: int, flag: int) -> int:
    g = _SZ(0)
    prop = make_prop(dev)
    check(
        hip.hipMemGetAllocationGranularity(
            ctypes.byref(g), ctypes.byref(prop), flag
        ),
        "hipMemGetAllocationGranularity",
    )
    return g.value


def import_handle(fd: int) -> ctypes.c_void_p:
    """ROCm 7.0.x wants &fd, 7.1.0+ wants (void*)(uintptr_t)fd -- mori hip_compat.hpp."""
    h = _P(0)
    err = hip.hipMemImportFromShareableHandle(
        ctypes.byref(h), _P(fd), HANDLE_TYPE_POSIX_FD
    )
    if err != 0:
        cfd = _I(fd)
        err = hip.hipMemImportFromShareableHandle(
            ctypes.byref(h), ctypes.byref(cfd), HANDLE_TYPE_POSIX_FD
        )
    check(err, "hipMemImportFromShareableHandle")
    return h


def enable_peer(peer_dev: int, my_dev: int) -> None:
    if peer_dev == my_dev:
        return
    err = hip.hipDeviceEnablePeerAccess(peer_dev, 0)
    # Consume the sticky last-error. hipDeviceEnablePeerAccess leaves it set
    # even on the benign AlreadyEnabled path, and a later torch op picks it up
    # via hipGetLastError() and raises "operation not supported" (observed on
    # gfx950). mori does the same -- cco_init.cpp mapPeer.
    hip.hipGetLastError()
    if err not in (0, ERR_PEER_ACCESS_ALREADY_ENABLED):
        print(f"  warn: hipDeviceEnablePeerAccess({peer_dev}) -> {err}")


def poke(addr: int, value: int) -> None:
    buf = ctypes.c_uint64(value)
    check(
        hip.hipMemcpy(_P(addr), ctypes.byref(buf), 8, MEMCPY_HTOD), "hipMemcpy H2D"
    )


def peek(addr: int) -> int:
    buf = ctypes.c_uint64(0)
    check(
        hip.hipMemcpy(ctypes.byref(buf), _P(addr), 8, MEMCPY_DTOH), "hipMemcpy D2H"
    )
    return buf.value


# ------------------------------------------------------------- fd exchange

def exchange_fds(rank: int, world: int, fd: int, sockdir: str) -> list[int]:
    """All-gather one fd per rank over unix sockets + SCM_RIGHTS.

    fds cannot ride torch.distributed; mori hits the same wall and solves it
    with LocalBootstrapNetwork ("Use cases: VMM shareable handle (file
    descriptor) exchange", local_bootstrap.hpp).

    Order matters: connect() to a listening socket completes from the backlog
    without the peer calling accept(), so every rank must finish connecting
    before anyone blocks on accept().  accept-then-connect deadlocks.
    """
    path = os.path.join(sockdir, f"r{rank}.sock")
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(path)
    srv.listen(world)
    dist.barrier()  # every socket is bound and listening

    clients = {}
    for step in range(1, world):
        peer = (rank + step) % world
        c = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        c.settimeout(60.0)
        c.connect(os.path.join(sockdir, f"r{peer}.sock"))
        clients[peer] = c

    srv.settimeout(60.0)
    for _ in range(world - 1):
        conn, _addr = srv.accept()
        try:
            socket.send_fds(conn, [b"x"], [fd])
        finally:
            conn.close()

    out = [-1] * world
    out[rank] = fd
    for peer, c in clients.items():
        try:
            _msg, fds, _flags, _addr = socket.recv_fds(c, 1, 1)
            if len(fds) != 1:
                raise RuntimeError(f"rank {rank}: no fd from peer {peer}")
            out[peer] = fds[0]
        finally:
            c.close()
    srv.close()
    return out


# ------------------------------------------------------------------ padding

def pad_group(epn: int, row_bytes: int, gran: int) -> int:
    """Smallest epn_padded >= epn with (epn_padded * row_bytes) % gran == 0.

    The group is the map unit so it must be granularity-aligned, and it must be
    a whole number of rows or the next slot starts mid-row and the row index
    formula pe*epn_padded + k breaks.  Rows *inside* a group need no alignment
    of their own -- that is why this is much weaker than per-row alignment.
    """
    step = gran // math.gcd(row_bytes, gran)
    return ((epn + step - 1) // step) * step


# --------------------------------------------------------------------- main

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--experts-per-rank", type=int, default=48)
    ap.add_argument("--prefetch-slots", type=int, default=4, help="B (inference)")
    ap.add_argument(
        "--row-bytes",
        type=int,
        default=7168 * 2048 * 2,
        help="one expert, one projection. default = DSv4 bf16 [H=7168, H'=2048]",
    )
    args = ap.parse_args()

    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    dev = rank % torch.cuda.device_count()
    torch.cuda.set_device(dev)

    epn, B, row_bytes = args.experts_per_rank, args.prefetch_slots, args.row_bytes
    gran_min = granularity(dev, GRANULARITY_MINIMUM)
    gran = granularity(dev, GRANULARITY_RECOMMENDED)

    epn_p = pad_group(epn, row_bytes, gran)
    slot_bytes = epn_p * row_bytes
    total = (world + 1) * slot_bytes  # +1: prefetch slots live past row E

    # gcd(row_bytes, granularity) small => epn_padded explodes (a 1,000,003-byte
    # row against a 2 MiB granularity needs 2,097,152 rows/group).  Diagnose it
    # instead of asking the driver to reserve petabytes.
    if epn_p > 4 * epn:
        if rank == 0:
            print(
                f"ABORT: row_bytes={row_bytes:,} is nearly coprime with "
                f"granularity={gran:,} (gcd={math.gcd(row_bytes, gran):,}), so a "
                f"whole-row group needs epn_padded={epn_p:,} vs epn={epn}.\n"
                f"       This pool cannot use the flat layout as-is -- pad each "
                f"row up to a granularity divisor first, then re-run."
            )
        dist.destroy_process_group()
        return 2

    if rank == 0:
        waste = (epn_p - epn) * row_bytes
        print("=" * 76)
        print(f"granularity   minimum={gran_min:,}  recommended={gran:,}")
        print(f"row_bytes     {row_bytes:,}  ({row_bytes / gran:.4f} x granularity)")
        print(f"epn {epn} -> epn_padded {epn_p}   group = {slot_bytes / 2**30:.3f} GiB")
        print(f"padding       {waste:,} B/group  ({waste / slot_bytes * 100:.2f}%)")
        print(f"reservation   {total / 2**30:.3f} GiB  ({world}+1 slots)")
        print(f"tail slot     holds {epn_p} rows, B={B} needs {B}")
        if B > epn_p:
            print(f"!! B={B} > epn_padded={epn_p}: tail slot too small")
        print("=" * 76)
    sys.stdout.flush()
    dist.barrier()

    sockdir = f"/tmp/probe_vmm_{os.environ.get('MASTER_PORT', 'x')}"
    if rank == 0:
        shutil.rmtree(sockdir, ignore_errors=True)
        os.makedirs(sockdir, exist_ok=True)
    dist.barrier()

    prop = make_prop(dev)
    base = _P(0)
    check(
        hip.hipMemAddressReserve(ctypes.byref(base), total, gran, None, 0),
        "hipMemAddressReserve",
    )
    if base.value % gran != 0:
        raise RuntimeError(f"reservation {base.value:#x} not granularity-aligned")

    handles = []
    peer_fds: list[int] = []
    try:
        # Own home group -> slot `rank`; local prefetch tail -> slot `world`.
        for slot, tag in ((rank, "home"), (world, "prefetch")):
            h = _P(0)
            check(
                hip.hipMemCreate(ctypes.byref(h), slot_bytes, ctypes.byref(prop), 0),
                f"hipMemCreate ({tag})",
            )
            handles.append(h)
            check(
                hip.hipMemMap(_P(base.value + slot * slot_bytes), slot_bytes, 0, h, 0),
                f"hipMemMap ({tag})",
            )

        fd = _I(-1)
        check(
            hip.hipMemExportToShareableHandle(
                ctypes.byref(fd), handles[0], HANDLE_TYPE_POSIX_FD, 0
            ),
            "hipMemExportToShareableHandle",
        )
        peer_fds = exchange_fds(rank, world, fd.value, sockdir)

        peer_devs = [0] * world
        dist.all_gather_object(peer_devs, dev)
        for pe in range(world):
            if pe == rank:
                continue
            enable_peer(peer_devs[pe], dev)
            h = import_handle(peer_fds[pe])
            handles.append(h)
            check(
                hip.hipMemMap(_P(base.value + pe * slot_bytes), slot_bytes, 0, h, 0),
                f"hipMemMap (peer {pe})",
            )

        # One SetAccess over the whole range -- every slot is backed by now.
        desc = MemAccessDesc()
        desc.location.type = LOCATION_TYPE_DEVICE
        desc.location.id = dev
        desc.flags = ACCESS_PROT_READWRITE
        check(
            hip.hipMemSetAccess(base, total, ctypes.byref(desc), 1), "hipMemSetAccess"
        )

        # Sign our own rows, then read everyone's back through the flat range.
        for k in range(epn):
            poke(base.value + (rank * epn_p + k) * row_bytes, (rank << 32) | k)
        poke(base.value + world * slot_bytes, 0xB10C0000 | rank)
        torch.cuda.synchronize()
        dist.barrier()

        bad = 0
        for pe in range(world):
            for k in range(epn):
                row = pe * epn_p + k
                got, want = peek(base.value + row * row_bytes), (pe << 32) | k
                if got != want:
                    bad += 1
                    if bad <= 5:
                        print(
                            f"[rank {rank}] row {row} (pe={pe},k={k}): "
                            f"got {got:#x} want {want:#x}"
                        )
        tail = peek(base.value + world * slot_bytes)
        tail_ok = tail == (0xB10C0000 | rank)
        if not tail_ok:
            print(f"[rank {rank}] prefetch slot: got {tail:#x}")

        ok = torch.tensor(
            [1 if (bad == 0 and tail_ok) else 0], dtype=torch.int32, device="cuda"
        )
        dist.all_reduce(ok)
        if rank == 0:
            n = int(ok.item())
            print(f"\n{'PASS' if n == world else 'FAIL'}: {n}/{world} ranks verified")
            if n == world:
                print(
                    f"row-contiguous [{world * epn_p} + {B}] VA works -- "
                    f"one grouped GEMM can address the whole range"
                )
        dist.barrier()
    finally:
        hip.hipMemUnmap(base, total)
        for h in handles:
            hip.hipMemRelease(h)
        hip.hipMemAddressFree(base, total)
        for i, f in enumerate(peer_fds):
            if f >= 0 and i != rank:
                os.close(f)
        if rank == 0:
            shutil.rmtree(sockdir, ignore_errors=True)

    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    sys.exit(main())
