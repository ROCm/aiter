# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""Probe 2: feed a VMM-stitched [E+B] weight tensor to aiter's fused_moe.

Probe 1 (``probe_vmm_contiguous.py``) proved the addressing: R peers' expert
weights can be mapped into one row-contiguous VA and read back by row index.
It verified with ``hipMemcpy``, which says nothing about whether the *GEMM*
will accept such a tensor.  This one closes that gap:

  ref  = ordinary local [E+B, ...] tensors, every expert materialised locally
  vmm  = [E+B, ...] over the stitched VA; each rank writes ONLY its own epn
         home experts, so every other expert's rows are peer memory over XGMI

Both go through the same ``fused_moe`` call with the same tokens and topk_ids
covering all E experts.  Bit-identical output == the grouped GEMM reads remote
rows correctly and the [E+B] layout is usable as-is.

Run:
    torchrun --nproc-per-node 8 op_tests/probe_fused_moe_vmm.py
"""

from __future__ import annotations

import argparse
import ctypes
import os
import shutil
import sys

import torch
import torch.distributed as dist

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from probe_vmm_contiguous import (  # noqa: E402
    ACCESS_PROT_READWRITE,
    GRANULARITY_RECOMMENDED,
    HANDLE_TYPE_POSIX_FD,
    LOCATION_TYPE_DEVICE,
    MemAccessDesc,
    check,
    enable_peer,
    exchange_fds,
    granularity,
    hip,
    import_handle,
    make_prop,
    pad_group,
)

_P, _I = ctypes.c_void_p, ctypes.c_int


class RawBuf:
    """Wrap a raw device pointer so torch.as_tensor can adopt it.

    Same trick mori uses for its symmetric tensors (mori/python/mori/shmem/
    tensor_utils.py: MoriShmemBuffer) -- no C++ extension needed.
    """

    def __init__(self, ptr: int, nbytes: int):
        self.__cuda_array_interface__ = {
            "data": (ptr, False),
            "shape": (nbytes,),
            "typestr": "<u1",
            "strides": None,
            "version": 3,
        }


class FlatPool:
    """One row-contiguous [(R+1) * epn_padded, ...] VA over R peers + local tail."""

    def __init__(self, tag, row_bytes, epn, rank, world, dev, sockdir):
        self.tag, self.rank, self.world, self.dev = tag, rank, world, dev
        gran = granularity(dev, GRANULARITY_RECOMMENDED)
        self.epn_p = pad_group(epn, row_bytes, gran)
        self.slot_bytes = self.epn_p * row_bytes
        self.total = (world + 1) * self.slot_bytes
        self.handles, self.peer_fds = [], []

        prop = make_prop(dev)
        self.base = _P(0)
        check(
            hip.hipMemAddressReserve(
                ctypes.byref(self.base), self.total, gran, None, 0
            ),
            f"[{tag}] hipMemAddressReserve",
        )
        for slot in (rank, world):
            h = _P(0)
            check(
                hip.hipMemCreate(
                    ctypes.byref(h), self.slot_bytes, ctypes.byref(prop), 0
                ),
                f"[{tag}] hipMemCreate",
            )
            self.handles.append(h)
            check(
                hip.hipMemMap(
                    _P(self.base.value + slot * self.slot_bytes),
                    self.slot_bytes, 0, h, 0,
                ),
                f"[{tag}] hipMemMap local",
            )

        fd = _I(-1)
        check(
            hip.hipMemExportToShareableHandle(
                ctypes.byref(fd), self.handles[0], HANDLE_TYPE_POSIX_FD, 0
            ),
            f"[{tag}] hipMemExportToShareableHandle",
        )
        os.makedirs(sockdir, exist_ok=True)
        dist.barrier()
        self.peer_fds = exchange_fds(rank, world, fd.value, sockdir)

        devs = [0] * world
        dist.all_gather_object(devs, dev)
        for pe in range(world):
            if pe == rank:
                continue
            enable_peer(devs[pe], dev)
            h = import_handle(self.peer_fds[pe])
            self.handles.append(h)
            check(
                hip.hipMemMap(
                    _P(self.base.value + pe * self.slot_bytes),
                    self.slot_bytes, 0, h, 0,
                ),
                f"[{tag}] hipMemMap peer {pe}",
            )

        desc = MemAccessDesc()
        desc.location.type = LOCATION_TYPE_DEVICE
        desc.location.id = dev
        desc.flags = ACCESS_PROT_READWRITE
        check(
            hip.hipMemSetAccess(self.base, self.total, ctypes.byref(desc), 1),
            f"[{tag}] hipMemSetAccess",
        )

    def tensor(self, dtype, shape) -> torch.Tensor:
        t = torch.as_tensor(RawBuf(self.base.value, self.total), device="cuda")
        return t.view(dtype)[: torch.Size(shape).numel()].view(*shape)

    def close(self):
        hip.hipMemUnmap(self.base, self.total)
        for h in self.handles:
            hip.hipMemRelease(h)
        hip.hipMemAddressFree(self.base, self.total)
        for i, f in enumerate(self.peer_fds):
            if f >= 0 and i != self.rank:
                os.close(f)


def expert_w(e: int, shape, dev) -> torch.Tensor:
    """Deterministic per-expert weights -- both sides must agree exactly."""
    g = torch.Generator(device=dev).manual_seed(9000 + e)
    return (
        torch.randn(*shape, generator=g, device=dev, dtype=torch.float32) * 0.05
    ).to(torch.bfloat16)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--experts-per-rank", type=int, default=2)
    ap.add_argument("--prefetch-slots", type=int, default=2)
    ap.add_argument("--hidden", type=int, default=1024)
    ap.add_argument("--inter", type=int, default=512)
    ap.add_argument("--tokens", type=int, default=256)
    args = ap.parse_args()

    from aiter import ActivationType, QuantType
    from aiter.fused_moe import fused_moe

    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    dev = rank % torch.cuda.device_count()
    torch.cuda.set_device(dev)
    devs = f"cuda:{dev}"

    epn, B = args.experts_per_rank, args.prefetch_slots
    E, H, I, n = world * epn, args.hidden, args.inter, args.tokens
    w1_shape, w2_shape = (2 * I, H), (H, I)
    w1_row = w1_shape[0] * w1_shape[1] * 2  # bf16
    w2_row = w2_shape[0] * w2_shape[1] * 2

    sockdir = f"/tmp/probe_fm_{os.environ.get('MASTER_PORT', 'x')}"
    if rank == 0:
        shutil.rmtree(sockdir, ignore_errors=True)
        print("=" * 74)
        print(f"E={E} (R={world} x epn={epn})  B={B}  H={H}  inter={I}  tokens={n}")
        print(f"w1 row {w1_row:,} B   w2 row {w2_row:,} B")
        print("=" * 74)
    dist.barrier()

    p1 = FlatPool("w1", w1_row, epn, rank, world, dev, sockdir + "/w1")
    p2 = FlatPool("w2", w2_row, epn, rank, world, dev, sockdir + "/w2")
    rows = (world + 1) * p1.epn_p
    assert p1.epn_p == epn and p2.epn_p == epn, "padding would shift row indices"

    vmm_w1 = p1.tensor(torch.bfloat16, (rows, *w1_shape))
    vmm_w2 = p2.tensor(torch.bfloat16, (rows, *w2_shape))
    if rank == 0:
        print(f"vmm_w1 {tuple(vmm_w1.shape)} contiguous={vmm_w1.is_contiguous()} "
              f"ptr={vmm_w1.data_ptr():#x}")

    # Reference: every expert materialised locally, ordinary allocation.
    ref_w1 = torch.zeros(rows, *w1_shape, dtype=torch.bfloat16, device=devs)
    ref_w2 = torch.zeros(rows, *w2_shape, dtype=torch.bfloat16, device=devs)
    for e in range(E):
        ref_w1[e] = expert_w(e, w1_shape, devs)
        ref_w2[e] = expert_w(1000 + e, w2_shape, devs)

    # VMM side: write ONLY our own home experts. Every other row is peer memory
    # and must arrive over XGMI -- deliberately never touched from here.
    for k in range(epn):
        e = rank * epn + k
        vmm_w1[e].copy_(ref_w1[e])
        vmm_w2[e].copy_(ref_w2[e])
    for b in range(B):  # local tail rows, unused by the GEMM here
        vmm_w1[world * epn + b].zero_()
        vmm_w2[world * epn + b].zero_()
    torch.cuda.synchronize()
    dist.barrier()

    # Sanity: does the flat view see peers' rows before we even call the GEMM?
    seen = sum(
        1 for e in range(E) if torch.equal(vmm_w1[e], ref_w1[e])
    )
    if seen != E and rank == 0:
        print(f"!! flat view sees only {seen}/{E} experts correctly")

    g = torch.Generator(device=devs).manual_seed(4242)
    x = (torch.randn(n, H, generator=g, device=devs, dtype=torch.float32) * 0.1).to(
        torch.bfloat16
    )
    topk_ids = (
        torch.arange(n, device=devs, dtype=torch.int32) % E
    ).view(n, 1)  # every expert exercised, incl. all 7 remote ranks
    topk_w = torch.ones(n, 1, dtype=torch.float32, device=devs)

    def run(w1, w2):
        return fused_moe(
            x, w1, w2, topk_w, topk_ids, None,
            ActivationType.Silu, quant_type=QuantType.No,
        )

    out_ref = run(ref_w1, ref_w2)
    torch.cuda.synchronize()
    out_vmm = run(vmm_w1, vmm_w2)
    torch.cuda.synchronize()

    exact = torch.equal(out_ref, out_vmm)
    if not exact:
        d = (out_ref.float() - out_vmm.float()).abs()
        print(
            f"[rank {rank}] MISMATCH max={d.max():.3e} mean={d.mean():.3e} "
            f"bad={(d > 0).sum().item()}/{d.numel()}"
        )
    ok = torch.tensor([1 if (exact and seen == E) else 0], dtype=torch.int32,
                      device=devs)
    dist.all_reduce(ok)
    if rank == 0:
        nok = int(ok.item())
        print(f"\n{'PASS' if nok == world else 'FAIL'}: {nok}/{world} ranks "
              f"bit-identical vs local reference")
        if nok == world:
            print(
                "fused_moe consumes the VMM-stitched [E+B] tensor unchanged; "
                "remote expert rows are read correctly over XGMI"
            )
    dist.barrier()

    del vmm_w1, vmm_w2
    p2.close()
    p1.close()
    if rank == 0:
        shutil.rmtree(sockdir, ignore_errors=True)
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    sys.exit(main())
