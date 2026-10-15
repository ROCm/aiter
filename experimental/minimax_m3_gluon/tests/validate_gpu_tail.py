# SPDX-License-Identifier: MIT
"""Four-GPU HIP-IPC mailbox and reduction smoke test."""

import os
import ctypes as C
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist

from minimax_m3_gluon.hip_ipc import DevicePointer
from minimax_m3_gluon.host import allocate_peer, allocate_workspace
from minimax_m3_gluon.tail import tail


def main():
    rank = int(os.environ["RANK"])
    torch.cuda.set_device(rank)
    dist.init_process_group("gloo", timeout=timedelta(seconds=120))
    try:
        for batch in (1, 4, 8, 16):
            state = allocate_workspace(batch, "cuda")
            with allocate_peer(batch, rank, dist.group.WORLD) as peer:
                tail.warmup(
                    state["SortOut"], peer.addresses, DevicePointer(peer.ptr.value),
                    state["Out"], state["TailEpochs"], state["TailErrors"],
                    B=batch, RANK=rank, num_warps=8, grid=(256,),
                )
                dist.barrier()
                for iteration in range(3):
                    # Distinct data for every rank, position and epoch.
                    indices = torch.arange(batch * 6144, device="cuda", dtype=torch.float32)
                    values = ((indices % 127) * 0.001 + rank * 0.125 + iteration * 0.01).bfloat16()
                    data = values.reshape(batch, 6144)
                    state["TailErrors"].zero_()
                    dist.barrier()
                    kernel = tail[(256,)](
                        data, peer.addresses, DevicePointer(peer.ptr.value),
                        state["Out"], state["TailEpochs"], state["TailErrors"],
                        B=batch, RANK=rank, num_warps=8,
                    )
                    torch.cuda.synchronize()
                    if iteration == 0:
                        hip_path = next(
                            line.split()[-1] for line in Path("/proc/self/maps").read_text().splitlines()
                            if "/libamdhip64.so" in line
                        )
                        hip = C.CDLL(hip_path)
                        occupancy = hip.hipModuleOccupancyMaxActiveBlocksPerMultiprocessor
                        occupancy.argtypes = [C.POINTER(C.c_int), C.c_void_p, C.c_int, C.c_size_t]
                        occupancy.restype = C.c_int
                        resident = C.c_int()
                        rc = occupancy(C.byref(resident), C.c_void_p(kernel.function), 512, kernel.metadata.shared)
                        assert rc == 0 and resident.value >= 1
                        assert kernel.n_spills == 0
                        if rank == 0:
                            if batch == 1:
                                isa = kernel.asm["amdgcn"]
                                assert "global_load_dwordx2" in isa and "sc1" in isa
                                print(
                                    "tail ISA:",
                                    [line.strip() for line in isa.splitlines()
                                     if "sc1" in line or "atomic_swap" in line][:6],
                                    flush=True,
                                )
                            print(f"B={batch}: tail VGPR={kernel.n_regs}, resident/CU={resident.value}", flush=True)
                    assert not state["TailErrors"].any().item(), f"rank {rank} poll timeout"
                    expected = torch.zeros_like(values, dtype=torch.float32)
                    for source in range(4):
                        expected += ((indices % 127) * 0.001 + source * 0.125 + iteration * 0.01).bfloat16().float()
                    assert torch.equal(state["Out"].flatten(), expected.bfloat16()), f"rank {rank} reduction differs"
                    dist.barrier()
                if rank == 0:
                    print(f"B={batch}: TP4 IPC reduction and 3 epochs OK", flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
