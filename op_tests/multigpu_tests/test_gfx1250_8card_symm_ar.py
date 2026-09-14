#!/usr/bin/env python3
# Copyright © Advanced Micro Devices, Inc. All rights reserved.
#
# MIT License
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""Cross-node smoke test for aiter CustomAllreduce on gfx1250 via mori symm_mem.

Drives the *real* CustomAllreduce class (not a hand-written kernel) end to end:
construct it on a gloo group, run custom_all_reduce(), and check the reduction.
This is the 6/8-card path that routes peer pointers through mori's torch
SymmetricMemory "MORI" backend, so it works across nodes on the MI450 fabric.

Run the SAME command on BOTH nodes, changing only --node_rank. torchrun does the
cross-node store handshake over master_addr/port; mori exchanges fabric handles
(or POSIX fds same-node) inside symm_mem.rendezvous().

    # node 0 (owns the rendezvous address):
    torchrun --nnodes=2 --node_rank=0 --nproc_per_node=4 \
        --master_addr=<node0-ip> --master_port=29500 \
        op_tests/multigpu_tests/test_gfx1250_8card_symm_ar.py

    # node 1:
    torchrun --nnodes=2 --node_rank=1 --nproc_per_node=4 \
        --master_addr=<node0-ip> --master_port=29500 \
        op_tests/multigpu_tests/test_gfx1250_8card_symm_ar.py

Single node (<=4 GPUs) works too for a quick local check:

    torchrun --nnodes=1 --nproc_per_node=4 \
        op_tests/multigpu_tests/test_gfx1250_8card_symm_ar.py
"""

import argparse
import os
import sys

# gfx1250 has no CK support; force the CK-free build so the JIT compile of the
# custom_all_reduce_gfx1250 kernel succeeds. Must be set before importing aiter
# (ENABLE_CK is read at aiter import time). setdefault keeps an explicit override.
os.environ.setdefault("ENABLE_CK", "0")
# Opt into the mori symm_mem transport — this is what makes 6/8-card / cross-node
# work. Must be set before importing custom_all_reduce (read at module import).
os.environ.setdefault("AITER_CUSTOM_AR_USE_SYMM_MEM", "1")

import torch
import torch.distributed as dist


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--elems", type=int, default=1024 * 1024, help="bf16 elements")
    p.add_argument("--iters", type=int, default=50, help="timed iterations")
    p.add_argument("--warmup", type=int, default=10)
    return p.parse_args()


def main():
    args = parse_args()

    # torchrun sets these; provide 1-rank defaults so a bare run is a smoke test.
    for key, val in (
        ("RANK", "0"),
        ("WORLD_SIZE", "1"),
        ("LOCAL_RANK", "0"),
        ("MASTER_ADDR", "127.0.0.1"),
        ("MASTER_PORT", "29500"),
    ):
        os.environ.setdefault(key, val)
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])

    def log(msg):
        print(f"[rank {rank}] {msg}", flush=True)

    # gloo for the store handshake only — CustomAllreduce requires a non-NCCL
    # group and does all peer addressing itself through symm_mem.
    dist.init_process_group("gloo")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    props = torch.cuda.get_device_properties(device)
    arch = getattr(props, "gcnArchName", "")
    if rank == 0:
        log(f"arch={arch}  world_size={world_size}  "
            f"AITER_CUSTOM_AR_USE_SYMM_MEM={os.environ.get('AITER_CUSTOM_AR_USE_SYMM_MEM')}")

    # Import after env + device are set (import triggers the gfx1250 JIT build).
    from aiter.dist.device_communicators.custom_all_reduce import CustomAllreduce

    ca = CustomAllreduce(group=dist.group.WORLD, device=device)

    # Fail loudly instead of silently degrading: a disabled CA, or one that fell
    # back off symm_mem, means cross-node would not actually be exercised.
    ok = torch.tensor([1 if not ca.disabled else 0], dtype=torch.int64)
    dist.all_reduce(ok, op=dist.ReduceOp.MIN)
    if ok.item() == 0:
        if rank == 0:
            log("FAILED: CustomAllreduce disabled on at least one rank "
                "(unsupported world_size, probe failure, or non-gfx1250).")
        dist.destroy_process_group()
        return 1

    if world_size > 1 and not ca._use_symm_mem:
        # Not fatal single-node (VMM/IPC also works), but on >1 node this means
        # the symm_mem probe failed and cross-node would not work.
        log("WARNING: CustomAllreduce is NOT on the symm_mem transport "
            "(fell back to VMM/IPC — same-node only).")

    # Each rank contributes (rank+1); the all-reduce sum is 1+2+...+world_size.
    # That sum (<= 36 for world<=8) is exact in bf16, so allclose is strict.
    val = float(rank + 1)
    want = float(world_size * (world_size + 1) // 2)
    inp = torch.full((args.elems,), val, dtype=torch.bfloat16, device=device)

    torch.cuda.synchronize()
    dist.barrier()

    out = ca.custom_all_reduce(inp)
    if out is None:
        log("FAILED: custom_all_reduce returned None (size did not fit the "
            "custom path).")
        dist.destroy_process_group()
        return 1
    torch.cuda.synchronize()

    max_abs_err = (out.float() - want).abs().max().item()
    errors = 0 if max_abs_err < 1e-3 else 1
    if errors:
        log(f"correctness FAILED: max_abs_err={max_abs_err}, want={want}, "
            f"got[0]={out[0].item()}")

    failed = torch.tensor([errors], dtype=torch.int64)
    dist.all_reduce(failed, op=dist.ReduceOp.SUM)
    if rank == 0 and failed.item() == 0:
        log(f"correctness: OK (sum == {want:.0f} across {world_size} ranks)")

    # ---- bandwidth ----
    for _ in range(args.warmup):
        ca.custom_all_reduce(inp)
    torch.cuda.synchronize()
    dist.barrier()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(args.iters):
        ca.custom_all_reduce(inp)
    end.record()
    torch.cuda.synchronize()
    ms = start.elapsed_time(end) / args.iters
    nbytes = inp.numel() * inp.element_size()
    # allreduce moves ~2*(N-1)/N * bytes of algorithmic traffic per rank.
    algbw = nbytes / (ms / 1e3) / 1e9
    if rank == 0:
        log(f"{ms * 1e3:8.1f} us/iter  |  {nbytes/1e6:.2f} MB  |  "
            f"{algbw:7.1f} GB/s (payload/time)")
        log("SUCCESS" if failed.item() == 0 else "FAILED")

    dist.barrier()
    ca.close()
    dist.destroy_process_group()
    return 0 if failed.item() == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
