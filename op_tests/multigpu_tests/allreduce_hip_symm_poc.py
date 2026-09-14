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
"""One-shot all-reduce over mori's torch SymmetricMemory backend, in HIP.

This is the all-reduce companion to ``all2all_hip.py`` and follows the same shape:
a gloo process group for the store handshake, mori registered as the "MORI"
SymmetricMemory backend, and a JIT-built HIP kernel driven straight off the peer
pointer array torch publishes (``hdl.buffer_ptrs_dev``).

Reduction model — pull / naive: every rank publishes its input into a symmetric
window; the kernel then reads *every* peer's window and sums in fp32, so after a
single barrier every rank holds the full reduction. This is the direct analog of
the gfx1250 ``ar_gfx1250_naive_unroll4`` kernel, just addressing peers through
symm_mem pointers instead of the C++ RankData table.

Single node (<=4 GPUs on gfx1250)::

    torchrun --nnodes=1 --nproc_per_node=4 allreduce_hip_symm_poc.py --elems 1048576

Eight GPUs across two 4-GPU nodes — run the SAME command on BOTH nodes, changing
only ``--node_rank`` (torchrun does the cross-node handshake over master_addr, then
mori exchanges fabric handles / POSIX fds during rendezvous)::

    # on node 0 (owns the rendezvous address):
    torchrun --nnodes=2 --node_rank=0 --nproc_per_node=4 \
        --master_addr=<node0-ip> --master_port=<port> allreduce_hip_symm_poc.py

    # on node 1:
    torchrun --nnodes=2 --node_rank=1 --nproc_per_node=4 \
        --master_addr=<node0-ip> --master_port=<port> allreduce_hip_symm_poc.py

The kernel is JIT-built on first run, so there is nothing to build first.
"""

import argparse
import gc
import os
import sys

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from mori.allocator import handle_type  # importing registers the "MORI" backend
from torch.utils.cpp_extension import load_inline

HIP_SOURCE = r"""
#include <c10/hip/HIPStream.h>

#include <algorithm>

constexpr int kThreads = 256;

// One-shot all-reduce (pull). peers[r] is rank r's input window; every rank sees
// the same set of windows after rendezvous. Each thread owns a float4 lane, reads
// that lane from all peers, sums in fp32, and writes the result to the local out.
// No remote writes and no per-peer handshake, so the caller barriers first (all
// inputs published) and the result is complete the moment the kernel returns.
__global__ void AllReduceSumPtrs(void** __restrict__ peers, float4* __restrict__ out,
                                 size_t n_vec, int world_size) {
  const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
  for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n_vec;
       i += stride) {
    float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);
    for (int r = 0; r < world_size; ++r) {
      const float4 v = reinterpret_cast<const float4*>(peers[r])[i];
      acc.x += v.x;
      acc.y += v.y;
      acc.z += v.z;
      acc.w += v.w;
    }
    out[i] = acc;
  }
}

// Enough blocks to fill the device but no more than there is work to do.
static int NumBlocks(int64_t n_vec) {
  static const int cus = [] {
    int v = 64, dev = 0;
    hipGetDevice(&dev);
    hipDeviceGetAttribute(&v, hipDeviceAttributeMultiprocessorCount, dev);
    return v;
  }();
  const int64_t need = (n_vec + kThreads - 1) / kThreads;
  return static_cast<int>(std::max<int64_t>(1, std::min<int64_t>(cus * 4, need)));
}

// out is local, n_elems floats. peers_dev is hdl.buffer_ptrs_dev.
void allreduce_sum_ptrs(const at::Tensor& out, int64_t peers_dev, int64_t n_elems,
                        int64_t world_size) {
  TORCH_CHECK(out.is_contiguous(), "out must be contiguous");
  TORCH_CHECK(out.scalar_type() == at::kFloat, "out must be float32");
  TORCH_CHECK(n_elems % 4 == 0, "n_elems must be a multiple of 4, got ", n_elems);
  TORCH_CHECK(out.numel() == n_elems, "out holds ", out.numel(), " elems, expected ", n_elems);

  const size_t n_vec = static_cast<size_t>(n_elems) / 4;
  hipLaunchKernelGGL(AllReduceSumPtrs, dim3(NumBlocks(n_vec)), dim3(kThreads), 0,
                     c10::hip::getCurrentHIPStream(),
                     reinterpret_cast<void**>(peers_dev),
                     reinterpret_cast<float4*>(out.data_ptr()), n_vec,
                     static_cast<int>(world_size));
  hipError_t err = hipGetLastError();
  TORCH_CHECK(err == hipSuccess, "allreduce_sum_ptrs launch failed: ", hipGetErrorString(err));
}
"""


def build():
    """JIT the kernel. Concurrent torchrun ranks share one build: torch holds a file lock."""
    return load_inline(
        name="allreduce_hip_kernel",
        cpp_sources=(
            "void allreduce_sum_ptrs(const at::Tensor&, int64_t, int64_t, int64_t);"
        ),
        cuda_sources=HIP_SOURCE,
        functions=["allreduce_sum_ptrs"],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3"],
    )


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--elems",
        type=int,
        default=1 << 20,
        help="float32 elements per rank (must be a multiple of 4)",
    )
    p.add_argument("--iters", type=int, default=20, help="timed iterations")
    p.add_argument("--warmup", type=int, default=5)
    return p.parse_args()


def main():
    args = parse_args()
    if args.elems % 4 != 0:
        raise SystemExit(f"--elems must be a multiple of 4, got {args.elems}")
    reduce_sum = build().allreduce_sum_ptrs

    # Defaults so a bare `python3 allreduce_hip_symm_poc.py` is a 1-rank smoke test;
    # torchrun sets these (WORLD_SIZE spans both nodes for the 2x4 run).
    for key, val in (
        ("RANK", "0"),
        ("WORLD_SIZE", "1"),
        ("LOCAL_RANK", "0"),
        ("MASTER_ADDR", "127.0.0.1"),
        ("MASTER_PORT", "29500"),
    ):
        os.environ.setdefault(key, val)
    rank_id = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])

    dist.init_process_group("gloo")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    group_name = dist.group.WORLD.group_name

    symm_mem.set_backend("MORI")
    symm_mem.enable_symm_mem_for_group(group_name)

    n = args.elems

    # Input window: this rank's contribution, published to peers. Constant value
    # (rank_id + 1) per rank makes the reduction exact in fp32 and trivial to
    # check: sum_{r=0}^{ws-1} (r + 1) == ws * (ws + 1) / 2.
    inp = symm_mem.empty(n, dtype=torch.float32, device=device)
    inp.fill_(float(rank_id + 1))
    # Ordinary local memory for the result (peers never read it).
    out = torch.empty(n, dtype=torch.float32, device=device)
    torch.cuda.synchronize()

    hdl = symm_mem.rendezvous(inp, group_name)
    peers_dev = hdl.buffer_ptrs_dev

    if rank_id == 0:
        print(
            f"kernel=HIP  world={world_size}  handle={handle_type(local_rank)}  "
            f"elems={n} ({n * 4 / 1024:.0f} KiB/rank)"
        )
        print(f"peers: {' '.join(hex(p) for p in hdl.buffer_ptrs)}")

    expected = float(world_size * (world_size + 1) // 2)

    def run_once():
        reduce_sum(out, peers_dev, n, world_size)
        torch.cuda.synchronize()
        dist.barrier()  # no device-side barrier in the backend yet

    # Everyone must have published inp before anyone reads peer windows.
    dist.barrier()
    run_once()

    # Correctness: every element must equal the full reduction.
    max_abs_err = (out - expected).abs().max().item()
    errors = int((out != expected).sum().item())
    if errors:
        bad = (out != expected).nonzero(as_tuple=True)[0][:4].tolist()
        print(
            f"[rank {rank_id}] {errors} mismatches (expected {expected}); "
            f"first at {bad} = {out[bad].tolist()}",
            flush=True,
        )
    failed = torch.tensor([errors], dtype=torch.int64)
    dist.all_reduce(failed, op=dist.ReduceOp.SUM)
    if rank_id == 0 and failed.item() == 0:
        print(f"correctness: OK (sum of 1..{world_size} = {expected:.0f}, max_abs_err={max_abs_err:.1e})")

    for _ in range(args.warmup):
        run_once()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    dist.barrier()
    start.record()
    for _ in range(args.iters):
        reduce_sum(out, peers_dev, n, world_size)
    end.record()
    torch.cuda.synchronize()
    ms = start.elapsed_time(end) / args.iters

    # Pull all-reduce moves (world_size - 1) remote input chunks into each rank;
    # the self chunk stays local. Report the aggregate read bandwidth.
    n_bytes = n * 4
    gbps = torch.tensor(
        [(world_size - 1) * n_bytes / (ms / 1e3) / 1e9], dtype=torch.float64
    )
    dist.all_reduce(gbps, op=dist.ReduceOp.SUM)
    if rank_id == 0:
        print(f"{ms * 1e3:7.1f} us/iter, {gbps.item():8.1f} GB/s aggregate")
        print("SUCCESS" if failed.item() == 0 else "FAILED")

    dist.barrier()
    del hdl, inp
    gc.collect()
    dist.destroy_process_group()
    return 0 if failed.item() == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
