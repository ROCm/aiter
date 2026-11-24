# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""The most a decode-attention kernel can reach against utils' roofline.

The roofline counts one dispatch plus the bytes at peak bandwidth. A paged
kernel also pays a dependent page-table read and DRAM latency, which at short
context are most of the time. This times a kernel that only streams the same
bytes after one dependent page-table read, cold (rotated through a >= 96 MiB
working set) under a HIP graph, as the attention harness does. Its %roof at a
configuration's bytes is that configuration's ceiling.

    python op_tests/op_benchmarks/rdna_attn/floor.py \\
        --hq 16 --hkv 16 --head-dim 32 --m 1 4

Prints the stream floor per size, then the ceiling per configuration, M and
context.
"""

import argparse
import bisect
import math
import pathlib
import statistics
import sys

import torch
from torch.utils.cpp_extension import load_inline

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import harness
import utils

_SRC = r"""
#include <hip/hip_runtime.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
typedef unsigned int u4 __attribute__((ext_vector_type(4)));
// Each workgroup reads its page index, the dependent load a paged kernel
// cannot avoid, then streams its share of the buffer in b128.
template <int UNR>
__global__ __launch_bounds__(256) void stream(const u4* __restrict__ p,
                                              const int* __restrict__ bt,
                                              long n16, unsigned* out) {
  const long stride = (long)gridDim.x * 256 * UNR;
  u4 acc = {0, 0, 0, 0};
  const long base = __builtin_amdgcn_readfirstlane(bt[blockIdx.x & 7]);
  for (long i = (long)blockIdx.x * 256 * UNR + threadIdx.x; i < n16;
       i += stride) {
    u4 v[UNR];
#pragma unroll
    for (int u = 0; u < UNR; ++u) {
      const long j = i + (long)u * 256 + base;
      v[u] = j < n16 ? p[j] : u4{0, 0, 0, 0};
    }
#pragma unroll
    for (int u = 0; u < UNR; ++u) acc ^= v[u];
  }
  const unsigned r = acc.x ^ acc.y ^ acc.z ^ acc.w;
  if (r == 0x12345678u) out[blockIdx.x] = r;
}
__global__ void empty(unsigned* out) {
  if (threadIdx.x == 999) out[0] = 1;
}
void run(torch::Tensor buf, torch::Tensor bt, int64_t n16, int64_t grid,
         int64_t unr, torch::Tensor out) {
  auto s = at::cuda::getCurrentCUDAStream();
  auto* p = (const u4*)buf.data_ptr();
  auto* o = (unsigned*)out.data_ptr();
  const int* b = bt.data_ptr<int>();
  if (unr < 0) {
    hipLaunchKernelGGL(empty, dim3(1), dim3(32), 0, s, o);
    return;
  }
#define L(U) hipLaunchKernelGGL((stream<U>), dim3(grid), dim3(256), 0, s, p, b, n16, o)
  if (unr == 1) L(1); else if (unr == 2) L(2); else if (unr == 4) L(4); else L(8);
}
"""

CONTEXTS = [128, 512, 1024, 4096, 8192, 16384, 32768]


def _module():
    sdk = pathlib.Path(torch.__file__).resolve().parents[1] / "_rocm_sdk_devel"
    return load_inline(
        "rdna_floor_probe",
        cpp_sources="void run(torch::Tensor, torch::Tensor, int64_t, int64_t, "
        "int64_t, torch::Tensor);",
        cuda_sources=_SRC,
        functions=["run"],
        extra_cuda_cflags=["-O3"],
        extra_ldflags=[f"-L{sdk / 'lib'}"],
    )


def _bench(mod, bufs, bts, n16, grid, unr, out):
    def fn():
        for b, t in zip(bufs, bts):
            mod.run(b, t, n16, grid, unr, out)

    fn()
    torch.cuda.synchronize()
    ms = statistics.median(harness.do_bench_cudagraph(fn))
    return ms * 1000 / len(bufs)


def measure(mod, sizes_kib, working_set_mb=96):
    """Best-of-grid stream time, in us, for each size."""
    out = torch.zeros(4096, dtype=torch.int32, device="cuda")
    floor = {}
    for kib in sizes_kib:
        nbytes = kib * 1024
        layers = max(10, -(-working_set_mb * 1024**2 // nbytes))
        bufs = [
            torch.randint(0, 255, (nbytes,), dtype=torch.uint8, device="cuda")
            for _ in range(layers)
        ]
        bts = [torch.zeros(8, dtype=torch.int32, device="cuda") for _ in bufs]
        n16 = nbytes // 16
        best = math.inf
        for grid in (8, 16, 20, 40, 80, 160, 320, 640, 1280):
            for unr in (1, 2, 4, 8):
                if grid * 256 * unr > 4 * n16 and grid > 8:
                    continue
                best = min(best, _bench(mod, bufs, bts, n16, grid, unr, out))
        floor[kib] = best
        del bufs, bts
        torch.cuda.empty_cache()
    return floor


def floor_us(floor, nbytes):
    """Log-log interpolation of the measured floor at nbytes."""
    pts = sorted(floor.items())
    xs = [math.log(k) for k, _ in pts]
    x = math.log(max(nbytes / 1024, pts[0][0]))
    i = min(max(bisect.bisect(xs, x), 1), len(xs) - 1)
    (_, t0), (_, t1) = pts[i - 1], pts[i]
    f = (x - xs[i - 1]) / (xs[i] - xs[i - 1])
    return math.exp(math.log(t0) + f * (math.log(t1) - math.log(t0)))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--hq", type=int, required=True)
    p.add_argument("--hkv", type=int, required=True)
    p.add_argument("--head-dim", type=int, required=True)
    p.add_argument("--window", type=int, default=0)
    p.add_argument("--m", type=int, nargs="+", default=[1, 4])
    p.add_argument("--contexts", type=int, nargs="+", default=CONTEXTS)
    p.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=[
            16,
            32,
            64,
            128,
            256,
            512,
            1024,
            2048,
            4096,
            8192,
            16384,
            65536,
            262144,
        ],
        help="stream sizes in KiB; they must bracket every configuration's KV",
    )
    args = p.parse_args()

    mod = _module()
    out = torch.zeros(16, dtype=torch.int32, device="cuda")
    tiny = [torch.zeros(16, dtype=torch.uint8, device="cuda") for _ in range(100)]
    bts = [torch.zeros(8, dtype=torch.int32, device="cuda") for _ in tiny]
    print(f"empty kernel: {_bench(mod, tiny, bts, 1, 1, -1, out):.2f} us")

    floor = measure(mod, args.sizes)
    print("\n| KiB | floor us | roofline us | %roof |")
    print("| --- | --- | --- | --- |")
    for kib, us in floor.items():
        roof = utils.DISPATCH_US + kib * 1024 / (utils.PEAK_GIBS * 1024**3) * 1e6
        print(f"| {kib} | {us:.2f} | {roof:.2f} | {roof / us * 100:.1f} % |")

    hq, hkv, d, w = args.hq, args.hkv, args.head_dim, args.window
    print(f"\n{hq}/{hkv}/{d}" + (f" w{w}" if w else "") + ": ceiling (%roof)")
    print("| M | " + " | ".join(str(s) for s in args.contexts) + " |")
    print("| --- " * (len(args.contexts) + 1) + "|")
    for m in args.m:
        r = []
        for s in args.contexts:
            roof = utils.roofline_us(hq, hkv, d, m, s, window=w)
            nbytes = (roof - utils.DISPATCH_US) * 1e-6 * utils.PEAK_GIBS * 1024**3
            r.append(roof / floor_us(floor, nbytes) * 100)
        print(f"| {m} | " + " | ".join(f"{x:.1f} %" for x in r) + " |")


if __name__ == "__main__":
    main()
