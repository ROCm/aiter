"""Per-kernel cost of back-to-back tiny kernels inside one CUDA graph (one GPU).

Each node is a 1-block kernel touching 4 KiB, so the replay time per node is the
launch / inter-kernel overhead of the graph, not the work.
"""

import os
import sys

import torch

N = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
ELEMS = int(sys.argv[2]) if len(sys.argv) > 2 else 1024  # floats each node writes
x = torch.zeros(ELEMS, device="cuda")


def body():
    for _ in range(N):
        x.add_(1.0)


for _ in range(3):
    body()
torch.cuda.synchronize()
g = torch.cuda.CUDAGraph()
s = torch.cuda.Stream()
with torch.cuda.stream(s):
    body()  # warm the stream
    torch.cuda.synchronize()
    with torch.cuda.graph(g, stream=s):
        body()
for _ in range(5):
    g.replay()
torch.cuda.synchronize()
times = []
for _ in range(10):
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record()
    g.replay()
    b.record()
    torch.cuda.synchronize()
    times.append(a.elapsed_time(b) * 1e3 / N)
times.sort()
env = {k: v for k, v in os.environ.items() if k.startswith(("DEBUG_CLR", "HIP_", "GPU_", "ROC_", "HSA_", "AMD_"))}
print(f"graph of {N} kernels writing {ELEMS * 4} B: {times[len(times) // 2]:.3f} us/kernel (min {times[0]:.3f})  env={env}")
