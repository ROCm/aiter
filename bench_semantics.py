import sys, torch
sys.path.insert(0, "op_tests")
import aiter
from aiter.test_common import make_generator, run_perftest
from test_mxfp8fp4gemm import _prep

M, N, K, apre = 512, 65536, 1536, 1
INTYPE = sys.argv[1] if len(sys.argv) > 1 else "a8w8"
ITERS, WARMUP, N_ROT = 400, 50, 8

gen = make_generator(0)
inp, _ = _prep(INTYPE, M, N, K, apre, "constant", "constant", gen)  # neutral scale, low-power data
kern = aiter.gemm_a8w8_mxfp8 if INTYPE == "a8w8" else aiter.gemm_a8w4_mxfp8


def run(A, B, sA, sB):
    return kern(A, B, sA, sB, dtype=torch.bfloat16, a_preshuffle=bool(apre), kernelName="")


# rotate copies to defeat L2 (like poc's N_BUF rotation)
rot = [(inp["A"], inp["B"], inp["sA"], inp["sB"])]
for _ in range(N_ROT - 1):
    rot.append(tuple(t.clone() for t in rot[0]))

for _ in range(WARMUP):
    run(*rot[0])
torch.cuda.synchronize()

# (A) aiter official: profiler per-kernel avg
_, us_aiter = run_perftest(run, *rot[0], num_iters=ITERS, num_warmup=WARMUP)

# (B) poc semantics: single event pair around N back-to-back launches / N, rotated
s = torch.cuda.Event(enable_timing=True)
e = torch.cuda.Event(enable_timing=True)
torch.cuda.synchronize()
s.record()
for i in range(ITERS):
    run(*rot[i % N_ROT])
e.record()
torch.cuda.synchronize()
us_throughput = s.elapsed_time(e) * 1000.0 / ITERS

# (C) single-shot latency: per-iter event+sync, median
lat = []
for i in range(ITERS):
    s2 = torch.cuda.Event(enable_timing=True)
    e2 = torch.cuda.Event(enable_timing=True)
    s2.record()
    run(*rot[i % N_ROT])
    e2.record()
    e2.synchronize()
    lat.append(s2.elapsed_time(e2) * 1000.0)
lat.sort()
us_latency = lat[len(lat) // 2]

# (D) CUDA graph back-to-back wall/N: capture N_ROT launches, replay -> no python/launch gaps
us_graph = float("nan")
try:
    g = torch.cuda.CUDAGraph()
    pool = torch.cuda.graph_pool_handle()
    # warm the capture stream
    sc = torch.cuda.Stream()
    sc.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(sc):
        for i in range(3):
            run(*rot[i % N_ROT])
    torch.cuda.current_stream().wait_stream(sc)
    with torch.cuda.graph(g, pool=pool):
        for i in range(N_ROT):
            run(*rot[i])
    REPLAYS = ITERS // N_ROT
    for _ in range(5):
        g.replay()
    torch.cuda.synchronize()
    s3 = torch.cuda.Event(enable_timing=True); e3 = torch.cuda.Event(enable_timing=True)
    s3.record()
    for _ in range(REPLAYS):
        g.replay()
    e3.record(); torch.cuda.synchronize()
    us_graph = s3.elapsed_time(e3) * 1000.0 / (REPLAYS * N_ROT)
except Exception as ex:
    print("graph capture failed:", ex)

print(f"\n===== {INTYPE} {M}x{N}x{K} constant/constant iters={ITERS} warmup={WARMUP} rot={N_ROT} =====")
print(f"(A) aiter profiler per-kernel avg   : {us_aiter:.2f} us")
print(f"(B) back-to-back wall/N (poc-style) : {us_throughput:.2f} us")
print(f"(C) per-iter sync, median (latency) : {us_latency:.2f} us")
print(f"(D) CUDA-graph wall/N (no host gaps): {us_graph:.2f} us")
print(f"ratio A/B = {us_aiter/us_throughput:.3f}")
