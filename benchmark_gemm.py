import torch, csv, sys
from aiter import dtypes, pertoken_quant, get_triton_quant, QuantType, gemm_a8w8_CK, gemm_a4w4
from aiter.ops.gemm_op_a8w8 import gemm_a8w8_blockscale
from aiter.ops.gemm_op_a4w4 import gemm_a4w4_blockscale
from aiter.ops.shuffle import shuffle_weight
from aiter.jit.utils.chip_info import get_gfx

torch.set_default_device("cuda")

def bench(func, *args, **kw):
    for _ in range(10): func(*args, **kw)
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    times = []
    for _ in range(50):
        s.record(); func(*args, **kw); e.record(); torch.cuda.synchronize()
        times.append(s.elapsed_time(e))
    return sorted(times)[25] * 1000

def bench_a8w8(M, N, K, qt):
    try:
        x, w = torch.randn((M,K), dtype=dtypes.bf16), torch.randn((N,K), dtype=dtypes.bf16)
        qd = dtypes.i8 if qt == "i8" else dtypes.fp8
        xq, xs = pertoken_quant(x, quant_dtype=qd)
        wq, ws = pertoken_quant(w, quant_dtype=qd)
        t = bench(gemm_a8w8_CK, xq, wq, xs, ws, torch.zeros([1,N], dtype=dtypes.bf16), dtypes.bf16)
        f = 2*M*N*K
        return {"M":M, "N":N, "K":K, "dtype":f"a8w8_{qt}", "backend":"CK",
                "time_us":round(t,2), "TFLOPS":round(f/(t*1e6),3), "GB/s":round((M*K+N*K+M*N*2)/(t*1e3),2)}
    except: return {"M":M, "N":N, "K":K, "dtype":f"a8w8_{qt}", "backend":"CK", "time_us":None, "TFLOPS":None, "GB/s":None}

def bench_a4w4(M, N, K):
    try:
        qf = get_triton_quant(QuantType.per_1x32)
        x, w = torch.randn((M,K), dtype=dtypes.bf16), torch.randn((N,K), dtype=dtypes.bf16)
        xq, xs = qf(x, shuffle=True)
        wq, ws = qf(w, shuffle=True)
        wq = shuffle_weight(wq, layout=(16,16))
        out = torch.empty(((M+31)//32)*32, N, dtype=dtypes.bf16)
        t = bench(gemm_a4w4, xq, wq, xs.view(torch.uint8), ws.view(torch.uint8), out, bpreshuffle=True)
        f = 2*M*N*K
        return {"M":M, "N":N, "K":K, "dtype":"a4w4", "backend":"ASM",
                "time_us":round(t,2), "TFLOPS":round(f/(t*1e6),3), "GB/s":round((M*K//2+N*K//2+M*N*2)/(t*1e3),2)}
    except: return {"M":M, "N":N, "K":K, "dtype":"a4w4", "backend":"ASM", "time_us":None, "TFLOPS":None, "GB/s":None}

def bench_a8w8_blockscale(M, N, K, qt):
    try:
        BLK = 128
        x, w = torch.randn((M,K), dtype=dtypes.bf16), torch.randn((N,K), dtype=dtypes.bf16)
        qd = dtypes.i8 if qt == "i8" else dtypes.fp8
        xq, xs = x.to(qd), torch.rand([M, (K+BLK-1)//BLK], dtype=dtypes.fp32)
        wq, ws = w.to(qd), torch.rand([(N+BLK-1)//BLK, (K+BLK-1)//BLK], dtype=dtypes.fp32)
        t = bench(gemm_a8w8_blockscale, xq, wq, xs, ws, dtypes.bf16)
        f = 2*M*N*K
        return {"M":M, "N":N, "K":K, "dtype":f"a8w8blk_{qt}", "backend":"CK",
                "time_us":round(t,2), "TFLOPS":round(f/(t*1e6),3), "GB/s":round((M*K+N*K+M*N*2)/(t*1e3),2)}
    except: return {"M":M, "N":N, "K":K, "dtype":f"a8w8blk_{qt}", "backend":"CK", "time_us":None, "TFLOPS":None, "GB/s":None}

def bench_a4w4_blockscale(M, N, K):
    try:
        qf = get_triton_quant(QuantType.per_1x32)
        x, w = torch.randn((M,K), dtype=dtypes.bf16), torch.randn((N,K), dtype=dtypes.bf16)
        xq, xs = qf(x, shuffle=True)
        wq, ws = qf(w, shuffle=True)
        out = torch.empty(((M+31)//32)*32, N, dtype=dtypes.bf16)
        t = bench(gemm_a4w4_blockscale, xq, wq, xs.view(torch.uint8), ws.view(torch.uint8), out, splitK=0)
        f = 2*M*N*K
        return {"M":M, "N":N, "K":K, "dtype":"a4w4blk", "backend":"CK",
                "time_us":round(t,2), "TFLOPS":round(f/(t*1e6),3), "GB/s":round((M*K//2+N*K//2+M*N*2)/(t*1e3),2)}
    except: return {"M":M, "N":N, "K":K, "dtype":"a4w4blk", "backend":"CK", "time_us":None, "TFLOPS":None, "GB/s":None}

SHAPES = [(1,1280,8192),(32,1280,8192),(64,1280,8192),(128,1280,8192),(256,1280,8192),
          (512,1280,8192),(1024,1280,8192),(2048,1280,8192),(4096,1280,8192),(8192,1280,8192),
          (1,8192,1024),(32,8192,1024),(64,8192,1024),(128,8192,1024),(256,8192,1024),
          (512,8192,1024),(1024,8192,1024),(2048,8192,1024),(4096,8192,1024),(8192,8192,1024),
          (2048,8192,8192)]

if __name__ == "__main__":
    print(f"GFX:{get_gfx()}, Device:{torch.cuda.get_device_name()}\n")
    results = []
    
    for dt, fn in [("A4W4", lambda m,n,k: bench_a4w4(m,n,k)),
                   ("A4W4-BLK", lambda m,n,k: bench_a4w4_blockscale(m,n,k)),
                   ("A8W8-INT8", lambda m,n,k: bench_a8w8(m,n,k,"i8")),
                   ("A8W8-FP8", lambda m,n,k: bench_a8w8(m,n,k,"fp8")),
                   ("A8W8BLK-INT8", lambda m,n,k: bench_a8w8_blockscale(m,n,k,"i8")),
                   ("A8W8BLK-FP8", lambda m,n,k: bench_a8w8_blockscale(m,n,k,"fp8"))]:
        print(f"{dt}:")
        for i, (M,N,K) in enumerate(SHAPES, 1):
            r = fn(M,N,K)
            results.append(r)
            print(f"  [{i}/{len(SHAPES)}] {M}x{N}x{K}: {r['TFLOPS']} TFLOPS" if r['TFLOPS'] else f"  [{i}/{len(SHAPES)}] SKIP")
    
    with open("gemm_benchmark_results.csv", "w") as f:
        w = csv.DictWriter(f, ["M","N","K","dtype","backend","time_us","TFLOPS","GB/s"])
        w.writeheader()
        w.writerows([r for r in results if r['TFLOPS']])
    
    print("\n" + "="*60)
    for dt in ["a4w4", "a4w4blk", "a8w8_i8", "a8w8_fp8", "a8w8blk_i8", "a8w8blk_fp8"]:
        v = [r['TFLOPS'] for r in results if r['dtype']==dt and r['TFLOPS']]
        if v: print(f"{dt:13s}: Avg={sum(v)/len(v):7.1f}, Max={max(v):7.1f}, Min={min(v):7.1f} TFLOPS")
    print("\nResults: gemm_benchmark_results.csv")
