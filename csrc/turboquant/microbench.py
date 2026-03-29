"""TurboQuant unified microbenchmark: correctness + performance for all kernel variants.

Tests single-layer GEMM accuracy (CosSim vs dequant reference) as a PPL proxy,
and latency vs BF16 F.linear baseline. Covers decode (B=1) and prefill (B=32+).

Usage (inside Docker):
    python3 microbench.py                    # all tests
    python3 microbench.py --only-fused       # just the fused kernel
    python3 microbench.py --only-flatmm      # just the flatmm pipeline
    python3 microbench.py --quick            # small shapes only
"""
import os, sys, math, time, argparse
import numpy as np
import torch
import torch.nn.functional as F
import importlib.util

# Load turboquant engine (direct import, no sglang dependency)
_tq_engine_path = os.environ.get("TQ_ENGINE_PATH",
    os.path.join(os.path.dirname(os.path.abspath(__file__)),
                 "../../python/sglang/srt/layers/quantization/turboquant_engine.py"))
_spec = importlib.util.spec_from_file_location("tq_engine", _tq_engine_path)
_tq = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_tq)

EXT_DIR = os.path.dirname(os.path.abspath(__file__))
CK_BASE = os.environ.get("CK_PATH",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../3rdparty/composable_kernel"))
DEVICE = None

def get_device():
    global DEVICE
    if DEVICE is None:
        DEVICE = torch.device("cuda")
    return DEVICE

def timeit(fn, n=200, warmup=20):
    dev = get_device()
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n): fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n * 1000

# ---- Reference: full dequant + matmul in PyTorch (ground truth) ----
def dequant_ref_matmul(x, packed_idx, codebook, norms, gs):
    """Full dequant reference: codebook[idx] * norm / sqrt(gs), then matmul."""
    dev = x.device
    N, half_K = packed_idx.shape
    K = half_K * 2
    inv_s = 1.0 / math.sqrt(gs)
    p_np = packed_idx.cpu().numpy()
    cb_np = codebook.cpu().float().numpy()
    n_np = norms.cpu().float().numpy()
    W = np.zeros((N, K), dtype=np.float32)
    for k in range(K):
        bi = k // 2
        idx = (p_np[:, bi] & 0xF) if k % 2 == 0 else ((p_np[:, bi] >> 4) & 0xF)
        g = k // gs
        nv = n_np[:, g] if norms.ndim == 2 else n_np
        W[:, k] = cb_np[idx.astype(np.int64)] * nv * inv_s
    return x.float() @ torch.from_numpy(W).to(dev).T

# ---- Test helpers ----
def make_test_data(N, K, gs, seed=42):
    torch.manual_seed(seed)
    W = torch.randn(N, K, dtype=torch.float32) * 0.02
    tq = _tq.turboquant_quantize_packed(W, bit_width=4, group_size=gs, seed=seed)
    return tq

def cos_sim(a, b):
    return F.cosine_similarity(a.flatten().unsqueeze(0), b.flatten().unsqueeze(0)).item()

# ---- DeepSeek-V3 layer shapes ----
LAYERS = {
    "small":     (128,   256,  128),
    "kv_b_proj": (512,   24576, 128),
    "gate_up":   (7168,  18432, 128),
    "down_proj": (9216,  7168,  128),
}
QUICK_LAYERS = {k: v for k, v in LAYERS.items() if k in ("small",)}
BATCH_SIZES = [1, 4, 32]

# ==================================================================
# Kernel: Fused dequant-GEMM (turboquant_fused_gemm.cu)
# ==================================================================
def build_fused():
    from torch.utils.cpp_extension import load
    return load(name="tq_fused_mb",
        sources=[os.path.join(EXT_DIR, "turboquant_fused_gemm.cu")],
        extra_include_paths=[os.path.join(EXT_DIR, "include")],
        extra_cuda_cflags=["-O3", "--offload-arch=gfx950", "-DUSE_ROCM", "-std=c++17"],
        verbose=False)

def test_fused(ext, layers, batch_sizes):
    dev = get_device()
    print("\n" + "="*75)
    print("FUSED DEQUANT-GEMM (turboquant_fused_gemm.cu)")
    print("="*75)

    # Correctness
    print(f"\n{'Layer':<14} {'B':>4} {'CosSim':>10} {'Status':>8}")
    print("-" * 40)
    for name, (K, N, gs) in layers.items():
        if N > 4096: continue  # ref dequant too slow for large N
        tq = make_test_data(N, K, gs)
        idx = tq["indices_packed"].to(dev)
        idx_T = idx.T.contiguous()
        cb = tq["codebook"].float().to(dev)
        norms = tq["norms"].float().to(dev)
        for B in batch_sizes:
            x = torch.randn(B, K, dtype=torch.float32, device=dev) * 0.1
            y_kern = ext.turboquant_fused_gemm(x, idx, idx_T, cb, norms, gs)
            y_ref = dequant_ref_matmul(x, idx, cb, norms, gs)
            cs = cos_sim(y_ref, y_kern.float())
            s = "PASS" if cs > 0.99 else "FAIL"
            print(f"  {name:<12} {B:>4} {cs:>10.6f} {s:>8}")

    # Performance
    print(f"\n{'Layer':<14} {'B':>4} {'BF16 ms':>10} {'Fused ms':>10} {'Speedup':>10}")
    print("-" * 55)
    for name, (K, N, gs) in layers.items():
        tq = make_test_data(N, K, gs)
        idx = tq["indices_packed"].to(dev)
        idx_T = idx.T.contiguous()
        cb = tq["codebook"].float().to(dev)
        norms = tq["norms"].float().to(dev)
        W_bf16 = torch.randn(N, K, dtype=torch.bfloat16, device=dev) * 0.02
        for B in batch_sizes:
            x_bf16 = torch.randn(B, K, dtype=torch.bfloat16, device=dev)
            x_f32 = x_bf16.float()
            t_bf16 = timeit(lambda: F.linear(x_bf16, W_bf16))
            try:
                t_fused = timeit(lambda: ext.turboquant_fused_gemm(x_f32, idx, idx_T, cb, norms, gs))
                print(f"  {name:<12} {B:>4} {t_bf16:>10.4f} {t_fused:>10.4f} {t_bf16/t_fused:>9.2f}x")
            except Exception as e:
                print(f"  {name:<12} {B:>4} {t_bf16:>10.4f} {'ERROR':>10}")

# ==================================================================
# Kernel: INT4 FlatMM pipeline (turboquant_int4_flatmm.hip)
# ==================================================================
def build_flatmm():
    from torch.utils.cpp_extension import load
    return load(name="tq_flatmm_mb",
        sources=[os.path.join(EXT_DIR, "turboquant_int4_flatmm.hip")],
        extra_include_paths=[EXT_DIR, CK_BASE + "/include"],
        extra_cuda_cflags=["-O3", "--offload-arch=gfx950", "-DUSE_ROCM", "-std=c++17"],
        verbose=False)

def test_flatmm(ext, layers, batch_sizes):
    dev = get_device()
    print("\n" + "="*75)
    print("INT4 FLATMM PIPELINE (turboquant_int4_flatmm.hip)")
    print("  NOTE: data NOT preshuffled -> correctness expected to fail")
    print("  This tests compilation + crash-freedom + raw throughput")
    print("="*75)

    # Correctness (expected to fail without preshuffle — just verify no crash)
    print(f"\n{'Layer':<14} {'B':>4} {'CosSim':>10} {'Status':>8}")
    print("-" * 40)
    for name, (K, N, gs) in layers.items():
        if N > 4096: continue
        tq = make_test_data(N, K, gs)
        idx = tq["indices_packed"].to(dev)
        norms = tq["norms"].float().to(dev)
        cb = tq["codebook"].float().to(dev)
        for B in batch_sizes:
            x = torch.randn(B, K, dtype=torch.bfloat16, device=dev)
            try:
                y_kern = ext.turboquant_int4_flatmm_gemm(x, idx, gs)
                y_ref = dequant_ref_matmul(x, idx, cb, norms, gs)
                cs = cos_sim(y_ref, y_kern.float())
                s = "PASS" if cs > 0.99 else ("NoPreSh" if cs < 0.1 else "LOW")
                print(f"  {name:<12} {B:>4} {cs:>10.6f} {s:>8}")
            except Exception as e:
                print(f"  {name:<12} {B:>4} {'crash':>10} {'CRASH':>8}")

    # Performance
    print(f"\n{'Layer':<14} {'B':>4} {'BF16 ms':>10} {'FlatMM ms':>10} {'Speedup':>10}")
    print("-" * 55)
    for name, (K, N, gs) in layers.items():
        tq = make_test_data(N, K, gs)
        idx = tq["indices_packed"].to(dev)
        W_bf16 = torch.randn(N, K, dtype=torch.bfloat16, device=dev) * 0.02
        for B in batch_sizes:
            x = torch.randn(B, K, dtype=torch.bfloat16, device=dev)
            t_bf16 = timeit(lambda: F.linear(x, W_bf16))
            try:
                t_fm = timeit(lambda: ext.turboquant_int4_flatmm_gemm(x, idx, gs))
                print(f"  {name:<12} {B:>4} {t_bf16:>10.4f} {t_fm:>10.4f} {t_bf16/t_fm:>9.2f}x")
            except Exception as e:
                print(f"  {name:<12} {B:>4} {t_bf16:>10.4f} {'CRASH':>10}")

# ==================================================================
# Single-layer PPL proxy: quantize-dequant round-trip SNR
# ==================================================================
def test_quant_quality(layers):
    print("\n" + "="*75)
    print("QUANTIZATION QUALITY (PPL proxy: SNR + CosSim on weight reconstruction)")
    print("="*75)
    print(f"\n{'Layer':<14} {'K':>6} {'N':>6} {'SNR dB':>10} {'CosSim':>10} {'RelErr%':>10}")
    print("-" * 60)
    for name, (K, N, gs) in layers.items():
        torch.manual_seed(42)
        W = torch.randn(N, K, dtype=torch.float32) * 0.02
        tq = _tq.turboquant_quantize_packed(W, bit_width=4, group_size=gs, seed=42)
        W_deq = _tq.turboquant_dequantize(tq, device="cpu")
        mse = ((W - W_deq) ** 2).mean().item()
        snr = 10 * math.log10(W.var().item() / (mse + 1e-30))
        cs = cos_sim(W, W_deq)
        rel_err = 100 * math.sqrt(mse) / (W.abs().mean().item() + 1e-30)
        print(f"  {name:<12} {K:>6} {N:>6} {snr:>10.2f} {cs:>10.6f} {rel_err:>9.2f}%")

# ==================================================================
# Main
# ==================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only-fused", action="store_true")
    parser.add_argument("--only-flatmm", action="store_true")
    parser.add_argument("--only-quality", action="store_true")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    layers = QUICK_LAYERS if args.quick else LAYERS

    print("TurboQuant Microbenchmark")
    print(f"Device: {torch.cuda.get_device_name(0)}")

    # Quality test (fast, no kernel build needed)
    if not args.only_fused and not args.only_flatmm:
        test_quant_quality(layers)

    # Fused kernel
    if not args.only_flatmm and not args.only_quality:
        print("\nBuilding fused kernel...")
        try:
            ext_fused = build_fused()
            print("Build OK")
            test_fused(ext_fused, layers, BATCH_SIZES)
        except Exception as e:
            print(f"Fused build FAILED: {e}")

    # FlatMM pipeline
    if not args.only_fused and not args.only_quality:
        print("\nBuilding flatmm kernel...")
        try:
            ext_flatmm = build_flatmm()
            print("Build OK")
            test_flatmm(ext_flatmm, layers, BATCH_SIZES)
        except Exception as e:
            print(f"FlatMM build FAILED: {e}")

    print("\n" + "="*75)
    print("DONE")
    print("="*75)

if __name__ == "__main__":
    main()
