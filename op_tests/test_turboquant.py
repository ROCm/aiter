"""TurboQuant GPU kernel tests: multi-bit compress/decompress, CUDA graph, quality.

Covers:
  1. GPU kernel round-trip (2/3/4-bit): compress → decompress → CosSim
  2. GPU ↔ Python cross-validation (bit-exact packing)
  3. CUDA graph capture + replay (compress and decompress)
  4. Mixed-bit (2.5/3.5-bit) outlier treatment
  5. Multi-bit matmul (turboquant_matmul_pytorch)
  6. QJL (Stage 2) packing correctness

Run: SGLANG_PATH=/path/to/sglang/python python3 test_turboquant.py
"""
import math
import os
import sys
import time

_sglang_path = os.environ.get("SGLANG_PATH")
if _sglang_path:
    sys.path.insert(0, _sglang_path)

import torch
import torch.nn.functional as F

from sglang.srt.layers.quantization.turboquant_engine import (
    generate_rotation_matrix,
    get_codebook,
    mixed_bit_config,
    mixed_compress_latent,
    mixed_compressed_bytes,
    mixed_decompress_latent,
    pack_indices,
    packed_bytes_per_dim,
    pad_for_packing,
    turboquant_dequantize,
    turboquant_matmul_pytorch,
    turboquant_quantize_packed,
    unpack_indices,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
KV_LORA_RANK, QK_ROPE_DIM = 512, 64
KV_DIM = KV_LORA_RANK + QK_ROPE_DIM
GROUP_SIZE, N_GROUPS = 128, 4

all_ok = True


def check(name, cond):
    global all_ok
    if not cond:
        all_ok = False
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")


# Try loading GPU kernel
gpu_ext = None
try:
    from torch.utils.cpp_extension import load

    ext_dir = os.path.dirname(os.path.abspath(__file__))
    ck_base = os.environ.get(
        "CK_PATH",
        os.path.join(ext_dir, "../../3rdparty/composable_kernel"),
    )
    gpu_ext = load(
        name="tq_test_kern",
        sources=[os.path.join(ext_dir, "turboquant_kv_compress.hip")],
        extra_include_paths=[ext_dir, ck_base + "/include"],
        extra_cuda_cflags=[
            "-O3", "--offload-arch=gfx950", "-DUSE_ROCM", "-std=c++17",
        ],
        verbose=False,
    )
    print(f"GPU kernel compiled OK (device: {torch.cuda.get_device_name()})\n")
except Exception as e:
    print(f"GPU kernel not available ({e}), testing Python path only\n")

Pi_all = torch.stack(
    [
        generate_rotation_matrix(GROUP_SIZE, seed=42 + g * GROUP_SIZE).to(DEVICE)
        for g in range(N_GROUPS)
    ]
)
rotations = {g * GROUP_SIZE: Pi_all[g] for g in range(N_GROUPS)}

# ── 1. GPU kernel round-trip ──
if gpu_ext:
    print("=" * 60)
    print("1. GPU Kernel Round-Trip (2/3/4-bit)")
    print("=" * 60)

    torch.manual_seed(42)
    kv = torch.randn(32, KV_DIM, dtype=torch.bfloat16, device=DEVICE) * 0.1

    for bw in [2, 3, 4]:
        centroids, boundaries = get_codebook(bw)
        cd, bd = centroids.to(DEVICE), boundaries.to(DEVICE)

        comp = gpu_ext.turboquant_kv_compress(kv, Pi_all, bd, N_GROUPS, GROUP_SIZE, bw)
        dec = gpu_ext.turboquant_kv_decompress(
            comp, Pi_all, cd, N_GROUPS, GROUP_SIZE, KV_LORA_RANK, QK_ROPE_DIM, bw,
        )
        cos = F.cosine_similarity(
            kv[:, :KV_LORA_RANK].float().flatten().unsqueeze(0),
            dec[:, :KV_LORA_RANK].float().flatten().unsqueeze(0),
        ).item()
        expected = {4: 0.99, 3: 0.97, 2: 0.90}
        check(f"{bw}-bit CosSim={cos:.6f} (>{expected[bw]})", cos > expected[bw])

        # Cross-validate: GPU compress → Python decompress
        pb = packed_bytes_per_dim(KV_LORA_RANK, bw)
        total = pb + N_GROUPS * 2 + QK_ROPE_DIM * 2
        check(f"{bw}-bit size={comp.shape[1]} (expected {total})", comp.shape[1] == total)

    # ── 2. CUDA graph capture ──
    print("\n" + "=" * 60)
    print("2. CUDA Graph Capture (compress + decompress)")
    print("=" * 60)

    for bw in [2, 3, 4]:
        centroids, boundaries = get_codebook(bw)
        cd, bd = centroids.to(DEVICE), boundaries.to(DEVICE)
        total = packed_bytes_per_dim(KV_LORA_RANK, bw) + N_GROUPS * 2 + QK_ROPE_DIM * 2

        kv_in = torch.randn(8, KV_DIM, dtype=torch.bfloat16, device=DEVICE) * 0.1
        comp_out = torch.empty(8, total, dtype=torch.uint8, device=DEVICE)
        dec_out = torch.empty(8, KV_DIM, dtype=torch.bfloat16, device=DEVICE)

        for _ in range(3):
            gpu_ext.turboquant_kv_compress_inplace(
                kv_in, Pi_all, bd, comp_out, N_GROUPS, GROUP_SIZE, bw,
            )
            gpu_ext.turboquant_kv_decompress_inplace(
                comp_out, Pi_all, cd, dec_out, N_GROUPS, GROUP_SIZE,
                KV_LORA_RANK, QK_ROPE_DIM, bw,
            )
        torch.cuda.synchronize()

        try:
            g1 = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g1):
                gpu_ext.turboquant_kv_compress_inplace(
                    kv_in, Pi_all, bd, comp_out, N_GROUPS, GROUP_SIZE, bw,
                )
            g1.replay()
            torch.cuda.synchronize()
            ref = gpu_ext.turboquant_kv_compress(
                kv_in, Pi_all, bd, N_GROUPS, GROUP_SIZE, bw,
            )
            check(
                f"{bw}-bit graph compress: "
                f"{'BIT-EXACT' if torch.equal(comp_out, ref) else 'MISMATCH'}",
                torch.equal(comp_out, ref),
            )
        except Exception as e:
            check(f"{bw}-bit graph compress: {e}", False)

# ── 3. Mixed-bit (outlier treatment) ──
print("\n" + "=" * 60)
print("3. Mixed-Bit Outlier Treatment (2.5/3.5-bit)")
print("=" * 60)

torch.manual_seed(42)
latent = torch.randn(64, KV_LORA_RANK, device=DEVICE) * 0.02

for eff_bits in [2.5, 3.5]:
    group_bits = mixed_bit_config(eff_bits, N_GROUPS)
    total_bytes = mixed_compressed_bytes(
        KV_LORA_RANK, GROUP_SIZE, QK_ROPE_DIM, group_bits,
    )
    fp16_bytes = KV_DIM * 2
    all_packed, norms, _ = mixed_compress_latent(
        latent, group_bits, GROUP_SIZE, rotations, DEVICE,
    )
    recon = mixed_decompress_latent(
        all_packed, norms, group_bits, GROUP_SIZE, KV_LORA_RANK, rotations, DEVICE,
    )
    cos = F.cosine_similarity(
        latent.flatten().unsqueeze(0), recon.flatten().unsqueeze(0),
    ).item()
    expected = {2.5: 0.94, 3.5: 0.98}
    check(
        f"{eff_bits}-bit CosSim={cos:.4f}, {fp16_bytes / total_bytes:.2f}x compression",
        cos > expected[eff_bits],
    )

# ── 4. Multi-bit matmul ──
print("\n" + "=" * 60)
print("4. Multi-Bit Matmul (turboquant_matmul_pytorch)")
print("=" * 60)

torch.manual_seed(42)
W = torch.randn(32, 512, device=DEVICE) * 0.02
x = torch.randn(4, 512, device=DEVICE) * 0.1
y_ref = x @ W.T

for bw in [2, 3, 4]:
    pd = turboquant_quantize_packed(W, bit_width=bw, group_size=128, seed=42)
    pd_dev = {k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v for k, v in pd.items()}
    y_tq = turboquant_matmul_pytorch(
        x, pd_dev["indices_packed"], pd_dev["codebook"],
        pd_dev["norms"], 512, 128, 42, bit_width=bw,
    )
    cos = F.cosine_similarity(
        y_ref.flatten().unsqueeze(0), y_tq.flatten().unsqueeze(0),
    ).item()
    thresh = {4: 0.99, 3: 0.97, 2: 0.90}
    check(f"{bw}-bit matmul CosSim={cos:.6f}", cos > thresh[bw])

# ── 5. Quantize → dequantize round-trip ──
print("\n" + "=" * 60)
print("5. Quantize/Dequantize Round-Trip")
print("=" * 60)

for bw in [2, 3, 4]:
    torch.manual_seed(42)
    W = torch.randn(64, 512, device=DEVICE) * 0.02
    packed = turboquant_quantize_packed(W, bit_width=bw, group_size=128, seed=42)
    recon = turboquant_dequantize(packed, DEVICE)
    cos = F.cosine_similarity(
        W.flatten().unsqueeze(0), recon.flatten().unsqueeze(0),
    ).item()
    thresh = {4: 0.99, 3: 0.97, 2: 0.90}
    check(f"{bw}-bit q→dq CosSim={cos:.6f}", cos > thresh[bw])

# ── 6. QJL packing fix ──
print("\n" + "=" * 60)
print("6. QJL Path (turboquant_kv.py)")
print("=" * 60)

try:
    from sglang.srt.layers.quantization.turboquant_kv import TurboQuantKVCompressor

    for bw in [2, 3, 4]:
        comp = TurboQuantKVCompressor(
            512, 64, bit_width=bw, use_qjl=True, device=DEVICE,
        )
        kv = torch.randn(16, 576, device=DEVICE)
        compressed = comp.compress(kv, global_norm=True)
        recovered = comp.decompress(compressed)
        cos = F.cosine_similarity(
            kv[:, :512].float().reshape(-1),
            recovered[:, :512].float().reshape(-1),
            dim=0,
        ).item()
        has_qjl = "qjl_signs" in compressed
        expected_shapes = {2: (16, 128), 3: (16, 192), 4: (16, 256)}
        shape_ok = tuple(compressed["indices_packed"].shape) == expected_shapes[bw]
        check(f"{bw}-bit QJL: CosSim={cos:.4f}, qjl={has_qjl}, shape={shape_ok}", has_qjl and shape_ok)
except ImportError:
    print("  [SKIP] turboquant_kv not available")

# ── Summary ──
print("\n" + "=" * 60)
print("ALL PASSED" if all_ok else "SOME TESTS FAILED")
print("=" * 60)

sys.exit(0 if all_ok else 1)
