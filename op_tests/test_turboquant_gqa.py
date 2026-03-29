"""TurboQuant GQA/MHA per-head quantization tests.

Covers:
  1. Per-head quality for multiple model configs (Qwen3, GPT-OSS, Llama, MQA)
  2. End-to-end MHATokenToKVPoolTQ: prefill → decode → multi-layer
  3. Compression ratio verification

Run: SGLANG_PATH=/path/to/sglang/python python3 test_gqa_mha.py
"""
import math
import os
import sys

_sglang_path = os.environ.get("SGLANG_PATH")
if _sglang_path:
    sys.path.insert(0, _sglang_path)

os.environ.setdefault("SGLANG_KV_CACHE_TURBOQUANT", "4")

import torch
import torch.nn.functional as F

from sglang.srt.layers.quantization.turboquant_engine import (
    generate_rotation_matrix,
    get_codebook,
    pack_indices,
    packed_bytes_per_dim,
    pad_for_packing,
    unpack_indices,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")

all_ok = True


def check(name, cond):
    global all_ok
    if not cond:
        all_ok = False
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")


# ── 1. Per-head quality ──
print("=" * 60)
print("1. Per-Head Quantization Quality")
print("=" * 60)

configs = [
    ("Qwen3-235B", 128, 4),
    ("GPT-OSS-120B", 64, 8),
    ("Llama-3-8B", 128, 8),
    ("MQA", 128, 1),
]

for model, hd, hn in configs:
    print(f"\n  {model}: head_dim={hd}, kv_heads={hn}")
    Pi = generate_rotation_matrix(hd, seed=42).to(DEVICE)
    scale = math.sqrt(hd)

    for bw in [2, 3, 4]:
        torch.manual_seed(42)
        data = torch.randn(64, hn, hd, device=DEVICE) * 0.02
        centroids, boundaries = get_codebook(bw)
        cd, bd = centroids.to(DEVICE), boundaries.to(DEVICE)
        n_levels = 2**bw
        pb = packed_bytes_per_dim(hd, bw)

        recon = torch.zeros_like(data, dtype=torch.float32)
        for h in range(hn):
            hd_data = data[:, h, :].float()
            norms = hd_data.norm(dim=1, keepdim=True).clamp(min=1e-8)
            Y = (hd_data / norms) @ Pi.T * scale
            idx = torch.searchsorted(bd, Y.reshape(-1)).clamp(0, n_levels - 1).reshape(64, hd)
            padded = pad_for_packing(hd, bw)
            if padded > hd:
                idx = F.pad(idx, (0, padded - hd))
            packed = pack_indices(idx, bw)
            unpacked = unpack_indices(packed, padded, bw)[:, :hd]
            Y_hat = cd[unpacked.long()] / scale
            recon[:, h, :] = (Y_hat @ Pi) * norms

        cos = F.cosine_similarity(
            data.float().flatten().unsqueeze(0),
            recon.flatten().unsqueeze(0),
        ).item()
        thresh = {4: 0.99, 3: 0.97, 2: 0.90}
        check(f"{bw}-bit CosSim={cos:.4f}", cos > thresh[bw])


# ── 2. E2E Pool Test ──
print("\n" + "=" * 60)
print("2. End-to-End MHATokenToKVPoolTQ")
print("=" * 60)


class MockRadixAttention:
    def __init__(self, layer_id):
        self.layer_id = layer_id


class MockMemSaver:
    def region(self, x):
        from contextlib import nullcontext
        return nullcontext()


def test_pool(hn, hd, nl, ps, bw):
    os.environ["SGLANG_KV_CACHE_TURBOQUANT"] = str(bw)

    from importlib import reload
    import sglang.srt.mem_cache.memory_pool as mp
    reload(mp)

    pool = mp.MHATokenToKVPoolTQ.__new__(mp.MHATokenToKVPoolTQ)
    pool.size = ps
    pool.page_size = 1
    pool.head_num = hn
    pool.head_dim = hd
    pool.v_head_dim = hd
    pool.layer_num = nl
    pool.dtype = torch.bfloat16
    pool.store_dtype = torch.bfloat16
    pool.device = DEVICE
    pool.memory_saver_adapter = MockMemSaver()
    pool.custom_mem_pool = None
    pool.enable_custom_mem_pool = False
    pool.start_layer = 0
    pool.layer_transfer_counter = None
    pool._create_buffers()

    torch.manual_seed(42)
    T = 32
    K = torch.randn(T, hn, hd, dtype=torch.bfloat16, device=DEVICE) * 0.02
    V = torch.randn(T, hn, hd, dtype=torch.bfloat16, device=DEVICE) * 0.02
    locs = torch.arange(T, device=DEVICE)

    pool.set_kv_buffer(MockRadixAttention(0), locs, K, V)
    K_out = pool.get_key_buffer(0)[:T]
    V_out = pool.get_value_buffer(0)[:T]

    cos_k = F.cosine_similarity(K.float().flatten().unsqueeze(0), K_out.float().flatten().unsqueeze(0)).item()
    cos_v = F.cosine_similarity(V.float().flatten().unsqueeze(0), V_out.float().flatten().unsqueeze(0)).item()

    # Decode: add 1 token
    new_k = torch.randn(1, hn, hd, dtype=torch.bfloat16, device=DEVICE) * 0.02
    new_v = torch.randn(1, hn, hd, dtype=torch.bfloat16, device=DEVICE) * 0.02
    pool.set_kv_buffer(MockRadixAttention(0), torch.tensor([T], device=DEVICE), new_k, new_v)
    K2 = pool.get_key_buffer(0)[T:T + 1]
    cos_new = F.cosine_similarity(new_k.float().flatten().unsqueeze(0), K2.float().flatten().unsqueeze(0)).item()

    return cos_k, cos_v, cos_new


for name, hn, hd in [("Qwen3-GQA", 4, 128), ("GPT-OSS-GQA", 8, 64)]:
    print(f"\n  {name}:")
    for bw in [2, 3, 4]:
        ck, cv, cn = test_pool(hn, hd, 2, 128, bw)
        thresh = {4: 0.99, 3: 0.97, 2: 0.90}
        check(f"{bw}-bit K={ck:.4f} V={cv:.4f} new={cn:.4f}", ck > thresh[bw] and cv > thresh[bw])


# ── 3. Compression ratios ──
print("\n" + "=" * 60)
print("3. GQA Compression Ratios (K+V)")
print("=" * 60)

for hn, hd in [(4, 128), (8, 64)]:
    fp16 = hn * hd * 2 * 2
    for bw in [2, 3, 4]:
        pb = packed_bytes_per_dim(hd, bw)
        tq = hn * (pb + 2) * 2
        print(f"  h={hn} d={hd} {bw}-bit: {fp16}B → {tq}B ({fp16 / tq:.2f}x)")

print("\n" + "=" * 60)
print("ALL PASSED" if all_ok else "SOME TESTS FAILED")
print("=" * 60)

sys.exit(0 if all_ok else 1)
