"""
Benchmark the 4 forward kernel stages to trace performance evolution.

Stage 1: tl.trans V-reuse, hardcoded BH=16 TK=32 w=4 s=1  (first version)
Stage 2: Separate K/V loads, hardcoded BH=16 TK=16 w=4 s=1  (removed tl.trans)
Stage 3: Separate K/V loads, BH=32 TK=16 w=4 s=2            (larger head tile)
Stage 4: tl.trans + autotune                                  (current version)

Each stage is tested with a fixed config (no autotune) except stage 4.
"""
import torch
import triton
import triton.language as tl
import argparse
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))


# ============================================================================
# Stage 1 & 4 kernel: tl.trans V-reuse
# ============================================================================
@triton.jit
def _fwd_kernel_trans(
    Q_ptr, KV_ptr, TopK_ptr, O_ptr, LSE_ptr,
    stride_q_t: tl.int64, stride_q_h: tl.int64,
    stride_kv_t: tl.int64,
    stride_o_t: tl.int64, stride_o_h: tl.int64,
    stride_topk_t: tl.int64,
    scale: tl.float32, num_heads: tl.int32,
    TOPK: tl.constexpr, BLOCK_H: tl.constexpr, TILE_K: tl.constexpr,
    D_V: tl.constexpr, D_ROPE: tl.constexpr,
):
    token_idx = tl.program_id(0)
    hg_idx = tl.program_id(1)
    offs_h = hg_idx * BLOCK_H + tl.arange(0, BLOCK_H)
    mask_h = offs_h < num_heads
    offs_v = tl.arange(0, D_V)
    offs_r = tl.arange(0, D_ROPE)

    q_base = token_idx * stride_q_t
    Q_lora = tl.load(Q_ptr + q_base + offs_h[:, None] * stride_q_h + offs_v[None, :],
                      mask=mask_h[:, None], other=0.0)
    Q_rope = tl.load(Q_ptr + q_base + offs_h[:, None] * stride_q_h + (D_V + offs_r[None, :]),
                      mask=mask_h[:, None], other=0.0)

    m_i = tl.full([BLOCK_H], float("-inf"), dtype=tl.float32)
    l_i = tl.full([BLOCK_H], 0.0, dtype=tl.float32)
    acc = tl.zeros([BLOCK_H, D_V], dtype=tl.float32)

    NUM_TILES: tl.constexpr = (TOPK + TILE_K - 1) // TILE_K
    topk_base = token_idx * stride_topk_t
    offs_tile = tl.arange(0, TILE_K)

    topk_pos = tl.load(TopK_ptr + topk_base + offs_tile, mask=offs_tile < TOPK, other=-1)
    topk_pos_next = topk_pos

    for t in range(NUM_TILES):
        tile_start = t * TILE_K
        valid = (tile_start + offs_tile) < TOPK
        valid = valid & (topk_pos != -1)
        if t + 1 < NUM_TILES:
            next_offs = (t + 1) * TILE_K + offs_tile
            topk_pos_next = tl.load(TopK_ptr + topk_base + next_offs, mask=next_offs < TOPK, other=-1)
        safe_pos = tl.where(valid, topk_pos, 0)

        # K^T layout: [D_V, TILE_K]
        K_lora = tl.load(KV_ptr + safe_pos[None, :] * stride_kv_t + offs_v[:, None],
                          mask=valid[None, :], other=0.0)
        K_rope = tl.load(KV_ptr + safe_pos[None, :] * stride_kv_t + (D_V + offs_r[:, None]),
                          mask=valid[None, :], other=0.0)

        S = tl.dot(Q_lora, K_lora) + tl.dot(Q_rope, K_rope)
        S *= scale
        S = tl.where(valid[None, :] & mask_h[:, None], S, float("-inf"))

        m_j = tl.max(S, axis=1)
        m_new = tl.maximum(m_i, m_j)
        m_new = tl.where(m_new > float("-inf"), m_new, 0.0)
        alpha = tl.exp(m_i - m_new)
        P = tl.exp(S - m_new[:, None])
        l_new = alpha * l_i + tl.sum(P, axis=1)
        acc = acc * alpha[:, None]

        # V via tl.trans
        V_lora = tl.trans(K_lora)
        acc += tl.dot(P.to(V_lora.dtype), V_lora)

        m_i = m_new
        l_i = l_new
        if t + 1 < NUM_TILES:
            topk_pos = topk_pos_next

    acc = acc / l_i[:, None]
    lse = m_i + tl.log(l_i)
    o_base = token_idx * stride_o_t
    tl.store(O_ptr + o_base + offs_h[:, None] * stride_o_h + offs_v[None, :],
             acc.to(Q_lora.dtype), mask=mask_h[:, None])
    tl.store(LSE_ptr + token_idx * num_heads + offs_h, lse, mask=mask_h)


# ============================================================================
# Stage 2 & 3 kernel: Separate K/V loads (no tl.trans)
# ============================================================================
@triton.jit
def _fwd_kernel_sep(
    Q_ptr, KV_ptr, TopK_ptr, O_ptr, LSE_ptr,
    stride_q_t: tl.int64, stride_q_h: tl.int64,
    stride_kv_t: tl.int64,
    stride_o_t: tl.int64, stride_o_h: tl.int64,
    stride_topk_t: tl.int64,
    scale: tl.float32, num_heads: tl.int32,
    TOPK: tl.constexpr, BLOCK_H: tl.constexpr, TILE_K: tl.constexpr,
    D_V: tl.constexpr, D_ROPE: tl.constexpr,
):
    token_idx = tl.program_id(0)
    hg_idx = tl.program_id(1)
    offs_h = hg_idx * BLOCK_H + tl.arange(0, BLOCK_H)
    mask_h = offs_h < num_heads
    offs_v = tl.arange(0, D_V)
    offs_r = tl.arange(0, D_ROPE)

    q_base = token_idx * stride_q_t
    Q_lora = tl.load(Q_ptr + q_base + offs_h[:, None] * stride_q_h + offs_v[None, :],
                      mask=mask_h[:, None], other=0.0)
    Q_rope = tl.load(Q_ptr + q_base + offs_h[:, None] * stride_q_h + (D_V + offs_r[None, :]),
                      mask=mask_h[:, None], other=0.0)

    m_i = tl.full([BLOCK_H], float("-inf"), dtype=tl.float32)
    l_i = tl.full([BLOCK_H], 0.0, dtype=tl.float32)
    acc = tl.zeros([BLOCK_H, D_V], dtype=tl.float32)

    NUM_TILES: tl.constexpr = (TOPK + TILE_K - 1) // TILE_K
    topk_base = token_idx * stride_topk_t
    offs_tile = tl.arange(0, TILE_K)

    topk_pos = tl.load(TopK_ptr + topk_base + offs_tile, mask=offs_tile < TOPK, other=-1)
    topk_pos_next = topk_pos

    for t in range(NUM_TILES):
        tile_start = t * TILE_K
        valid = (tile_start + offs_tile) < TOPK
        valid = valid & (topk_pos != -1)
        if t + 1 < NUM_TILES:
            next_offs = (t + 1) * TILE_K + offs_tile
            topk_pos_next = tl.load(TopK_ptr + topk_base + next_offs, mask=next_offs < TOPK, other=-1)
        safe_pos = tl.where(valid, topk_pos, 0)

        # K^T layout: [D_V, TILE_K] and [D_ROPE, TILE_K]
        K_lora = tl.load(KV_ptr + safe_pos[None, :] * stride_kv_t + offs_v[:, None],
                          mask=valid[None, :], other=0.0)
        K_rope = tl.load(KV_ptr + safe_pos[None, :] * stride_kv_t + (D_V + offs_r[:, None]),
                          mask=valid[None, :], other=0.0)

        S = tl.dot(Q_lora, K_lora) + tl.dot(Q_rope, K_rope)
        S *= scale
        S = tl.where(valid[None, :] & mask_h[:, None], S, float("-inf"))

        m_j = tl.max(S, axis=1)
        m_new = tl.maximum(m_i, m_j)
        m_new = tl.where(m_new > float("-inf"), m_new, 0.0)
        alpha = tl.exp(m_i - m_new)
        P = tl.exp(S - m_new[:, None])
        l_new = alpha * l_i + tl.sum(P, axis=1)
        acc = acc * alpha[:, None]

        # V via separate global load: [TILE_K, D_V]
        V_lora = tl.load(KV_ptr + safe_pos[:, None] * stride_kv_t + offs_v[None, :],
                          mask=valid[:, None], other=0.0)
        acc += tl.dot(P.to(V_lora.dtype), V_lora)

        m_i = m_new
        l_i = l_new
        if t + 1 < NUM_TILES:
            topk_pos = topk_pos_next

    acc = acc / l_i[:, None]
    lse = m_i + tl.log(l_i)
    o_base = token_idx * stride_o_t
    tl.store(O_ptr + o_base + offs_h[:, None] * stride_o_h + offs_v[None, :],
             acc.to(Q_lora.dtype), mask=mask_h[:, None])
    tl.store(LSE_ptr + token_idx * num_heads + offs_h, lse, mask=mask_h)


# ============================================================================
# FLOPS formula (same for all stages — matches the one used in test file)
# ============================================================================
def compute_tflops(total_tokens, num_heads, topk, d_qk, kv_lora_rank, ms):
    """
    FLOPs for sparse MLA forward:
      - Q_lora @ K_lora^T: total_tokens * num_heads * topk * 2 * kv_lora_rank
      - Q_rope @ K_rope^T: total_tokens * num_heads * topk * 2 * (d_qk - kv_lora_rank)
      - P @ V_lora:        total_tokens * num_heads * topk * 2 * kv_lora_rank
    Total: total_tokens * num_heads * topk * 2 * (d_qk + kv_lora_rank)
    """
    flops = total_tokens * num_heads * topk * 2 * (d_qk + kv_lora_rank)
    return flops / (ms * 1e-3) / 1e12


# ============================================================================
# Benchmark runner
# ============================================================================
CONFIGS = [
    # (batch, seq_len, num_heads, kv_lora_rank, rope_rank, topk)
    (1, 4096, 128, 512, 64, 1024),
    (1, 4096, 128, 512, 64, 2048),
    (1, 8192, 128, 512, 64, 1024),
    (1, 8192, 128, 512, 64, 2048),
    (1, 4096,  32, 256, 64, 1024),
    (1, 4096,  16, 512, 64, 1024),
]


def run_benchmark(kernel_fn, block_h, tile_k, num_warps, num_stages, label, configs=CONFIGS):
    """Run benchmark with a specific kernel and fixed config."""
    print(f"\n{'='*80}")
    print(f"  {label}")
    print(f"  BLOCK_H={block_h}  TILE_K={tile_k}  num_warps={num_warps}  num_stages={num_stages}")
    print(f"{'='*80}")
    print(f"  {'Config':<30s} {'ms':>8s} {'TFLOPS':>8s}")
    print(f"  {'-'*50}")

    for batch, seq_len, num_heads, kv_lora_rank, rope_rank, topk in configs:
        d_qk = kv_lora_rank + rope_rank
        total_tokens = batch * seq_len
        scale = 1.0 / (d_qk ** 0.5)

        torch.manual_seed(42)
        q = torch.randn(total_tokens, num_heads, d_qk, dtype=torch.bfloat16, device="cuda")
        kv = torch.randn(total_tokens, 1, d_qk, dtype=torch.bfloat16, device="cuda")
        topk_indices = torch.randint(0, total_tokens, (total_tokens, topk),
                                     dtype=torch.int32, device="cuda")
        o = torch.empty(total_tokens, num_heads, kv_lora_rank, dtype=torch.bfloat16, device="cuda")
        lse = torch.empty(total_tokens, num_heads, dtype=torch.float32, device="cuda")

        actual_bh = min(block_h, num_heads)
        grid = (total_tokens, triton.cdiv(num_heads, actual_bh))

        # Warmup
        try:
            for _ in range(5):
                kernel_fn[grid](
                    q, kv, topk_indices, o, lse,
                    q.stride(0), q.stride(1), kv.stride(0),
                    o.stride(0), o.stride(1), topk_indices.stride(0),
                    scale, num_heads,
                    TOPK=topk, BLOCK_H=actual_bh, TILE_K=tile_k,
                    D_V=kv_lora_rank, D_ROPE=rope_rank,
                    num_warps=num_warps, num_stages=num_stages,
                )
            torch.cuda.synchronize()
        except Exception as e:
            print(f"  B{batch}_S{seq_len}_H{num_heads}_topk{topk}  FAILED: {e}")
            continue

        # Benchmark
        ev0 = torch.cuda.Event(enable_timing=True)
        ev1 = torch.cuda.Event(enable_timing=True)
        reps = 100
        ev0.record()
        for _ in range(reps):
            kernel_fn[grid](
                q, kv, topk_indices, o, lse,
                q.stride(0), q.stride(1), kv.stride(0),
                o.stride(0), o.stride(1), topk_indices.stride(0),
                scale, num_heads,
                TOPK=topk, BLOCK_H=actual_bh, TILE_K=tile_k,
                D_V=kv_lora_rank, D_ROPE=rope_rank,
                num_warps=num_warps, num_stages=num_stages,
            )
        ev1.record()
        torch.cuda.synchronize()
        ms = ev0.elapsed_time(ev1) / reps

        tflops = compute_tflops(total_tokens, num_heads, topk, d_qk, kv_lora_rank, ms)
        label_cfg = f"B{batch}_S{seq_len}_H{num_heads}_topk{topk}"
        print(f"  {label_cfg:<30s} {ms:8.3f} {tflops:8.1f}")


def run_autotune_benchmark(label, configs=CONFIGS):
    """Run benchmark with the autotuned kernel (Stage 4)."""
    from aiter.ops.triton._triton_kernels.attention.deepseek_sparse_attention import sparse_mla_fwd

    print(f"\n{'='*80}")
    print(f"  {label}")
    print(f"  Autotune (best config selected per problem)")
    print(f"{'='*80}")
    print(f"  {'Config':<30s} {'ms':>8s} {'TFLOPS':>8s} {'Autotune':>40s}")
    print(f"  {'-'*82}")

    for batch, seq_len, num_heads, kv_lora_rank, rope_rank, topk in configs:
        d_qk = kv_lora_rank + rope_rank
        total_tokens = batch * seq_len
        scale = 1.0 / (d_qk ** 0.5)

        torch.manual_seed(42)
        q = torch.randn(total_tokens, num_heads, d_qk, dtype=torch.bfloat16, device="cuda")
        kv = torch.randn(total_tokens, 1, d_qk, dtype=torch.bfloat16, device="cuda")
        topk_indices = torch.randint(0, total_tokens, (total_tokens, topk),
                                     dtype=torch.int32, device="cuda")

        # Warmup (triggers autotune)
        for _ in range(5):
            o, lse_out = sparse_mla_fwd(q, kv, topk_indices, kv_lora_rank, scale)
        torch.cuda.synchronize()

        # Benchmark
        ev0 = torch.cuda.Event(enable_timing=True)
        ev1 = torch.cuda.Event(enable_timing=True)
        reps = 100
        ev0.record()
        for _ in range(reps):
            sparse_mla_fwd(q, kv, topk_indices, kv_lora_rank, scale)
        ev1.record()
        torch.cuda.synchronize()
        ms = ev0.elapsed_time(ev1) / reps

        tflops = compute_tflops(total_tokens, num_heads, topk, d_qk, kv_lora_rank, ms)

        # Get autotune best config
        from aiter.ops.triton._triton_kernels.attention.deepseek_sparse_attention import _sparse_mla_fwd_train_kernel
        try:
            best = _sparse_mla_fwd_train_kernel.best_config
            cfg = f"BH={best.kwargs['BLOCK_H']} TK={best.kwargs['TILE_K']} w={best.num_warps} s={best.num_stages}"
        except Exception:
            cfg = "N/A"

        label_cfg = f"B{batch}_S{seq_len}_H{num_heads}_topk{topk}"
        print(f"  {label_cfg:<30s} {ms:8.3f} {tflops:8.1f} {cfg:>40s}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=int, nargs="*", default=[1, 2, 3, 4],
                        help="Which stages to benchmark (1-4)")
    args = parser.parse_args()

    gpu_name = torch.cuda.get_device_name(0)
    print(f"GPU: {gpu_name}")
    print(f"Triton: {triton.__version__}")
    print(f"PyTorch: {torch.__version__}")

    if 1 in args.stage:
        run_benchmark(_fwd_kernel_trans, block_h=16, tile_k=32, num_warps=4, num_stages=1,
                      label="Stage 1: tl.trans V-reuse, BH=16 TK=32 w=4 s=1")

    if 2 in args.stage:
        run_benchmark(_fwd_kernel_sep, block_h=16, tile_k=16, num_warps=4, num_stages=1,
                      label="Stage 2: Separate K/V loads, BH=16 TK=16 w=4 s=1")

    if 3 in args.stage:
        run_benchmark(_fwd_kernel_sep, block_h=32, tile_k=16, num_warps=4, num_stages=2,
                      label="Stage 3: Separate K/V loads, BH=32 TK=16 w=4 s=2")

    if 4 in args.stage:
        run_autotune_benchmark(
            label="Stage 4: tl.trans + autotune (current version)")


if __name__ == "__main__":
    main()
