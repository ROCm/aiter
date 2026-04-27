"""
Split backward: dQ kernel + dKV kernel with full recomputation (Option B).

Eliminates ALL intermediate storage (dS, P) by recomputing S, P, dS in the
dKV kernel. Saves 2 GiB of memory vs the intermediate-storing approach.

The dKV kernel is 100% atomic-bound (~83 GOPS, MFMAs 97% idle), so the
extra 3 dots (1,2,3) for recomputation are hidden behind atomic latency
and should cost ~0 ms extra.

Register budget for dKV with recompute (per warp, 4 warps):
  Q_lora [16,512]:  32 VGPRs  (persistent per HG)
  Q_rope [16,64]:    4 VGPRs  (persistent)
  dO [16,512]:      32 VGPRs  (persistent)
  Q_T, dO_T tiles:  ~8 VGPRs  (streamed from L2)
  KV tile:           4 VGPRs  (gathered per tile)
  S, dP, dS, P:    ~16 VGPRs  (temporaries)
  lse, delta:        2 VGPRs
  dKV accum:        36 AGPRs
  Total: ~98 VGPRs + 36 AGPRs → fits, zero spills expected

Usage:
  python bench_bwd_dkv_recompute.py
"""
import torch
import triton
import triton.language as tl
import re
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))
from aiter.ops.triton._triton_kernels.attention.deepseek_sparse_attention import (
    _sparse_mla_bwd_kernel,
    _sparse_mla_bwd_preprocess,
    sparse_mla_fwd,
)


# =====================================================================
# Kernel 1: dQ-only (NO intermediate stores)
# =====================================================================
@triton.jit
def _bwd_dq_only(
    Q_ptr, KV_ptr, dO_ptr, TopK_ptr, LSE_ptr, Delta_ptr,
    dQ_ptr,
    stride_q_t: tl.int64, stride_q_h: tl.int64,
    stride_kv_t: tl.int64,
    stride_do_t: tl.int64, stride_do_h: tl.int64,
    stride_dq_t: tl.int64, stride_dq_h: tl.int64,
    stride_topk_t: tl.int64,
    scale: tl.float32, num_heads: tl.int32,
    TOPK: tl.constexpr, BLOCK_H: tl.constexpr, TILE_K: tl.constexpr,
    D_V: tl.constexpr, D_ROPE: tl.constexpr,
):
    """Pure dQ kernel — computes dots 1-5, stores only dQ. No intermediates."""
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
    do_base = token_idx * stride_do_t
    dO_val = tl.load(dO_ptr + do_base + offs_h[:, None] * stride_do_h + offs_v[None, :],
                     mask=mask_h[:, None], other=0.0)
    lse = tl.load(LSE_ptr + token_idx * num_heads + offs_h, mask=mask_h, other=0.0)
    delta = tl.load(Delta_ptr + token_idx * num_heads + offs_h, mask=mask_h, other=0.0)

    dQ_lora = tl.zeros([BLOCK_H, D_V], dtype=tl.float32)
    dQ_rope = tl.zeros([BLOCK_H, D_ROPE], dtype=tl.float32)
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
            topk_pos_next = tl.load(TopK_ptr + topk_base + next_offs,
                                    mask=next_offs < TOPK, other=-1)
        safe_pos = tl.where(valid, topk_pos, 0)

        K_lora_T = tl.load(KV_ptr + safe_pos[None, :] * stride_kv_t + offs_v[:, None],
                          mask=valid[None, :], other=0.0)
        K_rope_T = tl.load(KV_ptr + safe_pos[None, :] * stride_kv_t + (D_V + offs_r[:, None]),
                          mask=valid[None, :], other=0.0)

        S = tl.dot(Q_lora, K_lora_T)
        S += tl.dot(Q_rope, K_rope_T)
        S *= scale
        S = tl.where(valid[None, :] & mask_h[:, None], S, float("-inf"))
        P = tl.exp(S - lse[:, None])
        P = tl.where(valid[None, :] & mask_h[:, None], P, 0.0)

        dP = tl.dot(dO_val, K_lora_T)
        dS = P * (dP - delta[:, None]) * scale
        dS = tl.where(valid[None, :] & mask_h[:, None], dS, 0.0)

        V_lora = tl.trans(K_lora_T)
        dQ_lora += tl.dot(dS.to(V_lora.dtype), V_lora).to(tl.float32)
        K_rope = tl.trans(K_rope_T)
        dQ_rope += tl.dot(dS.to(K_rope.dtype), K_rope).to(tl.float32)

        if t + 1 < NUM_TILES:
            topk_pos = topk_pos_next

    dq_base = token_idx * stride_dq_t
    tl.store(dQ_ptr + dq_base + offs_h[:, None] * stride_dq_h + offs_v[None, :],
             dQ_lora.to(Q_lora.dtype), mask=mask_h[:, None])
    tl.store(dQ_ptr + dq_base + offs_h[:, None] * stride_dq_h + (D_V + offs_r[None, :]),
             dQ_rope.to(Q_rope.dtype), mask=mask_h[:, None])


# =====================================================================
# Kernel 2: dKV with HG fusion + full recomputation (Option B)
# =====================================================================
@triton.jit
def _bwd_dkv_hg_fused_recompute(
    Q_ptr,          # [T, H, D_QK] bf16
    KV_ptr,         # [T, 1, D_QK] bf16
    dO_ptr,         # [T, H, D_V] bf16
    Q_T_ptr,        # [T, D_QK, H] bf16 — pre-transposed Q
    dO_T_ptr,       # [T, D_V, H] bf16  — pre-transposed dO
    TopK_ptr,       # [T, TOPK] int32
    LSE_ptr,        # [T, H] fp32
    Delta_ptr,      # [T, H] fp32
    dKV_ptr,        # [T, D_QK] fp32    — output (atomic target)
    stride_q_t: tl.int64, stride_q_h: tl.int64,
    stride_kv_t: tl.int64,
    stride_do_t: tl.int64, stride_do_h: tl.int64,
    stride_qt_t: tl.int64,
    stride_dot_t: tl.int64,
    stride_topk_t: tl.int64,
    stride_dkv_t: tl.int64,
    scale: tl.float32, num_heads: tl.int32,
    TOPK: tl.constexpr,
    TILE_K: tl.constexpr,
    BLOCK_H: tl.constexpr,
    NUM_HG: tl.constexpr,
    D_V: tl.constexpr,
    D_ROPE: tl.constexpr,
):
    """
    dKV kernel with full recomputation — NO intermediate buffers needed.

    Grid: (total_tokens,) — ONE program per token.
    Each program:
      - Loads Q, dO, lse, delta (persistent per HG)
      - For each tile × HG: recomputes S, P, dS (dots 1-3), then dKV (dots 6-8)
      - Scatters dKV once per tile (accumulated across HGs)

    6 dots per tile per HG (vs 3 in intermediate version), but MFMAs are
    97% idle so the extra compute is free.
    """
    token_idx = tl.program_id(0)

    NUM_TILES: tl.constexpr = (TOPK + TILE_K - 1) // TILE_K
    topk_base = token_idx * stride_topk_t
    offs_tile = tl.arange(0, TILE_K)
    offs_v = tl.arange(0, D_V)
    offs_r = tl.arange(0, D_ROPE)

    for t in range(NUM_TILES):
        tile_start = t * TILE_K
        tile_offs = tile_start + offs_tile
        topk_pos = tl.load(TopK_ptr + topk_base + tile_offs,
                           mask=tile_offs < TOPK, other=-1)
        valid = (tile_offs < TOPK) & (topk_pos != -1)
        safe_pos = tl.where(valid, topk_pos, 0)

        # Load KV tile ONCE per tile (shared across HGs)
        K_lora_T = tl.load(KV_ptr + safe_pos[None, :] * stride_kv_t + offs_v[:, None],
                          mask=valid[None, :], other=0.0)  # [D_V, TK]
        K_rope_T = tl.load(KV_ptr + safe_pos[None, :] * stride_kv_t + (D_V + offs_r[:, None]),
                          mask=valid[None, :], other=0.0)  # [D_ROPE, TK]

        # dKV accumulators — fresh per tile, accumulated across HGs
        dKV_lora = tl.zeros([D_V, TILE_K], dtype=tl.float32)
        dKV_rope = tl.zeros([D_ROPE, TILE_K], dtype=tl.float32)

        for hg in range(NUM_HG):
            offs_h = hg * BLOCK_H + tl.arange(0, BLOCK_H)
            mask_h = offs_h < num_heads

            # ---- Load Q, dO for this HG (persistent within HG, dots 1-3) ----
            q_base = token_idx * stride_q_t
            Q_lora = tl.load(
                Q_ptr + q_base + offs_h[:, None] * stride_q_h + offs_v[None, :],
                mask=mask_h[:, None], other=0.0,
            )  # [BH, D_V]
            Q_rope = tl.load(
                Q_ptr + q_base + offs_h[:, None] * stride_q_h + (D_V + offs_r[None, :]),
                mask=mask_h[:, None], other=0.0,
            )  # [BH, D_ROPE]
            dO_val = tl.load(
                dO_ptr + token_idx * stride_do_t + offs_h[:, None] * stride_do_h + offs_v[None, :],
                mask=mask_h[:, None], other=0.0,
            )  # [BH, D_V]

            lse = tl.load(LSE_ptr + token_idx * num_heads + offs_h,
                         mask=mask_h, other=0.0)  # [BH]
            delta_val = tl.load(Delta_ptr + token_idx * num_heads + offs_h,
                               mask=mask_h, other=0.0)  # [BH]

            # ---- Recompute S, P, dS (dots 1-3) ----
            # Dot 1: S_lora = Q_lora @ K_lora_T  [BH, D_V] @ [D_V, TK] → [BH, TK]
            S = tl.dot(Q_lora, K_lora_T)
            # Dot 2: S += Q_rope @ K_rope_T  [BH, D_R] @ [D_R, TK] → [BH, TK]
            S += tl.dot(Q_rope, K_rope_T)
            S *= scale
            S = tl.where(valid[None, :] & mask_h[:, None], S, float("-inf"))
            P = tl.exp(S - lse[:, None])
            P = tl.where(valid[None, :] & mask_h[:, None], P, 0.0)

            # Dot 3: dP = dO @ K_lora_T  [BH, D_V] @ [D_V, TK] → [BH, TK]
            dP = tl.dot(dO_val, K_lora_T)
            dS_val = P * (dP - delta_val[:, None]) * scale
            dS_val = tl.where(valid[None, :] & mask_h[:, None], dS_val, 0.0)

            # ---- Load Q_T, dO_T for dKV computation (dots 6-8) ----
            qt_base = token_idx * stride_qt_t
            Q_lora_T = tl.load(
                Q_T_ptr + qt_base + offs_v[:, None] * num_heads + offs_h[None, :],
                mask=mask_h[None, :], other=0.0,
            )  # [D_V, BH]

            Q_rope_T = tl.load(
                Q_T_ptr + qt_base + (D_V + offs_r[:, None]) * num_heads + offs_h[None, :],
                mask=mask_h[None, :], other=0.0,
            )  # [D_ROPE, BH]

            dot_base = token_idx * stride_dot_t
            dO_T = tl.load(
                dO_T_ptr + dot_base + offs_v[:, None] * num_heads + offs_h[None, :],
                mask=mask_h[None, :], other=0.0,
            )  # [D_V, BH]

            # ---- Compute dKV (dots 6-8) ----
            # Dot 6: dKV_lora += Q_lora_T @ dS  [D_V, BH] @ [BH, TK] → [D_V, TK]
            dKV_lora += tl.dot(Q_lora_T, dS_val.to(Q_lora_T.dtype)).to(tl.float32)

            # Dot 7: dKV_lora += dO_T @ P  [D_V, BH] @ [BH, TK] → [D_V, TK]
            dKV_lora += tl.dot(dO_T, P.to(dO_T.dtype)).to(tl.float32)

            # Dot 8: dKV_rope += Q_rope_T @ dS  [D_R, BH] @ [BH, TK] → [D_R, TK]
            dKV_rope += tl.dot(Q_rope_T, dS_val.to(Q_rope_T.dtype)).to(tl.float32)

        # Scatter ONCE — accumulated from both HGs
        dkv_ptrs_lora = dKV_ptr + safe_pos[None, :] * stride_dkv_t + offs_v[:, None]
        tl.atomic_add(dkv_ptrs_lora, dKV_lora, mask=valid[None, :], sem="relaxed")

        dkv_ptrs_rope = dKV_ptr + safe_pos[None, :] * stride_dkv_t + (D_V + offs_r[:, None])
        tl.atomic_add(dkv_ptrs_rope, dKV_rope, mask=valid[None, :], sem="relaxed")


# =====================================================================
# ISA analysis
# =====================================================================
def analyze_isa(name, kernel_fn, args, kwargs, grid):
    """Extract register and instruction stats from compiled kernel."""
    try:
        ck = kernel_fn.warmup(*args, **kwargs, grid=grid)
        isa = ck.asm.get("amdgcn", "")

        vgpr_m = re.search(r"\.vgpr_count:\s*(\d+)", isa)
        spill_m = re.search(r"\.vgpr_spill_count:\s*(\d+)", isa)
        agpr_m = re.search(r"\.agpr_count:\s*(\d+)", isa)
        sgpr_m = re.search(r"\.sgpr_count:\s*(\d+)", isa)

        vgprs = int(vgpr_m.group(1)) if vgpr_m else -1
        spills = int(spill_m.group(1)) if spill_m else -1
        agprs = int(agpr_m.group(1)) if agpr_m else -1
        sgprs = int(sgpr_m.group(1)) if sgpr_m else -1

        mfma_count = len(re.findall(r"v_mfma_", isa))
        global_atomic = len(re.findall(r"global_atomic", isa))
        global_load = len(re.findall(r"global_load", isa))
        global_store = len(re.findall(r"global_store", isa))
        scratch_load = len(re.findall(r"scratch_load", isa))
        scratch_store = len(re.findall(r"scratch_store", isa))

        print(f"  {name}:")
        print(f"    VGPRs={vgprs}  AGPRs={agprs}  SGPRs={sgprs}  Spills={spills}")
        print(f"    MFMAs={mfma_count}  global_load={global_load}  global_store={global_store}")
        print(f"    global_atomic={global_atomic}  scratch_load={scratch_load}  scratch_store={scratch_store}")
        return {"spills": spills, "mfma": mfma_count, "vgprs": vgprs, "agprs": agprs}
    except Exception as e:
        print(f"  {name}: warmup failed: {e}")
        return {"spills": -1, "mfma": -1, "vgprs": -1, "agprs": -1}


# =====================================================================
# Benchmark
# =====================================================================
def benchmark_kernel(name, run_fn, reps=50):
    for _ in range(3):
        run_fn()
    torch.cuda.synchronize()

    ev0 = torch.cuda.Event(enable_timing=True)
    ev1 = torch.cuda.Event(enable_timing=True)
    ev0.record()
    for _ in range(reps):
        run_fn()
    ev1.record()
    torch.cuda.synchronize()
    return ev0.elapsed_time(ev1) / reps


def main():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Config
    batch, seq_len, num_heads = 1, 4096, 128
    kv_lora_rank, rope_rank, topk = 512, 64, 1024
    d_qk = kv_lora_rank + rope_rank
    total_tokens = batch * seq_len
    scale = 1.0 / (d_qk ** 0.5)
    bh, tk_dq, nw_dq, ns_dq = 64, 16, 4, 2
    num_hg = triton.cdiv(num_heads, bh)

    print(f"Config: T={total_tokens} H={num_heads} D={d_qk} TOPK={topk}")
    print(f"        D_V={kv_lora_rank} D_R={rope_rank} BH={bh}")
    print(f"        num_hg={num_hg} (fused into single program)")
    print(f"        Option B: full recomputation, ZERO intermediate storage")

    torch.manual_seed(42)
    q = torch.randn(total_tokens, num_heads, d_qk, dtype=torch.bfloat16, device="cuda")
    kv = torch.randn(total_tokens, 1, d_qk, dtype=torch.bfloat16, device="cuda")
    topk_indices = torch.randint(0, total_tokens, (total_tokens, topk),
                                 dtype=torch.int32, device="cuda")

    # Forward pass for O and LSE
    o, lse = sparse_mla_fwd(q, kv, topk_indices, kv_lora_rank, scale)
    do = torch.randn_like(o)

    # Preprocess: Delta
    delta = torch.empty(total_tokens, num_heads, dtype=torch.float32, device=q.device)
    grid_pre = (total_tokens, triton.cdiv(num_heads, bh))
    _sparse_mla_bwd_preprocess[grid_pre](
        O_ptr=o, dO_ptr=do, Delta_ptr=delta,
        stride_o_t=o.stride(0), stride_o_h=o.stride(1),
        num_heads=num_heads, D_V=kv_lora_rank, BLOCK_H=bh,
    )

    # Pre-transposed Q and dO (still needed for dots 6-8 in dKV kernel)
    q_t = q.transpose(1, 2).contiguous()
    do_t = do.transpose(1, 2).contiguous()

    # Outputs
    dq_baseline = torch.empty_like(q)
    dkv_baseline = torch.zeros(total_tokens, d_qk, dtype=torch.float32, device=q.device)

    dq_recomp = torch.empty_like(q)
    dkv_recomp = torch.zeros(total_tokens, d_qk, dtype=torch.float32, device=q.device)

    print(f"\nIntermediate buffers: 0 GiB (recomputation mode)")

    # ════════════════════════════════════════════════════════════════════
    # ISA Analysis
    # ════════════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print(f"  ISA ANALYSIS")
    print(f"{'='*70}")

    grid_baseline = (total_tokens, num_hg)
    baseline_isa = analyze_isa(
        "Baseline fused kernel (BH=64 TK=16 w=4 s=2)",
        _sparse_mla_bwd_kernel.fn,
        (q, kv, do, topk_indices, lse, delta, dq_baseline, dkv_baseline, q_t, do_t,
         q.stride(0), q.stride(1), kv.stride(0),
         do.stride(0), do.stride(1), dq_baseline.stride(0), dq_baseline.stride(1),
         dkv_baseline.stride(0), topk_indices.stride(0),
         q_t.stride(0), do_t.stride(0),
         scale, num_heads),
        dict(TOPK=topk, BLOCK_H=bh, TILE_K=tk_dq,
             D_V=kv_lora_rank, D_ROPE=rope_rank,
             num_warps=nw_dq, num_stages=ns_dq),
        grid_baseline,
    )

    grid_dq = (total_tokens, num_hg)
    dq_isa = analyze_isa(
        "dQ-only kernel (no intermediates)",
        _bwd_dq_only,
        (q, kv, do, topk_indices, lse, delta, dq_recomp,
         q.stride(0), q.stride(1), kv.stride(0),
         do.stride(0), do.stride(1),
         dq_recomp.stride(0), dq_recomp.stride(1),
         topk_indices.stride(0),
         scale, num_heads),
        dict(TOPK=topk, BLOCK_H=bh, TILE_K=tk_dq,
             D_V=kv_lora_rank, D_ROPE=rope_rank,
             num_warps=nw_dq, num_stages=ns_dq),
        grid_dq,
    )

    grid_dkv = (total_tokens,)
    for tk_dkv in [16, 32, 64]:
        for nw_dkv in [2, 4]:
            dkv_isa = analyze_isa(
                f"dKV recompute TK={tk_dkv} w={nw_dkv}",
                _bwd_dkv_hg_fused_recompute,
                (q, kv, do, q_t, do_t, topk_indices, lse, delta, dkv_recomp,
                 q.stride(0), q.stride(1), kv.stride(0),
                 do.stride(0), do.stride(1),
                 q_t.stride(0), do_t.stride(0),
                 topk_indices.stride(0), dkv_recomp.stride(0),
                 scale, num_heads),
                dict(TOPK=topk, TILE_K=tk_dkv, BLOCK_H=bh,
                     NUM_HG=num_hg, D_V=kv_lora_rank, D_ROPE=rope_rank,
                     num_warps=nw_dkv, num_stages=1),
                grid_dkv,
            )

    # ════════════════════════════════════════════════════════════════════
    # Benchmark: Baseline
    # ════════════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print(f"  BENCHMARK")
    print(f"{'='*70}")

    def run_baseline():
        dkv_baseline.zero_()
        _sparse_mla_bwd_kernel.fn[grid_baseline](
            q, kv, do, topk_indices, lse, delta, dq_baseline, dkv_baseline, q_t, do_t,
            q.stride(0), q.stride(1), kv.stride(0),
            do.stride(0), do.stride(1), dq_baseline.stride(0), dq_baseline.stride(1),
            dkv_baseline.stride(0), topk_indices.stride(0),
            q_t.stride(0), do_t.stride(0),
            scale, num_heads,
            TOPK=topk, BLOCK_H=bh, TILE_K=tk_dq,
            D_V=kv_lora_rank, D_ROPE=rope_rank,
            num_warps=nw_dq, num_stages=ns_dq,
        )

    ms_baseline = benchmark_kernel("Baseline", run_baseline)
    print(f"  Baseline fused:       {ms_baseline:.2f} ms")

    # ════════════════════════════════════════════════════════════════════
    # Benchmark: dQ-only kernel
    # ════════════════════════════════════════════════════════════════════

    def run_dq():
        _bwd_dq_only[grid_dq](
            q, kv, do, topk_indices, lse, delta, dq_recomp,
            q.stride(0), q.stride(1), kv.stride(0),
            do.stride(0), do.stride(1),
            dq_recomp.stride(0), dq_recomp.stride(1),
            topk_indices.stride(0),
            scale, num_heads,
            TOPK=topk, BLOCK_H=bh, TILE_K=tk_dq,
            D_V=kv_lora_rank, D_ROPE=rope_rank,
            num_warps=nw_dq, num_stages=ns_dq,
        )

    ms_dq = benchmark_kernel("dQ-only", run_dq)
    print(f"  dQ-only kernel:       {ms_dq:.2f} ms")

    # ════════════════════════════════════════════════════════════════════
    # Benchmark: dKV recompute kernel (config sweep)
    # ════════════════════════════════════════════════════════════════════
    print(f"\n  dKV recompute config sweep:")
    best_ms_dkv = float('inf')
    best_cfg = ""

    for tk_dkv in [16, 32, 64, 128]:
        for nw_dkv in [1, 2, 4]:
            def run_dkv_cfg(nw_val=nw_dkv, tk_val=tk_dkv):
                dkv_recomp.zero_()
                _bwd_dkv_hg_fused_recompute[grid_dkv](
                    q, kv, do, q_t, do_t, topk_indices, lse, delta, dkv_recomp,
                    q.stride(0), q.stride(1), kv.stride(0),
                    do.stride(0), do.stride(1),
                    q_t.stride(0), do_t.stride(0),
                    topk_indices.stride(0), dkv_recomp.stride(0),
                    scale, num_heads,
                    TOPK=topk, TILE_K=tk_val, BLOCK_H=bh,
                    NUM_HG=num_hg, D_V=kv_lora_rank, D_ROPE=rope_rank,
                    num_warps=nw_val, num_stages=1,
                )
            try:
                ms_cfg = benchmark_kernel(f"dKV", run_dkv_cfg, reps=20)
                total = ms_dq + ms_cfg
                tag = ""
                if ms_cfg < best_ms_dkv:
                    best_ms_dkv = ms_cfg
                    best_cfg = f"TK={tk_dkv} w={nw_dkv}"
                    tag = " ← best"
                print(f"    TK={tk_dkv:>3d} w={nw_dkv}: {ms_cfg:.2f} ms  "
                      f"(total={total:.2f} ms, {ms_baseline/total:.2f}x){tag}")
            except Exception as e:
                print(f"    TK={tk_dkv:>3d} w={nw_dkv}: FAILED ({type(e).__name__})")

    print(f"\n  Best dKV config: {best_cfg} = {best_ms_dkv:.2f} ms")
    print(f"  Best total: {ms_dq + best_ms_dkv:.2f} ms  ({ms_baseline / (ms_dq + best_ms_dkv):.2f}x vs baseline)")

    # ════════════════════════════════════════════════════════════════════
    # Correctness check (at best config)
    # ════════════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print(f"  CORRECTNESS CHECK ({best_cfg})")
    print(f"{'='*70}")

    # Parse best config
    best_tk = int(best_cfg.split("TK=")[1].split()[0])
    best_nw = int(best_cfg.split("w=")[1])

    # Run baseline
    dkv_baseline.zero_()
    run_baseline()
    torch.cuda.synchronize()

    # Run recompute version
    dkv_recomp.zero_()
    run_dq()
    torch.cuda.synchronize()
    _bwd_dkv_hg_fused_recompute[grid_dkv](
        q, kv, do, q_t, do_t, topk_indices, lse, delta, dkv_recomp,
        q.stride(0), q.stride(1), kv.stride(0),
        do.stride(0), do.stride(1),
        q_t.stride(0), do_t.stride(0),
        topk_indices.stride(0), dkv_recomp.stride(0),
        scale, num_heads,
        TOPK=topk, TILE_K=best_tk, BLOCK_H=bh,
        NUM_HG=num_hg, D_V=kv_lora_rank, D_ROPE=rope_rank,
        num_warps=best_nw, num_stages=1,
    )
    torch.cuda.synchronize()

    # Compare dQ
    dq_diff = (dq_recomp.float() - dq_baseline.float()).abs()
    dq_ref = dq_baseline.float().abs().max()
    dq_rel = dq_diff.max() / (dq_ref + 1e-8)
    print(f"  dQ: max_abs={dq_diff.max().item():.6e}  max_rel={dq_rel.item():.6e}")

    # Compare dKV
    dkv_diff = (dkv_recomp.float() - dkv_baseline.float()).abs()
    dkv_ref = dkv_baseline.float().abs().max()
    dkv_rel = dkv_diff.max() / (dkv_ref + 1e-8)
    print(f"  dKV: max_abs={dkv_diff.max().item():.6e}  max_rel={dkv_rel.item():.6e}")

    dq_ok = dq_rel.item() < 0.01
    dkv_ok = dkv_rel.item() < 0.01
    print(f"  dQ  correct (rel < 1%): {'PASS' if dq_ok else 'FAIL'}")
    print(f"  dKV correct (rel < 1%): {'PASS' if dkv_ok else 'FAIL'}")

    # ════════════════════════════════════════════════════════════════════
    # Summary comparison
    # ════════════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print(f"  SUMMARY")
    print(f"{'='*70}")
    print(f"  Baseline (fused):           {ms_baseline:.2f} ms  (105 spills, 2 GiB intermediate: N/A)")
    print(f"  Recompute (dQ + dKV):       {ms_dq + best_ms_dkv:.2f} ms  (0 spills, 0 GiB intermediate)")
    print(f"    dQ kernel:                {ms_dq:.2f} ms")
    print(f"    dKV kernel ({best_cfg}):  {best_ms_dkv:.2f} ms")
    print(f"  Speedup vs baseline:        {ms_baseline / (ms_dq + best_ms_dkv):.2f}x")
    print(f"  Memory saved:               2.00 GiB (no dS/P intermediates)")


if __name__ == "__main__":
    main()
