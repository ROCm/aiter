---
name: gpu-kernel-algorithms
description: GPU kernel algorithm reference covering FlashAttention, online softmax, tiling strategies, MoE dispatch, GEMM optimization, quantization schemes, and other fundamental algorithms used in AI kernel development. Use when implementing attention mechanisms, matrix multiplication, mixture-of-experts, normalization, RoPE, or quantization kernels from scratch. Covers the mathematical foundations and algorithmic techniques behind high-performance GPU kernels.
---

# GPU Kernel Algorithms Reference

## FlashAttention Algorithm

### Standard Attention (Naive)

```
Q, K, V ∈ ℝ^{N×d}
S = Q @ K^T / √d        # (N×N) attention scores - QUADRATIC memory!
P = softmax(S, dim=-1)   # (N×N) attention weights
O = P @ V                # (N×d) output
```

**Problem:** S and P are N×N matrices. For N=128K, that's 128GB in FP32. Cannot fit in GPU memory.

### FlashAttention: IO-Aware Tiling

**Key insight:** Never materialize the full N×N attention matrix. Process in tiles.

**Algorithm (forward pass):**

```
Initialize: O = 0, l = 0 (running softmax denominator), m = -∞ (running max)

For each block j of K, V (size B_c):
    Load K_j, V_j from HBM to SRAM
    
    For each block i of Q (size B_r):
        Load Q_i, O_i, l_i, m_i from HBM to SRAM
        
        # Compute block attention scores
        S_ij = Q_i @ K_j^T / √d        # (B_r × B_c) - fits in SRAM!
        
        # Online softmax update
        m_new = max(m_i, rowmax(S_ij))
        P_ij = exp(S_ij - m_new)
        l_new = exp(m_i - m_new) * l_i + rowsum(P_ij)
        
        # Update output with rescaling
        O_new = exp(m_i - m_new) * O_i + P_ij @ V_j
        
        Write O_new, l_new, m_new back to HBM
        
# Final normalization: O = O / l
```

**Complexity:** O(N²d / M) HBM accesses where M = SRAM size, vs O(N²d + N²) for standard.

### Online Softmax (Milakov & Gimelshein, 2018)

The key mathematical trick enabling FlashAttention. Compute softmax in a single pass:

```
# Standard softmax (2 passes):
m = max(x)                    # Pass 1: find max
out = exp(x - m) / sum(exp(x - m))  # Pass 2: normalize

# Online softmax (1 pass, streaming):
m_i = -∞, d_i = 0
for j = 1 to N:
    m_new = max(m_i, x_j)
    d_new = d_i * exp(m_i - m_new) + exp(x_j - m_new)
    m_i = m_new
    d_i = d_new
out_j = exp(x_j - m_N) / d_N
```

**In Triton (log2-based for speed):**

```python
# log2(e) = 1.44269504 - use exp2 instead of exp for hardware efficiency
m_ij = tl.maximum(m_i, tl.max(qk, 1))
qk -= m_ij[:, None]
p = tl.math.exp2(qk * 1.44269504)  # exp(x) = exp2(x * log2(e))
l_ij = tl.sum(p, 1)
alpha = tl.math.exp2((m_i - m_ij) * 1.44269504)
l_i = l_i * alpha + l_ij
acc = acc * alpha[:, None]  # Rescale previous accumulator
acc += tl.dot(p.to(v.dtype), v)
m_i = m_ij
```

### FlashAttention-2 Improvements

1. **Reduce non-matmul FLOPs:** Minimize softmax rescaling operations
2. **Better parallelism:** Parallelize over sequence length (not just batch×heads)
3. **Improved warp distribution:** Reduce shared memory communication between warps
4. **Result:** 50-73% of theoretical peak (vs 25-40% for FA-1)

### FlashAttention Backward Pass

Requires recomputation of attention weights (not stored):

```
# Given: dO (gradient of output), Q, K, V, O, LSE (log-sum-exp from forward)
# Compute: dQ, dK, dV

For each block:
    Recompute S_ij = Q_i @ K_j^T / √d
    Recompute P_ij = exp(S_ij - LSE_i)  # Using stored LSE
    dV_j += P_ij^T @ dO_i
    dP_ij = dO_i @ V_j^T
    dS_ij = P_ij * (dP_ij - (dO_i * O_i).sum(dim=-1))  # Softmax grad
    dQ_i += dS_ij @ K_j / √d
    dK_j += dS_ij^T @ Q_i / √d
```

## GEMM Tiling Strategy

### Basic Blocked GEMM

```
C[M,N] = A[M,K] × B[K,N]

For each (i_block, j_block) tile of C:    # Parallelized across CUs
    acc = zeros(BLOCK_M, BLOCK_N)
    For k_block in range(0, K, BLOCK_K):   # Sequential reduction
        a_tile = A[i*BM:(i+1)*BM, k*BK:(k+1)*BK]   # Load from HBM → registers
        b_tile = B[k*BK:(k+1)*BK, j*BN:(j+1)*BN]   # Load from HBM → registers
        acc += a_tile @ b_tile              # MFMA instruction
    C[i*BM:(i+1)*BM, j*BN:(j+1)*BN] = acc  # Store to HBM
```

### L2 Cache Optimization (Grouped Ordering)

Instead of row-major tile ordering, use grouped ordering to improve L2 cache reuse:

```python
# Standard ordering: tiles (0,0), (0,1), (0,2), ... → poor B reuse
# Grouped ordering: tiles in GROUP_SIZE_M groups
if GROUP_SIZE_M == 1:
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n
else:
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
```

### Split-K GEMM

For small M (decode phase), split K dimension across multiple blocks:

```
# Standard: 1 block computes full K reduction
# Split-K: splitK blocks each compute K/splitK reduction, then reduce

For each split s in range(splitK):
    partial[s] = A[:, s*Ks:(s+1)*Ks] @ B[s*Ks:(s+1)*Ks, :]
C = sum(partial, dim=0)  # Final reduction
```

Increases parallelism when M×N tiles < num_CUs.

## MoE (Mixture of Experts) Dispatch

### Token Routing

```
scores = router(hidden_states)           # (M, num_experts)
topk_weights, topk_ids = topk(softmax(scores), k=top_k)  # (M, top_k)
```

### Expert-Grouped GEMM

**Challenge:** Each expert processes a different number of tokens.

**Solution:** Sort tokens by expert, pad to block boundaries, grouped GEMM:

```
1. Histogram: count tokens per expert
2. Pad each count to BLOCK_M multiple
3. Sort token indices by expert
4. Run single GEMM kernel with expert-indexed weights:
   - Weight tensor: W[num_experts, N, K]
   - For each M-tile: look up expert_id, offset W by expert_id * stride_e
   - Gather input tokens via sorted_token_ids
```

### Two-Stage MoE

```
Stage 1: gate_up = x @ W1    # W1 = [experts, 2*inter_dim, dim]
         gate, up = split(gate_up)
         hidden = silu(gate) * up
Stage 2: output = hidden @ W2  # W2 = [experts, dim, inter_dim]
```

## Quantization Algorithms

### Per-Tensor Static Quantization
```
scale = max(|x|) / dtype_max
x_quant = clamp(round(x / scale), -dtype_max, dtype_max)
x_dequant = x_quant * scale
```

### Per-Token Dynamic Quantization
```
For each row i:
    scale[i] = max(|x[i,:]|) / dtype_max
    x_quant[i,:] = clamp(round(x[i,:] / scale[i]), -dtype_max, dtype_max)
```

### Block-Scale (Per-Block) Quantization
```
For each block (i,j) of size (BM, BK):
    scale[i,j] = max(|x[i*BM:(i+1)*BM, j*BK:(j+1)*BK]|) / dtype_max
    x_quant[block] = round(x[block] / scale[i,j])
```

### MXFP4 (Microscaling FP4)
- 4-bit floating point: 1 sign + shared exponent + 2-bit mantissa
- Shared exponent per 32 elements (group size = 32)
- Two FP4 values packed into one byte
- Scale = 2^(shared_exponent)

## RoPE (Rotary Position Embedding)

### NeoX-Style Rotation
```
# Split dimension in half
x1, x2 = x[..., :d//2], x[..., d//2:]
rotated = [-x2, x1]  # Concatenated
# Apply rotation
y = x * cos(θ) + rotated * sin(θ)
```

### GPT-J Style Rotation
```
# Interleaved pairs
x1, x2 = x[..., 0::2], x[..., 1::2]
rotated = stack([-x2, x1], dim=-1).flatten(-2)
y = x * cos(θ) + rotated * sin(θ)
```

### Frequency Computation
```
θ_i = base^(-2i/d)  for i = 0, 1, ..., d/2-1
freqs = position * θ_i  # (seq_len, d/2)
```

## RMSNorm Algorithm

```
# RMSNorm: y = x * w / sqrt(mean(x²) + ε)
rms = sqrt(mean(x², dim=-1) + ε)  # Per-row RMS
y = (x / rms) * weight            # Normalize and scale
```

**Fused Add + RMSNorm (residual connection):**
```
x_new = x + residual    # Add residual
y = rmsnorm(x_new, w)   # Normalize
# Also write x_new back as the new residual for next layer
```

## Roofline Model

The roofline model determines whether a kernel is compute-bound or memory-bound:

```
Arithmetic Intensity (AI) = FLOPs / Bytes transferred

Ridge Point = Peak FLOPS / Peak Bandwidth

If AI > Ridge Point: compute-bound (optimize compute)
If AI < Ridge Point: memory-bound (optimize memory access)
```

**MI300X ridge points:**
- FP16: 1307 TFLOPS / 5.3 TB/s ≈ 246 FLOPs/byte
- FP8: 2615 TFLOPS / 5.3 TB/s ≈ 493 FLOPs/byte

**Common kernel AI:**
- GEMM (large): ~M*N*K / (M*K + N*K + M*N) → high AI (compute-bound)
- GEMM (M=1): ~2*N*K / (N*K + N) ≈ 2 → low AI (memory-bound, GEMV)
- Attention: ~4*N*d / (3*N*d + N²) → varies with sequence length
- Elementwise: ~1-2 FLOPs/element → always memory-bound

## Research Papers

- **FlashAttention**: "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness" (Dao et al., 2022) - arxiv:2205.14135
- **FlashAttention-2**: "FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning" (Dao, 2023) - arxiv:2307.08691
- **FlashAttention-3**: "FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision" (Shah et al., 2024) - arxiv:2407.08610
- **Online Softmax**: "Online normalizer calculation for softmax" (Milakov & Gimelshein, 2018) - arxiv:1805.02867
- **MoE**: "Switch Transformers: Scaling to Trillion Parameter Models" (Fedus et al., 2021) - arxiv:2101.03961
- **RoPE**: "RoFormer: Enhanced Transformer with Rotary Position Embedding" (Su et al., 2021) - arxiv:2104.09864
- **RMSNorm**: "Root Mean Square Layer Normalization" (Zhang & Sennrich, 2019) - arxiv:1910.07467
- **GPTQ/AWQ**: Weight quantization papers for INT4/INT8 inference
