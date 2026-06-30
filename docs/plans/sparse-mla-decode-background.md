

# Background: Sparse MLA Decode

> **Prerequisite.** Parts 1–3 (attention, MQA/MLA, sparse attention) are covered in more
> depth in the sibling background document for the FP8 MQA-logits kernel:
> [`fp8-mqa-logits/00-background/attention-to-lightning-indexer.md`](../../knowledge-base/fp8-mqa-logits/00-background/attention-to-lightning-indexer.md).
> This document focuses on the **decode execution path** — what happens at inference time
> after the top-k token selection step, and how sparse attention changes the shape of that
> decode.

---

## Notation

| Symbol | Meaning | Typical value (DeepSeek V3/V4) |
|--------|---------|-------------------------------|
| `B` | batch size (number of decode requests in flight) | 1 – 256 |
| `T` | number of decode tokens (`= B` for single-token decode) | 1 – 256 |
| `H` | number of query heads | 128 |
| `D` | per-head dimension (after MLA absorption) | 576 (`= 512 + 64`) |
| `Dv` | value head dimension | 512 |
| `kv_lora_rank` | MLA KV latent dimension | 512 |
| `rope_rank` | MLA decoupled RoPE key dimension | 64 |
| `L` | full KV sequence length (context window) | up to 128k |
| `k` | sparse top-k budget (tokens attended per query) | 2048 |
| `BLOCK_K` | KV tile width in the decode inner loop | 16 or 32 |
| `KV_SPLITS` | number of parallel split-K partitions | power of 2, auto |

---

## Part 1 — Dense MHA Decode: The Starting Point

### 1.1 The single-token decode shape

When a language model generates text, it produces one token at a time. At each step, the
new token is projected to a **single query row** per head, while the keys and values come
from the **entire past context** stored in the KV cache:

```
Q  [1, H, D]         (one new query, H heads)
K  [L, H, D]         (all past keys, read from KV cache)
V  [L, H, Dv]        (all past values, read from KV cache)

attention output [1, H, Dv] = softmax(Q·Kᵀ / √D) · V
```

This is **bandwidth-bound**: you read the full K and V tensors (order of GBs for long
context) but do only `O(L · H · D)` arithmetic — very few FLOPs per byte. The GPU's
memory bandwidth is the bottleneck, not its compute units.

### 1.2 Paged KV cache

In real inference servers, tokens from different requests share GPU memory in a pooled
**paged KV cache** (borrowed from OS virtual memory). The KV sequences are not stored
contiguously — instead, each sequence's KV entries are scattered across fixed-size
**pages** (also called blocks), and a per-request **block table** maps logical page
indices to physical page slots.

```
kv_cache:     [total_pages, page_size, H, D]   (physical slots)
block_table:  [B, max_pages_per_seq]            (logical→physical map)
```

A decode kernel must **gather** the relevant pages for each request before doing the
attention dot products.

---

## Part 2 — MLA: Compressed KV Cache

### 2.1 Why the KV cache is large

In standard MHA each token caches `H × D` floats for K and the same for V. With
`H = 128`, `D = 128`, fp16, and 128k tokens that is ~8.6 GB **per layer**. Across
60+ layers, this dominates GPU memory and limits batch size.

### 2.2 MLA's low-rank latent trick

DeepSeek-V2 introduced **Multi-head Latent Attention (MLA)**: instead of caching the
full per-head K and V, it caches a compact **latent vector** `c` of size `kv_lora_rank`
(512 in DSv4) per token. At decode time the full K/V are reconstructed on the fly by
two learned up-projection matrices `W_UK`, `W_UV`.

```
Cached per token:   c [kv_lora_rank]         (≈ 512 floats)
                    k_R [rope_rank]           (≈ 64 floats, for RoPE)

Reconstructed:      K_content = c · W_UK     [H, kv_lora_rank]
                    K = [K_content | RoPE(k_R)]   [H, kv_lora_rank + rope_rank]
                    V = c · W_UV             [H, kv_lora_rank]
```

**The absorbed-query trick.** Because `K_content = c · W_UK`, the score can be rewritten:

```
Q_content · K_contentᵀ = (Q_content · W_UK^T) · cᵀ
```

By folding `W_UK^T` into the query projection, we can compute attention **directly
against the cached latent `c`** — no materialization of K_content needed. The decode
kernel's effective per-head dimension then becomes `kv_lora_rank + rope_rank = 576`.

This is the **"absorbed"** MLA representation used by `mla_decode_fwd` in aiter:

```
q: [T, H, D]     where D = kv_lora_rank + rope_rank = 576
kv: [L, 1, D]    shared latent, 1 KV head (pure MQA shape)
```

The cache is ~32× smaller than standard MHA at the same model quality.

---

## Part 3 — DeepSeek Sparse Attention (DSA)

### 3.1 The problem: full attention still scans L tokens

Even with MLA shrinking the per-token cache size, decode still touches **all L past
tokens** per step. At L = 128k that is a lot of bandwidth regardless.

**Sparse attention** observes that for any query, only a small subset of past tokens
contribute meaningfully. If those tokens can be identified cheaply, we can skip the rest.

### 3.2 The three-stage pipeline

DeepSeek V3.2+ runs sparse attention through three stages at each decode step:

```
[Query token]
     │
     ▼
Stage 1: Lightning Indexer (FP8 MQA logits kernel)
     │   Scores all L past tokens cheaply (few FP8 heads, no softmax)
     │   Output: score[t, s] for each past token s
     ▼
Stage 2: Top-k selection
     │   Keep the k highest-scoring tokens per query (k ≈ 2048)
     │   Output: kv_indices[t, :k] — the sparse index list
     ▼
Stage 3: Sparse MLA decode  ← Our focus
         Attend only to the k selected tokens
         Output: attention output vector per query head
```

### 3.3 What changes in sparse decode

The key structural difference from dense decode:

| Aspect | Dense decode | Sparse decode |
|--------|-------------|---------------|
| KV positions attended | All L tokens | Top-k selected (~2048) |
| KV index structure | Contiguous block table | Scattered `kv_indices[T, k]` |
| Bandwidth | Reads all L KV entries | Reads ~k KV entries (gather) |
| Arithmetic intensity | Low (bandwidth-bound) | Even lower — fewer tokens |
| Bottleneck | Sequential memory reads | Memory + gather overhead |

The sparse path gathers K/V from a **unified KV pool** using a flat list of slot indices,
rather than reading a contiguous range from the block table.

---

## Part 4 — The Sparse Decode Kernel: `pa_decode_sparse`

### 4.0 What the kernel computes

#### Operand shapes

| Tensor | Shape | Dtype | Notes |
|--------|-------|-------|-------|
| `q` | `[T, H, D]` | bf16 | Absorbed query: includes RoPE and W_UK^T fold |
| `unified_kv` | `[total_slots, D]` | bf16/fp8 | Flat latent cache; **one KV head (MQA)** |
| `kv_indices` | `[total_indices]` | int32 | Sparse slot list for all tokens |
| `kv_indptr` | `[T+1]` | int32 | CSR row pointers into `kv_indices` |
| `attn_sink` | `[H]` | fp32 | Per-head learnable sink log-weight |
| `out` | `[T, H, D]` | bf16 | Attention output in absorbed latent space |

Typical values for DeepSeek V3/V4: `H = 128`, `D = 576`, `T = 1..256`, sparse `k ≈ 2048`.

#### The attended set

For each decode token `t`, the top-k selector has pre-chosen `k_t` past token slots to
attend to. Their slot indices are:

```
S_t = kv_indices[ kv_indptr[t] : kv_indptr[t+1] ]   # length k_t ≈ k = 2048
```

Each slot `s ∈ S_t` identifies a row in `unified_kv`; that row is the absorbed KV
latent vector `c_s ∈ R^D` for that cached token.

#### Attention formula

For each query token `t` and head `h`:

```
# Query vector (absorbed, pre-scaled)
q_h = q[t, h, :] * softmax_scale           # shape [D],  softmax_scale = 1/√D

# Sparse KV gather (same latent for all heads — MQA)
c_j = unified_kv[ S_t[j], : ]              # shape [D],  j = 0..k_t-1

# Dot-product scores
score_j = q_h · c_j                        # scalar (full D-dim dot product)

# Softmax weights (with numerical stability via running max m)
w_j = exp(score_j) / Σ_j exp(score_j)

# Attention output
out[t, h, :] = Σ_j  w_j * c_j             # shape [D]
```

#### Key structural observations

1. **MQA (one KV head).** `unified_kv` stores a single `D`-dimensional vector per
   cached token, shared by all `H` query heads. The per-head dimension only affects `q`
   and `out` — the KV gather is head-independent. This means for a batch of `T` tokens
   with `k = 2048` sparse slots each, the kernel reads `T × k × D × 2` bytes of KV
   data regardless of `H`.

2. **Absorbed MLA latent space.** Both keys and values are the same cached latent `c_s`.
   The "key" role is served by the full `D = 576` vector (after absorbing `W_UK^T` into
   the query). The "value" role is served by the same `c_s`, with the distinction between
   `kv_lora_rank = 512` and `rope_rank = 64` resolved by the downstream output projection
   — not inside this kernel.

3. **Per-tile, per-split decomposition.** The sum `Σ_j` over `k_t` slots is never
   computed all at once. It is broken into tiles of `BLOCK_K = 16` slots and further
   partitioned across `KV_SPLITS` parallel CTAs, using the online softmax algorithm to
   make this associative (see §4.2–4.4).

4. **Attention sink as a virtual token.** A learnable per-head scalar `attn_sink[h]` is
   added to the softmax denominator as if there were an extra token with score
   `attn_sink[h]` and value zero. This absorbs "missing mass" when the true top-k tokens
   have collectively low softmax weight (see §4.5).

#### Matrix decomposition and MFMA tiles

The inner KV loop body reduces to two matrix multiplications per tile. The MFMA
(Matrix Fused Multiply-Accumulate) instructions on gfx942 operate on fixed-size tiles,
so the kernel shapes are chosen to match.

**MFMA tile on gfx942 for bf16:**
`v_mfma_f32_16x16x16_bf16` — computes `C[16,16] += A[16,16] × B[16,16]` in bf16
arithmetic, accumulating into fp32. One wave64 (64 lanes) executes one such instruction
cooperatively. This is the minimum MFMA tile and the one used here.

**One CTA = one token.** The kernel grid is `(T, H/BLOCK_H, KV_SPLITS)`. The `grid.x`
dimension fixes the token index `t` for the entire lifetime of a CTA. `Q_tile`,
`K_tile`, and `S_tile` are therefore all scoped to a **single decode token `t`**.

The conceptual score matrix for that token is `[H, k_t]` — `H=128` heads by `k_t ≈ 2048`
sparse KV slots. It is **never materialized**: the grid's `H/BLOCK_H` CTAs each own a
`[BLOCK_H, k_t]` horizontal strip, and within each CTA the KV dimension is further
tiled into `BLOCK_K`-wide chunks that are processed sequentially by the inner loop. Each
`[BLOCK_H, BLOCK_K]` fragment is consumed by the online softmax immediately and discarded.

```
Score matrix for token t (conceptual, never fully in memory):

              kv slot 0        kv slot 15 | kv slot 16      kv slot 31 | ...
head 0    ┌─────────────────────────────┐ ┌──────────────────────────┐
  ...     │   S_tile [16×16]  CTA(t,0,0)│ │  S_tile [16×16] CTA(t,0,0)  inner loop tile 1 ...
head 15   └─────────────────────────────┘ └──────────────────────────┘
head 16   ┌─────────────────────────────┐
  ...     │   S_tile [16×16]  CTA(t,1,0)│ ...
head 31   └─────────────────────────────┘
  ...
```

**The two matrix multiplications per KV tile** (each CTA, single token `t`, head block
`[h₀, h₀+BLOCK_H)`, KV slot tile `[j₀, j₀+BLOCK_K)`):

```
Step 1 — QK^T  (score computation):

  Q_tile  [BLOCK_H, D]        = [16, 576] bf16
          q[t, h₀:h₀+16, :]   — 16 absorbed query vectors for token t

  K_tile  [BLOCK_K, D]        = [16, 576] bf16
          unified_kv[S_t[j₀:j₀+16], :]
                               — 16 gathered KV latents from token t's sparse set
                                 (same rows for every head block — MQA)

  S_tile  [BLOCK_H, BLOCK_K]  = [16, 16] fp32
          S_tile[i, j] = score of head (h₀+i) attending to KV slot S_t[j₀+j]
          — a 2D tile of the full [H, k_t] score matrix, for token t only

  S_tile = Q_tile @ K_tile^T

  Decomposed along D = 576 = 36 × 16 (MFMA inner K-dimension = 16):
      for d in range(0, D, 16):
          S_tile += v_mfma_f32_16x16x16_bf16(Q_tile[:, d:d+16],
                                              K_tile[:, d:d+16].T)
  → 36 MFMA instructions per KV tile

Step 2 — PV  (weighted value accumulation):

  P_tile  [BLOCK_H, BLOCK_K]  = [16, 16] bf16   softmax weights after online-softmax update
  V_tile  [BLOCK_K, D]        = [16, 576] bf16   same gathered KV rows reused as values

  acc_tile[BLOCK_H, D]        = [16, 576] fp32   running output accumulator, token t

  acc_tile += P_tile @ V_tile

  Decomposed along D = 576 = 36 × 16:
      for d in range(0, D, 16):
          acc_tile[:, d:d+16] += v_mfma_f32_16x16x16_bf16(P_tile,
                                                            V_tile[:, d:d+16])
  → 36 MFMA instructions per KV tile
```

**Total MFMA budget per KV tile:** 36 (QK^T) + 36 (PV) = **72 MFMA instructions**.

**Why `BLOCK_H = 16` is the minimum.** The MFMA instruction requires its matrix
dimensions to be at least 16. A smaller `BLOCK_H` (e.g., `BLOCK_H=1` in the current
FlyDSL implementation) cannot use MFMA and must fall back to scalar `VALU` dot-product
with a 6-round butterfly wave-reduction across lanes — functional but leaving MFMA
throughput unused.

**D = 576 = 36 × 16 — exact fit.** No padding needed along the inner K-dimension of
the MFMA. `D = 512` (32 × 16) is similarly clean. Both are exact multiples of the
MFMA K-tile of 16.

**AGPR vs LDS for `acc_tile`.** The 16 × 576 fp32 accumulator `acc_tile` requires
16 × 576 × 4 = 36 KB. On gfx942, MFMA destinations are accumulator registers (AGPRs).
A `[16, 16]` MFMA C-tile uses 16 VGPRs per lane (one per C-row). For 36 tiles along D,
the full accumulator occupies 36 × 16 = 576 AGPRs per lane — well within gfx942's 512
AGPR limit if split across registers, but in practice typically kept in LDS or written
tile-by-tile to avoid AGPR pressure at large D.

**The Triton kernel** (`_pa_decode_sparse`) expresses both steps as `tl.dot()` calls,
which the Triton compiler lowers to sequences of `v_mfma_f32_16x16x16_bf16`. The
FlyDSL port (`pa_decode_sparse.py`) currently uses `BLOCK_H=1` with scalar ops; a
future MFMA-based FlyDSL version would raise `BLOCK_H` to 16 and issue MFMA explicitly.

### 4.1 Data layout

The aiter implementation (`aiter/ops/triton/attention/pa_decode_sparse.py`) uses a
**unified KV pool** with flat scatter/gather addressing:

```
unified_kv:  [total_slots, D]      bf16/fp16/fp8 — one row per cached token slot
kv_indices:  [total_indices]       int32 — sparse slot list, sentinel -1 = skip
kv_indptr:   [T+1]                 int32 — CSR row pointers into kv_indices
q:           [T, H, D]             bf16/fp16 — decode queries
attn_sink:   [H]                   fp32 — per-head attention sink
out:         [T, H, Dv]            bf16/fp16 — output
```

**`kv_indptr` and `kv_indices` form a CSR (Compressed Sparse Row) structure.**
For token `t`, its sparse KV slot list occupies:

```
kv_indices[ kv_indptr[t] : kv_indptr[t+1] ]
```

Example with 3 tokens having 3, 2, and 4 selected slots:

```
kv_indptr  = [0,    3,  5,       9]
kv_indices = [s0, s1, s2, s3, s4, s5, s6, s7, s8]
              ←token 0→  ←t1→  ←── token 2 ──→
```

Each `slot_j` in `kv_indices` is a **direct row index into `unified_kv`**. The KV
vector for that cached token is `unified_kv[slot_j, 0:D]`, i.e. the flat address of
element `v` is simply `slot_j * D + v`. There is no page-number / within-page-offset
decomposition — each row in `unified_kv` holds exactly one cached token's full
`D`-dimensional vector.

**Why a flat index list instead of a block table?** The top-k selector produces an
unordered list of arbitrary past token positions. There is no locality to exploit with
a block table. A flat gather list is the natural representation.

> **Note on naming.** `unified_kv` is sometimes described as `[total_pages, D]` in
> older comments. This is a historical artifact from paged-KV terminology. Functionally
> `total_pages == total_slots == total_cached_tokens` — there is no sub-page structure.

### 4.2 Split-K decomposition (flash-decode)

A single decode token attends to `k ≈ 2048` tokens — enough work for one GPU CU, but
not enough to fill MI300X's 304 CUs. To exploit parallelism across all CUs, the kernel
uses a **split-K** (flash-decode) strategy:

**Idea:** partition each token's sparse KV list into `KV_SPLITS` roughly equal chunks.
Each chunk is processed independently by a separate CTA (Cooperative Thread Array =
thread block), producing a **partial softmax state** `(m, l, acc)`. A second reduce
kernel then combines the partials.

```
Token t's sparse KV list:  [s₀, s₁, s₂, ..., s_{k-1}]   ← values from kv_indices;
                                                             each sⱼ is a row index
                                                             into unified_kv
                            ├──────────┤ ├──────────┤ ├──────────┤
                             split 0      split 1      split 2
                            (one CTA each, run in parallel)
                                         │
                                   reduce kernel
                                         │
                                   final output[t]
```

The grid is:

```
Grid: (T, ceil(H / BLOCK_H), KV_SPLITS)   — 3D
```

Each CTA owns one token `t`, one head-block `pid_h`, and one KV-split `pid_k`. The
auto-tuned `KV_SPLITS` targets ~256 total CTAs on gfx942/gfx950 (matching MI300X's
CU count) and is rounded up to the nearest power of 2.

### 4.3 The main kernel: `_pa_decode_sparse`

Each CTA in the main kernel:

1. **Loads the query** `q[t, h_block, :]` — the per-head query vectors for this CTA's
   head block. This is a small tile `[BLOCK_H, D]` — loaded once and stays in registers.

2. **Pre-scales** the query by `softmax_scale * log2(e)`. This switches the softmax to
   base-2 (`exp2`), which avoids a transcendental function and is faster on gfx9 hardware.

3. **Computes its KV range.** The token's KV list has `kv_len` entries. Split `pid_k`
   owns the range `[tile_start, tile_end)` of tiles, where each tile is `BLOCK_K`
   slots wide. If the range is empty (the split lands beyond the token's list), early-return.

4. **Inner KV loop.** For each tile of `BLOCK_K` slots:
   - Read slot indices from `kv_indices`
   - Gather `kv_raw[BLOCK_K, D]` from `unified_kv[slot]` (each `sⱼ` is a direct row index)
   - If FP8 KV: dequantize using per-group scales `kv_scales`
   - Compute scores `scores = q · kv^T`  — shape `[BLOCK_H, BLOCK_K]`
   - Update running online softmax `(m_i, l_i)` and output accumulator `acc`

5. **Write result.** Two paths:
   - `KV_SPLITS == 1`: fold the attention sink inline (see §4.5) and write final output
     directly to `out[t, h, :]`.
   - `KV_SPLITS > 1`: write the **partial softmax state** `(m_i, l_i, acc_i)` to
     intermediate buffers. These are *not* final attention scores — they are the
     numerically stable running statistics needed to reconstruct the full softmax once
     all splits are combined (see §4.4).

#### Online softmax (flash-decode variant)

The standard softmax needs all scores at once. The **online softmax** (used in
FlashAttention) processes tiles sequentially by tracking a running maximum `m` and a
running sum of shifted exponentials `l`:

```python
# Processing tile j:
m_new = max(m_old, max(scores_j))
l_new = exp2(m_old - m_new) * l_old  +  sum(exp2(scores_j - m_new))
acc_new = exp2(m_old - m_new) * acc_old  +  exp2(scores_j - m_new) · V_j
```

At the end: `output = acc / l`. This is mathematically identical to full softmax but
processes one tile at a time — a prerequisite for flash-decode's split-K parallelism.

#### Partial softmax state written by `KV_SPLITS > 1`

After the inner loop over its assigned KV tiles, each split CTA writes three tensors
(indexed by `[t, pid_k, h]`):

```
m_partial   [T, KV_SPLITS, H]       fp32 — running max (m_i) over this split's tiles
l_partial   [T, KV_SPLITS, H]       fp32 — running sum of exp weights (l_i)
acc_partial [T, KV_SPLITS, H, D]    fp32 — weighted value accumulator (acc_i), unnormalized
```

`m_i` and `l_i` are the online-softmax statistics needed to re-weight this split's
partial accumulator relative to the global maximum found across all splits.
`acc_i` is **not** divided by `l_i` yet — normalization happens in the reduce step
after all splits' `(m, l)` are combined into a global denominator.

### 4.4 The reduce kernel: `_pa_decode_sparse_reduce`

Grid: `(T, H)` — one CTA per token-head pair.

The reduce kernel merges the `KV_SPLITS` partial states and the attention sink into a
single normalized output. It is the same log-sum-exp associativity used by the online
softmax inner loop, applied across splits instead of across tiles.

#### Step 1 — mask stale splits

Not every split necessarily processes any tiles (if `kv_len` is short relative to
`KV_SPLITS`). The number of "live" splits is derived from `kv_len`:

```python
act_num_segments = ceil(ceil(kv_len / BLOCK_K) / KV_SPLITS)  # tiles per live split
```

Splits beyond the live range have their `m_p` forced to `-inf` and `l_p`, `acc_p` to
`0`, so they contribute nothing to the combination.

#### Step 2 — combine partial states across splits

Find the global maximum across all live split maxima:

```python
m_max = max(m_p[0], m_p[1], ..., m_p[KV_SPLITS-1])
```

Re-weight each split's accumulator by how much its local maximum differs from the
global maximum (the standard log-sum-exp rescaling):

```python
for each split p:
    alpha_p = exp2(m_p[p] - m_max)   # rescale factor; 1.0 for the split that held m_max
    # dead split (m_p == -inf) → alpha_p = 0, contribution is zeroed
    l_combined  += l_p[p]  * alpha_p
    acc_combined += acc_p[p] * alpha_p
```

After this loop, `(m_max, l_combined, acc_combined)` represent the merged softmax state
for all KV splits, as if the online softmax had processed them sequentially.

#### Step 3 — fold in the attention sink

The attention sink (§4.5) is treated as one additional virtual "split" with a scalar
weight and zero value:

```python
sink = attn_sink[h] * log2(e)          # convert to base-2 domain (if USE_EXP2)
m_final  = max(m_max, sink)
alpha_kv   = exp2(m_max - m_final)     # rescale the KV partials
alpha_sink = exp2(sink  - m_final)     # weight for the sink's virtual token

l_final   = l_combined * alpha_kv  +  alpha_sink          # sink contributes to denom
acc_final = acc_combined * alpha_kv                        # sink value = 0, no acc change
```

#### Step 4 — normalize and write output

```python
out[t, h, :] = acc_final / max(l_final, 1e-30)
```

The `max(..., 1e-30)` guards against division by zero when no valid KV tokens exist.

### 4.5 The attention sink

**What it is.** In long-context language models, attention scores are observed to
accumulate disproportionately on the very first few tokens of a sequence (the "initial
tokens"). This is called the **attention sink** phenomenon — those tokens act as a
"drain" for excess probability mass that would otherwise be spread thinly across the
context.

In sparse attention this matters: if those initial tokens are not selected by top-k (they
may not be among the semantically relevant tokens), the softmax denominator changes and
the output distribution shifts. The `attn_sink[H]` per-head scalar is a learnable
correction: it is added to the softmax denominator as a **virtual key with weight 1 and
value 0**, giving the model a way to absorb the "missing mass" gracefully:

```
l_final = l + exp(attn_sink[h] - m)         # sink as virtual key
acc_final = acc + exp(attn_sink[h] - m) · 0  # value = 0, so acc unchanged
output = acc_final / l_final
```

In practice this prevents attention collapse when the selected top-k tokens have low
overall scores. The sink is folded in during the reduce step (or inline when
`KV_SPLITS == 1`).

---

## Part 5 — FP8 KV Cache on gfx942

### 5.1 Why quantize the KV cache

The KV cache for a 128k-context session can be many GB per layer. Storing it in FP8
(8 bits per element, vs 16 for bf16) cuts that in half — more sequences fit in GPU
memory simultaneously, improving throughput at the same hardware budget.

### 5.2 Block-wise quantization

Rather than a single scale per tensor (too coarse) or per-element (too expensive),
the aiter sparse kernel uses **block-wise quantization** with `GROUP_SIZE = 64`:

```
kv_scales: [total_pages, D // 64]    fp32 — one scale per 64-element block
```

During the inner loop, after gathering `kv_raw` (raw fp8 bytes), each group of 64
elements is multiplied by the corresponding fp32 scale. This is then used in the dot
product against the fp16/bf16 query.

### 5.3 FN vs FNUZ — the gfx942 gotcha

FP8 has two flavors that differ in bit interpretation:

- **FN (E4M3FN)** — the OCP standard format; used natively by **gfx950 (MI355X)**
- **FNUZ (E4M3FNUZ)** — "unsigned zero" variant with a different exponent bias; used
  natively by **gfx942 (MI300X)**

The same 8 bits mean **approximately 2× different values** between the two formats. Any
kernel that handles FP8 on gfx942 must:
1. Use `float8_e4m3fnuz` dtype (not `float8_e4m3fn`)
2. Apply the `× 2` correction when loading FN-encoded data into a FNUZ-expecting kernel
   (see the FN→FNUZ conversion in `fp8-mqa-logits` idioms)

The aiter `pa_decode_sparse` wrapper sets `dtype=float8_e4m3fnuz` when calling the
gfx942/gfx950 Triton path.

---

## Part 6 — FlyDSL MFMA Kernel Implementation

This section describes the actual implementation in
`aiter/ops/flydsl/kernels/pa_decode_sparse.py`, function
`_compile_pa_decode_sparse_mfma`. The kernel replaces the Triton `_pa_decode_sparse`
main loop with a FlyDSL kernel that issues MFMA instructions explicitly, using a
4-wave (256-thread) CTA design.

### 6.1 Thread and wave organisation

Each CTA has **256 threads** arranged as **4 waves of 64 lanes**:

```
BLOCK_THREADS = 256 = 4 waves × 64 lanes
wave_id  = tid // 64           # 0, 1, 2, 3
lane     = tid % 64            # 0..63 within the wave
lane_row = lane % 16           # 0..15 — the MFMA M/N index (head or kv row)
lane_cgrp = lane // 16         # 0..3  — column group within the wave's D-slice
```

The 256-thread design is required because each wave handles only `D/4` D-columns.
With four waves together they cover all `D` columns for both the QK and PV matrix
multiplications, as described in §6.3 and §6.5.

**Block dimensions for D = 576:**

| Parameter | Value | Meaning |
|-----------|-------|---------|
| `BLOCK_H` | 16 | heads per CTA (= `_BLOCK_H`) |
| `BLOCK_K` | 16 | KV slots per tile (= `BLOCK_H` for MFMA) |
| `N_WAVES` | 4 | waves per CTA |
| `D_PER_WAVE` | 144 | D-columns owned by each wave (`D / 4`) |
| `D_CHUNKS_W` | 9 | MFMA K-chunks per wave (`D_PER_WAVE / 16`) |

### 6.2 LDS layout and bank-conflict mitigation

The CTA uses four LDS regions.

**gfx942 LDS bank geometry:**
- **32 banks**, each **4 bytes (32 bits)** wide
- Bank index for a byte address: `bank = (byte_offset // 4) % 32`
- Bank period (one full cycle through all banks): `32 × 4 = 128 bytes`

A conflict occurs when two threads in the same wave access **different addresses that
share the same bank index** in the same clock cycle — their accesses are serialized.
Any row stride that is an exact multiple of 128 bytes maps every row to the same
set of bank indices, causing N-way conflicts on cross-row (transposed) accesses.
Each LDS region is padded to shift the row stride off the 128-byte boundary:

| Region | Unpadded shape | Unpadded row stride | Pad | Padded shape | Padded row stride | Bytes | `stride % 128` |
|--------|---------------|--------------------|----|--------------|-------------------|-------|----------------|
| `scores_lds` | `[4, 16, 16]` f32 | `BLOCK_K × 4 = 64` B | +1 f32 | `[4, 16, 17]` f32 | `(BLOCK_K+1) × 4 = 68` B | 4352 | 68 ✓ |
| `kv_lds` | `[16, 576]` bf16 | `D × 2 = 1152` B | +8 bf16 | `[16, 584]` bf16 | `(D+8) × 2 = 1168` B | 18688 | 16 ✓ |
| `acc_lds` | `[16, 576]` f32 | `D × 4 = 2304` B | +8 f32 | `[16, 584]` f32 | `(D+8) × 4 = 2336` B | 37376 | 32 ✓ |
| `slots_lds` | `[16]` i32 | — | — | `[16]` i32 | — | 64 | — |
| `valids_lds` | `[16]` i32 | — | — | `[16]` i32 | — | 64 | — |
| **Total** | | | | | | **60544** | < 64 KB ✓ |

**`kv_lds` and `acc_lds`**: the unpadded strides (1152 B and 2304 B) are exact
multiples of 128, causing 4-way conflicts on every transposed access in the PV step.
Adding 8 elements per row (`KV_PAD = ACC_PAD = 8`) shifts the stride to 1168 B and
2336 B respectively.

**`scores_lds`**: each row holds `BLOCK_K = 16` f32 scores (one per KV slot), so the
unpadded stride is only `16 × 4 = 64` bytes — also an exact multiple of 128 (64 = 128/2,
so every pair of rows aliases). Adding just **1 f32 per row** (`ALIAS_ROW_STRIDE = BLOCK_K + 1 = 17`)
gives a stride of `17 × 4 = 68` bytes. Since 68 is not a multiple of 128, adjacent rows
no longer alias. The logical shape becomes `[N_WAVES=4, BLOCK_H=16, 17]` f32 with only
`4 × 16 × 1 × 4 = 256` bytes of wasted padding in 4352 total bytes.

`scores_lds` and `p_lds` share the same base LDS address. `scores_lds` is the larger
array (`[4, 16, 17]` f32) used for cross-wave QK score communication; `p_lds`
(`[16, 17]` f32) is wave 0's sub-region of `scores_lds`, reused to hold the softmax
probability matrix P for the PV step. After the cross-wave reduce writes the full P
into `p_lds` (step 5 in §6.4), the wave-indexed offsets of `scores_lds` are no longer
needed, so the overlap is safe.

```
LDS base:
  ├── [0]         scores_lds [N_WAVES=4, BLOCK_H=16, ALIAS_ROW_STRIDE=17] f32 (4352 B)
  │              └── p_lds  [BLOCK_H=16, ALIAS_ROW_STRIDE=17] f32         (alias of wave 0)
  ├── [4352]    kv_lds     [BLOCK_K=16, KV_ROW_STRIDE=584]   bf16       (18688 B)
  ├── [23040]   slots_lds  [BLOCK_K=16]                       i32        (64 B)
  └── [23104]   valids_lds [BLOCK_K=16]                       i32        (64 B)
```

Total LDS: **23168 bytes** (< 32 KB, enabling 2 CTAs/CU vs 1 CTA/CU at 60544 bytes).
The output accumulator is stored in VGPR iter-args rather than `acc_lds` (see §6.5).

Flat index formulas (all compile-time constants after padding).
`kv_slot` is the KV-tile row index (0..BLOCK_K-1), equal to `lane_row` at runtime:
```
scores_lds[w, head, kv_slot]  →  w * BLOCK_H * ALIAS_ROW_STRIDE + head * ALIAS_ROW_STRIDE + kv_slot
p_lds[head, kv_slot]          →  head * ALIAS_ROW_STRIDE + kv_slot   (= scores_lds[0, head, kv_slot])
kv_lds[kv_slot, d_col]        →  kv_slot * KV_ROW_STRIDE + d_col     (KV_ROW_STRIDE = D + 8)
```

### 6.3 Step-by-step per KV tile: QK matrix multiplication

**Inputs:** Q tile `[BLOCK_H=16, D=576]` bf16 in VRAM (per-head query), KV tile
`[BLOCK_K=16, D=576]` bf16 gathered from `unified_kv` via slot indices.

**Goal:** compute the score matrix `S[BLOCK_H, BLOCK_K]` = `Q @ K^T`.

Each wave is responsible for one D-slice of width `D_PER_WAVE = D/4 = 144`.

```
wave 0: d ∈ [0,   144)  →  D_CHUNKS_W=9 MFMA K-chunks
wave 1: d ∈ [144, 288)
wave 2: d ∈ [288, 432)
wave 3: d ∈ [432, 576)
```

Within a wave, the QK MFMA loop runs `D_CHUNKS_W = 9` iterations (one per 16-column
chunk of this wave's D-slice). Per iteration at chunk `dc`:

```
col_off = d_wave_base + dc * 16 + lane_cgrp * 4   # D-column base for this lane

# Q A-frag: Q[head, col_off..+3]  (4 bf16 elements per lane, 4 consecutive M-rows)
flat_q = t * H * D + h_lane * D + col_off
q_v4   = buffer_load(q_rsrc, flat_q // 2, vec_width=2)          # 4 bf16 as 2 i32

# KV B-frag for QK: KV[slot[lane_row], col_off..+3]  (same slot for all groups)
flat_kv = safe_slot * D + col_off
kv_v4   = buffer_load(kv_rsrc, flat_kv // 2, vec_width=2)       # 4 bf16 as 2 i32

# Write KV into kv_lds for PV reuse (transposed access needed there)
for v in 0..3:
    kv_lds[lane_row, col_off + v] = kv_v4[v]

# QK partial MFMA (accumulates into c_frag over D_CHUNKS_W iterations)
c_frag = mfma_f32_16x16x16_bf16(q_v4, kv_v4, c_frag)
```

**MFMA C-frag layout.** After `mfma_f32_16x16x16_bf16(A, B, C)`:
```
c_frag[v]  ↔  C[M = lane_cgrp*4 + v,  N = lane_row]
           ↔  S_partial[head = lane_cgrp*4+v,  kv_slot = lane_row]
```
Each lane accumulates a partial score for 4 (head, kv_slot) pairs over its wave's D-slice.
After all 9 iterations: `c_frag[v]` = sum of D/4 contributions to `S[head=lane_cgrp*4+v, kv_slot=lane_row]`.

**Simultaneously**, each lane writes its 4 KV bf16 values into `kv_lds[lane_row, col_off..+3]`.
Over all 9 iterations and 4 waves, this fills `kv_lds[0..15, 0..575]` — the complete KV
tile in row-major order. This staging is necessary because the PV step requires the KV
matrix in transposed access order relative to QK (see §6.5).

### 6.4 Online softmax with cross-wave LDS communication

After the QK loop, each wave holds a **partial score** `c_frag[v]` for only `D/4`
of the inner dimension. The full score `S[head, kv_slot] = Σ_{wave} c_frag_wave[v]` requires
summing the 4 waves' contributions. This is done via `scores_lds`.

#### Step 3 — write partial QK scores to scores_lds

All 4 waves write their partial c-frags to `scores_lds`:

```
scores_lds[wave_id, head, kv_slot] = c_frag[v]
  where head = lane_cgrp*4 + v,  kv_slot = lane_row

flat index = wave_id * BLOCK_H * ALIAS_ROW_STRIDE
           + (lane_cgrp*4+v) * ALIAS_ROW_STRIDE
           + lane_row
```

A `gpu.barrier()` ensures all waves have written before the next step.

#### Step 4 — cross-wave reduce, scale, and online softmax

All 4 waves independently perform the same reduction and softmax (redundant but avoids
further synchronisation). For each of its 4 (head, kv_slot) pairs:

```python
# Sum 4 wave partials → full score for (head=lane_cgrp*4+v, kv_slot=lane_row)
s_full = sum(scores_lds[w, head, kv_slot] for w in 0..3)

# Gate by validity (slot == -1 → mask to -inf)
s_scaled = s_full * softmax_scale * log2(e)          # base-2 domain
s_gated  = s_scaled if slot_valid else -inf

# Butterfly max over lane_row (kv_slot) dimension — finds max score across all 16 KV slots
# Each lane XOR-exchanges with neighbours at distance 1, 2, 4, 8
s_max = butterfly_max(s_gated, xor=[1, 2, 4, 8])    # wave-wide reduce

# Online softmax update (one of 4 independent per-head states)
m_new   = max(m_i, s_max)
alpha   = exp2(m_i - m_new)      # rescale factor for old accumulator
p_j     = exp2(s_gated - m_new)  # probability for this KV slot

# Butterfly sum over lane_row dimension — normalisation denominator
sum_p   = butterfly_sum(p_j, xor=[1, 2, 4, 8])

l_new   = alpha * l_i + sum_p
```

`m_i` and `l_i` are **ForOp iter-args** — they carry the running softmax state across
KV tiles. Each lane maintains 4 independent `(m_i, l_i)` pairs, one per head it owns
(`head = h_base + lane_cgrp*4 + v`).

**Base-2 softmax.** Q is pre-scaled by `softmax_scale * log2(e)` before the tile loop.
All exponentials use `exp2` (hardware instruction `v_exp2_f32`) rather than `exp`.
This is numerically equivalent (`exp(x) = exp2(x / ln2) = exp2(x * log2(e))`) but
avoids a separate multiply inside the inner loop.

**The butterfly reduction.** With `BLOCK_THREADS_MFMA = 256`, the butterfly uses XOR
offsets 1, 2, 4, 8 — these span all 16 lane_row values (the kv_slot dimension) within
each group of 16 consecutive lanes. Since all 256 threads run identical butterfly code
with `width=256`, each group of 16 lanes sharing the same `lane_cgrp` sees the correct
max/sum for its head.

#### Step 5 — write P to p_lds for PV

After the softmax update, each lane holds `p_j = exp2(score - m_new)` for its `(head,
kv_slot)` pair. All waves write the same values (the reduce in step 4 is identical across
all waves), so the writes are redundant but harmless:

```
p_lds[head, kv_slot] = p_j
  where head = lane_cgrp*4 + v,  kv_slot = lane_row

flat index = (lane_cgrp*4+v) * ALIAS_ROW_STRIDE + lane_row
```

A second `gpu.barrier()` makes `p_lds` and `kv_lds` visible to all waves before the
PV step.

**State summary at this point:**

| Value | Location | Owner |
|-------|----------|-------|
| `m_new[4]`, `l_new[4]` | VGPRs (ForOp iter-args) | each lane, 4 per-head values |
| `alpha[4]` | VGPRs | each lane, 4 rescale factors |
| P tile `[16, 17]` f32 | `p_lds` (= scores_lds base) | LDS, all waves share |
| KV tile `[16, 584]` bf16 | `kv_lds` | LDS, all waves share |
| Acc `[D_CHUNKS_W=9]` × `vec<4,f32>` | VGPRs (ForOp iter-args) | each lane, 9 acc chunks |

### 6.5 PV matrix multiplication

**Goal:** update the output accumulator `acc[BLOCK_H, D] += P[BLOCK_H, BLOCK_K] @ KV[BLOCK_K, D]`.

Each wave handles its D-slice (`D_PER_WAVE = D/4` columns) for all 16 heads. The loop
runs `D_CHUNKS_W = 9` MFMA iterations per wave.

#### Reading the PV operands from LDS

The MFMA `mfma_f32_16x16x16_bf16(A, B, C)` with M=16, K=16, N=16 requires:
- **A-frag**: `P[M=lane_row, K=lane_cgrp*4..+3]` — lane reads 4 elements from P's
  row indexed by `lane_row`, at KV columns `lane_cgrp*4..lane_cgrp*4+3`.
- **B-frag**: `KV[K=lane_cgrp*4..+3, N=lane_row]` — lane reads 4 elements from KV's
  columns indexed by the wave's current D-column, at KV rows `lane_cgrp*4..+3`.
- **C-frag**: `acc[M=lane_cgrp*4..+3, N=d_col]` — 4 f32 accumulator values.

**A-frag from p_lds (no transpose needed):**
```
p_flat = lane_row * ALIAS_ROW_STRIDE + (lane_cgrp*4 + v)
a_frag[v] = bf16(p_lds[p_flat])
```
P is stored in row-major order as `P[head, kv_slot]`. Reading `P[head=lane_row, kv_slot=kv_k]`
at flat index `lane_row * ALIAS_ROW_STRIDE + kv_k` gives the MFMA A-frag element
needed: `A[M=lane_row, K=kv_k]` = `P[head=lane_row, kv_slot=kv_k]`.

**B-frag from kv_lds (transposed access):**
```
kv_slot_pv = lane_cgrp*4 + v          # KV slot index for PV
d_col      = d_wave_base + dc*16 + lane_row
kv_flat_pv = kv_slot_pv * KV_ROW_STRIDE + d_col
b_frag[v]  = kv_lds[kv_flat_pv]
```

`kv_lds[kv_slot, d_col]` holds `KV[kv_slot, d_col]`. The QK step wrote
`kv_lds[lane_row, col_off+v]` = `KV[kv_slot=lane_row, d=col_off+v]`.
The PV step reads `kv_lds[kv_slot_pv, d_col]` = `KV[kv_slot=lane_cgrp*4+v, d=d_col]`.
This is a **transposed access pattern** — QK indexed by `(kv_slot=lane_row, d=lane_cgrp*4..+3)`,
PV indexes by `(kv_slot=lane_cgrp*4..+3, d=lane_row)` — which is why the KV tile must
live in LDS rather than VGPR registers: VGPR data cannot be shared across lanes.

**C-frag (VGPR iter-arg, rescale, MFMA):**
```
acc_frag = acc_i[dc]                        # VGPR iter-arg from previous tile

# Rescale with per-head alpha (online softmax correction)
acc_frag_scaled[v] = acc_frag[v] * alpha[v]

# PV MFMA: acc += P @ KV for this D-chunk
new_acc_frag = mfma_f32_16x16x16_bf16(a_frag, b_frag, acc_frag_scaled)

new_acc.append(new_acc_frag)                # collected; yielded as iter-arg at tile end
```

The `alpha[v] = exp2(m_old[v] - m_new[v])` rescale is applied before the MFMA. This
implements the online-softmax update `acc_new = alpha * acc_old + p * V` in a single
MFMA call: after multiplying `acc_old` by `alpha`, the MFMA adds `p * V` (via the
`P @ KV` product contribution) to it.

No barrier is needed after the PV loop — the accumulator lives in VGPRs, so there is
no LDS writeback to make visible.

#### Why VGPRs work for acc

Each lane holds `D_CHUNKS_W = 9` acc fragments of type `vec<4, f32>`. That is
`9 × 4 × 4 = 144` bytes = **36 VGPRs** per lane — well within the 512-VGPR budget on
gfx942. With 4 waves × 64 lanes = 256 threads per CTA, each wave's share is 36 VGPRs
for acc, leaving ample room for Q/KV loads, softmax scalars, and loop temporaries.

Eliminating `acc_lds` saves `BLOCK_H × (D+8) × 4 = 16 × 584 × 4 = 37376` bytes of
LDS, reducing total LDS from 60544 → **23168 bytes** and enabling 2 CTAs/CU instead
of 1, which is the primary source of the occupancy and performance improvement.

### 6.6 ForOp structure and loop-carried state

The KV tile loop is an `scf.ForOp` from `tile_start` to `tile_end`. Both the online
softmax scalars and the output accumulator are loop-carried as ForOp iter-args.

```
ForOp iter-args: [m_i[0], m_i[1], m_i[2], m_i[3],       # 4 f32: running max per head
                  l_i[0], l_i[1], l_i[2], l_i[3],        # 4 f32: running sum per head
                  acc_dc0, acc_dc1, ..., acc_dc8]          # 9 × vec<4,f32>: accumulator per D-chunk
                                                           # 44 iter-args total
```

Each lane holds 4 `(m_i, l_i)` pairs — one per head it serves
(`head = h_base + lane_cgrp*4 + v`). The 9 accumulator fragments `acc_dc[i]` are
`vec<4, f32>` — each holds 4 f32 output values for the 4 heads owned by this lane at
D-chunk `i`. Cross-lane communication uses the butterfly reduction and LDS as described in §6.4.

#### Initialization

`KV_SPLITS == 1` (single-pass, attention sink folded inline):
```
for v in 0..3:
    h_v    = h_base + lane_cgrp*4 + v
    sink_v = attn_sink[h_v]                     # per-head learnable log-weight
    m_i[v] = sink_v * log2(e)                   # virtual token with weight 1
    l_i[v] = 1.0                                # exp2(sink - m_i) = 1
acc_dc[0..8] = vec<4,f32>(0.0)                  # zero accumulator in VGPRs
```

The sink initialisation absorbs it as if it were the first KV token (score =
`attn_sink[h]`, value = 0). Every subsequent token is compared against this initial
maximum, so the sink's weight naturally enters the softmax denominator.

`KV_SPLITS > 1` (partial emit, sink handled by reduce kernel):
```
m_i[v] = -inf     # no prior max
l_i[v] = 0.0      # no prior mass
acc_dc[0..8] = vec<4,f32>(0.0)
```

#### Termination

After the ForOp: `m_final[4]`, `l_final[4]` from the last iteration's iter-args.

**`KV_SPLITS == 1` — direct output:**
```python
for v in 0..3:
    h_v = h_base + lane_cgrp*4 + v
    for dc in 0..D_CHUNKS_W:
        d_col   = d_wave_base + dc*16 + lane_row
        acc_val = vector.extract(acc_final[dc], v)    # from VGPR iter-arg
        out_val = bf16(acc_val / max(l_final[v], 1e-30))
        out[t, h_v, d_col] = out_val   (if h_v < H)
```

**`KV_SPLITS > 1` — partial emit:**
```python
for v in 0..3:
    h_v     = h_base + lane_cgrp*4 + v
    ml_flat = t * KV_SPLITS * H + pid_k * H + h_v
    if lane_row == 0:                           # one lane writes the scalar stats
        m_partial[ml_flat] = m_final[v]
        l_partial[ml_flat] = l_final[v]
    for dc in 0..D_CHUNKS_W:
        d_col   = d_wave_base + dc*16 + lane_row
        acc_val = vector.extract(acc_final[dc], v)    # from VGPR iter-arg
        acc_partial[ml_flat * D + d_col] = acc_val
```

The Triton `_pa_decode_sparse_reduce` kernel then combines these partial states across
all KV splits and folds in the attention sink (§4.4).

### 6.7 Tile-level execution timeline

```
[Tile loop body — 1 KV tile of BLOCK_K=16 slots]

1. WAVE 0 ONLY (lane_cgrp==0): load slot indices into slots_lds / valids_lds
   (16 slots × 1 lane each, via buffer_load)
   gpu.barrier()  — slots visible to all 256 threads

2. ALL WAVES: QK MFMA loop (D_CHUNKS_W=9 iterations per wave)
   Each iteration:
   a. Load Q A-frag from VRAM  (buffer_load, 4 bf16)
   b. Load KV B-frag from VRAM (buffer_load, 4 bf16)
   c. Write KV to kv_lds[lane_row, col_off..+3]  (4 bf16 → LDS)
   d. Issue mfma_f32_16x16x16_bf16(q_v4, kv_v4, c_frag)  (accumulates c_frag)

3. ALL WAVES: write c_frag to scores_lds[wave_id, head, kv_slot]
   gpu.barrier()  — all partial QK scores + kv_lds writes visible

4. ALL WAVES (independently, same result):
   a. Sum 4 wave partials from scores_lds → full score[head, kv_slot]
   b. Apply validity mask (slot == -1 → -inf)
   c. Butterfly max over lane_row: s_max
   d. Online softmax: m_new = max(m_i, s_max); alpha; p_j; butterfly sum_p; l_new
   e. Write p_j to p_lds[head, kv_slot]
   gpu.barrier()  — p_lds visible for PV A-frag reads

5. ALL WAVES: PV MFMA loop (D_CHUNKS_W=9 iterations per wave)
   Each iteration (dc = 0..8):
   a. Load A-frag: p_lds[lane_row * ALIAS_ROW_STRIDE + kv_k]  (4 bf16 from p_lds)
   b. Load B-frag: kv_lds[kv_slot_pv * KV_ROW_STRIDE + d_col]  (4 bf16 from kv_lds)
   c. Fetch C-frag from VGPR iter-arg acc_dc[dc]              (4 f32, no LDS load)
   d. Rescale C-frag by alpha[v]
   e. Issue mfma_f32_16x16x16_bf16(a_frag, b_frag, rescaled_c)  → new_acc[dc]
   (no barrier needed — acc lives in VGPRs)

→ yield (m_new[4], l_new[4], new_acc[0..8]) as next iteration's iter-args
```

**MFMA count per tile:** `D_CHUNKS_W × N_WAVES × 2 (QK + PV) = 9 × 4 × 2 = 72`
instructions — identical to the Triton analysis in §4.0.

### 6.8 MFMA operand layout summary

The `mfma_f32_16x16x16_bf16` instruction computes `C[16,16] += A[16,16] × B[16,16]`.
On gfx942 with a 64-lane wave, the operands are distributed as:

```
C-frag: lane l carries C[lane_cgrp*4 + v, lane_row]  for v ∈ {0,1,2,3}
A-frag: lane l carries A[lane_cgrp*4 + v, K_col]     for v ∈ {0,1,2,3}  (4 consecutive M-rows)
B-frag: lane l carries B[K_row, lane_row + ?]         (column of B)
```

The kernel uses this layout as follows:

| Step | M (row) | K (inner) | N (col) | A source | B source | C source |
|------|---------|-----------|---------|----------|----------|----------|
| QK | head = `lane_cgrp*4+v` | d chunk | kv_slot = `lane_row` | Q`[head, d]` from VRAM | KV`[kv_slot, d]` from VRAM | partial score (VGPR) |
| PV | head = `lane_row` | kv_slot = `lane_cgrp*4+v` | d = `d_col` | P`[head, kv_slot]` from `p_lds` | KV`[kv_slot, d]` from `kv_lds` | acc`[head, d]` from VGPR iter-arg |

Note the role swap between QK and PV: in QK, `lane_row` indexes the kv_slot (N) dimension;
in PV, `lane_row` indexes the head (M) dimension. This reversal is why the P matrix
must be stored in LDS in row-major `P[head, kv_slot]` order — the PV A-frag reads
`P[head=lane_row, kv_slot=kv_k]` in row-major order at `lane_row * ALIAS_ROW_STRIDE + kv_k`,
which matches exactly how the softmax wrote `P[head=lane_cgrp*4+v, kv_slot=lane_row]` at
`(lane_cgrp*4+v) * ALIAS_ROW_STRIDE + lane_row`.

### 6.9 Performance (T=128, H=128, D=576, kv\_len=2048)

| Kernel | Time | BW | TFLOPS | vs Triton |
|--------|------|-----|--------|-----------|
| FlyDSL 4-wave + VGPR acc | ~1.3 ms | ~250 GB/s | **29.75** | **2.07×** |
| FlyDSL 4-wave + LDS padding | 2.096 ms | 162.6 GB/s | 18.5 | 1.28× |
| Triton `_pa_decode_sparse` | 2.687 ms | 126.8 GB/s | 14.4 | 1.00× |

The VGPR accumulation design (Priority 1) eliminated `acc_lds` (37376 bytes) and moved
the output accumulator into ForOp VGPR iter-args (`9 × vec<4,f32>` per lane = 36 VGPRs).
This reduced LDS from 60544 → **23168 bytes**, enabling 2 CTAs/CU and producing a
**2.07× speedup** over Triton at T=128.

LDS row padding (`KV_PAD=8`, `ALIAS_ROW_STRIDE=BLOCK_K+1`) remains active to eliminate
bank conflicts on the transposed `kv_lds` accesses in the PV step.

---

## Glossary

- **Decode step** — one token-generation step; the model produces one new token given
  all previous tokens.
- **KV cache** — stored keys and values from all past tokens, reused at each decode step.
- **Paged KV cache** — KV cache stored in fixed-size pages, scattered in GPU memory, with
  a block table mapping logical to physical pages.
- **Unified KV pool** — a flat array of KV slots; indexed by scatter/gather index lists
  rather than a per-request block table.
- **MLA (Multi-head Latent Attention)** — DeepSeek's KV compression: caches a small
  latent `c` per token, reconstructs full K/V on the fly.
- **Absorbed MLA** — optimization where the up-projection `W_UK` is folded into the query
  projection; eliminates K_content materialization; gives `D = kv_lora_rank + rope_rank`.
- **DSA (DeepSeek Sparse Attention)** — attends to top-k selected tokens per query
  instead of all tokens; enabled by the lightning indexer (Stage 1).
- **Top-k selection (Stage 2)** — picks the k highest-scoring past tokens from the
  indexer output; produces the sparse `kv_indices` list.
- **Sparse MLA decode (Stage 3)** — the actual attention over the top-k selected tokens;
  the focus of this ticket.
- **Split-K (flash-decode)** — partitions a token's KV list into `KV_SPLITS` chunks,
  each processed by a separate CTA, then recombined; enables CU parallelism for small T.
- **Online softmax** — incremental softmax that processes one KV tile at a time without
  storing all scores; requires running `(m, l)` statistics; enables flash-decode.
- **Attention sink** — the phenomenon where initial tokens accumulate excess attention
  probability; corrected by a learnable per-head sink scalar in the softmax denominator.
- **FP8 (E4M3)** — 8-bit float format; `FNUZ` variant on gfx942, `FN` variant on gfx950.
- **BLOCK_K** — KV tile width in the sparse decode inner loop (16 or 32 in current code).
- **KV_SPLITS** — number of split-K partitions; auto-tuned to fill ~256 CTAs on gfx942.
- **gfx942 (MI300X, CDNA3)** — the target GPU architecture; 304 CUs, 5.3 TB/s HBM.
- **gfx950 (MI355X, CDNA4)** — the comparison GPU; 256 CUs, 8 TB/s HBM; has `async_copy`.
- **FlyDSL** — AMD's Python DSL for GPU kernels; generates MLIR → AMDGPU ISA; more
  control than Triton, more portable than hand-written assembly.
- **`pa_decode_sparse`** — the Triton split-K sparse paged attention decode kernel in aiter.
- **`mla_decode_fwd`** — the production MLA decode entry point in `aiter/mla.py`; uses
  ASM stage-1 + `mla_reduce_v1` C++ kernel.
- **`mla_reduce_v1`** — C++ kernel that combines split-K partials for the ASM path
  (under separate investigation in SILOTIGER-611).
- **CTA** — Cooperative Thread Array; one GPU thread block, runs on one CU.
- **CU** — Compute Unit; MI300X has 304. One CTA per CU at a time (at typical occupancy).
- **VGPR/SGPR/AGPR** — Vector/Scalar/Accumulator General-Purpose Registers; limited per CU.
- **Occupancy** — number of active waves (warps) per CU; limited by register pressure.
- **LDS** — Local Data Share; fast on-chip shared memory per CTA (~64 KB on gfx942).
- **HBM** — High Bandwidth Memory; large but slower GPU DRAM.

---

## References

**DeepSeek model line**
- DeepSeek-AI, [*DeepSeek-V2*](https://arxiv.org/abs/2405.04434) (MLA), 2024
- DeepSeek-AI, [*DeepSeek-V3 Technical Report*](https://arxiv.org/abs/2412.19437), 2024/2025
- Yuan et al., [*Native Sparse Attention (NSA)*](https://arxiv.org/abs/2502.11089), 2025

**Efficient attention kernels**
- Dao et al., [*FlashAttention*](https://arxiv.org/abs/2205.14135), 2022 (online softmax)
- Dao, [*FlashAttention-2*](https://arxiv.org/abs/2307.08691), 2023
- Shazeer, [*Flash-Decoding*](https://crfm.stanford.edu/2023/10/12/flashdecoding.html), 2023 (split-K decode)

**FP8 and AMD hardware**
- Micikevicius et al., [*FP8 Formats for Deep Learning*](https://arxiv.org/abs/2209.05433), 2022
- ROCm, [*HIP FP8 Numbers*](https://rocm.docs.amd.com/projects/HIP/en/latest/reference/fp8_numbers.html)
- AMD, [*Instinct MI300 CDNA3 ISA Reference Guide*](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-mi300-cdna3-instruction-set-architecture.pdf)

**Aiter implementation**
- `aiter/ops/triton/attention/pa_decode_sparse.py` — kernel launcher/wrapper
- `aiter/ops/triton/_triton_kernels/attention/pa_decode_sparse.py` — Triton kernels
- `aiter/mla.py` — `mla_decode_fwd` (ASM + reduce production path)
- `op_tests/triton_tests/attention/test_pa_decode_sparse.py` — unit tests
- `op_tests/test_mla_sparse.py` — integration test

---