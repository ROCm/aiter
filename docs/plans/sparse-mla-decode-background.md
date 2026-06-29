

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