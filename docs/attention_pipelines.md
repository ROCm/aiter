# Attention Pipelines in AIter

This document describes the attention kernel pipelines available in AIter,
their parallelization strategies, parameters, and when each is used.

---

## Feature Matrix

| Feature | Triton 2D | Triton 3D | CK-UA | CK-SK | CK-PK | CK-Fwd |
|---------|-----------|-----------|-------|-------|-------|--------|
| KV splitting | No | Yes (segments) | No | Yes (num_splits) | No | No |
| Paged KV cache | Yes | Yes | Yes | Yes | Yes | No |
| Sliding window | Yes | Yes | No | Yes | Yes | Yes |
| Causal mask | Yes | Yes | Yes | Yes | Yes | Yes |
| GQA | Yes | Yes | Yes (head-merge) | Yes (head-merge) | Yes | Yes |
| Softcap | Yes | Yes | No | Yes | Yes | Yes |
| ALiBi | Yes | Yes | No | No | No | No |
| Sinks | Yes | Yes | No | Yes | Yes | Yes |
| FP8 output | Yes | No | No | No | No | No |
| Compiled hdim | Any | Any | 64, 128 | 32-256 | 32-256 | 32-256 |
| Data types | fp16/bf16 | fp16/bf16 | fp16/bf16 | fp16/bf16/fp8 | fp16/bf16 | fp16/bf16/fp8 |

---

## 1. Triton 2D -- Single-Pass Unified Attention

**What:** Single-pass attention over the full KV range. Each workgroup processes
one Q-tile for one KV-head, iterating over all KV tiles sequentially.

**When to use:** Short KV sequences, sliding window attention, or when there are
enough Q-blocks x KV-heads to fill the GPU without KV splitting.

### Grid

```
grid = (num_kv_heads, total_num_q_blocks)

total_num_q_blocks = q_tokens // BLOCK_Q + num_seqs    (upper bound with ceil padding)
BLOCK_Q = BLOCK_M // num_queries_per_kv
```

Each workgroup `(kv_head, q_block)` processes one tile of Q tokens against ALL
KV tokens for that KV head.

**Example:** batch=4 decode (q_tokens=4), num_kv_heads=8, BLOCK_Q=2:
```
total_num_q_blocks = 4 // 2 + 4 = 6
grid = (8, 6) = 48 workgroups
```

### Parameters

```
Tensors:
  query_ptr          [total_q_tokens, num_query_heads, head_size]    bf16
  key_cache_ptr      [num_blks, block_size, num_kv_heads, head_size] bf16
  value_cache_ptr    [num_blks, block_size, num_kv_heads, head_size] bf16
  output_ptr         [total_q_tokens, num_query_heads, head_size]    bf16
  block_tables_ptr   [num_seqs, max_num_blocks_per_seq]              int32
  seq_lens_ptr       [num_seqs]                                      int32
  query_start_len_ptr [num_seqs + 1]                                 int32  (cu_seqlens_q)

Scalars:
  scale              float     softmax scale (1/sqrt(hdim))
  softcap            float     logits soft cap (0 = disabled)
  num_query_heads    int
  num_queries_per_kv int       GQA ratio
  BLOCK_SIZE         int       page block size (e.g. 32)
  HEAD_SIZE          int       head dimension (e.g. 64)
  SLIDING_WINDOW     int       1 + window_left (0 = no window)
  num_seqs           int

Compile-time flags:
  USE_ALIBI_SLOPES, USE_SOFTCAP, USE_SINKS, USE_FP8_OUTPUT, ALL_DECODE
```

### Pseudocode

```
kernel_2d(kv_head_idx, q_block_idx):
    # Find which sequence this Q block belongs to
    seq_idx = binary_search(query_start_len, q_block_idx)
    q_start = query_start_len[seq_idx]
    q_len   = query_start_len[seq_idx + 1] - q_start
    kv_len  = seq_lens[seq_idx]

    # Load Q tile (BLOCK_M rows x HEAD_SIZE cols)
    Q_tile = load_q(q_start + local_q_offset, kv_head_idx)

    # Iterate over KV tiles
    acc = zeros(BLOCK_M, HEAD_SIZE)     # output accumulator
    m = -inf                             # running max
    l = 0                                # running sum of exp

    for kv_tile_idx in range(0, kv_len, TILE_SIZE):
        # Page lookup
        page_idx = block_tables[seq_idx, kv_tile_idx // BLOCK_SIZE]
        K_tile = load_k(page_idx, kv_head_idx, kv_tile_idx % BLOCK_SIZE)
        V_tile = load_v(page_idx, kv_head_idx, kv_tile_idx % BLOCK_SIZE)

        # QK^T
        S = Q_tile @ K_tile.T * scale

        # Apply causal mask + sliding window
        S = apply_mask(S, q_positions, kv_tile_idx, SLIDING_WINDOW)

        # Online softmax update
        m_new = max(m, row_max(S))
        correction = exp(m - m_new)
        P = exp(S - m_new)
        l = l * correction + row_sum(P)
        acc = acc * correction + P @ V_tile
        m = m_new

    # Normalize
    output = acc / l
    store(output_ptr, seq_idx, kv_head_idx)
```

### Numerical Example

```
Setup: batch=1, q_len=1 (decode), kv_len=4, hdim=2, block_size=2, num_kv_heads=1

Q = [0.5, 0.3]                          # shape [1, 2]

block_table = [5, 2]                     # 2 pages of 2 tokens each
K_cache[page 5] = [[0.1, 0.2],          # kv token 0
                    [0.3, 0.4]]          # kv token 1
K_cache[page 2] = [[0.5, 0.6],          # kv token 2
                    [0.7, 0.8]]          # kv token 3

Tile 0 (page 5):  S = Q @ K[0:2].T = [0.05+0.06, 0.15+0.12] = [0.11, 0.27]
Tile 1 (page 2):  S = Q @ K[2:4].T = [0.25+0.18, 0.35+0.24] = [0.43, 0.59]

Full S = [0.11, 0.27, 0.43, 0.59] * scale
P = softmax(S)
output = P @ V_full
```

---

## 2. Triton 3D -- Split-KV Unified Attention

**What:** Splits the KV sequence into `NUM_SEGMENTS` parallel segments. Each
segment computes a partial softmax (local max + exp-sum + weighted output).
A separate reduce kernel merges the segments using the log-sum-exp trick.

**When to use:** Long KV sequences with low batch size, where Triton 2D doesn't
have enough workgroups to fill the GPU.

### Grid

```
# Attention kernel
grid = (total_num_q_blocks, num_kv_heads, NUM_SEGMENTS_PER_SEQ)

# Reduce kernel
grid = (total_q_tokens, num_query_heads)
```

Each attention workgroup `(q_block, kv_head, segment)` processes a slice of
KV tokens `[segment * kv_len/NUM_SEGMENTS, (segment+1) * kv_len/NUM_SEGMENTS)`.

**Example:** batch=1 decode, kv_len=8192, num_kv_heads=8, NUM_SEGMENTS=16:
```
attention grid = (1, 8, 16) = 128 workgroups
reduce grid    = (1, 64) = 64 workgroups      (64 = 8 kv_heads * 8 q_per_kv)
```

### Parameters

Same as Triton 2D, plus:
```
Intermediate buffers (allocated by host):
  segm_output   [total_q_tokens, num_query_heads, NUM_SEGMENTS, HEAD_SIZE_PADDED]  fp32
  segm_max      [total_q_tokens, num_query_heads, NUM_SEGMENTS]                    fp32
  segm_expsum   [total_q_tokens, num_query_heads, NUM_SEGMENTS]                    fp32

NUM_SEGMENTS_PER_SEQ     int    (power of 2, 8-128, chosen by heuristic)
```

### Pseudocode

```
# --- Attention kernel (per segment) ---
kernel_3d(q_block_idx, kv_head_idx, segment_idx):
    kv_start = segment_idx * (kv_len // NUM_SEGMENTS)
    kv_end   = (segment_idx + 1) * (kv_len // NUM_SEGMENTS)

    # Same flash-attention loop as 2D but only over [kv_start, kv_end)
    acc, m, l = flash_attn_loop(Q_tile, K[kv_start:kv_end], V[kv_start:kv_end])

    # Store partial results
    segm_output[q_idx, head, segment] = acc       # unnormalized
    segm_max[q_idx, head, segment]    = m         # local max
    segm_expsum[q_idx, head, segment] = l         # local exp-sum

# --- Reduce kernel ---
reduce_segments(q_idx, head_idx):
    # Merge all segments using log-sum-exp
    global_max = max over segments of segm_max[q, h, :]

    total_sum = 0
    output = zeros(HEAD_SIZE)
    for seg in range(NUM_SEGMENTS):
        correction = exp(segm_max[q, h, seg] - global_max)
        total_sum += segm_expsum[q, h, seg] * correction
        output += segm_output[q, h, seg] * correction

    output /= total_sum
    store(output_ptr[q, h], output)
```

### Numerical Example

```
Setup: kv_len=4, NUM_SEGMENTS=2

Segment 0 (tokens 0-1):  S0 = [0.11, 0.27]
  m0 = 0.27,  l0 = exp(0.11-0.27) + exp(0) = 0.852 + 1.0 = 1.852
  acc0 = softmax([0.11, 0.27]) @ V[0:2]

Segment 1 (tokens 2-3):  S1 = [0.43, 0.59]
  m1 = 0.59,  l1 = exp(0.43-0.59) + exp(0) = 0.852 + 1.0 = 1.852
  acc1 = softmax([0.43, 0.59]) @ V[2:4]

Reduce:
  global_max = max(0.27, 0.59) = 0.59
  c0 = exp(0.27 - 0.59) = 0.726
  c1 = exp(0.59 - 0.59) = 1.0
  total = l0*c0 + l1*c1 = 1.852*0.726 + 1.852*1.0 = 1.345 + 1.852 = 3.197
  output = (acc0*c0 + acc1*c1) / total
```

---

## 3. CK-UA -- CK Tile Unified Attention

**What:** Single-kernel paged-KV attention from Composable Kernel (CK Tile).
Key optimization: merges GQA head groups into the M (query) tile dimension,
giving better per-workgroup efficiency than Triton at moderate batch sizes.

> For how UA's `UnifiedAttentionPipeline` differs *internally* from the
> `BlockFmhaFwdV3Pipeline` it was forked from (signature diff, side-by-side
> Python pseudocode of the core loop, what UA added vs stripped), see
> [`unified_attention_vs_v3_pipeline.md`](unified_attention_vs_v3_pipeline.md).

**When to use:** Decode with moderate batch (128-256 seqs for 8 KV heads on
256 CUs). No sliding window. Selected automatically by the occupancy-based
selector when `cu_count*4 <= num_kv_heads*num_seqs <= cu_count*8`.

### Grid

```
# Decode grid (decode tier)
grid = dim3(num_kv_heads, num_seqs)
# Each workgroup (kv_head, seq) processes 1 seq, 1 kv_head, all q_heads in that group

# Prefill grid (medium/large tier)
grid = dim3(num_kv_heads * total_num_q_blocks)
# 1D grid, binary search to find seq_idx from q_block
```

**Example:** batch=256 decode, num_kv_heads=8, GQA-8 (64 q-heads):
```
grid = dim3(8, 256) = 2048 workgroups
Each workgroup: 1 kv_head, 1 seq, 8 q-heads merged into M dimension
```

### Parameters

```
Tensors:
  q_ptr                [num_tokens, num_heads_q, head_size]               bf16
  k_ptr                [num_blks, page_blk_size, num_kv_heads, head_size] bf16
  v_ptr                [num_blks, page_blk_size, num_kv_heads, head_size] bf16
  o_ptr                [num_tokens, num_heads_q, head_size]               bf16
  block_tables_ptr     [num_seqs, max_num_blocks_per_seq]                 int32
  seq_lens_ptr         [num_seqs]                                         int32
  query_start_len_ptr  [num_seqs + 1]                                     int32

Scalars:
  mask_type      int     0=no_mask, 2=causal
  scale_s        float   softmax scale
  scale          float   Q scale (quantization, usually 1.0)
  scale_k        float   K scale (quantization, usually 1.0)
  scale_v        float   V scale (quantization, usually 1.0)
  scale_out      float   output scale (usually 1.0)
  page_blk_size  int     page block size (32 or 64)
  num_seqs       int
  max_seqlen_q   int     (0 = unknown; used for tier selection)

Compile-time:
  kBlockM        int     Q tile size (16, 32, 64, 128, 256 depending on tier)
  kBlockQ        int     = kBlockM / num_queries_per_kv
  kPageBlockSize int     = page_blk_size (template parameter)
  kHeadDim       int     64 or 128
  IsMasking      bool
```

### Tile Tiers

```
Tier    | kBlockM | Warps | MFMA     | Grid       | Use case
--------|---------|-------|----------|------------|--------------------
Tiny    | 16      | 1     | 16x16x32 | 2D decode  | Pure decode (avg_q <= 2)
BS32    | 32      | 2     | 16x16x32 | 2D decode  | block_size=32 decode
Small   | 64      | 2     | 32x32x16 | 2D decode  | Short decode (avg_q <= 8)
Medium  | 128     | 4     | 32x32x16 | 1D prefill | All prefill
Large   | 256     | 8     | 32x32x16 | 1D prefill | Long prefill (no BS32)
```

Selection: `avg_q = num_tokens / num_seqs`, pick smallest tier where
`avg_q <= kBlockQ` and `max_seqlen_q <= kBlockQ`.

### GQA Head-Merging

With GQA-8 (8 Q-heads per KV-head), each workgroup computes attention for
all 8 Q-heads simultaneously by packing them into the M dimension:

```
# Without merge: kBlockM=16, each row is 1 Q token
M dimension = [q_token_0, q_token_1, ..., q_token_15]  (16 rows, mostly padding for decode)

# With GQA merge (kBlockM=16, kBlockQ=2, GQA-8):
# 2 Q tokens x 8 heads = 16 rows in M dimension
M dimension = [seq0_head0, seq0_head1, ..., seq0_head7,   # 8 heads for seq 0
               seq1_head0, seq1_head1, ..., seq1_head7]   # 8 heads for seq 1
```

All 8 Q-heads share the same K/V (one KV-head), so a single QK^T GEMM
computes scores for all heads at once.

### Pseudocode

```
kernel_ck_ua(kv_head_idx, seq_idx):    # decode grid
    q_start = query_start_len[seq_idx]
    q_len   = query_start_len[seq_idx + 1] - q_start
    kv_len  = seq_lens[seq_idx]

    # Load Q tile with GQA merge: [kBlockM x hdim]
    # For decode: 1 token x 8 heads = 8 rows, padded to kBlockM
    Q_tile = load_q_merged(q_start, kv_head_idx, num_queries_per_kv)

    # Page navigation
    num_pages = ceil(kv_len / page_blk_size)
    acc = zeros(kBlockM, hdim)
    m = -inf; l = 0

    for page_i in range(num_pages):
        phys_page = block_tables[seq_idx, page_i]

        # Async load K/V from paged cache to LDS (double-buffered)
        K_tile = async_load(k_ptr[phys_page])    # [page_blk_size x hdim]
        V_tile = async_load(v_ptr[phys_page])    # [page_blk_size x hdim]

        # MFMA GEMM: S = Q @ K^T
        S = mfma_gemm(Q_tile, K_tile)            # [kBlockM x page_blk_size]

        # Causal mask
        if IsMasking:
            apply_causal_mask(S, q_positions, page_i * page_blk_size)

        # Online softmax + accumulate
        m_new = max(m, row_max(S))
        P = exp(S - m_new)
        acc = acc * exp(m - m_new) + mfma_gemm(P, V_tile)
        l = l * exp(m - m_new) + row_sum(P)
        m = m_new

    output = acc / l
    # Write back: un-merge heads to [num_tokens, num_heads_q, hdim]
    store_unmerged(o_ptr, seq_idx, kv_head_idx, output)
```

### Numerical Example: GQA Head-Merge

```
Setup: GQA-4 (4 Q-heads per KV-head), 1 decode token, hdim=2

Q-heads for seq 0:
  head 0: [0.1, 0.2]
  head 1: [0.3, 0.4]
  head 2: [0.5, 0.6]
  head 3: [0.7, 0.8]

After merge into M dimension (kBlockM=4):
  Q_tile = [[0.1, 0.2],     # head 0
            [0.3, 0.4],     # head 1
            [0.5, 0.6],     # head 2
            [0.7, 0.8]]     # head 3

K (shared across all 4 heads, 1 KV-head):
  K = [[1.0, 0.0],          # kv token 0
       [0.0, 1.0]]          # kv token 1

S = Q_tile @ K.T =
  [[0.1, 0.2],              # head 0 scores
   [0.3, 0.4],              # head 1 scores
   [0.5, 0.6],              # head 2 scores
   [0.7, 0.8]]              # head 3 scores

One MFMA computes scores for ALL 4 heads simultaneously.
Each head gets its own softmax row -> its own attention output.
```

---

## 4. CK-SK -- CK FMHA Split-KV

**What:** CK Tile's split-KV flash attention. Splits the KV sequence across
`num_splits` workgroups, each computing partial results. A separate combine
kernel merges the partial max/sum/output using the log-sum-exp trick.

**When to use:** Long KV sequences with low batch, especially decode.
Called through `mha_varlen_fwd()` when `block_table` is provided.

### Grid

```
# Split-KV attention kernel
grid = dim3(batch * nhead_q * num_splits)
# Linearized: each workgroup handles 1 (batch, head, split) triple

# Combine kernel
grid = dim3(batch * nhead_q * max_seqlen_q)
# Merges splits for each (batch, head, q_position)

num_splits = heuristic(batch, num_heads_k, max_seqlen_q, head_size)
           = chosen to fill GPU: target = multiProcessorCount * 4
```

**Example:** batch=4, nhead_q=64, num_splits=16:
```
attention grid = dim3(4 * 64 * 16) = 4096 workgroups
combine grid   = dim3(4 * 64 * 1)  = 256 workgroups     (decode: max_seqlen_q=1)
```

### Parameters

```
Tensors:
  q              [total_q_tokens, num_heads_q, head_size]                bf16
  k              [num_blks, block_size, num_kv_heads, head_size]         bf16
  v              [num_blks, block_size, num_kv_heads, head_size]         bf16
  out            [total_q_tokens, num_heads_q, head_size]                bf16
  cu_seqlens_q   [batch + 1]                                            int32
  cu_seqlens_k   [batch + 1]                                            int32
  block_table    [batch, max_num_blocks_per_seq]                         int32

Intermediate (allocated internally):
  lse_acc        [batch, nhead_q, num_splits, max_seqlen_q]              float
  o_acc          [batch, nhead_q, num_splits, max_seqlen_q, head_size]   float

Scalars:
  max_seqlen_q, max_seqlen_k    int
  softmax_scale                  float
  logits_soft_cap                float    (0 = disabled)
  window_size_left, window_size_right   int    (-1 = no window)
  sink_size                      int
  is_causal                      bool
  num_splits                     int
```

### Pseudocode

```
# --- Split-KV attention kernel ---
kernel_splitkv(batch_idx, head_idx, split_idx):
    kv_len = seqlen_k[batch_idx]
    split_start = split_idx * ceil(kv_len / num_splits)
    split_end   = min(split_start + ceil(kv_len / num_splits), kv_len)

    # Flash attention loop over [split_start, split_end)
    # with paged KV lookup via block_table
    acc, m, l = flash_attn_loop(Q, K[split_start:split_end], V[split_start:split_end])

    # Store partial results
    o_acc[batch, head, split] = acc    # unnormalized
    lse_acc[batch, head, split] = m + log(l)

# --- Combine kernel ---
kernel_combine(batch_idx, head_idx, q_idx):
    global_max = max over splits of lse_acc[batch, head, :]

    output = zeros(head_size)
    total = 0
    for s in range(num_splits):
        w = exp(lse_acc[batch, head, s] - global_max)
        output += o_acc[batch, head, s] * w
        total += w
    output /= total
    store(out[batch, head, q_idx], output)
```

---

## 5. CK-PK -- CK FMHA PagedKV (Batch Prefill)

**What:** Non-split paged-KV forward attention from CK Tile. Unlike CK-SK,
this uses per-token page lookups (not tile-level navigation). Single kernel,
no combine step needed.

**When to use:** Prefill with paged KV cache. Called through
`mha_varlen_fwd_pagedkv()`.

### Grid

```
grid = dim3(nhead_q, total_num_q_tiles)
# where total_num_q_tiles = sum over seqs of ceil(seqlen_q[i] / bm0)
```

**Example:** batch=2, seqlen_q=[512, 256], nhead_q=64, bm0=128:
```
q_tiles = ceil(512/128) + ceil(256/128) = 4 + 2 = 6
grid = dim3(64, 6) = 384 workgroups
```

### Parameters

```
Tensors:
  q, k, v, out, cu_seqlens_q, cu_seqlens_k, block_table
  (same shapes as CK-SK)

Scalars:
  max_seqlen_q, max_seqlen_k, softmax_scale, logits_soft_cap
  is_causal, window_size_left, window_size_right, sink_size

Compile-time:
  bm0            int    Q tile size (16, 32, 128 depending on tier)
  bn0            int    KV tile size (32)
  page_size      int    page block size (32, 64, 128, 256)
```

### Pseudocode

```
kernel_pagedkv(head_idx, q_tile_idx):
    # Find batch and local Q tile position
    batch_idx, local_q_tile = find_batch(cu_seqlens_q, q_tile_idx, bm0)

    q_start = cu_seqlens_q[batch_idx] + local_q_tile * bm0
    kv_len  = seqlen_k[batch_idx]

    Q_tile = load_q(q_start, head_idx)

    acc = zeros(bm0, hdim)
    m = -inf; l = 0

    for kv_pos in range(0, kv_len, bn0):
        # Per-token page lookup (not tile-level)
        for t in range(bn0):
            token_idx = kv_pos + t
            page = block_table[batch_idx, token_idx // page_size]
            offset = token_idx % page_size
            K[t] = k_cache[page, offset, kv_head]
            V[t] = v_cache[page, offset, kv_head]

        S = Q_tile @ K.T * scale
        apply_mask(S)
        # online softmax + accumulate
        ...

    output = acc / l
    store(out, q_start, head_idx, output)
```

---

## 6. CK-Fwd -- CK FMHA Forward (Non-Paged)

**What:** Standard flash-attention forward from CK Tile for contiguous
(non-paged) KV tensors. Cannot use block_table.

**When to use:** Prefill when KV is stored contiguously in memory (no
paged KV cache). Rarely used in vLLM-style inference which always pages.

### Grid

```
grid = dim3(nhead_q, num_q_tiles)
# num_q_tiles = sum over batches of ceil(seqlen_q[i] / bm0)
```

### Parameters

```
Tensors:
  q    [total_q_tokens, num_heads_q, head_size]     bf16
  k    [total_kv_tokens, num_kv_heads, head_size]   bf16   (contiguous, not paged)
  v    [total_kv_tokens, num_kv_heads, head_size]   bf16
  out  [total_q_tokens, num_heads_q, head_size]     bf16
  cu_seqlens_q, cu_seqlens_k                        int32
  bias (optional)

Scalars:
  max_seqlen_q, max_seqlen_k, softmax_scale
  logits_soft_cap, dropout_p
  window_size_left, window_size_right
  is_causal
```

### Pseudocode

```
kernel_fwd(head_idx, q_tile_idx):
    batch_idx, local_q_tile = find_batch(cu_seqlens_q, q_tile_idx, bm0)
    q_start  = cu_seqlens_q[batch_idx] + local_q_tile * bm0
    kv_start = cu_seqlens_k[batch_idx]
    kv_len   = cu_seqlens_k[batch_idx + 1] - kv_start

    Q_tile = load_q(q_start, head_idx)

    # Direct access to contiguous K/V (no page table)
    acc, m, l = flash_attn_loop(Q_tile, K[kv_start:kv_start+kv_len],
                                         V[kv_start:kv_start+kv_len])
    output = acc / l
    store(out, q_start, head_idx, output)
```

---

## Key Concepts

### Paged KV Cache

Physical KV pages are stored in a pool. A `block_table[seq, page_col]`
maps logical page indices to physical page indices.

```
Physical KV pool:  [num_phys_pages, page_block_size, num_kv_heads, head_size]

Logical view for seq 0 (kv_len=5, page_block_size=2):
  block_table[0] = [7, 3, 1]    # 3 pages needed (ceil(5/2))

  Logical token 0 -> pool[7][0]
  Logical token 1 -> pool[7][1]
  Logical token 2 -> pool[3][0]
  Logical token 3 -> pool[3][1]
  Logical token 4 -> pool[1][0]  # last page partially filled

Lookup pseudocode:
  page_idx    = block_table[seq, token // page_block_size]
  page_offset = token % page_block_size
  kv_data     = kv_cache[page_idx, page_offset, kv_head, :]
```

### Causal Masking

For causal (autoregressive) attention, token at position `q_pos` can only
attend to KV tokens at positions `<= q_pos`.

```
Q positions:  0  1  2  3
KV positions: 0  1  2  3

Mask matrix (1 = attend, 0 = masked):
     KV: 0  1  2  3
Q 0:     1  0  0  0
Q 1:     1  1  0  0
Q 2:     1  1  1  0
Q 3:     1  1  1  1

For decode (q_pos = kv_len - 1): all KV tokens are visible -> no masking needed.
For prefill: triangular mask, but shifted by context_len if KV cache exists.
```

### Sliding Window Attention

Only attend to the most recent `window_left` KV tokens:

```
window_left = 3, q_pos = 7, kv_len = 8

KV positions:  0  1  2  3  4  5  6  7
               X  X  X  X  .  1  1  1    (X = masked, 1 = attend)
                              ^-------^
                              kv_pos >= q_pos - window_left

Benefit: KV iteration starts at max(0, q_pos - window_left) instead of 0.
Triton and CK-SK bound the iteration range.
CK-UA iterates the full range and masks (no early termination -> slow).
```

### Split-KV Reduce (Log-Sum-Exp Merge)

When KV is split across `S` segments, each produces:
- `o_s`: unnormalized output (weighted by local softmax)
- `m_s`: local max logit
- `l_s`: local sum of exponentials

Merge formula:
```
m_global = max(m_0, m_1, ..., m_{S-1})

For each segment s:
  c_s = exp(m_s - m_global)          # correction factor

total_l = sum(l_s * c_s for s in 0..S-1)
output  = sum(o_s * c_s for s in 0..S-1) / total_l
```

This is numerically equivalent to computing attention over the full KV range.

### CK-UA Selector Logic

```python
def should_use_ck_ua(max_seqlen_q, num_seqs, num_kv_heads, window_size,
                     block_size, max_seqlen_k, head_size, num_queries_per_kv):
    if max_seqlen_q != 1:           return False   # decode only
    if window_size != (-1, -1):     return False   # no sliding window
    if block_size < 32:             return False
    if block_size < 64 and max_seqlen_k < 256:
        return False                                # pre-existing CK pipeline bug guard

    # Only compiled for these configs
    if not ((head_size == 64 and num_queries_per_kv == 8) or
            (head_size == 128 and num_queries_per_kv == 1)):
        return False

    cu_count = get_num_sms()         # e.g. 256 on MI300X
    triton_2d_wgs = num_kv_heads * num_seqs

    # CK-UA wins in the "moderate occupancy" zone
    return cu_count * 4 <= triton_2d_wgs <= cu_count * 8
```

On a 256-CU GPU with 8 KV-heads: CK-UA activates for seqs in [128, 256].

---

## Files

| Pipeline | Key files |
|----------|-----------|
| Triton 2D/3D | `aiter/ops/triton/attention/unified_attention.py`, `aiter/ops/triton/_triton_kernels/attention/unified_attention.py` |
| CK-UA | `3rdparty/composable_kernel/example/ck_tile/42_unified_attention/` |
| CK-SK | `3rdparty/composable_kernel/example/ck_tile/01_fmha/codegen/ops/fmha_fwd_splitkv.py` |
| CK-PK | `3rdparty/composable_kernel/example/ck_tile/01_fmha/codegen/ops/fmha_pagedkv_prefill.py` |
| CK-Fwd | `3rdparty/composable_kernel/example/ck_tile/01_fmha/codegen/ops/fmha_fwd.py` |
| Selector | `_try_ck_unified_attention()` in `aiter/ops/triton/attention/unified_attention.py` |
| CK-SK/PK wrapper | `aiter/ops/mha.py`, `csrc/py_itfs_ck/mha_varlen_fwd_kernels.cu` |
