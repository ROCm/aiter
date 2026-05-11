# Attention Kernels in AIter -- Explained with Pseudocode

This document explains every major attention kernel in AIter using numpy-style
pseudocode with dimensions in comments. Each section is self-contained: you can
read just the kernel you care about.

All examples use the same model config (DeepSeek-V3/R1 style):

```python
nhead_q  = 64      # query heads
nhead_kv = 8       # key/value heads
hdim     = 64      # dimension per head
num_queries_per_kv = nhead_q // nhead_kv  # = 8 (GQA-8)
num_seqs = 4       # sequences in the batch (all doing decode unless noted)
page_size = 64     # tokens per KV cache page
```

---

## 0. Background

### The math

Multi-head attention computes, for each head h:

```
Output_h = softmax(Q_h @ K_h^T / sqrt(hdim)) @ V_h
```

The full output is all heads concatenated: `[Output_0, Output_1, ..., Output_{nhead_q-1}]`.

### GQA (Grouped Query Attention)

With GQA, multiple Q heads share one KV head. Every `num_queries_per_kv`
consecutive Q heads read the same K and V:

```python
Q = ...    # [num_seqs, nhead_q, hdim]    = [4, 64, 64]
K = ...    # [num_seqs, nhead_kv, seq_len, hdim] = per-seq KV cache
V = ...    # same shape as K

Output = np.zeros([num_seqs, nhead_q, hdim])  # [4, 64, 64]

for seq in range(4):
    for q_head in range(64):
        kv_head = q_head // 8       # heads 0-7   -> kv_head 0
                                     # heads 8-15  -> kv_head 1
                                     # ...
                                     # heads 56-63 -> kv_head 7
        q = Q[seq, q_head]           # [64]
        k = K[seq, kv_head]          # [kv_len, 64]
        v = V[seq, kv_head]          # [kv_len, 64]

        scores = q @ k.T / sqrt(64)  # [kv_len]
        Output[seq, q_head] = softmax(scores) @ v  # [64]
```

Heads 0-7 all read K[kv_head=0] and V[kv_head=0] but have different Q vectors,
so they compute different attention outputs.

### Paged KV cache

The GPU has a pool of fixed-size memory pages. Each page holds `page_size`
KV tokens. A sequence's KV tokens are scattered across non-contiguous pages.
`block_table[seq][col]` maps logical page index to physical page ID:

```python
kv_cache = ...  # [num_pages, page_size, nhead_kv, hdim]  -- physical pool

block_table = [
    [5, 12],       # seq0: 100 tokens -> pages 5 (tok 0-63), 12 (tok 64-99)
    [3, 7, 0, 9],  # seq1: 200 tokens -> 4 pages
    [10, 6],       # seq2: 80 tokens  -> 2 pages
    [2],           # seq3: 50 tokens  -> 1 page
]
seq_lens_k = [100, 200, 80, 50]

# To read KV token 130 of seq1:
page_col    = 130 // 64               # = 2 -> block_table[1][2] = page 0
page_offset = 130 % 64                # = 2 -> slot 2 within that page
data = kv_cache[0, 2, kv_head, :]     # [hdim]
```

### Flat Q array and cu_seqlens_q

Q tokens from all sequences are concatenated into a flat array.
`cu_seqlens_q` marks where each sequence starts:

```python
# 4 sequences doing decode (1 token each):
Q_flat = np.stack([Q[0], Q[1], Q[2], Q[3]])  # [4, nhead_q, hdim]
cu_seqlens_q = [0, 1, 2, 3, 4]
# Sequence i owns flat tokens from cu_seqlens_q[i] to cu_seqlens_q[i+1]
```

### Online softmax

All kernels use "online softmax" to avoid materializing the full score vector.
Instead of computing softmax over all KV at once, they process KV in tiles and
maintain running statistics:

```python
acc = np.zeros([hdim])   # weighted output accumulator
m = -np.inf              # running max score
l = 0.0                  # running sum of exp(scores)

for kv_tile in kv_tiles:
    S = q @ kv_tile.K.T / sqrt(hdim)           # scores for this tile
    m_new = max(m, np.max(S))
    correction = np.exp(m - m_new)              # rescale old accumulators
    P = np.exp(S - m_new)                       # softmax numerators
    l = l * correction + np.sum(P)
    acc = acc * correction + P @ kv_tile.V
    m = m_new

output = acc / l  # final normalized output
```

This produces the exact same result as full softmax but needs only O(hdim)
memory instead of O(kv_len).

---

## 1. CK-PK -- CK FMHA PagedKV (Single Kernel, No Head-Merge)

**What:** Single-kernel paged-KV attention from Composable Kernel. Each workgroup
handles one Q head. Designed for **prefill** where Q tiles have many tokens.

**Source:** `3rdparty/composable_kernel/include/ck_tile/ops/fmha/kernel/fmha_fwd_pagedkv_kernel.hpp`
**Python:** `aiter/ops/mha.py` -> `mha_varlen_fwd_pagedkv()`

```python
# === CONFIG ===
bm0 = 128    # Q tile height (tokens per workgroup)
bn0 = 32     # KV tile width (KV tokens per loop iteration)

# === GRID ===
# dim3(nhead_q, batch, q_tiles)
# One workgroup per (Q head, batch, Q tile)
total_q_tiles = ceil(max_seqlen_q / bm0)  # e.g. ceil(1/128) = 1 for decode
# Total WGs = nhead_q * num_seqs * total_q_tiles = 64 * 4 * 1 = 256

# Each workgroup: 4 warps, 256 threads (on AMD, 1 warp = 64 threads)

# === ONE WORKGROUP: q_head_idx=5, seq_idx=1, q_tile_idx=0 ===

def workgroup_ck_pk(q_head_idx, seq_idx, q_tile_idx):
    kv_head_idx = q_head_idx // 8         # = 0 (GQA-8 mapping)
    kv_len = seq_lens_k[seq_idx]          # = 200 for seq1

    # Load Q tile: bm0 rows of Q tokens for this head
    # For decode: only 1 token, rest is padding
    # For prefill: up to bm0 real tokens
    token_start = q_tile_idx * bm0
    Q_tile = np.zeros([bm0, hdim])                            # [128, 64]
    for row in range(bm0):
        tok = token_start + row
        if tok < seqlen_q[seq_idx]:
            Q_tile[row] = Q_flat[cu_seqlens_q[seq_idx] + tok, q_head_idx]

    # Accumulators (one per row)
    acc = np.zeros([bm0, hdim])    # [128, 64]
    m = np.full([bm0], -np.inf)    # [128]
    l = np.zeros([bm0])            # [128]

    # Loop over KV in chunks of bn0
    for kv_pos in range(0, kv_len, bn0):  # step by 32

        # Page lookup: resolve physical pages for this KV chunk
        K_tile = np.zeros([bn0, hdim])     # [32, 64]
        V_tile = np.zeros([bn0, hdim])     # [32, 64]
        for t in range(bn0):
            tok = kv_pos + t
            if tok < kv_len:
                page_col    = tok // page_size
                page_offset = tok % page_size
                phys_page = block_table[seq_idx][page_col]
                K_tile[t] = kv_cache[phys_page, page_offset, kv_head_idx, :]
                V_tile[t] = kv_cache[phys_page, page_offset, kv_head_idx, :]

        # Matrix multiply: all 256 threads cooperate on this MFMA
        S = Q_tile @ K_tile.T / sqrt(hdim)   # [128, 64] @ [64, 32] = [128, 32]

        # Mask out-of-range KV positions and causal violations
        for row in range(bm0):
            q_pos = token_start + row
            for col in range(bn0):
                kv_tok = kv_pos + col
                if kv_tok >= kv_len or kv_tok > q_pos:  # causal
                    S[row, col] = -np.inf

        # Online softmax update (each row independently)
        for row in range(bm0):
            m_new = max(m[row], np.max(S[row]))
            correction = np.exp(m[row] - m_new)
            P = np.exp(S[row] - m_new)               # [32]
            l[row] = l[row] * correction + np.sum(P)
            acc[row] = acc[row] * correction + P @ V_tile
            m[row] = m_new

    # Store results
    for row in range(bm0):
        tok = token_start + row
        if tok < seqlen_q[seq_idx]:
            out[cu_seqlens_q[seq_idx] + tok, q_head_idx] = acc[row] / l[row]
```

**Key points:**
- Grid iterates over **Q heads** (`nhead_q=64`), not KV heads
- Each WG handles 1 Q head -- Q heads sharing a KV head launch **separate** WGs
  that independently read the same K/V pages
- No head-merging: for decode, only 1 row out of `bm0=128` has real data
- Good for **prefill**: many Q tokens fill the tile, MFMA utilization is high

---

## 2. CK-UA -- CK Unified Attention (Single Kernel, Head-Merge)

**What:** Single-kernel paged-KV attention with GQA head-merging. All Q heads in
a GQA group are packed into the M dimension of a single workgroup, so one MFMA
computes scores for all heads at once.

**Source:** `3rdparty/composable_kernel/example/ck_tile/42_unified_attention/`
**Python:** `aiter/ops/unified_attention.py` -> `unified_attention_fwd()`

```python
# === CONFIG (tiny decode tier) ===
kBlockM = 16         # Q tile height (after merging heads into M)
kBlockQ = kBlockM // num_queries_per_kv  # = 16 // 8 = 2 token slots
kPageBlockSize = 64  # KV tile = one page

# === GRID (decode) ===
# dim3(num_kv_heads, num_seqs) = dim3(8, 4) = 32 workgroups
# One workgroup per (KV head, sequence)

# Each workgroup: 1 warp, 64 threads (tiny tier)
# Other tiers: small = 2 warps/128 threads, medium = 4 warps/256 threads

# === ONE WORKGROUP: kv_head_idx=0, seq_idx=1 ===

def workgroup_ck_ua(kv_head_idx, seq_idx):
    kv_len = seq_lens_k[seq_idx]              # = 200 for seq1
    q_start = cu_seqlens_q[seq_idx]           # where seq1's Q tokens start
    q_len = cu_seqlens_q[seq_idx + 1] - q_start  # = 1 for decode

    # === HEAD-MERGE: pack all Q heads for this KV group into M ===
    # For decode (1 token), this KV head serves Q heads 0..7
    # Pack them as rows: 1 token x 8 heads = 8 rows in M dimension
    Q_tile = np.zeros([kBlockM, hdim])          # [16, 64]
    for tok in range(q_len):                    # 0..0 for decode
        for h in range(num_queries_per_kv):     # 0..7
            row = tok * num_queries_per_kv + h
            q_head = kv_head_idx * num_queries_per_kv + h  # 0,1,2,...,7
            Q_tile[row] = Q_flat[q_start + tok, q_head]
    # Result for decode: rows 0-7 have real data (8 heads), rows 8-15 are padding

    num_pages = ceil(kv_len / kPageBlockSize)   # ceil(200/64) = 4

    acc = np.zeros([kBlockM, hdim])     # [16, 64]
    m = np.full([kBlockM], -np.inf)     # [16]
    l = np.zeros([kBlockM])             # [16]

    # Loop over KV one page at a time
    for page_i in range(num_pages):
        # Direct block_table lookup (no navigator abstraction)
        phys_page = block_table[seq_idx][page_i]

        # Load one full page of K and V
        K_tile = kv_cache[phys_page, :, kv_head_idx, :]  # [64, 64]
        V_tile = kv_cache[phys_page, :, kv_head_idx, :]  # [64, 64]

        # MFMA: scores for ALL 8 heads at once
        S = Q_tile @ K_tile.T / sqrt(hdim)
        # [16, 64] @ [64, 64] = [16, 64]
        # Row 0: head 0's scores against 64 KV tokens
        # Row 1: head 1's scores against 64 KV tokens
        # ...
        # Row 7: head 7's scores against 64 KV tokens
        # Rows 8-15: padding (zeros x K = zeros)

        # Causal mask
        for row in range(kBlockM):
            tok = row // num_queries_per_kv
            for col in range(kPageBlockSize):
                kv_tok = page_i * kPageBlockSize + col
                if kv_tok >= kv_len:
                    S[row, col] = -np.inf

        # Online softmax update
        for row in range(kBlockM):
            m_new = max(m[row], np.max(S[row]))
            correction = np.exp(m[row] - m_new)
            P = np.exp(S[row] - m_new)
            l[row] = l[row] * correction + np.sum(P)
            acc[row] = acc[row] * correction + P @ V_tile
            m[row] = m_new

    # Un-merge: write each row back to its Q head
    for tok in range(q_len):
        for h in range(num_queries_per_kv):
            row = tok * num_queries_per_kv + h
            q_head = kv_head_idx * num_queries_per_kv + h
            out[q_start + tok, q_head] = acc[row] / l[row]
```

**Key points:**
- Grid iterates over **KV heads** (`nhead_kv=8`), not Q heads
- Head-merge: 8 Q heads packed into M -> one MFMA computes all 8 heads' scores
- Only 32 WGs for decode (vs 256 for CK-PK), each doing 8x more useful work
- K/V loaded **once** per page, shared across all 8 heads in the group
- For decode with GQA-8: 8 useful rows out of 16 (50% utilization vs ~0.8% in CK-PK)

---

## 3. CK-SK -- CK FMHA Split-KV (Two Kernels)

**What:** Splits the KV sequence across multiple workgroups. Each computes a
partial result over its KV slice. A separate combine kernel merges the partials
using log-sum-exp. Needed when batch is too small to fill the GPU without
splitting.

**Source:** `3rdparty/composable_kernel/include/ck_tile/ops/fmha/kernel/fmha_fwd_splitkv_kernel.hpp`
**Python:** `aiter/ops/mha.py` -> `mha_varlen_fwd()` with `block_table`

```python
# === CONFIG ===
kM0 = 16           # Q tile height
num_splits = 8     # chosen by heuristic to fill GPU

# === GRID ===
# Attention kernel: dim3(q_tiles * num_splits, nhead_q, batch)
#   for decode: dim3(1 * 8, 64, 4) = 2048 WGs
# Combine kernel:  dim3(q_tiles, nhead_q, batch)
#   for decode: dim3(1, 64, 4) = 256 WGs

# With head-merge (decode + GQA): nhead_q -> nhead_kv in grid,
# effective Q length = seqlen_q * num_queries_per_kv

# === INTERMEDIATE BUFFERS ===
o_acc   = np.zeros([nhead_q, num_splits, total_q, hdim])  # [64, 8, 4, 64] fp32
lse_acc = np.zeros([nhead_q, num_splits, total_q])        # [64, 8, 4]     fp32

# ============================================================
# KERNEL 1: Split-KV Attention
# ============================================================
# Each WG processes ONE split of the KV range for one (head, batch, q_tile)

def workgroup_splitkv_attention(q_head_idx, seq_idx, split_idx):
    kv_head_idx = q_head_idx // 8
    kv_len = seq_lens_k[seq_idx]

    # This split's KV range
    kv_per_split = ceil(kv_len / num_splits)
    kv_start = split_idx * kv_per_split
    kv_end   = min(kv_start + kv_per_split, kv_len)
    if kv_start >= kv_len:
        return  # nothing to do for this split

    # Load Q tile (same Q for all splits -- they differ only in KV range)
    Q_tile = load_q(seq_idx, q_head_idx)    # [kM0, hdim]

    acc = np.zeros([kM0, hdim])
    m = np.full([kM0], -np.inf)
    l = np.zeros([kM0])

    # Loop only over THIS split's KV range (not all KV)
    for kv_pos in range(kv_start, kv_end, bn0):
        # Page lookup for this KV chunk
        K_tile, V_tile = page_lookup(seq_idx, kv_head_idx, kv_pos, bn0)

        S = Q_tile @ K_tile.T / sqrt(hdim)
        # ... causal mask ...
        # ... online softmax update (same as before) ...

    # Store PARTIAL results (not final -- combine kernel will merge)
    for row in range(kM0):
        tok = token_for_row(row)
        o_acc[q_head_idx, split_idx, tok]   = acc[row]    # unnormalized
        lse_acc[q_head_idx, split_idx, tok] = m[row] + log(l[row])  # log-sum-exp

# ============================================================
# KERNEL 2: Combine (runs after all split-KV WGs finish)
# ============================================================
# Each WG merges all splits for one (head, batch, q_tile)

def workgroup_combine(q_head_idx, seq_idx):
    tok = cu_seqlens_q[seq_idx]  # for decode: 1 token

    # Find global max across all splits
    global_max = -np.inf
    for s in range(num_splits):
        global_max = max(global_max, lse_acc[q_head_idx, s, tok])

    # Merge: rescale each split's partial to the global max, then sum
    total = 0.0
    output = np.zeros([hdim])
    for s in range(num_splits):
        # exp(local_lse - global_max) is the correction factor
        w = np.exp(lse_acc[q_head_idx, s, tok] - global_max)
        output += o_acc[q_head_idx, s, tok] * w
        total += w

    out[tok, q_head_idx] = output / total
```

**Key points:**
- Two kernel launches: attention (many WGs) then combine (fewer WGs)
- Needs intermediate `o_acc` and `lse_acc` buffers in fp32
- `num_splits` chosen by heuristic: target = `multiProcessorCount * 4` total WGs
- When batch is small and KV is long, splits provide parallelism the GPU needs
- Also supports head-merge for decode (`kMergeNumHeadGroupsSeqLenQ=true`):
  grid uses `nhead_kv` instead of `nhead_q`, effective Q length multiplied by
  `num_queries_per_kv`

---

## 4. CK-Fwd -- CK FMHA Forward (Non-Paged)

**What:** Standard flash-attention forward for contiguous (non-paged) KV tensors.
Same as CK-PK but without page table indirection. Simplest path.

**Source:** `3rdparty/composable_kernel/include/ck_tile/ops/fmha/kernel/fmha_fwd_kernel.hpp`
**Python:** `aiter/ops/mha.py` -> `mha_fwd()`

```python
# === GRID ===
# dim3(nhead_q, q_tiles, batch) = dim3(64, 1, 4) = 256 WGs

def workgroup_ck_fwd(q_head_idx, q_tile_idx, seq_idx):
    kv_head_idx = q_head_idx // 8
    kv_len = seq_lens_k[seq_idx]

    Q_tile = load_q(seq_idx, q_head_idx, q_tile_idx)  # [kM0, hdim]

    acc = np.zeros([kM0, hdim])
    m = np.full([kM0], -np.inf)
    l = np.zeros([kM0])

    # KV is contiguous -- direct indexing, no page table
    kv_start = cu_seqlens_k[seq_idx]

    for kv_pos in range(0, kv_len, bn0):
        # Direct memory access -- K/V are contiguous per sequence
        K_tile = K_flat[kv_start + kv_pos : kv_start + kv_pos + bn0, kv_head_idx, :]
        V_tile = V_flat[kv_start + kv_pos : kv_start + kv_pos + bn0, kv_head_idx, :]

        S = Q_tile @ K_tile.T / sqrt(hdim)   # [kM0, bn0]
        # ... causal mask, online softmax (same pattern) ...

    out[q_start + tok, q_head_idx] = acc / l
```

**Key points:**
- No page table, no block_table lookups -- K/V accessed by direct offset
- Otherwise identical to CK-PK: same grid over Q heads, same tile sizes
- Used for prefill when KV is stored contiguously (not in paged cache)
- Rarely used in vLLM-style inference (which always pages KV)

---

## 5. Triton 2D -- Single-Pass Unified Attention (Head-Merge)

**What:** Single-pass paged-KV attention written in Triton. Does GQA
head-merging just like CK-UA: multiple Q heads packed into BLOCK_M.

**Source:** `aiter/ops/triton/_triton_kernels/attention/unified_attention.py` -> `kernel_unified_attention_2d`
**Python:** `aiter/ops/triton/attention/unified_attention.py` -> `unified_attention()`

### BLOCK_M and BLOCK_Q

`BLOCK_M` is the total number of rows in the Q tile. With head-merging, those
rows interleave **tokens** and **heads**:

```python
BLOCK_M = 16         # total rows in Q tile
BLOCK_Q = BLOCK_M // num_queries_per_kv  # = 16 // 8 = 2 token slots

# The 16 rows are laid out as:
#   row 0:  token 0, head 0      row 8:  token 1, head 0
#   row 1:  token 0, head 1      row 9:  token 1, head 1
#   row 2:  token 0, head 2      row 10: token 1, head 2
#   row 3:  token 0, head 3      row 11: token 1, head 3
#   row 4:  token 0, head 4      row 12: token 1, head 4
#   row 5:  token 0, head 5      row 13: token 1, head 5
#   row 6:  token 0, head 6      row 14: token 1, head 6
#   row 7:  token 0, head 7      row 15: token 1, head 7
#
# For decode (1 token): rows 0-7 real, rows 8-15 padding -> 50% utilization
# For prefill (2+ tokens): all 16 rows real -> 100% utilization
```

### Binary search

The grid is `(num_kv_heads, total_num_q_blocks)`. Since `total_num_q_blocks` spans
all sequences, each workgroup must figure out which sequence its block belongs to.

```python
TILE_SIZE = 64       # KV tokens per loop iteration

# total_num_q_blocks: upper bound across all sequences
# Each sequence's tokens are grouped into blocks of BLOCK_Q (=2) tokens.
# The "+ seq_idx" accounts for per-sequence block alignment.
total_num_q_blocks = total_q_tokens // BLOCK_Q + num_seqs
# For decode (4 tokens, BLOCK_Q=2): 4 // 2 + 4 = 6

# Grid: (num_kv_heads=8, total_num_q_blocks=6) = 48 WGs
```

```python
# === ONE WORKGROUP: kv_head_idx=0, q_block_global_idx=3 ===

def workgroup_triton_2d(kv_head_idx, q_block_global_idx):
    # Binary search: find which sequence owns this global Q block.
    # cu_seqlens_q = [0, 1, 2, 3, 4] (token-space boundaries)
    # Convert to block-space: cu_seqlens_q[i] // BLOCK_Q + i
    #   seq 0: 0 // 2 + 0 = 0
    #   seq 1: 1 // 2 + 1 = 1
    #   seq 2: 2 // 2 + 2 = 3
    #   seq 3: 3 // 2 + 3 = 4
    #   end:   4 // 2 + 4 = 6
    # q_block_global_idx=3 -> binary search finds seq_idx=2
    seq_idx = binary_search(cu_seqlens_q, q_block_global_idx, BLOCK_Q)
    q_start = cu_seqlens_q[seq_idx]
    q_len   = cu_seqlens_q[seq_idx + 1] - q_start
    kv_len  = seq_lens_k[seq_idx]

    # Local Q block index within this sequence
    q_block_local = q_block_global_idx - (q_start // BLOCK_Q + seq_idx)
    if q_block_local * BLOCK_Q >= q_len:
        return  # padding block, exit early

    # === HEAD-MERGE (same idea as CK-UA) ===
    # Load BLOCK_M rows: BLOCK_Q tokens x num_queries_per_kv heads
    Q_tile = np.zeros([BLOCK_M, hdim])   # [16, 64]
    for row in range(BLOCK_M):
        tok  = q_block_local * BLOCK_Q + row // num_queries_per_kv
        head = kv_head_idx * num_queries_per_kv + row % num_queries_per_kv
        # row 0: tok 0, head 0
        # row 1: tok 0, head 1
        # ...
        # row 7: tok 0, head 7
        # row 8: tok 1, head 0
        # ... etc
        if tok < q_len:
            Q_tile[row] = Q_flat[q_start + tok, head]

    acc = np.zeros([BLOCK_M, hdim])
    m = np.full([BLOCK_M], -np.inf)
    l = np.zeros([BLOCK_M])

    # Loop over ALL KV tokens in TILE_SIZE chunks
    num_tiles = ceil(kv_len / TILE_SIZE)
    for j in range(num_tiles):
        # Per-position page lookup within the tile
        K_tile = np.zeros([TILE_SIZE, hdim])
        V_tile = np.zeros([TILE_SIZE, hdim])
        for t in range(TILE_SIZE):
            seq_offset = j * TILE_SIZE + t
            if seq_offset < kv_len:
                phys_page = block_table[seq_idx][seq_offset // page_size]
                slot      = seq_offset % page_size
                K_tile[t] = kv_cache[phys_page, slot, kv_head_idx, :]
                V_tile[t] = kv_cache[phys_page, slot, kv_head_idx, :]

        # MFMA: all 8 heads' scores at once
        S = Q_tile @ K_tile.T / sqrt(hdim)    # [16, 64]

        # Causal mask + sliding window
        for row in range(BLOCK_M):
            tok = q_block_local * BLOCK_Q + row // num_queries_per_kv
            q_pos = q_start + tok + (kv_len - q_len)  # absolute position
            for col in range(TILE_SIZE):
                kv_pos = j * TILE_SIZE + col
                if kv_pos >= kv_len or kv_pos > q_pos:
                    S[row, col] = -np.inf

        # Online softmax update (same as all other kernels)
        for row in range(BLOCK_M):
            m_new = max(m[row], np.max(S[row]))
            correction = np.exp(m[row] - m_new)
            P = np.exp(S[row] - m_new)
            l[row] = l[row] * correction + np.sum(P)
            acc[row] = acc[row] * correction + P @ V_tile
            m[row] = m_new

    # Store: un-merge heads
    for row in range(BLOCK_M):
        tok  = q_block_local * BLOCK_Q + row // num_queries_per_kv
        head = kv_head_idx * num_queries_per_kv + row % num_queries_per_kv
        if tok < q_len:
            out[q_start + tok, head] = acc[row] / l[row]
```

**Key points:**
- Very similar to CK-UA: head-merging, grid over KV heads, single pass
- Written in Triton (Python JIT compiled) vs CK-UA's C++/HIP
- Per-position page indirection within each TILE_SIZE chunk
- Supports features CK-UA doesn't: sliding window, softcap, ALiBi, sinks
- 2D kernel is used when: sliding window, short KV (<=512), or enough WGs to fill GPU

---

## 6. Triton 3D -- Split-KV Unified Attention (Two Kernels)

**What:** Splits KV into segments processed by parallel workgroups, then reduces.
Same concept as CK-SK but in Triton, and includes head-merging.

**Source:** `aiter/ops/triton/_triton_kernels/attention/unified_attention.py` -> `kernel_unified_attention_3d`, `reduce_segments`
**Python:** `aiter/ops/triton/attention/unified_attention.py` -> `unified_attention()`

```python
# === CONFIG ===
BLOCK_M = 16
BLOCK_Q = BLOCK_M // num_queries_per_kv  # = 2
TILE_SIZE = 64
NUM_SEGMENTS = 16   # chosen by heuristic based on GPU occupancy

# === GRID ===
# Attention: (total_q_blocks, num_kv_heads, NUM_SEGMENTS) = (6, 8, 16) = 768 WGs
# Reduce:    (total_q_tokens, num_query_heads)             = (4, 64)   = 256 WGs

# === INTERMEDIATE BUFFERS ===
segm_output = np.zeros([total_q, nhead_q, NUM_SEGMENTS, hdim])  # fp32
segm_max    = np.zeros([total_q, nhead_q, NUM_SEGMENTS])        # fp32
segm_expsum = np.zeros([total_q, nhead_q, NUM_SEGMENTS])        # fp32

# ============================================================
# KERNEL 1: Segment Attention (with head-merge)
# ============================================================

def workgroup_triton_3d(q_block_global_idx, kv_head_idx, segm_idx):
    seq_idx = binary_search(cu_seqlens_q, q_block_global_idx, BLOCK_Q)
    kv_len = seq_lens_k[seq_idx]

    # How many tiles per segment
    tiles_per_segment = ceil(kv_len / (NUM_SEGMENTS * TILE_SIZE))
    tile_start = segm_idx * tiles_per_segment
    tile_end   = min(tile_start + tiles_per_segment, ceil(kv_len / TILE_SIZE))
    if tile_start * TILE_SIZE >= kv_len:
        return  # this segment has no KV to process

    # Load Q tile with head-merge (same as Triton 2D)
    Q_tile = load_q_merged(seq_idx, kv_head_idx, q_block_global_idx)

    acc = np.zeros([BLOCK_M, hdim])
    m = np.full([BLOCK_M], -np.inf)
    l = np.zeros([BLOCK_M])

    # Loop only over THIS segment's tile range
    for j in range(tile_start, tile_end):
        K_tile, V_tile = page_lookup_tile(seq_idx, kv_head_idx, j, TILE_SIZE)
        S = Q_tile @ K_tile.T / sqrt(hdim)
        # ... causal mask, online softmax update ...

    # Store partial results for this segment
    for row in range(BLOCK_M):
        tok  = token_for_row(row)
        head = head_for_row(row, kv_head_idx)
        segm_output[tok, head, segm_idx, :] = acc[row]  # unnormalized
        segm_max[tok, head, segm_idx]        = m[row]
        segm_expsum[tok, head, segm_idx]     = l[row]

# ============================================================
# KERNEL 2: Reduce Segments
# ============================================================

def workgroup_reduce(query_token_idx, query_head_idx):
    seq_idx = token_to_seq(query_token_idx)
    kv_len = seq_lens_k[seq_idx]

    # How many segments actually contributed for this sequence
    tiles_per_segment = ceil(kv_len / (NUM_SEGMENTS * TILE_SIZE))
    act_num_segments = ceil(kv_len / (tiles_per_segment * TILE_SIZE))

    # Global max across all segments
    overall_max = -np.inf
    for s in range(act_num_segments):
        overall_max = max(overall_max, segm_max[query_token_idx, query_head_idx, s])

    # Rescale and merge
    overall_expsum = 0.0
    output = np.zeros([hdim])
    for s in range(act_num_segments):
        correction = np.exp(segm_max[query_token_idx, query_head_idx, s] - overall_max)
        rescaled_expsum = segm_expsum[query_token_idx, query_head_idx, s] * correction
        overall_expsum += rescaled_expsum
        output += segm_output[query_token_idx, query_head_idx, s, :] * correction

    out[query_token_idx, query_head_idx] = output / overall_expsum
```

**Key points:**
- Same split-KV + reduce pattern as CK-SK, but in Triton with head-merging
- `NUM_SEGMENTS` chosen to fill GPU: `ceil(target_prgms / num_2d_prgms)`,
  clamped to [8, 128], rounded to power of 2
- Used when: long KV, low batch, and 2D doesn't have enough WGs

---

## 7. Triton PA Decode -- Classic Paged Attention

**What:** The "classic" paged attention decode kernel, predating unified attention.
No head-merging. Separate code paths for MHA and GQA.

**Source:** `aiter/ops/triton/_triton_kernels/attention/pa_decode.py`
**Python:** `aiter/ops/triton/attention/pa_decode.py` -> `paged_attention_decode()`

```python
# === CONFIG ===
# V1 (single pass): no splitting
# V2 (partitioned): splits KV into max_num_partitions + reduce

# === GRID (V1, GQA path) ===
# (num_seqs, num_kv_heads, 1)  = (4, 8, 1) = 32 WGs

# Note: V1 MHA path uses (num_q_heads, num_seqs, 1) instead

# === ONE WORKGROUP (V1 GQA): seq_idx=1, kv_head_idx=0 ===

def workgroup_pa_decode_v1_gqa(seq_idx, kv_head_idx):
    kv_len = seq_lens_k[seq_idx]

    # Load ALL Q heads for this GQA group
    # (different from head-merge: handled via dot product, not merged M)
    q_heads = []                                 # list of [hdim]
    for h in range(num_queries_per_kv):          # 0..7
        q_head = kv_head_idx * num_queries_per_kv + h
        q_heads.append(Q_flat[seq_idx, q_head])  # [64]

    # Accumulators per Q head
    acc = [np.zeros([hdim]) for _ in range(num_queries_per_kv)]
    m = [-np.inf] * num_queries_per_kv
    l = [0.0] * num_queries_per_kv

    for kv_pos in range(0, kv_len, BLOCK_KV):
        # Page lookup
        for t in range(BLOCK_KV):
            tok = kv_pos + t
            if tok < kv_len:
                phys_block = block_table[seq_idx][tok // block_size]
                offset = tok % block_size
                K[t] = k_cache[phys_block, kv_head_idx, offset, :]
                V[t] = v_cache[phys_block, kv_head_idx, offset, :]

        # Compute scores for EACH Q head separately
        for h in range(num_queries_per_kv):
            scores = q_heads[h] @ K.T / sqrt(hdim)  # [BLOCK_KV]
            # ... online softmax update for acc[h], m[h], l[h] ...

    for h in range(num_queries_per_kv):
        q_head = kv_head_idx * num_queries_per_kv + h
        out[seq_idx, q_head] = acc[h] / l[h]

# === V2: PARTITIONED (for long KV) ===
# Same idea as split-KV / Triton 3D:
# Attention grid: (num_seqs, num_kv_heads, max_num_partitions)
# Reduce grid:    (num_seqs, num_kv_heads, 1)
# Each partition processes a slice of KV, stores partial result,
# reduce kernel merges via log-sum-exp.
```

**Key points:**
- No head-merging: Q heads processed sequentially within the WG (GQA path) or
  one WG per Q head (MHA path)
- K/V cache layout is `[num_blocks, num_kv_heads, block_size, hdim]`
  (different from unified attention's `[num_blocks, block_size, num_kv_heads, hdim]`)
- V2 adds KV partitioning + reduce for long sequences (same pattern as Triton 3D / CK-SK)
- Being superseded by unified attention for most use cases

---

## 8. Triton PA Prefill -- Context Attention

**What:** Prefill attention that reads from paged KV cache while also processing
the current step's new K/V tokens. Writes new K/V into the cache.

**Source:** `aiter/ops/triton/_triton_kernels/attention/pa_prefill.py`
**Python:** `aiter/ops/triton/attention/pa_prefill.py` -> `context_attention_fwd()`

```python
# === GRID ===
# (batch, head, q_blocks)  where q_blocks = ceil(max_input_len / BLOCK)
# e.g. for prefill with 512 new tokens: (4, 64, 8) = 2048 WGs
BLOCK = 64

# === ONE WORKGROUP: seq_idx=0, q_head_idx=3, q_block_idx=2 ===

def workgroup_pa_prefill(seq_idx, q_head_idx, q_block_idx):
    kv_head_idx = q_head_idx // 8
    input_len = b_seq_len[seq_idx]       # new tokens in this prefill
    past_len  = b_ctx_len[seq_idx]       # existing cached tokens

    q_start = q_block_idx * BLOCK
    if q_start >= input_len:
        return

    # Load Q tile from NEW tokens (not from cache)
    Q_tile = Q_new[b_start_loc[seq_idx] + q_start : ..., q_head_idx]  # [BLOCK, hdim]

    acc = np.zeros([BLOCK, hdim])
    m = np.full([BLOCK], -np.inf)
    l = np.zeros([BLOCK])

    # Part 1: attend to CACHED KV tokens (paged)
    for kv_pos in range(0, past_len, BLOCK):
        K_tile = np.zeros([BLOCK, hdim])
        V_tile = np.zeros([BLOCK, hdim])
        for t in range(BLOCK):
            tok = kv_pos + t
            if tok < past_len:
                phys_block = block_table[seq_idx][tok // block_size]
                offset = tok % block_size
                K_tile[t] = k_cache[phys_block, offset, kv_head_idx, :]
                V_tile[t] = v_cache[phys_block, offset, kv_head_idx, :]

        S = Q_tile @ K_tile.T / sqrt(hdim)
        # ... mask, online softmax update ...

    # Part 2: attend to NEW K/V tokens (contiguous, being generated now)
    for kv_pos in range(0, input_len, BLOCK):
        K_tile = K_new[b_start_loc[seq_idx] + kv_pos : ..., kv_head_idx]
        V_tile = V_new[b_start_loc[seq_idx] + kv_pos : ..., kv_head_idx]

        S = Q_tile @ K_tile.T / sqrt(hdim)
        # Causal mask: new token at position past_len + q_pos can only
        # attend to tokens at positions <= past_len + q_pos
        # ... mask, online softmax update ...

    out[b_start_loc[seq_idx] + q_start : ..., q_head_idx] = acc / l
```

**Key points:**
- Two-phase KV iteration: cached (paged) tokens, then new (contiguous) tokens
- No head-merging: one WG per Q head
- Used during prefill when KV cache already has past context
- GQA via `cur_kv_head = cur_head // num_queries_per_kv`

---

## 9. Comparison

### Feature Matrix

| Feature | CK-PK | CK-UA | CK-SK | CK-Fwd | Triton 2D | Triton 3D | PA Decode | PA Prefill |
|---------|-------|-------|-------|--------|-----------|-----------|-----------|------------|
| Head-merge | No | **Yes** | Yes (decode) | No | **Yes** | **Yes** | No | No |
| Paged KV | Yes | Yes | Yes | **No** | Yes | Yes | Yes | Yes |
| KV splitting | No | No | **Yes** | No | No | **Yes** | V2 only | No |
| Combine kernel | No | No | **Yes** | No | No | **Yes** | V2 only | No |
| Sliding window | Yes | **No** | Yes | Yes | Yes | Yes | No | No |
| Softcap | Yes | **No** | Yes | Yes | Yes | Yes | No | No |
| ALiBi | No | No | No | No | Yes | Yes | No | No |
| Sinks | Yes | **No** | Yes | Yes | Yes | Yes | No | No |
| FP8 output | No | No | No | No | Yes | No | No | No |

### When to Use Each

| Kernel | Best for | Why |
|--------|----------|-----|
| **CK-PK** | Prefill with paged KV | Many Q tokens fill the tile; no split overhead |
| **CK-UA** | Decode, moderate batch (128-256 seqs on MI300X) | Head-merge gives best MFMA utilization for decode |
| **CK-SK** | Decode, low batch + long KV | Split-KV provides parallelism when few sequences |
| **CK-Fwd** | Prefill with contiguous KV | No page indirection overhead |
| **Triton 2D** | Decode/prefill, enough WGs to fill GPU | Head-merge + full feature support (sliding window, softcap) |
| **Triton 3D** | Decode, long KV, low batch | Split-KV + head-merge when 2D can't fill GPU |
| **PA Decode** | Legacy decode path | Being replaced by unified attention |
| **PA Prefill** | Prefill with existing KV cache | Handles both cached and new K/V tokens |

### MFMA Utilization for Decode (1 token, GQA-8)

| Kernel | Q Tile | Useful Rows | Utilization | WGs per seq |
|--------|--------|-------------|-------------|-------------|
| CK-PK (bm0=128) | [128, 64] | 1 | 0.8% | 64 (one per Q head) |
| CK-UA tiny (kBlockM=16) | [16, 64] | 8 | 50% | 8 (one per KV head) |
| Triton 2D (BLOCK_M=16) | [16, 64] | 8 | 50% | 8 (one per KV head) |
| PA Decode V1 | per-head dot | N/A | N/A | 8 (GQA) or 64 (MHA) |

---

## 10. Routing / Selector Logic

### unified_attention() dispatcher

When vLLM calls `unified_attention()`, the kernel is selected in this order:

```python
def unified_attention(q, k, v, out, cu_seqlens_q, max_seqlen_q, ...):
    # Step 1: Try CK-UA (fastest for certain decode shapes)
    if _try_ck_unified_attention(q, k, v, out, ...):
        return  # done

    # Step 2: Choose Triton 2D vs 3D
    num_2d_prgms = num_kv_heads * total_num_q_blocks
    target_prgms = get_num_sms() * 2

    if use_2d_kernel(head_size, sliding_window, all_decode,
                     max_seqlen_q, max_seqlen_k,
                     target_prgms, num_2d_prgms):
        kernel_unified_attention_2d[grid](...)
    else:
        kernel_unified_attention_3d[grid](...)
        reduce_segments[grid](...)
```

### CK-UA selector

```python
def _try_ck_unified_attention(...):
    if max_seqlen_q != 1:          return False  # decode only
    if window_size != (-1, -1):    return False  # no sliding window
    if softcap != 0:               return False  # no softcap
    if alibi_slopes is not None:   return False  # no ALiBi
    if sinks is not None:          return False  # no sinks

    # Only compiled for specific GQA configs
    if not ((head_size == 64 and num_queries_per_kv == 8) or
            (head_size == 128 and num_queries_per_kv == 1)):
        return False

    # CK-UA wins in "moderate occupancy" zone
    cu_count = get_num_sms()                      # 256 on MI300X
    triton_2d_wgs = num_kv_heads * num_seqs
    if not (cu_count * 4 <= triton_2d_wgs <= cu_count * 8):
        return False
    # On MI300X with 8 KV-heads: CK-UA activates for 128-256 seqs

    unified_attention_fwd(out, q, k, v, block_table, seq_lens, ...)
    return True
```

### Triton 2D vs 3D selector

```python
def use_2d_kernel(head_size, sliding_window, all_decode,
                  max_seqlen_q, max_seqlen_k,
                  target_num_prgms, num_2d_prgms):
    return (
        (sliding_window > 0)          # 2D handles sliding window
        or (max_seqlen_k <= 512)      # short KV doesn't need splitting
        or (num_2d_prgms > target_num_prgms)  # already enough WGs
    )
```

### mha_varlen_fwd() dispatcher

When the flash-attention-compatible API is used:

```python
def mha_varlen_fwd(q, k, v, cu_seqlens_q, cu_seqlens_k, block_table, ...):
    if block_table is not None:
        # Paged KV -> CK-SK (split-KV with combine)
        num_splits = heuristic(batch, num_heads_k, max_seqlen_q, head_size)
        # allocate o_acc, lse_acc
        mha_fwd_splitkv(...)   # CK-SK attention + combine
    else:
        # Non-paged -> CK-Fwd
        mha_fwd(...)

# Separate entry for paged prefill without splitting:
def mha_varlen_fwd_pagedkv(q, k, v, block_table, ...):
    mha_fwd_pagedkv(...)  # CK-PK single kernel
```

### num_splits heuristic (CK-SK)

```python
def num_splits_heuristic(batch_nheads_mblocks, num_SMs):
    if batch_nheads_mblocks >= 0.8 * num_SMs:
        return 1   # enough work without splitting

    # Try each split count, pick smallest with good "wave efficiency"
    best_eff = 0
    for s in range(1, min(128, num_SMs) + 1):
        n_waves = (batch_nheads_mblocks * s) / num_SMs
        eff = n_waves / ceil(n_waves)
        best_eff = max(best_eff, eff)

    for s in range(1, min(128, num_SMs) + 1):
        eff = ...  # same calculation
        if eff >= 0.85 * best_eff:
            return s
    return 1
```

---

## Files

| Pipeline | Key files |
|----------|-----------|
| CK-PK | `3rdparty/composable_kernel/include/ck_tile/ops/fmha/kernel/fmha_fwd_pagedkv_kernel.hpp` |
| CK-UA | `3rdparty/composable_kernel/example/ck_tile/42_unified_attention/` |
| CK-SK | `3rdparty/composable_kernel/include/ck_tile/ops/fmha/kernel/fmha_fwd_splitkv_kernel.hpp` |
| CK-Fwd | `3rdparty/composable_kernel/include/ck_tile/ops/fmha/kernel/fmha_fwd_kernel.hpp` |
| Triton 2D/3D | `aiter/ops/triton/_triton_kernels/attention/unified_attention.py` |
| Triton PA Decode | `aiter/ops/triton/_triton_kernels/attention/pa_decode.py` |
| Triton PA Prefill | `aiter/ops/triton/_triton_kernels/attention/pa_prefill.py` |
| Selector | `aiter/ops/triton/attention/unified_attention.py` (`_try_ck_unified_attention`, `use_2d_kernel`) |
| CK-SK/PK wrapper | `aiter/ops/mha.py`, `csrc/py_itfs_ck/mha_varlen_fwd_kernels.cu` |
