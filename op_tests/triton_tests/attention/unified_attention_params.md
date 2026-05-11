# `unified_attention` parameters — annotated reference

Signature (from `aiter/ops/triton/attention/unified_attention.py`):

```python
unified_attention(
    q, k, v, out,
    cu_seqlens_q, max_seqlen_q,
    seqused_k,    max_seqlen_k,
    softmax_scale, causal, window_size,
    block_table, softcap,
    q_descale, k_descale, v_descale,
    alibi_slopes=None, output_scale=None, qq_bias=None,
    sinks=None,
)
```

The kernel implements **paged, varlen, causal attention** for both prefill
and decode. Q is laid out varlen‑packed (sequences concatenated along
axis 0); K/V live in a paged KV cache addressed by a per‑sequence block
table.

---

## A worked example — small mixed batch

Setup (chosen to make every value easy to see):

| Field | Value |
|---|---|
| `num_seqs`        | 3 |
| query lengths     | `[1, 1, 4]`  (seq 0 & 1 are decode, seq 2 is a 4‑token prefill) |
| KV lengths        | `[5, 8, 4]`  (current KV‑cache occupancy per sequence) |
| `num_query_heads` | 8 |
| `num_kv_heads`    | 1   (GQA ratio = 8) |
| `head_size`       | 64 |
| `block_size`      | 4   (KV cache page size) |
| dtype             | bfloat16 |
| `causal`          | `True` |

```
seq 0:  q-tokens = [Q0]                (1 token,  K-len = 5)
seq 1:  q-tokens = [Q1]                (1 token,  K-len = 8)
seq 2:  q-tokens = [Q2a, Q2b, Q2c, Q2d]  (4 tokens, K-len = 4)
                   ───────────────────
total_q_tokens = 1 + 1 + 4 = 6
```

---

## Tensor shapes & dtypes

### `q`, `out`
- shape: `[total_q_tokens, num_query_heads, head_size]` = `[6, 8, 64]`
- dtype: bf16 (or fp16, fp8 with descale)
- Sequences are concatenated along axis 0 in the order described by `cu_seqlens_q`.

### `k`, `v`  (paged KV cache)
- shape: `[num_blocks, block_size, num_kv_heads, head_size]`
- These blocks live in a *global* pool shared across all sequences. They
  are not per‑sequence; the `block_table` tells the kernel which physical
  block holds which logical position for each sequence.
- For our example the cache must hold:
  - seq 0: 5 KV tokens → ⌈5/4⌉ = **2 blocks**
  - seq 1: 8 KV tokens → ⌈8/4⌉ = **2 blocks**
  - seq 2: 4 KV tokens → ⌈4/4⌉ = **1 block**
- Total physical blocks ≥ 5. Say the pool has `num_blocks = 8`.
- shape: `[8, 4, 1, 64]`

---

## Index/length tensors

### `cu_seqlens_q`  — *cumulative* query lengths
- shape: `[num_seqs + 1]`, dtype `int32`
- `cu_seqlens_q[i+1] - cu_seqlens_q[i]` is the q‑length of seq `i`.
- Always starts at 0; last element = `total_q_tokens`.

For our example:
```
cu_seqlens_q = [0, 1, 2, 6]    # boundaries in q[]
                ^  ^  ^  ^
                |  |  |  └── end of seq 2
                |  |  └───── end of seq 1, start of seq 2
                |  └──────── end of seq 0, start of seq 1
                └─────────── start of seq 0
```

### `max_seqlen_q`
- scalar `int`. Upper bound on per‑sequence q length. Used to size autotune choices.
- Example: `max_seqlen_q = 4`.
- Pure decode → `max_seqlen_q == 1` (the kernel takes a faster path).

### `seqused_k`  — *current* KV length per sequence  ← **this is your question**
- shape: `[num_seqs]`, dtype `int32`
- `seqused_k[i]` is the **number of KV tokens currently valid for sequence i**.
- Comes from the inference engine's KV cache bookkeeping (e.g. vLLM stores this
  as the running token count per request). It is NOT the maximum — it is the
  *current* count, which grows by 1 each decode step.
- The kernel only attends to KV positions `[0, seqused_k[i])` for sequence `i`,
  ignoring everything past it (any KV beyond is uninitialized/stale).

For our example:
```
seqused_k = [5, 8, 4]
```

#### Why is it called `seqused_k`?

The name comes from the FlashAttention varlen API, where there are two ways
to communicate the K side of a varlen batch:

| API | What it means | Used by |
|---|---|---|
| `cu_seqlens_k` | Cumulative KV sequence lengths — fully describes the KV layout when K/V are *contiguous* varlen tensors. | classic varlen, no paging |
| **`seqused_k`** | Per‑sequence count of how many KV slots are **"used"** (i.e. currently filled with valid data). | paged KV cache |

Read it as **"sequences‑used K"** = *for each sequence, how many K (and V) slots are presently used.*

The distinction matters because with a paged KV cache:

- The `block_table` only describes *where* KV blocks live — `block_table[i]`
  may point to space for `max_blocks_per_seq * block_size` KV tokens, which
  is much more than what's actually filled.
- The block storage may be **pre‑allocated** (e.g. vLLM grabs blocks for
  several future tokens at once) so most of those slots contain stale or
  uninitialized memory.
- `seqused_k[i]` is what tells the kernel "of the slots reachable through
  `block_table[i]`, only the first `seqused_k[i]` actually contain valid
  K/V — stop there." Without this number, the kernel would read garbage
  past the end of each sequence.

So:
- `cu_seqlens_k` = "where K is, when K is contiguous"
- `seqused_k`    = "how much of K is real, when K is paged"

In the unified-attention kernel only `seqused_k` is needed (paged is the
default), and `cu_seqlens_k` does not appear in the API.

### `max_seqlen_k`
- scalar `int`. `max(seqused_k)`, used for grid sizing / autotune.
- Example: `max_seqlen_k = 8`.

### `block_table`  — paged-KV indirection
- shape: `[num_seqs, max_blocks_per_seq]`, dtype `int32`
- `block_table[i, j]` = physical block id (into `k`/`v`) holding logical KV positions
  `[j*block_size, (j+1)*block_size)` of sequence `i`.
- Slots beyond `⌈seqused_k[i] / block_size⌉` are unused (any value).

For our example with `block_size = 4`, `max_blocks_per_seq = ⌈8/4⌉ = 2`:

```
block_table = [
  [3, 7],    # seq 0: tokens [0..4) live in physical block 3, tokens [4..5) in block 7
  [1, 5],    # seq 1: tokens [0..4) → block 1, tokens [4..8) → block 5
  [2, *],    # seq 2: tokens [0..4) → block 2, second slot unused
]
```

(`*` = don't‑care; the kernel never reads it because `seqused_k[2] = 4`.)

---

## Scalars / behavior flags

### `softmax_scale`
- `float`. Multiplied into QKᵀ before the softmax.
- Conventional value: `1.0 / sqrt(head_size)` = `1.0 / sqrt(64)` = `0.125`.

### `causal`
- `bool`. Must be `True` for this kernel (asserted on entry).
- For each q‑token at logical position `p` in its sequence, only KV positions
  `[0, p+1)` can attend (combined with `seqused_k`).

### `window_size`
- tuple `(left, right)`. `(-1, -1)` disables the sliding window.
- `(W, 0)` restricts attention to KV positions `[p-W, p]` (sliding causal).
- Example: `window_size = (-1, -1)` (no window).

### `softcap`
- `float`. `0.0` disables. Otherwise applies `softcap * tanh(QKᵀ / softcap)`.
- Used by Gemma‑2 / some Llama‑3 variants.

---

## Optional / quantization

### `q_descale`, `k_descale`, `v_descale`
- Per‑tensor or per‑head dequant scales when q/k/v are FP8.
- `None` for bf16/fp16. The kernel asserts `q_descale is None`.

### `output_scale`
- Per‑output FP8 quantization scale. `None` to keep `out` in the input dtype.

### `alibi_slopes`
- shape: `[num_query_heads]` or `None`. Adds the standard ALiBi linear bias.

### `qq_bias`
- Optional `[num_query_heads, max_seqlen_q, max_seqlen_q]` Q‑Q attention bias
  (used by some custom decoders). `None` to disable.

### `sinks`
- shape `[num_query_heads]` or `None`. Adds an "attention sink" virtual token
  per head whose key/value contribution is just a learned scalar (StreamingLLM
  style). Asserted `sinks.shape[0] == num_query_heads` if provided.

---

## End-to-end example tensor allocation (PyTorch)

```python
import torch

num_seqs    = 3
q_lens      = [1, 1, 4]
k_lens      = [5, 8, 4]
nqh, nkh    = 8, 1
hdim        = 64
block_size  = 4
dtype       = torch.bfloat16
device      = "cuda"

total_q     = sum(q_lens)              # 6
max_q       = max(q_lens)              # 4
max_k       = max(k_lens)              # 8
num_blocks  = 8                        # global pool (>= 5 needed)
max_blk_per_seq = (max_k + block_size - 1) // block_size  # 2

q = torch.randn(total_q, nqh, hdim, dtype=dtype, device=device)
k = torch.randn(num_blocks, block_size, nkh, hdim, dtype=dtype, device=device)
v = torch.randn_like(k)
out = torch.empty_like(q)

cu_seqlens_q = torch.tensor([0, 1, 2, 6], dtype=torch.int32, device=device)
seqused_k    = torch.tensor(k_lens,        dtype=torch.int32, device=device)
block_table  = torch.tensor(
    [[3, 7],
     [1, 5],
     [2, 0]],          # second slot of seq 2 is unused; value irrelevant
    dtype=torch.int32, device=device,
)

from aiter.ops.triton.attention.unified_attention import unified_attention

unified_attention(
    q=q, k=k, v=v, out=out,
    cu_seqlens_q=cu_seqlens_q, max_seqlen_q=max_q,
    seqused_k=seqused_k,        max_seqlen_k=max_k,
    softmax_scale=1.0 / (hdim ** 0.5),
    causal=True,
    window_size=(-1, -1),
    block_table=block_table,
    softcap=0.0,
    q_descale=None, k_descale=None, v_descale=None,
)
```

---

## Quick mental model

```
            q[total_q_tokens, NQH, D]
                    │
                    │   sliced by cu_seqlens_q  ──►  per-sequence q rows
                    ▼
   For sequence i, attend to KV[0 .. seqused_k[i])
                    │
                    │   KV positions are paged: position p lives at
                    │   k[block_table[i, p // block_size], p % block_size]
                    ▼
            out[total_q_tokens, NQH, D]
```
Here `p` is a **logical KV position** (token index) within sequence `i`,
ranging over `0 .. seqused_k[i] - 1`. It is the timestep in the key/value
stream that a query token attends to. The paged-KV cache stores tokens in
fixed-size pages of `block_size`, so `p` is decomposed into:
- `p // block_size` → row in `block_table[i]`, which maps to a **physical page id** in the global KV pool `k` / `v`.
- `p % block_size`  → **slot offset** inside that page.
- `cu_seqlens_q` slices **q** on the token axis.
- `seqused_k` clips **how far back** in time each sequence's KV is valid.
- `block_table` translates logical KV positions to **physical pages** in the global pool.

---

## What is actually being computed (math)

Pure math, ignoring all GPU/tiling/numerical-stability details. For each
sequence `i`, each query token `t` in that sequence (logical position
`p_q = t`), and each query head `h`, the kernel computes a single output
vector `o[i, t, h] ∈ ℝ^D`.

### Notation

| Symbol | Meaning |
|---|---|
| $S$ | number of sequences (`num_seqs`) |
| $H_q$ | number of query heads (`num_query_heads`) |
| $H_k$ | number of KV heads (`num_kv_heads`) |
| $G = H_q / H_k$ | GQA group size (= `num_queries_per_kv`) |
| $D$ | head size |
| $L_q^{(i)}$ | query length of sequence $i$ |
| $L_k^{(i)}$ | **KV length of sequence $i$** (= `seqused_k[i]`) |
| $\mathbf{q}_{i,t,h} \in \mathbb{R}^D$ | the q-vector at sequence $i$, token index $t$, head $h$ |
| $\mathbf{k}_{i,p,h_k}, \mathbf{v}_{i,p,h_k} \in \mathbb{R}^D$ | KV vectors at logical position $p$ of seq $i$, KV head $h_k$ (resolved through `block_table`) |
| $s$ | `softmax_scale`, usually $1/\sqrt{D}$ |
| $h_k(h) = \lfloor h / G \rfloor$ | map each query head to its shared KV head |

### Indices, axes, and what gets contracted

Six indices show up. They split into two camps:

**"Free" / batch axes (one independent attention computation per value):**

| Index | Range | Meaning |
|---|---|---|
| $i$ | $0 \le i < S$ | which **sequence** in the batch |
| $t$ | $0 \le t < L_q^{(i)}$ | which **query token** *inside* sequence $i$ (q-local index) |
| $h$ | $0 \le h < H_q$ | which **query head** (e.g. 0..7 for 8 heads) |

For each combination $(i, t, h)$ the kernel produces **one independent
output vector** $\mathbf{o}_{i,t,h} \in \mathbb{R}^D$. Nothing is summed
across $i$, $t$, or $h$ — they're just the axes of the output tensor.

**Reduction axes (these get contracted/summed):**

| Index | Range | What it indexes | Reduced by |
|---|---|---|---|
| $p$ | $0 \le p < L_k^{(i)}$ | a **KV position** in sequence $i$'s timeline | softmax‑weighted sum |
| $D$ (vector axis, no name) | length $D$ | the **head‑dim coordinates** of $\mathbf{q}/\mathbf{k}/\mathbf{v}$ | dot product |

And one **derived** index, not free:

| Index | Definition | Meaning |
|---|---|---|
| $h_k$ | $h_k = \lfloor h / G \rfloor$ | which KV head this query head reads from (GQA mapping) |

So $h_k$ is **not summed over and not free** — it's a deterministic
function of $h$. With MHA ($G=1$) every query head has its own KV head;
with MQA ($H_k=1$) every query head shares the single KV head ($h_k = 0$).

### What each subscript on $\mathbf{q}, \mathbf{k}, \mathbf{v}$ means

- $\mathbf{q}_{i,t,h} \in \mathbb{R}^D$:
  - $i$ = sequence id
  - $t$ = q‑token index inside that sequence (its row in this call's $q$ tensor — ranging 0..$L_q^{(i)}-1$)
  - $h$ = query‑head id
  - the $D$ entries are the head‑dim coordinates of that one query vector
- $\mathbf{k}_{i,p,h_k}, \mathbf{v}_{i,p,h_k} \in \mathbb{R}^D$:
  - $i$ = sequence id (same sequence as the query — attention is per‑sequence)
  - $p$ = KV position inside that sequence (0..$L_k^{(i)}-1$)
  - $h_k$ = KV‑head id, derived from $h$ via $\lfloor h / G \rfloor$
  - these are looked up in the paged cache: `k[block_table[i, p // block_size], p % block_size, h_k, :]`

### The two contractions, in one place

For a fixed free triple $(i, t, h)$:

**1. Logit = contraction over the head‑dimension axis $D$** (a dot product):

$$
  \ell(p) \;=\; s \sum_{d=0}^{D-1} \bigl(\mathbf{q}_{i,t,h}\bigr)_d \;\cdot\; \bigl(\mathbf{k}_{i,p,h_k}\bigr)_d
\qquad \text{for each } p \in \mathcal{P}(i,t)
$$

Each dot product collapses two $D$-dim vectors into a single scalar
logit per KV position $p$. The valid $p$-set $\mathcal{P}(i,t)$ is
chosen by causal mask + sliding window + the `seqused_k`-driven length
clip.

**2. Output = contraction over the KV position axis $p$** (a
softmax‑weighted sum):

$$
  \mathbf{o}_{i,t,h} \;=\; \sum_{p \in \mathcal{P}(i,t)} \underbrace{\frac{\exp \ell(p)}{Z}}_{w(p)} \;\cdot\; \mathbf{v}_{i,p,h_k}
  \qquad
  Z = \sum_{p' \in \mathcal{P}(i,t)} \exp \ell(p')
$$

This collapses the $|\mathcal{P}(i,t)|$ V-vectors into a single $D$-dim
output vector for this $(i, t, h)$.

### Shape cheatsheet: what $p$, $\mathcal{P}(i,t)$, and $\ell(p)$ actually are

Easy to lose track of dimensions, so to be explicit:

| Object | Type / shape | Description |
|---|---|---|
| $p$ | a single **integer** | one KV slot index in sequence $i$'s timeline; $p \in \{0, 1, \dots, L_k^{(i)}-1\}$ |
| $\mathcal{P}(i,t)$ | a **set of integers** (subset of $\{0,\dots,L_k^{(i)}-1\}$) | the KV positions that q-token $(i,t)$ is *allowed* to attend to (after causal + window + length clip) |
| $\lvert \mathcal{P}(i,t) \rvert$ | a single integer | how many KV positions are in that set (between 1 and $L_k^{(i)}$) |
| $\ell(p)$ | a **scalar** ($\in \mathbb{R}$) | the pre-softmax attention score for one specific $p$ |
| $\bigl[\ell(p)\bigr]_{p \in \mathcal{P}(i,t)}$ | a **vector** of length $\lvert \mathcal{P}(i,t) \rvert$ | the full row of logits, one entry per allowed KV position |
| $w(p)$ | a scalar in $[0, 1]$ | post-softmax attention weight for that $p$ (entries sum to 1 across $\mathcal{P}$) |
| $\mathbf{q}_{i,t,h}, \mathbf{k}_{i,p,h_k}, \mathbf{v}_{i,p,h_k}$ | vectors in $\mathbb{R}^D$ | the actual feature vectors (head-dim long) |
| $\mathbf{o}_{i,t,h}$ | a vector in $\mathbb{R}^D$ | the output for this single q-token / head |

So to answer the three questions directly:

- **What is $p$?** A single integer. It indexes one KV position
  (one row of $\mathbf{k}, \mathbf{v}$) for sequence $i$.
- **What is $\mathcal{P}(i,t)$?** A *set* of such integers — the KV
  positions $p$ that q-token $(i,t)$ is allowed to look at, after
  applying causal mask, sliding window, and the $L_k^{(i)}$ length clip
  driven by `seqused_k`.
- **What's the dimension of $\ell(p)$?** A **scalar** for any single
  $p$. As you sweep $p$ over $\mathcal{P}(i,t)$, you accumulate
  $\lvert\mathcal{P}(i,t)\rvert$ scalars — i.e. a length-$\lvert\mathcal{P}\rvert$
  vector of pre-softmax logits per fixed $(i,t,h)$.

### Picture for one $(i,t,h)$ triple

Concrete example with $D = 4$, $L_k = 8$, $\mathcal{P}(i,t) = \{0,1,2,3,4\}$ (so $\lvert\mathcal{P}\rvert = 5$):

```
           p=0   p=1   p=2   p=3   p=4    (← KV positions in P(i,t))

q (D=4): [ q0, q1, q2, q3 ]                ← one D-vector

k:       [ k00,k01,k02,k03 ] [ k10,k11,k12,k13 ] [ k20,..] [ k30,..] [ k40,..]
                                                                              ← five D-vectors

ℓ:        ℓ0    ℓ1    ℓ2    ℓ3    ℓ4       ← five scalars (one per p), shape: |P|

           softmax over these 5 scalars
                       ↓
w:        w0    w1    w2    w3    w4       ← five scalars summing to 1

v:       [ v00,v01,v02,v03 ] [ v10,..]      [ v20,..] [ v30,..] [ v40,..]
                                                                              ← five D-vectors

o = w0·v0 + w1·v1 + w2·v2 + w3·v3 + w4·v4   ← one D-vector (length 4)
```

For $\lvert\mathcal{P}\rvert$ entries the QK matmul produces $\lvert\mathcal{P}\rvert$
scalars (one $\ell(p)$ per $p$), softmax turns them into $\lvert\mathcal{P}\rvert$
weights, and the AV matmul collapses the $\lvert\mathcal{P}\rvert$ V‑vectors back
into one D‑dim output.

### Picture with two q‑tokens ($L_q = 2$)

Same kind of diagram, but now the kernel call is processing **two**
query tokens for one sequence. Take $D = 4$, $L_k = 6$, $L_q = 2$,
causal mask, no window. This represents e.g. a chunked prefill: 4 prior
KV tokens already cached, plus 2 new q‑tokens whose K/V occupy slots 4
and 5.

#### Setup

$$
  \mathrm{abs\_q}(i, 0) = 6 - 2 + 0 = 4 \;\Rightarrow\; \mathcal{P}(i,0) = \{0,1,2,3,4\}\quad (\lvert\mathcal{P}\rvert = 5)
$$

$$
  \mathrm{abs\_q}(i, 1) = 6 - 2 + 1 = 5 \;\Rightarrow\; \mathcal{P}(i,1) = \{0,1,2,3,4,5\}\quad (\lvert\mathcal{P}\rvert = 6)
$$

The second q‑token can see one more KV slot than the first — its own
slot at $p = 5$.

#### Data layout

```
KV cache (6 vectors, each in ℝ^D=4):
              p=0           p=1           p=2           p=3           p=4           p=5
   k_p:  [k00,k01,k02,k03] [k10,..]      [k20,..]      [k30,..]      [k40,..]      [k50,..]
   v_p:  [v00,v01,v02,v03] [v10,..]      [v20,..]      [v30,..]      [v40,..]      [v50,..]
                                                                       ▲             ▲
                                                                       │             │
                                                                  "the new q-tokens' own K/V rows"
                                                                  (q[0] is at slot 4, q[1] is at slot 5)


Q tensor (2 vectors, each in ℝ^D=4):
              t=0           t=1
   q:    [q00,q01,q02,q03] [q10,q11,q12,q13]
         └── q_{i,0,h} ──┘ └── q_{i,1,h} ──┘
```

#### Each q‑token does its own attention computation

**For $t = 0$** ($\mathrm{abs\_q} = 4$, $\lvert\mathcal{P}\rvert = 5$):

```
q[0] ·ᵈ k[p]    for p ∈ {0,1,2,3,4}   →  ℓ_0(p) = 5 scalars

ℓ_0:           ℓ0(0)  ℓ0(1)  ℓ0(2)  ℓ0(3)  ℓ0(4)         ← length-5 vector

softmax → w_0:  w0(0)  w0(1)  w0(2)  w0(3)  w0(4)         ← 5 weights, sum to 1

o[0] = Σ_{p ∈ P(0)} w_0(p) · v[p]                         ← length-D=4 vector
     = w0(0)·v[0] + w0(1)·v[1] + w0(2)·v[2] + w0(3)·v[3] + w0(4)·v[4]
```

**Cannot** see `k[5]` / `v[5]` — that would be looking ahead in time
(slot 5 is q[1]'s own slot, which is later than q[0]'s slot 4).

**For $t = 1$** ($\mathrm{abs\_q} = 5$, $\lvert\mathcal{P}\rvert = 6$):

```
q[1] ·ᵈ k[p]    for p ∈ {0,1,2,3,4,5}  →  ℓ_1(p) = 6 scalars

ℓ_1:           ℓ1(0)  ℓ1(1)  ℓ1(2)  ℓ1(3)  ℓ1(4)  ℓ1(5)   ← length-6 vector

softmax → w_1:  w1(0)  w1(1)  w1(2)  w1(3)  w1(4)  w1(5)   ← 6 weights, sum to 1

o[1] = Σ_{p ∈ P(1)} w_1(p) · v[p]                          ← length-D=4 vector
     = w1(0)·v[0] + w1(1)·v[1] + ... + w1(5)·v[5]
```

Sees **all 6** KV positions, including its own slot at $p = 5$.

#### The dot products, written out

For $t = 0$ each logit is a $D$-dim dot product:

$$
  \ell_0(p) \;=\; s\,\sum_{d=0}^{D-1}\, \mathbf{q}_{i,0,h,d}\;\mathbf{k}_{i,p,h_k,d}
  \qquad p \in \{0,1,2,3,4\}
$$

i.e. concretely (with $D=4$ and $s = 1/\sqrt{D}$):

$$
  \ell_0(p)
  \;=\; s \;\bigl(q_{0,0}\,k_{p,0} \;+\; q_{0,1}\,k_{p,1} \;+\; q_{0,2}\,k_{p,2} \;+\; q_{0,3}\,k_{p,3}\bigr)
$$

producing 5 scalars $\bigl(\ell_0(0), \ldots, \ell_0(4)\bigr)$.

For $t = 1$ identical formula but $p$ ranges over $\{0,\dots,5\}$ and
the q-vector is $\mathbf{q}_{i,1,h}$ instead, producing 6 scalars.

The output collapses the |P| V-vectors back to one D-vector per t:

$$
  o_{i,0,h,d} \;=\; \sum_{p=0}^{4}\, w_0(p)\, v_{i,p,h_k,d}
  \qquad
  o_{i,1,h,d} \;=\; \sum_{p=0}^{5}\, w_1(p)\, v_{i,p,h_k,d}
$$

each evaluated for $d = 0, 1, 2, 3$.

#### Matrix notation (the textbook one-liner)

If you stack the q-tokens into a matrix $Q$ and the KV cache into
matrices $K$ and $V$, the whole computation becomes:

$$
  O \;=\; \mathrm{softmax}\!\left(\frac{Q\,K^{\!\top}}{\sqrt{D}} \;+\; M\right)\, V
$$

where:

- $Q \in \mathbb{R}^{L_q \times D}$  — q-tokens as rows (here $2 \times 4$)
- $K \in \mathbb{R}^{L_k \times D}$  — KV cache K-vectors as rows (here $6 \times 4$)
- $V \in \mathbb{R}^{L_k \times D}$  — KV cache V-vectors as rows (here $6 \times 4$)
- $M \in \mathbb{R}^{L_q \times L_k}$ — additive mask: $0$ where allowed, $-\infty$ where masked (here $2 \times 6$)
- $O \in \mathbb{R}^{L_q \times D}$  — output, q-tokens as rows (here $2 \times 4$)
- $\mathrm{softmax}$ is **row-wise** (each row independently sums to 1)

#### Concrete shapes for our 2-token example

```
Q    [2, 4]      K    [6, 4]      Kᵀ   [4, 6]      Q · Kᵀ   [2, 6]    (logits before scale/mask)

                                                  ┌─                          ─┐
                                                  │  ℓ0(0)  ℓ0(1) ... ℓ0(4) ℓ0(5) │
                                                  │  ℓ1(0)  ℓ1(1) ... ℓ1(4) ℓ1(5) │
                                                  └─                          ─┘

scale by 1/√D, then add causal mask M:

M    [2, 6]   =   ┌  0   0   0   0   0  -∞  ┐    ← row t=0: only p≤4 allowed
                  └  0   0   0   0   0   0  ┘    ← row t=1: all p allowed

→ row-wise softmax  →  W [2, 6]   (each row sums to 1, with W[0,5]=0)

W · V  [2, 6] · [6, 4]  =  O  [2, 4]              ← final output
```

The kernel doesn't actually allocate the full $2 \times 6$ logit matrix
— it walks the valid $p$-set per row and skips the $-\infty$ entries.
But this matrix form is the classic "scaled dot-product attention"
formulation from *Attention Is All You Need*; everything else
(`abs_q`, $\mathcal{P}(i,t)$, paged KV, varlen) is just bookkeeping
that turns this same matrix expression into something efficient for
ragged batches and paged caches.

#### Multi-head matrix form

Adding the head axis (and treating GQA's $h_k = \lfloor h/G \rfloor$ as
a broadcast on K/V):

$$
  O_h \;=\; \mathrm{softmax}\!\left(\frac{Q_h\,K_{h_k(h)}^{\!\top}}{\sqrt{D}} + M\right)\, V_{h_k(h)}
  \qquad \text{for each } h \in \{0, \ldots, H_q-1\}
$$

##### How many $O_h$ matrices do we get?

We get **one $O_h$ per query head**, so **$H_q$ of them**. For our
running example ($H_q = 8$, $H_k = 1$): **8 output matrices**, with all
8 sharing the same $K$ and $V$ (MQA — every query head maps to the
single KV head via $h_k(h) = \lfloor h/G \rfloor = 0$).

More generally:

| GQA regime | $H_q$ | $H_k$ | $G$ | # of $O_h$ | KV sharing |
|---|---:|---:|---:|---:|---|
| MHA | $H_q$ | $H_q$ | 1 | $H_q$ | each head has its own K/V |
| GQA | $H_q$ | $H_k$ | $H_q / H_k$ | $H_q$ | groups of $G$ query heads share one K/V |
| MQA | $H_q$ | 1 | $H_q$ | $H_q$ | all query heads share **one** K/V (our case) |

The number of $O_h$ matrices is **always $H_q$** — what GQA changes is
how many *distinct* $K_{h_k(h)} / V_{h_k(h)}$ slices feed into them.

##### Dimensions

| Symbol | Shape | Notes |
|---|---|---|
| $Q_h$ | $L_q \times D$ | one slice per query head; we have $H_q$ of them in total |
| $K_{h_k(h)}$, $V_{h_k(h)}$ | $L_k \times D$ | only $H_k$ distinct slices; $G = H_q/H_k$ query heads share each one |
| $K_{h_k(h)}^{\!\top}$ | $D \times L_k$ | transposed for the matmul |
| $Q_h K_{h_k(h)}^{\!\top}$ | $L_q \times L_k$ | logits before scaling/masking |
| $M$ | $L_q \times L_k$ | causal/window mask, shared across heads |
| $\mathrm{softmax}(\dots)$ | $L_q \times L_k$ | row-wise softmax over the $L_k$ axis |
| $O_h$ | $L_q \times D$ | per-head output; **$H_q$ such matrices** |
| $O$ (stacked) | $H_q \times L_q \times D$ | the full output for one sequence's chunk |

So you get **one $L_q \times D$ output matrix per query head**, and
**$H_q$ matrices in total** — all sharing the same $M$, all using
head-specific $Q_h$, and (after GQA broadcasting) shared $K_{h_k(h)}$ /
$V_{h_k(h)}$.

##### Concrete numbers

For the 2-token picture ($L_q = 2, L_k = 6, D = 4$) with our example
heads ($H_q = 8, H_k = 1, G = 8$):

- 8 query heads → **8 output matrices** $O_0, \ldots, O_7$.
- Each $O_h$ has shape $2 \times 4$.
- All 8 share the *same* $K$ and $V$ (MQA: $h_k(h) = 0$ for every $h$).
- Stacked: $O$ is shape $8 \times 2 \times 4 = 64$ scalars per sequence.

For the rocprofv3 decode benchmark ($L_q = 1, L_k = 4096, D = 64,
H_q = 8, H_k = 1$) per sequence:

- 8 output matrices, each $1 \times 64$.
- Stacked: $O$ is shape $8 \times 1 \times 64 = 512$ scalars per sequence
  ($248 \times 512 = 126{,}976$ across the full 248-sequence batch).

In the actual storage (PyTorch / aiter convention), the output tensor
is laid out as `out[total_q_tokens, H_q, D]` — i.e. $L_q$ on the
outermost axis, then heads, then head-dim — so `out[t, h, :]` reads
out the row $t$ of $O_h$ (a length-$D$ vector).

#### Einsum-style (the whole thing in one go)

If we lift both q-tokens at once and write it with explicit indices:

$$
  \ell_{i,t,h,p}
  \;=\; s \sum_{d=0}^{D-1} q_{i,t,h,d}\, k_{i,p,h_k(h),d}
  \qquad
  \text{for } t \in \{0,1\},\; p \in \mathcal{P}(i,t)
$$

$$
  o_{i,t,h,d}
  \;=\; \sum_{p \in \mathcal{P}(i,t)}
        \mathrm{softmax}_p\!\bigl(\ell_{i,t,h,\cdot}\bigr)(p)\;
        v_{i,p,h_k(h),d}
$$

Note that the softmax is **separately** applied to each row indexed by
$t$: row $t=0$ has 5 entries, row $t=1$ has 6 entries — they are
*independent* probability distributions, not one big softmax over 11
logits.

In NumPy/PyTorch einsum syntax (ignoring the masking, which is applied
on top):

```python
# shapes:
#   q:  (Lq, D)    here (2, 4)
#   K:  (Lk, D)    here (6, 4)
#   V:  (Lk, D)    here (6, 4)

logits = s * np.einsum("td, pd -> tp", q, K)    # (2, 6)  ← contract over D
logits = causal_mask(logits, abs_q=[4, 5])      # set upper-right to -inf
weights = softmax(logits, axis=-1)              # (2, 6)  ← softmax over p
o = np.einsum("tp, pd -> td", weights, V)       # (2, 4)  ← contract over p
```

Reading the einsum strings:

- `"td, pd -> tp"` — $d$ is contracted (head‑dim dot product); $t$ and $p$ become the two output axes (the logit matrix).
- `"tp, pd -> td"` — $p$ is contracted (softmax‑weighted V combine); $t$ and $d$ become the output axes (the final $L_q \times D$ output).

So in einsum terms, attention is just **two contractions back-to-back
with a softmax in the middle**: contract `D`, mask, softmax over `p`,
contract `p`.

#### The "logit matrix" view (with causal mask)

If you imagined a fully padded matrix of shape $L_q \times L_k = 2 \times 6$:

```
            p=0    p=1    p=2    p=3    p=4    p=5
  t=0  [   ℓ0(0)  ℓ0(1)  ℓ0(2)  ℓ0(3)  ℓ0(4)   −∞   ]   ← masked: p=5 > abs_q=4
  t=1  [   ℓ1(0)  ℓ1(1)  ℓ1(2)  ℓ1(3)  ℓ1(4)  ℓ1(5) ]   ← p=5 allowed: abs_q=5

  softmax independently along each row
                       ↓

            p=0    p=1    p=2    p=3    p=4    p=5
  t=0  [   w0(0)  w0(1)  w0(2)  w0(3)  w0(4)   0    ]   ← e^(-∞) = 0
  t=1  [   w1(0)  w1(1)  w1(2)  w1(3)  w1(4)  w1(5) ]
```

That's the classic **causal triangle**: each row has at most
$\mathrm{abs\_q}+1$ non‑zero entries. The `unified_attention` kernel
doesn't actually materialize this padded matrix — it iterates $p \in
\mathcal{P}(i,t)$ directly and skips the masked entries — but
conceptually this is what's happening.

#### Final output shape (per head $h$)

```
o = [ o[0], o[1] ]   ← two D=4 vectors

with    o[0] ∈ ℝ^4   from softmax over 5 logits, weighted sum of 5 V-vectors
        o[1] ∈ ℝ^4   from softmax over 6 logits, weighted sum of 6 V-vectors
```

#### What stays the same vs the 1‑token picture

- The **per‑$(i, t, h)$ recipe** is identical: dot Q with the allowed
  K's → softmax → weighted sum of the allowed V's → one D‑vector. The
  only thing that changes per q‑token is **which** K/V columns count,
  i.e. $\mathcal{P}(i, t)$.
- $\lvert\mathcal{P}\rvert$ varies row by row (5, then 6) — that's why
  this is "ragged" attention, and why the causal mask conceptually puts
  $-\infty$ in the upper‑right of the logit matrix.
- Adding more q‑tokens just adds more rows to that picture; the K/V
  "columns" stay the same (one per KV cache slot).

### Concrete sizes from our 3-sequence example

Recall: `q_lens=[1,1,4]`, `k_lens=[5,8,4]`, causal, no window/sinks.

| $(i, t)$ | $\mathrm{abs\_q}$ | $\mathcal{P}(i,t)$ | $\lvert\mathcal{P}\rvert$ | $\ell$ shape (per head) |
|---|---:|---|---:|---|
| (0, 0) | 4 | $\{0,1,2,3,4\}$        | 5 | length-5 vector of scalars |
| (1, 0) | 7 | $\{0,1,2,3,4,5,6,7\}$  | 8 | length-8 vector |
| (2, 0) | 0 | $\{0\}$                | 1 | length-1 vector (just one scalar) |
| (2, 1) | 1 | $\{0,1\}$              | 2 | length-2 vector |
| (2, 2) | 2 | $\{0,1,2\}$            | 3 | length-3 |
| (2, 3) | 3 | $\{0,1,2,3\}$          | 4 | length-4 |

So $\mathcal{P}$ is **not the same size for every q-token** — that's
exactly why the kernel needs ragged/variable-length handling. Each row
of logits has its own length.

### Einsum-style summary

In the spirit of `numpy.einsum`/PyTorch convention (with the GQA
broadcast over the head axis written explicitly):

$$
  \ell_{i,t,h,p} \;=\; s\;\sum_{d}\, q_{i,t,h,d}\, k_{i,p,\lfloor h/G \rfloor,d}
  \quad\quad
  o_{i,t,h,d} \;=\; \sum_{p}\, \mathrm{softmax}_p(\ell_{i,t,h,p})\, v_{i,p,\lfloor h/G \rfloor,d}
$$

Reading this:

- The first equation contracts over $d$ (head‑dim).
- The second contracts over $p$ (KV positions), but weighted by the
  *normalized* logits.
- $i, t, h$ never contract — they enumerate independent outputs.
- $h_k$ only appears via $\lfloor h/G \rfloor$ — it's just GQA broadcasting:
  many query heads $h$ share the same KV slice.

### A counting sanity check

For our 3‑sequence example (`q_lens=[1,1,4]`, `k_lens=[5,8,4]`,
`H_q=8`, `H_k=1`, `D=64`):

- Total *output vectors* produced = $\sum_i L_q^{(i)} \cdot H_q = (1+1+4) \cdot 8 = 48$ vectors of size $D=64$.
- Total *dot products* computed = $\sum_i L_q^{(i)} \cdot H_q \cdot |\mathcal{P}(i,t)|$, summed over $t$:
  - seq 0:  $1 \cdot 8 \cdot 5 = 40$
  - seq 1:  $1 \cdot 8 \cdot 8 = 64$
  - seq 2:  $8 \cdot (1 + 2 + 3 + 4) = 80$ (causal mask shrinks $\mathcal{P}$ per token)
  - **total = 184 dot products, each of length 64**.
- Each dot product touches $D = 64$ multiplies — i.e. $184 \times 64 = 11{,}776$ scalar mults for the QK part, then a similar count for the AV part.

That's the entire computation, before any GPU tiling.

### The key alignment formula: what is `abs_q`?

$$
  \boxed{\;\mathrm{abs\_q}(i, t) \;=\; L_k^{(i)} \;-\; L_q^{(i)} \;+\; t\;}
$$

In words: **the absolute position, in the full sequence's timeline,
of the $t$-th query token of sequence $i$.**

This is the single most important piece of indexing in the kernel,
because the causal mask, sliding window, and ALiBi bias are all defined
in terms of *absolute* positions — not in terms of the query‑local
index $t$.

#### What you might be missing: how the KV cache is updated

Most of the confusion around `Lk - Lq + t` evaporates once you accept
**one convention used by paged-attention engines (vLLM, SGLang, …)**:

> **Before this attention kernel runs, the K and V vectors of the
> *current* query tokens have already been written into the KV cache,
> and `seqused_k[i]` (= $L_k^{(i)}$) has already been bumped to include
> them.**

So at the moment the kernel executes:

- $L_k^{(i)}$ counts **everything**: old context **plus** the new q‑tokens.
- The Q tensor only carries the new q‑tokens (the small thing).
- The KV cache (`k`, `v`) carries old context **and** the new q‑tokens' K/V.
- Therefore the new q‑tokens occupy the **last $L_q^{(i)}$ slots** of the
  KV cache, at absolute positions
  $L_k^{(i)} - L_q^{(i)},\, L_k^{(i)} - L_q^{(i)} + 1,\, \dots,\, L_k^{(i)} - 1$.

Picture (for one sequence):

```
absolute pos:   0    1    2   ...   Lk-Lq-1     Lk-Lq    Lk-Lq+1   ...   Lk-1
KV cache  k:    ●    ●    ●   ...      ●           ●         ●     ...     ●
KV cache  v:    ●    ●    ●   ...      ●           ●         ●     ...     ●
                                                   │         │             │
                                                  q[0]      q[1]   ...   q[Lq-1]
              └─────────── "old" context ──────┤└───────── "new" q-tokens ────────┤
              (already attended in earlier calls)    (whose Q is in this call's q)
```

So `q[t]` is the **t-th of the trailing block**, and its absolute KV
position is therefore:

$$
  \mathrm{abs\_q}(i,t) \;=\; \underbrace{\bigl(L_k^{(i)} - L_q^{(i)}\bigr)}_{\text{first new slot}} \;+\; t.
$$

That's the formula. The intuition: **q‑tokens live at the end of the
cache**.

#### A timeline that should make it click

Imagine generating *"Hello world how are you doing ?"* one forward pass
at a time:

| Pass | What the engine does | Lq | Lk *(at the kernel call)* | abs_q for each t |
|---|---|---:|---:|---|
| **Prefill** | Receive prompt of 5 tokens. Compute K/V for all 5, write into the cache. Call attention with q=5, cache=5. | **5** | **5** | t=0..4 → 0..4 |
| **Decode 1** | Generate next token. Compute K/V for the *one* new token, append to cache. Call attention with q=1, cache=6. | **1** | **6** | t=0 → 5 |
| **Decode 2** | Append → cache=7. Call with q=1. | **1** | **7** | t=0 → 6 |
| **Decode 3** | Append → cache=8. Call with q=1. | **1** | **8** | t=0 → 7 |

Verify with the formula:

- Prefill: $\mathrm{abs\_q}(i, t) = 5 - 5 + t = t$. ✓ (q-token $t$ is at slot $t$)
- Decode step at token 6: $\mathrm{abs\_q} = 6 - 1 + 0 = 5$. ✓ (the new token is the 6th, index 5)
- Decode step at token 7: $\mathrm{abs\_q} = 7 - 1 + 0 = 6$. ✓
- Decode step at token 8: $\mathrm{abs\_q} = 8 - 1 + 0 = 7$. ✓

**Common gotcha #1.** Some readers expect "$L_k$ = how many tokens were
*already* in the cache **before** this call" — i.e. the old context only.
**That's not the convention here.** $L_k$ is the *post‑update* length,
including the new q‑tokens. (If it were pre‑update, the formula would
be `Lk_old + t`, not `Lk - Lq + t`.)

**Common gotcha #2.** "Why doesn't $t$ alone work for masking?"
Because $t$ is just the q‑token's offset *inside this kernel call's q
tensor* — it doesn't know how much KV history precedes it. In pure
decode $t = 0$ for the single new token, but that token's *real*
position might be slot 7, slot 1023, or slot 31999. Causal masking has
to compare against the real position, not against $t$.

#### Three concrete cases

| Regime | $L_k$ | $L_q$ | $t$ | $\mathrm{abs\_q}$ | Interpretation |
|---|---:|---:|---:|---:|---|
| Pure decode (1 new q-token) | 8 | 1 | 0 | **7** | the new q-token is the latest, slot 7 |
| Pure prefill (cold start)   | 5 | 5 | 0..4 | **0..4** | q-token $t$ is naturally at slot $t$ |
| Chunked prefill / spec decode | 10 | 4 | 0..3 | **6, 7, 8, 9** | 4 new q-tokens occupy slots 6..9 |

#### Why we need it (causal/window)

For each q‑token we then enforce, in *absolute* coordinates:

$$
  \text{(causal)} \quad p \;\le\; \mathrm{abs\_q}(i,t)
  \qquad\text{and}\qquad
  \text{(window)} \quad \mathrm{abs\_q}(i,t) - W_\ell \;\le\; p \;\le\; \mathrm{abs\_q}(i,t) + W_r
$$

That's it: `abs_q` exists purely so the *same* causal/window logic works
for prefill, decode, and chunked/mixed batches without any special
casing.

### Per-(sequence, query-token, head) computation

For a fixed $(i, t, h)$, with $\mathrm{abs\_q} = L_k^{(i)} - L_q^{(i)} + t$
and $h_k = \lfloor h / G \rfloor$:

**1. Valid KV-position set**

$$
  \mathcal{P}(i, t) \;=\; \bigl\{\, p \in \mathbb{Z} \;:\;
        0 \le p < L_k^{(i)},\;
        p \le \mathrm{abs\_q}\;\;\text{(causal)},\;
        \mathrm{abs\_q} - W_\ell \le p \le \mathrm{abs\_q} + W_r\;\;\text{(window)}\,\bigr\}
$$

(The window clause is dropped when `window_size = (-1, -1)`.)

**2. Pre-softmax logits**

$$
  \ell(p) \;=\; s\;\bigl\langle\, \mathbf{q}_{i,t,h},\; \mathbf{k}_{i,p,h_k}\,\bigr\rangle
            \quad\text{for each } p \in \mathcal{P}(i,t)
$$

Optional modifiers (applied in this order if enabled):

$$
  \begin{aligned}
    \text{softcap:}\quad &\ell(p) \;\leftarrow\; c\;\tanh\!\bigl(\ell(p)/c\bigr),\quad c=\texttt{softcap} \\
    \text{ALiBi:}\quad   &\ell(p) \;\leftarrow\; \ell(p) \;+\; \alpha_h\,(p - \mathrm{abs\_q}),\quad \alpha_h=\texttt{alibi\_slopes}[h] \\
    \text{qq-bias:}\quad &\ell(p) \;\leftarrow\; \ell(p) \;+\; \texttt{qq\_bias}[h, t, \cdot]
  \end{aligned}
$$

**3. Softmax weights**

Without sinks:

$$
  w(p) \;=\; \frac{\exp \ell(p)}{Z},\qquad
  Z \;=\; \sum_{p' \in \mathcal{P}} \exp \ell(p')
$$

With attention sinks ($\sigma_h$ := `sinks[h]` is a learned scalar logit):

$$
  Z \;=\; \exp \sigma_h \;+\; \sum_{p' \in \mathcal{P}} \exp \ell(p'),\qquad
  w(p) \;=\; \frac{\exp \ell(p)}{Z}
$$

The sink contributes mass $\exp(\sigma_h)/Z$ to the denominator only —
no virtual K/V vector is added to the output.

**4. Output**

$$
  \mathbf{o}_{i,t,h} \;=\; \sum_{p \in \mathcal{P}(i,t)} w(p)\;\mathbf{v}_{i,p,h_k} \;\in\; \mathbb{R}^D
$$

Optional FP8 output: $\mathrm{out}_{i,t,h} = \operatorname{quant}\bigl(\mathbf{o}_{i,t,h} \cdot \texttt{output\_scale}\bigr)$.

### Compact one-liner

For brevity, the entire kernel is just (per sequence $i$, query token $t$, head $h$):

$$
  \mathbf{o}_{i,t,h}
  \;=\;
  \mathrm{softmax}_p\!\Bigl(s\, \mathbf{q}_{i,t,h}^\top \mathbf{K}_{i,h_k}\Bigr)\bigm|_{p \in \mathcal{P}(i,t)}
  \;\cdot\; \mathbf{V}_{i,h_k}
$$

with $\mathcal{P}(i,t)$ capturing causality + sliding window + the
`seqused_k`-driven length clip, and $\mathbf{K}_{i,h_k}, \mathbf{V}_{i,h_k}$
being the per‑sequence per‑head paged KV materialized through `block_table`.

### Pseudocode (full kernel, math-only)

```python
for i in range(num_seqs):                       # each sequence
    Lq_i = cu_seqlens_q[i+1] - cu_seqlens_q[i]
    Lk_i = seqused_k[i]
    base_q = cu_seqlens_q[i]                    # row offset into q/out

    for t in range(Lq_i):                       # each query token
        abs_q = Lk_i - Lq_i + t                 # KV position aligned with this q-token

        for h in range(num_query_heads):        # each query head
            h_k = h // (num_query_heads // num_kv_heads)

            # 1. logits over valid KV positions
            logits = []
            valid_p = []
            for p in range(Lk_i):
                if p > abs_q:                       continue   # causal
                if window_size[0] >= 0 and p < abs_q - window_size[0]: continue
                if window_size[1] >= 0 and p > abs_q + window_size[1]: continue

                # paged-KV indirection
                phys_block = block_table[i, p // block_size]
                k_vec = k[phys_block, p % block_size, h_k, :]
                v_vec = v[phys_block, p % block_size, h_k, :]

                ell = softmax_scale * dot(q[base_q + t, h, :], k_vec)

                if softcap > 0:
                    ell = softcap * tanh(ell / softcap)
                if alibi_slopes is not None:
                    ell = ell + alibi_slopes[h] * (p - abs_q)

                logits.append(ell)
                valid_p.append(p)

            # 2. softmax (with optional sink)
            denom_extra = exp(sinks[h]) if sinks is not None else 0.0
            Z = denom_extra + sum(exp(l) for l in logits)
            w = [exp(l) / Z for l in logits]

            # 3. weighted sum of V
            o = zeros(head_size)
            for p, wi in zip(valid_p, w):
                phys_block = block_table[i, p // block_size]
                v_vec = v[phys_block, p % block_size, h_k, :]
                o = o + wi * v_vec

            out[base_q + t, h, :] = o
            # if output_scale is not None: out[...] = quantize(o * output_scale)
```

### Specializing to common cases

- **Pure decode** (`Lq[i] = 1` for all `i`):
  `t = 0`, `abs_q = Lk[i] − 1`, so each q‑token attends to *all* `Lk[i]`
  KV positions (no causal masking actually trims anything, since the q‑token
  *is* the latest KV position).

- **Pure prefill, single sequence** (`S = 1`, `Lq[0] = Lk[0]`):
  Every q‑token `t` attends to KV positions `[0, t]` — the textbook causal
  self‑attention with `Lq` rows and `Lk = Lq` columns.

- **MHA** (`G = 1`):
  `h_k = h`, every query head has its own dedicated KV.

- **MQA** (`H_k = 1`):
  All query heads share one KV head; `h_k = 0` for every `h`.

### Worked numbers for our 3-sequence example

Using `seqused_k = [5, 8, 4]`, `q_lens = [1, 1, 4]`, `head_size = 64`,
`softmax_scale = 1/√64 = 0.125`, causal, no window/softcap/alibi/sinks:

| seq i | t | abs_q | Valid KV positions |
|------:|--:|------:|---|
| 0 | 0 | 4 | `{0,1,2,3,4}`         (5 positions) |
| 1 | 0 | 7 | `{0,1,2,3,4,5,6,7}`   (8 positions) |
| 2 | 0 | 0 | `{0}`                 (1 position) |
| 2 | 1 | 1 | `{0,1}`               (2 positions) |
| 2 | 2 | 2 | `{0,1,2}`             (3 positions) |
| 2 | 3 | 3 | `{0,1,2,3}`           (4 positions) |

For each `(i, t)` the kernel produces 8 output vectors (one per query head)
of size 64, computed as `softmax(0.125 · Q · Kᵀ) · V` over those KV columns,
with all 8 heads sharing the same `K`/`V` (MQA, `H_k = 1`).

---

## GPU tiling / launch parameters (`BLOCK_M`, `BLOCK_Q`, `TILE_SIZE`)

Everything above describes **what** the kernel computes — pure math, no
GPU. This section describes **how** the work is split into tiles for the
Triton kernel. Three integer constants drive the tiling:

| Constant | Axis it tiles | Conventional name in attention literature |
|---|---|---|
| `BLOCK_M` | "expanded" Q‑rows axis (after replicating each q‑token $G$ times for GQA) | $M$ — the row dim of the QK matmul |
| `BLOCK_Q` | **original** q‑tokens per workgroup (`= BLOCK_M / G`) | (no standard name; aiter‑specific) |
| `TILE_SIZE` | KV positions per inner step | $N$ — i.e. `BLOCK_N` in FlashAttention papers |

> Note on naming: this Triton kernel uses **`TILE_SIZE`** for what
> FlashAttention‑style kernels call **`BLOCK_N`**. There is no symbol
> literally named `BLOCK_N` in `aiter/ops/triton/attention/unified_attention.py` —
> mentally treat `TILE_SIZE = BLOCK_N`.

### `BLOCK_M` — *(q‑token, head)*‑pair tile (the M‑axis of the QK matmul)

> **Heads up:** `BLOCK_M` is **not** "q‑tokens × q‑tokens" and **not**
> just "q‑tokens". It is the **fused *(q‑token, query‑head)* axis** of
> the QK matmul. The hard identity is
>
> $$\boxed{\;\mathrm{BLOCK\_M} \;=\; \mathrm{BLOCK\_Q} \;\times\; G\;}$$
>
> where $G = H_q / H_k = $ `num_queries_per_kv`.
> So if `BLOCK_Q = 2` distinct q‑tokens and $G = 8$ heads share one KV
> head, then `BLOCK_M = 16` rows — **2 tokens × 8 heads = 16 (token, head) pairs**,
> not "16 tokens".

You can see this directly in the kernel — each row $m$ of the Q tile
($0 \le m < \mathrm{BLOCK\_M}$) maps to:

```132:138:aiter/aiter/ops/triton/_triton_kernels/attention/unified_attention.py
    offs_m = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_SIZE_PADDED)
    offs_t = tl.arange(0, TILE_SIZE)
    query_pos = q_block_local_idx * BLOCK_Q + offs_m // num_queries_per_kv
    query_offset_1 = kv_head_idx * num_queries_per_kv + offs_m % num_queries_per_kv
```

- q‑token index = `q_block_local_idx * BLOCK_Q + m // G` → each q‑token contributes $G$ consecutive rows.
- query‑head index = `kv_head_idx * G + m % G` → cycles through the $G$ heads in the GQA group.

The QK matmul's output then has shape `(BLOCK_M, TILE_SIZE)` = $((q\text{-token}, \text{head})\text{-pairs} ,\, \text{KV positions})$, with all $G$ heads in the group sharing a *single* K/V load — that's the whole point of fusing the head axis into M instead of looping over heads.

**Naming‑map vs. FlashAttention literature**

| FlashAttention paper | aiter `unified_attention` |
|---|---|
| `BLOCK_M` = q‑tokens | `BLOCK_M` = q‑tokens × $G$ &nbsp;(= `BLOCK_Q · num_queries_per_kv`) |
| `BLOCK_N` = k‑tokens | `TILE_SIZE` = k‑tokens |
| (no equivalent) | `BLOCK_Q` = q‑tokens (the "real" token count) |

`BLOCK_M` is selected as follows:

```255:259:aiter/aiter/ops/triton/attention/unified_attention.py
    BLOCK_M = (
        16 if num_queries_per_kv <= 16 else triton.next_power_of_2(num_queries_per_kv)
    )
    BLOCK_Q = BLOCK_M // num_queries_per_kv
    assert BLOCK_Q >= 1
```

So:

- defaults to **16** when GQA group size $G \le 16$ (so a single q‑token's
  whole GQA group fits in one tile and `BLOCK_Q ≥ 1`),
- is rounded up to the next power of 2 when $G > 16$,
- is bumped to **64 (RDNA) / 128 (CDNA)** for **large prefills** (`max_seqlen_q ≥ 256`),
  to amortize per‑tile overhead over more *(token, head)* rows.

### `BLOCK_Q` — original q‑tokens per workgroup

This is the user‑facing tile size on the q‑sequence axis:

$$
  \boxed{\;\mathrm{BLOCK\_Q} \;=\; \frac{\mathrm{BLOCK\_M}}{G}\;}
  \qquad \text{where } G = H_q / H_k = \texttt{num\_queries\_per\_kv}.
$$

It tells you **how many real q‑tokens one workgroup processes**. Each
of those q‑tokens is then replicated $G$ times inside the workgroup to
form the $\mathrm{BLOCK\_M}$ rows.

Number of workgroups along the q‑axis (used to pick the kernel grid):

$$
  \mathrm{total\_num\_q\_blocks} \;=\; \left\lfloor \frac{\texttt{q.shape[0]}}{\mathrm{BLOCK\_Q}} \right\rfloor + \texttt{num\_seqs}
$$

(The `+ num_seqs` is an upper bound that absorbs the per‑sequence ceiling
without realizing per‑sequence query lengths on the host — see the
comment in the kernel launcher.)

### `TILE_SIZE` (≡ `BLOCK_N`) — KV tile

`TILE_SIZE` is the number of KV positions one inner iteration of the
softmax loop loads, scores, and reduces against. From the kernel:

```32:43:aiter/aiter/ops/triton/attention/unified_attention.py
    TILE_SIZE = 32 if arch.name == "gfx1201" else 16 if arch.is_rdna else 64
    ...
        TILE_SIZE = min(64, triton.next_power_of_2(block_size))
```

Selection logic:

- **Pure decode** (`max_seqlen_q == 1`): `TILE_SIZE = min(64, next_pow2(block_size))`.
  Tied to the page size so KV loads land on whole pages with no masking.
- **Prefill / mixed**: arch‑dependent default — `64` on CDNA (MI300X),
  `16` on RDNA, `32` on `gfx1201`.

It does **not** depend on `Lk`, GQA, or `BLOCK_M`. The KV‑axis loop runs

$$
  \left\lceil \frac{|\mathcal{P}(i,t)|}{\mathrm{TILE\_SIZE}} \right\rceil
$$

iterations per workgroup tile.

### Worked example — our default decode shape

Default benchmark: `num_seqs = 248`, `H_q = 8`, `H_k = 1` (so $G = 8$),
`hdim = 64`, `block_size = 64`, `max_seqlen_q = 1` (pure decode),
`max_seqlen_k = 4096`.

| Constant | Value | Why |
|---|---:|---|
| $G$ | **8** | $H_q / H_k = 8/1$ |
| `BLOCK_M` | **16** | $G = 8 \le 16$ → default branch |
| `BLOCK_Q` | **2** | $16 / 8$ |
| `TILE_SIZE` | **64** | decode + `min(64, next_pow2(block_size=64)) = 64` |
| `total_num_q_blocks` | **372** | $\lfloor 248 / 2 \rfloor + 248 = 124 + 248$ |

Each workgroup therefore covers a **`BLOCK_Q` × `TILE_SIZE` = 2 × 64**
slab of the `(q‑tokens) × (KV positions)` plane (with each q‑token
replicated 8× into 16 expanded QK rows internally). The KV loop runs
$\lceil 4096 / 64 \rceil = 64$ tiles per workgroup at full context length.

### How these connect to the math we already wrote

Going back to the multi‑head matrix form ($Q_h \in \mathbb{R}^{L_q \times D}$,
$O_h \in \mathbb{R}^{L_q \times D}$):

- The **row axis $L_q$** is tiled with stride `BLOCK_Q` across workgroups
  (one workgroup per `BLOCK_Q` × $G$ q‑rows × heads in the GQA group).
- The **softmax‑reduction axis $L_k$** is tiled with stride `TILE_SIZE`
  *inside* each workgroup (the inner KV loop with online softmax).
- The **head‑dim axis $D$** is **not tiled** — each tile loads the full
  $D$ at once and runs the dot product in registers / a single matmul.
- The **head axis $H_q$** is folded into `BLOCK_M` for the Q rows
  (replication by $G$) and into the kernel grid for the KV head $h_k$.

Math says "one $O_h \in \mathbb{R}^{L_q \times D}$ per query head, $H_q$ of
them". GPU tiling says: "process each $O_h$ in `BLOCK_Q`‑row × `TILE_SIZE`‑col
tiles, with online softmax along the `TILE_SIZE` axis, and bundle GQA
groups together via `BLOCK_M = BLOCK_Q · G`". Same computation, just
chunked.

---

## Triton vs CK‑tile: same `(token, head)` fusion, two abstraction levels

Both kernels solve **exactly the same indexing problem** for the Q tile:

> "Each row $m$ of the `BLOCK_M`‑tall Q tile must correspond to a unique
> *(q‑token, head‑in‑GQA‑group)* pair, with $G$ consecutive rows belonging
> to the same q‑token (one row per query head in that group)."

But Triton expresses it **imperatively** with explicit per‑lane arithmetic
(`m // G`, `m % G`), while CK‑tile expresses it **declaratively** through
tensor‑view transforms (`pad_tensor_view` + `make_merge_transform`). The
result is the same row‑to‑*(token,head)* mapping; the *programming model*
is what differs.

### Triton: explicit `m // G` / `m % G` per lane

The Triton kernel computes the $(token, head)$ pair for each row
directly, in registers:

```132:138:aiter/aiter/ops/triton/_triton_kernels/attention/unified_attention.py
    offs_m = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_SIZE_PADDED)
    offs_t = tl.arange(0, TILE_SIZE)
    query_pos = q_block_local_idx * BLOCK_Q + offs_m // num_queries_per_kv
    query_offset_1 = kv_head_idx * num_queries_per_kv + offs_m % num_queries_per_kv
```

Reading row $m$ ($0 \le m < \mathrm{BLOCK\_M}$):

$$
  \boxed{\;m \;=\; t \cdot G + g\;}
  \qquad
  t \;=\; m \mathbin{//} G,\quad g \;=\; m \bmod G
$$

with $t = $ q‑token‑within‑block, $g = $ head‑in‑group, $G = $ `num_queries_per_kv`.
The kernel then loads `q[query_pos, query_offset_1, :]` from global memory
and the resulting `(BLOCK_M, D)` tile already has rows interleaved by head
inside each token (a "head‑fastest" layout in the merged M axis).

### CK‑tile: declarative merge of two axes

The CK kernel never writes `m // G` or `m % G` anywhere. It instead
*describes* Q to the framework as a 3‑axis tensor `(q‑token, head‑in‑group, head‑dim)`,
pads it, and then **merges** the first two axes into one flat M axis. The
threads then index the merged tile linearly — the merge transform itself
encodes the `(t·G + g)` packing.

```363:389:aiter/3rdparty/composable_kernel/include/ck_tile/ops/unified_attention/kernel/unified_attention_kernel.hpp
        const auto q_dram = [&]() {
            const auto q_dram_base = make_naive_tensor_view<address_space_enum::global>(
                q_ptr,
                make_tuple(cur_batch_query_len, num_queries_per_kv, kHeadDim),
                make_tuple(kargs.query_stride_0, kargs.query_stride_1, 1),
                number<UnifiedAttentionPipeline::kAlignmentQ>{},
                number<1>{});

            const auto q_dram_pad =
                pad_tensor_view( // aling seqlen with kBlockQ and head dim with kHeadDimPadded
                    q_dram_base,
                    // block sizes
                    make_tuple(number<kBlockQ>{}, 1, kHeadDimPadded),
                    sequence<true, false, kPadHeadDimQ>{}); // pads to (seq_len_padded, num_head_q,
                                                            // kHeadDimPadded)

            const auto q_dram_merged = transform_tensor_view(
                q_dram_pad,
                make_tuple(make_merge_transform(make_tuple(query_len_padded, num_queries_per_kv)),
                           make_pass_through_transform(kHeadDimPadded)),
                make_tuple(sequence<0, 1>{}, sequence<2>{}),
                make_tuple(sequence<0>{},
                           sequence<1>{})); // flattens the first two dims, head idx is the fastest
                                            // changing dim in the merged dim

            return q_dram_merged;
        }();
```

Three steps, in order:

**1. Naive view — record the natural 3D shape.**

`make_naive_tensor_view` describes Q as a logical 3D tensor

$$
  \texttt{q\_dram\_base} : \bigl(\,L_q^{(i)},\; G,\; D\,\bigr)
$$

with strides `(query_stride_0, query_stride_1, 1)`. (`query_stride_0`
steps to the next *q‑token*; `query_stride_1` steps to the next *head*
inside the GQA group.) Nothing is read yet — this is just the shape +
stride descriptor.

**2. Pad the q‑token axis up to a multiple of `kBlockQ`.**

`pad_tensor_view(q_dram_base, (kBlockQ, 1, kHeadDimPadded), <true, false, kPadHeadDimQ>)`
produces a view of shape

$$
  \texttt{q\_dram\_pad} : \bigl(\,L_{q,\,\text{padded}}^{(i)},\; G,\; D_\text{padded}\,\bigr),
  \qquad
  L_{q,\,\text{padded}}^{(i)} \;=\; \mathrm{kBlockQ} \cdot
    \left\lceil \tfrac{L_q^{(i)}}{\mathrm{kBlockQ}} \right\rceil
$$

Out‑of‑bounds reads on the padded q‑token axis return 0; out‑of‑bounds
writes are masked. (This is what lets the kernel handle
`cur_batch_query_len` that is not divisible by `kBlockQ` — the same role
that Triton's bounds masks play, just hoisted into the descriptor.)

**3. Merge the first two axes into a single flat M axis.**

`transform_tensor_view` with `make_merge_transform(make_tuple(query_len_padded, num_queries_per_kv))`
produces a 2D view

$$
  \texttt{q\_dram\_merged} : \bigl(\,\underbrace{L_{q,\,\text{padded}}^{(i)} \cdot G}_{M\text{ axis}},\; D_\text{padded}\,\bigr)
$$

The order of the merge tuple matters and is the **whole game**:

- The first argument (`query_len_padded`) becomes the **slow** axis.
- The second argument (`num_queries_per_kv`) becomes the **fast** axis.

So the merged coordinate $m$ decomposes as

$$
  \boxed{\; m \;=\; t \cdot G + g \;} \qquad t \in [0,\, L_{q,\,\text{padded}}^{(i)}),\;\; g \in [0, G)
$$

— exactly the same `m // G`, `m % G` decomposition Triton is computing in
its lanes. The comment in the source even spells it out:

```text
flattens the first two dims, head idx is the fastest changing dim in the merged dim
```

### The tile window — where the M coord finally gets used

After the merge, CK opens a tile window into the merged view at the
correct row offset:

```395:398:aiter/3rdparty/composable_kernel/include/ck_tile/ops/unified_attention/kernel/unified_attention_kernel.hpp
        auto q_dram_window =
            make_tile_window(q_dram,
                             make_tuple(number<kBlockM>{}, number<kHeadDimPadded>{}),
                             {query_pos * num_queries_per_kv, 0});
```

with `query_pos = q_block_local_idx * kBlockQ` (the q‑token index inside
the sequence). The starting M offset is therefore

$$
  m_0 \;=\; \texttt{query\_pos} \cdot G \;=\; (q\_block\_local\_idx \cdot \mathrm{kBlockQ}) \cdot G \;=\; q\_block\_local\_idx \cdot \mathrm{kBlockM}
$$

and the window covers `kBlockM` rows × `kHeadDimPadded` columns of the
merged view — i.e. exactly $\mathrm{kBlockQ}$ q‑tokens × $G$ heads.

### Side‑by‑side: same arithmetic, two surfaces

| Concern | Triton | CK‑tile |
|---|---|---|
| Decomposition of M row | `t = m // G`, `g = m % G` written explicitly | implicit in `make_merge_transform((L_q_padded, G))` |
| Q‑token bounds (`L_q` not multiple of block) | masked at the load (`mask=...`) | absorbed by `pad_tensor_view(..., <true, false, …>)` |
| Head‑dim padding | `HEAD_SIZE_PADDED = next_pow2(HEAD_SIZE)` + mask | `kHeadDimPadded` + `pad_tensor_view(..., <…, …, kPadHeadDimQ>)` |
| Tile address arithmetic | `offs_m = tl.arange(0, BLOCK_M)` + base pointers + strides | `make_tile_window(q_dram, (kBlockM, kHeadDimPadded), {query_pos * G, 0})` |
| Where the math lives | inline in the kernel body | in the tensor descriptor / view chain |
| Programming model | imperative SIMT lane code | declarative tensor‑view transforms |

Both produce a `(BLOCK_M, HEAD_DIM)` Q tile whose row $m$ contains
`q[token = q_block_local_idx * BLOCK_Q + m // G,  head = kv_head_idx * G + m % G, :]`.

> **Mental model:** Triton lets you *see* the `m // G`, `m % G` arithmetic
> because each lane has to compute its own load address. CK‑tile hides
> the same arithmetic inside `make_merge_transform`, because the
> framework's job is to take a shape description + a stride description
> and produce the per‑thread offsets for you. They're isomorphic — and
> they have to be, because they're loading the *same* memory into the
> *same* `(BLOCK_M, D)` Q tile shape.

### Why it's nice that they line up exactly

Because `make_merge_transform((L_q_padded, G))` and Triton's
`offs_m // G` / `offs_m % G` produce **bit‑identical** `(token, head)`
mapping, the QK matmul and the per‑row softmax inside both kernels see
the same logical layout: $G$ consecutive M rows belong to the same
q‑token but different heads, so a single K/V load is reused across the
whole GQA group. That's the architectural trick that makes GQA cheap on
both backends — and it's expressed once per backend, in the language
each backend prefers.
