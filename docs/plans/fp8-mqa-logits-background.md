# Background: Attention, the DeepSeek Lightning Indexer, and the FP8 MQA Logits Kernel

> **Purpose.** This document builds up — from first principles — the background needed to
> work on the FP8 MQA logits kernel. It provides background on attention 
> and walks through: how attention works, the efficiency variants
> that lead to DeepSeek's design, what the "lightning indexer" is and where it fits.
>

### Notation used throughout

| Symbol | Meaning | Typical value (DeepSeek-V3-ish) |
|--------|---------|---------------------------------|
| `L` | sequence length (number of tokens) | 1 – 128k |
| `d_model` | model hidden size | 7168 |
| `H` | number of attention heads | 128 (main); 64 (indexer) |
| `d_head` | per-head dimension | 128 |
| `d_k`, `d_v` | **per-head** key/query and value widths | usually `= d_head` |
| `kv` subscript | a quantity over the KV sequence | — |

---

## Part 1 — Attention fundamentals

### 1.1 The setup: tokens, queries, keys, values

A transformer processes a sequence of **tokens** (sub-word units produced by a tokenizer).
Each token is represented by a vector — an *embedding*. The core operation, **attention**,
lets each token gather information from other tokens in the sequence.

#### Where the data comes from

The input to an attention layer is the sequence of token vectors coming out of the
**previous layer**. (For the very first layer, it is the token embeddings looked up from an
embedding table, plus positional information.) Stack all `L` token vectors row-by-row into
one **input matrix**:

```
X   with shape [L, d_model]
```

- `L` = sequence length (number of tokens),
- `d_model` = the model's hidden size (e.g. 7168 in DeepSeek-V3),
- each **row** of `X` is one token's vector.

#### Query, Key, and Value matrices

We have **Q, K, V matrices with one row per token.**, that is, query, key and value matrices.

Intuition for each:

- **Query (Q)** — "what is this token looking for?"
- **Key (K)** — "what does this token contain / advertise?"
- **Value (V)** — "what will this token contribute if attended to?"

#### How matrices Q, K, V are formed

The model holds three **learned weight matrices** (parameters, fixed after training). A single
projection produces **all `H` heads at once** — each head gets its own slice of width `d_head`,
and the slices are concatenated — so the output width is `H · d_head`:

```
W_Q : [d_model, H · d_k]
W_K : [d_model, H · d_k]      # K and Q must share per-head width d_k so their dot products are defined
W_V : [d_model, H · d_v]
```

Q, K, V are produced by multiplying the input `X` by each weight matrix — these are the
"linear projections" — and then **reshaped** to expose the head axis:

```
Q = X · W_Q   →  [L, H · d_k]   reshape →  [L, H, d_k]
K = X · W_K   →  [L, H · d_k]   reshape →  [L, H, d_k]
V = X · W_V   →  [L, H · d_v]   reshape →  [L, H, d_v]
```

Here `d_k` and `d_v` are the **per-head** widths (usually both `= d_head`); they are written
separately only to show that K and Q must match each other (for the dot product), while V's
width could in principle differ. In the common case `d_k = d_v = d_head = d_model / H`, the
combined output width `H · d_head` is exactly `d_model`, so each `W` is `[d_model, d_model]`.

#### Concrete example

Take `d_model = 4096`, `H = 32` heads (so `d_head = d_model/H = 128`), sequence `L = 1000`.
Note `H · d_head = 32 · 128 = 4096 = d_model`, which is why the combined projection width below
equals `d_model` — a consequence of `d_k = d_head`, not a coincidence to memorize.

| Tensor | Shape | What it is |
|--------|-------|------------|
| `X` (layer input) | `[1000, 4096]` | 1000 token vectors |
| `W_Q`, `W_K`, `W_V` (params) | `[4096, 4096]` = `[d_model, H·d_head]` | learned projections (all heads) |
| `Q`, `K`, `V` (combined) | `[1000, 4096]` reshaped to `[1000, 32, 128]` = `[L, H, d_head]` | per-token, per-head |
| one head's `Q` slice | `[1000, 128]` = `[L, d_head]` | the per-head view (`d_k = 128`) |
| `Q Kᵀ` per head | `[1000, 1000]` | all-pairs scores |

#### The decoding twist (previews §1.4)

During text generation, tokens are produced **one at a time**. At a given step, the *new*
token's input is a single row `X = [1, d_model]`, so its `Q` is just `[1, H, d_head]` — **one
query**. But `K` and `V` are the **accumulated** matrices for *all tokens so far*
(`[L_so_far, H, d_head]`), read back from the **KV cache**. This "one query row, many key
rows" asymmetry is exactly the shape our kernel deals with, and it is why reading the KV
data dominates cost.

> Reference: *Attention Is All You Need* (Vaswani et al., 2017).

### 1.2 Scaled dot-product attention

The similarity between a query and a key is their **dot product**: large when the two
vectors point in a similar direction. For a single query vector `q` and key vector `k`, each
of dimension `d_k`, the raw score is

```
score = q · k = Σ_{i=1..d_k} q[i] · k[i]
```

#### Why the scaling factor

Dot products of `d_k`-dimensional vectors grow with `d_k` (more terms summed), which pushes
softmax into saturated regions where gradients vanish. Dividing by `sqrt(d_k)` keeps the
variance roughly constant. So the scaled score is `(q · k) / sqrt(d_k)`.

#### From scores to weights to output

Do this for one query against **all** `L` keys → a length-`L` vector of scores. Apply
**softmax** to turn scores into non-negative weights that sum to 1, then take the
weighted sum of the value vectors. In matrix form, for all queries at once:

```
Attention(Q, K, V) = softmax( Q Kᵀ / sqrt(d_k) ) · V
```

Shapes: `Q Kᵀ` is `[L, L]` (every token vs. every token), softmax is applied **row-wise**,
and multiplying by `V [L, d_v]` gives an output `[L, d_v]` — one new vector per token.

#### Tiny worked example

One query `q = [1, 0]`, three keys `k₁=[1,0]`, `k₂=[0,1]`, `k₃=[1,1]`, `d_k = 2`
(skip the scale for clarity), values `v₁=[10], v₂=[20], v₃=[30]`:

```
raw scores: q·k₁=1, q·k₂=0, q·k₃=1
softmax([1,0,1]) ≈ [0.422, 0.155, 0.422]
output = 0.422·10 + 0.155·20 + 0.422·30 ≈ 17.9
```

The query "matched" k₁ and k₃ more than k₂, so it pulled mostly from v₁ and v₃.

#### The O(L²) cost and the causal mask

`Q Kᵀ` is an `L × L` matrix — every token against every token. This quadratic `O(L²)` cost
is the central scaling problem that sparse attention (Part 3) attacks: at `L = 128k` that is
~16 **billion** scores *per head per layer*.

In autoregressive (decoder) models a **causal mask** forces each token to attend only to
itself and earlier tokens (you may not look at the future). Concretely, before softmax the
upper triangle of `Q Kᵀ` is set to `-inf`, so those weights become 0:

```
       k0   k1   k2   k3
q0   [  s  -inf -inf -inf ]
q1   [  s    s  -inf -inf ]
q2   [  s    s    s  -inf ]
q3   [  s    s    s    s  ]
```

(Our kernel uses the same idea, but the allowed region is a per-row **window**
`[cu_starts[m], cu_ends[m])` rather than a simple triangle. A plain causal mask is just
the special case `start = 0, end = m+1`; the window generalizes this so it can *also*
express **context-parallel** KV sharding — see §4.3.)

### 1.3 Multi-head attention (MHA)

Instead of one attention, the model runs `H` in parallel — **heads** — each with its own
slice of width `d_head = d_model / H`. In practice the big `[L, d_model]` Q/K/V are simply
reshaped to `[L, H, d_head]`; head `h` does an independent `[L, L]` attention on its slice.

After all heads finish, their `[L, d_head]` outputs are **concatenated** back to
`[L, d_model]` and passed through one more learned matrix `W_O : [d_model, d_model]`.

#### Why multiple heads help

A single head produces **one** softmax distribution per query — i.e. one weighted average over
the values, driven by a single `q·k` similarity. That is a bottleneck: if a token needs to
gather information about *both*, say, the subject of its sentence *and* a related word far back,
one softmax must blend those into a single set of weights, so sharpening focus on one
relationship dilutes the other.

With `H` heads, each has its **own** `W_Q, W_K, W_V` slices and therefore computes similarity in
a **different projected subspace**. Each head gets an independent `q·k`, an independent softmax,
and writes a different slice of the output. So instead of one weighted average you get `H` of
them, concatenated and mixed by `W_O`. The result is **specialization**: probing trained
transformers reveals heads that do distinct jobs — one tracking the **previous token**
(positional), one linking **pronouns to antecedents** (coreference), one attending to a word's
**syntactic head**, "**induction heads**" that find an earlier occurrence of the current token
and predict what came next (a mechanism behind in-context learning). No single softmax could
express all of these at once.

Crucially this is **almost free**: heads *split* `d_model` (`d_head = d_model / H`) rather than
each using the full width, so `H` heads cost roughly the same parameters and FLOPs as one
full-width head — you partition the same budget into `H` smaller subspaces, and they run in
parallel. The trade-off is that more heads make each head **narrower** (smaller `d_head`), so
there is a sweet spot; very large `H` with tiny `d_head` stops helping.

> This framing also sets up Part 2: the specialization benefit lives on the **query** side
> (many query heads), while the *cost* of distinct K/V per head lives in the **KV cache**.
> MQA/GQA keep the many query heads but **share** K/V across them — which is exactly the
> "many query heads, one shared key" shape of the MQA-logits kernel.

Example (`d_model=4096, H=32, d_head=128, L=1000`): you effectively compute **32 separate**
`[1000, 1000]` score matrices, one per head.

### 1.4 The KV cache and why decoding is bandwidth-bound

During generation, tokens are produced one at a time. Re-computing K and V for *all*
previous tokens at every step would be wasteful, so they are stored in a **KV cache**. Each
new step:

1. computes the new token's Q, K, V (one row each),
2. **appends** the new K, V to the cache,
3. attends the new Q against the **entire** cached K/V.

#### Why this is memory-bound, with numbers

At step `t` you do a `[1, d_head]·[d_head, t]` score computation per head — that is a small
amount of arithmetic, but it must **read the whole cache** (`t` key vectors and `t` value
vectors per head) from memory. As `t` grows, the step is dominated by **reading the KV cache
out of HBM**, not by the math.

Cache size example: `H=128` heads, `d_head=128`, fp16 (2 bytes), `L=128k` tokens, K and V:

```
2 (K,V) × 128 heads × 128 dim × 2 bytes × 131072 tokens  ≈  8.6 GB   per layer
```

Multiplied across dozens of layers, the KV cache becomes huge and its bandwidth dominates.
This single fact motivates almost everything in Part 2 (shrink the cache) and Part 3
(read less of it).

---

## Part 2 — Making attention cheaper

### 2.1 Multi-Query Attention (MQA) and Grouped-Query Attention (GQA)

In standard MHA every head has its **own** K and V, so the KV cache scales with `H`. Two
ideas shrink it:

- **MQA**: all `H` query heads **share a single** K/V head. The KV cache (and the bandwidth
  to read it) drops by a factor of `H`. Trade-off: some quality loss.
  *Reference: Shazeer, 2019, "Fast Transformer Decoding: One Write-Head is All You Need."*
- **GQA**: a middle ground — query heads are split into `G` groups, each group sharing one
  K/V head. `G = 1` is MQA; `G = H` is full MHA. This is the de-facto standard in modern
  LLMs. *Reference: Ainslie et al., 2023.*

#### KV-cache size comparison

Reusing the §1.4 example (`H=128, d_head=128, fp16, L=128k`, per layer):

| Scheme | KV heads | KV cache / layer |
|--------|----------|------------------|
| MHA    | 128      | ~8.6 GB |
| GQA G=8| 8        | ~0.54 GB |
| MQA    | 1        | ~0.067 GB |

> **Why this matters for our kernel.** Our kernel is named **"MQA logits."** The lightning
> indexer has *many query heads* but a *single shared key stream* — exactly the MQA shape:
> one `K[n, :]` per KV position, scored against all the indexer's query heads `Q[m, h, :]`.
> That is the "multi-query" in the name (see Part 4).

### 2.2 FlashAttention — IO-aware kernels

Naively, attention materializes the full `[L, L]` score matrix in slow HBM, writes it,
reads it back for softmax, reads it again for the `·V` step — enormous memory traffic.
**FlashAttention** restructures this into **tiles** that fit in fast on-chip SRAM and
computes softmax **online** (incrementally, carrying a running max and sum), so the big
`[L, L]` matrix is **never written to HBM**. It is the *exact* same result, just memory-aware.
FlashAttention-2 improves how the work is split across GPU threads/blocks (~2× faster).

#### The tiling idea (sketch)

```
for each tile of queries Qi  (kept in SRAM):
    init running max m = -inf, running sum l = 0, accumulator o = 0
    for each tile of keys/values (Kj, Vj) streamed from HBM:
        S = Qi · Kjᵀ                 # small tile, in SRAM
        update m, l with rowmax/expsum of S      # online softmax
        o += softmax_partial(S) · Vj
    write o once
```

> *References: Dao et al., 2022 (FlashAttention); Dao, 2023 (FlashAttention-2).*

The relevance here is **structural**: our kernel follows the same playbook — tile the KV
dimension (`BLOCK_KV`), keep partial results in registers/LDS, and stream the FP8 keys from
memory once. We are writing a small, specialized attention-style kernel, so the same
IO-awareness principles apply (minimize HBM traffic, maximize on-chip reuse).

### 2.3 DeepSeek MLA — compressing the KV cache

DeepSeek-V2 introduced **Multi-head Latent Attention (MLA)**: instead of caching full K and V
per head, it caches a small **low-rank latent** vector per token, and reconstructs the
per-head K/V on the fly via learned up-projection matrices (a decoupled RoPE component
carries position). So the cache stores, say, a `[L, 512]` latent instead of a
`[L, H·d_head]` = `[L, 16384]` K/V — a large reduction — while keeping quality close to MHA.
DeepSeek-V3 scales this up (671B-param Mixture-of-Experts, trained in FP8).

> *References: DeepSeek-V2 (2024); DeepSeek-V3 Technical Report (2024/2025).*

MLA is the substrate on which DeepSeek's **sparse** attention (Part 3) is built.

---

## Part 3 — Sparse attention and the lightning indexer

### 3.1 The motivation

Even with MLA/GQA shrinking the cache, attention still **scans all previous tokens** —
`O(L²)` work over a sequence. For long contexts this dominates. **Sparse attention** observes
that for any given query, only a small subset of past tokens actually matters. If we could
**cheaply** identify the important tokens, we could attend only to those, turning `O(L²)`
into roughly `O(L · k)` for a budget of `k` selected tokens.

Numeric intuition: at `L = 128k` with a budget `k = 2048`, full attention scans 128k tokens
per query; sparse attention scans 2048 — about **64× fewer**.

The hard part is the "*cheaply identify*" step: a selector that is itself `O(L²)` and
expensive defeats the purpose. It must be **much** cheaper than the full attention it
replaces — which is exactly why the indexer runs in FP8 with few heads (Part 4).

DeepSeek's earlier **Native Sparse Attention (NSA)** explored hardware-aligned, natively
trainable sparsity with three branches (coarse token compression, fine-grained token
selection, and a sliding window). *Reference: Yuan et al., 2025 (NSA).*

### 3.2 DeepSeek Sparse Attention (DSA) and the lightning indexer

DeepSeek-V3.2(-Exp) introduces **DeepSeek Sparse Attention (DSA)**, built on top of MLA. The
*only* architectural change versus the previous model is DSA, and its cheap selector is the
**lightning indexer**.

The pipeline has three stages:

**Stage 1 — Indexer computes scores.** A lightweight, content-based pre-filter. For a query
token `t` and an earlier token `s`, it computes an **index score**:

```
I[t, s] = Σ_{j=1..H_I}   w[t, j] · ReLU( q_index[t, j] · k_index[s] )
          └────────────────────────────────────────────────────────┘
                 sum over the indexer's H_I heads
```

Reading it piece by piece:

- `q_index[t, j]` — the indexer's query for token `t`, head `j` (separate, small projection),
- `k_index[s]` — the indexer's **single shared** key for token `s` (MQA: one per position),
- `q_index[t,j] · k_index[s]` — a dot product (similarity),
- `ReLU(·)` — clamp negatives to 0 (the indexer's nonlinearity),
- `w[t, j]` — a per-(query, head) weight, also produced from token `t`,
- `Σ_j` — sum the weighted, ReLU'd scores over the indexer heads → one scalar per `(t, s)`.

The indexer uses **few heads** and **can run in FP8**, so it is cheap relative to the main
attention it gates.

**Stage 2 — Top-k token selection.** For each query, keep only the `k` highest-scoring tokens
(`k = 2048` in the released model); discard the rest. This produces the sparse mask.

**Stage 3 — Sparse attention over selected tokens.** The main MLA attention attends only
over the selected key/value entries:

```
u[t] = Attention( h[t], { c[s] : I[t, s] ∈ Top-k(I[t, :]) } )
```

This cuts attention cost substantially for long contexts (DeepSeek reports ~6–7× long-context
inference savings) at near-parity quality. The indexer is **trained** to mimic the main
model's attention distribution (a dense KL-divergence warm-up, then a sparse stage), and its
input is detached from the main graph so the two are optimized by separate losses.

> *References: DeepSeek-V3.2 (arXiv:2512.02556) and the DeepSeek-V3.2-Exp tech report/repo.
> The `I[t,s]` formula above is the paper's definition.*

### 3.3 Where our kernel sits

**Our kernel computes Stage 1 only** — the index scores `I[t, s]` (the "logits"). It does
**not** perform the top-k or the main attention; it produces the fp32 score matrix that a
downstream top-k consumes. Everything in the ticket — the FP8 Q·K dot, the per-K dequant
scale, the ReLU, the per-head weights, and the head reduction — is precisely this `I[t, s]`
formula realized as a GPU kernel.

```
[indexer Q,K in FP8] → OUR KERNEL: logits I[t,s] → top-k select → sparse MLA attention
                       └─ Stage 1 ─┘                └─ Stage 2 ─┘   └──── Stage 3 ────┘
```

---

## Part 4 — The FP8 MQA logits kernel in detail

### 4.1 What it computes (formula ↔ code ↔ shapes)

For each query row `m` and KV position `n` inside that row's window:

```
logits[m, n] = Σ_h  ReLU( <Q[m, h, :], K[n, :]> · kv_scale[n] ) · weights[m, h]
```

Mapping to the indexer math in §3.2: `m ≡ t` (query token), `n ≡ s` (key token), `h` ≡
indexer head, `weights[m,h] ≡ w[t,j]`, and `kv_scale[n]` is the FP8 dequantization scale for
key `n`. The output `logits` **is** `I[t, s]`.

Tensor shapes (DeepSeek dims: `NUM_HEADS = 64`, `HEAD_SIZE D ∈ {64, 128}`):

| Tensor      | Shape                          | dtype | Meaning |
|-------------|--------------------------------|-------|---------|
| `Q`         | `[seq_len, NUM_HEADS, D]`      | fp8   | indexer queries |
| `KV`        | `[seq_len_kv, D]`              | fp8   | shared indexer keys (MQA: one per position) |
| `kv_scales` | `[seq_len_kv]`                 | f32   | per-key dequant scale |
| `weights`   | `[seq_len, NUM_HEADS]`         | f32   | per-(query, head) weights |
| `cu_starts` / `cu_ends` | `[seq_len]`        | i32   | per-row attention window `[start, end)` |
| `logits`    | `[seq_len, seq_len_kv]`        | f32   | output scores (`-inf` outside window) |

Example sizes: `seq_len = 1024`, `seq_len_kv = 1024`, `NUM_HEADS = 64`, `D = 128`. Then `Q`
is `1024×64×128` fp8 (~8 MB), `KV` is `1024×128` fp8 (~128 KB), output `logits` is
`1024×1024` fp32 (~4 MB).

### 4.2 Why "MQA logits"

There is **one key vector per KV position** (`KV[n, :]`), scored against **all `NUM_HEADS`
query heads** (`Q[m, h, :]`). One shared K, many Q heads → multi-query. The result is a
matrix of **logits** (pre-top-k scores), hence "MQA logits."

### 4.3 Step-by-step semantics

For one query row `m`, over its window `[start, end) = [cu_starts[m], cu_ends[m])`:

1. **Dot product (FP8):** `s[h, n] = <Q[m,h,:], K[n,:]>` — an FP8 matrix multiply producing
   fp32 partial sums. Shape `[NUM_HEADS, BLOCK_KV]` per KV tile.
2. **Dequantize:** `s[h, n] *= kv_scale[n]` — the keys were quantized to FP8 with a per-key
   scale; this restores magnitude. Broadcast across heads (same scale for every `h`).
3. **ReLU:** `s = max(s, 0)` — the indexer's nonlinearity.
4. **Per-head weight:** `s[h, n] *= weights[m, h]` — broadcast across KV positions (same
   weight for every `n`).
5. **Head reduction:** `logits[m, n] = Σ_h s[h, n]` — sum over the head axis to one score
   per KV position. (This cross-head reduction is the trickiest part on the GPU; see §4.5.)
6. **Masking:** positions outside `[start, end)` are excluded so the later top-k ignores
   them. There are two policies, controlled by `clean_logits`: when **`True`**, the kernel
   explicitly writes `-inf` to every out-of-window position (the output is self-contained);
   when **`False`**, the kernel writes *only* in-window positions and leaves whatever was
   already in the buffer elsewhere, making the caller responsible for pre-filling `-inf`.
   The `True` path is the default and costs an `-inf` prefill of the whole output.

A small numeric walk-through for one `(m, n)` with `NUM_HEADS = 2`:

```
dots          = [<Q0,K>, <Q1,K>] = [ 3.0, -1.0 ]
× kv_scale(=2): [ 6.0, -2.0 ]
ReLU          : [ 6.0,  0.0 ]
× weights([0.5,0.7]): [ 3.0, 0.0 ]
Σ_h           : 3.0     →  logits[m,n] = 3.0
```

### 4.3.1 What the per-row window actually represents

The window `[cu_starts[m], cu_ends[m])` is more general than a causal triangle, and the test
data (`test_fp8_mqa_logits.py`) shows it carries **two** distinct meanings:

**(a) Causal masking — the common case.** With no context parallelism, the test sets

```
cu_starts[m] = 0
cu_ends[m]   = m + (seq_len_kv - seq_len)
```

Each query `m` may see every key from position 0 up to its own (offset) position. When
`seq_len_kv == seq_len` this is exactly the standard causal mask `end = m + 1` from §1.2. The
offset `seq_len_kv - seq_len` handles the decoding case where the KV cache already holds some
prefix tokens that *every* query is allowed to see.

**(b) Context-parallel (CP) KV sharding — why it is a `[start, end)` *window*, not just an
`end`.** In context parallelism the KV sequence is split across several devices ("CP ranks"),
and each rank computes logits only for the slice of keys **it owns**. That slice is an interior
band `[start, end)` of the full KV axis — hence a nonzero `start`. The test's
`generate_cp_test_data` builds exactly this: it chops the queries into two chunks and assigns
each chunk a KV band, deliberately pairing an *early* band with a *late* band
(`cp_id` vs `cp_size*2-1-cp_id`). That pairing is a **load-balancing** trick: under causal
masking, low-index queries have little work and high-index queries have a lot, so giving each
CP rank one light chunk and one heavy chunk equalizes work across ranks.

So the single `[start, end)` abstraction simultaneously expresses *causal masking* and
*which contiguous KV shard this rank is responsible for* — which is why the kernel takes
arbitrary per-row start/end arrays rather than assuming a triangle.

> This also explains the kernel's reverse row iteration (`row = num_programs - block_idx - 1`):
> processing high-index (heavy) rows first and low-index (light) rows last smooths the
> **tail effect** — the straggler imbalance when the last few CTAs on a CU finish at very
> different times.

### 4.4 FP8 number formats (and why gfx942 ≠ gfx950)

FP8 packs a float into 8 bits: 1 sign bit + exponent bits + mantissa bits. Two encodings are
standard:

- **E4M3** — 4 exponent, 3 mantissa bits: **more precision, less range** (max ≈ 448). Used
  for weights/activations — and here, for Q and K.
- **E5M2** — 5 exponent, 2 mantissa bits: **more range, less precision** (max ≈ 57344). Used
  for gradients.

Because FP8 has so few mantissa bits, values are typically stored as `fp8_value × scale`,
where `scale` is a higher-precision number per-tensor, per-row, or per-block. In our kernel
that is exactly `kv_scale[n]` — one scale per key, applied at step 2 above.

> *Reference: Micikevicius et al., 2022, "FP8 Formats for Deep Learning."*

#### FN vs FNUZ — the arch-specific gotcha

There are two **variants** of these encodings that matter on AMD hardware:

- **OCP "FN"** (e.g. `E4M3FN`) — the Open Compute Project standard; has signed zero and
  signed NaN.
- **"FNUZ"** (e.g. `E4M3FNUZ`) — "unsigned zero", a single NaN; uses a **different exponent
  bias** to gain one extra exponent value.

**These are not bit-compatible** — the same 8 bits mean roughly a factor-of-2 different value
between FN and FNUZ. Critically for us:

- **gfx942 (MI300X / CDNA3) uses the FNUZ variants.**
- **gfx950 (MI355 / CDNA4) uses the OCP (FN) variants.**

So our gfx942 kernel must select **`E4M3FNUZ`** (this is what `get_fp8_dtypes()` /
`get_hip_arch()` handle in the existing code). Data shared across the two arches needs
explicit conversion, not a raw reinterpret.

> *References: OCP Microscaling (MX) v1.0 spec; ROCm HIP FP8 docs; ROCm "Matrix Cores on
> CDNA3/CDNA4" blog.*

#### Why there is a `kv_scale` but no `q_scale`

A natural question: the formula dequantizes K with `kv_scale[n]`, but Q — also FP8 — has **no
scale** anywhere. (In the test, Q is cast straight to FP8 with `q.to(e4m3)`, no scale at all.)
Two facts explain this, and both come from where the operations sit relative to the ReLU:

1. **`kv_scale` is positive, so its placement is free.** Because `kv_scale[n] > 0` and ReLU is
   *positively homogeneous* (`ReLU(a·x) = a·ReLU(x)` for `a ≥ 0`), it is mathematically
   identical whether we scale before or after the ReLU:
   `ReLU(s · kv_scale) = kv_scale · ReLU(s)`. The kernel applies it *inside* (step 2, before
   ReLU) purely for convenience; the result is the same.

2. **A per-head Q scale would be redundant with `weights`.** Suppose Q for head `h` were stored
   as `q_fp8 · q_scale[h]`. By the same homogeneity, that per-head factor pulls straight through
   the ReLU and merges into the post-ReLU multiply by `weights[m, h]`:
   `weights[m,h] · ReLU(q_scale[h] · s) = (weights[m,h]·q_scale[h]) · ReLU(s)`. In other words
   the **per-head weight already absorbs any per-head Q magnitude**, so a separate `q_scale`
   stream would be wasted work. The kernel therefore needs only *one* scale stream (on K), and
   `weights` does double duty as the Q-side scale.

This is the conceptual reason the kernel carries exactly the scales it does — it is not an
arbitrary choice but a direct consequence of ReLU's positive homogeneity.

### 4.5 Matrix cores (MFMA) on CDNA3

AMD GPUs have **Matrix Cores** that execute **MFMA** (Matrix Fused Multiply-Add) instructions
— hardware that multiplies small fixed-size tiles in a single op, with many lanes cooperating.
CDNA3 (gfx942) has a **native FP8** instruction:

```
v_mfma_f32_16x16x32_fp8_fp8
```

The shape `16x16x32` means: it multiplies a `16×32` FP8 tile (A) by a `32×16` FP8 tile (B)
and **accumulates into a `16×16` FP32** tile (C). The `32` is the contraction (K) dimension
per instruction; FP32 accumulation is what gives acceptable accuracy from FP8 inputs.

#### Tiling our dot product onto MFMA

We need `s[H, BLOCK_KV] = Q[H, D] · K[D, BLOCK_KV]`. With `H = 64`, `BLOCK_KV = 128`,
`D = 128`:

- Split the output `H` axis into `H/16 = 4` row-tiles,
- split `BLOCK_KV` into `128/16 = 8` col-tiles,
- split the contraction `D` into `D/32 = 4` K-steps.

So one KV tile needs `4 × 8 × 4 = 128` MFMA instructions, with the four K-steps accumulating
into the same FP32 tile. These are distributed across the block's waves.

> The `16x16x32` instruction above is one choice. The Triton launcher actually selects the
> MFMA *non-K* tile size by problem size: `matrix_instr_nonkdim = 16` for `seq_len ≤ 1024`
> and `32` otherwise (the larger `32x32` MFMA amortizes issue overhead better once there is
> enough work to fill it). The FlyDSL port can start with the `16` variant and revisit.

> *References: AMD Matrix Instruction Calculator; AMD Instinct MI300 CDNA3 ISA Reference
> Guide; ROCm Matrix Cores blog.*

#### Why it is not MMA-bound

The matrix math here is tiny (`H` is only 64, `D` only 128). The real costs are **streaming
the FP8 keys from memory** and the **head reduction**. The reduction is the awkward part, and
it is worth being precise about *why*: the MFMA output tile is `[NUM_HEADS, BLOCK_KV]`, and we
must sum over the **head axis** — in the reference Triton kernel this is literally
`tl.sum(scores, axis=0)`, a reduction along the tile's **M (row) dimension**. But MFMA spreads
that M dimension *across the lanes and waves* of the block (each lane holds only a few rows of
the output), so "sum over heads" is not a free within-register reduction: it requires
**cross-lane** traffic (a warp butterfly via `ds_swizzle`/shuffle) and, when heads span more
than one wave, a **cross-wave** pass through LDS. Reducing along the *N* (column) dimension
would have been cheaper, but the head axis is the M dimension, so we pay for it.

The consequence: the kernel is **bandwidth/issue-bound**, and the optimization levers are
KV-load width and an efficient cross-head reduce — not raw MFMA throughput. This is why the
plan emphasizes pinning the MFMA fragment→lane layout and using wide FP8 loads.

### 4.6 Relationship to DeepGEMM, and how the kernel is validated

DeepSeek's **DeepGEMM** library (FP8 GEMM with fine-grained scaling) added "scoring kernels
(weighted ReLU MQA logits) for the lightning indexer" for DeepSeek-V3.2. The aiter tests for
this kernel are **adapted from DeepGEMM's attention tests**, which is why the reference torch
implementation (`ref_fp8_mqa_logits`) and the `calc_diff` similarity metric match DeepGEMM's
conventions.

#### The reference is "fake-quant," not an FP8 matmul

A subtlety that explains the tolerance: the reference does **not** run an FP8 matmul. The test
first quantizes K to FP8 and immediately **dequantizes back to bf16** (`kv_fp8 → kv` round-trip),
then `ref_fp8_mqa_logits` computes the whole dot/ReLU/weight/reduce in fp32. So the reference
models the *rounding error* introduced by storing K in FP8, but does its arithmetic in high
precision. Our kernel, by contrast, feeds FP8 straight into the MFMA. We are therefore
checking that "real FP8 matmul" stays close to "fp32 math on FP8-rounded inputs" — a numerical
match, never bit-exactness. (Q is cast directly to FP8 with no scale, consistent with §4.4.)

#### What `calc_diff` measures

The metric is a normalized similarity, **not** a max-abs or relative error:

```
calc_diff(x, y) = 1 − 2·⟨x, y⟩ / (‖x‖² + ‖y‖²)
```

This is `1 − cosine-like-similarity`: it is `0` exactly when `x == y`, grows toward `1` as they
diverge, and is dominated by the large-magnitude entries (so a few tiny logits being slightly
off barely moves it). That makes it the right tool for FP8 — it tolerates per-element rounding
while still catching systematic errors. The acceptance bar is **`calc_diff < 1e-3`**.

The test also asserts the **`-inf` masks match exactly** (`torch.equal` on the `== -inf`
masks) before comparing values — so masking/windowing must be bit-correct even though the
in-window values only need to match approximately.

> *Reference: DeepGEMM (github.com/deepseek-ai/DeepGEMM).*

---

## Part 5 — How this maps to our gfx942 work

- Existing implementations live in **aiter**: a Triton dense kernel (gfx942 primary), a Gluon
  dense kernel (gfx950-only, CDNA4 intrinsics), and Gluon/Triton paged paths.
- **Our task:** add a **FlyDSL** dense kernel for gfx942, because there is no FlyDSL
  MQA-logits kernel today and the math is gfx942-friendly (native FP8 MFMA, no scaled-MFMA
  requirement, tiny LDS use). It runs alongside Triton for A/B comparison.
- The reference Triton kernel to mirror is
  `aiter/ops/triton/_triton_kernels/attention/fp8_mqa_logits.py`; the launcher behavior to
  mirror is `aiter/ops/triton/attention/fp8_mqa_logits.py`.
- See the implementation plan: [`fp8-mqa-logits-flydsl-gfx942.md`](./fp8-mqa-logits-flydsl-gfx942.md).

---

## References (verified)

**Attention foundations**
- Vaswani et al., [*Attention Is All You Need*](https://arxiv.org/abs/1706.03762), 2017
- Shazeer, [*Fast Transformer Decoding: One Write-Head is All You Need*](https://arxiv.org/abs/1911.02150) (MQA), 2019
- Ainslie et al., [*GQA: Training Generalized Multi-Query Transformer Models...*](https://arxiv.org/abs/2305.13245), 2023

**Efficient attention kernels**
- Dao et al., [*FlashAttention*](https://arxiv.org/abs/2205.14135), 2022
- Dao, [*FlashAttention-2*](https://arxiv.org/abs/2307.08691), 2023

**DeepSeek model line**
- DeepSeek-AI, [*DeepSeek-V2*](https://arxiv.org/abs/2405.04434) (MLA), 2024
- DeepSeek-AI, [*DeepSeek-V3 Technical Report*](https://arxiv.org/abs/2412.19437), 2024/2025
- Yuan et al., [*Native Sparse Attention (NSA)*](https://arxiv.org/abs/2502.11089), 2025
- DeepSeek-AI, [*DeepSeek-V3.2*](https://arxiv.org/abs/2512.02556), 2025 ([HTML version](https://arxiv.org/html/2512.02556v1))
- DeepSeek-AI, [*DeepSeek-V3.2-Exp repository & tech report*](https://github.com/deepseek-ai/DeepSeek-V3.2-Exp) ([report PDF](https://github.com/deepseek-ai/DeepSeek-V3.2-Exp/blob/main/DeepSeek_V3_2.pdf))

**FP8 formats & AMD matrix cores**
- Micikevicius et al., [*FP8 Formats for Deep Learning*](https://arxiv.org/abs/2209.05433), 2022
- OCP, [*Microscaling Formats (MX) v1.0* spec](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf), 2023
- ROCm, [*HIP FP8 Numbers* docs](https://rocm.docs.amd.com/projects/HIP/en/latest/reference/fp8_numbers.html)
- ROCm, [*Matrix Cores on CDNA3 and CDNA4* blog](https://rocm.blogs.amd.com/software-tools-optimization/matrix-cores-cdna/README.html)
- AMD, [*Matrix Instruction Calculator*](https://github.com/ROCm/amd_matrix_instruction_calculator)
- AMD, [*Instinct MI300 CDNA3 ISA Reference Guide*](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-mi300-cdna3-instruction-set-architecture.pdf) (PDF; open in browser)

**FP8 GEMM / indexer kernels**
- DeepSeek-AI, [*DeepGEMM*](https://github.com/deepseek-ai/DeepGEMM)

---

## Glossary

- **Token** — a sub-word unit; the atomic input to a transformer.
- **Embedding** — the vector representation of a token.
- **d_model** — the model's hidden size (width of each token vector).
- **Q / K / V** — query, key, value matrices; `X·W_Q`, `X·W_K`, `X·W_V`.
- **Attention score / logit** — pre-softmax (or, here, pre-top-k) similarity value.
- **Softmax** — turns a score vector into non-negative weights summing to 1.
- **Head** — one of several parallel attention subspaces of width `d_head`.
- **Causal mask** — restricts each token to attend only to itself and earlier tokens.
- **KV cache** — stored keys/values from past tokens, reused during decoding.
- **MHA / MQA / GQA** — multi-head / multi-query / grouped-query attention (KV-sharing variants).
- **MLA** — Multi-head Latent Attention; low-rank KV-cache compression (DeepSeek-V2).
- **DSA** — DeepSeek Sparse Attention; selects top-k tokens via the lightning indexer.
- **Lightning indexer** — cheap content-based scorer producing index logits `I[t,s]`.
- **Top-k** — keep the `k` highest-scoring tokens per query; the sparsity mechanism.
- **FP8 (E4M3 / E5M2)** — 8-bit float formats; E4M3 = more precision, E5M2 = more range.
- **FN vs FNUZ** — OCP vs "unsigned-zero" FP8 variants; gfx950 uses FN, gfx942 uses FNUZ.
- **Dequantization scale** — higher-precision multiplier restoring an FP8 value's magnitude.
- **MFMA** — AMD Matrix Fused Multiply-Add (matrix-core) instruction.
- **CDNA3 / CDNA4** — AMD GPU architectures: gfx942 (MI300X) / gfx950 (MI355).
- **LDS** — Local Data Share; AMD on-chip shared memory (the "SRAM" FlashAttention uses).
- **HBM** — High-Bandwidth Memory; the GPU's large but slower DRAM.
- **BLOCK_KV** — the KV-tile width processed per inner-loop iteration (128 here).
- **Context parallelism (CP)** — splitting one sequence's KV across devices; each CP rank
  computes logits only for its KV band `[start, end)`. The per-row window encodes this.
- **Tail effect** — straggler imbalance when a CU's last CTAs finish at very different times;
  mitigated here by reverse row iteration (heavy rows first).
- **Positive homogeneity** — `ReLU(a·x) = a·ReLU(x)` for `a ≥ 0`; why a positive scale can
  move across the ReLU and why `weights` can absorb a per-head Q scale.
- **clean_logits** — flag: if `True`, the kernel writes `-inf` to all out-of-window positions;
  if `False`, it writes only in-window positions and the caller must pre-fill `-inf`.
- **calc_diff** — the test's normalized similarity error, `1 − 2⟨x,y⟩/(‖x‖²+‖y‖²)`; bar `< 1e-3`.
- **Fake-quant** — quantize→dequantize to a higher precision to model rounding without doing
  the low-precision matmul; how the torch reference treats FP8.
