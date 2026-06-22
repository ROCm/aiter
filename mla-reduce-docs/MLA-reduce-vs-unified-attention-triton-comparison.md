# Comparing two kernels: HIP MLA reduce vs Triton unified attention

Audience: junior engineer. Goal: understand how the HIP reduce kernel relates to
the Triton unified-attention kernels — what each does, why a "reduce" step exists
at all, and where they line up.

- **Kernel A** — `kn_mla_reduce_v1` (HIP), `csrc/kernels/mla/reduce.cu`
- **Kernel B** — the Triton **unified attention** file,
`aiter/ops/triton/_triton_kernels/attention/unified_attention.py`,
which actually contains **three** kernels (see §3).

---

## 1. The big picture in one paragraph

Attention turns a query into a weighted average of value vectors. You can compute
that **in one pass** (one worker walks all the keys) or you can **split the work**
(many workers each handle a slice of the keys, then a final worker merges their
results). The HIP kernel A is **only that final merge step**. The Triton file B
contains **both strategies** — a one-pass kernel, and a split pair (a stage-1
that produces partials + a stage-2 that merges them). So the closest match to
Kernel A inside file B is B's own merge kernel, called `reduce_segments`.

Analogy — averaging a giant list of numbers:

- **One pass:** one person reads the whole list, keeps a running average. Done.
- **Split:** give 8 people 1/8 each; each returns a partial average + how much
weight they saw; then **one person merges the 8 partials** into the true average.
**Kernel A is that merger. `reduce_segments` in file B is the same merger.**

---

## 2. The math both kernels share: online softmax

For each query, attention computes:

```
scores  = scale · (Q · Kᵀ)      # match query against keys
weights = softmax(scores)        # normalize to probabilities (sum = 1)
output  = weights · V            # weighted average of value vectors
```

The tricky part is `softmax`: to normalize you need the **max** and the **sum** of
exp(scores) over *all* keys. If you process keys in chunks you can't finish until
you've seen them all. The fix is **online softmax**: keep a running max `M` and
running sum `L`, and whenever a new chunk raises the max, **rescale** what you
already have:

```
new_max = max(old_max, chunk_max)
alpha   = exp(old_max − new_max)     # shrink factor for existing accumulator
acc     = acc · alpha + (this chunk's contribution)
L       = L   · alpha + (this chunk's sum)
```

This same rescale shows up in **every** kernel below — only the thing being
chunked changes (key-tiles in one-pass, or whole segments in the merge).

A speed note you'll see in the code: both use `exp2` (base-2 exponent) instead of
`exp`, because GPUs do `exp2` faster. They pre-multiply the scale by
`RCP_LN2 = 1/ln(2) = 1.4426…` so that `exp2(x·RCP_LN2) == exp(x)`. Same result,
cheaper instruction.

---

## 3. File B has three kernels (two strategies)

`unified_attention.py`:


| kernel                        | line | role                                                                             |
| ----------------------------- | ---- | -------------------------------------------------------------------------------- |
| `kernel_unified_attention_2d` | 55   | **one-pass** attention — whole thing in a single launch                          |
| `kernel_unified_attention_3d` | 461  | **split stage-1** — each program does ONE *segment* of the keys, writes partials |
| `reduce_segments`             | 848  | **split stage-2** — merges the segment partials into the final output            |


Triton calls the splits **"segments"** (`NUM_SEGMENTS_PER_SEQ`); the HIP world
calls them **"splits"**. Same concept: a chunk of the KV sequence handled by its
own worker.

### 3a. `kernel_unified_attention_2d` — the one-pass path

One program owns one query-block. It loops over key-tiles, and inside that loop it
does QK, the online-softmax rescale, and PV — all of it. Because one program walks
**all** the keys for its query, it finishes the softmax itself and writes the final
output directly. **No partials, no merge needed.**

### 3b. `kernel_unified_attention_3d` — the split stage-1 (producer)

Almost the same loop, but the launch grid gains a **third dimension**: `segm_idx`
(line 519). Each program only walks the key-tiles in **its** segment
(line 664: `for j in range(segm_idx·tiles_per_segment, …)`). It can't finish the
softmax because it only saw part of the keys, so instead of a final output it
writes **partials** for its segment:

- `segm_output_ptr` — the segment's unfinished weighted output (line 834)
- `segm_max_ptr` — the segment's running max `M` (line 844)
- `segm_expsum_ptr` — the segment's running sum `L` (line 845)

### 3c. `reduce_segments` — the split stage-2 (merger) ← matches Kernel A

This reads every segment's `(output, max, expsum)` and merges them (lines 901-931):

```python
overall_max    = tl.max(segm_max)                              # max across segments
segm_expsum   *= exp2(segm_max - overall_max)                  # rescale each sum
overall_expsum = tl.sum(segm_expsum)                           # total denominator
segm_output   *= exp2(segm_max - overall_max)[:, None]         # rescale each partial output
acc_sum        = tl.sum(segm_output, axis=0)                   # add the rescaled outputs
acc            = acc_sum / overall_expsum                      # divide → final
```

That is exactly the merge formula our HIP Kernel A computes:

```
final = Σ_i exp(LSE_i − max)·O_i  /  Σ_i exp(LSE_i − max)
```

---

## 3d. Tensor layouts and grid layouts

This section makes the shapes concrete. Two ideas to keep straight:

- **Tensor layout** = the *shape* of the data in memory (the arrays the kernel
reads/writes).
- **Grid layout** = how many *program instances* (Triton) or *workgroups* (HIP)
get launched, and what each one is responsible for. In Triton you read it off
`tl.program_id(axis)`; in HIP off `blockIdx`/`threadIdx`.

### The tensors (with example MLA-decode numbers)

Let `T` = number of query tokens, `H` = query heads (e.g. 128), `Dv` = head/value
dim (e.g. 512), `S` = number of segments (splits, e.g. 8).


| tensor                          | shape                                           | who writes / reads it                          |
| ------------------------------- | ----------------------------------------------- | ---------------------------------------------- |
| `query` (Q)                     | `[T, H, Dv]`                                    | input to both Triton paths                     |
| `key_cache` / `value_cache`     | `num_blocks, block_size,` `[num_kv_heads, Dv]` | paged KV cache, input                          |
| `output` (final)                | `[T, H, Dv]`                                    | written by `..._2d` or by `reduce_segments`    |
| `**segm_output`** (partial O)   | `**[T, H, S, Dv]**`                             | written by `..._3d`, read by `reduce_segments` |
| `**segm_max**` (partial max)    | `**[T, H, S]**`                                 | written by `..._3d`, read by `reduce_segments` |
| `**segm_expsum**` (partial sum) | `**[T, H, S]**`                                 | written by `..._3d`, read by `reduce_segments` |


The three `segm_*` tensors are the **partials** — the only tensors that exist
*because* the work was split. They carry an extra `**S` (segment) axis** that the
final output does not have; the whole job of the merge kernel is to collapse that
`S` axis away. (Allocated at unified_attention.py:434-455; only when
`NUM_SEGMENTS > 1`.)

Picture one `(token, head)` cell. The final output is a single `Dv`-long vector.
The partial for that same cell is `S` such vectors stacked — one per segment —
each with its own scalar `max` and `expsum`:

```
   final  output[t,h]            partial  segm_output[t,h]
   ┌────────────── Dv ──────┐     ┌────────────── Dv ──────┐  max   expsum
   │■■■■■■■■■■■■■■■■■■■■■■■■│ s0  │○○○○○○○○○○○○○○○○○○○○○○○○│  m0     l0
   └────────────────────────┘ s1  │○○○○○○○○○○○○○○○○○○○○○○○○│  m1     l1
            ▲                 s2  │○○○○○○○○○○○○○○○○○○○○○○○○│  m2     l2
            │                 …   │             …          │  …      …
            │                 s7  │○○○○○○○○○○○○○○○○○○○○○○○○│  m7     l7
            │                     └────────────────────────┘
            │                                 │
            └──── reduce_segments ────────────┘
              (LSE-weighted sum over the S rows → one row)
```

So `segm_output` is `[T, H, S, Dv]` and `output` is `[T, H, Dv]`: same `T,H,Dv`,
plus the extra `S` that the reduce eats. HIP packs each row's `(max, expsum)` into
**one** `partial_lse` scalar instead of two, but the shape story is identical.

Compare to the **HIP** Kernel A tensors (same idea, different packing):


| HIP tensor       | shape          | note                                               |
| ---------------- | -------------- | -------------------------------------------------- |
| `partial_output` | `[row, H, Dv]` | rows gathered indirectly via `reduce_partial_map`  |
| `partial_lse`    | `[row, H]`     | **one** LSE per (row,head) = packed `log(sum)+max` |
| `final_output`   | `[bs, H, Dv]`  | merged result                                      |


Two layout differences from Triton:

1. **Partial encoding.** Triton stores **two** small tensors (`segm_max` +
  `segm_expsum`); HIP stores **one** (`partial_lse`, the combined LSE). Same info.
2. **Segment axis.** Triton makes the split a real array axis `S`
  (`[T,H,S,Dv]`). HIP instead flattens partials into `row`s and uses a CSR index
   (`reduce_indptr` + `reduce_partial_map`) to say which rows belong to which
   merge group. Triton's is simpler/regular; HIP's is more flexible for
   variable-length / sparse groupings.

### The grid layouts

**Triton `..._2d` (one-pass)** — 2-D grid (unified_attention.py:366):

```
grid = (num_kv_heads, total_num_q_blocks)
program_id(0) = kv_head    program_id(1) = which query-block
```

Each program owns one query-block and walks **all** key-tiles itself.

**Triton `..._3d` (split stage-1)** — 3-D grid (unified_attention.py:528):

```
grid = (total_num_q_blocks, num_kv_heads, NUM_SEGMENTS)
program_id(0) = query-block   program_id(1) = kv_head   program_id(2) = segm_idx
```

The **third axis is the split.** Adding `S` segments multiplies the number of
programs by `S` — that's the whole point: manufacture `S×` more parallel work to
fill the GPU at decode. Each program walks only its segment's key-tiles
(`for j in range(segm_idx·tiles_per_segment, …)`) and writes one `S`-slice of the
`segm_`* partials.

**Triton `reduce_segments` (split stage-2)** — 2-D grid (unified_attention.py:591):

```
grid = (q.shape[0], num_query_heads)   # = (T, H)
program_id(0) = query token   program_id(1) = query head
```

**One program per (token, head).** Each reads that cell's `S` partials across the
segment axis, merges them (the 6 lines in §3c), and writes one final output row.
Note there is **no segment axis in the grid** — the segments are *looped/reduced
inside* each program, not spread across programs. That's what makes it a reduce.

**HIP Kernel A** — for contrast, its grid is **3-D** `(H, kNumThreadGroupPerBh, num_reduce_tile)` in grid-launch mode, or a flat 1-D persistent grid that
grid-strides over all `(head, q-group, tile)` work items. Within a workgroup, the
128 threads split the `Dv` output row (`Dv/128` floats each) — i.e. HIP also
parallelizes *across the head dimension inside the block*, which the Triton
reduce leaves to one program's tile.

### The split pipeline as dataflow

Follow one `(token t, head h)` cell through the two-stage split path. The `S`
segments fan **out** in stage-1 (one program each) and fan back **in** at stage-2
(one program reads all `S`):

```
            STAGE 1  (..._3d)                 STAGE 2  (reduce_segments)
        grid axis 2 = segment                 grid = (token, head)
        ────────────────────────              ─────────────────────

 keys   prog(t,h,s0) ─ walks ⅛ keys ─▶ segm_*[t,h,0] ┐
  for   prog(t,h,s1) ─ walks ⅛ keys ─▶ segm_*[t,h,1] │
 (t,h)  prog(t,h,s2) ─ walks ⅛ keys ─▶ segm_*[t,h,2] ├▶ prog(t,h) ─▶ output[t,h]
 split    …                              …           │   merges the      (final
  ×S    prog(t,h,s7) ─ walks ⅛ keys ─▶ segm_*[t,h,7] ┘   S partials       row)

        S programs per cell                   1 program per cell
        (fan-OUT: more parallelism)           (fan-IN: the reduce)
```

Compare the one-pass path, which never fans out — a single program does the whole
cell, so there are no partials and no second stage:

```
            ..._2d  (one-pass)
        grid = (kv_head, q_block)

 keys   prog(t,h) ─ walks ALL keys ─▶ output[t,h]      (no segm_*, no merge)
```

The split path trades **one** extra tensor axis (`S`) and **one** extra kernel
launch (the reduce) for `S×` more programs in stage-1 — worth it at decode when
there aren't otherwise enough `(token, head)` cells to fill the GPU.

### One picture

```
                 grid axes                         tensor written
 ..._2d     (kv_head, q_block)                  output[T,H,Dv]          ← no partials
 ..._3d     (q_block, kv_head, segment)         segm_*[T,H,S,(Dv)]      ← partials, +S axis
 reduce     (token, head)                        output[T,H,Dv]          ← S axis collapsed
 HIP A      (head, q_group, reduce_tile)         final_output[bs,H,Dv]   ← merges CSR groups
```

The **segment axis appears in stage-1's grid, lives in the partial tensors, and is
gone from stage-2's grid** — collapsed by the reduce. That single axis is the
entire reason the reduce kernel exists.

---

## 4. Kernel A (HIP reduce) vs `reduce_segments` (Triton merge)

These two do the **same job**. Differences are in encoding and tuning, not in math.


|                    | **HIP `kn_mla_reduce_v1`**                         | **Triton `reduce_segments`**                        |
| ------------------ | -------------------------------------------------- | --------------------------------------------------- |
| Job                | merge split partials → final output                | merge segment partials → final output               |
| Weight encoding    | one **LSE** = `log(sum)+max` per split             | **max** and **expsum** stored separately            |
| Exp base           | `expf` (base e)                                    | `exp2` (base 2)                                     |
| Code paths         | **two** — "simple" (2-3 splits) and "massive" (≥4) | **one** straightforward path                        |
| Skip final divide? | Yes, "massive" pre-normalizes the weights          | No, divides once at the end                         |
| How many splits?   | from CSR `reduce_indptr` (varlen, indirect gather) | from `NUM_SEGMENTS_PER_SEQ` (computed from seq_len) |
| Output partition   | 128 threads each own `Dv/128` floats of the row    | Triton tile handles the head dim                    |
| Launch             | grid-launch OR persistent grid-stride              | one program per (token, head)                       |
| Has matmul?        | **No** (pure reduction)                            | **No** (pure reduction)                             |
| Written in         | hand-tuned HIP/C++ (opus buffer ops)               | Triton (Python JIT)                                 |


The HIP kernel is the more heavily **hand-optimized** of the two: it specializes by
split count, stages the gather indices in LDS, software-pipelines the loads, and
pre-normalizes to skip a divide. The Triton version is simpler and leans on the
compiler. Both are **memory-bandwidth bound** — they just read partials, weight,
sum, and write; there's no heavy compute to hide behind.

---

## 5. Kernel A vs the one-pass kernel (`..._2d`)

These are **not** the same job and shouldn't be compared head-to-head:


|                         | **HIP `kn_mla_reduce_v1`**                | **Triton `..._2d` (one-pass)** |
| ----------------------- | ----------------------------------------- | ------------------------------ |
| Stage of attention      | only stage-2 (merge)                      | all stages (QK + softmax + PV) |
| Matmul?                 | No                                        | Yes (Q·Kᵀ and P·V)             |
| Online softmax over…    | splits                                    | key-tiles, inline              |
| Needs a partner kernel? | yes — it's the partner to a split stage-1 | no — self-contained            |


The one-pass kernel is the *alternative strategy* to the whole split pipeline that
Kernel A belongs to.

---

## 6. Why have a split path at all?

Both strategies give the **same answer** — the choice is about **keeping the GPU
busy**, not correctness.

- **One-pass** is simplest and best when there are **lots of queries** to spread
across the GPU's compute units (e.g. prefill, big batches).
- **Split (stage-1 + reduce)** wins at **decode**: the batch is small, so there are
**few queries** — not enough to fill the machine. Splitting each query's keys
across many workers manufactures more parallel work to fill all the compute
units. The cost is that you must then **merge the partials** — and that merge is
exactly Kernel A / `reduce_segments`.

---

## 7. Tie-back to our FlyDSL task

We are porting **Kernel A** (the HIP reduce) to FlyDSL for gfx942. File B's
`reduce_segments` is a useful **second reference** for the port because:

1. It's the **same merge math** in a higher-level language — easy to read.
2. It already uses `**exp2`**, which is one of the knobs our FlyDSL plan exposes
  (`use_exp2`).
3. It uses **one simple path** (no simple/massive split) — which is roughly what a
  *first* FlyDSL version should look like before we add optimizations.

When reading the HIP kernel for the port, remember: the part that makes it long and
complex (two paths, LDS-staged gather, software pipelining, skip-the-divide) is
**optimization**, not core logic. The core logic is the six lines of
`reduce_segments` in §3c.

---

## 8. How "segments" (Triton) differ from "splits" (HIP)

A segment and a split are the **same concept** — one worker's slice of the KV
sequence, producing one partial that the reduce later merges. They merge identically
(§3c == §4). The differences are entirely in **how the work is grouped, counted, and
addressed**, and those differences are what the FlyDSL port has to reproduce. Concretely:

### 8.1 Counting: uniform constexpr vs per-group variable

- **Triton segments are a single compile-time constant `NUM_SEGMENTS_PER_SEQ`**,
  chosen once on the host for the *whole launch* and baked into the kernel as a
  `tl.constexpr` (unified_attention.py launcher:147-154). The heuristic is
  "manufacture enough programs to fill the machine":
  `num_segments = ceil(target_num_prgms / num_2d_prgms)`, then clamped to
  `[MIN_SEGMENTS=8, MAX_SEGMENTS=min(128, ceil(max_seqlen_k/TILE_SIZE))]` and rounded
  up to a power of two. **Every sequence in the batch uses the same segment count.**
  Per-sequence variation in real KV length is absorbed *inside* the fixed count:
  `tiles_per_segment = cdiv(seq_len, num_segments·TILE_SIZE)` (line 552), and segments
  that fall past the end of a short sequence simply early-return
  (`if segm_idx·tiles_per_segment·TILE_SIZE >= seq_len: return`, line 554) or are
  masked out in the reduce (`act_num_segments`, segm_mask, lines 890-892).

- **HIP splits are a per-merge-group runtime count** read from a CSR structure:
  `num_splits = reduce_indptr[tile+1] − reduce_indptr[tile]`. There is **no global
  split constant** — each reduce-tile can have a *different* number of splits, decided
  by the metadata kernel that built `reduce_indptr`. The kernel even branches on it at
  runtime (simple ≤3 vs massive ≥4, plus the 64/256/LDS sub-tiers).

  **Consequence:** Triton's count is static → the compiler unrolls and sizes registers
  for exactly `NUM_SEGMENTS`. HIP's count is dynamic → it loops and *specializes by
  range* instead, which is precisely why the HIP kernel has multiple code paths the
  Triton one doesn't need.

### 8.2 Addressing: dense array axis vs CSR gather

- **Triton: the segment is a real, dense tensor axis.** Partials are
  `segm_output[T, H, S, Dv]`, `segm_max[T,H,S]`, `segm_expsum[T,H,S]`. Segment `s` of
  cell `(t,h)` lives at a computed stride offset — `…+ s·HEAD_SIZE_PADDED + …`
  (lines 918-921). Regular, contiguous, no indirection. The reduce loads the whole
  `S` row with one strided `tl.load` + a `segm_mask`.

- **HIP: the split is a flattened row reached through an indirection.** Partials are
  `partial_output[row, H, Dv]`; the `S` axis does **not** exist as a dimension.
  Instead `reduce_partial_map[reduce_indptr[tile] + s]` is a **gather index** giving
  the physical `row` for split `s`, and `reduce_indptr` (CSR offsets) says which slice
  of the map belongs to each tile. The kernel stages those gather indices into LDS
  once and reuses them.

  **Consequence:** Triton trades flexibility for regularity — a rectangular `[T,H,S,Dv]`
  block, easy to index, but it must allocate for the *max* segment count and mask the
  unused tail. HIP trades regularity for density — no wasted rows, variable group
  sizes, sparse/varlen-friendly — at the cost of an indirection layer (the CSR map +
  LDS staging) that the Triton version has no equivalent of.

### 8.3 What stays the same

- The **merge math** is identical (LSE-weighted online-softmax combine; §2).
- Both are **pure reductions, memory-bandwidth bound**, no matmul.
- Both **collapse the segment/split axis**: it exists in stage-1's grid and in the
  partial tensors, and is gone from the reduce's grid (§3d).
- A segment count of 1 / a single-split tile is the degenerate "no real split" case
  both handle specially (Triton: `if NUM_SEGMENTS==1` copy path, line 822 / 586; HIP:
  the simple path / early-exit sentinel).

### 8.4 Summary table

|                         | **Triton segment**                              | **HIP split**                                       |
| ----------------------- | ----------------------------------------------- | --------------------------------------------------- |
| Count is…               | one **constexpr** `NUM_SEGMENTS_PER_SEQ`, global | **runtime, per-tile** `indptr[t+1]−indptr[t]`        |
| Chosen by               | host heuristic (fill machine), pow2, clamp 8–128 | metadata kernel that builds the CSR                 |
| Same for all sequences? | **yes** (one constant for the launch)           | **no** (each reduce-tile may differ)                |
| Lives in memory as      | a dense tensor **axis** `S` in `[T,H,S,Dv]`     | flattened **rows**, no `S` axis                     |
| Addressed by            | computed stride `s·HEAD_SIZE_PADDED`            | **CSR gather** `reduce_partial_map` + `reduce_indptr` |
| Short-sequence handling | masked tail (`act_num_segments`, early-return)  | exact group size (no waste); early-exit sentinel    |
| Forces multiple paths?  | no — static count, compiler unrolls             | **yes** — dynamic count → simple/massive sub-tiers  |

### 8.5 Why it matters for the port

The HIP kernel's apparent complexity is largely a **consequence of the split being a
runtime, variable, gather-indexed quantity** rather than a fixed array axis. When
porting to FlyDSL, the CSR gather (`reduce_indptr` + `reduce_partial_map`) and the
LDS-staging of those indices are **correctness-critical** and have no counterpart in
the simpler Triton reference — `reduce_segments` can be read for the *math*, but its
dense, constexpr-segment addressing must **not** be copied; the port must keep HIP's
CSR/variable-split model.
