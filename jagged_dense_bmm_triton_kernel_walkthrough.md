# Walkthrough — Meta's Triton `jagged_dense_bmm_broadcast_add` Kernel

_A line-by-line tour of the reference kernel we're porting to AMD, written for a junior
engineer. Lots of pictures. Read top to bottom._

- **File:** `~/generative-recommenders/generative_recommenders/ops/triton/triton_jagged.py`
- **Kernel:** `jagged_dense_bmm_broadcast_add_kernel` (line 294)
- **Forward wrapper (with bias):** `triton_jagged_dense_bmm_add_fwd` (line 807)

---

## 0. The 30-second summary

We do **many matrix multiplies at once**, one per "group". Each group has a different number
of rows, but the rows are all packed into one big tensor. After multiplying, we add a small
bias vector that is **the same for every row in the group**.

```
   For each group b:   Out[group b] = Jagged[group b] @ Dense[b] + Bias[b]
                        └── (M_b × N) ──┘   (M_b × K) (K × N)   (1 × N broadcast)
```

That's it. Everything below is *how* one GPU kernel pulls this off when the row counts are
unknown on the CPU.

---

## 1. What "jagged" means

A normal batched tensor is a neat box: `B` groups, each with the same number of rows.
A **jagged** tensor has groups with *different* row counts, packed back-to-back with no
padding:

```
  Normal batch (padded)              Jagged (packed, no waste)
  ┌──────────────┐                   ┌──────────────┐  ← group 0 (3 rows)
  │ group0 row0  │                   │ g0 row0      │
  │ group0 row1  │                   │ g0 row1      │
  │ group0 PAD   │ ← wasted          │ g0 row2      │
  │ group0 PAD   │ ← wasted          ├──────────────┤  ← group 1 (1 row)
  ├──────────────┤                   │ g1 row0      │
  │ group1 row0  │                   ├──────────────┤  ← group 2 (4 rows)
  │ group1 PAD   │ ← wasted          │ g2 row0      │
  │ group1 PAD   │ ← wasted          │ g2 row1      │
  │ group1 PAD   │ ← wasted          │ g2 row2      │
  └──────────────┘                   │ g2 row3      │
                                     └──────────────┘
                                     Total rows L = 3+1+4 = 8
```

**How do we know where each group starts and ends?** A small array called `seq_offsets`
(a running total, a.k.a. prefix sum). For the example above:

```
  group:         0    1    2
  row counts:    3    1    4
  seq_offsets: [ 0,   3,   4,   8 ]     ← length is B+1
                 │    │    │    └── end of group 2
                 │    │    └─────── start of group 2 / end of group 1
                 │    └──────────── start of group 1 / end of group 0
                 └───────────────── start of group 0

  group b owns rows [ seq_offsets[b] , seq_offsets[b+1] )
  M_b (its row count) = seq_offsets[b+1] - seq_offsets[b]
```

**Key headache:** `seq_offsets` lives in **GPU memory**. The CPU launching the kernel does
NOT know how big each group is. So the kernel must read these offsets itself, on the GPU.

---

## 2. The four tensors (shapes & layout)

```
  Jagged (A)            Dense (B)                 Bias              Out
  packed input          one weight per group      one vec per grp   packed output
  shape (L, K)          shape (B, K, N)           shape (B, N)      shape (L, N)
  ┌────────┐ K          group0  group1  ...       ┌──────┐ N        ┌────────┐ N
  │ row0   │            ┌────┐  ┌────┐             │ b0   │         │ row0   │
  │ row1   │            │K×N │  │K×N │             │ b1   │         │ row1   │
  │ ...    │ L          │    │  │    │   B of them │ ...  │ B       │ ...    │ L
  │ rowL-1 │            └────┘  └────┘             └──────┘         │ rowL-1 │
  └────────┘            Dense[b] is the (K×N)                       └────────┘
                        weight matrix for group b
```

- All tensors are **row-major** (rows stored contiguously). BF16 data, FP32 accumulation.
- `Bias[b]` is one vector of length `N`. It gets added to **every row** of group `b`'s output
  — that's the "broadcast add".

> ⚠️ **Naming trap:** inside the kernel, the variable names are `K` (the shared/reduction
> dim) and `N` (the output width). But the HSTU *benchmark* names them differently (`D` and
> `K`). This walkthrough uses the *kernel's* names. Just know the benchmark labels don't match.

---

## 3. The launch grid — how work is split across the GPU

A GEMM output is too big to compute at once, so we chop it into **tiles**. Each GPU
workgroup ("program") computes one tile. The kernel is launched over a **3D grid**
(wrapper, line 821):

```
  grid = ( cdiv(N, BLOCK_N),          ← axis 0: which column-tile of the output
           cdiv(max_seq_len, BLOCK_M),← axis 1: which row-tile (sized by the BIGGEST group!)
           B )                        ← axis 2: which group
```

Picture one group's output, chopped into tiles. `program_id(0)` picks the column,
`program_id(1)` picks the row-band, `program_id(2)` picks the group:

```
                   N (output width)
            BLOCK_N  BLOCK_N  BLOCK_N
          ┌────────┬────────┬────────┐
  BLOCK_M │ (0,0)  │ (1,0)  │ (2,0)  │   each cell = one GPU program
          ├────────┼────────┼────────┤   cell label = (program_id(0), program_id(1))
  BLOCK_M │ (0,1)  │ (1,1)  │ (2,1)  │
          ├────────┼────────┼────────┤
  BLOCK_M │ (0,2)  │ (1,2)  │ (2,2)  │
          └────────┴────────┴────────┘
          ... and this whole picture is repeated B times (once per group, axis 2)
```

**The clever/simple trick:** axis 1 is sized by `max_seq_len` — the row count of the
*largest possible* group. So every group gets the *same* number of row-tiles launched. Groups
that are smaller will have extra tiles at the bottom that have no real work — those just
**exit early** (next section). This keeps the launch logic dead simple at the cost of a few
wasted (instantly-returning) programs.

```
  max_seq_len = 8 rows → 4 row-tiles launched per group (BLOCK_M=2)

  group 0 (M_b=3)        group 1 (M_b=1)        group 2 (M_b=8)
  ┌─────────┐ tile0 ✓    ┌─────────┐ tile0 ✓    ┌─────────┐ tile0 ✓
  │ rows0-1 │            │ row0    │            │ rows0-1 │
  ├─────────┤ tile1 ✓*   ├─────────┤ tile1 ✗    ├─────────┤ tile1 ✓
  │ row2    │            │ (none)  │ early-exit │ rows2-3 │
  ├─────────┤ tile2 ✗    ├─────────┤ tile2 ✗    ├─────────┤ tile2 ✓
  │ (none)  │ early-exit │ (none)  │ early-exit │ rows4-5 │
  ├─────────┤ tile3 ✗    ├─────────┤ tile3 ✗    ├─────────┤ tile3 ✓
  │ (none)  │ early-exit │ (none)  │ early-exit │ rows6-7 │
  └─────────┘            └─────────┘            └─────────┘
  ✓ = does work   ✗ = early-exit (wasted launch)   * = partial tile (masked)
```

---

## 4. The kernel, step by step

Now the kernel body (line 294). Each program runs this once for its `(off_n, off_m, off_b)`.

### Step A — figure out which group and which tile (lines 322–332)

```python
off_n = tl.program_id(0)               # my column-tile index
off_m = tl.program_id(1).to(tl.int64)  # my row-tile index  (int64: row counts get big)
off_b = tl.program_id(2)               # my group index

seq_start = tl.load(seq_offsets + off_b)       # where my group starts
seq_end   = tl.load(seq_offsets + off_b + 1)   # where my group ends
seq_len   = seq_end - seq_start                # M_b: my group's real row count

start_m = off_m * BLOCK_M               # first row (within group) this tile handles
start_n = off_n * BLOCK_N              # first column this tile handles

if start_m >= seq_len:                  # ← THE EARLY EXIT
    return                              # my tile is past the end of this group → quit
```

This is the heart of "jagged on the GPU": read the offsets, compute the real length, and
**bail out** if this tile fell off the end of a short group.

```
   start_m (where my tile begins)
        │
        ▼
   ┌──────────────────────────┐
   │■■■■■■■■■■■│ real rows      │   start_m < seq_len  →  DO WORK
   └──────────────────────────┘
                    seq_len ┘

   ┌──────────────────────────┐
   │ real rows │               │   start_m
   └──────────────────────────┘      │
              seq_len ┘               ▼ start_m >= seq_len  →  return (nothing to do)
```

### Step B — slide the pointers to MY group's data (lines 334–336)

The tensors are huge and shared. Each program advances its base pointers so it only "sees"
its own slice. This is just **pointer arithmetic** — no data is copied.

```python
Jagged += (seq_start + start_m) * stride_jm   # jump to my group's rows (+ my row-tile)
Dense  += off_b * stride_db                    # jump to Dense[b], my group's weight
Out    += seq_start * stride_om                # jump to my group's output rows
```

```
  Jagged (the big packed tensor)
  ┌──────────────────────────────────────────┐
  │ g0 rows │ g1 rows │■■■ g2 rows ■■■│ g3 ...│
  └──────────────────────────────────────────┘
             seq_start ┘──────► Jagged now points here (start of group 2)
                       + start_m more rows for this specific tile

  Dense                          Bias
  [ b0 | b1 |■b2■| b3 ]          [ b0 | b1 |■b2■| b3 ]
            └► Dense[b]                    └► Bias[b]
```

### Step C — the multiply loop (the "mainloop", lines 338–359)

This is a textbook tiled matrix multiply. We walk along the shared `K` dimension in chunks of
`BLOCK_K`, loading a slab of `Jagged` and a slab of `Dense`, multiplying them, and adding to a
running total kept in **FP32**.

```python
accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)   # FP32 running total
for k in range(0, K, BLOCK_K):
    jg = tl.load(jg_ptrs, mask=..., other=0.0)   # a (BLOCK_M × BLOCK_K) slab of A
    dn = tl.load(dn_ptrs, mask=..., other=0.0)   # a (BLOCK_K × BLOCK_N) slab of B
    accumulator += tl.dot(jg, dn)                 # multiply-accumulate
    jg_ptrs += BLOCK_K                            # slide right along K
    dn_ptrs += BLOCK_K * stride_dk               # slide down along K
```

```
  One output tile = sum over K of (A-slab × B-slab):

        BLOCK_K   BLOCK_K          BLOCK_N
        ┌────┐    ┌────┐          ┌────────┐
  BLOCK │ A0 │ ×  │ B0 │   BLOCK_K│   B0   │
   _M   └────┘    └────┘          └────────┘
                                  ┌────────┐
   then A1 × B1, A2 × B2 ...      │   B1   │
   adding each into the           └────────┘
   accumulator:                    ...

   accumulator (BLOCK_M × BLOCK_N) += A0·B0 + A1·B1 + ...   ← all in FP32
```

**The `mask=` arguments** handle edges so we never read past the real data:
- `offs_m < (seq_len - start_m)` → don't read rows beyond this group's last row (the partial
  bottom tile).
- `(k + offs_k) < K` and `offs_n < N` → don't read past the matrix edges when sizes aren't
  exact multiples of the block. Out-of-range reads return `0.0`, which is safe to add.

> For our HSTU shapes `K` (= bench `D`) is only 256–512 and `BLOCK_K` is 32–64, so this loop
> runs just **4–16 times**. That's *short* — which is why setup/finish overhead matters so
> much (see the dev journal's "amortize fixed costs" section).

### Step D — add the bias (lines 361–374)

Two modes. We care about the **broadcast** one (`ELEMENTWISE=False`):

```python
if HAS_BIAS:
    if ELEMENTWISE:                        # rare mode: bias is itself jagged (L, N)
        ...                                # load a bias value per output element
    else:                                  # ← THE MODE WE WANT (broadcast)
        bias_ptrs = Bias + off_b * stride_bias_b + offs_n  # one vector for group b
        bias = tl.load(bias_ptrs, mask=offs_n < N)
        accumulator += bias[None, :]       # add the SAME vector to every row
```

```
  Broadcast add: one (1 × N) bias vector, copied down across all BLOCK_M rows

           Bias[b]:  [ x0  x1  x2  ...  xN-1 ]   ← length N
                        │   │   │        │
                        ▼   ▼   ▼        ▼   (same values added to every row)
   accumulator   row0  +x0 +x1 +x2 ... +xN-1
   (BLOCK_M ×    row1  +x0 +x1 +x2 ... +xN-1
    BLOCK_N)     row2  +x0 +x1 +x2 ... +xN-1
                 ...
```

The `[None, :]` is the broadcast: it stretches the length-`N` vector across all the rows.
Bias is added in FP32 (matches the accumulator) for accuracy.

### Step E — write the result back (lines 376–386)

Cast the FP32 accumulator down to the output dtype (BF16) and store, again with a mask so we
only write real rows/columns:

```python
out = accumulator.to(Out.dtype.element_ty)        # FP32 → BF16
out_ptrs = Out + offs_m[:, None]*stride_om + offs_n[None, :]
tl.store(out_ptrs, out,
         mask=(offs_m[:, None] < (seq_len - start_m)) & (offs_n[None, :] < N))
```

```
   FP32 accumulator  ──cast──►  BF16 tile  ──masked store──►  Out[my group's rows]
   (in registers)               (smaller)                     (in HBM)
```

---

## 5. The whole flow on one page

```
                    ┌─────────────────────────────────────────────┐
   CPU launches  →  │  grid = (N-tiles, max_seq_len-tiles, B)      │
                    └─────────────────────────────────────────────┘
                                      │  (one program per cell)
                                      ▼
   ┌───────────────────────────────────────────────────────────────────┐
   │ EACH PROGRAM:                                                       │
   │                                                                     │
   │  A. read seq_offsets[b], seq_offsets[b+1]  → seq_len (M_b)          │
   │     if my row-tile is past seq_len → return (early exit)            │
   │                                                                     │
   │  B. slide pointers:  Jagged→my rows,  Dense→Dense[b],  Out→my rows  │
   │                                                                     │
   │  C. mainloop over K:   acc(FP32) += A_slab · B_slab     (tl.dot)    │
   │                                                                     │
   │  D. acc += Bias[b]  (broadcast the (1×N) vector across rows)        │
   │                                                                     │
   │  E. store  acc → BF16 → Out   (masked to real rows/cols)            │
   └───────────────────────────────────────────────────────────────────┘
```

---

## 6. How the wrapper calls it (forward, line 807)

`triton_jagged_dense_bmm_add_fwd` is the user-facing function that sets up the call:

```python
L, K = jagged.shape          # L = total packed rows, K = reduction dim
B, _, N = dense.shape        # B = groups, N = output width
out = torch.empty((L, N), ...)

grid = (cdiv(N, BLOCK_N), cdiv(max_seq_len, BLOCK_M), B)   # the 3D grid from §3

jagged_dense_bmm_broadcast_add_kernel[grid](
    seq_offsets=seq_offsets, Jagged=jagged, Dense=dense, Bias=bias, Out=out,
    N=N, K=K,
    stride_jm=..., stride_db=..., stride_dk=..., stride_dn=..., stride_om=...,
    HAS_BIAS=True, ELEMENTWISE=elementwise,   # forward-with-bias = broadcast
)
```

Notice the same kernel is reused for the **backward pass** too (lines 585, 865) — backward is
just more matrix multiplies with the operands swapped/transposed and `HAS_BIAS=False`. That's
out of scope for our v1 (forward only), but good to know the kernel is general.

---

## 7. Autotuning (lines 268–292)

The kernel is decorated with `@triton_autotune`. Triton tries many tile-size combos and picks
the fastest for each problem shape:

```python
configs: BLOCK_M ∈ {64,128}, BLOCK_N ∈ {64,128,256}, BLOCK_K ∈ {32,64},
         num_stages ∈ {3,5}, num_warps ∈ {4,8}
key = ["AUTOTUNE_MAX_SEQ_LEN", "N", "K", "ELEMENTWISE", "HAS_BIAS"]
```

- `BLOCK_M/N/K` = the tile sizes from §3–§4.
- `num_stages` = how many loop iterations are pipelined (prefetch depth).
- `num_warps` = how many warps/wavefronts per program.
- `key` = when any of these change, re-tune. (`AUTOTUNE_MAX_SEQ_LEN` is bucketed so we don't
  recompile for every tiny change in sequence length.)

When we write the AMD version, **this config space is our search space too** — but tuned for
CDNA (wavefront = 64, `waves_per_eu`, MFMA tile shapes), not NVIDIA warps.

---

## 8. What this means for our AMD port (the takeaways)

| Triton does this... | ...and our AMD kernel must do the same |
|---|---|
| Group index on grid axis 2 | Put `b` on a grid axis |
| Reads `seq_offsets` on the GPU | Read `seq_offsets` on device, compute `M_b` |
| `if start_m >= seq_len: return` | Early-exit tail tiles |
| Pointer bumps for Jagged/Dense/Out | Same pointer arithmetic (int64 for row base) |
| FP32 accumulate, add bias, cast BF16 | Same epilogue; bias broadcast = stride-0 D-tensor |
| Grid sized by `max_seq_len` | Same (uniform `M_i` → no wasted tiles) |

The Triton kernel is short and clean precisely because Triton hides the LDS staging and
pipelining. On AMD we re-create those by hand (CKTile's `UniversalGemmKernel` or a forked
FlyDSL `hgemm`), but the **logic above is exactly what we replicate**.

---

## Appendix — line index (quick jump)

| Lines | What |
|---|---|
| 268–292 | autotune config list + `@triton_autotune` decorator |
| 294–315 | kernel signature (args, strides, constexprs) |
| 322–332 | read group, compute `seq_len`, early-exit |
| 334–336 | pointer offsets to this group's data |
| 338–359 | the K mainloop (tiled multiply-accumulate) |
| 361–374 | bias add (broadcast vs elementwise) |
| 376–386 | cast to BF16 + masked store |
| 523–630 | autograd `Function` (forward + backward wiring) |
| 807–847 | `triton_jagged_dense_bmm_add_fwd` — the bias forward wrapper |
| 850–885 | backward jagged GEMM (reuses the same kernel) |
