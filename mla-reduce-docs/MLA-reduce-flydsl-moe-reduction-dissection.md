# FlyDSL `compile_moe_reduction` — dissection + suitability as the MLA-reduce base

Source: `aiter/ops/flydsl/kernels/moe_gemm_2stage.py :: compile_moe_reduction` (line 3590).
Companion to `MLA-reduce-HIP-kernel-dissection.md`. Question this answers: **is this kernel
a good structural base for the FlyDSL sparse-MLA reduce port?**

---

## PART A — Dissection of `compile_moe_reduction`

### A.0 What it computes
A plain **sum-reduction over the topk axis**, optionally masked:
```
X [tokens, topk, model_dim]  →  Y[t, d] = Σ_k X[t, k, d]   (k where valid_mask[t,k]=1)
```
Used as the non-atomic epilogue to `compile_moe_gemm2(accumulate=False)` so MoE expert
outputs are summed without atomic contention. It is a **pure elementwise reduction** — no
softmax, no LSE, no per-slot weighting beyond a 0/1 mask. Compute type is always fp32;
I/O is f16/bf16/f32.

### A.1 Compile-time config
| const | value | note |
|---|---|---|
| `BLOCK_SIZE` | 256 | threads/WG (4 wavefronts @ wave64) |
| `VEC_WIDTH` | 8 | logical fp32 elements processed per thread per tile-col step |
| `copy_vec_width` | `128/elem_bits` = 8 (f16/bf16), 4 (f32) | elems per buffer op (128-bit transactions) |
| `n_sub` | `VEC_WIDTH/copy_vec_width` = 1 (16-bit) or 2 (f32) | sub-vector loads per thread |
| `topk` | template constant | the **reduction length is a compile-time constant** |
| `model_dim` | template constant | drives static grid-y |

Everything that matters for control flow (`topk`, `model_dim`, dtype, `use_mask`) is
**baked at compile time**. The only runtime scalar is `i32_m_tokens`.

### A.2 Block & grid layout
- **Block**: `(BLOCK_SIZE=256, 1, 1)` — flat 1-D, 4 wavefronts.
- **Grid**: `(gx, gy_static, 1)` where
  - `gx = m_tokens` (runtime) — **one WG-row per token**, block_id("x") = token.
  - `gy_static = ceil(model_dim / (BLOCK_SIZE·VEC_WIDTH))` (compile-time) — tiles the
    model_dim columns; block_id("y") = column-tile.
- **Column partition**: `col_base = tile_idx·(BLOCK_SIZE·VEC_WIDTH) + tid·VEC_WIDTH`.
  Each thread owns `VEC_WIDTH=8` contiguous output columns. **Lane-partitioned output**,
  exactly like the HIP MLA kernel's `Dv/128` slice — no cross-thread comm on the output axis.
- This is a **2-D grid of (token × column-tile)**; there is no persistent/grid-stride mode
  and no fan-out heuristic — grid is sized directly to the problem.

### A.3 Memory hierarchy & staging
- **Buffer resources**: `_ptr_buffer_resource` (line 3658) builds a hardware buffer
  descriptor from the raw pointer with an explicit `num_records_bytes` byte-bound
  (`x_nbytes`, `y_nbytes`, `mask_nbytes`). This is the FlyDSL analogue of `opus::make_gmem`
  — **SGPR base + HW out-of-bounds drop**. The byte-bound is the OOB guard.
- **No LDS at all.** No `SmemAllocator`, no `gpu.barrier()`. The reduction axis (topk) is
  walked entirely in registers by each thread independently. (Contrast: the HIP MLA kernel
  needs LDS only for the *gather map* and *lse_scale* — see Part B.)
- **Registers**: `acc_vecs` = `n_sub` fp32 vectors of width `copy_vec_width`, the running
  sum accumulator. Inputs streamed HBM→VGPR via `buffer_load`, accumulated, cast, stored.
- **Access pattern**: `X[token, k, :]` is contiguous in `d`; the inner loop over `k` strides
  by `model_dim`. Each X element read **exactly once** → byte floor for a reduction.

### A.4 Control flow / guards (3-level nested `scf.IfOp`)
1. `tok_ok = token_idx < m_tokens` — runtime token bound (grid-x is runtime).
2. `col_ok = col_base < model_dim` — this column-tile has any work.
3. `end_ok = col_base + VEC_WIDTH <= model_dim` — **fast path vs tail path**:
   - **Fast path** (full vector in bounds): vectorized `buffer_load` of `copy_vec_width`,
     `range_constexpr(topk)` unrolled accumulate, optional mask `select`, `extf` to fp32,
     vectorized `buffer_store` after `truncf`.
   - **Tail path** (partial vector at the edge): per-lane scalar loop, same logic at
     `vec_width=1`. Handles `model_dim` not divisible by the tile width.

### A.5 The reduction loop
```
acc_vecs[si] = 0
for k in range_constexpr(topk):              # COMPILE-TIME unrolled
    if use_mask: mv_ok = mask[token,k] != 0
    for si in range_constexpr(n_sub):
        vec_e = buffer_load(X[(token·topk+k)·model_dim + col_base + si·cvw])
        if use_mask: vec_e = mv_ok.select(vec_e, 0)
        vec_c = vec_e.extf(f32) if 16-bit else vec_e
        acc_vecs[si] += vec_c
# store
for si: buffer_store(acc_vecs[si].truncf(elem), Y[...])
```
Key point: **`topk` is `range_constexpr`** — the reduction length is fully unrolled at
compile time. There is **no loop-carried `scf.for`**, no online/running-max bookkeeping,
no runtime trip count. It is the simplest possible reduction shape.

### A.6 Host launcher
`@flyc.jit launch_moe_reduction` (line 3801): grid `(m_tokens, gy_static, 1)`,
block `(256,1,1)`. `_MoeGemm2ReduceWrapper` (line 3827) handles the torch-side intermediate
buffer alloc, fake-tensor pointer marshalling (`_ptr_arg`), and the gemm2→reduce sequencing.

### A.7 Optimizations present
| # | optimization | where |
|---|---|---|
| 1 | lane-partitioned output (`VEC_WIDTH` cols/thread) | col_base, 3684 |
| 2 | 128-bit vectorized buffer load/store | copy_vec_width, 3716/3747 |
| 3 | SGPR buffer descriptors w/ byte-bound HW OOB drop | _ptr_buffer_resource, 3658 |
| 4 | `range_constexpr(topk)` full unroll | 3702 |
| 5 | fast-path / tail-path split (vector vs scalar edge) | end_ok, 3692 |
| 6 | fp32 accumulate, late truncate | acc_vecs, 3734/3740 |
| 7 | mask via `select` (branchless) | 3728 |
| 8 | each input read once (byte floor) | loop structure |

### A.8 What it does NOT have (relative to a reduction kernel zoo)
- No LDS / no barriers / no cross-thread reduction.
- No `scf.for` runtime loop / no loop-carried state.
- No gather/indirection — indices are arithmetic (`token·topk+k`), **not** a data-dependent
  `partial_map[i]`.
- No online-softmax / LSE / per-element rescaling.
- No persistent kernel / no work-fanout heuristic.
- No runtime-strided output (Y is contiguous).

---

## PART B — Is this a good base for the FlyDSL sparse-MLA reduce?

### B.1 Side-by-side of the two reductions

| dimension | `compile_moe_reduction` (FlyDSL) | `kn_mla_reduce_v1` (HIP, the spec) |
|---|---|---|
| math | `Σ_k X` (+ 0/1 mask) | **LSE-weighted online softmax** combine + normalize |
| reduction length | `topk`, **compile-time const** | `num_splits`, **runtime** (from `reduce_indptr`) |
| indexing | arithmetic `token·topk+k` | **data-dependent gather** `reduce_partial_map[i]` |
| per-slot weight | none (mask only) | `exp(LSE_s − global_LSE)`, needs a reduction first |
| cross-thread comm | none | warp-0 LSE max+sum reduction → LDS broadcast |
| LDS | none | gather-map + lse_scale staged in LDS |
| loop form | `range_constexpr` unroll | runtime `scf.for` w/ loop-carried `(acc,max,sum)` |
| output partition | `VEC_WIDTH` cols/thread | `Dv/128` floats/thread — **same idea** |
| buffer ops | SGPR rsrc + byte-bound OOB | SGPR make_gmem + 16B dwordx4 — **same idea** |
| dtype handling | f16/bf16/f32, fp32 accum, late trunc | fp32 in, bf16/fp16 out, fp32 accum — **same idea** |
| launch | static 2-D grid only | grid-launch **and** persistent + fanout heuristic |
| paths | fast vec / scalar tail | simple (<4 splits) / massive (≥4) sub-tiered |

### B.2 What genuinely transfers (reuse these)
1. **The scaffolding shell** — `@flyc.kernel` arg marshalling, `_ptr_buffer_resource`
   (buffer descriptor from raw ptr with byte-bound), the `@flyc.jit` launcher, the
   torch-side `_ptr_arg`/fake-tensor handling. This is ~40% of the boilerplate and it is
   directly copyable.
2. **Lane-partitioned vectorized output** — `col_base = tile·width + tid·VEC` and the
   128-bit `buffer_load`/`buffer_store` pattern map 1:1 to MLA's `Dv/128` lane slice. Reuse
   verbatim for the final-O store and the partial-O load.
3. **fp32-accumulate + late `truncf` to bf16/fp16** — exactly MLA's output cast.
4. **fast-path/tail-path structure** — MLA's `Dv` (512/128) is clean, but the pattern is
   good insurance for `Dv=128` or odd head dims.
5. **`select`-based masking** — reusable for the NaN→0 partial guard MLA needs.

### B.3 What must be added (the kernel does NOT cover these)
1. **Runtime reduction length** — MLA's `num_splits` is runtime, so the topk
   `range_constexpr` loop must become a **runtime `scf.for` with loop-carried state**. This
   is the single biggest structural change and a known FlyDSL pitfall (see
   `debug-flydsl-kernel`: loop-carried packing, range vs range_constexpr).
2. **Data-dependent gather** — read `reduce_indptr[tile]/[tile+1]`, then gather
   `partial_output[reduce_partial_map[start+s]]`. moe_reduction has *zero* indirection;
   you import the gather idiom from the jdbba KB (`flydsl-jdbba/00-layout-api-idioms`:
   device scalar read + runtime base-offset views).
3. **The online-softmax / LSE machinery** — running `max_lse`, `exp2/exp`, `sum_e`, the
   `old_scale/new_scale` rescale (simple path) OR the warp-0 global-LSE reduction + LDS
   `lse_scale` broadcast (massive path). None of this exists in moe_reduction. The wave
   reduce comes from `reduce.py` helpers; `exp2/rcp` from `rocdl`.
4. **LDS usage + barriers** — needed for the massive path's `lse_scale` broadcast and the
   staged gather map. moe_reduction uses no LDS, so you add `SmemAllocator` + `gpu.barrier()`
   from scratch (pattern: `flydsl-kernel-authoring` SmemAllocator section).
5. **Two-path dispatch (simple vs massive)** + optionally the split-count sub-tiers.
6. **Runtime-strided final output** (`stride_s_o/stride_h_o`) — moe_reduction's Y is
   contiguous; MLA must thread runtime strides into the store address.
7. **The 3-level work decomposition** (head × block × tile) + optional persistent launch.
   moe_reduction's flat (token × col-tile) grid is simpler than what MLA needs.

### B.4 Verdict

**Use it as the *scaffolding* base, not the *algorithmic* base.**

- As a **boilerplate/skeleton donor** it is the best starting point in the FlyDSL tree:
  it's the only existing FlyDSL kernel whose *purpose* is a reduction with vectorized
  buffer I/O and fp32-accum/late-cast, and its buffer-resource + launcher + dtype handling
  are directly reusable. Starting from it saves real time on the parts that are pure
  ceremony.
- As an **algorithm template it covers maybe the simple path's *shape* but none of its
  *substance*.** The defining features of MLA reduce — runtime split count, data-dependent
  gather, LSE-weighted online softmax, the warp-reduce + LDS-broadcast massive path, the
  two-path dispatch — are all absent. Those parts come from (a) the HIP kernel as the spec,
  (b) `reduce.py` for wave/block reduce, (c) the jdbba KB for the gather/varlen idioms,
  (d) `rocdl` for exp2/rcp.

**Recommended construction order:** fork the moe_reduction shell → swap the arithmetic
index for the `reduce_partial_map` gather → replace the `range_constexpr` sum with a
runtime `scf.for` online-softmax merge (simple path first, verify vs `torch_mla_reduce_v1`
at `n_splits∈{2,3}`) → add LDS + warp-reduce for the massive path. This reuses moe_reduction
for ~the I/O and host plumbing while sourcing the algorithm from the HIP reference.

**Risk note:** the easy reuse (vectorized I/O, dtype cast, launcher) is also the part that
was never the hard part. The hard parts (runtime loop-carried softmax, gather addressing,
warp-reduce-into-LDS) get *no* head start from this kernel — budget accordingly. Net: a
genuine but **modest** head start (~scaffolding + I/O), consistent with the plan's
"reuse compile_moe_reduction scaffolding" framing, not a drop-in algorithmic match.
</content>
