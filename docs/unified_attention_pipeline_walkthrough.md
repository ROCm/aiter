# CK‑UA Pipeline Walkthrough

A high‑level reading guide to
[`unified_attention_pipeline.hpp`](../3rdparty/composable_kernel/include/ck_tile/ops/unified_attention/pipeline/unified_attention_pipeline.hpp)
(1226 lines).

The kernel (`unified_attention_kernel.hpp`) is just orchestration — schedule, address
math, mask construction. **All the real work happens in this pipeline file.** It runs
the per‑Q‑block attention computation: paged KV walk, `Q·Kᵀ` matmul, online softmax,
masking, `P·V` matmul, and final normalization.

> **Mental model.** Think of it as a hand‑scheduled FlashAttention‑2 inner loop with
> double‑buffered LDS, async DRAM prefetch, and warp specialization — implemented as
> small reusable lambdas (`K_mem_load`, `fmha_alu0`, `gemm`, …) glued together by a
> short pre‑stage / main‑loop / post‑process spine.

---

## 0. Top‑level layout

```
lines     section
─────────────────────────────────────────────────────────────────
   1– 35  preamble: includes, ASM_MARKER macros, debug helpers
  36– 96  class header: template params, types, tile constants
  98–162  small helpers: GetSmemSize, simple LDS desc, s_waitcnt
 164–192  operator() signature
 194–340  setup: LDS windows, GEMM types, tile distributions, Q load
 340–415  paged-KV addressing: pointer rebasing, block_table walk
 416–565  KV-load / LDS-load helper lambdas
 575–805  online softmax + GEMM + mask helper lambdas
 820–972  CoreLoopScheduler-driven 4-phase iteration (warp specialization)
 974–994  fmha_post_process — final PV for the trailing block
 996–1040 pre-stage — bootstrap the pipeline
1042–1148 main loop — two flavors: 1-warpgroup serial, 2-warpgroup interleaved
1149–1180 final normalize: divide o_acc by rowsum l, return
1183–1222 thin overload that defaults all elementwise functions to identity
```

---

## 1. Class skeleton (lines 38–96)

```cpp
template <typename Problem_,
          typename Policy_ = UnifiedAttentionPipelineDefaultPolicy>
struct UnifiedAttentionPipeline { ... };
```

- **`Problem`** — datatypes (Q/K/V/O/Sacc/PCompute), `FmhaMask`, head‑dim, padding flags.
- **`Policy`** — strategy choices: alignments, GEMM types, register/LDS tile
  distributions, warp grouping. See companion file `*_default_policy.hpp`.

Compile‑time constants pulled from `Problem::UnifiedAttentionShape`:

| symbol | what |
|---|---|
| `kBlockSize` | threads per workgroup |
| `kBlockM` | Q tile rows = (q‑tokens × G heads) flattened |
| `kBlockQ` | Q tile rows / G — original q‑tokens per block |
| `kPageBlockSize` | KV‑axis tile (= `BLOCK_N`); aligned to paged‑KV page size |
| `kHeadDim` / `kHeadDimPadded` | head dim, possibly padded to next pow2 |
| `kAlignmentQ/K/V/O` | DRAM vector load widths (`Policy::GetAlignmentX`) |
| `kBlockPerCu` | `__launch_bounds__` hint (default 2) |

`GetSmemSize()` (line 98) reports the LDS budget — 4 K/V buffers (double‑buffered) +
P tile + O tile, staged by `Policy::GetSmemSize<Problem>()`.

---

## 2. `operator()` signature (lines 173–192)

```cpp
auto operator()(
    q_dram_block_window_tmp,    q_element_func,
    k_dram_block_window_tmp,    k_element_func,
    v_dram_block_window_tmp,    v_element_func,
    num_blocks, num_blocks_start,
    block_tables_ptr, block_table_offset, kv_page_size_in_blocks,
    s_acc_element_func, p_compute_element_func, o_acc_element_func,
    mask, scale_s, smem_ptr,
    k_row_stride = 0, v_row_stride = 0)
  -> O_acc tile;
```

Inputs from the kernel:

- `q/k/v_dram_block_window_tmp` — DRAM tile windows already pointing at the right
  Q‑block / KV‑head / page (built by the kernel with strides 0..3 we discussed earlier).
- `num_blocks_start` / `num_blocks` — the **per‑split KV page range** (`[start, end)`).
  When `num_splits=1`, this is `[0, total_num_kv_blocks)`.
- `block_tables_ptr + block_table_offset` — the page table row for this sequence.
- `mask` — the bottom‑right anchored causal mask (or no‑mask) constructed in the kernel.
- `scale_s` — `1/√d` (already includes the `log2(e)` factor for `exp2`‑based softmax).
- `smem_ptr` — workgroup LDS scratch.
- `*_element_func` — optional elementwise transforms (default `identity{}`).
  See §2.1 below for what each one does.
- `k_row_stride / v_row_stride` — used only by the **pointer‑rebase** path that avoids
  int32 overflow for large KV pools (>131K pages × `head_dim*nhead_kv` elements).

Returns the **normalized output tile** `O_acc` (shape `kBlockM × kHeadDimPadded`) for
this Q‑block. The kernel then writes it to global memory.

### 2.1 The six element‑wise hooks

The full `operator()` takes **six** elementwise function objects, one per logical
tensor in the algorithm:

| arg | applied to | when (pipeline location) |
|---|---|---|
| `q_element_func` | Q tile after DRAM load | once, just after `load_tile(q_dram_window)` (line 331) |
| `k_element_func` | K tile after DRAM/LDS load | hooked but currently `[[maybe_unused]]` — placeholder |
| `v_element_func` | V tile after DRAM/LDS load | hooked but currently `[[maybe_unused]]` — placeholder |
| `s_acc_element_func` | S = Q·Kᵀ result | hooked but currently `[[maybe_unused]]` |
| `p_compute_element_func` | P = exp2(...) before cast to fp16/bf16 | inside `fmha_alu1` (lines 679–680) |
| `o_acc_element_func` | O accumulator before return | once, after final normalize (line 1178) |

**What does the UA kernel actually pass?**

The kernel calls the **simpler 11‑arg overload** at the bottom of the file
(lines 1183–1222), which forwards `identity{}` for every element function:

```cpp
return operator()(q_dram_block_window_tmp,
                  identity{},                 // q_element_func
                  k_dram_block_window_tmp,
                  identity{},                 // k_element_func
                  v_dram_block_window_tmp,
                  identity{},                 // v_element_func
                  ...
                  identity{},                 // s_acc_element_func
                  identity{},                 // p_compute_element_func
                  identity{},                 // o_acc_element_func
                  mask, scale_s, smem_ptr, k_row_stride, v_row_stride);
```

So **for the standard CK‑UA path, every element function is the identity** — `q_element_func(x) == x`, `p_compute_element_func(x) == x`, etc. The
`tile_elementwise_in(q_element_func, origin_q)` on line 331 is therefore a no‑op
that the compiler folds away entirely.

**Why are the hooks there at all?**

For composability with **quant‑aware** variants. If you instantiate the pipeline
from a wrapper that needs to:

- dequantize fp8/int8/fp4 K/V on the fly (with per‑tile or per‑page scales),
- apply per‑head scale factors after `Q·Kᵀ`,
- bias / log‑softmax / softcap the scores before exponentiation,
- requantize O before writing back,

…you can pass non‑identity functions for the corresponding hook **without** touching
the pipeline body. The `tianxing/unified-attention-quantization` branch in CK uses
exactly this pattern for fp8 KV‑cache dequant. The 11‑arg overload covers the common
"no quant" case so callers don't need to spell six `identity{}`s.

**Concrete example.** A bf16 forward (your jsonl row 5) goes through the 11‑arg
overload → all `identity{}` → the `transformed_q = tile_elementwise_in(identity{}, origin_q)` call on line 331 emits zero instructions. So when you read the pipeline,
just mentally substitute `q_element_func(x) := x` everywhere unless you're
specifically reading a quant variant.

---

## 3. LDS sub‑windows (lines 209–234)

The pipeline carves the LDS slab into multiple typed views, **all at offset 0** (it
re‑uses the same physical LDS for different lifetimes):

```cpp
s_lds   // SaccDataType, shape kBlockM × kPageBlockSize  — softmax scores tile
p_lds   // PDataType,   shape kBlockM × kPageBlockSize  — softmax output tile (lives at smem + Policy::GetSmemSize)
o_lds   // PDataType,   shape kBlockM × kHeadDimPadded  — output staging
m_lds   // 1-D, kBlockM elements                         — row max for cross-warp sync
```

Why so many views over the same slab? Different stages of the algorithm need
different layouts and the pipeline overlaps them in time, never in space. CK‑tile
encodes that with multiple `tensor_view`s on the same pointer.

---

## 4. Policy queries: GEMM types + tile distributions (lines 239–278)

```cpp
constexpr auto gemm_0 = Policy::GetQKBlockGemm<Problem>();   // S = Q·Kᵀ
constexpr auto gemm_1 = Policy::GetPVBlockGemm<Problem>();   // O += P·V
```

These are fully specialized block‑GEMM objects encoding **MFMA shape, warp tiling,
and operand layouts**. The pipeline then calls them like generic functions.

Five register tile distributions are also instantiated here:

| distribution | tile | role |
|---|---|---|
| `MakeQRegTileDistribution` | Q in registers | persistent across all KV iters |
| `MakeKRegTileDistribution` | K (post‑LDS) | per inner iter, double‑buffered |
| `MakeVRegTileDistribution` | V (post‑LDS) | per inner iter, double‑buffered |
| `MakePRegTileDistribution` | P (softmax out) | feeds GEMM1 |
| `MakeKDramTileDistribution`, `MakeVDramTileDistribution` | DRAM staging | for `async_load_tile_raw` |

All are compile‑time constants — they erase to per‑lane register footprints with no
runtime cost.

---

## 5. Tile state (lines 280–338)

The pipeline manages an explicit **double‑buffer ping‑pong** for K and V LDS, plus a
ring of two `sp` slots so the softmax of step *j* and the GEMM of step *j+1* can
overlap:

```cpp
q_tile                              // distributed Q in registers (persistent)
union kv_tile_type { k_tile; v_tile } kv_tile;   // shared regs for K or V (alternating)
union sp_compute_type { sp_compute; p; } sp[2];  // S/P regs, ping-pong
o_acc                               // distributed O accumulator (persistent)
m, m_old, l                         // online-softmax row max / sum
sp_delta[2]                         // intermediate Δ = scale*(S − m)
```

Right after this:

```cpp
{ auto origin_q = load_tile(q_dram_window);
  auto transformed_q = tile_elementwise_in(q_element_func, origin_q);
  q_tile = transformed_q; }                         // load Q once
clear_tile(o_acc);
set_tile(m, bit_cast<float>(0xff7fffff));            // ≈ −∞ for first iter
clear_tile(l);
```

Q is loaded **once** and lives in registers for the whole KV walk. `m` is initialized
to *almost* `−∞` (`-FLT_MAX`) so the first running‑max step works.

---

## 6. Early exit + paged KV addressing (lines 342–414)

```cpp
if (FmhaMask::IsMasking && num_total_loop - num_blocks_start <= 0)
    return o_acc;       // no work after split-KV slicing
```

Then:

```cpp
block_table_offset += num_blocks_start;                        // skip earlier splits
kv_blk_idx_initial = block_tables_ptr_[block_table_offset];    // first physical page
```

There are two address modes (lines 372–398):

- **`use_ptr_rebase`** (large KV pools, hdim≤64) — `k_view.buf_.p_data_ = base + offset`
  in `long_index_t` so all subsequent address arithmetic stays in 64‑bit and avoids
  int32 overflow in `tensor_coordinate::get_offset()`. Window origin is reset to `{0, 0}`.
- **legacy `set_window_origin`** — original CK‑tile path, still used for hdim=128.

The first K and V DRAM windows are then constructed (lines 402–414) with the policy’s
DRAM tile distribution.

---

## 7. KV walk lambdas (lines 511–573)

Four small helpers do all the per‑iteration KV traffic:

```cpp
K_mem_load(buf)     // async DRAM → LDS K-buf, advance k_block_idx, look up next page
V_mem_load(buf)     // async DRAM → LDS V-buf, advance v_block_idx, look up next page
K_lds_load(buf)     // LDS K-buf → registers (kv_tile.k_tile)
V_lds_load(buf)     // LDS V-buf → registers (kv_tile.v_tile, transposed for PV)
```

`K_mem_load` and `V_mem_load` walk the page table:

```cpp
index_t page_blk_idx = block_tables_ptr_[block_table_offset
                                         + (block_idx / kv_page_size_in_blocks)];
// then either rebase the buffer pointer or set_window_origin to
//   page_blk_idx * PageSize + (block_idx % kv_page_size_in_blocks) * kPageBlockSize
```

So a **logical KV page** maps to a **physical page** via the block table; within a
physical page, the next `kPageBlockSize` slot is selected by modular arithmetic.
This is what makes KV "paged".

---

## 8. Online softmax + GEMM + mask helpers (lines 580–805)

The softmax is split across **three lambdas** so the compiler can schedule each piece
independently with the GEMMs and DRAM loads:

### `fmha_alu0(sp_reg_idx)` — running max + Δ
```text
m_latest = block_tile_reduce(sp.sp_compute, max)     // rowmax of new tile
m        = max(m_old, m_latest)                       // running max
sp_delta = fma(scale_s, sp.sp_compute, −scale_s · m)  // (S − m)*scale_s
```
On gfx950 with 32×32 MFMA it uses `permlane32_swap` instead of the generic block reduce
to halve the cross‑lane sync traffic.

> **Implementation note — the `fma` line is element-wise, not tensor-level.**
> `sp.sp_compute` is a `static_distributed_tensor<float, …>` (the gemm_0 output,
> register-resident across the warp). There is **no FMA overload** on
> `static_distributed_tensor`; the line above is shorthand for a compile-time
> double `sweep_tile_span` that walks every register element and emits one
> `v_fma_f32` per element:
>
> ```cpp
> for each (idx0, idx1) in sp.sp_compute's distributed spans:
>     sp_delta(idx0, idx1) = fma_impl_vsv(
>         sp.sp_compute(idx0, idx1),   //  S element  (VGPR, per-lane)
>         scale_s,                     //  scalar     (SGPR, wave-uniform)
>         -scale_s * m(idx0, idx1));   //  -scale_s·m (VGPR, per-lane)
> ```
>
> The spans are `constexpr`, so the whole nest unrolls into a flat sequence of
> FMAs on VGPRs (no real loop in the emitted ISA). `m` is itself a
> `static_distributed_tensor` reduced along K — one rowmax per thread (asserted
> by `m.thread_buf_.size() == 1`) — so `m(idx0, idx1)` is the same per-row scalar
> regardless of `idx1`, and the `-scale_s * m(i_j_idx)` factor is hoisted by CSE.
>
> `fma_impl_vsv` is a thin inline-asm wrapper around `v_fma_f32`. Its `vsv`
> suffix encodes operand placement: **v**ector, **s**calar, **v**ector —
> i.e. constraints `"v"`, `"s"`, `"v"`. Pinning `scale_s` to an SGPR saves one
> VGPR across all lanes and avoids a `v_mov_b32` pre-copy.
>
> **Why `−scale_s · m` and not just `−m`?** Because `m` is stored in **unscaled**
> units (rowmax of the raw `S = Q·Kᵀ` from gemm_0), so to compute
> `(S − m) · scale_s` in a single FMA both operands of the subtraction must be
> multiplied by `scale_s`: `fma(a=scale_s, b=S, c=−scale_s·m)`.

### `fmha_alu1(sp_reg_idx)` — exp + rowsum + rescale + cast
```text
sp.sp_compute = exp2(sp_delta)                        // P (unnormalized)
rowsum_p      = block_tile_reduce(sp.sp_compute, sum) // rowsum of P
l             = exp2(scale_s * (m_old - m)) * l + rowsum_p    // running sum
o_acc[0..R)   = o_acc[0..R) * o_acc_scale             // partial rescale (R = 6 regs)
sp.p          = pk_cvt(sp.sp_compute → fp16/bf16)     // P in compute dtype
```
Inline asm + chunked rescale (`fmha_alu_D_reg_cnt = 6`) so half the rescale happens
inside `fmha_alu1` and half inside `fmha_alu_D_upd`, hiding the latency.

### `fmha_alu_D_upd()` — finish the o_acc rescale
```text
o_acc_scale = exp2(scale_s * (m_old - m))
o_acc[R..end) *= o_acc_scale       // packed v_pk_mul_f32 inline asm
```

### `gemm(sp_reg_idx, gemm_idx)` and `cl_calc(...)`
Thin wrappers over `gemm_0` (S = Q·Kᵀ) and `gemm_1` (O += P·V). `cl_calc` additionally
folds in `fmha_alu0` for the GEMM1 case to keep the dependency chain tight.

### `fmha_mask(sp_reg_idx)`
Edge‑tile fast path:
```cpp
if (mask.IsEdgeTile(...))
    set_tile_if(sp.sp_compute, -inf, [&](idx) { return mask.IsOutOfBound(row, col); });
```
Inner tiles fully inside the allowed region skip the per‑pixel test entirely. Only
the diagonal/edge tile pays the masking cost.

---

## 8.1 `fmha_alu1` and `fmha_alu_D_upd` in detail

`fmha_alu0` is "pre‑exponentiation" (compute the new running max and the shift
`Δ = scale·(S − m)`). `fmha_alu1` and `fmha_alu_D_upd` together do everything **after**
the exponentiation: produce P, update the rowsum `l`, rescale `o_acc`, and cast P to
the GEMM1 input dtype. They're split into **two** lambdas precisely so the compiler
(and the warp‑specialization scheduler) can interleave the work with K/V LDS loads
and MFMA issue. Here's what each one actually does, line by line.

### `fmha_alu1(sp_reg_idx)` — five jobs in one lambda

**Reminder of preconditions** (set up earlier, by the **previous** iteration):

- `sp_delta(sp_reg_idx)` holds `scale_s · (S − m_new)` element‑wise, computed by
  `fmha_alu0` of the iter that produced this `S` tile.
- `m`, `m_old`, `l` are the running rowmax / previous rowmax / running rowsum from
  the same `fmha_alu0` step. They will **not** change again until the next
  `fmha_alu0`.
- `o_acc_scale` (a scalar) was set by the previous **`fmha_alu_D_upd`** to
  `exp2(scale_s · (m_old − m))` — the very scale we want to apply to the old `o_acc`
  to compensate for the new running max.

**Job 1: P = exp2(Δ)** — exponentiate in place

```cpp
sweep_tile_span(p_spans[0], [&](idx0) {
  sweep_tile_span(p_spans[1], [&](idx1) {
    sp.sp_compute(idx) = ck_tile::exp2(sp_delta(sp_reg_idx)(idx));
  });
});
```

This walks every register element of `sp_delta` and overwrites `sp.sp_compute` with
`exp2(Δ)`. After this point, `sp.sp_compute` holds `P_j` (the *unnormalized* softmax
of the j‑th tile, base‑2 form).

> **Why `exp2` and not `exp`?** `v_exp_f32` on AMD GCN is `exp2`. The kernel folds
> the natural‑log conversion into `scale_s` once at construction, so
> `scale_s_effective = (1/√d) · log₂(e)`. Then `exp2(scale_s · (S − m))` equals
> `exp((S − m) / √d)`. Saves one mul per element.

**Job 2: row‑sum of P and cross‑warp reduce**

```cpp
auto rowsum_p = block_tile_reduce(sp.sp_compute, dim=1, sum, init=0);
// then a cross-warp sync (permlane32_swap on gfx950 32x32, generic block_tile_reduce_sync otherwise)
```

`rowsum_p` is the per‑row sum of the new `P_j`. After the cross‑warp sync each thread
in the row holds the same final value (rowsum across the whole BLOCK_N axis = whole
KV tile). This is needed for the rescaled rowsum update next.

**Job 3: update the running rowsum `l`**

```cpp
sweep_tile_span(o_spans[0], [&](idx0) {
  const auto tmp = ck_tile::exp2(scale_s * (m_old[idx0] - m[idx0]));
  l(idx0) = detail::add_impl_vv(tmp * l[idx0], rowsum_p[idx0]);
});
```

Math:

$$
l_j \;=\; \underbrace{e^{\,\text{scale}\cdot(m_{j-1} - m_j)}}_{\text{old-l rescale factor}} \cdot l_{j-1} \;+\; \text{rowsum}(P_j)
$$

That `tmp` is the row‑wise rescale factor that compensates for the change in the
running max from `m_{j-1}` to `m_j`. Without it, summing rowsums computed with
different `m` values would be incoherent.

> **`detail::add_impl_vv` is inline‑asm.** The note in the code says: *"the compiler
> keeps moving the following instructions elsewhere because `l` is first consumed
> later"*. Re‑writing the final addition as inline asm creates an instruction‑order
> anchor so the dependent ops aren't sunk past the place they belong in the schedule.
> Same trick is used twice more in this lambda (for the partial `o_acc` rescale and
> the P→fp16/bf16 cast).

**Job 4: partial rescale of `o_acc` — first 6 registers**

```cpp
static_for<0, fmha_alu_D_reg_cnt, 1>{}([&](idx) {     // fmha_alu_D_reg_cnt = 6
    o_acc.thread_buf_[idx] = detail::mul_impl_vv(o_acc.thread_buf_[idx], o_acc_scale);
});
```

Multiplies the **first 6 registers** of `o_acc` by `o_acc_scale` (the scalar set by
the previous `fmha_alu_D_upd`). The remaining `o_acc.size() - 6` registers will be
rescaled later in `fmha_alu_D_upd`. Why split? See the next section.

**Job 5: cast P from fp32 → fp16/bf16, packed**

```cpp
static_for<0, sp.p.size(), 2>{}([&](idx) {           // process pairs of fp32
    float x = p_compute_element_func(sp.sp_compute[idx]);
    float y = p_compute_element_func(sp.sp_compute[idx+1]);
    if constexpr(is_fp16<PDataType>)
        sp.p[idx,idx+1] = detail::cvt_pk_fp16_f32(x, y);
    else
        sp.p[idx,idx+1] = cvt_pk_bf16_f32(x, y);
});
```

`sp_compute` is fp32 (matches `SaccDataType`). `gemm_1` wants P in `PDataType`
(fp16 or bf16) so that its A operand fits the MFMA shape. Each pair of fp32s is
converted with one packed instruction (`v_cvt_pk_fp16_f32` / `v_cvt_pk_bf16_f32`)
which produces both halves in a single VOP3P‑style instruction.

After this lambda, **`sp.p`** is the GEMM1‑ready P tile in fp16/bf16; **`sp.sp_compute`**
is now garbage (it was overwritten by `exp2` then read for the cast and is dead).
The `union sp_compute_type` reflects this — `sp_compute` and `p` share the same
register file.

### `fmha_alu_D_upd()` — finish the o_acc rescale

```cpp
o_acc_scale = exp2(scale_s * (m_old[0] - m[0]));     // scalar, shared across rows
fp32x2_t pk_o_acc_scale = {o_acc_scale, o_acc_scale};

// rescale o_acc[6..] using packed v_pk_mul_f32
static_for<6, o_acc.size(), 2>{}([&](idx) {
    fp32x2_t input = {o_acc[idx], o_acc[idx+1]};
    fp32x2_t output = detail::pk_mul_f32(input, pk_o_acc_scale);
    o_acc[idx]   = output.x;
    o_acc[idx+1] = output.y;
});
```

What it does:

1. **Compute `o_acc_scale`** as the same factor `exp2(scale_s · (m_old − m))` we
   already used inside `fmha_alu1` for `tmp` and the first 6 registers' rescale.
   It's set here *as a side‑effect for the next iteration's* `fmha_alu1` — i.e.
   "I'm writing the scale that the *next* `fmha_alu1` will read". This is what
   couples the two lambdas across iterations.
2. **Rescale `o_acc[6..end)`** with packed FP32 multiply (`v_pk_mul_f32`) — two
   floats per instruction, halving the issue count vs scalar `v_mul_f32`. The first
   6 elements were already done in `fmha_alu1` (job 4).

### Why split the rescale across two lambdas?

The constant `fmha_alu_D_reg_cnt = 6` (line 303) is the dial controlling the split.

```cpp
constexpr index_t fmha_alu_D_reg_cnt = 6; // threshold to decide how many fmha_alu_D_upd()
                                          // instructions should we move to fmha_alu1()
```

The motivation:

- `fmha_alu1` runs **between** the V LDS load and `gemm_1` (the PV matmul). The
  GPU is otherwise stalled waiting for the V tile's `lgkmcnt` to drain. Putting
  `o_acc` rescales here uses those idle cycles.
- `fmha_alu_D_upd` runs **between** `gemm_1` and the next `K_lds_load`. Different
  pocket of idle issue slots.
- Splitting the rescale `[0..6) | [6..end)` lets you fill **both** pockets without
  starving either. Six was tuned empirically — it's the number of `o_acc` registers
  that fit nicely in `fmha_alu1`'s slot before the cast‑to‑fp16 starts.

Concretely on MI300X with kBlockM=128, kHeadDim=64 and 4 warps, `o_acc.thread_buf_`
holds about 32 fp32 registers per thread (one per `(BLOCK_M × HEAD_DIM)` MFMA
tile element). 6 in `fmha_alu1`, 26 in `fmha_alu_D_upd` — roughly the sweet spot
for hiding both the `lgkmcnt` and the `vmcnt` waits.

### How `o_acc_scale` flows between iterations

```
ITER j-1                                  ITER j
─────────────────────                     ─────────────────────
fmha_alu0(j-1):                           gemm_1(P_{j-1} · V_{j-1})  ↓
  m_old = m;  m = max(m, rowmax(S_{j-1}))     using sp.p prepared by
  sp_delta = scale·(S_{j-1} - m)              fmha_alu1 of iter j-1

fmha_alu_D_upd():                         fmha_alu1(j-1):
  o_acc_scale = exp2(scale·(m_old - m))     // SAME m, m_old as fmha_alu0(j-1)
  o_acc[6..] *= o_acc_scale                 P_{j-1} = exp2(sp_delta)
                                            l = exp2(scale·(m_old - m)) * l + rowsum(P_{j-1})
                                            o_acc[0..6) *= o_acc_scale  // ← consumes scale set above
                                            cast P_{j-1} → fp16/bf16

(next gemm_0 runs here → S_j)             fmha_alu0(j):
                                            m_old <- m;  m <- max(...)
                                            ...
```

So `o_acc_scale` is a **single‑producer, single‑consumer scalar** that lives across
exactly one main‑loop iteration boundary. It's why the two lambdas have to be
called in the right order; you can't reorder `fmha_alu1` past the next `fmha_alu0`,
because that would clobber `m_old/m` and corrupt the scale.

### TL;DR for §8.1

| lambda | math it performs | side‑effects |
|---|---|---|
| `fmha_alu1` | `P = exp2(Δ)`,  `rowsum_p = ΣP`,  `l = exp2(scale·(m_old−m))·l + rowsum_p`,  `o_acc[0..6) *= o_acc_scale`,  cast P→fp16/bf16 | overwrites `sp.sp_compute` (now garbage), produces `sp.p` ready for `gemm_1` |
| `fmha_alu_D_upd` | `o_acc_scale = exp2(scale·(m_old−m))`,  `o_acc[6..] *= o_acc_scale` | sets the scalar that `fmha_alu1` of the **next** iteration will consume |

Together they implement the **online softmax rescale + P prep** half of the FlashAttn‑2 inner step, split across two pockets of idle issue cycles to maximize MFMA throughput.

---

## 9. CoreLoopScheduler iteration (lines 820–972)

Used by the **two‑warp‑group** path. Each call processes 2 KV pages (`pi=0,1`) split
across 4 explicit phases, with `__builtin_amdgcn_s_barrier()`/`s_waitcnt_*` between
them and `Scheduler::schedule(cl_p, phaseN)` injecting `sched_group_barrier`s for
deterministic instruction ordering.

```
phase0:  cl_calc(GEMM0)  +  fmha_alu1(prev)
phase1:  cl_load(K)      +  fmha_mask
phase2:  cl_calc(GEMM1)  +  fmha_alu_D_upd
phase3:  cl_load(V)
```

The two warp groups (`cl_p = 0` and `cl_p = 1`) run the same 4 phases but with
different LDS buffer indices — one group acts as **producer** (DRAM→LDS) while the
other acts as **consumer** (LDS→regs→MFMA), mirroring the warp‑specialization pattern
from Hopper FlashAttention‑3.

---

## 10. `fmha_post_process(d)` (lines 974–994)

After the main loop, one final `P·V` GEMM is needed for the **last** KV block whose
P was computed but whose V hadn't been multiplied yet. This lambda:

1. waits for the last V DRAM load,
2. loads V from LDS into registers,
3. calls `fmha_alu1` to finalize that last `sp` slot,
4. runs `gemm_1` for that last `P·V`.

Called once at end of `main_loops_exit` with `d=0` or `d=1` depending on whether
`num_total_loop` is even or odd.

---

## 10a. GEMM call‑site map — when does each matmul actually run?

The pipeline does only **two distinct matmuls** per inner step:

- **GEMM0**: `S = Q · Kᵀ` (uses `gemm_0` from policy)
- **GEMM1**: `O += P · V` (uses `gemm_1` from policy)

But they appear at **multiple call sites** because of the pipelined / double‑buffered
schedule. Here's the complete map, in execution order, for the easier‑to‑read
**single‑warp‑group** path. The two‑warp‑group path uses `cl_calc` from `core_loop`
but does the same two matmuls per KV page.

### The two helper lambdas

```cpp
auto gemm = [&](auto sp_reg_idx, auto gemm_idx) {        // line 700
    if constexpr (gemm_idx == 0)
        gemm_0(sp(sp_reg_idx).sp_compute, q_tile, kv_tile.k_tile);   // S = Q·Kᵀ
    else
        gemm_1(o_acc, sp(sp_reg_idx).p, kv_tile.v_tile);             // O += P·V
};

auto cl_calc = [&](auto sp_reg_idx, auto gemm_idx) {     // line 724
    /* same as gemm(...) but additionally calls fmha_alu0(other sp slot)
       at the end of the gemm_idx==1 case — bundles softmax with PV */
};
```

So **`gemm()` and `cl_calc()` are the only places `gemm_0` / `gemm_1` get called.**
Every matmul in the algorithm goes through one of them.

### Timeline (single‑warp‑group path)

Reading the source top‑to‑bottom from line 996:

```
PRE-STAGE (lines 996–1040)                                 KV block 0:
─────────────────────────────────────────────────
  K_mem_load(K0); … K_lds_load(K0)
  K_mem_load(K1); V_mem_load(V0)
  gemm(sp(0), GEMM0=0)    ←──── GEMM0 #1   S₀ = Q · K₀ᵀ      [line 1018]
  fmha_mask(0); fmha_alu0(0); fmha_alu_D_upd()
  ⇒ sp(0).sp_compute holds rescaled Δ for block 0
  ⇒ NO PV YET — V0 still loading

MAIN LOOP STEP 1 (lines 1054–1071)                         KV block 1:
─────────────────────────────────────────────────
  V_lds_load(V0); fmha_alu1(sp(0))   // finalize P₀ from sp(0)
  gemm(sp(0), GEMM1=1)   ←───── GEMM1 #1   O += P₀ · V₀      [line 1062]
  K_lds_load(K1)
  gemm(sp(1), GEMM0=0)   ←───── GEMM0 #2   S₁ = Q · K₁ᵀ      [line 1067]
  fmha_mask(1); fmha_alu0(1); fmha_alu_D_upd()

MAIN LOOP — even iter (lines 1075–1097)                    KV block i (i even, ≥2):
─────────────────────────────────────────────────
  V_lds_load(V buf 1); fmha_alu1(sp(1))
  gemm(sp(1), GEMM1=1)   ←───── GEMM1 #i   O += P_{i-1} · V_{i-1}  [line 1089]
  K_lds_load(K buf 0)
  gemm(sp(0), GEMM0=0)   ←───── GEMM0 #{i+1}   S_i = Q · K_iᵀ      [line 1094]

MAIN LOOP — odd iter (lines 1103–1123)                     KV block i (i odd, ≥3):
─────────────────────────────────────────────────
  V_lds_load(V buf 0); fmha_alu1(sp(0))
  gemm(sp(0), GEMM1=1)   ←───── GEMM1 #i                            [line 1115]
  K_lds_load(K buf 1)
  gemm(sp(1), GEMM0=0)   ←───── GEMM0 #{i+1}                        [line 1120]

POST-PROCESS (line 974)                                    last KV block:
─────────────────────────────────────────────────
  V_lds_load(V buf last); fmha_alu1(sp_last)
  gemm(sp_last, GEMM1=1)  ←──── GEMM1 #last   O += P_last · V_last  [line 993]
```

### The "lookback by 1" pattern

Notice the clean invariant: at any point in the main loop, the **GEMM0 for KV block
`i+1`** runs in the same iteration as the **GEMM1 for KV block `i`**. They are
*decoupled by one KV step*:

```
KV step:        0      1      2      3      4         …      N-1
                ┌──────┐
pre-stage:      │GEMM0₀│
                └──────┘
                       ┌──────┐ ┌──────┐
loop iter 1:           │GEMM1₀│ │GEMM0₁│
                       └──────┘ └──────┘
                                 ┌──────┐ ┌──────┐
loop iter 2:                     │GEMM1₁│ │GEMM0₂│
                                 └──────┘ └──────┘
                                                          ⋱
                                                    ┌──────┐
loop iter N-1:                                      │GEMM1_{N-2}│ │GEMM0_{N-1}│
                                                    └──────┘ └──────┘
                                                                    ┌──────┐
fmha_post_process:                                                  │GEMM1_{N-1}│
                                                                    └──────┘
```

So:
- **#GEMM0 calls = N** (one per KV block, `N = num_total_loop`)
- **#GEMM1 calls = N** (one per KV block, but offset by 1 in time)
- The first GEMM0 happens in **pre‑stage** (no PV yet — V₀ hasn't arrived)
- The last GEMM1 happens in **post‑process** (no further QK to pair it with)

This is the standard FlashAttention‑2 pipeline trick: **GEMM0(i+1) overlaps with
GEMM1(i) and softmax(i)**. Hardware-wise both GEMMs are MFMA pipelines on different
register operands, so the issue queue can keep them both busy simultaneously while
DRAM prefetches K_{i+2} and V_{i+1}.

### Two-warp-group path (cross‑reference)

For `NumWarpGroups == 2` the same matmuls happen, but inside the 4‑phase
`core_loop::iteration(pi)` (lines 829–969):

```
phase0: cl_calc(sp[1-pi], GEMM0)   // GEMM0 → sp[1-pi]
        fmha_alu1(sp[pi])           // finalize P from prev sp[pi]
phase1: cl_load(K) + fmha_mask
phase2: cl_calc(sp[pi], GEMM1)     // GEMM1 from sp[pi]
        fmha_alu_D_upd
phase3: cl_load(V)
```

Each `iteration(pi)` thus runs **one GEMM0 + one GEMM1**, just like one body of the
single‑warp‑group main loop. `core_loop` calls `iteration(0); iteration(1);`, so a
single `core_loop` invocation processes 2 KV pages and issues 2 GEMM0s + 2 GEMM1s.

### Summary table

| where | source line | GEMM | what it computes |
|---|---|---|---|
| pre‑stage | 1018 | GEMM0 | S₀ = Q · K₀ᵀ |
| main loop step 1 | 1062 | GEMM1 | O += P₀ · V₀ |
| main loop step 1 | 1067 | GEMM0 | S₁ = Q · K₁ᵀ |
| main loop even iter | 1089 | GEMM1 | O += P_{i−1} · V_{i−1} |
| main loop even iter | 1094 | GEMM0 | S_i  = Q · K_iᵀ |
| main loop odd iter | 1115 | GEMM1 | O += P_{i−1} · V_{i−1} |
| main loop odd iter | 1120 | GEMM0 | S_i  = Q · K_iᵀ |
| `core_loop` (2‑wg) | 863 / 936 | GEMM0 (via `cl_calc`) | S = Q · Kᵀ for next block |
| `core_loop` (2‑wg) | 887 / 963 | GEMM1 (via `cl_calc`) | O += P · V for current block |
| `fmha_post_process` | 993 | GEMM1 | O += P_last · V_last |

---

## 11. Pre‑stage (lines 996–1040)

The 4‑stage pipeline needs at least 1 K and 1 V already in flight before the main loop
can start. Pre‑stage:

```text
(1) K_mem_load(K0)   →   wait   →   K_lds_load(K0)
(2) prefetch K1 (if needed)  +  V_mem_load(V0)
(3) gemm_0(Q, K0)  +  fmha_mask  +  fmha_alu0  +  fmha_alu_D_upd
(4) early-exit if num_total_loop == 1
(5) prefetch K2 (if needed)
```

By the end of pre‑stage, `sp(0)` holds the (rescaled, scaled) score tile for KV block
0, V0 is loading into LDS, K1 is in LDS, and K2 is loading. The main loop picks up
from there.

---

## 12. Main loop (lines 1042–1148) — two implementations

### `NumWarpGroups == 1` — single warp group, serial pipeline

A hand‑unrolled `while(i_total_loops < num_total_loop)` walking pairs of KV pages,
ping‑ponging `(K buf 0, V buf 0)` ↔ `(K buf 1, V buf 1)` and `sp(0)` ↔ `sp(1)`:

```text
even step:                                  odd step:
  prefetch K_next → K buf 1                   prefetch K_next → K buf 0
  prefetch V_next → V buf 0                   prefetch V_next → V buf 1
  V_lds_load(buf 1)                           V_lds_load(buf 0)
  fmha_alu1(sp(1))                            fmha_alu1(sp(0))
  gemm_1: P(1) * V                            gemm_1: P(0) * V
  K_lds_load(buf 0)                           K_lds_load(buf 1)
  gemm_0: Q * K  → sp(0)                      gemm_0: Q * K  → sp(1)
  fmha_mask + alu0 + alu_D_upd                fmha_mask + alu0 + alu_D_upd
  i_total_loops++                              i_total_loops++
```

Async DRAM prefetch overlaps with all the compute below it.

### `NumWarpGroups == 2` — two warp groups, warp specialization

```cpp
if (warp_group_id == 0) { ... while (core_loop(0)); }
else                    { ... while (core_loop(1)); }
```

Each group runs its own variant of the 4‑phase scheduler. They cooperate via
`s_barrier` and `sched_group_barrier` — one fetches from DRAM, the other does MFMA,
they stay in lockstep.

---

## 13. `label_main_loops_exit:` and final normalization (lines 1149–1180)

```cpp
label_main_loops_exit:
    if (num_total_loop % 2)  fmha_post_process(number<1>{});
    else                     fmha_post_process(number<0>{});

    // O := O / l   (broadcast row-wise)
    sweep_tile_span(o_spans[0], [&](idx0) {
        const auto inv_l = (l[idx0] == 0.f) ? 0.f : 1.f / l[idx0];
        sweep_tile_span(o_spans[1], [&](idx1) {
            o_acc(idx0,idx1) *= inv_l;
        });
    });

    o_acc = tile_elementwise_in(o_acc_element_func, o_acc);
    return o_acc;
```

`l` is the running rowsum from online softmax. The final `1/l` divide turns the
unnormalized accumulator `Σ exp(S − m) · V` into the actual softmax output
`Σ softmax(S) · V`. The `l == 0` guard handles fully‑masked rows (no valid KV
positions for that q‑row) — output zero instead of NaN.

---

## 14. Online softmax math, in one frame

Across the KV walk the pipeline maintains three running quantities per row:

| symbol | math |
|---|---|
| `m` | running max of `S = scale_s · Q·Kᵀ` so far |
| `l` | running rowsum of `exp2(S − m)` so far |
| `o_acc` | running `Σ exp2(S − m) · V`  (rescaled on each `m` update) |

For a new tile `S_j`:

1. `m_new = max(m_old, rowmax(S_j))`
2. `α = exp2(scale_s · (m_old − m_new))`        ← rescale factor
3. `o_acc *= α`                                 ← rescale old contribution
4. `l *= α`
5. `P_j = exp2(scale_s · (S_j − m_new))`
6. `l += rowsum(P_j)`
7. `o_acc += P_j · V_j`

That's exactly what `fmha_alu0` + `fmha_alu1` + `fmha_alu_D_upd` + `gemm_1` realize,
but spread across the pipeline phases for instruction‑level scheduling.

At the end: `O = o_acc / l`.

---

## 15. Cheat sheet — symbols you'll keep seeing

| name | type | meaning |
|---|---|---|
| `sp[0..1]` | `union { sp_compute; p }` | ping-pong S→P registers |
| `m` / `m_old` | distributed tensor | running rowmax / previous rowmax |
| `l` | distributed tensor | running rowsum |
| `o_acc` | distributed tensor | unnormalized output accumulator |
| `q_tile` | distributed tensor | persistent Q in registers |
| `kv_tile` | `union { k_tile; v_tile }` | alternating K/V in registers |
| `sp_delta` | `[2]` distributed tensors | `scale_s·(S − m)` for `exp2` |
| `K/V buf 0..1` | LDS | double-buffered K/V tiles |
| `i_total_loops` | scalar | KV-block counter |
| `kv_page_size_in_blocks` | scalar | how many `kPageBlockSize` tiles fit in one paged-KV page |
| `K_mem_su_ld_insts` / `V_mem_su_ld_insts` | constexpr | # of `buffer_load` insts per K/V tile (= 1 for 32×128 typical) |
| `kBlockM` vs `kBlockQ` | constexpr | flattened (q,head) tile vs original q tile |

---

## 16. Reading order if you have 30 minutes

1. **Lines 38–104** — get oriented (template, types, `GetSmemSize`).
2. **Lines 173–334** — `operator()` setup: tile windows, GEMMs, Q load.
3. **Lines 511–573** — paged KV walk (`K_mem_load`, `V_mem_load`, …).
4. **Lines 580–783** — online softmax: `fmha_alu0`, `fmha_alu1`, `fmha_alu_D_upd`,
   `fmha_mask`.
5. **Lines 996–1040 and 1042–1148** — pre‑stage + main loop body (start with the
   `NumWarpGroups == 1` branch, it’s easier to read).
6. **Lines 1149–1180** — finalize.

Once those click, the 4‑phase `core_loop` (lines 820–972) is just the same algorithm
re‑expressed as warp‑specialized phases for the 2‑warp‑group path.

---

## 17. What the pipeline is NOT doing

To avoid confusion, here are things the kernel handles, **not** the pipeline:

- Picking `(seq_idx, kv_head_idx, q_block_local_idx)` from `blockIdx`.
- Computing `context_len`, `_max_seq_prefix_len`, `total_num_kv_blocks`.
- Slicing the KV page range across `num_splits` (`num_blocks_start / num_blocks` are
  passed in already‑sliced).
- Building the `FmhaMask` (the pipeline only consumes it).
- Allocating `smem_ptr` (passed in).
- The split‑KV combine pass (does **not exist** today; only the partition logic is in
  the kernel — the combine epilogue was the "WIP epilogue pending" piece).

---

## 18. Cross‑references to other docs

- `unified_attention_params.md` — the math + parameter glossary
  (`BLOCK_M`, `BLOCK_Q`, `TILE_SIZE`, GQA fold, etc.).
- `unified_attention_pipeline_default_policy.hpp` — the *strategy* knobs the pipeline
  consults at compile time (alignments, GEMM types, LDS layouts, register
  distributions).
- `unified_attention_kernel.hpp` — the orchestration layer that calls this pipeline.
