# `UnifiedAttentionPipeline` vs `BlockFmhaFwdV3Pipeline`

This doc compares the **internals** of CK Tile's two side-by-side forward
pipeline classes:

| Pipeline class | File |
|---|---|
| `BlockFmhaFwdV3Pipeline` (v3 forward, "asm-style") | [`block_fmha_fwd_v3_pipeline.hpp`](../3rdparty/composable_kernel/include/ck_tile/ops/fmha/pipeline/block_fmha_fwd_v3_pipeline.hpp) |
| `UnifiedAttentionPipeline` (UA) | [`unified_attention_pipeline.hpp`](../3rdparty/composable_kernel/include/ck_tile/ops/unified_attention/pipeline/unified_attention_pipeline.hpp) |

The companion docs operate at different layers and link in here:

- [`attention_pipelines.md`](attention_pipelines.md) — external grid + I/O for
  all six aiter attention backends.
- [`unified_attention_vs_fmha.md`](unified_attention_vs_fmha.md) — Python-level
  API surface comparison of `unified_attention_fwd` vs every
  `aiter.ops.mha.*` entry point.

The intended audience here is someone reading the two `.hpp` files side by
side and asking *"what did UA actually change?"*. The short answer: **UA is
the v3 forward pipeline retargeted at vLLM-style paged GQA, with most of
v3's `AttentionVariant` machinery stripped away.**

---

## 1. Lineage evidence

Three independent signals show UA was forked from v3, not written
from scratch:

**(a)** UA's pipeline header `#include`s the v3 pipeline header directly:

```6:9:aiter/3rdparty/composable_kernel/include/ck_tile/ops/unified_attention/pipeline/unified_attention_pipeline.hpp
#include "ck_tile/core.hpp"
#include "ck_tile/ops/unified_attention/pipeline/unified_attention_pipeline_default_policy.hpp"
#include "ck_tile/ops/fmha/pipeline/block_fmha_fwd_v3_pipeline.hpp"
#include "ck_tile/ops/reduce/block/block_reduce.hpp"
```

**(b)** The preamble macros are byte-for-byte identical between the two
files, including the `[POYENC]` author tag (a personal signature) and the
`ADD_SBARRIER_FOR_PHASE0`/`DEBUG_STMTS`/packed-fp32 guards:

```10:34:aiter/3rdparty/composable_kernel/include/ck_tile/ops/unified_attention/pipeline/unified_attention_pipeline.hpp
#define ENABLE_ASM_MARKER 1
#if ENABLE_ASM_MARKER
#define ASM_MARKER(marker)               \
    __builtin_amdgcn_sched_barrier(0);   \
    asm volatile("; [POYENC] " #marker); \
    __builtin_amdgcn_sched_barrier(0);
#endif
```

```12:36:aiter/3rdparty/composable_kernel/include/ck_tile/ops/fmha/pipeline/block_fmha_fwd_v3_pipeline.hpp
#define ENABLE_ASM_MARKER 1
#if ENABLE_ASM_MARKER
#define ASM_MARKER(marker)               \
    __builtin_amdgcn_sched_barrier(0);   \
    asm volatile("; [POYENC] " #marker); \
    __builtin_amdgcn_sched_barrier(0);
```

**(c)** UA's two-warp-group `core_loop` reuses the **exact same
`CoreLoopScheduler`** that's defined in the v3 header (the type is forward-
declared at `block_fmha_fwd_v3_pipeline.hpp:40-41` and specialized at
`:43-114`/`:117-188`). UA picks it up via the include in (a):

```820:867:aiter/3rdparty/composable_kernel/include/ck_tile/ops/unified_attention/pipeline/unified_attention_pipeline.hpp
        auto core_loop = [&](auto cl_p) {
            ...
            using Scheduler = CoreLoopScheduler<Problem, FmhaMask::IsMasking>;
            ...
                    cl_calc(xdl_SP_p01_reg_idx, gemm0);
                    fmha_alu1(xdl_SP_p23_reg_idx);

                    Scheduler::schedule(cl_p, number<0>{});
```

So the inline-asm scheduling barriers (which are what makes v3 "v3") are
the same instructions in both pipelines.

---

## 2. `operator()` signature diff

Both pipelines expose a "full" overload (with element-function hooks for
Q/K/V/S/P/O/LSE) and a "convenience" overload (forwards `identity{}` for
all of them). The full overloads tell the whole story:

### v3 forward — what the kernel hands the pipeline

```385:414:aiter/3rdparty/composable_kernel/include/ck_tile/ops/fmha/pipeline/block_fmha_fwd_v3_pipeline.hpp
    CK_TILE_DEVICE auto operator()(const QDramBlockWindowTmp& q_dram_block_window_tmp, // M0*K0 tile
                                   const QElementFunction& q_element_func,
                                   const KDramBlockWindowTmp& k_dram_block_window_tmp, // N0*K0 tile
                                   [[maybe_unused]] const KElementFunction& k_element_func,
                                   const VDramBlockWindowTmp& v_dram_block_window_tmp, // N1*K1 tile
                                   [[maybe_unused]] const VElementFunction& v_element_func,
                                   LSEDramBlockWindowTmp& lse_dram_window_tmp, // M0*1 tile
                                   const LSEElementFunction& lse_element_func,
                                   [[maybe_unused]] const SAccElementFunction& s_acc_element_func,
                                   const PComputeElementFunction& p_compute_element_func,
                                   const OAccElementFunction& o_acc_element_func,
                                   FmhaMask mask,
```

(continues with `scale_s`, `variant`, `variant_params`, `block_indices`,
`smem_ptr`)

### UA — same idea, but paged

```173:192:aiter/3rdparty/composable_kernel/include/ck_tile/ops/unified_attention/pipeline/unified_attention_pipeline.hpp
    CK_TILE_DEVICE auto operator()(
        const QDramBlockWindowTmp& q_dram_block_window_tmp, // kBlockM * kHeadDimPadded tile
        const QElementFunction& q_element_func,
        const KDramBlockWindowTmp& k_dram_block_window_tmp, // kPageBlockSize * kHeadDimPadded tile
        [[maybe_unused]] const KElementFunction& k_element_func,
        const VDramBlockWindowTmp& v_dram_block_window_tmp, // kHeadDimPadded * kPageBlockSize tile
        [[maybe_unused]] const VElementFunction& v_element_func,
        const index_t num_blocks,
        const index_t num_blocks_start,
        const void* block_tables_ptr,
        index_t block_table_offset,
        const index_t kv_page_size_in_blocks,
        [[maybe_unused]] const SAccElementFunction& s_acc_element_func,
        const PComputeElementFunction& p_compute_element_func,
        const OAccElementFunction& o_acc_element_func,
        FmhaMask mask,
        float scale_s,
        void* smem_ptr,
        long_index_t k_row_stride = 0,
        long_index_t v_row_stride = 0) const
```

### Diff

| Parameter | v3 | UA | Why |
|---|---|---|---|
| `LSEDramBlockWindow` + `lse_element_func` | yes | **dropped** | UA never returns LSE (`kStoreLSE` is dead code in v3 too — its `static_assert` requires `!kStoreLSE`, see below) |
| `AttentionVariant variant` + `AttentionVariantParams` | yes | **dropped** | This is where v3 hangs softcap, sliding-window, sinks, FP8 descales. UA only does causal/no-mask. |
| `BlockIndices block_indices` | yes | **dropped** | v3 needs `(batch, qo_head, kv_head)` to feed into `variant.LogitsMask(...)`. UA's mask is data-independent. |
| `num_blocks`, `num_blocks_start` | no | **added** | UA's outer loop counts in **pages**, not in `kN0` KV-tiles, and `num_blocks_start` lets the kernel express "skip the first N pages of this seq" (used for the splitkv-style KV-head sharding). |
| `block_tables_ptr`, `block_table_offset`, `kv_page_size_in_blocks` | no | **added** | The whole point: v3 reads a contiguous K/V tensor view, UA chases pointers through a page table on every iteration. |
| `k_row_stride`, `v_row_stride` (default 0) | no | **added** | Pointer-rebase escape hatch for >131K-page KV pools where `int32` `tensor_coordinate` offsets would overflow at `hdim<=64`; see `unified_attention_pipeline.hpp:366-414`. v3 doesn't need this because it never has that many K rows in one window. |

The v3 KV addressing is implicit in the `KDramBlockWindowTmp` itself —
the **kernel** picks `seqlen_k_start` and the pipeline calls
`move_tile_window` to advance. UA's KV addressing is **explicit in the
parameter list**.

---

## 3. The core loop, in Python pseudocode

Both pipelines follow the same five-phase choreography:

> **load Q -> prologue (K0, V0, first GEMM0) -> core loop (alternating
> GEMM0/GEMM1 with double-buffered KV prefetch) -> post-process (final
> PV) -> epilogue (normalize O)**

What changes is **how each iteration finds its K/V**.

### v3 forward

```python
def block_fmha_fwd_v3(q_win, k_win, v_win, lse_win,                # 385..414
                     mask, scale_s, variant, variant_params,
                     block_indices, smem):
    # ---- setup ----
    q_tile = q_element_func(load_tile(q_win))                       # 550..555
    o_acc, m, l = zeros(kM0, hdim), -inf, zeros(kM0)                # 557..559

    # ---- contiguous KV range comes from the mask, NOT a page table ----
    k_start, k_end = mask.GetTileRangeAlongX(q_origin, kM0, kN0)    # 562..563
    num_total_loop = ceil_div(k_end - k_start, kN0)
    kv_token_start = k_start

    if mask.IsMasking and num_total_loop <= 0:                      # 568..587
        if kStoreLSE: store_tile(lse_win, full(-inf))
        return o_acc

    # advance the K/V dram windows to seqlen_k_start (one move, then we'll
    # walk forward by kN0 each iteration via move_tile_window inside K_mem_load)
    k_dram_window = make_window(k_win, origin=(k_start, 0))         # 589..594
    v_dram_window = make_window(v_win, origin=(k_start, 0))         # 596..601

    # ---- prologue: K0, GEMM0(Q, K0), mask, alu0 ----
    K_mem_load(0); s_waitcnt(); K_lds_load(0)                       # 1177..1185
    if num_total_loop > 1: K_mem_load(1)                            # 1188..1192
    V_mem_load(0)
    gemm_0(sp[0])                                                   # 1195
    fmha_logits_trans(0)        # softcap/quant lives here          # 1196
    fmha_mask(0)                # variant.LogitsMask(...) at edges  # 1197
    fmha_alu0(0); fmha_alu_D_upd()                                  # 1199..1200
    kv_token_start += kN0                                           # 1202

    # ---- core loop: 2 warp-group cooperative pipeline ----
    if num_total_loop > 1:                                          # 1220..1238
        if warp_group_id == 0: while core_loop(0): pass             # uses CoreLoopScheduler
        else:                  while core_loop(1): pass

    # ---- final PV that the loop body deferred ----
    fmha_post_process(num_total_loop % 2)                           # 1241..1248

    # ---- epilogue ----
    if kStoreLSE:                                                   # 1251..1262
        store_tile(lse_win, m / ln(2) + log(l))   # *** v3 only ***
    o_acc *= where(l == 0, 0, 1/l)                                  # 1267..1281
    return o_acc_element_func(o_acc)
```

### UA — same skeleton, paged inside

```python
def unified_attention(q_win, k_win, v_win,                          # 173..192
                      num_blocks, num_blocks_start,
                      block_tables_ptr, block_table_offset,
                      kv_page_size_in_blocks, mask, scale_s, smem,
                      k_row_stride=0, v_row_stride=0):
    # ---- setup (identical to v3) ----
    q_tile = q_element_func(load_tile(q_win))                       # 329..334
    o_acc, m, l = zeros(kBlockM, hdim), -inf, zeros(kBlockM)        # 336..338

    # ---- early-out: this WG's slice of the page table is empty ----
    num_total_loop = num_blocks                                     # 342
    if mask.IsMasking and num_total_loop - num_blocks_start <= 0:   # 347..355
        return o_acc                                # *** no LSE branch ***

    # ---- paged KV addressing ----
    PageSize = kv_page_size_in_blocks * kPageBlockSize              # 359
    block_table_offset += num_blocks_start                          # 363
    phys_page = block_tables_ptr[block_table_offset + 0]            # 364

    # Optional pointer rebase to dodge int32 offset overflow on huge
    # KV pools (only for hdim<=64; see 366..414).
    if k_row_stride > 0 and v_row_stride > 0 and kHeadDim <= 64:
        rebase_kv_views(...)                                        # 372..414

    # ---- prologue: identical ASM_MARKER blocks as v3 ----
    K_mem_load(0); s_waitcnt(); K_lds_load(0)                       # 1000..1008
    if num_total_loop > 1: K_mem_load(1)                            # 1011..1014
    V_mem_load(0)
    gemm_0(sp[0])                                                   # 1018
    # NOTE: no fmha_logits_trans here -- variant pipeline absent
    fmha_mask(0)                                                    # 1020
    fmha_alu0(0); fmha_alu_D_upd()                                  # 1022..1023
    i_total_loops += 1                                              # 1025
    if i_total_loops >= num_total_loop: goto epilogue               # 1026..1029

    # ---- core loop: TWO specializations ----
    if NumWarpGroups == 1:                          # *** UA-only ***
        # Single-warp-group serial pipeline used by UA's decode tiers
        # (kBlockM in {16, 32, 64}). Hand-unrolled K/V double buffering.
        while i_total_loops < num_total_loop:                       # 1042..1126
            for ph in (even_step, odd_step):
                # Each step: prefetch next K & V via block_tables,
                #            run V_lds_load, fmha_alu1, gemm_1 (PV),
                #            then K_lds_load, gemm_0 (QK),
                #            fmha_mask, fmha_alu0, fmha_alu_D_upd.
                # KV step is exactly kPageBlockSize tokens / iteration.
                phys_page = block_tables_ptr[block_table_offset + i_total_loops]
                ...
                i_total_loops += 1
    else:                                           # NumWarpGroups == 2
        # Identical to v3's 2-WG pipeline.
        if warp_group_id == 0: while core_loop(cl_p=0): pass        # 1130..1138
        else:                  while core_loop(cl_p=1): pass        # 1140..1146
        # core_loop(...) iteration calls
        #   Scheduler::schedule(cl_p, phase) where Scheduler ==
        #   CoreLoopScheduler<Problem, IsMasking> from v3's header.

    # ---- final PV that the loop body deferred (same as v3) ----
    fmha_post_process(num_total_loop % 2)                           # 1149..1157

    # ---- epilogue: normalize O. NO lse store. ----
    o_acc *= where(mask.IsMasking and l == 0, 0, 1/l)               # 1160..1176
    return o_acc_element_func(o_acc)                                # 1178..1180
```

### What you'd see in a diff

If you actually `diff`'d the two `operator()` bodies after macro
expansion, the noise breaks down like this:

- **First ~30 lines (Q load, smem layout, double-buffered K/V LDS
  windows, online-softmax state init):** essentially identical token
  for token. Both call `Policy::GetSmemSizeKV<Problem>()` with the same
  layout, both clear `o_acc`, set `m = bit_cast<float>(0xff7fffff)`,
  clear `l`.
- **KV addressing block (~50 lines):** completely different. v3 derives
  `[seqlen_k_start, seqlen_k_end)` from the mask and walks one
  contiguous K/V window. UA loads a page-table entry per iteration and
  optionally rebases the K/V `tensor_view` pointers to avoid `int32`
  offset overflow.
- **Prologue (`pre-stage` ASM_MARKER block):** identical line-for-line
  *except* v3 has an extra `fmha_logits_trans(0)` call between
  `gemm_0` and `fmha_mask` (`block_fmha_fwd_v3_pipeline.hpp:1196`) —
  that's where v3 applies softcap and Q-quantization-scale variants.
  UA has no such transform because it has no `AttentionVariant`.
- **Core loop:** v3 has only the 2-WG branch. UA has a *new*
  `NumWarpGroups == 1` branch (`unified_attention_pipeline.hpp:1042-1126`)
  for its decode tiers, plus the same 2-WG branch using the same
  `CoreLoopScheduler` from v3's header.
- **Epilogue:** v3 conditionally stores LSE
  (`block_fmha_fwd_v3_pipeline.hpp:1250-1262`); UA simply doesn't.
  Both then divide `o_acc` by `l` with the same `where(l == 0, 0, 1/l)`
  mask-aware fallback.

---

## 4. What UA *adds* on top of v3

1. **Paged KV addressing.** Each iteration of UA's core loop computes
   `phys_page = block_tables_ptr[block_table_offset + i]` and rebases
   the K/V tile-window into the page pool
   (`unified_attention_pipeline.hpp:358-414`, `:523-565`). v3 never sees
   a page table.

2. **KV step = `kPageBlockSize` (not `kN0`).** UA fuses tiling and
   paging by making one core-loop iteration consume exactly one KV page
   (32 or 64 tokens). `fmha_mask` consequently uses
   `i_total_loops * kPageBlockSize` for its column-position math
   instead of v3's `kv_token_start`.

3. **1-warp-group decode specialization.** UA has a complete second
   pipeline at `unified_attention_pipeline.hpp:1042-1126` for
   `NumWarpGroups == 1`. This is what runs the four small tiers
   (`Tiny`/`BS32`/`Small`, `kBlockM` ∈ {16, 32, 64}) used at the 2D
   `(num_kv_heads, num_seqs)` decode grid. v3 hard-asserts
   `NumWarpGroups == 2` (`block_fmha_fwd_v3_pipeline.hpp:611-612`) and
   has no equivalent.

4. **GQA tile-merge into M.** `kBlockQ = kBlockM / num_queries_per_kv`
   is a UA-pipeline static constant
   (`unified_attention_pipeline.hpp:60-61`) and is asserted at runtime
   in the kernel
   (`unified_attention_kernel.hpp:245`). The mask's edge-tile
   granularity becomes `(kBlockQ, kPageBlockSize)` instead of v3's
   `(kM0, kN0)`. Effect: a single QK GEMM produces scores for all
   `num_queries_per_kv` Q-heads of one KV-head simultaneously.

5. **2D `(num_kv_heads, num_seqs)` decode grid.** Lives in the **kernel**
   (`unified_attention_kernel.hpp:233-263`), not the pipeline. The
   pipeline just sees a per-WG slice of the page table via
   `num_blocks_start` / `block_table_offset`. v3's varlen kernel uses a
   1D grid + binary search.

---

## 5. What UA *strips* away

Each row here corresponds to either a v3 `Problem::*` flag that UA
silently never sets, or a feature whose `static_assert` in v3 also
forbids it.

| v3 feature | Status in UA | Source |
|---|---|---|
| `kStoreLSE` (return per-row log-sum-exp) | not exposed | v3: `static_assert(!kStoreLSE, ...)` at `block_fmha_fwd_v3_pipeline.hpp:293-296`; UA's `operator()` simply never has an LSE epilogue branch. |
| `kHasDropout` | not exposed | same v3 `static_assert`. |
| `BiasEnum != NO_BIAS` (additive bias tensor) | not exposed | same v3 `static_assert`. |
| `QScaleEnum != NO_SCALE` (FP8 Q-quant variant) | not exposed | same v3 `static_assert`. UA's `scale_s/scale/scale_k/scale_v/scale_out` runtime knobs are unrelated -- they're plain multiplicative scalars, no per-call quant variant. |
| `kHasLogitsSoftCap` (`tanh(x/cap) * cap`) | absent | v3 routes this through `variant.LogitsTransform(...)` and `fmha_logits_trans(...)` at `block_fmha_fwd_v3_pipeline.hpp:1196`. UA has no `variant`, no `fmha_logits_trans`. |
| Sliding-window range, sinks | absent | v3 reads them from `mask.GetTileRangeAlongX(...)` and `variant.LogitsMask(...)`. UA's `FmhaMask` is causal-or-none only. |

The selector at
[`aiter/ops/triton/attention/unified_attention.py`](../aiter/ops/triton/attention/unified_attention.py)
already gates UA off whenever any of these features is requested:
`window_size != (-1, -1)`, `block_size < 32`, `head_size` and
`num_queries_per_kv` outside the compiled set, etc.

---

## 6. Shared infrastructure

Both pipelines share the same online-softmax decomposition into three
helpers (defined as private lambdas, but the names are identical
between the two files):

| Helper | Does |
|---|---|
| `fmha_alu0(buf)` | Row-wise `m_new = max(m, max(S))`, computes `sp_delta = scale_s * (S - m_new)` (or softcap on v3 only). |
| `fmha_alu1(buf)` | `P = exp2(sp_delta)`, `l_new = l*exp2(scale_s*(m_old-m)) + rowsum(P)`, rescales the first `fmha_alu_D_reg_cnt` registers of `o_acc`, casts `P` for the next PV GEMM. |
| `fmha_alu_D_upd()` | Scales the *remaining* `o_acc` registers by `exp2(scale_s*(m_old-m))` using `pk_mul_f32`. The split into "in-alu1" + "in-D_upd" is what gives the v3 family its name; both files use `constexpr index_t fmha_alu_D_reg_cnt = 6`. |

And both then call the same `CoreLoopScheduler::schedule(cl_p, phase)`
in their 2-WG core loops to insert the `sched_group_barrier(0x008,
...)` MFMA / `0x002` VALU / `0x004` SALU / `0x200` TRANS hints that
order instruction classes inside each phase. That scheduler **is
defined once**, in v3's header, and pulled in by UA via the `#include`.

---

## 7. Why the fork makes sense

- **vLLM uses paged KV everywhere.** Bolting paging behind v3's
  `KDramBlockWindow` would have meant rewriting the dram-window
  abstraction. Forking the pipeline and putting the page table in the
  signature is a smaller change.
- **`AttentionVariant` is not free.** Each variant call site
  (`LogitsTransform`, `QueryTransform`, `LogitsMask`) materializes
  through inlines that consume registers and instruction cache.
  Stripping them lets UA fit a wider `kBlockM` (256 in the Large tier)
  on the same WGP.
- **GQA-merge wants a different M decomposition.** v3 indexes M by Q
  *tokens*; UA wants M to mix Q-tokens with the
  `num_queries_per_kv` head dimension so a single QK GEMM produces all
  Q-head scores for one KV-head. That decomposition propagates into
  the LDS distribution policies and the mask edge-tile granularity,
  which are awkward to express through v3's `Problem` knobs alone.

The trade-off is exactly what the
[`unified_attention_vs_fmha.md`](unified_attention_vs_fmha.md) feature
matrix shows: UA wins paged GQA at the two compiled shapes (d=64 GQA-8
and d=128 MHA), and gives back softcap, sliding window, ALiBi, sink,
LSE, dropout, FP8, and most head-dim coverage in exchange.

---

## 8. File map

| Layer | File |
|---|---|
| v3 forward pipeline (parent) | [`block_fmha_fwd_v3_pipeline.hpp`](../3rdparty/composable_kernel/include/ck_tile/ops/fmha/pipeline/block_fmha_fwd_v3_pipeline.hpp) |
| v3 default policy (LDS layouts, GEMM choices) | [`block_fmha_fwd_v3_pipeline_default_policy.hpp`](../3rdparty/composable_kernel/include/ck_tile/ops/fmha/pipeline/block_fmha_fwd_v3_pipeline_default_policy.hpp) |
| UA pipeline (fork) | [`unified_attention_pipeline.hpp`](../3rdparty/composable_kernel/include/ck_tile/ops/unified_attention/pipeline/unified_attention_pipeline.hpp) |
| UA default policy | [`unified_attention_pipeline_default_policy.hpp`](../3rdparty/composable_kernel/include/ck_tile/ops/unified_attention/pipeline/unified_attention_pipeline_default_policy.hpp) |
| UA kernel (drives the pipeline; owns the 2D-vs-1D grid + GQA-merge + binary search) | [`unified_attention_kernel.hpp`](../3rdparty/composable_kernel/include/ck_tile/ops/unified_attention/kernel/unified_attention_kernel.hpp) |
| UA codegen (compiled shape envelope: d64 GQA-8, d128 MHA) | [`example/ck_tile/42_unified_attention/unified_attention.cpp`](../3rdparty/composable_kernel/example/ck_tile/42_unified_attention/unified_attention.cpp) |
| UA runtime selector (Triton wrapper that gates UA on/off) | [`aiter/ops/triton/attention/unified_attention.py`](../aiter/ops/triton/attention/unified_attention.py) |
| UA Python entry point | [`aiter/ops/unified_attention.py`](../aiter/ops/unified_attention.py) |
