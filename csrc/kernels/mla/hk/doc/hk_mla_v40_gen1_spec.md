<!--
HK MLA V40 Gen.1 — design spec.

Specifies *how* the bytes actually move — which lane of which wave touches
which LDS bank, which pinned VGPR holds what at which point in the loop,
and the contracts each manager and dispatch path obeys.

Audience: knows HipKittens primitives (buffer_load_lds, ds_read_b128/tr,
pinned-VGPR hex form), gfx950 mfma cadence, and the MI350 LDS bank model
(64 phys banks, 32-slot crossbar). We *use* those facts; we don't re-derive
them.
-->

# HK MLA V40 Gen.1 — Design Spec

---

## Chapter 1 — TL;DR + perf state

### What this kernel is

`mi35x_v40_fwd_decode_m16x8_fp8bf16_fp8bf16_gen1` is the V4 MLA decode kernel
for gfx950 (MI350). It implements *causal-free* per-token attention for the
DeepSeek-style MLA layout: one shared latent KV cache (one head) is read by
many query heads, with a small RoPE tail concatenated to a larger NoPE body.

There are **two partitions** of the same Gen.1 design, picked by the total
per-workgroup work $W = H \cdot \mathrm{mtp}$:

- **m16x8** ($W = 128$): 8 ptiles / workgroup, occupancy 1. Uses the newer
  `KvManager8to16bitsV2` KV pipeline (this doc's primary subject).
- **m16x4** ($W = 64$): 4 ptiles / workgroup, occupancy 2, `kBlockN = 32`.
  Uses the older `KvManager8to16bitsV1` pipeline (the V1-era layout most of
  Ch. 8's history describes).

This doc primarily specifies **m16x8**; m16x4 is called out where it diverges.

Key numbers (m16x8):

| Property | Value |
|---|---|
| Arch | gfx950 (MI350) |
| Tile name | **m16x8** — 8 ptiles per workgroup, each with m=16 rows |
| Ptile = | 1 wave (Gen.1 convention) |
| Total work per workgroup | $H \cdot \mathrm{mtp} = 128$ items |
| KV tile (`kBlockN`) | **64** rows = two 32-row sub-tiles A, B |
| $D_{\mathrm{NoPE}}$ | 448 elements, fp8 (one E8M0 scale per 64 elements) |
| $D_{\mathrm{RoPE}}$ | 64 elements, bf16 |
| $D_{\mathrm{QK}} = D_V$ | 512 ($= D_{\mathrm{NoPE}} + D_{\mathrm{RoPE}}$) |
| Q residency | half pinned-VGPR (Phase A), half LDS (Phase B) |
| KV residency | LDS, double-buffered (pong = 64 KiB) |
| Output residency | LDS bounce + VRAM (OManager V3 / V3NoStage) |
| Compiler scratch budget | m16x8: `amdgpu_num_vgpr(36)`; m16x4: `(44)` |

The router `aiter/mla.py::mla_v40_decode_fwd` dispatches to
`hk_mla_v40_decode_fwd`, which picks **m16x8** when
$H \cdot \mathrm{mtp} = 128$ and **m16x4** when $= 64$ (gfx950, fp8 Q/KV,
bf16 RoPE, page_size ∈ {1, 64}, experimental enabled). All valid $(H,
\mathrm{mtp})$ splits within a partition ride the same `mla_main` template.

### Perf state (v1.1, decode kernel only)

Measured decode-kernel-only (the `mla_reduce_v1` combine pass excluded),
gfx950, HIP 7.2.53211 / clang roc-7.2.2. Short = `b=4 c=4096`, long =
`b=33 c=63333`.

| partition | workload | page_size | µs | TFLOPS |
|---|---|---:|---:|---:|
| m16x8 (128,1) | short | 1  |  ~24.0 | ~179 |
| m16x8 (128,1) | short | 64 | ~194.3 |  ~22 |
| m16x8 (128,1) | long  | 1  | ~499.6 | ~1097 |
| m16x8 (128,1) | long  | 64 | ~615.3 |  ~890 |
| m16x4 (64,1)  | short | 1  |  ~17.9 | ~120 |
| m16x4 (64,1)  | long  | 1  | ~328.6 |  ~834 |

`page_size = 64` is markedly slower than `page_size = 1` — same ISA path
(one extra div/mod per row lookup), the gap is the metadata planner's
work distribution under page-size 64. v1.1 is ~1.3× faster than the v1
(kBlockN=32) baseline on the m16x8 path.

---

## Chapter 2 — Problem shape & notation

### 2.1 What MLA decode computes

For one decode step on one sequence position, the kernel computes

$$
S = Q\,K^{\top} \in \mathbb{R}^{m \times N_{kv}}, \qquad
P = \mathrm{softmax}\left( S/\sqrt{D_{\mathrm{QK}}} \right), \qquad
O = P\,V \in \mathbb{R}^{m \times D_V}
$$

with the V4 MLA dimensions:

| Symbol | Meaning | Value |
|---|---|---:|
| $D_{\mathrm{NoPE}}$ | non-positional head dim, fp8 elements | 448 |
| $D_{\mathrm{RoPE}}$ | RoPE tail head dim, bf16 elements | 64 |
| $D_{\mathrm{QK}}$ | $= D_{\mathrm{NoPE}} + D_{\mathrm{RoPE}}$ | 512 |
| $D_V$ | output / value head dim | 512 (= $D_{\mathrm{QK}}$) |
| $B_{\mathrm{rec}}$ | packed NoPE-record bytes / token (incl. dup E8M0 + pad) | 512 |
| $H$ | query heads sharing one KV head | varies (see 2.3) |
| $\mathrm{mtp}$ | multi-token-prediction tokens / step | varies (see 2.3) |
| $N_{kv}$ | KV context length for this batch element | varies |

The 512-byte packed record per token is $448$ fp8 NoPE values
+ $14$ duplicated E8M0 scale bytes + $50$ pad — see `kQkPackedNopeBytes`
in `hk_mla_utils.cuh`. The "512" is a *byte budget on disk*, not an
element count. The $14$ scale bytes are one E8M0 per $32$-element sub-tile
($448/32 = 14$), duplicated in pairs because the actual quant tile is
$64$ elements ($\mathrm{scales}[2i] = \mathrm{scales}[2i+1]$); the kernel
never reads the trailing $50$ pad bytes (contents undefined).

PV consumes the *full* $D_{\mathrm{QK}}$ slice (NoPE-in-bf16 + RoPE-in-bf16
after softmax cvt), so $D_V = D_{\mathrm{QK}} = 512$. This differs from
V3.2 where $V$ was the NoPE-only 512-wide slice.

MLA's defining property: a single latent KV row (one head) feeds many query
heads. That collapses KV bandwidth by a factor of $H$ vs full multi-head
attention. The kernel's job is to make those many Q rows reuse each KV load.

### 2.2 The mfma tile and "m=16"

We use the gfx950 fp8 mfma `v_mfma_f32_16x16x32_fp8_fp8` of shape

$$
(16 \times 32) \cdot (32 \times 16) \to (16 \times 16)
$$

for QK and the bf16 mfma `v_mfma_f32_16x16x32_f16` of the same shape for
PV (after the fp8 P from softmax is cast to bf16). Both have the same
m=16 / n=16 shape, so the per-ptile accumulator is a 16-row × 16-column
fp32 tile.

The "**m=16**" in the kernel name refers to this m dim: each ptile holds
16 rows of $Q$ in its mfma accumulators. Those 16 rows are the work items
this ptile is responsible for — see 2.3 for what "work item" means in MLA.

### 2.3 What "m16x8" means: ptiles and supported $(H, \mathrm{mtp})$

The total per-workgroup work is

$$
W = H \cdot \mathrm{mtp} = 128
$$

(read: 128 query heads in the $\mathrm{mtp}{=}1$ case, or fewer heads
multiplied by more predicted tokens).

The kernel splits $W$ into **8 groups**. Each group is owned by one
*processing tile*. We use the term **ptile** in this doc because
"Compute Unit" already means something specific on AMD GPUs. In **Gen.1**:

| Quantity | Value |
|---|---:|
| Groups (ptiles) per workgroup | 8 |
| Ptile = | 1 wave |
| Waves per workgroup | 8 |
| Work items per group | $W / 8 = 16$ |
| mfma m-dim | 16 (one row per work item) |

So Gen.1 is a single-wave-per-ptile design. The 8 ptiles share KV data
(loaded once into LDS) but each runs its own QK / softmax / PV stream over
its own 16 query rows.

The "x8" in `m16x8` is "**8** ptiles." Different splits of $W$ are valid
as long as $H \cdot \mathrm{mtp} = 128$ and each ptile still owns
16 work items:

| $H$ | $\mathrm{mtp}$ | $W$ | rows/ptile | supported? |
|---:|---:|---:|---:|:---:|
| 128 | 1 | 128 | 16 | ✓ (currently wired) |
| 64 | 2 | 128 | 16 | ✓ (same kernel template) |
| 32 | 4 | 128 | 16 | ✓ (same kernel template) |
| 16 | 8 | 128 | 16 | ✓ (same kernel template) |

The router dispatches m16x8 whenever `num_head * max_seqlen_q == 128`; all
four splits above are exercised by the test sweep. The sibling **m16x4**
partition ($W = H \cdot \mathrm{mtp} = 64$, 4 ptiles / workgroup) covers
$(64,1)$, $(32,2)$, $(16,4)$ — $(8,8)$ is rejected by the metadata planner's
supported-shape set. m16x4 runs the same `mla_main` shape at 4 waves /
occupancy 2 on the V1 KV pipeline with `kBlockN = 32`.

### 2.4 How the m=16 rows map to work items

The mfma's m=16 holds the 16 query rows owned by this ptile. Their packing
into the [token, head]-index space depends on which $(H, \mathrm{mtp})$
configuration is in play.

For the wired case ($H{=}128, \mathrm{mtp}{=}1$):

| m-row | token | local head index |
|---:|---:|---:|
| 0 | $t$ | $16 p + 0$ |
| 1 | $t$ | $16 p + 1$ |
| ⋮ | ⋮ | ⋮ |
| 15 | $t$ | $16 p + 15$ |

where $p \in [0, 8)$ is the ptile index for this workgroup. All 16 rows
share the same token $t$, varying only in head index.

For larger $\mathrm{mtp}$ (e.g. $H{=}16, \mathrm{mtp}{=}8$) the same 16
rows would partition into a 2-D grid of (head, predicted-token) pairs;
the exact mapping is set by the host layout of the $Q$ tensor.

### 2.5 Notation used throughout this doc

| Symbol | Meaning | Range |
|---|---|---|
| $w$ | warp index inside the workgroup | $w \in [0, 8)$ |
| $\ell$ | lane index inside a warp | $\ell \in [0, 64)$ |
| $p$ | ptile index | $p \in [0, 8)$; in Gen.1, $p = w$ |
| $t$ | thread index inside the workgroup | $t = 64 w + \ell$ |
| $i$ | KV chunk (tile) index in the main loop | $i \in [0, \lceil N_{kv}/N_{\mathrm{block}} \rceil)$ |
| $N_{\mathrm{block}}$ | KV tile size along the $N$ dim | **64** (`kBlockN`) = two 32-row sub-tiles A, B |
| $m \in [0,16)$ | row in this ptile's mfma accumulator | one row = one work item |

> **kBlockN = 64 (m16x8).** m16x8 processes a 64-row KV tile per main-loop
> iter, internally two 32-row sub-tiles (A at LDS offset 0, B at `+kSubPong`).
> This halves the barrier / softmax / loop overhead per KV row vs the original
> kBlockN=32. **m16x4 stays at kBlockN=32** on the V1 pipeline. Where the older
> text below says "32", read it as "one 32-row sub-tile" (m16x8) or the literal
> tile (m16x4).

Note: "warp" and "wave" mean the same thing on AMD; we say *warp* by
default and *wave* when the surrounding text is talking about HW
scheduling (e.g. wave priority via `setprio`).

## Chapter 3 — High-level dataflow

### 3.1 Tensors in flight

Each ptile holds **16 rows of $Q$** for its lifetime — these come from VMEM
once at the prologue, get split into a VGPR-resident half and an
LDS-resident half (see 3.3), and are reused for every KV tile.

Each iteration $i$ of the main loop processes one **KV tile** of shape
$N_{\mathrm{block}} \times D_{\mathrm{QK}} = 32 \times 512$ (one 32-token
window of the K/V latent, NoPE in fp8 + RoPE in bf16 once cast). The KV tile
is loaded into LDS once (shared by all 8 ptiles) and stays resident for one
iteration's QK+PV before the LDS slot is recycled by the next tile.

The output accumulator $\mathrm{oaccu} \in \mathbb{R}^{16 \times 512}$
(fp32) lives **entirely in pinned VGPRs** for the duration of the loop.
Only at epilogue is it cast to bf16 / fp32 and written to VRAM via a
small LDS "bounce" region.

### 3.2 Block diagram

Critical sp3 / mfma instruction at each edge is annotated in parentheses.

```
                              Global VMEM
              ┌───────────────────┬──────────────────────┐
              │                   │                      │
              ▼                   ▼                      ▼
         ┌─────────┐         ┌──────────┐          ┌──────────┐
         │   Q     │         │  K (fp8  │          │  V (fp8  │
         │ (fp8 +  │         │  + bf16  │          │ NoPE) ─  │
         │  bf16)  │         │   RoPE)  │          │ shares K │
         └────┬────┘         └─────┬────┘          │ LDS slot │
              │                    │               └────┬─────┘
              │ buffer_load_lds   buffer_load_dwordx4   │
              │ buffer_load_dwordx4 + cvt+scale         │
              │   (Phase 1 + 2 of QMgr)                 │
              ▼                    ▼                    │
   ┌────────────────────┐  ┌──────────────────────┐     │
   │  Q-LDS (LDS half = │  │ KV-LDS, DOUBLE pong  │     │
   │  Q[:, 256:512])    │  │ 64x512 bf16 = 64 KiB │◀────┘
   │  + Phase-1 staging │  │ (two 32-row sub A/B) │
   │  + KV raw-fp8 stage│  │ buf_A / buf_B swap   │
   └─────────┬──────────┘  └──────────┬───────────┘
             │                        │
             │ ds_read_b128            │ ds_read_b128 (K side)
             │                        │ ds_read_b64_tr_b16 (V side, transpose)
             ▼                        ▼
   ┌────────────────────┐  ┌──────────────────────┐
   │ q_vgpr  v64-v127   │  │ k0/k1/k2  v36-v47    │
   │ (Q[:,0:256] Ph.A;  │  │ (KV mfma operands,   │
   │  q_lds window Ph.B)│  │  3 tiles for N=64)   │
   └─────────┬──────────┘  └──────────┬───────────┘
             │                        │
             └─── v_mfma_f32_16x16x32_fp8_fp8 ───┘   ← QK
                              │
                              ▼
                  ┌─────────────────────┐
                  │ S tile (16x32 fp32) │   compiler scratch v0..v35
                  └─────────┬───────────┘
                            │ softmax (online)
                            │   v_max3 / warp_reduce(Max)
                            │   v_exp_f32 / warp_reduce(Add)
                            │   (hi warps: de-packed softmax, no v_pk_*)
                            ▼
                  ┌─────────────────────┐
                  │ p_comp  v48-v63     │  (fp32, 16xN/4, N/4 reg/lane)
                  │ p_mfma  v48-v55     │  (bf16 overlay, low half)
                  │   ↑ v_cvt_pk_bf16_f32 (pinned-DST)
                  └─────────┬───────────┘
                            │   ── reuses KV-LDS pong as V via transpose-read
                            ▼
              ── v_mfma_f32_16x16x32_f16 ──   ← PV
                            │   (interleaved with v_mul_f32 rescale)
                            ▼
                  ┌─────────────────────┐
                  │ oaccu   v128-v255   │  (fp32, 16x512, all pinned)
                  └─────────┬───────────┘
                            │  epilogue:
                            │   1) normalize by 1/row_sum_e:
                            │      hi warps  → hk::mul_vgpr (v_pk_mul_f32)
                            │      lo warps  → v_mul_f32_e32 sweep (de-packed)
                            │   2) OMgr V3 / V3NoStage
                            ▼
                  ┌─────────────────────┐
                  │ bounce LDS (per-warp│  ~2 KiB bf16 / ~4.5 KiB fp32
                  │  ds_write)          │  with sb8 inverse-perm un-swizzle
                  └─────────┬───────────┘
                            │  buffer_store_dwordx4 (coalesced)
                            ▼
                       VRAM output
```

Key sp3 ops to remember:

- **`v_mfma_f32_16x16x32_fp8_fp8`** — QK (single 16×32 × 32×16 → 16×16 fp32).
- **`v_mfma_f32_16x16x32_f16`** — PV (same shape, bf16 inputs).
- **`ds_read_b128`** — vanilla bf16 read for QK A-tile.
- **`ds_read_b64_tr_b16`** — transpose-read for V → mfma A-operand layout (Ch. 10.2).
- **`buffer_load_lds`** — direct vmem → LDS bypassing VGPRs (RoPE path + Q Phase 1 staging).
- **`v_cvt_scalef32_pk_bf16_fp8`** — fused fp8 → bf16 + e8m0 scale; emitted via pinned-DST asm wrapper (Ch. 5, Ch. 13).
- **`v_cvt_pk_bf16_f32`** — fp32 → bf16 pack for `p_comp → p_mfma` overlay.

The KV "double buffer" is the only inter-iteration LDS resident — every
iter writes the *next* tile into the *other* buffer while reading the
*current* tile out of the active one. See Ch. 8.

### 3.3 "Phase A / Phase B" — the KV-pipeline D-axis split

> **Terminology shift vs V1.** In the original V1 design, "Phase A/B" meant a
> $Q$-source split (half $Q$ in VGPR, half in LDS). The current m16x8 pins
> **all** of $Q$ in `q_vgpr` (`kQkGemmTiles = 16`, `kRopeInVgpr = true`), so
> QK reads every col-tile from VGPR and there is no Phase-B-Q-from-LDS. The
> `p_lds_q` region is now used only during the **Q prologue** load (Ch. 6-7).

In the current kernel, **Phase A / Phase B** name the two halves of the
**KV double-buffer pipeline**, run every iter by every warp:

| Phase | Work | LDS effect |
|---|---|---|
| **Phase A** | prefetch this warp's band of the *next* KV tile: 2 NoPE carriers → VGPR (`p0/p1`); staged strips → raw-fp8 staging LDS via `buffer_load_lds`; issue the index resolve for tile $i{+}2$ | writes staging LDS + VGPR carriers (no pong write yet) |
| **Phase B** | after QK: cvt+scale the carriers + staged strips fp8 → bf16 and `ds_write` them into `p_lds_kv_next`; hi warps DMA the RoPE tail directly | writes `p_lds_kv_next` (consumed next iter after the swap) |

Why the KV split: it lets the vmem latency of Phase A overlap the barrier +
QK, while Phase B's cvt+store overlaps softmax/PV. $Q$ is fully pinned, so
the reduction-axis work is entirely on the KV side now.

There is still no inter-warp swap of $Q$ ownership: each ptile keeps its own
16 rows in its own VGPRs. The KV side is what's shared across ptiles.

There is no inter-warp swap of $Q$ ownership: each ptile keeps its own 16
rows in its own VGPRs/LDS. The KV side is what's shared across ptiles.

### 3.4 KV tile timing (one iter, m16x8 V2)

A m16x8 iter processes one 64-row KV tile (sub-tiles A, B). Each warp owns
one 16-row **band** of that tile × one 256-col **tile** (Lo = tile 0,
Hi = tile 1 — see 3.5). Per warp per iter:

| Step | Work | Notes |
|---:|---|---|
| 0 | **deferred strip-3 consume** (lo only, non-first iter) | cvt+store the prev iter's staged strip 3 into the *current* pong before QK (Ch. 8) |
| 1 | Phase A: prefetch **next** band | 2 NoPE carriers → VGPR (`p0/p1`), strips 2,3 (lo) / strip 2 (hi) → staging LDS via `buffer_load_lds`; index resolve for tile $i{+}2$ |
| 2 | deferred PV of the **previous** tile (hi warps: `kHasPv`) | reads `p_lds_kv_next` |
| 3 | `s_barrier` + QK on tile $i$ | all 16 Q col-tiles from `q_vgpr` (kQkGemmTiles=16), K from `p_lds_kv_curr` |
| 4 | Phase B: cvt+store carriers + own staged strips → `p_lds_kv_next`; hi RoPE DMA | strip 3 (lo) is deferred to next iter's step 0 |
| 5 | softmax + pack `p_comp → p_mfma` | **hi** warps de-packed softmax (no `v_pk_*`); lo packed |
| 6 | PV of tile $i$ (lo warps: `kPvAtEnd`) | one `hk_mla_v40_pv_stage` contracts **both** sub-tiles A+B (single prologue) |
| 7 | epilogue only: normalize `oaccu` by `1/row_sum_e` | **lo** warps de-packed (`v_mul_f32_e32` sweep); hi packed (`v_pk_mul_f32`) |

The KV bound is the LDS double-buffer; loads are hidden under QK/PV by the
carrier + staging prefetch. The packed-ALU port is spread across the two warp
groups by *phase*: lo = packed softmax + de-packed oaccu-normalize; hi =
de-packed softmax + packed oaccu-normalize (they share a SIMD).

### 3.5 Who owns what (m16x8)

Two compile-time warp types (`enum WarpTypeM16x8`), 4 warps each:

| Warp type | Warps | Owns (of the 64×512 tile) | PV | softmax |
|---|---|---|---|---|
| `LoNoPEWarp` | 0–3 | band `w&3` × **tile 0** (cols 0–255, pure NoPE) | at call end (`kPvAtEnd`) | packed (`v_pk_*`) |
| `HiRoPEWarp` | 4–7 | band `w&3` × **tile 1** (cols 256–511, NoPE + RoPE tail) | deferred (`kHasPv`) | de-packed |

`band = w & 3` (rows `[band·16, +16)`); `sub_off = ((w>>1)&1)·kSubPong`
(sub-tile A/B); `row_tile = w & 1`. Warp $i$ and $i{+}4$ share the same
band/rows/`sub_off`/`row_tile`, differing only in tile — this is what lets
the deferred strip-3 store land in the right pong slot (Ch. 8).

| Resource | Owner / manager | Lifetime |
|---|---|---|
| `q_vgpr` v64..v127 (Q[:,0:256] Ph.A + `q_lds` window Ph.B) | `QManager…::load_q` | Phase A / Phase B of every iter |
| Q-LDS region (`p_lds_q`) | QManager Phase 2 | Phase B reads (whole loop) |
| KV-LDS pong (`p_lds_kv_curr/next`, 64 KiB each) | `KvManager8to16bitsV2` | one iter (then swapped) |
| KV raw-fp8 staging LDS | `KvManager8to16bitsV2` (strips 2,3) | within one iter (strip 3 spans to next) |
| `k0/k1/k2` v36..v47 | `KvManager8to16bitsV2::load_k_to_gpr` | inside one QK mfma pair |
| `p_comp` v48..v63 | softmax (`hk_mla_softmax.cuh`) | softmax → PV of one iter |
| `p_mfma` v48..v55 | PV gemm | overlay on p_comp |
| `oaccu` v128..v255 | PV gemm | whole loop |
| `s3_scale` (1 vgpr) | carried strip-3 e8m0 scale | across one iter (Ch. 8 deferred strip-3) |
| OMgr bounce LDS | `OManager…V4Gen1Swizzle*` | epilogue only |

The pinned VGPR layout is fixed by `amdgpu_num_vgpr(36)` (m16x8) on the
`__global__` plus inline-asm hex names — see Ch. 5. (m16x4 uses V1 managers,
`kBlockN=32`, and `amdgpu_num_vgpr(44)`.)

## Chapter 4 — LDS budget & layout

### 4.1 Budget and occupancy

V4 Gen.1 targets MI350 with `kOccupancy_=1`: one workgroup per CU at a
time, so the entire 160 KiB of LDS is available. The total budget at
that occupancy is bounded by

$$
\mathrm{kSzLdsKv} \,+\, \max(\mathrm{kSzLdsO},\, \mathrm{kSzLdsKv}) \,+\, \mathrm{kSzLdsQ} \,+\, \mathrm{kSzLdsStage} \le 160 \text{ KiB}
$$

(enforced by a `static_assert` in the kernel). The first pong holds `curr`;
the second pong region holds `next` **and** is overlaid by the O bounce at
epilogue, so it is budgeted at $\max(\mathrm{kSzLdsO}, \mathrm{kSzLdsKv})$.

Concretely (m16x8, kBlockN=64; from `get_lds_size_in_byte()` accessors):

| Region | Size | Owner | Source |
|---|---:|---|---|
| `kSzLdsKv` (one KV pong) | **64 KiB** | `KvManager8to16bitsV2` | $\mathrm{kBlockN} \cdot D_{\mathrm{QK}} \cdot \mathrm{sizeof(bf16)} = 64 \cdot 512 \cdot 2 = 65{,}536$ B (two 32-row sub-tiles A@0, B@`kSubPong`=32 KiB) |
| `kSzLdsStage` (KV raw-fp8 staging, sub-tile B) | **16 KiB** | `KvManager8to16bitsV2` | $32 \cdot \mathrm{kQkPackedNopeBytes} = 32 \cdot 512$; per-warp strips 2/3 slots |
| `kSzLdsQ` (Q final + Phase-1 staging overlay) | ≤ 64 KiB | `QManager8to16bitsV1` | `kFinalLdsBytes` |
| `kSzLdsO` (OMgr V3 bf16 bounce) | 16,896 B | `OManager16bitsV4Gen1Swizzle` | $8 \cdot 2112~\text{B}$ |
| `kSzLdsO` (OMgr V3 fp32 split bounce) | 34,816 B | `OManager32bitsV4Gen1Swizzle` | $8 \cdot 4352~\text{B}$ |
| `kSzLdsO` (V3NoStage variant) | 0 | `OManager32bitsV4Gen1SwNoStage` | direct VRAM, no bounce |

(m16x4 keeps the V1 pong at 32 KiB and has no `kSzLdsStage`.)

### 4.2 Layout (one snapshot, m16x8)

```
LDS address (bytes, low → high):

+0                           p_lds_kv_curr   ← KV pong A (64 KiB, sub A@0 / B@32K)
+0x10000 (64 KiB)            p_lds_kv_next   ← KV pong B (64 KiB)
                                              ─ during epilogue:
                                                OVERLAID by OMgr bounce
                                                (V3 bf16 or V3 fp32)
+p_lds_q  (after max(KV,O))  p_lds_q         ← Q final + Phase-1 staging
+p_lds_kv_stage (after Q)    p_lds_kv_stage  ← KV raw-fp8 staging (16 KiB)
```

`p_lds_kv_curr` and `p_lds_kv_next` swap pointers every iteration:

| Iter | `p_lds_kv_curr` points to | `p_lds_kv_next` points to |
|---|---|---|
| $i$ even | LDS base | LDS base + 64 KiB |
| $i$ odd | LDS base + 64 KiB | LDS base |

So at any moment one pong is being *read* (QK / PV mfma sources) and the
other is being *written* (prefetch + cvt for the next KV tile). The
raw-fp8 staging region is separate and per-warp private (not swapped).

### 4.3 Why O bounce overlays `p_lds_kv_next`, not `p_lds_q`

This is the comment block at lines 219–235 of the kernel file, distilled:

| Choice | Hazard |
|---|---|
| Overlay O bounce inside `p_lds_q` (the natural choice — Q is dead by epilogue) | Per-warp strides differ: QManager uses 8 KiB/warp, OMgr V3 uses 2112 B/warp (bf16) or 4352 B/warp (fp32). The mismatched per-warp strides create **cross-warp aliasing** with the *next* `work_idx`'s `load_q` — a fast warp that has already finished its epilogue would have its in-flight next-iter Q load racing a slow warp's OMgr bounce write. |
| Overlay O bounce inside `p_lds_kv_next` ✓ | Safe: `p_lds_kv_next` is **dead** on the global last iter (the swap is a no-op for the trailing tile). The next `work_idx`'s KV prologue writes into `p_lds_kv_curr` (= the *other* pong), not `p_lds_kv_next`, so the OMgr bounce's lingering bytes don't race anything. |

The cost is that O bounce + KV-next must both fit in $\max(\text{KV},\text{O})$
of space, which is why the `static_assert` budgets that maximum and the
allocator places the Q region after it.

### 4.4 Why Q is placed last (the kLdsHeadPadBytes story)

`QManager8to16bitsV1::p1_vmem_to_staging_chunk` pre-subtracts up to 192 B
from the LDS destination pointer (chunks 0/1/2/3 subtract 0/64/128/192).
This is a register-pressure trick — folding `kColInRecord` into the base
pointer keeps the per-lane address arithmetic to a single add. But it
means the *staging* base address ($\mathit{pLdsQ} - 192$) must still
land in a valid LDS region.

Putting Q **after** both KV pongs gives the pre-subtract enough headroom:
when warp 0 stages with `kColInRecord = 192`, the dst pointer falls
192 bytes earlier — inside the KV-next region, still valid LDS, and
harmlessly overwritten on the next iter's prefetch. Without that
headroom, the address would underflow mod $2^{32}$ and the store would
silently drop. The kernel encodes this with a second `static_assert`:

$$
\mathrm{kSzLdsKv} + \max(\mathrm{kSzLdsO},\, \mathrm{kSzLdsKv}) \ge \mathrm{QManager::kLdsHeadPadBytes} = 192
$$

### 4.5 Surviving bank-conflict notes

Two writer-side conflicts remain documented but mitigated:

| Site | Conflict | Mitigation |
|---|---|---|
| QManager Phase 2 NoPE writer (Site C) | 2-way `ds_write_b128` bank conflict | Vmem-load-side column-half-swap + reader XOR ([[v40-qlds-bank-conflict-swizzle]]) — not fixable by an LDS-write-address swap (Method 1 silently fails) |
| OManager V32 read path (legacy) | bank conflict on `ds_read` | Left in place; V32 only, not V40 |

Reader-side conflicts on V40 Q-LDS and KV-LDS were fully eliminated in
commits `f84c817b8` (Q+KV loads) and `3c55c6594` (all paths except
OMgrV32). The **writer**-side sub-tile-of-8 swizzle for Q-LDS and KV-LDS
(Ch. 7, Ch. 8) is the second layer that finally cleared the writer
2-way conflicts on those paths.

## Chapter 5 — Pinned VGPR map

### 5.1 The pinning contract

HK kernels reserve VGPRs by **two complementary mechanisms**:

1. `__attribute__((amdgpu_num_vgpr(N)))` on the `__global__` constrains the
   LLVM register allocator to use only `v0..v(N-1)` for its own scratch.
   **m16x8: N = 36** (`k_scratch_budget` at kBlockN=64); **m16x4: N = 44**
   (kBlockN=32).
2. Inline asm that names registers in **hex form** (`v[0x40]`, `v[0x80:0x81]`)
   reserves `v(N)..v255` for hand-pinned data. The compiler cannot rename these.

The whole map is generated from `HkMlaV40Regs<T>` in
`hk_mla_v40_fwd_decode_gen1_common.cuh` (the single source of truth,
parameterized by `kBlockN`); the kernel re-aliases the `k_*` constants
locally. If (1) is missing, the compiler may emit a decimal operand ≥ N that
silently overlaps hand-pinned data — corruption with no diagnostic. The audit
`[[check-unpinned-reg-usage]]` scans the post-`--save-temps` `.s` for decimal
`v ≥ N` and for `vgpr_spill_count > 0`. **Run it after every nontrivial
change.**

### 5.2 The full map (m16x8, kBlockN=64)

Compiler scratch: **36** (v0..v35). Hand-pinned: **220** (v36..v255).

| Range | Role | Owner / writer | Reader | Lifetime |
|---|---|---|---|---|
| v0..v35 | Compiler scratch: address arithmetic, fp8→bf16 cvt staging, e8m0 scale dwords, `ds_read`/`tr` buffers, **carried `s3_scale`** + the `row_kv_ld_next` cse-break hold | LLVM | LLVM | whole kernel; **budget=36, spill=0** |
| v36..v47 | `k_0 / k_1 / k_2` — three 16×32 KV mfma operand tiles (4 vgprs each). Doubles as the PV V-tile staging (`v_1`) during PV | `KvManager…::load_k_to_gpr` / `load_transposed_v_to_gpr` | QK / PV mfma (B / A operand) | inside one mfma pair |
| v48..v63 | `p_comp` — fp32 softmax output, `kBlockN/4 = 16` fp32/lane covering the 16×64 P-tile. **Overlaid**: `p_mfma` (bf16 P for PV) on v48..v55; the QK V-operand `v_0` on v60..v63 | softmax / PV cvt | softmax (rescale), PV | softmax → PV of one iter |
| v64..v127 | `q_vgpr` — **full** Q (all 512 cols) in mfma A-operand layout, `kQkGemmTiles=16` tiles; a small `q_lds` window at the top is prologue scratch | `QManager::load_q` (prologue) | QK mfma (A-operand), all col-tiles | whole loop (read-only after prologue) [^q-ro] |
| v128..v255 | `oaccu` — fp32 output accumulator, 128 fp32/lane covering 16×512 | PV mfma (C/D-operand) | epilogue (OManager) | whole loop |

Note the tight overlap in v48..v63: `p_comp`, its bf16 `p_mfma` overlay, and
the QK `v_0` operand all share that 16-vgpr window at different points in the
iter. `k_0` similarly doubles as a PV V-tile carrier.

[^q-ro]: Confirmed read-only after the prologue — see auto-memory
`[[v40-pinned-q-read-only-confirmed]]`. Any V40 bug that looks like "Q got
corrupted mid-loop" is **not** a Q-VGPR clobber; look downstream.

### 5.3 Why the layout is shaped this way

Going low → high (toward v255), the rationale per range (m16x8):

- **v0..v35 (compiler scratch).** Bounded by `amdgpu_num_vgpr(36)`. Holds
  per-iter dynamic state: address arithmetic for `buffer_load_lds`,
  `ds_read`/`tr` destination dwords, softmax intermediates, e8m0 scale
  dwords, and the two cross-iter scalars introduced this line of work — the
  carried `s3_scale` (deferred strip-3, Ch. 8) and the `row_kv_ld_next`
  cse-break hold (Ch. 13). **Zero spill** is the current measurement; new
  pinned data must come out of v36..v255, not here. Growing pinned data
  shrinks this window, so the deferred-strip-3 / cse-break carries are the
  minimum the scratch region can afford.
- **k_0 / k_1 / k_2 (v36..v47).** Three 16×32 KV mfma operand tiles. Three
  (not two) because kBlockN=64 contracts two 32-row sub-tiles per PV call;
  `k_0` also serves as a PV V-tile carrier (role-toggled across the QK→PV
  `s_waitcnt`).
- **p_comp / p_mfma / v_0 (v48..v63).** The 16-vgpr softmax window, triple-
  overlaid: `p_comp` (fp32, kBlockN/4), the `p_mfma` bf16 overlay on the low
  half (v48..v55), and the QK `v_0` operand on the high half (v60..v63). The
  overlay is safe because softmax→PV uses `low-to-high pack`
  (`pack_2f32_to_bf16_pair_pinned` in `hk_mla_utils.cuh`; gotcha: not the
  runtime-arg form — see Ch. 13). **Softmax: lo warps packed (`kSoftmaxUsePk
  = kPvAtEnd`), hi de-packed.**
- **q_vgpr (v64..v127).** Full Q, all 512 cols, in mfma A-operand layout
  (`kQkGemmTiles=16`, `kRopeInVgpr`). Read-only for the whole loop. A small
  `q_lds` window at the high end is prologue staging scratch only.
- **oaccu (v128..v255).** The biggest block, 128 vgprs, at the top so its
  base is a round `0x80` — simplifies the OManager offset arithmetic. Holds
  $16 \cdot 512 / 64 = 128$ fp32/lane. Normalized at epilogue by
  `1/row_sum_e`: hi warps via `hk::mul_vgpr` (`v_pk_mul_f32`), lo warps via a
  de-packed `v_mul_f32_e32` sweep (Ch. 10).

### 5.4 Compiler-scratch budget audit

The audit script reports three numbers:

| Number | Meaning | Current value (m16x8) |
|---|---|---:|
| budget | `N` from `amdgpu_num_vgpr(N)` | 36 |
| spill | `.vgpr_spill_count` in the kernel metadata | 0 |
| free gprs | `N - max_observed_decimal_v - 1` | tight (v0..v35) |

Zero spill is the invariant to protect. Any new pinned data, new inline-asm
clobber, or wider unroll could push the compiler over the 36-reg scratch
budget into spill (and into the pinned range — the classic clobber bug).
Always re-run the audit after touching:

- inline asm clobber lists in the managers
- `static_for` unroll factors in the kernel body
- any new `sched_barrier(0)` (it widens live ranges)

See `.claude/skills/check-unpinned-reg-usage/` for the script.

## Chapter 6 — QManager Phase 1 (vmem → staging LDS → pinned q_vgpr)

> **Full-Q-in-VGPR update.** The current m16x8 pins **all** of $Q$ (512 cols,
> NoPE + RoPE) in `q_vgpr` v64..v127 (`kQkGemmTiles=16`, `kRopeInVgpr=true`).
> The "VGPR half / LDS half" framing in Ch. 6-7 is V1 history: the prologue
> still stages $Q$ through LDS (Phase 1 NoPE, Phase 2 sb8-perm + RoPE) but
> everything lands in `q_vgpr`, and the QK loop reads Q only from VGPR. Read
> the mechanics below as "how the prologue lands each chunk in q_vgpr"; ignore
> references to a persistent Phase-B Q-LDS. Register ranges cited as v72.. are
> now v64.. (see Ch. 5.2).

Phase 1 fills the leading chunks of $Q$: $Q[:, 0{:}256]$ (the first 256 of
the 448 NoPE elements) into pinned `q_vgpr`. It runs once at the prologue,
before the main loop.

### 6.1 Geometry

| Symbol | Meaning | Value |
|---|---|---:|
| `kVgprHalfCols` | NoPE cols going to VGPR | 256 |
| `kP1ChunkCols` | cols per Phase-1 chunk | 64 |
| `kP1NumChunks` | chunks needed for 256 cols | 4 |
| `kP1StagingBytesPerWarp` | per-warp staging slot, one buffer | $16 \cdot 64 \cdot 1 = 1024$ B |
| `kP1NumStagingBuffers` | double-buffer slots | 2 |
| `kP1StagingBytesPerWarpTotal` | per-warp staging, both buffers | 2048 B |
| `kPackedNopeStride` | source bytes per token | 512 |
| `kScaleBaseOff` | first E8M0 scale byte in the 512-byte record | 448 |

Each warp covers 16 rows of $Q$ (`kTileM = 16`). Each chunk covers 64 cols.
So one Phase-1 iteration moves a $16 \times 64$ fp8 tile = 1024 B per warp,
plus 16 E8M0 scale bytes (one per row).

The staging is **per-warp private** — wave $w$'s staging sits inside wave
$w$'s own 8 KiB slice of the final 64 KiB Q-LDS region (see §4.4 and
§6.5). Because no inter-wave LDS traffic happens in Phase 1, there is
**no `__syncthreads()`** between Phase 1 and Phase 2.

### 6.2 The two-step pipeline per chunk

For each of the 4 chunks, Phase 1 issues two routines back-to-back. The
double-buffer means chunk $c+1$'s `p1_vmem_to_staging_chunk` issues while
chunk $c$'s `p1_staging_to_vgpr_chunk` is still consuming the prior buffer.

```
  chunk 0 → buf 0 → vmem→staging  ───┐
  chunk 1 → buf 1 → vmem→staging  ─┐ │
                                   │ └→ staging→vgpr chunk 0
  chunk 2 → buf 0 → vmem→staging   └→ staging→vgpr chunk 1
  ...
```

### 6.3 Step 1 — `p1_vmem_to_staging_chunk` (vmem fp8 → per-warp staging LDS)

This is a `buffer_load_dwordx4 lds:` direct vmem→LDS, plus a
`buffer_load_ubyte` for the E8M0 scale (which lands in a VGPR, used by
Step 2).

**Per-lane vmem offset (NoPE bytes):**

$$
\mathit{vOff}(\ell) = (\ell \gg 2) \cdot 512 + (\ell  \mathbin{\mathrm{and}}  3 \oplus (S \ll 1)) \cdot 16
$$

with $S = (\ell \gg 4)   \mathbin{\mathrm{and}}   1$. The bare expression
$(\ell{ \mathbin{\mathrm{and}} }3)\cdot 16$ would walk 4 lanes × 16 B = 64 B/row = exactly one
chunk row; the XOR-by-$2S$ swap on sub-tile row-bands (rows 4..7 and
12..15) is what makes the *reader* in Step 2 conflict-free at the b128
non-linear cycle. The **LDS write side** is unaffected — the HW pattern
for `buffer_load_dwordx4 lds:` is fixed at lane $\ell \to$ LDS offset
$\ell \cdot 16$, independent of any data permutation.

**Per-lane → staging-LDS byte (one chunk, buf 0):**

| lane $\ell$ | row in warp $= \ell{\gg}2$ | col-quad logical $= \ell  \mathbin{\mathrm{and}}  3$ | $S$ | col-quad physical (vmem) | staging LDS dst |
|---:|---:|---:|---:|---:|---:|
| 0 | 0 | 0 | 0 | 0 | $\ell{\cdot}16 = 0$ |
| 1 | 0 | 1 | 0 | 1 | 16 |
| 2 | 0 | 2 | 0 | 2 | 32 |
| 3 | 0 | 3 | 0 | 3 | 48 |
| 16 | 4 | 0 | 1 | **2** (XOR-flipped) | 256 |
| 17 | 4 | 1 | 1 | **3** | 272 |
| 18 | 4 | 2 | 1 | **0** | 288 |
| 19 | 4 | 3 | 1 | **1** | 304 |
| ⋮ | ⋮ | ⋮ | ⋮ | ⋮ | ⋮ |

So the 64-col chunk lands row-major in the staging slot: row $r$
occupies bytes $[r\cdot 64,\, r\cdot 64 + 64)$.

**Per-lane vmem offset (E8M0 scale):**

Each row $r \in [0,16)$ has its own scale dword (dup'd to 2 bytes for
alignment). For chunk $c$, the scale byte lives at byte $448 + 2c$ of the
512-byte record:

$$
\mathit{vOffScale}(\ell) = (\ell  \mathbin{\mathrm{and}}  15) \cdot 512, \qquad \mathit{iOff} = 448 + 2c
$$

Note: `scale_row = lane & 15` (**not** `lane >> 2`). Step 2's consumer
attributes lane $\ell$ to data row $\ell  \mathbin{\mathrm{and}}  15$, so the scale must
match that attribution. The mismatched form would scale lane $\ell$'s
fp8 by row $(\ell{\gg}2)$'s scale — silently wrong on near-uniform
data, catastrophic on outliers. (This is encoded in the comment at
line ~272 of the manager.)

### 6.4 The pre-subtract trick (and why Q is at the high end of LDS)

`buffer_load_dwordx4 lds:` adds its `i_offset` (an immediate) to **both**
the vmem source AND the LDS destination. The kernel exploits this:

- vmem side: $\mathit{iOffset} = \mathrm{kColInRecord} = c \cdot 64$
  → folds the chunk-base column into the immediate, saving a vgpr add.
- LDS side: dst is set to $\mathrm{staging} + \mathrm{kStagingI} - \mathrm{kColInRecord}$
  → cancels the spurious LDS shift.

This is identical to V32's known trick — fewer VGPRs in the inner loop.

**Hazard:** if $\mathrm{staging} < \mathrm{kColInRecord}_\mathrm{max} = 192$, the
LDS pointer underflows mod $2^{32}$ and the store silently drops. Warp 0
is the only warp where $\mathrm{staging} = \mathit{pLdsQ}$ exactly,
so the kernel places Q **after** the KV pongs + the O bounce (m16x8:
`kSzLdsKv` + `max(O, kSzLdsKv)`) — giving warp 0's staging at least 192 B
of preceding LDS to absorb the
subtract. The encoded `kLdsHeadPadBytes = 192` and the static assert in
the kernel guarantee this.

### 6.5 Step 2 — `p1_staging_to_vgpr_chunk` (staging LDS + scale → bf16 in q_vgpr)

This step:

1. drains both `vmcnt(0)` (staging vmem traffic + scale dword) AND
   `lgkmcnt(0)` (the LDS-write half of `buffer_load_lds` increments
   lgkmcnt on gfx9),
2. issues **one** `ds_read_b128` to bring 16 fp8 = 16 B/lane into the
   `fp8` vector,
3. converts each fp8 dword into bf16 directly into the caller-pinned
   q_vgpr slot, scaled by the E8M0 fp32 form.

**Per-lane LDS read address (mirrors the writer's swizzle):**

$$
\mathit{addrBase}(\ell) = \mathrm{staging} + (\ell  \mathbin{\mathrm{and}}  15) \cdot 64 + C_{\mathrm{phys}} \cdot 16
$$

with $C_{\mathrm{phys}} = (\ell{\gg}4)  \mathbin{\mathrm{and}}  3 \oplus ((\ell{\gg}2)  \mathbin{\mathrm{and}}  1) \ll 1$.

The two iter columns (`iter ∈ {0,1}` for the 2 mfma A-tiles of this chunk,
covering cols $[0,32)$ and $[32,64)$) **share `C_phys`** — they differ by
8 bytes, which folds into the `ds_read_b64` immediate offset. One b128
load satisfies both iters with no second address vgpr.

**Bank check** (one `ds_read_b128` per chunk, 4 non-linear cycles):

The non-linear b128 cycle 0 pairs lanes $(L, L{+}20)$. `+20` flips bit 4
and bit 2 of $L$ together. Bit 2 is $S$; bit 4 is bit 0 of the col-band
$\mathrm{cb} = (\ell{\gg}4)  \mathbin{\mathrm{and}}  3$. The writer XOR'd bit 1 of $\mathrm{cb}$
(via $S{\ll}1$), which `+20` does NOT touch — so the pair lands in
distinct quads. Per-lane quad

$$
q(\ell) = ((\ell  \mathbin{\mathrm{and}}  15)   \mathbin{\mathrm{and}}   3) \cdot 4 + C_{\mathrm{phys}}(\ell)
$$

distributes the 16 lanes of each cycle across distinct quads in $[0,16)$
— conflict-free on all 4 cycles. See the writer comment at lines
~252-265 of the manager for the algebra.

### 6.6 fp8 → bf16 cvt and scale application

`p1_staging_to_vgpr_chunk` uses `cvt_scalef32_pk_bf16_fp8_pinned` from
`hk_mla_utils.cuh` — a `__device__` wrapper around
`v_cvt_scalef32_pk_bf16_fp8` whose **destination VGPR is named in inline
asm hex form** (`v[0x...]`). The pinned-DST form is essential:

> The natural form `v_cvt_scalef32_pk_bf16_fp8 v[N]` (template int) is
> silently wrong because the assembler treats `N` as a constraint
> letter, not a register number — see auto-memory
> `[[v40-cvt-to-pinned-inline-asm-gotcha]]`. The pinned form encodes
> the register *number* directly in the asm string and routes through a
> `v_mov` trampoline when needed.

The fp8 vector layout is 4 dwords / lane (= 16 bytes = 16 fp8 values).
Each dword feeds **two** cvt calls (`opsel false` reads the low half,
`opsel true` reads the high half), producing 8 cvt calls per Phase-1
iteration. Each cvt writes 2 bf16 values = 1 dword into the pinned
q_vgpr slot:

| `kVgprChunkBase + i` | which dword in fp8 | opsel | writes |
|---:|---|:---:|---|
| +0 | fp8[0] | false | bf16 dw[0,1] = cols 0..3 of iter 0 |
| +1 | fp8[0] | true  | bf16 dw[2,3] = cols 4..7 of iter 0 |
| +2 | fp8[1] | false | iter 0, cols 8..11 |
| +3 | fp8[1] | true  | iter 0, cols 12..15 |
| +4 | fp8[2] | false | iter 1, cols 0..3 |
| +5 | fp8[2] | true  | iter 1, cols 4..7 |
| +6 | fp8[3] | false | iter 1, cols 8..11 |
| +7 | fp8[3] | true  | iter 1, cols 12..15 |

`kVgprChunkBase = GPR_NOPE_VGPR_START + 8 \cdot c` for chunk $c$, so all
4 chunks together write $4 \cdot 8 = 32$ vgprs/lane = the full
`q_vgpr` range v72..v103.

V4's NoPE scale layout: **one E8M0 scale per 64-col tile**, shared across
both 32-col mfma A-tiles within the chunk. The cvt scale_f is computed
once per chunk via `hk_mla::e8m0_to_f32` (which requires `asm volatile` —
see `[[v40-e8m0-to-f32-asm-required]]`).

### 6.7 Why the second sched_barrier matters

Between the `ds_read_b128` and the cvt calls, the code issues:

```cpp
__builtin_amdgcn_s_waitcnt(hk_mla::encode_s_waitcnt(/*lgkmcnt=*/0, /*vmcnt=*/-1));
__builtin_amdgcn_sched_barrier(0);
```

The `sched_barrier(0)` exists because cvt is a pure-SSA intrinsic. Without
the barrier, LLVM is free to hoist the cvt back above the s_waitcnt —
which then reads from a stale `fp8` vector. The KvManager has the same
construct; the QManager mirrors it for the same reason.

### 6.8 What's live after Phase 1

After Phase 1 completes (all 4 chunks done):

- `q_vgpr` v72..v103 holds $Q[:,0{:}256]$ in mfma A-operand layout.
- The 2 KiB/warp of staging LDS is **dead** — Phase 2 will immediately
  overwrite it as part of the same 8 KiB wave-private region.
- The E8M0 scale dwords have been consumed; their compiler-scratch
  vgprs are free.

No `__syncthreads()` is needed before Phase 2 because each wave only ever
read its own staging bytes.

## Chapter 7 — QManager Phase 2 (staging LDS → final Q-LDS, sb8 perm)

Phase 2 fills the **LDS half** of $Q$: $Q[:, 256{:}512]$ = the remaining
192 NoPE cols + 64 RoPE cols, into the 64 KiB per-WG Q-LDS region. Each
wave $w$ writes only into its own contiguous 8 KiB slice — that's the
**wave-major** invariant from §4.4. The same region overwrites the 2 KiB
Phase-1 staging without a barrier (intra-wave program order is enough).

### 7.1 Geometry and final layout

| Symbol | Value | Meaning |
|---|---:|---|
| `kLdsHalfCols` | 256 | bf16 cols in the LDS half |
| `kLdsHalfNopeCols` | 192 | NoPE cols (= 448 − 256) |
| `kLdsHalfRopeCols` | 64 | RoPE cols (= $D_{\mathrm{RoPE}}$) |
| `kP2ChunkCols` | 64 | cols per Phase-2 chunk |
| `kP2NumNopeChunks` | 3 | NoPE chunks at LDS-col [0, 64, 128] |
| `kSubBlockRows × kSubBlockCols` | 16 × 32 bf16 | one "sub-block" = a QK A-tile |
| `kSubBlockBytes` | 1024 | bytes per sub-block |
| `kWarpFinalBytes` | 8192 | bytes owned by one wave |
| `kFinalLdsBytes` | 65536 | total Q-LDS (8 waves × 8 KiB) |

The wave-major sub-block layout is:

$$
\mathit{subBlockByteOffset}(w, c) = w \cdot 8192 + c \cdot 1024
$$

where $w$ is the wave (row-tile) and $c \in [0, 8)$ is the col-tile index
in the wave's local 8-tile grid. Each wave owns the *contiguous* range
$[w \cdot 8192,\, (w+1) \cdot 8192)$ inside the 64 KiB region — the key
to no-barrier overlap with Phase 1 staging.

The 8 col-tiles per wave map to:

| col-tile $c$ | source | LDS cols held |
|---:|---|---:|
| 0 | Phase-2 NoPE chunk 0 (lo) | $[0, 32)$ |
| 1 | Phase-2 NoPE chunk 0 (hi) | $[32, 64)$ |
| 2 | Phase-2 NoPE chunk 1 (lo) | $[64, 96)$ |
| 3 | Phase-2 NoPE chunk 1 (hi) | $[96, 128)$ |
| 4 | Phase-2 NoPE chunk 2 (lo) | $[128, 160)$ |
| 5 | Phase-2 NoPE chunk 2 (hi) | $[160, 192)$ |
| 6 | Phase-2 RoPE chunk (lo) | $[192, 224)$ |
| 7 | Phase-2 RoPE chunk (hi) | $[224, 256)$ |

### 7.2 The sb8 permutation — why and how

The QK GEMM's `ds_read_b128_tr_b16` reader naturally lays out a 64-col
wave-tile in source col-element order $p \in [0, 64)$ — but at the
write side, naive `ds_write_b128` of 64 cols *as-is* hits a 2-way
`ds_write` bank conflict (the writer-side residue of the same Site C
collision we earlier mitigated on the read side).

The fix is a **sub-tile-of-8 permutation** ("sb8"). Treat each 64-col
wave-tile as 8 sub-tiles of width 8; store them in LDS in the order
$[0, 2, 4, 6, 1, 3, 5, 7]$. The reader is unaffected: the per-mfma K
rows just arrive in a permuted order, and matrix multiplication is
*commutative along the K reduction axis* — accumulating in a different
order yields the same fp32 sum (modulo rounding).

#### 7.2.1 Closed forms (forward and inverse)

For col-element $p \in [0, 64)$ (the lower 6 bits), decompose $p$ as
$(\mathit{sbD}, \text{inner3})$ where $\mathit{sbD} = (p \gg 3) \in [0,8)$
is the data sub-tile index, $\text{inner3} = p  \mathbin{\mathrm{and}}  7$ is the position
within the sub-tile.

**Forward perm** (data position → LDS position), bit-form:

$$
L = (p  \mathbin{\mathrm{and}}  7) \,\big|\, \Big(\big((p \gg 3)  \mathbin{\mathrm{and}}  1\big) \ll 5\Big) \,\big|\, \Big(\big((p \gg 3)  \mathbin{\mathrm{and}}  6\big) \ll 2\Big) \,\big|\, (p  \mathbin{\mathrm{and}}  \sim 0\mathrm{x}3F)
$$

Equivalently: **swap bits [3] and [5]** of $p$. The trailing
$(p  \mathbin{\mathrm{and}}  \sim 0\mathrm{x}3F)$ passes bits ≥ 6 through unchanged (those
indices live above one wave-tile).

Source: `sb8_perm_col_elems()` at `hk_mla_v40_buffer_managers_gen1.cuh`
lines 37–48.

**Inverse perm** (LDS → data), bit-form:

$$
p = (L  \mathbin{\mathrm{and}}  7) \,\big|\, \Big(\big((L \gg 5)  \mathbin{\mathrm{and}}  1\big) \ll 3\Big) \,\big|\, \Big((L  \mathbin{\mathrm{and}}  0\mathrm{x}18) \ll 1\Big) \,\big|\, (L  \mathbin{\mathrm{and}}  \sim 0\mathrm{x}3F)
$$

(`sb8_inv_perm_col_elems()`). Not an involution — the sub-tile perm
$[0,2,4,6,1,3,5,7]$ has inverse $[0,4,1,5,2,6,3,7]$.

#### 7.2.2 Forward perm table on $\mathit{sbD}$

| data $\mathit{sbD}$ | $0$ | $1$ | $2$ | $3$ | $4$ | $5$ | $6$ | $7$ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| LDS $sb_L$ | $0$ | $4$ | $1$ | $5$ | $2$ | $6$ | $3$ | $7$ |

Reading down the table: the *data* sub-tiles in even positions
($0, 2, 4, 6$) land in the *first half* of LDS sub-tiles
($0, 1, 2, 3$); the odd-positioned data sub-tiles ($1, 3, 5, 7$) land
in the *second half* ($4, 5, 6, 7$).

#### 7.2.3 Where the perm must apply (and where it doesn't)

| Side | Perm site | Reason |
|---|---|---|
| **LDS-dst writers** (manager controls the LDS dst address) | Forward `sb8_perm_col_elems` on the LDS dst col-element index | Manager-controlled writes can choose the dst; we put the perm here. |
| **`buffer_load_lds` writers** (HW fixes the LDS dst pattern) | Forward `sb8_perm_col_elems` on the **vmem-src col**, via permuted `v_offset` | `buffer_load_lds` always writes lane $T$ to LDS $T \cdot 16$. Algebra: permuting the source col is equivalent to permuting the dst, since the HW dst pattern is a bijection. |
| **PV reader (the V-side of the wave-tile)** | None | Reading from a permuted layout for K's reduction axis is fine — see 7.2.4. |
| **OManager epilogue** (final VRAM write) | Inverse `sb8_inv_perm_col_elems` on the per-lane VRAM col | Un-swizzles so the user sees natural col order. Ch. 11. |

#### 7.2.4 Why Q's D-axis must get the SAME perm as K's

This is the gotcha captured in `[[v40-sb8-perm-qk-reduction-axis]]`. A
naïve view would split: "permute K's LDS for bank-conflict win, leave
Q alone — they're independent operands." That's **wrong**.

QK is a reduction over the $D_{\mathrm{NoPE}}$ axis. The mfma
$\sum_d Q_{m,d} \, K_{n,d}$ requires Q and K to be addressed by the
*same* $d$ for each accumulation step. If K's $d$-axis is permuted in
LDS but Q's isn't, then mfma step $k$ multiplies $Q_{m,k}$ against
$K_{n, \mathrm{perm}(k)}$ — wrong product entirely.

So the sb8 perm must apply identically to both K's and Q's D-axes. PR-A
("KV-only" perm) is structurally impossible.

### 7.3 Phase 2 NoPE writer: `p2_vmem_to_vgpr_nope_chunk` + `p2_cvt_store_nope_chunk`

This is the V40 mirror of `KvManager8to16bitsV1::cvt_and_store_kv_tile`.
Split into two halves for double-buffering across chunks:

- `p2_vmem_to_vgpr_nope_chunk` issues 1× `buffer_load_dwordx4` (16 fp8)
  + 1× `buffer_load_ubyte` (E8M0 scale) per lane and returns the dwords.
- `p2_cvt_store_nope_chunk` drains `vmcnt(0)`, runs 8 cvts to bf16, and
  issues 2× `ds_write_b128` with the sb8 + Site-C compose.

**Per-lane vmem offset (NoPE):**

$$
\mathit{vOffNope}(\ell) = (\ell \gg 2) \cdot 512 + (\ell  \mathbin{\mathrm{and}}  3) \cdot 16, \quad \mathit{iOff} = 256 + 64 c
$$

with chunk $c \in \{0,1,2\}$. The vmem side is **straight** (no
swizzle) — the bank-conflict swizzle is on the LDS-write side instead,
mirroring KvManager.

**Per-lane vmem offset (scale):**

$$
\mathit{vOffScale}(\ell) = (\ell \gg 2) \cdot 512 + ((\ell  \mathbin{\mathrm{and}}  3) \gg 1), \quad \mathit{iOff} = 448 + c \cdot 2
$$

The `(\ell  \mathbin{\mathrm{and}}  3) \gg 1` gives 0 for col_group 0/1 and 1 for col_group
2/3 — both ds_write halves within a chunk share one scale (V4 packs one
E8M0 per 64-col tile, dup'd to 2 bytes).

**LDS write address (with sb8 forward perm + Site C row-XOR):**

After the cvt, each lane has 8 bf16 dwords (`lo_dw[0..3]` + `hi_dw[0..3]`)
covering its 16 fp8 inputs. The sb8 perm assignment is:

| lane's `col_group` | what `lo_dw` covers (data) | what `hi_dw` covers (data) | LDS col-tile target |
|---:|---|---|---:|
| 0 | data sub-tile 0 | data sub-tile 1 | `kColTileBase + 0` / `+1` |
| 1 | data sub-tile 2 | data sub-tile 3 | `kColTileBase + 0` / `+1` |
| 2 | data sub-tile 4 | data sub-tile 5 | `kColTileBase + 0` / `+1` |
| 3 | data sub-tile 6 | data sub-tile 7 | `kColTileBase + 0` / `+1` |

The reason: under sb8 forward perm, data sub-tiles $\{0,2,4,6\}$ all
land in LDS sub-tiles $\{0,1,2,3\}$ (the first half =
`sb_in_chunk = 0` = `kColTileBase`), and $\{1,3,5,7\}$ all land in
$\{4,5,6,7\}$ (the second half = `kColTileBase + 1`). So `lo_dw` and
`hi_dw` differ only by `+kSubBlockBytes = 1024 B` in the LDS imm
offset — one address VGPR + two `ds_write_b128`s.

**Site C row-XOR** (composes on top of sb8, disjoint bit):

$$
\mathit{byteInSbSwz} = (\mathit{colGroup} \ll 4) \oplus \big(((\mathit{rowInWarp} \gg 2)  \mathbin{\mathrm{and}}  1) \ll 5\big)
$$

The row-conditional XOR (rows 4..7 and 12..15 get bit 5 of
`byte_in_sb` flipped) operates on bit 5 of the sub-block byte address;
sb8 perm operates on bit 5 of the *col-element* index (which lives in
the col-tile selection, not the byte position). The bits are disjoint
so the two compositions don't clash.

Final per-lane LDS dst:

$$
\text{addr}(\ell) = \mathit{pLdsQ} + \mathit{subBlockByteOffset}(w, \text{kColTileBase}) + \mathit{rowInWarp} \cdot 64 + \mathit{byteInSbSwz}
$$

Two `ds_write_b128` are issued at this address, separated by immediate
offset `kSubBlockBytes = 1024 B` — the lo store at offset 0, the hi
store at offset 1024. One address VGPR, two writes, no second add.

### 7.4 Phase 2 RoPE writer: `p2_load_rope_chunk`

RoPE is bf16 already — no cvt. Two `buffer_load_dwordx4 lds:` cover
the 64-col tile (lo = cols [0,32), hi = cols [32,64), landing at LDS
col-tiles 6 and 7).

`buffer_load_lds` HW-fixes the LDS dst pattern (lane $T \to$ LDS
$T \cdot 16$), so the sb8 perm must apply on the **vmem-src col side**
instead:

$$
\mathit{colQuadSwz} = \mathit{colQuad} \oplus \big(((\mathit{rowInWarp} \gg 2)  \mathbin{\mathrm{and}}  1) \ll 1\big)
$$

(this is the Method-2 row-conditional half-swap; same Site C row pattern
as the NoPE writer above, just applied on the source side because the
dst is HW-fixed.)

$$
\mathit{vOffLo}(\ell) = \mathit{rowInWarp} \cdot 128 + \mathit{colQuadSwz} \cdot 32
$$

with `kRopeStride = 128 B = 64 bf16 cols`. The hi load shares
`v_off_lo` and uses `i_off = 16` (the +16 byte delta to reach
cols [8,16) within col_quad's 32-byte slot); the lo and hi LDS dsts
target col-tiles 6 and 7 respectively, with the hi dst
pre-subtracted by 16 to cancel the +16 imm offset on the LDS side.

### 7.5 What's live after Phase 2

After Phase 2 completes (3 NoPE chunks + 1 RoPE chunk):

- 64 KiB Q-LDS holds $Q[:, 256{:}512]$ in bf16, wave-major sub-block
  layout with the sb8 forward perm applied along the D-axis.
- Phase 1 staging is fully overwritten (the first 2 KiB of each wave's
  8 KiB slice).
- `q_vgpr` (v72..v103) holds $Q[:, 0{:}256]$ from Phase 1 — also with
  the sb8 perm applied (Step 2's reader used `C_phys = C_log XOR (S<<1)`,
  which on the per-chunk data layout is equivalent to the sb8 perm
  applied identically to the D-axis).

Both Q halves are ready for the main loop's QK Phase A (VGPR half) and
Phase B (LDS half). The KV side will apply the matching sb8 perm on K's
D-axis (Ch. 8) so QK accumulation is correct.

## Chapter 8 — KvManager double-buffered pipeline

KV is the dominant bandwidth consumer and the only inter-iteration LDS
resident. The KvManager hides VMEM latency via a **double-buffer pong**
scheme: while iter $i$'s QK/PV reads the *current* pong, Phase A prefetches
the *next* tile (into VGPR carriers + raw-fp8 staging LDS) and Phase B
cvt+stores it into the *next* pong.

> **V2 vs V1.** m16x8 uses `KvManager8to16bitsV2` (this section). m16x4 uses
> `KvManager8to16bitsV1` — the "Option 2" wave→tile map, kBlockN=32, single
> per-lane carrier — which the earlier revisions of this chapter described.
> V2 is derived from V1 (`class …V2 : public …V1`) and changes: (1) kBlockN=64
> double-tile, (2) a **band-per-warp** remap (each warp owns one 16-row band ×
> one 256-col tile → **one** `row_kv_ld` per lane, 2 warp types), (3) a
> **staging** path (strips 2,3 go vmem→raw-fp8-LDS via `buffer_load_lds`,
> converted later), and (4) **deferred strip-3** (§8.13).

### 8.1 Geometry and pong layout (V2, kBlockN=64)

| Symbol | Value | Source |
|---|---:|---|
| `kBlockN` | **64** | rows per KV tile = two 32-row sub-tiles A, B |
| `kSubPong` | 32768 | bytes of one 32-row sub-tile in a pong (B at `+kSubPong`) |
| `kQkNopeHeadDim` / `kQkRopeHeadDim` | 448 / 64 | fp8 NoPE / bf16 RoPE cols per token |
| `kQkHeadDim` | 512 | $= D_{\mathrm{QK}}$ bf16 cols per row in LDS |
| `kSubBlockRows × kSubBlockCols` | 16 × 32 bf16 | one sub-block (= one QK A-tile), 1024 B |
| `kTileCols` / `kColTilesPerTile` | 256 / 8 | one 256-col **tile**; col-tiles within it |
| `kWaveTileCols` | 64 | per-strip col width (a warp's band = 4 strips × 64) |
| **One pong** | $64 \cdot 512 \cdot 2 = $ **64 KiB** | full $\mathrm{kBlockN} \cdot D_{\mathrm{QK}}$ in bf16 |

Each 32-row sub-tile is stored col-major-sub-block exactly as V1
($\mathit{subBlockByteOffset}(r_{\mathrm{tile}}, c_{\mathrm{tile}}) =
(c_{\mathrm{tile}}\cdot 2 + r_{\mathrm{tile}})\cdot 1024$); sub-tile B is the
same layout at `+kSubPong`.

### 8.2 Warp → band map (V2)

Each of the 8 warps owns **one 16-row band × one 256-col tile** of the
64×512 KV tile (contrast V1, where a warp owned a 16×64 wave-tile in *both*
column halves and thus needed two row indices). With `band = w & 3`:

| quantity | expr | meaning |
|---|---|---|
| `band` | `w & 3` | rows `[band·16, +16)` of the 64-row tile |
| `tile` | `w >> 2` | 0 = cols 0–255 (Lo, pure NoPE); 1 = cols 256–511 (Hi, NoPE+RoPE) |
| `sub_off` | `((w>>1)&1)·kSubPong` | which 32-row sub-tile (A/B) this band lands in |
| `row_tile` | `w & 1` | 16-row half within the sub-tile |
| `row_kv_ld` base | `lane>>2` | the lane's row inside its band (0..15) |

A warp's 16×256 band = **4 col-strips of 16×64** = 4 dwordx4/lane (fp8).
Lo warps (0–3, tile 0) are pure NoPE. Hi warps (4–7, tile 1) are
NoPE(strips 0–2) + RoPE(strip 3, cols 448–511). `wave_is_rope_owner(w) =
(w >= 4)`.

**The key invariant:** warp $i$ and warp $i{+}4$ share `band`, `sub_off`,
`row_tile` and hence the *same* `row_kv_ld` — they differ only in `tile`.
This is what lets deferred / borrowed strip stores land in the correct pong
sub-block (§8.13).

### 8.3 The pong swap

Two LDS pointers, swapped each iter (`std::swap`, no data movement):

| Iter $i$ parity | `p_lds_kv_curr` | `p_lds_kv_next` |
|---|---|---|
| even | LDS base + 0 | LDS base + 64 KiB |
| odd | LDS base + 64 KiB | LDS base + 0 |

At iter $i$ entry, `p_lds_kv_curr` holds the finished tile to compute on;
`p_lds_kv_next` receives this iter's Phase B cvt+store. Swapped at iter end.

### 8.4 Carrier / staging / consume timeline (V2)

Each warp's 4 band strips split by transport:

| strips | transport | landing | consumed |
|---|---|---|---|
| 0, 1 | `prefetch_kv_nope` → VGPR **carriers** `p0/p1` (`buffer_load_dwordx4` + `ubyte` scale) | pinned VGPR | Phase B: cvt+store to pong |
| 2 (both), 3 (lo) | `prefetch_kv_nope_lds` → **raw-fp8 staging LDS** via `buffer_load_lds` (+ ubyte scale) | staging LDS | Phase B: `load_staged_kv_carrier` → cvt+store |
| 3 (hi) | `prefetch_kv_rope` → **direct** vmem→pong LDS | pong (bf16 already) | — (no cvt) |

Phase A issues carriers + staging before the pre-QK `s_barrier` so their
vmem latency overlaps the barrier + QK. Phase B, after QK, drains them with
graduated `wait_kv_loads<kIsRopeWarp, kVmCnt>` (vmcnt 6/4/2/0), converts
fp8→bf16 (`cvt_kv_tile_step`), and `store_kv_tile_step`s into
`p_lds_kv_next`. Lo warp strip 3 is **deferred** to the next iter (§8.13).

The prologue wrapper `async_load_k` does all four strips non-overlapped
directly into the current pong (no staging, no deferral) — this is why the
warp's first compute iter needs no deferred strip-3.

> **§8.5-8.12 note.** These sub-sections detail the shared V1 address
> primitives — the vmem address split, the Method-2 bank-conflict swizzle,
> `get_kv_ld_row`, `load_k_to_gpr`, RoPE DMA, the boundary carry. V2 reuses
> all of them; it only changes *which* strips go through carriers vs staging
> and adds deferred strip-3 (§8.13). The `prefetch_kv_tile` name below is the
> V1 entry; V2's `prefetch_kv_nope` / `prefetch_kv_nope_lds` are the same
> address math specialized by `<kColStrip, kTile>` template args, landing in a
> VGPR carrier or the staging LDS respectively.

### 8.5 NoPE prefetch — `prefetch_kv_tile` (NoPE branch)

Address split:

| Field | Per-lane / wave-uniform / immediate | Expression |
|---|---|---|
| `v_offset` (per-lane) | NoPE fp8 | $\mathit{rowKvLd} \cdot 512 + \mathit{colGroupSwz} \cdot 16$ |
| `s_offset` (wave-uniform) | NoPE fp8 | $c_{tileInHalf} \cdot 64$ |
| `i_offset` (immediate) | NoPE fp8 | $\text{kTileIdx} \cdot 256$ |
| `v_offset` (per-lane) | scale | $\mathit{rowKvLd} \cdot 512$ |
| `s_offset` (wave-uniform) | scale | $c_{tileInHalf} \cdot 2$ |
| `i_offset` (immediate) | scale | $448 + \text{kTileIdx} \cdot 8$ |

Two key choices:

1. **`row_kv_ld` is per-lane and must live in `v_offset`.** Each lane
   covers a distinct row of the 32-row KV tile (`row_kv_ld` is set up by
   `get_kv_ld_row_base_idx` + the page-index lookup in `get_kv_ld_row`,
   see 8.7). Routing `row_kv_ld` via `s_offset` would force
   `v_readfirstlane` and collapse all lanes onto row 0 — wrong by
   construction.
2. **The bank-conflict swizzle (Method 2) is on the vmem-load side.**
   For rows whose sub-tile-row bit is set
   (rows 4..7, 12..15, i.e. `(lane>>4)&1 == 1`), swap the 16 B chunk
   with the in-pair neighbour:

   $$
   \mathit{colGroupSwz} = \mathit{colGroup} \oplus \big(((\ell \gg 4)  \mathbin{\mathrm{and}}  1) \ll 1\big)
   $$

   Pairs with the matching XOR on `load_k_to_gpr`'s reader, and lets
   `cvt_and_store_kv_tile`'s LDS dst address stay straight — same
   pattern QManager Phase 2 ships.

### 8.6 NoPE cvt+store — `cvt_kv_tile_step` + `store_kv_tile_step`

After `wait_kv_loads<…, vmcnt=0>`, the carrier `KvTilePrefetch::nope_dw`
holds 4 fp8 dwords/lane. Four cvt steps produce 4 bf16 dwords/lane in a
single carrier `dw`:

| kStep | source | dst dwords |
|---:|---|---|
| 0 | `nope_dw[0]` (low + high fp8 pair) | `dw[0]`, `dw[1]` |
| 1 | `nope_dw[1]` | `dw[2]`, `dw[3]` |
| 2 | `nope_dw[2]` | `dw[0]`, `dw[1]` (overwrite lo carrier — safe: lo ds_write issued already) |
| 3 | `nope_dw[3]` | `dw[2]`, `dw[3]` |

Steps 0..1 fill the lo half, then `store_kv_tile_step<R, C, 0>` issues
the lo `ds_write_b128`. Steps 2..3 fill the hi half (reusing `dw`), then
`store_kv_tile_step<R, C, 1>` issues the hi `ds_write_b128` at imm
offset `kNumRowTiles * kSubBlockBytes = 2048 B`.

LDS write address (per-lane):

$$
\text{addr}(\ell) = p_{ldsKv} + \mathit{subBlockByteOffset}(r_{\mathrm{tile}}, c_{tileGlobalLo}) + (\ell \gg 2) \cdot 64 + (\ell  \mathbin{\mathrm{and}}  3) \cdot 16
$$

with

$$
c_{tileGlobalLo} = \text{kTileIdx} \cdot 8 + c_{tileInHalf} \cdot 2.
$$

Note this is the **straight** address — no swizzle on the LDS dst side
here, because the writer-side sb8 perm is *baked into the wave→tile
partition*: under Option 2 (8.2), each wave owns col-tiles
$(c_{tileInHalf} \cdot 2, c_{tileInHalf} \cdot 2 + 1)$,
and the column reordering across waves
$\{0,1,2,3\} \mapsto \{0,1,2,3\}, \{4,5,6,7\} \mapsto \{4,5,6,7\}$
within each half is the structural sb8 permutation. The 64-cols-per-wave
chunk preserves accumulation order along K's D-axis because Q's D-axis
gets the same partition.

### 8.7 Row lookup: `get_kv_ld_row_base_idx` + `get_kv_ld_row`

`row_kv_ld` is the **physical row number** in the flat KV-token space for
this lane's row of the 32-row tile. Two-step lookup:

1. **Per-lane local row in the tile**, set by
   `get_kv_ld_row_base_idx(warp_idx)`:

   $$
   \mathit{rowBaseIdx}(\ell, w) = (((w \gg 1)  \mathbin{\mathrm{and}}  1) \cdot 16) + (\ell \gg 2)
   $$

   This is just the lane's row within the 32-row tile (0..15 for upper
   half waves, 16..31 for lower).

2. **Page-index resolution**, set by
   `get_kv_ld_row<kCheckBoundary, kPageSize>(p_kv_indices, row_base_idx, kv_tile_start, kv_tile_end)`:

   - For `kPageSize == 1`: directly load $p_{kvIndices}[\mathit{rowBase} + \mathit{kvTileStart}]$.
   - For `kPageSize > 1`: split into $(\mathit{pageIdx}, \mathit{intraPage})$, look up the physical page number, return $\mathit{pagePhys} \cdot \text{kPageSize} + \mathit{intraPage}$.

   If `kCheckBoundary == true` and the global row index exceeds
   `kv_tile_end`, returns **−1**. The prefetcher then writes zeros (no
   vmem issue) — see `in_bounds` gates in `prefetch_kv_tile`.

This is the helper that was lifted to `hk_mla_utils.cuh` (inside
`namespace hk_mla`) so both V32 and V40 share one implementation.

### 8.8 RoPE prefetch — direct vmem → LDS

For waves 5 and 7 on `kTileIdx == 1`, RoPE prefetch issues two
`buffer_load_dwordx4 lds:` calls covering the 16×64 bf16 RoPE patch as
two 16×32 sub-blocks at LDS col-tiles 14 and 15.

Address split:

| Field | Expression |
|---|---|
| `v_offset` lo (per-lane) | $\mathit{rowKvLd} \cdot 128 + \mathit{colGroupSwz} \cdot 32$ |
| `i_offset` hi | 16 (the +16 B delta to reach cols [8,16) within col_quad) |
| LDS dst (lo, per-lane) | $p_{ldsKv} + \mathit{subBlockByteOffset}(r_{\mathrm{tile}}, 14) + \ell \cdot 16$ |
| LDS dst (hi, per-lane, **pre-subtracted**) | $p_{ldsKv} + \mathit{subBlockByteOffset}(r_{\mathrm{tile}}, 15) + \ell \cdot 16 - 16$ |

The pre-subtract on the hi dst cancels the +16 `i_offset` on the LDS
side (which advances both vmem AND LDS), so the hi load actually
lands at col-tile 15.

`buffer_load_lds` HW-fixes the LDS dst pattern (lane $T \to T \cdot 16$),
so the sb8 row-conditional swizzle must apply on the **vmem-src col**
side too:

$$
\mathit{colGroupSwz} = \mathit{colGroup} \oplus \big(((\ell \gg 4)  \mathbin{\mathrm{and}}  1) \ll 1\big)
$$

(Method 2 — same row pattern as the NoPE Method-2 vmem-side swizzle.)

The previous bug `[[v40-rope-prefetch-shared-m0-bug]]` (paraphrased from
the in-source comment): a single shared M0 with `i_off=0` and `i_off=16`
overlapped the two calls' lane slots — call 2 wrote each lane $T$ at
$M0 + (T+1) \cdot 16$, leaving sub-block 15 unwritten. The pre-subtract
fix above resolves this.

### 8.9 Consumer: `load_k_to_gpr`

QK mfma A-tile loader. Issues one `ds_read_b128` per call:

$$
\text{addr}(\ell) = p_{ldsKv} + \mathit{subBlockByteOffset}(\text{kRowOffset}/16, \text{kColOffset}/32) + \text{row} \cdot 64 + (\text{col} \cdot 2 \oplus \mathit{rowBankSwap})
$$

with `row = lane % 16`, `col = (lane / 16) * 8`, and

$$
\mathit{rowBankSwap} = ((\text{row} \gg 2)  \mathbin{\mathrm{and}}  1) \ll 5
$$

The XOR on bit 5 of the col-byte component is the **reader half** of
Method 1 (writer's row-conditional XOR on bit 5 of `byte_in_sb`). Same
pattern QManager Phase 2 ships.

`load_transposed_v_to_gpr` (for PV) is the same shape but uses
`ds_read_b64_tr_b16` (transpose read) — covered in Ch. 10.

### 8.10 The prefetch chain in the main loop

Each iter pushes a 3-deep state machine. At iter $i$ entry,
`p_lds_kv_curr` holds tile $i$ (drained by prior iter's wait) and
`p_lds_kv_next` holds tile $i{+}1$ in flight (prefetched at iter $i{-}1$).
During iter $i$:

1. **softmax → PV** on tile $i{-}1$'s data (still in `p_lds_kv_curr`
   from prev iter's perspective, now read for V).
2. **QK Phase A** on tile $i$ (Q from `q_vgpr` × K from `p_lds_kv_curr`).
   Interleaved with: `prefetch_kv_tile<…, kCheckBoundaryNext>` of tile
   $i{+}2$ into `p_lds_kv_next`, and `cvt_kv_tile_step` /
   `store_kv_tile_step` for tile $i{+}1$.
3. **QK Phase B** on tile $i$ (Q from LDS × K from `p_lds_kv_curr`).

At iter $i$ exit: swap `p_lds_kv_curr` and `p_lds_kv_next`.

So at any moment **four** tiles are "in flight" — $i{-}1$ (PV reading
the now-stale curr), $i$ (QK reading curr), $i{+}1$ (cvt+store filling
next), $i{+}2$ (vmem prefetch into VGPR carrier). The double pong
holds two; the `KvTilePrefetch` VGPR carrier holds the third.

### 8.11 Boundary handling — the slim-dispatch carry update

`mla_main`'s template params include `kCheckBoundaryNext` — when set,
`prefetch_kv_tile` runs with `kCheckBoundary = true`, calling
`get_kv_ld_row<true, ...>` which returns −1 for OOB rows and zero-fills
the carrier on those lanes.

Slim dispatch (Ch. 12) collapsed the per-iter `kCheckBoundaryNext`
branch by always passing `true`. The correctness fix it required: the
`row_kv_ld_next_next` carry — which remembers the resolved physical
row for the iter-after-next's prefetch — used to be gated on
`kCheckBoundaryNext == false`, so the always-true slim path never
updated it and subsequent iters re-prefetched from a stale row. The
carry is now gated on `kIsGlobalLast == false` instead; it updates on
every non-last iter, slim or not. See Ch. 12.

### 8.12 The wait-with-skip pattern

`wait_kv_loads<kRowOffset, kColOffset, kVmCnt>` issues the s_waitcnt +
sched_barrier on every wave **except** the two RoPE owners on the RoPE
half-tile (`kTileIdx == 1`, $w \in \{5, 7\}$). Those two waves'
`buffer_load_lds` traffic is synchronized later by an `s_barrier` (the
QK consumer reads from LDS, so the cross-wave sync point is the QK
barrier itself, not the per-tile wait). Skipping saves a few cycles per
iter for those waves.

Similarly, `store_kv_tile_step<…, kTileIdx=1>` early-returns for those
two waves — their RoPE path has no `ds_write`, only the direct
vmem→LDS that prefetch already issued.

### 8.13 Deferred strip-3 (V2, lo warps) — cross-iter software pipeline

Lo warps stage **two** NoPE strips (2 and 3) into raw-fp8 LDS in Phase A.
Converting both in the same iter's Phase B crowds the (already busy) cvt+store
block. So strip 3's **consume** (ds_read staging → cvt → store to pong) is
deferred by one iteration:

- **Iter N Phase A** issues the strip-3 `buffer_load_lds` (into staging slot 1)
  and its e8m0 scale into the *carried* VGPR `s3_scale`. Iter N Phase B
  cvt+stores strips 0,1,2 only — strip 3's two loads stay in flight.
- **Iter N+1 top**, *before* Phase A re-stages slot 1: `wait_kv_loads<…,0>` →
  `load_staged_kv_carrier<1>` → cvt (using `s3_scale`) → `store_kv_tile_step
  <3, tile0>` into `p_lds_kv_curr` (the tile this iter's QK is about to read,
  already swapped in). The store is drained by the existing pre-QK
  `lgkmcnt(0)` before the `s_barrier`, so QK sees a complete tile.

Why it's correct:

- **Only the scale carries in a VGPR.** The bulk NoPE data stays in private
  staging LDS (per-warp, warp-invariant address) — nothing else is hoisted.
- **Destination is `p_lds_kv_curr`.** Iter N staged the tile that becomes
  `curr` after iter N's swap; strip 3 completes that same tile.
- **Gate is `!kIsRopeWarp && !kIsFirstIter`.** The first compute iter's
  tile-0 strip 3 came from the prologue (`async_load_k`), so nothing is
  pending; every later iter (including the last/skip/epilogue iter) has a
  pending strip 3 from its predecessor and must flush it for its own QK.
- **WAR safety.** The iter N+1 consume ds_reads slot 1 and completes its
  `lgkmcnt(0)` *before* iter N+1 Phase A re-issues the strip-3
  `buffer_load_lds` into the same slot — program order guarantees no clobber.

Net effect: strip 3's vmem load gets a full extra iteration to retire (fully
hidden), and its cvt+store moves off Phase B onto the lighter iter-top window.

## Chapter 9 — Softmax

Online (Flash-style) softmax runs **once per KV tile**, between QK and PV.
It updates two per-row running scalars ($m$, $\ell$) and produces the fp32
P-tile in `p_comp` (v48..v63, **16 fp32/lane** at kBlockN=64 — a 16×64 tile),
then packs it into bf16 `p_mfma` (v48..v55 overlay) for PV.

> **kBlockN=64 + `kUsePk` update (m16x8).** With kBlockN=64 the P-tile is 16×64,
> so `p_comp` = `kBlockN/4 = 16` fp32/lane and the packed `p_mfma` = `kBlockN/8
> = 8` dwords/lane. The kernel uses the `_16`-width routines
> (`softmax_mask_p`, `softmax_p1_prescaled_16`, `max_16`), and each takes a
> `kUsePk` template arg. In softmax, `kSoftmaxUsePk = kPvAtEnd`, so **lo warps
> run packed** (`v_pk_*`) and **hi warps run de-packed** (`v_add_f32_e32` /
> `v_mov_b32` / `v_exp_f32` scalars). (The oaccu-normalize at epilogue is the
> opposite — `kFinalRescaleUsePk = !kPvAtEnd`, lo de-packed / hi packed, §10 —
> so each warp group carries exactly one packed phase.) Where the older text
> says "8 fp32 / 16×32" read "16 fp32 / 16×64"; where it says v120.. read v48..

### 9.1 The online recurrence

For each new tile $i$ producing local $S^{(i)}$:

$$
\begin{aligned}
m^{\mathrm{loc}} &= \max_j S^{(i)}_{:, j} & &\text{(per-row local max)} \\
m^{\mathrm{new}} &= \max(m^{\mathrm{old}}, m^{\mathrm{loc}}) & &\text{(running max)} \\
\alpha           &= \exp_2\big((m^{\mathrm{old}} - m^{\mathrm{new}}) \cdot \log_2 e\big) & &\text{(rescale factor)} \\
P^{(i)}          &= \exp_2\big((S^{(i)} - m^{\mathrm{new}}) \cdot \log_2 e\big) \\
\ell^{\mathrm{new}} &= \alpha \cdot \ell^{\mathrm{old}} + \sum_j P^{(i)}_{:, j} & &\text{(running denominator)} \\
\mathrm{oaccu}^{\mathrm{new}} &= \alpha \cdot \mathrm{oaccu}^{\mathrm{old}} + P^{(i)} V^{(i)} & &\text{(rescale + PV in Ch. 10)}
\end{aligned}
$$

Note the use of $\exp_2$ — gfx950 has `v_exp_f32` (base-2) but not a
native `v_exp_f32_e_base_e`. The standard trick: scale input by
$\log_2 e \approx 1.4426950408889634$ (the `log2e` constant) so
`v_exp_f32(x · log2e)` produces $e^x$.

On the **first iter** ($\mathrm{kIsFirstIter} = \mathrm{true}$):
$\alpha = 1$, $m^{\mathrm{old}}$ is treated as $-\infty$ via
`new_row_max = local_max`, and `oaccu`'s rescale is skipped (PV's mfma
initializes `oaccu` fresh with a 3-arg, non-accumulating form).

### 9.1.1 Deferred rescale (skip when the max barely moves)

Rescaling `oaccu` by $\alpha$ is a full-width pass over the 128-vgpr
`oaccu` tile every tile. But $\alpha = \exp_2((m^{\mathrm{old}} -
m^{\mathrm{new}}) \log_2 e)$ is **exactly 1** when $m^{\mathrm{new}} =
m^{\mathrm{old}}$, and *negligibly* below 1 when the max rises only
slightly. The softmax result $\mathrm{oaccu}/\ell$ is invariant to the
common reference $m$ as long as that reference is (a) shared by numerator
and denominator and (b) large enough that $\exp_2(S - m)$ does not
overflow fp32. So $m$ may be kept **stale** for as many tiles as the
overflow budget allows, deferring the rescale.

The kernel exploits this with a compile-time threshold
`T::kRescaleThreshold` (in `HkMlaV40DecodeFwdTraits`, logit units):

- Per lane, `local_max - row_max > kRescaleThreshold` decides whether
  *this lane's* rows need a rescale.
- The decision is promoted to a **wave-uniform** flag via
  `__builtin_amdgcn_ballot_w64(...) != 0`. `oaccu` is rescaled per wave
  (one $\alpha$ branch for the whole wave), but each lane owns different
  rows — so the rescale is only safe to skip when **every** active lane
  is under threshold.
- When the flag is false: keep `row_max` stale, set $\alpha = 1$, and
  skip the `oaccu` + `row_sum_e` rescale entirely. $P^{(i)} =
  \exp_2(S^{(i)} - m^{\mathrm{stale}})$ accumulates against the existing
  reference, staying consistent with the un-rescaled `oaccu`/$\ell$.
- When true: do the real rescale and advance $m^{\mathrm{new}} =
  \max(m^{\mathrm{old}}, m^{\mathrm{loc}})$, restoring headroom.

The default `kRescaleThreshold = 8.0` defers until the max would move by
more than $e^8 \approx 2981\times$ — far under the $e^{88}$ fp32 overflow
wall, so the fp32 accumulator's extra dynamic range stays well within
mantissa precision. The skip path is selected by a separate `kDoRescale`
instantiation of the PV gemm (§10), which omits all the interleaved
rescale multiplies. Setting the threshold $< 0$ disables the
optimization (every tile rescales).

### 9.2 Where $m$ and $\ell$ live

| State | Storage | Notes |
|---|---|---|
| $m$ (per row, this lane's share) | `float row_max;` local | 1 fp32/lane, persists across iters |
| $\ell$ (per row, this lane's share) | `float row_sum_e;` local | 1 fp32/lane, persists across iters |
| $S^{(i)}$ / $P^{(i)}$ | `p_comp` v48..v63 | 16 fp32/lane, 16×64 tile per warp |
| $P^{(i)}$ in bf16 for PV | `p_mfma` v48..v55 (overlay on p_comp low half) | 8 bf16x2 dwords/lane |
| $\alpha$ (rescale for this iter) | `float rescale;` local | 1 fp32/lane, lives only within this iter |

These compiler-scratch fp32 scalars sit in v0..v35 and are recreated
each iter (LLVM is free to choose where).

### 9.3 The softmax routines (m16x8, `_16` + `kUsePk`)

From `hk_mla_softmax.cuh` (all `_16`-width; each takes `kUsePk`):

| Routine | Inputs / Outputs | What it does |
|---|---|---|
| `softmax_mask_p<kCheckBoundary, GPR, kUsePk>(...)` | reads p_comp, writes p_comp | Applies the prescaled softmax scale and, on boundary tiles, fills out-of-range cols with −inf (via `set_ninf1/2<kUsePk>`) so exp → 0. De-packed when `kUsePk=false`. |
| `max_16<…>` + `warp_reduce` (inlined) | reads p_comp, writes row_max, rescale | Local row max (v_max3 ladder over 16, no `v_pk`) + cross-lane reduce; updates $m$, computes $\alpha$ (deferred-rescale threshold, §9.1.1). |
| `softmax_p1_prescaled_16<kIsFirstIter, k_p_comp_begin, comp_t, kUsePk>(...)` | reads p_comp + new_row_max + rescale, writes p_comp, row_sum_e | In-place `exp_2`, then row-sum reduce → running $\ell$. Packed or de-packed per `kUsePk` (§9.5). |

`kSoftmaxUsePk = kPvAtEnd` (true for lo warps): **lo = packed, hi =
de-packed** for the softmax block. The kernel inlines the row-max branch so
`kCheckBoundary` is a runtime branch around just `softmax_mask_p`.

### 9.4 The v_max3 ladder

At kBlockN=64, `max_16` reduces the per-lane local max over **16** fp32
values with a `v_max3_f32` tree (each `v_max3` folds 3 inputs), the
gfx950-minimum for 16 inputs. The kBlockN=32 form over 8 values was 4
instructions:

$$
\mathit{localMax} = \max\big(\max_3(\mathit{p0}, \mathit{p1}, \mathit{p2}), \max_3(\mathit{p4}, \mathit{p5}, \mathit{p6}), \max(\mathit{p3}, \mathit{p7})\big)
$$

`max_16<>` (m16x8) / `max_8<>` (m16x4) in `hk_mla_utils.cuh` ship this
ladder; the kernel inlines the same shape. After the per-lane reduce,
`warp_reduce<MaxFunctor>`
reduces across the 4 lanes per row-group via DPP / cross-lane permutes
— the mfma A-operand layout puts 16 rows × 4 lanes/row, so the
reduction window is 4 lanes wide.

### 9.5 The exp + add/mul block in `softmax_p1_prescaled_16` (+ `kUsePk`)

`softmax_p1_prescaled_16<…, kUsePk>` emits the add/mul/exp block over the
16 fp32 lanes. `kUsePk = kSoftmaxUsePk = kPvAtEnd` selects the ALU port:

- **`kUsePk = true` (lo warps):** packed — `v_pk_add_f32` (subtract `m`) +
  `v_pk_mul_f32` (× log2e) over 8 pairs, then `v_exp_f32` × 16, then a
  `v_pk`-based reduction tree to `local_sum_e`.
- **`kUsePk = false` (hi warps):** fully de-packed — `v_add_f32_e32` × 16,
  `v_exp_f32` × 16, then a scalar binary reduction tree. Helpers `set_ninf1`
  / `set_ninf2<kUsePk>` (in `hk_mla_softmax.cuh`) pick `v_mov_b32` vs
  `v_pk_mov_b32` for the boundary −inf fills; `softmax_mask_p<…, kUsePk>`
  is de-packed the same way. `max_16` has no `v_pk` and is shared.

Both paths warp-reduce `local_sum_e` → `row_sum_e` through a shared tail.
The fused asm stays one block so the compiler doesn't scatter the `v_exp`
issues across inline-asm boundaries and break the back-to-back issue that
hides exp latency.

### 9.6 Pack to bf16 for PV: `pack_2f32_to_bf16_pair_pinned`

After `softmax_p1_prescaled_16`, `p_comp` holds 16 fp32 / lane. PV's mfma
needs bf16, so the kernel issues **8** `v_cvt_pk_bf16_f32`s via
`pack_2f32_to_bf16_pair_pinned<DST, SRC>()`, with destinations
`p_mfma[0..7]` and sources `p_comp[0,2,…,14]`. With
`k_p_mfma_begin = k_p_comp_begin = 48`, the destination
`p_mfma[0..7]` **overlays the low half of `p_comp[0..15]`** — the
cvt reads sources before writing dst, and low-to-high pack order
ensures no instruction reads a vgpr that an earlier pack has
overwritten.

Why use the **pinned** form (`pack_2f32_to_bf16_pair_pinned`, takes
register *numbers* as template args) instead of the runtime-arg form
(`float_2_bf16_pair`)? The pinned form encodes the destination VGPR
number directly in the asm string, so the overlay is guaranteed. The
runtime-arg form would emit `v_cvt_pk_bf16_f32 v[N]` with `N` as a
constraint letter, which the assembler treats incorrectly — see
`[[v40-cvt-to-pinned-inline-asm-gotcha]]`.

### 9.7 The setprio interlude

Between the local-max computation and `softmax_p1`, the kernel drops
wave priority to 1 (`s_setprio 1`). This lets the KV writer waves
(still cvt'ing + ds_writing in parallel) make progress while this wave
is exp-heavy. Softmax is one rung in the loop-wide `3 → 2 → 1 → 0`
ladder — see Ch. 10.7.

### 9.8 What's live after softmax

After softmax + pack completes:

- `p_mfma` v48..v55 holds $P^{(i)}$ in bf16, ready for PV mfma.
- `p_comp` v48..v63 still holds the same data (low half bf16-overlay,
  high half stale fp32 — but PV only reads the low half via p_mfma).
- `row_max` and `row_sum_e` are updated; `rescale` ($= \alpha$) is in
  a local fp32 and is **the value passed to PV** to rescale `oaccu`
  inside the PV gemm loop.

### 9.9 Final normalization (after the loop)

At the end of the main loop (epilogue branch of `mla_main`), each row's
`oaccu` is divided by its final `row_sum_e`. The op is split by warp group
(`kFinalRescaleUsePk = !kPvAtEnd`):

- **hi warps** (`kFinalRescaleUsePk = true`): `hk::mul_vgpr(oaccu, oaccu,
  1/row_sum_e)` → packed `v_pk_mul_f32` over the 128-vgpr tile.
- **lo warps** (`false`): a `static_for<k_o_sz>` sweep of `v_mul_f32_e32`
  (de-packed), keeping lo off the packed-ALU port.

(`hk::mul` can't be used for the de-packed sweep — its scalar op uses an `"i"`
immediate constraint, invalid for the runtime `1/row_sum_e`; hence the
explicit `v_mul_f32_e32` inline-asm sweep.) The OManager epilogue then writes
the normalized `oaccu` to VRAM. See Ch. 10.9.

## Chapter 10 — PV gemm + oaccu rescale

PV is the second mfma sequence per KV tile. It accumulates the rescaled
running output:

$$
\mathrm{oaccu}^{\mathrm{new}} = \alpha \cdot \mathrm{oaccu}^{\mathrm{old}} + P^{(i)} V^{(i)}
$$

with $\alpha$ from softmax (`rescale` in the source). The
implementation interleaves the rescale's multiplies into the gemm so the
rescale costs nothing in wall-clock; it also pre-loads V via
**transpose reads** so the mfma A-operand sees V in the right layout.

### 10.1 Why PV is computed as `mma_ABt(oaccu, kv, p_mfma)`

MLA's PV is $O = P V$, but the mfma is $C = A B^\top$ shaped. So we
compute the **transpose** instead:

$$
O^\top = V^\top P^\top
$$

and read `oaccu` as $O^\top$ (col-major), `kv` as $V^\top$ in A-operand
layout, `p_mfma` as $P^\top$ in B-operand layout. This is identical to
how QK was already running ($K^\top Q^\top = S^\top$).

The mfma is `v_mfma_f32_16x16x32_f16` of shape

$$
(16 \times 32) \cdot (32 \times 16) \to (16 \times 16)
$$

producing a 16×16 fp32 output tile.

> **Merged PV over both sub-tiles (m16x8, kBlockN=64).** One
> `hk_mla_v40_pv_stage` call now contracts **both** 32-row sub-tiles A+B in a
> single invocation — one prologue, `num_pv_iter = kNumKvSub · kDIters =
> 2 · 16 = 32` iters (`kDIters = kVoHeadDim/(2·kTileM) = 16`), `row_base =
> (iter/16)·32` selecting sub-tile A vs B. This replaced the earlier
> two-call form (one prologue per sub-tile), removing a wasted second
> prologue. m16x4 (kBlockN=32) keeps the 16-iter single-sub-tile form.
>
> **Who runs PV when.** `kPvAtEnd = (kWarpType == LoNoPEWarp)`: **lo warps**
> run PV at the *end* of the call (after softmax); **hi warps** *defer* PV by
> one tile (`kHasPv`), running the previous tile's PV at the *start* of the
> call (its V is still alive in `p_lds_kv_next`). The epilogue drains any
> pending deferred PV.

With `kVoHeadDim = 512` and `kBlockN = 64`, each sub-tile's PV is 16 iters
(each iter covers 32 V-cols = 2 mfma A-tiles of `kv`); two sub-tiles = 32.

### 10.2 V from LDS via transpose-read: `load_transposed_v_to_gpr`

PV's mfma needs the V data laid out as $V^\top$ in mfma A-operand
order. The KV pong holds V in **K-order** (each row of the pong = one
KV token, cols indexed by D). The trick: gfx950's
`ds_read_b64_tr_b16` performs a bf16 transpose at LDS-read time —
4 lanes' 64 bf16 input bits get re-shuffled into the right output
layout for an mfma A-operand.

`KvManager::load_transposed_v_to_gpr<kRowOffset, kColOffset, GPR>`
issues one `ds_read_b64_tr_b16` per call, producing 2 dwords/lane in
`(GPR, GPR+1)`. Per PV iter the kernel issues **4** of these:

| Call | Row offset | Col offset | Dst |
|---|---:|---:|---|
| 1 | 0 | `iter*32 + 0` | `kv[k_kv_begin + 0..1]` |
| 2 | 16 | `iter*32 + 0` | `kv[k_kv_begin + 2..3]` |
| 3 | 0 | `iter*32 + 16` | `kv[k_kv_begin + 4..5]` |
| 4 | 16 | `iter*32 + 16` | `kv[k_kv_begin + 6..7]` |

After 4 reads, the KV carriers (`k0/k1/k2`, v36..v47 — the same range used
for QK K operands, role-toggled) hold 2 mfma A-tiles' worth of $V^\top$
data. (In V1/m16x4 this was `kv` at v112..v119.)

### 10.3 The interleaved PV iter (canonical pattern)

One PV iter, canonical case (`kIsFirstIter == false`, `has_next == true`):

| # | sp3 instructions issued | Purpose |
|---:|---|---|
| 1 | 4× `ds_read_b64_tr_b16` | load V via transpose into `kv_top` + `kv_bot` for THIS iter |
| 2 | 2× `mul_pair` (= 4× `v_mul_f32`) | rescale NEXT iter's `oaccu` sub-tile +0 and +1; hidden under ds_read latency |
| 3 | `s_waitcnt lgkmcnt(2)` | drain to 2 outstanding ds_reads |
| 4 | `mma_ABt(oaccu_a, kv_top, p_mfma, oaccu_a)` | first PV mfma |
| 5 | 1× `mul_pair` | rescale NEXT iter's sub-tile +2 (1 slot per mfma) |
| 6 | `s_waitcnt lgkmcnt(0)` | drain remaining ds_reads |
| 7 | `mma_ABt(oaccu_b, kv_bot, p_mfma, oaccu_b)` | second PV mfma |
| 8 | 1× `mul_pair` | rescale NEXT iter's sub-tile +3 |

So per PV iter:

- 4× `ds_read_b64_tr_b16` (V load)
- 2× `v_mfma_f32_16x16x32_f16` (the PV mfmas)
- 4× `v_mul_f32` rescaling the NEXT iter's oaccu base tile +0/+1 (interleaved with ds_read)
- 4× `v_mul_f32` rescaling the NEXT iter's oaccu base tile +2/+3 (1 mul_pair per mfma slot)
- 2× `s_waitcnt`

Total rescale per iter = 8 `v_mul_f32`. Over 16 iters = 128 `v_mul_f32`,
exactly the count to multiply the full 128-vgpr `oaccu` by `rescale` once.
In the **merged** 32-iter PV (kBlockN=64), the rescale is guarded to the
first sub-tile's iters (`row_base == 0`), so it still fires exactly 128
multiplies total across the merged call — sub-tile B accumulates onto the
already-rescaled `oaccu` with no further rescale.

### 10.4 First-iter and last-iter special cases

The canonical 3-arg accum `mma_ABt(oaccu, kv, p_mfma, oaccu)` is the
common case. Two branches:

- **`kIsFirstIter`**: skip the rescale entirely ($\alpha = 1$ on iter 0,
  there's no $\mathrm{oaccu}^{\mathrm{old}}$ to scale), and use the
  **3-arg init form** `mma_ABt(oaccu, kv, p_mfma)` (no accumulator).
- **`has_next == false`** (last iter): no next iter's oaccu to rescale,
  so the `mul_pair` interleave is dropped. Mfmas still issue accum
  (or init on `kIsFirstIter && last`).

The kernel emits these via two `if constexpr` branches: the special
case (init form on first iter, plain accum on last) emits only the
2 mfmas + 1 mid-`s_waitcnt`; the canonical case adds the 4 interleaved
`mul_pair`s.

### 10.4.1 `kDoRescale` — the two PV gemm instantiations

The whole PV gemm (prologue `pk_mul_pair` + the per-iter loop) is a lambda
templated on `bool kDoRescale`. The kernel picks the instantiation per tile
from the wave-uniform deferred-rescale decision (§9.1.1):

- `kIsFirstIter` → `kDoRescale = false` (oaccu init, nothing to rescale).
- else if `do_rescale` (the ballot fired) → `kDoRescale = true` — the full
  path with all the interleaved `pk_mul_pair`/`mul_pair`s above.
- else → `kDoRescale = false` — the max was kept stale, so $\alpha = 1$ and
  **every** rescale multiply is dropped, leaving a clean PV gemm (4 ds_read
  + 2 mfma + 2 s_waitcnt per iter, no VALU rescale).

Because the rescale multiplies are already hidden under ds_read / mfma
latency (§10.5), dropping them does not shorten a serial dependency chain —
the win is the reclaimed VALU/issue slots, which matters only at long
context where the running max plateaus and the skip path is taken on most
tiles. At short context the max keeps climbing within a wave, so
`do_rescale` is almost always true and the full path runs (no regression
beyond the one ballot).

### 10.5 Why this pattern hides everything

The numbers work out because gfx950's mfma occupancy budget per lane is
generous: each `v_mfma_f32_16x16x32_f16` issue spends ~32 cycles in the
mfma pipe, during which the VALU is free. The pattern fills both halves:

| Time slice within one PV iter | mfma pipe | VALU | LDS |
|---|---|---|---|
| t=0 | — | — | 4× ds_read in flight |
| t=1 | — | 2× mul_pair (rescale next +0, +1) | ds_read still draining |
| t=2 | mfma 1 | — | — |
| t=3 | mfma 1 in flight | mul_pair (rescale next +2) | — |
| t=4 | mfma 2 | — | — |
| t=5 | mfma 2 in flight | mul_pair (rescale next +3) | — |

The schedule is dense — there's no slot where mfma is idle waiting on
VALU or LDS. The rescale, naïvely a 128-mul standalone phase before PV,
is fully hidden.

### 10.6 `pv_v_aux` is dead in Gen.1

The pinned VGPR map in Ch. 5 lists `pv_v_aux` v104..v111 as "second
V-tile staging during PV." The kernel comment marks this as
**deferred — single-buffered in Gen.1**.

A double-buffered V load (pong `kv_top/bot` for current mfma, pong
`pv_v_aux` for next mfma's V) would let the next iter's
`ds_read_b64_tr_b16` overlap with the current iter's mfma — squeezing
another few percent. The architectural slot is there; the kernel body
just doesn't use it yet. The same registers are still reused as
`kv_alt` during QK Phase A (Ch. 8 fused pair-prefetch), so no VGPR
budget is wasted.

### 10.7 setprio ladder (loop-wide context)

Across one main-loop iter, the kernel issues `s_setprio` to a falling
ladder `3 → 2 → 1 → 0`:

| Phase | setprio | Why |
|---|---:|---|
| QK Phase A (mfma + KV prefetch interleave) | 3 (highest) | QK mfmas are the hot path; the KV writers (waves that converted+stored the *previous* tile) should yield |
| QK Phase B (q from LDS) | 2 | Still mfma-heavy but the KV writers are catching up |
| Softmax | 1 | exp/pk_add dominated; KV writers still need bandwidth |
| PV gemm (canonical body) | 0 | Allow KV writers and any outstanding LDS traffic to drain freely |

The ladder is set by `__builtin_amdgcn_s_setprio(N)` at phase boundaries.
This is one of the "fine-tuned setprio" wins in commit `2daf7dd6d`.

### 10.8 oaccu register grouping

`oaccu` v128..v255 = 128 vgprs/lane = 16 PV iters × 8 vgprs/iter.
Per iter $i$ the relevant slice is `v[128 + 8i .. 128 + 8i + 7]`,
which decomposes:

| oaccu sub-tile | vgprs (relative) | mfma output |
|---|---|---|
| `oaccu_a` (first 16 V-cols, low half) | +0..+3 | `mma_ABt(oaccu_a, kv_top, p_mfma)` |
| `oaccu_b` (first 16 V-cols, high half) | +4..+7 | `mma_ABt(oaccu_b, kv_bot, p_mfma)` |

The 4 muls per sub-tile (when rescaling) cover the 4 vgprs/lane of one
16×16 mfma output tile in col-major layout.

### 10.9 After PV completes

End of one main-loop iter:

- `oaccu` v128..v255 holds $\sum_{j \le i} (P^{(j)} V^{(j)})$ after the
  appropriate rescales — i.e. **incrementally correct** modulo the
  final $1/\ell$ division.
- `row_max`, `row_sum_e` are updated (in softmax).
- `kv` v112..v119 holds the last PV iter's V — dead, will be
  overwritten next iter's QK by the new tile's K.

At the global last iter the epilogue branch runs
`hk::mul_vgpr(oaccu, oaccu, 1/row_sum_e)` — one `v_mul_f32` per lane
per oaccu vgpr — then writes to VRAM via the OManager (Ch. 11).

## Chapter 11 — OManager V3 / V3NoStage epilogue

After the main loop ends, each warp owns a 16×512 fp32 `oaccu` tile
already normalized by $1/\ell$. The OManager:

1. Casts (bf16 path) or passes through (fp32 split path) the fp32 oaccu.
2. **Un-swizzles** the sb8 perm that was applied to the K/V D-axis in
   Ch. 7 / Ch. 8 — so the user sees natural col order in VRAM.
3. Coalesces 8 bf16 (or 4 fp32) per lane into one `buffer_store_dwordx4`.

Three variants are shipped:

| Manager | Output | Bounce LDS | When used |
|---|---|---:|---|
| `OManager16bitsV4Gen1Swizzle` | bf16 → final_output | 2112 B / warp ($\approx 16.5$ KiB) | `kEpilogueType = OutputFinal`, the common case |
| `OManager32bitsV4Gen1Swizzle` | fp32 → split_output | 4352 B / warp ($\approx 34$ KiB) | `kEpilogueType = OutputSplit`, with a bounce, when split-O LSE is needed |
| `OManager32bitsV4Gen1SwNoStage` | fp32 → split_output | 0 (direct) | `kEpilogueType = OutputSplit`, direct write when the LDS region is contended by the split-O reduction (see commit `15a8736c4`'s notes) |

### 11.1 Call shape: 64-cols-per-call, 8 calls per warp

The kernel emits 8 calls per warp (`num_pv_pair_iter = kVoHeadDim / (2·kBlockN) = 512/64 = 8`). For iter $i \in [0, 8)$, the call is
`output_to_vram_pair<GPR_BASE, kWaveTileColOff>(...)` with
`GPR_BASE = k_o_begin + 16i` and `kWaveTileColOff = 64i`
(where `k_o_begin = 128`, the first oaccu VGPR). Each call covers **one full 64-col wave-tile** =
16 fp32/lane = 16 vgprs. Compared to V2 (which covered 32 cols / call
across 16 calls), V3 batches twice the work per call — this is what
lets the un-swizzle resolve in a single LDS bounce round-trip per call.

### 11.2 V3 bf16 path — what happens inside one call

The bounce LDS layout for one warp:

| Quantity | Value |
|---|---:|
| `kNumRows` | 16 (= mfma m-dim) |
| `kNumCols` | 64 (one wave-tile width) |
| Padding elements per 2 rows | 4 (bank-conflict pad) |
| Padded elem count per 2 rows | $2 \cdot 64 + 4 = 132$ |
| Bytes per 2 padded rows | $132 \cdot 2 = 264$ B |
| **Per-warp bounce** | $8 \cdot 264 = 2112$ B (= 8 row-pairs, 16 rows total) |
| `kVramStElemPerLane` | 8 bf16 (= 1 `buffer_store_dwordx4`) |
| `kVramStLanePerRow` | $64 / 8 = 8$ |
| `kVramStRowsPerRnd` | $64 / 8 = 8$ (each round of stores covers 8 rows) |
| `kVramStNumRnds` | $16 / 8 = 2$ rounds |

Stages within one call:

| # | Stage | Per-lane action |
|---:|---|---|
| 1 | fp32 → bf16 pack | 8× `v_cvt_pk_bf16_f32` = 16 fp32 → 4× bf16x2 dwords (4 dwords/lane = 64 bf16/4lanes_per_col_band) |
| 2 | LDS-write (straight) | 4× `ds_write_b64` at stride `kMfmaCols·sizeof(out_t) = 32 B`, address = `lds_warp + v_offset_lds_st` |
| 3 | LDS-read (un-perm) | 2× `ds_read_b128` (covers 16 rows × 8 cols/lane) at the **sb8-inverse-permuted** col |
| 4 | VRAM-store | 2× `buffer_store_dwordx4` — round 1 at `v_offset_vram_st`, round 2 at `+ 8192 B` (= 8 rows × 512 cols × 2 B) |

The fine-grained `s_waitcnt lgkmcnt` between stages 3 and 4 drains LDS
reads one at a time so each `buffer_store_dwordx4` can fire as soon as
its source dwords are ready.

### 11.3 The sb8-inverse-perm un-swizzle (the heart of V3)

The writer side (stage 2) is **straight** — the writer just lays the
fp32-cvt'd bf16 into the bounce LDS in mfma layout. The un-swizzle
happens on the LDS-**read** side (stage 3) by computing the LDS
sub-tile index from the lane's desired VRAM col.

Per-lane mapping:

| Quantity | Expression |
|---|---|
| `row_lds_ld` | $\ell / 8$ (= 0..7, one row per lane in this round) |
| `lane_in_row` | $\ell \bmod 8$ (= 0..7, which 8-col chunk in the wave-tile this lane wants) |
| Desired VRAM sub-tile | `lane_in_row` (= 0..7, natural order) |
| LDS sub-tile holding that data | $\mathit{ldsSubtile} = \mathit{sb8Perm}(\mathit{laneInRow})$ |

Inline closed form (matches `sb8_perm_col_elems` restricted to a
sub-tile index):

$$
\mathit{ldsSubtile} = ((\mathit{laneInRow}   \mathbin{\mathrm{and}}   1) \ll 2) \,|\, ((\mathit{laneInRow}   \mathbin{\mathrm{and}}   6) \gg 1)
$$

This is the same closed form as Ch. 7.2.2's forward perm $[0,2,4,6,1,3,5,7]$
applied to the *3-bit sub-tile index* of `lane_in_row`. The reader
reads from $\mathit{ldsSubtile} \cdot 8$, finds the writer's data
there, and stores to VRAM at `lane_in_row * 8` — natural order in the
output buffer.

Why is this just $\mathit{sb8Perm}$ and not $\mathit{sb8Inv}$? The
reader is asking: "I want data at natural col $c = \mathit{laneInRow}$;
where is that data in LDS?" The writer put data at *permuted* position,
so the data the reader wants is at $\mathit{sb8Perm}(c)$. Mathematically
this equals the inverse perm applied to the reader's LDS address; either
form is correct, the code uses the forward direction because it's
cheaper to evaluate (a 3-bit table that constant-folds into the address
calculation).

### 11.4 V3 vs V3NoStage

The two split-O variants differ only in whether they use a per-warp
bounce LDS:

- **`OManager32bitsV4Gen1Swizzle`** uses a 4352 B/warp bounce — same un-swizzle
  pattern as the bf16 V3, scaled to fp32 with `kNumElemPerPaddedRow = 68`
  (= 64 + 4 pad). Total per-WG bounce = $8 \cdot 4352 = 34816$ B.
- **`OManager32bitsV4Gen1SwNoStage`** writes directly from oaccu VGPR to
  VRAM. No bounce LDS. Used when the LDS budget at epilogue time is
  contended by a downstream split-O reduction step that needs the
  same LDS region.

The "bounce-or-not" decision is made at host trait wiring time, not
per-call.

### 11.5 Why a bounce LDS at all (for V3)

The natural alternative — pack fp32 to bf16 in pinned VGPRs and emit
`buffer_store_dwordx4` directly — produces **uncoalesced** VRAM stores:
each lane wants natural-col-order, but the lane→col mapping after
sb8 perm + mfma layout gives 8 cols/lane that are *not* contiguous
in VRAM. The bounce LDS lets each warp:

1. Write its 16×64 wave-tile to LDS in *any* convenient layout (we
   choose the writer-straight layout above).
2. Re-read with `ds_read_b128` choosing the lane→col mapping that
   *will* produce coalesced VRAM stores (8 contiguous bf16 / lane in
   row order).

Removing the bounce (as V3NoStage does for fp32 split-O) trades VRAM
coalescing for LDS budget. For fp32 split-O the downstream reduction
step is the bottleneck, not the per-lane store coalescing, so the
trade is favorable.

### 11.6 The vmcnt(0) gate removal (commit `15a8736c4`)

A prior version of `OManager32bitsV4Gen1Swizzle` (and `V3NoStage`) issued a
`__builtin_amdgcn_s_waitcnt(0)` before each call's `buffer_store_dwordx4`.
At `b=33, c=63333` this added ~30k cycles per epilogue invocation —
~10 µs of pure stall. The gate was removed (the surrounding wait
already ensures store ordering); the perf win is row 6 of the
progression in Ch. 1.

### 11.7 Reuse of oaccu VGPRs as ds_read destinations

After stage 1 + 2 of the bf16 path (pack + write to bounce), the
source oaccu VGPRs `GPR_BASE..GPR_BASE+7` (= 8 of the 16 vgprs/lane
for this wave-tile) are **dead** — oaccu is not re-read by this
work_idx. So stage 3's `ds_read_b128` targets *those same VGPRs* as
its destination:

| ds_read | dst VGPRs |
|---|---|
| round 0 | `GPR_BASE + 0..3` |
| round 1 | `GPR_BASE + 4..7` |

This is one of the "OMgr pinned reg reuse" wins (commit `6d61ccff6`).
The compiler would otherwise allocate fresh unpinned scratch VGPRs to
hold the read results, and those could leak into pinned `q_vgpr` or
oaccu if the budget ever became tight.

### 11.8 After the epilogue completes

- VRAM holds the natural-col-order, sb8-un-swizzled output.
- LDS bounce is dead (no one reads it again this work_idx).
- The kernel exits if `work_start_idx + 1 >= work_end_idx`, else loops
  back to the next persistent work item (Ch. 12).

## Chapter 12 — Dispatch ladder & slim dispatch

The body of each KV-tile iter is a generic lambda `mla_main` with four
template params. The host wrapper expands those params into a per-warp
dispatch ladder so each iter sees the cheapest specialization for its
position in the warp's tile sweep.

### 12.1 `mla_main` template parameters

| Param | Type | Meaning |
|---|---|---|
| `kIsFirstIter` | `bool` | This is the warp's first compute iter — oaccu has no prior state, so rescale is skipped and PV uses the 3-arg init mfma. |
| `kSkipCompute` | `bool` | Warp is idle on this tile (e.g., causal-masked out, or this is the trailing epilogue-only run). Implies `!kIsFirstIter`. |
| `kEpilogueType` | `PvGemmEpilogueType` | `None` (continue the loop), `OutputFinal` (bf16 final via OMgr V3), or `OutputSplit` (fp32 split-O via OMgr V3 / V3NoStage). |
| `kCheckBoundaryNext` | `bool` | The *next* tile may be OOB (partial last tile). Affects `prefetch_kv_tile`'s boundary check (Ch. 8). |

Two derived flags inside the lambda:

| Derived | Definition |
|---|---|
| `kDoEpilogue` | `kEpilogueType != None` |
| `kIsGlobalLast` | `kSkipCompute \|\| kDoEpilogue` (no next tile — skip prefetch + wait + swap) |

Two static_asserts enforce sanity:

| Forbidden combo | Why |
|---|---|
| `kSkipCompute && kIsFirstIter` | A skip warp has no prior compute → "first iter" makes no sense. |
| `kIsGlobalLast && kCheckBoundaryNext` | Global-last means no next tile to load → no boundary check applies. |

> **Two derived flags added in V2 (m16x8).** `kHasPv = (!kPvAtEnd) &&
> (!kIsFirstIter) && ((!kSkipCompute) || kDoEpilogue)` selects the **deferred
> PV** of the previous tile at call start (hi warps). The **deferred strip-3**
> consume (§8.13) runs on `(!kIsRopeWarp) && (!kIsFirstIter)`. Both piggyback
> on the same four template params — no new template args.

### 12.2 Iter classification

For each warp, the host wrapper walks `[kv_start, kv_end)` in steps of
`kBlockN` (**64** for m16x8, 32 for m16x4) and classifies each iter:

| Iter class | $(\text{kIsFirstIter}, \text{kSkipCompute}, \text{kEpilogueType}, \text{kCheckBoundaryNext})$ |
|---|---|
| First-of-many (real) | `(true, false, None, …)` |
| Middle (real) | `(false, false, None, …)` |
| Warp's last real + global-last (combined) | `(true \| false, false, OutputFinal \| OutputSplit, false)` |
| Trailing skip-epilogue | `(false, true, OutputFinal \| OutputSplit, false)` |
| Pure idle warp | `(false, true, None, false)` |

The "trailing skip-epilogue" pattern handles the case where a warp ran
its last real tile *before* the global last tile (e.g., causal masking
made it idle on later tiles). It still needs to participate in the
final epilogue so its oaccu lands in the output.

### 12.3 Middle-iter peeling (zero per-iter branch)

The middle (all-`None`, all-fully-in-bounds) iters dominate runtime for
long-context inputs. The hot inner loop must have **zero per-iter
branches** to keep mfma cadence stable.

The ladder splits the middle range:

1. **Bulk middle loop** — while *both* the current and the iter-after-next
   tiles are fully in bounds, dispatch
   `(false, false, None, false)` (boundary-check off).
2. **Trailing middle iter** — if the loop exits with one middle iter
   still to do whose *next* tile is the global last (possibly
   partial), dispatch `(false, false, None, true)` (boundary-check on)
   **exactly once**, peeled out of the loop.

This pattern applies in **both** dispatch modes (slim and non-slim).
The thread-trace win measured against the per-iter `if` form was
~2–3 % on long contexts.

### 12.4 Slim vs non-slim dispatch

A compile-time flag `MLA_SLIM_DISPATCH` (default = 1 in Gen.1)
toggles between two ladders:

| Mode | Template-arg pattern | Number of `mla_main` instantiations |
|---|---|---:|
| Non-slim | `kCheckBoundaryNext` varies per iter class (true on the trailing middle iter, false in the bulk, true / false split for the warp's-last-real depending on `kv_len % kBlockN`) | ~20+ combos |
| **Slim** (default) | Always `kCheckBoundaryNext = true` *except* when forbidden by `kIsGlobalLast` | ~half of non-slim |

Slim dispatch drops the `kv_len % kBlockN == 0` and
`kv_len_eff % kBlockN == 0` fast-path specializations (rare in
practice — KV seq lengths in production are not multiples of 32).
The cost is **1 compare + 1 cmov per K-iter** inside
`prefetch_kv_tile`'s in-bounds gate.

Outcome:

- **40 % smaller kernel image** (fewer template instantiations →
  fewer ISA bytes → faster L1 instruction-cache fills, especially on
  the first cold launch).
- **Perf-neutral** on the hot path (the cmp+cmov is free given the
  surrounding mfma cadence).

### 12.5 The slim correctness fix

Slim required one Kv-manager change. The non-slim form gated the
`row_kv_ld_next_next` carry update — which remembers the resolved
physical row for the iter-after-next's prefetch — on
`kCheckBoundaryNext == false`. With slim's always-true, that condition
was permanently false; the carry never updated and subsequent iters
re-prefetched from a stale row, corrupting results from `b ≥ 33` at
long contexts.

Fix: re-gate the carry on `kIsGlobalLast == false` (regardless of
`kCheckBoundaryNext`). Now the carry updates on every non-last iter,
slim or not.

### 12.6 V32 dispatch ladder (out of scope but worth noting)

The V32 kernel family (in `mi3xx_v32_*` / `mi35x_v32_*`) uses a
similar but **richer** ladder — V32 emits `kCheckBoundaryNext = false`
for the middle path and uses a per-iter if-check for the last middle.
That's structurally less optimal than V40's peel-out form, but the V32
kernels are not in scope for this Gen.1 cleanup and are left as-is.

### 12.7 Per-warp causal offsets

Before the dispatch ladder runs, each warp computes a per-warp
**causal offset** that shifts its `kv_end_eff` back from the global
`kv_end`:

Let $G$ = `num_wave_group` (= qseqlen), $K$ = `waves_per_head`
(= num_qheads / kTileM = num_qheads / 16). The per-warp causal offset
$\delta_w$ is

$$
\delta_w = G - 1 - (w \gg \log_2 K)
$$

For MTP > 1 each warp owns a different query token; the causal mask
forbids attention to KV positions beyond the warp's own token. Warps
that own earlier query tokens have **smaller** effective KV ranges,
which means more skip iters for them and fewer real iters. The ladder
above handles this naturally via the `kv_len_eff <= 0`, `< kBlockN`,
`== kBlockN`, and `> kBlockN` branches at the top of §12.2.

### 12.8 Persistent work loop

The kernel is a **persistent** kernel: one workgroup processes
multiple `work_idx`'s from `params.p_work_indptr[worker_idx ..
worker_idx+1]` in a top-level loop. After the OManager epilogue
finishes one work_idx, the workgroup loads the next one's Q and
restarts the dispatch ladder. The KV double-buffer pong, OMgr bounce,
and Q-LDS region are all re-initialized; the pinned VGPRs carry no
cross-work-idx state (oaccu is reset to zero on the next work's
first iter via `kIsFirstIter = true`).

## Chapter 13 — Hazards & gotchas

Bugs whose root cause was not in the kernel logic itself — but which
*looked* exactly like kernel bugs and consumed real debugging time.
Each row is one rung future-you may hit. They are the kind of failure
ISA inspection + careful diff are good at; raw "add a printf" is not.

### 13.1 Compiler / inline-asm gotchas

| # | Symptom | Root cause | Fix |
|---:|---|---|---|
| 1 | `v_cvt_scalef32_pk_bf16_fp8 v[N]` with `N` as inline-asm template int silently produces garbage (consumer MFMA reads stale data). | The pinned-DST form IS correct, but the compiler can't see VALU→MFMA RAW hazard across the opaque inline-asm boundary. Without a manual `s_nop`, the cvt's writeback misses the MFMA's read window. | Use the `cvt_scalef32_pk_bf16_fp8_pinned` wrapper from `hk_mla_utils.cuh` which emits the `s_nop` (or use the `v_mov` trampoline form). |
| 2 | `e8m0_to_f32` returning wrong scale on V40 KV path — ~88 % output mismatch under `att`. | The pure-C++ form `bit_cast<float>(b << 23)` is SSA. LLVM's machine-sink/LICM hoists it cross-BB past the matching `s_waitcnt` to the original `buffer_load_ubyte` def site — racing the load. `sched_barrier(0)` is *intra-BB only*; cross-BB sinks ignore it. | `asm volatile("v_lshlrev_b32 …")`. `asm volatile` is the only cross-BB ordering construct LLVM honors against asm-volatile loads. |
| 3 | `PROBE_*_ITER` macros appear to "default to whatever the source says" but actually evaluate to 0 regardless. | `aiter/jit/optCompilerConfig.json` force-defines them via env var on the compile command line; source `#ifndef` defaults never fire. | Comment out the entry in `optCompilerConfig.json` for local debugging, then restore. |
| 9 | (m16x8/slim) An `s_waitcnt vmcnt(0)` appears between the `v_bitop3` col swizzle and the prefetch `buffer_load` in the KV-tile loop — a full vmem drain that kills carrier/staging overlap. | The prefetch address is `row_kv_ld << 9 + …`, and `row_kv_ld` comes from a `buffer_load_dword` in `get_kv_ld_row`. The compiler **rematerializes** that index load inline right before the address, so using the loaded value as a load *address* forces a `vmcnt(0)`. | Pin the already-resolved index in a VGPR so it can't be rematerialized: `asm volatile("" : "+v"(row_kv_ld_next));` right after `row_kv_ld_next = row_kv_ld_next_next;`. Costs 1 scratch VGPR (see Ch. 5.2). |
| 10 | (m16x8) De-packing the epilogue `oaccu` normalize with `hk::mul(oaccu, oaccu, 1/row_sum_e)` fails to compile (`invalid operand for inline asm constraint 'i'`). | `macros::mul`'s scalar op uses an `"i"` (immediate) constraint — valid only for a compile-time-constant scalar, not the runtime `1/row_sum_e`. `hk::mul_vgpr` uses `"v"` (packed). | For the de-packed sweep, emit `v_mul_f32_e32 v[%0], %1, v[%0]` with a `"v"` operand in a `static_for` over `k_o_sz` (Ch. 9.9 / 10). |

### 13.2 Memory-permutation / layout gotchas

| # | Symptom | Root cause | Fix |
|---:|---|---|---|
| 4 | "PR-A only permutes K's D-axis" reproduces with 100 % mismatch on QK output. | QK is a reduction over the D-axis. If K's D-axis is permuted in LDS but Q's isn't, mfma step $k$ multiplies $Q_{m,k}$ against $K_{n,\mathrm{perm}(k)}$ — wrong product. Q and K must be permuted **identically**. | The sb8 perm must apply to both writers (Ch. 7 and Ch. 8). KV-only PR is structurally impossible. |
| 5 | V40 Site C QK reader: 2-way `ds_read_b128` bank conflict. Method 1 (LDS-write address swap) compiles cleanly and *looks* right by hand-derived bank math — produces ~90 % output mismatch in practice. | The non-linear `ds_read_b128` cycle 0 pairs lanes $(L, L{+}20)$. `+20` flips bit 4 *and* bit 2 of $L$ together. The Method-1 swap targets bit 4 only — analytically equivalent to Method 2, but the HW cycle's actual data routing depends on bit 2 too, so the supposedly-fixed pair still lands on the same quad. | Method 2 (**vmem-load-side** col-half-swap + reader-side XOR). Lives in `prefetch_kv_tile` and Phase 2's NoPE writer. |
| 6 | V4 test harness: heads 1+ produce garbage RoPE values, head 0 is fine. | `quantize_v4_q` returns a non-contiguous `q_rope_bf16` slice (stride is on the head axis, not the elements axis). Without `.contiguous()`, only head 0 reads aligned bf16; heads 1+ read mis-aligned. | Always `.contiguous()` on `q_rope_bf16` between the quantizer and the kernel call. |

### 13.3 Host / metadata gotchas

| # | Symptom | Root cause | Fix |
|---:|---|---|---|
| 7 | Stochastic NaN in V40 output at `b ≥ 4`. `-ms` (max-splits) flag has no effect on perf or correctness. | `csrc/kernels/mla/metadata/v1_2_device.cuh:141` has a leftover debug override: `int32_t remain_payload = 0x7fffffff;` (was `= payload`). With splitting disabled, all works pile on `WG=0`; the epilogue's `p_lds_o` reads race the *next* work's `load_q` writes (intra-WG work-loop hazard). | Revert to `= payload`. |
| 11 | (host) After trimming branch-only changes back toward main, `module_mla_metadata` fails to build (`number of argument annotations does not match`), or `aiter.MlaVersion` is missing at runtime. | The `MLA_METADATA_PYBIND` macro in `rocm_ops.hpp` was reverted to main's arg list (no `MlaVersion` enum, `dtype_q/dtype_kv` instead of the four `dtype_*_nope/rope` + `mla_version`), but the C++ `get_mla_metadata_v1` still has the V40 signature → pybind arg-count mismatch. | Keep the V40 `MLA_METADATA_PYBIND` (enum + `mla_version` + `dtype_{q,kv}_{nope,rope}`) whenever the C++ signature has them. |

**Deferred strip-3 WAR (m16x8, §8.13).** Not a bug today, but the invariant
to preserve: the iter N+1 deferred consume must ds_read staging slot 1 and
complete its `lgkmcnt(0)` *before* iter N+1 Phase A re-issues the strip-3
`buffer_load_lds` into that slot. Reorder at your peril.

### 13.4 Confirmed-not-a-bug

| # | What was suspected | What's actually true |
|---:|---|---|
| 8 | The pinned Q VGPRs (v64..v127) might be getting clobbered mid-loop (any V40 numerical mismatch). | The pinned-q region is **read-only** after the prologue. Proven by ISA scan + runtime probe ([[v40-pinned-q-read-only-confirmed]]). When a V40 mismatch is being debugged, eliminate Q as suspect first — look downstream (K/V load, sb8 perm, OMgr un-perm). |

### 13.5 Tools to reach for

- `check-unpinned-reg-usage` skill (`.claude/skills/check-unpinned-reg-usage/`)
  — scans the post-`--save-temps` `.s` file for decimal `v ≥ N` (the scratch
  budget) and spill > 0. Run after every nontrivial change. Current state:
  m16x8 `budget=36 / spill=0`; m16x4 `budget=44 / spill=0`.
- ISA inspection (`-save-temps` + read the `.s` file) is the only way
  to catch compiler-hoist / inline-asm hazards above. Standard
  printf-debugging will not find them.
- `rocprofv3 --att` thread-trace dumps for cadence analysis — useful
  for finding the per-iter `if` branches that the middle-iter peel
  (Ch. 12.3) was designed to eliminate.

## Chapter 14 — File map

### 14.1 V40 Gen.1 active files

| File | What it contains |
|---|---|
| `aiter/mla.py::mla_v40_decode_fwd` | Python router → `hk_mla_v40_decode_fwd` (decode) + `mla_reduce_v1` (reduce). Picks m16x8 (`H·mtp=128`) or m16x4 (`=64`). |
| `csrc/kernels/mla/hk/mi35x_v40_fwd_decode_m16x8_fp8bf16_fp8bf16_gen1.cuh` | **m16x8** kernel (kBlockN=64, V2 KV pipeline): persistent work loop, `mla_main` lambda + dispatch ladder, pinned-VGPR & LDS layout, KV Phase A/B, deferred strip-3, softmax (kUsePk), merged PV (`hk_mla_v40_pv_stage`), epilogue (oaccu de-pack normalize). |
| `csrc/kernels/mla/hk/mi35x_v40_fwd_decode_m16x4_fp8bf16_fp8bf16_gen1.cuh` | **m16x4** kernel (kBlockN=32, V1 KV pipeline, occupancy 2). |
| `csrc/kernels/mla/hk/hk_mla_v40_fwd_decode_gen1_common.cuh` | `HkMlaV40Regs<T>` — the single source of truth for the pinned-VGPR map + `hk_mla_v40_pv_stage`/`hk_mla_v40_pv_gemm` (shared by both partitions). |
| `csrc/kernels/mla/hk/hk_mla_v40_buffer_managers_gen1.cuh` | V40-only managers: `QManager8to16bitsV1`, `KvManager8to16bitsV1`, **`KvManager8to16bitsV2` (: public V1)**, `OManager16bitsV4Gen1Swizzle`, `OManager32bitsV4Gen1Swizzle`, `OManager32bitsV4Gen1SwNoStage`. Also the sb8 perm helpers. |
| `csrc/kernels/mla/hk/hk_mla_softmax.cuh` | Online-softmax helpers, all `kUsePk`-templated: `softmax_mask_p`, `softmax_p1_prescaled_16`, `set_ninf1/2`. |
| `csrc/kernels/mla/hk/hk_mla_utils.cuh` | Shared with V32: traits, enums (`PvGemmEpilogueType`), and `namespace hk_mla` helpers: `e8m0_to_f32`, `encode_s_waitcnt`, `max_16`, `warp_reduce`, `cvt_scalef32_pk_bf16_fp8_pinned`, `pack_2f32_to_bf16_pair_pinned`, `get_kv_ld_row`. |

### 14.2 Adjacent (not V40 Gen.1 but referenced)

| File | Role w.r.t. V40 Gen.1 |
|---|---|
| `csrc/kernels/mla/hk/hk_mla_buffer_managers.cuh` | V32-shared managers (`QManager8bitsV1..V5`, `KvManager8bitsV1..V3`, `OManager16bitsV1..V2`, `OManager32bitsV1..V2`, `VtManager8bitsV1`). V40 Gen.1 does not include this. |
| `csrc/kernels/mla/hk/mi35x_v32_fwd_decode_m16x8_fp8_fp8.cuh`<br/>`csrc/kernels/mla/hk/mi35x_v32_fwd_decode_m16x4_fp8_fp8.cuh`<br/>`csrc/kernels/mla/hk/mi3xx_v32_fwd_decode_m16x8_fp8_fp8.cuh` | V32 sibling kernels. Share the per-iter-branch dispatch pattern noted in Ch. 12.6 — not in scope for Gen.1 cleanup. |
| `csrc/kernels/mla/hk_v32_decode_fwd.cu` | V32 host wrapper. |
| `csrc/kernels/mla/metadata/v1_2_device.cuh` | MLA work planner / metadata. Source of the `remain_payload` debug-override gotcha in Ch. 13.3. |
| `csrc/kernels/mla/reduce.cu` | Cross-WG reduction for split-O outputs (consumes the fp32 split tensor that V3 / V3NoStage produces). |
| `aiter/jit/optCompilerConfig.json` | Per-module hipify input set + force-defines. V40 Gen.1's entry lists the 4 `.cuh` files in §14.1. Source of the `PROBE_*_ITER` macro gotcha in Ch. 13.1. |

### 14.3 Tooling

| Path | Purpose |
|---|---|
| `.claude/skills/check-unpinned-reg-usage/` | ISA audit script — scans the post-`--save-temps` `.s` for decimal `v ≥ N` (scratch budget) and `spill > 0`. Run after every nontrivial change. Budgets: m16x8 = 36, m16x4 = 44; spill 0. |

## Chapter 15 — Glossary & cross-refs

### 15.1 Glossary

| Term | Definition |
|---|---|
| **D**, $D_{\mathrm{NoPE}}$, $D_{\mathrm{RoPE}}$, $D_{\mathrm{QK}}$, $D_V$ | Head dims: 448 fp8 NoPE, 64 bf16 RoPE, 512 = 448 + 64 (= $D_V$ in V4 since PV consumes the full bf16-cast slice). |
| **Gen.1** | This kernel family: one wave per ptile, m=16, 8 ptiles per WG. A Gen.2 is anticipated (m=32, 2 waves/ptile, 4 ptiles/WG); the `_gen1` postfix reserves room. |
| **m16x8** | Naming convention: "m=16 mfma rows per ptile, 8 ptiles per WG." `W = H·mtp = 128`, kBlockN=64, V2 KV pipeline, occupancy 1. |
| **m16x4** | The `W = 64` sibling: 4 ptiles per WG, kBlockN=32, V1 KV pipeline, occupancy 2. |
| **LoNoPEWarp / HiRoPEWarp** | The two m16x8 warp types (4 warps each). Lo = tile 0 (cols 0–255, pure NoPE), PV-at-end, packed softmax, de-packed oaccu-normalize. Hi = tile 1 (cols 256–511, NoPE+RoPE), deferred PV, de-packed softmax, packed oaccu-normalize. |
| **deferred strip-3** | m16x8 lo-warp cross-iter pipeline: strip 3's cvt+store is moved to the next iter's top (§8.13). |
| **MTP** | Multi-token-prediction. Number of query tokens predicted per decode step. Different $(H, \mathrm{mtp})$ combos all satisfying $H \cdot \mathrm{mtp} = 128$ ride the same kernel template (Ch. 2.3). |
| **NoPE / RoPE** | Non-positional (fp8, scaled by E8M0) vs RoPE tail (bf16). The two halves of $D$ are loaded and laid out by different paths (Ch. 7, Ch. 8). |
| **oaccu** | The fp32 output accumulator. 16 rows × 512 cols, lives entirely in pinned `v128..v255` (128 vgprs/lane). |
| **p_comp / p_mfma** | Softmax output (m16x8): `p_comp` = fp32 (v48..v63, 16/lane), `p_mfma` = bf16 (v48..v55, 8/lane, overlay on p_comp's low half). |
| **Phase A / Phase B** | A **D-axis split** of QK: Phase A reads Q from pinned VGPR (Q[:, 0:256]), Phase B reads Q from LDS (Q[:, 256:512]). Not a warp-role swap. |
| **pong** | One of two LDS slots holding a KV tile (m16x8: 64 KiB; m16x4: 32 KiB). Swap each iter. |
| **prefetch chain** | The KvManager's `prefetch → cvt+store → wait` 3-routine split that lets vmem latency hide under QK mfma. |
| **ptile** | One processing tile = one group's worth of work. Gen.1: ptile = 1 wave. The term is local to this doc; the AMD term "Compute Unit" means something different. |
| **sb8 perm** | Sub-tile-of-8 permutation $[0,2,4,6,1,3,5,7]$ applied to a 64-col wave-tile's D-axis. Reorders 8 sub-tiles of 8 elements each; equivalent to swapping bits [3] and [5] of the col-element index. Eliminates the 2-way `ds_write_b128` writer-side bank conflict. Applied identically to Q and K (Ch. 7.2.4). |
| **setprio ladder** | $3 \to 2 \to 1 \to 0$ per-phase wave-priority drop within one main-loop iter. Lets the slower waves (KV writers) catch up while the faster waves (this one) are in compute-bound phases. |
| **Site C** | The 2-way `ds_read_b128` bank conflict on the V40 QK reader. Mitigated by a row-conditional half-swap on the vmem-load side (Method 2 — Method 1 silently fails; see Ch. 13.1). |
| **slim dispatch** | Compile-time flag (`MLA_SLIM_DISPATCH=1`) that always passes `kCheckBoundaryNext=true`, halving the `mla_main` template instantiations. Perf-neutral, 40 % smaller kernel image. (Ch. 12.4) |
| **wave-tile** | One ptile's mfma A-operand tile along the D-axis: 16 rows × 64 cols of bf16. Each KvManager call covers exactly one wave-tile per wave. |
| **work_idx** | Index into the metadata planner's per-WG work list. The persistent kernel processes multiple work_idxs from `work_indptr[wg .. wg+1]` (Ch. 12.8). |

### 15.2 Notation legend (reprint of Ch. 2.5)

| Symbol | Meaning | Range |
|---|---|---|
| $w$ | warp index | $[0, 8)$ |
| $\ell$ | lane index | $[0, 64)$ |
| $p$ | ptile index | $[0, 8)$ (in Gen.1, $p = w$) |
| $t$ | thread index in WG | $t = 64w + \ell$ |
| $i$ | KV chunk (tile) index in the main loop | $[0, \lceil N_{kv} / N_{\mathrm{block}} \rceil)$ |
| $N_{\mathrm{block}}$ | KV tile size along $N$ | 32 (= `kBlockN`) |
| $m \in [0, 16)$ | row in this ptile's mfma accumulator | one row = one work item |
