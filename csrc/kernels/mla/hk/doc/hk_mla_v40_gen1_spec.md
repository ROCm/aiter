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

Key numbers:

| Property | Value |
|---|---|
| Arch | gfx950 (MI350) |
| Tile name | **m16x8** — 8 ptiles per workgroup, each with m=16 rows |
| Ptile = | 1 wave (Gen.1 convention) |
| Total work per workgroup | $H \cdot \mathrm{mtp} = 128$ items |
| $D_{\mathrm{NoPE}}$ | 448 elements, fp8 (one E8M0 scale per 64 elements) |
| $D_{\mathrm{RoPE}}$ | 64 elements, bf16 |
| $D_{\mathrm{QK}} = D_V$ | 512 ($= D_{\mathrm{NoPE}} + D_{\mathrm{RoPE}}$) |
| Q residency | half pinned-VGPR (Phase A), half LDS (Phase B) |
| KV residency | LDS, double-buffered |
| Output residency | LDS bounce + VRAM (OManager V3 / V3NoStage) |
| Compiler scratch budget | `__attribute__((amdgpu_num_vgpr(64)))` — v0..v63 |

The wrapper file `csrc/kernels/mla/hk_v40_decode_fwd.cu` dispatches to this
kernel for the $H{=}128, \mathrm{mtp}{=}1$ case on gfx950; other valid
m16x8 configurations ($H{=}64/\mathrm{mtp}{=}2$, etc. — see Ch. 2) are
not yet wired but ride the same kernel template.

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
| $B_{\mathrm{rec}}$ | packed NoPE-record bytes / token (incl. dup E8M0 + pad) | 576 |
| $H$ | query heads sharing one KV head | varies (see 2.3) |
| $\mathrm{mtp}$ | multi-token-prediction tokens / step | varies (see 2.3) |
| $N_{kv}$ | KV context length for this batch element | varies |

The 576-byte packed record per token is $448$ fp8 NoPE values
+ $16$ duplicated E8M0 scale bytes + $112$ pad — see `kQkPackedNopeBytes`
in `hk_mla_utils.cuh`. The "576" is a *byte budget on disk*, not an
element count.

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

Today the host wrapper (`hk_v40_decode_fwd.cu`) only invokes the kernel
when `num_head * max_seqlen_q == 128`; the case it dispatches in
practice is $H{=}128, \mathrm{mtp}{=}1$. The other valid combinations
ride the same `mla_main` template and would Just Work once the wrapper
is taught to pick them.

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
| $N_{\mathrm{block}}$ | KV tile size along the $N$ dim | 32 (`kBlockN`) |
| $m \in [0,16)$ | row in this ptile's mfma accumulator | one row = one work item |

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
   │  Q-LDS (LDS half = │  │  KV-LDS, DOUBLE pong │     │
   │  Q[:, 256:512])    │  │  (32x512 bf16 each)  │◀────┘
   │  + Phase-1 staging │  │  buf_A / buf_B swap  │
   └─────────┬──────────┘  └──────────┬───────────┘
             │                        │
             │ ds_read_b128            │ ds_read_b128 (K side)
             │                        │ ds_read_b64_tr_b16 (V side, transpose)
             ▼                        ▼
   ┌────────────────────┐  ┌──────────────────────┐
   │ q_vgpr  v72-v103   │  │ kv     v112-v119     │
   │ (Q[:, 0:256])      │  │ kv_alt v104-v111     │
   │ q_lds   v64-v71    │  │ (pair-fused QK)      │
   └─────────┬──────────┘  └──────────┬───────────┘
             │                        │
             └─── v_mfma_f32_16x16x32_fp8_fp8 ───┘   ← QK
                              │
                              ▼
                  ┌─────────────────────┐
                  │ S tile (16x32 fp32) │   compiler scratch v0..v63
                  └─────────┬───────────┘
                            │ softmax (online)
                            │   v_max3 / warp_reduce(Max)
                            │   v_exp_f32 / warp_reduce(Add)
                            ▼
                  ┌─────────────────────┐
                  │ p_comp  v120-v127   │  (fp32, 16x32, 8 reg/lane)
                  │ p_mfma  v120-v123   │  (bf16 overlay, low half)
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
                            │   1) hk::mul_vgpr(oaccu, oaccu, 1/row_sum_e)
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

### 3.3 Phase A vs Phase B (D-axis split, not a warp swap)

QK is computed in **two phases** that differ in where $Q$ comes from. This
is a split along the $D_{\mathrm{NoPE}} = 512$ reduction dimension, not a
warp-role swap — every warp goes through both phases on every iteration.

| Phase | $Q$ source | $Q$ slice | mfma cols | Notes |
|---|---|---|---|---|
| Phase A | pinned VGPR `q_vgpr` (v72..v103) | $Q[:, 0{:}256]$ | 256 / 32 = 8 mfma tiles | 32 vgprs/lane carry the bf16 A-operand in mfma layout |
| Phase B | LDS `p_lds_q` via `q_lds` (v64..v71) | $Q[:, 256{:}512]$ | 256 / 32 = 8 mfma tiles | LDS read happens just-in-time; q_lds holds 2 paired tiles at a time |

Why split:

- $Q$ as a whole would be $16 \cdot 512 \cdot 2~\text{B} = 16$ KiB / ptile
  in bf16 — too big to fully pin. Pinning the first half (8 KiB) leaves
  budget for the other state in v64..v255.
- The LDS-resident half (Phase B) frees `q_vgpr` for *the next iter's*
  use too — same physical VGPRs, reused naturally because Phase B reads
  occur after Phase A has consumed them.
- Phase B's `q_lds` is split into two 4-vgpr slots so two adjacent
  Phase-B mfma tiles can pair-fuse exactly like the Phase-A pairs do.

There is no inter-warp swap of $Q$ ownership: each ptile keeps its own 16
rows in its own VGPRs/LDS. The KV side is what's shared across ptiles.

### 3.4 KV tile timing (one iter)

For iter $i$, ptile $p$ (= warp $w$), the stages are:

| Step | Work | Notes |
|---:|---|---|
| 1 | softmax + write `p_comp` | $S$ from prev tile |
| 2 | QK Phase A on tile $i$ | fused with ds_read of next KV from buf$_{(i+1)\bmod 2}$ AND `prefetch_kv_tile` of tile $i{+}2$ into buf$_{i\bmod 2}$ |
| 3 | QK Phase B on tile $i$ | Q from LDS |
| 4 | PV mfma on tile $i{-}1$ | prev `p_comp`; KV from buf$_{(i-1)\bmod 2}$ (already drained by step 2's reads) |
| 5 | `oaccu` rescale | $4 \times$ `v_mul_f32` interleaved between PV mfmas |

The KV bound is the LDS double-buffer, not VMEM — loads are hidden
under compute by the prefetch chain in step 2.

### 3.5 Who owns what

| Resource | Owner / manager | Lifetime |
|---|---|---|
| `q_vgpr` v72..v103 | `QManager8to16bitsV1::load_q` | Phase A of every iter |
| `q_lds` v64..v71 | `QManager8to16bitsV1::load_q_lds_pair` | within Phase B of one iter |
| Q-LDS region | `QManager8to16bitsV1` Phase 2 stages it | Phase B reads (whole loop) |
| KV-LDS buf_A / buf_B | `KvManager8to16bitsV1` | one iter (then recycled) |
| `kv` / `kv_alt` v104..v119 | `KvManager8to16bitsV1::load_k_to_gpr` | inside one Phase A/B mfma pair |
| `p_comp` v120..v127 | softmax (`hk_mla_softmax.cuh`) | between softmax and PV of one iter |
| `p_mfma` v120..v123 | PV gemm | reads from p_comp overlay |
| `oaccu` v128..v255 | PV gemm | whole loop |
| OMgr bounce LDS | `OManager16bitsV3` / `OManager32bitsV3*` | epilogue only |

The pinned VGPR layout is fixed by `__attribute__((amdgpu_num_vgpr(64)))`
on the `__global__` plus inline-asm hex names — see Ch. 5.

## Chapter 4 — LDS budget & layout

### 4.1 Budget and occupancy

V4 Gen.1 targets MI350 with `kOccupancy_=1`: one workgroup per CU at a
time, so the entire 160 KiB of LDS is available. The total budget at
that occupancy is bounded by

$$
\mathrm{kSzLdsQ} \,+\, \mathrm{kSzLdsKv} \,+\, \max(\mathrm{kSzLdsO},\, \mathrm{kSzLdsKv}) \le 160 \text{ KiB}
$$

(enforced by a `static_assert` in the kernel).

Concretely (from the manager `get_lds_size_in_byte()` accessors):

| Region | Size | Owner | Source |
|---|---:|---|---|
| `kSzLdsKv` (one KV pong) | 32 KiB | `KvManager8to16bitsV1` | $\mathrm{kBlockN} \cdot D_{\mathrm{QK}} \cdot \mathrm{sizeof(bf16)} = 32 \cdot 512 \cdot 2 = 32{,}768$ B (NoPE 448 + RoPE 64 = 512 bf16 cols per row, see Ch. 8) |
| `kSzLdsQ` (Q final + Phase-1 staging overlay) | 64 KiB | `QManager8to16bitsV1` | `kFinalLdsBytes` |
| `kSzLdsO` (OMgr V3 bf16 bounce) | 16,896 B (~16.5 KiB) | `OManager16bitsV3` | $8 \text{ warps} \cdot 2112~\text{B}$ |
| `kSzLdsO` (OMgr V3 fp32 split bounce) | 34,816 B (~34 KiB) | `OManager32bitsV3` | $8 \text{ warps} \cdot 4352~\text{B}$ |
| `kSzLdsO` (V3NoStage variant) | 0 | `OManager32bitsV3NoStage` | direct VRAM, no bounce |

Two epilogue managers are sized; the kernel picks `max(V3, V3NoStage)` —
the **V3** path (with bounce) is taken when the output is bf16 final and
the **V3NoStage** path when writing fp32 split output.

Total at the wired config:
$32 + 32 + \max(34, 32) = \mathbf{98}$ **KiB** active LDS, well under 160 KiB.

### 4.2 Layout (one snapshot)

```
LDS address (bytes, low → high):

+0                           p_lds_kv_curr   ← KV pong A (32 KiB)
+0x8000  (32 KiB)            p_lds_kv_next   ← KV pong B (32 KiB)
                                              ─ during epilogue:
                                                OVERLAID by OMgr bounce
                                                (V3 bf16 or V3 fp32)
+0x10000 (64 KiB)            p_lds_q         ← Q final + Phase-1 staging
                                              (64 KiB total)
+0x20000 (128 KiB)           — unused —
```

`p_lds_kv_curr` and `p_lds_kv_next` swap pointers every iteration:

| Iter | `p_lds_kv_curr` points to | `p_lds_kv_next` points to |
|---|---|---|
| $i$ even | LDS base | LDS base + 32 KiB |
| $i$ odd | LDS base + 32 KiB | LDS base |

So at any moment one pong is being *read* (QK / PV mfma sources) and the
other is being *written* (prefetch + cvt for the next KV tile).

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

1. `__attribute__((amdgpu_num_vgpr(64)))` on the `__global__` constrains the
   LLVM register allocator to use only `v0..v63` for its own scratch.
2. Inline asm that names registers in **hex form** (`v[0x73]`,
   `v[0x77:0x78]`) reserves `v64..v255` for hand-pinned data. The compiler
   cannot rename these.

If (1) is missing, the compiler may emit a decimal operand like `v100`
that silently overlaps with hand-pinned data — corruption with no diagnostic.
The auto-memory entry `[[check-unpinned-reg-usage]]` describes the audit
script (`.claude/skills/check-unpinned-reg-usage`) that catches this by
scanning the post-`-save-temps` `.s` file for decimal `v ≥ 64` in any
kernel body. **Run this audit after every nontrivial change to the V40
kernel** — current state is `budget=64 / spill=0 / free=0`, so there is
zero headroom.

### 5.2 The full map

Total per-lane pinned VGPR count: **192** (v64..v255). Compiler scratch:
**64** (v0..v63). Sum: 256, the per-lane register file.

| Range | Bytes/lane | Role | Owner / writer | Reader | Lifetime |
|---|---:|---|---|---|---|
| `v[0x00..0x3F]`<br/>v0..v63 | 256 | Compiler scratch (cvt staging, scale dwords, `ds_read_b64_tr` buffers, address arithmetic) | LLVM register allocator | LLVM | whole kernel; **budget=64, spill=0** |
| `v[0x40..0x47]`<br/>v64..v71 | 32 | `q_lds` — Phase-B Q-from-LDS scratch (2 paired tiles: q_k0 + q_k1, 4 vgprs each) | `QManager::load_q_lds_pair` (in-loop) | Phase-B QK mfma (A-operand) | per-iter inside Phase B |
| `v[0x48..0x67]`<br/>v72..v103 | 128 | `q_vgpr` — Q[:, 0:256] in mfma A-operand layout, 8 base tiles of 16×32 bf16 | `QManager::load_q` (prologue) | Phase-A QK mfma (A-operand) | whole loop (read-only after prologue) [^q-ro] |
| `v[0x68..0x6F]`<br/>v104..v111 | 32 | `pv_v_aux` — second V-tile staging during PV gemm; **overlaid as `kv_alt`** during QK Phase A | KvManager (QK), PV gemm (PV) | PV mfma (B-operand) | role-toggled each phase |
| `v[0x70..0x77]`<br/>v112..v119 | 32 | `kv` — single 32×16 KV tile carrier (top half + bot half), no `kv_alt` shadow | `KvManager::load_k_to_gpr` | QK mfma (B-operand) | inside one mfma pair |
| `v[0x78..0x7B]`<br/>v120..v123 | 16 | `p_mfma` — bf16 P operand for PV, **overlaid on the low half of `p_comp`** | softmax cvt (bf16) | PV mfma (A-operand) | between softmax and PV of one iter |
| `v[0x78..0x7F]`<br/>v120..v127 | 32 | `p_comp` — fp32 softmax output, 8 fp32/lane covering 16×32 P-tile | softmax | softmax (rescale next iter), PV (via p_mfma overlay) | until PV consumes it |
| `v[0x80..0xFF]`<br/>v128..v255 | 512 | `oaccu` — fp32 output accumulator, 128 fp32/lane covering 16×512 | PV mfma (C/D-operand) | epilogue (OManager) | whole loop |

[^q-ro]: Confirmed read-only after Phase 1 prologue — see auto-memory
`[[v40-pinned-q-read-only-confirmed]]`. This matters when debugging a
mismatch: any V40 bug that looks like "Q got corrupted mid-loop" is
**not** a Q-VGPR clobber; look downstream.

### 5.3 Why the layout is shaped this way

Going low → high (toward v255), the rationale per range:

- **v0..v63 (compiler scratch).** Bounded by `amdgpu_num_vgpr(64)`. Holds
  the per-iter dynamic state: address arithmetic for `buffer_load_lds`,
  `ds_read_b64_tr` destination dwords, softmax intermediates, e8m0 scale
  dwords. **Zero free VGPRs and zero spill** is the current measurement —
  any new pinned data must come out of v64..v255, not here.
- **q_lds (v64..v71).** Phase B reads Q from LDS *into* these on every
  iter. Split into two 4-vgpr halves (`q_k0`, `q_k1`) so two adjacent
  Phase-B mfma tiles can issue back-to-back with one `ds_read` chain
  feeding both — same pair-fusion pattern as Phase A.
- **q_vgpr (v72..v103).** Phase A reads from these for the whole loop.
  Holds $Q[:, 0{:}256]$ in mfma A-operand layout = $(16 \text{ rows}) \cdot (256 \text{ cols}) / 64 \text{ lanes} / 2 \text{ elems/vgpr}$
  $= 32$ vgprs/lane. Placed adjacent to `q_lds` for symmetry.
- **pv_v_aux / kv_alt (v104..v111).** Dual-role: during QK Phase A,
  holds the *next* KV tile's V-side (called `kv_alt`) for pair-fused
  mfma; during PV, holds the staging V data for the second mfma B-operand.
  The dual role is safe because the QK→PV transition has an `s_waitcnt`
  barrier in between.
- **kv (v112..v119).** Single 32×16 KV carrier. There's no `kv_alt`
  shadow at this address — the alternate-tile carrier reuses pv_v_aux
  as noted above. Spec §4.2 explains the reasoning: one extra 8-vgpr
  shadow at this address would push p_comp past v127 and break the
  oaccu start address.
- **p_mfma / p_comp (v120..v127).** Overlay: `p_mfma` (bf16, 4 vgprs)
  occupies the **low half** of `p_comp` (fp32, 8 vgprs). The overlay is
  safe because softmax-to-PV is `low-to-high pack`: softmax writes
  high → low in the fp32 layout, then the cvt to bf16 packs the result
  back into the low half exactly where PV expects to read it. See
  the pinned-DST cvt helper `pack_2f32_to_bf16_pair_pinned` in
  `hk_mla_utils.cuh` (gotcha: don't use the runtime-arg
  `float_2_bf16_pair` form here — see Ch. 13).
- **oaccu (v128..v255).** The biggest single block, 128 vgprs. Lives at
  the top of the register file so its base address is a round
  `0x80` — simplifies the inline-asm offset arithmetic in the OManager
  epilogue. Holds $(16 \text{ rows}) \cdot (512 \text{ cols}) / 64 \text{ lanes}$
  $= 128$ fp32/lane.

### 5.4 Compiler-scratch budget audit

The audit script reports three numbers:

| Number | Meaning | Current value |
|---|---|---:|
| budget | `N` from `amdgpu_num_vgpr(N)` | 64 |
| spill | `.vgpr_spill_count` in the kernel metadata | 0 |
| free gprs | `N - max_observed_decimal_v - 1` | 0 (max observed v63) |

Zero free + zero spill = "fits exactly." Any new pinned data, new
inline-asm clobber, or wider unroll could push the compiler over budget
into spill. Always re-run the audit after touching:

- inline asm clobber lists in the managers
- `static_for` unroll factors in the kernel body
- any new `sched_barrier(0)` (it widens live ranges)

See `.claude/skills/check-unpinned-reg-usage/` for the script.

## Chapter 6 — QManager Phase 1 (vmem → staging LDS → pinned q_vgpr)

Phase 1 fills the **VGPR half** of $Q$: $Q[:, 0{:}256]$ (the first 256 of
the 448 NoPE elements) into pinned `q_vgpr` v72..v103. It runs once at
the prologue, before the main loop.

### 6.1 Geometry

| Symbol | Meaning | Value |
|---|---|---:|
| `kVgprHalfCols` | NoPE cols going to VGPR | 256 |
| `kP1ChunkCols` | cols per Phase-1 chunk | 64 |
| `kP1NumChunks` | chunks needed for 256 cols | 4 |
| `kP1StagingBytesPerWarp` | per-warp staging slot, one buffer | $16 \cdot 64 \cdot 1 = 1024$ B |
| `kP1NumStagingBuffers` | double-buffer slots | 2 |
| `kP1StagingBytesPerWarpTotal` | per-warp staging, both buffers | 2048 B |
| `kPackedNopeStride` | source bytes per token | 576 |
| `kScaleBaseOff` | first E8M0 scale byte in the 576-byte record | 448 |

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
\mathit{vOff}(\ell) = (\ell \gg 2) \cdot 576 + (\ell  \mathbin{\mathrm{and}}  3 \oplus (S \ll 1)) \cdot 16
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
576-byte record:

$$
\mathit{vOffScale}(\ell) = (\ell  \mathbin{\mathrm{and}}  15) \cdot 576, \qquad \mathit{iOff} = 448 + 2c
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
so the kernel places Q **after** the 32 KiB KV pongs + the O bounce —
giving warp 0's staging at least 192 B of preceding LDS to absorb the
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
\mathit{vOffNope}(\ell) = (\ell \gg 2) \cdot 576 + (\ell  \mathbin{\mathrm{and}}  3) \cdot 16, \quad \mathit{iOff} = 256 + 64 c
$$

with chunk $c \in \{0,1,2\}$. The vmem side is **straight** (no
swizzle) — the bank-conflict swizzle is on the LDS-write side instead,
mirroring KvManager.

**Per-lane vmem offset (scale):**

$$
\mathit{vOffScale}(\ell) = (\ell \gg 2) \cdot 576 + ((\ell  \mathbin{\mathrm{and}}  3) \gg 1), \quad \mathit{iOff} = 448 + c \cdot 2
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

KV is the dominant bandwidth consumer (one new 32-row tile per iter,
$32 \cdot 576$ B fp8 + $32 \cdot 128$ B bf16 RoPE = ~22 KiB / iter)
and the only inter-iteration LDS resident. The KvManager hides VMEM
latency via a **double-buffer pong** scheme: while iter $i$'s compute
reads from the *current* pong, iter $i+1$'s cvt+store fills the
*next* pong, and iter $i+2$'s prefetch issues into either pong's
in-flight slot.

### 8.1 Geometry and pong layout

| Symbol | Value | Source |
|---|---:|---|
| `kBlockN` | 32 | rows per KV tile |
| `kQkNopeHeadDim` | 448 | fp8 NoPE cols per token |
| `kQkRopeHeadDim` | 64 | bf16 RoPE cols per token |
| `kQkHeadDim` | 512 | $= D_{\mathrm{QK}}$ — total bf16 cols per row in LDS |
| `kSubBlockRows × kSubBlockCols` | 16 × 32 bf16 | one sub-block (= one QK A-tile) |
| `kSubBlockBytes` | 1024 | |
| `kNumRowTiles` | 2 | row-halves per KV tile |
| `kNumColTiles` | 16 | col-tiles per KV tile (= 512/32) |
| `kNumColTilesNope` | 14 | NoPE cols span col-tiles [0..14) |
| `kNumColTilesRope` | 2 | RoPE cols are col-tiles {14, 15} |
| `kTileCols` | 256 | one *half* of a KV tile along D (= 512/2) |
| `kColTilesPerTile` | 8 | col-tiles in one half |
| `kWaveColTilesPerWaveTile` | 2 | each wave owns 2 col-tiles (= 64 bf16 cols) |
| `kWaveTileCols` | 64 | per-wave col-tile width |
| **One pong** | $32 \cdot 512 \cdot 2 = $ **32 KiB** | full $\mathrm{kBlockN} \cdot D_{\mathrm{QK}}$ in bf16 |

Per pong, the 32×512 bf16 region is viewed as $2 \cdot 16 = 32$
sub-blocks of $16 \times 32$ bf16 each, stored in **col-major sub-block
order**:

$$
\mathit{subBlockByteOffset}(r_{\mathrm{tile}}, c_{\mathrm{tile}}) = (c_{\mathrm{tile}} \cdot 2 + r_{\mathrm{tile}}) \cdot 1024
$$

with $r_{\mathrm{tile}} \in \{0, 1\}$ (which 16-row half) and
$c_{\mathrm{tile}} \in [0, 16)$ (which 32-col strip). Strips 0..13 are
NoPE; strips 14..15 are RoPE.

### 8.2 Wave → tile map (Option 2, branchless)

The 8 waves of the workgroup partition the 32×512 KV tile into 4×4 grid
of $16 \times 64$ wave-tiles, with each wave owning one wave-tile per
half-tile (`kTileIdx ∈ {0,1}`):

$$
r_{\mathrm{tile}}(w) = (w \gg 1)  \mathbin{\mathrm{and}}  1, \qquad
c_{tileInHalf}(w) = ((w \gg 1)  \mathbin{\mathrm{and}}  2) \,|\, (w  \mathbin{\mathrm{and}}  1)
$$

resulting in:

| wave $w$ | $r_{\mathrm{tile}}$ | $c_{tileInHalf}$ | rows | cols in tile-0 | cols in tile-1 |
|---:|---:|---:|---|---|---|
| 0 | 0 | 0 | 0..15 | [0,64) | [256, 320) |
| 1 | 0 | 1 | 0..15 | [64,128) | [320, 384) |
| 2 | 1 | 0 | 16..31 | [0,64) | [256, 320) |
| 3 | 1 | 1 | 16..31 | [64,128) | [320, 384) |
| 4 | 0 | 2 | 0..15 | [128,192) | [384, 448) |
| 5 | 0 | 3 | 0..15 | [192,256) | **[448, 512) = RoPE** |
| 6 | 1 | 2 | 16..31 | [128,192) | [384, 448) |
| 7 | 1 | 3 | 16..31 | [192,256) | **[448, 512) = RoPE** |

For half-tile 1 (`kTileIdx == 1`), waves 5 and 7 land on
$c_{tileInHalf} = 3$, which corresponds to global col-tiles
{14, 15} = the RoPE region. Those two waves take the **RoPE path**
(direct vmem→LDS via `buffer_load_lds`, no cvt — RoPE is already bf16).
All other (wave, tile) combinations take the **NoPE path** (vmem fp8 →
VGPR → cvt → ds_write bf16).

`wave_is_rope_owner(w) = (w == 5) || (w == 7)`.

### 8.3 The pong swap

Two LDS pointers, swapped each iter:

| Iter $i$ parity | `p_lds_kv_curr` | `p_lds_kv_next` |
|---|---|---|
| even | LDS base + 0 | LDS base + 32 KiB |
| odd | LDS base + 32 KiB | LDS base + 0 |

At the entry of iter $i$, `p_lds_kv_curr` holds the **already-finished**
tile to compute on; `p_lds_kv_next` is the slot that the **prefetch +
cvt+store** of the *next* tile writes into. The pointers are swapped
at the bottom of the iter — no LDS data movement, just pointer math.

### 8.4 Prefetch / store / consume timeline

The KvManager splits each pong fill into 3 routines so the main loop
can interleave them with QK mfmas:

| Routine | What it does | Latency hidden behind |
|---|---|---|
| `prefetch_kv_tile<kRowOffset, kColOffset, kCheckBoundary>` | issues `buffer_load_dwordx4` (16 fp8) + `buffer_load_ubyte` (E8M0 scale) per lane → `KvTilePrefetch` carrier; or `buffer_load_lds` direct for RoPE | vmem latency hidden by the QK mfmas that follow |
| `wait_kv_loads<kRowOffset, kColOffset, kVmCnt=0>` | drains vmcnt to the requested level + `sched_barrier(0)` | (just a wait, no compute) |
| `cvt_kv_tile_step<kStep>` × 4 + `store_kv_tile_step<R,C,kStep>` × 2 | 4 cvts to bf16 dwords + 2× `ds_write_b128` per tile | cvt and store latencies overlap with mfma slots in the QK loop |
| `kv_tile_scale_f(prefetch)` | one ALU op: e8m0 → fp32 scale | hoisted once per tile |

For the prologue (cold start) the convenience wrapper `async_load_k` does
all of this non-overlapped: prefetch both half-tiles → wait → cvt+store
both. The main loop uses the split form to interleave with mfmas — see
the iteration timeline in §3.4.

### 8.5 NoPE prefetch — `prefetch_kv_tile` (NoPE branch)

Address split:

| Field | Per-lane / wave-uniform / immediate | Expression |
|---|---|---|
| `v_offset` (per-lane) | NoPE fp8 | $\mathit{rowKvLd} \cdot 576 + \mathit{colGroupSwz} \cdot 16$ |
| `s_offset` (wave-uniform) | NoPE fp8 | $c_{tileInHalf} \cdot 64$ |
| `i_offset` (immediate) | NoPE fp8 | $\text{kTileIdx} \cdot 256$ |
| `v_offset` (per-lane) | scale | $\mathit{rowKvLd} \cdot 576$ |
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

## Chapter 9 — Softmax

Online (Flash-style) softmax runs **once per KV tile**, between QK and PV.
It updates two per-row running scalars ($m$, $\ell$) and produces the
fp32 P-tile in `p_comp` (v120..v127), then packs it into bf16 `p_mfma`
(v120..v123 overlay) for PV.

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
`new_row_max = local_max`, and `oaccu`'s rescale is skipped.

### 9.2 Where $m$ and $\ell$ live

| State | Storage | Notes |
|---|---|---|
| $m$ (per row, this lane's share) | `float row_max;` local | 1 fp32/lane, persists across iters |
| $\ell$ (per row, this lane's share) | `float row_sum_e;` local | 1 fp32/lane, persists across iters |
| $S^{(i)}$ / $P^{(i)}$ | `p_comp` v120..v127 | 8 fp32/lane, 16×32 tile per warp |
| $P^{(i)}$ in bf16 for PV | `p_mfma` v120..v123 (overlay on p_comp low half) | 4 bf16x2 dwords/lane |
| $\alpha$ (rescale for this iter) | `float rescale;` local | 1 fp32/lane, lives only within this iter |

These compiler-scratch fp32 scalars sit in v0..v63 and are recreated
each iter (LLVM is free to choose where).

### 9.3 The three softmax routines

The kernel uses three helpers from `hk_mla_softmax.cuh`:

| Routine | Inputs / Outputs | What it does |
|---|---|---|
| `softmax_scale_p<kCheckBoundary, GPR>(col_start_idx, kv_end, softmax_scale)` | reads p_comp, writes p_comp | Element-wise multiply by `softmax_scale` ($= 1/\sqrt{D_{\mathrm{QK}}}$). On boundary tiles (`kCheckBoundary=true`), out-of-range cols are set to a very negative value so the upcoming exp produces 0. |
| `softmax_p0<kIsFirstIter, kCheckBoundary, GPR>(...)` | reads p_comp, writes row_max, rescale | Computes local row max (v_max3 ladder + warp_reduce), updates running $m$, computes $\alpha$. |
| `softmax_p1<kIsFirstIter, GPR>(...)` | reads p_comp + new_row_max + rescale, writes p_comp, row_sum_e | In-place exp: `p_comp ← exp_2((p_comp − new_row_max) · log2e)`. Then warp-reduces row sum and updates running $\ell$. |

The kernel does NOT call `softmax_p0` directly — it inlines the row-max
+ rescale logic and uses `max_8<…>` + `warp_reduce` directly. This is
equivalent and lets the kernel hoist `kCheckBoundary` into a runtime
branch around just the `softmax_scale_p` call. See lines 720–760 of the
kernel.

### 9.4 The v_max3 ladder

The per-lane local max over 8 fp32 values reduces to **4 instructions**
(2× `v_max3_f32` + 1× `v_max_f32` + 1× `v_max3_f32`), the gfx950-minimum
for 8 inputs:

$$
\mathit{localMax} = \max\big(\max_3(\mathit{p0}, \mathit{p1}, \mathit{p2}), \max_3(\mathit{p4}, \mathit{p5}, \mathit{p6}), \max(\mathit{p3}, \mathit{p7})\big)
$$

`max_8<>` in `hk_mla_utils.cuh` ships this ladder; the kernel inlines
the same shape. After the per-lane reduce, `warp_reduce<MaxFunctor>`
reduces across the 4 lanes per row-group via DPP / cross-lane permutes
— the mfma A-operand layout puts 16 rows × 4 lanes/row, so the
reduction window is 4 lanes wide.

### 9.5 The 8× exp + 4× pk_add in `softmax_p1`

`softmax_p1` emits one big inline-asm block:

1. `v_pk_add_f32` × 4: `p_comp[i:i+1] += {-m, -m}` (pair-add) for $i \in \{0,2,4,6\}$.
2. `v_pk_mul_f32` × 4: `p_comp[i:i+1] *= log2e_pk`.
3. `v_exp_f32` × 8: `p_comp[i] = v_exp_f32(p_comp[i])` for $i \in [0, 8)$.

Then 3 pair-adds reduce `p_comp[0..7]` to a scalar `local_sum_e`, which
warp-reduces to `row_sum_e`.

The fused asm is one block (not multiple smaller blocks) because the
compiler otherwise scatters the v_exp issues across the inline-asm
boundaries, breaking the back-to-back issue pattern that hides exp
latency.

### 9.6 Pack to bf16 for PV: `pack_2f32_to_bf16_pair_pinned`

After `softmax_p1`, `p_comp` holds 8 fp32 / lane. PV's mfma needs bf16,
so the kernel issues **4** `v_cvt_pk_bf16_f32`s via
`pack_2f32_to_bf16_pair_pinned<DST, SRC>()`, with destinations
`p_mfma[0..3]` and sources `p_comp[0,2,4,6]`. With
`k_p_mfma_begin = k_p_comp_begin = 120`, the destination
`p_mfma[0..3]` **overlays the low half of `p_comp[0..7]`** — the
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

- `p_mfma` v120..v123 holds $P^{(i)}$ in bf16, ready for PV mfma.
- `p_comp` v120..v127 still holds the same data (low half bf16-overlay,
  high half stale fp32 — but PV only reads the low half via p_mfma).
- `row_max` and `row_sum_e` are updated; `rescale` ($= \alpha$) is in
  a local fp32 and is **the value passed to PV** to rescale `oaccu`
  inside the PV gemm loop.

### 9.9 Final normalization (after the loop)

At the end of the main loop (in the epilogue branch of `mla_main`),
each row's `oaccu` is divided by its final `row_sum_e` via a single
`hk::mul_vgpr(oaccu, oaccu, 1.0f / row_sum_e)` over the full
128-vgpr `oaccu` tile — one `v_mul_f32` per vgpr. The OManager
epilogue routines then write the normalized `oaccu` to VRAM
unchanged.

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

producing a 16×16 fp32 output tile. With `kBlockN = 32` and
`kVoHeadDim = 512`, PV runs in **16 iters** (each iter covers 32 V-cols
= 2 mfma A-tiles = 2 base tiles of `kv`).

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

After 4 reads, `kv` (v112..v119) holds 2 mfma A-tiles' worth of
$V^\top$ data — 8 vgprs/lane = `kv_top` (4 vgprs) + `kv_bot` (4 vgprs).

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
which is exactly the count needed to multiply the full 128-vgpr `oaccu`
by `rescale` once.

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
| `OManager16bitsV3` | bf16 → final_output | 2112 B / warp ($\approx 16.5$ KiB) | `kEpilogueType = OutputFinal`, the common case |
| `OManager32bitsV3` | fp32 → split_output | 4352 B / warp ($\approx 34$ KiB) | `kEpilogueType = OutputSplit`, with a bounce, when split-O LSE is needed |
| `OManager32bitsV3NoStage` | fp32 → split_output | 0 (direct) | `kEpilogueType = OutputSplit`, direct write when the LDS region is contended by the split-O reduction (see commit `15a8736c4`'s notes) |

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

- **`OManager32bitsV3`** uses a 4352 B/warp bounce — same un-swizzle
  pattern as the bf16 V3, scaled to fp32 with `kNumElemPerPaddedRow = 68`
  (= 64 + 4 pad). Total per-WG bounce = $8 \cdot 4352 = 34816$ B.
- **`OManager32bitsV3NoStage`** writes directly from oaccu VGPR to
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

A prior version of `OManager32bitsV3` (and `V3NoStage`) issued a
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

### 12.2 Iter classification

For each warp, the host wrapper walks `[kv_start, kv_end)` in steps of
`kBlockN = 32` and classifies each iter:

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

### 13.4 Confirmed-not-a-bug

| # | What was suspected | What's actually true |
|---:|---|---|
| 8 | The pinned Q VGPRs (v72..v103) might be getting clobbered mid-loop (any V40 numerical mismatch). | Phase A's pinned-q region is **read-only** after Phase 1 prologue. Proven by ISA scan + runtime probe ([[v40-pinned-q-read-only-confirmed]]). When a V40 mismatch is being debugged, eliminate Q as suspect first — look downstream (K/V load, sb8 perm, OMgr un-perm). |

### 13.5 Tools to reach for

- `check-unpinned-reg-usage` skill (`.claude/skills/check-unpinned-reg-usage/`)
  — scans the post-`-save-temps` `.s` file for decimal `v ≥ 64` and
  spill > 0. Run after every nontrivial change. Current state:
  `budget=64 / spill=0 / free=0`. Zero headroom.
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
| `csrc/kernels/mla/hk_v40_decode_fwd.cu` | Host wrapper. Dispatches to the kernel for the wired `(H=128, mtp=1)` case on gfx950. |
| `csrc/kernels/mla/hk/mi35x_v40_fwd_decode_m16x8_fp8bf16_fp8bf16_gen1.cuh` | Main kernel: persistent work loop, `mla_main` lambda + dispatch ladder, pinned-VGPR & LDS layout, QK Phase A/B fusion, softmax, PV gemm + oaccu rescale, epilogue. |
| `csrc/kernels/mla/hk/hk_mla_v40_buffer_managers_gen1.cuh` | V40-only managers: `QManager8to16bitsV1`, `KvManager8to16bitsV1`, `OManager16bitsV3`, `OManager32bitsV3`, `OManager32bitsV3NoStage`. Also the sb8 perm helpers (`sb8_perm_col_elems`, `sb8_inv_perm_col_elems`). |
| `csrc/kernels/mla/hk/hk_mla_softmax.cuh` | Online-softmax helpers: `softmax_scale_p`, `softmax_p0`, `softmax_p1`, `softmax_p1_16`. Free functions, no class state. |
| `csrc/kernels/mla/hk/hk_mla_utils.cuh` | Shared with V32: traits (`HkMlaV40DecodeFwdTraits`, `HkMlaV32DecodeFwdTraits`), enums (`PvGemmEpilogueType`), and helpers under `namespace hk_mla`: `e8m0_to_f32`, `encode_s_waitcnt`, `max_8`, `sum_8`, `warp_reduce`, `cvt_scalef32_pk_bf16_fp8_pinned`, `pack_2f32_to_bf16_pair_pinned`, `float_2_bf16_pair`, `get_kv_ld_row`. |

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
| `.claude/skills/check-unpinned-reg-usage/` | ISA audit script — scans the post-`-save-temps` `.s` for decimal `v ≥ 64` and `spill > 0`. Run after every nontrivial change to this kernel. Current budget: 64 / spill: 0 / free: 0. |

## Chapter 15 — Glossary & cross-refs

### 15.1 Glossary

| Term | Definition |
|---|---|
| **D**, $D_{\mathrm{NoPE}}$, $D_{\mathrm{RoPE}}$, $D_{\mathrm{QK}}$, $D_V$ | Head dims: 448 fp8 NoPE, 64 bf16 RoPE, 512 = 448 + 64 (= $D_V$ in V4 since PV consumes the full bf16-cast slice). |
| **Gen.1** | This kernel family: one wave per ptile, m=16, 8 ptiles per WG. A Gen.2 is anticipated (m=32, 2 waves/ptile, 4 ptiles/WG); the `_gen1` postfix reserves room. |
| **m16x8** | Naming convention: "m=16 mfma rows per ptile, 8 ptiles per WG." |
| **MTP** | Multi-token-prediction. Number of query tokens predicted per decode step. Different $(H, \mathrm{mtp})$ combos all satisfying $H \cdot \mathrm{mtp} = 128$ ride the same kernel template (Ch. 2.3). |
| **NoPE / RoPE** | Non-positional (fp8, scaled by E8M0) vs RoPE tail (bf16). The two halves of $D$ are loaded and laid out by different paths (Ch. 7, Ch. 8). |
| **oaccu** | The fp32 output accumulator. 16 rows × 512 cols, lives entirely in pinned `v128..v255` (128 vgprs/lane). |
| **p_comp / p_mfma** | Softmax output: `p_comp` = fp32 (v120..v127, 8/lane), `p_mfma` = bf16 (v120..v123, 4/lane, overlay on p_comp's low half). |
| **Phase A / Phase B** | A **D-axis split** of QK: Phase A reads Q from pinned VGPR (Q[:, 0:256]), Phase B reads Q from LDS (Q[:, 256:512]). Not a warp-role swap. |
| **pong** | One of two 32 KiB LDS slots holding a KV tile. Swap each iter. |
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
