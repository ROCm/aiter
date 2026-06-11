# FP8 ASM FMHA vs CK-UA prefill — why it's ~2× and what to copy

**Shape studied:** `b=1, hq=hk=5, sq=sk=75600, d=dv=128, fp8 e4m3, non-causal`
(the `att_std_d128_fp8_noncausal_sq75600_h5` reference; FLOP = `2·b·hq·sq·sk·(d+dv) ≈ 1.463e13`).

This is the **fp8** sibling of the i8fp8 kernel in `i8fp8_asm_kernel_runbook.md`. Per the
colleague, the fp8 kernel runs on any recent branch — no worktree / no C++ delta — you
just rebuild the `.co` and drop it in the slot. Done; it is now wired into our test.

---

## 0. TL;DR

| Kernel | time | TFLOPS | vs CK-UA |
|---|---|---|---|
| **CK-UA** (ours, contiguous fp8, narrow 32x32x16) | 12.50 ms | **1171** | 1.00× |
| **CK-UA + wide 32x32x64 MMA** (kFp8WideMma, kv64) | **8.95 ms** | **1634** | **1.40× faster than narrow** |
| **ASM fp8** (`fwd_hd128_fp8.co`) — `@perftest` median | 6.94 ms | **2107** | 1.80× / **1.28× vs wide** |
| **ASM fp8** — standalone CUDA-event min | 6.34 ms | **2306** | 1.95× faster |

**UPDATE (wide-MMA milestone).** Switching the CK fp8 d128 prefill QK/PV warp gemm from the
narrow `v_mfma_f32_32x32x16` to the wide `v_mfma_f32_32x32x64_f8f6f4` (the same MMA the ASM
kernel uses) is a **+40%** standalone/harness win (1171 → 1634 TFLOPS), closing the gap to
the ASM kernel from **1.80× → 1.28×** (UA/ASM ratio 0.556 → 0.782). The enabler was that the
QK-C→PV-A FP8 P-relayout is barrier-free at 32x32x64 (cvt-only, no permute — same as the ASM
`_softmax_pack_P_fp8`), so it stays FA4-safe. Gated behind `-DUA_FP8_WIDE_MMA` (default on).
Harness 8.95ms == standalone 8.94ms (cross-validated). Attempts to also raise kTileKV to 128
were a dead end: 32x32x64+kv128 spills 617 VGPRs (256 cap), and a 16x16x128+kBlockM=128
variant that fits is ~3× slower (16x16 MFMA efficiency penalty). The remaining 1.28× is the
structural pipeline gap (symmetric compute / loader split / per-MFMA interleave) + the two
numeric tricks (exp2 ~+8%, rowmax-freeze ~+5%) documented below.

Correctness of the rebuilt `.co`: cosine **0.9984** (non-causal) / **0.9988** (causal) vs torch.

**The gap is mostly STRUCTURAL, not numeric.** Building the ASM kernel with both numeric
approximations OFF (exact softmax, same pipeline) still gives **2055 TFLOPS = 1.75× CK-UA**
(see §3.5). The Schraudolph `exp2` and the max-freeze/rollback together add only ~14% on
top. So the bulk of the loss is in how our **pipeline overlaps memory + MFMA + softmax**,
which matches our own ATT findings (exposed `memwait`, SOFTMAX phase ~7-12% longer than
MATRIX → barrier idle, only ~31-33% phase overlap). The three things they have that we do
not, re-ordered by *measured* contribution:

1. **Symmetric 8-wave compute + load-role specialization** (no softmax/matrix phase
   imbalance, fine-grained load/MFMA interleave) — holds most of the 1.75×.
2. **Schraudolph full-rate `exp2`** (we use the quarter-rate HW transcendental) — +8%.
3. **Max-freezing softmax** that *also skips the per-tile rowmax reduction* (we only skip
   the rescale today) — +5%.

---

## 1. How to run the comparison (wired into our test)

A new `--asmfp8` leg was added to `op_tests/test_unified_attention_ck.py`, mirroring
`--sagev1` (dense, per-tensor fp8, single-shape / non-SWA / non-top-left). It dispatches
`flash_attn_fp8_pertensor_func` → v3 ASM launcher → `hsa/gfx950/fmha_v3_fwd/fwd_hd128_fp8.co`.

```bash
# env: /venv has the ABI-correct torch (2.9.1, the one the CK JIT modules were built
# against) + triton symlinked in. aiter-venv's torch (2.11dev) segfaults on CUDA alloc
# on this box, unrelated to the kernel — use /venv.
GPU=0   # pick an idle one: rocm-smi --showuse
HIP_VISIBLE_DEVICES=$GPU /venv/bin/python3 op_tests/test_unified_attention_ck.py \
  -b 1 -sq 75600 -sk 75600 --num-heads 5,5 --head-size 128 --block-size 128 \
  --dtype fp8 --contiguous --asmfp8 --mask-type 0 --no-triton --no-reference
# report prints "ASMfp8 TFLOPs" and "UA vs ASMfp8 = 0.556x (ASMfp8 wins)".
```

Rebuild the `.co` from source (toggles `_USE_ROLLBACK` / `_USE_EXP2_APPROX` live at the top):

```bash
cd /root/diffusion-models-inference-private
python3 asm/fmha_sage_fwd/gfx950/mi350_fmha_hd128_fp8.py /tmp/fp8.s \
  --symbol _ZN5aiter24fmha_fwd_hd128_fp8_gfx950E
/opt/rocm/llvm/bin/clang++ -target amdgcn-amd-amdhsa -mcpu=gfx950 -x assembler /tmp/fp8.s -o /tmp/fp8.co
cp /tmp/fp8.co /root/aiter/hsa/gfx950/fmha_v3_fwd/fwd_hd128_fp8.co   # stock backed up as *.stock_bak
```

Static budget of the `.co`: **224 VGPR, 95 SGPR, 99,840 B LDS, 0 spills**, 2 waves/SIMD target.

---

## 2. ASM kernel architecture (gfx950, 512 threads = 8 waves)

Source: `asm/fmha_sage_fwd/gfx950/mi350_fmha_hd128_fp8.py`. `kTileQ=256, kTileKV=128`,
MFMA `v_mfma_f32_32x32x64_f8f6f4` for both QK and PV; P repacked to fp8 between them; O bf16.

**Wave roles (load specialization, NOT compute ping-pong).** `process_current_work()`
splits on `wave_id < 4`:
- waves **0-3** = **K-loaders** (`is_wave47=False`): stream each K tile HBM→LDS.
- waves **4-7** = **V-loaders** (`is_wave47=True`): stream each V tile HBM→LDS.
- **All 8 waves run the *same* `core_loop`**: `gemm_QK → fmha_softmax → gemm_PV`, over
  their own 32 query rows (`kSubSizeQ = 256/8`). V-LDS is shared and read transposed
  (`ds_read_b64_tr_b8`) by all 8 waves in PV.

**The "8-wave ping-pong" is a skew + issue-priority handoff, not a phase split.** The two
4-wave groups are phase-skewed and hand off the VALU issue port via `s_setprio`:
`gemm_QK` exits with `s_barrier(); s_setprio(0); s_barrier()` and `gemm_PV` enters with
`… s_setprio(1); s_barrier()`. So while one group is in MFMA (QK/PV), the other is in the
VALU softmax — the MFMA of one group hides the softmax of the other, with all waves on one
code path. (Contrast our design in §3.)

**Fine-grained latency hiding inside the MFMA loop.** `gemm_QK`/`gemm_PV` interleave, per
MFMA, *either* a `mem_load_K/V` (HBM→LDS for the **next** tile, `rep_idx^1`, double-buffered)
*or* a `lds_read_K/V` (LDS→VGPR for the current tile), governed by a running `lgkmcnt`
budget (`wait_cnt`). The K subtile-(n+1) LDS read is issued under subtile-n's MFMAs; V loads
drain just before the PV barrier so they overlap softmax. Loads are spread across the math,
not bunched at a prefetch point.

**Softmax — two big approximations (both on by default):**

- `_USE_EXP2_APPROX=True` — **Schraudolph 2^x**. The 64 per-tile exponentials are *not*
  `v_exp_f32` (quarter-rate transcendental). They are an affine map
  `bits = round(S·AC + bias)` via `v_fma_f32` + `v_cvt_u32_f32` reinterpreting the float
  bit pattern as 2^x — **full-rate VALU**. The per-tensor scale·descale·log2(e) is folded
  into the SGPR `_s_exp2_AC` so the QK MFMA runs on raw fp8 and the exponent carries all
  three factors.
- `_USE_ROLLBACK=True` — **max-freezing softmax with detect-after-exp + rollback**. After
  the first tile seeds `FA_max`, no-mask tiles **freeze the running max**: `gemm_QK` runs
  `with_max3=False` (no per-tile rowmax), and `_softmax_rescale_R_frozen` **drops the 32
  `v_pk_mul` O/L rescale** (delta≡1). Safety is a cheap per-lane row-sum + wave-OR ballot
  (`row_sum > 448`) — a lane's tile row-sum upper-bounds its max P, so this catches any tile
  that would overflow the fp8 P-pack **with no cross-lane reduction**. Only the rare
  overflow tile runs `_rollback_recover` (exact rescale reconstructed from P: `delta =
  1/max(P)`, `R*=delta`, `L=(L+sum)·delta`, `FA_max += log2(mx)/scale`). The unseeded /
  frozen-safe / rollback sub-paths emit identical `s_barrier` counts, so the two wave groups
  stay in lockstep even on different branches.

Net softmax cost on the common (no-mask, no-overflow) tile: **no `v_exp` transcendental, no
cross-lane rowmax (`permlane`), no O/L rescale** — just full-rate FMA/cvt + a local sum.

---

## 3. Side-by-side vs our CK-UA prefill

| Axis | CK-UA (ours) | ASM fp8 (theirs) | Gap? |
|---|---|---|---|
| Wave structure | 2 warp-groups **ping-pong asymmetric phases**: WG-A in SOFTMAX while WG-B in MATRIX, barrier-synced | 8 waves **symmetric compute** (all do QK→softmax→PV); only the *load* role is split (W0-3=K, W4-7=V) | **Yes** |
| Phase balance | SOFTMAX ~7-12% longer than MATRIX → barrier idle, ~31-33% overlap | skew + `s_setprio` overlaps MFMA(grp1) with VALU(grp2); no fat phase to stall on | **Yes** |
| `exp2` | `ck_tile::exp2` → HW `v_exp_f32` (**quarter-rate**) | Schraudolph affine `v_fma`+`v_cvt_u32` (**full-rate**) | **Yes (big)** |
| Per-tile rowmax | computed **every tile** (`block_tile_reduce` + `permlane32_swap`) | **frozen**, skipped on common tiles (rollback only on rare overflow) | **Yes (big)** |
| O/L rescale | `CONDITIONAL_RESCALE=1` already skips it between commits (τ=8) ✓ | frozen → delta≡1, rescale dropped | ~par |
| `s_setprio` handoff | dynamic setprio (W0-3/W4-7) already present ✓ | `s_setprio(0/1)` at gemm boundaries | ~par |
| Load hiding | prefetch (coarser) + the K-LDS-under-PV placement we just studied | per-MFMA `mem_load`/`lds_read` interleave w/ running `lgkmcnt`, double-buffered, dedicated loader waves | **Yes (medium)** |
| MFMA | fp8 MFMA | `v_mfma_f32_32x32x64_f8f6f4` (K=64) | ~par |
| Q tile | smaller | `kTileQ=256` (amortizes K/V streaming over more rows) | minor |

So two things we already converged on independently — **dynamic setprio** and a
**conditional/skipped rescale** — are confirmed as the right direction by the SOTA kernel.
The remaining, unclaimed wins are the three in the TL;DR.

## 3.5 Isolating the levers (built the ASM kernel with toggles off)

Rebuilt from `/tmp/fp8_{exact,exp2only}.py` (copies with `_USE_EXP2_APPROX` /
`_USE_ROLLBACK` flipped), deployed each `.co`, standalone CUDA-event **min** at sq75600/h5:

| ASM variant | TFLOPS (min) | vs CK-UA | marginal |
|---|---|---|---|
| `exact` — exp2 OFF, rollback OFF (**pipeline only**) | **2055** | **1.75×** | structural baseline |
| `exp2only` — exp2 ON, rollback OFF | 2225 | 1.90× | **+8.3%** (Schraudolph exp2) |
| `full` — exp2 ON, rollback ON | 2335 | 1.99× | **+5.0%** (max-freeze/rollback) |

Reading: the **exact** ASM kernel — same numerics class as ours, in fact *weaker* than ours
on the softmax (it does a full per-tile online rescale, which our `CONDITIONAL_RESCALE`
already skips) — is still **1.75×** faster. That 1.75× is the *pipeline* (symmetric
compute, loader split, per-MFMA load interleave, skew/`setprio`, `kTileQ=256`, hand
scheduling). The two numeric tricks are a clean, separable **+14%** on top. This is why §4
leads with the structural work, not the exp.

---

## 4. What to copy into CK-UA prefill (prioritized by *measured* contribution)

The isolation in §3.5 says: the structural pipeline holds ~1.75×, exp2 ~+8%, rollback ~+5%.
So lead with structure, then take the two cheap numeric wins.

**P0 — Close the structural pipeline gap (memwait + barrier idle).** This is where the 1.75×
lives and it maps onto bottlenecks we already measured. Two concrete, separable sub-items:

  - **P0a — Finer per-MFMA load/compute interleave + double-buffered loader split.** The ASM
    kernel issues, *inside* each QK/PV MFMA loop, one `mem_load` (next tile HBM→LDS,
    double-buffered) or one `lds_read` (current tile LDS→VGPR) per MFMA, under a running
    `lgkmcnt` budget, with waves 0-3 owning K loads and 4-7 owning V. That fully hides the
    DRAM stream behind math — directly attacking the exposed `memwait` at this hq=hk=5
    (zero-KV-reuse) shape. Port the budgeted interleave into our `fa4_matrix`/prefetch path
    (this also subsumes the K-LDS-placement micro-study). **Highest expected payoff.**
  - **P0b — Reconsider the asymmetric SOFTMAX↔MATRIX ping-pong vs symmetric compute.** Our
    2-warp-group split idles whenever the phases are unequal (softmax is 7-12% longer → the
    barrier idle we see). The ASM kernel keeps **all 8 waves on one QK→softmax→PV path** and
    overlaps MFMA(grp1)↔VALU(grp2) via phase-skew + `s_setprio`, specializing only the
    *load* role. Highest ceiling but most invasive — prototype after P0a/P1 and re-measure
    phase balance first.

**P1 — Schraudolph full-rate exp2 in the softmax (+~8%, cheap & local).** Replace
`ck_tile::exp2(scale_s·x)` (lowers to quarter-rate HW `v_exp_f32`) with the affine-bits
trick: precompute `AC = scale_s·log2(e)·C` and `D` to SGPRs, `P = u32_as_f32(round(S·AC +
D))` (full-rate `v_fma`+`v_cvt_u32`). Gate behind `UA_EXP2_APPROX` with an exact-SKU
fallback (keep exact on masked tiles, as the ASM kernel does). Validate cosine ≥ ~0.998.
  - Touch point: the `ck_tile::exp2(...)` sites near L2063-2067 and the `fmha_alu1`/exp step
    in `unified_attention_pipeline.hpp`.

**P2 — Freeze the rowmax, not just the rescale (+~5%).** We already skip the rescale
(`CONDITIONAL_RESCALE`) but still pay the per-tile `block_tile_reduce(max)` +
`permlane32_swap` cross-lane reduction (L1577) every tile. Adopt the ASM scheme: seed the
max on tile 0, then **freeze** it and skip the rowmax reduction; detect fp8-pack overflow
with a per-lane row-sum + wave-OR ballot against 448 (e4m3); only the rare flagged tile
reconstructs the exact rescale from P. Extension of `CONDITIONAL_RESCALE`, not a rewrite.

**P3 — Larger Q tile (`kTileQ=256`).** Amortizes each K/V tile's load over more query rows.
Mostly relevant once memwait/softmax are no longer the limiter.

### Suggested sequence
1. P1 exp2-approx (flagged) first — it's the cheapest, lowest-risk win and shrinks the
   SOFTMAX phase; re-run the ATT overlay at `sq75600 h5` to see phase balance shift.
2. P0a budgeted load interleave — target the exposed `memwait`; this is the biggest single
   structural lever that does *not* require rearchitecting the phase model.
3. P2 rowmax-freeze — further shrinks softmax.
4. Re-measure. If barrier idle still gates after 1-3, prototype P0b (symmetric compute).

---

## 5. Caveats / env notes

- Numbers were taken with the colleague using some GPUs; pick an idle device
  (`rocm-smi --showuse`) and prefer the standalone CUDA-event **min** (least throttled).
  The two ASM numbers (2107 `@perftest` median / 2306 standalone min) bracket the truth; a
  fully idle box should land near the colleague's ~2500.
- Both kernels here are **dense non-causal** at identical FLOP, so TFLOPS is directly
  comparable. The ASM kernel is the perf target for the CK-UA prefill at hd128.
- The ASM kernel's wins are partly **numeric approximations** (Schraudolph exp2, max-freeze).
  They are validated to cosine ~0.998 here and (per the source) on Wan2.2 e2e with rollback,
  but any copy into CK-UA must keep an exact SKU behind a flag and re-check accuracy gates.
- `_USE_ROLLBACK=False` makes the ASM kernel byte-stable (exact online softmax) — useful as
  an apples-to-apples "what does the pipeline structure alone buy" baseline if we want to
  separate the structural win (§4 P1) from the numeric wins (§4 P0).
