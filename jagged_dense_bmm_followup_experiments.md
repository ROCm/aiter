# jdbba follow-up experiments — W/C autotune, deployment skew, C-shuffle LDS, M register-tiling

Post-XCD-remap optimization experiments on AMD CDNA4 (MI355X / gfx950), each run
on a separate copy of the production kernel
(`aiter/ops/flydsl/kernels/jagged_dense_bmm_gen.py`, which carries the XCD remap)
so the production file stayed untouched. All are **negative for folding**, but
experiment 1 surfaced two deployment-relevant findings worth acting on, and
experiment 4 (M register-tiling) is documented with its current-hardware repro.

Measurement: rocprofv3 `--kernel-trace`, `read_us2.py <dir> jdbba p10` (10th
percentile device time; never median — see the main report's correction note).
Baseline = production kernel with the current remap defaults (`XCD_C_SMALL_K=32`,
`XCD_C_LARGE_K=60`, `XCD_W=8`). Headline shapes `(B, D=reduction K, Kout=output
N)`, uniform `M_i=7680` unless noted.

Baseline p10 (µs): B120_D256 ≈ 276, B120_D512 ≈ 787, B1024_D256 ≈ 2189,
B1024_D512 ≈ 6390. Measured noise floor ≈ 0.5% (default re-run 3×).

---

## Experiment 1 — finer (W,C) autotune + deployment-skew behavior

Clone: `aiter/ops/flydsl/kernels/jagged_dense_bmm_wc.py` (byte-identical to gen;
W/C swept purely via the `xcd_c`/`xcd_w` override args).

### 1A. Finer (W,C) sweep (uniform M_i=7680)

Swept `C ∈ {8,16,24,32,48,64,96,160,240}` at `W=8`, then `W ∈ {4,8,16}` at the
top-2 C per shape.

| shape | default (W8) | best found | best (W,C) | Δ vs default |
|---|---|---|---|---|
| B120_D256 | 277.1 (C32) | 273.3 | W16/C64 | −1.4% |
| B1024_D256 | ~2189 (C32) | 2162–2169 | W16/C64 | −1.2…−1.7% |
| B120_D512 | ~775 (C60) | 766.8–771 | W8/C240 or W16/C240 | −0.5…−1% |
| B1024_D512 | ~6390 (C60) | 6355–6372 | W16/C64 | −0.3…−0.6% |

**Verdict: finer tuning does not beat the defaults beyond the ~0.5% noise
floor.** The D512 C-curve is a flat plateau for `C≥32`; `W=8` is flat-optimal.
The only reproducible (3×) signal is **`W=16` on the D256 shapes (~1.2–1.7% on
B1024_D256)** — real but marginal.

**Recommendation: keep current defaults.** Optionally bump `XCD_W→16` for K≤256
if the ~1.5% D256 gain is worth a config change; not folded here because it's
within an order of magnitude of noise.

### 1B. Deployment-skew correctness + perf (the important finding)

Production deployment is varlen: `M_i ~ 0.95·Uniform(1, max_seq_len)` per group,
**not** the uniform `M_i=7680` the remap was tuned on. The remap's
`off_b = row // BM_TILES` decode assumes every group has the same number of
M-tiles (`BM_TILES = ceil(max_seq_len/BLOCK_M)`); the kernel launches that
uniform grid and early-exits (`if start_m < M_b`) on tiles past a group's real
length.

**Correctness — the remap stays correct under skew.** `_xcd_remap` is a provable
bijection over the full uniform tile grid; with `C=1,W=1` it is the exact
identity. Kernel cos vs torch eager (skewed seq_offsets):

| case | shape | L | cos |
|---|---|---|---|
| skewed M_i | B120_D512 | 433k | 0.999999 PASS |
| skewed M_i | B1024_D256 | 3.73M | 0.999999 PASS |
| empty (M_i=0) + single-row (M_i=1) | B1024_D256 | — | 0.999999 PASS |
| empty + single-row | B120_D256 | — | 0.999999 PASS |

**Perf — the remap does NOT help under skew; it slightly hurts.** B1024_D512
skewed, p10 µs, 3 seeds:

| C | seed1234 | seed7 | seed99 |
|---|---|---|---|
| **1 (no remap / identity)** | **3214.9** | **3223.2** | **3280.3** |
| 60 (default) | 3246.3 | 3291.2 | 3356.3 |
| 32 | 3277.1 | – | – |
| 96 | 3325.4 | – | – |

No-remap (C=1) is fastest, beating the C=60 default by ~2.0–2.3% consistently.
**Mechanism:** with the early-exit, clustering a group's M-tiles onto one XCD
also clusters its *skipped* (no-op) tiles, causing XCD load imbalance;
round-robin (identity) spreads the real work evenly. The remap's cross-block L2
reuse win (measured on the uniform grid) is outweighed by this imbalance once
group lengths vary.

**Recommendation (deployment-critical):** the remap was tuned and measured only
on the uniform grid. For the **skewed production distribution, disable the remap**
(`xcd_c=1`) — or gate the default on whether `M_i` is uniform. The uniform
headline numbers keep the remap; real varlen traffic should not. This is the
single most important outcome of these experiments.

### Pre-existing base-kernel bug found (not a remap bug)

B120_D512 with the combined pattern `M_i=[0, 1, 7680, …]` (empty + single-row +
full-envelope group together) GPU-faults (`0x..e00000`). It **reproduces
identically with the remap disabled** (`xcd_c=1`), so it is a pre-existing bug in
`jagged_dense_bmm_gen`, independent of XCD. It does **not** reproduce at B=4 with
the same pattern, nor at D256, nor with empty+single absent a full group — it
needs the large B=120 grid + a full-envelope (60-M-tile) group at N=512. Likely
the A/C buffer-descriptor bounding (`make_bounded_buffer_tensor` on C, or A_g
`max_size=True`) interacting with a full-envelope group at N=512; worth checking
the i64 C-bound and the last-tile masked store. **Not fixed here — flagged for
the base-kernel owner.**

---

## Experiment 3 — shrink the C-shuffle epilogue LDS to lift occupancy

Hypothesis: the kernel is occupancy-resource-limited (32 KB LDS/block → 2.8
waves/SIMD achieved vs a 5-wave ceiling). Shuffle the epilogue C tile in
N-strips so the row-major shared C tile shrinks from `BLOCK_M·BLOCK_N·2` (32 KB)
to `BLOCK_M·(BLOCK_N/S)·2`, hopefully lifting occupancy and hiding more of the
23% MemUnitStalled.

Clone: `aiter/ops/flydsl/kernels/jagged_dense_bmm_cstrip.py` — adds `EPI_STRIPS`
(S) through the factory and public entry; the epilogue stages a
`(BLOCK_M, BLOCK_N//S)` sC and loops S times, slicing the retiled bf16 fragment
on its trailing N-repeat axis (write→barrier→re-read N-contiguous→128b
store→barrier).

**Correctness:** S=2 passes all 4 shapes at Mi=512 and 7680, cos=0.999999.

**S=4 is not expressible:** the MFMA-16×16 C fragment's N-repeat granularity is
64 columns (retile trailing axis = 2 for BLOCK_N=128). 32-wide strips need
sub-slicing inside one contiguous fragment unit (`vector.extract` position 64 out
of range). Only S=2 (64-wide strips) is valid.

**The decisive finding — LDS/block never drops (the premise was false):**

| shape | A-staging LDS (`BLOCK_M·BLOCK_K·STAGES_A·2`) | epi full | epi S=2 | launched `max()` baseline → S=2 |
|---|---|---|---|---|
| D256 (BLOCK_K=128) | **64 KB** | 32 KB | 16 KB | 64 KB → **64 KB** |
| D512 (BLOCK_K=64) | **32 KB** | 32 KB | 16 KB | 32 KB → **32 KB** |

`epi_smem_bytes = max(smem_bytes, epi_tile)`, and the **A double-buffer staging
buffer is ≥ the epilogue tile** at default tiling. On D256 A-staging is 64 KB
(BLOCK_K=128) and swamps the epilogue; on D512 A-staging is 32 KB == the full
epilogue, so halving the epilogue to 16 KB leaves `max()` pinned at 32 KB.
**Launched LDS/block is identical on every shape (Δ 0 KB)** → occupancy cannot
rise (VGPR=84 and LDS both unchanged).

**Device time p10 (µs), Mi=7680, matched (W=8, C=32/60):**

| shape | baseline | S=2 | Δ |
|---|---|---|---|
| B120_D256 | 276.4 | 275.6 | +0.3% (noise) |
| B120_D512 | 787.2 | 792.6 | −0.7% |
| B1024_D256 | 2198.9 | 2186.7 | +0.6% (noise) |
| B1024_D512 | 6462.6 | 6454.7 | +0.1% (noise) |

**Verdict: not worth folding.** Shrinking the epilogue LDS doesn't help on any
shape because A-staging, not the epilogue, sets the LDS `max()`. The extra
barrier is at best neutral (slightly negative on D512). The C-strip path is
correct and reusable, but only buys peak-LDS headroom *after* the binding
resource is reduced.

**Reframed takeaway for future occupancy work:** to lift occupancy you must
shrink the *binding* resource — **A-staging LDS** (`STAGES_A` or `BLOCK_K`) and/or
**VGPR (84)** — not the epilogue. Any future occupancy lever should start there.

---

## Experiment 4 — M register-tiling (raise B reuse without growing LDS)

**Hardware/software for this run (the "current hardware"):** AMD Instinct
**MI355X** (gfx950 / CDNA4), 256 CU, 160 KB LDS/block, wave64, on host
`smci355-ccs-aus-m01-21` (EPYC 9575F). ROCm **7.13.0**, rocprofv3 **1.3.0**.
**GPU 5 only** (`HIP_VISIBLE_DEVICES=5`); all runs inside docker container
`anguyenh-dev`. Headline shapes `(B, D=reduction K, Kout=output N)`, uniform
`M_i=7680`.

Clone: `aiter/ops/flydsl/kernels/jagged_dense_bmm_mreg.py` (clone of
`jagged_dense_bmm_gen.py`, keeps the XCD remap unchanged).

### Hypothesis

B's per-group re-read count = number of M-tile blocks = `ceil(M_i/BLOCK_M)`. Make
each block compute `MM` M-subtiles that **share one B fragment** in the K-loop, so
one B-tile load amortizes over `MM` MFMAs → fewer Dense[b] re-fetches → less HBM
traffic on the bottleneck tensor. Avoid the naive `BLOCK_M=256` LDS blowup +
occupancy cliff by keeping `BLOCK_M=128` and register-blocking only the
MFMA/accumulators.

### Decomposition implemented

Each block covers `MM·BLOCK_M` rows (MM=2 → 256). In the inner K-loop the dense
K-tile is prefetched into **one** `mma_frag_B` and re-read once per k-subtile,
then `MM` MFMAs are issued against it — one per A-subtile, each with its own A
fragment and fp32 C accumulator (`mma_frag_A[j]`, `mma_frag_C[j]`). One B load,
MM MMAs.

- **LDS growth:** the swizzled A staging gains an outermost `MM` dim
  (`32 KB→64 KB` for D512); the O4 C-shuffle tile is **reused sequentially**
  across the MM subtile epilogues, so it does **not** grow (epi smem stays
  `max(64 KB, 32 KB)=64 KB` < 160 KB).
- **Grid:** `bm = ceil(M_i/(MM·BLOCK_M))` (M grid shrinks by MMx), fed unchanged
  into the XCD remap as `BM_TILES`.
- **Tail:** block guard `super_start_m < M_b`; per-subtile physical tile =
  `super_m_idx·MM + j`, kept uniform via `readfirstlane`. Fully-OOB subtiles are
  masked by the existing bounded-C buffer (rows ≥ M_b OOB-dropped) and the
  `max_size` A buffer (OOB reads return 0). Verified on a ragged `M_i=7000` case.

### Correctness (cos vs torch eager, all > 0.999)

| shape (B,D,Kout) | M_i=512 (C∈{16,32,60,120}) | M_i=7680 | M_i=7000 ragged |
|---|---|---|---|
| 120,256,256 | 1.000000 | 1.000000 | — |
| 120,512,512 | 1.000000 | 1.000000 | 1.000000 |
| 1024,256,256 | 1.000000 | 1.000000 | — |
| 1024,512,512 | 1.000000 | 1.000000 | 1.000000 |

### Device time p10 (µs), M_i=7680, interleaved same-session GPU 5

| shape | baseline+remap | mreg MM=2 | speedup |
|---|---|---|---|
| B120_D256 | 278.8 | 368.4 | **0.76× (regress)** |
| B120_D512 | 784.0 | 731.2 | **1.07×** |
| B1024_D256 | 2193.9 | 2801.3 | **0.78× (regress)** |
| B1024_D512 | 6469.1 | 6110.2 | **1.06×** |

Distributions clean (n=35 = 5 warmup + 30; B1024_D512 min/p10/median =
6069/6110/6150 baseline 6456/6469/6511).

**MM sweep (D512 only — D256 with BLOCK_K=128 exceeds 160 KB LDS at MM≥3):**
`MM=4` collapses occupancy (128 KB A-staging → ~1 block/CU): B1024_D512
6110 → **18187 µs (3× slower)**, B120_D512 731 → 2146 µs. `MM=3` GPU-faults
(memory access fault — suspected non-power-of-2 tiling constraint; not pursued).
**MM=2 is the sweet spot.**

### PMC (B1024_D512, whole-run sums, 2 counters/pass)

| counter | baseline | mreg MM=2 | Δ |
|---|---|---|---|
| L2 hit rate (TCC_HIT/(HIT+MISS)) | 73.2% | 65.5% | −7.7 pp |
| total L2 accesses (HIT+MISS) | 2.24e10 | 1.79e10 | **−20%** |
| TCC_EA0_RDREQ_DRAM (HBM reads) | 3.79e9 | 3.96e9 | **+4.5%** |
| TCC_EA0_WRREQ_DRAM (HBM writes) | 4.41e9 | 4.41e9 | +0.0% |

### Verdict: neutral / shape-dependent — **do NOT fold** (high confidence)

The core hypothesis (cut B's HBM traffic) **did not hold**: the XCD remap already
serves B re-reads from L2, so they weren't reaching DRAM. M register-tiling cut
total **L2** accesses ~20% (fewer B re-reads at the L2 level), but **DRAM reads
went up +4.5%**, not down. The modest ~6% D512 win therefore comes from *fewer
blocks* (less launch/epilogue overhead + one B fragment feeding 2 MFMAs improving
MFMA issue), not from bandwidth. D256 regresses 22–28% because `BLOCK_K=128`
doubles A-staging to 128 KB/block where the LDS/occupancy cost dominates with no
compute payoff on the shallow 2-iter K-loop. Not worth the LDS/occupancy risk.
The XCD remap remains the effective B-reuse lever.

---

## Experiment 5 — CDNA4 16x16x32 bf16 MFMA atom — **FOLDED IN** (high confidence)

### Hypothesis
The kernel uses the `MFMA(16,16,16,bf16)` atom; gfx950/CDNA4 has
`v_mfma_f32_16x16x32_bf16` (twice the K per instruction). Issue overhead is
visible (VALU/MFMA ratio 1.33). The wider-K atom halves MFMA issue and the
per-K-step staging that surrounds it.

### Change
`MFMA(16,16,16)` → `MFMA(16,16,32)`; K-permute tile `(4,4,2),(1,8,4)` →
`(8,4),(1,8)` (one 32-wide atom per perm group instead of 2 K=16 atoms); inner
loop fragment K index flattens from `(None, block_k_iter)` to `block_k_iter`.
Trip count `BLOCK_K // 32`, s2r copy, swizzle, remap, epilogue all unchanged.
Mirrors `preshuffle_gemm_v2`'s `use_mfma_k32` path.

### Correctness
All 4 headline shapes × full C grid at Mi=512 and Mi=7680: **cos = 1.000000**.

### Device time p10 (µs), Mi=7680, interleaved same-session GPU 4
| shape | baseline (remap) | mfma32 | speedup |
|---|---|---|---|
| B120_D256 | 273.8 | 263.3 | 1.04× |
| B120_D512 | 782.2 | 728.7 | 1.07× |
| B1024_D256 | 2174.2 | 2085.7 | 1.04× |
| B1024_D512 | 6387.8 | 5961.1 | 1.07× |

### PMC (B1024_D512)
MFMA insts 17.62B → 8.81B (**exactly 0.500×**); VALU insts −38%; VGPR 166→160.
Win is fewer issued instructions, not higher occupancy (no occupancy step crossed
on CDNA4).

### Verdict: **FOLD IN** — gated on gfx95* (auto-detected via `_use_mfma_k32`)
Consistent 4–7% across all shapes, bit-exact, larger on compute-heavier D512.
gfx942/MI300 lacks the 32-K atom → falls back to 16x16x16. Folded into
`jagged_dense_bmm_gen.py` with the `use_mfma_k32` kwarg (default auto). Stacks
independently with the XCD remap.

---

## Experiment 6 — persistent problem-visitor scheduler — **promising, not shippable as-is**

### Hypothesis
The static grid `(ceil(max_seq_len/BLOCK_M)·N_BLOCKS, 1, n_groups)` over-provisions
to the longest possible sequence per group and early-exits the rest. On skewed
(varlen) jagged data most launched blocks early-exit — pure overhead. A persistent
grid pulling only occupied tiles from a work-list (CUTLASS grouped-GEMM problem
visitor / MoE active-tile dispatch) eliminates that waste.

### Design (A: host-precomputed work-list)
Public entry copies `seq_offsets.cpu()`, builds group-major
`WORK_ITEMS[num_tiles,3]=(off_b, block_m_idx, block_n_idx)` over occupied tiles
only, launches `min(num_tiles, 512)` persistent blocks each looping
`for wi in range(blk_id, NUM_TILES, grid_dim)`. Group-major order keeps a group's
Dense[b] L2-resident. `tiled_mma`/copies MUST be built inside the loop (building
outside auto-promotes them to scf.for iter_args → breaks gemm lowering);
`SmemPtr._view_cache = None` reset per iteration.

### Correctness
UNIFORM Mi=7680 and SKEWED (power-law MSL·u⁴, ~27% empty, mean Mi≈1200): **cos =
1.000000** all 4 shapes.

### Device time p10 (µs)
| shape | UNIF base | UNIF persist | SKEW base | SKEW persist (kernel) | skew speedup |
|---|---|---|---|---|---|
| B120_D256 | 277 | 283 | 69 | 55 | 1.25× |
| B120_D512 | 788 | 949 | 327 | 174 | 1.88× |
| B1024_D256 | 2228 | 2314 | 527 | 387 | 1.36× |
| B1024_D512 | 6481 | 8020 | 2421 | 1280 | 1.89× |

### Verdict: high-value on skew, **two blockers before shipping**
1. The host `seq_offsets.cpu()` sync erases the win end-to-end (wall-clock
   628µs/8796µs) → **needs an on-device prefix kernel** to build the work-list.
2. Regresses ~20% on uniform D512 (loses the static grid's XCD-remap L2 reuse) →
   must be **skew-gated** (uniform → `_gen`, high-skew → persistent).
Right architecture for production jagged data; needs a second build pass.

---

## Experiment 7 — DirectToLDS for A — **regression, do NOT fold** (high confidence)

### Hypothesis
A is staged global→VGPR→LDS→VGPR→MFMA. `buffer_load_dwordx4 ... lds` writes
straight to LDS, cutting the VGPR round-trip (~32 VGPR/iter, 32 ds_write/iter) →
maybe enough occupancy for a memory-bound kernel. (Prior KB: DTLA alone regressed
8% on a compute-bound skinny-decode GEMM — speculative probe.)

### Change & finding
`fx.rocdl.BufferCopyLDS128b()` for the A load, `copy_frag_A` round-trip removed,
`s_waitcnt(0)` before the barrier preceding s2r. **Key correctness finding:**
`buffer_load_lds` writes contiguously and does NOT honor the XOR swizzle — with
`SwizzleType(3,3,3)` cos≈0.13; switching `sA` to a plain ordered layout gives
cos=1.0 (the swizzle only mattered for the deleted ds_write path).

### Device time p10 (µs), Mi=7680
| shape | baseline | dtla | speedup |
|---|---|---|---|
| B120_D256 | 278.9 | 348.0 | 0.80× |
| B120_D512 | 785.5 | 807.6 | 0.97× |
| B1024_D256 | 2205.7 | 2689.8 | 0.82× |
| B1024_D512 | 6471.9 | 6666.6 | 0.97× |

### Verdict: **regression (0.80–0.97×), do NOT fold**
ISA confirms the mechanism worked (A ds_write_b128 24→0, buffer_load…lds 0→16,
VGPR 194→168) — but same failure mode as the KB case: the immediate
`s_waitcnt vmcnt(0)` surrenders DTLA's latency-hiding, and losing the LDS write
swizzle adds s2r bank conflicts. Only worth revisiting paired with prefetch
(deferred vmcnt wait) + a swizzle-free / `ds_read_tr` transpose-read.

---

## Overall decision

**Folded into production (`jagged_dense_bmm_gen.py`): XCD remap + MFMA 16x16x32.**
Stacked vs the pre-remap baseline, B1024_D512 went 6728 → 6377 (remap) → 5951µs
(remap+mfma32), ≈1.13× total. Net guidance:

1. **Keep the remap defaults** (W=8, C=32/60) for the uniform headline shapes.
2. **MFMA 16x16x32 is on by default for gfx95*** (auto-detected); gfx942 falls
   back to 16x16x16. Both levers are independent and stack.
3. **Disable the remap for skewed/varlen deployment** (`xcd_c=1`) — it loses
   ~2% there due to early-exit XCD load imbalance. Consider gating the default on
   M_i uniformity.
4. **Drop the epilogue-LDS lever** (Exp 3); if occupancy is revisited, target
   A-staging LDS / VGPR instead.
5. **Drop M register-tiling** (Exp 4): does not cut HBM traffic (DRAM reads rise);
   the small D512 win is a block-count/MFMA-issue effect, D256 regresses.
6. **Persistent scheduler (Exp 6) is the path for skewed production data**
   (1.25–1.9×) but needs an on-device prefix kernel + skew gating first.
7. **Drop DirectToLDS-for-A (Exp 7)**: regresses standalone; revisit only with
   prefetch + swizzle-free transpose-read.
8. **Fix the base-kernel fault** at B120_D512 with empty+single-row+full-envelope
   groups (pre-existing, not XCD-related).

## Files (production kernel untouched)

Experiment 1: `aiter/ops/flydsl/kernels/jagged_dense_bmm_wc.py`,
`op_tests/flydsl_tests/{bench_headline_worker_wc.py,bench_skew_worker_wc.py,skew_correctness_wc.py,sweep_wc.sh}`

Experiment 3: `aiter/ops/flydsl/kernels/jagged_dense_bmm_cstrip.py`,
`op_tests/flydsl_tests/{test_cstrip_correct.py,bench_headline_worker_cstrip.py,bench_headline_worker_gen.py,run_cstrip_bench.sh,run_gen_bench.sh}`

Experiment 4: `aiter/ops/flydsl/kernels/jagged_dense_bmm_mreg.py`,
`op_tests/flydsl_tests/{test_mreg_correct_allshapes.py,bench_headline_worker_mreg.py}`

Experiment 5 (FOLDED IN): clone `aiter/ops/flydsl/kernels/jagged_dense_bmm_mfma32.py`,
`op_tests/flydsl_tests/{test_mfma32_correct.py,bench_headline_worker_mfma32.py}`
(the change itself lives in `jagged_dense_bmm_gen.py` behind `use_mfma_k32`).

Experiment 6: `aiter/ops/flydsl/kernels/jagged_dense_bmm_persist.py`,
`op_tests/flydsl_tests/{test_persist_correct.py,bench_headline_worker_persist.py,bench_skew_worker_persist.py,run_persist_bench.sh}`

Experiment 7: `aiter/ops/flydsl/kernels/jagged_dense_bmm_dtla.py`,
`op_tests/flydsl_tests/{test_dtla_correct.py,bench_headline_worker_dtla.py,run_dtla_bench.sh}`

## Reproduce

```bash
# EXP-1A finer (W,C) sweep, one shape:
docker exec -e HIP_VISIBLE_DEVICES=4 -e PYTHONPATH=/home/anguyenh/aiter:/home/anguyenh/generative-recommenders \
  anguyenh-dev bash -lc 'cd /home/anguyenh/aiter/op_tests/flydsl_tests && \
  bash sweep_wc.sh 1024 256 256 7680 "4 8 16" "8 16 24 32 48 64 96 160 240"'

# EXP-1B skew correctness (rand + edge):
docker exec -e HIP_VISIBLE_DEVICES=4 -e PYTHONPATH=/home/anguyenh/aiter:/home/anguyenh/generative-recommenders \
  anguyenh-dev bash -lc 'cd /home/anguyenh/aiter/op_tests/flydsl_tests && \
  python skew_correctness_wc.py 1024 256 256 7680 8 32 1234 rand'   # ... and "edge"

# EXP-1B perf under skew (C=1 = no-remap control):
docker exec -e HIP_VISIBLE_DEVICES=4 -e PYTHONPATH=/home/anguyenh/aiter:/home/anguyenh/generative-recommenders \
  anguyenh-dev bash -lc 'cd /home/anguyenh/aiter/op_tests/flydsl_tests && rm -rf /tmp/o && \
  rocprofv3 --kernel-trace -d /tmp/o -- python bench_skew_worker_wc.py 1024 512 512 7680 8 1 1234 && \
  python read_us2.py /tmp/o jdbba p10'

# EXP-3 correctness + bench (S=2):
docker exec -e HIP_VISIBLE_DEVICES=5 -e PYTHONPATH=/home/anguyenh/aiter:/home/anguyenh/generative-recommenders \
  anguyenh-dev bash -lc 'cd /home/anguyenh/aiter && \
  for s in "1024 512 512" "120 256 256" "120 512 512" "1024 256 256"; do \
    python3 op_tests/flydsl_tests/test_cstrip_correct.py $s 7680 2; done && \
  bash op_tests/flydsl_tests/run_gen_bench.sh && bash op_tests/flydsl_tests/run_cstrip_bench.sh 2'
```

### EXP-4 — M register-tiling (MI355X / gfx950, ROCm 7.13, rocprofv3 1.3, GPU 5)

All commands run **inside** container `anguyenh-dev` on **GPU 5**. Disable the JIT
disk cache while iterating so you don't pick up stale compiled artifacts.

```bash
# 4.1 Correctness — all 4 headline shapes (Mi=512 over a C grid) + Mi=7680
#     spot-check + ragged Mi=7000 tail. cos must be > 0.999. Arg = MM (default 2).
docker exec anguyenh-dev bash -lc 'cd /home/anguyenh/aiter && \
  FLYDSL_RUNTIME_ENABLE_CACHE=0 HIP_VISIBLE_DEVICES=5 \
  python op_tests/flydsl_tests/test_mreg_correct_allshapes.py 2'

# 4.2 Device-time benchmark — baseline (jagged_dense_bmm_gen) vs mreg MM=2,
#     interleaved on the same GPU for a fair same-session control. The worker
#     does 5 warmup + 30 timed launches; parse p10 (NOT median).
docker exec anguyenh-dev bash -lc 'cd /home/anguyenh/aiter && \
  OUT=$(mktemp -d /home/anguyenh/aiter/op_tests/flydsl_tests/_mreg.XXXX); \
  for s in "120 256 256" "120 512 512" "1024 256 256" "1024 512 512"; do \
    tag=$(echo $s | tr " " _); \
    HIP_VISIBLE_DEVICES=5 rocprofv3 --kernel-trace -d $OUT/base_$tag -- \
      python op_tests/flydsl_tests/bench_headline_worker.py flydsl $s 7680 >/dev/null 2>&1; \
    MREG_MM=2 HIP_VISIBLE_DEVICES=5 rocprofv3 --kernel-trace -d $OUT/mreg2_$tag -- \
      python op_tests/flydsl_tests/bench_headline_worker_mreg.py flydsl $s 7680 >/dev/null 2>&1; \
    bp=$(python op_tests/flydsl_tests/read_us2.py $OUT/base_$tag jdbba p10); \
    mp=$(python op_tests/flydsl_tests/read_us2.py $OUT/mreg2_$tag jdbba p10); \
    echo "SHAPE $tag  baseline=$bp  mreg2=$mp"; \
  done'
# rocprof DBs are written as root inside the container; remove via the container:
#   docker exec anguyenh-dev bash -lc "rm -rf /home/anguyenh/aiter/op_tests/flydsl_tests/_mreg.*"

# 4.3 PMC — confirm where the traffic went (B1024_D512). Use ONLY 2 counters per
#     pass: rocprofv3 --pmc with >2-3 counters core-dumps intermittently on this box.
#     L2 hit = TCC_HIT/(TCC_HIT+TCC_MISS); HBM reads = TCC_EA0_RDREQ_DRAM.
docker exec anguyenh-dev bash -lc 'cd /home/anguyenh/aiter && \
  OUT=$(mktemp -d /home/anguyenh/aiter/op_tests/flydsl_tests/_pmc.XXXX); \
  for cfg in base:bench_headline_worker.py mreg:bench_headline_worker_mreg.py; do \
    tag=${cfg%%:*}; w=${cfg##*:}; \
    MREG_MM=2 HIP_VISIBLE_DEVICES=5 rocprofv3 --pmc TCC_HIT TCC_MISS \
      -d $OUT/${tag}_hit -- python op_tests/flydsl_tests/$w flydsl 1024 512 512 7680 >/dev/null 2>&1; \
    MREG_MM=2 HIP_VISIBLE_DEVICES=5 rocprofv3 --pmc TCC_EA0_RDREQ_DRAM TCC_EA0_WRREQ_DRAM \
      -d $OUT/${tag}_dram -- python op_tests/flydsl_tests/$w flydsl 1024 512 512 7680 >/dev/null 2>&1; \
  done; \
  echo "OUT=$OUT (parse the *.db under each dir: SUM(rocpd_pmc_event.value) JOIN rocpd_info_pmc on pmc_id GROUP BY name)"'
```

To sweep MM, pass it as the test arg (`...allshapes.py 4`) and via the env var
(`MREG_MM=4`) for the bench worker. Note MM≥3 needs `D=512` (BLOCK_K=64) to stay
under 160 KB LDS; MM=3 currently GPU-faults, MM=4 runs but is 3× slower.
