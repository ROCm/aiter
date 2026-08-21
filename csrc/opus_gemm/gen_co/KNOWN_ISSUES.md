# Known correctness issues in the pre-compiled (`.co`) a16w16 families

One fixed, two open and guarded. All produce **wrong results, not faults**, so only a numeric
check sees them. Recorded so the next session starts from the reproducers.

Applies to `a16w16_4wave_co` (reference, `..._4wave_compute_...cuh`) and
`a16w16_4wave_wl_co` (`..._4wave_wl_...cuh`) unless noted.

---

## 1. Narrow `B_N` raced at large shapes — CAUSED BY TWO BUGS, BOTH FOUND

**Status:** two independent bugs, both found and both **fixed in the pipelines**
(see "The two bugs"). With the pad off and the fixes in, all 204 variants are
clean at 2–3 WG/CU over 400 repeats. The 1-WG/CU pad is still applied on top:
the fixes cost 1–2%, but dropping the pad is a separate, shape-dependent
performance question that has not been decided.

Non-deterministic wrong results: same seed, same data, the wrong-element count
changed every run, including runs that came out clean.

```
kid 21182  64x64x128 w1x4 c4x1  @ 2048x2048x7168   (before the fix)
  rep0 nbad=25416   rep1 nbad=24602   rep2 nbad=33105
```

**What is established.** This pipeline is only correct at one workgroup per CU.
Any tile whose A/B LDS segments fit twice in the 320 KB budget (<= 160 KB) let a
second workgroup co-reside, and every such variant raced.

VGPR and LDS both scale with the tile, so the raw correlation cannot separate
them — but the population contains the control group that does:

| group | n | LDS admits 2 WG | VGPR admits 2 WG | occupancy | result |
|---|--:|---|---|---|---|
| A | 61 | yes (<= 160 KB) | yes (2x320..450 <= 1024) | 2 WG/CU | **all wrong** |
| B | 71 | no (> 160 KB) | yes | 1 WG/CU | all correct |
| C | 72 | no | no (2xVGPR > 1024) | 1 WG/CU | all correct |

A and B differ **only** in the LDS-driven occupancy — both would allow two waves
per SIMD on registers alone. A is entirely wrong and B entirely correct, so the
variable is workgroups-per-CU, not tile size or register pressure.

**It IS a synchronization bug, and the damage geometry says which edge.** An
earlier reading in this file said it was not, on the strength of a "maximal
sync" experiment recorded as not helping. Re-run against the current tree,
maximal sync (`s_wait_tensorcnt(0)` + `s_wait_dscnt(0)` + a full barrier at
every tile) gives 0/61 on both shapes.

What the wrong elements look like is what localises it. With the pad off, at
`2048x2048x7168`, `tol = 4` against a noise floor of 1.0, across four variants
and twelve failing runs, every single one:

* touches exactly **one tile**, i.e. one workgroup;
* spans **all** of that tile's M rows;
* is confined in N to `[B_N/2, B_N/2 + a few)` — a short **prefix of wave 3's
  half of B**, never wave 2's half, never A;
* carries an error of ~7.6 against a `|ref|` of ~54 at those positions, where
  one whole K step contributes ~60 — so roughly an eighth of one step, about 16
  of the 128 K elements in a row.

Read that back into the pipeline. Wave 2 and wave 3 load the two halves of B's
`B_N` rows; the tile walk takes the N sub-groups in order, so wave 3's half is
the **last** region read in a K step.

Fill order decides what that means, and it is measured, not assumed: a
standalone probe that samples LDS mid-transfer (`s_sleep` instead of
`s_wait_tensorcnt`, sweeping the delay) shows the engine filling **rows in
ascending order** — at one delay step, rows 0..21 of a 32-row tile had landed
and 22..31 had not. So a reader that is too early misses the **tail** of a
region, while a writer that is too early clobbers its **head**.

The damage is at the head, sub-row, in the last-read region. That is a
write-after-read: the step refills the very slot it is reading (slot g%P is
consumed by step g and reloaded at its end for step g+P), and the refill lands
while a `ds_read` of that slot is still in flight.

### The two bugs

**(a) Write-after-read on the ring.** The handshake in the last msb-tile has to
carry this WAR as well as the fill it was written for, and the split pair gave
it neither half:

* this wave's reads of `cur` are ISSUED by then but not RETIRED, and a barrier
  orders waves, not their in-flight LDS traffic — the same distinction the
  epilogue's `ds_writes` already call out;
* `s_barrier_signal(-1)` and `s_barrier_wait(-1)` sit `kBarrierAhead` WMMAs
  apart, so a wave that is ahead can signal for the NEXT step into this step's
  count.

The fix retires first (`s_wait_dscnt(0)`) and moves the signal down to the wait,
so arrive-and-wait is one barrier. Failure rate scales with `k_steps`, which is
what says it is per-K-step.

**(b) A "zero-extent" TDM zero-fills LDS.** Affects the 108 of 204 variants
whose peeled step actually fuses C staging (`kFusedMsb = kNSub - 1 > 0`); the
`kNSub == 1` variants stage everything in the epilogue, after the drain, and
were never exposed. The pipeline
over-issues `kSlots` transfers past the last K step and relies on them writing
nothing: *"past K the D#'s tensor_dim0 saturates to 0, so a step beyond the last
is a zero-extent DMA that only bumps tensorcnt"*, which is also what let the
epilogue stage C in a ring slot one of them is aimed at, with no barrier in
front. **Measured on gfx1250, that is false.** A standalone probe pre-fills LDS
with a sentinel and issues a transfer whose origin is at or past either extent:
every one of the 2048 dwords of the tile comes back **zeroed**, not untouched —
for `origin0` at the extent, past it, and for `origin1` likewise. So the trailing
transfer zero-fills the slot C was just staged into. The drain that was supposed
to cover this sat *after* the peeled step, i.e. after the staging; the fix moves
it before. Failure rate is flat in `k_steps` at fixed grid (7/12/18/8 bad runs at
`k_steps` 8/16/32/64), which is what says it is per-workgroup.

They are independent, and each fix only closes its own (61 variants, 200
repeats, pad off):

| | 2048x2048x7168 | 4096x4096x2048 | 4096x4096x1024 |
|---|---|---|---|
| neither fix | 11/61 | 15/61 | 8/61 |
| (a) alone | 1/61 | 10/61 | 3/61 |
| (b) alone | 12/61 | 7/61 | 1/61 |
| **both** | **0/61** | **0/61** | **0/61** |

All 204 variants with both fixes and the pad off: 0 wrong over 100 repeats at
`2048x2048x7168`, `4096x4096x1024`, `2048x2048x8192`, `512x512x512`,
`129x257x384`, and 0 over 400 repeats at `4096x4096x2048`. (One isolated event
appeared in an earlier 100-repeat pass at that shape and did not reproduce in
400, so "zero" here is a bound, not a proof.)

Both are latent at 1 WG/CU: the shipped padded build is clean over 204 variants
x 200 repeats at `4096x4096x1024` and `4096x4096x2048`. Without a co-resident
workgroup the trailing transfer lands long before the staging, and the ring's
refill lands long after the reads.

### What the fixes cost

Measure this box's A/B by **interleaving** the two builds and taking a median
per kid, not by timing one build and then the other: a sequential pass puts all
the clock and thermal drift on one side. Timing the same pair sequentially first
said the fixes were 5.2% *faster* at `2048x2048x7168`; interleaved, on a shape
whose trial-to-trial spread is 6%, they are 2% slower. Constant inputs
(`--init const`) also drop the data distribution out of it — absolute latency
moves a lot (511 vs 715 us at `8192^3`) but the A/B ratio does not, which is
what makes it a usable control.

Shipped build, before the fixes vs after, median of 5 interleaved trials over
the 61 variants:

| shape | before | after | |
|---|---|---|---|
| `2048x2048x7168` | 49.3 us | 50.3 us | +2.1% |
| `4096x4096x2048` | 56.5 us | 57.6 us | +1.8% |
| `4096x4096x1024` | 33.5 us | 34.5 us | +2.9% |
| `8192x8192x8192`, 16 `128x256x128` kids | 511.4 us | 516.7 us | +1.0% |

So about 2%, uniformly, for two real races.

### Should the pad go?

Separate question, and the answer is per kid rather than per shape. No pad +
fixes against the shipped padded build, same interleaved method:

| shape | mean | per-kid p10 / median / p90 |
|---|---|---|
| `2048x2048x7168` | +9.2% | -11.5% / +9.5% / +39.6% |
| `4096x4096x2048` | +5.8% | -15.3% / +4.3% / +40.9% |
| `4096x4096x1024` | **-15.6%** | -26.4% / -12.9% / +1.2% |

A spread from -26% to +41% within one shape is exactly what the kid table plus
the tuner exist for: add the 61 entries a second time carrying
`-DOPUS_CO_NO_1WG_PAD` and a `variant` suffix, and let tuning pick. Note that
`build_co.py`'s `_expected_lds` reads the no-pad flag from the command line
rather than from the entry, which would have to follow.

The contrast that pins it: waiting one short of what the slot needs
(`s_wait_tensorcnt(kSlots - 1)`), so the step genuinely reads a fill still in
flight, produces ~3.2 M wrong elements across 1020 of 1024 tiles, whole tiles,
both operands, every run — nothing like the failure being diagnosed. So it was
never the counted `s_wait_tensorcnt` failing to cover its slot.

### Dead ends, so they are not walked twice

* **The TDM descriptor's LDS base.** `make_tdm` is handed
  `reinterpret_cast<uintptr_t>(smem_a)`, a *workgroup-relative* offset, so if the
  engine did not add the workgroup's LDS base a co-resident workgroup would aim
  at its neighbour's LDS. It fits every symptom and it is wrong: the engine adds
  the base. `HW_REG_LDS_ALLOC` decodes as `base_KB` in [11:0] and `size_KB` in
  [31:12] (calibrated over 32/96/153/160/200 KB allocations, base + size landing
  on the 320 KB budget every time), and it places **half** the workgroups at a
  non-zero base — all of which read back their own transfer at their own base.
  Adding the base in software makes it catastrophically worse (61/61 variants,
  ~2.6 M of 4.19 M elements wrong), the double-add signature.
* **Anything that only adds delay.** At `2048x2048x16384`, a bare `s_sleep`,
  which orders nothing, reaches 0/61 at +5.0% while `s_wait_tensorcnt(0)` needs
  +15.6% for the same result — so the full drain was never a principled fix
  either, just a poorly-priced delay. If a candidate fix is on that same
  cost-versus-correctness curve, it is a window-widener, not a fix.
* **A fixed tolerance at large K.** bf16's own output rounding grows with
  `sqrt(K)`: the whole population reports max-abs-err 1.000 at `K = 8192`, 1.997
  at `K = 16384`, 2.001 at `K = 32768`, bit-identical across kids. So `tol = 2`
  reads as "everything fails" at `K = 32768` and sits on the cliff at
  `K = 16384`. Measure at `K <= 8192` or scale the tolerance, and check the
  population agrees before believing any single kid.

An earlier reading in this file said the failure was **not** a synchronization
bug, on the strength of a "maximal sync" experiment recorded as not helping.
Re-run against the current tree, maximal sync gives 0/61.

**Reproducing.** `build_co.py --device-flag=-DOPUS_CO_NO_1WG_PAD` drops the pad
and puts the 61 affected variants back at 2–3 WG/CU; point `OPUS_GEN_CO_DIR` at
that output tree to run them without rebuilding the module. With the fixes in
they stay clean; to see the original failure, revert the two edits described
above. Expect a handful of variants wrong on a handful of runs out of 100 —
rare enough that a single repeat proves nothing, and that 20 repeats can come
out clean by luck.

Two traps this hit:

* **Both families were affected** (reference 16/64, wl 44/139), so it predated
  the wave-layout work — a defect found in new code is not necessarily from it.
* **`build_co.py` mirrors the LDS formula host-side** and did not know about the
  pad, so its `group_segment_fixed_size == traits` check failed the build. That
  check is why the drift was caught instead of shipped; keep the two in step.

Earlier notes in this file blamed a `B_N > N` tile-wider-than-matrix bug and then
"cross-candidate interference in mp_tuner". Both were wrong. The tell was that
standalone replays of the kids the tuner flagged came out clean — which proves
nothing about a race, and should have pointed at non-determinism sooner.

---

## 2. `B_K > 128` together with more than one msb group

**Status:** open, **guarded** by a `static_assert` in
`opus_gemm_pipeline_a16w16_4wave_wl_gfx1250.cuh`, so it cannot silently ship.
Only the `wl` family can express it.

```
kHalvesPerSlot > 2  &&  kNumNSub > 1   ->  ~2/3 of output elements wrong
```

Deterministic; reproduces on 10/10 sampled configurations from a clean build.
Either factor alone is fine: `128x64x256` (kHalves 4, kNumNSub 1) and
`96x96x128` (kHalves 2, kNumNSub 3) both verify.

Ruled out:

* **not a ds-count race** — forcing `s_wait_dscnt(0)` at every tile top gives
  bit-identical wrong output;
* **not the main K loop** — reproduces with `k_steps == 1`, i.e. prologue plus
  the peeled step only;
* **not non-determinism** — 20 repeats give the same bad count.

**Reproducer:** `96x96x256`, `wave_layout 2x2`, `P=3`, `cluster 4x4`, at `512^3`
(~68% of elements wrong). Builds in ~3 s.

---

## 3. `kExpM == 1` — no room for the barrier handshake

**Status:** guarded by `static_assert`, not a silent failure.

`B_M / (16 * TileM) == 1` leaves the prefetch window occupying the whole tile,
so there is nowhere ahead of it for the handshake. Fixing it means moving the
handshake into the previous tile — a restructure, not a constant. This is what
keeps `B_M = 16` out of reach even under the `1x4` layout (`B_M >= 32`).

---

## How to test for these

A plain tolerance check is a poor detector here — picking the tolerance is the
hard part. Two techniques that worked:

**Outlier detection across kids.** Every co kid computes the same GEMM with the
same fp32 accumulation, differing only in summation order, so on one input their
max-abs errors cluster tightly — on 4 of 5 shapes all 203 kids returned a
*bit-identical* max error. A kid an order of magnitude off is wrong, with no
tolerance to choose. This is what found issue 1.

**Repeat runs.** Any non-determinism across repeats with fixed data is a bug by
definition, whatever the magnitude.

Shape coverage that matters, and why:

* `N < B_N` / `M < B_M` — a tile wider than the whole matrix is a distinct path
  from a partially out-of-range last tile;
* large shapes (`2048x2048x*` and up) — issue 1 is invisible below that;
* `K % 64 != 0` — see the K-tail cliff in `README.md`;
* several repeats per kid, not one.
