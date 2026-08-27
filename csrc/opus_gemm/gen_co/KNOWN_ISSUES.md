# Pre-compiled (`.co`) a16w16 families: constraints and resolved defects

Two constraints are open, two defects were found and fixed. Neither open
constraint can ship a wrong answer — a `static_assert` refuses the
configuration — and both have been priced and declined rather than merely left
undone; the verdicts and their numbers are below. The resolved sections stay
because what they cost to find is in the reproducers and the dead ends, not in
the fix.

Applies to `a16w16_4wave_co` (reference, `..._4wave_compute_...cuh`) and
`a16w16_4wave_wl_co` (`..._4wave_wl_...cuh`) unless noted. Numbering is stable;
commits and `README.md` refer to these by number.

| | what | status |
|---|---|---|
| 1 | ring-buffer WAR + zero-extent TDM zero-fill | fixed, shipped in the tracked `.co` |
| 2 | `B_K > 128` with more than one msb group | open, `static_assert`; **declined**, blocks nothing that wins |
| 3 | `kExpM == 1` leaves no room for the handshake | open, `static_assert`; **declined**, measured below the budget floor |
| 4 | tuner silently never nominated the `wl` family | fixed |

One decision is still genuinely open: whether the 1-WG/CU LDS pad should go.

---

## Open constraints

### 2. `B_K > 128` together with more than one msb group

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


### 3. `kExpM == 1` — no room for the barrier handshake

**Status:** guarded by `static_assert`, not a silent failure.

`B_M / (16 * TileM) == 1` leaves the prefetch window occupying the whole tile,
so there is nowhere ahead of it for the handshake. Fixing it means moving the
handshake into the previous tile — a restructure, not a constant. This is what
keeps `B_M = 16` out of reach even under the `1x4` layout (`B_M >= 32`).


### Why neither is worth fixing

Measured before spending a restructure on it. Take the 38 shapes of the 84-shape
DSV4 sweep whose winner was a JIT `_ws` kid, and ask of each winning tile whether
the `.co` pipeline could express it:

| | shapes | |
|---|--:|---|
| legal today, simply had no entry | 25 | mostly `B_K = 256`, which the family had none of |
| blocked by issue 3 (`kExpM == 1`) | 10 | `32x32x*` needs `w2x2`; `16x64x256` needs `TileM=1` |
| blocked by issue 2 | **0** | every winning `B_K=256` tile has `kNSub == 1` already |
| geometrically impossible | 3 | `B_N = 32` has no legal layout |

So **issue 2 blocks nothing that won anything** — it is missing expressiveness,
not a missing win. And most of what issue 3 blocks is a duplicate: the same tile
under a different `wave_layout` is legal (`64x64x128 w4x1` is refused, `w1x4` and
`w2x2` are not). What only issue 3 can unlock is `B_M = 16`, which needs
`TileM = 1` and therefore `kExpM = 1`.

The first row is the one that paid. Adding 15 entries — `64x64x256` in two
layouts, `32x64x256` in one, five clusters each, no code change — and re-tuning
those 12 shapes moved 8 of them to `wl_co`, 1.156x geomean and up to 1.331x, and
took the group from 7/12 to 12/12 against triton.

The four it did **not** take are the useful part of that result: all four run at
`splitK` 2–5, and no `.co` family supports split-K. The eight it took are at
`splitK` 0 or 1. A `.co` entry can only win a shape that does not want split-K.

A second pass found 10 more shapes that could also reach the new tiles but had
not been re-tuned — adding candidates does nothing for a shape nobody re-tunes.
All 10 improved, 8/10 to 10/10 against triton.


### The probe that settled issue 3

The occupancy case for `B_M = 16` is real. Eight of the 84 shapes cannot be
filled by any legal `.co` tile without split-K, and `B_M = 16` would fill them;
they sit at `splitK` 3–5 for exactly that reason, so the family's usual
split-K handicap would not apply to them.

What kills it is the scheduling budget. This pipeline hides its ds_reads, its
handshake and its TDM issue in the gaps between WMMAs, and a K step only has
`(B_K/64) * kNumNSub * kExpM * kExpNPerSub * kExpKHalf` of them. Every `.co`
kernel that wins a shape today has **at least 16**:

| tile | WMMA / K step | shapes won |
|---|--:|--:|
| `128x256x128` | 128 | 26 |
| `64x64x256`, `64x128x128`, `128x64x128` | 32 | 17 |
| `32x64x256`, `64x64x128` | 16 | 7 |
| `32x64x128` | 8 | **0** |

That last row is measured, not extrapolated. `32x64x128 w1x4` is legal today, has
the same grid as `32x64x256` and therefore the same occupancy, and differs from
it only in K depth and budget — a controlled comparison of the budget alone.
Added as a probe and tuned head-to-head on the five shapes `32x64x256` owns, it
won **none**: 0.975–1.017x, i.e. noise. The probe entries were then removed.

`B_M = 16` reaches 8 with `B_K = 256` and 4 with `B_K = 128`, at or below the
level that just failed. Widening `B_N` to buy budget halves the grid and hands
the occupancy back, so there is no way out of the trade. Those eight shapes
belong to the `_ws` families, which are built for small tiles and can split K.

---


### Still open: should the 1-WG/CU pad go?

Separate question, and the answer is per kid rather than per shape. No pad +
fixes against the shipped padded build, same interleaved method:

| shape | mean | per-kid p10 / median / p90 |
|---|---|---|
| `2048x2048x7168` | +9.2% | -11.5% / +9.5% / +39.6% |
| `4096x4096x2048` | +5.8% | -15.3% / +4.3% / +40.9% |
| `4096x4096x1024` | **-15.6%** | -26.4% / -12.9% / +1.2% |

A spread from -26% to +41% within one shape is exactly what the kid table plus
the tuner exist for: add the 61 entries a second time carrying
`-DOPUS_CO_NO_1WG_PAD` and a `variant` suffix, and let tuning pick. Not done.

`build_co.py` now appends `--device-flag` to every entry's own flags and
`_expected_lds` takes the resulting `no_pad`, so a whole-family no-pad rebuild
passes the LDS check instead of failing it. That is the command-line path only —
per-entry no-pad variants would need `_expected_lds` to read the flag off the
entry as well.

The contrast that pins it: waiting one short of what the slot needs
(`s_wait_tensorcnt(kSlots - 1)`), so the step genuinely reads a fill still in
flight, produces ~3.2 M wrong elements across 1020 of 1024 tiles, whole tiles,
both operands, every run — nothing like the failure being diagnosed. So it was
never the counted `s_wait_tensorcnt` failing to cover its slot.


---

## Resolved

### 1. Narrow `B_N` raced at large shapes — two bugs, both fixed

**Status:** fixed, and the tracked `.co` are built from the fixed pipelines.
With the pad off and the fixes in, all 204 variants were clean at 2–3 WG/CU over
400 repeats.

The symptom was non-deterministic wrong results — same seed, same data, a
different wrong-element count every run, including runs that came out clean:

```
kid 21182  64x64x128 w1x4 c4x1  @ 2048x2048x7168   (before the fix)
  rep0 nbad=25416   rep1 nbad=24602   rep2 nbad=33105
```

**The control group that localised it.** VGPR and LDS both scale with the tile,
so the raw correlation cannot separate them, but the population splits three
ways:

| group | n | LDS admits 2 WG | VGPR admits 2 WG | occupancy | result |
|---|--:|---|---|---|---|
| A | 61 | yes (<= 160 KB) | yes (2x320..450 <= 1024) | 2 WG/CU | **all wrong** |
| B | 71 | no (> 160 KB) | yes | 1 WG/CU | all correct |
| C | 72 | no | no (2xVGPR > 1024) | 1 WG/CU | all correct |

A and B differ **only** in the LDS-driven occupancy — both would allow two waves
per SIMD on registers alone — so the variable is workgroups-per-CU, not tile size
or register pressure.

**The damage geometry named the edge.** At `2048x2048x7168`, across four variants
and twelve failing runs, every failure touched exactly one tile, spanned all of
its M rows, sat in a short prefix of wave 3's half of B — the last region read in
a K step, never wave 2's half and never A — and carried roughly an eighth of one
K step's contribution. A standalone probe that samples LDS mid-transfer shows the
engine filling rows in **ascending** order, so a reader that is early misses a
region's tail while a writer that is early clobbers its head. The damage was at
the head of the last-read region: a write-after-read.


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


### 4. The tuner silently never nominated the `wl` family

**Status:** fixed in `opus_gemm_tune.py`. Not a kernel bug, but it belongs here
because it failed the same way the others do: quietly, and only a measurement
saw it.

`kid_rejects_shape` answered `a16w16_4wave_co` early — no buffer resource, every
tail clamped by the D#, so no M/N/K alignment applies — and named only that tag.
`a16w16_4wave_wl_co` fell through to the rules below it, where a gfx942
whitelist, `BF16WS_EXACT_REDUCE_SHAPES`, refuses any `N` outside
`{64,128,256,512,1024,2048}` for a kid that "uses a bf16 workspace". The wl kids
have no workspace at all; they matched because `splitk_workspace_dtype` is a
field on every instance and its default had become `bf16_t`.

All 140 wl kids were therefore rejected at `N = 384 / 32320 / 129280`, and the
same rule took the `_ws` split-K families with them. At `256x384x7168` that left
**6 of 106 candidates, all one tile, none with split-K**, which the tuner then
reported as a flat ~25 us floor that did not scale with M — indistinguishable
from a slow kernel. With the family restored the same shape tunes to 7.0 us, and
across the 21 affected shapes opus gains 1.29x geomean, up to 3.65x.

The trap worth remembering: **a candidate that was never nominated reads exactly
like a candidate that lost.** Nothing in the tuner's output distinguishes them.
When a shape looks structurally slow, count the candidates before profiling the
winner — `candidate_kids_for_shape` minus `kid_rejects_shape` should not be
throwing away most of the set.

---


---

## Dead ends, so they are not walked twice

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
