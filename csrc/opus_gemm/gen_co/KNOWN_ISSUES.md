# Known correctness issues in the pre-compiled (`.co`) a16w16 families

One fixed, two open and guarded. All produce **wrong results, not faults**, so only a numeric
check sees them. Recorded so the next session starts from the reproducers.

Applies to `a16w16_4wave_co` (reference, `..._4wave_compute_...cuh`) and
`a16w16_4wave_wl_co` (`..._4wave_wl_...cuh`) unless noted.

---

## 1. Narrow `B_N` raced at large shapes — FIXED

**Status:** fixed by forcing 1 workgroup per CU. Kept here because the symptom
was subtle and the diagnosis is worth not repeating.

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

**What is NOT established: why 2 WG/CU breaks it.** Candidates, undistinguished:
the per-CU TDM request budget (a limit documented for the *split-K* pipeline in
`opus_gemm_common.py`, never measured for this one); or the cluster-scope
`s_barrier_signal(-1)` / `s_barrier_wait(-1)` handshake when two workgroups from
different clusters share a CU — the non-determinism and grid-size dependence fit
a barrier released early better than they fit a dropped transfer. Separating
these needs an ATT capture of a 2-WG/CU run; see `README.md` for the recipe.

The fix does not rest on the mechanism — it rests on "correct at 1 WG/CU", which
the A/B control establishes.

**Fix.** Pad `LDS_BYTES` past 160 KB when the real footprint is below it, so a
second workgroup cannot fit. The pad tail is never accessed. This is the same
trick `opus_cluster_tdm_splitk_ws_traits_gfx1250` already uses in this header
for the same reason. All 61 affected variants verify after the change.

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
