# Known correctness issues in the pre-compiled (`.co`) a16w16 families

Three open defects. All produce **wrong results, not faults**, so only a numeric
check sees them. Recorded so the next session starts from the reproducers.

Applies to `a16w16_4wave_co` (reference, `..._4wave_compute_...cuh`) and
`a16w16_4wave_wl_co` (`..._4wave_wl_...cuh`) unless noted.

---

## 1. Narrow `B_N` races at large shapes — SHIPPED SET FILTERED

**Status:** open. Affects BOTH families, so it predates the wave-layout work.
The 60 affected kids have been removed from `co_kernels.json`.

Non-deterministic: same seed, same data, the wrong-element count changes every
run — including runs that come out clean.

```
kid 21182  64x64x128 w1x4 c4x1  @ 2048x2048x7168
  rep0 nbad=25416   rep1 nbad=24602   rep2 nbad=33105
```

Scope over 203 kids x {2048x2048x7168, 2048x2048x4096, 4096x4096x4096} x 3 reps:

| `B_N` | racy / total |
|--:|--:|
| 64 | **50 / 50** |
| 128 | 10 / 51 |
| 192 | 0 / 51 |
| 256 | 0 / 51 |

Every `B_N = 64` kid races. `B_N >= 192` is clean. By family: reference 16/64,
wl 44/139 — **the reference family is affected**, so this is not a regression
from this session's generalisation work.

It is shape-dependent as well: `1024x1024x7168` is clean on kid 21182 while
`2048x2048x*` is not, so a small validation shape will not see it.

**This explains the transient tuner warnings** first seen on the PR #4749
`M=2048, K=7168` sweep (7 `maxDelta` warnings, then two clean re-runs). A race
produces exactly that: warnings that do not reproduce, and standalone replays of
the named kids that look fine. An earlier revision of this file concluded "co
family cleared" from those clean replays — that was wrong, and clean replays of a
racy kernel prove nothing.

**Danger:** at `1024x1024x7168` kid 21182's error stays under the tuner's
`0.1 * max|ref|` bound, so the tuner would *accept* it. The bound is not a
sufficient gate for this defect.

**Reproducer:** any dropped kid at `2048x2048x7168`, 3 repeats, compare the
wrong-element count across repeats.

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
