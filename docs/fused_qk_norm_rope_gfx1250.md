# fused_qk_norm_rope_group_quant on gfx1250

Companion to `qk_norm_rope_group_quant.md` (which is MI355/gfx950 work). Almost
nothing in that document transferred; see §1.

Machine: gfx1250, 256 CU / 1024 SIMD / **wave32** / 16 waves-per-SIMD ceiling.
Shape used throughout: DeepSeek-V4-Pro, D=512, RD=64, MQA (NK=1), G=32.

---

## 1. The headline: this kernel collapses on wave32

`hip/flydsl` ratio, same shapes, both arches:

| T | H | gfx950 (doc §5) | **gfx1250** |
|---|---|---|---|
| 64 | 16 | **0.94** (HIP faster) | **2.15** |
| 1024 | 16 | 1.03 | 1.25 |
| 16384 | 16 | **0.98** (HIP faster) | 1.66 |
| 1024 | 128 | 1.10 | 1.57 |
| 16384 | 128 | 1.03 | **2.49** |

gfx950 sits at 0.94–1.10. gfx1250 sits at 1.18–2.49, and gets **worse with more
heads** — the opposite of gfx950.

So this is not "the kernel needs tuning". It is structurally mismatched to
wave32, and **every constant in the dispatch heuristic was tuned on MI355**
(`FG_MANY_HEADS_MIN=128`, `PREFILL_Q_HEADS_PER_WAVE_*` = 3/8/16, tier thresholds
4/48/300 blocks-per-CU). Treat all of them as unvalidated here.

### Why wave32 hurts

| | wave64 (gfx950) | wave32 (gfx1250) |
|---|---|---|
| elements/lane (`vec_size_i`) | 8 | **16** |
| `in_chunk_bytes` per load | 1 chunk | **2 chunks** |
| threads per 1-wave block | 64 | **32** |

`vec_size_i = (HEAD_DIM/8 <= WARP_SIZE) ? 8 : 16` — 512/8 = 64, which is `<= 64`
on wave64 and `> 32` on wave32. Each lane carries twice the elements, so the
serial per-head dependency chain is twice as long.

---

## 2. What shipped (validated)

| change | knob | measured |
|---|---|---|
| Q-head TDM prefetch ring in the **coarse** kernel | `AITER_COARSE_Q_TDM_DEPTH=3` | **−11.3%** @H=32 T=1024, **−15.8%** @H=32 T=16384 |
| `%peak` column made arch-aware | (test file) | fixes a 2.2x overstatement |

Both paired `op_test`, idle GPU, CI excluding zero.

### 2.1 The coarse TDM ring

The Q-head loop was strictly serial:

    load head k -> wave_reduce (full-wave barrier) -> norm/rope/quant/store -> load head k+1

MLP = 1. ATT at T=16384 (78 waves) had `s_wait_loadcnt` at **57.9%** while
`FETCH_SIZE` was **1.089x** ideal — traffic already at the floor, latency simply
not hidden.

The ring issues `DEPTH` tiles in a prologue and refills one slot per iteration,
consuming with a dispatched `s_wait_tensorcnt<DEPTH-1>` so the other tiles stay
in flight across each head's reduce/rope/quant/store.

ATT before/after (78 vs 319 waves, both verified against the right checkout):

| category | DEPTH=0 | DEPTH=3 |
|---|---|---|
| `s_wait_loadcnt` | 12573 (57.9%) | **1619 (12.1%)** |
| VALU/LDS | 3707 (17.1%) | 5782 (43.3%) |
| `s_wait_xcnt` | 3904 (18.0%) | 3101 (23.2%) |
| `s_wait_tensorcnt` | 0 | 478 (3.6%) |
| **cyc/wave** | **21707** | **13354** |

The load wait collapses; the `ds_load` cost (+2075 VALU, +541 dscnt, +478
tensorcnt) is a third of what it buys.

An earlier in-tree comment said a vLLM-style 2-deep **register** prefetch here
measured neutral on MI355 because the extra live `vec_q_next` cost VGPR. That
objection does not apply to TDM: `tensor_load_to_lds` is scalar and its in-flight
data sits in LDS, so VGPR is unchanged (92 -> 90/94).

---

## 3. Knobs that are OFF (built, not validated)

All default to the original behaviour. Turn on with `-D<NAME>=<value>`.

| knob | default | status |
|---|---|---|
| `AITER_FG_USE_TDM` | 0 | **Validated −11.32% at xlarge** (H=128 T=16384, 570.07→505.56, 3 clean pairs, sd 0.10pp) but the same flag reaches **decode**, where TDM is neutral-to-harmful (a wave owns one head, so there is no second tile to overlap). One unclean run had H=16 T=64 at 6.99→10.26 us. Needs a per-tier gate before it can default on. |
| `AITER_XLARGE_USE_COARSE` | 0 | Never run. Routes the xlarge tier to coarse. Motivation in §4. |
| `AITER_COARSE_SCALE_ALL_LANES` | 0 | Never run. Drops the `tid % Q_REDUCE == 0` dedup on the Q scale store. |
| `AITER_FG_TOKENS_PER_WG` | 1 | Measured noise (−1.68% / +3.55%, CIs cross zero). |
| `AITER_FG_HEADS_PER_WAVE` | 1 | **Regression +20%** at decode (VGPR 66→86, occupancy 14→11, wave count halved). |
| `AITER_FG_HEADS_PER_BLOCK` | 1 | **Regression +4~5%** at decode. |

---

## 4. The largest remaining gap: rows per wave

At H=128 T=16384 the tier is `xlarge` -> FG, and FG is **one (token,head) row per
wave**:

| | HIP FG | flydsl TDM path |
|---|---|---|
| block | 32 thr = 1 wave | 256 thr = 8 waves |
| rows per WG | **1** | **128** |
| **rows per wave** | **1** | **16** |
| workgroups | **2,113,536** | **16,384** |
| waves | **2,113,536** | **131,072** |

129x the workgroups, 16x the waves. Every row pays its own kernarg read,
`positions` chase, cos/sin setup and descriptor build; flydsl amortises all of
that over 16 rows.

This tracks the measured ratios: H=128 (xlarge -> FG, 1 row/wave) is 2.49x, while
H=16/32 (large -> coarse, 8 rows/wave) are 1.53–1.66x.

`AITER_XLARGE_USE_COARSE=1` routes xlarge to coarse, landing at 36,864 WGs /
147,456 waves with 16 heads/wave — essentially flydsl's shape — and coarse
already carries the TDM ring. **Built, never run.** Note `FG_MANY_HEADS_MIN=128`
exists precisely to send H>=128 to FG on MI355, so this reverses an MI355
decision and needs correctness plus perf validation.

---

## 5. Occupancy and where time goes now

Coarse at T=16384 is **VGPR-capped at 10 waves/SIMD** (92–96 VGPR, `1024/96 = 10`),
against a 16 ceiling. Confirmed by ATT occupancy replay: peak 10, time-avg 9.0–9.2.

After the TDM ring the profile is no longer memory-wait bound:

| | share |
|---|---|
| VALU/LDS | **43.3%** |
| `s_wait_xcnt` | **23.2%** |
| `s_wait_loadcnt` | 12.1% |

The xcnt is concentrated in three in-loop sites, all the same pattern — an
exec-masked store followed by the EXEC restore, which is a WAR hazard forcing a
full `s_wait_xcnt 0x0` address-queue drain:

| cyc/wave | guarded op | hit |
|---|---|---|
| 1030 | `global_store_b16 ... offset:448` (scale store) | 2032 |
| 849 | `buffer_store_b128` (nope payload) | 2032 |
| 662 | exec region exit | 2032 |

`hit=2032 / 319 waves = 6.4` = the per-wave Q-head loop count, i.e. once per head.
Together ~19% of the kernel. `AITER_COARSE_SCALE_ALL_LANES` targets the first.

---

## 6. Tooling and reproduction

Everything lives outside the repo, under `/data/hwang/`:

| path | purpose |
|---|---|
| `prof_out/att_wide.sh` | ATT capture. `T= H= G= SE= BUF= CONSEC= DIR=` |
| `prof_out/ab_ctdm.sh`, `ab_h128.sh` | paired A/B over prebuilt `.so`, clean/DIRTY tagged |
| `prof_out/an.py` | paired-ratio stats (mean, sd, n, sign count, 95% CI) |
| `ktime_fused.py` | true GPU kernel duration via `--kernel-trace` |
| `attlib/` | ATT decoder libs (`LD_LIBRARY_PATH`) |

**On a new machine these paths will not exist.** `att_wide.sh` and the A/B
scripts hard-code `/data/hwang/aiter` and `/data/hwang/attlib`; fix those first.
The ATT decoder (`librocprof-trace-decoder.so`) must be on `LD_LIBRARY_PATH` or
captures silently produce no `stats_*.csv`.

### Traps that cost real time here

1. **`PYTHONPATH` must reach the traced child.** `rocprofv3 -- python ...` spawns
   python; without an exported `PYTHONPATH` the import falls back to a different
   checkout (`/app/aiter` on this box) and the whole trace is of the wrong
   binary. Two captures were silently invalid this way, and one produced a
   confident but meaningless conclusion. `att_wide.sh` now greps `capture.log`
   for the loaded path and exits non-zero on mismatch — keep that guard.
2. **aiter prints `finish build ... cost Xs` even when the build FAILED**, with a
   `failed jit build` line just above. Test for `failed jit build`, not for
   `finish build`.
3. **ATT sampling is time-windowed.** A 7 us kernel yields ~8 waves; 200 us yields
   ~1000. Use `--att-consecutive-kernels 16` for short kernels (8 -> 141 waves at
   T=64). Widening `--att-shader-engine-mask` alone did almost nothing (8 -> 9).
4. **Never compare wall-clock across builds with ATT.** On `inverse_rope` its span
   reported the shipped TDM commit 4.1% *slower* where paired ktime measured it
   4.63% faster. ATT is for within-build attribution only.
5. **Two variants per paired rep, not three.** Three raised paired-ratio noise
   from sd 0.9pp to 2.0pp — enough to hide the effects being chased.
6. **`%peak` was a hardcoded MI355 8 TB/s.** On gfx1250 (20 TB/s spec, ~17.9
   achievable on a 2:1 mix) it overstated by 2.2x — 94% reported where the real
   figure was 42%. Fixed, but check any older numbers quoted from that column.

---

## 7. Methodology note

Seven optimisation directions were tried before the first one worked, and the
common failure was **treating "I cannot measure an improvement" as "there is no
improvement available"**. Statements like "this is the shape's latency floor" and
"traffic is at the floor, nothing left" were written and were wrong: flydsl does
the same decode work in 3.96 us against 6.99, and the same prefill work at ~79%
of the bandwidth ceiling against 42%.

The fix was cheap and should have come first: **measure a known-faster reference
before optimising**. `op_test` compares against flydsl by dropping
`--no-flydsl`; it takes one run. With a 1.2–2.5x gap on the table, the analysis
changes from "3% left to shave" to "something structural is wrong", and the ATT
data reads completely differently.
