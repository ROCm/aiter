---
name: mxfp4-tuning-policy
description: How to choose, run and trust MXFP4 a4w4 MoE tuning on gfx950 — the GEMM1/GEMM2 search space and its constraint interactions, which axes actually pay, the env vars without which you silently benchmark the wrong kernel, and the measurement protocol (noise floor, repeats, per-stage attribution). Use when tuning or re-tuning any *_a4w4/*_fp4 tuned_fmoe.csv, when picking candidates for gemm_moe_tune.py --mxfp4-flydsl, when writing the --policy file for recommend_mxfp4_candidates.py, or when a tuned row's recorded us does not reproduce end to end.
argument-hint: [model config name, e.g. kimik3_a4w4, or a tuned_fmoe.csv path]
---

# MXFP4 a4w4 MoE tuning policy

Applies to the FlyDSL a4w4 *port* (`flydsl_mxmoe_g1_*` / `flydsl_mxmoe_g2_*` /
`flydsl_moe2_layout_*`) driven by
`csrc/ck_gemm_moe_2stages_codegen/gemm_moe_tune.py --mxfp4-flydsl`.

Numbers below were measured on gfx950 (MI355X) unless marked otherwise. Where a
claim is structural (read from code) rather than measured, it says so.

## 1. The search space

`Mxfp4FlydslTuner._g1_variants` enumerates a cross-product and lets the kernel's
own `_assert_supported` reject the illegal combinations, so the tuner can never
propose a name the runtime refuses.

| Axis | Values | Source |
|---|---|---|
| `(BM, use_nt, inline_quant)` | `(16,T,T) (32,F,F) (32,T,F) (64,F,F) (64,T,F) (128,F,F)` | `MXFP4_G1_VARIANTS["fp4"]` |
| GEMM2 family | `flydsl_moe2_layout_*` only (§2) | `get_flydsl_stage2_v2_kernels` |
| `BN` | 64, 128, 256 | `_G1_BN` |
| `num_waves` | 4, 2 | `_G1_NUM_WAVES` |
| `k_wave` | 1, 2, 4 | `_G1_K_WAVE` |
| `prefetch_hidden` | False, True — **inline-quant only** | gated inline |
| `xcd_swizzle` | 0, 2, 4 | `_G1_XCD_SWIZZLE` |
| `BK` | 256 only | hardcoded |

Not swept, and deliberately so: `interleave`, `a_dtype`/`out_dtype`,
`enable_bias`, `ksplit`, and `act` (taken from the row). Sweeping them needs a
matching torch reference that `_prepare_case`/`_torch_ref` do not build, so the
candidates would only fail the accuracy gate.

Measured survivor counts after `_assert_supported`:

| shape | GEMM1 variants | (g1, g2) pairs |
|---|---:|---:|
| kimik3_a4w4 NE896/topk16/h3584/i384 | 78 | **1248** |
| kimik2_fp4 NE384/topk8/h7168/i512 | 84 | ~2500 |

A token-2 kimik3 sweep took **~50 min for 989 timed candidates** on one GPU, so
a full model CSV is hours to days. Budget accordingly, or pre-select (§6).

### Constraint interactions worth knowing

- `BN64` ⇒ `BM32`, non-inline, separated gate/up.
- `num_waves == 2` ⇒ effective `BN64`.
- `k_wave > 1` ⇒ `BM32`, non-inline, `num_waves * k_wave <= 8`, and
  `D_HIDDEN/BK` divisible by `k_wave`. (`k_wave=4` is unreachable on kimik3:
  `3584/256 = 14` is not divisible by 4.)
- **`BM16` is the only inline-quant (`_f16in`) variant**, therefore the only one
  that can carry `_hpf`, and the only one that reads raw bf16 `hidden_states`.

## 2. Which axes actually pay

**`block_m`** dominates, and it is the axis a slate must never collapse on.

Track routed rows per expert (`token*topk/expert`). The shipped dispatch
heuristic is only a *prior*, and on kimi-k3 it is wrong at both ends: it returns
32 at token 256/512 where the tuned config is 16, and 128 at token >= 4096 where
the tuned config is 64. Measured tuned block_m for kimi-k3 (NE=896, topk=16):

| token | 2-512 | 1024 | 2048+ |
|---|---|---|---|
| block_m | 16 | 32 | 64 |

Other configs differ — kimi-k2/glm5/qwen do ship block_m 128 at large tokens — so
do not hard-code a ceiling. **Cover the tiers instead of trusting the prior**: a
candidate slate should contain at least three block_m tiers, including the one
the prior does *not* favour. Ranking by distance-from-prior alone is what caused
the prompt to hold a single tier (see §6).

**`prefetch_hidden` (`_hpf`)** hoists the next K-tile's hidden load out of the
inline-quant step. GEMM1 µs, kimik3 a4w4, `_hpf` vs the identical row without it:

| tok | 2 | 3 | 4 | 16 | 32 | 64 | 256 |
|---|---:|---:|---:|---:|---:|---:|---:|
| speedup | **1.20x** | 1.02x | 1.03x | 0.99x | 1.01x | 1.01x | 1.02x |

So: worth trying at very small token counts, noise above ~16. Cosine error is
identical either way — it is a scheduling change, not a numerical one.

**`BN`** has no reliable prior either. The intuitive "few routed rows -> BN128"
rule is *backwards* on kimi-k3: tokens 3-32 ship `BN256`, and tokens 128/512 ship
`BN128` with `xcd2`. Treat BN like block_m — stratify over it rather than predict
it.

**`xcd_swizzle`** tends to matter once blocks greatly outnumber CUs, and shows up
in a majority of kimi-k3's tuned rows (`_xcd2`/`_xcd4`). Do not spend the whole
slate on `xcd=0`.

### GEMM2: layout family only

**a4w4 pairs `flydsl_mxmoe_g1_*` with `flydsl_moe2_layout_*`, and nothing else.**
The native `flydsl_mxmoe_g2_a4w4_*` family is not a candidate: its BK=256
contraction requires `D_INTER % 256 == 0`, so it cannot serve `inter_dim` such as
384 at all, and the layout family spans the same tile space while carrying the
`sort_block_m` contract the port's intermediate needs. Some older shipped rows in
kimik2/glm5/qwen still name `mxmoe_g2`; re-tuning migrates them to the layout
equivalent.

`tile_n` should divide `model_dim`, `tile_k` should divide `inter_dim`. `atomic`
epilogs avoid a separate reduction; `reduce` epilogs need one and are the only
ones `AITER_FLYDSL_STAGE2_FP8` affects.


## 3. Env vars you must set, or you benchmark the wrong thing

**`AITER_SITUV2_A4W4=1`** — required for any SiTUv2 config (kimik3_a4w4).
Without it `fused_moe` picks `q_dtype_a=bf16` for per_1x32 SiTUv2 and dispatches
`gemm1_a16w4_port_*`; the a4w4 rows are never exercised and `kernelName1/2` are
ignored entirely. The failure is silent — you get plausible timings for a
different kernel family.

**`AITER_FLYDSL_STAGE2_FP8=1`** — the stage2 intermediate kimik3_a4w4 ships.
Only reaches `flydsl_moe2_layout_*` rows with a **`reduce`** epilog
(`_flydsl_v2_stage2_wrapper`: `_s2_fp8_inter = epilog == "reduce" and
_flydsl_stage2_fp8_enabled()`), i.e. 5 of kimik3's 17 rows. Omitting it makes
stage2 produce a bf16 intermediate instead of fp8 and moved one kimik3 row by
**1.9 %** — enough to invert a verdict. This bit us: a "confirmed regression" at
token 2048 was entirely this.

Silu configs (kimik2, glm5, qwen) reach a4w4 by default and have no
layout+reduce row among their changed rows, so neither flag applies to them.

## 4. Accuracy gates

- For fp4, `checkAllclose` "failed!" / high mismatch fractions are **meaningless**
  — they fire on correct output. Judge on `cosine_diff` (tuner) or `logits_diff`
  (`run_config`). Healthy is ~0.011 cosine / ~5e-4 logits; broken is O(0.1–1).
- **NaN must be rejected explicitly.** `float('nan') > errRatio` is `False`, so a
  candidate producing garbage passes a naive gate and, being fast, wins the sweep.
- SiTUv2 must be tuned *as* SiTUv2: the kernel name needs `_situv2` and both
  `situ_beta`/`situ_linear_beta` must reach the kernel, or the reference computes
  a different activation and every shape reports `logits_diff ≈ 0.14` with
  `out_norm/ref_norm ≈ 0.61` — a constant ratio, which is the signature of a
  scale disagreement rather than a bad kernel.

## 5. Measurement protocol

**Never diff recorded `us1`/`us2` across CSVs.** Those columns come from whichever
harness produced them and are not comparable. Re-measure both arms in one
harness. Observed: a row whose CSV claimed a 27 µs GEMM2 win measured 1.3 µs.

**The single-shot noise floor is ±3 %** at these sizes — wide enough to invert a
2 % result. Any row within ~1 % of parity must be re-run **three times in
alternating order** (`main, pr, main, pr, …`) and judged on the median. Doing
this once took seven apparent regressions down to one; the survivor had
non-overlapping triples.

**Per-stage attribution**: `rocprofv3 --kernel-trace --stats --output-format csv`,
sum `TotalDurationNs` per kernel, divide by the GEMM2 launch count (GEMM2 fires
once per `fused_moe` call). Exclude torch-reference and `fused_topk` kernels —
they run once, not per iteration. Roles: `gemm1_a4w4_port_*` / `mfma_moe1_*`
(GEMM1), `gemm2_a4w4_port_*` (GEMM2), `*sort*`/`*quant*` (prologue),
`moe_reduction_*` (reduce).

**Isolate a two-variable change with a 2×2.** When a row changes both GEMM1 and
GEMM2, cross them (`main+main`, `pr+pr`, `pr+main`, `main+pr`) before attributing
the delta. A per-stage table alone can mislead: the *same* GEMM1 kernel measured
233.8 µs after one GEMM2 and 252.4 µs after another, reproducibly — kernels
interact through L2 across iterations.

## 6. Choosing candidates (for `recommend_mxfp4_candidates.py --policy`)

You are picking what gets **benchmarked**, not guessing a winner. Spend the budget
on candidates that are individually plausible and collectively diverse — a slate
of near-identical kernels wastes the sweep.

1. **Stratify, do not sort.** Allocate the budget round-robin across every GEMM1
   axis the kernel name encodes -- `block_m`, `BN`, `xcd_swizzle`,
   `prefetch_hidden`, `k_wave`, `use_nt`, `num_waves` -- crossed with the GEMM2
   epilog. That is *every* axis the GEMM1 name encodes, and the completeness is
   the point: each axis was added only after a shipped config was measured to be
   hiding behind it, and the last one (`num_waves`) surfaced only when an
   upstream commit retuned a shape onto a `_w2` config that no prompt budget --
   96, 128, even 192 -- could reach. Each
   of those was added only after it was *measured* to be hiding a shipped
   config; none was added speculatively. Budget ~96 candidates (about 1.3x the
   ~72 cells) so every legal tier is represented. A global
   sort keyed on distance-from-prior is not safe here: with thousands of legal
   pairs against a 256-candidate prompt budget it filled the entire prompt with
   one `block_m`, and on kimi-k3 tokens 4096/8192/32768 that was block_m 128
   while the tuned config is 64 — the model could not have picked it at any
   temperature. A wrong prior must cost ordering, never coverage.
2. **Budget for two dimensions.** A slate is a `(GEMM1, GEMM2)` product. With
   `top_k` 8 and four block_m tiers, each GEMM1 gets one GEMM2 and the pairing is
   effectively unsearched — measured: on kimi-k3 tokens 256 and 4096-32768 the
   slate held the *right* GEMM1 and still lost, purely on its GEMM2 partner. Use
   `top_k` >= 16 for a full-range sweep, or pair a smaller GEMM1 set with several
   epilogs each.
3. At `token <= 8` on an inline-quant shape, always include an `_hpf` candidate
   and its non-`hpf` twin — that is where the 1.20x lives and the pair makes the
   effect attributable.
4. Vary one axis at a time across the slate so the result is interpretable.
5. Include at least one `atomic` and one `reduce` GEMM2 when both are legal. All
   GEMM2 candidates are `flydsl_moe2_layout_*`; never propose `flydsl_mxmoe_g2_*`.
6. Cover both `BN` values and both `xcd_swizzle != 0` options; neither has a
   trustworthy prior (§2).
7. **Stratify `k_wave` too.** An earlier version of this rule said not to spend
   slots on it unless BM32 non-inline was already competitive; that was wrong.
   Across glm5/kimi-k2/qwen, 7 of 9 GEMM1 misses were `_kw2` configs, and adding
   `k_wave` as a stratum recovered 8 of those 9 (other-model GEMM1 containment
   52/61 -> **60/61**). kimi-k3 could not have revealed this: `3584/256 = 14` is
   not divisible by 4, so its k_wave space is thin. Never generalise an axis
   prior from a single model config.

### Generalisation: the other three model configs

The rules above were derived on kimi-k3 and then checked, unchanged, against the
61 rows this PR changes in glm5 / kimi-k2 / qwen3.5-397B — different shapes
(NE 384/385, topk 8/9, `model_dim` 7168/6144/4096):

| model | rows | block_m | GEMM1 |
|---|---:|---:|---:|
| glm5_fp4 | 5 | 5/5 | 5/5 |
| qwen3_5_397b_fp4 | 4 | 4/4 | 4/4 |
| kimik2_fp4 | 52 | 52/52 | 51/52 |
| **total** | **61** | **61/61** | **60/61** |

`block_m` transferred perfectly and GEMM1 reached 98%, so the stratification
rules are not kimi-k3 artefacts. The single residual miss (kimi-k2 token 2048,
inter 256, `32x256x256_nt_xcd4`) sat at index 32 of a 64-candidate cell -- no
prompt budget could reach it -- because `use_nt` was not yet a stratum. Adding it, and then `num_waves`,
takes **prompt-level coverage to 64/64 distinct shapes across all four model
configs**: with a
96-candidate budget, every shipped GEMM1 is now offered to the model for every
shape it changes. That is the ceiling; `top_k` relative to the cell count converts it into slate
containment. Both remaining slate misses were re-run against the fully
stratified prompt:

| shape | `top_k` 32 | `top_k` 64 |
|---|---|---|
| kimi-k2 tok 2048, inter 256 | hit (recovered by the `use_nt` stratum alone) | hit |
| kimi-k3 tok 4 | miss | hit |

So with all six GEMM1 axes stratified, a 96-candidate prompt and `top_k` ~64
(about 0.9x the ~72 cells), **every shipped GEMM1 this PR changes is reachable in
the slate**. Below that, containment degrades gracefully -- it is a budget knob,
not a coverage cliff. Exact-pair containment is 0/61 by construction:
every one of those rows ships a `flydsl_mxmoe_g2_*` GEMM2, the family a4w4
excludes (§2).

### Measured: what actually moves reachability

Shipped-config containment for kimi-k3 (15 non-legacy shapes), each step
cumulative, measured against `kimik3_a4w4_tuned_fmoe.csv`:

| slate construction | block_m | GEMM1 |
|---|---:|---:|
| global sort by distance-from-prior, `top_k` 8 | 10/17 | 5/17 |
| + stratify `(block_m, epilog)` | **15/15** | 8/15 |
| + `top_k` 16, stratify BN | 15/15 | 8/15 |
| + stratify `xcd_swizzle` | 15/15 | 9/15 |
| + stratify `prefetch_hidden`, `top_k` 32, prompt 128 | 15/15 | 12/15 |
| + `--g2-per-g1 3` (expand each chosen GEMM1) | 15/15 | **14/15** |

And the measurement that actually matters -- slate winner vs the shipped config,
both benchmarked **in the same sweep** so no cross-run variance:

| slate | shapes where slate >= shipped |
|---|---:|
| `top_k` 8, no stratification | 4/15 |
| `top_k` 32, stratified | 5/15 |
| `top_k` 32 + `--g2-per-g1 3` (~90 candidates/shape) | 9/15 |
| + repeats on the near-parity shapes, `--g2-per-g1 0` | **11/15** |

Two of those six resolved once measured properly rather than asserted:

- **token 512** was noise: three alternating repeats give 320.9/320.8/320.3 vs
  321.5/321.4/319.6 — ratio **0.998**, spread 0.6%.
- **token 1024** was *not* noise, contrary to a first reading. Repeats put it at
  a reproducible 0.985 (spread 2.1%). Its GEMM1 was in the slate; the tuned GEMM2
  was not. For a fixed GEMM1 the layout family offers only ~16 partners and they
  are one-per-stratum, so stratifying degenerates to ranking and a top-3 take
  still missed it. With `--g2-per-g1 0` (every legal partner) the pair is in the
  slate and three repeats give **1.000**.

Do not call a sub-1% gap "noise" without running the repeats in §5 — one of these
two was real.

The remaining four are the `_sp` GEMM2 variants that are not enumerable (§9):
tokens 4096-32768 measure 0.961/0.981/0.978/0.979 and cannot do better while that
axis is missing. **That axis is worth real time.** Benchmarking each shipped
`_sp` config against its identical no-`_sp` twin (same GEMM1, same tile, same
epilog), three repeats each:

| token | `_sp` | no-`_sp` twin | ratio |
|---|---|---|---:|
| 4096 | 669/667/674 | 713/709/706 | **0.943** |
| 8192 | 1026/1024/1025 | 1042/1044/1039 | 0.984 |
| 16384 | 1780/1784/1786 | 1809/1813/1819 | 0.984 |
| 32768 | 3318/3344/3339 | 3380/3409/3439 | 0.979 |

So `spart` buys 1.6-5.7% at large tokens and the gap on those four shapes is a
**search-space** limitation, not a slate-quality one: no recommender can close it
while `build_flydslv2_gemm2_name` has no `sp` parameter. Note token 4 is a GEMM1-containment miss that still measured
**1.020** — exact-name containment is a proxy, throughput is the target.

Two lessons. **Stratification, not prompting, is what fixed coverage** — the
first step alone took block_m from 10/17 to 15/15, and each added stratum moved
GEMM1 containment while policy wording alone did not. And at the last step the
shipped GEMM1 was present in the prompt for **15/15** shapes with 33 distinct
GEMM1 offered; the 3 remaining misses are the model declining to pick 3 of 33
when choosing 31. So once the prompt is a proper cover, `top_k` relative to the
number of strata is the only remaining lever: `top_k >= strata` makes containment
exact by construction, at the cost of the model no longer selecting anything.

## 7. Known blind spot

The tuner ranks GEMM2 by its **isolated `us2`** and cannot see a cost a GEMM2
imposes on another kernel. A GEMM2 that is faster on its own stage can still lose
end to end through L2 displacement of the following GEMM1. If a tuned row's
recorded `us1 + us2` improves but e2e does not, suspect this and check with the
2×2 in §5.

> Not yet quantified on this branch: an earlier attempt to measure the `nt`
> component of that effect was run without `AITER_FLYDSL_STAGE2_FP8=1` and is
> void. The structural blind spot is real; the magnitude is unmeasured.

## 8. Wiring

```bash
# CPU-only: recommend candidates into a CSV
python csrc/ck_gemm_moe_2stages_codegen/recommend_mxfp4_candidates.py \
  -i aiter/configs/model_configs/kimik3_a4w4_untuned_fmoe.csv \
  -o /tmp/cand.csv --gfx gfx950 --cu-num 256 --top-k 8 \
  --policy .claude/skills/mxfp4-tuning-policy/SKILL.md

# GPU: tune only those candidates
AITER_SITUV2_A4W4=1 AITER_FLYDSL_STAGE2_FP8=1 \
python csrc/ck_gemm_moe_2stages_codegen/gemm_moe_tune.py --mxfp4-flydsl \
  -i <untuned.csv> -o <tuned.csv> --candidate-csv /tmp/cand.csv
```

`--gfx`/`--cu-num` are required because the tuner injects the *runtime* arch into
rows (`aiter/utility/base_tuner.py:369`); the recommender may run on a different
host, and the tuner errors if the CSV's arch does not match.

## 9. Known coverage limits

Two kinds of shipped row this tuner cannot propose:

- **`flydsl_moe1_*` GEMM1** (kimik3 tokens 1 and 8). That is the generic non-port
  flydsl family, dispatched by `fused_moe.py:2905` on the bare `flydsl_` prefix
  and tuned by a different path. Not a hole in the mxmoe space -- a different
  family. Re-tuning such a row here migrates it into the port.
- **`_sp<N>` GEMM2 variants -- accepted limitation, decided.**
  `get_flydsl_stage2_v2_kernels` enumerates no spatial-partition names, so a
  shipped row naming `_sp801`/`_sp1601` (kimi-k3 tokens 4096-32768) cannot be
  reproduced. A bare name resolves to the dispatcher default
  `MXFP4_G2_SPART=402`, so the sweep explores 402 only.

  This is deliberate, not an oversight. `spart` was measured to be worth
  **1.6-5.7%** on those shapes -- each shipped `_sp` config against its
  byte-identical no-`_sp` twin, three repeats: 0.943 / 0.984 / 0.984 / 0.979
  (tokens 4096 / 8192 / 16384 / 32768). Adding a `sp` parameter to
  `build_flydslv2_gemm2_name` plus a `spart_values` axis would make them
  reachable (measured 11/17 -> 15/17 kimi-k3 containment) at the cost of
  tripling the GEMM2 axis. **The call was to accept the gap instead.**

  Consequence to keep in mind: those four rows ship configs the tuner cannot
  propose, so a future kimi-k3 re-tune will replace them with non-`_sp`
  equivalents that are 1.6-5.7% slower. That is expected. Do not treat it as a
  regression, and do not re-open the axis without a fresh decision.

  With that gap accepted, candidate reachability is complete everywhere it is
  achievable: 64/64 prompt coverage against every shipped a4w4 config across the
  four model files, and 8/8 against glm5 optima found by exhaustive search. The
  four `_sp` rows are the only configs outside the space, by choice.

11 of kimi-k3's 17 shipped rows are exactly reachable; the rest are the
`flydsl_moe1_*` pair and the four `_sp` rows.

Related: `aiter-config-shape` (landing the tuned CSV without duplicate shapes).
