---
name: mxfp4-tuning-policy
description: How to choose, run and trust MXFP4 a4w4 MoE tuning on gfx950 — the GEMM1/GEMM2 search space and its constraint interactions, which axes actually pay, the env vars without which you silently benchmark the wrong kernel, and the measurement protocol (noise floor, repeats, per-stage attribution). Use when tuning or re-tuning any *_a4w4/*_fp4 tuned_fmoe.csv, when picking candidates for gemm_moe_tune.py --mxfp4-flydsl, when writing the --policy file for recommend_mxfp4_candidates.py, or when a tuned row's recorded us does not reproduce end to end.
argument-hint: [model config name, e.g. kimik3_a4w4, or a tuned_fmoe.csv path]
---

# MXFP4 a4w4 MoE tuning policy

Applies to the FlyDSL a4w4 *port* (`flydsl_mxmoe_g1_*` / `flydsl_moe2_layout_*`)
driven by `csrc/ck_gemm_moe_2stages_codegen/gemm_moe_tune.py --mxfp4-flydsl`.

Numbers were measured on gfx950 (MI355X). Claims that are structural (read from
code) rather than measured say so.

## 1. The search space

`Mxfp4FlydslTuner._g1_variants` enumerates a cross-product and lets the kernel's
own `_assert_supported` reject illegal combinations, so the tuner can never
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

Not swept, deliberately: `interleave`, `a_dtype`/`out_dtype`, `enable_bias`,
`ksplit`, `act` (taken from the row). Sweeping them needs a torch reference
`_prepare_case`/`_torch_ref` do not build, so candidates would only fail the
accuracy gate.

Constraint interactions:

- `BN64` ⇒ `BM32`, non-inline, separated gate/up.
- `num_waves == 2` ⇒ effective `BN64`.
- `k_wave > 1` ⇒ `BM32`, non-inline, `num_waves * k_wave <= 8`, and `D_HIDDEN/BK`
  divisible by `k_wave`. (`k_wave=4` is unreachable on kimi-k3: `3584/256 = 14`.)
- **`BM16` is the only inline-quant (`_f16in`) variant**, therefore the only one
  that can carry `_hpf`, and the only one reading raw bf16 `hidden_states`.

Legal pairs per shape run 1248 (kimi-k3) to 2688 (glm5). A token-2 kimi-k3 sweep
took **~50 min for 989 timed candidates** on one GPU, so an exhaustive model CSV
is hours to days. Budget accordingly, or pre-select (§6).

## 2. Which axes actually pay

**`block_m`** dominates, and is the axis a slate must never collapse on.

Track routed rows per expert (`token*topk/expert`). The shipped dispatch
heuristic is only a *prior*, and on kimi-k3 it is wrong at both ends: it returns
32 at token 256/512 where the tuned config is 16, and 128 at token >= 4096 where
the tuned config is 64. Measured tuned block_m (NE=896, topk=16):

| token | 2-512 | 1024 | 2048+ |
|---|---|---|---|
| block_m | 16 | 32 | 64 |

Other configs differ — kimi-k2/glm5/qwen do ship block_m 128 at large tokens — so
do not hard-code a ceiling. **Cover the tiers instead of trusting the prior.**

**`prefetch_hidden` (`_hpf`)** hoists the next K-tile's hidden load out of the
inline-quant step. GEMM1 µs, kimi-k3 a4w4, vs the identical row without it:

| tok | 2 | 3 | 4 | 16 | 32 | 64 | 256 |
|---|---:|---:|---:|---:|---:|---:|---:|
| speedup | **1.20x** | 1.02x | 1.03x | 0.99x | 1.01x | 1.01x | 1.02x |

Worth trying at very small token counts, noise above ~16. Cosine error is
identical either way — a scheduling change, not a numerical one. Confirmed
independently on glm5, where exhaustive search picks `_hpf` for half the smallest
shapes.

**`BN`** has no reliable prior either. The intuitive "few routed rows -> BN128"
rule is *backwards* on kimi-k3: tokens 3-32 ship `BN256`, tokens 128/512 ship
`BN128` with `xcd2`. Stratify over it rather than predict it.

**`xcd_swizzle`** tends to matter once blocks greatly outnumber CUs, and appears
in a majority of kimi-k3's tuned rows. Do not spend the whole slate on `xcd=0`.

**`spart`** (`_sp801`/`_sp1601`) is worth **1.6-5.7%** at large tokens — each
shipped kimi-k3 `_sp` config against its byte-identical no-`_sp` twin, three
repeats: 0.943 / 0.984 / 0.984 / 0.979 at tokens 4096 / 8192 / 16384 / 32768.
**It is not tunable**: `get_flydsl_stage2_v2_kernels` enumerates no
spatial-partition names, so a bare name always resolves to the dispatcher default
`MXFP4_G2_SPART=402`. Adding an `sp` axis was considered and **declined** — it
triples the GEMM2 space. Consequence to expect: a kimi-k3 re-tune replaces those
four rows with non-`_sp` equivalents that are 1.6-5.7% slower. That is not a
regression; do not re-open the axis without a fresh decision.

### GEMM2: layout family only

**a4w4 pairs `flydsl_mxmoe_g1_*` with `flydsl_moe2_layout_*`, and nothing else.**
The native `flydsl_mxmoe_g2_a4w4_*` family is not a candidate: its BK=256
contraction requires `D_INTER % 256 == 0`, so it cannot serve `inter_dim` such as
384 at all, and the layout family spans the same tile space while carrying the
`sort_block_m` contract the port's intermediate needs. Some older shipped rows in
kimik2/glm5/qwen still name `mxmoe_g2`; re-tuning migrates them.

`tile_n` should divide `model_dim`, `tile_k` should divide `inter_dim`. `atomic`
epilogs avoid a separate reduction; `reduce` epilogs need one and are the only
ones `AITER_FLYDSL_STAGE2_FP8` affects.

Also not tunable here: **`flydsl_moe1_*` GEMM1** (kimi-k3 tokens 1 and 8) is the
generic non-port flydsl family, dispatched by `fused_moe.py:2905` on the bare
`flydsl_` prefix and tuned by a different path. Re-tuning such a row migrates it
into the port.

## 3. Env vars you must set, or you benchmark the wrong thing

**`AITER_SITUV2_A4W4=1`** — required for any SiTUv2 config (kimik3_a4w4).
Without it `fused_moe` picks `q_dtype_a=bf16` for per_1x32 SiTUv2 and dispatches
`gemm1_a16w4_port_*`; the a4w4 rows are never exercised and `kernelName1/2` are
ignored entirely. The failure is silent — plausible timings for a different
kernel family.

**`AITER_FLYDSL_STAGE2_FP8=1`** — the stage2 intermediate kimik3_a4w4 ships.
Only reaches `flydsl_moe2_layout_*` rows with a **`reduce`** epilog
(`_flydsl_v2_stage2_wrapper`), i.e. 5 of kimi-k3's 17 rows. Omitting it makes
stage2 emit a bf16 intermediate instead of fp8 and moved one kimi-k3 row by
**1.9%** — enough to invert a verdict. This bit us: a "confirmed regression" at
token 2048 was entirely this.

Silu configs (kimik2, glm5, qwen) reach a4w4 by default; neither flag applies.

## 4. Accuracy gates

- For fp4, `checkAllclose` "failed!" / high mismatch fractions are **meaningless**
  — they fire on correct output. Judge on `cosine_diff` (tuner) or `logits_diff`
  (`run_config`). Healthy ~0.011 cosine / ~5e-4 logits; broken is O(0.1–1).
- **NaN must be rejected explicitly.** `float('nan') > errRatio` is `False`, so a
  candidate producing garbage passes a naive gate and, being fast, wins the sweep.
- SiTUv2 must be tuned *as* SiTUv2: the name needs `_situv2` and both
  `situ_beta`/`situ_linear_beta` must reach the kernel, or the reference computes
  a different activation and every shape reports `logits_diff ≈ 0.14` with
  `out_norm/ref_norm ≈ 0.61` — a constant ratio, the signature of a scale
  disagreement rather than a bad kernel.

## 5. Measurement protocol

**`run_config`'s `Kernel(us)` column is not a measurement.** It echoes the input
CSV's recorded `us`, so comparing two arms on it compares what each CSV already
claimed — a circular result that looks like a benchmark. The tell is that repeated
runs come back bit-identical (`37.0 / 37.0 / 37.0`). `E2E(us)` in the same table
*is* measured live, but includes host time and is destroyed by CPU contention.
For per-kernel numbers use rocprofv3 (below), and note that a call-count filter
does not separate the torch reference: `at::native*`, `rocprim*`, `Cijk_*`
(hipBLASLt) and `fillBufferAligned` loop too. Filter by kernel role name instead.

**Never diff recorded `us1`/`us2` across CSVs.** Those columns come from whichever
harness produced them and are not comparable. Re-measure both arms in one
harness. Observed: a row whose CSV claimed a 27 µs GEMM2 win measured 1.3 µs.

**The single-shot noise floor is ±3%** at these sizes — wide enough to invert a
2% result. Any row within ~1% of parity must be re-run **three times in
alternating order** (`main, pr, main, pr, …`) and judged on the median. Doing
this took seven apparent regressions down to one. Do not call a sub-1% gap
"noise" without running the repeats: of two such gaps, one was noise (0.998,
spread 0.6%) and the other was a reproducible 1.5% loss.

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

**Judge on throughput, not on kernel names.** A slate that misses the shipped
GEMM1 name can still win end to end (measured: kimi-k3 token 4, a name miss that
ran **1.020**). Name matching is a proxy; µs is the target.

## 6. Choosing candidates (for `recommend_mxfp4_candidates.py --policy`)

You are picking what gets **benchmarked**, not guessing a winner. Spend the budget
on candidates that are individually plausible and collectively diverse.

1. **Stratify, do not sort.** Allocate the budget round-robin across every GEMM1
   axis the kernel name encodes — `block_m`, `BN`, `xcd_swizzle`,
   `prefetch_hidden`, `k_wave`, `use_nt`, `num_waves` — crossed with the GEMM2
   epilog. Completeness is the point: each stratum was added only after a shipped
   config was *measured* to be hiding behind it, none speculatively. Budget ~96
   candidates (~1.3x the ~72 cells) so every legal tier is represented.

   A global sort keyed on distance-from-prior is **not safe**: with thousands of
   legal pairs against even a 256-candidate budget it filled the whole prompt with
   one `block_m` — on kimi-k3 tokens 4096/8192/32768 that was 128 while the tuned
   config is 64, unpickable at any temperature. A wrong prior must cost ordering,
   never coverage.
2. **Budget for two dimensions.** A slate is a `(GEMM1, GEMM2)` product. With
   `top_k` 8 and four block_m tiers each GEMM1 gets one GEMM2 and the pairing is
   effectively unsearched — measured: on kimi-k3 tokens 256 and 4096-32768 the
   slate held the *right* GEMM1 and still lost on its GEMM2 partner.

   For a fixed GEMM1 the layout family offers only ~16 partners, one per stratum,
   so stratifying GEMM2 degenerates to ranking and a top-3 take still missed a
   1.5% win. **`--g2-per-g1 0`** (every legal partner) is the reliable setting.
3. **Use `top_k` 64.** Once the prompt is a proper cover, `top_k` is the only
   remaining lever, and the failures it causes are the model declining to pick a
   config it *was* shown. Measured on glm5 against exhaustive per-shape optima
   (2688 pairs each, tokens 1-32768):

   | `top_k` | slate holds the true optimum | distinct GEMM1 offered |
   |---|---:|---:|
   | 32 | 41/64 | 31 |
   | **64** | **62/64** | 63 |

   Every miss at both budgets was present in the prompt and simply not chosen, so
   this is a selection budget, not coverage — including the two that survive at
   64 (`inter` 2048 tok 256, `inter` 1024 tok 512). 64 is ~0.9x the ~72 strata;
   at `top_k >= strata` containment becomes exact by construction, with the model
   no longer selecting anything. Below 32 it degrades gracefully — a knob, not a
   cliff.
4. At `token <= 8` on an inline-quant shape, always include an `_hpf` candidate
   **and its non-`hpf` twin** — that is where the 1.20x lives and the pair makes
   the effect attributable.
5. Vary one axis at a time across the slate so the result is interpretable.
6. Include at least one `atomic` and one `reduce` GEMM2 when both are legal. All
   GEMM2 candidates are `flydsl_moe2_layout_*`; never propose `flydsl_mxmoe_g2_*`.
7. Cover both `BN` values and both `xcd_swizzle != 0` options; neither has a
   trustworthy prior (§2).
8. **Stratify `k_wave` too.** Across glm5/kimi-k2/qwen, 7 of 9 GEMM1 misses were
   `_kw2`; adding the stratum recovered 8 of 9. kimi-k3 could not have revealed
   this — its k_wave space is thin (§1). Never generalise an axis prior from one
   model config.

## 7. Known blind spot

The tuner ranks GEMM2 by its **isolated `us2`** and cannot see a cost a GEMM2
imposes on another kernel. A GEMM2 faster on its own stage can still lose end to
end through L2 displacement of the following GEMM1. If a tuned row's recorded
`us1 + us2` improves but e2e does not, suspect this and check with the 2×2 in §5.

Bounded on glm5: an exhaustive 64-shape sweep (2688 pairs each) beat the shipped
config by a geomean of **1.039** end to end — 37 wins, 19 parity, 8 losses — but
two shapes went the *other* way by 10-13% (`inter_dim` 512 at tokens 16384 and
32768: 0.904 and 0.874, own-rep spread 2.9%/1.6%, so real). Those configs won the
tuner's own `us1 + us2` ranking and still lost end to end, which is the blind spot
biting. Treat it as a ~10% downside risk on the largest shapes, and re-check a
sweep winner end to end before shipping it there.

> The `nt` component specifically is still unquantified: an earlier attempt was
> run without `AITER_FLYDSL_STAGE2_FP8=1` and is void.

## 8. Wiring

```bash
# CPU-only: recommend candidates into a CSV
python csrc/ck_gemm_moe_2stages_codegen/recommend_mxfp4_candidates.py \
  -i aiter/configs/model_configs/kimik3_a4w4_untuned_fmoe.csv \
  -o /tmp/cand.csv --gfx gfx950 --cu-num 256 --top-k 64 --g2-per-g1 0 \
  --policy .claude/skills/mxfp4-tuning-policy/SKILL.md

# GPU: tune only those candidates
AITER_SITUV2_A4W4=1 AITER_FLYDSL_STAGE2_FP8=1 \
python csrc/ck_gemm_moe_2stages_codegen/gemm_moe_tune.py --mxfp4-flydsl \
  -i <untuned.csv> -o <tuned.csv> --candidate-csv /tmp/cand.csv
```

`--gfx`/`--cu-num` are required because the tuner injects the *runtime* arch into
rows (`aiter/utility/base_tuner.py:369`); the recommender may run on a different
host, and the tuner errors if the CSV's arch does not match.

Related: `aiter-config-shape` (landing the tuned CSV without duplicate shapes).
