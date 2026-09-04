---
name: mxfp4-tuning-policy
description: How to choose MXFP4 a4w4 MoE candidate kernels for a given shape on gfx950 — the legal GEMM1/GEMM2 search space and its constraint interactions, which axes actually pay and which have no trustworthy prior, and the stratification/top_k rules for building a candidate slate. Use when writing the --policy file for recommend_mxfp4_candidates.py, or when deciding which candidates a shape should be tuned over. For running and trusting the resulting numbers (env vars, accuracy gates, measurement protocol) see mxfp4-moe-benchmarking.
argument-hint: [model config name, e.g. kimik3_a4w4, or a tuned_fmoe.csv path]
---

# MXFP4 a4w4 MoE candidate-selection policy

How to turn one row of an untuned `*_fp4/_a4w4` fmoe CSV into a candidate slate
for the FlyDSL a4w4 *port* (`flydsl_mxmoe_g1_*` / `flydsl_moe2_layout_*`).

**Scope: choosing what to benchmark.** Running the sweep and trusting its output
— required env vars, accuracy gates, the noise floor and per-stage attribution —
is a separate concern; see the `mxfp4-moe-benchmarking` skill.

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
is hours to days. Budget accordingly, or pre-select (§3).

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

## 3. Choosing candidates (for `recommend_mxfp4_candidates.py --policy`)

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
3. **Use `top_k` 64, and do not expect more to help.** Once the prompt is a
   proper cover, `top_k` is the only remaining lever, and the failures it causes
   are the model declining to pick a config it *was* shown. Measured on glm5
   against exhaustive per-shape optima (all 64 shapes, 2688 pairs each):

   | `top_k` | best GEMM1 in slate | exact pair | pairs/shape | share of legal |
   |---|---:|---:|---:|---:|
   | 32 | 41/64 | 41/64 | 962 | 35% |
   | **64** | **62/64** | **61/64** | 1954 | 72% |
   | 80 | 58/64 | 57/64 | 2450 | 91% |
   | 96 (= prompt) | 64/64 | 64/64 | 2604 | 97% |

   **Containment is not monotonic in `top_k`.** 80 scores *worse* than 64, and its
   six misses are a different set of shapes — it recovers both shapes 64 missed
   and loses six others. The selection step is a model choosing a subset, not a
   ranking prefix, so a bigger budget reshuffles which configs get dropped rather
   than strictly adding. Do not tune `top_k` upward expecting a monotone climb.

   Full reachability exists but is degenerate: at `top_k` >= the prompt budget the
   model selects nothing, the slate *is* the pruned prompt, and containment is
   64/64 by construction — at 97% of the legal space, which is to say you have
   given up pre-selection and are back to an exhaustive sweep. 64 is the knee:
   97% of the optima for 72% of the space. Choose the last 2 shapes or the
   pruning, not both.

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
