---
name: mxfp4-moe-benchmarking
description: How to run MXFP4 a4w4 MoE tuning and trust the numbers on gfx950 — the env vars without which you silently benchmark a different kernel family, the fp4 accuracy gates (cosine_diff/logits_diff, not checkAllclose), the ±3% noise floor and repeat protocol, per-stage rocprofv3 attribution, and the columns that only look like measurements. Use when running gemm_moe_tune.py/run_config, when comparing two tuned CSVs, or when a tuned row's recorded us does not reproduce end to end. For choosing which candidates to benchmark see mxfp4-tuning-policy.
argument-hint: [model config name, e.g. kimik3_a4w4, or a tuned_fmoe.csv path]
---

# MXFP4 a4w4 MoE benchmarking and measurement

Applies to the FlyDSL a4w4 *port* driven by
`csrc/ck_gemm_moe_2stages_codegen/gemm_moe_tune.py --mxfp4-flydsl`. Candidate
*selection* is a separate concern — see the `mxfp4-tuning-policy` skill.

Numbers were measured on gfx950 (MI355X).
## 1. Env vars you must set, or you benchmark the wrong thing

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

## 2. Accuracy gates

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

## 3. Measurement protocol

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

## 4. Known blind spot

The tuner ranks GEMM2 by its **isolated `us2`** and cannot see a cost a GEMM2
imposes on another kernel. A GEMM2 faster on its own stage can still lose end to
end through L2 displacement of the following GEMM1. If a tuned row's recorded
`us1 + us2` improves but e2e does not, suspect this and check with the 2×2 in §3.

Bounded on glm5: an exhaustive 64-shape sweep (2688 pairs each) beat the shipped
config by a geomean of **1.039** end to end — 37 wins, 19 parity, 8 losses — but
two shapes went the *other* way by 10-13% (`inter_dim` 512 at tokens 16384 and
32768: 0.904 and 0.874, own-rep spread 2.9%/1.6%, so real). Those configs won the
tuner's own `us1 + us2` ranking and still lost end to end, which is the blind spot
biting. Treat it as a ~10% downside risk on the largest shapes, and re-check a
sweep winner end to end before shipping it there.

> The `nt` component specifically is still unquantified: an earlier attempt was
> run without `AITER_FLYDSL_STAGE2_FP8=1` and is void.

## 5. Running a tuning sweep

```bash
AITER_SITUV2_A4W4=1 AITER_FLYDSL_STAGE2_FP8=1 \
python csrc/ck_gemm_moe_2stages_codegen/gemm_moe_tune.py --mxfp4-flydsl \
  -i <untuned.csv> -o <tuned.csv> --candidate-csv /tmp/cand.csv
```

On a shared box, pick a GPU with zero *foreign* KFD processes
(`rocm-smi --showpids`, then `--showuse`) and report per-cell rep spread; a cell
whose own identical reps disagree by >3% was contended and its number is void.
Never `SIGSTOP` a running tuner to free GPUs — it holds the JIT baton lock
(`aiter/jit/build/lock_module_moe_asm`) and every concurrent aiter process then
blocks forever on `waiting for baton release`.

Related: `mxfp4-tuning-policy` (which candidates to benchmark).
