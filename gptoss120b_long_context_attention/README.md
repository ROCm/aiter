# gpt-oss-120B long-context prefill attention on MI455X (gfx1250) — work record

Backup of an optimization of the AITER gfx1250 Gluon prefill attention kernel for gpt-oss-120B at
ISL/OSL 70000/300 (SGLang, TP1). **This folder is a record only: nothing in it is wired into this branch's
AITER sources.**

## Base

All changes and measurements are against **`akao-amd/aiter` branch `draft-base1111`, commit `33b7a58`**
(with `akao-amd/sglang` `draft-base1111`). The patches in `patches/` apply cleanly there
(`git apply patches/0*.patch` at the aiter repo root).

This branch (`opt_gptoss120b_long_context_attention`) is based on upstream `ROCm/aiter` main instead. The
three AITER files differ heavily from `draft-base1111` there, and its variant 2 casts P to fp8 *without* the
x256 scale that `LOOP_VARIANT 3` below folds away. So the kernel change would not give the same speedup on
upstream; the config-table idea still applies, but needs its own measurement there.

Server setup measured: `--kv-cache-dtype fp8_e4m3`, `SGLANG_6003_PREFILL_Q=static` (fp8 query and fp8 KV
for prefill), `--page-size 64`, `--chunked-prefill-size 16384`, `--enable-mixed-chunk`, attention backend
`aiter`, AITER selected via `PYTHONPATH=$HOME/draftbase_1111/aiter`.

## Why

70000/300 CC=4, TP1: MI455X 160.1 output tok/s against GB300 237.3. Profiling showed ~85% of the gap is
prefill: full-attention `_unified_attention_gluon_kernel_2d` ~1.9x slower per call than GB300's
`fmhaSm103a` (chunk 3: ~17 ms against ~9.2 ms), dense GEMM ~2.3x slower. Decode is faster on MI455X.

## The change (3 files)

| File | Change |
|---|---|
| `aiter/ops/triton/configs/gfx1250/gluon/attention/unified_attention/DEFAULT.json` | 4 entries for a prefill chunk of >= 9216 tokens at head dim <= 64. fp8 full attention: `BLOCK_M 256, LOOP_VARIANT 3, NUM_BUFFERS 3` (was the generic `DT_fp8_fp8`: 128 / 2 / 2). 1K/1K and 8K/1K (chunks <= 8192) keep their original entries |
| `aiter/ops/triton/_gluon_kernels/gfx1250/attention/unified_attention_2d.py` | New `LOOP_VARIANT 3` (`attention_loop_fp8_folded`): variant 2 with the fp8 P x256 scale folded into the exponent (`exp2(x + 8) = 256 * exp2(x)`, exact), removing one multiply per score. Variants 0-2 untouched |
| `aiter/ops/triton/attention/unified_attention.py` | Two launcher guards (`BLOCK_M >= 16 * num_warps`; loop variant 0 forces 2 buffers) |

No environment variable (the AITER team does not accept new env knobs). No rebuild: pure Python/JSON, Triton
compiles the kernel on first use. `scripts/apply.sh` installs / `--restore`s on the server.

## Measurements (kernel benchmark, `scripts/bench_prefill_attention.py`, fp8, 16k chunk + 3 decode rows)

Per full-attention call, ms (same session; run-to-run noise <= 0.5%):

| KV prefix | Original (BLOCK_M 128, LV2) | BLOCK_M 256, LV2 | BLOCK_M 256, LV3, 2 buf | **BLOCK_M 256, LV3, 3 buf** |
|---|---|---|---|---|
| 0 | 3.61 | 3.48 | 3.14 | **2.90** |
| 16384 | 9.98 | 9.42 | 8.42 | **7.84** |
| 32768 | 16.40 | 15.32 | 13.71 | **12.66** |
| 49152 | 22.87 | 21.30 | 19.09 | **17.74** |

Relative error vs an fp32 reference is identical for all of them (`rel_err` 5.4e-2 / 1.5e-2 / 1.1e-2 / 9.6e-3;
the prefix-0 value is the fp8 rounding of P on rows that see 1-2 keys). GB300 at prefix 32768: ~9.2 ms.

End to end (SGLang serving benchmark, 70000/300 CC=4, one run on a shared node, not a same-session A/B):
169.5 output tok/s against 160.1 in the earlier baseline (+5.9%). Needs a clean same-machine A/B.

## What was tried and dropped

- **Ablation** (`scripts/make_ablations.py`, `ablate.sh`): removing the score exps saves 13.5%, removing the
  per-tile accumulator rescale 14.6%, both 27% -> the loop is softmax / vector-ALU bound.
- **Lazy rescale** (skip `acc *= alpha` unless a row max grows by > 0.75 log2): correct, but the block-uniform
  decision (cross-warp max + branch around the accumulators) made it **2.3x slower**; dropped
  (`scripts/make_lv3_diag.py`, `test_lv3_diag.sh` split it from the fold).
- **SGLang MIXED-batch decode split** (`patches/90_DROPPED_sglang_mixed_decode_split.patch`): run the 1-token
  decode rows of a mixed batch through the decode route. Correct, but only 0.05-0.12 ms per call (~0.5% end to
  end); dropped.
- Env-variable override for prefill configs: replaced by table entries (no env knobs upstream).

## Next ideas

- Cheaper exp (exp is ~13.5% of the loop; P now goes straight from exp to the fp8 cast).
- Config sweep around LV3 (`bench_prefill_attention.py --fp8 --sweep --prefix 32768`; the grid includes LV3).
- Dense GEMM (`_gemm_a16w16_gfx1250_compute_bound`, 256x256x64) is the other ~15% of the prefill gap.

## Layout

```
changed/aiter/    the 3 changed files (server copy names)
original/aiter/   the draft-base1111 originals they were made from
patches/          01-03: the change as git patches (repo paths); 90: the dropped SGLang experiment
scripts/          apply.sh (install/restore), bench_prefill_attention.py (kernel benchmark, accuracy check,
                  --fp8, --sweep, --kernel-file), bench.sh, test_lv3*.sh, make_*.py, ablate.sh, copy.sh
```
