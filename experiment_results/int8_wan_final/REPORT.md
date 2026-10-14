# gfx950 INT8 bpreshuffle GEMM: Wan readiness evidence

Branch `experiment/gfx950-int8-gemm`. Kernel: FlyDSL true-INT8 8-wave
`flydsl_bpreshuffle_8w_256x256x128_I8_I8_B16_1x8` (tuned id 1000062).
Hardware: MI355X, gfx950, 256 CUs, one idle GPU (`rocm-smi` 0% before the run).

## Frozen ABI (`aiter.gemm_a8w8_bpreshuffle`, INT8)

- `XQ`, `WQ`: contiguous signed `torch.int8`, same dtype; `WQ` is `(N, K)` in
  `shuffle_weight(..., layout=(16, 16))` order.
- `x_scale`: FP32, M elements (per row). `w_scale`: FP32, N elements (per column).
- Output: BF16 only. FP16 output is rejected before dispatch.
- `K % 128 == 0`, `K >= 256`. Bias is not fused (xDiT adds it after the GEMM).
- INT32 accumulation, then FP32 row/column scale, then BF16 store.
- gfx950 only. Untuned INT8 shapes use the same 8-wave kernel at its default
  256x256 tile (logged warning). Other archs raise `NotImplementedError`.
  The CK/CKTile bpreshuffle fallbacks are FP8-only; before this change an
  untuned INT8 call reached CK and returned NaN.

## Frozen shape matrix

Required, tuned in `aiter/configs/a8w8_bpreshuffle_tuned_gemm.csv`
(Wan 14B, 75,600 tokens at SP degree 8, hidden 5120, FFN 13824):

| M | N | K |
|---|---|---|
| 9450 | 5120 | 5120 |
| 9450 | 13824 | 5120 |
| 9450 | 5120 | 13824 |

Decision for the other SP/A2A M values: not tuned, supported through the
default-kernel fallback, validated for correctness only. Measured through the
public API (includes Python/launch overhead, so slower than the kernel-only
numbers below), bit-exact to Gluon in
`test_public_untuned_wan_m_uses_correct_fallback` for N=K=5120:

| M | Role | us (public API) | TOPS |
|---|---|---|---|
| 512 | cross-attention K/V | 49.6 | 542 |
| 9472 | packed A2A padding at SP8 | 195.6 | 2539 |
| 18900 | SP4 | 387.7 | 2556 |
| 37800 | SP2 | 795.0 | 2493 |
| 75600 | SP1 | 1532.6 | 2586 |

Small M (cross-attention, M=512) runs at 542-1389 TOPS across the Wan N/K
combinations (542 and 614 TOPS for the two 5120-wide outputs) with the 256x256
tile, which is low utilisation. It is correct but unoptimised;
keep it on BF16/other paths in xDiT if that matters. Flux shapes are not tuned
for the 8-wave kernel.

## Public-path tests (`op_tests/test_flydsl_int8_8wave.py`)

- All three Wan rows go through `aiter.gemm_a8w8_bpreshuffle`, select the tuned
  kernel, and are bit-exact to upstream Gluon; repeated launches are bit-stable.
- Untuned M fallback is bit-exact to Gluon.
- Rejections: FP16/FP32 output, bias, FP16 scales, wrong scale size, noncontiguous
  A or B, 1-D input, FP8/INT8 mix, unsupported K, mixed devices, non-gfx950.
- `torch.compile(fullgraph=True)` matches eager; CUDA-graph capture/replay is
  bit-exact with a stable output address and new inputs copied in place; a
  non-default stream executes the GEMM on the current stream.
- Result: 34 passed (this file + `op_tests/tuning_tests/test_flydsl_a8w8_8wave.py`),
  with two visible GPUs so the mixed-device test runs.

## Clean AOT deployment

1. `git archive HEAD` plus the working changes into a clean tree.
2. `pip wheel` (pure Python, `PREBUILD_KERNELS=0`), `pip install --target` into
   an empty directory.
3. `python -m aiter.aot.flydsl.gemm --csv int8_rows.csv` with
   `GPU_ARCHS=gfx950`, writing to the installed
   `aiter/jit/flydsl_cache` (empty before): 3 jobs compiled, 0 failed (2 unique
   cache entries, since the cache keys on K).
4. Fresh process, installed package only (repo not on the path),
   `FLYDSL_RUNTIME_RUN_ONLY=1`, no cache env override. All three Wan rows run
   without JIT; the negative control (K=384, not compiled) fails with
   `FLYDSL_RUNTIME_RUN_ONLY=1 but no usable AOT cache`, which shows run-only
   mode is active. See `run_only_probe.py`.
5. The public-path tests for the Wan rows, untuned M, compile, graph and stream
   (12 tests) pass in the same run-only environment (`run_only_pytest.log`).

Caveat: the full CK-prebuild wheel (`PREBUILD_KERNELS=1`) was not built; the
FlyDSL GEMM AOT step was exercised on the INT8 rows directly. The three INT8
rows are in the default `DEFAULT_CSVS` job collection (6836 jobs in total),
so the wheel build covers them. Shapes whose K differs from 5120/13824 still
JIT-compile on first use.

## Final paired benchmark

`bench_int8_gemm_gfx950.py --include-8wave --8wave-block-m 256 --8wave-block-n 256
--8wave-waves-per-eu 1 --8wave-xcd-swizzle 8 --include-asm --warmup 30
--iterations 200 --repeats 15 --confirmed-idle`, correctness on (the 8-wave
output is asserted bit-exact to Gluon; CK, Gluon, Gluon-preshuffle and ASM are
checked against each other). Raw samples: `int8_wan_final_256x256.csv`.

Median us over 15 repeats:

| M,N,K | flydsl_8wave | gluon | gluon_preshuffle | ck | asm_best | best other | speedup | repeats won |
|---|---|---|---|---|---|---|---|---|
| 9450,5120,5120 | 185.7 | 286.5 | 602.8 | 1189.2 | 309.2 (128x128) | gluon | 1.54x | 15/15 |
| 9450,13824,5120 | 494.9 | 773.6 | 1791.4 | 745.2 | 872.3 (160x128) | ck | 1.51x | 15/15 |
| 9450,5120,13824 | 459.6 | 674.8 | 1921.2 | 707.4 | 783.9 (128x128) | gluon | 1.47x | 15/15 |

"repeats won" means the FlyDSL sample was lower than every other backend in the
same paired repeat. FlyDSL spread is under 2.5% (min/max 183.0-187.0, 491.7-497.1,
457.1-463.1 us). The upstream gfx950 INT8 ASM kernels were tried for every
tile; the table shows the fastest correct tile per shape (weights in
`(16, 16)` layout, explicit zero bias, as the kernel requires).

## Validation run on the final tree

- `op_tests/test_flydsl_int8_8wave.py`, `op_tests/tuning_tests/test_flydsl_a8w8_8wave.py`: 34 passed.
- `op_tests/tuning_tests/test_csv_validation.py`, `test_config_shape_collision.py`:
  passed (41 tests plus 48 subtests, together with the 8-wave tuning test).
- `csrc/ck_gemm_a8w8_bpreshuffle/gen_instances.py` against the tuned CSV: exits 0
  and generates the CK instance set (the INT8 FlyDSL rows do not break codegen).
- Not run: `op_tests/tuning_tests/test_run_config.py`. It re-runs the CK tuner
  and rebuilds `module_gemm_a8w8_bpreshuffle` with every instance; I stopped it
  after about 25 minutes.

## Branch hygiene

- The triple-stage LDS experiment (`fc4b43f25c`) is reverted here and preserved on
  `experiment/gfx950-int8-triple-stage`. The 8-wave tuner dtype filter and its
  contract test were kept (`test_flydsl_a8w8_8wave.py`, renamed from the stage-3 file).
- Only the three validated gfx950 INT8 FlyDSL rows are in the tuned CSV.
- Out of scope, xDiT side: activation quantization, bias add, the INT8 backend
  and loader wiring.
