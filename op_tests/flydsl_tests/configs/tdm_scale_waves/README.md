# TDM scale waves: test configs (M256 shape)

Configs for testing `AITER_FLYDSL_TDM_SCALE_WAVES` in
`aiter/ops/flydsl/kernels/mxfp4_preshuffle_gfx1250_tdm.py` on gfx1250.

Shape: one MoE expert, 256 tokens, topk 1, model_dim 7168, inter_dim 32768, a4w4 (MXFP4
activations and weights), SiLU. gemm1 is M 256 x N 65536 x K 7168. AITER has no tuned row
for this shape, so these one-row configs pin the kernel: tile 256x256x256, 2x2 waves, 4
buffers, cluster 1x4.

| File | waves_per_tensor_tdm (gemm1, gemm2) |
|---|---|
| `cfg_t256_e1_i32768_wpt1.csv` | 1, 1 (the tuned row) |
| `cfg_t256_e1_i32768_wpt4.csv` | 4, 4 (same row otherwise) |

`waves_per_tensor_tdm` comes from the config. The env `AITER_FLYDSL_NUM_WAVES_PER_TENSOR_TDM`
applies only when the config leaves it unset.

## Run

From the repository root, with AITER installed from this branch (`ENABLE_CK=0` is enough;
the FlyDSL kernel is compiled on first use) and FlyDSL 0.4.0 (the version it pins):

```bash
CFG=op_tests/flydsl_tests/configs/tdm_scale_waves
TEST=op_tests/flydsl_tests/test_flydsl_grouped_gemm.py
export AITER_LOG_MORE=1          # prints the per-kernel timing table with the kernel names
ARGS="--scenario kernel --data-format a4w4 --experts 1 --tokens 256 --topk 1 --model-dim 7168
      --inter-dim 32768 --act silu --no-bias --no-check-aot-cache --data-init uniform --scale-init auto"

# baseline: waves_per_tensor_tdm=1
AITER_CONFIG_GROUPED_FMOE=$CFG/cfg_t256_e1_i32768_wpt1.csv python $TEST $ARGS

# A and B loaded by all four waves, A-scale by waves 0-1, B-scale by waves 2-3
AITER_CONFIG_GROUPED_FMOE=$CFG/cfg_t256_e1_i32768_wpt4.csv AITER_FLYDSL_TDM_SCALE_WAVES=2 \
    python $TEST $ARGS

# stock waves_per_tensor_tdm=4 (fails the check: 128 of the 256 output rows are wrong)
AITER_CONFIG_GROUPED_FMOE=$CFG/cfg_t256_e1_i32768_wpt4.csv python $TEST $ARGS
```

What to look for:
- Kernel names (in the profiler table) contain `_wpt1_`, `_wpt4_sw2_` and `_wpt4_` respectively.
- `[sanity a4w4 silu] kernel: logits_diff=... rel_l2=... (gate<0.01, ...)`: the test compares
  with its torch reference and raises `AssertionError` (exit status 1) above the gate.
- `[kernel-bench a4w4 silu] gemm1: us = ...` and `gemm2: us = ...`.

## Results (MI450X, 2026-10-27)

| Config | Check | gemm1 us | gemm2 us |
|---|---|---|---|
| wpt1 | logits_diff 0 | 47.31 | 66.86 |
| wpt4 + `AITER_FLYDSL_TDM_SCALE_WAVES=2` | logits_diff 0, output bit-identical to wpt1 | 47.67 | 71.23 |
| wpt4 (stock) | fails: 128 of 256 rows wrong | - | - |

Medians of 3 runs of the test's kernel-bench.
