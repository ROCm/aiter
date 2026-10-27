# TDM scale waves: test configs (DSR1 prefill shape)

Configs for testing `AITER_FLYDSL_TDM_SCALE_WAVES` in
`aiter/ops/flydsl/kernels/mxfp4_preshuffle_gfx1250_tdm.py` on gfx1250.

Shape: 64 MoE experts, 1536 tokens, topk 8, model_dim 7168, inter_dim 2048, a4w4 (MXFP4
activations and weights), SiLU, balanced routing (192 rows per expert). gemm1 is
64 x (M 192 x N 4096 x K 7168), gemm2 64 x (M 192 x N 7168 x K 2048). The config lookup
rounds 1536 tokens up to the 2048 bucket, so both files are the 2048-token row of
`aiter/configs/tuned_grouped_fmoe.csv`:

| File | waves_per_tensor_tdm (gemm1, gemm2) |
|---|---|
| `cfg_t2048_e64_i2048_wpt1.csv` | 1, 1 (the tuned row, same kernel as the default config) |
| `cfg_t2048_e64_i2048_wpt4.csv` | 4, 4 (same row otherwise) |

`waves_per_tensor_tdm` comes from the config. The env `AITER_FLYDSL_NUM_WAVES_PER_TENSOR_TDM`
applies only when the config leaves it unset.

## Run

From the repository root, with AITER installed from this branch (`ENABLE_CK=0` is enough;
the FlyDSL kernels are compiled on first use) and the FlyDSL wheel in the repository root
(`flydsl-0.3.3.dev882-cp312-cp312-linux_x86_64.whl`, which `run_dsr1.sh` installs; the
results below used it, while `requirements.txt` pins 0.3.4.1). That FlyDSL finds `ld.lld`
only through `ROCM_PATH`, which ROCm containers set; set it yourself elsewhere.

```bash
CFG=op_tests/flydsl_tests/configs/tdm_scale_waves
TEST=op_tests/flydsl_tests/test_flydsl_grouped_gemm.py
export ROCM_PATH=${ROCM_PATH:-/opt/rocm}
export AITER_USE_GROUPED_GEMM=1 AITER_MOE_EXPERT_BALANCE=true AITER_FLYDSL_MOE_EXPERT_SCHEDULING_MODE=1 \
       AITER_GROUPED_A4W4_A_PRESHUFFLE=1 AITER_FLYDSL_EXPLICIT_VGPR_PARTITION=1
export AITER_LOG_MORE=1          # prints the per-kernel timing table with the kernel names
ARGS="--scenario kernel --data-format a4w4 --experts 64 --tokens 1536 --topk 8 --model-dim 7168
      --inter-dim 2048 --act silu --no-bias --iters 128 --no-check-aot-cache --data-init uniform
      --scale-init uniform"

# baseline: waves_per_tensor_tdm=1
AITER_CONFIG_GROUPED_FMOE=$CFG/cfg_t2048_e64_i2048_wpt1.csv python $TEST $ARGS

# A and B loaded by all four waves, A-scale by waves 0-1, B-scale by waves 2-3
AITER_CONFIG_GROUPED_FMOE=$CFG/cfg_t2048_e64_i2048_wpt4.csv AITER_FLYDSL_TDM_SCALE_WAVES=2 \
    python $TEST $ARGS

# stock waves_per_tensor_tdm=4 (fails the check: about 60% of the output is NaN)
AITER_CONFIG_GROUPED_FMOE=$CFG/cfg_t2048_e64_i2048_wpt4.csv python $TEST $ARGS
```

What to look for:
- Kernel names (in the profiler table) contain `_wpt1_`, `_wpt4_sw2_` and `_wpt4_` respectively.
- `[sanity a4w4 silu] kernel: logits_diff=... rel_l2=... (gate<0.01, ...)`: the test compares
  with its torch reference and raises `AssertionError` (exit status 1) above the gate.
- `[kernel-bench a4w4 silu] gemm1: us = ...` and `gemm2: us = ...`.

## Results (MI450X, 2026-10-27)

| Config | Check | gemm1 us | gemm2 us |
|---|---|---|---|
| wpt1 | logits_diff 3.4249e-06 | 83.91 | 58.39 |
| wpt4 + `AITER_FLYDSL_TDM_SCALE_WAVES=2` | same, output bit-identical to wpt1 | 83.52 | 57.08 |
| wpt4 (stock) | fails: about 60% NaN | - | - |

Medians of 3 runs of the test's kernel-bench.
