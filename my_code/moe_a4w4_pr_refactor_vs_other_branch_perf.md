============== baseline summary ==============
# Baseline E64/T1536/topk8 benchmark

- branch: `hyg/moe_a4w4_pr_refactor`
- commit: `792c83f280125256bab7533f2413c68e1f7b44ed`
- MoE e2e timing: `testGraph=False`, `use_cuda_event=False`,
  `num_warmup=5`, `num_iters=20`, torch profiler
  `get_trace_perf(...).device_time_sum`
- repository: `/app/aiter`

## Python command

```bash
cd /app/aiter
ENABLE_CK=0 \
  AITER_MOE_EXPERT_BALANCE=true \
  AITER_LOG_MORE=1 \
  AITER_USE_GROUPED_GEMM=1 \
  AITER_GROUPED_DEBUG=0 \
  AITER_FLYDSL_MOE_EXPERT_SCHEDULING_MODE=1 \
  AITER_FLYDSL_GEMM1_FUSED_QUANT=1 \
  python3 -u my_code/test_flydsl_grouped_gemm_gfx1250.py --scenario bench --data-format a4w4 --act silu --no-bias --no-check-aot-cache --experts 64 --tokens 1536 --topk 8 --model-dim 7168 --inter-dim 2048 --iters 20 --const-init 0
```

AITER_FLYDSL_GEMM1_FUSED_QUANT=1 (pipeline=fused-quant)

| data | shape | mode | commit | GEMM1 samples (us) | GEMM1 median us | standalone quant samples (us) | GEMM1 + quant median us | GEMM1 pipeline vs baseline | GEMM1 TFLOP/s | GEMM1 effective R+W (TB/s) | GEMM1 ref out hash128 | GEMM1 out hash128 | GEMM2 samples (us) | GEMM2 median us | GEMM2 vs baseline | GEMM2 TFLOP/s | GEMM2 effective R+W (TB/s) | GEMM2 ref out hash128 | GEMM2 out hash128 | MoE e2e samples (us) | MoE e2e median us | MoE e2e vs baseline | pass | logits_diff | rel_l2 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| const0 | E64/T1536/topk8/M7168/I2048 | current-fused | 792c83f28012 | 97.197, 98.789, 103.628 | 98.789 | 0.000, 0.000, 0.000 | 98.789 | N/A | 7304.0 | 10.714 | c281c06c980fd4ca89d84615b26083b9 | c281c06c980fd4ca89d84615b26083b9 | 69.242, 65.687, 67.034 | 67.034 | N/A | 5382.0 | 10.273 | 8435f663d2aae0fe93d109c485cd9265 | 8435f663d2aae0fe93d109c485cd9265 | 229.51, 228.22, 228.89 | 228.89 | N/A | True | 0 | 0 |
================================================

================ yadai summary ================
# a4w4_prefill_v2_yadai E64/T1536/topk8 benchmark

- branch: `ROCm/dev/a4w4_prefill_v2_yadai`
- commit: `ef460aa109aee02e89952c389584a77f88d1e73f`
- MoE e2e timing: `testGraph=False`, `use_cuda_event=False`, `num_warmup=5`, `num_iters=20`, torch profiler `get_trace_perf(...).device_time_sum`

## GEMM kernel Python command

```bash
cd /data/yanguahe/code/wk_sp1/aiter_a4w4_prefill_v2_yadai
ENABLE_CK=0 \
  AITER_MOE_EXPERT_BALANCE=true \
  AITER_LOG_MORE=1 \
  AITER_USE_GROUPED_GEMM=1 \
  AITER_GROUPED_DEBUG=0 \
  AITER_FLYDSL_MOE_EXPERT_SCHEDULING_MODE=1 \
  AITER_META_DIR=/data/yanguahe/code/wk_sp1/aiter_a4w4_prefill_v2_yadai \
  PYTHONPATH=/data/yanguahe/code/wk_sp1/aiter_a4w4_prefill_v2_yadai \
  python3 -u op_tests/flydsl_tests/test_flydsl_grouped_gemm.py --scenario kernel --data-format a4w4 --experts 64 --tokens 1536 --topk 8 --model-dim 7168 --inter-dim 2048 --act silu --no-bias --no-check-aot-cache --warmup 5 --iters 20 --data-init zero --scale-init zero
```

## MoE e2e Python command

```bash
cd /data/yanguahe/code/wk_sp1/aiter_a4w4_prefill_v2_yadai
ENABLE_CK=0 \
  AITER_MOE_EXPERT_BALANCE=true \
  AITER_LOG_MORE=1 \
  AITER_USE_GROUPED_GEMM=1 \
  AITER_GROUPED_DEBUG=0 \
  AITER_FLYDSL_MOE_EXPERT_SCHEDULING_MODE=1 \
  AITER_META_DIR=/data/yanguahe/code/wk_sp1/aiter_a4w4_prefill_v2_yadai \
  PYTHONPATH=/data/yanguahe/code/wk_sp1/aiter_a4w4_prefill_v2_yadai \
  python3 -u /app/aiter/my_code/moe_prefill_yadai_compare_runs/20261002T041135Z/a4w4_prefill_v2_yadai/run_e2e_graph_false.py
```

| Metric | Samples (us) | Median (us) |
|---|---|---:|
| GEMM1 | 83.390, 83.670, 81.890 | 83.390 |
| GEMM2 | 59.540, 60.150, 60.040 | 60.040 |
| MoE e2e | 278.649, 279.353, 273.890 | 278.649 |

| Round | logits_diff | rel_l2 | pass |
|---:|---:|---:|:---:|
| 1 | 0.0000e+00 | 0.0000e+00 | True |
| 2 | 0.0000e+00 | 0.0000e+00 | True |
| 3 | 0.0000e+00 | 0.0000e+00 | True |
================================================
Baseline summary: /app/aiter/my_code/moe_prefill_yadai_compare_runs/20261002T041135Z/baseline/summary.md
Yadai summary: /app/aiter/my_code/moe_prefill_yadai_compare_runs/20261002T041135Z/a4w4_prefill_v2_yadai/summary.md
All logs: /app/aiter/my_code/moe_prefill_yadai_compare_runs/20261002T041135Z
