# A07-3 MoE Serving Baseline 与 Optimized 对比

## 目录

- [测试目标与环境](#测试目标与环境)
- [完整执行命令](#完整执行命令)
- [结论摘要](#结论摘要)
- [Serving benchmark 性能](#serving-benchmark-性能)
- [Serving trace 中的 MoE token 数](#serving-trace-中的-moe-token-数)
- [Random MoE 单测方法](#random-moe-单测方法)
- [Decode 与 Prefill 总 kernel 耗时](#decode-与-prefill-总-kernel-耗时)
- [Kernel name 对比](#kernel-name-对比)
- [结果解释与限制](#结果解释与限制)
- [结果文件](#结果文件)
- [Baseline 原始 Serving 结果](#baseline-原始-serving-结果)
- [Optimized 原始 Serving 结果](#optimized-原始-serving-结果)

## 测试目标与环境

目标是确认 A4W4 MoE prefill 优化只影响大 token prefill shape，不改变 decode 和小 token prefill 的 kernel 路径，并评估其对 Serving 端到端性能的影响。

```text
Machine:     a07-3 / heliosr-1b114-a07-3
Container:   hyg_fyd_e2e
Model:       /data/models/DeepSeek-R1-0528-MXFP4
ISL / OSL:   1024 / 1024
Requests:    256
Concurrency: 64

baseline:    314ab7d48a077f3ccc055e7d353040c10a5f7fe5
optimized:   b899eaa23ebda8985440332fe3f40570633d2d9e
```

Serving 性能来自不带 profiler 的正式运行。MoE token shape 来自单独的 profiler run；profiler 下的 Serving 延迟和吞吐不作为性能结论。

## 完整执行命令

server 是阻塞运行的，因此 start server 与 run benchmark 需要在两个终端中分别执行。每次切换 AITER commit 前，需要先关闭前一个 ATOM server。

### 不带 profiler 的 Serving 性能

#### Baseline server

```bash
cd /app/aiter
git checkout 314ab7d48a077f3ccc055e7d353040c10a5f7fe5

cd /app/scripts/dsr1
FAKE_EPLB=1 \
AITER_FLYDSL_MOE_EXPERT_SCHEDULING_MODE=1 \
bash /app/scripts/dsr1/serve_dsr1fp4_a8w8_tp1.sh \
  /data/models/DeepSeek-R1-0528-MXFP4
```

#### Baseline benchmark

```bash
cd /app/ATOM/scripts/performance
bash /app/scripts/dsr1/bench_dsr1fp4.sh \
  /data/models/DeepSeek-R1-0528-MXFP4

cp result.txt \
  /app/aiter/my_code/a07_serving_result_baseline_314ab7d_20260929.txt
```

#### Optimized server

```bash
cd /app/aiter
git checkout b899eaa23ebda8985440332fe3f40570633d2d9e

cd /app/scripts/dsr1
FAKE_EPLB=1 \
AITER_USE_GROUPED_GEMM=1 \
AITER_GROUPED_DEBUG=0 \
ENABLE_CK=0 \
FLYDSL_DUMP_IR=0 \
AITER_LOG_MORE=1 \
AITER_MOE_EXPERT_BALANCE=true \
AITER_FLYDSL_MOE_EXPERT_SCHEDULING_MODE=1 \
AITER_FLYDSL_GEMM1_A_PRESHUFFLE=1 \
AITER_FLYDSL_GEMM1_WAVES_PER_TENSOR_TDM=2 \
AITER_FLYDSL_GEMM1_MMA_GROUP=4 \
AITER_FLYDSL_GEMM1_FENCE_COVER_MMA=28 \
AITER_FLYDSL_GEMM1_DISABLE_XDL_ARB_STALL=0 \
AITER_FLYDSL_GEMM1_WMMA_REUSE=1 \
AITER_FLYDSL_GEMM1_OVERLAP_OUTPUT_STORE=1 \
AITER_FLYDSL_GEMM2_A_PRESHUFFLE=1 \
AITER_FLYDSL_GEMM2_A_PRESHUFFLE_PRODUCER=rowgroup \
AITER_FLYDSL_GEMM2_A_PRESHUFFLE_RPW=2 \
AITER_FLYDSL_GEMM2_A_PRESHUFFLE_PREFETCH=2 \
AITER_FLYDSL_GEMM2_WAVES_PER_TENSOR_TDM=2 \
AITER_FLYDSL_GEMM2_SCHEDULE_HINTS=1 \
AITER_FLYDSL_GEMM2_MMA_GROUP=4 \
AITER_FLYDSL_GEMM2_FENCE_COVER_MMA=28 \
AITER_FLYDSL_GEMM2_OVERLAP_OUTPUT_STORE=1 \
AITER_FLYDSL_GEMM2_OUTPUT_SPLIT_WM=3 \
AITER_FLYDSL_GEMM2_OUTPUT_WAVE_SPLIT=1 \
bash /app/scripts/dsr1/serve_dsr1fp4_a8w8_tp1.sh \
  /data/models/DeepSeek-R1-0528-MXFP4
```

#### Optimized benchmark

```bash
cd /app/ATOM/scripts/performance
bash /app/scripts/dsr1/bench_dsr1fp4.sh \
  /data/models/DeepSeek-R1-0528-MXFP4

cp result.txt \
  /app/aiter/my_code/a07_serving_result_optimized_b899eaa2_20260929.txt
```

### 用于 shape 统计的 profiler run

`TRACE=1` 会让启动脚本向 ATOM 添加 `--torch-profiler-dir` 和 `--mark-trace`。benchmark 必须直接调用 `atom.benchmarks.benchmark_serving` 并增加 `--profile`，因为 `bench_dsr1fp4.sh` 本身不会传递该参数。

#### Baseline trace server

```bash
cd /app/aiter
git checkout 314ab7d48a077f3ccc055e7d353040c10a5f7fe5

cd /app/scripts/dsr1
TRACE=1 \
TRACE_DIR=/data/yanguahe/code/wk_sp1/traces/a07_moe_tokens_baseline_314ab7d_20260928T1623Z \
ATOM_PROFILER_MORE=0 \
FAKE_EPLB=1 \
AITER_FLYDSL_MOE_EXPERT_SCHEDULING_MODE=1 \
bash /app/scripts/dsr1/serve_dsr1fp4_a8w8_tp1.sh \
  /data/models/DeepSeek-R1-0528-MXFP4
```

#### Optimized trace server

最终成功的数据来自 retry trace：

```bash
cd /app/aiter
git checkout b899eaa23ebda8985440332fe3f40570633d2d9e

cd /app/scripts/dsr1
TRACE=1 \
TRACE_DIR=/data/yanguahe/code/wk_sp1/traces/a07_moe_tokens_optimized_b899eaa2_retry_20260929T0017CST \
ATOM_PROFILER_MORE=0 \
FAKE_EPLB=1 \
AITER_USE_GROUPED_GEMM=1 \
AITER_GROUPED_DEBUG=0 \
ENABLE_CK=0 \
FLYDSL_DUMP_IR=0 \
AITER_LOG_MORE=1 \
AITER_MOE_EXPERT_BALANCE=true \
AITER_FLYDSL_MOE_EXPERT_SCHEDULING_MODE=1 \
AITER_FLYDSL_GEMM1_A_PRESHUFFLE=1 \
AITER_FLYDSL_GEMM1_WAVES_PER_TENSOR_TDM=2 \
AITER_FLYDSL_GEMM1_MMA_GROUP=4 \
AITER_FLYDSL_GEMM1_FENCE_COVER_MMA=28 \
AITER_FLYDSL_GEMM1_DISABLE_XDL_ARB_STALL=0 \
AITER_FLYDSL_GEMM1_WMMA_REUSE=1 \
AITER_FLYDSL_GEMM1_OVERLAP_OUTPUT_STORE=1 \
AITER_FLYDSL_GEMM2_A_PRESHUFFLE=1 \
AITER_FLYDSL_GEMM2_A_PRESHUFFLE_PRODUCER=rowgroup \
AITER_FLYDSL_GEMM2_A_PRESHUFFLE_RPW=2 \
AITER_FLYDSL_GEMM2_A_PRESHUFFLE_PREFETCH=2 \
AITER_FLYDSL_GEMM2_WAVES_PER_TENSOR_TDM=2 \
AITER_FLYDSL_GEMM2_SCHEDULE_HINTS=1 \
AITER_FLYDSL_GEMM2_MMA_GROUP=4 \
AITER_FLYDSL_GEMM2_FENCE_COVER_MMA=28 \
AITER_FLYDSL_GEMM2_OVERLAP_OUTPUT_STORE=1 \
AITER_FLYDSL_GEMM2_OUTPUT_SPLIT_WM=3 \
AITER_FLYDSL_GEMM2_OUTPUT_WAVE_SPLIT=1 \
bash /app/scripts/dsr1/serve_dsr1fp4_a8w8_tp1.sh \
  /data/models/DeepSeek-R1-0528-MXFP4
```

#### Baseline 和 Optimized 共用的 profiler benchmark

在对应版本的 trace server ready 后执行：

```bash
cd /app/ATOM/scripts/performance

python -m atom.benchmarks.benchmark_serving \
  --model=/data/models/DeepSeek-R1-0528-MXFP4 \
  --backend=vllm \
  --base-url=http://localhost:8000 \
  --dataset-name=random \
  --random-input-len=1024 \
  --random-output-len=1024 \
  --random-range-ratio=1.0 \
  --num-prompts=256 \
  --max-concurrency=64 \
  --request-rate=inf \
  --ignore-eos \
  --percentile-metrics=ttft,tpot,itl,e2el \
  --profile
```

#### 从 trace 提取 baseline token shape

```bash
cd /app/aiter

python3 my_code/extract_serving_moe_tokens.py \
  /data/yanguahe/code/wk_sp1/traces/a07_moe_tokens_baseline_314ab7d_20260928T1623Z/rank_0/DeepSeek-R1-0528-MXFP4_ts_20260928_162852_050.pt.trace.json.gz \
  my_code/a07_moe_tokens_serving_baseline_314ab7d.json \
  --aiter-commit 314ab7d48a077f3ccc055e7d353040c10a5f7fe5 \
  --model /data/models/DeepSeek-R1-0528-MXFP4 \
  --input-length 1024 \
  --output-length 1024 \
  --requests 256 \
  --max-concurrency 64
```

#### 从 trace 提取 optimized token shape

```bash
cd /app/aiter

python3 my_code/extract_serving_moe_tokens.py \
  /data/yanguahe/code/wk_sp1/traces/a07_moe_tokens_optimized_b899eaa2_retry_20260929T0017CST/rank_0/DeepSeek-R1-0528-MXFP4_ts_20260929_002214_426.pt.trace.json.gz \
  my_code/a07_moe_tokens_serving_optimized_b899eaa2.json \
  --aiter-commit b899eaa23ebda8985440332fe3f40570633d2d9e \
  --model /data/models/DeepSeek-R1-0528-MXFP4 \
  --input-length 1024 \
  --output-length 1024 \
  --requests 256 \
  --max-concurrency 64
```

## 结论摘要

- optimized 的 Total Token throughput 提升约 `1.82%`。
- Mean TPOT 从 `30.44 ms` 降至 `30.18 ms`，改善约 `0.85%`。
- random MoE 单测中，decode 总 CUDA kernel 时间仅回退 `0.091%`，属于噪声范围。
- 按各版本实际 trace calls 合并全部 prefill shape 后，optimized 总 CUDA kernel 时间改善 `37.67%`。
- 两版共同出现的 shape 中，只有 `prefill T=15375` 的 kernel name 集合不同。

## Serving benchmark 性能

| 指标 | baseline `314ab7d` | optimized `b899eaa2` | optimized 变化 |
|---|---:|---:|---:|
| Benchmark duration | 131.62 s | 129.27 s | 改善 1.79% |
| Request throughput | 1.95 req/s | 1.98 req/s | 提升 1.54% |
| Output token throughput | 1991.68 tok/s | 2027.89 tok/s | 提升 1.82% |
| Total Token throughput | 3983.37 tok/s | 4055.79 tok/s | 提升 1.82% |
| Mean TTFT | 1759.30 ms | 1436.73 ms | 改善 18.34% |
| Median TTFT | 1536.19 ms | 1648.45 ms | 回退 7.31% |
| P99 TTFT | 2851.69 ms | 2279.33 ms | 改善 20.07% |
| Mean TPOT | 30.44 ms | 30.18 ms | 改善 0.85% |
| Median TPOT | 30.61 ms | 29.98 ms | 改善 2.06% |
| P99 TPOT | 31.86 ms | 31.24 ms | 改善 1.95% |
| Mean ITL | 30.41 ms | 30.15 ms | 改善 0.85% |
| Median ITL | 29.38 ms | 29.33 ms | 改善 0.17% |
| P99 ITL | 30.06 ms | 30.08 ms | 回退 0.07% |
| Mean E2EL | 32900.14 ms | 32312.29 ms | 改善 1.79% |
| Median E2EL | 32909.54 ms | 32311.53 ms | 改善 1.82% |
| P99 E2EL | 32938.40 ms | 32319.43 ms | 改善 1.88% |

除 Median TTFT 外，其余主要指标均有收益。Median TTFT 与 Mean/P99 TTFT 的方向不一致，说明单轮测试中的请求分批和排队分布存在波动。

## Serving trace 中的 MoE token 数

trace 字段含义：

```text
scheduled_tokens = scheduler 选择的逻辑 token 数
moe_tokens       = CUDAGraph padding 后真正进入模型和 MoE 的物理行数
calls            = 使用该 shape 的 model forward 次数
```

### Baseline trace

| phase | MoE T | forward calls |
|---|---:|---:|
| decode | 64 | 4096 |
| prefill | 1025 | 4 |
| prefill | 3075 | 4 |
| prefill | 15375 | 16 |

### Optimized trace

| phase | MoE T | forward calls |
|---|---:|---:|
| decode | 64 | 4096 |
| prefill | 1025 | 3 |
| prefill | 3075 | 3 |
| prefill | 4100 | 1 |
| prefill | 15375 | 16 |

两边实际处理的 prefill token 总量相同：

```text
baseline:  4×1025 + 4×3075 + 16×15375 = 262400
optimized: 3×1025 + 3×3075 + 1×4100 + 16×15375 = 262400
```

`T=1025/3075/4100` 的组合差异来自异步请求到达、请求完成和 scheduler 分批时序，不代表模型输入总量发生变化。

## Random MoE 单测方法

```text
data_format=a4w4
experts=256
topk=8
model_dim=7168
inter_dim=2048
activation=silu
bias=False
data_init=uniform
scale_init=auto
seed=0
AITER_MOE_EXPERT_BALANCE=true
```

每个版本读取自己的 Serving trace JSON，并对每个 `(phase, T, calls)`：

1. 执行 5 次非计时 warmup。
2. 在 `torch.profiler` 中严格执行 `calls` 次完整 `fused_moe` forward。
3. 不进行 IQR 或异常 iteration 删除。
4. 汇总全部 CUDA kernel 的 `self_device_time_total`。
5. 每个 case 重复 3 轮。
6. 检查 correctness 和三轮 kernel name 稳定性。

这里的总耗时表示单个 MoE 算子按 Serving forward 次数重复执行后的 CUDA kernel 累计时间，不是整个模型所有 MoE layer 的累计时间。

## Decode 与 Prefill 总 kernel 耗时

Prefill 不按各 T 单独比较，而是按照各版本 Serving trace 的实际 calls，将每轮所有 prefill shape 的总时间相加后再比较。

| phase | baseline 三轮总耗时 (us) | baseline 中位数 | optimized 三轮总耗时 (us) | optimized 中位数 | optimized 变化 |
|---|---|---:|---|---:|---:|
| decode | 1,377,793.216, 1,377,744.659, 1,378,664.271 | 1,377,793.216 | 1,376,654.958, 1,379,042.078, 1,379,195.677 | 1,379,042.078 | 回退 0.091% |
| prefill | 42,060.481, 42,045.532, 42,136.863 | 42,060.481 | 26,217.848, 26,319.628, 26,150.312 | 26,217.848 | 提升 37.67% |

换算为毫秒：

```text
Decode:
baseline  = 1377.793 ms
optimized = 1379.042 ms
变化      = 回退 0.091%

Prefill:
baseline  = 42.060 ms
optimized = 26.218 ms
变化      = 提升 37.67%
```

Decode 每次 forward 的中位数：

```text
baseline  = 336.375 us
optimized = 336.680 us
变化      = 回退 0.091%
```

这个差异处于正常测量噪声范围，没有发现 optimized 对 decode kernel 性能造成实质影响。

## Kernel name 对比

| phase | T | kernel name 集合是否相同 |
|---|---:|:---:|
| decode | 64 | 是 |
| prefill | 1025 | 是 |
| prefill | 3075 | 是 |
| prefill | 15375 | 否 |

`T=4100` 只出现在 optimized trace 中，因此没有 baseline 同 shape 对照。

### Decode `T=64`

两个版本的 kernel name 完全相同：

```text
a8w4_tdm_fp4_t64x256x256_w1x4_b3_K7168_e256_act1_q1r4
a8w4_tdm_fp4_t64x256x256_w1x4_b3_K2048_e256
moe_token_multidest_quant_k8_fd7168_r4_fp4_pk8_scpk_ks14
moe_contiguous_psum_remap
moe_gather_reduce_bf16_d7168_tk8_sk1_v4_wbf16_frlds
moe_route
void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<int>, std::array<char*, 1ul> >(...)
```

### Prefill `T=15375`

Baseline 专属 kernel：

```text
a8w4_tdm_fp4_t64x256x256_w1x4_b3_K7168_e256_act1_q1r4
a8w4_tdm_fp4_t64x256x256_w1x4_b3_K2048_e256
moe_token_multidest_quant_fusepre_k8_fd7168_r4_fp4_pk8_hidtdm4_quant
moe_scatter_preshuffle_scale_b224_r4_k8_g
```

Optimized 专属 kernel：

```text
a8w4_tdm_fp4_t256x256x256_w2x2_b4_K7168_e256_act1_cn4_prefetch_eb8_apre_sh_rcw_mg4_fc28_xdl0_reuse_ostore2p_s4
a8w4_tdm_fp4_t256x256x256_w2x2_b4_K2048_e256_cn4_prefetch_apre_sh_mg4_fc28_ostore2p_s3_ow2
moe_quant_token_fd7168_fp4_pk8
moe_invert_route_rows_tk8
moe_scatter_preshuffled_a_fd7168_r32_lds_pe7_slds_skipempty
moe_quant_preshuffled_a_fd2048_rpw2_pf2_direct_hidtdm4_otdmw2
```

`T=15375` 每次 forward 的总 CUDA kernel 时间：

```text
baseline  = 2370.798 us
optimized = 1402.080 us
提升      = 40.86%
```

## 结果解释与限制

1. Decode trace 在两个版本中都只有 `T=64`，共 `4096` 个 model forward；kernel name 完全相同，性能差异约 `0.091%`。
2. 小 prefill shape `T=1025` 和 `T=3075` 的 kernel name 也完全相同。
3. 只有 `T=15375` 进入 optimized A-preshuffle 路径，符合“大 token prefill 才启用优化”的设计。
4. Serving trace 中每个请求实际贡献 `1025` 个 prefill token，benchmark 参数中的 `1024` 之外还包含一个模型输入 token。
5. 单测使用 `E=256/topk=8`。ATOM serving 的部分日志中会显示 shared expert 合并后的 `E=257/topk=9`；因此本单测验证的是目标 routed-expert GEMM 路径和开关隔离，不等同于逐字节复现整个 serving MoE 调用。
6. Serving benchmark 每个版本只运行一轮，TTFT 等分位数仍可能受到 scheduler 分批和系统抖动影响。
7. 第一次 optimized 全量 trace 在 `T=15375` 遇到一次 `HSA_STATUS_ERROR_MEMORY_FAULT`；重新启动后完整重跑成功，最终 optimized token JSON 来自成功的 retry trace。

## 结果文件

本地脚本：

```text
my_code/extract_serving_moe_tokens.py
my_code/profile_serving_moe_shapes.sh
```

本地结果副本：

```text
.codex_tmp/a07_moe_tokens_serving_baseline_314ab7d.json
.codex_tmp/a07_moe_tokens_serving_optimized_b899eaa2.json
.codex_tmp/a07_moe_shape_baseline_random_summary.json
.codex_tmp/a07_moe_shape_optimized_random_summary.json
.codex_tmp/a07_moe_shape_random_comparison.json
.codex_tmp/a07_serving_result_baseline_314ab7d_20260929.txt
.codex_tmp/a07_serving_result_optimized_b899eaa2_20260929.txt
```

远端结果：

```text
/app/aiter/my_code/a07_moe_tokens_serving_baseline_314ab7d.json
/app/aiter/my_code/a07_moe_tokens_serving_optimized_b899eaa2.json
/app/aiter/my_code/serving_moe_shape_profiles/a07_baseline_random_314ab7d_20260929T0026CST
/app/aiter/my_code/serving_moe_shape_profiles/a07_optimized_random_b899eaa2_20260929T0029CST
/app/aiter/my_code/serving_moe_shape_profiles/a07_baseline_vs_optimized_random_20260929.json
/app/aiter/my_code/a07_serving_result_baseline_314ab7d_20260929.txt
/app/aiter/my_code/a07_serving_result_optimized_b899eaa2_20260929.txt
```

## Baseline 原始 Serving 结果

```text
============ Serving Benchmark Result ============
Successful requests:                     256
Benchmark duration (s):                  131.62
Total input tokens:                      262144
Total generated tokens:                  262144
Request throughput (req/s):              1.95
Output token throughput (tok/s):         1991.68
Total Token throughput (tok/s):          3983.37
Concurrency:                             63.99
---------------Time to First Token----------------
Mean TTFT (ms):                          1759.30
Median TTFT (ms):                        1536.19
P99 TTFT (ms):                           2851.69
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          30.44
Median TPOT (ms):                        30.61
P99 TPOT (ms):                           31.86
---------------Inter-token Latency----------------
Mean ITL (ms):                           30.41
Median ITL (ms):                         29.38
P99 ITL (ms):                            30.06
----------------End-to-end Latency----------------
Mean E2EL (ms):                          32900.14
Median E2EL (ms):                        32909.54
P99 E2EL (ms):                           32938.40
==================================================
```

## Optimized 原始 Serving 结果

```text
============ Serving Benchmark Result ============
Successful requests:                     256
Benchmark duration (s):                  129.27
Total input tokens:                      262144
Total generated tokens:                  262144
Request throughput (req/s):              1.98
Output token throughput (tok/s):         2027.89
Total Token throughput (tok/s):          4055.79
Concurrency:                             63.99
---------------Time to First Token----------------
Mean TTFT (ms):                          1436.73
Median TTFT (ms):                        1648.45
P99 TTFT (ms):                           2279.33
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          30.18
Median TPOT (ms):                        29.98
P99 TPOT (ms):                           31.24
---------------Inter-token Latency----------------
Mean ITL (ms):                           30.15
Median ITL (ms):                         29.33
P99 ITL (ms):                            30.08
----------------End-to-end Latency----------------
Mean E2EL (ms):                          32312.29
Median E2EL (ms):                        32311.53
P99 E2EL (ms):                           32319.43
==================================================
```
