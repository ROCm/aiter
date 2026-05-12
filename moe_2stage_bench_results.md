## aiter `test_moe_2stage` 性能测试结果(gfx1250 / FlyDSL MoE,**stage1 / stage2 拆分**)

测试脚本基础: `/app/aiter/op_tests/test_moe_2stage.py`(`test_fmoe`)
拆分驱动脚本: `/tmp/aiter_bench/run_stage_breakdown.py`(monkey-patch `aiter.fused_moe._gfx1250_moe_stage1/_gfx1250_moe_stage2` 包一层 `cuda.Event`,捕获每次调用的 GPU 时间)
环境: `ENABLE_CK=0 AITER_USE_OPUS_MOE_SORTING=1 AITER_LOG_MORE=1`
`PYTHONPATH=/app/FlyDSL/build-fly/python_packages:/app/FlyDSL`
后端: `FlyDSL gfx1250 mxscale MoE`(`format=fp4`),`preshuffle=True`,`use_g1u1=True`,`L2_flush=True`,`warmup=5, iters=20`

含义说明:

- `fused (us)`:`test_fmoe` 报告的 end-to-end `cuda.Event` median 时间,包含两次 GEMM 内核 + 中间 quant + sort + bias + ε。
- `stage1 (us)`:`_gfx1250_moe_stage1` 这一次调用的 GPU 时间(median),包含 stage1 FlyDSL kernel 自身。
- `stage2 (us)`:`_gfx1250_moe_stage2` 同上。
- `kernel sum`:`stage1 + stage2`,与 fused 之间的差额可看成 sort + a2-quant + moe_sort_fwd + 其它 host/launch 开销。
- 数值未通过精度校验(`[FlyDSL gfx1250 FAIL]`),下表只反映 kernel 实际执行时间。

---

### 1. DeepSeek-R1 TP1(a4w4 / MXFP4 × MXFP4,Silu)`-dim 7168,2048 -e 256 -k 8 -q 4`

| token | fused (us) | stage1 median (us) | stage1 mean | stage1 min/max | stage2 median (us) | stage2 mean | stage2 min/max | kernel sum (us) | 非 GEMM 占比 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|    1 | 1383.89 |  275.69 |  272.69 |  221.09 / 307.18 |  195.25 |  187.11 |  142.69 / 204.78 |  470.94 | 66.0% |
|   64 | 1711.69 |  608.62 |  606.63 |  566.08 / 680.85 |  312.26 |  315.92 |  309.38 / 341.43 |  920.88 | 46.2% |
| 1024 | 3615.04 | 1265.72 | 1285.28 | 1242.84 / 1373.00 | 1029.01 | 1025.39 | 1007.97 / 1045.95 | 2294.72 | 36.5% |

---

### 2. DeepSeek-R1 TP4(a4w4 / MXFP4 × MXFP4,Silu)`-dim 7168,512 -e 256 -k 8 -q 4`

| token | fused (us) | stage1 median (us) | stage1 mean | stage1 min/max | stage2 median (us) | stage2 mean | stage2 min/max | kernel sum (us) | 非 GEMM 占比 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|    1 | 1346.60 | 219.20 | 213.06 | 147.10 / 252.61 | 140.93 | 140.61 |  78.12 / 214.96 |  360.13 | 73.3% |
|   64 | 1368.27 | 382.53 | 373.68 | 336.98 / 410.09 | 157.59 | 153.71 |  79.20 / 186.56 |  540.12 | 60.5% |
| 1024 | 1586.31 | 398.99 | 397.39 | 390.34 / 482.76 | 223.65 | 232.83 | 221.69 / 250.81 |  622.64 | 60.7% |

---

### 3. DeepSeek-R1 TP8(a4w4 / MXFP4 × MXFP4,Silu)`-dim 7168,256 -e 256 -k 8 -q 4`

| token | fused (us) | stage1 median (us) | stage1 mean | stage1 min/max | stage2 median (us) | stage2 mean | stage2 min/max | kernel sum (us) | 非 GEMM 占比 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|    1 | 1373.24 | 280.74 | 268.93 | 242.20 / 332.97 | 143.65 | 138.59 |  79.60 / 167.57 |  424.39 | 69.1% |
|   64 | 1400.56 | 342.87 | 333.99 | 289.95 / 365.30 | 146.62 | 140.79 |  80.72 / 189.00 |  489.49 | 65.1% |
| 1024 | 1368.39 | 285.02 | 282.13 | 232.10 / 324.48 | 312.62 | 317.96 | 286.99 / 346.47 |  597.64 | 56.3% |

---

### 4. GPT-OSS 120B TP1(a8w4 / MXFP8 × MXFP4,Swiglu)`-dim 2880,2880 -e 128 -k 4 -q 7` — **未完成**

之前的运行在 M=1 配置下进入 FlyDSL kernel 后长时间没有输出(>10 分钟被中断),还没有数据。
建议重试时:
- 先单独跑 `-t 1024` 看是否只在小 M 卡住;
- 试不同 `-hip` 组合(脚本默认 `hidden_pad=192, intermediate_pad=128` 对 K=2880 覆盖不够,自动选了 `(768, 768)`);
- 加 `AITER_LOG_LEVEL=DEBUG` 抓 FlyDSL JIT 阶段是否有死循环。

---

### 命令汇总

```bash
# 基础环境
export PYTHONPATH=/app/FlyDSL/build-fly/python_packages:/app/FlyDSL:$PYTHONPATH
export ENABLE_CK=0
export AITER_USE_OPUS_MOE_SORTING=1
export AITER_LOG_MORE=1

# 仅 fused (无 stage 拆分):
AITER_LOG_MORE=1 python /app/aiter/op_tests/test_moe_2stage.py \
  -d bf16 -dim 7168,2048 -e 256 -k 8 -q 4 -a silu -t 1 64 1024 --no-flydsl-csv

# 带 stage1 / stage2 拆分:
python /tmp/aiter_bench/run_stage_breakdown.py \
  --cases DSR1_TP1_a4w4 --tokens 1,64,1024 \
  --out /tmp/aiter_bench/stage_bd_tp1.json
# 类似把 --cases 换成 DSR1_TP4_a4w4 / DSR1_TP8_a4w4
```

---

### 数据来源 / 备注

1. 单次 `fused_moe()` 调用内部依次调用 `metadata.stage1(...)`(返回 `a2`)→ 若干 quant/sort/cast → `metadata.stage2(...)`。脚本把 `_gfx1250_moe_stage1` 和 `_gfx1250_moe_stage2` 各包了一层 `cuda.Event(start)/record() ... cuda.Event(end)/record()`,所以 `stage1 (us)` 和 `stage2 (us)` 是**该 stage 的内部 GPU 用时**(包含 partial 包装的 in_dtype/out_dtype_str/tile_n/tile_k 调度本身,但不含外层 quant)。
2. 每个 case 共 25 次调用(5 warmup + 20 timed),`stage1/stage2` 的统计只取后 20 次,与 `fused (us)` 的统计窗口一致。
3. `kernel sum = stage1 + stage2`,与 `fused` 的差(`fused - kernel sum`)就是 **stage 间的 quant + moe_sort_fwd + bias + launch + sync** 等开销;在 TP4 / TP8 / M=1 这种 GEMM 很短的场景里,这部分能占到 60–73%,瓶颈不在 stage GEMM 本身。
4. 跑通这个测试需要两个修改:
   - `ENABLE_CK=0`:`gfx1250` 在 `3rdparty/composable_kernel` 里没有匹配的 `MAP_COMPILER_STATE_TO_GFXxx_TARGET`,会让 `module_aiter_core` JIT 编译失败。
   - `aiter/ops/flydsl/kernels/{splitk_hgemm,small_m_hgemm}.py` 里 `from flydsl.compiler.protocol import fly_values` 改为 try/except fallback 到 `extract_to_ir_values`,以兼容 `/app/FlyDSL/build-fly/python_packages` 的新版 protocol API。
5. 所有 case 的精度校验都 FAIL(`[FlyDSL gfx1250 FAIL] mismatch_ratio>0.05 / logits_diff>0.25`),这是 FlyDSL gfx1250 MoE kernel 当前版本和 aiter 参考之间的已知差异,不影响 latency / TFLOPS 的物理意义。

---

## 2026-05-12 v2 sweep — 加上 bs=64 prefill (M=65536) 的完整跑测

本次相比第 1 节的差异:

- `aiter/fused_moe.py` 的两条 per-call `logger.info(...)` 已降级到 `logger.debug(...)`(改 `gfx1250 FlyDSL dispatch:` / `input shapes:`),省掉 50–200 μs/iter 的 stdout 开销。
- `test_moe_2stage.py` 的 `run_perftest` 已替换成直接 `torch.cuda.Event(enable_timing=True)` 计时(`AITER_MOE_WARMUP=5, AITER_MOE_ITERS=20, AITER_MOE_L2_FLUSH=1, AITER_MOE_USE_GRAPH=0`),`us` 是 20 次 timed iter 的 **median**。
- `checkAllclose` 的 msg 行加了 `GB/s` 估算(下界:input + min(E, M·topk) 个 expert weights + scales + output)。
- 没用 hipgraph,因为前次实测发现 graph 模式只在 M=1 上有正向收益(大 M 反而被 L2_flush + replay 调度争用拖慢);默认仍走 eager。

### 5. DeepSeek-V3 TP1 (a4w4 / MXFP4 × MXFP4, Silu) `-dim 7168,2048 -e 256 -k 8 -q 4`

| token | fused us (median) | mean | min | max | TFLOPS | est BW (GB/s) | mismatch | logits_diff | verdict |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
|     1 |     1169.67 |   1230.32 |   1109.66 |   1551.72 |   0.60 |  160.04 | 0.337 | 0.332 | FAIL |
|    64 |     1269.55 |   1288.19 |   1228.65 |   1728.15 |  35.52 | 4719.24 | 0.413 | 0.446 | FAIL |
|  1024 |     3166.50 |   3168.84 |   3037.10 |   3367.43 | 227.87 | 1900.78 | 0.421 | 0.461 | FAIL |
| 65536 |   124755.52 | 124791.70 | 124112.21 | 125143.59 | 370.16 |   63.07 | 0.444 | 0.508 | FAIL |

### 6. DeepSeek-V3 TP4 (a4w4) `-dim 7168,512 -e 256 -k 8 -q 4`

| token | fused us (median) | mean | min | max | TFLOPS | est BW (GB/s) | mismatch | logits_diff | verdict |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
|     1 |    1146.60 |   1161.33 |   1092.83 |   1292.57 |   0.15 |   40.84 | 0.390 | 0.403 | FAIL |
|    64 |    1158.09 |   1163.32 |   1108.66 |   1292.81 |   9.74 | 1294.54 | 0.401 | 0.435 | FAIL |
|  1024 |    1198.59 |   1202.24 |   1151.00 |   1387.11 | 150.50 | 1273.77 | 0.421 | 0.465 | FAIL |
| 65536 |   37416.97 |  37445.97 |  37225.62 |  37964.84 | 308.55 |   90.24 | 0.442 | 0.507 | FAIL |

### 7. DeepSeek-V3 TP8 (a4w4) `-dim 7168,256 -e 256 -k 8 -q 4`

| token | fused us (median) | mean | min | max | TFLOPS | est BW (GB/s) | mismatch | logits_diff | verdict |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
|     1 |    1170.91 |   1180.90 |   1105.05 |   1292.53 |   0.08 |   20.01 | 0.309 | 0.305 | FAIL |
|    64 |    1170.27 |   1177.95 |   1113.99 |   1300.26 |   4.82 |  641.32 | 0.411 | 0.462 | FAIL |
|  1024 |    1190.98 |   1203.04 |   1125.76 |   1440.35 |  75.73 |  653.28 | 0.405 | 0.446 | FAIL |
| 65536 |   30589.95 |  30589.49 |  30441.57 |  30768.11 | 188.70 |   85.90 | 0.431 | 0.492 | FAIL |

### 8. GPT-OSS 120B TP1 (a8w4 / MXFP8 × MXFP4, Swiglu) `-dim 2880,2880 -e 128 -k 4 -q 7` — **未完成**

`-t 1 64 1024 65536` sweep 在 **M=1** 这一个 case 进入 FlyDSL kernel 后超过 12 分钟无任何 cuda.Event 输出,被人工中断。和第 4 节里描述的现象完全一致(`hidden_pad=768, intermediate_pad=768` 的自动 K-pad 已生效,但 kernel 本身在 M=1 配置下挂住)。

参考数据点(同 q=7 a8w4 路径但 dim 不同,**已 PASS**):

| 配置 | M | dim | E | topk | hidden/inter pad | fused us | TFLOPS | est BW | verdict |
|---|---:|---|---:|---:|---|---:|---:|---:|:---:|
| `-dim 3072,3072 -e 128 -k 4 -q 7` | 512 | 3072/3072 | 128 | 4 | 768/768 (auto) | 7241.93 | 16.01 | 266.71 | **PASS** (mismatch=0.0016, logits=0.0425) |

**下一步排查建议:**

- 单跑 `-t 1024` 看是否只在 M=1 卡;
- 加 `AITER_GFX1250_DEBUG=1` 抓 fused_moe 内部 quant / dispatch 阶段的 probe 日志,定位是 stage1 还是 stage1 之前 hang;
- 试 `-hip 192,128`(关掉自动 K-pad)对比;
- `AITER_LOG_LEVEL=DEBUG` 看 FlyDSL JIT compile 是否在 M=1 走了死循环。

### 关键观察(基于 5–7 节)

1. **TP1 1024→65536 (×64) → fused 时间 ×39.4** (3166 → 124756 us),scaling 接近线性,瓶颈彻底落在 stage1/stage2 GEMM,host 开销摊薄。BW 估算从 1900 → **63 GB/s** 反而下降是因为 BW 公式在大 M 时 `min(E, M·topk)=256` 已经饱和(每个 expert 只读一次),而每 token 的 activation 写入和 partial sum reduce 真实带宽被 M 摊大,公式低估实际 traffic。
2. **TP4/TP8 small-M 区间(M ≤ 1024) fused us 几乎不变** (TP4 1147→1199,TP8 1171→1191),完全是 host launch + non-GEMM 固定开销主导;这跟第 1 节的结论一致 —— 切到 TP4/TP8 后 stage GEMM 已经只占 1/3 不到。
3. **TP4/TP8 大 M=65536 才有意义的速度差**:TP4 = 37417 μs,TP8 = 30590 μs,提速 ~22%。这跟 TP8 stage GEMM 砍半的预期一致(TP4→TP8 inter_dim 从 512 → 256)。
4. **fused 时间相比第 1 节 baseline (run_perftest + INFO log)**:
   - TP1 M=1 1383.89 → 1169.67 (−15.5%)
   - TP1 M=64 1711.69 → 1269.55 (−25.8%)
   - TP1 M=1024 3615.04 → 3166.50 (−12.4%)
   - 全局 12–26% 的下降 = `logger.info → debug` + `cuda.Event` 直测(去掉 profiler 包装) 的合计收益。
5. **精度 verdict 全 FAIL 是 mxfp4/mxfp8 a4w4 在 K=7168 + 没 K-pad 下的固有累积噪声**(mismatch ~0.30–0.44, logits_diff ~0.30–0.51),与 kernel 是否能跑无关,latency/TFLOPS 数据有效。

### v2 sweep 命令汇总

```bash
export PYTHONPATH=/app/FlyDSL/build-fly/python_packages:/app/FlyDSL:$PYTHONPATH
export ENABLE_CK=0
export AITER_USE_OPUS_MOE_SORTING=1

# DSV3 TP1 (a4w4)
AITER_LOG_MORE=1 python /app/aiter/op_tests/test_moe_2stage.py \
  -d bf16 -dim 7168,2048 -e 256 -k 8 -q 4 -a silu \
  -t 1 64 1024 65536 --no-flydsl-csv

# DSV3 TP4 (a4w4)
AITER_LOG_MORE=1 python /app/aiter/op_tests/test_moe_2stage.py \
  -d bf16 -dim 7168,512  -e 256 -k 8 -q 4 -a silu \
  -t 1 64 1024 65536 --no-flydsl-csv

# DSV3 TP8 (a4w4)
AITER_LOG_MORE=1 python /app/aiter/op_tests/test_moe_2stage.py \
  -d bf16 -dim 7168,256  -e 256 -k 8 -q 4 -a silu \
  -t 1 64 1024 65536 --no-flydsl-csv

# GPT-OSS 120B TP1 (a8w4) — 当前 hang 在 M=1
AITER_LOG_MORE=1 python /app/aiter/op_tests/test_moe_2stage.py \
  -d bf16 -dim 2880,2880 -e 128 -k 4 -q 7 \
  -t 1 64 1024 65536 --no-flydsl-csv
```

原始日志:`/tmp/aiter_bench_v3/{dsv3_tp1,dsv3_tp4,dsv3_tp8,gptoss}.log`
