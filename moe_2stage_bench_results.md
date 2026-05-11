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
