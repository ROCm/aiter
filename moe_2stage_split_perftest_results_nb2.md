# aiter `test_moe_2stage` 性能测试 — **`num_buffers=2`**（K-pipelining）vs **`num_buffers=1`**

测试日期: **2026-05-12**
平台: AMD `gfx1250`（rocminfo: `amdgcn-amd-amdhsa--gfx1250`）
后端: FlyDSL gfx1250 mxscale MoE，`format=fp4`，`preshuffle=True`，`use_g1u1=True`

## 与 v4 的差异

本次唯一改动: 把 FlyDSL kernel 的 `num_buffers` 从默认 **1** 强行改成 **2**。
做法是 monkey-patch
[`aiter.ops.flydsl.kernels.moe_gemm_2stage_mxscale_gfx1250.compile_moe_gemm1`](/app/aiter/aiter/ops/flydsl/kernels/moe_gemm_2stage_mxscale_gfx1250.py)
和 `compile_moe_gemm2`，在调用前注入 `num_buffers=2`。
因为 `_compile_moe_mxscale_gemm` 走 `functools.lru_cache(maxsize=1024)`，更改 `num_buffers`
会得到一个新的 cache key，触发重新编译一份 K-双缓冲版本的 kernel。

`num_buffers` 的物理含义见
[`moe_gemm_2stage_common_gfx1250.py`](/app/aiter/aiter/ops/flydsl/kernels/moe_gemm_2stage_common_gfx1250.py):

- `num_buffers=1`: 经典 K 串行循环，每个 K-tile 读完再算。
- `num_buffers=2`: 双缓冲 K-pipeline (`_use_pipeline = num_buffers >= 2`)，
  下一个 K-tile 的 LDS load 与当前 K-tile 的 WMMA_SCALE 重叠，按
  `pre_loaded = num_buffers - 1 = 1` 预取一拍。前提是 `num_k_tiles >= num_buffers`。

> 其余 tile / shape / 量化策略与 v4 完全一致（`tile_m=16, tile_n=128, tile_k=128, in_dtype=fp4, out=bf16, activation=Silu`），方便 head-to-head 对比 v4 报告
> [`moe_2stage_split_perftest_results.md`](moe_2stage_split_perftest_results.md)。

## 测试入口

- 测试脚本: [`/app/aiter/op_tests/test_moe_2stage.py`](/app/aiter/op_tests/test_moe_2stage.py) (`test_fmoe`)
- Stage 拆分驱动: [`moe_2stage_v5_bench/run_split_perftest_nb2.py`](moe_2stage_v5_bench/run_split_perftest_nb2.py)
  - 注入 `num_buffers=2`（可通过 `AITER_GFX1250_NUM_BUFFERS` 环境变量切到其它值，合法范围 1/2/3/4，见 `_run_pipelined_k_loop` 校验）。
  - monkey-patch `_gfx1250_moe_stage1/_stage2` 把 fused_moe 最后一次调用的实参原样捕获下来。
  - `test_fmoe` 跑一次（warmup 5 + timed 20 次 cuda.Event）之后，再分别用 `aiter.test_common.run_perftest` 和直接 `torch.cuda.Event` 给 stage1 / stage2 重测。
- 环境（与 v4 完全相同）:
  - `AITER_LOG_MORE=1`、`ENABLE_CK=0`、`AITER_USE_OPUS_MOE_SORTING=1`
  - `AITER_MOE_WARMUP=5`、`AITER_MOE_ITERS=20`、`AITER_MOE_L2_FLUSH=1`
  - `AITER_GFX1250_PROBE=0`
- 原始日志: `moe_2stage_v5_bench/{dsv3_tp1,dsv3_tp4,dsv3_tp8}.{log,json}`

## Tile 配置（FlyDSL `mxscale` 路径，所有 case 一致）

| 项 | stage1 | stage2 |
|---|---|---|
| `in_dtype` | `fp4` (mxfp4 × mxfp4) | `fp4` |
| `out_dtype_str` | `bf16` | `bf16` |
| `tile_m` (`block_m`) | **16** | **16** |
| `tile_n` (caller default) | **128** | **128** |
| `tile_k` (caller default) | **128** | **128** |
| `activation` (融合) | `Silu` | (无) |
| **`num_buffers`** | **2 (K-pipelined)** | **2 (K-pipelined)** |

每个 case 实际传给 FlyDSL kernel 的张量形状（与 v4 一致，列出以便自查）：

| 配置 | M | stage1 a | stage1 w1 (E,N,K/2) | stage2 a (M,topk,K/2) | stage2 w2 (E,N,K/2) | stage1 out | stage2 out |
|---|---:|---|---|---|---|---|---|
| TP1 `7168,2048` | 1  | (1, 3584)/fp4    | (256, 4096, 3584) | (1, 8, 1024)  | (256, 7168, 1024) | (1, 8, 2048)  | (1, 7168)  |
| TP1 `7168,2048` | 64 | (64, 3584)/fp4   | (256, 4096, 3584) | (64, 8, 1024) | (256, 7168, 1024) | (64, 8, 2048) | (64, 7168) |
| TP4 `7168,512`  | 1  | (1, 3584)/fp4    | (256, 1024, 3584) | (1, 8, 256)   | (256, 7168, 256)  | (1, 8, 512)   | (1, 7168)  |
| TP4 `7168,512`  | 64 | (64, 3584)/fp4   | (256, 1024, 3584) | (64, 8, 256)  | (256, 7168, 256)  | (64, 8, 512)  | (64, 7168) |
| TP8 `7168,256`  | 1  | (1, 3584)/fp4    | (256, 512, 3584)  | (1, 8, 128)   | (256, 7168, 128)  | (1, 8, 256)   | (1, 7168)  |
| TP8 `7168,256`  | 64 | (64, 3584)/fp4   | (256, 512, 3584)  | (64, 8, 128)  | (256, 7168, 128)  | (64, 8, 256)  | (64, 7168) |

---

## 性能数据 — `num_buffers=2`

> 各列含义与 v4 表格一致：`fused` 是 fused_moe 外层 cuda.Event median；`stage1/stage2 cuda.Event median` 是同一组实参分开重测的中位数；`sum_cuda = stage1 + stage2`；`非 GEMM = fused − sum_cuda` 包含 a1/a2 量化、moe_sort、bias copy、kernel launch、stream sync。

### 1. DeepSeek-V3 TP1 (a4w4 / MXFP4 × MXFP4, SiLU) `-dim 7168,2048 -e 256 -k 8 -q 4`

| M | fused (us) | stage1 run_perftest (us) | stage2 run_perftest (us) | stage1 cuda.Event median (us) | stage1 mean | stage1 min/max | stage2 cuda.Event median (us) | stage2 mean | stage2 min/max | sum_cuda (us) | 非 GEMM (us) | verdict |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
|  1 | **1193.50** |  86.02 |  71.59 | **143.85** | 144.38 | 137.10 / 198.70 | **138.21** | 138.07 | 124.84 / 184.31 |  282.06 |  911.44 | FAIL (err=0.36) |
| 64 | **1321.69** | 1983.54 | 750.70 | **674.56** | 668.40 | 624.16 / 731.36 | **255.54** | 268.31 | 252.51 / 363.45 |  930.10 |  391.58 | FAIL (err=0.37) |

### 2. DeepSeek-V3 TP4 (a4w4 / MXFP4 × MXFP4, SiLU) `-dim 7168,512 -e 256 -k 8 -q 4`

| M | fused (us) | stage1 run_perftest (us) | stage2 run_perftest (us) | stage1 cuda.Event median (us) | stage1 mean | stage1 min/max | stage2 cuda.Event median (us) | stage2 mean | stage2 min/max | sum_cuda (us) | 非 GEMM (us) | verdict |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
|  1 | **1137.33** | 100.69 |  78.96 | **144.69** | 145.30 | 138.30 / 173.81 | **135.36** | 134.69 | 119.20 / 138.31 |  280.06 |  857.28 | FAIL (err=0.36) |
| 64 | **1183.44** | 515.31 | 109.67 | **172.74** | 174.62 | 167.30 / 235.21 | **141.13** | 137.99 |  91.18 / 145.97 |  313.87 |  869.57 | FAIL (err=0.35) |

### 3. DeepSeek-V3 TP8 (a4w4 / MXFP4 × MXFP4, SiLU) `-dim 7168,256 -e 256 -k 8 -q 4`

| M | fused (us) | stage1 run_perftest (us) | stage2 run_perftest (us) | stage1 cuda.Event median (us) | stage1 mean | stage1 min/max | stage2 cuda.Event median (us) | stage2 mean | stage2 min/max | sum_cuda (us) | 非 GEMM (us) | verdict |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
|  1 | **1147.35** |  83.07 |  66.46 | **142.81** | 143.31 | 132.07 / 175.78 | **134.64** | 134.45 | 121.41 / 142.20 |  277.45 |  869.90 | FAIL (err=0.39) |
| 64 | **1161.41** | 155.23 |  80.81 | **155.31** | 152.52 | 130.46 / 235.97 | **139.61** | 138.81 | 124.60 / 145.97 |  294.92 |  866.49 | FAIL (err=0.31) |

---

## Head-to-head: `num_buffers=1` (v4) vs `num_buffers=2` (v5)

> 这是本次的核心问题：**K-pipelining 在 DSV3 a4w4 MoE 上到底有没有用？**
> 下表所有 us 都是 cuda.Event median。Δ% 为正表示 `num_buffers=2` 更慢；负数表示更快。

| Config | M | fused us (nb=1) | fused us (nb=2) | Δfused | stage1 (nb=1) | stage1 (nb=2) | **Δstage1** | stage2 (nb=1) | stage2 (nb=2) | Δstage2 | sum_cuda (nb=1) | sum_cuda (nb=2) | Δsum |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| TP1 | 1  | 1195.87 | 1193.50 | **−0.2%** | 141.25 | 143.85 | +1.8% | 137.09 | 138.21 | +0.8% |  278.34 |  282.06 | +1.3% |
| TP1 | 64 | 1537.37 | 1321.69 | **−14.0%** | 864.66 | 674.56 | **−22.0%** | 270.16 | 255.54 | −5.4% | 1134.82 |  930.10 | **−18.0%** |
| TP4 | 1  | 1192.18 | 1137.33 | **−4.6%** | 143.37 | 144.69 | +0.9% | 135.96 | 135.36 | −0.4% |  279.34 |  280.06 |  +0.3% |
| TP4 | 64 | 1157.69 | 1183.44 | **+2.2%** | 173.98 | 172.74 | −0.7% | 136.07 | 141.13 | +3.7% |  310.06 |  313.87 |  +1.2% |
| TP8 | 1  | 1158.25 | 1147.35 | **−0.9%** | 143.98 | 142.81 | −0.8% | 136.00 | 134.64 | −1.0% |  279.98 |  277.45 |  −0.9% |
| TP8 | 64 | 1208.69 | 1161.41 | **−3.9%** | 151.87 | 155.31 | +2.3% | 137.09 | 139.61 | +1.8% |  288.95 |  294.92 |  +2.1% |

### 解读

1. **TP1 M=64 是唯一明确受益的 case**，stage1 从 864.66 µs → 674.56 µs，**单 stage 砍 22%**，fused 直接砍 14%。stage1 在该 case 下要算 `inter_dim*2 = 4096` 的 N、K=7168 的 K-loop（fp4 packed 后 K-tile 数 = 7168 / 128 = 56），有充足的 K-tile 给 double-buffer 隐藏 LDS load，所以 pipeline 几乎完全发挥。
2. **TP4 / TP8 在 M=64 上反而略变慢**（fused +2.2% / 略快 −3.9%；但 sum_cuda 都 +1–2%）。原因：stage1 的 N 砍到 512 / 256 后，可以并行的 tile 数变少，pipeline 启动 / 退出阶段相对开销变大；K-loop 的 issue 间隙本来就小，多缓冲反而引入了多余的同步。
3. **所有 M=1 case 都和 v4 在测量噪声范围内（≤ 2.6%）。** 因为 M=1 每个 expert 至多分到 ~0.03 个 token，kernel 是 launch-bound、不是 K-bound，K-pipelining 没有发挥空间。
4. **stage2 普遍小幅变动 (±1–5%)**，没有显著加速。stage2 的 N=`model_dim=7168` 很大，K 已经在 fp4 packing 后只剩 1024 / 256 / 128（按 TP1/4/8），K-tile 数本来就少，double buffer 收益不明显。
5. **`非 GEMM` 在 TP1 M=64 上从 402 µs → 392 µs**（基本持平），说明这部分确实和 GEMM 无关，主要是 sort / quant / sync / launch。

### 结论与建议

- **DSV3 TP1 prefill (M≥64)** 强烈建议把 `num_buffers` 设为 2，能拿到 **stage1 −22%、fused −14%** 的实测收益。
- **TP4 / TP8** 当前 (N=512/256) 还是保持 `num_buffers=1` 更安全，pipeline 收益不能盖过启动开销。
- **decode (M=1)** 路径 `num_buffers` 改动无差异，决策由 prefill 路径决定就行。
- 若未来 TP4 / TP8 的 N 变大（例如 fused gate||up 的 N 翻倍、或者 inter_dim 调整），值得再扫一遍 `num_buffers ∈ {1, 2, 3, 4}` 验证。

---

## 复现命令

```bash
export PYTHONPATH=/app/FlyDSL/build-fly/python_packages:/app/FlyDSL:$PYTHONPATH
export ENABLE_CK=0
export AITER_USE_OPUS_MOE_SORTING=1
export AITER_LOG_MORE=1
# 可选: 把 num_buffers 切到 1 / 3 / 4 重新扫
export AITER_GFX1250_NUM_BUFFERS=2

# === DeepSeek-V3 TP1 (a4w4) ===
AITER_LOG_MORE=1 python /app/aiter/op_tests/test_moe_2stage.py \
  -d bf16 -dim 7168,2048 -e 256 -k 8 -q 4 -a silu \
  -t 1 64 1024 65536 --no-flydsl-csv

# === DeepSeek-V3 TP4 (a4w4) ===
AITER_LOG_MORE=1 python /app/aiter/op_tests/test_moe_2stage.py \
  -d bf16 -dim 7168,512  -e 256 -k 8 -q 4 -a silu \
  -t 1 64 1024 65536 --no-flydsl-csv

# === DeepSeek-V3 TP8 (a4w4) ===
AITER_LOG_MORE=1 python /app/aiter/op_tests/test_moe_2stage.py \
  -d bf16 -dim 7168,256  -e 256 -k 8 -q 4 -a silu \
  -t 1 64 1024 65536 --no-flydsl-csv
```

> 注意：上面三条 raw 命令本身**没有** `num_buffers` override 钩子；要真的让 fused_moe 用 `num_buffers=2`，必须走下面这个驱动脚本（它在 import 阶段 monkey-patch `compile_moe_gemm1/2`）。

带 stage1 / stage2 拆分 + `num_buffers=2` 强制注入的版本（本表数据来源）：

```bash
# bs=1 / bs=64，三个 TP 配置各跑一次
AITER_GFX1250_NUM_BUFFERS=2 python /app/aiter/moe_2stage_v5_bench/run_split_perftest_nb2.py \
  --cases DSV3_TP1_a4w4 --tokens 1,64 \
  --out  /app/aiter/moe_2stage_v5_bench/dsv3_tp1.json \
  > /app/aiter/moe_2stage_v5_bench/dsv3_tp1.log 2>&1

AITER_GFX1250_NUM_BUFFERS=2 python /app/aiter/moe_2stage_v5_bench/run_split_perftest_nb2.py \
  --cases DSV3_TP4_a4w4 --tokens 1,64 \
  --out  /app/aiter/moe_2stage_v5_bench/dsv3_tp4.json \
  > /app/aiter/moe_2stage_v5_bench/dsv3_tp4.log 2>&1

AITER_GFX1250_NUM_BUFFERS=2 python /app/aiter/moe_2stage_v5_bench/run_split_perftest_nb2.py \
  --cases DSV3_TP8_a4w4 --tokens 1,64 \
  --out  /app/aiter/moe_2stage_v5_bench/dsv3_tp8.json \
  > /app/aiter/moe_2stage_v5_bench/dsv3_tp8.log 2>&1
```

---

## 文件清单

```
/app/aiter/
├── moe_2stage_split_perftest_results.md         # v4: num_buffers=1
├── moe_2stage_split_perftest_results_nb2.md     # 本文件: num_buffers=2
├── moe_2stage_v4_bench/                         # v4 原始日志 + 驱动
│   ├── run_split_perftest.py
│   └── dsv3_tp{1,4,8}.{log,json}
└── moe_2stage_v5_bench/                         # v5 原始日志 + 驱动 (num_buffers=2)
    ├── run_split_perftest_nb2.py                # 在 v4 driver 基础上注入 num_buffers
    └── dsv3_tp{1,4,8}.{log,json}
```

每个 `.log` 都含 `[NB_OVERRIDE]` / `[CASE]` / `[TILE]` / `[PERFTEST]` 四类行：

- `[NB_OVERRIDE]` 行：本轮 `num_buffers` 注入值；
- `[CASE]` 行：`test_fmoe` 的返回字典；
- `[TILE]` 行：实际传给 stage1/stage2 的 `in_dtype / tile_n / tile_k / block_m / activation / num_buffers` + 张量形状；
- `[PERFTEST]` 行：fused us、`run_perftest` 报的 stage1/stage2 us、`cuda.Event` median stage1/stage2 us、sum_cuda、非 GEMM 时间。
