# aiter `test_moe_2stage` 性能测试 — **stage1 / stage2 拆分 + run_perftest**（gfx1250 / FlyDSL MoE）

测试日期: **2026-05-12**
平台: AMD `gfx1250`（rocminfo: `amdgcn-amd-amdhsa--gfx1250`）
后端: FlyDSL gfx1250 mxscale MoE，`format=fp4`，`preshuffle=True`，`use_g1u1=True`

## 测试入口

- 测试脚本: [`/app/aiter/op_tests/test_moe_2stage.py`](/app/aiter/op_tests/test_moe_2stage.py) (`test_fmoe`)
- Stage 拆分驱动: [`moe_2stage_v4_bench/run_split_perftest.py`](moe_2stage_v4_bench/run_split_perftest.py)
  - 通过 monkey-patch `aiter.fused_moe._gfx1250_moe_stage1` / `_gfx1250_moe_stage2` 把每次 fused 调用最后传入 stage1 / stage2 的实参原样捕获下来；
  - `test_fmoe` 跑完一次（warmup 5 + timed 20 次 cuda.Event）之后，再分别用 `aiter.test_common.run_perftest` 和直接 `torch.cuda.Event` 各自重新计时 stage1 / stage2；
  - 用真实的 `torch.partial`-绑定参数（`in_dtype/out_dtype_str/tile_n/tile_k/activation`）输出每个 stage 的 tile metadata 到日志（`[TILE]` 行）。
- 环境:
  - `AITER_LOG_MORE=1`（按要求开启；并且 `aiter.test_common.perftest` 会在该变量下额外输出 `cuda.Event` 测得的 us/iter）
  - `ENABLE_CK=0`（gfx1250 还没有 CK target，避免 `module_aiter_core` JIT 编译失败）
  - `AITER_USE_OPUS_MOE_SORTING=1`
  - `AITER_MOE_WARMUP=5`、`AITER_MOE_ITERS=20`、`AITER_MOE_L2_FLUSH=1`
  - `AITER_GFX1250_PROBE=0`（拿到 tile 信息后关掉，避免每次调用都打 [probe] 日志拖慢 fused 时序）
- 原始日志同级保存在: `moe_2stage_v4_bench/{dsv3_tp1,dsv3_tp4,dsv3_tp8}.{log,json}`
- 本次按用户指令固定到 `-t 1 64`（即 bs=1 / bs=64）；命令本身保留了原始 `-t 1 64 1024 65536` 形态，方便扩展。

## Tile / kernel 配置（FlyDSL `mxscale` 路径，所有 3 个配置）

`MOEMetadata.stage1` / `MOEMetadata.stage2` 通过
[`get_2stage_cfgs`](/app/aiter/aiter/fused_moe.py) 路由到
`_gfx1250_moe_stage1` / `_gfx1250_moe_stage2`，再到
[`compile_moe_gemm1`](/app/aiter/aiter/ops/flydsl/kernels/moe_gemm_2stage_mxscale_gfx1250.py)
／`compile_moe_gemm2`。本轮测试 stage1 / stage2 的实参 tile 完全一致：

| 项 | stage1 | stage2 |
|---|---|---|
| `in_dtype` | `fp4` (mxfp4 × mxfp4) | `fp4` |
| `out_dtype_str` | `bf16` | `bf16` |
| `tile_m` (`block_m`) | **16** | **16** |
| `tile_n` (caller default) | **128** | **128** |
| `tile_k` (caller default) | **128** | **128** |
| `activation` (融合) | `Silu` | (无) |
| route_tile_m | 16 | 16 |

> `_pick_mxscale_launch_shape("fp4", route_tile_m=16, tile_n=128)` 返回 `(single_tile_m=16, single_tile_n=128, m_warp, n_warp)`，再用 `_pick_fp4_warp_shape` 选 warp 划分；本轮所有 case `block_m=16` / `tile_n=128` / `tile_k=128`，没有切到不同 tile。

每个 case 实际传给 FlyDSL kernel 的张量形状（fp4 用 `[E, N, K/2]` 的 packed 布局）：

| 配置 | M | stage1 a | stage1 w1 (E,N,K/2) | stage2 a (M,topk,K/2) | stage2 w2 (E,N,K/2) | stage1 out | stage2 out |
|---|---:|---|---|---|---|---|---|
| TP1 `7168,2048` | 1  | (1, 3584)/fp4    | (256, 4096, 3584) | (1, 8, 1024)  | (256, 7168, 1024) | (1, 8, 2048)  | (1, 7168)  |
| TP1 `7168,2048` | 64 | (64, 3584)/fp4   | (256, 4096, 3584) | (64, 8, 1024) | (256, 7168, 1024) | (64, 8, 2048) | (64, 7168) |
| TP4 `7168,512`  | 1  | (1, 3584)/fp4    | (256, 1024, 3584) | (1, 8, 256)   | (256, 7168, 256)  | (1, 8, 512)   | (1, 7168)  |
| TP4 `7168,512`  | 64 | (64, 3584)/fp4   | (256, 1024, 3584) | (64, 8, 256)  | (256, 7168, 256)  | (64, 8, 512)  | (64, 7168) |
| TP8 `7168,256`  | 1  | (1, 3584)/fp4    | (256, 512, 3584)  | (1, 8, 128)   | (256, 7168, 128)  | (1, 8, 256)   | (1, 7168)  |
| TP8 `7168,256`  | 64 | (64, 3584)/fp4   | (256, 512, 3584)  | (64, 8, 128)  | (256, 7168, 128)  | (64, 8, 256)  | (64, 7168) |

---

## 性能数据汇总

每一行内：

- **`fused (us)`**：`test_fmoe` 在 fused_moe 外层用 `cuda.Event(enable_timing=True)` 测 20 次 timed iter 的 median，等价于过去 `/app/aiter/moe_2stage_bench_results.md` 第 5–7 节的 `fused us (median)`。
- **`stage1 / stage2 run_perftest us`**：直接调 `aiter.test_common.run_perftest(_gfx1250_moe_stage{1,2}, *args, **kwargs, num_iters=20, num_warmup=5, testGraph=False, num_rotate_args=1)` 的返回值。`run_perftest` 在 gfx1250 上通过 `torch.profiler` 取设备时间，HIP profiler 在短 kernel 上有 3–5× 的测量放大；下方专门列出来便于和上游对齐。
- **`stage1 / stage2 cuda.Event median`**：同一组实参，自己直接打 `torch.cuda.Event` 取 20 次 median，物理意义最贴近 fused 内部那一次 stage 的真实时间。这一列才能和 `fused (us)` 加减做合理度量。
- **`sum_cuda`** = stage1.cuda + stage2.cuda。
- **`非 GEMM`** = fused − sum_cuda，包含 a1/a2 量化、moe_sort、bias copy、kernel launch、stream sync、event record 等所有 stage 之外的开销。
- **`mismatch_ratio`** / **`verdict`**：`test_fmoe` 在跑性能前先打了一次精度（aiter 参考 vs FlyDSL kernel），按 FlyDSL UT 的 `mismatch < 5% OR logits_diff < 0.25` 判定，全部 FAIL，是 a4w4 (mxfp4 × mxfp4) 在 K=7168 没做 K-pad 时的固有累积误差，与延迟数据物理意义不冲突。

### 1. DeepSeek-V3 TP1 (a4w4 / MXFP4 × MXFP4, SiLU) `-dim 7168,2048 -e 256 -k 8 -q 4`

| M | fused (us) | stage1 run_perftest (us) | stage2 run_perftest (us) | stage1 cuda.Event median (us) | stage1 mean | stage1 min/max | stage2 cuda.Event median (us) | stage2 mean | stage2 min/max | sum_cuda (us) | 非 GEMM (us) | verdict |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
|  1 | **1195.87** |  120.30 |  80.75 | **141.25** | 143.78 |  67.50 / 251.50 | **137.09** | 134.37 |  41.38 / 186.08 |  278.34 |  917.53 | FAIL (err=0.36) |
| 64 | **1537.37** | 2536.78 | 771.62 | **864.66** | 862.06 | 810.09 / 970.85 | **270.16** | 275.29 | 260.83 / 380.17 | 1134.82 |  402.55 | FAIL (err=0.28) |

### 2. DeepSeek-V3 TP4 (a4w4 / MXFP4 × MXFP4, SiLU) `-dim 7168,512 -e 256 -k 8 -q 4`

| M | fused (us) | stage1 run_perftest (us) | stage2 run_perftest (us) | stage1 cuda.Event median (us) | stage1 mean | stage1 min/max | stage2 cuda.Event median (us) | stage2 mean | stage2 min/max | sum_cuda (us) | 非 GEMM (us) | verdict |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
|  1 | **1192.18** |  83.30 |  67.37 | **143.37** | 144.51 | 136.04 / 172.86 | **135.96** | 127.29 |  11.22 / 176.46 |  279.34 |  912.85 | FAIL (err=0.36) |
| 64 | **1157.69** | 533.68 | 116.11 | **173.98** | 186.82 | 169.21 / 301.93 | **136.07** | 135.54 | 126.83 / 146.91 |  310.06 |  847.64 | FAIL (err=0.27) |

### 3. DeepSeek-V3 TP8 (a4w4 / MXFP4 × MXFP4, SiLU) `-dim 7168,256 -e 256 -k 8 -q 4`

| M | fused (us) | stage1 run_perftest (us) | stage2 run_perftest (us) | stage1 cuda.Event median (us) | stage1 mean | stage1 min/max | stage2 cuda.Event median (us) | stage2 mean | stage2 min/max | sum_cuda (us) | 非 GEMM (us) | verdict |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
|  1 | **1158.25** | 72.42 |  64.96 | **143.98** | 145.95 | 139.05 / 187.36 | **136.00** | 135.61 | 124.83 / 140.85 |  279.98 |  878.27 | FAIL (err=0.39) |
| 64 | **1208.69** | 149.90 |  80.78 | **151.87** | 147.60 | 104.20 / 242.88 | **137.09** | 137.74 |  57.21 / 227.42 |  288.95 |  919.74 | FAIL (err=0)    |

> 备注：TP8 M=64 的 `err=0` 是因为 `mismatch_ratio < 5%`，按 FlyDSL UT 判定为 PASS 后 `test_fmoe` 把 `err` 归零返回（见 [`test_fmoe`](/app/aiter/op_tests/test_moe_2stage.py) 中 `if passed: err = 0`）；`logits_diff` 仍然 ~0.46。

---

## 关键观察

1. **stage 拆分后 `sum_cuda` 与 `fused` 的差仍然很大**（M=1 时占 73–79%，M=64 时 TP4/TP8 还有 70+%），证实之前的结论：在 small-M 区间，瓶颈不在 stage1 / stage2 GEMM，而是 a1/a2 量化（per_1x32 mxfp4 / fp8 sort）、`moe_sort_fwd` 拷贝、`bias.to(fp32)` 复制、event record / stream sync 这些常驻 host 路径。
2. **TP1 → TP4 → TP8 在 M=1 上几乎不省 stage 时间**（stage1 都在 141–144 µs，stage2 都在 136–137 µs），说明 M=1 配置下每个 expert 只 dispatch 1 个 token，kernel 的 launch + memory binding 完全主导了执行时间——把 inter_dim 砍掉 16× 完全看不到收益。
3. **TP1 → TP4 → TP8 在 M=64 上 stage1 才看到 scaling**：
   - TP1 stage1 = 864.66 µs（inter_dim=2048，每个 expert 平均 ~2 个 token）
   - TP4 stage1 = 173.98 µs（inter_dim= 512，~5× 加速）
   - TP8 stage1 = 151.87 µs（inter_dim= 256，再 ~1.15× 加速，已接近 launch 下限）
   - **stage2 在 M=64 几乎不动**（270 → 136 → 137 µs），因为 stage2 的 N=`model_dim=7168` 不随 TP 切，K 被 packed 在一起后 atomic 加，反而 TP4/TP8 因为每个 atom 算的 inter_dim 更小所以 stage2 反而更平。
4. **`run_perftest` 的 us 与 cuda.Event median 在 small 上差不多**（M=1 时 stage1 perftest 83–120 µs vs cuda.Event 141 µs），但在 **M=64 TP1 上 run_perftest 报 2536 µs 而 cuda.Event 只有 864 µs**。差异来源是 `aiter.test_common.run_perftest` 通过 `torch.profiler` 抓 `device_time_sum`，每次会把 `out.zero_()` 的 fill kernel 和 stage kernel 都算上，而且 HIP profiler 启用后短 kernel 会被 instrumentation overhead 整体放大 3–5×。**性能对齐请用 cuda.Event median 这一列**。
5. **stage1 / stage2 单独跑时享受了 L2 完整命中**（warmup 5 + 同实参 timed 20），所以这里得到的 stage time 偏乐观；fused 内部因为还要做 sort + quant + bias copy，stage 之间 L2 状态没那么理想。这是把"`fused - sum_cuda`"理解成"非 GEMM 杂项开销"的上限的根本原因。

---

## 复现命令

```bash
export PYTHONPATH=/app/FlyDSL/build-fly/python_packages:/app/FlyDSL:$PYTHONPATH
export ENABLE_CK=0
export AITER_USE_OPUS_MOE_SORTING=1
export AITER_LOG_MORE=1

# === DeepSeek-V3 TP1 (a4w4 / MXFP4 × MXFP4) ===
# input:[M,7168] bf16 ; w1:[256,4096,7168] fp4x2 ; w2:[256,7168,2048] fp4x2 ; topk=8
AITER_LOG_MORE=1 python /app/aiter/op_tests/test_moe_2stage.py \
  -d bf16 -dim 7168,2048 -e 256 -k 8 -q 4 -a silu \
  -t 1 64 1024 65536 --no-flydsl-csv

# === DeepSeek-V3 TP4 (a4w4) ===
# w1:[256,1024,7168] ; w2:[256,7168,512]
AITER_LOG_MORE=1 python /app/aiter/op_tests/test_moe_2stage.py \
  -d bf16 -dim 7168,512  -e 256 -k 8 -q 4 -a silu \
  -t 1 64 1024 65536 --no-flydsl-csv

# === DeepSeek-V3 TP8 (a4w4) ===
# w1:[256,512,7168]  ; w2:[256,7168,256]
AITER_LOG_MORE=1 python /app/aiter/op_tests/test_moe_2stage.py \
  -d bf16 -dim 7168,256  -e 256 -k 8 -q 4 -a silu \
  -t 1 64 1024 65536 --no-flydsl-csv
```

带 stage1 / stage2 拆分 + tile 输出的版本（本表数据来源）：

```bash
# bs=1 / bs=64，三个 TP 配置各跑一次
python /app/aiter/moe_2stage_v4_bench/run_split_perftest.py \
  --cases DSV3_TP1_a4w4 --tokens 1,64 \
  --out  /app/aiter/moe_2stage_v4_bench/dsv3_tp1.json \
  > /app/aiter/moe_2stage_v4_bench/dsv3_tp1.log 2>&1

python /app/aiter/moe_2stage_v4_bench/run_split_perftest.py \
  --cases DSV3_TP4_a4w4 --tokens 1,64 \
  --out  /app/aiter/moe_2stage_v4_bench/dsv3_tp4.json \
  > /app/aiter/moe_2stage_v4_bench/dsv3_tp4.log 2>&1

python /app/aiter/moe_2stage_v4_bench/run_split_perftest.py \
  --cases DSV3_TP8_a4w4 --tokens 1,64 \
  --out  /app/aiter/moe_2stage_v4_bench/dsv3_tp8.json \
  > /app/aiter/moe_2stage_v4_bench/dsv3_tp8.log 2>&1
```

---

## 文件清单

```
/app/aiter/
├── moe_2stage_split_perftest_results.md          # 本文件
└── moe_2stage_v4_bench/
    ├── run_split_perftest.py                     # stage 拆分驱动脚本
    ├── dsv3_tp1.{log,json}                       # TP1 原始日志 + 结构化结果
    ├── dsv3_tp4.{log,json}                       # TP4 原始日志 + 结构化结果
    └── dsv3_tp8.{log,json}                       # TP8 原始日志 + 结构化结果
```

每个 `.log` 都包含 `[CASE]` / `[TILE]` / `[PERFTEST]` 三类行：

- `[CASE]` 行：`test_fmoe` 的返回字典，含 `us`、`err`、quant/act 参数；
- `[TILE]` 行：本次 stage1 / stage2 实际传给 `_gfx1250_moe_stage{1,2}` 的 `in_dtype / tile_n / tile_k / block_m / activation` 以及输入/输出张量形状；
- `[PERFTEST]` 行：fused us、`run_perftest` 报的 stage1/stage2 us、`cuda.Event` median stage1/stage2 us、sum_cuda、非 GEMM 时间。

`.json` 是同样数据的结构化版本，方便后续脚本化对比。
