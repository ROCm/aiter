# gfx1250 E64/T1536/topk8 MoE tile 配置对比

## 测试环境

- 机器：`a07-3`
- 容器：`hyg_fyd_e2e`
- 仓库目录：`/app/aiter`
- 数据格式：A4W4
- activation：SiLU
- bias：关闭
- shape：
  - experts：64
  - tokens：1536
  - topk：8
  - model_dim：7168
  - inter_dim：2048
- 每个测试使用 `--iters 20`
- 性能测试前后均确认 GPU/KFD 空闲

## 对比版本

### 启用前

Commit：

```text
5d9e7dd9219eb50ab269d7e3b0fcaf1e13b4e064
```

该版本没有匹配到以下 grouped MoE CSV key：

```text
token=2048
model_dim=7168
inter_dim=2048
expert=64
topk=8
```

其中实际输入的 `tokens=1536` 经 `_get_padded_m()` 后使用 `token=2048` 进行配置查找。由于没有匹配的 tuned row，代码使用默认 tile。

实际 kernel：

```text
GEMM1: a8w4_tdm_fp4_t64x256x256_w1x4_b3_K7168_e64_act1_q1r4
GEMM2: a8w4_tdm_fp4_t64x256x256_w1x4_b3_K2048_e64
```

### 启用后

Commit：

```text
9d4ccd6295811d31aa8eba2a56a8a7cc2d25b8e5
```

该版本在 `aiter/configs/tuned_grouped_fmoe.csv` 中增加了 `token=2048, expert=64, topk=8` 的精确配置，GEMM1 和 GEMM2 使用：

```text
tile_m/tile_n/tile_k       = 256/256/256
tile_m2/tile_n2/tile_k2    = 256/256/256
m_warp/n_warp              = 2/2
m_warp2/n_warp2            = 2/2
num_buffers                = 4
num_buffer_stage2          = 4
cluster_n                  = 4
waves_per_tensor_tdm       = 2
next_stage_prefetch        = 1
```

实际 kernel：

```text
GEMM1: a8w4_tdm_fp4_t256x256x256_w2x2_b4_K7168_e64_act1_cn4_prefetch_eb8_apre_sh_rcw_mg4_fc28_xdl0_reuse_ostore2p_s4
GEMM2: a8w4_tdm_fp4_t256x256x256_w2x2_b4_K2048_e64_cn4_prefetch_apre_sh_mg4_fc28_ostore2p_s3_ow2
```

## 复现命令

### 进入测试环境

```bash
ssh -tt \
  -o ServerAliveInterval=30 \
  -o ServerAliveCountMax=3 \
  yanguahe@heliosr-1b114-a07-3.mnb.dcgpu

docker exec -it hyg_fyd_e2e bash
cd /app/aiter
```

### 启用前版本

切换到新增 tuned row 之前的 commit：

```bash
git -c advice.detachedHead=false checkout --detach \
  5d9e7dd9219eb50ab269d7e3b0fcaf1e13b4e064
```

在同一次运行中依次测试 random 和 const0：

```bash
ROUNDS=1 bash ./my_code/run_moe_prefill_switch_ab.sh e2e-both \
  --experts 64 \
  --tokens 1536 \
  --topk 8 \
  --model-dim 7168 \
  --inter-dim 2048
```

启用前测试日志：

```text
/app/aiter/my_code/moe_prefill_switch_ab_runs/20260929T140651Z
```

### 启用后版本

恢复优化分支并确认 commit：

```bash
git checkout hyg/moe_a4w4_pr
git rev-parse HEAD
```

预期输出：

```text
9d4ccd6295811d31aa8eba2a56a8a7cc2d25b8e5
```

测试 random 和 const0：

```bash
ROUNDS=1 bash ./my_code/run_moe_prefill_switch_ab.sh e2e-both \
  --experts 64 \
  --tokens 1536 \
  --topk 8 \
  --model-dim 7168 \
  --inter-dim 2048
```

启用后也可以分别复现两种数据：

```bash
ROUNDS=1 bash ./my_code/run_moe_prefill_switch_ab.sh e2e-random \
  --experts 64 \
  --tokens 1536 \
  --topk 8 \
  --model-dim 7168 \
  --inter-dim 2048

ROUNDS=1 bash ./my_code/run_moe_prefill_switch_ab.sh e2e-const0 \
  --experts 64 \
  --tokens 1536 \
  --topk 8 \
  --model-dim 7168 \
  --inter-dim 2048
```

本次启用后测试日志：

```text
const0: /app/aiter/my_code/moe_prefill_switch_ab_runs/20260929T140002Z
random: /app/aiter/my_code/moe_prefill_switch_ab_runs/20260929T140111Z
```

## 性能和正确性

### 启用前

| 数据 | GEMM1 | GEMM2 | fused MoE | logits_diff | rel_l2 | pass |
|---|---:|---:|---:|---:|---:|:---:|
| random | 135.889 us | 78.190 us | 253.09 us | 3.38491e-06 | 0.00260189 | True |
| const0 | 122.380 us | 73.498 us | 233.84 us | 0 | 0 | True |

### 启用后

| 数据 | GEMM1 | GEMM2 | fused MoE | logits_diff | rel_l2 | pass |
|---|---:|---:|---:|---:|---:|:---:|
| random | 103.385 us | 60.293 us | 221.25 us | 3.38491e-06 | 0.00260189 | True |
| const0 | 92.033 us | 55.465 us | 204.97 us | 0 | 0 | True |

## 性能提升

| 数据 | 指标 | 启用前 | 启用后 | 提升 |
|---|---|---:|---:|---:|
| const0 | GEMM1 | 122.380 us | 92.033 us | 24.80% |
| const0 | GEMM2 | 73.498 us | 55.465 us | 24.54% |
| const0 | fused MoE | 233.84 us | 204.97 us | 12.35% |
| random | GEMM1 | 135.889 us | 103.385 us | 23.92% |
| random | GEMM2 | 78.190 us | 60.293 us | 22.89% |
| random | fused MoE | 253.09 us | 221.25 us | 12.58% |

## 正确性结论

- 启用前和启用后的 random 测试均通过生产精度门限：

  ```text
  logits_diff = 3.38491e-06
  rel_l2      = 0.00260189
  ```

- 启用前和启用后的 const0 结果均完全一致：

  ```text
  logits_diff = 0
  rel_l2      = 0
  ```

- 新增 tuned row 只改变 kernel tile、cluster、A-preshuffle producer 和调度方式，没有引入可观察的精度回退。

## GEMM1 `cluster_m=1` 实验

在启用 `t256x256x256/w2x2/b4` 后，继续比较 GEMM1 的两种 cluster 几何：

```text
原版本：cluster_m=4, cluster_n=4，即 cluster=(4,4,1)
实验版：cluster_m=1, cluster_n=4，即 cluster=(4,1,1)
```

GEMM2 原本就是 `cluster=(4,1,1)`，本实验没有改变 GEMM2 的 cluster 几何。

### 代码改动

文件：

```text
aiter/ops/flydsl/kernels/mxfp4_preshuffle_gfx1250_tdm.py
```

核心修改：

```diff
-    cluster_m = 4 if fp4_prefill_schedule else 1
+    cluster_m = 1
```

实验版还为 GEMM1 增加 `_cm1` symbol 标记，以避免与原 4×4 kernel 的 JIT cache 混淆，并便于 profiler 确认实际运行版本：

```diff
+    _cluster_m = "_cm1" if fp4_prefill_schedule else ""
     _kname = (
         "a8w4_tdm_fp4"
         f"_t{tile_m}x{tile_n}x{tile_k}_w{m_warp}x{n_warp}"
         f"_b{num_buffers}_K{K}"
-        f"{_grouped}{_act}_cn4_prefetch{_epilogue_batch}_apre_sh"
+        f"{_grouped}{_act}_cn4{_cluster_m}_prefetch{_epilogue_batch}_apre_sh"
```

实际 GEMM1 symbol：

```text
4x4:
a8w4_tdm_fp4_t256x256x256_w2x2_b4_K7168_e64_act1_cn4_prefetch_eb8_apre_sh_rcw_mg4_fc28_xdl0_reuse_ostore2p_s4

4x1:
a8w4_tdm_fp4_t256x256x256_w2x2_b4_K7168_e64_act1_cn4_cm1_prefetch_eb8_apre_sh_rcw_mg4_fc28_xdl0_reuse_ostore2p_s4
```

GEMM2 在两个版本中相同：

```text
a8w4_tdm_fp4_t256x256x256_w2x2_b4_K2048_e64_cn4_prefetch_apre_sh_mg4_fc28_ostore2p_s3_ow2
```

### const0 三轮性能

两个版本都在 a07-3 GPU/KFD 空闲时使用以下命令运行：

```bash
ROUNDS=3 bash ./my_code/run_moe_prefill_switch_ab.sh \
  --experts 64 \
  --tokens 1536 \
  --topk 8 \
  --model-dim 7168 \
  --inter-dim 2048
```

原 4×4 cluster：

| 指标 | 三轮 samples | 中位数 |
|---|---|---:|
| GEMM1 | 90.155 us, 90.900 us, 91.214 us | 90.900 us |
| GEMM2 | 55.358 us, 55.054 us, 55.398 us | 55.358 us |
| fused MoE | 203.29 us, 203.40 us, 204.10 us | 203.40 us |

实验 4×1 cluster：

| 指标 | 三轮 samples | 中位数 |
|---|---|---:|
| GEMM1 | 84.555 us, 86.232 us, 84.808 us | 84.808 us |
| GEMM2 | 54.801 us, 55.685 us, 54.839 us | 54.839 us |
| fused MoE | 196.80 us, 198.87 us, 196.81 us | 196.81 us |

中位数对比：

| 指标 | 4×4 cluster | 4×1 cluster | 时间降低 | 性能提升 |
|---|---:|---:|---:|---:|
| GEMM1 | 90.900 us | 84.808 us | 6.091 us | 6.70% |
| GEMM2 | 55.358 us | 54.839 us | 0.518 us | 0.94% |
| fused MoE | 203.40 us | 196.81 us | 6.580 us | 3.24% |

两种 cluster 几何的 const0 结果均完全一致：

```text
logits_diff = 0
rel_l2      = 0
pass        = True
```

### 4×1 random 正确性和性能

复现命令：

```bash
ROUNDS=1 bash ./my_code/run_moe_prefill_switch_ab.sh e2e-random \
  --experts 64 \
  --tokens 1536 \
  --topk 8 \
  --model-dim 7168 \
  --inter-dim 2048
```

结果：

| GEMM1 | GEMM2 | fused MoE | logits_diff | rel_l2 | pass |
|---:|---:|---:|---:|---:|:---:|
| 97.237 us | 61.194 us | 216.17 us | 3.38491e-06 | 0.00260189 | True |

random 的误差与原 4×4 版本一致，没有精度回退。

### 性能变化原因

该测试启用 expert balance：

```text
1536 tokens × topk 8 / 64 experts = 192 routes/expert
```

每个 expert 的 192 行按 `tile_m=256` 对齐后只占一个 M tile。因此原 4×4 cluster 中，B/ScaleB 沿 M 方向的 expert-aware multicast 实际退化为 self-only，无法获得跨 M workgroup 的数据复用，但仍需承担 16-WG cluster、cluster barrier 和 co-residency 约束。

改成 4×1 后：

- 保留 A payload 沿 N 方向的 4-way multicast。
- 去掉 M 方向的 cluster barrier。
- 不再为本规模中没有实际收益的 B/ScaleB M 方向 multicast 组织 4 个 M peers。
- GEMM1 性能稳定提升约 6.7%，并传递为约 3.2% 的 fused MoE 提升。

### 4×1 实验复现流程

在本地修改并检查：

```bash
python -m black --check \
  aiter/ops/flydsl/kernels/mxfp4_preshuffle_gfx1250_tdm.py

python -m ruff check --no-cache \
  aiter/ops/flydsl/kernels/mxfp4_preshuffle_gfx1250_tdm.py
```

将实验 kernel 上传到 a07-3：

```bash
scp \
  aiter/ops/flydsl/kernels/mxfp4_preshuffle_gfx1250_tdm.py \
  yanguahe@heliosr-1b114-a07-3.mnb.dcgpu:/data/yanguahe/code/wk_sp1/mxfp4_preshuffle_gfx1250_tdm_cluster_m1.py
```

在 a07-3 创建独立 worktree：

```bash
docker exec hyg_fyd_e2e bash -lc '
  cd /app/aiter &&
  git worktree add --detach /tmp/aiter_cluster_m1_20260929 \
    9d4ccd6295811d31aa8eba2a56a8a7cc2d25b8e5
'
```

将上传文件复制进容器 worktree，并创建仅用于测试的临时 commit：

```bash
docker cp \
  /data/yanguahe/code/wk_sp1/mxfp4_preshuffle_gfx1250_tdm_cluster_m1.py \
  hyg_fyd_e2e:/tmp/aiter_cluster_m1_20260929/aiter/ops/flydsl/kernels/mxfp4_preshuffle_gfx1250_tdm.py

docker exec hyg_fyd_e2e bash -lc '
  cd /tmp/aiter_cluster_m1_20260929 &&
  git add aiter/ops/flydsl/kernels/mxfp4_preshuffle_gfx1250_tdm.py &&
  git -c user.name=Codex -c user.email=codex@local commit \
    -m "Temporary GEMM1 cluster_m1 benchmark"
'
```

运行三轮 const0：

```bash
docker exec hyg_fyd_e2e bash -lc '
  cd /tmp/aiter_cluster_m1_20260929 &&
  ROUNDS=3 bash ./my_code/run_moe_prefill_switch_ab.sh \
    --experts 64 \
    --tokens 1536 \
    --topk 8 \
    --model-dim 7168 \
    --inter-dim 2048
'
```

运行 random：

```bash
docker exec hyg_fyd_e2e bash -lc '
  cd /tmp/aiter_cluster_m1_20260929 &&
  ROUNDS=1 bash ./my_code/run_moe_prefill_switch_ab.sh e2e-random \
    --experts 64 \
    --tokens 1536 \
    --topk 8 \
    --model-dim 7168 \
    --inter-dim 2048
'
```

### 日志位置

```text
4×4 const0 三轮：
/app/aiter/my_code/moe_prefill_switch_ab_runs/20260929T143430Z

4×1 const0 三轮：
/app/aiter/my_code/moe_prefill_switch_ab_runs/cluster_m1_const0_rounds3_20260929T143553Z

4×1 random：
/app/aiter/my_code/moe_prefill_switch_ab_runs/cluster_m1_random_20260929T143747Z
```

4×1 实验结束后，容器正式仓库仍保持：

```text
hyg/moe_a4w4_pr
9d4ccd6295811d31aa8eba2a56a8a7cc2d25b8e5
```

## CSV 驱动的 `cluster_m=1` 与 t192 优化

本轮在前述 `t256x256x256/w2x2/b4` 配置基础上完成了两项优化：

1. 为 `aiter/configs/tuned_grouped_fmoe.csv` 新增 `cluster_m` 字段，通过 CSV 将 GEMM1 的 `cluster=(4,1,1)` 严格限制在当前调优规模。
2. 增加并修正 `t192x256x256/w2x2/b4` 调度，在通过 random MoE e2e 正确性后，将其写入当前规模的正式 tuned row。

### 最终 CSV 配置

只有以下精确 key 设置 `cluster_m=1`：

```text
token=2048
model_dim=7168
inter_dim=2048
expert=64
topk=8
```

实际输入为 `tokens=1536`，配置查询经过 padding 后使用 `token=2048` 作为 key。其他 CSV row 的 `cluster_m` 留空，GEMM1 继续使用默认 `cluster_m=4`。

目标 row 最终配置：

```text
tile_m/tile_n/tile_k       = 192/256/256
tile_m2/tile_n2/tile_k2    = 192/256/256
m_warp/n_warp              = 2/2
m_warp2/n_warp2            = 2/2
num_buffers                = 4
num_buffer_stage2          = 4
cluster_n                  = 4
cluster_m                  = 1
waves_per_tensor_tdm       = 2
next_stage_prefetch        = 1
```

因此 GEMM1 使用 `cluster=(4,1,1)`。未配置 `cluster_m` 的其他 GEMM1 规模仍默认使用 `cluster=(4,4,1)`。

### t192 实现修正

最初的 t192 版本仅放宽了 tile 检查，并直接复用了 t256 的调度参数。const0 无法暴露其中的错误，random 测试会产生错误结果。最终完成以下修正：

- t256 每个 k128 包含 32 条物理 WMMA，原调度固定使用 `FENCE_COVER_MMA=28`。
- t192 每个 k128 只有 24 条物理 WMMA。继续使用 28 会产生错误的 scheduler hint，因此改为按 tile 计算：t192 使用 `fc20`，t256 保持 `fc28`。
- t192 的 output store 从不均衡的 `4+2` 改为 `3+3`，使第一次 TDM store 后有更多独立 epilogue 工作可用于隐藏延迟。
- random 错误最终定位到 drain tail 的 `buf_ptr_opaque()` carry-target 地址。该技巧在 t192 生成的 LDS 地址形态下会读错 carry target，而全零数据会掩盖错误。t192 改用普通 foldable LDS 地址，t256 继续使用原来的 opaque 地址路径。

最终实际 kernel：

```text
GEMM1:
a8w4_tdm_fp4_t192x256x256_w2x2_b4_K7168_e64_act1_cn4_cm1_prefetch_eb8_apre_sh_rcw_mg4_fc20_xdl0_reuse_ostore2p_s3

GEMM2:
a8w4_tdm_fp4_t192x256x256_w2x2_b4_K2048_e64_cn4_prefetch_apre_sh_mg4_fc20_ostore2p_s3_ow2
```

### 相邻三轮 A/B 性能

测试环境：

```text
machine:   a07-3
container: hyg_fyd_e2e
repo:      /tmp/aiter_t192_cm1_20260929
shape:     E64/T1536/topk8/M7168/I2048
data:      const0
iters:     20
```

每次正式性能测试前后均使用 `/data/yanguahe/code/gpu_users.sh` 确认没有 GPU/KFD 占用。t256 和 t192 在同一时段相邻执行。

| 配置 | GEMM1 samples | GEMM1 median | GEMM2 samples | GEMM2 median | MoE e2e samples | MoE e2e median |
|---|---|---:|---|---:|---|---:|
| t256 + `cluster_m=1` | 84.302, 83.990, 84.760 us | 84.302 us | 55.657, 56.393, 54.970 us | 55.657 us | 198.51, 200.27, 198.48 us | 198.51 us |
| t192 + `cluster_m=1` | 80.897, 80.676, 81.435 us | 80.897 us | 54.489, 54.310, 54.034 us | 54.310 us | 194.50, 194.31, 193.15 us | 194.31 us |

| 指标 | t256 median | t192 median | t192 性能提升 |
|---|---:|---:|---:|
| GEMM1 | 84.302 us | 80.897 us | 4.04% |
| GEMM2 | 55.657 us | 54.310 us | 2.42% |
| fused MoE | 198.51 us | 194.31 us | 2.12% |

最终四个文件重新同步后的单轮 const0：

```text
GEMM1      = 81.399 us
GEMM2      = 54.708 us
fused MoE  = 194.60 us
logits_diff = 0
rel_l2      = 0
```

### 正确性

random MoE e2e：

```text
logits_diff = 3.3849e-06
rel_l2      = 2.6019e-03
pass        = True
```

const0：

```text
logits_diff = 0
rel_l2      = 0
pass        = True
```

const0 下 GEMM1、GEMM2 和最终 MoE 的 reference/output hash 均完全一致。random 通过 production `logits_diff < 0.01` 门限，且误差与此前正确版本一致。

### 复现环境

从 a07-3 主机进入容器：

```bash
docker exec -it hyg_fyd_e2e bash
cd /tmp/aiter_t192_cm1_20260929
```

当前临时测试树包含未提交的测试文件，因此直接运行 `env + python` 命令，不使用会检查 clean worktree 的 `run_moe_prefill_switch_ab.sh`。

### 检查 GPU 空闲

性能测试前后在 a07-3 主机执行：

```bash
/data/yanguahe/code/gpu_users.sh
```

如果输出显示其他 GPU/KFD 进程，运行结果只能用于正确性，不能作为性能数据。

### 切换目标 CSV row 的 tile

以下命令只修改 `/tmp/aiter_t192_cm1_20260929` 临时测试树中当前规模的 row。设置 `TILE_M=256` 可复现 t256 对照，设置 `TILE_M=192` 可恢复最终 t192 配置：

```bash
cd /tmp/aiter_t192_cm1_20260929

set_target_tile() {
  TILE_M="$1" python3 - <<'PY'
import os
from pathlib import Path

path = Path("aiter/configs/tuned_grouped_fmoe.csv")
lines = path.read_text(encoding="utf-8").splitlines()
header = lines[0].split(",")
tile_m = os.environ["TILE_M"]
matches = 0

for index in range(1, len(lines)):
    fields = lines[index].split(",")
    row = dict(zip(header, fields))
    if (
        row["token"] == "2048"
        and row["model_dim"] == "7168"
        and row["inter_dim"] == "2048"
        and row["expert"] == "64"
        and row["topk"] == "8"
    ):
        fields[header.index("tile_m")] = tile_m
        fields[header.index("tile_m2")] = tile_m
        lines[index] = ",".join(fields)
        matches += 1

assert matches == 1, f"expected one target row, got {matches}"
path.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"target row tile_m/tile_m2 set to {tile_m}/{tile_m}")
PY
}
```

复现 t256 对照配置：

```bash
set_target_tile 256
```

恢复最终 t192 配置：

```bash
set_target_tile 192
```

### 单轮 const0 正确性和性能

```bash
ENABLE_CK=0 \
AITER_MOE_EXPERT_BALANCE=true \
AITER_LOG_MORE=1 \
AITER_USE_GROUPED_GEMM=1 \
AITER_GROUPED_DEBUG=0 \
AITER_FLYDSL_MOE_EXPERT_SCHEDULING_MODE=1 \
FLYDSL_DUMP_IR=0 \
python3 -u my_code/test_flydsl_grouped_gemm_gfx1250.py \
  --scenario bench \
  --data-format a4w4 \
  --act silu \
  --no-bias \
  --no-check-aot-cache \
  --experts 64 \
  --tokens 1536 \
  --topk 8 \
  --model-dim 7168 \
  --inter-dim 2048 \
  --iters 20 \
  --const-init 0
```

该命令同时验证 const0 正确性，并输出 GEMM1、GEMM2 和 fused MoE 时间。性能数据仅在 GPU 空闲时有效。

### 单轮 random 正确性

```bash
ENABLE_CK=0 \
AITER_MOE_EXPERT_BALANCE=true \
AITER_LOG_MORE=1 \
AITER_USE_GROUPED_GEMM=1 \
AITER_GROUPED_DEBUG=0 \
AITER_FLYDSL_MOE_EXPERT_SCHEDULING_MODE=1 \
FLYDSL_DUMP_IR=0 \
python3 -u my_code/test_flydsl_grouped_gemm_gfx1250.py \
  --scenario bench \
  --data-format a4w4 \
  --act silu \
  --no-bias \
  --no-check-aot-cache \
  --experts 64 \
  --tokens 1536 \
  --topk 8 \
  --model-dim 7168 \
  --inter-dim 2048 \
  --iters 20
```

random 是判断 t192 drain/carry 地址是否正确的必要测试；只运行 `--const-init 0` 会掩盖错误 LDS 地址读取。

### 三轮 const0 性能

```bash
mkdir -p /tmp/e64_t192_final_logs

for round in 1 2 3; do
  log="/tmp/e64_t192_final_logs/const0_round_${round}.log"
  ENABLE_CK=0 \
  AITER_MOE_EXPERT_BALANCE=true \
  AITER_LOG_MORE=1 \
  AITER_USE_GROUPED_GEMM=1 \
  AITER_GROUPED_DEBUG=0 \
  AITER_FLYDSL_MOE_EXPERT_SCHEDULING_MODE=1 \
  FLYDSL_DUMP_IR=0 \
  python3 -u my_code/test_flydsl_grouped_gemm_gfx1250.py \
    --scenario bench \
    --data-format a4w4 \
    --act silu \
    --no-bias \
    --no-check-aot-cache \
    --experts 64 \
    --tokens 1536 \
    --topk 8 \
    --model-dim 7168 \
    --inter-dim 2048 \
    --iters 20 \
    --const-init 0 \
    2>&1 | tee "$log"
done
```

提取三轮关键结果：

```bash
grep -hE \
  'eager-profile:|fused_moe end-to-end us|bench kernel timing' \
  /tmp/e64_t192_final_logs/const0_round_*.log
```

### 完整 A/B 顺序

1. 检查 GPU 空闲。
2. 使用上述 CSV 切换命令设置 `TILE_M=256`。
3. 执行三轮 const0，保存 t256 日志。
4. 再次检查 GPU 空闲。
5. 使用上述 CSV 切换命令设置 `TILE_M=192`。
6. 执行三轮 const0，保存 t192 日志。
7. 执行一轮 random，确认 `logits_diff` 和 `rel_l2`。
8. 测试结束后再次确认 GPU/KFD 没有残留进程。

本轮原始日志位于容器内：

```text
t192 三轮 const0:
/tmp/t192_final_const0_r1.log
/tmp/t192_final_const0_r2.log
/tmp/t192_final_const0_r3.log

t256 相邻三轮 const0:
/tmp/t256_cm1_adjacent_const0_r1.log
/tmp/t256_cm1_adjacent_const0_r2.log
/tmp/t256_cm1_adjacent_const0_r3.log

最终同步文件后的 random:
/tmp/t192_final_exact_random.log

最终同步文件后的 const0:
/tmp/t192_final_exact_const0.log
```
