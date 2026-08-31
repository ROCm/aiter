# Comm-fused MoE 性能对比使用说明

本文说明两个受版本控制的性能入口：

- 算子级 TP8 A/B：`op_tests/multigpu_tests/test_flydsl_comm_fused_full_tp8_perf.py`
- InferenceX 端到端 A/B：`op_tests/multigpu_tests/run_inferencex_comm_fused_ab3.sh`

两者默认都在同一份当前代码中切换普通路径和通算融合路径。这样可以让输入、公共
primitive、模型配置和运行环境保持一致，适合判断通算融合本身的收益。

## 1. 对比口径

### 1.1 算子级 TP8 A/B

算子脚本在同一个 8-rank 进程组里运行两条路径：

```text
ordinary:
  Stage2 GEMM
  -> shared partial add
  -> BF16 TP all-reduce

fused:
  当前 comm_fused_moe.csv 选中的 production runner
  -> small / full / window / persistent / atomic
```

`ordinary` 是 `origin/main` 使用的算法路径，但不是从另一个 Git worktree 动态加载的代码。
如果当前分支修改了两条路径共享的 GEMM 或 collective primitive，这种改动会同时影响两边。

脚本会执行：

1. eager 正确性检查；
2. eager 延迟测量；
3. CUDA Graph 捕获；
4. ordinary/fused 交替测量；
5. 汇总每轮八个 rank 中最慢 rank 的延迟中位数。

### 1.2 InferenceX 端到端 A/B

端到端脚本启动两次相同配置的 ATOM server：

```text
Arm A: AITER_DISABLE_COMM_FUSED_MOE=1
Arm B: AITER_DISABLE_COMM_FUSED_MOE=0
```

每组场景使用相同 seed，并记录 AITer/ATOM commit、dirty worktree、配置文件哈希、服务日志和
benchmark 结果。它比较的是当前代码树中“禁用融合”和“启用融合”，同样不是自动切换到
`origin/main`。

## 2. 算子级 benchmark

### 2.1 前置条件

- 单机 8 张可互联的 gfx950/MI355X GPU；
- 已安装或挂载当前 AITer checkout；
- torch distributed 和 MORI/custom all-reduce 可用；
- 从仓库根目录启动，或将仓库根目录加入 `PYTHONPATH`。

### 2.2 快速运行

以下示例测量 `M=2048`、uniform routing，使用 3 轮、每轮 20 次 CUDA Graph replay：

```bash
cd /path/to/aiter
ulimit -c 0
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
HSA_COREDUMP_PATTERN=/dev/null \
MORI_SHMEM_HEAP_SIZE=40G \
COMM_FUSED_M=2048 \
COMM_FUSED_ROUTE=uniform \
COMM_FUSED_PERF_ROUNDS=3 \
COMM_FUSED_PERF_ITERS=20 \
torchrun --standalone --nproc_per_node=8 \
  op_tests/multigpu_tests/test_flydsl_comm_fused_full_tp8_perf.py
```

建议正式比较同时运行 `uniform` 和 `skew`：

```bash
for route in uniform skew; do
  COMM_FUSED_M=2048 \
  COMM_FUSED_ROUTE="$route" \
  COMM_FUSED_PERF_ROUNDS=5 \
  COMM_FUSED_PERF_ITERS=50 \
  HSA_COREDUMP_PATTERN=/dev/null \
  MORI_SHMEM_HEAP_SIZE=40G \
  torchrun --standalone --nproc_per_node=8 \
    op_tests/multigpu_tests/test_flydsl_comm_fused_full_tp8_perf.py
done
```

### 2.3 常用环境变量

| 变量 | 默认值 | 作用 |
| --- | ---: | --- |
| `COMM_FUSED_M` | `2048` | token bucket |
| `COMM_FUSED_ROUTE` | `uniform` | `uniform` 或 `skew` routing |
| `COMM_FUSED_PERF_ROUNDS` | `3` | 交替 A/B 轮数 |
| `COMM_FUSED_PERF_ITERS` | `20` | 每轮 Graph replay 次数 |
| `COMM_FUSED_COMPUTE` | `flydsl` | fused producer backend；默认使用 production FlyDSL 路径 |
| `COMM_FUSED_PROFILE_ONLY` | `0` | 只执行 profile replay，不输出完整 A/B |
| `COMM_FUSED_PROFILE_REPLAYS` | `1` | profile replay 次数 |
| `COMM_FUSED_PROFILE_GRAPH` | `0` | profile 时使用 CUDA Graph |

正式结果看下面这一行：

```text
FULL_RUNNER_M${M}_GRAPH ... ordinary_us=... fused_us=... speedup=...x
```

其中 `ordinary_us` 和 `fused_us` 都是每轮取最慢 rank、再跨轮取中位数。

## 3. InferenceX 端到端 benchmark

### 3.1 必需配置

脚本从自身位置自动推导 `AITER_REPO`。以下三个变量必须显式设置：

```bash
export CONTAINER=<running-atom-container>
export ATOM_REPO=/path/to/atom
export MODEL=/path/to/model
```

常用可选变量：

```bash
export DATA_ROOT=/path/to/benchmark-results
export JIT_DIR=/path/to/persistent-jit-cache
export HF_HOME_DIR=/path/to/huggingface-cache
export FMOE_TABLE=$AITER_REPO/aiter/configs/model_configs/dsv4_fp8fp4_tuned_fmoe.csv
export COMM_FUSED_TABLE=$AITER_REPO/aiter/configs/comm_fused_moe.csv
```

### 3.2 先做 preflight

```bash
op_tests/multigpu_tests/run_inferencex_comm_fused_ab3.sh --phase preflight
```

只查看场景矩阵、不启动 server：

```bash
op_tests/multigpu_tests/run_inferencex_comm_fused_ab3.sh \
  --phase synthetic --dry-run
```

运行单个场景：

```bash
op_tests/multigpu_tests/run_inferencex_comm_fused_ab3.sh \
  --phase synthetic \
  --scenario base_1024i_1024o_c64
```

运行完整 A/B 矩阵：

```bash
op_tests/multigpu_tests/run_inferencex_comm_fused_ab3.sh --phase all
```

中断后可用指定结果目录继续：

```bash
op_tests/multigpu_tests/run_inferencex_comm_fused_ab3.sh \
  --phase synthetic \
  --result-root /path/to/existing-result \
  --resume
```

结果目录包含每个 scenario 的三组配对结果以及 `summary.md`。脚本只会停止自己启动并记录的
server PID；如果容器里存在不属于本次运行的 ATOM server，它会拒绝继续，而不会主动清理。

## 4. 严格比较 origin/main

需要证明整个分支相对 `origin/main` 的收益时，应建立两个独立 worktree，并在同一节点、同一
容器、同一模型和同一 benchmark 参数下分别运行。不要让两个 worktree 共用同一个 JIT 目录：

```bash
git worktree add /path/to/aiter-main origin/main

AITER_REPO=/path/to/aiter-main \
AITER_JIT_DIR=/path/to/jit-main \
...运行 baseline...

AITER_REPO=/path/to/aiter-optimized \
AITER_JIT_DIR=/path/to/jit-optimized \
...运行 optimized...
```

算子级日常调优优先使用同进程 ordinary/fused A/B；只有最终归因或公共 primitive 发生变化时，
才需要上述严格 Git-ref 对比。

## 5. 运行纪律

- 每次记录 commit、节点、GPU 型号、route、M、rounds 和 iterations；
- 使用最慢 rank 延迟，不用 rank 0 或单轮最小值；
- 正式结果至少覆盖 uniform 和 skew；
- 性能比较前先通过 eager 和 Graph 正确性；
- 设置 `HSA_COREDUMP_PATTERN=/dev/null` 并执行 `ulimit -c 0`，避免 benchmark 产生 core dump；
- 短 smoke 使用少量轮次，参数确定后再执行正式长测。
