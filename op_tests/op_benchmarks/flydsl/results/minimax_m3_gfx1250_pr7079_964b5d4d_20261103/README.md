# MiniMax-M3 TP4：PR 7079 最新版 gfx1250 性能回退检查

这组 MiniMax-M3 TP4 合成 PA workload **未发现稳定、超过 5% 的性能回退**。
新版 `964b5d4d8` 与此前测过的 `15214dad7` 各运行三遍完整矩阵，84 个配置的等权几何平均延迟变化为 **+0.24%**。
83 个配置的变化在 **−2.81% 至 +2.74%**；另一个 FP8 长 context verify 配置在完整矩阵中为 +4.86%，单独交替复测为 −0.07%，未稳定复现。
5% 是本次筛查明显回退的工程阈值；原始小幅差异和波动保留在下表，这个阈值不代表统计置信区间。

两组针对性复测和一遍 main 对照也全部完成，共 **660 个测量点通过正确性检查**。
结果只覆盖 PA 算子；不同形状的等权统计不代表模型端到端延迟变化。

## 环境与版本

| 项目 | 设置 |
| --- | --- |
| PR | [ROCm/aiter #7079](https://github.com/ROCm/aiter/pull/7079) |
| 新版 PR | `964b5d4d8f7085497238f69fface36cb90ef36b2` |
| 上一版 PR | `15214dad75fc26d13402b1571de080cf56c514f3`，此前报告所测实现 |
| main 对照 | `6d8c8e9c247f318204303724f34d6d134c24d864`，新版 PR 合入的 main |
| GPU | 物理 GPU 3，gfx1250，256 CUs，432 GiB VRAM |
| GPU UUID | `33656162-3833-3532-6661-356238663661` |
| PyTorch / HIP | `2.11.0+rocm10.1.0a20260811-bkc.20260820` / `7.16.26315` |
| FlyDSL | `0.4.1`，源码 `f37f7a2b98f295902240f9f96612222003578b40`，起止均干净且版本相同 |
| 输入 | Hq16 / Hkv1 / D128；Q/O BF16；KV BF16 或 FP8 E4M3FN + FP32 per-token K/V scales |
| Dense | page128；decode qlen1；verify qlen4，causal mask |
| Sparse | page16；2048 selected tokens；4096-page shared KV pool |
| Prefill | 第 r 行 context 为 `min(r+1, 2048)` |
| GPU 隔离 | `HIP_VISIBLE_DEVICES=3`；在该卡串行运行，未修改时钟 |
| 空闲检查 | 起始利用率 0，显存基础占用约 173 MiB；结束恢复基础占用 |
| 随机种子 | 37 |
| Harness SHA256 | `46aa561f489625fb455719ef57f2367ab891c85fc99588c29ce1fd73216a60ed` |

GPU 2 当时有其他任务，GPU 0/1 也有显存占用，因此选 GPU 3。
上一版也在 GPU 3 上重测，避免直接与此前 GPU 0 的结果比较。
每遍完整测试的 GPU 起止记录和两段占用采样附在本目录。
main 使用 PR 合入的版本；检查时它与当前 `origin/main` 的 PA 文件没有变化。

## 计时与正确性

沿用 [MiniMax-M3 benchmark](../../bench_pa_minimax_m3.py)，每个 graph 包含 10 次完整 PA 调用；每个样本测量 5 次 replay，除以 50 次调用。
每轮取 5 个样本的最小值，每个进程取 **5 轮最小值的中位数**；主表再取 **3 个独立进程结果的中位数**。
每遍保持相同 case 顺序、输入、计时参数和环境。完整测试顺序为：

```text
previous → main → head → head → previous → previous → head
```

main 对照只跑一个完整进程，用于观察 PR 保留的优化收益；新旧 PR 的回退判断使用各三个进程。
两组针对性复测分别按 `previous → head → head → previous → previous → head` 执行。
编译、planning、分配、量化和 reference 检查均在计时外；测量包含需要的 partition reduction。

每个测量点检查 eager、graph replay、Q 取反且 output 填 NaN 后的 replay。
BF16 使用 `rtol=atol=0.005`；FP8 检查每行 relative L2 ≤ 0.05。
FP8 最大行 relative L2 为 **0.03444159**。7 遍完整矩阵 × 84 点，加 6 × 8 点 decode 复测和 6 × 4 点 verify 复测，共 **660 点全部通过**。

## 首轮可疑点与交替复测

首轮 ATOM BF16 dense decode bs8/8K 为 6.284 → 6.886 µs（+9.58%）。随后两遍完整矩阵的新版为 6.297 和 6.288 µs，三遍汇总为 +0.32%。
单独交替复测该形状和 bs4/32K，覆盖两个 profile 和两个 dtype，结果如下：

| 场景 | Profile | KV | 上一版 (µs) | 新版 (µs) | 延迟变化 |
| --- | --- | --- | ---: | ---: | ---: |
| Dense decode bs8 / 8K | atom | bf16 | 6.265 | 6.300 | +0.56% |
| Dense decode bs4 / 32K | atom | bf16 | 7.305 | 7.375 | +0.97% |
| Dense decode bs8 / 8K | atom | fp8 | 5.903 | 5.925 | +0.37% |
| Dense decode bs4 / 32K | atom | fp8 | 6.695 | 6.755 | +0.90% |
| Dense decode bs8 / 8K | pr | bf16 | 6.073 | 6.076 | +0.04% |
| Dense decode bs4 / 32K | pr | bf16 | 7.450 | 7.546 | +1.29% |
| Dense decode bs8 / 8K | pr | fp8 | 5.585 | 5.588 | +0.04% |
| Dense decode bs4 / 32K | pr | fp8 | 6.845 | 6.708 | -2.01% |

FP8 dense MTP4 verify bs3/220K 的 PR profile 在完整矩阵中为 23.430 → 24.568 µs（+4.86%）。
单独交替复测三个进程后为 23.795 → 23.778 µs（−0.07%）；ATOM profile 为 +0.15%。
完整矩阵和单独测量可能有输入地址、缓存状态与前序 case 的差异；本次没有确定具体原因。
保留两组样本，本次无法将 +4.86% 归因为稳定的实现回退。

| Profile | KV | 上一版 (µs) | 新版 (µs) | 延迟变化 |
| --- | --- | ---: | ---: | ---: |
| atom | bf16 | 33.519 | 33.804 | +0.85% |
| atom | fp8 | 23.948 | 23.985 | +0.15% |
| pr | bf16 | 33.569 | 33.449 | -0.36% |
| pr | fp8 | 23.795 | 23.778 | -0.07% |

该 verify 单独复测的 FP8 PR-profile 三次结果：上一版 `[23.795, 23.455, 24.289]` µs，新版 `[23.027, 24.154, 23.778]` µs；区间重叠且变化方向随进程改变。首轮异常与完整矩阵的长 context 差异保留在原始文件中。

## ATOM 配置：完整矩阵

ATOM dense table 补齐到 256K，context bound 为 None；sparse 使用推荐 split cap，prefill cap1。

时间单位为 µs；变化为 `新版 / 上一版 − 1`，正数表示变慢。两版输出路径相同。

| 场景 | BF16 上一版 | BF16 新版 | BF16 变化 | FP8 上一版 | FP8 新版 | FP8 变化 | 输出路径 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Dense decode bs1 / 8K | 5.537 | 5.564 | +0.49% | 5.345 | 5.378 | +0.61% | reducer |
| Dense decode bs4 / 8K | 5.956 | 5.969 | +0.22% | 5.681 | 5.725 | +0.79% | reducer |
| Dense decode bs8 / 8K | 6.277 | 6.297 | +0.32% | 5.908 | 5.946 | +0.64% | reducer |
| Dense decode bs32 / 8K | 9.342 | 9.361 | +0.21% | 7.851 | 7.850 | -0.01% | reducer |
| Dense decode bs4 / 32K | 7.313 | 7.513 | +2.74% | 6.704 | 6.725 | +0.32% | reducer |
| Dense decode bs3 / 220K | 29.562 | 29.552 | -0.03% | 15.103 | 14.679 | -2.81% | reducer |
| Dense MTP4 verify bs3 / 8K | 7.095 | 7.130 | +0.50% | 7.881 | 7.920 | +0.49% | reducer |
| Dense MTP4 verify bs3 / 32K | 9.099 | 9.173 | +0.81% | 8.757 | 8.804 | +0.54% | reducer |
| Dense MTP4 verify bs8 / 32K | 12.726 | 12.624 | -0.80% | 11.701 | 11.823 | +1.04% | reducer |
| Dense MTP4 verify bs3 / 220K | 33.572 | 33.713 | +0.42% | 24.102 | 24.085 | -0.07% | reducer |
| Sparse decode 4 rows | 4.699 | 4.689 | -0.20% | 4.829 | 4.931 | +2.12% | cluster |
| Sparse decode 64 rows | 5.643 | 5.562 | -1.43% | 5.562 | 5.600 | +0.68% | cluster |
| Sparse decode 128 rows | 8.018 | 8.083 | +0.82% | 7.753 | 7.765 | +0.16% | cluster |
| Sparse decode 256 rows | 14.644 | 14.662 | +0.12% | 14.128 | 14.180 | +0.37% | reducer |
| Sparse verify 3×4=12 rows | 4.776 | 4.751 | -0.52% | 4.995 | 4.952 | -0.85% | cluster |
| Sparse prefill 1024 rows | 15.371 | 15.260 | -0.72% | 14.392 | 14.444 | +0.36% | direct |
| Sparse prefill 4096 rows | 109.803 | 109.264 | -0.49% | 71.176 | 71.158 | -0.03% | direct |
| Sparse prefill 8192 rows | 252.876 | 253.048 | +0.07% | 162.361 | 162.196 | -0.10% | direct |
| Cluster boundary 239 rows, cap2 | 13.519 | 13.713 | +1.43% | 13.351 | 13.367 | +0.12% | cluster |
| Cluster boundary 240 rows, cap2 | 13.551 | 13.771 | +1.62% | 13.318 | 13.441 | +0.92% | cluster |
| Cluster boundary 241 rows, cap2 | 14.694 | 14.736 | +0.28% | 14.345 | 14.554 | +1.45% | reducer |

## PR 配置：完整矩阵

Dense 使用精确 table 和实际 context bound；其余使用默认 planner cap。

时间单位为 µs；变化为 `新版 / 上一版 − 1`，正数表示变慢。两版输出路径相同。

| 场景 | BF16 上一版 | BF16 新版 | BF16 变化 | FP8 上一版 | FP8 新版 | FP8 变化 | 输出路径 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Dense decode bs1 / 8K | 5.434 | 5.440 | +0.12% | 5.213 | 5.222 | +0.17% | reducer |
| Dense decode bs4 / 8K | 5.670 | 5.659 | -0.20% | 5.423 | 5.451 | +0.50% | reducer |
| Dense decode bs8 / 8K | 6.081 | 6.031 | -0.83% | 5.595 | 5.664 | +1.23% | reducer |
| Dense decode bs32 / 8K | 9.372 | 9.279 | -0.99% | 7.721 | 7.690 | -0.40% | reducer |
| Dense decode bs4 / 32K | 7.401 | 7.498 | +1.30% | 6.717 | 6.708 | -0.13% | reducer |
| Dense decode bs3 / 220K | 29.755 | 29.805 | +0.17% | 14.797 | 14.676 | -0.81% | reducer |
| Dense MTP4 verify bs3 / 8K | 6.495 | 6.515 | +0.30% | 7.301 | 7.279 | -0.30% | reducer |
| Dense MTP4 verify bs3 / 32K | 9.099 | 9.121 | +0.24% | 8.812 | 8.764 | -0.55% | reducer |
| Dense MTP4 verify bs8 / 32K | 12.705 | 12.622 | -0.66% | 11.974 | 11.650 | -2.70% | reducer |
| Dense MTP4 verify bs3 / 220K | 34.077 | 33.686 | -1.15% | 23.430 | 24.568 | +4.86% | reducer |
| Sparse decode 4 rows | 4.688 | 4.715 | +0.58% | 4.902 | 4.920 | +0.38% | cluster |
| Sparse decode 64 rows | 5.652 | 5.673 | +0.37% | 5.604 | 5.656 | +0.94% | cluster |
| Sparse decode 128 rows | 9.469 | 9.591 | +1.29% | 9.260 | 9.263 | +0.03% | reducer |
| Sparse decode 256 rows | 14.634 | 14.688 | +0.37% | 14.246 | 14.253 | +0.04% | reducer |
| Sparse verify 3×4=12 rows | 4.770 | 4.790 | +0.42% | 4.984 | 4.972 | -0.24% | cluster |
| Sparse prefill 1024 rows | 14.978 | 14.994 | +0.11% | 13.697 | 13.654 | -0.31% | direct |
| Sparse prefill 4096 rows | 111.580 | 111.834 | +0.23% | 70.927 | 71.143 | +0.31% | direct |
| Sparse prefill 8192 rows | 258.354 | 259.471 | +0.43% | 163.032 | 163.617 | +0.36% | direct |
| Cluster boundary 239 rows, cap2 | 13.467 | 13.584 | +0.86% | 13.436 | 13.451 | +0.11% | cluster |
| Cluster boundary 240 rows, cap2 | 13.584 | 13.630 | +0.34% | 13.419 | 13.358 | -0.45% | cluster |
| Cluster boundary 241 rows, cap2 | 14.679 | 14.732 | +0.36% | 14.347 | 14.571 | +1.56% | reducer |

## 新版相对 main 的收益

main 是合入 PR 的 `6d8c8e9c2`，一次完整进程；新版取三个进程的中位数。
下面对 18 个常规形状等权统计，**不含 239/240/241 cluster 边界点**。速度比为 `main / 新版`。

| Profile | KV | 速度比范围 | 等权几何平均速度比 |
| --- | --- | ---: | ---: |
| atom | bf16 | 1.006–1.781× | 1.230× |
| atom | fp8 | 1.001–1.434× | 1.205× |
| pr | bf16 | 1.024–1.750× | 1.254× |
| pr | fp8 | 1.013–1.402× | 1.243× |

## 逻辑 IO 吞吐

JSON/CSV 的 `logical_io_throughput_tb_s` 按 logical KV、FP8 scale、Q read 和 output write 的字节量除以完整 PA 延迟计算。
Sparse 行共享物理 KV pool，graph 反复访问相同输入，重复逻辑引用可以经缓存复用。
此指标是十进制 **逻辑 IO TB/s**；本次没有采集 HBM counters，不能作为物理显存读写带宽。
以下为新版 ATOM 配置的示例；所有配置均记录在 summary JSON/CSV。

| 场景 | BF16 时间 (µs) | BF16 逻辑 IO (TB/s) | FP8 时间 (µs) | FP8 逻辑 IO (TB/s) |
| --- | ---: | ---: | ---: | ---: |
| Dense decode bs4 / 8K | 5.969 | 2.816 | 5.725 | 1.517 |
| Dense MTP4 verify bs3 / 220K | 33.713 | 10.267 | 24.085 | 7.412 |
| Sparse decode 64 rows | 5.562 | 12.160 | 5.600 | 6.273 |
| Sparse prefill 8192 rows | 253.048 | 29.970 | 162.196 | 24.310 |

## 复现与原始记录

使用同一份 benchmark，`--source` 指定实现。每版运行三个独立进程，并按前述顺序交替。
完整 argv、环境、起止 UTC 时间与退出码记录于下面的 commands JSON；第一组完整对照可这样复现：

```bash
git fetch origin refs/pull/7079/head
git worktree add --detach /tmp/pa7079-old 15214dad75fc26d13402b1571de080cf56c514f3
git worktree add --detach /tmp/pa7079-new 964b5d4d8f7085497238f69fface36cb90ef36b2

HIP_VISIBLE_DEVICES=3 FLYDSL_RUNTIME_CACHE_DIR=/tmp/pa7079-cache \
python op_tests/op_benchmarks/flydsl/bench_pa_minimax_m3.py \
  --source /tmp/pa7079-old --profile both --rounds 5 \
  --suite dense-decode dense-verify sparse-decode sparse-verify sparse-prefill cluster-boundary \
  --label previous --output /tmp/pa7079-previous.json

HIP_VISIBLE_DEVICES=3 FLYDSL_RUNTIME_CACHE_DIR=/tmp/pa7079-cache \
python op_tests/op_benchmarks/flydsl/bench_pa_minimax_m3.py \
  --source /tmp/pa7079-new --profile both --rounds 5 \
  --suite dense-decode dense-verify sparse-decode sparse-verify sparse-prefill cluster-boundary \
  --label head --compare-to /tmp/pa7079-previous.json --output /tmp/pa7079-head.json
```

- [summary.json](summary.json) / [summary.csv](summary.csv)：84 点的三进程汇总、单独复测、每进程结果及波动范围。
- [previous.json](previous.json)、[repeat-previous-2.json](repeat-previous-2.json)、[repeat-previous-3.json](repeat-previous-3.json)：上一版三遍完整矩阵。
- [head.json](head.json)、[repeat-head-2.json](repeat-head-2.json)、[repeat-head-3.json](repeat-head-3.json)：新版三遍完整矩阵。
- [main.json](main.json)：main 完整矩阵对照。
- `recheck-{previous,head}-{1,2,3}.json`：两个 decode 形状复测；`verify-recheck-{previous,head}-{1,2,3}.json`：220K verify 复测。
- 每份原始结果均保留 CSV 和 stdout log；JSON 包含全部 GPU-event 样本、plan、reference error、源码提交。
- [commands.json](commands.json)、[repeat-commands.json](repeat-commands.json)、[recheck-commands.json](recheck-commands.json)、[verify-recheck-commands.json](verify-recheck-commands.json)：全部命令和退出码。
- [run-info.json](run-info.json)：实现文件 SHA256 与 FlyDSL 起止版本。
- `*-gpu-start.txt`、`*-gpu-end.txt`、[gpu-end.txt](gpu-end.txt)、[gpu3-monitor.csv](gpu3-monitor.csv)、[gpu3-repeat-monitor.csv](gpu3-repeat-monitor.csv)：GPU 起止状态与占用采样。
- [summarize.py](summarize.py)：用标准 Python 库重新生成 summary JSON/CSV，并检查完整性、正确性标记和对照参数。
- [SHA256SUMS](SHA256SUMS)：本次发布文件校验和。

此前 [2026-11-03 延迟和逻辑 IO 报告](../minimax_m3_gfx1250_20261103/README.md) 的原始结果和 FETCH_SIZE counter 说明继续保留。
