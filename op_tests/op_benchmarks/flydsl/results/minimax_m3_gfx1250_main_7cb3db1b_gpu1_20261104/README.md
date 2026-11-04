# MiniMax-M3 TP4：最新 main，gfx1250 GPU 1 benchmark

已用物理 **GPU 1** 更新到 main `7cb3db1b7`，并在同一卡上重测此前 benchmark 对应的 `15214dad7`。
两版各完成三遍 84 项完整矩阵；main 的等权几何平均延迟变化为 **+0.29%**，各项范围为 **-2.37% 至 +3.32%**，汇总中没有超过 5% 的变慢项。

**另有两次间歇性 GPU memory fault，均发生在处理 BF16 dense MTP4 verify、batch 3、query length 4、context 225280 的过程中。**
首轮完整矩阵和后续针对性复测各退出一次（SIGABRT，Python subprocess 返回 −6）。成功的重复进程没有消除这两次失败；具体 kernel、出错阶段和根因仍未确认。
下面保留完整矩阵的性能数据、所有已完成测量点和失败进程，避免把时延筛查当作稳定性结论。

## 环境与版本

| 项目 | 设置 |
| --- | --- |
| 最新 main | `7cb3db1b762a8c6637609208b6a1d99fb4127515`，2026-11-04 fetch |
| 旧版对照 | `15214dad75fc26d13402b1571de080cf56c514f3`，原 benchmark 分支 `55b1dde0f` 的 PA 源码与其相同 |
| PR 7384 前的诊断版本 | `e4973c4bed57a615068089fa02c88c64d644fd42` |
| GPU | 物理 GPU 1，gfx1250，256 CUs，432 GiB VRAM |
| GPU UUID | `34313632-6132-3030-6533-623732653737` |
| PyTorch / HIP | `2.11.0+rocm10.1.0a20260811-bkc.20260820` / `7.16.26315` |
| FlyDSL | `0.4.1`，源码 `f37f7a2b98f295902240f9f96612222003578b40` |
| GPU 隔离 | `HIP_VISIBLE_DEVICES=1`，进程内 device 0 |
| 架构配置 | `AITER_RUNTIME_GPU_ARCH=gfx1250`，`FLYDSL_GPU_ARCH=gfx1250` |
| 其他环境 | `AITER_USE_SYSTEM_TRITON=1`；kernarg preload 和 autotune 无额外覆盖 |
| 输入 | Hq16 / Hkv1 / D128；Q/O BF16；KV BF16 或 FP8 E4M3FN + FP32 per-token K/V scales |
| Dense | page128；decode qlen1；verify qlen4，causal mask |
| Sparse | page16；2048 selected tokens；4096-page shared KV pool |
| Prefill | 第 r 行 context 为 `min(r+1, 2048)` |
| 随机种子 | 37 |
| Harness SHA256 | `9f9a09e0f832617ef38131af636ee472e7009ad05ec884b55b530939a406aa92` |

benchmark 分支合入 main 后，PA 实现与上述 main revision 一致。
本轮涵盖 main 已合入的 PR 7079，并使用包含 MXFP4 集成、gfx1250-strict 架构适配和 PR 7384 pipeline 改动的源码。
本矩阵仍是 **D128、BF16/FP8 KV**，未测 MXFP4、D64 或 gfx1250-strict 运行配置。
旧版和最新 main 均在独立、干净的 frozen checkout 中导入；主矩阵起止源码 SHA256 与 FlyDSL 状态记录在 [run-info.json](run-info.json)。

GPU 1 起始利用率为 0，VRAM 基础占用约 173 MiB；每个主矩阵和针对性复测进程均保存 GPU 起止状态。
只在该卡串行运行，没有修改时钟；主矩阵另有 5 秒间隔的 [gpu1-monitor.jsonl](gpu1-monitor.jsonl) 占用采样。
benchmark 新增设备可见性和架构环境记录，对照会检查这些参数一致。

## 计时、正确性与进程记录

使用同一份 [benchmark](../../bench_pa_minimax_m3.py)。每个 graph 包含 10 次完整 PA 调用；每个 GPU-event 样本测 5 次 replay，除以 50 次调用。
每轮取 5 个样本的最小值，每个进程取 **5 轮最小值的中位数**；主表再取 **3 个完整进程的中位数**。
计时包含 partition reduction；planning、量化、分配、编译和 reference 检查均在计时外。

首轮 `previous → main` 在 main 中退出，原始文件保存在 `initial-attempt/`；随后单独复测出错形状通过。
主表使用接着执行的以下完整序列，首轮退出及其前面的成功点不进入主表：

```text
previous-1 → main-1 → main-2 → previous-2 → previous-3 → main-3
```

针对性复测按相同顺序尝试三个旧版和三个 main 进程，每个进程包含五个 dense 形状、两个 profile 和两个 dtype（20 点）。
`recheck-main-2` 在完成四点后退出；剩余预定进程继续完成。此组保留 3 个完整旧版、2 个完整 main 及 1 个失败 main。
诊断另交替运行 PR 7384 前 main 与最新 main，各五个 ATOM BF16 进程（每进程五形状）。

| 记录 | 已完成且检查通过的测量点 | 进程状态 |
| --- | ---: | --- |
| 主矩阵：旧版三遍、main 三遍 | 504 | 6/6 完成 |
| 针对性复测 | 104 | 旧版 3/3 完成；main 2/3 完成，失败进程保存 4 点 |
| PR 7384 前 main / 最新 main 诊断 | 50 | 两版各 5/5 完成 |
| 首轮及一次单独复测 | 94 | 旧版 84 点、失败 main 的 9 点、单独 main 的 1 点 |
| 总计 | **752** | **另有 2 次 GPU memory fault，触发点没有结果** |

每个已完成点检查 eager、graph replay，以及 Q 取反、output 填 NaN 后的 replay。
BF16 使用 `rtol=atol=0.005`；FP8 检查每行 relative L2 ≤ 0.05，观测最大值为 **0.03444159**。
汇总脚本核对完整进程的 84 点、干净源码、设备/环境/输入一致性，并把失败进程的 partial JSON 与退出码单独记录。

## 两次 GPU memory fault

| 进程 | 最后完成的形状 | 下一项 | 退出码 | Fault address |
| --- | --- | --- | ---: | --- |
| 首轮 main | ATOM BF16 dense verify bs8 / 32K | ATOM BF16 dense verify bs3 / 220K | −6 | `0x92000` |
| `recheck-main-2` | ATOM BF16 dense verify bs8 / 32K | ATOM BF16 dense verify bs3 / 220K | −6 | `0xbb000` |

两份日志都报告 `Memory access fault by GPU node-3 ... Page not present or supervisor privilege`；node-3 对应本次隔离的物理 GPU 1。
详见 [首轮日志](initial-attempt/main-1.log) 和 [针对性复测日志](recheck-main-2.log)。
同一 ATOM BF16 长 context 形状也在后续三遍主矩阵、一次单独检查、两遍完整针对性复测和五遍诊断中通过。

PR 7384 前版本与最新 main 的 PA 差异仅为 `pipeline.py` 和 `traits.py` 中该 PR 的改动，但本次各五遍诊断都通过，未定位因果关系。
现有 stdout 没有标出失败是在输入构造、PA、reference 还是 graph 的哪个阶段，不能把错误归因于特定 kernel 或 PR。
保留失败结果的 `status=running`，不改成 complete，也不将未测到的失败点填成延迟数据。

## 完整矩阵的延迟变化

以下按 18 个常规形状等权统计（每组不含 239/240/241 boundary 点）。正的延迟变化表示变慢，速度比为 `旧版 / main`。
所有 84 项的几何平均变化为 +0.29%；72 项常规配置为 +0.30%。
5% 仅为明显时延回退的工程筛查阈值，不代表统计置信区间，也不覆盖上面的内存访问错误。

| Profile | KV | 等权几何平均延迟变化 | 速度比范围 |
| --- | --- | ---: | ---: |
| atom | bf16 | +0.27% | 0.971–1.010× |
| atom | fp8 | +0.23% | 0.972–1.011× |
| pr | bf16 | +0.66% | 0.968–1.024× |
| pr | fp8 | +0.03% | 0.977–1.023× |

## ATOM：最新 main 延迟与逻辑 IO 吞吐

Dense table 补齐到 256K，context bound 为 None；sparse 使用推荐 split cap，prefill cap1。
时间为三进程汇总；变化相对本次 GPU 1 旧版汇总。两版输出路径相同。

| 场景 | BF16 µs | BF16 IO TB/s | BF16 延迟变化 | FP8 µs | FP8 IO TB/s | FP8 延迟变化 | 输出路径 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Dense decode bs1 / 8K | 5.578 | 0.753 | +0.13% | 5.408 | 0.401 | +0.28% | reducer |
| Dense decode bs4 / 8K | 5.928 | 2.836 | -0.01% | 5.773 | 1.504 | +1.15% | reducer |
| Dense decode bs8 / 8K | 6.362 | 5.285 | +0.74% | 5.958 | 2.915 | -0.08% | reducer |
| Dense decode bs32 / 8K | 9.727 | 13.825 | +2.98% | 7.885 | 8.811 | +2.86% | reducer |
| Dense decode bs4 / 32K | 7.478 | 8.978 | +0.01% | 6.795 | 5.097 | +0.89% | reducer |
| Dense decode bs3 / 220K | 29.312 | 11.806 | +0.10% | 14.719 | 12.124 | -1.08% | reducer |
| Dense MTP4 verify bs3 / 8K | 7.160 | 1.771 | +0.95% | 7.834 | 0.841 | -0.46% | reducer |
| Dense MTP4 verify bs3 / 32K | 9.151 | 5.511 | +0.75% | 8.818 | 2.954 | +0.63% | reducer |
| Dense MTP4 verify bs8 / 32K | 12.789 | 10.515 | +0.91% | 11.905 | 5.835 | +0.24% | reducer |
| Dense MTP4 verify bs3 / 220K | 33.458 | 10.345 | -0.29% | 23.465 | 7.608 | +0.09% | reducer |
| Sparse decode 4 rows | 4.616 | 0.916 | +0.51% | 4.818 | 0.456 | -0.02% | cluster |
| Sparse decode 64 rows | 5.540 | 12.207 | -0.22% | 5.633 | 6.236 | +0.07% | cluster |
| Sparse decode 128 rows | 8.032 | 16.841 | -0.11% | 7.738 | 9.079 | -0.04% | cluster |
| Sparse decode 256 rows | 14.779 | 18.305 | -0.31% | 14.422 | 9.742 | +0.84% | reducer |
| Sparse verify 3×4=12 rows | 4.762 | 2.663 | +0.56% | 4.947 | 1.331 | -0.19% | cluster |
| Sparse prefill 1024 rows | 15.207 | 18.221 | -0.99% | 14.446 | 10.172 | -0.07% | direct |
| Sparse prefill 4096 rows | 109.962 | 29.604 | -0.27% | 71.958 | 23.552 | -0.51% | direct |
| Sparse prefill 8192 rows | 253.666 | 29.897 | -0.49% | 162.235 | 24.304 | -0.39% | direct |
| Cluster boundary 239 rows, cap2 | 13.621 | 18.542 | +0.35% | 13.657 | 9.605 | +0.67% | cluster |
| Cluster boundary 240 rows, cap2 | 13.808 | 18.368 | +0.93% | 13.604 | 9.683 | +1.34% | cluster |
| Cluster boundary 241 rows, cap2 | 14.810 | 17.196 | +0.01% | 14.415 | 9.176 | -0.13% | reducer |

## PR profile：最新 main 延迟与逻辑 IO 吞吐

Dense 使用精确 table 和实际 context bound；其余使用默认 planner cap。`pr` 保留 benchmark 的历史 profile 名称，实际测量源码为最新 main。

| 场景 | BF16 µs | BF16 IO TB/s | BF16 延迟变化 | FP8 µs | FP8 IO TB/s | FP8 延迟变化 | 输出路径 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Dense decode bs1 / 8K | 5.439 | 0.773 | +0.09% | 5.205 | 0.417 | -0.14% | reducer |
| Dense decode bs4 / 8K | 5.697 | 2.951 | +0.45% | 5.459 | 1.591 | +0.00% | reducer |
| Dense decode bs8 / 8K | 6.116 | 5.497 | +2.11% | 5.593 | 3.105 | -2.21% | reducer |
| Dense decode bs32 / 8K | 9.408 | 14.295 | -2.37% | 7.910 | 8.782 | +2.31% | reducer |
| Dense decode bs4 / 32K | 7.594 | 8.842 | +3.32% | 6.765 | 5.120 | -1.34% | reducer |
| Dense decode bs3 / 220K | 29.450 | 11.751 | +0.42% | 14.797 | 12.060 | -0.28% | reducer |
| Dense MTP4 verify bs3 / 8K | 6.471 | 1.960 | +0.11% | 7.223 | 0.912 | -0.42% | reducer |
| Dense MTP4 verify bs3 / 32K | 9.393 | 5.369 | +2.66% | 8.752 | 2.976 | +0.27% | reducer |
| Dense MTP4 verify bs8 / 32K | 12.928 | 10.402 | +3.14% | 11.956 | 5.810 | +0.50% | reducer |
| Dense MTP4 verify bs3 / 220K | 33.584 | 10.306 | +0.16% | 23.773 | 7.509 | +1.12% | reducer |
| Sparse decode 4 rows | 4.685 | 0.902 | -0.27% | 4.939 | 0.445 | +0.37% | cluster |
| Sparse decode 64 rows | 5.553 | 12.179 | +0.09% | 5.633 | 6.236 | -0.61% | cluster |
| Sparse decode 128 rows | 9.631 | 14.045 | +1.77% | 9.288 | 7.564 | +0.28% | reducer |
| Sparse decode 256 rows | 14.942 | 18.105 | +1.05% | 14.450 | 9.723 | +0.88% | reducer |
| Sparse verify 3×4=12 rows | 4.742 | 2.675 | +0.07% | 4.980 | 1.322 | +0.13% | cluster |
| Sparse prefill 1024 rows | 15.204 | 18.225 | -0.18% | 13.724 | 10.707 | -0.35% | direct |
| Sparse prefill 4096 rows | 111.925 | 29.085 | -0.54% | 71.995 | 23.540 | +0.09% | direct |
| Sparse prefill 8192 rows | 258.978 | 29.284 | -0.10% | 163.962 | 24.048 | -0.02% | direct |
| Cluster boundary 239 rows, cap2 | 13.584 | 18.594 | +0.13% | 13.572 | 9.665 | -0.06% | cluster |
| Cluster boundary 240 rows, cap2 | 13.677 | 18.545 | -0.76% | 13.677 | 9.632 | +1.46% | cluster |
| Cluster boundary 241 rows, cap2 | 14.817 | 17.188 | +0.46% | 14.349 | 9.219 | -1.26% | reducer |

## 针对性复测的已完成进程

五个形状选自波动较大的 dense 配置和两次 fault 的触发项。
下表仅汇总 **3 个完整旧版进程与 2 个完整 main 进程**；每行 JSON 记录实际样本数。
失败的 `recheck-main-2` 不进入这张时延表，部分已完成点和退出记录仍保存。
此表用于观察成功进程的时延，主表仍使用三对三完整矩阵，稳定性判断同时考虑失败。

| 场景 | Profile | KV | 旧版 µs | main µs | 延迟变化 |
| --- | --- | --- | ---: | ---: | ---: |
| Dense decode bs8 / 8K | atom | bf16 | 6.285 | 6.299 | +0.22% |
| Dense decode bs32 / 8K | atom | bf16 | 9.430 | 9.500 | +0.74% |
| Dense decode bs4 / 32K | atom | bf16 | 7.326 | 7.260 | -0.90% |
| Dense MTP4 verify bs8 / 32K | atom | bf16 | 12.641 | 12.513 | -1.01% |
| Dense MTP4 verify bs3 / 220K | atom | bf16 | 33.453 | 33.242 | -0.63% |
| Dense decode bs8 / 8K | atom | fp8 | 5.946 | 5.932 | -0.23% |
| Dense decode bs32 / 8K | atom | fp8 | 7.909 | 7.879 | -0.38% |
| Dense decode bs4 / 32K | atom | fp8 | 6.691 | 6.722 | +0.47% |
| Dense MTP4 verify bs8 / 32K | atom | fp8 | 11.834 | 11.929 | +0.81% |
| Dense MTP4 verify bs3 / 220K | atom | fp8 | 23.106 | 22.883 | -0.96% |
| Dense decode bs8 / 8K | pr | bf16 | 6.107 | 5.932 | -2.85% |
| Dense decode bs32 / 8K | pr | bf16 | 9.399 | 9.331 | -0.73% |
| Dense decode bs4 / 32K | pr | bf16 | 7.554 | 7.445 | -1.45% |
| Dense MTP4 verify bs8 / 32K | pr | bf16 | 12.553 | 12.489 | -0.51% |
| Dense MTP4 verify bs3 / 220K | pr | bf16 | 33.308 | 33.366 | +0.18% |
| Dense decode bs8 / 8K | pr | fp8 | 5.616 | 5.697 | +1.43% |
| Dense decode bs32 / 8K | pr | fp8 | 7.750 | 7.750 | +0.01% |
| Dense decode bs4 / 32K | pr | fp8 | 6.753 | 6.831 | +1.15% |
| Dense MTP4 verify bs8 / 32K | pr | fp8 | 11.765 | 11.893 | +1.09% |
| Dense MTP4 verify bs3 / 220K | pr | fp8 | 23.160 | 23.367 | +0.89% |

例如 PR BF16 decode bs4/32K 在主矩阵中为 +3.32%，该组成功复测为 −1.45%；PR BF16 verify bs8/32K 从 +3.14% 变为 −0.51%。
小幅时延差异随进程和前序 case 变化，本次没有确定具体原因。原始样本、每进程中位数和 spread 保留在 summary JSON。

## 逻辑 IO 吞吐口径

表中的 IO TB/s 为十进制逻辑 tensor IO，沿用 [benchmark 定义](../../README_pa_minimax_m3.md#logical-io-throughput)：

```text
N = sum(context_lengths)
KV payload = 2 * N * 1 * 128 * KV element size
KV scales  = 2 * N * 1 * 4 for FP8; zero for BF16
Q + output = 2 * batch * query_length * 16 * 128 * 2
Logical IO TB/s = (KV payload + KV scales + Q/output) / latency_us / 1e6
```

Dense MTP4 共用 KV，每个 request 计一次；sparse 行共享 physical KV pool，但每行的逻辑访问分别计数。
Graph 反复访问同一输入，缓存复用可使逻辑吞吐超过物理显存带宽。本轮没有采集 HBM counters；metadata、scratch 和重复 kernel 读取不计入逻辑字节数。

## 复现与文件

使用两个 clean worktree 和同一份 harness，在物理 GPU 1 串行运行：

```bash
git fetch origin main
git worktree add --detach /tmp/m3-previous 15214dad75fc26d13402b1571de080cf56c514f3
git worktree add --detach /tmp/m3-main 7cb3db1b762a8c6637609208b6a1d99fb4127515

python op_tests/op_benchmarks/flydsl/results/minimax_m3_gfx1250_main_7cb3db1b_gpu1_20261104/run.py \
  --harness op_tests/op_benchmarks/flydsl/bench_pa_minimax_m3.py \
  --previous /tmp/m3-previous --main /tmp/m3-main \
  --flydsl /path/to/FlyDSL --output-root /tmp/m3-main-gpu1-new-run
```

`run.py` 自动设置 `HIP_VISIBLE_DEVICES=1`，采用上述三对三顺序，失败即停止并保存命令、日志和源码状态。
针对性与诊断复测的完整 argv 可在 commands JSON 中逐条重放；复现失败点可用：

```bash
HIP_VISIBLE_DEVICES=1 python op_tests/op_benchmarks/flydsl/bench_pa_minimax_m3.py \
  --source /tmp/m3-main --profile atom --dtype bf16 --rounds 5 \
  --case dense-decode-b8-q1-ctx8192 --case dense-decode-b32-q1-ctx8192 \
  --case dense-decode-b4-q1-ctx32768 --case dense-verify-b8-q4-ctx32768 \
  --case dense-verify-b3-q4-ctx225280 --output /tmp/m3-main-fault-recheck.json
```

该问题间歇出现，单次命令可能通过；保留退出码和日志。两次失败时使用的相同 cache 路径和完整环境见 commands JSON。

- [summary.json](summary.json) / [summary.csv](summary.csv)：84 点主汇总、针对性成功进程汇总、退出记录与样本数。
- `previous-{1,2,3}.json`、`main-{1,2,3}.json`：六遍完整矩阵，每份另附 CSV 和 stdout log。
- `recheck-{previous,main}-{1,2,3}.json`：针对性复测，`recheck-main-2.json` 保留 running 状态和 4 个已完成点。
- `diagnostic-{before7384,main}-{1,2,3,4,5}.json`：前后版本的 ATOM BF16 诊断。
- [commands.json](commands.json)、[recheck-commands.json](recheck-commands.json)、[diagnostic-commands.json](diagnostic-commands.json)：完整 argv、环境、UTC 起止和退出码。
- [initial-attempt/commands.json](initial-attempt/commands.json)、`initial-attempt/main-1.{json,log}`：首轮失败证据；该目录另保存旧版完整结果和出错形状的单独成功复测。
- [run-info.json](run-info.json)、`*-gpu-{start,end}.txt`、[gpu1-monitor.jsonl](gpu1-monitor.jsonl)：源码与软件状态、GPU 起止和占用记录。
- [run.py](run.py)、[summarize.py](summarize.py)：主矩阵 runner 与标准库汇总脚本；后者可重新生成 summary 并核对数据一致性。
- [SHA256SUMS](SHA256SUMS)：全部发布文件校验和。

此前 [GPU 3 PR 7079 回退检查](../minimax_m3_gfx1250_pr7079_964b5d4d_20261103/README.md) 和 [原始吞吐报告](../minimax_m3_gfx1250_20261103/README.md) 保留为历史记录，不直接与本轮 GPU 1 的延迟做比值。
