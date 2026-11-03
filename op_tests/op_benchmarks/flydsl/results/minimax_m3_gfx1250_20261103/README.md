# MiniMax-M3 TP4 FlyDSL PA 性能：gfx1250，2026-11-03

本次实测覆盖 18 个 MiniMax-M3 TP4 attention 形状，分别运行 BF16/FP8 KV、ATOM/PR 两种配置，另加 239/240/241 行 cluster 边界，共 **84 个测量点**。所有测量点通过参考结果和 CUDA graph 更新输入检查。主结果采用 ATOM 调用配置。

测量对象是一次 `aiter.ops.flydsl.pa_decode.pa_decode` 完整调用，包括需要的 partition reduction。输入为合成的 attention 张量；测试范围是 PA 算子，模型其他算子和多层执行时间不包含在下面的延迟中。

## 测试环境

| 项目 | 设置 |
| --- | --- |
| 测量开始时间 | 2026-11-03T00:15:33.132948+00:00 |
| 测量源码提交 | `e5403f61e732febe599860999db64935ea5c56c4`，测量时工作区干净 |
| PA 实现 | PR 7079 head `15214dad75fc26d13402b1571de080cf56c514f3`；上述提交仅增加 benchmark 与文档 |
| GPU | GPU 0，AMD Radeon Graphics，`gfx1250`，256 CUs，432 GiB VRAM |
| GPU UUID | `32636139-6565-3437-3732-643939363430` |
| PyTorch | `2.11.0+rocm10.1.0a20260811-bkc.20260820` |
| HIP | `7.16.26315` |
| FlyDSL | `0.4.1` |
| Attention | 本卡 Hq=16，Hkv=1，D=128，GQA=16；Q/O 为 BF16 |
| KV | BF16 无 scale；FP8 为 E4M3FN，K/V 各带 FP32 per-token scale |
| Dense | page128；decode qlen1，verify qlen4，共享请求 KV、使用 causal mask |
| Sparse | page16，16 个 logical128 blocks（2048 tokens），4096-page shared KV pool；verify 每个 query 独立一行、qlen1 |
| Prefill | 第 r 行的有效 context 为 `min(r+1, 2048)`，r 从 0 开始 |
| Plan | 默认 budget=512；ATOM sparse cap 来自 `get_recommended_splits(rows, 1)`，prefill cap1 |
| GPU 状态 | 运行前后检查为空闲；串行测试 GPU 0，未锁定/修改时钟。起止 GPU 状态见原始记录 |
| 随机种子 | 37，同形状在两个 profile 使用相同输入 |
| Harness SHA256 | `d0aa3073c901f5cbd481876519a59c96edef472c0c7b2139c6b22451563c45b9` |

## 计时与带宽口径

每个 CUDA graph 含 10 次完整 PA 调用。每个样本用 GPU events 计时 5 次 graph replay，再除以 50 次调用。每轮取 5 个样本的最小值，表中时间为 **5 轮最小值的中位数**；eager 和 graph 各预热 5 次。编译、plan 构建、KV 量化、table 构建、分配和正确性检查均在计时外。

表中带宽是 **逻辑张量 IO 的有效带宽**，使用十进制 TB/s。设 `N=sum(context_lengths)`，`T=batch×query_length`：

```text
KV payload bytes = 2 × N × Hkv × D × sizeof(KV)
KV scale bytes   = 2 × N × Hkv × 4（FP8）；BF16 为 0
Q + output bytes = 2 × T × Hq × D × 2
Logical IO bytes = KV payload bytes + KV scale bytes + Q/output bytes
Effective TB/s   = Logical IO bytes / latency_us / 10^6
```

Dense MTP4 的 KV 按每个请求计一次，4 个 query 共享 KV。Sparse 按每行有效 context 累加逻辑 KV，即使多行引用相同物理页；prefill 使用实际递增长度。统计不含 block table、plan metadata、scratch 或额外 kernel 读取。

Graph 重复读取相同输入，sparse 行共享 KV pool，所以该指标包含缓存复用效果，不能解释为硬件 HBM 实测吞吐，也不用于计算 HBM 峰值利用率。尤其是 4096/8192 行 sparse prefill，逻辑带宽可明显高于物理显存吞吐。

## ATOM 调用配置：主要结果

Dense table 补齐到 256K tokens，`max_context_length=None`。Sparse decode/verify 的 bound 为 2048，cap 在 4/64 行为 8、128 行为 4、256 行为 2；prefill cap1。

| 场景 | BF16 时间 (µs) | BF16 有效带宽 (TB/s) | FP8 时间 (µs) | FP8 有效带宽 (TB/s) | 输出路径 BF16/FP8 |
| --- | ---: | ---: | ---: | ---: | --- |
| Dense decode，bs1，ctx8192 | 5.515 | 0.762 | 5.362 | 0.405 | reducer |
| Dense decode，bs4，ctx8192 | 5.887 | 2.855 | 5.643 | 1.539 | reducer |
| Dense decode，bs8，ctx8192 | 6.296 | 5.340 | 5.866 | 2.961 | reducer |
| Dense decode，bs32，ctx8192 | 9.445 | 14.238 | 7.605 | 9.135 | reducer |
| Dense decode，bs4，ctx32768 | 7.269 | 9.236 | 6.681 | 5.184 | reducer |
| Dense decode，bs3，ctx225280 | 29.981 | 11.542 | 14.872 | 11.999 | reducer |
| Dense MTP4 verify，bs3，ctx8192 | 7.183 | 1.766 | 7.868 | 0.837 | reducer |
| Dense MTP4 verify，bs3，ctx32768 | 9.163 | 5.503 | 8.797 | 2.961 | reducer |
| Dense MTP4 verify，bs8，ctx32768 | 12.368 | 10.873 | 11.933 | 5.821 | reducer |
| Dense MTP4 verify，bs3，ctx225280 | 33.909 | 10.208 | 24.349 | 7.332 | reducer |
| Sparse decode，4 rows，ctx2048 | 4.553 | 0.928 | 4.782 | 0.459 | cluster |
| Sparse decode，64 rows，ctx2048 | 5.616 | 12.042 | 5.555 | 6.324 | cluster |
| Sparse decode，128 rows，ctx2048 | 8.224 | 16.449 | 7.778 | 9.032 | cluster |
| Sparse decode，256 rows，ctx2048 | 14.735 | 18.360 | 14.145 | 9.933 | reducer |
| Sparse verify，3×4=12 rows，ctx2048 | 4.745 | 2.673 | 4.970 | 1.325 | cluster |
| Sparse prefill，1024 rows，causal ctx≤2048 | 15.329 | 18.076 | 14.795 | 9.931 | direct |
| Sparse prefill，4096 rows，causal ctx≤2048 | 109.794 | 29.649 | 71.318 | 23.764 | direct |
| Sparse prefill，8192 rows，causal ctx≤2048 | 256.508 | 29.566 | 162.979 | 24.193 | direct |

## PR 测量配置：补充结果

Dense 使用 exact-sized table 和实际 context bound；其余使用默认 planner cap/budget，与 PR 7079 主性能表的设置一致。两种配置的字节统计相同，plan/dispatch 可不同。

| 场景 | BF16 时间 (µs) | BF16 有效带宽 (TB/s) | FP8 时间 (µs) | FP8 有效带宽 (TB/s) | 输出路径 BF16/FP8 |
| --- | ---: | ---: | ---: | ---: | --- |
| Dense decode，bs1，ctx8192 | 5.397 | 0.779 | 5.157 | 0.421 | reducer |
| Dense decode，bs4，ctx8192 | 5.631 | 2.985 | 5.385 | 1.613 | reducer |
| Dense decode，bs8，ctx8192 | 6.043 | 5.563 | 5.563 | 3.122 | reducer |
| Dense decode，bs32，ctx8192 | 9.376 | 14.342 | 7.800 | 8.906 | reducer |
| Dense decode，bs4，ctx32768 | 7.465 | 8.994 | 6.684 | 5.182 | reducer |
| Dense decode，bs3，ctx225280 | 29.570 | 11.703 | 14.739 | 12.107 | reducer |
| Dense MTP4 verify，bs3，ctx8192 | 6.491 | 1.954 | 7.260 | 0.907 | reducer |
| Dense MTP4 verify，bs3，ctx32768 | 9.172 | 5.498 | 8.788 | 2.964 | reducer |
| Dense MTP4 verify，bs8，ctx32768 | 12.687 | 10.600 | 11.855 | 5.860 | reducer |
| Dense MTP4 verify，bs3，ctx225280 | 33.823 | 10.234 | 23.863 | 7.481 | reducer |
| Sparse decode，4 rows，ctx2048 | 4.641 | 0.911 | 4.866 | 0.451 | cluster |
| Sparse decode，64 rows，ctx2048 | 5.564 | 12.156 | 5.569 | 6.307 | cluster |
| Sparse decode，128 rows，ctx2048 | 9.694 | 13.954 | 9.272 | 7.577 | reducer |
| Sparse decode，256 rows，ctx2048 | 14.793 | 18.288 | 14.224 | 9.879 | reducer |
| Sparse verify，3×4=12 rows，ctx2048 | 4.722 | 2.685 | 4.979 | 1.323 | cluster |
| Sparse prefill，1024 rows，causal ctx≤2048 | 15.187 | 18.245 | 13.751 | 10.685 | direct |
| Sparse prefill，4096 rows，causal ctx≤2048 | 113.101 | 28.782 | 71.835 | 23.593 | direct |
| Sparse prefill，8192 rows，causal ctx≤2048 | 260.914 | 29.066 | 164.677 | 23.943 | direct |

## Cluster 驻留边界

以下固定 cap2，以隔离 256-CU gfx1250 的 240-cluster 驻留上限。239/240 行使用 cluster merge，241 行使用 reducer。ATOM 的默认 split heuristic 在 239–241 行返回 cap3，因此这些是显式构造的边界测试。

### ATOM profile，cap2

| 场景 | BF16 时间 (µs) | BF16 有效带宽 (TB/s) | FP8 时间 (µs) | FP8 有效带宽 (TB/s) | 输出路径 BF16/FP8 |
| --- | ---: | ---: | ---: | ---: | --- |
| 239 rows，cap2 | 13.242 | 19.073 | 13.132 | 9.989 | cluster |
| 240 rows，cap2 | 13.370 | 18.970 | 13.118 | 10.042 | cluster |
| 241 rows，cap2 | 14.789 | 17.221 | 14.370 | 9.205 | reducer |

### PR profile，cap2

| 场景 | BF16 时间 (µs) | BF16 有效带宽 (TB/s) | FP8 时间 (µs) | FP8 有效带宽 (TB/s) | 输出路径 BF16/FP8 |
| --- | ---: | ---: | ---: | ---: | --- |
| 239 rows，cap2 | 13.517 | 18.685 | 13.061 | 10.043 | cluster |
| 240 rows，cap2 | 13.643 | 18.590 | 12.972 | 10.155 | cluster |
| 241 rows，cap2 | 14.778 | 17.234 | 14.360 | 9.211 | reducer |

## 轮间变化与正确性

下表记录大 prefill 场景五轮最小值的范围，供判断重复测量波动。范围包含本次全部五轮；上面的主表仍使用中位数。

| Profile | KV | Prefill rows | 五轮最小值范围 (µs) | 中位数 (µs) | (max−min)/median |
| --- | --- | ---: | ---: | ---: | ---: |
| atom | bf16 | 4096 | 109.615–115.479 | 109.794 | 5.34% |
| atom | bf16 | 8192 | 240.003–256.996 | 256.508 | 6.62% |
| atom | fp8 | 4096 | 71.059–76.530 | 71.318 | 7.67% |
| atom | fp8 | 8192 | 151.433–166.191 | 162.979 | 9.06% |
| pr | bf16 | 4096 | 112.731–119.323 | 113.101 | 5.83% |
| pr | bf16 | 8192 | 244.099–261.272 | 260.914 | 6.58% |
| pr | fp8 | 4096 | 71.584–76.984 | 71.835 | 7.52% |
| pr | fp8 | 8192 | 152.694–167.202 | 164.677 | 8.81% |

正确性：BF16 使用 FP32 reference 的 `rtol=atol=0.005`；FP8 检查每个 query row 的相对 L2 误差 ≤0.05，本次最大值为 **0.034442**。全部 84 个测量点还通过了 negated-Q、NaN-poisoned output 的 graph replay 检查。

## 复现与原始记录

从该分支仓库根目录运行，选择同一可见 GPU：

```bash
python op_tests/op_benchmarks/flydsl/bench_pa_minimax_m3.py \
  --profile both --device 0 --rounds 5 \
  --suite dense-decode dense-verify sparse-decode sparse-verify sparse-prefill cluster-boundary \
  --label minimax-m3-gfx1250-20261103 \
  --output /tmp/minimax-m3-gfx1250-20261103/results.json
```

- [results.json](results.json)：版本、完整精度字段、字节组成、每轮/每样本时间、实际 plan 和输出路径。
- [results.csv](results.csv)：84 个测量点的时间、有效带宽、字节数和正确性结果。
- [gpu-start.txt](gpu-start.txt) / [gpu-end.txt](gpu-end.txt)：测量起止 GPU 使用率、显存和时钟快照。
- [Benchmark 使用说明](../../README_pa_minimax_m3.md)。

原始结果 JSON SHA256：`03e50533bbe65e05e2e63783b624a8b7c3f560f14db3218c2c66d72ff4ecb86c`。
