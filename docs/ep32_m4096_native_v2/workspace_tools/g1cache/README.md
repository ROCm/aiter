# EP32/M4096 GEMM1 B-scale NT

此处保存已撤回的B scale NT实验：在已有B权重NT基础上，B scale的BufferCopy32b增加Python整数NT属性2（仅use_nt=True路径）；计算、布局、调度、等待均未改变。配对改善量级很小，后续默认入口没有证明合计收益，当前mxfp4_gemm1_ep32.py已恢复B scale默认策略，kernel名没有snt。下表是撤回之前的历史捕获。

以下为同进程交替原amd bench，各实现3次预热、30个样本，512 MiB只读flush。每行独立比较；没有跨卡推导收益。

| 捕获 | HIP | 原FC1 µs | NT FC1 µs | 原合计TB/s | NT合计TB/s |
|---|---:|---:|---:|---:|---:|
| amd_pair_20261112_201309_hip5 | 5 | 68.6200 | 68.2610 | 4.6635 | 4.6784 |
| amd_pair_20261112_202657_hip7 | 7 | 69.1610 | 68.9605 | 4.6602 | 4.6710 |

两次FC1中位数下降0.20–0.36 µs，量级很小，不宣称强稳定提速。逐对差值中位数分别-0.06和-0.2995 µs，30对中分别17和18对更快。这两个微小配对收益未成为当前默认保留依据；5T仍主要需要GEMM2改进。其他GPU尝试停在idle gate，不计为性能测试。

正确性：实际amd dump的6,285,312个有效FP8元素和全部scale，三次重放逐位一致。ISA与冻结原版7716条指令逐项比较，仅84条buffer_load_dword增加nt；160 VGPR、0 SGPR spill、零scratch、51712 B LDS。详见check_summary.json及isa_comparison.json。

baseline使用g1stream/baseline.py不可变源码；source/保存NT实验副本，source_manifest.json记录目标路径和hash。run.sh自动物化并清理实验文件；默认GEMM1文件不在清理列表。原始trace和逐样本结果见两个amd_pair目录。

复现：

```bash
cd /root/workspace/workflow_34/moonshot_ep32_moe_v2/cx
EP32_IGNORE_RESERVED_VRAM=1 bash g1cache/run.sh 5 check
EP32_IGNORE_RESERVED_VRAM=1 bash g1cache/run.sh 5 pair
```

使用指定Python/FlyDSL0.5.0和cc/wt_cache/flydsl050。每次GPU运行先执行指定gpu_idle.sh并检查连续10秒gfx/umc为0；仅按已有用户授权忽略保留显存。正式bench未改，未commit/push。
