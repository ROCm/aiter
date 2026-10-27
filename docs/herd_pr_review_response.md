# AITER #6527 评审逐条回复

核对对象：AITER PR [#6527](https://github.com/ROCm/aiter/pull/6527) 的远端版本 `266d1641ce`、本地合入的 `origin/main@7061cbd88b`，以及配套 ATOM PR [#2713](https://github.com/ROCm/ATOM/pull/2713)。下面按五项意见说明实际调用、修改和验证。

## 1. gfx1250 的旧 top-k HERD 入口是否被关闭

`_use_flydsl_herd` 控制普通 top-k 接口的自动 router drop。关闭其 gfx1250 分支不会关闭 ATOM #2713 的 K+1 → `fused_moe(herd_topk=K)` 入口，也不意味着显式的 gfx1250 `drop=0` 不支持。若设计目标是 gfx1250 统一使用 MoE 内的 fused drop，这个关闭可以是有意设计，不应直接判为 correctness 问题。

当前工作区为兼容旧调用保留 gfx950/gfx1250 的判断和 wave64/wave32 支持。新增的 `herd_topk` 参数则按调用约定限制为 gfx1250；gfx950 在 router 中完成 drop，向普通 MoE 传入 K 个专家。

ATOM #2713 的实际代码使用：

```python
# 输出宽度为 K+1；不做 drop、不归一化、不乘 routed scale。
topk_gating(
    candidate_weights,
    candidate_ids,
    router_logits,
    correction_bias,
    False,
    1.0,
    score_func="sqrtsoftplus",
)

fused_moe(
    ...,
    herd_topk=K,
    herd_routed_scaling_factor=routed_scale,
)
```

这里的 K+1 输出宽度不匹配旧 HERD 的 K 输出 profile，因此候选选择走普通 `topk_gating_fwd`，随后 MoE 完成一次 drop、归一化和缩放。AITER 提供的 `herd_topk_candidates` 是另一种候选选择入口；ATOM #2713 当前使用的是上面的 `topk_gating` 写法。

相关代码：[topk.py](../aiter/ops/topk.py)、[herd_topk.py](../aiter/ops/flydsl/herd_topk.py)。已增加旧入口的架构/profile 分发测试，以及真实的 `num_valid_rows` CUDA graph replay 测试。

## 2. gfx950 大 batch 报错和按 M 编译的问题

**调用范围：** `herd_topk` 只提供给 gfx1250 的配套 ATOM 调用。gfx950 在 top-k/router 阶段完成 drop，然后传普通 K 路由给 MoE；本次不支持从 gfx950 公共 MoE 入口进入 fused sorting drop。AITER 现在增加门控，非 gfx1250 传入非零 `herd_topk` 时明确报错，并在 finalize/后端分发前拒绝调用。

**M>128：** 评审指出的报错位于 gfx950 的 `flydsl_moe_sorting_herd`，不会被上述生产调用触发。ATOM #2713 还明确限制了 16–128 token decode。上一轮为 gfx950 添加的大 batch finalize 回退已撤回；不为范围外的调用扩展支持。

**JIT 编译：** `build_moe_sorting_herd_module(E, K, M, block_size)` 将 M 用于 LDS 容量和缓存 key，所以直接调用 gfx950 fused sorting helper 的确会按 M 编译。这个问题属于当前未启用的路径，本次不保留为其增加的固定 LDS/pointer 改造。

实际使用的两条路径没有这个 M 编译参数：

| 路径 | 编译/缓存参数 | M 的处理 |
| --- | --- | --- |
| gfx1250 fused drop：`herd_route_psum_remap` | `_get_compiled_route_psum_herd(topk)` | 运行时 Int32；E/max_m/tile_m 同样在运行时传入 |
| router drop：`herd_finalize_candidates` | K、E、归一化、输出精度、block threads、是否传 valid-row count | 运行时 rows，factory key 不包含 M |

新增测试在同一进程中遍历 route/psum 的 M=1/16/32/64/128，确认编译缓存始终只有一个实例，并检查专家 count、prefix sum、行位置和权重。router finalize 遍历 M=16/32/64/128，确认复用同一个 compiled callable。

相关代码：[gfx1250 门控](../aiter/fused_moe.py)、[route/psum cache](../aiter/ops/flydsl/grouped_moe_gfx1250.py)、[route/psum kernel](../aiter/ops/flydsl/kernels/moe_contiguous_psum.py)、[router finalize](../aiter/ops/flydsl/herd_topk.py)。将来若正式启用 gfx950 fused sorting，再处理该 helper 的 LDS 容量和缓存 key。

## 3. bias 是否与融合量化错误绑定，是否有遗漏 epilogue

保留 `5f88da51a4` 和 `e72fcd693d` 已有的 bias 实现。它们已经将 bias 与 fused requant 解耦，并让三个 epilogue 分支共用 `_biased_acc`。后续合并没有丢失这些修改；main 的 #6663 还补上了 bias 索引的 N tile 起点 `blk_n`。

已撤回 `c704a7f8cd` 中额外的 accumulator 阶段 bias 前移重构。这项重构不是满足第 3 点 review 所必需的修复，新增测试、CI、路由门控及其他修改继续保留。

原实现按 bf16/fp32/fp16 原生读取 bias，必要时转为 fp32，并在三个分支分别调用同一 helper：

| 输出分支 | bias 加法位置 |
| --- | --- |
| stage1 activation + fused requant | activation 和 MX quantization 前 |
| EP quantized scatter | route weighting 和量化前 |
| bf16/fp16 activation 或普通 GEMM 输出 | activation 或输出 dtype 转换前 |

新增 36 个 gfx1250 硬件组合：三种 bias dtype × requant 开/关 × SiLU/SwiGLU/SiTUv2 × stage 1/2。测试按专家和列生成不同 bias，将待验证 stage 的权重置零，使遗漏 bias 必须导致输出差异；PyTorch 参考使用同一 dtype 的逻辑 bias。测试同时覆盖多 N tile 的地址偏移。

也移除了 CSV 测试中“有 bias 就跳过非 256 对齐 intermediate”的旧逻辑：只有显式禁用 fused requant 时才保留 standalone quant 的对齐限制。

本机 gfx950 无法执行 gfx1250 TDM GEMM 数值测试，36 个组合交给 gfx1250 opt-in CI。已在本机针对 gfx1250 目标编译 15 个 GEMM 变体，覆盖三种 bias dtype、MXFP8/MXFP4 输入、融合 requant、非量化 activation 和普通输出。EP quantized scatter 未在本机完成编译/硬件验证，本机 MORI 包缺少 `mori.cco`。

相关代码：[epilogue](../aiter/ops/flydsl/kernels/mxfp4_preshuffle_gfx1250_tdm.py)、[bias 测试数据](../op_tests/flydsl_tests/test_flydsl_grouped_gemm.py)、[CI 测试](../op_tests/test_flydsl_herd.py)。

## 4. 本地测试未提交、CI 覆盖不足

意见成立，已新增 [op_tests/test_flydsl_herd.py](../op_tests/test_flydsl_herd.py)，覆盖：

- sigmoid/softmax/sqrtsoftplus 的 K+1 候选，以及 correction bias 只参与选择、不参与权重的语义；
- 实际使用的 route/psum 和 router finalize 跨 M 编译缓存复用；
- gfx950 传 `herd_topk` 时拒绝调用，gfx1250 的 drop=0/1 分发保持可用；
- grouped route/psum 的 expert count、行位置、prefix sum 和 gather weights；
- 旧 gfx1250/wave32 入口与 ATOM K+1 入口的分发；
- `num_valid_rows` 在 graph replay 中改变，padding 不参与 popularity，输出 IDs=-1、weights=0；
- gfx950 router drop + 普通 MoE，以及 gfx1250 `fused_moe(herd_topk=...)` 的 eager/graph 输出比较；
- 上述 36 个 gfx1250 bias/requant/stage 组合。

测试位于 `op_tests` 顶层，并提供直接执行入口，常规 Aiter CI 的 discovery 和 `python test_file.py` 执行方式都能运行它。已用 `split_tests.sh --dry-run` 确认进入 shard 列表。同时在 [ffm-bringup.yaml](../.github/workflows/ffm-bringup.yaml) 的 shard 0 中显式加入该测试，使 `ci:gfx1250-ffm-triton` job 执行 HERD 和 bias 测试。

本机 gfx950 验证：HERD 测试和 `fused_moe` 位置参数兼容性测试合计 26 passed、36 skipped。Black、Ruff、Python compilation、修改文件的 whitespace 检查及 workflow YAML 解析通过。36 个 skip 均为 gfx1250 bias 硬件测试；分发 mock 不能代替 gfx1250 硬件结果。

## 5. 新增 fused_moe 参数与 ATOM 配套依赖

这项意见是正确的依赖说明，无需调整公共参数位置。`herd_topk=0` 和 `herd_routed_scaling_factor=1.0` 追加在原有参数之后，旧位置参数调用继续成立，相关签名测试通过。

ATOM #2713 已明确依赖 AITER #6527，并在满足 decode 条件时分配 K+1 输出、选择未 drop 的候选，再传入上述两个参数。因此应将两个 PR 作为配套改动说明；只设置环境变量不能让未适配的 ATOM 调用自动使用 MoE 内的 fused drop。

## main 冲突处理和提交

在原分支 `cguo/gfx1250_herd` 提交本次修改。该分支同步了最新远端 PR 基线，并通过本地 merge commit `db2c31d9e5` 合入 `origin/main@7061cbd88b`，解决 `aiter/ops/topk.py` 冲突：同时保留 `need_renorm` 和 main 新增的 `num_valid_rows` 传递。

原本地分支的旧 HEAD 另存为备份分支；已有 `.gitignore`、submodule 和其他工作区文件不纳入本次提交。`origin/main` 是当前分支的祖先，index 中没有未解决冲突。提交信息不含 `Co-authored-by`，未推送到 GitHub。
