# MegaMoEV2 intra/inter backend 架构与 EP16 A4W4 实现

## 目标与范围

本实现为 `gfx950 + EP16（2 节点，每节点 8 GPU）` 增加 MegaMoEV2 A4W4
后端。通信使用 MORI `InterNodeV1LL`，激活在 dispatch 前按 `1x32`
blockwise MXFP4 量化，dispatch 在网络上传输 packed FP4；本地计算直接将 packed
FP4 和 E8M0 scale 传给 A4W4 `fused_moe`，combine 传输 BF16 输出。
BF16 入口复用 MegaMoEV2 的 FlyDSL `per_1x32_mx_quant(..., quant_mode="fp4")`，
不再调用通用 `get_torch_quant`；预量化入口则直接使用调用方提供的 FP4 与 scale。
后端会设置 `AITER_SITUV2_A4W4=1` 并关闭冲突的 A8W4 selector，调用方不需要
额外配置这一内部实现开关。

`MegaMoEV2` 只作为对外 facade，根据运行场景选择实现相同执行契约的 backend：

- `intra_node.py`：原有 A8W4 FlyDSL fused stage1/stage2；
- `inter_node.py`：EP16 A4W4 MORI dispatch、通用 fused MoE 和 MORI combine；
- `backend.py`：仅描述 `forward`/`forward_prequant` 的轻量内部协议。

两条路径不统一底层 buffer、workspace 或通信阶段，也不拆分 intra-node 的融合边界。
因此此次结构调整不会改变 kernel 调用及其性能路径。A4W4 入口会严格检查
gfx950、world size 16、hidden dimension 可被 32 整除，以及 experts 可被 EP 整除。

## 对外 API

```python
moe = MegaMoEV2(..., quant="a4w4", max_tok_per_rank=max_tokens)
output = moe(x_bf16, topk_weights, topk_ids)
# 或使用原有预量化入口：
output = moe.forward_prequant(x_fp4, x_scale, topk_weights, topk_ids)
```

`MegaMoEInterNodeContext` 保存接收 token、scale、权重、expert id、有效 token 数，
并在内部保存 source rank 原始 `topk_ids` 和生命周期状态。MORI combine 必须使用
源侧路由，不能使用 dispatch 返回的接收侧 expert id。当前只有一种 result 生命周期，
因此不再额外暴露一个 routing class。

MORI 返回真实的 top-k 路由，不在运行时追加 fake slot。当前 AITER EP 配置查找会
对 runtime top-k 减 1，因此该专用 tune CSV 的兼容查找 key 写为 15；kernel 实际
处理的 routed top-k 仍为 16。

MegaMoEV2 不对外暴露独立 dispatch、fused_moe、combine 方法；EP16 backend 在
`forward`/`forward_prequant` 内部依次完成三个阶段。专项测试可以直接检查内部 backend，
但这不属于稳定的用户接口。

## Rank 与设备

`rank` 是 EP 全局 rank（0..15），不可作为 node1 上的 CUDA device index。后端
所有本地 tensor 都使用 `torch.cuda.current_device()`，进程启动代码负责先设置
local rank。MORI config 仍使用全局 rank。

## MORI 配置

- kernel: `InterNodeV1LL`
- GPU per node: 8
- QP per peer: 2
- RDMA blocks: 64
- blocks: 96
- warps per block: 8

运行性能测试时仍需在两节点设置一致的 8-rail MORI 环境变量。测试脚本
`op_tests/multigpu_tests/test_ep16_a4w4_dispatch_moe_combine.py` 的正式结果通过
MegaMoEV2 公共接口得到；分阶段诊断计时仅在测试内部访问 backend。

## 生命周期约束

一个 `MegaMoEV2` 实例持有一个 MORI op，并沿用其 launch epoch/state；不要在各
iteration 重建实例。当前接口假设同一实例同一时刻只有一条 in-flight pipeline。

## 2026-09-03 checkpoint

- Python 架构已调整为薄 `MegaMoEV2` facade，以及独立的 intra/inter backend；
  对外仍只有 `quantize`、`forward`、`forward_prequant` 和 `__call__`。
- Intra EP8 A8W4 冒烟通过：BS16 relL2=0.059554，端到端约 0.3291 ms。
- Inter EP16 A4W4 冒烟通过：BS16 relL2=0.069863，BS128 relL2=0.069805。
- 未带专用 tune 的 100 次、末 20 次结果已保存在
  `trace_data/wrapper_ep16_a4w4_20260903/`。
- 新增的 local-expert tune 行使用 `expert=56` 是正确的。MORI 输入没有 fake slot，
  但当前 AITER EP 查表会将 runtime top-k 减一，因此其兼容 key 需要写成 `topk=15`；
  kernel 实际处理的 routed top-k 仍为 16。
- 新增 tune 行必须补齐 CSV 尾列；缺失的 `run_1stage` 会被 pandas 读为 NaN，并被
  当成 true，随后因 SiTUv2 不支持 one-stage 而丢弃配置。token=2048 行已经修正为
  `run_1stage=0, xbf16=0, flat=0`，但尚未实机复测是否命中和性能收益。
- 下一步恢复后：先以 `max_tok_per_rank=128` 运行 BS16/128，检查日志明确显示指定
  kernelName 且无 `heuristic fallback`；精度通过后再跑 100 次末 20 次性能对比。
