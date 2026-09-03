# MegaMoEV2 EP16 A4W4 独立 Dispatch/Combine 实现

## 目标与范围

本实现为 `gfx950 + EP16（2 节点，每节点 8 GPU）` 增加 MegaMoEV2 A4W4
后端。通信使用 MORI `InterNodeV1LL`，激活在 dispatch 前按 `1x32`
blockwise MXFP4 量化，dispatch 在网络上传输 packed FP4；本地计算直接将 packed
FP4 和 E8M0 scale 传给 A4W4 `fused_moe`，combine 传输 BF16 输出。
后端会设置 `AITER_SITUV2_A4W4=1` 并关闭冲突的 A8W4 selector，调用方不需要
额外配置这一内部实现开关。

原有 EP8 A8W4 FlyDSL fused stage1/stage2 路径保持不变。A4W4 入口会严格检查
gfx950、world size 16、hidden dimension 可被 32 整除，以及 experts 可被 EP 整除。

## API

```python
moe = MegaMoEV2(..., quant="a4w4", max_tok_per_rank=max_tokens)

# BF16 输入，由后端执行 blockwise FP4 量化
dispatched = moe.dispatch(x_bf16, topk_weights, topk_ids)

# 或者输入已经量化的 packed FP4 和 E8M0 scale
dispatched = moe.dispatch_prequant(x_fp4, x_scale, topk_weights, topk_ids)

local_output = moe.fused_moe(dispatched)
output, output_weights = moe.combine(local_output, dispatched)
```

`MegaMoEDispatchResult` 保存接收 token、scale、权重、expert id、有效 token 数，
并在内部保存 source rank 原始 `topk_ids` 和生命周期状态。MORI combine 必须使用
源侧路由，不能使用 dispatch 返回的接收侧 expert id。当前只有一种 result 生命周期，
因此不再额外暴露一个 routing class。

直接调用 `moe(x, weights, ids)` 仍可完成 dispatch、A4W4 fused_moe、combine
全流程；独立接口用于后续把通信与计算分别调度或测量。

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
`op_tests/multigpu_tests/test_ep16_a4w4_dispatch_moe_combine.py` 已改为仅通过
MegaMoEV2 的 standalone dispatch/combine API 调用通信。

## 生命周期约束

一个 `MegaMoEV2` 实例持有一个 MORI op，并沿用其 launch epoch/state；不要在各
iteration 重建实例。当前接口假设同一实例同一时刻只有一条 in-flight pipeline。
