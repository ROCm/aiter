# Task1 当前实现结构：Torch-owned split-K workspace

更新时间：`2026-08-13`

本文只描述当前代码。续接状态与性能证据见 `docs/task1_checkpoint.md`。

## 目标和最终边界

```text
原始：C++按stream持有并扩容workspace
当前：调用方传最终kid，Python/Torch按exact instance准备workspace，C++只launch
```

当前边界：

- kernel数值算法和exact-kid launch表；
- C++ generated launcher检查和architecture-specific split-K规则；
- A16 C ABI和pybind raw family ABI；
- gfx942 direct-pointer wave uniformization。

生产Python功能位于已有的`__init__.py`、`_arch.py`、`gemm_op_a16w16.py`和
`gemm_op_a8w8.py`。`aiter/jit/core.py`与原始Task1基线`ca68b4f...`内容一致。

## 唯一公开入口

```python
from aiter.ops.opus import opus_gemm

opus_gemm(
    XQ, WQ, Y,
    kid=final_kid,
    split_k=split_k,
    workspace=workspace,
)
```

调用方必须提供最终 `kid`，并负责 `Y`。`layout`、dtype和scale只检查该kid对应family的
输入规则，不参与选核。

A16上层调用策略保持基线优先级：显式kid直接进入exact public；没有显式kid时，
`aiter/tuned_gemm.py`先查tuned CSV，无有效row时调用现有A16文件内的per-arch private
heuristic，仍无合法OPUS kid时进入PyTorch fallback。tuned/heuristic只产生最终kid，不进入
public内部。

## A16调用链

```text
explicit kid，或 tuned CSV -> per-arch heuristic 得到的最终 kid
  -> opus_gemm
  -> 基线已有的 kernels_list.get(final kid)
  -> instance tag进入现有A16模块
  -> family模块校验dtype/layout；launcher校验runtime arch
  -> _launch_a16w16
  -> _check_a16w16_launch_layout(live tensors)
  -> _cached_explicit_a16w16_plan(scalars only)
       -> _resolve_exact_a16w16_config
       -> _plan_a16w16_workspace
  -> caller workspace or torch.empty(plan)
  -> OPUS-local _opus_gemm_a16w16_launch_ctypes_raw
       (first lazy build uses the existing pybind wrapper)
  -> opus_gemm_a16w16_launch_cabi
  -> opus_gemm_a16w16_launch
  -> workspace/direct exact-kid table
  -> generated launcher
```

`_cached_public_contract`为`lru_cache(maxsize=4096)`，缓存route/dtype/layout/option presence；
`_cached_explicit_a16w16_plan`为`lru_cache(maxsize=256)`，缓存A16 scalar/workspace plan。
key/value均为标量、registry metadata、dtype或shape；每次调用的Tensor、bias Tensor、workspace
Tensor与stream不进入缓存。

## Exact-kid legality

`_resolve_exact_a16w16_config`只验证传入id：

- kid必须属于runtime arch的A16 family；
- input为BF16，output受exact instance或reducer合同支持；
- shape满足该instance的OOB/tile/fused限制；
- bias只允许exact bias-aware kid；
- workspace kid解析allocation/launch split-K；
- gfx1250 batch和compile-time fused split规则；
- gfx942 BF16-workspace exact-N规则。

exact resolver不查CSV、不选择另一id、不运行heuristic。显式传入gfx942 kid
10210/10213/10216且N非法时直接报错；tuned/heuristic候选的legacy redirect在上层完成，最终
kid才进入这里。

## Workspace计划

`_plan_a16w16_workspace(config, batch, M, N, K)`返回：

```python
(shape, torch_dtype) | None
```

直接kid返回`None`。workspace kid从canonical instance读取 `B_M/B_N/B_K`、
`splitk_workspace_dtype`和fused metadata，并验证split不超过K tile上限。

| 架构/类型 | shape | dtype |
|---|---|---|
| gfx950 two-stage | `[S,batch,padM,padN]` | FP32 |
| gfx942 two-stage | `[S,batch,padM,padN]` | exact BF16/FP32 |
| gfx1250 two-stage | `[S,padM,padN]` | BF16 |
| gfx1250 fused | `[tilesM,tilesN,fuseS-1,B_M,B_N]` | exact BF16/FP32 |

caller workspace存在时Python不重新分配，C++仍逐次验证其device、dtype、contiguous、alignment
和capacity。直接kid若收到workspace会被拒绝。

## 多stream与graph

自动workspace是每次调用的新Torch Tensor；两个并发stream不会共享Python Tensor。graph capture
中的`torch.empty`进入graph private pool，replay不重新执行Python。C ABI使用调用时live stream，
切换并恢复HIP device/stream。最小ctypes装载和参数转换位于现有A16文件，不修改通用
`aiter/jit/core.py`。模块内不存在Tensor global或per-stream pointer cache。

## Registry查询

不同arch的kid编号段和合并后的`csrc/opus_gemm/opus_gemm_common.py::kernels_list`在
Task1原始基线`ca68b4f...`中已经存在。统一入口只做`kernels_list.get(kid)`直接查询；
instance tag决定进入现有A16或A8模块，dtype/layout/scale规则留在对应family文件中。
Task1/Task2都没有重新分配这些编号，也没有新增per-arch Python路由文件。

当前registry另包含PR #4320的45个gfx950 MXFP8 BMM global kid（`8000 + upstream kid`），
总数为2084。它们复用同一个`kernels_list`和现有A8W8 Python文件，不改变本节A16 workspace
流程。

## Build subset

`DEFAULT_COMPILED_KIDS_GFX*`是默认build的exact-id floor，并覆盖三个A16 caller-side heuristic
可能返回的全部kid。generator再合并tuned CSV中的有效OPUS id、sidecar和mandatory A8 id，
并按目标arch过滤；public/C++仍不读取这些集合来选核。

## 生产调用方

下列调用方已在其自身tuner/dispatcher解析最终id后调用统一入口；其中
`aiter/tuned_gemm.py`负责A16 tuned/heuristic/PyTorch fallback优先级：

```text
aiter/tuned_gemm.py
csrc/opus_gemm/opus_gemm_tune.py
csrc/gemm_a16w16/gemm_a16w16_tune.py
aiter/ops/gemm_op_a8w8.py
csrc/ck_gemm_a8w8_blockscale/gemm_a8w8_blockscale_tune.py
```

`aiter/ops/deepgemm.py`不再提供OPUS兼容shim。

## 验证要求

- 静态/CPU：registry unique route、exact resolver、workspace planning、scalar cache lifetime、
  production import和subset compile；
- GPU：A16/A8数值、错误合同、C ABI、graph、多stream和gfx950 140-kid exhaustive；
- 性能：同kid、split-K、Tensor、warmup/round/iters做相邻A/B，分别报告raw eager、public eager
  和public graph；
- 物理GPU仅允许4、5、6、7。

当前MI355X验收为：重点回归`126 passed, 17 skipped`，gfx950 A16 canonical sweep
`140 passed`。完整96-case首尾对比中，Task1私有C ABI相对最初C++内部workspace为raw eager
`-11.789%`、graph `-10.479%`；当前生产family为eager `-0.721%`、graph `-10.744%`。
