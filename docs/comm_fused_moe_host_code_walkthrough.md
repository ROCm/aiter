# 通算融合 MoE：FlyDSL Host 架构与逐行代码导读

本文对应提交：

```text
2c5bb5f93 feat: add windowed communication-fused MoE kernel
```

对应文件：

```text
aiter/ops/flydsl/comm_fused_moe_host.py
```

本文按该文件当前 326 行代码的行号解释。纯空行和括号续行与所属语句合并说明，避免把
文档写成大量没有实际信息的“这是一个右括号”。

## 1. 整体架构

调用关系：

```text
模型 / CommFusedMoeRuntime
        │
        ├─ 普通 FusedMoE 完成 routing、sorting、Stage1
        │
        └─ 通过 _stage2_override 调用 runner
                         │
             ┌───────────┴───────────┐
             │                       │
     _FullWidthRunner         _WindowedRunner
         M=2048                  M=32768
             │                       │
        G → L → B → RS → B → AG  七个列窗口流水
```

各阶段含义：

- `G`：Stage2 GEMM，生成压缩后的 routed result。
- `L`：本 rank 的 routed result 加 shared expert，并量化为通信 partial。
- `B`：TP rank 间 epoch barrier。
- `RS`：TP reduce-scatter，每个 rank 归约并保留 `M/TP` 行 shard。
- `AG`：TP all-gather，收集其他 rank 的 reduced shard。
- 最终每个 TP rank 都获得完整 `[M, H]` 输出。

Host 只负责：

1. 分配和持有长期 buffer。
2. 把 symmetric buffer 注册为 MORI external window。
3. 准备 FlyDSL kernel ABI。
4. 按固定顺序 launch kernel 和 barrier。

Host 不负责：

- routing、sorting、Stage1；
- 在线调参或读取 tuner 报告；
- 普通 MoE fallback；
- FlyDSL、Opus、ASM 跨后端选择；
- forward 热路径中的动态 planner。

## 2. 第 1～19 行：依赖和 Runner 缓存

### 第 1～2 行

```python
# SPDX-License-Identifier: MIT
"""FlyDSL host runner for communication-fused MoE."""
```

- 第 1 行：文件许可证。
- 第 2 行：说明该文件是通算融合 MoE 的 FlyDSL host，不是 GPU kernel 本体。

### 第 4～7 行

```python
import torch
import torch.distributed._symmetric_memory as symm_mem
import flydsl.expr as fx
from mori.cco import Communicator
```

- 第 4 行：分配普通 GPU tensor，并取得当前 CUDA stream。
- 第 5 行：分配可注册为跨 rank window 的 symmetric memory。
- 第 6 行：构造 FlyDSL `Int64`、pointer 等 launch 参数。
- 第 7 行：创建 MORI communicator 并注册 external window。

这四个依赖当前都在实际使用。

### 第 9～10 行

```python
from ... import full_width as full_width_kernels
from ... import windowed as windowed_kernels
```

Host 不再重复定义 `M/H/I/E/TOPK/TP/WINDOW`。

- `full_width_kernels`：当前 full-width KID。
- `windowed_kernels`：当前 windowed KID。

资源尺寸和 launch scalar 都直接从 kernel KID 取得，避免 host 与 kernel 出现两份不一致
的常量。

### 第 11～14 行

```python
from ...sync import (
    FLAT_VA_RANK_STRIDE,
    compile_epoch_barrier,
)
```

- `FLAT_VA_RANK_STRIDE`：相邻 TP rank 在 MORI flat VA 中的地址跨度。
- `compile_epoch_barrier`：编译 TP rank 间的 epoch barrier。

Barrier 被 full-width 和 windowed 共用，因此放在 `sync.py`。

### 第 15～16 行

```python
from ...tensor_shim import ptr_arg
from ...moe_kernels import _run_compiled
```

- `ptr_arg`：把 PyTorch tensor 的 `data_ptr()` 包装为 FlyDSL pointer 参数。
- `_run_compiled`：第一次调用时 JIT 编译并执行，后续调用缓存后的 launcher。

Host 不维护第二套 compiled-function cache。

### 第 19 行

```python
_RUNNERS = None
```

缓存已经构造好的 runner。它还负责间接维持以下资源生命周期：

- symmetric tensor；
- MORI communicator；
- external-window handle；
- flat-VA 映射。

当前 `_RUNNERS` 只支持一组模型 shape 和 TP group。后续做通用 shape+tuner 时，应改为按
shape、device 和 TP group 索引的缓存。

## 3. 第 22～69 行：五个公共 Helper

### 第 22～23 行：分配 Symmetric Tensor

```python
def _symmetric(device, shape) -> torch.Tensor:
    return symm_mem.empty(shape, dtype=torch.uint8, device=device)
```

使用 `uint8` 是因为这些 buffer 表示原始通信字节布局，不依赖 PyTorch 逻辑 dtype。

这里故意不调用整块 `zero_()`：

- payload 会被 GPU kernel 完整覆盖；
- 只有 epoch counter 需要初始化；
- 避免初始化数百 MiB 无用数据。

### 第 26～29 行：分配 Payload + Epoch Workspace

```python
def _workspace(device, payload_bytes: int) -> torch.Tensor:
    tensor = _symmetric(device, ((payload_bytes + 8 + 255) // 256 * 256,))
    tensor[payload_bytes : payload_bytes + 8].zero_()
    return tensor
```

- 第 26 行：输入实际 payload 字节数。
- 第 27 行：
  - `payload_bytes + 8` 为尾部 epoch counter 预留 8 bytes；
  - `+255 // 256 * 256` 将总长度向上对齐到 256 bytes；
  - 分配 symmetric memory。
- 第 28 行：只初始化 epoch counter，不清空 payload。
- 第 29 行：返回完整 workspace。

当前 production layout 的 payload 都是 8-byte 对齐，因此 epoch 可直接放在
`payload_bytes` 偏移处。

### 第 32～42 行：MORI Window 注册

```python
def _register(tp_group, rank: int, tp: int, tensors):
```

统一处理两个 runner 共用的 communicator 和 window 注册。

#### 第 33 行

```python
uid = Communicator.get_unique_id() if rank == 0 else None
```

只有 rank 0 创建 communicator UID。

#### 第 34～36 行

```python
comm = Communicator.init(
    tp, rank, tp_group.broadcast_object(uid), per_rank_vmm=FLAT_VA_RANK_STRIDE
)
```

- rank 0 将 UID 广播给其他 TP rank；
- 所有 rank 使用同一个 UID 创建 communicator；
- `per_rank_vmm` 固定每个 rank 的虚拟地址跨度。

#### 第 37～40 行

```python
windows = tuple(
    comm.register_external_window(tensor.data_ptr(), tensor.nbytes)
    for tensor in tensors
)
```

逐个注册 symmetric tensor。window handle 必须长期存活，因此后面保存到 runner 的
`self.windows` 中。

#### 第 41 行

```python
bases = tuple(w.local_ptr - rank * FLAT_VA_RANK_STRIDE for w in windows)
```

把本 rank 的 local pointer 转换为所有 rank 共用的 canonical flat base：

```text
peer_address = flat_base + peer_rank * FLAT_VA_RANK_STRIDE
```

kernel 因此不需要在每次 forward 查询 peer pointer。

#### 第 42 行

返回：

- communicator；
- window handle；
- 每个 window 的 flat base。

### 第 45～49 行：Launch Epoch Barrier

```python
def _barrier(workspace, flat_base, ready_offset, stream) -> None:
```

#### 第 46～48 行

构造 barrier launcher ABI：

- `workspace`：本 rank workspace pointer；
- `flat_base`：跨 rank 统一基地址；
- `ready_offset`：epoch counter 在 workspace 中的偏移；
- `stream`：当前 CUDA stream。

#### 第 49 行

```python
_run_compiled(compile_epoch_barrier(), args)
```

第一次编译 barrier kernel，之后复用缓存。Barrier 会：

1. 将本 rank 的 epoch 加一并 system-release；
2. 等待所有 TP rank 的 epoch 达到同一值并 system-acquire。

### 第 52～69 行：Stage2 公共 ABI

```python
def _stage2_args(stage2_args, stage2_kwargs, kernels):
```

Full-width 和 windowed 的 GEMM 入口拥有相同的前半段 ABI。

#### 第 53～54 行

```python
inter_states, w2 = stage2_args[0], stage2_args[2]
sorted_token_ids, sorted_expert_ids, num_valid_ids = stage2_args[3:6]
```

从普通 FusedMoE Stage2 参数中取出：

- Stage1 输出；
- W2 权重；
- sorted token ids；
- sorted expert ids；
- 有效排序项数量。

#### 第 55～69 行

返回的 ABI 依次为：

| 行号 | 参数 | 作用 |
| ---: | --- | --- |
| 56 | `inter_states` | Stage1 intermediate states |
| 57 | `w2` | Stage2 W2 权重 |
| 58 | `a2_scale.view(-1)` | 连续 A2 scale buffer |
| 59 | `w2_scale.view(-1)` | 连续 W2 scale buffer |
| 60 | `sorted_token_ids` | 排序后的 token id |
| 61 | `sorted_expert_ids` | 排序后的 expert id |
| 62 | `sorted_weights` | routing weight |
| 63 | `num_valid_ids` | 有效排序项数量 |
| 64 | `inter_states` | disabled-bias ABI 的合法 dummy pointer |
| 65 | `kernels.M` | token bucket |
| 66 | `kernels.H` | hidden size |
| 67 | `kernels.I` | 每 TP rank intermediate size |
| 68 | `shape[0] * 2` | common Stage2 kernel 当前要求的 expert-id 逻辑长度 |

第 64 行不会把 `inter_states` 当 bias 读取：

- 当前 kernel 编译时 `enable_bias=False`；
- ABI 仍要求传入合法 pointer；
- kernel 不会读取该参数；
- 复用已有 pointer 可以删除专用零长度 `empty_bias` tensor。

## 4. 第 72～144 行：Full-width Runner

该 runner 对整个 H 一次完成 Stage2 和通信：

```text
G → L → partial barrier → RS → reduced barrier → AG
```

### 第 73～76 行：基础状态

- 第 73 行：构造函数接收 TP group。
- 第 74 行：使用 full-width kernel KID。
- 第 75 行：保存当前 TP rank。
- 第 76 行：保存当前 GPU device。

### 第 77～81 行：Route Buffer

```python
self.route = torch.empty(
    (kernels.M, kernels.TOPK, kernels.H + kernels.H // 8),
    ...
)
```

每个 token、每个 top-k route 保存：

```text
H bytes FP8 payload + H/8 bytes scale metadata
```

它只在本 GPU 上由 `G` 写、由 `L` 读，不需要注册为通信 window。

### 第 82～83 行：Partial Workspace

```python
self.partial_ready = kernels.M * (kernels.H + kernels.H // 32)
self.partial = _workspace(self.device, self.partial_ready)
```

每个 token 保存：

```text
H bytes FP8 partial + H/32 bytes E8M0 scale
```

`L` 写本 rank partial，`RS` 通过 flat VA 读取所有 TP rank 的 partial。

### 第 84～88 行：Reduced Payload 和 Scale

```python
self.reduced_ready = kernels.SHARD_ROWS * kernels.H
self.reduced_payload = _workspace(self.device, self.reduced_ready)
self.reduced_scale = _symmetric(...)
```

每个 rank 负责 `M / TP` 行 reduced shard。`reduced_payload` 保存供其他
rank all-gather 的 MXFP8 数据，尾部包含 epoch；`reduced_scale` 保存对应
E8M0 scale。payload 和 scale 由同一个 `RS` kernel 产生。

### 第 89～91 行：最终输出

分配完整 `[M, H]` BF16 输出。每个 TP rank 最终都持有一份完整输出。
当 token 数需要 padding 时，runtime 直接复用 `runner.output` 作为
`[bucket, H]` shared staging buffer，不再为返回这个字段定义空转发方法。
该别名时序已通过 M=2048 output/shared alias 精度测试。

### 第 93～101 行：通信注册

- 第 93 行：按照 `partial → reduced payload → reduced scale` 组织 tensor。
- 第 94～96 行：创建 communicator 并注册三个 external window。
- 第 97～101 行：解包三个 flat base。

`self.comm` 和 `self.windows` 必须保存为成员，以维持 MORI 资源生命周期。

### 第 102～105 行：本 Rank 的 Reduced Shard

```python
shard_begin = self.rank * kernels.SHARD_ROWS
self.reduced_shard = self.output[shard_begin : shard_begin + kernels.SHARD_ROWS]
```

`RS` 直接写本 rank 负责的 BF16 shard；其他 rank 的 shard 由 `AG` 写入。

### 第 107～110 行：准备 Launch

- 第 107 行：runner 统一调用契约。
- 第 108 行：取得 full-width KID。
- 第 109 行：使用调用者当前 CUDA stream。
- 第 110 行：构造公共 Stage2 ABI。

### 第 111 行：G，Stage2 Compute

Stage2 GEMM 将结果写入 compact route buffer。

### 第 112～115 行：L，本地归约

输入：

- route；
- partial workspace；
- shared expert partial；
- stream。

`L` 将 top-k routed rows 与 shared expert 相加，并量化到 `self.partial`。

### 第 116 行：Partial Barrier

保证所有 TP rank 的 `L` 完成后，才允许任何 rank 的 `RS` 读取 peer partial。

### 第 117～127 行：RS，TP Reduce-Scatter

输入：

- partial flat base；
- reduced BF16 shard；
- reduced MXFP8 payload workspace；
- reduced scale；
- rank；
- stream。

`RS` 会：

1. 读取所有 TP rank 的 partial；
2. 完成跨 rank 归约；
3. 写本 rank 的 BF16 reduced shard；
4. 生成供其他 rank all-gather 的 payload 和 scale。

### 第 128～133 行：Reduced Payload Barrier

保证所有 rank 都完成 reduced payload 后，才允许 all-gather。

### 第 134～143 行：AG，TP All-Gather

输入：

- reduced payload flat base；
- reduced scale flat base；
- 完整 output；
- rank；
- stream。

每个 rank 读取其他 rank 的 reduced shard，填满完整 `[M, H]`。

### 第 144 行

返回完整 BF16 输出。

## 5. 第 147～202 行：Windowed Runner 初始化

M=32768 时按列切为七个 1024-column 窗口：

```text
G0

G1 + L0
G2 + L1 + RS0
G3 + L2 + RS1 + AG0
G4 + L3 + RS2 + AG1
G5 + L4 + RS3 + AG2
G6 + L5 + RS4 + AG3

L6 + RS5 + AG4
RS6 + AG5
AG6
```

### 第 148～151 行

保存 window kernel KID、rank 和 device。

### 第 152～159 行：双 Route Buffer

只分配 `SLOTS=2` 个 route buffer：

```text
slot = window % 2
```

因为 `G(n+2)` 开始前，`L(n)` 已消费完对应 route。每个 route row 保存：

```text
WINDOW bytes FP8 + WINDOW/8 bytes scale metadata
```

### 第 160～166 行：两个 Partial Slot

`partial_ready` 是一个逻辑窗口的全部 partial payload：

```text
M × (WINDOW FP8 bytes + WINDOW/32 scale bytes)
```

只分配两个物理 slot，不再为七个逻辑窗口各分一份。

### 第 167～171 行：两个 Reduced Payload Slot

每个 slot 保存：

```text
SHARD_ROWS × WINDOW
```

并在 payload 尾部包含 epoch。

### 第 172～175 行：两个 Reduced Scale Slot

每个 slot 保存当前 reduced shard window 的 scale。

### 第 176～178 行：完整输出

最终输出仍是完整 `[32768, 7168]` BF16 tensor。

### 第 180～185 行：注册六个 Window

注册顺序：

```text
2 partial → 2 reduced payload → 2 reduced scale
```

旧实现需要 `7 + 7 + 7 = 21` 个通信 window。

### 第 186～188 行：拆分 Flat Base

按照注册顺序把 bases 分为 partial、reduced payload 和 reduced scale 三组。

### 第 189～196 行：Reduced Shard Views

为七个逻辑窗口提前创建本 rank 的 reduced shard view，避免在 forward 热路径中重复创建
二维 slice。

### 第 197～202 行：All-Gather Output Views

为七个逻辑窗口提前创建完整行范围的 output view。`AG(window)` 写对应的
`[M, WINDOW]` 区域。

## 6. 第 204～246 行：Window ABI Helper 和 Tail

### 第 204～207 行：L 参数

```python
def _local_args(self, window, shared_partial):
```

- 第 205 行：逻辑窗口映射到两个物理 slot。
- 第 206 行：取得 shared expert 对应的列窗口首地址。
- 第 207 行：返回 route、partial、shared 三个 pointer。

这里使用 `[:, start:]`，因为 kernel 只读取固定 `WINDOW` 列，传入窗口首地址即可。

### 第 209～220 行：Collective 参数

- 第 210 行：计算 reduce-scatter 使用的 partial/reduced slot。
- 第 211 行：计算 all-gather 使用的 reduced slot。
- 第 212～220 行按 combined kernel ABI 返回：
  1. reduce-scatter partial flat base；
  2. 本 rank reduced BF16 shard；
  3. reduced payload workspace；
  4. reduced scale；
  5. all-gather reduced payload flat base；
  6. all-gather reduced scale flat base；
  7. all-gather output view。

### 第 222～246 行：Tail Launcher

Steady-state 结束后固定执行：

```text
L6 + RS5 + AG4
RS6 + AG5
AG6
```

#### 第 230～233 行

关闭的阶段使用窗口 0 提供合法 dummy pointer。

这不是 fallback：

- drain launcher ABI 固定要求所有 pointer 都存在；
- `local/reduce_scatter/all_gather is not None` 是编译期 specialization；
- 被关闭阶段的 pointer 不会被 kernel 读取。

#### 第 234～239 行

根据三个阶段是否存在，取得对应的固定 drain launcher。实际只编译三种版本：

```text
L+RS+AG
RS+AG
AG
```

#### 第 240～245 行

组合 L 参数、collective 参数、rank 和 stream。

#### 第 246 行

执行对应 tail kernel。

## 7. 第 248～312 行：Windowed 固定流水

### 第 248～254 行：G0

- 取得 window KID；
- 取得当前 stream；
- 构造公共 Stage2 ABI；
- 单独 launch 第 0 个窗口的 GEMM。

流水刚开始时没有可并行的 `L/RS/AG`，所以 `G0` 必须单独启动。

### 第 255～289 行：Steady State

循环次数当前为 `7 - 1 = 6`。

每轮 `local` 表示当前执行的 L 窗口。

#### 第 256 行

```python
has_reduce_scatter = local > 0
```

从第二轮开始可以执行前一个窗口的 reduce-scatter。

#### 第 257 行

```python
has_all_gather = local > 1
```

从第三轮开始可以执行前两个窗口的 all-gather。

#### 第 258～259 行

计算逻辑 reduce-scatter/all-gather 窗口。阶段尚未启用时使用窗口 0
提供 dummy ABI。

#### 第 260～274 行：Combined Cycle

每轮执行：

```text
G(local + 1)
+ L(local)
+ optional RS(local - 1)
+ optional AG(local - 2)
```

参数顺序为：

- 下一窗口 route output；
- Stage2 GEMM 公共 ABI；
- L 参数；
- collective 参数；
- rank；
- stream。

RS/AG 是否存在由 compile-time boolean 决定，不是 GPU 热路径动态 planner。

#### 第 275～281 行：Partial Barrier

`L(local)` 完成后同步对应 partial slot，保证下一轮 `RS(local)` 可以安全读取所有 rank 的
partial。

#### 第 282～289 行：Reduced Payload Barrier

只要本轮执行了 `RS(reduce_scatter)`，就同步对应 reduced slot。它同时保证：

1. 后续 `AG(reduce_scatter)` 可以安全读取 reduced payload；
2. 更晚逻辑窗口复用同一 2-slot scratch 前，旧的 RS/AG 读取已完成。

这是七份 scratch 能安全收缩为两份的关键同步。

### 第 291～311 行：Tail

当前 `last=6`。

- 第 292 行：执行 `L6 + RS5 + AG4`。
- 第 293～294 行：计算 window 6 和 window 5 的 scratch slot。
- 第 295～297 行：等待所有 rank 的 `L6` 完成。
- 第 298～303 行：等待所有 rank 的 `RS5` 完成。
- 第 304 行：执行 `RS6 + AG5`。
- 第 305～310 行：等待所有 rank 的 `RS6` 完成。
- 第 311 行：执行 `AG6`。
- 第 312 行：返回完整输出。

## 8. 第 315～326 行：Factory

### 第 315 行

Factory 接收 TP group、model dimension、per-TP intermediate dimension、expert 数和 top-k。

### 第 316 行

构造当前请求的 shape key：

```text
(H, I_per_tp, E, topk, TP)
```

### 第 317 行

当前暂时以 full-width KID 作为模型 shape 的事实源。Full-width 和 windowed 当前属于同一
模型 shape，只是 token bucket 和执行方式不同。

### 第 318～319 行

当前只允许已经验证的 exact shape，防止未调优 shape 错误进入固定 grid/kernel。

这不是最终通用生产架构。后续应替换为：

```text
shape
→ 查询离线 tuner winner
→ 得到 kernel KID
→ 构造该 shape 的 runner
```

### 第 320 行

声明修改模块级 `_RUNNERS`。

### 第 321～325 行

第一次调用时构造两个 token bucket：

- full-width runner；
- windowed runner。

之后重复使用同一组资源。

当前还有一个后续优化点：两个 runner 会同时创建。通用 shape 架构中可改为按 bucket 延迟
构造，避免从未使用的 bucket 提前占显存。

### 第 326 行

返回 `{token_bucket: runner}` 映射给 `CommFusedMoeRuntime`。

## 9. 当前资源变化

这轮 Host 收缩包括：

- 删除独立 shared tensor，复用 output；
- 删除 reduced payload view，直接传 reduced workspace 首地址；
- 删除 empty-bias tensor；
- symmetric workspace 只初始化 8-byte epoch；
- window partial/reduced-payload/reduced-scale 从七份收缩为两份；
- MORI external window 从 21 个减少到 6 个。

预计显存减少：

| Runner | 每 GPU 节省 |
| --- | ---: |
| M=2048 full-width | 约 28 MiB |
| M=32768 windowed | 约 633.6 MiB |
| 两个 runner 都构造 | 约 661.6 MiB |

## 10. 已完成验证

MI355X TP8 最终 Graph 结果：

| M | Route | 原基线 | 最终结果 |
| ---: | --- | ---: | ---: |
| 2048 | uniform | 266.702 us | 267.791 us |
| 2048 | skew | 271.788 us | 271.793 us |
| 32768 | uniform | 2224.386 us | 2228.116 us |
| 32768 | skew | 2221.514 us | 2225.370 us |

精度：

- M=2048：`max_abs=0.75`，`rel_l2≈0.02997`；
- M=32768：`max_abs=0.8125`，`rel_l2≈0.03013`。

其他验证：

- M=2048 output/shared alias：通过；
- M=32768 output/shared alias：通过；
- 普通 FlyDSL A8W4 Stage2：`3 passed`；
- runtime 接口测试：`3 passed`；
- `compileall`、`git diff --check`：通过。

## 11. 后续通用 Shape + Tuner 架构

当前 Host 已完成资源和固定流水收缩，但 factory 仍是单 shape 临时接入。后续合理结构是：

```text
shape + token bucket + TP
        │
        ▼
离线 tuner 枚举：
full-width / windowed
tile / grid / window / slots
        │
        ▼
保存 winner KID
        │
        ▼
factory 查表并按需创建 runner
        │
        ▼
同一套轻量 Host 执行固定 winner
```

泛化能力来自 shape→KID 的离线调优和映射，而不是在 forward 热路径中增加动态 planner。
