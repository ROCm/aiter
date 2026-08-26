# TPMoEStage1 阶段二设计：kernel 内 P2P all-gather

日期：2026-08-26
分支：`dev/all_gather_merge_stage1`
前置文档：`docs/superpowers/specs/2026-08-25-tp-moe-stage1-phase2-handoff.md`

---

## 0. 一句话

把 activation 的 all-gather 从一次 NCCL collective 改成融合 kernel 内部的 P2P store，与 GEMM1 合并成一次 launch；`moe_sorting` 完整保留为独立 kernel，不进融合 kernel。

---

## 1. 选定方案与被否掉的方案

用户在三个方案里选了 B。

| | 描述 | 估算（m_local=128） |
|---|---|---|
| A | 全融合，planner 也进 kernel | ~0.377 ms，快 29% |
| **B** | **保留外部 `moe_sorting`** | **~0.443 ms，快 17%** |
| C | 先 B 后 A 分两步走 | 同上，多一轮迭代 |

否掉 A 的理由：A 的 planner 不能照抄 MegaMoE。EP 下 dispatch 动作本身完成了按 expert 的排列，planner 只需产出 `TILE_ROW_BASE` 这种「第几个 tile 从第几行开始」的表；TP 下推送不产生任何排列，必须在 kernel 内复现 `moe_sorting` 的计数排序并产出 `(slot<<24)|token` 的逐行索引表，风险集中在这一块。

以上估算均为按成本模型 `T ≈ 20 µs + 17 µs × collective 次数 + bytes/275 GB/s` 的推算，非实测，且未扣除融合 kernel 自身的 ticket 与 ready flag 协议开销。

---

## 2. 立论所依赖的实测事实

以下每条都在本次设计过程中实测确认，不是沿用记忆。

**2.1 GEMM1 的 A buffer 是稠密的，没有按 topk 展开。**

```
m_global       = 1024
sorted_ids     = (18426,)   num_valid = 12288
a_fp8 (GEMM1 A)= (1024, 7168) fp8_e4m3   7.34 MB
a_scale        = (18432, 224) fp8_e8m0
row expansion  = a_rows / m_global = 1.0   (topk = 6)
```

推论：推送的目标行号 `src_rank * m_local + t` 静态可算，不依赖 sort 结果。

**2.2 mxfp8 量化是行内独立的，本地量化与 gather 后量化逐位相同。**

把 `[1024, 7168]` 整体量化，与切成 8 份各自量化后拼接，两者比较：

```
q 逐位相同 : True     不同元素数: 0 / 7340032
s 逐位相同 : True     不同元素数: 0 / 229376
```

推论：「先量化再传」不改变数值，每行跨卡字节数从 14336 降到 7392，量化工作量从 `m_global` 行降到 `m_local` 行。

**2.3 融合路径的 A-scale 是朴素 row-major，shuffle 在 LDS 里运行时现做。**

- 独立路径 `mixed_moe_gemm_2stage_common.py:1098-1104` 用排序后行号 `bx_m` 寻址 shuffle 过的 buffer，式子 `a_mni * stride_n0 + scale_lane_elem` 结构上要求 `x>>5` 在 wave 内是常量、`x&15` 等于 MFMA lane id，因此**无法**改成按 token id 寻址。
- 融合路径 `gemm_util.py:335-347` 读 `[rows, model_dim/32]` 的朴素 row-major，shuffle 由 `gemm_util.py:370-390` 在 LDS 里完成。

推论：融合 kernel 里 A 和 scale 都能保持稠密 rank-major 并用同一个 token id 间接寻址，`moe_mxfp4_sort` 整步删除，省掉一次约 4 MB 的写。

**2.4 全局排序不可省，单卡各排各的会让 GEMM1 工作量涨 6.8 倍。**

```
全局排序 : 路由  6144 条 ->  12288 行,  有效率  50.0%
单卡排序 : 路由   768 条 ->  10464 行,  有效率   7.3%  (×8 卡 = 83712 行)
GEMM1 行数比 = 6.8x
```

推论：`topk_ids` 的 6 KB gather 是 TP 相对 EP 的固有成本，绕不掉。

**2.5 MegaMoE 既不 gather `topk_ids` 也不 gather `route_weights`。**

`topk_ids` 只在本卡读，跨卡交换的是 384 个 int32 的计数矩阵（`dispatch.py:517-522`）；`route_weights` 搭着 payload 逐行推送，lane 0 各写一个 dword（`dispatch.py:901-907`）。EP 下没有任何一张卡需要全局路由表，TP 下每张卡都需要。

**2.6 `MegaMoEV2` 的输出 run-to-run 不可复现。**

在与 `main` 逐字节相同的生产代码上，固定 seed、清空 JIT cache 连跑两次，八个 rank 输出全不一致而编译 IR 完全一致。详见 6.3。这条不是本方案的立论依据，而是它否掉了「抽公共模块」那条路。

---

## 3. 流水线划分

### 融合 kernel 外面（三次 launch，全部用现成 kernel）

1. **本地量化。** `per_1x32_mx_quant(x_bf16)` 处理本卡 DP 分片的 `m_local` 行，产出 `x_q [m_local, 7168]` fp8 与 `x_scale [m_local, 224]` e8m0。
2. **一次 collective 收 metadata。** `topk_ids` 与 `route_weights` 形状同为 `[m_local, topk]`，拼成一个 int32 buffer 一次 `all_gather_into_tensor`，m_local=128 时 6 KB。
3. **`moe_sorting`。** 产出 `sorted_token_ids`、`sorted_weights`、`sorted_expert_ids`、`num_valid_ids`。原始 `topk_ids` 与 `route_weights` 此后不再被读取。

### 融合 kernel 里面（一次 launch）

4. **推送。** producer CTA 把本卡 `x_q` 与 `x_scale` 逐行写进所有 peer 的对称接收 buffer，目标行号 `tp_rank * m_local + t`。
5. **同步。** `fence_system_release` + `atomic_add_system` 打 flag，consumer `int32_wait_until_equals` 自旋后 `fence_system_acquire`。
6. **GEMM1。** work pool 取活，每个 M-tile 从 `sorted_token_ids` 解出 32 个 token id，按 id 取 A 行与 scale 行。

### 输出

仍为 `TPMoEStage1Output`，字段一个不变，下游仍是 `flydsl_moe2_layout_afp8_wfp4_bf16_t32x128x128_atomic_sbm32`。输出侧 scale 布局不变，阶段一已验证 MegaMoE `_s1_osd` 与 v2 排序 scale 逐位相同。

### 实测结果（m_local=128，替换原推算表）

下面是 2026-08-26 在本机 8× gfx950 上用 `bench_tp_moe_stage1.py` 实测的数字，计时方式与阶段一基线一致：每次迭代前 `dist.barrier()`，CUDA event 分段，30 次取中位数，再跨 rank `all_reduce(MAX)`。原推算值一并列出以便对照。

| | 基线（实测） | 方案 B 推算 | 方案 B 实测 |
|---|---|---|---|
| 量化 | 0.0282 | ~0.010 | 0.0762 |
| collective | 0.1264（3 次） | 0.037（1 次） | 0.0752（1 次） |
| `moe_sorting` | 0.0133 | 0.0191 | 0.0578 |
| 融合部分 | 0.3508 | ~0.377 | 0.3456 |
| 合计 | 0.5188 | ~0.443 | 0.5549 |

推算错在哪，可以逐行说清楚。

融合 kernel 那一行推算得相当准，而且方向还反了过来：推算认为把推送塞进 GEMM1 要多花约 24 µs，实测融合 kernel 比基线单独跑 GEMM1 还快 5 µs。也就是说 P2P 推送与跨 rank 握手的开销被 GEMM1 的访存完全盖住了，设计里最不确定的那一半反而是成立的。

错的是前三段。推算把前三段当成纯 GPU 工作来估，于是「本地量化只处理 128 行而不是 1024 行」「collective 从 3 次减到 1 次、字节数从 12.85 MB 减到 6 KB」这两条都被算成了实打实的节省。实测告诉我们这三段根本不是 GPU 受限的，而是被 host 端的 kernel 提交成本卡住。bench 里新加的 host 列直接把这一点量化了：融合路径前三段的 device 时间与 host 提交时间几乎逐段相等（0.0762 对 0.0766，0.0752 对 0.0711，0.0578 对 0.0576），说明 GPU 在这三段里一直空转等 CPU 下发命令。字节数减少了三个数量级，时间却一点没少。

基线那三段之所以看起来便宜，恰恰是因为它 GPU 工作多。三次 all-gather 一提交就把 GPU 队列填满，后面的 `moe_sorting` 和量化在提交时 GPU 还没追上，于是它们的 device 时间（0.0133、0.0282）比各自的 host 提交时间（0.0522、0.0418）还短，提交开销被前一段的 GPU 执行藏掉了。融合路径把 GPU 工作抽走之后，这层遮蔽也跟着没了。

结论是：这张推算表用的成本模型（`T ≈ 20 µs + 17 µs × collective 次数 + bytes/275 GB/s`）少了一项，也就是每段固定约 50–75 µs 的 host 提交成本，而这一项在当前的逐次 drain 计时协议下是完全暴露的。合计一项因此推算偏低约 0.11 ms。

### 3.5 NCCL 实现搬出生产模块

`TPMoEStage1` 只保留融合这一条实现，阶段一那条 NCCL 路径整个搬到 `op_tests/multigpu_tests/tp_moe_stage1_nccl_ref.py`，类名 `TPMoEStage1NCCLRef`。

**理由。** 这条路径的唯一用途是给融合实现当对拍参照，用完就删，两边不需要同步演进。留在生产模块里会逼出一层用不上的抽象，也让 `TPMoEStage1` 背着一条永远不会上线的分支。放进 `op_tests` 之后，它的位置本身就说明了它是一次性的，将来删除时不会有人误以为在删公开 API。

**代码重复是可接受的。** `_validate_call`、`_sort`、`_run_gemm1`、`_pack` 这几个方法会有两份，但它们不需要同步修改。唯一共享的是 `TPMoEStage1Output`，参照实现从生产模块 import 它而不是复制，因为这个 dataclass 正是两边都要满足的输出契约，复制会让契约有两个定义。

**`transport` 构造参数删除。** 一个类只剩一条实现，这个参数就没有取值可选了。今天刚做的 `nccl_allgather` / `fused_p2p` 改名（提交 `05e6e656a`）随之作废，那次改动成本很低，留在历史里无害。

**实施顺序。** 先把 NCCL 实现复制进 `op_tests` 并让阶段一的全部用例改跑参照实现，确认绿；再在 `TPMoEStage1` 里长出融合路径；最后删掉生产模块里的 NCCL 路径与 `transport` 参数。中间会有一段两份代码并存的窗口，这是有意的，为的是任何一步出问题都能立刻对比。

**`aiter/ops/flydsl/kernels/mega_moe/__init__.py` 不变**，仍然导出 `TPMoEStage1` 与 `TPMoEStage1Output` 两个名字。

---

## 4. 内存布局与构造前提

### 对称内存（构造时按 `max_tok_per_rank` 一次性分配）

`mori_shmem_create_tensor` 是集合操作，八张卡必须以相同顺序、相同大小调用，因此不能每次调用按 `m_local` 现开。记 `MTPR = max_tok_per_rank`，`P = tp_size`：

| 张量 | 形状 | MTPR=128 | MTPR=256 |
|---|---|---|---|
| `rx_x` | `[P*MTPR + 1, 7168]` fp8 | 7.34 MB | 14.68 MB |
| `rx_scale` | `[P*MTPR + 1, 224]` uint8 | 0.23 MB | 0.46 MB |
| `payload_ready` | `[2]` int32 | 8 B | 8 B |
| `launch_ready` | `[P]` int32 | 32 B | 32 B |

末尾多出的那一行是清零的 PAD 行，见 5.5。每个对称张量配一张 `int64[P]` 的 p2p 地址表，用 `build_p2p_table` 建。

**双缓冲。** `rx_x` 与 `rx_scale` 按 parity 开两份，实际形状是 `[2, P*MTPR + 1, ...]`，两份各自带一行 PAD 行。内存翻倍，MTPR=256 时合计 30.2 MB。默认开启，留构造参数可关。理由见 5.3。

### 本地（非对称）调度状态

沿用 MegaMoE 的形状：`work_head` 八个分片各占一条 64 字节 cache line、`epoch_gate` 与 `entry_count` 各 10 槽、`epoch_parity` 一个 int32、`epoch_expected` 两个 int32。

### 输出每次调用新分配

GEMM1 输出 `payload` 与 `osd` 是普通 torch 张量，不需 peer 可见，保持阶段一决定 11 的做法。这样融合实现与 NCCL 参照实现能在同一进程跑同一份输入而结果互不覆盖。

### 构造前提

`TPMoEStage1` 只有融合这一条实现（见第 3.5 节），所以下面两条是无条件的，不再按 transport 分情况：

1. 调用方必须已执行 `shmem_torch_process_group_init`。Mori 无查询接口，检查办法是调 `ms.shmem_npes()`，捕获异常并核对返回值等于 `tp_size`。这同时能抓到 shmem 通信域与 TP group 不一致的错误。
2. 必须给出 `max_tok_per_rank`，运行时 `m_local > max_tok_per_rank` 报错。

这一点修正了设计过程中的一个较早决定。当时问的是「新前提可否只限 `fused_p2p`」，答的是「可以，但只限 `fused_p2p`」。既然 NCCL 实现整个搬出 `aiter` 包，类里就没有另一条路径可以豁免了，前提自然变成无条件。搬出去的参照实现不受这两条约束，它照旧任意 `m_local`、零 Mori 依赖。

---

## 5. Kernel 内部结构与同步协议

### 5.1 grid 与角色

单个一维 grid，沿用 MegaMoE 的 ticket 机制：CTA 进来取一次 `atomic_add_agent`，号码除以 `launch_grid_x` 得 generation，余数是本轮角色编号。方案 B 无 planner，角色三类：

```
ticket 0                       owner：epoch 翻转、本地状态重置、跨卡启动握手
ticket 1..dispatch_blocks      producer：推送本卡 activation
ticket dispatch_blocks+1..     纯 consumer
```

owner 与 producer 干完各自的活后都汇入同一个 GEMM work pool。

### 5.2 执行顺序

```
owner:      翻 parity/expected -> 重置 work_head -> 向所有 peer 打 launch_ready
              -> 等所有 peer 的 launch_ready 追上本轮 -> 发布 epoch_gate
其余 CTA:   等 epoch_gate
producer:   推 x_q 行 + x_scale 行 -> fence_system_release
              -> 对 peer 的 payload_ready 做 atomic_add_system
全部 CTA:   等 payload_ready == expected -> fence_system_acquire -> 进 work pool
```

**等待只做一次，不是每个 tile 做一次。** MegaMoE 每取一个 work item 就等一次对应 expert 的 flag，因为 EP 下不同 expert 的数据来自不同源。TP 下任一 tile 的 32 行都散落在全部源上，所以进 work pool 前统一等一次即可，协议开销低于 MegaMoE。

### 5.3 双缓冲与启动握手

`launch_ready` 防的是真实的写后读冲突：`rx_x` 只有一份时，rank A 第 N+1 次的推送可能覆盖 rank B 还在读的第 N 次数据。让 A 等所有 peer 都进入第 N+1 次 launch 再写即可，因为同一条 stream 上 B 的第 N+1 个 kernel 启动意味着第 N 个已跑完。

但单缓冲会把这个握手卡在关键路径上：最慢的卡到达之前谁都不能推。按 parity 开两份之后，第 N+1 轮写的是另一份，握手只需保证无卡落后两轮以上，稳态下基本不阻塞。

### 5.4 producer 任务划分

本卡推 `m_local × tp_size` 行，包含写给自己的那份（当作本地拷贝，寻址逻辑统一，代价是 1/8 的额外流量）。

`blocks_per_destination = dispatch_blocks // tp_size`；block `b` 负责目标卡 `b // blocks_per_destination`，在该目标内按 `b % blocks_per_destination` 跨步取行；block 内一个 wave 负责一行，沿用 `row0 = warp, row_stride = num_waves`。行拷贝复用 `_copy_token_row` 的 dwordx4 模式。

每个 producer block 推完打一次 flag，因此每轮 `expected` 增加 `tp_size * blocks_per_destination`。

### 5.5 GEMM1 取数改动（本方案唯一的新 kernel 代码）

MegaMoE 的 `ATileLoader.for_tile(tile_row_base)` 假设 tile 的 32 行内存连续（EP 下推送动作完成了排列），TP 下不成立。

- **A loader**：从「一个 tile 基址 + 连续 32 行」改成「32 个独立行基址」，行基址 `token * 7168`。
- **scale loader**：从「一次 `[32, 224]` 连续块拷贝」改成「32 次 224 字节行拷贝」，行内仍合并访问。落 LDS 之后的 shuffle 与打包（`gemm_util.py:370-390`）完全不动。

**PAD 行。** `sorted_token_ids` 的填充哨兵是 `(topk << 24) | m_global`，解出的 token id ≥ `m_global`。在 `rx_x` 与 `rx_scale` 末尾各留一行清零行，取数时把越界 id 钳到该行，PAD 行乘积自然为零。这比依赖「下游会丢弃」稳妥，代价是两行内存。

### 5.6 复用清单

**照抄 MegaMoE、逻辑不变**（写进 TP 自己的 `collective_sched.py`，不改 MegaMoE 源文件，见第 6 节）：ticket 与 generation 计算、epoch/parity 协议、`epoch_gate` 本地重置、`launch_ready` 跨卡握手、分片 work pool 及 LDS 广播、`_copy_token_row`。其中 epoch/parity 那段在 TP 下要改 `expected` 的递增步长，work pool 那段要关掉逐 tile 等待，详见 6.2。

**直接调用现有实现、一行不改**：`AScaleLoader` 落 LDS 后的 shuffle 与打包、GEMM1 的 MFMA 主循环与 epilogue。

**需新写**：TP 的 producer 任务划分与推送、按 token id 取数的 A 与 scale loader、把这些接起来的 kernel 主体。

---

## 6. 调度同步代码的复用方式：拷贝，不改 MegaMoEV2

### 6.1 决定

TP 的融合 kernel 需要用到 MegaMoE 里四段与 EP 语义无关的调度同步代码。**做法是在 `aiter/ops/flydsl/kernels/mega_moe/collective_sched.py` 里写 TP 自己的一份，`mega_moe_stage1.py` 与 `dispatch.py` 一行不改。**

这推翻了本文档早先版本里「抽公共模块、旧文件改成 import」的写法，也恢复了阶段一决定 19「`MegaMoEV2` 完全冻结，纯增量」。推翻的理由见 6.3。

### 6.2 四段代码在 TP 下的差异

逐个核对过，真正逐字相同的只有前两个：

| 来源 | 行数 | TP 下是否要改 |
|---|---|---|
| `_copy_token_row`（`dispatch.py:183-201`） | 19 | 完全一样。参数只有行字节长度，推 activation 传 `model_dim`、推 scale 传 `model_dim/32` |
| ticket 与角色划分（`mega_moe_stage1.py:210-225`） | 16 | 完全一样。TP 的角色也是 owner、producer、consumer 三类 |
| 分片 work pool（`mega_moe_stage1.py:425-453`） | 28 | 结构一样，但 TP 只在进 pool 前统一等一次，要关掉挂在 `const_expr(not direct_fixed_slot)` 上的逐 tile 等待 |
| epoch/parity 与握手（`mega_moe_stage1.py:227-279`） | 45 | **真的不一样** |

最后一个的差异在 `expected` 的递增步长：

```python
next_expected = previous_expected + fx.Int32(fz_npes)
launch_epoch_lane = (next_expected // fx.Int32(fz_npes)) * fx.Int32(2) - next_parity_lane
```

EP 下每个源 rank 每个 expert 发布一次，步长是 `npes`。按 5.4 节，TP 是每个 producer block 发布一次，步长应该是 `npes * blocks_per_destination`，下面 `launch_epoch` 的推导也跟着要改。此外这段里还有三处 EP 专属的 `const_expr` 分支（`payload_tile_ready`、`external_grouping`、`direct_fixed_slot`），TP 全都用不上，所以 TP 版本会比原版短一半左右。

有一个设计选择能让它也变成完全一样：让 TP 每个源 rank 只发布**一次** ready flag，producer block 先在本地累加、最后一个 block 代表本卡向所有 peer 发布，步长就还是 `npes`。代价是多一次本地 atomic 和一次 block 间同步。这个选择留到实现时再定。

### 6.3 为什么放弃抽取

当初选择抽取的隐含前提是有一套硬守护能证明没改坏 `MegaMoEV2`：生成的 IR 逐字节相同，加上输出逐位相同。实测发现**第二条不成立**。

在与 `main` 逐字节相同的、完全未经修改的生产代码上，固定 seed、清空 JIT cache 连跑两次 `MegaMoEV2`：

```
MISMATCH rank0: 144991 bytes differ out of 1835008
...（八个 rank 全部不一致）
IR REPRODUCIBLE
```

编译产物完全一致，输出不一致。差异形态不像浮点重结合：128 行里有 19 行整行不同、其余行逐位相同，差异行的逐元素相对误差中位数约 4%，单行 relL2 最高 11%。看起来像某些 token 少算了一份 expert 贡献。

剩下的那条容差判据也不够硬。`test_mega_moe_v2.py` 的 `--rtol` 默认 0.10，是随 MegaMoE 最初的提交 `97d0c6e4c`（2026-08-10）一起进来的，代码里没有写为什么选这个数；同类量在阶段一 TP 测试里用的是 0.005 和 0.01。实测的 run-to-run 波动 2% 到 3% 塞进 0.10 还剩三四倍余量，但拿 0.005 去量每次都会红。

结论是：为 35 行完全相同的代码，去改一个自身可预测性存疑、且守护只剩一半的生产 kernel，风险与收益不成比例。拷贝的代价只是那 35 行有两份。

### 6.4 顺带记录的两个技术事实

这两条是为守护做验证时查清的，本方案不再用到，但将来真要重构 `MegaMoEV2` 时有用。

**只能比 `_ir_text`，不能比 `_source_ir`。** 后者第一行是 `#loc = loc("/abs/path.py":502:0)`，把绝对路径和行号烤进了 MLIR，函数换文件必然变；`_ir_text` 从 `module attributes {gpu.container_module}` 开始，不含任何源码位置信息。

**不能按 cache 目录名配对。** FlyDSL 的 cache key 包含 kernel 函数及其依赖的源码文本（`flydsl/compiler/jit_function.py` 的 `_jit_function_cache_key`），搬家后目录名全变，比对必须以哈希的多重集合为单位。

按这两条写的指纹工具曾提交在 `2647b3223`，因失去调用方且存在两个已知缺陷（workload 只编译出一部分时会误报成功、读的是全机共享目录）而在后续提交中删除，需要时从历史里取。

---

## 7. 测试

每个数值用例都要配负对照，证明故意搞坏之后它真的会失败。这是阶段一的教训，见前置文档第 4 节第 5 条。

| 用例 | 内容 | 负对照 |
|---|---|---|
| `case_fused_construct` | shmem 未初始化报错、`m_local > MTPR` 报错、`shmem_npes() != tp_size` 报错 | 不适用（校验类） |
| `case_fused_gather` | 推送后 `rx_x`/`rx_scale` 与 `all_gather(quantize(x))` **逐位相同** | 目标行号写成 `t * tp_size + rank`（行主序），必须失败 |
| `case_fused_numerics` | 对 `tp_moe_stage1_ref.py` 比 rel_l2 | gate/up 调换 0.75、scale 广播 bug 0.37、TP 分片窗口错位 1.33 |
| `case_fused_vs_ref` | `TPMoEStage1` 与 `TPMoEStage1NCCLRef` 同输入比结果，**设容差** | 见下 |
| `case_fused_repeat` | 连续调用十次，每次都要正确 | 去掉 epoch 翻转，必须失败 |
| `case_fused_skew` | rank 0 调用前 sleep，验证握手与双缓冲 | 关掉 `launch_ready`，必须失败 |

**`case_fused_vs_ref` 为何不能要求逐位相同：** 两条路径用不同的 GEMM1 实现（`mixed_moe_gemm_2stage_common` 对 `mega_moe/gemm1.py`），累加顺序不同。逐位相同的要求只放在 `case_fused_gather`，那里是纯拷贝，且 2.2 已证明量化顺序不影响结果。

---

## 8. 性能验收

`bench_tp_moe_stage1.py` 同时跑 `TPMoEStage1` 与 `TPMoEStage1NCCLRef`，测 m_local 64、128、256 三档。基线只测到 128，256 那档要先补参照实现的数。

**验收线：m_local=128 时总时间从 0.5365 ms 降到 0.48 ms 以内（至少快 10%）。** 达不到说明融合 kernel 自身的协议开销吃掉了收益，需要重新评估是否还值得走向方案 A。

### 8.1 实测结论：未达线

2026-08-26 实测，五档全部是融合路径更慢。

| m_local | 基线合计 | 融合合计 | 加速比 |
|---|---|---|---|
| 1 | 0.2200 | 0.3225 | 0.68× |
| 8 | 0.3906 | 0.4426 | 0.88× |
| 64 | 0.4722 | 0.5337 | 0.89× |
| 128 | 0.5188 | 0.5549 | 0.94× |
| 256 | 0.6263 | 0.6693 | 0.94× |

m_local=128 那一档要求 0.48 ms 以内，实测 0.5549 ms，不但没快 10%，反而慢了 7%。验收线没有达到。

但失败的原因不是设计文档预料的那个。文档猜的是融合 kernel 自身的协议开销吃掉收益，实测正相反：融合 kernel 在 m_local=128 时是 0.3456 ms，比基线单独跑 GEMM1 的 0.3508 ms 还快 5 µs，推送与握手是白送的。真正的原因是第 3 节分析的那一条，前三段被 host 提交成本卡死，砍掉的 GPU 工作换不来时间。

因此不该据此判定方案 A 不值得做。方案 A 把 planner 也收进 kernel，正好是继续减少 host 提交次数，方向与实测暴露出的瓶颈一致。真正需要先回答的问题是：在连续多层的推理循环里 host 能跑在 GPU 前面，这 0.2 ms 提交成本有多少是真实暴露的。当前计时协议每次迭代都 `barrier` 加 `synchronize` 把流水线抽干，是把这项成本暴露到最大的一种测法，而基线因为 GPU 工作多、队列深，受这种测法的惩罚比融合路径小。换一种连续下发的测法重新对比，是下一步该做的事，本次不做，以免变成为了凑线而改测法。

---

## 9. 已知取舍与后续

**方案 B 的结构性损失。** activation 推送本身不依赖 sort，理论上可与 `moe_sorting` 并行，但推送写在融合 kernel 内部而融合 kernel 必须等 sort 出结果才能启动，那 24 µs 被迫串行。方案 A 无此问题。若要救，可把推送拆成独立 kernel 与 `moe_sorting` 并流，多一次 launch（约 7 µs）换 19 µs 重叠，净赚约 12 µs，但这与「融合」的初衷相悖。先按单个融合 kernel 实现，实测后再定。

**本地量化与 collective 可并行。** 两者输入输出不相交，用独立 stream 加 event 可把量化那约 10 µs 藏进 collective 的 37 µs 里。收益约占总时间 2%，且与融合 kernel 正交，列为可选步骤，融合 kernel 跑通后再做，实测有效才留。需注意 NCCL 与量化 kernel 争 CU 的可能，必须实测确认无反效果。

**收益随 token 数缩水。** m_local=256 时 GEMM1 约翻倍到 0.7 ms 而固定开销基本不变，方案 A 的收益会从 29% 降到 20% 左右，B 相应更低。这是选定目标区间 64–256 时要接受的事实。
