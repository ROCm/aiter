# TPMoEStage1 阶段二交接文档

**日期：** 2026-08-25
**分支：** `dev/all_gather_merge_stage1`
**状态：** 阶段一已完成并验证，阶段二尚未开始

这份文档的目的是让阶段二的工作可以在没有前一轮对话上下文的情况下继续。所有数字和结论都是实测过的，不是推断。

---

## 1. 阶段一交付了什么

新增算子 `TPMoEStage1`，位于 `aiter/ops/flydsl/kernels/mega_moe/tp_moe_stage1.py`。它是 TP4/TP8 版本的 MoE Stage1，接收 DP 切分的 token 分片，内部完成 all-gather，输出对齐 aiter 通用 v2 FMoE GEMM2 的 ABI。

相对 main：5 个文件、3087 行新增、0 行删除。

```
aiter/ops/flydsl/kernels/mega_moe/tp_moe_stage1.py   405 行（新）
aiter/ops/flydsl/kernels/mega_moe/__init__.py        +2 行（唯一改动的既有文件）
op_tests/multigpu_tests/test_tp_moe_stage1.py        749 行（新，8 个用例）
op_tests/multigpu_tests/tp_moe_stage1_ref.py         202 行（新，torch fp32 参考）
docs/superpowers/plans/2026-08-24-tp-moe-stage1.md  1729 行（实施方案）
```

阶段一**没有写任何新 kernel**。它把现成组件串起来：

```
forward:          all-gather(BF16) → moe_sorting → fused_dynamic_mxfp8_quant_moe_sort
                  → flydsl_moe_stage1(v2_output_layout=True)
forward_prequant: all-gather(FP8+scale) → moe_sorting → moe_mxfp4_sort（只排 scale）
                  → flydsl_moe_stage1(v2_output_layout=True)
```

### 公开接口

```python
TPMoEStage1(
    model_dim, inter_dim, experts, topk, w1, w1_scale,   # inter_dim 是本 rank 分片
    group=None, tp_size=None, tp_rank=None,
    device=None, sort_block_m=32, swiglu_limit=0.0,
    stage1_kernel_name="flydsl_moe1_afp8_wfp4_bf16_t32x64x256_w4_gui_xcd4_kw4_fp8",
    transport="allgather_bf16",
)
.forward(x_bf16, route_weights, topk_ids) -> TPMoEStage1Output
.forward_prequant(x_fp8, x_scale, route_weights, topk_ids) -> TPMoEStage1Output
.quantize(x_bf16) -> (fp8, e8m0)
.m_logical_for(m_local) / .max_sorted_for(m_local)
```

`TPMoEStage1Output` 是 frozen dataclass，六个 tensor 加七个 host 标量：
`inter_sorted_quant`、`inter_sorted_shuffled_scale`、`sorted_token_ids`、`sorted_weights`、`sorted_expert_ids`、`num_valid_ids`、`m_logical`、`max_sorted`、`num_experts`、`model_dim`、`inter_dim`、`topk`、`sort_block_m`。

私有辅助方法（阶段二会复用）：`_all_gather_one`、`_all_gather_inputs`、`_validate_call`、`_sort`、`_run_gemm1`、`_pack`。

### 实测数值

| 用例 | 实测 | 阈值 |
|---|---|---|
| numerics（Stage1 对 torch fp32 逐行） | 0.00050 | 0.005 |
| prequant（两个入口等价性） | 恰好 0 | 1e-3 |
| e2e（Stage1 + GEMM2 + reduce-scatter 对全量 MoE） | 0.0040 | 0.01 |
| MegaMoEV2 冻结门（bs=128 / 512） | 0.057 / 0.053 | 0.10 |

e2e 的 0.0040 里有约 0.0031 是 bf16 atomic epilogue 的精度地板，实测隔离出来的，不是 bug。

---

## 2. 已敲定的设计决定

这 22 条是通过 grilling 逐条确认的，阶段二**不要重新讨论**，除非有新事实推翻。

1. 做 TP8/TP4 Stage1，不做 EP 中间物。`dev/tp_fuse_gemm1_v0` 分支忽略（用户判断其不可信，不追问原因）。
2. TP 语义：每卡持有全部 384 个 expert，`inter_dim` 切 tp 份。TP8 每卡 384，TP4 每卡 768。
3. 算子站在 all-gather **内侧**，自己做 gather。
4. `tp_size` 可配，只接受 4 和 8。
5. 有状态 class，落在 `mega_moe/tp_moe_stage1.py`。
6. `group=None` 默认 WORLD；device 用 `torch.cuda.current_device()`，不用 rank。
7. 两个入口 `forward` / `forward_prequant`，prequant 为主。
8. W1/W1_scale 由调用方传入已 shuffle 的。
9. `swiglu_limit` 暴露，默认 0.0。
10. `sort_block_m` 暴露，默认 32。
11. 输出每次调用新分配，不复用。
12. 各 rank token 数必须相等，文档化前提，不做运行时检查。
13. 输出对齐 v2 FMoE GEMM2，A 是 FP8 E4M3。
14. 下游由调用方直接调 `mxfp4_moe_gemm2`，不碰 `aiter/configs/`。
15. `sorted_expert_ids` 是 local expert id（TP 下等于 global id）；`sorted_token_ids` 是 `(slot<<24) | dense_global_token`。
16. 阶段一 baseline 无新 kernel。
17. 阶段二融合版 API 不变，`transport` 是接口位。
18. baseline gather BF16；`forward_prequant` 已经实现了 gather FP8。
19. `MegaMoEV2` 完全冻结，纯增量。
20. 验收用 torch fp32 逐行参考加端到端。
21. 不接 CI，本地 torchrun 8 卡验证。
22. DP group == TP group，`M_global = tp_size * M_local`。

下游目标 kernel 是 `flydsl_moe2_layout_afp8_wfp4_bf16_t32x128x128_atomic_sbm32`，来自 `docs/fp8_retune_config/dsv4_fp8fp4_tp8_k384_flydslv2_tuned_20260726_144002.csv:3`，解码为 `BM=32, BN=128, BK=128, epilog="atomic", SBM=32`。

模型形状是 DSV4-Pro：`model_dim=7168`、`experts=384`、`topk=6`、`swiglu_limit=10.0`，TP8 下 `inter_dim=384`。

---

## 3. 阶段二会用到的已验证事实

### MegaMoE Stage1 kernel 的结构

`compile_mega_moe_stage1`（`mega_moe_stage1.py:70-81`）的顶层 kernel 用 atomic ticket 分角色：

- 线程 0 对 `entry_count` 做 `atomic_add_agent`，`generation = ticket64 // launch_grid_x`，`ticket = ticket64 - generation*launch_grid_x`
- `ticket == 0` 是 owner/planner，`0 < ticket <= dispatch_blocks` 是 producer，其余是 consumer
- owner 翻转 parity、增加 expected、做跨 rank `LAUNCH_READY` 握手、初始化 work heads，然后发布 `epoch_gate`
- 所有 CTA 干完各自的活之后**都会并进同一个 GEMM work pool**，角色只是入口分工
- `grid_x = num_cu*grid_mult - planner_blocks - dispatch_blocks`

ready flag 有两个作用域。跨 rank 的用 system release/acquire：`LAUNCH_READY`、`COUNT_DONE`、`PAYLOAD_READY`、`TILE_READY`。本卡 CTA 之间的用 agent release/acquire：`PAIR_READY`、`PAIR_ORDER_READY`、`GROUP_DONE`、`epoch_gate`。`PLAN_READY` 是混合的。

### Stage1 和 Stage2 之间的缝

**两者之间没有任何共享的 epoch/parity 或 ready flag，只靠 stream ordering。** 契约就是 8 个 tensor：`_s1_out`、`_s1_osd`、`sorted_expert_ids`、`tile_row_base`、`num_valid`、`srcmap_em`、`wts_em`、`max_expert_tiles`。

### 最硬的耦合

`MegaMoEV2` 的 `self._s1_op = self.comb_op._gm`（`mega_moe_v2.py:78-80`）。Stage1 用的 `rx_em` / `scale_em` / `srcmap_em` / `wts_em` / `sorted_expert_ids` / `tile_row_base` / `num_valid` 全部归 combine op 内部的 `FlyDSLDispatchGroupMajorOp` 所有。阶段二如果要复用这套 buffer，要么直接构造 group-major op，要么绕开。

附带一个坑：`mega_moe_v2.py:83-86` 会回头改写 combine op 的 `max_blocks` 并重新分配它的两个 metadata tensor。

`FlyDSLDispatchGroupMajorOp` 在 main 上**没有** `local_only` 参数。被作废的 `dev/tp_fuse_gemm1_v0` 分支给它加过一个（`local_only=True` 时用普通 torch tensor 而不是 Mori symmetric，并跳过 `shmem_barrier_all`）。阶段二如果需要类似能力，要自己加，而且要注意决定 19 说 `MegaMoEV2` 冻结。

### 命名陷阱

Stage1 kernel 的形参叫 `sorted_token_ids`，但 `MegaMoEV2` 实际传进去的是 `op.tile_row_base`（`mega_moe_v2.py:236-242`）。名字和语义对不上。

### 数值和布局

- `_s1_osd` 的 scale shuffle 地址算术和 v2 的 sorted scale **逐位相同**，也等于 C++ 的 `mx_scale_shuffle_idx`（`csrc/include/mx_quant_utils.h:212-217`）。穷举验证过，0 处不一致。公式是
  `byte_off = (x>>5)*(S*32) + (y>>3)*256 + (y&3)*64 + (x&15)*4 + ((y>>2)&1)*2 + ((x>>4)&1)`，其中 `S = pad8(inter_dim/32)`。
- `TPMoEStage1Output.max_sorted` 是 **payload 行数**（`sort_block_m` 对齐后），不是 `sorted_ids` 的长度。两者在本配置下永不相等，因为 `384*32` 是 32 的倍数而 `6*(m_global-1)` 不是。生产路径 `aiter/fused_moe.py:2018` 用的就是 payload 行数。
- `moe_sorting` 只写 `[0, num_valid_ids[0])`，后面是 `torch.empty` 垃圾。任何内容断言都必须限定 `[:nvalid]`。`nvalid` 一定是 `sort_block_m` 的倍数。
- GEMM2 用 `token_id < i32_M` 门控 store（`mxmoe_gemm_v2.py:1006-1008`），所以 padding 行未初始化的 scale 不会落盘。
- `inter_dim=384` 有 12 个 scale group，pad 到 16。BK=128 时 GEMM2 **读不到** group 12-15，已用 poison 加同 scale 对照实测确认。BK 改成 256 会重新命中这个风险。
- `get_torch_quant(per_1x32)` 返回的 weight scale 是 **2D**，会把前导维压平。`w1 [E,2I,H]` 的 scale 是 `(E*2I, H/32)`，`w2 [E,H,I]` 的是 `(E*H, I/32)`。直接 `scale[e]` 会静默广播一行，形状对但数值错。
- `shuffle_scale_a16w4` 强制要求 2D 输入；`shuffle_weight_a16w4` 保持形状不变。W1 用 `True`，W2 用 `False`。
- `torch.cat` 对 `float8_e8m0fnu` **没有 CUDA kernel**，要先 `.view(torch.uint8)` 再拼。
- `moe_sorting(..., accumulate=False)` 会让 `moe_buf` 变成 `(0,0)` 占位并跳过清零 pass（`aiter/fused_moe.py:326-334`）。`_sort` 已经这么用了。

### 环境

这台机器有 8 张 gfx950。`mori.shmem` 和 `mori.ops` 可用（版本 1.2.1.dev20260619），但 **`mori.cco` 和 `mori.ops.dispatch_combine_v2` 装不了**（`ModuleNotFoundError`）。阶段一完全不需要 Mori，只用 `torch.distributed`；阶段二如果要做 kernel 内 P2P，就会需要 Mori symmetric memory。

CI 有 black gate（`.github/workflows/pre-checks.yaml:28-35` 跑 `psf/black@stable`）。提交前必须跑 `black` 和 `ruff check`。

---

## 4. 阶段一踩过的坑

这些是我写的实施方案里的**真错误**，全部由执行 agent 拒绝糊弄而暴露。记下来是为了避免阶段二重犯同类。

1. `max_sorted` 写成 `sorted_ids.shape[0]`，应该是 payload 行数。
2. 断言写了全张量，但 `moe_sorting` 只写 `[:nvalid]`。测出来 `m_local=8` 时尾部有 80 行垃圾值。
3. `dequant_w1_expert` 用了 `w1_scale[expert_id]`，而 scale 是 2D，静默广播。权重往返误差 0.44，修正后 0.115。
4. 参考实现在逐行循环里反量化权重，`m_local=128` 时会重复 6144 次 `[768,7168]` 的展开。改成按 expert 单条目缓存。
5. **`max(0.0, nan) == 0.0`**，且 `nan >= 0.05` 为 False。主数值门会把 NaN 静默吞掉。实验证明去掉守卫后注入 NaN 的测试会打印 `rel_l2=nan` 然后报 OK 并 exit=0。

教训是：每个数值用例都要配负对照，确认故意搞坏之后测试真的会失败。阶段一的三个数值用例都做了这件事，负对照分别是 0.75（gate/up 调换）、0.37（scale 广播 bug）、1.33（TP 分片窗口错位）。

---

## 5. 阶段二要做什么

目标是把 all-gather 融进 kernel 内部，让 GEMM1 能在数据边到边算，而不是等 all-gather 整体完成。

对外 API 不变。要做的是：

1. 放开 `transport="fused_allgather"` 分支，现在它抛 `NotImplementedError`。
2. 写一个新的 fused kernel，照 `mega_moe_stage1.py` 的 ticket/epoch 调度骨架，把 `dispatch.py` 的 `emit_dispatch_*` 换成 push-based all-gather，让 GEMM consumer 按 tile 等数据到位。
3. 复用现成的 `_pack()` 产出同一个 `TPMoEStage1Output`。

### 对拍方式

在同一个进程里构造两个实例，`transport` 分别是 `"allgather_bf16"` 和 `"fused_allgather"`，喂同一份输入，逐行比 `inter_sorted_quant * scale`。因为决定 11 是每次调用新分配，两份结果不会互相覆盖。

### 已有的性能基线

`forward`（gather BF16，每行 14336 字节）和 `forward_prequant`（gather FP8 加 scale，每行 7392 字节）的耗时差，就是「先 quant 再 gather」单独的收益。融合版再叠加通信计算重叠的收益。这两个数阶段一还没实测，阶段二应该先量出来，否则不知道重叠到底值多少。

### 尚未决定的问题

这些在阶段二开工前需要确认：

- 传输机制用什么。Mori SHMEM P2P 是最接近 MegaMoE 现有做法的，但阶段一完全没引入 Mori 依赖，加进来会改变算子的构造前提（需要外部先 `shmem_torch_process_group_init`）。
- 重叠的粒度。MegaMoE 的 compact path 支持按 expert 等 `PAYLOAD_READY`，也支持按 tile 等 `TILE_READY`。TP 下没有 expert 分组这回事（每卡全部 expert），所以粒度怎么定需要重新想。
- 要不要复用 `dispatch.py` 里的 `DispatchSlot` 和同步原语，还是另起一套。复用的话要处理决定 19 的冻结约束。
- `FlyDSLDispatchGroupMajorOp` 要不要加 `local_only`。TP 的 all-gather 需要 peer 可见的接收 buffer，但不需要 EP 那套 srcmap/wts_em。

### 已完成的基线测量（2026-08-25）

用 `op_tests/multigpu_tests/bench_tp_moe_stage1.py` 在 8 卡 gfx950 上量的，中位数，跨 rank 取 max，每次迭代前有 barrier 所以读数不含 rank skew。

BF16 入口的分段耗时（毫秒）：

| m_local | all-gather | moe_sorting | 量化 | GEMM1 | 总计 |
|---|---|---|---|---|---|
| 1 | 0.0749 | 0.0540 | 0.0203 | 0.0763 | 0.2256 |
| 8 | 0.0837 | 0.0557 | 0.0269 | 0.2372 | 0.4034 |
| 64 | 0.1002 | 0.0446 | 0.0413 | 0.3033 | 0.4894 |
| 128 | 0.1226 | 0.0191 | 0.0419 | 0.3530 | 0.5365 |

**关键结论一：all-gather 是延迟瓶颈，不是带宽瓶颈。** 拟合出来是
`T ≈ 20 µs + 17 µs × collective 次数 + bytes / 275 GB/s`。m_local=128 时 12.85 MB 的边际传输只花 46 µs，而三次 collective 的固定开销是 70 µs。

**关键结论二：决定 18 的前提不成立。** `forward_prequant` 虽然跨卡字节减半，但它为了单独传 scale 多做一次 collective，实测每一档都比 BF16 入口慢。「先 quant 再 gather 省带宽」这个理由在当前 token 规模下不成立。

**关键结论三：一次 kernel launch 约 7 µs。** 证据是解包两个小张量的耗时在 m_local 变化 128 倍时始终是 0.014 ms。

### 试过但无效的两条路（不要重复）

**`torch.distributed._coalescing_manager` 更慢。** m_local=128 时 0.2113 ms，比不用它的 0.1178 差了将近一倍。

**把所有输入打包成一个 buffer 做单次 all-gather，净收益接近零。** 拆解数据（m_local=128，BF16）：

```
3 次独立 collective   0.1148 ms
cat 打包              0.0107
1 次 collective       0.0806
解包 x                0.0159
解包 3 个小张量        0.0141
打包版合计            0.1051   ← 只省 10 µs，约总时间的 2%
```

省下 2 次 collective 的 34 µs，但 cat 加三次解包拷贝要花 41 µs。这一版实现过并跑通了全部数值用例，因为收益太小且 fp8 入口在小 m_local 反而变慢，已经退回。

### 对阶段二立论的修正

原来的立论是「重叠通信和计算，天花板 22% 到 32%」。测完之后这个说法**低估了收益，归因也不对**。

融合 kernel 真正消掉的是**启动开销**，不只是把通信藏起来。它把 gather、sort、量化、GEMM1 合成一次 launch，三次 collective 的 51 µs 和中间几次 kernel launch 一起消失。m_local=128 时非 GEMM1 的部分合计 0.1836 ms，全部吸收进 GEMM1 的话总时间从 0.5365 降到 0.353，是 34%。m_local=1 时非 GEMM1 部分占四分之三，比例更高。

这也解释了为什么打包和 coalescing 都无效：它们只是把开销在 collective 和 kernel launch 之间搬来搬去，总量不变。只有合成一个 kernel 才能真正消掉。

打折扣的地方是融合 kernel 自身的调度开销不是免费的，MegaMoE 那套 ticket、epoch、ready flag 协议要花时间，所以 34% 是上限而不是预期值。
