# TP MoE Stage1：把跨卡搬运改成接收端触发（pull）

日期：2026-08-31。分支 `dev/all_gather_merge_stage1_naive`，本地分支，未推送。

## 1. 改了什么

阶段二的融合 kernel 原来是发送端触发：每张卡的 staging CTA 把自己量化好的行写进所有对端的 receive buffer，写完由最后一个 CTA 去点亮对端的 `payload_ready`。现在增加了接收端触发的方向，并且把它设为默认。每张卡的 staging CTA 各自认领一个**源** rank，把那张卡 `tx` 缓冲区里的行读回本卡的 receive buffer。两个方向搬的字节数、落的偏移完全一样，所以下游 GEMM 看到的 A 操作数逐位相同。

方向由编译期常量 `pull` 选择，两个变体的 kernel 名不同，因此 `functools.cache` 和 FlyDSL 磁盘缓存都能区分。`TPMoEStage1(pull=...)` 一路传到 `compile_tp_fused_stage1`。保留 push 不是为了兼容，是因为它是目前唯一能做受控对比的参照实现。

## 2. pull 为什么不需要 payload_ready

push 必须有 `payload_ready`，因为写数据的是远端的 CTA，本地无从知道对方写完没有。pull 反过来，读数据的是本地 CTA，需要确认的只有一件事：源 rank 的量化 kernel 是否已经把 `tx` 填好。

这个条件已经被 `emit_launch_rendezvous` 覆盖了。看到某个 peer 发布的 launch epoch，意味着那个 peer 的融合 kernel 已经进入本轮；单条 stream 上量化 kernel 排在融合 kernel 前面，所以它必然已经退休。可见性方面，peer 在发布 epoch 之前有一次 `fence_system_release`，会把它 L2 里的量化结果写回去；本卡 owner CTA 在等到之后有一次 `fence_system_acquire`，staging CTA 在开始远端读之前又各自补了一次。

所以 pull 把 `payload_ready` 和维护它的那 `npes` 次 system-scope atomic 全部删掉了。换进来的是一个**设备内**的计数器：staging CTA 做完各自的搬运后加一，所有 CTA 等它涨到 `producer_blocks` 才进 GEMM。写者都在本设备上，用 agent scope 就够，不需要 system scope。

## 3. 代价：量化结果必须落在对称内存里

push 时每张卡只读自己的量化输出，普通 torch 张量即可。pull 要求 peer 能读到它，所以那块内存必须是 Mori 的对称分配，而且 `shmem_ptr_p2p` 的指针表在构造时就固定了，不能每次调用换地址。

`TPActivationGather` 因此多了一对常驻对称缓冲 `tx_x` / `tx_scale`，形状 `[2, max_tok_per_rank, row_bytes]`。量化 kernel 直接往里写，靠 `per_1x32_mx_quant` 新增的可选参数 `out=` / `scale_out=` 实现，这两个参数是纯增量的，不传就完全是原来的行为。之所以不用「量化完再拷贝进去」，是因为那样每次调用要多两次 kernel launch，而融合这件事本来就是为了省 launch。

`TPFusedStage1Runner.run` 里还有一个兜底：如果传进来的张量不在 `tx` 里（测试和 bench 会这样传），就替它拷一次。生产路径 `forward` 走的是直接写入，不会触发。

pull 强制要求双缓冲。它把缓冲区复用的风险方向反过来了，要排序的不再是「peer 写我的 receive buffer」和「我读」，而是「peer 下一轮的量化」和「我这一轮的读」。launch rendezvous 排不了这一对，因为量化在 peer 那条 stream 上排在 rendezvous 前面。两个 slot 才能拉开：peer 要到第 N+2 轮才会重写 slab p，那时我第 N 轮的 kernel 早已退休。`enable_pull=True` 配 `double_buffer=False` 会在第一次集合分配之前直接报错。

## 4. 正确性

三项，都在 8 卡 gfx950 上跑过，前两项各配了一个实测会红的负对照。

**pull 与 push 逐位相同。** `test_tp_moe_stage1.py --case pull_vs_push`，m_local 取 1/8/64/128，比 routed 行的 payload 和 mx scale。负对照是把 pull 的落点从 `source * m_local + row` 改成行主序的 `row * npes + source`，实测在 m_local=8 报 `127938 of 147456 bytes` 不符。m_local=1 两个公式恰好等价，所以负对照从 m_local=8 才开始报，这符合预期。

**pull 与 NCCL 参照逐位相同。** `--case ref_fidelity` 现在默认走 pull，仍然通过。`--case fused_numerics` 的 `rel_l2` 是 0.000000。

**歪斜下的握手是有效的。** 新增 `--case pull_skew`：rank 0 每轮 sleep 50 ms 制造歪斜，循环里不含任何集合操作（集合操作是八卡栅栏，会把刚造出来的歪斜抹平，push 那边的同类测试就栽过一次）。参照是 lockstep 下跑的 push 结果，不是 NCCL 参照，理由见下一节。负对照是把 `emit_launch_rendezvous` 里的跨卡等待摘掉、只留 gate 和计数器清零，实测第 1 轮就报错，各 rank 分别是 442342 到 1032122 字节不符，量级和 rank 间的差异都符合真实竞争的形态。

## 5. 顺带发现：融合路径与 NCCL 参照存在一处 1 ULP 差异

写 `pull_skew` 时最初拿 NCCL 参照当 oracle，第 5 轮稳定报「1179648 字节里差 1 个字节」，八个 rank 完全一致，位置固定在 `sorted_row=9761` 第 301 列，融合侧 86 参照侧 85，是相邻的 fp8 编码。

把 pull 换成 push 重跑，同一个字节、同一个值。**所以这不是 pull 引入的**，是融合 kernel 与 NCCL 参照之间既有的一处舍入差异，只是 `case_ref_fidelity` 测的那三组输入没碰上。目前没有调查，`pull_skew` 改用 push 当 oracle 绕开了它。如果以后要把「融合与参照逐位相同」写成硬约束，这一条得先查清楚。

## 6. 性能：只有 m_local=256 有可信信号

同一份 `do_tile`、同一套 tile 配置、同一份 loader，唯一区别是搬运方向。下表是融合 kernel 本身的 device 时间，单位 µs，30 次迭代取中位数再跨 rank 取最大，整套 bench 重复跑了 4 遍。

| m_local | pull（4 次） | push（4 次） | delta（pull − push） |
|---|---|---|---|
| 1 | 125.4 / 138.4 / 136.3 / 131.5 | 122.4 / 125.2 / 122.6 / 129.9 | +3.0 / +13.2 / +13.7 / +1.6 |
| 8 | 242.7 / 255.0 / 272.8 / 252.1 | 236.7 / 242.3 / 264.4 / 252.6 | +6.0 / +12.7 / +8.4 / −0.6 |
| 64 | 322.5 / 376.5 / 479.3 / 348.4 | 382.5 / 369.2 / 336.2 / 345.3 | −60.0 / +7.3 / +143.1 / +3.1 |
| 128 | 400.8 / 400.1 / 342.9 / 349.2 | 402.5 / 374.8 / 357.5 / 363.4 | −1.7 / +25.3 / −14.6 / −14.2 |
| 256 | 426.7 / 433.6 / 425.5 / 429.3 | 445.4 / 456.3 / 451.4 / 453.6 | −18.7 / −22.6 / −25.9 / −24.3 |

只有 m_local=256 这一行的符号是 4 比 0 一致的，pull 快 19 到 26 µs，约 5%，四次的极差只有 7 µs。m_local=1 和 8 也基本一致地朝另一个方向，pull 慢 1 到 14 µs，这符合直觉：数据量很小的时候搬运本身不是瓶颈，而 pull 多了一个 1024 个 CTA 一起等的设备内计数器。

m_local=64 和 128 这两行**不构成任何结论**。64 那行四次分别是 −60、+7、+143、+3，噪声远大于任何可能的效应。这与上一轮记录的现象一致：这套计时协议每次迭代抽干流水线再跨 rank 取最大，尾部噪声被放大。想在中间尺寸上得出结论，得先把测量稳定下来，那是阶段二结论文档第 7 节第二条。

一句话总结：pull 在最大尺寸上确实赚了约 5%，在最小尺寸上小亏，中间说不清。它还顺带删掉了一整套 ready flag 协议，这个简化本身有价值，和快慢无关。

## 7. 没做的事

没有查融合为什么整体上仍然比两次 launch 慢，那仍然是阶段二结论文档第 7 节的第一条。本次改的是通信方向，不触及融合边界带来的开销。

没有测方案一（接收端直接在 GEMM loader 里读远端、完全不落地）。在当前 `topk=6`、`N_TILES=3` 的配置下，每行原始 activation 会被读 18 次，pull 到寄存器意味着其中 7/8 走 XGMI，跨卡流量是现在的 18 倍。这个量级不值得实现来测。

`op_tests/multigpu_tests/test_tp_gemm1.py` 现在会 OOM。已确认这与本次改动无关：`git stash` 回到改动前跑同样命令，同样 OOM。

## 8. 改动清单

- `aiter/ops/flydsl/kernels/mega_moe/quant.py`：`per_1x32_mx_quant` 增加可选的 `out=` / `scale_out=`。
- `aiter/ops/flydsl/kernels/mega_moe/tp_gather.py`：`TPActivationGather` 增加 `enable_pull`、对称缓冲 `tx_x` / `tx_scale`、p2p 指针表，以及 `tx_views` / `is_tx_view` / `stage_source`。
- `aiter/ops/flydsl/kernels/mega_moe/tp_fused_stage1.py`：编译期 `pull` 开关，pull 分支的 staging 与设备内完成计数，kernel 签名增加四个 pull 专用参数。
- `aiter/ops/flydsl/kernels/mega_moe/tp_moe_stage1.py`：`pull` 构造参数，`quantize` 接受 `m_local` 以便直接写进对称 slab。
- `op_tests/multigpu_tests/test_tp_moe_stage1.py`：新增 `pull_vs_push` 与 `pull_skew` 两个 case。
- `op_tests/multigpu_tests/bench_tp_moe_stage1.py`：并列的 pull / push 完整 forward 两臂；受控的「两次 launch vs 一次 launch」对比仍然跑在 push 上，以便与已记录的数字可比。
