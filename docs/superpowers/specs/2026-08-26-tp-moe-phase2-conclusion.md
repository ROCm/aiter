# TP MoE 阶段二结论与交接

日期：2026-08-26
分支：`dev/all_gather_merge_stage1_naive`（HEAD `46a17228f`）

**一句话：融合 kernel 本身做成了并且是快的，但整条流水线更慢，因为瓶颈不是 GPU 而是 host 提交，而设计文档的立论建立在瓶颈是 GPU 之上。**

---

## 1. 交付了什么

代码全部是新增的，`MegaMoEV2` 执行路径上的五个文件与 `main` 逐字节相同（阶段一决定 19 守住了）。

| 文件 | 内容 |
|---|---|
| `mega_moe/collective_sched.py` | TP 自己的调度同步 helper：`copy_row`、`emit_ticket_and_roles`、`emit_epoch_rendezvous`、`emit_launch_rendezvous` |
| `mega_moe/tp_gather.py` | `TPActivationGather`：对称内存、p2p 表、独立的推送 kernel |
| `mega_moe/tp_gemm_util.py` | `TPATileLoader`、`TPAScaleLoader`：按 token id 取数 |
| `mega_moe/tp_gemm1.py` | 独立的 TP GEMM1，复用 `gemm1.do_tile` |
| `mega_moe/tp_fused_stage1.py` | 融合 kernel：推送 + 等待 + GEMM1 一次 launch |
| `mega_moe/tp_moe_stage1.py` | `forward` 已切到融合路径，NCCL 路径与 `transport` 参数已删除 |
| `op_tests/multigpu_tests/tp_moe_stage1_nccl_ref.py` | 阶段一 NCCL 实现的副本，一次性参照 |
| `op_tests/multigpu_tests/test_tp_gather.py` | 传输层四用例 |
| `op_tests/multigpu_tests/test_tp_gemm1.py` | loader 对拍 |

十五个用例全绿：阶段一十个（含 `ref_fidelity`、`fused_numerics`）、传输层四个、loader 一个。

---

## 2. 三处逐位相同的验证

这三条是本阶段最硬的成果，任何后续改动都应该继续守着它们。

**推送 vs NCCL。** `TPActivationGather.gather()` 在 m_local 为 1、2、7、8、64、128、256 时与 `dist.all_gather_into_tensor` 逐字节相同。负对照把目标行号从 rank 主序改成行主序，八个 rank 全部报错。

**取数 vs 原版 loader。** 同一份 `tp_gemm1.py` 编译两次，参照喂 host 预排好的行走原版连续 loader，被测喂稠密数据走 gather loader，四个尺寸逐字节相同。两个负对照分别打掉 A 和 scale 两条路径。

**融合路径 vs NCCL 参照。** `case_fused_numerics` 在四个 m_local 上 `rel_l2` 恰好是 0.000000，不是「在容差内」。两条路径虽然用不同的 GEMM1 实现，但沿 K 的累加顺序一致。

---

## 3. 性能：没达线，慢 7%

验收线是 m_local=128 总时间降到 0.48 ms 以内。实测（8 卡 gfx950，30 次中位数，每次迭代前 `barrier` 加 `synchronize`，CUDA event，跨 rank 取 max）：

| m_local | 参照 | 融合 | 比值 |
|---|---|---|---|
| 1 | 0.2200 | 0.3225 | 0.68× |
| 8 | 0.3906 | 0.4426 | 0.88× |
| 64 | 0.4722 | 0.5337 | 0.89× |
| 128 | 0.5188 | 0.5549 | 0.94× |
| 256 | 0.6263 | 0.6693 | 0.94× |

没有为凑数调过任何参数。

---

## 4. 融合 kernel 本身是成功的

m_local=128 时融合 kernel 跑 0.3456 ms，参照的独立 GEMM1 是 0.3508 ms。**快 5 µs，而且同时完成了 12.85 MB 的 P2P 推送和跨卡握手。**

各尺寸下融合 kernel 相对纯 GEMM1 的增量：m_local=1 加 45 µs（固定握手成本），64 加 22 µs，128 减 5 µs，256 加 19 µs。

推送塞进 GEMM1 基本免费，这正是设计想要的。方案里担心的 producer CTA 静态划分拖尾没有出现——没有 0.19 ms 的空间给它藏。

---

## 5. 真正的失败点：成本模型少了 host 提交这一项

阶段一拟合的模型是 `T ≈ 20 µs + 17 µs × collective 次数 + bytes/275 GB/s`。它少了每个 stage 约 50 到 75 µs 的 **host 提交开销**。

证据是 bench 新增的 host 提交列。融合路径前三步的设备耗时与 host 耗时几乎相等（m_local=128：0.0762/0.0766、0.0752/0.0711、0.0578/0.0576），GPU 全程在等 CPU 派发。最直白的一条是「本地量化」在 m_local 从 1 到 256 变化 256 倍的情况下稳定 0.076 ms，这不可能是 GPU 计算。

参照实现的上游看起来便宜是因为它三次 all-gather 把 GPU 队列填满了，后面 `moe_sorting` 和量化的派发开销被藏在队列阴影里（设备 0.0133 对 host 提交 0.0522）。**消掉 GPU 活的同时也消掉了藏派发开销的阴影。** 净结果：参照暴露 0.168 ms 上游开销，融合暴露 0.209 ms，尽管融合只搬 6 KB 而参照搬 12.85 MB。

这个开销在阶段一的分段测量里被队列藏住了，所以当时没看见，成本模型也就没有这一项。

---

## 6. 两条必须记住的保留意见

**计时协议与部署场景不匹配。** 现在这套每次迭代抽干流水线的协议测的是单次延迟，最大化暴露 host 派发，而且对队列更浅的融合版惩罚更重。真实推理是多层背靠背、host 跑在前面。按 GPU 忙碌时间估算，融合约 0.36 ms 对参照约 0.52 ms，那种协议下结论很可能反过来。**这是推算不是实测**，执行时刻意没有换协议重测，因为那与调参凑数只有一线之隔。补测的话应该作为并列的第二组数据，不替换现有表格。

**这组数据支持方案 A 而不是否定它。** 既然瓶颈是 host 提交次数，那把 planner 也收进 kernel、把上游从三次提交压到一次，正是数据指向的方向。当初否掉 A 的理由是 planner 风险集中，那个判断没变，但收益的来源和大小要重估。

---

## 7. 下一步的三个选项

按我的判断排序。

**补背靠背协议的基准。** 成本最低，直接回答「现在这个结论在真实场景下成立吗」。做法是多层循环连续调用、不在迭代间同步，只在最外层同步一次，测吞吐而非单次延迟。

**削 host 提交开销。** 76 µs 一次的本地量化远超一次 kernel launch 的约 7 µs，说明大头是 Python 侧的开销（张量分配、FlyDSL 参数封送、`fx.Stream` 构造、`_run_compiled` 的分派）。先 profile 再动手，这条与 kernel 无关，收益可能立竿见影。

**走方案 A。** 把 `moe_sorting` 收进 kernel，上游提交从三次降到一次。工作量最大，但如果前两条确认瓶颈确实在 host 提交，它是唯一能根治的路。

---

## 8. 阶段二过程中挖出的、与本任务无关但重要的两件事

**`MegaMoEV2` 的输出 run-to-run 不可复现。** 在与 `main` 逐字节相同的生产代码上，固定 seed、清空 JIT cache 连跑两次，八个 rank 输出全不一致而编译 IR 完全一致。128 行里 19 行整行不同，逐元素相对误差中位数 4%，单行 relL2 最高 11%。形态不像浮点重结合，更像某些 token 少算了一份 expert 贡献。`test_mega_moe_v2.py` 一直是绿的，因为 `--rtol` 默认 0.10 而实测波动只有 2% 到 3%。主假设是 capacity 溢出丢 token（`_use_direct_fixed_slot` 写死 `npes==8 and epr==48`，正好命中该配置，slot 用远程 `atomic_add` 抢）。**未调查。**

**FlyDSL 的磁盘缓存不认类的改动。** `_get_underlying_func`（`jit_function.py:375-388`）对类返回 `None`，所以 cache key 只收集可达函数的源码。改了 loader 这类的类体之后重跑会静默用旧二进制，负对照会假通过——实测撞到过。已写进 `CLAUDE.md` rigor 规则第 6 条，三个测试文件都已设 `FLYDSL_EXTRA_SOURCE_DIRS`。**仓库里其它 kernel 测试没有设，仍然有风险。**

---

## 9. 相关文档

- 设计：`docs/superpowers/specs/2026-08-26-tp-moe-stage1-fused-p2p-design.md`（第 3 节与第 8 节已用实测数字更新）
- 阶段一交接：`docs/superpowers/specs/2026-08-25-tp-moe-stage1-phase2-handoff.md`
- 四份实施方案：`docs/superpowers/plans/2026-08-26-tp-moe-phase2{a,b,c,d}-*.md`。其中 2a 已作废（抽取路线被否），文件开头有说明。
