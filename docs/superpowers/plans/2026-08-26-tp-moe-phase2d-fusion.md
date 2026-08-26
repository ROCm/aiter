# TP MoE 阶段二（下之二）：推送与 GEMM1 融合 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把 P2P 推送和 GEMM1 合成一次 kernel launch，让 `TPMoEStage1.forward` 走这条路，并测出它比阶段一的 NCCL 版快多少。

**Architecture:** 两块地基都已单独验证过：`TPActivationGather` 的推送与 NCCL 逐位相同，`TPATileLoader`/`TPAScaleLoader` 的按 token 取数与原版 loader 逐位相同。本方案把它们接成一个 kernel——owner 翻 epoch、producer 推送、全体等数据到齐、再一起跑 GEMM1 的 tile 循环——然后切换算子入口、删掉旧路径、量性能。

**Tech Stack:** Python 3.12、PyTorch、FlyDSL、Mori SHMEM、ROCm gfx950、8 卡 torchrun。

**依据文档：** `docs/superpowers/specs/2026-08-26-tp-moe-stage1-fused-p2p-design.md` 第 3 节、5.1 到 5.2 节、第 7 到 8 节。

**分支：** `dev/all_gather_merge_stage1_naive`。

---

## File Structure

| 文件 | 动作 | 职责 |
|---|---|---|
| `aiter/ops/flydsl/kernels/mega_moe/tp_gather.py` | 修改 | parity 移到 host，去掉每次调用的 GPU 同步 |
| `aiter/ops/flydsl/kernels/mega_moe/tp_fused_stage1.py` | 新建 | 融合 kernel：推送 + 等待 + GEMM1 |
| `aiter/ops/flydsl/kernels/mega_moe/tp_moe_stage1.py` | 修改 | `forward` 切到融合路径，删 NCCL 路径与 `transport` |
| `op_tests/multigpu_tests/test_tp_moe_stage1.py` | 修改 | 阶段一用例改跑 `TPMoEStage1NCCLRef` |
| `op_tests/multigpu_tests/bench_tp_moe_stage1.py` | 修改 | 加融合模式，测 m_local 64/128/256 |

**已有的三个测试文件不动**：`test_tp_gather.py`、`test_tp_gemm1.py` 继续守着两块地基，`tp_moe_stage1_nccl_ref.py` 是对拍参照。

---

## 一条必须先设的环境变量

**每个新写的 kernel 测试文件，顶部、`import flydsl` 之前，必须设 `FLYDSL_EXTRA_SOURCE_DIRS` 指向 `mega_moe` 包目录。**

FlyDSL 的 cache key 只收集可达**函数**的源码，`_get_underlying_func`（`jit_function.py:375-388`）对**类**返回 `None`，所以改了 `TPATileLoader` 这类的类体之后，`~/.flydsl/cache` 会原样返回上一次编译的二进制。方案三执行时实测撞到过：删掉 topk-slot 掩码后逐位对拍照样绿，因为 kernel 根本没重编。

`test_tp_gemm1.py` 里已经有现成写法，照抄。

---

## Task 1: parity 移到 host

**Files:**
- Modify: `aiter/ops/flydsl/kernels/mega_moe/tp_gather.py`

### 为什么

`gather()` 现在调 `current_parity()`，它做 `int(self.epoch_parity[0].item())`——一次 GPU 到 CPU 的同步，卡在每次调用的关键路径上。实测（8 卡 gfx950，中位数，跨 rank 取 max）：

| m_local | gather 总耗时 | `.item()` 同步 | 占比 |
|---|---|---|---|
| 8 | 0.0581 ms | 0.0153 ms | 26.3% |
| 128 | 0.0849 ms | 0.0152 ms | 17.9% |
| 256 | 0.1144 ms | 0.0145 ms | 12.7% |

稳定 15 µs。融合之后这条路径就是 `forward` 的关键路径，而本次改造的全部意义就是省这个量级的开销。

### 为什么 host 算得出来

parity 是确定的。每张卡调用 `forward` 的次数相同（这是 TP 的语义前提，阶段一决定 1 就写明了 DP group 等于 TP group），所以 host 自己数轮次即可，不需要问设备。

第 `r` 轮（从 0 起）用 parity `r % 2`。某个 parity `p` 在第 `r` 轮结束时被用过 `(r - p) // 2 + 1` 次，所以 `expected = ((r - p) // 2 + 1) * npes`。

这样 `epoch_parity` 和 `epoch_expected` 两个设备张量都不再需要，`emit_epoch_rendezvous` 里翻转 parity 那段也不再需要。

- [ ] **Step 1: 改 `__init__`**

删掉这两行：

```python
        self.epoch_parity = torch.zeros(1, dtype=torch.int32, device=dev)
        self.epoch_expected = torch.zeros(2, dtype=torch.int32, device=dev)
```

加一行：

```python
        # Round counter kept on the host: parity is deterministic (every rank
        # calls the same number of times), so deriving it here avoids a
        # GPU->CPU sync per call. Measured cost of that sync: ~15us, 13-26% of
        # the gather.
        self._round = 0
```

- [ ] **Step 2: 改 `current_parity`**

```python
    def current_parity(self):
        """Parity the NEXT gather() will write. Host-derived, no device read."""
        return (self._round % 2) if self.slots == 2 else 0

    def _expected_for(self, parity):
        """How many source-rank publishes payload_ready[parity] will hold after this round."""
        return ((self._round - parity) // 2 + 1) * self.tp_size
```

`slots == 1` 时 parity 恒为 0，`_expected_for(0)` 退化成 `(self._round // 2 + 1) * tp_size`，与设备端每轮加 `npes` 的行为不符。**所以单缓冲模式下 `_expected_for` 必须返回 `(self._round + 1) * self.tp_size`。** 实现时按 `self.slots` 分支处理，两条都要写。

- [ ] **Step 3: 改 `gather()`**

`parity` 和 `expected` 变成 kernel 参数，`addr_parity` / `addr_expected` 两个参数删掉。函数末尾加 `self._round += 1`。

注意 `self._round += 1` 必须在 `_run_compiled` **之后**，且即使 `_validate` 抛异常也不能自增——所以放在成功路径的最后一行，`return` 之前。

- [ ] **Step 4: 改 kernel**

`compile_tp_gather` 的签名去掉 `addr_parity`、`addr_expected`，加上 `parity: fx.Int32` 和 `expected: fx.Int32`。kernel 体内：

- 删掉 `parity_rsrc` / `expected_rsrc` 的构造，以及从它们读 `parity` / `expected` 的两行。
- `emit_epoch_rendezvous` 的调用改成不再传 `parity_rsrc` / `expected_rsrc`。

`collective_sched.emit_epoch_rendezvous` 需要一个不翻 parity 的变体。**不要改现有的那个**，`test_tp_gather.py` 之外还没有别的调用方，但保持它可用更省事。新加一个：

```python
@flyc.jit
def emit_launch_rendezvous(*, tid, is_owner, p_launch_ready, a_launch_ready,
        a_reset_counters, reset_count, gate_addr, gate_epoch, launch_epoch,
        npes, rank):
    """Rendezvous with every peer, reset per-launch counters, open the gate.

    Same as emit_epoch_rendezvous minus the parity flip: the caller derives
    parity and launch_epoch on the host, which removes a GPU->CPU sync from the
    per-call path. The peer wait still does the real work -- it is what stops
    rank A's round N+1 push from landing in a buffer rank B is still reading in
    round N, because on a single stream B entering round N+1 means B's round-N
    kernel retired.
    """
    if is_owner:
        if tid < fx.Int32(npes):
            peer = (tid + fx.Int32(rank)) % fx.Int32(npes)
            comm_ops.fence_system_release()
            launch_ready_table = _make_buffer_from_addr(p_launch_ready, fx.Int64)
            remote_launch_ready = _buffer_load(launch_ready_table, peer, fx.Int64)
            comm_ops.store_i32_system(remote_launch_ready, fx.Int32(rank), launch_epoch)
            mori_shmem.int32_wait_until_greater_than(
                a_launch_ready + fx.Int64(peer) * fx.Int64(4), launch_epoch - fx.Int32(1)
            )
            comm_ops.fence_system_acquire()
        if tid == fx.Int32(0):
            reset_rsrc = _make_buffer_from_addr(a_reset_counters, fx.Int32)
            for slot in range_constexpr(reset_count):
                _buffer_store(reset_rsrc, fx.Int32(slot * 16), fx.Int32(0), fx.Int32)
        fx.rocdl.s_waitcnt(0)
        fx.barrier()
        if tid == fx.Int32(0):
            comm_ops.fence_agent_release()
            comm_ops.store_i32_system(gate_addr, fx.Int32(0), gate_epoch)
        fx.rocdl.s_waitcnt(0)
        fx.barrier()
    else:
        if tid == fx.Int32(0):
            mori_shmem.int32_wait_until_equals(gate_addr, gate_epoch)
            comm_ops.fence_agent_acquire()
        fx.barrier()
```

`launch_epoch` 由 host 传，取 `self._round + 1`（单调递增即可，`int32_wait_until_greater_than` 只要求单调）。

- [ ] **Step 5: 四个用例全绿**

```bash
cd /root/workspace/aiter
for c in construct bitexact repeat skew; do
  printf "  %-10s " "$c"
  PYTHONPATH=. torchrun --standalone --nproc_per_node=8 \
      op_tests/multigpu_tests/test_tp_gather.py --case $c 2>&1 | tail -1
done
```

Expected：四行 OK。

`repeat` 这一条是关键：它连跑 12 次且 `m_local` 变化，host 的轮次计数与设备端 flag 的对应关系错了就会在第二次或第三次暴露。

- [ ] **Step 6: 负对照 — 轮次计数写错必须失败**

把 `self._round += 1` 临时注释掉（parity 永远是 0，expected 永远是 `npes`），重跑 `repeat`。

Expected：**FAIL**。第一轮会过，第二轮起 `payload_ready[0]` 已经累到 `2*npes` 而 `expected` 仍是 `npes`，`int32_wait_until_equals` 等不到相等值，会**挂死**而不是报错。所以这一步要加 `timeout 300`，超时即视为负对照生效。

```bash
cd /root/workspace/aiter
timeout 300 env PYTHONPATH=. torchrun --standalone --nproc_per_node=8 \
    op_tests/multigpu_tests/test_tp_gather.py --case repeat; echo "exit=$?"
```

Expected：`exit=124`（超时）或断言失败，两者都算负对照生效。改回去。

- [ ] **Step 7: 复测同步开销已消失**

```bash
cd /root/workspace/aiter
PYTHONPATH=. torchrun --standalone --nproc_per_node=8 \
    /root/.claude/jobs/1e723cc2/tmp/bench_parity.py 2>&1 | grep "^m_local="
```

该脚本会因为 `g.epoch_parity` 不存在而报错——把它那两行 `.item()` 测量删掉，只留 `gather total`。

Expected：m_local=128 的 `gather total` 从 0.0849 ms 降到 0.070 ms 附近。把实际数字贴出来；**如果没有明显下降，说明同步没真的去掉，停下来查**。

- [ ] **Step 8: 提交**

```bash
cd /root/workspace/aiter
python -m black --check aiter/ops/flydsl/kernels/mega_moe/tp_gather.py \
    aiter/ops/flydsl/kernels/mega_moe/collective_sched.py
git add aiter/ops/flydsl/kernels/mega_moe/tp_gather.py \
        aiter/ops/flydsl/kernels/mega_moe/collective_sched.py
git commit -m "perf(tp-moe): derive gather parity on the host, not from the device

current_parity() read epoch_parity[0].item(), a GPU->CPU sync on every call.
Measured at 15us regardless of size -- 26% of the gather at m_local=8, 18% at
128, 13% at 256. Parity is deterministic because every rank calls the same
number of times, so the host can just count rounds.

Drops the epoch_parity/epoch_expected device tensors and the parity flip from
the rendezvous. emit_epoch_rendezvous is left in place unused; the new
emit_launch_rendezvous is the flip-free variant.

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 2: 融合 kernel

**Files:**
- Create: `aiter/ops/flydsl/kernels/mega_moe/tp_fused_stage1.py`
- Modify: `op_tests/multigpu_tests/test_tp_moe_stage1.py`（加 `case_fused_numerics`）

### 结构

一个 kernel，grid `num_cu * grid_mult`，ticket 决定角色：

```
全体      取 ticket，定角色
owner     向所有 peer 打 launch_ready -> 等所有 peer -> 清计数器 -> 开 gate
非 owner  等 gate
producer  推 x_q 行 + x_scale 行 -> 本地 push_done 累加 -> 最后一个代表本卡向所有 peer 打 payload_ready
全体      等 payload_ready == expected -> fence_system_acquire
全体      跑 GEMM1 的 tile 循环
```

前四步照抄 `tp_gather.py` 的 kernel 体，最后一步照抄 `tp_gemm1.py` 的 `for flat in range(...)` 循环加 `do_scheduled_tile` 闭包。两边都是已验证过的代码。

### 三个接线要点

**LDS 复用。** `emit_ticket_and_roles` 用 `SharedStorage.pool` 的第 0 字节做广播，GEMM 用同一块 `pool` 做 A 的 ping/pong 和 cshuffle。先用后用互不干扰，MegaMoE 就是这么做的。所以只需要 `tp_gemm1.py` 那个 `SharedStorage`，不用再开一块。

**A 和 scale 从对称 buffer 取。** `TPATileLoader` 的 `x_tensor` 传 `rx_x[parity]`（host 按 parity 切好的视图），`sx_rsrc` 从 `rx_scale[parity]` 建。`total_rows` 是 `tp_size * mtpr + 1`，`pad_row` 是 `total_rows - 1`——那一行在构造时清过零，正是 PAD 行。

**work 分配不用 atomic pool。** `tp_gemm1.py` 用的是 `for flat in range(block_idx.x, total_work, GRID_X)` 静态划分，已经验证过。producer CTA 先干了推送的活，静态划分下它们的 GEMM 份额和别人一样，会略微拖尾。**先这么做，测出来再说**；真成为瓶颈时再换成 `collective_sched` 的分片 atomic pool。方案里不提前优化。

- [ ] **Step 1: 写融合 kernel**

新建 `aiter/ops/flydsl/kernels/mega_moe/tp_fused_stage1.py`。它 `import` 而不是复制：从 `tp_gather.py` 拿推送那段的写法参考，从 `tp_gemm1.py` 拿 GEMM 那段，从 `collective_sched.py` 拿 `copy_row`、`emit_ticket_and_roles`、`emit_launch_rendezvous`，从 `gemm1.py` 拿 `do_tile` 和 `_LdsF32View`，从 `gemm_util.py` 拿其余 loader，从 `tp_gemm_util.py` 拿两个 gather loader。

`compile_tp_fused_stage1` 的编译期参数是 `tp_gather` 那套（`model_dim`、`npes`、`rank`、`producer_blocks`、`num_waves`、`slots`）加 `tp_gemm1` 那套（`inter_dim`、`experts`、`total_rows`、`sort_block_m`、`tile_n`、`tile_k`、`num_cu`、`grid_mult`、`swizzle_a`、`pipe_weights`、`mfma_amajor`、`async_a_copy`、`waves_per_eu_hint`、`swiglu_limit`）。

kernel 运行期参数：两个输出张量、`rx_x[parity]` 与 `rx_scale[parity]` 两个 A 侧张量、`w`、`scale_w`、`tile_row_base`、`expert_ids`、`sorted_token_ids`、`num_valid_ids`、本卡的 `x_q` 与 `x_scale` 地址、四张 p2p 表的地址、`payload_ready` 与 `launch_ready` 的本地地址、`epoch_gate` 与 `entry_count` 与 `reset_counters` 的地址，以及标量 `m_local`、`parity`、`expected`、`launch_epoch`、`gate_epoch_base`、`tokens`、`x_slab_bytes`、`scale_slab_bytes`。

> 参数很多。实现时按「输出 / A 侧 / 权重 / 计划 / 本卡输入 / p2p 表 / 本地 flag / 标量」分组，每组之间空一行并写注释，否则接线时错位很难查。

**`always_valid=True` 与 `trb_rsrc` 的两个陷阱照旧**，见 `tp_gemm1.py` 里的注释，不要重新发明。

- [ ] **Step 2: 写 host 入口**

在 `tp_moe_stage1.py` 里加 `forward_fused`，先不动 `forward`：

```python
    def forward_fused(self, x_bf16, route_weights, topk_ids):
        """Fused entry: local quant, one metadata collective, sort, then one kernel."""
        m_local = self._validate_call(x_bf16, route_weights, topk_ids, torch.bfloat16)
        m_global = self.m_logical_for(m_local)
        x_q, x_scale = self.quantize(x_bf16)
        # topk_ids and route_weights have the same shape; ship them as one
        # int32 buffer so the metadata costs one collective instead of two.
        meta = torch.empty((m_local, 2 * self.topk), dtype=torch.int32, device=self.device)
        meta[:, : self.topk] = topk_ids
        meta[:, self.topk :] = route_weights.view(torch.int32)
        meta_g = self._all_gather_one(meta)
        ids_g = meta_g[:, : self.topk].contiguous()
        wts_g = meta_g[:, self.topk :].contiguous().view(torch.float32)
        sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids = self._sort(ids_g, wts_g)
        payload, scale = self._run_fused(
            x_q, x_scale, sorted_ids, sorted_expert_ids, num_valid_ids, m_local)
        return self._pack(payload, scale, sorted_ids, sorted_weights,
                          sorted_expert_ids, num_valid_ids, m_global)
```

`_run_fused` 负责懒构造 `TPActivationGather`（第一次调用时按 `m_local` 推出 `max_tok_per_rank`，或由构造参数给定）、算 parity/expected/launch_epoch、分配输出、启动融合 kernel。

**`max_tok_per_rank` 必须是构造参数**，不能按第一次调用的 `m_local` 推——对称内存是集合分配，八张卡必须一致，而第一次调用的 `m_local` 虽然各卡相同，但把分配时机推迟到 `forward` 里会让构造失败变成运行时失败。加进 `__init__` 签名，`fused` 路径必填。

- [ ] **Step 3: 写数值对拍**

在 `test_tp_moe_stage1.py` 加 `case_fused_numerics`：同一进程构造 `TPMoEStage1`（走 `forward_fused`）与 `TPMoEStage1NCCLRef`（走 `forward`），喂同一份输入。

**比对标准分两级：**

- `sorted_token_ids` / `sorted_expert_ids` / `sorted_weights` / `num_valid_ids` / `m_logical` / `max_sorted`：**逐位相同**。这些只由 `moe_sorting` 决定，两条路径喂给它的输入必须完全一样。
- `inter_sorted_quant` 与 `inter_sorted_shuffled_scale`：**设容差**，因为两条路径用的是不同的 GEMM1 实现（`mixed_moe_gemm_2stage_common` 对 `gemm1.do_tile`），累加顺序不同。掩掉 PAD 行之后比 rel_l2，阈值 `< 0.01`。

掩 PAD 行的写法照抄同文件里 `case_ref_fidelity` 的 `keep = (sorted_token_ids & 0x00FFFFFF) < m_logical`——**stage1 从不写 PAD 行，那些字节是 `torch.empty` 的残留内存**，不掩掉就是在比垃圾。

- [ ] **Step 4: 负对照**

把 `forward_fused` 里 `meta[:, self.topk:] = route_weights.view(torch.int32)` 临时改成 `= topk_ids`（权重被路由表覆盖），重跑。

Expected：**FAIL**，`sorted_weights` 逐位比对报错。

改回去。再把 `_run_fused` 传给 kernel 的 `parity` 临时固定成 0，跑 `case_fused_numerics` 两遍以上（用例内部要循环多个 `m_local`，所以一次运行就会跨轮）。

Expected：**FAIL 或挂死**，因为双缓冲失效后 peer 的读写会撞上。挂死同样算生效，用 `timeout 300` 包住。

- [ ] **Step 5: 跑通**

```bash
cd /root/workspace/aiter
PYTHONPATH=. torchrun --standalone --nproc_per_node=8 \
    op_tests/multigpu_tests/test_tp_moe_stage1.py --case fused_numerics
```

Expected：每个 `m_local` 一行，最后 `case_fused_numerics OK`。

- [ ] **Step 6: 提交**

```bash
cd /root/workspace/aiter
python -m black --check aiter/ops/flydsl/kernels/mega_moe/tp_fused_stage1.py \
    aiter/ops/flydsl/kernels/mega_moe/tp_moe_stage1.py \
    op_tests/multigpu_tests/test_tp_moe_stage1.py
git add aiter/ops/flydsl/kernels/mega_moe/tp_fused_stage1.py \
        aiter/ops/flydsl/kernels/mega_moe/tp_moe_stage1.py \
        op_tests/multigpu_tests/test_tp_moe_stage1.py
git commit -m "feat(tp-moe): fuse the P2P push and GEMM1 into one kernel launch

Both halves were verified in isolation first: the push against NCCL bit for
bit, the token-indexed loaders against the contiguous ones bit for bit. This
wires them into a single grid -- owner flips the epoch, producers push, every
CTA waits once for all sources, then all of them run the GEMM tile loop.

The wait is once per CTA rather than MegaMoE's once per tile: under TP every
tile's 32 rows are scattered across all source ranks, so no tile can start
early and per-tile readiness would only add protocol overhead.

forward_fused is added alongside forward; the switchover is a separate commit.

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 3: 切换入口，删掉 NCCL 路径

**Files:**
- Modify: `aiter/ops/flydsl/kernels/mega_moe/tp_moe_stage1.py`
- Modify: `op_tests/multigpu_tests/test_tp_moe_stage1.py`

- [ ] **Step 1: 阶段一用例改跑参照实现**

`test_tp_moe_stage1.py` 里 `case_all_gather`、`case_forward_contract`、`case_numerics`、`case_prequant_equivalence`、`case_end_to_end` 构造的 `TPMoEStage1(...)` 全部改成 `TPMoEStage1NCCLRef(...)`，并在文件顶部 import 它。

理由：这五个用例测的是阶段一那条 NCCL 流水线的行为，那条行为现在由参照实现承载。融合路径由 `case_fused_numerics` 和 `case_ref_fidelity` 覆盖。

`case_construct_validates`、`case_capacity`、`case_exports` 保持测 `TPMoEStage1`，因为它们测的是构造与导出。**注意 `case_exports` 里那三段 `transport` 校验要删掉**，因为参数没了。

- [ ] **Step 2: `forward` 切到融合路径**

把 `forward` 的函数体整个替换成 Task 2 Step 2 里 `forward_fused` 的内容，然后删掉 `forward_fused` 这个名字。保留末尾的 `__call__ = forward` 和 `forward_bf16 = forward`，这样对外的 API 表面一个字不变。

不要写成 `forward = forward_fused` 的别名形式：那样类里会同时存在两个可调用名字指向同一实现，读代码的人得先确认它们真的相同才敢改其中一个。

- [ ] **Step 3: 删掉不再用到的东西**

从 `tp_moe_stage1.py` 删除这些：

- `_all_gather_inputs`（融合路径不再收三个张量）
- `forward_prequant`
- `_TRANSPORT_NCCL`、`_TRANSPORT_FUSED`、`_TRANSPORTS` 三个常量
- 构造函数的 `transport` 参数、它的两处校验、以及 `self.transport`

**`_all_gather_one` 要保留**，融合路径用它收那一个打包好的 metadata buffer。

`moe_mxfp4_sort` 和 `fused_dynamic_mxfp8_quant_moe_sort` 两个 import 随 `forward_prequant` 和旧 `forward` 一起删掉。删完跑一次 `python -c "import ..."` 确认没有残留引用。

- [ ] **Step 4: 全量回归**

```bash
cd /root/workspace/aiter
for c in construct capacity exports; do
  printf "  %-18s " "$c"
  PYTHONPATH=. python op_tests/multigpu_tests/test_tp_moe_stage1.py --case $c 2>&1 | tail -1
done
for c in all_gather forward_contract numerics prequant e2e ref_fidelity fused_numerics; do
  printf "  %-18s " "$c"
  PYTHONPATH=. torchrun --standalone --nproc_per_node=8 \
      op_tests/multigpu_tests/test_tp_moe_stage1.py --case $c 2>&1 | tail -1
done
for c in construct bitexact repeat skew; do
  printf "  gather %-11s " "$c"
  PYTHONPATH=. torchrun --standalone --nproc_per_node=8 \
      op_tests/multigpu_tests/test_tp_gather.py --case $c 2>&1 | tail -1
done
printf "  gemm1 equiv        "
PYTHONPATH=. python op_tests/multigpu_tests/test_tp_gemm1.py --case equiv 2>&1 | tail -1
```

Expected：十五行全部以 `OK` 结尾。

- [ ] **Step 5: 确认 MegaMoE 仍未被碰**

```bash
cd /root/workspace/aiter
git diff --stat main...HEAD -- aiter/ops/flydsl/kernels/mega_moe/mega_moe_v2.py \
    aiter/ops/flydsl/kernels/mega_moe/mega_moe_stage1.py \
    aiter/ops/flydsl/kernels/mega_moe/dispatch.py \
    aiter/ops/flydsl/kernels/mega_moe/gemm1.py \
    aiter/ops/flydsl/kernels/mega_moe/gemm_util.py
```

Expected：无输出。

- [ ] **Step 6: 提交**

```bash
cd /root/workspace/aiter
git add -A aiter/ops/flydsl/kernels/mega_moe/tp_moe_stage1.py \
        op_tests/multigpu_tests/test_tp_moe_stage1.py
git commit -m "feat(tp-moe): switch TPMoEStage1 to the fused path, drop the NCCL one

forward now runs local quant, one packed metadata collective, moe_sorting, and
one fused kernel. The transport parameter goes away with the branch it
selected, and the phase-1 cases retarget to TPMoEStage1NCCLRef, which is what
now carries the old behaviour.

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 4: 性能验收

**Files:**
- Modify: `op_tests/multigpu_tests/bench_tp_moe_stage1.py`

- [ ] **Step 1: 加融合模式**

现有的 bench 分段测 `_all_gather_inputs` / `_sort` / 量化 / `_run_gemm1`。改成同时测两条路径：

- **参照**：`TPMoEStage1NCCLRef.forward`，沿用现有的分段计时。
- **融合**：`TPMoEStage1.forward`，分段成「本地量化 / metadata collective / moe_sorting / 融合 kernel」四段。

两条都测 m_local 1、8、64、128、256。基线只测到 128，256 那档要新测。

计时方式沿用现有写法：每次迭代前 `dist.barrier()`，CUDA event 计时，中位数，跨 rank `all_reduce(MAX)`。

- [ ] **Step 2: 跑**

```bash
cd /root/workspace/aiter
PYTHONPATH=. torchrun --standalone --nproc_per_node=8 \
    op_tests/multigpu_tests/bench_tp_moe_stage1.py --m-local 1,8,64,128,256
```

- [ ] **Step 3: 对照验收线**

设计文档第 8 节定的线是：**m_local=128 时总时间从 0.5365 ms 降到 0.48 ms 以内，即至少快 10%。**

把实测表格贴出来，逐档给出加速比，并明确写清是否达线。

**达不到也照实写，不要调参数凑数。** 达不到说明融合 kernel 自身的调度开销吃掉了收益，那是一个需要单独查的结论，不是一个要绕过的障碍。真达不到的话，第一个该查的是 producer CTA 在静态 work 划分下的拖尾——见 Task 2 的「work 分配不用 atomic pool」一节。

- [ ] **Step 4: 提交**

```bash
cd /root/workspace/aiter
python -m black --check op_tests/multigpu_tests/bench_tp_moe_stage1.py
git add op_tests/multigpu_tests/bench_tp_moe_stage1.py
git commit -m "bench(tp-moe): measure the fused path against the NCCL reference

Co-Authored-By: Claude <noreply@anthropic.com>"
```

- [ ] **Step 5: 更新设计文档的估算表**

设计文档第 3 节那张估算表是推算的，标注了「非实测」。用实测数字替换，并保留原推算值作对照，说明推算错在哪、错多少。

---

## Self-Review

**Spec 覆盖：** 第 3 节的流水线划分由 Task 2 Step 2 实现（本地量化、一次打包的 metadata collective、`moe_sorting`、融合 kernel）；5.1 的角色划分与 5.2 的执行顺序由 Task 2 Step 1 实现，「等待只做一次」写进了提交信息；第 7 节的融合数值用例由 Task 2 Step 3 实现；第 8 节的验收线由 Task 4 Step 3 执行。

**偏离 spec 之处：** spec 5.3 说双缓冲加 `launch_ready` 握手，Task 1 把 parity 的推导从设备挪到 host，握手保留、双缓冲保留，只是不再在设备上翻 parity。这是实测驱动的改动，15 µs 的同步开销占 gather 的 13% 到 26%。spec 5.4 说每个 producer block 发布一次 ready flag，实际实现（方案二已定）是每个源 rank 发布一次，理由写在方案二里。

**有意不覆盖：** spec 5.6 提到的 `collective_sched.emit_work_pool_loop` 本方案不用，因为 `tp_gemm1.py` 的静态 work 划分已经验证可行，没有证据表明需要 atomic pool。真测出拖尾再加。

**类型一致性：** `_expected_for` 在 `slots == 1` 与 `slots == 2` 下是两个不同的公式，Task 1 Step 2 明确要求两条都写。`forward_fused` 的 `meta` 打包依赖 `route_weights` 是 fp32 且与 `topk_ids` 同形状，这两条 `_validate_call` 已经校验过。

**已知风险：** Task 2 的 kernel 参数超过二十个，接线错位是最可能的失败模式，方案里要求按语义分组并加注释。Task 1 Step 6 的负对照会挂死而不是报错，方案里已写明用 `timeout` 判定。
