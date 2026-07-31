# Dispatch→GEMM1 融合：pull-分组 dispatch + 跨 CTA 全局 ring 重叠 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在已落地的 fp8-transport 基座（T-A，-20%）之上，用**跨 CTA 全局内存 ring** 把 "pull-分组 dispatch" 与 GEMM1 的计算重叠起来——生产者 CTA 把（按本地 expert 分组的）fp8 token tile 填进 HBM ring 槽并置就绪位，GEMM1 消费者 CTA 自旋等就绪、消费该槽、再置释放位——从而消除独立 a1 gather kernel + grouped-a1 一整趟 HBM 往返，并让 dispatch 与 GEMM1 跨核重叠，而**不依赖任何 CTA 内 warp 专精 / flydsl 命名屏障**。

**Architecture:** 明确放弃 CTA 内（workgroup 内）producer/consumer warp 专精——它需要 flydsl 侧从未验证过的命名屏障（`s_barrier_init/join`，仅在 opus C++ GEMM 验证过，flydsl 全路径只用过 `-1` 整-WG 屏障，成熟度未知）。改用**跨 CTA 粒度**的重叠：ring 在全局内存（HBM / 对称 arena），跨 CTA 同步只用 `flydsl_prims.py` 的系统作用域原子 + release/acquire fence + volatile 自旋——**这套原语正是现网 dispatch 的骨架，已验证**。重叠粒度比 DeepGEMM 的片上 L1 ring 粗（一个 CTA 一次吃一个 tile 槽），但零新原语风险，且可回退。

**Tech Stack:** Python 3, PyTorch, aiter FlyDSL（`flyc.jit`/`flyc.kernel`），cco 对称内存（`mori.cco.device.flydsl.Window.lsa_ptr`），`flydsl_prims`（系统原子/release-acquire fence/volatile 自旋），gfx1250 WMMA a8w4（fp8 激活 + MXFP4 权重）。

## Global Constraints

- 目标硬件仅 gfx1250；其它硬件走原路径不受影响。
- **同步原语铁律**：跨 CTA 协调只用 `aiter/ops/flydsl/dispatch_combine_v2/flydsl_prims.py` 的 `atomic_add_global` / `store_i32_system`(release) / `store_i64_system` / `fence_system_acquire` / `fence_system_release` / `load_i32_acquire` / `load_i64_acquire` / `spin_until_eq_i32/i64` / `spin_until_gt_i32`。**禁止**引入 flydsl 命名屏障（`s_barrier_init` / `s_barrier_join` / 非 `-1` 的 `s_barrier_signal/wait`）——本 plan 全程不做 CTA 内 warp 专精。
- 新 env 门控 `AITER_EP_CCTA_RING`（默认关）；`=0` 必须与当前主线逐 kernel / 逐字节一致。与 `AITER_EP_FP8_TRANSPORT`（基座，需先开）组合。
- 只作用于 a8w4（`data_format=="a8w4"`，`quant_mode=="fp8"`）。
- **就绪位/释放位语义**：release 存 + acquire 读，`load_i32_acquire` 为 volatile（防 LICM 把自旋读提出循环导致读到 stale）。每槽用**单调 generation** 计数（非 0/1 flag），规避 ring 复用的 ABA。
- CUDA graph 兼容：固定 grid + 固定 kernel 序列，无 host 端动态分配；ring 的 ready/freed 计数每次 launch 前由一发轻量 memset kernel 清零（进 graph）。
- 死锁安全：ring 深度 `S >= 消费者 CTA 并发数`，且生产者对某槽的复写必须等该槽上一占用者的 freed generation 到位（见 Task 0 模型）。
- 测试入口：`op_tests/multigpu_tests/test_mega_moe.py`，**非 `/app` 目录**运行（避免 `/app/triton` 遮蔽）；判据 `--acc_verify 1` 打印 `MEGA-CHECK PASS`。单 GPU 门禁在 `op_tests/flydsl_tests/`。

---

## 已落地基座（勿重做，作为前提）

见旧 plan `docs/superpowers/plans/2026-07-29-dispatch-gemm1-fusion.md` 与 spec `docs/superpowers/specs/2026-07-29-dispatch-gemm1-fusion-design.md`：

- **T-A（DONE，-20% 整层）**：dispatch 只传 fp8+e8m0（带宽减半），上游量化融进 rmsnorm（`fused_rms_mxfp8_quant`，30.5us）。`AITER_EP_FP8_TRANSPORT=1`。基座正确（2/4-rank `MEGA-CHECK PASS`，logits 逐位同基线）且净收益为正。
- **T-B.3/T-B.4（DONE，parity 绿）**：GEMM1 A-load 侧 `tensor_load_gather` + 真 rowmap（`_build_a_gather_rowmap`）已能从 fp8 `recv_x` 按 route 直接 gather A 行，单 GPU 逐字节 parity 通过。**但**该路径为求每-wave 均匀 gather 计数**强制关掉了 wave-spec**，加上 a1_payload 仍被物化，小配置实测 **+27% 回归**（88.8→112.8us）。本 plan 的跨 CTA ring 正是替代"CTA 内重建流水线"来回收这部分损失的路线。

**复用产物**：`tensor_load_gather` / `make_tensor_gather_descriptor`（`tdm_gather_shim.py`），`launch_gemm_a8w4_tdm` 的 `ep_a_gather`/`arg_a_rowmap`/`i32_num_recv_rows` 形参（T-B.3），`_build_a_gather_rowmap`（`grouped_moe_gfx1250.py`，T-B.4），`flydsl_prims.py` 全套原语。

---

## Task 0：跨 CTA 全局 ring 正确性 SPIKE（前置门 / 唯一 gating）

**目的**：在**动任何 GEMM/dispatch 代码之前**，用一个最小 flydsl 微核证明"多生产者 CTA + 多消费者 CTA 通过全局内存 ring（release/acquire + 自旋 + 单调 generation + 环回复用）在 gfx1250 上正确且不死锁"。这是整份 plan 能否成立的唯一前提；红则回退（见 Task 4 fallback），绿则 Task 1+ 只是把 payload 从"标记向量"换成"grouped fp8 tile"。**不碰对称内存**（单 GPU、纯本地 HBM，隔离同步风险）。

**Files:**
- Create: `aiter/aiter/ops/flydsl/kernels/crosscta_ring_spike_gfx1250.py`（自包含 `@flyc.jit` 微核 + host launcher）
- Test: `aiter/op_tests/flydsl_tests/test_crosscta_ring_spike.py`

**Interfaces:**
- Produces: `run_crosscta_ring_spike(G:int, S:int, P_BLK:int, C_BLK:int) -> torch.Tensor`
  - 起 `P_BLK` 个生产者 CTA + `C_BLK` 个消费者 CTA（`grid=(P_BLK+C_BLK,1,1)`，`bid<P_BLK` 为生产者），共处理 `G` 个任务、ring 深度 `S`。
  - 返回 `consumed[G]`(int32)：`consumed[i]` = 消费者读回的 task i 的 payload（生产者写入的值 = `i*7+13`）。生产者/消费者各用全局原子 claim 计数器领任务号。
  - ring/计数全在 HBM，函数内部分配并在 launch 前清零。

**同步模型（每槽单调 generation，MPMC claim）：**
- 全局计数器 `prod_next`(i32)、`cons_next`(i32) 初始 0；`ready[S]`、`freed[S]` 初始 0。
- 生产者 CTA 循环：`t = atomic_add_global(&prod_next, 1)`；`if t >= G: break`；`slot=t%S`；`gen=t//S + 1`；
  - `spin_until_eq_i32(&freed[slot], gen-1)`（等该槽上一占用者被消费；gen==1 时等 0，立即真）；
  - 写 `ring[slot] = t*7+13`；`fence_system_release()`；`store_i32_system(&ready[slot], 0, gen)`。
- 消费者 CTA 循环：`t = atomic_add_global(&cons_next, 1)`；`if t >= G: break`；`slot=t%S`；`gen=t//S + 1`；
  - `spin_until_eq_i32(&ready[slot], gen)`；`fence_system_acquire()`；
  - 读 `v = ring[slot]`；`consumed[t] = v`；`fence_system_release()`；`store_i32_system(&freed[slot], 0, gen)`。
- 约束 `S >= C_BLK`（否则消费者并发数 > 槽数，某槽 gen 未消费就被复写 → 死锁/错值）。

- [ ] **Step 1: 写失败测试**

新建 `aiter/op_tests/flydsl_tests/test_crosscta_ring_spike.py`：

```python
import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs gfx1250")


@pytest.mark.parametrize("G,S,P_BLK,C_BLK", [
    (1, 1, 1, 1),        # 退化：单任务单槽
    (64, 4, 1, 1),       # 单 P 单 C，环回 16 圈
    (256, 8, 2, 4),      # MPMC，S==2*C_BLK，环回
    (300, 8, 4, 8),      # S==C_BLK 边界（最紧不死锁）
])
def test_crosscta_ring_delivers_each_task_once_byte_exact(G, S, P_BLK, C_BLK):
    from aiter.ops.flydsl.kernels.crosscta_ring_spike_gfx1250 import (
        run_crosscta_ring_spike,
    )
    consumed = run_crosscta_ring_spike(G=G, S=S, P_BLK=P_BLK, C_BLK=C_BLK)
    assert consumed.shape[0] == G
    expected = torch.arange(G, device=consumed.device, dtype=torch.int32) * 7 + 13
    # 每个 task 恰好被消费一次且 payload 逐字节正确（乱序 claim 后按 task 号归位）
    torch.testing.assert_close(consumed.cpu(), expected.cpu(), atol=0, rtol=0)
```

- [ ] **Step 2: 运行确认失败**

Run: `cd /tmp && python -m pytest /app/aiter/op_tests/flydsl_tests/test_crosscta_ring_spike.py -v`
Expected: FAIL —— `ModuleNotFoundError` / `ImportError: cannot import name 'run_crosscta_ring_spike'`。

- [ ] **Step 3: 实现 spike 微核 + launcher**

新建 `aiter/aiter/ops/flydsl/kernels/crosscta_ring_spike_gfx1250.py`。**脚手架照抄** `dispatch_combine_v2/intranode_kernels.py` 的 `make_dispatch`：`@flyc.kernel(known_block_size=[WAVE,1,1])` 装饰 device body（`bid`=block id、`tid`=lane），`@flyc.jit def run(...): <kernel>(...).launch(grid=(P_BLK+C_BLK,1,1), ...)` 做 host launcher。原语从 `dispatch_combine_v2 import flydsl_prims as P` 引入。要点：

- ring/计数用裸 HBM 指针（`torch.zeros(..., int32, device=cuda)` 的 `.data_ptr()` 转 i64 传入），不走对称 arena。
- **每个任务只让 lane 0 干活**（`if tid == 0:`），payload 是标量 i32——spike 只验同步，不验数据搬运宽度。sink 到 `consumed` 用裸 global store（`store_i32_system(consumed_ptr, t, v)` 即可，或普通 store + 结束前 grid 无关，因每个 t 唯一）。
- 生产者/消费者用 `bid < P_BLK` 分流（`const_expr(P_BLK)` 已知 → 静态分支）。claim 循环用 `scf.WhileOp` 或 flydsl `while` 包 `atomic_add_global`；退出条件 `t >= G`。
- generation `gen = t//S + 1`、`slot = t%S` 用整数算术（`t` 是 runtime i32，`S` const → `arith` 运算）。
- 就绪/释放位：严格按上面"同步模型"的 release-store / acquire-spin / fence 顺序，**不得省略 `fence_system_release()`（写 payload 后、置 ready 前）与 `fence_system_acquire()`（等到 ready 后、读 payload 前）**——这是跨 CTA 可见性的关键。

- [ ] **Step 4: 运行确认通过（并观察是否 hang）**

Run: `cd /tmp && timeout 300 python -m pytest /app/aiter/op_tests/flydsl_tests/test_crosscta_ring_spike.py -v`
Expected: 4 个用例全 PASS。**若 hang（timeout 杀）= 死锁**，按序排查：(a) `S < C_BLK`？(b) generation 起始/环回算错（`freed` 初值 vs `gen-1`）？(c) 漏 fence？(d) `load_i32_acquire` 未 volatile 导致自旋读被 CSE。修正后重跑。

- [ ] **Step 5: 记录结论 + 提交**

把 spike 结论（S/C 边界、是否需 fence、最小可用配置）写进本 plan Task 0 末尾一行 `> SPIKE VERDICT: ...`。

```bash
cd /app/aiter
git add aiter/ops/flydsl/kernels/crosscta_ring_spike_gfx1250.py op_tests/flydsl_tests/test_crosscta_ring_spike.py docs/superpowers/plans/2026-07-31-crossCTA-ring-dispatch-gemm1.md
git commit -m "spike(ep): cross-CTA global ring correctness on gfx1250 (flydsl_prims, no named barriers)"
```

> SPIKE VERDICT（2026-07-31，单 gfx1250）：**绿**。`crosscta_ring_spike_gfx1250.py` + `test_crosscta_ring_spike.py`，6 组用例（含退化 1/1/1/1、1P1C 16 圈、MPMC S=2·C、`S==C_BLK` 边界、8192 任务/682 圈深环回）连跑 3× 全 PASS、byte-exact、无死锁（~4.5s/轮）。**结论**：跨 CTA 全局 ring 用 `flydsl_prims`（`store_i32_system` release / `spin_until_eq_i32` + `load_i32_acquire` volatile / `fence_system_acquire`+`fence_system_release` / `atomic_add_global`）在 flydsl+gfx1250 上正确可用，**零命名屏障、零 CTA 内 warp 专精**。关键实现要点已验证：(a) kernel body 用 runtime-start grid-stride `for range(bid, G, const)` 做确定性 MPMC 任务划分（避开未验证的动态 `while`/atomic-claim）；(b) 每槽单调 generation（`ready==gen` / `freed==gen-1`）规避环回 ABA；(c) 写 payload 后 `fence_release`→release-store ready、等 ready 后 `fence_acquire`→读 payload 的顺序不可省；(d) `S>=C_BLK` 且 grid 与 CU 共驻（本 spike 总块数 ≤18）。Task 1+ 只需把 payload 从标量 i32 换成 grouped fp8 tile。

---

## Task 1：ring arena region + generation 计数 + env 门控 + host 清零

**目的**：把 spike 验证过的 ring 落成生产可用的 arena 布局：grouped fp8 tile ring（payload）+ 对应 e8m0 scale ring + `ready[S]`/`freed[S]`/`prod_next`/`cons_next` 计数区，默认关时零改动。前置：Task 0 绿。

**Files:**
- Modify: `aiter/aiter/ops/flydsl/dispatch_combine_v2/dispatch_combine_op.py`（`EpDispatchCombineConfig` 属性区 + `EpDispatchCombineOp.__init__` regions；仿 `is_fused` / 旧 `_fused_q_regions` 风格）
- Test: `aiter/op_tests/flydsl_tests/test_ccta_ring_regions.py`（新建，纯函数，无 GPU）

**Interfaces:**
- Produces:
  - `EpDispatchCombineConfig.ccta_ring: bool`（env `AITER_EP_CCTA_RING`）
  - `EpDispatchCombineConfig.ring_depth: int`（默认 8，env `AITER_EP_CCTA_RING_DEPTH` 覆盖；Task 0 verdict 决定下限）
  - 模块级纯函数 `_ccta_ring_regions(cfg, tile_m:int, k_bytes:int, scale_bytes:int) -> list[(name,nbytes)]`（关时 `[]`）：
    - `"ring_a"`: `S * tile_m * k_bytes`（fp8 payload 槽）
    - `"ring_as"`: `S * tile_m * scale_bytes`（e8m0 scale 槽）
    - `"ring_ready"`: `S * 4`，`"ring_freed"`: `S * 4`，`"ring_prod"`: `4`，`"ring_cons"`: `4`
  - `EpDispatchCombineOp.ccta_ring_ptrs() -> dict[str,int]`（各 region 的 i64 device 指针）
  - `EpDispatchCombineOp.reset_ccta_ring()`：`ring_ready/freed/prod/cons` 清零（`torch` memset，供 launch 前调用；后续可换轻量 kernel 进 graph）

- [ ] **Step 1: 写失败测试**

```python
import torch
from aiter.ops.flydsl.dispatch_combine_v2.dispatch_combine_op import (
    EpDispatchCombineConfig, _ccta_ring_regions,
)

def _cfg(**ov):
    base = dict(rank=0, world_size=2, hidden_dim=512, max_num_inp_token_per_rank=128,
                num_experts_per_rank=4, num_experts_per_token=2, data_type=torch.float8_e4m3fn)
    base.update(ov); return EpDispatchCombineConfig(**base)

def test_ring_flag_defaults_off(monkeypatch):
    monkeypatch.delenv("AITER_EP_CCTA_RING", raising=False)
    assert _cfg().ccta_ring is False
    assert _ccta_ring_regions(_cfg(), tile_m=64, k_bytes=512, scale_bytes=16) == []

def test_ring_regions_present_when_on(monkeypatch):
    monkeypatch.setenv("AITER_EP_CCTA_RING", "1")
    cfg = _cfg(); S = cfg.ring_depth
    regs = dict(_ccta_ring_regions(cfg, tile_m=64, k_bytes=512, scale_bytes=16))
    assert regs["ring_a"] == S * 64 * 512
    assert regs["ring_as"] == S * 64 * 16
    assert regs["ring_ready"] == S * 4 and regs["ring_freed"] == S * 4
    assert regs["ring_prod"] == 4 and regs["ring_cons"] == 4
```

- [ ] **Step 2: 运行确认失败**

Run: `cd /tmp && python -m pytest /app/aiter/op_tests/flydsl_tests/test_ccta_ring_regions.py -v`
Expected: FAIL —— `cannot import name '_ccta_ring_regions'`。

- [ ] **Step 3: 实现 config 属性 + region 纯函数 + ptrs/reset**

`EpDispatchCombineConfig` 属性区加：

```python
    @property
    def ccta_ring(self) -> bool:
        return os.environ.get("AITER_EP_CCTA_RING", "0") in ("1", "true", "True", "yes", "on")

    @property
    def ring_depth(self) -> int:
        return int(os.environ.get("AITER_EP_CCTA_RING_DEPTH", "8"))
```

模块级加 `_ccta_ring_regions`（关时返回 `[]`，开时返回上面 6 个 region）。`__init__` 的 regions 列表构造后追加 `regions += _ccta_ring_regions(cfg, tile_m, k_bytes, scale_bytes)`（`tile_m`/`k_bytes`/`scale_bytes` 从 gemm1 a8w4 tile 配置取，见 `mxfp4_preshuffle_gfx1250_tdm.py` 的 `tile_m` / `A_ROW_B` / scale 行字节）。加 `ccta_ring_ptrs()`（`self.arena.local_ptr(name)` 转 i64）与 `reset_ccta_ring()`（对 ready/freed/prod/cons 四块 `.zero_()`）。

- [ ] **Step 4: 运行确认通过 + 提交**

Run: `cd /tmp && python -m pytest /app/aiter/op_tests/flydsl_tests/test_ccta_ring_regions.py -v`
Expected: PASS

```bash
cd /app/aiter && git add aiter/ops/flydsl/dispatch_combine_v2/dispatch_combine_op.py op_tests/flydsl_tests/test_ccta_ring_regions.py
git commit -m "feat(ep): cross-CTA ring arena regions + gate + host reset (default off)"
```

---

## Task 2：生产者——grouped fp8 tile 填 ring 槽 + 置就绪位

**目的**：把 spike 的"标记 payload"换成真 payload——生产者 CTA 按 `_build_a_gather_rowmap` 的 rowmap，从 fp8 `recv_x`（T-A 基座产出）gather 出一个 `[tile_m, K]` 的 grouped fp8 tile（+ 对应 e8m0 scale，preshuffle），写进 `ring_a`/`ring_as` 槽，`fence_system_release()` 后 `store_i32_system(ring_ready[slot], gen)`。**Phase-2a 源用 LOCAL `recv_x`**（push dispatch 照常先跑），隔离 ring 风险；跨 rank pull 留 Task 2b。

**Files:**
- Create: `aiter/aiter/ops/flydsl/kernels/ccta_ring_producer_gfx1250.py`（`@flyc.jit` 生产者核 + launcher）
- Modify: `aiter/aiter/ops/flydsl/grouped_moe_gfx1250.py`（`_grouped_a8w4_tdm_moe` 的 `_use_disp_q` 分支，`ccta_ring` 开时改起生产者核而非物化 a1_payload）
- Test: `aiter/op_tests/flydsl_tests/test_ccta_ring_producer.py`

**Interfaces:**
- Consumes: `recv_x`(fp8 `[num_recv_rows, K]`)、`out_scales`(e8m0)、`a_rowmap`（Task-B.4 `_build_a_gather_rowmap`）、`num_valid_tiles = ceil(contiguous_m/tile_m)`、Task 1 的 `ccta_ring_ptrs()`、`ring_depth`。
- Produces: `launch_ccta_ring_producer(recv_x, out_scales, a_rowmap, ring_ptrs, S, tile_m, ...)`：每 tile 号 `t` 由生产者 CTA 经 `atomic_add_global(ring_prod,1)` 领取，gather + preshuffle 写 `ring_a[slot]`/`ring_as[slot]`，release + `ready[slot]=gen`。

- [ ] **Step 1: 写失败测试（生产者单核 parity）**

单 GPU：起生产者核（消费者用**一发普通 kernel/torch** 直接把每个 ready 槽读出比对），断言 ring 槽内容 == 用 T-B.4 已验证路径物化的 grouped tile 逐字节：

```python
def test_producer_ring_tiles_match_materialized(monkeypatch):
    monkeypatch.setenv("AITER_EP_CCTA_RING", "1")
    # 构 recv_x/out_scales/a_rowmap（复用 test_tb_a_gather_parity 的构造）
    # ref = flydsl_moe_fused_quant_preshuffle(...) 物化 grouped a1（T-B.4 基准）
    # launch_ccta_ring_producer(...) 后，按 gen 顺序读 ring_a 槽，拼回 [contiguous_m, K]
    # torch.testing.assert_close(ring_tiles, ref_a1_payload, atol=0, rtol=0)
    ...
```

- [ ] **Step 2: 运行确认失败**

Run: `cd /tmp && python -m pytest /app/aiter/op_tests/flydsl_tests/test_ccta_ring_producer.py -v`
Expected: FAIL —— `launch_ccta_ring_producer` 不存在。

- [ ] **Step 3: 实现生产者核**

`ccta_ring_producer_gfx1250.py`：脚手架仿 `make_dispatch`。device body（每 CTA）：`while` 领 `t = atomic_add_global(ring_prod,1)`；`t >= num_valid_tiles` 退出；`slot=t%S`、`gen=t//S+1`；`spin_until_eq_i32(&freed[slot], gen-1)`；用 `tensor_load_gather`（复用 T-B.3 的 `make_tensor_gather_descriptor`，源=`recv_x`，行号=`a_rowmap[t*tile_m + r]`）把 `[tile_m,K]` fp8 读进 LDS/VGPR 并写 `ring_a[slot]`；scale 同法写 `ring_as[slot]`（preshuffle 索引复刻 `_grouped_a8w4_preshuffle_e8m0_scale`）；`fence_system_release()`；`store_i32_system(&ready[slot],0,gen)`。

- [ ] **Step 4: 运行确认通过**

Run: 同 Step 2。Expected: PASS（ring 槽逐字节等于物化基准）。

- [ ] **Step 5: `_grouped_a8w4_tdm_moe` 接生产者（开关下）+ 提交**

`grouped_moe_gfx1250.py` 的 `_use_disp_q` 分支：`if cfg.ccta_ring:` 调 `op.reset_ccta_ring()` + `launch_ccta_ring_producer(...)`（不再 `flydsl_moe_fused_quant_preshuffle` 物化 a1_payload），把 `ring_ptrs`/`S`/`tile_m` 透传给 Task 3 的 GEMM1 消费端；`else:` 走现路径。

```bash
cd /app/aiter && git add aiter/ops/flydsl/kernels/ccta_ring_producer_gfx1250.py aiter/ops/flydsl/grouped_moe_gfx1250.py op_tests/flydsl_tests/test_ccta_ring_producer.py
git commit -m "feat(ep): cross-CTA ring producer (pull-grouped fp8 tiles into ring, release ready)"
```

---

## Task 3：消费者——GEMM1 自旋消费 ring 槽 + 置释放位（跨 CTA 重叠）

**目的**：GEMM1 消费者 CTA 领 tile 号 → `spin_until_eq_i32(ready[slot], gen)` + `fence_system_acquire()` → 从 `ring_a`/`ring_as` 槽读 A/scale 直接进 A-load LDS（复用 T-B.3 gather 落 LDS 的下游）→ 算 WMMA → A-load 完成即 `store_i32_system(freed[slot], gen)` 放槽给生产者复用。**不做 CTA 内 warp 专精**——重叠来自"生产者 CTA 与消费者 CTA 并发跑在不同 CU"。前置：Task 2 绿。

**Files:**
- Modify: `aiter/aiter/ops/flydsl/kernels/mxfp4_preshuffle_gfx1250_tdm.py`（`launch_gemm_a8w4_tdm`：加 `ccta_ring` const + ring 指针形参；A-load prologue 从"gather recv_x"切到"spin+读 ring 槽"）
- Modify: `aiter/aiter/ops/flydsl/batched_gemm_mxfp4.py`（`flydsl_grouped_gemm_a8w4_masked` host 透传 ring 参数）
- Modify: `aiter/aiter/ops/flydsl/grouped_moe_gfx1250.py`（gemm1 调用处传 ring 参数）
- Test: `aiter/op_tests/flydsl_tests/test_ccta_ring_gemm1_parity.py`

**Interfaces:**
- Consumes: Task 2 生产者写好的 `ring_a`/`ring_as`/`ready`/`freed`/`ring_cons` 指针、`S`、`tile_m`、`num_valid_tiles`。
- Produces: `flydsl_grouped_gemm_a8w4_masked(..., ccta_ring=1, ring_a=, ring_as=, ring_ready=, ring_freed=, ring_cons=, ring_depth=)`：输出与"非 ring 路径 gather"逐字节一致。

- [ ] **Step 1: 写失败测试（端到端单 GPU parity）**

`test_ccta_ring_gemm1_parity.py`：同一 a8w4 gugu no-bias MoE，`AITER_EP_CCTA_RING={0,1}`（都开 `AITER_EP_FP8_TRANSPORT` 基座），断言 GEMM1 logits **逐字节相同**（`atol=0,rtol=0`）——生产者 ring + 消费者 spin 的联合路径 == T-B.4 直 gather 路径。

- [ ] **Step 2: 运行确认失败**

Run: `cd /tmp && python -m pytest /app/aiter/op_tests/flydsl_tests/test_ccta_ring_gemm1_parity.py -v`
Expected: FAIL —— `flydsl_grouped_gemm_a8w4_masked` 无 `ccta_ring` 参数 / 输出不一致。

- [ ] **Step 3: 实现消费端 A-load prologue**

`launch_gemm_a8w4_tdm`：`const_expr(ccta_ring)` 开时，把 tile 号来源从"grid 静态 tile 网格"改成"`t = atomic_add_global(ring_cons,1)` 领取"（消费者 CTA 持久循环，`t >= num_valid_tiles` 退出）；每 tile：`slot=t%S`、`gen=t//S+1`、`spin_until_eq_i32(&ready[slot],gen)`、`fence_system_acquire()`、从 `ring_a[slot]`/`ring_as[slot]` 用 TDM 连续块（**非 gather**，槽内已是连续 grouped tile）载入 A-load LDS（复用现 `add_tdm_loads` 连续路径，**wave-spec 保持默认 ON**——ring 槽是连续块，无 T-B.3 的非均匀 gather 计数问题）；算完 WMMA 的 A 消费后 `fence_system_release()` + `store_i32_system(&freed[slot],0,gen)`。scale 同法。

- [ ] **Step 4: 运行确认通过**

Run: 同 Step 2。Expected: PASS（逐字节）。

- [ ] **Step 5: host 透传 + 提交**

`batched_gemm_mxfp4.py` + `grouped_moe_gfx1250.py` 透传 ring 参数（`ccta_ring=0` 时全默认，向后兼容）。

```bash
cd /app/aiter && git add aiter/ops/flydsl/kernels/mxfp4_preshuffle_gfx1250_tdm.py aiter/ops/flydsl/batched_gemm_mxfp4.py aiter/ops/flydsl/grouped_moe_gfx1250.py op_tests/flydsl_tests/test_ccta_ring_gemm1_parity.py
git commit -m "feat(ep): GEMM1 cross-CTA ring consumer (spin ready, contiguous A-load, wave-spec on, release slot)"
```

---

## Task 4：EP mega 端到端正确性 + 性能验收 + fallback

**Files:** Test: `op_tests/multigpu_tests/test_mega_moe.py`（复用）

- [ ] **Step 1: 正确性（ring on）**

Run:
```bash
cd /tmp && AITER_EP_FP8_TRANSPORT=1 AITER_EP_CCTA_RING=1 \
ENABLE_CK=0 AITER_FORCE_A8W4=1 AITER_USE_GROUPED_GEMM=1 AITER_BF16_FP8_MOE_BOUND=0 \
torchrun --standalone --nproc_per_node=4 \
  /app/aiter/op_tests/multigpu_tests/test_mega_moe.py \
  -q a8w4_mxfp4 -e 384 -k 6 -hd 7168 -id 3072 \
  --combine scatter_fused --layers 2 --acc_verify 1
```
Expected: 2-rank 与 4-rank 均 `MEGA-CHECK PASS`，logits 与基线同量级。

- [ ] **Step 2: 回归（ring off）**

Run: 同上 `AITER_EP_CCTA_RING=0`。Expected: `MEGA-CHECK PASS`，与主线逐 kernel 一致。

- [ ] **Step 3: 性能 A/B（真实维度）**

Run: on/off 各 `--profile_table 1 --layers 61`。核对：(1) 独立 a1 gather kernel 消失、`grouped_a1`（`contiguous_m×hidden` fp8）物化+回读消失；(2) GEMM1 device time 因 ring 重叠 + wave-spec 恢复 **不劣于甚至优于** T-B.4 直 gather（对比小配置 +27% 回归是否消除）；(3) 整层 device time 相对 T-A 基座再降或持平。写回 spec §8.7 结果区。

- [ ] **Step 4: 记录 + 提交**

```bash
cd /app/aiter && git add docs/superpowers/specs/2026-07-29-dispatch-gemm1-fusion-design.md docs/superpowers/plans/2026-07-31-crossCTA-ring-dispatch-gemm1.md
git commit -m "docs(ep): record cross-CTA ring dispatch->gemm1 correctness + perf"
```

- [ ] **Task 2b（后续里程碑，跨 rank pull，本 plan 不展开 bite-sized）**：生产者从"读 LOCAL `recv_x`"升级为"经 `Window.lsa_ptr(peer, off)` 直接 pull 远端 send buffer 的 fp8 token"，配 per-expert 到齐信号（复用 dispatch 现有 `off_recv_num`/grid barrier + `spin_until_gt_i32`），**取代 push dispatch**（真 pull-分组 dispatch）。风险最高（跨 rank 就绪门控/死锁），仅在 Task 1-4 本地 ring 重叠证明有效后启动，届时补独立 detailed plan。

**Fallback（若 Task 0 SPIKE 红，或 Task 3 ring 重叠无净收益）**：
- Task 0 红（flydsl 跨 CTA release/acquire 不可靠）→ 退回**已验证的 `-1` split-phase barrier 弱解耦**（`pipeline_fence_signal/wait`，仍整-WG 汇合但比 T-B.3 全 lockstep 好），或直接停在 T-A 基座。
- Task 3 无净收益（ring 全局往返吃掉重叠收益）→ 保留 ring 正确性开关但默认关，记录实测，评估是否需 Phase 3 整层巨核（spec §7）。

---

## Self-Review

**Spec/架构 coverage**：
- "pull-分组 dispatch" → Task 2（grouped fp8 tile 填 ring，本地源）+ Task 2b（真跨 rank pull）✓
- "跨 CTA 全局 ring 重叠" → Task 0（原语正确性）+ Task 2/3（生产者/消费者跨 CTA）✓
- "明确不做 CTA 内 warp 专精" → Global Constraints 同步铁律（禁命名屏障）+ Task 3（消费者用连续 A-load，wave-spec ON，不 warp-specialize）✓
- "ring spike 放最前面" → Task 0 为唯一前置门，其余任务显式 gated on 其绿 ✓

**Placeholder scan**：Task 0/1 给了完整代码；Task 2/3 的 device body 标注确切参考函数（`tensor_load_gather` / `make_tensor_gather_descriptor` / `_grouped_a8w4_preshuffle_e8m0_scale` / `_build_a_gather_rowmap` / `add_tdm_loads`）与确切原语（`flydsl_prims` 全列名），执行者按名复刻，非 TBD。脚手架统一"照抄 `make_dispatch` 的 `@flyc.kernel`+`@flyc.jit run().launch`"。

**Type consistency**：`ccta_ring`(bool)/`ring_depth`(int)/`_ccta_ring_regions`/`ccta_ring_ptrs`/`reset_ccta_ring` 全程同名；ring region 名 `ring_a`/`ring_as`/`ring_ready`/`ring_freed`/`ring_prod`/`ring_cons` 在 Task 1 定义、Task 2/3 消费一致；generation 语义（`gen=t//S+1`、`ready==gen` / `freed==gen-1`）Task 0→2→3 统一；原语名与 `flydsl_prims.py` 逐一对齐。
