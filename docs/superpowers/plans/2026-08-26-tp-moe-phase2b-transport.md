# TP MoE 阶段二（中）：kernel 内 P2P 传输层 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 做出一个自成一体的 `TPActivationGather`，用 kernel 内 P2P store 完成 TP 组内的 activation all-gather，结果与 `dist.all_gather_into_tensor` 逐位相同。

**Architecture:** 三个新文件，互不依赖已有生产代码。`collective_sched.py` 放从 MegaMoE 照抄过来的调度同步 helper（TP 自己的一份，MegaMoE 源文件一行不改）。`tp_gather.py` 放 `TPActivationGather`，它在构造时用 Mori 分配对称内存并建 p2p 地址表，`gather()` 启动推送 kernel。测试证明输出与 NCCL 逐位相同，并且连续调用、rank 之间有偏差时都正确。

**Tech Stack:** Python 3.12、PyTorch、FlyDSL（`@flyc.jit` / `@flyc.kernel`）、Mori SHMEM、ROCm gfx950、8 卡 torchrun。

**依据文档：** `docs/superpowers/specs/2026-08-26-tp-moe-stage1-fused-p2p-design.md` 第 4 节、5.1 到 5.4 节、第 6 节。

**不在本方案范围内：** GEMM1 的融合、A 与 scale loader 的改动、`TPMoEStage1` 的任何改动、性能验收。那些在阶段二（下）。

**分支：** `dev/all_gather_merge_stage1_naive`。

---

## File Structure

| 文件 | 动作 | 职责 |
|---|---|---|
| `aiter/ops/flydsl/kernels/mega_moe/collective_sched.py` | 新建 | TP 自己的调度同步 helper，照抄 MegaMoE 的逻辑 |
| `aiter/ops/flydsl/kernels/mega_moe/tp_gather.py` | 新建 | `TPActivationGather`：对称内存、p2p 表、推送 kernel、host 入口 |
| `op_tests/multigpu_tests/test_tp_gather.py` | 新建 | 逐位相同、重复调用、rank 偏差、前提校验四组用例 |

**为什么不改 `TPMoEStage1`：** 传输层能独立验证，混进算子里就只能靠端到端数值间接判断。等它证明正确之后，阶段二（下）再接进去。

---

## 设计要点（实现前必须理解）

### 目标行号

`dest_row = tp_rank * m_local + row`，其中 `row` 是本卡的本地行号。这与 `dist.all_gather_into_tensor` 的 rank-major 语义完全一致，所以结果可以逐位对拍。注意用的是运行时的 `m_local`，不是 `max_tok_per_rank`，稠密排列。

### ready flag 的步长选择

设计文档 6.2 节留了一个选择：TP 是每个 producer block 发布一次（步长 `npes * blocks_per_destination`），还是每个源 rank 发布一次（步长 `npes`）。

**本方案选每个源 rank 发布一次。** 做法是 producer block 推完自己那份之后，对本地的 `push_done` 计数器做一次 `atomic_add_agent`；拿到 `dispatch_blocks - 1` 的那个 block 是最后一个，由它代表本卡向所有 peer 的 `payload_ready` 各发一次 `atomic_add_system`。

这么做的好处是 `emit_epoch_rendezvous` 可以照抄 MegaMoE，`expected` 的递增步长仍然是 `npes`，`launch_epoch` 的推导也不用改。代价是一次本地 atomic 和一次 block 间的隐式同步。

### 双缓冲

`rx_x` 和 `rx_scale` 的形状是 `[2, P*MTPR + 1, ...]`，按 parity 选用哪一份。第 N+1 轮写的是另一份，不会覆盖 peer 还在读的第 N 轮数据。末尾多出的那一行是清零的 PAD 行，本方案不用它，但先留着，阶段二（下）的 GEMM1 取数要用。

### 三个 helper 与 MegaMoE 的差异

`_copy_token_row` 和 `emit_ticket_and_roles` 逐字照抄。`emit_epoch_rendezvous` 照抄但删掉三处 EP 专属的 `const_expr` 分支（`payload_tile_ready`、`external_grouping`、`direct_fixed_slot`），因为 TP 全都用不上。work pool 那个 helper 本方案用不到，不写。

---

## Task 1: `collective_sched.py`

**Files:**
- Create: `aiter/ops/flydsl/kernels/mega_moe/collective_sched.py`

- [ ] **Step 1: 建文件**

```python
# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Collective scheduling and synchronisation helpers for the TP MoE kernels.

These are TP's own copies of four pieces of MegaMoE's scheduler. MegaMoE's
source files are deliberately NOT modified and NOT imported from -- see section
6 of docs/superpowers/specs/2026-08-26-tp-moe-stage1-fused-p2p-design.md for
why sharing was rejected. Only ~35 lines are genuinely identical between the
two; the rest diverges because TP has no expert-major routing, no capacity
overflow, and no per-tile payload readiness.

Do not "deduplicate" these against dispatch.py / mega_moe_stage1.py. The two
sides are allowed to evolve independently, and MegaMoEV2 is frozen.

These are trace-time helpers, not device functions: ``@flyc.jit`` bodies are
inlined during tracing, so factoring code in here emits no ``func.call``.
"""

# fmt: off

import flydsl.compiler as flyc
import flydsl.expr as fx
import mori.ir.flydsl as mori_shmem
from flydsl.expr import const_expr, range_constexpr
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec

from aiter.ops.flydsl.kernels import buffer_ops

from .. import communication_ops_utils as comm_ops
from .gemm_util import _buffer_load, _buffer_store, _make_buffer_from_addr


@flyc.jit
def copy_row(source_rsrc, destination_rsrc, lane, *, safe_end_i32, n_i32):
    """Copy one row global->global, dwordx4 per lane.

    Verbatim from MegaMoE's dispatch.py::_copy_token_row. ``n_i32`` is the row
    length in i32 words; ``safe_end_i32`` is the 512-word-aligned main body,
    i.e. ``(n_i32 // 512) * 512``. Handles both the 7168-byte activation row and
    the 224-byte scale row -- the latter has ``safe_end_i32 == 0`` and takes only
    the tail loop.
    """
    lane_offset = lane * fx.Int32(4)
    if const_expr(safe_end_i32 > 0):
        for column in range(lane_offset, safe_end_i32, 512):
            value0 = buffer_ops.buffer_load(source_rsrc, column, vec_width=4, dtype=fx.Int32)
            value1 = buffer_ops.buffer_load(source_rsrc, column + fx.Int32(256), vec_width=4, dtype=fx.Int32)
            buffer_ops.buffer_store(value0, destination_rsrc, column)
            buffer_ops.buffer_store(value1, destination_rsrc, column + fx.Int32(256))
    if const_expr(safe_end_i32 < n_i32):
        for column in range(lane_offset + safe_end_i32, n_i32, 256):
            value = buffer_ops.buffer_load(source_rsrc, column, vec_width=4, dtype=fx.Int32)
            buffer_ops.buffer_store(value, destination_rsrc, column)


def emit_ticket_and_roles(*, tid, lds_scratch, a_entry_count, a_epoch_gate,
        epoch_slot, launch_grid_x, producer_blocks):
    """Take this CTA's launch ticket and derive its role for the round.

    One atomic on thread 0, broadcast to the block through byte 0 of the LDS
    scratch. ``a_entry_count`` is a monotonically increasing i64 counter that is
    never reset: dividing by ``launch_grid_x`` recovers which launch this CTA
    belongs to, and the remainder is its role index within that launch.

    Returns ``(gate_addr, gate_epoch, is_owner, is_producer, producer_slot)``.
    """
    ticket_scratch = fx.recast_iter(fx.Int64, lds_scratch.ptr)
    ticket_view = fx.make_view(ticket_scratch, fx.make_layout(1, 1))
    if tid == fx.Int32(0):
        ticket64 = fx.Int64(
            comm_ops.atomic_add_agent(a_entry_count + fx.Int64(epoch_slot * 8), fx.Int64(1))
        )
        fx.ptr_store(Vec.from_elements([ticket64], fx.Int64), ticket_scratch)
    fx.barrier()
    ticket64 = Vec(ticket_view.load())[0]
    generation = ticket64 // fx.Int64(launch_grid_x)
    ticket = fx.Int32(ticket64 - generation * fx.Int64(launch_grid_x))
    gate_addr = a_epoch_gate + fx.Int64(epoch_slot * 4)
    gate_epoch = fx.Int32(generation + fx.Int64(1))
    is_owner = ticket == fx.Int32(0)
    is_producer = (ticket > fx.Int32(0)) & (ticket <= fx.Int32(producer_blocks))
    producer_slot = ticket - fx.Int32(1)
    return gate_addr, gate_epoch, is_owner, is_producer, producer_slot


def emit_epoch_rendezvous(*, tid, is_owner, parity_rsrc, expected_rsrc,
        p_launch_ready, a_launch_ready, a_reset_counters, reset_count,
        gate_addr, gate_epoch, npes, rank):
    """Flip the epoch, rendezvous with every peer, reset local state, open the gate.

    One indivisible if/else, copied from MegaMoE with its three EP-only
    const_expr branches removed. The owner CTA flips parity and expected,
    publishes its launch epoch to every peer and waits for theirs, zeroes the
    per-launch counters, then stores the gate; every other CTA waits on the gate.

    The peer wait is what stops rank A's round N+1 push from landing in a buffer
    rank B is still reading in round N: on a single stream, B entering round N+1
    means B's round-N kernel retired.

    ``next_parity_lane`` / ``launch_epoch_lane`` are rebound inside the nested
    ``if tid == 0`` and read after it; the readfirstlane pair must stay in this
    function or the SSA merge point moves.

    ``a_reset_counters`` is an i32 array of ``reset_count`` per-launch counters
    (this kernel uses one: push_done) that the owner zeroes each round.
    """
    if is_owner:
        next_parity_lane = fx.Int32(0)
        launch_epoch_lane = fx.Int32(0)
        if tid == fx.Int32(0):
            old_parity = _buffer_load(parity_rsrc, fx.Int32(0), fx.Int32)
            next_parity_lane = old_parity ^ fx.Int32(1)
            previous_expected = _buffer_load(expected_rsrc, next_parity_lane, fx.Int32)
            next_expected = previous_expected + fx.Int32(npes)
            _buffer_store(expected_rsrc, next_parity_lane, next_expected, fx.Int32)
            launch_epoch_lane = (
                (next_expected // fx.Int32(npes)) * fx.Int32(2) - next_parity_lane
            )
        next_parity = fx.Int32(fx.rocdl.readfirstlane(T.i32, next_parity_lane))
        launch_epoch = fx.Int32(fx.rocdl.readfirstlane(T.i32, launch_epoch_lane))
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
        fx.barrier()
        if tid == fx.Int32(0):
            fx.rocdl.s_waitcnt(0)
            comm_ops.fence_agent_release()
            _buffer_store(parity_rsrc, fx.Int32(0), next_parity, fx.Int32)
            fx.rocdl.s_waitcnt(0)
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

每个 reset counter 占 64 字节（`slot * 16` 个 i32），跟 MegaMoE 的 `work_head` 一样一条 cache line 一个，避免 false sharing。

- [ ] **Step 2: 确认能 import 且不牵连 MegaMoE**

```bash
cd /root/workspace/aiter
PYTHONPATH=. python -c "
from aiter.ops.flydsl.kernels.mega_moe import collective_sched as cs
print('helpers:', [n for n in dir(cs) if not n.startswith('__')][:8])
import aiter.ops.flydsl.kernels.mega_moe.dispatch as d
import aiter.ops.flydsl.kernels.mega_moe.mega_moe_stage1 as s1
print('megamoe still imports fine')
"
git diff --stat -- aiter/ops/flydsl/kernels/mega_moe/dispatch.py aiter/ops/flydsl/kernels/mega_moe/mega_moe_stage1.py
```

Expected：打印 helper 名字和 `megamoe still imports fine`，`git diff --stat` **无输出**（MegaMoE 一行没改）。

- [ ] **Step 3: 提交**

```bash
cd /root/workspace/aiter
python -m black --check aiter/ops/flydsl/kernels/mega_moe/collective_sched.py
git add aiter/ops/flydsl/kernels/mega_moe/collective_sched.py
git commit -m "feat(tp-moe): TP's own copy of MegaMoE's scheduling helpers

Copied rather than shared: MegaMoEV2 stays frozen, and only ~35 lines are
genuinely identical between the EP and TP versions. emit_epoch_rendezvous
drops MegaMoE's three EP-only const_expr branches.

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 2: `TPActivationGather` 的对称内存与构造前提

本 task 只做 host 侧，不写 kernel。`gather()` 先留一个抛 `NotImplementedError` 的桩，Task 3 填上。

**Files:**
- Create: `aiter/ops/flydsl/kernels/mega_moe/tp_gather.py`
- Create: `op_tests/multigpu_tests/test_tp_gather.py`

- [ ] **Step 1: 写 host 侧**

```python
# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""In-kernel P2P activation all-gather for tensor-parallel MoE.

Replaces ``dist.all_gather_into_tensor`` with plain stores into peer memory from
inside a kernel. On one node the 8 GPUs are XGMI-connected and peer memory is
directly addressable, so the transfer costs no separate launch and no collective
rendezvous -- the ~17us fixed cost per collective measured in phase 1 goes away.

Result is bit-identical to the NCCL all-gather: destination row is
``tp_rank * m_local + local_row``, i.e. the same rank-major layout.
"""

import torch

import mori.shmem as ms

_SUPPORTED_TP = (4, 8)
_RESET_COUNTERS = 1  # push_done


def _p2p_table(t, rank, npes, device):
    """i64[npes] table of intra-node P2P pointers to ``t`` on every peer."""
    table = torch.zeros(npes, dtype=torch.int64, device=device)
    for pe in range(npes):
        table[pe] = ms.shmem_ptr_p2p(t.data_ptr(), rank, pe)
    return table


class TPActivationGather:
    """Owns the symmetric receive buffers and runs the push kernel.

    Preconditions, checked in ``__init__``:
      * ``mori.shmem.shmem_torch_process_group_init`` has already been called
        by the caller, over a communicator whose size equals ``tp_size``
      * every rank constructs this with identical arguments, in the same order
        (``mori_shmem_create_tensor`` is collective)

    Buffers are sized by ``max_tok_per_rank`` once at construction; Mori's
    symmetric allocation cannot be resized per call.
    """

    def __init__(self, *, model_dim, tp_size, tp_rank, max_tok_per_rank,
                 device=None, num_waves=4, producer_blocks=32, double_buffer=True):
        if int(tp_size) not in _SUPPORTED_TP:
            raise ValueError(f"tp_size={tp_size} unsupported; expected one of {_SUPPORTED_TP}")
        if not (0 <= int(tp_rank) < int(tp_size)):
            raise ValueError(f"tp_rank={tp_rank} out of range for tp_size={tp_size}")
        if int(max_tok_per_rank) <= 0:
            raise ValueError("max_tok_per_rank must be positive")
        if int(model_dim) % 512 != 0:
            # copy_row moves dwordx4 chunks and the scale row is model_dim/32
            # bytes; both need 16-byte granularity, which model_dim % 512 == 0
            # guarantees for the activation row (model_dim bytes) and the scale
            # row (model_dim/32 bytes) simultaneously.
            raise ValueError(f"model_dim={model_dim} must be a multiple of 512")
        if int(producer_blocks) % int(tp_size) != 0:
            raise ValueError(
                f"producer_blocks={producer_blocks} must be divisible by tp_size={tp_size}"
            )

        try:
            shmem_npes = int(ms.shmem_npes())
        except Exception as exc:  # noqa: BLE001 - surface the real cause
            raise RuntimeError(
                "Mori SHMEM is not initialised; call "
                "mori.shmem.shmem_torch_process_group_init(<pg name>) before "
                f"constructing TPActivationGather ({type(exc).__name__}: {exc})"
            ) from exc
        if shmem_npes != int(tp_size):
            raise ValueError(
                f"Mori SHMEM world size {shmem_npes} != tp_size {tp_size}; the "
                "shmem communicator must be the TP group"
            )

        self.model_dim = int(model_dim)
        self.tp_size = int(tp_size)
        self.tp_rank = int(tp_rank)
        self.mtpr = int(max_tok_per_rank)
        self.num_waves = int(num_waves)
        self.producer_blocks = int(producer_blocks)
        self.slots = 2 if double_buffer else 1
        dev = device or torch.device("cuda", torch.cuda.current_device())
        if dev.type == "cuda" and dev.index is None:
            dev = torch.device("cuda", torch.cuda.current_device())
        self.device = dev

        self.scale_dim = self.model_dim // 32
        # +1 row: a zeroed PAD row the fused GEMM1 will clamp out-of-range token
        # ids to. Unused by this class, allocated now so the layout is stable.
        self.rows = self.tp_size * self.mtpr + 1
        self.row_bytes = self.model_dim
        self.scale_bytes = self.scale_dim

        # Slab byte counts are passed to the kernel as i32, and buffer resources
        # carry a 32-bit offset anyway (mega_moe_stage1.py:29).
        if self.rows * self.row_bytes >= (1 << 31):
            raise ValueError(
                f"symmetric slab {self.rows * self.row_bytes} bytes exceeds the i32 "
                f"offset ABI; lower max_tok_per_rank={self.mtpr}"
            )

        self.rx_x = ms.mori_shmem_create_tensor(
            (self.slots, self.rows, self.row_bytes), torch.uint8
        )
        self.rx_scale = ms.mori_shmem_create_tensor(
            (self.slots, self.rows, self.scale_bytes), torch.uint8
        )
        self.payload_ready = ms.mori_shmem_create_tensor((2,), torch.int32)
        self.launch_ready = ms.mori_shmem_create_tensor((self.tp_size,), torch.int32)
        for t in (self.rx_x, self.rx_scale, self.payload_ready, self.launch_ready):
            t.zero_()
        ms.shmem_barrier_all()

        self.p2p_rx_x = _p2p_table(self.rx_x, self.tp_rank, self.tp_size, dev)
        self.p2p_rx_scale = _p2p_table(self.rx_scale, self.tp_rank, self.tp_size, dev)
        self.p2p_payload_ready = _p2p_table(
            self.payload_ready, self.tp_rank, self.tp_size, dev
        )
        self.p2p_launch_ready = _p2p_table(
            self.launch_ready, self.tp_rank, self.tp_size, dev
        )

        # Local (non-symmetric) per-launch state.
        self.epoch_parity = torch.zeros(1, dtype=torch.int32, device=dev)
        self.epoch_expected = torch.zeros(2, dtype=torch.int32, device=dev)
        self.epoch_gate = torch.zeros(10, dtype=torch.int32, device=dev)
        self.entry_count = torch.zeros(10, dtype=torch.int64, device=dev)
        # One 64-byte cache line per counter, like MegaMoE's work_head.
        self.reset_counters = torch.zeros(_RESET_COUNTERS * 16, dtype=torch.int32, device=dev)

    def slab_bytes(self, kind):
        """Bytes per parity slab, for the kernel's parity offset arithmetic."""
        if kind == "x":
            return self.rows * self.row_bytes
        if kind == "scale":
            return self.rows * self.scale_bytes
        raise ValueError(kind)

    def _validate(self, x_q, x_scale):
        m_local = int(x_q.shape[0])
        if m_local <= 0:
            raise ValueError("m_local must be positive")
        if m_local > self.mtpr:
            raise ValueError(f"m_local={m_local} exceeds max_tok_per_rank={self.mtpr}")
        if x_q.dtype != torch.float8_e4m3fn or not x_q.is_contiguous():
            raise ValueError("x_q must be contiguous float8_e4m3fn")
        if tuple(x_q.shape) != (m_local, self.model_dim):
            raise ValueError(f"x_q must be [{m_local}, {self.model_dim}], got {tuple(x_q.shape)}")
        if x_scale.dtype not in (torch.uint8, torch.float8_e8m0fnu):
            raise ValueError(f"x_scale must be uint8 or float8_e8m0fnu, got {x_scale.dtype}")
        if not x_scale.is_contiguous():
            raise ValueError("x_scale must be contiguous")
        if tuple(x_scale.shape) != (m_local, self.scale_dim):
            raise ValueError(
                f"x_scale must be [{m_local}, {self.scale_dim}], got {tuple(x_scale.shape)}"
            )
        for name, t in (("x_q", x_q), ("x_scale", x_scale)):
            if t.device != self.device:
                raise ValueError(f"{name} is on {t.device}, expected {self.device}")
        return m_local

    def current_parity(self):
        """Parity the NEXT gather() will write. The kernel flips before pushing."""
        return int(self.epoch_parity[0].item()) ^ 1 if self.slots == 2 else 0

    def views(self, m_local, parity):
        """Views of the gathered result for a given round. Valid until the next gather."""
        n = self.tp_size * int(m_local)
        x = self.rx_x[parity, :n].view(torch.float8_e4m3fn)
        scale = self.rx_scale[parity, :n]
        return x, scale

    def gather(self, x_q, x_scale):
        raise NotImplementedError("push kernel lands in task 3")
```

- [ ] **Step 2: 写构造前提的测试（先写，必须能失败）**

创建 `op_tests/multigpu_tests/test_tp_gather.py`：

```python
# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""TPActivationGather correctness tests. Run with:

    torchrun --standalone --nproc_per_node=8 \
        op_tests/multigpu_tests/test_tp_gather.py --case <name>
"""

import argparse
import os
import sys

import mori.shmem as ms
import torch
import torch.distributed as dist

from aiter.ops.flydsl.kernels.mega_moe.quant import per_1x32_mx_quant
from aiter.ops.flydsl.kernels.mega_moe.tp_gather import TPActivationGather

MODEL_DIM = 7168


def _setup():
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    if not dist.is_initialized():
        dist.init_process_group("cpu:gloo,cuda:nccl", device_id=device)
    import torch._C._distributed_c10d as c10d

    c10d._register_process_group("default", dist.group.WORLD)
    ms.shmem_torch_process_group_init("default")
    return rank, world, device


def _teardown():
    try:
        ms.shmem_finalize()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _make_x(m_local, device, seed):
    g = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randn((m_local, MODEL_DIM), generator=g).to(
        device=device, dtype=torch.bfloat16
    ) * (MODEL_DIM**-0.25)
    return x.contiguous()


def _nccl_gather(t, world):
    out = torch.empty((t.shape[0] * world,) + tuple(t.shape[1:]), dtype=t.dtype, device=t.device)
    dist.all_gather_into_tensor(out, t.contiguous())
    return out


def case_construct():
    """Constructor preconditions. Runs under torchrun because shmem needs a PG."""
    rank, world, device = _setup()
    try:
        common = dict(model_dim=MODEL_DIM, tp_size=world, tp_rank=rank,
                      max_tok_per_rank=128, device=device)

        for kwargs, want in (
            (dict(tp_size=2, tp_rank=0), "tp_size"),
            (dict(tp_rank=world + 1), "out of range"),
            (dict(max_tok_per_rank=0), "positive"),
            (dict(model_dim=7000), "multiple of 512"),
            (dict(producer_blocks=30), "divisible"),
        ):
            merged = dict(common)
            merged.update(kwargs)
            try:
                TPActivationGather(**merged)
            except ValueError as exc:
                assert want in str(exc), (want, str(exc))
            else:
                raise AssertionError(f"expected ValueError containing {want!r} for {kwargs}")

        g = TPActivationGather(**common)
        assert g.rows == world * 128 + 1, g.rows
        assert g.scale_dim == MODEL_DIM // 32
        assert tuple(g.rx_x.shape) == (2, world * 128 + 1, MODEL_DIM)
        assert tuple(g.rx_scale.shape) == (2, world * 128 + 1, MODEL_DIM // 32)
        assert tuple(g.p2p_rx_x.shape) == (world,)
        assert int((g.p2p_rx_x != 0).sum()) == world, "every peer pointer must be non-null"
        assert g.p2p_rx_x[rank].item() == g.rx_x.data_ptr(), "self entry must be the local ptr"

        x = _make_x(4, device, 1)
        x_q, x_s = per_1x32_mx_quant(x, quant_mode="fp8")
        for bad, want in (
            ((x_q[:200000], x_s), "must be positive"),
            ((x_q.to(torch.uint8), x_s), "float8_e4m3fn"),
            ((x_q, x_s[:, :2]), "x_scale must be"),
        ):
            try:
                g._validate(*bad)
            except (ValueError, IndexError) as exc:
                assert want in str(exc) or isinstance(exc, IndexError), (want, str(exc))
            else:
                raise AssertionError(f"expected rejection for {want}")

        big = torch.empty((129, MODEL_DIM), dtype=torch.float8_e4m3fn, device=device)
        big_s = torch.empty((129, MODEL_DIM // 32), dtype=torch.uint8, device=device)
        try:
            g._validate(big, big_s)
        except ValueError as exc:
            assert "exceeds max_tok_per_rank" in str(exc), exc
        else:
            raise AssertionError("m_local > max_tok_per_rank must raise")

        if rank == 0:
            print("case_construct OK")
    finally:
        _teardown()


CASES = {"construct": case_construct}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default="construct")
    args = ap.parse_args()
    CASES[args.case]()


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: 跑测试**

```bash
cd /root/workspace/aiter
PYTHONPATH=. torchrun --standalone --nproc_per_node=8 \
    op_tests/multigpu_tests/test_tp_gather.py --case construct
```

Expected：`case_construct OK`。

- [ ] **Step 4: 负对照 — 证明 shmem 未初始化会被抓到**

```bash
cd /root/workspace/aiter
PYTHONPATH=. python -c "
import torch
from aiter.ops.flydsl.kernels.mega_moe.tp_gather import TPActivationGather
torch.cuda.set_device(0)
try:
    TPActivationGather(model_dim=7168, tp_size=8, tp_rank=0, max_tok_per_rank=128,
                       device=torch.device('cuda', 0))
except RuntimeError as e:
    print('CAUGHT:', str(e)[:90])
else:
    raise SystemExit('FAIL: uninitialised shmem was not rejected')
"
```

Expected：`CAUGHT: Mori SHMEM is not initialised; ...`。若它没抛而是崩溃或成功，说明 `ms.shmem_npes()` 在未初始化时的行为与假设不符，**停下来报告**，改用别的探测方式。

- [ ] **Step 5: 提交**

```bash
cd /root/workspace/aiter
python -m black --check aiter/ops/flydsl/kernels/mega_moe/tp_gather.py \
    op_tests/multigpu_tests/test_tp_gather.py
git add aiter/ops/flydsl/kernels/mega_moe/tp_gather.py \
        op_tests/multigpu_tests/test_tp_gather.py
git commit -m "feat(tp-moe): TPActivationGather symmetric memory and preconditions

Host side only; gather() still raises NotImplementedError. Allocates the
double-buffered symmetric receive buffers, builds the p2p pointer tables, and
rejects an uninitialised or mis-sized Mori communicator at construction rather
than at first launch.

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 3: 推送 kernel

**Files:**
- Modify: `aiter/ops/flydsl/kernels/mega_moe/tp_gather.py`（加 kernel 与 `gather()`）
- Modify: `op_tests/multigpu_tests/test_tp_gather.py`（加三组用例）

- [ ] **Step 1: 写 kernel**

在 `tp_gather.py` 的 import 段补上：

```python
import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
import mori.ir.flydsl as mori_shmem

from .. import communication_ops_utils as comm_ops
from ..tensor_shim import _run_compiled
from .collective_sched import copy_row, emit_epoch_rendezvous, emit_ticket_and_roles
from .gemm_util import _buffer_load, _buffer_store, _make_buffer_from_addr
```

在 `TPActivationGather` 之前加：

```python
# fmt: off
@functools.cache
def compile_tp_gather(*, model_dim: int, npes: int, rank: int, producer_blocks: int,
        num_waves: int = 4, slots: int = 2):
    """Push this rank's quantized rows into every peer's symmetric receive buffer."""
    TOTAL_THREADS = num_waves * 64
    ROW_BYTES = model_dim
    SCALE_BYTES = model_dim // 32
    ROW_I32 = ROW_BYTES // 4
    SCALE_I32 = SCALE_BYTES // 4
    ROW_SAFE_END = (ROW_I32 // 512) * 512
    SCALE_SAFE_END = (SCALE_I32 // 512) * 512
    BLOCKS_PER_DEST = producer_blocks // npes
    LAUNCH_GRID_X = 1 + producer_blocks
    EPOCH_SLOT = 0

    kernel_name = (
        f"tp_gather_d{model_dim}_n{npes}_pb{producer_blocks}"
        f"_w{num_waves}_s{slots}"
    )

    # Declared in the factory scope, not inside the kernel body -- this mirrors
    # mega_moe_stage1.py:160-163, where SharedStorage sits outside @flyc.kernel.
    @fx.struct
    class SharedStorage:
        scratch: fx.Array[fx.Int8, 64, 16]

    @flyc.kernel(name=kernel_name, known_block_size=[TOTAL_THREADS, 1, 1])
    def kernel(
        addr_x_q: fx.Int64, addr_x_scale: fx.Int64,
        addr_p2p_rx_x: fx.Int64, addr_p2p_rx_scale: fx.Int64,
        addr_p2p_payload_ready: fx.Int64, addr_p2p_launch_ready: fx.Int64,
        addr_payload_ready: fx.Int64, addr_launch_ready: fx.Int64,
        addr_parity: fx.Int64, addr_expected: fx.Int64,
        addr_epoch_gate: fx.Int64, addr_entry_count: fx.Int64,
        addr_reset: fx.Int64,
        m_local: fx.Int32, x_slab_bytes: fx.Int32, scale_slab_bytes: fx.Int32,
    ):
        tid = fx.thread_idx.x
        lane = tid & fx.Int32(63)
        warp = tid // fx.Int32(64)
        lds = fx.SharedAllocator().allocate(SharedStorage).peek()

        parity_rsrc = _make_buffer_from_addr(addr_parity, fx.Int32)
        expected_rsrc = _make_buffer_from_addr(addr_expected, fx.Int32)

        gate_addr, gate_epoch, is_owner, is_producer, producer_slot = emit_ticket_and_roles(
            tid=tid, lds_scratch=lds.scratch, a_entry_count=addr_entry_count,
            a_epoch_gate=addr_epoch_gate, epoch_slot=EPOCH_SLOT,
            launch_grid_x=LAUNCH_GRID_X, producer_blocks=producer_blocks)

        emit_epoch_rendezvous(
            tid=tid, is_owner=is_owner, parity_rsrc=parity_rsrc,
            expected_rsrc=expected_rsrc, p_launch_ready=addr_p2p_launch_ready,
            a_launch_ready=addr_launch_ready, a_reset_counters=addr_reset,
            reset_count=1, gate_addr=gate_addr, gate_epoch=gate_epoch,
            npes=npes, rank=rank)

        parity = _buffer_load(parity_rsrc, fx.Int32(0), fx.Int32, cache_modifier=1)
        expected = _buffer_load(expected_rsrc, parity, fx.Int32, cache_modifier=1)
        if const_expr(slots == 1):
            slab = fx.Int32(0)
        else:
            slab = parity
        x_base_off = fx.Int64(slab) * fx.Int64(x_slab_bytes)
        s_base_off = fx.Int64(slab) * fx.Int64(scale_slab_bytes)

        if is_producer:
            destination = producer_slot // fx.Int32(BLOCKS_PER_DEST)
            sub = producer_slot - destination * fx.Int32(BLOCKS_PER_DEST)
            rx_table = _make_buffer_from_addr(addr_p2p_rx_x, fx.Int64)
            sc_table = _make_buffer_from_addr(addr_p2p_rx_scale, fx.Int64)
            peer_x = _buffer_load(rx_table, destination, fx.Int64) + x_base_off
            peer_s = _buffer_load(sc_table, destination, fx.Int64) + s_base_off
            dest_base = fx.Int32(rank) * m_local
            row0 = sub + warp * fx.Int32(BLOCKS_PER_DEST)
            row_stride = fx.Int32(BLOCKS_PER_DEST * num_waves)
            for row in range(row0, m_local, row_stride):
                dest_row = dest_base + row
                copy_row(
                    _make_buffer_from_addr(addr_x_q + fx.Int64(row) * fx.Int64(ROW_BYTES), fx.Int32, 4),
                    _make_buffer_from_addr(peer_x + fx.Int64(dest_row) * fx.Int64(ROW_BYTES), fx.Int32, 4),
                    lane, safe_end_i32=ROW_SAFE_END, n_i32=ROW_I32)
                copy_row(
                    _make_buffer_from_addr(addr_x_scale + fx.Int64(row) * fx.Int64(SCALE_BYTES), fx.Int32, 4),
                    _make_buffer_from_addr(peer_s + fx.Int64(dest_row) * fx.Int64(SCALE_BYTES), fx.Int32, 4),
                    lane, safe_end_i32=SCALE_SAFE_END, n_i32=SCALE_I32)
            fx.rocdl.s_waitcnt(0)
            fx.barrier()
            # Last producer block on this rank publishes once to every peer, so
            # payload_ready advances by exactly npes per launch and the expected
            # step in emit_epoch_rendezvous stays npes.
            if tid == fx.Int32(0):
                comm_ops.fence_system_release()
                done = fx.Int32(comm_ops.atomic_add_agent(addr_reset, fx.Int32(1)))
                if done == fx.Int32(producer_blocks - 1):
                    pr_table = _make_buffer_from_addr(addr_p2p_payload_ready, fx.Int64)
                    for pe in range_constexpr(npes):
                        remote = _buffer_load(pr_table, fx.Int32(pe), fx.Int64)
                        comm_ops.atomic_add_system(
                            remote + fx.Int64(parity) * fx.Int64(4), fx.Int32(1))

        if tid == fx.Int32(0):
            mori_shmem.int32_wait_until_equals(
                addr_payload_ready + fx.Int64(parity) * fx.Int64(4), expected)
            comm_ops.fence_system_acquire()
        fx.barrier()

    @flyc.jit
    def launch(
        addr_x_q: fx.Int64, addr_x_scale: fx.Int64,
        addr_p2p_rx_x: fx.Int64, addr_p2p_rx_scale: fx.Int64,
        addr_p2p_payload_ready: fx.Int64, addr_p2p_launch_ready: fx.Int64,
        addr_payload_ready: fx.Int64, addr_launch_ready: fx.Int64,
        addr_parity: fx.Int64, addr_expected: fx.Int64,
        addr_epoch_gate: fx.Int64, addr_entry_count: fx.Int64, addr_reset: fx.Int64,
        m_local: fx.Int32, x_slab_bytes: fx.Int32, scale_slab_bytes: fx.Int32,
        stream: fx.Stream,
    ):
        kernel(
            addr_x_q, addr_x_scale, addr_p2p_rx_x, addr_p2p_rx_scale,
            addr_p2p_payload_ready, addr_p2p_launch_ready, addr_payload_ready,
            addr_launch_ready, addr_parity, addr_expected, addr_epoch_gate,
            addr_entry_count, addr_reset, m_local, x_slab_bytes, scale_slab_bytes,
            value_attrs={
                "rocdl.waves_per_eu": 2,
                "rocdl.flat_work_group_size": f"{TOTAL_THREADS},{TOTAL_THREADS}",
            },
        ).launch(grid=(LAUNCH_GRID_X, 1, 1), block=(TOTAL_THREADS, 1, 1), stream=stream)

    return launch
# fmt: on
```

`const_expr` 和 `range_constexpr` 都要加进 `tp_gather.py` 的 import：`from flydsl.expr import const_expr, range_constexpr`。

- [ ] **Step 2: 实现 `gather()`**

把 `TPActivationGather.gather` 的桩替换为：

```python
    def gather(self, x_q, x_scale, stream=None):
        """Push this rank's rows to every peer and wait for theirs.

        Returns ``(rx_x, rx_scale)`` views of shape ``[tp_size*m_local, ...]``,
        valid until the next call (double buffering gives one round of slack).
        """
        m_local = self._validate(x_q, x_scale)
        launch = compile_tp_gather(
            model_dim=self.model_dim, npes=self.tp_size, rank=self.tp_rank,
            producer_blocks=self.producer_blocks, num_waves=self.num_waves,
            slots=self.slots,
        )
        if stream is None:
            stream = fx.Stream(torch.cuda.current_stream())
        parity = self.current_parity()
        _run_compiled(
            launch,
            fx.Int64(x_q.data_ptr()), fx.Int64(x_scale.data_ptr()),
            fx.Int64(self.p2p_rx_x.data_ptr()), fx.Int64(self.p2p_rx_scale.data_ptr()),
            fx.Int64(self.p2p_payload_ready.data_ptr()),
            fx.Int64(self.p2p_launch_ready.data_ptr()),
            fx.Int64(self.payload_ready.data_ptr()),
            fx.Int64(self.launch_ready.data_ptr()),
            fx.Int64(self.epoch_parity.data_ptr()),
            fx.Int64(self.epoch_expected.data_ptr()),
            fx.Int64(self.epoch_gate.data_ptr()),
            fx.Int64(self.entry_count.data_ptr()),
            fx.Int64(self.reset_counters.data_ptr()),
            fx.Int32(m_local),
            fx.Int32(self.slab_bytes("x")), fx.Int32(self.slab_bytes("scale")),
            stream,
        )
        return self.views(m_local, parity)
```

- [ ] **Step 3: 写逐位相同的测试**

在 `test_tp_gather.py` 里加：

```python
def case_bitexact():
    """The push must reproduce dist.all_gather_into_tensor byte for byte."""
    rank, world, device = _setup()
    try:
        g = TPActivationGather(model_dim=MODEL_DIM, tp_size=world, tp_rank=rank,
                               max_tok_per_rank=256, device=device)
        for m_local in (1, 2, 7, 8, 64, 128, 256):
            x = _make_x(m_local, device, 3000 + rank * 13 + m_local)
            x_q, x_s = per_1x32_mx_quant(x, quant_mode="fp8")
            want_x = _nccl_gather(x_q, world)
            want_s = _nccl_gather(x_s, world)
            got_x, got_s = g.gather(x_q, x_s)
            torch.cuda.synchronize()
            assert got_x.shape == want_x.shape, (got_x.shape, want_x.shape)
            assert torch.equal(got_x.view(torch.uint8), want_x.view(torch.uint8)), (
                f"activation mismatch at m_local={m_local}: "
                f"{int((got_x.view(torch.uint8) != want_x.view(torch.uint8)).sum())} bytes"
            )
            assert torch.equal(got_s.view(torch.uint8), want_s.view(torch.uint8)), (
                f"scale mismatch at m_local={m_local}: "
                f"{int((got_s.view(torch.uint8) != want_s.view(torch.uint8)).sum())} bytes"
            )
            if rank == 0:
                print(f"  m_local={m_local} rows={got_x.shape[0]} bit-identical")
        if rank == 0:
            print("case_bitexact OK")
    finally:
        _teardown()


def case_repeat():
    """The epoch/parity protocol must survive repeated calls, not just the first."""
    rank, world, device = _setup()
    try:
        g = TPActivationGather(model_dim=MODEL_DIM, tp_size=world, tp_rank=rank,
                               max_tok_per_rank=128, device=device)
        for it in range(12):
            m_local = (it % 4) * 8 + 8
            x = _make_x(m_local, device, 5000 + rank * 7 + it)
            x_q, x_s = per_1x32_mx_quant(x, quant_mode="fp8")
            want_x = _nccl_gather(x_q, world)
            got_x, _ = g.gather(x_q, x_s)
            torch.cuda.synchronize()
            assert torch.equal(got_x.view(torch.uint8), want_x.view(torch.uint8)), (
                f"iteration {it} (m_local={m_local}) mismatch: "
                f"{int((got_x.view(torch.uint8) != want_x.view(torch.uint8)).sum())} bytes"
            )
        if rank == 0:
            print("case_repeat OK (12 iterations)")
    finally:
        _teardown()


def case_skew():
    """A slow rank must not corrupt a fast rank's buffer.

    Rank 0 sleeps before each call so the ranks enter the kernel far apart. The
    launch_ready handshake plus double buffering is what makes this safe; with
    either removed, a fast rank's round N+1 push lands in a buffer a slow rank
    is still reading from round N.
    """
    import time

    rank, world, device = _setup()
    try:
        g = TPActivationGather(model_dim=MODEL_DIM, tp_size=world, tp_rank=rank,
                               max_tok_per_rank=128, device=device)
        for it in range(6):
            if rank == 0:
                time.sleep(0.05)
            m_local = 64
            x = _make_x(m_local, device, 7000 + rank * 11 + it)
            x_q, x_s = per_1x32_mx_quant(x, quant_mode="fp8")
            want_x = _nccl_gather(x_q, world)
            got_x, _ = g.gather(x_q, x_s)
            torch.cuda.synchronize()
            bad = int((got_x.view(torch.uint8) != want_x.view(torch.uint8)).sum())
            assert bad == 0, f"iteration {it} rank {rank}: {bad} bytes differ under skew"
        if rank == 0:
            print("case_skew OK (6 iterations, rank 0 delayed 50ms)")
    finally:
        _teardown()


CASES = {
    "construct": case_construct,
    "bitexact": case_bitexact,
    "repeat": case_repeat,
    "skew": case_skew,
}
```

- [ ] **Step 4: 跑逐位相同**

```bash
cd /root/workspace/aiter
PYTHONPATH=. torchrun --standalone --nproc_per_node=8 \
    op_tests/multigpu_tests/test_tp_gather.py --case bitexact
```

Expected：七行 `m_local=N rows=8N bit-identical`，最后 `case_bitexact OK`。

首次运行会编译 kernel，慢几十秒属正常。

- [ ] **Step 5: 负对照 — 目标行号写错必须失败**

把 kernel 里的 `dest_base = fx.Int32(rank) * m_local` 临时改成 `dest_base = fx.Int32(rank)`（变成行主序而非 rank 主序），重跑 Step 4。

Expected：**FAIL**，`activation mismatch at m_local=... : NNNN bytes`。

看到失败后改回去。若它没失败，说明测试没有真的在比对，先修测试。

- [ ] **Step 6: 负对照 — 去掉握手必须在 skew 下失败**

先确认 `skew` 在正常代码下通过：

```bash
cd /root/workspace/aiter
PYTHONPATH=. torchrun --standalone --nproc_per_node=8 \
    op_tests/multigpu_tests/test_tp_gather.py --case skew
```

Expected：`case_skew OK (6 iterations, rank 0 delayed 50ms)`。

然后构造 `TPActivationGather(..., double_buffer=False)` 并把 `emit_epoch_rendezvous` 里 `if tid < fx.Int32(npes):` 那整段握手临时注释掉，重跑 `skew`。

Expected：**FAIL**，某次迭代出现字节不一致。

> 这个负对照可能不稳定复现（竞争窗口取决于调度）。如果连跑三次都不失败，**不要**据此认为握手是多余的——记录下来，说明这个用例的敏感度不足，握手保留。竞争测试证明不了不存在竞争。

改回去后确认 `skew` 恢复通过。

- [ ] **Step 7: 跑重复调用**

```bash
cd /root/workspace/aiter
PYTHONPATH=. torchrun --standalone --nproc_per_node=8 \
    op_tests/multigpu_tests/test_tp_gather.py --case repeat
```

Expected：`case_repeat OK (12 iterations)`。

> 这个用例专抓 epoch/parity 的 bug。单次调用下 flag 从零开始，什么都看不出来；`expected` 递增写错、`payload_ready` 没按 parity 分开、`reset_counters` 没清零，都要跑到第二次或第三次才暴露。

- [ ] **Step 8: 四个用例全绿 + black**

```bash
cd /root/workspace/aiter
for c in construct bitexact repeat skew; do
  printf "%-10s " "$c"
  PYTHONPATH=. torchrun --standalone --nproc_per_node=8 \
      op_tests/multigpu_tests/test_tp_gather.py --case $c 2>&1 | tail -1
done
python -m black --check aiter/ops/flydsl/kernels/mega_moe/tp_gather.py \
    aiter/ops/flydsl/kernels/mega_moe/collective_sched.py \
    op_tests/multigpu_tests/test_tp_gather.py
```

Expected：四行 OK，black 干净。

- [ ] **Step 9: 确认 MegaMoEV2 依然没被碰过**

```bash
cd /root/workspace/aiter
git diff --stat main...HEAD -- aiter/ops/flydsl/kernels/mega_moe/mega_moe_v2.py \
    aiter/ops/flydsl/kernels/mega_moe/mega_moe_stage1.py \
    aiter/ops/flydsl/kernels/mega_moe/dispatch.py \
    aiter/ops/flydsl/kernels/mega_moe/gemm1.py \
    aiter/ops/flydsl/kernels/mega_moe/gemm_util.py
```

Expected：**无输出**。

- [ ] **Step 10: 提交**

```bash
cd /root/workspace/aiter
git add aiter/ops/flydsl/kernels/mega_moe/tp_gather.py \
        op_tests/multigpu_tests/test_tp_gather.py
git commit -m "feat(tp-moe): in-kernel P2P activation all-gather

Replaces the NCCL collective with plain stores into peer memory from inside a
kernel. Output is bit-identical to dist.all_gather_into_tensor across m_local
1..256; verified it fails when the destination row formula is perturbed to
row-major.

One publish per source rank (the last producer block wins a local counter),
which keeps the expected step at npes and lets emit_epoch_rendezvous stay a
straight copy of MegaMoE's.

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## 验收

本方案完成的判据是四个用例全绿，且 `bitexact` 的负对照确认过会失败。

**不做性能测量。** 推送快不快在这一步无意义，因为收益来自与 GEMM1 合并成一次 launch，而 GEMM1 还没接进来。单独测一个推送 kernel 只会得到一个没有对照物的数字。性能验收在阶段二（下）。

---

## Self-Review

**Spec 覆盖：** 设计文档 4 节的对称内存布局、双缓冲、构造前提由 Task 2 实现；5.1 的 grid 与角色、5.2 的执行顺序、5.3 的双缓冲与握手、5.4 的 producer 任务划分由 Task 3 实现；第 6 节的拷贝而非抽取由 Task 1 实现，并在 Task 3 Step 9 验证 MegaMoE 未被改动。

**有意不覆盖：** 5.5 的 GEMM1 取数改动、5.6 的复用清单里属于 GEMM1 的部分、第 7 节除 gather 用例外的测试、第 8 节的性能验收。这些都依赖 GEMM1 融合，属于阶段二（下）。

**6.2 节留的设计选择已定：** 每个源 rank 发布一次 ready flag，而不是每个 producer block 发布一次。理由写在「设计要点」一节。

**类型一致性：** `compile_tp_gather` 的 16 个 kernel 参数与 `launch` 的前 16 个逐一对应，`gather()` 按同样顺序传参。`copy_row` 的关键字参数在 `collective_sched.py` 里是 `safe_end_i32` / `n_i32`，Task 3 的两处调用用的是同样的名字。`emit_epoch_rendezvous` 的 `reset_count=1` 与 `tp_gather.py` 里 `_RESET_COUNTERS = 1` 一致。

**已知风险：** Task 3 Step 6 的竞争负对照可能不稳定复现，方案里已写明不得据此删掉握手。另外 `ms.shmem_npes()` 在未初始化时的行为是假设的，Task 2 Step 4 专门验证它，不符就停下报告。
