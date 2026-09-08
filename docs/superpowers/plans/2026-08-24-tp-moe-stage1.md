# TPMoEStage1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 新增一个 TP4/TP8 的 MoE Stage1 算子 `TPMoEStage1`，输入是 DP 切分的 token 分片，内部完成 all-gather，输出对齐 aiter 通用 FlyDSL v2 FMoE GEMM2 的 6-tensor ABI。

**Architecture:** 每个 TP rank 持有**全部** expert，只持有 `inter_dim` 的 1/tp 分片（W1 沿 `2*inter_dim` 切、W2 沿 `inter_dim` 切）。DP 分片数 == TP 数，两者是同一个 process group。阶段一 baseline **不写任何新 kernel**——用 `torch.distributed.all_gather_into_tensor` 把 BF16 activation / topk_ids / route_weights 收齐，再串起现成的 `moe_sorting` → `fused_dynamic_mxfp8_quant_moe_sort` → `flydsl_moe_stage1(v2_output_layout=True)`。阶段二把 transport 换成 kernel 内融合的 all-gather，API 不变，两版可在同一进程内对拍。

**Tech Stack:** PyTorch + `torch.distributed`（gloo+nccl）、aiter FlyDSL MoE kernels、gfx950（MI355X）、torchrun 8 rank。

---

## 背景：已敲定的设计决定

| # | 决定 |
|---|---|
| 1 | 做 TP8/TP4 Stage1，不做 EP 中间物；`dev/tp_fuse_gemm1_v0` 忽略 |
| 2 | TP 语义：每卡全部 384 expert，`inter_dim` 切 tp |
| 3 | 算子站在 all-gather **内侧**，自己做 gather |
| 4 | `tp_size` 可配，仅接受 4 / 8 |
| 5 | 有状态 class `TPMoEStage1`，落在 `mega_moe/tp_moe_stage1.py` |
| 6 | `group=None`（默认 WORLD）取 `tp_size`/`tp_rank`；device 用 `torch.cuda.current_device()`，**不用 rank** |
| 7 | 两个入口 `forward` / `forward_prequant` |
| 8 | W1/W1_scale 由调用方传入已 shuffle 的 |
| 9 | `swiglu_limit` 暴露，默认 0.0 |
| 10 | `sort_block_m` 暴露，默认 32 |
| 11 | 输出每次调用新分配 |
| 12 | 各 rank token 数必须相等（文档化前提） |
| 13 | 输出对齐 v2 FMoE GEMM2，A = FP8 E4M3 |
| 14 | 下游由调用方直接调 `mxfp4_moe_gemm2`，不碰 `aiter/configs/` |
| 15 | `sorted_expert_ids` = local expert id；`sorted_token_ids` = `(slot<<24)\|global_token` |
| 16 | 阶段一 baseline 无新 kernel |
| 17 | 阶段二融合版 API 不变 |
| 18 | baseline gather BF16；融合版才先 quant 再 gather |
| 19 | `MegaMoEV2` 完全冻结，纯新增 |
| 20 | torch fp32 逐行参考 + 端到端验收 |
| 21 | 不接 CI，本地 torchrun 验证 |
| 22 | DP group == TP group，`M_global = tp_size * M_local` |

## 已验证的事实（实施时不要重新推导）

- `sorted_token_ids` 位布局：`(topk_slot << 24) | token_id`，padding sentinel `(topk << 24) | M`
  （`aiter/ops/flydsl/kernels/moe_sorting_kernel.py:17-20`）
- `num_valid_ids` 是 i32[2]：`[0]` = padded 有效行数，`[1]` = 真实 token 数
  （`moe_sorting_kernel.py:526-527`）；GEMM2 只读 `[0]`
- `flydsl_moe_stage1` **不做量化**，`a` 必须已是 FP8，`a1_scale` 必须已 sorted+shuffled
  （`aiter/ops/flydsl/moe_kernels.py:1609-1611`）
- v2 模式返回 `(payload, e8m0_scale)` 元组，payload 是 sorted-row-major
  （`moe_kernels.py:1874-1879`）
- `inter_dim=384` 时 `tile_n` 必须整除 384；`resolve_flydsl_stage1_tile_n(384, 64) -> 64`，
  但 `resolve_flydsl_stage1_tile_n(384, 256) -> 128`（静默降级并告警）
- `act="silu"` 且 `swiglu_limit` 为 0/None 时 clamp 解析成 `+inf`（不 clamp）
  （`moe_kernels.py:800-810`）
- 目标 stage1 kernel 名 `flydsl_moe1_afp8_wfp4_bf16_t32x64x256_w4_gui_xcd4_kw4_fp8` 可用
  `get_flydsl_kernel_params()` 解析出全部 tile 参数
- 目标 stage2 kernel 名 `flydsl_moe2_layout_afp8_wfp4_bf16_t32x128x128_atomic_sbm32`
  → `BM=32, BN=128, BK=128, epilog="atomic", SBM=32`
- 阶段一**不需要 Mori SHMEM**，只需要 `torch.distributed`

---

## File Structure

| 文件 | 职责 |
|---|---|
| `aiter/ops/flydsl/kernels/mega_moe/tp_moe_stage1.py`（新建） | `TPMoEStage1Output` 结构体、`TPMoEStage1` 类、容量计算、all-gather、sorting/量化编排、stage1 调用 |
| `aiter/ops/flydsl/kernels/mega_moe/__init__.py`（改） | 追加两个 lazy export，现有条目一字不动 |
| `op_tests/multigpu_tests/test_tp_moe_stage1.py`（新建） | 8 rank torchrun 测试，`--case` 选择单个用例 |
| `op_tests/multigpu_tests/tp_moe_stage1_ref.py`（新建） | torch fp32 参考实现（量化/反量化 + clamp SwiGLU），与测试分离便于复用 |

放在 `mega_moe/` 目录下是因为阶段二融合版要 import 同目录的 `dispatch.py` / `gemm1.py`。

---

### Task 1: 模块骨架、输出结构体与构造校验

**Files:**
- Create: `aiter/ops/flydsl/kernels/mega_moe/tp_moe_stage1.py`
- Test: `op_tests/multigpu_tests/test_tp_moe_stage1.py`

这一步只做**单进程**能验的部分：结构体定义、参数校验、容量公式。不碰 GPU 集合通信。

- [ ] **Step 1: 写失败的测试**

创建 `op_tests/multigpu_tests/test_tp_moe_stage1.py`：

```python
"""TPMoEStage1 correctness tests. Run with:

    torchrun --standalone --nproc_per_node=8 \
        op_tests/multigpu_tests/test_tp_moe_stage1.py --case <name>

Single-rank cases (construct/capacity) also run as plain `python3 <file>`.
"""

import argparse
import os
import sys

import torch
import torch.distributed as dist

from aiter.ops.flydsl.kernels.mega_moe.tp_moe_stage1 import (
    TPMoEStage1,
    TPMoEStage1Output,
)

NETWORK = dict(
    model_dim=7168,
    experts=384,
    topk=6,
    swiglu_limit=10.0,
)
STAGE1_KERNEL = "flydsl_moe1_afp8_wfp4_bf16_t32x64x256_w4_gui_xcd4_kw4_fp8"


def _fake_w1(experts, inter_dim, model_dim, device):
    """Byte-shaped stand-in for a preshuffled MXFP4 W1 (values are irrelevant here)."""
    w1 = torch.zeros(
        (experts, 2 * inter_dim, model_dim // 2), dtype=torch.uint8, device=device
    )
    w1_scale = torch.full(
        (experts, 2 * inter_dim, model_dim // 32), 0x7F, dtype=torch.uint8, device=device
    )
    return w1, w1_scale


def case_construct_validates():
    device = torch.device("cuda", 0)
    inter_dim = 384
    w1, w1_scale = _fake_w1(NETWORK["experts"], inter_dim, NETWORK["model_dim"], device)

    # tp_size must be 4 or 8
    try:
        TPMoEStage1(
            model_dim=NETWORK["model_dim"],
            inter_dim=inter_dim,
            experts=NETWORK["experts"],
            topk=NETWORK["topk"],
            w1=w1,
            w1_scale=w1_scale,
            tp_size=2,
            tp_rank=0,
            device=device,
        )
    except ValueError as exc:
        assert "tp_size" in str(exc), exc
    else:
        raise AssertionError("tp_size=2 must be rejected")

    # sort_block_m must divide the stage1 tile_m
    try:
        TPMoEStage1(
            model_dim=NETWORK["model_dim"],
            inter_dim=inter_dim,
            experts=NETWORK["experts"],
            topk=NETWORK["topk"],
            w1=w1,
            w1_scale=w1_scale,
            tp_size=8,
            tp_rank=0,
            device=device,
            sort_block_m=48,
        )
    except ValueError as exc:
        assert "sort_block_m" in str(exc), exc
    else:
        raise AssertionError("sort_block_m=48 must be rejected")

    op = TPMoEStage1(
        model_dim=NETWORK["model_dim"],
        inter_dim=inter_dim,
        experts=NETWORK["experts"],
        topk=NETWORK["topk"],
        w1=w1,
        w1_scale=w1_scale,
        tp_size=8,
        tp_rank=0,
        device=device,
        swiglu_limit=NETWORK["swiglu_limit"],
        stage1_kernel_name=STAGE1_KERNEL,
    )
    assert op.tp_size == 8
    assert op.sort_block_m == 32
    assert op.stage1_params["tile_m"] == 32
    assert op.stage1_params["tile_n"] == 64
    assert op.stage1_params["tile_k"] == 256
    assert op.stage1_params["gate_mode"] == "interleave"
    print("case_construct_validates OK")


def case_capacity():
    device = torch.device("cuda", 0)
    inter_dim = 384
    w1, w1_scale = _fake_w1(NETWORK["experts"], inter_dim, NETWORK["model_dim"], device)
    op = TPMoEStage1(
        model_dim=NETWORK["model_dim"],
        inter_dim=inter_dim,
        experts=NETWORK["experts"],
        topk=NETWORK["topk"],
        w1=w1,
        w1_scale=w1_scale,
        tp_size=8,
        tp_rank=0,
        device=device,
        stage1_kernel_name=STAGE1_KERNEL,
    )
    # M_global = tp_size * m_local; max_sorted matches moe_sorting's own formula.
    assert op.m_logical_for(1) == 8
    assert op.m_logical_for(128) == 1024
    # 8*6 + 384*32 - 6
    assert op.max_sorted_for(1) == 8 * 6 + 384 * 32 - 6
    # 1024*6 + 384*32 - 6
    assert op.max_sorted_for(128) == 1024 * 6 + 384 * 32 - 6
    print("case_capacity OK")


CASES = {
    "construct": case_construct_validates,
    "capacity": case_capacity,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", default="construct")
    args = parser.parse_args()
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    CASES[args.case]()


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 跑测试确认失败**

Run: `python3 op_tests/multigpu_tests/test_tp_moe_stage1.py --case construct`
Expected: FAIL — `ModuleNotFoundError: No module named 'aiter.ops.flydsl.kernels.mega_moe.tp_moe_stage1'`

- [ ] **Step 3: 写最小实现**

创建 `aiter/ops/flydsl/kernels/mega_moe/tp_moe_stage1.py`：

```python
# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
"""Tensor-parallel MoE Stage1.

Each TP rank owns ALL experts and one 1/tp shard of the intermediate
dimension. The caller passes its own DP token shard; this operator
all-gathers across the TP group (DP group == TP group), runs grouping,
GEMM1, SwiGLU and per-1x32 FP8 output quantization, and returns the
six tensors the ordinary FlyDSL v2 FMoE GEMM2 consumes.
"""

from dataclasses import dataclass

import torch

from aiter.ops.flydsl.moe_kernels import get_flydsl_kernel_params

_SUPPORTED_TP = (4, 8)
_DEFAULT_STAGE1_KERNEL = "flydsl_moe1_afp8_wfp4_bf16_t32x64x256_w4_gui_xcd4_kw4_fp8"


@dataclass(frozen=True)
class TPMoEStage1Output:
    """Everything ordinary FlyDSL v2 FMoE GEMM2 needs, plus host metadata."""

    inter_sorted_quant: torch.Tensor
    inter_sorted_shuffled_scale: torch.Tensor
    sorted_token_ids: torch.Tensor
    sorted_weights: torch.Tensor
    sorted_expert_ids: torch.Tensor
    num_valid_ids: torch.Tensor

    m_logical: int
    max_sorted: int
    num_experts: int
    model_dim: int
    inter_dim: int
    topk: int
    sort_block_m: int


class TPMoEStage1:
    """Stateful TP4/TP8 MoE Stage1 operator.

    Preconditions (documented, not checked at runtime):
      * every rank in the group calls with the same ``m_local``
      * ``w1`` / ``w1_scale`` are already preshuffled for the a16w4
        gate/up-interleaved layout
      * the group used here is both the DP group and the TP group
    """

    def __init__(
        self,
        *,
        model_dim: int,
        inter_dim: int,
        experts: int,
        topk: int,
        w1: torch.Tensor,
        w1_scale: torch.Tensor,
        group=None,
        tp_size: int | None = None,
        tp_rank: int | None = None,
        device: torch.device | None = None,
        sort_block_m: int = 32,
        swiglu_limit: float = 0.0,
        stage1_kernel_name: str = _DEFAULT_STAGE1_KERNEL,
        transport: str = "allgather_bf16",
    ):
        self.group = group
        if tp_size is None or tp_rank is None:
            import torch.distributed as dist

            if not dist.is_initialized():
                raise ValueError(
                    "TPMoEStage1 needs an initialized process group, or explicit "
                    "tp_size/tp_rank"
                )
            tp_size = dist.get_world_size(group)
            tp_rank = dist.get_rank(group)
        if int(tp_size) not in _SUPPORTED_TP:
            raise ValueError(f"tp_size={tp_size} unsupported; expected one of {_SUPPORTED_TP}")

        params = get_flydsl_kernel_params(stage1_kernel_name)
        if params is None:
            raise ValueError(f"unknown stage1 kernel name: {stage1_kernel_name}")
        if int(sort_block_m) != int(params["tile_m"]):
            raise ValueError(
                f"sort_block_m={sort_block_m} must equal the stage1 kernel tile_m="
                f"{params['tile_m']} ({stage1_kernel_name})"
            )
        if inter_dim % int(params["tile_n"]) != 0:
            raise ValueError(
                f"inter_dim={inter_dim} must be divisible by tile_n={params['tile_n']}"
            )
        if float(swiglu_limit) < 0:
            raise ValueError("swiglu_limit must be non-negative")
        if transport != "allgather_bf16":
            raise NotImplementedError(
                f"transport={transport!r} is not implemented yet; phase 1 only "
                "supports 'allgather_bf16'"
            )

        self.tp_size = int(tp_size)
        self.tp_rank = int(tp_rank)
        self.model_dim = int(model_dim)
        self.inter_dim = int(inter_dim)
        self.experts = int(experts)
        self.topk = int(topk)
        self.sort_block_m = int(sort_block_m)
        self.swiglu_limit = float(swiglu_limit)
        self.stage1_kernel_name = stage1_kernel_name
        self.stage1_params = params
        self.transport = transport
        self.device = device or torch.device("cuda", torch.cuda.current_device())
        self.w1 = w1
        self.w1_scale = w1_scale

    def m_logical_for(self, m_local: int) -> int:
        return self.tp_size * int(m_local)

    def max_sorted_for(self, m_local: int) -> int:
        """Mirror of moe_sorting's max_num_tokens_padded."""
        return self.m_logical_for(m_local) * self.topk + self.experts * self.sort_block_m - self.topk
```

- [ ] **Step 4: 跑测试确认通过**

Run: `python3 op_tests/multigpu_tests/test_tp_moe_stage1.py --case construct`
Expected: `case_construct_validates OK`

Run: `python3 op_tests/multigpu_tests/test_tp_moe_stage1.py --case capacity`
Expected: `case_capacity OK`

- [ ] **Step 5: 提交**

```bash
git add aiter/ops/flydsl/kernels/mega_moe/tp_moe_stage1.py \
        op_tests/multigpu_tests/test_tp_moe_stage1.py
git commit -m "feat(tp-moe): add TPMoEStage1 skeleton with construction validation

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 2: all-gather 三样输入

**Files:**
- Modify: `aiter/ops/flydsl/kernels/mega_moe/tp_moe_stage1.py`
- Modify: `op_tests/multigpu_tests/test_tp_moe_stage1.py`

`activation` / `topk_ids` / `route_weights` 三样都要按**同一个 rank 顺序**收齐，否则 `moe_sorting` 出来的 token id 索引不到对应的 activation 行。rank-major：`global_token = src_rank * m_local + local_token`。

- [ ] **Step 1: 写失败的测试**

在 `test_tp_moe_stage1.py` 里，`CASES` 定义之前插入分布式初始化辅助和新用例：

```python
def _setup_dist():
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    if not dist.is_initialized():
        dist.init_process_group("cpu:gloo,cuda:nccl", device_id=device)
    return rank, world, device


def case_all_gather():
    rank, world, device = _setup_dist()
    assert world in (4, 8), f"run this case with 4 or 8 ranks, got {world}"
    m_local = 5
    inter_dim = 384
    w1, w1_scale = _fake_w1(NETWORK["experts"], inter_dim, NETWORK["model_dim"], device)
    op = TPMoEStage1(
        model_dim=NETWORK["model_dim"],
        inter_dim=inter_dim,
        experts=NETWORK["experts"],
        topk=NETWORK["topk"],
        w1=w1,
        w1_scale=w1_scale,
        device=device,
        stage1_kernel_name=STAGE1_KERNEL,
    )
    assert op.tp_size == world and op.tp_rank == rank

    # Rank-identifiable payloads so we can assert the concatenation order.
    x = torch.full(
        (m_local, NETWORK["model_dim"]), float(rank), dtype=torch.bfloat16, device=device
    )
    ids = torch.full(
        (m_local, NETWORK["topk"]), rank, dtype=torch.int32, device=device
    )
    wts = torch.full(
        (m_local, NETWORK["topk"]), float(rank), dtype=torch.float32, device=device
    )

    gx, gwts, gids = op._all_gather_inputs(x, wts, ids)

    m_global = world * m_local
    assert gx.shape == (m_global, NETWORK["model_dim"]), gx.shape
    assert gids.shape == (m_global, NETWORK["topk"]), gids.shape
    assert gwts.shape == (m_global, NETWORK["topk"]), gwts.shape
    assert gx.dtype == torch.bfloat16 and gids.dtype == torch.int32
    assert gwts.dtype == torch.float32
    assert gx.is_contiguous() and gids.is_contiguous() and gwts.is_contiguous()

    for src in range(world):
        lo, hi = src * m_local, (src + 1) * m_local
        assert torch.all(gx[lo:hi] == float(src)), f"activation block {src} misordered"
        assert torch.all(gids[lo:hi] == src), f"topk_ids block {src} misordered"
        assert torch.all(gwts[lo:hi] == float(src)), f"weights block {src} misordered"

    if rank == 0:
        print("case_all_gather OK")
    dist.barrier()
    dist.destroy_process_group()
```

并把 `CASES` 改成：

```python
CASES = {
    "construct": case_construct_validates,
    "capacity": case_capacity,
    "all_gather": case_all_gather,
}
```

- [ ] **Step 2: 跑测试确认失败**

Run:
```bash
torchrun --standalone --nproc_per_node=8 \
    op_tests/multigpu_tests/test_tp_moe_stage1.py --case all_gather
```
Expected: FAIL — `AttributeError: 'TPMoEStage1' object has no attribute '_all_gather_inputs'`

- [ ] **Step 3: 写最小实现**

在 `tp_moe_stage1.py` 顶部 import 区加上：

```python
import torch.distributed as dist
```

并把 `__init__` 里那句局部 `import torch.distributed as dist` 删掉（改用模块级 import）。然后在 `max_sorted` 方法之后追加：

```python
    def _all_gather_inputs(self, x, route_weights, topk_ids):
        """Gather the three per-rank inputs in rank-major order.

        Returns (x_g, weights_g, ids_g) laid out so that
        ``global_token = src_rank * m_local + local_token``.
        """
        if self.tp_size == 1:
            return x.contiguous(), route_weights.contiguous(), topk_ids.contiguous()

        def _gather(t):
            t = t.contiguous()
            out = torch.empty(
                (t.shape[0] * self.tp_size,) + tuple(t.shape[1:]),
                dtype=t.dtype,
                device=t.device,
            )
            dist.all_gather_into_tensor(out, t, group=self.group)
            return out

        return _gather(x), _gather(route_weights), _gather(topk_ids)
```

`all_gather_into_tensor` 在 dim 0 上拼接，恰好就是 rank-major，无需 `movedim`/`reshape`。

- [ ] **Step 4: 跑测试确认通过**

Run:
```bash
torchrun --standalone --nproc_per_node=8 \
    op_tests/multigpu_tests/test_tp_moe_stage1.py --case all_gather
```
Expected: `case_all_gather OK`（只有 rank 0 打印），进程退出码 0

- [ ] **Step 5: 提交**

```bash
git add aiter/ops/flydsl/kernels/mega_moe/tp_moe_stage1.py \
        op_tests/multigpu_tests/test_tp_moe_stage1.py
git commit -m "feat(tp-moe): all-gather activation/ids/weights in rank-major order

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 3: `forward` —— sorting、量化、GEMM1，产出 6 个 tensor

**Files:**
- Modify: `aiter/ops/flydsl/kernels/mega_moe/tp_moe_stage1.py`
- Modify: `op_tests/multigpu_tests/test_tp_moe_stage1.py`

这一步把 baseline 串通。链路：`all-gather(BF16)` → `moe_sorting` → `fused_dynamic_mxfp8_quant_moe_sort` → `flydsl_moe_stage1(v2_output_layout=True)`。

**注意 `expert_ids` 的语义**：TP 下每卡持有全部 expert，`moe_sorting` 直接输出的就是 0..383 的 id，**不需要减任何 rank offset**（决定 15 里"local expert id"在 TP 下恰好等于 global id）。

本任务只验形状/dtype/编码，数值放 Task 4。

- [ ] **Step 1: 写失败的测试**

在 `test_tp_moe_stage1.py` 里追加：

```python
def _random_routes(m, experts, topk, device, seed):
    g = torch.Generator(device="cpu").manual_seed(seed)
    ids = torch.stack(
        [torch.randperm(experts, generator=g)[:topk] for _ in range(m)]
    ).to(device=device, dtype=torch.int32)
    w = torch.rand((m, topk), generator=g).to(device=device, dtype=torch.float32)
    return ids, w / w.sum(dim=-1, keepdim=True)


def case_forward_contract():
    rank, world, device = _setup_dist()
    inter_dim = 384
    model_dim = NETWORK["model_dim"]
    experts, topk = NETWORK["experts"], NETWORK["topk"]
    w1, w1_scale = _fake_w1(experts, inter_dim, model_dim, device)
    op = TPMoEStage1(
        model_dim=model_dim,
        inter_dim=inter_dim,
        experts=experts,
        topk=topk,
        w1=w1,
        w1_scale=w1_scale,
        device=device,
        swiglu_limit=NETWORK["swiglu_limit"],
        stage1_kernel_name=STAGE1_KERNEL,
    )

    for m_local in (1, 2, 4, 8, 16, 32, 64, 128):
        x = torch.randn((m_local, model_dim), dtype=torch.bfloat16, device=device)
        ids, wts = _random_routes(m_local, experts, topk, device, seed=1000 + rank)
        out = op.forward(x, wts, ids)

        assert isinstance(out, TPMoEStage1Output)
        m_global = world * m_local
        assert out.m_logical == m_global
        assert out.sort_block_m == 32
        assert out.inter_dim == inter_dim

        # moe_sorting's capacity is NOT sort_block_m-aligned; the stage1 payload is.
        # out.max_sorted is the payload row count, matching what the production path
        # feeds GEMM2 (`max_sorted = inter_states.shape[0]`, aiter/fused_moe.py:2018).
        sorted_len = op.max_sorted_for(m_local)
        n = -(-sorted_len // 32) * 32
        assert out.max_sorted == n, (out.max_sorted, n)

        assert out.inter_sorted_quant.shape == (n, inter_dim), out.inter_sorted_quant.shape
        assert out.inter_sorted_quant.dtype == torch.float8_e4m3fn
        assert out.sorted_token_ids.shape == (sorted_len,)
        assert out.sorted_token_ids.dtype == torch.int32
        assert out.sorted_weights.shape == (sorted_len,)
        assert out.sorted_weights.dtype == torch.float32
        assert out.sorted_expert_ids.shape == (n // 32,), out.sorted_expert_ids.shape
        assert out.num_valid_ids.shape == (2,)
        assert out.num_valid_ids.dtype == torch.int32
        assert out.num_valid_ids.device.type == "cuda"

        pad_rows = (n + 255) // 256 * 256
        pad_cols = ((inter_dim // 32) + 7) // 8 * 8
        assert out.inter_sorted_shuffled_scale.shape == (pad_rows, pad_cols), (
            out.inter_sorted_shuffled_scale.shape
        )

        # moe_sorting only writes rows [0, num_valid_ids[0]). The tail of the allocated
        # tensor is uninitialized torch.empty memory that no kernel reads: stage1
        # iterates ceil(num_valid/SBM) blocks, GEMM2 iterates cumsum[0]/BM. Scope every
        # content check to [:nvalid] — asserting over the full tensor reads garbage.
        nvalid = int(out.num_valid_ids[0].item())
        assert 0 < nvalid <= sorted_len, (nvalid, sorted_len)
        assert nvalid % 32 == 0, nvalid

        # token-id encoding: low 24 bits index the gathered batch, high 8 bits the slot
        packed = out.sorted_token_ids[:nvalid]
        tok = packed & 0x00FFFFFF
        slot = (packed >> 24) & 0xFF
        valid = tok < m_global
        assert torch.all(slot[valid] < topk), "valid rows must carry a real top-k slot"
        assert torch.all(tok[~valid] == m_global), "padding sentinel must be M_logical"
        assert torch.all(slot[~valid] == topk), "padding sentinel slot must be topk"
        assert torch.all(out.sorted_weights[:nvalid][~valid] == 0.0), "padding weight must be 0"
        assert int(valid.sum().item()) == m_global * topk, (
            f"expected {m_global * topk} routes, found {int(valid.sum().item())}"
        )

        used = out.sorted_expert_ids[: nvalid // 32]
        assert torch.all((used >= 0) & (used < experts)), "expert ids out of range"

    # per-call allocation: two calls must not alias
    x = torch.randn((8, model_dim), dtype=torch.bfloat16, device=device)
    ids, wts = _random_routes(8, experts, topk, device, seed=7 + rank)
    a = op.forward(x, wts, ids)
    b = op.forward(x, wts, ids)
    assert a.inter_sorted_quant.data_ptr() != b.inter_sorted_quant.data_ptr()
    assert a.sorted_token_ids.data_ptr() != b.sorted_token_ids.data_ptr()

    if rank == 0:
        print("case_forward_contract OK")
    dist.barrier()
    dist.destroy_process_group()
```

`CASES` 追加 `"forward_contract": case_forward_contract,`。

- [ ] **Step 2: 跑测试确认失败**

Run:
```bash
torchrun --standalone --nproc_per_node=8 \
    op_tests/multigpu_tests/test_tp_moe_stage1.py --case forward_contract
```
Expected: FAIL — `AttributeError: 'TPMoEStage1' object has no attribute 'forward'`

- [ ] **Step 3: 写最小实现**

`tp_moe_stage1.py` 的 import 区补上：

```python
from aiter.fused_moe import moe_sorting
from aiter.ops.flydsl.moe_kernels import flydsl_moe_stage1, get_flydsl_kernel_params
from aiter.ops.quant import fused_dynamic_mxfp8_quant_moe_sort
```

（把原来那行单独的 `get_flydsl_kernel_params` import 合并掉。）

在类里追加：

```python
    def _validate_call(self, x, route_weights, topk_ids, x_dtype):
        if x.dtype != x_dtype or not x.is_contiguous():
            raise ValueError(f"x must be contiguous {x_dtype}")
        if route_weights.dtype != torch.float32 or not route_weights.is_contiguous():
            raise ValueError("route_weights must be contiguous float32")
        if topk_ids.dtype != torch.int32 or not topk_ids.is_contiguous():
            raise ValueError("topk_ids must be contiguous int32")
        m_local = int(x.shape[0])
        if m_local <= 0:
            raise ValueError("m_local must be positive")
        if route_weights.shape != (m_local, self.topk):
            raise ValueError(
                f"route_weights must be [{m_local}, {self.topk}], got {tuple(route_weights.shape)}"
            )
        if topk_ids.shape != (m_local, self.topk):
            raise ValueError(
                f"topk_ids must be [{m_local}, {self.topk}], got {tuple(topk_ids.shape)}"
            )
        return m_local

    def _sort(self, topk_ids_g, weights_g, m_global):
        sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, _ = moe_sorting(
            topk_ids_g,
            weights_g,
            self.experts,
            self.model_dim,
            torch.bfloat16,
            block_size=self.sort_block_m,
        )
        return sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids

    def _run_gemm1(self, a_fp8, a_scale_sorted, sorted_ids, sorted_expert_ids, num_valid_ids):
        p = self.stage1_params
        payload, scale = flydsl_moe_stage1(
            a_fp8,
            self.w1,
            sorted_ids,
            sorted_expert_ids,
            num_valid_ids,
            out=None,
            topk=self.topk,
            tile_m=int(p["tile_m"]),
            tile_n=int(p["tile_n"]),
            tile_k=int(p["tile_k"]),
            a_dtype=str(p["a_dtype"]),
            b_dtype=str(p["b_dtype"]),
            out_dtype=str(p["out_dtype"]),
            act="silu",
            w1_scale=self.w1_scale,
            a1_scale=a_scale_sorted,
            sorted_weights=None,
            waves_per_eu=int(p.get("waves_per_eu", 3)),
            b_nt=int(p.get("b_nt", 0)),
            gate_mode=str(p.get("gate_mode", "separated")),
            xcd_swizzle=int(p.get("xcd_swizzle", 0)),
            k_wave=int(p.get("k_wave", 1)),
            swiglu_limit=(self.swiglu_limit or None),
            v2_output_layout=True,
        )
        return payload, scale

    def _pack(self, payload, scale, sorted_ids, sorted_weights, sorted_expert_ids,
              num_valid_ids, m_global):
        return TPMoEStage1Output(
            inter_sorted_quant=payload.view(torch.float8_e4m3fn),
            inter_sorted_shuffled_scale=scale,
            sorted_token_ids=sorted_ids,
            sorted_weights=sorted_weights,
            sorted_expert_ids=sorted_expert_ids,
            num_valid_ids=num_valid_ids,
            m_logical=m_global,
            max_sorted=int(payload.shape[0]),
            num_experts=self.experts,
            model_dim=self.model_dim,
            inter_dim=self.inter_dim,
            topk=self.topk,
            sort_block_m=self.sort_block_m,
        )

    def forward(self, x_bf16, route_weights, topk_ids) -> TPMoEStage1Output:
        """BF16 entry. Gathers bf16, then quantizes after sorting."""
        m_local = self._validate_call(x_bf16, route_weights, topk_ids, torch.bfloat16)
        m_global = self.m_logical_for(m_local)

        x_g, wts_g, ids_g = self._all_gather_inputs(x_bf16, route_weights, topk_ids)
        sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids = self._sort(
            ids_g, wts_g, m_global
        )
        a_fp8, a_scale_sorted = fused_dynamic_mxfp8_quant_moe_sort(
            x_g,
            sorted_ids=sorted_ids,
            num_valid_ids=num_valid_ids,
            token_num=m_global,
            topk=self.topk,
            block_size=self.sort_block_m,
            sorted_weights=None,
        )
        payload, scale = self._run_gemm1(
            a_fp8, a_scale_sorted, sorted_ids, sorted_expert_ids, num_valid_ids
        )
        return self._pack(
            payload, scale, sorted_ids, sorted_weights, sorted_expert_ids,
            num_valid_ids, m_global,
        )

    __call__ = forward
    forward_bf16 = forward
```

- [ ] **Step 4: 跑测试确认通过**

Run:
```bash
torchrun --standalone --nproc_per_node=8 \
    op_tests/multigpu_tests/test_tp_moe_stage1.py --case forward_contract
```
Expected: `case_forward_contract OK`

若首次跑报 FlyDSL 编译错误，把 `AITER_LOG_MORE=1` 打开重跑看 kernel 名，确认 `tile_n` 没被静默降级（`inter_dim=384` + `tile_n=64` 应当原样保留）。

- [ ] **Step 5: 提交**

```bash
git add aiter/ops/flydsl/kernels/mega_moe/tp_moe_stage1.py \
        op_tests/multigpu_tests/test_tp_moe_stage1.py
git commit -m "feat(tp-moe): baseline forward producing the v2 FMoE GEMM2 ABI

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 4: torch fp32 逐行数值参考

**Files:**
- Create: `op_tests/multigpu_tests/tp_moe_stage1_ref.py`
- Modify: `op_tests/multigpu_tests/test_tp_moe_stage1.py`

参考实现要**同时模拟 FP8 activation 量化和 MXFP4 权重**，否则容差只能放到 0.1 那种没有诊断力的水平（`test_mega_moe_v2.py` 就是因为漏了 activation 量化才用 `--rtol 0.10`）。

读回 `inter_sorted_shuffled_scale` 要用 canonical 的 shuffle 公式——这个公式已经验证过与 `csrc/include/mx_quant_utils.h:212-217` 和 kernel 侧逐位一致。

- [ ] **Step 1: 写失败的测试**

在 `test_tp_moe_stage1.py` 追加：

```python
from tp_moe_stage1_ref import (
    build_mxfp4_w1,
    dequant_w1_expert,
    per_1x32_fp8_quant_dequant,
    read_shuffled_scale,
    reference_inter_row,
)


def case_numerics():
    rank, world, device = _setup_dist()
    inter_dim = 384
    model_dim = NETWORK["model_dim"]
    experts, topk = NETWORK["experts"], NETWORK["topk"]
    limit = NETWORK["swiglu_limit"]

    w1_ref, w1_scale_ref, w1_shuf, w1_scale_shuf = build_mxfp4_w1(
        experts, inter_dim, model_dim, device, seed=2026
    )
    op = TPMoEStage1(
        model_dim=model_dim,
        inter_dim=inter_dim,
        experts=experts,
        topk=topk,
        w1=w1_shuf,
        w1_scale=w1_scale_shuf,
        device=device,
        swiglu_limit=limit,
        stage1_kernel_name=STAGE1_KERNEL,
    )

    worst = 0.0
    for m_local in (1, 4, 16, 64, 128):
        x = torch.randn(
            (m_local, model_dim), dtype=torch.bfloat16, device=device
        ) * (model_dim ** -0.25)
        ids, wts = _random_routes(m_local, experts, topk, device, seed=31 + rank)
        out = op.forward(x, wts, ids)
        torch.cuda.synchronize()

        # Rebuild the gathered inputs on the host side for the reference.
        x_g, wts_g, ids_g = op._all_gather_inputs(x, wts, ids)
        x_g_f32 = x_g.float()
        x_deq = per_1x32_fp8_quant_dequant(x_g_f32)

        nvalid = int(out.num_valid_ids[0].item())
        packed = out.sorted_token_ids[:nvalid]
        tok = (packed & 0x00FFFFFF).long()
        slot = ((packed >> 24) & 0xFF).long()
        valid = (tok < out.m_logical).nonzero(as_tuple=True)[0]

        scale_cols = inter_dim // 32
        got_scale = read_shuffled_scale(
            out.inter_sorted_shuffled_scale, nvalid, scale_cols
        )
        got = out.inter_sorted_quant[:nvalid].float() * got_scale.repeat_interleave(
            32, dim=-1
        )

        num, den = 0.0, 0.0
        # Rows are Expert-grouped, so a single-entry cache turns O(rows) weight
        # dequantizations into O(active experts). Without it this loop re-expands a
        # [2*inter, model_dim] tensor once per row and takes minutes.
        cur_e, cur_w1_deq = -1, None
        for r in valid.tolist():
            e = int(out.sorted_expert_ids[r // out.sort_block_m].item())
            t = int(tok[r].item())
            assert int(ids_g[t, int(slot[r].item())].item()) == e, (
                f"row {r}: sorted_expert_ids says {e} but topk_ids says "
                f"{int(ids_g[t, int(slot[r].item())].item())}"
            )
            assert abs(
                float(out.sorted_weights[r].item())
                - float(wts_g[t, int(slot[r].item())].item())
            ) < 1e-6, f"row {r}: route weight mismatch"

            if e != cur_e:
                cur_e, cur_w1_deq = e, dequant_w1_expert(w1_ref, w1_scale_ref, e, inter_dim)
            ref = reference_inter_row(x_deq[t], cur_w1_deq, limit)
            ref_q = per_1x32_fp8_quant_dequant(ref.unsqueeze(0)).squeeze(0)
            num += float(((got[r] - ref_q) ** 2).sum())
            den += float((ref_q**2).sum())

        rel_l2 = (num / max(den, 1e-30)) ** 0.5
        worst = max(worst, rel_l2)
        if rank == 0:
            print(f"m_local={m_local:4d} rows={len(valid):6d} rel_l2={rel_l2:.5f}")

    t = torch.tensor([worst], device=device)
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    worst = float(t.item())
    if rank == 0:
        print(f"case_numerics worst rel_l2={worst:.5f}")
    if worst >= 0.05:
        raise AssertionError(f"rel_l2={worst:.5f} exceeds 0.05")
    if rank == 0:
        print("case_numerics OK")
    dist.barrier()
    dist.destroy_process_group()
```

`CASES` 追加 `"numerics": case_numerics,`。文件顶部加 `sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))` 以便 import 同目录的 ref 模块。

- [ ] **Step 2: 跑测试确认失败**

Run:
```bash
torchrun --standalone --nproc_per_node=8 \
    op_tests/multigpu_tests/test_tp_moe_stage1.py --case numerics
```
Expected: FAIL — `ModuleNotFoundError: No module named 'tp_moe_stage1_ref'`

- [ ] **Step 3: 写参考实现**

创建 `op_tests/multigpu_tests/tp_moe_stage1_ref.py`：

```python
"""Torch fp32 reference for TPMoEStage1.

Models BOTH the per-1x32 FP8 activation quantization and the MXFP4 weights,
so the residual against the kernel is dominated by MFMA accumulation order
rather than by an unmodelled quantization step.
"""

import torch
import torch.nn.functional as F

from aiter import dtypes
from aiter.ops.shuffle import shuffle_scale_a16w4, shuffle_weight_a16w4
from aiter.utility.fp4_utils import (
    MxDtypeInt,
    e8m0_to_f32,
    f32_to_mx_e8m0_scale,
    mxfp4_to_f32,
)


def per_1x32_fp8_quant_dequant(x: torch.Tensor) -> torch.Tensor:
    """Per-1x32 MX FP8 E4M3 quantize-then-dequantize, in fp32."""
    x = x.float()
    grouped = x.view(*x.shape[:-1], -1, 32)
    amax = grouped.abs().amax(dim=-1)
    e8m0 = f32_to_mx_e8m0_scale(amax, dtype=MxDtypeInt.FP8_E4M3)
    scale = e8m0_to_f32(e8m0).unsqueeze(-1)
    q = (grouped / scale).clamp(-448.0, 448.0).to(dtypes.fp8)
    return (q.float() * scale).view_as(x)


def build_mxfp4_w1(experts, inter_dim, model_dim, device, seed):
    """Return (w1_ref, w1_scale_ref, w1_shuffled, w1_scale_shuffled).

    ``w1_ref`` keeps the UNSHUFFLED [E, 2*inter, model_dim/2] fp4x2 layout for
    the reference; the shuffled pair is what the kernel consumes.
    """
    import aiter

    g = torch.Generator(device="cpu").manual_seed(seed)
    w1_bf16 = (
        torch.randn((experts, 2 * inter_dim, model_dim), generator=g).to(
            device=device, dtype=dtypes.bf16
        )
        * (model_dim**-0.25)
    )
    quant = aiter.get_torch_quant(aiter.QuantType.per_1x32)
    w1_q, w1_scale = quant(w1_bf16, quant_dtype=dtypes.fp4x2)
    w1_shuf = shuffle_weight_a16w4(w1_q, 16, True)
    w1_scale_shuf = shuffle_scale_a16w4(w1_scale, experts, True)
    return w1_q, w1_scale, w1_shuf, w1_scale_shuf


def dequant_w1_expert(w1_q, w1_scale, expert_id, inter_dim):
    """Dequantize one expert's UNSHUFFLED W1 into fp32 [2*inter_dim, model_dim]."""
    experts, rows, _packed_cols = w1_q.shape
    assert rows == 2 * inter_dim, (rows, inter_dim)
    # ``per_1x32_f4_quant`` flattens the leading dims before reducing, so the
    # scale comes back 2D as (E * 2*inter_dim, model_dim//32) even for a 3D
    # weight. Restore the expert axis before indexing: ``w1_scale[expert_id]``
    # on the 2D form yields a single (model_dim//32,) row that broadcasts
    # silently over all 2*inter_dim rows -- shape-correct and numerically wrong.
    scale_e = w1_scale.reshape(experts, rows, -1)[expert_id]
    w = mxfp4_to_f32(w1_q[expert_id])
    s = e8m0_to_f32(scale_e).repeat_interleave(32, dim=-1)
    return (w * s).float()


def reference_inter_row(x_row_f32, w1_deq, swiglu_limit):
    """One route's GEMM1 + clamp + SwiGLU, fp32. Returns [inter_dim]."""
    inter_dim = w1_deq.shape[0] // 2
    gate_up = w1_deq @ x_row_f32
    gate = gate_up[:inter_dim]
    up = gate_up[inter_dim:]
    if swiglu_limit and swiglu_limit > 0:
        gate = gate.clamp(max=swiglu_limit)
        up = up.clamp(min=-swiglu_limit, max=swiglu_limit)
    return F.silu(gate) * up


def mx_scale_shuffle_idx(scale_n_pad: int, x: int, y: int) -> int:
    """Canonical MX scale shuffle address (csrc/include/mx_quant_utils.h:212-217)."""
    return (
        (x // 32 * scale_n_pad) * 32
        + (y // 8) * 256
        + (y % 4) * 64
        + (x % 16) * 4
        + (y % 8) // 4 * 2
        + (x % 32) // 16
    )


def read_shuffled_scale(scale_tensor, n_rows: int, n_kgroups: int) -> torch.Tensor:
    """Un-shuffle the stage1 output scale into a plain [n_rows, n_kgroups] fp32."""
    flat = scale_tensor.reshape(-1).view(torch.uint8)
    scale_n_pad = int(scale_tensor.shape[-1])
    xs = torch.arange(n_rows, dtype=torch.int64).view(-1, 1)
    ys = torch.arange(n_kgroups, dtype=torch.int64).view(1, -1)
    idx = (
        (xs // 32 * scale_n_pad) * 32
        + (ys // 8) * 256
        + (ys % 4) * 64
        + (xs % 16) * 4
        + (ys % 8) // 4 * 2
        + (xs % 32) // 16
    ).to(flat.device)
    return e8m0_to_f32(flat[idx.reshape(-1)]).view(n_rows, n_kgroups)
```

- [ ] **Step 4: 跑测试确认通过**

Run:
```bash
torchrun --standalone --nproc_per_node=8 \
    op_tests/multigpu_tests/test_tp_moe_stage1.py --case numerics
```
Expected: 每个 `m_local` 打印一行 `rel_l2`，最后 `case_numerics OK`，`worst rel_l2 < 0.05`

**如果超阈值，按这个顺序排查，不要直接放宽阈值：**
1. `gate_mode` —— 目标 kernel 是 `_gui`（gate/up interleaved）。确认参考用的是**未 shuffle** 的 `w1_ref` 且按 `[:inter]` / `[inter:]` 切两半；interleave 是 `shuffle_weight_a16w4(..., True)` 做的，不该出现在参考里
2. `swiglu_limit` —— 参考的 clamp 不对称：gate 只截上界，up 双边截
3. scale 读回 —— 用 `mx_scale_shuffle_idx` 对单个 `(row, kgroup)` 手算一次，和 `read_shuffled_scale` 的结果比对
4. padding 行 —— 确认只统计了 `tok < m_logical` 的行

- [ ] **Step 5: 提交**

```bash
git add op_tests/multigpu_tests/tp_moe_stage1_ref.py \
        op_tests/multigpu_tests/test_tp_moe_stage1.py
git commit -m "test(tp-moe): fp32 row-wise reference with fp8 activation quant modelled

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 5: `forward_prequant` —— 先 quant 再 gather

**Files:**
- Modify: `aiter/ops/flydsl/kernels/mega_moe/tp_moe_stage1.py`
- Modify: `op_tests/multigpu_tests/test_tp_moe_stage1.py`

`forward` 走的是 gather BF16（决定 18 的 baseline 路径）。`forward_prequant` 收的已经是 FP8，于是它天然就是**先 quant 再 gather**——每行跨卡 `7168 + 224 = 7392` 字节，而 BF16 是 `14336` 字节。所以阶段一就能拿到带宽对比，不用等融合版。

FP8 路径不能用 `fused_dynamic_mxfp8_quant_moe_sort`（那个会重新量化），要用**只排 scale** 的 `moe_mxfp4_sort`。

已验证的签名：
- `aiter.utility.fp4_utils.moe_mxfp4_sort(blockscale_e8m0, sorted_ids, num_valid_ids, token_num, block_size=32) -> Tensor`
- `aiter.ops.flydsl.kernels.mega_moe.quant.per_1x32_mx_quant(x, quant_mode='fp4', stream=None)`

- [ ] **Step 1: 写失败的测试**

`test_tp_moe_stage1.py` 追加：

```python
def case_prequant_equivalence():
    rank, world, device = _setup_dist()
    inter_dim = 384
    model_dim = NETWORK["model_dim"]
    experts, topk = NETWORK["experts"], NETWORK["topk"]
    w1_ref, w1_scale_ref, w1_shuf, w1_scale_shuf = build_mxfp4_w1(
        experts, inter_dim, model_dim, device, seed=2026
    )
    op = TPMoEStage1(
        model_dim=model_dim,
        inter_dim=inter_dim,
        experts=experts,
        topk=topk,
        w1=w1_shuf,
        w1_scale=w1_scale_shuf,
        device=device,
        swiglu_limit=NETWORK["swiglu_limit"],
        stage1_kernel_name=STAGE1_KERNEL,
    )

    worst = 0.0
    for m_local in (1, 8, 64, 128):
        x = torch.randn(
            (m_local, model_dim), dtype=torch.bfloat16, device=device
        ) * (model_dim**-0.25)
        ids, wts = _random_routes(m_local, experts, topk, device, seed=77 + rank)

        a = op.forward(x, wts, ids)
        x_q, x_scale = op.quantize(x)
        assert x_q.dtype == torch.float8_e4m3fn, x_q.dtype
        assert x_q.shape == (m_local, model_dim), x_q.shape
        b = op.forward_prequant(x_q, x_scale, wts, ids)
        torch.cuda.synchronize()

        # Routing metadata does not depend on quantization at all -> must match exactly.
        assert torch.equal(a.sorted_token_ids, b.sorted_token_ids)
        assert torch.equal(a.sorted_expert_ids, b.sorted_expert_ids)
        assert torch.equal(a.num_valid_ids, b.num_valid_ids)
        assert torch.equal(a.sorted_weights, b.sorted_weights)
        assert a.m_logical == b.m_logical and a.max_sorted == b.max_sorted

        nvalid = int(a.num_valid_ids[0].item())
        cols = inter_dim // 32
        va = a.inter_sorted_quant[:nvalid].float() * read_shuffled_scale(
            a.inter_sorted_shuffled_scale, nvalid, cols
        ).repeat_interleave(32, dim=-1)
        vb = b.inter_sorted_quant[:nvalid].float() * read_shuffled_scale(
            b.inter_sorted_shuffled_scale, nvalid, cols
        ).repeat_interleave(32, dim=-1)
        rel = float(((va - vb) ** 2).sum() ** 0.5 / max(float((va**2).sum() ** 0.5), 1e-30))
        worst = max(worst, rel)
        if rank == 0:
            print(f"m_local={m_local:4d} forward-vs-prequant rel_l2={rel:.6f}")

    t = torch.tensor([worst], device=device)
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    worst = float(t.item())
    if worst >= 0.02:
        raise AssertionError(f"forward vs forward_prequant rel_l2={worst:.6f} exceeds 0.02")
    if rank == 0:
        print(f"case_prequant_equivalence OK (worst rel_l2={worst:.6f})")
    dist.barrier()
    dist.destroy_process_group()
```

`CASES` 追加 `"prequant": case_prequant_equivalence,`。

- [ ] **Step 2: 跑测试确认失败**

Run:
```bash
torchrun --standalone --nproc_per_node=8 \
    op_tests/multigpu_tests/test_tp_moe_stage1.py --case prequant
```
Expected: FAIL — `AttributeError: 'TPMoEStage1' object has no attribute 'quantize'`

- [ ] **Step 3: 写实现**

`tp_moe_stage1.py` import 区追加：

```python
from aiter.ops.flydsl.kernels.mega_moe.quant import per_1x32_mx_quant
from aiter.utility.fp4_utils import moe_mxfp4_sort
```

类里追加：

```python
    def quantize(self, x_bf16):
        """Local per-1x32 BF16 -> FP8 E4M3 + E8M0. Same routine MegaMoEV2 uses."""
        return per_1x32_mx_quant(x_bf16, quant_mode="fp8")

    def forward_prequant(
        self, x_fp8, x_scale, route_weights, topk_ids
    ) -> TPMoEStage1Output:
        """Prequantized entry.

        Gathers FP8 payload + E8M0 scale instead of BF16, i.e. quantize-then-gather.
        Per row this moves ``model_dim + model_dim/32`` bytes instead of
        ``model_dim * 2``.
        """
        m_local = self._validate_call(
            x_fp8, route_weights, topk_ids, torch.float8_e4m3fn
        )
        if not x_scale.is_contiguous():
            raise ValueError("x_scale must be contiguous")
        if x_scale.shape != (m_local, self.model_dim // 32):
            raise ValueError(
                f"x_scale must be [{m_local}, {self.model_dim // 32}], "
                f"got {tuple(x_scale.shape)}"
            )
        m_global = self.m_logical_for(m_local)

        x_g, wts_g, ids_g = self._all_gather_inputs(x_fp8, route_weights, topk_ids)
        scale_g = self._all_gather_one(x_scale)
        sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids = self._sort(
            ids_g, wts_g, m_global
        )
        a_scale_sorted = moe_mxfp4_sort(
            scale_g.view(m_global, 1, -1),
            sorted_ids,
            num_valid_ids,
            m_global,
            self.sort_block_m,
        )
        payload, scale = self._run_gemm1(
            x_g, a_scale_sorted, sorted_ids, sorted_expert_ids, num_valid_ids
        )
        return self._pack(
            payload, scale, sorted_ids, sorted_weights, sorted_expert_ids,
            num_valid_ids, m_global,
        )
```

并把 `_all_gather_inputs` 里的内部闭包提成一个可复用的方法（`forward_prequant` 要单独 gather scale）：

```python
    def _all_gather_one(self, t):
        t = t.contiguous()
        if self.tp_size == 1:
            return t
        out = torch.empty(
            (t.shape[0] * self.tp_size,) + tuple(t.shape[1:]),
            dtype=t.dtype,
            device=t.device,
        )
        dist.all_gather_into_tensor(out, t, group=self.group)
        return out

    def _all_gather_inputs(self, x, route_weights, topk_ids):
        """Gather the three per-rank inputs in rank-major order.

        Returns (x_g, weights_g, ids_g) laid out so that
        ``global_token = src_rank * m_local + local_token``.
        """
        return (
            self._all_gather_one(x),
            self._all_gather_one(route_weights),
            self._all_gather_one(topk_ids),
        )
```

（替换 Task 2 里那份带内部闭包的实现。）

- [ ] **Step 4: 跑测试确认通过**

Run:
```bash
torchrun --standalone --nproc_per_node=8 \
    op_tests/multigpu_tests/test_tp_moe_stage1.py --case prequant
```
Expected: 每个 `m_local` 打印 `rel_l2`，最后 `case_prequant_equivalence OK`

再跑一次 Task 4 确认没回归：
```bash
torchrun --standalone --nproc_per_node=8 \
    op_tests/multigpu_tests/test_tp_moe_stage1.py --case numerics
```
Expected: `case_numerics OK`

**若 `rel_l2` 明显大于 0.02**：说明 `per_1x32_mx_quant(quant_mode="fp8")` 和 `fused_dynamic_mxfp8_quant_moe_sort` 的舍入不一致。先分别 dump 两者对同一个 tensor 的 e8m0 scale 比对，再决定是统一量化函数还是接受差异——**不要直接放宽阈值**。

- [ ] **Step 5: 提交**

```bash
git add aiter/ops/flydsl/kernels/mega_moe/tp_moe_stage1.py \
        op_tests/multigpu_tests/test_tp_moe_stage1.py
git commit -m "feat(tp-moe): add forward_prequant that gathers fp8 instead of bf16

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 6: 端到端 —— 接 v2 GEMM2 + reduce-scatter

**Files:**
- Modify: `op_tests/multigpu_tests/tp_moe_stage1_ref.py`
- Modify: `op_tests/multigpu_tests/test_tp_moe_stage1.py`

这一步只写**测试**，不改算子——决定 14 说了下游由调用方接。目的是证明输出 ABI 真的能被目标 kernel 吃下，并且整条 TP 链的数值是对的。

链路：`TPMoEStage1` → `mxfp4_moe_gemm2(epilog="atomic")` → `reduce_scatter_tensor` → 和「用全局未切分权重算的 torch fp32 全量 MoE」比。

目标 kernel `flydsl_moe2_layout_afp8_wfp4_bf16_t32x128x128_atomic_sbm32` 解出来是 `BM=32, BN=128, BK=128, epilog="atomic", SBM=32, use_nt=False`。约束都满足：`7168 % 128 == 0`、`384 % 128 == 0`。

- [ ] **Step 1: 写失败的测试**

先在 `tp_moe_stage1_ref.py` 追加全局权重的构造与参考：

```python
def build_global_weights(experts, inter_global, model_dim, device, seed):
    """Build UNSHARDED bf16 W1/W2, quantize to MXFP4, return refs + quantized.

    W1 logical layout is [E, 2*inter_global, model_dim] with the gate half
    first and the up half second.
    """
    import aiter

    g = torch.Generator(device="cpu").manual_seed(seed)
    w1 = (
        torch.randn((experts, 2 * inter_global, model_dim), generator=g).to(
            device=device, dtype=dtypes.bf16
        )
        * (model_dim**-0.25)
    )
    w2 = (
        torch.randn((experts, model_dim, inter_global), generator=g).to(
            device=device, dtype=dtypes.bf16
        )
        * (inter_global**-0.25)
    )
    quant = aiter.get_torch_quant(aiter.QuantType.per_1x32)
    w1_q, w1_s = quant(w1, quant_dtype=dtypes.fp4x2)
    w2_q, w2_s = quant(w2, quant_dtype=dtypes.fp4x2)
    return w1_q, w1_s, w2_q, w2_s


def shard_w1(w1_q, w1_s, tp_rank, tp_size, inter_global):
    """Take this rank's [start, start+I_rank) window out of BOTH halves.

    W1 is column-parallel: TP shards the N axis (2*inter_global).

    MEASURED: get_torch_quant(per_1x32) flattens leading dims, so w1_s arrives
    2D as (E * 2*inter_global, model_dim//32). Reshape to 3D to slice the N
    axis, then flatten back -- shuffle_scale_a16w4 asserts a 2D input.
    """
    experts = w1_q.shape[0]
    i_rank = inter_global // tp_size
    lo = tp_rank * i_rank

    def _sl(t):
        return torch.cat(
            (
                t[:, lo : lo + i_rank],
                t[:, inter_global + lo : inter_global + lo + i_rank],
            ),
            dim=1,
        ).contiguous()

    q = _sl(w1_q)
    s = _sl(w1_s.reshape(experts, 2 * inter_global, -1))
    return q, s.reshape(experts * 2 * i_rank, -1).contiguous()


def shard_w2(w2_q, w2_s, tp_rank, tp_size, inter_global, model_dim):
    """W2 is row-parallel: TP shards the contraction axis (inter_global).

    The fp4x2 payload packs two values per byte so its last dim is halved; the
    scale's last dim is inter/32. MEASURED: w2_s arrives 2D as
    (E * model_dim, inter_global//32).
    """
    experts = w2_q.shape[0]
    i_rank = inter_global // tp_size
    lo = tp_rank * i_rank
    q = w2_q[:, :, lo // 2 : (lo + i_rank) // 2].contiguous()
    s = w2_s.reshape(experts, model_dim, -1)[:, :, lo // 32 : (lo + i_rank) // 32]
    return q, s.reshape(experts * model_dim, -1).contiguous()


def reference_full_moe(x_g_bf16, ids_g, wts_g, w1_q, w1_s, w2_q, w2_s, swiglu_limit):
    """Full unsharded MoE in fp32, modelling both activation quantizations."""
    m, model_dim = x_g_bf16.shape
    inter_global = w1_q.shape[1] // 2
    x_deq = per_1x32_fp8_quant_dequant(x_g_bf16.float())
    # w2_s is 2D (E * model_dim, inter_global//32); restore the expert axis once.
    w2_s3 = w2_s.reshape(w2_q.shape[0], model_dim, -1)
    out = torch.zeros((m, model_dim), dtype=torch.float32, device=x_deq.device)
    for e in torch.unique(ids_g).tolist():
        rows, slots = (ids_g == e).nonzero(as_tuple=True)
        if rows.numel() == 0:
            continue
        w1_deq = dequant_w1_expert(w1_q, w1_s, e, inter_global)
        w2_deq = (
            mxfp4_to_f32(w2_q[e])
            * e8m0_to_f32(w2_s3[e]).repeat_interleave(32, dim=-1)
        ).float()
        for r, s in zip(rows.tolist(), slots.tolist()):
            inter = reference_inter_row(x_deq[r], w1_deq, swiglu_limit)
            inter = per_1x32_fp8_quant_dequant(inter.unsqueeze(0)).squeeze(0)
            out[r] += (w2_deq @ inter) * float(wts_g[r, s].item())
    return out
```

再在 `test_tp_moe_stage1.py` 追加端到端用例：

```python
from aiter.ops.flydsl.kernels.mxmoe_dispatcher import mxfp4_moe_gemm2
from aiter.ops.shuffle import shuffle_scale_a16w4, shuffle_weight_a16w4
from tp_moe_stage1_ref import (
    build_global_weights,
    reference_full_moe,
    shard_w1,
    shard_w2,
)

GEMM2_BM, GEMM2_BN, GEMM2_BK = 32, 128, 128


def case_end_to_end():
    rank, world, device = _setup_dist()
    model_dim = NETWORK["model_dim"]
    experts, topk = NETWORK["experts"], NETWORK["topk"]
    limit = NETWORK["swiglu_limit"]
    inter_global = 384 * world          # TP8 -> 3072, TP4 -> 1536
    inter_dim = inter_global // world   # this rank's shard == 384

    w1_q, w1_s, w2_q, w2_s = build_global_weights(
        experts, inter_global, model_dim, device, seed=4096
    )
    w1_loc, w1_s_loc = shard_w1(w1_q, w1_s, rank, world, inter_global)
    w2_loc, w2_s_loc = shard_w2(w2_q, w2_s, rank, world, inter_global, model_dim)

    op = TPMoEStage1(
        model_dim=model_dim,
        inter_dim=inter_dim,
        experts=experts,
        topk=topk,
        w1=shuffle_weight_a16w4(w1_loc, 16, True),
        w1_scale=shuffle_scale_a16w4(w1_s_loc, experts, True),
        device=device,
        swiglu_limit=limit,
        stage1_kernel_name=STAGE1_KERNEL,
    )
    w2_u8 = shuffle_weight_a16w4(w2_loc, 16, False).view(torch.uint8)
    w2_scale_u8 = shuffle_scale_a16w4(w2_s_loc, experts, False).view(torch.uint8)

    for m_local in (1, 8, 64, 128):
        x = torch.randn(
            (m_local, model_dim), dtype=torch.bfloat16, device=device
        ) * (model_dim**-0.25)
        ids, wts = _random_routes(m_local, experts, topk, device, seed=555 + rank)
        s1 = op.forward(x, wts, ids)

        # epilog="atomic" accumulates, so the buffer must start at zero.
        partial = torch.zeros(
            (s1.m_logical, model_dim), dtype=torch.bfloat16, device=device
        )
        mxfp4_moe_gemm2(
            inter_sorted_quant=s1.inter_sorted_quant,
            inter_sorted_shuffled_scale=s1.inter_sorted_shuffled_scale,
            w2_u8=w2_u8,
            w2_scale_u8=w2_scale_u8,
            sorted_expert_ids=s1.sorted_expert_ids,
            cumsum_tensor=s1.num_valid_ids,
            sorted_token_ids=s1.sorted_token_ids,
            sorted_weights=s1.sorted_weights,
            out=partial,
            M_logical=s1.m_logical,
            max_sorted=s1.max_sorted,
            NE=experts,
            D_HIDDEN=model_dim,
            D_INTER=inter_dim,
            topk=topk,
            BM=GEMM2_BM,
            BN=GEMM2_BN,
            BK=GEMM2_BK,
            a_dtype="fp8",
            b_dtype="fp4",
            epilog="atomic",
            SBM=s1.sort_block_m,
            out_dtype="bf16",
        )
        torch.cuda.synchronize()

        # Stage1 leaves the output scale of padding rows uninitialized (measured in
        # Task 5). GEMM2 gates the store on `token_id < i32_M`
        # (mxmoe_gemm_v2.py:1006-1008) so that garbage must never reach `out` -- but
        # a NaN leaking through the shared CShuffle LDS would be silent, so check.
        assert torch.isfinite(partial).all(), "GEMM2 output contains non-finite values"

        # TP partials -> sum across ranks, keep only this rank's DP shard.
        got = torch.empty((m_local, model_dim), dtype=torch.float32, device=device)
        dist.reduce_scatter_tensor(got, partial.float().contiguous(), group=op.group)

        x_g, wts_g, ids_g = op._all_gather_inputs(x, wts, ids)
        ref_full = reference_full_moe(
            x_g, ids_g, wts_g, w1_q, w1_s, w2_q, w2_s, limit
        )
        ref = ref_full[rank * m_local : (rank + 1) * m_local]
        rel = float(
            ((got - ref) ** 2).sum() ** 0.5 / max(float((ref**2).sum() ** 0.5), 1e-30)
        )
        t = torch.tensor([rel], device=device)
        dist.all_reduce(t, op=dist.ReduceOp.MAX)
        rel = float(t.item())
        if rank == 0:
            print(f"m_local={m_local:4d} end-to-end rel_l2={rel:.5f}")
        if rel >= 0.05:
            raise AssertionError(f"m_local={m_local} end-to-end rel_l2={rel:.5f} >= 0.05")

    if rank == 0:
        print("case_end_to_end OK")
    dist.barrier()
    dist.destroy_process_group()
```

`CASES` 追加 `"e2e": case_end_to_end,`。

- [ ] **Step 2: 跑测试确认失败**

Run:
```bash
torchrun --standalone --nproc_per_node=8 \
    op_tests/multigpu_tests/test_tp_moe_stage1.py --case e2e
```
Expected: FAIL — `ImportError: cannot import name 'build_global_weights' from 'tp_moe_stage1_ref'`

- [ ] **Step 3: 按 Step 1 补齐 ref 模块**

把 Step 1 里 `tp_moe_stage1_ref.py` 的四个新函数写进去。注意 `shard_w2` 里 fp4x2 payload 的最后一维是**半字节打包**的，所以切片下标要除 2；scale 的最后一维是 `inter/32`，下标除 32。

- [ ] **Step 4: 跑测试确认通过**

Run:
```bash
torchrun --standalone --nproc_per_node=8 \
    op_tests/multigpu_tests/test_tp_moe_stage1.py --case e2e
```
Expected: 每个 `m_local` 打印 `end-to-end rel_l2`，最后 `case_end_to_end OK`

**排查顺序（超阈值时）：**
1. 先单独跑 `--case numerics`。如果 Stage1 是对的，问题在 GEMM2 参数或 reduce-scatter
2. `partial` 是否 zero-init —— `epilog="atomic"` 是累加，脏 buffer 直接出错
3. W2 的 shuffle 标志位是 `False`（不是 W1 的 `True`）
4. `shard_w2` 的 fp4x2 下标除 2 有没有漏
5. `D_INTER` 传的是**本 rank 分片** 384，不是 `inter_global`

- [ ] **Step 5: 提交**

```bash
git add op_tests/multigpu_tests/tp_moe_stage1_ref.py \
        op_tests/multigpu_tests/test_tp_moe_stage1.py
git commit -m "test(tp-moe): end-to-end stage1 + v2 gemm2 + reduce-scatter vs full MoE

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 7: 公开导出、冻结校验、阶段二接缝

**Files:**
- Modify: `aiter/ops/flydsl/kernels/mega_moe/__init__.py:8-16`
- Modify: `op_tests/multigpu_tests/test_tp_moe_stage1.py`

`__init__.py` 现在是一个 `_LAZY` 字典（`:8-16`），加两个条目是**纯增量**——现有条目和 `__getattr__` / `__dir__` 一字不动，满足决定 19。

- [ ] **Step 1: 写失败的测试**

`test_tp_moe_stage1.py` 追加：

```python
def case_exports():
    import aiter.ops.flydsl.kernels.mega_moe as mm

    assert "TPMoEStage1" in mm.__all__
    assert "TPMoEStage1Output" in mm.__all__
    assert mm.TPMoEStage1 is TPMoEStage1
    assert mm.TPMoEStage1Output is TPMoEStage1Output
    # existing exports must survive untouched
    for name in (
        "MegaMoEConfig",
        "MegaMoEV2",
        "Stage1Config",
        "Stage2Config",
        "compile_gemm1",
        "gemm1_kernel",
        "select_mega_moe_config",
    ):
        assert name in mm.__all__, f"existing export {name} disappeared"
        assert getattr(mm, name) is not None

    # phase-2 seam: the knob exists and rejects unimplemented transports clearly
    device = torch.device("cuda", 0)
    w1, w1_scale = _fake_w1(NETWORK["experts"], 384, NETWORK["model_dim"], device)
    try:
        TPMoEStage1(
            model_dim=NETWORK["model_dim"],
            inter_dim=384,
            experts=NETWORK["experts"],
            topk=NETWORK["topk"],
            w1=w1,
            w1_scale=w1_scale,
            tp_size=8,
            tp_rank=0,
            device=device,
            transport="fused_allgather",
        )
    except NotImplementedError as exc:
        assert "fused_allgather" in str(exc), exc
    else:
        raise AssertionError("unimplemented transport must raise NotImplementedError")
    print("case_exports OK")
```

`CASES` 追加 `"exports": case_exports,`。

- [ ] **Step 2: 跑测试确认失败**

Run: `python3 op_tests/multigpu_tests/test_tp_moe_stage1.py --case exports`
Expected: FAIL — `AssertionError: assert 'TPMoEStage1' in mm.__all__`

- [ ] **Step 3: 写实现**

把 `aiter/ops/flydsl/kernels/mega_moe/__init__.py` 的 `_LAZY` 字典改成（**只加两行，其余原样**）：

```python
_LAZY = {
    "MegaMoEConfig": "mega_moe_config",
    "MegaMoEV2": "mega_moe_v2",
    "Stage1Config": "mega_moe_config",
    "Stage2Config": "mega_moe_config",
    "TPMoEStage1": "tp_moe_stage1",
    "TPMoEStage1Output": "tp_moe_stage1",
    "compile_gemm1": "gemm1",
    "gemm1_kernel": "gemm1",
    "select_mega_moe_config": "mega_moe_config",
}
```

- [ ] **Step 4: 跑测试确认通过**

Run: `python3 op_tests/multigpu_tests/test_tp_moe_stage1.py --case exports`
Expected: `case_exports OK`

- [ ] **Step 5: 跑 `MegaMoEV2` 回归，证明老路径没被碰坏**

Run:
```bash
MORI_SHMEM_HEAP_SIZE=40G torchrun --standalone --nproc_per_node=8 \
    op_tests/multigpu_tests/test_mega_moe_v2.py \
    --network v4_pro --bs-list 128,512 --iters 10 --accuracy-max-bs 512 --rtol 0.10
```
Expected: 退出码 0，无 `AssertionError`

同时确认本次改动对既有文件是纯增量：
```bash
git diff --stat main...HEAD -- aiter/
```
Expected: 只有 `mega_moe/__init__.py` 有改动且是 `2 insertions(+)`，`tp_moe_stage1.py` 是新增文件

- [ ] **Step 6: 提交**

```bash
git add aiter/ops/flydsl/kernels/mega_moe/__init__.py \
        op_tests/multigpu_tests/test_tp_moe_stage1.py
git commit -m "feat(tp-moe): export TPMoEStage1 and reserve the phase-2 transport seam

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## 全量回归命令

```bash
# 单进程用例
python3 op_tests/multigpu_tests/test_tp_moe_stage1.py --case construct
python3 op_tests/multigpu_tests/test_tp_moe_stage1.py --case capacity
python3 op_tests/multigpu_tests/test_tp_moe_stage1.py --case exports

# 8 卡用例
for c in all_gather forward_contract numerics prequant e2e; do
  torchrun --standalone --nproc_per_node=8 \
      op_tests/multigpu_tests/test_tp_moe_stage1.py --case "$c" || echo "FAILED: $c"
done

# 老路径回归
MORI_SHMEM_HEAP_SIZE=40G torchrun --standalone --nproc_per_node=8 \
    op_tests/multigpu_tests/test_mega_moe_v2.py \
    --network v4_pro --bs-list 128,512 --iters 10 --accuracy-max-bs 512 --rtol 0.10
```

TP4 验证把 `--nproc_per_node` 改成 4 即可——`inter_global = 384 * world` 会自动变成 1536，本 rank 分片仍是 384。注意 TP4 的**真实**目标形状是 `inter_dim=768`（`inter_global=3072`），要验那个得把测试里的 `inter_global` 固定成 3072；本方案默认按 384 分片跑通即可，768 留到后续。

## 阶段二（融合版）的接缝

阶段一已经把接缝留好了，阶段二**不改 API**，只做三件事：

1. `transport="fused_allgather"` 分支放开（现在是 `NotImplementedError`）
2. 新增一个 fused kernel：照 `mega_moe_stage1.py` 的 ticket/epoch 调度骨架，把 `dispatch.py` 的 `emit_dispatch_*` 换成 push-based all-gather，让 GEMM1 consumer 按 tile 等数据到位
3. 复用 `_pack()` 出同一个 `TPMoEStage1Output`

对拍方式：同一进程里构造两个实例（`transport="allgather_bf16"` 和 `"fused_allgather"`），喂同一份输入，逐行比 `inter_sorted_quant * scale`。因为决定 11 是「每次调用新分配」，两份结果不会互相覆盖，可以直接放在一起比。

阶段一同时给出了带宽基线：`forward`（gather BF16，每行 14336 B）与 `forward_prequant`（gather FP8+scale，每行 7392 B）的耗时差，就是「先 quant 再 gather」这一项单独的收益；融合版再叠加通信计算重叠的收益。

## 已知风险

| 风险 | 说明 | 触发条件 |
|---|---|---|
| pad K-group 填充值不一致 | v2 的 torch 参考期望 `0x7F`(=1.0)，MegaMoE 写 `0x00`，v2 kernel 路径不填 | 仅当 `inter_dim % 256 != 0`。TP8 的 384 命中（12 个 group pad 到 16）。**但推演显示 BK=128 时读不到 pad 区**：K 循环 `ceil(384/128)=3` 个 tile，`tilesPerScaleChunk=256/128=2`，tile 0/1 → chunk 0 (y=0..7)，tile 2 → chunk 1 且 `ikxdl=2%2=0` → y=8..11，共 12 个真实 group。风险因此降级，但仍是 Task 6 数值异常时的排查项之一（BK 若改成 256 会重新命中） |
| `per_1x32_mx_quant` vs `fused_dynamic_mxfp8_quant_moe_sort` 舍入 | 两条入口用了不同的量化实现 | Task 5 直接测这一点 |
| `tile_n` 静默降级 | `resolve_flydsl_stage1_tile_n(384, 256) -> 128` 且只告警一次 | 换 stage1 kernel 名时要重新确认 `inter_dim % tile_n == 0` |
| `moe_sorting` 在 E=384 大 M 下未被测试覆盖 | 仓库里没有 `num_experts=384` 的现成用例 | Task 3 的 `m_local=128`（M=1024）会第一次压到它。**已通过** |
| padding 行的 output scale 未初始化 | Task 5 实测：stage1 不给 padding 行写 scale，两条入口都一样，是既有行为 | GEMM2 用 `token_id < i32_M` 门控 store（`mxmoe_gemm_v2.py:1006-1008`），垃圾不会落盘。Task 6 加 `isfinite` 断言兜底 |
| 断言范围写全张量 | `torch.empty` 尾部是垃圾；`max(0.0, nan)` 还会把 NaN 静默吞掉 | 所有内容断言必须限定 `[:nvalid]`，并显式 `isfinite` 检查 |

## Self-Review

**1. 决定覆盖**：22 条决定逐条对应——1/2/3/4→Task 1 构造与 `tp_size` 校验；5→文件路径与类名；6→Task 1 的 `group`/`device` 处理；7→Task 3 + Task 5；8→测试里传已 shuffle 的 W1；9/10→Task 1 构造参数；11→Task 3 的 per-call 分配断言；12→文档化前提（`_validate_call` 不做跨卡检查）；13/15→Task 3 的编码断言；14→Task 6 直接调 `mxfp4_moe_gemm2`；16/18→Task 3 的 baseline 链路；17→Task 7 的 transport 接缝；19→Task 7 Step 5 的回归；20→Task 4 + Task 6；21→无 CI 改动；22→Task 2 的 rank-major 断言 + Task 6 的 reduce-scatter。

**2. 占位符扫描**：无 TBD/TODO；每个改代码的步骤都给了完整代码；每个命令都给了预期输出；数值超阈值时给的是排查顺序而不是「适当调整容差」。

**3. 类型一致性**：`TPMoEStage1Output` 的字段名在 Task 1 定义后，Task 3/5/6 全程使用同一套（`inter_sorted_quant` / `inter_sorted_shuffled_scale` / `sorted_token_ids` / `sorted_weights` / `sorted_expert_ids` / `num_valid_ids` / `m_logical` / `max_sorted` / `sort_block_m`）。方法名 `_all_gather_one` / `_all_gather_inputs` / `_sort` / `_run_gemm1` / `_pack` / `_validate_call` / `quantize` / `m_logical_for` / `max_sorted_for` 前后一致。**方法带 `_for` 后缀，输出结构体字段不带**（`out.m_logical` / `out.max_sorted`）——字段名刻意对齐 `mxfp4_moe_gemm2` 的形参 `M_logical=` / `max_sorted=`，方法改名是为了避免把 bound method 误塞进 int 字段。Task 5 明确要求用新的 `_all_gather_one` 版本替换 Task 2 里的闭包实现。
