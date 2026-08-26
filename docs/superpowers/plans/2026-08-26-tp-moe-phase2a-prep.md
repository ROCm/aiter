> **⚠️ 本方案已作废（2026-08-26），只有 Task 1 执行了。**
>
> Task 2 到 Task 8 全部取消。原因是它们服务于「把四个 helper 从 `mega_moe_stage1.py` /
> `dispatch.py` 抽进共享模块」这条路线，而该路线在执行 Task 3 时被否掉了：Task 3 的
> 可复现性关卡失败，`MegaMoEV2` 在与 `main` 逐字节相同的代码上两次运行输出就不一致
> （八个 rank 全部不一致，而编译 IR 完全一致），所以「输出逐位相同」这条守护不存在。
> 决定改为拷贝，`MegaMoEV2` 一行不改，守护随之不再需要。
>
> **实际执行结果：**
> - Task 1 完成，`TPMoEStage1NCCLRef` 与 `case_ref_fidelity` 已提交（`36806710f`、`7c020eac4`）。
> - Task 2 完成后又删除（`2647b3223` 加 `7386405aa`），工具失去调用方且有已知缺陷。
> - Task 3 执行到 Step 3 失败并按方案要求停止，快照脚本未提交。
> - Task 4 到 Task 8 未执行。
>
> 现行设计见 `docs/superpowers/specs/2026-08-26-tp-moe-stage1-fused-p2p-design.md` 第 6 节。
> 四个 helper 的 TP 版本改为在阶段二（下）与使用它们的 kernel 一起写。

---

# TPMoEStage1 阶段二（上）：准备工作 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把阶段一的 NCCL 实现复制成一份测试参照，并把 MegaMoE 里四个与 EP 无关的调度同步 helper 抽进新模块 `collective_sched.py`，全程用「生成的 IR 逐字节相同」守住冻结约束。

**Architecture:** 两件互不相干的准备工作。第一件是纯 Python 层的复制，产出 `TPMoEStage1NCCLRef`，用一个「两者输出逐位相同」的测试证明副本忠实。第二件是 FlyDSL 层的代码搬家，四个 helper 逐个搬、逐个验证，每搬一个就跑一次 IR 指纹比对。搬完之后 `MegaMoEV2` 的行为和生成代码都必须与搬家前完全一致。本方案**不写任何融合 kernel 代码**，那是阶段二（下）的事。

**Tech Stack:** Python 3.12、PyTorch、FlyDSL（`@flyc.jit` / `@flyc.kernel` tracing）、Mori SHMEM、ROCm gfx950、8 卡 torchrun。

**依据文档：** `docs/superpowers/specs/2026-08-26-tp-moe-stage1-fused-p2p-design.md`（第 3.5 节与第 6 节）。

**分支：** `dev/all_gather_merge_stage1_naive`。

---

## File Structure

| 文件 | 动作 | 职责 |
|---|---|---|
| `op_tests/multigpu_tests/tp_moe_stage1_nccl_ref.py` | 新建 | `TPMoEStage1NCCLRef`，阶段一 NCCL 实现的忠实副本，一次性参照，将来删除 |
| `op_tests/multigpu_tests/flydsl_ir_fingerprint.py` | 新建 | 导出 FlyDSL JIT cache 里全部 `CompiledArtifact` 的 `sha256(_ir_text)`，用于冻结守护 |
| `op_tests/multigpu_tests/megamoe_v2_snapshot.py` | 新建 | 固定 seed 跑一次 `MegaMoEV2`，把输出张量存盘，用于逐位比对 |
| `aiter/ops/flydsl/kernels/mega_moe/collective_sched.py` | 新建 | 四个与 EP 无关的调度同步 helper |
| `aiter/ops/flydsl/kernels/mega_moe/dispatch.py` | 修改 | 删除 `_copy_token_row`，改为 import |
| `aiter/ops/flydsl/kernels/mega_moe/mega_moe_stage1.py` | 修改 | 三段内联代码改为调用 helper |
| `op_tests/multigpu_tests/test_tp_moe_stage1.py` | 修改 | 新增 `case_ref_fidelity` |

**为什么参照实现放在 `op_tests` 而不是 `aiter` 包内：** 它的位置本身就说明它是一次性的。将来删除时不会有人误以为在删公开 API。见 spec 第 3.5 节。

---

## 关于冻结守护的两个已验证事实

执行本方案的人必须先理解这两点，否则第 4 到 7 个 task 的验证步骤会看起来莫名其妙。

**第一，只能比 `_ir_text`，不能比 `_source_ir`。** 后者第一行就是 `#loc = loc("/root/workspace/aiter/.../mega_moe_stage2.py":502:0)`，把绝对路径和行号烤进了 MLIR。函数换个文件、换个行号，这个字段必然变，但生成的代码可能一条指令都没动。

**第二，不能按 cache 目录名配对。** FlyDSL 的 cache key 包含 kernel 函数及其依赖的源码文本（`flydsl/compiler/jit_function.py:572-578`），所以搬家后目录名和 `.pkl` 文件名全都不一样。比对必须以哈希的多重集合为单位，两次运行之间要清空 cache。

**第三，引入函数边界不改 IR。** `@flyc.jit` 在 tracing 期间（`ir.Context` 存活时）是普通 Python 调用，函数体直接内联进 trace，不发 `func.call`：

```python
# flydsl/compiler/jit_function.py:1357-1359
def __call__(self, *args, **kwargs):
    if ir.Context.current is not None:
        return self.func(*args, **kwargs)
```

真正会破坏 IR 一致性的是语句重排或常量折叠结果改变，不是函数边界本身。

---

## Task 1: `TPMoEStage1NCCLRef` — 忠实副本

**Files:**
- Create: `op_tests/multigpu_tests/tp_moe_stage1_nccl_ref.py`
- Modify: `op_tests/multigpu_tests/test_tp_moe_stage1.py`（新增 `case_ref_fidelity`，注册进 `CASES`）

- [ ] **Step 1: 复制文件**

```bash
cd /root/workspace/aiter
cp aiter/ops/flydsl/kernels/mega_moe/tp_moe_stage1.py \
   op_tests/multigpu_tests/tp_moe_stage1_nccl_ref.py
```

- [ ] **Step 2: 改模块 docstring**

把 `op_tests/multigpu_tests/tp_moe_stage1_nccl_ref.py` 开头的整段 docstring（第 3 到 10 行，从 `"""Tensor-parallel MoE Stage1.` 到 `"""`）整体替换为：

```python
"""Phase-1 NCCL implementation of TP MoE Stage1, kept as a test reference.

This is a verbatim copy of ``aiter/ops/flydsl/kernels/mega_moe/tp_moe_stage1.py``
as of the commit that introduced the fused P2P transport, with the ``transport``
parameter removed. It exists ONLY so the fused implementation has something to
be checked against, and it will be deleted once the fused path is trusted.

Do not import this from production code. Do not fix bugs here that were not also
fixed in the real operator -- the two are deliberately allowed to diverge.
"""
```

- [ ] **Step 3: 改 import，共享输出契约**

把这一段：

```python
from aiter.fused_moe import moe_sorting
from aiter.ops.flydsl.moe_kernels import flydsl_moe_stage1, get_flydsl_kernel_params
from aiter.ops.quant import fused_dynamic_mxfp8_quant_moe_sort
from aiter.utility.fp4_utils import moe_mxfp4_sort

from .quant import per_1x32_mx_quant
```

替换为：

```python
from aiter.fused_moe import moe_sorting
from aiter.ops.flydsl.kernels.mega_moe.quant import per_1x32_mx_quant
from aiter.ops.flydsl.kernels.mega_moe.tp_moe_stage1 import TPMoEStage1Output
from aiter.ops.flydsl.moe_kernels import flydsl_moe_stage1, get_flydsl_kernel_params
from aiter.ops.quant import fused_dynamic_mxfp8_quant_moe_sort
from aiter.utility.fp4_utils import moe_mxfp4_sort
```

`TPMoEStage1Output` 从生产模块 import 而不是复制，因为这个 dataclass 是两边都要满足的输出契约，复制会让契约有两个定义。

- [ ] **Step 4: 删掉本地的 `TPMoEStage1Output` 定义**

删除整个 dataclass，即从 `from dataclasses import dataclass` 那一行、以及：

```python
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
```

整块删除。

- [ ] **Step 5: 删掉 transport 相关代码**

删除这三行常量及其上方的三行注释：

```python
# How the operator collects every rank's activation shard. The names describe the
# mechanism, not the payload dtype: both entry points (bf16 ``forward`` and fp8
# ``forward_prequant``) run over whichever transport is selected.
_TRANSPORT_NCCL = "nccl_allgather"  # dist.all_gather_into_tensor, outside the kernel
_TRANSPORT_FUSED = "fused_p2p"  # phase 2: in-kernel P2P over Mori SHMEM
_TRANSPORTS = frozenset({_TRANSPORT_NCCL, _TRANSPORT_FUSED})
```

删除构造函数签名里的这一行：

```python
        transport: str = _TRANSPORT_NCCL,
```

删除构造函数体里的这一段：

```python
        if transport not in _TRANSPORTS:
            raise ValueError(
                f"unknown transport={transport!r}; expected one of {sorted(_TRANSPORTS)}"
            )
        if transport != _TRANSPORT_NCCL:
            raise NotImplementedError(
                f"transport={transport!r} is not implemented yet; only "
                f"{_TRANSPORT_NCCL!r} is available"
            )
```

删除这一行：

```python
        self.transport = transport
```

- [ ] **Step 6: 改类名**

```python
class TPMoEStage1:
    """Stateful TP4/TP8 MoE Stage1 operator.
```

改为：

```python
class TPMoEStage1NCCLRef:
    """Phase-1 NCCL reference. Same behaviour as the original TPMoEStage1.
```

其余 docstring 内容保持不变。

- [ ] **Step 7: 跑 black 并确认能 import**

```bash
cd /root/workspace/aiter
python -m black op_tests/multigpu_tests/tp_moe_stage1_nccl_ref.py
PYTHONPATH=. python -c "
import sys, os
sys.path.insert(0, 'op_tests/multigpu_tests')
from tp_moe_stage1_nccl_ref import TPMoEStage1NCCLRef
from aiter.ops.flydsl.kernels.mega_moe.tp_moe_stage1 import TPMoEStage1Output
import inspect
p_ref = list(inspect.signature(TPMoEStage1NCCLRef.__init__).parameters)
print('ref ctor params:', p_ref)
assert 'transport' not in p_ref, 'transport must be gone'
print('shares Output contract:', TPMoEStage1NCCLRef.forward.__doc__ is not None)
"
```

Expected：打印出不含 `transport` 的参数列表，无 assert 失败。

- [ ] **Step 8: 写保真测试（先写，必须能失败）**

在 `op_tests/multigpu_tests/test_tp_moe_stage1.py` 的 `case_exports` 之后、`CASES` 之前插入：

```python
def case_ref_fidelity():
    """TPMoEStage1NCCLRef must be a faithful copy: bit-identical to production.

    The two are the same code today, so anything other than bit-equality means
    the copy was botched. Once the fused path lands this case is what proves the
    reference still represents phase-1 behaviour.
    """
    from tp_moe_stage1_nccl_ref import TPMoEStage1NCCLRef

    rank, world, device = _setup_dist()
    model_dim = NETWORK["model_dim"]
    experts, topk = NETWORK["experts"], NETWORK["topk"]
    inter_dim, limit = 384, NETWORK["swiglu_limit"]

    _, _, w1_shuf, w1_scale_shuf = build_mxfp4_w1(
        experts, inter_dim, model_dim, device, seed=4242
    )
    common = dict(
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
    prod = TPMoEStage1(**common)
    ref = TPMoEStage1NCCLRef(**common)

    for m_local in (1, 8, 64):
        g = torch.Generator(device="cpu").manual_seed(9100 + rank * 17 + m_local)
        x = (
            torch.randn((m_local, model_dim), generator=g).to(
                device=device, dtype=torch.bfloat16
            )
            * (model_dim**-0.25)
        )
        ids, wts = _random_routes(m_local, experts, topk, device, seed=71 + rank)

        a = prod.forward(x, wts, ids)
        b = ref.forward(x, wts, ids)

        assert type(b).__name__ == "TPMoEStage1Output", type(b)
        assert a.m_logical == b.m_logical, (a.m_logical, b.m_logical)
        assert a.max_sorted == b.max_sorted, (a.max_sorted, b.max_sorted)
        nvalid = int(a.num_valid_ids[0].item())
        assert nvalid == int(b.num_valid_ids[0].item())
        for name in (
            "sorted_token_ids",
            "sorted_expert_ids",
            "sorted_weights",
            "num_valid_ids",
        ):
            ta, tb = getattr(a, name), getattr(b, name)
            n = nvalid if name != "sorted_expert_ids" else nvalid // prod.sort_block_m
            n = min(n, ta.shape[0])
            assert torch.equal(ta[:n], tb[:n]), f"{name} differs at m_local={m_local}"
        qa = a.inter_sorted_quant.view(torch.uint8)[:nvalid]
        qb = b.inter_sorted_quant.view(torch.uint8)[:nvalid]
        assert torch.equal(qa, qb), f"payload differs at m_local={m_local}"
        sa = a.inter_sorted_shuffled_scale.view(torch.uint8)
        sb = b.inter_sorted_shuffled_scale.view(torch.uint8)
        assert torch.equal(sa, sb), f"scale differs at m_local={m_local}"
        if rank == 0:
            print(f"  m_local={m_local} nvalid={nvalid} bit-identical")

    dist.barrier()
    dist.destroy_process_group()
    if rank == 0:
        print("case_ref_fidelity OK")
```

并把 `CASES` 改为：

```python
CASES = {
    "construct": case_construct_validates,
    "capacity": case_capacity,
    "all_gather": case_all_gather,
    "forward_contract": case_forward_contract,
    "numerics": case_numerics,
    "prequant": case_prequant_equivalence,
    "e2e": case_end_to_end,
    "exports": case_exports,
    "ref_fidelity": case_ref_fidelity,
}
```

- [ ] **Step 9: 负对照 — 证明这个测试真的能失败**

临时在 `tp_moe_stage1_nccl_ref.py` 的 `_sort` 里把 `block_size=self.sort_block_m` 改成 `block_size=64`，然后跑：

```bash
cd /root/workspace/aiter
PYTHONPATH=. torchrun --standalone --nproc_per_node=8 \
    op_tests/multigpu_tests/test_tp_moe_stage1.py --case ref_fidelity
```

Expected：**FAIL**，`AssertionError: sorted_token_ids differs at m_local=1` 或类似。

看到失败后把 `block_size` 改回 `self.sort_block_m`。

> 如果这一步**没有**失败，说明测试没有真正比对，必须先修测试再继续。阶段一有过 `max(0.0, nan) == 0.0` 静默吞掉 NaN 的教训，每个比对用例都要先证明它会失败。

- [ ] **Step 10: 跑通测试**

```bash
cd /root/workspace/aiter
PYTHONPATH=. torchrun --standalone --nproc_per_node=8 \
    op_tests/multigpu_tests/test_tp_moe_stage1.py --case ref_fidelity
```

Expected：

```
  m_local=1 nvalid=... bit-identical
  m_local=8 nvalid=... bit-identical
  m_local=64 nvalid=... bit-identical
case_ref_fidelity OK
```

- [ ] **Step 11: 确认阶段一原有用例没被弄坏**

```bash
cd /root/workspace/aiter
for c in construct capacity exports; do
  PYTHONPATH=. python op_tests/multigpu_tests/test_tp_moe_stage1.py --case $c 2>&1 | tail -1
done
for c in all_gather forward_contract prequant; do
  PYTHONPATH=. torchrun --standalone --nproc_per_node=8 \
      op_tests/multigpu_tests/test_tp_moe_stage1.py --case $c 2>&1 | tail -1
done
```

Expected：六行分别以 `case_construct_validates OK`、`case_capacity OK`、`case_exports OK`、`case_all_gather OK`、`case_forward_contract OK`、`case_prequant_equivalence OK` 结尾。

- [ ] **Step 12: 提交**

```bash
cd /root/workspace/aiter
python -m black --check op_tests/multigpu_tests/tp_moe_stage1_nccl_ref.py \
    op_tests/multigpu_tests/test_tp_moe_stage1.py
git add op_tests/multigpu_tests/tp_moe_stage1_nccl_ref.py \
        op_tests/multigpu_tests/test_tp_moe_stage1.py
git commit -m "test(tp-moe): copy the NCCL implementation into a disposable reference

TPMoEStage1NCCLRef is a verbatim copy of the phase-1 operator minus the
transport parameter. It exists so the upcoming fused implementation has
something to be checked against, and it lives under op_tests so that
deleting it later is obviously not an API removal.

case_ref_fidelity proves the copy is faithful by requiring bit-equality
on every output field. Verified it fails when the copy's sort block size
is perturbed.

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 2: IR 指纹工具

**Files:**
- Create: `op_tests/multigpu_tests/flydsl_ir_fingerprint.py`

- [ ] **Step 1: 写工具**

创建 `op_tests/multigpu_tests/flydsl_ir_fingerprint.py`：

```python
# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Dump a path-independent fingerprint of every compiled FlyDSL kernel.

Used to prove that a pure code-movement refactor did not change the generated
code. Two facts make the naive approaches wrong:

  * ``CompiledArtifact._source_ir`` embeds ``loc("/abs/path.py":line:col)``, so it
    changes whenever a function moves file or line even if the emitted code is
    identical. Only ``_ir_text`` is path-free -- it starts at
    ``module attributes {gpu.container_module}`` and carries no source location.
  * The cache directory name is derived from the kernel's SOURCE TEXT
    (flydsl/compiler/jit_function.py:572-578), so directory names change too.
    Fingerprints must therefore be compared as a sorted multiset, not pairwise
    by directory.

Usage:

    rm -rf ~/.flydsl/cache
    <run the workload>
    python op_tests/multigpu_tests/flydsl_ir_fingerprint.py --out before.txt

    <apply the refactor>

    rm -rf ~/.flydsl/cache
    <run the same workload>
    python op_tests/multigpu_tests/flydsl_ir_fingerprint.py --out after.txt

    diff before.txt after.txt && echo "IR UNCHANGED"
"""

import argparse
import hashlib
import os
import pathlib
import pickle
import sys


def fingerprints(cache_root: pathlib.Path) -> list[str]:
    """Sorted sha256 of every artifact's _ir_text. Path- and order-independent."""
    out = []
    unreadable = []
    for pkl in sorted(cache_root.rglob("*.pkl")):
        try:
            artifact = pickle.loads(pkl.read_bytes())
        except Exception as exc:  # noqa: BLE001 - report, do not mask
            unreadable.append(f"{pkl}: {type(exc).__name__}: {exc}")
            continue
        ir = getattr(artifact, "_ir_text", None)
        if not ir:
            unreadable.append(f"{pkl}: no _ir_text")
            continue
        out.append(hashlib.sha256(ir.encode()).hexdigest())
    if unreadable:
        print(f"WARNING: {len(unreadable)} artifact(s) skipped:", file=sys.stderr)
        for line in unreadable[:10]:
            print(f"  {line}", file=sys.stderr)
    return sorted(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--cache",
        default=os.path.expanduser("~/.flydsl/cache"),
        help="FlyDSL JIT cache root",
    )
    ap.add_argument("--out", required=True, help="where to write the sorted hashes")
    args = ap.parse_args()

    root = pathlib.Path(args.cache)
    if not root.is_dir():
        raise SystemExit(f"cache root does not exist: {root}")
    hashes = fingerprints(root)
    if not hashes:
        raise SystemExit(
            f"no artifacts under {root}; did the workload actually compile anything?"
        )
    pathlib.Path(args.out).write_text("\n".join(hashes) + "\n")
    print(f"{len(hashes)} artifacts -> {args.out} ({len(set(hashes))} distinct)")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 在当前 cache 上验证工具能跑**

```bash
cd /root/workspace/aiter
python op_tests/multigpu_tests/flydsl_ir_fingerprint.py --out /tmp/probe.txt
```

Expected：形如 `437 artifacts -> /tmp/probe.txt (360 distinct)`，具体数字取决于当前 cache 内容，只要 artifacts 数大于 0 即可。

- [ ] **Step 3: 验证工具对同一份 cache 是稳定的**

```bash
cd /root/workspace/aiter
python op_tests/multigpu_tests/flydsl_ir_fingerprint.py --out /tmp/probe2.txt
diff /tmp/probe.txt /tmp/probe2.txt && echo "STABLE"
```

Expected：打印 `STABLE`，无 diff 输出。

- [ ] **Step 4: 验证 `_source_ir` 确实不可用（记录证据，不写进代码）**

```bash
cd /root/workspace/aiter
python - <<'PY'
import pathlib, pickle
pkl = next(pathlib.Path.home().joinpath(".flydsl/cache").rglob("*.pkl"))
art = pickle.loads(pkl.read_bytes())
print("_ir_text    contains '.py':", ".py" in art._ir_text)
print("_source_ir  contains '.py':", ".py" in art._source_ir)
print("_source_ir  first line:", art._source_ir.splitlines()[0][:100])
PY
```

Expected：

```
_ir_text    contains '.py': False
_source_ir  contains '.py': True
_source_ir  first line: #loc = loc("/root/workspace/aiter/aiter/ops/flydsl/kernels/mega_moe/...py":NNN:0)
```

这一步只是让执行者亲眼确认为什么工具只读 `_ir_text`。

- [ ] **Step 5: 提交**

```bash
cd /root/workspace/aiter
python -m black --check op_tests/multigpu_tests/flydsl_ir_fingerprint.py
git add op_tests/multigpu_tests/flydsl_ir_fingerprint.py
git commit -m "test: add a path-independent FlyDSL IR fingerprint tool

Guards pure code-movement refactors. Reads only CompiledArtifact._ir_text,
because _source_ir embeds loc(\"/abs/path.py\":line:col) and would flag every
file move as a change. Compares as a sorted multiset because the cache
directory name is derived from the kernel's source text.

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 3: MegaMoEV2 输出快照工具 + 抓取重构前基线

**Files:**
- Create: `op_tests/multigpu_tests/megamoe_v2_snapshot.py`

- [ ] **Step 1: 写快照脚本**

创建 `op_tests/multigpu_tests/megamoe_v2_snapshot.py`：

```python
# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Run MegaMoEV2 once with a fixed seed and save the output, for bit-exact diffing.

Second half of the freeze guard: flydsl_ir_fingerprint.py proves the generated
code did not change; this proves the numbers did not either.

    rm -rf ~/.flydsl/cache
    torchrun --standalone --nproc_per_node=8 \
        op_tests/multigpu_tests/megamoe_v2_snapshot.py --out /tmp/before
    python op_tests/multigpu_tests/flydsl_ir_fingerprint.py --out /tmp/before_ir.txt
"""

import argparse
import os
import sys

import torch
import torch.distributed as dist

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from test_mega_moe_v2 import (  # noqa: E402
    NETWORKS,
    _cleanup,
    _make_inputs,
    _next_power_of_two,
    _quantize_weights,
    _setup_dist,
)

from aiter.ops.flydsl.kernels.mega_moe import MegaMoEV2  # noqa: E402


def compare(prefix_a, prefix_b, world):
    """Bit-exact comparison of two snapshot sets. Returns a list of mismatches."""
    bad = []
    for r in range(world):
        ta = torch.load(f"{prefix_a}.rank{r}.pt")
        tb = torch.load(f"{prefix_b}.rank{r}.pt")
        if ta.shape != tb.shape or ta.dtype != tb.dtype:
            bad.append(
                f"rank{r}: {tuple(ta.shape)}/{ta.dtype} vs {tuple(tb.shape)}/{tb.dtype}"
            )
            continue
        ba, bb = ta.view(torch.uint8), tb.view(torch.uint8)
        if not torch.equal(ba, bb):
            n = int((ba != bb).sum().item())
            bad.append(f"rank{r}: {n} bytes differ out of {ba.numel()}")
    return bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", help="output prefix; rank N -> <out>.rankN.pt")
    ap.add_argument(
        "--compare",
        nargs=2,
        metavar=("A", "B"),
        help="compare two prefixes bit-exactly and exit; no GPUs needed",
    )
    ap.add_argument("--world", type=int, default=8, help="rank count for --compare")
    ap.add_argument("--network", choices=list(NETWORKS), default="v4_pro")
    ap.add_argument("--tokens", type=int, default=128)
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()

    if args.compare:
        bad = compare(args.compare[0], args.compare[1], args.world)
        for line in bad:
            print("MISMATCH", line)
        if bad:
            raise SystemExit(1)
        print(f"IDENTICAL across {args.world} ranks")
        return
    if not args.out:
        raise SystemExit("--out is required unless --compare is given")

    rank, world, device = _setup_dist()
    try:
        net = NETWORKS[args.network]
        if net["experts"] % world:
            raise ValueError(f"experts={net['experts']} not divisible by world={world}")
        local_experts = net["experts"] // world
        packed = _quantize_weights(
            net["model_dim"], net["inter_dim"], local_experts, rank, args.seed, device
        )
        w1, w1_scale, w2, w2_scale = packed[0], packed[1], packed[2], packed[3]
        x, weights, ids = _make_inputs(
            args.tokens,
            net["model_dim"],
            net["experts"],
            net["topk"],
            rank,
            args.seed,
            device,
        )
        moe = MegaMoEV2(
            rank=rank,
            world_size=world,
            quant="a8w4",
            w1=w1,
            w1_scale=w1_scale,
            w2=w2,
            w2_scale=w2_scale,
            model_dim=net["model_dim"],
            inter_dim=net["inter_dim"],
            experts=net["experts"],
            topk=net["topk"],
            max_tok_per_rank=max(16, _next_power_of_two(args.tokens)),
            swiglu_limit=net["swiglu_limit"],
        )
        out = moe.forward(x, weights, ids)
        torch.cuda.synchronize()
        path = f"{args.out}.rank{rank}.pt"
        torch.save(out.detach().cpu(), path)
        print(f"rank{rank}: saved {tuple(out.shape)} {out.dtype} -> {path}")
        dist.barrier()
    finally:
        _cleanup()


if __name__ == "__main__":
    main()
```

`--compare` 不碰 GPU，也不初始化 process group，所以基线比对可以在单机单进程下跑。

- [ ] **Step 2: 抓取重构前的基线**

```bash
cd /root/workspace/aiter
rm -rf ~/.flydsl/cache
PYTHONPATH=. torchrun --standalone --nproc_per_node=8 \
    op_tests/multigpu_tests/megamoe_v2_snapshot.py --out /tmp/mm_before
python op_tests/multigpu_tests/flydsl_ir_fingerprint.py --out /tmp/mm_before_ir.txt
```

Expected：八行 `rankN: saved (128, 7168) torch.bfloat16 -> /tmp/mm_before.rankN.pt`，然后一行 `NNN artifacts -> /tmp/mm_before_ir.txt (MMM distinct)`。

- [ ] **Step 3: 验证基线本身是可复现的（关键，别跳）**

```bash
cd /root/workspace/aiter
rm -rf ~/.flydsl/cache
PYTHONPATH=. torchrun --standalone --nproc_per_node=8 \
    op_tests/multigpu_tests/megamoe_v2_snapshot.py --out /tmp/mm_ctrl
python op_tests/multigpu_tests/flydsl_ir_fingerprint.py --out /tmp/mm_ctrl_ir.txt
python op_tests/multigpu_tests/megamoe_v2_snapshot.py --compare /tmp/mm_before /tmp/mm_ctrl
diff /tmp/mm_before_ir.txt /tmp/mm_ctrl_ir.txt && echo "IR REPRODUCIBLE"
```

Expected：`IDENTICAL across 8 ranks` 和 `IR REPRODUCIBLE`。

> 如果这一步就不一致，说明 `MegaMoEV2` 本身有不确定性（例如 atomic epilogue 的累加顺序随调度变化），那么「输出逐位相同」这条守护不成立，必须**停下来报告**，改用 `_ir_text` 单条守护加上 `test_mega_moe_v2.py` 的容差测试。不要在基线不可复现的情况下继续往下做。

- [ ] **Step 4: 提交工具**

```bash
cd /root/workspace/aiter
python -m black op_tests/multigpu_tests/megamoe_v2_snapshot.py
python -m black --check op_tests/multigpu_tests/megamoe_v2_snapshot.py
git add op_tests/multigpu_tests/megamoe_v2_snapshot.py
git commit -m "test: add a MegaMoEV2 output snapshot/compare tool for the freeze guard

Pairs with flydsl_ir_fingerprint.py: that one proves the generated code did
not change, this one proves the numbers did not either.

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 4: 抽取 `_copy_token_row`（唯一的纯剪切粘贴）

**Files:**
- Create: `aiter/ops/flydsl/kernels/mega_moe/collective_sched.py`
- Modify: `aiter/ops/flydsl/kernels/mega_moe/dispatch.py`（删除 183-201 行的函数定义，改为 import）

- [ ] **Step 1: 建新模块并粘贴函数**

创建 `aiter/ops/flydsl/kernels/mega_moe/collective_sched.py`：

```python
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
# ruff: noqa: B023, SIM102
"""Collective scheduling and synchronisation helpers shared by MegaMoE and TP MoE.

Everything here is EP-agnostic: no expert-major routing, no srcmap encoding, no
per-expert histograms, no capacity/compact fork. Those stay in dispatch.py.

These are trace-time helpers, not device functions. ``@flyc.jit`` bodies are
inlined during tracing (flydsl/compiler/jit_function.py:1357-1359 returns
``self.func(*args, **kwargs)`` whenever an ``ir.Context`` is live), so factoring
code in here does not introduce a ``func.call`` and does not change the emitted
IR. What WOULD change it is reordering statements or altering constant folding.
"""

# fmt: off

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr

from aiter.ops.flydsl.kernels import buffer_ops


@flyc.jit
def _copy_token_row(source_rsrc, destination_rsrc, lane, *, fz_safe_end_i32, fz_n_i32):
    lane_offset = lane * fx.Int32(4)
    if const_expr(fz_safe_end_i32 > 0):
        for column in range(lane_offset, fz_safe_end_i32, 512):
            value0 = buffer_ops.buffer_load(
                source_rsrc, column, vec_width=4, dtype=fx.Int32
            )
            value1 = buffer_ops.buffer_load(
                source_rsrc, column + fx.Int32(256), vec_width=4, dtype=fx.Int32
            )
            buffer_ops.buffer_store(value0, destination_rsrc, column)
            buffer_ops.buffer_store(value1, destination_rsrc, column + fx.Int32(256))
    if const_expr(fz_safe_end_i32 < fz_n_i32):
        for column in range(lane_offset + fz_safe_end_i32, fz_n_i32, 256):
            value = buffer_ops.buffer_load(
                source_rsrc, column, vec_width=4, dtype=fx.Int32
            )
            buffer_ops.buffer_store(value, destination_rsrc, column)
```

函数体与 `dispatch.py:183-201` 逐字符相同。

- [ ] **Step 2: 从 `dispatch.py` 删除原定义**

删除 `aiter/ops/flydsl/kernels/mega_moe/dispatch.py` 里从 `@flyc.jit` 到函数结尾的整段（原 183-201 行），即上面 Step 1 中 `@flyc.jit` 开始的那 19 行。

- [ ] **Step 3: 在 `dispatch.py` 加 import**

把 `dispatch.py` 的最后一行 import：

```python
from .. import communication_ops_utils as comm_ops
```

改为：

```python
from .. import communication_ops_utils as comm_ops
from .collective_sched import _copy_token_row
```

- [ ] **Step 4: 确认没有循环 import 且能编译**

```bash
cd /root/workspace/aiter
PYTHONPATH=. python -c "
from aiter.ops.flydsl.kernels.mega_moe import dispatch, collective_sched
assert dispatch._copy_token_row is collective_sched._copy_token_row
print('import ok, same object')
"
```

Expected：`import ok, same object`

- [ ] **Step 5: 跑守护**

```bash
cd /root/workspace/aiter
rm -rf ~/.flydsl/cache
PYTHONPATH=. torchrun --standalone --nproc_per_node=8 \
    op_tests/multigpu_tests/megamoe_v2_snapshot.py --out /tmp/mm_t4
python op_tests/multigpu_tests/flydsl_ir_fingerprint.py --out /tmp/mm_t4_ir.txt
python op_tests/multigpu_tests/megamoe_v2_snapshot.py --compare /tmp/mm_before /tmp/mm_t4
diff /tmp/mm_before_ir.txt /tmp/mm_t4_ir.txt && echo "IR UNCHANGED"
```

Expected：`IDENTICAL across 8 ranks` 和 `IR UNCHANGED`。

> 任何一条不过就**立刻停下**，不要继续抽下一个。回退这次改动，报告差异内容。

- [ ] **Step 6: 提交**

```bash
cd /root/workspace/aiter
python -m black --check aiter/ops/flydsl/kernels/mega_moe/collective_sched.py \
    aiter/ops/flydsl/kernels/mega_moe/dispatch.py
git add aiter/ops/flydsl/kernels/mega_moe/collective_sched.py \
        aiter/ops/flydsl/kernels/mega_moe/dispatch.py
git commit -m "refactor(mega-moe): move _copy_token_row into collective_sched

Pure cut-and-paste of dispatch.py:183-201 -- the row copy is EP-agnostic and
the TP push kernel needs it too. Verified MegaMoEV2 output is bit-identical
and the multiset of generated _ir_text hashes is unchanged.

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 5: 抽取 `emit_ticket_and_roles`

**Files:**
- Modify: `aiter/ops/flydsl/kernels/mega_moe/collective_sched.py`（新增 helper）
- Modify: `aiter/ops/flydsl/kernels/mega_moe/mega_moe_stage1.py:210-225`

- [ ] **Step 1: 在 `collective_sched.py` 末尾加 helper**

追加到 `collective_sched.py`（同时把 `Vec` 和 `comm_ops` 加进 import）：

先把 import 段改为：

```python
import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr
from flydsl.expr.typing import Vector as Vec

from aiter.ops.flydsl.kernels import buffer_ops

from .. import communication_ops_utils as comm_ops
```

再追加：

```python
def emit_ticket_and_roles(*, tid, a_buf, a_entry_count, a_epoch_gate,
        grid_epoch_slot, launch_grid_x, dispatch_blocks):
    """Take this CTA's launch ticket and derive its role for the round.

    One atomic per CTA on thread 0, broadcast to the block through byte 0 of the
    LDS pool. ``a_entry_count`` is a monotonically increasing i64 counter that is
    never reset; dividing by ``launch_grid_x`` recovers which launch this CTA
    belongs to, and the remainder is its role index within that launch.

    Returns ``(ticket, gate_addr, gate_epoch, compact_owner, compact_producer,
    producer_slot)``.
    """
    ticket_scratch = fx.recast_iter(fx.Int64, a_buf.ptr)
    ticket_view = fx.make_view(ticket_scratch, fx.make_layout(1, 1))
    if tid == fx.Int32(0):
        ticket64 = fx.Int64(
            comm_ops.atomic_add_agent(a_entry_count + fx.Int64(grid_epoch_slot * 8), fx.Int64(1))
        )
        fx.ptr_store(Vec.from_elements([ticket64], fx.Int64), ticket_scratch)
    fx.barrier()
    ticket64 = Vec(ticket_view.load())[0]
    generation = ticket64 // fx.Int64(launch_grid_x)
    ticket = fx.Int32(ticket64 - generation * fx.Int64(launch_grid_x))
    gate_addr = a_epoch_gate + fx.Int64(grid_epoch_slot * 4)
    gate_epoch = fx.Int32(generation + fx.Int64(1))
    compact_owner = ticket == fx.Int32(0)
    compact_producer = (ticket > fx.Int32(0)) & (ticket <= fx.Int32(dispatch_blocks))
    producer_slot = ticket - fx.Int32(1)
    return ticket, gate_addr, gate_epoch, compact_owner, compact_producer, producer_slot
```

语句顺序与 `mega_moe_stage1.py:210-225` 完全一致，一行都没有重排。

- [ ] **Step 2: 在 `mega_moe_stage1.py` 替换内联代码**

把 `mega_moe_stage1.py:210-225` 整段：

```python
        ticket_scratch = fx.recast_iter(fx.Int64, a_buf.ptr)
        ticket_view = fx.make_view(ticket_scratch, fx.make_layout(1, 1))
        if tid == fx.Int32(0):
            ticket64 = fx.Int64(
                comm_ops.atomic_add_agent(a_entry_count + fx.Int64(grid_epoch_slot * 8), fx.Int64(1))
            )
            fx.ptr_store(Vec.from_elements([ticket64], fx.Int64), ticket_scratch)
        fx.barrier()
        ticket64 = Vec(ticket_view.load())[0]
        generation = ticket64 // fx.Int64(launch_grid_x)
        ticket = fx.Int32(ticket64 - generation * fx.Int64(launch_grid_x))
        gate_addr = a_epoch_gate + fx.Int64(grid_epoch_slot * 4)
        gate_epoch = fx.Int32(generation + fx.Int64(1))
        compact_owner = ticket == fx.Int32(0)
        compact_producer = (ticket > fx.Int32(0)) & (ticket <= fx.Int32(dispatch_blocks))
        producer_slot = ticket - fx.Int32(1)
```

替换为：

```python
        ticket, gate_addr, gate_epoch, compact_owner, compact_producer, producer_slot = (
            emit_ticket_and_roles(
                tid=tid, a_buf=a_buf, a_entry_count=a_entry_count, a_epoch_gate=a_epoch_gate,
                grid_epoch_slot=grid_epoch_slot, launch_grid_x=launch_grid_x,
                dispatch_blocks=dispatch_blocks))
```

- [ ] **Step 3: 加 import**

把 `mega_moe_stage1.py` 里这一行：

```python
from .. import communication_ops_utils as comm_ops
```

之后插入（保持 import 块的字母序，放在 `from .dispatch import (` 之前）：

```python
from .collective_sched import emit_ticket_and_roles
```

- [ ] **Step 4: 跑守护**

```bash
cd /root/workspace/aiter
rm -rf ~/.flydsl/cache
PYTHONPATH=. torchrun --standalone --nproc_per_node=8 \
    op_tests/multigpu_tests/megamoe_v2_snapshot.py --out /tmp/mm_t5
python op_tests/multigpu_tests/flydsl_ir_fingerprint.py --out /tmp/mm_t5_ir.txt
python op_tests/multigpu_tests/megamoe_v2_snapshot.py --compare /tmp/mm_before /tmp/mm_t5
diff /tmp/mm_before_ir.txt /tmp/mm_t5_ir.txt && echo "IR UNCHANGED"
```

Expected：`IDENTICAL across 8 ranks` 和 `IR UNCHANGED`。

- [ ] **Step 5: 提交**

```bash
cd /root/workspace/aiter
git add aiter/ops/flydsl/kernels/mega_moe/collective_sched.py \
        aiter/ops/flydsl/kernels/mega_moe/mega_moe_stage1.py
git commit -m "refactor(mega-moe): extract emit_ticket_and_roles into collective_sched

Statement order preserved exactly; the LDS scratch stays owned by the caller
and is passed in, since emit_work_pool_loop aliases the same byte. Verified
MegaMoEV2 output is bit-identical and the generated IR is unchanged.

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 6: 抽取 `emit_epoch_rendezvous`

这是四个里最容易出错的一个，因为它是一整条 `if compact_owner: ... else: ...` 语句（`mega_moe_stage1.py:227-279`），包含 parity 翻转、`launch_ready` 跨卡握手、`epoch_gate` 本地重置三件事。**必须整条一起抽**，拆开就得改分支结构。

另有一个陷阱：`next_parity_lane` 和 `launch_epoch_lane` 在嵌套的 `if tid == 0` 内被重新绑定、退出后再读。那对 `readfirstlane` 必须跟 `if tid == 0` 留在同一个 helper 里，否则 SSA 合并点位置变化，IR 文本就变了。

**Files:**
- Modify: `aiter/ops/flydsl/kernels/mega_moe/collective_sched.py`
- Modify: `aiter/ops/flydsl/kernels/mega_moe/mega_moe_stage1.py:227-279`

- [ ] **Step 1: 追加 helper**

先把 `collective_sched.py` 的 import 段补成：

```python
import flydsl.compiler as flyc
import flydsl.expr as fx
import mori.ir.flydsl as mori_shmem
from flydsl.expr import const_expr, range_constexpr
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec

from aiter.ops.flydsl.kernels import buffer_ops

from .. import communication_ops_utils as comm_ops
from .gemm_util import _buffer_load, _buffer_store, _make_buffer_from_addr
```

再追加：

```python
def emit_epoch_rendezvous(*, tid, compact_owner, parity_rsrc, expected_rsrc,
        a_payload_ready_rows, p_launch_ready, a_launch_ready, a_work_head, a_work_tail,
        a_group_done, gate_addr, gate_epoch, fz_npes, fz_rank, fz_tile_m,
        payload_tile_ready, external_grouping, direct_fixed_slot):
    """Flip the epoch, rendezvous with every peer, reset local state, open the gate.

    One indivisible if/else: the owner CTA flips parity and expected, publishes
    its launch epoch to every peer and waits for theirs, resets the work pool,
    then stores the gate; every other CTA just waits on the gate. Splitting the
    three phases apart would change the branch structure, and the handshake
    needs the launch_epoch the flip produces.

    ``next_parity_lane`` / ``launch_epoch_lane`` are rebound inside a nested
    ``if tid == 0`` and read after it -- the readfirstlane pair must stay in the
    same function as that branch or the SSA merge point moves.
    """
    if compact_owner:
        next_parity_lane = fx.Int32(0)
        launch_epoch_lane = fx.Int32(0)
        if tid == fx.Int32(0):
            old_parity = _buffer_load(parity_rsrc, fx.Int32(0), fx.Int32)
            next_parity_lane = old_parity ^ fx.Int32(1)
            previous_expected = _buffer_load(expected_rsrc, next_parity_lane, fx.Int32)
            next_expected = previous_expected + fx.Int32(fz_npes)
            _buffer_store(expected_rsrc, next_parity_lane, next_expected, fx.Int32)
            launch_epoch_lane = (
                (next_expected // fx.Int32(fz_npes)) * fx.Int32(2) - next_parity_lane
            )
        next_parity = fx.Int32(fx.rocdl.readfirstlane(T.i32, next_parity_lane))
        launch_epoch = fx.Int32(fx.rocdl.readfirstlane(T.i32, launch_epoch_lane))
        if const_expr(payload_tile_ready):
            if tid == fx.Int32(0):
                comm_ops.store_i32_system(a_payload_ready_rows, fx.Int32(0), fx.Int32(fz_tile_m))
                comm_ops.fence_system_release()
            fx.barrier()
        if tid < fx.Int32(fz_npes):
            peer = (tid + fx.Int32(fz_rank)) % fx.Int32(fz_npes)
            comm_ops.fence_system_release()
            launch_ready_table = _make_buffer_from_addr(p_launch_ready, fx.Int64)
            remote_launch_ready = _buffer_load(launch_ready_table, peer, fx.Int64)
            comm_ops.store_i32_system(remote_launch_ready, fx.Int32(fz_rank), launch_epoch)
            mori_shmem.int32_wait_until_greater_than(
                a_launch_ready + fx.Int64(peer) * fx.Int64(4), launch_epoch - fx.Int32(1)
            )
            comm_ops.fence_system_acquire()
        if tid == fx.Int32(0):
            work_head_rsrc = _make_buffer_from_addr(a_work_head, fx.Int32)
            for shard in range_constexpr(8):
                _buffer_store(work_head_rsrc, fx.Int32(shard * 16), fx.Int32(0), fx.Int32)
            _buffer_store(_make_buffer_from_addr(a_work_tail, fx.Int32), fx.Int32(0), fx.Int32(0), fx.Int32)
            if const_expr(external_grouping or direct_fixed_slot):
                group_done_rsrc = _make_buffer_from_addr(a_group_done, fx.Int32)
                for destination in range_constexpr(fz_npes if direct_fixed_slot else 1):
                    _buffer_store(group_done_rsrc, fx.Int32(destination), fx.Int32(0), fx.Int32)
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

注意这个 helper **不返回任何东西**。`next_parity` 和 `launch_epoch` 都是 `if compact_owner` 分支内部的局部量，出了分支就没人用；调用方接下来读的是 `parity_rsrc`/`expected_rsrc` 的内容，不是这两个 Python 名字。

- [ ] **Step 2: 在 `mega_moe_stage1.py` 替换**

把 `mega_moe_stage1.py` 从 `if compact_owner:`（原 227 行）到 `fx.barrier()`（原 279 行，`else:` 分支的最后一行）整段，替换为：

```python
        emit_epoch_rendezvous(
            tid=tid, compact_owner=compact_owner, parity_rsrc=parity_rsrc,
            expected_rsrc=expected_rsrc, a_payload_ready_rows=a_payload_ready_rows,
            p_launch_ready=p_launch_ready, a_launch_ready=a_launch_ready,
            a_work_head=a_work_head, a_work_tail=a_work_tail, a_group_done=a_group_done,
            gate_addr=gate_addr, gate_epoch=gate_epoch, fz_npes=fz_npes, fz_rank=fz_rank,
            fz_tile_m=fz_tile_m, payload_tile_ready=payload_tile_ready,
            external_grouping=external_grouping, direct_fixed_slot=direct_fixed_slot)
```

紧随其后的两行（原 280-282）保持不动：

```python
        payload_parity = _buffer_load(parity_rsrc, fx.Int32(0), fx.Int32, cache_modifier=_SC0_CACHE)
        payload_expected = _buffer_load(expected_rsrc, payload_parity, fx.Int32, cache_modifier=_SC0_CACHE)
```

- [ ] **Step 3: 更新 import**

把 `mega_moe_stage1.py` 的：

```python
from .collective_sched import emit_ticket_and_roles
```

改为：

```python
from .collective_sched import emit_epoch_rendezvous, emit_ticket_and_roles
```

- [ ] **Step 4: 跑守护**

```bash
cd /root/workspace/aiter
rm -rf ~/.flydsl/cache
PYTHONPATH=. torchrun --standalone --nproc_per_node=8 \
    op_tests/multigpu_tests/megamoe_v2_snapshot.py --out /tmp/mm_t6
python op_tests/multigpu_tests/flydsl_ir_fingerprint.py --out /tmp/mm_t6_ir.txt
python op_tests/multigpu_tests/megamoe_v2_snapshot.py --compare /tmp/mm_before /tmp/mm_t6
diff /tmp/mm_before_ir.txt /tmp/mm_t6_ir.txt && echo "IR UNCHANGED"
```

Expected：`IDENTICAL across 8 ranks` 和 `IR UNCHANGED`。

> 这个 task 最可能在这一步失败。若 `diff` 有输出，先怀疑两件事：`payload_tile_ready` 那段是不是被挪到了 parity 翻转和握手之间（原始顺序就是这样，不要「优化」）；`readfirstlane` 那两行是不是被移出了 helper。

- [ ] **Step 5: 提交**

```bash
cd /root/workspace/aiter
git add aiter/ops/flydsl/kernels/mega_moe/collective_sched.py \
        aiter/ops/flydsl/kernels/mega_moe/mega_moe_stage1.py
git commit -m "refactor(mega-moe): extract emit_epoch_rendezvous into collective_sched

The parity flip, launch_ready handshake and epoch_gate reset are one if/else
statement and move together -- splitting them would change the branch structure,
and the handshake consumes the launch_epoch the flip produces. The readfirstlane
pair stays with its nested if tid==0 so the SSA merge point does not move.

Verified MegaMoEV2 output is bit-identical and the generated IR is unchanged.

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 7: 抽取 `emit_work_pool_loop`

**Files:**
- Modify: `aiter/ops/flydsl/kernels/mega_moe/collective_sched.py`
- Modify: `aiter/ops/flydsl/kernels/mega_moe/mega_moe_stage1.py:425-453`

- [ ] **Step 1: 追加 helper**

```python
def emit_work_pool_loop(*, tid, a_buf, ticket, a_work_head, total_work,
        wait_tile_payload, do_scheduled_tile, work_shards, direct_fixed_slot):
    """Drain the sharded work pool until every tile is claimed.

    Each shard's head sits on its own 64-byte cache line, and the interleave
    ``work = shard + local*WORK_SHARDS`` covers [0, total_work) exactly once while
    spreading one M-tile's N-tiles across shards. ``work`` and ``has_work`` are
    broadcast to the block through the same LDS scratch byte the ticket used.

    ``wait_tile_payload`` and ``do_scheduled_tile`` are Python callables taking a
    single flat work id; passing them as parameters is IR-neutral because they
    are inlined at trace time.
    """
    consumer_active = fx.Int32(1) == fx.Int32(1)
    work_scratch = fx.recast_iter(fx.Int32, a_buf.ptr)
    work_scratch_view = fx.make_view(work_scratch, fx.make_layout(1, 1))
    work_shard = ticket & fx.Int32(work_shards - 1)
    while consumer_active:
        if tid == fx.Int32(0):
            local_work = fx.Int32(
                comm_ops.atomic_add_agent(
                    a_work_head + fx.Int64(work_shard) * fx.Int64(64), fx.Int32(1)
                )
            )
            work = work_shard + local_work * fx.Int32(work_shards)
            fx.ptr_store(Vec.from_elements([work], fx.Int32), work_scratch)
        fx.barrier()
        work = Vec(work_scratch_view.load())[0]
        if tid == fx.Int32(0):
            has_work = (work < total_work).select(fx.Int32(1), fx.Int32(0))
            if has_work != fx.Int32(0):  # noqa: SIM102 - keep the device and compile-time branches separate.
                if const_expr(not direct_fixed_slot):
                    wait_tile_payload(work)
            fx.ptr_store(Vec.from_elements([has_work], fx.Int32), work_scratch)
        fx.barrier()
        has_work = Vec(work_scratch_view.load())[0]
        if has_work != fx.Int32(0):
            if const_expr(not direct_fixed_slot):
                comm_ops.fence_system_acquire()
            do_scheduled_tile(work)
        consumer_active = has_work != fx.Int32(0)
```

`WORK_SHARDS` 在调用方是模块常量，这里改名为参数 `work_shards`，值一样，`fx.Int32(work_shards - 1)` 与原来的 `fx.Int32(WORK_SHARDS - 1)` 折叠出同一个常量。

- [ ] **Step 2: 在 `mega_moe_stage1.py` 替换**

把从 `# Control CTAs join the work pool after dispatch.`（原 425 行）到 `consumer_active = has_work != fx.Int32(0)`（原 453 行）整段替换为：

```python
        # Control CTAs join the work pool after dispatch.
        emit_work_pool_loop(
            tid=tid, a_buf=a_buf, ticket=ticket, a_work_head=a_work_head,
            total_work=total_work, wait_tile_payload=_wait_tile_payload,
            do_scheduled_tile=_do_scheduled_tile, work_shards=WORK_SHARDS,
            direct_fixed_slot=direct_fixed_slot)
```

- [ ] **Step 3: 更新 import**

```python
from .collective_sched import (
    emit_epoch_rendezvous,
    emit_ticket_and_roles,
    emit_work_pool_loop,
)
```

- [ ] **Step 4: 跑守护**

```bash
cd /root/workspace/aiter
rm -rf ~/.flydsl/cache
PYTHONPATH=. torchrun --standalone --nproc_per_node=8 \
    op_tests/multigpu_tests/megamoe_v2_snapshot.py --out /tmp/mm_t7
python op_tests/multigpu_tests/flydsl_ir_fingerprint.py --out /tmp/mm_t7_ir.txt
python op_tests/multigpu_tests/megamoe_v2_snapshot.py --compare /tmp/mm_before /tmp/mm_t7
diff /tmp/mm_before_ir.txt /tmp/mm_t7_ir.txt && echo "IR UNCHANGED"
```

Expected：`IDENTICAL across 8 ranks` 和 `IR UNCHANGED`。

- [ ] **Step 5: 提交**

```bash
cd /root/workspace/aiter
git add aiter/ops/flydsl/kernels/mega_moe/collective_sched.py \
        aiter/ops/flydsl/kernels/mega_moe/mega_moe_stage1.py
git commit -m "refactor(mega-moe): extract emit_work_pool_loop into collective_sched

The two per-tile callbacks are passed as plain Python callables, which is
IR-neutral because flyc.jit inlines at trace time. Verified MegaMoEV2 output is
bit-identical and the generated IR is unchanged.

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 8: 收尾验证

**Files:** 无新增改动，只跑验证。

- [ ] **Step 1: 全量数值回归**

```bash
cd /root/workspace/aiter
PYTHONPATH=. torchrun --standalone --nproc_per_node=8 \
    op_tests/multigpu_tests/test_mega_moe_v2.py --bs-list 1,8,128 2>&1 | tail -20
```

Expected：每个 batch size 一行 `[MEGA-V2] bs=N relL2=0.0xxxxx ...`，无 `AssertionError`，退出码 0。

- [ ] **Step 2: dispatch/combine 回归**

```bash
cd /root/workspace/aiter
PYTHONPATH=. torchrun --standalone --nproc_per_node=8 \
    op_tests/multigpu_tests/test_dispatch_combine.py 2>&1 | tail -10
```

Expected：退出码 0。若该脚本需要额外参数导致启动失败，记录实际报错并在本 task 的完成说明里写清楚，不要伪装成通过。

- [ ] **Step 3: 阶段一 TP 用例全绿**

```bash
cd /root/workspace/aiter
for c in construct capacity exports; do
  PYTHONPATH=. python op_tests/multigpu_tests/test_tp_moe_stage1.py --case $c 2>&1 | tail -1
done
for c in all_gather forward_contract numerics prequant e2e ref_fidelity; do
  echo "--- $c ---"
  PYTHONPATH=. torchrun --standalone --nproc_per_node=8 \
      op_tests/multigpu_tests/test_tp_moe_stage1.py --case $c 2>&1 | tail -1
done
```

Expected：九个用例全部以 `OK` 结尾。

- [ ] **Step 4: black 全量检查**

```bash
cd /root/workspace/aiter
python -m black --check \
    aiter/ops/flydsl/kernels/mega_moe/collective_sched.py \
    aiter/ops/flydsl/kernels/mega_moe/dispatch.py \
    aiter/ops/flydsl/kernels/mega_moe/mega_moe_stage1.py \
    op_tests/multigpu_tests/tp_moe_stage1_nccl_ref.py \
    op_tests/multigpu_tests/flydsl_ir_fingerprint.py \
    op_tests/multigpu_tests/megamoe_v2_snapshot.py \
    op_tests/multigpu_tests/test_tp_moe_stage1.py
```

Expected：`All done! ... N files would be left unchanged.`

CI 跑的是 `psf/black@stable`（`.github/workflows/pre-checks.yaml:28-35`），提交前必须过。

- [ ] **Step 5: 写完成报告**

在本方案文件末尾追加一节「执行记录」，写清楚：四次 IR 守护的实际结果、Task 3 Step 3 的基线可复现性结论、以及任何一处与方案预期不符的地方。不要只写「全部通过」，要贴命令输出。

---

## Self-Review

**Spec 覆盖：** spec 第 3.5 节的「NCCL 实现搬到 `op_tests`」由 Task 1 实现；「`TPMoEStage1Output` 共享」由 Task 1 Step 3 实现；「实施顺序：先复制，再长融合路径，最后删旧路径」——本方案只做第一步，后两步属于阶段二（下）。spec 第 6.1 节的四个 helper 由 Task 4 到 7 逐个实现；第 6.2 节的两条守护由 Task 2、3 提供工具，并在 Task 4 到 7 每一步执行。spec 第 5 节（kernel 内部结构）、第 7 节（融合测试）、第 8 节（性能验收）**不在本方案范围内**，属于阶段二（下）。

**未覆盖且有意为之：** `transport` 参数的删除、`TPMoEStage1` 里 NCCL 路径的删除。这两件事必须等融合路径能跑之后再做，否则中间会有一段算子不可用。

**类型一致性：** `TPMoEStage1Output` 在参照实现中通过 import 复用，不存在两个定义。`emit_ticket_and_roles` 返回六元组，`mega_moe_stage1.py` 按同样顺序解包。`emit_epoch_rendezvous` 无返回值，调用方随后从 `parity_rsrc` 读取，与原代码一致。`emit_work_pool_loop` 的 `work_shards` 参数对应调用方的 `WORK_SHARDS` 常量。

**已知风险：** Task 3 Step 3 可能发现 `MegaMoEV2` 的输出本身不可复现（atomic epilogue 的累加顺序随调度变化）。方案里已经写明这种情况要停下来报告并降级守护标准，而不是继续往下做。
