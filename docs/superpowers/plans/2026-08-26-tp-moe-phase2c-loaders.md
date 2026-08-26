# TP MoE 阶段二（下之一）：按 token id 取数的 A 与 scale loader Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 让 GEMM1 能直接从稠密的 rank-major activation buffer 里按 `sorted_token_ids` 取行，不再需要事先把行物理排列好，也不再需要把 scale 展开成排序后的 shuffle 布局。

**Architecture:** 两个新的 loader 类，复制 `gemm_util.py` 的 `ATileLoader` 与 `AScaleLoader`，只改地址计算：行基址从「tile 基址加连续行号」改成「按 token id 查表」。`do_tile` 原样 import 复用，230 行 MFMA 主循环一行不抄。验收靠一个物理对拍：把同一份数据在 host 上先排好交给原版 loader，与让新 loader 自己去 gather，两者输出必须**逐位相同**。

**Tech Stack:** Python 3.12、PyTorch、FlyDSL、ROCm gfx950。本方案是单卡的，不需要 torchrun。

**依据文档：** `docs/superpowers/specs/2026-08-26-tp-moe-stage1-fused-p2p-design.md` 第 2.3 节与 5.5 节。

**不在本方案范围内：** 推送与 GEMM1 的融合、`TPMoEStage1` 的改动、性能验收。那些在阶段二（下之二）。

**分支：** `dev/all_gather_merge_stage1_naive`。

---

## File Structure

| 文件 | 动作 | 职责 |
|---|---|---|
| `aiter/ops/flydsl/kernels/mega_moe/tp_gemm_util.py` | 新建 | `TPATileLoader`、`TPAScaleLoader`，按 token id 取数 |
| `aiter/ops/flydsl/kernels/mega_moe/tp_gemm1.py` | 新建 | `compile_tp_gemm1` / `run_tp_gemm1`：独立的 TP GEMM1 kernel |
| `op_tests/multigpu_tests/test_tp_gemm1.py` | 新建 | 逐位对拍与负对照 |

**新模块必须放在 `mega_moe/` 目录下。** FlyDSL 的 cache key 只收集与 `@flyc.jit` 启动器**同目录**的依赖源码（`jit_function.py::_is_user_function` 用 `os.path.dirname` 比较）。放到别处的话，`gemm1.py` 和 `gemm_util.py` 的源码不会进 cache key，将来它们一改就会命中陈旧缓存。

---

## 四个必须先理解的约束

这四条都不是从 API 表面能看出来的，写错了不会崩，会静默算错。

### 1. `load_step` 一条 VMEM 都不能发（最危险的一条）

`gemm1.py:152-156` 有一个硬编码的 vmcnt 屏障：

```python
if const_expr(async_a_copy):
    wait_lds_barrier(NUM_ACC_N * _PACK + NUM_B_SCALE)
```

它靠数「`prefetch_to_lds` 之后还会发多少条 VMEM 指令」来确认 A 的 global→LDS 搬运已经落地。这个计数成立的前提是：`b_scale.load_step` 发 `NUM_B_SCALE` 条，`b_loader.load_next` 发 `NUM_ACC_N * _PACK` 条，而 **`a_scale.load_step` 发零条**——因为它读的是 LDS 里 `stage()` 好的副本（`gemm_util.py:393-399` 是 `ds_read`）。

所以 `TPAScaleLoader` 必须保持「`stage()` 把整块搬进 LDS，`load_step()` 只读 LDS」的结构。**绝对不能**为了省事把 token 查表挪进 `load_step` 里直接读显存。那样屏障会提前放行，读到写了一半的 A tile，结果是随机错数。

### 2. `load_step` 的返回形状是硬约束

必须返回**恰好 `M_REPEAT // 2` 个标量 i32**。`gemm1.py:97-104` 和 `:184-191` 按这个数目解包循环携带的状态。多一个少一个都会让 `sa` 和 `sb` 的槽位错位，不报错，直接算错。原版 `AScaleLoader.load_step` 返回 `self._n_groups = m_repeat // _PACK` 个。

### 3. `__init__` 与 `for_tile` 的分工不能乱

`ATileLoader.__init__` 建的是与 tile 无关的东西（`self._tx`、`self._dma_atom`），它们的 SSA 值产生在持久化 tile 循环**之外**；`for_tile` 建的是每个 tile 各一份的。任何依赖 `tile_row_base` 的东西——包括 token 查表——必须放在 `for_tile` 里，放进 `__init__` 会在循环外只算一次，全部 tile 共用第一个 tile 的地址。

### 4. LDS 布局一个字节都不能动

`AS2RLoader.load_operand`（`gemm_util.py:233-251`）按「LDS 行号即 tile 内槽位」读数，它的 XOR swizzle 与 `ATileLoader.store` 必须逐位匹配（`gemm_util.py:239` 的注释写明了）。本方案只改**源地址**，LDS 里放什么、放在哪，与原版完全一致。`TPAScaleLoader.stage` 写进 LDS 的仍然是 `[sort_block_m, model_dim/32]` 的 row-major 块。

---

## 两个新 loader 相对原版的差异

### `TPATileLoader`

原版 `for_tile`（`gemm_util.py:112-127`）把 buffer resource 的基址设成 `x + tile_row_base * row_bytes`，`num_records` 只有 `sort_block_m * row_bytes`——一个只覆盖本 tile 的窗口。gather 之后行散落在整个 A 里，这个窗口不成立。

改法是：**resource 覆盖整个 A 张量**，每行的偏移用 token id 算。这不是可选项，是硬约束：buffer resource 住在 SGPR 里、是 wave-uniform 的，而 token id 是 per-lane 的，所以不可能给每行建一个 resource，只能一个宽 resource 加 per-lane 的 32 位偏移。

代价是 `num_records` 是 32 位字节数，所以 `total_rows * model_dim` 必须小于 4 GiB。`model_dim=7168` 时约 60 万行，远够用，但要断言。

`gemm_util.py:144-146` 的

```python
row = lin // fx.Int32(chunks_per_row)
row_byte = row * fx.Int32(self._row_bytes)
```

改成先查表再算：

```python
row = lin // fx.Int32(chunks_per_row)
tok = _buffer_load(self._tok_rsrc, tile_row_base_i32 + row, fx.Int32)
tok = (tok & fx.Int32(0x00FFFFFF)).min(fx.Int32(self._pad_row))
row_byte = tok * fx.Int32(self._row_bytes)
```

`& 0x00FFFFFF` 是因为 `sorted_token_ids` 的高 8 位存的是 topk slot。`min(pad_row)` 把填充哨兵（token id 等于 `m_global`）钳到 A 末尾那一行清零行，这样 PAD 行的乘积自然是零。

默认配置下 `chunks_per_row = 256/16 = 16`，`total_chunks = 32*16 = 512`，`total_threads = 256`，所以每个线程多两次 `buffer_load`，而且在 `for_tile` 里，每个 tile 只做一次，不进 K 循环。

`prefetch_to_lds`（`async_a_copy=True` 时才用）同理，但它的 `row` 在 K 循环内部算。**必须在 `for_tile` 里把每线程的 token id 预先算好存在 `self` 上**，否则每个 K step 都会重查一次。

### `TPAScaleLoader`

原版 `stage`（`gemm_util.py:335-366`）是一次**扁平的线性拷贝**：`base = tile_row_base * n_scale`，每个线程搬第 `lin` 个 16 字节块。整段代码里没有行的概念，一个 16 字节块可以跨两行。

gather 之后源行不连续，所以必须改成二维分解：

```python
chunks_per_row = n_scale // 16
row = lin // chunks_per_row
cir = lin - row * chunks_per_row
tok = _buffer_load(self._tok_rsrc, tile_row_base_i32 + row, fx.Int32)
tok = (tok & fx.Int32(0x00FFFFFF)).min(fx.Int32(self._pad_row))
src_group = (tok * fx.Int32(self._n_scale) + cir * fx.Int32(16)) // fx.Int32(16)
```

LDS 目标地址**不变**，仍然是 `lin * 16`，等价于 `row * n_scale + cir * 16`。

这带来一条新的约束：**`n_scale % 16 == 0`，即 `model_dim % 512 == 0`。** 原版只要求 `sort_block_m * n_scale % 16 == 0`。`model_dim=7168` 时 `n_scale = 224 = 14 * 16`，成立，但要写断言。`chunks_per_row = 14`，`total = 32 * 14 = 448`，与原版一样是 448，所以原版那个带谓词的 `copy_chunk` 结构（448 不能被 256 或 512 整除，才需要它）原样保留。

`sx_rsrc` 本来就是全张量描述符（`max_size=True`），这一侧不用改。

`load_step`、`_read_scale_lds`、i32 打包、以及下游全部不动。

---

## Task 1: 两个 loader

**Files:**
- Create: `aiter/ops/flydsl/kernels/mega_moe/tp_gemm_util.py`

- [ ] **Step 1: 建文件**

```python
# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""A-operand loaders that gather rows by token id instead of reading them contiguously.

MegaMoE's EP dispatch physically permutes rows into expert-major order as a side
effect of pushing them, so its loaders read 32 contiguous rows starting at a tile
base. TP's all-gather produces a dense rank-major buffer and expresses the
permutation as an index list, so the loaders have to gather.

These are copies of gemm_util.ATileLoader / AScaleLoader with the address
computation changed and nothing else. gemm_util.py is frozen; see section 6 of
docs/superpowers/specs/2026-08-26-tp-moe-stage1-fused-p2p-design.md.

FOUR CONSTRAINTS THAT ARE NOT VISIBLE IN THE API AND FAIL SILENTLY:

1. ``load_step`` MUST issue zero VMEM instructions. gemm1.py:152-156 has a
   hardcoded ``wait_lds_barrier(NUM_ACC_N * _PACK + NUM_B_SCALE)`` that proves
   the A prefetch DMAs retired by counting the VMEM issued after them. That count
   assumes the A-scale loader reads LDS, not memory. Move the token lookup into
   load_step and the barrier releases early, giving half-written A tiles and
   silently wrong numbers.
2. ``load_step`` must return EXACTLY ``m_repeat // 2`` scalar i32 values.
   gemm1.py:97-104 unpacks the loop-carried state by that count.
3. Anything depending on ``tile_row_base`` belongs in ``for_tile`` / ``stage``,
   never ``__init__``: __init__ runs outside the persistent tile loop.
4. The LDS layout is unchanged. AS2RLoader.load_operand reads LDS by tile-local
   slot and its XOR swizzle must match store() bit for bit.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, range_constexpr
from flydsl.expr.typing import Vector as Vec

from .gemm_util import _PACK, _buffer_load, _make_buffer


class TPATileLoader:
    """Dense rank-major A rows gathered by token id, gmem->reg->LDS."""

    def __init__(self, *, row_bytes, sort_block_m, k_step_bytes, total_threads,
                 swizzle=False, x_tensor=None, tok_rsrc=None, pad_row=None,
                 total_rows=None, async_copy=False):
        assert x_tensor is not None and tok_rsrc is not None
        assert pad_row is not None and total_rows is not None
        # num_records is a 32-bit byte count, so the whole A tensor must fit.
        assert total_rows * row_bytes < (1 << 32), (
            f"A tensor {total_rows * row_bytes} bytes exceeds the 32-bit buffer "
            "num_records; a per-row descriptor is impossible because buffer "
            "resources are wave-uniform (SGPR) and token ids are per-lane"
        )
        self._sort_block_m = sort_block_m
        self._k_step_bytes = k_step_bytes
        self._total_threads = total_threads
        self._swizzle = swizzle
        self._row_bytes = row_bytes
        self._tx = fx.thread_idx.x
        self._wave = self._tx // 64
        self._x_tensor = x_tensor
        self._tok_rsrc = tok_rsrc
        self._pad_row = int(pad_row)
        self._async_copy = bool(async_copy)
        # One wide descriptor over the whole A tensor, built once.
        self._rsrc = _make_buffer(
            x_tensor, fx.Int32, 4, max_size=False,
            num_records_bytes=total_rows * row_bytes,
        )
        if const_expr(self._async_copy):
            assert total_threads % 64 == 0
            assert (sort_block_m * 16) % total_threads == 0
            assert row_bytes % 16 == 0 and k_step_bytes % 16 == 0
            self._dma = fx.rocdl.make_buffer_tensor(
                fx.Tensor(fx.make_view(
                    fx.get_iter(x_tensor),
                    fx.make_layout(total_rows * row_bytes, 1))),
                max_size=False, num_records_bytes=total_rows * row_bytes,
            )
            self._tile_dma = fx.logical_divide(self._dma, fx.make_layout(1, 1))

    def _row_of(self, tile_row_base_i32, row_i32):
        """sorted slot -> dense A row. Padding slots clamp to the zeroed pad row."""
        tok = _buffer_load(self._tok_rsrc, tile_row_base_i32 + row_i32, fx.Int32)
        tok = tok & fx.Int32(0x00FFFFFF)
        return tok.min(fx.Int32(self._pad_row))

    def for_tile(self, tile_row_base_i32):
        """Precompute this tile's per-thread global byte offsets and LDS slots."""
        chunks_per_row = self._k_step_bytes // 16
        row_stride_i32 = self._k_step_bytes // 4
        total_chunks = self._sort_block_m * chunks_per_row
        self._chunks = []
        for c in range_constexpr(0, total_chunks, self._total_threads):
            lin = fx.Int32(c) + fx.Int32(self._tx)
            row = lin // fx.Int32(chunks_per_row)
            chunk = lin - row * fx.Int32(chunks_per_row)
            row_byte = self._row_of(tile_row_base_i32, row) * fx.Int32(self._row_bytes)
            if const_expr(self._swizzle):
                col_i32 = chunk * fx.Int32(4)
                swz = row * fx.Int32(row_stride_i32) + (
                    col_i32 ^ ((row & fx.Int32(15)) << fx.Int32(2))
                )
                lds_byte = swz * fx.Int32(4)
            else:
                lds_byte = lin * fx.Int32(16)
            self._chunks.append((lds_byte, row_byte + chunk * fx.Int32(16)))
        if const_expr(self._async_copy):
            # Hoisted out of the K loop on purpose: row/token are K-independent,
            # and re-reading the token table per K step would both cost VMEM and
            # perturb the hardcoded vmcnt in gemm1.py:152-156.
            self._dma_rows = []
            total = self._sort_block_m * 16
            for round_base in range_constexpr(0, total, self._total_threads):
                physical = fx.Int32(round_base) + fx.Int32(self._tx)
                row = physical // fx.Int32(16)
                physical_chunk = physical - row * fx.Int32(16)
                if const_expr(self._swizzle):
                    logical_chunk = physical_chunk ^ (row & fx.Int32(15))
                else:
                    logical_chunk = physical_chunk
                src_row_byte = self._row_of(tile_row_base_i32, row) * fx.Int32(self._row_bytes)
                self._dma_rows.append((round_base, src_row_byte, logical_chunk))

    def load_regs(self, k_step_byte_off):
        """Read this K-step's 16-byte chunks gmem->reg. Only the K offset varies."""
        koff = fx.Int32(k_step_byte_off)
        regs = []
        for lds_byte, chunk_base in self._chunks:
            group = (chunk_base + koff) // fx.Int32(16)
            regs.append((lds_byte, _buffer_load(self._rsrc, group, fx.Int32, 4)))
        return regs

    def store(self, lds_dst, regs, base_i32=0):
        """Scatter loaded chunks into LDS. Byte-identical to ATileLoader.store."""
        base_bytes = fx.Int32(base_i32) * fx.Int32(4)
        for lds_byte, v in regs:
            dst = fx.make_view(
                fx.add_offset(
                    fx.recast_iter(fx.Int32, lds_dst.ptr),
                    (base_bytes + lds_byte) // fx.Int32(4),
                ),
                fx.make_layout(4, 1),
            )
            fragment = fx.make_rmem_tensor(4, fx.Int32)
            fragment.store(Vec(v))
            fx.copy(fx.make_copy_atom(fx.UniversalCopy128b(), fx.Int32), fragment, dst)

    def prefetch_to_lds(self, k_step_byte_off, lds_dst, base_i32=0):
        """Direct global->LDS copies using the token ids cached by for_tile."""
        koff = fx.Int32(k_step_byte_off)
        base_bytes = fx.Int32(base_i32) * fx.Int32(4)
        lds_f8 = fx.recast_iter(fx.Float8E4M3FN, lds_dst.ptr)
        for round_base, src_row_byte, logical_chunk in self._dma_rows:
            src_byte = src_row_byte + koff + logical_chunk * fx.Int32(16)
            src = fx.slice(self._tile_dma, (None, src_byte))
            wave_base = base_bytes + fx.Int32((round_base + self._wave * 64) * 16)
            dst = fx.make_view(fx.add_offset(lds_f8, wave_base), fx.make_layout(1, 1))
            fx.copy(self._dma_atom, src, dst)


class TPAScaleLoader:
    """Per-1x32 E8M0 A scales gathered by token id, staged to LDS once per tile."""

    def __init__(self, *, scale_rsrc, m_repeat, model_dim, sort_block_m,
                 total_threads, tok_rsrc, pad_row):
        # stage() decomposes into (row, chunk-in-row), so a row must be a whole
        # number of 16-byte chunks. The original was a flat linear copy and only
        # needed sort_block_m*n_scale % 16 == 0.
        assert (model_dim // 32) % 16 == 0, (
            f"model_dim={model_dim} gives n_scale={model_dim // 32}, which is not "
            "a multiple of 16; the gathered stage() needs whole chunks per row"
        )
        self._rsrc = scale_rsrc
        self._n_scale = model_dim // 32
        self._lane = fx.thread_idx.x % 64
        self._n_groups = m_repeat // _PACK
        self._sort_block_m = sort_block_m
        self._total_threads = total_threads
        self._tx = fx.thread_idx.x
        self._tok_rsrc = tok_rsrc
        self._pad_row = int(pad_row)

    def stage(self, lds_ascale, tile_row_base_i32):
        """Gather this tile's e8m0 rows into LDS as a [sort_block_m, n_scale] block.

        Row-decomposed, unlike the contiguous original: source rows are scattered
        but the LDS destination is identical, so everything downstream is unchanged.
        """
        chunks_per_row = self._n_scale // 16
        total_chunks = self._sort_block_m * chunks_per_row

        @flyc.jit
        def copy_chunk(lin: fx.Int32):
            if lin < fx.Int32(total_chunks):
                row = lin // fx.Int32(chunks_per_row)
                cir = lin - row * fx.Int32(chunks_per_row)
                tok = _buffer_load(self._tok_rsrc, tile_row_base_i32 + row, fx.Int32)
                tok = (tok & fx.Int32(0x00FFFFFF)).min(fx.Int32(self._pad_row))
                src_group = (tok * fx.Int32(self._n_scale) + cir * fx.Int32(16)) // fx.Int32(16)
                v = _buffer_load(self._rsrc, src_group, fx.Int32, 4)
                dst = fx.make_view(
                    fx.add_offset(fx.recast_iter(fx.Int32, lds_ascale.ptr), lin * fx.Int32(4)),
                    fx.make_layout(4, 1),
                )
                fragment = fx.make_rmem_tensor(4, fx.Int32)
                fragment.store(v)
                fx.copy(fx.make_copy_atom(fx.UniversalCopy128b(), fx.Int32), fragment, dst)

        for c in range_constexpr(0, total_chunks, self._total_threads):
            # 32*14 = 448 chunks does not divide a 256/512-thread CTA, hence the
            # predicate inside copy_chunk. Same as the original.
            copy_chunk(fx.Int32(c) + fx.Int32(self._tx))

    def load_step(self, lds_ascale, kstep_i32):
        """One packed i32 per pack-group, read from LDS. MUST issue zero VMEM."""
        lane_row = fx.Int32(self._lane % 16)
        col0 = kstep_i32 * fx.Int32(8) + fx.Int32(self._lane // 16)
        out = []
        for g in range_constexpr(self._n_groups):
            r0 = fx.Int32(g * 32) + lane_row
            r1 = r0 + fx.Int32(16)
            b = []
            for ksub in range_constexpr(_PACK):
                for rr in (r0, r1):
                    b.append(self._read_scale_lds(lds_ascale, rr, col0 + fx.Int32(ksub * 4)))
            out.append(b[0] | (b[1] << fx.Int32(8)) | (b[2] << fx.Int32(16)) | (b[3] << fx.Int32(24)))
        return out

    def _read_scale_lds(self, lds_ascale, row_i32, col_i32):
        off = row_i32 * fx.Int32(self._n_scale) + col_i32
        ptr = fx.recast_iter(fx.Uint8, fx.add_offset(lds_ascale.ptr, fx.make_int_tuple(off)))
        v = fx.make_view(ptr, fx.make_layout(1, 1)).load()
        return Vec(v, dtype=fx.Uint8)[0].to(fx.Int32)
```

- [ ] **Step 2: 确认能 import 且没碰冻结文件**

```bash
cd /root/workspace/aiter
PYTHONPATH=. python -c "
from aiter.ops.flydsl.kernels.mega_moe.tp_gemm_util import TPATileLoader, TPAScaleLoader
from aiter.ops.flydsl.kernels.mega_moe.gemm_util import ATileLoader, AScaleLoader
import inspect
for tp, orig, names in ((TPATileLoader, ATileLoader, ('for_tile','load_regs','store','prefetch_to_lds')),
                        (TPAScaleLoader, AScaleLoader, ('stage','load_step','_read_scale_lds'))):
    for n in names:
        a = inspect.signature(getattr(tp, n)); b = inspect.signature(getattr(orig, n))
        assert str(a) == str(b), (n, str(a), str(b))
    print(tp.__name__, 'duck-types', orig.__name__, '->', names)
"
git diff --stat -- aiter/ops/flydsl/kernels/mega_moe/gemm_util.py aiter/ops/flydsl/kernels/mega_moe/gemm1.py
```

Expected：两行 `duck-types`，`git diff --stat` **无输出**。

方法签名必须逐字相同，因为 `do_tile` 是按位置调用它们的。

- [ ] **Step 3: 提交**

```bash
cd /root/workspace/aiter
python -m black --check aiter/ops/flydsl/kernels/mega_moe/tp_gemm_util.py
git add aiter/ops/flydsl/kernels/mega_moe/tp_gemm_util.py
git commit -m "feat(tp-moe): A and A-scale loaders that gather rows by token id

TP's all-gather leaves activations dense and rank-major and expresses the MoE
permutation as an index list, so the loaders have to gather rather than read 32
contiguous rows. Address computation is the only change; the LDS layout, the
MFMA operand assembly and everything downstream are untouched.

The A descriptor now spans the whole tensor because buffer resources are
wave-uniform and token ids are per-lane, so a per-row descriptor is impossible.

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Task 2: TP GEMM1 kernel 与逐位对拍

**Files:**
- Create: `aiter/ops/flydsl/kernels/mega_moe/tp_gemm1.py`
- Create: `op_tests/multigpu_tests/test_tp_gemm1.py`

### 对拍怎么做

不能拿 `flydsl_moe_stage1` 当参照，那是另一套 kernel（`mixed_moe_gemm_2stage_common`），累加顺序不同，只能设容差。

改成**物理对拍**，这样能要求逐位相同：同一个 `tp_gemm1.py` 编译出两个 kernel，唯一区别是 loader。

- **参照**：host 先把 A 和 scale 按 `sorted_token_ids` 物理排好（`a_perm = a_dense[tok]`），交给**原版** `ATileLoader` / `AScaleLoader`，`trb_rsrc` 传 `[0, 32, 64, ...]`。
- **被测**：A 和 scale 保持稠密，交给 `TPATileLoader` / `TPAScaleLoader`，同一张 `trb_rsrc`，外加 `sorted_token_ids`。

同一个 `do_tile`、同一套 MFMA、同一个 epilogue，只有取数地址不同，所以输出必须**逐位相同**。任何差异都只可能来自 loader。

- [ ] **Step 1: 写 kernel 模块**

创建 `aiter/ops/flydsl/kernels/mega_moe/tp_gemm1.py`。它 fork `build_fused_gemm1`（`gemm1.py:269-333`）的构造部分，其中只有两处 loader 构造要改；`do_tile` 直接 import。

用 `gather: bool` 这个编译期常量选 loader，这样参照与被测走同一份代码。

```python
# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Standalone TP MoE GEMM1: dense rank-major A gathered by sorted_token_ids.

Forks only the builder half of gemm1.build_fused_gemm1 -- the 230-line do_tile
MFMA pipeline is imported and reused unmodified, because it takes the loaders as
parameters and never touches their internals.

``gather=False`` selects the original contiguous loaders. That mode exists so
the test can run the identical kernel over host-permuted data and require the
two to agree bit for bit; it is not a production path.
"""

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr
from flydsl.runtime.device import get_rocm_arch

from ..tensor_shim import _run_compiled
from .gemm1 import _LdsF32View, do_tile
from .gemm_util import (
    AScaleLoader,
    AS2RLoader,
    ATileLoader,
    BScaleLoader,
    BWeightLoader,
    MfmaScaleGU,
    SiluQuantEpilogue,
    TileScheduler,
    _buffer_load,
    _make_buffer,
)
from .tp_gemm_util import TPAScaleLoader, TPATileLoader


@functools.cache
def compile_tp_gemm1(*, model_dim: int, inter_dim: int, experts: int, total_rows: int,
        gather: bool = True, sort_block_m: int = 32, tile_n: int = 256, tile_k: int = 256,
        num_waves: int = 4, num_cu: int = 256, grid_mult: int = 4,
        swizzle_a: bool = True, pipe_weights: bool = True, mfma_amajor: bool = False,
        async_a_copy: bool = False, waves_per_eu_hint: int = 2, swiglu_limit: float = 0.0):
    arch = get_rocm_arch()
    if not str(arch).startswith("gfx95"):
        raise RuntimeError(f"tp_gemm1 targets gfx95x, got {arch}")

    NUM_WAVES = num_waves
    TOTAL_THREADS = NUM_WAVES * 64
    n_per_wave = tile_n // NUM_WAVES
    N_TILES = (2 * inter_dim) // tile_n
    M_REPEAT = sort_block_m // 16
    NUM_ACC_N = n_per_wave // 16
    assert NUM_ACC_N % 2 == 0, "NUM_ACC_N must be even"
    A_K_STEP_BYTES = tile_k
    K_ITERS = model_dim // tile_k
    a_lds_size = sort_block_m * A_K_STEP_BYTES
    a_lds_i32 = a_lds_size // 4
    cs_tile_n = tile_n // 2
    lds_pool_bytes = max(2 * a_lds_size, sort_block_m * cs_tile_n * 4)
    n_scale_bytes = sort_block_m * (model_dim // 32)
    PAD_ROW = total_rows - 1  # last row of A is the zeroed padding row

    @fx.struct
    class SharedStorage:
        pool: fx.Array[fx.Int8, lds_pool_bytes, 16]
        A_scale: fx.Array[fx.Int8, n_scale_bytes, 16]

    kernel_name = (
        f"tp_gemm1_{'gather' if gather else 'contig'}"
        f"_t{sort_block_m}x{tile_n}x{tile_k}_w{NUM_WAVES}_d{model_dim}_i{inter_dim}"
    )

    @flyc.kernel(name=kernel_name, known_block_size=[TOTAL_THREADS, 1, 1])
    def tp_gemm1_kernel(
        out: fx.Tensor, out_scale: fx.Tensor, x: fx.Tensor, scale_x: fx.Tensor,
        w: fx.Tensor, scale_w: fx.Tensor, tile_row_base: fx.Tensor,
        expert_ids: fx.Tensor, sorted_token_ids: fx.Tensor, num_valid_ids: fx.Tensor,
        tokens: fx.Int32,
    ):
        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        a_buf = lds.pool
        a_scale_lds = lds.A_scale
        c_tile = _LdsF32View(fx.recast_iter(fx.Float32, lds.pool.ptr))
        wave_id = fx.thread_idx.x // fx.Int32(64)

        w_rsrc = _make_buffer(w, fx.Int32, 4)
        sw_rsrc = _make_buffer(scale_w, fx.Int32)
        sx_rsrc = _make_buffer(scale_x, fx.Int32, 4)
        trb_rsrc = _make_buffer(tile_row_base, fx.Int32)
        expert_rsrc = _make_buffer(expert_ids, fx.Int32)
        tok_rsrc = _make_buffer(sorted_token_ids, fx.Int32)
        nv_rsrc = _make_buffer(num_valid_ids, fx.Int32)
        scale_cols = (inter_dim // 32 + 7) // 8 * 8
        os_nbytes = tokens * fx.Int32(scale_cols) + fx.Int32(8192)
        os_rsrc = _make_buffer(out_scale, fx.Int8, max_size=False, num_records_bytes=os_nbytes)

        sched = TileScheduler(expert_rsrc=expert_rsrc, inter_dim=inter_dim, expert_offset=0)
        n_wave_base = wave_id * fx.Int32(n_per_wave)

        if const_expr(gather):
            a_gather = TPATileLoader(
                row_bytes=model_dim, sort_block_m=sort_block_m,
                k_step_bytes=A_K_STEP_BYTES, total_threads=TOTAL_THREADS,
                swizzle=swizzle_a, x_tensor=x, tok_rsrc=tok_rsrc,
                pad_row=PAD_ROW, total_rows=total_rows, async_copy=async_a_copy)
            a_scale = TPAScaleLoader(
                scale_rsrc=sx_rsrc, m_repeat=M_REPEAT, model_dim=model_dim,
                sort_block_m=sort_block_m, total_threads=TOTAL_THREADS,
                tok_rsrc=tok_rsrc, pad_row=PAD_ROW)
        else:
            a_gather = ATileLoader(
                row_bytes=model_dim, sort_block_m=sort_block_m,
                k_step_bytes=A_K_STEP_BYTES, total_threads=TOTAL_THREADS,
                swizzle=swizzle_a, x_tensor=x, async_copy=async_a_copy)
            a_scale = AScaleLoader(
                scale_rsrc=sx_rsrc, m_repeat=M_REPEAT, model_dim=model_dim,
                sort_block_m=sort_block_m, total_threads=TOTAL_THREADS)

        a_s2r = AS2RLoader(k_step_bytes=A_K_STEP_BYTES, swizzle=swizzle_a)
        b_loader = BWeightLoader(
            w_rsrc=w_rsrc, model_dim=model_dim, n_per_wave=n_per_wave,
            num_acc_n=NUM_ACC_N, k_step_bytes=A_K_STEP_BYTES, cache_modifier=0)
        b_scale = BScaleLoader(
            sw_rsrc=sw_rsrc, model_dim=model_dim, n_per_wave=n_per_wave,
            num_acc_n=NUM_ACC_N)
        mfma = MfmaScaleGU(num_acc_n=NUM_ACC_N, m_repeat=M_REPEAT)
        epi = SiluQuantEpilogue(
            out_rsrc=None, out_scale_rsrc=os_rsrc, sorted_rsrc=trb_rsrc,
            c_tile=c_tile, inter_dim=inter_dim, sort_block_m=sort_block_m,
            tile_n=tile_n, n_per_wave=n_per_wave, num_acc_n=NUM_ACC_N,
            wave_id=wave_id, always_valid=True, out_tensor=out,
            swiglu_limit=swiglu_limit)

        num_valid = _buffer_load(nv_rsrc, fx.Int32(0), fx.Int32)
        num_m_tiles = (num_valid + fx.Int32(sort_block_m - 1)) // fx.Int32(sort_block_m)
        total_work = num_m_tiles * fx.Int32(N_TILES)

        flat = fx.block_idx.x
        while flat < total_work:
            m_tile = flat // fx.Int32(N_TILES)
            n_tile = flat - m_tile * fx.Int32(N_TILES)
            n_tile_base = n_wave_base + n_tile * fx.Int32(tile_n)
            expert = sched.expert_of(m_tile)
            do_tile(m_tile, n_tile_base, expert, sched, a_gather, a_s2r, b_loader,
                b_scale, a_scale, mfma, epi, a_buf, a_scale_lds, a_lds_i32,
                K_ITERS, M_REPEAT, NUM_ACC_N, A_K_STEP_BYTES, pipe_weights,
                mfma_amajor, async_a_copy, trb_rsrc)
            flat = flat + fx.Int32(num_cu * grid_mult)

    @flyc.jit
    def launch(out: fx.Tensor, out_scale: fx.Tensor, x: fx.Tensor, scale_x: fx.Tensor,
            w: fx.Tensor, scale_w: fx.Tensor, tile_row_base: fx.Tensor,
            expert_ids: fx.Tensor, sorted_token_ids: fx.Tensor,
            num_valid_ids: fx.Tensor, tokens: fx.Int32, stream: fx.Stream):
        tp_gemm1_kernel(
            out, out_scale, x, scale_x, w, scale_w, tile_row_base, expert_ids,
            sorted_token_ids, num_valid_ids, tokens,
            value_attrs={
                "rocdl.waves_per_eu": waves_per_eu_hint,
                "rocdl.flat_work_group_size": f"{TOTAL_THREADS},{TOTAL_THREADS}",
            },
        ).launch(grid=(num_cu * grid_mult, 1, 1), block=(TOTAL_THREADS, 1, 1), stream=stream)

    return launch
```

**注意 `always_valid=True` 不能改。** `SiluQuantEpilogue` 把 `sorted_rsrc` 当成「按槽位索引的 token 表」读（`gemm_util.py:662`），而我们传进去的 `trb_rsrc` 是「按 tile 索引的行基址表」，两种语义不兼容。只因为 `always_valid=True` 让那段成为死代码才安全。改成 `False` 会用行槽位去索引一张长度只有 `n_tiles` 的表，读到越界内存。

**`trb_rsrc` 用的是 `max_size=True`，没有硬件越界钳位**，所以 host 分配的 `tile_row_base` 表必须至少有 `num_m_tiles` 个 int32。

- [ ] **Step 2: 写 host 入口**

在 `tp_gemm1.py` 末尾加：

```python
def run_tp_gemm1(*, x, scale_x, w, scale_w, tile_row_base, expert_ids,
        sorted_token_ids, num_valid_ids, max_sorted, model_dim, inter_dim,
        experts, total_rows, gather=True, sort_block_m=32, swiglu_limit=0.0,
        stream=None, **cfg):
    """Allocate outputs and launch. Returns (payload_fp8, packed_mx_scale)."""
    import torch

    dev = x.device
    out = torch.empty((max_sorted, inter_dim), dtype=torch.float8_e4m3fn, device=dev)
    prows = ((max_sorted + 255) // 256) * 256
    pcols = (((inter_dim // 32) + 7) // 8) * 8
    out_scale = torch.zeros(prows * pcols + inter_dim, dtype=torch.uint8, device=dev)
    launch = compile_tp_gemm1(
        model_dim=model_dim, inter_dim=inter_dim, experts=experts,
        total_rows=total_rows, gather=gather, sort_block_m=sort_block_m,
        swiglu_limit=swiglu_limit, **cfg)
    if stream is None:
        stream = fx.Stream(torch.cuda.current_stream())
    _run_compiled(
        launch, out, out_scale, x.view(torch.uint8), scale_x, w.view(torch.uint8),
        scale_w, tile_row_base, expert_ids, sorted_token_ids, num_valid_ids,
        fx.Int32(int(max_sorted)), stream)
    return out, out_scale
```

- [ ] **Step 3: 写对拍测试**

创建 `op_tests/multigpu_tests/test_tp_gemm1.py`。单进程，不需要 torchrun。

```python
# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""TP GEMM1 loader equivalence. Single process:

    python op_tests/multigpu_tests/test_tp_gemm1.py --case equiv
"""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from aiter.fused_moe import moe_sorting  # noqa: E402
from aiter.ops.flydsl.kernels.mega_moe.quant import per_1x32_mx_quant  # noqa: E402
from aiter.ops.flydsl.kernels.mega_moe.tp_gemm1 import run_tp_gemm1  # noqa: E402
from tp_moe_stage1_ref import build_mxfp4_w1  # noqa: E402

MODEL_DIM, EXPERTS, TOPK, INTER = 7168, 384, 6, 384
SBM = 32


def _build(m_global, seed, device):
    g = torch.Generator(device="cpu").manual_seed(seed)
    ids = torch.stack(
        [torch.randperm(EXPERTS, generator=g)[:TOPK] for _ in range(m_global)]
    ).to(device=device, dtype=torch.int32)
    w = torch.rand((m_global, TOPK), generator=g).to(device=device, dtype=torch.float32)
    w = w / w.sum(-1, keepdim=True)
    sids, sw, seid, nv, _ = moe_sorting(
        ids, w, EXPERTS, MODEL_DIM, torch.bfloat16, block_size=SBM, accumulate=False
    )
    x = torch.randn((m_global, MODEL_DIM), generator=g).to(
        device=device, dtype=torch.bfloat16
    ) * (MODEL_DIM**-0.25)
    x_q, x_s = per_1x32_mx_quant(x, quant_mode="fp8")
    # One zeroed padding row at the end; gathered padding slots clamp to it.
    a = torch.cat([x_q.view(torch.uint8), torch.zeros_like(x_q[:1].view(torch.uint8))], 0)
    s = torch.cat([x_s, torch.zeros_like(x_s[:1])], 0)
    return sids, sw, seid, nv, a.contiguous(), s.contiguous()


def case_equiv():
    device = torch.device("cuda", 0)
    _, _, w1_shuf, w1_scale_shuf = build_mxfp4_w1(EXPERTS, INTER, MODEL_DIM, device, seed=99)

    for m_global in (8, 64, 512, 1024):
        sids, _, seid, nv, a_dense, s_dense = _build(m_global, 1000 + m_global, device)
        nvalid = int(nv[0].item())
        n_tiles_m = nvalid // SBM
        max_sorted = ((sids.shape[0] + SBM - 1) // SBM) * SBM
        total_rows = a_dense.shape[0]
        pad_row = total_rows - 1

        trb = (torch.arange(n_tiles_m + 64, dtype=torch.int32, device=device) * SBM)

        # Reference: permute on the host, feed the ORIGINAL contiguous loaders.
        tok = (sids[:nvalid] & 0x00FFFFFF).clamp(max=pad_row).long()
        a_perm = a_dense[tok].contiguous()
        s_perm = s_dense[tok].contiguous()
        pad = (-a_perm.shape[0]) % SBM
        if pad:
            a_perm = torch.cat([a_perm, a_perm[:1].expand(pad, -1)], 0).contiguous()
            s_perm = torch.cat([s_perm, s_perm[:1].expand(pad, -1)], 0).contiguous()
        ref_ids = torch.arange(a_perm.shape[0], dtype=torch.int32, device=device)
        out_ref, os_ref = run_tp_gemm1(
            x=a_perm, scale_x=s_perm, w=w1_shuf, scale_w=w1_scale_shuf,
            tile_row_base=trb, expert_ids=seid, sorted_token_ids=ref_ids,
            num_valid_ids=nv, max_sorted=max_sorted, model_dim=MODEL_DIM,
            inter_dim=INTER, experts=EXPERTS, total_rows=a_perm.shape[0],
            gather=False, sort_block_m=SBM)

        # Under test: dense A, gather by token id.
        out_got, os_got = run_tp_gemm1(
            x=a_dense, scale_x=s_dense, w=w1_shuf, scale_w=w1_scale_shuf,
            tile_row_base=trb, expert_ids=seid, sorted_token_ids=sids,
            num_valid_ids=nv, max_sorted=max_sorted, model_dim=MODEL_DIM,
            inter_dim=INTER, experts=EXPERTS, total_rows=total_rows,
            gather=True, sort_block_m=SBM)
        torch.cuda.synchronize()

        pa = out_ref.view(torch.uint8)[:nvalid]
        pb = out_got.view(torch.uint8)[:nvalid]
        assert torch.equal(pa, pb), (
            f"m_global={m_global}: payload differs on "
            f"{int((pa != pb).sum())} of {pa.numel()} bytes"
        )
        rows = ((nvalid + 255) // 256) * 256
        cols = (((INTER // 32) + 7) // 8) * 8
        sa = os_ref[: rows * cols]
        sb = os_got[: rows * cols]
        assert torch.equal(sa, sb), (
            f"m_global={m_global}: mx scale differs on {int((sa != sb).sum())} bytes"
        )
        assert torch.isfinite(out_ref.float()).all(), f"m_global={m_global}: ref non-finite"
        print(f"  m_global={m_global} nvalid={nvalid} tiles={n_tiles_m} bit-identical")

    print("case_equiv OK")


CASES = {"equiv": case_equiv}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default="equiv")
    args = ap.parse_args()
    torch.cuda.set_device(0)
    CASES[args.case]()


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: 跑对拍**

```bash
cd /root/workspace/aiter
PYTHONPATH=. python op_tests/multigpu_tests/test_tp_gemm1.py --case equiv
```

Expected：四行 `m_global=N nvalid=... tiles=... bit-identical`，最后 `case_equiv OK`。

首次运行要编译两个 kernel（`gather` 与 `contig` 各一个），慢一两分钟属正常。

- [ ] **Step 5: 负对照一 — 去掉 topk slot 的掩码必须失败**

把 `TPATileLoader._row_of` 里的 `tok = tok & fx.Int32(0x00FFFFFF)` 临时删掉（直接用原值），重跑 Step 4。

Expected：**FAIL**。`sorted_token_ids` 高 8 位存 topk slot，不掩掉就会把 slot 号当成行号的高位，取到完全错误的行。

看到失败后改回去。

- [ ] **Step 6: 负对照二 — scale loader 的行分解写错必须失败**

把 `TPAScaleLoader.stage` 里的 `cir = lin - row * fx.Int32(chunks_per_row)` 临时改成 `cir = lin % fx.Int32(16)`（一个看似合理但错误的分解，因为 `chunks_per_row` 是 14 不是 16），重跑 Step 4。

Expected：**FAIL**，报 mx scale 或 payload 不一致。

> 这一条专门验证 scale 的二维分解是真的在起作用。如果它不失败，说明测试没有覆盖到 scale 路径，先修测试。

改回去。

- [ ] **Step 7: 确认冻结文件没被碰**

```bash
cd /root/workspace/aiter
git diff --stat main...HEAD -- aiter/ops/flydsl/kernels/mega_moe/gemm1.py \
    aiter/ops/flydsl/kernels/mega_moe/gemm_util.py \
    aiter/ops/flydsl/kernels/mega_moe/mega_moe_stage1.py \
    aiter/ops/flydsl/kernels/mega_moe/dispatch.py \
    aiter/ops/flydsl/kernels/mega_moe/mega_moe_v2.py
```

Expected：**无输出**。

- [ ] **Step 8: 阶段一与传输层用例不受影响**

```bash
cd /root/workspace/aiter
for c in construct capacity exports; do
  printf "  %-16s " "$c"
  PYTHONPATH=. python op_tests/multigpu_tests/test_tp_moe_stage1.py --case $c 2>&1 | tail -1
done
for c in numerics e2e; do
  printf "  %-16s " "$c"
  PYTHONPATH=. torchrun --standalone --nproc_per_node=8 \
      op_tests/multigpu_tests/test_tp_moe_stage1.py --case $c 2>&1 | tail -1
done
printf "  %-16s " "gather bitexact"
PYTHONPATH=. torchrun --standalone --nproc_per_node=8 \
    op_tests/multigpu_tests/test_tp_gather.py --case bitexact 2>&1 | tail -1
```

Expected：六行全部以 `OK` 结尾。

- [ ] **Step 9: 提交**

```bash
cd /root/workspace/aiter
python -m black --check aiter/ops/flydsl/kernels/mega_moe/tp_gemm1.py \
    op_tests/multigpu_tests/test_tp_gemm1.py
git add aiter/ops/flydsl/kernels/mega_moe/tp_gemm1.py op_tests/multigpu_tests/test_tp_gemm1.py
git commit -m "feat(tp-moe): standalone TP GEMM1 over dense token-indexed activations

Forks only the builder half of build_fused_gemm1; do_tile is imported and
reused, so the 230-line MFMA pipeline is shared rather than copied.

Equivalence is proved physically rather than by tolerance: the same kernel is
compiled twice, once with the original contiguous loaders fed host-permuted
rows and once with the gathering loaders fed dense rows, and the two outputs
must match bit for bit. Verified it fails when the topk-slot mask is dropped
and when the scale loader's row decomposition is perturbed.

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Self-Review

**Spec 覆盖：** 设计文档 2.3 节（融合路径的 A-scale 是 row-major、shuffle 在 LDS 现做）是本方案能成立的前提；5.5 节的两处 loader 改动由 Task 1 实现，PAD 行钳位也在里面。

**有意不覆盖：** 推送与 GEMM1 的融合、`TPMoEStage1.forward` 的切换、NCCL 路径的删除、性能验收。全部留给阶段二（下之二），因为它们依赖本方案产出的 `compile_tp_gemm1` 的实际形态。

**四条隐形约束都写进了代码注释**，不只写在方案里：`load_step` 零 VMEM、返回 `m_repeat // 2` 个标量、`for_tile` 与 `__init__` 的分工、LDS 布局不变。将来有人来改这两个 loader，注释就在他眼前。

**类型一致性：** Task 1 Step 2 用 `inspect.signature` 逐字比对新旧 loader 的方法签名，因为 `do_tile` 是按位置调用的。`run_tp_gemm1` 的参数顺序与 `launch` 的前 11 个一致。`PAD_ROW = total_rows - 1` 与测试里 `pad_row = total_rows - 1` 一致，A 和 scale 都在末尾追加了一行零。

**已知风险：** `async_a_copy` 默认 `False`，所以 Task 1 里 `prefetch_to_lds` 那条路径本方案不会被执行到，等于没测。方案里保留了它的实现（因为阶段二下之二可能要开），但**它是未验证代码**。真要开启时必须先补一个 `async_a_copy=True` 的对拍用例，尤其要盯 `gemm1.py:152-156` 那个硬编码 vmcnt。
