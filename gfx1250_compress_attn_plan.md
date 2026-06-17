# gfx1250 fused_compress_attn 移植计划

## 分支
`jli/gfx1250/compress_attn` (基于 main 70edd9b53)

## 问题背景

`fused_compress_attn.py` (2468行) 和 `fused_compress_attn_hca.py` (1251行) 是 wave64 专用内核 (BLOCK_THREADS=64)。
在 gfx1250 (RDNA4, wave32) 上运行时报错:
```
LLVM ERROR: Cannot select: intrinsic %llvm.amdgcn.permlane32.swap
```

原因: `gpu.shuffle xor %x, 32, 64` 被 MLIR pass 降级为 `rocdl.permlane32.swap`，gfx1250 不支持该指令。

## 确认的范围
- **全部三个 shape**: csa_main (D=512, BF16), csa_indexer (D=128, FP8), hca_main (D=512, ratio=128, BF16)
- **两种路径都做**: legacy 单 wave + K-split 多 wave
- **原文件保留不动** (MI355X gfx950 还需要)

---

## 新建文件

### 1. `aiter/ops/flydsl/kernels/fused_compress_attn_gfx1250.py` (~2600行)

从 `fused_compress_attn.py` 复制修改。包含:
- `_build_kernel(...)` — 单 wave32 内核
- `_build_kernel_ksplit(...)` — K-split 多 wave 内核
- `csa_ksplit_num_waves_gfx1250(plan_capacity)` — 自动选 NW
- `compile_flydsl_fused_compress_attn_gfx1250(...)` 
- `compile_flydsl_fused_compress_attn_ksplit_gfx1250(...)`
- `flydsl_fused_compress_attn_gfx1250(...)` — 公开入口

### 2. `aiter/ops/flydsl/kernels/fused_compress_attn_hca_gfx1250.py` (~1350行)

从 `fused_compress_attn_hca.py` 复制修改。包含:
- Kernel A: `_build_compress_forward_kernel(...)` — 多 wave K-split compress
- Kernel B: `_build_norm_rope_scatter_kernel(...)` — 单 wave norm+rope+scatter
- `hca_per_n_config_gfx1250(plan_capacity)`
- `flydsl_hca_compress_attn_gfx1250(...)`

### 3. `op_tests/test_flydsl_compress_attn_gfx1250.py`

gfx1250 专用测试，复用现有 reference 做正确性校验。

---

## 修改的现有文件 (仅加 dispatch 路由)

### `fused_compress_attn.py` — `flydsl_fused_compress_attn()` 函数顶部加:
```python
from aiter.jit.utils.chip_info import get_gfx as _get_gfx
if _get_gfx() == "gfx1250":
    from .fused_compress_attn_gfx1250 import flydsl_fused_compress_attn_gfx1250
    return flydsl_fused_compress_attn_gfx1250(
        kv_in=kv_in, score_in=score_in, ...所有参数透传...
    )
```

### `fused_compress_attn_hca.py` — `flydsl_hca_compress_attn()` 函数顶部同理路由到 gfx1250 版本。

---

## 核心代码变更详解

### 1. BLOCK_THREADS: 64 → 32

模块级常量修改。所有派生量自动跟随:

| 量 | wave64 | wave32 |
|---|---|---|
| BLOCK_THREADS | 64 | 32 |
| VEC (D=512) | 8 | 16 |
| VEC (D=128) | 2 | 4 |
| ROPE_THREAD_LO (D=512) | 56 | 28 |
| ROPE_THREAD_LO (D=128) | 32 | 16 |
| PAIRS_PER_THREAD (D=512) | 4 | 8 |
| PAIRS_PER_THREAD (D=128) | 1 | 2 |
| log2_block | 6 | 5 |

断言需扩展: `assert VEC in (2, 4, 8)` → `assert VEC in (2, 4, 8, 16)`

### 2. wave_reduce_add / wave_reduce_max

**无需手动改循环逻辑** — 已参数化:
```python
log2_block = int(math.log2(BLOCK_THREADS))  # 5 instead of 6
for sh_exp in range_constexpr(log2_block):
    off = BLOCK_THREADS // (2 << sh_exp)     # 16,8,4,2,1 instead of 32,16,8,4,2,1
    peer = shuffle_xor(off, BLOCK_THREADS)   # width=32 instead of 64
```

`shuffle_xor(offset≤16, width=32)` 在 RDNA 上正常工作 (已由 `flash_attn_func_gfx1201.py` 验证)。

### 3. VEC=16 的 load 路径 (新增代码)

`_load_f32_vec`: 现有 VEC≤4 (单次 dwordxN) 和 VEC=8 (2× dwordx4)。
VEC=16 需要 **4× dwordx4** load:
```python
else:  # VEC == 16
    quarter = 4
    out = []
    for q in range_constexpr(4):
        r = buffer_ops.buffer_load(rsrc,
            ArithValue(off) + arith.constant(q * quarter, type=i32),
            vec_width=quarter, dtype=f32)
        for i in range_constexpr(quarter):
            out.append(vector.extract(r, static_position=[i], dynamic_position=[]))
    return out
```

`_load_bf16_vec_then_f32`: VEC=16 → 8 dwords → **2× dwordx4** load，然后 bitcast 为 vec<16,bf16>，逐元素 extf 到 f32。

### 4. VEC=16 的 BF16 store 路径 (新增代码)

现有: `dwords = (VEC+1)//2`。VEC=16 → dwords=8。硬件最大 `buffer_store` 是 dwordx4。
需拆成 **2× dwordx4** store:
```python
if const_expr(dwords <= 4):
    buffer_ops.buffer_store(bf16_as_i32, out_rsrc, cache_off_dw)
else:  # dwords == 8, VEC=16
    lo = vector.extract_strided_slice(bf16_as_i32, offsets=[0], sizes=[4], strides=[1])
    hi = vector.extract_strided_slice(bf16_as_i32, offsets=[4], sizes=[4], strides=[1])
    buffer_ops.buffer_store(lo, out_rsrc, cache_off_dw)
    buffer_ops.buffer_store(hi, out_rsrc, ArithValue(cache_off_dw) + c4_i32)
```

### 5. K-split 内核的 wid/lid 计算

```python
# wave64 原来:
c_64 = arith.constant(64, type=i32)  # 实际引用 BLOCK_THREADS
wid = tid // c_64
lid = tid % c_64

# wave32 改为:
c_WS = arith.constant(BLOCK_THREADS, type=i32)  # = 32
wid = tid // c_WS
lid = tid % c_WS
```

`BLOCK_TH = BLOCK_THREADS * NW` = 32 * NW (was 64 * NW)。

### 6. FP8 quant 路径 (csa_indexer)

- `rocdl.cvt_pk_fp8_f32` **在 gfx1250 上可用** (qk_norm_rope_quant.py 第120行确认)
- FP8 dtype: gfx1250 用 `e4m3fn` (OCP)，由 `_fp8_const()` 运行时解析
- D=128 / BLOCK_THREADS=32 → VEC=4。VEC=4 的 FP8 packing 路径 **已存在**:
  ```python
  elif const_expr(VEC == 4):
      pk = rocdl.cvt_pk_fp8_f32(i32, fp8_inputs[0], fp8_inputs[1], c_p0, 0)
      pk = rocdl.cvt_pk_fp8_f32(i32, fp8_inputs[2], fp8_inputs[3], pk, 1)
      dword = pk  # 4 bytes packed into 1 i32, no pair-coop needed
  ```
  无需 `shuffle_xor(1, ...)` pair cooperation — 比 wave64 的 VEC=2 路径更简单。

### 7. Preshuffle 决策

**gfx1250 强制 `preshuffle=False` (线性布局)**。原因:
- `_PRESHUFFLE_TILE=16` 是 MFMA 16×16 tile layout (gfx9/gfx94/gfx95 CDNA)
- gfx1250 用 WMMA，tile 布局不同，MFMA preshuffle 公式会产生错误的 cache 布局
- gfx1250 的 attention consumer 支持线性 FP8 布局 (`pa_mqa_logits.py` 第459行确认)

在 `flydsl_fused_compress_attn_gfx1250()` 中:
```python
_preshuffle = False  # MFMA preshuffle is gfx9-only; force linear layout
```

### 8. HCA Kernel A (compress_forward)

- `BLOCK_TH = 32 * NW` (was 64 * NW)
- wid/lid 用 32 而非 64
- `slice_size` 约束改为 `% 32 == 0`
- VEC = SLICE_SZ // 32 (slice_size=64 → VEC=2, slice_size=32 → VEC=1)
- **无 shuffle_xor** — 跨 wave reduction 完全走 LDS + barrier

### 9. HCA Kernel B (norm_rope_scatter)

- `BLOCK_THREADS=32`, D=512 → VEC=16
- `wave_reduce_add` 自动适配 (log2_block=5, width=32)
- 需要 VEC=16 的 load/store 分支 (同上)
- 断言 `assert VEC == 8` → `assert VEC in (8, 16)`

### 10. 内核命名

加 `"w32"` 后缀避免 JIT cache 冲突:
```python
_name_parts = ["fused_compress_attn_w32", f"D{D}", ...]
```

### 11. 编译 hints

```python
_DEFAULT_COMPILE_HINTS = {
    "waves_per_eu": 8,
    "fast_fp_math": True,
    "unsafe_fp_math": True,
}
```
与 wave64 版本相同。

### 12. Auto-tuning 函数 (初始值，需硬件调优)

```python
def csa_ksplit_num_waves_gfx1250(plan_capacity):
    if plan_capacity <= 512: return 4
    if plan_capacity <= 1024: return 2
    return 1

def hca_per_n_config_gfx1250(plan_capacity):
    if plan_capacity <= 64: return 32, 8
    if plan_capacity <= 256: return 64, 8
    if plan_capacity <= 1024: return 256, 4
    return 512, 1
```

---

## BLOCK_THREADS 在原文件中的所有引用位置

### fused_compress_attn.py
| 行 | 上下文 |
|---|---|
| 104 | 定义: `BLOCK_THREADS = 64` |
| 194 | `VEC = D // BLOCK_THREADS` |
| 205 | `assert D % BLOCK_THREADS == 0` |
| 247 | `log2_block = int(math.log2(BLOCK_THREADS))` |
| 299-301 | `wave_reduce_add`: `off = BLOCK_THREADS // (2 << sh_exp)`, `shuffle_xor(off, BLOCK_THREADS)` |
| 309-311 | `wave_reduce_max`: 同上 |
| 1030 | FP8 pair-coop: `shuffle_xor(1, BLOCK_THREADS)` |
| 1204-1208 | launch: `block=(BLOCK_THREADS, 1, 1)` |
| 1272 | ksplit: `VEC = D // BLOCK_THREADS` |
| 1276 | `BLOCK_TH = BLOCK_THREADS * NW` |
| 1282 | `assert D % BLOCK_THREADS == 0` |
| 1333 | `log2_block = int(math.log2(BLOCK_THREADS))` |
| 1372 | `c_64 = arith.constant(BLOCK_THREADS, type=i32)` |
| 1387-1388 | `wid = divsi(tid, c_64)`, `lid = remui(tid, c_64)` |
| 1668-1674 | ksplit wave-0 `wave_reduce_add` |
| 1832-1840 | ksplit wave-0 `wave_reduce_max` |
| 1896 | ksplit FP8 pair-coop: `shuffle_xor(1, BLOCK_THREADS)` |
| 2053-2057 | ksplit launch: `block=(BLOCK_TH, 1, 1)` |

### fused_compress_attn_hca.py
| 行 | 上下文 |
|---|---|
| 70 | 定义: `BLOCK_THREADS = 64` |
| 676 | `VEC = D // BLOCK_THREADS` |
| 680 | `assert D % BLOCK_THREADS == 0` |
| 688 | `log2_block = int(math.log2(BLOCK_THREADS))` |
| 718-722 | `wave_reduce_add` (Kernel B) |
| 780 | `sq_full = wave_reduce_add(sq_local)` |
| 988 | launch: `block=(BLOCK_THREADS, 1, 1)` |

### shuffle_xor 所有调用位置
| 文件 | 行 | 上下文 |
|---|---|---|
| fused_compress_attn.py | 301 | `wave_reduce_add` butterfly |
| fused_compress_attn.py | 311 | `wave_reduce_max` butterfly |
| fused_compress_attn.py | 1030 | FP8 VEC=2 pair-coop |
| fused_compress_attn.py | 1672 | ksplit `wave_reduce_add` |
| fused_compress_attn.py | 1837 | ksplit `wave_reduce_max` |
| fused_compress_attn.py | 1896 | ksplit FP8 pair-coop |
| fused_compress_attn_hca.py | 722 | Kernel B `wave_reduce_add` |

---

## 注意事项

### VEC=16 寄存器压力
- loop-carry state: 3×VEC = 48 个 f32 = 48 VGPRs (wave64 是 3×8=24)
- `enable_prefetch_input=True` 额外加 3×VEC = 48 VGPRs
- RDNA4 有 512 VGPRs/wave，可以承受，但建议 gfx1250 默认 `enable_prefetch_input=False`

### repo 中的 gfx1250 命名惯例
- 文件: `*_gfx1250.py` (扁平文件，如 `gemm_mxscale_gfx1250.py`)
- 或子目录包: `fmha_gfx1250/` (仅 FMHA 用了这种方式)
- dispatch: `from aiter.jit.utils.chip_info import get_gfx; if get_gfx() == "gfx1250": ...`
- 低层: `from flydsl.runtime.device import get_rocm_arch`

### flydsl 包 bug (不在本次修改范围内)
`is_rdna_arch()` 不识别 gfx1250 (只匹配 `gfx120*`)。但 `convert-gpu-to-rocdl` pass 接收 `chipset=gfx1250` (从 `get_rocm_arch()` 来)，所以 shuffle 降级实际上是 MLIR pass 行为，改 `is_rdna_arch` 会影响 `wave64` flag。本次通过新建 wave32 内核文件来绕过。

---

## 测试策略

### 正确性测试
- 三个 shape × bs=[1,2,4,8,16,32,128,512] × mtp=[0,3] × mode=[decode,prefill]
- K-split 变体: csa_main, csa_indexer, hca_main
- 对比 `fused_compress_attn_reference` (torch 实现)
- BF16 容差: rtol=1e-2, atol=2e-2, tol_err_ratio=0.02
- FP8 容差: rtol=1e-2, atol=1e-2, tol_err_ratio=0.05

### 回归防护
- MI355X 上确认 gfx950 dispatch 不受影响
