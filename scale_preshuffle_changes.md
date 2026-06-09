# Scale Preshuffle 修改说明

## 背景

gfx1250 WMMA 指令 `wmma_scale_f32_16x16x128_f8f6f4` 的 `scaleBType` 参数选择的是 scale VGPR 中的**奇偶字节**：

- `scaleBType=0` → 偶数字节 (byte 0, 2) → wm=even 的 scale 数据
- `scaleBType=1` → 奇数字节 (byte 1, 3) → wm=odd 的 scale 数据

这与 `aiter.ops.shuffle.shuffle_scale` 的 non-guinterleave 格式一致（N_Pack 最内层，stride=1）。

原来的 `_grouped_a8w4_preshuffle_e8m0_scale` 产生的 layout 不符合这一格式，导致 scale 错误。

---

## 修改内容

### 1. `aiter/ops/flydsl/grouped_moe_gfx1250.py`

**函数 `_grouped_a8w4_preshuffle_e8m0_scale`（约 line 147）**

**核心变化**：改成 NP（wm pack）最内层的格式。

每个 4 字节 DWORD 的布局（以 kgrp=0 为例）：
```
byte 0: wm=even, K-group 0  (opsel=0 → 偶数字节 → wm=even)
byte 1: wm=odd,  K-group 0  (opsel=1 → 奇数字节 → wm=odd)
byte 2: wm=even, K-group 1
byte 3: wm=odd,  K-group 1
byte 4: wm=even, K-group 2  (kgrp=1，+SCALES_PER_WMMA=4 字节偏移)
byte 5: wm=odd,  K-group 2
byte 6: wm=even, K-group 3
byte 7: wm=odd,  K-group 3
```

**新代码**：

```python
def _grouped_a8w4_preshuffle_e8m0_scale(scale, warp_tile, scale_k_per_tile=4):
    scale = scale.view(torch.uint8).contiguous()
    E, _, k_scale = scale.shape
    wmma_rep = int(warp_tile) // 16
    k_groups = k_scale // scale_k_per_tile
    k_wmma_steps = scale_k_per_tile // 4

    if wmma_rep < 2:
        # wmma_rep=1（activation scale 常见情况）：保持原始格式
        g = scale.view(E, -1, 1, 16, k_groups, k_wmma_steps, 4)
        g = g.permute(0, 1, 3, 4, 5, 2, 6).contiguous()
        return g.reshape(E, -1, k_scale)

    # wmma_rep >= 2：NP 最内层（shuffle_scale 风格）
    # 每个 DWORD: [wm_even_k0, wm_odd_k0, wm_even_k1, wm_odd_k1]
    g = scale.view(E, -1, wmma_rep // 2, 2, 16, k_groups, k_wmma_steps, 2, 2)
    #              E  Ms  wm_pair        NP  l16  kg        kws            kgrp kpack
    g = g.permute(0, 1, 4, 5, 6, 2, 7, 8, 3).contiguous()
    #             E  Ms  l16  kg  kws  wm  kgrp kpack NP(innermost)
    return g.reshape(E, -1, k_scale * wmma_rep)
```

---

### 2. `aiter/ops/flydsl/kernels/gemm_mxscale_gfx1250.py`

#### `_load_b_and_scales`（FP8/A8W4 路径）

A8W4 强制走 opsel 路径（不再依赖 `use_scale_opsel` flag）：

```python
# 原来
if const_expr(use_scale_opsel):
    b_scales = b_scales_all[::2]
    a_scales = a_scales_all[::2]
else:
    b_scales = b_scales_all
    a_scales = a_scales_all

# 修改后
if const_expr(is_a8w4 or use_scale_opsel):
    b_scales = b_scales_all[::2]
    a_scales = a_scales_all[::2]
else:
    b_scales = b_scales_all
    a_scales = a_scales_all
```

#### `_emit_wmma`（opsel 索引）

A8W4 强制使用 wm//2 和 wm%2：

```python
# 原来
if const_expr(use_scale_opsel):
    a_scale_idx = wm // 2
    a_opsel = wm % 2
else:
    a_scale_idx = wm
    a_opsel = 0

# 修改后
if const_expr(is_a8w4 or use_scale_opsel):
    a_scale_idx = wm // 2
    a_opsel = wm % 2
else:
    a_scale_idx = wm
    a_opsel = 0
```

B scale（FP8/A8W4 路径）同理：

```python
if const_expr(is_a8w4 or use_scale_opsel):
    b_scale_idx = wn // 2
    b_opsel = wn % 2
else:
    b_scale_idx = wn
    b_opsel = 0
```

---

### 3. `aiter/ops/flydsl/kernels/moe_scatter_copy_preshuffle_scale.py`

fused scatter + preshuffle kernel 整体重写，产生新的字节交错格式。

**核心变化**：每个 work item 处理 `(lane, sd, wm_pair)`，同时 gather wm=even 和 wm=odd 两行，用位运算生成两个 kgrp DWORD：

```python
# 以下是伪代码
src0 = src[token(wm_pair*2,   lane), sd]   # wm=even 的 4 字节 scale
src1 = src[token(wm_pair*2+1, lane), sd]   # wm=odd  的 4 字节 scale

# kgrp=0 DWORD: bytes [wm0_K0, wm1_K0, wm0_K1, wm1_K1]
dst_kgrp0 = (src0 & 0xFFFF) | ((src1 & 0xFFFF) << 16)

# kgrp=1 DWORD: bytes [wm0_K2, wm1_K2, wm0_K3, wm1_K3]
dst_kgrp1 = (src0 >> 16) | (src1 & 0xFFFF0000)

# dwordx2 store（kgrp=0 和 kgrp=1 相邻）
store [dst_kgrp0, dst_kgrp1] at dst_off
```

---

### 4. `aiter/ops/quant.py`

`MxScaleRoundMode` import 加 fallback，解决 pybind .so 不包含该符号时的 ImportError：

```python
# 原来
from ..utility.mx_types import (
    MX_DEFAULT_ROUND_MODE,
    MxDtypeInt,
    MxScaleRoundMode,
    MxScaleRoundModeInt,
)

# 修改后
from ..utility.mx_types import (
    MX_DEFAULT_ROUND_MODE,
    MxDtypeInt,
    MxScaleRoundModeInt,
)
try:
    from ..utility.mx_types import MxScaleRoundMode
except ImportError:
    MxScaleRoundMode = MxScaleRoundModeInt  # fallback: pybind .so lacks it
```

---

### 5. `aiter/ops/shuffle.py`

注释修正（原来写的 `K_Lane = 4` 是错的）：

```python
# gfx1250 uses wave32; K_Lane = warp_size // N_Lane = 32 // 16 = 2
warp_size = 32
N_Lane = 16
K_Lane = warp_size // N_Lane  # 2
```

---

## 尚未解决的问题

### 1. `_precompute_scale_lane_bases` 的 kgrp offset

当前代码为 FP4/A8W4 仍保留：
```python
base = base + lane_kgrp * arith.index(SCALES_PER_WMMA)  # +4 字节
```

理论上应该**去掉**：新的 NP-innermost layout 中，kgrp=0 和 kgrp=1 两个半 warp 应读取同一个 VGPR（即同一地址），硬件自动用字节 0,1 给 kgrp=0，字节 2,3 给 kgrp=1。

去掉后 cmodel 验证（rel_l2）应该会大幅降低。但目前遇到了 `HIP error: invalid device function` 阻断了进一步验证。

### 2. `HIP error: invalid device function`

在 cmodel 上跑 a4w4 real-gemm 时出现，具体表现：

```
File "grouped_moe_gfx1250.py", line 697, in _maybe_grouped_gfx1250_a8w4_moe
    _bias1_arg = _bias1_arg.to(dtype)
torch.AcceleratorError: HIP error: invalid device function
```

已确认：
- FlyDSL 编译目标是 `gfx1250` ✓
- 原始代码（未改动）也有相同错误（MxScaleRoundMode import 修好之后）
- 原因未明：可能是 FP4 WMMA kernel 在该版本 cmodel 上的兼容性问题

---

## cmodel 验证结论

用 cmodel 对比了两种 layout：

| Layout | DWORD 格式 | opsel 语义 | rel_l2（cmodel） |
|--------|-----------|-----------|-----------------|
| A：NP innermost | `[wm0_k0, wm1_k0, wm0_k1, wm1_k1]` | 偶数字节=wm0，奇数字节=wm1 | **92**（更小）|
| B：NP second | `[wm0_k0, wm0_k1, wm1_k0, wm1_k1]` | 低2字节=wm0，高2字节=wm1 | 169 |

→ **Layout A（NP innermost）方向正确**，与 `shuffle_scale` 格式一致。  
→ rel_l2=92 仍大的原因：kgrp offset 未去掉，导致 kgrp=1 的 lane 读到错误的 VGPR。
