# gfx1250 GEMM2 A8W4 TDM noact 深度解析：从路由、布局到逐 PC ISA

<!-- markdown-toc-generator:start -->
## Table of Contents

- [Authoritative Sources](#authoritative-sources)
- [Benchmark / Trait Configuration](#benchmark-trait-configuration)
  - [固定 workload 与路由几何](#sec-fixed-workload-and-routing-geometry)
  - [编译资源与代码对象](#sec-compiled-resource-and-code-object)
  - [PC、计数和证据口径](#sec-pc-counting-and-evidence-convention)
- [1. Kernel overview and software pipeline](#1-kernel-overview-and-software-pipeline)
  - [1.1 软件调用路径与 GEMM2 数学定义](#sec-1-1-call-path-and-mathematical-definition)
  - [1.2 Workgroup / wave / lane 分工](#sec-1-2-workgroup-wave-lane-geometry)
  - [1.3 Grid、DeepGEMM swizzle 与 expert lookup](#sec-1-3-grid-swizzle-and-expert-lookup)
  - [1.4 LDS arena：逻辑 74,752 B 与实际 75,776 B](#sec-1-4-lds-arena-logical-vs-allocated)
  - [1.5 两级 TDM ring buffer 与六级 K pipeline](#sec-1-5-two-buffer-tdm-pipeline)
  - [1.6 VOP3PX2 WMMAScale、硬件 operand swap 与 E8M0](#sec-1-6-vop3px2-wmmascale-and-operand-swap)
  - [1.7 Wait、barrier 与可见性](#sec-1-7-waits-barriers-and-visibility)
  - [1.8 完整 kernel-relative PC phase map](#sec-1-8-full-pc-phase-map)
- [2. Full GM/LDS/VGPR layout maps](#2-full-gmldsvgpr-layout-maps)
  - [2.1 A2 payload：GM -> LDS](#sec-2-1-a2-gm-to-lds)
  - [2.2 A2 payload：LDS -> VGPR](#sec-2-2-a2-lds-to-vgpr)
  - [2.3 B2 weight：GM -> LDS](#sec-2-3-b2-gm-to-lds)
  - [2.4 B2 weight：LDS -> VGPR](#sec-2-4-b2-lds-to-vgpr)
  - [2.5 SA：A2 的 E8M0 scale，GM -> LDS -> VGPR](#sec-2-5-sa-layout)
  - [2.6 SB：B2 的 n32k4 E8M0 scale，GM -> LDS -> VGPR](#sec-2-6-sb-layout)
  - [2.7 A2/B2/SA/SB -> WMMAScale -> 8 个 accumulator fragments](#sec-2-7-wmma-and-accumulator-layout)
  - [2.8 bias2：GM -> VGPR -> FP32 accumulator](#sec-2-8-bias2-epilogue)
  - [2.9 Output：accumulator -> BF16 VGPR -> LDS -> GM](#sec-2-9-output-layout)
  - [2.10 端到端 tensor transition 总图](#sec-2-10-end-to-end-transition-map)
  - [2.11 Tensor residency 总结](#sec-2-11-residency-summary)
  - [2.12 Low-level call chains and issue counts](#sec-2-12-low-level-call-chains-and-issue-counts)
- [3. Final summary and verification boundary](#3-final-summary-and-verification-boundary)

<!-- markdown-toc-generator:end -->

本文只分析下列 **GEMM2** 编译实例：

```text
gemm_a8w4_tdm_t16x512x128_w1x4_b2_e384_afp8_outbf16_noact_bias1_qout0_qrep1_v1
```

它是固定 MoE workload 中的 stage-2 grouped GEMM：以 GEMM1 后的 MXFP8 activation
`a2_payload` 为左数学矩阵，以每个 expert 的 MXFP4 `w2` 为右数学矩阵，加 `bias2`，
不执行 activation，不量化输出，最终写 BF16 `grouped_out`。

文中的 ISA PC 均为 **kernel symbol-relative byte PC**，区间均采用半开形式
`[start, end)`。ELF 中函数符号虚拟地址为 `0x1900`，因此需要 ELF VA 时只需
`ELF_VA = 0x1900 + kernel_relative_PC`；运行时装载地址仍可重定位。

## Authoritative Sources

以下文件按“越靠近最终机器码，越能裁决当前编译实例”的原则使用：

1. **最终 ISA 与 metadata**
   - `aiter/my_code/flydsl_dump/gemm_a8w4_tdm_t16x512x128_w1x4_b2_e384_afp8_outbf16_noact_bias1_qout0_qrep1_v1/21_final_isa.s`
2. **嵌入代码对象与最终 LLVM IR**
   - 同目录 `19_gpu_module_to_binary.mlir`
   - 同目录 `20_llvm_ir.ll`
3. **FlyDSL lowering 证据**
   - 同目录 `00_origin.mlir`
   - 同目录 `03_fly_layout_lowering.mlir`
   - 同目录 `07_fly_promote_regmem_to_vectorssa.mlir`
   - 同目录 `08_convert_fly_to_rocdl.mlir`
4. **当前 kernel source**
   - `aiter/aiter/ops/flydsl/kernels/mxfp4_preshuffle_gfx1250_tdm.py`
   - `aiter/aiter/ops/flydsl/kernels/gemm_common_gfx1250.py`
5. **wrapper 与上层调用**
   - `aiter/aiter/ops/flydsl/batched_gemm_mxfp4.py`
   - `aiter/aiter/ops/flydsl/grouped_moe_gfx1250.py`
   - `aiter/aiter/fused_moe.py`
6. **固定 workload 与编译日志**
   - `aiter/my_code/run_gemm.log`
   - `aiter/my_code/run_gemm.sh`
7. **硬件语义交叉参考**
   - `C:/Users/yanguahe/Documents/code/llm-wiki/mi400_hw_wiki/raw/papers/mi400_hd_txt/architecture/subsystem/SH/MI400_Shader_Programming#65.txt`
   - 重点为 MI400 Shader Programming Guide §4.3.7（wait/dependency，印刷页 85、88）、
     §4.6.12（WMMA，印刷页 143-165）、§4.10（TDM，印刷页 197-206）。

> **gfx1250 适用性警告。** 上述 guide 描述 MI450/MI400，而本报告对象是 gfx1250。
> 本文仅用它解释在当前 gfx1250 ISA 中已经直接观察到的同族概念：
> wave32 WMMA、4-DWORD VOP3PX2 block-scale encoding、TDM D#、TENSORcnt 及 LDS/DS
> dependency。MI400 的具体吞吐、容量、clause errata、XNACK 行为和产品级限制均不自动外推到
> gfx1250。发生冲突时，以当前 `21_final_isa.s`、代码对象和 gfx1250 lowering 为准。

## Benchmark / Trait Configuration

<a id="sec-fixed-workload-and-routing-geometry"></a>
### 固定 workload 与路由几何

```text
GPU                         : gfx1250
wavefront                   : 32
token count T               : 4096
topk                        : 6
route count                 : T * topk = 24576
expert count E              : 384
hidden/model dimension N    : 7168
intermediate dimension K    : 768
contiguous_m                : 30720
A2 format                   : MXFP8 E4M3, 1 byte/value
B2 format                   : MXFP4 E2M1, 2 values/byte
scale format                : E8M0, one byte per K=32 block
output                      : BF16
accumulator                 : FP32
activation                  : none
bias                        : enabled; actual tensor is stage-2 bias2
tile                        : M16 x N512 x K128
workgroup                   : 128 threads = 4 wave32
warps                       : m_warp=1, n_warp=4
buffers                     : 2
```

固定 launch 几何：

| Quantity | Formula | Value |
|---|---:|---:|
| M tiles | `ceil(30720 / 16)` | 1920 |
| N tiles | `ceil(7168 / 512)` | 14 |
| Grid | `1920 * 14` | 26880 workgroups |
| Block | `4 * 32` | 128 threads |
| K tiles | `768 / 128` | 6 |
| WMMA N fragments / wave | `(512 / 4) / 16` | 8 |
| FP32 accumulator fragments / wave | `1 * 8` | 8 |

`contiguous_m` 不是有效 route 数本身。`4096*6=24576` 个 route 先按 expert 聚集，再为
16-row M tile 做 prefix padding；`m_tile_map[e]` 保存 expert `e` 的 tile-aligned exclusive
end row。`30720-24576=6144` 是本 workload 的静态容量余量，不代表每个 expert 都固定填充
16 行。

<a id="sec-compiled-resource-and-code-object"></a>
### 编译资源与代码对象

| Resource | Dump value | Interpretation |
|---|---:|---|
| VGPR | 172 | 包括 payload fragments、scale、8 个 FP32 accumulator fragments 和 epilogue 临时值 |
| SGPR | 84 | kernarg、TDM D#、tile/expert 地址与循环常量 |
| AGPR | 0 | gfx1250 WMMAScale 结果留在普通 VGPR |
| VGPR spill | 0 | 无 scratch spill |
| SGPR spill | 0 | 无 SGPR spill |
| Private segment | 0 B | 无 private scratch |
| Fixed group segment | 0 B | **不是没有 LDS**；本 kernel 使用 dynamic LDS |
| Dynamic LDS launch | 75,776 B (`0x12800`) | `00_origin.mlir` 的 `gpu.launch_func` 明确给出 |
| Kernarg | 176 B | expanded tensor descriptors + pointers + scalars |
| Required workgroup | `128 x 1 x 1` | 4 wave32 |
| Kernel code size | 7,104 B (`0x1bc0`) | ELF symbol size |

代码对象中的 1,109 条 encoded instructions 恰好覆盖 `0x0000..0x1bbf`：

| Encoding width | Instruction count | Bytes |
|---:|---:|---:|
| 4 B | 597 | 2388 |
| 8 B | 405 | 3240 |
| 12 B | 59 | 708 |
| 16 B | 48 | 768 |
| **Total** | **1109** | **7104 (`0x1bc0`)** |

48 条 16-byte instruction 全部是
`v_wmma_scale_f32_16x16x128_f8f6f4`。TDM load/store 是 12-byte encoding；
GFX12 dual-issue `v_dual_* :: v_dual_*` 也是单条 encoded instruction，而不是表中两条。

<a id="sec-pc-counting-and-evidence-convention"></a>
### PC、计数和证据口径

本文同时区分三种计数：

| Term | Meaning |
|---|---|
| Static ISA count | `21_final_isa.s` 中出现的 opcode site 数 |
| Per-wave dynamic issue | 一个有效 wave 走完整 valid-expert path 时执行的次数 |
| Per-WG wave-issue count | 4 个 wave 的动态次数之和；不是唯一 cache transaction 数 |

本实例没有 wave-specialized TDM：4 个 wave 都执行相同静态 TDM site，但每个 wave 的 D#
含自己的 4-row/8-n16-row segment。因此：

- static `tensor_load_to_lds` = 24；
- per-wave dynamic = 24；
- per-WG wave issues = 96；
- static WMMAScale = 48；
- per-WG wave issues = 192。

若 `expert == 384`（padding sentinel），workgroup 在 `0x0684` 直接跳到 `s_endpgm`，这些
valid-path 动态计数不发生。

---

## 1. Kernel overview and software pipeline

<a id="sec-1-1-call-path-and-mathematical-definition"></a>
### 1.1 软件调用路径与 GEMM2 数学定义

实际调用链：

```text
aiter.fused_moe
  -> _maybe_grouped_gfx1250_a8w4_moe
  -> _grouped_a8w4_tdm_moe
  -> flydsl_grouped_gemm_a8w4_masked
  -> launch_gemm_a8w4_tdm
  -> @flyc.kernel gemm_a8w4_tdm_t16x512x128_...
```

TDM shortcut 默认由 `AITER_GROUPED_A8W4_TDM=1` 开启，并要求 `stage1_weight_layout ==
"gugu"` 且非 expert-parallel path。GEMM2 调用明确传入：

```python
flydsl_grouped_gemm_a8w4_masked(
    grouped_out, a2_payload, w2_u8, a2_scale, w2s_i32, psum,
    n_experts=E, contiguous_m=contiguous_m, N=model_dim, K=inter_dim,
    tile_m=tile_m2, tile_n=tile_n2, tile_k=tile_k2,
    out_is_f16=out_is_f16, a_is_fp4=_a_is_fp4,
    stage1_act=0, bias=_b2, num_buffers=num_buffers2,
)
```

因此 generated name 中：

- `noact` 对应 `stage1_act=0`；
- `qout0` 对应 `stage1_quant_out=0`；
- `bias1` **只表示布尔值 `has_bias=1`**，不是“使用 stage-1 bias1”；
- 实际传入 `arg_bias` 的 tensor 是 `_b2`，即 stage-2 `bias2`。

对属于 expert `e` 的有效 contiguous row `r`：

```text
C[r,n] =
  sum_{k=0}^{767}
    dequant_fp8(A2[r,k], SA[r, floor(k/32)])
  * dequant_fp4(B2[e,n,k], SB[e,n, floor(k/32)])
  + bias2[e,n]

grouped_out[r,n] = BF16(C[r,n])
```

数学矩阵形状：

| Tensor | Logical shape | Logical dtype | Mathematical role |
|---|---|---|---|
| A2 | `[30720, 768]` | MXFP8 E4M3 | math-left activation |
| B2 | `[384, 7168, 768]` | MXFP4 E2M1 | expert weight; math uses `B2[e]^T` |
| SA | `[30720, 24]` scale bytes | E8M0 | A2 K-block-32 row scales |
| SB | `[384, 7168, 24]` scale bytes | E8M0 | B2 K-block-32 output-column scales |
| bias2 | `[384, 7168]` | BF16 | post-accumulation bias |
| grouped_out | `[30720, 7168]` | BF16 | routed stage-2 output |

<a id="sec-1-2-workgroup-wave-lane-geometry"></a>
### 1.2 Workgroup / wave / lane 分工

每个 workgroup 覆盖一个 `16 x 512` output tile。`m_warp=1`，因此 4 个 wave 共享相同
16 行；`n_warp=4`，每个 wave 负责连续 128 列。

<div style="overflow-x: auto;">
<table style="white-space: nowrap; width: max-content; min-width: 1100px;">
  <thead>
    <tr>
      <th align="right">wave</th>
      <th align="right"><code>wave_m</code></th>
      <th align="right"><code>wave_n</code></th>
      <th>Output rows</th>
      <th>Output columns</th>
      <th>B2 n16 groups</th>
      <th>Accumulator fragments</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td align="right">0</td><td align="right">0</td><td align="right">0</td>
      <td><code>blk_m + 0..15</code></td><td><code>blk_n + 0..127</code></td>
      <td><code>0..7</code></td><td><code>wn0..wn7</code></td>
    </tr>
    <tr>
      <td align="right">1</td><td align="right">0</td><td align="right">1</td>
      <td><code>blk_m + 0..15</code></td><td><code>blk_n + 128..255</code></td>
      <td><code>8..15</code></td><td><code>wn0..wn7</code></td>
    </tr>
    <tr>
      <td align="right">2</td><td align="right">0</td><td align="right">2</td>
      <td><code>blk_m + 0..15</code></td><td><code>blk_n + 256..383</code></td>
      <td><code>16..23</code></td><td><code>wn0..wn7</code></td>
    </tr>
    <tr>
      <td align="right">3</td><td align="right">0</td><td align="right">3</td>
      <td><code>blk_m + 0..15</code></td><td><code>blk_n + 384..511</code></td>
      <td><code>24..31</code></td><td><code>wn0..wn7</code></td>
    </tr>
  </tbody>
</table>
</div>

wave 内：

```text
lane   = tid % 32
lane16 = lane % 16
kgrp   = lane // 16
row    = lane16
```

<div style="overflow-x: auto;">
<table style="white-space: nowrap; width: max-content; min-width: 920px;">
  <thead>
    <tr>
      <th>Lane range</th>
      <th align="right"><code>lane16</code></th>
      <th align="right"><code>kgrp</code></th>
      <th>Output row</th>
      <th>Each <code>wn</code> output columns</th>
      <th>A2/B2 fragment effect</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><code>0..15</code></td><td align="right"><code>lane</code></td><td align="right">0</td>
      <td><code>row = lane</code></td><td><code>wn*16 + 0..7</code></td>
      <td>first half of the per-row/per-column WMMA lane layout</td>
    </tr>
    <tr>
      <td><code>16..31</code></td><td align="right"><code>lane-16</code></td><td align="right">1</td>
      <td><code>row = lane-16</code></td><td><code>wn*16 + 8..15</code></td>
      <td>second half; lane <code>x</code> and <code>x+16</code> share one logical output row</td>
    </tr>
  </tbody>
</table>
</div>

<a id="sec-1-3-grid-swizzle-and-expert-lookup"></a>
### 1.3 Grid、DeepGEMM swizzle 与 expert lookup

Launch 先生成线性 grid：

```text
m_tiles = ceil(M / 16)
n_tiles = ceil(N / 512)
grid.x  = m_tiles * n_tiles
```

kernel 内再做 16-M-tile group swizzle：

```text
TILES_PER_GROUP = 16
blocks_per_group = total_n_tiles * 16
group             = bid_x // blocks_per_group
group_first_tile  = group * 16
in_group          = bid_x - group * blocks_per_group
group_tiles       = min(total_m_tiles - group_first_tile, 16)
m_tile            = group_first_tile + (in_group % group_tiles)
blk_m             = m_tile * 16
blk_n             = (in_group // group_tiles) * 512
```

该映射让一个 group 内先遍历固定的 16 个相邻 M tiles，再推进 N tile，改善同一 expert
权重 slab 的局部性。最后一个不足 16 M tiles 的 group 使用 `group_tiles` 收缩取模范围。

expert lookup 是在 `m_tile_map = psum` 上求：

```text
expert = lower_bound(psum, first value > blk_m)
mn_oob = psum[min(expert, 383)] - blk_m
```

`E=384` 时 source 展开 10 次 binary-search iteration。当前 ISA 中 10 次 search load 与
最后一次 extent load 的 PC 为：

| Purpose | Kernel-relative PCs |
|---|---|
| 10 binary-search loads | `0x003c`, `0x02d8`, `0x0314`, `0x0350`, `0x0390`, `0x03e0`, `0x041c`, `0x0458`, `0x04a0`, `0x04e8` |
| final `psum[expert]` / `mn_oob` | `0x0530` |
| `expert >= 384` padding skip | branch at `0x0684` |

第一次 load 被 backend 提前到 swizzle 算术期间：

```asm
; rel PC 0x003c, first midpoint = 192, byte offset = 192*4 = 768
global_load_b32 v2, v1, s[4:5] offset:768
```

M-tail 由 `mn_oob` 同时约束 A2 TDM load 和 output TDM store。相反，B2、SB 和 bias2
没有 N-tail predicate。虽然 launch 使用 `ceil(N/512)`，该 specialization 必须满足
**N 已 padding 或 `N % 512 == 0`**；本 workload `7168/512=14`，恰好满足。

<a id="sec-1-4-lds-arena-logical-vs-allocated"></a>
### 1.4 LDS arena：逻辑 74,752 B 与实际 75,776 B

每个 ring slot 的编译期布局：

| Region | Formula | Bytes | Hex |
|---|---:|---:|---:|
| A2 | `16 * (128 + 16)` | 2304 | `0x0900` |
| B2 | `(512/16) * ((128/2)*16)` | 32768 | `0x8000` |
| SA | `16 * 1 * 4` | 64 | `0x0040` |
| SB | `(512/32) * (128/4) * 4` | 2048 | `0x0800` |
| unpadded slot | sum | 37184 | `0x9140` |
| 512-B aligned pitch | `align_up(37184,512)` | 37376 | `0x9200` |

两级 ring 的逻辑 arena：

```text
ARENA_B = max(2 * 37376, C_STORE_B)
        = max(74752, 16*512*2)
        = 74752 B = 73.00 KiB = 0x12400
```

但 `tile_m <= 64` 时 source 为避免 SA 越界读到旧的 `0xFF` E8M0 NaN scale，先清零整个
arena。每轮 128 threads 各写 16 B，因此 `_zblk = 2048 B`；实际分配向 2048 B 上取整：

```text
_arena = ceil(74752 / 2048) * 2048
       = 37 * 2048
       = 75776 B = 74.00 KiB = 0x12800
```

这解释了三个容易混淆的数字：

| Layer | Reported size | Meaning |
|---|---:|---|
| source occupancy print | 74,752 B | ring/C-store 的逻辑最大 footprint |
| launch dynamic shared | 75,776 B | 实际 `gpu.launch_func dynamic_shared_memory_size` |
| ELF metadata `.group_segment_fixed_size` | 0 B | 仅表示没有 **fixed/static** group segment |

`00_origin.mlir` 明确含：

```mlir
%c75776_i32 = arith.constant 75776 : i32
gpu.launch_func ... dynamic_shared_memory_size %c75776_i32 ...
```

完整 LDS 地址图：

| Buffer | A2 | B2 | SA | SB | alignment/tail |
|---|---|---|---|---|---|
| slot 0 | `[0x0000,0x0900)` | `[0x0900,0x8900)` | `[0x8900,0x8940)` | `[0x8940,0x9140)` | `[0x9140,0x9200)` |
| slot 1 | `[0x9200,0x9b00)` | `[0x9b00,0x11b00)` | `[0x11b00,0x11b40)` | `[0x11b40,0x12340)` | `[0x12340,0x12400)` |
| rounded tail | - | - | - | - | `[0x12400,0x12800)` |
| C-store reuse | `[0x0000,0x4000)` | - | - | - | overwrites slot 0 only after final fence |

ISA 在 `0x053c..0x0664` 发出 37 个 `ds_store_b128` site。每个 wave 的每个 lane 都执行，
覆盖 `37 * 128 * 16 = 75,776 B`，随后：

```asm
; 0x0678
s_wait_dscnt 0x0
; 0x067c
s_barrier_signal -1
; 0x0680
s_barrier_wait -1
```

<a id="sec-1-5-two-buffer-tdm-pipeline"></a>
### 1.5 两级 TDM ring buffer 与六级 K pipeline

`K=768`、`tile_k=128`、`KWS=1`，所以最终 ISA 完全展开 6 个 K tiles。活跃代码采用
mid-compute prefetch：

1. prologue issue K0；
2. steady K0..K4：等待当前 slot ready，读取 B2/SB/SA，issue 下一 K tile 到另一个 slot，
   再读取 A2 并执行 8 次 WMMAScale；
3. drain K5，不再 prefetch；
4. final `pipeline_fence(0)` 后复用 LDS slot 0 作为 C staging。

每个 K tile 有 4 个静态 TDM load site：

<div style="overflow-x: auto;">
<table style="white-space: nowrap; width: max-content; min-width: 1180px;">
  <thead>
    <tr>
      <th align="right">kt</th>
      <th align="right">ring slot</th>
      <th>K range</th>
      <th>A2 TDM PC</th>
      <th>B2 TDM PC</th>
      <th>SA TDM PC</th>
      <th>SB TDM PC</th>
      <th>WMMA PC range</th>
    </tr>
  </thead>
  <tbody>
    <tr><td align="right">0</td><td align="right">0</td><td><code>0..127</code></td><td><code>0x07e8</code></td><td><code>0x09e4</code></td><td><code>0x0a40</code></td><td><code>0x0a4c</code></td><td><code>0x0c44..0x0cc4</code></td></tr>
    <tr><td align="right">1</td><td align="right">1</td><td><code>128..255</code></td><td><code>0x0b54</code></td><td><code>0x0bf0</code></td><td><code>0x0bfc</code></td><td><code>0x0c08</code></td><td><code>0x0e5c..0x0edc</code></td></tr>
    <tr><td align="right">2</td><td align="right">0</td><td><code>256..383</code></td><td><code>0x0db8</code></td><td><code>0x0e10</code></td><td><code>0x0e1c</code></td><td><code>0x0e28</code></td><td><code>0x105c..0x10dc</code></td></tr>
    <tr><td align="right">3</td><td align="right">1</td><td><code>384..511</code></td><td><code>0x0fb8</code></td><td><code>0x1010</code></td><td><code>0x101c</code></td><td><code>0x1028</code></td><td><code>0x126c..0x12ec</code></td></tr>
    <tr><td align="right">4</td><td align="right">0</td><td><code>512..639</code></td><td><code>0x11c4</code></td><td><code>0x120c</code></td><td><code>0x1228</code></td><td><code>0x1238</code></td><td><code>0x1470..0x14f0</code></td></tr>
    <tr><td align="right">5</td><td align="right">1</td><td><code>640..767</code></td><td><code>0x13cc</code></td><td><code>0x1424</code></td><td><code>0x1430</code></td><td><code>0x143c</code></td><td><code>0x15dc..0x1660</code>, 中间有 DS wait</td></tr>
  </tbody>
</table>
</div>

第一组完整计算骨架：

```asm
; K0 ready
0x0a58: s_wait_tensorcnt 0x0
0x0a5c: s_barrier_signal -1
0x0a60: s_barrier_wait -1

; B2 LDS -> VGPR: 16 x ds_load_b128
0x0a68: ds_load_b128 v[16:19], v166
        ...
0x0ae0: ds_load_b128 v[108:111], v166 offset:7680

; SB + SA scale loads
0x0ae8: ds_load_2addr_b32 ...
        ...
0x0b1c: ds_load_b32 v114, v168

; issue K1 to slot 1
0x0b54: tensor_load_to_lds ...   ; A2
0x0bf0: tensor_load_to_lds ...   ; B2
0x0bfc: tensor_load_to_lds ...   ; SA
0x0c08: tensor_load_to_lds ...   ; SB

; A2 LDS -> VGPR
0x0c20: ds_load_b128 v[0:3], v170
0x0c28: ds_load_b128 v[4:7], v170 offset:32
0x0c30: ds_load_b128 v[8:11], v170 offset:64
0x0c38: ds_load_b128 v[12:15], v170 offset:96
0x0c40: s_wait_dscnt 0x0

; 8 output N16 fragments
0x0c44: v_wmma_scale_f32_16x16x128_f8f6f4 ...
        ...
0x0cb4: v_wmma_scale_f32_16x16x128_f8f6f4 ...
```

<a id="sec-1-6-vop3px2-wmmascale-and-operand-swap"></a>
### 1.6 VOP3PX2 WMMAScale、硬件 operand swap 与 E8M0

MI400 guide §4.6.12（印刷页 143）将 WMMA 定义为一整个 wave 共同持有一组 A/B/C/D
matrix，并计算 `A*B+C -> D`；不是每 lane 一次独立矩阵乘。Table 59（印刷页 145）列出
`V_WMMA_SCALE_F32_16X16X128_F8F6F4` 的：

```text
Matrix A : 16x128 FP4/FP6/FP8
Matrix B : 128x16 FP4/FP6/FP8
C/D      : 16x16 FP32
```

guide §4.6.12（印刷页 147）和 §4.6.12.6（印刷页 154-155）还说明：

- scaled opcode 使用 4-DWORD VOP3PX2 encoding；
- block-scale 值为 8-bit E8M0/E5M3/E4M3；
- `SCALE` 版本每 32 个 K values 使用一个 scale；
- scale 在普通 dot product 后、累加前应用；
- WMMA 为 wave32 operation。

当前 gfx1250 代码对象独立验证了其中与本实例有关的部分：每条 scaled WMMA 恰为 16 B，
opcode 是 `v_wmma_scale_f32_16x16x128_f8f6f4`，wave size metadata 为 32，A/B scale 均
为 E8M0 block-32。MI400 guide 的性能速率、reuse/clause errata 不作为 gfx1250 结论。

FlyDSL 调用呈现一个容易误读的硬件 operand swap：

```python
fx.gemm(
    wmma_atom,
    c_frags[idx],
    wt[wn],       # first hardware matrix operand: B2 / weight
    act[i],       # second hardware matrix operand: A2 / activation
    c_frags[idx],
    scale_a=sb_k[wn],
    scale_b=sa_k[wm],
)
```

| View | Math-left | Math-right | HW first matrix operand | HW second matrix operand | Scale pairing |
|---|---|---|---|---|---|
| GEMM2 | A2 activation | `B2[e]^T` weight | B2 FP4 fragment | A2 FP8 fragment | first/B2 uses SB; second/A2 uses SA |

第一条 ISA 直接显示这种顺序：

```asm
; 0x0c44
v_wmma_scale_f32_16x16x128_f8f6f4 \
  v[72:79],       /* D/C FP32 */ \
  v[16:23],       /* first matrix operand: B2 FP4 */ \
  v[0:15],        /* second matrix operand: A2 FP8 */ \
  0,              /* initial C */ \
  v116,           /* SB */ \
  v114            /* SA */ \
  matrix_a_fmt:MATRIX_FMT_FP4
```

这里的“Matrix A/B”是硬件 instruction operand 命名，不能直接等同于应用层 `A2/B2`
变量名。数学结果的最终 row/column 解释由 WMMAScale lane layout 与 epilogue store mapping
共同恢复。

<a id="sec-1-7-waits-barriers-and-visibility"></a>
### 1.7 Wait、barrier 与可见性

MI400 guide §4.10（印刷页 197-199）说明 TDM 可以与 shader 指令并行，descriptor 由
SGPR D# 提供；completion 由 TENSORcnt 跟踪，`S_WAIT_TENSORCNT` 一次等待的是 tensor
instruction completion。§4.3.7（印刷页 85、88）区分 TENSORcnt、DScnt、LOADcnt 等
dependency counters。

本 kernel 的同步协议是：

```text
TDM writes LDS
  -> s_wait_tensorcnt(0)
  -> workgroup barrier signal/wait
  -> every wave may read the completed slot

VGPR -> LDS output stores
  -> s_wait_dscnt(0)
  -> workgroup barrier signal/wait
  -> tensor_store_from_lds
  -> s_wait_tensorcnt(0)
```

静态 wait/barrier 计数：

| Opcode | Static count | Main role |
|---|---:|---|
| `s_wait_tensorcnt` | 8 | 6 K-slot fences + final compute fence + output store completion |
| `s_barrier_signal` | 9 | zero-init、6 K slots、postcompute、C store |
| `s_barrier_wait` | 9 | 与 signal 配对 |
| `s_wait_dscnt` | 9 | zero stores、A/B LDS reads、C LDS writes |
| `s_wait_loadcnt` | 19 | expert map 和 bias2 global loads |
| `s_wait_kmcnt` | 3 | scalar kernarg/data dependencies |
| `s_wait_xcnt` | 4 | address translation completion |
| `s_wait_alu` | 37 | gfx1250 ALU dependency |

六个 K-slot fence：

```text
0x0a58, 0x0ccc, 0x0edc, 0x10dc, 0x12ec, 0x14f8
```

postcompute barrier 被 scheduler 拆开：

```asm
0x1670: s_wait_tensorcnt 0x0
0x1674: s_barrier_signal -1
        ; output/bias address arithmetic is scheduled here
0x16d8: s_barrier_wait -1
```

这不是少了一半 barrier；signal 与 wait 之间的独立地址计算正是在利用 split form 隐藏延迟。

<a id="sec-1-8-full-pc-phase-map"></a>
### 1.8 完整 kernel-relative PC phase map

| Phase | ISA source lines | Relative PC range | ELF VA range | Main work |
|---|---:|---:|---:|---|
| Entry / kernarg preload | 8-17 | `[0x0000,0x003c)` | `[0x1900,0x193c)` | mode setup、initial scalar loads |
| Swizzle + hoisted lookup | 18-158 | `[0x003c,0x02a4)` | `[0x193c,0x1ba4)` | grid division、first psum load |
| Expert lower-bound | 159-296 | `[0x02a4,0x053c)` | `[0x1ba4,0x1e3c)` | remaining search + `mn_oob` |
| LDS zero + barrier | 297-340 | `[0x053c,0x0684)` | `[0x1e3c,0x1f84)` | clear all 75,776 B |
| Padding branch | 341 | `[0x0684,0x0688)` | `[0x1f84,0x1f88)` | skip expert sentinel |
| D# setup + initial TDM | 342-521 | `[0x0688,0x0a58)` | `[0x1f88,0x2358)` | A2/B2/SA/SB K0 descriptors |
| K0 + prefetch K1 | 522-609 | `[0x0a58,0x0ccc)` | `[0x2358,0x25cc)` | first 8 WMMAs |
| K1 + prefetch K2 | 610-678 | `[0x0ccc,0x0edc)` | `[0x25cc,0x27dc)` | accumulate |
| K2 + prefetch K3 | 679-743 | `[0x0edc,0x10dc)` | `[0x27dc,0x29dc)` | accumulate |
| K3 + prefetch K4 | 744-812 | `[0x10dc,0x12ec)` | `[0x29dc,0x2bec)` | accumulate |
| K4 + prefetch K5 | 813-879 | `[0x12ec,0x14f8)` | `[0x2bec,0x2df8)` | accumulate |
| K5 drain | 880-922 | `[0x14f8,0x1670)` | `[0x2df8,0x2f70)` | last 8 WMMAs |
| Final fence / address setup | 923-949 | `[0x1670,0x16f0)` | `[0x2f70,0x2ff0)` | prepare bias/C addresses |
| Bias load + unpack | 950-1033 | `[0x16f0,0x194c)` | `[0x2ff0,0x324c)` | 8 BF16 vector loads |
| Bias FP32 add | 1034-1065 | `[0x194c,0x1a4c)` | `[0x324c,0x334c)` | 32 packed adds |
| BF16 conversion | 1066-1097 | `[0x1a4c,0x1b4c)` | `[0x334c,0x344c)` | 32 packed converts |
| C LDS stores | 1098-1110 | `[0x1b4c,0x1ba0)` | `[0x344c,0x34a0)` | 8 `ds_store_b128` |
| C fence + TDM store | 1111-1115 | `[0x1ba0,0x1bbc)` | `[0x34a0,0x34bc)` | LDS -> GM |
| End | 1117 | `[0x1bbc,0x1bc0)` | `[0x34bc,0x34c0)` | `s_endpgm` |

---

## 2. Full GM/LDS/VGPR layout maps

记号：

```text
e       : expert index
r       : contiguous-M row
n       : output column
kt      : K-tile index 0..5
k       : local K in a K128 tile
wave_n  : wave index 0..3
wn      : per-wave N16 fragment 0..7
lane16  : lane % 16
kgrp    : lane // 16
slot    : kt % 2
PITCH   : 0x9200 bytes
```

“logical layout”描述数学 tensor；“physical/preshuffled layout”描述 TDM 和 WMMA 实际读取的
bytes/dwords。两者不可混用。

<a id="sec-2-1-a2-gm-to-lds"></a>
### 2.1 A2 payload：GM -> LDS

#### Transition table：A2 logical/physical GM -> TDM -> LDS

| Stage | Shape / coordinate | Storage | Address / stride | Verified primitive |
|---|---|---|---|---|
| Logical GM | `[30720,768]`, `(r,k)` | MXFP8 E4M3 | row-major, 1 B/value | wrapper contract |
| Physical GM tile | `[16,128]` | 2048 payload bytes | `arg_a + blk_m*768 + kt*128` | TDM D# |
| Per-wave TDM segment | `[4,128]` | 512 bytes/wave | GM row stride 768 | `tensor_load_to_lds` |
| LDS tile | `[16,128]` with padded row | 2304 bytes | `slot*0x9200 + row*144 + k` | D# pad interval 128, amount 16 |

MI400 guide §4.10.2（印刷页 199-200）给出的 D# 概念与此对应：`global_addr` 是 tile
起点，`tensor_dim_stride` 是 GM line stride，`lds_addr` 是 tile LDS 起点；load 可以规则性
插入 LDS padding。这里 padding 由 source 明确设为每 128 B 插 16 B。

<div style="overflow-x: auto;">
<table style="white-space: nowrap; width: max-content; min-width: 1180px;">
  <thead>
    <tr>
      <th align="right">wave</th>
      <th>GM rows</th>
      <th>GM byte offset from tile base</th>
      <th>LDS rows</th>
      <th>LDS byte base, slot 0</th>
      <th>LDS byte base, slot 1</th>
      <th>Bytes moved / K tile</th>
    </tr>
  </thead>
  <tbody>
    <tr><td align="right">0</td><td><code>blk_m+0..3</code></td><td><code>0*768</code></td><td><code>0..3</code></td><td><code>0x0000</code></td><td><code>0x9200</code></td><td>512</td></tr>
    <tr><td align="right">1</td><td><code>blk_m+4..7</code></td><td><code>4*768</code></td><td><code>4..7</code></td><td><code>0x0240</code></td><td><code>0x9440</code></td><td>512</td></tr>
    <tr><td align="right">2</td><td><code>blk_m+8..11</code></td><td><code>8*768</code></td><td><code>8..11</code></td><td><code>0x0480</code></td><td><code>0x9680</code></td><td>512</td></tr>
    <tr><td align="right">3</td><td><code>blk_m+12..15</code></td><td><code>12*768</code></td><td><code>12..15</code></td><td><code>0x06c0</code></td><td><code>0x98c0</code></td><td>512</td></tr>
  </tbody>
</table>
</div>

A2 是唯一使用 `mn_oob` row extent 的 input payload；最后一个 expert M tile 超出的行由
TDM OOB 语义读为零。

<a id="sec-2-2-a2-lds-to-vgpr"></a>
### 2.2 A2 payload：LDS -> VGPR

#### Transition table：A2 LDS -> DS reads -> VGPR WMMA operand

| Stage | Per-lane representation | Address formula | ISA / count |
|---|---|---|---|
| LDS | one padded A2 row | `row*144 + kt_local*128 + kgrp*16` | source `load_a` |
| DS read | four 16-B vectors | `b0 + {0,32,64,96}` | 4 × `ds_load_b128` / K stage |
| VGPR | 16 dwords = 64 bytes FP8 | shuffle concatenates four vectors | WMMAScale second matrix operand |

在本实例 `KWS=1`，`kt_local=0`；外层 `kt` 已经体现在 TDM 的 GM offset 和 ring slot。

<div style="overflow-x: auto;">
<table style="white-space: nowrap; width: max-content; min-width: 1100px;">
  <thead>
    <tr>
      <th>Lane range</th>
      <th align="right"><code>lane16</code></th>
      <th align="right"><code>kgrp</code></th>
      <th>LDS base before immediate offsets</th>
      <th>DS immediate offsets</th>
      <th>VGPR payload</th>
      <th>Logical role</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><code>0..15</code></td><td align="right"><code>0..15</code></td><td align="right">0</td>
      <td><code>slot*0x9200 + lane16*144</code></td>
      <td><code>0,32,64,96</code></td><td>16 dwords / 64 FP8 bytes</td>
      <td>A2 activation operand, rows 0..15</td>
    </tr>
    <tr>
      <td><code>16..31</code></td><td align="right"><code>0..15</code></td><td align="right">1</td>
      <td><code>slot*0x9200 + lane16*144 + 16</code></td>
      <td><code>0,32,64,96</code></td><td>16 dwords / 64 FP8 bytes</td>
      <td>paired half of the same logical row</td>
    </tr>
  </tbody>
</table>
</div>

6 stages 共 24 个静态 A2 `ds_load_b128`。第一组在 `0x0c20..0x0c40`，slot 1 的下一组
在 `0x0e38..0x0e58`。精确的 VGPR byte-to-matrix-element permutation 属于
WMMAScale wave lane layout；本报告验证 LDS 地址和 16-dword payload，但不凭变量名臆造
guide 未直接覆盖的 gfx1250 lane permutation。

<a id="sec-2-3-b2-gm-to-lds"></a>
### 2.3 B2 weight：GM -> LDS

#### Transition table：B2 logical -> preshuffled GM -> TDM -> LDS

| Stage | Shape / coordinate | Storage | Address / stride | Verified primitive |
|---|---|---|---|---|
| Logical weight | `[384,7168,768]` | MXFP4 E2M1 | 4-bit/value | application tensor |
| Packed physical | `[384,7168,384]` bytes | 2 values/byte | gfx1250 shuffled weight | wrapper `uint8` view |
| n16 physical row | one 16-column group | 6144 bytes | `(K/2)*16 = 6144` | `Kp16` |
| One K128 tile | `[32 n16 groups,1024 bytes]` | 32768 bytes | `kt*1024` within each n16 row | TDM D# |
| LDS | `[32,1024]` | 32768 bytes | base `slot*0x9200 + 0x900` | `tensor_load_to_lds` |

GM base：

```text
b_outer_row = expert*(N/16) + blk_n/16
b_off0      = b_outer_row * 6144
B2_GM       = arg_b + b_off0 + kt*1024
```

<div style="overflow-x: auto;">
<table style="white-space: nowrap; width: max-content; min-width: 1240px;">
  <thead>
    <tr>
      <th align="right">wave</th>
      <th>Output columns</th>
      <th>n16 groups</th>
      <th>GM offset from B2 tile base</th>
      <th>LDS base, slot 0</th>
      <th>LDS base, slot 1</th>
      <th>Bytes moved / K tile</th>
    </tr>
  </thead>
  <tbody>
    <tr><td align="right">0</td><td><code>0..127</code></td><td><code>0..7</code></td><td><code>0*6144</code></td><td><code>0x0900</code></td><td><code>0x9b00</code></td><td>8192</td></tr>
    <tr><td align="right">1</td><td><code>128..255</code></td><td><code>8..15</code></td><td><code>8*6144</code></td><td><code>0x2900</code></td><td><code>0xbb00</code></td><td>8192</td></tr>
    <tr><td align="right">2</td><td><code>256..383</code></td><td><code>16..23</code></td><td><code>16*6144</code></td><td><code>0x4900</code></td><td><code>0xdb00</code></td><td>8192</td></tr>
    <tr><td align="right">3</td><td><code>384..511</code></td><td><code>24..31</code></td><td><code>24*6144</code></td><td><code>0x6900</code></td><td><code>0xfb00</code></td><td>8192</td></tr>
  </tbody>
</table>
</div>

这里列出的 GM offset 是 preshuffled n16-row 间距，不是 math-logical `[n,k]` row-major
offset。TDM 不理解“FP4 matrix”；它只搬运 32 个 1024-B physical rows。

<a id="sec-2-4-b2-lds-to-vgpr"></a>
### 2.4 B2 weight：LDS -> VGPR

#### Transition table：B2 LDS -> DS reads -> VGPR WMMA operand

| Stage | Per-lane representation | Address formula | ISA / count |
|---|---|---|---|
| LDS | one n16/K128 preshuffled row | `0x900 + (wave_n*8+wn)*1024 + kgrp*256 + lane16*16` | source `load_b` |
| DS read | two 16-B vectors | `b0` and `b0+512` | 2 × `ds_load_b128` / `wn` |
| VGPR | 8 dwords = 32 packed bytes | shuffle of two vectors | WMMAScale first matrix operand |

<div style="overflow-x: auto;">
<table style="white-space: nowrap; width: max-content; min-width: 1260px;">
  <thead>
    <tr>
      <th align="right"><code>wn</code></th>
      <th>Wave-local output columns</th>
      <th>LDS n16 row</th>
      <th><code>kgrp=0</code> base addition</th>
      <th><code>kgrp=1</code> base addition</th>
      <th>DS addresses</th>
      <th>VGPR result</th>
    </tr>
  </thead>
  <tbody>
    <tr><td align="right">0</td><td><code>0..15</code></td><td><code>wave_n*8+0</code></td><td><code>lane16*16</code></td><td><code>256+lane16*16</code></td><td><code>b0, b0+512</code></td><td>8 dwords</td></tr>
    <tr><td align="right">1</td><td><code>16..31</code></td><td><code>wave_n*8+1</code></td><td><code>lane16*16</code></td><td><code>256+lane16*16</code></td><td><code>b0, b0+512</code></td><td>8 dwords</td></tr>
    <tr><td align="right">2</td><td><code>32..47</code></td><td><code>wave_n*8+2</code></td><td><code>lane16*16</code></td><td><code>256+lane16*16</code></td><td><code>b0, b0+512</code></td><td>8 dwords</td></tr>
    <tr><td align="right">3</td><td><code>48..63</code></td><td><code>wave_n*8+3</code></td><td><code>lane16*16</code></td><td><code>256+lane16*16</code></td><td><code>b0, b0+512</code></td><td>8 dwords</td></tr>
    <tr><td align="right">4</td><td><code>64..79</code></td><td><code>wave_n*8+4</code></td><td><code>lane16*16</code></td><td><code>256+lane16*16</code></td><td><code>b0, b0+512</code></td><td>8 dwords</td></tr>
    <tr><td align="right">5</td><td><code>80..95</code></td><td><code>wave_n*8+5</code></td><td><code>lane16*16</code></td><td><code>256+lane16*16</code></td><td><code>b0, b0+512</code></td><td>8 dwords</td></tr>
    <tr><td align="right">6</td><td><code>96..111</code></td><td><code>wave_n*8+6</code></td><td><code>lane16*16</code></td><td><code>256+lane16*16</code></td><td><code>b0, b0+512</code></td><td>8 dwords</td></tr>
    <tr><td align="right">7</td><td><code>112..127</code></td><td><code>wave_n*8+7</code></td><td><code>lane16*16</code></td><td><code>256+lane16*16</code></td><td><code>b0, b0+512</code></td><td>8 dwords</td></tr>
  </tbody>
</table>
</div>

每 K stage 16 个 B2 `ds_load_b128`，6 stages 共 96 个静态 site。K0 的 immediate offset
序列为 `0,512,1024,...,7680`，位于 `0x0a68..0x0ae8`。

<a id="sec-2-5-sa-layout"></a>
### 2.5 SA：A2 的 E8M0 scale，GM -> LDS -> VGPR

#### Transition table：SA logical -> preshuffled GM -> LDS -> VGPR

| Stage | Shape / representation | Address / stride | Transfer |
|---|---|---|---|
| Logical SA | `[30720,24]` bytes | one E8M0 byte per A2 row per K32 | quantization contract |
| Physical SA | `[30720,6]` i32 | 4 adjacent K32 scale bytes / dword | wrapper `.view(int32)` |
| Per-K128 GM tile | `[16,1]` i32 | base dword `blk_m*6 + kt` | 64 B TDM payload |
| LDS | 16 dwords | `slot*0x9200 + 0x8900 + row*4` | `tensor_load_to_lds` |
| VGPR | one i32/lane | source formula may add `kgrp*4` | `ds_load_b32` |

<div style="overflow-x: auto;">
<table style="white-space: nowrap; width: max-content; min-width: 1000px;">
  <thead>
    <tr>
      <th align="right">kt</th>
      <th>K values</th>
      <th>Logical E8M0 byte indices</th>
      <th>Physical SA dword</th>
      <th>GM byte advance</th>
      <th>Ring slot</th>
    </tr>
  </thead>
  <tbody>
    <tr><td align="right">0</td><td><code>0..127</code></td><td><code>0..3</code></td><td>0</td><td><code>0</code></td><td>0</td></tr>
    <tr><td align="right">1</td><td><code>128..255</code></td><td><code>4..7</code></td><td>1</td><td><code>4</code></td><td>1</td></tr>
    <tr><td align="right">2</td><td><code>256..383</code></td><td><code>8..11</code></td><td>2</td><td><code>8</code></td><td>0</td></tr>
    <tr><td align="right">3</td><td><code>384..511</code></td><td><code>12..15</code></td><td>3</td><td><code>12</code></td><td>1</td></tr>
    <tr><td align="right">4</td><td><code>512..639</code></td><td><code>16..19</code></td><td>4</td><td><code>16</code></td><td>0</td></tr>
    <tr><td align="right">5</td><td><code>640..767</code></td><td><code>20..23</code></td><td>5</td><td><code>20</code></td><td>1</td></tr>
  </tbody>
</table>
</div>

SA LDS read 公式：

```text
warp_lds_row = lane16
SA_byte = slot*0x9200 + 0x8900
        + lane16*4 + kgrp*4
```

`kgrp=1` 可读到 nominal row footprint 后一个 dword，这正是 source 对 `tile_m<=64` 先把
整个实际 arena 清零的原因。清零行为和 source rationale 是 verified；某个具体 lane 最终
取 dword 内哪一个 E8M0 byte，由 WMMAScale scale selector 决定。

<a id="sec-2-6-sb-layout"></a>
### 2.6 SB：B2 的 n32k4 E8M0 scale，GM -> LDS -> VGPR

#### Transition table：SB logical -> n32k4 GM -> LDS -> VGPR

| Stage | Shape / representation | Address / stride | Transfer |
|---|---|---|---|
| Logical SB | `[384,7168,24]` scale bytes | one E8M0 byte / output col / K32 | weight quantization |
| Preshuffled GM | `[384,7168/32,24*32]` bytes | n32k4; 768 B / N32 super-row | wrapper contract |
| int32 view | `[384,224,192]` i32 | `K4=768/4=192` dwords | kernel address units |
| Per-K128 GM tile | `[16 N32 supers,32 dwords]` | `sb_off0 + kt*32` dwords | 2048 B TDM |
| LDS | `[16,32]` i32 | `slot*0x9200 + 0x8940` | 2048 B |
| VGPR | one i32 per `wn` per lane | formula below | eight SB dwords / K stage / lane |

GM base：

```text
N_SUPERS  = ceil(N/32) = 224
sb_off0   = expert*(224*192) + (blk_n/32)*192    # i32 units
kt advance = 32 i32 = 128 bytes
```

<div style="overflow-x: auto;">
<table style="white-space: nowrap; width: max-content; min-width: 1160px;">
  <thead>
    <tr>
      <th align="right">wave</th>
      <th>Output columns</th>
      <th>N32 supers</th>
      <th>GM dword offset from tile base</th>
      <th>LDS byte base, slot 0</th>
      <th>LDS byte base, slot 1</th>
      <th>Bytes moved</th>
    </tr>
  </thead>
  <tbody>
    <tr><td align="right">0</td><td><code>0..127</code></td><td><code>0..3</code></td><td><code>0*192</code></td><td><code>0x8940</code></td><td><code>0x11b40</code></td><td>512</td></tr>
    <tr><td align="right">1</td><td><code>128..255</code></td><td><code>4..7</code></td><td><code>4*192</code></td><td><code>0x8b40</code></td><td><code>0x11d40</code></td><td>512</td></tr>
    <tr><td align="right">2</td><td><code>256..383</code></td><td><code>8..11</code></td><td><code>8*192</code></td><td><code>0x8d40</code></td><td><code>0x11f40</code></td><td>512</td></tr>
    <tr><td align="right">3</td><td><code>384..511</code></td><td><code>12..15</code></td><td><code>12*192</code></td><td><code>0x8f40</code></td><td><code>0x12140</code></td><td>512</td></tr>
  </tbody>
</table>
</div>

SB LDS read：

```text
col_rel = wave_n*128 + wn*16 + lane16
SB_byte = slot*0x9200 + 0x8940
        + ((col_rel//32)*32 + (col_rel%32))*4
```

每 K stage source 产生 8 个 `load_sb` dword result，外加 1 个 SA dword。最终 backend 将部分
相邻操作合并为 `ds_load_2addr_b32`。6 stages 的 scale result 总数为 54 dwords：

```text
28 * ds_load_b32 + 13 * ds_load_2addr_b32 * 2 results = 54
```

<a id="sec-2-7-wmma-and-accumulator-layout"></a>
### 2.7 A2/B2/SA/SB -> WMMAScale -> 8 个 accumulator fragments

#### Transition table：VGPR operands/scales -> FP32 accumulator

| Input | VGPR width / lane | Hardware role | Scale |
|---|---:|---|---|
| B2 FP4 | 8 dwords | first WMMAScale matrix operand | SB dword |
| A2 FP8 | 16 dwords | second WMMAScale matrix operand | SA dword |
| C/D | 8 f32 | one N16 output fragment | initial zero, then self-accumulate |

`wmma_m_rep=1`，`wmma_n_rep=8`，所以每 wave 有 8 个
`fx.make_rmem_tensor(8, Float32)`。一个 fragment 每 lane 8 个 FP32，8 fragments 合计
每 lane 64 FP32。

<div style="overflow-x: auto;">
<table style="white-space: nowrap; width: max-content; min-width: 1320px;">
  <thead>
    <tr>
      <th align="right">fragment / wn</th>
      <th>Wave-local N16 columns</th>
      <th><code>kgrp=0</code> store columns</th>
      <th><code>kgrp=1</code> store columns</th>
      <th>K0 accumulator VGPRs</th>
      <th>K0 B2 VGPRs</th>
      <th>K0 SB VGPR</th>
      <th>Shared SA VGPR</th>
    </tr>
  </thead>
  <tbody>
    <tr><td align="right">0</td><td><code>0..15</code></td><td><code>0..7</code></td><td><code>8..15</code></td><td><code>v[72:79]</code></td><td><code>v[16:23]</code></td><td><code>v116</code></td><td><code>v114</code></td></tr>
    <tr><td align="right">1</td><td><code>16..31</code></td><td><code>16..23</code></td><td><code>24..31</code></td><td><code>v[64:71]</code></td><td><code>v[24:31]</code></td><td><code>v48</code></td><td><code>v114</code></td></tr>
    <tr><td align="right">2</td><td><code>32..47</code></td><td><code>32..39</code></td><td><code>40..47</code></td><td><code>v[56:63]</code></td><td><code>v[32:39]</code></td><td><code>v49</code></td><td><code>v114</code></td></tr>
    <tr><td align="right">3</td><td><code>48..63</code></td><td><code>48..55</code></td><td><code>56..63</code></td><td><code>v[48:55]</code></td><td><code>v[40:47]</code></td><td><code>v117</code></td><td><code>v114</code></td></tr>
    <tr><td align="right">4</td><td><code>64..79</code></td><td><code>64..71</code></td><td><code>72..79</code></td><td><code>v[40:47]</code></td><td><code>v[80:87]</code></td><td><code>v118</code></td><td><code>v114</code></td></tr>
    <tr><td align="right">5</td><td><code>80..95</code></td><td><code>80..87</code></td><td><code>88..95</code></td><td><code>v[32:39]</code></td><td><code>v[88:95]</code></td><td><code>v119</code></td><td><code>v114</code></td></tr>
    <tr><td align="right">6</td><td><code>96..111</code></td><td><code>96..103</code></td><td><code>104..111</code></td><td><code>v[16:23]</code></td><td><code>v[96:103]</code></td><td><code>v112</code></td><td><code>v114</code></td></tr>
    <tr><td align="right">7</td><td><code>112..127</code></td><td><code>112..119</code></td><td><code>120..127</code></td><td><code>v[24:31]</code></td><td><code>v[104:111]</code></td><td><code>v113</code></td><td><code>v114</code></td></tr>
  </tbody>
</table>
</div>

K0 的 C operand 为 inline zero；K1..K5 的 C operand 是同一 fragment 的旧 FP32 值。register
allocator 会在 B2 operand 最后一次使用后复用其 VGPR，因此不能把某个物理 `vNN` 在整个
kernel 中永久命名为“weight”或“accumulator”。

每个 wave 的 48 次 WMMA：

```text
6 K stages * 8 N16 fragments = 48
```

每 workgroup 的数学工作：

```text
16 * 512 * 768 MAC
= 6,291,456 MAC
= 12,582,912 FLOPs
```

<a id="sec-2-8-bias2-epilogue"></a>
### 2.8 bias2：GM -> VGPR -> FP32 accumulator

#### Transition table：bias2 GM -> BF16 VGPR -> widened FP32 -> accumulator

| Stage | Shape / representation | Address | ISA |
|---|---|---|---|
| Logical GM | `[384,7168]` BF16 | expert-major | stage-2 `bias2` |
| Per-lane load | 8 contiguous BF16 | `bias2 + expert*N + col_rel` | `global_load_b128` |
| Widened VGPR | 8 FP32 | BF16 high-half reconstruction / conversion | ISA unpack sequence |
| Accumulator update | 8 FP32 | `acc += bias` | 32 × `v_pk_add_f32` total |

```text
col_rel = wave_n*128 + wn*16 + kgrp*8
```

<div style="overflow-x: auto;">
<table style="white-space: nowrap; width: max-content; min-width: 1080px;">
  <thead>
    <tr>
      <th align="right">wave</th>
      <th>Bias columns</th>
      <th>Static vector loads</th>
      <th>Per-load lane-group addresses</th>
      <th>Accumulator targets</th>
    </tr>
  </thead>
  <tbody>
    <tr><td align="right">0</td><td><code>blk_n+0..127</code></td><td>8 × 16 B</td><td><code>wn*16 + kgrp*8</code></td><td><code>frag0..7</code></td></tr>
    <tr><td align="right">1</td><td><code>blk_n+128..255</code></td><td>8 × 16 B</td><td><code>128 + wn*16 + kgrp*8</code></td><td><code>frag0..7</code></td></tr>
    <tr><td align="right">2</td><td><code>blk_n+256..383</code></td><td>8 × 16 B</td><td><code>256 + wn*16 + kgrp*8</code></td><td><code>frag0..7</code></td></tr>
    <tr><td align="right">3</td><td><code>blk_n+384..511</code></td><td>8 × 16 B</td><td><code>384 + wn*16 + kgrp*8</code></td><td><code>frag0..7</code></td></tr>
  </tbody>
</table>
</div>

exact ISA：

```asm
; 0x16f0..0x1750: 8 static BF16 vector loads
global_load_b128 v[0:3],   v[96:97], off
global_load_b128 v[4:7],   v[96:97], off offset:32
global_load_b128 v[8:11],  v[96:97], off offset:64
global_load_b128 v[12:15], v[96:97], off offset:96
global_load_b128 v[80:83], v[96:97], off offset:128
global_load_b128 v[84:87], v[96:97], off offset:160
global_load_b128 v[88:91], v[96:97], off offset:192
global_load_b128 v[92:95], v[96:97], off offset:224

; 0x194c..0x1a4c
v_pk_add_f32 ...  ; 32 static instructions
```

`stage1_act=0` 使 SiLU/SwiGLU 分支在 compile time 消失；因此这是 **bias-only noact
epilogue**。`f32_swiglu_limit` 仍存在于通用 ABI，但本 specialization 不使用。

<a id="sec-2-9-output-layout"></a>
### 2.9 Output：accumulator -> BF16 VGPR -> LDS -> GM

#### Transition table：FP32 accumulator -> BF16 VGPR -> LDS -> TDM -> GM

| Stage | Shape / representation | Address / mapping | ISA |
|---|---|---|---|
| FP32 accumulators | 8 fragments × 8 f32/lane | row=`lane16`, cols by `wave_n/wn/kgrp` | WMMAScale D |
| BF16 VGPR | 64 BF16/lane = 32 dwords | packed pairs | 32 × `v_cvt_pk_bf16_f32` |
| LDS C staging | `[16,512]` BF16 | `(row*512 + col)*2` | 8 × `ds_store_b128` / wave |
| TDM store segment | `[4,512]` BF16 / wave | LDS base `wave*4096` | `tensor_store_from_lds` |
| Logical GM | `[30720,7168]` BF16 | `arg_c + blk_m*N + blk_n`, row stride `N` | output `grouped_out` |

BF16 conversion exact range：

```text
0x1a4c..0x1b4c : 32 x v_cvt_pk_bf16_f32
```

LDS address：

```text
row_rel = lane16
col_rel = wave_n*128 + wn*16 + kgrp*8
C_lds_byte = (row_rel*512 + col_rel) * 2
```

<div style="overflow-x: auto;">
<table style="white-space: nowrap; width: max-content; min-width: 1220px;">
  <thead>
    <tr>
      <th align="right">wave</th>
      <th>Columns staged by wave</th>
      <th>Per-lane row</th>
      <th>8 DS-store immediate offsets</th>
      <th>TDM-assigned rows</th>
      <th>TDM LDS base</th>
      <th>GM row stride</th>
    </tr>
  </thead>
  <tbody>
    <tr><td align="right">0</td><td><code>0..127</code></td><td><code>lane16</code></td><td><code>0,32,...,224</code></td><td><code>0..3</code></td><td><code>0x0000</code></td><td><code>N=7168</code></td></tr>
    <tr><td align="right">1</td><td><code>128..255</code></td><td><code>lane16</code></td><td><code>0,32,...,224</code></td><td><code>4..7</code></td><td><code>0x1000</code></td><td><code>N=7168</code></td></tr>
    <tr><td align="right">2</td><td><code>256..383</code></td><td><code>lane16</code></td><td><code>0,32,...,224</code></td><td><code>8..11</code></td><td><code>0x2000</code></td><td><code>N=7168</code></td></tr>
    <tr><td align="right">3</td><td><code>384..511</code></td><td><code>lane16</code></td><td><code>0,32,...,224</code></td><td><code>12..15</code></td><td><code>0x3000</code></td><td><code>N=7168</code></td></tr>
  </tbody>
</table>
</div>

所有 4 个 wave 先把各自 128-column stripe 写满 16 行 LDS；barrier 后，TDM 再把 row
维度分成 4-row segments。两种分工是正交的：DS store 按 N stripe 分，TDM store 按 M
rows 分。

```asm
; 0x1b50..0x1ba0
ds_store_b128 v130, v[0:3]
ds_store_b128 v130, v[4:7] offset:32
...
ds_store_b128 v130, v[28:31] offset:224

0x1ba0: s_wait_dscnt 0x0
0x1ba4: s_barrier_signal -1
0x1ba8: s_barrier_wait -1
0x1bac: tensor_store_from_lds s[44:47], s[4:11]
0x1bb8: s_wait_tensorcnt 0x0
```

输出 TDM D# 使用 `mn_oob` 作为 M extent，因而 M tail writes 被 drop。N extent 没有 tail
mask，再次要求 N 为 512 对齐或物理 padding。

<a id="sec-2-10-end-to-end-transition-map"></a>
### 2.10 端到端 tensor transition 总图

```text
A2 logical MXFP8 GM [M,K]
  -> physical row-major FP8 bytes
  -> TDM [16,128], GM stride 768
  -> padded LDS A slot [16,144 B]
  -> 4 x ds_load_b128/lane
  -> 16-dword FP8 VGPR operand
                                                \
                                                 -> VOP3PX2 WMMAScale
B2 logical MXFP4 GM [E,N,K]                     -> 8 FP32 accumulator fragments/lane
  -> packed+preshuffled n16 physical rows       /
  -> TDM [32,1024 B], GM stride 6144
  -> LDS B slot [32,1024 B]
  -> 2 x ds_load_b128/wn/lane
  -> 8-dword packed-FP4 VGPR operand

SA logical E8M0 [M,K/32] -> packed i32 -> TDM -> LDS -> ds_load_b32 -> WMMAScale scale_b
SB logical E8M0 [E,N,K/32] -> n32k4 -> TDM -> LDS -> ds_load_b32 -> WMMAScale scale_a

bias2 BF16 GM [E,N]
  -> 8 x global_load_b128/wave
  -> widen FP32
  -> v_pk_add_f32 into accumulators

FP32 accumulators
  -> v_cvt_pk_bf16_f32
  -> packed BF16 VGPR
  -> 8 x ds_store_b128/wave
  -> LDS C [16,512]
  -> tensor_store_from_lds
  -> grouped_out BF16 GM [M,N]
```

每条箭头的 layout 口径：

| Transition | Logical unit | Physical unit | Key stride/padding | OOB behavior |
|---|---|---|---|---|
| A2 GM -> LDS | FP8 element | byte | GM 768 B/row; LDS 144 B/row | M rows bounded by `mn_oob` |
| A2 LDS -> VGPR | K128 row | four B128 reads | 32-B interleave | LDS dependency via DScnt |
| B2 GM -> LDS | FP4 element | packed byte / n16 row | GM 6144 B/n16; LDS 1024 B/n16 | no N-tail bound |
| B2 LDS -> VGPR | n16/K128 fragment | two B128 reads | second read `+512` | LDS dependency via DScnt |
| SA GM -> LDS | E8M0 byte | i32 of 4 scales | 6 dwords/A row | arena zero protects over-read |
| SA LDS -> VGPR | four scales | one i32 | row/kgrp formula | WMMAScale selects scale byte |
| SB GM -> LDS | E8M0 byte | n32k4 i32 | 192 dwords/N32 full K | no N-tail bound |
| SB LDS -> VGPR | four scales | one i32 per wn/lane | 32 dwords/K128/N32 | WMMAScale selects scale byte |
| VGPR -> accumulator | FP4/FP8 + E8M0 | VOP3PX2 registers | K128 | FP32 accumulation |
| bias2 GM -> accumulator | BF16 | B128 vector then FP32 | expert-major stride N | no N-tail bound |
| accumulator -> LDS | FP32 then BF16 | packed dword/B128 | row stride 1024 B | valid rows later masked |
| LDS -> GM | BF16 tile | TDM D# | GM row stride N | M tail dropped |

<a id="sec-2-11-residency-summary"></a>
### 2.11 Tensor residency 总结

```text
A2:
  GM row-major FP8 -> TDM -> padded LDS A0/A1 -> VGPR[16 dwords/lane] -> WMMA second operand.

B2:
  GM packed+preshuffled FP4 n16 rows -> TDM -> LDS B0/B1
  -> VGPR[8 dwords/wn/lane] -> WMMA first operand.

SA:
  GM E8M0 row scale bytes packed as int32 -> TDM -> LDS SA0/SA1
  -> one scale dword -> WMMA scale for A2 activation.

SB:
  GM n32k4 E8M0 packed scale -> TDM -> LDS SB0/SB1
  -> eight scale dwords -> WMMA scales for B2 fragments.

Accumulator:
  8 fragments * 8 FP32/lane, VGPR only across all six K128 stages.

bias2:
  GM BF16 -> VGPR -> FP32 add; never enters LDS independently.

Output:
  accumulator+bias FP32 -> packed BF16 VGPR -> LDS [16,512]
  -> TDM store -> grouped_out GM BF16.
```

<a id="sec-2-12-low-level-call-chains-and-issue-counts"></a>
### 2.12 Low-level call chains and issue counts

| Source statement | Role | Lowered primitive | Final ISA | Static count |
|---|---|---|---|---:|
| `issue(...): fx.copy(A atom,...)` | A2 GM -> LDS | `rocdl.tensor.load.to.lds` | `tensor_load_to_lds` | 6 |
| `issue(...): fx.copy(B atom,...)` | B2 GM -> LDS | `rocdl.tensor.load.to.lds` | `tensor_load_to_lds` | 6 |
| `issue(...): fx.copy(SA atom,...)` | SA GM -> LDS | `rocdl.tensor.load.to.lds` | `tensor_load_to_lds` | 6 |
| `issue(...): fx.copy(SB atom,...)` | SB GM -> LDS | `rocdl.tensor.load.to.lds` | `tensor_load_to_lds` | 6 |
| `load_a` | LDS A2 -> VGPR | raw LLVM addrspace(3) load | `ds_load_b128` | 24 |
| `load_b` | LDS B2 -> VGPR | raw LLVM addrspace(3) load | `ds_load_b128` | 96 |
| `load_sa/load_sb` | LDS scales -> VGPR | i32 addrspace(3) loads | `ds_load_b32`, `ds_load_2addr_b32` | 28 + 13 |
| `fx.gemm(WMMAScale,...)` | K128 matrix MAC | ROCDL scaled WMMA intrinsic | `v_wmma_scale_f32_16x16x128_f8f6f4` | 48 |
| `fx.ptr_load(bias_map...)` | bias2 GM -> VGPR | LLVM vector load | `global_load_b128` | 8 |
| `acc + bias` | bias epilogue | packed FP32 arithmetic | `v_pk_add_f32` | 32 |
| `Float32.to(BFloat16)` | output conversion | vector `fptrunc` | `v_cvt_pk_bf16_f32` | 32 |
| `lds_store_b128_raw` | BF16 VGPR -> LDS | LLVM addrspace(3) store | `ds_store_b128` | 8 |
| arena clear | LDS zero init | LLVM addrspace(3) store | `ds_store_b128` | 37 |
| `fx.copy(atomC,LDS,GM)` | output LDS -> GM | `rocdl.tensor.store.from.lds` | `tensor_store_from_lds` | 1 |

静态、per-wave 与 per-WG issue：

<div style="overflow-x: auto;">
<table style="white-space: nowrap; width: max-content; min-width: 1120px;">
  <thead>
    <tr>
      <th>Opcode / result</th>
      <th align="right">Static ISA</th>
      <th align="right">Per valid wave</th>
      <th align="right">Per valid WG wave-issues</th>
      <th>Notes</th>
    </tr>
  </thead>
  <tbody>
    <tr><td><code>tensor_load_to_lds</code></td><td align="right">24</td><td align="right">24</td><td align="right">96</td><td>4 tensors × 6 K tiles</td></tr>
    <tr><td><code>tensor_store_from_lds</code></td><td align="right">1</td><td align="right">1</td><td align="right">4</td><td>one 4-row segment / wave</td></tr>
    <tr><td>A2 <code>ds_load_b128</code></td><td align="right">24</td><td align="right">24</td><td align="right">96</td><td>4 per K tile</td></tr>
    <tr><td>B2 <code>ds_load_b128</code></td><td align="right">96</td><td align="right">96</td><td align="right">384</td><td>16 per K tile</td></tr>
    <tr><td>scale dword results</td><td align="right">54</td><td align="right">54</td><td align="right">216</td><td>28 single + 13 dual-address instructions</td></tr>
    <tr><td><code>v_wmma_scale...</code></td><td align="right">48</td><td align="right">48</td><td align="right">192</td><td>8 fragments × 6 K tiles</td></tr>
    <tr><td>bias <code>global_load_b128</code></td><td align="right">8</td><td align="right">8</td><td align="right">32</td><td>128 columns / wave</td></tr>
    <tr><td><code>v_pk_add_f32</code></td><td align="right">32</td><td align="right">32</td><td align="right">128</td><td>bias-only FP32 add</td></tr>
    <tr><td><code>v_cvt_pk_bf16_f32</code></td><td align="right">32</td><td align="right">32</td><td align="right">128</td><td>64 BF16 values/lane</td></tr>
    <tr><td>C <code>ds_store_b128</code></td><td align="right">8</td><td align="right">8</td><td align="right">32</td><td>one 128-column stripe / wave</td></tr>
    <tr><td>zero <code>ds_store_b128</code></td><td align="right">37</td><td align="right">37</td><td align="right">148</td><td>all 75,776 B</td></tr>
  </tbody>
</table>
</div>

“Per valid WG wave-issues”不能直接当作 DRAM transaction 或 LDS bank request 数；TDM 会在硬件
内部拆分操作，global/LDS coalescing 也可能合并 lane requests。

完整 lowering call chain：

```text
grouped_moe_gfx1250._grouped_a8w4_tdm_moe
  -> batched_gemm_mxfp4.flydsl_grouped_gemm_a8w4_masked
  -> mxfp4_preshuffle_gfx1250_tdm.launch_gemm_a8w4_tdm

fx.rocdl.make_tdm_atom + fx.copy
  -> fly.copy / fly.copy_atom_call
  -> rocdl.tensor.load.to.lds / tensor.store.from.lds
  -> llvm.amdgcn.tensor.{load.to.lds,store.from.lds}
  -> tensor_load_to_lds / tensor_store_from_lds

lds_load_b128_raw / lds_load_b32_raw
  -> LLVM addrspace(3) load
  -> ds_load_b128 / ds_load_b32 / ds_load_2addr_b32

fx.rocdl.WMMAScale + fx.gemm
  -> rocdl.wmma.scale.f32.16x16x128.f8f6f4
  -> llvm.amdgcn.wmma.scale.f32.16x16x128.f8f6f4
  -> 4-DWORD VOP3PX2 v_wmma_scale_f32_16x16x128_f8f6f4

pipeline_fence
  -> tensor_wait + gpu.barrier
  -> s_wait_tensorcnt + s_barrier_signal/s_barrier_wait
```

---

## 3. Final summary and verification boundary

本 GEMM2 的核心不是普通“global load + dot + global store”，而是完整的：

```text
contiguous-M route swizzle
  -> in-kernel expert lower_bound
  -> 2-slot TDM GM->LDS ring
  -> DS LDS->VGPR re-gather
  -> 6 x K128 VOP3PX2 WMMAScale accumulation
  -> bias2-only FP32 epilogue
  -> BF16 VGPR->LDS staging
  -> TDM LDS->GM store
```

关键定量结论：

- 固定 workload：`M=30720, N=7168, K=768, E=384`；
- grid：26880 workgroups，每组 4 wave32；
- 逻辑 LDS footprint：74,752 B；
- 实际 dynamic LDS allocation：75,776 B；
- metadata fixed LDS：0 B，仅因为使用 dynamic LDS；
- 6 个完全展开的 K128 stages；
- static TDM loads：24；
- static VOP3PX2 WMMAScale：48；
- 8 个 FP32 accumulator fragments / wave；
- bias-only noact：8 个 BF16 global vector loads、32 个 packed FP32 adds；
- output：32 个 packed BF16 converts、8 个 LDS B128 stores、1 个静态 TDM store；
- VGPR/SGPR：172/84，零 spill；
- kernel code：1109 instructions、7104 B。

### Verified facts

- source formulas、logical/preshuffled shapes、LDS offsets和 launch geometry；
- `00_origin.mlir` 中 75,776-B dynamic LDS；
- `19_gpu_module_to_binary.mlir` 的 ELF symbol VA/size；
- 所有 kernel-relative PC ranges；
- opcode static counts、resource metadata 和 zero spills；
- A2/B2/SA/SB/bias2/output 的实际 GM/LDS/VGPR transition；
- generated `bias1` 是 `has_bias=1`，实际 tensor 是 `bias2`；
- 当前 workload 的 `N=7168` 对 512 整除。

### Derived but deterministic

- `30720/16 * 7168/512 = 26880` workgroups；
- 4-wave workgroup 的 192 个动态 WMMA wave-issues；
- full output tile 的 `12,582,912` FLOPs；
- 37 rounds × 2048 B = 75,776 B zero coverage。

### Inference / non-claims

- 物理 VGPR 角色名只在相应 live range 内成立，register allocator 会复用寄存器；
- A2/B2 packed bytes 到 WMMAScale 内部 matrix element 的最后一级 lane permutation 由硬件
  instruction layout 决定；本文只把 source/ISA 可验证的 LDS 地址、VGPR 宽度和 output
  mapping 标为事实；
- MI400 guide 只作 ISA-family 语义交叉参考，不声称 gfx1250 与 MI450 有相同吞吐、容量、
  clause errata 或 XNACK 行为；
- static ISA count 不等于 cache transaction 数；
- 本报告没有从该代码对象推断实际 occupancy 或运行时 memory latency。

最后，虽然通用 source 用 `ceil(N/512)` 建 grid，本 specialization 对 B2、SB、bias2 和
output N dimension 没有显式 tail mask。任何复用该配置的 workload 都必须保持
`N % 512 == 0` 或提供合法物理 padding；当前 `N=7168` 满足这一前提。
