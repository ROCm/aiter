# gfx1250 GEMM1 A8W4 TDM + Gated-SiLU 深度分析

<!-- markdown-toc-generator:start -->
## Table of Contents

- [Authoritative Sources](#authoritative-sources)
- [Benchmark / Trait Configuration](#benchmark-trait-configuration)
  - [Fixed workload and routing geometry](#sec-fixed-workload-and-routing-geometry)
  - [Compiled resource usage](#sec-compiled-resource-usage)
  - [Address convention](#sec-address-convention)
- [1. Kernel overview and software pipeline](#1-kernel-overview-and-software-pipeline)
  - [1.1 Source call path and lowering chain](#sec-1-1-source-call-path-and-lowering-chain)
  - [1.2 Grid swizzle and expert lookup](#sec-1-2-grid-swizzle-and-expert-lookup)
  - [1.3 End-to-end software pipeline](#sec-1-3-end-to-end-software-pipeline)
  - [1.4 Full kernel-relative PC phase map](#sec-1-4-full-kernel-relative-pc-phase-map)
  - [1.5 Dynamic LDS arena and two-buffer ring](#sec-1-5-dynamic-lds-arena-and-two-buffer-ring)
  - [1.6 TDM descriptors and exact tensor instructions](#sec-1-6-tdm-descriptors-and-exact-tensor-instructions)
  - [1.7 Prologue, 27 steady iterations, and drain](#sec-1-7-prologue-27-steady-iterations-and-drain)
  - [1.8 Waits, barriers, and scheduling boundaries](#sec-1-8-waits-barriers-and-scheduling-boundaries)
  - [1.9 VOP3PX2 WMMAScale and hardware operand swap](#sec-1-9-vop3px2-wmmascale-and-hardware-operand-swap)
  - [1.10 Bias, clamp, gated-SiLU, and BF16 epilogue](#sec-1-10-bias-clamp-gated-silu-and-bf16-epilogue)
- [2. Full GM/LDS/VGPR layout maps](#2-full-gmldsvgpr-layout-maps)
  - [2.1 Notation and complete tensor residency map](#sec-2-1-notation-and-complete-tensor-residency-map)
  - [2.2 A activation: GM -> LDS -> VGPR](#sec-2-2-a-activation-gm---lds---vgpr)
    - [2.2.1 A GM logical/physical layout](#sec-2-2-1-a-gm-logicalphysical-layout)
    - [2.2.2 A GM -> LDS TDM layout](#sec-2-2-2-a-gm---lds-tdm-layout)
    - [2.2.3 A LDS -> VGPR WMMA-B layout](#sec-2-2-3-a-lds---vgpr-wmma-b-layout)
  - [2.3 SA activation scale: GM -> LDS -> VGPR](#sec-2-3-sa-activation-scale-gm---lds---vgpr)
  - [2.4 B weight: GM -> LDS -> VGPR](#sec-2-4-b-weight-gm---lds---vgpr)
    - [2.4.1 B logical versus preshuffled physical layout](#sec-2-4-1-b-logical-versus-preshuffled-physical-layout)
    - [2.4.2 B GM -> LDS TDM layout](#sec-2-4-2-b-gm---lds-tdm-layout)
    - [2.4.3 B LDS -> VGPR WMMA-A layout](#sec-2-4-3-b-lds---vgpr-wmma-a-layout)
  - [2.5 SB weight scale: GM -> LDS -> VGPR](#sec-2-5-sb-weight-scale-gm---lds---vgpr)
  - [2.6 VGPR operands -> FP32 accumulators](#sec-2-6-vgpr-operands---fp32-accumulators)
  - [2.7 Bias GM -> VGPR and accumulator -> gated-SiLU VGPR](#sec-2-7-bias-gm---vgpr-and-accumulator---gated-silu-vgpr)
  - [2.8 BF16 VGPR -> LDS -> GM output](#sec-2-8-bf16-vgpr---lds---gm-output)
  - [2.9 Summary: every tensor transition](#sec-2-9-summary-every-tensor-transition)
  - [2.10 Low-level call chains and issue counts](#sec-2-10-low-level-call-chains-and-issue-counts)
- [3. Final summary](#3-final-summary)
  - [3.1 Verified facts versus inference](#sec-3-1-verified-facts-versus-inference)
  - [3.2 Key conclusions](#sec-3-2-key-conclusions)

<!-- markdown-toc-generator:end -->

本文只分析下面这一份 **GEMM1** 编译实例：

```text
gemm_a8w4_tdm_t16x256x256_w1x4_b2_e384_afp8_outbf16_silu_bias1_qout0_qrep1_v1
```

它是 gfx1250 grouped MoE stage-1 的 A8W4 gate/up projection：输入 activation 为
MXFP8 E4M3，weight 为 MXFP4 E2M1，A/B 两侧都使用 block-32 E8M0 scale；计算完成后在
FP32 accumulator 上加 BF16 bias，按 GUGU 相邻列做 clamp 和
`SiLU(gate) * up`，转成 BF16，经 LDS staging 后由 TDM 写回 GM。

本文严格区分四类概念：

1. **逻辑 layout**：数学矩阵的 `(M,N,K)` 坐标。
2. **物理/preshuffled layout**：kernel 实际收到的 packed byte / `n32k4` tensor。
3. **静态指令数**：最终 ISA 中出现的 opcode site 数。
4. **动态指令数**：一个 wave 在本 workload 的真实控制流中执行的次数。

所有 PC 结论均来自 `19_gpu_module_to_binary.mlir` 内嵌 ELF，而不是把汇编行号误当作地址。

## Authoritative Sources

下列文件是本文的事实来源，按优先级排列。

| Source | Role |
|---|---|
| `aiter/my_code/flydsl_dump/gemm_a8w4_tdm_t16x256x256_w1x4_b2_e384_afp8_outbf16_silu_bias1_qout0_qrep1_v1/21_final_isa.s` | 最终 gfx1250 ISA、资源 metadata、静态 opcode 计数 |
| 同目录 `19_gpu_module_to_binary.mlir` | 内嵌 code-object ELF、真实 `.text` VMA、函数 byte size、launcher dynamic LDS |
| 同目录 `20_llvm_ir.ll` | TDM/WMMA intrinsic、descriptor 值、loop trip count、bias/SiLU IR |
| 同目录 `00_origin.mlir`、`03_fly_layout_lowering.mlir`、`08_convert_fly_to_rocdl.mlir` | FlyDSL high-level copy/GEMM 到 atom、ROCDL intrinsic 的 lowering 对应 |
| `aiter/aiter/ops/flydsl/kernels/mxfp4_preshuffle_gfx1250_tdm.py` | tile、LDS、TDM job、DS read、WMMAScale、epilogue 的源公式 |
| `aiter/aiter/ops/flydsl/kernels/gemm_common_gfx1250.py` | `fused_silu_swiglu_elem`、LDS helper、barrier/fence 定义 |
| `aiter/aiter/ops/flydsl/batched_gemm_mxfp4.py` | grouped GEMM wrapper、参数 ABI、dummy quant-scale |
| `aiter/aiter/ops/flydsl/grouped_moe_gfx1250.py` | routing、contiguous-M、GUGU/non-EP TDM path、GEMM1/GEMM2 串联 |
| `aiter/aiter/fused_moe.py` | public `fused_moe_` 到 grouped gfx1250 path 的 dispatch |
| `aiter/op_tests/test_flydsl_grouped_gemm_gfx1250.py` | workload 构造、balanced routing、weight/scale shuffle、reference |
| `aiter/my_code/run_gemm.log` | 本次编译日志、逻辑 arena 打印、correctness/perf 上下文 |
| `C:/Users/yanguahe/Documents/code/llm-wiki/mi400_hw_wiki/raw/papers/mi400_hd_txt/architecture/subsystem/SH/MI400_Shader_Programming#65.txt` | GFX12 TDM、WMMA block-scale、VOP3PX2、packed convert 的硬件语义参考 |

**MI400 guide 适用性声明。** 该 guide 明确描述 MI450-B0/MI400 家族，而本 kernel target
是 `gfx1250`。因此本文只在实际 gfx1250 ISA/LLVM 已经出现相同 opcode、encoding 或 counter
模型时引用 guide 的共性说明。MI450 专属 throughput、tapeout 差异、bug workaround、LDS
allocation granularity 和 NOP 数量都不能直接外推到 gfx1250。每处引用均注明 section/page，
并把“实际 dump 已验证”与“guide 解释性参考”分开。

## Benchmark / Trait Configuration

<a id="sec-fixed-workload-and-routing-geometry"></a>
### Fixed workload and routing geometry

本文固定 workload 来自 `run_gemm.log` 与 test runner：

```text
Arch                  : gfx1250, wave32
Tokens T              : 4096
Experts E             : 384
topk                  : 6
Model/hidden dim H    : 7168
Intermediate dim I    : 768
GEMM1 pre-act N       : 2 * I = 1536
GEMM1 K               : H = 7168
Data format           : A8W4 = MXFP8 E4M3 activation x MXFP4 E2M1 weight
Scale                 : E8M0, one byte per 32 K values
Stage-1 row layout    : GUGU = [gate0, up0, gate1, up1, ...]
Activation            : gated SiLU, limit = 7.0
Bias                  : BF16 bias1 enabled
Output                : BF16, shape (1, 30720, 768)
```

`AITER_MOE_EXPERT_BALANCE=true` 使 `4096*6/384=64`，每个 expert 精确接收 64 条 route：

| Quantity | Formula | Value |
|---|---|---:|
| Valid routes | `T * topk` | 24,576 |
| Tile-M alignment | `tile_m` | 16 |
| Rows per expert | `24576 / 384` | 64 |
| Valid M tiles per expert | `64 / 16` | 4 |
| `contiguous_M` | `align_up(T*topk + E*16 - topk, 16)` | 30,720 |
| Capacity tail | `30720 - 24576` | 6,144 rows |
| Total M tiles | `30720 / 16` | 1,920 |
| Valid M tiles | `24576 / 16` | 1,536 |
| Sentinel/padding M tiles | `6144 / 16` | 384 |
| N tiles | `1536 / 256` | 6 |
| Grid | `1920 * 6` | 11,520 workgroups |
| Valid-compute workgroups | `1536 * 6` | 9,216 |
| Sentinel-skipped workgroups | `384 * 6` | 2,304 |

Balanced workload 下：

```text
m_tile_map[e] = (e + 1) * 64, e = 0..383
```

`run_gemm.log` 的最后一组 profiler sample 报告 GEMM1 device average `2127.1 us`
（7 次），同一轮 fused-MoE end-to-end 为 `5254.94 us`，correctness 为
`logits_diff=4.1258e-06, rel_l2=2.8725e-03, pass=True`。这些数字用于标识 dump
上下文，不应当视为隔离 GEMM1 microbenchmark：日志中还包含 profiler overhead、routing、
quant、GEMM2 和 gather。

<a id="sec-compiled-resource-usage"></a>
### Compiled resource usage

| Property | Verified value | Evidence / interpretation |
|---|---:|---|
| Workgroup | `128 x 1 x 1` | 4 wave32 |
| VGPR | 103 | `.amdhsa_next_free_vgpr`, `.vgpr_count` |
| SGPR | 88 numbered / 90 metadata | 额外 2 个为 reserved VCC |
| AGPR | 0 | accumulator 也在 VGPR |
| VGPR spills | 0 | metadata |
| SGPR spills | 0 | metadata |
| Private segment | 0 B | 无 scratch |
| Fixed LDS metadata | 0 B | LDS 是 dynamic，不代表“不用 LDS” |
| Logical ring arena | 78,848 B | `2 * PITCH = 2 * 39,424` |
| Actual dynamic LDS launch | 79,872 B | tile-M≤64 zero-fill rounding 后 `0x13800` |
| Kernarg size | 176 B | metadata |
| Function code size | `0x1a08 = 6664 B` | embedded ELF symbol span |
| Static physical instructions | 1,104 | 最终 ISA instruction lines，VOPD 每行算一条 encoding |
| Static WMMAScale | 16 | loop body 8 + drain 8 |
| Dynamic WMMAScale / wave | 224 | `27*8 + 8` |

逻辑 arena 与实际 allocation 的差异必须明确：

```text
STAGE_A  =  4,352 B
STAGE_B  = 32,768 B
STAGE_SA =    128 B
STAGE_SB =  2,048 B
sum      = 39,296 B
PITCH    = align_up(39,296, 512) = 39,424 B
2*PITCH  = 78,848 B = 77 KiB                 # source/log print 的 ARENA

_zblk    = 16 * block = 16 * 128 = 2,048 B
_arena   = align_up(78,848, 2,048) = 79,872 B = 78 KiB  # launch 的 dynamic LDS
```

`21_final_isa.s` 的 `.amdhsa_group_segment_fixed_size 0` 只描述 fixed LDS；真正值在
`19_gpu_module_to_binary.mlir` launcher 的
`dynamic_shared_memory_size = 79872`。

<a id="sec-address-convention"></a>
### Address convention

`21_final_isa.s` 没有 PC annotation。本文从 `19_gpu_module_to_binary.mlir` 的 ELF 恢复：

```text
.text file offset : 0x900
.text ELF VMA     : 0x1900
.text section size: 0x1c00
kernel size       : 0x1a08
kernel VMA span   : 0x1900 .. 0x3307
```

本文写法：

- `+0x0b40`：kernel-relative PC，最稳定。
- `VMA 0x2440`：ELF `.text` 中的地址，等于 `0x1900 + 0x0b40`。
- runtime code address 仍可能由 loader relocation，不应把 ELF VMA 当成运行时绝对 VA。

---

## 1. Kernel overview and software pipeline

每个 workgroup 负责一个 `(16 M rows, 256 pre-activation N columns)` tile。四个 wave
沿 N 维分片，每个 wave 负责 `16 x 64`，内部有 4 个 `16 x 16` accumulator fragment。
K=7168 被分成 28 个 `tile_k=256`，每个 K tile 又由两条 K=128 WMMAScale step 完成。

高层数据流：

```text
A MXFP8 + SA E8M0 ─┐
                    ├─ TDM GM->LDS -> DS LDS->VGPR ─┐
B MXFP4 + SB E8M0 ─┘                                ├─ WMMAScale -> FP32 acc
Bias BF16 GM ---------------------------------------┘
FP32 acc -> bias add -> clamp -> gated-SiLU -> BF16 VGPR
         -> LDS output staging -> TDM LDS->GM
```

<a id="sec-1-1-source-call-path-and-lowering-chain"></a>
### 1.1 Source call path and lowering chain

Production call path:

```text
fused_moe_
  -> _maybe_grouped_gfx1250_a8w4_moe
     -> [gfx1250, per_1x32, G1U1, SiLU, A8W4, GUGU, non-EP TDM branch]
     -> _grouped_a8w4_tdm_moe
        -> flydsl_moe_fused_quant_preshuffle        # hidden -> A/SA
        -> flydsl_grouped_gemm_a8w4_masked          # GEMM1
           -> launch_gemm_a8w4_tdm
              -> generated kernel
        -> flydsl_moe_fused_quant_preshuffle        # BF16 y -> GEMM2 A/SA
```

因为 `bias1 != None`：

```text
_fuse_quant = (not _is_fp4) and (_b1 is None) = false
```

所以 GEMM1 kernel 是 `qout0`，先输出 BF16 `y`，再由独立 quant kernel 生成 GEMM2
输入，而不是在本 epilogue 内量化。

Compiler lowering 对应关系：

| Dump stage | Relevant representation | Exact site count |
|---|---|---:|
| `00_origin` .. `02_fly_canonicalize` | `fly.copy` + `fly.gemm` | 9 copy + 16 gemm |
| `03_fly_layout_lowering` .. `07_fly_promote_regmem_to_vectorssa` | TDM/MMA atom calls、register SSA | 25 atom calls |
| `08_convert_fly_to_rocdl` .. `18_reconcile_unrealized_casts` | `rocdl.tensor.*` + `rocdl.wmma.scale.*` | 9 tensor + 16 WMMA |
| `20_llvm_ir.ll` | `llvm.amdgcn.tensor.*` + `llvm.amdgcn.wmma.scale.*` | 8 load + 1 store + 16 WMMA |
| `21_final_isa.s` | machine ISA | 8 tensor load + 1 tensor store + 16 WMMAScale |

<a id="sec-1-2-grid-swizzle-and-expert-lookup"></a>
### 1.2 Grid swizzle and expert lookup

源公式：

```text
total_n_tiles   = ceil(N / 256) = 6
total_m_tiles   = ceil(M / 16)  = 1920
blocks_per_group = total_n_tiles * TILES_PER_GROUP
                 = 6 * 16 = 96

group            = bid_x // 96
group_first_tile = group * 16
in_group         = bid_x - group * 96
group_tiles      = min(total_m_tiles - group_first_tile, 16)
m_tile           = group_first_tile + in_group % group_tiles
blk_m            = m_tile * 16
blk_n            = (in_group // group_tiles) * 256
```

本 workload 的 `total_m_tiles=1920` 可被 16 整除，所以所有 group 的
`group_tiles=16`；每连续 96 个 block 按 N-major group 内转置成 16 个 M tile × 6 个
N tile。

<div style="overflow-x: auto;">
<table style="white-space: nowrap; width: max-content; min-width: 1180px;">
  <thead>
    <tr>
      <th align="right"><code>bid_x</code> in one group</th>
      <th align="right"><code>in_group % 16</code></th>
      <th align="right"><code>in_group // 16</code></th>
      <th align="right"><code>blk_m</code></th>
      <th align="right"><code>blk_n</code></th>
      <th>Interpretation</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td align="right"><code>96*g + 0..15</code></td>
      <td align="right"><code>0..15</code></td>
      <td align="right">0</td>
      <td align="right"><code>(16*g + 0..15)*16</code></td>
      <td align="right">0</td>
      <td>同一 16-tile M group 的第 0 个 N tile</td>
    </tr>
    <tr>
      <td align="right"><code>96*g + 16..31</code></td>
      <td align="right"><code>0..15</code></td>
      <td align="right">1</td>
      <td align="right"><code>(16*g + 0..15)*16</code></td>
      <td align="right">256</td>
      <td>同一批 M tile，N 前进 256</td>
    </tr>
    <tr>
      <td align="right">...</td>
      <td align="right">...</td>
      <td align="right">...</td>
      <td align="right">...</td>
      <td align="right">...</td>
      <td>...</td>
    </tr>
    <tr>
      <td align="right"><code>96*g + 80..95</code></td>
      <td align="right"><code>0..15</code></td>
      <td align="right">5</td>
      <td align="right"><code>(16*g + 0..15)*16</code></td>
      <td align="right">1280</td>
      <td>第 6 个、也是最后一个 N tile</td>
    </tr>
  </tbody>
</table>
</div>

Expert lookup 是固定 384-entry `m_tile_map` 上的 unrolled lower-bound：

```text
lo = 0, hi = 384
repeat ceil(log2(384)) + 1 = 10 times:
    mid = (lo + hi) >> 1
    go_right = tile_map[min(mid, 383)] <= blk_m
    lo = go_right ? mid + 1 : lo
    hi = go_right ? hi      : mid
expert = lo
```

它等价于找到第一个 `tile_map[expert] > blk_m` 的 expert。10 次查找的
`global_load_b32` kernel-relative PCs 为：

```text
+0x003c, +0x02c8, +0x0300, +0x033c, +0x0398,
+0x0408, +0x0440, +0x0478, +0x04b8, +0x04f8
```

随后 `+0x053c` 的第 11 次 load 取得：

```text
mn_oob = tile_map[min(expert, 383)] - blk_m
```

balanced workload 中每个 expert 64 行、边界 16 对齐，所以四个有效 tile 的
`mn_oob` 依次为 64、48、32、16，均足以覆盖当前 16 行；capacity tail 查到
`expert=384`，在 `+0x0698` 跳到 `+0x1a04 s_endpgm`。该 branch 在 arena zero-init
之后，因此 sentinel workgroup 仍执行统一的 LDS 初始化，但不执行 TDM/GEMM/epilogue。

<a id="sec-1-3-end-to-end-software-pipeline"></a>
### 1.3 End-to-end software pipeline

下面按 `mxfp4_preshuffle_gfx1250_tdm.py` 中的实际变量名和控制流展开。当前
specialization 的关键编译期常量为：

```text
K_TILES       = K / tile_k = 7168 / 256 = 28
KWS           = tile_k / WMMA_K = 256 / 128 = 2
num_buffers   = 2
PITCH         = 0x9a00 = 39,424 B
TDM_PER       = 4                    # WAVE_SPEC=false
wmma_m_rep    = 1
wmma_n_rep    = 4
n_acc         = 4                    # 4 个 16x16 FP32 fragments / wave
FRONT         = [0]
BACK          = []                   # m-rep 只有 1，不存在 back-M fragment
```

源码中的四个 TDM `Job` 是：

```text
A  : GM FP8  [16,256]  -> LDS [16,272 B]，每 row 尾部 pad 16 B
B  : GM FP4  [16,2048 B physical rows] -> LDS [16,2048 B]
SA : GM i32  [16,2]    -> LDS [16,2] i32
SB : GM i32  [8,64]    -> LDS [8,64] i32
```

`issue(slot, kt)` 会为同一个 K tile 发出这四种 `fx.copy(TDM atom)`。因为
`tile_m=16 < 64`，`WAVE_SPEC=false`，四个 wave 都经过相同的四个
`tensor_load_to_lds` site；各 wave 的 D# 负责自己的 tensor segment。

```text
Setup / pre-dispatch:
  [S0] 由 block_id 计算 (blk_m, blk_n)
       expert = lower_bound(m_tile_map, blk_m)
       mn_oob = m_tile_map[min(expert, E-1)] - blk_m

  [S1] 分配 dynamic LDS：
       logical ring = 2 * PITCH = 78,848 B
       actual arena = align_up(78,848, 2,048) = 79,872 B

  [S2] 因 tile_m <= 64，全部 128 threads 协作清零 LDS：
       for zi in 0..38:                              # 39 rounds
         ds_store_b128(
           addr = (tid + zi*128) * 16,
           data = 16-byte zero
         )
       workgroup_barrier()
       # 清零也覆盖 SA 可能读取的 padding dword，避免 stale 0xff E8M0 形成 NaN

  [S3] if expert == E:                               # padding sentinel 384
         return                                      # 不执行 TDM/GEMM/epilogue

  [S4] 创建四个 TDM jobs：
       job_A:
         GM base = arg_a + blk_m*7168
         tile    = [16,256] bytes
         GM row stride  = 7168 B
         LDS row stride = 272 B
         outer extent   = mn_oob

       job_B:
         b_outer_row = expert*(N/16) + blk_n/16
         GM base     = arg_b + b_outer_row*57344
         tile        = [16,2048] bytes
         GM row stride = 57344 B

       job_SA:
         GM base = arg_scale_a + blk_m*56 i32
         tile    = [16,2] i32
         GM row stride = 56 i32

       job_SB:
         sb_off0 = expert*(N/32)*(K/4) + (blk_n/32)*(K/4)
         GM base = arg_scale_b + sb_off0 i32
         tile    = [8,64] i32
         GM row stride = K/4 = 1792 i32

  [S5] 创建 4 个 FP32 accumulator fragments：
       for wn in 0..3:
         c_frags[wn] = zeros(8 x f32 per lane)


TDM prologue:
  [P0] for i in range(num_buffers - 1):              # 只执行 i=0
         issue(slot=0, kt=0)

       此时只预取第一个 K256 tile：
         buffer0 = { A0, B0, SA0, SB0 }
         buffer1 = empty


Steady loop:
  n_steady = K_TILES - (num_buffers - 1)
           = 28 - 1
           = 27

  for kt in 0..26:
    [C0 ready/reuse fence]
      slot = kt % 2
      buf  = base_ptr + slot*0x9a00

      pipeline_fence(outstanding=0):
        s_wait_tensorcnt 0
        workgroup_barrier()

      # fence 同时保证：
      # 1. 当前 slot 的 A/B/SA/SB 已由 TDM 写完；
      # 2. 所有 wave 都可以安全读取当前 slot；
      # 3. 该 slot 上一轮的使用已经结束，可以在 alternate slot 上继续流水。

    [C1 issue current B/SB/SA LDS reads for KSL0]
      wt[0..3], sb_k[0..3], sa_k[0] =
        load_b_and_scales(buf, ksl=0)

      wt[wn]:
        2 x ds_load_b128 -> 8 packed-FP4 dwords/lane
      sb_k[wn]:
        1 x ds_load_b32  -> 4 E8M0 bytes for weight K0..127
      sa_k[0]:
        1 x ds_load_b32  -> 4 E8M0 bytes for activation K0..127

    [C2 mid-compute prefetch of next K256 tile]
      prefetch_kt = kt + 1
      sched_barrier(0)
      issue(
        slot = prefetch_kt % 2,
        kt   = prefetch_kt
      )
      sched_barrier(0)

      # 当前 tile 的 B/scale DS reads 已经发出；
      # 下一 tile 的 A/B/SA/SB TDM 写入 alternate buffer，
      # 与当前 tile 后续的 A LDS read + WMMAScale 重叠。

    [C3 compute KSL0: local K 0..127 of current K256 tile]
      act = load_a(buf, wm=0, ksl=0)
            # 4 x ds_load_b128 -> 16 FP8 dwords/lane

      s_wait_dscnt 0

      for wn in 0..3:
        c_frags[wn] =
          WMMAScale_16x16x128(
            dst/acc = c_frags[wn],
            src0    = wt[wn],       # hardware Matrix A = FP4 weight
            src1    = act,          # hardware Matrix B = FP8 activation
            scale_a= sb_k[wn],      # weight E8M0
            scale_b= sa_k[0]        # activation E8M0
          )

    [C4 issue current B/SB/SA LDS reads for KSL1]
      wt[0..3], sb_k[0..3], sa_k[0] =
        load_b_and_scales(buf, ksl=1)

      # load_b/load_sb/load_sa 地址增加到当前 K256 tile 的后半个 K128。

    [C5 compute KSL1: local K 128..255]
      act = load_a(buf, wm=0, ksl=1)
      s_wait_dscnt 0

      for wn in 0..3:
        c_frags[wn] =
          WMMAScale_16x16x128(
            c_frags[wn], wt[wn], act,
            scale_a=sb_k[wn], scale_b=sa_k[0]
          )

    [C6 scheduling-only tail]
      sched_dsrd / sched_mfma hints describe:
        KSL0: B+scale DS reads -> A DS reads -> 4 WMMA
        KSL1: B+scale DS reads -> A DS reads -> 4 WMMA
      sched_barrier(0)

    yield:
      c_frags[0..3]                                  # loop-carried FP32 accumulators
      alternate buffer contains tile kt+1


Drain:
  # steady loop 已计算 kt0..kt26，并在 kt26 中预取 kt27 到 slot1
  for j in range(num_buffers - 1):                   # 只执行 j=0
    [D0] kt   = n_steady + j = 27
         slot = kt % 2 = 1
         buf  = buffer1

    [D1] pipeline_fence(outstanding=0)

    [D2] compute_ktile(buf, prefetch_kt=None):
         load B/SB/SA KSL0
         load A KSL0
         4 WMMAScale
         load B/SB/SA KSL1
         load A KSL1
         4 WMMAScale

         # 不再 issue 新 TDM tile。

  result:
    c_frags[0..3] 已沿完整 K=7168 累加：
      28 K256 tiles * 2 K128 steps * 4 WN
      = 224 WMMAScale / wave


Post-compute fence and epilogue:
  [E0] accs = load(c_frags[0..3])
       pipeline_fence(outstanding=0)
       # 最终 tensor wait + WG barrier；
       # buffer0 随后可安全复用为 output LDS staging。

  [E1] compile-time branch selection：
       stage1_act       = 1      # SiLU
       stage1_quant_out = 0      # qout0
       has_bias         = 1
       out dtype        = BF16

       STORE_N   = tile_n / 2 = 128
       out_stride= i32_n / 2  = 768
       is_swiglu = false
       limit     = 7.0

  [E2] for wn in 0..3:
       row_rel = lane16
       col_rel = wave_n*64 + wn*16 + kgrp*8
       acc     = c_frags[wn][0..7]

       bias = load_bf16x8(
         arg_bias + expert*i32_n + col_rel
       )                                             # global_load_b128
       acc = fp32(acc) + fp32(bias)

       for p in 0..3:
         gate = min(acc[2*p],     +7.0)
         up   = clamp(acc[2*p+1], -7.0, +7.0)

         sig  = rcp(1 + exp2(-log2(e)*gate))
         hv[p]= gate * sig * up                      # SiLU(gate) * up

       hv_bf16 = cast_bf16(hv[0..3])
       lds_store_b64(
         LDS_base
           + (row_rel*128 + col_rel/2)*2,
         hv_bf16
       )

       # 每个 lane 每个 wn 写 4 个 BF16；
       # 4 wn 共写 16 个 BF16/lane。

  [E3] workgroup_barrier()
       # backend 在 barrier 前插入 s_wait_dscnt 0，
       # 保证四个 wave 的 [16,128] BF16 tile 已完整写入 LDS。

  [E4] output TDM store:
       out_col_off = blk_n / 2
       c_off_rt    = blk_m*768 + out_col_off

       GM view:
         shape  = [16,128]
         stride = [768,1]
         outer row extent = mn_oob

       fx.copy(
         TDM store atom,
         LDS BF16 [16,128] at base_ptr,
         GM y[blk_m:blk_m+16, blk_n/2:blk_n/2+128]
       )
       s_wait_tensorcnt 0
```

该源码流水的关键点不是“先完整搬完所有 K，再开始 GEMM”，而是：

1. prologue 只向 `buffer0` 预取 `kt0`；
2. 每轮先确认当前 buffer ready，再发起当前 B/scale 的 LDS reads；
3. 在读取当前 A 和执行 8 条 WMMAScale 之前，向 alternate buffer 发起下一 tile
   的四路 TDM；
4. 下一轮的 `pipeline_fence(0)` 再负责确认这些 TDM 写入完成；
5. `c_frags` 始终驻留在 VGPR，直到 28 个 K256 tile 全部累加结束。

与源码一一对应的压缩阶段表：

| Pipeline phase | Memory / descriptor work | Compute / epilogue work | Current state |
|---|---|---|---|
| Setup | expert lookup；39 轮 `ds_store_b128` 清零 79,872 B LDS；构造 A/B/SA/SB jobs | 4 个 FP32 accumulator fragments 清零 | no K tile resident |
| TDM prologue | `issue(0,0)`，四路 TDM 将 kt0 写入 buffer0 | none | buffer0=kt0，buffer1 empty |
| Steady `kt=0..26` | fence current；KSL0 中 `issue((kt+1)%2,kt+1)` | 每轮 2 KSL × 4 WN = 8 WMMAScale | current compute + alternate-buffer DMA |
| Drain `kt=27` | fence buffer1；无下一次 issue | 最后 8 WMMAScale | full K accumulated |
| Post fence | `pipeline_fence(0)`，允许 LDS base 被 output staging 复用 | none | final FP32 `c_frags` |
| Bias + gated-SiLU | 4×`global_load_b128` bias / lane | BF16→FP32 add；clamp；exp2；rcp；两次 multiply | 16 BF16 results/lane |
| Output staging | 4×`ds_store_b64` / lane；WG barrier | none | LDS BF16 `[16,128]` |
| Output TDM | `tensor_store_from_lds` + final tensor wait | none | GM `y[16,128]`, row stride 768 |

<a id="sec-1-4-full-kernel-relative-pc-phase-map"></a>
### 1.4 Full kernel-relative PC phase map

| Kernel-relative range | ELF VMA range | Dominant source phase | Exact anchors |
|---|---|---|---|
| `+0x0000..003b` | `0x1900..193b` | scheduler mode、kernarg loads | first instruction at VMA `0x1900` |
| `+0x003c..0547` | `0x193c..1e47` | swizzle、expert bisect、address arithmetic | expert loads listed above |
| `+0x0548..067f` | `0x1e48..1f7f` | full dynamic-arena zero | 39 `ds_store_b128` |
| `+0x0680..069b` | `0x1f80..1f9b` | zero completion、barrier、sentinel branch | DS wait `+0x068c`; branch `+0x0698` |
| `+0x069c..0b3f` | `0x1f9c..243f` | TDM descriptor construction + kt0 issue | A `+0x07cc`; B `+0x09b8`; SA `+0x0a04`; SB `+0x0b34` |
| `+0x0b40..0e0b` | `0x2440..270b` | 27-iteration steady loop | header `+0x0b40`; backedge `+0x0e08` |
| `+0x0e0c..0ffb` | `0x270c..28fb` | final K-tile drain | tail WMMA starts `+0x0eec` |
| `+0x0ffc..1007` | `0x28fc..2907` | final TDM fence + barrier | wait `+0x0ffc` |
| `+0x1008..109f` | `0x2908..299f` | bias/output address setup | output row/column arithmetic |
| `+0x10a0..10cf` | `0x29a0..29cf` | four bias vector loads | `+0x10a0,+0x10ac,+0x10b8,+0x10c4` |
| `+0x10d0..12a7` | `0x29d0..2ba7` | BF16 unpack、bias add、clamp scheduling | first packed bias add in this region |
| `+0x12a8..196f` | `0x2ba8..326f` | gated-SiLU exp2/range repair/rcp/mul | first `-log2(e)` literal at `+0x12ac` |
| `+0x1970..19c7` | `0x3270..32c7` | BF16 pack + output LDS addresses | first `v_cvt_pk_bf16_f32 +0x1970` |
| `+0x19c8..19e7` | `0x32c8..32e7` | VGPR -> LDS output | four `ds_store_b64` |
| `+0x19e8..19f3` | `0x32e8..32f3` | output LDS wait/barrier | `+0x19e8,+0x19ec,+0x19f0` |
| `+0x19f4..1a03` | `0x32f4..3303` | LDS -> GM TDM store | tensor store `+0x19f4`; wait `+0x1a00` |
| `+0x1a04..1a07` | `0x3304..3307` | exit | `s_endpgm` |

这些区间是 compiler scheduling 后的 dominant phase。TDM prefetch、DS read、descriptor SALU
和 WMMA 在 steady body 中交错，不能把 source statement 强行映射成互不重叠的“大块”；
上表边界和下面的单指令 PC 则是 ELF byte-exact。

<a id="sec-1-5-dynamic-lds-arena-and-two-buffer-ring"></a>
### 1.5 Dynamic LDS arena and two-buffer ring

单 buffer 的精确 byte map：

| Relative byte range | Size | Tensor | Logical tile | LDS row stride |
|---|---:|---|---|---:|
| `0x0000..0x10ff` | 4,352 B | A | `16 x 256` FP8 | 272 B = 256 useful + 16 pad |
| `0x1100..0x90ff` | 32,768 B | B | physical `16 x 2048` bytes | 2,048 B |
| `0x9100..0x917f` | 128 B | SA | physical `16 x 2` i32 | 2 i32 |
| `0x9180..0x997f` | 2,048 B | SB | physical `8 x 64` i32 | 64 i32 |
| `0x9980..0x99ff` | 128 B | pitch pad | none | none |

两级 ring：

| Buffer | Base | A | B | SA | SB | Pitch pad |
|---|---:|---|---|---|---|---|
| 0 | `0x0000` | `0000..10ff` | `1100..90ff` | `9100..917f` | `9180..997f` | `9980..99ff` |
| 1 | `0x9a00` | `9a00..aaff` | `ab00..12aff` | `12b00..12b7f` | `12b80..1337f` | `13380..133ff` |
| Zero-tail only | `0x13400` | none | none | none | none | `0x13400..0x137ff`, 1,024 B extra allocation |

`tile_m<=64` 时 source 注释说明 SA 某些未被 WMMAScale 选中的 lane 会读到一个
unpopulated slot；若旧值是 `0xff`，按 E8M0 可形成有害值。因此 kernel 用 39 条静态
`ds_store_b128`，每条由 128 threads 合计清 2,048 B：

```text
39 * 128 threads * 16 B = 79,872 B
```

这解释了为什么必须清 actual rounded allocation，而不只是 logical ring。

GEMM 完成后 LDS base 被复用为 output staging。source 为通用 no-act 情况保守计算
`C_STORE_B=16*256*2=8192 B`；本 `stage1_act=1` 的真实输出只有
`16*128*2=4096 B`，位于 `0x0000..0x0fff`，且只在所有 GEMM 完成并 fence 后覆盖
buffer0。

<a id="sec-1-6-tdm-descriptors-and-exact-tensor-instructions"></a>
### 1.6 TDM descriptors and exact tensor instructions

FlyDSL 建立四个 2D TDM load job：

| Job | GM view | GM outer stride | LDS offset/shape/stride | K-tile immediate advance |
|---|---|---:|---|---:|
| A | `(16,256)` i8 | 7,168 bytes | `+0x0000`, `(16,256):(272,1)` | 256 bytes |
| B | `(16,2048)` i8 | 57,344 bytes | `+0x1100`, `(16,2048):(2048,1)` | 2,048 bytes |
| SA | `(16,2)` i32 | 56 i32 | `+0x9100`, `(16,2):(2,1)` | 8 bytes |
| SB | `(8,64)` i32 | 1,792 i32 | `+0x9180`, `(8,64):(64,1)` | 256 bytes |

Lowered group-0 address descriptor 形式为：

```text
[1, lds_addr, global_addr_low, global_addr_high | 0x80000000]
```

`20_llvm_ir.ll` 中可验证的 raw group-1 descriptor：

| Tensor | Raw descriptor words 0..7 | Verified meaning |
|---|---|---|
| A | `[0x07500000,0xffff0000,extent-packed,...|0x01000000,4,7168,0,0]` | i8、每 wave 4 rows、A row stride7168、256+16 LDS pad |
| B | `[0,0xffff0000,full-extent,...|0x08000000,4,57344,0,0]` | i8、每 wave 4 physical rows |
| SA | `[0x00020000,0xffff0000,full-extent,...|0x00020000,4,56,0,0]` | i32、每 wave 4 rows |
| SB | `[0x00020000,0xffff0000,full-extent,...|0x00400000,2,1792,0,0]` | i32、每 wave 2 N-super rows |
| Output | `[0x00010000,0xffff0000,mn_oob-packed,...|0x00800000,4,768,0,0]` | BF16、每 wave 4 output rows、GM stride768 |

上述 base、extent、stride 和 raw words 是 dump-verified；对每一个 mode bit 的命名解释则
不能只凭当前 dump 完整恢复。MI400 guide §4.10.1–4.10.4（printed pp.197–202，
extract markers Page207–212）说明 D# 由 SGPR groups 描述 global base、LDS base、
tensor/tile dimensions、stride 和 padding，且 `S_WAIT_TENSORCNT` 跟踪完成。该模型与实际
gfx1250 lowering 一致，但 MI450 的具体限制不自动等价于 gfx1250。

Initial K-tile 0 的四条 12-byte tensor instruction：

```asm
; kernel-relative PC        ISA
+0x07cc                     tensor_load_to_lds s[44:47], s[16:23] ; A
+0x09b8                     tensor_load_to_lds s[4:7],   s[24:31] ; B
+0x0a04                     tensor_load_to_lds s[44:47], s[36:43] ; SA
+0x0b34                     tensor_load_to_lds s[80:83], s[4:11]  ; SB
```

Steady-loop 中 prefetch next K tile：

```asm
+0x0c04                     tensor_load_to_lds s[44:47], s[16:23] ; next A
+0x0c68                     tensor_load_to_lds s[60:63], s[24:31] ; next B
+0x0c74                     tensor_load_to_lds s[44:47], s[36:43] ; next SA
+0x0cb0                     tensor_load_to_lds s[60:63], s[4:11]  ; next SB
```

MI400 guide §4.10（printed p.197 onward）提供三个有用、但需带 gfx1250 caution 的
硬件模型：

- tensor op 不按 lane 执行、忽略 EXEC；因此本 kernel 的 expert branch 必须 workgroup-uniform。
- tensor load/store 由 `TENSORcnt` 追踪，wave 内 tensor ops 保序，不同 wave 之间不保序。
- TDM 可与 shader 指令并行，且可在 LDS 中按规则插入 padding。

<a id="sec-1-7-prologue-27-steady-iterations-and-drain"></a>
### 1.7 Prologue, 27 steady iterations, and drain

编译期：

```text
K_TILES = 7168 / 256 = 28
num_buffers = 2
prologue issues = num_buffers - 1 = 1 K tile
n_steady = K_TILES - (num_buffers - 1) = 27
drain tiles = num_buffers - 1 = 1
```

控制流：

```text
Prologue:
    issue(buffer0, kt0)

for kt = 0..26:                          # 27 iterations
    s = kt % 2
    wait_tensorcnt(0); workgroup_barrier
    read B/SB/SA from buffer s
    issue(buffer((kt+1)%2), kt+1)        # mid-compute prefetch
    read A and execute 8 WMMAScale

Drain:
    wait buffer1
    compute kt27 with 8 WMMAScale
    no prefetch
```

最终 ISA loop：

```asm
+0x0b40 .LBB0_2:
+0x0b44     s_wait_tensorcnt 0x0
            s_cselect_b32 s45, 0, 0x9a00
            s_cselect_b32 s58, 0x9a00, 0
...
+0x0b74     s_barrier_signal -1
+0x0b78     s_barrier_wait -1
+0x0b80     ds_load_b128 v[62:65], v95
...
+0x0c04     tensor_load_to_lds s[44:47], s[16:23]
...
+0x0c68     tensor_load_to_lds s[60:63], s[24:31]
+0x0c74     tensor_load_to_lds s[44:47], s[36:43]
...
+0x0cb0     tensor_load_to_lds s[60:63], s[4:11]
...
+0x0ce8     v_wmma_scale_f32_16x16x128_f8f6f4 ...
...
+0x0e08     s_cbranch_scc1 .LBB0_2
```

Main-loop 8 个静态 WMMAScale site：

```text
+0x0ce8 +0x0cf8 +0x0d08 +0x0d18
+0x0da4 +0x0db8 +0x0dc8 +0x0dd8
```

Drain 8 个 site：

```text
+0x0eec +0x0f0c +0x0f1c +0x0f2c
+0x0fb8 +0x0fc8 +0x0fd8 +0x0fe8
```

所以：

```text
static WMMAScale sites / wave = 8 loop + 8 drain = 16
dynamic WMMAScale / wave      = 27*8 + 8 = 224
dynamic WMMAScale / WG        = 224 * 4 waves = 896 wave-instruction instances
```

“静态 16”和“动态 224”描述不同对象，不能互相替代。

<a id="sec-1-8-waits-barriers-and-scheduling-boundaries"></a>
### 1.8 Waits, barriers, and scheduling boundaries

| Purpose | Static PC(s) | Dynamic count / wave | Why required |
|---|---|---:|---|
| Zero stores complete | `s_wait_dscnt +0x068c` | 1 | arena clear 对所有 wave 可见 |
| Zero WG barrier | `+0x0690/+0x0694` | 1 pair | 避免任何 wave 提前使用未清 LDS |
| Steady tensor ready | `s_wait_tensorcnt +0x0b44` | 27 | `nb=2` 时 outstanding threshold 为 0 |
| Steady WG barrier | `+0x0b74/+0x0b78` | 27 pairs | 不同 wave 的 cooperative TDM segment 对全 WG 可见 |
| Main KSL0 DS ready | `s_wait_dscnt +0x0ce4` | 27 | 第一组 WMMA operands/scales ready |
| Main KSL1 DS ready | `+0x0da0,+0x0db4` | 27 each | 第二组 operands 与 scheduler dependency |
| Drain tensor ready | `+0x0e38` | 1 | kt27 buffer1 ready |
| Drain WG barrier | `+0x0e48/+0x0e4c` | 1 pair | kt27 LDS visibility |
| Drain DS waits | `+0x0ee0,+0x0fb4` | 1 each | 两个 KSL |
| Post-compute fence | `+0x0ffc,+0x1000,+0x1004` | 1 | epilogue 前结束 outstanding tensor work |
| Output DS/barrier | `+0x19e8,+0x19ec,+0x19f0` | 1 | 所有 BF16 LDS stores 对 TDM store 可见 |
| Output tensor wait | `+0x1a00` | 1 | GM write 完成后退出 |

Static ISA 中有 4 个 `s_wait_tensorcnt` site，动态执行：

```text
27 steady + 1 drain + 1 post-compute + 1 output-store = 30 / wave
```

Static barrier pair 有 5 处，动态执行：

```text
1 zero + 27 steady + 1 drain + 1 post-compute + 1 output = 31 pairs / wave
```

Source 中的 `rocdl.sched_*` 是 compiler scheduling hints；最终 ISA 没有同名 runtime
instruction。它们体现在 WMMAScale/DS/TDM 的排布、90 条 `s_delay_alu` 和 13 条
`v_nop` 中。

<a id="sec-1-9-vop3px2-wmmascale-and-hardware-operand-swap"></a>
### 1.9 VOP3PX2 WMMAScale and hardware operand swap

Source atom：

```python
WMMAScale(16, 16, 128, Float4E2M1FN, Float8E4M3FN, Float32)
```

数学计算是：

```text
Z[M,N] += A_activation[M,K] * W_weight[N,K]
```

但 hardware WMMA 的 `Matrix A/SRC0` 与软件 tensor 名称交换：

| View | Mathematical role | Hardware role | ISA registers | Scale operand |
|---|---|---|---|---|
| `arg_b` / weight | 数学右矩阵 `W^T` | Matrix A / `SRC0`, FP4 | e.g. `v[62:69]` | `scale_a = SB`, e.g. `v96` |
| `arg_a` / activation | 数学左矩阵 A | Matrix B / `SRC1`, FP8 | `v[0:15]` | `scale_b = SA`, e.g. `v94` |
| `c_frags[wn]` | FP32 partial sum | Matrix C / VDST accumulator | `v16:23`, `v24:31`, `v32:39`, `v40:47` | none |

代表性 ISA：

```asm
; +0x0ce8
v_wmma_scale_f32_16x16x128_f8f6f4 \
    v[16:23], v[62:69], v[0:15], v[16:23], v96, v94 \
    matrix_a_fmt:MATRIX_FMT_FP4
```

这条指令应读成：

```text
FP32_acc[16x16] +=
    dequant(FP4_weight_fragment, SB_E8M0) *
    dequant(FP8_activation_fragment, SA_E8M0)
```

而不是由 `arg_a/arg_b` 名字推断 `SRC0=activation`。

MI400 guide 的相关参考：

- §4.6.12（printed pp.143–158）：WMMA 是 wave-level matrix op，矩阵跨全部 32 lanes，
  `A*B+C -> D`，仅支持 wave32。
- §4.6.12.6（printed pp.153–158）：F4/F6/F8 block-scale；每 32 K values 一个 8-bit
  scale，E8M0 是合法组合。
- §11.3.7（printed pp.321–322，extract Page331–332）：VOP3PX2 为 4-DWORD/128-bit
  encoding，前 64 bits 是 scale-load 部分，后 64 bits 是 WMMA 部分。

实际 gfx1250 ELF 同样显示每条 WMMAScale 为 16 bytes；因此 **encoding width 和当前
operand semantics 是 dump-verified**。Guide 进一步说明 software 看见一个 VOP3PX2，
hardware 可把它视为 back-to-back `LD_SCALE + WMMA`，scale 只服务紧随的这一条 WMMA。
这一内部解释用于理解，而 MI450 的 clause bug、NOP workaround 和 rate 不外推。

<a id="sec-1-10-bias-clamp-gated-silu-and-bf16-epilogue"></a>
### 1.10 Bias, clamp, gated-SiLU, and BF16 epilogue

对每个 `(row,j)`，GUGU 相邻两列：

```text
z_g = accumulator[row, 2*j]     + bias[expert, 2*j]
z_u = accumulator[row, 2*j + 1] + bias[expert, 2*j + 1]

g = min(z_g, +7.0)              # gate 只有上界 clamp
u = min(max(z_u, -7.0), +7.0)   # up 双边 clamp

sigmoid(g) = 1 / (1 + exp(-g))
           = 1 / (1 + exp2(-log2(e) * g))

y[row,j] = BF16(g * sigmoid(g) * u)
```

精确 ISA chain：

1. `+0x10a0,+0x10ac,+0x10b8,+0x10c4`：四个 `global_load_b128`，每个 lane
   取得 8 个 BF16 bias。
2. BF16 不通过单独 `v_cvt` 解包，而是用 `v_and_b32 0xffff0000` 与
   `v_lshlrev_b32 16` 形成对应 FP32 bit pattern。
3. 16 个 `v_pk_add_f32` 把 32 个 pre-activation values/lane 与 bias 相加。
4. 48 个 clamp compare（16 pairs × 3）：
   gate `min(+L)`，up `max(-L)` 后 `min(+L)`。
5. 乘 `-log2(e)`，ISA literal `0xbfb8aa3b`。
6. 对 exp2 input 做 range repair：
   `v_cmp_gt_f32 ... 0xc2fc0000`（-126），必要时加 `0x42800000`（64），
   `v_exp_f32` 后用 `v_ldexp_f32` 和 `0xffffffc0`（-64）恢复指数。
7. 加 1，16 个 `v_rcp_f32`。
8. 16 个 `v_pk_mul_f32` 完成 `g*sigmoid(g)` 和再乘 `u`。
9. 8 个 `v_cvt_pk_bf16_f32`，每条把两个 FP32 source 打包为一个 BF16 pair。
10. 4 个 `ds_store_b64`/lane 写入 LDS。

代表性尾部：

```asm
+0x1970  v_cvt_pk_bf16_f32 v15, v14, v15
...
          v_cvt_pk_bf16_f32 v4, v8, v9
+0x19c8  ds_store_b64 v64, v[14:15]
+0x19d0  ds_store_b64 v65, v[2:3]
+0x19d8  ds_store_b64 v66, v[0:1] offset:32
+0x19e0  ds_store_b64 v64, v[4:5] offset:48
+0x19e8  s_wait_dscnt 0x0
+0x19ec  s_barrier_signal -1
+0x19f0  s_barrier_wait -1
+0x19f4  tensor_store_from_lds s[24:27], s[16:23]
+0x1a00  s_wait_tensorcnt 0x0
```

Guide references and cautions：

- §4.6.6（printed pp.127–128，extract Page137–138）确认
  `V_CVT_PK_BF16_F32` 把两个 source VGPR 转成一个 packed VGPR，F32 packed convert
  使用 round-to-nearest-even。
- Guide §4.6.10（printed p.142）讨论的是 `V_S_EXP/V_S_RCP` pseudo-scalar SGPR
  形式，**不是**本 kernel 的 lane-wise `V_EXP_F32/V_RCP_F32`。不能用该节替代实际 ISA。
- `exp2` 的 `+64 -> v_exp -> ldexp(-64)` 精确 range-repair 序列来自本 gfx1250 ISA
  和 LLVM OCML lowering；guide 未规定本 kernel 的这个 compiler algorithm。

---

## 2. Full GM/LDS/VGPR layout maps

<a id="sec-2-1-notation-and-complete-tensor-residency-map"></a>
### 2.1 Notation and complete tensor residency map

Notation：

```text
tid      = 0..127
wave     = tid // 32 = 0..3
lane     = tid % 32
lane16   = lane % 16
kgrp     = lane // 16 = 0 or 1
wave_n   = wave % 4 = wave
wn       = 0..3
ksl      = 0..1                # one tile_k=256 has two WMMA_K=128 steps
buf      = (K-tile index % 2) * 0x9a00
```

Tensor residency overview：

| Tensor | GM physical storage | LDS | VGPR | Accumulator/output role |
|---|---|---|---|---|
| A | uint8 FP8 `(1,30720,7168)` | ring A `[16,272B]` | 16 dwords/lane | hardware WMMA Matrix B |
| SA | uint8 E8M0 `(1,30720,224)`, ABI i32 `(...,56)` | ring SA `[16,2] i32` | one selected dword/lane | hardware `scale_b` |
| B | uint8 packed FP4, preshuffled `(384,1536,3584)` | ring B physical `[16,2048]` | 8 dwords/fragment/lane | hardware WMMA Matrix A |
| SB | uint8 E8M0 n32k4 `(384,48,7168)`, ABI i32 | ring SB `[8,64] i32` | four dwords/lane/KSL | hardware `scale_a` |
| Bias | BF16 `(384,1536)` | none | 4×8 BF16 then FP32/lane | added to FP32 acc |
| Acc | none | none | four × 8 FP32/lane | `16x64` output/wave |
| Activated C | none | `[16,128]` BF16 at LDS base | 16 BF16/lane | TDM store source |
| Y | BF16 `(1,30720,768)` | output staging source | none after store | GM final GEMM1 output |

<a id="sec-2-2-a-activation-gm---lds---vgpr"></a>
### 2.2 A activation: GM -> LDS -> VGPR

<a id="sec-2-2-1-a-gm-logicalphysical-layout"></a>
#### 2.2.1 A GM logical/physical layout

**A GM logical/physical transition table**

| Level | Shape / dtype | Coordinate meaning | Address |
|---|---|---|---|
| Original hidden | BF16 `(4096,7168)` | token × hidden K | upstream |
| Routed logical A | MXFP8 `(24576,7168)` | route row × K | valid subset |
| Capacity physical A | uint8 `(1,30720,7168)` | contiguous-M capacity × K | `arg_a` |
| Kernel tile | uint8 `(16,256)` | rows `blk_m..+15`, K `kt*256..+255` | `arg_a + blk_m*7168 + kt*256` |

每 byte 是一个 FP8 E4M3 code；数学值还需乘 SA：

```text
A_f32[r,k] =
    fp8_e4m3_decode(A_byte[r,k]) *
    e8m0_decode(SA_byte[r, floor(k/32)])
```

<a id="sec-2-2-2-a-gm---lds-tdm-layout"></a>
#### 2.2.2 A GM -> LDS TDM layout

**A GM -> LDS transition table**

| Item | Formula |
|---|---|
| GM byte | `A_GM[(blk_m+r)*7168 + kt*256 + k]` |
| Tile | `r=0..15`, `k=0..255` |
| LDS byte | `buf + r*272 + k` |
| Row padding | `buf + r*272 + 256..271` |
| Useful bytes | `16*256 = 4096` |
| LDS footprint | `16*272 = 4352` |
| OOB bound | outer extent uses `mn_oob`; TDM right-OOB read returns zero by descriptor semantics |

All-wave cooperative partition：

<div style="overflow-x: auto;">
<table style="white-space: nowrap; width: max-content; min-width: 1240px;">
  <thead>
    <tr>
      <th align="right">wave</th>
      <th align="right">A rows</th>
      <th>GM base for this wave segment</th>
      <th>LDS base for this wave segment</th>
      <th align="right">Segment bytes useful</th>
      <th align="right">Segment LDS bytes with pad</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td align="right">0</td>
      <td align="right">0..3</td>
      <td><code>arg_a + (blk_m+0)*7168 + kt*256</code></td>
      <td><code>buf + 0*1088</code></td>
      <td align="right">1,024</td>
      <td align="right">1,088</td>
    </tr>
    <tr>
      <td align="right">1</td>
      <td align="right">4..7</td>
      <td><code>arg_a + (blk_m+4)*7168 + kt*256</code></td>
      <td><code>buf + 1*1088</code></td>
      <td align="right">1,024</td>
      <td align="right">1,088</td>
    </tr>
    <tr>
      <td align="right">2</td>
      <td align="right">8..11</td>
      <td><code>arg_a + (blk_m+8)*7168 + kt*256</code></td>
      <td><code>buf + 2*1088</code></td>
      <td align="right">1,024</td>
      <td align="right">1,088</td>
    </tr>
    <tr>
      <td align="right">3</td>
      <td align="right">12..15</td>
      <td><code>arg_a + (blk_m+12)*7168 + kt*256</code></td>
      <td><code>buf + 3*1088</code></td>
      <td align="right">1,024</td>
      <td align="right">1,088</td>
    </tr>
  </tbody>
</table>
</div>

<a id="sec-2-2-3-a-lds---vgpr-wmma-b-layout"></a>
#### 2.2.3 A LDS -> VGPR WMMA-B layout

Source address：

```text
row = lane16
b0  = buf + row*272 + ksl*128 + kgrp*16
loads = ds_load_b128(b0 + 0, 32, 64, 96)
```

四次 16-byte DS load 经 shuffle 形成 `vector<16xi32>`，即 64 B/lane。32 lanes 合计
2,048 B，恰好是一份 `16x128` FP8 WMMA matrix。

**A LDS -> VGPR transition table**

<div style="overflow-x: auto;">
<table style="white-space: nowrap; width: max-content; min-width: 1420px;">
  <thead>
    <tr>
      <th>lane group</th>
      <th align="right">logical row</th>
      <th align="right"><code>kgrp</code></th>
      <th><code>v0..v3</code></th>
      <th><code>v4..v7</code></th>
      <th><code>v8..v11</code></th>
      <th><code>v12..v15</code></th>
      <th>WMMA role</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>lanes 0..15</td>
      <td align="right"><code>lane</code></td>
      <td align="right">0</td>
      <td><code>Kbase+0..15</code></td>
      <td><code>Kbase+32..47</code></td>
      <td><code>Kbase+64..79</code></td>
      <td><code>Kbase+96..111</code></td>
      <td>hardware Matrix B rows/columns per gfx1250 FP8 layout</td>
    </tr>
    <tr>
      <td>lanes 16..31</td>
      <td align="right"><code>lane-16</code></td>
      <td align="right">1</td>
      <td><code>Kbase+16..31</code></td>
      <td><code>Kbase+48..63</code></td>
      <td><code>Kbase+80..95</code></td>
      <td><code>Kbase+112..127</code></td>
      <td>同一 16 rows 的另一半 K strip</td>
    </tr>
  </tbody>
</table>
</div>

其中 `Kbase = kt*256 + ksl*128`。这个 map 与 MI400 guide
§4.6.12.6.1.1 的 wave32 16-VGPR FP8 layout 一致，也由实际 source 的四次
`ds_load_b128 + shuffle` 和 ISA 的 `v[0:15]` operand 共同验证。

<a id="sec-2-3-sa-activation-scale-gm---lds---vgpr"></a>
### 2.3 SA activation scale: GM -> LDS -> VGPR

SA 每 32 个 K values 一个 E8M0 byte。每个 A row 有 `7168/32=224` scale bytes，
ABI 以 4 bytes bit-view 为 i32，因此 row stride 是 56 i32。

**SA GM -> LDS transition table**

| Item | Formula / layout |
|---|---|
| GM byte shape | `(1,30720,224)` E8M0 |
| ABI i32 shape | `(1,30720,56)` |
| K-tile scale bytes / row | `256/32 = 8 B = 2 i32` |
| GM i32 | `SA[(blk_m+r)*56 + kt*2 + q]`, `q=0,1` |
| LDS i32 | `buf/4 + 0x9100/4 + r*2 + q` |
| Tile shape | `(16,2)` i32 |
| Footprint | `16*2*4 = 128 B` |

**SA per-wave GM -> LDS map**

<div style="overflow-x: auto;">
<table style="white-space: nowrap; width: max-content; min-width: 1020px;">
  <thead>
    <tr>
      <th align="right">wave</th>
      <th align="right">rows</th>
      <th>GM i32 base</th>
      <th>LDS byte base</th>
      <th align="right">bytes</th>
    </tr>
  </thead>
  <tbody>
    <tr><td align="right">0</td><td align="right">0..3</td><td><code>(blk_m+0)*56 + kt*2</code></td><td><code>buf+0x9100</code></td><td align="right">32</td></tr>
    <tr><td align="right">1</td><td align="right">4..7</td><td><code>(blk_m+4)*56 + kt*2</code></td><td><code>buf+0x9120</code></td><td align="right">32</td></tr>
    <tr><td align="right">2</td><td align="right">8..11</td><td><code>(blk_m+8)*56 + kt*2</code></td><td><code>buf+0x9140</code></td><td align="right">32</td></tr>
    <tr><td align="right">3</td><td align="right">12..15</td><td><code>(blk_m+12)*56 + kt*2</code></td><td><code>buf+0x9160</code></td><td align="right">32</td></tr>
  </tbody>
</table>
</div>

**SA LDS -> VGPR transition table**

| Lane | Source formula | Selected useful dword | WMMA role |
|---|---|---|---|
| lanes 0..15, `ksl=0` | `buf+0x9100 + lane16*8 + 0` | 4 E8M0 bytes for K `0..127` | `scale_b` |
| lanes 0..15, `ksl=1` | `buf+0x9100 + lane16*8 + 4` | 4 E8M0 bytes for K `128..255` | `scale_b` |
| lanes 16..31 | exact source adds `kgrp*4` | scale-select 未使用的 half 可能读下一 slot | zero-fill 防止 stale `0xff` 污染 |

精确 source：

```text
byte = lane16*(AS_INNER*4) + kgrp*4 + ksl*wmma_m_rep*4
     = lane16*8 + kgrp*4 + ksl*4
ds_load_b32(buf + 0x9100 + byte)
```

哪些 half/lane 被 VOP3PX2 scale selector 使用是硬件 layout contract；上述地址公式和
zero-fill 原因是 source-verified，未选 lane 的 microarchitectural consumption 不做超出
guide 的推断。

<a id="sec-2-4-b-weight-gm---lds---vgpr"></a>
### 2.4 B weight: GM -> LDS -> VGPR

<a id="sec-2-4-1-b-logical-versus-preshuffled-physical-layout"></a>
#### 2.4.1 B logical versus preshuffled physical layout

**B logical -> physical transition table**

| View | Shape | Element meaning |
|---|---|---|
| Logical weight | `(E,N,K) = (384,1536,7168)` FP4 | one E2M1 value |
| Packed row-major size notion | `(384,1536,3584)` bytes | two FP4 per byte |
| Actual kernel input | same outer byte shape but gfx1250 preshuffled | 不能按普通 `W[e,n,k/2]` 直接解释 |
| Kernel physical 2D view | `(N/16, (K/2)*16) = (96,57344)` bytes/expert | outer row represents 16 logical N rows |
| One TDM tile | `(16,2048)` bytes | N tile256 × K tile256 packed FP4 |

GUGU row order：

```text
logical N = [gate0, up0, gate1, up1, ..., gate767, up767]
```

<a id="sec-2-4-2-b-gm---lds-tdm-layout"></a>
#### 2.4.2 B GM -> LDS TDM layout

`b_outer_row = expert*(N/16) + blk_n/16`，`Kp16=(K/2)*16=57344`。

**B GM -> LDS transition table**

| Item | Formula |
|---|---|
| GM physical byte | `B[(expert*96 + blk_n/16 + outer)*57344 + kt*2048 + inner]` |
| `outer` | `0..15`, each covers 16 logical N rows |
| `inner` | `0..2047`, preshuffled 16-row × K256 payload |
| LDS byte | `buf + 0x1100 + outer*2048 + inner` |
| Footprint | `16*2048 = 32768 B` |

<div style="overflow-x: auto;">
<table style="white-space: nowrap; width: max-content; min-width: 1280px;">
  <thead>
    <tr>
      <th align="right">wave</th>
      <th align="right">physical outer rows</th>
      <th>Logical N rows represented</th>
      <th>GM base increment from tile base</th>
      <th>LDS base</th>
      <th align="right">bytes</th>
    </tr>
  </thead>
  <tbody>
    <tr><td align="right">0</td><td align="right">0..3</td><td><code>blk_n+0..63</code></td><td><code>0*4*57344</code></td><td><code>buf+0x1100</code></td><td align="right">8,192</td></tr>
    <tr><td align="right">1</td><td align="right">4..7</td><td><code>blk_n+64..127</code></td><td><code>1*4*57344</code></td><td><code>buf+0x3100</code></td><td align="right">8,192</td></tr>
    <tr><td align="right">2</td><td align="right">8..11</td><td><code>blk_n+128..191</code></td><td><code>2*4*57344</code></td><td><code>buf+0x5100</code></td><td align="right">8,192</td></tr>
    <tr><td align="right">3</td><td align="right">12..15</td><td><code>blk_n+192..255</code></td><td><code>3*4*57344</code></td><td><code>buf+0x7100</code></td><td align="right">8,192</td></tr>
  </tbody>
</table>
</div>

<a id="sec-2-4-3-b-lds---vgpr-wmma-a-layout"></a>
#### 2.4.3 B LDS -> VGPR WMMA-A layout

Source：

```text
b0 = buf + 0x1100
   + (wave_n*4 + wn)*2048
   + ksl*1024
   + kgrp*256
   + lane16*16

fragment = shuffle(
    ds_load_b128(b0),
    ds_load_b128(b0 + 512)
)  # vector<8xi32>
```

**B LDS -> VGPR transition table**

<div style="overflow-x: auto;">
<table style="white-space: nowrap; width: max-content; min-width: 1480px;">
  <thead>
    <tr>
      <th align="right">wave</th>
      <th align="right"><code>wn</code></th>
      <th>Logical N rows</th>
      <th>lanes 0..15 / <code>kgrp=0</code></th>
      <th>lanes 16..31 / <code>kgrp=1</code></th>
      <th>VGPR fragment</th>
      <th>WMMA role</th>
    </tr>
  </thead>
  <tbody>
    <tr><td align="right">0</td><td align="right">0</td><td><code>0..15</code></td><td><code>Kbase+0..31,64..95</code></td><td><code>Kbase+32..63,96..127</code></td><td><code>v62:69</code> in first loop KSL</td><td>Matrix A / SRC0 FP4</td></tr>
    <tr><td align="right">0</td><td align="right">1</td><td><code>16..31</code></td><td>same K strip pattern</td><td>same K strip pattern</td><td><code>v70:77</code></td><td>Matrix A / SRC0 FP4</td></tr>
    <tr><td align="right">0</td><td align="right">2</td><td><code>32..47</code></td><td>same</td><td>same</td><td><code>v78:85</code></td><td>Matrix A / SRC0 FP4</td></tr>
    <tr><td align="right">0</td><td align="right">3</td><td><code>48..63</code></td><td>same</td><td>same</td><td><code>v86:93</code></td><td>Matrix A / SRC0 FP4</td></tr>
    <tr><td align="right">1..3</td><td align="right">0..3</td><td><code>wave*64 + wn*16 .. +15</code></td><td>same FP4 K mapping</td><td>same FP4 K mapping</td><td>allocator-dependent</td><td>各 wave 自己的 16x64 N slice</td></tr>
  </tbody>
</table>
</div>

`Kbase=kt*256+ksl*128`。8 dwords/lane = 32 B/lane，32 lanes 合计 1,024 B，
等于 `16x128` 个 FP4 nibbles。Guide §4.6.12.6.1.3（printed pp.155–156）
给出同样的 wave32 8-VGPR FP4 matrix layout。

<a id="sec-2-5-sb-weight-scale-gm---lds---vgpr"></a>
### 2.5 SB weight scale: GM -> LDS -> VGPR

Logical weight scale shape 是 `(384,1536,224)` E8M0 bytes；preshuffle 后为：

```text
(E, N/32, (K/32)*32) = (384,48,7168) bytes
```

ABI flatten 后 bit-view 为 i32。一个 N-super-row=32 output rows；一个 K tile 有
8 scale bytes/output row，因此一个 super-row 有 `32*8=256 B=64 i32`。

**SB GM -> LDS transition table**

| Item | Formula / layout |
|---|---|
| GM outer stride | `K4 = K/4 = 1792 i32` |
| Tile base | `expert*ceil(N/32)*1792 + (blk_n/32)*1792` |
| Tile GM i32 | `base + s*1792 + kt*64 + j`, `s=0..7`, `j=0..63` |
| LDS i32 | `buf/4 + 0x9180/4 + s*64 + j` |
| Tile | `(8,64)` i32 |
| Footprint | `8*64*4=2048 B` |

**SB per-wave GM -> LDS map**

<div style="overflow-x: auto;">
<table style="white-space: nowrap; width: max-content; min-width: 1100px;">
  <thead>
    <tr>
      <th align="right">wave</th>
      <th align="right">N super-rows</th>
      <th>Logical N columns</th>
      <th>GM i32 increment</th>
      <th>LDS byte base</th>
      <th align="right">bytes</th>
    </tr>
  </thead>
  <tbody>
    <tr><td align="right">0</td><td align="right">0..1</td><td><code>blk_n+0..63</code></td><td><code>0*2*1792</code></td><td><code>buf+0x9180</code></td><td align="right">512</td></tr>
    <tr><td align="right">1</td><td align="right">2..3</td><td><code>blk_n+64..127</code></td><td><code>1*2*1792</code></td><td><code>buf+0x9380</code></td><td align="right">512</td></tr>
    <tr><td align="right">2</td><td align="right">4..5</td><td><code>blk_n+128..191</code></td><td><code>2*2*1792</code></td><td><code>buf+0x9580</code></td><td align="right">512</td></tr>
    <tr><td align="right">3</td><td align="right">6..7</td><td><code>blk_n+192..255</code></td><td><code>3*2*1792</code></td><td><code>buf+0x9780</code></td><td align="right">512</td></tr>
  </tbody>
</table>
</div>

**SB LDS -> VGPR transition table**

Source address：

```text
col_rel = wave_n*64 + wn*16 + lane16
idx = (col_rel//32)*64 + ksl*32 + (col_rel%32)
SB_dword = ds_load_b32(buf + 0x9180 + idx*4)
```

| `ksl` | Loaded dword bytes | Scale blocks represented | WMMA usage |
|---:|---|---|---|
| 0 | four E8M0 bytes | K `0..31,32..63,64..95,96..127` | `scale_a` for first K=128 WMMA |
| 1 | four E8M0 bytes | K `128..159,160..191,192..223,224..255` | `scale_a` for second K=128 WMMA |

每个 `wn` 一个 SB dword，所以第一组代表性 registers 为 `v96,v97,v98,v99`；它们分别
配对四个 B fragments。E8M0 是 exponent-only block scale；guide §4.6.12.6
（printed pp.153–154）确认 block-32 scale 是每 32 K values 一个 8-bit value，
而 §4.6.12.6.2（printed pp.156–158）说明一条 dword 可承载 K=128 的四个 scale bytes。

<a id="sec-2-6-vgpr-operands---fp32-accumulators"></a>
### 2.6 VGPR operands -> FP32 accumulators

**VGPR operands -> accumulator transition table**

| `wn` | Hardware Matrix A / weight | Hardware Matrix B / activation | Scale A/B | FP32 C/D accumulator |
|---:|---|---|---|---|
| 0 | `v62:69` FP4 | `v0:15` FP8 | `v96` SB / `v94` SA | `v16:23` |
| 1 | `v70:77` FP4 | `v0:15` FP8 | `v97` SB / `v94` SA | `v24:31` |
| 2 | `v78:85` FP4 | `v0:15` FP8 | `v98` SB / `v94` SA | `v32:39` |
| 3 | `v86:93` FP4 | `v0:15` FP8 | `v99` SB / `v94` SA | `v40:47` |

每个 wave 的 logical output ownership：

<div style="overflow-x: auto;">
<table style="white-space: nowrap; width: max-content; min-width: 1420px;">
  <thead>
    <tr>
      <th align="right">wave</th>
      <th align="right"><code>wave_n</code></th>
      <th>Pre-activation N span</th>
      <th align="right"><code>wn</code></th>
      <th>16x16 fragment N</th>
      <th>lane 0..15 acc columns</th>
      <th>lane 16..31 acc columns</th>
      <th>Accumulator VGPRs</th>
    </tr>
  </thead>
  <tbody>
    <tr><td align="right">0</td><td align="right">0</td><td><code>0..63</code></td><td align="right">0..3</td><td><code>wn*16..wn*16+15</code></td><td><code>+0..7</code></td><td><code>+8..15</code></td><td><code>v16:47</code> split 4×8</td></tr>
    <tr><td align="right">1</td><td align="right">1</td><td><code>64..127</code></td><td align="right">0..3</td><td><code>64+wn*16..+15</code></td><td><code>+0..7</code></td><td><code>+8..15</code></td><td>allocator-dependent, same logical layout</td></tr>
    <tr><td align="right">2</td><td align="right">2</td><td><code>128..191</code></td><td align="right">0..3</td><td><code>128+wn*16..+15</code></td><td><code>+0..7</code></td><td><code>+8..15</code></td><td>same</td></tr>
    <tr><td align="right">3</td><td align="right">3</td><td><code>192..255</code></td><td align="right">0..3</td><td><code>192+wn*16..+15</code></td><td><code>+0..7</code></td><td><code>+8..15</code></td><td>same</td></tr>
  </tbody>
</table>
</div>

Source epilogue 把 accumulator vector 解释为：

```text
row_rel = lane16
col_rel = wave_n*64 + wn*16 + kgrp*8
acc[0..7] = preactivation[row_rel, col_rel + 0..7]
```

因此每个 lane 持有四个 fragments × 8 FP32 = 32 FP32 accumulator values；一个 wave
覆盖 `16x64=1024` preactivation outputs，四个 wave 覆盖整个 `16x256`。

<a id="sec-2-7-bias-gm---vgpr-and-accumulator---gated-silu-vgpr"></a>
### 2.7 Bias GM -> VGPR and accumulator -> gated-SiLU VGPR

**Bias GM -> VGPR transition table**

| Item | Formula / layout |
|---|---|
| Bias physical/logical shape | BF16 `(384,1536)`, GUGU |
| Per-fragment base | `bias + expert*1536 + wave_n*64 + wn*16 + kgrp*8` |
| ISA load | one `global_load_b128` = 8 BF16 |
| Loads/lane | 4, offsets `0,32,64,96` bytes relative to wave/`kgrp` base |
| Total loaded per wave | `16 rows * 64 cols * 2 B = 2048 B` |
| Destination | unpack to FP32, add to corresponding 8-value accumulator fragment |

**Accumulator -> gated-SiLU VGPR transition table**

| Input pair | Clamp | Nonlinearity | Output |
|---|---|---|---|
| `acc[2p] + bias[2p]` | `g=min(x,7)` | `g*rcp(1+exp2(-log2(e)*g))` | gate factor FP32 |
| `acc[2p+1] + bias[2p+1]` | `u=min(max(x,-7),7)` | multiply by gate factor | one FP32 output |
| Four pairs / fragment | 4 outputs | packed with `v_cvt_pk_bf16_f32` | 4 BF16 |

每个 lane：4 fragments × 4 outputs = 16 BF16。每个 wave：32 lanes ×16 =512 BF16；
四个 wave：2048 BF16，正好等于 `16x128` activated output tile。

<a id="sec-2-8-bf16-vgpr---lds---gm-output"></a>
### 2.8 BF16 VGPR -> LDS -> GM output

`STORE_N = tile_n/2 = 128`。

**BF16 VGPR -> LDS transition table**

| Item | Formula / layout |
|---|---|
| Output row | `row_rel = lane16` |
| Output column base | `col_out = col_rel/2` |
| LDS byte | `(row_rel*128 + col_out)*2` |
| Store | `ds_store_b64`, 4 consecutive BF16 |
| Stores/lane | 4, one per `wn` |
| Wave column span | `wave_n*32 .. wave_n*32+31` |
| Full LDS tile | BF16 `(16,128)`, row stride256 B, footprint4096 B |

<div style="overflow-x: auto;">
<table style="white-space: nowrap; width: max-content; min-width: 1180px;">
  <thead>
    <tr>
      <th align="right">wave</th>
      <th>Pre-activation N</th>
      <th>Post-SiLU output N</th>
      <th>Rows written</th>
      <th>LDS column range</th>
      <th align="right">WG-produced bytes</th>
    </tr>
  </thead>
  <tbody>
    <tr><td align="right">0</td><td><code>0..63</code></td><td><code>0..31</code></td><td><code>0..15</code> via lane16</td><td><code>0..31</code></td><td align="right">1,024</td></tr>
    <tr><td align="right">1</td><td><code>64..127</code></td><td><code>32..63</code></td><td><code>0..15</code></td><td><code>32..63</code></td><td align="right">1,024</td></tr>
    <tr><td align="right">2</td><td><code>128..191</code></td><td><code>64..95</code></td><td><code>0..15</code></td><td><code>64..95</code></td><td align="right">1,024</td></tr>
    <tr><td align="right">3</td><td><code>192..255</code></td><td><code>96..127</code></td><td><code>0..15</code></td><td><code>96..127</code></td><td align="right">1,024</td></tr>
  </tbody>
</table>
</div>

输出 barrier 后，TDM 不再按“谁写了哪些 N 列”分工，而是按 4-row segments 读完整 LDS
rows。

**Output LDS -> GM transition table**

| Item | Formula / layout |
|---|---|
| LDS source | BF16 `(16,128):(128,1)` at base0 |
| GM row stride | `i32_n/2 = 1536/2 = 768` BF16 |
| GM tile base | `arg_c + blk_m*768 + blk_n/2` |
| GM element | `Y[(blk_m+r)*768 + blk_n/2 + c]` |
| TDM extent | outer rows bounded by `mn_oob` |
| ISA | `tensor_store_from_lds +0x19f4` |

<div style="overflow-x: auto;">
<table style="white-space: nowrap; width: max-content; min-width: 1080px;">
  <thead>
    <tr>
      <th align="right">TDM issuing wave</th>
      <th align="right">Output rows moved</th>
      <th>LDS byte base</th>
      <th>GM BF16 base</th>
      <th align="right">bytes</th>
    </tr>
  </thead>
  <tbody>
    <tr><td align="right">0</td><td align="right">0..3</td><td><code>0x000</code></td><td><code>(blk_m+0)*768 + blk_n/2</code></td><td align="right">1,024</td></tr>
    <tr><td align="right">1</td><td align="right">4..7</td><td><code>0x400</code></td><td><code>(blk_m+4)*768 + blk_n/2</code></td><td align="right">1,024</td></tr>
    <tr><td align="right">2</td><td align="right">8..11</td><td><code>0x800</code></td><td><code>(blk_m+8)*768 + blk_n/2</code></td><td align="right">1,024</td></tr>
    <tr><td align="right">3</td><td align="right">12..15</td><td><code>0xc00</code></td><td><code>(blk_m+12)*768 + blk_n/2</code></td><td align="right">1,024</td></tr>
  </tbody>
</table>
</div>

<a id="sec-2-9-summary-every-tensor-transition"></a>
### 2.9 Summary: every tensor transition

```text
A:
  GM uint8/FP8 [1,30720,7168]
  -> TDM tensor_load_to_lds
  -> LDS ring A [16,272B] (256B useful + 16B pad/row)
  -> ds_load_b128 x4 / KSL / lane
  -> VGPR 16 dwords/lane
  -> WMMAScale hardware Matrix B / SRC1.

SA:
  GM E8M0 bytes [1,30720,224], ABI i32 [1,30720,56]
  -> TDM
  -> LDS ring SA [16,2] i32
  -> ds_load_b32
  -> one packed 4-scale dword/lane
  -> WMMAScale scale_b.

B:
  logical FP4 [384,1536,7168]
  -> packed+preshuffled GM uint8 [384,1536,3584]
  -> physical TDM view [96,57344] bytes/expert
  -> LDS ring B [16,2048] bytes
  -> ds_load_b128 x2 / fragment / KSL / lane
  -> VGPR 8 dwords/fragment/lane
  -> WMMAScale hardware Matrix A / SRC0.

SB:
  logical E8M0 [384,1536,224]
  -> n32k4 GM bytes [384,48,7168], ABI i32
  -> TDM
  -> LDS ring SB [8,64] i32
  -> ds_load_b32, four scales packed/dword
  -> WMMAScale scale_a.

Accumulator:
  WMMAScale FP32 -> four c_frags x 8 FP32/lane, VGPR only.

Bias:
  GM BF16 [384,1536]
  -> global_load_b128 x4/lane
  -> BF16 bit-unpack to FP32 VGPR
  -> add to accumulators.

Activation/output:
  FP32 acc+bias
  -> gate/up clamp
  -> exp2 range repair + reciprocal + multiply
  -> v_cvt_pk_bf16_f32
  -> BF16 VGPR
  -> ds_store_b64 x4/lane
  -> LDS [16,128] BF16
  -> tensor_store_from_lds
  -> GM Y [1,30720,768] BF16.
```

<a id="sec-2-10-low-level-call-chains-and-issue-counts"></a>
### 2.10 Low-level call chains and issue counts

| Source statement | Role | Lowest-level primitive | ISA form | Static sites | Dynamic / wave |
|---|---|---|---|---:|---:|
| `issue(...A...)` | A GM -> LDS | `llvm.amdgcn.tensor.load.to.lds` | `tensor_load_to_lds` | 2 | 28 |
| `issue(...B...)` | B GM -> LDS | same | `tensor_load_to_lds` | 2 | 28 |
| `issue(...SA...)` | SA GM -> LDS | same | `tensor_load_to_lds` | 2 | 28 |
| `issue(...SB...)` | SB GM -> LDS | same | `tensor_load_to_lds` | 2 | 28 |
| `load_a` | A LDS -> VGPR | LLVM addrspace(3) vector load | `ds_load_b128` | 16 | 224 |
| `load_b` | B LDS -> VGPR | LLVM addrspace(3) vector load | `ds_load_b128` | 32 | 448 |
| `load_sa` | SA LDS -> VGPR | LLVM i32 LDS load | `ds_load_b32`/combined scheduling | included below | 56 logical |
| `load_sb` | SB LDS -> VGPR | LLVM i32 LDS load | `ds_load_b32`, some combined as `ds_load_2addr_b32` | included below | 224 logical |
| `fx.gemm(WMMAScale)` | scaled GEMM | WMMA-scale intrinsic | `v_wmma_scale_f32_16x16x128_f8f6f4` | 16 | 224 |
| bias `ptr_load` | bias GM -> VGPR | global vector load | `global_load_b128` | 4 | 4 |
| `fused_silu_swiglu_elem` | gated-SiLU | VALU/TRANS | cmp/exp/ldexp/rcp/pk_mul | one unrolled epilogue | once |
| `lds_store_b64_raw` | BF16 VGPR -> LDS | LLVM addrspace(3) store | `ds_store_b64` | 4 | 4 |
| output `fx.copy` | LDS -> GM | tensor-store intrinsic | `tensor_store_from_lds` | 1 | 1 |

说明：A/B logical load counts 与 physical DS opcode counts不同；`ds_load_2addr_b32`
一条 encoding 同时取两个 b32。

最终 ISA 静态计数与本 workload 动态计数：

| Opcode / family | Static physical sites | Dynamic / wave | Notes |
|---|---:|---:|---|
| All physical instructions | 1,104 | control-flow dependent | VOPD 每条 encoding 算 1 |
| WMMAScale | 16 | 224 | 27×8 + drain8 |
| `tensor_load_to_lds` | 8 | 112 | 28 K tiles × 4 tensors |
| `tensor_store_from_lds` | 1 | 1 | output |
| `ds_load_b128` | 48 | 672 | 27×24 + tail24 |
| `ds_load_2addr_b32` | 7 | 111 | 27×4 + tail3 |
| `ds_load_b32` | 6 | 58 | 27×2 + tail4 |
| Total physical DS reads | 61 | 841 | 672+111+58 |
| `ds_store_b128` | 39 | 39 | arena zero |
| `ds_store_b64` | 4 | 4 | BF16 output |
| `global_load_b32` | 11 | 11 | expert lookup10 + `mn_oob`1 |
| `global_load_b128` | 4 | 4 | bias |
| `s_wait_tensorcnt` | 4 | 30 | steady27 + drain + post + output |
| `s_wait_dscnt` | 7 | `1 + 27*3 + 2 + 1 = 85` | zero、steady、drain、output |
| barrier signal/wait pairs | 5 | 31 pairs | zero + steady27 + drain + post + output |
| `v_exp_f32` | 16 | 16 | one per output pair/lane |
| `v_ldexp_f32` | 16 | 16 | exp2 range repair |
| `v_rcp_f32` | 16 | 16 | sigmoid denominator |
| `v_pk_add_f32` | 16 | 16 | acc + bias |
| `v_pk_mul_f32` | 16 | 16 | gate*sigmoid and *up |
| `v_cvt_pk_bf16_f32` | 8 | 8 | 16 BF16 outputs/lane |
| `v_cmp_gt_f32` | 64 | 64 | 48 clamp +16 range checks |
| `v_dual_*` encodings | 45 | phase dependent | 每条包含两个 VALU components |
| `s_delay_alu` | 90 | phase dependent | compiler dependency schedule |
| `v_nop` | 13 | phase dependent | hazard/schedule spacing |

Call chains：

```text
add_tdm_loads / issue
  -> fx.copy(TDM atom)
  -> rocdl.tensor.load.to.lds
  -> llvm.amdgcn.tensor.load.to.lds
  -> tensor_load_to_lds

load_a / load_b
  -> lds_load_b128_raw
  -> LLVM load addrspace(3), vector<4xi32>
  -> ds_load_b128

load_sa / load_sb
  -> lds_load_b32_raw
  -> LLVM load addrspace(3), i32
  -> ds_load_b32 / ds_load_2addr_b32

mma_rows
  -> fx.gemm(WMMAScale atom, scale_a=SB, scale_b=SA)
  -> rocdl.wmma.scale.f32.16x16x128.f8f6f4
  -> llvm.amdgcn.wmma.scale...
  -> v_wmma_scale_f32_16x16x128_f8f6f4

fused_silu_swiglu_elem
  -> clamp selects + exp2 + rocdl.rcp
  -> OCML/LLVM lowering
  -> v_cmp/v_cndmask/v_exp/v_ldexp/v_rcp/v_pk_mul

lds_store_b64_raw
  -> LLVM store vector<2xi32>, addrspace(3)
  -> ds_store_b64

output fx.copy
  -> rocdl.tensor.store.from.lds
  -> llvm.amdgcn.tensor.store.from.lds
  -> tensor_store_from_lds
```

---

## 3. Final summary

<a id="sec-3-1-verified-facts-versus-inference"></a>
### 3.1 Verified facts versus inference

**Dump/source verified：**

- fixed workload、routing geometry、grid=11,520、balanced `m_tile_map`；
- tile/WG/K-loop specialization；
- logical ring 78,848 B 与 actual dynamic allocation 79,872 B；
- VGPR/SGPR/AGPR/spill/private-segment metadata；
- embedded ELF 的 PC、VMA、function byte size；
- 所有 TDM/DS/WMMA/bias/SiLU/output opcode sites 和静态计数；
- 27 steady iterations、1 drain、每 wave 224 WMMAScale；
- A/B/SA/SB 的 GM/LDS address formulas；
- hardware SRC0=weight、SRC1=activation 和 `scale_a=SB, scale_b=SA`；
- gated-SiLU 的 clamp、`-log2(e)`、exp2 range repair、rcp、BF16 pack；
- fixed LDS metadata 为 0、launcher dynamic LDS 为 79,872 的表面矛盾及其原因。

**Guide-supported interpretation，且已注明 gfx1250 caution：**

- TDM descriptor/counter/EXEC-independent 模型；
- block-32 E8M0 scale 的 WMMA layout；
- VOP3PX2 软件单指令、硬件 scale-load+WMMA 的解释；
- packed BF16 convert 的 two-source/RNE 行为。

**不从当前材料过度推断：**

- 未文档化的 gfx1250 TDM flag bit 名称；
- MI450 专属 throughput、NOP 数、clause bug、LDS allocation granularity；
- WMMA 内部 XDL pipeline latency 和实际 occupancy；
- runtime cache transaction 数、bank-conflict 实测值；
- runtime relocated code VA；
- 未被当前 WMMAScale selector 使用的 scale-lane 是否在内部真正读取。

<a id="sec-3-2-key-conclusions"></a>
### 3.2 Key conclusions

1. 该 GEMM1 不是普通 row-major A8W4 GEMM。B 已按 gfx1250 WMMA/TDM preshuffle，
   SB 已转成 `n32k4`；逻辑 shape 只能用于数学解释，不能直接用于 byte addressing。
2. TDM 同时搬运 A、B、SA、SB，两个 `0x9a00`-pitch LDS buffers 构成 ring。source
   打印的 78,848 B 是逻辑 ring，真正 launch 分配 79,872 B，并全部清零。
3. 初始只 prefetch kt0；`.LBB0_2` 执行 27 次，计算 kt0..26 并 mid-compute prefetch
   kt1..27；随后 drain kt27。没有额外隐藏的第 29 个 K tile。
4. 最终 ISA 只有 16 个 WMMAScale site，但 loop 展开语义使一个 wave 动态执行 224 条，
   一个 workgroup 执行 896 个 wave-level instances。
5. hardware operand 与软件 tensor 名称交换：weight FP4 是 Matrix A/SRC0，activation
   FP8 是 Matrix B/SRC1；SB 是 `scale_a`，SA 是 `scale_b`。
6. E8M0 scale 每 32 K values 一个 byte；每个 K=128 WMMAScale 需要四个 scale bytes，
   恰好由一个 i32 携带。
7. Bias 直接 GM->VGPR，不进 LDS。FP32 accumulator 加 bias 后，相邻 GUGU 列做
   `g=min(g,7)`、`u=clamp(u,-7,7)`、`g*sigmoid(g)*u`。
8. `sigmoid` 精确实现为 `exp2(-log2(e)*g)`，compiler 插入 `+64/ldexp(-64)` range
   repair；最后 8 个 packed convert 生成 16 BF16/lane。
9. 输出不是 direct VGPR->GM：先由 4 个 `ds_store_b64`/lane 形成 LDS `[16,128]`，
   barrier 后由 `tensor_store_from_lds` 写入 GM `[1,30720,768]`。
10. `.amdhsa_group_segment_fixed_size 0` 与大量 LDS 指令并不冲突；它只说明 fixed LDS
    为 0，本 kernel 使用 launcher 指定的 dynamic LDS。
