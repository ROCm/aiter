# `gemm1.v0.s` ISA 分析：`v_dual_cndmask_b32`

## 1. 统计结论

`gemm1.v0.s` 中共有：

- 12 条包含 `v_dual_cndmask_b32` 的 VOPD/VOPD3 复合指令；
- 19 个 `v_dual_cndmask_b32` 子操作；
- 按 `::` 两侧的严格配对形式，可分为 4 种；
- 按 `v_dual_cndmask_b32` 自身的编码方式，可分为 2 种：隐式 VCC 的 VOPD 和显式 selector 的 VOPD3。

## 2. 基本语义

`v_dual_cndmask_b32` 对 wave 中每个活动 lane 执行一次 32-bit 条件选择：

```text
VDST[lane] = condition[lane] ? SRC1[lane] : SRC0[lane]
```

- `condition[lane] = 1` 时选择 `SRC1`；
- `condition[lane] = 0` 时选择 `SRC0`；
- `b32` 表示原样复制 32-bit 数据，不进行整数或浮点格式转换；
- 不在 `EXEC` 中的 lane 不写目的 VGPR；
- 当数据源为 SGPR 时，该值广播到 wave 的所有 lane。

因此，虽然条件可能由整数比较或浮点比较产生，`v_dual_cndmask_b32` 本身只是按位复制被选中的 32-bit 数据。

例如：

```asm
v_dual_cndmask_b32 v2, v4, v2 :: v_dual_cndmask_b32 v1, v5, v1
```

等价于同时执行：

```text
v2 = VCC[lane] ? old_v2 : v4
v1 = VCC[lane] ? old_v1 : v5
```

这里目的寄存器同时也是一个输入寄存器，表示“条件成立时保留旧值，否则用另一个候选值替换”。

## 3. 两种编码形式

### 3.1 VOPD：隐式使用 VCC

语法为：

```asm
v_dual_cndmask_b32 vdst, src0, src1
```

这种写法没有显式给出 selector，选择条件隐式来自：

```text
VCC[lane]
```

本 kernel 使用 wave32，所以实际使用的是 `vcc_lo` 的 32 个条件位。

例如：

```asm
v_cmp_lt_i32_e32 vcc_lo, s34, v1
v_dual_cndmask_b32 v1, v3, v4 :: v_dual_cndmask_b32 v2, v2, v6
```

其含义为：

```text
v1 = vcc_lo[lane] ? v4 : v3
v2 = vcc_lo[lane] ? v6 : old_v2
```

文件中有 8 条这种 VOPD bundle，共包含 14 个 CNDMASK 子操作：

- 第 313、348、381、392、403、415 行：每条包含两个 CNDMASK；
- 第 978、997 行：每条包含一个 CNDMASK。

普通 VOPD 是 64-bit 编码，两个目的 VGPR 必须一个为偶数、另一个为奇数。例如：

```asm
v_dual_cndmask_b32 v2, ... :: v_dual_cndmask_b32 v1, ...
```

其中 `v2` 为偶数 VGPR，`v1` 为奇数 VGPR。

### 3.2 VOPD3：显式给出 selector

语法为：

```asm
v_dual_cndmask_b32 vdst, src0, src1, selector
```

例如：

```asm
v_dual_cndmask_b32 v2, v2, v3, vcc_lo :: v_dual_bitop2_b32 v5, 1, v3 bitop3:0x54
```

左侧操作为：

```text
v2 = vcc_lo[lane] ? v3 : old_v2
```

CDNA5 的 VOPD3 CNDMASK selector 可以是 VCC，也可以是任意 SGPR；selector 被计算为一次 SGPR 读取。本文件中的显式 selector 都是 `vcc_lo`。

文件中有 4 条这种 VOPD3 bundle，共包含 5 个 CNDMASK 子操作：

- 第 321 行：一个 CNDMASK；
- 第 335 行：两个 CNDMASK；
- 第 960、985 行：各一个 CNDMASK。

VOPD3 是 96-bit 编码。与普通 VOPD 相比，它取消了目的 VGPR 必须一奇一偶的限制，但两个目的寄存器仍不能相同或重叠。

例如：

```asm
v_dual_cndmask_b32 v2, v2, v6, vcc_lo :: v_dual_cndmask_b32 v4, v4, v1, vcc_lo
```

两个目的寄存器 `v2`、`v4` 都是偶数，普通 VOPD 无法编码，因此必须使用 VOPD3。

第 321 行也必须使用 VOPD3，因为 `v_dual_bitop2_b32` 只存在于 VOPD3 的 OPY 操作位置。

## 4. 按左右配对形式分类

### 4.1 `CNDMASK :: CNDMASK`

共有 7 条：

- 第 313、335、348、381、392、403、415 行。

示例：

```asm
v_dual_cndmask_b32 v1, v5, v1 :: v_dual_cndmask_b32 v2, v2, v3
```

等价于：

```text
v1 = VCC[lane] ? old_v1 : v5
v2 = VCC[lane] ? v3 : old_v2
```

这种形式一次完成两组独立的条件选择。在本 kernel 前半段，它主要用于同时更新成对的索引或地址候选，避免使用分支。

### 4.2 `CNDMASK :: BITOP2`

共有 1 条：

```asm
v_dual_cndmask_b32 v2, v2, v3, vcc_lo :: v_dual_bitop2_b32 v5, 1, v3 bitop3:0x54
```

左侧根据 `vcc_lo` 更新 `v2`：

```text
v2 = vcc_lo[lane] ? v3 : old_v2
```

右侧的 `bitop3:0x54` 对两个输入实现按位 OR，因此：

```text
v5 = 1 | v3
```

这样可以在选择一个索引候选的同时，并行产生另一个置最低位的候选值。

### 4.3 `CNDMASK :: LSHLREV`

共有 2 条，位于第 978、997 行。

示例：

```asm
v_dual_cndmask_b32 v17, s54, v0 :: v_dual_lshlrev_b32 v10, 16, v10
```

左侧执行条件选择，右侧同时执行：

```text
v10 = v10 << 16
```

该组合把浮点值的条件截断与另一路独立的 16-bit 半字重排放进同一个 VOPD bundle。

### 4.4 `LSHLREV :: CNDMASK`

共有 2 条，位于第 960、985 行。

示例：

```asm
v_dual_lshlrev_b32 v58, 16, v9 :: v_dual_cndmask_b32 v32, s54, v16, vcc_lo
```

左侧执行：

```text
v58 = v9 << 16
```

右侧执行：

```text
v32 = vcc_lo[lane] ? v16 : s54
```

这一方向使用 VOPD3。例如上面的目的寄存器 `v58`、`v32` 都是偶数，不满足普通 VOPD 的奇偶目的寄存器约束。

OPX 和 OPY 支持的 opcode 集合并不完全相同，因此左右位置不是任意可交换的：

- CNDMASK 在 OPX 和 OPY 中都受支持；
- 普通 VOPD 的 `LSHLREV` 位于 OPY，因此紧凑形式写成 `CNDMASK :: LSHLREV`；
- VOPD3 扩展了 OPX opcode，可以写成 `LSHLREV :: CNDMASK`；
- `BITOP2` 只存在于 VOPD3 的 OPY，因此第 321 行必须写成 `CNDMASK :: BITOP2`。

## 5. 在本 kernel 中的实际功能

### 5.1 前半段：索引和地址候选选择

第 313～415 行附近先通过 `v_cmp_lt_i32` 产生逐 lane 的 `vcc_lo`，随后使用 CNDMASK 在两组索引或地址候选之间选择。

例如：

```asm
v_cmp_lt_i32_e32 vcc_lo, s34, v4
v_add_nc_u32_e32 v5, 1, v3
v_dual_cndmask_b32 v1, v5, v1 :: v_dual_cndmask_b32 v2, v2, v3
```

对每个 lane：

```text
v1 = (s34 < v4) ? old_v1 : v5
v2 = (s34 < v4) ? v3 : old_v2
```

这种无分支选择方式可以避免修改 `EXEC`，也能把两个相关的候选更新压缩到一次 VOPD 发射中。

### 5.2 后半段：浮点上界和下界截断

后半段使用 SGPR 作为 `SRC0`，该 SGPR 的值广播到所有 lane。

上界截断示例：

```asm
v_cmp_gt_f32_e32 vcc_lo, s54, v16
v_dual_lshlrev_b32 v58, 16, v9 :: v_dual_cndmask_b32 v32, s54, v16, vcc_lo
```

对普通非 NaN 数值：

```text
v32 = (s54 > v16) ? v16 : s54
    = min(v16, s54)
```

类似地，第 978、985 行也计算相应输入与 `s54` 的上界截断。

代码前面通过：

```asm
s_sub_f32 s0, 0, s54
```

计算：

```text
s0 = -s54
```

因此下面的操作形成下界截断：

```asm
v_cmp_gt_f32_e64 vcc_lo, v3, -s54
v_dual_cndmask_b32 v35, s0, v3 :: v_dual_lshlrev_b32 v14, 16, v14
```

对普通非 NaN 数值：

```text
v35 = (v3 > -s54) ? v3 : -s54
    = max(v3, -s54)
```

需要注意，这里使用的是比较加条件选择，而不是 IEEE `minimumNumber`/`maximumNumber` 指令，所以 NaN 情况由比较结果和选择方向决定，不能简单视为完全等价的 IEEE min/max。

## 6. 发射与执行的特殊之处

### 6.1 同一个 wave 的双 VALU 发射

`::` 两侧不是顺序执行，而是一个复合指令中的两个子操作：

- 左侧是 OPX，进入 coreMACC；
- 右侧是 OPY，进入 sideMACC；
- 两个操作由同一个 wave32 在同一周期发射和开始执行。

因此，VOPD/VOPD3 属于同一 wave 内的双发射，不要与 MI450 从两个不同 wave 各选择一条 VALU 的跨-wave双发射机制混淆。

### 6.2 仅支持 wave32

VOPD/VOPD3 只对 wave32 合法。本 kernel 明确配置为：

```asm
.amdhsa_wavefront_size32 1
```

所以 `vcc_lo` 的 32 个 bit 恰好对应 wave32 的 32 个 lane。

### 6.3 两个子操作必须相互独立

OPX 和 OPY 在同一周期执行，不能把左侧本次产生的结果直接作为右侧本次操作的输入，反之亦然。两个子操作读取的是发射前已经可用的源值。

本文件中的配对都满足这一点，例如：

```asm
v_dual_cndmask_b32 v2, v4, v2 :: v_dual_cndmask_b32 v1, v5, v1
```

左侧不读取或写入右侧使用的寄存器，右侧也不依赖左侧本次产生的结果。

### 6.4 目的寄存器限制

普通 64-bit VOPD：

- 两个目的 VGPR 必须一个为偶数、另一个为奇数。

96-bit VOPD3：

- 目的 VGPR 可以具有相同奇偶性；
- 但两个目的寄存器不能相同或重叠。

这正是第 335、960、985 行使用 VOPD3 的重要原因。

### 6.5 VGPR 源 bank 和读端口限制

VOPD 的两个子操作同时读取源操作数，因此必须满足 VGPR source-cache 端口限制：

- `SRCX0` 与 `SRCY0` 必须是同一个、同宽度 VGPR，或者位于不同的 VGPR bank；
- `SRCX1` 与 `SRCY1` 同样如此；
- 软件可按 `VGPR编号 % 4` 判断架构规定的 bank；
- VOPD3 的对应源位置也有类似约束。

这些是硬件正确性约束，不只是潜在的性能优化建议。通常由汇编器和编译器负责保证。

### 6.6 DPP 和 literal 限制

- VOPD 和 VOPD3 都不能使用 DPP；
- 普通 VOPD 最多携带一个共享的 32-bit literal，不支持 64-bit literal；
- VOPD3 不允许 literal。

### 6.7 `V_CMP → CNDMASK` 的 VCC 快速转发

普通 main VALU 流水线较深，但 CDNA5 对下面的依赖提供专用快速转发：

```text
V_CMP -> V_CNDMASK
```

比较产生的 VCC 可以零等待地转发给紧随其后的 CNDMASK，因此可以看到：

```asm
v_cmp_lt_i32_e32 vcc_lo, s34, v1
v_dual_cndmask_b32 v1, v3, v4 :: v_dual_cndmask_b32 v2, v2, v6
```

中间不需要为新产生的 `vcc_lo` 插入 4 个普通 VALU wait-state。

该快速通路只解决条件掩码的依赖：

- CNDMASK 的 `SRC0`、`SRC1` 仍必须已经就绪；
- CNDMASK 产生的目的 VGPR 也仍具有正常 VALU 流水线延迟；
- 后续依赖者仍由硬件 scoreboard、数据转发或显式 `s_delay_alu` 保证正确时序。

### 6.8 吞吐率不等于结果零延迟

对于这些单周期子操作，一个 VOPD bundle 可以在一个周期内启动两个 wave32 VALU 操作，从而提高 coreMACC 和 sideMACC 的利用率。

但是 main VALU 仍是多级流水线。所谓“双发射”表示两个操作可以同时进入流水线，并不表示两个结果在发射周期内立即可供后续指令使用。

## 7. 总结

`gemm1.v0.s` 对 `v_dual_cndmask_b32` 的使用可以概括为：

1. 前半段用两个并行 CNDMASK 无分支地更新索引和地址候选；
2. 用 `CNDMASK :: BITOP2` 同时完成条件选择和索引最低位置位；
3. 后半段把 CNDMASK 的浮点上下界截断与独立的 16-bit 左移重排配对；
4. 根据目的寄存器奇偶、第三源和 OPX/OPY opcode 能力，在 64-bit VOPD 与 96-bit VOPD3 之间选择；
5. 利用 `V_CMP -> CNDMASK` 的 VCC 快速转发，使比较和条件选择能够紧密排列；
6. 通过一次同-wave双发射同时使用 coreMACC 和 sideMACC，提高普通 VALU 的执行密度。

---

# `gemm1.v0.s` ISA 分析：`tensor_load_to_lds`

## 1. 统计与分类结论

`gemm1.v0.s` 中静态出现 8 条 `tensor_load_to_lds`，但需要区分三种不同的计数方式：

1. **按指令语法分类：1 种。** 全部使用两个 SGPR descriptor group，执行普通二维 tensor load。
2. **按搬运的数据功能分类：4 种。** 分别搬运 A、B、SA 和 SB。
3. **按软件流水阶段分类：2 种。** 前四个静态 site 负责初始 K tile 的 prologue 预取，后四个 site 负责 steady-loop 中下一个 K tile 的双缓冲预取。

因此，回答“有几种用法”时，最有意义的结论是：

```text
8 个静态指令 site
= 4 种 tensor 数据功能
× 每种 2 个流水阶段 site
```

四种 tensor 数据分别是：

| 类别 | 数据 | GM 逻辑/物理 tile | LDS 起始偏移 |
|---|---|---|---:|
| A | FP8 activation | `[16,256]` bytes | `0x0000` |
| B | preshuffled FP4 weight | `[16,2048]` bytes | `0x1100` |
| SA | activation E8M0 scale | `[16,2]` i32 | `0x9100` |
| SB | preshuffled weight E8M0 scale | `[8,64]` i32 | `0x9180` |

## 2. 指令的基本语义

汇编语法为：

```asm
tensor_load_to_lds s[group0_first:group0_last], s[group1_first:group1_last]
```

该指令把 SGPR 中的 Tensor DMA Descriptor（D#）提交给 TDM 硬件，异步执行：

```text
Global Memory 中的二维 tensor tile
        ↓
按照 descriptor 给出的维度、stride 和 padding 搬运
        ↓
LDS 中指定的目标区域
```

两个操作数的用途为：

- 第一个操作数是 4 个 SGPR，提供 D# group 0；
- 第二个操作数是 8 个 SGPR，提供 D# group 1；
- group 0 主要包含有效 descriptor 标志、LDS 地址和 57-bit Global 地址；
- group 1 主要包含元素大小、tensor/tile 维度、GM stride、LDS padding 等信息。

本文件中的所有 `tensor_load_to_lds` 都只使用 group 0 和 group 1，未提供 group 2/group 3，所以都是二维 tensor load，不是 3D～5D、gather 或 descriptor-iteration 形式。

这类指令还有几个重要特征：

- 它不使用 VGPR 搬运数据，数据直接从 GM 进入 LDS；
- 它不是逐 lane 执行的 VALU/VMEM 操作；
- 它忽略 `EXEC`，即使 `EXEC==0` 也会提交 tensor operation；
- 一条 tensor 指令只产生一次 Tensor-Done，而不是每个内部 memory transaction 产生一次；
- 完成状态由 `TENSORcnt` 跟踪。

## 3. 用法一：加载 A——FP8 activation

初始 K tile 的静态 site 为：

```asm
tensor_load_to_lds s[44:47], s[16:23]
```

steady-loop 中预取下一个 K tile 时再次使用：

```asm
tensor_load_to_lds s[44:47], s[16:23]
```

它搬运 activation A：

```text
GM:  FP8 A tile [16,256 bytes]
LDS: offset 0x0000
```

完整 workgroup tile 的参数为：

| 属性 | 值 |
|---|---:|
| A 数据格式 | FP8 E4M3，每个元素 1 byte |
| tile 行数 | 16 |
| 每行当前 K tile 数据 | 256 bytes |
| GM 行 stride | 7168 bytes |
| LDS 行有效数据 | 256 bytes |
| LDS 行尾 padding | 16 bytes |
| LDS 行 stride | 272 bytes |
| 每个 K tile 的 GM 地址增量 | 256 bytes |

本 kernel 有四个 wave，未采用 wave-specialized TDM。四个 wave 都执行该指令，但每个 wave 的 descriptor 地址和 outer segment 不同：

```text
每个 wave 搬运 4 行
4 wave × 4 行 = 完整的 16 行 A tile
```

A 是四种 load 中唯一启用 LDS padding 的类型。其 descriptor 的首 DWORD 为：

```text
0x07500000
```

对应的关键设置为：

- `data_size=1 byte`；
- 启用 LDS padding；
- 每写入 256 bytes 后跳过 16 bytes；
- LDS 中形成 `(16,256):(272,1)` 的布局。

padding 使下一行从新的 LDS bank 相位开始，有利于后续 `ds_load_b128` 的访问布局，并避免把连续的 256-byte 行简单叠在同一 bank 相位上。

A descriptor 还带有根据 `mn_oob` 计算的有效行边界。tile 超出有效 activation 行数时，TDM 对越界读取返回零。因此它同时承担：

1. GM 到 LDS 的二维搬运；
2. M 方向尾块的 OOB 保护；
3. LDS 行 padding 布局生成。

## 4. 用法二：加载 B——preshuffled FP4 weight

初始 K tile 使用：

```asm
tensor_load_to_lds s[4:7], s[24:31]
```

steady-loop 中使用更新后的 group 0：

```asm
tensor_load_to_lds s[60:63], s[24:31]
```

第二个 descriptor group 始终是 `s[24:31]`，说明两条指令描述的是同一种 tensor 几何；第一个 group 不同，是因为当前 GM 地址和 LDS 双缓冲地址已经重新计算。

它搬运 preshuffled weight B：

```text
GM:  packed/preshuffled FP4 B tile [16,2048 bytes]
LDS: offset 0x1100
```

参数为：

| 属性 | 值 |
|---|---:|
| 物理元素单位 | packed byte |
| 外层物理行数 | 16 |
| 每条物理行 | 2048 bytes |
| GM 物理行 stride | 57344 bytes |
| LDS 行 stride | 2048 bytes |
| LDS padding | 无 |
| 每个 K tile 的 GM 地址增量 | 2048 bytes |

这里的 `[16,2048]` 是 weight preshuffle 后的物理 view，不应直接解释成普通数学矩阵的 16 行 × 2048 列。FP4 每个数只占 4 bit，并且 kernel 输入已经按照 WMMA 所需的 16×16 byte tile 形式重排。

同样由四个 wave 协作：

```text
每个 wave 搬运 4 条物理行
4 wave × 4 行 = 16 条物理行
```

B 不需要 A 那样的 16-byte 行 padding，因为 preshuffled B 的 2048-byte LDS 行布局已经与后续 weight fragment 的 `ds_load_b128` 地址公式匹配。

## 5. 用法三：加载 SA——activation E8M0 scale

初始和循环 site 分别为：

```asm
tensor_load_to_lds s[44:47], s[36:43]
```

```asm
tensor_load_to_lds s[44:47], s[36:43]
```

它搬运 activation scale SA：

```text
GM:  SA tile [16,2] i32
LDS: offset 0x9100
```

descriptor 的首 DWORD 为：

```text
0x00020000
```

其中 `data_size=4 bytes`，因此 TDM 以 i32 为元素单位搬运。这里并不表示一个 scale 是 FP32；实际每个 i32 打包四个 8-bit E8M0 scale。

对于一个 `K=256` tile：

```text
scale block size = 32
256 / 32 = 8 个 E8M0 scale / activation row
8 byte = 2 个 i32
```

参数为：

| 属性 | 值 |
|---|---:|
| tile | `[16,2]` i32 |
| 每行 scale 数 | 8 个 E8M0 byte |
| GM 行 stride | 56 i32 |
| LDS 行 stride | 2 i32 |
| 每个 K tile 的 GM 地址增量 | 8 bytes |
| LDS padding | 无 |

四个 wave 各搬运四行 SA，合计覆盖与 A tile 相同的 16 个 activation 行。

后续 LDS 读取把一个 packed i32 作为四个 E8M0 scale byte，传给 `v_wmma_scale_f32_16x16x128_f8f6f4` 的 activation scale 输入。

## 6. 用法四：加载 SB——weight E8M0 scale

初始 K tile 使用：

```asm
tensor_load_to_lds s[80:83], s[4:11]
```

steady-loop 使用：

```asm
tensor_load_to_lds s[60:63], s[4:11]
```

它搬运 weight scale SB：

```text
GM:  preshuffled SB tile [8,64] i32
LDS: offset 0x9180
```

SB descriptor 同样使用：

```text
data_size = 4 bytes
```

每个 i32 打包四个 E8M0 scale byte。SB 的物理布局是针对 weight 的 `n32k4` preshuffle，而不是普通的 `[N,K/32]` 行主序矩阵，所以 descriptor view 为 `[8,64]` i32。

参数为：

| 属性 | 值 |
|---|---:|
| tile | `[8,64]` i32 |
| GM 外层 stride | 1792 i32 |
| LDS 行 stride | 64 i32 |
| 每个 K tile 的 GM 地址增量 | 256 bytes |
| LDS padding | 无 |

四个 wave 各搬运两个 N-super-row：

```text
每个 wave 2 行
4 wave × 2 行 = 8 行
```

后续 `ds_load_b32` 从该区域取得 packed E8M0 bytes，并把它们作为 FP4 weight 对应的 block scale。

## 7. Prologue 与 steady-loop 两种流水阶段

### 7.1 Prologue：加载 K tile 0

前四条静态 tensor load 分散在 descriptor/address 初始化代码中：

```asm
tensor_load_to_lds s[44:47], s[16:23]  // A0
tensor_load_to_lds s[4:7],   s[24:31]  // B0
tensor_load_to_lds s[44:47], s[36:43]  // SA0
tensor_load_to_lds s[80:83], s[4:11]   // SB0
```

它们共同完成：

```text
buffer0 = { A(K tile 0), B(K tile 0), SA(K tile 0), SB(K tile 0) }
```

此时 buffer1 还未填充。

### 7.2 Steady loop：预取下一个 K tile

循环中的四条静态 site 为：

```asm
tensor_load_to_lds s[44:47], s[16:23]  // next A
tensor_load_to_lds s[60:63], s[24:31]  // next B
tensor_load_to_lds s[44:47], s[36:43]  // next SA
tensor_load_to_lds s[60:63], s[4:11]   // next SB
```

它们在计算当前 K tile 时，把下一块数据加载到另一个 LDS slot。两个 slot 的间距为：

```text
PITCH = 0x9a00 = 39424 bytes
```

形成如下双缓冲：

```text
计算 buffer0 中的 tile kt
    同时 TDM 填充 buffer1 中的 tile kt+1

计算 buffer1 中的 tile kt+1
    同时 TDM 填充 buffer0 中的 tile kt+2
```

本 kernel 参数为：

```text
K            = 7168
tile_k       = 256
K_TILES      = 7168 / 256 = 28
num_buffers  = 2
```

因此：

```text
Prologue:     预取 tile 0
Steady loop:  27 次，每次预取下一个 tile
Drain:        计算最后一个已经预取的 tile，不再发出 tensor load
```

对于有效 workgroup，每个 wave 动态执行：

```text
4 条 prologue load + 27 × 4 条 loop load
= 112 条 tensor_load_to_lds
```

四个 wave 合计为 448 个 wave-instruction instance。这里的动态计数是 wave 指令实例数，不等同于 TDM 内部拆分出的 memory transaction 数。

## 8. 完成跟踪和同步语义

### 8.1 `TENSORcnt`

每发出一条 `tensor_load_to_lds`，对应的 tensor operation 由 `TENSORcnt` 跟踪。完成等待使用：

```asm
s_wait_tensorcnt 0x0
```

该等待保证当前 wave 之前提交的 tensor load 已完成对 LDS 的写入。

同一个 wave 的 tensor load/store 相互保序，但需要注意：

- 不同 wave 的 tensor operation 不互相排序；
- tensor operation 与普通 VMEM/SMEM/DS 指令不自动排序；
- `s_wait_loadcnt`、`s_wait_dscnt` 不能代替 `s_wait_tensorcnt`。

### 8.2 为什么还需要 workgroup barrier

四个 wave 分别提交同一种 tensor 的不同 segment。某个 wave 的 `s_wait_tensorcnt 0` 只保证该 wave 自己提交的 TDM 已完成，不能单独证明另外三个 wave 的 segment 已完成。

因此消费 LDS tile 前还需要：

```asm
s_wait_tensorcnt 0x0
s_barrier_signal -1
s_barrier_wait -1
```

其含义为：

1. 每个 wave 等待自己的 A/B/SA/SB tensor load 完成；
2. 四个 wave 在 workgroup barrier 汇合；
3. barrier 之后，完整的协作式 LDS tile 才能被所有 wave 安全读取。

### 8.3 与计算重叠

`tensor_load_to_lds` 是异步 TDM 操作。提交后 wave 可以继续执行 SALU、VALU、DS load 和 WMMA，而 TDM 在后台搬运下一 K tile。

steady-loop 特意把 next-tile 的四条 tensor load 放在当前 tile 的计算中间，使：

```text
下一 tile 的 GM→LDS 延迟
与
当前 tile 的 LDS→VGPR + WMMAScale
```

尽可能重叠。

## 9. 本文件没有使用的 tensor load 变体

虽然 CDNA5 TDM descriptor 还支持其他能力，但 `gemm1.v0.s` 中这 8 条 load 都没有使用：

- gather row-index 模式；
- descriptor iteration；
- cluster multicast/workgroup mask；
- TDM 完成后的 LDS atomic-barrier arrive；
- 3D、4D 或 5D tensor descriptor；
- tensor load instruction clause。

所以本文件中的差异完全来自四套二维 tensor 几何、数据格式、地址和 LDS 布局，而不是四种不同 opcode。

## 10. 总结

`gemm1.v0.s` 中的 `tensor_load_to_lds` 可以概括为：

1. **A load**：加载 `[16,256]` FP8 activation，处理 M-tail OOB，并在 LDS 每行增加 16-byte padding；
2. **B load**：加载 `[16,2048]` packed/preshuffled FP4 weight，无 padding；
3. **SA load**：加载 `[16,2]` i32，其中每个 i32 打包四个 activation E8M0 scale；
4. **SB load**：加载 `[8,64]` i32 的 n32k4 preshuffled weight scale；
5. 每种数据各有 prologue 和 steady-loop 两个静态 site，共 8 条；
6. 四个 wave 分段协作搬运，每个 wave 动态执行 112 条 tensor load；
7. `s_wait_tensorcnt` 保证单 wave TDM 完成，workgroup barrier 保证四个 wave 的 LDS 分片全部可见；
8. 双缓冲使下一 K tile 的 GM→LDS 搬运与当前 tile 的 WMMAScale 计算重叠。
