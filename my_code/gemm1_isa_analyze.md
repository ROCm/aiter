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

# `gemm1.v0.s` ISA 分析：`s_delay_alu` 与 `s_wait_alu`

## 1. 统计口径与证据基线

本节计数基于当前 `my_code/gemm1.v0.s`，ISA 行号也是该版本的实际文本行号。扫描时该文件的 SHA-256 为：

```text
5d0259d1f75d1011cd00a11bf1b6871911990adb3b6185ddb8846a673f038390
```

这里把“唯一形式”定义为指令助记符之后的符号 operand 组合完全相同；对本文件而言，每个不同组合对应不同的 `SIMM16` 字段组合。注释中的指令名不计数。一次性 Python 扫描脚本使用锚定到真实指令行开头的正则，并以 `Counter` 独立统计总数、唯一文本和行号：

```python
from collections import Counter
from pathlib import Path
import re

p = Path("my_code/gemm1.v0.s")
rows = [
    (n, line.strip())
    for n, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1)
    if re.match(r"^\s*s_(?:delay|wait)_alu\b", line)
]
for op in ("s_delay_alu", "s_wait_alu"):
    selected = [(n, text) for n, text in rows if text.startswith(op)]
    counts = Counter(text for _, text in selected)
    print(op, "total", len(selected), "unique", len(counts))
    for text, count in counts.items():
        print(count, [n for n, current in selected if current == text], text)
```

脚本结果与第二次 `rg` 锚定扫描一致：

| 指令 | 静态总数 | 唯一 operand/编码形式 | 本节采用的互斥语义场景 |
|---|---:|---:|---:|
| `s_delay_alu` | 90 | 34 | 7 |
| `s_wait_alu` | 30 | 7 | 4 |
| 合计 | 120 | 41 | 11 |

下文用“**文档事实**”表示直接由硬件文档给出的语义，用“**数据流推断**”表示根据当前 ISA 的附近生产者、消费者和控制流得到的解释。后者不冒充硬件规范。

## 2. 查阅的本地硬件资料

### 2.1 MI400 Shader Programming Guide

文件：

```text
C:\Users\yanguahe\Documents\code\llm-wiki\mi400_hw_wiki\raw\papers\mi400_hd_txt\architecture\subsystem\SH\MI400_Shader_Programming#65.txt
```

直接使用的章节和文本行如下：

| 章节 | 文本页标 | 文件行 | 用途 |
|---|---:|---:|---|
| 3.4.9 Scheduling Mode | Page 52 | 2910-2932 | `DEP_MODE=2`、`VA_VDST`/`VM_VSRC` 检查关闭，以及 XDL stall bit 的争议说明 |
| 4.3.7.1 VALU Ordering | Page 95 | 5532-5547 | 不同 VALU pipeline 的完成顺序 |
| 4.3.7.2 Memory Dependency Counters | Page 95-97 | 5548-5680 | `DScnt`、`TENSORcnt`、`VA_VDST`、`VM_VSRC` 的增减条件 |
| 4.3.7.4 Expert Scheduling Mode | Page 100-102 | 5826-6000 | `S_WAIT_ALU` 阈值、mode 2 暴露的 hazard、非零阈值限制 |
| 4.3.8 ALU Instruction Software Scheduling | Page 102-104 | 6001-6104 | `S_DELAY_ALU`、`instid`、`instskip`、TRANS/XDL scoreboard |
| 5.3.1 Hardware Enforced Interlocks | Page 224-226 | 15196-15341 | ALU scoreboard 与 instruction dependency counter 的边界 |
| 5.3.3 Instruction Co-issue Scheduling | Page 228-229 | 15443-15481 | `S_DELAY_ALU`/`S_WAIT_ALU` co-issue |
| 5.3.4-5.3.5 SALU/VALU Scheduling | Page 229-230 | 15506-15565 | SALU forwarding、VALU pipeline 深度与 fast forwarding |

### 2.2 CDNA5 ISA

文件：

```text
C:\Users\yanguahe\Documents\hardware_file\MI450\amd-instinct-cdna5-instruction-set-architecture.txt
```

直接使用的章节和文本行如下：

| 章节 | 文本页标 | 文件行 | 用途 |
|---|---:|---:|---|
| 5.7 Data Dependency Resolution | Page 65 | 3607-3614 | `DISABLE_XDL_ARB_STALL` 和 WMMA back-to-back 发射 |
| 5.8 ALU Instruction Software Scheduling | Page 66-67 | 3620-3697 | delay 可零周期执行、`INSTID`/`SKIP`、SALU/VALU/TRANS/XDL 分类 |
| 15.5 SOPP Instructions: `S_DELAY_ALU` | Page 292-294 | 19627-19720 | `SIMM16` 位段、所有实际 operand code、正确性定位 |
| 5.3 Instruction Clauses | Page 52 | 2884-2916 | clause 内 `S_DELAY_ALU` 的限制 |

### 2.3 MI400 corpus 补充交叉核对

为调查 `DISABLE_XDL_ARB_STALL` 的 bit 位置，还查阅了：

```text
C:\Users\yanguahe\Documents\code\llm-wiki\mi400_hw_wiki\raw\papers\mi400_hd_txt\block\sq.txt
```

其中 2.24 `GFXIPARCH-935: Disable SQ ALU scheduling stall for WMMA via SCHED_MODE` 行 23159-23186 明确把 `DISABLE_VALU_ARB_STALL` 定义为 `SQ_WAVE_SCHED_MODE bit[2]`。

## 3. 两条指令解决的是不同问题

### 3.1 `s_delay_alu`：ALU scoreboard 的发射调度提示

**文档事实：**

- `SIMM16[3:0]` 是 `instid0`，`[6:4]` 是 `instskip`，`[10:7]` 是 `instid1`（Programming Guide 4.3.8，行 6024-6031；CDNA5 ISA 15.5，行 19630-19639）。
- `VALU_DEP_1..4` 从普通 VALU scoreboard 倒数引用第 1..4 条已发射 VALU；`TRANS32_DEP_1..3` 从 16/32-bit transcendental scoreboard 倒数引用。16/32-bit XDL/WMMA 也按 TRANS 项跟踪（Programming Guide 行 6049-6086；CDNA5 ISA 行 3667-3697）。
- `SALU_CYCLE_1..3` 的含义是为先前 SALU 工作请求 1..3 个周期的间隔，不是“倒数第 N 条 SALU”（Programming Guide 行 6065-6071、6092-6094；CDNA5 ISA 行 3683-3689）。
- `instid0` 作用于 delay 后的第一个目标；省略 `instskip` 而同时给出 `instid1` 时，两项都作用于同一个目标（`SAME`）。`NEXT` 让 `instid1` 作用于第二个目标；`SKIP_k` 在第一个目标后跳过 `k` 条指令，再绑定第二项依赖。
- 如果 scoreboard 已经 ready，`s_delay_alu` 可以零周期执行；MI400 允许它与前一条 SALU/VALU co-issue（Programming Guide 行 6022、15443-15479；CDNA5 ISA 行 3634）。

本文件实际出现的 operand 词汇可按下表解码：

| operand | 当前文件中的取值 | 含义 |
|---|---|---|
| `instid0(X)` | 每条 delay 都有 | 把依赖类型 `X` 绑定到 delay 后第一个目标 |
| `instid1(X)` | 可选 | 为同一个或更后的目标再编码一项依赖 |
| `VALU_DEP_N` | `N=1..4` | 目标相对普通 VALU scoreboard 中倒数第 `N` 个生产者 |
| `TRANS32_DEP_N` | `N=1,2`；`N=3` 合法但未出现 | 目标相对 TRANS32/XDL scoreboard 中倒数第 `N` 个生产者 |
| `SALU_CYCLE_N` | `N=1..3` | 请求相对先前 SALU 工作的 `N` cycle 间隔，不按生产者条数倒数 |
| 省略 `instskip` | 仅行 657 的双 `instid` | `SAME`：两项依赖都绑定第一个目标 |
| `instskip(NEXT)` | 有 | `instid1` 绑定紧接第一个目标之后的第二个目标 |
| `instskip(SKIP_k)` | `k=1..4` | 在第一个目标后跳过 `k` 条指令，再绑定 `instid1` |

最重要的正确性边界是：`s_delay_alu` 是避免依赖指令过早进入 ALU pipeline、改善 wave 间可调度性的提示。CDNA5 ISA 明确说省略它仍应保持功能正确，只可能让依赖指令在 ALU 内部停顿并损失性能（行 19706-19719）。特别是 `SALU_CYCLE_N` 只表达周期间隔，**不能当成生产者已经完成的通用证明**。本 kernel 的 mode 2 也没有关闭 SALU/VALU 自身的 scoreboard interlock。

### 3.2 `s_wait_alu`：显式等待跨 pipeline dependency counter

**文档事实：**

- `VA_VDST` 在 VALU 发射时增加，在结果写回 GPR 后减少，表示尚未完成的 VALU destination（Programming Guide 行 5623-5630、5917-5924）。
- `VM_VSRC` 在 global/buffer/image/flat/scratch/LDS 指令赢得发射时增加，在该指令读完所有源 VGPR 后减少（行 5641-5647、5925-5927）。
- `depctr_x(K)` 的条件是 `x <= K`。`K=0` 是 full wait；非零 `K` 是 partial wait，可保留至多 `K` 个更年轻的无关项。
- `VM_VSRC` 硬件 counter 是 4 bit，但 wait 字段只有 3 bit：0..6 是真实阈值，7 表示“不等待”。`VA_VDST` 是 5 bit，但 `s_wait_alu` 只能访问低 4 bit（行 5904-5907、5964-5966、5979-5990）。
- `s_wait_alu` 可以与前一条 SALU/SMEM、VALU 或 VMEM co-issue，并在 counter 达标时让下一条指令立即前进；它不能与前一条 `s_delay_alu` co-issue（行 15443-15481）。

因此，`VA_VDST` 解决当前文件中的“VALU 写 VGPR，随后 DS/VMEM 读该地址/数据或写同一 VGPR”的 RAW/WAW；`VM_VSRC` 解决“DS/VMEM 尚未读取地址或 store data VGPR，随后 VALU/另一条内存指令覆盖该 VGPR”的 WAR。两者都不表示内存访问本身已经返回或提交。

## 4. `s_delay_alu`：34 种唯一形式、90 条

### 4.1 七个互斥语义场景汇总

| 语义场景 | 指令数 | 当前文件中的用途 |
|---|---:|---|
| 纯 SALU cycle | 20 | SGPR/SCC/carry 生产者到 SALU 或 VALU 消费者的间隔 |
| SALU + TRANS32 双目标 | 2 | SGPR→RCP 与 RCP→`v_readfirstlane` 打包 |
| 普通 VALU→普通 VALU | 60 | 索引、地址、unpack、clamp、SiLU/EXP 周围的 ordinary VALU RAW 调度 |
| `VALU_DEP` 绑定 non-ALU | 1 | 行 788 绑定行 789 `tensor_load_to_lds`，按文档没有 ALU delay 作用 |
| SALU + 普通 VALU 双目标 | 5 | 一个编码同时覆盖两个不同 pipeline 的消费者 |
| 纯 TRANS32 | 1 | 两条 EXP→`v_ldexp` 链 |
| 普通 VALU + TRANS32 双目标 | 1 | 普通 VALU→add 与 EXP→`v_ldexp` 打包 |
| 合计 | 90 | 完整互斥分区；唯一 operand 形式仍为 34 种 |

### 4.2 场景一：纯 SALU cycle，20 条、5 种

- 7 次，行 104、119、187、210、237、650、918：`s_delay_alu instid0(SALU_CYCLE_1)`
- 1 次，行 111：`s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_2) | instid1(SALU_CYCLE_1)`
- 8 次，行 116、180、197、200、218、230、247、250：`s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)`
- 2 次，行 177、227：`s_delay_alu instid0(SALU_CYCLE_3) | instskip(NEXT) | instid1(SALU_CYCLE_3)`
- 2 次，行 205、255：`s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)`

**数据流推断：** 行 111 的两个目标可以直接从文本定位：行 112 的 `s_and_b32` 消费行 110 写出的 `s2`；跳过行 113、114 后，行 115 的 `s_ashr_i32` 消费行 114 新写的 `s3`。行 116 又分别约束行 117 和 118。行 650 是 SALU→VALU：行 649 的 `s_sub_co_ci_u32` 写 `s2`，行 651 的 `v_mul_lo_u32` 把 `s2` 当 scalar source。

```asm
// L110-L118
s_cselect_b32 s2, -1, 0
s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_2) | instid1(SALU_CYCLE_1)
s_and_b32 s2, s2, s4
s_sub_co_ci_u32 s2, s3, 0
s_add_co_i32 s3, s52, 15
s_ashr_i32 s4, s3, 31
s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)
s_lshr_b32 s4, s4, 28
s_add_co_i32 s4, s3, s4
```

这些都是“请求至少若干 SALU 周期间隔”，而不是完成 counter；如果自然间隔已经足够，delay 不停顿。

### 4.3 场景二：SALU + TRANS32，2 条、1 种

- 2 次，行 172、222：`s_delay_alu instid0(SALU_CYCLE_2) | instskip(SKIP_1) | instid1(TRANS32_DEP_1)`

两处是同构的整数除法辅助序列。以行 172 为例：

```asm
// L156-L175
s_cvt_f32_u32 s7, s6
...
s_delay_alu instid0(SALU_CYCLE_2) | instskip(SKIP_1) | instid1(TRANS32_DEP_1)
v_rcp_iflag_f32_e32 v2, s7
v_nop
v_readfirstlane_b32 s7, v2
```

**数据流推断：** `instid0` 给行 173 的 RCP 留出读取 `s7` 的 SALU 间隔；`SKIP_1` 跳过行 174，使 `instid1` 绑定行 175，等待最近的 TRANS32 RCP 结果 `v2` 可被 `v_readfirstlane_b32` 消费。行 222-225 对 `s6/v2` 重复同一模式。

### 4.4 场景三/四：纯普通 VALU operand，61 条、21 种

- 4 次，行 290、342、587、637：`s_delay_alu instid0(VALU_DEP_2)`
- 5 次，行 312、338、406、414、417：`s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)`
- 7 次，行 315、326、384、395、788、839、892：`s_delay_alu instid0(VALU_DEP_1)`
- 4 次，行 322、380、391、402：`s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)`
- 1 次，行 334：`s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_1) | instid1(VALU_DEP_1)`
- 1 次，行 350：`s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_2)`
- 2 次，行 429、1019：`s_delay_alu instid0(VALU_DEP_3)`
- 1 次，行 601：`s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_2) | instid1(VALU_DEP_3)`
- 11 次，行 606、1014、1059、1064、1083、1088、1107、1112、1131、1156、1161：`s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)`
- 1 次，行 614：`s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_2)`
- 2 次，行 667、1211：`s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_3) | instid1(VALU_DEP_4)`
- 7 次，行 673、980、989、1070、1094、1118、1137：`s_delay_alu instid0(VALU_DEP_4)`
- 3 次，行 782、785、1009：`s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_4)`
- 1 次，行 909：`s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_3)`
- 1 次，行 912：`s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_1) | instid1(VALU_DEP_3)`
- 2 次，行 962、999：`s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_1) | instid1(VALU_DEP_4)`
- 1 次，行 1027：`s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_2) | instid1(VALU_DEP_3)`
- 1 次，行 1032：`s_delay_alu instid0(VALU_DEP_3) | instskip(SKIP_2) | instid1(VALU_DEP_4)`
- 1 次，行 1037：`s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_2) | instid1(VALU_DEP_4)`
- 1 次，行 1042：`s_delay_alu instid0(VALU_DEP_4) | instskip(SKIP_3) | instid1(VALU_DEP_4)`
- 4 次，行 1054、1078、1102、1126：`s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_3)`

按目标类型再分，60 条真正绑定 ordinary VALU，构成场景三；行 788 的 1 条绑定 non-ALU，单列为场景四。这 61 条 operand 形式落在三个实际代码区域：

1. 行 290-429 的索引/地址构造。例如行 314 `v_add` 生产 `v3`，行 315 delay 后由行 316 `v_lshrrev` 消费；行 338 同时覆盖行 339 的 64-bit shift 和行 340 的下一步 64-bit shift。
2. 行 587-673、759-912 的 pointer、`v_readfirstlane`、LDS/WMMA 前后准备。例如行 601 把行 600 的 `v[2:3]` 链到行 602，并把另一项依赖绑定到行 605 的 `v_or`。行 788 是一个必须单独指出的例外：它的下一条目标行 789 是 `tensor_load_to_lds`，不是 ALU。按 Programming Guide 行 6036“delay 可应用于任意 opcode，但对 non-ALU 没有用途”，该实例不会证明行 790 的 `v102` 生产关系，应视为编译器留下的无效/零作用调度提示。
3. 行 962-1161、1211 的 clamp/SiLU 展开。大量 `VALU_DEP_3/4` 在 compare、cndmask、mul、add 之间交错，使普通 main-pipeline 工作能与 `v_exp_f32` 并行，而不把 EXP 错算进普通 VALU 的倒数编号。

例如：

```asm
// L1014-L1026
s_delay_alu instid0(VALU_DEP_4) | instskip(NEXT) | instid1(VALU_DEP_4)
v_cndmask_b32_e64 v50, 0, 0x42800000, s1
v_cndmask_b32_e64 v51, 0, 0x42800000, s2
v_cndmask_b32_e32 v22, s0, v23, vcc_lo
...
s_delay_alu instid0(VALU_DEP_3)
v_dual_add_f32 v38, v38, v50 :: v_dual_add_f32 v39, v39, v51
...
v_exp_f32_e32 v38, v38
v_exp_f32_e32 v39, v39
```

这里 `VALU_DEP_N` 的 `N` 按已发射普通 VALU 指令计数，不是源文件行距，也不是要固定等待 `N` 个 cycle。

### 4.5 场景五：SALU + 普通 VALU，5 条、5 种

- 1 次，行 631：`s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_2) | instid1(VALU_DEP_1)`
- 1 次，行 657：`s_delay_alu instid0(VALU_DEP_4) | instid1(SALU_CYCLE_1)`
- 1 次，行 661：`s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(SALU_CYCLE_1)`
- 1 次，行 759：`s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_4) | instid1(VALU_DEP_3)`
- 1 次，行 774：`s_delay_alu instid0(VALU_DEP_2) | instskip(SKIP_3) | instid1(SALU_CYCLE_1)`

行 631 的 `SALU_CYCLE_1` 绑定行 632 `s_add_co_i32`，`SKIP_2` 跳过行 633、634 后把 `VALU_DEP_1` 绑定行 635 `v_mul_u64`。行 657 没有显式 `instskip`，所以是唯一的 `SAME` 实例：行 658 的 `v_add_nc_u64` 同时依赖普通 VALU 产生的 `v[8:9]` 和 SALU 产生的 `s[4:5]`。

### 4.6 场景六：纯 TRANS32，1 条、1 种

- 1 次，行 1189：`s_delay_alu instid0(TRANS32_DEP_2) | instskip(SKIP_2) | instid1(TRANS32_DEP_1)`

```asm
// L1173-L1193
v_exp_f32_e32 v35, v35
...
s_delay_alu instid0(TRANS32_DEP_2) | instskip(SKIP_2) | instid1(TRANS32_DEP_1)
v_ldexp_f32 v34, v34, v52
v_cndmask_b32_e32 v18, s54, v22, vcc_lo
v_cmp_gt_f32_e32 vcc_lo, s54, v24
v_ldexp_f32 v35, v35, v53
```

**数据流推断：** 最近两条 TRANS 是行 1173 的 EXP `v35` 和行 1168 的 EXP `v34`。`TRANS32_DEP_2` 保护行 1190 对 `v34` 的消费，`SKIP_2` 把 `TRANS32_DEP_1` 绑定到行 1193 对 `v35` 的消费。

### 4.7 场景七：普通 VALU + TRANS32，1 条、1 种

- 1 次，行 1230：`s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(TRANS32_DEP_1)`

行 1231 的 `v_add_f32` 使用普通 VALU 链；跳过行 1232-1234 后，行 1235 的 `v_ldexp_f32 v31` 消费行 1216 的 `v_exp_f32 v31`。一条 delay 分别查询 main 和 TRANS scoreboard，避免让中间独立 SALU/VALU 无谓等待。

当前文件的 `TRANS32_DEP` 实例只实际指向 RCP/EXP；文档所说的“XDL/WMMA 也作为 TRANS 项”是架构能力，本文件没有一条可由附近数据流证明是在用 `TRANS32_DEP_N` 等待 WMMA。

## 5. `s_wait_alu`：7 种唯一形式、30 条

### 5.1 七种唯一文本

- 13 次，行 79、292、317、328、344、386、397、409、735、791、841、920、1308：`s_wait_alu depctr_va_vdst(0)`
- 11 次，行 311、324、336、353、382、393、404、426、521、875、933：`s_wait_alu depctr_vm_vsrc(0)`
- 1 次，行 375：`s_wait_alu depctr_va_vdst(3)`
- 2 次，行 421、433：`s_wait_alu depctr_va_vdst(1)`
- 1 次，行 729：`s_wait_alu depctr_vm_vsrc(5)`
- 1 次，行 731：`s_wait_alu depctr_vm_vsrc(4)`
- 1 次，行 867：`s_wait_alu depctr_vm_vsrc(6)`

### 5.2 四个互斥语义场景

| 语义场景 | 指令数 | 唯一形式数 | 达标条件 |
|---|---:|---:|---|
| `VA_VDST` full wait | 13 | 1 | 所有较早 VALU destination 已完成 |
| `VA_VDST` partial wait | 3 | 2 | `VA_VDST <= 1` 或 `<= 3` |
| `VM_VSRC` full wait | 11 | 1 | 所有较早 DS/VMEM 已读完源 VGPR |
| `VM_VSRC` partial wait | 3 | 3 | `VM_VSRC <= 4/5/6` |
| 合计 | 30 | 7 | 完整互斥分区 |

### 5.3 `VA_VDST(0)`：VALU 写回后才让 DS/VMEM 使用，13 条

最短的完整 RAW 链在 kernel 开头：

```asm
// L43, L79-L80
v_mov_b32_e32 v16, 0
...
s_wait_alu depctr_va_vdst(0)
global_load_b32 v1, v16, s[12:13] offset:768
```

行 24 已进入 mode 2，global load 不再由硬件自动等 `VA_VDST==0`。行 79 保证行 43 写出的地址 `v16` 已可供 VMEM 读取。

其余 full wait 的附近数据流是：

- 行 292：行 291 写 `v2`，行 293 把 `v2` 当 global 地址。
- 行 317、328、344：行 316 写 `v3`→行 318 读地址；行 327 写 `v6`→行 329；行 343 写 `v[8:9]`→行 345。
- 行 386、397、409：行 385/396 写 `v3`→行 387/398；行 408 写 `v4`→行 410。行 410 同时读、写 `v4`，兼有 RAW/WAW。
- 行 735：行 730、732 产生 `v95/v100/v101`，行 736-746 的 DS load 读取这些地址。
- 行 791：行 790 产生 `v102`，行 792-795 的 DS load 读取该地址。
- 行 841：行 830-840 产生 `v53/v96/v97/v98/v52`，行 842-857 的 DS load 使用这些地址。
- 行 920：行 915 产生 `v[50:51]`，行 922-925 的 global-load clause 使用该地址。
- 行 1308：行 1296-1304 写出 BF16 store data，行 1309-1312 的 DS store 读取这些 VGPR。

这些等待只证明 VALU destination 写回；例如行 735 之后的 DS load 是否已经返回仍由行 796 的 `s_wait_dscnt 0` 证明，而不是由行 735 证明。

### 5.4 `VA_VDST(K>0)`：为何非零阈值成立，3 条

**数据流推断：**

- 行 375 `depctr_va_vdst(3)`：行 357 的 `v_lshrrev` 生成 global load 行 376 所需的 `v3`；其后恰有行 358-360 三条独立 main-pipeline `v_add`。允许最多 3 项在途即可保留这三条年轻工作，同时要求更老的 `v3` 生产者完成。
- 行 421 `depctr_va_vdst(1)`：行 419 生成地址 `v3`，行 420 是唯一更年轻的独立 `v2` 写；`<=1` 足以让行 422 读取 `v3`。
- 行 433 `depctr_va_vdst(1)`：行 431 生成行 434 的地址兼 destination `v1`，行 432 的 compare 是唯一更年轻 VALU；`<=1` 允许 compare 留在途而要求 `v1` ready。

```asm
// L357-L376
v_lshrrev_b32_e32 v3, 1, v3
v_add_nc_u32_e32 v11, 0x12000, v6
v_add_nc_u32_e32 v12, 0x12800, v6
v_add_nc_u32_e32 v13, 0x13000, v6
s_wait_alu depctr_va_vdst(3)
global_load_b32 v4, v3, s[12:13] scale_offset
```

这三个证明依赖于相关项和年轻项都在同一 ordinary VALU ordering class，且中间没有会按运行时 `EXEC` 跳过、从而改变计数的 VALU。Programming Guide 明确警告 TRANS、XDL 和普通 VALU 可彼此乱序完成，修改 `EXEC` 也会使非零阈值的静态位置失效（行 5921-5924、5967-5978）。因此非零值不是 cycle，也不能脱离这段具体顺序复用。

### 5.5 `VM_VSRC(0)`：源地址/数据读完后才覆盖，11 条

行 311 是典型的 global-address WAR：

```asm
// L293, L311-L313
global_load_b32 v3, v2, s[12:13] scale_offset
...
s_wait_alu depctr_vm_vsrc(0)
s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)
v_dual_cndmask_b32 v2, v4, v2 :: v_dual_cndmask_b32 v1, v5, v1
```

global load 必须先锁存旧 `v2` 地址，之后行 313 才能覆盖 `v2`。同构站点为：

- 行 324：行 318 的 global load 读 `v3`，行 325 写 `v3`。
- 行 336：行 329 读地址 `v6`，行 337 写 `v[6:7]`。
- 行 353：行 345 读 `v[8:9]`，行 354/355 开始覆盖 `v8/v9`。
- 行 382、393、404：行 376/387/398 读 `v3`，行 383/394/405 覆盖 `v3`。
- 行 426：行 422 读 `v3`，行 427 写 `v[2:3]`。
- 行 521：此前行 435-473 的 DS store 读 `v[2:5]`，行 522 开始回收 `v3`；但行 477 已有 `s_wait_dscnt 0`，所以按文档“只有 DS pending 时 DScnt==0 也保证 VM_VSRC==0”，行 521 通常应已满足，是保守边界而非额外内存完成等待。
- 行 875：行 868-874 的 DS load 都读地址 `v53`，行 876 的 DS load destination 覆盖 `v52:v53`；这里 full wait 直接阻止 address WAR。
- 行 933：行 922-925 的 global-load clause 都读 `v[50:51]`，而行 931 只等到 `LOADcnt<=3`；行 934/935 将覆盖 `v51/v50`，所以行 933 的 source-read full wait 不能由 partial load completion 替代。

在行 311、324、336、353、382、393、404、426 前，附近已经有 `s_wait_loadcnt 0`。完成一个 load 必然晚于它读取地址，因此这些 `VM_VSRC(0)` 很可能零停顿；它们仍明确编码 mode 2 下的 WAR 边界，并可利用 co-issue 低成本执行。

### 5.6 `VM_VSRC(K>0)`：部分排空及当前文件的实际条件，3 条

- 行 729：`depctr_vm_vsrc(5)`，随后行 730 重写 `v95/v100`。
- 行 731：`depctr_vm_vsrc(4)`，随后行 732 重写 `v101`。
- 行 867：`depctr_vm_vsrc(6)`，随后开始行 868-874 的第二批 DS load。

**数据流推断：** 从循环寄存器角色看，上一轮行 801-808 用 `v95`、行 809-810 用 `v100`、行 811 用 `v101` 作为 DS 地址，下一轮行 730/732 重建这些地址，因此 5、4 是编译器按同一 LDS ordering group 给出的剩余项上限。不过当前控制流在行 816 已执行 `s_wait_dscnt 0`，才由行 827 回跳到行 709；首次进入循环前也在行 477 清空了 DScnt。依据 Programming Guide 行 5931-5937，这意味着若 pending source 都来自 DS，则这些旧地址读已经完成。所以当前路径上的行 729/731 很可能一开始就达标，不应声称它们一定发生了实际 stall。

行 867 更明显是轻量 partial guard：行 858 已清空 DScnt，之后到行 867 之前只新发出行 859 一条 DS load，因此仅从附近静态序列看 `VM_VSRC<=6` 必然成立。没有证据把它归因于一个必须等待的同名 VGPR WAR；更可靠的结论是它保留了编译器的 mode-2 counter bookkeeping，并通常零停顿。

Programming Guide 行 5982-5988 规定 `VM_VSRC` 只在各自 ordering group 内保序：LDS/Flat 是一组，Flat/Scratch/Global/Buffer/Image 是另一组。跨组存在多个依赖时，必须针对各组相对位置取能覆盖全部依赖的最小阈值；因此不能把本文件的 4/5/6 搬到另一段 VMEM 流水中。

## 6. 与 kernel 开头 `SCHED_MODE` 的关系

当前 ISA 行 24：

```asm
s_setreg_imm32_b32 hwreg(HW_REG_WAVE_SCHED_MODE, 0, 2), 2
```

把 `DEP_MODE[1:0]` 设为 2。Programming Guide 3.4.9 行 2917-2923 和 4.3.7.4 行 5850-5859、5931-5961 一致说明：mode 2 **只关闭**发射前对 `VA_VDST` 和 `VM_VSRC` 的硬件检查，其余依赖机制保留。正常 mode 0 会用不比较实际寄存器号的 conservative counter check 自动挡住这些跨 pipeline hazard；mode 2 为了消除 false dependency stall，把责任交给软件。

所以本 kernel 需要显式 `s_wait_alu` 或等价的完成等待：

- VALU 写地址/store data 后，DS/VMEM 发射前等 `VA_VDST`；
- DS/VMEM 尚未读完地址/store data 时，后续覆盖 VGPR 前等 `VM_VSRC`；
- 若已知所有 pending 都属于 DS，`s_wait_dscnt 0` 可以提供更强的 `VM_VSRC==0` 保证；但它不能替代 `VA_VDST`。

这不意味着所有依赖都变成软件负责。Programming Guide 行 5865-5867 明确说 `SA_M0`、`SA_EXEC`、`VA_EXEC` 无论 `SCHED_MODE` 如何都由硬件检查；mode 2 也没有关闭普通 SALU/VALU scoreboard。故本文件中的 `s_delay_alu` 仍是性能调度提示，而 `s_wait_alu` 才是 mode 2 下这两类跨 pipeline 正确性 interlock。

当前 ISA 行 63 还单独写：

```asm
s_setreg_imm32_b32 hwreg(HW_REG_WAVE_SCHED_MODE, 4, 1), 1
```

它意图关闭 WMMA 后的多周期 arbiter stall，让一条 wave 可连续发出约 5 条 WMMA，再执行独立指令。该位改变 co-execution/发射节奏，不关闭 `VA_VDST`/`VM_VSRC`，也不让 WMMA 结果立即 ready。由于 XDL/WMMA 可与普通 VALU 乱序完成，它反而是使用非零 `VA_VDST` 阈值时必须谨慎区分 pipeline 的原因之一。

## 7. 与 `s_wait_dscnt`、`s_wait_tensorcnt` 的边界

| wait | 等待对象 | 能证明什么 | 不能证明什么 |
|---|---|---|---|
| `s_wait_alu depctr_va_vdst(K)` | VALU destination counter | 足够老的 VALU 已写回 GPR | global/LDS/TDM 已完成 |
| `s_wait_alu depctr_vm_vsrc(K)` | DS/VMEM source-read counter | 足够老的内存指令已读完地址/store data VGPR | load 已返回、store 已提交 |
| `s_wait_dscnt K` | LDS/Flat 的 DS completion counter | DS load return 或 DS store/atomic 完成到阈值 | 普通 global load、TDM 或较早 VALU 写回 |
| `s_wait_tensorcnt K` | TDM operation counter | 本 wave 发出的 tensor DMA 完成到阈值 | 普通 DS/VMEM、其他 wave 的 TDM、VALU counter |

Programming Guide 行 5588-5597、5802-5805 定义 `DScnt` 在 LDS 指令发射时增加，load/atomic return 或 store 写入 LDS 时减少；这是比“已读源 VGPR”更晚的完成点。行 5618-5622、14100-14104 定义 `TENSORcnt` 独立跟踪 TDM，且 tensor 与其他 memory type、其他 wave 不排序。

当前文件能看到三者明确分工：

- 行 735 `VA_VDST(0)` 让行 730/732 的地址写回，行 736-746 才发 DS load；行 796 的 `s_wait_dscnt 0` 再等这些 load 返回。
- 行 875 `VM_VSRC(0)` 只保证行 868-874 已读旧 `v53`，行 883 的 `s_wait_dscnt 0` 才保证新一批 DS load 返回供 WMMA 使用。
- 行 711、835、889、1317 的 `s_wait_tensorcnt 0` 等 TDM；它们不能替代 `VA_VDST` 或普通 DS completion。跨 wave 的 LDS tile 可见性还需相邻 workgroup barrier。

## 8. 文档冲突与采用原则

### 8.1 `DISABLE_XDL_ARB_STALL` 是 bit 2 还是 bit 4

- Programming Guide 3.4.9 的字段表写 bit 2（行 2925-2930），紧接着又写“本应在 bit4，register spec 错误地称其为 `DISABLE_VALU_ARB_STALL`”（行 2931-2932）。
- 同一 Programming Guide 4.3.7.4.2 再次写 bit 2（行 5991-6000）。
- CDNA5 ISA 5.7 也写 bit 2（行 3607-3614）。
- MI400 corpus 的 `block\sq.txt` 2.24.1 实现说明同样写 bit 2（行 23173-23186）。
- 当前 `gemm1.v0.s` 行 63 实际写的是 bit 4。

因此，字段表、CDNA5 ISA 和补充实现说明三处证据都支持 bit 2，只有 Programming Guide 的一句勘误式备注指向 bit 4，而当前 kernel 也写 bit 4。仅凭现有文本仍不能证明目标 stepping 最终采用哪一位；若按多数文档证据审查，行 63 值得另行核验。本任务不修改 ISA，只记录“当前 kernel 写 bit 4，意图为 disable WMMA arb stall”，不把 bit 4 宣称为已无歧义证明。这个冲突不影响行 24 的 `DEP_MODE=2`，也不影响 30 条 `s_wait_alu` 的必要性分析。

### 8.2 `instskip` 是否计入特殊 control instruction

- Programming Guide 4.3.8 行 6029-6031 说所有类型指令都计入 SKIP。
- CDNA5 ISA 5.8 行 3641-3644 说所有类型都计入，但排除 `S_SET_VGPR_MSB` 和 `S_WAIT_ALU`。
- Programming Guide 另在行 2310 明说 `S_SET_VGPR_MSB` 会被 `S_DELAY_ALU` skip-count 计入。

这是直接冲突。当前 90 条 delay 的实际 skip window 内没有落入这些有争议的特殊指令，所以本节列出的第二目标行号不受影响；若以后重排代码，应以目标 stepping 的最终硬件/assembler 规范复核。

### 8.3 clause 内是否合法

- Programming Guide Page 79 行 4650-4670 的表说 `S_DELAY_ALU` 在 clause 内 legal 但 pointless。
- 同一文档 4.3.8 行 6037 说不应在 VALU clause 内使用。
- CDNA5 ISA 15.5 行 19720 明说在 `S_CLAUSE` 创建的 clause 内 illegal。

当前文件没有把任何 `s_delay_alu` 放在 active clause 内；例如行 920 的 `s_wait_alu` 位于行 921 `s_clause` 之前。因此该冲突不改变当前统计或数据流结论，保守规则是把 delay 放在 `s_clause` 前。

### 8.4 `VA_VDST` 是否全局按序减少

Programming Guide 行 5967 写“incremented and decremented in order”，但行 5532-5546、5627-5630、5921-5924 又明确说 core/side、TRANS、DP、XDL 可彼此乱序完成。采用更具体的 pipeline-ordering 描述：只在同一 completion class 内利用相对顺序，跨 TRANS/XDL/main 或可能改变 `EXEC` 时不从非零阈值推导某个特定生产者已完成。本文件三个非零 `VA_VDST` 站点都只跨同类普通 VALU。

## 9. 正确性结论

1. `s_delay_alu` 的 90 条、34 种形式主要是 ALU 发射调度信息；七个语义场景分别为 20/2/60/1/5/1/1 条，其中行 788 是绑定 non-ALU 的零作用实例。它可以零周期执行，`SALU_CYCLE_N` 不是完成保证。
2. `s_wait_alu` 的 30 条、7 种形式是 mode 2 下显式 dependency-counter wait；四个语义场景分别为 `VA full=13`、`VA partial=3`、`VM full=11`、`VM partial=3`。
3. `VA_VDST` 保护 VALU→DS/VMEM 的地址、数据和 destination RAW/WAW；`VM_VSRC` 保护 DS/VMEM 先读地址/data、后续覆盖的 WAR。
4. full wait 清空相关可见 counter；partial wait 只在已证明 ordering class、相对年龄和 `EXEC` 行为时安全。当前三个 partial `VM_VSRC` 站点从附近控制流看很可能已经达标，不能把它们描述成必然发生的实际 stall。
5. `s_wait_alu` 不等待内存完成；`DScnt`、`LOADcnt`、`TENSORcnt` 分别负责相应 memory operation 的完成语义。
