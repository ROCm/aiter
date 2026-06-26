# flydsl vs asm gemm2 ATT 逐指令对比分析与优化建议

> 数据来源：
> - asm：`ui_output_agent_38647_dispatch_9347`（kernel `moe_2stage_down`，gfx942）
> - flydsl：`ui_output_agent_38725_dispatch_9358`（kernel `moe_gemm2_0`，gfx942）
>
> ATT `code.json` 字段：`ISA, _, LineNumber, Source, Codeobj, Vaddr, Hit, Latency, Stall, Idle`
> （Hit = 采样命中次数，Latency = 累计延迟，Stall = 累计停顿，Idle = 空闲）

本文用两份 ATT 的**逐指令 Stall/Latency 实测**定位 flydsl gemm2 慢在哪里、为什么，以及要追平/超过 asm 必须改什么。结论与之前静态 ISA 分析一致，但更精确：**flydsl 的时间不是花在 MFMA，而是花在(1) 访存等待没被隐藏 和 (2) 一个极其臃肿的 epilogue（bf16 RNE 转换 + 逐元素 scale gather）**。

---

## 一、两个 kernel 的"时间都花在哪"——按 Stall 归类

### asm `moe_2stage_down`（主循环 Hit≈7872）

| 类别 | 代表指令 / 行 | 单条 Latency 量级 | 条数 | 占比定性 |
|---|---|---|---|---|
| **输出写出** | `global_store_dwordx4 ... nt sc1`（475/489/503/517/704/718/732/746） | 0.5M–2.0M | 8 | 大 |
| **权重 W2 加载** | `buffer_load_dwordx4 a[..], v2, s[20:23]`（379/397/415/434/608/626/644/663…） | 0.7M–1.6M | ~10 | 大 |
| **barrier 同步** | `s_barrier`（459/688） | 1.7M–1.8M | 2 | 中 |
| MFMA | `v_mfma_f32_16x16x32_fp8_fp8` ×96 | 0.08M–0.15M | 96 | 分散、被良好重叠 |
| epilogue | `v_pk_mul_f32` + `v_perm_b32` | 0.03M–0.06M | ~64 | **小** |

**asm 特征**：计算（MFMA）与访存**重叠良好**（用 `s_waitcnt vmcnt(6)/vmcnt(4)` 部分等待，保持多笔 load 在飞）；瓶颈是不可避免的 HBM 写出 + 权重读 + 2 次 barrier。epilogue 极轻（`v_pk_mul_f32` 乘 scale + `v_perm_b32` 直接截断打包 bf16）。

### flydsl `moe_gemm2_0`（主体 Hit≈15160）

| 类别 | 代表指令 / 行 | 单条 Latency 量级 | 条数 | 占比定性 |
|---|---|---|---|---|
| **访存等待（最大）** | `s_waitcnt vmcnt(0)`（行10 **8.51M**、行160 **7.86M**、行45 5.48M、行41 2.11M）；`vmcnt(2)`（行132 5.40M、行191 5.12M） | 2M–8.5M | 多 | **极大** |
| **epilogue VALU（次大）** | `v_mul_f32`+`v_bfe_u32`+`v_add3_u32`+`v_or_b32`+`v_cmp_u_f32`+`v_cndmask_b32`（行340–735，数百条） | 0.06M–0.18M | **数百** | **极大** |
| **CShuffle LDS 写** | `ds_write_b16_d16_hi`（16-bit 半宽写，~64 条） | 0.14M–0.17M | ~64 | 中 |
| **scale gather** | 窄 `buffer_load_dword v.., s[36:39]`（271/284/293/307/…，逐元素带 cndmask 边界判定） | 0.11M–0.37M | 数十 | 中 |
| barrier | `s_barrier`（135/163/194/257/737） | 0.6M–2.4M | 5 | 中 |
| 输出原子 | `global_atomic_pk_add_bf16`（760/785/811/…） | 0.14M–0.61M | 16 | 中（非主因） |
| MFMA | `v_mfma_f32_16x16x32_fp8_fp8` ×~40 | 0.13M–0.32M | ~40 | 分散 |

**flydsl 特征**：
- `s_waitcnt vmcnt` 的累计 Stall **数千万 cycle**，远超其他一切——**访存延迟完全没被计算隐藏**（loads 发出后马上 `vmcnt(0)` 全清等待）。
- 启动处 `buffer_load_dword v34`（行8，载入排序后 token 数等控制量）紧跟 `s_waitcnt vmcnt(0)`（行10）= **8.51M**，是单点最大停顿——一个**串行化的控制值依赖加载**。
- epilogue 不是"乘个 scale 打包"那么简单，而是**逐元素**做：2 次 scale 乘（激活 scale `v35` + 行 scale `v1`）→ **bf16 RNE 舍入**（`v_bfe_u32`+`v_add3_u32`+`v_or_b32`+`v_cmp_u_f32`+`v_cndmask_b32`，5 条 VALU/元素）→ `ds_write_b16_d16_hi` 半宽写 LDS。数百条这种指令累计吞掉巨量时间。

---

## 二、根因对比（为什么 flydsl 慢 ~2×）

| 维度 | asm | flydsl | 差距来源 |
|---|---|---|---|
| **访存隐藏** | 多笔 dwordx4 在飞 + 部分 `vmcnt(N)` 等待，与 MFMA 重叠 | 频繁 `vmcnt(0)` 全清，启动还有 8.5M 串行控制加载 | **访存延迟未隐藏（最大头）** |
| **bf16 转换** | `v_perm_b32` 截断打包（1 op / 2 元素） | RNE 舍入 5 op/元素（bfe+add3+or+cmp+cndmask） | **epilogue VALU ~10×** |
| **scale 应用** | scale 预载入寄存器，`v_pk_mul_f32` 批量乘 | 逐元素窄 `buffer_load_dword` gather + cndmask 边界判定 + 2 次标量乘 | epilogue 访存 + VALU 膨胀 |
| **LDS epilogue** | `ds_write_b64`（64-bit） | `ds_write_b16_d16_hi`（16-bit 半宽，条数翻倍） | LDS 指令数 ×2+ |
| **barrier** | 2 次（~3.5M） | 5 次（~6M） | 流水级更多、同步更重 |
| **输出** | 非原子宽 store（2.4GB partial + 独立 reduce） | `global_atomic_pk_add_bf16`（2 bf16/次） | 已验证非主因 |

> 注意：与"步骤5"静态 ISA 结论相比，ATT 实测把**epilogue（尤其 bf16 RNE + scale gather）**的权重抬得更高——它和访存等待是 flydsl 的两个并列大头，而不是次要项。

---

## 三、要追平 / 超过 asm 的优化建议（按 ATT 实测收益排序）

### 🅰 改 epilogue 的 bf16 转换：RNE 5-op → 截断/硬件打包（收益极大、改动相对集中）

flydsl 当前每个输出元素用 5 条 VALU 做 RNE 舍入（`v_bfe_u32`+`v_add3_u32`+`v_or_b32`+`v_cmp_u_f32`+`v_cndmask_b32`），数百条累计是 epilogue 的主成本；asm 只用 `v_perm_b32` 取高 16 位（截断）。

- **建议**：在 flydsl gemm2 的 f32→bf16 epilogue 用 `v_cvt_pk_bf16_f32`（如目标 ISA 支持）或 `v_perm_b32` 截断打包，一次处理 2 个元素。即使保留 RNE，也应改用单指令打包路径，而非展开成 5 条 VALU。
- **预期**：epilogue VALU 量级直接砍数倍，这是 ATT 里仅次于访存等待的第二大块。

### 🅱 隐藏访存延迟：消除 `vmcnt(0)` 全清 + 加深预取（收益最大）

flydsl 的 `s_waitcnt vmcnt(0)` 累计停顿数千万 cycle，是头号开销。

1. **加深 A/B tile 预取（multi-buffer）**：像 asm 那样让多笔 `buffer_load_dwordx4` 在飞，主循环用 `s_waitcnt vmcnt(N)`（部分等待）而非 `vmcnt(0)`（全清），使 MFMA 与下一块 load 重叠。
2. **修掉启动串行加载（行8+行10 = 8.51M）**：把控制量（sorted token 数 / expert offset，当前 `buffer_load_dword v34` 后立即 `vmcnt(0)`）尽量改用标量 `s_load` 提前发射，并把它的等待与后续地址计算/setup 重叠，而不是一上来就全清等待。
3. 这一项对应 gfx942 的延迟隐藏机制：算术强度低（K=192）时必须靠足够多的在途访存 + 流水重叠来填满内存延迟。

### 🅲 scale 处理：从"逐元素 gather"改为"预载入批量乘"（收益大）

flydsl epilogue 里有数十条窄 `buffer_load_dword ... s[36:39]`（逐元素取行/专家 scale）+ cndmask 边界判定 + 2 次标量乘；asm 把 scale 预载入 VGPR（v6/v8/v10/v12/v13）后用 `v_pk_mul_f32` 批量乘。

- **建议**：把当前 tile 需要的 per-row / per-expert scale **一次性合并加载**（宽读或预载入 LDS/寄存器），epilogue 内改用 `v_pk_mul_f32` 成对乘，去掉逐元素窄 load 和 cndmask gating。
- **副作用**：同时减少窄访存（呼应 🅱），并消掉大量 cndmask/cmp VALU。

### 🅳 CShuffle LDS 写：`ds_write_b16_d16_hi`（16-bit）→ `ds_write_b64/b128`（收益中）

flydsl 用半宽 16-bit LDS 写，条数是 asm `ds_write_b64` 的 2 倍以上。打包成 bf16x2/x4 后用宽 ds 写，减半以上 LDS 指令与相应 `lgkmcnt` 等待。

### 🅴 减少 barrier：5 → 接近 asm 的 2（收益中）

flydsl 5 次 `s_barrier`（~6M）对应更多流水级。合并/减少 stage 数（与 🅱 的多 buffer 预取协同），降低同步停顿。

### 🅵 输出原子：维持（非主因）

ATT 证实 `global_atomic_pk_add_bf16` 累计仅 ~5M，远小于访存等待与 epilogue。保持 atomic（大 M 下 L2 友好），优先级最低。

---

## 四、优先级与预期

| 优化 | 对应 ATT 大头 | 预期收益 | 改动量 |
|---|---|---|---|
| 🅱 加深预取 / 去 `vmcnt(0)` 全清 / 修启动串行 load | `s_waitcnt vmcnt` 数千万 cycle | **最大** | 中（codegen 调度/预取） |
| 🅰 bf16 RNE 5-op → 单指令打包/截断 | epilogue VALU 数百条 | **大** | 中（epilogue codegen） |
| 🅲 scale 预载入批量乘，去逐元素 gather | epilogue 窄 load + cndmask | 大 | 中 |
| 🅳 LDS 写 16-bit → b64/b128 | `ds_write_b16_d16_hi` ~64 条 | 中 | 中 |
| 🅴 barrier 5→2 | `s_barrier` ~6M | 中 | 中~重 |
| 🅵 输出原子维持 | atomic ~5M | — | 无 |

> **一句话**：要让 flydsl gemm2 追平甚至超过 asm，核心是把 ATT 里两座大山铲掉——**(🅱) 用深预取 + 部分 vmcnt 等待把访存延迟藏进 MFMA**，**(🅰/🅲) 把臃肿的 epilogue（RNE 5-op 转换 + 逐元素 scale gather）压成 asm 那样的 `v_pk_mul_f32 + v_perm_b32` 紧凑形式**。这两项做到位，再叠加 🅳/🅴，flydsl 完全有条件达到甚至略超 asm（因为 asm 还背着 2.4GB partial 写 + 独立 reduce，而 flydsl 用 atomic 省掉了 reduce）。

---

## 五、按本文落地的改动与实测（2026-06-12）

目标 shape：token=32768, model_dim=4096, inter_dim=192, expert=193, topk=9, fp8/per_tensor, gfx942。配置 `t64x128_atomic_bnt0` + `block_m2=64`。基线 gemm2 ≈ 5781us。

### ✅ 🅰 已落地：bf16 epilogue RNE → 截断（收益验证为真）

- **改法**（`moe_gemm_2stage.py` 的 `write_row_to_lds`，新增 `_cvt_out`）：bf16 恰好是 f32 的高 16 位，所以 f32→bf16 用 `bitcast→ >>16 →trunci→bitcast`（~2 op，无 `v_cmp_u_f32`/`v_cndmask`）替代 MLIR `arith.truncf` 默认的 5-op RNE 舍入序列，等价于 asm 的 `v_perm_b32` 截断。
- **env 开关** `FLYDSL_MOE_STAGE2_BF16_TRUNC`（默认开，仅对 bf16 输出生效；f16 仍走 RNE）。
- **实测**：gemm2 **5781 → 5416us（-6.3%）**，e2e 12830 → 12462us。正确性 pass，cos 0.99999，**max_delta 0.01171875 不变**（截断未抬高误差，因数值本就 fp8 量级粗）。稳定可复现（5416~5502）。

> 印证了 ATT 的判断：RNE 转换确实在 epilogue 关键路径上（不是被 stall 隐藏的那部分 VALU）。这也修正了上一轮"删 sentinel 掩码无收益"的困惑——那些 `v_cmp_u_f32`/`v_cndmask` 多数是 RNE 舍入，不是 sentinel 掩码。

### ⏸ 🅲 scale 预载：本 shape 无收益（已验证）

per_tensor 下把逐行 `buffer_load(sx, ts2)` 换成入口一次性标量加载，上一轮已实测**无加速**（scale gather 不在关键路径）。保持现状。

### ⏸ 🅳 LDS 半宽写：结构性受限

CShuffle 映射下同一 thread 的 2 个 `ni` 写到相距 16 列的 LDS 位置（非相邻），无法直接打包成 b32/b64；要宽写需重做 CShuffle 的 acc→LDS 布局，风险大，暂缓。

### ❌ 🅱 访存延迟隐藏 / 提占用：经实验**证伪**（提占用对本 kernel 无效）

投入 🅱 时先查 `use_async_copy` 崩溃，并用硬件计数器把"提占用"假设逐条做实验，得到一个**决定性的负结果**。

#### B.1 async_copy 崩溃根因（已查清，权威）

- 复现：手工构造 `_async` kernel 名跑通，LLVM 后端崩溃：
  `ExpandIntegerOperand ... llvm.amdgcn.raw.ptr.buffer.load.lds ... load (s128) ... i32<16>` → `LLVM ERROR: Do not know how to expand this operator's operand!`
- 根因（LLVM 源码核实）：**gfx942 的 `buffer_load_lds` 单次只支持 1/2/4 字节传输**（gfx950 才加 12/16 字节）。flydsl 该路径发的是 16B（dwordx4）DMA，gfx942 无法 lower。
- 该 async 路径**只把 X（A2 激活，约 56MB）**搬到 LDS；W2（约 3.6GB，主流量）始终走寄存器加载，不经 async。
- gfx942 上要修需把 16B 拆成 4×4B `buffer_load_lds`，但 dword 版硬件按 4B 跨 lane 排布，**无法复现 read 端期望的 lane-major 16B LDS 布局**，必须连带改写**共享的热路径 LDS 读**（`lds_load_packs_k64`）——高风险。
- **已落地（卫生修复）**：在 `compile_moe_gemm2` 入口对 gfx942 + `use_async_copy` 直接抛 `ValueError`，把"后端崩溃"变成可读报错（tuner 本就不在 gfx942 枚举 async，此守卫仅防手工 override）。

#### B.2 占用率实验：**提占用对本 kernel 无效（核心负结果）**

gemm2 资源：`arch_vgpr=92`、`accum_vgpr=44`、LDS=16640B、256 线程/WG（=4 waves，分摊到 4 个 SIMD = 每 SIMD 1 wave/WG）。gfx942 每 SIMD 256 arch VGPR → 256/92 = **仅 2 waves/SIMD（≈25% 占用）**。表面看"提占用"是大抓手，但实验逐条证伪：

| 实验 | arch_vgpr | 理论 waves/SIMD | gemm2 us | 结论 |
|---|---|---|---|---|
| `waves_per_eu=2` | 92 | （限到更低） | 7146 | 占用太低，明显变慢 |
| **`waves_per_eu=3`（默认）** | 92 | 2 | **5418** | 基准 |
| `waves_per_eu=4` | 92 | 2（VGPR 卡住） | 5503 | 无改善（编译器无法在不 spill 下提占用） |
| `waves_per_eu=5/6` | 92 | 2 | 5418 | 与默认持平（hint 饱和） |
| `n_per_wave=16` | **64** | **4（占用翻倍）** | **6252** | **占用翻倍反而更慢！** |

> **决定性结论：`n_per_wave=16` 把 arch_vgpr 砍到 64、可达 4 waves/SIMD（占用翻倍），gemm2 却从 5418 变慢到 6252。** 说明本 kernel **不是占用受限**——提占用带来的延迟隐藏，远不及每 wave 计算效率损失（`num_acc_n` 4→2、数据复用下降、2× wave 争抢同一份 W2 带宽）。`waves_per_eu≥3` 早已饱和也印证这一点。

#### B.3 真正的瓶颈：W2 的 HBM 带宽/延迟（占用与 async 都治不了）

"21.8% 非-LDS 等待"是等 **W2 全局加载**（3.6GB，每个 expert block 都要重读，K=192 算术强度极低 → 本质 W2 带宽受限）。token=32768 时 grid 已有 8192 个 workgroup，GPU 早已塞满活；每 CU 再多并发 wave 只会**加剧对同一批 W2 字节的争抢**，并不能隐藏延迟（所以 B.2 提占用无效）。

- async_copy 只动 X（56MB），**碰不到 W2**；即使修好 gfx942 dword 版，其唯一收益是释放 X 的 ~8 个 VGPR 以提占用——而 B.2 已证明提占用无效。**故 async_copy 修复对本 shape 无价值，停止投入。**
- asm 之所以快，是用 VGPR=208 + 完全展开换极深流水，达成**更高有效带宽**（asm 也仅 ~36% 带宽利用）。这属 flydsl JIT 调度器的根本能力差距，非单点 codegen 可补。

> **本轮净结果**：① 🅰 bf16 截断把 gemm2 从 5781 拉到 ~5416us（-6.3%，默认开、正确性不变）；② 查清并卫生修复 async 崩溃（gfx942 ISA 限制，转为可读报错）；③ 用实验**证伪"提占用/async 能追平 asm"**——本 kernel 是 W2 HBM 带宽受限，占用与 async-X 均无效。距 asm 核（3162us）的剩余差距源于 JIT 调度器无法生成 asm 级深流水/高有效带宽，非配置或单点 codegen 可达。

#### B.4 W2 流量 / L2 复用 / tile 大小：全部实测，无进一步收益

继续沿"W2 带宽受限"找减流量的杠杆，逐条实测（默认配置 t64x128 atomic = gemm2 ~5416us 为基准）：

- **L2 命中率 = 55.6%**（TCC_HIT/(HIT+MISS)）：W2 在 L2 有部分复用（同 expert 的相邻 workgroup 共享），约 44% 落 HBM。
- **增大 tile_m 减 W2 重读**（tile 数 = padded_rows/tile_m，越大 W2 重读越少）：实测**全部更慢**——`t128x64`=7981、`t128x128`=9555、`t128x64 reduce`=8090。原因是 tile_m=128 的累加器 VGPR 暴涨，占用跌破 2 waves/SIMD，得不偿失。
- **减小 tile_m**（`t32x128`，更多 tile/更多 W2 读）：5398~5482，与 t64x128（稳定 5417）**基本持平**（噪声内）。说明 tile 数（W2 读次数）不是线性主导，L2 复用足够；真正约束是占用/效率平衡。
- **split-K=3**（`kb3`，把 K=192 拆 3×64，3× 并发 workgroup 想加强延迟隐藏）：13075，**大幅更差**——3× 输出原子写流量主导，且 grid 本就有 8192 WG 不缺并发。
- **persist（XCD 局部性持久 kernel）**：bf16/fp8 gemm2 路径未接线（kernel 名 MISS），不可用。

> **最终结论（已彻底搜索）**：gemm2 在 token=32768 这一 shape 上处于稳健局部最优 ~5416us。配置/结构层面的所有杠杆（tile_m、n_per_wave、waves_per_eu、split-K、async、persist）要么更差要么持平；唯一真实收益是 🅰 bf16 截断（-6.3%）。本 kernel 受 W2 HBM 有效带宽约束，而提占用/减 tile 流量都无法转化为加速（占用非瓶颈、L2 复用已足够）。要继续逼近 asm 核（3162us）只能靠 flydsl JIT 在 K 循环生成 asm 级深寄存器流水以提高**有效带宽**，属调度器能力，非 stage2 配置或单点 codegen 可达。

#### B.5 深流水改造（K 循环全 B 预取）：实现 + 实测**证伪**——W2 是 HBM 带宽饱和，非延迟受限

按"对齐 asm 深寄存器流水"的方向，实现了 **deep B prefetch**（env `FLYDSL_MOE_STAGE2_DEEP_B`，默认关）：把默认的"1-tile-ahead 软件流水"改为**把全部 K-tile 的 W2 load 一次性在 prologue 发射**（inter_dim=192/tile_k=64 = 3 个 tile，24 条 B load 全部前置），让所有 W2 全局加载尽量并发、asm 式地藏 HBM 延迟。代码用最小改动实现：prologue 构建 `_b_all_deep` 列表，主循环改为从列表取 B 而非循环内再发 `load_b_tile`，保留原 barrier/scheduler/ping-pong 结构。

实测（t64x128 atomic，正确性 pass cos 0.99999）：

| 指标 | 默认（1-ahead） | DEEP_B（全预取） |
|---|---|---|
| gemm2 us | 5416 | **5442（持平/微差）** |
| 非-LDS 访存等待 / wave | 21.8% | **26.8%（更差）** |
| L2 命中率 | 55.6% | 56.9%（几乎不变） |
| arch_vgpr | 92 | 88 |

> **决定性诊断：把全部 W2 load 前置，访存等待不降反升（21.8%→26.8%）、L2 命中几乎不变、性能持平。** 说明 W2 **不是 load-issue 延迟受限**，而是 **HBM 带宽饱和**——一次性灌 24 条 load 只会让 VMEM 队列排得更长（每请求等待更久），反不如交错的 1-ahead 预取把 HBM 管道喂得平稳。**更深的预取流水无法加速一个带宽饱和的 kernel。**

**这彻底关闭了"深流水追平 asm"的路径**：asm 核 3162us 的优势不可能来自更深的 B 预取（flydsl 已是 16B 宽读且 W2 已带宽饱和），只可能来自**减少 W2 的 HBM 流量**——即靠 XCD 感知的 workgroup 调度提高 expert 权重在 L2 的时间局部性（把同 expert 的 tile 排到同一 XCD 的 L2 上复用）。这属 grid 调度/持久化 kernel 的能力，flydsl 当前 gemm2 路径未暴露（persist 未接线），且 B.4 的 tile_m 实验已表明单纯改 tile 形状无法捕获该局部性。

> **DEEP_B 去留**：env 默认关闭、正确性已验、对本 shape 无益；保留为可选开关（K-tile 更多的 shape 上理论上可能有用），不影响默认路径。

#### 🔚 deep-pipeline 方向总结
deep-pipeline / 全 B 预取已实现并实测，**证伪**其对本 shape 的价值：非流水/延迟/占用受限。

---

## 六、重大修正：gemm2 是**输出写受限（90% HBM 是写）**，不是 W2 读受限

用 `TCC_EA0_RDREQ_DRAM` / `TCC_EA0_WRREQ_DRAM`（真正打到 HBM/DRAM 的读/写请求）测 gemm2：

| HBM DRAM 流量 | 占比 |
|---|---|
| **读（W2 等）** | **9.8%** |
| **写（输出累加）** | **90.2%** |

> **结论彻底修正**：W2 读大部分命中 L2（55.6%），真正打 HBM 的读只占 9.8%；**90% 的 HBM 流量是输出写**。所以前面所有围绕"W2 带宽 / XCD 局部性 / 预取"的方向都是次要的——**真正的瓶颈是输出的原子累加写**。

### 6.1 根因：gfx942 无原生 bf16 原子加 → CAS 重试环，被 topk=9 竞争放大

gemm2 atomic 模式把每个专家块的结果 `global atomicrmw fadd`（bf16）累加到 `out[token, model_dim]`。gfx942 **没有** `buffer_atomic_pk_add_bf16`，该 bf16 原子加被仿真为 **CAS 重试环**（load→unpack→add→pack→cmpxchg→失败重试）。topk=9 意味着**同一个 token 输出位置被 9 个不同专家块并发累加**，竞争激烈 → CAS 反复重试 → **写流量放大**，正是 90% 写的来源。

### 6.2 实证：reduce 模式（流式写、无 CAS）gemm2 便宜 ~730us

| 模式（均含 🅰 bf16 截断） | gemm2 us | +reduce kernel | e2e us | 正确性 |
|---|---|---|---|---|
| **atomic** t64x128（默认） | 5416 | — | 12463 | pass |
| reduce t64x128 | 5206 | +672 | 12936 | pass |
| **reduce t32x256** | **4684（-13.5%）** | +672 | 12427 | pass |

reduce 模式 gemm2 把结果**流式 store** 到 `[tokens, topk, model_dim]` 临时缓冲（每个 token-slot 唯一地址、无竞争、无 CAS），gemm2 直降到 4684us，**实证了 atomic CAS 放大约值 730us**。但 reduce 需要一个独立的 `topk_sum` 规约 kernel（~672us，读 2.4GB 临时缓冲求和），**e2e 与 atomic 基本持平（12427 vs 12463）**。

### 6.3 评估：后续可能的写优化方向与 ROI

1. **f32 原子累加 + 末尾转换**（中等工作量，需碰 stage2 外）：gfx942 有原生 `global_atomic_add_f32`（单指令、无 CAS 重试）。改为累加到 f32 scratch `[tokens, model_dim]` 再单遍转 bf16。避免 CAS 重试放大，且**无需 topk 规约**（f32 scratch 本身就是累加结果）。代价：f32 写字节翻倍（4.8GB vs bf16 基线 2.4GB）+ 转换 kernel（~150us）。**净收益取决于 CAS 放大倍数是否 >2×**；需实测。涉及 scratch 分配 + 转换 kernel，超出"只改 stage2"。
2. **切默认到 reduce t32x256**：gemm2 指标更低（4684），但 e2e 持平（reduce kernel 吃掉收益）。仅当下游单独计 gemm2 才有意义。
3. **维持 atomic + bf16 截断**（当前）：已落地 -6.3% 的实质收益，e2e 12463。

> **核心评估**：e2e 无论 atomic/reduce 都卡在同一量级，因为输出累加（2.4GB 量级 partial，无论哪条路）本身是带宽密集，且 flydsl 在读和写两侧都比手写 asm 达成更低的有效带宽（JIT vs asm 的固有差距）。

### 6.4 落地：选定 reduce t32x256（已写入 tuned 配置）

干净头对头（同轮，正确性 pass）：

| 配置 | gemm2 us | e2e us | cos |
|---|---|---|---|
| atomic t64x128（旧默认） | 5413 | 9915 | 0.99999 |
| **reduce t32x256（选定）** | **4684（-13.4%）** | **9878** | 1.00000 |

已将 `hy3_fp8_pertensor_tuned_fmoe.csv` 中 token=32768 行的 `kernelName2` 改为 `flydsl_moe2_afp8_wfp8_bf16_t32x256x64_reduce_bnt0`、`block_m2=32`，并验证 fused_moe 正确分发（gemm2 4684、e2e 9878、cos 1.0）。

### 6.5 f32 原子累加方案：经**流量核算证伪**（会更差，放弃）

之前提出"f32 原生原子免 CAS"作为减写流量的候选。**仔细核算流量后否定该方案**（这是我之前漏算的）：

| 写路径 | HBM 流量（输出处理） | 说明 |
|---|---|---|
| **reduce（选定）** | 2.4GB 流式写 + 2.4GB 规约读 ≈ **5GB** | partial=`[token,topk,model_dim]` bf16，流式、无竞争 |
| bf16 atomic | 6e8 个 half2 原子RMW ×8B ≈ 4.8GB + CAS 重试放大 | gfx942 仿真 CAS |
| **f32 atomic** | 1.2e9 个**标量**原子RMW ×8B ≈ **9.6GB** + 转换 | **f32 RMW 字节翻倍 + gfx942 无 pk-f32 原子→标量原子op数翻倍** |

> **结论：f32 原子方案会让输出 HBM 流量从 ~5GB 涨到 ~9.6GB（近 2×），必然回退，放弃。** gfx942 没有 packed-f32 原子加，f32 既翻倍字节又翻倍原子 op 数。**reduce 模式（流式写 + 规约）本就是流量最优的写路径**，已选定落地。

## 七、rocprof-compute（rocprofiler-compute 3.4）speed-of-light 分析（reduce t32x256）

对选定的 reduce t32x256 gemm2 做全量计数器采集（13 个 pass），关键 SOL：

| 维度 | 实测 | 含义 |
|---|---|---|
| **ALU 活跃**（VALU+MFMA busy / wave_cy） | **15.9%** | ALU 闲置 84% → 严重非计算受限 |
| **指令等待**（inst-wait / wave_cy） | **35.3%** | 三成以上周期在等访存 |
| VALU : MFMA | 9.6 : 1 | VALU 多但藏在 stall 下 |
| **L2 命中率** | **79.2%** | 比 atomic（55.6%）高很多——流式写不做 RMW 读回、不污染 L2 |
| 每 dispatch 读 / 写 | **265 MB / 2416 MB** | write:read = **9.1**，写仍占 90% |
| **达成 HBM 带宽** | **~514 GB/s（写）≈ 峰值 10%** | 2.4GB partial 写花 4.7ms，远低于峰值 5.3TB/s |
| VGPR / occupancy | VGPR=52、8 waves/WG | **不受 VGPR 限**（256/52=4 waves/SIMD 余量） |

**诊断结论**：
1. reduce gemm2 的 2.4GB/dispatch 写正是 reduce 模式的 partial 流式写（`[tokens*topk*model_dim]` bf16 = 2.4GB，精确吻合），它就是 90% 写流量的来源。
2. **kernel 严重访存停顿（ALU 仅 16% 忙、35% 在等），但达成 HBM 带宽只有峰值的 ~10%**——既没吃满计算也没吃满带宽，说明是**访存并行度/重叠不足**（s_waitcnt 串行化），即 JIT 调度达不到 asm 的访存重叠深度。这与前面"预取/占用"实验的结论一致，并用 SOL 量化了：**asm 同样搬 2.4GB 却只花 3162us（~760 GB/s，1.5× 于 flydsl），差距就在达成带宽**。
3. VGPR=52、占用有余量 → 占用不是瓶颈（再次印证提占用无用）。

> rocprof-compute 印证了全部前述结论：gemm2 既非计算受限、也非 VGPR/占用受限，而是**访存重叠不足导致达成带宽仅 ~10% 峰值**。这是 JIT codegen 相对手写 asm 的固有差距，无法靠 stage2 配置/单点改动消除。

---

## 八、store 合并度验证（rocprof-compute 计数器）+ 与 asm 对比

针对"低达成带宽是否因 store 未合并"做了直接验证（reduce t32x256，rocprof-compute pmc_perf）：

| 计数器 | 值 | 结论 |
|---|---|---|
| `TCC_EA0_WRREQ_sum` | 3.02e8 | 总写请求 |
| `TCC_EA0_WRREQ_64B_sum` | 3.02e8 | **64B 写占比 = 100%** |
| 32B（部分/未合并）写 | 0 | **完全合并，零浪费** |
| `TCC_TOO_MANY_EA_WRREQS_STALL` | 0 | 写队列不堵 |
| `RDREQ_32B` | 0 | 读也 100% 64B 合并 |

代码侧确认：`_store_nt = 2 if not accumulate else 0` → reduce 模式 `nontemporal=True`。**flydsl reduce 的 store = 100% 合并的 64B + non-temporal**。

与 asm 对比（asm store 来自 ISA：`global_store_dwordx4 ... nt sc1`）：

| | flydsl reduce | asm down |
|---|---|---|
| store 宽度/合并 | 64B 全合并 | dwordx4(16B/thread) 合并 |
| non-temporal | 是 | 是（nt） |
| 写 2.4GB partial 用时 | ~4684us | ~3162us |
| **达成写带宽** | **~514 GB/s（~10% 峰值）** | **~760 GB/s（~14% 峰值，1.5×）** |

> **结论：store 本身不是问题——flydsl 已是"全合并 64B + non-temporal"，与 asm 的 store 特性等价；写队列也不堵（STALL=0）。** 两者达成写带宽都远低于峰值（<15%），差距纯粹在**访存级并行度（MLP）/ 重叠深度**：asm 用更深的寄存器流水让更多 store 同时在飞，达成 1.5× 带宽。这再次落到"JIT 调度器达不到 asm 的重叠深度"这一固有差距，而非任何可调的 store 属性。

---

### 🔚 最终结论
stage2 内部所有写路径已穷尽：reduce t32x256 是 flydsl 侧的流量最优解（gemm2 4684、e2e 9878）。**追平 asm 核（3162us）在 flydsl gemm2 codegen 内不可达**——asm 同样走"流式写 partial + 规约"，但其 gemm2 流式写达成的有效带宽显著高于 flydsl（4684 vs 3162，1.48×），这是 JIT 调度 vs 手写 asm 的固有带宽差距，已被本文件多轮实验（预取/占用/tile/写模式）反复证实。要拿到 asm 性能，唯一确定路径是按 M 段把该 shape 分发到 asm（e2e 8602 vs 9878，且本身即 asm 性能）。
