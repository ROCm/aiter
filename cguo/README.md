# gfx1250 naive grouped MXFP4 MoE GEMM

目标：把线上 kernel `aiter/ops/flydsl/kernels/mxfp4_preshuffle_gfx1250_tdm.py`
拆成一个可读的 FlyDSL naive 版本，再一步一步还原线上的优化，每一步都验证正确性并和线上对比速度。

## 文件

| 文件 | 作用 |
|---|---|
| `naive_grouped_gemm_mxfp4.py` | naive kernel：`build_naive_grouped_gemm`（编译期配置）、`preshuffle_weights`（权重转线上布局）、`naive_grouped_gemm_mxfp4`（host 入口） |
| `test_naive_grouped_gemm.py` | 正确性：和 torch fp32 参考对比，并检查专家之间的 padding 行没有被写 |
| `bench_vs_prod.py` | 和线上 kernel 对比速度：dsv41 Flash TP4 a4w4，按 `dsv41_flash_tuned_grouped_fmoe.csv` 的 tile / wave / num_buffers 配置 |

运行（容器内，在仓库根目录；不要在 `/tmp` 下运行，`/tmp/aiter` 会遮住 aiter 包）：

```bash
HIP_VISIBLE_DEVICES=0 python cguo/test_naive_grouped_gemm.py
HIP_VISIBLE_DEVICES=0 python cguo/bench_vs_prod.py --tokens 1 16 128 1024 8192 32768
```

shape：E=384，topk=6，gemm1 N=1152 K=5120，gemm2 N=5120 K=576；均衡路由（与 tuner 一致）。

## 已完成的优化

按实际完成顺序排列；编号沿用最初的路线图。

### step0：naive 基线
- 每个 lane 用 `buffer_load` 直接把 A / W / scale 从 global 读到 WMMA 寄存器布局，每个 workgroup 1 个 wave。
- 输出：每个 lane 直接 `buffer_store_b128` 写 bf16。
- 路由和线上一致：专家起点按 tile_m 对齐，`psum[e] = starts[e] + rows[e]`。
  - 找专家：lane 并行数 psum，`shuffle_xor` 求和，`readfirstlane` 广播。
  - grid 按最坏情况开，尾部没有活的 tile 由 `has_work` 提前退出。
- K 在 host 端补零到 tile_k 的倍数。

### step3：TDM 把 A / W / scale 从 global 搬到 LDS
- 每个操作数一个 TDM 描述符：`make_tiled_tdm_atom` + `tdm_partition` + `fx.copy`。
- 数据再用 `ds_load_b128` / `ds_load_b32` 从 LDS 读到寄存器。
- TDM 是 wave 级标量指令，`s_wait_tensorcnt` 只统计本 wave 发出的 copy，所以要再加 barrier 等其他 wave。
- gfx1250 有个 hazard：TDM 发出后紧跟分支不行，需要先插一条 `s_nop 0`。

### step2：m_warp × n_warp 个 wave
- 每个 wave 负责输出里的一块 `WARP_M x WARP_N`。
- 4 个 TDM copy 分给各个 wave（wave i 发第 i 个）。
- tile / wave / num_buffers 和线上 CSV 完全对齐。

### step4：多 buffer 流水
- LDS 里开 `NB = num_buffers` 个 stage，TDM 比 WMMA 提前 NB-1 个 K tile 发出。
- 分 prologue / steady / drain 三段；`s_wait_tensorcnt(COPIES * newer)` 只等最老的那个 tile。

### step4b：寄存器双缓冲
- 两组 fragment 寄存器：当前 K step 做 WMMA 的同时，用 `ds_load` 读下一个 K step。
- 跨 tile 时提前读下个 tile 的 step 0，每个 tile 只需一次 barrier。
- 效果几乎为 0，说明瓶颈不在这里。之后的 tile_k 探测发现真正的瓶颈是 64 B 的短行，于是做了 step1。

### step1：权重换成线上布局（(16,16) preshuffle + n32k4 scale）
- 用 `shuffle_weight(w, layout=(16, 16))`：16 行一组，组内 16 B 块号 c、行 r 的位置是 `(c*16 + r)*16`。
  - 一个 K tile 在每组内是一段连续的 `16 * tile_k/2` 字节，tile_k=128 时是 1 KB。
- 用 `shuffle_scale_n32k4(w_scale)`：32 行一组，每个 WMMA K step 128 B。
- TDM box 的变化（以 gemm2 T=16 为例；TDM 指令条数不变，变的是每行长度）：

  | 操作数 | step4b | step1 |
  |---|---|---|
  | W | 64 行 × 64 B | 4 行 × 1024 B |
  | W scale | 64 行 × 4 B | 2 行 × 128 B |

  TDM 按行发 burst，每行有固定开销，短于 128 B 的行跑不满带宽（见 `mega_moe_gfx1250/tdm_prims.py` 的 `TDM_ROW_FLOOR_BYTES`）。
- LDS → 寄存器：16 个 lane 读连续的 256 B，没有 bank 冲突。
- 效果：T=16 gemm1 78.4 → 25.0 us，权重读带宽 3.6 → 11.3 TB/s（线上 12.0 TB/s）。

### step5：epilogue 改为 LDS 暂存 + TDM store（对应线上约第 2774–2832 行）
1. 第一次 `fx.barrier()`：所有 wave 读完输入 stage 后，C 才能复用这块 LDS。
2. 每个 lane 用 `ds_store_b128` 把 8 个 bf16 写进 LDS。行距是 `tile_n + 16` 列，用来打散 bank（同线上 `STORE_PAD`）。
3. 第二次 `fx.barrier()`，然后每个 wave 发一条 TDM store（`num_warps=NUM_WAVES`，按行切分）。
   - 全局 view 是 `(rows_end - blk_m, tile_n)`，越界的 padding 行和 pad 列由硬件丢弃。
4. 最后 `s_wait_tensorcnt(0)`：TDM 读完 LDS 之后 wave 才能退出。
5. 在 epilogue 前加 `rocdl.sched_barrier(0)`。原先每行一个边界分支，天然隔开了调度区域；去掉分支后，调度器会把最后一个 K tile 和 epilogue 混在一起排：
   - 128×256×128 的 VGPR 418 → 550，每个 SIMD 从 2 个 wave 降到 1 个；
   - 128×384×256 涨到 1023 个 VGPR，并出现 scratch 溢出。

   加上 `sched_barrier` 后恢复原样。
- 效果：大 M 的 gemm2 从慢 2.7–2.8 倍降到 1.12–1.13 倍。

## 性能进度

单位 us，naive 版；最后两列是线上 kernel 和当前的 naive/线上比值（step5 那次 benchmark）。
step3 用的是 naive 自己扫出来的 tile，和线上不可比，所以表从 step2 开始。

| T | gemm | 配置 | step2 | step4 | step4b | step1 | step5 | 线上 | naive/线上 |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 1 | 16x64x256 w1x1 b2（线上 sk4） | 10.6 | 8.2 | 8.0 | 7.1 | 7.1 | 4.2 | 1.69 |
| 1 | 2 | 16x128x128 w1x2 b2 | 3.9 | 3.3 | 3.7 | 3.4 | 3.6 | 3.4 | 1.07 |
| 16 | 1 | 16x384x128 w1x2 b3 | 78.8 | 77.5 | 78.4 | 25.0 | 24.8 | 23.6 | 1.05 |
| 16 | 2 | 16x64x128 w1x1 b2 | 19.0 | 18.7 | 19.5 | 10.6 | 10.4 | 9.3 | 1.12 |
| 128 | 1 | 16x128x512 w1x2 b4 | 188.6 | 161.1 | 148.4 | 83.2 | 83.1 | 74.5 | 1.12 |
| 128 | 2 | 16x128x128 w1x2 b3 | 82.3 | 69.9 | 71.9 | 45.8 | 45.3 | 38.9 | 1.17 |
| 1024 | 1 | 32x128x256 w1x2 b2 | 179.0 | 164.3 | 162.3 | 91.1 | 90.3 | 81.1 | 1.11 |
| 1024 | 2 | 32x128x128 w1x2 b3 | 107.7 | 96.5 | 98.6 | 69.0 | 51.3 | 47.1 | 1.09 |
| 8192 | 1 | 128x128x256 w1x2 b2 | 328.5 | 313.7 | 322.9 | 239.1 | 203.7 | 129.5 | 1.57 |
| 8192 | 2 | 128x128x128 w1x2 b2 | 307.7 | 283.6 | 292.7 | 308.2 | 130.6 | 115.9 | 1.13 |
| 32768 | 1 | 128x384x256 w2x2 b4 | 1406.3 | 1003.8 | 966.7 | 566.2 | 428.0 | 377.1 | 1.13 |
| 32768 | 2 | 128x256x128 w2x2 b4 | 1050.7 | 891.6 | 893.1 | 928.8 | 366.3 | 327.5 | 1.12 |

## 过程中的发现

- **Occupancy 不是差距来源。** 用 clang `-Rpass-analysis=kernel-resource-usage` 对 dump 出的 IR 统计，naive 和线上每行的 VGPR、LDS、每个 CU 驻留的 WG 数几乎一样。
  - prefill 阶段每个 SIMD 只有 1–2 个 wave，T=1 / T=16 的 gemm1 填不满 256 个 CU。
  - 这些是静态值；实际 occupancy 需要用 rocprof 计数器测。
- **线上 K tail 越界读（K=576、tile_k=128）。** `add_tdm_loads` 的 view extent 用了 `inner * K_TILES`（`mxfp4_preshuffle_gfx1250_tdm.py` 第 1028 行），没有用按 K_TAIL 截断的长度。于是最后一个 K tile：
  - A 每行多读 32 B（读进下一行）；
  - W 每个 16 行组多读 512 B（读进下一组）。

  后果：最后一个专家有行且后面的内存未映射时会 GPU fault；专家有 ≥2 行时误差 1.4–2.9%。把 A 的 pitch 补大后误差恢复到 0.2%，由此确认原因。这是 #7655（6850fe1773）引入的回退；fused_moe 的 verify 仍能通过（rel_l2 2.6e-3 → 4.2e-3）。
  - `bench_vs_prod.py` 里用 `a_row_stride_bytes` 和权重尾部补零绕开了这个问题；线上代码没有改。

## Epilogue 分析

### 输出布局和 topk 归约
- GEMM 不做 topk 求和：每个路由在 `grouped_out` 里占一行，`grouped_out` 和 GEMM 输入一样用 contiguous-M 布局。
- 以 T=16、topk=6、E=384、均衡路由为例：
  - 96 个路由，专家 e 唯一的那一行在第 16e 行；
  - `contiguous_m = min(96 + 384·16 − 6, 96 + 96·15) = 1536` 行，`grouped_out` 是 1536 × 5120 × 2 B = 15.7 MB，有效的只有 96 行（6.25%）。
- padding 行只占地址空间，不产生读写：GEMM 写回时丢掉它们，后面的 reduce 也只读有效行。真正的浪费在计算上：每个 16 行的 M tile 只有 1 行有效。
- TP 路径：gemm2 之后单独跑 `flydsl_moe_gather_reduce`（`kernels/moe_gather_reduce.py`）。
  - 每个 workgroup 负责一个 token：`out[t] = Σ_k w[t,k] · grouped_out[topids_to_rows[t,k]]`，fp32 累加，不用 atomic，结果确定。
  - 开 `doweight_stage1` 时权重改在 gemm1 里乘，gather 权重全为 1。
- EP 路径：gemm2 的 epilogue 先乘路由权重再截断成 bf16，再用 TDM gather-store 直接写进目标 rank 的 combine buffer；combine 只做不加权求和。
- 为什么不在 GEMM 里求和：同一个 token 的 6 个结果来自 6 个专家、6 个不同的 workgroup，只能靠 atomic 或跨 WG 同步。
- 两个 benchmark 都只测 grouped GEMM，不含 reduce。

### 为什么先写 LDS 再用 TDM 写回
LDS 在这里起换布局的作用：把 WMMA 累加器布局（lane = 行，每个 lane 一次拿 8 列）换成行主序。主要收益在写回粒度。
以 T=32768 gemm2（128×256，w2×2）为例，每个 WG 输出 64 KB：

| | 每个 lane 直接 `buffer_store_b128` | LDS + TDM |
|---|---|---|
| 写全局的指令 | 4 wave × 32 条 | 4 条（每个 wave 1 条） |
| 内存系统看到的写段 | 2048 段 × 32 B（每条指令写 16 行各 32 B，行距 10 KB） | 128 行 × 512 B |
| 128 B cache line | 由 4 条不同指令拼满 | 一次写满 |

次要收益：
- 向量内存指令和地址计算少了，逐行边界分支（exec mask）也没有了；
- padding 行和 pad 列由 TDM 越界裁剪；
- TDM 是异步 DMA。

多走一步 LDS 代价小：
- 每个 CU 的 LDS 带宽远高于它能分到的 HBM 带宽；
- 行距补 16 列，`ds_store_b128` 达到 b128 写入的下限（不补是 16 路冲突）；
- C 复用输入 stage，只有 C 比 stage 大时 LDS 才变大（128×128：34 → 36 KB）。

实测：T=32768 gemm2 写回 2.01 GB，kernel 929 → 366 us，按写回量算的带宽下限 2.17 → 5.5 TB/s。
收益和"输出字节 / 计算量"成正比：

| 场景 | 效果 |
|---|---|
| gemm2（N=5120，K=576） | 受益最大 |
| gemm1（N=1152，K=5120） | 次之 |
| decode | 基本不变 |

### 代码分段（`naive_grouped_gemm_mxfp4.py` 第 349–391 行）
例子：128×256×128，w2×2，`WM_REP = WN_REP = 4`，`ACC_LEN = 16`，`C_PITCH = 272`。

1. **第 350–353 行，`sched_barrier(0)` + 第一次 barrier。**
   - 累加器布局：lane l 是第 l%16 行；值 v 是第 `(v//8)*16 + (l//16)*8 + v%8` 列。
   - `sched_barrier` 只约束编译器的调度，不生成指令。
   - barrier 保证所有 wave 读完输入 stage 后，C 才覆盖它：C 占 128×272×2 = 68 KB，落在 102 KB 的 stage 区开头。
2. **第 354–361 行，寄存器 → LDS。**
   - 每个 WMMA 块：fp32 → bf16（8 条 `v_cvt_pk_bf16_f32`）；按 f=0/1 各取 8 个值，`ds_store_b128` 写到 `(row*C_PITCH + col)*2`。
   - 例：wave 3、lane 21，`wm=1, wn=2, f=1` → row 85，col 216，字节偏移 46672。
   - 每个 wave 32 条 `ds_store_b128`。padding 行也照写，没有分支。
3. **第 362 行，第二次 barrier。** TDM 按行分给各 wave（wave w 负责第 32w–32w+31 行、全部 256 列），和计算时的 2×2 分块不同，所以必须等所有 wave 写完。
4. **第 364–374 行，全局 view。**
   - `bf16_ptr`：`ptr_arg` 传进来的指针只按 1 B 对齐，所以显式造一个 16 B 对齐的 bf16 指针。
   - view 形状是 `(rows_end - blk_m, tile_n)`，行数是运行时值：decode 时为 1，prefill 时 ≥128。
   - 列数 256 小于 LDS 行宽 272，所以 pad 列会被丢弃。
5. **第 375–379 行，TDM store atom。** box 为 `(tile_m, C_PITCH)`；LDS 布局是紧密的，不需要 TDM pad 字段；`num_warps=NUM_WAVES` 让每个 wave 负责 32 行。
6. **第 380–385 行，LDS view。** 从 LDS 偏移 0 开始，带一个长度为 1 的 stage 维，形状和读取路径的 `(box, NB)` 一致。
7. **第 386–389 行，`tdm_partition`。** 用 `wave` 选出本 wave 的 32 行：全局地址和 LDS 地址都按 32 行偏移，每个 wave 的有效行上限是 `max(0, rows_end − blk_m − 32·wave)`。
8. **第 390–391 行，发 store + 等待。**
   - 每个 wave 1 条 `tensor_store_from_lds`。
   - `s_wait_tensorcnt(0)`：TDM 读完 LDS 前 wave 不能退出。这条等待紧跟在 TDM 后面，也顺带满足了"TDM 后不能紧跟分支"的 hazard。

### 编译后的 TDM store 描述符（128×256 w2×2，从 LLVM IR 解出）

| 字段 | 值 | 含义 |
|---|---|---|
| data_size | `1<<16` | 元素 2 B（bf16） |
| tensor_dim0 | 256 | 全局每行有效列数，第 256–271 列丢弃 |
| tensor_dim1 | `max(0, rows_end − blk_m − 32·wave)`（运行时） | 越过专家有效行的行丢弃 |
| tile_dim0 | 272 | 每行从 LDS 读 `C_PITCH` 个元素 |
| tile_dim1 | 32 | 每个 wave 32 行 |
| stride | 5120 | 全局行距 N |
| LDS 地址 | `lds + wave × 8704` 个元素 | 32 行 × 272 |
| 全局地址 | `c + (blk_m·N + blk_n)·2 + wave·32·10240` B | |

每个 wave 的 epilogue：128 条 cvt、32 条 `ds_store_b128`、2 次 barrier、1 条 `tensor_store_from_lds`、1 条 `s_wait_tensorcnt`。

### epilogue 能否和最后的 WMMA 重叠
不需要等所有 WMMA 算完，硬约束只有两条：
1. 块 (wm, wn) 的 cvt 只依赖这一块最后一条 WMMA 的结果，由硬件按寄存器依赖等待；
2. 写 LDS 前所有 wave 的 `ds_load` 都要完成（第一次 barrier）。WMMA 只读寄存器，barrier 等的是 `ds_load`，不是 WMMA。

现在已有的重叠：
- `sched_barrier` 只决定发射顺序。WMMA 流水执行，barrier 不等它的结果。
- epilogue 处理块的顺序和 `mma_step` 一样，最早用到的 `acc[0][0]` 是最后一个 K step 里最早发出的 WMMA，`acc[3][3]` 留到最后。
- 这个配置每个 SIMD 有 2 个 wave，一个做 epilogue 时矩阵单元可以给另一个用。
- 真正串行的是两次 barrier（等 WG 内最慢的 wave）和最后的 `s_wait_tensorcnt`。

证据：不加 `sched_barrier` 时，编译器把最后的 WMMA 挪到 barrier 之后，和 epilogue 交错（ISA 里第一条 `ds_store` 在第 918 行，WMMA 排到第 1076 行），说明这样做是合法的。但它同时把最后一个 tile 的 `ds_load` 整批提前，VGPR 418 → 550，所以加了 `sched_barrier`。

更深的重叠，按改动从小到大：
1. **第一次 barrier 提前到最后一个 K step 之前。** 有寄存器双缓冲，最后一个 K step 的数据上一步已经读进寄存器，这一步没有 `ds_load`。再用 `sched_group_barrier` 控制交错，比如 WMMA(0,0)、WMMA(0,1)、块 (0,0) 的 cvt+store、WMMA(0,2)……既藏住 cvt/store，又控住 VGPR。收益上限约是每个 wave 128 条 cvt 加 32 条 `ds_store` 的时间。
2. **C 用独立的 LDS 区域，去掉第一次 barrier。** 128×256 会从 102 KB 涨到 170 KB，每个 CU 从 2 个 WG 降到 1 个，得不偿失；小 tile 几乎免费（16×64 的 C 只有 2.5 KB）。
3. **persistent kernel + 延迟等待（线上 `defer_c_wait`）。** C 放在最后一个 pipeline stage，TDM store 发出后不等，直接进入下一个 tile 的 prologue（线上第 543–545 行、第 2830 行，用于 persistent stage1 的量化输出）。

线上普通的非 persistent 路径和 naive 一样：主循环结束 → `stage_fence(0)`（第 2232–2234 行）→ 写 LDS → TDM store，也没做第 1 条那种交错。

## 后续优化方案

按预期收益和依据排序：

1. **A 的 LDS padding（线上 `LDS_PAD_A = 16`）。** A 在 LDS 里每行是一个 K tile，16 个 lane 读 16 行时：
   - tile_k=128（行距 64 B）：起始 bank 只有 4 种，4 路冲突；
   - tile_k=256（行距 128 B）：起始 bank 只有 2 种，8 路冲突。

   每行补 16 B（行距 80 B 或 144 B）后 16 个 lane 的起始 bank 互不相同。
   - 用 TDM 的 LDS 布局 stride 实现即可（`make_layout(box, (tile_bytes + 16, 1))`，atom 会自动生成 padInterval / padAmount）。
2. **kernel 内处理 K tail，去掉 host 端补零。** gemm2 的 K=576 补到 640 白白多了 11% 的 WMMA 和读取。
   - 最后一个 K tile 只做 `ceil(K_TAIL/128)` 个 WMMA K step，TDM 用正确的 extent 做越界保护，避免重蹈线上的越界读。
3. **split-K（线上 `k_waves`）。** T=1 的 gemm1 线上用 sk4，naive 还差 1.69 倍。
   - 每组 wave 算一段 K，部分和经 LDS 归约（线上的 `K_PARTIAL_B` / `GROUP_ARENA_B`）。
4. **T=8192 gemm1（1.57 倍）。** 先用 rocprof 定位，候选包括：
   - A scale 的 TDM 行太短：tile_k=256 时每行只有 8 B，远低于 128 B 下限（A 每行 128 B，刚好在下限）；
   - 线上的 A scale 预排布 `_grouped_a8w4_preshuffle_e8m0_scale`；
   - WMMA 和 ds_load 的交错调度（`sched_group_barrier`）。
5. **A / A scale 预排布。** 线上 `a_preshuffled`：gemm1 的 epilogue 直接按 K32/M16 块写出 fp4，A scale 用 m16k4 布局。这样 A 的 TDM 行变长，LDS 也不需要 padding。
6. **cluster + B multicast。** 同一专家的多个 M tile 共享同一块 W，prefill 阶段可以减少 L2/HBM 流量。
7. **融合 epilogue。** bias、SwiGLU / SiTUv2、gemm1 输出的 MX 量化（`stage1_quant_out`，可直接写成 gemm2 要的预排布 A）、gemm2 的 EP scatter。
8. **epilogue 和最后一个 K step 交错。** 把第一次 barrier 提前，用 `sched_group_barrier` 让 cvt/`ds_store` 和最后的 WMMA 交错（见上面「epilogue 能否和最后的 WMMA 重叠」）。收益有限，先测 T=8192 / 32768 的 gemm2。
9. **persistent kernel / tile 调度。** 解决小 M 填不满 GPU、大 M 尾轮不均的问题；顺带可以让 TDM store 和下一个 tile 的主循环重叠（线上 `defer_c_wait`）。
