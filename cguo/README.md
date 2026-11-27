# gfx1250 naive grouped MXFP4 MoE GEMM

目标：把线上 kernel `aiter/ops/flydsl/kernels/mxfp4_preshuffle_gfx1250_tdm.py`
拆成一个可读的 FlyDSL naive 版本，再一步一步还原线上的优化，每一步都验证正确性并和线上对比速度。

## 文件

| 文件 | 作用 |
|---|---|
| `naive_grouped_gemm_mxfp4.py` | naive kernel：`build_naive_grouped_gemm`（编译期配置）、`preshuffle_weights`（权重转线上布局）、`preshuffle_act_scale`（A scale 按 (M tile, K tile) 分块）、`naive_grouped_gemm_mxfp4`（host 入口） |
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
5. 在 epilogue 前加 `rocdl.sched_barrier(0)`。原先每行一个边界分支，天然隔开了调度区域。去掉分支后，主循环和 epilogue 成了同一个调度区域，调度器为了让 WMMA 和末尾的 cvt / `ds_store` 交错，把 WMMA 越推越晚。
   - WMMA 只碰寄存器，可以越过 barrier；`ds_load` 被 barrier 固定在原段不动。
   - 于是 `ds_load` 比 WMMA 超前的距离从 1 个 K tile 涨到 2.2 个，好几个 tile 的 fragment 同时占着寄存器。
   - 128×256×128 的 VGPR 416 → 550，每个 SIMD 从 2 个 wave 降到 1 个；128×384×256 涨到 1023 个 VGPR，并出现 scratch 溢出。

   加上 `sched_barrier` 后恢复原样。逐段数据见「epilogue 能否和最后的 WMMA 重叠」。
- 效果：大 M 的 gemm2 从慢 2.7–2.8 倍降到 1.12–1.13 倍。

### step6：A 的 LDS 行补 16 B（对应线上 `LDS_PAD_A = 16`）
- **问题**：A 在 LDS 里每行是一个 K tile，行距是 2 的幂（fp4 下 tile_k=128 为 64 B，256 为 128 B）。一条 `ds_load_b128` 里 16 个 lane 读 16 行，起始 bank 只有 1–4 种，因此有 4–16 路冲突。
- **做法**：`A_LDS_PITCH = A_TILE_BYTES + 16`，行距变成 80 B、144 B……，16 个 lane 的起始 bank 两两不同。
  - `tdm_k_tile` 加了一个 `lds_pitch` 参数：LDS 布局按 `(lds_pitch, 1)` 做 stride，atom 自动生成描述符的 pad 字段。
  - `load_act` 按新的行距读。W 已经是 preshuffle 布局，本来就没有冲突，scale 也不需要处理。
- **验证（只编译）**：A 描述符 group1 的第一个字段从 `0` 变成 `0x06D00000`，即 pad_enable=1，每 16 dword（64 B）补 4 dword（16 B）。VGPR 变化在 ±5 以内。每个 stage 的 LDS 多出 `tile_m × 16 B`，例如 T=8192 gemm1 从 68 涨到 72 KB，和线上相同。
- **效果**：收益和 A 在 LDS 读取中的占比一致。
  - T=8192 gemm1（w1×2，每个 wave 128 行）：每步 A 16 条、W 8 条 `ds_load_b128`，快了 14%（1.57 → 1.34 倍）。
  - T=32768 gemm1（w2×2）：A 8 条、W 24 条，快了 8.5%（1.13 → 1.04 倍）。
  - 大 M 的 gemm2 快 3–4%；decode 基本不变。

### step7：A scale 按 (M tile, K tile) 连续排布（对应线上 A scale 的 m16k4 预排布）
- **问题**：A scale 原来按行存（`[M, K/32]`），每个 K tile 的 TDM box 是 tile_m 行、每行只有 tile_k/32 B。例如 T=8192 gemm1 是 128 行 × 8 B，远低于 TDM 每行 128 B 的带宽下限。
  - 这个配置是 w1×2：wave 0 同时发 A（128 行）和 A scale（128 行）两个 copy，每个 tile 要处理的行数翻倍。
  - 它又只有 2 个 buffer，TDM 只能提前 1 个 tile，这部分延迟几乎全部暴露。
- **证据**（对照实验，见「剩余差距分析」）：线上改用按行存的 A scale（`row_major_ascale=1`）后，T=8192 gemm1 从 128.6 慢到 158.5 us（+23%）；其他 gemm1 行只慢 0–2%。
- **做法**：
  - 新增 `preshuffle_act_scale(a_scale, tile_m, tile_k)`：K 补到 tile_k 的倍数，排成 `[M/tile_m][Kp/tile_k][tile_m][tile_k/32]`，返回 4 维张量，用形状表明布局。和 `preshuffle_weights` 一样在 GEMM 之外做；实际流水线里由量化 kernel 直接写出这个布局，线上的 m16k4 布局也是同样的目的。
  - kernel：A scale 的 TDM 每个 M tile 只有一行（`AS_ROW_BYTES`），K tile kt 是这一行里第 kt 个连续块（`AS_BLOCK_BYTES = tile_m × tile_k/32`）。数据落到 LDS 里仍然是按行排列，`load_act_scale` 不用改。
  - host：a_scale 的形状里已经没有原始 K，所以改用 dtype 判断 a4w4 / a8w4（a8w4 要求传入 `torch.float8_e4m3fn`）。
- **验证（只编译）**：这个 box 只有一行，被编成一维 TDM 描述符（T=8192 gemm1：每个 K tile 一段 1024 B）；VGPR 不变；正确性 15/15。
- **效果**：
  - T=8192 gemm1：175.7 → 164.7 us（1.34 → 1.25 倍）；
  - T=8192 gemm2：125.3 → 122.1 us；
  - 其余行变化在 ±3% 以内，属于噪声。
  - 对照实验里同样布局的 T=8192 gemm1 测到 151.6 us，而 benchmark 单次测量是 162–165 us。两者的 ISA 只差 32 行 SGPR 写法（VGPR 都是 506）。后来在同一进程里交替跑了 3 轮，单位 us：

    | 行 | step6 | 对照实验 naive_as | step7 |
    |---|---|---|---|
    | T=8192 gemm1 | 175.7 / 180.4 / 177.5 | 162.6 / 160.3 / 152.3 | 160.2 / 158.3 / 155.9 |
    | T=8192 gemm2 | 127.6 / 127.7 / 129.8 | 126.6 / 124.9 / 127.4 | 125.4 / 125.6 / 126.3 |
    | T=32768 gemm1 | 392.8 / 391.8 / 390.9 | 389.1 / 390.1 / 388.5 | 388.8 / 388.3 / 389.7 |

    step7 和 naive_as 落在同一个范围里，是同一个 kernel。T=8192 gemm1 单次测量本身就会在 152–163 us 之间浮动，151.6 是对照实验里偏低的一次。按最小值算，step7 在这一行比 step6 快 11%（175.7 → 155.9）。

### step8：每个 wave 各发自己那一份 TDM，去掉归属分支和 `s_nop`
- **问题**：原来每个 copy 由一个 wave 发（`if wave == i % NUM_WAVES`），每个 copy 后面跟一条 `s_nop 0`，用来避开 gfx1250"TDM 后面不能紧跟分支"的 hazard。T=8192 gemm1 的主循环每个 K tile 因此有 6 个分支、3 条 `s_nop`，循环体也被切成了好几个基本块。
- **做法**：
  - `tdm_k_tile` 的 atom 用 `num_warps=NUM_WAVES`，`tdm_partition(atom, wave, make_layout(NUM_WAVES, 1), ...)`，每个操作数的 box 由所有 wave 分摊。
  - 每个 wave 每个 K tile 发 4 条 TDM（`COPIES = 4`），原来"wave 数要整除 4"的断言删掉。
  - `issue` 不再有分支，也不再插 `s_nop`。
  - 线上默认是每 2 个 wave 分一个 tensor（`num_waves_per_tensor_tdm=2`），4 个 wave 时分成两组，组之间仍有归属判断；naive 用最简单的"全部 wave 分"，完全没有分支。
- **验证（只编译）**：
  - 6 个配置的 ISA 里都没有"TDM 后紧跟分支或 label"，`s_nop` 为 0，每个 kernel 只剩 1–2 个分支（`has_work` 和循环回跳）。
  - 多 wave 配置的 VGPR 下降：128×256 w2×2 从 414 降到 378，a8w4 32×128 w1×2 从 186 降到 149。
  - T=8192 gemm1 主循环每个 K tile：分支 6 → 1，`s_nop` 3 → 0。循环体成了一个基本块，调度器自己把 `s_wait_dscnt` 从 4 拆到 10（线上 11）。VALU 从 16 变成 25，应该是每个 wave 都要算自己那份 TDM 的地址偏移。
  - 正确性 15/15。
- **效果**：naive 自身只有两行有明显变化：T=16 gemm1 25.1 → 23.3 us（−7%，追平线上），T=1 gemm1 7.3 → 7.0 us（−4%）。其余行在 ±2% 内。
  - 表里部分比值的改善来自线上这一次测得偏慢（例如 T=128 gemm2 线上 39.6 → 41.0 us）。
  - 等待粒度已经接近线上，T=8192 gemm1 仍差 1.23 倍，说明剩下的差距不在等待次数和分支上（后来 step9 找到了原因：workgroup 的发射顺序）。
- **测量方式**：这一列逐行测，每行开跑前确认没有别人的 GPU 进程，运行中每秒采样一次；出现别人的进程就作废这一行，等 GPU 空了重跑。

### step9：workgroup 按线上的分组 swizzle 发射（对应线上第 671–695 行，`TILES_PER_GROUP = 16`）
- **问题**：rocprofv3 ATT 显示，T=8192 gemm1 的 naive 有 59% 的时间卡在等 TDM，线上只有 24%。原因是 naive 的 grid 让共用同一个 A tile 的 9 个 workgroup 分散到不同的 XCD，而且在时间上相隔很远，A 在 L2 里留不住。分析见下面「Workgroup 发射顺序、XCD 与 L2」。
- **做法**：
  - grid 从二维的 (M tile 数, N tile 数) 改成一维的 M tile 数 × N tile 数。
  - kernel 开头用线上同样的公式把 block id 换算成 (m_tile, n_tile)：每 `GROUP_M = 16` 个 M tile 一组，组内 M 在内层、N 在外层，最后一组可能不满 16 个。
  - 新增 kernel 参数 `m_tiles`；A scale 的 TDM 改用换算后的 `m_tile`。
  - 共用一个 A tile 的 N 个 workgroup，编号是 j、j+16、…，`% 8` 都相同，所以落在同一个 XCD、同一个 L2 上，而且编号相邻。
- **验证**：正确性 15/15；6 个编译配置的 VGPR 不变。
- **效果**：同一进程里和 step8 交替跑 3 轮取最小值，单位 us：

  | 行 | step8 | step9 | 变化 |
  |---|---|---|---|
  | T=8192 gemm1 | 155.5 | **128.6** | **−17%** |
  | T=8192 gemm2 | 124.2 | 118.4 | −5% |
  | T=1024 gemm1 | 84.5 | 79.9 | −5% |
  | T=32768 gemm1 | 388.7 | 374.5 | −4% |
  | T=32768 gemm2 | 353.3 | 343.8 | −3% |
  | T=1024 gemm2 | 50.5 | 49.3 | −2% |
  | T=16 gemm2 | 10.1 | 10.4 | +3% |
  | 其余 5 行 | | | ±1% 以内 |

  - T=8192 gemm1 和线上的差距从 1.23 倍降到 1.02 倍（benchmark 单次测量：线上 133.1 us，naive 136.0 us）。
  - 收益大小和"A 有多大、被几个 N tile 复用"一致：gemm1 每行的 A 是 2.5 KB，复用 3–9 次，收益最大；gemm2 的 A 很小（每行 288 B），收益只有 2–5%。
  - T=16 gemm2 的 +3% 在测量噪声边缘：decode 阶段每个专家只有 1 行，W 没有复用，换顺序也不改变读取量。

## 性能进度

单位 us，naive 版；最后两列是线上 kernel 和当前的 naive/线上比值（step9 那次 benchmark）。
step3 用的是 naive 自己扫出来的 tile，和线上不可比，所以表从 step2 开始。

| T | gemm | 配置 | step2 | step4 | step4b | step1 | step5 | step6 | step7 | step8 | step9 | 线上 | naive/线上 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 1 | 16x64x256 w1x1 b2（线上 sk4） | 10.6 | 8.2 | 8.0 | 7.1 | 7.1 | 7.0 | 7.3 | 7.0 | 7.3 | 4.5 | 1.62 |
| 1 | 2 | 16x128x128 w1x2 b2 | 3.9 | 3.3 | 3.7 | 3.4 | 3.6 | 3.6 | 3.8 | 3.7 | 3.7 | 3.6 | 1.03 |
| 16 | 1 | 16x384x128 w1x2 b3 | 78.8 | 77.5 | 78.4 | 25.0 | 24.8 | 25.0 | 25.1 | 23.3 | 24.1 | 23.3 | 1.04 |
| 16 | 2 | 16x64x128 w1x1 b2 | 19.0 | 18.7 | 19.5 | 10.6 | 10.4 | 10.3 | 10.4 | 10.3 | 10.5 | 9.4 | 1.12 |
| 128 | 1 | 16x128x512 w1x2 b4 | 188.6 | 161.1 | 148.4 | 83.2 | 83.1 | 79.3 | 77.3 | 78.0 | 79.3 | 75.5 | 1.05 |
| 128 | 2 | 16x128x128 w1x2 b3 | 82.3 | 69.9 | 71.9 | 45.8 | 45.3 | 45.4 | 45.3 | 44.5 | 42.4 | 40.3 | 1.05 |
| 1024 | 1 | 32x128x256 w1x2 b2 | 179.0 | 164.3 | 162.3 | 91.1 | 90.3 | 86.9 | 87.8 | 87.5 | 83.3 | 84.3 | 0.99 |
| 1024 | 2 | 32x128x128 w1x2 b3 | 107.7 | 96.5 | 98.6 | 69.0 | 51.3 | 50.6 | 50.7 | 49.7 | 49.6 | 47.3 | 1.05 |
| 8192 | 1 | 128x128x256 w1x2 b2 | 328.5 | 313.7 | 322.9 | 239.1 | 203.7 | 175.7 | 164.7 | 161.5 | **136.0** | 133.1 | **1.02** |
| 8192 | 2 | 128x128x128 w1x2 b2 | 307.7 | 283.6 | 292.7 | 308.2 | 130.6 | 125.3 | 122.1 | 121.1 | 117.4 | 113.2 | 1.04 |
| 32768 | 1 | 128x384x256 w2x2 b4 | 1406.3 | 1003.8 | 966.7 | 566.2 | 428.0 | 391.5 | 395.8 | 393.0 | 381.9 | 376.8 | 1.01 |
| 32768 | 2 | 128x256x128 w2x2 b4 | 1050.7 | 891.6 | 893.1 | 928.8 | 366.3 | 350.2 | 348.3 | 347.8 | 335.3 | 330.1 | 1.02 |

step9 这一列是 benchmark 的单次测量，同样逐行加了保护；和 step8 的精确对比以上面 step9 一节的交替测量为准。

每次 benchmark 之间线上本身也有 ±3% 左右的波动（例如 T=16 gemm1 线上测到过 22.7 / 23.3 / 23.6 / 23.8 us），比值的小幅变化以此为准。

## 过程中的发现

- **Occupancy 不是差距来源。** 用 clang `-Rpass-analysis=kernel-resource-usage` 对 dump 出的 IR 统计，naive 和线上每行的 VGPR、LDS、每个 CU 驻留的 WG 数几乎一样。
  - prefill 阶段每个 SIMD 只有 1–2 个 wave，T=1 / T=16 的 gemm1 填不满 256 个 CU。
  - 这些是静态值；实际 occupancy 需要用 rocprof 计数器测。
  - step6 给 A 加 padding 后，T=16 gemm1 的 LDS 从 79.7K 涨到 80.4K，每个 CU 驻留的 WG 从 4 个降到 3 个。这和线上一致（81.0K，3 个），测速上也没有看出变化。
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
- C 复用输入 stage，只有 C 比 stage 大时 LDS 才变大。step5 时 128×128 从 34 涨到 36 KB；step6 给 A 补 padding 后，stage 本身变成 38 KB（和线上相同），C 不再额外占 LDS。

实测：T=32768 gemm2 写回 2.01 GB，kernel 929 → 366 us，按写回量算的带宽下限 2.17 → 5.5 TB/s。
收益和"输出字节 / 计算量"成正比：

| 场景 | 效果 |
|---|---|
| gemm2（N=5120，K=576） | 受益最大 |
| gemm1（N=1152，K=5120） | 次之 |
| decode | 基本不变 |

### 代码分段（`naive_grouped_gemm_mxfp4.py` 第 376–418 行）
例子：128×256×128，w2×2，`WM_REP = WN_REP = 4`，`ACC_LEN = 16`，`C_PITCH = 272`。

1. **第 377–380 行，`sched_barrier(0)` + 第一次 barrier。**
   - 累加器布局：lane l 是第 l%16 行；值 v 是第 `(v//8)*16 + (l//16)*8 + v%8` 列。
   - `sched_barrier` 只约束编译器的调度，不生成指令。
   - barrier 保证所有 wave 读完输入 stage 后，C 才覆盖它：C 占 128×272×2 = 68 KB，落在 102 KB 的 stage 区开头。
2. **第 381–388 行，寄存器 → LDS。**
   - 每个 WMMA 块：fp32 → bf16（8 条 `v_cvt_pk_bf16_f32`）；按 f=0/1 各取 8 个值，`ds_store_b128` 写到 `(row*C_PITCH + col)*2`。
   - 例：wave 3、lane 21，`wm=1, wn=2, f=1` → row 85，col 216，字节偏移 46672。
   - 每个 wave 32 条 `ds_store_b128`。padding 行也照写，没有分支。
3. **第 389 行，第二次 barrier。** TDM 按行分给各 wave（wave w 负责第 32w–32w+31 行、全部 256 列），和计算时的 2×2 分块不同，所以必须等所有 wave 写完。
4. **第 391–401 行，全局 view。**
   - `bf16_ptr`：`ptr_arg` 传进来的指针只按 1 B 对齐，所以显式造一个 16 B 对齐的 bf16 指针。
   - view 形状是 `(rows_end - blk_m, tile_n)`，行数是运行时值：decode 时为 1，prefill 时 ≥128。
   - 列数 256 小于 LDS 行宽 272，所以 pad 列会被丢弃。
5. **第 402–406 行，TDM store atom。** box 为 `(tile_m, C_PITCH)`；LDS 布局是紧密的，不需要 TDM pad 字段；`num_warps=NUM_WAVES` 让每个 wave 负责 32 行。
6. **第 407–412 行，LDS view。** 从 LDS 偏移 0 开始，带一个长度为 1 的 stage 维，形状和读取路径的 `(box, NB)` 一致。
7. **第 413–416 行，`tdm_partition`。** 用 `wave` 选出本 wave 的 32 行：全局地址和 LDS 地址都按 32 行偏移，每个 wave 的有效行上限是 `max(0, rows_end − blk_m − 32·wave)`。
8. **第 417–418 行，发 store + 等待。**
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

不加 `sched_barrier` 会怎样：编译器把 WMMA 挪到 epilogue 的 barrier 之后，和 cvt / `ds_store` 交错。这说明 WMMA 越过 barrier 是合法的，但这种推迟会沿着主循环一路往前传导。

统计方法：对 128×256×128 w2×2（K=640，5 个 K tile）编译两版 ISA，按 `s_barrier` 切段，再从后往前推活跃寄存器。
每次主循环 barrier 表示"tile k 已在 LDS"。每个 tile、每个 wave 有 28 条 `ds_load`（104 个 VGPR）和 16 条 WMMA；"超前"指截至这一段，`ds_load` 比 WMMA 多做的 tile 数。

| 段 | 读的 tile | 有 `sched_barrier`：ds_load / WMMA，超前 | 没有：ds_load / WMMA，超前 |
|---|---|---|---|
| 第 0 次 barrier 后 | tile 0 | 28 / 0，1 | 28 / 0，1 |
| 第 1 次后 | tile 1 | 28 / 16，1 | 28 / 16，1 |
| 第 2 次后 | tile 2 | 28 / 24，0.5 | 28 / 4，1.75 |
| 第 3 次后 | tile 3 | 28 / 16，0.5 | 28 / 12，2.0 |
| 第 4 次后 | tile 4 | 28 / 24，0 | 28 / 13，2.2 |
| epilogue（第 5 次后） | — | WMMA 0，cvt 128，store 32 | WMMA 35，cvt 120，store 32 |
| 活跃 VGPR 峰值 | | 396 = 累加器 192 + fragment 181（约 1.7 个 tile） | 537 = 累加器 256 + fragment 275（约 2.6 个 tile） |

- **`ds_load` 不会动**：它不能挪到"tile k 已到 LDS"的 barrier 之前（数据还没到），也不能挪到下一个 barrier 之后（那之后这个 stage 会被下一个 TDM 覆盖）。所以每个 tile 的 28 条 `ds_load` 固定在自己那一段。
- **有 `sched_barrier`**：这是设计上的寄存器双缓冲。tile_k=128 时一个 tile 只有 1 个 K step，等价于读 tile k+1 的同时算 tile k，`ds_load` 超前 1 个 tile。WMMA 在 epilogue 之前全部发完。
- **没有 `sched_barrier`**：中间几段只发了 4、12、13 条 WMMA，最后 35 条（2.2 个 tile）挤进 epilogue。`ds_load` 超前到 2.2 个 tile，峰值时 16 个累加器全活着，再加上约 2.6 个 tile 的 fragment，分配到 550 个 VGPR（超过 512），每个 SIMD 只能放 1 个 wave。
- 这相当于编译器自动加深了一层软件流水：`ds_load` 的延迟藏得更好，但每个 SIMD 少一个 wave。两者谁快尚未实测。

更深的重叠，按改动从小到大：
1. **第一次 barrier 提前到最后一个 K step 之前。** 有寄存器双缓冲，最后一个 K step 的数据上一步已经读进寄存器，这一步没有 `ds_load`。再用 `sched_group_barrier` 控制交错，比如 WMMA(0,0)、WMMA(0,1)、块 (0,0) 的 cvt+store、WMMA(0,2)……既藏住 cvt/store，又控住 VGPR。收益上限约是每个 wave 128 条 cvt 加 32 条 `ds_store` 的时间。
2. **C 用独立的 LDS 区域，去掉第一次 barrier。** 128×256 会从 102 KB 涨到 170 KB，每个 CU 从 2 个 WG 降到 1 个，得不偿失；小 tile 几乎免费（16×64 的 C 只有 2.5 KB）。
3. **persistent kernel + 延迟等待（线上 `defer_c_wait`）。** C 放在最后一个 pipeline stage，TDM store 发出后不等，直接进入下一个 tile 的 prologue（线上第 543–545 行、第 2830 行，用于 persistent stage1 的量化输出）。

线上普通的非 persistent 路径和 naive 一样：主循环结束 → `stage_fence(0)`（第 2232–2234 行）→ 写 LDS → TDM store，也没做第 1 条那种交错。

## 剩余差距分析（step6 之后测）

### 这次 benchmark 里线上实际开了什么
调优 CSV 中这 12 行只有 T=1 gemm1 用了 `split_k1=4`。`cluster_m` / `cluster_n` 写的是 -1，`_select_cluster_*` 会把它解析成 1，所以 cluster 和 multicast 都没开。除 split-K 外，两边配置相同，剩下的差别都在实现细节。

### 对照实验（单位 us，同一进程里交替跑 3 轮取最小值，GPU 上没有别人的进程）
| T | gemm | 线上 | 线上，A scale 按行存 | 线上，关 split-K | naive（step6） | naive，A scale 连续 |
|---|---|---|---|---|---|---|
| 1 | 1 | 4.5 | 4.4 | 6.7 | 7.3 | 7.4 |
| 16 | 1 | 23.7 | 23.6 | | 25.5 | 24.9 |
| 128 | 1 | 72.8 | 73.1 | | 77.7 | 77.1 |
| 1024 | 1 | 81.1 | 81.7 | | 86.2 | 85.5 |
| 8192 | 1 | 128.6 | 158.5 | | 175.6 | 151.6 |
| 32768 | 1 | 374.5 | 380.2 | | 392.3 | 389.1 |
| 1 | 2 | 3.6 | | | 3.8 | 3.8 |
| 16 | 2 | 9.5 | | | 10.6 | 10.5 |
| 128 | 2 | 40.2 | | | 45.2 | 44.9 |
| 1024 | 2 | 46.2 | | | 51.0 | 50.9 |
| 8192 | 2 | 111.2 | | | 128.2 | 124.7 |
| 32768 | 2 | 323.2 | | | 353.3 | 353.2 |

（gemm2 的 K=576 有半个 128 的尾块，线上不支持按行存的 A scale，所以那一列只有 gemm1。）

- **split-K**：T=1 gemm1 这一行 2.8 us 的差距里，有 2.2 us（79%）来自 split-K。这一行只有 108 个 workgroup，填不满 256 个 CU；去掉 split-K 后 naive 只比线上慢 1.09 倍。
- **A scale 布局**：只在 T=8192 gemm1 上影响很大，其他行 ≤3%。step7 已处理。

### 主循环：ISA 对比（T=8192 gemm1，每个 K tile）
两边的 WMMA（32 条）和 `ds_load_b128`（48 条）完全相同，差别在：

| 每个 K tile | 线上 | naive |
|---|---|---|
| `s_wait_dscnt` | 11 | 4 |
| 分支 / `s_nop` | 1 / 0 | 6 / 3 |
| VALU | 3 | 16 |
| scale 的 `ds_load` | 6 | 13 |

- **等待粒度**：线上每次只等下一条 WMMA 需要的那几个 `ds_load`；naive 一大批读完才统一等一次，LDS 延迟藏得少。
- **发 TDM 的方式**：naive 用 `if wave == i` 决定哪个 wave 发哪个 copy，每个 copy 带一个分支和一个 `s_nop`，每个 tile 还要重算 stage 地址；线上每个 wave 各发自己那一份，没有分支。

### 主循环：扫 K（gemm2 的形状 N=5120，K 取 128 的倍数，naive 用连续 A scale）
把耗时拟合成 `t = a + b × (K/128)`，a 是固定开销，b 是每个 128 的 K tile 的开销：

| T | 配置 | 线上 a / b | naive a / b | b：naive 比线上 |
|---|---|---|---|---|
| 16 | 16×64×128 w1×1 b2 | −1.1 / 2.35 | −2.1 / 2.90 | +23%（K=1024 处有突跳，拟合不准） |
| 128 | 16×128×128 w1×2 b3 | 5.2 / 7.61 | 4.3 / 7.89 | +4% |
| 1024 | 32×128×128 w1×2 b3 | 8.7 / 8.00 | 10.6 / 8.33 | +4% |
| 8192 | 128×128×128 w1×2 b2 | 46.4 / 11.89 | 25.2 / 19.09 | +61% |

- 固定开销两边差不多（T=8192 时 naive 反而更低），差距在每个 tile 的开销上，也就是主循环。
- 只有 2 个 buffer 的配置差距最大：TDM 只能提前 1 个 tile，而 naive 的等待又是大块的。

### 已排除（在这个 benchmark 里两边相同）
- **找专家**：线上第 727–750 行用的算法和 naive 相同（lane 并行数 psum，再 `shuffle_xor` 求和）。
- **K 尾块**：目前两边读的字节数和 WMMA 条数相同。线上最后一个 K tile 有越界读的 bug，也按整个 tile 去读；线上的尾块同样做 1 个 128 的 WMMA 步（K=576 的最后 64 补零）。等线上修好这个 bug，gemm2 线上会少读最多 11%，这一项才会变成差距。
- **occupancy、epilogue**：两边已经一致。
- **主循环里的 VALU 地址计算**：试过，没有收益，见下一节。

### 试过但没有保留：预先算好 LDS 地址（step8 之后）
- **问题**：step8 之后主循环每个 K tile 有 25 条 VALU，线上只有 3 条。其中约 16 条是 `v_add_nc_u32 vX, s_stage, v_lane`：stage 号 `kt % NB` 是运行时的值，编译器给每个 WMMA 块、每次读都单独留了一个循环外的 lane 偏移寄存器，每个 tile 都要给它们加一次 stage 基址。其余 8 条是填 hazard 的 `v_nop`。
- **做法**：
  1. **中间版**：每个读取器（A、W、A scale、W scale）只保留一个 lane 偏移，WMMA 块、K step、第几次读的差别都作为编译期常量折进 `ds_load` 的立即数 offset。每个 tile 的 VALU 从 25 降到 15（7 次加法和 8 条 `v_nop`）。
  2. **展开版**：再把稳态循环按 NB 展开，每份循环体里的 stage 号都是编译期常量，stage 偏移也进了立即数。循环里的 VALU 变成 0。
  - 两个版本的正确性都是 15/15，`s_nop` 为 0，没有任何 TDM 后紧跟分支。
- **结果**（同一进程里交替跑 3 轮取最小值，单位 us，GPU 上没有别人的进程）：

  | 行 | step8 | 中间版 | 展开版 |
  |---|---|---|---|
  | T=1 gemm1 | 7.3 | 7.3 | 7.4 |
  | T=16 gemm1 | 23.9 | 23.7 | 23.6 |
  | T=128 gemm1 | 75.8 | 75.3 | 77.2 |
  | T=128 gemm2 | 44.0 | 43.3 | 43.5 |
  | T=1024 gemm1 | 85.2 | 86.1 | 84.6 |
  | T=8192 gemm1 | 155.4 | 153.6 | 152.7 |
  | T=8192 gemm2 | 123.8 | 123.4 | 126.2 |
  | T=32768 gemm1 | 384.3 | 403.0 | 393.2 |
  | T=32768 gemm2 | 353.3 | 356.8 | 357.2 |

  另外 3 行（T=1 gemm2、T=16 gemm2、T=1024 gemm2）三个版本之间相差都不超过 0.5 us。
- **结论**：除了 T=32768 gemm1，其余各行三个版本的差别都在 ±2% 以内；T=32768 gemm1 两个新版本反而慢 2–5%。展开版还让这一行的 VGPR 从 674 涨到 786，代码量也变大。所以 VALU 地址计算不是瓶颈：和每个 tile 的 32 条 WMMA、48 条 `ds_load` 比，它的开销很小。代码回退到 step8。

## Workgroup 发射顺序、XCD 与 L2（step9 的依据）

结论：让共用同一个 A tile 的 workgroup 落在同一个 XCD 上，并且在差不多同一时间运行，这个 A tile 就只需要从显存读一次进这个 XCD 的 L2。在 T=8192 gemm1 这一行，应该让 A 留在 L2：A 被复用 9 次，B 一次复用都没有。但在更大的 batch 下情况会反过来。

step9 用线上的分组 swizzle 验证了这一点：T=8192 gemm1 从 155.5 us 降到 128.6 us（−17%），和线上的差距从 1.23 倍降到 1.02 倍。

### rocprofv3 ATT 的证据（T=8192 gemm1，step8）
用 `rocprofv3 -i input.yaml -- python prof_driver.py ...` 抓了 thread trace（`advanced_thread_trace: true`，`att_target_cu: 1`）。下面是被 trace 的 CU 上所有 wave 的周期合计：

| | naive | 线上 |
|---|---|---|
| kernel 耗时（profiler 下） | 169.5 / 171.2 us | 140.1 / 140.6 us |
| wave 总周期 | 1,516,308 | 1,119,928 |
| 卡在 `s_wait_tensorcnt`（等 TDM 数据到 LDS） | **893,692（59%）** | 267,695（24%） |
| 卡在 `s_wait_dscnt` | 182,101 | 239,829 |
| 卡在 barrier | 57,635 | 104,752 |

- 两边主循环的结构差不多：一次 TDM 从发出到被等待，中间都隔着大约一个 tile 的计算。所以是 naive 的 TDM 本身更慢，数据来源更远。LDS 等待和 barrier 反而是线上更多。
- 抓 trace 时遇到的坑：rocprofv3 1.3.5 同时开 `sys_trace: true` 和 ATT 时，在最后写 rocpd 数据库那一步会段错误崩溃，ATT 也来不及解码；关掉 `sys_trace` 就能正常解码出 `ui_output_*` 和 `stats_*.csv`。所以分两遍跑：
  - 一遍 `sys_trace: true`：崩溃之前 kernel / HIP / HSA 的 CSV 已经写好；
  - 一遍 ATT 专用：`sys_trace: false`，加 `kernel_iteration_range: "[2-3]"`，只抓缓存预热之后的 dispatch。
  - 两份 yaml 都要加 `att_library_path` 指向 `librocprof-trace-decoder.so` 所在目录。

### gfx1250 的拓扑
- KFD topology：`Num_Xcc = 8`、`Cu_Count = 256`，也就是每个 XCD 32 个 CU；一共 16 个 shader engine，每个 XCD 2 个。
- rocminfo 报的 L2 是 4096 KB。MI300 / gfx950 上这个数是按单个 XCD 报的，所以应是每个 XCD 有一个独立的 4 MB L2，共 32 MB。gfx1250 上没有实测确认。
- workgroup 和 XCD 的对应关系：workgroup w 在 XCD `w % 8` 上（aiter `mega_moe_large_m.py` 第 1948 行的注释，这段是 gfx950 的代码，gfx950 同样是 8 个 XCD × 32 个 CU；gfx1250 未实测）。XCD 内部再由硬件派到当时有空闲资源的 CU 上。

### 例子：T=8192 gemm1（128×128×256，w1×2，b2）
- 384 个专家，每个专家正好 128 行，也就是一个 M tile。N=1152，切成 9 个 N tile。A tile e（128 × 2560 B = 320 KB）被 (e, n0)…(e, n8) 这 9 个 workgroup 用到；每个专家的 W tile 只被它唯一的那个 M tile 用到。
- **naive（step8 及以前的 grid）**：grid 是 (765, 9)，M 在最内层，workgroup 编号 w = e + 765·n。765 % 8 = 5，所以 A tile 0 的 9 个使用者落在 XCD 0、5、2、7、4、1、6、3、0 上：几乎每个 XCD 都要从显存单独读一次这份 A，而且两次使用之间编号相隔 765，在时间上也离得很远。
- **线上 / step9**：一维 grid，大小 = M tile 数 × N tile 数。每 16 个 M tile 分成一组，组内 M 在内层。组内第 j 个 M tile 的 9 个使用者，编号是 j、j+16、…、j+128，`% 8` 全都是 j % 8，所以落在同一个 XCD 上，而且都在相邻的 144 个编号之内，几乎同时驻留。这份 A 只读一次进 L2，之后 8 次都命中。
- **每个 CU 上跑什么**：两边分发的单位都是 workgroup，不会把 9 个 N tile 塞进同一个 CU。每个 workgroup 2 个 wave，输出 128×128；每个 wave 输出 128（M）× 64（N），也就是 8 个 16 行的 M 块 × 2 个 32 列的 N 块。每个 K tile 分 2 个 K step、每步 16 条 `v_wmma_scale_f32_32x16x128_f4`，20 个 K tile 共 640 条。受 VGPR 限制每个 SIMD 驻留 2 个 wave，所以每个 CU 8 个 wave、4 个 workgroup。整卡一次驻留 1024 个 workgroup，3456 个有效 workgroup 大约要跑 3.4 轮。

### A 和 B 谁留在 L2 更划算
- 复用次数：A tile = N tile 数（N / tile_n）；B（W）tile = 同一专家的 M tile 数（每个专家的行数 / tile_m）。
- 能留在 L2 时，少从显存读的量 =（复用次数 − 1）× 这份数据的总量：

| 行 | A 总量 × 复用次数 | 可省的读取 | W 总量 × 复用次数 | 可省的读取 | 该留谁 |
|---|---|---|---|---|---|
| T=8192 gemm1 | 126 MB × 9 | 约 1.0 GB | 1.13 GB × 1 | 0 | **A** |
| T=8192 gemm2 | 16 MB × 40 | 约 0.6 GB | 0.57 GB × 1 | 0 | A |
| T=32768 gemm1 | 503 MB × 3 | 约 1.0 GB | 1.13 GB × 4 | **约 3.4 GB** | **W** |

- **decode 和中等 batch**：每个专家只有一个 M tile，W 没有任何复用，只有 A 值得留在 L2。这一行这一点最明显：A 有 9 个使用者，而且每份 A 有 320 KB。
- **大 batch prefill**：每个专家的 M tile 变多，W 的复用越来越重要。step8 的 grid 里同一专家的几个 M tile 编号相邻，对 W 复用天然友好，所以 T=32768 gemm1 之前只慢 1.03 倍。
- **线上 swizzle 两头兼顾，但偏向 A**：同一个 A tile 的使用者一定在同一个 XCD 上；同一专家的几个 M tile 在组内编号相邻，按 `w % 8` 会分散到不同的 XCD，W 还是要在多个 XCD 各读一份。

## 后续优化方案

按预期收益和依据排序（A 的 LDS padding、A scale 连续布局、无分支的 TDM 发射、分组 swizzle 已分别在 step6–step9 完成）。step9 之后，除了 T=1 gemm1，其余 11 行都在线上的 1.12 倍以内，其中 10 行在 1.05 倍以内：

1. **split-K（线上 `k_waves`）。** 只影响 T=1 gemm1，但这一行还差 1.62 倍，其中约 79% 来自 split-K。
   - 每组 wave 算一段 K，部分和经 LDS 归约（线上的 `K_PARTIAL_B` / `GROUP_ARENA_B`）。
2. **decode 的 gemm2（T=16 慢 1.12 倍）。** 先用 rocprofv3 ATT 抓这一行，看是固定开销还是 TDM 等待。
   - 顺带在 gfx1250 上实测 workgroup → XCD 的映射（每个 workgroup 读一下所在 XCD 的硬件寄存器），确认 `w % 8` 这条规则。
3. **A 数据预排布（线上 `a_preshuffled`）。** 需要 gemm1 的 epilogue 直接按 K32/M16 块写出 fp4。这样 A 的 TDM 行变长，LDS 也不需要 padding。
4. **kernel 内处理 K tail，去掉 host 端补零。** 现在不是差距来源（原因见上面"已排除"），但正确的做法是 gemm2 少读最多 11%：最后一个 K tile 用正确的 extent 做越界保护，只做 `ceil(K_TAIL/128)` 个 WMMA K step。
5. **cluster + B multicast。** 同一专家的多个 M tile 共享同一块 W，prefill 阶段可以减少 L2/HBM 流量（这次 benchmark 里线上也没开）。按「A 和 B 谁留在 L2 更划算」的分析，大 batch 下 W 的复用比 A 更重要，而现在的 swizzle 会把同一专家的几个 M tile 分到不同 XCD 上，这正是 multicast 或按 XCD 重排可以补的地方。
6. **融合 epilogue。** bias、SwiGLU / SiTUv2、gemm1 输出的 MX 量化（`stage1_quant_out`，可以直接写成 gemm2 要的预排布 A）、gemm2 的 EP scatter。
7. **epilogue 和最后一个 K step 交错。** 把第一次 barrier 提前，用 `sched_group_barrier` 让 cvt/`ds_store` 和最后的 WMMA 交错（见「epilogue 能否和最后的 WMMA 重叠」）。收益有限。
8. **persistent kernel / tile 调度。** 解决小 M 填不满 GPU、大 M 尾轮不均的问题；顺带可以让 TDM store 和下一个 tile 的主循环重叠（线上 `defer_c_wait`）。
