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
8. **persistent kernel / tile 调度。** 解决小 M 填不满 GPU、大 M 尾轮不均的问题。
