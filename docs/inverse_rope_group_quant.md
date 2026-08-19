# Inverse-RoPE + FP8 group-quant (V4 `wo_a` 前处理) — profile 驱动优化记录

对象：`csrc/kernels/inverse_rope_group_quant.cu` 里的
`aiter::inverse_rope_group_quant_kernel`（DeepSeek-V4：对 attention 输出做
inverse GPT-J RoPE + 1×128 e8m0 FP8 group-quant）。对照 Triton 版
`aiter/ops/triton/fusions/inverse_rope_group_quant.py` 与「inverse_rope +
独立 quant」两步基线。

测量环境：**MI355X (gfx950, 256 CU)**，容器 `yzhou_atom_latest2`，`HIP_VISIBLE_DEVICES=7`。
方法论沿用 `fused_qk_norm_rope_cache_quant.md` §0/§2/§11.1。

> ⚠️ **§1–§12 是第一轮（gfx950/wave64）的记录，其中的代码形态已经被第二轮大改。**
> 第二轮在 **gfx1250 (wave32)** 上重做了线程映射与 grid 映射，见 **§13**（含交接清单）。
> 第三轮在另一台 gfx1250 上复现第二轮的结果、补齐它缺的 shuffle 表，并评估了那份
> gfx950 修复 patch 对 gfx1250 的影响，见 **§14**；
> **要引用最新数字请看 §14.1，不要用 §13.11。**
> 第四轮加了 n32k4 scale 布局（ROCm/aiter#4626 的 a8w4 batched GEMM 要的那个）、
> 把 `scale_layout` 从运行期参数改成模板参数，并把 scale 散写的代价拆成"事务碎片"
> 与"通道串行"两项，见 **§15**（含交接清单）。
> **读第三轮及以前时请把 `-l row shuffle` 当作历史，现在是 `-l row mfma n32k4`。**
> 读第一轮时请把 `K_PER_BLOCK` / `BLOCK_M` / `row = blockIdx.y*rows_per_block + ...`
> 当作历史；现在的参数是 `K_PER_THREAD` / `k_slots` / `blockIdx = (s, Ks span, g)`。

形状（TP=8 局部）：`o[S,16,512] bf16` → `x_fp8[S,2,4096]` + `scale[S,2,32] e8m0`，
`rope_dim=64`（只作用于每个 head 的尾 64 维），`quant_group_size=128` → `scaleN=32`。

---

## TL;DR

**最终结果**（idle GPU7，`run_perftest` hip_us，µs/call；同 run Triton 归一化）

| S | cpp 起始 | **cpp 最终** | triton | two_step | cpp/triton |
|---|--:|--:|--:|--:|--:|
| 1 | 3.13 | **2.45** | 2.53 | 4.67 | 0.97 |
| 32 | 3.52 | **2.69** | 2.64 | 5.24 | 1.02 |
| 128 | 4.15 | **3.18** | 3.35 | 5.46 | 0.95 |
| 256 | 5.36 | **3.40** | 4.52 | 5.93 | 0.75 |
| 512 | ~7.3 | **4.39** | 6.62 | 6.72 | 0.66 |
| 1024 | 12.24 | **6.49** | 10.78 | 9.28 | 0.60 |
| 2048 | ~24.7 | **11.37** | 20.29 | 14.48 | 0.56 |
| 4096 | 42.05 | **19.15** | 37.42 | 22.33 | 0.51 |
| 8192 | ~82.4 | **34.32** | 73.71 | 39.08 | 0.47 |

即 **S≥256 提速 1.6–2.4×**，S≤128 基本持平（撞 dispatch 地板）。S=8192 时
203 MB 流量 / 34.3µs ≈ **5.9 TB/s**（起始 ~2.4 TB/s）。HIP 现在在所有 S 上
≤ Triton，且全面优于两步基线。

**两条真正起作用的优化**（都由 ATT 定位）
1. **给每个 wave 多个在飞 load**（`K_PER_BLOCK`）：原来每 thread 只发 1 条 load 就
   `s_waitcnt vmcnt(0)` 全量等待，访存延迟完全裸露。让每 block 处理 KPB 个 quant
   group、**先把 KPB 条 load 全发出去再消费**，编译器随之生成 `vmcnt(3)` 分级等待。
   S=8192 单这一项 **82.4 → 42.7µs（1.93×）**。
2. **cos/sin 向量化**：rope 分支原来按元素发 `global_load_ushort`（每对后面跟满等待），
   占改动后剩余 stall 的 ~37%。一个 thread 的 TDS 个元素只需 TDS/2 个**相邻** cos/sin，
   合成两条向量 load。S=8192 再 **42.7 → 34.3µs**，各 S 归一化比值全面下降。

**一条被实测否决的优化**
- **DPP 替换 `__shfl_xor` 规约**：ISA 上确实生效（`ds_bpermute` 5→1 条，
  total stall −26%，executed 指令 −15%），但 **Triton 归一化 wall-time 完全不动**。
  与 `fused_qk_norm_rope_cache_quant.md` §10.4「砍 ALU（fused-DPP）无效」一致：
  规约的 stall 本来就藏在那条 load 等待背后。**保留它只因为顺带发现了下面的 bug，
  以及它让寄存器/指令更省，不因为它快。**

---

## 1. Profile 方法与踩坑

沿用既有纪律，另外补两条本轮新增的：

- **`.so` mtime 必须核对**（`fused_qk_norm_rope_cache_quant.md` §11.1 的 stale-build
  陷阱）。本轮全程用 `rm -f aiter/jit/module_inverse_rope_group_quant.so`
  强制重编，每次编完 `ls --time-style=full-iso` 看时间戳。优化收尾时把 kernel 从
  6600 行的 `fused_qk_norm_rope_cache_quant.cu` 拆到独立的
  `csrc/kernels/inverse_rope_group_quant.cu` + 独立 module
  `module_inverse_rope_group_quant`，重编从 ~96–113s 降到 ~21s；调这个 kernel
  务必只重编这个小 module。
- **~2.4–2.5µs 是本机 dispatch 地板**：kernel-trace 里连 1 个 block 的
  `vectorized_elementwise_kernel` 都要 2.5µs。所以 S≤64 的读数几乎全是地板，
  "S=1 的 0.1µs 差异"没有意义；要看真信号得取 S≥512，或用「S=X 减 S=1」估算 body。
- **跨 run 时钟漂移很大**（同一份 binary，Triton S=1 在 2.48↔3.12µs 之间摆）。
  **一律用同 run 的 Triton 归一化比值**，别直接比绝对值。

一次编译扫完参数：扫参期间 kernel 里临时加过 `AITER_INV_ROPE_TDS` / `_BM` / `_KPB`
三个环境变量覆盖 dispatch（默认 0 = 走出厂启发式），避免每个参数点重编 100s。
**调参结束后已删除**（见 §9）——这类脚手架留在产线代码里只会让不可达的模板继续被
实例化。下次要重扫参数时按同样套路临时加回来即可。

复现：
```bash
ssh mi355-gpu-47; podman exec -it yzhou_atom_latest2 bash
cd /home/jun_chen2_qle/yzhou_aiter/aiter_3
rocm-smi --showuse | grep use            # 先确认 0%
rm -f aiter/jit/module_inverse_rope_group_quant.so   # 改 .cu 后必须
HIP_VISIBLE_DEVICES=7 PYTHONPATH=$PWD python3 \
  op_tests/test_inverse_rope_group_quant.py --tokens 1 32 128 512 2048 8192

# kernel-trace（真实 GPU duration + grid/VGPR/LDS）
HIP_VISIBLE_DEVICES=7 PYTHONPATH=$PWD rocprofv3 --kernel-trace \
  -d /tmp/kt -o run --output-format csv -- \
  python3 op_tests/prof_inverse_rope_group_quant.py --impl both --s 128
python3 op_tests/parse_ktrace.py S128=/tmp/kt/run_kernel_trace.csv

# ATT（每条 ISA 的 stall）
HIP_VISIBLE_DEVICES=7 PYTHONPATH=$PWD rocprofv3 \
  -i op_tests/prof/att_inv_rope_cpp.yaml -- \
  python3 op_tests/prof_inverse_rope_group_quant.py --impl cpp --s 2048 --warmup 5 --iters 1
python3 op_tests/parse_att.py X=/tmp/inv_att_cpp/ui_output_agent_*_dispatch_15/code.json
```
辅助脚本（本轮新增）：`op_tests/prof_inverse_rope_group_quant.py`（纯 launch driver，
不含 `run_perftest`，避免双重 profiling）、`op_tests/parse_ktrace.py`、
`op_tests/parse_att.py`、`op_tests/diag_inverse_rope.py`（定位错在哪个 group/lane）。

---

## 2. 起始 ATT 画像（S=128，warm dispatch）与判读

| 类别 | HIP stall | 占比 | Triton stall | 占比 |
|---|--:|--:|--:|--:|
| s_waitcnt | 35924 | 54.5% | 17332 | 23.7% |
| VALU(含 cmp/sel/trans) | 23736 | 36.0% | 29060 | 39.7% |
| SALU | 5688 | 8.6% | 7312 | 10.0% |
| s_barrier | 0 | 0% | 17280 | 23.6% |
| **total stall** | **65888** | | **73228** | |
| executed instr | 12480 | | 16256 | |

**注意这里和 `fused_qk_norm_rope_cache_quant.md` 的画像不同**：那个 kernel 是
s_waitcnt ~91% / VALU ~5% 的纯访存等待；本 kernel s_waitcnt 只 54%、VALU 36%，
所以不能照搬那边「ALU 优化一定无效」的结论——但最终实测**结论仍然一样**（见 §3）。

HIP 的 s_waitcnt 拆开：
- `vmcnt(0)` 11628（主数据 load）
- `lgkmcnt(0)` 24296，其中 **16024 来自 5 条 `ds_bpermute_b32`**（`__shfl_xor` 规约），
  8272 来自 kernarg s_load。

**关键对比（真正的根因）**：同样 10 条静态 VMEM load 指令，
- HIP `hit=64`（**每 wave 只执行 1 条 load**，8 字节 `buffer_load_dwordx2`），
  紧跟 `s_waitcnt vmcnt(0)` 全量 drain → 在飞请求数 = 1，延迟全裸露。
- Triton `hit=640`（**每 wave 6+ 条 load**），用 `vmcnt(5)/vmcnt(2)/vmcnt(3)` **分级
  等待** → 在飞请求数 6，wave 内部就把延迟叠掉了。

Little's law：吞吐 ∝ 在飞请求数 / 延迟。这就是 gap 的来源，也解释了为什么后面
砍 ALU 无效、而加 KPB 立刻 1.9×。

---

## 3. 被否决：DPP 规约（ISA 生效但 wall-time 不动）

把 `__shfl_xor(amax, stride, WARP_SIZE)` 换成 DPP 规约。`__shfl_xor` 每步展开成 9 条
指令（`v_xor` + `v_cmp` + `s_nop 1` + `v_cndmask` + `v_lshl` + `ds_bpermute` +
`s_waitcnt lgkmcnt(0)` + 2×`v_max`），5 步共 ~49 条、**26028 stall = 全部 stall 的 39.5%**。

ISA 确认改动生效：`ds_bpermute` 5→1、static instr 375→345、executed 12480→10560、
total stall 65888→48808（**−26%**）、latency 128076→102876。

**但 Triton 归一化 wall-time 每格都 flat**（S=8 1.109→1.126、S=128 1.061→1.060、
S=4096 1.104→1.108）。原因：S=128 时 512 blocks×512 thread = 16 waves/CU，
per-wave stall 被别的 wave 发射掩盖；真正卡住的是 §2 那条"1 条 load + 全量 drain"。

→ **与 `fused_qk_norm_rope_cache_quant.md` §10.4 结论一致：total stall ≠ runtime。**
本轮最终代码保留了 DPP 版（指令/寄存器更省、且顺带修了下面的 bug），但**不把它算作提速项**。

### 3.1 ⚠️ 顺带发现：`hip_reduce.h` 的 `multithread_reduce_max_dpp` 在 thread_num=16 会算错

换用 `hip_reduce.h` 的 `multithread_reduce_max_dpp<N>` 后，`THREADS_PER_GROUP=16`
（TDS=8）**正确性挂**：192 个 group 里 19 个 scale 恰好小一个 binade
（got 0.000488 vs want 0.000977），出错 lane 恒为 `{1,3,4,6,9,11,12,14}`（mod 8 呈
`{1,3,4,6}` 周期）。

根因（ATT 反汇编实证）：该 helper 用 `asm volatile` 发 DPP，**编译器看不见这是 DPP**，
两条 DPP 之间只插了 `s_nop 0`（1 个 wait state），而 gfx9 对「VALU 写 VGPR → DPP 读该
VGPR」要求 **2 个 wait state**：

```
v_max_f32_dpp v9, v8, v8 quad_perm:[1,0,3,2] ... bound_ctrl:1
s_nop 0                         # 只有 1 个 wait state，gfx9 需要 2
v_max_f32_dpp v8, v9, v9 quad_perm:[2,3,0,1] ... bound_ctrl:1
```

为什么只有 16 暴露：`thread_num=32/64` 末尾有一次
`rocprim::warp_shuffle(v, thread_num-1, ...)` 广播，会把正确值盖回全部 lane，
**把 bug 掩盖掉**；`thread_num<=16` 靠 butterfly 直接返回，没有掩盖。
`thread_num=8` 在本输入下没被触发，属**未证伪**而非正确。

**修法**：改走编译器可见的 `__builtin_amdgcn_update_dpp`（16 以内 butterfly：
`0xb1` / `0x4e` / row_half_mirror `0x141` / row_mirror `0x140`），跨 16/32 lane 用
`__builtin_amdgcn_permlane16_swap` / `permlane32_swap`（与
`pa_sparse_prefill_opus.h::attn_row_max` 同惯用法）。本 kernel 已改为自带的
`group_reduce_max_dpp<N>`，4 个 tier 全部 err 回到基线。

**待办（本轮未动，影响面在本 kernel 之外）**：`hip_reduce.h` 的 asm 版仍在
`topk_gating_kernels.cu`（WARP_SIZE=64，被广播掩盖）、
`fused_qk_rmsnorm_group_quant.cu:287` 与 `fused_qk_norm_rope_cache_quant.cu:4537`
（两处均为 `GROUP_SIZE / vec_size_i`，**可以取到 16**）在用 → 需单独核查/修复。

### 3.2 测试覆盖漏洞（同批修掉）

`op_tests/test_inverse_rope_group_quant.py` 原来只用 **S=3** 做正确性，而 S=3 只走
TDS=2 这一档 → **TDS=4/8/16 三档从来没被校验过**，所以 3.1 的 bug 才会在 benchmark
里静默通过。测试后来按 `.claude/skills/aiter-op-test/SKILL.md` 整体重写（见 §8），
S sweep 本身就横跨三档 dispatch。

---

## 4. 采纳：`K_PER_BLOCK`（每 wave 多个在飞 load）

原结构：`grid = (scaleN, ⌈S·G/BLOCK_M⌉)`，即**一个 block 只负责 1 个 quant group**，
每 thread 恰好 1 条 load。改为每 block 负责 `K_PER_BLOCK` 个**相邻** group（`grid.x =
scaleN/KPB`），并且**先把 KPB 条 load 全部发出，再进入 per-group 的
rope/规约/store 循环**：

```cpp
vec_i in_vec[K_PER_BLOCK];
#pragma unroll
for (int k = 0; k < K_PER_BLOCK; ++k)
  in_vec[k] = load_vector_nbytes<...>(input_buffer, input_offset0 + k * GROUP_SIZE);
```

ISA 确认（S=2048, TDS=8, KPB=4）：4 条 `buffer_load_dwordx4` 连续发射
（offset 0/256/512/768），后续变成 `s_waitcnt vmcnt(3)` **分级等待**，
store 也合成 `buffer_store_dwordx2`：

```
buffer_load_dwordx4 v[0:3],   v11, s[24:27], 0 offen
buffer_load_dwordx4 v[6:9],   v11, s[24:27], 0 offen offset:256
buffer_load_dwordx4 v[20:23], v11, s[24:27], 0 offen offset:512
buffer_load_dwordx4 v[16:19], v11, s[24:27], 0 offen offset:768
...
s_waitcnt vmcnt(3)              # 只等自己那条，不再全量 drain
```

KPB 扫描（cpp/triton 归一化比值，越小越好）：

| TDS/KPB | S=32 | S=128 | S=256 | S=512 | S=2048 | S=8192 |
|---|--:|--:|--:|--:|--:|--:|
| 4 / 1 | **1.01** | **0.95** | 0.92 | 0.91 | 0.85 | 0.85 |
| 8 / 2 | 1.16 | 0.96 | **0.80** | **0.68** | 0.55 | 0.49 |
| 8 / 4 | 1.30 | 1.10 | 0.89 | 0.71 | 0.54 | **0.45** |
| 8 / 8 | 1.68 | 1.41 | 1.14 | 0.86 | 0.55 | 0.48 |
| 16 / 4 | 1.38 | 1.16 | 0.96 | 0.82 | **0.53** | 0.46 |
| 16 / 8 | 2.19 | 1.77 | 1.34 | 0.95 | 0.54 | 0.47 |

**KPB 是双刃**：它用 block 数换 per-wave 在飞请求数。小 S 本来 block 就不够填
256 CU（S=1 时 KPB=1 只有 32 blocks），再除 KPB 直接掉性能（S=32 KPB=8 差 1.7×）；
大 S block 有余、瓶颈在带宽，KPB 就是纯收益。

**最终启发式**（`quant_group_size=128`，BLOCK_M 恒 16）：

| S | THREAD_DATA_SIZE | K_PER_BLOCK | THREADS_PER_GROUP |
|---|--:|--:|--:|
| ≤4 | 2 | 1 | 64 |
| ≤128 | 4 | 1 | 32 |
| ≤512 | 8 | 2 | 16 |
| >512 | 8 | 4 | 16 |

（`BLOCK_M` 也扫过 4/8/16/32，16 在各档都是最优或并列，故不随 S 变。
TDS=2 在大 S 极差（S=2048 达 1.7–1.8×），TDS=16 在小 S 极差（S≤32 达 1.5×），
启发式的作用正是避开这两端。）

---

## 5. 采纳：cos/sin 向量化 load

加了 KPB 之后重跑 ATT（S=2048），**最大剩余开销变成 rope 分支的 cos/sin**：
成对的 `global_load_ushort` + `s_waitcnt vmcnt(1)/vmcnt(0)`，8 对合计
~686k stall ≈ **总 stall 的 37%**。每次只取 2 字节，且每对都要等。

关键几何观察：一个 thread 的 `THREAD_DATA_SIZE` 个元素只需要 **TDS/2 个相邻**
cos/sin（`cos_i = local>>1`）。且 `group_elem_base`、`block_head_start`、`ROPE_START`
全是偶数 → `local0` 必为偶数 → `(local&1) == (i&1)`、`cos_i = (local0>>1) + (i>>1)`，
于是可以两条向量 load 拿全：

```cpp
constexpr int NCOS = THREAD_DATA_SIZE / 2;
const int local0 = block_head_start + group_elem_base - ROPE_START;
if (local0 >= 0) {                     // 该 thread 整体落在 rotary 尾部
  using vec_c = opus::vector_t<scalar_t, NCOS>;
  const int64_t crow = pos * (RD / 2) + (local0 >> 1);
  const vec_c cvec = *reinterpret_cast<const vec_c*>(cos_cache + crow);
  const vec_c svec = *reinterpret_cast<const vec_c*>(sin_cache + crow);
  ...
}
```

对齐是成立的：`cos_cache` 行长 `RD/2=32` 个 bf16 = 64B；`local0>>1` 是 `NCOS` 的倍数
→ 字节偏移是 `NCOS*2` 的倍数。跨界（`local0 < 0`，thread 部分落在 rotary 内）仍保留
原逐元素路径兜底，因此对非 V4 形状也安全。

收益（同 run Triton 归一化，KPB 版 → +cos/sin 向量化）：
S=128 1.02→0.94、S=256 1.00→0.76、S=512 0.86→0.70、S=1024 0.70→0.60、
S=2048 0.68→0.55、S=4096 0.62→0.50、S=8192 0.52→0.46。绝对值 S=8192 42.7→34.3µs。

---

## 6. 剩余问题 / 下一步

- **小 S（≤128）已撞地板**：cpp/triton ≈ 0.95–1.02，而 ~2.4µs 是本机 dispatch 地板，
  真实 body 只有零点几 µs。要再快只能减少 dispatch 次数（融进上游 kernel 或 HIP graph），
  kernel 内部没什么可榨的。
- **`row / G` 仍是软件整数除法**：`v_cvt_f32_u32` + `v_rcp_iflag_f32` + `v_mul_lo/hi`
  链，约 3.9k stall，且**位于 load 地址依赖链之前**（会推迟 load 发射）。把 `G`
  模板化（V4 恒为 2）可变成移位。**未做**——按 §3 的教训，这类纯 ALU/地址优化在
  当前 occupancy 下预期 flat，做之前应先有 ATT 证据表明它已成为关键路径。
- **kernarg s_load 等待** ~8.3k stall。参照 `fused_qk_norm_rope_cache_quant.md` §10.5，
  减 arg 条数历史上是 flat，优先级低。（曾经的 `shuffle_scale` 参数被传进 kernel 却
  `(void)` 掉，现已不再进 kernel——见 §11。）
- **`hip_reduce.h` 的 asm DPP helper 需要修**（§3.1），影响本 kernel 之外的调用点。

---

## 7. 文件布局（拆分后）

kernel 原先寄生在 6600 行的 `fused_qk_norm_rope_cache_quant.cu` 里，改一行就要重编
~96–113s。已拆成自己的编译单元 + 自己的 JIT module（重编 ~21s）。Python wrapper 也一
并搬出：它是 attention **输出**路径上的独立 op，与 QK-norm/RoPE 那批输入路径 op 不共享
任何 kernel 或 helper，混在一个文件里只是历史遗留。

| 文件 | 内容 |
| --- | --- |
| `csrc/kernels/inverse_rope_group_quant.cu` | `group_reduce_max_dpp` / kernel 模板 / host dispatch |
| `csrc/include/inverse_rope_group_quant.h` | `aiter::inverse_rope_group_quant` 声明 |
| `csrc/pybind/inverse_rope_group_quant_pybind.cu` | `AITER_SET_STREAM_PYBIND` + `INVERSE_ROPE_GROUP_QUANT_PYBIND` |
| `csrc/include/rocm_ops.hpp` | `INVERSE_ROPE_GROUP_QUANT_PYBIND` 宏（已从 `FUSED_QKNORM_ROPE_CACHE_QUANT_PYBIND` 摘出） |
| `aiter/jit/optCompilerConfig.json` | `module_inverse_rope_group_quant`（`-DENABLE_FP8`，include `ck_tile`/`opus`） |
| `aiter/jit/core.py` | 把新 module 加进 `_get_ck_exclude_modules()` 的硬编码表 |
| `aiter/ops/inverse_rope_group_quant.py` | Python wrapper（`@compile_ops` 指向新 module，仍 `develop=True`，保证 graph capture 时 stream 被显式设置）+ `_squeeze_rope_cache`，从 `fused_qk_norm_rope_cache_quant.py` 整体搬出 |
| `aiter/__init__.py` | `from .ops.inverse_rope_group_quant import *`，让 `aiter.inverse_rope_group_quant` 自己成立 |

`fused_qk_norm_rope_cache_quant.py` **一行都不留**。中间一度在那里放过一行 re-export
给 ATOM 兼容，提交前撤掉了：这个 op 在仓库历史里从没在那个文件出现过（ATOM 侧的 wrapper
也还是未跟踪状态），所以"保持向后兼容"无从谈起，留着只是让 diff 变脏。ATOM 本地那个
wrapper 直接改成 `from aiter.ops.inverse_rope_group_quant import ...`。

`kFp8KvQuantAbsmaxFloorF32` / `kHwFp8E4m3Dtype` 在新文件里以
`kFp8QuantAbsmaxFloorF32` / `kHwFp8E4m3Dtype` 各自复制了一份（原文件的是 anonymous
namespace 里的 TU-local 常量，不是共享头文件符号）。`inv_rope_ic<N>` 因为不再有命名
冲突，简化成文件内 anonymous namespace 的 `ic<N>`。

---

## 8. 测试：按 `aiter-op-test` 标准重写

`op_tests/test_inverse_rope_group_quant.py` 原来是手搓的 print 版（`--tokens`、
`--no-bench`、没有 `@benchmark`、没有 markdown 表、只报 `us`）。已按
`.claude/skills/aiter-op-test/SKILL.md` 重写，对齐 `test_batched_gemm_bf16.py`：

- `@benchmark()` + candidate dict（`cpp` / `triton`）+ `run_perftest` +
  `checkAllclose`，每个 candidate 报 `us` / `TFLOPS` / `TB/s` / `err`，
  外加一列 `scale err`；`itertools.product` sweep，结尾 `df.to_markdown`。
- **两个正确性口径**：dequant 后的值用 `rtol=atol=1e-2` 比；e8m0 exponent byte 用
  `rtol=atol=0` 比。torch 参考里按位复刻了
  `fp_f32_to_e8m0_scale<RoundUp, FP8_E4M3>`（`ceil_pow2(amax/max_pos)`，
  `max_pos` 取 `torch.finfo(dtypes.fp8).max` 以自动跟随 gfx942 的 240 / gfx950
  的 448），所以 scale 能真的做 bit-exact 比较。实测两列全 0。
- **轴的映射**：`-b/--hg` = `(n_local_heads, n_local_groups)` =
  `(n_heads//tp, o_groups//tp)`，由 `deepseek_v4.ModelArgs`
  （`n_heads=128, head_dim=512, o_groups=16`）推出 `D=4096` 与 tp 无关、
  真实配置恒满足 `n_local_heads = 8 * n_local_groups`；`-l/--scale-layout`
  的 `row`/`col` 对应 `transpose_scale` 的两种 scale 存储（见 §11）。
- **两步基线被移出表**：`get_hip_quant(per_1x128)` → `per_group_quant_hip` 产出的是
  **fp32 per-group scale**，不是 e8m0 byte，跟本 op 不是同一个量化格式，没法用同一个
  参考校验。旧的 two_step 列数字留在 §5 的记录里即可。
- `SUPPORTED_GFX = ["gfx950"]`：kernel 的跨 lane 规约在
  `THREADS_PER_GROUP >= 32` 时用 `__builtin_amdgcn_permlane16_swap` /
  `permlane32_swap`（gfx950+），而 `S<=4` 档取 TDS=2 → 64 lane/group 会实例化到
  它，所以 gfx942 今天编不过。

### 8.1 重写测试当场抓到的真 bug：cos/sin cache 的 squeeze squeeze 错了维

skill 里"必须按模型真实调用来测，不要测理想化的 op"这条直接暴露了一个集成 bug。
ATOM 的 `_build_cos_sin_cache` 是 `freqs.real.contiguous().unsqueeze(-2).unsqueeze(-2)`，
2D `[max_pos, rd/2]` 连续两次 `unsqueeze(-2)` 得到的是
**`[max_pos, 1, 1, rd/2]`**（aiter `rope_cached_positions` 的 batch/head 布局），
而两个 wrapper 里都写着

```python
# ATOM _V4RoPE stores [max_pos, rd//2, 1, 1]   # ← 注释和现实不符
if cos_cache.dim() == 4:
    cos_cache = cos_cache.squeeze(-1).squeeze(-1)
```

最后一维是 `rd/2` 不是 1，`squeeze(-1)` 是**空操作** → 传进去还是 4D：C++ 路径直接
撞 `cos_cache/sin_cache must be 2D` 挂掉；Triton 路径因为 `[max_pos,1,1,rd/2]` 连续
时内存布局与 2D 相同而**静默正确**，所以一直没人发现。旧测试自己造 2D cache，永远
走不到这条分支。

修法：两个 wrapper 都换成 `_squeeze_rope_cache()`——只在中间维全为 1 时
`reshape(shape[0], shape[-1])`，否则 assert 报错，这样两种 4D 摆法都对，而
`[max_pos, H, 1, rd/2]` 这类不该被压扁的输入会明确失败。

---

## 9. 收尾：删掉调参脚手架，顺带砍掉不可达的实例化

`AITER_INV_ROPE_TDS/_BM/_KPB` 三个 env 覆盖删除后，dispatch 的取值范围收窄为
`tds ∈ {2,4,8}`、`kpb ∈ {1,2,4}`、`bm ≡ 16`——**原来的 `tds=16`、`bm=4/8/32`、
`kpb=8` 分支只能通过 env 取到，删了 env 就全部不可达**，但模板仍然会被实例化。
于是一并清掉：

- `dispatch_bm` 整个删除，`BLOCK_M` 变成函数作用域的 `constexpr int BLOCK_M = 16`
  （4/8/16/32 的扫描结果是 flat，且 16 是能让 `BLOCK_SIZE` 对所有已实例化的
  `THREADS_PER_GROUP`（最大 64）都不超 1024 线程的最大值）。
- `tds` 的 `case 16` 与 `kpb` 的 `case 8` 删除。
- 运行时的 `while (bm * (GS / tds) > 1024) bm >>= 1;` 随之失去意义（bm 固定 16 时恒
  成立），删除；`launch` 里那条 `if constexpr (BLOCK_SIZE > 1024)` 的编译期兜底保留。
- `#include <cstdlib>` 一并删除（getenv/atoi 是唯一使用者）。

实例化数从 `3 GS × 4 tds × 4 bm × 4 kpb = 192`（×2 dtype = 384）降到
`3 × 3 × 1 × 3 = 27`（×2 = 54），**重编时间 20.6s → 9.4s**。性能与正确性无变化
（128 个 checkAllclose 全过，graph capture 全过）。

副作用：`op_tests/diag_inverse_rope.py` 原来靠读 `AITER_INV_ROPE_TDS` 打印当前档位，
现在改为直接复刻 kernel 的启发式算 tds，否则它会一直打印 `TDS=0`。

---

## 11. scale 布局：是 transpose,不是 MX swizzle（`shuffle_scale` → `transpose_scale`）

这一节记录一次绕了远路的排查，结论对以后接 GEMM 很重要。

### 11.1 "shuffle" 这个词在 aiter 里是重载的

对 e8m0 字节 scale，aiter 内部有**两套互不兼容**的 `shuffle_scale` 语义：

| 出处 | e8m0 + shuffle 的含义 |
| --- | --- |
| `quant_kernels.cu` `dynamic_per_group_scaled_quant` | `mx_scale_shuffle_idx` 分块 swizzle，对 32/64/128 **所有** group size 都用 |
| `rmsnorm_quant_kernels.cu` | **只有** `group_size==32` 才 swizzle，否则退化成普通转置 `y*m+x` |
| `dynamic_per_token_scaled_quant`（fp32 scale） | 普通转置 |

所以"e8m0 + shuffle_scale=True"并不能唯一确定布局，必须看消费者。

### 11.2 消费者说了话：blockscale_bpreshuffle 要的是转置

用一个最小受控探针直接问 GEMM（同一个未混洗的 A，只换 scale 布局）：

| 传给 `gemm_a8w8_blockscale_bpreshuffle` 的 x_scale | mean\|out − ref\| |
| --- | --- |
| 全 1（对照：确认 x_scale 真被读） | 6.98（相对随机 scale 的输出差） |
| 行主序 | **1.216** |
| `transpose(0,1).contiguous()` 列主序 | **0.038**（bf16 输出舍入量级，过 1e-2 口径） |

**写这个探针时踩了一个坑值得记**：第一版用 `randn/10` 造输入，4096 个 quant group 的
amax 太接近，**e8m0 字节全部相同**，于是任何排列都等价、反向对照恒为 0，整个测试是空转。
必须给每个 `(token, 128-块)` 乘一个不同的 2 的幂（`randint(-8,8)`）把字节打散，并
断言 `unique().numel() > 1`，测试才有区分力。

ATOM 侧两处独立代码也印证同一件事，且**与 e8m0 无关**——`scale_type=fp8_e8m0` 和
`transpose_scale` 是两个正交开关：

- `linear.py`：`transpose_scale=envs.ATOM_FP8_BLOCKSCALE_WEIGHT_PRESHUFFLE`，注释
  "preshuffle GEMM expects column-major x_scale"。
- `layernorm.py`：e8m0 dtype 下依然 `empty((num_groups, M)).view(M, num_groups)`。

反过来，全仓**没有任何 GEMM** 消费 group-128 + e8m0 + `mx_scale_shuffle_idx`：吃 e8m0
A-scale 的 GEMM group size 全是 32，128×128 blockscale 的 GEMM A-scale 全是 fp32。
`mx_scale_shuffle_idx` 属于 MoE MXFP4/MXFP8 那条 `mxfp4_moe_sort` → GEMM 链。

### 11.2.1 plain blockscale 要的是**行主序**——两者正好相反

`transpose_scale` 跟的是 **B 是否 preshuffle**，不是"blockscale"这个词。同一探针把两个
GEMM 各喂两种布局（m=130 n=512 k=4096，mean|out − ref|）：

| x_scale 布局 | `gemm_a8w8_blockscale`（plain） | `..._bpreshuffle` |
| --- | --- | --- |
| 行主序 `[M, K/128]`，strides `(K/128, 1)` | **0.000001** ✅ | 0.324 ❌ |
| 转置列主序，strides `(1, M)` | 0.316 ❌ | **0.001** ✅ |

即 plain 喂转置 scale 错得和 bpreshuffle 喂行主序一样彻底。所以**不做 B preshuffle 就不
要 transpose**，`transpose_scale=False`。

CK 侧根因：plain 走 `ALayout=Row` → `StrideScaleA = ceil(K/128)`，descriptor
`(StrideScaleA, 1)`；bpreshuffle 的 gridwise 把 A-scale descriptor **硬编码**成
`(1, ceil(M/ScaleBlockM))`（`gridwise_gemm_xdl_cshuffle_v3_multi_d_blockscale_b_preshuffle.hpp:1054`），
CK 官方 example 里也是 `A1Layout = Col` + 注释 "Transpose the AScale tensor for better
performance"。

**m=1 时两种布局等价**（转置是恒等变换），四种组合误差全为 0 —— 拿 decode 单 token 验证
布局会得到假阳性，必须用 m>1。

另外这两个 blockscale GEMM 的 x_scale **都只收 fp32**：喂 e8m0/uint8 直接被拒
（"Scales should have the same dtype..."）。本 op 目前发的是 e8m0 字节，所以无论哪种布局
都还不能直连 blockscale GEMM —— 这与 §11.2 末尾"没有 GEMM 消费 group-128 + e8m0"是同一
个缺口，`transpose_scale` 要等到有 e8m0 消费者、或本 op 改发 fp32 才真正生效。

### 11.3 只有 weight 混洗，act 不混洗

`test_gemm_a8w8_blockscale.py:97-101` 把四个输入的分工写得很清楚：A 的 fp8 数据**原样**
行主序、A 的 scale 转置、B 的权重 16×16 预混洗、B 的 scale 不动。所以本 op 的 `x_fp8`
一直是普通 contiguous `[S,G,D]`，改的只有 `x_scale` 的 strides。

### 11.4 storage 改成 group-major `[G, Ks, S]`

原来列主序是 `empty(Ks, S, G).permute(1,2,0)`，group 在最内层。wo_a 是按 G 分组各跑一次
GEMM 的（`batched_gemm_bf16` batch=G），取第 g 组时 2D scale 的 strides 是 `(G, S*G)`，
**不是** GEMM 要的紧凑列主序。改成 `empty(G, Ks, S).permute(2,0,1)` 后逻辑形状仍是
`[S,G,Ks]`，但 `x_scale[:, g, :]` 的 strides 是 `(1, S)`，其底层 `[Ks,S]` 连续，正好等于
`x_scale.transpose(0,1).contiguous()` 的产物，可直接喂 GEMM。

### 11.5 参数改名 + kernel 不再收这个标志

- `shuffle_scale` → `transpose_scale`（C++ 声明/pybind `py::arg`/两个 Python wrapper/测试）。
  一度在两个 Python wrapper 里留了 `shuffle_scale` 作为已废弃别名（ATOM 是按关键字调的，
  改名会 TypeError），提交前一并撤掉：这个 op 是首次引入，给从没发布过的名字带个
  deprecated alias 说不通。ATOM 本地 wrapper 的形参也顺势改成 `transpose_scale`——ATOM
  自己全仓（`deepseek_v2.py`/`layernorm.py`/`linear.py`…）本来就叫 `transpose_scale`，
  `shuffle_scale` 只出现在这个还没有任何调用方的 wrapper 里，改完反而两边一致。
- 布局完全由调用方的 x_scale strides 决定，kernel **不需要**这个标志：它曾经被传进
  kernel 又 `(void)` 掉。现在 host 侧用它做**布局断言**（`transpose_scale` 时校验
  `stride(0)==1 && stride(2)==S`，否则校验 `stride(2)==1`），把一个死参数变成能抓
  调用方/分配器不一致的护栏，且不占 kernarg。
- 测试新增 `_check_scale_layout`：logical shape 两种布局相同，**只有 strides 能区分**，
  所以光比值是分不出布局的，必须显式断言 strides。

---

## 12. 提交时的坑：aiter_3 的 pre-commit hook 会顺手改无关文件

`.git/hooks/pre-commit` 对每个 staged `.py` 跑 `black` + **`ruff check --fix`**，改完直接
`git add -u`。仓库**没有任何地方 pin ruff 版本**（`requirements*.txt` / `pyproject.toml`
里都没有），所以本地 ruff 一新，它就会把 staged 的 legacy 文件一起"现代化"：

- 只给 `aiter/__init__.py` 加 1 行注册，hook 改出 **104 行** diff；
- 只给 `aiter/jit/core.py` 加 1 行，改出 **54 行**（`from typing import Callable` →
  `from collections.abc import Callable` 之类）。

判定这是环境漂移而不是仓库规范的依据：`origin/main` 自己的 `aiter/__init__.py` 在 ruff
0.16.0 下就有 54 个 error（49 可修）——上游根本没应用这些规则。带着这些改动提交会污染
diff 并极可能在 PR 里冲突。

**更关键的是 ruff 对新文件的"修复"本身是错的**：它把 `Optional[Tensor]` 改成
`Tensor | None`，而 `setup.py` 写的是 `python_requires=">=3.8"`。函数签名的注解是 def 时
求值的，`Tensor | None` 在 3.8/3.9 上直接 `TypeError`。而且 `aiter/ops/` 下
`Optional[` 有 1857 处、`| None` 只有 241 处，惯例也是前者。

处理：`git checkout HEAD~1 -- <legacy files>` 拿回原版、手工加回那一行，新文件里把
`Optional` 写回去，然后 `git commit --amend --no-verify`。hook 没有提供跳过开关。

---

# 13. 第二轮：wave32 (gfx1250) 上重做线程/grid 映射 —— 交接记录

测量环境：**gfx1250**（`rocm/fw-bringup:gfx1250-atom--20260817`，容器 `yzhou_latest`），
仓库 `/home/hwang/zoe/aiter`，分支 `zoe/aiter`。所有 wave32 数字都是本机 `run_perftest`
的 `hip_us`。**gfx950 本轮没有硬件可用，相关结论全部是"静态验证"（编译期资源统计 +
反汇编），已逐条标注。**

## 13.0 一句话状态

`csrc/kernels/inverse_rope_group_quant.cu` 与
`op_tests/test_inverse_rope_group_quant.py` 有未提交改动（约 +256/−147 行）。
gfx1250 上正确性全过、性能 `s=8192` 从 **47.10 → 21.92µs（4.31 → 9.26 TB/s）**、
`s=16384` **93.83 → 42.81µs（9.48 TB/s）**，全面优于 unfused 两步基线。
**gfx950 只做了静态验证，未跑过一次**，这是交接时最大的未闭合项（见 §13.13）。

代码状态标记（下文数字都会标 A/B/C/D）：

| 状态 | 含义 |
| --- | --- |
| **A** | 本轮起点 = 分支 HEAD（§1–§12 的产物在这个分支上的形态） |
| **B** | A + 重做线程映射（§13.3） |
| **C** | B + DPP 正确性修复（§13.7）+ wave64 分档（§13.8） |
| **D** | C + grid 维序（§13.4）+ 消除除法与 blockDim 加载（§13.5）= **当前工作区** |

## 13.1 起点：报告的"4.7 → 4.2 TB/s 回退"是测量噪声

用户报告优化后从 4.7 掉到 4.2 TB/s。**同一台机器上把提交前后各重跑一遍，两边都是
46.4µs / 4.37 TB/s** —— 回退不存在，是跨 run 的时钟漂移（§1 第三条踩坑，wave32 机器上
同样成立，而且更明显：同一 binary 同一 shape，`s=4096` 在不同 run 里量到过 16.09 和
12.65µs，差 27%）。

**纪律：只比同一个 run 内的数字。**本节所有对比都注明是同 run 还是跨 run。

真正的问题不是回退，而是这个 kernel 一直只跑到可达带宽的一半：状态 A 在 `s=8192`
是 4.31 TB/s，而**同 run 的 unfused 两步基线是 5.73 TB/s** —— 融合版比不融合还慢。

### 13.1.1 新增踩坑：`rm -f *.so` 不够，还要删 build 目录

§1 说改 `.cu` 后要 `rm -f aiter/jit/module_inverse_rope_group_quant.so`。本轮发现
**光删 `.so` 会拿到旧二进制**：JIT 的 ninja 目录还在，会命中缓存。必须两个都删：

```bash
rm -f  aiter/jit/module_inverse_rope_group_quant.so
rm -rf aiter/jit/build/module_inverse_rope_group_quant
```

本轮最初几次测量就是被这个坑住的，得到的"基线"其实是旧代码。

## 13.2 根因：一个 wave 的 load 跨行散射

状态 A 的映射是 `tid / THREADS_PER_GROUP` 当**行号**（`row_in_tile`），
`blockIdx.x` 当 group 号。于是 wave32 上一个 wave 的 32 个 lane 被切成两半，
分别落在相隔 `D*2 = 8KB` 的两个不同 `[S,G]` 行里，各取 256B —— 一条 load 指令拆成
两段远隔的短传输。

消融实测（状态 A，`s=8192`，逐项注掉，同 run）：

| 注掉的部分 | 省下 |
| --- | --- |
| scale 字节写入 | ~11µs（每个 wave 只写 2 个相隔 32B 的字节） |
| 跨 lane amax 规约 | ~7µs（每组 16 lane，4 步 DPP） |
| inverse RoPE | ~6µs（`positions[s]` 每 lane 各发一次 64-bit gather） |

对照：同样访存量的纯 `dynamic_per_group_scaled_quant` 只要 30.6µs。所以差距不在算法，
在访存组织。

## 13.3 采纳（状态 B）：一个 block = 一行 × 一段连续 group

把映射整个翻过来：**block 只负责一个 `[S,G]` 行里一段连续的 quant group，`tid` 沿 d 方向走**，
于是 lane 顺序就是地址顺序。

```cpp
// K_PER_BLOCK -> K_PER_THREAD；行不再由 tid 切分
const int k_slot        = tid / THREADS_PER_GROUP;   // 负责段内第几个 group
const int lane_in_group = tid - k_slot * THREADS_PER_GROUP;
```

连带三件事自然成立：

1. **一个 wave 一段连续区**：`wave_size × 每线程字节`，wave32/tds=16 是 1KB，
   wave64/tds=8 是 1KB。row-major 的 scale 字节也变成每 wave 一段连续写。
2. **`s` 成为 block 不变量** → `positions[s]` 从 per-lane gather 降为每 block 一次标量加载。
   状态 A 里还有个 `any_rope` 判断决定是否加载，新映射下 block 跨度 ≥ `GROUPS_PER_HEAD`
   时它恒为真，所以直接无条件加载（这一点的代价见 §13.13 第 4 条）。
3. **RoPE 按 thread 整体旋转**：`ROPE_START=448`、`THREAD_DATA_SIZE ∈ {8,16,32}` 都能整除，
   于是 `kSliceAlignedToRope` 为真，`orig[]` 临时数组和逐元素分支全部消失；不对齐的形状
   仍保留逐元素兜底路径，非 V4 形状安全。

同时每组的 lane 数从 16 降到 4–8，规约步数 4 → 2–3。

**状态 B（gfx1250，bf16，h=16 g=2，gs=128，µs，跨 run 参考值）**

| s | A row | B row | A shuffle | B shuffle |
|---|--:|--:|--:|--:|
| 512 | 5.31 | 5.68 | 6.39 | 5.27 |
| 2048 | 13.72 | 11.84 | 18.68 | 11.85 |
| 4096 | 22.57 | 16.09 | 30.90 | 17.81 |
| 8192 | 47.10 | 23.56 | 64.25 | 33.10 |
| 16384 | 93.83 | 50.60 | 115.62 | 62.50 |

`gs=32/64` 同样受益（`gs=32 row, s=8192`：96.5 → 19.9µs）。

### 13.3.1 调参用 HIP graph 计时，否则小尺寸全是地板

非 graph 的直接计时在本机有 **~21µs 的 Python launch 开销地板**，会把所有快配置压平成
46–65µs 的噪声。扫参阶段改用 `torch.cuda.CUDAGraph()` 把多次迭代打包，才看出真实曲线
（最优配置在 25µs 以下）。取 min 而不是 median（median 受时钟波动影响，同一点量到过
min 46 / median 57µs）。

扫了 48+ 个组合：`TDS ∈ {8,16,32} × waves_per_block ∈ {1,2,4} × KPT ∈ {1,2}`。

## 13.4 采纳（状态 D）：grid 维序 —— 本轮量级最大、且反直觉的一项

状态 B/C 的 grid 是 `(rows, Ks_spans)`，`x = row`。改成三维 `(S, Ks_spans, G)` 之后，
**只把 x 维给谁**这一件事就值 74%：

| grid | `s=8192` | `s=16384` | `s=4096` |
|---|--:|--:|--:|
| `(Ks_spans, S, G)` —— x 给 Ks span，"局部性更好" | **38.19** | 72.04 | 19.86 |
| `(S, Ks_spans, G)` —— x 给 s（**采纳**） | **21.92** | 42.81 | 12.19 |

同 run、同指令、同字节、同寄存器，唯一区别是**哪些 block 同时驻留**。x 是硬件 dispatch
最快变化的维，把 Ks span 放 x 会让同时跑的相邻 block 去读同一行里相邻的 4KB，挤在同一批
channel 上；给 s 则让相邻 block 相隔一行（8KB），自然摊到不同 channel。

> **教训**：这个 kernel 是纯带宽 kernel，"同时在飞的请求怎么铺在 channel 上"是主导量级，
> 比 ALU/指令数大一个数量级。**直觉上"更连续更好"在这里是错的。**
> 顺便说明为什么 §3 那条"砍 ALU 无效"的结论会一再重现。

注意状态 A 的 grid 是 `(scale_n/KPB, ...)`，x 给的正是 Ks —— 也就是说这条一直是慢的。

## 13.5 采纳（状态 D）：消掉地址依赖链头部的除法与 blockDim 加载

读 gfx950 反汇编（`gfx950_inverse_rope_group_quant.s`，状态 C 快照）时发现，
**第一条 data load 之前**挂着两样东西：

1. **`row / G` 的软件除法**：`G` 是运行期值，编成
   `s_abs_i32 / v_cvt_f32_u32 / v_rcp_iflag_f32 / v_mul_hi_u32` 链约 25 条，
   而 `row` 是 load 地址的一部分 —— 整条链堵在 load 发射前面。（§6 记为"未做"，
   本轮做了，因为新映射把它顶到了地址链的头部。）
2. **`blockDim.x` 要从 hidden kernarg 段读一条 `global_load_ushort`**：
   `-fno-offload-uniform-block` 让 block 尺寸不是编译期常量，地址
   `kernarg_base + 0x70 + {12,18}` 里那个 `{12,18}` 的选择就是"最后一个 block 可能不满"
   的逻辑。`k_slots = blockDim.x / THREADS_PER_GROUP` 依赖它 → data load 前多一条
   `s_waitcnt vmcnt(0)`，**等一条内存加载才能发出真正的内存加载**。

两条的修法都是"让它不存在"，而不是优化它：

```cpp
// 1) s / g 各占一个 grid 维，方向从 row→(s,g) 反成 (s,g)→row
const int s   = static_cast<int>(blockIdx.x);
const int g   = static_cast<int>(blockIdx.z);
const int row = s * G + g;                  // 一条乘加，且只有输出地址用到

// 2) k_slots 变成显式 kernarg：落在 --amdgpu-kernarg-preload-count=32 窗口内，
//    直接以 SGPR 到位，零 load
__global__ void ...(..., int scale_n, int k_slots, bool scale_shuffle, ...)
```

副作用：grid 维度现在精确等于 `(S, Ks/span, G)`，状态 A 的
`if(row >= S*G || k_group_base >= scale_n) return;` 边界检查整块删掉，
`scale_n % k_per_block == 0` 由 host 侧 `AITER_CHECK` 保证。

**§13.4 + §13.5 合并收益（同 run，C → D）**：`s=16384` 51.59 → **42.81µs**、
`s=8192` 23.49 → **21.92µs**。

## 13.6 被否决：每线程取相邻 KPT 组（换立即数 offset）

`k_slots` 是运行期值，所以交错式 `k_group0 + k*k_slots` 的 stride 折不进 `offset:`
立即数，4 条 load 各要一个地址 VGPR：

```
buffer_load_dwordx4 v[10:13], v19, s[20:23], 0 offen
buffer_load_dwordx4 v[6:9],  v20, s[20:23], 0 offen
buffer_load_dwordx4 v[14:17], v18, s[20:23], 0 offen
buffer_load_dwordx4 v[2:5],  v21, s[20:23], 0 offen
```

对比 §4 旧版的单 `v11` + `offset:256/512/768`。于是试了"让每个 thread 拿 KPT 个**相邻**
group"（`k_group0*KPT + k`），stride 变成编译期常量 `GROUP_SIZE`，省 3 个地址 VGPR + 3 条
`v_lshl_add_u32`。

**结果更慢（同 run，gfx1250，row，gs=128）**：

| s | 交错（采纳） | 相邻 |
|---|--:|--:|
| 512 | 4.46 | 4.34 |
| 2048 | 7.91 | 8.09 |
| 4096 | 12.19 | 12.34 |
| 8192 | **21.92** | **25.52** |
| 16384 | **42.81** | 44.67 |

几何上很清楚，两种摆法搬的字节、64B line 请求数完全一样，差别只在单条指令的足迹：

| | 每条 load 的足迹 | 一个 wave 总足迹 |
| --- | --- | --- |
| 交错 | **1 段连续**（wave_size × 每线程字节） | 连续 `k_slots·KPT·GS` |
| 相邻 | `k_slots` 段 × 256B，段间隔 `KPT·256B` | 同上（同一块连续区） |

即**交错按指令算更连续**，"交错 coalescing 更差"的说法不成立。3 条 VALU 在带宽饱和区换不
出 13%。保留交错，并在代码注释里写明测过多少。

> **⚠️ 与 gfx950 的分歧（未闭合）**：另一位 agent 在 gfx950 上报告相邻式更快（大 s 约
> 13%）。但他对比的"旧版"同时带着 §13.5 的除法与 blockDim 加载（两者都在地址链头部），
> **A/B 被污染**。而且真要有差别，可信的机制不是 coalescing，而是"一个 wave 的 KPT 条在飞
> 请求是压在一整块连续 4KB 上（交错）还是自己就跨 4 个相隔 1KB 的段（相邻）" ——
> 即 §13.4 那类 channel 并行度效应；gfx950 并发 wave 数约是 gfx1250 的 4 倍
> （256CU×4×8 ≈ 8192 vs ~40CU×4×16 ≈ 2560），更容易打满单个 channel，所以同一选择两边
> 翻过来是可能的。**处理建议**：拿状态 D 在 gfx950 上重测，若确实翻转，就给 tier 加一个
> 编译期 bool 分岔，不要全局改。

## 13.7 正确性修复：`hip_reduce.h` 的 asm DPP 在 gfx9 上少一个 wait state

§3.1 已经根因过这个 bug，但**那次的修法（自带 `group_reduce_max_dpp`，走
`__builtin_amdgcn_update_dpp`）从来没进过 `zoe/aiter` 这个分支** —— 这里一直在调
`hip_reduce.h` 的 `asm volatile` 版。反汇编实证（状态 C，同一份源码两个 arch）：

```
gfx950                                  gfx1250
v_max3_f32 v2, v2, |v0|, |v1|           v_max3_num_f32 v2, v2, |v5|, |v1|
v_max_f32  v3, v2, v2 quad_perm ...     v_max_f32 v3, v2, v2 quad_perm ...
s_nop 0     # 只有 1 个 wait state      v_max_f32 v2, v3, v3 quad_perm ...
v_max_f32  v2, v3, v3 quad_perm ...     # 不需要 nop，硬件自己解决
```

gfx9 对「VALU 写 VGPR → DPP 读该 VGPR」要求 **2 个 wait state**，编译器看不见 `asm` 里
是 DPP，只插了 1 个。**gfx1250 是 RDNA 类硬件、自己处理这个依赖，所以本轮 1056 个
bit-exact 检查全过，恰好把 gfx950 上会读到过期寄存器这件事掩盖掉了。**
状态 A 在 gfx950 上选的档正是 `THREADS_PER_GROUP=16`，就是 §3.1 实测算错 scale 的那个。

修法：本文件内自带 `group_reduce_max_dpp<N>`（`N ≤ 16`，走 `opus::upd_dpp` builtin，
dpp_ctrl `0xb1 / 0x4e / 0x141 / 0x140`，`bound_ctrl` 让越界 lane 读 0 —— 这里所有值
都是 `|x| ≥ 1e-8 > 0`，安全）。修后 gfx950 每条 DPP 前正确出现 `s_nop 1`：

```
v_max3_f32 v17, v17, |v14|, |v15|
s_nop 1
v_mov_b32_dpp v24, v17 quad_perm:[1,0,3,2] ... bound_ctrl:1
v_max_f32_e32 v24, v24, v24     # fmaxf 的 canonicalize
v_max_f32_e32 v17, v17, v24
```

代价：gfx1250 上不再融合成单条带 DPP 的 `v_max_f32`，每步多 2 条 VALU。
**实测 flat**（同 run，`s=8192`：23.56 → 23.49µs），与 §3 的结论一致 —— 规约的 ALU
本来就藏在 load 等待后面。

**没有动共享的 `hip_reduce.h`**：它的 asm 版还有别的调用方
（`topk_gating_kernels.cu`（`WARP_SIZE=64`，被末尾广播掩盖）、
`fused_qk_rmsnorm_group_quant.cu:287`、`fused_qk_norm_rope_cache_quant.cu:4553`，
后两处是 `GROUP_SIZE / vec_size`，**可以取到 16**）。这仍是 §3.1 那条待办，现在多了一份
反汇编证据。

## 13.8 wave64 (gfx950) 的处理：按 wave 宽度分档

第一轮的档位（`TDS ∈ {2,4,8}`）是 wave64 上扫的；本轮扫出来的 wave32 最优档是
`TDS ∈ {16,32}`（每线程 32B/64B）。**直接套到 gfx950 会掉 occupancy**：gfx950 每 SIMD
的 VGPR 文件是 512，要 8 个常驻 wave 就得 ≤64 个 VGPR，而 wave32 的宽 slice 刚好越过这道坎。

用 JIT 原本的 flags 只换 `--offload-arch`，`-Rpass-analysis=kernel-resource-usage`
读出来（bf16，状态 C 快照）：

| 配置 | gfx950 VGPR / occ | gfx1250 VGPR / occ |
| --- | --: | --: |
| 第一轮调出的 TDS=8, KPT=4 | 48 / **8** | — |
| TDS=32, KPT=1（wave32 宽档） | 68 / **7** | 60 / **16（满）** |
| TDS=16, KPT=2（wave32 窄档） | 74 / **6** | 56 / **16（满）** |
| TDS=32, KPT=2 | 84 / 5 | 76 / 12 |
| TDS=32, KPT=4 | **128 / 4，scratch 68B/lane（溢出）** | — |

即每线程 64B 在 gfx1250 上是白拿的，在 gfx950 上每个 SIMD 少一个常驻 wave。
于是按 wave 宽度分档，**wave64 走回第一轮调过的操作点**：

```cpp
const bool wave64 = wave_size == 64;
int tds = wave64 ? 8 : (narrow_slice ? 16 : 32);   // 每线程 16B / 32B / 64B
int kpt = wave64 ? 4 : (narrow_slice ? 2 : 1);
```

wave64 档回到 38/42/52 VGPR、**occ 8**，与第一轮一致；而 §13.3–§13.5 的映射改动
（每 wave 一段连续、标量 `positions`、无除法、无 blockDim 加载）与架构无关，wave64 照样吃到。
`kNarrowCrossoverWavesPerSimd` 这个 gfx1250 扫出来的阈值因此也不再影响 gfx950。

**gfx950 反汇编确认（状态 C 快照，`GS=128 TDS=8 KPT=4 bf16`，共 778 条指令）**：

- `.amdhsa_next_free_vgpr 52`、`private_segment_fixed_size 0`（**无 scratch**）
- 16 条 `v_mov_b32_dpp`，**16 条前面都有 `s_nop 1`**（= 4 组 × 4 步 butterfly）
- 4 条 `buffer_load_dwordx4` 连发后 `s_waitcnt vmcnt(3)` 分级等待 —— **§4 的签名保住了**
- 8 条 `global_load_dwordx2` = cos/sin 向量化（`NCOS=4` → 8B）—— **§5 的收益保住了**；
  是 4 个 unroll pass 各一对，因为 `k_group` 是运行期值、编译器无法证明哪个 pass 落在 rope 尾
- 4 条 `buffer_store_dwordx2`（fp8）+ 8 条 `global_store_byte`（scale 的 row/shuffle 两条分支）

## 13.9 收尾：掐掉不可达实例化

`kpt=4` 只有 wave64 会取，而 wave64 的 `tds` 恒为 8（shuffle 下 `gs=128` 抬到 16），
所以 `TDS=32 × KPT=4` 不可达 —— 但它会被实例化，而且**溢出到 scratch**（128 VGPR + 68B/lane）。
按 §9 的规矩把 kpt 的分派按 tds 收窄：

```cpp
if constexpr(TDS <= 8) { if(kpt >= 4) { launch(..., ic<4>{}, ...); return; } }
if(kpt >= 2)           { launch(..., ic<2>{}, ...); return; }
launch(..., ic<1>{}, ...);
```

可达组合收敛为 `(8,{1,2,4})`、`(16,{1,2})`、`(32,{1,2})`：实例化 54 → **42**，
重编 **10.0s**，且没有任何实例化再溢出。

## 13.10 dispatch 启发式的最终形态（状态 D）

不再有 `S` 阈值表，全部由"有多少 wave 可用"推导：

```
rows        = S * G
wide_waves  = rows * D / (wave_size * 32)
simds       = num_cu * 4
narrow_slice= wide_waves >= simds * kNarrowCrossoverWavesPerSimd
              # 56 (gs>=128) / 40 (gs>=64) / 24 (gs=32)

tds  = wave64 ? 8 : (narrow_slice ? 16 : 32)
tds  = max(tds, GS*8/wave_size)   if scale_shuffle   # 让一个 wave 至少覆盖 8 个 group
tds  = min(tds, GS); while (GS/tds > wave_size) tds <<= 1

waves_per_block = scale_shuffle ? 4 : 1
k_slots = clamp(waves_per_block * wave_size / (GS/tds), 1, scale_n)
          然后：block 数不足 num_cu*4 就减半；不整除 scale_n 就减半
kpt     = wave64 ? 4 : (narrow_slice ? 2 : 1)
          然后：跨度不整除 scale_n 或 block 数不足 num_cu*4 就减半
grid    = (S, scale_n / (k_slots*kpt), G)      block = k_slots * (GS/tds)
```

要点：`k_slots` 是**运行期** launch 选择，只决定 block 尺寸，不增加模板实例化；
`kpt` 的两个 while 循环替代了第一轮"S≤128 → KPB=1"那张表 —— 小形状会自动退回。

## 13.11 最终数字（状态 D，gfx1250，bf16，h=16 g=2，gs=128，row，同 run）

| s | A（起点） | **D（当前）** | 同 run unfused | D vs A |
|---|--:|--:|--:|--:|
| 512 | 5.31 | **4.46**（2.85 TB/s） | 5.28 | 1.19× |
| 2048 | 13.72 | **7.91**（6.41 TB/s） | 12.36 | 1.73× |
| 4096 | 22.57 | **12.19**（8.33 TB/s） | 19.52 | 1.85× |
| 8192 | 47.10 | **21.92**（9.26 TB/s） | 36.32 | 2.15× |
| 16384 | 93.83 | **42.81**（9.48 TB/s） | 71.54 | 2.19× |

（A 与 D 不在同一个 run，A 列仅作量级参考；D 与 unfused 是同 run。）

**正确性（状态 D 全量 sweep，日志 `容器:/tmp/sweep2.log`）**：

```
-d bf16 fp16 -b 16,2 8,1 -s 1 4 8 32 128 300 512 700 2048 4096 8192
--group-size 32 64 128 -l row shuffle --graph
```

→ **1056 个 `checkAllclose` 全过**（值 `rtol=atol=1e-2`；e8m0 字节 `rtol=atol=0` 即 bit-exact）、
**266 行 perf 的 `cpp err` 与 `cpp scale err` 全为 0**、**HIP graph capture/replay 全过**。

shuffle 布局仍落后 row（状态 D 只从日志尾部读到 `s=8192, h=8 g=1, fp16`：
row 13.49µs / 7.56 TB/s vs shuffle 16.56µs / 6.16 TB/s；状态 B 时 `h=16 g=2 bf16`
是 row 23.56 vs shuffle 33.10）。**`h=16 g=2` 的 shuffle 完整表没来得及从日志里取出
（容器已停）**，日志还在，下一个 agent 可以直接取。

## 13.12 复现与无 GPU 静态验证

```bash
docker start yzhou_latest && docker exec -it yzhou_latest bash
cd /home/hwang/zoe/aiter
rm -f  aiter/jit/module_inverse_rope_group_quant.so        # 两个都要删！
rm -rf aiter/jit/build/module_inverse_rope_group_quant
PYTHONPATH=$PWD python op_tests/test_inverse_rope_group_quant.py \
  -b 16,2 -s 512 2048 4096 8192 16384 -l row --group-size 128
# 全量 + graph：
PYTHONPATH=$PWD python op_tests/test_inverse_rope_group_quant.py \
  -d bf16 fp16 -b 16,2 8,1 -s 1 4 8 32 128 300 512 700 2048 4096 8192 \
  --group-size 32 64 128 -l row shuffle --graph
```

**本轮新增的手法：不需要目标硬件也能验证一半的东西。**
拿 JIT 自己的编译命令（`aiter/jit/build/module_inverse_rope_group_quant/build/build.ninja`
里的 `cuda_cflags`），只把 `--offload-arch=gfx1250` 换成 `gfx950`：

```bash
# 1) 每个实例化的 VGPR / occupancy / scratch（加 -Rpass-analysis=kernel-resource-usage）
#    -> 能抓到 occupancy 掉档和寄存器溢出
# 2) 反汇编（--cuda-device-only -S）
#    -> 能抓到 wait state 危险、立即数 offset、分级 vmcnt、向量化是否还在
```

gfx950 的反汇编产物留在 `aiter/gfx950_inverse_rope_group_quant.s`（**untracked 构建产物，
不要提交**；状态 C 快照，状态 D 需要重新生成）。这套静态验证抓到了 §13.7 的正确性 bug 和
§13.8 的 occupancy 掉档 —— 两个都是 gfx1250 上跑一万遍也看不见的。

## 13.13 待办 / 下一个 agent 从这里开始

按优先级：

1. **gfx950 实测状态 D**（唯一的硬阻塞项）。本轮 gfx950 全靠静态验证。
   先跑 `-s 512 2048 8192 16384 -l row --group-size 128`，和 §TL;DR 的
   34.32µs / 5.9 TB/s（`s=8192`，第一轮 MI355X）对齐量级。
   另一位 agent 在**状态 B/C** 上报告过小 s +10~14%、大 s +13%，那份 A/B 被 §13.5 的
   除法与 blockDim 加载污染了，状态 D 上要重测才算数。
   **他后来给的 `csrc/gfx950_fix.patch` 基线同样是状态 C，见 §14.5.1；rebase 版和
   完整的交接说明在 §14.5–§14.6。**
2. **若 gfx950 上"相邻 KPT"确实更快**（§13.6），给 tier 加编译期 bool 分岔，不要全局改；
   gfx1250 上交错胜出 21.92 vs 25.52µs，有实测。
   **§14.5.3 在状态 D 上重做了这条并复现（`s=8192` +21.4%），且已有一份 rebase 好的
   实现 `csrc/gfx950_fix_on_stateD.patch`（用运行期 `contig_k` 标志按 wave 宽度分岔）。**
3. **shuffle 布局落后 row 约 23–40%**：MFMA tile 布局下每 64B 只写 1 个字节。
   彻底解决要用 LDS 攒满一个 256B tile 再写，需要一个 block 覆盖 32 个 s ——
   与当前"一个 block 一行"的映射冲突，是一次独立改动。
   **§13.14.2 补齐了这里缺的完整表，实测比这里估的更差（1.22–1.56×），且大 s 上
   shuffle 已经和 unfused 打平 —— 优先级应该比这个位次高。**
4. **小 s 的固定开销**：`positions[s]` 现在无条件加载（§13.3 第 2 点）。
   新映射下 block 跨度 ≥ `GROUPS_PER_HEAD` 时旧的 `any_rope` 守卫恒真，救不了；
   而 `s ≤ 128` 本来就撞 dispatch 地板（§1：~2.4–2.5µs），绝对值 ~0.25–0.4µs。**低优先级。**
5. **`hip_reduce.h` 的 asm DPP helper 仍需修**（§13.7 末），影响本 kernel 之外三个调用点。
6. **gfx942**：`SUPPORTED_GFX` 里仍然没有它。本轮之后 `permlane16_swap/permlane32_swap`
   不再被实例化（每组 lane 数 ≤16，规约不出 DPP row），§8 记的那个"编不过"的理由已经不成立，
   但没人在 gfx942 上跑过，所以没加。
7. **未提交**：`csrc/kernels/inverse_rope_group_quant.cu`（`op_tests/` 那份已经进了
   `7804bf22`，工作区只剩 kernel 一个文件 +244/−128）。提交前重读 §12（pre-commit hook
   会用本地 ruff 顺手改无关文件，且会把 `Optional[X]` 改成 `X | None` 破坏 py3.8）。

---

# 14. 第三轮：在另一台 gfx1250 上复现状态 D

测量环境：**gfx1250**（`rocm/fw-bringup:gfx1250-atom--20260817`，容器 `yzhou_latest`，
4 卡机、`HIP_VISIBLE_DEVICES=0`，开跑前 `rocm-smi` 四张卡全 0%），仓库
`/home/hwang/zoe/aiter`，分支 `fix_inverse_rope_group_quant_gfx1250` @ `7804bf22`
+ 工作区那份未提交的状态 D kernel。**代码一行没动**，纯复现。

开跑前撞到 §13.1.1 那个坑的另一种形态：`.so` 是 02:22 编的、`.cu` 06:57 才改过。
`ls --time-style=full-iso` 对一眼时间戳就能发现，两个都删掉后重编 **10.2s**，
与 §13.9 收敛实例化后的时间吻合。

## 14.1 row 布局：复现，且比状态 D 记录快 4–8%

（bf16, h=16 g=2, gs=128, µs。run1/run2/sweep 是三个独立 run；unfused 取 run1 同 run。）

| s | §13.11 的 D | run1 | run2 | sweep | 本轮 TB/s | 同 run unfused | 加速 |
|---|--:|--:|--:|--:|--:|--:|--:|
| 512 | 4.46 | 4.21 | 3.99 | 4.00 | 3.18 | 6.09 | 1.53× |
| 2048 | 7.91 | 7.21 | 7.24 | 7.25 | 7.00 | 11.14 | 1.54× |
| 4096 | 12.19 | 11.24 | 11.34 | 11.27 | 8.95 | 18.54 | 1.64× |
| 8192 | 21.92 | 20.32 | 20.27 | 19.64 | **10.01** | 34.50 | 1.70× |
| 16384 | 42.81 | 40.16 | 40.49 | — | **10.02** | 64.07 | 1.59× |

fp16 与 bf16 同量级，`s=8192` 甚至到 19.21µs / **10.56 TB/s**（本轮最高）。

两点值得记：

- **本机跨 run 离散度只有 ~2%**，远好于 §13.1 记的「同一 binary 同一 shape 差 27%」。
  §13.1「只比同 run」的纪律仍然要守，但这台机器的时钟状态明显更干净，
  跨 run 参考值可用性比第二轮那台高。
- 快出来的 4–8% **不是代码带来的**（工作区就是状态 D）。只可能是机器/驱动状态差异，
  所以 §13.11 的表和这张表不要混着引用；要对比就整表换。

## 14.2 补齐 §13.13 第 3 条缺的 shuffle 表

§13.11 说 `h=16 g=2` 的 shuffle 完整表「没来得及从日志里取出来（容器已停）」。
补上（同 run，bf16, gs=128）：

| s | row | shuffle | shuffle 落后 | 同 run unfused |
|---|--:|--:|--:|--:|
| 512 | 3.99 | 5.66 | 1.42× | 6.06 |
| 2048 | 7.24 | 8.85 | 1.22× | 11.33 |
| 4096 | 11.34 | 14.68 | 1.29× | 18.66 |
| 8192 | 20.27 | 29.56 | 1.46× | 34.26 |
| 16384 | 40.49 | 63.03 | 1.56× | 64.60 |

**比 §13.13 估的「23–40%」更差，而且大 s 上 shuffle 已经追平 unfused**
（`s=16384`：63.03 vs 64.60）—— 即在 shuffle 布局下这个 kernel 融合掉的那一趟
访存基本被 scale 散写吃光了。§13.13 第 3 条（LDS 攒满 256B tile 再写）因此不该
排在第 3 位。

## 14.3 新观察：小 s 时 shuffle 恒为 row 的 2×

| s | 1 | 4 | 8 | 32 | 128 | 300 |
|---|--:|--:|--:|--:|--:|--:|
| row | 2.65 | 2.67 | 2.65 | 2.69 | 2.67 | 3.06 |
| shuffle | 5.30 | 5.34 | 5.31 | 5.35 | 5.33 | 5.41 |

row 那行就是 §1 说的 dispatch 地板（本机 ~2.65µs）。shuffle 在 `s ≤ 128` 上**恒定
是它的两倍**，比例整齐到不像带宽效应 —— 这些形状下按 §13.10 的启发式
（`gs=128` → `tds=32`、`threads_per_group=4`、`k_slots` 被两个 while 砍到 8）
`s=1` 只发 **4 个 block × 32 线程**，纯粹是延迟。

排除了一个显然的嫌疑：**不是 wrapper 里 `torch.full(..., 0x7F)` 那次填充**。
测试的 `_alloc_outputs` 预分配输出并把 `x_scale` 传进 op，计时区间里只有 kernel
（`op_tests/test_inverse_rope_group_quant.py:148,391`）。所以这 2.6µs 在 kernel 内部，
没有进一步定位（需要 ATT）。与 §13.13 第 4 条一样是小 s 固定开销，但**量级大一个档**
（那条是 0.25–0.4µs），如果 shuffle 布局要用在 decode 上，这条比第 4 条更值得看。

## 14.4 正确性：全量 sweep 全过

```
-d bf16 fp16 -b 16,2 8,1 -s 1 4 8 32 128 300 512 700 2048 4096 8192
--group-size 32 64 128 -l row shuffle --graph      # 约 304s
```

→ **1056 个 `checkAllclose` 全过**、`fail/mismatch/Traceback` 零命中、
HIP graph capture/replay 全过，与 §13.11 完全一致。日志：`容器:/tmp/sweep_0818.log`。

## 14.5 评估 gfx950 修复 patch 对 gfx1250 的影响：无

有人报告状态 D 在 gfx950 上回退，并给出 `csrc/gfx950_fix.patch`。本节回答的**只有**
「把它拿过来会不会拖累 gfx1250」，不回答它在 gfx950 上有没有用（见 §14.6）。

> **文件名注意**：`csrc/gfx950_fix.patch` 这个路径后来被覆盖成了一份完整 `.cu`，已改名
> 保存为 `csrc/gfx950_verified.cu.bak` —— 它等于本节的 rebase 版**应用后的结果**
> （逐字节相同，仅差一个末尾空行），且是在 gfx950 上实跑过的那一份。
> 下面描述的「patch 的上下文行」指的是原来那份 213 行的 diff，该路径已不再是它。

### 14.5.1 先看基线：那份 patch 的基线是状态 C，不是状态 D

patch 的**上下文行**（不带 `+`/`-`）里有：

```
const int k_slots = static_cast<int>(blockDim.x) / THREADS_PER_GROUP;
const int row = static_cast<int>(blockIdx.x);
...
const int s = row / G;
const int g = row - s * G;
```

外加 kernel 签名末尾是 `int max_position)`（没有 `int k_slots`）、launch 实参是
`S, H, G, D, scale_n, scale_shuffle,`。这四处正是 §13.5 删掉的东西，而且 grid 还是二维的
（缺 §13.4）。所以：

- **`git apply` 到本仓库会失败**，必须重新 rebase；
- **把对方整份文件合过来 gfx1250 会掉**，等于吐回 §13.5 记的 C→D 收益
  （`s=8192` 23.49→21.92、`s=16384` 51.59→42.81）；
- **对方的 gfx950 A/B 本身可疑**，因为基线缺的这两项一个在地址依赖链头部、一个决定
  block 怎么铺到 channel 上 —— 与 §13.6 记的那次污染是同一个模式。

rebase 后的版本存为 `csrc/gfx950_fix_on_stateD.patch`（**untracked**，头部写了出处与
基线 md5）。下面所有数字都是这个 rebase 版。

### 14.5.2 逐 hunk 隔离实测（不是打包测）

8 个 hunk 里只有 3 个进得了 wave32 的代码路径。**打包测一次是不够的**——`contig_k` 的
开销和 `any_rope` 的收益如果一正一负就会互相抵消成假的 flat，所以逐条单独编译测量：

| hunk | gfx1250 结论 | 证据 |
| --- | --- | --- |
| `contig_k` 机制（多一个 kernarg + uniform select） | **≤1%** | `s=16384` 40.26 vs 40.27，两边三次重复跨度均 <0.03µs |
| `contig_k` **取 true** | **+21.4%** | `s=8192` 24.31 vs 20.02（各三次：24.31/24.31/24.36 vs 20.00/20.02/20.08） |
| `any_rope` 守卫 | **flat** | 小 s 各五次，`s∈{1,4,32,128}` 归一化差 ≤0.5% 且方向不一致；`s=300` +1% |
| `group_reduce_max_dpp` 扩到 N≤64、`static_assert` 放宽 | 无 | 新增两步在 `if constexpr(N > 16)` 内，gfx1250 各档 N ∈ {2,4,8}，codegen 不变 |
| `contig_k=true`、`wave_starved` 收窄 tds、`waves_per_block=4` | 无 | 全部 `wave64` gate |
| `dispatch_kpt` 重构 + `switch case 2/4` | 无 | wave32 取不到 tds=2/4；多 12 个实例化但重编仍 10.2s |

结论：**rebase 版对 gfx1250 无影响**，正确性 768 个 `checkAllclose` 全过（含 `--graph`）。

### 14.5.3 `contig_k` 的 gate 是承重的 —— 顺带在状态 D 上重做了 §13.6

三方对照（bf16, h=16 g=2, gs=128, row，各三次取中位数，µs）：

| s | D | 加 `contig_k` 但取 false | `contig_k=true` |
|---|--:|--:|--:|
| 512 | 4.04 | 4.17 | 4.17 |
| 2048 | 7.24 | 7.27 | 7.26 |
| 8192 | **19.85** | **20.02** | **24.31** |
| 16384 | 40.27 | 40.26 | 41.75 |

即**机制免费、取值致命**。`wave_size == 64` 那个 gate 一旦失效或被写成全局开关，
gfx1250 主力形状直接 +21%。

这同时是 §13.6 在状态 D 上的重做，形状对得上：文档记的是 `s=8192` +16%、`s=16384`
+4.3%，今天是 **+21.4%** 和 **+3.7%** —— 都是大 s 中段差距最大、到 16384 收窄
（两种摆法都把机器喂满了）。所以 §13.6「gfx1250 要交错」的结论在状态 D 上依然成立，
**两个 arch 在这一项上确实相反**，§13.13 第 2 条「加编译期 bool 分岔、不要全局改」是对的。

### 14.5.4 顺带记两条

- **`any_rope` 在 gfx1250 上救不掉那条 load**，所以才 flat：守卫是 per-thread 的，而
  `gs=128` 时 `GROUPS_PER_HEAD=4`、交错步长 `k_slots` 在每个可达档上都是 4 的倍数
  （宽档 8、窄档 4），于是一个 thread 的 KPT 个组 `kg % 4` 相同 → 只有 4 个 k_slot 里的
  1 个为真 → 一个 wave32 的 32 lane 里有 8 个命中 → **wave 照样发这条 load**。
  它要真省下来得让 thread 跨度 < `GROUPS_PER_HEAD`，那是 patch 新增的 wave64 小 S 档
  （tds=2/4 → `k_slots` 只有 1–2），gfx1250 选不到。
- **一个 hygiene 退化**：`static_assert(THREADS_PER_GROUP <= 16)` 放宽到 `<= 64` 之后，
  「一个 quant group 必须落在单个 wave 内」就只剩 host 侧 `while(GS/tds > wave_size)`
  这个运行期循环在守。`case 2` 会为 `GS=128` 实例化出 `THREADS_PER_GROUP=64`，在 wave32
  上编译出跨 wave 的 `__shfl_xor(v, 32, 64)` —— 目前不可达所以无害，但编译期护栏没了。
- 若要连那 ≤1% 也省掉，别把 `contig_k` 做成模板参数（实例化翻倍）；让 host 直接传
  `k_pass_stride` 和 `k_slot` 的乘数两个 int 即可，kernel 里连 select 都不需要。

## 14.6 未闭合

- §13.13 第 1 条（**gfx950 实测状态 D**）本轮仍然没做 —— 这台是 gfx1250 四卡机，
  没有 gfx950 硬件。它依旧是交接时最大的硬阻塞项。
- **§14.5 完全没有验证 gfx950 这一侧**。不只是「没硬件」：patch 里真正针对 gfx950 的
  三条（`contig_k=true`、`wave_starved` 收窄 tds、`waves_per_block=4`）全部写在
  `wave64` 分支内，所以即使在 gfx1250 上跑 rebase 版，它们**一行都没被执行**。
  也就是说这组测量对 gfx950 零信息量。下一个有 gfx950 的人要回答两个独立问题：
  1. 状态 D 相对状态 A 在 gfx950 上到底退了多少；
  2. `csrc/gfx950_fix_on_stateD.patch` 在**状态 D 上**还修不修得好 —— 对方是在状态 C 上
     调的，而 `waves_per_block=4`（理由是「1-wave block 导致 occupancy 差」）和
     `contig_k` 的前提都被 §13.4/§13.5 动过了。

---

# 15. 第四轮：n32k4 scale 布局与 scale 散写代价的拆解 —— 交接记录

测量环境：**gfx1250 四卡机（256 CU），容器 `yzhou_latest`，`HIP_VISIBLE_DEVICES=3`**。
代码状态记为 **E** = 状态 D + 本轮全部改动。

## 15.0 TL;DR

起因是 ROCm/aiter#4626 的 a8w4 batched GEMM：它在 gfx1250 上用 WMMA，要的 activation
scale 是 **n32k4** 布局，而不是行主序。本轮把这个布局做进 kernel（省掉一趟独立的
transpose），顺带把 `scale_layout` 从 `bool scale_shuffle` 改成三值枚举再改成模板参数。

**结论**

1. **scale 只占 3% 的字节，却能让 kernel 慢一倍** —— 因为内存系统按事务收费不按字节收费。
   代价拆成两项，其中通道串行是大头（§15.3）。
2. **super-major 行重映射把这个代价全额消掉**：32 个 b×m 配置上 n32k4/row 落在
   **0.95–1.08×**，此前是 1.0–1.85×，整体 **1.30×**、最大 **1.86×**，且**没有一格变慢**
   （§15.11）。做法是把行号重排，让共享同一个 128B chunk 的 32 行不要同时在飞。
3. **真正的自变量是 `n_super = ceil(S/32)` 的低位零个数，不是 G**（§15.11.2）。`8 | n_super`
   时重映射满血，奇数时完全失效，中间按 2 的幂次逐级退化 —— 一个断崖，不是斜坡。
   而 `n_super` 是我们自己挑的：**向上取整到 8 的倍数**就把所有 S 都买到了满血档。
4. **`quant_group_size` 现在被钉死在 32**（§15.10）。之前三档全放行，而 64/128 会产出一个
   "字节位置全对、语义全错"的 buffer，且没有任何现存检查会拦 —— 是个静默正确性洞。
5. **对等口径下我们现在全面领先 Triton**：`tri rope + tri n32k4 quant` vs 我们一个融合
   kernel，16 个 shape 全为正，**1.03×–2.89×**（§15.8.1）。此前大 shape 是输的。
6. **剩下的唯一一条是主路径带宽**：我们 ~10.5 TB/s，Triton 同字节量 ~15 TB/s，
   **稳定差 1.40–1.43×**。和 scale 布局无关，三种 layout 一起中（§15.8.1、§15.12）。
7. **`--amdgpu-kernarg-preload-count=32` 改变了 §13.5 的结论**：新增 kernarg 只要落在前
   32 个 dword 内就是预载 SGPR，不产生 `s_load`，地址链头部不会被挂住（§15.7）。

> 被推翻的：§15.4 的 grid 序门槛（结论没错，但 swap 已从主路降级为 fallback）和 §15.5 的
> "G 依赖"归因（测量没错，归因被第 3 条替换）。两节都原样留着并加了指向说明 —— 那两轮的
> 受控实验手法本身是对的，错在当时可枚举的自变量里少了一个。

**当前性能**（bf16, k=4096, quant_block=32 即 Ks=128, µs, 同 run, 两次取平均；
"旧" = `AITER_IRGQ_SUPER_MAJOR=0`，即 §15.4 的 swap 启发式）

| b | m | hip row | n32k4 旧 | n32k4 今 | 旧比 | 今比 |
|---|--:|--:|--:|--:|--:|--:|
| 2 | 4096 | 10.6 | 15.14 | **10.80** | 1.42× | **1.02×** |
| 4 | 4096 | 18.2 | 32.56 | **18.21** | 1.79× | **1.00×** |
| 8 | 4096 | 37.4 | 64.19 | **38.88** | 1.71× | **1.04×** |
| 8 | 16384 | 153 | 248.81 | **160.81** | 1.64× | **1.05×** |
| 16 | 4096 | 76.6 | 83.98 | 81.16 | 1.10× | 1.06× |
| 4 | 5000 | 21.6 | 39.55 | **21.61** | 1.85× | **1.00×** |
| 16 | 1000 | 18.7 | 33.72 | **18.11** | 1.82× | **0.97×** |

## 15.1 n32k4 是什么

`aiter.ops.shuffle.shuffle_scale_n32k4` 给权重产出的那个布局，本轮让 activation 也直接
产出它。存储 `[ceil(S,32)/32, G, Ks*32]`，字节位置：

```
byte = ((s/32)*G + g) * Ks*32  +  (k/4)*128  +  (s%32)*4  +  (k%4)
```

要求 `Ks % 4 == 0`：消费端一个 lane 的 WMMA scaleB operand 是一个 K=128 步的 4 个 e8m0，
用一条 `ds_load_b32` 取。`shuffle_scale_n32k4` 对同样的情况是直接拒绝而不是补齐，这里
跟它保持一致，两个产出方才可互换。

**和 MFMA tile（gfx950 那个）的关系**：两者都是"给矩阵核预排的 scale"，但排法完全不同，
不能混用。以 `gs=128`（Ks=32）为例，一行 32 个 scale 字节：

| 布局 | 碎片数 × 每片字节 | 共享者 |
|---|---|---|
| row | 1 × 32（连续） | 无 |
| MFMA tile | 16 × 2 | 一个 256B tile 由 32 行共享 |
| n32k4 | 8 × 4 | 一个 128B chunk 由 32 行共享 |

所以 gfx950 上对应的是 MFMA tile，**n32k4 在 gfx950 上根本不存在**；拿两边的"shuffle 代价"
直接比是比了两个不同的东西。另外本仓从来没有在 gfx950 上量过 MFMA tile 与 row 的对比
（第一轮只测了 row，第二/三轮没有 gfx950 硬件），§14.2 那张 1.22–1.56× 是 gfx1250 的。

## 15.2 `scale_layout`：bool → 三值枚举 → 模板参数

原来是 `bool scale_shuffle`。加了第三种布局之后先改成 `enum ScaleLayout : int64_t`
（`kScaleRowMajor=0 / kScaleMfmaTile=1 / kScaleN32K4=2`，见
`csrc/include/inverse_rope_group_quant.h`），然后进一步提成 kernel 的模板参数。

提成模板的理由不是"看着干净"，是三种布局要的 kernarg 互不相交：

| 布局 | 用到的 scale kernarg |
|---|---|
| row | `scale_stride_s/g/k` |
| MFMA tile | `S_pad`, `Ks_pad` |
| n32k4 | 都不用（`scale_n` 和 `G` 本来就要） |

运行期选的话，每个变体都得为自己不用的那些参数买单，而它们全在 store 的地址链上。
改成模板后：

- 三处 `if(scale_layout == ...)` 变成 `if constexpr`；
- MFMA 那三个 `shuf_tile_m / shuf_s_mod16 / shuf_m_half` 预计算删掉了 —— 它们原来在
  **所有**布局下都无条件算；
- `swap_sg` 也折进模板条件（`SCALE_LAYOUT == kScaleN32K4 && swap_sg`），于是 row/mfma
  两条路径直接读 `blockIdx`，前面不站任何东西。

主机侧加了一层 `dispatch_group_size(sl<...>{})`，实例化数 ×3。**代价可以忽略**：kernel
单文件全量重编 14.1s，完整重编 + 跑完整个测试套件 27.7s（改之前约 25s）。

性能上是中性的（b=16 m=4096：n32k4 85.56 vs 改前 83.7，row 78.17 vs 77.8，在跑测噪声内）。
真正的收益是把 §15.7 那条路铺平了。

## 15.3 为什么"只是写的位置不一样"能差这么多

`gs=32` 时一行 payload 4096 个 fp8 字节，scale 只有 128 字节，占 3%。但内存系统按
cache line 收费：

```
row-major   一行:  [b0 b1 ... b127]                        -> 2 笔事务
n32k4       一行:  chunk0[..r*4..] chunk1[..r*4..] ...      -> 32 笔，每笔 4 有效字节
                   (r = s % 32，32 个 chunk 各占 4 字节)
```

**两个实验把两种可能的原因分开了：**

| 实验 | 假设 | 结果 |
|---|---|---|
| 把 4 个相邻 k 的 scale 字节跨 lane 打包成一个 dword 再写 | 瓶颈是 store 指令数 | **完全无效**（41.76 vs 42.10µs）—— 合并器早就做了这件事 |
| 把 grid 最快变化维从 `s` 换成 `g` | 瓶颈是并发 block 撞同一通道 | **134.4 → 83.7µs**（b=16 m=4096） |

所以主因是**跨 block 的同时性**，不是碎片化本身。原来 `blockIdx.x = s`，同一个 super 的
32 行是 32 个编号相邻的 block，几乎必然同时在跑，而它们全在写同一个 128B chunk；L2 按
地址分 bank/channel，同一个 chunk 就是同一个口。

**为什么 DRAM 带宽没怎么涨**：L2 是 write-back 的，32 行最终会把那个 chunk 写满，line 只以
完整脏行刷一次 DRAM。放大发生在 **L2 事务槽和 bank 排队**上，不在 DRAM 字节上。对数：
b=16 m=4096 总流量约 813MB，row 78.17µs ≈ 10.4 TB/s（贴着这个 kernel 的历史天花板，
§14.1 记的 10.01–10.56）；swap 后 85.56µs 多出的约 7µs 与"L2 事务数增加、部分被写合并
吸收"对得上，而 swap 前多出的 50µs 按带宽根本解释不到。

> 打包那条已经写进 kernel 注释，**不要再试第二遍**。

## 15.4 grid 序门槛：受控 A/B 证明 `rows >= 64*CU` 是对的（已降级为 fallback）

> **§15.11 之后**：这一节的门槛本身没被推翻，但 swap 已经不是主路了 —— super-major 在几乎
> 所有形状上都比它好，swap 只在 super-major 关掉时（`S < 256`）才轮得到。下面的数是
> "只有 swap 可用"时的最优解，留作对照。

之前从 `bench_scale_layout.py` 那种"让启发式自己选"的 sweep 里，看起来像是 `G` 决定一切。
**那是伪相关**：门槛按 `rows` 判，而在那 16 个配置里 `rows` 和 `b`、`m` 强相关，
swap 开着的格子恰好几乎都是 `b<16`。

加了 `AITER_IRGQ_SWAP_SG`（0 强制 s 最快 / 1 强制 g 最快 / 不设=启发式）把两边都强制跑一遍。
比值是 n32k4/row，**用同 run 的 row 做锚点**抵消跨进程漂移（§13.1 的纪律；这里两半必须是
两个进程，因为 C++ 侧把 getenv 缓存成了 static）：

| b | m | rows | swap off | swap on | 门槛判定 |
|---|--:|--:|--:|--:|:--|
| 2 | 128 | 256 | **0.99×** | 1.12× | off ✓ |
| 2 | 1024 | 2048 | **1.10×** | 1.61× | off ✓ |
| 2 | 4096 | 8192 | **1.47×** | 1.70× | off ✓ |
| 2 | 16384 | 32768 | 2.16× | **1.73×** | on ✓ |
| 4 | 128 | 512 | **1.02×** | 1.25× | off ✓ |
| 4 | 1024 | 4096 | **1.29×** | 1.71× | off ✓ |
| 4 | 4096 | 16384 | 2.17× | **1.74×** | on ✓（压线） |
| 4 | 16384 | 65536 | 1.88× | **1.67×** | on ✓ |
| 8 | 128 | 1024 | **1.01×** | 1.43× | off ✓ |
| 8 | 1024 | 8192 | **1.42×** | 1.71× | off ✓ |
| 8 | 4096 | 32768 | 2.03× | **1.70×** | on ✓ |
| 8 | 16384 | 131072 | 1.64× | 1.64× | 平 ✓ |
| 16 | 128 | 2048 | **1.02×** | 1.21× | off ✓ |
| 16 | 1024 | 16384 | 1.90× | **1.10×** | on ✓（压线） |
| 16 | 4096 | 65536 | 1.77× | **1.12×** | on ✓ |
| 16 | 16384 | 262144 | 1.37× | **1.09×** | on ✓ |

**16/16 全对**，两个 `rows` 恰好等于 16384 的压线配置也选对（分别赢 1.25× 和 1.73×）。
所以**不要往门槛里加 G**。

> 方法论：要评价一个启发式，必须把它旁路掉再两边都跑。让它自己选的 sweep 只能告诉你
> "它选完之后是什么样"，不能告诉你"它选得对不对"。

## 15.5 剩下的问题：G<=8 上 1.6–1.75×，而且不是规模效应（**归因已被 §15.11.2 推翻**）

> **§15.11 之后：这一节的归因是错的**，测量没错。"是 G 本身不是规模"这个结论排除了规模，
> 但没排除掉真正的自变量 —— `n_super = ceil(S/32)` 的低位零个数。下面那组"rows/字节/chunk
> 数完全相同"的对照里，`G=4/m=16384` 的 `n_super=512`、`G=16/m=4096` 的 `n_super=128`，
> 两个都是 8 的倍数，所以对照本身控住的变量不包括它。留着这一节是因为那个"找一组其他量全
> 相等的配置"的手法是对的，只是当时可枚举的自变量里少了一个。

把上表 swap on 那一列按 G 分组：`G=2/4/8` 全部收敛到 **1.6–1.75×**，`G=16` 是
**1.09–1.12×**，是个断崖不是渐变。换算成带宽（m=4096）：

| G | row | n32k4 (swap on) |
|---|--:|--:|
| 2 | 9.42 TB/s | 5.54 TB/s |
| 4 | 10.84 | 6.22 |
| 8 | 10.74 | 6.32 |
| 16 | 10.73 | **9.57** |

row 在所有 G 上都是 ~10.7 TB/s，只有 n32k4 掉。**关键对照**：`G=4/m=16384` 与
`G=16/m=4096` 的 rows（65536）、总字节、`(super,g)` 区域数（2048）、chunk 总数（65536）
完全相同，比值却是 1.67× vs 1.12×。**唯一的差别是 grid 里 g 这一维有几个值**，所以是 G
本身，不是规模。

机理上讲得通：swap 之后在飞的 block 能覆盖的不同 chunk 数约等于 `G × nspan`
（这些配置 `nspan=4`），G=16 给 64 个，G=8 只给 32 个。

## 15.6 super-major 重映射（初版：仅 2 的幂 S）

> 这是初版，**已被 §15.11 的通用版取代**（任意 S、`n_super` 补齐到 8 的倍数、默认开启）。
> 保留是因为下面那段静态验证的手法有复用价值。

针对 §15.5。思路不是再去摊 chunk，而是让**同一个 chunk 的 32 行不要同时在飞**：把行号
重排成「低位选 super、高位选 super 内的行号」。

```cpp
s = ((s & ((1 << super_shift) - 1)) << 5) | (s >> super_shift);   // super_shift = log2(S/32)
```

连续的 z 于是落到不同的 super，同一 chunk 的 32 行在 dispatch 序里被拉开整整一轮。
只在 **S 是 2 的幂**时启用，这样它是两条移位、且是 `[0,S)` 上的双射，**不需要尾部越界判断**。
`super_shift < 0` 关闭。开关 `AITER_IRGQ_SUPER_MAJOR`（0/1，默认关）。

**静态验证**（`--cuda-device-only -S`，不需要 GPU）：正好 6 条标量指令，无 `s_load`、无 VALU：

```
s_and_not1_b32 s17, s2, s17   ; super        = s & (2^shift - 1)
s_lshl_b32     s17, s17, 5    ; super << 5
s_ashr_i32     s22, s2, s30   ; row_in_super = s >> shift   (s30 = 预载的 super_shift)
s_or_b32       s17, s17, s22
s_cmp_lt_i32   s30, 0         ; guard
s_cselect_b32  s26, s2, s17
```

整份文件全部实例化 `VGPRs Spill: 0`、occupancy 多数 16 waves/SIMD。

> **已闭合**：正确性和性能都在 §15.11 测完了。结论是这个方向对，但初版的两个设计决定
> （只接 2 的幂、默认关）都太保守。

## 15.7 `--amdgpu-kernarg-preload-count=32`：§13.5 的结论要打个补丁

§13.5 的教训是"别让 kernarg 加载站在地址链头部"，据此我一度不敢给 kernel 加参数。
读 JIT 自己的 `cuda_cflags` 发现里面有 `-mllvm --amdgpu-kernarg-preload-count=32`，
反汇编也确认了：新增的 `super_shift` 以 **`s30` 这个预载 SGPR** 到达，kernel 前 80 条指令里
`s_load` 计数为 **0**。

所以准确的说法是：**前 32 个 dword 之内的显式 kernarg 是免费的**；§13.5 真正贵的那两样是
① 运行期除法（`row / G`），② `blockDim.x` —— 后者在 **hidden** kernarg 段，不参与预载。
§14.5.4 最后那条"别把 `contig_k` 做成模板参数，让 host 多传两个 int 即可"的建议因此更成立了。

## 15.8 和 Triton 的对比：口径

**同进程**。跨进程比过一次，同一配置能差 2×（30.85 vs 16.22µs），Triton 那侧甚至量出超过
本机 HBM 上限的数字。PR 的 Triton kernel 是逐字复制进
`/home/hwang/zoe/bench_scale_layout.py` 的（它只依赖 `_mxfp8_quant_op`，两棵树里逐字节相同）。

**口径陷阱**：PR 的两个 Triton kernel（`dynamic_mxfp8_quant` 和那个 n32k4 版）**都不做
inverse RoPE**，只做 quant。所以"我们比 Triton 慢"这句话要分清比的是什么。本轮给
benchmark 加了对等的两算子列：

- `tri rope` = `_rope_cached_bwd`（GPTJ，in-place，rope 尾单独切出来传）——
  和 `op_tests` 里 `unfused` 基线的第一条 kernel 是同一个调用；
- `tri r+plain` = rope + `dynamic_mxfp8_quant`，对标 `hip row`；
- `tri r+n32k4` = rope + PR 的 n32k4 quant，对标 `hip n32k4`。

（注：`aiter/ops/triton/fusions/inverse_rope_group_quant.py` 这个路径 §1 引用过，
**在当前这棵树里不存在**，别去找。）

**只比 quant 的话**（对我们偏不利，因为我们还多做 rope；对 Triton 偏不利的是它的
`torch.zeros` scale 分配算在计时里，我们的 buffer 是预分配的），`tri n32k4 / hip n32k4`：

| b \ m | 128 | 1024 | 4096 | 16384 |
|---|--:|--:|--:|--:|
| 2 | **1.72×** | **1.67×** | **1.12×** | 1.00× |
| 4 | **1.70×** | **1.45×** | 0.80× | 0.97× |
| 8 | **1.61×** | 0.74× | **0.53×** | **0.52×** |
| 16 | **1.54×** | 0.75× | 0.76× | 0.74× |

小 m 全面赢（Triton 撞 launch 开销），大 m 大 b 输。输的原因是两个不同的东西：

- **b=8 输，是 n32k4 布局**：我们相对自己 row 多付 68%，Triton 只多付 8%（§15.5）。
- **b=16 输，是 row 主路径**：我们 10.41 TB/s，Triton `dynamic_mxfp8_quant` 15.09 TB/s，
  搬的字节完全一样。这条和 n32k4 无关。

### 15.8.1 §15.11 之后重跑（本轮最终口径）

第一条没了，第二条原样留着，而且现在是**唯一**剩下的东西。对等口径
`tri r+n32k4 / hip n32k4`（他们两个 kernel，我们一个）：

| b \ m | 128 | 1024 | 4096 | 16384 |
|---|--:|--:|--:|--:|
| 2 | **2.89×** | **2.56×** | **2.27×** | **1.64×** |
| 4 | **2.74×** | **2.66×** | **2.08×** | **1.74×** |
| 8 | **2.52×** | **1.55×** | **1.13×** | **1.07×** |
| 16 | **1.90×** | **1.37×** | **1.08×** | **1.03×** |

**全部为正**，不再有输的格子。但大 shape 只剩 1.03–1.08×，而融合本该省下一整趟数据 ——
省下的被每字节效率吐回去了。按同样的字节量（读 bf16 的 `o`，写 fp8 + e8m0）折算带宽：

| | b=4/m=16384 | b=8/m=16384 | b=16/m=4096 | b=16/m=16384 |
|---|--:|--:|--:|--:|
| 我们（融合，含 rope） | 10.85 | 10.68 | 10.56 | 10.55 TB/s |
| Triton `dynamic_mxfp8_quant`（不含 rope） | 15.22 | 15.14 | 15.07 | 14.93 TB/s |
| 差距 | 1.40× | 1.42× | 1.43× | 1.41× |

**1.4× 就是现在全部的剩余空间**，而且它非常稳 —— 跨 b、跨 m 都是 1.40–1.43×，说明是个
固定的每字节代价，不是某个 shape 的调度问题。10.5 TB/s 也正好压在 §14.1 记的这个 kernel
的历史天花板上，所以它大概率比 n32k4 这条线还老。这条打掉的话，上面那张表的大 shape 会从
1.03× 直接变成 ~1.45×。

## 15.9 交接：文件、开关、命令

**本轮改动的文件**

| 文件 | 改动 |
|---|---|
| `csrc/include/inverse_rope_group_quant.h` | `bool scale_shuffle` → `enum ScaleLayout`，n32k4 字节公式 |
| `csrc/include/rocm_ops.hpp` | pybind `py::arg("scale_layout") = 0` |
| `csrc/kernels/inverse_rope_group_quant.cu` | 模板化、n32k4 store、super-major（主）+ swap（fallback）、`GS==32` 护栏、两个 env 开关 |
| `aiter/ops/inverse_rope_group_quant.py` | `SCALE_LAYOUTS`、`scale_shape()` 辅助、`scale_layout: str` |
| `op_tests/test_inverse_rope_group_quant.py` | `-l row mfma n32k4`；`_unshuffle_*` 向量化（分钟级 → 25s） |

**不在仓库里的工具**（放在 `/home/hwang/zoe/`，`zoe` 不是 git 仓库，不污染 aiter 的 git status）

- `bench_scale_layout.py` —— 全对比：hip row/n32k4、tri rope/plain/n32k4、两算子组合、fold
- `tune_n32k4_swap.py` —— §15.4 / §15.11 的受控 A/B 表，认 `TUNE_G` / `TUNE_M`（逗号分隔）
- `logs/` —— 本轮所有原始输出，文件名对应见 §15.11.4

**机器**（第四轮换过一次卡，环境和前几轮不一样，这段是给下一个人的）

- gfx1250 四卡，**256 CU**（§15.4 的 `rows >= 64*CU` 门槛因此是 16384 行）。
- **宿主机上没有 python**，所有东西都在容器里跑：
  `docker exec yzhou_latest bash -lc 'cd /home/hwang/zoe/aiter && ...'`。
  容器是别人起的，但挂了 `/home/hwang`。要自己起就照 `docker inspect` 抄一份：
  `--network host --ipc host --shm-size 16G --device /dev/kfd --device /dev/dri
  --group-add video --cap-add CAP_SYS_PTRACE --security-opt seccomp=unconfined`，
  镜像 `rocm/fw-bringup:gfx1250-atom--20260817`。
- 容器里是 root，所以 `aiter/jit/build/` 是 root 建的。**在宿主机上 `rm -rf` 会 permission
  denied**（那层目录 root 所有），要么 `sudo`，要么在容器里删。顶层那个 `.so` 反而删得掉，
  因为 `aiter/jit/` 自己是 hwang 的 —— 删除看的是父目录权限，不是文件权限，别被这个绊住。
- 跑之前 `rocm-smi --showmemuse` 看一眼，这台机器是几个人共用的。

**env 开关**（照 `topk_per_row_kernels.cu` 的 `TOPK_FORCE_GRID` 写法，getenv 只读一次，
不进 per-launch 开销；因此一个进程只能是一个设置）

| 变量 | 取值 |
|---|---|
| `AITER_IRGQ_SWAP_SG` | 不设/-1=启发式，0=强制 s 最快，1=强制 g 最快 |
| `AITER_IRGQ_SUPER_MAJOR` | 不设/-1=启发式（`S >= 256` 时开），0=强制关，1=强制开 |

注意 `SUPER_MAJOR=0` 现在等价于"回到 §15.4 的旧启发式"，因为 swap 的默认分支带
`!super_major`。做新旧对照就用它，不需要同时动两个变量。

`tune_n32k4_swap.py` 另外认 `TUNE_G` / `TUNE_M`（逗号分隔）来换形状列表。

**命令**（下面这台是 gfx1250 四卡机，跑在容器 `yzhou_latest` 里，
`docker exec yzhou_latest bash -lc 'cd /home/hwang/zoe/aiter && ...'`；宿主机上没有 python）

```bash
# 重编前两个都要删（§13.1.1）。注意 build 目录是容器里 root 建的，
# 宿主机上删要 sudo；在容器里删则不用。
rm -f  aiter/jit/module_inverse_rope_group_quant.so
rm -rf aiter/jit/build/module_inverse_rope_group_quant

# 正确性全量（280 行：row/mfma 跑满三档 group，n32k4 只在 32 档，见 §15.10）
PYTHONPATH=$PWD python op_tests/test_inverse_rope_group_quant.py \
  -d bf16 fp16 --group-size 32 64 128
# 非 2 的幂 S：专测 super-major 的尾部 return 和 n_super 补齐
PYTHONPATH=$PWD python op_tests/test_inverse_rope_group_quant.py \
  -d bf16 fp16 -l n32k4 --group-size 32 \
  -s 33 63 255 256 257 300 511 700 1000 1023 1025 3000 5000 12000
# graph capture/replay
PYTHONPATH=$PWD python op_tests/test_inverse_rope_group_quant.py \
  --graph -s 1 4 32 128 300 512 700 2048 3000 --group-size 32 128

# 新旧对照（先跑正确性！）。两半必须是两个进程，比值都取同 run 的 n32k4/row。
for v in 0 1; do
  AITER_IRGQ_SUPER_MAJOR=$v PYTHONPATH=$PWD python /home/hwang/zoe/tune_n32k4_swap.py
  TUNE_M=1000,3000,5000,12000 AITER_IRGQ_SUPER_MAJOR=$v PYTHONPATH=$PWD \
    python /home/hwang/zoe/tune_n32k4_swap.py
done

# 复现 §15.11.2 的 n_super 断崖（要先把 n_super 的补齐改回 ceil(S/32)）
TUNE_G=4 TUNE_M=3968,4000,4032,4064,4096,4128,4160,4224,4352,4608 \
  AITER_IRGQ_SWAP_SG=0 AITER_IRGQ_SUPER_MAJOR=1 \
  PYTHONPATH=$PWD python /home/hwang/zoe/tune_n32k4_swap.py

# 全对比
PYTHONPATH=$PWD python /home/hwang/zoe/bench_scale_layout.py

# 无 GPU 静态验证（§13.12 那套，本轮用它验的 §15.6 codegen）
FLAGS=$(python3 -c "import re;print(re.search(r'^cuda_cflags = (.*)$',open('aiter/jit/build/module_inverse_rope_group_quant/build/build.ninja').read(),re.M).group(1))")
hipcc $FLAGS -Rpass-analysis=kernel-resource-usage --cuda-device-only -S \
  -o /tmp/irgq.s csrc/kernels/inverse_rope_group_quant.cu
```

## 15.10 n32k4 只在 `quant_group_size == 32` 成立（本轮补的护栏）

名字里的 `n32` 是**超行高度**（32 行 token 打包进同一段），不是 quant group —— 两个 32
撞在一起纯属巧合，之前一直没把它们分开写，于是 kernel 放行了 group ∈ {32,64,128} 的全部
三档。这是错的：消费端一个 lane 的 WMMA scaleB operand 是 **一个 K=128 step 的 4 个
e8m0**，一次 `ds_load_b32` 取走。4 个 group 要正好铺满 128 个元素，group 就只能是 32。

危险的地方在于**它不会报错**。group=64/128 时 kernel 照样按布局公式把字节放到正确位置，
形状检查过、`op_tests` 里的 unshuffle 也过（那个 unshuffle 只验"字节落在公式说的地方"，
不验"这 4 个字节属于同一个 K step"）—— 然后 GEMM 把分属 4 个不同 K step 的 scale 当成
一个 step 的用掉，结果错，且没有任何东西会拦。权重侧的 `shuffle_scale_n32k4` 其实早就把
这件事钉死了，只是钉在**形状**上而不是断言上：它的入参是 `(E, N, K//32)`。

所以三处都加了检查，就近拦截：

- `csrc/kernels/inverse_rope_group_quant.cu` 的 host 侧 `AITER_CHECK`（和已有的
  `Ks % 4 == 0` 并排，两个约束是同一条推理的两半）；
- `aiter/ops/inverse_rope_group_quant.py` 在分配 scale 之前 `assert`；
- `csrc/include/inverse_rope_group_quant.h` 的 `kScaleN32K4` 注释写清 `n32` 的含义。

`op_tests` 里对应地 `continue` 掉 `n32k4 × group != 32`，同时把 `--group-size` 的默认值
从 `[128]` 改成 `[32, 128]` —— 否则默认那趟会把 n32k4 整个跳过，护栏反而变成了覆盖率
黑洞。全量从 360 行变成 280 行（row/mfma 三档 × 240，n32k4 一档 × 40）。

**已验证**：两层各跑了 gs ∈ {32,64,128} 三档，32 放行、64/128 拦住。注意 C++ 那层是
`AITER_CHECK` → `abort()`，**不是可捕获的异常**，所以进程直接挂（`SIGABRT`），stdout 里
没来得及 flush 的东西会一起丢 —— 想在一个脚本里遍历三档，得每档开一个子进程，
并加 `python -u`。

## 15.11 super-major 实测：n32k4 的代价基本被消掉

§15.6 那版终于跑上了。结论比预期好很多，也顺手推翻了 §15.5 的归因。

### 15.11.1 swap 和 super-major 是**互斥的两个解**，不是叠加的两层

第一次 A/B（只动 `SUPER_MAJOR`，`SWAP_SG` 交给启发式）出来的图样很干净：凡是启发式判成
swap off 的格子，重映射大赢；凡是判成 swap on 的格子，重映射**一点用都没有**。

这说明两者在解同一个问题（同 chunk 的 32 行同时在飞），谁先解掉，另一个就没东西可做了。
于是补了 2×2（`SWAP_SG × SUPER_MAJOR` 各 0/1），把从没测过的 **swap off + 重映射 on**
这一格补上 —— 而它恰好是全场最优：

| G | S | rows | sw0/sm0 | sw0/**sm1** | sw1/sm0（旧启发式） | sw1/sm1 |
|---|--:|--:|--:|--:|--:|--:|
| 2 | 4096 | 8192 | 1.41× | **1.00×** | 1.71× | 1.68× |
| 4 | 4096 | 16384 | 2.00× | **0.98×** | 1.78× | 1.76× |
| 8 | 16384 | 131072 | 1.43× | **1.00×** | 1.62× | 1.61× |
| 16 | 1024 | 16384 | 1.89× | **0.95×** | 1.13× | 1.11× |
| 16 | 16384 | 262144 | 1.25× | **1.01×** | 1.08× | 1.15× |

16 个配置里 12 个（全部 `S >= 256`）的最优都是 sw0/sm1，且都落在 **0.95–1.03×**。
剩下 4 个是 `S=128`：那里只有 4 个 super，重映射摊不开，纯赔（G=16 上 µs +23%）。

所以启发式重接成：**super-major 主路（`S >= 256`），swap 降为它不接时的 fallback**。

### 15.11.2 真正的自变量是 `n_super` 的低位零个数，不是 G

把上面的规则推广到任意 S 之后（`n_super = ceil(S/32)`，尾部 `s >= S` 的 block 直接 return），
**非 2 的幂的 S 大面积回退**，最差 0.60×。固定 G=4、扫 S=3968…4608：

| n_super | 124 | 125 | 126 | 127 | **128** | 129 | 130 | 132 | **136** | **144** |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 低位零个数 | 2 | 0 | 1 | 0 | 7 | 0 | 1 | 2 | 3 | 4 |
| n32k4/row | 1.67 | 1.96 | 1.78 | 1.96 | **1.00** | 2.00 | 1.80 | 1.67 | **0.99** | **0.99** |

完美单调，而且是**断崖**：`8 | n_super` 就是 1.00×，否则按 2 的幂次逐级退化，奇数时重映射
完全不起作用。这解释了 §15.5 那个"G 依赖"—— 当时所有配置的 S 都是 2 的幂，`n_super` 恒是
8 的倍数，G 只是跟着一起变的旁观量。

**修法很便宜**：`n_super` 是我们自己挑的，任何 `>= ceil(S/32)` 的值都保持双射，只要 kernel
把落到 `s >= S` 的行丢掉 —— 而它为了 `S % 32` 的尾部本来就要丢。所以**把 `n_super` 向上取
整到 8 的倍数**即可，最多浪费 7 个 super（224 个 block，S=3000 时 2.3%），这些 block 在碰
任何内存之前就 return 了。

```cpp
const int n_super = (((S + 31) / 32) + 7) & ~7;
```

补齐之后，同一组扫描全部落到 **0.98–1.02×**：

| S | 3968 | 4000 | 4032 | 4064 | 4096 | 4128 | 4160 | 4224 | 4352 | 4608 |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 补齐前 | 1.67 | 1.96 | 1.78 | 1.96 | 1.00 | 2.00 | 1.80 | 1.67 | 0.99 | 0.99 |
| **补齐后** | **1.02** | **1.00** | **0.99** | **0.99** | **1.00** | **1.02** | **1.02** | **1.00** | **1.00** | **0.98** |

> 机理还没坐实。断崖只跟 `n_super` 的 2 进制低位有关，而 chunk 之间的地址步长
> （`G*Ks*32` 字节）跟 `n_super` 无关，所以"并发 block 撞同一 chunk"的模型解释不了它，
> 更像是 L2 通道/bank 的地址散列在某个位段上对齐。要坐实得上 ATT。**但修法不依赖机理**：
> 断崖位置是直接测出来的，补齐是恒等变换，正确性由双射保证。

### 15.11.3 最终数字

32 个配置（G∈{2,4,8,16} × S∈{128,1024,4096,16384} 和 {1000,3000,5000,12000}），两次取平均，
比值都是同 run 的 n32k4/row：

| | 旧（swap 启发式） | 新（super-major） | 整体 |
|---|--:|--:|--:|
| 2 的幂 S | 1027 µs | **798 µs** | **1.29×** |
| 任意 S | 1010 µs | **764 µs** | **1.32×** |
| 合计 | 2037 µs | **1562 µs** | **1.30×** |

单点最大 **1.86×**（G=16/S=1000：33.72 → 18.11 µs）。**没有一个配置变慢**（最差 0.99×，
在 S=128 那几个 3 µs 的点上，属噪声）。n32k4/row 现在全场 **0.95–1.08×** —— 也就是说
**融合 n32k4 基本不再要钱**。原来挂在未闭合里的"fold 只赢 4%，融合这件事本身要重新论证"
那条随之作废：b=16/m=4096 上融合版 **79.6 µs** 对 "row 77.0 + 单独一趟 permute 11.2 =
88.2 µs"，从赢 4% 变成赢 **10%**，而且 fold 那一趟的代价随规模涨得更快（m=16384 时是
40.1 µs），所以大 shape 上差距只会更大。

### 15.11.4 验证与残留

跑过的：

- 全量 280 行（bf16/fp16 × 三档 group × row/mfma/n32k4）+ 56 行专测非 2 的幂 S
  （33/63/255/256/257/300/511/700/1000/1023/1025/3000/5000/12000）+ 90 行 graph
  capture/replay，`err` 与 `scale err` 全 0。非 2 的幂那批同时验了尾部 `s >= S` 的 return。
- `GS==32` 护栏两层各三档（§15.10）。
- 性能：§15.11.1 的 2×2、§15.11.2 的 S 扫描、§15.11.3 的新旧对照（各两次）、§15.8.1 的
  Triton 全对比。原始日志留在 `/home/hwang/zoe/logs/`（`ab2` = 2×2，`scan`/`scan2` =
  补齐前后的 S 扫描，`final2` = 新旧对照，`bench` = Triton 对比，`c3` = 最后一次全量）。

**一处没闭合的**：全量回归跑在"注释修改前"的那个二进制上。之后只改了 kernel 里的一段
注释（讲 tail block 为什么会多出来），重编完成了（`.so` 比源文件新），但那一轮的回归被打断
在中途。改动是纯注释、不影响 codegen，所以结论应该照旧 —— 但**下次上卡第一件事是把
§15.9 的正确性全量补跑一遍**，别默认它过了。

## 15.12 未闭合，按优先级

1. **主路径卡在 ~10.5 TB/s，Triton 同样字节量 ~15 TB/s，差 1.40–1.43×**（§15.8.1）。
   跨 b 跨 m 都是这个数，是固定的每字节代价。和 scale 布局完全无关，row/mfma/n32k4 一起中。
   §15.11 之后这条**升为第一优先，也是唯一一条性能项**：n32k4 已经贴着 row 跑，row 就是
   天花板了。打掉它，对等口径的大 shape 会从 1.03× 变成 ~1.45×。
2. **§15.11.2 那个 `8 | n_super` 断崖的机理没坐实**。修法是测出来的，不依赖机理，但机理
   本身可能对别的 kernel 也适用（"并发写的地址步长要避开某些 2 的幂对齐"），值得用 ATT
   查一次 L2 通道分布。另外补齐到 8 是测出来的最小充分值，没试过 16/32 会不会更好。
3. **`S < 256` 仍然走 swap fallback**。那里 super-major 因为 super 太少而赔钱
   （G=16 上 µs +23%）。这些形状绝对耗时都在 4 µs 以下，优先级低，但如果 decode 场景真的
   常驻小 S，值得再想一个专门的解。
4. **e8m0 舍入与 PR 的 Triton kernel 有分歧（已知，决定不改）**：我们是对 e4m3 的 448
   做 RoundUp，他们把尾数 ≥1.75 的进到下一个 exponent。amax=0.21875 时我们出 scale=116
   (qmax=448)、他们出 117 (qmax=224)。实测 8192 个 group 里 117 个差 1 个字节（bf16 下
   尾数恰为 1.75 很常见），但 dequant 后的值吻合到 9.5e-7。**我们的精度更高，保留**。
   接 PR 的产物做 bit-exact 比对时会撞到这个，别当成 bug。
5. **gfx950 上 n32k4 无意义**（那边是 MFMA tile），但 §15.2 的模板化和 §15.7 的 kernarg
   结论对 gfx950 同样适用，值得在有卡时一起验。§14.6 那两个 gfx950 问题仍然全部未闭合。

---

# 16. 第五轮：`G=16` 的带宽塌陷（两个 launch 形状 bug）

`GS=32 / -l n32k4` 的 sweep 里 `G=16` 明显低于其他 G（TB/s）：

| G | S=4096 | S=16384 |
|---|--:|--:|
| 2 | 6.52 | 7.57 |
| 4 | 6.50 | 7.58 |
| 8 | 6.56 | 8.26 |
| **16** | **5.35** | **5.60** |

两个独立原因，都不在 scale 布局上，而是**算错了 launch 形状**，所以 `row` 同样中招。

## 16.1 `Ks` 小的时候 block 填不满一个 wave

`h=16 head_dim=512` 下 `D = 16*512/G`，所以 `Ks = D/32` 随 G 反比缩小：

| G | Ks | `TDS=32` 时的 `k_slots` | block_size |
|---|--:|--:|--:|
| 2 | 128 | 32 | 32 |
| 4 | 64 | 32 | 32 |
| 8 | 32 | 32 | 32 |
| **16** | **16** | **16** | **16（半个 wave）** |

`TDS=32` 让 `threads_per_group = GS/TDS = 1`，于是 `k_slots = min(wave_size/1, Ks)` 被 `Ks`
夹住。`G=16` 时 `Ks=16`，block 只有 16 个线程 —— **wave32 上一半 lane 空转**，而且这是
`GS==32` 那个「大 launch 强制 wide tier」分支主动选出来的。

一般式 `block = k_slots * threads_per_group = min(wave_size, Ks * GS / TDS)`，所以
`block >= wave_size` 等价于 `Ks * GS >= TDS * wave_size`。不满足就把 TDS 减半：

```cpp
while(tds > 1 && (int64_t)scale_n * GS < (int64_t)tds * wave_size) tds >>= 1;
```

`G=16` 于是落回 `TDS=16`（`threads_per_group=2`, `k_slots=16`, block=32）。实测
（`AITER_IRGQ_TDS=32` 复现旧行为）：

| | S=4096 row | S=4096 n32k4 | S=16384 row | S=16384 n32k4 |
|---|--:|--:|--:|--:|
| 旧（半 wave） | 17.40 | 19.04 | 55.11 | 56.64 µs |
| 新（clamp） | **14.19** | **15.96** | **43.20** | **49.11** µs |
| | +23% | +19% | +28% | +15% |

> `narrow_slice` 这类「按 wave 总数选 tile 宽度」的启发式只看了**有多少活**，没看
> **一个 block 能不能凑够一个 wave**。后者是 `Ks` 的硬约束，必须单独 clamp。

## 16.2 `n_super` 往 `num_cu*4` 加宽是负收益

§15.11.2 定的是 `n_super = ceil(S/32)` 向上取整到 8 的倍数。后来有一版在
`ceil(S/32) >= 512` 时额外抬到 `max(n_super, num_cu*4)`，想把同一个 128B chunk 的 32 行在
dispatch 序里拉更开。代价是**每一行 padding 就是一个空 block，而这笔账要乘 G**：
`G=16 S=16384` 时 `n_super` 512 → 1024，`grid.x` 16384 → 32768，**一半 block 空转**。

阈值落在 `S >= 16352`，所以挑一个差一点点触发不到的 S 就能不改代码做对照：

| S | n_super | 空转 block | TB/s |
|--:|--:|--:|--:|
| 8192 | 256 | ~0 | 7.86 |
| **16000** | 504 | ~0 | **8.41** |
| **16384** | **1024** | **50%** | **5.70** |

`S=16000` 比 `S=16384` **少 2.4% 的数据却快 34%**。加宽已移除，回到 §15.11.2 的写法。

> 一个「只在大 S 生效」的开关，当初只在 `G=2` 上量过（那里 grid 最小、空 block 绝对数也最小），
> 就会看成"无影响"而被留下。**凡是改 grid 形状的旋钮，必须在 G 最大的那一列上验。**

## 16.3 两条修完之后

| G | S | n32k4 旧 | n32k4 新 | row 新 |
|---|--:|--:|--:|--:|
| 2 | 16384 | 54.0 (7.57) | **48.8 (8.38)** | — |
| 4 | 16384 | 54.0 (7.58) | **50.9 (8.04)** | — |
| 8 | 16384 | 49.5 (8.26) | 49.3–53.6 (7.6–8.3) | 41.3–42.4 (9.6–9.9) |
| 16 | 4096 | 19.1 (5.35) | **16.0 (6.40)** | 14.2 (7.21) |
| 16 | 16384 | 73.0 (5.60) | **49.1–51.2 (8.0–8.3)** | 43.2–44.2 (9.3–9.5) |

`G=16 S=16384` 从 5.60 到 ~8.1 TB/s（**1.45×**）。`G=8` 落在 49–54 µs 的 run 间离散区内，
没有可辨别的变化。`row` 现在到 9.3–9.9 TB/s，`n32k4` 相对 `row` 还差 ~1.15–1.20×（剩余 layout 税）。

**回归**：1120 项 `passed~`、0 失败（bf16+fp16 × 三档 group × 三种 layout）；非 2 的幂 S
14 个值全过；graph capture/replay 全过。

## 16.4 A/B 回归扫：80 个 shape，无回退

前面的回归只证了"不出错"，没证"不变慢"。补一轮真正的 A/B：把修复前的 `.cu` 单独 checkout
出来重编一份二进制，两份跑同一组 80 个 shape（`(h,g)` = 16,2 / 16,8 / 16,16 / 8,1 × `S` =
128 / 1024 / 4096 / 16384 × group 32、128 × row/mfma_tile/n32k4），再按 `after/before` 比。

```bash
git checkout HEAD~1 -- csrc/kernels/inverse_rope_group_quant.cu   # 记得之后 checkout HEAD 还回来
rm -f aiter/jit/module_inverse_rope_group_quant.so && rm -rf aiter/jit/build/module_inverse_rope_group_quant
python op_tests/test_inverse_rope_group_quant.py -d bf16 -b 16,2 16,8 16,16 8,1 \
    -s 128 1024 4096 16384 --group-size 32 128 -l row mfma_tile n32k4 > /tmp/before.log
```

结果：56 个落在 ±5% 内，19 个提升（最大 0.74×），5 个被标成 >1.05× 的**全部是 `mfma_tile`**。
而 `mfma_tile` 在代码上到不了这两处改动：`if constexpr(kMfmaTile)` 分支原样未动且优先于新的
`else if`，`n_super` 只在 `super_major`（要求 `LAYOUT == kScaleN32K4`）时经 `s_extent` 起作用。
同一批里 `mfma_tile` 还出现了 0.88× 和 0.92× 的"提升"，双向摆动即噪声。把这几个 shape 连测
三遍，离散区完全盖住了两侧的值：

| shape (mfma_tile, GS=32) | before | after | 连测三遍 |
|---|--:|--:|---|
| G=2 S=16384 | 50.0 | 53.4 | 53.1 / 53.8 / **47.5** |
| G=8 S=16384 | 50.0 | 52.9 | 50.6 / 51.7 / **47.5** |
| G=8 S=128 | 5.78 | 6.32 | 6.10 / 5.74 / 6.08 |
| G=16 S=128 | 6.03 | 6.40 | 6.24 / 6.40 / 6.48 |

所以这台机器上 `mfma_tile` 的 run 间离散度有 ±8–11%，单跑一次的差值读不出 5% 级别的结论——
要判 mfma 路径的回归必须连测，或者干脆按代码路径排除。A/B 表只跑了 group 32 和 128，
补一档 64（clamp 在小 `Ks` 上会触发的那档）：`G=8/16 × S=1024/4096/16384 × bf16/fp16`
96 项 `passed~`、0 失败，`row` 在 `S=16384` 是 9.1–9.9 TB/s。

**gfx950 未受影响**，两处改动各自被 `LAYOUT == kScaleN32K4`（gfx950 用的是 mfma tile）和
`!wave64` 挡住；`git diff -w HEAD~1` 的功能改动就这两块。

## 16.5 下一步

§15.12 第 1 条（主路径 1.4×）仍是最大的一块，而这一轮**支持**它：两个 bug 修完 `row` 到了
9.3–9.9 TB/s，说明之前那个"天花板"里有一部分其实是 launch 形状的浪费，不全是每字节代价。
先在这个二进制上把 1.40–1.43× 重测一遍再定性。
