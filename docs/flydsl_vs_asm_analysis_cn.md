# asm vs flydsl 性能差异分析（shape: token=32768, model_dim=4096, inter_dim=192, E=193, topk=9, fp8 per_tensor, silu, g1u1, gfx942）

> 端到端基线（本机实测，warmup=10/iters=100，稳定）：
> - **asm（asmjit 1-stage）：~8600 us**
> - **flydsl（CK gemm1 + flydsl gemm2 的 2-stage）：~14500 us（慢 1.69×）**

> Kernel 级分解（profiler，us/iter）：
>
> | 组件 | asm 路径 | flydsl 路径 |
> |---|---|---|
> | gemm1（gate/up 上投影） | `moe_2stage_gateup` 3684 | CK `kernel_moe_gemm` 3103 |
> | gemm2（down 下投影） | `moe_2stage_down` 3162 | flydsl `moe_gemm2_0` **7475** |
> | 激活量化 bf16→fp8 | `dynamic_per_token_scaled_quant` ×2 = **759** | `data_to_scale` ×2 = 3108 + `scaled_quant` ×2 = 500 = **3608** |
> | final reduce | `moe_gemm_final_reduce` 837 | (折叠进 gemm2 atomic) |
> | moe_sorting | ~160 | ~330 |
> | 合计 | ~8604 | ~14537 |

分析进度（本轮按"分步审批"逐项重新核实源码）：
- [x] 步骤 1：量化部分差异分析（已逐行核实源码，见下）
- [x] 步骤 2：gemm2（asm `moe_2stage_down` vs flydsl `moe_gemm2_0`）差异分析（已反汇编 asm + dump flydsl ISA 实证）
- [x] 步骤 3：moe_sorting 差异分析（已核实调用路径，见末尾）

---

## 步骤 1：量化部分的差异与原因

### 结论先行

两条路径量化耗时差约 **4.75×（759us vs 3608us）**，**根因不是实现细节，而是激活量化的"粒度"被选成了不同的方案**：

| | asm 路径 | flydsl / CK 2-stage 路径 |
|---|---|---|
| 激活量化粒度 | **per_Token**（每行一个 scale） | **per_Tensor**（整张 tensor 一个 scale） |
| 需要几个 kernel | **1 个**（融合：边读边算 scale 边量化） | **3 个**（initScale + data_to_scale + scaled_quant） |
| 读数据次数 | **1 遍**（数据进 VGPR 后直接量化复用） | **2 遍**（先全表求 max，再全表应用 scale） |
| 是否需要全局规约 | 否（每行独立） | **是**（要先得到整张表的全局 max） |
| 实测耗时 | 759 us | 3608 us |

### 为什么 asm 是 per_Token、flydsl 是 per_Tensor？（关键证据）

虽然两次测试我都传了 `--quant-type per_tensor`，但 **asm 的 asmjit 路径把激活量化硬编码成了 per_Token**，配置里的 `per_Tensor` 只作用于**权重**量化。证据在 `aiter/fused_moe_asmjit_aot.py`：

```120:126:/opt/aiter/aiter/fused_moe_asmjit_aot.py
        quant_func = aiter.get_hip_quant(aiter.QuantType.per_Token)
        hidden_states_q, hidden_states_scale = quant_func(
            hidden_states,
            scale=None,
            quant_dtype=w1.dtype,
            num_rows=None,
        )
```

```165:170:/opt/aiter/aiter/fused_moe_asmjit_aot.py
        gemm1_out_q, gemm1_out_scale = quant_func(
            gemm1_out.view(B * TOPK, -1),
            scale=None,
            quant_dtype=w2.dtype,
            num_rows=None,
        )
```

- 第 120 行 `aiter.get_hip_quant(aiter.QuantType.per_Token)` —— 写死 per_Token。
- 配置字符串里的 `quant_type`（per_Tensor）只通过 `quant_type_w=f"QuantType.{qtype_str}"` 传给 gemm kernel，控制的是**权重**怎么用 scale，不影响激活量化粒度。

而 flydsl/CK 的 2-stage 路径在 `aiter/fused_moe.py` 里按用户配置的 `quant_type=per_Tensor` 走 `get_quant(quant_type)` → 调用 `dynamic_per_tensor_quant`，于是激活就是真 per_Tensor。

### per_Tensor 为什么必然更慢（算法层面）

per_Tensor 要求**整张 tensor 共用一个 scale = 全局 max / FP8_MAX**。这导致：

1. **无法把"求 scale"和"量化"融合进一个 kernel**：必须先扫一遍整张表求出全局 max（`data_to_scale_kernel`），全局 max 算完后才能开始量化（`scaled_quant_kernel`）。→ 至少 **2 遍读数据**。
2. per_Token 则每行有自己的 scale，行与行独立，所以可以**一个 kernel 一遍搞定**：把一行数据读进寄存器(VGPR) → 算这行的 max → 直接用寄存器里的数据量化写出。

看 `csrc/kernels/quant_kernels.cu` 的两个 host 入口就一目了然：

per_Tensor 启 3 个 kernel：
```676:687:/opt/aiter/csrc/kernels/quant_kernels.cu
        AITER_DISPATCH_FLOATING16_TYPES(input.scalar_type(), "scaled_quant_kernel", [&] {
            using input_dtype = typename t2opus<scalar_t>::type;
            aiter::initializeScale<<<dim3(1), dim3(64), 0, stream>>>(
                scale.data_ptr<float>(), 1, 0.0f);
            aiter::data_to_scale_kernel<input_dtype, opus::fp8_t><<<grid, block, 0, stream>>>(
                scale.data_ptr<float>(), reinterpret_cast<input_dtype*>(input.data_ptr()), cols);
            aiter::scaled_quant_kernel<<<grid, block, 0, stream>>>(
                reinterpret_cast<opus::fp8_t*>(out.data_ptr()),
                reinterpret_cast<input_dtype*>(input.data_ptr()),
                scale.data_ptr<float>(),
                cols);
        });
```

per_Token 只启 1 个融合 kernel（读一行→算 scale→量化复用寄存器数据）：
```370:396:/opt/aiter/csrc/kernels/quant_kernels.cu
    auto res         = data_to_per_row_scale<DTYPE_I, DTYPE_O, thread_data_size>(input, cols);
    float row_scale  = std::get<0>(res);
    DTYPE_I* vec_ptr = std::get<1>(res);

    if(threadIdx.x == 0)
    {
        ...
        scale[token_idx] = row_scale;
    }
    ...
        scaled_quant_vgpr_impl<DTYPE_I, DTYPE_O, thread_data_size>(out, vec_ptr, &row_scale, cols, row_offset);
```

### 为什么 data_to_scale 这一遍格外慢（实现层面，3108us 远超访存理论）

光"2 遍 vs 1 遍"只能解释 ~2×，但实测 `data_to_scale`(3108us) 单独就是 `scaled_quant`(500us) 的 **6×**，两者都只读一遍数据，差距来自 `data_to_scale_kernel` 的**线程映射极其低效**，尤其对**中间结果的量化**：

- grid/block 固定为 `grid(rows), block(BlockSize=256)`，即**一个 block 处理一行**，全靠 `atomicMaxFloat` 把每行的 max 汇聚到**唯一一个全局 float**。
- 这条路径上有**两次** per_tensor 量化：
  - ① hidden `[32768, 4096]`：32768 个 block，每 block 256 线程各读 1 个 16 元素向量，尚可。
  - ② 中间结果 `gemm1_out` `[B·TOPK, inter_dim] = [294912, 192]`：**cols 只有 192**，`num_vecs = 192/16 = 12`，即**每个 block 的 256 个线程里只有 12 个在干活**（利用率 ~5%），却要起 **~29.5 万个 block**，每个 block 末尾还要做一次 256 路 `block_reduce` + 一次 `atomicMaxFloat` 到同一个全局地址 → **极端的 block 数 + 原子争用 + 线程闲置**。这一项是 3108us 里的大头。

对照 per_Token 同一份中间结果 `[294912, 192]`：`192 ≤ 8·256`，走 `thread_data_size=8` 分支，一行进寄存器一遍量化完，实测 `Li8` 版本仅 603us。

### 逐行核实补充（本轮重新读源码确认）

精确把 profiler 的 4 个量化 kernel 一一对上源码（`BlockSize=256`）：

| profiler 项 | 对应数据 | 源码路径 | 线程映射 | 实测 |
|---|---|---|---|---|
| `data_to_scale` ×2 | hidden`[32768,4096]` + 中间`[294912,192]` | `dynamic_per_tensor_quant` 起 `initializeScale`+`data_to_scale_kernel`+`scaled_quant_kernel`（quant_kernels.cu:676-687） | 每行 1 block，`data_to_per_row_scale<…,0>` 先全行求 max，再 `block_reduce<256>` + thread0 `atomicMaxFloat` 到**唯一全局地址** | 3108us |
| `scaled_quant` ×2 | 同上两张表 | `scaled_quant_kernel`（同上，第二个 kernel） | 每行 1 block，只读 1 遍 + 写出，无规约无原子 | 500us |
| `dynamic_per_token Li16` | hidden`[32768,4096]`，cols=4096≤16·256 → `thread_data_size=16` | `dynamic_per_token_scaled_quant`（quant_kernels.cu:820-826 + DISPATCH 宏:646-662） | 每行 1 block，读进 VGPR→算行 max→`scale[token]=row_scale`（**无全局原子争用**）→ 直接用寄存器量化 | 156us |
| `dynamic_per_token Li8` | 中间`[294912,192]`，cols=192≤8·256 → `thread_data_size=8` | 同上 | 同上（融合一遍） | 603us |

**两条关键证据，确认根因：**

1. **粒度不同导致 kernel 数不同**：`dynamic_per_tensor_quant`（quant_kernels.cu:664）无论如何都要 3 个 kernel（init+求 max+量化），因为全局 scale 必须先全表规约出来；`dynamic_per_token_*`（:710）走 `else` 分支（:818-843）只 1 个融合 kernel。

2. **per_Tensor 的"求 max"那一遍格外慢的实现根因 = 单点全局原子 + 块内大量闲置线程**：
   - `data_to_scale_kernel`（:236-246）对中间结果 `cols=192`：`vec_size=16` → `num_vecs=12`，即 256 线程里**只有 12 个**载入数据，却仍执行 256 路 `block_reduce`，再由 thread0 对**同一个 float 地址**做 `atomicMaxFloat`。
   - 中间结果有 **~29.5 万个 block**，全部抢这一个原子地址 → 严重串行化。这就是 3108us 的主要来源（远超 ~300us 的访存理论值）。
   - per_Token 的 `scale[token_idx]=row_scale`（:384）写的是**各自独立**的地址，没有任何争用。

3. **asm 为何用 per_Token**：`fused_moe_asmjit_aot.py:120` 硬编码 `get_hip_quant(QuantType.per_Token)`；2-stage 路径 `fused_moe.py:670` 用 `get_quant(quant_type)` 跟随用户的 `per_Tensor`。两者在 GEMM 里对**权重** scale 的用法一致（都按 `quant_type_w=per_Tensor`），差别**只在激活量化粒度**。

> 附带发现：代码库其实已有**融合版 per_tensor 量化**（`fused_moe.py:933` 的 `dynamic_per_tensor_quant_fp8_i8_fused_small` / `_nozero`），目前只在 small-M direct 路径用。这是步骤 1 优化时的现成抓手。

### 对优化的启示（先记录，后续步骤再展开）

1. **最划算**：让 2-stage 路径的激活量化也走 per_Token 融合 kernel（像 asm 那样），单这一步可省 ~2.8ms（3608→~760us）。需确认 flydsl gemm2 是否能吃 per_token 的激活 scale。
2. 若必须保留 per_Tensor：可改用代码库里已有的**融合 per_tensor 量化**（`fused_moe.py` 中 `_direct_per_tensor_quant_cached` / `dynamic_per_tensor_quant_fp8_i8_fused_small`，small-M direct 路径在用），避免"两遍 + 低效 block 映射"。
3. `data_to_scale_kernel` 对**窄 cols（如 192）**的线程映射应重写：让一个 block 吃多行、提高线程利用率、用层级规约替代单点 atomic。

---

## 步骤 2：gemm2（下投影）的差异与原因

### gemm2 在干什么

下投影 GEMM 形状（每个 expert）：A2 = 中间结果 `[M_sorted, K=inter_dim=192]`，W2 = `[E, N=model_dim=4096, K=192]`，输出 `[M, N=4096]`。
**注意 K 只有 192，非常短**（tile_k=64 时只有 3 次 K 迭代）→ MFMA 计算量很小，kernel 不是算力瓶颈，而是**访存 + 写出/累加（epilogue）瓶颈**。同时因为 topk=9，每个原始 token 有 9 份要累加。

### 结论先行

| | asm `moe_2stage_down` | flydsl `moe_gemm2_0`（`t32x128x64_atomic_w2_bnt0`） |
|---|---|---|
| topk 如何累加 | **写出全部 9 份部分结果**到 `[B, TOPK, N2]`，**再用独立 kernel `moe_gemm_final_reduce_bf16` 规约** | **gemm2 内部用 global atomic-add 直接累加**进输出（`_atomic` 模式） |
| 用不用原子操作 | **不用** | **用**，且 gfx942 上 bf16 只能走**慢速 `global_atomic_pk_add_bf16`（raw pointer）** |
| 耗时 | down 3162 + reduce 837 ≈ **4000 us** | **7475 us**（reduce 已折叠进来） |
| 即便扣掉 asm 的 reduce | gemm 本体 3162 | 7475，**约 2.36×** |

**根因：两者把"topk 的 9 份结果合并"的策略完全不同。** asm 选择"空间换原子"——多写一份 `[M·9, 4096]` 的部分结果，再用一个高效的规约 kernel 求和，全程**零原子**；flydsl 选择"原子折叠"——每个 tile 直接 atomic-add 到最终输出，省掉了 reduce kernel，但在 **gfx942 上 bf16 没有快速 buffer 原子指令**，只能用慢速 global 原子，加上 topk=9 造成的**写竞争**，反而更慢。

### 证据

flydsl 这个变体 `flydsl_moe2_afp8_wfp8_bf16_t32x128x64_atomic_w2_bnt0` 解析为：tile_m=32 / tile_n=128 / tile_k=64，`atomic` 模式，`waves_per_eu=2`，`b_nt=0`（B 普通缓存）。`n_per_wave=32` → 4 个 wave / 256 线程。

gfx942 上 bf16 输出被迫走 global 原子（慢路径）：
```2352:2360:/opt/aiter/aiter/ops/flydsl/kernels/moe_gemm_2stage.py
    # gfx950+ has buffer_atomic_pk_add_bf16 → bf16 can use buffer atomics (same as f16).
    # gfx942 only has global_atomic_pk_add_bf16 → must use global atomics with raw pointer.
    _has_buffer_atomic_bf16 = str(gpu_arch).startswith(("gfx95", "gfx12"))
    _needs_global_atomic_bf16 = out_is_bf16 and not _has_buffer_atomic_bf16
    if out_is_bf16:
        if not supports_bf16_global_atomics(gpu_arch):
            raise ValueError(
                f"out_dtype='bf16' requires bf16 global atomics ({bf16_global_atomics_arch_description()}), got arch={gpu_arch!r}"
            )
```

flydsl 用**同一套 atomic-add epilogue** 同时完成 topk 合并和 split-K 合并：
```2215:2223:/opt/aiter/aiter/ops/flydsl/kernels/moe_gemm_2stage.py
    k_batch:
      Split-K factor along the inter_dim K axis. Default 1 = no split. With
      ``k_batch > 1``, the launch grid grows along Z (``grid=(gx, gy, k_batch)``)
      so each WG processes only ``inter_dim / k_batch`` of the K reduction; the
      individual partials are merged via the SAME atomic-add epilogue stage2
      already uses for topk accumulation. ...
```

asm 则是"写满 topk 部分结果 → 独立规约"，全程不碰原子：
```171:212:/opt/aiter/aiter/fused_moe_asmjit_aot.py
        gemm2_out = torch.empty(
            B, TOPK, N2, dtype=torch.bfloat16, device=gemm1_out_q.device
        )
        hsaco.fmoe_asmjit.moe_2stage_down(
            [grid_down],
            [256],
            ...
            gemm2_out,  # cur_out,
            ...
            with_silu=False,
            BLOCK_TILE_SIZE_M=kcfgs.BLOCK_M,
            BLOCK_TILE_SIZE_N=DOWN_BLOCK_TILE_SIZE_N,
            quant_type_w=f"QuantType.{qtype_str}",
            dyn=kcfgs.use_dyn_sched,
        )
        num_WG = num_CU * 4
        num_tokens_wg = B // num_WG
        num_extra_tokens = B % num_WG
        hsaco.fmoe_asmjit.moe_gemm_final_reduce_bf16(
            [num_WG],
            [64],
            gemm2_out,
            cur_out,
            num_tokens_wg,
            num_extra_tokens,
            B,
            TOPK=TOPK,
            OC=N2,
        )
        return cur_out
```

### 反汇编实证（本轮新增，最硬的证据）

asm 的 `moe_2stage_down` 虽是预编译 hsaco（仓库无源码），但可直接反汇编。对应我们 shape 的 `.co`：
`hsa/gfx942/fmoe_asmjit/moe_2stage_down-...-TOPK=9-K=192-N=4096-...-BLOCK_TILE_SIZE_M=64-BLOCK_TILE_SIZE_N=128-quant_type_w=QuantType.per_Tensor-dyn=False.co`

```bash
/opt/rocm/llvm/bin/llvm-objdump -d --mcpu=gfx942 <co>
```

**asm down kernel ISA 关键指令（实测）**：

| 指令 | 次数 | 含义 |
|---|---|---|
| `v_mfma_f32_16x16x32_fp8_fp8` | 144 | fp8 MFMA GEMM 主体 |
| `buffer_load_dwordx4` | 36 | 经 LDS 暂存的 tiled 加载（16B/次，硬件边界检查） |
| `ds_write_b64` / `ds_read_b128` | 16 / 8 | LDS 暂存 |
| `v_pk_mul_f32` / `v_perm_b32` | 32 / 32 | epilogue：乘 scale 反量化 + 打包 bf16 |
| **`global_store_dwordx4 ... nt sc1`** | **8** | **普通非时序 store（8 bf16/次），零原子** |
| `global_atomic*` | **0** | **完全没有原子** |

资源：`VGPR=208, LDS=19456B, block=256`。epilogue 片段：
```
v_pk_mul_f32 v[72:73], v[72:73], v[48:49]   ; acc * scale（反量化）
v_perm_b32   v72, v72, v73, s26             ; 两个 f32 → 打包成 bf16
global_store_dwordx4 v[56:57], v[96:99], off nt sc1   ; 普通写出
```

flydsl 的 `moe_gemm2_0` 是 JIT 编译，用 `FLYDSL_DUMP_IR=1` dump 出最终 ISA（`~/.flydsl/debug/moe_gemm2_0/17_final_isa.s`）：

**flydsl gemm2 kernel ISA 关键指令（实测）**：

| 指令 | 次数 | 含义 |
|---|---|---|
| `v_mfma_f32_16x16x32_fp8_fp8` | 24 | fp8 MFMA（tile 比 asm 小，静态条数不可直接比） |
| `buffer_load_dword(x4)` | 22 + 6 | 加载 |
| `ds_write_b16_d16_hi` / `ds_read_b128` | 16 / 8 | CShuffle epilogue 经 LDS |
| **`global_atomic_pk_add_bf16`** | **8** | **全局原子加（2 bf16/次），输出全程靠原子** |
| `global_store*` | **0** | **输出没有一条普通 store** |

实测原子写出片段（`syncscope="agent"`、绕过 L1）：
```
global_atomic_pk_add_bf16 v[4:5], v7, off
global_atomic_pk_add_bf16 v[4:5], v1, off offset:128
```

**两边 ISA 对照一句话总结**：

| | asm `moe_2stage_down` | flydsl `moe_gemm2_0` |
|---|---|---|
| 输出指令 | `global_store_dwordx4`（8 bf16/次，普通 nt store） | `global_atomic_pk_add_bf16`（**2 bf16/次，原子加**） |
| 同样写 N=4096 一行 | 4096/8 = **512 次普通 store** | 4096/2 = **2048 次原子**，且 9 份 topk 抢同一行 → 严重串行 |
| topk 合并 | 延后到独立 reduce kernel | gemm2 内 atomic 直接合并 |

→ 与源码 `moe_gemm_2stage.py:3733-3748`（`_needs_global_atomic_bf16` 且 `accumulate=True` → `llvm.AtomicRMWOp(fadd, ..., syncscope="agent")`）和 `_e_vec=2`（:2416-2417，每原子只 2 bf16）完全吻合。

### 次要差异

- **tile 形状**：asm down 用 `BLOCK_TILE_SIZE_N=128`、block 256；flydsl 用 tile 32×128×64、256 线程。N=4096 两者都切 128 列；M 维度都很大（~29 万行）不缺并行度，所以差距主因不是 occupancy，而是 epilogue。
- **B（权重）缓存**：当前变体 `bnt0` 用普通缓存读 W2。W2 很大（193·4096·192·1B ≈ 145MB）且每个 expert 复用，缓存策略影响有限。
- **K 短**：K=192 让算术强度极低，进一步放大了 epilogue（写出/原子）在总时间里的占比。

### 对优化的启示（先记录）

1. **最可能见效**：把 flydsl gemm2 的 topk 合并从"bf16 global 原子"改成 asm 那种"**写部分结果 + 独立规约**"，或改用 flydsl 已有的 `reduce` / `split_reduce`(`_sr`) 非原子模式，规避 gfx942 的慢 bf16 原子。
2. 尝试 `out_dtype=f16/f32` 原子（f16 是 half2 快原子；f32 标量原子）配合最后一次类型转换，看是否快于 bf16 global 原子。
3. tile/`n_per_wave`/`waves_per_eu`/`b_nt` 仍有调参空间，但量级上不如 ①。

---

## 步骤 3：moe_sorting 的差异与原因

### 结论先行

**两条路径用的是完全相同的 sorting kernel（`ck_tile::MoeSortingMultiPhaseKernel`），实现没有任何区别。唯一的差别是调用次数：flydsl 2-stage 路径 sort 了两次，asm 只 sort 一次。**

| | asm（asmjit / stage0） | flydsl 2-stage |
|---|---|---|
| moe_sorting 调用次数 | **1 次**（BLOCK_M=64） | **2 次**（stage1 用 block_m=128，stage2 用 block_m2=32） |
| profiler 每个 sorting kernel 的 count | 1 | **2** |
| 实测 sorting 总耗时 | `P23`128.98 + `P0_v1`16.98 + `P1`9.23 + `Clear`6.55 ≈ **161us** | 上述每项 ×2 ≈ **328us** |
| 单次 sort 成本 | 同上 | `P23` 260.92/2 ≈ 130 ≈ 与 asm 单次一致 |

→ 即"同样的活干了两遍"，所以约 2×；**kernel 本身一样快**。

### 为什么 flydsl 要 sort 两次（核实到的调用路径）

1. **asm 走 stage0 早退，自己只 sort 一次**：`fused_moe.py:382` 一旦 `metadata.stage0` 非空（asmjit），直接 `return metadata.stage0(...)`，**完全绕过** dispatch 层的 sorting 分支；`fused_moe_asmjit_aot.py:108` 内部用 `kcfgs.BLOCK_M=64` 只调一次 `moe_sorting`，gateup 和 down 共用这一份排序。

2. **flydsl 2-stage 因两个 stage 的 block_m 不同 → 被迫 sort 两次**：
   - tuned 配置里 `block_m=128`（stage1 CK gemm1 的最优 tile）、`block_m2=32`（stage2 flydsl gemm2 `t32x128x64` 的 tile_m=32），见 `hy3_fp8_pertensor_tuned_fmoe.csv` token=32768 行。
   - `fused_moe.py:429` 判断：`run_1stage or block_size_M1 == block_size_M2` 才共用一次排序；这里 `run_1stage=False` 且 `128 != 32`，落入 `else`（:447-471），对 `topk_ids` **用 block_m=128 排一次给 stage1、用 block_m2=32 再排一次给 stage2**。
   - 排序结果（`sorted_token_ids` / `sorted_expert_ids` 的分块 padding）依赖 block_m，所以两个 block_m 不能复用，必须各排一次。

```429:471:/opt/aiter/aiter/fused_moe.py
    elif metadata.run_1stage or block_size_M1 == block_size_M2:
        sorted_ids1, ... = moe_sorting(..., block_size_M1, ...)   # 单次
        sorted_ids2 = sorted_ids1   # 复用
        ...
    else:
        sorted_ids1, ... = moe_sorting(..., block_size_M1, ...)   # stage1: block_m=128
        sorted_ids2, ... = moe_sorting(..., block_size_M2, ...)   # stage2: block_m2=32（第二次）
```

> 注：direct small-M 路径（`flydsl_moe*_direct`）会走 `fused_moe.py:419` 的 `direct_2stage` 分支，**完全不调 moe_sorting**（直接用 topk_ids）。但本 shape（token=32768）用的是非 direct 的 CK+flydsl 2-stage，所以是上面的"两次排序"。

### 对优化的启示（先记录）

1. 影响量级最小（161→328us，差 ~167us），优先级低于 gemm2(🅰) 和量化(🅱)。
2. 若让 stage1/stage2 统一 block_m（例如都用 32 或都用 128），即可省去第二次排序 → 回到单次 ~161us。代价是其中一个 stage 可能偏离它的最优 tile，需要权衡（要重新 tune 验证 stage gemm 变慢的幅度是否小于省下的 ~167us）。
3. sorting kernel 本身（`MoeSortingMultiPhaseKernel`）两条路径一致，无需单独优化它。

---

## 步骤 4：gemm2 实测优化（已落地，本轮新增）

### 重要结论修正：大 M 下"原子加"不是瓶颈

步骤 2 留下的头号假设是"把 bf16 global 原子改成非原子写+独立 reduce 能加速"。**实测把这个假设证伪了**（shape: token=32768）：

| gemm2 模式 | `moe_gemm2_0` us | 额外 reduce us | e2e device us |
|---|---|---|---|
| atomic（原配 `t32x128_w2_bnt0`） | 7479.8 | 0 | 14545 |
| **reduce（非原子写+topk_sum，`t32x128_reduce_w2`）** | **9274.9（更慢）** | +708.7 | 17056（更差） |

原因：M=32768 下 gemm2 是**显存带宽瓶颈**。
- atomic 模式写入 `out[token, model_dim]`≈268MB 小缓冲，9 份 topk 累加进同一地址（L2 友好），**总 HBM 写流量小**。
- reduce 模式写 `[token*topk, model_dim]`≈2.4GB 的 partial（写流量 ×9），再读回 2.4GB 做规约 → 带宽翻数倍。

对照 asm：asm 的 `moe_2stage_down` 用**同样的非原子写** 2.4GB partial 只要 3162us，flydsl 的 reduce 模式要 9275us。**所以真正的差距是 flydsl gemm2 这个 GEMM kernel 本身的效率（约 2.9×），不是累加方式。** atomic（7479）已是 flydsl 两种模式里更优的那个。

> 步骤 2 的"启示 ①/②（atomic→reduce / 换 f16/f32 原子）"在大 M 下作废；在小 M（带宽不紧张、原子争用占主导）可能仍成立，需另测。

### 真正生效的优化：tile_m=64（必须同步 block_m2=64）

在 **atomic 模式**下扫 tile/waves/bnt/split-K，发现 `tile_m` 从 32 提到 64、`tile_n=128` 不变是甜点（这恰好和 asm down kernel 的 `BLOCK_TILE_SIZE_M=64, BLOCK_TILE_SIZE_N=128` 一致，见步骤 2 反汇编文件名）：

| 候选（atomic, 已对齐 block_m2=tile_m） | gemm2 us | 备注 |
|---|---|---|
| `t32x128_w2_bnt0`（原配） | 7479.8 | 基线 |
| **`t64x128_bnt0`（block_m2=64）** | **5873.8** | **最优，-21.5%** |
| `t64x256_bnt0`（bm64） | 8424 | tile_n 太大 |
| `t128x128_bnt0`（bm128） | 10897 | tile_m=128 padding 浪费 |
| `t64x64_bnt0`（bm64） | 25608 | tile_n=64 灾难 |
| `t128x64_bnt0`（bm128） | 23274 | 同上 |
| `t32x128_w2_bnt0_kb3`（split-K=3） | 40459 | split-K 在大 M 下灾难 |

waves_per_eu 对 t64x128 几乎无影响（5930~5933）；`bnt0` 略胜 `bnt2`（5874 vs 6053）。

### 关键陷阱：`tile_m` 必须等于 `block_m2`（否则结果是垃圾）

第一次只改 `kernelName2=t64x128` 而没动 `block_m2=32` 时，**性能看着快但正确性 cos=0.0048（垃圾）**。根因：
- `fused_moe.py:1321` 传给 `flydsl_moe_stage2` 的 `sort_block_m` 来自 kernel 名解析，普通名里没编码 → 取 0；
- `moe_kernels.py:1954` `_sbm = sort_block_m if >0 else tile_m` → 退化成 `tile_m=64`；
- 但 stage2 的 `sorted_ids2` 实际是用 **配置里的 `block_m2=32`** 排出来的（步骤 3）→ 分块 padding 布局按 32，gemm2 按 64 读 → **错位**。

**修复 = 让 `block_m2` 与 `tile_m` 一致**（基线 t32x128 之所以对，正是因为 block_m2=32==tile_m=32）。配 `block_m2=64` 后：

| | 基线 t32x128 / bm32 | 优化 t64x128 / bm64 |
|---|---|---|
| 正确性 `check_result` | pass，cos=0.999990 | **pass，cos=0.999990，max_delta=0.0117** |
| `moe_gemm2_0` | 7479.8 us | **5873.8 us（-21.5%）** |
| e2e device/iter | 14545 us | **12931 us（-11.1%）** |

### 已落地改动

`aiter/configs/model_configs/hy3_fp8_pertensor_tuned_fmoe.csv` token=32768 行：
- `block_m2`: 32 → **64**
- `kernelName2`: `flydsl_moe2_afp8_wfp8_bf16_t32x128x64_atomic_w2_bnt0` → **`flydsl_moe2_afp8_wfp8_bf16_t64x128x64_atomic_bnt0`**
- `us2`/`us`/`tflops`/`bw` 按实测 breakdown 比例刷新为估计值（精确值需跑一次完整 tuner 复测）。

### 仍存在的差距与后续方向

优化后 flydsl gemm2 5874us 仍明显慢于 asm 的 down+reduce ≈ 3999us。剩余差距来自 flydsl JIT kernel 的 epilogue/调度效率（asm 是手写汇编）。

### codegen 层开关已逐一实测（无进一步收益）

用 monkeypatch 注入 `get_flydsl_kernel_params` 的覆盖项（real pipeline、带正确性校验），并对 `persist` 用一次性环境变量门控临时测后还原。全部基于已优化的 `t64x128_bnt0`（baseline gemm2 ≈ 5892us）：

| 开关 | 取值 | gemm2 us | 结论 |
|---|---|---|---|
| `use_async_copy` | True | — | **编译崩溃**（LLVM `Do not know how to expand this operator's operand`，async buffer-load→LDS 在 t64x128/fp8/gfx942 上 lowering 失败） |
| `n_per_wave` | 16 | 6760（更慢） | 只对 tile_m=16 有益；tile_m=64 反而增加 wave 数拖慢 |
| `persist`(`_persist_m`) | -1（round-robin） / 4 | 5892 / 5892（持平） | 大 M 并行度已充足，调度模式无影响 |
| `b_nt` | 0 vs 2 | 5874 vs 6053 | 已在步骤4扫过，bnt0 最优 |
| `waves_per_eu` | 0~4 | ≈5930（几乎不变） | 已扫过，无影响 |
| `mfma_variant` | mfma16k128/32k64 | — | 仅 fp4(tile_k=128) 路径用，fp8 不适用 |

→ **结论：gemm2 在 flydsl 框架内的可调空间已挖尽，t64x128_bnt0（≈5874us）是该 shape 的实际最优。** 进一步缩小与 asm 的差距需要改 flydsl gemm2 的 codegen 内核本身（epilogue/inner-loop 调度），属重型工作；或接受固有差距，转向占比同样大的量化（🅱，`data_to_scale` 3108us）和 gemm1。

---

## 步骤 5：优化后 stage2（t64x128）与 asm 的深入差异分析（同 tile 对齐）

优化后两边 tile 完全一致（asm down 的 `.co` 文件名也是 `BLOCK_TILE_SIZE_M=64-BLOCK_TILE_SIZE_N=128`），第一次可以做真正 apples-to-apples 的对比。

### 5.1 用 reduce 模式隔离 epilogue：差距不在写出，而在 GEMM 核

同样 t64x128、同样非原子 store（flydsl `reduce` 模式 = 和 asm down 一模一样的写出方式）实测：

| 写出方式（均 t64x128） | gemm2 核 us | +reduce us | 合计 |
|---|---|---|---|
| flydsl reduce（非原子 store） | **5563** | +679（`_topk_sum`） | 6242 |
| flydsl atomic（bf16 原子加） | **5892** | 0 | 5892 |
| **asm down（非原子 store）** | **3162** | +837（`final_reduce`） | 3999 |

**两个反直觉的结论（修正步骤 2 的旧判断）**：
1. **bf16 global 原子加开销其实很小**：atomic(5892) 仅比 flydsl 自己的非原子 store(5563) 慢 ~329us。步骤 2 在 t32x128 上观察到的"原子很慢"是被过小的 tile 放大的假象。atomic 仍是 flydsl 最优解（省掉 679us 的独立 `_topk_sum`）。
2. **真正的差距全在 GEMM 核**：同样非原子 store、同样 tile，flydsl 5563us vs asm 3162us，**差 2401us（慢 76%），与写出/原子无关**。

### 5.2 同 tile ISA 指令构成对比（最硬证据）

- flydsl：`~/.flydsl/debug/moe_gemm2_0/17_final_isa.s`（`FLYDSL_DUMP_IR=1` 重新 dump，t64x128 atomic）
- asm：`llvm-objdump -d` 上面那个 `BLOCK_TILE_SIZE_M=64-N=128` 的 `.co`

| 指标 | asm `moe_2stage_down` | flydsl `moe_gemm2_0` | 含义 |
|---|---|---|---|
| `v_mfma_f32_16x16x32_fp8_fp8` | **144** | **48** | asm 把 M-repeat / K-loop 静态展开 3×，直线代码、ILP 更高；flydsl 仍卷着循环 |
| `buffer_load_dwordx4`（16B 宽） | **36**（全部来自单一 buffer s[20:23]） | **9** | asm 把 A+W 的流式加载几乎全合并成 16B 宽读 |
| `buffer_load_dword`（4B 窄） | **4**（仅 scale/meta） | **38**（s[4:7]×18、s[36:39]×16…） | flydsl 主 A/X gather 也是 dwordx4，但额外发射大量窄读（scale / sorted_ids / gather 元数据 / 次级加载），窄读∶宽读 ≈ 38∶9 vs asm 4∶36 → 整体访存远不如 asm 集中合并 |
| `ds_write` / `ds_read`（LDS） | 21 / 16 | **36 / 40** | flydsl 的 CShuffle epilogue 走更重的 LDS 往返 |
| `s_waitcnt`（同步/等待） | **21** | **61** | flydsl ~3× 的等待点 → 流水停顿多（窄读 + LDS 往返造成更多依赖） |
| 反量化/打包 | `v_pk_mul_f32`×32 + `v_perm_b32`×32（内联紧凑） | 0（融进别处/LDS 路径） | asm 内联 dequant→pack→宽 store |
| 输出指令 | `global_store_dwordx4`×8（64 bf16/次，普通 nt） | `global_atomic_pk_add_bf16`×16（32 bf16/次，原子） | asm 一次写 2× 数据、指令数一半、且非原子 |
| VGPR | **208** | **82** | asm 用满寄存器 → 深度软件流水（边算边预取）；flydsl 寄存器少、更依赖 LDS、停顿更多 |
| LDS | 19456 B | 16640 B | — |

### 5.3 差距归因

按贡献从大到小：

按 HBM 带宽利用率估算：gemm2 总 HBM 流量 ≈ 6GB（输出 bf16 2.4GB + W2 ≈ 3.6GB + A 56MB），MI300X ≈ 5.3TB/s 理论上限 ≈ 1.13ms。**asm 3162us ≈ 36% 带宽利用，flydsl 5563us ≈ 20%**。gemm2 是带宽/调度受限（K=192 算术强度极低），差距即"达成带宽"之差，来源：

1. **软件流水深度（主因）**：asm VGPR=208 vs flydsl 82。asm 用大量寄存器把"预取下一块 + 计算当前块"深度重叠（`s_waitcnt` 仅 21）；flydsl 寄存器预算低、`s_waitcnt` 多 3×（61），等待 in-flight 加载时频繁停顿，达成带宽上不去。
2. **次级元数据窄读**：经 SGPR→参数精确映射核实，**A2/B2 矩阵数据两边都是 16B `buffer_load_dwordx4` 宽读**（flydsl `load_b_pack_k32` fp8 路径 vec_width=4、X 走 `buffer_copy_gmem16_dwordx4`）。flydsl 的 38 条 4B 窄读全是次级元数据：sorted_token_ids gather ×18 + a2_scale 逐行 ×16 + w_scale/杂项 ×4。其中 per_Tensor 下 a2_scale 本是标量却被展开成逐行向量读（16 条窄读 + 对应 waitcnt），是可优化的冗余。（注：flydsl 静态 48 MFMA 是 K-loop 部分卷循环，asm 144 是完全展开，静态宽读条数 9 vs 36 不等价，不作为"访存更碎"的硬证据。）
3. **CShuffle LDS epilogue**：flydsl 的 LDS 往返（ds 76 条）比 asm（37 条）重；atomic/reduce 两模式都有这部分。
4. **写出方式（次要，~329us）**：bf16 原子加 vs 宽 nt store——量级最小，且 atomic 反而省掉了独立 reduce，是 flydsl 的合理选择。

> 一句话：**优化后剩余的 ~1900us 差距，本质是 JIT 生成内核（浅流水 + 访存更碎 + 重 LDS epilogue，达成带宽 ~20%）与手写汇编（深流水 + 宽合并访存 + 紧凑 epilogue，达成带宽 ~36%）的固有差异，集中在 GEMM 核的访存调度，而非写出/原子。** 仅靠现有配置/开关无法消除，需在 flydsl codegen 层改寄存器调度/预取深度与次级加载合并（重型工作）。

---

## 步骤 6：flydsl 优化建议（按 ROI 排序）

### 第一梯队：工程性、低风险、立刻可做

1. **按 M 区间分流 kernel（最高 ROI）**。本 shape 实测：
   - 大 M（如 token≥8192）：asm 1-stage（e2e 8602us）比最优 flydsl 2-stage（12931us）快 ~33%。**大 M 直接用 asm**。
   - 小 M：flydsl `*_direct` 路径（kernel 内 topk 规约、不排序）反而是赢家（见 tuned 配置 token≤8 行）。
   - 建议在 tuned 配置/dispatch 里明确按 M 选择，而不是让 flydsl 2-stage 覆盖全 M 段。
2. **保证 `block_m2 == stage2 tile_m`（已落地）**。这是正确性前提，也顺带 -21.5% gemm2（步骤 4）。建议在 tuner 里把这个约束写死，避免再次产生错配的"看着快但结果错"的行。
3. **统一 stage1/stage2 的 block_m 省第二次排序**（步骤 3，~167us）。优先级低，但零风险时可顺手做。

### 第二梯队：flydsl codegen 改造（高价值、重型，需重测正确性）

针对 gemm2 达成带宽仅 ~20%（asm ~36%）：

4. **提高软件流水深度（最值博）**：当前 VGPR 仅 82、`s_waitcnt` 61。让 codegen 在 K 循环上做 double/triple-buffer 预取（多预取一两个 K-tile 到寄存器，与 MFMA 重叠），把 VGPR 预算往上提到 ~150+。这是 asm 用 208 VGPR 换深流水的核心，预计是缩小差距的主要抓手。
5. **减少次级窄读**：38 条 4B 窄读多来自 scale / sorted_ids / gather 元数据。per_Tensor 量化下 a2/w2 scale 是标量，应在 kernel 入口一次性读进 SGPR 广播，而不是循环里反复窄读；sorted_token_ids 也可一次性宽读进 LDS 复用。
6. **精简 CShuffle epilogue 的 LDS 往返**（ds 76 vs asm 37）：评估能否像 asm 那样直接在寄存器里 dequant(`v_pk_mul`)+pack(`v_perm`) 后宽 store，绕过 LDS 重排；或减少 epilogue 的 LDS stage 次数。
7. **`use_async_copy` 当前在 t64x128/fp8 上编译崩溃**（步骤 4）——若要走 async（buffer-load→LDS DMA）预取路线，需先修这个 LLVM lowering bug，它本可同时服务于 #4 的预取深度。

### 第三梯队：换 stage（与 gemm2 同量级的其他瓶颈）

8. **量化 🅱**：flydsl 路径里 `data_to_scale_kernel` 占 3108us（与 gemm2 节省同量级），是 per_Tensor 多 pass + global 原子求 scale 导致；可考虑融合成单 kernel / 减少原子。**下一个最该做的**。
9. **gemm1**：flydsl 2-stage 的 gemm1 走 CK，本身不慢，优先级低。

> 总结：**短期最划算是 #1（大 M 走 asm）+ #8（量化）**；要让 flydsl gemm2 本身追上 asm，核心是 #4（加深预取流水、提 VGPR），属 codegen 重型改造。

---

## 步骤 7：硬件计数器实测 —— 推翻"带宽受限"假设，确认 gemm2 是 **VALU 发射受限**（重大修正）

之前（步骤 5）把差距归因为"达成带宽 20% vs 36%、需加深流水/提 VGPR"。但加 VGPR 的实验（n_per_wave=64）反而 -22%，与"带宽受限"矛盾。这次用 `rocprofv3` 采集 gemm2（`moe_gemm2_0`，t64x128 atomic）的指令计数器，得到**确定性结论**：

| 计数器（16 次 dispatch 汇总） | 值 | 占总指令 | 与 MFMA 之比 |
|---|---|---|---|
| **SQ_INSTS_VALU** | 5.92e9 | **63%** | **12.8 : 1** |
| SQ_INSTS_LDS | 7.21e8 | 7.7% | 1.6 |
| SQ_INSTS_VMEM_RD | 4.45e8 | 4.7% | 1.0 |
| SQ_INSTS_MFMA | 4.62e8 | 4.9% | 1.0 |
| SQ_BUSY_CYCLES / GRBM_GUI_ACTIVE | ≈1.0 | SQ 近 100% 忙 | — |

**结论：每条 MFMA 要配 12.8 条 VALU（干净 GEMM 应 ~1-2），SQ 发射端被 VALU 占满 → gemm2 是 VALU-issue-bound，不是 HBM 带宽 / VGPR / 访存延迟受限。** 这解释了为何加 VGPR/预取全部无效：瓶颈在指令发射，不在内存等待。

### 7.1 VALU 来源（静态 ISA，单 M-block 1055 条 v_，与动态 12.8:1 吻合）

| VALU 指令 | 条数 | 来源 |
|---|---|---|
| `v_mul_f32` | 192 | 逐元素**标量** dequant（acc×sx×sw×tw）。**asm 用 `v_pk_mul_f32`×32 打包 2 个 f32 —— 6× 差距** |
| `v_cmp_*` + `v_cndmask` | ~211 | 每行 sentinel 有效性掩码（padding 行清零） |
| 地址 math（add/lshl/mul u32/i32） | ~196 | gather / LDS swizzle / 写出地址 |
| `v_bfe_u32`+`v_or`+`v_add3` | ~250 | sorted_token_ids 解码 + CShuffle 索引 |

根因一句话：**flydsl 的 gather + CShuffle codegen 为每个输出元素铺了大量标量 VALU（标量 dequant、逐行掩码、索引解码），把 SQ 发射占满；asm 用打包 dequant + 极少 VALU + 宽 store 把这部分压到最低。** 这与 K=192、VGPR、预取深度都无关。

### 7.2 本次落地的改动（stage2 codegen，安全、已验正确）

1. **dequant 标量重结合**（`write_row_to_lds`）：把 `(v*sx)*sw*tw` 改为先提 `sx_row = sx*tw`（每行不变量，循环外算一次），元素内只 `v*(sx_row*sw)`，每元素少 1 条 `v_mul_f32`。
2. **写出路径死代码下沉**（`store_pair`）：把仅 f16/gfx950 分支用到的 `t/idx0/idx_elem*`（含一条整型乘）移进对应 `else`，保证 gfx942 bf16 原子路径不发射（LLVM 实测已 DCE，属保险+表意）。

**实测（t64x128_atomic_bnt0，block_m2=64，正确性 pass cos=0.99999、max_delta 不变）：gemm2 5892 → 5785 us（-1.8%），e2e 12955 → 12834 us。** 稳定可复现。

### 7.3 为什么没能在一晚内追平 asm（诚实结论）

- **纯配置已到地板**：atomic/reduce × 全 tile/wave/bnt（匹配 block_m2）彻底扫完，最优就是 ~5785us（reduce 的 tile_n=256→e_vec=8 宽 epilogue 思路：gemm2 5871 但 +708 topk_sum，e2e 更差）。配置无法接近 asm 的 3162（核）/3999（含 reduce）。
- **要真正砍 VALU 需重构 epilogue**，风险高、不宜无人值守过夜落地：
  - **打包 dequant**（`v_pk_mul_f32`）：需把 epilogue 从"逐 (mi,ii) 单元素"改成"批量 ii 凑 vec、按 vec 乘 scale"，触及 CShuffle 的 LDS 布局，正确性风险大。
  - **全有效块快路径**：moe_sorting 把每个 expert padding 到 block_m 整数倍，**只有每个 expert 的最后一个 block 含 padding**，其余内部 block 全有效。可对"整块有效"的 block 走**免逐行掩码**的 epilogue（省掉 ~211 条 cmp/cndmask），是最大的一块安全空间，但需加运行时分支 + 两份 epilogue，需充分验证。

> **给次晨的行动建议（按 ROI）**：① 实现 7.3 的"全有效块快路径"（去掉大多数 block 的 sentinel 掩码 VALU），预计是最大单点收益；② 打包 dequant（对齐 asm 的 `v_pk_mul_f32`）；③ 二者都属 epilogue codegen 改造，应在白天有验证条件时分步落地、每步跑 `check_result`。本次已先行落地 7.2 的两项零风险改动并验证。

---

## 步骤 8：用 **等待周期(wait-cycle)** 计数器再修正 —— 既不是 VALU 也不是 LDS，而是 **全局访存延迟 + barrier 停顿（低占用）**

步骤 7 看"指令条数占比"（VALU 63%、12.8:1）得出"VALU 发射受限"。但这只说明 VALU **指令多**，不代表它在**关键路径**上。于是落地了"全有效块快路径"实测验证：

### 8.1 全有效块快路径实测：正确但**零加速**（关键反证）

实现了 block 级 `scf.if blk_all_valid`（uniform，barrier 安全）：整块有效时走免逐行掩码 epilogue，去掉 ~20% 的 VALU（cmp/cndmask + 索引）。结果：
- 正确性 pass（cos 0.99999、max_delta 不变）——快路径逻辑正确；
- **性能 5785 → 5887us，不升反微降**（多出的 `blk_all_valid` 探测 + 分支 + 两份 epilogue 代码）。

> **结论：砍掉 20% VALU 对 wall-clock 毫无帮助 → gemm2 不是 VALU-issue-bound。** SQ_BUSY≈100% 被误读了：它把 `s_waitcnt`/`s_barrier` 停顿也算作"忙"。该改动已保留为 **env 开关 `FLYDSL_MOE_STAGE2_FASTVALID`，默认关闭**（对 epilogue 占比大的 shape 仍可能有用）。
>
> 另注：全局强制免掩码**不可行**——padding 行 sentinel `t=tokens`，无谓词会让其原子写到 `out` 第 `tokens` 行（越界），触发缺页风暴/挂死。所以掩码只能在"整块确证有效"时去掉。

### 8.2 等待周期分解（`SQ_WAIT_*`，真正的瓶颈）

| 指标 | 占 `SQ_WAVE_CYCLES` | 含义 |
|---|---|---|
| **LDS 指令等待** `SQ_WAIT_INST_LDS` | **1.3%** | LDS bank conflict 虽 ~0.85/op，但**几乎不在关键路径** |
| **非-LDS 指令等待**（全局 VMEM `s_waitcnt`） | **21.8%** | 等全局访存（X/A2 gather + W2 加载）回来 |
| `SQ_WAIT_ANY`（含 barrier 等所有等待） | **~48%** | 近一半 wave 周期在等 |
| VALU+MFMA 单元忙 `SQ_VALU_MFMA_BUSY` | **~25%** | **ALU 闲置 75%** → 绝非计算/VALU 受限 |

> **最终定性（已三次用硬件计数器收敛）：gemm2 是 _访存延迟 + barrier 停顿_ 受限，根因是 _占用率太低_（每 workgroup 仅 4 waves，藏不住延迟），不是 VALU、不是 LDS bank conflict、不是 HBM 带宽峰值。** ALU 闲 75%、近一半周期在等待即铁证。

### 8.3 这把方向彻底厘清了——真正的抓手只有"提占用 / 加深延迟隐藏"

- 全局访存延迟（~22%）+ barrier 停顿要靠**更多在飞 wave**来藏。但本 shape 4 waves 已是 tile/VGPR 配置上限：`n_per_wave=64`（更少 wave）实测 -22%，配置空间也扫到地板（~5785us）。
- asm 用 **VGPR=208 + 完全展开**换来极深软件流水（`s_waitcnt` 仅 21），把同样的访存延迟隐藏掉，从而达成更高有效带宽。flydsl JIT 默认调度做不到这个深度。
- 因此**追平 asm 的唯一正道 = flydsl codegen 层重做 K 循环的多级寄存器预取/软件流水（提 VGPR 预算、减少 barrier 依赖）**，属重型改造，且前序 `use_async_copy` 在该配置上还有 LLVM lowering 崩溃需先修。

### 8.4 本轮净结果（代码状态）

- 保留步骤 7.2 两项零风险改动（dequant 重结合 + 死码下沉），gemm2 ≈ 5785us（run-to-run 抖动 ±80us）。
- 新增"全有效块快路径"`FLYDSL_MOE_STAGE2_FASTVALID`（**默认关**，正确、对本 shape 无收益）。
- 默认最优配置仍为 `t64x128_atomic_bnt0` + `block_m2=64`。
- **诚实结论：在不重做 codegen 软件流水的前提下，stage2 已到约 5785us 的工程地板，无法靠配置/局部 VALU/LDS 改动追平 asm 的 3162/3999us；下一步只有投入 K 循环预取流水的重型 codegen 改造。**
