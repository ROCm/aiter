# `pa_mqa_logits_fp4_prefill` VALU 优化日志

本文件记录对 `pa_mqa_logits_fp4_prefill.py` 内核后处理路径的 **VALU 指令削减** 迭代优化。
该内核在 gfx950 上是 **VALU-issue bound**（实测 `VALUBusy ≈ 97.6%`，`MfmaUtil ≈ 22%`，
L2 命中率 ≈ 94%，HBM 基本空闲），因此削减 VALU 指令数几乎线性转化为运行时收益。

## 测量方法（可复现）

固定工作负载：`bs=2 ctx=8000 n_q=8000`（`total_tokens=16000`，`grid=16000` CTA，
`heads=64 head_dim=128 block_k=256 kv_block_size=64`）。

- **正确性**（每次改动后必须过）：

  ```bash
  python op_tests/test_flydsl_pa_mqa_logits_fp4_prefill.py   # 4 个 correctness case, 要求 cos_exact>0.99
  ```

- **wall time**（GPU 需空闲，取多次 best）：

  ```bash
  python -m op_tests.bench_flydsl_pa_mqa_logits_fp4_prefill_valu --iters 50 --warmup 15 --reps 4
  ```

- **VALU 指令数**（与其它 GPU 负载无关，最可靠的削减度量）：

  ```bash
  printf 'pmc: SQ_INSTS_VALU SQ_INSTS_VALU_MFMA_F6F4 SQ_WAVES\n' > /tmp/pmc.txt
  rocprofv3 -i /tmp/pmc.txt --kernel-include-regex 'pa_mqa_logits_fp4_prefill.*' \
    -d /tmp/rpvalu -o v --output-format csv -- \
    python -m op_tests.bench_flydsl_pa_mqa_logits_fp4_prefill_valu --iters 4 --warmup 2 --reps 1
  # SQ_INSTS_VALU = 每次 dispatch 发射的 wave64 VALU 指令数
  ```

硬件：gfx950，256 CU × 4 SIMD16，wave64，2.4 GHz。峰值 VALU 发射率
≈ `256 × 2.4e9 × 1 = 6.14e11` wave64-instr/s。VALU 理论下限 = `SQ_INSTS_VALU / 6.14e11`。

## 结果汇总

| 步骤 | 改动 | SQ_INSTS_VALU | vs 基线 | wall best (us) | vs 基线 | 正确性 | 结论 |
|---|---|---|---|---|---|---|---|
| baseline | `relu*w+sum` 已用 `fma` 融合（上一轮成果） | 464,640,000 | — | 868.96 | — | cos=1.0 | 基准（本轮重测 wall；VALU 与旧基准一致） |
| opt1 | `_bperm_xor_add` 预计算 `lane*4`、用常量 xor 替代每次 `*4` | 464,640,000 | 0 | 862.1 | +0.1% | cos=1.0 | **回退**：编译器已自动优化，无收益 |
| opt2 | 把标量 `weight_scale` 折叠进 hoisted 的 `w_per_lane`（每 wave 一次），删除 `_post_process_nt` 里每 nt 的 `* weight_scale` | 456,960,000 | **-1.65%** | 852.79 | **-1.9%** | cos=1.0 | **采纳**：省 ~120 VALU/wave |
| opt3 | 窗口判定改用无符号区间技巧 `u32(out_token-local_start) < u32(win_len)` + 合并 writer 掩码为单次 select | 459,200,000 | +0.49% | — | — | cos=1.0 | **回退**：反而 +35 VALU/wave，编译器对原始两比较+两 select 的降级更优（`.to(Uint32)` 引入额外转换） |
| opt4 | 删除 `_extract_kvs_scales`：不再软件 `(packed>>8*nt)&0xFF` 抽取 kv-scale，直接把 packed dword 作为 `scaleB`，用 MFMA 的 `opselB=nt` 硬件字节选择 | 448,768,000 | **-3.42%** | 847.34 | **-2.5%** | cos=1.0 | **采纳**：省 ~128 VALU/wave；相对 opt2 再 -1.79% |
| opt5 | 地址计算里对 `kv_block_size`(64) 和 4 的 `//`/`%` 全部改为移位/掩码（操作数恒为非负 token/字节偏移） | 383,872,000 | **-17.4%** | 807.22 | **-7.1%** | cos=1.0 | **采纳（最大单笔收益）**：有符号 floordiv/rem by 2^k 的符号修正序列每 wave 约 1014 VALU，编译器无法证明非负故未消除 |
| opt6 | kv buffer_load 的 nt/k_tile 偏移改用立即数 soffset：`N_PHYS==1` 保证 4 个 nt 共享同一物理块且 `token_in_block` 无回绕，故 tile(nt,k_tile) 相对 nt=0 基址只差编译期常量 `k_tile*_stride_kv_ktile + nt*MFMA_N*_kv_chunk_bytes`。base VGPR 偏移只算一次，每个 nt 的常量增量交给指令 `soffset`（HW 免费相加，nt*256≤768 可进 12-bit 立即数），删除 nt=1..3 的 `_mod_kb`+地址乘加 | 376,000,000 | **-19.1%** | 803.46 | **-7.5%** | cos=1.0 | **采纳**：相对 opt5 再 -2.05%（-123 VALU/wave），wall 810.28→803.46（同机重测基线）。VGPR 仍 52。编译器此前未自动把 nt 常量增量折进 inst_offset |

> 注：本日志的 baseline 已经包含"把 `mul+add` 融合成 `fma`"这一轮优化（相对最初的
> `mul+add` 版本，VALU 从 540.10M → 464.64M，-14%，wall 984.7 → 861.2 us，-12.8%）。

## 优化候选点（VALU 来源）

后处理路径 `_post_process_nt` + 每 chunk 的 `_extract_kvs_scales` 是 VALU 主要来源。
每 wave 每 chunk ≈ 227 条 VALU（464.64M / 64000 waves / 32 chunk）。候选：

1. ~~`_bperm_xor_add`：每次归约里 `peer_byte = (lane ^ sh) * 4` 的 `* 4` 每次重算。~~
   （opt1 已试，编译器自动优化，无收益，已回退。）
2. ~~标量 `weight_scale` 每 nt 乘一次 → 折叠进 hoisted 的 `w_per_lane`。~~
   （opt2 已采纳，-1.65%。）
3. ~~`_extract_kvs_scales`：逐 nt 的 `>> 8*nt` + `& 0xFF`。~~
   （opt4 已采纳：改用 MFMA `opselB` 硬件字节选择，整段删除，-1.79%。）
4. 窗口 store 地址：`in_window`/`is_writer` 的比较 + 两次 `select`（每 nt ~6 VALU）。
   opt3 已试无符号技巧但反而更差、已回退；此处编译器降级已较优，暂无进一步空间。
5. `relu + fma` 主路径（每 chunk ~128 VALU）：`maximumf(4) + fma(4)` per (mi,nt)，是最大来源，但为算术本质（16 个 relu + 16 个 madd），已是理论下限，难以削减。
6. q-scale 抽取 `qs_dws[..] >> 8*(mi%4)`：可同样用 `opselA` 硬件字节选择替换，但该抽取是 **每 wave hoist 一次**（仅 m_tiles 条），VALU 收益可忽略，暂不做。
7. ~~地址计算里 `//kv_block_size`、`%kv_block_size`、`//4` 的有符号除/模。~~
   （opt5 已采纳，-17.4%，最大单笔收益。）
8. ~~kv buffer_load 每个 nt 重算 `token_in_block`+地址乘加。~~
   （opt6 已采纳：nt/k_tile 的偏移差是编译期常量，改用指令 `soffset` 立即数，
   base 只算一次，-2.05%。）

## ISA/IR 级别分析方法（opt5 的来源）

直接统计 flydsl 编译产物（`~/.flydsl/cache/launch_pa_mqa_logits_fp4_prefill_*/*.pkl`
里的 `_source_ir` MLIR）**主循环 `scf.for` body 内**的算子直方图，是定位 VALU 的最快手段：

```python
import pickle
d = pickle.load(open(".../<hash>.pkl", "rb"))
open("/tmp/k.mlir", "w").write(d._source_ir)   # 内核 MLIR（lowering 前）
```

某轮循环 body（一个 chunk）实测：`math.fma×64`、`arith.muli×65`、`arith.addi×53`、
`arith.maximumf×16`（=64 标量 `v_max`）、`floordivsi×6`、`remsi×4`。**整数地址算子总数
甚至超过 fma**。其中 muli 全是 `var × 2^k`（会被后端强度削减为移位，廉价），但
`floordivsi`/`remsi` 是**有符号**除/模，即使除数是 2 的幂也会展开符号修正序列——这正是
opt5 命中的隐藏大头。核心 `relu(64) + fma(64)=128` 是算术本质，已是下限。
