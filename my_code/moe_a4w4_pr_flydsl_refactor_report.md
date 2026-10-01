# MoE A4W4 FlyDSL 重构报告

## 1. 工作范围

本文记录针对 [yanguahe/aiter#1](https://github.com/yanguahe/aiter/pull/1)
开展的 FlyDSL 代码重构、重构前后性能对比、结果精度和 ISA 一致性验证。

- 源分支：`hyg/moe_a4w4_pr`
- 源 HEAD：`04cc526b8f06f1e54f836718964e8f2e2c844fa8`
- PR merge-base：`105615e75430e171bc3854dfcb7abf4fe92dc645`
- 重构 worktree：`a4w4-refactor`
- 重构分支：`codex/moe_a4w4_pr_refactor`
- AITER 固定的 FlyDSL 版本：`0.3.4.1`
- 测试机器：`a07-3`
- 测试容器和目录：`hyg_fyd_e2e:/app/aiter`

本次工作全部在独立 worktree 中完成，原始 `aiter` 工作区没有被修改，也没有创建
commit。

重构范围仅限 PR 修改的非 `my_code/` 实现。最终通过逐行范围审计确认：五个重构文件中
所有被修改或删除的原 HEAD 行都位于 PR 修改范围内。PR 中其余非 `my_code/` 文件也已
检查，但不需要进行 FlyDSL kernel 代码重构，因此保持不变：

- `.gitattributes`：repository text/EOL 属性。
- `aiter/configs/tuned_grouped_fmoe.csv`：tuning 数据。
- `aiter/fused_moe.py`：reference 的 `return_per_route` 接口处理。
- `aiter/test_common.py`：profiler 结果处理。

本次重构参考了 `rocm/main:.claude/skills/` 中以下规范：

- `flydsl-kernel-authoring`
- `flydsl-tile-programming`
- `kernel-code-cleanup`
- `llvm`
- `api-stability`
- `gemm-optimization`
- `oob-detection`
- `isa-resource-diff`
- `llvm/references/arch-gfx1250.md`

## 2. 重构内容

### `aiter/ops/flydsl/grouped_gemm_mxfp4.py`

- 将 A-preshuffle capability 判断拆分为负责解析选项的 public helper 和接收已解析
  launch 参数的 `_supports_gfx1250_a_preshuffle_resolved`。
- 避免 optimized 路径重复解析相同的 env/CSV 参数。
- 保持 generic launcher 的行为和 public function signature 不变。

### `aiter/ops/flydsl/grouped_moe_gfx1250.py`

- 合并 GEMM1/GEMM2 A-preshuffle 的公共 enable 条件。
- 删除重复的初始赋值和冗余 Boolean 表达式。
- 保持 producer 选择、layout、fallback 条件、buffer 顺序和 launch 参数不变。

### `aiter/ops/flydsl/kernels/moe_fused_route_quant_scatter.py`

- 删除约 127 行重复的 MX E8M0 scale 生成代码，统一复用
  `quant_utils.emit_mx_e8m0_scale`。
- 删除 PR 新增代码对私有 `_DTYPE_CFG` 和 `_M` 的直接依赖。
- 将 PR 新增代码中的 raw vector wrapper 替换为公开的 `fx.Vector` API。
- 将 PR 新增代码中的 `ArithValue` wrapper 替换为 typed `fx.Int32` 和
  `fx.Float32`。
- 将必须保留的 raw `scf` import 局部化，并记录保留原因：对应 module-level IR
  emitter 不经过 FlyDSL AST rewrite。

### `aiter/ops/flydsl/kernels/mxfp4_preshuffle_gfx1250_tdm.py`

- 删除对内部 `mega_moe_gfx1250.vector` 的依赖，改用 `Vec(...)`。
- 将 raw LLVM `s.setreg` import 局部化。
- 删除未使用的常量和临时变量。
- 保留 tuned kernel 的 raw LDS byte pointer 和 index lowering。机械替换为
  `peek().ptr` 和 `fx.Index` 后，t256/E96 specialization 可以运行，但 t192/E64
  specialization 会发生 segmentation fault。因此撤销了该不安全迁移，并在代码旁记录
  这一语义边界。

### `aiter/ops/flydsl/moe_kernels.py`

- 将 PR 新增的 GEMM1 A-preshuffle 路径提取为
  `_TokenQuantThenScatterPath`：
  - 每个 source token 只量化一次；
  - invert route-to-row map；
  - 将 payload 和 ScaleA scatter 到 tuned GEMM 使用的布局。
- 将 PR 新增的 GEMM2 A-preshuffle 路径提取为
  `_GroupedRowsQuantPath`：
  - 条件满足时使用 rowgroup quant kernel；
  - 否则保留 compact-route 加 scatter fallback。
- 保持原有 prequantized、token-multidest、generic route 和 generic non-route
  路径不变。
- 保持 kernel builder、allocation shape、launch 参数和 fallback predicate 不变。

最终 worktree diff 只包含以下五个实现文件和本文档：

```text
aiter/ops/flydsl/grouped_gemm_mxfp4.py
aiter/ops/flydsl/grouped_moe_gfx1250.py
aiter/ops/flydsl/kernels/moe_fused_route_quant_scatter.py
aiter/ops/flydsl/kernels/mxfp4_preshuffle_gfx1250_tdm.py
aiter/ops/flydsl/moe_kernels.py
my_code/moe_a4w4_pr_flydsl_refactor_report.md
```

## 3. 测试方法

baseline 数据来自应用重构前的 PR HEAD。final 数据来自将 worktree 中最终五个实现文件
copy 到 `hyg_fyd_e2e:/app/aiter`，并逐个核对本地与容器内 SHA256 之后的测试结果。

baseline 日志保存在 a07-3：

```text
/app/aiter/.codex_refactor/baseline_20261001/
```

final 日志保存在：

```text
/app/aiter/.codex_refactor/final_scope_20261001/
```

其中 `e96_random_retry.log` 是有效的 E96 random baseline 日志；第一次生成的
`e96_random.log` 没有完整测试结果。

每批性能测试前均运行 `/data/yanguahe/code/gpu_users.sh` 和
`rocm-smi --showuse --showmemuse`，确认没有 GPU/KFD 用户、GPU use 为 0%、VRAM
allocated 为 0%。每批测试结束后再次检查，没有发现测试残留进程。

公共测试命令如下：

```bash
FLYDSL_RUNTIME_ENABLE_CACHE=0 \
AITER_USE_GROUPED_GEMM=1 \
AITER_GROUPED_DEBUG=0 \
ENABLE_CK=0 \
FLYDSL_DUMP_IR=0 \
AITER_LOG_MORE=1 \
AITER_MOE_EXPERT_BALANCE=true \
AITER_FLYDSL_MOE_EXPERT_SCHEDULING_MODE=1 \
python3 -u my_code/test_flydsl_grouped_gemm_gfx1250.py \
  --scenario bench \
  --data-format a4w4 \
  --experts <experts> \
  --tokens <tokens> \
  --topk <topk> \
  --model-dim 7168 \
  --inter-dim <inter_dim> \
  --act silu \
  --no-bias \
  --no-check-aot-cache \
  --iters 20 \
  [--const-init 0]
```

三组测试规模分别为：

```bash
# E64/T1536/topk8/I2048
--experts 64 --tokens 1536 --topk 8 --inter-dim 2048

# E96/T16384/topk6/I3072
--experts 96 --tokens 16384 --topk 6 --inter-dim 3072

# E256/T16384/topk8/I2048
--experts 256 --tokens 16384 --topk 8 --inter-dim 2048
```

random 测试不传 `--const-init`；const0 测试增加 `--const-init 0`。profiler 排除
一次 warm-up launch 后，报告 19 次实际测量的平均时间。

## 4. 重构前后性能

表中的负数表示重构后更快。每项都是一次 20-iteration profiler 测量，低于约 1% 的
变化应视为机器状态和动态时钟带来的正常波动。

| Shape | Data | GEMM1 before | GEMM1 after | GEMM1 delta | GEMM2 before | GEMM2 after | GEMM2 delta | MoE before | MoE after | MoE delta |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| E64/T1536/topk8/I2048 | random | 93.935 us | 89.274 us | -4.661 us (-4.96%) | 68.443 us | 67.778 us | -0.665 us (-0.97%) | 234.87 us | 227.44 us | -7.43 us (-3.16%) |
| E64/T1536/topk8/I2048 | const0 | 78.552 us | 76.490 us | -2.062 us (-2.63%) | 59.281 us | 61.131 us | +1.850 us (+3.12%) | 210.66 us | 208.89 us | -1.77 us (-0.84%) |
| E96/T16384/topk6/I3072 | random | 732.979 us | 722.253 us | -10.726 us (-1.46%) | 437.359 us | 450.826 us | +13.467 us (+3.08%) | 1432.78 us | 1435.57 us | +2.79 us (+0.19%) |
| E96/T16384/topk6/I3072 | const0 | 566.256 us | 566.848 us | +0.592 us (+0.10%) | 361.176 us | 360.821 us | -0.355 us (-0.10%) | 1173.20 us | 1172.84 us | -0.36 us (-0.03%) |
| E256/T16384/topk8/I2048 | random | 688.774 us | 696.339 us | +7.565 us (+1.10%) | 500.089 us | 499.186 us | -0.903 us (-0.18%) | 1478.41 us | 1484.93 us | +6.52 us (+0.44%) |
| E256/T16384/topk8/I2048 | const0 | 538.914 us | 536.856 us | -2.058 us (-0.38%) | 402.121 us | 400.323 us | -1.798 us (-0.45%) | 1212.65 us | 1206.21 us | -6.44 us (-0.53%) |

测试没有显示稳定的性能回退。E96 const0 的 GEMM1、GEMM2、MoE e2e 从
`566.256 / 361.176 / 1173.20 us` 变为
`566.848 / 360.821 / 1172.84 us`，基本不变。

## 5. 重构前后结果精度

所有 random 和 const0 case 的 MoE output hash 在重构前后均 bitwise 一致。
random 使用原有的 `logits_diff < 0.01` gate；const0 输出与 reference bitwise 一致。

| Shape | Data | Baseline accuracy | Refactored accuracy | Baseline MoE hash128 | Refactored MoE hash128 | Reference hash128 | Result |
|---|---|---|---|---|---|---|---|
| E64/T1536/topk8/I2048 | random | `logits_diff=3.3849e-06`, `rel_l2=2.6019e-03` | 相同 | `5069dae4a9dc8e4eeafe5773d2694405` | `5069dae4a9dc8e4eeafe5773d2694405` | `5a378a8f5577e415d7504b0e45a03938` | pass |
| E64/T1536/topk8/I2048 | const0 | `logits_diff=0`, `rel_l2=0` | 相同 | `6bebf6409ef198fe1a0255681f4f784f` | `6bebf6409ef198fe1a0255681f4f784f` | `6bebf6409ef198fe1a0255681f4f784f` | bitwise equal |
| E96/T16384/topk6/I3072 | random | `logits_diff=3.3980e-06`, `rel_l2=2.6069e-03` | 相同 | `1556fc617347e2dabc9cff19dbfd822b` | `1556fc617347e2dabc9cff19dbfd822b` | `1a5d22911ba167160b4f2c12092a5193` | pass |
| E96/T16384/topk6/I3072 | const0 | `logits_diff=0`, `rel_l2=0` | 相同 | `21291d9023c8af8a6324fe20f346a967` | `21291d9023c8af8a6324fe20f346a967` | `21291d9023c8af8a6324fe20f346a967` | bitwise equal |
| E256/T16384/topk8/I2048 | random | `logits_diff=3.3842e-06`, `rel_l2=2.6016e-03` | 相同 | `a62355d90642228b640e688fedd17538` | `a62355d90642228b640e688fedd17538` | `ae8a7e815af826cc2167172d3cdde40d` | pass |
| E256/T16384/topk8/I2048 | const0 | `logits_diff=0`, `rel_l2=0` | 相同 | `21291d9023c8af8a6324fe20f346a967` | `21291d9023c8af8a6324fe20f346a967` | `21291d9023c8af8a6324fe20f346a967` | bitwise equal |

random GEMM1 output hash 在重构前后也完全一致：

| Shape | GEMM1 output hash128 before/after |
|---|---|
| E64/T1536/topk8/I2048 | `aa5af718fdd2ab981a4e6c7c0f3537ee` |
| E96/T16384/topk6/I3072 | `fe33745fff774ee077f3893ed331625f` |
| E256/T16384/topk8/I2048 | `90867ecd573480b8eac23a0be2aad119` |

在三个 const0 case 中，GEMM1、GEMM2 和最终 MoE output 均与对应 reference
bitwise 一致。

## 6. ISA 一致性

E96 const0 pipeline 在重构前后分别使用下面的方式重新编译和 dump：

```bash
AITER_USE_GROUPED_GEMM=1 \
AITER_GROUPED_DEBUG=0 \
ENABLE_CK=0 \
FLYDSL_DUMP_IR=1 \
FLYDSL_RUNTIME_ENABLE_CACHE=0 \
AITER_LOG_MORE=0 \
AITER_MOE_EXPERT_BALANCE=true \
AITER_FLYDSL_MOE_EXPERT_SCHEDULING_MODE=1 \
python3 -u my_code/test_flydsl_grouped_gemm_gfx1250.py \
  --scenario bench \
  --data-format a4w4 \
  --experts 96 \
  --tokens 16384 \
  --topk 6 \
  --model-dim 7168 \
  --inter-dim 3072 \
  --act silu \
  --no-bias \
  --no-check-aot-cache \
  --iters 2 \
  --const-init 0
```

dump 目录：

```text
/app/aiter/.codex_refactor/isa_before_e96/
/app/aiter/.codex_refactor/isa_final_e96_scope/
```

九个 final ISA 文件全部逐字节一致：

| Kernel | SHA256 |
|---|---|
| `a8w4_tdm_fp4_t256x256x256_w2x2_b4_K7168_e96_act1_cn4_prefetch_eb8_apre_sh_rcw_mg4_fc28_xdl0_reuse_ostore2p_s4` | `9e70d5b2256958618e4240873722bbdebfbf6c83a84953d259e3e84e7480d934` |
| `a8w4_tdm_fp4_t256x256x256_w2x2_b4_K3072_e96_cn4_prefetch_apre_sh_mg4_fc28_ostore2p_s3_ow2` | `8f8455edccdcf71128dd239461131f599f08a8cb5f51a7d423fc7c1ad3ca3161` |
| `moe_quant_token_fd7168_fp4_pk8_hidtdm7` | `db5eaca22c7bcae060789e7f526dddb77ea320adf0472546f77cd8d00fb282b4f` |
| `moe_invert_route_rows_tk6` | `44163558fffc36f3ae125c72c0a53b318725ef32b224d56e2d48624ec379f6f8` |
| `moe_scatter_preshuffled_a_fd7168_r32_lds_pe7_slds_skipempty` | `089f36a48fe2499155d69c97dc5bb41eefa7c2c0790beb854851f65f77dfe78c` |
| `moe_quant_preshuffled_a_fd3072_rpw2_pf2_direct_hidtdm6_otdmw2` | `ee5da153fa72a22b851b1740a6d832ff8fc6e655f7facc0ce4a01b2f9f091ae8` |
| `moe_route` | `193d7b14a94b2fd1200324260117f5c32b89d5b4dc150192370baf3e3d702248` |
| `moe_contiguous_psum_remap` | `7bdc1e48507430e25d4ff39a1033941516e7d32680ceb75b53ba0b7e91b01b88` |
| `moe_gather_reduce_bf16_d7168_tk6_sk1_v4_wbf16_frlds` | `6ca96a46b3b2c6ef3caf955a03db46fcc47f6635ab94480852c8fd5a22a016940` |

这说明重构没有改变 E96 pipeline 中 GEMM、quant、scatter、route 或 gather kernel
最终生成的指令，也没有改变 code object 的 resource declaration。

## 7. 其他验证

完成最终 scope cleanup 后，单独验证了原有 prequantized scatter 路径，参数为
`E=1`、`max_m=32`、`feat_dim=256`、`topk=2`、`wmma_rep=2`：

```text
payload_equal True
scale_equal True
payload_hash e0b7524cee636c7bda710eee21c8a6b9
scale_hash bdeb00d75e535487b77f3108be1b3d1a
```

最终静态检查：

```text
python -m ruff check ...     passed
python -m compileall -q ...  passed
git diff --check             passed
```

为避免越过 PR scope，没有将 whole-file formatter 建议写回到文件。远端测试过程中没有执行
`git clean`、`git checkout .` 或其他广泛的 repository cleanup 命令。
