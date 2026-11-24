# MegaMoE TP (gfx1250) benchmark 说明

MI455X (gfx1250, 256 CU) x 4，TP4。对比对象：

- **mega**：`aiter/ops/flydsl/kernels/mega_moe_tp_1250/`（一个常驻 kernel 完成通信 + 路由 + GEMM1/2 + reduce）
- **baseline**：aiter 的 `fused_moe` + 通信（RCCL / custom all-reduce(Lamport) / Gluon all-reduce、custom all-gather / reduce-scatter 里取最快）

两种模式：

- `ar`：每个 rank 都有全部 T 个 token，输出也是全部 token（MoE + all-reduce）
- `ag_rs`：每个 rank 只有自己的 T/4 个 token（all-gather + MoE + reduce-scatter）

计时方式：CUDA graph 里连续 20 次调用，replay 5 次取最小，4 个 rank 取最大（`torch.cuda.Event`）。baseline 的 ag_rs 在 >= 8192 tokens 用 eager 计时。

---

## 1. 环境

```bash
cd /home/binding2/moe_1026/aiter                      # 分支 binding_mega_moe_tp_gfx1250_up1109
export PYTHONPATH=/home/binding2/moe_1026/pydeps/flydsl053:$PWD   # FlyDSL 0.5.3（upstream 当前 pin 的版本）
```

- **本目录下的 `bench/` 是全部 benchmark 脚本的快照**（和本 README 一起随 kernel 代码保存）。脚本内部用两个固定路径互相调用：
  shell / 工具脚本在 `/tmp/mmtp/`，Python 入口脚本在 `/home/binding2/moe_1026/scripts/`。
  **换机器 / 重启后先把它们放回去**（在仓库根目录 `aiter/` 下执行）：

  ```bash
  B=aiter/ops/flydsl/kernels/mega_moe_tp_1250/bench
  mkdir -p /tmp/mmtp /home/binding2/moe_1026/scripts /home/binding2/moe_1026/data_1114/raw
  cp $B/*.sh $B/tab_*.py $B/rowcfg.py $B/rebuild_jit.py $B/gen_fixed_csv.py /tmp/mmtp/
  cp $B/run_mm.py $B/baseline_moe_ar.py $B/baseline_m31_sh8.py $B/compile_mm.py $B/pad_inter.py $B/gpu_free.sh \
     /home/binding2/moe_1026/scripts/
  ```

  （工作区里另有一份持久副本：`/home/binding2/moe_1026/scripts/` 和 `scripts/mmtp/`，含各轮实验的全部 `r2xx_chain.sh`。
  改了脚本之后记得同步回 `bench/`。）

- Python 入口脚本：`run_mm.py`（mega）、`baseline_moe_ar.py`（全 MXFP4 baseline）、
  `baseline_m31_sh8.py`（M3.1 A8W8 shared 的 baseline）、`compile_mm.py`（离线编译）、`pad_inter.py`（被 `run_mm.py` 引用）。
  它们从 `op_tests/flydsl_tests/test_flydsl_grouped_gemm.py` 取权重生成和 fp32 参考。
- 合并 upstream 之后要重建 JIT 模块（它们不会自动重建）：

  ```bash
  AITER_REBUILD=1 python /tmp/mmtp/rebuild_jit.py module_custom_all_reduce module_custom_all_reduce_gfx1250 module_moe_asm module_quant
  rm aiter/jit/module_aiter_core.so && python -c "import aiter"
  python /tmp/mmtp/gen_fixed_csv.py /home/binding2/moe_1026/data_1114/tuned_grouped_fmoe_fixed_<日期>.csv   # baseline 用的合并 tuned 表
  ```

- 输出：原始日志 `data_1114/raw/<tag>*.log`，汇总 csv `data_1114/perf_<tag>*.csv`，每轮总结 `data_1114/opt_summ_1120_v<N>.txt`。

## 2. 模型 shape（脚本里的 `<model>`）

| model | 参数 | 说明 |
|---|---|---|
| `dsv41` | `--model-dim 5120 --experts 384 --topk 6 --inter-total 2304` | DSv4.1，inter 576/rank |
| `m31` | `--model-dim 6144 --experts 129 --topk 5 --inter-total 3072 --shared-expert` | MiniMax M3.1，128 routed + 1 shared，inter 768/rank |
| `kimi31` | `--model-dim 3584 --experts 896 --topk 16 --inter-total 3072 --act situv2` | Kimi 3.1，inter 768/rank |

单卡验证时 `--inter-total` 用每 rank 的值（576 / 768 / 768），即 TP4 每个 rank 的 GEMM 形状。

## 3. 一键 benchmark

### 3.1 M3.1 + A8W8 shared expert（当前重点）

```bash
bash /tmp/mmtp/bench_m31_sh8.sh <tag> ["ag_rs ar"] ["16 32 64 128 256 512 16384 32768"] [both|base|mega]
# 例: bash /tmp/mmtp/bench_m31_sh8.sh r300
```

依次跑：baseline B1（`sep`：routed `fused_moe` E128/top-4 + shared expert 单独的 MXFP8 dense 链路）、
baseline B2（`fuse`：分支 `binding_moe_m3.1_opt_gfx1250` 的 `fused_moe(shared_w*)`，只到 512 tokens）、
mega（`--sh8`），最后打印表并写 `data_1114/perf_<tag>_m31sh8.csv`
（列：`mode,tokens,b1_sep_us,b1_sep_moe_us,b2_fuse_us,b2_fuse_moe_us,mega_sh8_us,speedup_vs_b1,speedup_vs_b2,err,logits_diff`）。

只重跑 mega：最后一个参数给 `mega`（复用同一个 tag 已有的 baseline 日志）。

### 3.2 全 MXFP4（三个模型）

```bash
bash /tmp/mmtp/bench_1120.sh <model> <ar|ag_rs> <tag> ["<tokens>"] [both|base|mega]
bash /tmp/mmtp/full_r200.sh <tag>          # 3 模型 x 2 模式 x 8 个 token 数，写 data_1114/perf_<tag>_all.csv
```

### 3.3 等 GPU 安静后再跑

机器是共享的。`quiet_run.sh` 等到 4 张卡连续 N 分钟没有任何进程才启动命令：

```bash
nohup bash /tmp/mmtp/quiet_run.sh 4 4 bash /tmp/mmtp/bench_m31_sh8.sh r300 > /tmp/mmtp/r300.log 2>&1 &
#                         安静分钟数 ^ ^ 最多等几小时
```

## 4. 分步命令

### 4.1 mega：单卡验证（改过 kernel 或用新参数时，4 卡之前必须先做）

```bash
bash /tmp/mmtp/tp1g.sh <model> <ar|ag_rs> "<tokens>" "<MEGA_CFG>"
MM_EXTRA="--sh8" bash /tmp/mmtp/tp1g.sh m31 ag_rs "16 64 512 32768" "$(python3 /tmp/mmtp/rowcfg.py m31 ag_rs 64 | tail -1)"
```

- 在 GPU0 上以 TP1 跑 TP4 的每 rank 形状，和 fp32 参考比较；要求 `err=0` 且 `logits_diff < 1e-3`，否则打印 `STOP`。
- 第 4 个参数是 `MEGA_CFG` 覆盖。TP1 查不到 TP4 的 tuned 行，所以要把该 token 数实际会用的 tuned 行拷进来：
  `rowcfg.py <model> <mode> <tokens>` 会打印这一行。**4 卡要跑的每个 (token 数, 参数组合) 都要先用完全相同的参数做单卡验证。**
- 每次运行后检查 dmesg 新增的 GPU fault（`dmf.sh`），以及运行期间 GPU 上是否出现别的进程（标 `DIRTY`，结果作废）。

### 4.2 mega：4 卡计时 / 参数扫描

```bash
bash /tmp/mmtp/sweepm.sh <model> <ar|ag_rs> <tag> "<tokens>" "<cfg1>" "<cfg2>" ...     # cfg "" = 默认
MM_EXTRA="--sh8" bash /tmp/mmtp/sweepm.sh m31 ag_rs r301 "16 32 64" "" "s1.num_buffers=4"
```

每个 cfg 起一个 torchrun（4 进程），日志 `data_1114/raw/<tag>_<i>.log`，行格式
`T=<tokens> err=<kernel 错误位> rel_l2=.. logits_diff=.. us=<每次调用 us>`。
同一批比较必须用**同一个 token 列表**（列表里最大的 token 数决定 arena 大小，会影响小 token 的时间）。

直接调用：

```bash
torchrun --nproc_per_node=4 ../scripts/run_mm.py [--ar] <shape 参数> --tokens 16 64 256 --check --bench [--sh8] [--stamps]
```

`--check` 和参考比较；`--bench` 计时；`--sh8` M3.1 的 shared expert 用 A8W8；`--stamps`（配合 `MEGA_TS=1`）打印 kernel 内各阶段时间戳。

### 4.3 mega：阶段时间戳（kernel 自己的计时器，不是 profiler）

```bash
MM_EXTRA="--sh8" bash /tmp/mmtp/stampsm.sh m31 ag_rs <tag> "16 64" ""
```

输出每个阶段结束时间的 中位数/最大值（us，各 CTA 从自己启动算起）：
`agrows/ag`（AG 推送）、`agwait`（等齐）、`b1`（路由计数）、`scan`、`place`（放置）、`b2`、`G1`、`G2`、`rs`、`y`。
各 rank 的启动有错位，只看相对量。

### 4.4 baseline

```bash
# 全 MXFP4
torchrun --nproc_per_node=4 ../scripts/baseline_moe_ar.py <shape 参数> --mode ar|ag_rs --tokens ...
# M3.1 A8W8 shared
bash /tmp/mmtp/bench_sh8_base.sh <ar|ag_rs> <tag> ["<tokens>"]
torchrun --nproc_per_node=4 ../scripts/baseline_m31_sh8.py --model-dim 6144 --experts 129 --topk 5 --inter-total 3072 \
    --shared-expert --mode ar --variant sep|fuse --tokens ...
```

baseline 日志行：
`T=.. moe=<只 MoE> ar_rccl= ar_ca= ar_gluon= lamport=.. moe+ar=<总> (<用了哪种通信>)`（ar），
`T=.. moe= ag= rs= ag_rccl= ag_ca= rs_rccl= rs_ca= rccl_tot=<只用 RCCL 的旧口径> ag+moe+rs=<总>`（ag_rs）。
baseline 运行时用环境变量 `AITER_CONFIG_GROUPED_FMOE` 指向合并好的 tuned 表（脚本里已设置），避免 aiter 扫描并改写源 csv。

### 4.5 离线编译和驻留检查（不占 GPU）

```bash
export FLYDSL_GPU_ARCH=gfx1250
HID=6144 EXP=129 TOPK=5 INTER_TOTAL=3072 ACT=swiglu MEGA_CFG="ag_ll=1,rs_ll=1,sh8=1,cta_per_cu=2" \
  FLYDSL_DEBUG_DUMP_ASM=1 FLYDSL_DUMP_DIR=/tmp/mmtp/asm_v python ../scripts/compile_mm.py 64
grep -m1 amdhsa_next_free_vgpr /tmp/mmtp/asm_v/*/*final_isa.s        # VGPR 数
grep -m1 amdhsa_group_segment_fixed_size /tmp/mmtp/asm_v/*/*final_isa.s   # LDS 字节
bash /tmp/mmtp/compile_all.sh                                        # 三个模型的默认路径全部编译一遍
```

常驻 kernel 要求所有 CTA 同时驻留：**VGPR 数 <= 1024 / 每 SIMD 的 wave 数**（每 CU 2 个 CTA：512；3 个：341；4 个：256），
**LDS x 每 CU 的 CTA 数 <= 320KB**。超了不会报错，表现为等待超时（`err=64`）或挂住。

## 5. 常用环境变量

| 变量 | 作用 |
|---|---|
| `MEGA_CFG="k=v,g1.tile_k=512,s1.num_buffers=4"` | 覆盖 LaunchCfg 字段（`g1./g2.` routed GEMM1/2，`s1./s2.` shared A8W8 的 tile），在 tuned 行之后生效 |
| `MM_EXTRA="--sh8"` | 传给 `run_mm.py` 的额外参数（`tp1g.sh` / `sweepm.sh` / `stampsm.sh`） |
| `MEGA_TS=1` | 编入阶段时间戳（`stampsm.sh` 已设置） |
| `MEGA_LL_TOKENS`（默认 512） | LL 通信协议的 token 上限（ag_rs <= 该值；AR 的 push <= 该值 / tp） |
| `MEGA_TUNED_KEY=<tokens>` / `MEGA_TUNED_BUCKET=0` | 强制用某一行 tuned 配置（-1：不用）/ 关闭"向上取最近的 tuned 行" |
| `MEGA_ARENA_ALIGN`（默认 2MiB） | arena 通信区对齐 |
| `GWAIT=<分钟>` | 脚本等 GPU 空闲的最长时间 |
| `CRUN_ABORT=0` | 发现别的进程时不终止自己的 run（默认立即终止） |

tuned 配置在 `aiter/ops/flydsl/kernels/mega_moe_tp_1250/tuned_mega_moe_tp.csv`
（列 `model_dim,experts,inter,tp,tokens,mode,cfg,us,note`；`mode` 为空 / `ar` / `ag_rs`，A8W8 shared 另读 `sh8` / `ar_sh8` / `ag_rs_sh8`）。

## 6. 安全规则（都出过事故）

1. **一次只跑一个 GPU 任务**，GPU 空闲才启动（脚本里的 `gwait.sh`）。4 卡的常驻 kernel 不能和别的任务同时跑：
   所有等待会超时，严重时驱动 evict queue 失败、GPU 卡在 100% 需要 reset。`crun.sh` 发现第二个进程会立即终止自己的 run。
2. **新 kernel / 新参数先单卡验证**（4.1），用和 4 卡完全相同的参数；`err=0` 不代表结果对，必须看 `logits_diff`。
3. **不要用 rocprofv3 的 PMC / 多卡 profiling**（会导致 SMU 无响应）；看阶段耗时用 4.3 的时间戳。
4. 每步之后看 `bash /tmp/mmtp/dmf.sh`（dmesg 里 fault / evict 失败的行数）有没有增加；有新 fault 就停下来。
5. `sweepm.sh` / `stampsm.sh` 等脚本结尾会 `pkill -f "[r]un_mm.py"`：**调用它们的那条命令行里不要出现 `run_mm.py` 这个字符串**
   （否则会把调用它的 shell 一起杀掉）。停一条后台链时，kill 命令单独执行。
6. 正在跑 benchmark 时不要改 `mega_moe_tp_1250/*.py`（FlyDSL 编译时会重新读源码）。
7. 以 root 运行后把新文件的属主还回去：`find /home/binding2/moe_1026 -user root -exec chown -h 11174:11174 {} +`。

## 7. 脚本清单（`bench/`）

| 脚本 | 用途 |
|---|---|
| `bench_m31_sh8.sh`、`tab_sh8.py` | M3.1 A8W8：baseline B1/B2 + mega，出表 |
| `bench_sh8_base.sh` | 只跑 M3.1 A8W8 的 baseline（sep / fuse） |
| `bench_1120.sh`、`tab_1120.py`、`full_r200.sh` | 全 MXFP4 的 baseline + mega，出表 |
| `tp1g.sh`、`rowcfg.py` | 单卡验证；打印某 token 数用到的 tuned 行 |
| `sweepm.sh` | 4 卡 mega 计时 / 参数扫描 |
| `stampsm.sh` | 4 卡阶段时间戳 |
| `compile_all.sh`、`compile_wpf.sh` | 离线编译 |
| `gwait.sh`、`quiet_run.sh`、`crun.sh`、`dmf.sh`、`gwatch.sh` | 等 GPU 空闲 / 等安静窗口 / 运行时检测别的进程 / dmesg fault 计数 / 记录 GPU 上的进程 |
| `rebuild_jit.py`、`gen_fixed_csv.py` | 合并 upstream 后重建 JIT 模块、生成 baseline 的 tuned 表 |
| `run_mm.py`、`baseline_moe_ar.py`、`baseline_m31_sh8.py`、`compile_mm.py`、`pad_inter.py`、`gpu_free.sh` | Python 入口脚本和 GPU 空闲检查（放到 `/home/binding2/moe_1026/scripts/`） |
| `killchain.sh` | 停一条后台实验链（单独一条命令执行） |
| `r233_chain.sh`、`r234_chain.sh` | 实验链的例子：shared tile 调优、全量单卡验证 + 4 卡全表 |
