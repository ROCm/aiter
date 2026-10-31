# 在新 container 中运行 FlyDSL vs gluon（HERD 模式）e2e 对比

在 DeepSeek-V4-Flash、单卡 TP1 下，对比 opt aiter 里两种 MoE 后端在 HERD 模式下的端到端性能：
- **FlyDSL**：FlyDSL grouped GEMM；
- **gluon**：Triton/gluon decode MoE。

在 `/data/binding/moe_1031/bench/` 下执行 `bash run_flydsl_vs_gluon.sh` 即可一次跑完，约 25 分钟。2026-10-31 的结果见 `/data/binding/moe_1031/flydsl_vs_gluon_v2.txt`。

## 1. 对比哪几组

三组配置用的是同一份 opt aiter 和同一份打过补丁的 ATOM，只有 MoE 相关的环境变量不同：

| 配置 | `ATOM_USE_TRITON_MOE` | `AITER_FLYDSL_USE_HERD` | `AITER_TRITON_USE_HERD` | 启动脚本 |
|---|---|---|---|---|
| FlyDSL+HERD | 0 | 1 | 1（原脚本自带，FlyDSL 路径不读） | `serve_dsv4_tp1_flydsl_herd.sh` |
| FlyDSL 不开 HERD（基线，可选） | 0 | 不设 | 1（同上） | `serve_dsv4_tp1.sh` |
| Triton(gluon)+HERD | 1 | 不设 | 1 | `serve_dsv4_tp1_triton_herd.sh` |

其余环境变量和 `/app/scripts/dsv4/serve_dsv4.sh` 完全一致，只把 `-tp 4` 改成 `-tp 1`。

负载是 `/app/scripts/dsv4/bench_dsv4.sh` 的默认参数：ISL 1024、OSL 1024、CONC 64、256 个请求。每组都按同样的流程跑：
1. 起 server；
2. 冷启动跑 1 轮（run1），这一轮包含首次 JIT 编译，只作参考；
3. 预热；
4. 正式跑 3 轮（run2～4），取均值；
5. 停 server。

### FlyDSL HERD 为什么需要打补丁

- **aiter #6527**：给 gfx1250 加了 `fused_moe(herd_topk=K)`，让 grouped MoE 的 route kernel 丢掉每个 token 最冷门的那个候选 expert。opt aiter 已经包含这个 PR，不需要额外处理。
- **ATOM #2713**：让 ATOM 在 decode 时按 Top-(K+1) 路由，并给 `fused_moe` 传 `herd_topk`。这个 PR 还没合入，镜像里的 `/app/ATOM` 也没有，所以要在 `/app/ATOM` 的副本上打补丁。不打的话，`AITER_FLYDSL_USE_HERD=1` 在 gfx1250 上不起任何作用。

## 2. 新 container 的前提条件

| 项目 | 要求 | 检查方法 |
|---|---|---|
| 镜像 | 和 10-31 用的镜像相同：`/app/ATOM` 为 `r1_1012_rebased`（`f19778df4`），flydsl 0.3.3.dev886，`/app/triton-mi450`，torch 2.11 + ROCm 10.1 | `git -C /app/ATOM log --oneline -1`；`pip list \| grep -E "^(amd-aiter\|flydsl\|triton\|torch) "` |
| GPU | 至少 1 张 gfx1250，且没有别的任务在用（脚本要独占约 25 分钟） | `rocm-smi`，VRAM% 应为 0；`rocm-smi --showpids` |
| 端口 | 8000 空闲 | `curl -sf localhost:8000/health && echo busy` |
| `/data` | 已挂载，并包含模型 `/data/models/DeepSeek-V4-Flash`、opt aiter `/data/binding/moe_1031/opt/aiter` 和本目录 `bench/` | `ls /data/models/DeepSeek-V4-Flash/config.json /data/binding/moe_1031/opt/aiter/setup.py` |
| 网络 | 能访问 PyPI。`pip install -e .` 需要在隔离环境里下载构建依赖和 flydsl 0.4.1 | `pip download flydsl==0.4.1 --no-deps -d /tmp/x` |
| 用户 | root（要往系统 site-packages 里装包） | `id -u` 输出 0 |

`/data` 上已经准备好、可以直接复用的内容：
- **opt aiter**：CK submodule 已初始化，aiter 的 JIT 模块也已编译（`aiter/jit/*.so`）；
- **打过补丁的 ATOM 副本**：`/data/binding/moe_1031/atom_pr2713`，分支 `r1_1012_rebased_pr2713_port`。

## 3. 一键运行（推荐）

```bash
cd /data/binding/moe_1031/bench

# 1) 先空跑：只检查前置条件、打印执行计划，不碰 GPU，也不改 Python 环境
bash run_flydsl_vs_gluon.sh --dry-run

# 2) 正式运行，约 25 分钟
bash run_flydsl_vs_gluon.sh
```

新 container 里，`--dry-run` 的输出应该类似下面这样：

```
==== 0/7 prerequisites
ok: model /data/models/DeepSeek-V4-Flash, port 8000 free
==== 1/7 python env backup (/data/binding/moe_1031/bench/env_backup/<hostname>)
would back up the current env to /data/binding/moe_1031/bench/env_backup/<hostname>
==== 2/7 aiter#6527 in /data/binding/moe_1031/opt/aiter
ok: cfb6a59c1 update
==== 3/7 ATOM#2713 copy (/data/binding/moe_1031/atom_pr2713)
already patched: /data/binding/moe_1031/atom_pr2713 (f19778df4, branch r1_1012_rebased_pr2713_port)
==== plan (dry run, nothing executed)
  REQUIRE_LOG='FlyDSL HERD route' ... run_config.sh .../flydsl_herd serve_dsv4_tp1_flydsl_herd.sh
  FORBID_LOG='FlyDSL HERD route' ... run_config.sh .../flydsl_noherd serve_dsv4_tp1.sh
  REQUIRE_LOG='Triton MoE experts' ... run_config.sh .../triton_herd serve_dsv4_tp1_triton_herd.sh
```

如果第 3 步出现 `warning: /app/ATOM is at <commit>, the copy is at f19778df4`，说明新镜像里的 ATOM 和副本不是同一个版本，处理方法见 4.1。

正式运行依次执行以下步骤：
1. 备份当前 container 的 Python 环境，存到 `bench/env_backup/<hostname>/`；
2. 检查 aiter #6527 是否在 opt 里；
3. 准备打过补丁的 ATOM 副本（已存在就复用）；
4. 用 `pip install -e` 安装 opt aiter，保留系统 triton，flydsl 会升到 0.4.1；
5. 依次跑三组配置；
6. 检查 HERD 是否真正生效；
7. 生成报告。

脚本退出时会自动恢复 Python 环境，中途失败也会恢复。

常用参数：

| 参数 | 作用 |
|---|---|
| `--no-baseline` | 不跑 FlyDSL 不开 HERD 那组，省约 7 分钟 |
| `--unit-test` | 额外跑 aiter 的 `op_tests/test_flydsl_herd.py`，约 4 分钟；10-31 在 gfx1250 上的结果是 73 passed、1 skipped |
| `--keep-env` | 跑完不恢复环境，opt aiter 保持安装状态 |
| `--tag NAME` / `--out FILE` | 指定结果目录名和报告路径 |

可以用环境变量覆盖的路径：
- `AITER_DIR`：aiter 目录，默认 opt；
- `ATOM_SRC`：源 ATOM，默认 `/app/ATOM`；
- `ATOM_DIR`：打过补丁的 ATOM 副本；
- `MODEL`：模型路径；
- `BACKUP_DIR`：环境备份目录；
- `RESULTS_ROOT`：结果的根目录。

产出位置：
- **结果目录**：`bench/results/<时间戳>/`，下面每组一个子目录 `flydsl_herd/`、`flydsl_noherd/`、`triton_herd/`。每个子目录里有 `server.log`、`run_config.log`、`run1`～`run4/bench.log`、`prewarm.log` 和 `flydsl_cache/`。
- **报告**：`bench/results/<时间戳>/flydsl_vs_gluon.txt`。
- **总日志**：`bench/results/<时间戳>/run_flydsl_vs_gluon.log`。

> 说明：一键脚本在 10-31 只做了 `--dry-run` 和静态检查，还没有在 GPU 上完整跑过一次。它执行的流程，和产出 `flydsl_vs_gluon_v2.txt` 的那次手动运行相同。第一次正式使用时，请看一下第 5 节列的检查项。

## 4. 新 container 要注意的地方

### 4.1 ATOM 补丁副本：复用还是重新做

- **新镜像的 `/app/ATOM` 还是 `f19778df4`**：直接复用 `/data/binding/moe_1031/atom_pr2713`，不用做任何事。
- **新镜像的 ATOM 版本不同，而你想测新版本**：用一个新的副本路径，脚本会从新的 `/app/ATOM` 复制一份并打补丁：
  ```bash
  ATOM_DIR=/data/binding/moe_1031/atom_pr2713_$(git -C /app/ATOM rev-parse --short HEAD) bash run_flydsl_vs_gluon.sh
  ```
- **补丁打不上**（脚本提示 `the port does not apply`）：说明新 ATOM 的 `atom/model_ops/moe.py` 结构变了，需要参照 `patches/atom_pr2713.diff`（原 PR）手工移植。改动集中在 `Mxfp4MoEMethod` 的 5 处，`patches/atom_pr2713_port_r1_1012_rebased.diff` 是在 `r1_1012_rebased` 上的移植示例：
  1. 从 `aiter.ops.topk` import `_FLYDSL_USE_HERD`、`_FLYDSL_HERD_MIN_M`、`_FLYDSL_HERD_MAX_M`；
  2. 在 `__init__` 里设置 `self.use_flydsl_herd`，条件是：HERD 开关已打开、gfx1250、gate/up 交错权重、非 EP、非 EPLB；
  3. 在算 `topk_ids` 的地方，如果 `_flydsl_herd_step()` 返回 True，就改为 `FusedMoE.select_experts(top_k=K+1, renormalize=False, ...)`；
  4. 给 `fused_moe` 的额外参数里加上 `herd_topk=K` 和 `herd_routed_scaling_factor=routed_scaling_factor`；
  5. 新增 `_flydsl_herd_step()`：只接受 token 数在 16～128 之间的 decode 步，拒绝 hash/custom 路由、grouped topk、fused shared expert、Triton 路径和 modular kernel。

  另外，一键脚本靠日志 `FlyDSL HERD route` 判断 HERD 是否被走到，所以移植时要把 `_log_flydsl_herd_route()` 这条只打印一次的日志也加上。要么就不用一键脚本，按第 6 节手动运行。
- **如果 ATOM 是基于 main 的新版本**：可以先试原 PR：`git apply --include=atom/model_ops/moe.py patches/atom_pr2713.diff`。原 PR 里同样没有那条日志，需要自己补上。

### 4.2 aiter 的 JIT 产物

`opt/aiter/aiter/jit/*.so` 是在 10-31 的镜像里编出来的（torch 2.11 + ROCm 10.1），镜像相同就可以直接用。

如果新镜像的 torch 或 ROCm 版本不同，起 server 时可能报 `undefined symbol` 之类的 import 错误。这时先清掉旧产物，让它重新编译（首次起 server 会多花约 5 分钟）：

```bash
rm -rf /data/binding/moe_1031/opt/aiter/aiter/jit/*.so /data/binding/moe_1031/opt/aiter/aiter/jit/build
```

### 4.3 环境备份按 container 分开存放

`run_flydsl_vs_gluon.sh` 默认把备份放在 `bench/env_backup/<hostname>/`，在 docker 里 hostname 就是 container ID。每个 container 恢复的都是自己的 site-packages。

不要把别的 container 的备份（例如 `e2e_results/backup`）传给 `BACKUP_DIR`：镜像不同时，恢复会把另一个镜像的文件拷进来。

### 4.4 第一次起 server 会更慢

新 container 的 `~/.triton/cache` 是空的，第一组配置起 server 时要现场编译所有 Triton kernel，会比 10-31 的约 1.5 分钟慢几分钟。这不影响测量，因为编译都发生在冷启动轮和预热阶段，正式的 run2～4 期间新编译的 kernel 数仍应为 0。

### 4.5 多卡 container

脚本固定用 TP1，跑在第 0 张可见的卡上。要换卡就先设置 `HIP_VISIBLE_DEVICES`：

```bash
HIP_VISIBLE_DEVICES=2 bash run_flydsl_vs_gluon.sh
```

注意：只有同型号的卡（gfx1250，256 CU）才能和 10-31 的数字直接比较。

### 4.6 不在共享 `/data` 上的机器

至少要把下面这些拷过去：
- `/data/models/DeepSeek-V4-Flash`
- `/data/binding/moe_1031/opt/aiter`，包含 `.git` 和 `3rdparty/composable_kernel`
- `/data/binding/moe_1031/bench`

路径不同的话，用环境变量 `MODEL`、`AITER_DIR`、`ATOM_DIR` 指定。

如果 opt 的 CK submodule 是空的，`env.sh install` 会先尝试从 `/app/aiter/3rdparty/composable_kernel` 本地 clone，失败再从 GitHub 拉。

## 5. 怎么判断结果有效

1. **总日志里每一步都要正常通过**（`run_flydsl_vs_gluon.log`），尤其是这几项：
   - FlyDSL+HERD 组出现 `check ok: 'FlyDSL HERD route' found 5 times`，对应 decode CUDA graph 的 bs=16/32/48/64/128；
   - 第 6 步出现 `ok: ... HERD route kernel built`；
   - Triton 组出现 `check ok: 'Triton MoE experts' found ...`。

   任何一项不满足，脚本都会直接报错退出。
2. **每组的 `run_config.log` 里要确认两点**：
   - 加载路径正确：`aiter=/data/binding/moe_1031/opt/aiter`，`atom=<ATOM_DIR>`，且 `moe_py_modified=yes`；
   - 正式测试期间没有新编译：`new FlyDSL kernels during clean runs: 0`，`new Triton cache entries ...: 0`。
3. **报告里的“路径与 HERD 校验”一节要满足：**
   - 只有 FlyDSL+HERD 这组有 HERD route 的 token 数和 `launch_route_psum_herd` kernel；
   - Triton+HERD 这组的 decode MoE 显示为 Triton/gluon。
4. **和 10-31 的参考数据比较**（预热后 3 轮均值）：

   | 配置 | 输出吞吐 (tok/s) | Mean TPOT (ms) | Mean TTFT (ms) |
   |---|---:|---:|---:|
   | FlyDSL 不开 HERD | 6,296.91 | 9.53 | 651.98 |
   | FlyDSL+HERD | 6,689.32（+6.23%） | 8.94（-6.22%） | 646.66 |
   | Triton(gluon)+HERD | 6,908.22（+9.71%） | 8.64（-9.41%） | 647.13 |

   括号里是相对 FlyDSL 不开 HERD 的变化。FlyDSL+HERD 比 Triton+HERD 吞吐低 3.17%，TPOT 高 3.51%。

   **正常波动：** 同一个 server 里，轮与轮之间吞吐的 std 约 15～40 tok/s；不同次启动的 server 之间差 0.5%～1%。复现结果落在这个范围内就算一致。

## 6. 不用一键脚本时：手动分步

```bash
cd /data/binding/moe_1031/bench
B=$PWD/env_backup/$(hostname)
bash env.sh backup $B                                   # 只需第一次
bash patch_atom_herd.sh /app/ATOM /data/binding/moe_1031/atom_pr2713
bash env.sh install /data/binding/moe_1031/opt/aiter    # AITER_USE_SYSTEM_TRITON=1 已经内置

export AITER_DIR=/data/binding/moe_1031/opt/aiter ATOM_DIR=/data/binding/moe_1031/atom_pr2713
R=$PWD/results/manual_$(date +%m%d_%H%M)
REQUIRE_LOG='FlyDSL HERD route'  bash run_config.sh $R/flydsl_herd   serve_dsv4_tp1_flydsl_herd.sh
FORBID_LOG='FlyDSL HERD route'   bash run_config.sh $R/flydsl_noherd serve_dsv4_tp1.sh
REQUIRE_LOG='Triton MoE experts' bash run_config.sh $R/triton_herd   serve_dsv4_tp1_triton_herd.sh

python3 report_flydsl_vs_gluon.py --results $R          # 生成 $R/flydsl_vs_gluon.txt
bash env.sh restore $B
```

`run_config.sh` 会自己把 `PYTHONPATH` 设成 `ATOM_DIR:AITER_DIR`。如果是自己手动起 server，就必须手动 `export PYTHONPATH=$ATOM_DIR:$AITER_DIR`，否则容器默认的 `PYTHONPATH=/app/aiter` 会让你测到 `/app/aiter`。

## 7. 恢复环境

脚本正常结束或中途失败都会自动执行 `env.sh restore`，恢复后会对比 `pip freeze`，输出 `restored: pip freeze identical`。

如果是被 Ctrl-C 打断，或者 server 还没停，需要手动恢复：

```bash
pkill -INT -f "^python -m atom.entrypoints.openai_server"      # 等 rocm-smi 的 VRAM% 回到 0
bash /data/binding/moe_1031/bench/env.sh restore /data/binding/moe_1031/bench/env_backup/$(hostname)
bash /data/binding/moe_1031/bench/env.sh status                 # 应显示 /app/aiter 和 flydsl 0.3.3.dev886
```

## 8. 常见问题

| 现象 | 原因 / 处理 |
|---|---|
| `required server log 'FlyDSL HERD route' not found` | HERD 没有被走到，可能有这几种情况：<br>• 用的不是打过补丁的 ATOM，看 `run_config.log` 里的 `atom=` 和 `moe_py_modified`；<br>• 启动脚本里没有 `AITER_FLYDSL_USE_HERD=1`；<br>• GPU 不是 gfx1250；<br>• 开了 EP 或 EPLB。 |
| `launch_route_psum_herd was not built` | ATOM 传了 `herd_topk`，但 aiter 没走 HERD 的 route kernel。检查 aiter 是否包含 #6527（第 2 步会检查），以及 `AITER_DIR` 是否正确 |
| `import check failed: aiter resolves to '/app/aiter/...'` | `PYTHONPATH` 指向了别处。`env.sh` 和 `run_config.sh` 会自动设置，手动跑时需要自己 `export` |
| pip 报 `externally-managed-environment` | 需要加 `--break-system-packages`，`env.sh` 已经加了 |
| 装完 aiter 后 triton 变成了 PyPI 版本 | 安装时漏了 `AITER_USE_SYSTEM_TRITON=1`（`env.sh` 已内置）。按第 7 节恢复环境；triton 不在备份范围内，需要从镜像重新安装 `/app/triton-mi450` |
| 起 server 时 import 报错或 `undefined symbol` | 镜像不同导致 JIT `.so` 不兼容，按 4.2 清掉后重新编译 |
| `port 8000 is serving` / `an ATOM server is running` | 有别的 server 在跑，先停掉；如果显存还没释放，等 `rocm-smi` 的 VRAM% 回到 0 |
| `Benchmark duration` 和 TTFT 波动很大 | 正式测试期间有首次 JIT 编译。看 `run_config.log` 里新编译的 kernel 数是否为 0；不为 0 就重新跑 |
| 报告最后提示 `save_csv.py` 找不到 | `bench_dsv4.sh` 最后一步调用的脚本本来就不存在，不影响结果 |

## 9. 文件清单（`/data/binding/moe_1031/bench/`）

| 文件 | 用途 |
|---|---|
| `run_flydsl_vs_gluon.sh` | 一键脚本（HERD 模式），用 `--help` 查看用法 |
| `run_config.sh` | 跑一个配置：起 server、冷启动 1 轮、预热、正式 3 轮、停 server；支持 `REQUIRE_LOG` / `FORBID_LOG` 检查 |
| `serve_dsv4_tp1*.sh` | 4 个启动脚本：FlyDSL、FlyDSL+HERD、Triton+HERD、Triton 不开 HERD |
| `make_serve_scripts.sh` | 从 `/app/scripts/dsv4/serve_dsv4.sh` 重新生成上面 4 个启动脚本 |
| `bench_dsv4.sh` | 原版 `/app/scripts/dsv4/bench_dsv4.sh` 的拷贝，逐字节相同 |
| `prewarm.sh` | 预热：依次发送 k 个并发的 1024-token 请求，k = 1..16 |
| `patch_atom_herd.sh` | 复制 `/app/ATOM` 并打 ATOM #2713 移植补丁；`CHECK_ONLY=1` 时只检查 |
| `patches/atom_pr2713.diff` | ATOM #2713 原始 diff（基于 ATOM main） |
| `patches/atom_pr2713_port_r1_1012_rebased.diff` | 移植到 `r1_1012_rebased` 的 diff，附带 `FlyDSL HERD route` 日志 |
| `env.sh` | 环境管理：`backup`、`install`、`restore`、`status` |
| `report_flydsl_vs_gluon.py` | 从结果目录生成报告，只读文本日志 |
| `herd_check.py` | 在 GPU 上验证路由层面的 HERD：Triton `routing()` 会让 batch 内用到的 expert 变少；FlyDSL 那一行检查的是 aiter 在 `topk_gating` 里的自动 HERD，在 gfx1250 上本来就是 False |
