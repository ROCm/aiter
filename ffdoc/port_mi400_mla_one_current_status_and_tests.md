# MI400 MLA 当前移植状态与测试命令

日期：2026-05-13

## 范围

本文记录 `poc_kl/mi400/mla` 移植到 `aiter` 的当前状态，目标架构为 `gfx1250`。

路径说明：

- 用户指定目标路径：`/home/feifei/Desktop/repo/aiter`
- 实际解析到的仓库路径：`/home/feifei/repo/aiter`

## 当前状态

资产与元数据：

- 当前只接入 1 个 gfx1250 汇编 kernel：`mla_a8w8_qh16_1tg_16mx4_64nx1_np.co`。
- `.co` 位于 `aiter/hsa/gfx1250/mla/`。
- `aiter/hsa/gfx1250/mla/mla_asm.csv` 当前只有 1 条 gfx1250 行。
- CSV 仍使用原 MLA 字段：`qType,kvType,Gqa,ps,qSeqLen,prefill,causal,lse,knl_name,co_name`。

运行时支持：

- `gfx1250` decode 路径已和 gfx9 路径分开：`mla_decode_stage1_asm_fwd` 中检测到 `gfx1250` 后会提前 `return mla_decode_mi400_dispatch(...)`。
- `gfx1250` 当前通过原 `get_heuristic_kernel_mla(...)` 命中单个 fp8/fp8、GQA16、qSeqLen4 kernel。
- `gfx942/gfx950` 仍继续走原有 `KernelArgs` 和原 heuristic 路径。
- 目前没有 `modelVersion` / `variantId` 选择逻辑，也没有接入 v4/sparse kernel。

当前限制：

- 仅支持非 persistent decode。
- 当前 gfx1250 runtime 要求 fp8/fp8、`nhead_kv == 1`、`gqa_ratio == 16`、`max_seqlen_q == 4`、`page_size == 64`、`num_kv_splits == 1`、Q head dim 576、output head dim 512。
- `q_scale` / `kv_scale` 必须为 fp32，且至少包含 batch 个元素。
- `splitData` / `splitLse` 必须为 fp32；gfx1250 不能复用 bf16 `o.view(...)` 作为 `splitData`。
- `lse` 必须为 `nullptr`，当前 smoke 不支持 LSE 输出。
- 当前 Python 环境缺少 `torch`，所以没有在本机完成 JIT runtime / pytest 测试。

## 相关文件

主要源码与元数据：

- `aiter/hsa/gfx1250/mla/mla_asm.csv`
- `aiter/hsa/gfx1250/mla/mla_a8w8_qh16_1tg_16mx4_64nx1_np.co`
- `aiter/csrc/py_itfs_cu/asm_mla.cu`
- `aiter/aiter/mla.py`
- `aiter/op_tests/test_mla_mi400.py`
- `aiter/op_tests/_mla_mi400_repro.py`

当前 gfx1250 `.co` 清单：

- `mla_a8w8_qh16_1tg_16mx4_64nx1_np.co`

## 已执行命令

校验 CSV 中每行都能找到 `.co` 和符号：

```bash
cd /home/feifei/repo/aiter
python3 - <<'PY'
import csv, pathlib, subprocess, sys
base = pathlib.Path('/home/feifei/repo/aiter/hsa/gfx1250/mla')
ok = True
rows = list(csv.DictReader((base / 'mla_asm.csv').open()))
for row in rows:
    co = base / row['co_name']
    if not co.exists():
        print(f'missing co: {row["co_name"]}')
        ok = False
        continue
    out = subprocess.check_output(['llvm-readelf', '-s', str(co)], text=True, errors='replace')
    if row['knl_name'] not in out:
        print(f'missing symbol: {row["knl_name"]} in {row["co_name"]}')
        ok = False
print(f'checked {len(rows)} gfx1250 MLA csv rows')
sys.exit(0 if ok else 1)
PY
```

当前预期结果：

```text
checked 1 gfx1250 MLA csv rows
```

检查当前 Python 环境：

```bash
cd /home/feifei/repo/aiter
python3 - <<'PY'
try:
    import torch
    print('torch ok', torch.__version__)
except Exception as e:
    print(type(e).__name__ + ':', e)
PY
```

结果：

```text
ModuleNotFoundError: No module named 'torch'
```

## 需要在 gfx1250 机器上执行的命令

构建或测试时建议显式设置：

- `ENABLE_CK=0`：绕开 gfx1250 下 CK 编译断言。
- `ROCM_HOME=/opt/rocm`：确保 JIT 链接时能找到 `libamdhip64.so`。
- `GPU_ARCHS=gfx1250`、`AITER_GPU_ARCHS=gfx1250`：限制编译目标和 runtime dispatch。
- `AITER_ASM_DIR=/home/feifei/repo/aiter/hsa`：指向当前仓库中的 asm 资产。

构建 `module_mla_asm`：

```bash
cd /home/feifei/repo/aiter
ROCM_HOME=/opt/rocm ENABLE_CK=0 GPU_ARCHS=gfx1250 AITER_GPU_ARCHS=gfx1250 \
  python3 -c "from aiter.jit.core import build_module; build_module('module_mla_asm')"
```

运行当前 smoke（只验证 loader/launcher 和 shape）：

```bash
cd /home/feifei/repo/aiter
ROCM_HOME=/opt/rocm ENABLE_CK=0 \
GPU_ARCHS=gfx1250 AITER_GPU_ARCHS=gfx1250 AITER_ASM_DIR=/home/feifei/repo/aiter/hsa \
  python3 -m pytest op_tests/test_mla_mi400.py -k smoke -q
```

运行正式数值测试：

```bash
cd /home/feifei/repo/aiter
ROCM_HOME=/opt/rocm ENABLE_CK=0 \
GPU_ARCHS=gfx1250 AITER_GPU_ARCHS=gfx1250 AITER_ASM_DIR=/home/feifei/repo/aiter/hsa \
  python3 -m pytest op_tests/test_mla_mi400.py -k numerics -q
```

该测试固定输入、运行 `mla_decode_fwd` 后显式 `torch.cuda.synchronize()`，检查 `out` / `attn_logits` / `attn_lse` 没有 nan/inf，并用同一份 fp8 Q/KV 计算 PyTorch causal MLA reference，对 `out` 做余弦误差断言。成功时预期结果：

```text
1 passed
```

注意：`-k smoke` 仍只检查 load/launch 和 shape，不能证明数值正确；需要数值验证时使用 `-k numerics`。

运行强同步 + nan/inf 检查 repro：

```bash
cd /home/feifei/repo/aiter
rm -rf aiter/jit/build/module_mla_asm aiter/jit/module_mla_asm.so
ROCM_HOME=/opt/rocm ENABLE_CK=0 \
GPU_ARCHS=gfx1250 AITER_GPU_ARCHS=gfx1250 AITER_ASM_DIR=/home/feifei/repo/aiter/hsa \
  python3 -u op_tests/_mla_mi400_repro.py
```

成功时关键输出应包含：

```text
launch returned, syncing...
sync OK.
out shape: (4, 16, 512) dtype: torch.bfloat16
nan count: 0 inf count: 0
DONE
```
