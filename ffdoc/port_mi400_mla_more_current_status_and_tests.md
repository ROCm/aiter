# MI400 MLA 当前移植状态与测试命令

日期：2026-05-13

## 范围

本文记录 `poc_kl/mi400/mla` 移植到 `aiter` 的当前状态，目标架构为 `gfx1250`。

路径说明：

- 用户指定目标路径：`/home/feifei/Desktop/repo/aiter`
- 实际解析到的仓库路径：`/home/feifei/repo/aiter`

## 当前状态

资产与元数据：

- 已将 `poc_kl/mi400/mla/shaders/*.s` 中的 12 个汇编 kernel 编译为 gfx1250 `.co`。
- 已将 12 个 `.co` 放入 `aiter/hsa/gfx1250/mla/`。
- 已更新 `aiter/hsa/gfx1250/mla/mla_asm.csv`，当前包含 12 条 gfx1250 行。
- CSV 新增 `passSize`、`modelVersion`、`variantId` 字段，用于区分同形状的不同 kernel，避免 heuristic 随机命中。

运行时支持：

- `modelVersion=3`、`variantId=1..8`：已接入当前 `mla_decode_mi400_dispatch` 路径。
- `modelVersion=4`、`variantId=9..12`：`.co` 与 CSV 已接入，但运行时会主动阻断，因为 v4 需要独立的 QROPE/KVROPE ABI。
- `gfx1250` decode 路径已和 gfx9 路径分开：`mla_decode_stage1_asm_fwd` 中检测到 `gfx1250` 后会提前 `return mla_decode_mi400_dispatch(...)`。
- `gfx942/gfx950` 仍继续走原有 `KernelArgs` 和原 heuristic 路径。

当前限制：

- 仅支持非 persistent decode。
- 当前 gfx1250 runtime 要求 fp8/fp8、`nhead_kv == 1`、`page_size == 64`、`num_kv_splits == 1`、Q head dim 576、output head dim 512。
- 当前 v3 runtime 使用 `q_scale` / `kv_scale` 作为 v3 slot 16/17 的 `descale_q/descale_k` 指针。
- v4 不能复用这两个字段，因为 v4 slot 16/17 是 QROPE/KVROPE 指针。
- 当前 Python 环境缺少 `torch`，所以没有在本机完成 JIT runtime / pytest 测试。

## 相关文件

主要源码与元数据：

- `aiter/hsa/gfx1250/mla/mla_asm.csv`
- `aiter/csrc/py_itfs_cu/asm_mla.cu`
- `aiter/aiter/mla.py`

当前 gfx1250 `.co` 清单：

- `mla_a8w8_qh16_1tg_16mx4_64nx1_np.co`
- `mla_a8w8_qh16_1tg_16mx4_64nx1.co`
- `mla_a8w8_qh16_1tg_16mx4_64nx1_np_3p.co`
- `mla_a8w8_qh16_1tg_16mx4_64nx1_np_1tdm.co`
- `mla_a8w8_qh16_1tg_16mx4_128nx1_np_3p.co`
- `mla_a8w8_qh16_1tg_16mx1_32nx4_np_3p.co`
- `mla_a8w8_qh16_1tg_16mx2_32nx4_np_3p.co`
- `mla_a8w8_qh32_1tg_32mx4_64nx1.co`
- `mla_a8w8_qh16_1tg_16mx4_64nx1_np_nm.co`
- `mla_a8w8_qh16_1tg_16mx4_64nx1_sparse.co`
- `mla_a8w8_qh16_1tg_16mx4_64nx1_sparse_msb_cycling.co`
- `mla_a8w8_qh16_1tg_16mx4_64nx1_sparse_msb_cycling_pure_issue.co`

## 已执行命令

在 `poc_kl` 中编译 12 个 `.s`：

```bash
cd /home/feifei/repo/poc_kl/mi400/mla
for s in /home/feifei/repo/poc_kl/mi400/mla/shaders/*.s; do
  base=$(basename "$s" .s)
  echo "=== asm: ${base}.co ==="
  amdclang++ -ggdb -g -x assembler -target amdgcn--amdhsa --offload-arch=gfx1250 "$s" -o "/home/feifei/repo/poc_kl/mi400/mla/${base}.co"
done
```

复制 `.co` 到 aiter：

```bash
cp /home/feifei/repo/poc_kl/mi400/mla/*.co /home/feifei/repo/aiter/hsa/gfx1250/mla/
```

安装临时代码生成依赖，不修改仓库依赖文件：

```bash
python3 -m pip install --target /tmp/aiter_codegen_deps numpy pandas
```

运行 codegen dry-run：

```bash
cd /home/feifei/repo/aiter
mkdir -p /tmp/aiter_mla_codegen_all
PYTHONPATH=/tmp/aiter_codegen_deps AITER_GPU_ARCHS='gfx1250;gfx950;gfx942' \
  python3 hsa/codegen.py -m mla --output_dir /tmp/aiter_mla_codegen_all
```

结果：

- 命令成功。
- 生成 `/tmp/aiter_mla_codegen_all/asm_mla_configs.hpp`。
- 生成头文件包含 12 条 `gfx1250` 行。
- 生成头文件包含 `passSize`、`modelVersion`、`variantId` 字段。

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

结果：

```text
checked 12 gfx1250 MLA csv rows
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

构建 `module_mla_asm`：

```bash
cd /home/feifei/repo/aiter
GPU_ARCHS=gfx1250 AITER_GPU_ARCHS=gfx1250 \
  python3 -c "from aiter.jit.core import build_module; build_module('module_mla_asm')"
```

运行当前 smoke：

```bash
cd /home/feifei/repo/aiter
GPU_ARCHS=gfx1250 AITER_GPU_ARCHS=gfx1250 AITER_ASM_DIR=/home/feifei/repo/aiter/hsa \
  python3 -m pytest op_tests/test_mla_mi400.py -k smoke -q
```

指定 v3 变体：

```bash
cd /home/feifei/repo/aiter
AITER_MLA_MI400_VARIANT_ID=1 \
GPU_ARCHS=gfx1250 AITER_GPU_ARCHS=gfx1250 AITER_ASM_DIR=/home/feifei/repo/aiter/hsa \
  python3 -m pytest op_tests/test_mla_mi400.py -k smoke -q
```

当前可选择的 v3 变体：

- `1..5`：qSeqLen 4，GQA 16。
- `6`：qSeqLen 1，GQA 16。
- `7`：qSeqLen 2，GQA 16。
- `8`：qSeqLen 4，GQA 32。

`9..12` 是 v4/sparse 行，当前会主动报错，直到 v4 QROPE/KVROPE ABI 接入完成。
