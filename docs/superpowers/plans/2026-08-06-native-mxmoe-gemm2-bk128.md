# Native MXMoE GEMM2 BK128 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 为原生 FlyDSL MXMoE GEMM2 增加显式 `BK=128` 支持，使实际 `inter_dim=384` 可正确运行，并完整接通 runtime、AOT、tuner 与现有 epilogue。

**Architecture:** 保留原生 GEMM2 的 BN256、排序布局、流水线和 epilogue，只把 K tile 几何从固定“两段 K128”泛化为由 BK 编译期推导。BK 由 `kernelName2` 精确指定并贯穿 parser、runtime、AOT、cache 和 tuner；BK128/BK256 使用不同内部 GPU symbol。

**Tech Stack:** Python 3、PyTorch/pytest、FlyDSL、MLIR/ROCDL、gfx950 scaled MFMA、AITer FMoE tuner/AOT。

## Global Constraints

- 原生 GEMM2 只支持 `BN=256`。
- 新合法集合是 `BK in {128, 256}`；不增加 BN128。
- `kernelName2` 中的 BK 是精确选择，不允许 runtime 自动改写。
- 配置继续按 `w1/w2` 实际 shape 查找，不改变 CSV schema 或 shape key。
- native CSV 的 `inter_dim` 是实际、未 padding 的 K；本计划不实现 BK128
  padding/`inter_real` 验收，也不移除 runtime wrapper 既有的外部 BK256
  `D_INTER_REAL` padding 能力。
- 不改 GEMM1，不合并 native/v2 实现。
- 现有 11 个 native BM/NT/epilogue 组合必须全部保留。
- 内部 GPU symbol 对 BK128 和 BK256 都显式追加
  `_core<epoch>_bk<BK>`。
- BK256 的 BM>=32 保持内存访问与计算语义等价；BM16 有意修复 inactive
  wave 在长流水 refill 中写入相邻 LDS slot 的既有问题。不得声称
  normalized IR/ISA 仅有 symbol 差异。
- 实现依据：`docs/superpowers/specs/2026-08-06-native-mxmoe-gemm2-bk128-design.md`。

## File Structure

- Modify: `aiter/ops/flydsl/mxfp4_kname.py`
  - 负责 native/v2 GEMM2 名称中的 BM/BN/BK 与 epilogue 标志解析。
- Modify: `aiter/fused_moe.py`
  - 将解析出的 BN/BK 传入 native stage2 两条输出路径。
- Modify: `aiter/ops/flydsl/mxfp4_gemm2_kernels.py`
  - 负责 host-side tile/shape/variant fail-fast 与 launcher cache。
- Modify: `aiter/aot/flydsl/mxfp4_moe.py`
  - 负责 native AOT job、dedup key 和 compile 参数。
- Modify: `csrc/ck_gemm_moe_2stages_codegen/gemm_moe_tune.py`
  - 负责按实际 K 生成 BK128/BK256 native 候选。
- Modify: `aiter/ops/flydsl/kernels/mxfp4_gemm_common.py`
  - 修正 K256 粒度 B-scale chunk 的 ceil stride。
- Modify: `aiter/ops/flydsl/kernels/mxfp4_gemm2.py`
  - 实现 BK 派生的 LDS、B-load、scale、MFMA half 和 K-loop。
- Create: `op_tests/flydsl_tests/test_native_mxmoe_gemm2_bk128.py`
  - host-only parser、runtime、AOT、tuner 与 validation 测试。
- Create: `op_tests/flydsl_tests/test_native_mxmoe_gemm2_bk128_gpu.py`
  - gfx950 原生 BK128/BK256 correctness、epilogue 和 high-level smoke。

---

### Task 1: 建立显式 BK parser、runtime 传播与 host validation

**Files:**
- Add: `docs/superpowers/plans/2026-08-06-native-mxmoe-gemm2-bk128.md`
- Create: `op_tests/flydsl_tests/test_native_mxmoe_gemm2_bk128.py`
- Modify: `aiter/ops/flydsl/mxfp4_kname.py:7-9,108-180`
- Modify: `aiter/fused_moe.py:1623-1741,1844-1933`
- Modify: `aiter/ops/flydsl/mxfp4_gemm2_kernels.py:68-147`

**Interfaces:**
- Produces: `_parse_mxfp4_g2_kname(kname: str) -> dict`，包含 `BM`, `BN`, `BK`, `atomic`, `use_nt`, `mxfp4out`, `cshuffle`, `xcd_swizzle`。
- Produces: `parse_g2_kname_any(kname: str) -> dict`，native/v2 均包含 `BM`, `BN`, `BK`。
- Produces: `_mxfp4_a4w4_stage2` 新增 keyword-only 参数 `BN: int` 与
  `BK: int`，其余现有参数保持不变。
- Consumes later: Task 2 的 AOT/tuner 和 Task 3 的 device kernel 均依赖上述字段精确传播。

- [ ] **Step 1: 创建会失败的 parser、runtime 和 validation 测试**

创建 `op_tests/flydsl_tests/test_native_mxmoe_gemm2_bk128.py`：

```python
# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import pytest
import torch

from aiter.ops.flydsl.mxfp4_gemm2_kernels import _assert_supported
from aiter.ops.flydsl.mxfp4_kname import (
    _parse_mxfp4_g2_kname,
    parse_g2_kname_any,
)


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        (
            "flydsl_mxmoe_g2_a4w4_16x256x128_atomic_nt",
            {
                "BM": 16,
                "BN": 256,
                "BK": 128,
                "atomic": True,
                "use_nt": True,
                "mxfp4out": False,
                "cshuffle": False,
            },
        ),
        (
            "flydsl_mxmoe_g2_a4w4_128x256x256_f4out",
            {
                "BM": 128,
                "BN": 256,
                "BK": 256,
                "atomic": False,
                "use_nt": False,
                "mxfp4out": True,
                "cshuffle": False,
            },
        ),
        (
            "flydsl_mxmoe_g2_a4w4_64x256x128_cshuffle",
            {
                "BM": 64,
                "BN": 256,
                "BK": 128,
                "atomic": False,
                "use_nt": False,
                "mxfp4out": False,
                "cshuffle": True,
            },
        ),
    ],
)
def test_native_g2_parser_preserves_tiles_and_flags(name, expected):
    parsed = _parse_mxfp4_g2_kname(name)
    for key, value in expected.items():
        assert parsed[key] == value
    unified = parse_g2_kname_any(name)
    assert unified["v2"] is False
    for key, value in expected.items():
        assert unified[key] == value


def test_stage2_fw_forwards_native_tiles(monkeypatch):
    from aiter import fused_moe

    called = {}

    def capture(*args, **kwargs):
        called.update(kwargs)
        return args[9]

    monkeypatch.setattr(fused_moe, "_mxfp4_a4w4_stage2", capture)

    inter = torch.empty((32, 192), dtype=torch.uint8)
    w1 = torch.empty((2, 768, 128), dtype=torch.uint8)
    w2 = torch.empty((2, 256, 192), dtype=torch.uint8)
    ids = torch.zeros(32, dtype=torch.int32)
    weights = torch.ones(32, dtype=torch.float32)
    scale = torch.empty(1, dtype=torch.uint8)
    out = torch.empty((1, 256), dtype=torch.bfloat16)

    result = fused_moe._mxfp4_a4w4_stage2_fw(
        inter,
        w1,
        w2,
        ids,
        ids,
        ids,
        out,
        2,
        w2_scale=scale,
        a2_scale=scale,
        block_m=32,
        sorted_weights=weights,
        kernelName2="flydsl_mxmoe_g2_a4w4_32x256x128_atomic_nt",
        reverse_sorted=ids,
    )

    assert result is out
    assert (called["BM"], called["BN"], called["BK"]) == (32, 256, 128)
    assert called["atomic"] is True
    assert called["use_nt"] is True
    assert called["D_INTER"] == 384


def _validation_kwargs(**overrides):
    kwargs = {
        "NE": 2,
        "D_HIDDEN": 256,
        "D_INTER": 384,
        "topk": 2,
        "BM": 32,
        "use_nt": False,
        "atomic": True,
        "mxfp4out": False,
        "cshuffle": False,
        "BN": 256,
        "BK": 128,
    }
    kwargs.update(overrides)
    return kwargs


def test_native_validation_accepts_k384_bk128():
    _assert_supported(**_validation_kwargs())


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"BK": 256}, "multiple of 256"),
        ({"BK": 64}, "BK must be one of"),
        ({"BN": 128}, "BN=256"),
    ],
)
def test_native_validation_rejects_wrong_tile_contract(overrides, message):
    with pytest.raises(NotImplementedError, match=message):
        _assert_supported(**_validation_kwargs(**overrides))
```

- [ ] **Step 2: 运行测试并确认按预期失败**

Run:

```bash
python -m pytest \
  op_tests/flydsl_tests/test_native_mxmoe_gemm2_bk128.py \
  -q
```

Expected: FAIL；parser 尚无 `BN/BK`，stage2 尚未转发它们，validation 尚未拒绝 BN128/BK64。

- [ ] **Step 3: 在 kernel-name parser 中保留 BN/BK**

更新 `aiter/ops/flydsl/mxfp4_kname.py` 顶部格式注释为 `<BM>x<BN>x<BK>`，并将 native G2 return dict 改为：

```python
    return {
        "BM": nums["BM"],
        "BN": nums["BN"],
        "BK": nums["BK"],
        "splitk": "kSplitK" in nums,
        "kSplitK": nums.get("kSplitK", 0),
        "atomic": atomic,
        "use_nt": "NT" in flags,
        "mxfp4out": mxfp4out,
        "cshuffle": cshuffle,
        "xcd_swizzle": nums.get("xcd_swizzle", 0),
    }
```

将 `parse_g2_kname_any()` 的两个 return dict 补齐：

```python
        return {
            "v2": True,
            "BM": v2["tile_m"],
            "BN": v2["tile_n"],
            "BK": v2["tile_k"],
            "atomic": v2["epilog"] == "atomic",
            "use_nt": v2["use_nt"],
            "mxfp4out": False,
            "cshuffle": False,
        }
```

```python
    return {
        "v2": False,
        "BM": p2["BM"],
        "BN": p2["BN"],
        "BK": p2["BK"],
        "atomic": p2["atomic"],
        "use_nt": p2["use_nt"],
        "mxfp4out": p2["mxfp4out"],
        "cshuffle": p2["cshuffle"],
    }
```

- [ ] **Step 4: 将 BN/BK 贯穿 native stage2 两条输出路径**

在 `aiter/fused_moe.py::_mxfp4_a4w4_stage2` 的 keyword-only 参数中加入：

```python
    BM,
    BN,
    BK,
    device,
```

在该函数内两次调用 `flydsl_mxfp4_gemm2()` 时都加入：

```python
        BN=BN,
        BK=BK,
```

在 `_mxfp4_a4w4_stage2_fw()` 调用 `_mxfp4_a4w4_stage2()` 时加入：

```python
        BM=cfg["BM"],
        BN=cfg["BN"],
        BK=cfg["BK"],
```

- [ ] **Step 5: 增加 native wrapper 的精确 tile validation**

在 `aiter/ops/flydsl/mxfp4_gemm2_kernels.py::_assert_supported` 开头加入：

```python
    if BN != 256:
        raise NotImplementedError(
            f"flydsl mxfp4 gemm2 native kernel requires BN=256, got BN={BN}"
        )
    if BK not in (128, 256):
        raise NotImplementedError(
            f"flydsl mxfp4 gemm2 native kernel BK must be one of (128, 256), got BK={BK}"
        )
```

保留随后的 `D_INTER % BK`、`D_HIDDEN % BN` 和 `_SUPPORTED` 检查。

- [ ] **Step 6: 运行 host 测试与 linter**

Run:

```bash
python -m pytest \
  op_tests/flydsl_tests/test_native_mxmoe_gemm2_bk128.py \
  -q
```

Expected: PASS。

Run:

```bash
python -m ruff check \
  aiter/ops/flydsl/mxfp4_kname.py \
  aiter/fused_moe.py \
  aiter/ops/flydsl/mxfp4_gemm2_kernels.py \
  op_tests/flydsl_tests/test_native_mxmoe_gemm2_bk128.py
```

Expected: PASS，无新增诊断。

- [ ] **Step 7: 提交 parser/runtime contract**

```bash
git add \
  docs/superpowers/plans/2026-08-06-native-mxmoe-gemm2-bk128.md \
  aiter/ops/flydsl/mxfp4_kname.py \
  aiter/fused_moe.py \
  aiter/ops/flydsl/mxfp4_gemm2_kernels.py \
  op_tests/flydsl_tests/test_native_mxmoe_gemm2_bk128.py
git commit -m "feat(fmoe): propagate native mxmoe gemm2 tiles"
```

---

### Task 2: 接通 native AOT 与 tuner 的 BK specialization

**Files:**
- Modify: `op_tests/flydsl_tests/test_native_mxmoe_gemm2_bk128.py`
- Modify: `aiter/aot/flydsl/mxfp4_moe.py:52-97,100-217,259-290`
- Modify: `csrc/ck_gemm_moe_2stages_codegen/gemm_moe_tune.py:5740-5808`

**Interfaces:**
- Consumes: Task 1 的 native parser `BN/BK` 字段。
- Produces: native AOT job 字段 `BN: int`, `BK: int`，且 `_job_key()` 区分两者。
- Produces: `Mxfp4FlydslTuner._g2_kname(bm, use_nt, epilog, bk, bn=256) -> str`。
- Produces: `_candidate_rows()` 对 K384 只生成 BK128、对 K512 生成 BK128/BK256。

- [ ] **Step 1: 追加 AOT 与 tuner 的失败测试**

先在 `op_tests/flydsl_tests/test_native_mxmoe_gemm2_bk128.py` 顶部 import
区加入 `import csv`，再追加：

```python
def _write_native_csv(path, *, inter_dim, kernel_name2):
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "topk",
                "model_dim",
                "expert",
                "inter_dim",
                "kernelName1",
                "kernelName2",
                "cu_num",
                "act_type",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "topk": 2,
                "model_dim": 256,
                "expert": 2,
                "inter_dim": inter_dim,
                "kernelName1": "flydsl_mxmoe_g1_a4w4_32x256x256",
                "kernelName2": kernel_name2,
                "cu_num": 256,
                "act_type": "ActivationType.Silu",
            }
        )


def test_native_aot_preserves_bk128_and_stage1_k384(tmp_path):
    from aiter.aot.flydsl.mxfp4_moe import _job_key, parse_csv

    csv_path = tmp_path / "native_bk128.csv"
    _write_native_csv(
        csv_path,
        inter_dim=384,
        kernel_name2="flydsl_mxmoe_g2_a4w4_32x256x128_atomic",
    )
    jobs = parse_csv(str(csv_path))
    stage1 = next(job for job in jobs if job["stage"] == 1)
    stage2 = next(job for job in jobs if job["stage"] == 2)

    assert stage1["D_INTER"] == 384
    assert (stage2["BN"], stage2["BK"]) == (256, 128)
    assert stage2["D_INTER"] == 384
    assert stage2["D_INTER_REAL"] is None
    assert _job_key(stage2) != _job_key({**stage2, "BK": 256})


def test_native_aot_compile_forwards_tiles(monkeypatch):
    from aiter.aot.flydsl import mxfp4_moe
    from aiter.ops.flydsl import mxfp4_gemm2_kernels

    called = {}
    monkeypatch.setattr(
        mxfp4_gemm2_kernels,
        "flydsl_mxfp4_gemm2",
        lambda **kwargs: called.update(kwargs),
    )
    mxfp4_moe._compile_stage2(
        {
            "stage": 2,
            "kernel_name": "flydsl_mxmoe_g2_a4w4_32x256x128_atomic",
            "BM": 32,
            "BN": 256,
            "BK": 128,
            "use_nt": False,
            "NE": 2,
            "N_OUT": 256,
            "epilog": "atomic",
            "D_INTER": 384,
            "D_INTER_REAL": None,
            "topk": 2,
            "xcd_swizzle": 0,
        }
    )
    assert (called["BN"], called["BK"]) == (256, 128)


_TUNER_KEYS = [
    "gfx",
    "cu_num",
    "token",
    "model_dim",
    "inter_dim",
    "expert",
    "topk",
    "act_type",
    "dtype",
    "q_dtype_a",
    "q_dtype_w",
    "q_type",
    "use_g1u1",
    "doweight_stage1",
]


def _tuner_row(inter_dim):
    from aiter import ActivationType, QuantType, dtypes

    return {
        "gfx": "gfx950",
        "cu_num": 256,
        "token": 4,
        "model_dim": 256,
        "inter_dim": inter_dim,
        "expert": 2,
        "topk": 2,
        "act_type": ActivationType.Silu,
        "dtype": dtypes.bf16,
        "q_dtype_a": dtypes.fp4x2,
        "q_dtype_w": dtypes.fp4x2,
        "q_type": QuantType.per_1x32,
        "use_g1u1": True,
        "doweight_stage1": False,
    }


@pytest.mark.parametrize(
    ("inter_dim", "expected_bks"),
    [(384, {128}), (512, {128, 256}), (320, set())],
)
def test_native_tuner_candidate_bks(monkeypatch, inter_dim, expected_bks):
    from csrc.ck_gemm_moe_2stages_codegen import gemm_moe_tune

    monkeypatch.setattr(
        gemm_moe_tune,
        "get_flydsl_stage2_v2_kernels",
        lambda *args, **kwargs: {},
    )
    tuner = gemm_moe_tune.Mxfp4FlydslTuner.__new__(
        gemm_moe_tune.Mxfp4FlydslTuner
    )
    tuner.keys = _TUNER_KEYS
    candidates = tuner._candidate_rows(_tuner_row(inter_dim))
    native_names = [
        candidate["kernelName2"]
        for candidate in candidates
        if candidate["kernelName2"].startswith("flydsl_mxmoe_g2_")
    ]
    bks = {_parse_mxfp4_g2_kname(name)["BK"] for name in native_names}
    assert bks == expected_bks
```

- [ ] **Step 2: 运行新增测试并确认失败**

Run:

```bash
python -m pytest \
  op_tests/flydsl_tests/test_native_mxmoe_gemm2_bk128.py \
  -q
```

Expected: FAIL；native AOT job 尚无 BN/BK，K384仍被向上对齐到512，tuner只生成 BK256。

- [ ] **Step 3: 让 native AOT 使用名称中的 BK 和实际 K**

在 `aiter/aot/flydsl/mxfp4_moe.py::_job_key()` 的 native stage2 tuple 中加入：

```python
        job["BM"],
        job["BN"],
        job["BK"],
        job["use_nt"],
```

在 `parse_csv()` 读取 `kn2` 后，分别解析 v2/native：

```python
            v2_g2 = parse_flydsl_v2_gemm2_kernel(kn2)
            native_g2 = (
                _parse_mxfp4_g2_kname(kn2)
                if isinstance(kn2, str)
                and kn2.startswith("flydsl_mxmoe_g2_a4w4_")
                else None
            )
            if v2_g2 is not None:
                stage2_bk = v2_g2["tile_k"]
                stage2_d_inter = (
                    (inter_dim + stage2_bk - 1) // stage2_bk
                ) * stage2_bk
                stage2_d_inter_real = (
                    inter_dim if inter_dim != stage2_d_inter else None
                )
            elif native_g2 is not None:
                stage2_bk = native_g2["BK"]
                if inter_dim % stage2_bk != 0:
                    raise ValueError(
                        f"native MXMOE GEMM2 requires inter_dim % BK == 0, "
                        f"got inter_dim={inter_dim}, BK={stage2_bk}, "
                        f"kernelName2={kn2!r}"
                    )
                stage2_d_inter = inter_dim
                stage2_d_inter_real = None
            else:
                stage2_d_inter = ((inter_dim + 255) // 256) * 256
                stage2_d_inter_real = (
                    inter_dim if inter_dim != stage2_d_inter else None
                )
```

将 native/v2 共用的 stage1 job 中间维改为：

```python
                        "D_INTER": stage2_d_inter,
```

在 native stage2 job 中加入并使用：

```python
                        "BN": p2["BN"],
                        "BK": p2["BK"],
                        "D_INTER": stage2_d_inter,
                        "D_INTER_REAL": stage2_d_inter_real,
```

在 `_compile_stage2()` 调用中加入：

```python
        BN=job["BN"],
        BK=job["BK"],
```

- [ ] **Step 4: 让 tuner 显式生成合法 BK 集合**

将 `_g2_kname` 改为：

```python
    @staticmethod
    def _g2_kname(bm, use_nt, epilog, bk, bn=256):
        name = f"flydsl_mxmoe_g2_a4w4_{bm}x{bn}x{bk}"
        if epilog == "atomic":
            name += "_atomic" + ("_nt" if use_nt else "")
        elif epilog == "nonatomic_mxfp4":
            name += "_f4out"
        elif epilog == "nonatomic_cshuffle":
            name += "_cshuffle"
        return name
```

在 `_candidate_rows()` 中用实际 K 枚举：

```python
        inter_dim = int(row["inter_dim"])
        native_bks = []
        if inter_dim % 128 == 0:
            native_bks.append(128)
        if inter_dim % 256 == 0:
            native_bks.append(256)
```

将 native candidate 内层改为：

```python
                if bm in g2_bms:
                    for bk in native_bks:
                        for _, n2, ep in sorted(v for v in G2 if v[0] == bm):
                            cands.append(
                                self._candidate_row(
                                    row,
                                    bm,
                                    kn1,
                                    self._g2_kname(bm, n2, ep, bk),
                                )
                            )
```

保留 path-B v2 candidate 枚举不变。

- [ ] **Step 5: 运行 AOT/tuner host 测试**

Run:

```bash
python -m pytest \
  op_tests/flydsl_tests/test_native_mxmoe_gemm2_bk128.py \
  -q
```

Expected: PASS。

Run:

```bash
python -m ruff check \
  aiter/aot/flydsl/mxfp4_moe.py \
  csrc/ck_gemm_moe_2stages_codegen/gemm_moe_tune.py \
  op_tests/flydsl_tests/test_native_mxmoe_gemm2_bk128.py
```

Expected: PASS。

- [ ] **Step 6: 提交 AOT/tuner 传播**

```bash
git add \
  aiter/aot/flydsl/mxfp4_moe.py \
  csrc/ck_gemm_moe_2stages_codegen/gemm_moe_tune.py \
  op_tests/flydsl_tests/test_native_mxmoe_gemm2_bk128.py
git commit -m "feat(fmoe): add BK-aware native mxmoe AOT tuning"
```

---

### Task 3: 泛化 native GEMM2 K core 并跑通首个 K384/BK128

**Files:**
- Create: `op_tests/flydsl_tests/test_native_mxmoe_gemm2_bk128_gpu.py`
- Modify: `aiter/ops/flydsl/kernels/mxfp4_gemm_common.py:232-265`
- Modify: `aiter/ops/flydsl/kernels/mxfp4_gemm2.py:43-134,170-186,324-584`

**Interfaces:**
- Consumes: Task 1/2 的显式 `BK`。
- Produces: `compile_gemm2_a4w4_port` 接受 keyword-only
  `BN=256, BK=128|256` specialization。
- Produces: `tiling(BM: int, KH_TILE: int) -> tuple[int, int, int]`，第三项为每 wave 的 A-load group 数。
- Preserves: epilogue interfaces 与 launcher 参数 ABI。

- [ ] **Step 1: 创建最小 high-level GPU 回归**

创建 `op_tests/flydsl_tests/test_native_mxmoe_gemm2_bk128_gpu.py`：

```python
# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import pytest
import torch

from aiter import ActivationType, dtypes
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.flydsl.utils import is_flydsl_available
from aiter.test_common import checkAllclose
from csrc.ck_gemm_moe_2stages_codegen.gemm_moe_tune import Mxfp4FlydslTuner


_SKIP_GFX950_FLYDSL = pytest.mark.skipif(
    get_gfx() != "gfx950" or not is_flydsl_available(),
    reason="gfx950 FlyDSL required",
)


def _check_close(ref, out, label):
    assert torch.isfinite(out).all(), f"{label}: output contains NaN or Inf"
    err = checkAllclose(ref, out, msg=label, atol=1.0, rtol=0.05)
    assert err == 0 or err <= 0.05, f"{label}: error ratio {err} exceeds 0.05"


@pytest.mark.parametrize(
    ("inter_dim", "bk"),
    [
        pytest.param(384, 128, id="k384-bk128"),
        pytest.param(512, 256, id="k512-bk256"),
    ],
)
@_SKIP_GFX950_FLYDSL
def test_native_atomic_bm32_high_level(inter_dim, bk):
    token, model_dim, expert, topk, bm = 33, 256, 2, 2, 32
    activation = ActivationType.Silu
    data = Mxfp4FlydslTuner._prepare_case(
        token,
        model_dim,
        inter_dim,
        expert,
        topk,
        dtypes.bf16,
        seed=123,
    )
    kn1 = Mxfp4FlydslTuner._g1_kname(bm, False, False)
    kn2 = Mxfp4FlydslTuner._g2_kname(bm, False, "atomic", bk)

    out = Mxfp4FlydslTuner._port_e2e(
        data,
        kn1,
        kn2,
        topk,
        expert,
        model_dim,
        dtypes.bf16,
        activation,
        4.0,
        25.0,
    )
    ref = Mxfp4FlydslTuner._torch_ref(
        data,
        topk,
        dtypes.bf16,
        activation,
        4.0,
        25.0,
    )
    _check_close(ref.float(), out.float(), f"native_bm32_k{inter_dim}_bk{bk}")
```

- [ ] **Step 2: 验证 BK256 baseline 通过、BK128 按预期失败**

Run:

```bash
python -m pytest \
  op_tests/flydsl_tests/test_native_mxmoe_gemm2_bk128_gpu.py \
  -k "k512-bk256" \
  -q
```

Expected: PASS。

Run:

```bash
python -m pytest \
  op_tests/flydsl_tests/test_native_mxmoe_gemm2_bk128_gpu.py \
  -k "k384-bk128" \
  -q
```

Expected: FAIL at `compile_gemm2_a4w4_port` 的 `BN==BK==256` assertion。

- [ ] **Step 3: 在改 core 前保存 BK256 IR/ISA baseline**

Run:

```bash
rm -rf /tmp/native_mxmoe_bk256_before
FLYDSL_DUMP_IR=1 \
FLYDSL_DUMP_DIR=/tmp/native_mxmoe_bk256_before \
FLYDSL_RUNTIME_ENABLE_CACHE=0 \
python -m pytest \
  op_tests/flydsl_tests/test_native_mxmoe_gemm2_bk128_gpu.py \
  -k "k512-bk256" \
  -q
```

Expected: PASS，并在 `/tmp/native_mxmoe_bk256_before` 下生成 native stage2 的 `20_llvm_ir.ll` 与 `21_final_isa.s`。

保存同一 BK256 case 的性能 baseline：

```bash
python - <<'PY'
from pathlib import Path

from aiter import ActivationType, dtypes
from aiter.test_common import run_perftest
from csrc.ck_gemm_moe_2stages_codegen.gemm_moe_tune import Mxfp4FlydslTuner

token, model_dim, inter_dim, expert, topk, bm = 33, 256, 512, 2, 2, 32
data = Mxfp4FlydslTuner._prepare_case(
    token,
    model_dim,
    inter_dim,
    expert,
    topk,
    dtypes.bf16,
    seed=123,
)
kn1 = Mxfp4FlydslTuner._g1_kname(bm, False, False)
kn2 = Mxfp4FlydslTuner._g2_kname(bm, False, "atomic", 256)


def call():
    return Mxfp4FlydslTuner._port_e2e(
        data,
        kn1,
        kn2,
        topk,
        expert,
        model_dim,
        dtypes.bf16,
        ActivationType.Silu,
        4.0,
        25.0,
    )


call()
_, us = run_perftest(call, num_warmup=20, num_iters=101)
Path("/tmp/native_mxmoe_bk256_before_us.txt").write_text(f"{float(us):.8f}\n")
print(f"BK256 before: {float(us):.4f} us")
PY
```

Expected: 写入 `/tmp/native_mxmoe_bk256_before_us.txt` 并打印稳定的中位数。

- [ ] **Step 4: 修正公共 B-scale K256 chunk stride**

将 `aiter/ops/flydsl/kernels/mxfp4_gemm_common.py::kbs_c_k1_for` 改为：

```python
def kbs_c_k1_for(k):
    return (k + 255) // 256
```

该改动对 K256/K512 不改变结果，对 K384 从1个 chunk修正为2个。

- [ ] **Step 5: 泛化 A global-to-LDS geometry 与 symbol**

将 `tiling` 替换为：

```python
def tiling(BM, KH_TILE):
    lanes_per_row = KH_TILE // 16
    rows_per_call = 64 // lanes_per_row
    n_load_waves = min(4, BM // rows_per_call)
    rows_per_wave = BM // n_load_waves
    return n_load_waves, rows_per_wave, rows_per_wave // rows_per_call
```

将 `_issue_a_load_lds` 替换为：

```python
def _issue_a_load_lds(
    aq_rsrc, saq_base_i32, slot, kt, car, lane, slot_bytes, lds_row, KH_TILE, k_half
):
    lanes_per_row = KH_TILE // 16
    lane_row = lane // fx.Int32(lanes_per_row)
    lane_col = (lane % fx.Int32(lanes_per_row)) * fx.Int32(16)
    mask = _lds_swizzle_mask(lds_row + lane_row, KH_TILE)
    voffset = (lane_col ^ mask) + car * fx.Int32(k_half)
    off_i32 = fx.Int32(slot * slot_bytes) + lds_row * fx.Int32(KH_TILE)
    lds_ptr = _lds_ptr3(saq_base_i32, off_i32)
    rocdl.raw_ptr_buffer_load_lds(
        aq_rsrc,
        lds_ptr,
        fx.Int32(16),
        voffset,
        fx.Int32(kt * KH_TILE),
        fx.Int32(0),
        fx.Int32(0),
    )
```

在 `compile_gemm2_a4w4_port()` 开头使用：

```python
    assert BN == 256, f"only BN==256 supported, got BN={BN}"
    assert BK in (128, 256), f"BK must be one of (128, 256), got BK={BK}"
```

将 compile-scope tiling 和 symbol 改为：

```python
    _n_load_waves, _rows_per_wave, _load_groups = tiling(BM, KH_TILE)
```

```python
    _tag = (
        f"ne{NE}_h{N_OUT}_i{_K}{_rtag}_bm{BM}"
        f"{'_nt' if use_nt else ''}_{_epi_tag}_bk{BK}"
    )
```

将 `_issue_all_a_loads()` 的 inner loops 改为：

```python
        def _issue_all_a_loads(m_row0):
            lanes_per_row = KH_TILE // 16
            rows_per_call = 64 // lanes_per_row
            for slot in range_constexpr(kStages):
                for group in range_constexpr(_load_groups):
                    lds_row = (
                        wave * fx.Int32(_rows_per_wave)
                        + fx.Int32(group * rows_per_call)
                    )
                    car = m_row0 + lds_row + (
                        lane // fx.Int32(lanes_per_row)
                    )
                    _issue_a_load_lds(
                        aq_rsrc,
                        saq_base_i32,
                        slot,
                        slot,
                        car,
                        lane,
                        _slot_bytes,
                        lds_row,
                        KH_TILE=KH_TILE,
                        k_half=_K_HALF,
                    )
```

- [ ] **Step 6: 泛化 body constants、A fragment 与 B fragment**

在 `_gemm2_body()` 中使用：

```python
    _kHalves = BK // 128
    _tilesPerScaleChunk = 256 // BK
    _kScaleSubBlocks = max(1, _kMChunks // 2)
    _n_load_waves, _rows_per_wave, _load_groups = tiling(BM, KH_TILE)
    _lanes_per_row = KH_TILE // 16
    _rows_per_call = 64 // _lanes_per_row
```

`a_scale_s_base` 的长度改为 `_kScaleSubBlocks`。替换三个 load helper：

```python
    def scale_chunk_tile(kt):
        return kt // _tilesPerScaleChunk

    def shift_scale_word(scale, kt):
        if const_expr(_tilesPerScaleChunk == 1):
            return scale
        shift = fx.Int32((kt % _tilesPerScaleChunk) * 16)
        return arith.shrui(scale, _raw(shift))

    def load_a_scale_tile(kt):
        chunk_kt = scale_chunk_tile(kt)
        out = [None] * _kScaleSubBlocks
        for sub in range_constexpr(_kScaleSubBlocks):
            out[sub] = buffer_ops.buffer_load(
                ascale_rsrc,
                (v_voff_scale + fx.Int32(chunk_kt * 256)) // fx.Int32(4),
                vec_width=1,
                dtype=T.i32,
                soffset_bytes=a_scale_s_base[sub],
            )
        return out

    def load_b_scale_tile(kt):
        chunk_kt = scale_chunk_tile(kt)
        imm = chunk_kt * (kBS_stride_k0_dw * 4)
        out = [None, None]
        for mw in range_constexpr(2):
            out[mw] = buffer_ops.buffer_load(
                bscale_rsrc,
                (v_voff_scale + fx.Int32(imm)) // fx.Int32(4),
                vec_width=1,
                dtype=T.i32,
                soffset_bytes=b_scale_s_base[mw],
            )
        return out

    def load_b_tile(kt):
        v_voff_b = (
            (lane_div_16 * fx.Int32(256))
            + (lane_mod_16 * fx.Int32(16))
            + fx.Int32(kt * _kHalves * 1024)
        )
        out = [[None] * _kHalves for _ in range(4)]
        for j in range_constexpr(4):
            for half in range_constexpr(_kHalves):
                if const_expr(kt * _kHalves + half >= _n_real_half):
                    continue
                frag = buffer_ops.buffer_load(
                    bq_rsrc,
                    (v_voff_b + fx.Int32(half * 1024)) // fx.Int32(4),
                    vec_width=4,
                    dtype=T.i32,
                    cache_modifier=b_aux,
                    soffset_bytes=b_load_s_base[j],
                )
                out[j][half] = Vec(frag)
        return out
```

替换 body 的 A load/read：

```python
    def issue_a_load_lds(slot, kt):
        for group in range_constexpr(_load_groups):
            lds_row = (
                wave * fx.Int32(_rows_per_wave)
                + fx.Int32(group * _rows_per_call)
            )
            car = m_row + lds_row + (
                lane // fx.Int32(_lanes_per_row)
            )
            _issue_a_load_lds(
                aq_rsrc,
                saq_base_i32,
                slot,
                kt,
                car,
                lane,
                _slot_bytes,
                lds_row,
                KH_TILE=KH_TILE,
                k_half=_K_HALF,
            )

    def issue_a_ds_read(slot):
        lane_row = lane_mod_16
        lane_col = lane_div_16 * fx.Int32(16)
        mask = _lds_swizzle_mask(lane_row, KH_TILE)
        base_ptr = _lds_ptr3(saq_base_i32, fx.Int32(0))
        a = [[None] * _kHalves for _ in range(_kMChunks)]
        for half in range_constexpr(_kHalves):
            lds_col = (lane_col + fx.Int32(half * 64)) ^ mask
            for i in range_constexpr(_kMChunks):
                lds_row = lane_row + fx.Int32(i * 16)
                byte_off = (
                    fx.Int32(slot * _slot_bytes)
                    + lds_row * fx.Int32(KH_TILE)
                    + lds_col
                )
                a[i][half] = llvm.load(
                    T.vec(4, T.i32), _gep3(base_ptr, byte_off)
                )
        return a
```

- [ ] **Step 7: 泛化 scaled MFMA half 与真实 kt**

将 `mfma_cluster` 替换为：

```python
    def mfma_cluster(b_tile, a, a_scale_sub, b_scale_slot, init, kt):
        shifted_a_scale = [
            shift_scale_word(a_scale_sub[sub], kt)
            for sub in range_constexpr(_kScaleSubBlocks)
        ]
        shifted_b_scale = [
            shift_scale_word(b_scale_slot[mw], kt)
            for mw in range_constexpr(2)
        ]
        for J in range_constexpr(4):
            mni = J // 2
            in_b = J % 2
            sb = shifted_b_scale[mni]
            for sub in range_constexpr(_kScaleSubBlocks):
                sa = shifted_a_scale[sub]
                i0 = sub * 2
                for half in range_constexpr(_kHalves):
                    if const_expr(kt * _kHalves + half >= _n_real_half):
                        continue
                    for row_group in range_constexpr(2):
                        i = i0 + row_group
                        if const_expr(i >= _kMChunks):
                            continue
                        acc_in = (
                            zero4
                            if const_expr(init and half == 0)
                            else accm[i][J]
                        )
                        accm[i][J] = (
                            rocdl.mfma_scale_f32_16x16x128_f8f6f4(
                                mfma_res_ty,
                                [
                                    a[i][half],
                                    b_tile[J][half],
                                    acc_in,
                                    4,
                                    4,
                                    2 * half + row_group,
                                    sa,
                                    2 * half + in_b,
                                    sb,
                                ],
                            )
                        )
```

所有调用都显式传 `kt`。短路径保持：

```python
            mfma_cluster(
                b[kt],
                a,
                a_scale_sub,
                b_scale_v[kt],
                init=(S == 0),
                kt=kt,
            )
```

长路径两个调用分别改为：

```python
            mfma_cluster(
                b[kt],
                a,
                a_scale_sub,
                b_scale_v[kt],
                init=(OFFSET == 0),
                kt=kt,
            )
```

```python
            mfma_cluster(
                b[kt],
                a,
                a_scale_sub,
                b_scale_v[kt],
                init=False,
                kt=kt,
            )
```

并把所有 `range_constexpr(_kSubBlocks)` 改为
`range_constexpr(_kScaleSubBlocks)`；A load 循环独立使用 `_load_groups`。

- [ ] **Step 8: 运行最小 GPU 测试**

Run:

```bash
FLYDSL_RUNTIME_ENABLE_CACHE=0 \
python -m pytest \
  op_tests/flydsl_tests/test_native_mxmoe_gemm2_bk128_gpu.py \
  -q
```

Expected: `k384-bk128` 与 `k512-bk256` 均 PASS。

- [ ] **Step 9: 运行 host 回归与 linter**

Run:

```bash
python -m pytest \
  op_tests/flydsl_tests/test_native_mxmoe_gemm2_bk128.py \
  -q
python -m ruff check \
  aiter/ops/flydsl/kernels/mxfp4_gemm_common.py \
  aiter/ops/flydsl/kernels/mxfp4_gemm2.py \
  op_tests/flydsl_tests/test_native_mxmoe_gemm2_bk128_gpu.py
```

Expected: 全部 PASS。

- [ ] **Step 10: 提交 core**

```bash
git add \
  aiter/ops/flydsl/kernels/mxfp4_gemm_common.py \
  aiter/ops/flydsl/kernels/mxfp4_gemm2.py \
  op_tests/flydsl_tests/test_native_mxmoe_gemm2_bk128_gpu.py
git commit -m "feat(flydsl): add native mxmoe gemm2 BK128 core"
```

---

### Task 4: 扩展到全部 11 个 native variant 与三组 K/BK

**Files:**
- Modify: `op_tests/flydsl_tests/test_native_mxmoe_gemm2_bk128_gpu.py`
- Modify if a test exposes a core defect: `aiter/ops/flydsl/kernels/mxfp4_gemm2.py`

**Interfaces:**
- Consumes: Task 3 的 direct `flydsl_mxfp4_gemm2`，并显式传入
  `BN=256, BK=128|256`。
- Produces: 33-case GPU correctness matrix，覆盖 atomic/nonatomic/cshuffle/f4out。
- Produces: direct test helper `_run_native_case(bm, use_nt, epilog, inter_dim, bk)`。

- [ ] **Step 1: 追加 direct stage2 运行与 reduce helper**

在 GPU test 文件顶部 import 区加入：

```python
import aiter

from aiter.fused_moe import _mxfp4_a4w4_stage1_fw, moe_sorting
from aiter.ops.flydsl.moe_common import (
    DEFAULT_SITUV2_BETA,
    DEFAULT_SITUV2_LINEAR_BETA,
)
from aiter.ops.flydsl.mxfp4_gemm2_kernels import flydsl_mxfp4_gemm2
```

追加 helper：

```python
_G1_VARIANT = {
    16: (True, True),
    32: (False, False),
    64: (False, False),
    128: (False, False),
}


def _run_native_case(bm, use_nt, epilog, inter_dim, bk):
    token, model_dim, expert, topk = bm + 1, 256, 2, 2
    activation = ActivationType.Silu
    data = Mxfp4FlydslTuner._prepare_case(
        token,
        model_dim,
        inter_dim,
        expert,
        topk,
        dtypes.bf16,
        seed=1000 + bm + inter_dim + bk,
    )
    g1_nt, g1_inline = _G1_VARIANT[bm]
    kn1 = Mxfp4FlydslTuner._g1_kname(bm, g1_nt, g1_inline)
    atomic = epilog == "atomic"
    sti, sw, sei, nvi, moe_buf, m_indices, reverse_sorted = moe_sorting(
        data["topk_ids"],
        data["topk_weights"],
        expert,
        model_dim,
        dtypes.bf16,
        block_size=bm,
        accumulate=atomic,
        output_aux=True,
    )
    inter_q, inter_scale = _mxfp4_a4w4_stage1_fw(
        data["input"],
        data["w1_a16"],
        data["w2_a16"],
        sti,
        sei,
        nvi,
        None,
        topk,
        block_m=bm,
        w1_scale=data["w1s_a16"],
        kernelName1=kn1,
        m_indices=m_indices,
        moe_buf=moe_buf,
        activation=activation,
        situ_beta=DEFAULT_SITUV2_BETA,
        situ_linear_beta=DEFAULT_SITUV2_LINEAR_BETA,
    )

    max_sorted = sti.shape[0]
    mxfp4out = epilog == "nonatomic_mxfp4"
    cshuffle = epilog == "nonatomic_cshuffle"
    out = (
        moe_buf
        if atomic and moe_buf.numel()
        else torch.zeros((token, model_dim), dtype=dtypes.bf16, device="cuda")
    )
    flat_out_scale = None
    if atomic:
        flat_out = out
    elif mxfp4out:
        flat_out = torch.full(
            (max_sorted, model_dim // 2),
            0xFF,
            dtype=torch.uint8,
            device="cuda",
        )
        flat_out_scale = torch.full(
            (max_sorted, model_dim // 32),
            0xFF,
            dtype=torch.uint8,
            device="cuda",
        )
    else:
        flat_out = torch.empty(
            (max_sorted, model_dim), dtype=dtypes.bf16, device="cuda"
        )

    flydsl_mxfp4_gemm2(
        inter_sorted_quant=inter_q,
        inter_sorted_shuffled_scale=inter_scale,
        w2_u8=data["w2_a16"],
        w2_scale_u8=data["w2s_a16"],
        sorted_expert_ids=sei,
        cumsum_tensor=nvi,
        sorted_token_ids=sti,
        sorted_weights=sw,
        flat_out=flat_out,
        M_logical=token,
        max_sorted=max_sorted,
        BM=bm,
        use_nt=use_nt,
        atomic=atomic,
        mxfp4out=mxfp4out,
        NE=expert,
        D_HIDDEN=model_dim,
        D_INTER=inter_dim,
        topk=topk,
        flat_out_scale=flat_out_scale,
        cshuffle=cshuffle,
        D_INTER_REAL=None,
        BN=256,
        BK=bk,
        xcd_swizzle=0,
    )

    if mxfp4out:
        assert (flat_out != 0xFF).any(), "f4out values were not written"
        assert (flat_out_scale != 0xFF).any(), "f4out scales were not written"
        aiter.mxfp4_moe_scatter_reduce_q(
            flat_out_q=flat_out,
            flat_out_scale=flat_out_scale,
            reverse_sorted=reverse_sorted,
            sorted_weights=sw,
            out=out,
            NE=expert,
            TOPK=topk,
            D_HIDDEN=model_dim,
            MB=bm,
        )
    elif not atomic:
        aiter.mxfp4_moe_scatter_reduce(
            flat_out=flat_out,
            reverse_sorted=reverse_sorted,
            sorted_weights=sw,
            out=out,
            NE=expert,
            TOPK=topk,
            D_HIDDEN=model_dim,
            MB=bm,
        )

    ref = Mxfp4FlydslTuner._torch_ref(
        data,
        topk,
        dtypes.bf16,
        activation,
        DEFAULT_SITUV2_BETA,
        DEFAULT_SITUV2_LINEAR_BETA,
    )
    return ref.float(), out.float()
```

- [ ] **Step 2: 追加 33-case 参数矩阵**

```python
_NATIVE_VARIANTS = [
    pytest.param(16, False, "atomic", id="bm16-atomic"),
    pytest.param(16, True, "atomic", id="bm16-atomic-nt"),
    pytest.param(32, False, "atomic", id="bm32-atomic"),
    pytest.param(32, True, "atomic", id="bm32-atomic-nt"),
    pytest.param(64, False, "atomic", id="bm64-atomic"),
    pytest.param(64, True, "atomic", id="bm64-atomic-nt"),
    pytest.param(128, False, "nonatomic", id="bm128-nonatomic"),
    pytest.param(128, False, "nonatomic_mxfp4", id="bm128-f4out"),
    pytest.param(32, False, "nonatomic_cshuffle", id="bm32-cshuffle"),
    pytest.param(64, False, "nonatomic_cshuffle", id="bm64-cshuffle"),
    pytest.param(128, False, "nonatomic_cshuffle", id="bm128-cshuffle"),
]

_K_CASES = [
    pytest.param(384, 128, id="k384-bk128"),
    pytest.param(512, 128, id="k512-bk128"),
    pytest.param(512, 256, id="k512-bk256"),
]


@pytest.mark.parametrize(("bm", "use_nt", "epilog"), _NATIVE_VARIANTS)
@pytest.mark.parametrize(("inter_dim", "bk"), _K_CASES)
@_SKIP_GFX950_FLYDSL
def test_native_full_variant_matrix(bm, use_nt, epilog, inter_dim, bk):
    ref, out = _run_native_case(bm, use_nt, epilog, inter_dim, bk)
    _check_close(
        ref,
        out,
        f"native_{epilog}_bm{bm}_nt{int(use_nt)}_k{inter_dim}_bk{bk}",
    )
```

- [ ] **Step 3: 先运行最敏感的 K384/BK128 矩阵**

Run:

```bash
FLYDSL_RUNTIME_ENABLE_CACHE=0 \
python -m pytest \
  op_tests/flydsl_tests/test_native_mxmoe_gemm2_bk128_gpu.py \
  -k "full_variant_matrix and k384-bk128" \
  -q
```

Expected: 11 PASS。若某个 BM/epilogue失败，只修复该测试暴露的 core defect，不改 epilogue API 或扩大设计范围。

- [ ] **Step 4: 运行 K512/BK128 与 BK256 对照**

Run:

```bash
FLYDSL_RUNTIME_ENABLE_CACHE=0 \
python -m pytest \
  op_tests/flydsl_tests/test_native_mxmoe_gemm2_bk128_gpu.py \
  -k "full_variant_matrix and k512" \
  -q
```

Expected: 22 PASS。

- [ ] **Step 5: 运行完整 GPU 文件与 host 回归**

Run:

```bash
FLYDSL_RUNTIME_ENABLE_CACHE=0 \
python -m pytest \
  op_tests/flydsl_tests/test_native_mxmoe_gemm2_bk128_gpu.py \
  op_tests/flydsl_tests/test_native_mxmoe_gemm2_bk128.py \
  -q
```

Expected: 全部 PASS。

- [ ] **Step 6: 运行 linter**

Run:

```bash
python -m ruff check \
  aiter/ops/flydsl/kernels/mxfp4_gemm2.py \
  op_tests/flydsl_tests/test_native_mxmoe_gemm2_bk128_gpu.py
```

Expected: PASS。

- [ ] **Step 7: 提交完整 correctness 矩阵**

```bash
git add \
  aiter/ops/flydsl/kernels/mxfp4_gemm2.py \
  op_tests/flydsl_tests/test_native_mxmoe_gemm2_bk128_gpu.py
git commit -m "test(flydsl): cover native mxmoe gemm2 BK128 matrix"
```

---

### Task 5: 验证 AOT、缓存隔离、BK256 IR/ISA 与 tuner 性能

**Files:**
- Verify only; no source file should change.
- Temporary artifacts: `/tmp/native_mxmoe_bk128_aot.csv`, `/tmp/native_mxmoe_bk256_after`, `/tmp/native_mxmoe_tune_*.csv`

**Interfaces:**
- Consumes: Tasks 1-4 的完整 feature。
- Produces: 最终验证证据，不产生仓库文件或额外 commit。

- [ ] **Step 1: 运行全部新增 host/GPU 测试**

```bash
FLYDSL_RUNTIME_ENABLE_CACHE=0 \
python -m pytest \
  op_tests/flydsl_tests/test_native_mxmoe_gemm2_bk128.py \
  op_tests/flydsl_tests/test_native_mxmoe_gemm2_bk128_gpu.py \
  -q
```

Expected: 全部 PASS。

- [ ] **Step 2: 运行相关既有 FlyDSL MoE 回归**

```bash
python -m pytest \
  op_tests/flydsl_tests/test_flydsl_moe_a8w4.py \
  -q
```

Expected: PASS；v2 BK128/BK256 行为无回退。

- [ ] **Step 3: 验证 cold-cache 与 warm-cache 路径**

```bash
rm -rf /tmp/native_mxmoe_bk128_cache
FLYDSL_RUNTIME_CACHE_DIR=/tmp/native_mxmoe_bk128_cache \
python -m pytest \
  op_tests/flydsl_tests/test_native_mxmoe_gemm2_bk128_gpu.py \
  -k "native_atomic_bm32_direct_stage2 and k384-bk128" \
  -q
FLYDSL_RUNTIME_CACHE_DIR=/tmp/native_mxmoe_bk128_cache \
python -m pytest \
  op_tests/flydsl_tests/test_native_mxmoe_gemm2_bk128_gpu.py \
  -k "native_atomic_bm32_direct_stage2 and k384-bk128" \
  -q
```

Expected: 两次均 PASS；第一次生成缓存，第二次复用同一缓存。

- [ ] **Step 4: 用同一 CSV AOT 编译 BK128 与 BK256**

生成临时 CSV：

```bash
python - <<'PY'
import csv

path = "/tmp/native_mxmoe_bk128_aot.csv"
fields = [
    "topk",
    "model_dim",
    "expert",
    "inter_dim",
    "kernelName1",
    "kernelName2",
    "cu_num",
    "act_type",
]
rows = [
    {
        "topk": 2,
        "model_dim": 256,
        "expert": 2,
        "inter_dim": 384,
        "kernelName1": "flydsl_mxmoe_g1_a4w4_32x256x256",
        "kernelName2": "flydsl_mxmoe_g2_a4w4_32x256x128_atomic",
        "cu_num": 256,
        "act_type": "ActivationType.Silu",
    },
    {
        "topk": 2,
        "model_dim": 256,
        "expert": 2,
        "inter_dim": 512,
        "kernelName1": "flydsl_mxmoe_g1_a4w4_32x256x256",
        "kernelName2": "flydsl_mxmoe_g2_a4w4_32x256x128_atomic",
        "cu_num": 256,
        "act_type": "ActivationType.Silu",
    },
    {
        "topk": 2,
        "model_dim": 256,
        "expert": 2,
        "inter_dim": 512,
        "kernelName1": "flydsl_mxmoe_g1_a4w4_32x256x256",
        "kernelName2": "flydsl_mxmoe_g2_a4w4_32x256x256_atomic",
        "cu_num": 256,
        "act_type": "ActivationType.Silu",
    },
]
with open(path, "w", newline="") as file:
    writer = csv.DictWriter(file, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
print(path)
PY
python -m aiter.aot.flydsl.mxfp4_moe \
  --csv /tmp/native_mxmoe_bk128_aot.csv
```

Expected: stage2 三个 job 均显示 `[OK]`；K512/BK128 与 K512/BK256 不被 dedup。

- [ ] **Step 5: 生成 after IR/ISA 并与 baseline 比较**

```bash
rm -rf /tmp/native_mxmoe_bk256_after
FLYDSL_DUMP_IR=1 \
FLYDSL_DUMP_DIR=/tmp/native_mxmoe_bk256_after \
FLYDSL_RUNTIME_ENABLE_CACHE=0 \
python -m pytest \
  op_tests/flydsl_tests/test_native_mxmoe_gemm2_bk128_gpu.py \
  -k "native_atomic_bm32_direct_stage2 and k512-bk256" \
  -q
```

标准化内部 symbol 后比较 LLVM IR 与 ISA：

```bash
python - <<'PY'
from pathlib import Path
import re

roots = {
    "before": Path("/tmp/native_mxmoe_bk256_before"),
    "after": Path("/tmp/native_mxmoe_bk256_after"),
}
for label, root in roots.items():
    candidates = [
        path
        for path in root.glob("**/21_final_isa.s")
        if "gemm2_a4w4_port" in str(path.parent)
    ]
    assert len(candidates) == 1, (label, candidates)
    for filename in ("20_llvm_ir.ll", "21_final_isa.s"):
        text = (candidates[0].parent / filename).read_text()
        text = re.sub(
            r"gemm2_a4w4_port_[A-Za-z0-9_.$]+",
            "gemm2_a4w4_port_NORMALIZED",
            text,
        )
        Path(f"/tmp/{label}_{filename}").write_text(text)
PY
diff -u /tmp/before_20_llvm_ir.ll /tmp/after_20_llvm_ir.ll
diff -u /tmp/before_21_final_isa.s /tmp/after_21_final_isa.s
```

Expected: 不假设两个 `diff` 无输出。BM>=32 必须保持 load/address/MFMA
语义等价；BM16 必须保留 inactive-wave A-load guard。记录 normalized ISA
的全部剩余差异、A-load/guard 数量和 VGPR/SGPR/LDS/MFMA，而不是把差异
描述为仅 symbol 变化。只有出现未设计的语义或性能回退时才增加
BK-sensitive 编译期分支，并重新执行 Tasks 3-5 的测试。

- [ ] **Step 6: 运行两组 shape 的 tuner smoke 与 benchmark**

生成临时 untuned CSV：

```bash
python - <<'PY'
import csv

path = "/tmp/native_mxmoe_tune_untuned.csv"
fields = [
    "gfx",
    "cu_num",
    "token",
    "model_dim",
    "inter_dim",
    "expert",
    "topk",
    "act_type",
    "dtype",
    "q_dtype_a",
    "q_dtype_w",
    "q_type",
    "use_g1u1",
    "doweight_stage1",
]
base = {
    "gfx": "gfx950",
    "cu_num": 256,
    "token": 4,
    "model_dim": 256,
    "expert": 2,
    "topk": 2,
    "act_type": "ActivationType.Silu",
    "dtype": "torch.bfloat16",
    "q_dtype_a": "torch.float4_e2m1fn_x2",
    "q_dtype_w": "torch.float4_e2m1fn_x2",
    "q_type": "QuantType.per_1x32",
    "use_g1u1": 1,
    "doweight_stage1": 0,
}
with open(path, "w", newline="") as file:
    writer = csv.DictWriter(file, fieldnames=fields)
    writer.writeheader()
    writer.writerow({**base, "inter_dim": 384})
    writer.writerow({**base, "inter_dim": 512})
print(path)
PY
rm -f /tmp/native_mxmoe_tune_result.csv
python csrc/ck_gemm_moe_2stages_codegen/gemm_moe_tune.py \
  --mxfp4-flydsl \
  --untune_file /tmp/native_mxmoe_tune_untuned.csv \
  --tune_file /tmp/native_mxmoe_tune_result.csv \
  --mp 1 \
  --warmup 5 \
  --iters 31 \
  --all
```

Expected:

- K384 日志只出现 native `x256x128` 候选；
- K512 日志同时出现 native `x256x128` 与 `x256x256`；
- 每个候选先通过 cosine correctness，再打印 `us`；
- 输出 CSV 每个 shape 只保留最低 `us` winner。

- [ ] **Step 7: 对比 BK256 改动前后性能**

```bash
python - <<'PY'
from pathlib import Path

from aiter import ActivationType, dtypes
from aiter.test_common import run_perftest
from csrc.ck_gemm_moe_2stages_codegen.gemm_moe_tune import Mxfp4FlydslTuner

token, model_dim, inter_dim, expert, topk, bm = 33, 256, 512, 2, 2, 32
data = Mxfp4FlydslTuner._prepare_case(
    token,
    model_dim,
    inter_dim,
    expert,
    topk,
    dtypes.bf16,
    seed=123,
)
kn1 = Mxfp4FlydslTuner._g1_kname(bm, False, False)
kn2 = Mxfp4FlydslTuner._g2_kname(bm, False, "atomic", 256)


def call():
    return Mxfp4FlydslTuner._port_e2e(
        data,
        kn1,
        kn2,
        topk,
        expert,
        model_dim,
        dtypes.bf16,
        ActivationType.Silu,
        4.0,
        25.0,
    )


call()
_, after_us = run_perftest(call, num_warmup=20, num_iters=101)
before_us = float(
    Path("/tmp/native_mxmoe_bk256_before_us.txt").read_text().strip()
)
after_us = float(after_us)
regression = (after_us / before_us - 1.0) * 100.0
print(
    f"BK256 before={before_us:.4f} us after={after_us:.4f} us "
    f"regression={regression:.2f}%"
)
if regression > 3.0:
    raise SystemExit(
        "BK256 regression exceeds 3%; inspect IR/ISA and repeat measurement"
    )
PY
```

Expected: regression `<=3%`。若超过阈值，先重复测量排除噪声；可复现时恢复 BK256 compile-time 专用分支。

- [ ] **Step 8: 最终工作树与提交历史检查**

```bash
git status --short
git log --oneline -5
```

Expected: 只有用户原先存在的未跟踪文件；本计划涉及的源代码和测试均已提交，最近历史包含四个独立提交。

---

### Final-review corrections and implementation deviations

These corrections supersede conflicting preservation wording above:

1. **BM16 wave guard**
   - Both initial and rotating-pipeline A loads are issued only by
     `n_load_waves`.
   - For BM16/BK256, the old long-pipeline refill let waves 2/3 write rows
     beyond the BM16 slot into adjacent rotating LDS slots. The guard is an
     intentional fix; BM>=32 BK256 remains semantically equivalent.
   - K512/BK256 still has only its already-guarded two initial static A-load
     sites. The guard's semantic delta is visible in a long pipeline:
     K1024 changes guarded groups `1 -> 3` and derived dynamic wave-load
     executions `12 -> 8`.

2. **Direct-stage2 test deviation**
   - The final 33-case matrix uses a deterministic direct-stage2 fixture built
     from exact quantized operands instead of routing every matrix case through
     native stage1 as originally sketched in Task 4.
   - This isolates GEMM2 variants and makes expert/scale/output writes
     discriminative. The separate K384/BK128 high-level smoke retains complete
     fused dispatch coverage, so the deviation does not remove end-to-end
     validation.

3. **Cache dependency and symbol**
   - `launch_gemm2` directly references the module-level `_gemm2_body`
     JitFunction. FlyDSL therefore records
     `jit:_gemm2_body:<manager_key>` automatically.
   - `NATIVE_GEMM2_CORE_CACHE_EPOCH` remains only for explicit forced
     cache/symbol invalidation; the symbol contract is
     `_core<epoch>_bk<BK>`.

4. **Final BM16/BK256 K512 evidence**
   - Base source: commit
     `375c47524f86c91f03c12227fed27e39170088b8`, extracted without a worktree.
   - Correctness: before/after passed and produced bitwise-equal outputs.
   - Normalized ISA is not identical: the K512 diff contains an independent
     A-address instruction reorder. Both versions use 2 static A-load sites,
     1 guarded group/4 dynamic wave-load executions, 90 VGPR, 29 SGPR,
     20,480 LDS bytes, and 16 static MFMA instructions.
   - Same-process 12-round alternating medians were 8.50897 us before and
     8.50521 us after (-0.044%).
   - Reproducible artifacts:
     `/tmp/native_mxmoe_bm16_bk256_ab/static_isa_summary.json`,
     `/tmp/native_mxmoe_bm16_bk256_ab/k512_normalized_isa.diff`, and
     `/tmp/native_mxmoe_bm16_bk256_ab/bm16_bk256_k512_alternating_perf.json`.

