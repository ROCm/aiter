# V2 GEMM Turbo SPART Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a permanent, independently switchable unsigned/direct-coordinate SPART optimization to `feat/v2-gemm-turbo`, validate exact mapping equivalence, and select its default from this branch's own correctness, IR/ISA, and five-run stage2 measurements.

**Architecture:** Preserve the current signed and flatten/redivide implementation as the switch-off path. The switch-on path changes only non-persistent SPART mapping: it uses unsigned arithmetic for non-negative coordinates and passes resolved `(m_block_idx, n_block_idx)` directly into the current GEMM2 body. Current `g2_kstages`, tile support, B pipeline, LDS, MFMA, grid, and epilogue behavior remain unchanged.

**Tech Stack:** Python, pytest, FlyDSL, MLIR/AMDGPU ISA, gfx950, `op_tests/test_moe_2stage.py`.

---

## File Map

- Create `op_tests/flydsl_tests/test_v2_gemm2_spart_opt.py`: CPU-safe TDD coverage for control/cache plumbing, unsigned helpers, full mapping equivalence, and coordinate resolution.
- Modify `aiter/ops/flydsl/kernels/mxmoe_dispatcher.py`: add the permanent control, cache/tag identity, unsigned mapper, and direct-coordinate dispatch.
- Modify `aiter/ops/flydsl/kernels/mxmoe_gemm_v2.py`: add optional precomputed coordinates and preserve flat-ID fallback.
- Read only `/root/workspace/aiter/docs/fp8_retune_config/glm5_fp4_pathB_flydslv2_tuned_20260722_141743.csv`: drive the exact GLM correctness/performance case.
- Generate IR/ISA and logs only under `/tmp`.
- Do not stage or modify `3rdparty/composable_kernel`.

## Task 1: Write Failing Focused Tests

**Files:**
- Create: `op_tests/flydsl_tests/test_v2_gemm2_spart_opt.py`
- Test: `op_tests/flydsl_tests/test_v2_gemm2_spart_opt.py`

- [ ] **Step 1: Create the test module with control and helper tests**

Create the file with:

```python
import pytest

import aiter.ops.flydsl.kernels.mxmoe_dispatcher as dispatcher
from aiter.ops.flydsl.kernels import mxmoe_gemm_v2


def test_get_g2_keys_and_forwards_spart_opt(monkeypatch):
    calls = []
    monkeypatch.setattr(dispatcher, "G2_CACHE", {})
    monkeypatch.setattr(
        dispatcher,
        "compile_gemm2_a4w4_port",
        lambda **kwargs: calls.append(kwargs) or object(),
    )
    monkeypatch.setenv("MXFP4_G2_KSTAGES", "2")
    monkeypatch.setenv("MXFP4_G2_BHOIST", "1")
    monkeypatch.setenv("MXFP4_G2_ASCALE_PF", "1")
    monkeypatch.setenv("MXFP4_G2_SPART", "402")
    monkeypatch.setenv("MXFP4_G2_BF16_LDS", "0")

    common = {
        "BM": 32,
        "BN": 128,
        "BK": 256,
        "use_nt": True,
        "HIDDEN_MAX": 8192,
        "epilog": "atomic",
        "INTER_MAX": 8192,
        "a_dtype": "fp4",
    }
    for enabled in (False, True):
        monkeypatch.setenv("MXFP4_G2_SPART_OPT", "1" if enabled else "0")
        dispatcher.get_g2(**common)

    assert [call.get("g2_spart_opt") for call in calls] == [False, True]
    assert [call["g2_kstages"] for call in calls] == [2, 2]


def test_unsigned_spart_helpers_wrap_operands(monkeypatch):
    calls = []

    def as_uint32(value):
        calls.append(("u32", int(value)))
        return int(value)

    def as_int32(value):
        calls.append(("i32", int(value)))
        return int(value)

    monkeypatch.setattr(dispatcher.fx, "Uint32", as_uint32)
    monkeypatch.setattr(dispatcher.fx, "Int32", as_int32)

    udiv = getattr(dispatcher, "_udiv_i32", None)
    umod = getattr(dispatcher, "_umod_i32", None)
    assert udiv is not None
    assert umod is not None
    assert udiv(9, 4) == 2
    assert umod(9, 4) == 1
    assert calls == [
        ("u32", 9),
        ("u32", 4),
        ("i32", 2),
        ("u32", 9),
        ("u32", 4),
        ("i32", 1),
    ]


def test_resolve_tile_coords_precomputed_and_fallback_paths():
    class NoDivide:
        def __floordiv__(self, _other):
            raise AssertionError("flat tile id must not be divided")

    resolve = getattr(mxmoe_gemm_v2, "_resolve_tile_coords", None)
    assert resolve is not None
    assert resolve(NoDivide(), 48, 3, 7) == (3, 7)
    assert resolve(99, 48) == (2, 3)
    with pytest.raises(AssertionError):
        resolve(99, 48, 2, None)
    with pytest.raises(AssertionError):
        resolve(99, 48, None, 3)
```

- [ ] **Step 2: Add the host numeric shim and complete mapping test**

Append:

```python
@pytest.mark.parametrize(
    ("m0", "n0", "group_num", "m01"),
    [
        (257, 48, 4, 2),
        (3, 5, 4, 2),
        (4, 7, 4, 2),
        (7, 6, 5, 3),
    ],
)
def test_unsigned_spart_mapping_matches_signed_permutation(
    monkeypatch, m0, n0, group_num, m01
):
    uint32_constructions = 0

    class HostPredicate:
        def __init__(self, value):
            self.value = bool(value)

        def select(self, true_value, false_value):
            return true_value if self.value else false_value

    class HostInt:
        def __init__(self, value):
            self.value = int(value)

        @staticmethod
        def _value(other):
            return int(other)

        def __int__(self):
            return self.value

        def __add__(self, other):
            return type(self)(self.value + self._value(other))

        def __radd__(self, other):
            return type(self)(self._value(other) + self.value)

        def __sub__(self, other):
            return type(self)(self.value - self._value(other))

        def __rsub__(self, other):
            return type(self)(self._value(other) - self.value)

        def __mul__(self, other):
            return type(self)(self.value * self._value(other))

        def __rmul__(self, other):
            return type(self)(self._value(other) * self.value)

        def __floordiv__(self, other):
            return type(self)(self.value // self._value(other))

        def __rfloordiv__(self, other):
            return type(self)(self._value(other) // self.value)

        def __mod__(self, other):
            return type(self)(self.value % self._value(other))

        def __rmod__(self, other):
            return type(self)(self._value(other) % self.value)

        def __le__(self, other):
            return HostPredicate(self.value <= self._value(other))

        def __lt__(self, other):
            return HostPredicate(self.value < self._value(other))

    class HostInt32(HostInt):
        pass

    class HostUint32(HostInt):
        def __init__(self, value):
            nonlocal uint32_constructions
            super().__init__(value)
            uint32_constructions += 1

    monkeypatch.setattr(dispatcher.fx, "Int32", HostInt32)
    monkeypatch.setattr(dispatcher.fx, "Uint32", HostUint32)

    def output_sequence(use_unsigned):
        return [
            tuple(
                int(coord)
                for coord in dispatcher._spart_output_tile_index(
                    HostInt32(block_id),
                    HostInt32(m0),
                    n0,
                    group_num,
                    m01,
                    use_unsigned=use_unsigned,
                )
            )
            for block_id in range(m0 * n0)
        ]

    signed = output_sequence(use_unsigned=False)
    assert uint32_constructions == 0
    unsigned = output_sequence(use_unsigned=True)

    assert uint32_constructions > 0
    assert unsigned == signed
    assert all(0 <= m < m0 and 0 <= n < n0 for m, n in unsigned)
    assert len(unsigned) == m0 * n0
    assert set(unsigned) == {(m, n) for m in range(m0) for n in range(n0)}
```

- [ ] **Step 3: Run the new test file and verify RED**

Run:

```bash
/opt/venv/bin/python -m pytest op_tests/flydsl_tests/test_v2_gemm2_spart_opt.py -q
```

Expected: failures because `g2_spart_opt`, `_udiv_i32`, `_umod_i32`, `_resolve_tile_coords`, and the mapper's `use_unsigned` argument are absent. Fix only test-mechanics errors before production code.

## Task 2: Implement Control, Unsigned Mapping, and Direct Coordinates

**Files:**
- Modify: `aiter/ops/flydsl/kernels/mxmoe_dispatcher.py:42-78`
- Modify: `aiter/ops/flydsl/kernels/mxmoe_dispatcher.py:81-184`
- Modify: `aiter/ops/flydsl/kernels/mxmoe_dispatcher.py:241-308`
- Modify: `aiter/ops/flydsl/kernels/mxmoe_dispatcher.py:425-500`
- Modify: `aiter/ops/flydsl/kernels/mxmoe_gemm_v2.py:180-280`
- Test: `op_tests/flydsl_tests/test_v2_gemm2_spart_opt.py`

- [ ] **Step 1: Add unsigned helpers and preserve the switch-off mapper**

Add above `_spart_output_tile_index`:

```python
def _udiv_i32(a, c):
    return fx.Int32(fx.Uint32(a) // fx.Uint32(c))


def _umod_i32(a, c):
    return fx.Int32(fx.Uint32(a) % fx.Uint32(c))
```

Change the mapper signature to:

```python
def _spart_output_tile_index(
    block_1d_id, M0, N0, group_num, m01, *, use_unsigned=False
):
```

Inside it use:

```python
    div = _udiv_i32 if use_unsigned else lambda a, b: a // b
```

Replace the mapper arithmetic with:

```python
    gn = fx.Int32(group_num)
    n0 = fx.Int32(N0)
    m01c = fx.Int32(m01)
    mn = M0 * n0
    group_size = div(mn + gn - fx.Int32(1), gn)
    big_group_num = gn - (group_size * gn - mn)
    group_id_y = div(block_1d_id, gn)
    group_id_x = (
        _umod_i32(block_1d_id, gn)
        if use_unsigned
        else block_1d_id - group_id_y * gn
    )
    remap_a = group_id_x * group_size + group_id_y
    remap_b = group_id_x * group_size + big_group_num - group_id_x + group_id_y
    remap = (group_id_x <= big_group_num).select(remap_a, remap_b)
    idx_M0 = div(remap, n0)
    idx_N0 = remap - idx_M0 * n0
    M0_tmp = div(M0, m01c)
    M0_mod = _umod_i32(M0, m01c) if use_unsigned else M0 - M0_tmp * m01c
    M01_adapt = (idx_M0 < (M0 - M0_mod)).select(m01c, M0_mod)
    idx_M00 = div(idx_M0, m01c)
    idx_M01 = (
        _umod_i32(idx_M0, m01c)
        if use_unsigned
        else idx_M0 - idx_M00 * m01c
    )
    idx_local = idx_N0 + idx_M01 * n0
    N_out = div(idx_local, M01_adapt)
    loc_mod = idx_local - N_out * M01_adapt
    return loc_mod + idx_M00 * m01c, N_out
```

- [ ] **Step 2: Add the permanent compile control and kernel tag**

Add `g2_spart_opt=None` immediately after `g2_spart=None` in `compile_gemm2_a4w4_port`.

After current SPART parse/validation, add provisional default-on parsing:

```python
    if g2_spart_opt is None:
        g2_spart_opt = os.environ.get("MXFP4_G2_SPART_OPT", "1") == "1"
    g2_spart_opt = bool(g2_spart_opt) and g2_spart > 0 and not persist
```

Add:

```python
    spart_opt_tag = "_spartopt" if g2_spart_opt else ""
```

Insert `{spart_opt_tag}` immediately after `{spart_tag}` in the final kernel tag.

- [ ] **Step 3: Add coordinate resolution to the current GEMM2 body**

Add above `gemm2_body_v2`:

```python
def _resolve_tile_coords(
    bx_i32,
    num_n_blocks,
    precomputed_m_block_idx=None,
    precomputed_n_block_idx=None,
):
    if precomputed_m_block_idx is not None:
        assert precomputed_n_block_idx is not None
        return precomputed_m_block_idx, precomputed_n_block_idx
    assert precomputed_n_block_idx is None
    m_block_idx = bx_i32 // num_n_blocks
    n_block_idx = bx_i32 - m_block_idx * num_n_blocks
    return m_block_idx, n_block_idx
```

Add these positional defaults after `i32_npad` and before `*`:

```python
    precomputed_m_block_idx=None,
    precomputed_n_block_idx=None,
```

Replace the current body coordinate calculation with:

```python
    m_block_idx, n_block_idx = _resolve_tile_coords(
        bx_i32,
        num_n_blocks,
        precomputed_m_block_idx,
        precomputed_n_block_idx,
    )
```

Do not add `g2_spart_opt` to the body keyword-only parameters.

- [ ] **Step 4: Forward direct coordinates from the current dispatcher**

Change the nested helper to:

```python
        def run_unit(
            unit_bx, precomputed_m_block_idx=None, precomputed_n_block_idx=None
        ):
            gemm2_body_v2(
                lds_base_i32,
                arg_ascale,
                arg_bq,
                arg_bscale,
                arg_eids,
                arg_stids,
                arg_sweights,
                i32_M,
                i32_max_m_blocks,
                arg_out,
                unit_bx,
                lane,
                wave,
                arg_aq,
                i32_inter,
                i32_hidden,
                i32_kpad,
                i32_npad,
                precomputed_m_block_idx,
                precomputed_n_block_idx,
                BM=BM,
                BN=BN,
                BK=BK,
                use_nt=use_nt,
                INTER_MAX=INTER_MAX,
                aStages=aStages,
                a_dtype=a_dtype,
                use_reduce=use_reduce,
                topk=topk,
                has_pad=has_pad,
                SBM=SBM,
                g2_kstages=g2_kstages,
                g2_bhoist=g2_bhoist,
                g2_ascale_pf=g2_ascale_pf,
                g2_bf16_lds=g2_bf16_lds,
                route_out_fp8=route_out_fp8,
            )
```

In the non-persistent SPART branch, call:

```python
                m_block_idx, n_block_idx = _spart_output_tile_index(
                    bx_i32,
                    total_m_blocks,
                    num_n_blocks,
                    g2_group_num,
                    g2_m01,
                    use_unsigned=g2_spart_opt,
                )
                if const_expr(g2_spart_opt):
                    issue_all_a_loads(m_block_idx * BM)
                    rocdl.sched_barrier(0)
                    run_unit(fx.Int32(0), m_block_idx, n_block_idx)
                else:
                    unit_bx = m_block_idx * num_n_blocks + n_block_idx
                    issue_all_a_loads(m_block_idx * BM)
                    rocdl.sched_barrier(0)
                    run_unit(unit_bx)
```

Retain the current naive and persistent branches exactly.

- [ ] **Step 5: Add the control to `get_g2` cache identity and forwarding**

After reading `g2_spart`, add:

```python
    g2_spart_opt = os.environ.get("MXFP4_G2_SPART_OPT", "1") == "1"
    g2_spart_opt = bool(g2_spart_opt) and g2_spart > 0 and not persist
```

Insert `g2_spart_opt` immediately after `g2_spart` in the cache key, and forward `g2_spart_opt=g2_spart_opt` immediately after `g2_spart=g2_spart` in the compile call.

- [ ] **Step 6: Run the focused tests and verify GREEN**

```bash
/opt/venv/bin/python -m pytest op_tests/flydsl_tests/test_v2_gemm2_spart_opt.py -q
```

Expected: all tests PASS.

- [ ] **Step 7: Format and run static checks**

```bash
/opt/venv/bin/python -m black \
  aiter/ops/flydsl/kernels/mxmoe_dispatcher.py \
  aiter/ops/flydsl/kernels/mxmoe_gemm_v2.py \
  op_tests/flydsl_tests/test_v2_gemm2_spart_opt.py
/opt/venv/bin/python -m ruff check \
  aiter/ops/flydsl/kernels/mxmoe_dispatcher.py \
  aiter/ops/flydsl/kernels/mxmoe_gemm_v2.py \
  op_tests/flydsl_tests/test_v2_gemm2_spart_opt.py
/opt/venv/bin/python -m py_compile \
  aiter/ops/flydsl/kernels/mxmoe_dispatcher.py \
  aiter/ops/flydsl/kernels/mxmoe_gemm_v2.py \
  op_tests/flydsl_tests/test_v2_gemm2_spart_opt.py
git diff --check
```

Expected: no errors.

- [ ] **Step 8: Commit implementation and tests**

```bash
git add \
  aiter/ops/flydsl/kernels/mxmoe_dispatcher.py \
  aiter/ops/flydsl/kernels/mxmoe_gemm_v2.py \
  op_tests/flydsl_tests/test_v2_gemm2_spart_opt.py
git commit -m "perf: add spart optimization to v2 gemm turbo"
```

## Task 3: Run Adjacent Tests and Preserve Branch State

**Files:**
- Verify: `op_tests/flydsl_tests/test_v2_gemm2_spart_opt.py`
- Verify: `op_tests/flydsl_tests/test_flydsl_moe_a8w4.py`
- Preserve: `3rdparty/composable_kernel`

- [ ] **Step 1: Run focused and adjacent tests**

```bash
/opt/venv/bin/python -m pytest \
  op_tests/flydsl_tests/test_v2_gemm2_spart_opt.py \
  op_tests/flydsl_tests/test_flydsl_moe_a8w4.py -q
```

If the adjacent file requires unsupported runtime resources in the isolated worktree, report the exact collection/runtime limitation and retain the focused test as the blocking unit gate; do not edit the adjacent test.

- [ ] **Step 2: Verify only scoped tracked paths changed**

```bash
git diff --check
git status --short
git diff --stat HEAD~1..HEAD
git diff --submodule=short -- 3rdparty/composable_kernel
```

Expected: the implementation commit contains only the two kernel files and the new focused test. The pre-existing composable-kernel checkout difference is neither staged nor changed by this task.

## Task 4: Validate Correctness and Fresh IR/ISA for Off and On

**Files:**
- Read-only config: `/root/workspace/aiter/docs/fp8_retune_config/glm5_fp4_pathB_flydslv2_tuned_20260722_141743.csv`
- Generate outside repository: fresh `/tmp/v2_gemm_turbo_spart_0.*` and `/tmp/v2_gemm_turbo_spart_1.*` directories and logs.

- [ ] **Step 1: Run fixed-seed correctness with SPART-opt disabled**

From the isolated implementation worktree, run:

```bash
MXFP4_G2_SPART_OPT=0 \
PYTHONPATH="$PWD" \
AITER_CONFIG_FMOE=/root/workspace/aiter/docs/fp8_retune_config/glm5_fp4_pathB_flydslv2_tuned_20260722_141743.csv \
AITER_MOE_EXPERT_BALANCE=true \
/opt/venv/bin/python -c '
import random
import runpy
import sys
import numpy as np
import torch
random.seed(123)
np.random.seed(123)
torch.manual_seed(123)
sys.argv = [
    "op_tests/test_moe_2stage.py",
    "-q", "4", "-dim", "6144,512", "-e", "257", "-k", "9", "-t", "16",
    "--no-flydsl-csv",
]
runpy.run_path("op_tests/test_moe_2stage.py", run_name="__main__")
'
```

Expected: RC0, no NaN, no accuracy-threshold warning, no excessive-logits warning; record mismatch fraction, max delta, and `logits_diff`.

- [ ] **Step 2: Run fixed-seed correctness with SPART-opt enabled**

```bash
MXFP4_G2_SPART_OPT=1 \
PYTHONPATH="$PWD" \
AITER_CONFIG_FMOE=/root/workspace/aiter/docs/fp8_retune_config/glm5_fp4_pathB_flydslv2_tuned_20260722_141743.csv \
AITER_MOE_EXPERT_BALANCE=true \
/opt/venv/bin/python -c '
import random
import runpy
import sys
import numpy as np
import torch
random.seed(123)
np.random.seed(123)
torch.manual_seed(123)
sys.argv = [
    "op_tests/test_moe_2stage.py",
    "-q", "4", "-dim", "6144,512", "-e", "257", "-k", "9", "-t", "16",
    "--no-flydsl-csv",
]
runpy.run_path("op_tests/test_moe_2stage.py", run_name="__main__")
'
```

Expected: RC0, no NaN, no accuracy-threshold warning, no excessive-logits warning. Compare mismatch fraction, max delta, and `logits_diff` without claiming bitwise identity.

- [ ] **Step 3: Generate a fresh switch-off IR/ISA dump**

Create the output directory and run the dump with:

```bash
SPART0_DUMP_DIR=$(mktemp -d /tmp/v2_gemm_turbo_spart_0.XXXXXX)
MXFP4_G2_SPART_OPT=0 \
PYTHONPATH="$PWD" \
FLYDSL_RUNTIME_ENABLE_CACHE=0 \
FLYDSL_DUMP_IR=1 \
FLYDSL_DUMP_DIR="$SPART0_DUMP_DIR" \
AITER_CONFIG_FMOE=/root/workspace/aiter/docs/fp8_retune_config/glm5_fp4_pathB_flydslv2_tuned_20260722_141743.csv \
AITER_MOE_EXPERT_BALANCE=true \
/opt/venv/bin/python op_tests/test_moe_2stage.py \
  -q 4 -dim 6144,512 -e 257 -k 9 -t 16 --no-flydsl-csv --kernel
```

- [ ] **Step 4: Generate a fresh switch-on IR/ISA dump**

Create a second directory and run the enabled dump with:

```bash
SPART1_DUMP_DIR=$(mktemp -d /tmp/v2_gemm_turbo_spart_1.XXXXXX)
MXFP4_G2_SPART_OPT=1 \
PYTHONPATH="$PWD" \
FLYDSL_RUNTIME_ENABLE_CACHE=0 \
FLYDSL_DUMP_IR=1 \
FLYDSL_DUMP_DIR="$SPART1_DUMP_DIR" \
AITER_CONFIG_FMOE=/root/workspace/aiter/docs/fp8_retune_config/glm5_fp4_pathB_flydslv2_tuned_20260722_141743.csv \
AITER_MOE_EXPERT_BALANCE=true \
/opt/venv/bin/python op_tests/test_moe_2stage.py \
  -q 4 -dim 6144,512 -e 257 -k 9 -t 16 --no-flydsl-csv --kernel
```

- [ ] **Step 5: Compare exact GEMM2 artifacts**

Locate the `gemm2_a4w4_port_*` subdirectory in each dump and record:

- kernel symbol and `_spartopt` tag identity;
- origin and layout-lowered `floordivsi/divui/remsi/remui` counts;
- presence or absence of `unit_bx` flatten/redivision;
- final ISA total, SALU, VALU, branch, VMEM, LDS, wait, barrier, and MFMA counts;
- `.vgpr_count`, `.sgpr_count`, LDS, private segment, VGPR spill, and SGPR spill;
- grid/workgroup size and dynamic K/MFMA semantics.

Expected:

- switch-off retains the current signed path;
- switch-on uses unsigned/direct coordinates and removes targeted mapping overhead;
- both have identical MFMA, grid, LDS, output path, and zero spills/private-memory regressions.

## Task 5: Measure Performance and Select the Default

**Files:**
- Modify only if the measured default changes: `aiter/ops/flydsl/kernels/mxmoe_dispatcher.py`
- Read-only runner/config: `op_tests/test_moe_2stage.py`, absolute path-B config above.

- [ ] **Step 1: Run five interleaved rounds**

Create one fresh log directory and run the ten processes serially:

```bash
set -euo pipefail
SPART_PERF_LOG_DIR=$(mktemp -d /tmp/v2_gemm_turbo_spart_perf.XXXXXX)
for SPART_ROUND in 1 2 3 4 5; do
  MXFP4_G2_SPART_OPT=0 \
  PYTHONPATH="$PWD" \
  AITER_CONFIG_FMOE=/root/workspace/aiter/docs/fp8_retune_config/glm5_fp4_pathB_flydslv2_tuned_20260722_141743.csv \
  AITER_MOE_EXPERT_BALANCE=true \
  /opt/venv/bin/python op_tests/test_moe_2stage.py \
    -q 4 -dim 6144,512 -e 257 -k 9 -t 16 --no-flydsl-csv --kernel \
    >"$SPART_PERF_LOG_DIR/off_round${SPART_ROUND}.log" 2>&1

  MXFP4_G2_SPART_OPT=1 \
  PYTHONPATH="$PWD" \
  AITER_CONFIG_FMOE=/root/workspace/aiter/docs/fp8_retune_config/glm5_fp4_pathB_flydslv2_tuned_20260722_141743.csv \
  AITER_MOE_EXPERT_BALANCE=true \
  /opt/venv/bin/python op_tests/test_moe_2stage.py \
    -q 4 -dim 6144,512 -e 257 -k 9 -t 16 --no-flydsl-csv --kernel \
    >"$SPART_PERF_LOG_DIR/on_round${SPART_ROUND}.log" 2>&1
done
```

Check every process return code while executing. Do not run processes in parallel and do not rerun outliers.

- [ ] **Step 2: Calculate stable statistics**

For each path report five raw `us_stage2` values, sorted values, median, minimum, maximum, mean, absolute delta, and percent change `(on/off - 1) * 100`.

Extract `us_stage2` from the markdown summary in each log; do not use total two-stage `us` or ATT timing.

- [ ] **Step 3: Apply the deterministic default policy**

If the enabled median is no slower, keep both environment defaults at `"1"`. If it is slower, change both default literals to `"0"`:

```python
os.environ.get("MXFP4_G2_SPART_OPT", "1")
```

The two locations are the compile-function fallback and `get_g2` environment read. The permanent control, cache key, compile arg, and kernel tag remain regardless of default.

- [ ] **Step 4: Re-run final verification after any default change**

```bash
/opt/venv/bin/python -m black --check \
  aiter/ops/flydsl/kernels/mxmoe_dispatcher.py \
  aiter/ops/flydsl/kernels/mxmoe_gemm_v2.py \
  op_tests/flydsl_tests/test_v2_gemm2_spart_opt.py
/opt/venv/bin/python -m ruff check \
  aiter/ops/flydsl/kernels/mxmoe_dispatcher.py \
  aiter/ops/flydsl/kernels/mxmoe_gemm_v2.py \
  op_tests/flydsl_tests/test_v2_gemm2_spart_opt.py
/opt/venv/bin/python -m py_compile \
  aiter/ops/flydsl/kernels/mxmoe_dispatcher.py \
  aiter/ops/flydsl/kernels/mxmoe_gemm_v2.py \
  op_tests/flydsl_tests/test_v2_gemm2_spart_opt.py
/opt/venv/bin/python -m pytest op_tests/flydsl_tests/test_v2_gemm2_spart_opt.py -q
git diff --check
```

- [ ] **Step 5: Commit a measured default adjustment only when needed**

If the default changes:

```bash
git add aiter/ops/flydsl/kernels/mxmoe_dispatcher.py
git commit -m "perf: select measured spart optimization default"
```

Do not create an empty commit when the provisional default remains selected.

## Task 6: Final Review and Report

**Files:**
- Verify: all scoped implementation/test files
- Preserve: `3rdparty/composable_kernel` and unrelated files

- [ ] **Step 1: Verify final branch state**

```bash
git diff --check
git status --short
git log --oneline --decorate -5
git diff --stat 992407015..HEAD
git diff --submodule=short -- 3rdparty/composable_kernel
```

Expected: scoped implementation/test commits only; the pre-existing submodule checkout difference is not staged or changed by this feature.

- [ ] **Step 2: Produce the final comparison**

Report:

- exact commits and files changed;
- RED and GREEN test evidence;
- signed/unsigned mapping equivalence coverage;
- two fixed-seed numerical results;
- IR/ISA arithmetic and instruction deltas;
- resource/spill/grid/MFMA comparison;
- five raw timings per path and median/range;
- selected default and whether the performance delta is statistically established;
- explicit confirmation that B-split, `g2_kstages`, BN64/BK128, composable-kernel, CSVs, and unrelated files were not modified.
