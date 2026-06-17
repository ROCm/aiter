# MHA ENABLE_CK Dispatch Test Report

## Summary

This report covers the MHA dispatch changes on `jim/dev/mha_fix`, tested on a
gfx942 machine, and provides follow-up validation plans for gfx950 and gfx1250.

The dense MHA path was changed so the unified C++ dispatcher owns the
ASM/CK decision:

- Forward: `mha_fwd` tries v3 ASM first, then falls back to CK when
  `ENABLE_CK=1`; with `ENABLE_CK=0`, unsupported ASM returns to Python and
  falls back to Triton.
- Backward: `mha_bwd` tries v3 ASM first, then falls back to CK when
  `ENABLE_CK=1`; with `ENABLE_CK=0`, Python avoids AITER backward when the
  v3 ASM backward coverage is not sufficient.
- Dense decode/small-Q policy is preserved in C++: `seqlen_q <= 128` skips v3
  ASM forward so `ENABLE_CK=1` uses CK and `ENABLE_CK=0` uses Triton fallback.

Varlen code paths were reverted to main behavior. They were tested only as a
regression check.

## Test Environment

- Machine architecture: `gfx942`
- Container: `jim_aiter_test`
- Repository branch: `jim/dev/mha_fix`
- Comparison branch: `origin/main`, checked out in `/home/jimguo12/aiter-main-test`
- JIT cache cleanup before test:

```bash
cd /workspace/aiter/aiter/jit
rm -rf build __pycache__ *.so
```

For the main worktree, `3rdparty/composable_kernel` had to be initialized before
`ENABLE_CK=1` tests could build CK kernels:

```bash
git submodule update --init --recursive 3rdparty/composable_kernel
```

## Code Paths Under Test

### Dense Forward

Expected behavior:

- `ENABLE_CK=0`
  - v3 ASM-supported shapes: use v3 ASM.
  - v3 ASM-unsupported shapes: return unsupported and fall back to Triton.
- `ENABLE_CK=1`
  - v3 ASM-supported shapes: use v3 ASM.
  - v3 ASM-unsupported shapes: fall back to CK inside `mha_fwd`.
- `seqlen_q <= 128`
  - skip v3 ASM by policy.
  - `ENABLE_CK=1`: CK path.
  - `ENABLE_CK=0`: Triton fallback.

### Dense Backward

Expected behavior:

- `ENABLE_CK=0`
  - use AITER backward only when the v3 ASM backward path is covered.
  - otherwise route the full autograd op to Triton up front.
- `ENABLE_CK=1`
  - use `mha_bwd`, which tries v3 ASM and falls back to CK.

### Varlen

Varlen code was restored to main behavior. Expected behavior matches main:

- `ENABLE_CK=0`: public varlen API uses Triton fallback.
- `ENABLE_CK=1`: CK/ASM varlen behavior follows main dispatch.

## gfx942 Test Results

### Dense MHA

Command:

```bash
for ck in 0 1; do
  ENABLE_CK=$ck PYTHONPATH=/workspace/aiter python3 op_tests/test_mha.py \
    -d bf16 \
    -d_qk_v 32,32 40,40 64,64 128,128 160,160 192,128 \
    -c true \
    -l false \
    -det false \
    -gr 1 8 \
    -i BSHD
done
```

Result on current branch:

- `ENABLE_CK=0`: pass
- `ENABLE_CK=1`: pass

Observed coverage:

- `d=64`, `d=32`, `d=40`, `d=160`: v3 forward unsupported on gfx942 dense path;
  `ENABLE_CK=0` falls back to Triton, `ENABLE_CK=1` falls back to CK.
- `d=128`: v3 forward and v3 backward ASM are exercised.
- `d=192,d_v=128`: v3 forward ASM is exercised; for `ENABLE_CK=0` training,
  backward coverage is not sufficient, so Python routes to Triton before forward.

### Dense Small-Q Policy

Command:

```bash
for ck in 0 1; do
  ENABLE_CK=$ck PYTHONPATH=/workspace/aiter python3 - <<'PY'
import torch
import aiter

for sq in (128, 512):
    q = torch.randn(2, sq, 16, 128, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(2, 512, 16, 128, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(2, 512, 16, 128, device="cuda", dtype=torch.bfloat16)
    out, lse = aiter.flash_attn_func(
        q, k, v, dropout_p=0.0, causal=True,
        deterministic=False, return_lse=True
    )
    torch.cuda.synchronize()
    print("sq", sq, tuple(out.shape), tuple(lse.shape))
PY
done
```

Result:

- `sq=128`: no v3 forward ASM load observed, confirming small-Q skips ASM.
- `sq=512`: v3 forward ASM load observed.
- Both `ENABLE_CK=0` and `ENABLE_CK=1` completed successfully.

### Varlen Regression Check

Direct varlen fwd/bwd command:

```bash
for ck in 0 1; do
  ENABLE_CK=$ck PYTHONPATH=/workspace/aiter python3 - <<'PY'
import torch
import aiter

B, SQ, SK, H, D = 2, 256, 256, 16, 128
cu_q = torch.arange(0, (B + 1) * SQ, SQ, device="cuda", dtype=torch.int32)
cu_k = torch.arange(0, (B + 1) * SK, SK, device="cuda", dtype=torch.int32)
q = torch.randn(B * SQ, H, D, device="cuda", dtype=torch.bfloat16, requires_grad=True)
k = torch.randn(B * SK, H, D, device="cuda", dtype=torch.bfloat16, requires_grad=True)
v = torch.randn(B * SK, H, D, device="cuda", dtype=torch.bfloat16, requires_grad=True)
out, lse = aiter.flash_attn_varlen_func(
    q, k, v, cu_q, cu_k, SQ, SK,
    dropout_p=0.0, causal=True, deterministic=False, return_lse=True
)
out.float().sum().backward()
torch.cuda.synchronize()
print(tuple(out.shape), tuple(lse.shape), tuple(q.grad.shape))
PY
done
```

Result on current branch:

- `ENABLE_CK=0`: pass
- `ENABLE_CK=1`: pass

`op_tests/test_mha_varlen.py` command:

```bash
ENABLE_CK=1 PYTHONPATH=/workspace/aiter python3 op_tests/test_mha_varlen.py \
  -b 2 \
  -nh 16 \
  -s 256,256 \
  -d_qk_v 128,128 \
  -dp 0.0 \
  -c true \
  -l false \
  -det false \
  -gr 1 \
  -dt bf16 \
  -i BSHD
```

Result:

- `ENABLE_CK=1`: pass
- `ENABLE_CK=0`: the main benchmark passes, but the script's automatic
  sequence-padding subtest fails. This is identical to `origin/main`.

Failure details for `ENABLE_CK=0` sequence-padding subtest:

```text
padding_scenario=mixed
out_diff=2.921875
tol=0.0625
AssertionError
```

The failing subtest is documented as validating CK group-mode padded-token
behavior. With `ENABLE_CK=0`, varlen follows main behavior and routes to Triton,
so this failure is not a regression from this change.

## Main Branch Comparison

The same tests were run in `/home/jimguo12/aiter-main-test` at `origin/main`.

| Test | Current branch | origin/main | Notes |
| --- | --- | --- | --- |
| Dense `test_mha.py`, `ENABLE_CK=0` | Pass | Pass | Same result |
| Dense `test_mha.py`, `ENABLE_CK=1` | Pass | Pass | Same result |
| Direct varlen fwd/bwd, `ENABLE_CK=0` | Pass | Pass | Same result |
| Direct varlen fwd/bwd, `ENABLE_CK=1` | Pass | Pass | Same result |
| Varlen `test_mha_varlen.py`, `ENABLE_CK=1` | Pass | Pass | Same result |
| Varlen `test_mha_varlen.py`, `ENABLE_CK=0` | Fails seq-padding subtest | Fails same subtest | Existing main behavior |

## gfx950 Test Plan

### Goals

- Validate that dense forward uses v3 ASM on supported gfx950 shapes.
- Validate that dense forward small-Q (`seqlen_q <= 128`) still prefers CK when
  `ENABLE_CK=1`.
- Validate `ENABLE_CK=0` unsupported fallback to Triton.
- Validate dense backward v3 ASM path, including gfx950-specific conversion
  behavior (`how_v3_bf16_cvt` forced to supported mode).

### Setup

```bash
cd /workspace/aiter
cd aiter/jit && rm -rf build __pycache__ *.so && cd -
python3 - <<'PY'
from aiter.jit.utils.chip_info import get_gfx
assert get_gfx() == "gfx950"
PY
```

### Dense Correctness

```bash
for ck in 0 1; do
  ENABLE_CK=$ck PYTHONPATH=/workspace/aiter python3 op_tests/test_mha.py \
    -d bf16 \
    -d_qk_v 64,64 128,128 192,128 \
    -c true \
    -l false \
    -det false \
    -gr 1 8 \
    -i BSHD
done
```

### Small-Q Policy

```bash
for ck in 0 1; do
  ENABLE_CK=$ck PYTHONPATH=/workspace/aiter python3 - <<'PY'
import torch
import aiter

for sq in (128, 512):
    q = torch.randn(2, sq, 16, 128, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(2, 512, 16, 128, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(2, 512, 16, 128, device="cuda", dtype=torch.bfloat16)
    out, lse = aiter.flash_attn_func(
        q, k, v, dropout_p=0.0, causal=True,
        deterministic=False, return_lse=True
    )
    torch.cuda.synchronize()
    print("sq", sq, tuple(out.shape), tuple(lse.shape))
PY
done
```

Expected:

- `sq=128`: no v3 forward ASM load.
- `sq=512`: v3 forward ASM load for supported shapes.

### Optional Varlen Regression

Run only as a regression check, since varlen code is intended to remain main
behavior in this change:

```bash
ENABLE_CK=1 PYTHONPATH=/workspace/aiter python3 op_tests/test_mha_varlen.py \
  -b 2 -nh 16 -s 256,256 -d_qk_v 128,128 \
  -dp 0.0 -c true -l false -det false -gr 1 -dt bf16 -i BSHD
```

## gfx1250 Test Plan

### Goals

- Validate that gfx1250-specific dense sink ASM behavior remains unchanged.
- Validate that the restored Python local helper
  `can_impl_fmha_fwd_with_sink_asm()` still routes supported gfx1250 cases to
  `fmha_fwd_with_sink_asm`.
- Validate that unsupported gfx1250 dense cases fall back through the new dense
  dispatcher logic.

### Setup

```bash
cd /workspace/aiter
cd aiter/jit && rm -rf build __pycache__ *.so && cd -
python3 - <<'PY'
from aiter.jit.utils.chip_info import get_gfx
assert get_gfx() == "gfx1250"
PY
```

### Dense Sink ASM

Run the existing sink ASM tests:

```bash
ENABLE_CK=0 PYTHONPATH=/workspace/aiter python3 op_tests/test_fmha_fwd_with_sink_asm.py
```

If runtime is too long, start with a narrow manual smoke test:

```bash
ENABLE_CK=0 PYTHONPATH=/workspace/aiter python3 - <<'PY'
import torch
import aiter

B, SQ, SK, H, D = 2, 512, 512, 16, 128
q = torch.randn(B, SQ, H, D, device="cuda", dtype=torch.bfloat16)
k = torch.randn(B, SK, H, D, device="cuda", dtype=torch.bfloat16)
v = torch.randn(B, SK, H, D, device="cuda", dtype=torch.bfloat16)
out, lse = aiter.flash_attn_func(
    q, k, v, dropout_p=0.0, causal=True,
    deterministic=False, return_lse=True
)
torch.cuda.synchronize()
print(tuple(out.shape), tuple(lse.shape))
PY
```

### Dense Fallback

Run unsupported-shape and CK-enabled fallback checks:

```bash
for ck in 0 1; do
  ENABLE_CK=$ck PYTHONPATH=/workspace/aiter python3 op_tests/test_mha.py \
    -d bf16 \
    -d_qk_v 64,64 128,128 \
    -c true \
    -l false \
    -det false \
    -gr 1 \
    -i BSHD
done
```

Expected:

- gfx1250 sink-specific supported cases should still take the dedicated
  `fmha_fwd_with_sink_asm` path.
- Non-sink or unsupported dense shapes should fall back according to
  `ENABLE_CK`.

## Notes And Risks

- The current machine is gfx942, so gfx950/gfx1250 sections are plans only.
- The gfx1250 sink path was intentionally restored to main structure and was not
  validated on gfx942.
- Varlen source files and public behavior were restored to main. Remaining
  changes are dense MHA dispatch and compile-flag cleanup.
