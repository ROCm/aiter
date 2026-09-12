---
name: debug-aiter-op
description: >
 Systematically debug a failing `op_tests/test_*.py`. Covers
 tolerance-vs-precision mismatches, NaN/Inf propagation in FP8/FP4 paths,
 stride / layout bugs, dtype-dispatch holes, wrong backend selection (CK vs
 Triton vs ASM), JIT/rebuild issues, and the AITER-specific debug toggles
 (`AITER_LOG_MORE`, `AITER_LOG_TUNED_CONFIG`, `AITER_LOG_LEVEL=DEBUG`,
 `AITER_REBUILD`). Use when an op_test crashes, produces wrong output, or
 passes on one dtype/shape but fails on another.
 Usage: /debug-aiter-op 
allowed-tools: Bash Read Edit Grep Glob
---

# Debug a Failing AITER Op Test

Before anything else:

```bash
rm -rf aiter/jit/build aiter/jit/module_.so /tmp/aiter_configs
AITER_REBUILD=1 AITER_LOG_MORE=1 AITER_LOG_LEVEL=DEBUG \
    python3 op_tests/test_.py 2>&1 | tee /tmp/aiter_debug.log
```

A stale JIT cache or stale merged tuning CSV is the #1 source of "my fix didn't
apply".

## 1. Classify the failure

| Symptom | Likely cause | Go to |
|---------|--------------|-------|
| Crashes before any output | JIT / import failure | §2 |
| `checkAllclose` mismatch at every element | Wrong backend wired up, transpose bug | §3 |
| Small relative error (<5%) in FP8/FP4 | Expected quantization noise, or wrong scale | §4 |
| NaN/Inf in output | Softmax / normalization issue, overflow | §5 |
| Passes at M=1024, fails at M=3 | Missing tiny-M config, edge-case masking | §6 |
| Passes on bf16, fails on fp16 | Dtype-dispatch table missing a branch | §7 |
| Passes once, fails on second call | Stateful buffer not zeroed, cache poisoning | §8 |
| Perf much worse than expected | Wrong tuned config selected | §9 |
| GPU hang / timeout | Infinite loop, bad sync, OOB memory | §10 |

## 2. Import / JIT failure

Run with the debug env and look for the first error:

```bash
AITER_REBUILD=1 AITER_LOG_MORE=1 python3 -c "import aiter; from aiter.ops. import *"
```

Jump to the `aiter-jit-debug` skill if you see:
- `ModuleNotFoundError: No module named 'module_xxx'`
- `undefined symbol: ...`
- `hipErrorNoBinaryForGpu`
- `fatal error: /path/to/header not found`

## 3. All-elements-wrong correctness failure

Checklist:

1. **Transpose**: many AITER GEMMs expect `w: (N, K)` internally transposed.
   Mixing up `(N, K)` vs `(K, N)` produces a permutation of the correct result.
   Read the wrapper's docstring; print `x.shape`, `w.shape`, and compare with
   the reference's expectation.

2. **Backend selection**: some ops have multiple backends picked by a dispatch
   function. For example, `aiter.ops.attention` can route to CK tile, ASM, or
   a fallback depending on `dtype`, `head_dim`, or `causal`. Print which path
   is taken:

   ```python
   import os; os.environ["AITER_LOG_LEVEL"] = "DEBUG"
   from aiter.ops.attention import ...   # triggers DEBUG logs
   ```

3. **Strides / contiguity**: AITER kernels usually require contiguous inputs.
   Check `x.is_contiguous()`; `.contiguous()` if necessary before calling.

4. **Data types**: the test's reference may run in FP32 while the kernel runs
   in BF16; widen the tolerance (`rtol=1e-2, atol=1e-2`) or cast the reference
   to the same dtype before comparing.

5. **All-ones isolation test**: fill every input with `1.0` and run both sides.
   If reference and kernel both produce the same constant but different from
   expected, your reference has a bug; if only the kernel is off, it's an
   addressing / layout bug:

   ```python
   x.fill_(1.0); w.fill_(1.0)
   ```

## 4. Small relative error (1–5 %) in FP8 / FP4

Common and usually expected:

- **FP8 PV in attention**: ~0.03 max error vs BF16 is inherent to the FP8 data
  path. Widen `atol=5e-3`.
- **Per-tensor vs per-row quant**: if the reference uses per-row quant and the
  kernel uses per-tensor (or vice versa), expect 1–3% mismatch. Verify the
  quant mode matches.
- **Scale factor mismatch**: common bug — applying `v_scale` twice (once in
  prob scaling, once in PV output). Print `scale_q`, `scale_k`, `scale_v`
  before the call and compare.
- **FP4 `mxfp4`**: group-size mismatches between reference and kernel cause
  block-local differences. `group_size=32` is the common default; verify.

## 5. NaN / Inf

### Softmax `-inf - (-inf) = NaN`

If **all** attention keys are masked (out of context), `qk_max = -inf` and
`exp(s - qk_max) = exp(-inf - (-inf)) = exp(NaN) = NaN`. Check the ref side
has the same guard.

### Division by zero in normalization

`rms_norm` divides by `sqrt(mean + eps)`. If `eps=0` or the input is all zeros
the output is NaN. Default eps in AITER is `1e-6` — if you're calling with `0`
you'll hit this.

### Host-side sanity check before the assert

```python
torch.cuda.synchronize()
print("nan%:", float(got.isnan().float().mean()))
print("inf%:", float(got.isinf().float().mean()))
print("range:", float(got.min()), float(got.max()))
```

## 6. Edge cases (tiny or non-power-of-2 shapes)

Triton kernels: check the config JSON has an `M_LEQ_4` or `M_LEQ_8` entry,
otherwise the `any` fallback is used with block sizes that may be wrong for
tiny shapes. See `aiter-triton-kernel` skill §2.3.

CK kernels: `errRatio` tolerance in the tuner may have excluded some instances
that worked for your small shape. Rerun the tuner with `--errRatio 0.1` just
for that shape and inspect the winner.

Also check for:
- **Implicit padding**: some kernels pad M/N/K to a multiple of the block size
  and trust that your output tensor is big enough. If `out.shape` isn't
  padded you'll get OOB stores.
- **Head-dim alignment**: attention kernels often require `head_dim %
  {64,128} == 0`. Fall back to Triton / torch for the remainder.

## 7. Dtype dispatch holes

```cpp
AT_DISPATCH_FLOATING_TYPES_AND2(
    at::ScalarType::Half, at::ScalarType::BFloat16,
    input.scalar_type(), "my_op", [&] { ... });
```

Missing a branch (e.g. omitting BF16 in an `AT_DISPATCH_FLOATING_TYPES`)
manifests as `RuntimeError: "my_op" not implemented for 'BFloat16'`.
Fix: use `AT_DISPATCH_FLOATING_TYPES_AND2` (or `AND3` including FP8).

For FP8: `at::ScalarType::Float8_e4m3fnuz` on ROCm ≤7.0, `Float8_e4m3fn` on
ROCm 7.x. Check `torch.version.hip` and match.

## 8. Stateful buffers, cache poisoning

- **AITER tuning CSV**: if `AITER_CONFIG_GEMM_A8W8` points to a hand-edited
  CSV with a bad row, every subsequent call picks the bad kernel. Restore the
  default: `unset AITER_CONFIG_GEMM_A8W8` and `rm -rf /tmp/aiter_configs`.
- **Ragged / Paged attention**: KV-cache tensors must be zeroed before reuse
  across test cases.
- **Triton autotune cache**: `~/.triton/cache/`. Wipe it when you change the
  kernel body or Triton version.
- **FlyDSL cache**: `~/.flydsl/cache/`.

## 9. Perf much worse than expected

```bash
AITER_LOG_TUNED_CONFIG=1 AITER_LOG_MORE=1 python3 op_tests/test_my_op.py
```

Look for `config file not found` / `using default config` / `kernelId=-1`
messages. Then:

1. Check `/tmp/aiter_configs/*_tuned_.csv` — is your shape there?
2. Rebuild: `AITER_REBUILD=1 rm aiter/jit/module_*.so && python3 ...`
3. If the shape is missing, add it to the untuned CSV and re-run the tuner
   (`aiter-ck-tune` skill).
4. For Triton ops: verify the right `{arch}-NAME.json` was loaded — hard-code
   a `config={...}` dict to bisect.

## 10. GPU hang

Signs: test hangs with no CPU activity; `rocm-smi` shows one CU at 100%.

Causes and fixes:

- **`scf.for` with `stop < start`**: unsigned comparisons on 32-bit indices
  overflow. Print the bounds before the launch.
- **Divergent barrier**: `gpu.barrier()` / `__syncthreads()` with a divergent
  predicate deadlocks. Make sure ALL threads in the workgroup reach the
  barrier.
- **OOB memory access on buffer_load/store**: the kernel may silently write
  to random memory until it hits a page fault; can look like a hang.
  Re-run with `HIP_DEVICE_LIB_PATH=... HSA_NO_SCRATCH_RECLAIM=1` (as
  documented in ROCm debug docs) to enable pagefault detection.
- **Recovery**: `sudo rocm-smi --gpureset -d 0` or reboot. In Docker,
  `docker restart ` is usually enough.

## Diagnostic workflow summary

```
1. rm -rf aiter/jit/build aiter/jit/*.so /tmp/aiter_configs
2. AITER_LOG_MORE=1 AITER_LOG_LEVEL=DEBUG AITER_REBUILD=1 AITER_LOG_TUNED_CONFIG=1 \
       python3 op_tests/test_.py 2>&1 | tee /tmp/dbg.log
3. Classify with §1 and jump to the matching section.
4. All-ones isolation test → layout bugs.
5. NaN / Inf count on the output → softmax / normalization bugs.
6. Narrow to a single (dtype, M, N, K) tuple that reproduces.
7. Bisect backend by forcing a fallback (e.g. call the torch reference
   directly from the wrapper to confirm the test driver itself is fine).
```

## Useful env toggles

| Env var | Effect |
|---------|--------|
| `AITER_LOG_MORE=1` | Verbose JIT + wrapper logging. |
| `AITER_LOG_LEVEL=DEBUG` | Everything. `WARNING`/`ERROR` also valid. |
| `AITER_LOG_TUNED_CONFIG=1` | Print which tuned config & kernelId was used. |
| `AITER_REBUILD=1` | Re-JIT every `module_*` on next touch. |
| `AITER_CONFIG_GEMM_A8W8=/path/custom.csv` | Override merged tuned CSV (see `aiter/jit/core.py`). |
| `GPU_ARCHS=gfx942;gfx950` | Limit archs compiled. |
| `TORCH_SHOW_CPP_STACKTRACES=1` | C++ backtraces for CHECK failures. |

## Common pitfalls checklist

- [ ] Cache cleared (`rm -rf aiter/jit/build aiter/jit/*.so /tmp/aiter_configs`).
- [ ] Input is contiguous (`x.is_contiguous()`).
- [ ] Input dtype is supported (check the AT_DISPATCH list).
- [ ] Input shape aligned to the kernel's block / head-dim requirements.
- [ ] For Triton: config JSON has `M_LEQ_*` entries covering small shapes.
- [ ] For CK: tuned CSV covers this `(cu_num, M, N, K)`.
- [ ] Softmax guards against `-inf - (-inf)` and division-by-zero.
- [ ] All-ones isolation test passes (rules out layout bugs).
- [ ] Reference implementation uses the same quantization mode / scales.
- [ ] Tolerance is realistic for the dtype (BF16: 1e-2, FP8: 5e-3).
