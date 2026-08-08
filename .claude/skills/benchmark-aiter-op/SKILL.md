---
name: benchmark-aiter-op
description: >
 Benchmark an AITER operator and compare it against a reference (torch) or an
 alternative backend (CK vs Triton vs ASM vs FlyDSL). Covers the two parallel
 AITER benchmarking idioms — `@perftest` from `aiter/test_common.py` (used
 inside `op_tests/test_*.py`) and `triton.testing.do_bench` (used by the
 dedicated drivers in `op_tests/op_benchmarks/triton/bench_*.py` and
 `op_tests/op_benchmarks/hip/`) — plus L2 cache rotation, correct warmup,
 and producing a roofline-style report (latency / TFLOPS / bandwidth / %
 of peak). Use when the user asks to benchmark an op, compare backends,
 generate a perf table for a PR, or reproduce a published number.
 Usage: /benchmark-aiter-op  [--dtype ] [--shape M,N,K]
allowed-tools: Bash Read Edit Grep Glob
---

# Benchmark an AITER Operator

AITER has two parallel benchmarking idioms — **pick the one that matches
the host file**, don't mix them:

| Use case | Idiom | Where it lives |
|----------|-------|----------------|
| Correctness tests with a quick perf number | `@perftest` | `op_tests/test_*.py` |
| Dedicated bench driver for a PR table | `triton.testing.do_bench` | `op_tests/op_benchmarks/triton/bench_*.py`, `op_tests/op_benchmarks/hip/` |
| Instruction-level hotspot analysis | `rocprofv3` ATT | See `capture-kernel-trace` / `kernel-trace-analysis` |

Every existing driver under `op_tests/op_benchmarks/triton/` uses
`triton.testing.do_bench`. **Do not introduce `@perftest` there** — it
breaks the shape / backend / json-export scaffolding the drivers share.

## 1. Pick the right entry point

| Goal | Command |
|------|---------|
| Quick latency of an op next to a correctness check | `python3 op_tests/test_<op>.py` |
| Full sweep for PR | `python3 op_tests/op_benchmarks/triton/bench_<op>.py` |
| Compare CK vs Triton vs ASM for the same shape | Small ad-hoc driver — see §4 |

List available bench drivers:

```bash
ls op_tests/op_benchmarks/triton/ | Select-String '^bench_'
ls op_tests/op_benchmarks/hip/
```

Existing drivers include `bench_gemm_a8w8.py`, `bench_gemm_a16w16.py`,
`bench_rmsnorm.py`, `bench_extend_attention.py`, `bench_fp8_mqa_logits.py`,
`bench_topk_topp_sampling.py`, and many more.

## 2. Idiom A — `@perftest` (inside `op_tests/test_*.py`)

`aiter/test_common.py` provides `@perftest(num_iters=101, num_warmup=2,
testGraph=False, num_rotate_args=0, needTrace=False)`. Decorated functions
return `(output, latency_us)` instead of just `output`.

```python
from aiter.test_common import perftest, checkAllclose

@perftest()
def run_aiter(x, w):
    return gemm_a8w8(x, w)

@perftest()
def run_torch(x, w):
    return torch.mm(x.to(torch.bfloat16), w.to(torch.bfloat16).T)

y_aiter, us_aiter = run_aiter(x, w)
y_ref, us_ref = run_torch(x, w)

checkAllclose(y_aiter, y_ref, rtol=1e-2, atol=1e-2)
print(f"aiter={us_aiter:7.1f} us   torch={us_ref:7.1f} us   speedup={us_ref/us_aiter:.2f}x")
```

Key `@perftest` options:

| Option | Default | Purpose |
|--------|---------|---------|
| `num_iters` | `101` | Total timed iterations |
| `num_warmup` | `2` | Warmup passes (not timed) |
| `testGraph` | `False` | Wrap in `torch.cuda.CUDAGraph` to remove host overhead |
| `num_rotate_args` | `0` | Rotate through N deep-copies of inputs each iter to defeat L2 cache. Use `>=3` for memory-bound ops. |
| `needTrace` | `False` | Attach `torch.profiler` context |

For memory-bound ops (normalization, quant, elementwise) `num_rotate_args=0`
(auto-computed based on free memory + L2 size) is usually correct. The
default logic picks enough rotations to flush L2 between iterations.

## 3. Idiom B — `triton.testing.do_bench` (inside `op_tests/op_benchmarks/.../bench_*.py`)

`do_bench` is what every driver under `op_tests/op_benchmarks/triton/`
actually uses (grep any file there to confirm). It handles warmup and
returns median ms per call.

```python
import torch
import triton
import triton.testing

from aiter.ops.triton.gemm.basic.gemm_a16w16 import gemm_a16w16

M = N = K = 4096
x = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
w = torch.randn(N, K, dtype=torch.bfloat16, device="cuda")

ms = triton.testing.do_bench(
    lambda: gemm_a16w16(x, w),
    warmup=25,   # ms of warmup (NOT number of iters)
    rep=100,     # ms of timed region
    return_mode="median",
)

flops = 2 * M * N * K
print(f"{ms*1e3:.1f} us   {flops / (ms*1e-3) / 1e12:.1f} TFLOPS")
```

Key flags: `warmup` and `rep` are in **milliseconds**, not iterations.
Typical values are `warmup=25, rep=100`. For a full worked example,
read `op_tests/op_benchmarks/triton/bench_gemm_a16w16.py`.

## 4. Structure of a bench driver

The existing drivers under `op_tests/op_benchmarks/triton/` follow this
pattern (built on `do_bench`, reusing shared argparse + json helpers in
`op_tests/op_benchmarks/triton/utils/`):

```python
# op_tests/op_benchmarks/triton/bench_my_op.py
import argparse, json, torch
import triton.testing
from aiter.ops.triton.category.my_op import my_op


def gen_inputs(M, N, K, dtype):
    x = torch.randn(M, K, dtype=dtype, device="cuda")
    w = torch.randn(N, K, dtype=dtype, device="cuda")
    return x, w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dtype", default="bf16")
    ap.add_argument("--shape", default="1024,1024,1024")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}[args.dtype]
    M, N, K = map(int, args.shape.split(","))
    x, w = gen_inputs(M, N, K, dtype)

    # Correctness first
    y_aiter = my_op(x, w)
    y_ref   = x @ w.T
    torch.testing.assert_close(y_aiter, y_ref, rtol=1e-2, atol=1e-2)

    ms_aiter = triton.testing.do_bench(lambda: my_op(x, w), warmup=25, rep=100,
                                       return_mode="median")
    ms_ref   = triton.testing.do_bench(lambda: x @ w.T,    warmup=25, rep=100,
                                       return_mode="median")

    flops   = 2 * M * N * K
    bytes_  = (M*K + N*K + M*N) * x.element_size()
    tflops  = flops / (ms_aiter * 1e-3) / 1e12
    bw      = bytes_ / (ms_aiter * 1e-3) / 1e9   # GB/s

    row = {"M": M, "N": N, "K": K, "dtype": args.dtype,
           "us_aiter": ms_aiter * 1e3, "us_ref": ms_ref * 1e3,
           "tflops": tflops, "bw_GBps": bw,
           "speedup": ms_ref / ms_aiter}
    print(json.dumps(row, indent=2))
    if args.json:
        with open(args.json, "a") as f: f.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
```

Run across a shape sweep:

```bash
for mnk in 1024,1024,1024 2048,2048,2048 4096,4096,4096 8192,8192,8192; do
  python3 op_tests/op_benchmarks/triton/bench_my_op.py --shape $mnk --json /tmp/results.jsonl
done
```

## 5. Comparing backends (CK vs Triton vs ASM vs torch)

Many AITER ops have multiple backends selected by a Python dispatcher. To
force a particular backend, call it directly:

```python
# CK:
from aiter.ops.gemm_op_a8w8 import gemm_a8w8 as gemm_a8w8_ck
# Triton:
from aiter.ops.triton.gemm.basic.gemm_a8w8 import gemm_a8w8 as gemm_a8w8_triton
# Torch reference:
def gemm_a8w8_ref(xq, wq, x_scale, w_scale, dtype):
    return (xq.to(torch.float32) * x_scale) @ (wq.to(torch.float32).T * w_scale).T.to(dtype)

@perftest(num_rotate_args=5)
def _ck(*a):    return gemm_a8w8_ck(*a)
@perftest(num_rotate_args=5)
def _tri(*a):   return gemm_a8w8_triton(*a)
@perftest(num_rotate_args=5)
def _torch(*a): return gemm_a8w8_ref(*a)

y_ck,   us_ck   = _ck(*inputs)
y_tri,  us_tri  = _tri(*inputs)
y_ref,  us_ref  = _torch(*inputs)
```

Report: pick whichever wins at a given shape; the winner often changes between
small-M and large-M regimes.

Inside a dedicated bench driver, swap the `@perftest` calls above for
`triton.testing.do_bench` calls (same lambdas).

## 6. Roofline numbers

For a PR table, report at minimum:

- **Latency** (us)
- **Throughput** — TFLOPS (compute-bound ops) or GB/s (memory-bound ops)
- **% of peak** — against the known hardware peak (see below)
- **Speedup** vs baseline

Hardware peak reference (per `CONTRIBUTE.md`):

| GPU | Arch | FP16 TFLOPS | BF16 TFLOPS | FP8 TFLOPS | HBM BW |
|-----|------|-------------|-------------|------------|--------|
| MI300X | gfx942 | ~380 | ~380 | ~770 | 5.3 TB/s (some SKUs 3.2 TB/s) |
| MI350X | gfx950 | ~1000 | ~1000 | ~2000 | 8.0 TB/s |

Arithmetic intensity threshold to decide compute- vs memory-bound:

```
AI_threshold = peak_TFLOPS / peak_BW
e.g. MI300X FP16: 380 / 5.3 = ~72 FLOP/byte
```

Anything below ~72 FLOP/byte is memory-bound → report GB/s utilization.
Above → report TFLOPS utilization.

## 7. PR-ready table format

Lifted from `CONTRIBUTE.md`:

```markdown
## Performance Results (MI300X, gfx942)

| Config                   | Baseline | Optimized | Speedup |
|--------------------------|----------|-----------|---------|
| BF16, M=1024 N=4096 K=7168 | 180 μs  | 150 μs    | 1.20x   |
| BF16, M=2048 N=8192 K=7168 | 720 μs  | 600 μs    | 1.20x   |

- Bandwidth utilization (Optimized): 78% of peak
- Arithmetic intensity: 0.83 FLOP/byte (memory-bound)
- Kernel: `_gemm_a16_w16_kernel_BLOCK_SIZE_M_64_BLOCK_SIZE_N_128_...` (from trace)
```

## 8. Typical gotchas

| Issue | Fix |
|-------|-----|
| First call is 10–100x slower | JIT / autotune compile. Use warmup iters, and ignore the first call in manual timers. `do_bench` already warms up by default. |
| Two consecutive benches give very different numbers | L2 pollution — for `@perftest` set `num_rotate_args=5` (or higher); for `do_bench` pass `grad_to_none=[...]` or rotate inputs manually. |
| Mixed `@perftest` + `do_bench` in one file | Don't. Dedicated bench drivers use `do_bench`; unit-test files use `@perftest`. Timing semantics differ (`warmup`/`rep` are iters vs ms). |
| Torch ref is faster than AITER at small M | Expected; AITER kernels are tuned for throughput, torch calls cuBLAS which has a highly tuned small-M path. |
| Numbers vary ±5% between runs | Normal GPU jitter. Run 3 times and report the median. |
| `torch.profiler` prints are empty | Need `needTrace=True` on `@perftest`, OR wrap manually in `torch.profiler.profile(...)`. |
| Bench driver crashes on first iter | Correctness bug; run the `test_.py` first to confirm. |
| Suspicious 2–3x speedup | Check that both paths produce matching output with `checkAllclose`. |

## 9. When to escalate to rocprofv3

If AITER is slower than expected **and** `checkAllclose` passes, collect an
ATT trace and hand off to `kernel-trace-analysis`:

```bash
# see capture-kernel-trace skill
rocprofv3 --stats --kernel-trace -f csv -- python3 op_tests/test_my_op.py -m 1024 -n 4096 -k 7168
```

Look at `*_kernel_stats.csv` to confirm the expected kernel actually ran
(and only once per iteration). A wrapper that falls back to a slow path is
easy to miss without this check.
