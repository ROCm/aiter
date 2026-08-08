---
name: bisect-perf-regression
description: >
 Binary-search the AITER commit that introduced a performance regression
 using `git bisect run`. Covers building a deterministic benchmark
 driver (isolated shape, warmed, L2-flushed via `num_rotate_args`),
 wrapping it in a pass/fail script with a user-supplied threshold,
 rebuilding AITER correctly between commits (JIT cache invalidation,
 `AITER_REBUILD=1`), skipping broken commits, and reporting the first
 bad commit with its diff.
 Use this skill when the user says "bisect perf", "find the regression
 commit", "git bisect for AITER", or reports "kernel X got slower after
 pulling main".
allowed-tools: Bash Read Grep Glob Write
---

# Bisect an AITER Performance Regression

Goal: find the exact AITER commit that caused a named op to get slower on
a specific shape / backend. Deliverable: a commit SHA + one-line diff
summary + a reproducer.

## Step 1: Lock down a reproducer

Before bisecting, confirm:

- **Which op** (`aiter.gemm_a8w8`, `aiter.fused_moe`, `aiter.flash_attn_*`, …).
- **Which backend** — HIP, CK, Triton, ASM. Force it if the op has an
  env-var selector (grep the op's wrapper to see).
- **Exact shape / dtype / q_type**.
- **Known-good commit** (baseline) and **known-bad commit** (usually `HEAD`).
  If the user only has "main", use the commit before the last known good
  release as `--good`.

Write a standalone benchmark driver. Keep it < 40 lines and deterministic:

```python
# /tmp/aiter_bench_repro.py
import os, sys, time, torch, aiter
from aiter.test_common import perftest

torch.manual_seed(0)

M, N, K = 4096, 4096, 4096
dtype = torch.bfloat16

@perftest(num_iters=50, num_warmup=20, num_rotate_args=4)
def run(a, b):
    return aiter.gemm_a16w16(a, b)

a = torch.randn(M, K, device="cuda", dtype=dtype)
b = torch.randn(N, K, device="cuda", dtype=dtype).t().contiguous()

_, t_us = run(a, b)
tflops = 2 * M * N * K / (t_us * 1e-6) / 1e12
print(f"TFLOPS={tflops:.2f} time_us={t_us:.2f}")
sys.exit(0)
```

Tips:

- `num_rotate_args=4` flushes L2 between iterations — critical for stable
  numbers.
- Warmup ≥ 10 iterations; the first launch includes JIT overhead.
- Pin the GPU clock if possible (`rocm-smi --setperflevel high`).
- Set `HIP_VISIBLE_DEVICES=0` so runs are on one GPU.

Run on `HEAD` and `HEAD~N` (a known-good) manually once to confirm the
regression is real and reproducible:

```bash
git checkout <known-bad>
AITER_REBUILD=1 python /tmp/aiter_bench_repro.py
git checkout <known-good>
AITER_REBUILD=1 python /tmp/aiter_bench_repro.py
```

Target a **20%+ delta**. Anything smaller is probably noise; tighten the
benchmark first.

## Step 2: Decide a pass/fail threshold

Pick a TFLOPS number clearly between good and bad:

```
good:  210 TFLOPS
bad:   145 TFLOPS
threshold: 185 TFLOPS  (anywhere in the gap; err toward good)
```

## Step 3: Write the bisect runner

`git bisect run` expects a script that exits `0` for good, `1` for bad,
`125` for "skip this commit" (broken / uncompilable).

```bash
cat > /tmp/aiter_bisect.sh <<'EOF'
#!/usr/bin/env bash
set -u
cd "$(git rev-parse --show-toplevel)"

# --- update submodules that AITER depends on (CK, etc.) ---
git submodule update --init --recursive 2>/dev/null || true

# --- clean JIT cache; AITER caches compiled .so per (arch, commit) ---
rm -rf aiter/jit/build 2>/dev/null || true

# --- rebuild ---
export AITER_REBUILD=1
export GPU_ARCHS=${GPU_ARCHS:-gfx942}
export MAX_JOBS=${MAX_JOBS:-$(nproc)}

# If the user installed with PREBUILD_KERNELS=1, reinstall in-place.
# Otherwise JIT-compile on first import.
if ! pip install -e . --no-build-isolation --quiet 2>/tmp/build.log; then
  echo "build failed on $(git rev-parse --short HEAD), skipping"
  cat /tmp/build.log | tail -20
  exit 125
fi

# --- run benchmark ---
out=$(python /tmp/aiter_bench_repro.py 2>&1) || {
  echo "benchmark crashed on $(git rev-parse --short HEAD), skipping"
  echo "$out" | tail -20
  exit 125
}

tflops=$(echo "$out" | sed -n 's/.*TFLOPS=\([0-9.]*\).*/\1/p')
echo "commit=$(git rev-parse --short HEAD) tflops=$tflops"

THRESHOLD=185   # <-- adjust per Step 2
awk -v t="$tflops" -v th="$THRESHOLD" 'BEGIN{ exit !(t>=th) }'
EOF
chmod +x /tmp/aiter_bisect.sh
```

Notes:

- Exit `125` (not `1`) for build / crash, so `git bisect` skips those
  commits instead of blaming them.
- The `THRESHOLD` is the only number that needs tuning between
  investigations.
- `rm -rf aiter/jit/build` is the single most important step — without
  it, `git bisect` keeps loading a stale `.so` and "finds" a regression
  at a random commit.

## Step 4: Run the bisect

```bash
git bisect start
git bisect good <known-good-sha>
git bisect bad  <known-bad-sha>    # usually HEAD
git bisect run /tmp/aiter_bisect.sh | tee /tmp/bisect.log
```

After it finishes, `git` will print the first bad commit. To inspect:

```bash
FIRST_BAD=$(git rev-parse BISECT_HEAD)
git show --stat $FIRST_BAD
git bisect reset
```

Save the log:

```bash
cp /tmp/bisect.log ./bisect_<op>_<arch>.log
```

## Step 5: Confirm & produce the report

Re-run the benchmark on the first-bad commit and its parent, to rule out
noise:

```bash
git checkout $FIRST_BAD^ && AITER_REBUILD=1 python /tmp/aiter_bench_repro.py
git checkout $FIRST_BAD  && AITER_REBUILD=1 python /tmp/aiter_bench_repro.py
```

If the delta holds, write the report:

```
### Perf regression: aiter.<op> on (M,N,K,dtype)=(...)

- Arch: gfx942
- Backend: CK   (forced via <env var> = <value>)
- Good: <SHA>  <TFLOPS_good> TFLOPS
- Bad:  <SHA>  <TFLOPS_bad>  TFLOPS   (-XX%)
- First bad commit: <SHA>   "<commit subject>"

#### Diff summary
<files changed, lines +/->

#### Reproducer
/tmp/aiter_bench_repro.py  (attached below)

#### Suggested follow-up
- Review `<files touched by first-bad commit>`.
- If CK tile: check whether `aiter/configs/tuned_gemm_*.csv` lost a row.
- If Triton: compare `get_gemm_config()` output for this shape on both commits.
- If JIT config: diff `aiter/jit/optCompilerConfig.json`.
```

## Common pitfalls

| Symptom | Cause / fix |
|---------|-------------|
| `git bisect` blames a commit that "can't" regress the kernel (e.g. a README change) | Stale JIT cache. Ensure `rm -rf aiter/jit/build` runs every iteration, and use `AITER_REBUILD=1`. |
| Many commits in the middle fail to build | Good — mark them `exit 125`. If > 50% fail, widen `good`/`bad` range to skip the broken era. |
| TFLOPS is noisy between runs on the same commit | Raise `num_iters` / `num_warmup`, pin clocks (`rocm-smi --setperflevel high`), reserve the GPU with `HIP_VISIBLE_DEVICES`. |
| Submodule commits don't match AITER commit | Add `git submodule update --init --recursive` at the top of the runner (already in the template). |
| First-bad commit is a tuning CSV change | Not a real regression, just missing a tuned shape. Fix by re-running the tuner for that shape; see `aiter-ck-tune` / `aiter-moe-tuning`. |
| First-bad commit is a `optCompilerConfig.json` edit | JIT module definition changed; diff the module entry and rebuild locally to confirm. |
| Kernel name changed across the range | `--kernel-include-regex` / env-var backend selector differs. Make the repro explicit: force the backend via env var, don't rely on defaults. |
| `PREBUILD_KERNELS=1` previously | Reinstall step can take 10+ min/commit. Switch to pure JIT for the bisect (`unset PREBUILD_KERNELS`; `pip install -e . --no-build-isolation` without prebuilds). |

## References
- `benchmark-aiter-op` skill — how to write the perftest driver.
- `aiter-jit-debug` skill — if the build fails on many commits.
- `aiter-ck-tune` / `aiter-moe-tuning` — if the regression turns out to be a missing tuned row.
- `git help bisect`, esp. `git bisect run` semantics.
