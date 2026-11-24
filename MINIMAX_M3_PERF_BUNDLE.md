# MiniMax-M3.1 perf bundle (internal testing branch)

This branch (`p1-integration`) is a convenience bundle of 3 independent perf PRs from the
MiniMax-M3.1 TP4/MI355X optimization project, cherry-picked onto a current `origin/main`
so a colleague can test all of them together without checking out 3 branches. **This is
not meant to be merged as-is** — it exists purely so the individual PRs (still open,
under normal review) can be exercised together. Merge status/history belongs on the PRs
themselves, not here.

## What's bundled

1. **[#8465](https://github.com/ROCm/aiter/pull/8465) — MiniMax-M3 prefill scorer + sparse attention: per-forward launch plans**
   Builds the launch plan once per forward instead of re-validating it every layer.
   Measured: scorer 43→9 us/call, sparse attention 16-18→7-8 us/call (host-side).

2. **[#8472](https://github.com/ROCm/aiter/pull/8472) — fused_qknorm_idxrqknorm: issue row/slot_mapping/norm-rope loads up front**
   Measured: device kernel time -1% to -2.7% (-6.6% at 512 tokens).

3. **[#8473](https://github.com/ROCm/aiter/pull/8473) — same kernel: nvfp4 quantizer via hardware fp8/fp4 converts (stacked on #8472), plus a NaN-handling fix**
   Measured: 4.6%-11.9% faster than main.

## Reproducing the numbers

Each PR's own description has the exact repro command / test plan for its own number —
use those rather than re-deriving a command here, since they're kept in sync with that
PR's own test file:
- `op_tests/test_flydsl_prefill_launch_plan.py` (PR #8465)
- `op_tests/test_fused_qknorm_nvfp4.py` (PRs #8472 / #8473 — covers both, #8473 stacks
  on #8472 in the same file)

## How this branch was built

Cherry-picked from each PR's own branch onto `origin/main` (no rebasing of the original
PR branches themselves):
- `af127aaeb` — from `ganyi/flydsl-prefill-launch-plan` (#8465)
- `6a3663ef9` — from `ganyi/fused-qknorm-idxr-early-loads` (#8472)
- `9e851f3c8` — from `ganyi/fused-qknorm-idxr-hw-convert` (#8473; its branch already
  contained #8472's commit, so only its own top commit was cherry-picked here to avoid
  double-applying)

All three applied with zero conflicts. Verified here: all changed Python files compile
(`python3 -I -m py_compile`); no local ROCm/torch install in this environment, so an
actual `import aiter` / GPU op-test run was **not** performed here — do that on a real
ROCm box before relying on this branch for anything beyond code review. The `.cu` file
has no leftover merge-conflict markers but was not compiled here (no local hipcc).
