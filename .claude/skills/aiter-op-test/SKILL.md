---
name: aiter-op-test
description: Standard structure for aiter op_tests under op_tests/test_*.py — @benchmark + run_perftest candidate loop, a torch reference, a final markdown summary table, a __main__ guard so the module is importable, and faithful reproduction of the real model call (output buffer, layout, shapes). Use whenever writing, rewriting, or extending any aiter unit/perf test, or adding model-derived shapes (e.g. DeepSeek-V4) to an existing one.
argument-hint: [op name or op_tests/test_*.py file]
---

# aiter op_test standard

How every aiter op test in `op_tests/test_*.py` must be built. The canonical
reference in-tree is **`op_tests/test_quant.py`** — match its shape. A test is
both a **correctness check** (vs a torch reference) and a **perf sweep** that
ends in a **markdown summary table**.

Follow this whenever you create a new `test_*.py`, rewrite an old one, or add
shapes/candidates to an existing one.

## The hard rules

1. **Mirror `test_quant.py`.** Same imports, same decorator, same table-at-the-end
   flow. Don't invent a different structure.
2. **`@benchmark()` on the test fn.** It logs the function's call args (the shape
   params) as table columns automatically and merges the dict you `return`. So the
   test fn signature *is* the table's left-hand columns — name params accordingly.
3. **Candidates live in a dict; build `ret` in a loop.** Per candidate record raw
   `us`, plus **`TFLOPS` and `TB/s`**, plus `err` — `ret[f"{name} us"]`,
   `ret[f"{name} TFLOPS"]`, `ret[f"{name} TB/s"]`, `ret[f"{name} err"]`. Never
   hand-write ratio columns. (TFLOPS/TB-s section below.)
4. **torch is the reference only** — compute it, compare against it, but do **not**
   time it and do **not** put it in the table. (A pure-torch candidate is allowed
   only when torch *is* one of the kernels under test, e.g. `torch.einsum`.)
5. **Time with `run_perftest`, check with `checkAllclose`** — both, for every
   candidate. Compare in fp32 (`.to(dtypes.fp32)`).
6. **End with a markdown summary table — one per test function.** Sweep the shape
   lists with `itertools.product`, collect per-shape dicts into a `pd.DataFrame`,
   print via `aiter.logger.info("... :\n%s", df.to_markdown(index=False))`. A file
   with several test fns of different arg signatures emits **one table each** —
   never force-merge them (it scatters NaN columns). Mandatory — a test with no
   summary table is incomplete.
7. **`__main__` guard.** All argparse + the sweep loop go inside `main()`, called
   under `if __name__ == "__main__": main()`. The reference (`run_torch`) and the
   `@benchmark` test fn stay at module top level so other scripts can
   `import` them for combination testing.
8. **Standard argparse only.** Use `-d/--dtype`, `-b/--batch`, `-s/--mnk` plus
   *op-specific sweep axes* as needed (e.g. `--layout`, `--modes`, `--mtp`). Those
   are legitimate data lists. **Do not** add bespoke behavior-toggle flags
   (no `--dsv4`, no `--only-*`) — every flag is a list the sweep iterates.
9. **Run clean on every supported card.** Gate on `get_gfx()` in `main()` so the
   test passes on all supported archs; arch-unsupported ops/candidates are filtered
   out *before* launch. Prefer the kernel's arch-dispatching wrapper over a
   file-per-arch (full section below).

## Canonical template

```python
# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import argparse
import itertools

import aiter
import pandas as pd
import torch
import torch.nn.functional as F
from aiter import dtypes
from aiter.test_common import (
    benchmark,
    checkAllclose,
    run_perftest,
)
from aiter.jit.utils.chip_info import get_gfx  # "gfx942", "gfx950", "gfx1250", ...

torch.set_default_device("cuda")

SUPPORTED_GFX = ["gfx942", "gfx950"]  # every card this op is built/validated for


def run_torch(x, weight, dtype=dtypes.bf16):
    # Reference only: fp32 math, cast back. Not timed, not in the table.
    ...
    return out.to(dtype)


@benchmark()  # call args (b, m, n, k, dtype, ...) become the table's left columns
def test_op(b, m, n, k, dtype, layout):
    # build inputs/outputs in the layout the MODEL actually uses (see below);
    # ref = run_torch(...)
    candidates = {
        "triton": lambda: ...,        # the path the model really runs
        "torch_einsum": lambda: ...,  # optional torch kernel under test
    }
    if <kernel supported on this arch + config>:   # e.g. get_gfx() != "gfx1250"
        candidates["ck"] = lambda: ...   # else skip it (see rules below)

    flops = 2 * b * m * n * k                                   # roofline numerator
    nbytes = (b * m * k + b * n * k + b * m * n) * x.element_size()

    ret = {"gfx": get_gfx()}             # record the card in the table
    for name, fn in candidates.items():
        out, us = run_perftest(fn)
        err = checkAllclose(ref.to(dtypes.fp32), out.to(dtypes.fp32),
                            rtol=1e-2, atol=1e-2, msg=f"{name}: <op>")
        ret[f"{name} us"] = us
        ret[f"{name} TFLOPS"] = flops / us / 1e6
        ret[f"{name} TB/s"] = nbytes / us / 1e6
        ret[f"{name} err"] = err
    return ret


def main():
    # Whole-op arch gate goes HERE, not inside test_op: @benchmark always returns
    # the call-args dict, so an in-fn `return` still emits an args-only row.
    if get_gfx() not in SUPPORTED_GFX:
        aiter.logger.warning("<op> unsupported on %s; skipping", get_gfx())
        return

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter, description="config input of test"
    )
    parser.add_argument("-d", "--dtype", type=dtypes.str2Dtype, nargs="*", default="bf16,", ...)
    parser.add_argument("-b", "--batch", type=int, nargs="*", default=[...], ...)
    parser.add_argument("-s", "--mnk", type=dtypes.str2tuple, nargs="*", default=[...], ...)
    # add -l/--layout ONLY if the op has real layout variants
    args = parser.parse_args()

    for dtype in args.dtype:          # one table per outer-config element
        df = []
        for layout, b, (m, n, k) in itertools.product(   # sweep via itertools.product
            args.layout, args.batch, args.mnk
        ):
            df.append(test_op(b, m, n, k, dtype, layout))
        df = pd.DataFrame(df)
        aiter.logger.info("<op> summary (markdown):\n%s", df.to_markdown(index=False))


if __name__ == "__main__":
    main()
```

`op_tests/test_batched_gemm_bf16.py` is a complete worked instance of this
template (GEMM with triton/CK/einsum candidates, `bmn`/`mbn` layout, V4 shapes).

## Faithful to the real model call (do not test an idealized op)

The point of the test is the kernel **as the model invokes it**, not a clean
textbook version. Reproduce exactly:

- **Preallocated output buffers.** If the model passes `YQ=`/`out=` a buffer it
  allocated, allocate and pass the same — don't let the kernel allocate its own.
- **Real tensor layout, including non-contiguous views.** If the model feeds a
  transposed view (e.g. `o.transpose(0, 1)`), build the input as a transposed
  view of a contiguous tensor — not a fresh contiguous tensor of that shape.
- **Couple input and output layout.** They are linked in the model. If you sweep
  an output layout (`bmn` vs `mbn`), the *input* must follow: the `mbn` (model)
  case is a transposed view of `[m, b, k]` (physically `mbk`) **and** a transposed
  view of `[m, b, n]` — not contiguous `[b, m, k]`.
- **For a torch.einsum candidate, switch the layout by editing the subscript
  string**, not by transposing afterwards: `->sgr` is physically `[m,b,n]` (mbn),
  `->gsr` is physically `[b,m,n]` (bmn). Feed it the model's natural contiguous
  operand (e.g. `o` is contiguous `[s,g,d]`, so `x.transpose(0,1).contiguous()`).

A test that quietly uses contiguous inputs when the model uses a transposed view
gives the wrong perf **and hides correctness bugs** (see next rule).

## Report TFLOPS and TB/s, not just `us`

A bare `us` doesn't say whether a kernel is compute- or memory-bound. Always add
both roofline metrics per candidate, derived from the same `us`:

```python
flops = 2 * b * m * n * k                                   # GEMM: 2*M*N*K mul-add
nbytes = (b * m * k + b * n * k + b * m * n) * x.element_size()  # in + weight + out
ret[f"{name} TFLOPS"] = flops / us / 1e6   # us -> s is 1e-6, FLOP -> T is 1e-12
ret[f"{name} TB/s"]   = nbytes / us / 1e6
```

Count the FLOPs and bytes the *op* actually does (adjust the formula per op: a
quant/norm/attention kernel has its own element-traffic and arithmetic). Use
`tensor.element_size()` for dtype width so fp8/bf16/fp16 are handled. Reading the
table: small `m` (decode) is memory-bound (high TB/s, low TFLOPS); large `m` is
compute-bound (TFLOPS approaches peak).

## Multiple test functions → multiple tables; reference patterns

- **One `@benchmark` fn per distinct arg signature, one table each.** A file may
  hold several (e.g. a main bf16/fp8 sweep + an fp8 nm-asm cross-check with
  different columns). Give each its own `pd.DataFrame` + `aiter.logger.info(...)`
  via a tiny `summarize(name, rows)` helper. Forcing them into one table scatters
  NaN columns and is unreadable.
- **Two correctness shapes are common:**
  - *Multi-candidate* (this template): several kernels vs one torch reference, all
    timed, dict-loop.
  - *Single-kernel-vs-reference, in-place output* (e.g. cache-writing kernels):
    clone the output buffer, run the kernel into one copy and the reference into
    another, then `checkAllclose`. Still record `us`/TFLOPS/TB-s/err.
- **Prefer a shared reference from `aiter.ops.torch_ref`** when one exists; only
  hand-write `run_torch` when there is none.
- **Do all the asserts a case needs; record one representative `err`.** A quant
  path checks the dequantized output *and* bit-exact scales: use
  `tol_err_ratio=` for the fraction of allowed element mismatches (fp8/bf16
  rounding), and `rtol=0, atol=0` for values that must match to the bit (scales).

## Skip a candidate in configs it does not support

Some kernels are only correct for some layouts/dtypes. Running them anyway pollutes
the table with wrong-but-fast numbers. **Conditionally add** such a candidate and
leave its cells `nan` elsewhere — e.g. `batched_gemm_bf16_CK` returns garbage
(`err ≈ 0.99`) on a non-contiguous `mbk` input, so it is only added for `bmn`:

```python
if layout == "bmn":
    candidates["ck"] = lambda: aiter.batched_gemm_bf16_CK(x, weight)
```

When you skip something, say so in a code comment with the reason. The non-zero
`err` column is exactly how you discover these — never silently drop a candidate
because its error is high; first confirm whether it's a real bug or an unsupported
config, then skip with a comment.

## Run on every supported card (arch gating)

A test must run **clean on every currently-supported card** — today `gfx942`
(MI300) and `gfx950` (MI35x), plus any arch the kernel specifically targets.
Never assume one GPU. Detect arch at runtime and filter *before* launching:

```python
from aiter.jit.utils.chip_info import get_gfx   # "gfx942" / "gfx950" / "gfx1250" / ...
```

- **One test, all archs — drive the arch-dispatching wrapper; do NOT write a file
  per arch.** Most aiter kernels expose a public wrapper that routes to the
  wave64/wave32 (or gfx-specific) implementation internally by `get_gfx()` — call
  that wrapper and the single test covers every arch. Only the *behavioral* arch
  differences need handling in the test (e.g. gfx1250 uses a linear FP8 layout, so
  force `preshuffle=False` there; keep the kernel call and the reference in sync).
  Never import an `*_gfx1250`/arch-suffixed kernel directly to make a parallel
  test file.
- **Op not built/supported on this arch → skip in `main()`** with an allow-list
  early `return` (skips the whole sweep cleanly — no rows):
  ```python
  SUPPORTED_GFX = ["gfx942", "gfx950", "gfx1250"]
  if get_gfx() not in SUPPORTED_GFX:
      aiter.logger.warning("<op> unsupported on %s; skipping", get_gfx())
      return
  ```
  Do **not** gate by returning from inside the `@benchmark` fn: that wrapper always
  returns the call-args dict, so an in-fn `return` still emits an args-only NaN
  row. Prefer a **positive allow-list** (`not in [...]`) over a deny-list so an
  unknown new card doesn't silently run an unbuilt kernel and crash.
- **One candidate / one sub-check unsupported on this arch → drop just it**, with a
  warning naming arch + reason — e.g. the fp8 nm-asm cross-checks are wave64-only:
  ```python
  if get_gfx() != "gfx1250":
      summarize("hca_fp8", [test_hca_fp8(bs) for bs in args.fp8_bs])
  else:
      aiter.logger.warning("gfx1250: skipping wave64-only fp8 cross-checks")
  ```
- **Record the card**: put `"gfx": get_gfx()` in the returned dict so one table is
  self-describing across cards.

In-tree precedent: the flydsl `fused_compress_attn` wrappers dispatch wave64/wave32
internally (one `test_flydsl_compress_attn.py` covers all archs); `test_deepgemm.py`
/ `test_gemm_a4w4.py` allow-list a single arch; `test_gemm_a8w8.py` does per-arch
candidate/dtype/shape gating with warnings.

## Deriving model shapes (don't guess)

When adding "test op X for model Y" shapes:

1. **Read the real `config.json`** for the actual dims (`grep`/`python -json`),
   don't assume the dataclass defaults.
2. **Map model semantics → the kernel's `(b, m, n, k)`** and write the mapping in
   a comment. Worked example — DeepSeek-V4 grouped output LoRA
   (`atom/models/deepseek_v4.py`, `batched_gemm_bf16(o.transpose(0,1), wo_a, YQ=y)`):
   - `b` (batch) = `n_local_groups` = `o_groups // tp`
   - `m` = num_tokens (the swept dim)
   - `n` = `o_lora_rank`
   - `k` = `n_heads * head_dim // o_groups`
3. **Cover the real parallelism configs**, because they change `b`:
   - V4-Flash: `o_groups=8`, `tp8 → b=1`, `tp2 → b=4`
   - V4-Pro: `o_groups=16`, `tp8 → b=2`, **dp** (tp1 attn, full groups) `→ b=16`, `tp1 → b=16`
   Put the candidate `b` values in `-b` defaults and the `(m,n,k)` rows in `-s`;
   the sweep's cross product covers each config, identifiable by the `b`/`n`/`k`
   columns. Ask the user for the tp/dp set if it isn't given.

## Workflow when asked to write/extend a test

1. Make the edit (smallest change that follows the rules; for layout/equation
   tweaks, edit the one operand or subscript string in place).
2. `python3 -c "import ast; ast.parse(open(path).read())"` then a tiny subset run
   to confirm it executes and `err == 0`.
3. **Run the requested sweep and paste the markdown table verbatim** — the table
   *is* the deliverable. When the user asks for "the result", give the raw
   `df.to_markdown` block, not a re-summary.
4. Keep `import <module>` side-effect-free (the `__main__` guard) so the user can
   compose tests.

## Anti-patterns (all previously rejected)

- ❌ Custom behavior-toggle flags (`--dsv4`, `--only-*`) — every flag is a swept list.
- ❌ Hand-written ratio columns (`triton/ck`) — table holds raw `us`/TFLOPS/TB-s/err
  per candidate; compute ratios outside if needed.
- ❌ Reporting only `us` without TFLOPS and TB/s.
- ❌ torch reference timed into the table.
- ❌ Module-level argparse/sweep that runs on `import`.
- ❌ Gating arch by returning from inside `@benchmark` (emits an args-only NaN row);
  gate in `main()` instead.
- ❌ A separate `test_*_gfx<arch>.py` per arch when one dispatching wrapper covers
  all — merge into one test.
- ❌ Force-merging test fns with different arg signatures into one table (NaN scatter).
- ❌ Idealized contiguous inputs when the model uses a transposed view.
- ❌ Leaving a wrong-result kernel in the table instead of skipping its config.
- ❌ Hardcoding one arch / no `get_gfx()` gate — crashes on other supported cards.
- ❌ Deny-list arch checks that let an unknown new card run an unbuilt kernel.
- ❌ Deep manual nested loops for the sweep instead of `itertools.product`.
- ❌ No final summary table.
