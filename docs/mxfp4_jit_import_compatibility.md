# MXFP4 JIT import compatibility: reproduction and validation

## Root cause and scope

This is a compatibility fix for **CPython 3.10.9's source-block parser**, not a
claim that all Python 3.10 versions fail. It was found during RTP-LLM integration.

`inspect.BlockFinder` in 3.10.9 ends its decorator state at an inner closing
parenthesis, without tracking nesting depth. A subsequent inline `lambda` in
the decorator can then be mistaken for the source block being inspected. Source
extraction stops before the decorated function's `def`. Triton/Gluon JIT requires
that definition and raises `ValueError: No function definition found for kernel`.

This matches [CPython issue gh-83035](https://github.com/python/cpython/issues/83035)
and its [3.10 backport](https://github.com/python/cpython/pull/100080).
The tested 3.10.12 interpreter already has the corrected parser and does **not**
reproduce the original import failure. A multiline lambda alone is not a
sufficient description of the trigger: decorator parsing and nested parentheses
are the important details.

Moving the predicates out of the decorator avoids this parser behavior. Both
Triton and Gluon now use `utils/mxfp4_heuristics.py`; no Python or Triton monkey
patch is installed, and no GPU kernel body is changed.

## Exact reproducing environment

- Original container image:
  `hub.docker.alibaba-inc.com/isearch/rtp_llm_dev_rocm:2026_04_15_21_09_b045964`.
- Local Docker image ID:
  `sha256:f474cdb53ac01c65f3c46fa8a508fcf232bc2f53a8a40d0c8f580f6b853c3038`.
- Failing interpreter: `/opt/conda310/bin/python`, CPython **3.10.9**,
  GCC 11.2.0. RTP-LLM's Bazel configuration explicitly selects this interpreter.
- Control interpreter in the same container: `/usr/local/bin/python3`,
  CPython **3.10.12**.
- Same Triton distribution for both: **3.8.0+amd.rocm7.2.0.git111ff227**.
- AITER base: `fedccf0af` (parent of the initial fix `ef2323211`).
- Neither `sitecustomize` nor `usercustomize` was loaded in the package-import
  reproductions; `inspect.getsourcelines` remained the original callable before
  and after imports. Earlier diagnostic monkey patches were not used.

The following local `inspect.py` hashes match byte-for-byte the files downloaded
from the corresponding official CPython release tags:

| Interpreter | SHA256 of `inspect.py` |
| --- | --- |
| 3.10.9 | `ed67bf67157f52d3e942d4fcd381f427336def212b3989bab2cf7c89ed9b4c09` |
| 3.10.12 | `98cc184ae793fa1c45de2f28de2539f6d63bf7bec8338df580549f8263baa905` |

Sources: [3.10.9](https://raw.githubusercontent.com/python/cpython/v3.10.9/Lib/inspect.py),
[3.10.12](https://raw.githubusercontent.com/python/cpython/v3.10.12/Lib/inspect.py).

## Pure-standard-library reproduction

Save the following as `inspect_repro.py`. Run with `python -I -S inspect_repro.py`
to exclude site initialization, PYTHONPATH injection, AITER and Triton entirely:

```python
import inspect
from functools import partial


def heuristics(values):
    return lambda fn: fn


@heuristics({
    "first": lambda args: args["M"] % args["BM"] == 0
    and args["N1"] % (args["BN"]) == 0,
    "second": lambda args: args["M"] % args["BM"] == 0
    and args["N2"] % (args["BN"]) == 0,
})
def before():
    pass


def aligned(args, n):
    return args["M"] % args["BM"] == 0 and args[n] % args["BN"] == 0


@heuristics({"first": partial(aligned, n="N1"),
             "second": partial(aligned, n="N2")})
def after():
    pass


for fn in (before, after):
    print(fn.__name__, "def " + fn.__name__ in inspect.getsource(fn))
```

Expected: 3.10.9 prints `before False`, `after True`; 3.10.12 prints both `True`.
No custom `inspect` implementation is necessary to reproduce this failure.

## Base/head evidence

Both source revisions were imported in separate processes using the same
interpreter and Triton installation. The base was exported into a separate
directory, without changing the working branch.

| Interpreter | Base AITER imports | Fixed AITER imports |
| --- | --- | --- |
| CPython 3.10.9 | FAIL: original Gluon JIT `ValueError` | PASS: all five kernel sources contain their definitions |
| CPython 3.10.12 | PASS | PASS: all five kernel sources contain their definitions |

Run the updated regression suite from the selected source checkout:

```bash
env -u PYTHONPATH /opt/conda310/bin/python -m pytest op_tests/triton_tests/quant/test_mxfp4_jit_import.py -q
env -u PYTHONPATH /usr/local/bin/python3 -m pytest op_tests/triton_tests/quant/test_mxfp4_jit_import.py -q
```

The fixed source passes **60 tests** on each interpreter. Each of the five
kernels has an import test, a no-inline-lambda compatibility test, and ten
named, hardcoded alignment truth-table cases. The tests assert both modules
use the same predicate function, and cover the iterative kernel's
`BLOCK_SIZE_N1 * NUM_ITER` contract without recomputing expected answers.
The 60 tests comprise 5 public-wrapper import/source checks, 5 AST checks,
and 50 parameterized truth-table cases. The latter execute 100 predicate
calls in total (1 + 3 + 2 + 1 + 3 predicates per set of ten inputs), not
1,620 checks. Kernel objects are obtained through
`aiter.ops.triton.quant.fused_mxfp4_quant`; only the import-free AST checks
read internal source files directly, relative to the repository root.

Copying the import/AST tests onto the base gives:

- 3.10.9: 10 failures (5 package-import checks plus 5 AST compatibility checks).
- 3.10.12: 5 import checks pass; all 5 AST compatibility checks fail.

Thus modern-Python CI can reject reintroduction of the incompatible pattern,
even though that interpreter no longer exhibits native source truncation.
The AST check is a compatibility invariant, not an emulation of an older Python.

## Host overhead and validation limits

### Review-fix revalidation (2026-09-21)

After correcting the relocated test's repository-root path and routing imports
through the public wrapper, the commands above passed 60/60 tests on CPython
3.10.9 (3.57 s) and 3.10.12 (3.86 s) in `xjn_gdn_721`, as user
`xingjunna.xjn`. The current user installation reports Triton `3.8.0`; this
rerun is distinct from the AMD-fork base/head reproduction recorded above.
No new GPU performance or base/head reproduction run is claimed by this rerun.

GPU function bodies remain AST-identical to the base. This does **not** imply
zero Python launch overhead. A host-only measurement used the actual Triton
`Heuristics.run`, with a no-op downstream launch, on one pinned CPU: 9 alternating
base/head rounds, 100,000 calls per round, median microseconds per call.

| Kernel | Base | Fixed | Delta |
| --- | ---: | ---: | ---: |
| `_gluon_fused_rms_mxfp4_quant_kernel` | 3.429 | 3.754 | +0.325 |
| `_gluon_fused_reduce_rms_mxfp4_quant_kernel` | 5.184 | 6.099 | +0.915 |
| `_fused_rms_mxfp4_quant_kernel` | 4.320 | 4.909 | +0.589 |
| `_fused_reduce_act_mul_and_dynamic_mxfp4_quant_kernel` | 3.609 | 3.915 | +0.306 |
| `_fused_reduce_rms_mxfp4_quant_kernel` | 5.276 | 6.282 | +1.006 |

These are host microbenchmark measurements, not GPU latency or end-to-end
performance. No zero-overhead claim is made. The repository's
`bench_quant_mxfp4.py` benchmarks `dynamic_mxfp4_quant`, not these five fused
heuristic-decorated kernels, and was not used to claim their launch overhead.

No gfx1250 device execution or downstream model validation is claimed here.
Maintainers should enable `ci:all` (or `ci:sglang`, `ci:atom`, `ci:vllm`) and
review the downstream results before merge. Those labels/results are not
modified or claimed by this local validation.
