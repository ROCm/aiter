# Test selection — supporting evidence

Generated from `dbbe0be70` (2026-09-15). Every figure below is produced by a script over the
tree at that commit; none are hand-entered. See *Reproducing* at the end.

This documents why [`select_tests.py`](select_tests.py) resolves changed files to tests
through a reversed import graph rather than by naming convention, call-graph analysis, or
developer-authored decorators — and what the resulting selection actually measures.

---

## 1. Naming convention alone is not enough

Matching each op module to `test_<name>.py` at the mirrored path, over 124 op modules
and 118 test files in `op_tests/triton_tests`:

| outcome | count |
| --- | ---: |
| exact mirrored-path match | 75 |
| basename matches, directory differs | 5 |
| no match at all | 44 |

Convention resolves 60% of op modules. The unmatched 44 include
both genuine helpers and ops whose tests are named differently:

```
  attention/fav3_sage_attention_mxfp4_wrapper.py
  attention/hstu_attention.py
  attention/lean_atten_paged.py
  attention/mha_fused_bwd.py
  attention/mha_onekernel_bwd.py
  attention/mla_decode.py
  attention/pa_mqa_logits.py
  attention/pod_attention.py
  attention/utils.py
  comms/all_gather.py
  comms/fused/reduce_scatter_rmsnorm_quant_all_gather.py
  comms/iris.py
  … and 32 more
```

Critically, this only covers the op-wrapper layer. The files PRs most often touch have no
test-name mirror at all:

| directory | .py files | non-.py files |
| --- | ---: | ---: |
| `aiter/ops/triton/_triton_kernels` | 169 | 0 |
| `aiter/ops/triton/_gluon_kernels` | 36 | 0 |
| `aiter/ops/triton/utils` | 44 | 2 |
| `aiter/ops/triton/configs` | 0 | 540 |

Only 8 test files import `_triton_kernels` directly. A convention or decorator scheme
keyed on op modules is structurally blind to the kernel and config files where most changes land.

## 2. Why not a static call graph

Import edges are dominated by shared utilities. Counting test files that import each module:

| module | test files reaching it |
| --- | ---: |
| `aiter.ops.triton.utils._triton.kernel_repr` | 101 |
| `aiter.ops.triton.utils.logger` | 98 |
| `aiter.ops.triton.utils._triton.arch_info` | 93 |
| `aiter.ops.triton.utils` | 88 |
| `aiter.ops.triton.utils.config_utils` | 76 |

For an **audit** these edges are noise and would need call-site analysis to filter. For **test
selection** they are correct: changing one of these genuinely should run everything that
depends on it. That difference is why plain import closure suffices and a full interprocedural
call graph buys nothing — selection wants the over-approximation.

Two things a call graph still cannot see:

- Ops registered dynamically into `torch.library.Library("aiter", ...)` by
  `torch_compile_guard` and reached via `torch.ops.aiter.<name>` — no AST edge exists.
- Runtime backend selection. `gemm_a16w16.py` gates on `_is_gluon_available()` against
  `_GLUON_SUPPORTED_ARCHS`, so which kernel actually runs is an arch-time decision.

## 3. Why not developer-authored decorators

Decorators record *declared intent*, which cannot be audited. The live counterexample is in
the tree today:

- `op_tests/triton_tests/torch_compile/test_compile_gemm_a16w16.py` is named for the op.
- It contains 0 references to `aiter` and none to `torch.ops`.
- It compiles `torch.mm` and compares against eager; it never calls the aiter op.

A decorator naming that file as the op's test would be wrong, would read as correct to any
reviewer, and nothing would catch it. For selection specifically the failure is worse than
mislabelling: a forgotten decorator yields a **silently skipped test**.

The graph also reaches tests that no per-op annotation would list, because tests import each
other for shared fixtures. Changing the gemm A16W16 kernel selects 7 tests; 4 are reachable
only via `test_gemm_a16w16`, which the others import `get_x_vals` from.

## 4. The graph

- **1,117** Python modules indexed across `aiter/` and `op_tests/`
- **3,416** import edges, reversed once at startup
- **118** test files in the triton suite, **304** across all of `op_tests/`
- Full triton suite estimated at **10,337s**
  (parsed from `FILE_TIMES` in `split_tests.sh`, so timings have one source of truth)

Fan-out for representative changed files:

| changed file | route | tests selected |
| --- | --- | ---: |
| `_triton_kernels/gemm/basic/gemm_a16w16.py` | `import-graph` | 7 |
| `gemm/basic/gemm_a16w16.py` | `import-graph` | 7 |
| `_triton_kernels/common/splitk_reduce.py` | `import-graph` | 58 |
| `utils/_triton/arch_info.py` | `import-graph` | 93 |
| `configs/gfx1250/gluon/moe/a4w4/DEFAULT.json` | `config-category:moe` | 14 |
| `configs/gfx950/triton/gemm/batched_gemm_a16w16/DEFAULT.json` | `config-category:gemm` | 62 |
| `pyproject.toml` | `**escalate** always-full` | 118 |
| `docs/tuning.md` | `ignored` | 0 |

## 5. Historical backtest

Replaying the last 120 commits that touched `aiter/ops/triton`, measuring selected GPU time as a
share of the full triton suite:

| metric | value |
| --- | ---: |
| commits replayed | 120 |
| median GPU time selected | 9.0% |
| mean GPU time selected | 36.0% |
| escalated to full suite | 14 (11%) |

Distribution of that share, one row per decile:

```
    0- 10%  ############################################################# 61
   10- 20%  ######                                   6
   20- 30%                                           0
   30- 40%  #                                        1
   40- 50%                                           0
   50- 60%  ###############                          15
   60- 70%  ####                                     4
   70- 80%  ###########                              11
   80- 90%  ####                                     4
   90-100%  ##################                       18
```

The distribution is **bimodal, not long-tailed**: 67 commits select under 20% of the suite and
52 select over 50%, with just 1 in the 20–50% valley between them. Commits are
either local (a kernel plus its wrapper) or hub changes that legitimately invalidate most of
the suite. Because the middle is nearly empty, there is little to gain from trying to narrow
the hub cases — the win is in the lower cluster, and it is already captured.

Escalation causes over the same window:

| cause | changed paths |
| --- | ---: |
| `unmapped-config` | 50 |
| `always-full` | 10 |

> Backtesting runs the **HEAD** import graph against historical file lists, so these are good
> estimates rather than exact replays. Run shadow mode before trusting them.

## 6. Safety properties

Selection is allowed to be too broad and never too narrow. These are checked exhaustively,
not sampled.

| property | result |
| --- | --- |
| a changed test file always selects itself | 118/118 |
| an op's selection ⊆ its kernel's selection | 330/330 |
| empty changeset selects nothing | 0 tests |
| unmappable path under `aiter/ops/` escalates | yes |

## 7. Kernel vs. op wrapper

Changing only the op wrapper works, and is the easier case: the wrapper sits closer to the
tests than the kernel does, and nothing reachable from the wrapper depends on the kernel below
it. The two are deliberately **not** symmetric for a shared kernel.

`_triton_kernels/common/splitk_reduce.py` is imported by 11 op wrappers:

| changed file | tests selected |
| --- | ---: |
| the kernel itself | **58** |
| `gemm/basic/gemm_a16w16.py` | 7 |
| `gemm/basic/gemm_a16w8_blockscale.py` | 1 |
| `gemm/basic/gemm_a16wfp4.py` | 3 |
| `gemm/basic/gemm_a8w8.py` | 40 |
| `gemm/basic/gemm_a8w8_blockscale.py` | 42 |

Touch the shared kernel and every dependent op's tests run. Touch one wrapper and only its
own tests run, because the others are genuinely unaffected. A symmetric answer here would
mean the edge direction was wrong.

## 8. Ops that select zero tests

The one case where selection silently runs nothing is an op with no test anywhere. These are
**pre-existing coverage gaps**, not selection bugs — they are equally untested today with the
full suite running — but selection makes them visible.

33 modules select no test in any suite. 22 are offline tuning harnesses
under `utils/_triton/tunning/` and are not on any test path. The remainder are worth a look:

```
  _triton_kernels/attention/pod_attention.py
  _triton_kernels/fusions/fused_routing_from_topk.py
  _triton_kernels/gated_delta_rule/fused_qkvzba_split.py
  attention/mla_decode.py
  attention/pod_attention.py
  fusions/fused_routing_from_topk.py
  gemm/basic/gemm_afp4wfp4_pre_quant_atomic.py
  gemm/batched/batched_gemm_afp4wfp4_pre_quant.py
  utils/_triton/gemm_tune_check.py
  utils/_triton/moe_common.py
  utils/moe_common.py
```

Suggested follow-up: gate CI so an op selecting zero tests needs either a test or an explicit
entry in a known-untested list, so the set can only shrink.

## 9. Guarded invariants

Both of these have already drifted once in this repo, and either would quietly collapse
selection toward the full suite, so `check_invariants()` warns on every run.

**Config path depth.** The rule reads `<op_name>` positionally from
`configs/<arch>/<backend>/<category>/<op_name>/<SHAPE>.json`, which is
9 components deep at HEAD. The tree has already migrated from a flat
layout once:
371 changed config paths in the last 300 commits are old-style and no longer resolve.
A second migration would silently shift which component holds the op name.

**Eager triton imports from the package root.** `aiter/__init__.py` currently imports:

```
  aiter.ops.triton.comms
  aiter.ops.triton.comms.all_gather
  aiter.ops.triton.comms.reduce_scatter
```

These are intentional (the Iris comms re-exports) and allowlisted in `ROOT_TRITON_ALLOWED`.
Any *new* eager triton import from the root would put every triton change upstream of every
test that does `import aiter`, and the tool flags it.

## 10. Prior art in this repo

`.github/scripts/select_triton_tests.py` (added 2026-01, #1682) already implements Triton
test selection over a `networkx` dependency graph. Two things to know about it:

**Its invocation in `triton-test.yaml` is commented out**, so no CI job runs it today.

**It no longer runs at all.** It resolves kernel config files under the pre-migration flat
layout `configs/gemm/`, which no longer exists, and aborts on every invocation:

```
  CRITICAL|Required directory [aiter/ops/triton/configs/gemm] doesn't exist.
```

It is a worked example of the exact failure mode §9 guards against: a positional
assumption about the config tree that broke silently when the layout moved, in a script
nothing was running often enough to notice.

One idea in it is better than the replacement and worth porting: it resolves config
references by parsing the f-string templates in op source and matching them against real
config paths, rather than reading `<op_name>` positionally from the path. That would remove
the `config-category` fallback and its over-selection (§4, the 62-test row).

Recommended: reconcile the two rather than run both — keep the reversed-graph core, port
the template-matching config rule, and delete the dead script.

## 11. Reproducing

```bash
# selection for the current branch
.github/scripts/select_tests.py --test-type triton --base origin/main --explain

# what CI runs: emit a list, then shard it as usual
.github/scripts/select_tests.py --test-type triton --base origin/main -o triton_selected.list
.github/scripts/split_tests.sh --shards 8 --test-type triton --select-from triton_selected.list

# shadow mode: emit the FULL suite, write the would-be selection to <out>.selected
.github/scripts/select_tests.py --test-type triton --base origin/main --shadow -o triton_selected.list
```

This document is generated, not maintained by hand. Regenerate it with the script that
produced it after any change to the selection rules.
