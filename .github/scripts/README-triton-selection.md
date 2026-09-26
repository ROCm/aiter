# Selecting Triton unit tests

`select_triton_tests.py` writes the test files affected by a change, one path per
line. The selection covers `op_tests/triton_tests/`, including tests for Gluon
kernels. It selects entire test files, so their existing parametrized cases run.

```bash
python .github/scripts/select_triton_tests.py \
  --source HEAD --target origin/main --output selected_triton_tests.list
```

The source/target form uses the merge-base diff. For a checked-out PR merge
commit, use `--merge-ref HEAD`; this compares the merge result to its first
parent, the base branch. The checkout must contain the source or merge result
being tested, because dependency discovery reads files in the checkout.
`--all` explicitly selects the full suite.

The output is passed to `split_tests.sh` for sharding. An empty selection means
there are no selected Triton unit tests; it must not turn into an unfiltered
`pytest` invocation. Selection reasons are printed to stderr and to the GitHub
step summary when `GITHUB_STEP_SUMMARY` is set.

## Kernel and test changes

The selector follows imports transitively from tests through wrappers, kernels,
and test helpers. If test A imports wrapper B, and B imports kernel C, a change
to C selects A. This also finds fused operations that reuse other kernels.
Tests that reuse another test's reference implementation or input generator
are dependencies too.

Same-name tests supplement the import graph, including the `test_compile_`
naming convention. Changed tests run themselves and their importing tests.
Deleted files are not scheduled, but their surviving importers are still
considered. The selector retains conservative folder/full-suite fallbacks when
it cannot establish a safe subset.

This is a module-level dependency graph. Imports of unused helpers, reference
implementations, or conditionally selected backends can select extra files.
Arbitrary runtime dispatch and constructed module names are not generally
recoverable from imports. Keep explicit import paths or the supported naming
convention for new tests, and add a selector regression when introducing a new
dispatch pattern. Tests with no discoverable source mapping accompany every
non-empty selection; the summary identifies them.

Recognized dynamic loaders (`import_module`, `__import__`, and
`spec_from_file_location`) also make their consuming tests accompany every
non-empty selection, even when those tests have other static imports. Literal
module names contribute edges, but cannot prove all runtime targets are known.
The graph covers the Triton source and test trees; new dispatch through an
external helper needs a regression demonstrating that its kernel dependencies
are still discoverable.

## Config changes

A config path has this layout:

```text
aiter/ops/triton/configs/<arch>/<backend>/<op>/<family>/<file>.json
```

Changing a config selects the entire op test folder and the tests that depend
on that op's source modules, including tests in other folders and paired compile
tests. The family name alone is insufficient: for example,
`rmsnorm_large_m_small_n` is consumed by `rmsnorm.py`, and preshuffled GEMM
families share source files with their other variants.

There are a few layout exceptions:

- Attention includes the separate `chunk_delta_attn` source and test category.
- MHC configs belong to the `fusions` source and test category.
- GMM uses root-level `gmm.py` and `test_gmm.py`, instead of an op folder.

Keep these category relationships in sync if the source or config layout
changes. Unknown layouts or configs with no test mapping fall back to the full
suite.

Config selection does not restrict by architecture or backend. Loaders can use
architecture fallbacks, and wrappers can support multiple backends. Running an
op's tests still does not guarantee that every changed tuning entry is read:
the runner must support the intended architecture/backend, and the test cases
must exercise the relevant shapes and dispatch route. Config validation and
coverage of new specialized shapes remain necessary alongside test selection.

## Conservative fallbacks and scope

Shared machinery, CI changes, unrecognized relevant source layouts, and
selection/diff failures use the full Triton suite. Markdown and `.gitkeep`
changes do not select tests; Triton benchmark changes alone do not select unit
tests.

Changes to other `aiter/` or `op_tests/` files and root package/test configuration
also run the full suite because those dependencies are outside the import graph.
CI path filters include these inputs. Main pushes, manual runs, and PRs carrying
`ci:triton-full` run the full suite. `ci:triton-300x` independently enables MI300X.

This selector does not replace other CI suites. Tests outside
`op_tests/triton_tests/`, including multi-GPU and tuning tests, retain their own
CI coverage. When extending the selector, verify both inclusion of an impacted
test and exclusion of an unrelated op; also cover deleted/renamed modules and
the full-suite fallback.

Run the selector's CPU-only regression suite without importing Torch or Triton:

```bash
python -m unittest discover -s .github/scripts/tests -v
```
