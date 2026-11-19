## Summary

Adds two opt-in modes to the one-workgroup FlyDSL decode top-k:

- `finite_only` never selects a NaN, `+inf`, or `-inf` column. A row with fewer than `k` finite values returns those in column order, then `-1`.
- `compact` is a long-row path. After the first radix pass it keeps the candidates for the threshold in LDS and runs the remaining passes over them, instead of rereading the whole row for each pass.

It also adds an `arch` override to `flydsl_top_k_per_row_decode`, and `topk_select` passes the arch it already probes. Default behavior is unchanged. Nothing in this PR calls `compact` or `finite_only` through the public wrapper; the QSA indexer that needs them is a follow-up.

## Motivation

- By default NaN ranks above `+inf` in the radix key, so a NaN score is selected ahead of the largest finite score. Callers that need finite-only selection currently have to sanitize scores first.
- On a long row (more than 20000 columns), each radix pass rereads the full score buffer. When the first threshold leaves only a few thousand candidates, the remaining passes can run over those candidates alone.

## Changes

- `aiter/ops/flydsl/kernels/topk/topk_per_row_decode_persistent.py`: `build_topk_per_row_decode_one_workgroup_module` gains `compact` and `finite_only`, both default `False`. The full-row radix passes move into `finish_global`, and the new `finish_compact` and `finish_finite` helpers are added. The default build still runs the same selection, but its code is restructured and has not been benchmarked.
- `aiter/ops/flydsl/topk/topk_per_row.py`: `flydsl_top_k_per_row_decode` gains keyword-only `arch` and `finite_only`. `finite_only` with a row wider than 20000 columns raises `ValueError`.
- `aiter/ops/topk_select.py`: the decode backend passes `arch=_device_arch(...)`, so the device is not probed again on that call.
- `op_tests/test_topk_per_row.py`: two regressions, described under "Changed op tests".

## Performance (if applicable)

Not measured for this PR. The kernel's default path was restructured, and `compact` and `finite_only` have no benchmark here. The numbers in the QSA description are for a different branch and are not evidence for this one.

| Hardware | Workload/configuration | Before | After | Change |
|----------|------------------------|--------|-------|--------|
| gfx950 | Not measured | — | — | — |

## Testing

- [x] Unit tests added or updated
- [ ] Performance benchmarks run
- [ ] Tested on MI250X
- [ ] Tested on MI300X

Unchecked items are explained under "Validation and unverified scope".

## Documentation

- [x] Docstrings updated
- [ ] User guide updated
- [ ] Performance guide updated
- [ ] No documentation change is needed, with the reason stated below

## Dependencies

- [x] No new third-party dependencies added
- [ ] New dependencies are listed and justified below

## Breaking Changes

None. The new `flydsl_top_k_per_row_decode` argument is keyword-only, and both new builder arguments default to `False`.

## PR Self-Review

### Reviewed revision

`d4017b653c2d9a9a0399ddf7f3543c88b1851297`

### Behavior and intent

With `finite_only=False` and `compact=False`, selection is intended to match the previous kernel. Setting `finite_only=True` excludes NaN, `+inf`, and `-inf` columns, so a row with fewer than `k` finite values returns those in column order, then `-1`. Setting `compact=True` enables the long-row candidate path for rows over 20000 columns with between 1 and 4096 first-threshold candidates. The existing behavior is insufficient because NaN is selected by default and long rows reread the full buffer on every pass.

### Validation and unverified scope

Run on one gfx950 GPU through `gpuq`, using the `flydsl_venv` interpreter:

- `finite-only` regression (`test_flydsl_decode_finite_only_drops_nonfinite`): PASS.
- `compact` regression (`test_flydsl_decode_compact_long_row`): PASS.

Both regressions ran through a temporary wrapper that loads `op_tests/test_topk_per_row.py` up to its argparse block and calls one test function, which skips the sweep. The file's own command line (`--regressions compact -c 8 -k 512 -b 4 -n 1`) has not been run yet.

Lint: `ruff check` passes on the four changed files. `ruff format --check` reports three hunks in `op_tests/test_topk_per_row.py`, and `origin/main` has the same hunks, so they predate this PR and were left alone.

Not verified:

- The full `op_tests/test_topk_per_row.py` sweep and every other regression in that file.
- `op_tests/test_topk_select.py` and the other top-k tests.
- Performance of the restructured default path, and of `compact` and `finite_only`.
- MI250X and MI300X. The only GPU used here is gfx950.

### PR-specific question

No automated PR Self-Review comment has been posted for this branch yet, so there is no question to answer.

### Changed op tests

- `op_tests/test_topk_per_row.py`:
  - `finite-only` (`test_flydsl_decode_finite_only_drops_nonfinite`) checks that NaN and both infinities are never selected under `finite_only=True`, including short rows that need `-1` padding. It also checks that the default order still ranks a quiet NaN above `+inf`.
  - `compact` (`test_flydsl_decode_compact_long_row`) builds the one-workgroup kernel with `compact=True` and checks three rows: a tie that fits the candidate limit, a ranked row, and a tie one past the limit that must fall back to the full-row passes.

  Neither is redundant with the existing tests. The existing sweeps never pass `finite_only` and never build `compact=True`. The public wrapper does not set `compact`, so only a direct builder call can reach that path. The compact test imports `_COMPACT_LIMIT` and `_COMPACT_MIN_ROW` so its row sizes follow the kernel's own bounds.
