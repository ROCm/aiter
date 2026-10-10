# Ticket 01 lifecycle implementation and evidence

Base: fixed main `882ea79dd7cc7091f859a5bedac914523ac77bbd`, entered through
integration tip `559e83fa7d750204a9164cdf97777c830fa760ff`. Tracker state is
maintained only in the integration worktree; this document records implementation
evidence and does not close a ticket.

## Behavior delivered

`--all` selection keeps input-only `gate_mode` while filtering device rows. The
winner CSV removes that metadata. Unsupported requested rows remain eligible for
failure reporting even when a valid tuned row shares their shape key. Returned
shape failures and empty candidate universes do not prevent later valid winners
from being written. Retuning replaces historical rows only with finite positive
validated winners; failed retuning preserves old valid rows.

Shared summary compares shape keys rather than all input columns and normalizes
string/enum representations. MXFP4 summary matches the requested shape across
the scheduler's assigned runtime identity, while the written winner retains that
assigned gfx/CU. Ordinary tuners retain their full device keys and timing/profile
semantics.

Shared CSV loading and sorting infer missing gfx per legacy row, preserving
explicit identities. The MXFP4 boundary filters tagged fallback rows before
dropping `_tag`, supplies numeric defaults, preserves ordinary fMoE's `nt`, and
projects to the production schema. Empty selections create no workers and do
not corrupt existing results or revive device-filtered rows.

MXFP4 prune/full validate their existing scheduled runtime GPU set before
tuning. Runtime `gcnArchName` and CU properties supply result identity;
`GPU_ARCHS`, global-agent identity and unused visible devices do not impose the
gfx950 restriction. Prune/full retain candidate order, mode-specific universe,
parallelism normalization and the existing direct/Queue supervision selection.

## Meaningful red → green slices

| Slice | Red observation | Repair and green behavior |
| --- | --- | --- |
| All-selection metadata | `KeyError: gate_mode` in measured row | Preserve explicit input metadata at selection; same/separate files pass |
| Same-file output | `gate_mode` leaked into winner CSV | Strip at dedicated output boundary |
| Unsupported already tuned | Command exited zero and skipped unsupported row | Keep unsupported requests before tuned dedup; failure plus later winner persists |
| Runtime preflight | Unsupported participating GPU reached worker launch | Validate participating logical properties before direct/worker work |
| Legacy shared CSV | Tagged row survived; missing gfx/defaults stayed invalid | Row-wise migration, tag filtering and numeric normalization |
| Mixed supported CU | Successful GPU1 row was reported unfinished | Match requested shape independently of scheduler-assigned gfx/CU |
| Ordinary summary | Failed enum-valued row also reported unfinished | Canonical string key comparison, ordinary schema retained |
| Invalid first latency | NaN first candidate blocked a later finite winner | Gate finite positive times before winner comparison, preserve attempt order |
| Runtime selection | Build/global gfx could mislabel the current GPU | Single-slot prune/full retain the caller current GPU; multi-slot retains first logical GPUs |

The tests inject controlled enumeration/measurement/scheduling at the seams
approved in the specification, exercise full input→CSV→sort→summary/exit behavior,
and assert persisted timing/schema and later-row behavior. The ordinary fMoE
suite enumerates real CK tasks and uses real post-processing/profiles/entry
points; see `ordinary-regression.md`.

## Environment and verification limits

Python: `/root/workspace/tmp_dev/mxfp4-port-env/bin/python`, Python 3.12,
torch `2.8.0+rocm7.1.0.git7a520360`, ROCm `7.1.25424-4179531dcd`, FlyDSL
`0.3.4.1`, einops `0.8.2`. Controlled final checks use target CK
`96afae1d54fc79c26857de7e1683b0dde1ff3c6f`. Normal full-module imports require
the ROCm/FlyDSL toolchain and may compile CPU enum bindings. The system Triton
`3.4.0+rocm7.1.0.gitf9e5bf54` is below the general Gluon minimum; controlled
checks use the repository's `AITER_USE_SYSTEM_TRITON=1` override and retain its
warning. These checks are not GPU numerical/deployment acceptance.

Final guarded verification passed **55 controlled lifecycle tests** (38 MXFP4,
17 ordinary fMoE) in **13.76 s** with 85 warnings, retained in
`isolated-lifecycle-final.txt`. The related existing infrastructure suite passed
**55 tests in 8.261 s**, retained in `isolated-shared-infra-green.txt`. Syntax,
Black, Ruff and diff whitespace checks passed. All final test commands used
empty ROCR/HIP/CUDA visibility, outside-repo JIT output and target CK. The
controlled fixtures forbid actual worker starts, GPU allocation/data and
measurement boundaries; explicit CPU dummy allocations for import/build custom
ops are allowed. The existing multiprocessing suite uses its CPU workers.

The final commands can be reproduced with:

```bash
export PATH=/root/workspace/tmp_dev/mxfp4-port-env/bin:$PATH
export ROCR_VISIBLE_DEVICES='' HIP_VISIBLE_DEVICES='' CUDA_VISIBLE_DEVICES=''
export GPU_ARCHS=gfx950 CU_NUM=256 AITER_USE_SYSTEM_TRITON=1
export CK_DIR=/root/workspace/tmp_dev/aiter-mxfp4-integration/3rdparty/composable_kernel
export AITER_JIT_DIR=/root/workspace/tmp_dev/mxfp4-port-notes/isolated-01-jit
python -m pytest -q op_tests/tuning_tests/test_mxfp4_tuner_lifecycle.py op_tests/tuning_tests/test_ordinary_tuner_lifecycle.py
python -m op_tests.tuning_tests.run_controlled_tuner_checks
```

Import-time discovery metadata is simulated only in the test bootstrap;
production identity/preflight functions still execute against each test's
controlled runtime properties. Discovery caches are cleared after import.

`isolated-lifecycle-checks.txt.gz` retains the red guard run (12 failures caused by
blocking explicit CPU dummy allocations); `isolated-shared-infra.txt.gz` and
`isolated-shared-infra-final.txt.gz` retain failed runner attempts. A stdin runner
could not be re-imported by CPU spawn workers, and the first file runner did not
bootstrap child imports. The retained file runner fixes both issues. These failed
attempts are excluded from the green counts. `controlled-checks.txt.gz` is the earlier
96-check intermediate run and does not establish final acceptance.

The early uncontrolled GPU execution incident and incomplete raw-log retention
are explicitly recorded in `incident.md`; no incident output is counted as
acceptance. All actual numerical, production, AOT and calibration acceptance is
owned by later tickets and the coordinator.

Long intermediate/failed logs are losslessly gzip-compressed; these files contain complete shell-captured output. The incident console excerpts remain partial because those two original tool outputs were not written to disk.
