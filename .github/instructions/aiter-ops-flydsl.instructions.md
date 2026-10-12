---
applyTo: "aiter/ops/flydsl/**,aiter/ops/*.py,op_tests/**/*flydsl*.py,op_tests/flydsl_tests/**,op_tests/op_benchmarks/flydsl/**,aiter/aot/flydsl/**,aiter/jit/utils/chip_info.py"
---

# AITER FlyDSL / ops wrappers — PR review rules

Open and apply these in-repo skills. They are the FlyDSL review contract; this
file does not repeat them.

| Change | Read |
|---|---|
| New or changed `@flyc.kernel` / `@flyc.jit` body, layouts, copy/MMA, LDS | `.claude/skills/flydsl-kernel-authoring/SKILL.md` |
| Migration off legacy IR (`ArithValue`, `fx.Index`, `buffer_ops`, raw dialects, `SmemPtr`), helper reuse, `_run_compiled` | `.claude/skills/flydsl-kernel-code-cleanup/SKILL.md` |
| A FlyDSL test (`op_tests/**/*flydsl*.py`, `op_tests/flydsl_tests/**`) or any new FlyDSL test suite | `.claude/skills/aiter-op-test/SKILL.md` |

Judge the kernel against aiter's pinned FlyDSL in `requirements.txt`, not a
newer local FlyDSL checkout. Cleanup maps to that pin.

Flag a test that ignores aiter-op-test: second suite under
`op_tests/flydsl_tests/` when `op_tests/test_<op>.py` already exists, missing
`get_gfx()` gate in `main()`, or a local compare instead of `checkAllclose`.

## Architecture helpers (do not invent a mixed-device finding)

Both helpers live in `aiter.jit.utils.chip_info`, and the context picks one:

- Runtime dispatch (wrappers, launchers, kernel or tuned-config selection)
  uses `get_gfx_runtime()`. It detects the live GPU.
- Tests and build-time codegen use `get_gfx()`. It follows `GPU_ARCHS`, and
  when that lists several targets it returns the last one, which can differ
  from the installed GPU.

AITER assumes a homogeneous node. These helpers are the shared contract;
almost every kernel already uses them.

Do **not** flag either helper (including its process-level cache) as
process-global, first-`rocminfo`-agent, or mixed-device wrong-ISA.
Do **not** ask this kernel for `gcnArchName`,
`torch.cuda.get_device_properties` for architecture, a per-tensor arch cache,
or a heterogeneous-host test. If the helper is wrong, fix `chip_info.py`, not
the launcher.

Still flag:

- This op's allow-list or fallback (wrong targets, silent default to another
  ISA, missing gate before a wave-size-specific path).
- ISA, atom, LDS, and launch attributes that do not match the architectures
  this kernel claims.
- A **new** local architecture probe (`gcnArchName`, `get_device_properties`
  read for architecture, a `rocminfo` parse) instead of the shared helpers.
  It is a finding even when the rest of the PR uses the helpers.
- `get_gfx()` **added** to a runtime dispatch path; ask for
  `get_gfx_runtime()`. Calls on unchanged lines are out of scope.
- A change to `chip_info.py` itself, when that file is in the diff.
- Allocating workspace or outputs with `device="cuda"` instead of `x.device`.
  That is the tensor's device **id**, not its ISA. Do not "fix" ISA detection
  by reading properties off that device.
