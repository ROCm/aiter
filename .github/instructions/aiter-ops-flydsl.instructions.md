---
applyTo: "aiter/ops/flydsl/**,aiter/ops/*.py,op_tests/test_flydsl*.py,op_tests/flydsl_tests/**,aiter/aot/flydsl/**"
---

# AITER FlyDSL / ops wrappers — PR review rules

Open and apply these in-repo skills. They are the FlyDSL review contract; this
file does not repeat them.

| Change | Read |
|---|---|
| New or changed `@flyc.kernel` / `@flyc.jit` body, layouts, copy/MMA, LDS | `.claude/skills/flydsl-kernel-authoring/SKILL.md` |
| Migration off legacy IR (`ArithValue`, `fx.Index`, `buffer_ops`, raw dialects, `SmemPtr`), helper reuse, `_run_compiled` | `.claude/skills/flydsl-kernel-code-cleanup/SKILL.md` |
| `op_tests/test_flydsl*.py` or any new FlyDSL test suite | `.claude/skills/aiter-op-test/SKILL.md` |

Judge the kernel against aiter's pinned FlyDSL in `requirements.txt`, not a
newer local FlyDSL checkout. Cleanup maps to that pin.

Flag a test that ignores aiter-op-test: second suite under
`op_tests/flydsl_tests/` when `op_tests/test_<op>.py` already exists, missing
`get_gfx()` gate in `main()`, or a local compare instead of `checkAllclose`.

## Architecture helpers (do not invent a mixed-device finding)

Runtime ISA for dispatch is `get_gfx_runtime()` from
`aiter.jit.utils.chip_info`. Tests and build-time codegen use `get_gfx()`.
AITER assumes a homogeneous node. These helpers are the shared contract;
almost every kernel already uses them.

Do **not** flag ordinary `get_gfx_runtime()` / `get_gfx()` (including a
process-level cache) as process-global, first-`rocminfo`-agent, or
mixed-device wrong-ISA. Do **not** ask this kernel for `gcnArchName`,
`torch.cuda.get_device_properties` for architecture, a per-tensor arch cache,
or a heterogeneous-host test. If the helper is wrong, fix `chip_info.py`, not
the launcher.

Still flag:

- This op's allow-list or fallback (wrong targets, silent default to another
  ISA, missing gate before a wave-size-specific path).
- ISA, atom, LDS, and launch attributes that do not match the architectures
  this kernel claims.
- A **new** local architecture probe that diverges from the shared helpers.
- A change to `chip_info.py` itself, when that file is in the diff.
- Allocating workspace or outputs with `device="cuda"` instead of `x.device`.
  That is the tensor's device **id**, not its ISA. Do not "fix" ISA detection
  by reading properties off that device.
