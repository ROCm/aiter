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
`op_tests/flydsl_tests/` when `op_tests/test_<op>.py` already exists, a missing
or `get_gfx()`-based gate in `main()` (use `get_gfx_runtime()`), or a local
compare instead of `checkAllclose`. Flag a test that would still pass with the
change reverted: constant scales, a clamp it never triggers, a pre-zeroed output,
an unchecked `checkAllclose` result.

aiter-specific FlyDSL notes (where code goes, `tensor_shim` reuse, compiled-handle
caches) are in `aiter/ops/flydsl/AGENTS.md`.

## Before you report

Each item below is a pattern that produced wrong review comments on aiter FlyDSL
PRs. Check it before posting.

- Recompute LDS, buffer and offset sizes with every term and show the arithmetic.
  LDS capacity per arch comes from `chip_info.get_lds_capacity_bytes()` (gfx950
  is 160 KiB). Members of an `@fx.union` alias, so their LDS use is the maximum,
  not the sum.
- `gpu.barrier()` and `s_barrier` synchronize one workgroup. Report a
  cross-workgroup deadlock only when the code spin-waits on memory that another
  workgroup writes.
- Read the whole function, and the helpers it calls, before reporting a missing
  guard, import, barrier or stream. `mfma.call(..., set_prio=True)` emits an
  `s_barrier`.
- `const_expr` and `range_constexpr` resolve at trace time. A tail that
  predication or a compile-time branch covers is handled.
- Do not predict a lowering or compile failure on a path the PR's own tests run.
  Check FlyDSL API typing against the pinned version before reporting it.
- `flyc.compile(exe, *args)` with live device arguments compiles **and launches
  once**. Do not ask for an extra call of the returned handle on that path; that
  launches twice.
- A cudagraph-safe or metadata-only wrapper must not read device tensor contents
  on the host. Before proposing a tighter bound, name a legal call it would
  still accept: an empty shard, a zero-length row, the caller's real layout.
- Comment only on lines this PR changes. Skip byte-identical moves and code that a
  merge from main brought into the branch. On a config-only PR, review only the
  config.
- Post one comment per finding class. Do not repost an unanswered comment. Do not
  re-raise a finding the author rejected with evidence (ISA, measurement, a test
  that fails before the fix) unless you have new evidence. Since the PR
  self-review gate, every Copilot finding needs an author reply, so each extra
  comment costs the author time.

## Still check

These found real bugs in the same period:

- Raw-pointer launchers: every tensor passed as a pointer is on the launch device
  (not CPU), has the expected `stride(-1)`/`stride(0)`, alignment and length,
  and i32 offset products are widened before they can reach 2^31 bytes.
- A cache of compiled FlyDSL handles (`_run_compiled`, `FastJitLauncher`) keys
  everything that is not a runtime value: device, tensor rank and unit-stride
  axis, constexprs and mode globals. A handle reused across devices fails to
  launch, and the error is only printed.
- New tests are collected by CI (top-level `op_tests/test_*.py`, run as
  `python3 <file>`).
- New or changed tuned-config rows are compared against the rows they replace.
  A slower row or an ASM-to-FlyDSL switch needs a stated reason.
- A dispatch gate declines unsupported features (soft-cap, padded `cu_seqlens`,
  descales, fnuz dtypes on gfx942) instead of silently dropping them.
- One kernel backend per PR (`README.md`), with stacked PRs when one depends on
  another. Do not apply it to pure renames or one-line test or device fixes.

## Architecture helpers (do not invent a mixed-device finding)

Both helpers live in `aiter.jit.utils.chip_info`, and the context picks one:

- Code that asks which GPU it runs on uses `get_gfx_runtime()`, which detects
  the live GPU: runtime dispatch (wrappers, launchers, kernel or tuned-config
  selection), test gates in `main()`, and the card a test records.
- Build-time codegen and AOT use `get_gfx()`. It follows `GPU_ARCHS`, and when
  that lists several targets it returns the last one, which can differ from the
  installed GPU (`GPU_ARCHS='gfx950;gfx942'` gives `gfx942` on an MI355X).

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
- `get_gfx()` **added** to a runtime dispatch path or a test gate; ask for
  `get_gfx_runtime()`. Calls on unchanged lines are out of scope.
- A change to `chip_info.py` itself, when that file is in the diff.
- Allocating workspace or outputs with `device="cuda"` instead of `x.device`.
  That is the tensor's device **id**, not its ISA. Do not "fix" ISA detection
  by reading properties off that device.
