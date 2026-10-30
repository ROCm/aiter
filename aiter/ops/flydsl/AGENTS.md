# Agent instructions: FlyDSL ops

This file is for coding agents, such as Claude Code, Codex and Copilot. It applies
to `aiter/ops/flydsl/`, `aiter/aot/flydsl/` and the FlyDSL op tests. It holds only
what is specific to aiter. The FlyDSL API itself is documented in FlyDSL's skills
(`flydsl-kernel-authoring`, `kernel-code-cleanup`, `api-stability`) at the
version below.

- **FlyDSL version.** Use the version `requirements.txt` pins, not a newer local
  FlyDSL checkout. Check the imported module with
  `python -c 'import flydsl; print(flydsl.__version__, flydsl.__file__)'`.
- **Where code goes.** `@flyc.kernel` code and shared kernel helpers live in
  `aiter/ops/flydsl/kernels/`; launch wrappers, compile helpers and public op
  entry points live in `aiter/ops/flydsl/*.py`; AOT prebuild lists live in
  `aiter/aot/flydsl/`.
- **Reuse `aiter.ops.flydsl.kernels.tensor_shim`.** Use its `_run_compiled` and
  `ptr_arg` for cached dispatch, pointer extraction and failed-compile cleanup.
  `moe_kernels._run_compiled(exe, args)` is a tuple-argument adapter over it.
  Do not add a second copy in a wrapper module.
- **Compiled-handle caches.** `flyc.compile(exe, *args)` with live device
  arguments compiles and launches once; it does not launch under `COMPILE_ONLY`
  or when any argument is a CPU tensor or placeholder. The returned handle skips
  argument checks: only data pointers, scalars and the stream may change between
  calls, its fast path does not fill in defaulted arguments, and a launch
  failure is printed, not raised. Any cache of handles (`_run_compiled`'s
  `specialization_key`, `FastJitLauncher`) must therefore key the device, each
  tensor's rank and unit-stride axis, constexprs and mode globals. Verified on
  FlyDSL 0.4.0 and 0.4.1, MI355X: a handle reused on a view without the
  unit-stride axis it was compiled for writes outside the view, and one reused on
  another device does not launch.
- **Legacy `buffer_ops`.** `aiter/ops/flydsl/kernels/buffer_ops.py` exists for
  older kernels. New code uses FlyDSL buffer tensors and `fx.copy`.
- **Raw ops.** A raw `rocdl`, `llvm`, `arith` or inline-asm call stays only with
  a comment at the call naming the stable API that is missing or slower, backed
  by ISA or benchmark evidence. Prefer typed `fx` operations (`.bitcast(...)`,
  operators, `fx.arith.cmpi`) over raw `arith.*`.
- **Architectures.** aiter targets gfx942, gfx950 and gfx1250; check ISA, MMA
  atoms and LDS sizes against each arch the kernel claims
  (`aiter.jit.utils.chip_info`). Use `get_gfx_runtime()` for dispatch and test
  gates, `get_gfx()` for build-time codegen and AOT.
- **Tests.** Follow `.claude/skills/aiter-op-test/SKILL.md`; extend the existing
  `op_tests/test_<op>.py`.
- **Review checks.** `.github/instructions/aiter-ops-flydsl.instructions.md`.
