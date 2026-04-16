# AITER Claude Skills

This folder contains a set of [Claude skills](https://docs.anthropic.com/en/docs/claude-code/skills)
tailored for working on [AITER](https://github.com/ROCm/aiter) — AMD's centralized
repository of high-performance AI operators on ROCm.

These skills encode AITER-specific conventions (JIT compilation, `optCompilerConfig.json`,
tuning CSVs, the HIP / CK / Triton / FlyDSL / ASM backend split, `op_tests/` layout,
pre-commit formatting, …) so an agent can plug into the codebase without having to
re-discover them every session.

The structure mirrors `ROCm/FlyDSL/.claude/skills` and the same skills can also be
dropped into `.cursor/skills/` for Cursor Agent users.

## Skills at a glance

| Skill | Purpose |
|-------|---------|
| [`build-aiter`](./build-aiter/SKILL.md) | Install / build AITER (`setup.py develop`, `PREBUILD_KERNELS`, `GPU_ARCHS`). |
| [`aiter-add-operator`](./aiter-add-operator/SKILL.md) | End-to-end recipe for adding a new op: C++/HIP kernel → pybind → `optCompilerConfig.json` → Python wrapper → `op_tests/` test. |
| [`aiter-jit-debug`](./aiter-jit-debug/SKILL.md) | Diagnose JIT compile / module-load failures (missing `.so`, hipify issues, `AITER_REBUILD`, cache at `aiter/jit/build`). |
| [`aiter-triton-kernel`](./aiter-triton-kernel/SKILL.md) | Author Triton kernels following the `aiter/ops/triton/` layout, `get_gemm_config`, `make_kernel_repr`, and arch-string conventions. |
| [`aiter-ck-tune`](./aiter-ck-tune/SKILL.md) | Tune & integrate CK-tile kernels (GEMM A8W8, FMOE, etc.) via `gemm_a*_tune.py` and the `untuned_*.csv` / `tuned_*.csv` flow. |
| [`debug-aiter-op`](./debug-aiter-op/SKILL.md) | Systematically debug a failing `op_tests/test_*.py` (correctness, NaN, dtype, stride, backend selection). |
| [`benchmark-aiter-op`](./benchmark-aiter-op/SKILL.md) | Benchmark an op with `@perftest`, compare AITER vs reference vs alternative backends, report roofline numbers. |
| [`aiter-moe-tuning`](./aiter-moe-tuning/SKILL.md) | Tune the fused-MoE op (`tuned_fmoe.csv`, `untuned_fmoe.csv`) and per-model configs under `aiter/configs/model_configs/`. |
| [`format-code`](./format-code/SKILL.md) | Run AITER's pre-commit-compatible formatting: `black==26.3.0`, `ruff==0.15.7`, `clang-format-18`, plus the copyright-year bump. |
| [`capture-kernel-trace`](./capture-kernel-trace/SKILL.md) | Capture an `rocprofv3` ATT trace for an AITER kernel. |
| [`kernel-trace-analysis`](./kernel-trace-analysis/SKILL.md) | Analyze an ATT trace to find top stall hotspots mapped back to source. |
| [`bisect-perf-regression`](./bisect-perf-regression/SKILL.md) | Binary-search which AITER commit introduced a perf regression. |

## Using the skills

Claude Code / Codex: skills are auto-loaded from the repo's `.claude/skills/`
folder. Invoke explicitly with `/ ` (e.g. `/build-aiter`) or just describe
the task and let the agent pick the matching skill from the frontmatter.

Cursor Agent: copy (or symlink) the `.claude/skills/` directory to
`.cursor/skills/` in this repo, or to `~/.cursor/skills-cursor/` for global scope.

## Conventions used by every skill

- **GPU archs**: `gfx942` (MI300X) and `gfx950` (MI350X). Never parse product names;
  compare against the arch string (see `aiter/ops/triton/README.md`).
- **JIT cache**: compiled `.so` and `build/` live under `aiter/jit/`.
  `AITER_REBUILD=1` forces a rebuild, `AITER_LOG_MORE=1` increases verbosity.
- **Tuning data**: lives as CSV under `aiter/configs/` (plus `model_configs/`
  for per-model overrides). Every tuned op has a matched `*_untuned_*.csv` +
  `*_tuned_*.csv` pair.
- **Tests**: standalone Python scripts under `op_tests/`; run with
  `python op_tests/test_ .py` (no pytest required) or the CI helper
  `bash .github/scripts/aiter_test.sh`. Triton tests under `op_tests/triton_tests/`
  use `pytest`.
- **Pre-commit**: black / ruff / clang-format-18 with specific pinned versions
  (see [CONTRIBUTE.md](../../CONTRIBUTE.md)). The `format-code` skill reproduces
  that pipeline.

## Validating the skills

A lightweight validator is shipped with the skill set:

```bash
python .claude/skills/validate.py           # format + fact check; exits 0 on pass
python .claude/skills/validate.py --strict  # also fail on warnings
```

It verifies that every `SKILL.md` has proper YAML front matter, that
`name:` matches the directory, and that every AITER-repo path cited in
inline code actually exists in the checkout. The full test plan
(including a behavioral test matrix) lives in [`TESTS.md`](./TESTS.md).

## Contributing a new skill

Follow the format used by every file here:

```markdown
---
name: my-skill
description: >
 One-paragraph description. Claude reads this to decide when to activate the skill.
 Usage: /my-skill 
allowed-tools: Bash Read Edit Grep Glob
---

# My Skill

## Step 1 ...
## Step 2 ...
```

Keep skills **task-oriented**, **AITER-specific**, and **runnable**
(prefer concrete shell commands and file paths over abstract advice).
