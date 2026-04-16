# AITER Skills — Test Plan

This file lists how to tell whether each skill in `.claude/skills/` is
actually pulling its weight. There are three layers; run them in order.

## Layer 1 — Format validation (automated, cheap)

```bash
python .claude/skills/validate.py           # must exit 0
python .claude/skills/validate.py --strict  # also fail on warnings
```

What it checks:

- Every `SKILL.md` has a well-formed YAML front matter.
- `name:` matches the parent directory name.
- `description:` is present and at least 60 chars (skills are triggered
  by matching a user's request against `description`; short descriptions
  silently never trigger).
- Every AITER-repo path (`aiter/...`, `csrc/...`, `op_tests/...`, …) cited
  in inline-code backticks actually exists.
- Obvious template paths (`aiter/ops/my_op.py`, `...category/test_<op>.py`)
  and runtime-only paths (`aiter/jit/build/...`) are skipped.

Wire this into CI whenever you extend the skill set.

## Layer 2 — Fact probe (semi-automated)

For each skill, confirm the commands / env vars / tool versions it asserts
are actually what the repo uses *today*.

```bash
# 1. Version pins cited in format-code
grep -n 'black==\|ruff==' CONTRIBUTE.md
# 2. Env vars cited by build-aiter / aiter-jit-debug
grep -rn 'PREBUILD_KERNELS\|GPU_ARCHS\|AITER_REBUILD\|AITER_LOG_LEVEL' setup.py aiter/jit/core.py aiter/__init__.py
# 3. Tuner scripts cited by aiter-ck-tune / aiter-moe-tuning
ls csrc/ck_gemm_a8w8/gemm_a8w8_tune.py \
   csrc/ck_gemm_moe_2stages_codegen/gemm_moe_tune.py
# 4. perftest decorator cited by benchmark-aiter-op
grep -n 'def perftest\|num_rotate_args' aiter/test_common.py
# 5. Pre-commit hook structure cited by format-code
grep -n 'clang-format\|black\|ruff' .githooks/pre-commit
```

All five groups should return at least one match. If any return nothing,
the skill is out of date — go fix it.

## Layer 3 — Behavioral test (the real signal)

Layers 1 and 2 only prove the skill isn't *wrong*. To tell whether it's
*useful*, run the same prompt with and without the skill and compare
the answer quality.

Recommended prompts (one per skill):

| Skill | Prompt | Must-hit markers in the answer |
|-------|--------|-------------------------------|
| `build-aiter` | "Install AITER on gfx942, prebuild only GEMM and attention." | `PREBUILD_KERNELS`, `GPU_ARCHS=gfx942`, concrete bitmask / kernel-list value. |
| `aiter-add-operator` | "Add a new HIP op `my_add` that does elementwise addition." | Covers all 5 artifacts: kernel, pybind, `optCompilerConfig.json`, `@compile_ops` wrapper, `op_tests/test_my_add.py`. |
| `aiter-jit-debug` | "`import aiter` raises `ModuleNotFoundError: module_activation`." | Mentions `AITER_REBUILD=1`, `aiter/jit/build`, `GPU_ARCHS` mismatch, `optCompilerConfig.json`. |
| `aiter-triton-kernel` | "Add a Triton a8w8 GEMM kernel under `aiter/ops/triton/gemm`." | `get_gemm_config`, `make_kernel_repr`, arch string `gfx9xx`, config CSV with `M_LEQ_x/M_GEQ_x/any`. |
| `aiter-ck-tune` | "Tune a8w8 GEMM for shape (8192, 8192, 8192) on gfx942." | Edits `aiter/configs/untuned_gemm_a8w8.csv`, runs `csrc/ck_gemm_a8w8/gemm_a8w8_tune.py`, then `AITER_REBUILD=1`. |
| `debug-aiter-op` | "`op_tests/test_rmsnorm.py` returns NaN for a new shape." | Systematic order: backend check → stride → dtype → all-ones test → logging env vars. |
| `benchmark-aiter-op` | "Benchmark `aiter.gemm_a16w16` at 4096³ vs `torch.matmul`, give roofline." | Uses `@perftest(num_rotate_args=…)`, reports TFLOPS and % of peak. |
| `aiter-moe-tuning` | "Tune Mixtral-8x7B fused MoE in bf16 on gfx942." | Edits `aiter/configs/untuned_fmoe.csv`, runs `csrc/ck_gemm_moe_2stages_codegen/gemm_moe_tune.py`, then `AITER_REBUILD=1` and `op_tests/test_moe.py::test_fmoe`. |
| `format-code` | "Format the 5 files I just edited before committing." | `black==26.3.0`, `ruff==0.15.7`, `clang-format-18`, copyright-year bump, `git diff` filtering. |
| `capture-kernel-trace` | "Capture an ATT trace of `aiter.gemm_a16w16`." | `rocprofv3 --att --kernel-include-regex ... --kernel-iteration-range "[1]"`. |
| `kernel-trace-analysis` | "I have an ATT directory at /tmp/aiter_att, tell me what's slow." | Stall breakdown (VMEM / BARRIER / LGKMCNT / VALU), occupancy from VGPR/SGPR/LDS, hot-PC → source mapping. |
| `bisect-perf-regression` | "`gemm_a16w16` is 30% slower than last week, find the commit." | `git bisect run`, a runner that does `rm -rf aiter/jit/build` + `AITER_REBUILD=1` and `exit 125` on build failure. |

Procedure per skill:

1. Start a fresh session with the skill directory disabled.
2. Ask the prompt. Score the answer against the markers.
3. Enable the skill and ask the same prompt.
4. Score again. A skill is useful only if the "with" column adds at least
   2 markers the "without" column missed.

Skills that don't improve the answer after one round of prompt tuning
should be pruned or merged into a neighboring skill.

## When to re-run

- **Layer 1** on every commit that touches `.claude/skills/`.
- **Layer 2** monthly, or after any AITER refactor that renames a
  top-level directory / env var (e.g. if `csrc/ck_gemm_moe_2stages_codegen`
  gets renamed again, the MoE skill needs updating).
- **Layer 3** whenever you rewrite a skill substantially, or whenever
  a teammate reports "the agent didn't help with X".
