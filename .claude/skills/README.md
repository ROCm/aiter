# AITER Claude Skills

This folder contains a set of [Claude skills](https://docs.anthropic.com/en/docs/claude-code/skills)
tailored for working on [AITER](https://github.com/ROCm/aiter) — AMD's
centralized repository of high-performance AI operators on ROCm.

**What a skill is, in one sentence:** a small Markdown file that teaches
the AI agent a project-specific workflow (commands, file paths, pitfalls)
so you don't have to re-explain AITER every time you open a session.

These skills encode AITER conventions (JIT compilation,
`optCompilerConfig.json`, tuning CSVs, the HIP / CK / Triton / FlyDSL /
ASM backend split, `op_tests/` layout, pre-commit formatting, …).

The structure mirrors `ROCm/FlyDSL/.claude/skills` and works for both
Claude Code / Codex and Cursor Agent.

---

## If you're brand new: 5-minute quickstart

### 1. Prerequisites

| You want to … | You need |
|---|---|
| **Use the skills** (agent reads them as context) | An agent that supports skills: Claude Code, Codex CLI, or Cursor Agent. That's it. No GPU needed just to read skills. |
| **Actually build AITER** | Linux + ROCm (6.x) + a supported GPU (gfx942 / MI300X, or gfx950 / MI350X). Windows hosts can read/edit code but CK and `PREBUILD_KERNELS` are disabled in `setup.py` — use WSL2 for a real build. |
| **Follow skills like `aiter-ck-tune`, `aiter-moe-tuning`, `capture-kernel-trace`** | A ROCm box with the target GPU + `rocprofv3`. |
| **Follow skills like `aiter-add-operator`, `format-code`, `build-aiter`** | Any machine where you can clone the repo is fine (no GPU required for reading the walkthrough). |

### 2. Install the skills

The files in this folder *are* the install — you just need the agent to
see them.

**Claude Code / Codex CLI** (auto-discovery):

```bash
# Already done — anything committed under .claude/skills/ in this repo
# is auto-loaded when the agent runs from the repo root.
claude --version   # any recent version
```

**Cursor Agent** (repo-local):

```powershell
# Windows PowerShell (run from repo root)
New-Item -ItemType Directory -Force .cursor | Out-Null
Copy-Item -Recurse -Force .claude\skills .cursor\skills
```

```bash
# Linux / macOS / WSL (run from repo root)
mkdir -p .cursor && cp -r .claude/skills .cursor/skills
```

Or make them available in **every** Cursor project:

```bash
# Linux / macOS
cp -r .claude/skills/* ~/.cursor/skills-cursor/
```

```powershell
# Windows PowerShell
Copy-Item -Recurse -Force .claude\skills\* $HOME\.cursor\skills-cursor\
```

### 3. Verify installation

Run the validator — it also double-checks the skills haven't drifted out
of sync with the repo (renamed file, missing tuner script, …):

```bash
python .claude/skills/validate.py
# expected: "Checked 13 SKILL.md files  errors: 0  warnings: 0"
```

Then ask the agent a probe prompt and watch it trigger a skill:

> *"Install AITER on a gfx942 box, prebuild forward kernels only."*

You should see the agent lean on `build-aiter` (it will mention
`PREBUILD_KERNELS=2`, `GPU_ARCHS=gfx942`, and point you at `setup.py`). If
it doesn't, see **Troubleshooting** at the bottom.

### 4. Suggested "first hour" for a new contributor

1. **`build-aiter`** — get an editable install working, confirm
   `import aiter` succeeds.
2. **`format-code`** — install the pre-commit formatters once, so your
   first PR doesn't bounce on lint.
3. **`aiter-add-operator`** — walk through the 5-file recipe even if you
   don't plan to add an op today; it's the best map of the codebase.
4. **`debug-aiter-op`** — bookmark this one; you'll need it the first
   time a test fails.

Everything else is task-triggered — the agent will pull it in when you
describe a matching problem.

---

## Skills at a glance

Grouped by the task you're trying to do.

### Setup & contribution hygiene
| Skill | When to use |
|-------|-------------|
| [`build-aiter`](./build-aiter/SKILL.md) | Installing / rebuilding AITER, picking a `PREBUILD_KERNELS` level, or debugging a bad `GPU_ARCHS` install. Thin hook around `setup.py` + `docs/installation.rst`. |
| [`format-code`](./format-code/SKILL.md) | Before every commit: runs AITER's pinned `black==26.3.0` / `ruff==0.15.7` / `clang-format-18` + copyright-year bump. |

### Writing a new operator
| Skill | When to use |
|-------|-------------|
| [`aiter-add-operator`](./aiter-add-operator/SKILL.md) | End-to-end recipe: HIP/C++ kernel → pybind → `optCompilerConfig.json` → Python wrapper → `op_tests/test_*.py`. Start here when adding any HIP/CK op. |
| [`aiter-triton-kernel`](./aiter-triton-kernel/SKILL.md) | Same but for Triton: `aiter/ops/triton/**` layout, `get_gemm_config`, `make_kernel_repr`, arch-string handling, JSON tuning configs. |

### Tuning
| Skill | When to use |
|-------|-------------|
| [`aiter-ck-tune`](./aiter-ck-tune/SKILL.md) | Tuning an existing CK GEMM (A8W8, A16W16, A4W4, …) for a new shape. Thin hook pointing at each op's `csrc/ck_*/README.md`. |
| [`aiter-moe-tuning`](./aiter-moe-tuning/SKILL.md) | Tuning the fused-MoE op — `tuned_fmoe.csv`, `untuned_fmoe.csv`, per-model configs under `aiter/configs/model_configs/`. |

### Debugging & benchmarking
| Skill | When to use |
|-------|-------------|
| [`debug-aiter-op`](./debug-aiter-op/SKILL.md) | `op_tests/test_*.py` returns NaN / wrong answer / wrong backend. Ordered playbook: backend → contiguity → dtype dispatch → all-ones → logging. |
| [`benchmark-aiter-op`](./benchmark-aiter-op/SKILL.md) | Measuring latency / TFLOPS / % of peak. Covers **both** AITER idioms: `@perftest` (inside `op_tests/test_*.py`) and `triton.testing.do_bench` (inside `op_tests/op_benchmarks/.../bench_*.py`). |
| [`aiter-jit-debug`](./aiter-jit-debug/SKILL.md) | `ModuleNotFoundError: module_*` at import, hipify errors, `AITER_REBUILD=1`, stale `aiter/jit/build/` cache. |
| [`bisect-perf-regression`](./bisect-perf-regression/SKILL.md) | An op got slower between two commits — drives `git bisect run` with a JIT-aware test script. |

### Profiling
| Skill | When to use |
|-------|-------------|
| [`capture-kernel-trace`](./capture-kernel-trace/SKILL.md) | Capture an `rocprofv3` ATT trace for an AITER kernel (`--kernel-include-regex`, `--kernel-iteration-range`). |
| [`kernel-trace-analysis`](./kernel-trace-analysis/SKILL.md) | Read an ATT directory: stall breakdown (VMEM/BARRIER/LGKMCNT/VALU), occupancy from VGPR/SGPR/LDS, hot-PC → source. |

---

## How to invoke a skill

You don't have to — just describe what you want and the agent picks the
skill whose `description:` frontmatter matches. Two equivalent ways:

| Style | Example prompt |
|-------|----------------|
| Natural language (preferred) | *"Help me add a new HIP op `my_add` that does element-wise addition."* |
| Explicit slash command | `/aiter-add-operator my_add` |

If the agent doesn't mention a skill in its answer, either (a) the prompt
was too vague to match any `description:`, or (b) the skills aren't
loading — see **Troubleshooting**.

---

## Conventions every skill assumes

- **GPU archs**: `gfx942` (MI300X) and `gfx950` (MI350X). Never parse
  product names; compare against the arch string
  (`aiter/jit/utils/chip_info.get_gfx()`).
- **JIT cache**: compiled `.so` and `build/` live under `aiter/jit/`.
  `AITER_REBUILD=1` forces a rebuild, `AITER_LOG_MORE=1` +
  `AITER_LOG_LEVEL=DEBUG` increase verbosity.
- **Tuning data**: lives as CSV under `aiter/configs/` (plus
  `model_configs/` for per-model overrides). Every tunable CK op has a
  matched `*_untuned_*.csv` + `*_tuned_*.csv` pair.
- **Tests**: standalone Python scripts under `op_tests/`; run with
  `python op_tests/test_<op>.py` (no pytest required). Triton tests
  under `op_tests/triton_tests/` use `pytest`. Bench drivers live in
  `op_tests/op_benchmarks/{triton,hip}/`.
- **Pre-commit**: black / ruff / clang-format-18 with specific pinned
  versions (see [`CONTRIBUTE.md`](../../CONTRIBUTE.md)). The
  `format-code` skill reproduces that pipeline exactly.

---

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| Agent never mentions any skill | You're probably running outside the repo root. Claude Code / Codex auto-load `.claude/skills/` **relative to cwd**; Cursor needs the files copied into `.cursor/skills/` (see quickstart). |
| `python .claude/skills/validate.py` reports missing paths | Repo layout changed (CK codegen dirs get renamed occasionally). Fix the path in the offending `SKILL.md` or open an issue. |
| A skill gives outdated advice | Skills are best-effort snapshots. Every skill either points at a live README or inlines the current convention; run Layer 2 of [`TESTS.md`](./TESTS.md) to spot-check facts. |
| Want to add your own skill | See below. |

---

## Validating the skills

A lightweight validator ships with the skill set:

```bash
python .claude/skills/validate.py           # format + fact check; exits 0 on pass
python .claude/skills/validate.py --strict  # also fail on warnings
```

It verifies that every `SKILL.md` has proper YAML front matter, that
`name:` matches the directory, and that every AITER-repo path cited in
inline code actually exists in the checkout. The full test plan
(including a behavioral A/B test matrix and recorded baselines) lives
in [`TESTS.md`](./TESTS.md).

---

## Contributing a new skill

Follow the format used by every file here:

```markdown
---
name: my-skill
description: >
 One-paragraph description. Claude reads this to decide when to activate the skill.
 Mention concrete trigger phrases ("add a new op", "tune gemm", …).
 Usage: /my-skill <arg>
allowed-tools: Bash Read Edit Grep Glob
---

# My Skill

## Step 1 ...
## Step 2 ...
```

Rules of thumb:

- **Task-oriented**, not topic-oriented — a skill answers *"how do I
  do X"*, not *"what is X"*.
- **AITER-specific** — if the content is identical to upstream ROCm /
  PyTorch / Triton docs, link to those instead.
- **Runnable** — prefer concrete shell commands and real file paths
  over abstract advice.
- **Pointer + gotchas over restating docs** — if there's already a
  `README.md` or `docs/` page in the repo, link to it and only add the
  pitfalls agents actually fall into. See `build-aiter` and
  `aiter-ck-tune` for the compact hook style.
- **Validate before opening a PR** — `python .claude/skills/validate.py`
  must pass, and if the skill is high-frequency, run Layer-3 A/B from
  `TESTS.md` and record the baseline.
