# Agent instructions: Triton and Gluon ops

This file is for coding agents, such as Claude Code, Codex and Copilot. It
applies to `aiter/ops/triton/`, `op_tests/triton_tests/` and
`op_tests/op_benchmarks/triton/`.

The rules are in the files below. People and agents use the same files. Read
them before you change or review code here.

| File | Rules for |
| ---- | --------- |
| `aiter/ops/triton/README.md` | Kernels, configs, tests, benchmarks, comments and PRs |
| `aiter/ops/triton/configs/AGENTS.md` | Config files and config loaders |
| `op_tests/triton_tests/README.md` | Unit tests |
| `.github/instructions/aiter-ops-triton.instructions.md` | Code review checks |

Do not copy rules into this file. To change a rule, edit the file that has it.
Do not add more `CLAUDE.md` files. The `CLAUDE.md` in this folder only imports
the `AGENTS.md` files for Claude Code. When you add an `AGENTS.md` file, import
it there.

Claude Code loads the two README files through these imports:

@README.md
@../../../op_tests/triton_tests/README.md
