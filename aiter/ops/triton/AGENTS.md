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
| `aiter/ops/triton/utils/_triton/tuning/AGENTS.md` | Tuning scripts |
| `op_tests/triton_tests/README.md` | Unit tests |
| `.github/instructions/aiter-ops-triton.instructions.md` | Code review checks |

Do not copy rules into this file. To change a rule, edit the file that has it.

`aiter/ops/triton/CLAUDE.md` is the only `CLAUDE.md` file, and it must stay.
It imports the `AGENTS.md` files for Claude Code. When you add an `AGENTS.md`
file, add an import for it there. Do not add a `CLAUDE.md` file in a different
folder.

Claude Code loads the two README files through these imports:

@README.md
@../../../op_tests/triton_tests/README.md
