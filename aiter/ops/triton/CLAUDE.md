This is the only `CLAUDE.md` file for the Triton folders, and it must stay.
Claude Code reads the rules in the `AGENTS.md` files through these imports.
The imports also work when Claude Code does not read `AGENTS.md` files itself,
for example when the repository root has a `CLAUDE.md`.
A `CLAUDE.md` at root supersedes an `AGENTS.md` file in triton since Claude Code prefers `CLAUDE.md` over `AGENTS.md`.


@AGENTS.md
@configs/AGENTS.md
@../../../op_tests/triton_tests/AGENTS.md
@../../../op_tests/op_benchmarks/triton/AGENTS.md
@utils/_triton/tuning/AGENTS.md
