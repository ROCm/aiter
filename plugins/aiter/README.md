# aiter Claude Code plugin

Packages the agent skills under [`.claude/skills/`](../../.claude/skills/) as a Claude Code plugin,
so they can be used outside an aiter checkout or pinned per user. `skills/` here is a symlink to
that directory, so a skill has one source of truth. Inside an aiter checkout, the same skills
already load as project skills without installing anything.

| Skill | Use |
|---|---|
| `aiter-config-shape` | Add or merge tuned config CSVs under `aiter/configs/**` without duplicate shapes |
| `aiter-op-test` | Write or extend `op_tests/test_*.py` to the aiter op-test standard |
| `create-pr` | Open or update a ROCm/aiter PR that passes aiter-bot's PR Self-Review |
| `opus-kernel-best-practice` | Compile-time guidance for OPUS (`opus.hpp`) HIP kernels |
| `opus-module-build-optimization` | Reduce first-call JIT build time of OPUS-based modules |
| `review-pr` | Advisory AI review of aiter/FlyDSL PRs |
| `validate-kernel-pr` | Reproducible base-vs-head GPU validation of a kernel PR |

FlyDSL authoring and cleanup skills (`flydsl-kernel-authoring`, `kernel-code-cleanup`, ...) are
maintained upstream in [ROCm/FlyDSL](https://github.com/ROCm/FlyDSL/tree/main/.claude/skills) and
are not duplicated here.

## Install

```bash
claude plugin marketplace add ROCm/aiter --sparse .claude-plugin plugins .claude/skills
claude plugin install aiter@rocm-aiter
```

Or, in a session: `/plugin marketplace add ROCm/aiter`, then `/plugin install aiter@rocm-aiter`.

`--sparse` keeps the marketplace clone to the plugin files instead of the whole repository. Keep
`.claude/skills` in the list: `skills/` links there, and without it the plugin loads no skills.

Plugin skills are namespaced, for example `/aiter:review-pr`. Inside an aiter checkout, the
project copy is also available as `/review-pr`.

## Develop

Edit skills in `.claude/skills/<name>/`, then check the plugin with:

```bash
claude plugin validate .            # marketplace (.claude-plugin/marketplace.json)
claude plugin validate .claude/skills
claude plugin details aiter@rocm-aiter
```

Bump `version` in `.claude-plugin/plugin.json` when a release should reach installed users.
