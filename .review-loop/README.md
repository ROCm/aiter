# .review-loop — @aiter-bot review orchestration

The orchestration layer behind `@aiter-bot review` (triggered by
`.github/workflows/aiter-review-bot.yml`). It reuses the in-repo `review-pr` skill
(`.claude/skills/review-pr/`) — it does not reimplement it: `fetch.sh` wraps the skill's
`fetch.sh`, and `gates.sh`/`collect.sh` wrap the skill's `triage.py`.

Chain: `run_one.sh <pr>` — fetch → headless GLM worker (Step 1b–8) → independent GLM refuter
(Step 7.7) → seven gates → collect; then `publish.sh <pr> --post` posts the card as aiter-bot
(holds 🔴 HIGH RISK for a human, posts non-🔴, dedups by head SHA, skips merged/closed PRs).

- `prompts/` — verbatim-quoting prompt templates for the worker and refuter agents.
- `ci/authorized.txt` — the aiter-team allowlist of who may trigger.
- `ci/preflight.sh` — runner-side environment self-check.

The GLM backend (`claude-glm`) is provisioned on the self-hosted runner, not in this repo.
