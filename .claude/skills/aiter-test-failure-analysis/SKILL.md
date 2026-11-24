---
name: aiter-test-failure-analysis
description: Analyze ROCm/aiter main-branch Aiter Test failures, classify failed jobs and steps, extract uploaded test logs when available, rank suspicious commits since the last successful Aiter Test run, and produce a GitHub Actions summary report. Use for scheduled or manual Aiter Test failure triage.
argument-hint: "[owner/repo] [run-id]"
---

# Aiter Test Failure Analysis

Use this skill to inspect recent `Aiter Test` workflow failures on `ROCm/aiter` main and
produce an evidence-based summary for CI triage.

## Workflow

Run the bundled reporter from the repository root:

```bash
.claude/skills/aiter-test-failure-analysis/scripts/run_report.sh ROCm/aiter
```

Environment knobs:

- `AITER_TEST_FAILURE_RUN_ID`: analyze a specific workflow run id instead of scanning.
- `AITER_TEST_FAILURE_LOOKBACK`: recent Aiter Test runs to scan; default `120`.
- `AITER_TEST_FAILURE_AGENT`: headless analysis agent; default is `AITER_REVIEW_AGENT`
  from the runner environment, then `claude-glm`.
- `AITER_TEST_FAILURE_AGENT_TIMEOUT`: agent timeout in seconds; default `1800`.
- `AITER_TEST_FAILURE_SKIP_DOWNLOADS=1`: skip log/artifact downloads for dry runs.
- `AITER_BOT_TOKEN`: optional token used only by the post-analysis auto-revert
  script to push a revert branch and create a PR. The analysis agent is run with
  GitHub tokens removed from its environment.
- `GITHUB_STEP_SUMMARY`: summary output file; GitHub Actions provides this.
- `GH_TOKEN` or `GITHUB_TOKEN`: token with `actions:read` and `contents:read`.

## What The Reporter Does

1. Finds the latest completed failed `Aiter Test` run on `main`, or uses
   `AITER_TEST_FAILURE_RUN_ID` when provided.
2. Reads failed jobs and failed steps, separating aggregate jobs such as
   `Aiter Test Gate` from root failures.
3. Downloads only small test-log artifacts (`standard-test-log-*`, `multigpu-test-*`,
   and `aiter_shards`) and avoids wheel artifacts.
4. Extracts likely failure snippets from `latest_test.log` and job logs.
5. Finds the latest successful Aiter Test run before the failed run and ranks commits in
   that range by path and keyword overlap with the failure surface.
6. Writes `evidence.json` and `culprit.json` for downstream automation.
7. Gives the evidence to the configured headless agent for a concise report. If the agent
   is unavailable, it writes the deterministic fallback report instead.
8. In the GitHub Actions workflow, a separate `box308-pr-revert` job may create
   a revert PR only when `culprit.json` has `auto_revert_ready: true` and
   `confidence: high`.

## Reporting Rules

- State whether logs or artifacts could not be downloaded.
- Do not blame a commit solely because it is in the range; mark suspects by evidence
  strength.
- Treat `check-signal` and `Aiter Test Gate` as aggregate jobs unless their own logs show
  a root failure.
- Distinguish build failures, standard shard test failures, and multi-GPU smoke/test
  failures.
