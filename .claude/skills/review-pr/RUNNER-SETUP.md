# Runner setup — aiter-review-bot

The `@aiter-bot review` workflow (`.github/workflows/aiter-review-bot.yml`) runs on a
self-hosted runner labeled `self-hosted, box308`. This is what that runner must provide.
Verify the box with `bash .claude/skills/review-pr/preflight.sh` — it checks every item below
and prints a fix hint for each red. This doc explains how to make those checks pass; it holds
no machine-specific paths (each site fills in its own).

## 1. Register the runner (needs repo admin)

Creating a runner registration token requires **repo admin** on this repository (a `maintain`
or `push` role is not enough — the `registration-token` API returns 403). Either get an admin
to add the runner, or reuse a runner your org already has by giving it the labels above.

Install the runner on a **data volume with room to spare**, not `$HOME` or `/`: each review
checks out the repo and builds two worktrees (~180 MB apiece) under the runner's `_work`, and
a box whose root fs fills up will fail mid-review. Register with the labels `box308`.

## 2. Box config via the runner `.env` (loaded into every job)

The workflow's review step sets no environment on purpose, so the box-specific config lives in
`<runner-dir>/.env`, which the GitHub Actions runner injects into every job. Set at least:

    TMPDIR=<data-volume>/tmp                 # review scratch (WORK) lands here, off the root fs
    AITER_REVIEW_AGENT=<path>/claude          # the headless Claude entrypoint for this box
    ANTHROPIC_BASE_URL=http://localhost:30000 # the model endpoint (a local GLM here)
    ANTHROPIC_MODEL=<model>                    # e.g. a local GLM served by the box
    ANTHROPIC_SMALL_FAST_MODEL=<model>         # same model, or the session-title call warns
    PATH=<data-volume>/bin:/usr/local/bin:/usr/bin:/bin   # a gh >= 2.24 must be first (fetch needs baseRefOid)
    GH_CONFIG_DIR=<data-volume>/gh-config      # gh auth stored here, NOT as GH_TOKEN in the env

## 3. gh auth off the job environment

`fetch.sh` calls `gh` to read the PR. Authenticate it under `GH_CONFIG_DIR` (from a token with
`public_repo` read) rather than exporting `GH_TOKEN`, so the headless review agent never
inherits a token from its environment:

    GH_CONFIG_DIR=<data-volume>/gh-config gh auth login --with-token < token-file

## 4. Bot identity

Set the repo secret `AITER_BOT_TOKEN` to the bot account's PAT (`public_repo`). The workflow
uses it only in the claim and publish steps — never in the review step — so the review agent
cannot read it.

## 5. Verify

    bash .claude/skills/review-pr/preflight.sh    # all green = @aiter-bot review runs end to end
