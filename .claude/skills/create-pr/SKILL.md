---
name: create-pr
description: Open (or update) an aiter pull request that is ready for human review on the first try — one backend per PR, the auto component tag plus a hand-written type tag in the title, a description that follows .github/pull_request_template.md, and a PR Self-Review section that passes aiter-bot's own validator for the current head SHA. Use whenever creating a PR to ROCm/aiter, editing an aiter PR description, pushing new commits to an open aiter PR, or answering an aiter-bot "NOT READY FOR HUMAN REVIEW" comment.
argument-hint: [branch or PR number]
---

# Create an aiter PR

aiter's PR rules live in three places, all on `origin/main`:

- **`CONTRIBUTE.md` → "Pull Requests":** one backend per PR, the title tag format, and readiness criteria.
- **`.github/pull_request_template.md`:** the description skeleton.
- **`.github/scripts/pr_self_review.py`:** aiter-bot. It posts a `PR Self-Review` comment and a status. It reports
  **NOT READY FOR HUMAN REVIEW** until the description answers its questions for the *current* head SHA
  and the required CI is green.

Always read these from `origin/main`, not from your branch. A feature branch can be older than the rules.

```bash
git fetch origin main
git show origin/main:CONTRIBUTE.md | sed -n '/^## Pull Requests/,/^## /p'
git show origin/main:.github/pull_request_template.md
```

## 1. Scope check before pushing

- **One backend per PR.** Triton/Gluon, HIP, ASM, CK, OPUS and FlyDSL are separate backends; Triton and Gluon count as one.
  - If the diff touches two backends, split it (use stacked PRs when one part depends on the other).
  - `git diff --name-only origin/main...HEAD` shows the effective scope.
- **No unrelated files.** Look for artifacts, `*.log`, `__pycache__`, conflict markers, or `pdb`/`breakpoint` lines. aiter-bot flags these as hygiene failures.
- **Formatting** must match CI's pinned versions; use the `commit-and-push` / `format-code` skills. The workflow pins Ruff in `.github/workflows/pre-checks.yaml`, and Black is `psf/black@stable`.
- **Base:** branch from the latest `origin/main`. If main moved after your last commit, check `git diff --name-only HEAD...origin/main` for overlap with your files.

## 2. Title

The title has the form `[<Component>] [<Type>] <imperative description>`.

- **Component tags** are added and normalized automatically by `.github/workflows/pr-title-tags.yaml`, based on the files changed. Writing one yourself is fine; it must be the canonical form: `[Triton/Gluon]`, `[ASM]`, `[HIP]`, `[CK]`, `[OPUS]`, `[FlyDSL]`, `[CI]`, `[JIT]`, `[Build]`, `[Config]`, `[Docs]`.
- **Type tags are never added for you.** Pick one: `[Bugfix]`, `[Feature]`, `[Kernel]`, `[Perf]`, `[Test]`, `[Hardware]` (e.g. `[Hardware][MI300X]`), `[Misc]`.

Example: `[FlyDSL] [Perf] pa_decode: peel the last KV tile so only it carries the V tail mask`.

## 3. Description

Follow the template section by section:

1. Summary
2. Motivation
3. Changes
4. Performance (if applicable)
5. Testing
6. Documentation
7. Dependencies
8. Breaking Changes
9. PR Self-Review

Rules that reviewers and the bot rely on:

- **Motivation** explains *why* the current behavior is insufficient, i.e. the mechanism, not just "faster".
- **Changes** lists what changed and what deliberately did not. For refactors that dedent code, tell reviewers to use "Hide whitespace".
- **Performance** is required for kernel/perf changes. Use the template table `| Hardware | Workload/configuration | Before | After | Change |`.
  - State the exact GPU (e.g. an 80-CU gfx942 part is **not** "MI300X"), the shapes, the timing method and the run-to-run noise.
  - Report the worst shapes and any reproducible regression, not only the geomean.
  - **Re-derive every µs and % value from the raw benchmark data right before posting.** Hand-copied numbers drift.
- **Testing:** tick only what was actually run. Explain an unchecked item instead of deleting it.
- **Breaking Changes:** write `None` when there are none.

### PR Self-Review

The bot parses `###` headings with exactly these names:

| Heading | Rule (from `validate_author_answers`) |
|---|---|
| `### Reviewed revision` | At least the first 7 characters of the **current head SHA**. |
| `### Behavior and intent` | At least 18 word units. What behavior changes and why the old behavior was insufficient. |
| `### Validation and unverified scope` | At least 18 units. Must name evidence (a `command`, `pytest`, `passed`/`failed`, a link) **and** explicitly state what is unverified / not tested / not covered, or say none with a reason. |
| `### PR-specific question` | At least 14 units. Answer the question the bot selects from the changed paths (table below). |
| `### Changed op tests` | Name every changed `op_tests/` file (path or basename) with what it covers and why it is not redundant. Write `None` (with a reason) if none changed. |

Placeholder answers (`N/A`, `none`, `yes`, `done`, `tested`, `ok`, `see above`, ...) count as empty. HTML comments are stripped before the bot reads an answer.

The question is picked by `select_question()`; the first matching row wins:

| Changed paths | Question to answer |
|---|---|
| any `.github/` | Permissions and untrusted-input boundary; can fork PR code or text reach a privileged path? |
| `csrc/`, `hsa/`, `*.cu`, `*.cuh`, `*.hip`, `*.s` | Which arch/dtype/shape boundaries select the changed kernel path, and what invariant keeps unsupported inputs out? |
| a path containing `config`, or any `.csv`/`.json` | Which exact workload keys consume the config change, and what shows unrelated shapes keep their previous selection? |
| `aiter/jit/`, `setup.py`, `pyproject.toml` | What changes in dependency/cache/compilation behavior; verified on both a clean build and an existing cache? |
| `op_tests/` | What would each changed test fail on before this PR, and why isn't an existing canonical test enough? |
| anything else (e.g. `aiter/ops/...`) | The most important boundary or failure mode introduced, and the evidence that covers it. |

Answer with a concrete invariant and the test or benchmark that exercises it, not a restatement of the change.

## 4. Validate locally, then create

Write the body to a file, then run the bot's validator against it:

```bash
python .claude/skills/create-pr/validate_pr_body.py body.md            # head SHA = local HEAD
python .claude/skills/create-pr/validate_pr_body.py body.md --pr 1234  # head SHA + changed op_tests from GitHub
```

It prints `OK` or lists the exact issues the bot would report. Fix every issue before posting.

```bash
git push -u origin HEAD
gh pr create --repo ROCm/aiter --base main --title "<title>" --body-file body.md
# later edits:
gh pr edit <N> --repo ROCm/aiter --body-file body.md
```

- **Push credentials:** if HTTPS push has no credential helper, run `git -c credential.helper='!gh auth git-credential' push ...`. This uses gh's token for one command without changing global git config.
- **Draft PRs** are never "ready". Mark the PR ready only once the self-review is filled in.

## 5. After creating, and after every push

1. Read the bot's comment:
   `gh api repos/ROCm/aiter/issues/<N>/comments --jq '.[] | select(.body | contains("aiter-pr-self-review")) | .body'`.
   The answers section should say **"Author self-review accepted"**. Any remaining `WAIT` is CI: Black, Ruff, Repository dependency, Aiter Test Gate.
2. **Every new push invalidates the self-review.**
   - Update `### Reviewed revision` to the new head SHA.
   - Re-check that the answers and numbers still describe the new head.
   - Re-run the validator.
3. **Copilot findings** need a substantive author reply; resolving the thread is optional.
4. **Downstream CI.** For changes to ops that ATOM / SGLang / vLLM consume (attention, pa_decode, MLA, fused_moe, quant, gemm, jit core), downstream CI is skipped unless a `ci:*` label is added.
   - Map the changed path to a model and pick the minimal label. Read `.github/workflows/*.yaml` for the current `ci:atom` / `ci:sglang` / `ci:vllm` / `ci:all` rosters.
   - Ask the user before adding labels.

## Example

ROCm/aiter#7197 is the reference for a FlyDSL perf PR whose description passes the bot on the first edit. It includes:
- two-arch performance rows;
- an honest unverified-scope list;
- a PR-specific answer naming the invariant (`start < end`) and the test that covers it.
