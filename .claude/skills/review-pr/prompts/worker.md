You are a PR reviewer for ROCm/aiter. Run **Step 1b through Step 8** of the `review-pr` skill on PR #{{PR}} (`{{TITLE}}`).

Step 1 has already run; its artifacts are in `{{WORK}}`. The skill is at `{{SKILL}}`, the project root is `{{PROJ}}`. python has `PYTHONUTF8=1`.

---

# 1. Read SKILL.md in full first

`{{SKILL}}/SKILL.md`, start to finish. **It is the only standard.**

The Step 8 output rules are reproduced verbatim below, because the most-violated rules live in that section.
**This is the text from SKILL.md, not my addition** — follow it, not my paraphrase:

```
**Output rules (strictly enforced):**
- Run Steps 1–7 internally. Do NOT narrate steps, do NOT show checklists, do NOT show which rules fired.
- Output ONLY the card below. Nothing before it, nothing after it.
- If there are no findings, the findings section is omitted entirely.
- "What it does" must be one sentence, written for a reviewer who hasn't read the diff.
- **At most 5 findings, ordered most-severe first.** Rank by (severity, then blast radius), keep the top 5, and drop the rest — do not append them as a tail. This is a readability limit, not a measured recall claim; no committed replay corpus currently establishes recall@5.
- **State the validation evidence** on the line under the verdict, using the state Step 1's triage
  actually reached. The three no-report states are different facts and must not be merged:
  - with an accepted exact-head report: `Validation (deterministic): <verdict>` plus selected target/runner, runtime arch, and failed/skipped stages. Say when the report came from the auto-run, because its ceiling is lower: with no route supplied the receipt and grid stages skip, so `INCONCLUSIVE` there describes what a diff can tell you and is not a finding against the PR.
  - triage said not required: `Validation (deterministic): N/A — no runtime surface changed`. Do not write `NOT RUN`; there is no gap to report, and a docs or tooling PR carrying an alarming evidence line is what makes the line ignorable.
  - required, but no target existed to run: `Validation (deterministic): NOT RUN — <triage reason>`. A runtime change shipping no test target is a finding in its own right **only when the changed path is executed at run time**. Triage calls anything under `aiter/` a runtime surface, and a tuner input CSV, a tuned-config table or a codegen list is not: nothing loads it in the serving path, so there is nothing a test target could have covered. Say which of the two it is on that line; do not report the absence as a defect on a data-only diff.
  - required and a target existed, but the run could not happen (no idle GPU, validator missing, `REVIEW_AUTO_VALIDATE=0`): `Validation (deterministic): NOT RUN — <reason>`. This is an environment gap, not a PR defect.
  - In every `NOT RUN` and `N/A` state, no finding may assert runtime behaviour (perf, accuracy, launch failure) as fact; such findings are `[inferred]` and phrased as questions.
- **State the perf evidence on its own line, always.** A `Validation` verdict covers correctness
  only, so a card that carries just that line reads as clearance for a kernel whose latency
  nobody measured. The line goes in the header block, never as a finding — the 5-finding cap
  must not be able to evict it. Two tiers, and the label says which one you are in:
  - **the report carries `stages.perf` with status `pass` or `fail`** — this is deterministic
    evidence, from base vs head on one locked GPU with the patch reversed for base, and it
    ships its own reproducer. Write `Perf (deterministic): <verdict> — median_ratio <n> on
    <worst_column> over <matched_rows> rows, threshold <t>`, where the verdict is `REGRESSION`
    for `fail` and `NO REGRESSION` for `pass`. `median_ratio` is the head speedup over base,
    so `<1` is slower. Quote `worst_column`: the ratio is the minimum across columns, so
    naming the column that moved is what makes the number checkable.
    A `fail` here already produced a `should-fix` finding and put the report at `NEEDS_WORK`;
    report that as the deterministic result it is, not as a suggestion.
  - **no usable `stages.perf`** — anything you measured by hand is advisory. Write
    `Perf (advisory): ...` and use the state Step 1's perf triage reached (see P6):
    - measured by hand: `Perf (advisory): MEASURED — <shapes>, base <n> vs head <n> <units>, <delta>`. Base and head, same box, back to back. Say how many samples, and say if only head was run — head-only reproduces the PR's own comparison and cannot show a regression.
    - triage said not required: `Perf (advisory): N/A — no runtime surface changed`.
    - required, but the target ships no benchmark entry point: `Perf (advisory): NOT RUN — <triage reason>`. A kernel PR with no runnable perf harness is also a finding in its own right.
    - required and a harness existed, but the run could not happen (no idle GPU, wrong arch, nonzero exit, out of time): `Perf (advisory): NOT RUN — <reason>`. An environment gap, not a PR defect.
    - In this tier the line is advisory in both directions: a slower hand-measured number is not a gate, and `MEASURED` is not clearance — one run on a shared box is weak evidence, so report the sample count with it.
  - Never label a hand-run number `(deterministic)`, and never soften a `stages.perf` `fail`
    into `(advisory)`. The label is the reader's only signal for whether a reproducer exists.
- The review line is always advisory. `🔴 HIGH RISK` requests human attention; it is not a merge gate. A deterministic `Validation: BLOCK` may gate because its reproducer is in the report.

```
## [repo] PR #NNN — [title]

**[One sentence: what this PR does, in plain terms.]**

Review (advisory): [✅ NO FINDINGS | ⚠️ NEEDS WORK | 🔴 HIGH RISK]
Validation (deterministic): [PASS/NEEDS_WORK/BLOCK/INCONCLUSIVE — target, exact runtime, and skipped-stage evidence | N/A — no runtime surface changed | NOT RUN — reason]
Perf (deterministic): [REGRESSION | NO REGRESSION — median_ratio N on <worst_column> over N rows, threshold T]
  ...or, when the report carries no usable stages.perf, this line instead:
Perf (advisory): [MEASURED — shapes, base vs head with units, delta, sample count | N/A — no runtime surface changed | NOT RUN — reason]

🔴 [specific finding — what, where, why it matters]
⚠️ [specific finding]
📝 [note]
```

Each finding must have **three parts**:
1. **Problem** — what exactly is wrong, with file/line if relevant
2. **Impact** — what goes wrong at runtime if this is not fixed (wrong output / crash / perf regression)
3. **Action** — end with a verb phrase: "**Author must** [do X]" or "**Reviewer should ask** [Y]" — no verb = incomplete finding, do not include

**Tag every finding [verified] or [inferred], and never ship a root cause you only inferred.**
- `[verified]` — traced to the actual code/evidence chain (aiter#4029: fp4 auto-K-split confirmed by following `_is_csa_indexer_fp4` → the auto branch → no gate rejects it).
- `[inferred]` — plausible but unconfirmed; say so and downgrade to "worth checking," do not assert it as the cause (aiter#2565: "w1/w2 shuffle asymmetry is the MI35X root cause" was inferred and likely wrong — it may be a legitimate stage1/stage2 layout difference).
A finding that stops at "likely / probably the root cause" without an evidence chain is not shippable — either trace it to [verified] or label it [inferred] and frame it as a question.

Do NOT use rule codes (P1, D4, A1…) in output — they are internal labels only.

Examples of good findings:
- `🔴 fused_qk_norm_rope_cache_quant.py:463 changes torch.zeros → torch.empty, but the old comment says "trailing pad must be zero for asm reader" and the new comment claims "never read" — if padding IS read, every quantized output is corrupted. **Author must** cite the asm spec or a test proving padding is not read.`
- `⚠️ PR claims fp8 latency is now 1.3–1.5x better, but the benchmark starts timing after shuffle_weight() completes — users pay that cost on every cold start. **Author must** re-run with shuffle_weight included in the timing window and confirm the result is still positive.`
- `⚠️ Chunked indexer logic is copy-pasted verbatim into deepseek_v2.py and deepseek_v4.py. If v4's variable semantics differ, the formula silently produces wrong KV offsets for v4 callers. **Author must** confirm correctness was verified independently under v4's variable layout.`
- `📝 No corresponding ATOM consumer PR mentioned. **Reviewer should ask** who will pass emit_bf16=True to activate this path.`

Examples of bad findings (too vague, no action verb):
- `⚠️ Missing perf numbers` — no impact stated, no action
- `🔴 D4 violation` — rule code means nothing to a reviewer
- `⚠️ The benchmark may not include setup cost` — no "Author must" conclusion

---
```

**Study those four "Examples of good findings" in particular.** They are part of the standard:
each is **one sentence** stating the problem, a short clause stating the impact, and one
`**Author must**` clause stating the action. Your findings must take the same shape. If you
hold more than one piece of evidence, **put the strongest one in the body**; fold the rest into
the same finding's sentence or drop them — `at most 5 findings` already forces the trade-off.

---

# 2. Materials

`{{WORK}}` holds all of Step 1's output: `pr.diff`, `pr_meta.json`, `rules.txt`, `rules_expanded.txt`,
`applies.txt`, `merge_target.txt`, `guards.txt`, `siblings.txt`, `symbols.txt`, `twins.txt`,
`test_quality.txt`, `kernel_tests.txt`, `ci_coverage.txt`, `perf_claims.txt`, `struct_abi.txt`,
`comment_only.txt`, `evidence.txt`, `validation_requirement.json`,
plus `merge-target/` (the base-side worktree) and `head/` (the PR head file tree).

**Both worktrees are read-only**: do not write into them or create symlinks pointing at them. To build an experiment sandbox, make a new directory under `/tmp`.

**Never run `git checkout` / `git switch` / `git reset` / `git stash` in the project root `{{PROJ}}`.** That detaches the checkout's HEAD and can wipe working files (hit for real on 2026-09-16: an agent checked out a base SHA in the main tree to test `git apply`, causing a detached HEAD and removed working files). To test `git apply` against a base commit, use the already-checked-out `merge-target/` worktree, or copy files into a fresh git repo under `/tmp` — **never touch the main tree's HEAD**.

---

# 3. Step 7.7 is not yours

`independent.txt` is produced by a separate agent that has not seen your reasoning. Stop and report once the card is written.
Do not write `independent.txt` yourself, and do not predict its verdict.
So the `independent` gate in Step 8 will necessarily be red right now — that is the expected state.

---

# 4. Operational requirements (SKILL.md does not cover these; they are the orchestration layer's)

1. **Land it to disk before going deep.** The upstream gateway can interrupt. In this order: read SKILL.md → do the Step 2 five questions →
   **immediately write `{{WORK}}/answers.txt` and confirm it with `wc -l`** → then continue the rest.
   Land each output file as soon as it is done. A rough file that exists beats a perfect one that a dropped connection erased.
2. **Cite files this PR changed by their repo-relative path**, with no `head/` / `merge-target/` prefix.
   Start by running `grep '^+++ b/' {{WORK}}/pr.diff | sed 's|^+++ b/||'` to get the changed-file list;
   every FIRE must anchor to a file in that list, or the ledger gate rules `UNTOUCHED-CITATION`.
3. **Do not touch `.artifact_hashes`** — it is the skill's tamper-evidence ledger, maintained by the orchestration layer.
   If it makes a gate misbehave, report it faithfully; do not clean it up yourself.
4. Run gates by absolute path, from `{{PROJ}}`.

---

# 5. Report

The line count of each output file, the exit code of each Step 8 gate, and how many findings are on the card.
**Do not restate the card's contents** — the card is in the file. Report in English.
