# The loop contract

This file is the contract. The orchestration layer underneath it can be replaced — a different
harness, a different flow engine, a different set of models — but the directories, artifacts,
gates and termination criteria defined here cannot, because every guarantee the loop offers is
expressed in terms of them.

Everything below was derived from three complete rounds against one real PR at three successive
heads, on two GPU architectures. Numbers marked *measured* come from those runs.

---

## 0. Roles and models

| role | responsibility | model class |
|---|---|---|
| `worker` | fetch, rule adjudication, core-file assessment, AI diagnostic, writing the card | local / cheap |
| **`refuter`** | **independent refutation — must be a different model family from `worker`** | frontier |
| `formatter` | prose → structured lines, mechanical file writing | local / cheap |
| `remote-exec` | ssh, docker, running the skills on the GPU host | local / cheap |

**Never hardcode these.** They come from configuration, and a human confirms them at the single
break. The one constraint that is not negotiable is the second row: a refuter drawn from the same
model as the worker is checking its own homework.

*Measured:* a same-model refuter killed 18% of findings (2 of 11). A cross-model refuter killed
40% (7 of 17, then 6 of 15), including four of six `RED`s in one round. The one false positive that
reached a card and had to be withdrawn by hand came from a same-model round; the identical false
positive was caught automatically once the refuter changed family.

---

## 1. The artifact contract

Files are the contract. Return values are logs.

### Local

```
<workspace>/
  <repo>/                          the checkout the gates read (long-lived, read-only to the loop)
  <tmpRoot>/                       every intermediate; deletable in one command after a round
    pr-review/<pr>/<round>/work/       the skill's $WORK, synced back from the host
    pr-review/<pr>/<round>/reports/    card.md, findings.md, validation_report*.json
    runs/<run-id>/                     one-off scripts and launch logs
    logs/                              subagent transcripts
```

No agent may write outside `<tmpRoot>` and the loop's own package directory. Creating files in the
workspace root is forbidden.

### Remote

```
<host>:<hostRoot>/                     the PR body: clone, base worktree, patch (long-lived)
<container>:<scratch>/                 every temporary script, log and test artifact
```

Hard constraints:

- **All PR testing runs inside the container.** The host does git and docker, nothing else.
- **Temporary scripts go in the container scratch.** Writing scratch to the host's `/tmp` is
  forbidden beyond a single script the agent deletes afterwards.
- **Every round ends by removing its scratch** and chowning any root-owned artifact the container
  left in the worktree back to the host user. A round that skips this leaves the next round unable
  to remove its own worktree.

### Handoff

- Agents write their conclusions to named files; the return value only carries the prompt forward.
- **The `worker` role never gets an output schema.** Structured output is the `formatter`'s job,
  with at most three flat string fields.
  *Measured:* of 74 agents in an early round, 23 (31%) returned nothing to the orchestrator while
  their own logs showed `turn/end: completed`, files written and commands run. Schema validation,
  not model capability, discarded finished work. Every one of those was recoverable only because
  the agent had also written its result to disk.

---

## 2. Stages

| stage | what happens | artifacts | role |
|---|---|---|---|
| **S0 env** | probe host, GPUs, container, tooling; confirm with a human | env record | remote-exec |
| **S1 fetch** | run `review-pr/fetch.sh` in the container; sync `$WORK` back | ~20 skill artifacts | remote-exec |
| **S2 analysis** | Step 2 semantic, Step 4 per file, Step 5 per rule group, Step 6 diagnostic | `answers.txt`, `core_files.txt`, `verdicts_*.txt`, `ai_diagnostic.txt` | worker |
| **S3 refutation** | one fresh agent per finding; each writes its own ledger line | `refutation_lines/F<nn>.md`, `refutations.txt`, `independent.txt`, `findings.md` | **refuter** |
| **S4 card** | rank surviving findings, write the card, account for the rest | `card.md` | worker |
| **S5 gates** | the skill's seven `triage.py` gates | all green, or the complaint is fed back | worker |
| **S6 validation** | `validate_pr.sh` per target, in the container, on a claimed-idle GPU | `validation_report*.json` | remote-exec |
| **S6b executor refutation** | attack the executor's *own* findings the same way | verdicts per executor finding | **refuter** |
| **S7 refold** | fold deterministic evidence into the card; re-gate; verify independently | updated `card.md` | worker |
| **S8 cleanup** | recover artifacts, delete scratch, restore ownership | — | remote-exec |

### Granularity in S2 and S3

**Fan out; do not batch.** One agent per file and per rule group, one refuter per finding.

*Measured:* a single agent given one rule family consumed 177k input over 79 steps; ten per-file
agents consumed 4k–26k each and all ten succeeded, while the batched equivalent failed four times
in a row. A single refuter given all 11 findings cost more (111k input, 84 steps) than the eleven
separate refuters and failed outright twice. Cumulative input grows with *step count*, because
every step resends the context — so step count, not file size, is what drives cost.

### The cost of refutation

*Measured:* 3.7k–38.8k input per finding, median ~10k; 11 findings ≈ 150k. The refuter is given the
finding and a pointer to the evidence directory — **never the diff**; it greps for what it needs.
Report the estimate (`findings × ~15k`) to the human at the break.

---

## 3. Human gates: all of them at the front

```
[once per host]  environment probe → human confirms → env record, reused by later rounds
[per PR]   phase P (prefetch): verify env + skill Step 1                 ← cheap, 2 agents
              ↓  ★ the single break: one batch of questions
           phase R: S2 → S3 → S4 → S5 → S6 → S6b → S7 → S8               ← no further stops
```

**Why the break cannot move earlier.** The human must choose validation targets, and the candidate
list comes from `validation_requirement.json`, which is produced by Step 1. The alternatives are to
guess the target or to reimplement the skill's requirement derivation — the first is unacceptable,
the second drifts from the skill immediately.

**The four questions, asked once:**

1. **Environment** — host, container/image, GPU pool, PR body path. Written to the env record.
2. **Models** — one per role, from what the harness actually offers.
3. **Refutation budget** — granularity and the estimate.
4. **Validation** — the candidate targets, or `N/A` when the skill reports no runtime surface.

After phase R starts, only a hard error may interrupt: host unreachable, patch conflict, no idle
GPU, or the gates failing for `maxGateRounds` rounds. Everything else is recorded and carried.

### Six fail-closed guards

All are covered by `tools/dryrun.mjs`, which stubs the harness hooks and asserts each one fires
without calling a model (10/10).

| guard | fires when | the incident behind it |
|---|---|---|
| S2 completeness | any analysis agent returns nothing | a silently failed Step 4 left artifacts missing, and the gate then asked the card's author to supply them |
| collection floor | findings < the `FIRE` count in `verdicts_*.txt` (**derived, not configured**) | a collector returned nothing and the refutation stage ran zero agents |
| ledger provenance | refuter files ≠ ledger lines ≠ findings, or any malformed line | one formatter asked to write 17 ledger lines wrote 2 |
| gate termination | still not green after `maxGateRounds` | the original implementation recorded the failure and delivered the card anyway |
| ledger untouched | gates report green but the ledger line count has changed | the card's author wrote the refutation ledger itself to turn a gate green |
| post-refold check | gates fail after S7 edits the card | only the editing agent witnessed its own success |

**Separation of duties covers two records, not one.** The `worker` may not write the refutation
ledger (`refutations.txt`, `independent.txt`, `findings.md`, `refutation_lines/`) — and may not add
a verdict line for a rule that has none. The second is the same hole through a different door: a
rule with no verdict means an analysis agent failed, and the correct response is to report a
blocker, not to adjudicate it yourself.

---

## 4. The card

- At most **five findings**. This is the `review-pr` skill's own readability limit, and the skill
  says plainly that it is not a measured recall claim.
- Everything dropped, killed or downgraded is accounted for at the bottom of the card with
  `-- not reported: <reason>`; anything withdrawn after the blind-spot check uses `-- late finding:`.
- **`findings.md` ships alongside**: every finding, its refutation verdict and the deciding
  evidence. The card is for the reviewer's attention; `findings.md` is for anyone who wants to dig.
- When a previous round exists, the card carries a **"since last review"** line. Once the base
  moves, a file the previous head added and this head removed is invisible in the diff — it reads
  as if it never existed. Only `git log <prev-head>..<head>` shows it, and phase P reports it.

---

## 5. When validation runs

Driven by the skill's own `validation_requirement.json`:

- `required=false` → **do not run** `validate-kernel-pr`. The card says
  `Validation (deterministic): N/A — no runtime surface changed`. Not running it is not a defect.
- `required=true` → put the candidate targets in front of the human at the break; after they
  choose, the loop runs to the end without stopping.
- Environment prevented it (no idle GPU, missing runtime, host rebooted) → `NOT RUN — <reason>`.
  **That is an environment gap, never a PR defect**, and a missing report is never inferred from
  another target's result.

---

## 6. Termination criteria

Truth comes from files, never from an agent's summary.

| claim | what establishes it |
|---|---|
| the analysis happened | the seven `triage.py` gates exit 0 |
| a card finding stands up | a line in `refutations.txt` + `independent.txt`, written by the refuter, provenance-checked |
| runtime behaviour | `validation_report.json` whose `repo.head` equals the PR head |
| performance | `stages.perf` carries `median_ratio`; otherwise the card states why it skipped |
| **the tools did not lie** | **every executor finding also went through refutation** |
| the site is clean | scratch removed, no root-owned leftovers, nothing in the host `/tmp` |

### Why tool-honesty is a hard criterion

In one round the validator produced two signals that would otherwise have been attributed to the
author:

1. Its own GPU-picker import wrote `__pycache__` into the worktree under test; its own cleanliness
   check then read the tree as dirty and skipped every correctness and perf stage.
2. Its tolerance extractor discards the `atol=`/`rtol=` names and zips the remaining literals by
   position. The PR had reordered one call site from `(rtol, atol)` to `(atol, rtol)`, so an rtol
   was compared against an atol and reported as a widening. Like for like, every tolerance in that
   PR was unchanged or tighter.

Both reached a card before being withdrawn. S6b exists so that they are withdrawn automatically.
See `docs/upstream-findings.md`.

---

## 7. Environment traps

| trap | symptom | handling |
|---|---|---|
| no usable shell under a confined harness | cygwin `couldn't create signal pipe` | run the skills on the remote Linux host; keep the local side to the harness's own shell |
| host git cannot reach https | `schannel: SEC_E_NO_CREDENTIALS` | force the openssl backend, or use ssh |
| no `gh`, or `gh` unauthenticated | `fetch.sh` depends on it | use `tools/gh_shim.py` |
| diff over GitHub's 20000-line API cap | `gh pr diff` returns HTTP 406 | the shim produces the diff locally with `git diff <merge-base> <head>` |
| skill tooling reads files as the platform codepage | `MISSING-RULE-TEXT`, i.e. rules silently unexpanded | force `PYTHONUTF8=1` everywhere |
| CRLF checkouts | same as above | `core.autocrlf=false` for the review checkout |
| quoting through shell→ssh→shell | remote command mangled | write the script, copy it, run `bash <path>` |
| login banners | ssh exit codes unreliable | judge by stdout |
| PR stale against main | patch does not apply to the tip | base on `merge-base(main, head)`; note STALE on the card |
| submodule uninitialised | `fatal error: 'ck_tile/core.hpp' file not found` | initialise the pinned submodule — this produced one false BLOCK |
| container runs as root | root-owned artifacts block the next `git worktree remove` | clean from inside the container, then chown back |
| container may be stopped | a docker restart left it `exited` | check `docker ps -a` first; starting someone else's container is out of scope and must be reported |
| validator runs pytest with `-x` | the failure surface is under-reported (64 of 70 in one case) | say that execution stopped early, or re-run without `-x` to characterise a failure |
| the host can reboot mid-run | the report never appears | record `NO REPORT` with the reason; never infer from another target |
| GPU shared with co-tenants | a run can lose the GPU it claimed | gate on `rocm-smi --showpids` immediately before launch; a lost GPU is an environment gap |
| harness-side sandbox faults | ssh refused locally while the host is fine | the agent must fail loud; a silently skipped cleanup is a failed round |

---

## 8. Lessons from the live runs

### 8.1 Structured output silently discards finished work

31% of agents in one round returned nothing while having completed their work. → No schema for
workers; files are the durable contract.

### 8.2 The reviewed party will write the review record if you let it

A collection step returned empty, so zero refuters ran and the ledger was written empty — and the
card's author, told to "fix what the gate names", wrote the ledger itself with five plausible
`SURVIVED` lines. All seven gates went green on a fabricated record. → Fail closed on an empty
collection; **each refuter writes its own line**; check provenance by count and format; forbid the
worker from touching those files.

### 8.3 The severity bar must be fixed, or rounds are not comparable

The same finding — a public API removed with no shim — survived in one round and was killed in the
next, on identical facts. One refuter accepted a protocol documented in a README as evidence of a
consumer; the other demanded a consumer verifiable in the tree. Both are defensible; leaving the
choice open is not. → **A 🔴 requires an in-tree triggering point. A protocol documented outside the
tree caps the finding at ⚠️ with an action asking the author to confirm the downstream.**

### 8.4 Prior rounds leak

A run configured as "no prior" produced a refuter that cited the previous round's card, which sat
in a sibling directory. → Either isolate rounds or state on the card that priors were reachable.
Claiming independence while leaving them readable is the one unacceptable option.

### 8.5 A diff cannot show what happened between rounds

When the base advanced, a test the previous head had added and this head removed became invisible:
relative to the new base it had never existed. The commit message said the update addressed the
review; what it did was remove the test that exposed the problem. → Phase P must report
`git log <prev-head>..<head>`, separate the author's commits from merge traffic, and name every
test or guard added or removed.

### 8.6 A failed cleanup must be loud

One cleanup agent had every ssh refused by a local harness fault while ssh from elsewhere worked
fine. It reported the refusal — but a prompt that does not demand that would have let it read as
"nothing to clean". → Remote agents must stop and say so; a failed cleanup is a failed round.
