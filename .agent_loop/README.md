# .agent_loop

An orchestration layer that drives the two review skills in this repo — `.claude/skills/review-pr`
and `.claude/skills/validate-kernel-pr` — as one repeatable loop, so that a PR goes from a number
to a gated verdict card plus deterministic on-GPU evidence without a human holding it by the hand.

It is **not** a third skill. The skills decide *what* a review means; this decides *who runs what,
in which order, and what may not be believed*. It calls their CLI surface and nothing else.

```
[once per host] environment probe → human confirms → env record
[per PR]  phase P (prefetch): verify env + skill Step 1 fetch        ← cheap, 2 agents
             ↓  ★ the single human break: targets, models, budget — asked once
          phase R: analysis fan-out → independent refutation → card → seven gates
                   → on-GPU validation → refute the executor → refold → cleanup   ← no stops
```

## Why a loop rather than one review pass

Three things only a loop can do, each of which happened during the runs this was built from:

- **It re-reviews the same PR as it moves.** When the base advances, a file the previous head added
  and this head removed is invisible in the diff — it reads as if it never existed. Phase P reports
  `git log <prev-head>..<head>` for exactly this reason.
- **It puts the tools on trial too.** A validator finding is a claim, not a fact. Stage S6b refutes
  the executor's own findings; on the runs behind this package that withdrew four false ones,
  including a "comparison tolerance widened" verdict that was an artifact of positional matching.
- **It makes the review record falsifiable.** Every finding on the card was attacked by a fresh
  agent that never saw the reasoning behind it, and that agent — not the card's author — wrote the
  ledger line recording the outcome.

## Quick start

```bash
cp config/loop.example.json config/loop.json && $EDITOR config/loop.json

node tools/dryrun.mjs            # zero-token: proves the guards fire. 10/10 or do not proceed.
```

Then, from an agent harness that provides a `workflow` tool (the scripts are workflow script
bodies, not standalone programs):

1. run `workflows/prefetch.workflow.js` with `args` built from your config plus `{ pr, sinceHead }`;
2. **stop and ask a human** the four questions phase P's report makes answerable — which validation
   targets (the skill refuses to pick when several candidates exist), which model per role, the
   refutation budget, and whether the environment record still holds;
3. run `workflows/run.workflow.js` with those answers. It does not stop again.

## What it guarantees, and what it does not

**Guarantees.** Every finding on the card was attacked by an independent agent that did not see the
analysis; the ledger recording that was written by the refuters themselves and is checked for
count and format before a card can be certified; the seven gates are judged by the skill's own exit
codes, never by a model's say-so; deterministic claims come only from a `validation_report.json`
whose `repo.head` matches the PR head; a run that cannot establish any of this aborts rather than
producing a green result.

**Does not guarantee.** That the findings are *correct* — that is what a human reads the card for.
That the review is *complete*: the set of files given to the per-file assessment is still chosen by
the caller. That the executor never produces false signals — hence S6b. See
[docs/known-limitations.md](docs/known-limitations.md), which is deliberately specific.

## Layout

| path | what |
|---|---|
| `loop.md` | the contract: roles, artifact layout, eight stages, gates, termination, environment traps, and the lessons behind each guard |
| `workflows/prefetch.workflow.js` | phase P |
| `workflows/run.workflow.js` | phase R |
| `tools/dryrun.mjs` | local harness that stubs the workflow hooks and asserts every guard fires — no model calls |
| `tools/gh_shim.py` | drop-in `gh` for hosts without it, or for a diff over GitHub's 20000-line API cap |
| `tools/agent_cost.py`, `dump_subagent_log.py`, `list_run_agents.py` | after-the-fact inspection: token accounting, readable transcripts, and which agents a run actually started |
| `config/loop.example.json` | every installation-specific value, with the reasoning inline |
| `docs/known-limitations.md` | what is still broken or unverified |
| `docs/upstream-findings.md` | four reproducible defects this loop found **in the skills it drives** |
| `VALIDATED-AGAINST.md` | the skill commit, PR heads, hardware and measurements this was validated on |

## The one number worth knowing

The refuter must not be the same model as the worker. Measured over three rounds of the same PR:
a same-model refuter killed **18%** of findings; a cross-model refuter killed **40%**, including
four of six `RED`s in one round. The false positive that a same-model round shipped to the card —
and that had to be withdrawn by hand — was caught automatically once the refuter changed family.
