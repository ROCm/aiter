---
name: review-pr
description: Advisory AI code review for aiter and FlyDSL PRs. Catches perf regressions, silent correctness bugs, dispatch gate holes, and AI-generated code patterns, but never acts as a merge gate. Invoke with a PR number (optionally owner/repo#N) and, when one exists, a validation report path. Step 1 triages whether the PR changes runtime surface at all and, when it does and the PR ships a single test target, runs validate-kernel-pr itself; a PR with no runtime surface is reported N/A rather than unvalidated. That run also times the target on base and head back to back on one locked GPU, so a kernel PR's latency is measured rather than assumed. The review line stays advisory; deterministic correctness and perf results are judged only from a head-matched report.
argument-hint: <PR number> [owner/repo] [validation-report]
---

# aiter PR Review — advisory tier

This skill supplies hints to a human reviewer. Its judgement is stochastic and never blocks a
merge. Only a reproducible blocker from an explicitly supplied, head-matched
`validation_report.json` may be used as a deterministic gate.

## Promotion bar

This skill is advisory now and stays advisory until both conditions below hold. Neither holds today,
so no part of it may gate a merge.

- **False clearance is measured and near zero for every family that raises a red verdict.** The
  number that matters is not recall, and not a spot check: it is the rate at which the tool reports
  nothing wrong when something is wrong. No committed replay corpus establishes it, so that number
  does not currently exist.
- **The judgement relied on is not an LLM's.** An LLM judgement never gates a merge, whatever its
  measured accuracy. Only a reproducible blocker carried by a head-matched `validation_report.json`
  may gate, because the report ships its reproducer with it.

Until then `🔴 HIGH RISK` requests human attention and nothing more. Whoever proposes a rule edit as
an improvement, or proposes letting this tool gate a merge, owns building the corpus and measuring
against it. This bar lives in the header rather than in an issue because a header is read on every
use and an issue sinks.

---

## Step 1 — Fetch

```bash
# Everything Step 1 does is in fetch.sh; read its output, then keep the printed $WORK.
"$(git rev-parse --show-toplevel)/.claude/skills/review-pr/fetch.sh" "$@"
```

Read the diff and PR body before proceeding.

### Step 1b — Derive the applicable rules, and collect the evidence they need

**Step 1b writes its artifacts into `$WORK`. Read them; each explains its own output,
so what follows is the map, not the manual.**

| file | what it answers | the trap it exists for |
|---|---|---|
| `rules_expanded.txt` | the full text of exactly the rules this diff derives | reading all 51 means attending to none |
| `evidence.txt` | how a removed guard is handled on head | prose telling you to grep was read and not acted on (aiter#5143) |
| `symbols.txt` | first-party imports that do not resolve against the merge target | a **rebase** signal, not invented code — #4994's import was valid when written |
| `twins.txt` | which existing file each new file was copied from | the defect is the *asymmetry* between them, not the copy |
| `test_quality.txt` | assertion count, tolerances, shapes of added tests | zero assertions may mean a helper asserts — read before firing |
| `ci_coverage.txt` | whether a CI job will ever run the added tests | HK6 is satisfied by a file in a directory nothing scans |
| `perf_claims.txt` | every number claimed, and which name no baseline | a signed delta and a `before \| after` table already carry theirs |
| `struct_abi.txt` | structs whose pinned layout this diff shifts | the assertions exist to force a code-object rebuild |
| `comment_only.txt` | the non-prose lines of a comment-dominated diff | 7963 lines that reduce to none (aiter#4062) |

A `SKIPPED:` or empty artifact means that axis was **not checked** — say so rather than
reading silence as clean.


**Read `$WORK/rules_expanded.txt` — it is the full text of exactly the rules in
`$WORK/rules.txt`, and it is the rule list for this review.** They are derived from paths, added/deleted
lines and the title, and the derivation is conservative — a family it cannot decide
structurally is included, never dropped. Over 597 open PRs every one matched at least one
family, and no family fired on more than half of them.

**Cross-file verification — `$WORK/evidence.txt` already holds it; read that file before
writing any finding about a removed guard or a changed signature.** The diff shows changed
lines, not the whole story, and prose telling you to go and grep is not enough: it was in
this skill already and was read and not acted on, producing a `q_out is not None` finding on
aiter#5143 that was withdrawn once head was read (`q_out` is `std::optional` on both sides,
every call site is `has_value() ? data_ptr() : nullptr`, and the kernel guards
`if(is_q && q_out != nullptr)`). The collector puts those three lines in front of you.
Where it produced nothing, grep the *entire* symbol family yourself — `.cu` + `.cuh` + `.h`
together, since sync/fence/atomics or the other half of a scatter often live in the header
(aiter#3802: a "kernel has no sync" finding was false, the barrier was in the `.cuh`;
aiter#4098: "compares raw uint8 vs float" was false, the reader had a conditional
`maybe_view_fp8()` the diff never showed).

**Classify every CI failure before blaming the PR.** A red check is not automatically the PR's fault:
- Read the failed *step*. `check-signal` / "Wait for Checks" timeouts, "Expected exactly one wheel artifact", and dep-resolver noise are **infra flakes**, not code failures (aiter#3593, #4171).
- Compare against main: if main fails the same shard in the same window, it's baseline/flaky, not a regression introduced here.
- Expired logs (`HTTP 410 Gone`) on old runs mean the failure is months-stale and meaningless against today's main — ask for a rebase + fresh run instead of quoting it (aiter#2565).

---

## Step 2 — Semantic Understanding (answer all 5 before rules)

Work through these by reading the diff, not the description alone, and **write the answers
into `$WORK/answers.txt`, one line each, `Q1:` … `Q5:`** — Step 8 will not let you write a
card without them. Inline `_Answer:_` blanks left nothing behind: a review that skipped
Step 2 was textually identical to one that did it, which is the same hole D9 had before its
scan moved into Step 1.

**Q1 — What specifically changed computationally?**
Not "improves perf" — what algorithm/formula/data flow changed?
_Answer:_

**Q2 — Hardware scope: which arch(es), precision(s), execution phase(s)?**
gfx942 / gfx950 / gfx1250? fp16/bf16/fp8? decode / prefill / both?
_Answer:_

**Q3 — Does this change any public aiter API?**
New symbol in `aiter/ops/*.py`, new kwarg on existing op, change to `aiter/__init__.py`?
_Answer:_

**Q4 — Performance claim: what is the mechanism?**
Not "faster" — WHY is it faster? (fewer memory round-trips, fewer kernel launches, better tiling?)
_Answer:_

**Q5 — Does the description explain WHY or only WHAT?**
"Fuses kernels for speedup" = surface. "Eliminates intermediate HBM write between rmsnorm and quant" = understanding.
If surface-level only → treat as elevated AI-code risk.
_Answer:_

---

## Step 3 — PR Type Classification

**Step 1b already derived this.** Read `$WORK/rules.txt` and work only those families;
`$WORK/rules_expanded.txt` holds their text. Do not classify by hand — the failure mode was
self-application: a model asked to tick 20 types over 44 rules attends to none of them.

The family → rule mapping lives in `MAPPING.md`, generated by `triage.py mapping` so it
cannot drift from the deriver. The hand-written copy that used to sit here had: it claimed
D9 was derived (D9 is scanner-backed and deliberately is not) and omitted 21 rules that are,
including every Triton rule. `tests/` fails if the committed mapping and the code disagree.

Measured over 600 open aiter PRs: median 15 rules against a 49-rule set (31%), every PR
matched at least one family, no family fired on more than half, and the expensive full
Step 4 assessment fires on 6% rather than on everything touching `aiter/ops/`.

---

## Step 4 — Core File Risk Assessment

**Write one line per backbone file this diff touches into `$WORK/core_files.txt` — Step 8
gates on it.** This step was prose only, so a review that skipped it was textually identical
to one that performed it — the hole Step 2 had before `answers.txt`. Format:

```
<path> TIER1|TIER2|TIER3 COVERED|GAP|N/A -- <reason naming what THIS PR changed>
aiter/fused_moe.py TIER2 COVERED -- num_local_experts is threaded through to moe_sorting_fwd and op_tests/test_moe.py adds a DSv3 TP=8 decode case for it
aiter/__init__.py TIER1 GAP -- the new `from .ops.gemm_op_a4w4 import *` sits above the rest of the block, so an ImportError inside it truncates the namespace silently
```

`COVERED` = the blast radius below is exercised by this PR's tests or is unreachable from
the change; `GAP` = it is not, and that goes on the card as a finding; `N/A` = the change
cannot reach it at all (comment, docstring). Touching no Tier 1/2 file: write one
`NONE -- <reason>` line naming the Tier 3 files it does touch.

The gate rejects: a backbone file with no line (`UNASSESSED`); a tier recorded below the
table's (`TIER-MISMATCH` — downgrading is not a way past the checks a tier requires); a
formulaic or sub-30-character reason (`NO-EVIDENCE`); a reason naming no file or symbol
this PR changes (`UNANCHORED` — "core file, large blast radius" is equally true of every
PR ever opened against that file); a line about a file the diff does not contain
(`UNTOUCHED-FILE`); `NONE` declared while a backbone file is present (`UNDECLARED-CORE`).
A missing `core_files.txt` is a hard failure.

**What makes a file "backbone"?** Apply these three questions to any file in the diff — including new files not in the table below.

```
Q1 — If this file has a syntax error or fails to import, does `import aiter`
     still succeed?  → NO → Tier 1 (system-critical: aiter itself breaks)
Q2 — Does it hold the Python dispatch that selects which kernel runs for an op
     class, AND is that op used by >1 production model family (DSv3, Kimi…)?
     → YES → Tier 2 (op-class critical: wrong result for ALL users of that op)
Q3 — Is it the public aiter API for an op (`from aiter import X` lands here)?
     → YES → Tier 2 (signature change silently breaks all consumers)
Otherwise → Tier 3 (individual kernel or model-specific code).
```

The table is the snapshot the gate demands a line for; use Q1/Q2/Q3 on new files and add a
line for any you judge Tier 1 or 2. Ranked by commit frequency (2025–2026), blast radius:

| Tier | File | Git commits | Blast radius | Failure mode |
|------|------|-------------|-------------|--------------|
| **1** | `aiter/jit/core.py` | 182 | **ALL ops** — JIT compilation engine | Any import of aiter fails; zero ops load |
| **1** | `aiter/__init__.py` | 52 | **ALL** vLLM/SGLang/ATOM users | `ImportError` or silent namespace truncation below broken import |
| **2** | `aiter/fused_moe.py` | 119 | All MoE models (DeepSeek, Kimi, MiniMax) | Wrong expert routing, silent accuracy drop |
| **2** | `aiter/ops/mha.py` | 89 | All MHA attention paths | Wrong attention output, crash |
| **2** | `aiter/ops/attention.py` | 66 | MLA/paged attention dispatch | Wrong KV, accuracy drop |
| **2** | `aiter/ops/gemm_op_a8w8.py` | 59 | All FP8 quantized GEMM | Wrong matmul result, silent accuracy drop |
| **2** | `aiter/mla.py` | 57 | All MLA decode/prefill (DSv3/Kimi) | Wrong KV, accuracy drop, crash |
| **2** | `aiter/tuned_gemm.py` | 52 | All GEMM-backed ops | `assert False` crash or silent fallback to slow path |
| **2** | `aiter/ops/moe_op.py` | 51 | MoE op dispatch table | Wrong dispatch, wrong expert weights |
| **2** | `aiter/ops/quant.py` | 49 | All quantization paths | Wrong scale, silent accuracy drop |
| **3** | `aiter/ops/*.py` (a single op's wrapper), individual kernel `.py`/`.cu` | varies | Consumers of that one op | `AttributeError` at call time in downstream |

**Why `aiter/ops/*.py` is Tier 3 and not Tier 1**, and what happens to the assessment if it is not: `rules.md` § Tiering.

**`aiter/__init__.py` special rule**: The import block must NOT be wrapped in try/except.
Any new import added here → check the imported module for bare `ImportError` paths that
could silently truncate the namespace.

**`aiter/jit/core.py` special rule**: This file bootstraps the entire JIT compilation pipeline.
A syntax error, wrong default, or broken env-var handling here means zero aiter ops load.
Changes here require e2e smoke test across all GPU arch targets.

**What the reason has to answer** — this is what makes a `COVERED` checkable.
For **Tier 1** files (`jit/core.py`, `__init__.py` — these two only):
- Every public symbol changed, and its callers across aiter itself (`grep -rn '<symbol>' aiter/`). A caller not covered by the PR's test is a `GAP`.
- For `__init__.py`: does the new import have a bare `ImportError` path that could silently truncate the namespace?
- For `jit/core.py`: is there an e2e smoke test that loads all kernels on gfx942 AND gfx950 after this change?
- If this change is wrong, what breaks and how would it be detected? (all ops fail / one op family fails / silent wrong value)

For **Tier 2** files (fused_moe, mha, attention, gemm, mla, tuned_gemm, quant):
- Which model families (DSv3, Kimi, MiniMax, GLM…) use this op? Is at least one from each family in the test?
- Are production shapes tested? At minimum: decode (M=1, TP=4/TP=8) AND prefill (ISL=4096, TP=4/TP=8).
- Does the change affect gfx942 only, gfx950 only, or both? If both, are both arch paths tested?

**AI code red flag — verbatim duplication across backbone files:** Same algorithm copy-pasted into 2+ backbone files with only variable names changed. See D5.

---

## Step 5 — Rule Checklist

**Adjudicate every rule in `$WORK/rules.txt`, one line each, into `$WORK/verdicts.txt`.**
The derivation already cut the list to what this diff can actually trigger — 12 rules at the
median over 597 PRs, 14 on the Triton subset — so there is no rule here you may pass over
because the list looked long. Format, one per rule id:

```
<RULE-ID> FIRE|CLEAR|N/A — <the specific reason, naming file:line, symbol, or the condition>
```

`CLEAR` means you looked and it does not apply *to this diff*; it is a claim, and the reason
is what makes it checkable. "ok", "n/a", "fine" are not reasons — Step 8's gate rejects them.

Step 8 will not let you write a verdict card until every derived rule has a line with a
reason. This is the same move as running the D9 scan inside Step 1 rather than asking for it
mid-checklist: on a 14-PR controlled run the revised D9 prose caught 0 of 3 known overflow
defects and the scanner it names was never once invoked. A checklist a reviewer marks off to
itself decays under load, silently, and the queue ahead is large.

Six failure categories — work all six in order. Advisory severity per finding:
🔴 high risk / ⚠️ should fix / 📝 note. These labels prioritize human attention; they do not
themselves gate a merge.

**🔴 evidence threshold — before firing any 🔴, write down the concrete input that triggers it.** Name the specific shape / scale / dtype / arch / value that makes the finding fire (e.g. "at `token_id` > 16M with H=32, D=128 the int32 product exceeds 2^31", or "when `arch=='gfx1250'` with fp4 input the branch assumes fp8"). If you cannot state a concrete triggering case, the 🔴 is unproven — **downgrade to ⚠️ ("worth checking") or drop it.** A 🔴 that reads as a definite defect but names no demonstrable triggering input is exactly how a false positive lands on a maintainer's PR. This threshold applies to every rule below — including those whose own text omits an explicit FP self-check (e.g. D9): the same index expression is safe in a capped/small-batch kernel and unsafe only at a scale you must actually exhibit.

| Category | Core question | Key triggers |
|---|---|---|
| **A. Coverage gaps** | Same bug elsewhere? Same code other configs? | `_opt`, `_prefill_opt`, `_v2`; shared path; broad `if` condition |
| **B. Silent bypass** | Does every input reach the right branch? | gated-off param; string alias; non-aligned dim; proxy metric |
| **C. Hardcoded arch/dtype** | Does the constant break on another GPU or fp8 flavor? | `240.0`, `448.0`; arch name for fnuz; `bf16` fixed |
| **D. Uninitialized state** | Is the buffer clean before atomic/kernel launch? | `::empty()`+`atomic_fmax`; `fill_(0)` missing |
| **E. Cross-repo sync** | Does the consumer know about this change? | new aiter symbol; default-preserving new param; plugin bridge |
| **F. Resource duplication** | Does the change double GPU memory silently? | new `_preshuffled`/`_quantized` weight alongside original |

---

**The rule bodies are not in this file.** `$WORK/rules_expanded.txt`, written by Step 1b,
holds the full text of exactly the rules this diff derives — read that. All 51 blocks live in
`rules.md`; the median PR needs 103 of their 383 lines, so keeping them here loaded 280 lines
of irrelevant rule text into every review, on top of a skill that is already long enough that
"read all of it" is a hope rather than a guarantee. Cutting *which rules you are told to
check* from 44 to 12 while still shipping all 44 rule texts was half a fix.


## Step 6 — AI Code Diagnostic

For each question below, note if the answer is a warning sign:

| Question | Warning sign |
|----------|-------------|
| Does description explain mechanism (WHY) or just action (WHAT)? | Only WHAT → elevated risk |
| Are perf numbers suspiciously clean? (exact 2.0x, 1.5x, 3.0x) | Could be cherry-picked or fabricated |
| Are perf claims only trace screenshots with no numeric values? | Screenshots ≠ numbers; reviewer will ask |
| Does the test only cover M=1 or M=16? | AI defaults to toy shapes |
| Are gated-off parameters asserted or silently ignored? | Silent → B1 violation |
| Does code introduce `sys.path`, `os.environ` mutations at module level? | Global state leak → HK3 |
| Were unrelated files committed alongside the actual change? | AI commit artifact → HK2 |
| Is the new default path revertible? | No env-var gate → D2 violation |
| Is "Test Plan" / "Test Result" section left as template comment? | Empty = untested, AI-generated description |
| PR description footer says "🤖 Generated with Claude Code" or similar AI attribution? | Author may not understand the change — elevated manual review priority |

**Structural verification — the table above is a cheap pre-filter; a clean description does not make the code correct.** When the diff touches code, AI fails in specific *structural* ways. Run these checks and report each as a finding tagged `[verified]`/`[inferred]`, ending in an action verb (per the finding format below):

1. **Unresolved imports — `$WORK/symbols.txt` already holds this.** Step 1b resolved every
   first-party import the diff adds against the branch this PR merges into. Read that file;
   do not redo it by hand. Each line is a *rebase* signal, not an accusation of invention
   (aiter#4994's import was valid when written; the module was deleted from main 19 hours
   before it merged). For symbols the static sweep cannot judge — a new kwarg on an existing
   call, an attribute, an enum member — grep those specifically; that residue is what is
   left for you, not the imports.
   → `🔴 [symbol] on [line] not found in [module] / signature mismatch — confirm it exists or it is a hallucinated API`
2. **Twin divergence (copy-paste half-adapted).** Identify mirrored code — fwd/bwd, v2/v3, prefill/decode, gfx942/gfx950. Compare field by field; any asymmetry (one side int64 the other int32, one masked the other not, one stride order flipped) is an unfinished copy. This is the signature AI kernel bug (cf. D9's fwd-int32/bwd-int64 case).
   → `🔴/⚠️ [detail] differs between [twin A] and [twin B] — copy-paste left [side] unadapted`
3. **Claim/comment ↔ code, and number provenance.** Does the code actually enforce the invariant the description or a comment asserts? Then take the single most impressive number in the PR and trace it to its source (script output / log line). A number or PR/issue citation you cannot trace is `[unverified]` — never repeat it as fact. (This skill's own P5 once shipped a fabricated "1.14x" for aiter#4166 that the PR never claimed — verify, do not trust.)
   → `⚠️ [claim/comment] asserted but code does [X]; or [number] not traceable to any output — mark [unverified] and ask for the source`
4. **Safety theater.** For each new `if`/`try`/`assert` guard: is it reachable, will it ever fire, does `except: pass` swallow a real error? AI adds defensive code that is unreachable or silently hides failures.
   → `⚠️ guard on [line] is [unreachable / swallows errors] — remove it or make it actually enforce the invariant`
5. **Test calibrated to pass, not to falsify.** Is the reference impl structurally a twin of the kernel (the same bug lives in both, so they always agree)? Is `atol`/`rtol` loosened with no justification? Does it assert against the kernel's own output? AI writes tests that pass rather than tests that could catch a regression.
   → `⚠️ test [name] cannot fail because [mirrored ref / loose tol / self-comparison] — replace with an independent oracle`
6. **Magic constant without derivation.** A new tile size / threshold / epsilon / literal — is there a stated derivation or tuning basis, or does it merely look plausible?
   → `📝 constant [value] on [line] has no stated derivation — ask for the tuning/source basis`

**Write one line per check into `$WORK/ai_diagnostic.txt`** — `1:` … `6:`, each naming what
you looked at and what you found (`clean` alone is not an answer). Step 8 will not let you
write a card without them. These six are where a diff that reads well hides its defects, and
prose asking for them left nothing behind: a review that skipped all six was textually
identical to one that ran them.

If 3+ table signs OR any structural check above fires: note "elevated AI code risk — recommend thorough manual verification of the dispatch logic and test coverage." Regardless of the table count, when the diff changes code the structural checks are mandatory — a clean, well-written description is itself something AI produces easily.

---

## Step 7 — Free-Form Review

After the rule checklist, read the diff as a domain expert:
- Does the approach make sense given the hardware constraints?
- Are there correctness concerns not caught by the rules above?
- LDS limits: gfx942 = 64KB, gfx950 = 64KB per CU. gfx1250 (RDNA4) has 320KB LDS per CU but `ds_read`/`ds_write` immediate offset is only 16-bit (max 65535 = 64KB). If LDS allocation exceeds 64KB on gfx1250, the compiler uses VGPRs for the LDS address → VGPR spill → perf regression or compile failure. Real example (PR#4031): reviewer caught OPUS kernel on gfx1250 would hit this.
- For new Triton kernels: BLOCK_SIZE choices, num_warps, num_stages — are they reasonable for MI300X? Large BLOCK_SIZE can push LDS over limit causing test failures (Real example: PR#3808, 10 LDS-exhaustion failures in Triton batched GEMM configs).
- `.contiguous()` before kernel calls when tensor may have non-standard strides?
- For mixed FP8 dtype paths (fn vs fnuz): gfx942 KV cache is fnuz by default, but Q quantization may emit fn (e.g., DSv4 Flash fused indexer). A kernel handling mixed fn/fnuz inputs needs explicit dtype dispatch — silent dtype mismatch compiles but produces wrong values. Real example (PR#3913): reviewer asked "why is there a mixed FN/FNUZ path?" and asked for `if arch == "gfx942":` guard on the fnuz *conversion* path.
- For FlyDSL/assembly kernels: hardware tile size constants (MFMA M=16, N=16, K=32 for MI300X FP8) should be named constants, not raw magic numbers (16, 32) scattered across the kernel. Real example (PR#3913): vpietila asked "add named constants MFMA_M=16, MFMA_N=16, MFMA_K=32 and use them throughout."

---

## Step 7.5 — Blind-Spot Check

Before writing the verdict, answer this one question in full:

**"Is there any correctness risk, resource hazard, or behavioral edge case in this diff that none of Steps 1–7 above caught?"**

Append it to `$WORK/answers.txt` as a `BLIND:` line. If the answer is yes, add it to the
findings. A bare "no" is rejected: it costs nothing to write and carries nothing, so say
what you looked for and did not find.

## Step 7.6 — Refutation

**Try to kill each finding before reporting it.** One line per attempt in
`$WORK/refutations.txt`: `RED|WARN|NOTE SURVIVED|KILLED -- what you opened and what it said`.
`rules.md` § Refutation says what to attack; a killed finding never reaches the card.

---

## Step 8 — Verdict

**Before writing the card, run the four gates below. A red gate means that step did not
happen; go back to it rather than reporting.**

```bash
"$SKILLS_ROOT/review-pr/triage.py" answers "$WORK/answers.txt" || {
  echo "Step 2 / 7.5 not answered — the card must not be written yet" >&2
  exit 1
}
"$SKILLS_ROOT/review-pr/triage.py" diagnostic "$WORK/ai_diagnostic.txt" || {
  echo "Step 6 structural checks not recorded — the card must not be written yet" >&2
  exit 1
}
"$SKILLS_ROOT/review-pr/triage.py" corefiles "$WORK/core_files.txt" "$WORK/pr.diff" || {
  echo "Step 4 not performed for every backbone file — the card must not be written yet" >&2
  exit 1
}
"$SKILLS_ROOT/review-pr/triage.py" ledger "$WORK/rules.txt" "$WORK/verdicts.txt" \
  "$WORK/pr.diff" || {
  echo "rule pass incomplete — the card must not be written yet" >&2
  exit 1
}

# Then write the card to $WORK/card.md and run the last two gates against it. The four gates
# above check that the work happened; this one checks that the card reports THAT work.
"$SKILLS_ROOT/review-pr/triage.py" card "$WORK/card.md" "$WORK/verdicts.txt" \
  "$WORK/ai_diagnostic.txt" "$WORK/answers.txt" "$WORK/pr.diff" || {
  echo "a finding in the card is not nailed down — fix or drop it before reporting" >&2
  exit 1
}
"$SKILLS_ROOT/review-pr/triage.py" refutations "$WORK/refutations.txt" "$WORK/pr.diff" \
  "$WORK/card.md" || { echo "a finding reached the card unattacked" >&2; exit 1; }
```

The first names any of Step 2's five questions or Step 7.5's blind-spot question left
unanswered (`UNANSWERED`) or answered with a formula rather than an answer
(`NO-SUBSTANCE`). The third names every Tier 1/2 backbone file in the diff with no
assessment line (`UNASSESSED`), and every line whose reason is formulaic (`NO-EVIDENCE`)
or anchored in nothing this PR changes (`UNANCHORED`); it computes the backbone set from
the diff itself, so a missing or `NONE`-declaring artifact does not get past it. The
fourth lists each derived rule with no verdict line (`UNADJUDICATED`), each verdict with no
reason (`NO-EVIDENCE`), and each `FIRE` whose reason cites a file this PR does not change
(`UNTOUCHED-CITATION`) — a defect claimed against untouched code is a stale or invented
citation. `CLEAR` may cite anything: "E5 CLEAR: `aiter/__init__.py` unchanged" is correct
*because* that file is absent from the diff. The gate fails closed: an empty or missing `rules.txt` is the state of
a run whose Step 1b never happened, so it exits non-zero rather than waving the run through.
The gate checks that each rule was answered, not that the answer is right — it closes the
cheapest failure, which is not answering.

The card gate asks one question per finding: **is this nailed down?** It runs in both
directions. Nothing on the card may be untraceable: not a file the PR does not change, not a
rule no verdict marked `FIRE`, not a 🔴 naming neither a value nor two code identifiers — the
red threshold above, finally enforced. And nothing adjudicated `FIRE` may quietly not reach
the card: report it, change the verdict, or write `-- not reported: <reason>` on the verdict
line. The 5-finding cap is a real reason; leaving it out silently is not one.

Write the card to `$WORK/card.md` before running the gate — a missing file is a hard
failure, because not writing it was the cheapest way past this check. Whether a finding is
*correct* is not checked, and is what a human reads the card for.

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

## Adding New Rules

When a human reviewer catches something real that this skill missed:
1. Add the rule body to **`rules.md`**, with a real PR example as evidence
2. Add its id to the family in `triage.py` that the diff would trigger — a rule no
   derivation emits is never read, and `tests/test_review_skill.py` fails on a rule id the
   deriver can emit with no body in `rules.md`
3. Commit with message: `review-pr: add R[N] from PR#[NNN] — [one line description]`

The skill grows from real review history, not hypothetical patterns.

**Nothing new goes in this file.** SKILL.md is budgeted at 500 lines and holds only the
prose every review needs; `tests/test_review_skill.py` fails past it, and fails on any code
fence over 30 lines. That budget exists because this file went 487 → 632 → 1372 lines in
under two months while its instruction content barely moved: at 1210 lines, 825 were shell
and 299 were prose. Conditional content goes in `rules.md` and reaches the reviewer through
`triage.py expand`; executable content goes in a script and gets called, like `fetch.sh`.
Raising the budget is a decision to make this file less likely to be read in full — make it
in a commit that says why.
