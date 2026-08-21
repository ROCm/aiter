---
name: downstream-impact-check
description: Decide whether an aiter PR can break vLLM, SGLang or ATOM, by tracing the changed files through aiter's own imports to a module the downstream actually imports. Answers the reachability question with a scan instead of from memory, so the label-gated downstream CI is requested when it is needed.
argument-hint: <PR number>
---

# Downstream impact

aiter's downstream tests are **skipped by default** and only run when a `ci:*` label is added. A
PR that changes something vLLM or SGLang depends on can pass every aiter check, merge green, and
surface days later as a downstream incident.

The judgement that gates all of this — *is this reachable from a consumer?* — is normally made
from memory. It is greppable, and memory gets it wrong in the direction that hurts.

---

## The incident this is built from

aiter#4530 changed one file: `aiter/ops/triton/_triton_kernels/moe/moe_routing/topk.py`. Kernel
internals. No public signature touched. Nothing a "did the API change?" review would stop.

It broke vLLM's gpt-oss MXFP4 MoE on MI355 — `test_gpt_oss_attention_quantization` — after vLLM
bumped aiter `0.1.16.post5 → 0.1.19` (vllm#49361, the bump whose test plan was blank). vLLM
shipped a hotfix (vllm#50859) naming aiter#4530 as the companion fix. In the same week SGLang
reverted its aiter pin from `v0.1.19` back to `9127c94` (sglang#32879).

One internal kernel edit, two downstreams disrupted, and the aiter PR looked local.

## The check

```bash
.claude/skills/downstream-impact-check/scan_downstream_consumers.py \
    --diff "$WORK/pr.diff" --aiter . --root /path/to/vllm --root /path/to/sglang
```

It walks aiter's own imports **upward** from every changed file and tests each module in the
closure against the downstream checkouts, stopping at the first module a downstream imports.
Replaying #4530 it reconstructs the chain:

```
aiter/ops/triton/_triton_kernels/moe/moe_routing/topk.py
  <- aiter/ops/triton/moe/moe_routing/topk.py
  <- aiter/ops/triton/moe/moe_routing/routing.py
  <- aiter/ops/triton/moe/moe_op_gemm_a16w4.py
  -> imported by vLLM aiter_mxfp4_w4a8_moe.py
```

and reports no reachability for SGLang, which is correct — SGLang does not import that subtree.

**Do not ask "did a public symbol change".** The first version of this script asked exactly that
and reported *no downstream impact* for #4530, because its public signature never moved. The
question is reachability, not signature.

## Reading the output

| output | what it licenses |
|---|---|
| `REACHED <module>` with a chain | Ask for the downstream CI label. Name it from the reached module's consumers, and confirm the current `ci:*` definitions in `.github/workflows/` first — the model roster rotates and a stale mapping produces a confidently wrong recommendation. |
| nothing reachable | Reads as no impact **for the checkouts you passed**, through **static imports only**. Dispatch by string, env var or entry-point registry will not appear. |
| `NO DOWNSTREAM CHECKOUT AVAILABLE` | Unmeasured, not absent. Never record it as "no downstream impact". |

## What still needs a human

The scan establishes reachability. It does not establish *harm* — a reached op whose behaviour is
unchanged is fine, and a change behind an arch gate the downstream never hits is fine. Two
judgements stay with the reviewer:

- **Does the behaviour actually change for the reached caller?** Numeric output, dtype, arity,
  default dispatch. A pure-additive arch-gated path is exempt.
- **Is a version bump involved?** vllm#49361 was a bump with an empty test plan, and the incident
  above followed it. A bump PR whose test plan is blank is the highest-risk shape in this
  repository's downstream history; ask for the plan before the label.

## Limitations, stated

- Static imports only, and only against checkouts you provide. There is no coverage of ATOM here
  because no ATOM checkout was available when this was written.
- Depth-limited to 5 import hops.
- Reachability is measured against the downstream's *current* main, not the version pinned to the
  aiter release under review. A consumer that has since dropped the call will read as unreachable.
- Validated on one incident (#4530 → vllm#50859 / sglang#32879) plus one negative (SGLang, where
  the correct answer is "not reachable"). One positive and one negative is thin; treat the
  chain output as evidence and the label recommendation as a suggestion.
