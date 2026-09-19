# Validated against

This loop consumes the **CLI surface** of two skills in this repository. That surface is still
evolving, so a release pins what it was exercised against. If a pinned command or flag has moved,
the loop can still appear to work while drawing the wrong conclusion — check before trusting a run.

## Loop

    .agent_loop 0.1.0

## Skills

    ROCm/aiter .claude/skills @ the commit this directory was added on

**`review-pr` — the seven gates, all of which must exist and keep their argument order:**

    triage.py answers      <answers.txt>
    triage.py diagnostic   <ai_diagnostic.txt>
    triage.py corefiles    <core_files.txt> <pr.diff> <project-root>
    triage.py ledger       <rules.txt> <verdicts.txt> <pr.diff>
    triage.py card         <card.md> <verdicts.txt> <ai_diagnostic.txt> <answers.txt> <pr.diff> <late_findings.txt>
    triage.py refutations  <refutations.txt> <pr.diff> <card.md>
    triage.py independent  <independent.txt> <card.md>

    fetch.sh <PR> <owner/repo>            # honours REVIEW_AUTO_VALIDATE=0

Artifacts the loop reads by name from `fetch.sh`'s scratch dir: `pr.diff`, `pr_meta.json`,
`rules.txt`, `rules_expanded.txt`, `applies.txt`, `merge_target.txt`, `guards.txt`, `siblings.txt`,
`symbols.txt`, `twins.txt`, `test_quality.txt`, `kernel_tests.txt`, `ci_coverage.txt`,
`perf_claims.txt`, `struct_abi.txt`, `comment_only.txt`, `evidence.txt`,
`validation_requirement.json`, and the `head/` and `merge-target/` worktrees.

**`validate-kernel-pr` — the flags the loop passes:**

    validate_pr.sh --repo --patch --head-sha --target --label --out
                   [--expected-route] [--shape-argnames] [--shape-env] [--no-perf]

Report fields the loop reads: `verdict`, `findings[]`, `stages.*.status` and their notes,
`test_selection`, `execution_receipt`, `stages.correctness_repo_tests.stats`, `stages.perf`
(`median_ratio`, `worst_column`, `matched_rows`, `threshold`, `baseline_method`), `arch_coverage`.

## Pull request

    ROCm/aiter#4961, reviewed at three successive heads:
      round 1  head a65d5ffee51752b9051a76328631994dd4231248  base 24a62b1c122f23645a19b9d8b0abd4750c59359b
      round 2  head a05fe49e1c28d36b21419b543e62a89357ca2aaf  base f0321c0e8927d1d90a29385433f71e592b1c51f5
      round 3  head 1f24221fe2dfd3b36e742a628eee4ebc8abe1aea  base 226ee790953feedcc9d5e17323db0a019a177d23

    diff size 21.7k-22.0k lines, 68-74 changed files, 30-32 of 53 rules derived

## Hardware

    gfx950  (MI350X, 8 GPUs)      — correctness, policy, interface and JIT-cache targets
    gfx1250 (MI450X ES, 1 GPU)    — the CO-integration and split-K targets that skip on gfx950

## Measurements

    subagents over the whole exercise      211
    tokens                                  in 6.82M / out 5.86M
    one full phase R on a 21.7k-line diff   ~45 agents, in ~1.5M / out ~1.1M
    refutation cost                         3.7k-38.8k input per finding, median ~10k

    refutation kill rate, same-model refuter        18%   (2 of 11)
    refutation kill rate, cross-model refuter       40%   (7 of 17, 6 of 15)
    dry-run guard scenarios                         10/10

## What a real run produced

Round 2's validation on gfx1250 found a reproducible failure in the PR's own added test
(`correctness_repo_tests: fail`, all other stages passing), which a dedicated debugging agent traced
to a kernel id registered in Python but absent from the generated C++ dispatch table on any fresh
build — and whose one-line fix it verified on hardware. Round 3, on a head where that test had been
removed, produced no correctness failure. Both rounds independently withdrew validator artifacts as
non-defects. That is the behaviour this package exists to reproduce.
