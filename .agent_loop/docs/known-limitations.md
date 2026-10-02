# Known limitations

Written so that nobody has to discover these by being misled. Each entry says what is missing, what
it costs, and what would fix it.

## Coverage

**The per-file assessment set is chosen by the caller, not derived.** `args.files` and
`args.ruleGroups` are filled in by whoever starts phase R, from the fetch report. On the runs behind
this package, 9 of 68 changed files were assessed. The `corefiles` gate derives the *backbone* set
from the diff and will refuse a card that ignores one of those, so a Tier-1/Tier-2 omission is
caught — but a Tier-3 omission is not. **Fix:** derive the file set from the diff with the same
tiering questions the gate uses, and let the caller only add to it.

**Rule partitioning is manual.** `ruleGroups` splits the derived rule ids into batches for parallel
adjudication. A rule that lands in no group is caught by the `ledger` gate, but the split itself is
hand-made. **Fix:** partition programmatically from `rules.txt`.

## Provenance

**The ledger check counts, it does not authenticate.** Phase R verifies that the number of
refuter-written files equals the number of ledger lines equals the number of findings, and that
every line is well formed. It cannot detect a party overwriting the *first line* of an existing
`F<nn>.md`. The separation-of-duties instruction forbids it and nothing observed has attempted it,
but the guarantee is procedural, not cryptographic. **Fix:** have each refuter record a hash of its
own file, and verify hashes at assembly.

**Cross-round isolation is not enforced.** Artifacts from a previous round sit in a sibling
directory and agents can and do find them: in one round a refuter cited the previous round's card in
its evidence, despite the run being configured as "no prior". Either isolate the rounds or state on
the card that priors were reachable — claiming independence while leaving them readable is the one
unacceptable option. **Fix:** per-round workspace roots, and a prompt-level prohibition.

## Operation

**Human answers are not persisted.** Targets, model roles and budget are re-asked every round;
the environment record is written but never read back by phase P (`envKnown` is passed in by the
caller). **Fix:** a `state.json` per PR that the break writes and phase P reads.

**No budget enforcement.** `policy.budget` is documented in the config and honoured by nothing. A
round is roughly 45 agents and 2.6M tokens; nothing stops a pathological run from doing far more.
**Fix:** count agents in the script and abort past the cap.

**Resume is partial.** `resume: true` skips refuters whose file already exists. Analysis outputs,
validation reports and the card are always redone. **Fix:** the same existence check for the S2
artifacts, which are already all on disk.

**Cleanup failure is a warning, not an error.** If a cleanup agent cannot reach a host, the run logs
it and still returns. The remote side is then left dirty for the next round, which discovers it as a
permission failure when removing a worktree. **Fix:** treat a failed cleanup as a failed round.

## Environment

**Windows hosts hit a long list of avoidable traps** — no usable bash under a confined harness,
`schannel` TLS failures for git over https, CRLF breaking the skill's own rule parsing, and
PowerShell quoting mangling remote commands. All are worked around in `loop.md` §7, and none of them
exist on a Linux host. **Running the loop from Linux removes most of that section.**

**One GPU per host is common and matters.** A host may expose a single GPU shared with other users'
long-running containers. The GPU gate (`rocm-smi --showpids`) is checked immediately before launch,
but nothing holds the GPU for the duration beyond the validator's own `flock`, and a co-tenant can
claim it mid-run. A run that loses the GPU should be reported as an environment gap, never as a PR
defect.

**A host can reboot mid-run.** It happened. The validation agent returns no report, and the card
records `NO REPORT` with the reason rather than inferring a verdict from another target. That is the
correct behaviour, but the round is then incomplete and must be finished by hand.

## Scope

**Validated against one repository and one PR.** Three rounds of the same PR, on two GPU
architectures. The loop contract is repo-agnostic, but nothing here has been run against a second
repository, and the skill CLI surface it depends on is pinned in `../VALIDATED-AGAINST.md` precisely
because that surface is still evolving.

**The card's five-finding cap is a readability limit, not a recall claim.** It comes from the
`review-pr` skill, which says so itself. Everything dropped is accounted for on the card and kept in
full in `findings.md`.
