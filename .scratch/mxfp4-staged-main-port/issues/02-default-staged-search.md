# 02: Deliver complete default-environment staged search

**What to build:** A supported default-environment invocation runs staged search from the main-compatible CLI to a numerically valid, deployable winner, with the full reference/fallback/cache and process-recovery contract.

**Blocked by:** 01: Make the existing MXFP4 tuning lifecycle safe.

**Status:** ready-for-agent

- [ ] Extend the single search selector to prune/full/staged, default prune; positive explicit screening top-K is valid only with staged, default 2, distinct from routing top-k. Preserve prune/full candidate order and supervision mechanisms, with 01's all-mode MXFP4 device preflight.
- [ ] Derive staged records only from the same M_est-pruned layout-v2 pair universe, equal sorting/G1/G2 BM. Planning executes no GPU pairs and excludes native/re-tile expansion.
- [ ] Validate actual input/gate/activation/quantization and BM-derived intermediate contracts before grouping. Preserve complete stage/pair identities and stable original order with domain-consistent vocabulary.
- [ ] G1 screens once per BM with ranking reused; G2/pipeline groups use BM plus atomic/non-atomic, reduce/scatter sharing non-atomic. Kernel-only G1 timing includes BM16 inline quantization; sort/external quant/buffer preparation stays untimed.
- [ ] Preserve deterministic producer/atomic-consumer preferences and retries within the admitted same-shape/same-BM pruned candidates, Torch layout-faithful self-test and untimed pair bootstrap. Bootstrap does not compete or enter cache; failed reference fixtures send affected groups to exhaustive normal pair validation.
- [ ] Each G2 attempt owns a fresh reference-producer intermediate; score the complete output path including atomic initialization or reduction/scatter. Default activation/limit parameters match direct G1, reference and production calls.
- [ ] Rank returned positive finite screening scores, tie-break by original order and keep at most K. If either stage has no successful screening for a group, exhaust its admitted normal pairs rather than skip it.
- [ ] Validate shortlist products through the full normal pair path with configurable cosine_diff gate, default 0.1, and positive finite timing; if none is valid, exhaust remaining admitted group pairs, including screening-failed configurations, and choose the fastest valid result across all usable groups.
- [ ] Cache all returned validation outcomes as scalar records and never rerun pair identities during expansion. Fallback cannot recover M_est-excluded candidates. Empty/no-winner shapes return invalid directly and later shapes continue.
- [ ] Every staged shape uses a fresh private-Pipe worker even on one GPU. Separate result arrival and exit; join/inspect/close before GPU release; bound handles by active workers and maintain per-GPU measurement exclusivity.
- [ ] Retain main's --mp normalization/cap by participating visible GPUs and current nonempty batch rows; schedule only the preflighted participating set. A single slot still launches a fresh worker for each staged shape, and an empty batch launches none.
- [ ] Python shape errors, fatal exit and clean EOF/no-result become shape failures. Unexpected IPC/parent/cancellation errors stop launches, clean owned workers/endpoints/handles and propagate the original exception. Test ordering with coordinated events, not fixed sleeps.
- [ ] Own/restore per-operation SIGALRM timeout, default 1800, without parent/shape hard deadlines or permanent-live-hang recovery. Prune/full never enter the staged supervisor.
- [ ] Shape/CLI policy tests verify source projection, K/ties, M_est boundaries, BM contracts, reference retries/direct fallback, invalid/empty shortlists, exhaustive fallback/cache, empty/fatal outcomes, resource cleanup and later-row continuation. With fixed returned pair outcomes, staged still finds a valid prune pair when independent screening/reference fixtures fail.
- [ ] At the first real GPU integration, apply the specification's test discipline: select and verify the physical card through acceptance's PCI/UUID mapping, check idle before each independent test process and run compared GPU tasks sequentially on that same card. Multi-device logic remains covered by simulated fixtures.
- [ ] A real small staged invocation writes a canonical winner CSV, matches explicit A4W4 dtype/quantization, dispatches exact names and uses matching default precompiled entries on the idle-checked target card. Treat independent scores as ranking evidence, not additive final latency.
- [ ] Deliver default-path documentation and focused checks/production evidence according to the [specification](../spec.md); full 27-search calibration belongs to 04.

## Internal delivery order

Keep one ticket and deliver these steps in order; each step can be reviewed and committed independently with its focused checks passing:

1. CLI and candidate planning, verified without GPU pair execution.
2. Screening, reference/fallback policy and pair cache, verified with controlled screening and validation outcomes.
3. Process supervision and scheduling, verified with fake shape functions and coordinated process events.
4. Connect the tested policy and supervisor to real GPU execution, then complete winner CSV, exact production replay and matching default AOT evidence under the test discipline above.

Intermediate commits are implementation steps; the ticket is complete only after the whole path and all acceptance criteria pass.
