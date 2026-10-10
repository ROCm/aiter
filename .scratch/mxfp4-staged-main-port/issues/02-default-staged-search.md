# 02: Deliver complete default-environment staged search

**What to build:** A supported default-environment invocation runs staged search from the main-compatible CLI to a numerically valid, deployable winner, with the full reference/fallback/cache and process-recovery contract.

**Blocked by:** 01: Make the existing MXFP4 tuning lifecycle safe.

**Status:** resolved

- [x] Extend the single search selector to prune/full/staged, default prune; positive explicit screening top-K is valid only with staged, default 2, distinct from routing top-k. Preserve prune/full candidate order and supervision mechanisms, with 01's all-mode MXFP4 device preflight.
- [x] Derive staged records only from the same M_est-pruned layout-v2 pair universe, equal sorting/G1/G2 BM. Planning executes no GPU pairs and excludes native/re-tile expansion.
- [x] Validate actual input/gate/activation/quantization and BM-derived intermediate contracts before grouping. Preserve complete stage/pair identities and stable original order with domain-consistent vocabulary.
- [x] G1 screens once per BM with ranking reused; G2/pipeline groups use BM plus atomic/non-atomic, reduce/scatter sharing non-atomic. Kernel-only G1 timing includes BM16 inline quantization; sort/external quant/buffer preparation stays untimed.
- [x] Preserve deterministic producer/atomic-consumer preferences and retries within the admitted same-shape/same-BM pruned candidates, Torch layout-faithful self-test and untimed pair bootstrap. Bootstrap does not compete or enter cache; failed reference fixtures send affected groups to exhaustive normal pair validation.
- [x] Each G2 attempt owns a fresh reference-producer intermediate; score the complete output path including atomic initialization or reduction/scatter. Default activation/limit parameters match direct G1, reference and production calls.
- [x] Rank returned positive finite screening scores, tie-break by original order and keep at most K. If either stage has no successful screening for a group, exhaust its admitted normal pairs rather than skip it.
- [x] Validate shortlist products through the full normal pair path with configurable cosine_diff gate, default 0.1, and positive finite timing; if none is valid, exhaust remaining admitted group pairs, including screening-failed configurations, and choose the fastest valid result across all usable groups.
- [x] Cache all returned validation outcomes as scalar records and never rerun pair identities during expansion. Fallback cannot recover M_est-excluded candidates. Empty/no-winner shapes return invalid directly and later shapes continue.
- [x] Every staged shape uses a fresh private-Pipe worker even on one GPU. Separate result arrival and exit; join/inspect/close before GPU release; bound handles by active workers and maintain per-GPU measurement exclusivity.
- [x] Retain main's --mp normalization/cap by participating visible GPUs and current nonempty batch rows; schedule only the preflighted participating set. A single slot still launches a fresh worker for each staged shape, and an empty batch launches none.
- [x] Python shape errors, fatal exit and clean EOF/no-result become shape failures. Unexpected IPC/parent/cancellation errors stop launches, clean owned workers/endpoints/handles and propagate the original exception. Test ordering with coordinated events, not fixed sleeps.
- [x] Own/restore per-operation SIGALRM timeout, default 1800, without parent/shape hard deadlines or permanent-live-hang recovery. Prune/full never enter the staged supervisor.
- [x] Shape/CLI policy tests verify source projection, K/ties, M_est boundaries, BM contracts, reference retries/direct fallback, invalid/empty shortlists, exhaustive fallback/cache, empty/fatal outcomes, resource cleanup and later-row continuation. With fixed returned pair outcomes, staged still finds a valid prune pair when independent screening/reference fixtures fail.
- [x] At the first real GPU integration, apply the specification's test discipline: select and verify the physical card through acceptance's PCI/UUID mapping, check idle before each independent test process and run compared GPU tasks sequentially on that same card. Multi-device logic remains covered by simulated fixtures.
- [x] A real small staged invocation writes a canonical winner CSV, matches explicit A4W4 dtype/quantization, dispatches exact names and uses matching default precompiled entries on the idle-checked target card. Treat independent scores as ranking evidence, not additive final latency.
- [x] Deliver default-path documentation and focused checks/production evidence according to the [specification](../spec.md); full 27-search calibration belongs to 04.

## Internal delivery order

Keep one ticket and deliver these steps in order; each step can be reviewed and committed independently with its focused checks passing:

1. CLI and candidate planning, verified without GPU pair execution.
2. Screening, reference/fallback policy and pair cache, verified with controlled screening and validation outcomes.
3. Process supervision and scheduling, verified with fake shape functions and coordinated process events.
4. Connect the tested policy and supervisor to real GPU execution, then complete winner CSV, exact production replay and matching default AOT evidence under the test discipline above.

Intermediate commits are implementation steps; the ticket is complete only after the whole path and all acceptance criteria pass.

## Answer

Implemented the existing prune/full/staged selector and positive staged screening top-K, shared pruned layout-v2 planning, full stage/pair identity and BM/accumulation grouping, deterministic checked references, positive finite stage ranking, group-local exhaustive fallback, scalar returned-outcome pair cache, fresh private-Pipe workers even with one slot, resource cleanup and owned operation SIGALRM restoration. Prune/full retain their original scheduling/supervision; staged preflights first selected visible devices. Actual default screening uses kernel-only G1 timing and fresh candidate-local G2 intermediates with complete final output handling.

Phase commits: `65a4eba35` (CLI/planning), `60cbce1f6` (controlled reference/screen/fallback/cache), `2c902bf08` (supervision/timeouts), `09c9b5ef27aaf322435308166f0b0dd0c5bbd1f4` (GPU execution), with deterministic exit-before-read and exact replay supplements. Final implementer `a1a8d2fd9d3be00fbdaf145409413914c9e9f832`, evidence `05a7244fb`, integration merge `00f9c6073ff3d078ee5ead9688f7dc6804a337e6`.

Acceptance: 155 unique focused controlled checks passed, plus Black/Ruff/syntax/code whitespace checks. Four real tasks ran sequentially after per-task idle and UUID/PCI checks on physical GPU7 (PCI `0000:f5:00.0`, gfx950/256 CUs), all exit 0. The prescribed `(4,3072,512,128,4)` Swiglu A4W4 staged smoke screened 12 G1/32 G2, validated 8 pairs without fallback, wrote a canonical winner with `us=us1=15.3478` microseconds and `us2=0`, and completed tuner summary in 107.2228 seconds. Exact public production error was `0.011169739046099525`; fresh-cache AOT compiled all 4 selected-CSV jobs; precompiled-only exact production error was `0.011173635721687503` with `FLYDSL_RUNTIME_RUN_ONLY=1`. All numerical checks used the unchanged finite cosine-diff gate <=0.1.

See [implementation evidence](../../../docs/mxfp4_staged_main_port/02/implementation.md), [GPU results](../../../docs/mxfp4_staged_main_port/02/gpu/README.md), [task sequence](../../../docs/mxfp4_staged_main_port/02/gpu/task-sequence.json) and adjacent full logs/configuration/results/compressed traces. Synchronized diagnostics are separately labelled from normal continuous tuner kernel-duration sum. This small M_est1 smoke does not establish scatter/I384/large-BM/Situv2 or extended settings; those remain required in tickets 03–04.
