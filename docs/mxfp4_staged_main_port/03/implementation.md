# Configured output and activation implementation

Ticket03 starts at integration `9c035b435716c8148696f61e63d4330ba2c33089`.
The canonical tracker lives only in the integration worktree. Snapshot/route
forwarding is commit `2308ce0bf`; configured AOT is commit `9260524a5`.

The MXFP4 CLI resolves its tuning limit and two output switches once, before
input selection. Programmatic invocations attach the same snapshot to their
arguments; spawn workers inherit both those arguments and the untouched process
environment. Direct G1, references, bootstrap and pair validation receive that
limit in prune/full/staged. Finite positive configured limits apply to Silu or
Swiglu; Situv2 with a configured limit is rejected before already-tuned filtering.
Default Situv2 keeps the reference parameters 4/25. Stage2 screening and normal
pairs now receive the original topk weights so production's FP8 path executes
its final weighted reduction. Atomic/default paths retain their output handling.

Controlled checks exercise CLI/shape behavior, CSV pre-dedup rejection, original
route-weight forwarding, the real production stage2 wrapper at the external
kernel boundary, and fresh spawned CPU workers. All controlled invocations use
empty ROCm/HIP/CUDA visibility, synthetic import discovery and fail-closed real
GPU allocation/measurements/workers. The worker target is explicitly admitted
and only probes environment/configuration and production switch resolution.

The configured replay harness derives no candidate registry. Its separate pair
request files are selected from the real admitted source by `create_fixtures.py`.
`--action pair` checks the unchanged Torch numerical gate, measures the complete
pair with ordinary `run_perftest` and replaces that request's zero timing with a
valid canonical CSV row. `--action replay` verifies exact public metadata, A4W4
quantization, named launch arguments, actual target dtype and complete final
reduction/scatter. AOT and run-only replay use matching limit/flags and an isolated
FlyDSL cache. Cold AOT jobs and exact run-only success provide deployment evidence;
plain fallback execution cannot satisfy the harness assertions.

The clamp fixture multiplies hidden values by16 and requantizes both the Torch
reference input and actual pipeline input under the same per-1x32 FP4 contract.
It compares limits5 and7 on that fixed data and requires distinguishable outputs.
Nonuniform route weights are the literal sequence1..topk normalized to sum1.
Numerical output always uses the same ordinary Torch reference, including lossy
FP4 scatter; no reference quantization is added to conceal an error.

GPU execution and final measured results are recorded separately under `gpu/`.
Only the root coordinator launches GPU tasks, with the designated UUID/PCI idle
gate and one independent process at a time. All22 specified pairs completed all4 actions (88 tasks). The two independent
matching-limit clamp controls also passed, giving90 sequential task records.
See [the GPU evidence](gpu/README.md) for exact names, numerical errors,
AOT job configuration, kernel sums, process timing and full traces.

The documented selected-CSV batch path uses a guarded file and the existing
collector/worker pool. Its31 controlled AOT checks include an independent CPU
subprocess through the real pool with the external FlyDSL compiler substituted.
The full default model corpus includes Situv2 and therefore cannot accept a
custom limit globally; selected compatible CSVs provide matching batch jobs.

Controlled coverage has231 unique cases: the189 recorded default/configuration
regression cases,2 caller-owned CSV reuse cases,9 additional custom direct/
bootstrap parameter cases and31 AOT cases. The repeated24 execution checks
include the prior15 and those9 new cases. Required syntax/Black/Ruff/diff
checks pass. Omitted public Situv2 API1/1 remains a separate final acceptance
check; this ticket records the tuner default reference parameters4/25 explicitly.
