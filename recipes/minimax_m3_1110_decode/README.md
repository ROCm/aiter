# MiniMax-M3 2026-11-10 recipe: guarded TP4 MoE decode overlays

This is an **opt-in, version-pinned recipe experiment**, not a tuning-table
change for current main. Current main's grouped TDM load interface differs
from the measured recipe, so its default configs are deliberately untouched.

Pinned server:
- Image: rocm/fw-bringup@sha256:f743523a3b3469a1c0c53cc49031c2b816d8f35bb8ea01e9b68f057591a2e8c4
- AITER: 55cde6a0c49cc04f39f6141e312938a8c48e61fb
- ATOM: 7b7e7961451a8d9dddac9574d855efea6754ecfe
- FlyDSL 0.5.0, C5 four gfx1250 GPUs
- TP4, EAGLE3 K3 with real acceptance, FP8 KV, TG=1, MONO=0,
  online PTPC FP8 for non-MoE layers, original INT4 communication.

## Default M16 change

Only the FP4 MoE row for D=6144, I=768, E=129, top-k=5, padded M=16:
- GEMM1 tile N: 384 -> 128
- N waves: 2 -> 1
- Intra-workgroup K waves: 1 -> 2
- TDM issue-owner waves: auto -> 1
- B-only TDM cache hint: 0 -> 1

The tensor dtype, quantization, activation, routing, shared expert, gather,
GEMM2 tile and model/sampling settings are unchanged. M16 dispatch serves
actual token counts 9–16. Other buckets remain unchanged unless the optional M32 profile below is explicitly selected.

`make_overlay.py` appends a TP-only (`ep_fused=0`) row, preserving the original
wildcard row for EP. It requires the pinned AITER commit, hashes of both critical
source files, and the exact original tuning fields. It refuses different source,
an existing output, missing/changed fields and already-patched input. Baseline
files are never edited. Pin the entire image above: this helper does not
independently validate the ATOM version or every dependency.

## Enable

Inside the pinned image, generate a new full overlay from its effective CSV:

    python /path/to/make_overlay.py \
      --aiter-root /app/aiter \
      --baseline /tmp/aiter_configs/tuned_grouped_fmoe.csv \
      --output /workspace/m3-m16-overlay.csv

Pass the following to the server environment before startup:

    AITER_CONFIG_GROUPED_FMOE=/workspace/m3-m16-overlay.csv

Restart to capture new CUDA graphs. Roll back by removing that override and
restarting; no kernel source or default config has been modified.

## Validate

CPU guards (run anywhere):

    pytest -q test_overlay.py

Whole-MoE GPU benchmark in the pinned image (exclusive GPU):

    python bench_moe.py --aiter-root /app/aiter \
      --baseline /tmp/aiter_configs/tuned_grouped_fmoe.csv --output bench.json

Additional signed-FP4, varied-scale and activation-amplitude checks:

    python check_numeric.py --aiter-root /app/aiter \
      --baseline /tmp/aiter_configs/tuned_grouped_fmoe.csv --output numeric.json

The benchmark constructs synthetic tensors with the pinned recipe's input-only
fixture, then calls the real public fused_moe. It rotates four distinct weight
buffers, captures independent A/A/B graphs of 32 calls, replays before checking,
changes input/route/padding values and alternates timing order. It is not model
serving. M9–16 testing found 1.17–1.39x whole-MoE speedups; M16 repeats were about
1.18x. Tested outputs were bit-identical, including 64 additional old/new
comparisons across signed random packed weights, four low/high scale ranges,
activation amplitudes and two seeds. This is finite test coverage, not a proof
of identical model outputs under dynamic batching or a full accuracy evaluation.

## Serving methodology and limitations

Use the original MLPerf Endpoints client at 4b385eb, unmodified, and dataset SHA256
1beb24c882122df96571cf11b390acbea388944038bc55c78b891475459014ae.
Concurrency 8, independent warmup seeds, measured 5-minute issue cap, 8192 output
cap, temperature 1, streaming, tool delays and salt enabled. These are exploratory
partial trajectories, not an official MLPerf submission. Both sides disable the
separate PullQ4 prefetch experiment and the earlier gather patch.

Each side starts fresh on the same four GPUs. Order is A->B followed by B->A;
no profiler or simultaneous GPU microbenchmarks during serving. Report OSL,
turn counts, real EAGLE acceptance and inline score alongside throughput/latency.
Output lengths differ, so raw output tok/s changes cannot all be attributed to
kernel work. No GSM8K or matched full-trajectory accuracy claim.

One A2 **warmup-only** client invocation logged a malformed-tool chat-template
fallback to whitespace counting; its warmup metrics are not performance evidence.
The measured pass is a fresh client process and must be checked separately.
All measured logs are scanned for that fallback.

Fixed-512-token completion probes are only decode diagnostics, NOT AgentX:
they use ignore_eos and short synthetic prompts. Their counts are fixed but
generated content and acceptance can still vary. They must not replace the
original multi-turn recipe numbers.

Control-host test date: 2026-11-11. C5's clock was behind and logs have a
2026-11-07 date. No machine clock or GPU clock settings were changed.


## Measured multi-turn AgentX results

Two independent 5-minute issue-cap pairs, run A->B then B->A with fresh servers:

| Pair | Metric | Baseline | M16 overlay | Change |
|---|---|---:|---:|---:|
| 1 | TPOT p50 (ms) | 3.997 | 3.822 | -4.4% |
| 1 | TPOT mean (ms) | 3.963 | 3.796 | -4.2% |
| 1 | System output tok/s | 868.0 | 967.0 | +11.4% |
| 1 | TTFT p50 (ms) | 102.84 | 100.24 | -2.5% |
| 1 | Mean output tokens/turn | 459.4 | 520.2 | +13.2% |
| 1 | EAGLE acceptance | 17.884% | 17.897% | approximately unchanged |
| 2 | TPOT p50 (ms) | 4.187 | 3.796 | -9.3% |
| 2 | TPOT mean (ms) | 4.133 | 3.755 | -9.1% |
| 2 | System output tok/s | 821.7 | 877.5 | +6.8% |
| 2 | TTFT p50 (ms) | 101.96 | 104.84 | +2.8% (worse) |
| 2 | Mean output tokens/turn | 449.8 | 438.8 | -2.5% |
| 2 | EAGLE acceptance | 17.936% | 17.622% | lower, not a fixed-rate shortcut |

All 2,344 measured turns completed without failures. Each measured client log
was checked: no chat-template fallback or tokenization error. Two pairs show
lower TPOT and higher observed output tok/s, but different trajectories/output
lengths prevent attributing the entire tok/s change to the kernel. TTFT does not
consistently improve. These are limited screening results, not a statistical
performance guarantee or matched accuracy evaluation. Inline scores and complete
run details are in results.json.

The fixed-512-token diagnostic found c4 TPOT p50 1.769 -> 1.631 ms and
2.003 -> 1.847 ms (~8% lower); c8 was effectively unchanged. Batch-8 verification
usually uses M32, outside this M16 overlay, so this is not a general batch-size
optimization. Fixed-length overall throughput was noisy and is not claimed.

The final standalone public benchmark reproduced 1.15–1.37x whole-MoE speedups,
with A/A within about 1%, four independent weight buffers, changed inputs/routes
and exact old/new outputs. The public numeric script passed 32 additional
candidate comparisons exactly. CPU guard tests: 34 passed. TP-only dispatch was
checked against the service config and EP selected its unchanged original row.

## M32 incremental experiment

`make_overlay.py --token 32` opts into a separately measured M32 profile.
The default remains `--token 16`. To compose both, first generate the M16
overlay as above, then run:

    python /path/to/make_overlay.py --token 32 \
      --aiter-root /app/aiter \
      --baseline /workspace/m3-m16-overlay.csv \
      --output /workspace/m3-m16-m32-overlay.csv

Use the new file for `AITER_CONFIG_GROUPED_FMOE` and restart the server.
The M32 profile appends only a TP-specific row for actual M17–32:
GEMM1 K tile 1024 -> 256 and buffers 2 -> 3; GEMM2 N tile 256 -> 64,
N waves 2 -> 1 and buffers 3 -> 2; B cache hint 0 -> 1. Other tiles,
quantization, route semantics, and model/sampling settings stay unchanged.
The original M32 wildcard EP row and all M16 rows are preserved.
Both tools reject already-patched or unmeasured inputs.

The benchmark and numeric scripts accept the same `--token 32` option.
Run them with the original recipe CSV, not a CSV already patched for the
selected bucket. Numeric validation requires finite, **bit-identical** outputs.
The old M16 commands and their default bucket remain available.

### Incremental AgentX evidence: M16 baseline vs M16+M32

Fresh TP4 servers; 5-minute issue cap A->B, then 10-minute cap B->A.
Client, dataset, EAGLE3 and sampling settings are as above. These results are
incremental comparisons, not directly comparable to the earlier M16 pairs.

| Pair | Metric | M16 baseline | M16+M32 | Change |
|---|---|---:|---:|---:|
| 5 min | TPOT p50 (ms) | 3.887 | 3.779 | -2.8% |
| 5 min | TPOT mean (ms) | 3.853 | 3.716 | -3.5% |
| 5 min | System output tok/s | 939.4 | 878.9 | **-6.4% (worse)** |
| 5 min | TTFT p50 (ms) | 103.76 | 103.54 | -0.2% |
| 5 min | Mean output tokens/turn | 520.9 | 460.0 | different |
| 5 min | EAGLE acceptance | 17.589% | 18.050% | different |
| 10 min | TPOT p50 (ms) | 3.878 | 3.842 | -0.9% |
| 10 min | TPOT mean (ms) | 3.828 | 3.763 | -1.7% |
| 10 min | System output tok/s | 934.7 | 964.4 | +3.2% |
| 10 min | TTFT p50 (ms) | 100.77 | 95.69 | -5.0% |
| 10 min | Mean output tokens/turn | 447.8 | 491.5 | different |
| 10 min | EAGLE acceptance | 18.799% | 19.037% | different |

All 3,704 measured turns completed, with no client tokenization fallback.
An offline reconstruction using the exact pinned client's structured output
and after-first-chunk rules matches each run's output-token total and mean
TPOT. Output-length bins and inline scores are included in `results_m32.json`.
Those bins are diagnostic, not randomized or matched-request comparisons.

**No stable system-throughput gain is established.** Both TPOT point estimates
are lower, but batch composition, output lengths and acceptance differ;
these two pairs alone do not establish a statistical guarantee. There is no
full-trajectory or independent accuracy evaluation, and no TTFT guarantee.

### Real-data replay scope

An eager diagnostic server captured rank-0 activation, routing and weight
snapshots for MoE ordinals 0, 1, 12, 26, 40 and 56 at M32. These local-only
model fixtures are NOT distributed in this PR. This is not a TP4 service replay
and is not exhaustive coverage of all layers or ranks.

Each saved baseline output and each candidate replay were checked exactly.
Randomized A/A/B graph timing, rotating all six layers, measured:
- first process: 31.863 -> 29.651 us/MoE (1.0746x);
- independent repeat: 32.200 -> 29.980 us/MoE (1.0740x).

Single-layer hot-cache results vary substantially; ordinal 12 is slightly
slower with the candidate in the first process. The six-layer cycle is the
reported aggregate, not a claim that every layer improves. Replaying synthetic
inputs with `bench_moe.py` is independently reproducible but need not reproduce
the same real-route timing. All per-layer and A/A results are retained.

These incremental tests were run on control-host date 2026-11-11; C5's
uncorrected clock labels them 2026-11-08. GPU clock settings were not changed.

### Release validation and fixed-length diagnostics

The shared public tools were rerun after adding `--token`:
- CPU guards: **69 passed**, Ruff check/format and `git diff --check` passed.
- Public whole-MoE M17–32: **1.044–1.097x**, all outputs exact, including
  changed activation/routing and dead-tail graph replay; A/A varied by up to
  about 1.8%. Default M9–16 regression remained 1.165–1.384x.
- Signed packed FP4, varied scales and activation amplitudes: **32 finite,
  bit-identical cases per bucket**, 64 total.

Four fresh diagnostic servers ran A->B->B->A, c8, 32 requests per point,
2048 output tokens/request (`ignore_eos=True`), two request sets at each of
approximately 1.8K/14.4K input tokens. All 512 measured requests succeeded.
Every paired request hash and input/output token count matched. All eight
paired TPOT means were lower (0.36–5.08%); observed output tok/s was higher
(0.48–4.45%). These are **synthetic fixed-length diagnostics, not AgentX**.

In each pair, 19–26 of the 32 generated text hashes matched. Real EAGLE3
acceptance was roughly 85–93% and differed between most pairs, much higher
than AgentX. Fixed-length counts remove the output-length confound but do
not remove acceptance, content or dynamic-batching differences. The client
TPOT here uses server output usage minus one, not Endpoints' structured
after-first-chunk token count; the two metric populations must not be pooled.
All per-point results, exact acceptance counts and paired deltas are retained
in `results_m32.json`. There is still no stable AgentX-throughput claim.
