# MiniMax-M3 2026-11-10 recipe: TP4 M16 MoE decode overlay

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

## Change

Only the FP4 MoE row for D=6144, I=768, E=129, top-k=5, padded M=16:
- GEMM1 tile N: 384 -> 128
- N waves: 2 -> 1
- Intra-workgroup K waves: 1 -> 2
- TDM issue-owner waves: auto -> 1
- B-only TDM cache hint: 0 -> 1

The tensor dtype, quantization, activation, routing, shared expert, gather,
GEMM2 tile and model/sampling settings are unchanged. M16 dispatch serves
actual token counts 9–16. Other buckets remain unchanged.

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
