# Kimi K3 Gluon monokernel and B8 hybrid

Research snapshot for gfx950/MI355X, single-node TP8 decode, batch 1–8.
The kernels, HIP-IPC support, checkpoint references, and AMD vLLM integration
are included here. This directory is an explicit experiment entrypoint; it
does not change AITER's global operator dispatch or any NVIDIA implementation.

## Implementations

| Adapter | B8 execution | Validation at export |
|---|---|---|
| `integration/monokernel.py` | One resident kernel for the fused region, including routed experts | TP8 vLLM concurrency 1–8 measured |
| `integration/hybrid5.py` | Gluon front → AITER sorting → AITER MoE1 → AITER MoE2 → Gluon tail | TP8 vLLM C1–4/C8, 92 layers × 8 ranks, graph replay checked |
| `integration/hybrid4.py` | Gluon front including metadata/zeroing → AITER MoE1 → AITER MoE2 → Gluon tail | Standalone correctness and changed-input graph replay passed; serving and uncontended performance checks pending |

The hybrid adapters retain the monokernel dispatch for smaller physical batches.
The five-kernel reference has no MoE output copy. The four-kernel candidate also
removes the separate sorting launch. Hybrid B8 experts use the baseline A4W4
SiTUv2 kernels; the full monokernel uses A16W4. Prefill stays on the existing
A16W4 vLLM path in the integrated variants.

This is not an entire attention/decode layer in one launch. On the 68 KDA layers,
the fused region starts at attention **output projection**, then includes its
TP reduction, post-attention AttnRes/RMSNorm, router/top-16, latent down projection,
shared expert gate/up and activation, routed experts, latent reduction/norm/up,
shared down projection, final reduction/combine, and residual-prefix update.
The attention core remains separate. On the 24 MLA layers the integrated region
starts at MoE; all 92 layers use the fused MoE path. Fusion stops at the layer
boundary. ISL/OSL in the saved serving results are both 1024.

The package copies retain separate tuning choices so one batch's optimization
cannot silently alter another batch:

| Physical batch / purpose | Kernel package |
|---|---|
| B1 | `k3_gluon_serving_front32` |
| B2/B4 | `k3_gluon_serving_b4grid512` |
| B3 | `k3_gluon_serving_b4tiles` |
| Other small shapes | `k3_gluon_serving_tuned` |
| Full B8 monokernel | `k3_gluon_serving_b8grid512` |
| Five-kernel B8 reference | `k3_gluon_hybrid_b8` |
| Four-kernel B8 candidate | `k3_gluon_hybrid_sort_b8` |
| Shared host/reference utilities | `k3_gluon` |

## Synchronization and metadata

The resident kernels use epoch-tagged local and HIP-IPC mailboxes. Polls use
side-effecting `global_load_dwordx2 ... sc1` assembly and payload reads use
coherent `.cv` loads. The host checks zero spills and enough resident CTA capacity
before launch; a 512-CTA grid must fit two CTAs on each of 256 CUs. First eager
launches synchronize ranks after compilation so lazy compilation is not charged
against a peer's watchdog. Same-stream launch ordering connects the hybrid's
front, expert stages, and tail.

For B8, top-16 produces 128 routes and at most eight rows per expert. The
four-kernel front uses spare CTAs 376–431 to zero the MoE2 accumulation buffer.
CTA 432 acquires `RouteReady` and builds one BM16 block per active expert.
Pairwise comparisons count earlier occurrences and distinct smaller experts;
padding and valid rows are written to disjoint positions. This avoids a
histogram scan and a clear-then-scatter barrier. Packed token IDs, routing
weights, expert IDs, valid-row counts, row maps, and reverse maps match the
selected AITER two-stage ABI. Token IDs within each token's top-16 must be
distinct. The two AITER expert kernels and their grids are retained.

The FlyDSL reference is ROCm/FlyDSL PR 1204, commit
`21a3d1ee6ba2d6c0f33e3feb56e23cbf9e79c085`. Its monokernel directly assigns expert
tasks to routes; the hybrid instead constructs metadata for the baseline AITER
expert kernels.

## Environment

See [evidence/environment.json](evidence/environment.json) for the recorded
Torch, HIP, Triton, vLLM, and FlyDSL versions. The experiment used Python 3.12,
Torch `2.13.0+git733fca1`, HIP `7.2.53211`, a Triton 3.8 build with Gluon atomics,
FlyDSL `0.3.4.1`, and vLLM `0.30.1rc1.dev648+g92044241a`.

The compiler must expose Gluon `atomic_load`/`atomic_store`, support `.cv`
coherent loads, and provide CDNA4 scaled MFMA layouts. The stock benchmark
image lacked those Gluon atomics; an ABI-compatible compiler build was used.
Compiler binaries and model weights are not bundled. Install AITER with the
BM16 `_adaptive_moe_sort` / `fused_moe_2stages` A4W4 SiTUv2 path and the selected
`flydsl_mxmoe_g1_a4w4_16x256x256_f16in_nt_situv2_xcd4` and
`flydsl_moe2_layout_afp4_wfp4_bf16_t16x128x128_atomic_nt_sbm16` implementations.
The B8 hybrid is specialized to this ABI; a different tuning row/layout needs
validation before use.

Use an idle eight-GPU allocation for timing. The GPU experiment was paused
because another job occupied all eight GPUs. The latest candidate's final
microbenchmarks overlap that job, so they are not evidence of a speedup.

## Standalone checks and benchmarks

From this directory, with the compatible Python/compiler/AITER environment:

```bash
export PYTHONPATH="$PWD:$PWD/benchmarks${PYTHONPATH:+:$PYTHONPATH}"
python benchmarks/test_fused_sort.py
python benchmarks/run_b8.py --variant monokernel --model /path/to/Kimi-K3 \
  --blocks 1 --output results/monokernel-b8-blocks1.json
python benchmarks/run_b8.py --variant hybrid5 --model /path/to/Kimi-K3 \
  --blocks 1 --output results/hybrid5-b8-blocks1.json
python benchmarks/run_b8.py --variant hybrid4 --model /path/to/Kimi-K3 \
  --blocks 1 --output results/hybrid4-b8-blocks1.json
```

Repeat with `--blocks 8` to exercise the longer AttnRes case. The harness uses
checkpoint layer 5, TP8, 128 invocations per captured graph, and three repeats.
It checks numerical outputs, mailbox errors, occupancy, and graph replay before
reporting timings. The hybrid harness also changes graph inputs and churns the
allocator before replay. `--print-command` is a CPU-only configuration check.
The underlying scripts retain some historical ablation switches; native-dense
and saved-crash replay modes are intentionally unavailable in this snapshot.

## AMD vLLM integration

The patch targets the recorded vLLM revision and edits only
`vllm/models/kimi_k3/amd/{linear,kda}.py`. Apply it to a clean matching checkout,
then select the adapter:

```bash
git -C /path/to/vllm apply "$PWD/integration/vllm-kimi-k3.patch"
python integration/install_adapter.py --variant hybrid4 --vllm-source /path/to/vllm
export PYTHONPATH="$PWD:$PWD/integration:/path/to/vllm${PYTHONPATH:+:$PYTHONPATH}"
bash integration/serve.sh /path/to/Kimi-K3
```

Use `--variant monokernel` or `--variant hybrid5` to select those implementations.
The selected adapter dispatches automatically by physical batch. The server
recipe uses TP8, FP8 KV, max model length 4096, max sequences 8, 8192 batched
tokens, no speculative decoding, and no prefix caching. It binds port 8243 on
localhost and enables benchmark-only worker RPCs. `KIMI_K3_RESULTS` optionally
chooses the directory for diagnostic dumps and traces.

The saved results used the InferenceX serving benchmark client:

```bash
python integration/benchmark.py \
  --benchmark-script /path/to/InferenceX/utils/bench_serving/benchmark_serving.py \
  --model /path/to/Kimi-K3 --concurrency 8 --output results/hybrid4-serving
```

This runs random ISL=1024/OSL=1024, seed 42, two warmups, and
`max(10, 2*concurrency)` prompts. After measuring C8, repeat C1–4 as protected
cases. Do not accept a B8 optimization that regresses the retained smaller-batch
variants. The worker extension exposes all-layer numerical validation and
mailbox inspection outside the timing interval.

## Evidence and remaining work

[evidence/validation.json](evidence/validation.json) contains the saved
monokernel and five-kernel serving metrics and the four-kernel correctness
results. Total token throughput counts input plus output; output throughput
counts generated tokens only. The five-kernel C8 repeat is kept separate from
its primary run, whose initial TTFT increase did not recur.

The four-kernel metadata checks covered 23 routing cases plus four changed-input
graph replays. Full-region checks at one/eight AttnRes blocks passed with zero
mailbox errors and no spills. Its TP8 vLLM run failed before model loading due to
occupied GPU memory. Uncontended paired timings, all-layer vLLM validation,
C1–4 regression checks, and a trace proving four launches per region remain
pending. C5–7 have not been remeasured for either hybrid. lm-eval has not run.
TP4 end-to-end results are unavailable because the checkpoint does not fit the
tested four-GPU allocation.

The numerical reference uses FP32 dense math, the baseline AITER experts, and
rank-ordered TP sums. Baseline dense-projection rounding was separately observed
to amplify through A4W4 experts; operator checks are not an lm-eval substitute.

Kernel copies are deliberately separate. `source-manifest.json` records original
and exported hashes and whether each original Python AST is unchanged. The hybrid
host now adds an AITER dispatch guard; its entry records that intentional change.
The device kernels retain the original arithmetic and schedules.

## Four-kernel source integration

`k3_gluon_hybrid_sort_b8/pipeline.py` is the shared launch contract used by both
`integration/hybrid4.py` and `benchmarks/bench_hybrid4.py`. It checks the supported
B8/TP8/512-CTA/eight-warp schedule, supplies the fused metadata to the front,
advances the epoch once, invokes the prepared AITER experts, and passes their
returned output directly to the tail. No sort or copy is enqueued by this wrapper.
The AITER metadata transform rejects a different expert kernel pair, block size,
one-stage implementation, or prequantization path. Preparation still initializes
metadata once outside graph capture; the repeated decode path prepares it in the
Gluon front. The exact four device launches require a future GPU trace.

CPU-only wiring checks (no Torch, Triton, or HIP imports):

```bash
python -m unittest discover -s tests -v
```

These check launch order, epoch options, output pointer identity and unsupported
configurations. The current shared wrapper/dispatch guard have not been run on a
GPU. Previous standalone evidence applies to the original candidate only.

## Reusable Codex skill

[build-monokernel/SKILL.md](skills/build-monokernel/SKILL.md) describes the general
workflow, with detailed references for synchronization, IPC collectives, hybrid
metadata, occupancy, tuning, numerical checks and serving validation. The K3 case
study identifies case-specific assumptions and links to implementation anchors.
Copy the `build-monokernel` folder into your Codex skills directory to reuse it.
