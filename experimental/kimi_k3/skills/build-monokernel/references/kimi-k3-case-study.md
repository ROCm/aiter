# Kimi K3 reference implementation and lessons

## Source anchors

The reference snapshot is in ROCm/aiter branch `vinayak/monokernel_kimik3`, under
`experimental/kimi_k3`. Initial published commit:
`3656b09a8fec166e6b2a92dd3bf6bc0e2df55adb`. Later branch commits include the shared
hybrid pipeline and dispatch checks. Use the checked-out source as authority.

Repository location:
https://github.com/ROCm/aiter/tree/vinayak/monokernel_kimik3/experimental/kimi_k3

The FlyDSL reference used commit
`21a3d1ee6ba2d6c0f33e3feb56e23cbf9e79c085`, PR 1204. Do not assume the PR's latest
state matches that revision. Read the corresponding stage of the pinned
reference before porting an operation and write down semantic differences.

| File relative to the snapshot | What to study |
|---|---|
| `hip_ipc.py` | HIP IPC allocation, handle exchange, mapping, lifetime |
| `k3_gluon_hybrid_sort_b8/primitives.py` | Tagged 64-bit peer words, poll ISA, fixed-order reduction |
| `k3_gluon_hybrid_sort_b8/coherent_memory.py` | `.cv` payload loads |
| `k3_gluon_hybrid_sort_b8/moe_front.py` | Routing and store completion |
| `k3_gluon_hybrid_sort_b8/moe_complete.py` | Separate ready flags and dependencies |
| `k3_gluon_hybrid_sort_b8/perf_front32.py` | Dense producers, 32-row tiles/four K splits |
| `k3_gluon_hybrid_sort_b8/perf_reference_chain.py` | Stage boundaries, spare-CTA zeroing and routing consumer |
| `k3_gluon_hybrid_sort_b8/sort_metadata.py` | Pairwise stable BM16 routing metadata |
| `k3_gluon_hybrid_sort_b8/host.py` | AITER weight views, metadata ABI, retained experts |
| `k3_gluon_hybrid_sort_b8/pipeline.py` | Source-only launch contract and exact dispatch guard |
| `k3_gluon_hybrid_sort_b8/perf_reference_tail.py` | External expert output, TP/norm/up/shared/residual tail |
| `integration/{monokernel,hybrid5,hybrid4}.py` | Serving selection, warmup, occupancy, persistent ownership |
| `integration/vllm-kimi-k3.patch` | AMD model integration and output-projection boundary |
| `benchmarks/test_fused_sort.py` | GPU metadata and replay cases; do not run in source-only mode |
| `benchmarks/bench_hybrid4.py` | Independent output checks and captured region timings |
| `tests/test_hybrid_pipeline.py` | CPU-only orchestration/configuration checks |
| `evidence/{environment,validation}.json` | Saved environment, measured results and limitations |

## Exact scope and precision

Target: gfx950/MI355X, single-node TP8, decode batches 1–8, serving ISL=1024,
OSL=1024. K3 has hidden width 7168, routed latent width 3584, 896 routed experts,
and top-16 selection. The captured implementation uses a SiTUv2 activation with
beta 4, linear beta 25, separated gate/up, and group-32 expert scales.

For 68 KDA layers, Gluon starts at attention output projection, then TP reduction,
post-attention AttnRes/RMSNorm, router/top-k, latent down, shared gate/up/activation,
routed experts, latent reduction/norm/up, shared down, final reduction/combine and
residual-prefix update. The attention core remains outside. For 24 MLA layers,
the integration starts at MoE; all 92 layers use the fused MoE path. It does not
fuse continuously from one layer's MoE to the next layer's attention core.

Full Gluon mono uses BF16 dense values and FP32 router scores with A16W4 experts.
The hybrid B8 retains baseline A4W4 experts. The FlyDSL comparison used different
fusion scope and dense/router precision (MXFP8 dense/BF16 router in the recorded
comparison). Thus end-to-end numbers compare implementations; they do not isolate
fusion alone. Preserve this caveat when using FlyDSL as the performance reference.

## The four-launch candidate

1. Gluon front runs projection/AttnRes where applicable, dense MoE front, routing,
   shared activation, fused metadata, and output zeroing.
2. AITER MoE1 consumes BF16 latent input and prepared BM16 metadata, quantizes
   inline, runs gate/up and SiTUv2 with the selected packed format.
3. AITER MoE2 consumes the intermediate and accumulates into the front-zeroed
   output buffer using the selected atomic implementation.
4. Gluon tail reads the actual returned output directly, then completes latent TP
   reduction/norm/up, shared down, final TP combine and residual update.

B8's front has 376 dense producers. CTAs 376–431 zero 512 BF16 elements each:
56*512 = 8*3584. CTA 432 waits on the eight RouteReady publications and constructs
metadata for the 128 routes. Grid is 512 CTAs with eight 64-lane warps/CTA. The
recorded compilation fitted two CTAs/CU on 256 CUs with zero spills. Future edits
and compilers must re-establish this; the number 512 is not intrinsically safe.

`prepare_metadata` still runs an AITER sorter once during preparation to allocate
and initialize ABI-compatible buffers, before capture/timing. The repeated
pipeline does not call it. Do not mistake the setup call for a fifth decode
launch, or remove initialization before expert warmup without replacing it with
valid metadata. The shared launch wrapper sets front `END=2, BEGIN=True`, tail
`START=4, END=4, BEGIN=False`, passes identical metadata, and supplies MoE's actual
returned tensor to `ExternalRouted`. Same-stream ordering replaces the inter-kernel
mailboxes at these two boundaries; internal/front and TP mailboxes remain.

Selected expert kernels:

- `flydsl_mxmoe_g1_a4w4_16x256x256_f16in_nt_situv2_xcd4`
- `flydsl_moe2_layout_afp4_wfp4_bf16_t16x128x128_atomic_nt_sbm16`

The pipeline rejects different selected names, non-BM16, one-stage dispatch or
separate prequant. This preserves the known metadata/launch contract; it does not
prove future versions of the same wrappers still issue only two kernels. Trace
when GPUs become available. This is Gluon surrounding retained baseline AITER
expert kernels; the experts themselves need not be rewritten in Gluon.

## Performance lessons and retained variants

Separate packages preserve tuning across physical batches: front32 for B1,
512-grid specialization for B2/B4, b4tiles for B3, a general tuned variant for
other small shapes, and a dedicated B8 mono/hybrid. Do not overwrite a shared
compile-time global to tune only one batch.

The early slowdown had strong ISA evidence for repeated acquire/release-related
cache operations. The complete explicit-coherence protocol improved the matched
B8 region from roughly 1151 to 436 microseconds in the recorded profiled ablation;
additional staged expert, load and layout changes reduced it further. These are
historical ablation numbers, not current serving latencies or isolated predictions.

The later full B8 monokernel was around 26.553 ms TPOT versus FlyDSL 26.917 ms,
but the serving baseline was around 21.845 ms. Being faster than another
monokernel did not make it faster than the unfused baseline. The evidence motivated
retaining larger-grid A4W4 MoE kernels while fusing the surrounding small work.

The validated five-kernel no-copy hybrid measured B8 TPOT 20.075 ms; an independent
repeat measured 20.271 ms. Removing the output copy eliminated 92 launches per
decode step (profile count 1318 -> 1226). These values are saved evidence, not a
promise for another environment. Consult validation.json for complete metrics and
run identity rather than mixing TTFT from one run with throughput from another.

The four-kernel candidate passed standalone numerical checks at one/eight
AttnRes blocks, 23 metadata cases and four changed-input graph replays in the
original experiment. Its last performance samples overlapped another job and
must not be presented as a speedup. vLLM validation failed before model loading
because GPUs were occupied. Later source-only launch/ABI edits have no new GPU
validation. All-layer serving checks, uncontended paired timing, a four-launch
trace, and protected smaller-batch regressions remain pending. lm-eval is unrun.

## Reproduction and adapting to another model

Start from the snapshot README/environment file, not an assumed stock Triton
wheel. The experiment used Torch 2.13.0+git733fca1, HIP 7.2.53211, Triton 3.8 with
Gluon atomics, FlyDSL 0.3.4.1 and vLLM 0.30.1rc1.dev648+g92044241a. Keep compiler
capabilities, weight layouts and AITER ABI aligned; no compiler binaries or model
weights are bundled. GPU commands in the README require a separately authorized
idle allocation. CPU tests are explicitly separate.

For a new model, retain the workflow and protocol reasoning, then replace shape
constants, expert activation/routing, TP/EP layout, metadata packing, tile mapping,
precision and serving hooks. Recompute CTA capacity and traffic. Re-prove top-k
uniqueness and per-expert block bounds before adopting the pairwise metadata
kernel. An EP model may need dispatch/all-to-all rather than a TP all-reduce.
An MLA or paged-attention integration has KV/indexer dependencies absent from this
front/tail example. For TP4, rederive shard sizes and CTA assignments and check
weight capacity; a WORLD substitution does not extend this TP8 specialization.
