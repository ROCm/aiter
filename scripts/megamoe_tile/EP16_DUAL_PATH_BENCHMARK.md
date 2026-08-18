# EP16 dual-path benchmark audit

## Locked target

```text
tokens/rank  128
H            7168
I            3072
E            896 (56/rank)
topk         16
EP           16 = 2 nodes x 8 GPUs
activation   SiLU
quant        A4W4
route        bench_mega_moe_v2.make_inputs(..., "rank-balanced-hot", 0.6)
```

`E=384` is the separate DSV4 expert-count configuration. The driver rejects it
explicitly; this benchmark always constructs E896 routes, a 896-entry expert
mask, MORI `num_experts_per_rank=56`, and local weight tensors with leading
dimension 56.

With `topk == EP`, MegaMoE's rank-balanced generator selects exactly one route
on every EP rank for every source token. The static MORI receive count is
therefore `16 * 128 = 2048` rows/rank.

## Baseline reuse

`bench_megamoe_tile_ep16_dual_path.py` uses MORI `InterNodeV1LL` directly:

```text
dispatch  block=256 rdma=128 warps=8
combine   block=256 rdma=128 warps=4
max_total_recv_tokens=0
```

Dispatch carries packed A4 plus 224 raw E8M0 scales. The explicit local path
selects the one route owned by this rank, performs local-expert sort,
`moe_mxfp4_sort`, GMM1+SiLU+A4 requant, weighted GMM2, and scatters the result
back to MORI dispatch-row order. `combine` receives `weights=None` because GMM2
already applied the route weight, and receives the original source-rank top-k
IDs because MORI retains its dispatch routing state internally.

## Candidate wire geometry

```text
aggregate chunk       524288 B (4 QPs)

dispatch record       4096 B
dispatch batch/QP     32
dispatch records/chunk 128
dispatch active bytes 524288
dispatch chunks/rank  1

partial record        14592 B
partial batch/QP      8
partial records/chunk 32
partial active bytes  466944
partial chunks/rank   4

remote LSA fanout entries/rank  128 * 8 = 1024
```

The AITER wire helpers support this geometry and their H7168 GPU pack/unpack
tests pass. The reusable candidate factory is:

```text
op_tests.multigpu_tests.bench_megamoe_tile_ep16_dual_path:build_candidate_path
```

It packs one dispatch chunk, fans local and remote records into 2048 compact
LSA inbox slots, runs explicit H1/H2, performs EP8 node reduction, returns 4
partial chunks, and final-combines the two node partials. `--paths both` uses
this built-in factory; `--candidate-factory module:function` remains available
for alternate implementations. A baseline-only run is explicitly labelled
`MEGAMOE_EP16_BASELINE_SMOKE` and is never reported as a comparison.

## Timing contract

- Warmup correctness and static receive-count validation occur outside timing.
- After prime/JIT and all warmups, every rank performs a CUDA synchronize and
  one Gloo barrier before timed work.
- `--barrier-mode once` leaves timed iterations free of Gloo collectives;
  `each` inserts a Gloo barrier before, never inside, every timed iteration.
- No Gloo numerical reference/all-reduce occurs inside a timed iteration.
- Every stage has HIP start/end events.
- A separate GPU E2E event spans all stages and exposes enqueue gaps between
  stage events.
- Host critical path spans enqueue through final device synchronize.
- Every sample is reduced with `MAX` across all 16 ranks outside timing.
- Per-iteration rank MAX is deferred until the complete local timed loop has
  finished, preserving continuous operations in `once` mode.
- `--tail-iters N` reports tail mean/p50/p95 while legacy median fields continue
  to summarize all timed iterations.
- Reports include stage sum, GPU E2E minus stage sum, and host minus GPU E2E.
- Each iteration still ends in local CUDA synchronize and therefore measures
  independent round latency, not cross-iteration enqueue throughput.
- When both paths are connected, candidate-vs-baseline relative L2 is checked
  after timing and reduced by rank maximum.
- Correctness diagnostics also report logits diff, max absolute error, norm
  ratio, per-token relative-L2 min/p50/max, and prime-vs-timed nondeterminism.
