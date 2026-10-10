# FlyDSL paged-attention decode benchmark

Run from the AITER repository root in a ROCm environment with FlyDSL:

```bash
python -m op_tests.op_benchmarks.flydsl.bench_pa_decode --help
python -m op_tests.op_benchmarks.flydsl.bench_pa_decode \
  --batch-sizes 16 64 --max-context-lens 4096 16384 \
  --distributions uniform log-uniform bimodal --kv-dtypes fp8 bf16 \
  --output benchmark-results/pa-decode.json
```

Defaults benchmark BF16 queries, FP8 E4M3 caches, per-token scales, Hq64/Hkv8,
D128, page128, transposed V, one query per sequence, budget512, and uniformly
sampled unrounded contexts. `--data-init uniform` samples Q/K/V from
U(-0.5,0.5); `--data-init normal` uses N(0,0.5). Both are reproducible with
`--seed`. FP8 caches are quantized from these random values; BF16 caches use
native BF16 compute on gfx1250. Input initialization runs in chunks to avoid
retaining a large FP32 copy of the KV cache.

Each sequence owns separate physical pages, shuffled across the cache pool.
Only the pages needed by the sampled contexts are allocated; there is no
repeated tiny shared KV pool. Last-page padding contains initialized finite
data. Context distributions are uniform over tokens, log-uniform over
`log(1 + tokens)`, or bimodal over the shortest and longest eighths of the
configured range. For batch sizes above one, minimum and maximum lengths are
included, then `--empty-fraction` replaces a sampled fraction with zeros.

Exact lengths override the batch-size and context-range sweep. For example,
exercise empty/short contexts, page/tile boundaries, causal MTP, and windows:

```bash
python -m op_tests.op_benchmarks.flydsl.bench_pa_decode \
  --context-lengths 0 1 127 128 129 255 256 257 1027 \
  --query-length 4 --window 257 --kv-dtypes fp8 bf16 \
  --check-sequences 9
```

Lengths include the query tokens. Early MTP query rows without visible tokens
produce zero output. `--v-layout plain`, `--query-dtype fp16`, `--page-size 16`,
and `--kv-scale per-tensor` cover alternative storage/compute modes.

Sweep explicit workgroup budgets on identical inputs with:

```bash
python -m op_tests.op_benchmarks.flydsl.bench_pa_decode \
  --batch-sizes 64 --max-context-lens 16384 --head-dim 256 \
  --workgroup-budgets 128 256 512 1024 2048 4096
```

Budget order is shuffled reproducibly. Each candidate reports its actual
capacity and active task count; different requested budgets can yield the
same plan capacity. The benchmark does not install autotuning entries.

Before measurement, the benchmark checks all output for finite values and
compares the shortest, longest, and randomly sampled sequences with FP32
causal attention over the actual dequantized cache. The default checks up to
four sequences; increase `--check-sequences` to the batch size to check all
rows. Setting it to zero disables reference checks, while finite checks remain.
FP8 tolerances are `atol=rtol=0.005`; native BF16 uses `atol=0.002, rtol=0.01`.
These defaults match the unit tests' small uniform inputs. Larger normally
distributed values can exceed them because the FP8 kernel rounds Q and P as
well as K/V. Failed reference checks abort before timing. Use explicit
`--atol`/`--rtol` overrides when evaluating a different error budget; the
effective tolerances and whether reference checking ran are saved per result.

On this gfx1250 host, the validation sweep completed 24 configurations using
page128, BF16 queries, per-token FP8 or native BF16 caches, three context
distributions, batch16/64, and maximum lengths 4096/16384. An additional
page16/plain-V run using FP16 queries and native BF16 caches encountered a
GPU page fault with FlyDSL 0.3.4.1 and HIP 7.16.26315. That combination has not
been validated by this benchmark; its diagnostic log is saved alongside the
local results. The cause has not been established.

GPU events time seven rounds, each with ten replays of a graph containing
32 decode/reduce calls. Compilation, random initialization, quantization,
reference checks, and scratch allocation occur before measurement. Plans are
built before capture; `--include-plan` additionally captures and times an
in-place GPU plan refresh for every call. Contexts stay fixed during timing.
The measured operations are reported explicitly in JSON/CSV.

Reported useful KV bandwidth is `2 * sum(context lengths) * Hkv * D *
cache_element_bytes / latency`. For windows, each context contributes the
union of tokens visible to its MTP rows, capped at `window + query_length - 1`.
K/V bytes are counted once across GQA heads and MTP rows. This excludes scales,
queries, output, scratch, page padding, and repeated physical loads. Graph
replays reuse resident inputs without cache flushing; bandwidth is an
algorithmic metric that includes cache reuse, not a physical memory counter.

JSON preserves every context length, checked sequence, raw timing sample,
GPU/software metadata, and CLI setting. The sibling CSV contains scalar
results. JSON status is `RUNNING` while a sweep is in progress, `COMPLETE`
after all cases finish, and `FAILED` when an exception aborts the sweep.
Only completed candidates are saved; an interrupted run is not a completed
sweep. Keep other GPU work idle when comparing performance.
