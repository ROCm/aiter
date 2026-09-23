# Offline FlyDSL PA-decode budget tuning

`aiter/ops/flydsl/pa_decode_tuning.py` benchmarks the native PA decoder and
reducer with a fixed work plan. It produces benchmark results and explicit
static-shape budget recommendations. It does **not** change runtime defaults,
refresh plans during timing, install a lookup hook, download models, or measure
whole-model inference. The offline sweep and keyed persistence follow
the approach of the commit-pinned
[FlyDSL metadata tuner](https://github.com/ROCm/FlyDSL/blob/76ca04c92a9b7459d63a9f8fded6a1f35257b723/kernels/attention/pa_metadata_tuning.py).
This tool additionally reports the smallest budget within 97% of the best speedup.

## Commands

Run from the repository root. Direct-path execution is intentional: these CPU
commands do not import `aiter`, Torch, Triton, or FlyDSL, and need only Python's
standard library:

```bash
python -B aiter/ops/flydsl/pa_decode_tuning.py --help
python -B aiter/ops/flydsl/pa_decode_tuning.py --list-models
python -B aiter/ops/flydsl/pa_decode_tuning.py \
  --model qwen3-32b --tp-size 2 \
  --batch-sizes 4,64 --context-lengths 4096,32768 \
  --query-lengths 1,4 --windows 0,1024 \
  --length-mode varlen --seed 0 \
  --budget-cu 0.5,1,2,4,8,16 \
  --architecture gfx950 --num-cu 256 --dry-run
```

Dry-run requires explicit architecture and CU count; it never probes a GPU.
It prints the static key, separate benchmark-input metadata, and deduplicated
candidate capacities. It does not write result files, even if `--output` is set.

For an actual sweep, use a supported ROCm PyTorch/FlyDSL environment and an idle
device. Architecture and CU count are detected; explicit values must agree.
The tool prepends its own repository root for imports, verifies the resolved
native host/planner/tile/reducer paths, and records those paths and source hashes.

```bash
python -B aiter/ops/flydsl/pa_decode_tuning.py \
  --model qwen3-235b-a22b --tp-size 4 \
  --batch-sizes 4,64 --context-lengths 4096,32768 \
  --query-lengths 1,4 --windows 0 \
  --length-mode varlen --seed 17 \
  --budgets 128,256,512,1024,2048,4096 \
  --rounds 16 --iterations 100 --warmup 5 \
  --device 0 --output /tmp/pa-budget-results.json
```

The default CSV path is the JSON path with a `.csv` suffix; `--csv` overrides it.
Both must be distinct new paths with existing parent directories. JSON is saved
after each shape using atomic replacement for progress updates. CSV contains
one row per candidate capacity, including failures. Nonfinite values, OOM, and
exceptions cannot become recommendations. A run with any failed shape exits
nonzero and preserves completed records. Existing output files are not resumed
or overwritten by a new CLI invocation.

## Attention geometry and workload controls

Preset names describe conventional MHA/GQA **attention geometries**, not loaded
models. The verified global dimensions are:

| Preset | Hq | Hkv | D | Source |
| --- | ---: | ---: | ---: | --- |
| `mqa-synthetic` | 16 | 1 | 128 | Synthetic MQA-like geometry; not an official M3 configuration |
| `qwen3-235b-a22b` | 64 | 4 | 128 | [Pinned Qwen config](https://huggingface.co/Qwen/Qwen3-235B-A22B/resolve/8efa61729e24bd65b1d152b5ab5409052aa80e65/config.json) |
| `qwen3-32b` | 64 | 8 | 128 | [Pinned Qwen config](https://huggingface.co/Qwen/Qwen3-32B/resolve/9216db5781bf21249d130ec9da846c4624c16137/config.json) |
| `qwen3-8b` | 32 | 8 | 128 | [Pinned Qwen config](https://huggingface.co/Qwen/Qwen3-8B/resolve/b968826d9c46dd6066d109eabc6255188de91218/config.json) |
| `gemma3-4b` | 8 | 4 | 256 | [Google reference implementation](https://github.com/google/gemma_pytorch/blob/014acb7ac4563a5f77c76d7ff98f31b568c16508/gemma/config.py), text decoder |

Gemma dimensions were checked in Google's public reference implementation, not
an authenticated HF checkpoint download. Gemma has both local (`W=1024`) and
global (`W=0`) text-attention layers; sweep them separately. Config windows,
normalization, RoPE, logit softcaps, and model-specific scaling are not applied
automatically. This tool uses `softmax_scale = D**-0.5`; it is not a model-quality
or end-to-end latency test. Qwen contexts beyond the cited checkpoints' 40,960
positions are kernel stress tests, not validated native-context inference.

Use `--config /local/model/config.json` instead of `--model` to read a local HF
configuration, including nested `text_config`. No network requests are made.
Explicit `head_dim` is authoritative: `hidden_size/Hq` would incorrectly give
64 for Qwen3-235B, 80 for Qwen3-32B, and 320 for Gemma 3 4B. Only when `head_dim`
is absent is it inferred from the **original configuration's** integral
`hidden_size/num_attention_heads`. A later query-head override does not change
that inferred D. An absent `num_key_value_heads` defaults to the original Hq
(MHA); malformed/null head counts are rejected. MLA configurations with a
non-null `kv_lora_rank` are unsupported.

`--shape Hq,Hkv,D`, `--query-heads`, `--kv-heads`, and `--head-dim` are **global**
overrides applied before TP; individual flags take precedence over `--shape`.
Hq must divide evenly by both Hkv and TP. Hkv normally divides evenly by TP.
When TP exceeds Hkv, `--allow-kv-replication` permits only explicit even
replication with `TP % Hkv == 0`, one local KV head, and replication factor
`TP/Hkv`. For example, Qwen3-235B TP4 gives local `(Hq,Hkv,G,D)=(16,1,16,128)`;
TP8 with explicit replication gives `(8,1,8,128)`. Uneven sharding is rejected.
An unsharded attention geometry does not imply all model weights fit one GPU.

The CLI sweeps the Cartesian product of batch/context/QL/window/page/dtype/
quantization/layout lists. Supported settings include:

- `--dtypes bfloat16,float16`; FP8 KV is E4M3FN on gfx950 or E4M3FNUZ on gfx942.
- `--quant-modes per_token,per_tensor`; scales are FP32.
- `--page-sizes 16,64,128`; `--trans-v yes|no|both`.
- D=64 or multiples of 128 through 1024; batch in `[1,4096]`.
- `--windows 0,1024`: 0 is dense, positive W includes each query's own token.
- `--query-lengths 1,4`: QL4 is multi-token/speculative decode, not four
  independent single-token requests. Context lengths include these query tokens.

Uniform inputs use the context upper bound for every row. Seeded varlen inputs
draw lengths uniformly from `[QL,context_length]`, then set the first row to QL
and the last to the upper bound when B>1; B1 uses the upper bound. Q and K/V are
nontrivial seeded uniform random inputs in `[-0.5,0.5)`. K/V are initialized in
page chunks and quantized to the actual FP8 cache representation. Per-token
scales use row maxabs; per-tensor scales use the known random bound divided by
FP8 max. The reference dequantizes these actual caches, not pre-quantized data.

## Budget, validation, and timing semantics

The default candidates are `128,256,512,1024,2048,4096`. Alternatively,
`--budget-cu 0.5,1,2,4,8,16` scales by detected CUs; products must be positive
integers. The actual native baseline request `2*CUs` is always included.
`max_partitions` stays at the actual CU count for every candidate.

For local KV-head count H, batch B, and CU count P, the native planner capacity is:

```text
C = min(B * P, max(B, ceil(workgroup_budget / H)))
```

Budgets with the same C share one benchmark, retaining all budget aliases.
The baseline-capacity candidate launches with the actual `2*CUs` request;
other candidates launch with their smallest alias. Recommendations use the
smallest alias for the chosen capacity. The budget is across KV heads **before**
native query splitting, not an independent budget per KV head or a strict CTA
ceiling. Batch floors, CU caps, query splitting, and unused plan slots can make
the actual grid differ substantially. Active plan slots and per-row partition
counts are saved. Native selector decisions are not overridden by the tuner.

Each candidate must pass eager execution, graph replay, and post-timing replay
against the full, streamed, dequantized FP32 attention reference, with fixed
`atol=rtol=0.005`. Every output element must be finite and within tolerance.
The reference covers the causal QL mask and, when enabled, each query's window;
it does not sample keys. TF32 is disabled and FP32 matmul precision set to
`highest` for both the CLI and direct tuning API, then previous flags are
restored. Work/reduction plan tensors must remain unchanged. A failed baseline
disables the recommendation for the entire shape, even if another budget passes.

Some short MTP inputs exceed this tolerance because of FP8 probability/value-scale
rounding in the decoder, including the pre-tuning implementation. These failures
are retained as `BASELINE_FAILED` with no recommendation. Passing checks validate
the measured inputs, not every future Q/K/V tensor sharing the same static key.

Inputs, reference, plan, scratch allocation, JIT compilation, warmup, and
correctness checks are outside timing. Each graph contains 100 native
decode-plus-reducer calls by default; the event duration divided by that count
is one latency sample. There are 16 rounds by default. Candidate order rotates
each round and reverses direction after each full cycle, including for two
candidates. All candidates share the same inputs but have independent plans
and scratch. The plan is never refreshed inside the captured graph.

Results include every sample, median/min/max latency, common-baseline speedup
(`baseline median / candidate median`), per-round speedups, and logical unique-KV
TB/s. Ranking uses the ratio of medians, not the median of per-round ratios.
The best budget maximizes this speedup; the conservative choice is the smallest
budget reaching at least **97% of the best speedup**. This is a descriptive
tolerance, not a confidence interval or a statistically established tie.
Repeat sweeps and increase rounds/iterations when differences are near noise.

The fixed numerator is FP8 unique visible K+V bytes:

```text
dense:    2 * Hkv_local * D * sum(L_i)
windowed: 2 * Hkv_local * D * sum(min(L_i, W + QL - 1))
TB/s:     unique_KV_bytes / (median_latency_us * 1e6)
```

This excludes scales, padding, metadata, scratch, repeated reads and query
traffic. It is **logical useful-KV bandwidth**, not measured physical HBM
bandwidth, model tokens/s, or end-to-end speedup. Allocation/quantization, KV
updates, plan refresh, QKV/O projections, normalization, RoPE, TP communication,
and other model layers are not timed.

## Explicit static-shape lookup

The configuration key is an explicit whitelist: architecture/CUs and FP8 dtype;
B, context upper bound, QL, local/global Hq/Hkv/G/D, TP and KV replication;
page size, Q dtype, quantization, V layout, window and softmax scale; plus schema
and native host/planner/tile/reducer/tuner source SHA256s. Model labels, config
paths, and benchmark samples are not keys.

In particular, **length distributions, seeds, exact lengths, length hashes and
input-generator identity are not configuration fields and do not affect
lookup**. They live only in the result's `benchmark_input` metadata (CSV uses
`benchmark_*` columns) for reproducing the measurement. Uniform and varlen runs
with the same static shape produce the same lookup key. A recommendation from
one sample is not a guarantee of optimality for every distribution.

One result file may contain only one record per static key. Duplicate keys,
including different sample metadata under the same key, are explicitly rejected
on save/load and cannot silently select the first record. Use separate result
files for repeat seeds/distributions and explicitly choose a reviewed run; this
tool does not automatically merge or aggregate such runs.

For CPU-only use, import the file directly to avoid `aiter.__init__` runtime
imports. Load and look up **once before plan creation**, not on every decode:

```python
import importlib.util
from pathlib import Path

path = Path("aiter/ops/flydsl/pa_decode_tuning.py").resolve()
spec = importlib.util.spec_from_file_location("pa_budget_tuning", path)
tuning = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tuning)

results = tuning.load_results("/tmp/pa-budget-results.json")
model = tuning.resolve_model("qwen3-235b-a22b", tp_size=4)
shape = tuning.make_shape(model, 64, 32768, 4)  # Distribution is not a lookup key.
key = tuning.make_key(shape, architecture="gfx950", num_cu=256)
budget = tuning.lookup_budget(results, key)  # Smallest within 97%; None on miss.
best_budget = tuning.lookup_budget(results, key, best=True)
```

`load_results` validates schema, source-key format and unique static keys.
`lookup_budget` operates on the loaded data without file or GPU access, checks
the candidate/baseline/accuracy/source gates and recomputes the saved selection.
Wrong geometry, source revision, architecture or CUs, malformed data and failed
records return `None`. This explicit validation has CPU cost; no hot-path
zero-overhead claim is made. The caller chooses whether to use a returned budget
for its own `plan_pa_decode(..., max_partitions=actual_cus,
workgroup_budget=budget)` call, or preserve its existing default when it is None.
There is no automatic runtime table load or default change.

## CPU regression tests

```bash
python -B -m unittest op_tests.tuning_tests.test_pa_decode_tuning -v
```

These tests cover geometry/config/TP validation, sample-independent keys,
capacity aliasing, balanced ordering, strict selection gates, JSON/CSV and
CPU-only CLI behavior with runtime imports explicitly blocked. GPU eager/graph
correctness and actual performance require a separate ROCm execution.
