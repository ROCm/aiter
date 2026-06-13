# Unified-Attention per-model tuner

Tunes the Triton **unified-attention** 2D / 3D kernels per model and writes a
shape-keyed config table that AITER consults at runtime. This is the attention
analogue of the FMOE / GEMM tuned-config mechanism under `aiter/configs/`.

## How it fits together

| Piece | Path |
|---|---|
| Runtime lookup | `aiter/ops/triton/utils/ua_config.py` |
| Base table | `aiter/configs/tuned_ua.csv` |
| Per-model tables (auto-merged) | `aiter/configs/model_configs/*_tuned_ua.csv` |
| Shape/workload input | `aiter/configs/untuned_ua.csv` |
| Tuner | `csrc/unified_attention_tune/unified_attention_tune.py` |
| Env override of table path | `AITER_CONFIG_UA` |

At dispatch time, `unified_attention()` looks up a row by **attention-shape
signature** (`num_query_heads`, `num_kv_heads`, `head_size`, `block_size`,
`q_dtype`, `kv_dtype`, sliding-window / sink flags) plus an **operating point**
(`phase` ∈ {prefill, decode}, a context-length bucket, a batch bucket). A match
selects the kernel (2D vs 3D) and its launch config; no match keeps the built-in
heuristics. Only the non-gfx12, non-shuffled Triton path is covered.

## Bucketing and fallback

Lookup is intentionally **coarse-bucketed** to bound the number of distinct
Triton compilations during a serving run:

- context (`max_seqlen_k`) buckets (upper bounds):
  `512, 2048, 8192, 32768, 65536, 131072, ∞` — the first edge is the dispatch
  boundary below which the 2D kernel is always used, so a bucket never straddles
  the 2D/3D split. The upper end (64k / 128k) covers long-context production
  workloads (e.g. 50k and 100k ISL land in distinct buckets).
- batch (`num_seqs`) buckets (upper bounds): `4, 16, 64, 256, ∞`.

The lookup runs **per `unified_attention()` call** on the live `max_seqlen_k` /
`num_seqs`, so a single serving process adapts per-request as ISL/OSL/conc vary
(and as context grows during a decode). It is *not* fixed at launch.

Context is matched first (it decides 2D vs 3D): within the chosen context
bucket the batch bucket is tried exact→down→up, then the context bucket falls
**down** to a shorter-context config (which degrades gracefully on a longer
context; the reverse can be worse than the default), and only as a last resort
up. The looked-up `NUM_SEGMENTS` is always **clamped** to the valid range for
the actual sequence length, so a long-context 3D config cannot over-segment a
short context.

## Workflow

### 1. Get the shapes to tune

Columns:

```
num_query_heads,num_kv_heads,head_size,block_size,q_dtype,kv_dtype,
sliding_window,has_sinks,phase,ctx_bucket,bs_bucket
```

- `head_size` = the kernel head dim (e.g. 64 for GPT-OSS — *not*
  `hidden_size / num_heads`); `block_size`, `kv_dtype` come from the serving
  recipe (`--block_size`, `--kv-cache-dtype`).
- `sliding_window` = 0 for full attention, else the effective window length
  (e.g. 128). `has_sinks` = 1 for attention sinks (e.g. GPT-OSS).
- `phase`, `ctx_bucket`, `bs_bucket` enumerate the operating points, using the
  bucket upper bounds above.

**Recommended — capture from a real run** instead of hand-authoring. Launch
your serving/benchmark with `AITER_UA_DUMP_UNTUNED=1` (or a path); on every
lookup miss `unified_attention` appends the *exact* observed signature and
bucket to the CSV (deduped per process), mirroring FMOE's online-tune capture.
This records ground truth — real `head_size`, dtype, `block_size`, window, and
the actual ctx/bs buckets the workload sweeps — so you never mis-type a field:

```bash
AITER_UA_DUMP_UNTUNED=/work/untuned_ua.csv vllm serve ...   # run your benchmark
python3 csrc/unified_attention_tune/unified_attention_tune.py \
    -i /work/untuned_ua.csv -o aiter/configs/model_configs/mymodel_tuned_ua.csv
```

### 2. Run the tuner (inside the container with vLLM + AITER)

```bash
python3 csrc/unified_attention_tune/unified_attention_tune.py \
    -i aiter/configs/untuned_ua.csv \
    -o aiter/configs/model_configs/mymodel_tuned_ua.csv
```

For each row the tuner benchmarks the default heuristic, sweeps 2D and 3D launch
configs (forcing each kernel), and writes the winner — **only if it beats the
default by `--min_improvement_pct` (default 3%)** at the sampled operating
point. The guarantee holds at the bucket's sample point; since one row covers a
bucket range, verify with an end-to-end run (or set `AITER_BYPASS_TUNE_CONFIG=1`
to fall back to heuristics).

### 3. Use it

Any `aiter/configs/model_configs/*_tuned_ua.csv` is auto-merged with the base
`tuned_ua.csv` for the current `gfx` + CU count. To pin a specific file:

```bash
export AITER_CONFIG_UA=/path/to/mymodel_tuned_ua.csv
```

To ignore the table entirely and run on the built-in heuristics, set the same
global switch the other AITER ops honor:

```bash
export AITER_BYPASS_TUNE_CONFIG=1
```

## Tuned CSV schema

```
gfx,cu_num,num_query_heads,num_kv_heads,head_size,block_size,q_dtype,kv_dtype,
sliding_window,has_sinks,phase,ctx_bucket,bs_bucket,
kernel,TILE_SIZE,NUM_SEGMENTS,BLOCK_M,num_warps,num_stages,waves_per_eu,
us,errRatio,_tag
```

`kernel` ∈ {`2d`, `3d`}. For 2D rows `NUM_SEGMENTS=0`; `BLOCK_M=0` means "use
the kernel default" (only large prefill records a non-zero `BLOCK_M`). For 3D
rows `BLOCK_M=0`.

## Notes / limitations

- Triton recompiles per distinct config; keep buckets coarse and warm up the
  configs your deployment hits.
- The table covers the non-gfx12, non-shuffled, bf16/fp8 Triton path. gfx12
  (gfx1250) Gluon, shuffled-KV-cache, and nvfp4 (`uint8` q/kv) keep their
  existing heuristics.
- `phase` is decode (`max_seqlen_q == 1`) or prefill (everything else, incl.
  mixed/chunked batches); the prefill `BLOCK_M` override is only applied when
  the live `max_seqlen_q >= 256`, matching the heuristic.
