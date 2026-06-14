# Unified-Attention per-model tuner

Tunes the Triton **unified-attention** 2D / 3D kernels per model and writes a
shape-keyed config table that AITER consults at runtime.

## How it fits together

| Piece | Path |
|---|---|
| Runtime lookup | `aiter/ops/triton/utils/ua_config.py` |
| Base table | `aiter/configs/tuned_ua.csv` |
| Per-model tables (auto-merged) | `aiter/configs/model_configs/*_tuned_ua.csv` |
| Shape/workload input | `aiter/configs/untuned_ua.csv` |
| Tuner | `csrc/unified_attention_tune/unified_attention_tune.py` |
| Env override of table path | `AITER_CONFIG_UA` |

At dispatch time, for **decode** calls (`max_seqlen_q == 1`), `unified_attention()`
looks up a row by **attention-shape signature** (`num_query_heads`,
`num_kv_heads`, `head_size`, `block_size`, `q_dtype`, `kv_dtype`,
sliding-window / sink) plus an **operating point** (a context-length bucket and
the **exact batch size**). A match selects the kernel (2D vs 3D) and its launch
config; a miss — and every prefill/mixed call — keeps the built-in heuristics.
Only the non-gfx12, non-shuffled, bf16/fp8 path is covered.

## Bucketing and fallback

**Batch is matched exactly, not bucketed.** vLLM quantizes decode `num_seqs` to
a fixed set of CUDA-graph capture sizes (and pads up to them), so the kernel
only ever sees a known, finite set of batch sizes. Keying on the exact batch
means a batch you didn't tune cleanly **defers to the heuristic** (which picks
the kernel correctly for that batch) instead of reusing a neighbor's config
across the occupancy-driven 2D/3D boundary.

**Context is bucketed** (it is continuous — KV length grows token by token).
Bucket upper bounds for `max_seqlen_k`: `512, 2048, 8192, 32768, 65536, 131072,
∞`. The first edge is the dispatch boundary below which the 2D kernel is always
used; the upper end (64k/128k) keeps long-context workloads (50k vs 100k) in
distinct buckets. Context buckets are coarse so the set of distinct launch
configs stays small — each config JITs once and is reused.

The lookup runs **per call** on the live `max_seqlen_k` / `num_seqs`, so a
serving process adapts per-request as ISL/OSL/conc vary; it is *not* fixed at
launch. On a context-bucket miss it falls **down** to a shorter-context config
(which degrades gracefully on a longer context; the reverse can be worse than
the heuristic) and up only as a last resort. The looked-up `NUM_SEGMENTS` is
always **clamped** to the live sequence length, so a long-context 3D config
cannot over-segment a short context.

## Workflow

### 1. Get the shapes to tune

Columns (decode only):

```
num_query_heads,num_kv_heads,head_size,block_size,q_dtype,kv_dtype,
sliding_window,has_sinks,phase,ctx_bucket,bs
```

- `head_size` = the kernel head dim (e.g. 64 for GPT-OSS — *not*
  `hidden_size / num_heads`); `block_size`, `kv_dtype` come from the serving
  recipe (`--block_size`, `--kv-cache-dtype`).
- `sliding_window` = 0 for full attention, else the effective window length
  (e.g. 128). `has_sinks` = 1 for attention sinks (e.g. GPT-OSS).
- `phase` is `decode`; `ctx_bucket` uses the bucket upper bounds above; `bs` is
  the **exact** batch size (a CUDA-graph capture size).

**Recommended — capture from a real run** instead of hand-authoring. Launch
your serving/benchmark with `AITER_UA_DUMP_UNTUNED=1` (or a path); on every
decode lookup miss `unified_attention` appends the *exact* observed signature,
context bucket, and batch size to the CSV (deduped per process), mirroring
FMOE's online-tune capture. This records ground truth — real `head_size`,
dtype, `block_size`, window, and the exact (ctx-bucket, batch) points the
workload hits — so you never mis-type a field and you tune exactly the batches
you serve:

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
sliding_window,has_sinks,phase,ctx_bucket,bs,
kernel,TILE_SIZE,NUM_SEGMENTS,BLOCK_M,num_warps,num_stages,waves_per_eu,
us,errRatio,_tag
```

`kernel` ∈ {`2d`, `3d`}. 2D rows set `NUM_SEGMENTS=0`. `phase` (always `decode`)
and `BLOCK_M` (always `0`; decode uses the kernel default) are carried in the
schema but unused today — reserved for a future prefill table.

## Notes / limitations

- **Decode only.** Prefill and mixed prefill+decode batches keep the heuristic:
  their shapes are not CUDA-graph-quantized (chunked prefill produces dynamic
  `max_seqlen_q`/`num_seqs`), so a per-bucket table can't represent them safely,
  and the heuristic already computes `BLOCK_M`/`BLOCK_Q` from the live shape.
- The table covers the non-gfx12, non-shuffled, bf16/fp8 Triton path. gfx12
  (gfx1250) Gluon, shuffled-KV-cache, and nvfp4 (`uint8` q/kv) keep their
  existing heuristics.
- Exact-batch keying assumes decode runs under CUDA graphs (the common case). In
  eager decode `num_seqs` is unquantized, so untuned batches simply fall back to
  the heuristic.
