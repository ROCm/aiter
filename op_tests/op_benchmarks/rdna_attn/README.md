# RDNA attention: adding, tuning and checking a configuration

`aiter.ops.rdna_attn` (gfx1150..gfx1153, and RDNA4's gfx1200 and gfx1201) runs
decodes of up to 8 query tokens per sequence on
`csrc/kernels/rdna_attn/rdna_decode_attn.cuh` and everything else on
`rdna_prefill_attn.cuh`. Its tables, in `csrc/kernels/rdna_attn/`:

| File | Entries | Used for |
| --- | --- | --- |
| `rdna_decode_configs.h` | decode builds, one per M | one sequence |
| `rdna_decode_batch_configs.h` | decode builds, one per M | a batch of sequences |
| `rdna_prefill_configs.h` | prefill builds, one per step of M | one sequence |
| `rdna_prefill_batch_configs.h` | per batch cell, a prefill build and head group | a batch of sequences |

Each file has a table per GPU family: gfx1151's (1536 VGPRs a SIMD lane),
gfx1150's (1024), which gfx1152 and gfx1153 also use, and gfx1201's, which
gfx1200 also uses. The tools tune and write the tables of the GPU they run on.
Prefill `v_t` 2 loads V^T with GLOBAL_LOAD_TR (gfx12; gfx11 builds it as 1).

A *configuration* is `(Hq, Hkv, D, window, page size)`, in fp16 and bf16.

| Tool | What it does |
| --- | --- |
| `tune.py` | searches the launch knobs of a configuration and writes the tables |
| `matrix.py` | times the shapes of `shapes.csv` against their roofline |
| `floor.py` | the most a decode kernel can reach against the roofline |
| `harness.py` | serves and times a batch as a server runs it; used by the others |
| `jit.py` | standalone builds of any knobs, for the points the tables lack |
| `utils.py` | the roofline model and the knobs |

Every time is compared against the roofline of `utils.py`: the longer of the
call's bytes at peak bandwidth and its FLOPs at peak throughput, plus a
dispatch, with the peaks measured on each GPU family.

## Requirements

- A gfx1150..gfx1153 or RDNA4 GPU, with aiter importable (`pip install -e .`,
  or `PYTHONPATH=$PWD` from the tree root).
- `D` in 32, 48, 64, 72, 80, 96, 128, 256, 512, `Hq % Hkv == 0`, and a page
  size that is a multiple of 16.
- Nothing else running on the GPU.

`jit.py` keeps its builds in `RDNA_JIT_CACHE` (default
`~/.cache/aiter/rdna_jit`).

## Decode

Tune each M in fp16 (bf16 reuses the choice), then apply:

```bash
for m in 1 2 3 4 5 6 7 8; do
  python op_tests/op_benchmarks/rdna_attn/tune.py \
      --hq 16 --hkv 2 --head-dim 256 --m $m \
      --points points.jsonl --decisions decisions.jsonl
done
python op_tests/op_benchmarks/rdna_attn/tune.py \
    --apply decisions.jsonl --block-sizes 16
```

- `--window W`, `--block-size P`, `--reuse` to resume, `--limit 20` for a
  smoke run.
- `--apply` refuses decisions measured on another kernel source or entry. It
  removes the built `module_rdna_attn*`, which hold the tables as they were;
  they are rebuilt from the headers at their next use.

Add the configuration to `shapes.csv`.

## Batched decode

Tune each M over batch specs (`batch_spec.py`):

```bash
for m in 1 2 3 4 5 6 7 8; do
  python op_tests/op_benchmarks/rdna_attn/tune.py \
      --hq 16 --hkv 2 --head-dim 256 --m $m \
      --batch 2q${m}s4k 2q${m}s32k 4q${m}s16k 8q${m}s4k 8q${m}s8k 12q${m}s2k \
              16q${m}s2k 32q${m}s1k 64q${m}s1k q${m}s32k_15q${m}s1k \
      --points points_batch.jsonl --decisions decisions_batch.jsonl
done
python op_tests/op_benchmarks/rdna_attn/tune.py \
    --apply decisions_batch.jsonl --block-sizes 16
```

A batch beyond the 64 MiB scratch budget is not served.

## Prefill

Tune every step of M in one run:

```bash
python op_tests/op_benchmarks/rdna_attn/tune.py --prefill \
    --hq 16 --hkv 2 --head-dim 256 \
    --points points.jsonl --decisions decisions.jsonl
```

An entry serves M from its `min_query_len` up to the next entry's. Steps start
at 16, 32, ..., 8192; the first entry also serves 9 to 15 tokens. `--steps
8192` tunes only that step; `--apply` keeps the others. A step's winner is the
finalist closest to the roofline that loses at most `--prefill-loss-limit`
(2%) to the current entry at any cell; `--reselect decisions.jsonl --decisions
out.jsonl` chooses stored decisions again from their A/B samples, without
timing. `--non-causal` tunes the encoder builds (every token sees every key of
its sequence).

## Batched prefill

Per cell of (mean query tokens, mean keys per token), the prefill build and
head group, once the prefill entries exist (`--non-causal` for the encoder
builds):

```bash
python op_tests/op_benchmarks/rdna_attn/tune.py --prefill-batch \
    --hq 16 --hkv 2 --head-dim 256 --decisions decisions_prefill_batch.jsonl
python op_tests/op_benchmarks/rdna_attn/tune.py \
    --apply decisions_prefill_batch.jsonl
```

`--apply` refuses the cells if the prefill entries changed since.

## Check

```bash
pytest tests/test_rdna_attn_tables.py
python3 op_tests/test_rdna_attn.py
python op_tests/op_benchmarks/rdna_attn/matrix.py \
    --hq 16 --hkv 2 --head-dim 256 --m 1 2 3 4 5 6 7 8 > matrix.md
python op_tests/op_benchmarks/rdna_attn/matrix.py \
    --hq 16 --hkv 2 --head-dim 256 --batch decode prefill-batch mixed \
    > batches.md
```

Add `--prefill` (with `--m` and `--prefixes`), `--non-causal` (with `--m` or
`--segments`), `--windowed`, `--block-size P` and `--spec-tokens N` as needed.
The `path` column must read `kernel` or `prefill`. Forcing a knob
(`--waves 4`), `--ablate` or `--jit` times jit.py builds instead of the
modules'. `floor.py` gives the ceiling of `%roof` for a decode configuration.
