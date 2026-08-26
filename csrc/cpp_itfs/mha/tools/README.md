# CK MHA Tuning & Deployment Workflow

End-to-end pipeline for tuning MHA forward tile configurations against
your workload's *observed* shapes and plugging the tuned result back into
Composable Kernel (CK) tile at runtime. All three drivers live in the
same directory as this README:

- [`mha_count_shape.py`](./mha_count_shape.py) &mdash; aggregate a shape log
  into per-group CSVs and expand a `max_seqlen` sweep.
- [`mha_tune.py`](./mha_tune.py) &mdash; enumerate legal
  `FmhaFwdTileSize` candidates, build them, benchmark them, and pick the
  winner per `max_seqlen`.
- [`mha_gen_runtime_json.py`](./mha_gen_runtime_json.py)
  &mdash; merge multiple per-group tuned CSVs into a single JSON that
  CK-tile reads at JIT time via
  `CK_TILE_FMHA_FWD_CUSTOM_TUNE_CONFIG_FILE`.

> **Scope of validation.** Only the CK-tile forward path in
> **group (varlen) mode** has been end-to-end validated with this
> workflow so far. **Batch mode** and other MHA variants (backward,
> splitkv, appendkv, pagedkv, `mha_batch_prefill`, ...) are wired
> through the same infrastructure but have **not** been thoroughly
> tested; treat them as best-effort and verify manually before rolling
> them into production.

---

## Pipeline at a glance

```
   ┌──────────────┐  1. collect    ┌───────────────┐
   │ Your service │───────────────>│ shape log     │
   │  (inference) │                │  *.log        │
   └──────────────┘                └──────┬────────┘
                                          │ 2. process
                     mha_count_shape.py     ▼
       ┌─────────────────────────────────────────────────────┐
       │ mha_group_*.csv        + mha_groups_summary.csv     │
       │ mha_untune_*.csv       (max_seqlen sweep per group) │
       └─────────────────────────────────────────────────────┘
                                          │ 3. tune (per group)
                     mha_tune.py          ▼
       ┌─────────────────────────────────────────────────────┐
       │ mha_tuned_*.csv        (best tile per max_seqlen)   │
       └─────────────────────────────────────────────────────┘
                                          │ 4. merge (all groups)
                     mha_gen_runtime_json.py
                                          ▼
       ┌─────────────────────────────────────────────────────┐
       │ <deployment>.json      (compact CK-consumable)      │
       └──────────────┬──────────────────────────────────────┘
                      │ 5. enable
                      ▼
   ┌───────────────────────────────────────────────────────┐
   │ CK_TILE_FMHA_FWD_CUSTOM_TUNE_CONFIG_FILE=<...>.json   │
   │ (re-launch your service; aiter JIT picks it up)       │
   └───────────────────────────────────────────────────────┘
```

Run `python <script>.py --help` (and `<script>.py <subcmd> --help`) for
the authoritative and up-to-date option list. This document only calls
out the flags that matter for the common path.

---

## 1. Collect shape traces from your service

The dumper is built into aiter and enabled via two environment
variables (see [`csrc/include/mha_fwd_dump.h`](../../../include/mha_fwd_dump.h)):

| Env var                            | Meaning                                                  |
|------------------------------------|----------------------------------------------------------|
| `AITER_DUMP_MHA_FWD_INFO=<stride>` | Enable dumping. `1` = every call. `N>1` = subsample 1/N. |
| `AITER_DUMP_MHA_FWD_INFO_FILE`     | Output file path. Used verbatim (no `$pid` expansion).   |

Default path when `AITER_DUMP_MHA_FWD_INFO_FILE` is unset:
`/tmp/mha_dump_info_<pid>.log`. When enabled, the first successful open
prints `[MHA_FWD] AITER_DUMP_MHA_FWD_INFO enabled, writing to <path>` on
stderr. Every completed record is `fflush()`-ed, so partial data
survives external `SIGKILL`.

```bash
# Example: sample 1-in-10 forward calls, write to a workload-specific path.
AITER_DUMP_MHA_FWD_INFO=10 \
AITER_DUMP_MHA_FWD_INFO_FILE=/path/to/logs/workload.log \
  <launch-your-inference-service>
```

**Tip.** Choose the stride so that hot shapes have enough samples per
`max_seqlen_q` bucket without producing gigabytes of log. Aggregating
across multiple runs is fine &mdash; just `cat run1.log run2.log > combined.log`
before Step 2.

---

## 2. Process the log into per-group / per-sweep CSVs

### 2.1 Group by `(mode, dtype, hdim_q, hdim_v, mask_type)`

```bash
python mha_count_shape.py group -i /path/to/logs/workload.log -d ./mha_logs/
```

For each group, `group` writes one deduplicated
`mha_group_<gid>_<sig>.csv` (rows = unique `(seqlens_q, seqlens_k)`
combinations sorted by observed count), plus a cross-group
`mha_groups_summary.csv`, and prints per-group `max_seqlen_q` / `seqlens_q`
/ `seqlens_k` Top-K distributions on the terminal to help you pick the
tuning sweep.

Key options (`mha_count_shape.py group --help` for the full list):

- `-i / --input_log` &mdash; input log file (single file; concatenate
  first if you have multiple).
- `-d / --out_dir` &mdash; output directory.
- `--topk` &mdash; how many top entries to print per distribution
  (terminal only; CSVs always contain the full unique set).

### 2.2 Expand a `max_seqlen` sweep per group

Read the terminal stats from 2.1 to identify the dense hot region and
the tail, then combine `--range` (arithmetic segments, closed interval)
and `--singletons` (exact-observed lengths) into an untuned sweep:

```bash
# Dense step below 2048, coarser above.
python mha_count_shape.py generate_tune_range \
    -i ./mha_logs/mha_group_0_<sig>.csv \
    --range 512:2048:128 --range 2048:4224:256

# Different group: single dense range.
python mha_count_shape.py generate_tune_range \
    -i ./mha_logs/mha_group_1_<sig>.csv \
    --range 512:2560:128
```

Output: `mha_untune_<gid>_<sig>.csv`, colocated with the input. See
`mha_count_shape.py generate_tune_range --help` for `-o / --output`,
range/singletons semantics (union, dedup, sort), etc.

---

## 3. Tune each group

```bash
python mha_tune.py bench \
    -i ./mha_untune_0_<sig>.csv \
    --ck-root  /path/to/composable_kernel/ \
    --work-dir ./mha_group_0/ \
    --tune-hdim-q 80 --tune-hdim-v 96 \
    --nhead-q 16 --nhead-k 16 \
    --bias n --lse 0 --p-drop 0.0 \
    --warmup 5 --repeat 50 -j 64 -w 32
```

`mha_tune.py bench` is a one-shot entry point: it **auto-runs `enum`
and/or `build` first if their products are missing**, and short-circuits
to bench-only when everything is already on disk. The pipeline can also
be run stage by stage (`mha_tune.py enum` -> `mha_tune.py build` ->
`mha_tune.py bench`) when you want to inspect intermediate artifacts.

### Frequently used options

Match the shape parameters to the actual configuration of the target
MHA group so the bench numbers reflect production dispatch:

| Option                          | What it controls                                                                 |
|--------------------------------|-----------------------------------------------------------------------------------|
| `--tune-hdim-q / --tune-hdim-v` | Compiled `hdim_q` / `hdim_v` (CK requires certain padded values, e.g. 72 -> 80). |
| `--nhead-q / --nhead-k`         | Head counts passed to the bench binary; also filters codegen variants.           |
| `--bias / --lse / --p-drop`     | Feature switches; filter codegen pipelines and are passed to the bench.          |
| `--warmup / --repeat`           | Bench iteration counts per shape.                                                |
| `-w / --workers`                | Number of tune configs built in parallel.                                        |
| `-j / --jobs`                   | `cmake --build -j` for a single config.                                          |
| `--allow-mfma-16`               | Extend the enumerable mfma set (needed when hdim is not multiple of 32).         |
| `--build-target`                | Bench target name (default matches `tile_example_fmha_fwd`).                     |
| `--no-fresh`                    | Skip purging a stale build directory (fast re-runs).                             |

See `mha_tune.py --help` and `mha_tune.py <subcmd> --help` for every
option including `--limit`, `--dry-run`, `--stop-on-error`, mask helpers,
etc.

### Intermediate artifacts you can inspect

- `--work-dir/<hq>_<hv>/tile_candidates.json` &mdash; enumerated tiles.
- `--work-dir/<hq>_<hv>/tune_configs/*.json` &mdash; per-tile CK JSON that
  the `build` stage feeds to codegen. Filenames encode
  `b<bm0>x<bn0>x<bk0>x<bn1>x<bk1>x<bk0max>_r<...>_w<...>_o<occupancy>`.
- `--work-dir/<hq>_<hv>/build_<tile>/` &mdash; one cmake build tree per
  tile. Failed configs land here with logs.
- `--work-dir/bench/` &mdash; per-shape bench JSON produced by the last
  stage.

### Output

```
[done] tuned csv written to: <work-dir>/mha_tuned_<gid>_<sig>.csv
[done] per-shape bench json under: <work-dir>/bench
```

Each row of `mha_tuned_*.csv` records the winning tile for one
`max_seqlen` sample plus the achieved metrics and a human-readable
tile expression that the merger later re-parses.

Repeat Step 3 for every group you want to tune.

---

## 4. Merge tuned CSVs into a CK-consumable JSON

```bash
python mha_gen_runtime_json.py \
    --in ./mha_group_0/mha_tuned_0_<sig0>.csv \
    --in ./mha_group_1/mha_tuned_1_<sig1>.csv \
    --out ./<deployment>.json \
    --target gfx942 \
    --print-summary --compact
```

The merger:

1. Groups tuned rows per `(dtype, compiled_hdim_q, compiled_hdim_v)`.
2. Sorts by `max_seqlen`, computes integer-midpoint boundaries, folds
   consecutive rows that pick the same tile into one interval, and OR-s
   non-contiguous runs of the same tile.
3. Emits every tile with a `cpp_constraint` string (e.g.
   `a.max_seqlen_q < 704`, `a.max_seqlen_q >= 1344 && a.max_seqlen_q < 1984`)
   that CK-tile's forward dispatcher evaluates at runtime.
4. `--compact` strips debug metadata (samples, intervals, `meta.*`) for
   deployment; drop `--compact` to keep a fully annotated payload for
   diffing / audit.

Key options (`mha_gen_runtime_json.py --help` for the
rest):

- `--in` (repeat as needed) &mdash; per-group tuned CSVs to merge.
- `--out` &mdash; deployment JSON path (**required**).
- `--target` &mdash; GPU arch tag written into `target` (required by
  CK's `_build_custom_tune_factory`).
- `--print-summary` &mdash; also print a per-bucket breakdown on stderr.
- `--compact` &mdash; minimal deployment payload.
- `--constraint-var` &mdash; C++ variable name used inside
  `cpp_constraint` (default `a.max_seqlen_q`, matching CK's inner
  dispatcher).

Unknown fields in the JSON are ignored by CK, so keeping the annotated
(non-compact) copy around for auditing is safe.

---

## 5. Enable at runtime

Clear stale aiter JIT artifacts for MHA (both the `.so` and the build
tree) so the next launch triggers a fresh JIT compile that picks up the
new JSON. Then export the env var and re-launch the service:

```bash
# 1. Clean previous MHA JIT cache (paths depend on your aiter install).
#    Typical locations under aiter's jit build root:
#      <aiter-jit-root>/build/mha_varlen_fwd_*/
#      <aiter-jit-root>/*.so                  (mha-related)

# 2. Point CK-tile at the merged JSON and start the service.
CK_TILE_FMHA_FWD_CUSTOM_TUNE_CONFIG_FILE=/path/to/<deployment>.json \
  <launch-your-inference-service>
```

On startup, CK-tile's `_build_custom_tune_factory` parses the JSON,
overrides `get_hdim_tile_size_dict()` with the tuned tiles, and every
generated `fmha_fwd_v2/v3` dispatcher wraps the tile pick in the
`cpp_constraint` predicate, so different `max_seqlen_q` values land on
different tuned tiles at runtime.

**Verification checklist**

- Startup log shows the CK codegen path picking up the JSON (typically a
  line mentioning the custom factory / arch base).
- The first inference call for each hot `max_seqlen_q` should hit the
  tuned tile rather than CK's built-in defaults.
- If a link error like
  `multiple definition of fmha_fwd_<..._traits_<...>, gfx9_t>(...)`
  appears, verify that your CK checkout carries the per-occupancy
  `kOccupancy_` traits parameter; without it, two tiles that differ
  *only* in `F_occupancy` collide at link time.

---

## Recap of the current validation matrix

| Path / feature                                | Status                              |
|-----------------------------------------------|-------------------------------------|
| CK-tile forward, **group (varlen)** mode      | **End-to-end validated.**           |
| CK-tile forward, **batch** mode               | Wired through, **not** validated.   |
| CK-tile forward, splitkv / appendkv / pagedkv | Wired through, **not** validated.   |
| CK-tile forward, `fmha_batch_prefill`         | Wired through, **not** validated.   |
| Backward pass                                 | **Not covered.**                    |

Anything outside the "validated" row above should be treated as
best-effort: the tooling produces artifacts, but no perf/correctness
regression has been signed off yet. Please share results (or issues)
when you exercise those paths so the matrix can be updated.

---

## Troubleshooting

- **`bench` complains about missing `tile_candidates.json`.** Fixed:
  `bench` now auto-runs `enum` and `build` when their products are
  missing. If the message still shows up, verify you are on the latest
  `mha_tune.py`.
- **All configs fail to `cmake configure` / build.** Confirm
  `--ck-root` points at a CK checkout that carries the custom-tuning
  factory (`_build_custom_tune_factory`) and the `kOccupancy_` traits
  parameter. Older CK forks will not accept the tune JSONs emitted by
  Step 3.
- **Merged JSON is loaded but perf does not change.** Make sure the
  aiter MHA JIT cache was purged before re-launching, and that
  `CK_TILE_FMHA_FWD_CUSTOM_TUNE_CONFIG_FILE` is visible to the process
  that actually runs the kernels (not only the parent shell).
- **`AITER_DUMP_MHA_FWD_INFO` is enabled but the log stays empty.**
  Check the stderr line that reports the actual sink path; if the
  target directory is not writable, the dumper falls back to stderr and
  prints a warning.
